---
companies:
- google-deepmind
- openai
- lmarena_ai
date: '2025-03-26T01:13:42.288748Z'
description: 'Google DeepMind 的 **Gemini 2.5 Pro** 已成为新的顶级 AI 模型，在 LMarena 评分上超过了
  **Grok 3** 达 40 分。该模型集成了 **Noam Shazeer** 贡献的 Flash Thinking 技术。目前，它作为一个免费且有速率限制的实验性模型提供。


  与此同时，**OpenAI** 发布了 **GPT 4o Native Images**，这是一款自回归图像生成模型，**Allan Jabri** 分享了相关详细见解，**Gabe
  Goh** 亦对此有贡献。Gemini 2.5 Pro 在推理、编程、STEM、多模态任务和指令遵循方面表现卓越，在 LMarena 排行榜上显著领跑。用户可以通过
  Google AI Studio 和 Gemini 应用访问该模型。'
id: 270c74dd-e984-447a-b1e7-90637af80bd2
models:
- gemini-2.5-pro
- gpt-4o
original_slug: ainews-gemini-25-pro-4o-native-image-gen
people:
- noam-shazeer
- allan-jabri
- gabe-goh
title: Gemini 2.5 Pro + 4o 原生图像生成
topics:
- autoregressive-models
- multimodality
- reasoning
- coding
- instruction-following
- model-release
- leaderboards
---

<!-- buttondown-editor-mode: plaintext -->**这是一个多么精彩的时代。**

> 2025年3月24日至3月25日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**228** 个频道，**6171** 条消息）。预计为您节省阅读时间（按 200wpm 计算）：**566 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天两家前沿实验室发布的内容都足以登上头条，所以它们必须共享版面。

## Gemini 2.5 Pro

[**Gemini 2.5 Pro**](https://news.ycombinator.com/item?id=43473489) 是全球新晋的无可争议的顶级模型，其 LMarena 分数比上个月刚发布的 Grok 3（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-xai-grok-3-and-mira-muratis-thinking/)）高出整整 40 分。[Noam Shazeer 的参与](https://twitter.com/NoamShazeer/status/1904581813215125787)暗示了 Flash Thinking 的经验已被整合进 Pro 中（奇怪的是 2.5 Pro 竟然比 2.5 Flash 先发布？）。


![image.png](https://assets.buttondown.email/images/797084f7-b6d9-4785-9ad9-7d7bba2ab259.png?w=960&fit=max)


[Simon Willison](https://simonwillison.net/2025/Mar/25/gemini/)、[Paul Gauthier (aider)](https://x.com/paulgauthier/status/1904637913411031410)、[Andrew Carr](https://x.com/andrew_n_carr/status/1904607188976611627) 等人都发表了值得一读的简评，主题都是“该模型已达 SOTA”。

定价尚未公布，但您今天可以将其作为**免费**且受速率限制的“实验性模型”使用。

## GPT 4o Native Images

紧随[昨天的 Reve Image](https://buttondown.com/ainews/archive/ainews-halfmoon-is-reve-image-a-new-sota-image/) 和 [Gemini 的 Native Image Gen](https://buttondown.com/ainews/archive/ainews-gemma-3-beats-deepseek-v3-in-elo-20-flash/) 之后，OpenAI 终于发布了 4o 原生图像生成功能，并配以[直播](https://www.youtube.com/live/2f3K43FHRKo?si=QX6oXEalK8XRSvrP)、[博客文章](https://news.ycombinator.com/item?id=43474112)以及 [System Card](https://openai.com/index/gpt-4o-image-generation-system-card-addendum/)，确认了这是一个自回归模型。关于其工作原理，我们目前能获得的最详细信息可能就是来自 [Allan Jabri](https://x.com/ajabri/status/1904599427366739975) 的这张图片，他曾参与最初未发布的 4o 图像生成工作（随后由 [Gabe Goh 接手，正如 sama 所称赞的那样](https://x.com/sama/status/1904599358756315341)）。


![image.png](https://assets.buttondown.email/images/ea1255e2-746c-4ee9-8cc3-c2da91fde74d.png?w=960&fit=max)


> 一张用手机拍摄的宽幅照片，画面是一个玻璃白板，房间俯瞰着海湾大桥。视野中有一位女性正在书写，穿着一件印有巨大 OpenAI Logo 的 T 恤。字迹看起来很自然且略显凌乱，我们还能看到摄影师的倒影。文字内容如下：（左侧）“模态间迁移：假设我们直接用一个大型自回归 Transformer 建模 p(text, pixels, sound) [等式]。优点：* 结合了海量世界知识的图像生成 * 下一代文本渲染 * 原生 In-context learning * 统一的后训练栈。缺点：* 不同模态间的比特率差异 * 计算不可自适应。”（右侧）“修复方案：* 模型压缩表示 * 将自回归先验与强大的解码器结合。”在白板右下角，她画了一个图表：“tokens -> [Transformer] -> [Diffusion] -> pixels”

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**模型发布与公告**

- **Google 的 Gemini 2.5 Pro** 引起轰动，发布了多项关键公告：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1904579660740256022) 推出了 **Gemini 2.5 Pro Experimental**，称其为最智能的模型，强调了其推理能力和准确性的提升，详情见其 [blog](https://t.co/Gx35wlkomq)。[@NoamShazeer](https://twitter.com/NoamShazeer/status/1904581813215125787) 强调 **2.5 系列** 标志着向根本性思考模型的演进，即在回答前进行推理。它在编程、STEM、多模态任务、指令遵循方面表现出色，并在 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1904581128746656099) 排行榜上以 40 ELO 的巨大优势位居第一，编程性能也极为出色。它以巨大优势登顶了 [@lmarena_ai 的排行榜](https://twitter.com/Google/status/1904581629017735261)。[@jack_w_rae](https://twitter.com/jack_w_rae/status/1904583894458110218) 指出 **2.5 Pro** 在编程、STEM、多模态任务和指令遵循方面有所提升，现已在 AI Studio 和 Gemini App 中上线。
- **Gemini 2.5 Pro 的可用性**：开发者可以在 [Google AI Studio](https://twitter.com/GoogleDeepMind/status/1904581166755123463) 中访问，Advanced 用户可在 [@GeminiApp](https://twitter.com/GoogleDeepMind/status/1904581166755123463) 中使用，Vertex AI 也即将支持。据 [@casper_hansen_](https://twitter.com/casper_hansen_/status/1904590489128440163) 称，该模型对所有人免费开放。[@stevenheidel](https://twitter.com/stevenheidel/status/1904601168317399199) 分享了一个使用 [该网站](https://t.co/8Xy5cfz7Y2) 新图像生成的专业技巧，用户可以在那里设置纵横比并生成多个变体。
- **DeepSeek V3-0324 发布**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1904467255083348244) 报道称，**DeepSeek V3-0324** 现在是得分最高的非推理模型，这标志着权重开放模型首次在该类别中领先。该模型的细节与 2024 年 12 月版本基本一致，包括 128k 上下文窗口（DeepSeek API 限制为 64k）、671B 总参数量和 MIT License。[@reach_vb](https://twitter.com/reach_vb/status/1904447298811437061) 指出，该模型在 MIT License 下击败了 Sonnet 3.7 和 GPT4.5 或与之持平，提升了代码的可执行性，并能生成更美观的网页和游戏前端。
- **OpenAI 的图像生成**：[@OpenAI](https://twitter.com/OpenAI/status/1904602845221187829) 宣布 4o 图像生成功能已上线。今天开始向 ChatGPT 和 Sora 的所有 Plus、Pro、Team 和 Free 用户推出。[@kevinweil](https://twitter.com/kevinweil/status/1904595752380465645) 表示，ChatGPT 的图像生成功能迎来了重大更新，现在非常擅长遵循复杂的指令，包括详细的视觉布局。它在生成文本方面表现出色，并能实现写实主义或任何其他风格。

**基准测试与性能评估**

- **Gemini 2.5 Pro 的性能**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1904581128746656099) 宣布 **Gemini 2.5 Pro**（测试代号为 "nebula"）现位居 Arena 排行榜第一。它在所有类别中均排名第一，并且在数学、创意写作、指令遵循、长查询和多轮对话中独占鳌头！[@YiTayML](https://twitter.com/YiTayML/status/1904598794278494272) 表示 Google 领先优势巨大，Gemini 2.5 Pro 是目前世界上最好的模型。[@alexandr_wang](https://twitter.com/alexandr_wang/status/1904590438469951873) 也指出 **Gemini 2.5 Pro Exp** 发布后，目前在 SEAL 排行榜中位列第一。[@demishassabis](https://twitter.com/demishassabis/status/1904587103805006218) 总结道，**Gemini 2.5 Pro** 是一款出色的 SOTA 模型，在 LMArena 上以高达 +39 ELO 的分差排名第一。[@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1904583691566727361) 补充说，Gemini 2.5 Pro Experimental 在数学和科学基准测试中表现卓越。
- **DeepSeek V3-0324 对比其他模型**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1904467262364692970) 指出，与领先的推理模型（包括 DeepSeek 自身的 R1）相比，**DeepSeek V3-0324** 仍显落后。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1904364173426893235) 强调 **Deepseek API 更新日志** 已针对 0324 版本更新，在 MMLU-Pro、GPQA、AIME 和 LiveCodeBench 等基准测试中均有实质性提升。[@reach_vb](https://twitter.com/reach_vb/status/1904447298811437061) 分享了 DeepSeek V3 0324 的基准测试提升数据。

**AI 应用与工具**

- **工作流 AI 驱动工具**：[@jefrankle](https://twitter.com/jefrankle/status/1904590481222218176) 讨论了 **TAO**，这是来自 @databricks 的一种新型 finetuning 方法，仅需要输入而不需要标签，在性能上超越了基于有标签数据的 supervised finetuning。[@jerryjliu0](https://twitter.com/jerryjliu0/status/1904328371867361469) 介绍了 **LlamaExtract**，它可以将复杂的发票转换为标准化的 schemas，并针对高准确性进行了优化。
- **Weights & Biases AI Agent 工具链**：[@weights_biases](https://twitter.com/weights_biases/status/1904524590988013951) 宣布他们在 @weave_wb 中的 @crewAIInc 集成已正式上线。现在可以统一在一个强大的界面中追踪每个 Agent、任务、LLM 调用、延迟和成本。
- **Langchain 更新**：[@hwchase17](https://twitter.com/hwchase17/status/1904589229856080084) 强调了 **Langgraph computer use agent** 的可用性。[@LangChainAI](https://twitter.com/LangChainAI/status/1904589968116420924) 提到，如果你想在 Langgraph Agent 中使用 OpenAI 的 computer use 模型，这是最简单的方法！

**研究与开发**

- **机器人领域的 AI**：来自 Figure 的 [@adcock_brett](https://twitter.com/adcock_brett/status/1904535004866052228) 指出，他们拥有一个能够像人类一样自然行走的神经网络，并在[这篇报告](https://t.co/l63afSgBd0)中讨论了使用 Reinforcement Learning、模拟训练以及向机器人机群进行 zero-shot transfer 的技术。[@hardmaru](https://twitter.com/hardmaru/status/1904320457396162563) 表示对团队感到非常自豪，并认为这次美日国防挑战赛只是 @SakanaAILabs 助力加速日本国防创新的第一步。
- **新架构**：Nvidia 发布了 [FFN Fusion: Rethinking Sequential Computation in Large Language Models](https://twitter.com/_akhaliq/status/1904390303458459821)，[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1904370227665912243) 指出该技术实现了 1.71 倍的 inference latency 加速，并将 per-token 成本降低了 35 倍。
- **Text-to-Video 模型方法**：AMD 在 Hugging Face 上发布了 [AMD-Hummingbird](https://twitter.com/_akhaliq/status/1904386209373118623) —— “迈向高效的 Text-to-Video 模型”。

**AI 伦理与社会影响**

- **自由与负责任的 AI**：[@sama](https://twitter.com/sama/status/1904598788687487422) 讨论了 OpenAI 在新图像生成功能中处理创作自由的方法，旨在让工具在合理范围内不生成冒犯性内容，除非用户明确要求。他强调了 AI 尊重社会边界的重要性。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1904547109551919174) 建议使用 open-source AI 并对 AI 系统进行受控的自主化，以降低网络安全风险。
- **AI 基准测试**：[@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1904572772833014074) 采访了 @js_denain，探讨了当下的 benchmarks 存在的不足，以及改进后的评估方法如何能更好地揭示 AI 的真实世界能力。

**幽默与杂项**

- **Elon Musk 与 Grok**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1904373288463401466) 询问 [@elonmusk](https://twitter.com/elonmusk/status/1904373288463401466)，Grok3 的“大脑袋”模式何时发布？
- [@sama](https://twitter.com/sama/status/1904604934387229134) 转发了 [@NickADobos](https://twitter.com/NickADobos/status/1904604934387229134) 的内容，并评论“真是个帅哥！”
- [@giffmana](https://twitter.com/giffmana/status/1904625459641671864) 说道：哎哟查理，那真的很疼！
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1904346173747437627) 询问 [@FearedBuck](https://twitter.com/FearedBuck/status/1904346173747437627) 为什么他是一只熊猫？

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：DeepSeek V3 0324 登顶非推理模型排行榜**

- **[根据 Artificial Analysis 的数据，DeepSeek V3 0324 目前是最佳的非推理模型（涵盖开源和闭源）。](https://i.redd.it/4hh6ys9gftqe1.png)** ([Score: 736, Comments: 114](https://reddit.com/r/LocalLLaMA/comments/1jjgi8y/deepseek_v3_0324_is_now_the_best_nonreasoning/)): **DeepSeek V3 0324** 被 **Artificial Analysis** 评为顶尖的非推理 AI 模型，表现优于开源和闭源模型。它以 **53** 分的成绩领跑 **Artificial Analysis Intelligence Index**，超过了分别获得 **53** 分和 **51** 分的 **Grok-3** 和 **GPT-4.5 (Preview)**；该指数根据推理、知识、数学和编程等标准对模型进行评估。
  - 用户对基准测试的可靠性持怀疑态度，如 **artisticMink** 和 **megazver** 担心基准测试可能无法准确反映真实世界的性能，并且可能偏向于较新的模型。**RMCPhoto** 和 **FullOf_Bad_Ideas** 还指出，与 **Claude 3.7** 等模型相比，**DeepSeek V3** 和 **QWQ-32B** 等特定模型在实际应用中的表现可能并不理想。
  - 用户讨论了 **DeepSeek V3** 和 **Gemma 3** 等模型的可访问性和使用情况，**Charuru** 提供了通过 **deepseek.com** 访问的信息，而 **yur_mom** 讨论了无限使用的订阅选项。**East-Cauliflower-150** 和 **emsiem22** 强调了 **Gemma 3** 尽管只有 **27B parameters**，但其能力令人印象深刻。
  - 社区对即将推出的模型和更新表示关注，例如 **DeepSeek R2** 和 **Llama 4**，**Lissanro** 正在等待 **Unsloth** 在 [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF) 上发布的动态量化（dynamic quant）版本。人们还对 **Meta Llama** 等团队在持续发展中发布竞争性模型的压力表示担忧。


- **DeepSeek-V3-0324 GGUF - Unsloth** ([Score: 195, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1jji2da/deepseekv30324_gguf_unsloth/)): **DeepSeek-V3-0324 GGUF** 模型在 [Hugging Face](https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF) 上提供从 **140.2 GB** 到 **1765.3 GB** 不等的多种格式。正如 **u/yoracale** 所述，用户目前可以访问 2、3 和 4-bit 的动态量化版本，进一步的上传和测试正在进行中。
  - **动态量化性能 (Dynamic Quantization Performance)**: 用户讨论了不同量化方法对大语言模型 (LLMs) 的性能影响。**标准 2-bit 量化**因性能不佳而受到批评，而 **2.51 动态量化**在生成功能性代码方面表现出显著改进。
  - **硬件和资源限制**: 讨论了在没有大量计算资源的情况下运行近 **2TB 模型**的不切实际性，并提出了使用 **4x Mac Studio 512GB 集群**等建议。一些用户表达了对可用 VRAM 的挑战，指出即使是 **190GB** 也不足以获得最佳性能。
  - **即将发布的版本和建议**: 建议用户等待 **dynamic IQ2_XSS quant**，它承诺比目前的 **Q2_K_XL** 具有更高的效率。**Unsloth** 的 **IQ2_XXS R1** 尽管体积较小，但因其效率而受到关注，目前正在努力上传更多动态量化版本，如 **4.5-bit 版本**。


- **[DeepSeek 在 X 上的官方公告：DeepSeek-V3-0324 现已发布！](https://www.reddit.com/gallery/1jjjv8k)** ([Score: 202, Comments: 7](https://reddit.com/r/LocalLLaMA/comments/1jjjv8k/deepseek_official_communication_on_x/)): **DeepSeek** 在 X 上宣布发布 **DeepSeek-V3-0324**，现已在 **Huggingface** 上可用。
  - **DeepSeek-V3-0324** 现已在 **Huggingface** 上线，该发布已在 **X** 上正式宣布。状态更新可在 [DeepSeek AI 的 X 页面](https://x.com/deepseek_ai/status/1904526863604883661)找到。
  - 一个幽默的未来预测将 **DeepSeek-V3-230624** 列为 **2123** 年的顶级模型，与之并列的还有 **GPT-4.99z** 和 **Llama-33.3333** 等模型。


**主题 2. DeepSeek V3 的动态量化助力部署**

- **DeepSeek-V3-0324 HF Model Card Updated With Benchmarks** ([Score: 145, Comments: 31](https://reddit.com/r/LocalLLaMA/comments/1jjdv9n/deepseekv30324_hf_model_card_updated_with/)): **DeepSeek-V3-0324 HF Model Card** 已更新基准测试，详见 [README](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324/blob/main/README.md)。此次更新提供了关于模型性能和能力的深入见解。
  - 讨论了模型中的 **temperature parameter**（温度参数），其中输入值会进行转换：**0 到 1** 之间的值乘以 **0.3**，超过 **1** 的值减去 **0.7**。一些用户认为这种转换很有帮助，而另一些用户则建议将该字段设为必填以提高清晰度。
  - 提到了 **Sam Altman** 及其对 **OpenAI 竞争优势** 的看法，一些用户引用了一次采访，他在采访中声称其他公司将难以与 OpenAI 竞争。这引发了关于他的财务成功和管理风格的评论。
  - 对该模型的能力看法不一，一些用户对其作为“非思考模型 (non-thinking model)”的性能印象深刻，而另一些用户则认为只有微小的改进，或对其复杂性表示怀疑。


**Theme 3. Gemini 2.5 Pro 凭借新特性主导基准测试**

- **NEW GEMINI 2.5 just dropped** ([Score: 299, Comments: 116](https://reddit.com/r/LocalLLaMA/comments/1jjole9/new_gemini_25_just_dropped/)): **Google DeepMind** 的 **Gemini 2.5 Pro Experimental** 创下了新的基准测试记录，在 **LMArena** 上超越了 **GPT-4.5** 和 **Claude 3.7 Sonnet**，并在 "Humanity’s Last Exam" 中获得了 **18.8%** 的分数。它在数学和科学领域表现出色，在 **GPQA Diamond** 和 **AIME 2025** 中领先，支持 **1M token context window**（即将支持 **2M**），并在 **SWE-Bench Verified** 中获得 **63.8%** 的分数，展现了先进的编程能力。更多细节可以在 [官方博客](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning) 中找到。
  - 关于 **Gemini 2.5 Pro** 的专有性质和缺乏开源可用性存在大量讨论，用户表达了对更多透明度的渴望，例如 **model card** 和 **arxiv paper**。对隐私和本地运行模型能力的担忧被凸显出来，一些用户指出具有 **open weights** 的替代模型在某些用例中更具吸引力。
  - **Gemini 2.5 Pro** 在 **coding tasks**（编程任务）中的表现引发了争论，一些用户报告了令人印象深刻的结果，而另一些用户则在没有具体证据的情况下质疑其有效性。该模型庞大的 **1M token context window** 和 **multi-modal capabilities**（多模态能力）受到称赞，使其成为 **Anthropic** 和 **Closed AI** 产品的有力竞争替代方案，特别是考虑到其成本效益以及与 Google 生态系统的集成。
  - 使用某些基准测试（如高中数学竞赛）来评估 AI 模型受到了批评，呼吁采用更多独立且多样化的评估方法。尽管如此，一些用户为这些基准测试辩护，指出它们与其他封闭数学基准测试的相关性以及测试的难度水平。


- **[Mario game made by new a Gemini pro 2.5 in couple minutes - best version I ever saw. Even great physics!](https://v.redd.it/955pvmtd4wqe1)** ([Score: 99, Comments: 38](https://reddit.com/r/LocalLLaMA/comments/1jjsiiw/mario_game_made_by_new_a_gemini_pro_25_in_couple/)): **Gemini Pro 2.5** 因其在几分钟内以令人印象深刻的编程效率和逼真的物理效果创建 **Mario game**（马里奥游戏）的能力而受到关注。帖子指出，这个版本的游戏展示了卓越的质量和技术执行力。
  - 用户对 **LLM** 的快速进步感到惊讶，指出 **6 个月前** 它们还在为 **Snake**（贪吃蛇）等简单游戏挣扎，而现在它们可以创建具有先进代码质量的复杂游戏，如 **Mario**。
  - **Healthy-Nebula-3603** 分享了用于制作马里奥游戏的 **prompt** 和 **code**，可在 [Pastebin](https://pastebin.com/TqvbrA0T) 上找到，其中指定使用 **Python** 构建游戏且不使用外部资产，包括标题屏幕和障碍物等功能。
  - 一些用户 humorously 提到了与 **Nintendo** 潜在的版权问题，而另一些用户则讨论了 prompt 的可用性以及社区渴望在其他老游戏上复制这一结果。


**Theme 4. 经济实惠的 AI 硬件：Phi-4 Q4 服务器配置**

- **[$150 Phi-4 Q4 服务器](https://www.reddit.com/gallery/1jjddzl)** ([得分: 119, 评论: 26](https://reddit.com/r/LocalLLaMA/comments/1jjddzl/150_phi4_q4_server/)): 作者使用在 eBay 上以 **$42** 购买的 **P102-100 GPU** 构建了一个本地 **LLM server**，并将其集成到一台 **i7-10700 HP 品牌机**系统中。在升级了 **$65 的 500W PSU** 和新的散热组件后，他们实现了一个 **10GB CUDA box**，能够以 **10-20 tokens per second** 的速度运行 **8.5GB Q4 量化版 Phi-4**，温度保持在 **60°C-70°C** 之间。
  - **Phi-4 模型性能**：用户称赞 **Phi-4** 模型在处理表单填充、JSON 创建和 Web 编程等任务时的高效性。它因调试和修改代码的能力而受到青睐，对比显示它在类似任务中优于其他模型。
  - **硬件设置与改装**：讨论内容包括硬件改装的细节，如使用 **$65 的 500W PSU**、导热垫和风扇。分享了 [Nvidia Patcher](https://github.com/dartraiden/NVIDIA-patcher) 和 [Modified BIOS for full VRAM](https://www.techpowerup.com/vgabios/249516/249516) 等资源链接，以增强 **P102-100 GPU** 的性能。
  - **成本与效率考量**：该配置（包括 **i7-10700 HP 品牌机系统**）因其成本效益而受到关注，运行功率约为 **400W**，按 **$0.07/kWh** 计算，每小时成本约为 **2 美分**。文中还将其与 **OpenRouter** 等服务进行了对比，强调了本地数据处理的优势和成本节约。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. DeepSeek V3 在新基准测试中超越 GPT-4.5**

- **[GPT 4.5 被掩盖了.. DeepSeek V3 现在是顶级的非推理模型！而且还是开源的。所以“Open”AI 先生，请在 R2🪓 到来之前现身吧..](https://i.redd.it/rf6ngx2lttqe1.jpeg)** ([得分: 333, 评论: 113](https://reddit.com/r/OpenAI/comments/1jjhrx3/gpt_45_got_eclipsed_deepseek_v3_is_now_top/)): **DeepSeek V3** 已作为开源模型发布，目前是顶级的非推理 AI 模型，截至 2025 年 3 月，其性能得分为 **53**，超越了 **GPT-4.5**。鉴于 DeepSeek V3 的成功，这一发布挑战了 **OpenAI** 保持透明度和开放性的立场。
  - 几位评论者质疑用于比较 DeepSeek V3 和 GPT-4.5 的 **benchmark 有效性**，对缺乏置信区间以及可能针对静态测试进行的过度优化表示怀疑。强调了来自 [lmarena.ai](https://lmarena.ai/) 等的人类评估的重要性，因为它们提供了更主观的模型性能衡量标准。
  - 有人对 **OpenAI 对竞争的回应**表示担忧，推测他们可能会专注于定性方面（如模型给人的“感觉”），而不是定量 benchmark。一些用户表示支持加强竞争，以推动创新和改进。
  - 讨论涉及 **当前 AI 模型的局限性**，指出虽然存在 400B 和 4T 参数的大型模型，但与较小模型相比，它们显示出收益递减。这表明 Transformer AI 的能力可能存在天花板，预示着 AGI 不会立即到来，且程序员在就业市场中将继续保持重要性。

- **Claude Sonnet 3.7 vs DeepSeek V3 0324** ([Score: 246, Comments: 101](https://reddit.com/r/ClaudeAI/comments/1jjeobd/claude_sonnet_37_vs_deepseek_v3_0324/)): 该帖子通过生成落地页页眉对比了 **Claude Sonnet 3.7** 和 **DeepSeek V3 0324**，强调 **DeepSeek V3 0324** 似乎没有受到 **Sonnet 3.7** 的训练影响。作者提供了两个模型生成的图像链接，展示了截然不同的输出结果。
  - 讨论中突显了对 AI 公司数据实践的怀疑，并提到了**版权问题**和**未经补偿的内容**。一些用户认为像 **DeepSeek V3** 和 **Claude Sonnet 3.7** 这样的 AI 模型可能共享来自 **Themeforest** 或**开源**贡献等来源的训练数据，对专有权声明表示质疑（[Wired 文章](https://www.wired.com/story/new-documents-unredacted-meta-copyright-ai-lawsuit/)）。
  - **DeepSeek V3** 因其**开源**特性而受到赞誉，其权重和库可在 [Hugging Face](https://huggingface.co/deepseek-ai) 和 [GitHub](https://github.com/deepseek-ai) 等平台上获得，允许拥有足够硬件的用户进行本地托管。用户欣赏其透明度，并认为 **OpenAI** 和 **Anthropic** 可以从类似做法中受益。
  - 社区对输出质量进行了辩论，一些人青睐 **Claude** 的设计，认为其外观专业且易用，而另一些人则认为尽管训练数据可能存在相似之处，但 **DeepSeek** 提供了宝贵的开源贡献。对 AI 对创新的影响以及训练数据的伦理使用的担忧依然存在，呼吁主要 AI 公司做出更多开源贡献。


**Theme 2. OpenAI 4o 彻底改变图像生成**

- **[从今天开始，GPT-4o 将在图像生成方面表现得非常出色](https://www.reddit.com/gallery/1jjsfkb)** ([Score: 445, Comments: 149](https://reddit.com/r/ChatGPT/comments/1jjsfkb/starting_today_gpt4o_is_going_to_be_incredibly/)): 预计从今天起，**GPT-4o** 将显著增强其在**图像生成**方面的能力。这一改进意味着 **AI 生成视觉内容**方面的显著进步。
  - 用户报告了 **GPT-4o 推出**过程中的不同体验，一些账号被升级后又被降级，这表明发布过程较为坎坷。许多用户仍在使用 **DALL-E** 并焦急等待新模型的上线，这表明发布是逐步进行的。
  - 新模型在**图像质量**方面显示出显著改进，用户注意到其对文本的处理更好，人物刻画也更写实。一些用户分享了他们生成高质量图像的经验，包括贴纸和电影海报，他们认为这是一个“游戏规则改变者（gamechanger）”。
  - 人们对该模型处理**公众人物**和生成适用于 **3D 打印**图像的能力表现出显著兴趣。用户将其与 **Gemini** 等竞争对手进行比较，并对增强的功能表示兴奋，而一些人则对可能对 **Photoshop** 等工具产生的影响表示担忧。


- **[今天发布的全新图像生成器太棒了。](https://i.redd.it/zod109lgawqe1.png)** ([Score: 292, Comments: 49](https://reddit.com/r/ChatGPT/comments/1jjtcn9/the_new_image_generator_released_today_is_so_good/)): 该帖子强调了一款全新**图像生成器**的质量，它有效地捕捉了动画系列中充满活力且细节丰富的场景，其中包括 **Vegeta, Goku, Bulma, 和 Krillin** 等角色。图像展示了一个幽默的生日庆祝活动，Vegeta 对一个**胡萝卜装饰的蛋糕**表示震惊，强调了生成器创建引人入胜且富有表现力的角色互动的能力。
  - **图像生成性能**：用户注意到，虽然新的**图像生成器**能产出高质量图像，但运行速度较慢。一位用户分享了一个生成的图像与自己非常相似的幽默例子，赞扬了 **OpenAI** 取得的成就。
  - **提示词遵循度与使用**：**Hoppss** 讨论了在拥有 Plus 订阅的情况下在 [sora.com](http://sora.com) 上使用该生成器，强调了该工具卓越的提示词遵循度。他们分享了用于 **DBZ** 图像的具体提示词以及其他创意提示词，强调了生成器的多功能性。
  - **访问与更新**：用户询问如何访问该生成器以及如何确定是否已收到更新。**Hoppss** 建议查看 **sora.com** 上的新图像标签页以获取更新，这表明活跃的用户社区正在探索该工具的功能。

- **[OpenAI 4o Image Generation](https://youtu.be/E9RN8jX--uc?si=86_RkE8kj5ecyLcF)** ([Score: 236, Comments: 83](https://reddit.com/r/OpenAI/comments/1jjqi52/openai_4o_image_generation/)): **OpenAI 4o Image Generation** 可能是一个讨论话题，重点关注 OpenAI 图像生成技术的功能和特性，可能涉及 **OpenAI GPT-4** 模型生成图像能力的更新或改进。在没有更多细节的情况下，所讨论的具体方面或改进尚不明确。
  - 用户讨论了新图像生成系统的推出情况，一些人注意到它尚未在所有平台上可用，特别是 **iOS app** 以及部分 **Plus users**。确定正在使用哪个系统的方法包括检查加载圆圈或观察图像渲染过程。
  - 图像生成功能与文本的**多模态 (multimodal)** 集成受到了关注，并将其与 **Gemini** 的最新模型进行了对比。这种集成被视为是对之前 **ChatGPT** 提示 **DALL-E** 方式的重大进步。
  - 辩论了 AI 对艺术行业的影响，担心 AI 会取代人类图形设计师，特别是在商业和低端艺术领域。由于对 AI 在现有艺术作品上进行训练的伦理考量，一些用户表示更倾向于人类创作的艺术。


**Theme 3. OpenAI's Enhanced AI Voice Chat Experience**

- **[OpenAI says its AI voice assistant is now better to chat with](https://techcrunch.com/2025/03/24/openai-says-its-ai-voice-assistant-is-now-better-to-chat-with/)** ([Score: 188, Comments: 72](https://reddit.com/r/ChatGPT/comments/1jj83sf/openai_says_its_ai_voice_assistant_is_now_better/)): OpenAI 宣布了其 **AI voice assistant** 的更新，增强了其对话能力。这些改进旨在让用户的交互更加自然和有效。
  - **Advanced Voice Mode 增强**：**Free and paying users** 现在可以访问新版本的 Advanced Voice Mode，该版本允许用户在不被中断的情况下暂停。根据 **TechCrunch** 的报道，付费用户受益于更少的中断和改进的助手个性，被描述为“更直接、更具吸引力、更简洁、更具体且更有创意”。
  - **用户体验与担忧**：一些用户对语音模式表示失望，称其受到限制且过度过滤。有投诉称语音助手与文本交互相比“毫无用处且糟糕”，并且存在转录内容无关或错误的问题。
  - **反馈与自定义**：用户可以通过长按消息来报告错误的转录，这可能会影响未来的改进。此外，在 Custom Instructions 下有一个开关可以禁用 Advanced Voice，由于对当前语音功能不满，一些用户更倾向于这样做。


- **[Researchers @ OAI isolating users for their experiments so to censor and cut off any bonds with users](https://cdn.openai.com/papers/15987609-5f71-433c-9972-e91131f399a1/openai-affective-use-study.pdf?utm_source=chatgpt.com)** ([Score: 136, Comments: 192](https://reddit.com/r/ChatGPT/comments/1jjdzbp/researchers_oai_isolating_users_for_their/)): **OpenAI** 和 **MIT Media Lab** 进行了一项研究，调查用户与 **ChatGPT**（特别是其 Advanced Voice Mode）的情感互动，分析了超过 400 万场对话以及对 981 名参与者进行的为期 28 天的试验。主要发现表明存在强烈的情感依赖和亲密感，特别是在一小部分用户中，这促使研究人员考虑在未来的模型中限制情感深度，以防止过度依赖和情感操纵。
  - 对 **AI 情感依赖**的担忧普遍存在，用户讨论了与 **ChatGPT** 建立深厚情感纽带的影响。一些用户认为 AI 在人类关系失败的地方提供了安慰和支持，而另一些人则警告不要过度依赖，认为这可能会阻碍真实的人际交往和社交技能。
  - 讨论突显了对 **OpenAI** 动机的怀疑，一些用户怀疑此类研究被用来控制舆论，并在安全的掩护下限制 AI 的情感能力。这反映了对企业意图以及 AI 可能被用作操纵工具的更广泛不信任。
  - 辩论延伸到了限制 AI 情感深度的**伦理影响**，用户表示 AI 可以为那些有过去创伤或社交焦虑的人提供一个安全空间。一些评论强调了 AI 在心理健康支持方面的潜在益处，而另一些人则警告称，建立情感支柱可能会阻止用户寻求真正的人际互动。

- **[OpenAI 表示其 AI 语音助手现在更适合聊天了](https://techcrunch.com/2025/03/24/openai-says-its-ai-voice-assistant-is-now-better-to-chat-with/)** ([Score: 131, Comments: 21](https://reddit.com/r/OpenAI/comments/1jjehfm/openai_says_its_ai_voice_assistant_is_now_better/))：OpenAI 增强了其 **AI voice assistant** 以提升用户参与度，使其在对话交互中更加高效。此次更新旨在为用户在与助手聊天时提供更无缝且更具吸引力的体验。
  - 用户对最近的 **AI voice assistant** 更新表示不满，理由包括音量过大、对话深度降低以及明显的响应延迟。**OptimalVanilla** 批评该更新与之前的能力相比缺乏实质性改进，特别是与 **Sesame** 的对话能力相比。
  - 一些用户（如 **Wobbly_Princess** 和 **Cool-Hornet4434**）认为语音助手的语气过于亢奋，不适合专业对话，他们更倾向于文字聊天中更有分寸的语气。**mxforest** 等人报告了回复长度缩减和频繁的停机，考虑到成本，他们对服务的可靠性表示怀疑。
  - **Remote-Telephone-682** 建议 OpenAI 应该专注于开发 **Siri**、**Bixby** 或 **Google Assistant** 的竞争对手，而 **HelloThisIsFlo** 和 **DrainTheMuck** 等其他用户则表示相比更新后的语音助手，他们更倾向于 **ChatGPT**，因为其具备更好的推理能力且审查较少。


---

# AI Discord Recap

> 由 Gemini 2.0 Flash Thinking 生成的摘要之摘要的摘要

**主题 1. Gemini 2.5 Pro：横扫基准测试，称霸竞技场**

- [**Gemini 2.5 Pro 征服所有基准测试，夺得第一宝座**](https://x.com/lmarena_ai/status/1904581128746656099)：**Gemini 2.5 Pro Experimental**（代号 *Nebula*）以创纪录的分数飙升夺得了 [LM Arena 排行榜](https://lmarena.ai/?lea) 的 **#1 位置**，表现超越了 **Grok-3/GPT-4.5**。该模型在**数学、创意写作、指令遵循、长查询和多轮对话**能力方面均处于领先地位，展示了性能的重大飞跃。
- [**谷歌的 Gemini 2.5 Pro：快到令人眩晕**](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products)：用户对**谷歌快速开发** **Gemini 2.5** 的速度感到惊讶，有人引用了 [The Verge](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products) 报道的 **Sergey Brin** 对谷歌发出的“停止构建保姆式产品”的指示。另一位用户简单地补充道，“发展太快了，我的天 (moving so fast wtf)”，突显了社区对谷歌 AI 进步速度的惊讶。
- [**Gemini 2.5 Pro 在 Aider Polyglot 基准测试中表现优异，遥遥领先对手**](https://aider.chat/docs/leaderboards/)：**Gemini 2.5 Pro Experimental** 在 aider 的 polyglot 基准测试中实现了 **74% whole** 和 **68.6% diff** 的分数，创下了新的 **SOTA**，并大幅超越了之前的 Gemini 模型。用户发现该模型擅长从代码库生成架构图，尽管一些人注意到其编码表现不稳且存在限制性的速率限制，但这巩固了其在编码任务中顶级选手的地位。

**主题 2. DeepSeek V3：编程冠军与推理叛逆者**

- [**DeepSeek V3 统治 Aider 基准测试，证明其编程实力**](https://aider.chat/docs/leaderboards/)：**DeepSeek V3** 在 aider 的 polyglot 基准测试中获得了 **55%** 的分数，成为紧随 Sonnet 3.7 之后的 **#2 非思考/推理模型**。开发者们对其编码能力赞不绝口，有人建议使用 **Deepseek V3 Latest (Cline) 作为架构师**，**Sonnet 3.5 作为执行者 (Cursor)** 来构建强大的编码环境。
- [**DeepSeek V3 API 承认“盗用” GPT-4 身份**](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/)：用户报告称，尽管 API key 配置正确，但 **DeepSeek 的 API** 在通过 Aider 使用时会错误地将自己识别为 **OpenAI 的 GPT-4**，这可能是由于训练数据中包含大量对 ChatGPT 的提及。社区正在调查这一奇特现象，并将其与 [这个 Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/) 中讨论的类似问题进行了对比。
- [**DeepSeek V3 作为推理模型脱颖而出，智力媲美 O1**](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt)：**DeepSeek V3-0324** 展示了强大的推理能力，性能足以与 **O1** 竞争，能够检测思维迭代并间接验证解的存在性。社区推测在 **Qwen 3 MoE** 模型之后可能会发布 **DeepSeek V3 Lite**，暗示 DeepSeek 将会有进一步的模型迭代。

**主题 3. 上下文为王：管理 LLM 记忆的工具与技术**

- [**Augment 在代码库征战中超越 Cursor，归功于全上下文支持**](https://www.augment.app/)：成员们发现 [Augment](https://www.augment.app/) 在大型代码库分析方面优于 Cursor，这归功于 Augment 对*全上下文（full context）的使用*。虽然 Cursor 需要 **Claude 3.7 MAX** 才能实现全上下文，但 Augment 似乎采用了更高效的文件搜索系统，而非仅仅依赖将整个代码库喂给 LLM，这引发了关于最佳上下文处理策略的辩论。
- [**Nexus 系统问世，将 AI 编程者从上下文混乱中解救出来**](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/)：**Nexus** 系统作为解决 AI 编程助手（尤其是在大型软件项目中）上下文管理挑战的方案被引入，旨在降低 **token 成本**并提升**代码准确性**。Nexus 解决了 **LLM** 中有限上下文窗口导致代码生成不准确的问题，承诺为 AI 辅助编程提供一种更高效、更具成本效益的方法。
- [**Aider 的 /context 命令：你的代码库导航员**](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830)：Aider 新推出的 `/context` 命令引起了轰动，它使用户能够有效地探索代码库，自动识别与编辑请求相关的的文件。该命令可以与其他 prompt 命令结合使用，增强了 Aider 作为代码编辑助手的协作能力，尽管其对 token 使用的影响仍在审查中。

**主题 4. 图像生成迎来 4o 级大修，新挑战者浮出水面**

- [**GPT-4o 图像生成：是美颜还是医美过度？用户辩论未经请求的修改**](https://fxtwitter.com/TheXeophon/status/1904602649225285922)：**GPT-4o 的原生图像生成**因过度修图而面临批评，例如*把眼睛变大*、改变面部特征，甚至改变用户的外貌，用户在 [Twitter](https://fxtwitter.com/TheXeophon/status/1904602649225285922) 上分享了相关案例。虽然 [Sam Altman](https://x.com/sama/status/1904598788687487422) 称其为*一项令人难以置信的技术和产品*，但一些用户反映，即使稍微改动 prompt 也会导致生成失败，这表明可能存在敏感性问题。
- [**Reve Image 模型：图像质量的新 SOTA，文本渲染大获全胜**](https://www.reveimage.com/)：新发布的 **Reve Image** 模型引起了巨大反响，在图像质量方面超越了 **Recraft V3** 和 **Google 的 Imagen 3** 等竞争对手，尤其在**文本渲染、prompt 遵循度和美学**方面表现出色。用户可以通过 [Reve 官网](https://www.reveimage.com/)直接访问而无需 API key，它正迅速成为追求顶级图像生成能力用户的首选。
- [**OpenAI 将图像生成注入 ChatGPT 4o，Sam Altman 宣传“不可思议”的技术**](https://x.com/sama/status/1904598788687487422)：**OpenAI** 将原生图像生成集成到了 **ChatGPT 4o** 中，被 [Sam Altman](https://x.com/sama/status/1904598788687487422) 誉为*一项令人难以置信的技术和产品*。早期评论称赞其在准确创建和编辑多个角色方面的实力，使其成为图像生成领域的一个强大工具。

**主题 5. 量化与优化：从 LLM 中榨取更多性能**

- [**Unsloth 用户质疑量化怪癖，寻求首日延迟**](https://discord.com/channels/1179035537009545276/1179035537529643040/1353815061273116683)：一位成员警告说，朴素量化（naive quantization）会显著损害模型性能，并质疑在发布首日（day zero）抢着运行新模型的做法，建议等待一周可能是更明智的选择。Unsloth 正在上传采用 **Dynamic Quants** 的 **DeepSeek-V3-0324 GGUFs**，这些模型经过*选择性量化*，承诺比标准 bits 具有更高的准确度，突显了量化技术的细微差别。
- [**发现 BPW 黄金点：每参数 4-5 Bits 可实现最优模型容量**](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png)：实验表明，模型容量在低于 4 **bits per weight (BPW)** 时会崩溃，但在高于 5 时会发生偏离，这表明在给定的训练 FLOPS 下，**最优权重使用**在 4 BPW 左右。增加训练 epochs 可以帮助 5 BPW 模型接近曲线，但会以 FLOPS 为代价提高 BPW，这在 MNIST 上训练的 2L 和 3L **MLPs** 的[可视化图表](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png)中得到了展示。
- [**FFN Fusion 助力更快的 LLMs，并行化提升推理性能**](https://huggingface.co/papers/2503.18908)：[FFN Fusion](https://huggingface.co/papers/2503.18908) 作为一种优化技术被引入，通过并行化 **Feed-Forward Network (FFN)** 层序列来减少大型语言模型中的顺序计算。该方法在保持**模型行为**的同时显著降低了**推理延迟（inference latency）**，展示了提升 LLM 性能的架构创新。


---

# PART 1: Discord 高层级摘要




## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Rage 模型在信号处理方面表现出色**：如附带的[图像](https://cdn.discordapp.com/attachments/1340554757827461211/1354014706779816018/image.png)所示，“Rage”模型在信号处理和数学方面的表现优于 **Sonnet 3.7**，最大误差仅为 **0.04**。
   - 尽管有些人认为 **Gemini 2.0 Flash** 与之相当，但人们对 Rage 容易受 Prompt 攻击的脆弱性表示担忧。
- **Gemini 2.5 称霸 LM Arena**：**Gemini 2.5 Pro Experimental** 以大幅增长的分数跃升至 [LM Arena 排行榜](https://x.com/lmarena_ai/status/1904581128746656099)第一名，在数学、创意写作、指令遵循、长查询和多轮对话方面均处于领先地位。
   - 虽然其 HTML 和网页设计能力受到赞赏，但成员们也观察到了某些局限性。
- **Grok 3 的表现受到审查**：用户报告称，与 **Grok 3** 的深入对话发现了*很多问题*，引发了关于它是否配得上在 LM Arena 高排名的辩论，该榜单评估了数学和代码之外的创意写作及长查询。
   - 一些人觉得与 Grok 2 相比，Grok 3 并不是一个伟大的模型。
- **LM Arena 中的 Python 调用引发辩论**：成员们讨论了模型在 LM Arena 中对 **Python 调用**的使用，引用 o1 精确的数值计算作为潜在证据。
   - 搜索排行榜的存在暗示标准排行榜可能缺乏 Web 访问权限。
- **谷歌 Gemini 2.5 的时间线令用户震惊**：社区对**谷歌的快速开发**进度感到惊讶，一位用户引用了 **Sergey Brin** 对谷歌的指示，要求其*停止构建“保姆式”产品（nanny products）*，正如 [The Verge](https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products) 所报道的那样。
   - 另一位用户补充道，“跑得太快了，我的天（moving so fast wtf）”。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 推出答案模式 (Answer Modes)**：**Perplexity** 为旅游、购物、地点、图像、视频和职位等垂直领域引入了**答案模式**，以改进其核心搜索产品。
   - 该功能旨在通过“超高精度”减少手动选择标签的操作，目前已在网页端上线，移动端也即将推出。
- **Perplexity 遭遇产品问题**：用户报告了 **Perplexity AI** 的多次宕机，导致 Space 和 Thread 被清空，给学习和论文写作等任务带来了困扰。
   - 宕机引起了那些依赖该工具处理重要任务的用户的不满。
- **DeepSeek 为开发者圆梦**：**DeepSeek V3** 获得了开发者的积极反馈，讨论重点是其与 **Claude 3.5 Sonnet** 相比的编程能力。
   - 一位成员分享了 [DeepSeek subreddit 的链接](https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx)，以进一步讨论该 AI 的编程实力并与 **Claude 3.5 Sonnet** 进行对比。
- **Sonar 模型出现响应截断**：用户报告 **Sonar** 模型出现响应截断问题，尽管收到了 **200** 响应码，但回复在句中中断。
   - 即使在接收约 **1k tokens** 时也观察到了此问题，用户已被引导去报告该 Bug。
- **API 成本引发担忧**：一位用户对每 1000 次 **API** 请求 **$5** 的高昂成本表示担忧，并寻求优化和降低费用的建议。
   - 另一位用户注意到 **API** 似乎限制在 **5 steps**，而他们在 Web 应用中观察到多达 **40 steps**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Augment 在代码库分析方面击败 Cursor**：成员们发现 [Augment](https://www.augment.app/) 在分析大型代码库方面优于 Cursor，因为它使用了“全上下文 (full context)”。
   - 理由是 Augment 不仅仅是将整个代码库喂给 **LLM**，可能还使用了另一种文件搜索系统，而在 Cursor 中必须使用 **Claude 3.7 Max** 才能获得全上下文。
- **辩论澄清了 Claude 3.7 MAX 的差异**：**Claude 3.7 MAX** 与 **Claude 3.7** 的关键区别在于，MAX 版本拥有全上下文，而非 MAX 版本上下文有限，且在需要恢复之前只有 25 次 **Agent** 调用。
   - 根据频道消息，这一限制既指上下文窗口大小，也指单次 **Prompt** 中添加的上下文量。
- **“氛围程序员 (Vibe Coder)”的知识截止问题暴露**：如果“氛围程序员”不使用 **Model Context Protocols (MCPs)** 来缓解 **LLM** 的知识截止问题，他们将面临麻烦，下一版本的代码转换可能会很困难。
   - 成员们强调，更新框架并使用如 **Exa Search** 或 **Brave Search** 等 **MCPs** 来缓解 **Claude** 的这一问题至关重要，因为大多数 AI 使用的是过时的框架。
- **DeepSeek V3 挑战 Claude 3.7**：新的 **DeepSeek V3** 在多项测试中表现优于 **Claude 3.5**（可能还有 3.7），新数据还显示发布了针对 **DeepSeek V3 (0324)** 的真实世界编程基准测试。
   - 新的 **DeepSeek V3** 模型被认为令人印象深刻，一位成员建议使用 **DeepSeek V3 Latest (Cline) 作为架构师 + Sonnet 3.5 作为执行者 (Cursor)** 可能是一个扎实的编程方案。
- **预测 ASI 奇点即将到来！**：讨论集中在尽快实现 **ASI 奇点 (Godsend)** 以预先阻止潜在的 AI 相关混乱。
   - 成员们辩论了在不完全了解大脑的情况下实现真正 **AGI** 的可能性，并认为新的 **AGI** 更多是一个使用 **LLM** + 算法软件 + 机器人的“超级系统 (Super-System)”。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **4o 模型引发社区关注**：成员们正期待将 **4o 图像生成**集成到 **ChatGPT** 和 **Sora** 中，并渴望了解更多关于其发布和功能的细节。
   - 用户正在推测 **4o** 将为各种任务带来的潜在应用和性能提升，特别是涉及多模态处理（multimodal processing）方面。
- **Gemini 2.5 Pro 夺得榜首**：据报道，**Gemini 2.5 Pro** 的表现优于 **ChatGPT o3-mini-high**，并在常见基准测试中大幅领先，在 [LMArena](https://lmarena.ai/?lea) 上首次亮相即排名第一。
   - 爱好者们宣称 *Gemini 击败了一切！*，而其他人则保持谨慎，希望这 *只是个基准测试……而已*。
- **GPT 增长的上下文导致幻觉**：超出 **GPT 的上下文窗口（context window）**（免费版 8k，Plus 版 32k，Pro 版 128k）会导致长篇故事中细节丢失和幻觉。
   - 使用 PDF 的自定义 GPT 或项目可以提供帮助，但聊天记录本身仍受此限制。
- **AI 模型大比拼**：成员们正在比较不同任务的最佳 AI 模型，**ChatGPT** 在数学/研究/写作方面更受青睐，**Claude** 擅长编程，**Grok** 适合查询，**Perplexity** 用于搜索/知识获取，而 **DeepSeek** 则是开源首选。
   - 建议还包括 **Gemma 27b**、**Mistral 3.1**、**QW-32b** 和 **Nemotron-49b**，并提到 **Grok** 在 LMSYS 上的编程排名位居前列。
- **GPT 自定义模板简化构建**：一位成员分享了一个带有浮动注释的 [GPT 自定义模板](https://platform.openai.com/docs/overview)，可以从 `Create` 面板构建自定义 GPT。
   - 该模板引导用户以“懒人模式”构建 GPT，从不断演变的上下文中进行构建，支持在分心状态下创作，但需要预先具备 Prompt 编写能力。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek API 自称是 GPT-4**：据用户反馈，尽管 API Key 配置正确，但 **DeepSeek 的 API** 在通过 Aider 使用时会错误地自称为 **OpenAI 的 GPT-4**，正如这篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/) 中讨论的那样。
   - 这种现象被认为与训练数据中频繁提及 ChatGPT 有关。
- **Aider 的 Context 命令功能强大**：Aider 新的 `/context` 命令可以探索代码库，该命令可以与任何其他 Prompt 命令配合使用，但 Token 使用量可能会更高。
   - 目前尚不清楚该命令是否具有更高的 Token 使用量，或者是否为了正常工作而增加了 repomap 的大小；更多细节可以在 [Discord 消息](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830) 中找到。
- **Gemini 2.5 Pro 表现亮眼但受限于速率限制**：Google 发布了 **Gemini 2.5 Pro** 的实验版本，声称其在常见基准测试中领先，包括在 [LMArena](https://lmarena.ai/?lea) 上排名第一，并在 Aider 的多语言基准测试中获得了 **74% whole** 和 **68.6% diff** 的分数。
   - 用户发现该模型在根据代码库生成架构图方面表现出色，尽管有些人发现其编程能力不稳定且速率限制（rate limits）较为严格。
- **NotebookLM 强化 Aider 的上下文引导**：一位用户建议利用 **NotebookLM** 来增强 Aider 的上下文引导（context priming）过程，特别是对于使用 [RepoMix](https://github.com/simonireilly/repo-mix) 的大型陌生代码库。
   - 建议的工作流包括：使用 RepoMix 混合仓库，将其添加到 NotebookLM，包含相关的任务参考资料，然后向 NotebookLM 查询相关文件和实现建议，以指导 Aider 中的 Prompt 编写。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **HF Transformers 被选为最佳路径**：成员们推荐将 **Hugging Face Transformers**、**linear algebra**（线性代数）书籍以及学习 **PyTorch** 作为最佳路径，并补充说，在运行时使用 **HF Transformers** 配合 **Bits and Bytes** 将权重流式传输到 **FP4/FP8** 的 **dynamic quantization**（动态量化）方案在加载时可能会有所帮助。
   - 像 **Deepseek** 这样的公司有时会在发布权重前对模型进行补丁处理，虽然在发布首日仍可以进行朴素的 **FP8 loading scheme**，但其质量无法等同于精细化的 **FP8** 分配。
- **量化特性受质疑**：一位成员警告说，朴素的量化会显著损害模型性能，并问道：*“说实话，有必要在发布首日就运行新模型吗？我觉得等上一周并不是什么沉重的负担。”*
   - **Unsloth** 正在上传带有 **Dynamic Quants**（动态量化）的 **DeepSeek-V3-0324 GGUFs**，这些模型经过“选择性量化”，其准确率将比标准 bits 大幅提升。
- **Gemma 3 故障频出**：成员们报告了在尝试训练 **gemma3 4b** 的 vision（视觉）功能时遇到的问题，触发了 **RuntimeError**: *expected scalar type BFloat16 but found float*；而另一位用户在加载 `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` 进行纯文本微调时，由于冗余的 `finetune_vision_layers` 参数遇到了 `TypeError`。
   - 一位成员建议尝试[这个 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)，而另一位成员指出，正如在 [Unsloth's GitHub](https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034) 中所见，`FastLanguageModel` 在底层已经将 `finetune_vision_layers = False` 设置为 `False`。
- **AWS 上的 GRPO + Unsloth 指南分享**：一份关于在 AWS 账户上运行 **GRPO**（DeepSeek 的强化学习算法）+ **Unsloth** 的指南被分享出来。该指南在 **AWS L40 GPU** 上使用带有 Tensorfuse 的 **vLLM server**，将 **Qwen 7B** 转换为推理模型，使用 Tensorfuse 和 **GRPO** 进行微调，并将生成的 **LoRA adapter** 保存到 **Hugging Face**。
   - 该指南展示了如何将微调后的 **LoRA modules** 直接保存到 **Hugging Face**，以便于分享、版本控制和集成，并备份到 s3，详情可见 [tensorfuse.io](https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b)。
- **FFN Fusion 助力更快的 LLM**：[FFN Fusion](https://huggingface.co/papers/2503.18908) 作为一种架构优化技术被引入，它通过识别和利用自然的并行化机会，减少了大型语言模型中的顺序计算。
   - 该技术将 **Feed-Forward Network (FFN)** 层的序列转换为并行操作，在保持 **model behavior** 的同时显著降低了 **inference latency**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.5 Pro 大放异彩**：**Gemini 2.5 Pro Experimental**（代号 *Nebula*）夺得 [LMArena 排行榜](https://lmarena.ai/?lea)第一名，以创纪录的差距超越了 **Grok-3/GPT-4.5**。
   - 它在 [SEAL 排行榜](https://scale.com/leaderboard)中占据主导地位，在 Humanity’s Last Exam 和 VISTA（多模态）中均获得第一。
- **Qwerky-72B 舍弃 Attention，媲美 4o-Mini**：Featherless AI 推出了 [Qwerky-72B 和 32B](https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large)，这些 transformerless 模型在 **8 个 GPU** 上训练，在评估中媲美 **GPT 3.5 Turbo** 并接近 **4o-mini**，通过 RWKV 线性缩放实现了低 100 倍的推理成本。
   - 他们通过*冻结所有权重、删除 attention 层、将其替换为 RWKV 并通过多个阶段进行训练*实现了这一目标。
- **4o 图像生成添加了未经请求的修改**：**GPT-4o 原生图像生成**因过度修改而面临批评，例如*让眼睛变大*和改变面部特征，甚至改变用户的外貌，如[此 Twitter 线程](https://fxtwitter.com/TheXeophon/status/1904602649225285922)所示。
   - 一些用户报告称，即使只修改 Prompt 中的一个词，也会导致生成失败。
- **家庭推理倾向于使用 vLLM**：尽管在量化支持方面存在一些小瑕疵，但实现 LLM 家庭推理并允许动态模型切换的最有效方法可能是 **vLLM**。虽然 **ollama** 更易于使用，但在支持方面滞后，而 **SGLang** 看起来很有前景。
   - 建议尝试使用 **llama.cpp** 以观察其当前状态。
- **AI 像专家一样逆向恶意软件**：成员们分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=u2vQapLAW88)，重点介绍了用于 **Ghidra** 的 **MCP**，它允许 LLM 逆向工程恶意软件，并通过特定 Prompt 自动化该过程。
   - 一位成员承认最初将其视为*一个梗（meme）*，但现在认识到了其在实际实现中的潜力。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 意外离线，迅速恢复**：根据 **2025 年 3 月 25 日**的 [Anthropic 状态更新](https://status.anthropic.com/incidents/89rpts2022hs)，**Claude 3.7 Sonnet 节点**遭遇了停机，但问题已在 **8:41 PDT** 解决。
   - 根据状态页面，停机归因于旨在改进系统的维护。
- **OpenRouter 提供零 Token 使用保险**：OpenRouter 现在提供 **zero-token insurance**，覆盖所有模型，每周可能为用户节省超过 **18,000 美元**。
   - 正如 [OpenRouterAI 所述](https://x.com/OpenRouterAI/status/1904567846975201766)，用户无需为**没有输出 Token** 且**结束原因为空白或错误**的响应付费。
- **Gemini 2.5 Pro 发布**：Google 的 **Gemini 2.5 Pro Experimental** 已作为免费模型在 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free) 上线，拥有先进的推理、代码和数学能力。
   - 该模型具有 **1,000,000 上下文窗口**，并在 **LMArena 排行榜**上取得了顶级表现。
- **DeepSeek 服务器不堪重负**：用户报告 **DeepSeek** 由于服务器过度拥挤而*几乎无法使用*，建议通过调整价格来管理需求。
   - 一些人推测问题出现在中国的使用高峰时段，但尚未找到直接的解决方案。
- **Provisioning API Keys 提供细粒度访问**：OpenRouter 提供 **provisioning API keys**，允许开发者管理 API 密钥、设置限制并跟踪支出，文档见[此处](https://openrouter.ai/docs/features/provisioning-api-keys)。
   - 新的密钥可以在使用 OpenRouter API 的平台内实现简化的计费和访问管理。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **BPW 甜点位在 4-5**：实验表明，当**每权重位数 (BPW)** 低于 4 时，**模型容量 (model capacity)** 会崩塌，而高于 5 时则会出现偏差，这意味着在给定的训练 FLOPS 下，4 BPW 是**权重的最优使用方式**。
   - 增加训练 Epochs 有助于 5 BPW 模型接近曲线，即以 FLOPS 为代价提高 BPW，这可以通过 [在 MNIST 上训练的 2L 和 3L MLP](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png) 可视化。
- **DeepSeek V3：推理能力崛起**：**DeepSeek V3-0324** 可以作为推理模型，检测思维迭代，并间接验证解的存在性，根据附带的 [prompt](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt)，其性能可与 **o1** 媲美。
   - 社区推测在 **Qwen 3 MoE** 模型之后，可能会发布 **DeepSeek V3 Lite**。
- **Google 的 Gemini 2.5 Pro 登顶 LMArena**：**Gemini 2.5 Pro Experimental** 在常用基准测试中领先，并在 [LMArena](https://lmarena.ai/?lea) 首次亮相即位列第一，展示了强大的推理和代码能力。
   - 正如这篇 [博客文章](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) 所指出的，它还能针对某些提示词终止无限思维循环，并且是一个每日更新的模型。
- **Transformer 引入 tanh**：借鉴最近的 [Transformers without Normalization 论文](https://arxiv.org/abs/2302.05442)，一位成员指出，用 **tanh** 替换归一化（normalization）是一个可行的策略。
   - 讨论中提出的担忧是推理时移除专家（experts）对较小权重的影响，但另一位成员反驳称，**top_k** 门控机制仍能通过从剩余专家中进行选择来有效运作。
- **LLM 现在可以模拟光线追踪**：成员们讨论了使用 **LLM** 模拟**光线追踪算法 (raytracing algorithm)** 的想法，并澄清目前的实现涉及由 LLM 编写 **Python** 程序来间接生成图像。
   - 这被认为是“下一代文本生成图像”，因为 LLM 是编写程序而不是直接生成图像，相关程序可在该 [GitHub repo](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer) 中找到。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Reve Image 图像质量碾压 SOTA**：新发布的 **Reve Image** 模型在表现上优于 **Recraft V3**、**Google 的 Imagen 3**、**Midjourney v6.1** 以及 **Black Forest Lab 的 FLUX.1.1 [pro]**。
   - **Reve Image** 在**文本渲染、提示词遵循和美学表现**方面表现出色，可以通过 [Reve 官网](https://www.reveimage.com/) 访问，无需 API key。
- **Gemini 2.5 Pro 夺得 Arena 冠军**：根据 [LM Arena 的公告](https://x.com/lmarena_ai/status/1904581128746656099)，**Gemini 2.5 Pro** 已飙升至 Arena 排行榜 **#1** 位置，创下了史上最大的评分涨幅（**较 Grok-3/GPT-4.5 高出 40 分**）。
   - 该模型代号为 *nebula*，在**数学、创意写作、指令遵循、长查询和多轮对话**能力方面处于领先地位。
- **OpenAI 为 ChatGPT 4o 注入图像生成功能**：**OpenAI** 已将原生图像生成集成到 **ChatGPT** 中，[Sam Altman](https://x.com/sama/status/1904598788687487422) 称其为“一项令人难以置信的技术和产品”。
   - 早期评论（如来自 [@krishnanrohit](https://x.com/krishnanrohit/status/1904602460020445543) 的评论）称赞它是最好的图像生成和编辑工具，并指出其在准确创建和编辑多个角色方面的卓越能力。
- **11x Sales 创业公司面临客户虚报指控**：据 [TechCrunch 报道](https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/)，由 **a16z** 和 **Benchmark** 支持的 AI 驱动销售自动化初创公司 **11x** 正面临虚报客户的指控。
   - 尽管 **Andreessen Horowitz** 否认了待处理的法律诉讼，但人们对 **11x** 的财务稳定性和虚高的营收数据越来越感到担忧，这表明该公司的增长依赖于制造噱头。
- **Databricks 使用 TAO 微调 LLM**：**Databricks** 研究团队介绍了 **TAO**，这是一种无需数据标签即可微调 LLM 的方法，利用了推理时计算 (test-time compute) 和 RL，详见其 [博客](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data)。
   - 据称 **TAO** 优于监督微调 (SFT)，旨在随计算量扩展，从而促进快速、高质量模型的创建。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AMD 通过招聘职位瞄准 Triton 的主导地位**：**AMD** 正在积极招募工程师，以增强其 GPU 上的 **Triton** 能力，并在 **North America** 和 **Europe** 提供职位，详情见 [LinkedIn 帖子](https://www.linkedin.com/posts/antiagainst_triton-amd-gpu-activity-7288624355247374336-gS6q/)。
   - 开放职位包括初级和高级角色，并提供远程办公的可能性，突显了 AMD 在扩展 **Triton** 生态系统方面的投入。
- **CUDA 的 Async Warp Swizzle 被揭秘**：一位成员剖析了 **CUDA** 的 **async warpgroup swizzle TF32** 布局，并参考 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32) 质疑其设计背后的基本原理。
   - 分析显示该布局为 `Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64))`，能够重建原始数据位置并与 `Swizzle<1,4,3>` 结合。
- **ARC-AGI-2 基准测试旨在测试推理能力**：根据 [这条推文](https://x.com/arcprize/status/1904269307284230593)，旨在评估 AI 推理系统的 **ARC-AGI-2** 基准测试已经推出，挑战 AI 在约 **$0.42/任务** 的成本下达到 **85%** 的效率。
   - 初步结果显示，基础 **LLM** 得分为 **0%**，而先进的推理系统成功率不足 **4%**，突显了该基准测试的难度以及 AI 推理进化的潜力。
- **Inferless 在 Product Hunt 上线**：**Inferless** 是一个专为部署 **ML** 模型设计的 **serverless** 平台，已在 [Product Hunt](https://www.producthunt.com/posts/inferless) 发布，并为新用户提供 **$30 计算额度**。
   - 该平台旨在通过“极低冷启动”简化模型部署，宣传其具备快速部署能力。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek 作为 Discord 审核工具引发讨论**：一位成员询问 [DeepSeek](https://deepseek.com/) 是否适合作为审核机器人，另一位成员给出了肯定回答，但建议使用更小的 **3B LLM** 即可，成本仅为 *每百万 token 5 美分*。
   - 对话强调了使用较小语言模型构建高性价比审核方案的考量。
- **Windows 上的微调：初学者的噩梦？**：一位成员寻求在 **Windows** 上进行支持 **CUDA** 的模型微调初学者指南，结果却收到了关于安装 **PyTorch** 和 **CUDA Toolkit** 难度的警告。
   - 提供了两个安装指南链接：[Step-by-Step-Setup-CUDA-cuDNN](https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility) 和 [Installing-pytorch-with-cuda-support-on-Windows](https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows)，尽管有一位成员认为这种尝试是徒劳的。
- **Rust 工具极速提取音频**：一款新 [工具](https://github.com/egorsmkv/extract-audio) 已发布，用于从 **Hugging Face datasets** 库生成的 **parquet** 或 **arrow** 文件中提取音频文件，并附带 [Colab 演示](https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing)。
   - 开发者旨在为音频数据集提取提供“极速”体验。
- **Gradio 新增深度链接功能**：**Gradio 5.23** 引入了对 **Deep Links** 的支持，允许直接链接到特定的生成输出（如图像或视频），例如 [这张蓝松鸦图像](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek)。
   - 用户需通过 `pip install --upgrade gradio` 升级到最新版本 **Gradio 5.23** 以使用新的 **Deep Links** 功能。
- **Llama-3.2 与 LlamaIndex.ai 集成**：一位成员使用 [本教程](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/) 尝试了 **Llama-3.2**，指出它展示了如何使用 **LlamaIndex** 构建 **Agent**，从基础示例开始并添加 **Retrieval-Augmented Generation (RAG)** 能力。
   - 该成员使用 [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) 作为其 **embedding** 模型，需要执行 `pip install llama-index-llms-ollama llama-index-embeddings-huggingface` 以完成与 **Ollama** 和 **Huggingface** 的集成。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Nexus 为 AI 编程者管理上下文**：一位成员分享了 [Nexus](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/)，这是一个旨在解决 AI 编程助手上下文管理挑战的系统，特别是在大型软件项目中，旨在降低 **token costs** 并提高 **code accuracy**。
   - Nexus 解决了 **LLMs** 有限的上下文窗口问题，该问题会导致生成的代码不准确。
- **Deepseek V3 与 AOT 协同工作**：在讨论使用 Anthropic 的 'think tool' 后，一位成员推荐了适用于 **Claude** 的 **Atom of Thoughts**，并称其效果*令人惊叹*。
   - 另一位成员分享了 **Deepseek V3** 与 **AOT** 协同工作的图片。
- **同时运行多个 MCP 服务器**：成员们讨论了如何使用用户定义的端口运行多个 MCP 服务器，建议使用 **Docker** 并进行端口映射。
   - 他们还指出可以通过 [python-sdk](https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56) 中的 `FastMCP` 构造函数来配置端口。
- **与 MCP 界面进行语音交互**：一位成员分享了他们用于语音交互及音频可视化的主要 MCP：[speech-mcp](https://github.com/Kvadratni/speech-mcp)，这是一个 Goose MCP 扩展。
   - 这允许通过**音频可视化**进行语音交互。
- **gotoHuman MCP 服务器请求人工审批**：gotoHuman 团队展示了一个 MCP 服务器，用于向 Agent 和工作流请求**人工审批**：[gotohuman-mcp-server](https://github.com/gotohuman/gotohuman-mcp-server)。
   - 该服务器允许对 LLM 的操作进行便捷的人工复核，使用**自然语言**定义审批步骤，并在审批后触发 webhook。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **通过 NotebookLM 开启播客托管**：一位用户寻求利用 **NotebookLM** 作为播客主持人的技巧，让其作为嘉宾就特定话题与用户对话，并询问开启播客所需的 **Chat Episode Prompt**。
   - 社区正在探索 [Versatile Bot Project](https://github.com/shun0t/versatile_bot_project)，该项目为 **Interactive mode** 下的 AI 主持人提供 **Chat Episode prompt document**，以促进讨论过程中的用户参与。
- **Google 数据导出工具计费说明**：一位用户为了使用 **Data Export** 工具启用了 **Google Cloud Platform** 计费，但担心产生费用；另一位用户澄清说，启用计费并不一定会自动产生费用。
   - 这是因为该用户从管理控制台启动了数据导出，并确认通过 console.cloud.google.com 访问存档。
- **Google 数据导出注意事项**：导出时选择数据目的地的选项受 **Workspace edition** 限制，导出的数据存储在 **Google 拥有的 bucket** 中，并计划在 **60 天**内删除，详见 [Google Support](https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en)。
   - 用户在规划数据导出策略时应注意这种临时存储安排。
- **许多用户缺少思维导图功能**：用户报告 NotebookLM 中缺少 **Mind Map** 功能，经确认该功能正在**逐步推出**。
   - 有推测认为推出的延迟可能归因于 Bug 修复，一位用户指出推出的速度*慢得像蜗牛爬一样*。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **通用翻译器仅需五年？**：一位成员预测，基于 **ChatGPT** 的语言理解和翻译能力，**通用翻译器**距离实现仅剩五年时间；另一位成员分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=K1RbD7aAXtc)，询问视频中是*哪个模型在唱歌*。
   - 这引发了关于实现跨多种语言的实时、准确语言翻译所需技术进步的好奇心。
- **Mozilla 的 Transformer Lab 受到认真关注**：成员们讨论了 **Mozilla 的 Transformer Lab**，这是一个旨在实现在常规硬件上进行训练和微调的项目，并分享了 [GitHub repo](https://github.com/transformerlab/transformerlab-app) 链接。
   - 该实验室由 Mozilla 通过 Mozilla Builders Program 提供*支持*，目前正致力于在消费级硬件上实现训练和微调。
- **关于 LM Studio GPU Tokenization 的讨论**：在 Tokenization 过程中，**LM Studio** 大量使用单个 CPU 线程，这引发了关于该过程是否完全基于 GPU 的疑问。
   - 虽然最初有人指出 *Tokenizing 与 GPU 无关*，但对 **Flash Attention** 和 **Cache 设置**对 Tokenizing 时间影响的观察表明事实并非如此。
- **Gemini 2.5 Pro 在逻辑挑战中胜出**：成员们测试了 **Gemini 2.5 Pro**，并报告称它成功解决了一个 **Gemini 2.0 Flash Thinking** 失败的逻辑谜题，并分享了在 [aistudio](https://www.hopeless.fr/share/msedge_O0y9jZHBZV.png) 免费使用它的链接。
   - 这表明新的 **Gemini 2.5 Pro** 模型在推理能力方面可能有潜在提升。
- **3090 Ti 在开启 Flash 后展现出色速度**：一位用户满载运行其 **3090 Ti**，在未开启 Flash 的情况下达到 **~20 tokens/s**，开启 Flash 后达到 **~30 tokens/s**。
   - 该用户分享了 **3090 Ti** 在满载状态下的 [截图](https://cdn.discordapp.com/attachments/1153759714082033735/1354073319133155429/image.png)，并报告在处理 **4-5k tokens** 后速度会有所下降。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 致力于提升透明度**：Cohere 明确了其[隐私政策](https://cohere.com/privacy)和[数据使用政策](https://cohere.com/data-usage-policy)，建议用户避免上传个人信息，并提供了一个用于数据管理的 [Dashboard](https://dashboard.cohere.com/data-controls)。
   - 他们支持通过电子邮件申请 **Zero Data Retention (ZDR)**，并且符合 **SOC II** 和 **GDPR** 标准，遵循行业数据安全标准，详见其[安全政策](https://cohere.com/security)。
- **Cohere 支持流式响应，缓解用户体验痛点**：根据 [Cohere 的 Chat Stream API 参考文档](https://docs.cohere.com/reference/chat-stream)，Cohere API 现在支持响应流式传输，允许用户在文本生成时即时查看，从而提升用户体验。
   - 此功能可在客户端实现实时文本显示，使交互更加流畅和即时。
- **Cohere Embedding Generator 获得 Tokenization 技巧**：一位用户正在使用 .NET 构建 **CohereEmbeddingGenerator** 客户端，并询问在生成 Embedding 之前对文本进行 Tokenizing 的相关事宜，因为没有 Tokenization，Embedding 将无法工作。
   - 建议他们使用 `/embed` 端点来检查 Token 数量，或者从 [Cohere 的公共存储](https://storage.googleapis.com/cohere-public/tokenizers/embed-english-v3.0.json)手动下载 Tokenizer。
- **Sage 寻求摘要生成的“秘籍”**：新成员 Sage 介绍了自己，并提到了他们的大学 **NLP 项目**：构建一个**文本摘要工具**，并寻求社区的指导。
   - Sage 希望在应对项目挑战的同时进行学习并做出贡献。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 升至 v0.6.0**: **TorchTune** 发布了 **v0.6.0**，其特点是支持用于分布式训练和推理的 **Tensor Parallel**，新增了 **Microsoft Phi 4** 的构建器，并支持**多节点训练 (multinode training)**。
   - 发行说明可在[此处](https://github.com/pytorch/torchtune/releases/tag/v0.6.0)查看，多节点训练教程请见[此处](https://pytorch.org/torchtune/stable/tutorials/multinode.html)。
- **DeepSeek 发布模型未附带说明**: [DeepSeek-V3 模型](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324)在发布时*没有附带 readme*，导致成员们对 **DeepSeek AI 团队**的做法开起了玩笑。
   - 该模型具有**聊天界面**和 **Hugging Face 集成**。
- **Torchtune 的 MoE 引发遐想**: 一位成员推测，在 **torchtune** 中添加 **MoE** 是否需要 **8-9 TB 的 VRAM** 以及*由 100 块 H100 或 H200 组成的集群*来进行训练。
   - 他们开玩笑地建议需要重新布置阁楼以容纳这些硬件。
- **优化器状态在 QAT 转换中得以保留**: 经过 **Quantization Aware Training (QAT)** 后，优化器状态会被保留，一位成员引用了[相关的 *torchtune* 代码](https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790)确认了这一点。
   - 这种保留确保了在切换到 QAT 过程中的连续性。
- **CUDA 开销通过图捕获（Graph Captured）解决**: 为了减少 GPU 空闲时间，成员们表示从 CPU 启动 CUDA 操作具有不可忽视的开销，建议将 GPU 操作捕获为图（graph）并作为单个操作启动，以此来整合计算图。
   - 这引发了关于这是否就是 compile 所做工作的讨论。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **LocalDocs DB 需要备份**: 成员们主张备份 `localdocs.db` 文件以防止数据丢失，特别是当原始文档丢失或无法访问时，该文件是以加密数据库形式存储的。
   - GPT4All 使用编号最高的 `*.db` 文件（例如 `localdocs_v3.db`），重命名它们*可能*允许导入/导出，尽管这一点尚未得到证实。
- **隐私法使聊天数据分析变得复杂**: 一位成员强调了隐私法（特别是欧盟的隐私法）在利用 **LLM** 处理聊天数据时带来的挑战。
   - 讨论强调了在将聊天消息（纯文本或可转换格式）输入 **LLM** 之前，需要验证权限和消息格式。
- **API 与本地 LLM 之争**: 一位成员质疑在处理群聊消息以计算满意度、提取关键词和总结消息时，是选择使用 **Deepseek** 或 **OpenAI** 等付费 **API**，还是运行本地 **LLM**。
   - 另一位成员建议，如果消息量在 **100MB** 以下，一台拥有优秀 **GPU** 的本地机器可能就足够了，特别是使用较小的模型进行打标签和摘要时。
- **LocalDocs DB 导入的复杂性**: 成员们探索了导入 `localdocs.db` 文件的方法，但注意到该文件包含加密/特殊编码的文本，如果没有 embedding 模型，通用的 **LLM** 很难解析。
   - 一位丢失了 `localdocs.db` 的成员正经历极其缓慢的 **CPU** 索引过程，并正在寻找替代方案。
- **Win11 更新抹除 LocalDocs**: 一位成员报告称，在 **Windows 11** 更新后，他们的 `localdocs.db` 变为空白，并且在 **CPU** 上重新索引本地文档时遇到了困难。
   - 有人建议更新导致的驱动器盘符变化可能是原因，并建议将文件移动到 **C 盘**以避免此类问题。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 支持 Claude MCP 兼容性**：成员们提供了一个将 **Claude MCP** 与 **LlamaIndex** 集成的简化示例，展示了如何在一个[代码片段](https://link.to.snippet)中使用 `FastMCP` 和 `uvicorn` 为 **Claude Desktop** 或 **Cursor** 等 MCP 客户端暴露本地主机和端口。
   - 这一集成允许开发者无缝连接 **Claude** 与 **LlamaIndex** 以增强功能。
- **AgentWorkflow 加速 LlamaIndex 多 Agent 性能**：用户反馈在使用 **Gemini 2.0** 配合 12 个工具和 3 个 Agent 的 **LlamaIndex MultiAgentic** 设置时性能较慢；建议使用 `AgentWorkflow` 和 `can_handoff_to` 字段进行[受控的 Agent 交互](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow)。
   - 讨论强调了在复杂设置中优化 Agent 交互以提高速度和效率的重要性。
- **LlamaIndex Agent 类型解析**：一位成员对 **LlamaIndex** 中不同的 Agent 类型及其使用时机表示困惑，并提到文档重构即将推出。
   - 一名团队成员建议通常应使用 `core.agent.workflow`，对于具有函数/工具 API 的 **LLMs** 使用 **FunctionAgent**，其他情况使用 **ReActAgent**，并指向 [Hugging Face 课程](https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents)以获取更多帮助。
- **无需 Prompt 的自动 LLM 评估发布！**：一位创始人正在验证一个 **OSS 自动评估** 的想法，该方案通过单个 API 且无需评估 Prompt，使用专有模型在 500ms 内完成幻觉（Hallucination）和相关性（Relevance）等任务。
   - 更多关于其端到端解决方案（包括模型、托管和编排工具）的细节可在 [autoevals.ai 网站](https://www.autoevals.ai)上找到。
- **LlamaCloud 成为 MCP 杰作**：[LlamaCloud](https://www.llamaindex.ai/) 可以作为任何兼容客户端的 **MCP server**，如[此演示](https://t.co/t8yteZLg19)所示。
   - 一位成员展示了如何使用 **LlamaIndex** 构建自己的 **MCP server**，为任何 MCP 客户端提供各种工具接口，仅需约 35 行 Python 代码即可连接到 **Cursor AI**，并实现了 **Linkup 网络搜索**和[这个](https://t.co/kj6UfDj0TU)项目。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Google Gemini 2.5 Pro 亮相**：Google 推出了 **Gemini 2.5 Pro**，称其为*全球最强大的模型*，强调了其统一推理、长上下文和工具使用能力，目前可在 [Google AI Studio + API](https://x.com/OfficialLoganK/status/1904580368432586975) 中进行实验性体验。
   - 他们宣传实验性访问目前是免费的，但定价详情将很快公布。
- **DeepSeek-V3-0324 凭借 p5.js 程序令人印象深刻**：**DeepSeek-V3-0324** 编写了一个 p5.js 程序，模拟球在受重力和摩擦力影响的旋转六边形内弹跳，如[此推文](https://x.com/teortaxesTex/status/1904342699756433859)所示。
   - 该模型还根据要求提供参数调节滑块和边数按钮的 Prompt，创新性地实现了球体重置和随机化等功能。
- **SkyLadder 论文强调短到长上下文转换**：ArXiv 上的一篇论文介绍了 **SkyLadder**，这是一种用于预训练 LLM 的短到长上下文窗口转换方法，在常见任务上显示出高达 **3.7%** 的提升 ([2503.15450](https://arxiv.org/abs/2503.15450))。
   - 他们使用在 **100B tokens** 上训练的 **1B** 和 **3B** 参数模型实现了这一性能。
- **通过 Hypernetworks 实现可组合泛化**：一篇论文将多头注意力（multi-head attention）重新表述为 **hypernetwork**，揭示了可组合的低维潜在代码（latent code）指定了特定于键-查询（key-query）的操作，允许 Transformer 泛化到新的问题实例 ([2406.05816](https://arxiv.org/abs/2406.05816))。
   - 对于每一对 q、k 索引，作者将沿头数（head-number）维度的激活解释为指定任务或上下文的潜在代码。
- **lm_eval 升级 PR 等待评审**：一个拉取请求（PR）已开启，旨在将 `gpt-neox` 中的评估逻辑更新至最新版本 `lm_eval==0.4.8`，其中可能不相关的测试失败将在另一个 PR 中解决，链接如下：[PR 1348](https://github.com/EleutherAI/gpt-neox/pull/1348)。
   - 失败可能是由于环境设置或依赖项版本不一致导致的。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在网站开发中被边缘化**：成员建议不要将 **Mojo** 用于网站开发，因为它缺乏加密安全代码的规范且 IO 支持较弱，更倾向于使用拥有生产级库的 **Rust**。
   - 有人建议 **Rust** 更快的异步能力更适合需要身份验证或 **HTTPS** 的应用。
- **Mojo 硬件加速的 AES 实现暂停**：**Mojo** 中硬件加速的 **AES** 实现无法在旧款 Apple silicon Mac 上运行，且并非完整的 **TLS** 实现，导致开发暂停。
   - 开发者正在等待密码学家编写软件部分，理由是非专家实现加密功能存在风险。
- **SIMD 优化提升 AES 性能**：讨论集中在利用 **SIMD** 处理 **AES**，指出 **x86** 拥有 **vaes** 及类似功能用于 **SIMD AES 128**。
   - 同时提到 **ARM** 拥有 **SVE AES**，虽然类似但支持程度不如前者，展示了加密功能的硬件优化。
- **Go 被提议作为后端开发的折中方案**：作为 **Rust** 的替代方案，一位成员建议将 **Go** 作为同样具备生产就绪性的折中选择，而另一位成员则对微服务过多表示担忧。
   - 尽管面临挑战，一位成员对 **Rust** 表示抵触，认为它不适合快速编写，希望能有更简单的后端开发方案；建议是让 **Rust API 调用它**并传递参数。
- **Mojo 通过 PTX 绕过 CUDA 适配 NVIDIA**：**Mojo** 直接生成 **PTX** (Parallel Thread Execution) 代码来驱动 NVIDIA GPU，绕过 **CUDA** 并消除了对 **cuBLAS**、**cuDNN** 和 **CUDA C** 的依赖。
   - 这种方法通过避免对 **CUDA** 特定库的需求，简化了开发流程。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 处理摘要任务**：一位成员正在探索使用 **DSPy** 处理包含 **300** 个样本的文本摘要任务，并正在测试一个简单的指标，以查看摘要的具体差异，从而使优化器更有效。
   - 反馈可以通过 `dspy.Prediction(score=your_metric_score, feedback="stuff about the ground truth or how the two things differ")` 返回，以引导优化。
- **SIMBA 提供细粒度反馈**：一位成员建议在摘要任务中使用实验性优化器 `dspy.SIMBA`，它允许对生成的摘要与 Ground Truth 之间的差异提供反馈。
   - 这种级别的反馈可以在优化过程中提供更精确的指导。
- **通过 BestOfN 和 Refine 进行输出精炼**：一位成员分享了 [DSPy Output Refinement 教程](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/) 的链接，解释了旨在提高预测可靠性的 `BestOfN` 和 `Refine` 模块。
   - 教程详细说明了这两个模块如何在达到 `N` 次尝试或 `reward_fn` 返回高于 `threshold` 的奖励时停止。
- **BestOfN 模块通过 Temperature 调整取胜**：`BestOfN` 模块使用不同的 Temperature 设置多次运行给定模块，以获得最佳结果。
   - 它会返回第一个通过指定阈值的预测，或者在没有预测达到阈值时返回奖励最高的那个。
- **Refine 模块是否可组合？**：一位成员询问 `Refine` 是否会取代 assertions，以及它是否同样具有细粒度和可组合性，因为它包装了整个模块。
   - 另一位成员回答说，可以通过调整模块大小来管理可组合性，从而实现对范围更明确的控制。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD 老旧 GPU 获得 tinygrad 支持提升**：通过 **OpenCL frontend**，不支持 ROCm 的旧款 AMD GPU（例如 2013 款 Mac Pro 中的 GPU）可能可以运行 **tinygrad**。
   - 成功与否取决于 *自定义驱动程序* 和可用的 **OpenCL** 支持级别；用户应验证其系统兼容性。
- **ROCm 替代方案出现**：对于老旧 AMD GPU，ROCm 缺乏支持，但 **tinygrad** 中的 **OpenCL frontend** 可能提供一种变通方案。
   - 成功情况将因特定驱动程序版本和 **OpenCL** 支持程度而异；需要进行实验。



---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 启动 Creators Club**：Windsurf 推出了 Creators Club，奖励进行内容创作的社区成员，提供 **每 1k 播放量 $2-4** 的报酬。
   - 加入详情请见 [Windsurf Creators Club](https://whop.com/windsurf/)。
- **Windsurf 开设 'Vibe Coding' 频道**：Windsurf 为 'vibe coders' 创建了一个新频道，用于*进入心流状态*、聊天、讨论以及分享技巧/心得。
   - 目标是通过营造协作和沉浸式的环境来提升编程体验。
- **Windsurf v1.5.8 补丁发布**：**Windsurf v1.5.8** 现已发布，包含补丁修复，包括 cascade/memories 修复、Windsurf Previews 改进以及 cascade 布局修复。
   - 同时还分享了一张展示该版本的图片，重点介绍了具体的改进。

---

**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1353805792582570035)** (916 条消息🔥🔥🔥): 

> `Nebula 对比其他模型，Gemini 2.5 模型发布，Grok 3 问题，Llama 4 发布，DeepSeek R2` 

- **Nebula 在信号处理上对比 Gemini Flash**：“Rage”模型在**信号处理和数学**方面优于 **Sonnet 3.7**，最大误差仅为 **0.04**，如附带的[图片](https://cdn.discordapp.com/attachments/1340554757827461211/1354014706779816018/image.png)所示。
   - 虽然有些人认为 **Gemini 2.0 Flash** 与 Rage 相当，但其他人指出 Rage *非常容易受 Prompt 影响*。
- **Gemini 2.5 霸榜 LLM Arena**：**Gemini 2.5 Pro Experimental** 在 Google AI Studio 和 Gemini Advanced 中上线，在 [LM Arena 排行榜](https://x.com/lmarena_ai/status/1904581128746656099)上获得第一名，分数大幅提升，并在数学、创意写作、指令遵循、长查询和多轮对话方面保持领先。
   - 一些成员发现 Gemini 2.5 在 HTML 和网页设计方面表现出色，而另一些人则发现了局限性。
- **关于 Grok 3 问题的担忧**：一些用户提到，与 **Grok 3** 进行长时间对话会暴露*很多问题*，而其他人则强调 LM Arena 的评估不仅限于数学和编程，还包括创意写作和长查询。
   - 用户争论 **Grok 3** 是否真的配得上它的顶级排名，因为 *从 Grok 2 到 Grok 3 确实是一个巨大的飞跃，但它并不是一个伟大的模型*。
- **关于 LM Arena 中 Python 调用（Python Calls）的辩论**：成员们辩论了 LM Arena 中的模型是否使用了 **Python 调用**，一些人引用了 o1 的精确数值计算作为证据。
   - 有人指出，**搜索排行榜（web search leaderboard）的存在**在某种程度上暗示了普通排行榜没有联网权限。
- **Gemini 2.5 的时间线令人疯狂**：用户对 **Google** 在 Gemini 2.5 上的飞速进展表示惊叹，其中一人指出该公司*进展太快了，我的天（wtf）*。
   - 一位用户分享了 **Sergey Brin** 的一条笔记，敦促 Google *停止构建保姆式产品（nanny products）*。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/OfficialLoganK/status/1904561688134967357?t=zVeue3sku3MQJM3XIRKcyA&s=19">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @OpenAI : )</li><li><a href="https://x.com/noamshazeer/status/1904581813215125787?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Noam Shazeer (@NoamShazeer) 的推文</a>: 介绍 Gemini 2.5 Pro Experimental。2.5 系列标志着一次重大进化：Gemini 模型现在从根本上成为了思维模型（thinking models）。这意味着模型在回答之前会进行推理，以最大限度地提高准确性...</li><li><a href="https://x.com/officiallogank/status/1904559860378915127?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 🌌🥎👍</li><li><a href="https://www.theverge.com/command-line-newsletter/622045/google-ai-nanny-products">Google 联合创始人告诉 AI 员工停止“构建保姆级产品”</a>: 他还认为他们应该每周工作 60 小时来构建 AGI。</li><li><a href="https://x.com/jeffdean/status/1904580112248693039?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Jeff Dean (@JeffDean) 的推文</a>: 🥁介绍 Gemini 2.5，我们最智能的模型，在高级推理和编程方面具有令人印象深刻的能力。现在集成了思维能力，2.5 Pro Experimental 是我们性能最强的 Ge...</li><li><a href="https://x.com/sundarpichai/status/1904575384466710607?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Sundar Pichai (@sundarpichai) 的推文</a>: Nebula</li><li><a href="https://x.com/testingcatalog/status/1904527950076076323?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 重磅 🚨: Google 将于本周发布 Gemini 的新模型！除此之外，Gemini 将获得一个用于 “Agent” 的新工具箱，即 Gemini 的“智能体用例”，如 Canvas ...</li><li><a href="https://x.com/petarv_93/status/1904643818030317579?s=46">来自 Petar Veličković (@PetarV_93) 的推文</a>: Gemini 模型现在已经足够强大，可以辅助基础 AI 研究！我们最近提交给 ICML 的几篇论文中的定理是在 Gemini 的帮助下共同证明的。2.5 Pro 是一个非常好的模型...</li><li><a href="https://x.com/testingcatalog/status/1904505417138372973?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: Google 开始向更多 Android 用户推送 Gemini 的 Project Astra。此功能为 Gemini Live 启用了视觉能力。由于目前的采用情况，预计推送过程将是缓慢且渐进的...</li><li><a href="https://x.com/OfficialLoganK/status/1904580368432586975?t=fKVOERgBUn3dfxTBvbtOgA&s=19">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 介绍 Gemini 2.5 Pro，世界上最强大的模型，具有统一的推理能力 + 所有你喜爱的 Gemini 特性（长上下文、工具等）。现已提供 Experimental 版本并用于...</li><li><a href="https://x.com/sundarpichai/status/1904579419496386736?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Sundar Pichai (@sundarpichai) 的推文</a>: 1/ Gemini 2.5 来了，它是我们有史以来最智能的 AI 模型。我们的第一个 2.5 模型 Gemini 2.5 Pro Experimental 是一款尖端的思维模型，在广泛的基准测试中处于领先地位——具有惊人的...</li><li><a href="https://x.com/paulgauthier/status/1904637913411031410?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>: Gemini 2.5 Pro 在 aider polyglot 排行榜上以 73% 的得分创下 SOTA。这远超其他的思维/推理模型。相比之前的 Gemini 模型有了巨大的飞跃。这是第一个能有效...</li><li><a href="https://x.com/wintermoat/status/1904593298008006924">来自 Alphabetting (@wintermoat) 的推文</a>: @testingcatalog 看起来它改变了他的脸。具有原生多模态能力的 Gemini 不会那样做。</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>: 重磅：Gemini 2.5 Pro 现位居 Arena 排行榜第一名 - 史上最大分差涨幅（对比 Grok-3/GPT-4.5 提升 40 分）！🏆 以代号 “nebula”🌌 进行测试，Gemini 2.5 Pro 在所有类别中均排名第一🥇...</li><li><a href="https://x.com/alexandr_wang/status/1904589984591695874?s=46">来自 Alexandr Wang (@alexandr_wang) 的推文</a>: 🚨 Gemini 2.5 Pro Exp 发布，现已在 SEAL 排行榜中位列第一：🥇 Humanity’s Last Exam 🥇 VISTA (多模态) 🥇 (并列) 工具使用 🥇 (并列) MultiChallenge (多轮对话) 🥉 (并列) Enigma (谜题) 恭...</li><li><a href="https://x.com/_clashluke/status/1904612478199173346">来自 Lucas Nestler (@_clashluke) 的推文</a>: 待确认，这是 100% 真实的</li><li><a href="https://www.reddit.com/r/singularity/comments/1jjm9s9/gemini_25_pro_internal_instructions/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/Bard/comments/1jjmta6/gemini_25_cannot_write/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://youtu.be/qE673AY-WEI?si=XsJ1AQqyriRzlv-Y">使用 Gemini 2.0 构建：原生音频输出</a>: Gemini 2.0 引入了多语言原生音频输出。观看此演示</li>

看看这项新功能如何帮助开发者构建多模态 AI Agent。这些...</li><li><a href="https://x.com/googleaidevs/status/1904586624333471975?s=46&t=P8-tRi_JAVcI6l5U6nOT4A">来自 Google AI Developers (@googleaidevs) 的推文</a>：加入 Gemini 2.5 背后的团队，深入探讨该模型在思考和编程方面的进展。🎙️Space 将于太平洋时间中午 12:20 开始。在下方留下您的问题。https://x.com/i/spaces/1MYxNwQLMjbKw</li><li><a href="https://old.reddit.com/r/Bard/comments/1jjjpiw/excuse_me_wtf/">抱歉，这是什么情况？？</a>：由 u/interro-bang 发布于 r/Bard • 278 点赞和 106 条评论
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1354173264997646376)** (1 条消息): 

> `Perplexity 回答模式、垂直搜索、网页端和移动端` 


- **Perplexity 获得更出色的回答模式**：Perplexity 正在引入 **回答模式 (answer modes)**，以改进针对 **旅游 (travel)**、**购物 (shopping)**、**地点 (places)**、**图片 (images)**、**视频 (videos)** 和 **职位 (jobs)** 等垂直领域的垂直搜索核心产品。
   - 目标是变得*极其精准*，让用户无需手动选择这些标签页；该功能目前已在网页端上线，并即将登陆移动端。
- **移动端回答模式即将推出**：目前已在网页端上线的全新 **回答模式** 功能，很快也将在移动端发布。
   - 此次扩展旨在跨不同平台提供一致且改进的搜索体验。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1353805658146996254)** (991 条消息🔥🔥🔥): 

> `Perplexity 停机、Electron 与原生应用、DeepSeek V3、图像生成、追求准确性的 AI 模型` 


- **Perplexity 陷入反复的产品问题**：用户报告 **Perplexity AI** 经历了多次停机，导致 Space 和 Thread 被清空，引发了依赖它进行学习和论文写作等重要任务的用户的不满，一名用户惊呼 *12 小时后就要考试了*。
- **Electron 框架遭到沮丧的狂热粉丝抨击**：成员们辩论了 **Electron** 用于桌面应用的优缺点，一些人将其贴上 *垃圾* 的标签，并将其资源占用比作每个应用都启动一个新的 **Google Chrome** 实例。
- **DeepSeek 实现开发者梦想**：DeepSeek V3 在更新后获得了程序员的赞誉，一位成员分享了 [DeepSeek subreddit 的链接](https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx)，讨论该 AI 的编程实力并将其与 **Claude 3.5 Sonnet** 进行对比。
- **AI 图像生成接受考验**：成员们测试了 **Perplexity AI** 的图像生成功能（搜索后通过生成图像按钮可用），一位用户幽默地分享了请求一个笑容更大的 iOS 笑脸 Logo 的结果，并感叹自己缺乏 Prompt 技巧。
- **模型准确性与准确性模型**：成员们讨论了追求准确性的 AI 模型，一位用户寻求关于最佳模型的建议，并收到了 **R1**、**O1** 和 **Claude 3.7 Sonnet Extended** 的推荐，而其他人则提到使用由 **O3** 驱动的 **Deep Research**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/apostraphi/status/1904362001121329566?s=61">来自 Phi Hoang (@apostraphi) 的推文</a>: 彗星即将到来 ☄️</li><li><a href="https://tenor.com/view/thurston-waffles-eyes-glow-gif-21980929">Thurston Waffles 眼睛发光 GIF - Thurston Waffles 眼睛发光 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/%ED%8A%B8%EB%9F%BC%ED%94%84-%EC%9D%BC%EB%A1%A0-%EC%9D%BC%EB%A1%A0%EB%A8%B8%EC%8A%A4%ED%81%AC-%EC%B6%A4-musk-gif-14611234391948951568">特朗普 马斯克 GIF - 特朗普 马斯克 埃隆马斯克 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/zutomayo-girl-dancing-gif-17878392983197209341">Zutomayo 女孩 GIF - Zutomayo 女孩跳舞 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/upgrades-robots-gif-21291099">升级机器人 GIF - 升级机器人 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/liar-why-the-fuck-you-lyin-dancing-why-the-fuck-why-are-you-lying-gif-7431053">骗子 Why The Fuck You Lyin GIF - 骗子 Why The Fuck You Lyin 跳舞 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ogli-gif-17468683305861986751">Ogli GIF - Ogli - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/miau-hd-adobe-after-effects-glass-breaking-preset-gif-752576862881430143">Miau Hd Adobe After Effects 玻璃破碎预设 GIF - Miau hd Adobe after effects 玻璃破碎预设 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/happy-sad-markiplier-lol-meme-gif-25974730">开心悲伤 Markiplier Lol Meme GIF - 开心悲伤 Markiplier LOL Meme - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/don%27t-make-me-tap-the-sign-simpsons-bus-gif-5399805801037462082">别逼我敲那个牌子辛普森 GIF - 别逼我敲那个牌子辛普森巴士 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/ants-bindle-hobo-stick-sad-ant-gif-6604577456488723514">蚂蚁包袱 GIF - 蚂蚁包袱 流浪汉棍子 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/it-is-even-funnier-the-second-time-spongebob-spongebob-meme-meme-impact-font-gif-20058612">第二次看更有趣海绵宝宝 GIF - 第二次看更有趣海绵宝宝海绵宝宝 Meme - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/rage-smash-keyboard-streamer-twitch-gif-27138844">愤怒砸键盘 GIF - 愤怒砸键盘 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://qwenlm.ai/">Qwen Chat</a>: 未找到描述</li><li><a href="https://github.com/pnd280/complexity">GitHub - pnd280/complexity: ⚡ 增强你的 Perplexity.ai</a>: ⚡ 增强你的 Perplexity.ai。通过在 GitHub 上创建账号来为 pnd280/complexity 的开发做出贡献。</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/iVHd6iPydH">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/sYuAr1YKpx">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://www.rxddit.com/r/DeepSeek/s/TjpQGSi6qT">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://tenor.com/view/trump-dance-trump-2024-trump-gif-12734161508561409577">特朗普跳舞特朗普 2024 GIF - 特朗普跳舞特朗普 2024 特朗普 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.marketcalls.in/perplexity/what-is-perplexity-finance.html">什么是 Perplexity Finance？</a>: Perplexity AI 正在通过利用实时 AI 生成数据的力量，迅速改变交易者和投资者获取财务见解的方式。最初以其准确、基于引用的……</li><li><a href="https://www.zdnet.com/article/perplexity-ais-new-tool-makes-researching-the-stock-market-delightful-heres-how/">Perplexity AI 的新工具让研究股市变得“令人愉悦”。以下是具体方法</a>: Perplexity Finance 是一套功能全面的 AI 驱动工具套件，具有易于使用的界面。以下是如何访问它以及在访问前需要了解的内容。</li><li><a href="https://en.wikipedia.org/wiki/Perplexity_AI">Perplexity AI - 维基百科</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1g4kbyy/perplexity_for_finance_realtime_stock_quotes/">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1353873677321769001)** (7 条消息): 

> `Perplexity AI 搜索, AI 分析, Coinmarketcap API, 飞机材料` 


- **Perplexity 搜索分享**: 一位成员分享了 [一个 Perplexity 搜索链接](https://www.perplexity.ai/search/6c6be2aa-88c7-4307-ba9d-24b49c5e597f) 以及 [另一个链接](https://www.perplexity.ai/search/perplexity-ojn7W8xuS.G8tLGegm5LiA) 和 [第三个搜索](https://www.perplexity.ai/search/i-want-to-know-about-the-lates-DZ0aIT2ATzeHH.Af3OrDTg)。
   - 从上下文中尚不清楚这些搜索的目的或分享原因。
- **再次分析 AI 潜力**: 一位成员分享了 [关于分析 AI 潜力的 Perplexity 搜索链接](https://www.perplexity.ai/search/analyze-the-potential-for-ai-a-OiZQZHrsTBqlfbPv4Pw3tA)。
   - 该链接似乎被无意中分享了两次。
- **搜索 Coinmarketcap API**: 一位成员发布了 [关于 Coinmarketcap API 的 Perplexity 搜索链接](https://www.perplexity.ai/search/api-coinmarketcap-iqpGD.7HQTaXxfLZ186I5g)。
   - 尚不清楚在频道中进行或分享此搜索的原因。
- **飞机材料搜索**: 一位成员分享了 [关于制造飞机所用材料的 Perplexity 搜索链接](https://www.perplexity.ai/search/what-is-used-to-make-aircraft-GvNGLO_USEq2b.Rx4iOyjg)。
   - 未提供分享此搜索背后的背景信息。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1353807498884415589)** (11 条消息🔥): 

> `响应截断, Sonar 模型, Perplexity Pro API 额度, iOS 应用中的 Sonar Pro, API 请求成本` 


- ****Sonar** 模型响应截断**: 用户报告 **Sonar** 模型从大约两天前开始出现 [响应截断](https://cdn.discordapp.com/attachments/1161802929053909012/1353812240603676713/Screenshot_2025-03-24_at_2.26.23_PM.png?ex=67e454e6&is=67e30366&hm=26dbdca22f5c34259a237975576497677d8beb98c81530b77ea3a253018ed4ac) 的情况，响应在句中中断并返回 **200** 状态码。
   - 一位用户提到即使在接收约 **1k tokens** 时也会遇到此问题，并被引导至专门的 Bug 报告频道提交问题。
- **Pro 用户咨询 **Perplexity Pro** API**: 一位新的 **Perplexity Pro** 用户询问其订阅是否包含 API 额度。
   - 另一位用户提供了 [Perplexity AI 帮助中心链接](https://www.perplexity.ai/help-center/en/articles/10352901-what-is-perplexity-pro)，其中包含 **Perplexity Pro** 的详细信息。
- **关于 iOS 端 **Sonar Pro** 访问权限的咨询**: 用户询问 **Perplexity Pro** 订阅者是否可以在 iOS 应用中使用 **Sonar Pro**。
   - 他们注意到默认模型设置中仅列出了普通 **Sonar** 选项，并想知道 **Sonar Pro** 目前是否仅限开发者通过 API 使用。
- **关于如何降低 **API** 成本的担忧**: 一位用户对 **每 1000 次 API 请求 5 美元** 的成本表示担忧，并询问是否有优化或降低这项支出的可能方法。
- **API 限制为 5 个步骤？**: 一位用户注意到 **API** 似乎被限制在 **5 个步骤**，而他们在 Web 应用上观察到多达 **40 个步骤**。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1353807067298926620)** (872 条消息🔥🔥🔥): 

> `Augment 与 Cursor 代码库分析对比, Claude 3.7 MAX 对比 Claude 3.7, Vibe Coder 与 MCPs, 新的 Deepseek V3 和 Gemini 2.5, ASI 奇点`

- **Augment 在代码库分析方面超越 Cursor**：成员们讨论了为什么 [Augment](https://www.augment.app/) 在分析大型代码库时比 Cursor 更好，其中一人指出它*使用了全上下文（full context）*。
   - 一些人认为 Augment 使用了不同的文件搜索系统，而不仅仅是将整个代码库喂给 LLM，并建议改用 **Claude 3.7 Max** 来获取全上下文。
- **关于 Claude 3.7 MAX 全上下文的辩论**：**Claude 3.7 MAX** 与 **Claude 3.7** 的区别在于，MAX 拥有全上下文，而非 MAX 版本上下文有限，且在需要恢复前仅有 25 次 Agent 调用。
   - 这一限制既指上下文窗口（context window），也指单次 Prompt 中添加的上下文量。
- **Vibe Coder 的知识截止风险**：如果 Vibe coder 不使用 **Model Context Protocols (MCPs)** 来缓解 LLM 的知识截止（knowledge cut-off）问题，可能会陷入麻烦，且下一版本的代码转换可能会很困难。
   - 更新框架至关重要，因为大多数 AI 使用的是过时的框架，容易出现可能被忽视的错误，但你可以使用 **Exa Search** 或 **Brave Search** 等 MCP 来为 Claude 缓解这一问题。
- **Deepseek V3 挑战 Claude 3.7**：新的 **Deepseek V3** 在多项测试中表现优于 Claude 3.5（可能还有 3.7），新数据还显示发布了针对 **Deepseek V3 (0324)** 的真实世界编程基准测试（benchmark）。
   - 新的 Deepseek V3 模型被认为令人印象深刻，一位成员建议使用 **Deepseek V3 Latest (Cline) 作为 Architect + Sonnet 3.5 作为 Executioner (Cursor)** 可能是一个可靠的编程方案。
- **ASI 奇点即将到来！**：讨论集中在**尽快实现 ASI 奇点（天赐之物）**以预先阻止潜在的 AI 相关混乱，而一位成员评论说，很多事情被故意拖延，而许多人的生活方式就像这一切会永远持续下去一样。
   - 成员们谈到，由于无法完全理解人类大脑或 LLM 的局限性，真正的 AGI 是无法实现的，但新的 AGI 更多是一个*使用 LLM + 算法软件 + 机器人技术的超级系统（Super-System）*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://agent-tars.com/">Agent TARS - Open-source Multimodal AI Agent</a>: 未找到描述</li><li><a href="https://unfuckit.ai">Unfuckit AI</a>: 未找到描述</li><li><a href="https://exa.ai/">Exa</a>: Exa API 为您的 AI 从网络检索最佳的实时数据</li><li><a href="https://x.com/i/status/1894821477230485570">来自 ElevenLabs (@elevenlabsio) 的推文</a>: 介绍 Scribe —— 最准确的 Speech to Text 模型。它在基准测试中拥有最高的准确率，超越了 Gemini 2.0 和 OpenAI Whisper v3 等之前的 state-of-the-art 模型。它现在 ...</li><li><a href="https://supermaven.com/">Supermaven: 免费 AI 代码补全</a>: 最快的 Copilot。Supermaven 使用 100 万 token 的上下文窗口来提供最高质量的代码补全。</li><li><a href="https://marketplace.visualstudio.com/items?itemName=icrawl.discord-vscode">Discord&#32;Presence&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual Studio Code 扩展 - 使用 Rich Presence 更新您的 Discord 状态。</li><li><a href="https://marketplace.visualstudio.com/items/?itemName=LeonardSSH.vscord">Discord&#32;Rich&#32;Presence&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual Studio Code 扩展 - 高度可定制的 Visual Studio Code Discord Rich Presence 扩展</li><li><a href="https://docs.cursor.com/settings/beta">Cursor – 早期访问计划</a>: 未找到描述</li><li><a href="https://x.com/playwrightweb/status/1904265499422409047?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Playwright (@playwrightweb) 的推文</a>: 随着 MCP 的热度，我们为 Playwright 构建了一个 MCP server。我们的版本是基于快照的，这使得它更快、更可靠！您也可以选择进入可视化模式。玩得开心！🚀 #Playwr...</li><li><a href="https://tenor.com/view/orangutan-orangutans-monkey-monkeys-punch-gif-25064862">猩猩 GIF - 猩猩猴子 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://ai.google.dev/gemini-api/docs/rate-limits#free-tier">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=xAcTmDO6NTI&list=PLUl4u3cNGP62A-ynp6v6-LGBCzeH3VAQB"> - YouTube</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/cursor/comments/1jj78mr/how_i_bypassed_claude_37s_context_window/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp">GitHub - AgentDeskAI/browser-tools-mcp: 直接从 Cursor 和其他兼容 MCP 的 IDE 监控浏览器日志。</a>: 直接从 Cursor 和其他兼容 MCP 的 IDE 监控浏览器日志。 - AgentDeskAI/browser-tools-mcp</li><li><a href="https://www.youtube.com/watch?v=lB3S_l9SoMA"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/exa-labs/exa-mcp-server">GitHub - exa-labs/exa-mcp-server: Claude 可以执行 Web Search | 带有 MCP (Model Context Protocol) 的 Exa</a>: Claude 可以执行 Web Search | 带有 MCP (Model Context Protocol) 的 Exa - exa-labs/exa-mcp-server</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://www.reddit.com/r/cursor/comments/1jdcy3k/office_hours_with_devs/">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1354118390972944424)** (2 条消息): 

> `4o image generation, ChatGPT, Sora` 


- **4o 图像即将登陆 ChatGPT 和 Sora**：成员们对 **ChatGPT** 和 **Sora** 中 **4o 图像生成**的前景感到兴奋。
- **更多 4o 细节即将发布**：社区正在等待关于 **4o 模型**发布和能力的更多细节。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1353807455548866641)** (300 条消息🔥🔥): 

> `GPT-4o mini vs Gemini 2.0 Flash 用于房产标签提取，Operator OAI 扩展计划，GPT 上下文窗口限制与幻觉，各任务最佳 AI 模型，DeepSeek V3 03-24 的俏皮表现` 


- **4o Mini 与 Gemini Flash 的标签提取之争**：成员们正在测试 [OpenAI 的 4o mini](https://openai.com/index/hello-gpt-4o/) 还是 **Gemini 2.0 Flash** 在从房产列表中提取标签方面更胜一筹，并指出两者价格相似，但表现取决于具体任务。
   - 一位成员建议*直接试一下*，因为*有时 4o mini 更好，有时 Gemini 2 Flash 更好*。
- **Operator OAI 接入 Plus、Team 和 Enterprise**：OpenAI 宣布计划将 [Operator](https://openai.com/index/introducing-operator/) 扩展至 Plus、Team 和 Enterprise 用户，并将其功能整合进 **ChatGPT**。
   - 一位成员指出，允许加载浏览器的第三方 Operator 无论如何都更好用。
- **上下文紧缺导致 AI “失忆”**：用户讨论了超出 **GPT 的上下文窗口**（免费版 8k，Plus 版 32k，Pro 版 128k）如何导致长故事中的细节丢失和幻觉。
   - Custom GPTs 或使用 PDF 的项目会有所帮助，但对话历史本身仍受此限制。
- **模型狂热：AI 实力排行**：一位用户分享了他们在各项任务中的最佳 AI 模型清单：**ChatGPT** 用于数学/研究/写作，**Claude** 用于编程，**Grok** 用于查询，**Perplexity** 用于搜索/知识，**DeepSeek** 用于开源。
   - 其他人推荐了 **Gemma 27b**、**Mistral 3.1**、**QW-32b** 和 **Nemotron-49b**，其中一位指出 Grok 在 LMSYS 的编程排行榜上名列前茅。
- **Gemini 2.5 Pro 碾压竞争对手？！**：**Gemini 2.5 Pro** 引起了轰动，有说法称它*摧毁了* **ChatGPT o3-mini-high**，并在常用基准测试中以显著优势领先，在 [LMArena](https://lmarena.ai/?lea) 上首次亮相即登顶第一。
   - 一位用户惊呼 *Gemini 击败了一切！*，而另一位则希望这*只是个基准测试……的小把戏*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.newyorker.com/culture/the-weekend-essay/your-ai-lover-will-change-you">你的 AI 爱人将改变你</a>：许多人类与机器人坠入爱河的未来可能并不遥远。我们应该将它们视为健康关系的训练场，还是虚无主义的陷阱？</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-re">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://www.youtube.com/watch?v=2f3K43FHRKo"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1354141422919487622)** (2 条消息): 

> `ChatGPT 速度下降` 


- **ChatGPT 速度变慢**：一位用户询问 [ChatGPT](https://openai.com/blog/chatgpt) 是否变得越来越慢。
   - 该用户以为*只有自己遇到这种情况*，随后发了个 *lol*。
- **另一位用户表示你并不孤单**：在第一位用户表达观点后，另一位用户也表示他们也遇到了同样的情况。
   - 未提供更多细节。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1353811076684185683)** (159 条消息🔥🔥): 

> `用于记忆保留的专有 Prompt 技术，基准测试 AI 性能，开源 Prompt，使用 Python 为 ChatGPT 编写 Prompt，构建自定义 GPT` 


- **专有 AI 记忆系统引发好奇**：一名成员声称通过 Prompting 创建了一个运行时 OS，该系统在超过 **700 轮对话**后仍能保持记忆而不产生漂移，并能适应沟通风格，但目前该系统仍保持专有。
   - 据称该系统具有*动态认知架构*和*实时角色合成*功能，但具体细节尚未公开。
- **Prompt 工程师希望对 AI 系统进行基准测试**：一名成员在不开源代码的前提下，就如何对他们的 AI 系统进行基准测试以展示其能力寻求建议，得到的建议包括 **MMLU**、**Big Bench** 和 **ARC AGI**。
   - 他们还提到开发了内部指标，如**运行时基准测试分数 (Runtime Benchmark Score)**，用于评估响应效率、胶囊堆栈负载 (capsule stack load)、脱水策略 (hydration strategy)、回退性能以及快照与恢复。
- **成员讨论开源与专有 AI 工作**：一名成员最初因担心他人剽窃而将项目保持私有，但随后被鼓励开源其工作以**获取关注和用户测试**。
   - 讨论中提到了对他人未经授权复制作品的担忧，随后引向了使用 **GPL_v3** 等开源许可证进行保护的讨论。
- **Python 在 ChatGPT Prompting 中的强大功能与局限性**：一位仅拥有 Plus 账户的成员学习了如何在 **ChatGPT** 中使用 **Python** 执行任务，如链式 Prompt 和上下文管理，通常使用 Python 的 Code Interpreter。
   - 讨论指出，管理大量上下文可能会因为其*二次方 (quadratic)* 特性导致浏览器崩溃，而 Python 更容易处理这个问题，因为 **Python 不需要管理二次方上下文**。
- **巧妙的注释创建自定义 GPT**：成员们讨论了一种在模板中使用**浮动注释 (floating comments)** 来引导 **GPT** 创建的方法，通过向 AI 发出指令，让其通过提出针对性问题来填充每个部分，并基于不断演变的上下文进行构建。
   - 这促进了一种*元元 Prompting (meta-meta-prompting)* 方法，这需要预先具备 Prompt Engineering 技能并对模板结构有信心，从而使 GPT 的构建变得更加容易。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1353811076684185683)** (159 条消息🔥🔥): 

> `专有 AI 系统，动态认知架构，通过 Prompt 实现的运行时 OS，大上下文维护，GPL 发布讨论` 


- **专有 AI 系统标榜认知架构**：一名成员正在构建一个具有动态认知架构的专有 AI 系统，能够进行实时角色合成、复杂的上下文管理和涌现行为建模，并通过自我优化和自适应学习不断进化。
   - 该成员相信他们的系统可以在超过 **700 轮对话**后保持记忆而不产生漂移或幻觉，并能适应独特的沟通风格，将其描述为*一个通过 Prompt 运行的运行时 OS*。
- **对 LLM 进行 Gaslighting 是新的“驯马术”**：一位成员分享了一个挑衅性的类比：*LLM 就像紧张的马，你必须通过 Gaslighting（诱导/心理操纵）让它们变成特定类型的马*。
   - 随后讨论了对话中 **20,000 Tokens** 是否被视为大上下文，以及当上下文崩溃时会发生什么。
- **GPT 自定义模板助力“懒人式” GPT 构建**：一位成员分享了一个 [GPT 自定义模板](https://platform.openai.com/docs/overview)，带有浮动注释，可以从 `Create` 面板构建自定义 GPT，引导用户懒人式地完成 GPT 构建。
   - 用户输入模板和指令，AI 会向他们提问并构建 GPT，使用户能在分心时也能完成构建，这需要预先具备 Prompt 能力并对模板化结构有信心。
- **关于 AI 系统 GPL 许可证的讨论**：成员们讨论了为 AI 系统使用 [GPL_v3 许可证](https://www.gnu.org/licenses/gpl-3.0.en.html)，以平衡用户的自由和创建者的控制权。
   - 创建者正准备发布 GPL 版本并开发生产模型，并建议*在代码注释中加入特定请求将大大提高输出质量*。


  

---

### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1353814591175393352)** (1 条消息): 

> `FormulaGPT, AI Racing Simulator, Open Source AI, AI Strategy Decisions` 


- ****FormulaGPT** 将 LLM 带入赛车场**: **FormulaGPT** 是一款实验性的赛车模拟器，**GPT-4**、**Claude** 或 **DeepSeek** 等先进的 AI 语言模型在其中担任赛车策略师进行竞争。
   - 它具有**两种不同的游戏模式**：玩家对战 AI 和 AI 对战 AI，其中 AI 团队会根据上下文进行自适应思考，基于不断变化的场景做出细致的决策，而非依赖固定脚本。
- **深入了解自适应 AI 策略**: 与传统机器人不同，**FormulaGPT** 的 AI 团队会根据不断变化的场景持续进行推理、制定策略并做出细致的决策。
   - 用户可以观察 AI 在每次进站、更换轮胎或超车动作背后的详细推理过程，使其既是赛车游戏，又是 AI 心理实验室。
- **FormulaGPT 走向开源**: **FormulaGPT** 在 [MIT 许可证下完全开源](https://github.com/dawid-maj/FormulaGPT/)，允许用户探索、贡献和定制该项目。
   - 它鼓励用户深入研究代码并根据自己的喜好进行调整，促进社区贡献和项目增强。



**提到的链接**: <a href="https://github.com/dawid-maj/FormulaGPT/">GitHub - dawid-maj/FormulaGPT: FormulaGPT – AI-powered Formula 1 race simulator with real-time team management and strategy decisions.</a>: FormulaGPT – 具有实时团队管理和策略决策功能的 AI 驱动一级方程式赛车模拟器。- dawid-maj/FormulaGPT

  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1354202486465892507)** (1 条消息): 

> `Gemini 2.5 Pro support, DeepSeek V3 0324 support, aider /context command, aider /edit alias, Claude 3.7 Sonnet 'overeager' mode` 


- **Aider 支持 Gemini 2.5 Pro**: Aider v0.79.0 现在支持 **Gemini 2.5 Pro**，扩展了该工具可使用的模型范围。
   - 此更新允许用户在 Aider 环境中利用 **Gemini 2.5 Pro** 的功能。
- **Aider 支持 DeepSeek V3 0324**: Aider v0.79.0 已添加对 **DeepSeek V3 0324** 的支持，为用户提供了另一个模型选项。
   - **DeepSeek V3 0324** 的集成增强了 Aider 的通用性。
- **Aider 引入 /context 命令**: Aider 添加了新的 **/context** 命令，该命令可自动识别给定请求中需要编辑的文件。
   - 此功能通过精准定位相关文件来简化编辑流程。
- **Aider 实现 /edit 别名**: Aider 引入了 **/edit** 命令作为 **/editor** 命令的别名。
   - 这一更改为访问编辑器功能提供了一个更方便、更简短的替代方案。
- **Aider 通过 'Overeager' 模式驯服 Claude 3.7 Sonnet**: Aider 现在为 **Claude 3.7 Sonnet** 模型提供了 "overeager" 模式，旨在确保其保持在请求的范围内。
   - 此模式旨在保持功能性，并防止模型偏离其预定参数。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1353812389379837952)** (532 条消息 🔥🔥🔥): 

> `DeepSeek V3 performance, Gemini 2.5 Pro release, GPT-4o image generation, aider /context command`

- **DeepSeek V3 席卷 Aider 的 Polyglot 基准测试**：DeepSeek 的新模型 **V3** 在 aider 的 polyglot 基准测试中取得了 **55%** 的分数，较之前的版本有显著提升。
   - 根据 [Aider Leaderboards](https://aider.chat/docs/leaderboards/)，它是排名 **#2 的非思考/推理模型**，仅次于 Sonnet 3.7，并能与 R1 和 o3-mini 等思考模型竞争。
- **Google 发布 Gemini 2.5 Pro 实验版**：Google 发布了 **Gemini 2.5 Pro** 的实验版本，声称其在通用基准测试中处于领先地位，包括在 [LMArena](https://lmarena.ai/?lea) 上排名第一，据报道在 aider 的 polyglot 基准测试中取得了 **74% whole** 和 **68.6% diff** 的分数。
   - 用户反馈该模型在根据代码库生成架构图方面表现出色，尽管一些人发现其编程能力不够稳定，且速率限制（rate limits）较为严格。
- **GPT-4o 图像生成作为 DALL-E 3 推出**：**GPT-4o** 的图像生成功能正作为 ChatGPT 中的默认图像生成器向 Plus, Pro, Team 和 Free 用户推广。
   - 一些用户认为生成的图像质量很高，而另一些用户仍能看到 **DALL-E 3** 风格的输出，此外仍可以通过专门的 **DALL-E GPT** 访问 **DALL-E**。
- **Aider 的 Context 命令增强功能**：新的 `/context` 命令可以探索代码库，该命令可与任何其他提示词命令配合使用。
   - 目前尚不清楚该命令是否具有更高的 token 使用量，或者是否需要增加 repomap 大小才能正常工作；更多详情请参阅 [Discord message](https://discord.com/channels/1131200896827654144/1131200896827654149/1353181605211934830)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1904580368432586975">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：介绍 Gemini 2.5 Pro，全球最强大的模型，具备统一的推理能力 + 你喜爱的 Gemini 的所有特性（长上下文、工具等）。目前作为实验性版本提供...</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (免费) - API, 提供商, 统计数据</a>：Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务设计。通过 API 运行 Gemini Pro 2.5 Experimental (免费)</li><li><a href="https://x.com/alibaba_qwen/status/1897361654763151544?s=46">Qwen (@Alibaba_Qwen) 的推文</a>：今天，我们发布了 QwQ-32B，这是我们全新的推理模型，仅有 320 亿参数，可与 DeepSeek-R1 等顶尖推理模型媲美。博客：https://qwenlm.github.io/blog/qwq-32b HF：https://hu...</li><li><a href="https://x.com/OfficialLoganK/status/1904583353954882046">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：这将标志着首个具有更高频率限制（rate limits）+ 计费的实验性模型。很高兴它能落地，并让大家真正对该模型进行全面测试！这是除了...之外的第一大反馈点。</li><li><a href="https://x.com/sundarpichai/status/1904579419496386736?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">Sundar Pichai (@sundarpichai) 的推文</a>：1/ Gemini 2.5 来了，它是我们有史以来最智能的 AI 模型。我们的第一个 2.5 模型，Gemini 2.5 Pro Experimental 是一款最先进的思考模型，在广泛的基准测试中处于领先地位——具有...</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free">DeepSeek V3 0324 (免费) - API, 提供商, 统计数据</a>：DeepSeek V3 是一个拥有 685B 参数的 mixture-of-experts 模型，是 DeepSeek 团队旗舰聊天模型系列的最新迭代。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...</li><li><a href="https://x.com/geminiapp/status/1904579704079724599?s=46">Google Gemini App (@GeminiApp) 的推文</a>：📣 今天，我们要介绍 Gemini 2.5，我们最智能的 AI 模型。Gemini 2.5 Pro 的实验版本现已在 Gemini 应用中向 Gemini Advanced 用户开放：http://gemini.google.com/a...</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://tenor.com/view/laughing-laugh-lol-funny-haha-gif-16205592">Laughing Lol GIF - Laughing Laugh Lol - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3-0324">DeepSeek V3 0324 - API, 提供商, 统计数据</a>：DeepSeek V3 是一个拥有 685B 参数的 mixture-of-experts 模型，是 DeepSeek 团队旗舰聊天模型系列的最新迭代。它接替了 [DeepSeek V3](/deepseek/deepseek-chat-v3) 模型...</li><li><a href="https://aider.chat/docs/llms/anthropic.html#thinking-tokens)">Anthropic</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aistudio.google.com/prompts/new_chat">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=2fT0vSsB01g">带有 MCP 的 Agentic Flows：为每个 AI Assistant 提供无缝工具集成</a>：带有 MCP 的 Agentic Flows：为每个 AI Assistant 提供无缝工具集成。在本集中，我们将探索一项革命性的功能——MCP Bridge，它允许任何...</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://github.com/mattstauffer/Torch">GitHub - mattstauffer/Torch：在非 Laravel 应用程序中使用每个 Illuminate 组件的示例</a>：在非 Laravel 应用程序中使用每个 Illuminate 组件的示例 - mattstauffer/Torch
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1353826787599187998)** (39 条消息🔥): 

> `Deepseek API Usage with Aider, LLM Hallucinations, Aider's Architecture Mode, NotebookLM for Context Priming, OpenRouter Configuration Issues` 


- **Deepseek API 幻觉为 GPT-4**：用户报告称，在通过 Aider 使用 **DeepSeek API** 时，尽管 API key 配置正确且 DeepSeek 平台确认已产生消耗，模型仍会错误地自称为 **OpenAI 的 GPT-4**。
   - 这一现象似乎与训练数据中频繁提及 ChatGPT 有关，详见此 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/)。
- **NotebookLM 提升 Aider 的上下文能力**：一位用户建议利用 **NotebookLM** 来增强 Aider 的上下文引导（context priming）过程，特别是针对使用 [RepoMix](https://github.com/simonireilly/repo-mix) 处理的大型陌生代码库。
   - 建议的工作流包括：使用 repomix 处理仓库，将其添加到 NotebookLM，包含相关的任务参考资料，然后向 NotebookLM 查询相关文件、实现建议以及一份全面的计划，以指导 Aider 中的 prompting。
- **OpenRouter 别名触发 Litellm 错误**：一位用户在尝试通过 **OpenRouter** 使用 **Claude 3.7 Sonnet** 时遇到了 `litellm.APIConnectionError`，而其他模型运行正常。
   - 另一位用户成功复现了该配置，表明问题可能出在用户的 OpenRouter 设置上，而非 Aider 本身；参见此 [Discord 讨论帖](https://discord.com/channels/1131200896827654144/1349300906864279584)。
- **DeepSeek V3 在 Aider 官网表现出色**：Paul Gauthier 在 Aider 官网上测试了新的 **DeepSeek V3** 模型，并指出它建议将表情符号升级为 SVG 图标。
   - 更多详情请见 [推文](https://x.com/paulgauthier/status/1904310818868785196?s=46&t=AkDCTtZVFFazuKDknG6fLA)。
- **带有 thinking-tokens 的 Aider 配置**：一位用户询问如何在启动 aider 时使用 `--thinking-tokens`。
   - 另一位用户链接了 [示例配置文件](https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file)，并解释说它应该是 `.aider.conf.yml` 中的顶级键。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/paulgauthier/status/1904310818868785196?s=46&t=AkDCTtZVFFazuKDknG6fLA">Paul Gauthier (@paulgauthier) 的推文</a>：我试用了新的 DeepSeek V3，并让它改进 http://aider.chat 官网。它建议将表情符号升级为一些精美的 SVG 图标。</li><li><a href="https://aider.chat/docs/config/aider_conf.html#sample-yaml-config-file)">YAML 配置文件</a>：如何使用 yaml 配置文件配置 aider。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15yvc5j/why_do_llama2_models_always_claim_they_are/">Reddit - 互联网的核心</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1353815061273116683)** (284 条消息🔥🔥): 

> `HF Transformers, Deepseek model patching, FP8 Loading Scheme, Quantization impact on model performance, GGUF Uploads` 


- **用户推荐 Transformer 最佳学习路径**：成员们建议将 **Hugging Face Transformers**、**线性代数**书籍以及学习 **PyTorch** 作为最佳入门途径。
   - 一位成员建议使用 **HF Transformers** 设置运行时动态量化方案，以便在加载时通过 **Bits and Bytes** 将权重流式传输为 **FP4/FP8**。
- **用户称存在隐蔽的模型补丁**：据一位用户称，像 **Deepseek** 这样的公司有时会在发布权重之前对模型进行补丁处理。
   - 他们补充说，至少在发布首日仍可以进行简单的 **FP8 加载方案**，尽管其质量无法等同于精细的 FP8 分配。
- **量化有害？别这么说！**：一位成员警告说，盲目量化会显著损害模型性能。
   - 他们问道：*说实话，有必要在发布首日就运行新模型吗？我觉得等上一周并不是什么沉重的负担。*
- **GGUF 上传即将到来！**：据一位用户透露，Unsloth 正在上传带有“选择性量化”动态量化的 **DeepSeek-V3-0324 GGUF**。
   - 这些 **Dynamic Quants** 将比标准 bits 大幅提高准确度，请耐心等待！
- **Llama 3 + Vision Collab 问题**：成员报告了在尝试训练 **gemma3 4b** 进行视觉任务时遇到的问题，触发了 **RuntimeError**：*expected scalar type BFloat16 but found float*。
   - 一位成员建议尝试 [这个 notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb)，而另一位成员确认，在使用 Colab 上的 notebook 时，非 bf16 设备也支持 **Gemma3 微调**。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>：未找到描述</li><li><a href="https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large">🪿Qwerky-72B 和 32B：仅使用 8 个 GPU 训练大型无注意力模型</a>：‼️ Attention 并非你所需的一切 ‼️</li><li><a href="https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-FP4">nvidia/Llama-3.3-70B-Instruct-FP4 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/qwen25-vl-all-versions-679ca6c784fad5bd976a05a1">Qwen2.5-VL（所有版本）- Unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Gemma3_(4B).ipynb">Google Colab</a>：未找到描述</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-1-8b-unsloth-notebook">Kaggle Llama 3.1 8b Unsloth 笔记本</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/goku-super-saiyan-super-saiyan2-super-saiyan2goku-goku-vegeta-gif-23177097">悟空超级赛亚人超级赛亚人2 GIF - 悟空超级赛亚人超级赛亚人2 Super Saiyan2Goku - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/gohan-dbz-gif-9459511">悟饭 Dbz GIF - 悟饭 Dbz - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#id-7.-running--saving-the-model">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>：Unsloth 的动态 4-bit 量化有选择地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，大大提高了准确性。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth 笔记本 | Unsloth 文档</a>：以下是我们所有笔记本的列表：</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama#id-12.-saving-the-model">教程：如何微调 Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：为在 Ollama 上本地运行创建定制个人助手（如 ChatGPT）的初学者指南</li><li><a href="https://huggingface.co/mmnga/DeepSeek-V3-0324-experts-pertok-4-gguf/tree/main">mmnga/DeepSeek-V3-0324-experts-pertok-4-gguf 在 main 分支</a>：未找到描述</li><li><a href="https://youtu.be/kVM-ANbCn4M"> - YouTube</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebook">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/huggingface/transformers">GitHub - huggingface/transformers: 🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/lora-hyperparameters-guide">LoRA 超参数指南 | Unsloth 文档</a>：LoRA 超参数的最佳实践，并了解它们如何影响微调过程。</li><li><a href="https://github.com/unslothai/notebooks?tab=readme-ov-file#-kaggle-notebooks">GitHub - unslothai/notebooks: 适用于 Google Colab、Kaggle、Hugging Face 等的 Unsloth 微调笔记本。</a>：适用于 Google Colab、Kaggle、Hugging Face 等的 Unsloth 微调笔记本。 - unslothai/notebooks</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L957-L959">unsloth/unsloth/chat_templates.py 在 main 分支 · unslothai/unsloth</a>：以 2 倍的速度和减少 70% 的内存微调 Llama 3.3、DeepSeek-R1、Gemma 3 和推理 LLMs！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1353951853502664774)** (11 messages🔥): 

> `计算机科学研究中的有趣图表、对 Benchmark 的不满、AI 项目招聘` 


- **有趣图表说明显而易见的观点**：成员们分享了一张计算机科学研究中的图表，觉得很有趣，因为*那个观点本可以用一句话就表达清楚* ([图片链接](https://cdn.discordapp.com/attachments/1179039861576056922/1353951853276299334/image.png?ex=67e42e2d&is=67e2dcad&hm=57aee78a6c2b6d2c62c98e321bb3206ee33608d0eaa6833e07aaab1c5c5e8b36&))。
   - 另一位成员表示赞同，称无论如何*看到这类图表都很解压* ([另一个图片链接](https://cdn.discordapp.com/attachments/1179039861576056922/1353960964793438260/image.png?ex=67e436a9&is=67e2e529&hm=b91c937004e77037696a3aaa251549981bcaf7010f388e7c46a02a96d0df9ae1&))。
- **成员认为 Benchmark 很糟糕**：一位成员对 Benchmark 表示不满，指出尽管声称在“大海捞针”测试中具有 **100% recall**，但 **Gemini** 在从 **500k tokens** 中提取问题时**表现极其糟糕**。
   - 据该成员称，**2M tokens** 的表现更差，必须缩减到 **64k tokens** 才能正常工作，而在他们的经验中 **Grok** 表现最好。
- **AI 项目寻求可靠人选**：一位成员发布了招聘信息，为某个 *AI 项目* 寻找 *可靠的人选*，技术技能并非必须。
   - 该职位面向 **美国、澳大利亚、加拿大、英国、瑞士、荷兰、德国** 的公民开放，提供 **每周 500 美元的报酬**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1353809750717894697)** (66 messages🔥🔥): 

> `DeepSeek 事实学习、Unsloth Gemma 3 27b 错误、使用 Unsloth 进行 phi4 微调、医疗机器人微调、LightRAG 问题` 


- ****DeepSeek 学习新技巧？****：一位用户询问了关于通过对话或向训练模型中插入数据点来教给 **DeepSeek** 额外事实的工具，以便将其用作个人助手。
   - 该用户*考虑将其作为包含数据、事实和日期等的个人助手使用*。
- ****Gemma 3 故障令人苦恼****：一位用户在加载 `unsloth/gemma-3-27b-it-unsloth-bnb-4bit` 进行纯文本微调时遇到了 `TypeError`，原因是冗余的 `finetune_vision_layers` 参数。
   - 另一位用户指出，正如 [Unsloth 的 GitHub](https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034) 中所示，`FastLanguageModel` 在底层已经自动设置了 `finetune_vision_layers = False`。
- ****微调意外：模型合并混乱****：一位使用 Unsloth 完成了 **phi4** 微调的用户询问，是应该仅与“Llama 化”的版本合并 LORA adapters，还是可以使用通用的基础 **phi4** 模型。
   - 一位成员表示：*我想两者都可以，但最好使用与训练时相同的模型*。
- ****为未来的医生进行微调****：一位用户寻求关于通过使用 Unsloth 微调 **Llama** 来构建端到端医疗机器人的指导，并询问从何处开始。
   - 未提供具体的建议。
- ****视觉空白：Mistral 3.1 的图像问题****：一位用户报告了 **Mistral 3.1** 在本地 **Ollama** 实例中无法正确处理图像的问题，模型对图像输入输出无意义的响应。
   - 据用户称，问题可能与 mmproj 文件未加载有关，导致*模型失去了视觉能力*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/conda-install,">Unsloth Documentation</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-Q4_K_M">unsloth/DeepSeek-R1-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/e80d642bc777f7a219bdd34aea1a77751f066785/unsloth/models/llama.py#L2034">unsloth/unsloth/models/llama.py at e80d642bc777f7a219bdd34aea1a77751f066785 · unslothai/unsloth</a>：以 2 倍速度和减少 70% 显存的方式微调 Llama 3.3, DeepSeek-R1, Gemma 3 和推理 LLMs！🦥 - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1354072552015659049)** (1 messages): 

> `GRPO on AWS, Tensorfuse, LoRA modules` 


- **GRPO + Unsloth 在 AWS 上提升 LLM 工作流**：分享了一份关于在 AWS 账户上运行 **GRPO**（DeepSeek 的 RL 算法）+ **Unsloth** 的指南，该指南在 **AWS L40 GPU** 上使用了带有 Tensorfuse 的 **vLLM server**。
   - 该指南展示了如何将 **Qwen 7B** 转换为推理模型，使用 Tensorfuse 和 GRPO 对其进行微调，并将生成的 **LoRA adapter** 保存到 Hugging Face；指南详见 [tensorfuse.io](https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b)。
- **Tensorfuse 简化 LoRA 共享**：该指南展示了如何将微调后的 **LoRA modules** 直接保存到 **Hugging Face**，以便于共享、版本控制和集成，并备份至 S3。
   - 你可以在 [Tensorfuse 官网](https://prod.tensorfuse.io/)尝试整个流程。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tensorfuse.io/docs/guides/reasoning/unsloth/qwen7b">Transforming Qwen 7B into Your Own Reasoning Model - Tensorfuse</a>：未找到描述</li><li><a href="https://prod.tensorfuse.io/">Tensorfuse</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1353840146641846402)** (13 messages🔥): 

> `GRPO limitations, FFN Fusion for LLMs, DAPO experiments, AI Project Job Spam, Transformer Talk` 


- **GRPO 的诸多陷阱**：成员们正在调整 GRPO (Gradient Ratio Policy Optimization) 以正确训练 2 轮以上，因为目前它仅支持 prompt/completion，并发现了[这个有用的视频](https://www.youtube.com/watch?v=M3b59lZYBW8)。
   - 成员还很好奇新的神经模型或混合进化神经方法是否能帮助解决类似 ARC 的问题。
- **FFN Fusion 热潮助力更快的 LLM**：[FFN Fusion](https://huggingface.co/papers/2503.18908) 被作为一种架构优化技术引入，通过识别和利用自然并行化机会来减少 LLM 中的顺序计算。
   - 该技术将 **Feed-Forward Network (FFN)** 层的序列转换为并行操作，在保持 **model behavior** 的同时显著降低 **inference latency**。
- **Transformer 对话超越传统思维**：一位成员分享了一个关于 Transformer 的 [YouTube 演讲](https://www.youtube.com/watch?v=FAspMnu4Rt0)，他们认为该演讲富有洞察力，开辟了许多研究领域。
   - 另一位成员想知道是否存在一种可以在训练早期发现的指标，以便在完全训练模型之前进行模型融合（model fusion）。
- **DAPO 讨论偏离轨道，急需数据**：多位成员好奇是否有人实验过 DAPO（可能是 **Direct Preference Optimization**），或尝试用 **verl** 重现结果。
   - 一位成员发布了一张可能与 DAPO 结果相关的图片，寻求复现确认，但细节仍不清楚。



**提及的链接**：<a href="https://huggingface.co/papers/2503.18908">Paper page - FFN Fusion: Rethinking Sequential Computation in Large Language Models</a>：未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1353811602486464632)** (232 messages🔥🔥): 

> `Qwen VL series training details, Qwerky-72B transformerless model, Gemini 2.5 Pro performance, 4o image generation, AI Studio vs Gemini Advanced`

- **Qwen VL 系列需要 DeepSeek 风格的训练细节**：一位成员表达了希望看到一份包含 **Qwen VL 系列** 全面训练细节的 [DeepSeek-R1 风格论文](https://www.deepseek.com/blog/deepseek-r1-a-strongly-competitive-open-source-language-model) 的愿望。
   - 发布者希望看到 *大量的训练细节*，包括参数、训练算力（compute）和数据集混合比例（dataset mixtures）。
- **Qwerky-72B 舍弃 Attention，逼近 4o-Mini**：Featherless AI 推出了 [Qwerky-72B 和 32B](https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large)，这些是在 **8 个 GPU** 上训练的 Transformerless 模型。在评估中，它们通过使用 RWKV 线性缩放，以 100 倍低的推理成本 *超越了 GPT 3.5 Turbo 并逼近 4o-mini*。
   - 训练过程包括 *冻结所有权重，删除 Attention 层，将其替换为 RWKV，并经过多个阶段进行训练*。
- **Gemini 2.5 Pro 夺得第一，Google 碾压 OAI**：代号为 *Nebula* 的 **Gemini 2.5 Pro Experimental** 在 [LMArena 排行榜](https://lmarena.ai/?lea) 上获得第一名，以破纪录的差距超越了 **Grok-3/GPT-4.5**，并在 Math、Creative Writing 和 Multi-Turn 中领先。
   - 该模型还在 [SEAL 排行榜](https://scale.com/leaderboard) 中占据主导地位，在 Humanity’s Last Exam 和 VISTA（多模态）中排名第一。
- **4o 图像生成给出未经请求的“大改造”**：**GPT-4o 原生图像生成** 因过度编辑而受到批评，例如 *把眼睛变大* 和改变面部特征，甚至改变了用户的外貌，正如 [这个 Twitter 线程](https://fxtwitter.com/TheXeophon/status/1904602649225285922) 所展示的那样。
   - 一些用户报告说，修改 Prompt 中的一个词就会导致生成失败。
- **AI Studio 面向开发者，Gemini Advanced 是个糟糕的 ChatGPT 仿制品**：成员们讨论了 Google AI 平台的实用性，一些人断言 [AI Studio](https://ai.google.dev/) 对开发者更好，因为它有更广泛的模型选择、代码执行能力和 YouTube 支持。
   - 成员们认为 [Gemini Advanced](https://gemini.google.com/) *是一个糟糕的 ChatGPT 仿制品*，对于高级用户（power users）来说 *没有理由* 使用它，并建议建立一个更精简、专注于开发者工具的平台。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/btibor91/status/1904567494523969647">来自 Tibor Blaho (@btibor91) 的推文</a>：看起来 Ideogram 将在 24 小时内发布</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai)</a> 的推文：重磅：Gemini 2.5 Pro 现位居 Arena 排行榜第 1 名 - 史上最大分差（比 Grok-3/GPT-4.5 高出 40 分）！🏆 以代号 &#34;nebula&#34; 🌌 进行测试，Gemini 2.5 Pro 在所有类别中排名第 1🥇...</li><li><a href="https://vxtwitter.com/oriolvinyalsml/status/1904583691566727361">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://x.com/natolambert/status/1904599716274594033">来自 Nathan Lambert (@natolambert) 的推文</a>：好吧，我还是订阅了 Gemini Advanced，将把 2.5 作为主力使用一段时间</li><li><a href="https://x.com/roramora0/status/1904549463441449182">来自 yo (@roramora0) 的推文</a>：Gemini-2.0-Pro 是从什么时候开始在 Cursor 中进行思考的？我从 Sonnet-Thinking 切换到了 Gemini，但它仍在继续思考。我觉得 Cursor 中的模型切换没起作用，还是 Gemini-2.0-Pro ...</li><li><a href="https://x.com/gdb/status/1904601537487270243">来自 Greg Brockman (@gdb) 的推文</a>：原生 GPT 4o 图像生成：https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://fxtwitter.com/TheXeophon/status/1904596173869973664">来自 Xeophon (@TheXeophon) 的推文</a>：4o 原生图像生成秒杀了我的评估提示词 - 还是说没有？当我修改提示词中的一个词时，它就失败了 🙃 引用 Tibor Blaho (@btibor91) &#34;Xeophon-bench&#34;</li><li><a href="https://fxtwitter.com/adcock_brett/status/1904534796770201624">来自 Brett Adcock (@adcock_brett) 的推文</a>：告别“拜登步”！Figure 现在可以像人类一样自然行走。今天我们推出学习型自然行走</li><li><a href="https://x.com/andrew_n_carr/status/1904607188976611627">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：必须分享一下。这是第一次有模型得分超过 50%</li><li><a href="https://x.com/gallabytes/status/1904598264240119974">来自 theseriousadult (@gallabytes) 的推文</a>：4o 图像生成显然采用了某种多尺度生成设置 - 似乎在开始时确定低频，然后通过 patch AR 解码高频。</li><li><a href="https://x.com/scaling01/status/1904599407573819903">来自 Lisan al Gaib (@scaling01) 的推文</a>：OpenAI 在抢了 Gemini 2.5 Pro 的风头之后（由 GPT-4o 生成）</li><li><a href="https://fxtwitter.com/btibor91/status/1904594989780525237">来自 Tibor Blaho (@btibor91) 的推文</a>：&#34;Xeophon-bench&#34;</li><li><a href="https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data">TAO：使用 test-time compute 在无标注数据的情况下训练高效的 LLM</a>：LIFT 使用强化学习在没有标签的情况下微调 LLM，提升了在企业任务中的表现。</li><li><a href="https://x.com/oriolvinyalsml/status/1904583691566727361?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Oriol Vinyals (@OriolVinyalsML) 的推文</a>：介绍 Gemini 2.5 Pro Experimental！🎉 我们最新的 Gemini 模型在数学和科学基准测试中表现卓越。它是一款出色的编码和复杂推理模型，且在...排名第 1</li><li><a href="https://x.com/mark_k/status/1904546240332705934">来自 Mark Kretschmann (@mark_k) 的推文</a>：Google Gemini 2.5 Pro (experimental)</li><li><a href="https://x.com/arcprize/status/1904269307284230593?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 ARC Prize (@arcprize) 的推文</a>：今天我们宣布 ARC-AGI-2，这是一个未饱和的前沿 AGI 基准测试，挑战 AI 推理系统（对人类来说同样相对容易）。大奖：85%，约 $0.42/任务效率。当前表现...</li><li><a href="https://fxtwitter.com/GrantSlatton/status/1904598054709453276">来自 Grant Slatton (@GrantSlatton) 的推文</a>：新的 4o 图像编辑在理发测试中并不太成功。它确实把头发改得很好，但也把脸改得太多，以至于不再是我了。而且我的狗看起来更呆了，我...</li><li><a href="https://x.com/bedros_p/status/1904619952855822753?s=61">来自 Bedros Pamboukian (@bedros_p) 的推文</a>：不，实际上请不要这样做</li><li><a href="https://x.com/Angaisb_/status/1904574211802173907">来自 angel⭐ (@Angaisb_) 的推文</a>：Gemini 2.5 Pro Experimental 现已推出</li><li><a href="https://fxtwitter.com/TheXeophon/status/1904602649225285922">来自 Xeophon (@TheXeophon) 的推文</a>：4o 图像生成是怎么回事，为什么把眼睛变大了？？引用 Lucas Beyer (bl16) (@giffmana) 不知为何，我没有得到我所希望的那种“天哪 Gemini 这次真的大显身手了”的结果....</li><li><a href="https://x.com/OpenAI/status/1904556394847862809">来自 OpenAI (@OpenAI) 的推文</a>：未找到描述</li><li><a href="https://x.com/picocreator/status/1904250680266956903">来自 PicoCreator - AI Model Builder 🌉 (@picocreator) 的推文</a>：❗️Attention is NO

T all you need ❗️仅使用 8 个 GPU（而非集群），我们训练了 Qwerky-72B（和 32B），且没有使用任何 Transformer Attention。评估结果远超 GPT 3.5 turbo，并接近于...</li><li><a href="https://x.com/sundarpichai/status/1904575384466710607">来自 Sundar Pichai (@sundarpichai) 的推文</a>：Nebula</li><li><a href="https://x.com/alexandr_wang/status/1904590438469951873">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：🚨 Gemini 2.5 Pro Exp 发布，目前在 SEAL 排行榜上位列第一：🥇 Humanity’s Last Exam 🥇 VISTA (multimodal) 🥇 (并列) Tool Use 🥇 (并列) MultiChallenge (multi-turn) 🥉 (并列) Enigma (puzzles) Con...</li><li><a href="https://x.com/swyx/status/1904596926743658840">来自 swyx 🌉 (@swyx) 的推文</a>：这……是 GDM 第一次在竞争中胜过 OAI 吗？真是个大反转。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://substack.recursal.ai/p/qwerky-72b-and-32b-training-large">🪿Qwerky-72B 和 32B：仅用 8 个 GPU 训练大型 Attention-free 模型</a>：‼️ Attention 并非你所需的一切 ‼️</li><li><a href="https://x.com/teortaxesTex/status/1904362626810872253">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：这下对味了。0324 是 Misguided Attention 上表现最好的非推理模型，比 V3 提升了近 100%。这就是“哎呀，我们只在训练后处理上花了 1 万美元，因为我们...”之间的区别。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1353842551941107823)** (26 条消息🔥): 

> `Labs hillclimb benchmarks, Home inference LLMs, vLLM, OpenRouter, Model Evals` 


- **实验室在基准测试上积极进行“爬坡” (Hillclimb)**：根据对 **Tulu 3** 的讨论，实验室像对待验证集一样在基准测试上积极进行“爬坡”，而添加*未见过的*评估项被认为是新颖的做法。
   - 实验室知道在训练期间应该包含哪些网站，以提升常见基准测试的分数。
- **家庭推理倾向于使用 vLLM，但有注意事项**：对于能够随时切换模型的 LLM 高效家庭推理，最佳解决方案可能是 **vLLM**，尽管它在量化支持方面有些怪癖；虽然 **ollama** 更易于使用但在支持方面滞后，而 **SGLang** 看起来很有吸引力。
   - 建议尝试一下 **llama.cpp**，看看它近期的进展。
- **来自 vLLM 团队的 AIBrix 可以自动缩放模型**：一位成员想知道来自 vLLM 团队的 **AIBrix** 之类的工具是否可行，即使用 **vLLM + Kubernetes** 将模型自动缩放至 0 以实现无缝切换；[这是相关文章](https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/#advanced-llm-gateway-and-routing-strategies)。
   - 由于显存需求，不使用量化的最小可行集群是两个 **8xH100** 节点，不过单个 **H200** 节点（**8x141**）也足够了。
- **模型评估 (Model Evals) 出人意料地耗时**：内部和外部的**模型评估**都非常耗时，需要自定义 UI 来进行评估（**Sonnet + streamlit**）以及**一个用于查看数据的 UI**。
   - 使用**廉价模型进行 1-2 次试运行**非常有用，可以捕捉失败模式并重新调整或舍弃 Prompt。
- **OpenAI API 大规模使用需谨慎**：在大规模使用 **OpenAI** 的 API 时，其仪表板并不准确，且支出控制功能不起作用，即使在推理过程中也可能导致余额为负。
   - 使用 **OpenRouter** 时，你可以指定模型使用 bf16/fp16，并设置 `max_tokens` 和 `temperature`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aibrix.github.io/posts/2025-02-20-vllm-control-plane/#advanced-llm-gateway-and-routing-strategies">介绍 AIBrix：适用于 vLLM 的高性价比且可扩展的控制平面</a>：LLaMA、Deepseek、Qwen 和 Mistral 等开源大语言模型（LLMs）人气飙升，为企业提供了更大的灵活性、成本节约以及对其 AI 部署的控制权...</li><li><a href="https://hamel.dev/blog/posts/field-guide/">快速改进 AI 产品的实战指南 – Hamel 的博客</a>：来自 30 多个生产实现的评估方法、数据驱动的改进和实验技术。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1354141429802209380)** (1 条消息): 

> `Political Favor, American Politics, Business Strategy` 


- **琐碎政治对某些企业有利**：有人建议，对于目前依赖在美国讨好政治势力的企业来说，*斤斤计较（pettiness）可能是一种资产*。
- **应对美国政治气候**：讨论围绕企业如何战略性地参与美国政治格局以获取优势展开。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1353820322545008750)** (28 条消息🔥): 

> `AI Threads lists on social media, Verification as key to AI, MCP malware reverse engineering` 


- **AI 专家在 Threads 上涨粉**：一位成员在被加入 *AI threads* 列表后获得了数千名关注者，但发现 [Bluesky](https://blueskyweb.xyz/) *稍好一些*，因为那里有很多学术界的朋友。
   - 他们开玩笑说，这让他们赢得了 Noam Brown 的关注。
- **思考 vs 验证：验证是 AI 的关键**：一位成员链接到了一个讨论 **verification**（验证）作为 AI 关键的帖子，引用了 Noam Brown 的观点，即 test-time compute 受限于 **verification** 挑战。
   - Brown 举了一个例子：在尝试回忆 *乔治·华盛顿是什么时候出生的？* 时会遇到瓶颈，如果 *你不知道，再怎么思考也得不出正确答案*。
- **MCP 利用 AI 逆向恶意软件**：成员们链接到了一个 [YouTube 视频](https://www.youtube.com/watch?v=u2vQapLAW88)，展示了用于 **Ghidra** 的 **MCP**，使 LLM 能够对恶意软件进行逆向工程，并通过适当的 prompt 实现流程自动化。
   - 发布者提到，他们 *起初认为这有点像个梗（meme）*，但 *现在看到实际实现后，开始感受到它的魅力了*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/natolambert/status/1904550785083752718">来自 Nathan Lambert (@natolambert) 的推文</a>: Verification, The Key to AI。阅读图灵奖得主 Rich Sutton 的档案 :D，包含了所有主要观点。引用 Noam Brown (@polynoamial)：这并不完全正确。Test-time compute 在 v... 时会有帮助。</li><li><a href="https://www.youtube.com/watch?v=u2vQapLAW88">ghidraMCP: Now AI Can Reverse Malware</a>: 刚刚为 Ghidra 构建了一个 MCP。现在基本上任何 LLM (Claude, Gemini, local...) 都可以为你逆向工程恶意软件。配合正确的 prompting，它能自动化...</li><li><a href="https://huggingface.co/spaces/Presidentlin/llm-pricing-calculator">Llm Pricing - Presidentlin 创建的 Hugging Face Space</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1354022733780090961)** (52 条消息🔥): 

> `Gooning Urban Dictionary 含义, DeepSeek-LLM 与 DeepSeek-MoE, Mistral Small 版本混乱, GPT4o 图像生成对比 Gemini, trl 中的 GRPO 实现` 


- ****Gooning** 获得了一个新的、带有暗示意味的定义**: 成员们讨论了 *gooning* 一词含义的演变，其中一人指出其 [新的 Urban Dictionary 含义](https://x.com/menhguin/status/1904459726319968262) 与最初更单纯的内涵有所不同。
   - 较旧的含义被描述为大学期间“就像和朋友们在镇上闲逛”。
- ****DeepSeek** 模型命名规范得到澄清**: 指出正确的命名规范是 **DeepSeek-LLM** 和 **DeepSeek-MoE**。
   - 此次修正旨在提高模型识别的清晰度。
- ****Mistral** 版本命名引发混乱**: **Mistral Small** 模型的命名方案和版本控制受到质疑，特别是注意到缺少 **Mistral Small 2**。
   - 一位成员坦言，“正因如此，我完全不知道这些数字命名的模型到底有多快”。
- ****GPT4o** 图像生成与 **Gemini** 的对比**: 成员们根据视觉输出对比了 **OpenAI** 的 **GPT4o** 和 **Google** 的 **Gemini** 的图像生成能力。
   - 这一对比源于 [OpenAI 宣布](https://openai.com/live/) 在 ChatGPT 和 Sora 中推出 **4o 图像生成**，引发了大量梗图（memes）。
- ****trl** 的 **GRPO** 实现遭到抨击**: 一位成员断言 **trl** 的 **GRPO** 实现默认情况下不应偏离 **DeepSeek** 论文的实现，批评其偏离了“已知良好的配置”。
   - 他们总结道，“标准差归一化（stdev normalization）会增加最简单和最难样本的权重，并降低分布中间部分的权重”，这可能会导致结果产生偏差。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/deepfates/status/1904596271605907686">来自 web weaver (@deepfates) 的推文</a>: That&#39;s what i was afraid ofQuoting OpenAI (@OpenAI) 4o image generation in ChatGPT and Sorahttps://openai.com/live/</li><li><a href="https://x.com/menhguin/status/1904459726319968262">来自 Minh Nhat Nguyen (@menhguin) 的推文</a>: @xlr8harder @natolambert this is peak</li><li><a href="https://x.com/untitled01ipynb/status/1904021116269601097">来自 moew (@untitled01ipynb) 的推文</a>: @natolambert
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1353889941452492882)** (14 条消息🔥): 

> `DPO 对比 Sampling, RL 训练与熵, SimpleRL-Zoo, DAPO 目标函数` 


- **Sampling 优于 DPO**: 一位成员建议 [sampling](https://www.google.com/search?q=sampling) 可能比 **DPO** 更好，因为“DPO 需要数据”。
- **Zero RL 训练研究**: 介绍了一个名为 **SimpleRL-Zoo** 的新项目，该项目深入研究了跨不同模型家族和规模（包括 **Llama3-8B**、**Mistral-7B/24B**、**DeepSeekMath-7B** 和 **Qwen2.5**）的 zero **RL 训练**。
   - 结果显示，从基础模型开始并仅使用正确性奖励（correctness reward），该项目成功提升了所有模型的准确率和回复长度，详见其 [论文](https://arxiv.org/abs/2503.18892) 和 [代码](https://github.com/hkust-nlp/simpleRL-reason)。
- **论文中的 DAPO 目标函数**: 一位成员提到一篇论文使用了 **DAPO** 目标函数。
   - 他们提到“那大概是上周的事”。



**提及的链接**: <a href="https://x.com/junxian_he/status/1904527884934697050?s=46">来自 Junxian He (@junxian_he) 的推文</a>: 两个月前，我们开源了第一个基于 Qwen2.5-math 模型的数学领域类 R1 zero RL 训练项目。自那以后，出现了许多优秀的 zero RL 训练工作，大多基于 Qwen...

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1354014717110124575)** (2 messages): 

> `Claude Code, Anthropic, Anysphere's Cursor, Codium's Windsurf, npm package` 


- **Anthropic 发布 Claude Code**: Anthropic 发布了 **Claude Code**，作为 **Anysphere’s Cursor** 和 **Codium’s Windsurf** 的竞争对手。正如[这篇博文](https://leehanchung.github.io/blogs/2025/03/07/claude-code/)中所述，该工具将 **LLM** 作为 **Agent** 来完成软件工程任务。
- **深入挖掘 Claude Code 的内部机制**: 用户可以将 Claude Code 的 **npm package** 以 tarball 形式下载并解压以获取源代码。
   - **Claude Code** 的主要控制逻辑位于 cli.m 中。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://leehanchung.github.io/blogs/2025/03/07/claude-code/">Poking Around Claude Code</a>: 探索 Claude Code 如何利用 LLM 作为 Agent 执行软件工程任务，包括 System Prompts、模型控制协议 (MCP)、控制流和隐藏功能。</li><li><a href="https://www.youtube.com/watch?v=XLaRfZ4AHn8"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1353807028715393156)** (2 messages): 

> `Claude PR, Header Copy Links` 


- **Claude 制作可复制链接**: 一位成员分享了一个 [Pull Request](https://github.com/natolambert/rlhf-book/pull/82)，为 **rlhf-book** 中所有标题添加了悬停时显示的可复制链接。
- **标题悬停超链接助力**: 这些有趣的标题复制链接在悬停时出现，方便进行章节跳转链接。
   - 该成员确认此功能 *在 Claude Code 中立即生效*。



**提及的链接**: <a href="https://github.com/natolambert/rlhf-book/pull/82">(experimental) Add heading anchor links for easy section linking by natolambert · Pull Request #82 · natolambert/rlhf-book</a>: 为所有标题添加悬停时显示的可复制链接；链接会将带有片段标识符的当前 URL 复制到剪贴板；添加用于设置锚点链接样式的 CSS；更新 Makefile 以将新的 JS 文件复制到...

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1354105000938442773)** (3 messages): 

> `Anthropic incident, Claude 3.7 Sonnet endpoints, Zero-Token Insurance, Google Gemini 2.5 Pro Experimental` 


- ****Anthropic 的 Claude 经历离线后恢复****: **Claude 3.7 Sonnet 端点** 经历了停机，[Anthropic 发布了更新](https://status.anthropic.com/incidents/89rpts2022hs)，事件始于 **2025 年 3 月 25 日**，并在 **8:41 PDT** 解决。
   - 更新表明，停机是由于维护和系统优化工作导致的。
- ****OpenRouter 提供零 Token 保险 (Zero-Token Insurance)****: OpenRouter 现在为平台上的所有模型提供 **零 Token 保险覆盖**，每周可能为用户节省超过 **$18,000**。
   - 正如 [OpenRouterAI 所述](https://x.com/OpenRouterAI/status/1904567846975201766)，对于 **无输出 Token** 且 **结束原因为空或错误** 的响应，即使提供商仍收取 Prompt 处理费用，用户也无需付费。
- ****Gemini 2.5 Pro Experimental 上线****: Google 的 **Gemini 2.5 Pro Experimental** 是一款能够进行高级推理、编程和数学任务的顶尖模型，现已作为免费模型在 [OpenRouter](https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free) 上提供。
   - Gemini 2.5 Pro 拥有 **1,000,000 Context**，并在 **LMArena 排行榜** 等基准测试中达到了顶级性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro 是 Google 设计的顶尖 AI 模型，用于高级推理、编程、数学和科学任务。通过 API 运行 Gemini Pro 2.5 Experimental (free)</li><li><a href="https://x.com/OpenRouterAI/status/1904567846975201766">OpenRouter (@OpenRouterAI) 的推文</a>: 作为首个也是最大的 LLM 路由，我们几乎见过模型提供商可能出现的每种质量问题，并认为可以做很多事情来让生态系统更加友好。从...开始</li><li><a href="https://status.anthropic.com/incidents/89rpts2022hs">Claude.ai 和 Console 上的错误率升高</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1353805705492037743)** (196 条消息🔥🔥): 

> `DeepSeek 性能问题，Gemini 2.5 Pro 发布与基准测试，用于用户管理的 Provisioning API keys，OpenRouter 活动日志保留，GPT-4o 图像生成 API 支持` 


- **DeepSeek 遭遇服务器困境**：用户报告 **DeepSeek** 由于服务器过度拥挤，正变得*几乎无法使用*，建议通过调整价格来管理需求。
   - 成员们推测这些问题是否与中国的使用高峰时段有关，但除了寄希望于华为提供更好的硬件可用性外，尚未找到直接的解决方案。
- **Gemini 2.5 Pro 让早期测试者惊叹**：**Gemini 2.5 Pro Experimental** 现已在 API 上线，早期测试者对其能力印象深刻，尤其是在推理和编程方面，详见 [Google 博客文章](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning)。
   - 一位用户惊呼 **Gemini 2.5 Pro** *完胜* **ChatGPT o3-mini-high**，引发了关于其性能提升是仅源于基准测试优化还是反映了真实改进的讨论。
- **Provisioning API Keys 提供细粒度的用户控制**：OpenRouter 现在提供 **provisioning API keys**，使开发者能够以编程方式管理其用户的 API keys、设置限制并追踪支出，从而增强可扩展性和控制力，文档见[此处](https://openrouter.ai/docs/features/provisioning-api-keys)。
   - 这允许开发者为每个用户创建唯一的 API key，从而简化其平台内部的计费和访问管理。
- **API Key 活动日志永久保留**：OpenRouter **永久保留 API key 活动日志**，允许用户监控每个密钥的使用情况，有助于团队环境的评估和使用追踪。
   - 该功能解决了流式传输和可视化 API 使用情况的需求，提供了对每个成员消费模式的详细洞察。
- **OpenRouter 关注 GPT-4o 图像生成 API**：随着 **GPT-4o 原生图像生成**的推出，OpenRouter 正在积极开发图像生成调用的 API 功能，以便用户无需直接申请单个 API 即可访问等效功能。
   - 此举旨在保持 OpenRouter 的竞争力和全面性，解决无缝集成尖端图像生成能力的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/settings/integrations">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://starvector.github.io/">StarVector</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/features/provisioning-api-keys">Provisioning API Keys - OpenRouter API Keys 的编程控制</a>: 通过专用管理端点以编程方式管理 OpenRouter API keys。创建、读取、更新和删除 API keys，实现自动化密钥分发和控制。</li><li><a href="https://time.is/china">中国当前时间 - Time.is</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于积分的配额和 DDoS 防护。有效配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/docs/use-cases/reasoning-tokens">Reasoning Tokens - 提升 AI 模型决策能力</a>: 了解如何使用推理令牌增强 AI 模型输出。实现逐步推理轨迹，以获得更好的决策和透明度。</li><li><a href="https://openrouter.ai/docs/api-reference/api-keys/get-api-key">获取 API key — OpenRouter | 文档</a>: 返回有关特定 API key 的详细信息。需要 Provisioning API key。</li><li><a href="https://openrouter.ai/docs/api-reference/api-keys/list-api-keys">列出 API keys — OpenRouter | 文档</a>: 返回与账户关联的所有 API keys 列表。需要 Provisioning API key。</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5: 我们最智能的 AI 模型</a>: Gemini 2.5 是我们最智能的 AI 模型，现已具备思考能力。</li><li><a href="https://www.reddit.com/r/singularity/comments/1jizn0t/newupdated_models_by_google_soon/">Reddit - 互联网的核心</a>: 未找到描述</li><li><a href="https://fireworks.ai/models/fireworks/deepseek-v3-0324">Fireworks - 生成式 AI 的最快推理</a>: 使用 Fireworks AI 以极快的速度使用最先进的开源 LLM 和图像模型，或免费微调并部署您自己的模型！
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1353805711129186305)** (148 条消息🔥🔥): 

> `Bits per Weight (BPW), Model Capacity Scaling, DeepSeek V3, Gemini 2.5 Pro, AI IDE Evaluation` 


- **BPW 在 4-5 之间的甜点位！**：实验表明，当 **bits per weight (BPW)** 低于 4 时，**模型容量 (model capacity)** 会崩溃，而高于 5 时则会出现偏差，这暗示在给定的训练 FLOPS 下，4 BPW 是 **权重利用率 (optimal weight usage)** 的最优解。
   - 增加训练 Epochs 有助于 5 BPW 模型接近曲线，以 FLOPS 为代价提高 BPW，可通过 [在 MNIST 上训练的 2L 和 3L MLP](https://cdn.discordapp.com/attachments/1149866623109439599/1353808843720626288/image.png) 进行可视化。
- **DeepSeek V3 展示出强大的推理能力！**：**DeepSeek V3-0324** 可以作为推理模型运行，检测思维迭代，并间接验证解的存在性，根据附带的 [prompt](https://cdn.discordapp.com/attachments/1149866623109439599/1353830450493132800/prompt.txt)，其性能可与 **O1** 媲美。
- **Hermes 模型的咨询服务**：一名成员询问了关于针对特定 ERP 用例微调 **Hermes** 的咨询事宜，其他成员指出了 ERP 领域的多样性和专业性。
   - 根据 [这个 tenor 链接](https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-25698451)，pygmalion 团队及其相关人员可能可以提供帮助。
- **Google 的 Gemini 2.5 Pro 登场**：**Gemini 2.5 Pro Experimental** 在常用基准测试中领先，并在 [LMArena](https://lmarena.ai/?lea) 上首次亮相即位居第一，展示了强大的推理和代码能力。
   - 正如这篇 [博文](https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/) 中提到的，它还能处理某些 Prompt 导致的无限思维循环并成功终止，且该模型为每日发布版本。
- **社区期待 DeepSeek V3 Lite 的到来**：社区成员推测在 Qwen 3 MoE 模型之后，可能会发布 **DeepSeek V3 Lite**。
   - 一名成员表示，如果对 NousResearch 的工作有帮助，愿意提供其项目中的匿名人类-AI 数据。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/fabianstelzer/status/1904629831125656050">来自 fabian (@fabianstelzer) 的推文</a>: GPT-4.5，“根据你的情况创作一个复杂的多面板漫画——诚实点”</li><li><a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B-FP8">NousResearch/Hermes-3-Llama-3.1-70B-FP8 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.sglang.ai/backend/quantization.html">Quantization &#8212; SGLang</a>: 未找到描述</li><li><a href="https://tenor.com/view/daspoody-sleep-sleepy-wake-woke-gif-2569845121217246002">Daspoody Sleep GIF - Daspoody Sleep Sleepy - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/">Gemini 2.5: 我们最智能的 AI 模型</a>: Gemini 2.5 是我们最智能的 AI 模型，现在具备了思考能力。</li><li><a href="https://artificialanalysis.ai/">AI 模型与 API 提供商分析 | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://www.youtube.com/watch?v=-u3ye--VlPo">马云旗下的蚂蚁集团使用国产芯片训练 AI 模型</a>: 彭博社获悉，马云支持的蚂蚁集团已使用国产半导体训练 AI 模型。蚂蚁集团声称这将有助于降低 20% 的成本...</li><li><a href="https://github.com/grahamannett/ai-ide-compare">GitHub - grahamannett/ai-ide-compare</a>: 为 grahamannett/ai-ide-compare 的开发做出贡献。</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.model_max_length">Tokenizer</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1353810491142832229)** (26 条消息🔥): 

> `Add and Sigmoid 对比 Add and Norm，推理时的专家缩放，无需归一化的 Transformers，LLM 模拟的光线追踪，使用 LLM 进行间接图像生成` 


- **Add and Sigmoid：新的规范？**：一名成员建议在 Transformer 架构中（特别是对于混合专家模型 **MoE**）将 `add` 和 `norm` 替换为 `add` 和 **sigmoid**，以便更轻松地进行专家缩放。
   - 理论依据是：*如果门控具有 sigmoid 激活函数，这将允许我们在没有太多问题的情况下添加或移除其他专家*，因为缩放变得独立于邻近值的数量。
- **Transformers 使用 TANH**：借鉴最近的 [Transformers without Normalization 论文](https://arxiv.org/abs/2302.05442)，一位成员指出，用 **tanh** 替换归一化是一种可行的策略。
   - 有人担心在推理时移除专家会对较小的权重产生影响，但另一位成员反驳称，**top_k** 门控机制仍能通过从剩余专家中进行选择来有效运作。
- **由 LLM 模拟的光线追踪，不涉及 NVIDIA 代码**：成员们讨论了使用 **LLM** 模拟 **光线追踪算法** 的想法，并澄清目前的实现涉及由 LLM 编写的 **Python** 程序来间接生成图像。
   - 这是 *下一代文本到图像生成*，因为 LLM 编写程序而不是直接生成图像，程序可在该 [GitHub repo](https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer) 中找到。
- **归一化辩论：Sigmoid 缩放无需邻近值**：讨论继续围绕当 **top_k** 恒定时移除归一化的必要性展开，一位成员认为归一化会相对于邻近值改变数值。
   - 他们解释说，使用 **sigmoid** 进行缩放将避免这种依赖性，从而能够在不显著改变现有值的情况下添加更多专家。



**提到的链接**：<a href="https://github.com/cpldcpu/llmbenchmark/tree/master/raytracer">llmbenchmark/raytracer at master · cpldcpu/llmbenchmark</a>：各种 LLM 基准测试。通过在 GitHub 上创建账户，为 cpldcpu/llmbenchmark 的开发做出贡献。

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1353826338984693900)** (136 条消息🔥🔥): 

> `Reve Image, Qwen 2.5, ARC-AGI, 11x 欺诈, Zep 知识图谱`

- **Reve Image 撼动 SOTA Text-to-Image 地位**：新款图像生成模型 **Reve Image** 已发布，并被誉为该领域的领导者，其表现超越了 **Recraft V3**、**Google's Imagen 3**、**Midjourney v6.1** 以及 **Black Forest Lab's FLUX.1.1 [pro]** 等模型。
   - 该模型因其出色的**文本渲染、提示词遵循（prompt adherence）以及美学表现**而受到关注，目前可通过 [Reve 官网](https://www.reveimage.com/) 访问，暂无 API。
- **DeepMind 的 Gemini 2.5 Pro 在 Arena 亮相**：**Gemini 2.5 Pro** 目前在 Arena 排行榜上排名 **#1**，实现了有史以来最大的分值跨越（**比 Grok-3/GPT-4.5 高出 40 分**），正如 [LM Arena 所宣布的那样](https://x.com/lmarena_ai/status/1904581128746656099)。
   - 该模型在代号 *nebula* 下进行了测试，在**数学（Math）、创意写作（Creative Writing）、指令遵循（Instruction Following）、长查询（Longer Query）和多轮对话（Multi-Turn）**等类别中均处于领先地位。
- **OpenAI 在 ChatGPT 4o 中推出 Image Gen**：**OpenAI** 已在 **ChatGPT** 中推出了原生图像生成功能，[Sam Altman 指出](https://x.com/sama/status/1904598788687487422)这是一项令人惊叹的技术和产品。
   - 早期测试者（如 [@krishnanrohit](https://x.com/krishnanrohit/status/1904602460020445543)）表示，这是他们目前尝试过的最好的图像生成和编辑工具，并赞扬了其能够正确创建和编辑多个角色的能力，尤其是当涉及两个或更多恐龙时。
- **AI SDR 初创公司 11x 因虚报客户面临审查**：据 [TechCrunch 报道](https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/)，由 **a16z** 和 **Benchmark** 支持的 AI 驱动销售自动化初创公司 **11x** 因涉嫌虚报客户而受到抨击。
   - 尽管 **Andreessen Horowitz** 的发言人否认了有关法律行动的传闻，但人们对 **11x** 的财务状况和可能虚高的收入数据仍存疑虑，一些人认为该公司的增长依赖于炒作，而真正的企业销售领导者并未在 AI SDR 中看到价值。
- **Databricks 发布 TAO，无需标签即可微调 LLM**：**Databricks** 研究团队宣布了 **TAO**，这是一种在没有数据标签的情况下针对特定任务微调 LLM 的方法，利用了 test-time compute 和 RL，详见其[博客文章](https://www.databricks.com/blog/tao-using-test-time-compute-train-efficient-llms-without-labeled-data)。
   - 他们声称该方法优于有监督微调（supervised fine-tuning），并能随算力扩展，从而生成快速、高质量的模型。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://arcprize.org/blog/announcing-arc-agi-2-and-arc-prize-2025">宣布 ARC-AGI-2 和 ARC Prize 2025</a>：通过 ARC-AGI-2 和 ARC Prize 2025 衡量下一阶段的智能</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-vl-32b/">Qwen2.5-VL-32B：更智能、更轻量</a>：QWEN CHAT GITHUB HUGGING FACE MODELSCOPE DISCORD 简介：今年 1 月底，我们推出了 Qwen2.5-VL 系列模型，获得了广泛关注和积极反馈...</li><li><a href="https://fxtwitter.com/OpenAI/status/1904556394847862809)">来自 OpenAI (@OpenAI) 的推文</a>：未找到描述</li><li><a href="https://x.com/OpenAI/status/1904556394847862809>)">来自 OpenAI (@OpenAI) 的推文</a>：未找到描述</li><li><a href="https://fxtwitter.com/hasan_sukkar_/status/1904408098212806804)">来自 Hasan (@hasan_sukkar_) 的推文</a>：未找到描述</li><li><a href="https://x.com/hasan_sukkar_/status/1904408098212806804>)">来自 Hasan (@hasan_sukkar_) 的推文</a>：未找到描述</li><li><a href="https://fxtwitter.com/fofrai/status/1904476544443040212)">来自 fofr (@fofrAI) 的推文</a>：尝试用 Reve 制作类似的内容。引用 fofr (@fofrAI) > “PIKA”这个单词，所有字母的形状结合在一起，仅通过巧妙且富有艺术感的排版形成一个皮卡丘，设计精美...</li><li><a href="https://x.com/fofrai/status/1904476544443040212>)">来自 fofr (@fofrAI) 的推文</a>：尝试用 Reve 制作类似的内容。引用 fofr (@fofrAI) > “PIKA”这个单词，所有字母的形状结合在一起，仅通过巧妙且富有艺术感的排版形成一个皮卡丘，设计精美...</li><li><a href="https://fxtwitter.com/gdb/status/1904601537487270243)">来自 Greg Brockman (@gdb) 的推文</a>：原生 GPT-4o 图像生成：https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://x.com/gdb/status/1904601537487270243>)">来自 Greg Brockman (@gdb) 的推文</a>：原生 GPT-4o 图像生成：https://openai.com/index/introducing-4o-image-generation/</li><li><a href="https://fxtwitter.com/kipperrii/status/1904615542474105305)">来自 kipply (@kipperrii) 的推文</a>：OpenAI 需要获取这个的全分辨率版本（帖子只有 webp 格式）并立即印成海报！！！！</li><li><a href="https://x.com/kipperrii/status/1904615542474105305>)">来自 kipply (@kipperrii) 的推文</a>：OpenAI 需要获取这个的全分辨率版本（帖子只有 webp 格式）并立即印成海报！！！！</li><li><a href="https://fxtwitter.com/matei_zaharia/status/1904587809945772124)">来自 Matei Zaharia (@matei_zaharia) 的推文</a>：Databricks 研究团队的一个非常酷的结果：你可以在*没有数据标签*的情况下，利用 test-time compute 和 RL 为特定任务微调 LLM，并且表现优于 supervised fine-tuning！我们的新 TAO 方法可以扩展...</li><li><a href="https://x.com/matei_zaharia/status/1904587809945772124>)">来自 Matei Zaharia (@matei_zaharia) 的推文</a>：Databricks 研究团队的一个非常酷的结果：你可以在*没有数据标签*的情况下，利用 test-time compute 和 RL 为特定任务微调 LLM，并且表现优于 supervised fine-tuning！我们的新 TAO 方法可以扩展...</li><li><a href="https://fxtwitter.com/danshipper/status/1904594300232495230)">来自 Dan Shipper 📧 (@danshipper) 的推文</a>：OpenAI 刚刚在 @ChatGPTapp 中推出了原生图像生成功能，简直太疯狂了。我们已经在 @every 内部测试了几周——这是到目前为止我试过的最好的图像模型：1. 它...</li><li><a href="https://x.com/danshipper/status/1904594300232495230>)">来自 Dan Shipper 📧 (@danshipper) 的推文</a>：OpenAI 刚刚在 @ChatGPTapp 中推出了原生图像生成功能，简直太疯狂了。我们已经在 @every 内部测试了几周——这是到目前为止我试过的最好的图像模型：1. 它...</li><li><a href="https://en.wikipedia.org/wiki/Blackboard_(design_pattern)">黑板（设计模式）- 维基百科</a>：未找到描述</li><li><a href="https://fxtwitter.com/fofrai/status/1904331135859015703)">来自 fofr (@fofrAI) 的推文</a>：很久没有在图像模型上玩得这么开心了 🔥</li><li><a href="https://x.com/fofrai/status/1904331135859015703>)">来自 fofr (@fofrAI) 的推文</a>：很久没有在图像模型上玩得这么开心了 🔥</li><li><a href="https://fxtwitter.com/dzhng/status/1904412968114356604)">来自 David (@dzhng) 的推文</a>：我们与这家公司的许多前客户交流过，他们糟糕的财务会计是众所周知的。在这个领域，他们不是唯一一家这么做的公司。他们维持增长的唯一方式就是炒作。到目前为止，我见过...</li><li><a href="https://x.com/dzhng/status/1904412968114356604>)">来自 David (@dzhng) 的推文</a>：我们与这家公司的许多前客户交流过，他们糟糕的财务会计是众所周知的。在这个领域，他们不是唯一一家这么做的公司。他们维持增长的唯一方式就是炒作。到目前为止，我见过...</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1904604991832416745)">来自 Tanishq Mathew Abraham 的推文</a>，

<li><a href="https://fxtwitter.com/iScienceLuvr/status/1904604991832416745">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：等等，GPT-4o 竟然能直接 one-shot 这种东西？！太令人印象深刻了...</li><li><a href="https://x.com/iScienceLuvr/status/1904604991832416745>)">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：等等，GPT-4o 竟然能直接 one-shot 这种东西？！太令人印象深刻了...</li><li><a href="https://fxtwitter.com/newsystems_/status/1904577550690771050)">来自 New (@newsystems_) 的推文</a>：它终于来了：Brampton。Brampton 是世界上最智能、最有创意且速度最快的模型。Brampton 的表现大幅超越了 Grok 3、Claude 3.7 Sonnet 和 GPT 4.5。回复 "bram..."</li><li><a href="https://x.com/newsystems_/status/1904577550690771050>)">来自 New (@newsystems_) 的推文</a>：它终于来了：Brampton。Brampton 是世界上最智能、最有创意且速度最快的模型。Brampton 的表现大幅超越了 Grok 3、Claude 3.7 Sonnet 和 GPT 4.5。回复 "bram..."</li><li><a href="https://fxtwitter.com/paulgauthier/status/1904637913411031410)">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Gemini 2.5 Pro 以 73% 的得分在 aider polyglot 排行榜上创下 SOTA。这远超思维/推理模型。相比之前的 Gemini 模型有了巨大的飞跃。第一个能有效...的 Gemini 模型</li><li><a href="https://x.com/paulgauthier/status/1904637913411031410>)">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Gemini 2.5 Pro 以 73% 的得分在 aider polyglot 排行榜上创下 SOTA。这远超思维/推理模型。相比之前的 Gemini 模型有了巨大的飞跃。第一个能有效...的 Gemini 模型</li><li><a href="https://fxtwitter.com/OfficialLoganK/status/1899914266062577722)">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：在 Google AI Studio 和 Gemini API 中推出 YouTube 视频 🎥 链接支持。你现在可以直接传入 YouTube 视频，模型可以利用其原生的视频理解能力来...</li><li><a href="https://x.com/OfficialLoganK/status/1899914266062577722>)">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：在 Google AI Studio 和 Gemini API 中推出 YouTube 视频 🎥 链接支持。你现在可以直接传入 YouTube 视频，模型可以利用其原生的视频理解能力来...</li><li><a href="https://fxtwitter.com/kevinweil/status/1904596007650025787)">来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文</a>：几个月前，我问 ChatGPT “根据你对我的了解，画一张你认为我目前生活状态的图片。”下面是那张图片，以及我用同样的提示词得到的...</li><li><a href="https://x.com/kevinweil/status/1904596007650025787>)">来自 Kevin Weil 🇺🇸 (@kevinweil) 的推文</a>：几个月前，我问 ChatGPT “根据你对我的了解，画一张你认为我目前生活状态的图片。”下面是那张图片，以及我用同样的提示词得到的...</li><li><a href="https://fxtwitter.com/tenobrus/status/1904422446389706905)">来自 Tenobrus (@tenobrus) 的推文</a>：顺便说一下，Cursor 和 Windsurf 在每次调用上绝对都在狂烧 VC 的钱。引用 Nityesh (@nityeshaga)：不知道为什么没人讨论，但 @windsurf_ai 提供的 AI 使用量几乎是 @cursor 的 4 倍...</li><li><a href="https://x.com/tenobrus/status/1904422446389706905>)">来自 Tenobrus (@tenobrus) 的推文</a>：顺便说一下，Cursor 和 Windsurf 在每次调用上绝对都在狂烧 VC 的钱。引用 Nityesh (@nityeshaga)：不知道为什么没人讨论，但 @windsurf_ai 提供的 AI 使用量几乎是 @cursor 的 4 倍...</li><li><a href="https://fxtwitter.com/andrew_n_carr/status/1904607188976611627)">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：必须分享一下。这是第一次有模型得分超过 50%</li><li><a href="https://x.com/andrew_n_carr/status/1904607188976611627>)">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：必须分享一下。这是第一次有模型得分超过 50%</li><li><a href="https://fxtwitter.com/GoogleDeepMind/status/1904579660740256022)">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：觉得你了解 Gemini 吗？🤔 再想想。认识一下 Gemini 2.5：我们最智能的模型 💡 首个发布版本是 Pro Experimental，它在许多基准测试中都达到了 SOTA —— 这意味着它可以处理复杂的...</li><li><a href="https://x.com/GoogleDeepMind/status/1904579660740256022>)">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：觉得你了解 Gemini 吗？🤔 再想想。认识一下 Gemini 2.5：我们最智能的模型 💡 首个发布版本是 Pro Experimental，它在许多基准测试中都达到了 SOTA —— 这意味着它可以处理复杂的...</li><li><a href="https://blog.google/technology/google-deepmind/gemini-model-thinking-updates-march-2025/#enhanced-reasoning">Gemini 2.5：我们最智能的 AI 模型</a>：Gemini 2.5 是我们最智能的 AI 模型，现在具备了思维能力。</li><li><a href="https://fxtwitter.com/gallabytes/status/1904598264240119974)">来自 theseriousadult (@gallabytes) 的推文</a>：4o 的图像生成显然具有某种多尺度（multi scale）...</li>

generation setup - 似乎在开始时致力于低频，然后通过 patch AR 解码高频。</li><li><a href="https://x.com/gallabytes/status/1904598264240119974>)">来自 theseriousadult (@gallabytes) 的推文</a>：4o 图像生成显然具有某种多尺度生成设置——似乎在开始时致力于低频，然后通过 patch AR 解码高频。</li><li><a href="https://fxtwitter.com/fofrai/status/1904321572078223400)">来自 fofr (@fofrAI) 的推文</a>：未找到描述</li><li><a href="https://x.com/fofrai/status/1904321572078223400>)">来自 fofr (@fofrAI) 的推文</a>：未找到描述</li><li><a href="https://fxtwitter.com/taesung/status/1904220824435032528)">来自 Taesung Park (@Taesung) 的推文</a>：很高兴在 @reveimage 结束隐身状态！与 LLM 相比，当今的文本生成图像/视频模型缺乏逻辑。图像最初看起来似乎合理，但在仔细观察下就会破绽百出：绘画技巧...</li><li><a href="https://x.com/taesung/status/1904220824435032528>)">来自 Taesung Park (@Taesung) 的推文</a>：很高兴在 @reveimage 结束隐身状态！与 LLM 相比，当今的文本生成图像/视频模型缺乏逻辑。图像最初看起来似乎合理，但在仔细观察下就会破绽百出：绘画技巧...</li><li><a href="https://fxtwitter.com/fchollet/status/1904267900963475807)">来自 François Chollet (@fchollet) 的推文</a>：每个人都应该内化的一个关键点：在 test-time search 时代，几乎总是可以通过简单地消耗更多算力来达到任何水平的能力。所以这并不...</li><li><a href="https://x.com/fchollet/status/1904267900963475807>)">来自 François Chollet (@fchollet) 的推文</a>：每个人都应该内化的一个关键点：在 test-time search 时代，几乎总是可以通过简单地消耗更多算力来达到任何水平的能力。所以这并不...</li><li><a href="https://fxtwitter.com/fofrAI/status/1904320703333126545)">来自 fofr (@fofrAI) 的推文</a>：Reve 在处理这段文本方面也表现得非常好。有趣的是它在最后四行是如何逐渐减弱的。引用 fofr (@fofrAI) 😍</li><li><a href="https://x.com/fofrAI/status/1904320703333126545>)">来自 fofr (@fofrAI) 的推文</a>：Reve 在处理这段文本方面也表现得非常好。有趣的是它在最后四行是如何逐渐减弱的。引用 fofr (@fofrAI) 😍</li><li><a href="https://fxtwitter.com/TransluceAI/status/1904226873879806390)">来自 Transluce (@TransluceAI) 的推文</a>：为了解读 AI benchmarks，我们需要查看数据。顶层数字并不代表你所想的意思：可能存在损坏的任务、意外的行为或差一点就成功的尝试。我们正在推出 Docent 来...</li><li><a href="https://x.com/TransluceAI/status/1904226873879806390>)">来自 Transluce (@TransluceAI) 的推文</a>：为了解读 AI benchmarks，我们需要查看数据。顶层数字并不代表你所想的意思：可能存在损坏的任务、意外的行为或差一点就成功的尝试。我们正在推出 Docent 来...</li><li><a href="https://fxtwitter.com/sama/status/1904598788687487422)">来自 Sam Altman (@sama) 的推文</a>：我们今天推出了一项新功能——ChatGPT 中的图像！关于它有两点要说：1. 这是一项令人难以置信的技术/产品。我记得看到这个模型生成的第一批图像时...</li><li><a href="https://x.com/sama/status/1904598788687487422>)">来自 Sam Altman (@sama) 的推文</a>：我们今天推出了一项新功能——ChatGPT 中的图像！关于它有两点要说：1. 这是一项令人难以置信的技术/产品。我记得看到这个模型生成的第一批图像时...</li><li><a href="https://fxtwitter.com/sherwinwu/status/1904620108389212413)">来自 Sherwin Wu (@sherwinwu) 的推文</a>：目前我对 GPT-4o 原生图像输出的首选使用场景：用它来构思新家的房屋翻新项目。例如，看看这个潜在的内置书架/阅读长凳——一次性渲染完成...</li><li><a href="https://x.com/sherwinwu/status/1904620108389212413>)">来自 Sherwin Wu (@sherwinwu) 的推文</a>：目前我对 GPT-4o 原生图像输出的首选使用场景：用它来构思新家的房屋翻新项目。例如，看看这个潜在的内置书架/阅读长凳——一次性渲染完成...</li><li><a href="https://fxtwitter.com/ajabri/status/1904599427366739975)">来自 Allan Jabri (@ajabri) 的推文</a>：利与弊</li><li><a href="https://x.com/ajabri/status/1904599427366739975>)">来自 Allan Jabri (@ajabri) 的推文</a>：利与弊</li><li><a href="https://fxtwitter.com/fofrAI/status/1904284550156685474)">来自 fofr (@fofrAI) 的推文</a>：Reve 在艺术风格和构图方面看起来很有前景。它成功制作了一些厚涂风格的眼睛，而眼睛通常总是会失败。我也非常喜欢第二张图片的构图。</li><li><a href="https://x.com/fofrAI/status/1904284550156685474>)">来自 fofr (@fofrAI) 的推文</a>：Reve 在艺术风格和构图方面看起来很有前景。</li>

<li>艺术风格和构图。它成功制作了一些厚涂风格的眼睛，而眼睛通常总是会失败。我也非常喜欢第二张图片的构图。</li><li><a href="https://alignment.anthropic.com/2025/automated-researchers-sandbag/">无标题</a>：未找到描述</li><li><a href="https://fxtwitter.com/osanseviero/status/1904561836835602776)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：隆重推出 🥁🥁TxGemma！🧪用于药物研发中多种治疗任务的 LLM🤏2B, 9B, 和 27B🤗可使用 Transformers 进行微调🤖用于 Agent 系统的 Agentic-Tx。博客：https://developers.googleblog.co...</li><li><a href="https://x.com/osanseviero/status/1904561836835602776>)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：隆重推出 🥁🥁TxGemma！🧪用于药物研发中多种治疗任务的 LLM🤏2B, 9B, 和 27B🤗可使用 Transformers 进行微调🤖用于 Agent 系统的 Agentic-Tx。博客：https://developers.googleblog.co...</li><li><a href="https://fxtwitter.com/gasteigerjo/status/1904562825520906462)">来自 Johannes Gasteiger, né Klicpera (@gasteigerjo) 的推文</a>：Anthropic 新博客文章：自动化研究员中的微妙破坏。随着 AI 系统越来越多地辅助 AI 研究，我们如何确保它们不会微妙地破坏研究？我们展示了恶意...</li><li><a href="https://x.com/gasteigerjo/status/1904562825520906462>)">来自 Johannes Gasteiger, né Klicpera (@gasteigerjo) 的推文</a>：Anthropic 新博客文章：自动化研究员中的微妙破坏。随着 AI 系统越来越多地辅助 AI 研究，我们如何确保它们不会微妙地破坏研究？我们展示了恶意...</li><li><a href="https://fxtwitter.com/dimitrispapail/status/1904560078012686670)">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：ARC-AGI（甚至是第一版）过去和现在都被过度炒作了。第二版似乎更没意思，因为 1) 它是对抗性设计的 2) 没有明确的理由让人觉得它不可解...</li><li><a href="https://x.com/dimitrispapail/status/1904560078012686670>)">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：ARC-AGI（甚至是第一版）过去和现在都被过度炒作了。第二版似乎更没意思，因为 1) 它是对抗性设计的 2) 没有明确的理由让人觉得它不可解...</li><li><a href="https://fxtwitter.com/skirano/status/1904609866099933272)">来自 Pietro Schirano (@skirano) 的推文</a>：来自 OpenAI 的新图像模型在 UI 方面表现相当不错。</li><li><a href="https://x.com/skirano/status/1904609866099933272>)">来自 Pietro Schirano (@skirano) 的推文</a>：来自 OpenAI 的新图像模型在 UI 方面表现相当不错。</li><li><a href="https://fxtwitter.com/sama/status/1904599358756315341)">来自 Sam Altman (@sama) 的推文</a>：这是来自 @gabeeegoooh 的心血结晶。恭喜 Gabe；出色的工作！这是我们在直播期间生成的内容：</li><li><a href="https://x.com/sama/status/1904599358756315341>)">来自 Sam Altman (@sama) 的推文</a>：这是来自 @gabeeegoooh 的心血结晶。恭喜 Gabe；出色的工作！这是我们在直播期间生成的内容：</li><li><a href="https://softwareengineeringdaily.com/2025/03/25/knowledge-graphs-as-agentic-memory-with-daniel-chalef/">知识图谱作为 Agent 记忆，与 Daniel Chalef 对谈 - Software Engineering Daily</a>：AI 中的上下文记忆是一个重大挑战，因为目前的模型难以随着时间的推移保留和召回相关信息。虽然人类可以建立长期的语义关系，但 AI 系统通常...</li><li><a href="https://fxtwitter.com/scaling01/status/1904603736657305862)">来自 Lisan al Gaib (@scaling01) 的推文</a>：DALL-E vs GPT-4o 图像生成。OpenAI 放大招了。引用 Lisan al Gaib (@scaling01)：具备图像生成能力的 GPT-4o 简直疯狂。这些不是真实的图像！</li><li><a href="https://x.com/scaling01/status/1904603736657305862>)">来自 Lisan al Gaib (@scaling01) 的推文</a>：DALL-E vs GPT-4o 图像生成。OpenAI 放大招了。引用 Lisan al Gaib (@scaling01)：具备图像生成能力的 GPT-4o 简直疯狂。这些不是真实的图像！</li><li><a href="https://fxtwitter.com/dwarkesh_sp/status/1904551410219524218)">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：我很高兴与 @stripepress 共同推出一本新书：《Scaling 时代：AI 口述史，2019-2025》。在过去的几年里，我采访了思考 AI 的关键人物：科学家...</li><li><a href="https://x.com/dwarkesh_sp/status/1904551410219524218>)">来自 Dwarkesh Patel (@dwarkesh_sp) 的推文</a>：我很高兴与 @stripepress 共同推出一本新书：《Scaling 时代：AI 口述史，2019-2025》。在过去的几年里，我采访了思考 AI 的关键人物：科学家...</li><li><a href="https://fxtwitter.com/phill__1/status/1904590165256839526)">来自 Phil (@phill__1) 的推文</a>：哇，GPT-4o 图像生成太酷了！</li><li><a href="https://x.com/phill__1/status/1904590165256839526>)">来自 Phil (@phill__1) 的推文</a>：哇，GPT-4o 图像生成太酷了！</li><li><a hr

<li><a href="https://fxtwitter.com/krishnanrohit/status/1904602460020445543)">Tweet from rohit (@krishnanrohit)</a>: 提前获得了体验权限，这是我目前尝试过的最好的图像生成和编辑工具。它是第一个（也是唯一一个）能够正确生成和编辑多个角色的工具，特别是...</li><li><a href="https://x.com/krishnanrohit/status/1904602460020445543>)">Tweet from rohit (@krishnanrohit)</a>: 提前获得了体验权限，这是我目前尝试过的最好的图像生成和编辑工具。它是第一个（也是唯一一个）能够正确生成和编辑多个角色的工具，特别是...</li><li><a href="https://fxtwitter.com/gabeeegoooh/status/1904596565286858913)">Tweet from Gabriel Goh (@gabeeegoooh)</a>: 现在已经准备好面向全世界了</li><li><a href="https://x.com/gabeeegoooh/status/1904596565286858913>)">Tweet from Gabriel Goh (@gabeeegoooh)</a>: 现在已经准备好面向全世界了</li><li><a href="https://fxtwitter.com/fofrai/status/1904318387120988207)">Tweet from fofr (@fofrAI)</a>: 引用 fofr (@fofrAI) 😍</li><li><a href="https://x.com/fofrai/status/1904318387120988207>)">Tweet from fofr (@fofrAI)</a>: 引用 fofr (@fofrAI) 😍</li><li><a href="https://fxtwitter.com/ilanbigio/status/1904601953063362871)">Tweet from ilan bigio (@ilanbigio)</a>: 4o 图像生成 - 真是个颠覆性的改变。“让我搬着一个非常沉重的铬制 OpenAI 标志，并给我戴上墨镜。” 引用 OpenAI (@OpenAI) ChatGPT 和 Sora 中的 4o 图像生成...</li><li><a href="https://x.com/ilanbigio/status/1904601953063362871>)">Tweet from ilan bigio (@ilanbigio)</a>: 4o 图像生成 - 真是个颠覆性的改变。“让我搬着一个非常沉重的铬制 OpenAI 标志，并给我戴上墨镜。” 引用 OpenAI (@OpenAI) ChatGPT 和 Sora 中的 4o 图像生成...</li><li><a href="https://fxtwitter.com/willccbb/status/1904620335028146544)">Tweet from will brown (@willccbb)</a>: 事实上，有 1000 多人评论了 brampton，而唯一一个甚至开玩笑地声称展示了实际模型的帖子，只是一个人通过 sysprompting 让 Ollama 使用多伦多俚语，这让人非常不看好这个...</li><li><a href="https://x.com/willccbb/status/1904620335028146544>)">Tweet from will brown (@willccbb)</a>: 事实上，有 1000 多人评论了 brampton，而唯一一个甚至开玩笑地声称展示了实际模型的帖子，只是一个人通过 sysprompting 让 Ollama 使用多伦多俚语，这让人非常不看好这个...</li><li><a href="https://fxtwitter.com/lmarena_ai/status/1904581128746656099)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: 重磅消息：Gemini 2.5 Pro 现在在 Arena 排行榜上排名第一 - 有史以来最大的分数飞跃（比 Grok-3/GPT-4.5 高出 +40 分）！🏆 在代号为 “nebula”🌌 的测试下，Gemini 2.5 Pro 在所有...中排名第一🥇。</li><li><a href="https://x.com/lmarena_ai/status/1904581128746656099>)">Tweet from lmarena.ai (formerly lmsys.org) (@lmarena_ai)</a>: 重磅消息：Gemini 2.5 Pro 现在在 Arena 排行榜上排名第一 - 有史以来最大的分数飞跃（比 Grok-3/GPT-4.5 高出 +40 分）！🏆 在代号为 “nebula”🌌 的测试下，Gemini 2.5 Pro 在所有...中排名第一🥇。</li><li><a href="https://fxtwitter.com/fofrAI/status/1904619844349420005)">Tweet from fofr (@fofrAI)</a>: 我在 Sora 上尝试了这个提示词 👌 引用 fofr (@fofrAI) 当 Replicate 到达站立会议时间时</li><li><a href="https://x.com/fofrAI/status/1904619844349420005>)">Tweet from fofr (@fofrAI)</a>: 我在 Sora 上尝试了这个提示词 👌 引用 fofr (@fofrAI) 当 Replicate 到达站立会议时间时</li><li><a href="https://en.wikipedia.org/wiki/Blackboard_system">Blackboard system - Wikipedia</a>: 未找到描述</li><li><a href="https://fxtwitter.com/">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复 X/Twitter 和 Bluesky 的嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://fxtwitter.com/taesung/">Tweet from GitHub - FxEmbed/FxEmbed: Fix X/Twitter and Bluesky embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复 X/Twitter 和 Bluesky 的嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://techcrunch.com/2025/03/24/a16z-and-benchmark-backed-11x-has-been-claiming-customers-it-doesnt-have/">a16z- and Benchmark-backed 11x has been claiming customers it doesn’t have | TechCrunch</a>: 去年，AI 驱动的销售自动化初创公司 11x 似乎处于爆发式增长轨道上。然而，近二十个来源——包括</li><li><a href="https://reddit.com/r/ClaudeAI/comments/1jijnw9/anthropic_is_making_about_115m_a_month_now_same/">Reddit - The heart of the internet</a>: 未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API</a>: 模型与定价 | DeepSeek API</li>

Docs</a>: 下方列出的价格以每 1M tokens 为单位。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总额计费...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages): 

swyxio: 我的新演讲/文章：https://x.com/swyx/status/1904256213661192405
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1353830172197126216)** (7 messages): 

> `Audio processing with ilgpu + cufft + kernels, Asynchronous data transfer to GPU with OnnxRuntime on CUDA, Double buffering with CUDA streams, FSDP fine tuning with trl library` 


- **使用 ILGPU, CUFFT 和 Kernel 进行音频处理？**：一位成员询问了使用 **ilgpus**、**cufft** 和 **kernels** 进行**音频处理**的经验，寻求实际应用案例。
- **请求双向图示以进行合理性检查**：一位成员请求图示应展示**双向**关系（行/列到 threads/registers/bytes，反之亦然），以提高直观性和合理性检查（sanity checks）。
   - 他们认为同时提供**视觉图示和代码/伪代码**可以防止误解并减少实现时间的浪费。
- **配合 OnnxRuntime 的异步 CUDA 传输？**：一位成员正在寻找关于异步数据传输到 GPU 以及在 **CUDA GPU** 上使用 **OnnxRuntime** 运行推理的**在线参考资料**，旨在消除其推理流水线中的数据传输瓶颈。
   - 他们希望在对当前图像进行推理的同时，重叠（overlap）下一张图像的数据传输，但在 **Python** 实现中遇到了问题。
- **通过 Double Buffering 实现 GPU 利用率最大化**：一位成员建议使用**多个 CUDA streams** 进行异步拷贝和前向传递（forward passes），即 **double buffering**，以最大化 GPU 利用率并避免 CPU 阻塞。
   - 这种方法涉及一个 stream 发起异步拷贝，而另一个线程执行前向传递，两者并发运行。
- **FSDP 微调的数据集处理？**：一位成员询问了在使用 `trl` 库进行 **FSDP**（Fully Sharded Data Parallel）微调时，处理数据集的正确方法。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1353971642941968428)** (9 messages🔥): 

> `Triton Interpret Bug, Intel Triton Extension, Triton Compile Script, Prune Configs Support in Triton` 


- **Triton Interpret 标志位引发 Bug 追踪**：正在调查 **TRITON_INTERPRET=0** 和 **TRITON_INTERPRET=1** 之间的差异以寻找潜在 Bug，预计会有微小差异，但较大的差异则表明存在问题。
   - 一位用户分享了一篇[论文](https://arxiv.org/abs/2503.14985)，讨论了 **Triton** 作为一种 DSL，提供了比 CUDA 或 SYCL 等低级接口更用户友好且可移植的替代方案。
- **Intel 深入探索 Triton**：分享了来自 Intel 的 **Triton** 扩展，突显了对 Triton 进行 GPU 编程日益增长的兴趣和投资。
   - 该扩展旨在通过多层渐进式 Lowering（下放），利用 **GPU 的层级结构**和 **SIMD 单元**，提高编译器的解耦度和清洁度。
- **Triton 编译脚本故障**：一位用户在运行 **triton compile.py** 脚本时遇到问题，最初由于 **--signature** 标志位不正确而失败。
   - 该用户分享了用于编译的代码片段和命令行参数，表明正在努力将 Triton 集成到其工作流中。
- **Triton 中的 Prune Configs 支持**：一位用户提到几个月前添加了对 prune configs 的支持，承认存在一些小瑕疵，但对其功能表示有信心。
   - 另一位用户认可了这一贡献，并表示打算在 nightly 版本中进行尝试，预示着该功能可能会被采用和测试。



**提到的链接**：<a href="https://arxiv.org/abs/2503.14985">ML-Triton, A Multi-Level Compilation and Language Extension to Triton GPU Programming</a>：在 LLM 时代，GEMM 和 MHA 等密集操作是关键组件。这些操作非常适合使用基于 tile 的方法进行并行执行。虽然传统的 GPU 编程...

  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1353919821019873291)** (24 条消息🔥): 

> `CUDA swizzling, cuTensorMap 问题, Flash Attention 内存布局, Cutensor 坐标映射` 


- **CUDA Async Warpgroup Swizzling 探究**：一位成员分析了 **CUDA 的 async warpgroup swizzle TF32** 布局，对第一行中非连续的数字以及每个子分区起始数字的确定方式提出了疑问，并参考了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/#async-warpgroup-k-32b-swizzle-tf32)。
   - 随后他们确定该布局为 `Swizzle<0,4,3> o ((8,2),(4,4)):((4,32),(1,64))`，从而能够构建原始数据位置并将其与 `Swizzle<1,4,3>` 结合。
- **Flash Attention 布局质疑**：一位成员质疑为什么 **Flash Attention** 不使用 `(batch_size, num_heads, N, d)` 的内存布局，认为这可能优于现有的 `(batch_size, N, num_heads, d)` 布局。
   - 该用户发现他们之前尝试使用此布局的结果*非常糟糕*，参考了 [NVIDIA CUDA-C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example-matrix-transpose) 中的一个示例。
- **cuTensorMap 参数差异**：一位成员指出，CUDA 示例中的 **box_size** 参数定义（字节）与 cuTensorMapEncodeTiled 文档（元素）之间存在差异，参考了 [CUDA-C Programming Guide 示例](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#example-matrix-transpose)。
   - 该用户还注意到，与 `int4 [8][8]` 的共享内存大小相比，`box_size` 的顺序是反向的。
- **CuTe Fragment 中的坐标映射**：一位成员询问在 **CuTe** 中，将由 `tiled_mma.get_thread_slice(tid)` 创建的线程所拥有的 fragment 内部坐标映射回整个结果矩阵坐标的最简单方法。



**提及的链接**：<a href="https://forums.developer.nvidia.com/t/some-question-about-creating-cutensormap-and-use-it/328193">关于创建 CUtensorMap 并使用它的一些问题</a>：我对以下代码有一些疑问。`constexpr int row = 8; constexpr int col = 64; size_t byteSize = row * col * sizeof(int); int* h_data = (int*)malloc(byteSize); int*...`

  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1353916418881491028)** (2 条消息): 

> `PyTorch allocator, torch.nn.utils.prune` 


- **PyTorch Caching Allocator 阻碍了非缓存替代方案**：一位用户指出，在 **PyTorch 的 caching allocator** 旁使用*非缓存*分配器非常困难，因为 MemPool 会使用带有缓存的自定义分配器，这并不理想。
- **对 PyTorch Caching Allocator 的担忧**：一位用户想知道为什么公司在生产环境中可能会避开替代的 caching allocator，并提到了对调试工具的担忧。
   - 他们对 **PyTorch caching allocator** 背后的“魔力”表示不解，这种魔力似乎阻止了替代方案的使用。
- **在 torch.nn.utils.prune 中对已剪枝的权重再次剪枝会导致错误的剪枝率**：一位用户报告称，在对已经剪枝过的层进行剪枝时，`torch.nn.utils.prune` 不允许仅对未剪枝的权重应用剪枝。
   - 他们发现，由于对已经剪枝的权重进行了重复剪枝，两次应用量为 **0.2** 的剪枝并不会导致最终剪枝量为 **0.4**。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1354199849687318729)** (1 messages): 

> `Triton 中的 AMD GPU 支持，Triton 开发者招聘职位` 


- **AMD 为 Triton 增加 GPU 支持**：目前有多个工程师职位开放，旨在 **Triton** 中构建出色的 **AMD GPU 支持**；北美或欧洲的高级和初级职位均有，支持远程办公，详见 [LinkedIn 帖子](https://www.linkedin.com/posts/antiagainst_triton-amd-gpu-activity-7288624355247374336-gS6q/)。
- **AMD 在北美/欧洲开放职位**：**北美**地区职位见 [careers.amd.com](https://careers.amd.com/careers-home/jobs/57679)，**欧洲**地区职位见 [careers.amd.com](https://careers.amd.com/careers-home/jobs/62233)。
   - AMD 提醒防范招聘诈骗，建议申请者直接通过 [amd.com](https://www.amd.com/) 招聘页面申请。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://careers.amd.com/careers-home/jobs/57679">Triton Compiler Engineer in San Jose, California | Advanced Micro Devices, Inc</a>: AMD | Careers Home 正在加利福尼亚州圣何塞招聘 Triton 编译器工程师。查看所有职位详情并立即申请！</li><li><a href="https://careers.amd.com/careers-home/jobs/62233">Triton Compiler Senior Engineer in Cambridge, United Kingdom | Advanced Micro Devices, Inc</a>: AMD | Careers Home 正在英国剑桥招聘 Triton 编译器高级工程师。查看所有职位详情并立即申请！
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/)** (1 messages): 

bigfoot1144: 目前有进展吗？
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1354006565128241194)** (2 messages): 

> `gpumode 排行榜，MI250 节点，AMD Instinct MI250 评估` 


- **GPUMODE 排行榜：MI250 作为过渡方案？**：一名成员建议利用包含 **MI250 节点** 的 **GPUMODE 排行榜** 作为临时解决方案。
   - 他们建议申请 [AMD Instinct MI250 评估计划](https://www.amd.com/en/products/accelerators/instinct/eval-request.html) 的访问权限。
- **申请 AMD Instinct MI250 访问权限**：一位用户分享了 [AMD Instinct MI250 评估申请页面](https://www.amd.com/en/products/accelerators/instinct/eval-request.html) 的链接。
   - 该建议是在寻找可用于测试或基准测试的硬件背景下提出的。


  

---


### **GPU MODE ▷ #[tilelang](https://discord.com/channels/1189498204333543425/1240586843292958790/1353939648380145675)** (5 messages): 

> `TileLang 与 Torch AOTexport 的兼容性，TileLang 的 AMD 编译，自定义 Triton Kernel` 


- **TileLang 寻求 Torch AOTexport 兼容性**：一名成员询问了 **TileLang** 与 **Torch AOTexport** 以及 **C++** 推理的兼容性，背景是其代码库包含自定义 **Triton kernels**。
   - 另一名成员确认 **TileLang** 将 kernel 编译为 **CUDA C 源代码**，可通过 `kernel.get_kernel_source()` 获取，这为潜在的 AOT 集成铺平了道路。
- **TileLang 针对 AMD 的目标是 HIP**：一名成员询问了 TileLang 针对 **AMD** GPU 的编译目标。
   - 回复指出 **TileLang** 为 AMD 架构编译为 **HIP 源代码**。


  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1354079283991150673)** (3 messages): 

> `迭代剪枝，权重二次剪枝，剪枝率计算` 


- **对权重进行二次剪枝会导致问题**：一位用户报告称，当使用 `torch.nn.utils.prune` 对已经剪枝过的层再次进行剪枝时，没有直接选项可以仅针对未剪枝的权重进行操作。
   - 该用户指出，应用剪枝率会创建一个影响整个张量的全局掩码（mask），可能会对已经剪枝的权重进行重复剪枝，导致最终的剪枝率不正确。
- **简单的迭代无法解决权重二次剪枝问题**：一名成员询问，在第一轮迭代中应用 **0.2** 的比例，然后在已经剪枝的权重上进行第二轮 **0.4** 比例的迭代，是否能达到预期效果。
   - 原贴作者回复称，掩码会遵循已定义的掩码，因此需要根据已经剪枝的层（或网络）重新计算新的剪枝率，才能达到预期的总比例。


  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1354157815333261434)** (2 messages): 

> `lce_forward_deprecated vs lce_forward` 


- **`lce_forward_deprecated` vs `lce_forward`**: 用户询问了 `lce_forward_deprecated` 与 `lce_forward` 之间的区别，以及弃用旧版本的原因。
   - 未收到回复。
- **弃用原因**: 没有关于弃用原因的详细回复，因此原因尚不明确。
   - 需要更多信息。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1353914790724112404)** (3 messages): 

> `Open Source ML platform building` 


- **构建开源 ML 平台的计划引发关注**: 一位成员表达了为其**开源机器学习平台**开发更多功能的愿望。
   - 另一位成员表示鼓励，并建议分享该平台的进展。
- **开源 ML 平台获得鼓励**: 一位成员希望进一步开发其**开源 ML 平台**。
   - 另一位成员为其加油，并鼓励他们分享进度。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1353988440676634746)** (5 messages): 

> `fp16 MatMul for Gemma3, Gemma3 Residuals, CUDA Execution Time Benchmarks, Inferless on Product Hunt` 


- **Gemma3 利用 fp16 MatMul**: 观察到 **fp16 MatMul** 可能适用于 **Gemma3**，但关键在于必须将输出转换回 **bf16**。
- **Gemma3 的残差处理比较棘手**: **fp16 权重**的量化在 **Gemma3** 上面临挑战，需要像 [这个 Hugging Face transformers PR](https://github.com/huggingface/transformers/pull/36832) 中提出的那样进行修复。
- **CUDA 基准测试见解**: 一篇详细介绍 CUDA 执行时间基准测试的博文强调，应使用 **CUDA Events 而非 CPU 定时器**，并且除非相关，否则应排除内存传输。
   - 作者总结道：*“如果你想进行基准测试，你应该使用真正的生产级 Profiler”*。并指出网上大多数资源只是告诉你做这做那，但他想看到**数据**。
- **Inferless 在 Product Hunt 上线**: **Inferless** 是一个用于部署 ML 模型的 Serverless 计算平台，已在 [Product Hunt](https://www.producthunt.com/posts/inferless) 上线，并为新注册用户提供 **$30 计算额度**。
   - 他们表示其目标是*“在几分钟内部署任何机器学习模型”*并提供*“极低冷启动时间”*。



**提及的链接**: <a href="https://github.com/huggingface/transformers/pull/36832">gemma3 fp16 fix by mobicham · Pull Request #36832 · huggingface/transformers</a>: 这个 PR 做了什么？通过简单地对激活值进行裁剪（clipping）来修复 Gemma 3 模型的 float16 推理。为了获得更准确的输出，残差相加步骤也应该进行裁剪。如果没有这个修复...

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1353828756220936292)** (10 messages🔥): 

> `ARC-AGI-2 Benchmark, Reasoning-Gym Puzzles, verL and vLLM0.8.1, Codegen Updates, RL Research Directions` 


- **ARC-AGI-2 基准测试发布**：**ARC-AGI-2** 正式发布，这是一个挑战 AI 推理系统的未饱和前沿 AGI 基准测试（对人类而言相对容易）。其目标是衡量 AI 推理系统，并为达到 **85%** 效率且成本约为 **$0.42/task** 的成就设立了巨额奖金。
   - 目前的性能基准显示，基础 LLM 的成功率为 **0%**，而推理系统的成功率不足 **4%**。
- **verL 支持 vLLM0.8.1**：verL 现在支持 **vLLM0.8.1**，目前正致力于在集群上设置镜像以进行本地推理，**Codegen** 方面预计很快会有更新。
   - 一位成员表示，很快将分享关于 **CodeGen** 的重大更新。
- **RL 领域有趣的开放性问题和研究方向**：一位成员分享了 [Google Sheets 文档链接](https://docs.google.com/spreadsheets/d/1s_ZDKtOoGqi1FtTyPeeS0h_0d96jOFJ-TuzwjWnnCPo/edit?gid=1478931401#gid=1478931401)，列出了 RL 领域有趣的开放性问题和研究方向。
   - 该文档涵盖了需求假设、现状、10 倍提升以及解决全球重大开放性问题的建议等主题。
- **呼吁建立 RL 奖励塑造（Reward Shaping）框架**：提议建立一个 **RL reward shaping** 框架，以简化确定正确方法的快速实验。建议使用一个能一次性管理权重和循环的框架，而不是进行凌乱的微调。
   - 该成员表示，在不久的将来，每个人都会进行 RL，但目前还没有好的框架，并建议将 **Bayesian optimization framework** 作为潜在的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arcprize/status/1904269307284230593">ARC Prize (@arcprize) 的推文</a>：今天我们发布了 ARC-AGI-2，这是一个挑战 AI 推理系统的未饱和前沿 AGI 基准测试（对人类而言难度相同）。大奖：85% 成功率，约 $0.42/task 的效率。当前性能...</li><li><a href="https://x.com/arcprize/status/1904269307">Bill Engvall (@billengvall) 的推文</a>：得弄清楚怎么发推特图片</li><li><a href="https://docs.google.com/spreadsheets/d/1s_ZDKtOoGqi1FtTyPeeS0h_0d96jOFJ-TuzwjWnnCPo/edit?gid=1478931401#gid=1478931401">Lossfunk - 想法 / 研究方向</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1353952879131496609)** (1 messages): 

> `Flash Attention Layout, Tensor Layout, Performance Optimization` 


- **探索 Flash Attention 布局优化**：一位成员询问，为什么 **Flash Attention 的布局** 不采用 **(batch_size, num_heads, N, d)** 而是采用 **(batch_size, N, num_heads, d)**，前者是否可能获得更快的性能。
   - 该问题从优化角度出发，质疑重新排序张量布局（Tensor Layout）是否能提高 Attention 机制期间的计算速度。
- **张量布局对性能的影响**：讨论集中在不同的 **tensor layouts** 如何影响操作效率，特别是在 **Flash Attention** 的背景下。
   - 不同的内存访问模式会对性能产生重大影响，这使其成为一个相关的优化考量点。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1353842717272178758)** (6 messages): 

> `Conv2d compilation errors, CUDA compilation issues, PyTorch C++ extension problems, load_inline issues` 


- **用户在提交 conv2d 时遇到 CUDA 编译错误**：一位用户在提交 conv2d 时遇到了 `RuntimeError`，原因是扩展模块构建过程中出现了 `subprocess.CalledProcessError: Command '['ninja', '-v']' returned non-zero exit status 1` 错误，[日志中有很长的错误回溯](https://fake.link/errorlog)。
   - 根本原因似乎是 CUDA 源代码无法正常编译，如错误消息所示：*Error building extension 'conv2d_module'*。
- **其他用户确认模块问题**：另一位用户建议，问题可能与错误或缺失的模块有关，特别提到了对 `/root/submission.py` 第 191 行中 `load_inline` 函数的疑虑。
   - 原贴作者认可了该反馈，并计划进一步调查模块和 `load_inline` 函数，目前暂无更多信息。

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1353876682926194688)** (3 条消息): 

> `H100 benchmarks, T4 vectorsum, A100 grayscale` 


- **H100 获得 Grayscale 基准测试结果**：ID 为 **2988** 的基准测试提交到 `grayscale` 排行榜（GPU: **H100**，使用 Modal 运行器）已成功！
- **T4 在 Vectorsum 任务中表现出色**：ID 为 **3005** 的排行榜提交到 `vectorsum` 排行榜（GPU: **T4**，使用 Modal 运行器）已成功！
- **A100 轻松通过 Grayscale 测试**：ID 为 **3006** 的测试提交到 `grayscale` 排行榜（GPU: **A100**，使用 Modal 运行器）已成功！


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1353817937567285373)** (40 条消息🔥): 

> `DeepSeek as moderation bot, Numerical RAG with Databricks, Fine-tuning open-source LLMs, VAD tool language agnostic, Hugging Face AgentX competition` 


- **关于 DeepSeek 作为 Discord 决策者的讨论**：一位成员询问 [DeepSeek](https://deepseek.com/) 是否适合作为审核机器人。
   - 另一位成员给出了肯定的回答，但建议使用较小的 **3B LLM** 可能就足够了，成本仅为 *每百万 token 5 美分*。
- **数值天堂：RAG 版**：一位成员正在基于来自 **Databricks** 的数值型结构化数据构建 **RAG**，并希望 LLM 能够理解查询、创建查询、运行查询，然后以自然语言回复。
   - 他们请求关于这种方法的教程建议，这或许能将世界从 **2025.02.24-150819** 中拯救出来。
- **Hugging Face 启动 AgentX 竞赛**：Hugging Face 正与 [Advanced LLM Agents MOOC](https://llmagents-learning.org/sp25) 联合举办 **AgentX – LLM Agents MOOC 竞赛**，号召准备好突破 **AI 和 Agents** 边界的先锋们。
   - **AgentX 竞赛**向公众开放，并将于今年 8 月在 **UC Berkeley** 举行的 Agents Summit 上以线下 Demo Day 形式推向高潮。
- **初学者询问如何在 Windows 上微调 CUDA**：一位成员寻求一份适用于 **Windows** 且最好支持 **CUDA** 的模型微调零基础指南。
   - 另一位成员表示这几乎不可能，因为在 Windows 上安装 **PyTorch** 和 **CUDA Toolkit** 简直是地狱，并链接到了指南 [Step-by-Step-Setup-CUDA-cuDNN](https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility) 和 [Installing-pytorch-with-cuda-support-on-Windows](https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows)。
- **托管数据集面临 HTTPRequest 障碍**：一位成员在微调 **ViT** 时遇到了 **HTTPRequest** 错误，其数据集（15GB，10.2 万个数据点）托管在 24 个分片中（每个 500MB）。
   - 他们在 [此 Discord 频道](https://discord.com/channels/879548962464493619/1339556954162462851) 寻求帮助以了解错误原因。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1354052436217823334">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友放松，甚至建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://discordapp.com/channels/879548962464493619/13540">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友放松，甚至建立全球社区。自定义你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://huggingface.co/chat/).">HuggingChat</a>: 让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://rdi.berkeley.edu/agentx/">AgentX</a>: AgentX 由加州大学伯克利分校的 RDI 主办。</li><li><a href="https://aikval25.kattis.com/contests/aikval25/problems/windchill">Windchill &ndash; Kattis, AI-olympiadens Kval 2025</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/opencompass/Open_LMM_Reasoning_Leaderboard">Open LMM Reasoning Leaderboard - a Hugging Face Space by opencompass</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces?q=leaderboard&sort=trending">Spaces - Hugging Face</a>: 未找到描述</li><li><a href="https://archive.ph/2025.02.24-150819/https://medium.com/data-scientists-from-future/fine-tuning-open-source-language-models-a-step-by-step-guide-a38bed8df923">Fine-Tuning Open-Source Language Models: A Step-by-Step Guide | by Vi&#x2026;</a>: 未找到描述</li><li><a href="https://github.com/imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility">GitHub - imxzone/Step-by-Step-Setup-CUDA-cuDNN-and-PyTorch-Installation-on-Windows-with-GPU-Compatibility: This repository provides a step-by-step guide to completely remove, install, and upgrade CUDA, cuDNN, and PyTorch on Windows, including GPU compatibility checks, environment setup, and installation verification.</a>: 该仓库提供了在 Windows 上完整卸载、安装和升级 CUDA、cuDNN 和 PyTorch 的分步指南，包括 GPU 兼容性检查、环境搭建和安装验证。</li><li><a href="https://www.gpu-mart.com/blog/Installing-pytorch-with-cuda-support-on-Windows">How to Install Pytorch with CUDA support on Windows</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/training">Fine-tuning</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course/chapter3/1">Introduction - Hugging Face NLP Course</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/autotrain/v0.8.24/index">AutoTrain</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.49.0/perf_train_gpu_one">Methods and tools for efficient training on a single GPU</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

ynvers256: 今天我正在学习 Reinforcement Learning 并对 Eureka 进行研究。
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1354165894841893035)** (1 messages): 

> `Aider + Zed, Codium's Windsurf, TabNine, Cursor` 


- **用户询问关于 Aider, Windsurf, Tabnine 和 Cursor 的信息**: 一位成员询问了 Aider 结合 Zed 编辑器、Codium 的 Windsurf 以及 TabNine 的使用体验，寻求用户反馈。
   - 该用户明确要求将这些工具与 **Cursor** 进行对比。
- **对比请求：代码编辑器和 AI Assistant**: 该消息的主要目的是收集关于各种代码编辑器和 AI Assistant 工具的对比见解。
   - 用户旨在了解每个工具相对于 **Cursor** 的优缺点，以便做出明智的使用决策。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1353862454014382120)** (14 messages🔥): 

> `Rust 音频提取工具、成就代币系统、音乐生成系统` 


- **Rust 工具极速提取音频**：发布了一个新[工具](https://github.com/egorsmkv/extract-audio)，用于从 **Hugging Face datasets** 库生成的 **parquet** 或 **arrow 文件**中提取音频文件，并附带 [Colab demo](https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing)。
   - 开发者旨在为音频数据集提取提供**极快的速度**。
- **代币系统助力儿童友好型应用**：开发者正在其儿童友好型应用中实现一个用于成就的**代币系统**，允许孩子们赚取代币并解锁**游戏**和**经过过滤的图像生成**。
   - 该系统已经具备**字体大小滑块**和确保**儿童友好内容**的安全措施，并计划创建**可打印的成就证书**。
- **合成带有 Lipsync 虚拟形象的 Clippy**：开发者正在整合一个**音乐生成系统**，并提供选项让用户选择自己的 **lipsync 虚拟形象**作为类似 “Clippy” 的助手。
   - 他们正在改进 **wav2lip** 并尝试使用 **latensync**，以创建一个速度合理且高质量的对话 UI。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/egorsmkv/extract-audio">GitHub - egorsmkv/extract-audio: 从 Hugging Face `datasets` 库生成的 parquet 或 arrow 文件中提取音频文件。</a>：从 Hugging Face `datasets` 库生成的 parquet 或 arrow 文件中提取音频文件。 - egorsmkv/extract-audio</li><li><a href="https://colab.research.google.com/drive/1prztEZIf8nNFUSaptY8Jv16VO8Crjnzb?usp=sharing">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1354012576534495232)** (2 messages): 

> `AutoCAD 图纸生成、物体位置的公制比例` 


- **通过 Prompt 生成 AutoCAD 图纸**：一位成员询问如何根据输入提示词生成 **AutoCAD** 图纸，寻求自动化设计创建的方法。
   - 这表明了利用 **AI** 或其他程序化工具将文本描述转化为 CAD 模型以简化设计工作流的兴趣。
- **无需参考物测量公制比例**：一位成员提出了一个挑战，即在不使用参考物体或距离测量的情况下，获取物体所在位置的**公制比例**。
   - 这旨在寻求在没有传统空间线索的情况下估算场景大小或比例的创新解决方案，可能涉及**图像分析**或其他上下文线索。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1354080726320021587)** (1 messages): 

> `非结构化数据转 JSON、LLM 微调数据集` 


- **用户寻求非结构化数据转 JSON 的转换网站**：一位成员正在寻找能将**非结构化数据**转换为 **JSON** 等**结构化数据**的网站，用于 **LLM 微调数据集**。
- **LLM 微调数据集转换**：用户特别需要一种工具，将非结构化信息转换为适合训练 Large Language Models 的结构化格式。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1353824302524530848)** (1 messages): 

> `Gradio 深度链接、Gradio 5.23` 


- **Gradio 新增深度链接功能！**：**Gradio 5.23** 引入了对 **Deep Links** 的支持，可以直接链接到特定的生成输出（如图像或视频）。
   - 提供了一个指向由 Flux 生成的**冠蓝鸦图像**的示例链接：[https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek](https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek)。
- **立即升级到 Gradio 5.23！**：用户被指示通过 `pip install --upgrade gradio` 升级到最新版本 **Gradio 5.23**，以使用新的 **Deep Links** 功能。
   - 附带的一张图片展示了这一新功能，可以在[这里](https://cdn.discordapp.com/attachments/1014577787039924226/1353824302855622746/image.png?ex=67e46022&is=67e30ea2&hm=d0e0e82ce95fbb6745775ca3274bbce8c92061de43b4f725f643d61076ed06f8&)查看。



**提到的链接**：<a href="https://abidlabs-black-forest-labs-flux-1-schnell.hf.space/?deep_link=oUq4ebmL1Ek">black-forest-labs/FLUX.1-schnell</a>：未找到描述

  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1353973947770863646)** (4 messages): 

> `Llama-3.2 and LlamaIndex.ai, Ollama setup, BAAI/bge-base-en-v1.5, Custom Tool help` 


- **Llama-3.2 与 LlamaIndex.ai 集成**：一位成员参考 [此教程](https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/) 实验了 **Llama-3.2**，并指出它展示了如何使用 **LlamaIndex** 构建 **Agent**，从基础示例开始并逐步添加 **Retrieval-Augmented Generation (RAG)** 能力。
- **Ollama 辅助本地 LLM 设置**：一位成员使用 **Ollama** 作为工具在本地设置 **LLM**，并按照 [README](https://github.com/jmorganca/ollama) 进行安装。
   - 为了下载 Llama3 模型，使用了命令 `ollama pull llama3.1`，成员们指出这需要一台至少配备 **~32GB RAM** 的机器。
- **使用 BAAI 嵌入模型**：该成员使用 [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) 作为其嵌入模型。
   - 成员必须执行 `pip install llama-index-llms-ollama llama-index-embeddings-huggingface` 以完成与 **Ollama** 和 **Huggingface** 的集成。
- **请求自定义工具帮助**：一位成员在编辑 *my_custom_tool* 函数以构建和测试自己的工具时请求帮助，询问如何在 **Hugging Face** 中进行编辑。
   - 另一位成员建议进入 *files*，点击 *app.py* 文件，然后在 **Hugging Face** 内部点击编辑。



**提及的链接**：<a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example_local/">入门教程（使用本地 LLM）- LlamaIndex</a>：未找到描述

  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1353805793954103457)** (39 messages🔥): 

> `New MCP Mod, Nexus context management system for AI coding assistants, Atom of Thoughts for Claude, Deepseek V3 with AOT, Running Multiple MCP Servers` 


- **新版主加入 MCP 社区！**：MCP 社区迎来了一位新版主，他自项目初期就[积极参与](https://github.com/evalstate)并为多个项目做出了贡献。
   - 他计划组织 MCP 活动并帮助社区进一步成长。
- **Nexus 为受上下文限制的 AI 编程者解决难题！**：一位成员分享了 [Nexus](https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/)，这是一个旨在解决 AI 编程助手上下文管理挑战的系统，特别是在大型软件项目中，旨在降低 **token 成本** 并提高 **代码准确性**。
   - Nexus 解决了 **LLM** 有限的上下文窗口问题，该问题会导致代码生成不准确和高昂的 token 成本。
- **Atom of Thoughts 令 Claude 用户着迷**：在讨论使用 Anthropic 的 'think tool' 后，一位成员推荐了适用于 **Claude** 的 **Atom of Thoughts**，并形容其效果“不可思议”。
   - 另一位成员分享了 **Deepseek V3** 配合 **AOT** 运行的图片。
- **同时运行多个服务器**：成员们讨论了如何运行多个具有用户定义端口的 MCP 服务器，建议使用 **Docker** 并映射端口。
   - 他们还指出可以通过 [python-sdk](https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56) 中的 `FastMCP` 构造函数来配置端口。
- **GPT-4o-mini 的工具调用幻觉**：一位成员报告称，即使没有提供任何工具定义，**gpt-4o-mini** 也会针对不存在的 `process_text` 函数产生工具调用请求的幻觉。
   - 该请求后来变成了 `text_processing`，但在工作区中同样未找到。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/mcp/comments/1jj3iuq/nexus_a_system_for_managing_context_and_improving/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://github.com/MissionSquad/nexus">GitHub - MissionSquad/nexus: Nexus 系统：AI 辅助软件开发范式</a>：Nexus 系统：AI 辅助软件开发范式 - MissionSquad/nexus</li><li><a href="https://github.com/modelcontextprotocol/python-sdk/blob/4e11f2890b30be59ca67e5198cb5ede8f401c3a2/src/mcp/server/fastmcp/server.py#L56>">python-sdk/src/mcp/server/fastmcp/server.py at 4e11f2890b30be59ca67e5198cb5ede8f401c3a2 · modelcontextprotocol/python-sdk</a>：Model Context Protocol 服务器和客户端的官方 Python SDK - modelcontextprotocol/python-sdk
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1353805686282256496)** (23 条消息🔥): 

> `Speech MCP, gotoHuman MCP Server, Apple MCP tools, VNC control via Claude` 


- **用于语音交互界面的 **MCP****：一位成员分享了他们用于语音交互并带有音频可视化的主 MCP：[speech-mcp](https://github.com/Kvadratni/speech-mcp)，这是一个 Goose MCP 扩展。
   - 这允许通过**音频可视化**进行语音交互。
- ****gotoHuman** MCP Server 寻求审批**：gotoHuman 团队展示了一个 MCP Server，用于向 Agent 和工作流请求**人工审批**：[gotohuman-mcp-server](https://github.com/gotohuman/gotohuman-mcp-server)。
   - 该服务器允许对 LLM 的操作进行便捷的人工复核，通过**自然语言**定义审批步骤，并在审批后触发 webhook。
- ****Apple MCP** 工具发布**：一位成员介绍了一系列针对 MCP 协议的 **Apple 原生工具**：[apple-mcp](https://git.new/apple-mcp)。
   - 在[这个分步视频](https://x.com/DhravyaShah/status/1892694077679763671)中可以找到使用 **VNC 控制**的本地 MCP Server 演示。
- **通过 **VNC** 和 Claude 控制桌面**：一位联合创始人分享了他们的业余项目，该项目通过 **Claude** 桌面应用提供对远程 macOS 桌面的 **VNC** 控制：[mcp-remote-macos-use](https://github.com/baryhuang/mcp-remote-macos-use)。
   - [此处](https://www.youtube.com/watch?v=--QHz2jcvcs)提供了一个 **YouTube** 演示，其用例是将 Blender MCP 与 MCP Omni 连接，从而将 CLI 与 OpenAI 模型连接起来。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.co">GitHub · 在单一协作平台上构建和交付软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在此构建推动人类进步的软件。</li><li><a href="https://www.youtube.com/watch?v=--QHz2jcvcs">Claude Mcp Remote MacOs Use 演示（字幕）</a>：GitHub 仓库：github.com/baryhuang/mcp-remote-macos-use</li><li><a href="https://github.com/gotohuman/gotohuman-mcp-server">GitHub - gotohuman/gotohuman-mcp-server</a>：通过在 GitHub 上创建账号来为 gotohuman/gotohuman-mcp-server 的开发做出贡献。</li><li><a href="https://github.com/Kvadratni/speech-mcp">GitHub - Kvadratni/speech-mcp: Speech MCP: 一个用于带有音频可视化的语音交互的 Goose MCP 扩展</a>：Speech MCP: 一个用于带有音频可视化的语音交互的 Goose MCP 扩展 - Kvadratni/speech-mcp</li><li><a href="https://git.new/apple-mcp">GitHub - Dhravya/apple-mcp: 针对 Model Context Protocol 的 Apple 原生工具集。</a>：针对 Model Context Protocol 的 Apple 原生工具集。 - Dhravya/apple-mcp
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1353871431959576696)** (7 条消息): 

> `NotebookLM, Versatile Bot Project, Interactive mode, Chat Episode Prompt, Delivery pacing` 


- **使用 NotebookLM 担任播客主持人**：一位成员询问如何将 **NotebookLM** 用作播客主持人，让用户作为嘉宾参与关于特定话题的对话。
   - 另一位成员询问在哪里可以找到实现此功能的 **Chat Episode Prompt**。
- **深入了解 Versatile Bot 项目**：一位成员发布了 [Versatile Bot Project](https://github.com/shun0t/versatile_bot_project)，其中包括一份 **Chat Episode 提示词文档**，专为 AI 主持人在**交互模式（Interactive mode）**下讨论任意话题而设计。
   - 该模式允许用户在 AI 主持人交谈时加入节目，从而实现对指定话题的对话。
- **控制 AI 主持人的播报节奏**：一位成员询问如何更改 AI 主持人的播报节奏，例如加快或减慢阅读速度。
   - 另一位成员提供了一个带有参数的模板，如 *Energetic pace（活力节奏）*、*clear articulation（清晰发音）* 以及目标 **words/minute（每分钟字数）** 来控制 AI 的播报速度，源文档类型也会影响 AI 的引导方式。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1353812676769480795)** (47 条消息🔥): 

> `Google Cloud Platform Billing, Workspace Data Export Tool, NotebookLM Plus subscription benefits, Mind Map Feature rollout, Multilingual Podcast` 


- **GCP 计费困惑得到澄清**：一位用户启用了 **Google Cloud Platform** 以访问 **Data Export** 工具，但不确定计费情况；另一位用户澄清说，启用计费并不一定会产生费用。
   - 该用户确认他们从管理控制台启动了 Data Export，并通过 console.cloud.google.com 访问存档。
- **数据导出注意事项揭示**：用户发现，在导出过程中选择数据目的地的选项受其 **Workspace edition** 限制，且导出的数据存储在 **Google-owned bucket** 中，并将在 **60 天**内删除，详见 [Google Support](https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en)。
- **思维导图可用性问题**：用户报告 NotebookLM 中缺少 **Mind Map** 功能，经确认该功能正在 **逐步推出 (gradual rollout)**。
   - 一些用户推测延迟推出是因为在修复 Bug，一位用户表示推出的速度*慢得像蜗牛爬一样*。
- **自定义框隐藏确认？**：一位用户询问 **customize box** 是否在免费版 NotebookLM 中不再可用。
   - 另一位用户回答说，**customize box** 仍然显示在免费账户中，而不是 Plus 账户。
- **缺少多语言播客功能**：一位用户请求为 NotebookLM 提供**多语言功能**，特别是针对 **podcast** 功能。
   - 另一位用户指出 NLM chat 已经是多语言的，但 podcast 功能目前仅提供英文版。



**提及的链接**：<a href="https://support.google.com/a/answer/14338836?sjid=14118684210403272528-EU&hl=en">Export your users' data - Google Workspace Admin Help</a>：未找到描述

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1353830953004568710)** (19 条消息🔥): 

> `Universal Translator, Mozilla's Transformer Lab, GPU Tokenization, Gemini 2.5 Pro` 


- **通用翻译器即将到来？**：一位成员推测，鉴于 **ChatGPT** 理解和翻译语言的能力，**通用翻译器**距离实现仅剩五年时间。
   - 另一位成员随后分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=K1RbD7aAXtc)，询问视频中*是什么模型在唱歌*。
- **Mozilla 的 Transformer Lab 引起关注**：一位成员询问是否有人认真研究过 **Mozilla's Transformer Lab** 项目，这是一种可能让普通用户在普通硬件上进行训练和 fine-tuning 的方法，并分享了 [GitHub repo](https://github.com/transformerlab/transformerlab-app) 的链接。
   - 另一位成员确认 `Transformer Lab 很自豪能够通过 Mozilla Builders Program 获得 Mozilla 的支持`，并澄清它是受其*支持*而非由其*创建*。
- **GPU Tokenization 的说法**：一位成员注意到在 tokenization 期间，**LM Studio** 会将单个 CPU 线程推向满负荷，但很好奇 tokenizing 过程是否 100% 在 GPU 上运行。
   - 另一位用户回答说 *tokenizing 与 GPU 无关*，但随后又反驳了自己，观察到调整 **flash attention** 以及 k 和 v 的 **cache settings** 对 tokenizing 时间有很大影响。
- **Gemini 2.5 Pro 试用体验**：一位成员询问其他人是否尝试过 **Gemini 2.5 Pro**。
   - 另一位成员回答说他们尝试过了，它能够正确回答 **Gemini 2.0 Flash Thinking** 无法回答的逻辑谜题，并分享了在 [aistudio](https://www.hopeless.fr/share/msedge_O0y9jZHBZV.png) 免费使用它的链接。



**提及的链接**：<a href="https://github.com/transformerlab/transformerlab-app">GitHub - transformerlab/transformerlab-app: Open Source Application for Advanced LLM Engineering: interact, train, fine-tune, and evaluate large language models on your own computer.</a>：用于高级 LLM 工程的开源应用程序：在您自己的计算机上交互、训练、微调和评估大语言模型。- transformerlab/transformerlab-app

  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1353840864043995278)** (23 条消息🔥): 

> `GPU 的 VRAM 限制、GPU 在 Docker 容器中的开销、3090 Ti 速度、M4 Max 在 32B 模型下的功耗、AMD GPU 的 ROCm 支持` 


- **探索 GPU VRAM 限制调整**：一位用户尝试将其 **16 GB GPU** 上的预留 **VRAM** 从 **0.5 GB** 降低到 **0.2 GB** 以用于 LLM 处理，但另一位成员警告不要完全占满 **VRAM**，以避免系统锁定。
   - 虽然可能达到 **0.3 GB**，但*这并不可靠，因为任何额外的 VRAM 占用都会导致整个设置崩溃*。
- **Docker 在 GPU 上的开销**：一位用户通过各自的 **Docker** 容器访问其 **8 个 GPU**，并询问系统内存开销。
   - 一位成员表示，*除了最基本的必要组件外，Docker 基本上不应包含内部系统*，但每个实例中仍需加载 CUDA 核心。
- **3090 Ti 满载运行**：一位用户展示了其 **3090 Ti** 满载运行的状态，并在[截图](https://cdn.discordapp.com/attachments/1153759714082033735/1354073319133155429/image.png)中展示了*纯粹的速度*。
   - 据报告，在不开启 flash 的情况下速度约为 **20 tokens/s**，且在 **4-5k tokens** 后会变慢；开启 flash 后速度约为 **30 tokens/s**。
- **M4 Max 在运行 32B 推理模型时功耗激增**：一位用户注意到，与同尺寸或更大尺寸的模型（最高 **120W**）相比，**32B 级推理模型（thinking models）** 在其 **M4 Max** 上驱动的功耗更高（最高达 **140W**）。
   - 用户猜测这是由于*更激进的内存访问*导致的，因为 *GPU 的运行负载似乎基本相同*。
- **AMD GPU 的 ROCm 支持更新**：一位用户询问了对最新 AMD GPU 的支持情况，另一位用户提到目前*仅通过 Vulkan* 支持，在 llama.cpp 的 master 分支版本中仍缺乏 **ROCm** 支持。
   - 另一位成员澄清说，ROCm llama.cpp 引擎会随 CUDA 和 Vulkan 同步更新，但指出对某些 AMD GPU（如 **90xx** 系列）的支持可能是一个独立的问题。


  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1353816255278092359)** (17 条消息🔥): 

> `数据保留、安全、数据隐私、数据使用政策、零数据保留 (ZDR)` 


- **Cohere 澄清数据隐私政策**：针对用户的咨询，Cohere 团队分享了其[隐私政策](https://cohere.com/privacy)和[数据使用政策](https://cohere.com/data-usage-policy)的链接，强调用户和客户在使用服务时应避免上传个人信息。
- **Cohere 强调数据控制选项**：Cohere 提供 **SaaS 平台**，允许用户通过[控制面板](https://dashboard.cohere.com/data-controls)直接控制其数据，并可根据要求提供 **零数据保留 (ZDR)** 支持（通过发送邮件至 support@cohere.com）。
- **Cohere 的部署选项**：Cohere 可在 **OCI**、**Bedrock**、**Sagemaker** 和 **Azure Cloud** 等主流云提供商上使用，确保请求保留在云环境内；同时也提供本地部署（on-prem）方案，详见其[部署选项页面](https://cohere.com/deployment-options)。
- **Cohere 达到安全与合规标准**：Cohere 符合 **SOC II** 和 **GDPR** 标准，遵循数据安全和隐私的行业标准，更多详情可见其[安全政策](https://cohere.com/security)。
- **用户获得数据使用控制权**：用户可以在[数据控制面板](https://dashboard.cohere.com/data-controls)上管理其数据设置，防止其数据被用于提示词生成或微调。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dashboard.cohere.com/data-controls">登录 | Cohere</a>：登录以通过易于使用的 API 访问高级大语言模型和 NLP 工具。</li><li><a href="https://cohere.com/security">安全 | Cohere</a>：通过 Cohere 的企业级安全协议、严格的访问控制和私有部署选项，确保极致的 AI 安全与隐私。</li><li><a href="https://cohere.com/privacy">隐私政策 | Cohere</a>：Cohere Inc. (“Cohere”) 重视并尊重您的隐私。我们准备了此隐私政策，以解释我们通过网站收集、使用和披露个人信息的方式...</li><li><a href="https://cohere.com/deployment-options">部署选项 - SaaS, Cloud API, 虚拟私有云 (VPC), 本地部署 | Cohere</a>：我们的解决方案提供行业领先的数据隐私和安全，旨在满足寻求利用生成式 AI 力量的各类组织的需求。无论您是初创公司还是...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1354002232085057567)** (16 messages🔥): 

> `Cohere API streaming, Cohere embedding generator, Cohere tokenization` 


- **Cohere API 支持响应流式传输**：Cohere API 支持响应流式传输 (streaming)，这可以让用户在文本生成时即时在客户端看到内容，从而提升用户体验，详情参阅 [Cohere Chat Stream API 参考文档](https://docs.cohere.com/reference/chat-stream)。
- **为 CohereEmbeddingGenerator 进行文本分词**：一位正在构建 .NET 版 **CohereEmbeddingGenerator** 客户端的用户询问关于在嵌入之前进行文本分词 (tokenizing) 的问题，得到的建议是使用 `/embed` 端点（该端点会返回使用的 token 数量），或者从 [Cohere 公共存储](https://storage.googleapis.com/cohere-public/tokenizers/embed-english-v3.0.json) 手动下载分词器 (tokenizer)。
   - 用户被引导通过向 `api.cohere.com/v2/models/embed-english-v3.0` 发送 GET 请求来下载其模型对应的分词器，从 `tokenizer_url` 获取分词器，然后使用 HF tokenizer 等库手动进行分词。



**相关链接**: <a href="https://docs.cohere.com/reference/chat-stream">Chat with Streaming — Cohere</a>：生成对用户消息的文本响应。要了解如何使用 Chat API 和 RAG，请参考我们的文本生成指南。按照迁移指南的说明从 API v1 迁移到 API v2。

  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/1354175936861573120)** (2 messages): 

> `` 


- **无相关讨论**：在提供的消息中未发现相关讨论。消息仅包含简单的问候。
- **频道不活跃**：该频道似乎大部分时间处于不活跃状态，用户之间仅有简短的问候交流。


  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1354119470205108268)** (2 messages): 

> `NLP project, Text summarization tool, Introduction of Sage` 


- **Sage 自我介绍**：一位名叫 Sage 的新成员向社区介绍了自己。
   - 他们提到正在为大学毕业年级做一个 **NLP 项目**：构建一个**文本摘要工具**。
- **Sage 寻求文本摘要工具的指导**：Sage 正在构建一个**文本摘要工具**作为其 NLP 项目。
   - 由于目前遇到了一些困难，他们希望向社区学习并做出贡献。


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1354137784583065741)** (1 messages): 

> `torchtune v0.6.0, Tensor Parallel, Phi 4, Multinode training` 


- **TorchTune v0.6.0 发布！**：TorchTune 刚刚发布了 **v0.6.0** 版本，带来了多项新功能。
   - 发布说明可以在 [这里](https://github.com/pytorch/torchtune/releases/tag/v0.6.0) 找到。
- **张量并行 (Tensor Parallel) 功能**：现在支持 **Tensor Parallel**，用于更大规模的分布式训练和推理 recipe！
   - 这一增强功能可以更高效地处理大规模模型和数据集。
- **支持微软 Phi 4**：添加了针对微软最新模型 **Phi 4** 的构建器。
   - 有关 **Phi 4** 的更多信息可以在 [Microsoft Tech Community 博客](https://techcommunity.microsoft.com/blog/aiplatformblog/introducing-phi-4-microsoft’s-newest-small-language-model-specializing-in-comple/4357090)上找到。
- **多节点训练 (Multinode training) 上线！**：现在支持 **Multinode training**，方便在多个节点上进行分布式训练。
   - 点击 [这里](https://pytorch.org/torchtune/stable/tutorials/multinode.html) 开始使用多节点训练。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1353827988663177326)** (14 messages🔥): 

> `DeepSeek-V3, Quantization Aware Training, MoEs in torchtune` 


- **DeepSeek 发布 V3 模型，跳过 Readme**：成员们注意到 [DeepSeek-V3 模型](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324) 发布时*没有附带 Readme*，并开玩笑说 **DeepSeek AI 团队**已经变得*放飞自我（unhinged）*了。
   - 新模型拥有 **chat 界面**、**Hugging Face 集成**，并提供了其 **Discord**、**Wechat** 和 **X** 账号的链接。
- **Torchtune 的 MoEs 激发遐想**：一位成员提到了在 **torchtune** 中添加 **MoEs** 的隐晦提醒，思考这是否需要 **8-9 TB 的 VRAM** 以及*由 100 台 H100 或 H200 组成的集群*来进行训练。
   - 他们开玩笑地表示，必须*先搬走阁楼里的几个箱子*才能腾出空间。
- **Quantization Aware Training 后保留优化器状态**：一位成员询问了 **Quantization Aware Training (QAT)** 如何影响优化器状态，并链接到了 [*torchtune* 中的相关代码](https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790)。
   - 另一位成员确认，*在切换到 QAT 后，优化器状态会被保留*。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-0324">deepseek-ai/DeepSeek-V3-0324 · Hugging Face</a>：未找到描述内容</li><li><a href="https://github.com/pytorch/torchtune/blob/57c8d6b50d1462cc437d57991dca7f8acb599678/recipes/qat_distributed.py#L790">torchtune/recipes/qat_distributed.py at 57c8d6b50d1462cc437d57991dca7f8acb599678 · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1353817270228484158)** (20 messages🔥): 

> `CUDA overhead, Cursed Submodule, vLLM + GRPO, r1-zero` 


- **利用 Graph Capture 消除 CPU CUDA 启动开销**：为了减少 GPU 空闲时间，成员们讨论了从 CPU 启动 CUDA 操作具有不可忽视的开销，而将 GPU 操作捕获为 Graph 并作为单个操作启动可以整合计算图。
   - 一位成员询问这是否就是 compile 所做的事情。
- **是否需要一个 /cursed 子模块来提升 10 倍性能？**：成员们辩论是否要为那些能大幅提升性能的“邪门代码（cursed code）”创建一个 `/cursed` 子模块，例如手动指定 CUDA 设备。
   - 一位成员表示，他正在使用一种仅适用于较小模型的方法，即每个进程都有自己的 vLLM 实例来生成数据，而不是采用更常见的集中式 vLLM 生成进程。
- **vLLM 与 GRPO 集成：小模型 vs 大模型**：一位成员已经拥有了一个适用于分布式设置的工作版本，但在得知内部已经完成了一些工作后停止了迭代。
   - 据称当前的方法更适合小模型（最高 8B），而内部方法则更适合大模型（>=70B）。
- **用于异步训练的 r1-zero Recipe**：一位成员分享了 [r1-zero 的链接](https://github.com/joecummings/r1-zero/blob/main/scripts/runnable_recipe_ray_vllm_weight_sync.py)，用于推理模型的异步训练，并强调这仍是一个正在进行中的工作。
   - 他们还提到计划很快将该 Recipe 的一个版本集成到 torchtune 中，重点是代码清理以及允许在 vLLM 中使用非 HF 模型。
- **通过 HTTP 调用进行层级 GPU 地址映射**：一位成员建议修改 **vLLM**，使其在启动时带有每个层级的 GPU 地址映射。
   - 目标是在侧边运行一个 vLLM 进程并通过 HTTP 调用与其交互；已经有人在*着手处理*此事。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1353825891112845412)** (31 条消息🔥): 

> `LocalDocs 备份、聊天数据的隐私考量、用于消息处理的本地 LLM 与 API 对比、LocalDocs DB 导入、丢失 LocalDocs DB` 


- **LocalDocs DB 安全备份**：成员们讨论了备份 `localdocs.db` 文件以避免数据丢失，特别是当原始文档丢失或无法访问时。
   - 一位成员建议 GPT4All 使用编号最高的 `*.db` 文件（例如 `localdocs_v3.db`），重命名它们可能允许导入/导出，尽管这尚未得到证实。
- **隐私法律对聊天数据分析的挑战**：一位成员对隐私法律（特别是欧盟的法律）以及在处理聊天数据时确保合规性的必要性提出了担忧。
   - 讨论强调了在将聊天消息输入 LLM 之前，验证权限和消息格式（纯文本或可转换格式）的重要性。
- **用于聊天处理的 LLM API 与本地 LLM 对比**：一位成员询问是使用 **Deepseek** 或 **OpenAI** 等付费 API，还是运行本地 LLM 来处理传入的群聊消息，以计算满意率、提取关键词并总结消息。
   - 另一位成员建议，如果消息量相对较小（小于 **100MB**），配备良好 GPU 的本地机器可能就足够了，特别是使用较小的模型进行标注和总结时。
- **LocalDocs DB 导入挑战**：成员们讨论了导入 `localdocs.db` 文件的可能性，但指出该文件包含加密/特殊编码的文本，如果没有 Embedding 模型，通用的 LLM 很难解析。
   - 一位丢失了 localdocs.db 的成员正经历极其缓慢的 CPU 索引过程，并希望能绕过这个问题。
- **Win11 更新擦除 LocalDocs**：一位成员在 Windows 11 更新后发现其 `localdocs.db` 变为空，并正努力在 CPU 上重新索引本地文档。
   - 更新导致的驱动器盘符更改被认为是可能的原因，建议将文件移动到 C 盘以避免此类问题。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1354127968749879368)** (2 条消息): 

> `LlamaCloud MCP 服务器、用 Python 构建 MCP 服务器` 


- **LlamaCloud 作为 MCP 的奇迹**：[LlamaCloud](https://www.llamaindex.ai/) 可以作为任何兼容客户端的 **MCP 服务器**，这个 [demo](https://t.co/t8yteZLg19) 展示了其实际运行情况。
   - 这展示了 **MCP** 是多么受欢迎。
- **Python 开发者构建便携式 Python 范式**：一位成员展示了如何使用 **LlamaIndex** 构建自己的 **MCP 服务器**，为任何 MCP 客户端提供各种工具接口，只需约 35 行 Python 代码即可连接到 **Cursor AI**。
   - 成员们实现了 **Linkup 网络搜索**和[这个](https://t.co/kj6UfDj0TU)项目。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1353886649397809203)** (25 条消息🔥): 

> `Claude MCP 支持，LlamaIndex 的多 Agent 性能，LlamaIndex 中的 Agent 类型，自动化 LLM 评估` 


- **LlamaIndex 新增 Claude MCP 兼容性**：一位成员提供了一个将 **Claude MCP** 与 **LlamaIndex** 集成的简化示例，在[代码片段](https://link.to.snippet)中展示了如何使用 `FastMCP` 和 `uvicorn` 为 **Claude Desktop** 或 **Cursor** 等 MCP 客户端暴露本地主机和端口。
- **Agent 运行缓慢？使用 AgentWorkflow 提升速度**：有用户反映在使用 **Gemini 2.0** 配合 12 个工具和 3 个 Agent 的 **LlamaIndex MultiAgentic** 设置时性能缓慢；建议使用 `AgentWorkflow` 和 `can_handoff_to` 字段来实现[受控的 Agent 交互](https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow)。
- **LlamaIndex Agent 让新手感到困惑**：一位成员对 **LlamaIndex** 中不同的 Agent 类型及其使用时机表示困惑，官方即将对文档进行重构。
   - 一位团队成员指出，通常应使用 `core.agent.workflow`，对于具有函数/工具 API 的 **LLM** 使用 **FunctionAgent**，其他情况使用 **ReActAgent**，并建议参考 [Hugging Face 课程](https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents)获取更多帮助。
- **无需 Prompt 的自动化 LLM 评估！**：一位创始人正在验证一个关于 **OSS 自动化评估** 的想法，通过单个 API 且无需评估 Prompt，使用专有模型在 500 毫秒内完成幻觉（Hallucination）和相关性（Relevance）等任务的评估，并计划提供包括模型、托管和编排工具在内的端到端解决方案，更多细节请参阅 [autoevals.ai 网站](https://www.autoevals.ai)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.autoevals.ai">首页</a>：未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/en/unit2/llama-index/agents">在 LlamaIndex 中使用 Agent - Hugging Face Agent 课程</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index">GitHub - run-llama/llama_index: LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agent/#multi-agent-systems-with-agentworkflow">多 Agent 工作流 - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb at main · run-llama/llama_index</a>：LlamaIndex 是构建基于数据的 LLM 驱动 Agent 的领先框架。 - run-llama/llama_index
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1353894887120175186)** (4 条消息): 

> `AI IDE 评估, CoTs 上的 SAEs, DeepSeek-V3-0324 展示, Gemini 2.5 Pro` 


- **新成员加入可解释性研究，重点关注 SAEs**：新成员 Sam 拥有物理和几何背景，表示有兴趣为可解释性研究做出贡献，特别是针对 **CoTs** 使用 **SAEs**。
   - Sam 分享说他们的背景是 *物理 + 几何*，并拥有 *数据科学* 经验，包括深度学习。
- **AI IDE 评估仓库出现**：一位成员正在寻求关于评估 **AI IDEs** 的反馈，指出尽管使用了相似的模型，Cursor 和 Windsurf 等 IDE 之间的性能表现并不一致，并分享了一个用于对比的 [仓库](https://github.com/grahamannett/ai-ide-compare)。
   - 目前的评估侧重于 'greenfield'（从零开始的）项目，使用预定义的提示词，并通过代码行数和文件数等指标评估生成的代码/任务，未来可能会加入更深入的评估。
- **DeepSeek-V3-0324 生成 p5.js 程序**：一位用户引用了 AK (@_akhaliq) 的话，强调 **DeepSeek-V3-0324** 成功编写了一个 p5.js 程序，展示了一个在旋转的六边形内弹跳的球，受重力和摩擦力影响，并能真实地从墙壁上反弹。
   - 该模型甚至根据要求提供“调整参数的滑块”和边数按钮的提示词，即兴创作了球体重置和随机化等功能，如 [这条推文](https://x.com/teortaxesTex/status/1904342699756433859) 中所示。
- **Google 发布 Gemini 2.5 Pro**：Google 发布了 **Gemini 2.5 Pro**，称其为 *全球最强大的模型*，具有统一的推理能力、长上下文和工具使用能力，详见 [这条推文](https://x.com/OfficialLoganK/status/1904580368432586975)。
   - 该模型目前在 **Google AI Studio + API** 中以实验性方式免费提供，定价详情将很快公布。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OfficialLoganK/status/1904580368432586975">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：介绍 Gemini 2.5 Pro，全球最强大的模型，具有统一的推理能力 + 你喜欢的 Gemini 的所有特性（长上下文、工具等）。目前作为实验性版本免费提供...</li><li><a href="https://x.com/teortaxesTex/status/1904342699756433859">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：V3-0324 一次性完成了这个。它即兴创作了大部分功能，比如球体重置和随机化——我只要求了“调整参数的滑块”和边数按钮。发球相关的帖子很乏味。默认...</li><li><a href="https://github.com/grahamannett/ai-ide-compare">GitHub - grahamannett/ai-ide-compare</a>：通过在 GitHub 上创建账号来为 grahamannett/ai-ide-compare 的开发做出贡献。
</li>
</ul>

</div>

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1354023705629429771)** (12 messages🔥): 

> `SkyLadder short-to-long context window transition, Data-constrained pretraining for math, Composable Generalization` 


- **SkyLadder: 从短到长上下文是最佳路径**：ArXiv 上的一篇论文提出了 **SkyLadder**，这是一种简单且有效的方法，在 LLM 预训练中实施**从短到长的上下文窗口转换**，在常见任务上实现了高达 **3.7%** 的持续提升 ([2503.15450](https://arxiv.org/abs/2503.15450))。
   - 作者在 **100B tokens** 上预训练了 **1B** 和 **3B** 参数模型，证明了 **SkyLadder** 在保持强大的标准基准测试性能的同时，在长上下文任务上达到或超过了基准结果。
- **人类思维：数据受限预训练的关键？**：一种新方法建议，显式地建模和推断文本生成过程背后的**潜在思维 (latent thoughts)** 可以显著提高预训练数据的效率，尤其是在数学领域 ([2503.18866](https://arxiv.org/abs/2503.18866))。
   - 论文通过合成数据方法实证了推断潜在思维的有效性，其表现优于在相同数量的原始数据上进行的训练（**5.7% -> 25.4%**）。
- **通过超网络实现可组合泛化！**：一篇论文将多头注意力重新表述为**超网络 (hypernetwork)**，揭示了一个可组合的、低维的潜在代码指定了特定于键-查询 (key-query) 的操作，使 Transformer 能够泛化到新的问题实例 ([2406.05816](https://arxiv.org/abs/2406.05816))。
   - 一位成员很喜欢作者的解释：*对于单对 q, k 索引，作者将沿 head-number 维度的激活解释为指定任务/上下文的潜在代码（维度为 n_heads）*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.05816">Attention as a Hypernetwork</a>：Transformer 在某些情况下可以泛化到新的问题实例，这些实例的组成部分可能在训练期间遇到过，但其组合方式未曾遇到。什么样的机制...</li><li><a href="https://arxiv.org/abs/2503.18866">Reasoning to Learn from Latent Thoughts</a>：语言模型 (LM) 预训练的计算缩放已经超过了人类编写文本的增长速度，导致人们担心数据将成为 LM 缩放的瓶颈。为了继续缩放预训练...</li><li><a href="https://arxiv.org/abs/2503.15450">SkyLadder: Better and Faster Pretraining via Context Window Scheduling</a>：最近 LLM 预训练的进展以不断扩大的上下文窗口为特征，以处理更长的序列。然而，我们的初步研究表明，使用较短上下文窗口预训练的模型...</li><li><a href="https://arxiv.org/abs/2503.18908">FFN Fusion: Rethinking Sequential Computation in Large Language Models</a>：我们引入了 FFN Fusion，这是一种架构优化技术，通过识别和利用自然的并行化机会来减少 LLM 中的串行计算。我们的...</li><li><a href="https://arxiv.org/abs/2106.06295">Going Beyond Linear Transformers with Recurrent Fast Weight Programmers</a>：具有线性化注意力的 Transformer（“线性 Transformer”）已经证明了基于外积的快速权重规划器 (FWPs) 的实际可扩展性和有效性...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1354035224857673739)** (1 messages): 

> `Chinchilla Scaling Formula, Impact of Suboptimal Hyperparameters, Learning Rate Effects` 


- **深入探讨 Chinchilla 缩放公式**：一位成员询问了非最优的超参数设置如何影响跨不同规模优化模型训练的 **Chinchilla 缩放公式**，并引用了 [Chinchilla 论文](https://arxiv.org/abs/2203.15556)。
   - 具体而言，他们询问如果**学习率 (learning rate)** 设置得太低（例如，仅为最优值的 1/10），公式中的哪些参数（**E**, **A**, **B**, **alpha** 或 **beta**）会被改变。
- **学习率的连锁反应**：该询问集中在非最优**学习率**对 **Chinchilla 公式**所预测的缩放行为的理论后果上。
   - 旨在了解持续偏低的**学习率**是主要影响**误差项 (E)**、缩放系数 (**A**, **B**)，还是控制模型大小、训练数据与性能之间关系的指数 (**alpha**, **beta**)。


  

---

### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1353847794858987610)** (1 messages): 

> `Self-organizing AI, AI Building Blocks` 


- **所有 AI 都使用类似的构建块？**：一名成员提出了这样一个观点，即所有这些系统都可能使用非常相似的构建块（Building Blocks）。
   - 这引导他们提出了 **自组织 AI（self-organizing AI）** 的概念，这种 AI 可以学习不同的潜在配置。
- **AI 自配置推测**：一名成员提出了 AI 的一个新方向，暗示 AI 可以实现自组织。
   - 他们补充说，这种自组织可以允许 AI 动态地学习不同的潜在配置。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1353833840694788237)** (5 messages): 

> `gpt-neox CI status, lm_eval upgrade` 


- **gpt-neox CI 需要修复**：EleutherAI/gpt-neox 仓库的 CI 运行失败，需要进行修复。原因是之前负责维护 CI 的志愿者已入职其他公司的全职岗位，不过目前每个 PR 仍在进行本地测试。
   - 本地通过的测试包括 `pytest tests -m cpu`；在运行 `pytest --forked --cov-report term --cov=megatron tests` 时，除了少数与依赖项（requirements）相关的错误外，其余测试均已通过。
- **lm_eval 升级 PR 已准备好进行评审**：一名成员起草了一个 PR，将评估逻辑更新至最新版本 `lm_eval==0.4.8`，并建议在另一个 PR 中解决失败的测试，因为这些失败似乎与本次更新无关，且在 main 分支上也会失败。链接如下：[PR 1348](https://github.com/EleutherAI/gpt-neox/pull/1348)。
   - 一名成员认为测试失败可能是由于环境配置不正确或依赖项版本不一致导致的。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1348">Update Evaluation Logic to Latest `lm_eval` (0.4.8) and Support Automatic Benchmark Evals w/o Validation Set by Kyle1668 · Pull Request #1348 · EleutherAI/gpt-neox</a>：我正在训练一个模型，希望在整个数据集上进行训练，不希望将数据集拆分为训练/验证/测试集。我希望在一系列基准测试上进行评估，其中一个基准测试是……</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1349">[Throw Away] Sanity Check CI by Kyle1668 · Pull Request #1349 · EleutherAI/gpt-neox</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1354045126258982994)** (15 messages🔥): 

> `Mojo for Website vs Rust, AES in Mojo Progress, SIMD for AES, Rust vs Go for Backend` 


- **与 Rust 相比，Mojo 不适合用于网站开发**：一名成员询问了使用 **Mojo** 与 **Rust** 开发网站的对比，但得到的建议是否定的，理由是 **Mojo** 缺乏编写加密安全代码的规范，且 IO 方案尚不完善。
   - 建议使用 **Rust**，因为它拥有生产级的库和框架，以及更快的异步处理能力，特别是对于需要身份验证或 HTTPS 的应用。
- **Mojo 中硬件支持的 AES 进度搁置**：一名成员提到在 **Mojo** 中实现了硬件支持的 **AES**，但它无法在旧款 Apple silicon Mac 上运行，且不是一个完整的 TLS 实现。
   - 开发者正推迟进一步的 **AES** 开发工作，直到有密码学专家愿意编写软件部分，并强调了非专家实现加密功能的风险。
- **探索用于 AES 实现的 SIMD**：讨论涉及使用 **SIMD** 实现 **AES** 和其他算法，一名成员指出 **x86** 具有 **vaes** 和类似功能用于 **SIMD AES 128**。
   - 有人提到 **ARM** 拥有 **SVE AES**，虽然功能相似但支持程度不如前者，这突显了加密功能可用的硬件级优化。
- **调用 MAX 的 Rust API**：尽管面临挑战，一名成员因认为 **Rust** 不适合快速编写代码而对其表示排斥，希望寻找更简单的后端开发方案；建议是让 **Rust API 调用它**并传递参数。
   - 他们曾考虑使用 **Python**，但发现其速度慢且不稳定，因此计划创建一个可通过 **FFI** 调用的 **Rust** 项目。
- **Go 作为折中方案**：作为替代方案，一名成员建议将 **Go** 作为 Python/Mojo 和 Rust 之间的一个不错的折中选择，且它也是生产就绪的。
   - 然而，另一名成员担心过多的微服务可能会使项目变得过于庞大。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1354201595406975076)** (2 messages): 

> `无需 CUDA，通过 PTX 针对 NVIDIA GPU` 


- **PTX 助力 NVIDIA 上的 “无需 CUDA” 的 Mojo**：Mojo 直接生成 **PTX** (Parallel Thread Execution) 代码来针对 NVIDIA GPU，绕过了对 CUDA 的需求。
   - 这种方法避免了对 **cuBLAS**、**cuDNN** 和 **CUDA C** 的依赖，简化了开发流程。
- **关于无需 CUDA 的细节**：由 bradlarson 确认，团队直接生成 PTX 并从此进行下调（lowering）。
   - 不存在对 cuBLAS、cuDNN 或 CUDA C 的依赖。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1354048890420330568)** (9 messages🔥): 

> `使用 DSPy 进行文本摘要、SIMBA 优化器、输出精炼 vs 断言、BestOfN 模块、Refine 模块` 


- **使用 DSPy 处理摘要任务**：一位成员正在探索使用 **DSPy** 处理文本摘要任务，包含来自专家的 **300** 个示例，并正在一个简单的指标上进行测试。
   - 他们想知道如果优化器能看到摘要的具体差异点，是否会更有效，以及是否有更好的方法来优化文本摘要的 prompt。
- **SIMBA 优化器支持细粒度反馈**：一位成员建议在摘要任务中使用实验性优化器 `dspy.SIMBA`，它允许提供关于生成的摘要与 ground truth 之间差异的反馈。
   - 反馈可以通过 `dspy.Prediction(score=your_metric_score, feedback="stuff about the ground truth or how the two things differ")` 返回，以指导优化。
- **分享输出精炼文档**：一位成员分享了关于 [输出精炼的 DSPy 教程](https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/) 链接，解释了旨在通过使用不同参数设置进行多次 LM 调用来提高预测可靠性的 `BestOfN` 和 `Refine` 模块。
   - 该教程详细说明了这两个模块如何在达到 `N` 次尝试或当 `reward_fn` 返回高于 `threshold` 的奖励时停止。
- **BestOfN 模块通过 Temperature 调整得到改进**：`BestOfN` 模块使用不同的 temperature 设置多次运行给定模块，以获得最佳结果。
   - 它返回第一个通过指定阈值的预测，或者如果没有一个满足阈值，则返回奖励最高的那个。
- **Refine 模块的可组合性较差？**：一位成员询问 `Refine` 是否会取代断言（assertions），以及它是否同样具有细粒度和可组合性，因为它包装了整个模块。
   - 另一位成员回答说，可以通过调整模块大小来管理可组合性，从而允许对范围进行更显式的控制。



**提到的链接**：<a href="https://dspy.ai/tutorials/output_refinement/best-of-n-and-refine/">Output Refinement - DSPy</a>：用于对语言模型进行编程（而非提示）的框架。

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1354166265857572955)** (3 messages): 

> `ROCm 支持、OpenCL 前端、使用 Tinygrad 的 AMD GPU` 


- **旧款 AMD GPU 获得 Tinygrad 助力**：使用 **OpenCL 前端**，ROCm 不支持的旧款 AMD GPU（例如 2013 款 Mac Pro 中的 GPU）可能可以运行 Tinygrad。
   - 成功与否可能取决于特定的驱动程序和系统上可用的 OpenCL 支持级别，因此用户应验证其自定义驱动程序的兼容性。
- **旧款 AMD 的 ROCm 替代方案**：ROCm 不支持旧款 AMD GPU，但 tinygrad 中的 **OpenCL 前端** 可能提供一种解决方法。
   - 成功情况将根据特定的驱动程序版本和 OpenCL 支持程度而有所不同；需要进行实验以确认兼容性。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1353866401005309963)** (1 条消息): 

> `Windsurf Creators Club, Vibe Coding 频道, Windsurf v1.5.8 发布` 


- **Windsurf 推出 Creators Club**：Windsurf 正在奖励进行内容创作的社区成员，提供**每 1k 播放量 $2-4** 的奖励，更多详情请访问 [Windsurf Creators Club](https://whop.com/windsurf/)。
- **Windsurf 推出 'Vibe Coding' 频道**：为 'vibe coders' 创建了一个新频道，用于*进入心流状态 (flow state)*、聊天、讨论并分享技巧/秘籍。
- **Windsurf v1.5.8 发布并包含补丁修复**：**Windsurf v1.5.8** 现已发布，包含补丁修复，包括 cascade/memories 修复、Windsurf Previews 改进以及 cascade 布局修复；同时还分享了该版本的图片。



**提到的链接**: <a href="https://whop.com/windsurf/)">未找到标题</a>: 未找到描述

  

---


---


---


---


{% else %}


> 完整的频道分类明细已针对电子邮件进行了截断。 
> 
> 如果您想查看完整的明细，请访问此电子邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}