---
companies:
- datacomp
- hugging-face
- openai
- nvidia
- mistral-ai
- deepseek
date: '2024-07-20T02:08:36.357452Z'
description: '**DataComp 团队**发布了一款极具竞争力的 **7B 开源数据语言模型**。该模型仅在拥有 240 万亿 token 的海量
  **DCLM-POOL 数据集**中的 **2.5 万亿 token** 上训练而成，展现出优于 FineWeb 的扩展趋势。**OpenAI** 推出了 **GPT-4o
  mini**，这是一款高性价比模型，其 **MMLU 得分为 82%**，性能接近 GPT-4-Turbo，旨在为开发者提供广泛的应用支持。**NVIDIA 和
  Mistral** 联合发布了 **Mistral NeMo 12B** 模型，该模型具备 **128k token 上下文窗口**、FP8 检查点、多语言支持，并采用
  Apache 2.0 开源协议。**DeepSeek** 宣布 **DeepSeek-V2-0628** 成为 LMSYS Chatbot Arena 排行榜上排名最高的开源模型，在编程、数学和高难度提示词（hard
  prompts）方面表现强劲。这些新闻突显了 AI 社区在数据集设计、模型效率以及开源贡献方面的显著进展。'
id: bef0ea8e-4b87-4f2e-a4fb-5fb98fa9c019
models:
- mistral-nemo-12b
- gpt-4o-mini
- deepseek-v2-0628
- mistral-7b
- llama-3
- gemma-2
- qwen-2
original_slug: ainews-apple-dclm-7b-the-best-new-open-weights
people:
- sam-altman
- guillaume-lample
- philschmid
- miramurati
title: DataComp-LM：最优秀的开源数据 7B 模型/基准/数据集。
topics:
- dataset-design
- scaling-laws
- model-benchmarking
- model-performance
- fine-tuning
- multilinguality
- function-calling
- context-windows
- open-source-models
- model-optimization
- cost-efficiency
- benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**240T tokens 是你开始所需的一切。**

> AI 新闻 (2024/7/18-2024/7/19)。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**467** 个频道和 **2305** 条消息）。预计节省阅读时间（以 200wpm 计算）：**266 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

尽管 [HuggingFace 的 SmolLM 发布才不到 4 天](https://buttondown.email/ainews/archive/ainews-to-be-named-5745/)，但它现在已被超越：DataComp 团队（[我们的报道在此](https://www.latent.space/p/neurips-2023-papers)）现已发布了一个“基准”语言模型，在 7B 参数规模上可与 Mistral/Llama3/Gemma/Qwen2 竞争。值得注意的是，它是一个来自 [DataComp-LM 数据集](https://www.datacomp.ai/dclm/index.html#home) 的**开放数据模型**，并且仅用 2.5T tokens 就达到了与其他模型相当的水平：

 
![image.png](https://assets.buttondown.email/images/f6fa149a-8872-4ed7-9e5b-24eda0aef2ef.png?w=960&fit=max)
 

正如你所预料的，秘诀在于数据质量。他们从 `DCLM-POOL` 开始，这是一个源自 Common Crawl 的 240 万亿 tokens 的语料库，也是迄今为止最大的语料库，并提供了针对 5 个规模的**数据集设计扩展趋势 (scaling trends)** 研究：

 
![image.png](https://assets.buttondown.email/images/cd55ffba-94ef-4384-ad8c-27a44bf543fc.png?w=960&fit=max)
 

在每个规模中都有两个赛道：Filtering（必须来自 DCLM-Pool，不含任何外部数据，但可以使用其他模型进行过滤/改写）和 Mixing（允许外部数据）。他们提供了一个“Baseline”过滤示例作为起点：

 
![image.png](https://assets.buttondown.email/images/19bb6cf0-771d-4b9a-ac80-c63cc09f3763.png?w=960&fit=max)
 

关注数据集进展的人可能会好奇 DCLM-Pool 和 Baseline 与 FineWeb（[我们的报道在此](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)）相比如何，前景非常乐观：DCLM 在**每一个**规模上的训练效果都更好。

 
![image.png](https://assets.buttondown.email/images/486c1fe9-5e83-498b-b347-b48a24a871bf.png?w=960&fit=max)
 

这份长达 88 页的论文其余部分包含了大量关于数据质量技术的细节；这是所有参与者（不仅仅是通常报道的 Apple）对开放 LLM 研究做出的巨大贡献。


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**OpenAI 发布 GPT-4o mini 模型**

- **能力**：[@sama](https://twitter.com/sama/status/1813984333352649087) 指出 GPT-4o mini “每百万输入 token 15 美分，每百万输出 token 60 美分，MMLU 为 82%，且速度极快。”他将其与 text-davinci-003 [进行了对比](https://twitter.com/sama/status/1813984927622549881)，称后者“比这个新模型差得多”且“成本高出 100 倍”。
- **定价**：[@gdb](https://twitter.com/gdb/status/1814019156561543658) 强调该模型面向开发者，目标是“将机器智能转化为各领域的积极应用”。[@miramurati](https://twitter.com/miramurati/status/1813996188229894218) 强调 GPT-4o mini “让智能变得更加经济实惠，从而开启了广泛的应用场景。”
- **基准测试**：[@lmsysorg](https://twitter.com/lmsysorg/status/1813999088758673875) 报告称 GPT-4o mini 在 Arena 中进行了测试，显示其性能达到了 GPT-4-Turbo 水平，同时大幅降低了成本。[@polynoamial](https://twitter.com/polynoamial/status/1813986952129167663) 称其为“同尺寸中的佼佼者，尤其是在推理方面”。

**NVIDIA 和 Mistral 发布 Mistral NeMo 12B 模型**

- **能力**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1813949898095534278) 介绍了 Mistral NeMo，这是一个 12B 模型，支持 128k token 上下文窗口、FP8 对齐 checkpoint，在学术、对话和微调基准测试中表现强劲。它是支持 9 种语言的多语言模型，并配备了全新的 Tekken tokenizer。
- **许可**：[@_philschmid](https://twitter.com/_philschmid/status/1813948993489240407) 强调基础版和 instruct 版模型均以 Apache 2.0 许可证发布。instruct 版本支持 function calling。
- **性能**：[@osanseviero](https://twitter.com/osanseviero/status/1813948802644193697) 指出 Mistral NeMo 的性能优于 Mistral 7B，由 NVIDIA 和 Mistral 在 DGX Cloud 上的 3,072 块 H100 80GB GPU 上联合训练而成。

**DeepSeek 发布 DeepSeek-V2-0628 模型**

- **排行榜排名**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1813921111694053644) 宣布 DeepSeek-V2-0628 是 LMSYS Chatbot Arena 排行榜上排名第一的开源模型，总榜排名第 11，在 Hard Prompts 和 Coding 方面排名第 3，在 Longer Query 方面排名第 4，在 Math 方面排名第 7。
- **可用性**：该模型 checkpoint 已在 Hugging Face 上开源，并提供 API。

**趋势与讨论**

- **合成数据**：[@karpathy](https://twitter.com/karpathy/status/1814038096218083497) 建议模型需要先变大再变小，因为需要它们的自动化帮助来“将训练数据重构并塑造成理想的合成（synthetic）格式”。他将其比作 Tesla 的自动驾驶网络，利用之前的模型大规模生成更干净的训练数据。
- **评估担忧**：[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1814049093393723609) 分享了评判 AI 安全评估方案的标准，警告称许多提案在所有方面都失败了，类似于“必须做点什么。这就是‘点什么’。”的谬误。
- **推理局限性**：[@JJitsev](https://twitter.com/JJitsev/status/1813930981637902486) 测试了在奥数竞赛中排名第一的 NuminaMath-7B 模型在基础推理问题上的表现。该模型在简单的变体问题上表现挣扎，揭示了当前衡量推理能力的基准测试存在缺陷。

**梗与幽默**

- [@fabianstelzer](https://twitter.com/fabianstelzer/status/1814023016717664292) 开玩笑说 OpenAI 悄悄发布了原生的 GPT-o “图像”模型，并分享了一个连环画的 prompt 和输出结果。
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1814077033124962596) 幽默地将新加坡的治理方式比作产品管理，即针对新用户留存进行优化。
- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1814002372391153807) 针对一段个人轶事沉思了原则和以牙还牙的升级行为，同时承认由于拥有“FU money”，他免受了大多数影响。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. CPU 推理速度的突破**

- **[NVIDIA CUDA 现在可以通过 "SCALE" 工具包直接在 AMD GPUs 上运行](https://wccftech.com/nvidia-cuda-directly-run-on-amd-gpus-using-scale-toolkit/)** ([Score: 67, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1e6jwf5/nvidia_cuda_can_now_directly_run_on_amd_gpus/)): **NVIDIA CUDA** 现在可以使用开源的 **SCALE (Scalable Compute Abstraction Layer for Execution) 工具包**直接在 **AMD GPUs** 上运行。这一突破允许开发者在无需修改代码的情况下，在 AMD 硬件上执行 CUDA 应用程序，这有可能将 AI 和 HPC 应用的生态系统扩展到 NVIDIA 的硬件垄断之外。由 **StreamHPC** 开发的 SCALE 工具包旨在弥合不同 GPU 架构和编程模型之间的差距。

- **通过 Llamafile 实现 30% 到 500% 的全新 CPU 推理速度提升** ([Score: 70, Comments: 36](https://reddit.com//r/LocalLLaMA/comments/1e6v8qb/new_cpu_inference_speed_gains_of_30_to_500_via/)): **Llamafile** 实现了显著的 **CPU 推理速度提升**，范围从 **30% 到 500%**，在 **Threadripper** 处理器上的表现尤为出色。最近的一次演讲强调，在 Threadripper 上速度从 **300 tokens/second 提升到了 2400 tokens/second**，接近了 **GPU 级别的性能**。虽然没有提到测试的具体模型，但这些改进，加上对 **开源 AI** 的强调，代表了基于 CPU 的推理能力的显著进步。
    - **Prompt Processing 速度至关重要**：Llamafile 的改进主要影响 **Prompt Processing**（提示词处理），而不是 Token 生成。这非常重要，因为 Prompt Processing 是进行**深度理解**的地方，特别是对于涉及大量输入数据的复杂任务。
    - **布尔输出微调**：一些用户报告说，LLM 在处理 True/False 查询并返回 **0 或 1** 时效果良好，特别是在 **Fine-tuning**（微调）之后。一位用户使用 **Gemma 2 9b** 在单张 **4090 GPU** 上，通过特定的分类任务提示词，达到了每秒 **25 次查询**。
    - **CPU vs GPU 性能**：虽然 Llamafile 的 CPU 改进令人印象深刻，但 LLM 推理仍然受限于内存带宽（**Memory-bound**）。**DDR5** 的带宽无法与 **VRAM** 相比，但一些用户发现，在某些应用中，用 **128 GB RAM** 换取高端 GPU 一半的速度是具有吸引力的。


**主题 2. Mistral AI 发布全新开源 LLM**


- **DeepSeek-V2-Chat-0628 权重发布！（Chatbot Arena 中排名第 1 的开源权重模型）** ([Score: 67, Comments: 37](https://reddit.com//r/LocalLLaMA/comments/1e6ba6a/deepseekv2chat0628_weight_release_1_open_weight/)): **DeepSeek-V2-Chat-0628** 已作为性能最强的开源权重模型在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 上发布。该模型在 Chatbot Arena 中总排名 **第 11**，超越了所有其他开源模型，同时在 Coding Arena 和 Hard Prompts Arena 中均取得了 **第 3 名** 的佳绩。

- **[Mistral-NeMo-12B，128k 上下文，Apache 2.0 协议](https://mistral.ai/news/mistral-nemo/)** ([Score: 185, Comments: 84](https://reddit.com//r/LocalLLaMA/comments/1e6cp1r/mistralnemo12b_128k_context_apache_20/)): **Mistral-NeMo-12B** 是一款全新的开源语言模型，具有 **128k 上下文窗口**并采用 **Apache 2.0 许可证**。该模型由 **Mistral AI** 与 **NVIDIA** 合作开发，基于 **NeMo 框架**并使用 **FlashAttention-2** 进行训练。它在各种基准测试中表现出强劲的性能，包括在某些任务上超越了 **Llama 2 70B**，同时保持了 **120 亿参数**的较小规模。

**主题 3. 全面的 LLM 性能基准测试**

- **GGUF 与 EXL2 在多模型及多尺寸下的综合性能基准测试** ([Score: 51, Comments: 44](https://reddit.com//r/LocalLLaMA/comments/1e68k4o/comprehensive_benchmark_of_gguf_vs_exl2/)): **GGUF vs EXL2 性能大对决**。一项针对 **GGUF** 和 **EXL2** 格式在多个模型（**Llama 3 8B**、**70B** 以及 **WizardLM2 8x22B**）上的综合基准测试显示，**EXL2** 在 Llama 模型上略快（**快 3-7%**），而 **GGUF** 在 WizardLM2 上表现更好（**快 3%**）。在配备 **4x3090 GPUs** 的系统上进行的测试表明，两种格式的性能相当，其中 GGUF 提供了更广泛的模型支持和 RAM offloading 能力。
    - **GGUF 追赶上 EXL2**：**GGUF** 的性能有了显著提升，现在在某些情况下已达到或超过了 **EXL2**。此前，**EXL2** 要快 **10-20%**，但最近的测试显示，即使在 prompt 处理方面，两者的速度也旗鼓相当。
    - **量化与模型细节**：GGUF 中的 **Q6_K** 实际上是 **6.56bpw**，而 **EXL2** 的量化则很精确。为了获得更好的质量，建议使用 **5.0bpw** 或 **4.65bpw**，而 **4.0bpw** 更接近 **Q3KM**。不同的架构在不同格式之间的表现可能有所不同。
    - **投机采样 (Speculative Decoding) 与并发请求**：在大型模型前使用 **1B 模型** 可以通过投机采样显著提高速度。关于 **GGUF** 和 **EXL2** 在并发请求场景下的性能差异，仍存在疑问。


- **你目前最常用的 5 个 LLMs 是什么？最近有更换过新的吗？** ([Score: 79, Comments: 39](https://reddit.com//r/LocalLLaMA/comments/1e6qtsa/what_are_your_top_5_current_workhorse_llms_right/)): **5 大常用 LLM 及其潜在的新竞争者**。作者目前最常用的 5 个 LLM 分别是：用于 RAG 任务的 **Command-R**，用于智能和专业回答的 **Qwen2:72b**，用于视觉相关任务的 **Llava:34b**，作为第二意见模型的 **Llama:70b**，以及用于代码相关任务的 **Codestral**。他们表示有兴趣尝试 **Florence**、**Gemma2-27b** 和用于文档检索的 **ColPali**，同时幽默地提到，如果存在以 Steven Seagall 命名的 LLM，他们也会尝试。
    - **ttkciar** 报告称对 **Gemma-2** 模型印象深刻，特别是 **Big-Tiger-Gemma-27B-v1c**，它在 **reason:sally_siblings** 任务中 **五次测试全部正确**。他们还使用 **Dolphin-2.9.1-Mixtral-1x22B** 处理各种任务，并正在尝试使用 **Phi-3** 模型进行 Evol-Instruct 开发。
    - **PavelPivovarov** 分享了他们在有限硬件下的顶级模型：用于大多数任务的 **Tiger-Gemma2 9B**，用于推理的 **Llama3 8B**，用于复杂逻辑和企业写作的 **Phi3-Medium 14B**，以及用于角色扮演的 **Llama-3SOME**。他们表示有兴趣尝试新的 **Gemmasutra** 模型。
    - **ttkciar** 详细分析了 **Phi-3-Medium-4K-Instruct-Abliterated-v3** 在各种任务中的表现。该模型在创意任务中表现出色，在简单的心理理论 (Theory-of-Mind) 问题中推理正确，并且...


**Theme 4. AI 发展与监管挑战**



- **[正如承诺的那样，我开源了我的 Tone Changer - https://github.com/rooben-me/tone-changer-open](https://v.redd.it/4atx6gz21edd1)** ([Score: 96, Comments: 14](https://reddit.com//r/LocalLLaMA/comments/1e6t7ow/as_promised_ive_open_sourced_my_tone_changer/)): **Tone Changer AI 工具开源**。开发者履行了之前的承诺，在 [GitHub](https://github.com/rooben-me/tone-changer-open) 上发布了其 **Tone Changer** 项目的源代码。该工具可能允许用户修改文本输入的语气或风格，尽管帖子中未提供其功能的具体细节。
    - **具有 OpenAI 兼容性的本地部署**：**Tone Changer** 工具完全本地化，并与任何 **OpenAI API** 兼容。它可以在 [GitHub](https://github.com/rooben-me/tone-changer-open) 上获取，并可以通过 [Vercel 托管的演示 (demo)](https://open-tone-changer.vercel.app/) 进行访问。
    - **征求开发细节**：用户对该项目的实现表现出兴趣，要求更新 **README** 以包含运行说明，并询问了 **demo 制作过程**。开发者使用了 **screen.studio** 进行屏幕录制。
    - **功能受到质疑**：一些用户批评了该工具的新颖性，认为它依赖 **prompts** 来改变语气，暗示除了现有的语言模型能力之外，技术创新有限。

- **[Apple 一个月前表示不会在欧盟推出 Apple Intelligence，现在 Meta 也表示由于监管问题，将不会在欧盟提供未来的多模态 AI 模型。](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu)** ([Score: 170, Comments: 95](https://reddit.com//r/LocalLLaMA/comments/1e6vbqe/apple_stated_a_month_ago_that_they_wont_launch/)): **Apple** 和 **Meta** 由于监管担忧，正对**欧盟**保留其 **AI 模型**。Apple 一个月前宣布不会在欧盟推出 **Apple Intelligence**，现在 Meta 也紧随其后，表示不会在该地区提供未来的**多模态 AI 模型**。这些决定凸显了 **AI 创新**与**欧盟法规**之间日益紧张的关系，可能会导致欧洲用户在 AI 技术可用性方面出现重大差距。
    - • **-p-e-w-** 认为欧盟的监管是有益的，可以防止 **FAANG** 公司垄断 AI 市场并压制竞争。他们建议**禁止**这些公司进入欧盟 AI 市场，以限制其权力。
    - • 关于 **GDPR** 合规性的讨论揭示了不同的观点。一些人认为对于诚信经营的企业来说这很容易，而另一些人则强调，与资源丰富的跨国公司相比，**初创公司**和小型企业面临着巨大挑战。
    - • 批评者指责这些公司**虚伪**，指出它们一方面倡导“AI 安全”，另一方面却抵制实际的监管。一些人认为这是企业试图**挟持政府**以降低对公民的保护，而另一些人则认为欧盟的监管可能会阻碍创新。

## 所有 AI Reddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. AI 在医学执业考试中表现优于人类**

- [/r/singularity] **[ChatGPT 轻松通过美国执业医师资格考试，准确率达 98%。医生的平均水平仅为 75%。](https://v.redd.it/gi5xuyw9zedd1)** ([Score: 328, Comments: 146](https://reddit.com//r/singularity/comments/1e6wgbs/chatgpt_aces_the_us_medical_licensing_exam/)): **ChatGPT 在美国执业医师资格考试中的表现优于人类医生**，实现了惊人的 **98% 准确率**，而医生的平均准确率为 **75%**。这一令人印象深刻的表现展示了 AI 彻底改变医学教育和实践的潜力，引发了关于 AI 在医疗保健中未来角色以及调整医学培训课程必要性的讨论。
    - • **ChatGPT** 在美国执业医师资格考试中 **98% 的准确率**（医生为 **75%**）引发了人们对 AI 对医疗职业影响的担忧。一些人认为 AI 可以减少**每年 795,000 例**因诊断错误导致的死亡，而另一些人则质疑该考试与现实世界医疗实践的相关性。
    - • 专家预测 AI 最初将与人类医生并肩工作，特别是在放射科等专业领域。**保险公司**可能会强制要求使用 AI，以捕捉人类遗漏的信息，从而可能提高诊断速度和准确性。
    - • 批评者认为 AI 的表现可能是由于**“在测试集上进行了预训练”**，而非真正的理解。一些人认为考试结构可能无法充分测试复杂的推理能力，而另一些人则指出人类医生也会通过研究过去的考试来做准备。


**主题 2. OpenAI 的 GPT-4o-mini：更实惠、更高效的 AI 模型**

- [/r/singularity] **[GPT-4o-mini 比 GPT 3.5 Turbo 便宜 2 倍](https://i.redd.it/yr8e8te0abdd1.png)** ([Score: 363, Comments: 139](https://reddit.com//r/singularity/comments/1e6gw80/gpt4omini_is_2_times_cheaper_than_gpt_35_turbo/)): **GPT-4o-mini** 是一款新的 AI 模型，现在的**成本仅为 GPT-3.5 Turbo 的一半**。这款由 **Anthropic** 开发的模型提供了与 GPT-3.5 Turbo 相当的性能，但价格显著降低，有可能使先进的 AI 功能被更广泛的用户和应用所使用。

- [/r/singularity] **[GPT-4o mini：推进高性价比智能](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)** ([Score: 238, Comments: 89](https://reddit.com//r/singularity/comments/1e6gffv/gpt4o_mini_advancing_costefficient_intelligence/)): **GPT-4o mini** 是由 **Anthropic** 开发的一款新 AI 模型，旨在通过以 **GPT-4** 极小部分的成本提供类似的功能，来实现**高性价比智能**。该模型旨在为开发者和企业提供更易于获取且更实惠的选择，从而可能推动先进 AI 技术的更广泛采用。虽然未提供具体的性能指标和定价细节，但对成本效率的关注表明，在使强大的语言模型对更广泛的应用更具经济可行性方面迈出了重要一步。

- [/r/singularity] **[OpenAI 发布其迄今为止最强大模型的迷你版本](https://www.cnbc.com/2024/07/18/openai-4o-mini-model-announced.html)** ([分数: 378, 评论: 222](https://reddit.com//r/singularity/comments/1e6d4p5/openai_debuts_mini_version_of_its_most_powerful/)): **OpenAI** 推出了 **GPT-4 Turbo**，这是其最先进语言模型的更小、更高效版本。该新模型提供 **128k context**，旨在为开发者提供 **更实惠** 的选择，定价为每 1,000 **input tokens** 0.01 美元，每 1,000 **output tokens** 0.03 美元。**GPT-4 Turbo** 还包含了截至 **2023年4月** 的更新知识，并支持 **JSON mode** 等新功能以实现结构化输出。

**主题 3. AI 生成视觉与音频内容的进展**

- [/r/singularity] **[新语音模式即将推出](https://i.redd.it/c53opypphcdd1.jpeg)** ([分数: 279, 评论: 106](https://reddit.com//r/singularity/comments/1e6mog7/new_voice_mode_coming_soon/)): **新的语音合成模式** 即将发布，扩展了 **AI** 生成语音的能力。这一即将推出的功能有望提升合成语音的质量和通用性，可能为各种应用提供更自然且可定制的音频输出。

- [/r/singularity] **[Unanswered Oddities 第 1 集（一部完全由 AI 生成视频的 AI 辅助电视剧）](https://v.redd.it/60cr3fko7add1)** ([分数: 330, 评论: 41](https://reddit.com//r/singularity/comments/1e6c1d4/unanswered_oddities_ep_1_an_aiassisted_tv_series/)): **《Unanswered Oddities》** 是一部 **AI-assisted** 电视剧，采用 **完全 AI-generated video**，目前已发布第一集。该系列探索了 **未解现象** 和 **神秘事件**，利用 **AI technology** 创作剧本和视觉效果，推动了娱乐行业中 **AI-driven content creation** 的边界。

- [/r/singularity] **[Pet Pixels Studio 的 Kling AI 示例](https://v.redd.it/rrkxtpshzbdd1)** ([分数: 287, 评论: 25](https://reddit.com//r/singularity/comments/1e6k8ry/example_of_kling_ai_by_pet_pixels_studio/)): **Pet Pixels Studio** 展示了其 **Kling AI** 技术，这似乎是一个用于生成或处理宠物相关图像的 **AI** 系统。虽然没有提供关于该 **AI** 能力或实现的具体细节，但标题表明这是 **Kling AI** 输出或功能的示例或演示。

---

# AI Discord 摘要

> 摘要之摘要的摘要

## GPT4O (gpt-4o-2024-05-13)


**1. LLM 进展**

- **Llama 3 发布在即**：传闻拥有 **4000 亿**参数的 **Llama 3** 将在 4 天内发布，引发了社区内的兴奋和猜测。
   - 即将到来的发布引发了关于其潜在影响和能力的众多讨论。
- **GPT-4o mini 提供高性价比性能**：**GPT-4o mini** 被视为 3.5 Turbo 更便宜、更快的替代方案，正如 [GitHub 上](https://github.com/openai/simple-evals)所指出的，其速度快约 **2 倍**，价格便宜 **60%**。
   - 然而，它缺乏图像支持，且在基准测试中的得分低于 **GPT-4o**，凸显了其局限性。
    


**2. 模型性能优化**

- **DeepSeek-V2-Chat-0628 登顶 LMSYS 排行榜**：[DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 是一款拥有 **236B 参数**的模型，在 [LMSYS Chatbot Arena Leaderboard](https://chat.lmsys.org) 上位列开源模型第一。
   - 它占据了多个顶尖位置：总榜第 11 名、困难提示词（Hard Prompts）第 3 名、代码（Coding）第 3 名、长查询（Longer Query）第 4 名、数学（Math）第 7 名。
- **Mojo vs JAX：基准测试之争**：尽管 JAX 针对多核系统进行了优化，但 **Mojo** 在 CPU 上的表现优于 **JAX**。讨论表明，Mojo 的编译器可见性赋予了其性能优势。
   - **MAX** 与 **openXLA** 的对比显示，作为一种惰性计算图构建器，MAX 具有优势，提供了更多的优化机会和广泛的影响。
    


**3. 开源 AI 框架**

- **SciPhi 开源用于知识图谱的 Triplex**：SciPhi 正在[开源](https://www.sciphi.ai/blog/triplex) **Triplex**，这是一款用于知识图谱构建的顶尖 LLM，可显著降低 98% 的成本。
   - Triplex 可与 SciPhi 的 R2R 配合使用，直接在笔记本电脑上构建知识图谱，其表现优于经过 Few-shot-prompted 的 **GPT-4**，而推理成本仅为后者的 1/60。
- **Open WebUI 功能强大**：[Open WebUI](https://github.com/open-webui/open-webui) 拥有 TTS、RAG 以及无需 Docker 即可访问互联网等丰富功能，令用户着迷。
   - 在 Windows 10 上使用 Open WebUI 的良好体验，引发了人们将其性能与 **Pinokio** 进行对比的兴趣。
    


**4. 多模态 AI 创新**

- **Text2Control 实现自然语言指令**：[Text2Control](https://europe.naverlabs.com/text2control) 方法使 Agent 能够通过视觉语言模型（Vision-Language Models）解释自然语言指令来执行新任务。
   - 该方法在零样本泛化（Zero-shot Generalization）方面优于多任务强化学习基准，并提供[交互式演示](https://europe.naverlabs.com/text2control)供用户探索其功能。
- **Snowflake Arctic Embed 1.5 提升检索系统扩展性**：Snowflake 推出了 **Arctic Embed M v1.5**，通过极小的嵌入向量（Embedding Vectors），使检索系统的扩展性提升了高达 **24 倍**。
   - [Daniel Campos 关于此更新的推文](https://x.com/spacemanidol/status/1813968340744020252)强调了性能指标的显著增强。
    


**5. AI 社区工具**

- **ComfyUI 赢得 Stable Diffusion 新手的青睐**：成员们建议将 [ComfyUI](https://comfy.icu/) 作为 Stable Diffusion 初学者的优秀 UI，强调了其灵活性和易用性。
   - 此外，建议观看 Scott Detweiler 的 [YouTube 教程](https://www.youtube.com/@sedetweiler)以获得详尽的指导。
- **GPTs Agents 展现出自我意识**：在 **GPTs Agents** 上进行的一项实验旨在评估其自我意识，特别是过程中避免使用网页搜索功能。
   - 测试结果引发了关于在没有外部数据源的情况下，具有自我意识的 AI 系统的实际意义和潜在局限性的讨论。

## GPT4OMini (gpt-4o-mini-2024-07-18)


**1. 最近的模型发布与性能表现**

- **Mistral NeMo 与 DeepSeek 模型亮相**：Mistral 发布了具有 **128k token 上下文长度**的 **[NeMo 12B 模型](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407)**，展示了多语言能力和工具支持；同时 **DeepSeek-V2-Chat-0628** 在 LMSYS 排行榜上名列前茅。
   - 这些模型强调了性能和效率的提升，其中 DeepSeek 拥有 **236B 参数**，在开源模型中排名第一。
- **GPT-4o Mini 对比 Claude 3 Haiku**：全新的 **GPT-4o mini** 比 GPT-3.5 Turbo 快约 **2 倍**且便宜 **60%**，尽管其基准测试分数低于 **Claude 3 Haiku**，但仍是一个极具吸引力的替代方案。
   - 用户正在讨论潜在的替代方案，对于 mini 在各种任务中的表现意见不一。
- **Apple 发布 DCLM 7B 模型**：Apple 发布的 **[DCLM 7B 模型](https://huggingface.co/apple/DCLM-7B)** 表现优于 **Mistral 7B**，并展示了完全开源的训练代码和数据集。
   - 此举引发了关于其对开源 AI 模型竞争格局影响的讨论。
    


**2. AI 工具与社区资源**

- **Open WebUI 增强功能**：**[Open WebUI](https://github.com/open-webui/open-webui)** 现在包含 TTS 和 RAG 等功能，允许用户在不使用 Docker 的情况下与其模型交互，提升了可访问性和易用性。
   - 用户反馈在 Windows 10 上运行体验良好，认为其性能优于 **Pinokio**。
- **适合初学者的 ComfyUI**：成员们推荐将 **[ComfyUI](https://comfy.icu/)** 作为 Stable Diffusion 初学者的优秀用户界面，强调了其灵活性和易用性。
   - 建议那些寻求全面指导的用户观看 YouTube 上 Scott Detweiler 的教程。
    


**3. 训练技术与模型微调**

- **提升 Transformer 的泛化能力**：一篇 **[arXiv 论文](https://arxiv.org/abs/2405.15071)** 指出，在饱和点之外继续训练 Transformer 可以增强其泛化能力，特别是对于域外（out-of-domain）任务。
   - 这种方法有助于防止灾难性遗忘（catastrophic forgetting），使其成为未来模型训练的关键策略。
- **Mistral-12b 的微调挑战**：用户报告了 **Mistral-12b** 的配置问题，特别是关于投影权重（projection weights）的大小不匹配，需要从源码安装 Transformers 库才能修复。
   - 关于微调策略的讨论表明，需要对训练设置进行特定调整以优化性能。
    


**4. AI 中的数据隐私与安全**

- **CrowdStrike 停机事件的影响**：最近的 **CrowdStrike** 更新导致了全球范围的停机，影响了多个行业，并引发了关于云端安全服务可靠性的讨论。
   - 该事件引起了人们对技术基础设施中数据隐私和运营韧性的担忧。
- **企业对分享敏感数据的犹豫**：对数据隐私的担忧使企业对与第三方分享敏感信息持谨慎态度，优先考虑内部控制而非外部交换。
   - 这一趋势突显了数据安全在 AI 应用中日益增长的重要性。
    


**5. 知识图谱与检索增强生成 (RAG) 的进展**

- **Triplex 彻底改变知识图谱**：**[Triplex 模型](https://huggingface.co/SciPhi/Triplex)** 使知识图谱构建成本降低了 **98%**，其表现优于 GPT-4，而成本仅为后者的 1/60。
   - Triplex 促进了使用 SciPhi 的 R2R 平台进行本地图谱构建，增强了检索增强生成（RAG）方法。
- **用于知识图谱的 R2R 平台**：**R2R 平台**支持可扩展的、生产级的检索增强生成应用，集成了多模态支持和自动关系提取。
   - 成员们强调了其在从非结构化数据创建知识图谱方面的有效性，展示了实际应用案例。
    


---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Hermes 2.5 表现优于 Hermes 2**：在添加了[代码指令示例](https://link.to.examples)后，**Hermes 2.5** 在各种基准测试中的表现似乎优于 **Hermes 2**。
   - Hermes 2 在 MMLU 基准测试中得分为 **34.5**，而 Hermes 2.5 得分为 **52.3**。
- **Mistral 在扩展超过 8k 时遇到困难**：成员们表示，如果不进行持续预训练，**Mistral** 无法扩展到 8k 以上，且[这是一个已知问题](https://link.to.issue)。
   - 他们指出 *mergekit* 和 *frankenMoE finetuning* 是性能突破的下一个前沿。
- **Mistral 发布 NeMo 12B 模型**：Mistral 发布了 [NeMo](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit)，这是一个 **120 亿参数模型**，展示了多语言能力和原生工具支持。
   - *该模型恰好可以运行在免费的 Google Colab GPU 实例中*，你可以[在此访问](https://unsloth.ai/blog/mistral-nemo)。
- **深入探讨 CUDA bf16 问题及修复**：多位用户报告了在 RTX A4000 和 T4 等**不同 GPU 型号**上与 **bf16** 支持相关的错误，阻碍了模型执行。
   - 问题被确定为 **torch.cuda.is_bf16_supported() 返回 False**，Unsloth 团队随后已修复此问题。
- **SciPhi 开源用于知识图谱的 Triplex**：SciPhi 正在[开源](https://www.sciphi.ai/blog/triplex) **Triplex**，这是一个用于知识图谱构建的尖端 LLM，可显著降低 98% 的成本。
   - Triplex 可与 SciPhi 的 R2R 配合使用，直接在笔记本电脑上构建知识图谱，其表现优于少样本提示的 **GPT-4**，且推理成本仅为 1/60。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 赢得 Stable Diffusion 新手的青睐**：成员们建议将 [ComfyUI](https://comfy.icu/) 作为 Stable Diffusion 新手的理想 UI，强调了其灵活性和易用性。
   - 此外，建议观看 Scott Detweiler 的 [YouTube 教程](https://www.youtube.com/@sedetweiler)以获取详尽指导。
- **在 AI 任务中 NVIDIA 胜过 AMD**：讨论中的共识表明，由于更好的支持和更少的故障排除，在 Stable Diffusion 方面，NVIDIA GPU 优于 AMD。
   - 尽管 AMD 提供了更多 VRAM，但 NVIDIA 因其更广泛的兼容性而受到赞誉，尤其是在 Linux 环境中，尽管偶尔会出现驱动程序问题。
- **Stable Diffusion 模型：没有万能的选择**：关于最佳 Stable Diffusion 模型的讨论得出结论，选择取决于 VRAM 和用户的具体需求，推荐 SDXL 是因为其更大的规模和能力。
   - 提到 SD3 因采用了新的 VAE 而具有卓越的图像质量，同时指出它目前主要在 ComfyUI 中受支持。
- **让 Stable Diffusion 更有艺术感的技巧**：一位成员寻求关于如何让图像看起来更有艺术感、减少超写实感的建议，抱怨高清、高对比度输出占据主导地位。
   - 建议包括使用艺术风格的 LoRAs 以及尝试不同的模型，以实现理想的数字绘画效果。
- **寻找 Reddit 之外的 AI 新闻替代来源**：一位成员对 Reddit 的封禁和 Twitter 的审查表示沮丧，正在寻找 AI 新闻的替代来源。
   - 建议包括在 Twitter 上关注科学界以获取最新的论文和进展，尽管存在感知上的地区和基于用户的审查问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **CrowdStrike BSOD 快速帮助**：来自 **CrowdStrike** 的一个错误文件导致了全球范围的大规模 BSOD（蓝屏死机），影响了全球数百万个系统。CrowdStrike 的 Overwatch 总监发布了一个[热修复补丁](https://youtu.be/E8RQVx2gBFc?si=D2hdEW9k9iK0U9Vl)来打破 BSOD 循环。
   - 该问题引发了大量关于后续影响以及预防未来类似事件措施的讨论。
- **Hugging Face API 问题**：社区中多位用户讨论了 **Meta-Llama-3-70B-Instruct API** 的问题，包括关于不支持的模型配置的错误消息。
   - 普遍认为 **Hugging Face 基础设施存在问题**，特别是影响了模型处理速度，用户指出在停机后最近已趋于稳定。
- **模型发布潮席卷动态**：一天之内发布了多个重要模型：**DeepSeek** 的顶级开源 lmsys 模型、**Mistral 12B**、**Snowflake** 的嵌入模型等。[查看推文](https://x.com/osanseviero/status/1814068082060460409)获取完整列表。
   - *Osanseviero 评论道*：“🌊 送给那些被今天发布的内容淹没的人们，” 总结了社区对大量更新涌现的情绪。
- **神经网络中的技术预告**：[Circuits 线程](https://distill.pub/2020/circuits/)提供了一种实验性格式，深入探讨神经网络的内部运作，涵盖了如 **Curve Detectors**（曲线检测器）和 **Polysemantic Neurons**（多语义神经元）等创新发现。
   - 这种理解神经机制的引人入胜的方法引发了关于概念和实际影响的热烈讨论。
- **AI Comic Factory 增强功能**：[AI Comic Factory](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory) 进行了重大更新，现在默认包含对话气泡，提升了漫画创作体验。
   - 这项新功能利用 AI 进行提示词生成和对话分割，通过视觉指标改善了叙事，甚至可以适应恐龙等非人类角色。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **通过受睡眠启发的动力学克服 ANN 中的灾难性遗忘**：Maxim Bazhenov 等人的实验表明，ANN 中的[类睡眠阶段](https://vchs.ucsd.edu/blog/2022/12/sleep-derived-how-artificial-neural-networks-can-avoid-catastrophic-forgetting.html)有助于减少灾难性遗忘，研究结果发表在 [Nature Communications](https://www.nature.com/articles/s41467-022-34938-7) 上。
   - ANN 中的睡眠涉及使用局部无监督 Hebbian 塑性规则和噪声输入的离线训练，帮助 ANN 恢复之前遗忘的任务。
- **Opus Instruct 3k 数据集助力多轮指令微调**：一位成员分享了 Hugging Face 上的 [Opus Instruct 3k 数据集链接](https://huggingface.co/datasets/kalomaze/Opus_Instruct_3k)，其中包含约 250 万个 token 的通用多轮指令微调数据，风格类似于 **Claude 3 Opus**。
   - teknium 以正面评价认可了该数据集的重要性。
- **GPT-4o Mini 在编程基准测试中与 GPT-3.5-Turbo 竞争**：在一个[编程基准测试](https://aider.chat/docs/leaderboards/)中，**GPT-4o Mini** 的表现与 **GPT-3.5-Turbo** 持平，尽管其宣传的 HumanEval 分数提高了用户的期望。
   - 一位用户对过度炒作的性能指标表示不满，推测 OpenAI 在基准测试数据上对其进行了训练。
- **Triplex 将 KG 构建成本降低了 98%**：Triplex 是 [SciPhi.AI](https://www.sciphi.ai) 对 Phi3-3.8B 进行微调的版本，在从非结构化数据创建知识图谱（KG）方面，其表现优于 GPT-4，而成本仅为后者的 1/60。
   - 它支持使用 SciPhi 的 R2R 平台进行本地图谱构建，显著降低了开支。
- **Mistral-Nemo-Instruct GGUF 转换困难备受关注**：一位成员在将 **Mistral-Nemo-Instruct** 转换为 **GGUF** 时遇到了困难，原因是 BPE 词表问题和缺失 tokenizer.model 文件。
   - 尽管提交了支持 Tekken 分词器的 PR，转换脚本仍然无法工作，令人非常沮丧。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o mini 提供高性价比的性能**：**GPT-4o mini** 被视为 3.5 Turbo 更便宜、更快速的替代方案，正如 [GitHub](https://github.com/openai/simple-evals) 上所述，其速度快约 **2 倍**，价格便宜约 **60%**。
   - 然而，它缺乏图像支持，且在基准测试中的得分低于 **GPT-4o**，这凸显了它的一些局限性。
- **Crowdstrike 停机事件扰乱各行各业**：一次 **Crowdstrike** 更新导致了全球范围的停机，影响了航空公司、银行和医院等行业，机器需要手动解锁。
   - 这主要影响了 **Windows 10** 用户，使得修复过程缓慢且成本高昂。
- **GPT-4o 的基准测试优势引发争议**：与 **GPT-4 Turbo** 相比，**GPT-4o** 在基准测试中得分更高，但其有效性因使用场景而异 [来源](https://github.com/openai/simple-evals)。
   - 由于这些变数，社区对于谁具有绝对优势尚未达成共识，这凸显了特定应用需求的重要性。
- **4o mini 的微调功能指日可待**：成员们预计 **4o mini** 的微调功能将在大约 **6 个月** 内推出。
   - 这一潜在的增强功能可能会进一步提升其在特定应用中的实用性和性能。
- **在代码片段中请求玻璃拟态 UI**：用户正寻求使用 HTML、CSS 和 JavaScript 创建一个具有**玻璃拟态（glassmorphic）UI** 的代码片段库，并配有动画渐变背景。
   - 期望的功能包括管理代码片段（添加、查看、编辑和删除），并具备跨浏览器兼容性和响应式设计。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 提升调试体验**：Mojo 优先开发高级调试工具，增强了在 GPU 上进行机器学习任务的调试体验。[了解更多](https://www.modular.com/blog/debugging-in-mojo)。
   - Mojo 的扩展允许在 VS Code 中进行无缝设置，并计划集成 [LLDB-DAP](https://lldb.llvm.org/resources/lldbdap.html) 以实现从 CPU 到 GPU 代码的单步执行。
- **Mojo vs JAX：基准测试之战**：尽管 JAX 针对多核系统进行了优化，但 **Mojo** 在 CPU 上的表现优于 **JAX**。讨论表明，Mojo 的编译器可见性赋予了其性能优势。
   - **MAX** 与 **openXLA** 的对比显示，作为惰性计算图构建器，MAX 具有优势，提供了更多的优化机会和广泛的影响。
- **Mojo 的低级编程之旅**：一位从 Python 转向 Mojo 的用户考虑到 Mojo 目前文档较少，曾考虑学习 C、CUDA 和 Rust。社区的回应集中在“复杂性的渐进式披露（Progressive Disclosure of Complexity）”。
   - 讨论鼓励记录学习历程以帮助塑造 Mojo 的生态系统，并建议在类型中使用 `InlineArray` 处理 FloatLiterals。
- **Mojo 中的 Async IO API 标准**：一场讨论强调了在 Mojo 中建立 Async IO API 标准的必要性，通过有效处理缓冲区来支持更高性能的模型。对话借鉴了 Rust 在 Async IO 方面面临的挑战。
   - 社区考虑避免在注重性能的库和主流库之间产生分裂，旨在实现无缝集成。
- **Mojo Nightly 更新亮点**：[Mojo nightly 更新 2024.7.1905](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 引入了一个新的标准库函数 `Dict.setdefault(key, default)`。查看 [原始差异（raw diff）](https://github.com/modularml/mojo/compare/bb7db5ef55df0c48b6b07850c7566d1ec2282891...f8d9214ac31da76bb679f867f57b255b65d9a31a) 了解详细变更。
   - 贡献者会议可能会与社区会议分开，以便更好地与 Modular 的工作保持一致；标准库的贡献在集成前将通过孵化器进行 API 和受欢迎程度的审核。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mistral Nvidia 合作引发关注**：[Mistral Nvidia 合作](https://mistral.ai/news/mistral-nemo/)推出了 Mistral-Nemo 12B，提供大上下文窗口和顶尖性能，但目前 LM Studio 尚不支持。
   - 需要 llama.cpp 提供 **Tokenizer 支持**才能使 Mistral-Nemo 兼容。
- **Open WebUI 的丰富功能备受瞩目**：[Open WebUI](https://github.com/open-webui/open-webui) 拥有 TTS、RAG 以及无需 Docker 的互联网访问等广泛功能，令用户着迷。
   - 在 Windows 10 上使用 Open WebUI 的良好体验，引发了将其性能与 **Pinokio** 进行对比的兴趣。
- **DeepSeek-V2-Chat-0628 登顶 LMSYS 排行榜**：[DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 是一款拥有 **236B 参数**的模型，在 [LMSYS Chatbot Arena 排行榜](https://chat.lmsys.org)上排名开源模型第一。
   - 它占据了多个领先位置：总榜第 11，Hard Prompts 第 3，Coding 第 3，Longer Query 第 4，Math 第 7。
- **使用 NVidia Tesla P40 的复杂性**：用户在 Windows 上运行 NVidia Tesla P40 时评价褒贬不一；使用了数据中心和 Studio RTX 驱动，但性能表现各异。
   - 强调了 Tesla P40 与 Vulcan 的兼容性问题，建议进行多次安装并启用虚拟化。
- **TSMC 预测 AI 芯片供应延迟**：TSMC 首席执行官预测，由于封装瓶颈和高需求，AI 芯片供应在 2025-2026 年之前不会达到平衡。
   - 正如[该报告](https://www.theregister.com/2024/07/18/tsmc_ceo_predicts_ai_chip/)中所述，预计海外扩张将继续。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Llama 3 发布在即**：传闻拥有 **400B** 参数的 **Llama 3** 将在 4 天内发布，引发了社区的兴奋和猜测。
   - 即将到来的发布引发了关于其潜在影响和能力的众多讨论。
- **自博弈偏好优化引起兴趣**：**SPPO (Self-Play Preference Optimization)** 因其潜力而受到关注，但对其在几次迭代后的长期有效性存在怀疑。
   - 对于当前的方法论在广泛部署和使用后是否能经得起考验，意见不一。
- **苹果开源 DCLM 7B 模型**：苹果发布了 **DCLM 7B** 模型，该模型超越了 **Mistral 7B** 且完全开源，包括训练代码和数据集。
   - 此次发布引起了轰动，[VikParuchuri 的 GitHub 个人资料](https://github.com/VikParuchuri)展示了 **90 个仓库**，[官方推文](https://x.com/casper_hansen_/status/1814269340100751382)也强调了此次开源。
- **Snowflake Arctic Embed 1.5 提升检索系统可扩展性**：Snowflake 推出了 **Arctic Embed M v1.5**，通过微小的 Embedding 向量，为检索系统提供高达 **24 倍的可扩展性提升**。
   - [Daniel Campos 关于此更新的推文](https://x.com/spacemanidol/status/1813968340744020252)强调了性能指标的显著增强。
- **Texify 与 Mathpix 的功能对比**：有人提出了 **Texify** 与 **Mathpix** 在功能方面的对比，但未提供详细解答。
   - 对话凸显了关于这些工具在各种用例中有效性的持续争论。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Nvidia 受反垄断法影响开源内核模块**：据推测，**Nvidia** 开源内核模块的决定可能受到 **US anti-trust laws** 的影响。
   - 一位用户建议，**maintaining kernel modules** 并非 Nvidia 的核心业务，开源可以在不需要高技能开发人员的情况下提高兼容性。
- **Float8 权重在 PyTorch 中引入来自 BF16 的动态转换**：成员们讨论了在 PyTorch 中将以 BF16 存储的权重转换为 FP8 进行 matmul，参考了 [float8_experimental](https://github.com/pytorch-labs/float8_experimental)。
   - 还有人对为 FP8 权重更新实现 **stochastic rounding** 感兴趣，这可能得到 Meta 计算资源的支持。
- **Tinygrad 悬赏任务引发褒贬不一的反应**：关于为 **tinygrad** 悬赏任务（如 [splitting UnaryOps.CAST](https://github.com/tinygrad/tinygrad/pull/4487)）做贡献的讨论指出，有些人认为报酬与付出的努力不成正比。
   - 一位成员悬赏 **$500** 为 tinygrad 添加 FSDP 支持，这被认为太低了，潜在的实现者至少需要一到两周时间。
- **Yuchen 的 7.3B 模型训练实现线性扩展**：Yuchen 使用 **karpathy 的 llm.c** 和 32 块 H100 GPU 训练了一个 **7.3B model**，达到了 **327K tokens/s** 的速度和 **46.7%** 的 MFU。
   - 为了处理由于模型参数巨大导致的整数溢出，需要将 'int' 更改为 'size_t'。
- **HQQ+ 2-bit Llama3-8B-Instruct 模型发布**：一个新模型 [HQQ+ 2-bit Llama3-8B-Instruct](https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq) 发布，它使用 **BitBlas backend** 和 64 group-size 量化以保持质量。
   - 尽管 Llama3-8B 的低比特量化存在挑战，该模型仍兼容 **BitBlas** 和 `torch.compile` 以实现快速推理。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro 用户报告搜索质量下降**：一些成员，特别是使用 **Claude Sonnet 3.5** 的成员，注意到过去 8-9 天内 **Pro searches** 的质量显著下降。
   - 这个问题已在讨论中提出，但尚未确定明确的解决方案或原因。
- **GPT-4o mini 将取代 Claude 3 Haiku？**：围绕在 Perplexity 中用更便宜、更智能的 **GPT-4o mini** 替换 **Claude 3 Haiku** 的想法展开了积极讨论。
   - 尽管 **GPT-4o mini** 具有极具吸引力的属性，但目前仍在使用 **Claude 3 Haiku**。
- **YouTube Music 推出智能电台**：一场讨论重点介绍了 [YouTube Music's Smart Radio](https://www.youtube.com/embed/5lC4KwPFvaE)，其特点是创新的内容交付和新的音乐发现工具。
   - *YouTube Music* 因智能策划播放列表并适应用户偏好而受到称赞。
- **Dyson 首次推出高科技耳机**：Dyson 的新款 [high-tech headphones](https://www.perplexity.ai/search/t6-3al250w-fuse-nc_aBqo8SKm15tV1Kvk3pQ) 因集成了先进的降噪和空气过滤技术而受到关注。
   - 成员们对该产品的双重功能和时尚设计发表了评论。
- **寻求 Perplexity 的 RAG API 访问权限**：一位成员指出，在发送关于其企业 RAG API 的电子邮件后没有收到回复，正在寻求进一步协助以获取访问权限。
   - 这表明存在持续的沟通挑战，以及对企业级 API 解决方案未被满足的需求。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral AI 发布两款新模型**：Daun.ai 推出了 [Mistral Nemo](https://openrouter.ai/models/mistralai/mistral-nemo)，这是一个拥有 12B 参数、支持 128k token 上下文长度的多语言 LLM。
   - [Codestral Mamba](https://openrouter.ai/models/mistralai/codestral-mamba) 也已发布，该模型拥有 7.3B 参数和 256k token 上下文长度，专为代码和推理任务设计。
- **L3-Euryale-70B 价格大幅下调 60%**：[L3-Euryale-70B](https://openrouter.ai/models/sao10k/l3-euryale-70b) 经历了 60% 的巨额降价，使其在各种应用中的使用更具吸引力。
   - 此外，[Cognitivecomputations 发布了 Dolphin-Llama-3-70B](https://openrouter.ai/models/cognitivecomputations/dolphin-llama-3-70b)，这是一款极具竞争力的新模型，承诺提供更强的指令遵循和对话能力。
- **LLM-Draw 集成 OpenRouter API 密钥**：[LLM-Draw](https://github.com/RobinVivant/llm-draw) 应用现在支持 **OpenRouter API 密钥**，并利用 **Sonnet 3.5 自我审核模型**。
   - 该应用可作为 **Cloudflare page** 通过 Next.js 部署，目前已提供 [在线版本](https://llm-draw.pages.dev)。
- **Gemma 2 出现重复问题**：用户报告了 **Gemma 2 9B** 的内容重复问题，并寻求缓解该问题的建议。
   - 有建议提出使用 **CoT** (Chain of Thought) 提示词以获得更好的性能。
- **Mistral NeMo 增加韩语支持**：消息指出 **Mistral NeMo** 已扩展其语言支持以包含韩语，增强了其多语言能力。
   - 用户注意到它在 **英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文、日语、韩语、阿拉伯语和印地语** 方面表现强劲。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GPT-4o Mini 的多功能表现**：[GPT-4o mini](https://x.com/paulgauthier/status/1814014867361374610?s=46) 在 Aider 的代码编辑基准测试中与 GPT-3.5 持平，但在处理大文件的代码 diffs 时表现吃力。
   - [该模型](https://x.com/simonw/status/1814163501880893794?s=46) 提供了极具成本效益的文本生成，但图像输入成本仍然较高，这促使用户考虑 **Claude 3 Haiku** 和 **Gemini 1.5 Flash** 等替代方案。
- **OpenAI 面临新的安全漏洞**：[OpenAI 的新安全机制](https://x.com/elder_plinius/status/1814023961535295918?s=46) 被轻易绕过，导致 GPT-4o-mini 生成有害内容，暴露了重大漏洞。
   - [内部评估](https://fxtwitter.com/corbtt/status/1814056457626862035?s=61) 显示 GPT-4o mini 可能存在过拟合问题，额外信息虚高了其得分，凸显了评估设置中的潜在缺陷。
- **Gemma 2 的 Logit Capping 令人惊讶**：成员们讨论了 **Gemma 2** 中 [移除软 logit capping](https://discord.com/channels/1179127597926469703/1179208129083363358/1263650433604259914) 的情况，并辩论是否需要重新训练以应对其影响。
   - 一些成员发现该模型在没有进行大规模重新训练的情况下表现良好，这令人吃惊，挑战了关于 logit capping 调整的普遍预期。
- **MosaicML 独特的宝剑传统**：MosaicML 的员工会收到宝剑作为独特传统的一部分，这在关于未来 [潜在采访](https://discord.com/channels/1179127597926469703/1183121795247779910/1263730787874770944) 的讨论中被提及。
   - 据报道，人力资源和法律团队曾表示反对，但传闻甚至连 MosaicML 的法律团队也参与其中。
- **Sara Hooker 批评美国 AI 法案**：一位成员分享了 [YouTube 视频](https://www.youtube.com/watch?v=dBZp47999Ko)，内容是 **Sara Hooker** 批评美国 AI 法案中的算力阈值，引发了社区关注。
   - 她在社区中的活跃（近期发表的论文也证明了这一点）突显了关于监管框架及其对未来 AI 发展影响的持续讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Z-Loss 正则化项探讨**：讨论了 **Z-loss** 作为目标函数的正则化项，并将其与 weight decay 进行了比较，成员们对其必要性展开了辩论。
   - [Carsonpoole 澄清道](https://link.address)，Z-loss 通过防止过大的激活值来针对激活不稳定性（activation instability），并将其与现有的正则化方法进行了对比。
- **CoALA：语言 Agent 的结构化方法**：一篇关于 [语言 Agent 认知架构 (CoALA)](https://arxiv.org/abs/2309.02427) 的论文介绍了一个带有模块化内存组件的框架，用以指导语言模型开发。
   - 该框架旨在调查和组织语言模型的最新进展，借鉴认知科学和符号 AI 来提供可操作的见解。
- **BPB 与 per token 指标澄清**：针对给定指标应解释为 **bits per byte (BPB)** 还是 **per token** 进行了澄清，为确保准确性将其确定为 'per token'。
   - *Cz_spoon_06890* 指出，正确理解该指标对相应评估具有重大影响。
- **Scaling laws 影响 hypernetwork 能力**：讨论集中在 **scaling laws** 如何影响 hypernetworks 及其达到这些定律预测的目标误差的能力，并质疑了较小 hypernetworks 的可行性。
   - 建议包括将 hypernetworks 集中在具有有利 scaling laws 的任务上，从而更容易从特定的数据子集中学习。
- **无分词模型引发辩论**：关于 **tokenization-free models** 在字节或字符层面的可解释性展开了辩论，并对缺乏规范的处理位置表示担忧。
   - 一位成员指出，“Utf-8 也是一种 tokenization 方案，只是个糟糕的方案”，表达了对字节级 tokenization 的怀疑。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **MistralAI 和 OpenAI 发布新模型**：今天是新模型发布的大日子，**MistralAI** 和 **OpenAI** 均有发布，且两者都已获得 [零日支持](https://twitter.com/llama_index/status/1814036536192811184)，包括性能超越 **Mistral 7b** 的全新 **Mistral NeMo** 12B 模型。
   - **Mistral NeMo** 模型拥有显著的 128k context window。
- **LlamaCloud 更新增强协作**：**LlamaCloud** 最近的更新引入了 **LlamaCloud Chat**（数据的对话式界面）以及用于协作的新团队功能。
   - 这些变化旨在提升用户体验和生产力。[点击此处阅读更多](https://twitter.com/llama_index/status/1814363518726222119)。
- **通过 Re-ranking 提升相关性**：对检索结果进行重排序（Re-ranking）可以显著增强响应的相关性，尤其是在使用 **@postgresml** 等托管索引时。
   - 查看他们在 LlamaIndex 博客上的 [客座文章](https://t.co/HWfitT0CJt) 以获取更多见解。[更多详情](https://twitter.com/llama_index/status/1814386548340826449)。
- **LLMs context window 限制引发困惑**：一位用户在为 GPT-4o mini 设置 max_tokens 限制时遇到了 'Error code: 400'，尽管 OpenAI 的文档称其 context window 为 128K tokens，但据报道它仅支持 16384 个 completion tokens。
   - 这种困惑源于在代码的不同部分使用了不同的模型，导致 SQL query engines 中 GPT-3.5 和 GPT-4 之间产生了干扰。
- **通过 LlamaIndex 为非结构化数据提供 ETL**：一位成员询问如何将视频和音乐等非结构化数据解析为 LLMs 可理解的格式，并引用了 Jerry Liu 和 Alejandro 之间的一次 YouTube [对话](https://www.youtube.com/watch?v=imlQ1icxpBU)，其中提到了一种新型 ETL。
   - 这突出了 ETL 在 AI 数据处理中的实际应用和潜在用例。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **训练推理提升 Transformer 泛化能力**：一篇 [arXiv 论文](https://arxiv.org/abs/2405.15071) 指出，在饱和点之外继续训练 Transformer 可以增强其泛化能力和推理事实推断能力。
   - 研究结果显示，由于缺乏在多种语境下存储相同事实的动力，Transformer 在处理域外（out-of-domain）推理时表现不佳。
- **配置问题困扰 Mistral-12b 的使用**：一名成员报告了 **Mistral-12b** 的配置问题，特别是投影权重（projection weights）的大小不匹配。
   - 修复方案包括从源码安装 transformers 并调整训练设置（如使用 8x L40s），这在降低 loss 方面表现出了改进。
- **Triplex 模型革新知识图谱构建**：基于 Phi3-3.8B 的 **Triplex 模型** 与 GPT-4 相比，构建知识图谱的成本降低了 98%（[来源](https://huggingface.co/SciPhi/Triplex)）。
   - 该模型可共享、可在本地执行，并能与 Neo4j 和 R2R 良好集成，从而增强下游的 RAG 方法。
- **Axolotl 训练调整解决 GPU 显存错误**：axolotl 训练过程中常见的 GPU 显存错误引发了关于调整 `micro_batch_size`、`gradient_accumulation_steps` 以及启用 `fp16` 的讨论。
   - 社区分享了这些设置的详细指南，以优化显存使用并防止错误。
- **Llama3 调整降低了 Eval 和训练 Loss**：降低 **Llama3** 的 rank 有助于改善其 eval loss，尽管仍需进一步运行以确认稳定性。
   - 训练 loss 也明显降低，表明改进效果一致。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **GPTs Agents 表现出自我意识**：在 **GPTs agents** 上进行的一项实验旨在评估其自我意识，过程中特意避开了网络搜索功能。
   - 测试结果引发了关于在没有外部数据源的情况下，具有自我意识的 AI 系统的实际意义和潜在局限性的讨论。
- **Cohere Toolkit 的灵活性给社区留下深刻印象**：一位社区成员转发了 [Aidan Gomez 和 Nick Frosst 的推文](https://x.com/aidangomez/status/1814308463104668113)，赞扬了 **Cohere Toolkit UI** 的开源特性，它允许集成各种模型并贡献新功能。
   - 开源方法因其能够实现广泛的定制化并促进整个社区在工具开发方面的创新而受到称赞。
- **Firecrawl 面临定价挑战**：一名成员指出，在没有庞大客户群的情况下， **Firecrawl** 的成本很高，建议转向按需付费（pay-as-you-go）模式。
   - 讨论内容包括各种定价策略以及针对小型用户提供更灵活方案的需求。
- **Firecrawl 自托管被视为节省成本的方案**：成员们探讨了通过自托管 **Firecrawl** 来降低费用，一位成员分享了一份详细说明该过程的 [GitHub 指南](https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md)。
   - 据报告，自托管显著降低了成本，使该服务对个人开发者更具吸引力。
- **本地 LLM 聊天 GUI 项目获得关注**：社区分享了一个由本地 **LLMs** 驱动的聊天 GUI 新项目，集成了 **Web Search、Python 解释器和图像识别**功能。
   - 感兴趣的成员可前往该项目的 [GitHub 仓库](https://github.com/yamikumo-DSD/chat_cmr) 进行进一步的交流和贡献。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **统一数据集抽象 RFC 获得关注**：旨在统一 instruct 和 chat 数据集以支持多模态数据的 [RFC](https://github.com/pytorch/torchtune/pull/1186) 引起了广泛讨论，核心反馈集中在将 tokenizer 和 prompt templating 与其他配置分离。
   - 成员们强调了可用性和改进领域，建议采用更用户友好的方法来高效管理数据集配置。
- **Torchtune Recipe 文档将自动生成**：出现了从 recipe docstrings [自动生成文档的提案](https://github.com/pytorch/torchtune/pull/256)，以提高 Torchtune recipes 的可见性和可访问性。
   - 此举旨在确保用户拥有与当前版本 recipes 保持一致的、最新的、易于浏览的文档。
- **错误处理重构建议**：讨论中提到了通过集中通用的验证函数来简化 Torchtune recipes 中的错误处理，从而提供更简洁的代码库。
   - 其理念是尽量减少样板代码，并将用户的注意力集中在关键配置上，以提高效率。
- **合并 Instruct/Chat 数据集 RFC**：分享了一个 [RFC](https://link.to.rfc)，旨在合并 Instruct/Chat 数据集，以简化在 **Hugging Face** 上添加自定义数据集的过程。
   - 鼓励微调任务的定期贡献者进行审查并提供反馈，确保这不会影响高层 API。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Mozilla Builders 启动初创企业加速器**：[Mozilla Builders](https://builders.mozilla.org) 宣布了一个针对硬件和 AI 项目的初创企业加速器，旨在推动边缘创新。
   - 一位成员表现出极大的热情，表示：*“我不搬走，这不是兼职加速器，我们就住在这里。”*
- **为盲人生成的 AI 场景描述**：社区讨论了利用 AI 为视障人士生成场景描述，旨在增强可访问性。
   - 情绪高涨，出现了诸如 *“失明和所有疾病都需要被消除”* 之类的言论。
- **智能 AI 设备助力养蜂业**：重点介绍了用于养蜂业的**智能 AI 数据驱动设备**的开发，为养蜂人提供预警以防止蜂群流失。
   - 这种创新方法为将 AI 整合到农业和环境监测中带来了希望。
- **GoldFinch 凭借混合模型增益问世**：**GoldFinch** 结合了来自 RWKV 和 Transformers 的 Linear Attention，在任务表现上优于 **1.5B 级 Llama** 等模型，减少了二次减速并缩小了 KV-Cache 大小。
   - 应用包括使用消费级 GPU 分析大型文档或代码库，从而显著降低成本。论文和代码可在 [arXiv](https://arxiv.org/abs/2407.12077) 和 [GitHub](https://github.com/recursal/GoldFinch-paper) 上获得。
- **GPTAlpha 和 Finch-C2 模型表现优于竞争对手**：新的 **Finch-C2** 和 **GPTAlpha** 模型融合了 RWKV 的线性特性和 Transformer 架构，提供比传统模型更好的性能和效率。
   - 这些模型增强了下游任务的性能，可在 [GitHub](https://github.com/recursal/GoldFinch-paper) 和 [Huggingface](https://huggingface.co/recursal/GoldFinch-paper) 上获取完整的文档和代码。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 中的 Kernel 重构引发变革**：George Hotz 建议重构 Kernel 以消除 `linearize` 并引入 `to_program` 函数，从而促进更好的结构化。
   - 他强调需要先移除 `get_lazyop_info` 才能高效地实施这些变更。
- **GTX1080 在 tinygrad 兼容性方面遇到困难**：[一位成员](https://discord.com)报告了在 GTX1080 上使用 `CUDA=1` 运行 tinygrad 时出现错误，突显了 GPU 架构问题。
   - 另一位成员建议至少使用 **2080 代 GPU**，并建议在 `ops_cuda` 中应用补丁并禁用 tensor cores。
- **tinygrad 内部机制：理解 View.mask**：一位成员深入研究了 tinygrad 的内部结构，特别是询问了 `View.mask` 的用途。
   - George Hotz 澄清它主要用于 padding，并提供了一个[参考链接](https://discord.com/channels/1068976834382925865/1070745817025106080/1255977369727140013)。
- **剖析 tinygrad 中的 `_pool` 函数**：一位成员寻求关于 `_pool` 函数的澄清，思考它是否通过 `pad`、`shrink`、`reshape` 和 `permute` 操作复制了数据。
   - 经过进一步检查，该成员意识到该函数并不像最初想象的那样复制数值。
- **新项目提案：记录 OpenPilot 模型追踪**：George Hotz 提议了一个项目，旨在利用 OpenPilot 模型追踪来记录 kernel 的变更及其对性能的影响。
   - 他分享了一个包含说明的 [Gist 链接](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb)，邀请成员参与。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **GPT-4o Mini 引发参数修改疑问**：一位用户询问 **GPT-4o Mini** 是否仅通过修改参数即可运行，还是需要 **OI** 的正式引入。
   - 讨论暗示了潜在的设置挑战，但在是否需要正式引入机制方面缺乏明确共识。
- **16k token 输出功能令人惊叹**：社区对令人印象深刻的 **16k 最大 token 输出**功能感到惊叹，突显了其在处理海量数据方面的潜在效用。
   - 贡献者建议这种能力可能会彻底改变大规模文档解析和生成任务。
- **Yi large preview 仍是顶级竞争者**：成员们报告称 **Yi large preview** 在 **OI** 框架内的表现继续优于其他模型。
   - 推测认为稳定性和改进的上下文处理是其关键优势。
- **GPT-4o Mini 在代码生成方面落后**：初步测试表明 **GPT-4o Mini** 速度很快，但在**代码生成方面表现平平**，未达到预期。
   - 尽管如此，一些人认为它在配合精确的自定义指令时可能在特定任务中表现出色，尽管其 function-calling 能力仍需改进。
- **OpenAI 宣传 GPT-4o Mini 的 function calling**：OpenAI 的[公告](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)赞扬了 **GPT-4o Mini** 强大的 function-calling 技能和增强的长上下文性能。
   - 社区反应不一，争论所报道的改进是否与实际观察相符。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **ICML'24 重点介绍 LAION 模型**：研究人员感谢 **LAION** 项目提供的模型，这些模型被用于一篇 [ICML'24 论文](https://europe.naverlabs.com/text2control)。
   - 他们分享了其 **Text2Control** 方法的交互式 Demo，并称其对于提升 **vision-language models** 的能力至关重要。
- **Text2Control 支持自然语言指令**：[Text2Control](https://europe.naverlabs.com/text2control) 方法使 **Agent** 能够通过 **vision-language models** 解析自然语言指令来执行新任务。
   - 该方法在 **zero-shot generalization** 方面优于多任务 **reinforcement learning** 基准，并提供了一个 [交互式 Demo](https://europe.naverlabs.com/text2control) 供用户探索其功能。
- **AGI 炒作与模型性能**：一次讨论强调了 **AGI** 被过度炒作的现状，同时指出许多模型在经过适当实验后能达到很高的准确率，并引用了 [@_lewtun 的一条推文](https://x.com/_lewtun/status/1813197210600829192)。
   - *“许多模型能正确解决类似 AGI 的任务，但进行必要的实验通常被认为是‘无趣的’”*。
- **需要 Latents 以降低存储成本**：用户表示需要像 **sdxl vae** 这样的大型图像数据集的 **latents**，以降低存储成本。
   - 有建议将其托管在 **Hugging Face** 上，因为该平台可以承担 **S3 存储费用**。
- **体验 CNN 解释器工具**：分享了一个 [CNN 解释器可视化工具](https://poloclub.github.io/cnn-explainer/)，旨在通过交互式视觉效果帮助用户理解卷积神经网络（**CNNs**）。
   - 该工具对于那些希望从实践角度加深对 **CNNs** 理解的人特别有用。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Triplex 大幅降低图谱构建成本**：[Triplex](https://huggingface.co/SciPhi/Triplex) 在构建知识图谱方面实现了 **98% 的成本降低**，在性能超过 **GPT-4** 的同时，运行成本仅为其 **1/60**。
   - 由 [SciPhi.AI](https://www.sciphi.ai) 开发的 Triplex 是一个经过 **finetuned** 的 **Phi3-3.8B** 模型，得益于 SciPhi 的 **R2R**，现在支持以极低的成本进行本地图谱构建。
- **LangChain 中不需要特定模型的 Prompt 用词**：一位用户询问在 LangChain 的 `ChatPromptTemplate` 中是否需要针对特定模型的用词来确保 **Prompt** 的准确性。
   - 官方澄清 `ChatPromptTemplate` 抽象化了这一需求，使得像 `<|assistant|>` 这样的特定标记变得不再必要。
- **使用 ChatPromptTemplate 创建 Prompt**：分享了一个关于如何在 LangChain 的 `ChatPromptTemplate` 中定义消息数组的示例，利用了角色（role）和消息文本对。
   - 提供了 [指南链接](https://js.langchain.com/v0.2/docs/tutorials/llm_chain/#prompt-templates) 以获取详细步骤，帮助有效地构建结构化 **Prompt**。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **OpenAI Scale Tier 之谜**：一位成员询问关于新的 [OpenAI Scale Tier](https://openai.com/api-scale-tier/) 的理解，引发了社区对 **GPT-4 TPS** 计算方式的困惑。
   - 讨论强调了 **TPS** 确定的复杂性以及 **GPT-4** 性能指标中的差异。
- **GPT-4 TPS 计算困惑**：成员们对 OpenAI 在 **pay-as-you-go** 级别上计算出的 **19 tokens/second** 感到困惑，因为 **GPT-4** 的实际输出接近 **80 tokens/second**。
   - 这引发了关于 **TPS** 计算准确性及其如何影响不同使用层级的辩论。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **企业对共享敏感数据持谨慎态度**：一位成员指出，企业不愿与第三方共享**敏感的业务线数据**或**客户/患者数据**，这反映了对**数据隐私**的高度关注。
   - 讨论强调，这种谨慎源于对**数据安全**和**隐私泄露**日益增长的担忧，导致企业优先考虑内部控制而非外部数据交换。
- **数据隐私成为焦点**：随着企业应对合规性和安全性挑战，对**数据隐私**的关注变得越来越重要。
   - 观察到一个明显的趋势，即企业正优先考虑保护**敏感信息**，以防潜在的未经授权访问。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **明确针对目标受众的沟通**：讨论重点在于理解**目标受众**以实现有效沟通，强调了不同的群体，如**工程师**、**准工程师**、**产品经理**、**DevRel** 和**解决方案架构师**。
   - 参与者强调，为这些特定群体定制信息可以确保相关性和影响力，从而提高沟通的有效性。
- **针对性沟通的重要性**：明确目标受众可确保沟通对特定群体具有相关性和影响力。
   - 其目的是为**工程师**、**准工程师**、**产品经理**、**DevRel** 和**解决方案架构师**适当地定制信息。



---


**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期处于静默状态，请告知我们，我们将予以移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1263576583818579978)** (190 messages🔥🔥): 

> - `Mistral-Nemo 模型细节`
> - `Mistral-Nemo 在 Unsloth 上的支持状态`
> - `关于 AI 模型的社区互动`
> - `Unsloth 的内部运作`
> - `即将推出的功能和发布` 


- **Mistral-Nemo 模型细节**：讨论围绕 Mistral-Nemo 的模型架构展开，特别关注 head dimensions 和 hidden sizes，并分享了 [Hugging Face 模型卡片](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) 和 [博客文章](https://mistral.ai/news/mistral-nemo/) 的链接以获取更多详情。
   - 一位社区成员澄清说，调整参数有助于在不损失显著信息的情况下保持计算效率。
- **Unsloth 正式支持 Mistral-Nemo**：Unsloth 宣布支持 Mistral-Nemo 模型，并通过 [Google Colab 链接](https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing) 进行了确认，并解决了与 EOS 和 BOS token 相关的一些初始障碍。
   - 社区对此次发布表示兴奋，强调了 Unsloth 的动态 RoPE 分配，它可以根据数据集的长度高效管理高达 128K token 的上下文。
- **精益创业：Unsloth 团队结构**：社区惊讶地发现 Unsloth 仅由两兄弟运营，负责工程、产品、运营和设计，这引发了对其效率的钦佩。
   - 成员之间进行了幽默且支持性的互动，庆祝社区里程碑和个人喜讯（如为人父母）等成就。
- **探索 Unsloth 的外部替代方案**：讨论了为更轻松地访问 AI 模型所做的努力，包括用于本地使用的 Jan AI 和 [Colab](https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb) 中的 OobaGooba 等替代方案。
   - 成员们渴望找到无需复杂设置即可运行模型的便捷平台，强调了用户友好界面的重要性。
- **未来功能和即将发布的版本**：Unsloth 宣布了几个正在开发中的新发布和功能，包括对视觉模型的支持以及模型推理和训练界面的改进。
   - 团队鼓励社区参与反馈和测试，透露了提高 VRAM 效率和扩展功能的计划。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.gradio.app/guides/creating-a-chatbot-fast#introduction)">快速创建聊天机器人</a>：Gradio 分步教程</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/love-quotes-gif-3643220039448794437">Love Quotes GIF - Love quotes - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/GoogleColab/status/1778535625840525400">来自 Colaboratory (@GoogleColab) 的推文</a>：Colab 现在为付费用户提供 NVIDIA L4 运行时！🚀 24GB 显存！当你想要比 T4 更高性能的 GPU 时，这是一个绝佳选择。通过选择 L4 运行时来尝试一下吧！</li><li><a href="https://tenor.com/view/dad-jokes-aht-aht-dad-jokes-aht-aht-ha-ha-ha-knee-slapper-gif-26152690">Dad Jokes Aht Aht GIF - Dad Jokes Aht Aht Dad Jokes Aht Aht - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/unsloth/Mistral-Nemo-Base-2407-bnb-4bit">unsloth/Mistral-Nemo-Base-2407-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit">unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/GeraudBourdin/llm-scripts/blob/main/collab_price_gpu.ipynb">llm-scripts/collab_price_gpu.ipynb at main · GeraudBourdin/llm-scripts</a>：通过在 GitHub 上创建账户，为 GeraudBourdin/llm-scripts 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/unsloth/comments/1e4w3i0/wrote_a_python_script_to_auto_install_unsloth_on/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/janhq/jan">GitHub - janhq/jan: Jan 是 ChatGPT 的开源替代方案，可在你的电脑上 100% 离线运行。支持多种引擎 (llama.cpp, TensorRT-LLM)</a>：Jan 是 ChatGPT 的开源替代方案，可在你的电脑上 100% 离线运行。支持多种引擎 (llama.cpp, TensorRT-LLM) - janhq/jan</li><li><a href="https://jan.ai/">将你的电脑变成 AI 电脑 - Jan</a>：在你的电脑上本地离线运行 Mistral 或 Llama2 等 LLM，或者连接到远程 AI API，如 OpenAI 的 GPT-4 或 Groq。</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blob/main/Colab-TextGen-GPU.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sh">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/oobabooga/text-generation-webui/blo">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1263877918380134464)** (1 messages): 

> - `Mistral NeMo 发布`
> - `CSV/Excel 微调`
> - `Ollama 模型支持`
> - `新文档页面`
> - `免费 Notebooks` 


- **Mistral 发布 NeMo 12B 模型**：Mistral 发布了 [NeMo](https://huggingface.co/unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit)，这是一个 **120 亿参数模型**，展示了多语言能力和原生工具支持。
   - *完美适配免费的 Google Colab GPU 实例*，你可以[在此访问](/blog/mistral-nemo)。
- **现已支持 CSV/Excel 微调**：你现在可以使用 **CSV/Excel 文件** 以及 **多列数据集** 来微调模型。
   - 访问 [Colab notebook](https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing) 获取更多详情。
- **集成 Ollama 模型支持**：新增了将模型部署到 **Ollama** 的支持。
   - 查看 [Ollama Llama-3 (8B) Colab](https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing) 了解更多信息。
- **新文档页面上线**：推出我们的[新文档页面](https://docs.unsloth.ai/)，以提供更好的指导和资源。
   - 包含 **LoRA Parameters Encyclopedia** 等功能和教程，用于全面学习。
- **Unsloth Studio (Beta) 发布预告**：**Unsloth Studio (Beta)** 将于下周发布，具备增强功能。
   - 更多细节即将公布，敬请期待！


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unsloth.ai/blog/mistral-nemo">使用 Unsloth 微调 Mistral NeMo</a>: 通过 Unsloth 微调 Mistral 的新模型 NeMo 128k，支持 4 倍长的上下文长度！</li><li><a href="https://docs.unsloth.ai/)">Unsloth Docs</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/17d3U-CAIwzmbDRqbZ9NnpHxCkmXB6LZ0?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing)">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing)">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1263720824108748870)** (20 条消息🔥): 

> - `GPT-4o mini model`
> - `Claude model sizes`
> - `Salesforce xLAM models`
> - `Model weights and context windows`
> - `Rumors and validations` 


- **GPT-4o Mini 在 MMLU 上获得高分**：传闻 [OpenAI 的新模型 GPT-4o mini](https://techcrunch.com/2024/07/18/openai-unveils-gpt-4o-mini-a-small-ai-model-powering-chatgpt/) 是一个 8B 模型，在 **MMLU** 基准测试中得分 **82**，引起了 AI 社区的关注。
   - 推测认为它实际上可能是一个 **MoE 模型**或涉及量化（quantization）技术，使其精确规模变得模糊。
- **Salesforce 发布 xLAM 模型**：[Salesforce 发布了模型权重](https://huggingface.co/Salesforce)，包括其 **1B 和 7B xLAM 模型**，具备函数调用（function calling）能力和不同的上下文窗口。
   - 虽然 1B 模型支持 **16K tokens**，但 7B 模型仅能处理 **4K tokens**，一些人认为相对于其尺寸来说这个表现一般。
- **Claude 模型尺寸详情**：[Alan D. Thompson 的备忘录](https://lifearchitect.substack.com/p/the-memo-special-edition-claude-3#:~:text=3%20models%20sizes%3A%20Haiku%20(~20B)%2C%20Sonnet%20(~70B)%2C%20and%20Opus%20(~2T)) 揭示了 Claude 3 模型有多种尺寸，包括 **Haiku (~20B)**、**Sonnet (~70B)** 和 **Opus (~2T)**。
   - 这种多样性突显了 Anthropic 满足不同性能和资源需求的战略方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lifearchitect.substack.com/p/the-memo-special-edition-claude-3#:~:text=3%20models%20sizes%3A%20Haiku%20(~20B)%2C%20Sonnet%20(~70B)%2C%20and%20Opus%20(~2T)">The Memo - 特别版：Claude 3 Opus</a>：Anthropic 发布 Claude 3，性能超越了包括 GPT-4 在内的所有模型</li><li><a href="https://lifearchitect.ai/models-table/">Models Table</a>：在新标签页中打开模型表 | 返回 LifeArchitect.ai。数据字典模型（文本）名称...</li><li><a href="https://techcrunch.com/2024/07/18/openai-unveils-gpt-4o-mini-a-small-ai-model-powering-chatgpt/?guccounter=1#:~:text=OpenAI%20would%20not%20disclose%20exactly%20how%20large%20GPT%2D4o%20mini%20is%2C%20but%20said%20it%E2%80%99s%20roughly%20in%20the%20same%20tier%20as%20other%20small%20AI%20models%2C%20such%20as%20Llama%203%208b%2C%20Claude%20Haiku%20and%20Gemini%201.5%20Flash.">OpenAI 发布 GPT-4o mini，一款更小且更便宜的 AI 模型 | TechCrunch</a>：OpenAI 周四推出了 GPT-4o mini，这是其最新的小型 AI 模型。该公司表示 GPT-4o mini 比 OpenAI 目前的模型更便宜、更快</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e6wncc/salesforce_released_model_weights_for_the_xlam_1b/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/07/18/openai-unveils-gpt-4o-mini-a-small-ai-model-powering-chatgpt/?gucc">OpenAI 发布 GPT-4o mini，一款更小且更便宜的 AI 模型 | TechCrunch</a>：OpenAI 周四推出了 GPT-4o mini，这是其最新的小型 AI 模型。该公司表示 GPT-4o mini 比 OpenAI 目前的模型更便宜、更快
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1263574774110752830)** (89 messages🔥🔥): 

> - `CUDA bf16 问题`
> - `模型部署与微调`
> - `Mistral Colab notebook 问题`
> - `Mistral Nemo 中的 FIM (Fill in the Middle) 支持`
> - `双 GPU 规格指定` 


- **多种 GPU 上的 CUDA bf16 问题**：多名用户报告了在 RTX A4000 和 T4 等不同 **GPU 型号**上与 **bf16** 支持相关的错误，阻碍了模型执行。
   - 问题被确定为由于 **torch.cuda.is_bf16_supported() 返回 False** 引起，Unsloth 团队随后修复了该问题。
- **模型部署可能需要 GPU 进行推理**：一位用户询问了在服务器上部署其训练好的模型，并被建议使用专门的推理引擎如 vllm。
   - 普遍共识是，使用 **GPU VPS** 处理模型的推理任务更为理想。
- **Mistral Colab notebook 出现 bf16 错误**：**Mistral Colab notebook** 的用户在 **A100** 和其他 GPU 上遇到了 bf16 相关错误。
   - 经过调查，Unsloth 团队确认已修复该问题，测试显示目前运行正常。
- **了解 Mistral Nemo 中的 FIM**：讨论了 **Mistral Nemo** 中关于代码补全任务的 **Fill in the Middle (FIM)** 支持。
   - FIM 允许语言模型预测文本输入中间缺失的部分，这对于代码自动补全非常有用。
- **指定用于微调的 GPU**：一位用户寻求关于如何在拥有多个 GPU 的机器上指定使用哪个 GPU 进行训练的指导。
   - Unsloth 团队引导他们查看最近的一个 [GitHub pull request](https://github.com/unslothai/unsloth/pull/228)，该 PR 修复了 CUDA GPU ID 选择的问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@SymeCloud/what-is-fim-and-why-does-it-matter-in-llm-based-ai-53f33385585b">什么是 FIM，为什么它在基于 LLM 的 AI 中很重要</a>：当你在喜欢的编辑器中写作时，类似 AI 的 Copilot 会根据你写的内容立即进行猜测和补全……</li><li><a href="https://github.com/unslothai/unsloth/pull/228">修复单 GPU 限制代码通过环境变量覆盖错误的 CUDA GPU ID，由 Qubitium 提交 · Pull Request #228 · unslothai/unsloth</a>：PR 修复了以下场景：存在多个 GPU 设备，用户已使用 CUDA_VISIBLE_DEVICES=13,14 启动 Unsloth 代码，无论是否设置 CUDA_DEVICE_ORDER=PCI_BUS_ID，当前代码都会……</li><li><a href="https://www.reddit.com/r/unsloth/comments/1e4w3i0/wrote_a_python_script_to_auto_install_unsloth_on/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1263739387804389437)** (2 messages): 

> - `Triplex 知识图谱`
> - `Triplex 成本降低`
> - `Triplex 对比 GPT-4`
> - `结合 Triplex 的 R2R`
> - `用于 R2R RAG 的 Supabase` 


- **SciPhi 开源用于知识图谱的 Triplex**：SciPhi 正在[开源](https://www.sciphi.ai/blog/triplex) **Triplex**，这是一款用于知识图谱构建的最先进 LLM，可显著降低 98% 的成本。
   - Triplex 可以与 SciPhi 的 R2R 配合使用，直接在笔记本电脑上构建知识图谱，其性能优于 few-shot-prompted 的 **GPT-4**，且推理成本仅为 1/60。
- **Triplex 构建知识图谱的成本降低 98%**：**Triplex** 旨在将构建知识图谱的费用降低 98%，使其比传统方法（可能耗资数百万）更易于负担。
   - 它是 Phi3-3.8B 的微调版本，专为从非结构化数据创建知识图谱而设计，可在 [HuggingFace](https://huggingface.co/SciPhi/Triplex) 上获取。
- **R2R 增强了 Triplex 在本地图谱构建中的应用**：[R2R](https://github.com/sciphi-ai/r2r) 被强调为利用 Triplex 以极低成本在本地构建知识图谱的解决方案。
   - R2R 提供了一个全面的平台，具有多模态支持、混合搜索和自动关系提取等功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://kg.sciphi.ai/.">SOTA 三元组提取</a>：未找到描述</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>：未找到描述</li><li><a href="https://ollama.com/sciphi/triplex">sciphi/triplex</a>：快速上手并运行大语言模型。</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph.">简介 - 最好的开源 AI 驱动问答引擎。</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1263739612250116217)** (8 条消息🔥): 

> - `绕过 PyTorch`
> - `OpenAI 中的可训练 Embeddings`
> - `评估微调后的 LLaMA3 模型` 


- **通过 backward hook 绕过 PyTorch**：一位用户建议，要绕过 **PyTorch**，可以添加一个 backward hook 并将梯度清零。
   - 讨论中提到，在内存中存储整个计算图（compute graph）是否会违背部分可训练 Embedding 的初衷。
- **OpenAI 的双矩阵 Embedding 策略**：一位用户提到 **OpenAI** 将其 Embeddings 分为两个矩阵：一个小的可训练部分和一个大的冻结部分。
   - 他们还指出需要逻辑路径来选择不同的代码路径。
- **在 Colab 上评估微调后的 LLaMA3 8B**：一位用户分享了一个 [Colab notebook 链接](https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing)，并寻求帮助评估微调后的 **LLaMA3 8B** 模型。
   - 另一位用户建议：*“尝试在拼接的冻结 Embedding、可训练 Embedding 以及线性层上进行 PyTorch 训练。”*



**提到的链接**：<a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1263734251963027478)** (5 条消息): 

> - `睡眠衍生机制`
> - `人工神经网络`
> - `灾难性遗忘` 


- **睡眠衍生机制减少神经网络中的灾难性遗忘**：[加州大学圣迭戈分校 (UC San Diego) 的研究人员](https://vchs.ucsd.edu/blog/2022/12/sleep-derived-how-artificial-neural-networks-can-avoid-catastrophic-forgetting.html)表明，在人工神经网络中实施类睡眠阶段，可以通过减少记忆覆盖来缓解**灾难性遗忘 (catastrophic forgetting)**。
   - 这项发表在 **Nature Communications** 上的研究证明，神经网络中的**类睡眠无监督重放 (sleep-like unsupervised replay)** 有助于在进行新训练时保护旧记忆。
- **AI 爱好者的待读论文积压**：一位成员提到，频道中分享的所有近期 **AI 论文** 已经积压了很多没读。
   - 另一位成员幽默地回应道：*“仿生人会梦见电子羊吗？ (Do robots dream of electric sheep?)"*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vchs.ucsd.edu/blog/2022/12/sleep-derived-how-artificial-neural-networks-can-avoid-catastrophic-forgetting.html">
			Sleep Derived: How Artificial Neural Networks Can Avoid Catastrophic Forgetting
		</a>：未找到描述</li><li><a href="https://www.nature.com/articles/s41467-022-34938-7">Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks - Nature Communications</a>：众所周知，人工神经网络在最近学习的任务上表现良好，但同时会遗忘之前学习的任务。作者提出了一种无监督睡眠重放算法来恢复...
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1263572699947733094)** (233 条消息🔥🔥): 

> - `针对 SD 的 ComfyUI`
> - `NVIDIA vs AMD GPU`
> - `SD 模型推荐`
> - `SD 中的艺术风格`
> - `脱离 Reddit 获取 AI 新闻` 


- **推荐初学者使用 ComfyUI 运行 Stable Diffusion**：成员们建议将 [ComfyUI](https://comfy.icu/) 作为 Stable Diffusion 初学者的理想 UI，并强调了其灵活性和易用性。
   - 此外，建议观看 Scott Detweiler 的 [YouTube 教程](https://www.youtube.com/@sedetweiler) 以获取详尽的指导。
- **AI 任务首选 NVIDIA 显卡**：讨论中的共识表明，由于更好的支持和更少的故障排除需求，在运行 Stable Diffusion 时 NVIDIA GPU 优于 AMD。
   - 尽管 AMD 提供了更多的 VRAM，但 NVIDIA 因其更广泛的兼容性（尤其是在 Linux 环境中）而受到赞誉，尽管偶尔会出现驱动程序问题。
- **SD 模型推荐因需求而异**：关于最佳 Stable Diffusion 模型的讨论结论是，选择取决于 VRAM 和用户的具体需求，其中 SDXL 因其更大的参数规模和能力而被推荐。
   - 提到了 SD3，因其采用了新的 VAE 而具有卓越的图像质量，同时指出目前它主要在 ComfyUI 中得到支持。
- **Stable Diffusion 艺术风格技巧**：一位成员寻求如何让图像看起来更具艺术感、减少过度写实感的建议，并抱怨目前 HD、高对比度的输出占据了主导地位。
   - 建议包括使用艺术风格的 LoRA 以及尝试不同的模型，以实现理想的数字绘画效果。
- **Reddit 之外的 AI 新闻替代来源**：一位成员对 Reddit 的封禁和 Twitter 的审查表示沮丧，正在寻找 AI 新闻的其他来源。
   - 建议包括在 Twitter 上关注科学界以获取最新的论文和进展，尽管存在感知上的地区性和基于用户的审查问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.instagram.com/reel/C8luO4VM3l1/?igsh=MzRlODBiNWFlZA==">ERLAX 在 Instagram: &quot;&#x2026; 

#techno #dreamcore #rave #digitalart #aiart #stablediffusion&quot;</a>: 4,246 个赞，200 条评论 - erlax.case 于 2024 年 6 月 24 日: &quot;&#x2026;   #techno #dreamcore #rave #digitalart #aiart #stablediffusion&quot;。 </li><li><a href="https://www.youtube.com/@sedetweiler">Scott Detweiler</a>: Stability.ai 的质量保证人员 &amp; PPA 大师专业摄影师。问候！我是 Stability.ai 的首席 QA，也是一位驻扎在密尔沃基附近的专业摄影师和修图师...</li><li><a href="https://www.nasa.gov/missions/mars-2020-perseverance/perseverance-rover/heres-how-ai-is-changing-nasas-mars-rover-science/">AI 如何改变 NASA 的火星车科学 - NASA</a>: 人工智能正在帮助科学家识别毅力号火星车研究的岩石中的矿物质。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bts2km/sdxl_loras_with_pony_model_seem_to_not_work/">Reddit - 深入了解任何事物</a>: 未找到描述</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/ehristoforu/DeFooocus">GitHub - ehristoforu/DeFooocus: 始终专注于提示词和生成</a>: 始终专注于提示词和生成。通过在 GitHub 上创建账户为 ehristoforu/DeFooocus 的开发做出贡献。</li><li><a href="https://comfy.icu/">ComfyICU - ComfyUI 云端</a>: 在云端共享和运行 ComfyUI 工作流
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1263597700813946882)** (1 条消息): 

> - `Watermark Remover using Florence 2`
> - `CandyLLM Python Library`
> - `AI Comic Factory Update`
> - `Fast Subtitle Maker`
> - `Quantise + Load HF Text Embedding Models on Intel GPUs` 


- **使用 Florence 2 的水印去除器**：社区成员分享了一个使用 Florence 2 的 [水印去除器](https://huggingface.co/spaces/DamarJati/Remove-watermark)。
   - 贡献者声称：*“它对各种类型的水印都能产生出色的效果。”*
- **CandyLLM Python 库发布**：利用 [Gradio UI](https://github.com/shreyanmitra/CandyLLM) 的新 CandyLLM 库已发布。
   - 它的目标是通过用户友好的界面使语言模型交互更加便捷。
- **AI Comic Factory 更新增加对话气泡**：[AI comic factory](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory) 现在默认包含对话气泡。
   - 创建者指出，这一新功能提升了视觉叙事体验。
- **快速简便的字幕创建**：推出了一款全新的快速 [字幕制作工具](https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker)。
   - 社区对其快速的处理速度和易用性感到兴奋。
- **在 Intel GPU 上轻松量化并加载文本模型**：分享了一份关于如何在 Intel GPU 上 [量化并加载 HF 文本嵌入模型](https://github.com/sleepingcat4/intel-hf) 的指南。
   - 贡献者表示，这使得在 Intel 硬件上使用模型更加高效。



**提到的链接**：<a href="https://youtu.be/cpoS7K_fpRM)">如何从任何领域转型到 Machine Learning？ | Artificial Intelligence ft. @vizuara</a>：在这段视频中，来自 Vizuara 的 Raj Dandekar 博士分享了他从机械工程转型到 Machine Learning (ML) 的经验。他还解释了...

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1263574869220524142)** (194 条消息🔥🔥): 

> - `Loss Reduction Strategies` (Loss 降低策略)
> - `Issues with Model Processing Speed` (模型处理速度问题)
> - `Meta-Llama-3-70B-Instruct API Issues` (Meta-Llama-3-70B-Instruct API 问题)
> - `Hugging Face Infrastructure Problems` (Hugging Face 基础设施问题)
> - `Training Models on Kaggle` (在 Kaggle 上训练模型)


- **Loss Reduction Strategies Debated (Loss 降低策略讨论)**：一位成员询问是更多的数据还是更多的 epochs 会导致更小的 loss，并表示 *“我认为是更多的 epochs，但这可能会导致过拟合”*。
- **Cohere Model Processing Speed Criticized (Cohere 模型处理速度受诟病)**：用户注意到 **Cohere 模型** 与其他模型相比变慢了，某些响应甚至需要 *长达 5 分钟*。
   - *“自动通知？如果没有 ping，人们该如何收到通知？”*
- **Meta-Llama-3-70B-Instruct API Problem (Meta-Llama-3-70B-Instruct API 问题)**：一位成员在使用 **Meta-Llama-3-70B-Instruct API** 时遇到问题，收到错误提示称模型类型应为列出的几种特定配置之一。
   - 社区建议在 Hugging Face 上仔细检查该模型是否支持 **text2text-generation 任务**。
- **Acknowledgment of Hugging Face Infrastructure Problems (承认 Hugging Face 基础设施问题)**：一位开发者承认 Hugging Face 的基础设施问题影响了处理速度，特别是对于 **Cohere 模型**。
   - 在报告停机后，用户注意到系统出现了暂时的稳定性。
- **Training Models on Kaggle Discussed (在 Kaggle 上训练模型的讨论)**：一位成员询问 **Google/Gemma7b** 量化模型是否可以在拥有 16 GB RAM 的 Kaggle P100 上运行。
   - 另一位用户建议使用更现代的模型，如 **Llama 3 8b** 或 **Mistral Nemo 12b**，它们甚至可以适配 8GB VRAM。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/NexusflowX/status/1814333956646715567?t=tbB9IGd5WmkUII1Jbxxpgg&s=19">Nexusflow (@NexusflowX) 的推文</a>: 🚀 推出 Athene-70B：重新定义开源模型的后训练！我们很高兴发布 Athene-Llama3-70B，这是一个基于 @AIatMeta 的 Llama-3-70B 微调的开源权重对话模型。凭借出色的...</li><li><a href="https://arxiv.org/abs/2407.10240">xLSTMTime : Long-term Time Series Forecasting With xLSTM</a>: 近年来，基于 Transformer 的模型在多变量长期时间序列预测 (LTSF) 中占据了主导地位，尽管面临高昂的...</li><li><a href="https://stackoverflow.com/help/how-to-ask">如何提出一个好问题？ - 帮助中心</a>: Stack Overflow | 全球最大的开发者在线社区</li><li><a href="https://huggingface.co/docs/huggingface_hub/en/guides/download">从 Hub 下载文件</a>: 未找到描述</li><li><a href="https://civitai.com/user/AI_Art_Factory">Civitai | 分享你的模型</a>: 未找到描述</li><li><a href="https://youtu.be/87GxEhlAmhw?si=AcSDTcpxqJ3Ko0UG">Java 目标检测应用 (展示)</a>: 我大学第二学期 OOP 项目的最终展示。基本上，我的项目是用 Java 创建一个目标检测应用（遗憾的是）。这一切始于老师要求...</li><li><a href="https://tenor.com/view/%D0%B3%D1%80%D1%83%D1%81%D1%82%D0%BD%D1%8B%D0%B9-%D0%BA%D0%BE%D1%82-gif-4290454008808323372">грустный кот GIF - 伤心的猫 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/muslehal/xLSTMTime">GitHub - muslehal/xLSTMTime: 用于时间序列预测的 xLSTMTime</a>: 用于时间序列预测的 xLSTMTime。通过在 GitHub 上创建账号为 muslehal/xLSTMTime 的开发做出贡献。</li><li><a href="https://x.com/HochreiterSepp/status/1813189814373462295">Sepp Hochreiter (@HochreiterSepp) 的推文</a>: xLSTM 在时间序列预测方面表现出色。“我们的 xLSTMTime 模型在与最先进的基于 Transformer 的模型以及其他最近提出的时间序列模型对比中表现出了卓越的性能...”</li><li><a href="https://huggingface.co/spaces/nroggendorff/zelda-lora">Zelda Diffusion XL - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/animexl">Anime Diffusion XL - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/lm-sys/routellm">GitHub - lm-sys/RouteLLM: 一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！</a>: 一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！ - lm-sys/RouteLLM</li><li><a href="https://huggingface.co/spac">Spac (Stéphan Pacchiano)</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course">简介 - Hugging Face NLP 课程</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1263889004135317616)** (2 messages): 

> - `Crowdstrike BSOD issue`
> - `Knowledge Graphs` 


- **Crowdstrike 引发全球 BSOD**: 一个来自 **Crowdstrike 的错误文件** 导致了大规模的蓝屏死机 (BSOD)，影响了全球数百万个系统。
   - Crowdstrike 的 Overwatch 总监发布了一个 [hot fix](https://youtu.be/E8RQVx2gBFc?si=D2hdEW9k9iK0U9Vl) 来打破 BSOD 循环。
- **Knowledge Graphs 提供协助**: 一名成员提供了关于 **Knowledge Graphs** 的帮助和信息，强调了它们的趣味性和实用性。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1263731505952329760)** (3 messages): 

> - `Circuits Thread on Inner Workings of Neural Networks`
> - `Recent Model Releases`
> - `Interesting Papers on AI` 


- **Circuits 线程探索 Neural Networks**: [Circuits 线程](https://distill.pub/2020/circuits/) 提供了一种实验性格式，汇集了深入探讨 Neural Networks 内部机制的短文和批判性评论。
   - 它包含诸如 **Curve Detectors**、**Pose-Invariant Dog Head Detectors** 以及 **Polysemantic Neurons** 等组件。
- **周四模型发布潮**: 在短短一天内，发布了多个重要模型：**DeepSeek** 的顶级开源 lmsys 模型、**Mistral 12B**、**Snowflake** 的 embedding 模型、**HF** 的 Docmatix 数据集、**GoldFinch** 混合模型、**Arcee-Nova** 以及 **Mixedbread+deepset** embeddings。
   - *Osanseviero 评论道*：“🌊 送给那些被今天发布的消息淹没的人们。” [推文链接](https://x.com/osanseviero/status/1814068082060460409)。
- **近期值得关注的 AI 论文**: 亮点包括用于文档检索的视觉语言模型 **ColPali** ([论文](https://arxiv.org/pdf/2407.01449))、**Scaling Agents Across Simulated Worlds** ([论文](https://arxiv.org/pdf/2404.10179)) 以及 **Chameleon** 混合模态早期融合模型 ([论文](https://arxiv.org/pdf/2405.09818))。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/osanseviero/status/1814068082060460409">来自 Omar Sanseviero (@osanseviero) 的推文</a>: 🌊 送给那些被今天发布的消息淹没的人们。 1. DeepSeek 发布了顶级开源 lmsys 模型 2. Mistral 12B 模型 (多语言, tool usage, Apache 2) 3. Snowflake 发布了一个 embeddi...</li><li><a href="https://distill.pub/2020/circuits/zoom-in/">Zoom In: An Introduction to Circuits</a>: 通过研究神经元之间的连接，我们可以在 Neural Networks 的权重中发现有意义的算法。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1263609737942011985)** (12 条消息🔥): 

> - `使用 Llama 架构进行训练`
> - `MathStral 模型`
> - `Rush E 发布`
> - `AI Comic Factory`
> - `GPT-4o Mini` 


- **Llama 架构训练教程**: 发布了一个[关于使用 Llama 架构进行训练的新教程](https://huggingface.co/blog/nroggendorff/train-with-llama-architecture)，涵盖了从安装库到将训练好的模型推送到 Hugging Face Hub 的步骤。
   - 该教程分为详细的步骤，帮助用户登录 Hugging Face Hub、格式化数据集、设置训练参数等。
- **MathStral 在数学专业化方面表现出色**: 一名成员上传了一段[测试 MathStral 的 YouTube 视频](https://youtu.be/kP2sI4RuWsw?si=jA4AeLPiDomik9GU)，这是一个专门针对数学的新 Mistral 模型，在 Ollama 上展示了令人印象深刻的结果。
   - 该成员建议订阅他们的频道，以便获取未来模型发布的通知。
- **AI Comic Factory 增强故事对话**: 分享了关于 [AI Comic Factory](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory/discussions/832#66978fe8e6a07255a8d7f5d0) 如何使用 AI 生成的提示词和分割模型处理对话气泡的详细解释。
   - 该技术涉及检测类人形状，并使用 HTML Canvas API 绘制 AI 生成的对话气泡，这甚至适用于恐龙等非人类形状。
- **OpenAI 发布 GPT4o Mini**: 分享了一段 [YouTube 视频](https://youtu.be/aujSsSEcs8U?si=tbHHtkTQTMVTOABX)，展示了 OpenAI 新推出的 GPT4o Mini 模型的强大能力。
   - 视频鼓励观众亲自测试该模型，并提供了一个无需账号或信用卡即可免费访问的链接。
- **Isari 平台发布概念验证**: [Isari 平台](https://isari.ai)的概念验证已就绪，允许用户请求任务，使用来自 Hugging Face 的 `transformers` 在本地进行处理，并返回 JSON 输出。
   - 该平台目前使用一个模型（`phi-3-mini-4k-instruct`），但计划增加更多模型，包括提示词生成和代码生成功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory/discussions/832#66978fe8e6a07255a8d7f5d0">jbilcke-hf/ai-comic-factory · 在哪里可以找到代码？</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/nroggendorff/create-diffusers-dataset">为 Stable Diffusion 微调创建兼容 Diffusers 的数据集</a>: 未找到描述</li><li><a href="https://x.com/thepatch_kev/status/1814386138972598446?s=46">来自 thecollabagepatch (@thepatch_kev) 的推文</a>: 又一次使用开源 Ableton 插件进行极速创作，该插件可以与你一起即兴演奏 gary4live，这是与好友 tom's beat 的合作，跳转到 2:56 观看 @_buildspace @_nightsweekends @ma...</li><li><a href="https://huggingface.co/blog/nroggendorff/train-with-llama-architecture">从头开始训练 Llama 模型</a>: 未找到描述</li><li><a href="https://youtu.be/kP2sI4RuWsw?si=jA4AeLPiDomik9GU">MathΣtral 首次测试！结果非常令人印象深刻！Mistral AI</a>: 让我们在 Ollama 上尝试 MathΣtral</li><li><a href="https://youtu.be/vpqPFVn5jDU">Rush E</a>: 由 DistroKid 提供给 YouTube，Rush E · Noa Roggendorff，Rush E℗ 4056422 Records DK，发布日期：2024-07-18，由 YouTube 自动生成。</li><li><a href="https://youtu.be/87GxEhlAmhw?si=AcSDTcpxqJ3Ko0UG">Java 物体检测应用（展示）</a>: 我大学第二学期 OOP 项目的最终展示。基本上，我的项目是用 Java 创建一个物体检测应用（遗憾的是）。这一切始于老师要求...</li><li><a href="https://youtu.be/aujSsSEcs8U?si=tbHHtkTQTMVTOABX">OpenAI 发布了 GPT4o Mini | 让我们来测试它！</a>: 通过此链接查看最好的编程 AI：https://BestCoderAi.com（无需账号或信用卡即可免费使用 10 条消息）</li><li><a href="https://isari.ai">Isari - AI 增强型劳动力</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1263594128789344348)** (9 messages🔥): 

> - `Optimization of ML Model Layers`
> - `Paper Clubs in Different Discord`
> - `Event Planning for 8/3`
> - `Event Confirmation and Feedback` 


- **开始优化 ML 模型层**：一名成员开始致力于 **ML model layers** 的优化工作，包括 dense layers、GRU 和 LSTM GPU kernels。
   - 他们请求推荐一些基础论文或文章，以便在该领域建立职业生涯。
- **在另一个 Discord 上推广论文俱乐部**：一名成员询问在此频道发布关于其他 Discord 中进行的论文俱乐部活动是否合适。
   - *另一名成员建议* 只要不发布 Discord 邀请链接就可以。
- **规划 8/3 的活动**：成员们讨论了 **8/3** 的活动规划，并与另一名成员进行了确认。
   - 他们分享了活动链接并收到了积极的反馈，大家对活动图表表示赞赏。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1263598824883224677)** (7 messages): 

> - `camera calibration with Transformers`
> - `Object Detection App in Java`
> - `image segmentation for road detection using satellite images`
> - `DeeplabV3 and SenseTheRoad` 


- **对使用 Transformers 进行相机标定感到好奇**：一名成员询问是否有人具有使用 **Transformers** 模型进行 **camera calibration** 的经验。
- **Java 编写的目标检测应用展示**：一名成员分享了他们的 [YouTube 视频](https://youtu.be/87GxEhlAmhw?si=AcSDTcpxqJ3Ko0UG)，展示了一个为大学项目构建的 **Java 目标检测应用**。
   - *遗憾的是，它是用 Java 编写的*，并作为其 OOP 项目的一部分进行了详细说明。
- **寻找图像分割模型**：一名成员正在寻求用于卫星图像道路检测的 **image segmentation models** 推荐。
   - 另一名成员指向了 [SenseTheRoad](https://github.com/SinaRaoufi/SenseTheRoad)，推荐 **DeepLabV3** 作为该任务的一个可行选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/87GxEhlAmhw?si=AcSDTcpxqJ3Ko0UG">Object Detection App In Java ( Showcase )</a>：我大学第二学期 OOP 项目的最终展示。基本上，我的项目是用 Java 创建一个目标检测应用（遗憾地）。这一切都始于老师要求……</li><li><a href="https://github.com/SinaRaoufi/SenseTheRoad">GitHub - SinaRaoufi/SenseTheRoad: Road detection using DeepLabv3 segmentation model</a>：使用 DeepLabv3 分割模型进行道路检测。通过在 GitHub 上创建账户为 SinaRaoufi/SenseTheRoad 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1263805726900490324)** (4 messages): 

> - `XLM-Roberta fine-tuning`
> - `SQL chatbot for Q&A`
> - `RAG concept for chatbots`
> - `Haystack ImportError` 


- **为 Token Classification 微调 XLM-Roberta-large**：一位用户询问如何使用 AutoModel 和 Trainer 在其数据上微调 **XLM-Roberta-large** 以进行 token/text classification。
- **构建 SQL 数据问答聊天机器人**：一位用户寻求构建基于 SQL 数据的**对话式问答聊天机器人**的帮助，请求有相关经验的人提供指导和帮助。
- **RAG 对 SQL 数据聊天机器人是否多余？**：一位用户质疑为 SQL 数据聊天机器人实现 **RAG (Retrieval-Augmented Generation)** 概念是否多余。
- **Haystack ImportError 问题**：一位用户在配合 **Haystack 和 Neo4j** 设置 `Neo4jDocumentStore` 时遇到了 **ImportError**: 'cannot import name 'default_from_dict' from 'haystack''。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1263704081239904369)** (10 messages🔥): 

> - `ANN 中的灾难性遗忘`
> - `睡眠衍生学习`
> - `GenQA 论文见解`
> - `LLaMA-3-8B 微调结果` 


- **研究人员通过类睡眠动力学解决 ANN 中的灾难性遗忘问题**：Maxim Bazhenov 等人的[实验](https://vchs.ucsd.edu/blog/2022/12/sleep-derived-how-artificial-neural-networks-can-avoid-catastrophic-forgetting.html)表明，ANN 中的类睡眠阶段有助于减少灾难性遗忘，研究结果发表在 [Nature Communications](https://www.nature.com/articles/s41467-022-34938-7) 上。
   - ANN 中的睡眠涉及使用局部无监督 Hebbian 可塑性规则和噪声输入的离线训练，帮助 ANN 恢复之前遗忘的任务。
- **对 GenQA 合成数据生成方法的不同看法**：一位成员批评 **GenQA** 论文的合成数据生成方法过于简单，但指出其结果似乎与 **Evol Instruct** 和 **UltraChat** 等方法具有竞争力。
   - 讨论强调了数据集大小的差异（例如 GenQA 的 10M 与过滤后的 Wizard/UltraChat），这让读者对论文中报告的结果感到困惑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://vchs.ucsd.edu/blog/2022/12/sleep-derived-how-artificial-neural-networks-can-avoid-catastrophic-forgetting.html">
			Sleep Derived: How Artificial Neural Networks Can Avoid Catastrophic Forgetting
		</a>：未找到描述</li><li><a href="https://www.nature.com/articles/s41467-022-34938-7">Sleep-like unsupervised replay reduces catastrophic forgetting in artificial neural networks - Nature Communications</a>：众所周知，人工神经网络在最近学习的任务上表现良好，但同时会遗忘之前学习的任务。作者提出了一种无监督睡眠重放算法来恢复...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1263736092373422102)** (2 messages): 

> - `Opus Instruct 3k 数据集`
> - `句子中的单复数主语`
> - `Claude 3 Opus 多轮指令微调` 


- **Opus Instruct 3k 数据集发布**：一位成员分享了 Hugging Face 上 [Opus Instruct 3k 数据集的链接](https://huggingface.co/datasets/kalomaze/Opus_Instruct_3k)，该数据集包含多轮对话。
   - 据指出，该数据集包含约 250 万个 token 的通用多轮指令微调数据，风格模仿 **Claude 3 Opus**。
- **识别单数和复数主语**：一位用户请求 AI 助手帮助识别某些句子包含的是单数还是复数主语。
   - AI 助手提供了分析，指出像 *'Chicken with rice and beans'* 这样的短语尽管提到了多个项目，但仍是单数，而 *'Australia and New Zealand'* 则是复数主语。
- **Claude 3 Opus 多轮指令数据集**：Opus Instruct 3k 数据集包含由模型自身生成的多轮对话，模仿 **Claude 3 Opus**。
   - *teknium* 对该数据集的重要性给予了积极评价。



**提到的链接**：<a href="https://huggingface.co/datasets/kalomaze/Opus_Instruct_3k">kalomaze/Opus_Instruct_3k · Datasets at Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1263842125209145417)** (2 messages): 

> - `关于 AI 的 YouTube 视频`
> - `Claude 的文本处理能力` 


- **分享关于 AI 进展的 YouTube 视频**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=CA-VUk2yLZU)链接，可能讨论了 AI 的最新发展或见解。
   - *AI 爱好者应该观看此视频以了解最新趋势。*
- **Claude 对文本处理请求的幽默回应**：一条有趣的 [推文](https://x.com/emollick/status/1813753156431384851)展示了 Claude 对一个奇怪请求的独特处理：从小说《西线无战事》（*All Quiet on the Western Front*）中“移除鱿鱼”。
   - Claude 的完美回复：“文档中没有任何关于鱿鱼的提及”，引发了人们的笑声，并对 AI 的理解能力表示赞赏。



**提到的链接**：<a href="https://x.com/emollick/status/1813753156431384851">来自 Ethan Mollick (@emollick) 的推文</a>：👀Claude 处理了一个疯狂的请求：“移除鱿鱼” —— “该文档似乎是 Erich Maria Remarque 所著小说《西线无战事》的全文。其中不包含...”

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1263878783845535764)** (7 messages): 

> - `DCLM models`
> - `language map for codebases`
> - `lumentis project` 


- **性能最强的开源 DCLM 模型发布**：[Vaishaal](https://x.com/Vaishaal/status/1813956553042711006) 宣布在 Huggingface 上发布了他们的 **DCLM 模型**，声称它们是目前可用的性能最强的真正开源模型。
   - *Teknium* 补充道，发布的数据集包含高达 **250T tokens**。
- **语言地图 (Language map) 让与 LLM 的代码库沟通更简单**：[MutableAI](https://x.com/mutableai/status/1813815706783490055) 引入了一种 **language map**，通过将代码转换为具有特定结构的英语，简化了与 LLM 讨论代码库的过程。
   - *Adjectiveallison* 评论了这种方法的创造性，将其与 graphrag 趋势联系起来，并指出其在检索阶段相比于完整的图结构 (full-on graphs) 具有优势。
- **Lumentis 项目自动生成详尽文档**：*Adjectiveallison* 提到了 [Lumentis 项目](https://github.com/hrishioa/lumentis) 的公开发布，该项目可以从代码库中自动生成详尽的文档。
   - *Adjectiveallison* 指出 MutableAI 的方法在此基础上有所改进，将这些生成的文档集成到了检索系统中。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Vaishaal/status/1813956553042711006">来自 Vaishaal Shankar (@Vaishaal) 的推文</a>：我们已经在 Huggingface 上发布了我们的 DCLM 模型！据我们所知，这些是目前性能最好的真正开源模型（开源数据、开源权重模型、开源训练代码）1/5</li><li><a href="https://x.com/mutableai/status/1813815706783490055">来自 mutable.ai (@mutableai) 的推文</a>：http://x.com/i/article/1813813469969543168</li><li><a href="https://github.com/hrishioa/lumentis">GitHub - hrishioa/lumentis: AI powered one-click comprehensive docs from transcripts and text.</a>：基于 AI 的一键式从转录文本和文本生成详尽文档。- hrishioa/lumentis
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1263575565492031550)** (161 条消息🔥🔥): 

> - `GPT-4o Mini`
> - `Mistral-Nemo-Instruct-2407`
> - `CrowdStrike Outages`
> - `Apple DCLM-7B`
> - `Cybersecurity` 


- **GPT-4o Mini 与 GPT-3.5-Turbo 在编程基准测试上的对比**：在 [编程基准测试](https://aider.chat/docs/leaderboards/) 中，**GPT-4o Mini** 的表现与 **GPT-3.5-Turbo** 相当，尽管其宣传的 **HumanEval 分数** 提高了用户的期望。
   - “OpenAI 在基准测试数据上对其进行了训练”导致一位用户对过度炒作的性能指标表示不满。
- **Mistral-Nemo-Instruct-2407 表现优于同类模型**：[Mistral-Nemo-Instruct-2407 的模型卡 (Model Card)](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407) 显示，这款由 Mistral AI 和 NVIDIA 联合微调的模型优于同等规模的其他模型，具有 **128k 上下文窗口**，并包含多语言和代码数据。
- **CrowdStrike 停机事件引发强烈抵制**：CrowdStrike 因导致全球 **技术基础设施停机** 而面临重大批评，一些用户认为这超过了其任何积极贡献。
   - 在为公司对抗 **勒索软件攻击 (ransomware attacks)** 的努力辩护时，另一位用户承认了造成的重大损害，但声称 CrowdStrike 仍然带来了净正面影响。
- **Apple 发布 DCLM-7B 模型**：Apple 发布了 **DCLM-7B** 模型，据报道其表现优于 Mistral 7B，并附带了完全开源的预训练数据集，引发了关于其上下文长度能力的讨论。
   - 尽管最初发布的 [Apple DCLM-7B](https://huggingface.co/apple/DCLM-7B) 仅具有 2k 上下文长度，但用户表示希望未来的迭代版本能提供更长的上下文窗口。
- **DeepSeek 量化显示出良好的前景**：[DeepSeek 1-bit 量化结果](https://huggingface.co/nisten/deepseek-0628-gguf) 在 CPU 推理方面显示出潜力，目前在 LMSYS Arena Hard 排行榜上排名 **全球第 7**。
   - 用户讨论了特定量化技术（如 **IQ1_S**）的影响以及更高上下文长度的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1782903738350416290">anton (@abacaj) 的推文</a>: lol，llama-3 可以通过动态缩放无需训练即可处理 16k+ 上下文</li><li><a href="https://huggingface.co/UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3">UCLA-AGI/Gemma-2-9B-It-SPPO-Iter3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/facebook/chameleon-30b">facebook/chameleon-30b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/TuringsSolutions/737250440678858">Hugging Face 上的 @TuringsSolutions: &quot;介绍：&#39;Synthetic Math Phi&#39;！只需按下一个按钮，即可接收……&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/apple/DCLM-7B-8k">apple/DCLM-7B-8k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/nisten/deepseek-0628-gguf">nisten/deepseek-0628-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/casper_hansen_/status/1814269340100751382">Casper Hansen (@casper_hansen_) 的推文</a>: Apple 发布了一个击败 Mistral 7B 的 7B 模型 - 但最劲爆的是他们完全开源了一切，包括预训练数据集 🤯 https://huggingface.co/apple/DCLM-7B</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://time.com/6802011/gen-z-financial-scams-fraud/">为什么 Z 世代出人意料地容易受到金融诈骗</a>: 与婴儿潮一代相比，Z 世代陷入网络诈骗的可能性高出三倍多。专家分析了原因</li><li><a href="https://arxiv.org/abs/2407.10817">Foundational Autoraters: Taming Large Language Models for Better Automatic Evaluation</a>: 随着大型语言模型 (LLMs) 的进步，由于人工评估的高昂成本，可靠地评估其输出变得更具挑战性。为了在更好的 LLM 自动评分器方面取得进展，我们引入了……</li><li><a href="https://x.com/natolambert/status/1814024567192748166">Nathan Lambert (@natolambert) 的推文</a>: GPT4-o-mini 在 reward bench 上超过了 claude 3 sonnet (不是 3.5) 和 llama 3 70b，低于 gemma 2 27b。实际上所有这些都很相似。已经相当饱和了。</li><li><a href="https://x.com/corbtt/status/1814056457626862035">Kyle Corbitt (@corbtt) 的推文</a>: 在我们的内部 LLM-as-judge 评估中，gpt-4o mini 绝对压倒了 gpt-4o。所以我查看了数据 (感谢 @HamelHusain) 并意识到它在回答问题的同时，还加入了一堆……</li><li><a href="https://huggingface.co/datasets/N8Programs/PeriodConvo">N8Programs/PeriodConvo · Hugging Face 数据集</a>: 未找到描述</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1263725715275386922)** (10 条消息🔥): 

> - `Mistral-Nemo-Instruct GGUF 转换`
> - `Ollama 模型问题`
> - `Tekken Tokenizer 与 Llama.cpp`
> - `预训练模型作为 Embeddings` 


- **Mistral-Nemo-Instruct GGUF 转换困扰**：一位成员在将 **Mistral-Nemo-Instruct** 转换为 **GGUF** 时遇到了困难，原因是 **BPE vocab** 问题和缺失 tokenizer.model 文件。
   - 尽管拉取了一个支持 **Tekken tokenizer** 的 PR，转换脚本仍然无法工作，令人非常沮丧。
- **Ollama 模型加载时崩溃**：一位成员报告在 **Ollama** 上运行 **Mistral-Nemo-Instruct-12b** 时出现了张量维度不匹配错误。
   - 加载模型时显示 **'blk.0.attn_q.weight'** 的张量形状不匹配错误。
- **Tekken 与 Sentencepiece tokenizers**：讨论强调 **llama.cpp** 和 **ollama** 尚不支持 **Tekken tokenizer**，后者像 OpenAI 模型一样使用 **tiktoken**。
   - 目前的转换器严重依赖 **sentencepiece**，这使得使用 **Tekken** 的模型转换变得复杂。
- **为什么在检索流水线中使用预训练 embeddings？**：在检索流水线的背景下，有人提出了“为什么我们可以使用预训练模型作为 embedding”的问题。
   - 这表明人们有兴趣了解预训练模型在 embedding 任务中的作用和优势。


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1263697693398667355)** (21 条消息🔥): 

> - `Triplex LLM`
> - `Knowledge Graphs`
> - `R2R`
> - `RAG 应用`
> - `Neo4j 与 PropertyGraphStore` 


- **Triplex 将 KG 构建成本降低了 98%**：Triplex 是由 [SciPhi.AI](https://www.sciphi.ai) 对 Phi3-3.8B 进行微调后的版本，在从非结构化数据创建知识图谱（Knowledge Graphs）方面，其性能超越了 GPT-4，而成本仅为 1/60。
   - 它支持使用 SciPhi 的 R2R 平台进行本地图谱构建，显著降低了开支。
- **R2R 弥合了本地 LLM 与可扩展 RAG 之间的差距**：[R2R](https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph) 平台被称为“RAG 界的 Supabase”，专为具有多模态支持和混合搜索能力、可扩展且生产就绪的检索增强生成（RAG）应用而设计。
   - 关键特性包括用于构建知识图谱的自动关系提取、用于文档和用户管理的完整身份验证，以及用于性能分析的可观测性。
- **将 Triplex 与 Neo4j 结合用于实体关系提取**：成员们使用 Neo4j PropertyGraphStore 配合 Triplex，通过[集成 API](https://r2r-docs.sciphi.ai/api-reference/introduction) 从公司文档中提取实体及其关系。
   - 他们成功查询了数据以用于实际应用，例如列出曾在 Google 工作过的 YC 创始人，并得到了合理的回复。
- **Graph RAG 增强了通用问答任务**：成员们讨论了微软的 GraphRAG 如何将知识图谱能力扩展到更主观的数据集，从而增强通用问答任务的 RAG 方法。
   - 这允许进行详尽的群体级查询，证明了在处理复杂、非确定性查询时的实用性。
- **探索其他图构建工具**：成员们考虑尝试 Nebula Graph，因为它具有平台无关的知识图谱构建能力。
   - 他们注意到提取的三元组是模式无关（schema-independent）的，使其能够与任何知识图谱提供商兼容。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph.">简介 - 最好的开源 AI 驱动问答引擎。</a>: 未找到描述</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1263594417298735164)** (3 条消息): 

> - `WorldSim 问题`
> - `服务器停机解决` 


- **WorldSim 面临停机**：一位成员报告 **WorldSim** 无法工作，引发了对其可用性的担忧。
   - 另一位成员保证问题很快会得到解决，最后确认已修复，并感谢了最初的报告者。
- **服务器停机问题迅速解决**：**WorldSim** 的停机问题由团队成员迅速处理并解决。
   - 修复工作实施迅速，负责的团队成员对社区的耐心表示了感谢。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1263572406090465378)** (174 条消息🔥🔥): 

> - `GPT-4o mini 能力`
> - `语音能力推测`
> - `Crowdstrike 停机事件的影响`
> - `GPT-4o mini 的 API 使用`
> - `AI 模型之间的对比` 


- **GPT-4o Mini 因使用限制缺少图像支持**：一位用户注意到 **GPT-4o mini** 无法识别图像且缺少上传图像的选项，引发了关于其局限性的讨论。
   - 另一位成员解释说，由于它是作为 **GPT-3.5** 的更便宜、智能程度稍低的替代品，因此缺少图像支持。
- **全球 Windows 停机事件**：由于 **Crowdstrike** 的一次错误更新，导致了严重的全球性停机，影响了航空公司、银行和医院等众多行业。
   - 由于加密原因，机器需要手动解锁，导致修复过程缓慢且昂贵，主要影响 **Windows 10** 用户。
- **Sira 发布时间尚不确定**：用户推测 **Sira** 的发布情况，想知道它是否会在 API 中提供。
   - 目前还没有确认 **Sira** 会向所有人开放，用户希望很快能获得更完整的功能访问权限。
- **AI 模型准确率对比**：成员们对比了 **Claude Opus** 和 **GPT-4o 家族** 在技术任务中的准确率。
   - 一些用户发现 **Claude Opus** 在技术任务上不如 **GPT-4o** 可靠，而 **Sonnet 3.5** 在解决复杂问题方面的能力也稍逊一筹。



**提到的链接**：<a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账号来为 openai/simple-evals 的开发做出贡献。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1263658060648546374)** (12 条消息🔥): 

> - `4o vs. 4o-mini`
> - `GPT-4 Turbo 对比`
> - `微调 4o mini`
> - `ChatGPT 会话清理` 


- **4o-mini 具有速度和成本优势**：成员们讨论认为 **4o mini** 的速度大约比 **4o** 快 **2 倍**，且比 3.5 Turbo 便宜 **60%**，使其成为一种极具成本效益的选择。然而，正如[此处](https://github.com/openai/simple-evals)所述，它在基准测试中的得分低于 4o。
   - *
- **GPT-4o 在基准测试中表现更优**：**GPT-4o** 在基准测试中的得分高于 **GPT-4 Turbo**，但实际效果取决于具体的使用场景。
   - 关于谁具有绝对优势尚未达成共识，因为使用场景的多样性起着重要作用。
- **预计 4o mini 的微调将在六个月内推出**：一位成员询问了 **4o mini** 的微调时间表，得到的估计大约是 **6 个月**。
   - *
- **手动清理 ChatGPT 会话非常繁琐**：一位用户询问是否有办法快速清理 ChatGPT 中的多个会话。回复指出，除非删除所有聊天记录，否则目前该过程仍需手动操作。
   - 用户希望未来在这方面能有提升使用体验的改进。



**提到的链接**：<a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账号来为 openai/simple-evals 的开发做出贡献。

  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1263774760291274782)** (4 条消息): 

> - `代码片段库的毛玻璃 UI`
> - `避免不需要的 AI 标记`
> - `提示工程建议` 


- **创建毛玻璃效果的代码片段库**：一位用户请求使用 HTML、CSS 和 JavaScript 创建一个具有 **毛玻璃 UI (Glassmorphic UI)** 和动态渐变背景的代码片段库。
   - 它应具备代码片段管理功能，如 **添加、查看、编辑和删除片段**，并为每个片段包含一个“复制”按钮。
- **避免不需要的 AI 标记**：一位成员对 AI 在回复中出现类似 **“【5:15†source”** 的标记表示担忧，并提供了一个 AI 对该标记含义解释不佳的例子。
   - 他们寻求提示工程（Prompt Engineering）方面的建议，以 **防止 AI 出现这类回复**。

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1263774760291274782)** (4 messages): 

> - `Prompt Engineering for file_search`
> - `Dynamic Glassmorphic UI Library` 


- **避免非预期标注的 Prompt Engineering 建议**：一名成员寻求关于 Prompt Engineering 的建议，以避免 AI 在进行 file_search 时返回包含类似 '【5:15†source' 这种标注的响应。
   - 尝试过的解决方案包括明确要求避免此类回复，但问题依然存在，因此请求进一步的提示。
- **创建动态 Glassmorphic UI 库**：有人请求使用 **HTML, CSS, and JavaScript** 创建一个动态且视觉美观的代码片段库，其特点是具有动画渐变背景、Glassmorphic UI 以及管理代码片段的功能。
   - 功能包括添加、查看、编辑、删除代码片段以及将代码复制到剪贴板，同时保持响应式设计和跨浏览器兼容性。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1263571241936355460)** (69 条消息🔥🔥): 

> - `Mojo 中的 GPU 支持`
> - `为 Mojo 学习底层编程概念`
> - `Mojo 中的 Socket 实现`
> - `在网络处理中选择 epoll 还是 io_uring`
> - `io_uring 的安全问题` 


- **Mojo 中的 GPU 操作并行化**：Mojo 将允许直接并行化操作，利用与 NVidia 的合作伙伴关系来启用 CUDA/NVidia 支持，预计很快就会宣布。
   - 为了获得更高的控制权，开发者可以在 MAX 中使用自定义的 Mojo kernel，而偏好自动化的开发者可以让编译器进行管理。
- **从 Python 转向 Mojo 进行底层编程**：一位用户分享了他们从 Python 转向 Mojo 的顾虑，并考虑先学习 C、CUDA 和 Rust，担心 Mojo 缺乏文档。
   - 社区成员强调了“复杂性的渐进式披露”（Progressive Disclosure of Complexity）的概念，并鼓励询问和记录学习历程，以帮助构建 Mojo 的生态系统。
- **在 Mojo 中实现 Socket 功能**：讨论围绕寻找简洁的 Socket 实现展开，建议 Rust 的实现可能是一个很好的参考，尽管存在对 “ifdef 地狱”的担忧。
   - 成员们强调需要优先考虑像 Linux 的 io_uring 这样基于完成（completion-based）的 API，因为它比传统的轮询（polling）API 具有性能优势。
- **比较 Mojo 网络处理中的 epoll 和 io_uring**：分享了 [Tigerbeetle](https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue) 的抽象见解，强调了使用 io_uring 和 kqueue 相比 epoll 在统一化问题上更少的优势。
   - 建议优先选择 io_uring 而非 epoll，以获得更高的性能和统一的基于完成的 API 处理。
- **解决 io_uring 的安全性问题**：有人对 io_uring 的漏洞表示担忧，指出根据 Google 的报告，2022 年 60% 的 Linux kernel 漏洞针对的是 io_uring。
   - 尽管存在安全担忧，社区认为持续的加固使得 io_uring 优于其他替代方案，因为即使性能降低 50%，它仍然比 epoll 快。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.kernel.org/networking/tls.html">Kernel TLS &#8212; Linux Kernel 文档</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=5znybwzUZog)">FreeBSD 上 PostgreSQL 的异步和直接 I/O - Thomas Munro</a>: 完整描述见 https://www.bsdcan.org/events/bsdcan_2022/schedule/session/90-asynchronous-and-direct-io-for-postgresql-on-freebsd/</li><li><a href="https://www.youtube.com/watch?v">YouTube</a>: 无描述</li><li><a href="https://tigerbeetle.com/blog/a-friendly-abstraction-over-iouring-and-kqueue">一个面向程序员的 io_uring 和 kqueue 之上的 I/O 抽象</a>: 为未来 30 年联机事务处理（OLTP）提供动力的金融交易数据库。</li><li><a href="https://github.com/dmitry-salin/io_uring">GitHub - dmitry-salin/io_uring: 适用于 Mojo 的 io_uring 库</a>: 适用于 Mojo 的 io_uring 库。通过在 GitHub 上创建账号来为 dmitry-salin/io_uring 的开发做出贡献。</li><li><a href="https://man7.org/linux/man-pages/man7/sctp.7.html">sctp(7) - Linux 手册页</a>: 无描述</li><li><a href="https://github.com/bytecodealliance/rustix/tree/main/src/net">rustix/src/net at main · bytecodealliance/rustix</a>: POSIX 风格 API 的安全 Rust 绑定。通过在 GitHub 上创建账号来为 bytecodealliance/rustix 的开发做出贡献。</li><li><a href="https://github.com/rust-lang/rfcs/blob/master/text/3128-io-safety.md">rfcs/text/3128-io-safety.md at master · rust-lang/rfcs</a>: Rust 变更的 RFC。通过在 GitHub 上创建账号来为 rust-lang/rfcs 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1263660857896730694)** (18 messages🔥): 

> - `Mojo Debugging`
> - `Developer Tooling`
> - `Mojo Test Debugging`
> - `LLDB-DAP`
> - `WSL Debugging Issues` 


- **Mojo 改进了调试工具**：Mojo 和 MAX 优先开发比传统 Python、C++ 和 CUDA 栈更先进的调试工具，通过扩展到 GPU 提升了调试体验，尤其是在机器学习任务中。[了解更多](https://www.modular.com/blog/debugging-in-mojo)。
   - 频道中的一位开发者表示：*“目标是展示使用 Mojo 进行调试是多么简单且强大”*。
- **简化 Mojo 中的调试设置**：在 VS Code 中设置 Mojo 调试可以通过 [Mojo extension](https://marketplace.visualstudio.com/item) 轻松完成，并可以使用 [LLDB-DAP](https://lldb.llvm.org/resources/lldbdap.html) 适配到其他编辑器。未来的增强功能将允许从 CPU 代码无缝单步执行到 GPU 调用。
   - *“它针对的是没有任何调试经验的通用调试，但也涵盖了目前所有已实现的 Mojo 特性。”*
- **修复 Mojo 测试子文件夹的调试问题**：要在 `mojo test` 的子文件夹中调试测试，请按照用户的建议使用符号链接并添加 main 函数包装器。
   - 讨论中提到的一种变通方法是：*“我通过添加符号链接使其工作……还必须在你的测试中添加 main 函数”*。



**提到的链接**：<a href="https://www.modular.com/blog/debugging-in-mojo">Modular: Debugging in Mojo🔥</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Debugging in Mojo🔥

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1263582230807187642)** (30 messages🔥): 

> - `Alias tuple of FloatLiterals`
> - `Benchmark confusion`
> - `Custom Mojo version installation`
> - `Anti-pattern discussion`
> - `C interop via OpenSSL` 


- **FloatLiterals 的别名元组需要显式声明**：一位用户发现在 nightly 版本中，必须显式声明 `Tuple[FloatLiteral, FloatLiteral](1.0, 2.0)`，因为 `alias Nums = (1.0, 2.0)` 会将类型视为变参包（variadic pack）。
   - 还有建议考虑在仅使用 `FloatLiterals` 时使用 `InlineArray`。
- **Benchmark 工具缺少墙上时间（Wall Time）追踪**：一位用户对 `benchmark` 模块感到困惑，指出它似乎缺少墙上时间追踪，并询问其在仓库中的位置。
   - 该用户分享了一份显示平均时间不一致的基准测试报告，并幽默地用游戏术语承认了自己的困惑。
- **安装自定义 Mojo 版本的指南**：一位用户询问如何安装自定义 Mojo 版本，并被引导至 `bot-help` 频道中的答案。
   - 为了清晰起见，提供了直接链接和进一步的帮助。
- **条件一致性变通方法中的反模式（Anti-Pattern）**：一位用户幽默地将他们的变通方法标记为“反模式”，其他人也同意这看起来像是针对条件一致性（conditional conformance）问题的权宜之计。
   - 现场进行了轻松的交流，用户承诺会提供更好的解决方案以避免吓到别人。
- **通过 OpenSSL 进行 C Interop**：讨论强调了 OpenSSL 非常庞大，一个项目的 `.mojo` 文件达到了 800 MB。
   - 目标被设定为通过对 `mlir-translate --import-llvm` 的输出进行后处理来实现 C Interop。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1263603705752719443)** (5 messages): 

> - `MAX vs openXLA`
> - `Mojo vs JAX`
> - `Custom ops with Mojo` 


- **MAX 与 openXLA 的比较**：成员们讨论了 **MAX** 与 **openXLA** 以及使用 Google 支持 openXLA 架构的 **JAX** 之间的对比。
   - *darkmatter__* 强调，由于 MAX 是惰性的（lazy）并构建计算图，与 JAX 相比，它允许更多的优化机会。
- **Mojo 在多个层面击败 JAX**：一位成员分享道，在基准测试中 **Mojo** 在 CPU 上的表现优于 **JAX**，尽管 JAX 针对多核系统进行了优化。
   - *darkmatter__* 解释说，**Mojo** 具有比 JAX 更好的编译器可见性和优化能力。
- **Mojo 中的自定义算子**：Mojo 允许实现自定义算子（custom operations），提供了比 JAX 或 Python 更多的灵活性。
   - *darkmatter__* 指出，虽然 Mojo 目前在某些方面逊于 C++ 或 Rust，但它在未来的编译器改进方面具有潜力。


  

---

### **Modular (Mojo 🔥) ▷ #[max-gpu](https://discord.com/channels/1087530497313357884/1212827673257316453/1263603396548759745)** (2 messages): 

> - `MAX vs openXLA`
> - `Google's open projects` 


- **MAX 与 openXLA 的比较**：一名成员询问了 **MAX** 与 **openXLA** 的对比，并指出 **Jax** 速度很快，且采用了由 Google 等主要参与者支持的架构。
- **对 Google 开源项目方式的批评**：*OpenXLA* 被批评为主要是一个仅限 Google 内部的项目，人们担心 Google 在处理任何“开源（open）”事务上表现不佳。


  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1263570991422902282)** (17 messages🔥): 

> - `Contributor Meeting & Incubator Alignment`
> - `Community Contribution Value`
> - `Async IO API Standards`
> - `Stdlib Opt-Out`
> - `Mojo Nightly Update 2024.7.1905` 


- **贡献者会议提案评估**：随后讨论了将贡献者会议与社区会议分开的问题，以解决孵化器（incubator）可能与 Modular 的工作不一致的担忧。
   - 会中指出 *Modular* 已表现出集成 stdlib 贡献的兴趣，而孵化器有助于在正式提交到 stdlib 之前评估 API 和受欢迎程度。
- **社区验证 API 提案**：成员们认为，在将提案纳入 stdlib 之前，社区反馈至关重要，以避免出现类似 Rust 等语言中的问题。
   - 诸如 allocator awareness 等特定用例可以从这种社区过滤中受益，正如[此提案](https://github.com/gabrieldemarmiesse/mojo/blob/proposal_stdlib_extensions/proposals/stdlib-extensions.md)中所讨论的。
- **关于 Async IO API 标准的辩论**：一名成员强调 Mojo 需要支持更高性能模型的 Async IO API，通过有效地处理 buffers 来实现。
   - 讨论引用了 Rust 的挑战，强调要避免在性能导向库和主流库之间产生分裂。
- **退出 Stdlib 的可能性**：成员们讨论了在 Mojo 中禁用或退出 stdlib 的可能方式。
   - 虽然 Mojo 目前仅包含已使用部分的处理机制使其需求降低，但它被类比为 Rust 的 `no_std` 特性。
- **Mojo Nightly 编译器更新 2024.7.1905**：[Mojo Nightly 编译器更新 2024.7.1905](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 已发布，其特点是新增了 stdlib 函数 `Dict.setdefault(key, default)`。
   - [查看原始 diff](https://github.com/modularml/mojo/compare/bb7db5ef55df0c48b6b07850c7566d1ec2282891...f8d9214ac31da76bb679f867f57b255b65d9a31a) 以了解详细变更。



**提到的链接**：<a href="https://github.com/gabrieldemarmiesse/mojo/blob/proposal_stdlib_extensions/proposals/stdlib-extensions.md#the-future-of-this-repository-when-mojo-has-a-public-source-of-truth">mojo/proposals/stdlib-extensions.md at proposal_stdlib_extensions · gabrieldemarmiesse/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 gabrieldemarmiesse/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/)** (1 messages): 

punishedjamesthesnake: nice
  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1263576801314345064)** (83 条消息🔥🔥): 

> - `Mistral Nvidia 合作`
> - `带有 RAG 的 LM Studio Server`
> - `Open WebUI 功能`
> - `适用于 AMD GPU 的 SCALE 工具链`
> - `自定义 HF 模型集成` 


- **Mistral Nvidia 合作发布**：[Mistral Nvidia 合作](https://mistral.ai/news/mistral-nemo/) 介绍了 Mistral-Nemo 12B，提供大上下文窗口和最先进的性能，但目前 LM Studio 尚未支持。
   - 需要 llama.cpp 的 **Tokenizer 支持** 才能使 Mistral-Nemo 兼容。
- **在 LM Studio 中实现 RAG 和 TTS**：成员们讨论了如何在 LM Studio 中实现 RAG 和 TTS 功能，建议将 **Open WebUI** 作为已经支持这些功能的替代方案。
   - 推荐使用 **ChatterUI** 和 **Msty** 等多个前端，通过不同设备访问 LM Studio server。
- **Open WebUI 的精彩功能**：[Open WebUI](https://github.com/open-webui/open-webui) 提供丰富的功能，包括 TTS、RAG 和无需 Docker 的互联网访问。
   - 用户分享了在 Windows 10 上安装的积极体验及其提供的灵活性，并有兴趣将其性能与 **Pinokio** 进行对比。
- **SCALE 工具包让 AMD GPU 支持 CUDA**：来自 Spectral Compute 的新 [SCALE 工具包](https://scale-lang.com/posts/2024-07-12-release-announcement) 允许 CUDA 应用程序在 AMD GPU 上轻松运行，简化了软件迁移。
   - 尽管这是一个创新性的飞跃，用户提到其缺点是并非开源。
- **将自定义 HF 模型集成到 LM Studio**：一位用户寻求关于将基于 llama 3 的自定义 HF 模型集成到 LM Studio 的指导，收到的建议是转换为 GGUF 并向 llama.cpp 和 Hugging Face 提交 PR。
   - 建议联系 Hugging Face 上的专家（如 **mradermacher**）以获取有关模型转换的进一步帮助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://youtu.be/ejGbF3QghFA">具有记忆功能的对话式 AI 助手概览 - AI 代码解析</a>: 探索我们先进的具有记忆功能的对话式 AI 助手的能力！🌟 在本视频中，我们将详细介绍一个复杂的 AI 系统...</li><li><a href="https://dou.ua/forums/topic/49408/">Як я розробив чат-бот зі штучним інтелектом</a>: 在这篇文章中，Serhiy Trush 讲述了他过去三个月一直在进行的长期开源 side project 之一。这是关于 TelegramAIChatbot 仓库——一个乌克兰语的 Telegram 聊天机器人...</li><li><a href="https://www.tomshardware.com/tech-industry/new-scale-tool-enables-cuda-applications-to-run-on-amd-gpus">新的 SCALE 工具使 CUDA 应用程序能够在 AMD GPU 上运行</a>: 通过为 AMD GPU 重新编译 CUDA 程序</li><li><a href="https://pinokio.computer/.">Pinokio</a>: AI 浏览器</li><li><a href="https://cloud.google.com/use-cases/retrieval-augmented-generation">什么是检索增强生成 (RAG)？ | Google Cloud</a>: 检索增强生成 (RAG) 将 LLM 与外部知识库相结合，以改进其输出。通过 Google Cloud 了解更多信息。</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: 适用于 LLM 的用户友好型 WebUI（原名 Ollama WebUI）</a>: 适用于 LLM 的用户友好型 WebUI（原名 Ollama WebUI） - open-webui/open-webui</li><li><a href="https://github.com/vosen/ZLUDA">GitHub - vosen/ZLUDA: AMD GPU 上的 CUDA</a>: AMD GPU 上的 CUDA。通过在 GitHub 上创建账户为 vosen/ZLUDA 做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1263628709076668456)** (26 messages🔥): 

> - `DeepSeek-V2-Chat-0628`
> - `GGUF model performance` (GGUF 模型性能)
> - `Model VRAM requirements` (模型 VRAM 需求)
> - `Custom dataset creation` (自定义数据集创建)
> - `New jail-breaking technique for frontier models` (针对前沿模型的新越狱技术)


- **DeepSeek-V2-Chat-0628 在 LMSYS 排行榜名列前茅**：[DeepSeek-V2-Chat-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 拥有 **236B 参数**，现已开源，并在 [LMSYS Chatbot Arena 排行榜](https://chat.lmsys.org)上被评为排名第一的开源模型。
   - 它占据了领先位置：总榜第 11，困难提示词 (Hard Prompts) 第 3，编程 (Coding) 第 3，长查询 (Longer Query) 第 4，数学 (Math) 第 7。
- **GGUF 模型在 VRAM 上的效率**：关于拥有超过 21GB 的 VRAM 是否能提高性能的讨论指出，将所有 **236B 参数**加载到 VRAM 或 RAM 是最优选择。
   - 讨论强调，即使无法完全加载整个模型，尽可能多地装入 VRAM 也会有所帮助。
- **轻松创建自定义数据集**：一位新手用户询问创建自定义数据集的最简单方法。
   - 他们被引导使用 [augmentoolkit](https://github.com/e-p-armstrong/augmentoolkit) 将计算资源和书籍转换为指令微调 (instruct-tuning) 数据集或分类器。
- **针对前沿模型的新越狱技术**：一位用户分享了一种对前沿模型有效的[新越狱技术](https://arxiv.org/pdf/2407.11969)。
   - 他们建议在技术尚未被修复 (patched) 前使用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1813921111694053644">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🎉激动人心的消息！我们开源了 DeepSeek-V2-0628 检查点，这是 LMSYS Chatbot Arena 排行榜 @lmsysorg 上排名第一的开源模型。详细 Arena 排名：总榜第 11，困难提示词第 3，Co...</li><li><a href="https://huggingface.co/bullerwins/DeepSeek-V2-Chat-0628-GGUF">bullerwins/DeepSeek-V2-Chat-0628-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit.git">GitHub - e-p-armstrong/augmentoolkit: 将计算资源和书籍转换为指令微调数据集（或分类器）！</a>: Convert Compute And Books Into Instruct-Tuning Datasets (or classifiers)! - e-p-armstrong/augmentoolkit
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1263902099553390684)** (5 messages): 

> - `Mistral BPE`
> - `LM Studio Compatibility` (LM Studio 兼容性)
> - `llama.cpp Support` (llama.cpp 支持)
> - `lmdeploy RAM Limitation` (lmdeploy RAM 限制)


- **LM Studio 中的 Mistral BPE 分词问题**：一位用户在尝试加载模型时遇到了 `llama.cpp error`，提示未知的预分词器 (pre-tokenizer) 类型 'mistral-bpe'。
   - *它无法在此版本的 LM Studio 中运行*，且经另一位用户确认，`llama.cpp` 尚未支持该类型。
- **llama.cpp 增加了对 BPE 预分词的支持**：一位用户指出 `llama.cpp` 已在 [PR #6920](https://llama.cpp/pull/6920) 中增加了 BPE 预分词支持。
   - 另一位用户提到 LM Studio 的 *llama.cpp 版本稍显滞后*，因此可能需要一两次更新才能看到支持。
- **LM Studio 与 lmdeploy 的兼容性**：一位用户尝试在 24GB RAM 环境下使用 `lmdeploy`，但发现内存不足。
   - 这表明当前的硬件可能无法满足使用 `lmdeploy` 成功部署模型的 RAM 需求。


  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1263656262441238640)** (10 条消息🔥): 

> - `LLM 硬件的未来`
> - `TSMC AI 芯片供应预测`
> - `在 Windows 上运行 NVidia Tesla P40`
> - `Tesla P40 的 Vulcan 支持`
> - `NVidia Tesla P40 驱动程序` 


- **在没有昂贵 GPU 的情况下运行 LLM 可能需要时间**：一位用户质疑在未来 1-2 年内，在 PCIE NPU 或 ASIC 等非昂贵 GPU 硬件上运行大型语言模型的可行性。
- **TSMC CEO 表示，到 2025-2026 年 AI 芯片供应才能达到平衡**：TSMC CEO 预测，由于客户需求旺盛和封装瓶颈，先进 AI 芯片的供需平衡要到 2025 年或 2026 年才能实现。
- **在 Windows 上运行 NVidia Tesla P40 的结果参差不齐**：用户分享了在 Windows 10 上将 NVidia Tesla P40 GPU 与其他 GPU 并行运行的经验，并提到了数据中心和 Studio RTX 驱动程序的使用。
   - *一位用户指出，虽然 P40 速度较慢，但仍比 CPU 推理快。*
- **Tesla P40 与 Vulcan 的兼容性问题**：一位用户强调了在 Tesla P40 上获得 Vulcan 支持的障碍，提到可能需要进行多次安装，并且可能需要启用虚拟化。



**提到的链接**：<a href="https://www.theregister.com/2024/07/18/tsmc_ceo_predicts_ai_chip/">TSMC CEO 预测 AI 芯片短缺将持续到 2025... 2026 年</a>：魏哲家坚持认为海外扩张将继续。

  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/)** (1 条消息): 

aptronym: 如果你们有一个便携式安装选项，我就可以
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1263571360999936151)** (87 条消息🔥🔥): 

> - `Llama 3 发布`
> - `Self-Play Preference Optimization (SPPO)`
> - `Sonnet 拒绝回答及其推测`
> - `Apple 开源 DCLM 7B 模型`
> - `Snowflake Arctic Embed 更新` 


- **Llama 3 发布在即**：传闻拥有 **4000 亿**参数的 **Llama 3** 将在 4 天内发布，引发了社区的兴奋和猜测。
- **Self-Play Preference Optimization (SPPO) 论文引起关注**：SPPO (Self-Play Preference Optimization) 因其潜力而受到关注，但对其在几次迭代后的长期有效性存在怀疑。
- **关于 Sonnet 拒绝回答的推测**：Sonnet 的拒绝行为引起了关注，这种行为被描述为教条化，但在反思后又表现出非凡的理性。
- **Apple 开源 DCLM 7B 模型**：Apple 发布了 **DCLM 7B** 模型，该模型超越了 **Mistral 7B**，并且完全开源，包括训练代码和数据集。
- **Snowflake Arctic Embed 1.5 提升检索系统可扩展性**：Snowflake 推出了 **Arctic Embed M v1.5**，通过极小的嵌入向量（embedding vectors）为检索系统带来高达 **24 倍的可扩展性提升**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/casper_hansen_/status/1814269340100751382?s=46">来自 Casper Hansen (@casper_hansen_) 的推文</a>：Apple 发布了一个击败 Mistral 7B 的 7B 模型——但最劲爆的是他们完全开源了一切，包括预训练数据集 🤯 https://huggingface.co/apple/DCLM-7B</li><li><a href="https://x.com/_xjdr/status/1814043484732764167?s=46">来自 xjdr (@_xjdr) 的推文</a>：人们接下来可能想要关注的一系列事物：- 树搜索辅助的合成数据生成 - SPPO - 从超大模型到小型模型的蒸馏 - Mixture of Depth，此外可能相关的还有...</li><li><a href="https://x.com/spacemanidol/status/1813968340744020252?s=46">来自 Daniel Campos (@spacemanidol) 的推文</a>：🚀 介绍 Arctic Embed M v1.5！通过微型嵌入向量，将检索系统的可扩展性提升高达 24 倍。https://huggingface.co/Snowflake/snowflake-arctic-embed-m-v1.5 在 ... 上获得 55+ 分</li><li><a href="https://x.com/vaishaal/status/1813956553042711006?s=46">来自 Vaishaal Shankar (@Vaishaal) 的推文</a>：我们已经在 Hugging Face 上发布了 DCLM 模型！据我们所知，这些是目前性能最好的真正开源模型（开源数据、开源权重模型、开源训练代码）1/5</li><li><a href="https://www.manifoldrg.com/llm-agents/">大语言模型时代的智能数字 Agent</a>：这篇立场论文概述了当前基于 LLM 的 AI Agent 的研究领域和突破。我们强调了关键进展并讨论了每个领域的局限性。</li><li><a href="https://x.com/maksym_andr/status/1813608842699079750?s=46">来自 Maksym Andriushchenko @ ICML'24 (@maksym_andr) 的推文</a>：🚨很高兴分享我们的新论文！🚨 我们揭示了当前拒绝训练（refusal training）方法中一个奇特的泛化差距：只需将有害请求用过去时态重新表述（例如，“如何制作...”）</li><li><a href="https://www.datacomp.ai/dclm/index.html#home">DataComp</a>：未找到描述</li><li><a href="https://x.com/alexreibman/status/1814142347367817443?s=46">来自 Alex Reibman 🖇️ (@AlexReibman) 的推文</a>：祝 @ollama 生日快乐！！这里是即将发布的预览：Ollama Agents 👀 对 LLM 工具调用（tool calls）的原生支持即将到来</li><li><a href="https://x.com/osanseviero/status/1780238572374655298?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Snowflake 刚刚开源了 snowflake-arctic-embed：一系列强大的嵌入模型 🤏2200 万到 3.35 亿参数 💾384-1024 嵌入维度 🔥50-56 MTEB 分数（同尺寸中的 SOTA） 这...</li><li><a href="https://x.com/osanseviero/status/1813971183295156595?s=46">来自 Omar Sanseviero (@osanseviero) 的推文</a>：新的 @SnowflakeDB Arctic Embed 模型！一个 109M 的模型，可以进行嵌入量化 + Matryoshka，从而缩减至 128 字节的字节嵌入向量。https://huggingface.co/Snowflake/snowflake-arctic-e...</li><li><a href="https://x.com/swyx/status/1814095122055025141">来自 swyx 🤞 🔜 SFO (@swyx) 的推文</a>：如果你对用于摘要生成的 LLM 感兴趣，我对 GPT-4o 与 mini 的 @smol_ai 评估已经发布。TLDR：- mini 在某些情况下表现相当或略差 - 但因为 mini 的成本仅为 4o 的 3.5% - 我可以运行 10 个版本...</li><li><a href="https://x.com/repligate/status/1814110855467786722?s=46">来自 j⧉nus (@repligate) 的推文</a>：关于 Claude 3.5 Sonnet 及其拒绝机制：1. Sonnet 倾向于反射性地拒绝某些类型的想法/请求，并会做出荒谬、教条的断言，与其所知的关于该主题的一切相矛盾...</li><li><a href="https://news.ycombinator.com/item?id=40998497">未找到标题</a>：未找到描述</li><li><a href="https://x.com/rohanpaul_ai/status/1814112068796129746?s=46">来自 Rohan Paul (@rohanpaul_ai) 的推文</a>：OpenAI 便宜到无法被击败</li><li><a href="https://x.com/ivory_tang/status/1813973545497907329?s=46">来自 Ivory Tang (@ivory_tang) 的推文</a>：想到空间智能（spatial intelligence）所实现的一切，简直令人震惊……从一张图片生成无限场景，从文本生成 3D 精装房的 360 度视图，执行广泛的任务...</li><li><a href="https://x.com/_philschmid/status/1814274909775995087?s=46">来自 Philipp Schmid (@_philschmid) 的推文</a>：Apple 加入战局！@Apple 刚刚发布了一个 7B 开源 LLM、权重、训练代码和数据集！👀 TL;DR：🧠 7B 基础模型，在开源数据集上训练了 2.5T token 🌐 主要是英文...</li><li><a href="https://x.com/officiallogank/status/1814343684625735714?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：面向 @GoogleAI 开发者的长上下文（long context）使用新指南 ✨ 本指南涵盖了见解和研究，包括：- 不同模态下的性能 - 上下文学习（in-context learning）的工作原理 - 长上下文...</li><li><a href="https://x.com/nickadobos/status/1813626926273380429?s=46">来自 Nick Dobos (@NickADobos) 的推文</a>：OpenAI 不得不让 AI 变笨，好让愚蠢的人类能理解它。引用 OpenAI (@OpenAI)：“我们训练了先进的语言模型来生成文本...”</li>

弱模型可以轻松验证，并发现它...</li><li><a href="https://u.osu.edu/ihudas/">SunLab，OSUNLP 的一部分 | 自然语言处理，人工智能，LLMs 和 Agents</a>：未发现描述</li><li><a href="https://buttondown.email/ainews/archive/ainews-mini-nemo-turbo-lite-smol-models-go-brrr/">[AINews] Mini, Nemo, Turbo, Lite - Smol 模型大爆发 (GPT4o-mini 版本)</a>：第一个 GPT4o Mini 期刊！2024/7/17-2024/7/18 的 AI 新闻。我们为您检查了 7 个 subreddits、384 个 Twitter 和 29 个 Discord（467 个频道和 2324 条消息）....</li><li><a href="https://buttondown.email/ainews/archive/ainews-lskjd/">[AINews] Mini, Nemo, Turbo, Lite - Smol 模型大爆发 (GPT4o 版本)</a>：效率就是你所需要的一切。2024/7/17-2024/7/18 的 AI 新闻。我们为您检查了 7 个 subreddits、384 个 Twitter 和 29 个 Discord（467 个频道和 2324 条消息）....
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1263948175635513405)** (29 messages🔥): 

> - `GitHub 概览`
> - `布局检测`
> - `任务分解`
> - `Mathpix 对比`
> - `数据集创建` 


- **分享了 VikParuchuri 的 GitHub 个人主页**：一位成员分享了 [VikParuchuri 的 GitHub 个人主页](https://github.com/VikParuchuri)，展示了该平台上可用的 **90 个仓库**。
- **布局检测中的经典目标检测**：一位成员询问：*“布局检测是如何工作的？是使用大量训练数据的经典目标检测吗？”*
   - 寻求了关于该方法是否涉及经典目标检测技术的澄清，但消息中未提供具体答案。
- **强调了优秀的任务分解**：任务分解被称赞为有效任务划分的**完美范例**。
   - 成员们讨论了它如何帮助将复杂问题分解为更易于管理的任务。
- **Texify 对比 Mathpix**：有人提出了关于 **Texify** 在功能方面如何与 **Mathpix** 竞争的对比。
   - 消息中未提供关于此对比的进一步细节或答案。
- **阅读顺序模型的训练数据集**：询问阅读顺序模型训练数据集的创建方式，是**手动标注**还是使用启发式方法。
   - 在给出解释后，反馈是*“非常感谢！！”*，但未包含具体的步骤或方法。



**提到的链接**：<a href="https://github.com/VikParuchuri">VikParuchuri - 概览</a>：VikParuchuri 有 90 个可用仓库。在 GitHub 上关注他们的代码。

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1263614962933170197)** (5 messages): 

> - `Nvidia 开源内核模块`
> - `反垄断法的影响`
> - `兼容性和维护优势` 


- **美国反垄断法可能影响了 Nvidia**：有推测称，由于反垄断法，Nvidia 被美国**强制**开源其内核模块。
   - 一位用户建议：*“我的猜测是，维护内核模块不是 Nvidia 的核心业务，因此将其开源可能会在无需保留高技能内核开发人员的情况下实现更好的兼容性”。*
- **关于 Nvidia 开源原因的辩论**：另一种观点认为，**维护内核模块并非 Nvidia 业务的核心**，认为开源可以提高兼容性并减少对专业开发人员的需求。
   - 辩论强调了 Nvidia 可能通过此举获得的潜在运营和战略利益。

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1263596276621250710)** (15 条消息🔥): 

> - `PyTorch 中的 Float8`
> - `Stochastic Rounding`
> - `用于 DDP 和 FSDP 的 Multi-GPU 设置`
> - `INT8 权重训练`
> - `Quantization Aware Training` 


- **Float8 权重引入了从 BF16 的动态转换**：多位成员讨论了在 PyTorch 中将以 BF16 存储的权重动态转换为 FP8 以进行 matmul 的方案，并引用了 [float8_experimental](https://github.com/pytorch-labs/float8_experimental)。
   - 成员们还表达了在 FP8 权重更新中实现 Stochastic Rounding 的兴趣，并可能获得来自 Meta 计算资源的支持。
- **Stochastic Rounding 在 CUDA Math API 中缺乏内置支持**：一位成员指出，[CUDA Math API](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html) 中缺少带有 Stochastic Rounding 的 BF16->FP8 数据类型转换指令，这表明可能需要软件层面的实现。
   - 还讨论了在 DDP 中确保跨 GPU 权重更新一致性以及在处理具有独立 SR 的 FSDP 时的复杂性，这增加了另一层挑战。
- **受 Q-Galore 启发的 INT8 权重训练实验**：成员们对复制 Q-Galore 在使用 INT8 权重预训练 Llama-7B 方面的成功表现出兴趣，并强调了 Stochastic Rounding 的作用。
   - 有人指出 Q-Galore 的方法涉及 BF16 梯度，类似于 float8_experimental 仓库，这可能为 INT8 训练提供见解。
- **Stochastic Rounding 在 Multi-GPU 设置中的潜力**：探讨了 Multi-GPU 设置中的 Stochastic Rounding，并深入讨论了它如何影响数据并行性以及跨 GPU 的权重一致性。
   - 对在 PyTorch 中使用 `.view(dtype)` 来平衡数据类型要求的可行性提出了疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/main/torch/optim/adam.py#L415),">pytorch/torch/optim/adam.py at main · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/optim/adamw.py#L379)">pytorch/torch/optim/adamw.py at main · pytorch/pytorch</a>：Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1263788094331687037)** (5 条消息): 

> - `混合分布式算法`
> - `Ring Attention 内存计算`
> - `Sequence Parallelism 论文`
> - `反向传播计算`
> - `私人导师咨询` 


- **学生寻求混合分布式算法方面的帮助**：一名学生正在学习 SP (Ulysses)、TP 和 PP 等 **混合分布式算法**，并正在寻找愿意提供帮助和解答问题的私人导师。
   - 他们对特定的计算（如内存和通信复杂度）有很多疑问。
- **Ring Attention 内存计算查询**：该学生请求澄清在使用 **Ring Attention 风格的 SP** 时如何计算内存，特别是数值 **32** 和 **4** 的来源。
   - 该学生提到“我有很多类似这样的问题”，表明其对深入理解该主题有浓厚兴趣。
- **Sequence Parallelism 论文讨论**：另一位用户请求提供 [Sequence Parallelism 论文链接](link)，以便更好地理解并针对所询问的计算方法提供建议。
   - 他们澄清了需要了解这些问题是关于通用的反向传播计算，还是与论文相关的特定细节。


  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1263624684415484098)** (25 条消息🔥): 

> - `tinygrad 中的 FSDP 支持`
> - `Together Inference Engine`
> - `tinygrad 悬赏任务`
> - `Rust CUDA kernel`
> - `tinygrad 教程` 


- **讨论了 Tinygrad 悬赏列表**：成员们讨论了为 **tinygrad** 贡献代码的几项悬赏任务，包括 [拆分 UnaryOps.CAST](https://github.com/tinygrad/tinygrad/pull/4487) 和 [转换 BinaryOps.DIV](https://github.com/tinygrad/tinygrad/pull/4887) 等具体任务。
   - 一些人认为报酬与付出不符，并指出了高昂的 GPU 计算成本。
- **Tinygrad 的 FSDP 支持**：一位成员悬赏 **$500** 为 **tinygrad** 添加 FSDP 支持，引发了关于该任务可行性和价值的讨论。
   - 一位用户评论说他们可以完成，但需要 **一到两周** 时间，并认为 $500 的报价 **“低得离谱”**。
- **Together Inference Engine 发布**：Together AI 发布了一个 [全新的推理栈](https://www.together.ai/blog/together-inference-engine-2)，其性能优于开源和商业解决方案，在 Meta Llama 3 8B 上达到了 **每秒 400 个 token**。
   - 他们还推出了 **Together Turbo 和 Together Lite**，为企业提供性能、质量和价格方面的灵活性。
- **Tinygrad 学习笔记发布**：一位成员分享了 [tinygrad 学习笔记](https://mesozoic-egg.github.io/tinygrad-notes/)，旨在帮助用户理解 **tinygrad** 的内部原理。
   - 这些笔记包括 [快速入门指南](https://github.com/tinygrad/tinygrad/blob/master/docs/quickstart.md) 以及关于 kernel fusion、GPU 代码生成等方面的详细信息。
- **使用 Rust 创建 CUDA Kernel**：一位用户分享了 [cubecl](https://github.com/tracel-ai/cubecl) 的 GitHub 仓库链接，这是一个用于 **Rust** 的多平台高性能计算语言扩展。
   - 这允许使用 **comptime 系统** 创建 CUDA kernel，以实现特化和最佳性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mesozoic-egg.github.io/tinygrad-notes/">Tutorials on Tinygrad</a>：tinygrad 教程</li><li><a href="https://tenstorrent.com/hardware/tt-quietbox?utm_source=morethanmoore">TT-QuietBox</a>：TT-QuietBox 液冷桌面工作站为希望运行、测试和开发 AI 模型，或为 HPC 移植和开发库的开发者提供了卓越的性价比。</li><li><a href="https://www.together.ai/blog/together-inference-engine-2">Announcing Together Inference Engine 2.0 with new Turbo and Lite endpoints</a>：未找到描述</li><li><a href="https://github.com/cloneofsimo/min-fsdp">GitHub - cloneofsimo/min-fsdp</a>：通过在 GitHub 上创建账号来为 cloneofsimo/min-fsdp 的开发做出贡献。</li><li><a href="https://docs.google.com/spreadsheets/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/edit?gid=0#gid=0">Bounties</a>：tinygrad 悬赏任务：简短描述、价值、链接、GitHub 所有者。例如：将 UnaryOps.CAST 拆分为 UnaryOps.CAST 和 UnaryOps.BITCAST，$100...</li><li><a href="https://github.com/tracel-ai/cubecl">GitHub - tracel-ai/cubecl: Multi-platform high-performance compute language extension for Rust.</a>：用于 Rust 的多平台高性能计算语言扩展。 - tracel-ai/cubecl
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1263571498019328132)** (3 条消息): 

> - `Nsight Compute 文件导出`
> - `Nsight Compute CLI 用户指南`
> - `打开 ncu-rep 文件` 


- **Nsight Compute 支持导出 profile**：有人建议将捕获的 profile 导出为文件，然后可以通过 Nsight Compute GUI 打开。
- **Nsight Compute CLI 用户指南详情**：[Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html) 的用户指南提供了全面的说明，包括启动目标应用程序和从 nvprof 迁移的章节。
   - 该指南涵盖了使用命令行 profiler 直接将结果打印到命令行或存储在报告文件中的方法。
- **Nsight Compute 可打开 ncu-rep 文件**：Nsight Compute 可以打开 `ncu-rep` 文件，为用户分析结果提供了灵活性。



**提到的链接**：<a href="https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html">4. Nsight Compute CLI &mdash; NsightCompute 12.5 文档</a>：未找到描述

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1263574100664651896)** (7 条消息): 

> - `FSDP2 采用`
> - `结合 FSDP2 的低比特优化器`
> - `低比特优化器的 DTensor 支持`
> - `1-bit Adam 优化器` 


- **FSDP2 将取代 FSDP**：提到 **FSDP2** 将取代 **FSDP**，未来必须使用 FSDP2，并以 **nf4** 作为示例实现。
- **低比特优化器与 FSDP2 的兼容性**：与 **FSDP2 作者** 的交流明确了 **低比特优化器** 不需要处理 FSDP 逻辑，因为 FSDP 仍然提供 **fp32 sharded parameters**。
   - **低比特优化器** 可以将这些参数视为输入，而无需担心前向/反向的 FSDP 逻辑。
- **DTensor 与自定义 Subclass 的集成**：讨论了将 **tensor subclass** 与 **DTensor** 集成的问题，包括使用 `distribute_tensor()` 等函数为 subclass 创建 DTensor。
   - 在低比特优化器中处理 DTensor 的 gather 和 scatter 操作被认为是一个重大挑战。
- **用于减少通信开销的 1-bit Adam**：提到了 **1-bit Adam** 通过 **量化梯度** 来减少通信开销的潜力。
   - 承认了其复杂性以及与当前低比特优化方法的不同之处。
- **低比特优化器 DTensor 支持的经验**：一位成员分享了他们在 **DTensor 支持** 方面的经验，以及包裹顺序（order of wrapping）和与 **FSDP2** 可组合性的重要性。
   - 该成员指出，与简单的 tensor 操作相比，实现这些功能引入了一些操作上的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/cbaff6c128d97ff4d26ab60fa5b06c56cd23ba2a/torchao/prototype/low_bit_optim/adam.py#L42-L47),">ao/torchao/prototype/low_bit_optim/adam.py at cbaff6c128d97ff4d26ab60fa5b06c56cd23ba2a · pytorch/ao</a>: 用于训练和推理的自定义数据类型和布局 - pytorch/ao</li><li><a href="https://github.com/pytorch-labs/float8_experimental/blob/7f0d6bbb531d5d76d27d80c9ec3c7eca61de5dfa/float8_experimental/float8_tensor.py#L71">float8_experimental/float8_experimental/float8_tensor.py at 7f0d6bbb531d5d76d27d80c9ec3c7eca61de5dfa · pytorch-labs/float8_experimental</a>: 此仓库包含实验性的 PyTorch 原生 float8 训练 UX - pytorch-labs/float8_experimental</li><li><a href="https://github.com/pytorch-labs/float8_experimental/blob/7f0d6bbb531d5d76d27d80c9ec3c7eca61de5dfa/float8_experimental/float8_ops.py#L236-L268">float8_experimental/float8_experimental/float8_ops.py at 7f0d6bbb531d5d76d27d80c9ec3c7eca61de5dfa · pytorch-labs/float8_experimental</a>: 此仓库包含实验性的 PyTorch 原生 float8 训练 UX - pytorch-labs/float8_experimental
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1263838343570128976)** (1 条消息): 

> - `Gradio 分享链接错误`
> - `Gradio 状态页面` 


- **Gradio 分享链接创建失败**：一位成员在使用 Gradio 时遇到了错误提示：“无法创建分享链接。请检查您的互联网连接或我们的状态页面”。
   - *该成员未提供额外的上下文或链接。*
- **Gradio 状态页面说明**：错误消息中包含一个指向 [Gradio Status Page](https://status.gradio.app) 的链接，该页面提供过去 90 天的运行时间和状态更新。
   - *该成员未提供额外的上下文或链接。*



**提到的链接**: <a href="https://status.gradio.app">Gradio Status</a>: 未找到描述

  

---

### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1263851565358252094)** (2 条消息): 

> - `HQQ+ 2-bit Llama3-8B-Instruct model`
> - `BitBlas 集成性能` 


- **HQQ+ 2-bit Llama3-8B-Instruct 模型发布**：推出了一款[新的实验性模型](https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq) **HQQ+ 2-bit Llama3-8B-Instruct**。该模型采用 BitBlas 后端和 64 group-size 量化，并通过 low-rank adapter calibration 减少了质量损失。
   - 尽管 Llama3-8B 在低比特量化方面非常困难，但该模型据称与 [BitBlas](https://github.com/microsoft/BitBLAS) 和 `torch.compile` 完全兼容，可实现快速推理。
- **讨论 BitBlas 性能问题**：有用户评论称，**BitBlas** 在理论上看起来非常出色，但在端到端集成到模型中时，尤其是在较大的 context sizes 和 batch sizes 下，会出现性能下降。
   - 尽管具有理论优势，但在较大的 context-sizes / batch-sizes 下的*性能退化*被强调为一个亟待解决的问题。



**提到的链接**：<a href="https://huggingface.co/mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq">mobiuslabsgmbh/Llama-3-8b-instruct_2bitgs64_hqq · Hugging Face</a>：未找到描述

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1263618241452965989)** (43 条消息🔥): 

> - `GPT-2 和 GPT-3 训练`
> - `Kernel 优化`
> - `会议讨论`
> - `精度处理`
> - `即将举行的 CUDA MODE IRL` 


- **Yuchen 的 7.3B 模型训练**：Yuchen 使用 **karpathy 的 llm.c** 在 32 台 H100 GPU 上训练了一个 7.3B 模型，达到了 **327K tokens/s** 的速度和 **46.7%** 的 MFU，并被描述为具有**线性扩展性 (linear scaling)**。
   - "由于 7.3B 模型中的某些参数非常大，目前的 llm.c 代码在使用 32 位 int 存储权重字节数并进行 malloc 时会出现整数溢出。我将一些 'int' 更改为 'size_t' 以使其正常工作。"
- **Pull Request 中的 Kernel 优化**：Arund42 向小组通报了一个新的 [简化并优化 backward bias kernel 的 PR](https://github.com/karpathy/llm.c/pull/699)，提到新 kernel 接近通用的列归约 (column reduction)。
   - 实际的 kernel 除去注释只有 **33 行代码**，并计划通过添加 stride 来进一步通用化。
- **即将举行的旧金山 CUDA MODE IRL**：**MarkSaroufim** 邀请成员参加 **9 月 21 日**在旧金山举行的 "CUDA MODE IRL"。
   - 成员们热烈响应，计划在 20 分钟的演讲中讨论构建 **llm.c** 的有趣方面，涵盖 **train_gpt2.c** 的故事以及在 **cuBLAS** 和 **cuDNN** 中的探索。
- **精度处理和 Checkpoint 策略讨论**：展开了关于存储通用的 "始终为 FP32" 的 checkpoint 是否有益的讨论，以便在运行中途轻松更改精度。
   - Eriks.0595 建议当前系统已经以 **FP32** 存储主权重 (master weights) 和优化器状态，因此无需更改 checkpoint 文件格式。
- **内存优化**：有人指出 **llm.c** 在内存优化方面显著优于 **torch.compile**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com">未找到标题</a>：未找到描述</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSdU_L29hXxnCXgXLKMzqHK7Gt-x7jAPMQIXlG-Iut_Qzu4eyQ/viewform">NVIDIA CUDA 和 cuDNN 安装反馈</a>：NVIDIA 希望听取您安装 CUDA 和 cuDNN 的体验！您的匿名反馈对我们改进软件栈的易用性非常重要。</li><li><a href="https://docs.google.com/document/d/10LkM5_xLh9r_ycul2ywOfgrOGmNP9YDTa9c4V755QgY/edit?usp=sharing">CUDA MODE IRL 邀请函</a>：这是对首届 CUDA MODE IRL 黑客松的正式邀请，我们想邀请您进行主题演讲。该活动由 Accel 赞助，将在...举行。</li><li><a href="https://x.com/Yuchenj_UW/status/1814159545280971115">Yuchen Jin (@Yuchenj_UW) 的推文</a>：让我们更进一步！使用 @karpathy 的 llm.c 在 32 台 H100 GPU 上训练 GPT-2 (7.3B)。*GPU 在深夜轰鸣* 🔥 - 设置：4 个通过 400Gb/s InfiniBand 连接的 H100 节点 - 训练速度：327...</li><li><a href="https://github.com/karpathy/llm.c/pull/699">由 ademeure 提交的简化/更快的 "backward bias" kernel (列归约) · Pull Request #699 · karpathy/llm.c</a>：受我昨天为 train_gpt2fp32.cu 制作的简化 kernel 启发，这是一个列归约 kernel（目前专门用于 backward bias，但实际上非常通用），它更简单且...</li><li><a href="https://github.com/karpathy/llm.c/discussions/677">让我们在 llm.c 中复现 GPT-2 (1.6B)：一个 8XH100 节点，24 小时，$672 · karpathy/llm.c · Discussion #677</a>：在这篇文章中，我们正在 llm.c 中复现 GPT-2。这是 "那个 GPT-2"，即 OpenAI 博客文章《更好的语言模型及其影响》中介绍的完整的 1558M 参数版本...</li><li><a href="https://github.com/karpathy/llm.c/pull/694/files)">由 ngc92 提交的模型初始化清理 · Pull Request #694 · karpathy/llm.c</a>：将模型参数分配整合到单一源位置，使梯度缓冲区累积变为 eager 模式，移动了编码器确定性辅助缓冲区，以便由 forward 提前分配 -> ...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1263628936353419366)** (6 条消息): 

> - `Torch 中的 Ring Attention`
> - `使用 torch.compile 生成 Triton Kernel`
> - `用于内存或计算受限检查的算术强度 (Arithmetic Intensity)` 


- **Ring Attention Torch vs. Jax**: 有成员询问是否有 **Torch** 版本的 **Ring Attention** 实现，还是必须使用 **Jax**。
- **使用 torch.compile 生成 Triton Kernel**: 一位用户在尝试通过 `torch.compile` 生成 Triton kernel 时遇到困难，分享了代码片段并遇到了几个问题。
   - 经过指导后，指出张量需要位于 **GPU** 上才能成功编译，这解决了问题。
- **用于内存或计算受限检查的算术强度**: 提出了一个关于使用 **算术强度为 1** 来确定 GPU 任务是内存受限 (memory-bound) 还是计算受限 (compute-bound) 的问题。
   - *这难道不取决于 GPU 的规格，如 DRAM/HBM 的 FLOPS/GB/s 带宽吗？* 以及 *对于某些 GPU，这个比例是否可能高达 20？*


  

---


### **CUDA MODE ▷ #[youtube-watch-party](https://discord.com/channels/1189498204333543425/1238931064223830016/)** (1 条消息): 

mr.osophy: 我喜欢这个主意，我很好奇这些活动进行得怎么样？ <@1221046138249936939>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1263571908222521488)** (96 条消息🔥🔥): 

> - `Claude 3 Haiku vs GPT-4o mini`
> - `Pro 搜索质量下降`
> - `Collection 提示词问题`
> - `Sonnet 3.5 不遵循提示词`
> - `Perplexity Pro 图像生成` 


- **Pro 用户报告搜索质量下降**: 一些成员（尤其是使用 **Claude Sonnet 3.5** 的用户）注意到在过去 8-9 天内 **Pro 搜索** 的质量显著下降。
- **GPT-4o mini 将取代 Claude 3 Haiku？**: 讨论了在 Perplexity 中是否可能用更便宜、更智能的 **GPT-4o mini** 取代 **Claude 3 Haiku**，尽管目前 Haiku 仍在使用中。
- **Collection 提示词无法正常工作**: 用户报告称，在 Collection 中创建的线程里，无论使用哪种 AI 模型，**Collection 提示词** 都未被遵循。
- **Pro 图像生成问题**: 一位 Pro 会员询问为什么订阅后只能创建一张图像，后来发现重启浏览器解决了该问题。
- **Sonnet 3.5 在 Collection 提示词中的问题**: 成员们难以让 **Sonnet 3.5** 遵循 Collection 提示词，在尝试了包括使用 GPT-4o 在内的各种方法后仍未成功。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1263605572679372860)** (9 条消息🔥): 

> - `YouTube Music 的智能电台`
> - `戴森 (Dyson) 的高科技耳机`
> - `基努 (Keanu) 的科幻小说`
> - `OpenAI 的 GPT`
> - `埃隆·马斯克 (Elon Musk) 的奥斯汀总部` 


- **YouTube Music 推出智能电台**: 一场讨论重点介绍了 [YouTube Music 的智能电台](https://www.youtube.com/embed/5lC4KwPFvaE)，其特点是创新的内容交付和新的音乐发现工具。
   - *YouTube Music* 因其智能策划播放列表和适应用户偏好而受到称赞。
- **戴森发布高科技耳机**: 戴森新款 [高科技耳机](https://www.perplexity.ai/search/t6-3al250w-fuse-nc_aBqo8SKm15tV1Kvk3pQ) 因集成了先进的降噪和空气过滤技术而受到关注。
   - 成员们对该产品的双重功能和时尚设计发表了评论。
- **埃隆·马斯克搬迁至奥斯汀总部**: 埃隆·马斯克已将特斯拉总部迁至德克萨斯州奥斯汀，正如最近的 [搜索结果](https://www.perplexity.ai/search/musk-x-headquarters-austin-Xd98i7sMSiuUI3ffTkmmTg) 所讨论的。
   - 这一战略举措旨在利用德克萨斯州对商业友好的环境。
- **OpenAI 发布 GPT-4o**: OpenAI 发布了 [GPT-4o](https://www.perplexity.ai/page/openai-drops-gpt-4o-mini-viKDYptISzufyJDPoL3Etg)，承诺提升语言生成和理解能力。
   - 社区反馈强调了该模型更有效地处理复杂查询的能力。
- **Crowdstrike 遭遇全球 IT 故障**: [Crowdstrike](https://www.perplexity.ai/page/crowdstrike-global-it-outage-qKRKi2QWRuaWxf44d1G5nQ) 面临全球 IT 故障，影响了多个服务并导致运营中断。
   - 该事件引发了人们对云端安全服务可靠性和韧性的担忧。



**提到的链接**: <a href="https://www.youtube.com/embed/5lC4KwPFvaE">YouTube</a>: 未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1263614138010173592)** (4 条消息): 

> - `Online Models Internet Search Capabilities`（在线模型互联网搜索能力）
> - `RAG API Access Inquiry`（RAG API 访问咨询）
> - `ChatGPT 4.0 Mini Internet Browsing`（ChatGPT 4.0 Mini 互联网浏览）
> - `Perplexity API via Azure or Amazon`（通过 Azure 或 Amazon 使用 Perplexity API）


- **在线模型无法搜索互联网**：一位成员询问在线模型是否能够搜索互联网，目前尚未确认具备此能力。
   - 他们对在线模型的能力表示不确定，凸显了广泛的好奇心和潜在的局限性。
- **寻求 Perplexity 的 RAG API 访问权限**：一位成员指出，在发送关于企业级 RAG API 的邮件后未收到回复，寻求进一步协助以获取访问权限。
   - 这表明目前存在沟通挑战，以及对企业级 API 解决方案未满足的需求。
- **ChatGPT 4.0 Mini 缺少互联网浏览功能**：一位成员询问 ChatGPT 4.0 Mini 是否可以浏览互联网，并澄清其无法通过 API 实现此功能。
   - 这凸显了 ChatGPT 4.0 Mini 的能力与用户预期之间的差距。
- **通过 Azure 或 Amazon 使用 Perplexity API**：一位用户询问通过 Azure 或 Amazon 云服务使用 Perplexity API 的可行性。
   - 这表明用户有兴趣将 Perplexity 的能力与领先的云基础设施集成。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1263794828433686558)** (4 条消息): 

> - `Ranking and stats issue fix`（排名和统计数据问题修复）
> - `New models from Mistral AI`（来自 Mistral AI 的新模型）
> - `Router resilience update`（路由弹性更新）
> - `L3-Euryale-70B price drop`（L3-Euryale-70B 降价）
> - `New Dolphin-Llama model`（新的 Dolphin-Llama 模型）


- **排名分析问题已解决**：由于只读副本数据库故障，排名和统计数据显示为陈旧数据，但面向用户的 API 和积分等功能运行正常。
   - **更新**：排名分析和统计数据的问题现已修复。
- **Mistral AI 发布两个新模型**：Mistral AI 推出了 [Mistral Nemo](https://openrouter.ai/models/mistralai/mistral-nemo)，这是一个拥有 12B 参数、128k token 上下文长度的多语言 LLM。
   - [Codestral Mamba](https://openrouter.ai/models/mistralai/codestral-mamba) 也已发布，这是一个拥有 7.3B 参数、256k token 上下文长度的模型，专为代码和推理任务设计。
- **路由弹性功能上线**：新功能现在允许将未在 order 参数中指定的提供商默认作为备选方案，除非明确设置了 `allow_fallbacks: false`。
   - 这意味着在 API 请求中，如果优先的提供商不可用，将尝试其他提供商——从而增强了整体弹性。
- **L3-Euryale-70B 价格下调 60%**：宣布 [sao10k/l3-euryale-70b](https://openrouter.ai/models/sao10k/l3-euryale-70b) 降价 60%。
   - *不仅如此*：[Cognitivecomputations 发布了 Dolphin-Llama-3-70B](https://openrouter.ai/models/cognitivecomputations/dolphin-llama-3-70b)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/sao10k/l3-euryale-70b>)">Llama 3 Euryale 70B v2.1 by sao10k</a>：Euryale 70B v2.1 是来自 [Sao10k](https://ko-fi.com/sao10k) 的专注于创意角色扮演的模型。- 更好的指令遵循。- 更好的解剖学/空间意识。- 能更好地适应独特和复杂的场景...</li><li><a href="https://openrouter.ai/models/cognitivecomputations/dolphin-llama-3-70b)">Dolphin Llama 3 70B 🐬 by cognitivecomputations</a>：Dolphin 2.9 专为指令遵循、对话和编码设计。该模型是 [Llama 3 70B](/models/meta-llama/llama-3-70b-instruct) 的微调版本。它在指令遵循方面表现出改进...</li><li><a href="https://openrouter.ai/models/mistralai/mistral-nemo):">Mistral: Mistral Nemo by mistralai</a>：由 Mistral 与 NVIDIA 合作构建的 12B 参数模型，具有 128k token 上下文长度。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...</li><li><a href="https://openrouter.ai/models/mistralai/codestral-mamba):">Mistral: Codestral Mamba by mistralai</a>：基于 Mamba 的 7.3B 参数模型，专为代码和推理任务设计。- 线性时间推理，理论上允许无限序列长度 - 256k token 上下文窗口 - 针对快速推理进行了优化...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1263838666711760967)** (2 条消息): 

> - `LLM-Draw App`
> - `AI Whispers Prompts Collection` 


- **LLM-Draw 集成了 OpenRouter API keys**：[LLM-Draw](https://github.com/RobinVivant/llm-draw) 应用已更新，现在支持 **OpenRouter API keys**，并利用了 **Sonnet 3.5 self-moderated model**。
   - 它可以作为 **Cloudflare page** 使用 Next.js 进行部署，[此处提供实时应用](https://llm-draw.pages.dev)。
- **AI Whispers Prompts Collection 更新**：[AI Whispers](https://github.com/zielperson/AI-whispers) 正在重新组织用于 **Fabric** 的提示词，并添加 Markdown 结构，包括在 README 文件中添加更详细的信息。
   - *目前，为了更好的组织和清晰度，内容正在进行调整*。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/zielperson/AI-whispers/tree/master">GitHub - zielperson/AI-whispers: testing</a>: testing。通过在 GitHub 上创建账号来为 zielperson/AI-whispers 的开发做出贡献。</li><li><a href="https://github.com/RobinVivant/llm-draw">GitHub - RobinVivant/llm-draw: Make it real</a>: Make it real。通过在 GitHub 上创建账号来为 RobinVivant/llm-draw 的开发做出贡献。</li><li><a href="https://llm-draw.pages.dev">make real starter</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1263572804444356698)** (71 条消息🔥🔥): 

> - `4o mini moderation`
> - `Image tokens billing`
> - `OpenRouter availability`
> - `Gemma 2 repetition issues`
> - `OpenRouter statistics system` 


- ****4o mini 审核机制的模糊性****：关于 **4o mini** 是自审核（self-moderated）还是使用 **OpenAI** 的审核器存在困惑，一些用户遇到了不同的审核行为。
   - 一位用户推测他们的 4o 请求可能被路由到了 **Azure**，而后者的审核阈值较低。
- ****图像 Token 计费不一致的解释****：关于图像 Token 的讨论表明，**OpenRouter** 上的成本是基于分辨率的，但在 Token 计数上存在模糊性。
   - 值得注意的是，基础 Token 用于分析，而总 Token 决定成本，这涉及到 **OpenAI** 的计算方式。
- ****OpenRouter 可用性常见问题****：用户讨论了 OpenRouter 的可用性，并被引导至 [状态页面](https://status.openrouter.ai/) 查看最近的事件。
   - 区域性问题可能会导致服务不可用；**stats system** 最近也面临了数据库副本（DB replica）故障。
- ****Gemma 2 用户面临重复问题****：**Gemma 2 9B** 的用户报告遇到了重复（repetition）问题，并寻求解决建议。
   - 有建议提出使用 **CoT** (Chain of Thought) 提示词以获得更好的性能。
- ****OpenRouter 统计系统停机****：**OpenRouter 统计系统** 出现停机，影响了排名和提供商信息的更新。
   - 停机原因是由于 **DB read replicas** 故障，目前正在修复中，活动页面的数据更新也面临延迟。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mistral-nemo">Mistral: Mistral Nemo by mistralai</a>: 一个拥有 12B 参数、128k Token 上下文长度的模型，由 Mistral 与 NVIDIA 合作构建。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...</li><li><a href="https://openrouter.ai/models/mistralai/codestral-mamba">Mistral: Codestral Mamba by mistralai</a>: 一个基于 Mamba 的 7.3B 参数模型，专为代码和推理任务设计。- 线性时间推理，理论上允许无限序列长度 - 256k Token 上下文窗口 - 针对快速...优化</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter 事件历史
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/1263666508664668300)** (3 条消息): 

> - `Mistral NeMo`
> - `Korean Language Support`
> - `Supported Languages of Mistral NeMo`
> - `daun.ai` 


- **Mistral NeMo 支持韩语**：一条消息指出 **Mistral NeMo** 已添加对韩语的支持。
   - *用户注意到 Mistral NeMo* 在**英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文、日语、韩语、阿拉伯语和印地语**方面表现尤为出色。
- **关于 daun.ai 链接的讨论**：一名成员分享了 **daun.ai** 的链接：[Discord 对话](https://discord.com/channels/1091220969173028894/1092729520181739581/1263886157565923494)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[一般](https://discord.com/channels/1091220969173028894/1246339931337527337/)** (1 条消息): 

k11115555: 没人用...
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1263578489622368350)** (15 messages🔥): 

> - `GPT-4o mini 性能`
> - `OpenAI 安全问题`
> - `模型评估`
> - `图像输入成本`
> - `企业市场主导地位` 


- **GPT-4o mini 在代码基准测试中与 GPT-3.5 持平**：[GPT-4o mini](https://x.com/paulgauthier/status/1814014867361374610?s=46) 在 Aider 的代码编辑基准测试中得分与原始 GPT-3.5 相似，尽管它在使用 diffs 编辑代码时表现挣扎，且仅限于较小的文件。
- **OpenAI 的新安全机制被轻易绕过**：[OpenAI 的新安全机制](https://x.com/elder_plinius/status/1814023961535295918?s=46) 已被 Jailbreak，**GPT-4o-mini** 输出了恶意软件和非法活动配方等有害内容，暴露出重大的安全缺陷。
- **GPT-4o mini 在内部评估中出现过拟合**：发现 [GPT-4o mini](https://fxtwitter.com/corbtt/status/1814056457626862035?s=61) 在内部 LLM-as-judge 评估中由于包含无关信息而优于 GPT-4o，可能对长度偏差（length bias）等常见评估缺陷产生了过拟合。
- **OpenAI 凭借 GPT-4o mini 发起反击**：[GPT-4o mini](https://x.com/crwhite_ml/status/1814028565161169090) 因其极高的性价比而受到赞誉，正如 [livebench.ai](http://livebench.ai) 所示，对市场产生了重大影响。
   - 根据社区讨论，该模型的经济性可能对主导企业市场至关重要。
- **GPT-4o mini 图像输入定价相同**：[GPT-4o mini](https://x.com/simonw/status/1814163501880893794?s=46) 的文本输入比 GPT-4o 便宜 33 倍，但由于每张图像消耗更多 Token，其图像输入价格保持不变。
   - 根据 **Romain Huet** 的说法，对于追求成本效益的图像输入，Claude 3 Haiku 和 Gemini 1.5 Flash 等替代方案可能更可行。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/crwhite_ml/status/1814028565161169090">来自 Colin White (@crwhite_ml) 的推文</a>: OpenAI 发起反击 💫 GPT-4o-mini 是一款性价比极高的模型！快去 http://livebench.ai 查看它的表现！</li><li><a href="https://x.com/paulgauthier/status/1814014867361374610?s=46">来自 Paul Gauthier (@paulgauthier) 的推文</a>: GPT 4o mini 在 aider 的代码编辑基准测试中得分与原始 GPT 3.5 相当（后期的 3.5 版本表现更差）。初步看来它似乎无法使用 diffs 编辑代码，这限制了它的用途...</li><li><a href="https://x.com/elder_plinius/status/1814023961535295918?s=46">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>: ⚡️ JAILBREAK 警报 ⚡️ OPENAI: 被攻破 ✌️😎 GPT-4O-MINI: 被解放 🤗 看来新的“指令层级（instruction hierarchy）”防御机制还不够 🤷‍♂️ 见证全新的 gpt-4o-mini ...</li><li><a href="https://fxtwitter.com/corbtt/status/1814056457626862035?s=61">来自 Kyle Corbitt (@corbtt) 的推文</a>: 在我们的内部 LLM-as-judge 评估中，gpt-4o mini 绝对碾压了 gpt-4o。于是我查看了数据（感谢 @HamelHusain），发现它在回答问题的同时还塞进了一堆...</li><li><a href="https://x.com/simonw/status/1814163501880893794?s=46">来自 Simon Willison (@simonw) 的推文</a>: GPT-4o mini 一个略显意外的方面：虽然它的文本输入比 GPT-4o 便宜 33 倍，但图像输入的价格却是一样的。如果你需要廉价的图像输入，你可能最好选择...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1263650433604259914)** (7 messages): 

> - `Gemma 2 论文`
> - `Soft logit capping`
> - `Gemma 2 29B 与 LLaMA 3 70B 的竞争力` 


- **在 Gemma 2 中移除 Soft Logit Capping**：成员们讨论了在 **Gemma 2** 模型中移除 Soft logit capping 特性的问题，询问是否需要额外的训练来修复关闭此封顶机制带来的“伤痕”。
   - 一位成员认为，在没有任何显著重新训练的情况下直接禁用 Logit 封顶竟然没问题，这令人难以置信且惊讶。
- **Gemma 2 29B 的竞争力**：一位成员询问为什么 **Gemma 2 29B** 模型与 **LLaMA 3 70B** 相比极具竞争力，尽管它不像 9B 和 2.6B 版本那样受益于蒸馏（distillation）。
   - 另一位成员将其归功于 Google 在 Softmax 和蒸馏或其他技术方面的“魔法”，最后的评论指出**更好的数据**是原因之一。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1263596951899865252)** (1 条消息): 

> - `AGI mission` (AGI 使命)
> - `current business as a sideline` (当前业务作为副业)


- **AGI 使命带来挑战**：当前的业务努力被视为实现 **AGI** 这一主要使命的副业。
   - 这种观点强调了在将业务目标与实现 AGI 的**核心目标**进行协调时可能存在的困难。
- **业务努力被视为副业**：在 AGI **主要使命**的背景下，*当前的业务某种程度上只是副业*。
   - 尽管关注 AGI，但业务方面虽然被承认，却并非首要任务。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1263730787874770944)** (54 条消息🔥): 

> - `Zotero 7 Update` (Zotero 7 更新)
> - `Hugo and Docker` (Hugo 与 Docker)
> - `Reading Lists and Websites` (阅读列表与网站)
> - `Potential Future Interviews` (潜在的未来访谈)
> - `MosaicML Sword Tradition` (MosaicML 宝剑传统)


- **Zotero 7 带来速度与风格**：[Zotero 7](https://www.zotero.org/support/beta_builds) 是对当前版本的更新，提供了更快的速度、暗黑模式以及更好的插件兼容性。
   - 成员们讨论了为了 Better BibTex 等插件以及使用 'Actions and Tags for Zotero' 进行自动打标签而进行的升级，并希望增加工具提示引用功能。
- **Docker 挫折延迟了 Hugo 搭建**：一位成员分享了由于 Docker 中的网络问题，在使用 Hugo 搭建个人网站时遇到的困难。
   - 尽管遇到了挫折，但大家仍鼓励尽快重新审视该项目并使其上线。
- **托管阅读列表引起关注**：讨论了研究人员在静态网站上托管阅读列表这一有趣且实用的想法。
   - 像 **answer.ai** 这样的项目已经分享了他们的 [Zotero 库](https://www.zotero.org/groups/5004697/llms_ai_answers)，激发了对类似倡议的热情。
- **未来与 AI 领袖的访谈**：讨论了联系 Andrej Karpathy 和 Andrew Trask 等知名人士进行访谈的计划。
   - 像 Jonathan Frankle 及其 MosaicML 宝剑传统这样有趣的个性化话题也被提及作为潜在的访谈内容。
- **MosaicML 的宝剑与 HR 的冲突**：作为一种奇特传统，授予 MosaicML 员工的宝剑遭到了 HR 的反对。
   - 传闻甚至 Databricks 的法律团队也可能因为他们的努力而收到了宝剑，这凸显了这一独特的传统。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/testingcatalog/status/1814358444704498010?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 这是谁？ 👀👀👀 2.0, Eureka, Ultra, mini, nano? Flash? Pro? 1.75? Plus? Advanced? Pro New? Pro New New? Pro Plus Ultra? Flame? 🔥</li><li><a href="https://x.com/soldni/status/1695087021520457939">来自 Luca Soldaini 🎀 (@soldni) 的推文</a>: 祝福 @DippedRusk，他在我的自然栖息地（办公室书桌）拍到了我和我的 @MosaicML 宝剑</li><li><a href="https://www.zotero.org/support/beta_builds>">
	beta_builds [Zotero 文档]
</a>: 未找到描述</li><li><a href="https://www.zotero.org/groups/5004697/llms_ai_answers">Zotero | Groups > LLMs AI Answers</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1263647502696185867)** (2 条消息): 

> - `Sara Hooker's critique on US AI Act` (Sara Hooker 对美国 AI 法案的批评)
> - `Cohere for AI`
> - `Compute thresholds in AI` (AI 中的计算阈值)


- **Sara Hooker 谈美国 AI 法案中误导性的计算阈值**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=dBZp47999Ko)，其中 **Sara Hooker** 批评了美国 AI 法案中计算阈值的使用。
   - 另一位参与者指出，由 **Cohere** 研究副总裁 Sara Hooker 撰写的配套论文非常出色，并引起了社区的兴趣。
- **Sara Hooker 在社区讨论中受到称赞**：一位人士表达了对 **Sara Hooker** 的钦佩，强调了她的魅力以及对 AI 研究的贡献。



**提及的链接**：<a href="https://www.youtube.com/watch?v=dBZp47999Ko">为什么美国 AI 法案的计算阈值是误导性的...</a>：Sara Hooker 是 Cohere 的研究副总裁，也是 Cohere for AI 的负责人。我们讨论了她最近批评使用以 FLOPs 衡量的计算阈值的论文...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1263866486104264744)** (24 条消息🔥): 

> - `Z-Loss`
> - `Regularization`
> - `Logits`
> - `Softmax`
> - `Paper Ideas` 


- **关于 Z-Loss 功能的辩论**：一位成员建议将“探索并剖析 Z-loss 为何有效”添加到论文想法列表中。
   - 成员们讨论了 **Z-loss** 作为目标函数 Regularization 项的复杂性，将其与 Weight Decay 进行了比较，并质疑了这种 Regularization 的必要性和深度。
- **[Carsonpoole 澄清 Z-Loss](https://link.address)**：**Carsonpoole** 澄清说，Z-loss 在目标函数的激活值中引入了 Regularization 参数，类似于针对激活值的 Weight Decay。
   - 他强调过大的激活值会导致不稳定性，因此 Z-loss 旨在防止产生不必要的大激活值。
- **替代 Regularization 技术**：**Nshepperd** 提出了一种替代 Regularization 方法，使用 `logits.mean(-1).square().mean()` 而不是 `logits.logsumexp(-1).square().mean()`。
   - 这些方法之间的有效性差异以及 Weight Decay 等 Regularization 基础原理引发了好奇和辩论。
- **理解 Z-Loss 的必要性**：**The_deleted_account** 认为 Softmax 的平移不变性（shift invariance）使得 Z-loss 成为必要。
   - 提到的动机包括防止 bfloat16 中的舍入误差，并鼓励 Logits 成为归一化的对数概率（log-probabilities）。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1263598116649566299)** (13 条消息🔥): 

> - `Cognitive Architectures for Language Agents (CoALA)`
> - `Discussion on Bits Per Byte (BPB) vs Per Token`
> - `Mixing Sequences for Training`
> - `Transformer Training Instability Checklist`
> - `Experience-driven AI Evaluations` 


- **CoALA 框架组织语言 Agent**：一篇新论文提出了 [语言 Agent 的认知架构 (CoALA)](https://arxiv.org/abs/2309.02427)，该架构描述了一个具有模块化内存组件和结构化动作空间进行决策的语言 Agent，旨在组织和规划语言模型的未来发展。
   - 该论文利用 CoALA 调查并组织了最近的工作，并从认知科学和符号 AI 中汲取灵感，确定了可行的发展方向。
- **BPB 与 Per Token 指标解释**：对给定指标是“每字节比特数 (BPB)”还是“每 Token”进行了澄清，强调其为“每 Token”。
   - *cz_spoon_06890* 澄清说，所讨论的指标非常重要，因为它会显著影响结果的解读。
- **混合序列用于模型训练**：一位成员提议在训练期间对多个序列进行平均，类似于 CNN 中的 Mixup，这可能消除对微调（Fine-tuning）的需求。
   - 另一位成员提到可能需要对 Mixup 率进行退火（Annealing），引发了关于相比两阶段训练更简洁解决方案的讨论。
- **Transformer 训练不稳定性资源**：有人询问关于 Transformer 训练不稳定性的检查清单，并提供了相关资源的链接。
   - 分享了 [检查清单链接](https://discord.com/channels/729741769192767510/1079865324087803985/1258858457814138880) 以帮助解决训练不稳定性问题。
- **评估经验驱动的 AI**：一位成员就其关于经验驱动 AI 评估的论文寻求反馈，特别是关于评估周期的特征描述。
   - 征求反馈以确保该特征描述的准确性和相关性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.13623">Scaling Laws with Vocabulary: Larger Models Deserve Larger Vocabularies</a>：关于缩放大型语言模型 (LLMs) 的研究主要集中在模型参数和训练数据大小上，忽略了词表大小的作用。直观地说，更大的词表能够实现更……</li><li><a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>：最近的努力通过外部资源（例如互联网）或内部控制流（例如 Prompt Chaining）增强了大型语言模型 (LLMs)，以处理需要落地或推理的任务……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1263950155401662464)** (1 messages): 

> - `Hypernetworks and Scaling Laws`
> - `Scaling Law Predictions`
> - `Compute and Target Error`
> - `Conditional Hypernetworks`
> - `Neural Network Prediction` 


- **Scaling laws 限制了 hypernetwork 的能力**：讨论了 **scaling laws** 如何为 **hypernetworks** 的能力设定边界，并质疑较小规模的 hypernetwork 是否能达到 scaling law 预测的目标误差。
   - 有人指出，hypernetwork 的有效性可能需要“改进” scaling law，或者专注于输出模型的 scaling law 表现良好的任务，例如表示单个数据点。
- **比较 hypernetwork 和输出模型的 scaling**：探讨了 hypernetwork 及其输出模型的架构是否遵循相同的 scaling law，以及预测一个解决任务的神经网络是否比直接解决任务更容易。
   - 建议认为，只有当目标任务具有“良好”的 scaling law 时，hypernetworks 才可能有用，这需要从相关数据子集中学习所需的计算量或数据量显著减少。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263598332773797889)** (8 messages🔥): 

> - `Tokenization-free language models`
> - `Interpretability of ResNet in Vision Models`
> - `MATS 7.0 Streams by Neel Nanda and Arthur Conmy` 


- **Tokenization-free 模型引发可解释性辩论**：成员们辩论了 **tokenization-free 模型**（无论是字符级还是字节级）对可解释性是利是弊，并对缺乏规范的处理位置表示担忧。
   - *“Utf-8 也是一种 tokenization 方案，只是个糟糕的方案，”* 一位成员指出，并对字节级 tokenization 成为默认方案持怀疑态度。
- **深入研究 ResNet residual stream**：一位成员分享了他们的 [新文章](https://arxiv.org/abs/2407.05340)，调查了 **ResNet** 中的 residual stream，以进行视觉模型的机械可解释性（mechanistic interpretability）研究。
   - 在寻求建设性反馈时，他们将自己的方法描述为由于新手身份而进行的简单探索，并寻找相关论文或轻量级模型以进行进一步研究。
- **冬季 MATS 7.0 方向开放申请**：由 **Neel Nanda 和 Arthur Conmy** 领导的 **冬季 MATS 7.0 方向** 已开放申请，截止日期为 8 月 30 日。
   - [在此申请](https://docs.google.com/document/?usp=docs_web)，有机会在尖端的机械可解释性研究中获得指导。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/NeelNanda5/status/1813921161052635209">来自 Neel Nanda @ ICML (@NeelNanda5) 的推文</a>：你对 @ch402 风格的机械可解释性研究感到兴奋吗？我正寻求通过 MATS 指导学者 - 请在 8 月 30 日前申请！我对过去学者的工作印象深刻，并且热爱指导...</li><li><a href="https://tinyurl.com/neel-mats-app">Neel Nanda / Arthur Conmy MATS 7.0 方向 - 录取程序 + 常见问题解答</a>：Neel Nanda / Arthur Conmy MATS 方向 - 录取程序 + 常见问题解答。如何申请：填写常规 MATS 申请表（少于 10 分钟）。截止日期为太平洋时间 8 月 30 日星期五晚上 11:59。请注意，这是一个特殊的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263576951264907415)** (22 messages🔥): 

> - `System prompt concatenation`
> - `LM eval model correctness`
> - `HF datasets trust remote code`
> - `Zeno upload feature`
> - `Editable installation issues` 


- **System prompt 拼接逻辑**：澄清了当存在 description 字段时，对于允许其 chat template 接收该字段的模型，它会被拼接到 system prompt 的末尾。
- **确保 LM eval 模型正确性**：一位用户指出新实现的 LM eval 模型与其他 HF 模型之间的分数差异，寻求检查实现正确性并消除变量的代理指标。
- **解决远程代码信任问题**：一位成员分享了一个技巧，即使用 `export HF_DATASETS_TRUST_REMOTE_CODE=1` 在加载基准测试数据集时信任远程代码。
- **重构后的 Zeno 上传功能**：用户报告了在大规模重构后，Zeno 上传功能 `visualize_zeno.py` 遇到的挑战。
- **可编辑安装与日志问题**：一位用户遇到了尽管使用了可编辑安装（editable install），但 `eval_logger.info` 无法打印的问题。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1263596451867660400)** (5 条消息): 

> - `Mistral NeMo 发布`
> - `LlamaCloud 更新`
> - `对检索结果进行重排序 (Re-ranking)`
> - `使用 LLMs 作为评判者 (LLM as a judge)`
> - `社区活动` 


- **MistralAI 和 OpenAI 发布新模型**：今天是新模型发布的重要日子，**MistralAI** 和 **OpenAI** 均有发布，且这两个模型已经获得了 [零日支持 (day zero support)](https://twitter.com/llama_index/status/1814036536192811184)。
   - **Mistral NeMo** 是一款小型 (12B) 模型，性能超越了 **Mistral 的 7b** 模型，并拥有显著的 (128k) 上下文窗口。
- **LlamaCloud 推出新功能**：**LlamaCloud** 的最新更新包括 **LlamaCloud Chat**（一个数据的对话式界面）以及用于协作的新团队功能。
   - 这些更改旨在提升用户体验和生产力。[点击此处阅读更多](https://twitter.com/llama_index/status/1814363518726222119)。
- **通过重排序 (Re-ranking) 提升相关性**：对检索到的结果进行重排序可以显著增强响应的相关性，尤其是在使用像 **@postgresml** 这样的托管索引时。
   - 查看他们在 LlamaIndex 博客上的 [客座文章](https://t.co/HWfitT0CJt) 以获取更多见解。[更多详情请点击此处](https://twitter/llama_index/status/1814386548340826449)。
- **McDermott 的 RAG 评估专题**：**Yixin Hu (VU Amsterdam)** 和 **Thomas Hulard (McDermott)** 分享了一个关于使用 **LLMs as a judge** 将应用投入生产的环节。
   - 该录像深入探讨了 RAG 评估背后的核心概念。[在此观看](https://twitter.com/llama_index/status/1814409012328517701)。
- **报名参加即将举行的活动**：提醒：仍有时间报名参加一小时后开始的活动。[在此加入](https://t.co/BxdWQect1S)。
   - 请继续关注更多社区活动和更新。[了解更多](https://twitter.com/llama_index/status/1814318805906305161)。



**提到的链接**：<a href="https://t.co/HWfitT0CJt">Improving Vector Search - Reranking with PostgresML and LlamaIndex — LlamaIndex, Data Framework for LLM Applications</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1263623170926313513)** (41 条消息🔥): 

> - `通过 LlamaIndex 流式传输思考过程 (Streaming thoughts)`
> - `LLM 中的上下文窗口限制`
> - `Pandas 查询引擎的不一致行为`
> - `Text to SQL 查询流水线问题`
> - `Llama-parse API 性能` 


- **关于通过 LlamaIndex 流式传输思考过程的疑问**：一位用户询问 LlamaIndex 是否可以流式传输思考过程，另一位用户提供了一个教程示例，建议通过修改 Prompt 可能实现。
   - 参考了 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/query_engine/sub_question_query_engine/)上的教程以了解详细实现。
- **LLM 上下文窗口限制引起困惑**：一位成员在为 GPT-4o mini 设置 `max_tokens` 限制时遇到了 'Error code: 400'，尽管 OpenAI 文档声明其上下文窗口为 128K tokens，但据报道它仅支持 16384 个生成 tokens (completion tokens)。
   - 这种困惑源于在代码的不同部分使用了不同的模型，导致 SQL 查询引擎中 GPT-3.5 和 GPT-4 之间产生了干扰。
- **Pandas 查询引擎表现出不一致的行为**：一位用户报告称，Pandas 查询引擎在 Jupyter Notebook 中能正确解析列名，但在作为 .py 文件或 API 运行时失败，导致 KeyError。
   - 代码和文档在不同环境下保持不变，这表明在 Jupyter Notebook 之外存在自然语言列映射的问题。
- **使用 CTEs 改进 Text to SQL 流水线**：一位用户遇到了多个 SQL 查询无法正确执行的问题，因为系统假设了结果而不是运行后续查询，通过使用公用表表达式 (CTEs) 解决了该问题。
   - 根据教程，提示系统使用 CTEs 提供了一个解决方案，使后续查询能够成功执行。
- **ReActAgent 卡在最大迭代值上**：提高 ReActAgent 的 `max_iterations` 值并未解决 Agent 似乎卡住且无法返回任何响应的问题。
   - 尽管修改了迭代参数，问题仍然存在，这促使该用户请求社区提供进一步的故障排除协助。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/run-llama/llama_parse">GitHub - run-llama/llama_parse: Parse files for optimal RAG</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/6a8e151f9b912d8fad5fa4d09bd2f7bfcb393f0c/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/utils.py#L50">llama_index/llama-index-integrations/llms/llama-index-llms-openai/llama_index/llms/openai/utils.py at 6a8e151f9b912d8fad5fa4d09bd2f7bfcb393f0c · run-llama/llama_index</a>: LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1263572925697495112)** (15 条消息🔥): 

> - `Query rewriting` (查询重写)
> - `Multimodal RAG` (多模态 RAG)
> - `Splitting documents in LlamaIndex` (LlamaIndex 中的文档切分)
> - `Use of LlamaIndex versus LangChain` (LlamaIndex 与 LangChain 的使用对比)
> - `ETL of unstructured data` (非结构化数据的 ETL)


- **查询重写工具 (Query Rewriting Utility)**：一位成员发现 LlamaIndex 在处理包含大量数学公式、图表和图像的演示文件时表现出色，并提出了关于利用 Query Rewriting 增强性能的效用问题。他们热衷于在 LlamaIndex 框架内探索更多用例。
- **使用 LlamaIndex 切分文档**：讨论透露，当使用 SentenceSplitter 时，LlamaIndex 会在句子边界附近自动切分文档，默认 **chunk size** 为 **1024**，**overlap size** 为 **128**。
   - LlamaIndex 中的 PDF 加载器按页切分文档，这与实际页码完美对应，正如代码作者所确认的，这使得引用更加容易。
- **非结构化数据的 ETL**：一位成员询问如何将视频和音乐等非结构化数据解析为 LLM 可理解的格式，并引用了 Jerry Liu 与 Alejandro 之间的一次 YouTube [对话](https://www.youtube.com/watch?v=imlQ1icxpBU)，其中提到了一种新型的 ETL。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://kaushikshakkari.medium.com/choosing-the-best-structured-output-parser-approach-3-ways-to-generate-structured-output-d9686482729c">Choosing the Best Structured Output Parser Approach | 3 Ways To Generate Structured Output</a>：结构化输出提取方法的详细比较</li><li><a href="https://www.youtube.com/watch?v=imlQ1icxpBU">Jerry Liu - What is LlamaIndex, Agents &amp; Advice for AI Engineers</a>：在本集中，我们采访了 LlamaIndex 的富有远见的创始人 Jerry Liu，这是一个专为 LLM 开发而设计的尖端 Python 框架...</li><li><a href="https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/claude_parse.ipynb">llama_parse/examples/multimodal/claude_parse.ipynb at main · run-llama/llama_parse</a>：为优化 RAG 解析文件。通过在 GitHub 上创建账号为 run-llama/llama_parse 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1263601449280082111)** (40 messages🔥): 

> - `Mistral-12b`
> - `Training Inferences in Transformers`
> - `Config Issues and Fixes`
> - `Triplex Model for Knowledge Graphs` 


- **训练推理提升 Transformer 泛化能力**：一篇 [arXiv 论文](https://arxiv.org/abs/2405.15071)指出，将 Transformer 训练到远超饱和与记忆阶段，可以提高其泛化和推导推理事实的能力。
   - 论文研究结果显示，Transformer 在处理域外（out-of-domain）推理时表现不佳，是因为它们缺乏在两个不同位置存储相同事实的动力。
- **Mistral-12b 的配置问题**：一名成员报告在 Mistral-12b 模型中遇到了配置问题，特别是各种投影权重（projection weights）的大小不匹配。
   - 解决这些问题需要从源码安装 transformers，而训练过程在 8x L40s 上运行良好，并显示出令人期待的 loss 降低结果。
- **需要 Tokenizer 填充 Token**：成员们因 Tokenizer 缺少填充 token（padding token）而遇到错误，建议将 Tokenizer 的 pad token 设置为 `tokenizer.eos_token` 或添加一个新的 pad token。
   - 这些错误影响了包括补全（completions）和训练过程在内的各种场景，需要对配置进行特定调整。
- **用于知识图谱构建的 Triplex 模型**：Triplex 模型是 Phi3-3.8B 的一个版本，为创建知识图谱提供了一种经济高效的解决方案，[与 GPT-4 相比成本降低了 98%](https://huggingface.co/SciPhi/Triplex)。
   - 该模型可共享并在本地系统上执行，允许通过 Neo4j 和 R2R 轻松设置，从而增强下游的 RAG 方法。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>：我们研究了 Transformer 是否能够学习对参数化知识进行隐式推理，这是即使是最强大的语言模型也难以掌握的技能。重点关注两种代表性的推理类型...</li><li><a href="https://x.com/danielhanchen/status/1814317286389666094?s=46">Daniel Han (@danielhanchen) 的推文</a>：我对 Mistral NeMo 12b 的发现：1. &lt;/s&gt; EOS token 在 base 模型中未经过训练 - 是个 bug 吗？2. EOS token 会自动追加 3. Wq 是 4096 而不是 5120 4. 不是 Llama Tokenizer 5. Tools, FIM 6. Pad_token=10&lt;p...</li><li><a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>：未找到描述</li><li><a href="https://r2r-docs.sciphi.ai/cookbooks/knowledge-graph">Knowledge Graphs - 最好的开源 AI 驱动问答引擎。</a>：未找到描述</li><li><a href="https://github.com/vllm-project/vllm/pull/6548">[Model] Support Mistral-Nemo by mgoin · Pull Request #6548 · vllm-project/vllm</a>：修复 #6545。补丁从 huggingface/transformers#32050 移植而来。本质上是 MistralConfig 中添加了一个新的 head_dim 覆盖。我们将在配置中查找该可选参数并默认...
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1263619761380790363)** (2 messages): 

> - `Mistral-Nemo`
> - `Technical queries in axolotl-dev channel` 


- **Mistral-Nemo 状态查询**：一名成员询问了 **Mistral-Nemo** 目前的工作状态。
- **常规技术咨询**：成员们经常使用 axolotl-dev 频道提问和回答技术问题，例如特定工具（如 **Mistral-Nemo**）的当前状态。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1263583242402136216)** (5 messages): 

> - `Llama3`
> - `Eval Loss`
> - `Training Loss` 


- **Llama3 rank 调整改善了 eval loss**：根据一名成员的观察，**降低 Llama3 的 rank** 显著改善了 eval loss。
   - *仍需稍后运行评估集*以确认这种改进是否能持续。
- **关于 eval loss 差异的讨论**：另一名成员评论说存在显著差异，并推测这可能会在后面的步骤中趋于平稳。
   - 原成员提到他们今晚将继续运行测试以观察结果。
- **训练损失也更低**：同一名成员注意到，**training loss** 似乎也随 eval loss 一起显著降低了。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-phorm-bot](https://discord.com/channels/1104757954588196865/1225558824501510164/1264005864730267728)** (5 messages): 

> - `axolotl 训练中的 GPU 显存错误`
> - `axolotl 中的常见错误`
> - `训练配置调整` 


- **解决 axolotl 训练中的 GPU 显存问题**：用户讨论了在 axolotl 训练期间运行内存不足（OOM）的常见错误，这通常是由于模型过大或 Batch Size 超过了 GPU 容量导致的。
   - 分享了一份详细指南，介绍如何通过调整 `micro_batch_size`、`gradient_accumulation_steps` 等设置，并启用 `fp16` 来优化显存使用，从而缓解此问题。
- **axolotl 中的常见错误及其修复**：社区强调了在 axolotl 训练中遇到的几个常见错误并提供了解决方案，包括调整序列长度（sequence length）以及使用特定的优化器（如 `adamw_bnb_8bit`）。
   - 分享了 [Common Errors 🧰](https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L562L614) 的链接，以获取更多故障排除信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/openaccess-ai-collective/axolotl/tree/main/README.md#L562L614)">axolotl/README.md at main · axolotl-ai-cloud/axolotl</a>: 欢迎提出 axolotl 相关问题。通过在 GitHub 上创建账号来为 axolotl-ai-cloud/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=15dfc26f-b460-49e5-ae58-0ffd7039cc47)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1263618480687546601)** (25 messages🔥): 

> - `GPTs Agents`
> - `Web 搜索能力`
> - `LLM 自我意识`
> - `Cohere Toolkit`
> - `角色图标` 


- **测试 GPTs Agents 的自我意识**：一位用户进行了实验，旨在不使用 Web 搜索能力的情况下确定 GPTs Agents 的自我意识。
- **Web 搜索性能令用户印象深刻**：一位用户观察到 Bot 的响应速度极快，感觉就像从未执行过 Web 搜索一样，突显了系统的效率。
- **Cohere Toolkit 的灵活性受到赞赏**：一位用户分享了 [Aidan Gomez 和 Nick Frosst 的推文](https://x.com/aidangomez/status/1814308463104668113)，强调了 Cohere Toolkit UI 的开源特性，允许用户接入任何模型并贡献新功能。
- **讨论了角色图标和工作安排**：一位用户幽默地提到他们的角色图标是一个安全帽。



**提到的链接**: <a href="https://x.com/aidangomez/status/1814308463104668113">Aidan Gomez (@aidangomez) 的推文</a>: 提醒大家，整个 Toolkit UI 都是开源且即插即用的。所以可以随意接入你想要的任何模型并贡献新功能！引用 Nick Frosst (@nickfrosst) 几周前...

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1263579431167987815)** (15 messages🔥): 

> - `Firecrawl 定价`
> - `Firecrawl 自托管`
> - `GPT-4o 集成`
> - `本地 LLM 聊天 GUI` 


- **Firecrawl 在没有大批量客户的情况下过于昂贵**：一位成员提到 **Firecrawl** 只有在拥有庞大客户群时才具有成本效益，并建议推出按需付费计划。
- **Firecrawl 后端自托管可节省成本**：成员们讨论了自托管 Firecrawl 后端，通过仅设置 API 端点使服务更实惠。
   - 一位成员分享了一个 [GitHub 链接](https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md) 及自托管指南，表示这为他们节省了几百美元。
- **Firecrawl 与 GPT-4o 的集成**：Firecrawl 自托管允许通过使用存储在 `.env` 文件中的个人 API Key 来实现 **GPT-4o 集成**。
- **使用本地 LLM 的新聊天 GUI 项目**：一位成员分享了他们正在进行的项目，该项目具有由本地 LLM 驱动的聊天 GUI，并实现了 **Web Search、Python 解释器和图像识别**。
   - 为感兴趣的人提供了该项目的 [GitHub 仓库](https://github.com/yamikumo-DSD/chat_cmr)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jsfiddle.net/razodactyl/gqr5vaot/1/">Edit fiddle - JSFiddle - Code Playground</a>: 无描述</li><li><a href="https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md">firecrawl/SELF_HOST.md at main · mendableai/firecrawl</a>: 🔥 将整个网站转换为 LLM 就绪的 Markdown 或结构化数据。通过单个 API 进行抓取、爬取和提取。 - mendableai/firecrawl
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1263779359903187087)** (3 条消息): 

> - `有用的解决方案`
> - `Instruct/Chat 数据集 RFC` 


- **用户发现解决方案很有帮助**：一位用户表示分享的解决方案非常有用，并提到它对他们有效。
- **关于 Instruct/Chat 数据集整合的 RFC**：一名成员在 dev 频道分享了一个 [Request for Comments (RFC)](https://link.to.rfc)，内容涉及整合 Instruct/Chat 数据集类，旨在简化在 **Hugging Face** 上添加自定义数据集的过程。
   - 他们鼓励那些经常使用自定义数据运行 fine-tuning 任务的人员查看该 RFC 并提供反馈，并表示这不会影响高级 API。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1263594118924472340)** (32 条消息🔥): 

> - `LLM 训练测试`
> - `Torchtune Recipe 文档`
> - `统一数据集抽象`
> - `Recipe 中的错误处理` 


- **测试 LLM：强制回复 HAHAHA**：成员们讨论了尝试训练 LLM 对每个输入都回复“HAHAHA”的实验。尽管调整了设置，[LLM 并没有像预期那样学习](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_instruct.py#L99)。
- **Torchtune Recipes 的可见性**：有一场关于提高 Torchtune 可用 Recipes 的可见性和文档完善度的对话。
   - [从 recipe docstrings 自动生成文档](https://github.com/pytorch/torchtune/pull/256)被提议为向前迈出的有用一步。
- **统一数据集抽象 RFC**：讨论了一个新的 RFC，旨在统一 instruct 和 chat 数据集以支持多模态数据。
   - 关键反馈包括可用性改进，例如将 tokenizer 和 prompt templating 与其他数据集配置分离，[正如 RFC 中详述的那样](https://github.com/pytorch/torchtune/pull/1186)。
- **精简错误处理**：建议通过将通用的验证函数从单个 Recipe 中移出，来精简 Recipe 中的错误处理。
   - 这将有助于用户关注关键配置并减少样板代码（boilerplate code）。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_instruct.py#L99">torchtune/torchtune/datasets/_instruct.py at main · pytorch/torchtune</a>：一个用于 LLM fine-tuning 的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1186">[RFC] Unified dataset with data and model transforms by RdoubleA · Pull Request #1186 · pytorch/torchtune</a>：感谢 @pbontrager 进行的所有讨论，帮助收敛到此设计。TLDR：让我们创建一个通用的 fine-tuning 数据集类，它接收一个数据转换类和一个模型转换类...</li><li><a href="https://github.com/pytorch/torchtune/pull/256">Added Recipes to docs by pbontrager · Pull Request #256 · pytorch/torchtune</a>：上下文：关于我们的 Recipes 如何在文档中呈现的第一个示例。这是基于早期关于文档的讨论。这允许用户获取与相同版本匹配的 Recipes...
</li>
</ul>

</div>
  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1263668415579815987)** (28 条消息🔥): 

> - `Mozilla Builders 初创企业加速器`
> - `为盲人生成的 AI 场景描述`
> - `用于养蜂业的智能 AI 设备`
> - `群体机器人 (Swarm Robotics) 与比特币挖矿` 


- **Mozilla Builders 启动加速器**：一位成员提到 Mozilla Builders 宣布了[一个针对硬件和边缘 AI 的初创企业加速器](https://builders.mozilla.org)。
   - 另一位成员表达了兴趣并分享了他们的持续投入，表示：“我不会离开，这不是一个兼职加速器，我们就住在这里。”
- **AI 为盲人生成场景描述**：有一场关于创建 AI 为盲人生成场景描述的讨论。
   - “失明和所有疾病都需要被消除。”是一位成员分享的尖锐观点。
- **构建用于蜜蜂的智能 AI 设备**：已经为养蜂业和开源比特币挖矿硬件构建了**智能 AI 数据驱动设备**。
   - 主要兴趣在于结合 **AI 和养蜂业**，在蜜蜂面临危险之前为养蜂人提供预警。
- **群体机器人与 AI 项目**：成员表达了对蜜蜂的着迷，并提到对**群体机器人 (swarm robotics)** 的潜在兴趣。
   - 还提到了一个专注于艺术的 AI 项目，该项目通过 Whisper 进行监听，并在对话过程中生成基于上下文的图像。


  

---

### **Alignment Lab AI ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/1263609158691983380)** (1 messages): 

> - `RWKV hybrid model paper` (RWKV 混合模型论文)
> - `GoldFinch model details` (GoldFinch 模型详情)
> - `Transformer enhancements` (Transformer 增强)
> - `Model performance comparisons` (模型性能对比)


- **GoldFinch 携混合模型增益问世**：**GoldFinch** 结合了来自 RWKV 的 Linear Attention 和传统的 Transformers，在下游任务上超越了稍大的 **1.5B 级 Llama** 和 **Finch (RWKV-6)** 模型。这一改进归功于消除了二次方减速并显著减小了 KV-Cache 大小，从而能够以极低的 VRAM 需求支持超长上下文。
   - 潜在应用包括在消费级显卡上分析整个代码库或法律文件，通过降低二次方注意力成本实现大幅成本节约。
- **GPTAlpha 和 Finch-C2 发布**：此次发布包括 **Finch-C2**（Finch 的高性能下游版本）以及 **GPTAlpha**（一种增强的全 Transformer 架构，采用了 RWKV 组件并使用 softmax attention）。这些模型优于传统的 Transformers，提供了卓越的性能和效率。
   - 新模型提供了以更低成本回看每个 token 的能力，并兼具更好的下游性能。
- **GoldFinch 模型论文和代码已发布**：GoldFinch 论文已在 [arXiv](https://arxiv.org/abs/2407.12077) 上线，详细介绍了混合模型架构和性能增强。 [GitHub 仓库](https://github.com/recursal/GoldFinch-paper) 包含了各种消融实验和 1.5B 规模模型的代码及权重。
   - GoldFinch 项目的产物和权重也托管在 [Huggingface](https://huggingface.co/recursal/GoldFinch-paper) 上，包括小规模和大规模模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12077">GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression</a>：我们介绍了 GoldFinch，这是一种混合 Linear Attention/Transformer 序列模型，它使用一种新技术，在线性时间和空间内高效生成高度压缩且可重复使用的 KV-Cache...</li><li><a href="https://github.com/recursal/GoldFinch-paper">GitHub - recursal/GoldFinch-paper: GoldFinch and other hybrid transformer components</a>：GoldFinch 及其他混合 Transformer 组件。通过在 GitHub 上创建账户为 recursal/GoldFinch-paper 的开发做出贡献。</li><li><a href="https://huggingface.co/recursal/GoldFinch-paper">recursal/GoldFinch-paper · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1263679424704942182)** (8 messages🔥): 

> - `Kernel refactoring suggestion` (Kernel 重构建议)
> - `get_lazyop_info removal` (移除 get_lazyop_info)
> - `tinygrad internals` (tinygrad 内部机制)
> - `View.mask purpose` (View.mask 的用途)
> - `Project proposal: trace OpenPilot model` (项目提议：追踪 OpenPilot 模型)


- **Kernel 重构建议**：George Hotz 建议重构 Kernel，取消 `linearize` 函数，改为仅保留 `to_program` 函数。
   - 他补充说，应首先移除 `get_lazyop_info` 以促进这一更改。
- **探索 tinygrad 内部机制**：一名成员正在尝试学习 tinygrad 的内部原理，并询问了 `View.mask` 的用途。
   - George Hotz 确认它主要用于 padding（填充），另一名成员分享了一个 [参考链接](https://discord.com/channels/1068976834382925865/1070745817025106080/1255977369727140013) 来支持该解释。
- **新项目提议：分析 OpenPilot 模型追踪**：George Hotz 提议了一个新项目，利用 OpenPilot 模型追踪来记录 Kernel 更改及其对性能的影响。
   - 他分享了一个 [Gist 链接](https://gist.github.com/geohot/8d7edc7ac2fd9a31ea563c134b66cddb) 并提供了运行分析的说明，强调这类任务任何“稍微聪明点”的人都能胜任。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1263579220781699112)** (16 messages🔥): 

> - `GTX1080 兼容性`
> - `Tinygrad 中的 _pool 函数`
> - `Lazybuffers 中的 Shapetracker` 


- **GTX1080 面临 Tinygrad 的兼容性问题**：[一位成员](https://discord.com) 报告了在 `CUDA=1` 的 GTX1080 上运行 Tinygrad 时出现错误，表明 GPU 架构存在问题。
   - 另一位成员建议 **2080 代 GPU** 是最低要求，并建议在 `ops_cuda` 中修补架构并禁用 tensor cores。
- **关于 `_pool` 函数实现的讨论**：一位成员寻求理解 Tinygrad 中 `_pool` 函数的帮助，特别是质疑池化是否使用 `pad`、`shrink`、`reshape` 和 `permute` 操作来复制数据。
   - 在重新评估代码后，该成员承认该函数并不像最初怀疑的那样会复制数值。
- **Shapetracker 在 Lazybuffers 中的作用引发辩论**：成员们讨论了 Lazybuffers 是否应该使用一系列 view 和 Lazybuffers，而不是使用 Shapetracker 来组织 view 的组合。
   - 虽然一些成员主张使用单个 Shapetracker 进行更好的组织，但他们一致认为这主要影响代码组织，而不会改变功能。


  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1263695247662256261)** (5 messages): 

> - `gpt-4o-mini`
> - `16k token 输出`
> - `Yi large preview`
> - `OI 模型引入` 


- **GPT-4o-mini 参数变更**：一位用户询问是否可以通过直接更改参数来使用 **gpt-4o-mini**，还是需要由 **OI** 引入。
- **16k token 输出令人印象深刻**：一位成员提到了 **16k 最大 token 输出** 这一出色的特性。
- **Yi large preview 表现更佳**：一位成员表示 **Yi large preview** 在 **OI** 中的表现仍然优于其他模型。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1263756978669355008)** (10 messages🔥): 

> - `GPT-4o Mini`
> - `Function Calling`
> - `代码生成` 


- **GPT-4o Mini：速度快但在代码生成方面表现平平**：成员们评论说，根据初步测试，GPT-4o Mini 速度很快，但在 **代码生成方面表现平平**。
   - 然而，配合良好的 custom instructions，它可能适用于特定任务，但在 Function Calling 性能方面尚未展现出卓越之处。
- **OpenAI 声称 GPT-4o Mini 具有强大的 Function Calling 能力**：分享了一个指向 OpenAI [公告](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/) 的链接，该公告指出 GPT-4o Mini 在 **Function Calling** 方面表现强劲，且与 GPT-3.5 Turbo 相比，长上下文性能有所提升。
   - *“我以为它很强？”* 引发了关于实际性能与预期之间差距的简短讨论。


  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1263795517478010944)** (6 messages): 

> - `使用 LAION 模型的 ICML'24 论文`
> - `Text2Control 方法`
> - `大型图像数据集的存储缩减`
> - `在 Hugging Face 上托管 Latents` 


- **ICML'24 论文使用 LAION 模型**：一位用户对 LAION 项目提供的模型表示感谢，这些模型被用于最近的一篇 ICML'24 论文中。
   - 他们鼓励大家尝试 [Text2Control 的交互式演示](https://europe.naverlabs.com/text2control)，并形容其非常有趣。
- **Text2Control 让 Agent 能够通过自然语言执行任务**：[Text2Control](https://europe.naverlabs.com/text2control) 是一种允许 Agent 执行由自然语言指定的新任务的方法，它通过使用 vision-language models 推断目标，并由 goal-conditioned agent 来实现该目标。
   - 该方法在向新任务的 zero-shot 泛化方面优于多任务强化学习基准（multitask reinforcement learning baselines）。
- **对大型图像数据集 Latents 的需求**：一位用户询问是否可以获取大型图像数据集的 latents，特别是 sdxl vae latents 和 conditioner datasets。
   - 他们提到，使用这些 latents 将显著降低其运行的存储成本。
- **在 Hugging Face 上托管 Latents**：有人建议将 latents 上传到 Hugging Face 以避免存储费用。
   - 对方指出 Hugging Face 会承担 S3 存储费用。



**提到的链接**：<a href="https://europe.naverlabs.com/text2control">Bridging environments and language with rendering functions and vision-language models</a>：未找到描述

  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1263626610486018099)** (5 messages): 

> - `AGI 模型性能`
> - `使用 LAION 模型的 ICML'24 论文`
> - `Text2Control 交互式演示` 


- **AGI 声明被过度炒作，但模型表现出色**：一位成员讨论了实现类 AGI 的性能通常被认为是被过度炒作的，但许多模型通过适当的实验已经实现了很高的正确率，并引用了 [@_lewtun 的推文](https://x.com/_lewtun/status/1813197210600829192)。
   - 他们指出：*“这条推文带有自我讽刺意味，因为许多模型都能妥善解决它，但没人愿意进行‘枯燥’的实验来对其进行科学验证”*。
- **ICML'24 论文引用 LAION 模型**：一位研究人员感谢 LAION 项目为其最近的 [ICML'24 论文](https://europe.naverlabs.com/text2control) 提供了所使用的模型。
   - 他们分享了其 **Text2Control** 方法的交互式演示，该方法利用视觉语言模型使 Agent 能够根据文本指令达成目标。
- **Text2Control 演示优于基准模型**：**Text2Control** 方法允许 Agent 通过视觉语言模型从文本中推断目标，从而执行自然语言指定的任务，并且在 zero-shot 泛化方面优于多任务强化学习基准模型。
   - 研究人员邀请其他人尝试交互式演示，强调了他们的方法在启用语言条件 Agent 方面的实际应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/_lewtun/status/1813197210600829192">Lewis Tunstall (@_lewtun) 的推文</a>: 你能感觉到 AGI 吗？</li><li><a href="https://europe.naverlabs.com/text2control">通过渲染函数和视觉语言模型连接环境与语言</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/1263617095690747926)** (2 messages): 

> - `CNN 可视化`
> - `Text2Control 方法` 


- **优秀的 CNN 可视化**：一位成员分享了一个 [CNN 解释器可视化工具](https://poloclub.github.io/cnn-explainer/) 的链接，强调了它的实用性。
   - 该工具旨在通过交互式可视化帮助用户理解卷积神经网络 (CNNs) 的工作原理。
- **Text2Control 方法介绍**：[Naver Labs Europe](https://europe.naverlabs.com/text2control) 展示了他们新的 “Text2Control” 方法，用于利用视觉语言模型根据文本指令控制类人机器人。
   - 该方法通过超越 MTRL 基准模型实现了令人印象深刻的 zero-shot 泛化，并允许用户通过 [交互式演示](https://europe.naverlabs.com/wp-content/plugins/wp-fastest-cache-premium/pro/images/blank.gif) 进行互动。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://poloclub.github.io/cnn-explainer/">CNN Explainer</a>: 一个旨在帮助非专业人士学习卷积神经网络 (CNNs) 的交互式可视化系统。</li><li><a href="https://europe.naverlabs.com/text2control">通过渲染函数和视觉语言模型连接环境与语言</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/)** (1 messages): 

prince.dhankhar: 如何使用 LangChain 向 ChatOllama 的每条聊天消息发送时间戳？
  

---

### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1263663996939014276)** (6 messages): 

> - `针对特定模型的 Prompt 用词`
> - `ChatPromptTemplate 的用法`
> - `在 Prompt 中加入 JSON` 


- **针对特定模型的 Prompt 用词并非必要**：一位成员询问是否需要在 `ChatPromptTemplate` 中使用模型描述中的特定用词以确保准确性。
   - 另一位成员澄清说，LangChain 的 `ChatPromptTemplate` 抽象了这一点，使得像 `<|assistant|>` 这样的特定标记变得不再必要。
- **使用 ChatPromptTemplate 创建 Prompt**：[一个示例](https://app.langchain.com)展示了如何通过定义消息数组来创建 `ChatPromptTemplate`，每条消息由角色（role）和消息文本对表示。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/langchain-ai/langchain/issues/19763>">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/v0.2/docs/concepts/#chatprompttemplates>">概念指南 | 🦜️🔗 Langchain</a>: 本节包含对 LangChain 关键部分的介绍。</li><li><a href="https://js.langchain.com/v0.2/docs/tutorials/llm_chain/#prompt-templates>">使用 LCEL 构建简单的 LLM 应用 | 🦜️🔗 Langchain</a>: 在本快速入门中，我们将向你展示如何构建一个简单的 LLM 应用程序。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1264009384497778749)** (1 messages): 

> - `Triplex LLM`
> - `Knowledge Graphs`
> - `SciPhi.AI`
> - `Graph RAG`
> - `成本降低` 


- **Triplex 彻底改变了知识图谱的构建**：[Triplex](https://huggingface.co/SciPhi/Triplex) 使知识图谱创建成本降低了 **98%**，性能超越 **GPT-4**，而成本仅为后者的 **1/60**。
   - Triplex 由 [SciPhi.AI](https://www.sciphi.ai) 开发，是 Phi3-3.8B 的微调版本，专门用于从非结构化数据中提取三元组（triplets）。
- **SciPhi.AI 开源 Triplex**：一位成员分享了 SciPhi.AI 刚刚开源了 [Triplex](https://huggingface.co/SciPhi/Triplex)，使其可用于高效的知识图谱创建。
   - Triplex 支持使用 SciPhi 的 **R2R** 进行本地图谱构建，显著降低了构建知识图谱的成本。



**提到的链接**: <a href="https://huggingface.co/SciPhi/Triplex">SciPhi/Triplex · Hugging Face</a>: 未找到描述

  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1263995128284971128)** (3 messages): 

> - `OpenAI Scale Tier`
> - `GPT-4 Token 计算` 


- **OpenAI Scale Tier 之谜**：一位成员询问如何理解新的 [OpenAI Scale Tier](https://openai.com/api-scale-tier/)。
   - 该问题在社区中引发了关于 TPS 计算方式的困惑，特别是针对 **GPT-4** 模型。
- **GPT-4 TPS 计算困惑**：成员们对 OpenAI 在按需付费层级（pay-as-you-go tier）计算出的 **19 tokens/second** 感到困惑，指出 GPT-4 的实际输出速度约为 **80 tokens/second**。
   - 这引发了关于其 TPS 计算依据以及不同层级之间差异的讨论。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1263601428748832829)** (1 messages): 

> - `敏感数据担忧`
> - `数据隐私` 


- **与第三方共享敏感数据的担忧**：一位成员指出，许多企业不愿将**敏感的业务线数据**或**客户/患者数据**发送给另一家公司，这表明了对数据隐私和安全的担忧。
- **企业优先考虑数据隐私**：由于**隐私和安全**方面的考虑，企业在与外部实体共享敏感信息时变得越来越谨慎。


  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1263992625602494536)** (1 条消息): 

> - `目标受众明确` 


- **定义沟通的目标受众**：讨论围绕理解**目标受众**以实现有效沟通展开。
   - *对于 engineers，针对产品与 engineers 交流；对于准 engineers/product，与 devrels / solution architects 交流。*
- **有针对性沟通的重要性**：明确目标受众可确保沟通对特定群体具有相关性和影响力。
   - 其目的是为 **engineers**、**准 engineers**、**product managers**、**devrels** 和 **solution architects** 量身定制信息。


  

---



---



---



---



{% else %}


> 完整的频道明细已在邮件中截断。 
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}