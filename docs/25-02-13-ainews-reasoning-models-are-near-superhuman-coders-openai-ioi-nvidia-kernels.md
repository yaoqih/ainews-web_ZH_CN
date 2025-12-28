---
companies:
- openai
- nvidia
- ollama
- elevenlabs
- sakana-ai
- apple
date: '2025-02-14T02:42:41.628781Z'
description: '以下是该文本的中文翻译：


  **o3模型**在**2024年国际信息学奥林匹克竞赛（IOI）中摘得金牌**，并在**Codeforces上排名前99.8%**，超越了大多数人类。这证明了强化学习（RL）方法优于传统的归纳偏置（inductive
  bias）方法。**英伟达的DeepSeek-R1**能够自主生成GPU内核，其性能甚至超过了部分专家设计的内核，展示了简单而有效的AI驱动优化能力。**OpenAI**更新了**o1和o3-mini**模型，使其在ChatGPT中支持文件和图像上传，并发布了**DeepResearch**——这是一款基于**o3模型（结合强化学习）**的强大研究助手，具备深度思维链（CoT）推理能力。**Ollama**推出了基于**Qwen2.5**微调的**OpenThinker模型**，其表现优于部分DeepSeek-R1蒸馏模型。**ElevenLabs**已成长为一家估值达33亿美元的公司，专注于AI语音合成，且并未开源其技术。研究亮点还包括：**Sakana
  AI实验室的TAID知识蒸馏方法**在**ICLR 2025**上获得Spotlight论文奖，以及**苹果公司关于混合专家模型（MoEs）缩放法则**的研究。此外，开源AI对科学发现的重要性也再次得到了强调。'
id: 7b69cd80-a3f6-41e9-b1a7-5deff8ab7c50
models:
- o3
- o1
- o3-mini
- deepseek-r1
- qwen-2.5
- openthinker
original_slug: ainews-reasoning-models-are-near-superhuman
people:
- alex-wei
- karpathy
- abacaj
- awnihannun
title: 推理模型已具备接近超人类的编程能力（OpenAI IOI、英伟达内核）
topics:
- reinforcement-learning
- gpu-kernel-optimization
- fine-tuning
- knowledge-distillation
- scaling-laws
- chain-of-thought-reasoning
- model-accessibility
---

<!-- buttondown-editor-mode: plaintext -->**RL 就是你所需要的一切。**

> 2025年2月12日至2月13日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（包含 **211** 个频道和 **5290** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**554 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

这是两条不同新闻的汇总，但它们有着相同的主题：

- [o3 在 2024 年 IOI 中摘得金牌，并获得了与人类精英选手相当的 Codeforces 评分](https://reddit.com/r/MachineLearning/comments/1io4c7r/r_o3_achieves_a_gold_medal_at_the_2024_ioi_and/) —— 特别是其 Codeforces 评分处于 [99.8 百分位](https://reddit.com/r/OpenAI/comments/1iok7f3/gg_there_are_only_7_american_coders_better_than_o3/) —— 仅有 199 名人类选手的表现优于 o3。值得注意的是，团队成员 Alex Wei [指出](https://x.com/alexwei_/status/1889727571106918694)，与 RL 的“惨痛教训 (bitter lesson)”相比，所有的“归纳偏置 (inductive bias)”方法都失败了。 
![image.png](https://assets.buttondown.email/images/9051ed8f-279d-430a-beb0-ea4b94d18522.png?w=960&fit=max)

- 在 [利用 DeepSeek-R1 和推理时间扩展 (Inference Time Scaling) 自动化 GPU Kernel 生成](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/) 中，Nvidia 发现 DeepSeek-R1 可以编写自定义 Kernel，且“在某些情况下，**比资深工程师开发的优化 Kernel 表现更好**”。 
![image.png](https://assets.buttondown.email/images/7d517b27-cb0d-4131-91ed-e6115310c4a5.png?w=960&fit=max)


在 Nvidia 的案例中，解决方案也极其简单，这引发了许多人的惊愕。

![image.png](https://assets.buttondown.email/images/80e7cf66-efc5-417e-98a8-091e9edffac3.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**AI 工具与资源**

- **OpenAI 关于 o1、o3-mini 和 DeepResearch 的更新**：[OpenAI](https://twitter.com/OpenAI/status/1889822643676913977) 宣布 **o1 和 o3-mini 现在在 ChatGPT 中支持文件和图像上传**。此外，[DeepResearch 现已向所有 Pro 用户开放](https://twitter.com/OpenAI/status/1889812348581634146)，支持移动端和桌面端应用，**进一步扩大了可用性**。

- **使用 Ollama 在本地分发开源模型**：[@ollama](https://twitter.com/ollama/status/1889784880923394257) 讨论了**为开发者在本地分发和运行开源模型**，并强调这是对托管型 OpenAI 模型的补充。

- **@karpathy 的“LLM 深度解析”**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1889826850119229597) 分享了 **@karpathy 制作的一段 3 小时以上的免费视频**，探讨了 **ChatGPT 等 AI 模型是如何构建的**，包括**预训练、后训练、推理**以及如何有效使用模型等主题。

- **ElevenLabs 在 AI 语音合成领域的历程**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1889841943271842291) 详细介绍了 **@elevenlabsio 如何从一个周末项目演变为一家价值 33 亿美元的公司**，在不开源的情况下提供 AI 驱动的 TTS、语音克隆和配音工具。

- **OpenAI 的 DeepResearch 是一款令人惊叹的研究助手**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1890008648841396679) 评测了 **OpenAI 的 DeepResearch**，这是一款由**带有 RL 的强大 o3 模型**驱动的虚拟研究助手，专为**深度思维链 (chain-of-thought) 推理**而设计，并强调了其功能和优势。

- **OpenThinker 模型发布**：[@ollama](https://twitter.com/ollama/status/1890130798353031389) 宣布推出 **OpenThinker 模型**，该模型基于 **Qwen2.5** 进行微调，在某些基准测试中**超越了 DeepSeek-R1 蒸馏模型**。

**AI 研究进展**

- **DeepSeek R1 生成优化 Kernel**：[@abacaj](https://twitter.com/abacaj/status/1889847093046702180) 报告称，他们**让 R1 循环运行了 15 分钟**，生成的代码在某些情况下**“优于资深工程师开发的优化 Kernel”**。

- **开源 AI 对科学发现的重要性**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1889783322693476491) 强调，**如果不对开源 AI 进行投资**，可能会阻碍那些**负担不起闭源模型**的西方大学的科学发现。

- **Sakana AI Labs 的 'TAID' 论文在 ICLR2025 获得关注**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1889996178651312183) 宣布其新的**知识蒸馏方法 'TAID'** 被评为 **ICLR2025 的 Spotlight 论文（前 5%）**。

- **Apple 关于 Scaling Laws 的研究**：[@awnihannun](https://twitter.com/awnihannun/status/1890063767343706386) 重点介绍了 Apple 最近关于 **MoE 和知识蒸馏的 Scaling Laws** 的两篇论文，贡献者包括 **@samira_abnar、@danbusbridge** 等人。

**AI 基础设施与效率**

- **提倡在数据中心而非移动端进行 AI 计算**：[@JonathanRoss321](https://twitter.com/JonathanRoss321/status/1890045784738955546) 认为，与数据中心相比，**在移动设备上运行 AI 的能源效率较低**，并使用类比说明了效率差异。

**AI 安全与防护**

- **AI Web Agent 漏洞曝光**：[@micahgoldblum](https://twitter.com/micahgoldblum/status/1890078592929026329) 展示了攻击者如何**诱导 Anthropic 的 Computer Use 等 AI Web Agent 发送钓鱼邮件或泄露信用卡信息**，凸显了底层 LLM 的脆弱性。

- **Meta 的自动化合规加固 (ACH) 工具**：[@AIatMeta](https://twitter.com/AIatMeta/status/1890137619608268871) 介绍了他们的 **ACH 工具**，该工具通过**基于 LLM 的测试生成**来加固平台以防止回归，从而增强合规性和安全性。

**AI 治理与政策**

- **法国行动峰会的见解**：[@sarahookr](https://twitter.com/sarahookr/status/1889948845729259775) 分享了对**法国行动峰会**的观察，指出此类峰会作为**重要 AI 讨论的催化剂**非常有价值，并强调了了解**国家努力和科学进展**的重要性。

- **从“AI 安全”转向“负责任的 AI”**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1890076882391167317) 主张将对话从**“AI 安全”转向“负责任的 AI”**，认为这将**加速 AI 带来的益处**，并在不阻碍发展的情况下更好地解决实际问题。

**梗/幽默**

- **“Rizz GPT”与封锁后的社交挑战**：[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1890079063169224831) 幽默地评论道，**Z 世代正在开发各种版本的“Rizz GPT”**，因为他们的**大脑因封锁而受损**，不知道如何进行正常的对话。

- **“传染病的大日子”**：[@stevenheidel](https://twitter.com/stevenheidel/status/1890083974971875409) 发布了一条神秘消息，称今天是**“传染病的大日子”**，增添了一抹幽默感。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1. Google 的 FNet：通过傅里叶变换提升 LLM 效率的潜力**

- **这篇论文可能是 Google 自己都还没意识到的突破** ([Score: 362, Comments: 24](https://reddit.com/r/LocalLLaMA/comments/1io4s4s/this_paper_might_be_a_breakthrough_google_doesnt/))：**2022 年**的 **FNet** 论文探索了使用**傅里叶变换**来混合 Token，暗示在模型训练中可能获得巨大的效率提升。作者推测，复制这种方法或将其集成到更大的模型中可能会带来 **90% 的速度提升和内存减少**，为 AI 模型效率的进步提供了重大机遇。
  - **效率与收敛挑战**：用户反馈称，虽然 **FNet** 有效，但其效果不如传统的 Attention 机制，特别是在小模型中，并且面临严重的收敛问题。这让人对其在更大模型中的可扩展性和有效性产生怀疑。
  - **替代方案与对比**：讨论中提到了其他模型，如 **Holographic Reduced Representations (Hrrformer)**（声称以更少的训练获得更优的性能）和 **M2-BERT**（在基准测试中显示出更高的准确性）。这些替代方案凸显了在评估训练速度、准确性和泛化能力之间权衡的复杂性。
  - **集成与实现**：**FNet** 代码可在 [GitHub](https://github.com/google-research/google-research/tree/master/f_net) 上获得，但由于其是使用 **JAX** 实现的，将其与 Transformer 等现有模型集成并非易事。用户讨论了潜在的混合方法，例如创建 **fnet-llama** 或 **fnet-phi** 等变体，以探索性能差异和幻觉倾向。


**主题 2. 为 70B LLM 自建高性能服务器：策略与成本**

- **谁在组装能运行 70B 本地 LLMs 的电脑？** ([分数: 108, 评论: 160](https://reddit.com/r/LocalLLaMA/comments/1io811j/who_builds_pcs_that_can_handle_70b_local_llms/)): 讨论了构建能够运行 **70B 参数本地 LLMs** 的家用服务器，重点是使用价格合理的旧服务器硬件来最大化核心数、RAM 和 GPU RAM。作者询问是否有专门从事此类服务器组装的专业人士或公司，因为他们无法承担配备高端 GPU 的家用服务器通常所需的 **$10,000 到 $50,000** 的成本。
  - **Psychological_Ear393** 建议，使用 **Epyc 7532**、**256GB RAM** 和 **MI60 GPUs** 等组件，可以在 **$3,000** 以下构建用于 **70B 参数 LLMs** 的家用服务器。一些用户如 **texasdude11** 分享了使用双 **NVIDIA 3090s** 或 **P40 GPUs** 的配置以实现高效性能，并提供了详细的组装和操作指南及 YouTube 视频。
  - **NVIDIA A6000 GPU** 因其在运行 **70B 模型** 时的速度和能力而受到关注；然而，其约 **$5,000** 的价格较为昂贵。替代方案包括配备多个 **RTX 3090** 或 **3060 GPUs** 的设置，用户如 **Dundell** 和 **FearFactory2904** 建议使用二手组件进行更具成本效益的组装。
  - 用户讨论了使用 **Macs**（特别是配备 **128GB RAM** 的 **M1 Ultra**）高效运行 **70B 模型** 的可行性，尤其是在聊天应用中，正如 **synn89** 所指出的。未来的潜在选择包括等待 **Nvidia Digits** 或 **AMD Strix Halo**，它们可能为家用推理任务提供更好的性能。


**主题 3. Gemini 2.0 在 OCR 基准测试和上下文处理中的主导地位**

- **[Gemini 在视频 OCR 基准测试任务中击败了所有人。完整论文：https://arxiv.org/abs/2502.06445](https://i.redd.it/8u7jixwzmwie1.jpeg)** ([分数: 114, 评论: 26](https://reddit.com/r/LocalLLaMA/comments/1ioikl0/gemini_beats_everyone_is_ocr_benchmarking_tasks/)): **Gemini-1.5 Pro** 在视频 OCR 基准测试任务中表现出色，实现了 0.2387 的 **字符错误率 (CER)**、0.2385 的 **词错误率 (WER)** 以及 76.13% 的 **平均准确率**。尽管 **GPT-4o** 以 76.22% 的整体准确率略高且 WER 最低，但 **Gemini-1.5 Pro** 因其优于 **RapidOCR** 和 **EasyOCR** 等模型的性能而备受瞩目。[完整论文](https://arxiv.org/abs/2502.06445)
  - **RapidOCR** 被指出是 PaddleOCR 的一个分支，预计其得分与原版偏差极小。人们对使用 **Gemini-1.5 Pro** 探索直接 PDF 处理能力很感兴趣，并提供了在 **Google Cloud Vertex AI** 上实现的链接 [点击此处](https://cloud.google.com/vertex-ai/generative-ai/docs/samples/generativeaionvertexai-gemini-pdf#generativeaionvertexai_gemini_pdf-python)。
  - 用户表示 OCR 基准测试需要包含手写识别，**Azure FormRecognizer** 因处理草体文本而受到称赞。一位用户报告称，与其他语言模型相比，**Gemini 2.0 Pro** 在俄语手写笔记上的表现异常出色。
  - 有人呼吁在多种语言和模型之间进行更广泛的比较，包括 **Gemini 2**、**Tesseract**、**Google Vision API** 和 **Azure Read API**。尽管对 **Gemini** 处理简单任务的方式感到有些沮丧，但用户承认其在视觉标注方面的进步，且 **Moondream** 被强调为一个极具前景的新兴模型，并计划将其添加到 [OCR 基准测试仓库](https://github.com/video-db/ocr-benchmark) 中。

- **[NoLiMa：超越字面匹配的长上下文评估 —— 终于有一个好的基准测试能展示 LLM 在长上下文下的表现有多糟糕。所有模型在仅 32k 上下文时性能就大幅下降。](https://i.redd.it/95ysyjzs8sie1.png)** ([Score: 402, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1io3hn2/nolima_longcontext_evaluation_beyond_literal/)): **NoLiMa 基准测试** 强调了 **LLMs** 在长上下文长度下显著的性能退化，在 **GPT-4、Llama、Gemini 1.5 Pro 和 Claude 3.5 Sonnet** 等模型中，仅在 **32k tokens** 处就出现了明显的下降。图表和表格显示，在这些扩展长度下，得分降至基础得分的 50% 以下，表明在增加上下文时维持性能面临巨大挑战。
  - **性能退化与基准测试对比**：**NoLiMa 基准测试** 显示了像 **llama3.1-70B** 这样的 **LLMs** 存在实质性的性能退化，其在 32k 上下文长度下的得分为 43.2%，而相比之下在 **RULER** 上的得分为 94.8%。该基准测试被认为比之前的 **LongBench** 更具挑战性，后者侧重于多项选择题，无法充分捕捉跨上下文长度的性能退化。
  - **模型性能与架构担忧**：关于 **o1/o3** 等 **reasoning models** 如何处理长上下文存在大量讨论，部分模型在难题子集上表现不佳。当前架构的局限性（如 attention 机制的平方复杂度）被强调为维持长上下文性能的障碍，这表明需要像 **RWKV** 和 **linear attention** 这样的新架构。
  - **未来模型测试与预期**：参与者对测试 **Gemini 2.0-flash/pro** 和 **Qwen 2.5 1M** 等新模型表现出兴趣，希望在长上下文场景中能有更好的表现。对于模型能有效处理 128k tokens 的说法存在怀疑，一些用户强调实际应用通常在 8k tokens 以下的上下文中表现最好。


**主题 4. 来自 DeepSeek 的创新架构见解：专家混合与 Token 预测**

- **从零开始构建 DeepSeek | 由 MIT 博士毕业生授课** ([Score: 245, Comments: 28](https://reddit.com/r/LocalLLaMA/comments/1iohk4o/lets_build_deepseek_from_scratch_taught_by_mit/))：一位 MIT 博士毕业生正在推出一系列关于从零构建 **DeepSeek 架构** 的综合教育课程，重点关注基础元素，如 **Mixture of Experts (MoE)**、**Multi-head Latent Attention (MLA)**、**Rotary Positional Encodings (RoPE)**、**Multi-token Prediction (MTP)**、**Supervised Fine-Tuning (SFT)** 和 **Group Relative Policy Optimisation (GRPO)**。该系列包含 35-40 个深度视频，总时长超过 40 小时，旨在让参与者具备独立构建 DeepSeek 组件的能力，使他们跻身前 0.1% 的 ML/LLM 工程师行列。
  - **对资历的怀疑**：一些用户对强调 MIT 等名校资历表示怀疑，认为内容的质量应该独立于作者的背景。有人呼吁评判内容时应脱离创作者的学术或职业背景。
  - **缺失的技术细节**：讨论的一个重点是该系列忽略了 **Nvidia 的 Parallel Thread Execution (PTX)** 作为 CUDA 的高性价比替代方案，这被视为在探讨 DeepSeek 的效率和成本效益方面存在缺失。这表明，理解技术底层原理而不仅仅是功能，对于理解 DeepSeek 的架构至关重要。
  - **算力的不确定性**：关于 DeepSeek 开发中使用的实际算力存在争论，一些用户批评了网上流传的推测数字。讨论强调了准确数据（特别是在数据集和算力资源方面）对于理解和复制 AI 系统的重要性。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. OpenAI 将 o3 合并为统一的 GPT-5**

- **[OpenAI 取消其 o3 AI 模型，转而发布“统一”的下一代版本](https://techcrunch.com/2025/02/12/openai-cancels-its-o3-ai-model-in-favor-of-a-unified-next-gen-release/)** ([Score: 307, Comments: 66](https://reddit.com/r/ChatGPT/comments/1ioagn0/openai_cancels_its_o3_ai_model_in_favor_of_a/)): **OpenAI** 已决定取消其 **o3 AI 模型**项目，转而专注于开发包含 **GPT-5** 的**统一下一代版本**。这一战略转变表明，资源和精力正向单一且更先进的 AI 模型整合。
  - 关于 **OpenAI** 将 **o3 模型**集成到 **GPT-5** 的举动究竟代表了进步还是缺乏创新，存在重大争议。一些用户认为这种集成是为了提高易用性的战略简化，而另一些用户则将其视为模型开发的停滞迹象，并引用 **DeepSeek R1** 作为具有竞争力的免费替代方案。
  - 许多评论者对自动模型选择方式导致的**用户控制权丧失**表示担忧，担心这可能导致次优结果。像 **whutmeow** 和 **jjjiiijjjiiijjj** 这样的用户更倾向于手动选择模型，担心算法决策可能会优先考虑公司成本而非用户需求。
  - 讨论还强调了对所用术语的困惑，几位用户纠正了 **o3 被取消**的说法，澄清其正在被集成到 **GPT-5** 中。这引发了对误导性标题以及对 **OpenAI** 战略方向和领导层潜在影响的担忧。


- **Altman 把潜台词说了出来** ([Score: 126, Comments: 59](https://reddit.com/r/OpenAI/comments/1iotr9r/altman_said_the_silent_part_out_loud/)): **Sam Altman** 宣布，最初计划作为 **GPT-5** 的 **Orion** 将改为以 **GPT-4.5** 的名义发布，并将其定性为最后一个非 **CoT** 模型。来自 **Bloomberg**、**The Information** 和 **The Wall Street Journal** 的报道指出，**GPT-5** 面临重大挑战，其相对于 **GPT-4** 的提升幅度小于前代产品。此外，**o3 模型**将不会单独发布，而是集成到名为 **GPT-5** 的统一系统中，这可能是由于 **ARC benchmark** 揭示的高昂运营成本，如果被用户广泛且随意地使用，可能会给 OpenAI 带来财务压力。
  - **硬件与成本**：在不合适的硬件（如 **Blackwell chips**）上运行模型的低效率，以及由于重复查询导致的成本数据误读，被认为是影响 **GPT-4.5** 和 **GPT-5** 发布和运营的因素。**Deep Research** 查询的定价为每次 **$0.50**，计划为 Plus 会员提供 **10** 次，为免费用户提供 **2** 次。
  - **模型演进与挑战**：人们普遍认为**非 CoT 模型**可能已经达到了可扩展性极限，从而促使向具有推理能力的模型转变。这种转型被视为必要的演进，一些人认为 **GPT-5** 代表了一个新方向，而非对 **GPT-4** 的迭代改进。
  - **推理与模型选择**：讨论强调了推理模型的潜在优势，一些用户指出像 **o3** 这样的推理模型可能会“过度思考”，而另一些用户则建议使用混合方法为特定任务选择最合适的模型。关于具有可调思考时间的模型成本高昂的概念也引发了辩论，以及 OpenAI 实施使用上限以管理开支的可能性。

- **我 50 多岁了，刚刚让 ChatGPT 为我的网站写了一个 JavaScript/HTML 计算器。我被震撼到了。** ([Score: 236, Comments: 62](https://reddit.com/r/ChatGPT/comments/1iosoyp/im_in_my_50s_and_i_just_had_chatgpt_write_me_a/)): 作者已年过五旬，他使用 **ChatGPT** 为自己的网站创建了一个 JavaScript/HTML 计算器，并对其理解模糊指令和优化代码的能力感到印象深刻，认为这就像与 Web 开发者对话一样。尽管之前很少使用 AI，但他对其能力感到惊讶，并回顾了自 1977 年以来观察技术进步的漫长历史。
  - 用户分享了 **ChatGPT** 辅助完成各种编程任务的经验，从创建 SQL 查询到构建维基和服务器，强调了它在引导学习陌生技术方面的效用。**FrozenFallout** 和 **redi6** 强调了它在简化复杂流程和错误处理方面的作用，即使对于技术知识有限的人也是如此。
  - **Front_Carrot_1486** 和 **BroccoliSubstantial2** 对 AI 的飞速发展表达了共同的惊叹，将其与过去的技术变革相提并论，并指出了见证技术从科幻变为现实的代际视角。他们赞赏 AI 提供解决方案和替代方案的能力，尽管偶尔会出现错误。
  - 进一步探索的建议包括尝试使用 **Cursor** 会员以获得更令人印象深刻的体验（由 **TheoreticalClick** 建议），以及在 AI 指导下探索应用开发（由 **South-Ad-9635** 提到）。


**主题 2. Anthropic 和 OpenAI 增强推理模型**

- **[OpenAI 将其最先进推理模型的速率限制提高了 7 倍。现在轮到你了，Anthropic。](https://i.redd.it/jauypbvupwie1.jpeg)** ([Score: 486, Comments: 72](https://reddit.com/r/ClaudeAI/comments/1ioitxd/openai_increased_its_most_advanced_reasoning/)): **OpenAI** 显著提高了其高级推理模型 **o3-mini-high** 的速率限制，为 **Plus 用户** 提高了 **7 倍**，现在每天允许使用多达 **50 次**。此外，**OpenAI o1** 和 **o3-mini** 现在支持在 **ChatGPT** 中上传文件和图像。
  - 用户对 **Anthropic** 在竞争压力下表现出的紧迫感缺失表示强烈不满，一些人由于缺乏引人注目的更新或功能而取消了订阅。担忧包括该公司对安全和内容审核的关注超过了创新，可能失去竞争优势。
  - **OpenAI** 的 **o3-mini-high** 模型速率限制的提高受到了积极评价，尤其是 **Plus 用户**，他们非常欣赏这种增强的访问权限。然而，一些人认为 **OpenAI** 优先考虑 API 商业客户而非 Web/App 用户，导致后者的限制较低。
  - 存在一种对 **Anthropic** 失望的情绪，用户觉得他们对安全和企业客户的关注掩盖了创新和对市场竞争的响应。一些用户对 **Claude** 的局限性以及缺乏具有类似能力的替代方案感到沮丧。


- **[The Information：Claude 混合推理模型可能在未来几周内发布](https://www.theinformation.com/articles/anthropic-strikes-back?utm_source=ti_app)** ([Score: 160, Comments: 44](https://reddit.com/r/ClaudeAI/comments/1iom1k0/the_information_claude_hybrid_reasoning_model_may/)): 据报道，Anthropic 将在未来几周内发布 **Claude 混合推理模型**，提供一个滑动缩放功能，当设置为 0 时会退回到非推理模式。据称该模型在某些编程基准测试中优于 **o3-mini**，并且在典型的编程任务中表现出色，而 **OpenAI** 的模型在学术和竞赛编程方面更胜一筹。
  - **Anthropic 对安全的关注**受到了一些用户的批评，他们将其与 **OpenAI** 减少的审查以及 **Gemini 2.0** 模型进行了比较，后者因限制较少而受到称赞。一些用户认为审查工作无关紧要，而另一些人则将其视为不必要的企业迎合。
  - 人们对 **Claude 混合推理模型** 在写作任务中的有效性持怀疑态度，担心它可能会遇到与 **o3-mini** 类似的问题。用户表示需要更大且更有效的上下文窗口（context windows），并指出 **Claude** 所谓的 200k token 上下文在超过 32k token 后开始显著退化。
  - 用户讨论了**上下文窗口**和**输出复杂性**的重要性，一些人认为 **o3-mini-high** 的输出过于复杂，而另一些人则强调需要一个在超过 64k token 后仍能保持完整性的上下文窗口。

- **[Deep reasoning coming soon](https://www.reddit.com/gallery/1ior2du)** ([Score: 121, Comments: 42](https://reddit.com/r/ClaudeAI/comments/1ior2du/deep_reasoning_coming_soon/)): 标题为 **"Deep reasoning coming soon"** 的帖子，正文内容为 **"Hhh"**，缺乏实质性内容和上下文，无法提供详细摘要。
  - **代码输出担忧**: **Estebansaa** 对深度推理的价值表示怀疑，如果它不能超越目前 **300-400 行代码** 的输出，并达到 **o3** 每次请求超过 **1000 行** 的能力。**Durable-racoon** 质疑是否需要如此大量的输出，认为即使是 **300 行** 的代码审查起来也可能让人不堪重负。
  - **API 访问问题**: **Hir0shima** 等人讨论了 API 访问的挑战，强调了高昂的成本和频繁的错误。**Zestyclose_Coat442** 指出即使发生错误也会产生意外费用，而 **Mutare123** 提到了达到响应限制的可能性。
  - **发布紧迫感**: **Joelrog** 指出，最近关于深度推理的公告仍处于较短的时间窗口内，反对急躁情绪，并强调公司通常会遵守其发布计划。


---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Exp 生成的摘要之摘要

**主题 1. 推理 LLM 模型 - 新发布趋势**

*   [**Nous Research 推出 DeepHermes-3 以实现卓越推理**](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview): **Nous Research** 发布了 **DeepHermes-3 Preview**，展示了在统一推理和直觉语言模型能力方面的进展。该模型需要特定的带有 `<think>` 标签的系统提示词（System Prompt）来启用长思维链（Chain of Thought）推理，从而增强系统性问题解决能力。基准测试报告显示其在数学推理方面有**显著增强**。
*   [**Anthropic 计划推出集成推理功能的 Claude 版本**](https://x.com/steph_palazzolo/status/1890058003493343453): **Anthropic** 准备发布一个新的 **Claude** 模型，结合了传统 LLM 和推理 AI 的能力，通过基于 Token 的滑动标尺进行控制，适用于编程等任务。传闻该模型在多个基准测试中可能优于 **OpenAI** 的 **o3-mini-high**。这些模型及其新能力的亮相标志着 AI 系统混合思考新时代的开始。
*   [**智力规模化，Elon 承诺推出 Grok-3**](https://ca.finance.yahoo.com/news/elon-musk-says-grok-3-064145838.html): **Elon Musk** 宣布 **Grok 3** 即将发布，吹嘘其拥有超越现有模型的卓越推理能力，暗示其将达到“可怕的聪明”新高度，预计在大约两周内发布。与此同时，他正以 **974 亿美元** 竞购 OpenAI 的非营利资产。这一公告预计将改变游戏规则。

**主题 2. 小而强大的 LLM 与工具改进**

*   [**DeepSeek-R1 像老板一样生成 GPU Kernel**](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940): **NVIDIA** 的博客文章介绍了 LLM 生成的 GPU Kernel，展示了 **DeepSeek-R1** 加速 **FlexAttention** 并在 🌽KernelBench Level 1 上实现 **100% 数值正确性**，同时在 KernelBench 基准测试的 Level-2 问题上达到 96% 的准确率。这通过分配额外资源实现了计算密集型任务的自动化，但也引发了对基准测试本身的关注。
*   [**Hugging Face Smolagents 问世，简化你的工作流**](https://huggingface.co/spaces/m-ric/open_Deep-Research): **Hugging Face** 推出了 **smolagents**，这是一个轻量级的 Agent 框架，可作为 `deep research` 的替代方案，**6 个步骤** 的处理时间约为 **13 秒**。用户可以修改原始代码，以便在本地服务器运行时扩展执行，提供了极佳的适应性。
*    [**Codeium 的 MCP 提升编程能力**](https://codeium.com/blog/windsurf-wave-3): **Windsurf Wave 3** (Codeium) 引入了 **Model Context Protocol (MCP)** 等功能，集成了多个 AI 模型以提高效率和输出质量，允许用户配置工具调用到自定义的 **MCP 服务器**，并获得更高质量的代码。社区对这一混合 AI 框架感到兴奋！

**主题 3. Perplexity 财经仪表盘与 AI 模型分析**

*   [**Perplexity 推出全能 Finance Dashboard**](https://www.perplexity.ai/search?q=%s&focus=[internet,scholar,writing,wolfram,youtube,reddit]&copilot=[true,false]): **Perplexity** 发布了新的 [Finance Dashboard](https://www.perplexity.ai/search?q=%s&focus=[internet,scholar,writing,wolfram,youtube,reddit]&copilot=[true,false])，提供市场摘要、每日亮点和收益快报。用户正请求在 Web 端和移动端 App 中添加专门的仪表盘按钮。
*   [**Perplexity AI 模型性能受到严格审查**](https://www.perplexity.ai/search?q=%s&focus=[internet,scholar,writing,wolfram,youtube,reddit]&copilot=[true,false]): **Perplexity AI** 使用的模型受到质疑。关于 AI 模型的争论浮出水面，特别是 **R1** 与 **DeepSeek** 和 **Gemini** 等替代方案相比的效率和准确性。
*   [**DeepSeek R1 在推理能力上击败 OpenAI**](https://openrouter.ai/deepseek/deepseek-r1/providers): 一位用户报告称 **DeepSeek R1** 在处理复杂的 SIMD 函数时表现出令人印象深刻的推理能力，在 OpenRouter 上的表现优于 **o3-mini**。HuggingFace 上的用户赞扬了 V3 的编程能力，而 Unsloth AI 的一些用户看到它成功处理了合成数据和 GRPO/R1 蒸馏任务。

**主题 4. 挑战与创意解决方案**

*   [**DOOM 游戏被压缩进二维码**](https://github.com/Kuberwastaken/backdooms): 一名成员成功地将一款名为 **The Backdooms** 的可玩 DOOM 启发式游戏塞进了一个二维码中，占用空间不到 **2.4kb**，并以 **MIT license** 开源该项目供他人实验。该项目使用了类似 .kkrieger 的压缩技术，并有一篇博文记录了实现方法。
*   [**移动设备限制 RAM，促使寻找替代方案**](https://discord.com/channels/1053877538025386074/1149866623109439599/1339308516321787955): 移动用户指出 **12GB 手机仅允许 2GB 的可用内存**，阻碍了模型性能，有人建议使用约 100 美元的 **16GB ARM SBC** 作为便携式计算的替代方案。如果你没有高端手机，那就升级它。
*   [**Hugging Face Agents 课程导致用户连接中断**](https://discord.com/channels/879548962464493619/1329142738440028273/1339279913928232971): 由于用户在 Agents 课程期间遇到连接问题，一名成员建议将端点更改为新链接 (https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud)，并指出需要更新模型名称为 **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**，这可能是由过载引起的。尝试更换浏览器，如果一切都失败了……断开连接。

**主题 5. 数据、版权与声明**

*   [**美国拒绝 AI 安全协议，声称要保持竞争优势**](https://arstechnica.com/ai/2025/02/us-and-uk-refuse-to-sign-ai-safety-declaration-at-summit/): **美国**和**英国**拒绝签署联合 AI 安全声明，美国领导人强调他们致力于保持 AI 领导地位。官员们警告说，在 AI 领域与威权国家接触可能会损害基础设施安全。
*   [**汤森路透赢得具有里程碑意义的 AI 版权案**](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/): 汤森路透在针对 Ross Intelligence 的重大 [AI 版权案](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/) 中获胜，判定该公司通过复制 Westlaw 的材料侵犯了其版权。美国巡回法院法官 Stephanos Bibas 驳回了 Ross 的所有辩护，称其*全都站不住脚*。
*   [**LLM Agents MOOC 证书处理延迟**](https://discord.com/channels/1280234300012494859/1280370030609170494/1339332225757610034): 许多 LLM Agents 课程的参与者尚未收到之前的证书，必须等待手动发送，而其他需要帮助寻找证书的人被提醒，**declaration form** 是证书处理所必需的。我的证书在哪？


---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 的动态量化减少内存占用**：用户解释说，在 **DeepSeek 模型**中实现的动态量化有助于在保持性能的同时减少内存使用，尽管目前主要适用于特定模型，并详细说明了其[优势](https://unsloth.ai/blog/deepseekr1-dynamic)。
   - 动态量化是一项正在进行的工作，旨在减少 VRAM 占用并*运行 Unsloth 的 1.58-bit 动态 GGUF 版本*。
- **GRPO 训练平台期与提示词正则表达式微调**：关于在 **GRPO 训练**期间获得预期奖励的问题被提出，用户观察到性能指标出现平台期以及生成长度出现意外变化，更多细节见 [Unsloth 的 GRPO 博客文章](https://unsloth.ai/blog/r1-reasoning)。
   - 一位用户报告称通过修改正则表达式（regex）获得了更好的训练结果，但指标的不一致仍然是一个问题，且对 **Llama3.1 (8B)** 性能指标的影响尚不明确。
- **Rombo-LLM-V3.0-Qwen-32b 表现亮眼**：新模型 **Rombo-LLM-V3.0-Qwen-32b** 已发布，在各项任务中展现出令人印象深刻的性能，更多细节见 [Reddit 帖子](https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/)。
   - 详细介绍了如何通过 Patreon 支持模型开发者的工作，每月仅需 5 美元即可为未来的模型投票并访问私有仓库。
- **Lavender 方法增强 VLMs**：**Lavender** 方法作为一种监督微调技术被引入，它利用 **Stable Diffusion** 提高了视觉语言模型（**VLMs**）的性能，代码和示例可在 [AstraZeneca 的 GitHub 页面](https://astrazeneca.github.io/vlm/)找到。
   - 该方法实现了性能提升，包括在 20 个任务上增长了 **+30%**，在 OOD WorldMedQA 上提升了 **+68%**，展示了文本-视觉注意力对齐（attention alignment）的潜力。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DOOM 游戏被压缩进二维码**：一名成员成功将一款名为 **The Backdooms** 的可玩 DOOM 启发游戏塞进了一个二维码中，占用空间不足 **2.4kb**，并以 **MIT 许可证**在 [GitHub](https://github.com/Kuberwastaken/backdooms) 开源了该项目。
   - 作者在[这篇博客文章](https://kuberwastaken.github.io/blog/Projects/How-I-Managed-To-Get-Doom-In-A-QR-Code)中记录了实现方法，提供了关于技术挑战和解决方案的见解。
- **Steev AI 助手简化模型训练**：一个团队推出了 Steev，这是一个旨在自动化 AI 模型训练的 AI 助手，减少了对持续监督的需求，更多信息请访问 [Steev.io](https://www.steev.io/)。
   - 其目标是简化 AI 训练过程，消除乏味且重复的任务，让研究人员能够专注于模型开发和创新的核心环节。
- **Rombo-LLM V3.0 在编程方面表现出色**：新模型 **Rombo-LLM-V3.0-Qwen-32b** 已发布，在编程和数学任务中表现优异，如这篇 [Reddit 帖子](https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/)所示。
   - 该模型使用的 **Q8_0 量化**显著提升了效率，使其能够在没有高计算需求的情况下执行复杂任务。
- **Agents 课程验证出现问题**：**Hugging Face AI Agents 课程**的许多参与者报告了通过 Discord 验证账户时遇到的持续问题，导致反复出现连接失败。
   - 推荐的解决方案包括退出登录、清除缓存以及尝试不同的浏览器，少数幸运儿最终完成了验证过程。
- **建议为 Agent 连接使用新端点**：针对用户在 Agents 课程中遇到的连接问题，一名成员建议将端点（endpoint）更改为新链接 (https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud)，并指出需要将模型名称更新为 **deepseek-ai/DeepSeek-R1-Distill-Qwen-32B**。
   - 这一修复方案可能会解决课程参与者在 Agent 工作流中使用 LLM 时遇到的一系列近期问题。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Pro 用户获得 Deep Research 访问权限**：**Deep research 访问权限**现已面向所有 **Pro 用户**开放，涵盖移动端和桌面端应用（iOS, Android, macOS 和 Windows）。
   - 这增强了在各种设备上的研究能力。
- **OpenAI o1 & o3 支持文件和图像上传**：**OpenAI o1** 和 **o3-mini** 现在支持在 ChatGPT 中进行*文件*和*图像上传*。
   - 此外，Plus 用户的 **o3-mini-high 限制**已提升 **7 倍**，每天允许最多 **50 次上传**。
- **OpenAI 发布 Model Spec 更新**：OpenAI 分享了 [Model Spec](https://openai.com/index/sharing-the-latest-model-spec/) 的**重大更新**，详细说明了对模型行为的预期。
   - 该更新强调了对**可定制性**、**透明度**以及培养*思想自由*氛围的承诺。
- **OpenAI 的所有权面临审查**：讨论围绕 Elon Musk 收购 OpenAI 的可能性展开，许多人对此表示怀疑，并希望如果发生收购，能将技术开源。
   - 用户推测大型科技公司可能会将利润置于公共利益之上，从而导致对 AI 过度控制的担忧。
- **GPT-4o 设有免费限制**：Custom GPTs 在 **GPT-4o** 模型上运行，其限制根据各种因素每天都在变化，*只有部分固定值*，如 AVM 为 **15min/month**。
   - 用户必须根据其**地区和使用时区**来关注限制。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **o3-mini 落后于 Claude**：用户观察到 **OpenAI 的 o3-mini** 模型在 tool calling 方面表现不如 **Claude**，通常需要多次 prompt 才能达到预期结果，这引发了不满。
   - 许多人表示 **Claude** 的推理模型在工具使用方面表现出色，并建议集成类似于 Cline 的 Plan/Act 模式以改善用户体验。
- **混合 AI 模型令开发者兴奋**：社区对 **Anthropic 即将推出的混合 AI 模型**表现出兴奋，据报道，在利用最大推理能力时，该模型在编程任务上超越了 **OpenAI 的 o3-mini**。
   - 这种期待源于新模型在编程基准测试中的高性能，表明相对于目前的替代方案，它可以显著提升编码工作流。
- **Tool Calling 引发关注**：用户对 **o3-mini** 在 tool calling 方面的灵活性和效率有限表示不满，质疑其在真实编程场景中的实用性。
   - 讨论强调了对 AI 模型简化复杂编码任务的需求，并建议建立 prompting 最佳实践以引导出更高质量的代码。
- **MCP 使用成为讨论话题**：**MCP (Multi-Channel Processor)** 的概念作为一种通过集成多个 AI 模型来提高效率和产出的工具浮出水面。
   - 用户一直在分享利用 MCP 服务器优化编码工作流并克服单一模型局限性的经验和策略。
- **Windsurf 定价令人不满**：讨论涉及 **Windsurf** 僵化的定价，特别是限制用户使用自己的 key，这导致了用户的不满。
   - 许多用户表示相比竞争对手更青睐 **Cursor** 的功能和实用性，强调了其在性价比和整体用户体验方面的优势。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepHermes-3 统一推理能力**：Nous Research 发布了 **DeepHermes-3 Preview**，这是一个将推理能力与传统语言模型相结合的 LLM，可在 [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) 上获取。
   - 该模型需要特定的带有 `<think>` 标签的系统提示词（system prompt）来启用长链条思维链（chain of thought）推理，从而增强系统化问题解决能力，并在数学推理方面表现出**显著提升**。
- **Nvidia 的 LLM 内核加速**：[Nvidia 的博客文章](https://x.com/anneouyang/status/1889770174124867940)介绍了由 LLM 生成的 GPU 内核，这些内核在加速 **FlexAttention** 的同时，在 🌽KernelBench Level 1 上实现了 **100% 的数值正确性**。
   - 这标志着 GPU 性能优化取得了显著进展，成员们还推荐了 **r1 kimik** 和 **synthlab** 的论文，以获取有关 **LLM 进展** 的最新信息。
- **移动设备 RAM 限制**：成员们注意到 12GB 的手机仅允许 2GB 的可访问内存，这阻碍了他们运行模型的能力。
   - 一位用户建议购买 16GB 的 ARM SBC 用于便携式计算，这样可以在旅行时以约 100 美元的价格运行小型 LLM，为感兴趣的人提供了一个实惠的选择。
- **美国拒绝 AI 安全协议**：据 [ArsTechnica 报告](https://arstechnica.com/ai/2025/02/us-and-uk-refuse-to-sign-ai-safety-declaration-at-summit/)，**美国**和**英国**拒绝签署一份联合 AI 安全宣言，美国领导人强调他们致力于保持 AI 领导地位。
   - 官员们警告说，在 AI 领域与威权国家接触可能会损害国家基础设施安全，并引用了 **CCTV** 和 **5G** 作为通过补贴出口以施加不当影响的例子。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Groq DeepSeek R1 70B 冲刺 1000 TPS**：OpenRouter 用户正在庆祝 **Groq DeepSeek R1 70B** 的加入，其吞吐量达到了 **每秒 1000 个 token**，并提供参数自定义和速率限制调整。该公告发布在 [OpenRouter AI 的 X 账号](https://x.com/OpenRouterAI/status/1889726731571044538)上。
   - 这是旨在增强用户与平台交互的更广泛集成的一部分。
- **新的排序调整提升 UX**：用户现在可以在账户设置中自定义模型提供商的默认排序，专注于**吞吐量**或平衡速度与成本。如 [OpenRouter 的推文](https://x.com/OpenRouterAI/status/1890061196885360647)所述，要在任何模型名称后附加 `:nitro` 即可访问最快的可用提供商。
   - 此功能允许用户根据自己的优先级定制体验。
- **API 采用原生 Token 计数**：OpenRouter 计划将 API 中的 `usage` 字段从 GPT token 归一化切换为模型的**原生 token 计数**，并正在征求用户反馈。
   - 有推测认为这一变化可能会影响像 **Vertex** 这样具有不同 token 比例的模型。
- **Deepseek R1 在推理方面击败 OpenAI**：一位用户报告称，**Deepseek R1** 在处理复杂的 SIMD 函数时表现出令人印象深刻的推理能力，优于表现“固执”的 **o3-mini**。
   - 团队正在探索这一选项，并承认了用户对审核问题的担忧。
- **用户抱怨 Google 的速率限制**：由于资源耗尽，用户经常遇到来自 Google 的 **429 错误**，尤其是影响到 Sonnet 模型。
   - OpenRouter 团队正在积极解决由 Anthropic 容量限制引起的日益严重的速率限制问题。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 发布 Finance Dashboard**：Perplexity 发布了全新的 [Finance Dashboard](https://www.perplexity.ai/search?q=%s&focus=[internet,scholar,writing,wolfram,youtube,reddit]&copilot=[true,false])，提供市场摘要、每日亮点和收益快报。
   - 用户请求在 Web 端和移动端 App 中添加仪表盘的专用按钮。
- **对 AI 模型性能产生质疑**：关于 AI 模型的讨论不断涌现，特别是 **R1** 与 **DeepSeek** 和 **Gemini** 等替代方案相比的效率和准确性，以及首选的使用方式和性能指标。
   - 成员们分享了使用经验，并提出了可以改善用户体验的功能和特性。
- **Perplexity 支持服务受到批评**：一名用户报告称，Perplexity 的客户服务响应缓慢且缺乏支持，涉及 Pro 账户已付费但无法访问的问题。
   - 这引发了关于建立清晰沟通和高效支持团队必要性的讨论。
- **API 遭遇大规模 500 错误**：多名成员报告所有 API 调用均出现 **500 错误**，导致生产环境故障。
   - 在 API 恢复正常之前，这些错误持续了一段时间。
- **对 Cerebras 上的 Sonar 充满热情**：一名成员表达了成为 **Cerebras** 上 **Sonar** API 版本 **beta tester** 的强烈兴趣。
   - 该成员表示他们已经梦想了几个月，表明了对这一集成的潜在兴趣。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Wave 3 正式上线！**：Windsurf **Wave 3** 引入了 **Model Context Protocol (MCP)**、可自定义的应用图标、增强的 **Tab to Jump** 导航以及 **Turbo Mode**，详见 [Wave 3 博客文章](https://codeium.com/blog/windsurf-wave-3)。
   - 此次更新还包括重大升级，如自动执行命令和改进的额度可见性，详见[完整变更日志](https://www.codeium.com/changelog)。
- **更新后 MCP 服务器选项难以找到？**：更新 Windsurf 后，部分用户报告难以找到 **MCP server options**，该问题通过重新加载窗口得到解决。
   - 此问题强调了刷新界面的重要性，以确保 **MCP settings** 按预期显示，从而允许配置对用户定义 **MCP servers** 的工具调用。
- **Cascade 饱受性能问题困扰**：用户报告 **Cascade model** 出现性能迟缓和频繁崩溃，通常需要重启才能恢复功能。
   - 报告的挫败感包括运行期间响应时间慢和 CPU 占用率增加，突显了稳定性问题。
- **Codeium 1.36.1 旨在修复 Bug**：**Codeium 1.36.1** 的发布旨在解决现有问题，建议用户在此期间切换到 **pre-release** 版本。
   - 之前修复 **2025** 写作问题的尝试未能成功，突显了此次更新的必要性。
- **Windsurf Chat 饱受不稳定性困扰**：Windsurf 聊天用户正经历频繁的卡死、对话历史丢失和工作流中断。
   - 建议的解决方案包括重新加载应用程序并报告 Bug，以解决这些关键的**稳定性问题**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen-2.5 VL 面临性能瓶颈**：用户报告 **Qwen-2.5 VL** 模型存在响应速度慢和内存问题，特别是在后续提示词（follow-up prompts）之后，会导致显著的延迟。
   - 该模型的 **内存占用激增**，可能依赖于 SSD 而非 VRAM，这在高配置机器上尤为明显。
- **Speculative Decoding 需要调整设置**：上传与 **Speculative Decoding**（投机采样）相关的模型时遇到困难，经过排查发现用户需要调整设置并确保选择了兼容的模型。
   - 该问题强调了将模型配置与所选 **speculative decoding** 功能相匹配的重要性。
- **Tesla K80 PCIe 引发讨论**：讨论了使用售价 60 美元、拥有 24GB VRAM 的 **Tesla K80 PCIe** 执行 LLM 任务的潜力，引发了对功耗和兼容性的担忧。
   - 用户建议，虽然价格低廉，但 K80 过旧的架构和潜在的安装问题可能使得 **GTX 1080 Ti** 成为更好的替代方案。
- **SanDisk 通过 HBF 内存大幅提升 VRAM**：**SanDisk** 推出了新型高带宽闪存，能够在 GPU 上实现 **4TB 的 VRAM**，旨在用于需要高带宽和低功耗的 AI 推理应用。
   - [据 Tom's Hardware 报道](https://www.tomshardware.com/pc-components/dram/sandisks-new-hbf-memory-enables-up-to-4tb-of-vram-on-gpus-matches-hbm-bandwidth-at-higher-capacity)，这种 **HBF memory** 将自己定位为未来 AI 硬件中传统 **HBM** 的潜在替代品。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell 的 Tensor Memory 受到审视**：讨论澄清了 **Blackwell GPU 的 tensor memory** 完全由程序员管理，具有专用的分配函数（[详见此处](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions)），并且它也是矩阵乘法中寄存器的替代品。
   - 关于 **tensor memory** 在处理 **sparsity**（稀疏性）和 **microtensor scaling**（微张量缩放）时的效率出现了争论，如果未充分利用，可能会导致容量浪费，并增加在 Streaming Multiprocessors 上拟合累加器的复杂性。
- **D-Matrix 推广创新的 Kernel 工程**：**D-Matrix** 正在招聘 Kernel 开发人员，邀请具有 CUDA 经验的人员联系并探索其独特技术栈中的机会，建议联系 [LinkedIn 上的 Gaurav Jain](https://www.linkedin.com/in/gauravjain14/) 以了解其创新硬件和架构。
   - **D-Matrix 的 Corsair** 技术栈旨在提高速度和能效，可能改变大规模推理的经济效益，并声称对 H100 GPU 具有竞争优势，强调 AI 的可持续解决方案。
- **SymPy 简化反向传播推导**：成员们对使用 [SymPy](https://www.sympy.org/en/index.html) 推导算法的反向传播（backward pass）以管理复杂性表现出兴趣。
   - 围绕 `gradgradcheck()` 遇到的问题进行了讨论，涉及非预期的输出行为，意在澄清要点并在问题持续时在 GitHub 上进行跟进，这暗示了维护准确中间输出的复杂性。
- **Reasoning-Gym 改进评估指标**：**Reasoning-Gym** 社区讨论了 **MATH-P-Hard** 上的性能下降，并发布了一个针对 **Graph Coloring Problems**（图着色问题）的新 Pull Request，通过统一提示词对数据集进行标准化以简化评估流程，提高输出的机器兼容性，详见 [此处的 PR](https://github.com/open-thought/reasoning-gym/pull/120)。
   - 诸如 **Futoshiki puzzle dataset** 等更新旨在提供更简洁的解算器和改进的逻辑框架（如[此 PR](https://github.com/open-thought/reasoning-gym/pull/60) 所示），并建立了一种跨数据集平均分数的标准方法，以实现一致的报告。
- **DeepSeek 自动化 Kernel 生成**：NVIDIA 展示了使用 [DeepSeek-R1 模型](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940) 自动为 GPU 应用生成数值正确的 Kernel，并在推理期间对其进行优化。
   - 生成的 Kernel 在 KernelBench 基准测试的 Level-1 问题上达到了 100% 的准确率，在 Level-2 问题上达到了 96%，但也有人对该基准测试的饱和度表示担忧。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **GRPO 极大提升 Tulu 流水线性能**：在 Tulu 流水线中从 **PPO** 切换到 **GRPO** 带来了 **4倍** 的性能提升，正如 [Costa Huang 所宣布的](https://x.com/natolambert/status/1889730488199209393)，新的 **Llama-3.1-Tulu-3.1-8B** 模型在 MATH 和 GSM8K 基准测试中均展现出进步。
   - 这一转变标志着自去年秋天推出的早期模型以来的显著演进。
- **Anthropic 的 Claude 获得推理滑块**：根据 [Stephanie Palazzolo 的推文](https://x.com/steph_palazzolo/status/1890058003493343453)，Anthropic 即将推出的 Claude 模型将融合传统的 LLM 与推理 AI，允许开发者通过 **滑动刻度** 微调推理水平，在多个基准测试中可能超越 **OpenAI 的 o3-mini-high**。
   - 这代表了专为编程任务设计的模型训练和运行能力的转变。
- **DeepHermes-3 深度思考，但成本更高**：**Nous Research** 推出了 **DeepHermes-3**，这是一款集成了推理与语言处理的 LLM，可以切换长思维链（Chain of Thought）以提高准确性，但代价是更高的计算需求，正如 [Nous Research 的公告](https://x.com/NousResearch/status/1890148000204485088)所述。
   - 评估指标以及与 Tulu 模型的比较引发了关于基准测试分数差异的辩论，特别是遗漏了与官方 **8b distill release** 的对比，后者拥有更高的分数（**GPQA 约为 36-37%**，而 **r1-distill 约为 49%**）。
- **EnigmaEval 的谜题难倒了 AI**：Dan Hendrycks 发布了 **EnigmaEval**，这是一套复杂的推理挑战，AI 系统在其中表现挣扎，在普通谜题上得分低于 **10%**，在 MIT 级别的挑战上得分为 **0%**，详见 [Hendrycks 的推文](https://fxtwitter.com/DanHendrycks/status/1890091724594393140)。
   - 该评估旨在突破 AI 推理能力的极限。
- **OpenAI 释放 AGI 策略转变信号**：Sam Altman 暗示 OpenAI 目前的 Scaling Up 策略将不再足以实现 AGI，并建议在计划发布 **GPT-4.5** 和 **GPT-5** 时进行转型；[OpenAI 将整合其系统](https://x.com/sama/status/1889755723078443244?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)以提供更无缝的体验。
   - 他们还将解决社区对模型选择步骤的不满。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **关于 PPO-Clip 效用的辩论出现**：成员们重新讨论了将 [PPO-Clip](https://example.com) 应用于不同模型以生成 Rollouts 的方法，呼应了过去对话中的类似想法。
   - 一位成员根据之前的尝试，对这种方法的有效性表示怀疑。
- **Forgetting Transformer 性能调优**：围绕 **Forgetting Transformer** 展开了讨论，特别是将激活函数从 sigmoid 更改为 tanh 是否能对性能产生积极影响。
   - 对话还涉及引入负注意力权重，强调了注意力机制中潜在的复杂性。
- **Delphi 的引用变得更加容易**：成员们建议将来自论文和 [GitHub 页面](https://github.com/delphi) 的 **Delphi** 引用结合起来，以实现全面的归因。
   - 还有人建议在引用常见论文时使用 *arXiv 自动生成的 BibTeX 条目*，以实现标准化。
- **深入探讨长上下文模型的挑战**：成员们强调了对当前长上下文模型基准测试（如 **HashHop**）以及解决 1-NN 的迭代性质相关挑战的担忧。
   - 针对这些长上下文模型所宣称的理论可行性提出了质疑。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 用户未能保存进度**：一位用户报告称，由于在 **Stable Diffusion** 中未启用自动保存功能，导致生成的图像丢失，随后询问了图像恢复选项。
   - 该问题的解决方案涉及调试其使用的 Web UI 版本，以确定适当的保存设置。
- **Linux 系统导致 ComfyUI 用户出现 OOM 错误**：一位从 Windows 切换到 Pop Linux 的用户在 **ComfyUI** 中遇到了显存溢出 (**OOM**) 错误，尽管此前在 Windows 上运行正常。
   - 社区讨论了确认系统更新和推荐驱动程序的问题，并强调了不同操作系统之间依赖项的差异。
- **角色一致性挑战困扰 AI 模型**：一位用户在跨模型保持一致的角色设计方面遇到困难，引发了使用 **Loras** 以及 **FaceTools** 和 **Reactor** 等工具的建议。
   - 建议强调了应选择为特定任务设计的模型。
- **Stability 的创意放大器 (Creative Upscaler) 仍未发布**：用户询问了 **Stability 创意放大器** 的发布状态，并断言该工具尚未发布。
   - 讨论内容包括模型能力对内存和性能等硬件要求的限制。
- **账号共享受到质疑**：一位用户请求为即将开展的项目借用美国的 **Upwork** 账号，这引发了质疑。
   - 成员们对“借用”账号的可行性和潜在影响表示担忧。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 通过 GPT-4.5/5 统一模型**：根据其 [路线图更新](https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg)，OpenAI 正在通过在即将发布的 **GPT-4.5** 和 **GPT-5** 中整合 **O 系列模型**和工具来简化其产品线。
   - 此举旨在通过更紧密地集成所有工具和功能，为开发者和用户简化 AI 体验。
- **Anthropic 紧随其后推出推理 AI**：根据 [Stephanie Palazzolo 的推文](https://x.com/steph_palazzolo/status/1890058003493343453)，Anthropic 计划很快推出一款新的 **Claude** 模型，该模型结合了传统 LLM 能力与推理 AI，并可通过基于 Token 的滑动标尺进行控制。
   - 这呼应了 OpenAI 的做法，标志着将高级推理直接整合到 AI 模型中的行业趋势。
- **DeepHermes 3 推理 LLM 预览版发布**：Nous Research 发布了 **DeepHermes 3** 的预览版，这是一款将推理能力与传统响应功能相结合的 LLM，旨在提升性能，详情见 [HuggingFace](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview)。
   - 新模型寻求提供更高的准确性和功能性，作为 LLM 发展的一大进步。
- **Meta 使用 LLM 驱动的工具强化合规性**：Meta 推出了其**自动化合规强化 (ACH)** 工具，该工具利用基于 LLM 的测试生成技术，通过创建未检测到的故障进行测试，从而增强软件安全性，详见 [Meta 工程博客](https://engineering.fb.com/2025/02/05/security/revolutionizing-software-testing-llm-powered-bug-catchers-meta-ach/)。
   - 该工具旨在通过自动生成针对代码中特定故障条件的单元测试，来增强隐私合规性。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Hugging Face 发布 Smolagents**：Hugging Face 推出了 [smolagents](https://huggingface.co/spaces/m-ric/open_Deep-Research)，这是一个替代 `deep research` 的 Agent 框架，**6 个步骤**的处理时间约为 **13 秒**。
   - 用户可以修改原始代码，以便在本地服务器运行时延长执行时间，从而提供适应性。
- **Musk 声称 Grok 3 超越对手**：Elon Musk 宣布他的新 AI 聊天机器人 **Grok 3** 即将发布，并且在推理能力上*超越了现有模型*，预计在大约两周内推出。
   - 此前，Musk 的投资集团在与 OpenAI 持续的法律纠纷中，出价 **974 亿美元**收购其非营利资产。
- **Thomson Reuters 赢得里程碑式 AI 版权案**：Thomson Reuters 在针对 Ross Intelligence 的重大 [AI 版权案](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/)中获胜，法院判定该公司通过复制 Westlaw 的材料侵犯了其版权。
   - 美国巡回法院法官 Stephanos Bibas 驳回了 Ross 的所有辩护，称其*没有一个能站得住脚*。
- **强化学习（Reinforcement Learning）的创新方法**：讨论中出现了一种在新的 Reinforcement Learning 模型中使用 logits 作为中间表示的方法，强调了为了有效采样而延迟 Normalization 的重要性。
   - 该提案包括用 Energy-based methods 取代 softmax，并整合多目标训练范式，以实现更有效的模型性能。
- **新工具加速文献综述**：一位成员介绍了一个用于快速文献综述的新工具，可在 [Deep-Research-Arxiv](https://github.com/GitsSaikat/Deep-Research-Arxiv) 获取，强调了其简单性和可靠性。
   - 此外，还提到了一个 Hugging Face 应用，该应用旨在实现快速高效的文献综述。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 招聘开源工程师**：LlamaIndex 正在招聘一名全职开源工程师以增强其框架，欢迎对**开源、Python 和 AI** 充满热情的人士加入，详见其[职位公告](https://twitter.com/llama_index/status/1889724678970978588)。
   - 该职位提供了为 LlamaIndex 框架开发尖端功能的机会。
- **Nomic AI Embedding 模型助力 Agentic 工作流**：LlamaIndex 重点介绍了 **Nomic AI** 的最新研究，强调了 Embedding 模型在改进 **Agentic Document Workflows** 中的作用，并在[推文](https://twitter.com/llama_index/status/1889725475502665951)中进行了分享。
   - 社区期待该 Embedding 模型能带来更好的 AI 工作流集成。
- **LlamaIndex 与 Google Cloud**：LlamaIndex 推出了与 **Google Cloud** 数据库的集成，方便进行数据存储、Vector 管理、文档处理和聊天功能，详见[此帖](https://twitter.com/llama_index/status/1890109073615626388)。
   - 这些增强功能旨在利用云端能力，简化并保障数据访问的安全性。
- **微调（Fine Tuning）LLM 的讨论**：一位成员在 **#ai-discussion** 频道询问了关于对模型进行 Finetune 的合理理由。
   - 频道内没有提供更多额外信息来回答这个问题。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 创造了令人欲罢不能的 AI 播客**：用户们称赞 **NotebookLM** 能够快速将文字内容转化为播客，并强调了其在 Spotify 和 Substack 等平台上进行内容营销的潜力。
   - 爱好者们认为播客是一种内容营销工具，突出了其巨大的潜在受众覆盖范围和**创作的便捷性**。
- **通过 AI 播客开辟新收入渠道**：用户正在探索利用 AI 创作播客来产生收入，通过运行一个两人制 AI 播客，在专注于快速创作内容的同时，每月可赚取约 **$7,850**。
   - 他们声称，利用 [Substack](https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true.) 等工具，AI 驱动的播客创作可以带来 **300% 的自然触达率**和内容消耗增长。
- **AI 生成的播客主持人库引发关注**：社区成员讨论了创建一个 **AI 生成的播客主持人库**的潜力，展示多样化的主题和内容风格。
   - 爱好者们对协作和分享独特的 AI 生成音频体验感到兴奋，以增强社区参与度。
- **社区期待 NotebookLM 的多语言支持**：用户渴望 **NotebookLM** 能将其功能扩展到英语以外的其他语言，这凸显了全球范围内对易用的 AI 工具日益增长的兴趣。
   - 尽管可以调整语言设置，但音频功能目前仍仅限于**英语输出**，这引起了社区成员的沮丧。
- **探索 NotebookLM Plus 的功能与优势**：据一位成员介绍，**NotebookLM Plus** 提供了诸如交互式播客等对学生有益的功能，而这些功能在免费版中可能无法使用。
   - 另一位用户建议转向 **Google AI Premium** 以获取捆绑功能，这引发了关于 *“Google NotebookLM 真的很棒……”* 的讨论。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 发布新职位空缺**：一位成员分享了 **Modular** 发布了[新的职位招聘](https://discord.com/channels/1087530497313357884/1098713601386233997/)。
   - 这一消息引起了在 **Mojo** 生态系统中寻找机会的成员们的兴奋。
- **Mojo 和类型（Sum Types）引发辩论**：成员们将 **Mojo** 的和类型与 **Rust-like** 的和类型以及 **C-style** 的枚举进行了对比，指出 `Variant` 解决了许多需求，但参数化 Trait（parameterized traits）具有更高的优先级。
   - 一位用户使用 variant 模块实现的“权宜之计”的联合类型（hacky union type）凸显了当前 **Mojo** 实现的局限性。
- **Mojo 语境下的 ECS 定义得到澄清**：一位成员澄清了 **Mojo** 语境下 **ECS** 的定义，指出状态应该与行为分离，类似于 **Unity3D** 中的 **MonoBehavior** 模式。
   - 社区成员一致认为，一个遵循 **ECS 原理**的示例中，状态驻留在组件（components）中，而行为驻留在系统（systems）中。
- **使用 Unsafe Pointers 进行函数包装**：关于在 **Mojo** 的结构体中存储和管理函数的讨论引出了一个使用 `OpaquePointer` 安全处理函数引用的示例。
   - 交流中包含了完整的示例，并承认了在使用 `UnsafePointer` 时管理生命周期和内存的复杂性。
- **MAX 最小化了 CUDA 依赖**：**MAX** 仅在内存分配等核心功能上依赖 **CUDA driver**，从而最小化了对 **CUDA** 的依赖。
   - 一位成员指出，MAX 在使用 **GPU**（尤其是 **NVIDIA** 硬件）时采取了精简的方法，以实现最佳性能。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 客户端 Bug 困扰用户**：成员们分享了使用 MCP 客户端的经验，重点推荐了 [wong2/mcp-cli](https://github.com/wong2/mcp-cli) 的开箱即用功能，同时指出 *客户端 Bug 频出是一个普遍现象*。
   - 开发者们讨论了尝试绕过现有工具限制的方法。
- **OpenAI 模型进入 MCP 领域**：新用户对 MCP 的功能表示兴奋，并询问 **Claude** 之外的模型是否会支持 MCP。
   - 有人指出，虽然 MCP 与 OpenAI 模型兼容，但像 **Open WebUI** 这样的项目可能不会优先考虑它。
- **Claude Desktop 用户遭遇使用限制**：用户反映 **Claude Desktop** 的使用限制非常麻烦，并建议 Glama 的服务可以作为一种变通方案。
   - 一位成员强调了这些限制如何影响他们的使用场景，并指出 **Glama** 提供了更便宜、更快速的替代方案。
- **Glama Gateway 挑战 OpenRouter**：成员们将 [Glama 的 Gateway](https://glama.ai/gateway) 与 **OpenRouter** 进行了对比，指出 Glama 的成本更低且有隐私保证。
   - 虽然 **Glama** 支持的模型较少，但因其快速和可靠而受到称赞，使其成为某些应用的稳健选择。
- **Open WebUI 引起关注**：多位用户对 **Open WebUI** 表示好奇，提到了其丰富的功能集以及最近关于 MCP 支持的路线图更新。
   - 成员们对其易用性给出了正面评价，并希望能够完全从 **Claude Desktop** 迁移出来。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **DeepSeek-R1 自动化 GPU Kernel 生成**：一篇博文强调了 **DeepSeek-R1 模型** 在改进 GPU Kernel 生成方面的应用，通过使用 *Test-time Scaling* 在推理期间分配更多计算资源来提升模型性能，链接见 [NVIDIA 技术博客](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)。
   - 文章指出，AI 可以通过在选择最佳方案前评估多种结果来**有效地制定策略**，这模仿了人类解决问题的过程。
- **Tinygrad Graph Rewrite Bug 令成员沮丧**：成员们调查了由于一个潜在 Bug 导致的 CI 失败，该 Bug 中**错误的缩进**导致 `bottom_up_rewrite` 从 `RewriteContext` 中被移除。
   - 还考虑了 **Graph** 处理中更深层次的问题，例如错误的重写规则或顺序。
- **Windows CI Backend 变量传递修复**：一位成员注意到 **Windows CI** 未能在步骤之间传递 Backend 环境变量，并提交了一个 [GitHub Pull Request](https://github.com/tinygrad/tinygrad/pull/9047) 来解决此问题。
   - 该 PR 通过在 CI 执行期间利用 `$GITHUB_ENV` 来确保 Backend 变量持久化。
- **Tinygrad 承诺比 PyTorch 更高的性能收益**：用户讨论了从 **PyTorch** 切换到 **tinygrad** 的利弊，考虑学习曲线是否值得，特别是在**成本效率**或掌握**底层原理**方面。
   - 与 PyTorch 相比，使用 tinygrad 最终可能会带来**更便宜的硬件**支持或**更快的模型**，提供优化和资源管理方面的优势。
- **社区警告不要提交 AI 生成的代码**：成员们强调在提交前要检查代码 Diff，指出微小的空格变化可能导致 PR 被关闭，并敦促不要直接提交由 **AI** 生成的代码。
   - 社区建议使用 AI 进行**头脑风暴和反馈**，尊重成员的时间，并鼓励原创贡献。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents Hackathon 获胜者公布**：根据 [Dawn Song 的推文](https://x.com/dawnsongtweets/status/1889686697564315963)，**LLM Agents MOOC Hackathon** 公布了来自 **127 个国家**的 **3,000 名参与者**中的获胜团队，彰显了全球 AI 社区的极高参与度。
   - 顶尖代表机构包括 **UC Berkeley**、**UIUC**、**Stanford**、**Amazon**、**Microsoft** 和 **Samsung**，完整的提交作品可在 [Hackathon 网站](https://rdi.berkeley.edu/llm-agents-hackathon/)上查看。
- **2025 春季 MOOC 正式启动**：**2025 春季 MOOC** 正式开课，面向更广泛的 AI 社区，并邀请大家转发 [Dawn Song 教授的公告](https://x.com/dawnsongtweets/status/1889355520294944829)。该课程基于 2024 秋季课程的成功经验，当时有超过 **1.5 万名注册学员**。
   - 更新后的课程涵盖了 **Reasoning & Planning**、**Multimodal Agents** 以及 **AI for Mathematics and Theorem Proving** 等高级主题，并邀请所有人参加**每周一太平洋时间下午 4:10 的直播课程**。
- **MOOC 证书发放问题**：多位用户反映未收到之前课程的证书，一名学生请求重发，另一名学生需要帮助查找，但处理可能需要等到周末。
   - Tara 指出 **Ninja Certification** 没有正式评分，并建议在指定频道中针对其他学生的提交内容测试 Prompt。此外，处理证书需要提交 **declaration form**（声明表单）。
- **新 AI/ML 入门者寻求指导**：一位新成员表达了希望在 **AI/ML 领域**入门以及了解模型训练技术的意愿，但频道内尚未提供相关指导。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 支持分布式推理**：用户现在可以使用 **Torchtune** 在多个 GPU 上运行分布式推理，实现细节请查看 [GitHub recipe](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py)。
   - 使用带有 **vLLM** 的保存模型可获得额外的速度提升。
- **Torchtune 仍缺少 Docker 镜像**：目前 **Torchtune** 尚无可用的 **Docker 镜像**，这导致部分用户安装困难。
   - 唯一的安装方式是参考 GitHub 上的[安装指南](https://github.com/pytorch/torchtune?tab=readme-ov-file#installation)。
- **Checkpointing 分支通过测试**：新的 **checkpointing 分支**已成功克隆，且在初步测试中表现良好。
   - 计划进一步测试 **recipe_state.pt** 功能，并可能更新关于恢复训练的文档。
- **团队积极协作处理 Checkpointing PR**：团队成员对 **checkpointing PR** 表现出极大的热情和积极的协作态度。
   - 这突显了团队对于改进 Checkpointing 流程的共同承诺。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **NVIDIA 利用 DeepSeek-R1 扩展推理**：NVIDIA 的实验展示了利用 **DeepSeek-R1** 模型进行“推理时扩展”（inference-time scaling）以优化 GPU attention kernels，通过评估多种结果来实现更好的问题解决能力，详见其 [博客文章](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/)。
   - 这种技术在推理期间分配额外的计算资源，模拟了人类解决问题的策略。
- **LangChain 与 DSPy 的选择权衡**：关于何时选择 **LangChain** 而非 **DSPy** 的讨论强调两者用途不同。一位成员建议，如果 **DSPy** 的学习曲线太陡峭，应优先考虑成熟的 **LangChain** 方法。
   - 对话强调了根据项目需求评估采用新框架复杂性的重要性。
- **DSPy 2.6 更新日志揭晓**：一位用户询问了 **DSPy 2.6** 的更新日志，特别是关于 Signatures 的 **instructions** 与旧版本相比的有效性。
   - 澄清显示这些指令自 2022 年起就已存在，详细的更新日志可在 **GitHub** 上查看，尽管未提供具体链接。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 接入 Deepseek R1**：**GPT4All v3.9.0** 允许用户在本地下载并运行 **Deepseek R1**，重点在于离线功能。
   - 然而，在本地运行全量模型非常困难，目前似乎仅限于 **13B** 参数模型等较小变体，其性能表现不如全量版本。
- **LocalDocs 困扰用户**：有用户报告 **LocalDocs** 功能较为基础，处理 **TXT** 文档时提供准确结果的概率仅约 **50%**。
   - 用户怀疑这些限制是由于使用了 **Meta-Llama-3-8b instruct** 模型或设置不当造成的。
- **NOIMC v2 等待实现**：尽管已确认发布，但成员们对 **NOIMC v2** 模型尚未得到妥善实现感到困惑。
   - 分享了 [nomic-embed-text-v2-moe 模型](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) 的链接，强调了其多语言性能和能力。
- **多语言 Embeddings 支持 100 种语言**：**nomic-embed-text-v2-moe** 模型支持约 **100 种语言**，相对于同等规模的模型具有高性能，且具备灵活的 Embedding 维度，并完全开源。
   - 分享了其 [代码](https://github.com/nomic-ai/contrastors)。
- **社区寻求将 Prompt 转换为代码的工具**：一位用户正在寻求关于将 **英语 Prompt** 转换为可用代码的工具建议。
   - 需要具体的建议。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 混乱的评分**：一位用户发现 **Rerank 3.5** 在不同批次处理文档时会给出不同的评分，这超出了他们的预期，因为它是 cross-encoder。
   - 这种评分的可变性被描述为“反直觉的”。
- **Cohere 在 Salesforce 的 BYOLLM 中遇到困难**：一位成员询问如何将 **Cohere** 作为 LLM 与 Salesforce 的 BYOLLM 开放连接器配合使用，并提到了 [api.cohere.ai](https://api.cohere.ai/v2/chat) 聊天端点的问题。
   - 他们正尝试按照 Salesforce 支持人员的建议，创建一个 https REST 服务来调用 Cohere 的聊天 API。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1339280039195181127)** (956 条消息🔥🔥🔥): 

> `Unsloth 中的 GRPO 更新, 模型的 VRAM 需求, DeepSeek 的动态量化, 模型合并及其影响, LLM 的微调策略` 

- **Unsloth 中的 GRPO 更新**：用户分享了关于 Unsloth 新 GRPO 更新的发现，指出其在各种强化学习实验中的有效性。
   - 一位用户强调了更好管理 VRAM 的必要性，因为他们在不同的训练设置中遇到了内存使用不匹配和 OOM 错误。
- **模型的 VRAM 需求**：关于各种模型 VRAM 需求的讨论提到，在运行带有一定上下文的模型时，大约需要模型权重 1.5 倍的 VRAM。
   - 该估算旨在帮助用户根据上下文长度和模型大小评估其硬件能力。
- **DeepSeek 的动态量化**：解释了动态量化技术，特别是其在 DeepSeek 模型中的实现方式及其优势。
   - 用户分享了动态量化如何帮助减少内存占用同时保持性能的见解，尽管目前主要适用于特定模型。
- **模型合并及其影响**：关于模型合并的伦理和可行性的对话强调了对原作者贡献和署名的担忧。
   - 虽然合并可以增强能力，但也有关于在开源社区中补偿原模型创建者的讨论。
- **LLM 的微调策略**：几位用户讨论了不同的微调策略，包括使用合成数据和 R1 蒸馏过程来提升模型性能。
   - 对话指出，微调中的实践经验和共享发现可以显著促进更好的训练方法论。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/collections/unsloth/deepseek-r1-all-versions-678e1c48f5d2fce87892ace5">DeepSeek R1 (所有版本) - unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/agentica-org/DeepScaleR-Preview-Dataset">agentica-org/DeepScaleR-Preview-Dataset · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2501.06252">Transformer-Squared: 自适应 LLMs</a>：自适应大语言模型 (LLMs) 旨在解决传统微调方法带来的挑战，这些方法通常计算密集，且在处理多样化任务时能力较为静态...</li><li><a href="https://github.com/agentica-project/deepscaler/tree/main/scripts/train">deepscaler/scripts/train (main 分支) · agentica-project/deepscaler</a>：让 LLMs 的强化学习平民化。通过在 GitHub 上创建账号，为 agentica-project/deepscaler 的开发做出贡献。</li><li><a href="https://docs.jax.dev/en/latest/quickstart.html">快速入门 &#8212; JAX 文档</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1889726411478278183">Unsloth AI (@UnslothAI) 的推文</a>：使用我们的免费 Notebook，利用 DeepSeek 的 GRPO 算法训练你自己的推理 LLM！你将让 Llama 3.1 (8B) 具备思维链 (chain-of-thought) 能力。Unsloth 使 GRPO 的 VRAM 占用减少了 80%。指南：https:...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth 需求 | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF/tree/main">bartowski/DeepSeek-R1-Distill-Qwen-7B-GGUF (main 分支)</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/open-r1/OpenR1-Math-Raw">open-r1/OpenR1-Math-Raw · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benchmarks">Unsloth 基准测试 | Unsloth 文档</a>：想知道 Unsloth 有多快吗？</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>：又称持续微调。Unsloth 允许你进行持续预训练，使模型能够学习一种新语言。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=v2GniOB2D_U">创建用于微调 LLMs 的训练数据</a>：🚀 掌握 LLM 微调：从 PDF 到 JSONL 文件 🚀 欢迎来到 APC Mastery Path！在本综合教程中，我们将深入探讨创建...的过程</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">量化 (Quantization)</a>：未找到描述</li><li><a href="https://huggingface.co/agentica-org/DeepScaleR-1.5B-Preview">agentica-org/DeepScaleR-1.5B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`">CUDA 语义 &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF">unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/allenai/s2orc?tab=readme-ov-file#download-instructions">GitHub - allenai/s2orc: S2ORC: Semantic Scholar 开放研究语料库：https://www.aclweb.org/anthology/2020.acl-main.447/</a>：S2ORC: Semantic Scholar 开放研究语料库：https://www.aclweb.org/anthology/2020.acl-main.447/ - allenai/s2orc</li><li><a href="https://docs.unsloth.ai/basics/unsloth-benc">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/SakanaAI/self-adaptive-llms/tree/main">GitHub - SakanaAI/self-adaptive-llms：一个自适应框架 🐙，可实时让 LLMs 适应未见过的任务！</a>：一个自适应框架 🐙，可实时让 LLMs 适应未见过的任务！ - SakanaAI/self-adaptive-llms</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/fused_linear_cross_entropy.py">Liger-Kernel/src/liger_kernel/ops/fused_linear_cross_entropy.py (main 分支) · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton 内核。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/apple/ml-cross-entropy/blob/main/cut_cross_entropy/cce_lse_forward.py#L79">ml-cross-entropy/cut_cross_entropy/cce_lse_forward.py (main 分支) · apple/ml-cross-entropy</a>：通过在 GitHub 上创建账号，为 apple/ml-cross-entropy 的开发做出贡献。</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/ops/triton/cross_entropy.py">flash-attention/flash_attn/ops/triton/cross_entropy.py (main 分支) · Dao-AILab/flash-attention</a>：快速且内存高效的精确 a

ttention。通过在 GitHub 上创建账号，为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L25>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/cross_entropy.py#L264>">Liger-Kernel/src/liger_kernel/ops/cross_entropy.py at main · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1339514787897282560)** (3 条消息): 

> `Unsloth 重新介绍，Wendel 的 AI 推荐，Deepseek 的发布` 


- **Unsloth 带着新特性重新推出**：标题为 ["Re-introducing Unsloth"](https://www.youtube.com/shorts/6VDpjGFivYw) 的 YouTube 视频详细介绍了各项增强功能，允许用户使用 Unsloth 更快地 finetune 和训练自己的 LLMs。
   - 描述中强调了通过 Unsloth 的新功能使 LLM 训练过程变得更加简单，并鼓励用户采用。
- **Wendel 推荐 Unsloth**：在视频 ["Embrace the Coming AI Revolution with Safe Local AI!"](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) 中，Wendel 重点介绍了几个以 Unsloth 为特色的创新。
   - *直接引用 Wendel 的话*，他讨论了 Unsloth 在当前 AI 领域产生的重大影响及其在未来进步中的潜力。
- **AI 工业革命随 Deepseek 开启**：Wendel 讨论了 [Deepseek 的发布](https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ) 如何撼动 AI 世界，并标志着 AI 工业革命的开始。
   - 他敦促观众通过利用处于领先地位的前沿 AI 工具（如 Unsloth）来拥抱这些变化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/rPf5GCQBNn4?si=S7UNe8xboIwqQLuQ">Embrace the Coming AI Revolution with Safe Local AI!</a>: Deepseek 的发布撼动了 AI 世界，我们正处于 AI 工业革命的边缘！Wendell 为你详细讲解如何应对……</li><li><a href="https://www.youtube.com/shorts/6VDpjGFivYw">Re-introducing Unsloth  #ai #llm</a>: 重新介绍 Unsloth，轻松 finetune 和训练 LLMs，使用 Unsloth 获得更快速度 https://unsloth.ai/
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1339296014984544377)** (108 条消息🔥🔥): 

> `Llama 3.2 问题、GRPO 训练挑战、结构化数据模型、在本地模型中使用 Unsloth、模型配置与安装` 


- **Llama 3.2 与 BitsandBytes 错误排查**：用户在 WSL 上运行 **Llama 3.2 11B** 时遇到 BitsandBytes 相关问题，建议如果问题持续存在，请尝试创建新的 conda 环境。
   - 部分用户在离线状态下遇到加载时间过长的问题，用户认为这可能源于与 Hugging Face 的连接错误。
- **使用 Llama 模型进行 GRPO 训练**：关于在 **GRPO 训练**期间获取预期奖励的问题被提出，用户观察到性能指标陷入平台期以及生成长度出现意外变化。
   - 一位用户报告称通过修改正则表达式获得了更好的训练结果，但指标不一致的问题仍然存在。
- **结构化数据提取的挑战**：关于针对结构化数据微调 Llama 模型的讨论表明，实现所需的输出格式存在困难，有人指出 XML schema 提取值的准确率较低。
   - 用户建议对输出进行评分，而不是仅仅依赖模型的推理能力来改进结果。
- **使用本地缓存加载 FastLanguageModel**：用户分享了从本地缓存加载 **FastLanguageModel** 的方法，强调在导入前进行环境配置以确保顺利执行。
   - 一位用户在生成输出时遇到的 attention_mask 错误引起了关注，提醒需要确保预先正确定义所有张量（tensor）组件。
- **模型安装与连接问题**：多位用户在从 Hugging Face 下载大型模型文件时遇到困难，提示连接错误并建议将手动下载作为替代方案。
   - 用户对 HF 发布的模型与 Unsloth 版本之间的差异感到好奇，特别是关于影响训练的仓库配置。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb),">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/errors#evaluation-loop-also-oom-or-crashing">错误 | Unsloth 文档</a>: 要修复设置中的任何错误，请参阅下文：</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://unsloth.ai/blog/r1-reasoning">在本地训练你自己的 R1 推理模型 (GRPO)</a>: 你现在可以使用 Unsloth 在本地 100% 复现你自己的 DeepSeek-R1 推理模型。使用 GRPO。开源、免费且对初学者友好。</li><li><a href="https://anotherwrapper.com/open-deep-research">Open Deep Research - 开源 AI 研究助手</a>: 发现 OpenAI Deep Research、Google Gemini 和 Anthropic Claude 的开源替代方案。由 GPT-4o-mini 提供支持，该工具提供全面的市场分析和...</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/mistral-benchmark#Breakdown">Unsloth 更新：Mistral 支持及更多</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了滑动窗口注意力（sliding window attention）、初步的 Windows 和 DPO 支持，以及...</li><li><a href="https://github.com/instructor-ai/instructor">GitHub - instructor-ai/instructor: LLM 的结构化输出</a>: LLM 的结构化输出。通过在 GitHub 上创建账户为 instructor-ai/instructor 的开发做出贡献。</li><li><a href="https://docs.unsloth.ai/basics/datasets-101#getting-started">数据集 101 | Unsloth 文档</a>: 学习创建用于微调的数据集的所有要点！</li><li><a href="https://huggingface.co/docs/transformers/main/en/chat_templating">聊天模板</a>: 未找到描述</li><li><a href="https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/">Llama 3.1 | 模型卡片与提示词格式</a>: Llama 3.1 - 最强大的开源模型。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1339503091367809035)** (2 条消息): 

> `Rombo-LLM-V3.0-Qwen-32b, DeepSeek-R1 Performance, Llama 3.1B Fine-tuning, Resources for Training Reasoning Models` 


- **Rombo-LLM-V3.0-Qwen-32b 发布了！**：新模型 **Rombo-LLM-V3.0-Qwen-32b** 已发布，在各项任务中展现出令人印象深刻的性能。详情请见 [Reddit 帖子](https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/)。
   - 在 **Patreon** 上支持开发者的工作，每月仅需 5 美元即可为未来的模型投票并访问私有仓库。
- **DeepSeek-R1 在复杂任务中表现卓越**：DeepSeek-R1 扩展了其处理数学和编程任务的能力，展示了显著的多功能性。重点讨论了它与标准模型的不同运作方式，包括其对 **real RL** 方法的使用。
   - 详尽的 [视频](https://www.youtube.com/live/bbFEYPx9Hpo?si=-YvREf39uO10vwxy) 和 [幻灯片](https://www.canva.com/design/DAGe5nXTLas/HKlqBg40KhNHZizSIo_Uuw/edit?utm_content=DAGe5nXTLas&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 记录了昨天训练环节中讨论的见解。
- **利用 GRPO + LoRA 提升 Llama 3.1B**：社区探索了如何结合 GRPO 与 LoRA/QLoRA 技术来增强 **Llama 3.1B**，使其性能与更高阶的模型并驾齐驱。这一性能对比涵盖了从基础模型到经过微调的推理模型等多种配置。
   - 参与者展示了不同方法之间的差异，强调了结合 **CoT** 提示与高级微调以提升推理能力的优势。
- **获取训练资源**：分享了宝贵的资源，包括用于动手训练和探索推理模型的 [Google Colab Notebook](https://colab.research.google.com/drive/1iLTEK_KD-ZfzRQhTIvMgBqIjPMWijnq3?usp=sharing)。强调了 **DeepSeekMath** 的贡献，因为它对于理解统一范式具有重要意义。
   - 查看[此处](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)的图解指南，进一步了解 DeepSeek-R1 的架构和功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://www.youtube.com/live/bbFEYPx9Hpo?si=-YvREf39uO10vwxy)">Deepseek-R1 &amp; 训练你自己的推理模型</a>: DeepSeek 正在席卷全球应用商店，但其最新突破背后的原因是什么？加入我们，深入探讨 DeepSeek-R1，这是首个大型推理模型 (LR...</li><li><a href="https://www.canva.com/design/DAGe5nXTLas/HKlqBg40KhNHZizSIo_Uuw/edit?utm_content=DAGe5nXTLas&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)">惊人简单的图形设计软件 – Canva</a>: 惊人简单的图形设计软件 – Canva</li><li><a href="https://colab.research.google.com/drive/1iLTEK_KD-ZfzRQhTIvMgBqIjPMWijnq3?usp=sharing)">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1339284424004468901)** (6 条消息): 

> `Transformer 在表格数据上的表现，微调 Mistral 模型，LoRA Checkpoint 的推理指令，Reasoning Agent 的开发，针对 VLM 的 Lavender 方法` 


- **Transformer 在表格数据上表现不佳**：有人对 **Transformer** 在**表格数值数据**上的糟糕表现表示担忧，无论使用何种格式，并指出每个 LLM 计算总和及平均值的方式都不同。
   - *现有的每一个 LLM* 似乎都以不同的方式处理表格数据，这表明该架构在处理此类信息方面存在根本性问题。
- **微调 Mistral Small 的建议**：有人询问了为文本拟人化项目微调 **Mistral small** 模型时建议的样本数量，表示他们是该流程的新手。
   - 讨论了微调策略，重点关注实现有效性能所需的训练样本数量。
- **LoRA Checkpoint 及使用说明**：一位成员提供了一个 [LoRA Checkpoint 链接](https://huggingface.co/sathvikask/r1-1.5b-RL-gsm8k)，并包含了通过 [推理指南](https://github.com/sathvikask0/r1-distilled-RL) 使用该模型的步骤。
   - 该模型目前无法通过受支持的 Inference Provider 获取，且训练在 **418 步**时停止，需要对其性能进行进一步分析。
- **对 Reasoning Agent 的兴趣**：有人提出了是否有人正在研究 **Reasoning Agent** 的问题，突显了社区对高级 AI 应用的关注。
   - 这反映了人们对开发能够有效执行逻辑推理任务的模型的兴趣日益浓厚。
- **Lavender 方法增强视觉语言模型**：**Lavender** 方法被介绍为一种简单的监督微调技术，它利用 **Stable Diffusion** 提高了视觉语言模型 (VLM) 的性能。
   - 该方法通过更好的文本-视觉注意力对齐，实现了显著的性能提升，包括在 20 个任务上提升了 **+30%**，在 OOD WorldMedQA 上提升了 **+68%**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://astrazeneca.github.io/vlm/">Lavender: Diffusion Instruction Tuning</a>: 未找到描述</li><li><a href="https://huggingface.co/sathvikask/r1-1.5b-RL-gsm8k">sathvikask/r1-1.5b-RL-gsm8k · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/sathvikask0/r1-distilled-RL">GitHub - sathvikask0/r1-distilled-RL</a>: 通过在 GitHub 上创建账户，为 sathvikask0/r1-distilled-RL 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1339281292256350363)** (51 条消息🔥): 

> `Agent 模板问题、Embedding 模型与性能、深度强化学习课程、LLama 垃圾信息行为、ViT 投影维度` 


- **首次 Agent 模板执行困难**：一位用户表示在创建新副本后运行“First Agent 模板”时遇到困难，并寻求执行步骤方面的指导。
   - 其他成员引导他们前往特定的 Discord 频道寻求帮助。
- **关于 Embedding 模型大小的讨论**：用户质疑为什么超过 7B 的 Embedding 模型很少，评论指出 Embedding 模型通常被设计为低成本且快速。
   - 一位成员指出，较大的模型性能未必优于较小的模型，因为它们往往会对基准测试（benchmarks）产生过拟合。
- **深度强化学习课程参与**：一位用户询问是否有关于深度强化学习课程的聊天或频道，并被引导至相关的 Discord 线程。
   - 进一步的信息强调了参与者可获得的课程内容和路径，包括一个用于讨论的 Discord 服务器。
- **LLama 模型垃圾信息问题**：有用户反映 LLama 模型会大量发送“!”，用户建议这可能源于代码中的配置错误。
   - 一位成员提供了一个 Discord 消息链接，其中可能包含处理此问题的更多见解。
- **ViT 投影维度查询**：一位用户寻求关于 ViT 合适投影维度的建议，询问该维度应该大于还是小于 patch 维度。
   - 他们参考了原始论文对维度的选择，并分享了使用较小值时好坏参半的结果，正在寻找公认的处理方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/agents-course/unit0/introduction">Welcome to the 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/deep-rl-course">Welcome to the 🤗 Deep Reinforcement Learning Course - Hugging Face Deep RL Course</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/v0.28.1/guides/download#download-an-entire-repos">Download files from the Hub</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/deep-rl-course/en/unit0/introduction">Welcome to the 🤗 Deep Reinforcement Learning Course - Hugging Face Deep RL Course</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/huggingface_hub/v0.28.1/guides/download#download-an-entire-repository">Download files from the Hub</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>: 未找到描述</li><li><a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/murilofarias10/Python/blob/main/VAM_AI/Course_HF/New_PROMPT_AI.ipynb">Python/VAM_AI/Course_HF/New_PROMPT_AI.ipynb at main · murilofarias10/Python</a>: 我的 Python 项目。通过在 GitHub 上创建账号为 murilofarias10/Python 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding)">LM Studio Docs | LM Studio Docs</a>: 了解如何使用 LM Studio 在本地运行 Llama、DeepSeek、Phi 和其他 LLM。</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta Releases</a>: LM Studio 的 Beta 和发行候选版本
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1339331541997719704)** (10 条消息🔥): 

> `Tensor Parallelism 中的重叠通信 (Overlapping Communication)、Agents 课程、模糊聚类 (Fuzzy Clustering)、特殊 Token 的使用、重复的重要性` 


- **理解 Tensor Parallelism 中的重叠通信**：一位成员分享了关于 **Tensor Parallelism 中重叠通信 (overlapping communication)** 的见解，强调了其在处理效率方面的重要性。
   - 讨论中提出了关于 Tensor Parallelism **应用场景**的问题，引发了进一步的探究和讨论。
- **Agents 课程中的疑问**：另一位成员正在学习 **Agents 课程的 Unit 1**，并对 Agent 术语中的 *reasoning* 和 *reflection* 感到困惑。
   - 他们寻求澄清这些术语是否可以**互换使用**，这似乎引起了其他有类似困惑的成员的共鸣。
- **流数据上的模糊聚类**：一位成员正在深入研究针对流数据的 **fuzzy clustering 技术**，旨在探索其有效性和应用。
   - 该话题突显了对各种数据处理流中相关的**数据驱动方法**的兴趣。
- **重复的作用**：在一次轻松的交流中，一位成员指出重复是关键，并引用了关于使用 ♻️ 表情符号表示循环操作的讨论。
   - “*重复至关重要*”得到了成员们的呼应，强调了其在学习中的重要性。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1339310344295944213)** (14 条消息🔥): 

> `QR Code DOOM, AI 模型训练助手, 新 LLM 发布, Joker 玩笑生成器, 深度研究员系统` 


- **DOOM 在二维码内运行！**: 一名成员成功创建了一款名为 **The Backdooms** 的受 DOOM 启发的自制游戏，它完全容纳在单个二维码中，大小保持在 **2.4kb** 以下。
   - 该项目以 **MIT 许可证** 开源，允许他人在 [此处](https://github.com/Kuberwastaken/backdooms) 进行实验。
- **介绍 Steev，你的 AI 训练助手**: 一个团队推出了 Steev，旨在简化 AI 模型训练流程，从而消除过程中所需的持续监督。
   - 他们邀请感兴趣的用户在 [Steev.io](https://www.steev.io/) 探索该应用。
- **用于编程任务的 Rombo-LLM V3.0 发布**: 一个名为 **Rombo-LLM-V3.0-Qwen-32b** 的新模型已发布，并在 Reddit 帖子里详细展示了其功能。
   - 该模型在编程和数学方面表现出色，**Q8_0 量化** 进一步增强了其能力。
- **玩笑生成器在 Hugging Face 上线**: 一名成员介绍了一个 **Jokes Generator**，它通过一个用户友好的 Gradio 聊天界面从 Joker Rest API 获取玩笑。
   - 可以通过 [此链接](https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI) 在他们的 Hugging Face Space 体验该应用。
- **深度研究员（Deep Researchers）概念讨论**: 一位用户分享了他们使用两个不同的 LLM 进行来回讨论，以提取对特定主题更深层次见解的方法。
   - 该方法利用一个较大的模型将发现结果汇编成一份连贯的研究报告，更多详情可在其 [GitHub](https://github.com/solarkyle/Adversarial-Researchers/tree/main) 查看。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/xyizko/xo-JokeGen-NoAI">Xo JokeGen NoAI - a Hugging Face Space by xyizko</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/open-neo/kyro-n1-67ab2e7bbc76a9aab3030c21">Kyro-n1 - a open-neo Collection</a>: 未找到描述</li><li><a href="https://pypi.org/project/llm-wrapper-cli/.">Client Challenge</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/.kkrieger">.kkrieger - Wikipedia</a>: 未找到描述</li><li><a href="https://www.steev.io/">steev</a>: 用于 ML 研究的实验性 AI Agent。</li><li><a href="https://github.com/solarkyle/Adversarial-Researchers/tree/main">GitHub - solarkyle/Adversarial-Researchers</a>: 通过在 GitHub 上创建账号来为 solarkyle/Adversarial-Researchers 的开发做出贡献。</li><li><a href="https://github.com/karam-koujan/mini-pytorch">GitHub - karam-koujan/mini-pytorch</a>: 通过在 GitHub 上创建账号来为 karam-koujan/mini-pytorch 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/Kuberwastaken/backdooms">GitHub - Kuberwastaken/backdooms: A self-contained game that fits inside a QR code inspired by DOOM 1993 and The Backrooms</a>: 一个受 DOOM 1993 和 The Backrooms 启发，能装进二维码的自包含游戏 - Kuberwastaken/backdooms</li><li><a href="https://kuberwastaken.github.io/blog/Projects/How-I-Managed-To-Get-Doom-In-A-QR-Code">How I Managed To Get Doom In A QR Code</a>: 是的，这确实是整个游戏。如果你想玩，扫描并开始。DOOM 以其自 1993 年以来的各种移植版本而闻名，到处都能运行，甚至有“它能运行...”的梗。</li><li><a href="https://www.reddit.com/">reddit</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1339634403726463067)** (10 messages🔥): 

> `Bingbin Liu 演讲，技术困难，会议录制` 


- **Bingbin Liu 分享关于 Attention Glitches 的见解**：由 Bingbin Liu 主讲的关于论文 *Exposing Attention Glitches with Flip-Flop Language Modeling* 的 Reading Group 会议即将开始，[论文链接在此](https://proceedings.neurips.cc/paper_files/paper/2023/file/510ad3018bbdc5b6e3b10646e2e35771-Paper-Conference.pdf)。
   - 鼓励参与者加入在 Zoom 上举行的讨论并在会议期间提问。
- **会议期间出现技术困难**：成员们对今天遇到的 **Technical Difficulties** 表示歉意，这导致会议转向了 Zoom 会议。
   - 提供的 [Zoom 链接](https://mcgill.zoom.us/j/85033055096) 被多次分享，以确保每个人都能加入。
- **会议将为缺席成员录制**：与会者得到保证，会议将为不方便使用 Zoom 或无法参加直播的人进行 **录制**。
   - 这确保了尽管存在技术问题，每个人都能获得分享的见解。
- **对演讲表示感谢**：与会者对演讲表示赞赏，指出背景信息增加了对论文理解的深度。
   - 一位成员特别感谢了 Bingbin Liu 和另一位演讲者的精彩分享。



**提到的链接**：<a href="https://mcgill.zoom.us/j/85033055096">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1339690447219720265)** (1 messages): 

> `Canny 边缘滤波器，Sobel 滤波器，结合 Diffusion Model 的 ControlNet` 


- **Canny 和 Sobel 滤波器作为预处理工具**：在集成 Machine Learning (ML) 模型之前，使用 **Canny edge** 或 **Sobel 滤波器** 是一个关键方法，因为并非每个过程都需要 ML 组件。
   - 这些滤波器可以作为一个重要的 **Pre-processing Stage**（预处理阶段），辅助 ML 学习不同的下游任务。
- **ControlNet 利用边缘过滤**：结合 Diffusion Model 的 **ControlNet** 采用 **Canny edge 过滤后的图像** 来生成与原始图像保持 **Structural Consistency**（结构一致性）的输出。
   - 这种方法说明了传统的图像处理技术如何增强现代 ML 模型的能力。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1339312167215956062)** (8 messages🔥): 

> `预训练模型行为，Tool Messages 的 Tokenization，使用 LoRA 进行 Fine-tuning，End Token 生成问题，训练技术` 


- **预训练模型预测下一个 Token**：Pretrained Models 实际上是根据其训练语料库来 **预测下一个 Token**。在指令数据集上进行 Fine-tuning 可以让模型优化其回答并减少幻觉（Hallucinations）。
   - 一位成员指出，这种训练有助于模型理解 **'user'** 和 **'assistant'** 等角色。
- **对 Tool Message Tokenization 的好奇**：一位成员询问 **Tool Messages** 是否像 System 和 Human/Assistant 消息一样被 Tokenized 并发送给 Transformer。另一位成员推测模型是基于 Tool Messages 的响应进行推理的。
   - 这表明理解 Tool Messages 的处理方式对于模型功能至关重要。
- **使用 LoRA 对 Qwen 模型进行 Fine-tuning**：在用 LoRA 对 **Qwen** 模型进行 Fine-tuning 后，一位成员发现预合并（Pre-merged）模型在执行指令方面表现更好，但难以生成 End Tokens。合并后的模型产生了乱码响应，引发了关于权重合并过程的疑问。
   - 成员们担心训练期间低质量的数据可能会导致这些问题，从而影响模型正确结束回答的能力。
- **End Token 生成解释**：一位成员指出，只有当 End Tokens 是最可能的下一个 Token 时才会生成，这表明在防止无限循环方面存在挑战。他们寻求关于如何有效教导模型识别 End Tokens 的明确方法。
   - 另一位成员建议对指令/回答对使用 **Supervised Fine-tuning (SFT)**，这可以帮助模型学习回答应该在哪里结束。
- **理解训练技术**：当 Learning Rate 过高时，会发生快速训练，导致模型权重发生破坏性变化。Length 指的是训练中的 **Epochs** 或 Steps 数量，表示训练阶段持续的时间。
   - 成员们强调了 **Fine-tuning** 的重要性，以避免剧烈变化，同时保留模型对语言的理解。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1339345956369469490)** (18 条消息🔥): 

> `Agent 课程支持、Endpoint 变更、模型名称更新、学习小组咨询、测试新工具` 


- **Agent 课程频道混淆**：一位成员强调需要切换频道以获得更好的 Agent 课程支持，建议前往专门的 "agent course" 板块。
   - 支持入口对于解决用户面临的常见问题仍然至关重要。
- **更换过载的 Endpoint**：一位用户建议将 Endpoint 更改为新链接，以解决由于过载导致的连接问题：[new endpoint](https://jc26mwg228mkj8dw.us-east-1.aws.endpoints.huggingface.cloud)。
   - 他们还指出需要将模型名称更新为 'deepseek-ai/DeepSeek-R1-Distill-Qwen-32B' 才能正常运行。
- **加入学习小组仍具挑战**：一位用户询问如何加入 Agent 课程的学习小组，表示对流程不确定。
   - 另一位成员也提出了同样的问题，凸显了社区对共享资源的共同需求。
- **测试工具的困惑**：一位在课程中创建了工具的成员询问，在测试时是否需要编写 Prompt。
   - 关于激活问题的并行咨询引出了对 LLM 可能过载的建议。
- **常见故障排除技巧**：建议用户检查日志以发现潜在问题，并提示缺少 `HF_TOKEN` 等定义是常见问题。
   - 这强调了在用户设置中进行正确配置对于成功交互的重要性。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1339279913928232971)** (717 条消息🔥🔥🔥): 

> `Hugging Face Agents Course, Discord 验证问题, 学习小组与协作, 模型访问与部署, 课程完成与证书` 


- **Hugging Face Agents Course 概览**：参与者们对加入 **Hugging Face AI Agents Course** 感到非常兴奋，分享了他们的背景和所在地，包括来自印度、加拿大和巴西等国家。许多用户表达了学习 Agent 的渴望，并希望在整个课程期间与同行协作。
- **Discord 验证问题**：一些用户在通过 Discord 验证其 Hugging Face 账号时遇到问题，导致反复出现连接错误以及对流程的困惑。用户建议通过登出并使用不同的浏览器来解决这些问题，部分用户最终成功完成了验证过程。
- **协作与学习小组**：许多参与者寻求与其他学员建立联系以进行协作学习，特别是对组建课程学习小组感兴趣。用户分享了他们的 LinkedIn 个人资料，并表达了在整个学习过程中互相支持的意愿。
- **模型访问与调试**：参与者讨论了在使用 Agent 时面临的技术挑战，特别是模型过载和错误信息的问题。还有关于 Agent 如何处理工具错误以及它们是否能自主调试问题的咨询，强调了彻底测试的重要性。
- **课程完成与证书**：用户确认已完成第一单元并收到了证书，一些用户好奇如何查看测验答案以便进一步学习。还提出了关于课程认证流程以及生成证书所需时间的问题，体现了对课程内容的积极参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wxknx1kg971u7k1n.us-east-1.aws.endpoints.huggingface.cloud'`">未找到标题</a>: 未找到描述</li><li><a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud'">未找到标题</a>: 未找到描述</li><li><a href="https://pflgm2locj2t89co.us-east-1.aws.endpoints.huggingface.cloud',">未找到标题</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/introduction">Agent 简介 - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/ml-for-3d-course/en/unit0/introduction">欢迎来到 🤗 Machine Learning for 3D Course - Hugging Face ML for 3D Course</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/m-ric/beam_search_visualizer">Beam Search Visualizer - Hugging Face Space (由 m-ric 创建)</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit0/introduction">欢迎来到 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/en/unit0/introduction">欢迎来到 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/agents">AgentS (Sean M. Murphy)</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course/unit1/final-quiz">第 1 单元测验 - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/agents-course">欢迎来到 🤗 AI Agents Course - Hugging Face Agents Course</a>: 未找到描述</li><li><a href="https://www.youtube.com/live/iLVyYDbdSmM">欢迎来到 Agents 课程！课程介绍与问答</a>: 在 Agents Course 的首次直播中，我们将解释课程的运作方式（范围、单元、挑战等）并回答您的问题。不要错过...</li><li><a href="https://huggingface.co/blog/smolagents#%E2%9C%85-when-to-use-agents--%E2%9B%94-when-to-avoid-them">smolagents 介绍：用代码编写动作的简单 Agent。</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.2-1B">meta-llama/Llama-3.2-1B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/what-a-sunny-day-gif-25989302">天气真好 GIF - What A Sunny Day - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/agents-course">agents-course (Hugging Face Agents Course)</a>: 未找到描述</li><li><a href="https://uningenieur.fr">aperrot 🍹 主页 - aperrot 🍹 主页</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/agents-course/certificates/tree/main">agents-course/certificates at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1339746840513478839)** (5 条消息): 

> `DeepSeek AI-HPC, Granite 3.2 MoE, GPT-3.5 Data Distillation` 


- **DeepSeek AI-HPC 的高性价比协同设计**：[YouTube 视频](https://youtu.be/wGWn3eVPvH8)标题为 "DeepSeek 🐋 | Fire-flyer AI-HPC"，讨论了在计算能力需求日益增长的背景下，针对深度学习的高性价比软件和硬件协同设计。
   - *DeepSeek 采用了创新解决方案*，以应对快速发展的深度学习技术所带来的挑战。
- **大语言模型（Large Language Models）的新研究**：[arXiv](https://arxiv.org/abs/2408.14158) 上概述了由 [Wei An](https://arxiv.org/search/cs?searchtype=author&query=An,+W) 等人撰写的论文，探讨了大语言模型（LLM）的进展及其对 AI 发展的影响。
   - 该研究中采用的具体方法论细节为该领域的尖端发展提供了见解。
- **Granite 3.2 MoE 预览洞察**：一位用户分享了对 Granite 3.2 MoE 的印象，认为它可能仅从 **GPT-3.5** 进行了数据蒸馏（Data Distillation），暗示其学习范围存在局限性。
   - 据指出，该模型的训练数据仅更新至 **2021年**，这引发了对其在近期发展中适用性的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2408.14158">Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning</a>: 深度学习 (DL) 和大语言模型 (LLMs) 的快速进步呈指数级增加了对计算能力和带宽的需求。结合更高算力的高昂成本...</li><li><a href="https://youtu.be/wGWn3eVPvH8">DeepSeek 🐋 | Fire-flyer AI-HPC:  A Cost-Effective Software Hardware Co-design for Deep Learning</a>: 深度学习 (DL) 和大语言模型 (LLMs) 的快速进步呈指数级增加了对计算能力和带宽的需求。结合...</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V3-Base">deepseek-ai/DeepSeek-V3-Base · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1339414632678424667)** (3 条消息): 

> `Deep Research Access, File & Image Uploads in ChatGPT, Model Spec Update` 


- **Pro 用户现可使用 Deep Research**：OpenAI 宣布 **Deep Research** 现已面向所有 **Pro 用户**开放，支持包括移动端和桌面端应用（iOS, Android, macOS 和 Windows）在内的多个平台。
   - 此功能增强了在各种设备上的研究能力，触达了更广泛的用户群体。
- **ChatGPT 增强：支持文件与图片上传**：更新后，**OpenAI o1** 和 **o3-mini** 现已支持在 ChatGPT 中进行*文件*与*图片上传*。
   - 此外，Plus 用户的 **o3-mini-high 限制**已提升 **7 倍**，允许每天最多 **50 次上传**。
- **Model Spec 重大更新**：OpenAI 分享了 [Model Spec](https://openai.com/index/sharing-the-latest-model-spec/) 的**重大更新**，详细说明了对模型行为的预期。
   - 该更新强调了对**可定制性**、**透明度**的承诺，并致力于营造一种*智力自由*的氛围，供用户使用 AI 进行探索和创作。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1339284898216804354)** (347 条消息🔥🔥): 

> `OpenAI future ownership, AI model capabilities, Fictional violence in AI, Current AI tools and platforms, Comparison of AI models` 


- **对 OpenAI 所有权的担忧**：讨论围绕 Elon Musk 收购 OpenAI 的可能性展开，许多人对此表示怀疑，并希望如果发生收购，能将技术开源（open-sourcing）。
   - 用户推测大型科技公司可能会将利润置于公共利益之上，从而引发对 AI 过度控制的担忧。
- **AI 模型功能与过滤**：参与者讨论了不同 AI 模型在处理虚构暴力方面的差异，重点介绍了用于无过滤创意写作的工具，如 Sudowrite。
   - 一些用户指出，在使用 AI 服务时保持隐私和安全设置的重要性，以防止滥用。
- **新兴 AI 工具与模型**：社区分享了对不同 AI 工具的见解，包括 DeepSeek 及其相对于大公司模型的优势，以及 AI 写作助手的各项功能。
   - 用户强调了运行大型开源模型需要强大的硬件支持，并对 AI 能力的潜在未来感到兴奋。
- **AI 模型的对比讨论**：关于 GPT-4、O3 以及 GPT-5 等新兴技术有效性的辩论此起彼伏，还幽默地提到了假设的“GPT Megazord”。
   - 成员们担心持续将模型合并为一个可能会导致 AI 推理中出现意想不到的结果。
- **关于数据隐私的警告**：用户强调了 AI 公司数据收集的相关问题，特别是关于恶意行为者利用 AI 的情况以及此类行为的潜在后果。
   - 对话提出了关于安全措施是否充分以及 AI 技术的伦理影响等问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gemini.google.com/app">‎Gemini - 激发灵感的聊天工具</a>：Bard 现已更名为 Gemini。从 Google AI 获取写作、规划、学习等方面的帮助。</li><li><a href="https://arxiv.org/abs/2405.04517">xLSTM: Extended Long Short-Term Memory</a>：在 20 世纪 90 年代，恒定误差轮转（constant error carousel）和门控（gating）作为 Long Short-Term Memory (LSTM) 的核心思想被引入。自那时起，LSTM 经受住了时间的考验，并为众多领域做出了贡献...</li><li><a href="https://tenor.com/view/meme-gif-26461359">Meme GIF - Meme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=7dd_r-aqecw">用儿童机器人伴侣 Moxie 重新思考社会发展</a>：凭借蓝色的身体和大大的动漫眼睛，Moxie 想成为你孩子的朋友。这款由 AI 驱动的机器人被《时代》杂志评为 2020 年最佳发明之一，旨在...</li><li><a href="https://community.openai.com/t/webrtc-real-time-api-with-microcontroller/1059806">基于微控制器的 WebRTC Real-Time API</a>：嗨！在第 9 天的演示中，我们看到一个带有微控制器的毛绒玩具正在调用 WebRTC Real-Time API（链接：YouTube 直播）。您能提供更多关于整体架构的细节吗？例如...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1339308028737884361)** (12 条消息🔥): 

> `Custom GPT Models, Hiring Experts, Limits of Free Plan` 


- **自定义 GPT 运行在 GPT-4o 上**：一位成员确认自定义 GPT（Custom GPTs）在 **GPT-4o** 模型上运行，回答了关于底层模型的查询。
- **寻求特定领域的专业知识**：一位寻求招聘的成员表示需要为其初创公司寻找专家，请有**丰富经验**的人员与其联系。
   - 另一位成员幽默地询问“像专业人士一样吃披萨”是否算作相关经验，建议在请求中说明更多细节。
- **了解免费版限制**：一位用户询问如何核实各种模型在免费版下的限制，询问是否涉及消息数量和文件附件。
   - 一位成员回答说，限制每天都会根据各种因素发生变化，*只有部分固定数值*，例如 AVM 为 **15分钟/月**。
- **免费版使用指南**：尽管有人询问免费版限制的指南，但据称目前并没有真正的**粗略指南**可供参考。
   - 用户必须根据其**所在地区和使用时区**来观察限制情况。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1339280116894797865)** (16 条消息🔥): 

> `Function Calling 问题, Prompt 分享, 使用 CoT 和 ToT, 错误解读, Prompt Engineering 讨论` 


- **System Prompt 中的 Function Calling 挑战**：一位成员分享了在其 Prompt 中与客户状态相关的 Function Calling 持续出现的问题，指出 AI 回复与函数触发之间存在偏差。
   - 他们强调了在客户互动后正确调用 `determine_status` 函数的重要性，以避免丢失 leads。
- **鼓励 Prompt 分享**：成员们讨论了在频道中分享 Prompt 的适当性，鼓励提出问题和展开讨论，而不仅仅是信息堆砌。
   - 一位成员对分享冗长的 Prompt 表示迟疑，但被告知欢迎关于异常 Prompt 的探讨。
- **为 Functions 使用 CoT 和 ToT**：另一位成员提到利用 Chain of Thought (CoT) 和 Tree of Thought (ToT) 策略来有效处理模糊的客户回复。
   - 他们强调，精确结构化的 Prompt 应该有助于实现正确的 Function Calling 序列。
- **Prompt 中错误识别的价值**：有人建议通过要求模型解读 Prompt 来识别错误，重点关注潜在的冲突或歧义。
   - 该策略被推荐用于增强 Prompt 的清晰度并解决模型的误解。
- **关于 'Boomer Prompts' 的讨论**：一位成员幽默地提出了关于 'boomer prompt' 含义的问题，暗示了 Prompt 的文化或代际背景。
   - 这引发了人们对语言和 Prompt 惯例在不同受众之间可能存在差异的兴趣。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1339280116894797865)** (16 条消息🔥): 

> `Function calling 问题, Prompt 分享实践, 使用 CoT 和 ToT, ChatGPT 对比 playground, 解读 prompts` 


- **Prompt 中的 Function calling 混乱**：一位用户讨论了其 System Prompt 中 Function calling 面临的挑战，强调了 AI 有时无法指示状态，导致 Pipedrive 中潜在的 leads 丢失。
   - 他们分享了使用函数对客户回复进行分类的结构化方法，但在 AI 行为的一致性方面仍面临问题。
- **讨论中的 Prompt 分享礼仪**：另一位用户询问了关于发布 Prompt 的事宜，引发了关于在 Discord 频道内分享的最佳实践的讨论。
   - 成员们建议，提问或观察比单纯的“信息堆砌”更能产生良好的讨论。
- **在 Prompt 中利用 CoT 和 ToT**：一位用户解释说，他们结合了 `Chain of Thought (CoT)` 和 `Tree of Thought (ToT)` 策略，以确定在模糊的客户互动中何时调用函数。
   - 他们希望获得关于其 Prompt 结构的反馈，以改进功能性。
- **ChatGPT 与 playground 之间的差异**：用户注意到使用 ChatGPT 与使用 playground 的工作方式不同，模型处理 Prompt 和错误的方式也各不相同。
   - 建议通过识别错误模式来优化 Prompt 指令，以获得更好的结果。
- **解读 Prompt 冲突以提高清晰度**：一位成员推荐了一种技术，即要求模型在不执行指令的情况下解读 Prompt，以发现潜在的歧义。
   - 这种策略有助于揭示意想不到的冲突并改进整体的 Prompt 设计。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1339280783881539715)** (392 messages🔥🔥): 

> `Cursor IDE Features, OpenAI o3-mini vs. Claude, Anthropic's Hybrid AI Model, Tool Calling and Coding, MCP Server Utilization` 


- **Cursor 在 o3-mini 上的用户体验**：用户发现 **o3-mini** 模型在 Tool Calling 能力方面表现不佳，通常需要多次 Prompt 才能达到预期效果，导致用户挫败感增加，并对其在编程任务中的价值产生怀疑。
   - 许多人提到，与 **o3-mini** 相比，**Claude** 的推理模型在工具使用方面表现更出色，这引发了关于集成类似于 Cline 的 Plan/Act 模式以改善用户体验的讨论。
- **对 Anthropic 混合模型的期待**：用户对 Anthropic 即将推出的**混合 AI 模型**充满期待，据报道，该模型在利用最大推理能力时，在编程任务中的表现优于 **OpenAI 的 o3-mini**。
   - 新模型在编程基准测试中的高性能表现表明，与当前产品相比，它能显著增强编程工作流。
- **对 Tool Calling 有效性的担忧**：用户对 **o3-mini** 在 Tool Calling 方面的灵活性和效率不足表示不满，对其在实际编程环境中的实用性表示担忧。
   - 持续的讨论显示，用户希望 AI 模型能简化复杂的编程任务，并建议建立 Prompt 最佳实践，以引导 AI 输出更高质量的代码。
- **关于 MCP 使用的观点**：**MCP (Multi-Channel Processor)** 的概念在讨论中出现，被视为通过集成多个 AI 模型来提高效率和产出的增强编程任务的工具。
   - 用户分享了利用 MCP 服务器优化编程工作流并解决单个模型局限性的各种经验和策略。
- **市场竞争与定价**：对话涉及了 AI 模型的定价策略，用户指出 **Windsurf** 不允许用户使用自己的 Key，缺乏灵活性，导致用户对其价值不满。
   - 许多用户表示相比竞争对手更青睐 **Cursor** 的功能和实用性，并指出了其在成本效益和用户体验方面的优势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/darth-vader-alter-the-deal-empire-strikes-back-star-wars-gif-15971205">Darth Vader Alter The Deal GIF - Darth Vader Alter The Deal Empire Strikes Back - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/funny-gif-27151298">Funny GIF - Funny - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑能力的量化基准。</li><li><a href="https://forum.cursor.com/t/claude-not-detecting-lint-errors-half-the-time-after-writing-code/50455">Claude not detecting lint errors half the time after writing code</a>：Claude 在编写代码后有一半的时间无法检测并修复 Lint 错误。Linter 错误会立即显示在代码编辑器中，但只有进行后续 Prompt 才能让 AI 修复该问题...</li><li><a href="https://forum.cursor.com/t/supervisory-agent-to-guide-worker-agent/49395/7">&quot;Supervisory&quot; agent to guide &quot;worker&quot; agent</a>：Aider 展示了这是实现最佳 AI 编程 Agent 的方式。此外，我认为实现这种流程的 Agent 对 Cursor 来说将是绝对的规则改变者。所以，非常棒的建议！</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>：介绍 Wave 3，这是我们对 Windsurf 编辑器的第三批更新。</li><li><a href="https://x.com/pontusab/status/1890038188934410482">Tweet from Pontus Abrahamsson — oss/acc (@pontusab)</a>：直接从你的 package.json 生成你自己的优化 Cursor 规则，现已在 Cursor Directory 上线！构建工具：◇ @nextjs - 框架 ◇ @vercel - 托管 ◇ @aisdk - AI 工具包 ◇ @xai - LLM ◇ @shadcn - U...</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>：未找到描述</li><li><a href="https://status.cursor.com/">Cursor Status</a>：未找到描述</li><li><a href="https://www.augmentcode.com/?">Augment Code – Developer AI for real work</a>：体验真正理解你代码库的 AI 平台。我们的开发者 AI 帮助团队更快地编写代码，做出更明智的决策，并解锁集体知识。立即免费试用。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1339707671850254398)** (2 条消息): 

> `DeepHermes-3 Preview, Long chain of thought reasoning, LLM advancements, Hugging Face Model Links` 


- **发布 DeepHermes-3 Preview**：Nous Research 推出了 **DeepHermes-3 Preview**，这是一款开创性的 LLM，它**统一了推理**与传统语言模型的能力，现已在 [Hugging Face](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview) 上线。
   - 该模型展示了在 **LLM annotation** 和 **function calling** 方面的增强，并且是首批在单一框架中处理长链思维（long chains of thought）的模型之一。
- **使用 LongCoT 所需的专业知识**：为了启用长链思维推理，用户必须使用特定的 system prompt，包括要求将内部思考过程包含在 `<think>` 标签中的指令。
   - 这种直接的方法鼓励在得出解决方案之前进行更深层次的系统化推理，从而提升模型性能。
- **基准测试报告改进**：DeepHermes-3 的早期基准测试显示，在数学推理方面有**显著增强**，在 **Google Proof Question Answering (GPQA)** 方面也有小幅提升。 
   - 该模型旨在通过社区反馈和对其功能的进一步探索来完善其**推理能力**。
- **致谢开源协作**：DeepHermes-3 的开发对支持数据收集、评估和训练工作的关键社区成员表示感谢。
   - 这种协作精神对于深度推理模型的持续进步和增强用户可控性（steerability）至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-8B-Preview-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1339308516321787955)** (268 条消息🔥🔥): 

> `DeepHermes-3 model preview, Reasoning capabilities, Mobile performance limitations, Comparisons with other models, Accessibility of hardware for running models` 


- **DeepHermes-3 模型预览版发布**：Nous Research 推出了 DeepHermes-3 Preview，这是一款统一了推理和直觉能力的新型 LLM，同时允许切换长链思维以提高准确性。
   - 该模型已在 Hugging Face 上发布，供用户测试其能力，例如处理多轮对话。
- **关于移动设备限制的讨论**：成员们讨论了在移动设备上使用 AI 模型的限制，特别是关于 RAM 占用和后台应用管理。
   - 一位用户对他们 12GB 的手机仅允许 2GB 可访问内存表示沮丧，这阻碍了他们运行模型的能力。
- **模型性能对比**：DeepHermes-3 与 DeepSeek 模型的对比突出了后者在数学问题上的强劲表现，但也指出了其对话能力较弱。
   - 用户指出，虽然 DeepSeek 模型在特定任务上表现出色，但 DeepHermes-3 旨在提供通用的对话和推理能力。
- **测试模型的潜在硬件解决方案**：一位用户建议购买 16GB 的 ARM SBC 用于便携式计算，以便在旅行时运行小型 LLM。
   - 这些设备的价格从 8GB 的约 80 美元到 16GB 的 100-140 美元不等，为感兴趣的人提供了经济实惠的选择。
- **远程访问中 X Forwarding 的效用**：讨论了将 X Forwarding 作为在远程 Linux 服务器上运行图形应用程序的方法，从而有效地实现远程桌面功能。
   - 然而，用户表示现在不是购买新设备的时候，特别是考虑到目前的财务状况。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://zed.dev/blog/edit-prediction">Zed 现在通过 Zeta（我们的全新开源模型）预测你的下一次编辑 - Zed 博客</a>：来自 Zed 博客：一个能预判你下一步动作的工具。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1/commit/8a58a132790c9935686eb97f042afa8013451c9f">更新 tokenizer_config.json · deepseek-ai/DeepSeek-R1 at 8a58a13</a>：未找到描述</li><li><a href="https://fxtwitter.com/dreamworks2050/status/1890164583249375377">来自 M4rc0𝕏 (@dreamworks2050) 的推文</a>：DEEPHERMES-LLAMA-3-8B 思考模式：开启 - 首个 RUNGGUF - F16，由 @NousResearch 提供 🔥MacBook Pro M4 Max：28.98t/s</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">来自 Nous Research (@NousResearch) 的推文</a>：介绍 DeepHermes-3 预览版，这是一款统一了推理和直觉语言模型能力的新型 LLM。https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview DeepHermes 3 构建于...</li><li><a href="https://huggingface.co/Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF/tree/main">Joseph717171/Hermes-3-Llama-3.1-8B-OQ8_0-F32.EF32.IQ4_K-Q8_0-GGUF at main</a>：未找到描述</li><li><a href="https://f-droid.org/packages/superfreeze.tool.android/">SuperFreezZ 应用停止器 | F-Droid - 免费开源 Android 应用仓库</a>：完全冻结应用的所有后台活动。</li><li><a href="https://tenor.com/view/apparently-its-a-big-deal-big-deal-big-deal-apparently-it-is-a-big-deal-gif-26730751">Apparently Its A Big Deal Big GIF - Apparently Its A Big Deal Big Deal - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview#prompt-format-for-function-calling">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kSNKuHX9AZo">比亚迪股价在披露此事后暴涨...</a>：比亚迪股价在披露此事后暴涨...澳大利亚最好的太阳能公司刚刚安装了我的新太阳能系统。点击这里查看：https...</li><li><a href="https://www.youtube.com/watch?v=P_fHJIYENdI)">AI 做过的最有用的事</a>：世界上最重大的问题可能会通过使用 AI 解锁的微小分子来解决。今天就通过 https://ve42.co/hostinger 将你的伟大创意上线 - 代码 ...</li><li><a href="https://forms.gle/s4dG8RYVmcu1e1Cg7">Deepfake 技术：威胁还是工具</a>：本调查旨在评估公众对 Deepfake 技术的认知，并为研究目的收集数据。您的回答将帮助我们了解人们对 Deepfake 的识别程度及其潜在风险...</li><li><a href="https://x.com/sama/status/1889755723078443244">来自 Sam Altman (@sama) 的推文</a>：OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望更好地分享我们的预期路线图，并更好地简化我们的产品线。我们希望 AI 能为你“开箱即用”；我们真的...</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding)">LM Studio 文档 | LM Studio 文档</a>：学习如何使用 LM Studio 在本地运行 Llama, DeepSeek, Phi 以及其他 LLM。</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta 版本</a>：LM Studio 的 Beta 和发行候选版本</li><li><a href="https://www.cnbc.com/2025/02/11/ken-griffin-says-trumps-bombastic-trade-rhetoric-is-a-mistake-thats-eroding-trust-in-the-us.html">Ken Griffin 表示特朗普“夸张”的贸易言论是一个错误，正在侵蚀对美国的信任</a>：这位亿万富翁对冲基金创始人的评论是在特朗普周一晚间签署一项对钢铁和铝进口征收 25% 关税的命令后发表的。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1339642166640640123)** (2 条消息): 

> `Llama-3B-Instruct 上的 SFT，基础模型性能损失，特定领域挑战` 


- **SFT 导致 Llama-3B-Instruct 性能问题**：一位成员报告称，在 **Llama-3B-Instruct** 上以 **2e-4** 的学习率进行了 **SFT**。
   - 他们注意到在第一个 epoch 期间，基础模型的性能出现了显著下降（使用 **Winogrande** 进行测量）。
- **性能下降与领域特定性有关**：性能问题似乎源于**巴西葡萄牙语**的**技术领域**。
   - 该成员寻求建议，以克服在这一特定领域面临的挑战。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339398214503370873)** (3 条消息): 

> `Nvidia 关于 GPU kernels 的博客文章，LLM 报告论文，LLM 的最先进方法` 


- **Nvidia 展示 LLM 生成的 GPU kernels**：来自 [Nvidia 的新博客文章](https://x.com/anneouyang/status/1889770174124867940) 强调，LLM 生成的 GPU kernels 在 **KernelBench Level 1** 上实现了 **100% 的数值正确性**，并展示出优于 **FlexAttention** 的加速效果。
   - 这一进展表明在优化 LLM 的 GPU 性能方面取得了重大突破。
- **寻找更新的 LLM 报告论文**：一名成员正在积极寻找涵盖推理模型等**最先进方法 (State of the art methods)** 的近期 LLM 报告论文，并指出 2024 年 2 月的综述论文已经过时。
   - 这反映了社区对 LLM 方法论最新研究和进展的渴望。
- **LLM 进展的相关论文**：针对寻找更新 LLM 论文的需求，teknium 推荐了 **r1 kimik** 和 **synthlab** 论文，认为它们是最相关的选择。
   - 这表明成员们正在分享有价值的资源，以辅助研究和开发工作。



**提到的链接**：<a href="https://x.com/anneouyang/status/1889770174124867940">来自 Anne Ouyang (@anneouyang) 的推文</a>：Nvidia 的新博客文章：LLM 生成的 GPU kernels 显示出优于 FlexAttention 的加速效果，并在 🌽KernelBench Level 1 上实现了 100% 的数值正确性。

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1339301821877059716)** (1 条消息): 

> `美国 AI 安全宣言，国际 AI 合作，对威权政体的担忧` 


- **美英拒绝 AI 安全协议**：在最近的一次峰会上，**美国**和**英国**拒绝签署一份联合 AI 安全宣言，美国领导人强调他们致力于保持 AI 领导地位。
   - Vance 警告不要与**威权政体**结盟，指出这些政体过去曾滥用技术侵犯国家安全。
- **国际治理分歧**：由包括**中国**、**印度**和**德国**在内的多个国家签署的宣言，重点在于加强 AI 治理的国际合作。
   - 然而，一名美国官员表示，美国不赞成有关**多边主义 (multilateralism)** 的措辞，并反对解释协作框架的相关条款。
- **对基础设施安全的担忧**：Vance 警告说，在 AI 领域与威权国家接触可能会导致国家**信息基础设施**受损，并引用了 **CCTV** 和 **5G** 的例子。
   - 他将这些技术描述为**廉价**但获得大量补贴的出口产品，可能会使各国受到威权势力的影响。



**提到的链接**：<a href="https://arstechnica.com/ai/2025/02/us-and-uk-refuse-to-sign-ai-safety-declaration-at-summit/">美英在峰会上拒绝签署 AI 安全宣言</a>：美国的立场较拜登政府发生了“180 度大转弯”。

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1339398214503370873)** (3 条消息): 

> `Nvidia LLM 生成的 GPU Kernels，近期 LLM 报告论文，r1 kimik 和 synthlab 论文` 


- **Nvidia 凭借 LLM kernels 实现 100% 正确性**：[Nvidia 的新博客文章](https://x.com/anneouyang/status/1889770174124867940) 强调，LLM 生成的 GPU kernels 显示出优于 **FlexAttention** 的加速效果，并在 **KernelBench Level 1** 上实现了 **100% 的数值正确性**。
   - *这可能会显著提升使用这些模型的开发者的性能指标。*
- **寻找最新的 LLM 报告论文**：@pier1337 对涵盖近期最先进方法（包括推理模型）的 **LLM 报告论文** 表示感兴趣，并称之前的论文已过时。
   - 他们发现 **2024 年 2 月的 LLM 综述论文** 很有用，但现在正在寻找更及时的信息。
- **Teknium 推荐的相关论文**：作为回应，teknium 建议 **r1 kimik** 和 **synthlab 论文** 是获取 LLM 进展最新信息最相关的来源。
   - *这些论文可能为那些研究前沿推理模型的人员提供实质性的见解。*



**提到的链接**：<a href="https://x.com/anneouyang/status/1889770174124867940">来自 Anne Ouyang (@anneouyang) 的推文</a>：Nvidia 的新博客文章：LLM 生成的 GPU kernels 显示出优于 FlexAttention 的加速效果，并在 🌽KernelBench Level 1 上实现了 100% 的数值正确性。

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1339286171007717408)** (11 条消息🔥): 

> `Groq DeepSeek R1 70B 发布，OpenRouter 新的排序偏好，API usage 字段更新，Token 计数对比，关于模型排名一致性的讨论` 


- **Groq DeepSeek R1 70B 提供创纪录的速度**：OpenRouter 宣布新增 **Groq DeepSeek R1 70B**，记录到了令人印象深刻的 **1000 tokens per second** 吞吐量，并支持各种参数，同时提供增加速率限制（rate limits）的选项。
   - 这是与 [OpenRouter AI](https://x.com/OpenRouterAI/status/1889726731571044538) 更广泛集成的一部分，旨在最大化用户与平台的交互。
- **新的默认排序选项提升用户体验**：现在，用户可以通过更改设置来轻松调整模型提供商的默认排序偏好，以**专注于吞吐量**或在速度与成本之间取得平衡。
   - 此外，在任何模型名称后附加 `:nitro` 可确保用户访问可用的最快提供商，正如 [OpenRouter](https://x.com/OpenRouterAI/status/1890061196885360647) 的公告所述。
- **API usage 字段可能切换到原生 token 计数**：一项拟议的更新建议将 API 中的 `usage` 字段从 GPT token 归一化更改为模型的**原生 token 计数**，目前正在征求用户反馈。
   - 用户对模型排名和一致性提出了担忧，强调了在不同模型之间保持**公平比较**的重要性。
- **Token 计数差异引发讨论**：关于从 GPT 的归一化计数切换到原生 token 计数将如何影响 **Vertex** 等模型的推测不断，且对不同 token 比例的担忧依然存在。
   - 回复确认虽然存在细微差异，但不会像以前基于字符的模型那样极端，因此不会导致颠覆性的变化。
- **呼吁在 usage 报告中增加额外功能**：有人建议在 API 中加入一个显式返回 **GPT token 计数**的字段，反映了对更全面使用指标的需求。
   - 这与正在进行的关于提高模型比较和使用报告清晰度及透明度的讨论相一致。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1890061196885360647">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 新功能：你现在可以在账户设置中更改任何模型的默认提供商排序。如果你在意速度，请按“吞吐量”排序 🚀；按“默认”排序则平衡了运行时间、价格和吞吐量...</li><li><a href="https://x.com/OpenRouterAI/status/1889726731571044538">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 很高兴宣布 @GroqInc 正式上线 OpenRouter！⚡️- 包括创纪录速度的 1000 TPS 蒸馏版 DeepSeek R1 70B - 支持大量参数 - 如果你愿意，可以自带密钥（BYOK）以获得速率限制提升...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1339280395333533788)** (257 条消息🔥🔥): 

> `OpenAI o3-mini 功能，Deepseek R1 的问题，自我审核的 OpenAI 端点，Google 的速率限制错误，使用 AI 模型进行 YouTube 内容创作` 


- **面向 Tier 3 用户的 OpenAI o3-mini 功能**：在等待 8 天后，一位用户报告称 OpenAI 为其 Tier 3 密钥（之前是 Tier 2）启用了 o3-mini。
   - 他们对等待时间表示沮丧，但指出现在可以通过 BYOK 使用 OpenAI 额度。
- **Deepseek R1 展示了卓越的推理能力**：一位用户分享了使用 Deepseek R1 的经验，在处理复杂的 SIMD 函数时，与 o3-mini 相比，它展示了令人印象深刻的推理能力。
   - 他们称 o3-mini “固执”，暗示其在推理任务中效果较差。
- **讨论自我审核的 OpenAI 端点**：一位用户对是否提供自我审核的 OpenAI 端点表示关注，期望获得更低的延迟和一致的结果。
   - 团队表示他们正在探索这一选项，并承认了用户对审核问题的担忧。
- **Google 的速率限制问题引发沮丧**：用户报告由于资源耗尽收到来自 Google 的 429 错误，影响了他们对 Sonnet 模型的使用。
   - OpenRouter 团队提到，他们正在解决由 Anthropic 容量限制引起的日益严重的速率限制问题。
- **用于创建 YouTube 缩略图和标题的最佳 AI**：一位用户询问了旨在最大化点击率的生成 YouTube 内容的最佳 AI 模型。
   - 另一位用户建议通过跟踪表现来优化模型输出，尽管对现有工具表示不满。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://trytest.in,">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/sama/status/1889755723078443244">Sam Altman (@sama) 的推文</a>: OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为您“直接可用”；我们重新...</li><li><a href="https://openrouter.ai/rankings/programming?view=week">LLM 排名：编程 | OpenRouter</a>: 根据编程提示词的使用情况对语言模型进行排名和分析</li><li><a href="https://openrouter.ai/docs/use-cases/for-providers#for-providers">提供商集成 - 将您的模型添加到 OpenRouter</a>: 了解如何将您的 AI 模型与 OpenRouter 集成。为提供商提供的完整指南，使其模型可通过 OpenRouter 的统一 API 使用。</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: LLM 的统一接口。为您的提示词找到最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: R1 – 提供商状态</a>: 查看提供商状态并向 DeepSeek: R1 发起负载均衡请求 - DeepSeek R1 已发布：性能与 [OpenAI o1](/openai/o1) 相当，但已开源并具有完全开放的推理 Token。它...</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API 速率限制 - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 防护。有效地配置和监控您的模型使用限制。</li><li><a href="https://openrouter.ai/docs/quickstart">OpenRouter 快速入门指南</a>: 开始使用 OpenRouter 针对数百个 AI 模型的统一 API。了解如何使用 OpenAI SDK、直接 API 调用或第三方框架进行集成。</li><li><a href="https://openrouter.ai/docs/features/web-search#customizing-the-web-plugin">网络搜索 - AI 模型的实时网络落地 (Web Grounding)</a>: 在您的 AI 模型响应中启用实时网络搜索功能。通过 OpenRouter 的网络搜索功能，为任何模型的输出添加事实性的、最新的信息。</li><li><a href="https://openrouter.ai/docs/features/provider-routing#ignoring-providers)">提供商路由 - 智能多提供商请求管理</a>: 智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://openrouter.ai/docs/features/provider-routing#floor-price-shortcut">提供商路由 - 智能多提供商请求管理</a>: 智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">提供商路由 - 智能多提供商请求管理</a>: 智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://community.openai.com/t/are-openai-credits-expiring/511215">OpenAI 额度会过期吗？</a>: 自仪表板更改以来，我没有看到关于额度过期日期的警告。是他们忘了放，还是放在别处了，或者额度不再过期了？</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/error-code-429">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1339378601531019391)** (1 条消息): 

> `功能反馈` 


- **功能反馈请求**：一位成员对新功能表示热烈欢迎，称 *“这看起来太棒了！”*，并鼓励其他人分享任何缺失的功能。
   - 该消息强调了社区通过用户反馈改进产品的关注点。
- **鼓励社区参与**：同一位成员通过 *“如果您发现任何缺失的功能，请告诉我们”* 这句话鼓励就功能发现进行持续沟通。
   - 这表明在收集用户输入和增强整体体验方面采取了主动的方法。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1339306866986975305)** (230 条消息🔥🔥): 

> `Perplexity Finance Dashboard, AI 模型性能, 客户支持问题, 推荐链接与折扣, AI 在技术中的应用` 


- **探索 Perplexity Finance Dashboard**：成员们讨论了新发布的 [Perplexity Finance Dashboard](https://www.perplexity.ai/search?q=%s&focus=[internet,scholar,writing,wolfram,youtube,reddit]&copilot=[true,false])，并确认这是否是 Perplexity 推出的首个此类仪表盘。
   - 用户希望在网页端和移动端 App 上能有专门的仪表盘按钮。
- **对 AI 模型性能的担忧**：关于 AI 模型的辩论，特别是 **R1** 与 DeepSeek 和 Gemini 等替代方案相比的效率和准确性，引发了关于首选用途和性能指标的讨论。
   - 成员们分享了他们的经验，提到了可以改善用户体验的具体特性和功能。
- **对客户支持体验的抱怨**：一位用户对 Perplexity 客服在处理账户问题时的响应缓慢和缺乏支持表示沮丧，具体涉及被扣除 Pro 账户费用但无法访问的问题。
   - 这引发了关于支持团队需要清晰沟通和协助的讨论。
- **推荐链接与折扣讨论**：成员们讨论了各种优惠和推荐链接，包括免费 Pro 订阅代码的获取方式。
   - 一些成员声称通过 Revolut 等服务的促销活动获得了延长的订阅期。
- **AI 工具及其局限性**：讨论强调了 AI 的矛盾性：在创建先进技术方面能力出众，但在处理基础任务准确性（特别是编码场景）时却表现挣扎。
   - 一位用户希望 AI 能更智能，能更准确地遵循文档和建议的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/guides/model-cards">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/hasbulla-hasbik-cute-meme-influencer-gif-21732737">Hasbulla Hasbik GIF - Hasbulla Hasbik 可爱 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/CuriousCharjan/status/1889807714576113845">来自 avdhesh.eth (@CuriousCharjan) 的推文</a>：天哪，这响应太疯狂了！这是代码。在结账时添加：FREEPPLXNTUOSS。引用 avdhesh.eth (@CuriousCharjan) 回复此推文以获得 6 个月的免费 Perplexity Pro。我会私信你代码！</li><li><a href="https://x.com/pplxfinance/status/1889742180421337120?s=61">来自 Perplexity Finance (@PPLXfinance) 的推文</a>：您每日获取最新市场洞察的来源——现已在 Perplexity 上线。市场摘要、每日亮点、收益快照，以及您理解背后原因所需的一切。Fi...</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 状态</li><li><a href="https://one.google.com/u/2/explore-plan/notebooklm?utm_source=notebooklm&utm_medium=web&utm_campaign=notebooklm_settings&pli=1&g1_landing_page=5&pageId=none">未找到标题</a>：未找到描述</li><li><a href="https://x.com/elder_plinius/status/1890028958907089059?t=Kv46N8eXldfN35QN-zmGhQ&s=19">来自 Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>：哈哈哈 💉💦 引用 Djah 〰️ (@Djahlor) 什么？？？@elder_plinius 是你做的吗？？</li><li><a href="https://x.com/perplexity_ai/status/1889366732432674961">来自 Perplexity (@perplexity_ai) 的推文</a>：我们很高兴地宣布，Million Dollar Questions 抽奖活动的获胜者是 Kaylee Edmondson！Kaylee 是一位来自田纳西州纳什维尔的小企业主。恭喜 Kaylee。感谢...</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1ilzw2e/i_made_a_chrome_extension_to_highlight_evidence/">Reddit - 深入探索</a>：我制作了一个 Chrome 扩展程序来突出显示证据</li><li><a href="https://sonar.perplexity.ai">Perplexity 的 Sonar</a>：使用由 Perplexity 创建的最佳 AI 问答引擎 API 进行构建。利用市面上最快、最便宜且具备搜索增强（search grounding）功能的产品为您的产品赋能。提供无与伦比的实时、全网范围的...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1339305657215488134)** (21 条消息🔥): 

> `EU AI Investment, Llama Model, DND Campaign Features, AI's Performance on Integer Queries, OpenAI Bid Situation` 


- **欧盟 AI 投资见解**：链接指向关于 [欧盟 AI 投资](https://www.perplexity.ai/search/eu-ai-investment-aE_wZ53LRUCrT.ntggaGZQ) 的讨论，重点介绍了近期推动欧洲 AI 发展的资金和政策。
   - 强调了需要制定强有力的战略，以跟上全球 AI 发展的步伐。
- **探索 Llama 模型能力**：用户分享了一个讨论 [Llama 模型](https://www.perplexity.ai/search/llama-model-gbV8Cv7ARLej0CKOEl8u0Q) 的链接，详细介绍了其架构和在 AI 应用中的用例。
   - 对话探讨了其相较于同类模型的潜在优势。
- **探索 DND 战役功能**：用户报告了 Perplexity AI 的 [DND 战役](https://www.perplexity.ai/search/start-dnd-campaign-where-your-YSt8QNOrSE.E87vCkaUQwA) 功能，该功能同时支持 DM 和玩家角色。
   - 他们询问了如何邀请朋友，并分享了关于游戏动态的见解。
- **AI 在整数查询上的表现受关注**：一项讨论揭示了对 AI 正确处理整数查询能力的挫败感，如该 [查询](https://www.perplexity.ai/search/what-is-the-smallest-integer-t-9EE9T0XnTiGJDu0ir1BecA) 所示。
   - 成员们推测了 AI 的学习曲线和改进策略。
- **马斯克对 OpenAI 竞购的变数**：一篇链接文章透露，如果某些条件得不到满足，马斯克可能会 [撤回对 OpenAI 的竞购](https://www.perplexity.ai/page/musk-to-withdraw-bid-if-openai-z5zXTCfGSMac79T.IzlL5w)。
   - 对话围绕这一潜在撤回对 AI 领域的影响展开。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1339352406453911592)** (11 条消息🔥): 

> `API 500 Error, Beta Testing for Sonar API on Cerebras` 


- **API 广泛出现 500 错误**：多位成员报告所有 API 调用均出现 **500 错误**，其中一位成员指出生产环境出现故障。
   - 一位用户表示“这不太妙”，这些错误持续了一段时间，随后另一位成员提到 API 现在似乎已恢复正常。
- **对在 Cerebras 上进行 Sonar Beta 测试感兴趣**：一位成员表达了成为 **Cerebras** 上 **Sonar** 的 API 版本 **Beta 测试员** 的热情，并表示他们已经梦寐以求好几个月了。
   - 他们的测试提议表明了对这些工具集成相关创新的潜在兴趣。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1339643524978577469)** (2 条消息): 

> `AI Engineering Summit 门票, Windsurf Wave 3 特性, Model Context Protocol, 可自定义应用图标, Turbo Mode` 


- **赢取 AI Engineering Summit 门票！**：我们将送出 **3 张门票**，参加 **2 月 20-21 日**在纽约市举行的 [AI Engineering Summit](https://www.ai.engineer/summit/2025)。填写表格即可参与，但您必须位于纽约市地区才有资格获得门票，链接在[这里](https://forms.gle/WM67ZgQngXaY4stq7)。
   - *不包含差旅费用*，但参会者将见到 Windsurf 的产品工程负责人，并获得活动专属周边。
- **Windsurf Wave 3 发布特性**：Windsurf **Wave 3** 引入了令人兴奋的新功能，包括 **Model Context Protocol (MCP)**、可自定义的应用图标以及增强的 **Tab to Jump** 导航。重大升级还包括用于自动执行命令的 **Turbo Mode** 以及改进的额度可见性。
   - 在 [Wave 3 博客文章](https://codeium.com/blog/windsurf-wave-3)中阅读完整更新，并在[此处](https://www.codeium.com/changelog)查看完整的更新日志。
- **Model Context Protocol 增强**：Cascade 支持 **Model Context Protocol (MCP)**，允许用户配置对用户定义的 MCP 服务器的工具调用。无论执行结果如何，每次 MCP 工具调用都消耗一个 flow action 额度。
   - 所有**个人计划**用户均可使用此新功能，可以通过点击 Cascade 输入工具栏中的锤子图标进行设置。
- **自定义图标现已上线！**：Windsurf 现在允许付费用户在 Mac (Beta) 上使用**自定义应用图标**，风格包括 Classic, Blueprint, Hand-drawn 和 Valentine。这些图标在系统范围内生效，但需要重启应用才能使更改生效。
   - 所有付费用户计划都可以访问此功能，进一步增强了应用的个性化。
- **面向 Cascade 用户的 Turbo Mode**：Cascade 中新引入的 **Turbo Mode** 通过自动执行命令和支持拖放图片上传来简化命令执行。这些增强功能还显著改进了补全（completions）并扩展了 @docs 选项。
   - 鼓励用户探索这些作为 Wave 3 发布内容一部分的功能，并加入专门频道的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forms.gle/WM67ZgQngXaY4stq7">来自 Windsurf 的 AI Engineering Summit 纽约门票 </a>：为了感谢我们的社区，我们将送出三张免费的纽约 AI Engineering Summit 门票！这场为期两天的活动将于 2 月 20-21 日举行，是听取顶级 AI 专家分享的绝佳机会...</li><li><a href="https://codeium.com/blog/windsurf-wave-3">Windsurf Wave 3</a>：介绍 Wave 3，这是我们对 Windsurf 编辑器的第三批更新。</li><li><a href="https://x.com/windsurf_ai/status/1890161230876381249">来自 Windsurf (@windsurf_ai) 的推文</a>：Wave 3 来了！本次更新包含：⏩ Tab to Jump 🔗 MCP 集成 ⚡ Turbo Mode 🎨 自定义图标……以及更多。</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor 更新日志 | Windsurf Editor 和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1339302510166544415)** (13 条消息🔥): 

> `Codeium Release 1.36.1, 故障排除, 实习机会, 即将发布的公告` 


- **Codeium Release 1.36.1 修复问题**：最新版本 **1.36.1** 将于明天上线，似乎解决了现有问题，建议在此期间切换到 **pre-release** 版本。
   - 有人提到过去 **2025** 年的一些记录未能改善现状。
- **用户需要故障排除建议**：一位用户对某个在 Cursor 中运行良好但在其他情况下出现问题的问题表示沮丧，并请求故障排除。
   - 另一位用户建议通过 [codeium.com/support](https://codeium.com/support) 联系支持团队寻求帮助。
- **VPN 解决连接问题**：一名成员指出，在使用 **VPN** 时应用程序运行正常，这表明可能存在网络相关问题。
   - 这意味着某些连接问题可能具有地域针对性。
- **寻找全栈开发实习生**：一位成员宣布他们正在**寻找全栈开发实习生**，在社区内提供了机会。
   - 这反映了持续招聘新人才的努力。
- **期待更多公告**：一位用户暗示今天可能还有其他公告，特别是针对 **NYC** 以外的用户，预示着更多消息即将到来。
   - 这一言论引发了关于成为令人兴奋的 **Wave 3** 发布内容一部分的猜测。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1339284880621568034)** (244 条消息🔥🔥): 

> `AI 生成决策的担忧，Windsurf 聊天问题，MCP 服务器功能，Cascade 性能问题，功能请求与建议` 


- **对 AI 生成决策的担忧**：用户对 AI 生成决策的可靠性表示了极大的担忧，指出持续的错误可能导致潜在的经济损失。
   - 社区讨论强调了分析 AI 输出的重要性，并确认了 AI 一致性方面存在的持续问题。
- **Windsurf 聊天稳定性问题**：多位用户报告 Windsurf 聊天频繁卡死、对话历史丢失，以及对工作流造成的重大干扰。
   - 建议包括重新加载应用程序和提交 Bug 报告，以解决这些关键的稳定性问题。
- **MCP 服务器可见性问题**：部分用户在更新 Windsurf 后无法找到 MCP 服务器选项，从而触发了重新加载窗口等故障排除步骤。
   - 已确认刷新界面通常会使 MCP 设置按预期显示。
- **Cascade 性能与可用性问题**：用户报告 Cascade 模型存在性能迟缓和崩溃问题，通常需要强制重启才能恢复功能。
   - 用户分享了持续的挫败感，特别是关于响应能力不足以及运行期间 CPU 占用率升高的问题。
- **功能请求与建议**：用户的反馈强调了 Windsurf 需要更多可定制功能，例如 Markdown 导出选项和专门的 Prompt。
   - 鼓励社区成员在官方反馈平台提交请求，以便在未来的更新中予以考虑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://marketplace.visualstudio.com/items?itemName=avli.clojure">Clojure&#32;-&#32;Visual&#32;Studio&#32;Marketplace</a>: Visual&#32;Studio&#32;Code&#32;扩展&#32;-&#32;为&#32;Visual&#32;Studio&#32;Code&#32;提供&#32;Clojure&#32;nREPL&#32;支持</li><li><a href="https://shitposting.pictures/ElRlAJulppNd">一张人工精选的恶搞图片</a>: 未找到描述</li><li><a href="https://codeium.canny.io/">Codeium Feedback</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.canny.io/feature-requests">Feature Requests | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://www.youtube.com/watch?v=OIV1vKm59Xg">Windsurf Wave 3 Updates: Tab to Jump, MCP, Custom App Icons, Turbo Mode &amp; More</a>: Windsurf Wave 3 更新来了！🚀 查看让 Windsurf 更加强大的最新功能：Tab 键跳转 ⏩ 轻松在文件内导航以进行...</li><li><a href="https://status.codeium.com">Codeium Status</a>: 未找到描述</li><li><a href="https://directory.llmstxt.cloud">llms.txt directory</a>: 未找到描述</li><li><a href="https://mintlify.com/blog/simplifying-docs-with-llms-txt">Simplifying docs for AI with /llms.txt</a>: 为什么我们正在为 LLM 处理文档提供更好的方式。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1339289598458859551)** (115 条消息🔥🔥): 

> `Qwen-2.5 VL 模型性能问题、模型上传与兼容性、在 LM Studio 中使用模板、GPU 使用与规格、LM Studio 错误与故障排除` 


- **Qwen-2.5 VL 模型性能问题**：用户报告 **Qwen-2.5 VL** 模型存在响应速度慢和内存管理问题，特别是在发送后续 Prompt 时，会导致明显的延迟。
   - 模型在 Prompt 之后 **内存占用激增**，这表明它可能依赖于 SSD 而非高效的 VRAM，这在高配置机器上尤为明显。
- **模型上传与兼容性见解**：一位用户询问了关于上传与 **Speculative Decoding** 相关的模型时遇到的困难，发现尽管使用了最新版本，其模型仍不兼容。
   - 故障排除显示，用户需要调整设置并确保选择了兼容的模型，才能使 **speculative decoding** 功能正常运行。
- **在 LM Studio 中使用模板**：关于在系统 Prompt 中粘贴 Jinja 模板的问题得到了澄清：模板应放置在 **LM Studio** 的不同板块中。
   - 分享了截图以帮助用户在界面中进行 **模板管理 (template management)**。
- **关于 GPU 规格的讨论**：用户对 GPU 兼容性和规格表示关注，特别是关于 **Tesla K80** 型号及其运行能力。
   - 关于 PCIe 和 SXM2 使用的查询凸显了对旧款 GPU 特性如何适配现代配置的困惑。
- **LM Studio 错误与故障排除**：一位用户报告在运行查询时出现 "error: received prediction-error" 消息，引发了关于更新 **LM Studio** 版本和运行时的讨论。
   - 反馈包括建议检查硬件兼容性，因为部分用户因缺乏 AVX2 指令集支持而遇到问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://installers.lmstudio.ai/win32/x64/0.3.6-8/LM-Studio-0.3.6-8-x64.exe">未找到标题</a>：未找到描述</li><li><a href="https://www.ebay.com/itm/116443919451">NVIDIA TESLA K80 24GB GDDR5 GPU GRAPHICS CARD   699-22080-0200-511 no cable  | eBay</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/system-requirements">System Requirements | LM Studio Docs</a>：LM Studio 在 Mac (M1/M2/M3/M4)、Windows (x64/ARM) 和 Linux (x64) 上支持的 CPU、GPU 类型</li><li><a href="https://github.com/lmstudio-ai/mlx-engine/issues">lmstudio-ai/mlx-engine</a>：👾🍎 适用于 LM Studio 的 Apple MLX 引擎。通过在 GitHub 上创建账号为 lmstudio-ai/mlx-engine 的开发做出贡献。</li><li><a href="https://www.ebay.com/itm/275857855418">Nvidia P100-SXM2-16GB P100 PCIe 16 GB Tesla GPU  | eBay</a>：未找到描述</li><li><a href="https://v0.dev">v0 by Vercel</a>：与 v0 聊天。通过简单的文本 Prompt 生成 UI。复制、粘贴、交付。</li><li><a href="https://www.reddit.com/r/KoboldAI/comments/1iodziq/rombollmv30qwen32b_release_and_q8_0_quantization/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/430">Markdown input is rendered instead of displayed as raw text in chat · Issue #430 · lmstudio-ai/lmstudio-bug-tracker</a>：LM Studio 版本？LM Studio 0.3.9 (Build 6) 操作系统？Windows 11 Bug 内容？当用户以 Markdown 格式输入文本（例如 # 标题、斜体、加粗）时，它会被渲染...</li><li><a href="https://web.archive.org/web/20250110120850/https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>：在你的电脑上本地运行 Llama, Mistral, Phi-3。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1339286663083462737)** (120 messages🔥🔥): 

> `LLM 推理性能、GPU 对比、AI 硬件发展、Intel vs AMD CPU、游戏 vs 推理` 


- **用于 LLM 任务的 K80 GPU**：一位用户考虑购买一块 60 美元的 Tesla K80 PCIe（带 24GB VRAM）用于一个 8B LLM 项目，但对电源需求和适配器兼容性表示担忧。
   - 讨论建议虽然 K80 价格实惠，但许多人发现其配置存在问题，建议可能改用 GTX 1080 Ti。
- **推理性能预期**：用户讨论了不同 GPU 的预期性能，如果配置得当，Tesla K80 预计在运行 R1 Llama 8b Q4_K 时可达到约 30 tokens per second。
   - 担忧主要集中在 K80 较旧的架构上，这可能会限制其与较新选项相比的性能。
- **用于 Optiplex 的 Amazon AD 连接器**：一位寻求升级 Dell Optiplex 7020 的用户考虑使用 PSU 适配器来支持 Tesla K80 的电源需求，这超出了系统的标准容量。
   - 这种设置引发了其他人指出的潜在兼容性和性能问题，建议在操作前保持谨慎。
- **可互换 GPU 的优势**：对话强调了根据能效与 VRAM 需求选择 GPU 的偏好，指出对于某些 AI 任务，GTX 1080 Ti 可能是更合理的选择。
   - 还提到了在购买前租用 GPU 进行基准测试（benchmarks）是衡量性能的一种实用方法。
- **SanDisk 推出 HBF 内存**：SanDisk 推出了一种新型高带宽闪存（HBF），可在 GPU 上实现 4TB 的 VRAM 容量，目标是寻求高带宽和低功耗需求的 AI 推理应用。
   - 这种创新的内存解决方案定位为未来 AI 硬件开发中传统高带宽内存（HBM）的潜在替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/pc-components/dram/sandisks-new-hbf-memory-enables-up-to-4tb-of-vram-on-gpus-matches-hbm-bandwidth-at-higher-capacity">SanDisk 的新型高带宽闪存可在 GPU 上实现 4TB VRAM，在高容量下匹配 HBM 带宽</a>：为 AI GPU 配备 4TB 内存。</li><li><a href="https://videocardz.com/newz/amd-reportedly-working-on-gaming-radeon-rx-9000-gpu-with-32gb-memory">(已更新) 据报道 AMD 正在开发具有 32GB 显存的游戏显卡 Radeon RX 9070 XT - VideoCardz.com</a>：来自 Chiphell 的新传闻称，据称 Radeon RX 9000 显卡的显存容量是 RX 9070 的两倍...</li><li><a href="https://github.com/Nicoolodion/RTX-3070-16GB-GUIDE">GitHub - Nicoolodion/RTX-3070-16GB-GUIDE: 将 RTX 3070 改装为 16 GB VRAM 的指南</a>：将 RTX 3070 改装为 16 GB VRAM 的指南。通过在 GitHub 上创建账户为 Nicoolodion/RTX-3070-16GB-GUIDE 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=L1NPFFRTzLo">NVIDIA RTX 5090 PCIe 5.0 vs. 4.0 vs. 3.0 x16 扩展基准测试</a>：赞助商：Amazon 上的 Arctic Liquid Freezer III - https://geni.us/NrMtDT。此基准测试对比了 NVIDIA RTX 5090 GPU 上的 PCIe 版本差异。我们正在测试...</li><li><a href="https://youtu.be/COcHHX2MdKs">Pciex16 vs x8 vs x4 - 游戏测试。</a>：Pci express x16 vs x8 vs x4 - 使用 RTX 3070 进行测试，1440p。测试详情：在 2560 x 1440 分辨率下测试。室内环境温度 - 30 度，PCIe 版本 3.0，CPU...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

shindeirou: 有人知道 nvjet 是在哪个 toolkit 版本引入 cublas 的吗？
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1339404750290030673)** (8 条消息🔥): 

> `PyTorch Profiler Tracing, Fused MM Activation with Triton, Triton GEMM Performance, Autotuning in Triton` 


- **PyTorch Profiler 显示 10ms 间隔**：一位成员分享了 PyTorch Profiler 的追踪结果分析，指出连续行之间存在 **10ms 间隔**，特别是在 `column_sum` 之后紧接着 CUTLASS kernel 调用。
   - 结论是如果没有 warm-up，可能会出现显著的延迟，从而影响性能。
- **通过 warm-up 解决问题**：在评估追踪信息后，得出的结论是运行一个 **for 循环** 可以消除与 warm-up 相关的延迟问题。
   - 该成员表示，这种方法将缓解之前观察到的执行时间中的 bubble。
- **在 Triton 中实现 Fused MM 激活**：一位成员询问了针对非对称矩阵（维度为 **M=2500**, **N=512**, **K=512**）的最快 tiled MM kernel，并参考了 Triton [MM tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 进行指导。
   - 他们强调，理解正确的 **block_size** 选择（应为 16 的倍数）对于获得最佳性能至关重要。
- **建议使用 A8W8 GEMM kernel**：针对 Fused MM 的咨询，一位成员推荐了 **A8W8 (persistent) GEMM**，认为这是 Triton 中针对特定维度的最快选项。
   - 他们建议运行 **max-autotune**，以确定针对特定硬件需求的最佳 autotuning 设置。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1339289677009653813)** (18 条消息🔥): 

> `Blackwell GPU Tensor Memory, CUTLASS CUTE and SGEMM, NCCL Issues with Blackwell, Tensor Memory Programmer Management, Accessing GB200 GPU Resources` 


- **关于 Blackwell Tensor Memory 管理的澄清**：关于 **Blackwell GPU** 中的新 tensor memory 是硬件管理还是软件管理存在争议；一些用户声称它完全由程序员管理，并配有专门的分配函数 [详情见此](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-memory-alloc-manage-instructions)。另一位成员指出，tensor memory 的作用是在矩阵乘法中取代寄存器，而不仅仅是为了取代 shared memory。
- **关于 CUTLASS CUTE 功能的询问**：一位用户对 **sgemm_v1.cu** 提出了疑问，注意到多线程在内存上的操作及其在 128x8 tile 结构中的整体工作流。他们寻求关于多余线程的作用、内存访问重叠以及计算过程中线程映射的澄清。
- **使用 Blackwell GPU 时的 NCCL 错误**：一位成员报告在尝试使用 Blackwell GPU 进行分布式训练时遇到了 **NCCL** 错误，即使使用了最新的 nightly 版本。报告的错误强调了 `invalid argument` 问题，该问题在不同的 NCCL 版本中持续存在 [提供的详情](https://link.to/nccLErrorInfo)。
- **对 Tensor Memory 效率的担忧**：成员们讨论了 tensor memory 如何适应 **sparsity**（稀疏性）和 **microtensor scaling** 相关的潜在低效问题，这可能导致容量浪费。一位成员强调，如果 256KiB 的可用容量中使用了 250KiB，将使得在 streaming multiprocessors 上放置累加器变得复杂。
- **获取 GB200 GPU 的挑战**：一位用户对获取 **GB200 GPU** 访问权限的困难表示沮丧，并指出潜在供应商缺乏回应。有人提出了关于替代供应商的建议，并提到了 **LLM inference** 的高需求以及等待队列的问题。



**提到的链接**：<a href="https://x.com/lambdaapi/status/1890028876954489125?s=46">Lambda (@LambdaAPI) 的推文</a>：我们只知道我们的 NVIDIA HGX B200 没问题 🙂

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1339279962502463714)** (7 条消息): 

> `SymPy for Backward Pass, Torch Compile for Optimization, Fast Hadamard Transform in Quantized Attention, Gradient Formula Simplification, Issues with gradgradcheck()` 


- **SymPy 可能简化反向传播推导**：一位成员对使用 [SymPy](https://www.sympy.org/en/index.html) 推导算法的反向传播（backward pass）表示好奇，认为这有助于管理复杂性。
   - 他们似乎对展示其在代码中实际应用的示例感兴趣。
- **使用 torch.compile 进行图优化**：有人建议利用带有 `TORCH_LOGS=aot_graphs` 的 `torch.compile` 来优化计算图，以获得更好的性能。
   - 另一位成员认可了这一技巧，但对与手写图相比的优化程度表示担忧。
- **用于高效 Attention 的快速 Hadamard 变换**：有人提出疑问，为什么某些量化 Attention 方法需要 [Fast Hadamard Transform](https://github.com/Dao-AILab/fast-hadamard-transform) 来保证性能，而像 SageAttention 这样的方法则不需要。
   - 他们讨论了最近一篇提出改进现有方法的论文，重点介绍了量化技术和性能指标。
- **梯度公式简化的复杂性**：一位成员纠正了另一位的理解，确认对推导实际的 *gradient formula* 感兴趣，但对能否手动简化表示不确定。
   - 他们提到了在使用 `gradgradcheck()` 时遇到的问题，涉及意外的输出行为——这可能表明在维持准确的中间输出方面存在复杂性。
- **需要澄清 gradgradcheck() 中的输出检查**：有人对 `gradgradcheck()` 在返回零矩阵时的行为表示担忧，认为它检查的是中间输出而非最终输出。
   - 讨论显示了澄清这些点的意图，并考虑如果问题持续存在，将在 GitHub 上进行后续跟进。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/Dao-AILab/fast-hadamard-transform">GitHub - Dao-AILab/fast-hadamard-transform: Fast Hadamard transform in CUDA, with a PyTorch interface</a>: CUDA 中的快速 Hadamard 变换，带有 PyTorch 接口 - Dao-AILab/fast-hadamard-transform</li><li><a href="https://arxiv.org/abs/2411.10958">SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization</a>: 虽然线性层的量化已被广泛使用，但其在加速 Attention 过程中的应用仍然有限。为了进一步提高 Attention 计算的效率，相比于...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

iron_bound: https://arxiv.org/abs/2502.07202
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1339400326154817557)** (8 messages🔥): 

> `D-Matrix 招聘工作、Kernel 编程讲座、架构讨论、性能预测、编程模型开发` 


- **D-Matrix 寻求 Kernel 开发人才**：D-Matrix 正在积极为其 Kernel 开发工作招聘人才，并邀请具有 CUDA 经验的人士联系并探索其独特技术栈中的机会。
   - 鼓励潜在候选人联系 [LinkedIn 上的 Gaurav Jain](https://www.linkedin.com/in/gauravjain14/)，以深入了解其创新的硬件和架构。
- **提议举办关于 D-Matrix 编程的讲座**：有人提议安排一场讲座来介绍 D-Matrix 编程，旨在吸引潜在的招聘对象。
   - Gaurav 对这一想法表示热烈欢迎，并指出该频道是进行此类讨论的理想场所。
- **技术讲座筹备计划**：Gaurav 确认愿意讨论 D-Matrix 的架构和编程模型，该模型目前处于早期开发阶段。
   - 他计划在结束为期三周的美国境外旅行回国后，协调讲座的具体细节。
- **D-Matrix 的性能展望**：D-Matrix 的 Corsair stack 旨在实现最优的速度和能效，声称在规模化推理经济学方面具有变革潜力。
   - 性能预测突显了其相对于 H100 GPU 的竞争优势，强调了 AI 领域的可持续解决方案。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://jobs.ashbyhq.com/d-Matrix/f78a6ec1-b881-401b-aa39-03f269d9fa10">Software Engineer, Senior Staff Kernels</a>：软件工程师，高级资深 - Kernels</li><li><a href="https://www.d-matrix.ai/)">d-Matrix Corsair. Built For Generative AI | d-Matrix AI</a>：d-Matrix 为规模化生成式 AI 提供全球最高效的 AI 计算解决方案
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1339329377623605309)** (30 messages🔥): 

> `CUDA 代码结构、错误处理最佳实践、CUDA 内存清理、Kernel 启动索引、C vs C++ CUDA 开发` 


- **寻求 CUDA 代码结构反馈**：一位用户在 [GitHub](https://gist.github.com/saptarshichaudhuri/4c3c63448279c8b87ba2fe5ce83d8de9) 上分享了一个矩阵乘法示例，寻求关于其 CUDA 代码结构的反馈，特别是围绕**错误处理**和**内存清理**的实践。他们在学习 **CUDA 编程基础**时寻求**建设性的指导**。
- **错误处理的一致性**：审阅者指出用户代码中 `cudaCheckError` 宏的使用不一致，建议对所有 CUDA 调用进行一致的错误检查。他们强调，如果错误不可恢复，显式的资源清理可能不是必需的，因为操作系统/驱动程序可以处理。
- **Kernel 启动索引方案问题**：讨论中提到了一个 Kernel 索引方案产生重复索引的问题，导致矩阵的某些行未被计算。将 **blockDim** 从 **(2, 2, 1)** 更改为 **(4, 1, 1)** 解决了该问题，突显了理解 **grid 布局**的必要性。
- **资源管理的最佳实践**：审阅者指出，虽然有效管理资源是良好的习惯，但如果程序因错误退出，显式清理的需求就会降低，因为操作系统会处理。他们还提到，在 **main()** 结束时释放资源可以实现代码重用，而无需担心内存泄漏。
- **CUDA 中 C 与 C++ 的考量**：一位用户询问在编写生产级 CUDA 代码时对 **C 或 C++** 的偏好，对此得到的澄清是 CUDA 通常与 C++ 开发保持一致，以便更好地兼容当前实践。他们认识到保持 **C++ 导向**对于在学习并行编程时保持更新的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/saptarshichaudhuri/4c3c63448279c8b87ba2fe5ce83d8de9">Sample matrix multiplication - CUDA</a>：CUDA 矩阵乘法示例。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/nvidia/cccl">GitHub - NVIDIA/cccl: CUDA Core Compute Libraries</a>：CUDA 核心计算库。通过创建账号为 NVIDIA/cccl 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1339287560803061893)** (8 条消息🔥): 

> `CUDA 内存模型困惑，分块矩阵乘法表格错误，Tile 大小澄清，印刷材料中的拼写错误` 


- **CUDA 内存模型关注点**：一位 CUDA 初学者提出了一个关于代码片段违反 **C++ 内存模型**的问题，原因是 scan/prefix sum 示例中缺少 thread fence。
   - 他们寻求澄清 CUDA 文档是否承认了这一疏忽，并收到了社区成员褒贬不一的回复。
- **分块矩阵乘法表中的拼写错误**：一位成员指出分块矩阵乘法相关表格的**第 7 列**可能存在错误，根据他们的分析，操作数是不正确的。
   - 另一位参与者确认了这一点，指出**第 7 列**仅仅是重复了**第 4 列**的索引，并指出了在线 PDF 中的多处拼写错误。
- **分块矩阵乘法中 Tile 大小的澄清**：一位成员询问分块矩阵乘法中的 **tiles** 是否应该与 blocks 的大小相同。
   - 对话揭示了围绕**第 4 列**准确性的一些困惑，其中一位成员反思了他们之前关于该列的错误。
- **期待第 5 版的修订**：一位参与者表示希望**第 5 版**的编辑能迅速但不草率地处理这些已确认的拼写错误。
   - 对话强调了材料中仍存在许多排版错误，并对更新表示期待。



**提到的链接**：<a href="https://stackoverflow.com/questions/79429440/cuda-memory-model-why-acquire-fence-is-not-needed-to-prevent-load-load-reorderi.">CUDA memory model: why acquire fence is not needed to prevent load-load reordering?</a>：我正在阅读《Programming Massively Parallel Processors》一书，并注意到了以下实现“多米诺风格”scan 的代码片段：&#xA;if (threadIdx.x == 0) {&#x...

  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1339316722976423978)** (2 条消息): 

> `动态量化，问题解决` 


- **torchao 中可用的动态量化选项**：成员们讨论了用户应该能够直接从 [torchao](https://link.to.torchao) 尝试 **FP8 或 INT8 动态量化**。
   - 一位成员表示，根据最近的讨论，这些选项似乎现在可以进行测试了。
- **先前问题的解决**：一位成员提到了之前的一个 issue，表示根据最近的讨论，该问题似乎已**基本解决**。
   - 另一位成员确认**这已经解决**，表明了该事项的进展。


  

---


### **GPU MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/)** (1 条消息): 

shindeirou: 抱歉伙计，没看到那条消息。那是 excalidraw + PP

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1339382879687544895)** (6 messages): 

> `Inference-time scaling, AI and compute efficiency, Documentation of hardware, Conspiracy theories in AI, Personal coding documentation` 


- **推理时扩展 (Inference-time scaling) 成为一项关键技术**：一种名为 [_inference-time scaling_](https://blogs.nvidia.com/blog/ai-scaling-laws/) 的新扩展定律在 AI 领域备受关注，它通过在推理过程中分配更多计算资源来评估多个结果，从而提高性能。
   - 这种技术被称为 _AI 推理 (AI reasoning)_ 或 _长思考 (long-thinking)_，使模型能够像人类一样解决复杂问题。
- **关注 AI 计算效率**：近年来，人们在降低 AI 计算需求方面做出了巨大努力，例如通过 [FlashAttention](https://hazyresearch.stanford.edu/blog/2023-01-12-flashattention-long-sequences) 等技术。
   - 包括 [H3](https://hazyresearch.stanford.edu/blog/2023-01-20-h3) 和 [Monarch Mixer](https://hazyresearch.stanford.edu/blog/2023-12-11-truly-subquadratic) 在内的各种模型，旨在利用现有计算资源更高效地运行 AI。
- **对硬件文档的担忧**：有人指出硬件文档通常不完善，导致在 AI 应用中难以有效使用。
   - 一位成员幽默地表示，他们不会因为别人的文档能力差而批评对方，这反映了大家对这一问题的共同感受。
- **关于故意模糊化的猜测**：讨论中提到技术文档质量差是无意的还是故意的选择，这引发了对 AI 相关阴谋论的思考。
   - 一位成员表示，在考虑这些可能性时，感觉自己像个 _阴谋论者 (conspiracy theorist)_。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://hazyresearch.stanford.edu/blog/2024-05-12-tk">GPUs Go Brrr</a>: 如何让 GPU 变快？</li><li><a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">Automating GPU Kernel Generation with DeepSeek&#x2d;R1 and Inference Time Scaling | NVIDIA Technical Blog</a>: 随着 AI 模型扩展其解决更复杂挑战的能力，一种名为测试时扩展 (test&#x2d;time scaling) 或推理时扩展 (inference&#x2d;time scaling) 的新扩展定律正在兴起。也被称为 AI 推理 (AI reasoning) ...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1339320713240252539)** (3 messages): 

> `FSDP, Liger Kernel, User Defined Kernels` 


- **在 Liger Kernel 上使用 FSDP 遇到困难**：一位成员表达了在尝试让 **FSDP** 与 **Liger Kernel** 协同工作数小时后的挫败感。
   - 他们正在寻求帮助，这表明可能存在误解或技术问题。
- **询问 FSDP 版本**：另一位成员询问发帖者使用的是 **FSDP 版本 1** 还是 **版本 2**。
   - 他们认为 **用户自定义 Kernel (user-defined kernels)** 应该不会有太多问题，暗示了潜在的兼容性。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1339659282496356496)** (7 messages): 

> `CUDA Kernel Optimizations, Performance Comparisons with PyTorch, cuBLAS vs. CUDA, Matrix Multiplication Techniques` 


- **CUDA Transformer Kernel 中的优化**：一位用户在他们的 Transformer CUDA Kernel 中实现了循环展开 (loop unrolling) 和 Warp 级归约 (warp-level reductions) 等各种优化，在不使用 cuBLAS 的情况下达到了 PyTorch **1/3 的性能**。
   - 尽管进行了优化，他们觉得进一步的改进空间很小，而且代码已经非常复杂了。
- **关于 cuBLAS 和 CUDA 的误解**：一位成员澄清说 **cuBLAS** 是针对矩阵乘法优化的高级 API，与他们用底层编写的 CUDA 实现形成对比。
   - 他们强调，可以在不深入研究 PTX 的情况下，用 CUDA 编写更快的矩阵乘法。
- **性能改进的潜力**：另一位用户鼓励探索 CUDA 中的其他优化，建议无需复杂的编程技术即可实现改进。
   - 他们参考了一个详细介绍适用于 GPU 实现的各种技术的资源。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/prateekshukla1108/100-daysofcuda/tree/main/day18">100-daysofcuda/day18 at main · prateekshukla1108/100-daysofcuda</a>: 为 100 天 CUDA 挑战编写的 Kernel。通过在 GitHub 上创建一个账户，为 prateekshukla1108/100-daysofcuda 的开发做出贡献。</li><li><a href="https://ppc.cs.aalto.fi/ch4/v2/">Chapter 4: V2</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1339347516361146368)** (70 条消息🔥🔥): 

> `DeepSeek-R1 与 Inference-Time Scaling，KernelBench 基准测试性能，GPU Kernel 优化挑战，Project Popcorn 与开放协作，使用 LLMs 进行系统编程` 


- **DeepSeek-R1 自动化 GPU Kernel 生成**：NVIDIA 展示了使用 [DeepSeek-R1 模型](https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940) 为 GPU 应用自动生成*数值正确的 Kernel*，并在推理过程中对其进行优化。
   - 然而，结果缺乏性能指标的细节，且有人对当前基准测试的饱和度表示担忧，特别是在更高级别上。
- **KernelBench 准确率与性能疑问**：最近的报告指出，NVIDIA 的工作流在 KernelBench 基准测试的 *Level-1 问题上达到了 100% 的准确率*，在 *Level-2 问题上达到了 96%*，但性能数据仍未公布。
   - 有人担心该基准测试可能已经饱和，并建议关注性能指标以衡量真正的有效性。
- **GPU Kernel 优化作为核心挑战**：讨论强调，虽然 GPU Kernel 编程是一个小众领域，但由于其对性能和资源利用的影响，在软件工程中被认为非常重要。
   - 成员们指出，优化 Kernel 可以显著节省*计算成本*和*能源消耗*，从而影响更广泛的软件工程实践。
- **Project Popcorn 的开放协作努力**：一位成员强调，一旦初始“任务”发布（目标是在 3 月 16 日左右的 GTC 期间），对 Project Popcorn 的贡献将变得更加容易。
   - 目前正努力以*开源*方式构建该项目，尽管某些方面（如数据发布）需要更正式的批准。
- **LLMs 在代码生成方面的挑战**：讨论了利用 LLMs 进行代码生成的尝试，重点关注应用编程与系统编程挑战之间的鸿沟——特别是在性能优化方面。
   - 一位参与者提出，精心的设计和高性能目标使 Kernel 优化成为衡量 LLM 推理能力的极具吸引力的基准。



**提及的链接**：<a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/?ncid=so-link-284103&linkId=100000338909940">使用 DeepSeek-R1 和 Inference Time Scaling 自动化 GPU Kernel 生成 | NVIDIA 技术博客</a>：随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为 Test-time Scaling 或 Inference-time Scaling 的新缩放法则正在兴起。也被称为 AI 推理 ...

  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1339393220370763857)** (47 条消息🔥): 

> `Graph Coloring Problems, Reasoning-Gym Dataset Evaluations, Futoshiki Puzzle Dataset, Game of Life Outputs, Standardization of Reporting Scores` 


- **Miserlou 提交的 Graph Coloring Problems PR**：提交了一个关于 **Graph Coloring Problems**（图着色问题）的新 Pull Request，要求实现一种相邻顶点不共享相同颜色的着色方法，具体细节见 [此 PR](https://github.com/open-thought/reasoning-gym/pull/120)。
   - 讨论强调这些更改将提高输出的机器兼容性。
- **MATH-P-Hard 上的性能问题**：成员们注意到在 **MATH-P-Hard** 上性能显著下降，表明模型存在**对原始推理模式的偏见**，这影响了模型在更难示例上的有效性，详见[此推文线程](https://x.com/kaixuanhuang1/status/1889366696403804507?s=46&t=E50tvry4ancj_GB5agsQ7w)。
   - 提到的好消息是，模型在面对较简单的扰动时表现稳健。
- **Reasoning-Gym 数据集更新**：贡献了包括 **Futoshiki 谜题数据集**在内的内容，旨在提供更简洁的求解器和改进的逻辑框架，详情见 [此 PR](https://github.com/open-thought/reasoning-gym/pull/60)。
   - 此外，数据集正在通过统一的 Prompt 进行标准化，以简化评估流程。
- **评估输出存储设置**：创建了一个**评估仓库**（evaluation repository）用于存储输出的 JSON 文件和相关脚本，以保持主仓库的整洁，这是在讨论如何维持组织结构后由多名成员提议的。
   - 讨论还包括使用一个**中央 Google 表格**来追踪评估和结果，以便更好地进行协作监督。
- **评分报告标准化**：呼吁建立一种确定各数据集平均分的标准方法，建议使用 **50 个样本**以确保报告的一致性，正如成员讨论中所强调的那样。
   - 目标是确保在评估各种模型的排行榜上具有可靠性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/kaixuanhuang1/status/1889366696403804507?s=46&t=E50tvry4ancj_GB5agsQ7w">Kaixuan Huang (@KaixuanHuang1) 的推文</a>：我们观察到 MATH-P-Hard 上的性能显著下降，而 MATH-P-Simple 上的性能下降可以忽略不计。这表明模型偏向于原始的推理分布...</li><li><a href="https://github.com/open-thought/reasoning-gym-eval/">GitHub - open-thought/reasoning-gym-eval: reasoning-gym 任务数据集的 LLM 补全集合</a>：reasoning-gym 任务数据集的 LLM 补全集合 - open-thought/reasoning-gym-eval</li><li><a href="https://x.com/teortaxestex/status/1889774968969294010">Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex) 的推文</a>：令人惊叹的论文。递归自我改进的一个奇招：模型迭代地标记自己的训练数据，并从逐渐变难的示例中学习。必须 1) 生成适当难度的问题...</li><li><a href="https://docs.google.com/spreadsheets/d/1qk2BgxzfRZzTzMQnclCr47ioykgltbGkMJUHO2sH6Gw/edit?gid=0#gid=0">reasoning-gym-eval</a>：未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/125">Miserlou 修改 Game of Life 输出格式 · Pull Request #125 · open-thought/reasoning-gym</a>：要求更具机器友好性的输出格式，附带说明和示例。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/120">Miserlou 添加 Graph Coloring Problems · Pull Request #120 · open-thought/reasoning-gym</a>：请为该图提供一种着色方案，使得每个顶点都不与相同颜色的顶点相连。该图具有以下属性：边：[(0, 2), (0, 4), (0, 7), (0, 8), (1,...</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/60">olliestanley 添加 Futoshiki 谜题生成器 · Pull Request #60 · open-thought/reasoning-gym</a>：关闭 #54。现有的求解器很乱且难以理解，所以我实现了一个新的。即使在这段代码中，逻辑规则也不容易遵循，但非常值得，因为它们加速了...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1339290200454598748)** (133 条消息🔥🔥): 

> `Tulu 模型中的 GRPO vs PPO, Anthropic 即将推出的 Claude 模型, DeepHermes-3 预览版发布, EnigmaEval 推理挑战, 越狱挑战结果`

- **GRPO 在 Tulu 模型中表现优于 PPO**：Costa Huang 宣布，从 **PPO** 切换到 **GRPO** 使 Tulu 流线的性能提升了 **4 倍**，新的 **Llama-3.1-Tulu-3.1-8B** 模型在 MATH 和 GSM8K 测试中表现出更好的结果。
   - 转向 GRPO 展示了相比去年秋季发布的先前模型的显著增强。
- **Anthropic 的 Claude 模型即将问世**：Anthropic 的下一个 Claude 模型将把传统的 LLM 与推理 AI 相结合，允许开发者通过 **滑动刻度 (sliding scale)** 调整推理级别，在各种基准测试中可能超越 OpenAI 的 o3-mini-high。
   - 这种创新方法标志着商业编程任务在模型训练和操作能力方面的转变。
- **DeepHermes-3 预览版发布**：Nous Research 发布了 **DeepHermes-3**，这是一款结合了推理与语言处理的 LLM，能够切换长思维链 (chains of thought) 以提高准确性，代价是计算需求增加。
   - 由于基准测试得分的差异，该模型的性能指标以及与 Tulu 模型的比较引发了疑问。
- **EnigmaEval 提出新的推理挑战**：Dan Hendrycks 宣布了 **EnigmaEval**，这是一套复杂的推理挑战，AI 系统在其中表现挣扎，在普通谜题上的得分低于 **10%**，在 MIT 级别的挑战中得分为 **0%**。
   - 引入这种严格的评估旨在突破 AI 推理能力的边界。
- **Anthropic 越狱挑战赛结果揭晓**：在 Anthropic 组织的越狱挑战赛中，参与者发送了超过 **300,000 条消息**，并生成了一个 **通用越狱 (universal jailbreak)**，获胜者获得了 5.5 万美元奖金，展示了极高的参与度。
   - 该挑战反映了在提高 AI 模型安全措施方面的持续努力，特别是新引入的宪法分类器 (constitutional classifiers)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://openweight.org/">Open Weight Definition (OWD)</a>：未找到描述</li><li><a href="https://x.com/NousResearch/status/1890148000204485088">Nous Research (@NousResearch) 的推文</a>：介绍 DeepHermes-3 Preview，这是一款统一了推理和直觉语言模型能力的新型 LLM。https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview DeepHermes 3 构建于...</li><li><a href="https://opensource.org/ai/open-source-ai-definition">开源 AI 定义 (The Open Source AI Definition) – 1.0</a>：1.0 版本前言 为什么我们需要开源人工智能 (AI)？开源已经证明，在消除学习、使用、共享的障碍后，每个人都能获得巨大的利益...</li><li><a href="https://opensourcealliance.org/">开源联盟 (Open Source Alliance)</a>：团结全球开源社区，塑造软件自由的未来。</li><li><a href="https://x.com/lmarena_ai/status/1889741530757210524">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：人们都在询问哪些类型的编程语言？根据我们检索到的文件类型，Python 和 Markdown 是目前为止人们提问中最常见的相关语言。</li><li><a href="https://x.com/gm8xx8/status/1889879054406336544">𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文</a>：蒸馏缩放定律 (Distillation Scaling Laws)  这项研究提出了一种蒸馏缩放定律，用于根据教师模型和学生模型之间的计算资源分配来估计学生模型的性能。它提供了指导方针...</li><li><a href="https://x.com/natolambert/status/1889730488199209393">Nathan Lambert (@natolambert) 的推文</a>：Costa 正试图让 GRPO 在没有 Bug 的情况下飞速运行，结果我们的性能比去年秋天发布的 Tülu 模型好得多。从 PPO 切换到 GRPO 使增益翻了 4 倍...</li><li><a href="https://x.com/stalkermustang/status/1890144205038842219">Igor Kotenkov (@stalkermustang) 的推文</a>：宝贝快醒醒，AIME 2 LLM 结果出炉了。o3-mini 是王者，Gemini 完蛋了，R1 表现“尚可”。</li><li><a href="https://x.com/janleike/status/1890155264101486792">Jan Leike (@janleike) 的推文</a>：@caleb_parikh 他们发送了 7,867 条消息，并将其中 1,408 条传给了自动评分器。我们估计他们总共可能在这上面花费了超过 40 小时。</li><li><a href="https://x.com/Dorialexander/status/1890122850339811642">Alexander Doria (@Dorialexander) 的推文</a>：@TheXeophon 新机构。https://opensourcealliance.org/ 定义刚刚在峰会期间发布并得到了合理的宣传（背景：似乎是因为意见分歧而从 Open Source Initiative 分叉出来的...</li><li><a href="https://x.com/theinformation/status/1889831938346852380">The Information (@theinformation) 的推文</a>：独家：Anthropic 预计 2027 年营收将飙升至 345 亿美元。作为 OpenAI 的主要挑战者，Anthropic 预计 2027 年营收将高达 345 亿美元，高于今年的 37 亿美元...</li><li><a href="https://x.com/dylan522p/status/1889939130668417225">Dylan Patel (@dylan522p) 的推文</a>：新的 OpenAI 模型规范允许性内容。就在我们说话的时候，数百万第三世界的标注员正被分配最古怪的角色扮演任务。成千上万的 AI 裁判正在被启动以...</li><li><a href="https://x.com/NeginRaoof_/status/1889739171826377008">Negin Raoof (@NeginRaoof_) 的推文</a>：宣布 OpenThinker-32B：从 DeepSeek-R1 蒸馏出的最佳开源数据推理模型。我们的结果表明，经过验证的 R1 标注的大型、精心策划的数据集可以产生 SoTA 推理模型...</li><li><a href="https://x.com/steph_palazzolo/status/1890058003493343453">Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：Anthropic 的下一个 Claude 模型即将推出。它将是传统 LLM + 推理 AI 的结合，开发者可以通过以 Token 为单位的滑动条来调整其推理程度...</li><li><a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">使用 DeepSeek-R1 和推理时间缩放自动化 GPU Kernel 生成 | NVIDIA 技术博客</a>：随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为测试时间缩放 (test-time scaling) 或推理时间缩放 (inference-time scaling) 的新缩放定律正在出现。也被称为 AI 推理...</li><li><a href="https://fxtwitter.com/neilhoulsby/status/1889952572431122891">Neil Houlsby (@neilhoulsby) 的推文</a>：我很高兴地宣布我加入了 Anthropic 瑞士分部！🇨🇭 Anthropic 正在苏黎世设立新办公室，扩展其全球业务。我非常兴奋能在这里组建团队...</li><li><a href="https://x.com/oliviergodement/status/1889789220664852610">Olivier Godement (@oliviergodement) 的推文</a>：@mikeknoop API 将支持 o3！我们将提供诸如推理力度 (reasoning effort) 之类的调节选项，以充分发挥新前沿模型的性能。我们正在研究打包方案...</li>

ge o3 在更广泛的 GPT-5 系统中，...</li><li><a href="https://fxtwitter.com/PiotrPadlewski/status/1889960617915879614">Piotr Padlewski (@PiotrPadlewski) 的推文</a>：很高兴加入 @neilhoulsby 在 Anthropic 位于苏黎世的新办公室，从事 multimodal 研究。在 Reka 度过了一段不可思议的旅程后，是时候开启新篇章了！很感激有机会为 LLM/VL 做出贡献...</li><li><a href="https://x.com/togethercompute/status/1889743684977168547">Together AI (@togethercompute) 的推文</a>：自发布 DeepSeek-R1 以来，我们看到大批公司寻求在生产中部署 reasoning models——但高效扩展它们仍然是一个挑战。今天，我们正在扩展我们的 ultr 之外...</li><li><a href="https://x.com/btibor91/status/1890061119274004829">Tibor Blaho (@btibor91) 的推文</a>：The Information 报道称，Anthropic 将在未来几周发布一款混合 AI 模型，它可以在快速响应和 deep reasoning 之间切换，并为开发者提供独特的滑动标尺来控制...</li><li><a href="https://x.com/swyx/status/1889929794936295426?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">swyx 🔜 @aidotEngineer NYC (@swyx) 的推文</a>：转发这条友好的回复。我觉得最糟糕的情况是 {COMPETITOR} 似乎正在走向的那条路，即训练不同的模型 + 使用 model router 来营造出 agin 的假象...</li><li><a href="https://x.com/TheXeophon/status/1889762840384266578">Xeophon (@TheXeophon) 的推文</a>：随着 GPT-5 成为一个（更加）黑盒的系统，我希望学术界最终能从付费产品测试员转变为专门使用 open models</li><li><a href="https://x.com/OpenAI/status/1889822643676913977">OpenAI (@OpenAI) 的推文</a>：两个你会喜欢的更新——📁 OpenAI o1 和 o3-mini 现在在 ChatGPT 中支持文件和图像上传 ⬆️ 我们为 Plus 用户将 o3-mini-high 的限制提高了 7 倍，达到每天 50 次</li><li><a href="https://x.com/tsarnick/status/1889913600325902704">Tsarathustra (@tsarnick) 的推文</a>：Elon Musk 表示 Grok 3 将在“一两周内”发布，它“聪明得吓人”，展示出的 reasoning 技能优于任何已发布的 AI 模型</li><li><a href="https://tenor.com/view/old-boomer-history-84years-many-years-ago-gif-18534104">Old Boomer GIF - Old Boomer History - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/DanHendrycks/status/1890091724594393140">Dan Hendrycks (@DanHendrycks) 的推文</a>：我们正在发布 EnigmaEval，这是一个长而复杂的 reasoning 挑战集合，需要多人花费数小时或数天才能解决。最好的 AI 系统在普通谜题上的得分低于 10%，而对于...</li><li><a href="https://fxtwitter.com/janleike/status/1890141865955278916">Jan Leike (@janleike) 的推文</a>：我们 jailbreaking 挑战的结果：经过 5 天、超过 300,000 条消息以及估计 3,700 个集体小时，我们的系统被攻破了。最终有 4 位用户通过了所有关卡，1 位发现了通用 jailbreak。我们...</li><li><a href="https://x.com/nrehiew_/status/1889737259835969735">wh (@nrehiew_) 的推文</a>：在相同的 TULU3 数据集上，GRPO > PPO。这里的直觉是什么？GRPO 难道就是天命所归的 RL 算法吗？引用 Costa Huang (@vwxyzjn) 🔥 allenai/Llama-3.1-Tulu-3-8B（使用 PPO 训练）-> a...</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Sam Altman (@sama) 的推文</a>：OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 对你来说能“直接可用”；我们...</li><li><a href="https://x.com/pitdesi/status/1889830141116948753">Sheel Mohnot (@pitdesi) 的推文</a>：Anthropic 预计今年基本收入为 22 亿美元（高于 2024 年的约 5 亿美元），并预计 2027 年达到 120 亿美元。OpenAI 的收入约为 Anthropic 的 5 倍，预计 2027 年达到 440 亿美元。为了实现这些预测，他们必须...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1339669334880030783)** (5 条消息): 

> `notebookLM 性能, GPT-5 模型界面` 


- **notebookLM 在基础任务上表现挣扎**：一位用户表达了挫败感，称 **notebookLM** 的表现就像一个过时的模型，虽然响应迅速，但在从 **24 份 PDF** 的 benchmark 中创建 markdown 表格等任务上失败了。
   - 用户的担忧凸显了 **markdown 格式化**的问题，促使人们考虑改用 **Deep Research**。
- **对 GPT-5 单一界面模型的担忧**：一位用户对 **Sama 关于合并 GPT-5 模型**的公告做出了反应，强调了了解哪些模型正在用于任务分配的重要性。
   - *“我很确定 notebookLM 也是这种情况，”* 他们评论道，表明产品版本已导致用户不满。

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1339712571506294844)** (2 条消息): 

> `DH3 评估指标，ImitationLearn 公司合法性` 


- **DH3 评估指标引发质疑**：笔记指出 DH3 仅针对 **'reasoning on'** 指标展示了两个特定的评估，而 **'reasoning off'** 图表则显示了所有指标。
   - 有人对他们忽略与官方 **8b distill release** 的对比表示担忧，后者的分数更高，DH3 的 **GPQA** 约为 **36-37%**，而 **r1-distill** 约为 **49%**。
- **ImitationLearn 的公司信誉受到质疑**：讨论引用了一位成员对 *ImitationLearn* 合法性的不确定，称其可能只是看起来“很酷（swaggy）”，而这也许才是最重要的。
   - 这种模糊性让社区对该公司在该领域的真实性产生了质疑。



**提到的链接**：<a href="https://fxtwitter.com/kalomaze/status/1890153665333457140">来自 kalomaze (@kalomaze) 的推文</a>：dh3 笔记 1. 他们只为 “reasoning on” 展示了这两个特定的评估；“reasoning off” 图表是唯一显示所有指标的 2. 他们没有与官方 8b di... 进行对比

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1339281605004496929)** (36 条消息🔥): 

> `受审模型讨论、OpenThinker-32B 模型发布、推理 Token 扩展、聊天机器人 Prompt 指南、社区对 RL 的评论` 


- **对受审模型的担忧**：成员们对 DeepSeek 模型的审查制度表示不满，引发了关于如何对其进行去审查（de-censor）的讨论，并提出了 OpenThinker-32B 模型等替代方案。
   - *有人指出，这些担忧在社区内很普遍*。
- **OpenThinker-32B 发布**：一款名为 **OpenThinker-32B** 的新推理模型已发布，该模型通过利用精选数据进行问题分类和推理任务，表现良好。
   - 团队对此次发布表示庆祝，并评论称在成功对齐（Alignment）模型方面取得了重大进展。
- **关于推理 Token 扩展的辩论**：关于推理 Token 在达到一定问题规模后出现下降的讨论非常热烈，观察结果表明在 **30 位数字** 左右存在限制。
   - 一位用户提醒说，这些发现是基于小样本量的，因此鼓励他人保守地解读结果。
- **聊天机器人 Prompt 的最佳实践**：一位用户分享了一份指南，强调了避免使用“老派 Prompt”（boomer prompts）的重要性，并建议使用直接的指令以及分隔符（delimiters）以确保清晰。
   - *另一位成员幽默地提到，他们感觉这些指南仿佛是在针对自己，突显了社区的参与度。*
- **社区对强化学习（Reinforcement Learning）的评论**：成员们对近期强化学习综述结果的混乱和复杂性发表了评论，反映了对需要进行整理汇总的挫败感。
   - 尽管存在挑战，但人们也认可可以从更有趣的发现中汲取有价值的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/__nmca__/status/1889741584922751092">Nat McAleese (@__nmca__) 的推文</a>: @stalkermustang @ahelkky o3 采样了许多解决方案，并使用学习到的函数来挑选最佳方案 —— 对于 codeforces，我们为每个问题采样了 1,162 个样本</li><li><a href="https://tenor.com/view/avatar-aang-aang-atla-avatar-the-last-airbender-avatar-gif-23087281">Avatar Aang Aang GIF - Avatar Aang Aang Atla - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/colin_fraser/status/1889821087623733251">Colin Fraser (@colin_fraser) 的推文</a>: @littmath 也许并非巧合，在这个阈值下，推理 Token 似乎停止随问题规模扩展</li><li><a href="https://x.com/colin_fraser/status/1889816761090072708">Colin Fraser (@colin_fraser) 的推文</a>: @littmath 取决于你对“可靠”的标准，但在 30 位数字后似乎下降得很厉害。样本量较小（每个 n 为 10 个），所以对任何观点都要持保留态度</li><li><a href="https://x.com/madiator/status/1889772019492987225">Mahesh Sathiamoorthy (@madiator) 的推文</a>: 我们意外地对模型进行了去审查！我们使用的 Qwen-instruct 是经过审查和对齐的。DeepSeek-R1 distilled 模型也是经过审查和对齐的。当我们使用数学推理数据对 Qwen 模型进行 SFT 时...</li><li><a href="https://x.com/elder_plinius/status/1890028958907089059?s=46">Pliny the Liberator 🐉 (@elder_plinius) 的推文</a>: 哈哈哈 💉💦 引用 Djah 〰️ (@Djahlor) 什么？？？@elder_plinius 是你干的吗？？</li><li><a href="https://x.com/OpenAIDevs/status/1890147300493914437">OpenAI Developers (@OpenAIDevs) 的推文</a>: 正如你们中一些人所注意到的，在使用 o 系列模型时要避免“老派 Prompt”。相反，要简单直接，并给出具体的指导方针。分隔符（xml 标签）将有助于为模型保持整洁，并且...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1339313836733497396)** (4 messages): 

> `DeepSeek Announcement, OpenAI's Roadmap Update, OLMo GitHub Issue, AI Security Reviewers, Rust's Future Value Proposition` 


- **DeepSeek 预告即将发布**：一名成员暗示明天将有关于 **DeepSeek** 的**重大**消息发布。
   - *AK* 预告了这一公告，暗示这对社区可能具有重大意义。
- **OpenAI AGI 策略的转变**：Sam Altman 透露，目前单纯扩大规模（scaling up）的策略将不再足以实现 AGI，这表明 OpenAI 在准备发布 **GPT-4.5** 和 **GPT-5** 之际，其方法论正在发生转变。
   - OpenAI 将整合其系统以提供更无缝的体验，同时解决社区对模型选择步骤的不满。
- **关于 AllenAI 创立的澄清**：GitHub 上提出的一个 Issue 质疑 **AllenAI** 是否由 **Jeff Bezos** 创立，OLMo 2 断言这不是事实。
   - 这一询问突显了对 **OLMo** 等 AI 项目起源进行澄清的必要性。
- **AI 审查器即将带来变革**：随着 AI 安全审查器准备大规模发现并修复源代码中的 Bug，关于编程技能价值缩水的问题随之而来。
   - 一位成员思考，在这些进步中，几年后 **Rust** 的价值主张（value proposition）究竟会是什么。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/untitled01ipynb/status/1889960924582457410">来自 loss (@untitled01ipynb) 的推文</a>：现在任何程序员的价值主张究竟是什么？引用 xlr8harder (@xlr8harder)：既然我们正处于 AI 安全审查器能够大规模检测并修复安全/内存 Bug 的关头……</li><li><a href="https://github.com/allenai/OLMo/issues/787">你们是由 Jeff Bezos 创立的吗？· Issue #787 · allenai/OLMo</a>：❓ 问题：Olmo2 说 AllenAI 是由 Jeff Bezos 创建的。这是真的吗？我是 OLMo 2，由艾伦人工智能研究所 (Ai2) 开发的 AI 语言模型，也被称为 "All..."</li><li><a href="https://x.com/untitled01ipynb/status/1889751694365388821">来自 loss (@untitled01ipynb) 的推文</a>：ak 看到了什么？引用 AK (@_akhaliq)：明天将有关于 DeepSeek 的重大消息发布。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1339307854124941393)** (20 messages🔥): 

> `Dwarkesh, Noam Shazeer, Jeff Dean, Podcast Interviews, Science History` 


- **Dwarkesh 与 Noam 及 Jeff 的标志性播客**：围绕 Noam Shazeer 和 Jeff Dean 在 [Dwarkesh 的播客](https://open.substack.com/pub/dwarkesh/p/jeff-dean-and-noam-shazeer?r=68gy5&utm_medium=ios)上的表现展开了讨论，成员们对其内容表示兴奋。
   - *天哪*，这一集确实是**标志性**的，听众们非常欣赏 **Noam** 的风格，尤其是他的 **OutdoorResearch 帽子**，这引发了额外的讨论。
- **对 Dwarkesh 播客的兴趣**：几位成员提到希望参加 **Dwarkesh** 的播客，强调这会很有趣，但目前不是首要任务。
   - 一位成员指出，多个人推荐过它，并提到“我可以邀请他”，引用了他们的播客经验。
- **对 Noam 帽子选择的致敬**：对话转向了幽默的方向，一位成员形容 Noam 的帽子是**标志性**的，并将其与网上找到的照片进行对比，指出这是他经常戴的那顶。
   - 另一位补充道，“我有同款帽子，但颜色不同”，展示了成员们对共同时尚选择的共鸣。



**提到的链接**：<a href="https://www.google.com/search?num=10&sca_esv=fd2f423473d1beed&sxsrf=AHTn8zqRKL3KlFd8kwS5ozut-AB-4NPovA:1739410797255&q=noam+shazeer&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWpA-dk4wpBWOGsoR7DG5zJBnsX62dbVmWR6QCQ5QEtPRqut5gkyra9fZFbsKm1oGezOI6DQjxNKZ2V8dXgJRWA_TJMoTMoaAT3sFlmqfwsFU7xKyaCESU9pcEBIOWtbh8Q57l_jotrwukFQfQsaj_ShBIVC3RtBGfnv0evqLdoaTjhVTpVso9nbb1qUVYZwxrh2LzRlg&sa=X&ved=2ahUKEwjS98CVwr-LAxWFOTQIHSa4JO4QtKgLegQIBxAB&biw=1080&bih=1084&dpr=1.33)">noam shazeer - Google 搜索</a>：未找到描述

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1339319316377763921)** (5 条消息): 

> `Aged beautifully, SnailBot` 


- **关于随时间演变的评论**：一位成员评论说某些事物 *aged beautifully*（经受住了时间的考验/随时间推移愈发经典），可能暗示随着时间的推移产生了积极的变化。
   - 这引发了与另一位成员表达认可的轻松交流。
- **SnailBot 新闻警报**：SnailBot 发布了一条针对特定受众的通知，可能表示更新或重要信息。
   - 未提供关于 SnailBot 消息内容的具体细节。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1339291957737951263)** (14 条消息🔥): 

> `Dataset perplexity evaluation, Post-training datasets, Online courses in AI and tech, Collaboration opportunities at Eleuther AI` 


- **寻求高效的数据集困惑度评估**：一位成员询问关于数据集困惑度（perplexity）评估的**高效实现**，要求能在 **multi-GPU** 系统上快速运行。
   - 另一位成员建议使用 **lm-eval-harness** 作为潜在的解决方案。
- **关于最佳后训练数据集的讨论**：有人提问关于最佳的后训练（post-training）数据集，**SmolTalk** 和 **Tulu-3** 被提及为可能的选项。
   - 对话还涉及了关于将 **reward models** 与 **SFT objectives** 结合的咨询。
- **AI 与技术学习资源**：一位巴西成员征求关于 AI 和技术/商业领域**最佳在线课程**的建议，以便保持更新。
   - 另一位成员引导他们前往特定频道获取学习机会资源。
- **有兴趣参与 Eleuther AI 的研究**：一位成员表达了对在 Eleuther AI **合作研究项目**的兴趣，并寻求如何贡献的指导。
   - 他们询问是否有任何**开放研究项目**可供参与。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1339291696856436889)** (126 条消息🔥🔥): 

> `PPO-Clip with Alternative Models, Memory Mechanisms in Models, Forgetting Transformer, Evaluation of Long Context Models, Temporal Causality in Attention` 


- **关于 PPO-Clip 与替代模型的讨论**：成员们讨论了将 [PPO-Clip](https://example.com) 应用于不同模型以生成 rollouts 的想法，并回顾了过去对话中的类似观点。
   - 一位成员根据之前的尝试，对这种方法的有效性表示怀疑。
- **探索记忆机制**：讨论强调了一位成员关于通过使用 **dummy tokens** 和为稀疏注意力机制增加额外门控来改进模型记忆的概念。
   - 讨论中出现了关于该实现如何影响模型架构中的因果性和并行化的担忧。
- **Forgetting Transformer 框架**：围绕 **Forgetting Transformer** 展开了对话，讨论了将激活函数从 sigmoid 更改为 tanh 是否会对性能产生积极影响。
   - 还提出了引入负注意力权重的想法，强调了注意力机制更趋复杂的潜力。
- **长上下文模型评估的挑战**：成员们反思了与当前长上下文模型基准测试（如 **HashHop**）相关的挑战，以及解决 1-NN 的迭代性质。
   - 针对长上下文模型方法所宣称的理论可行性提出了质疑。
- **时间因果性与注意力机制**：讨论了在**时间因果性**（temporal causality）中对向量进行搜索的有效性，强调了语言数据非平稳性（non-stationarity）的独特性。
   - 成员们辩论了 softmax attention 和门控 RNN 是否能充分模拟各种数据类型中的非平稳分布。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.06773">On the Emergence of Thinking in LLMs I: Searching for the Right Intuition</a>: 最近的 AI 进展（如 OpenAI 的新模型）正在将 LLM 转变为 LRM (Large Reasoning Models)，这些模型在推理过程中执行推理，消耗额外的时间和计算资源以获得更高质量的结果...</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows — Magic</a>: 关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://github.com/SmerkyG/gptcore/blob/main/model/experimental/memtention.py">gptcore/model/experimental/memtention.py at main · SmerkyG/gptcore</a>: 用于创建和训练尖端 LLM 的快速模块化代码 - SmerkyG/gptcore
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1339437784670277763)** (2 messages): 

> `引用 Delphi，引用 Sparsify，论文的 BibTeX 条目，GitHub 引用` 


- **关于引用 Delphi 的指导**：<@177739383070261248> 询问了引用 **Delphi** 的正确方式，考虑同时引用论文和 [GitHub 页面](https://github.com/delphi)。
   - 正如讨论中其他人所同意的，**结合引用**以实现全面的归属标注是一个好主意。
- **关于 Sparsify 引用的想法**：<@177739383070261248> 表示计划同时引用 **Sparsify GitHub 页面**，并强调多重引用能提升研究质量。
   - 使用来自 arXiv 的**自动生成 BibTeX 条目**可能有利于标准化。
- **关于 BibTeX 输出的讨论**：有人建议对常见论文使用 *arXiv 自动生成的 BibTeX 条目*，这可以简化引用流程。
   - 成员们建议为各种 **GitHub 仓库**创建基础的 BibTeX 条目，以保持一致性。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1339313037651214458)** (92 messages🔥🔥): 

> `Stable Diffusion 保存问题，切换到 Linux 使用 ComfyUI，模型推荐与性能，AI 角色设计一致性，Upwork 账号借用` 


- **Stable Diffusion 缺少自动保存功能**：一位用户意识到他们在 Stable Diffusion 中没有开启自动保存选项，并询问如何找回之前生成的图像。
   - 其他人建议检查所使用的 Web UI 版本，以确定可用的保存选项。
- **Linux 上的 ComfyUI OOM 错误**：一位从 Windows 切换到 Pop Linux 的用户在运行 ComfyUI 时遇到了显存溢出（OOM）错误，尽管之前运行成功。
   - 讨论内容包括确认系统更新和推荐驱动，强调了不同操作系统之间依赖关系的差异。
- **选择正确的 AI 模型**：一位用户表达了在不同模型间保持一致角色设计的挑战，引发了关于使用 Loras 以及 FaceTools 和 Reactor 等工具的讨论。
   - 建议包括根据任务探索不同的模型，重点关注专门为某些功能设计的模型。
- **追求放大（Upscaling）和创意工具**：用户询问 Stability 的创意放大器（creative upscaler）何时发布，有人断言该工具尚未发布。
   - 这引发了关于模型能力以及某些模型是否符合内存和性能要求的对话。
- **Upwork 账号借用咨询**：一位成员寻求借用美国 Upwork 账号用于即将开展的项目，引发了关于此类安排可行性的质疑。
   - 这引发了对“借用”账号概念及其影响的怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#black-forest-labs-flux1-models>">SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI</a>: SwarmUI（原 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调易用的强大工具、高性能和可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/mcm">mcm - 概览</a>: mcm 有 47 个公开仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1339280233894903889)** (81 messages🔥🔥): 

> `AI Agents, OpenAI 路线图更新, Apple 产品发布, DeepHermes 3 模型, Meta 的自动化合规强化工具`

- **OpenAI 的最新路线图更新**：OpenAI 分享了关于其即将推出的 GPT-4.5 和 GPT-5 模型的重大更新，旨在统一各种模型类型并简化其产品线。
   - 关键特性包括整合 O-series 模型和所有工具，为开发者和用户创造更流畅的 AI 体验。
- **Anthropic 的新 Claude 模型**：Anthropic 计划很快推出下一个 Claude 模型，该模型将结合传统的 LLM 能力与推理 AI，开发者可以通过基于 token 的滑动条进行控制。
   - 这种方法与 OpenAI 在其近期公告中采用的方法类似，表明了在 AI 模型中整合推理能力的趋势。
- **对 Apple 产品发布会的期待**：人们对 Apple 即将发布的产品充满期待，包括传闻中的新 iPhone SE 和期待已久的 Apple TV 更新。
   - Tim Cook 对此次发布进行了预热，引发了关于这些产品额外功能和升级芯片的猜测。
- **DeepHermes 3 模型亮相**：Nous Research 发布了 DeepHermes 3 的预览版，这是一款将推理与传统响应能力相结合的 LLM，增强了其性能。
   - 该模型旨在提供更高的准确性和功能性，代表了 LLM 发展中的重要一步。
- **Meta 的自动化合规强化工具**：Meta 推出了其自动化合规强化 (ACH) 工具，利用基于 LLM 的测试生成技术，通过创建未检测到的故障进行测试，从而增强软件安全性。
   - 这一先进工具旨在通过自动生成针对代码中特定故障条件的单元测试，来提高隐私合规性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/OpenAI/status/1889822643676913977">来自 OpenAI (@OpenAI) 的推文</a>：两个你会喜欢的更新——📁 OpenAI o1 和 o3-mini 现在支持在 ChatGPT 中上传文件和图像 ⬆️ 我们为 Plus 用户将 o3-mini-high 的限制提高了 7 倍，达到每天最多 50 次</li><li><a href="https://x.com/steph_palazzolo/status/1890058003493343453">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：Anthropic 的下一个 Claude 模型即将推出。它将是传统 LLM + 推理 AI 的结合，开发者可以通过以 token 衡量的滑动刻度来调整其推理程度...</li><li><a href="https://x.com/glean/status/1889706504812683728">来自 Glean (@glean) 的推文</a>：欢迎来到 Agent 时代 🚀 我们很高兴地宣布推出 𝐆𝐥𝐞𝐚𝐧 𝐀𝐠𝐞𝐧𝐭𝐬——我们的横向 Agent 环境，使员工和企业能够大规模地构建、运行、管理和治理 AI Agent...</li><li><a href="https://x.com/swyx/status/1889810524696891903">来自 swyx 🔜 @aidotEngineer NYC (@swyx) 的推文</a>：转发 @JeffDean：我很高兴能加入我的好朋友兼同事 @NoamShazeer，与 @dwarkesh_sp 进行一场超过 2 小时的对话，讨论广泛的...</li><li><a href="https://x.com/glennsolomon/status/1889717350456315960?s=46">来自 Glenn Solomon (@glennsolomon) 的推文</a>：很荣幸共同领投 @FAL 的 B 轮融资 🚀 AI 驱动的创造力取决于其背后的基础设施。fal 是为 Canva、Perplexity 等提供生成式媒体动力的推理层！很高兴能合作...</li><li><a href="https://x.com/romainhuet/status/1889804638914007458?s=46">来自 Romain Huet (@romainhuet) 的推文</a>：@NickADobos @sama 我们的开发者平台仍然是重中之重，我们的 API 将支持 o3 的推理能力！是的，我们将继续提供你所需的所有控制——比如“推理努力（reasoning effort）”设置...</li><li><a href="https://share.snipd.com/episode/645ae532-40fd-43ff-9ee4-eb76c8fd56fe">Jeff Dean &amp; Noam Shazeer – 在 Google 的 25 年：从 PageRank 到 AGI</a>：Jeff Dean &amp; Noam Shazeer – 在 Google 的 25 年：从 PageRank 到 AGI</li><li><a href="https://x.com/scaledcognition/status/1889721166421479751?s=46">来自 Scaled Cognition (@ScaledCognition) 的推文</a>：我们是 Scaled Cognition，正在开发首个专门为 Agent 应用训练的模型：1. 我们的第一个系统 APT-1，目前在 Agent 基准测试中排名第一。2. 它由一支美国团队开发...</li><li><a href="https://x.com/winstonweinberg/status/1889713028234416371?s=46">来自 Winston Weinberg (@winstonweinberg) 的推文</a>：很高兴宣布由 @sequoia 领投的 D 轮融资，参与方包括 @conviction、@kleinerperkins、@OpenAI、@GVteam、@conviction、@eladgil 和 @LexisNexis。感谢我们的客户、团队、投资...</li><li><a href="https://x.com/natolambert/status/1890065738515505501?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：新演讲！我想留出空间来问：这股新的 RL 兴趣浪潮将走向何方？这与我们在 ChatGPT 之后通过 Alpaca 等“重新发现”RLHF 时相比如何？哪些因素促成了这...</li><li><a href="https://x.com/fleetingbits/status/1889759187913367571?s=46">来自 FleetingBits (@fleetingbits) 的推文</a>：OpenAI 正在转向销售 Agent 而非模型。一些想法：1) 你将不再能构建自己的系统，因为 OpenAI 已经为你打包好了；2) 你将购买某种智能水平...</li><li><a href="https://www.youtube.com/watch?v=YXTYbr3hiFU">一场意想不到的强化学习复兴</a>：我们所处的语言模型研究时代，普遍坚信推理和新的强化学习（RL）训练...</li><li><a href="https://engineering.fb.com/2025/02/05/security/revolutionizing-software-testing-llm-powered-bug-catchers-meta-ach/">变革软件测试：推出 LLM 驱动的漏洞捕获器</a>：它是什么：Meta 的自动化合规加固（ACH）工具是一个用于变异引导、基于 LLM 的测试生成系统。ACH 通过生成未检测到的故障（变异）来加固平台防止回归……</li><li><a href="https://x.com/GoogleDeepMind/status/1890054036168356283">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：🎥 我们最先进的视频生成模型 Veo 2 现已在 @YouTube Shorts 中可用。通过 Dream Screen 功能，创作者可以：✨ 制作无缝融入其叙事的全新片段...</li><li><a href="https://x.com/zach_nussbaum/status/1890088381742256446?s=46">来自 Zach Nussbaum (@zach_nussbaum) 的推文</a>：许多 Embedding 模型，尤其是多语言模型，已经从 BERT-base 规模扩展到了 7B Mistral 规模。但为什么 Embedding 没有借鉴 LLM 的经验，利用混合专家模型（Mixture of Experts）...</li><li><a href="https://x.com/dejavucoder/status/1889884453889253844?s=46">来自 sankalp (@dejavucoder) 的推文</a>：你在笑。他们把 DeepSeek R1 放在一个带有简单验证器的循环中，在某些情况下，它在编写 GPU Kernel 方面表现优于 NVIDIA 的资深工程师，而你却在笑？</li>

<li><a href="https://x.com/airkatakana/status/1889371928080818382">来自 Air Katakana (@airkatakana) 的推文</a>：Hugging Face 上排名第三的热门模型</li><li><a href="https://x.com/nrehiew_/status/1889851293835076024?s=46">来自 wh (@nrehiew_) 的推文</a>：这个 Agent 框架，那个 Agent 框架。你需要的只是一个 while 循环。引用 anton (@abacaj)：呃，可能要结束了……他们让 R1 循环运行了 15 分钟，它生成了：“在某些情况下比资深工程师开发的优化内核更好”……</li><li><a href="https://x.com/abacaj/status/1889847093046702180?s=46">来自 anton (@abacaj) 的推文</a>：呃，可能要结束了……他们让 R1 循环运行了 15 分钟，它生成了：“在某些情况下比资深工程师开发的优化内核更好”。引用 Anne Ouyang (@anneouyang)...</li><li><a href="https://x.com/pitdesi/status/1889830141116948753?s=46">来自 Sheel Mohnot (@pitdesi) 的推文</a>：Anthropic 预计今年基本收入为 22 亿美元（高于 2024 年的约 5 亿美元），并预测 2027 年达到 120 亿美元。OpenAI 的收入约为 Anthropic 的 5 倍，预测 2027 年达到 440 亿美元。为了实现这些预测，他们必须……</li><li><a href="https://youtu.be/LP5OCa20Zpg?si=DOMco-LipDJwYPP-">构建 AI Agent 的技巧</a>：Anthropic 的 Barry Zhang（应用 AI）、Erik Schultz（研究）和 Alex Albert（Claude 关系）讨论了 AI Agent 的潜力、应避免的常见陷阱……</li><li><a href="https://x.com/MatthewBerman/status/1890081482104008920?t=V3aeg7FX8ZvIKtvhHtl-xA&s=19">来自 MatthewBerman (@MatthewBerman) 的推文</a>：新的研究论文展示了 LLM 如何在输出单个 token 之前进行内部“思考”！与 Chain of Thought 不同，这种“潜意识推理（latent reasoning）”发生在模型的隐藏空间中。大量……</li><li><a href="https://x.com/matt_barrie/status/1889907121803895101?s=46">来自 Matt Barrie (@matt_barrie) 的推文</a>：“它是在最大的算力上训练的，使用了大量的合成数据，如果它得到的数据是错误的，它会对此进行反思并将其删除。即使没有微调，Grok 3 基础模型也比……”</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-8B-Preview">NousResearch/DeepHermes-3-Llama-3-8B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/tim_cook/status/1890068457825394918">来自 Tim Cook (@tim_cook) 的推文</a>：准备好迎接家庭的新成员。2 月 19 日，星期三。#AppleLaunch</li><li><a href="https://x.com/dimitrispapail/status/1889747709491351734?s=46">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：o3 无法进行超过几位数的乘法……但我认为乘法、加法、迷宫求解和由易到难的泛化实际上在标准 Transformer 上是可以解决的……通过递归自我改进……</li><li><a href="https://www.youtube.com/watch?v=LP5OCa20Zpg&ab_channel=Anthropic">构建 AI Agent 的技巧</a>：Anthropic 的 Barry Zhang（应用 AI）、Erik Schultz（研究）和 Alex Albert（Claude 关系）讨论了 AI Agent 的潜力、应避免的常见陷阱……</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1inoi6b/openai_silently_rolls_out_o1_o3mini_and_o3mini/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/sama/status/1889755723078443244?s=46&t=JE84TqLviekDnEt8MAT-Eg">来自 Sam Altman (@sama) 的推文</a>：OpenAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“直接工作”；我们……</li><li><a href="https://x.com/OpenAI/status/1889781541259321466">来自 OpenAI (@OpenAI) 的推文</a>：今天我们分享了 Model Spec 的重大更新——这是一份定义我们希望模型如何表现的文档。此次更新强化了我们对可定制性、透明度和智力……的承诺。</li><li><a href="https://x.com/alexwei_/status/1889727569421087217?s=46">来自 Alexander Wei (@alexwei_) 的推文</a>：来自 OpenAI 推理团队的 o3 x 竞赛编程更新！🧵 早在 8 月，我们就冲刺准备让 o1 参加 2024 年国际信息学奥林匹克竞赛……</li><li><a href="https://x.com/aravsrinivas/status/1889742679912628267?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：在此推文下回复你认为非常令人印象深刻且（或）有用的 ChatGPT 深度搜索（deep search）提示词（或链接）</li><li><a href="https://x.com/iscienceluvr/status/1889872445039059445?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：使用连续概念进行 LLM 预训练。Meta 的新论文介绍了一个新的预训练框架，模型必须预测从预训练的稀疏自编码器（sparse autoencoder）中学习到的“连续概念”……</li><li><a href="https://x.com/joannejang/status/1889786393829974290?s=46">来自 Joanne Jang (@joannejang) 的推文</a>：📖 Model Spec v2！这是概述我们对 OpenAI 模型预期行为要求的最新版本文档。其中一些新增内容应该没有争议，但有些肯定会引起……</li>

<li><a href="https://x.com/anneouyang/status/1889770174124867940?s=46">来自 Anne Ouyang (@anneouyang) 的推文</a>：Nvidia 的新博客文章：LLM 生成的 GPU kernels 显示出比 FlexAttention 更快的速度，并在 🌽KernelBench Level 1 上实现了 100% 的数值正确性</li><li><a href="https://x.com/nomic_ai/status/1889721438300442749?s=46">来自 Nomic AI (@nomic_ai) 的推文</a>：Nomic Embed Text V2 现已发布 - 首个通用 Mixture-of-Experts (MoE) 嵌入模型 - 在同尺寸的多语言 MIRACL 基准测试中达到 SOTA 性能 - 支持 100 多种语言 - Tr...</li><li><a href="https://x.com/aravsrinivas/status/1889668709356023942?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：需要毫不含糊地澄清这一点：我们仍然认为 NVIDIA 是无与伦比且独一无二的，是目前为止的行业领导者。我们与他们的关系没有改变。我喜欢 Andrew 和 Cerebras...</li><li><a href="https://x.com/teortaxestex/status/1889926068041384371?s=46">来自 Teortaxes▶️ (DeepSeek 推特🐋铁粉 2023 – ∞) (@teortaxesTex)</a>：我相信 Elon。一个理论是，他们一直在发布平庸的内容并理顺组织架构，但最终已经赶上，并将成为第一个发布该权重级别设计精良且训练有素的模型的人...</li><li><a href="https://x.com/avischiffmann/status/1889827327074595205?s=46">来自 Avi (@AviSchiffmann) 的推文</a>：我觉得我们正在 friend 解决最难的工程问题之一。在与 LLM 聊天时创建、更新和查询“记忆”已经勉强可行，但由于 friend 一直在倾听...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio: 新播客发布！ https://x.com/latentspacepod/status/1890101440615453025
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1339296307470143528)** (50 条消息🔥): 

> `获取研究论文、TinyStories 预训练、预训练基础模型、RL 中的中间 Logits、架构与优化挑战` 


- **获取研究论文变得简单**：成员们讨论了获取研究论文的方法，建议使用 [Anna's Archive](https://annas-archive.org/) 等资源，并直接联系作者获取 PDF 访问权限。
   - 一位用户对寻找学术论文时值得收藏的资源表示感谢。
- **TinyStories 提供预训练架构**：一位成员推荐将 TinyStories 用于预训练数据集，称其提供了一系列架构、预训练模型和详细的研究论文。
   - 对话强调了 TinyStories 是专门为促进有效训练小模型而创建的。
- **在有限硬件上优化预训练**：用户寻求适用于性能较低硬件的预训练数据集建议，对 GPT-2 或 Phi 系列模型用于原型项目表示兴趣。
   - 有人指出，TinyStories 可以为在消费级硬件上进行小规模预训练提供一个易于入门的切入点。
- **强化学习的创新方法**：讨论了在新的强化学习模型中使用 Logits 作为中间表示的方法，强调了为了有效采样而延迟归一化。
   - 该提议包括用基于能量的方法取代 Softmax，并集成多目标训练范式以获得更有效的模型性能。
- **训练小模型的挑战**：成员们讨论了在消费级硬件上训练小模型时面临的常见障碍，强调了优化技术的重要性。
   - 一位用户幽默地指出，选择云解决方案通常比本地设置更容易训练，因为本地设置需要大量的优化工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://annas-archive.org/">Anna’s Archive</a>：未找到描述</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/13j0spj/r_tiny_language_models_below_10m_parameter">Reddit - 探索一切</a>：未找到描述</li><li><a href="https://openreview.net/forum?id=O-XJwyoIF-k">Minimum Width for Universal Approximation</a>：宽限制网络的通用近似性质已被作为深限制网络经典通用近似结果的对偶问题进行了研究。然而，临界宽度...</li><li><a href="https://www.reddit.com/r/MachineLearning/comments/13j0spj/r_tiny_language_models_below_10m_parameters_or/">Reddit - 探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1339319616958234704)** (6 messages): 

> `Language Models 的 Forward citation，Monte Carlo Tree Diffusion，平衡语言数据集的挑战，论文讨论日程安排` 


- **探索 Language Models 中的 Forward Citations**：一位成员注意到关于 Language Models 的 Forward citation，指出如果在像日语和英语这样的 **balanced corpora** 上进行训练，它会根据目标语言（如英语对应法语，日语对应中文）进行激活。
   - 他们质疑这是否可以扩展到更多语言，讨论了一种主导语言与多种平衡语言之间的权衡。
- **介绍 Monte Carlo Tree Diffusion**：分享了一篇关于 [Monte Carlo Tree Diffusion](https://arxiv.org/abs/2502.07202) 的论文链接，详细介绍了它如何将 Diffusion Models 的生成强度与 MCTS 的自适应搜索能力相结合。
   - 该框架将 denoising 重新概念化为一个树状结构过程，从而实现对计划的迭代评估和优化。
- **测试更大规模平衡语言的挑战**：一位参与者评论说，由于除了英语之外，任何语言都难以获得提供超过 **3-5B** natural tokens 的开源数据集，因此测试更大规模的平衡语言非常困难。
   - 他们表达了从零开始使用平衡方法进行 pretraining 的兴趣，例如使用 **4 种语言** 的 **20M tokens**。
- **每日论文讨论更新**：一位成员提到他们今天必须跳过论文讨论，但预计明天会有更充实的会议。
   - 另一位成员表示本周是忙碌的冲刺周，不确定是否能参加本周的讨论，但预计下周恢复。



**Link mentioned**: <a href="https://arxiv.org/abs/2502.07202">Monte Carlo Tree Diffusion for System 2 Planning</a>: Diffusion models 最近已成为一种强大的规划工具。然而，与 Monte Carlo Tree Search (MCTS) 不同——后者的性能会随着额外的 test-time computation (TTC) 自然提升，...

  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1339580959934382081)** (5 messages): 

> `Hugging Face 的 Smolagents，不使用 Tokens 的新 AI 模型，新颖的 Language Model 架构` 


- **Hugging Face 发布 Smolagents**：Hugging Face 推出了 `deep research` 的替代方案，即他们的 Agent 框架 [smolagents](https://huggingface.co/spaces/m-ric/open_Deep-Research)，在 **6 个步骤** 下的运行处理时间约为 **13 秒**。
   - 原始代码可以进行修改，以便在本地服务器运行时延长执行时间，展示了对用户的适应性。
- **AI 可以在没有 Tokens 的情况下思考吗？**：一段 [YouTube 视频](https://www.youtube.com/watch?v=ZLtXXFcHNOU) 引发了关于模型是否可以在不使用 tokens 的情况下进行思考的讨论，挑战了 AI 的传统观念。
   - 演讲者通过质疑 AI 运行的基本机制来吸引观众，并鼓励注册以获取定期更新。
- **探索一种新颖的 Language Model 架构**：[arXiv](https://arxiv.org/abs/2502.05171) 上的一篇论文概述了一种新的 Language Model，它通过 latent space 中的隐式推理来扩展 test-time computation，并展开到任意深度。
   - 该模型拥有 **35 亿个 parameters**，在无需专门训练的情况下实现了 reasoning benchmarks 的性能提升，挑战了传统方法。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: 我们研究了一种新颖的 Language Model 架构，它能够通过在 latent space 中进行隐式推理来扩展 test-time computation。我们的模型通过迭代一个 recurrent block 来工作，从而展开...</li><li><a href="https://www.youtube.com/watch?v=ZLtXXFcHNOU">New AI Model &quot;Thinks&quot; Without Using a Single Token</a>: 模型可以在不使用 tokens 的情况下思考吗？！真的吗？？加入我的 Newsletter 以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅：https://www....</li><li><a href="https://huggingface.co/spaces/m-ric/open_Deep-Research">Open Deep-Research - a Hugging Face Space by m-ric</a>: 暂无描述</li><li><a href="https://github.com/huggingface/smolagents/tree/main/examples/open_deep_research">smolagents/examples/open_deep_research at main · huggingface/smolagents</a>: 🤗 smolagents：一个极简的 Agents 库。Agents 编写 Python 代码来调用 tools 并编排其他 Agents。 - huggingface/smolagents
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1339292099027271710)** (20 条消息🔥): 

> `Elon Musk 的 Grok 3，Thomson Reuters AI 版权案，OpenAI 路线图更新，文献综述工具，预训练语言模型发布实践` 


- **Elon Musk 的 Grok 3 准备挑战 OpenAI**：Elon Musk 宣布他的新 AI 聊天机器人 **Grok 3** 即将发布，声称其在推理能力上超越了现有模型，预计将在大约两周内推出。
   - 在此公告发布之前，Musk 的投资团队在与 OpenAI 持续的法律纠纷中，出价 **974 亿美元** 拟收购 OpenAI 的非营利资产。
- **Thomson Reuters 赢得重大 AI 版权诉讼案**：Thomson Reuters 在针对 Ross Intelligence 的重大 [AI 版权诉讼案](https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/) 中获胜，法院判定该公司因复制 Westlaw 的材料而侵犯了其版权。
   - 美国巡回法院法官 Stephanos Bibas 驳回了 Ross 的所有辩护，称其**没有一个站得住脚**。
- **OpenAI 路线图揭示未来计划**：OpenAI 分享了其路线图更新，强调 **GPT-4.5** 将是最后一个非 chain-of-thought 模型，之后将把其 O 系列和 GPT 系列模型合并为一个统一系统。
   - **GPT-5** 模型即将发布，将具备更强的能力，并为不同的订阅层级提供更多访问权限。
- **来自 GitHub 的新文献综述工具**：一位成员介绍了一个用于快速文献综述的新工具 [Deep-Research-Arxiv](https://github.com/GitsSaikat/Deep-Research-Arxiv)，强调其简单性和可靠性。
   - 此外，还提到了一个 Hugging Face 应用，旨在以快速高效为目标促进文献综述。
- **预训练语言模型发布实践中的挑战**：最近的一项研究分析了 Hugging Face 上的 **52,227 个 PTLMs**，并发现了发布实践中的不一致性，包括 **148 种不同的命名方式**。
   - 研究显示，超过 **40%** 的模型权重文件更改未在版本控制实践或文档中体现，这表明存在显著的知识鸿沟。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.msn.com/en-us/news/technology/openai-pledges-that-its-models-won-t-censor-viewpoints/ar-AA1yVNbM?ocid=msedgdhp&pc=U531&cvid=be2ee583152548cb8377075422221335&ei=46">MSN</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/AlignAI/Deep-Research-Arxiv">Deep Research Arxiv - AlignAI 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2409.10472">Towards Semantic Versioning of Open Pre-trained Language Model Releases on Hugging Face</a>: 在 Hugging Face (HF) 等模型注册平台上，开源预训练语言模型 (PTLMs) 的激增为构建相关产品的公司带来了机遇和挑战....</li><li><a href="https://www.wired.com/story/thomson-reuters-ai-copyright-lawsuit/">Thomson Reuters 赢得美国首例重大 AI 版权案</a>: Thomson Reuters 的裁决对生成式 AI 公司与权利持有者之间的斗争具有重大影响。</li><li><a href="https://slashdot.org/story/25/02/13/1154209/musk-says-new-ai-chatbot-outperforms-rivals-nears-launch">Musk 称新 AI 聊天机器人超越竞争对手，即将发布 - Slashdot</a>: Elon Musk 周四表示，他的 AI 初创公司 xAI 将在两周内发布 Grok 3，他声称这款新聊天机器人超越了现有的 AI 模型。Musk 在迪拜世界政府峰会上发表讲话时引用了内部数据...</li><li><a href="https://x.com/sama/status/1889755723078443244?t=EgnihPXVoD2fsS9ag5u5SA&s=19">Sam Altman (@sama) 的推文</a>: OPENAI 关于 GPT-4.5 和 GPT-5 的路线图更新：我们希望在分享预期路线图方面做得更好，并在简化产品供应方面做得更好。我们希望 AI 能为你“直接工作”；我们...</li><li><a href="https://github.com/GitsSaikat/Deep-Research-Arxiv">GitHub - GitsSaikat/Deep-Research-Arxiv: 快速、简单、可靠地进行文献综述</a>: 快速、简单、可靠地进行文献综述。通过在 GitHub 上创建一个账户来为 GitsSaikat/Deep-Research-Arxiv 的开发做出贡献。</li><li><a href="https://ca.finance.yahoo.com/news/elon-musk-says-grok-3-064145838.html">Elon Musk 表示 Grok 3 处于最后阶段，表现优于所有聊天机器人</a>: Elon Musk 周四表示，他的 AI 聊天机器人、ChatGPT 的挑战者 Grok 3 正处于开发的最后阶段，将在大约一两周内发布。“Grok 3 拥有非常强大的推理能力...”
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1339284317175681236)** (3 条消息): 

> `LlamaIndex Open Source Engineer Position, Nomic AI Embedding Model, Google Cloud Integrations` 


- **LlamaIndex 招聘全职开源工程师**：LlamaIndex 正在寻求一名全职开源工程师来扩展其框架，目标人群是热爱 **开源、Python 和 AI** 的人士。感兴趣的候选人可以在 [职位发布](https://twitter.com/llama_index/status/1889724678970978588) 找到更多信息。
   - 这一机会邀请有能力的个人为 LlamaIndex 框架开发令人兴奋的新功能。
- **Nomic AI 推进 Embedding Model**：LlamaIndex 对 **Nomic AI** 的一项新工作表示兴奋，该工作强调了 Embedding Model 对 **Agentic Document Workflows** 的重要性。正如其 [推文](https://twitter.com/llama_index/status/1889725475502665951) 中所述，这一进展对于提高工作流质量和效率至关重要。
   - 社区非常期待看到该 Embedding Model 如何改进与 AI 工作流的集成。
- **LlamaIndex 与 Google Cloud 集成**：LlamaIndex 推出了与 **Google Cloud** 数据库的新集成，支持数据存储、Vector Store、Document Store 和 Chat Store 等多种功能。更多关于这些集成的特性可以在其详细帖子 [此处](https://twitter.com/llama_index/status/1890109073615626388) 中探索。
   - 这些增强功能旨在简化并安全地访问数据，同时利用云端能力。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1339306772166610954)** (65 条消息🔥🔥): 

> `Query Engine Tool with Metadata, Exhaustive RAG Search Techniques, Vector Database Preferences, LlamaIndex Configuration for Unicode, AI Agents and Workflow Implementation` 


- **创建元数据驱动的 Query Engine Tools**：一位用户询问如何开发利用基于元数据的预定义过滤器的 Query Engine Tools，而无需创建多个索引。
   - 另一位用户确认可以使用 `QueryEngineTool.from_defaults` 并配合适当的过滤器来实例化查询工具。
- **探索详尽 RAG 搜索的方法**：一位用户就进行详尽 RAG 搜索的最佳方法寻求建议，特别是当 top k 较低但考虑了许多 chunk 时。
   - 他们提到将 *autorag* 和查询合成（query synthesizing）视为全面搜索数据的潜在解决方案。
- **讨论 Vector Database 选择**：成员们讨论了他们偏好的 Vector Database；一位用户提到尝试使用 *Milvus*，而另一位确认他们使用的是 *Pinecone*。
   - 另一位参与者提到在 Docker 容器中使用 *Redis* 进行向量数据处理。
- **使用 LlamaIndex 存储和显示越南语文本**：一位成员面临 LlamaIndex 将越南语文本转换为 Unicode 转义序列而不是显示正确字符的问题。
   - 他们寻求关于如何配置 LlamaIndex 与 Qdrant 以处理并正确显示越南语文本的帮助。
- **UV 在环境管理中的效用**：用户分享了使用 `uv` 工具管理虚拟环境的经验，讨论了其优缺点。
   - 一位用户分享了一个 bash 函数，通过别名简化环境切换和配置文件调整。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.pantsbuild.org/dev/reference/targets/uv_requirements">uv_requirements | Pantsbuild</a>: 在 `pyproject.toml` 的 `[tool.uv]` 部分下为每个条目生成一个 `python_requirement`。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/query_engine/RouterQueryEngine/">Router Query Engine - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_workflow_basic/">AgentWorkflow Basic Introduction - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/multi_agents/#multi-agent-workflows">Multi-agent workflows - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 条消息): 

pier1337: 微调（finetune）一个模型有哪些好的理由？
  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1339344449183944754)** (10 条消息🔥): 

> `NotebookLM 的播客功能，利用 AI 实现内容变现，AI 生成的播客主持人，比较 AI 来源，将内容转换为音频` 


- **NotebookLM 在 AI 播客领域表现惊艳**：一位用户称赞 NotebookLM 让快速创建播客变得非常简单，声称它可以将文字内容转化为**值得一口气听完的音频**，而无需真人录音。
   - 围绕播客的热情凸显了其作为**内容营销工具**的力量，并强调了在 Spotify 和 Substack 等平台上触达大量潜在受众的潜力。
- **AI 播客：一种新的收入来源**：一位用户概述了通过运营一个双人 AI 播客实现 **$7,850/月** 潜在收入的方案，重点在于快速创作内容的效率。
   - 该帖子强调，使用 AI 进行播客创作可以大幅增加内容消费量，并建议自然触达率（organic reach）可提升 **300%**。
- **AI 生成的主持人引发关注**：一场关于创建 **AI 生成播客主持人库**的讨论展开，展示了用户贡献的各种主题和内容风格。
   - 成员们对协作和分享独特的 AI 生成音频体验的潜力感到兴奋，这增强了社区参与度。
- **比较 AI 来源可能比较棘手**：一位成员指出，**NotebookLM** 在处理多个来源时存在相似化对待的问题，但发现它在分析求职者评分方面仍然有效。
   - 提到使用引用（citations）进一步阐明了来源比较过程，强调了其在评估中的实际用途。
- **对多语言支持的需求**：用户询问 **NotebookLM** 何时会将其功能扩展到**其他语言**。
   - 这凸显了让全球更广泛受众能够使用 AI 工具的日益增长的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://drive.google.com/file/d/1f4CNO1N9J657e9lPEz_JL_ESAHluoBTD/view?usp=sharing">Captain America_ Brave New World Review.wav</a>：未找到描述</li><li><a href="https://millionai.substack.com/p/create-ai-podcasts-in-seconds-without?r=297y6u&utm_medium=ios&triedRedirect=true.">🎙️几秒钟内创建播客（无需开口）🤐</a>：我如何通过双人 AI 播客额外赚取 $7850/月 🎧（无代码）
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1339284758932488322)** (49 条消息🔥): 

> `NotebookLM 功能，每日使用限制，语言支持，在社区分享想法，音频生成问题` 


- **NotebookLM Plus 提供独特功能**：成员们讨论了 **NotebookLM Plus** 如何使学生受益，并提供交互式播客等免费版本可能没有的功能。
   - 一位成员建议转向 **Google AI Premium** 以获取捆绑功能，并表示：“我发现 Google NotebookLM 真的很好……”。
- **对每日聊天限制的困惑**：一位成员询问 NotebookLM 是否存在每日聊天限制，另一位成员确认了这一点，并指出这些限制是在几个月前宣布的。
   - 从 12 月开始，免费用户和 Plus 用户分别引入了 **50/500** 条消息的限制。
- **当前多语言音频的局限性**：尽管有关于支持其他语言的咨询，用户对**音频生成**仅支持英文表示沮丧。
   - 虽然成员们分享了在设置中切换语言的方法，但目前的音频功能仍仅限于**仅限英文**的输出。
- **社区互动与支持**：成员们讨论了 Discord 小组的宗旨，将其视为 NotebookLM 用户之间进行**社区互动**、支持和协作的平台。
   - 另一位成员强调，论坛允许分享想法和解决方案，而无需在网上进行大量搜索，从而增强了社区连接性。
- **音频合成准确性的挑战**：一位用户报告称，当将音频概览（audio overview）集中在特定来源时会出现**不准确**的情况，并建议采用更广泛的方法可能会产生更好的结果。
   - 社区建议将**通用笔记本**与特定笔记本混合使用，以避免混淆并提高音频输出质量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://www.reddit.com/r/notebooklm/comments/1iobfsf/turn_entire_youtube_playlists_to_large_text_books/">Reddit - 深入了解任何事物</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 条消息): 

eggsquad: 新的 Modular 职位发布 👀
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1339286299604947044)** (40 条消息🔥): 

> `Mojo sum types, ECS vs 组件架构, Mojo 函数包装, Mojo 中的内存管理, 新版本 v25.1` 


- **关于 Mojo Sum Types 和 `Variant` 的讨论**：成员们讨论了类 Rust 的 sum types 与 C 风格 enums 之间的区别，强调 `Variant` 可以满足许多需求，但参数化 traits 是即将推出的功能中更高优先级的任务。
   - 一位用户使用 variant 模块实现了一个权宜的（hacky）union 类型，引发了关于当前 Mojo 实现局限性的对话。
- **关于 ECS 实现的误解**：对 ECS 的定义进行了澄清，指出状态应与行为分离，这一概念类似于 Unity3D 中的 MonoBehavior 模式。
   - 成员们确认原始示例确实遵循了 ECS 原则，将状态放在组件中，行为放在系统中。
- **在 Mojo 中使用 Unsafe Pointers**：随后讨论了如何在 Mojo 的结构体中存储和管理函数，并提供了一个利用 `OpaquePointer` 安全管理函数引用的示例。
   - 用户分享了完整的示例，承认在使用 `UnsafePointer` 时管理生命周期和内存的复杂性。
- **`add_fun` 函数调用中的错误**：一位用户遇到了关于 `add_fun` 函数调用中别名（aliasing）的特定编译器错误，这引发了关于 Mojo 中可能存在的内存管理问题的讨论。
   - 参与者讨论了函数式编程模式和可变性约束，分享了他们在当前语言局限性方面的经验。
- **发布新版本 v25.1**：一位成员宣布了 Mojo v25.1 的发布，表达了对潜在新功能和改进的期待。
   - 这一消息引起了人们对新版本中引入的更新和变化的兴趣。



**提到的链接**：<a href="https://zig.news/david_vanderson/faster-interface-style-2b12">未找到标题</a>：未找到描述

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1339286519420162058)** (11 条消息🔥): 

> `MAX CUDA 使用, NVIDIA 后端 Bug, Mojo API tensor 类型, 寻求帮助的论坛` 


- **MAX 最小化了 CUDA 依赖**：MAX 仅在必要时使用 **CUDA**，主要依靠 **CUDA driver** 进行内存分配等功能。
   - 正如一位成员指出的，MAX 意味着一种精简的 GPU 使用方式，特别是在 **NVIDIA** 硬件上以获得最佳性能。
- **棘手的 NVIDIA 后端 Bug 等待处理**：一位成员对分享 **NVIDIA 后端 Bug** 表示犹豫，称这些 Bug 非常严重。
   - 其他人鼓励报告任何 **MAX** Bug，强调他们渴望调查即使是最具挑战性的问题。
- **Mojo API 中的 Tensor 类型问题**：有人担心在访问 **Mojo API** 时可能使用了错误的 tensor 类型。
   - 这表明必须理解**特定的操作**，以确保 MAX 内部正常的 GPU 功能。
- **寻求 CUDA 问题的详细信息**：一位用户对导致问题的 NVIDIA 库表示困惑，并建议在论坛上发布具体细节以获得社区支持。
   - 这种方法符合社区共同诊断并更有效地修复 MAX 问题的努力。



**提到的链接**：<a href="https://forum.modular.com/">Modular</a>：与我们一起构建 AI 的未来，了解 MAX、Mojo 和 Magic。

  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1339303563884494910)** (50 条消息🔥): 

> `MCP 开发, OpenAI 模型在 MCP 中的应用, Claude Desktop 的使用限制, Glama Gateway 对比, Open WebUI 特性` 


- **MCP 开发见解**：成员们讨论了各种 MCP 客户端的使用体验，其中 [wong2/mcp-cli](https://github.com/wong2/mcp-cli) 因其开箱即用的功能而受到关注。
   - *客户端存在 Bug 是一个共同话题*，多位开发者分享了他们尝试绕过现有工具限制的方案。
- **OpenAI 模型在 MCP 中的使用**：新用户对 MCP 的功能表示兴奋，并询问 Claude 以外的模型是否也能支持 MCP。
   - 有人指出，虽然 MCP 与 OpenAI 模型兼容，但像 Open WebUI 这样的其他项目可能不会优先考虑它。
- **讨论 Claude Desktop 的使用限制**：用户注意到 Claude Desktop 的使用限制已成为一个问题，讨论建议 Glama 的服务可以提供一种变通方案。
   - 一位成员强调，这些限制极大地影响了他们的使用场景，并指出 Glama 如何提供更便宜、更快速的替代方案。
- **Glama Gateway vs. OpenRouter**：成员们将 [Glama's gateway](https://glama.ai/gateway) 与 OpenRouter 进行了对比，指出其在低成本和隐私保障方面的优势。
   - 虽然 Glama 支持的模型较少，但因其快速和可靠而受到称赞，使其成为某些应用的可靠选择。
- **对 Open WebUI 的好奇**：几位用户对 Open WebUI 表示好奇，提到了其丰富的功能集以及最近关于 MCP 支持的路线图更新。
   - 成员们对其易用性给予了正面评价，并表达了希望完全从 Claude Desktop 迁移出来的愿望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://glama.ai/gateway">Gateway</a>: 快速、可靠的 AI 网关</li><li><a href="https://jryng.com/thoughts/why-open-webui">Timothy J. Baek - 为什么我要构建 Open WebUI：关于自主性、多样性和人类的未来</a>: 未找到描述</li><li><a href="https://glama.ai/models/">领先的 LLM 模型</a>: 企业级安全、隐私，具备 Agent、MCP、提示词模板等功能。</li><li><a href="https://github.com/luohy15/y-cli">GitHub - luohy15/y-cli: 一个支持 MCP 客户端的微型 AI 模型终端聊天应用</a>: A Tiny Terminal Chat App for AI Models with MCP Client Support - luohy15/y-cli
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1339316816253550602)** (16 messages🔥): 

> `Windows CI 问题, DeepSeek-R1 模型实验, 图重写挑战, AI 与代码提交礼仪` 


- **Windows CI 在环境变量处理上遇到困难**：一位成员指出 **Windows CI** 未能在步骤间传递后端环境变量，导致每次都默认选择 **CLANG**。他们链接了一个解决此问题的 [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/9047)。
   - 该 PR 通过在 CI 执行期间利用 `$GITHUB_ENV` 确保后端变量在各步骤中持久化。
- **DeepSeek-R1 自动化 GPU kernel 生成**：分享的一篇博文讨论了 **DeepSeek-R1 模型** 实验，展示了在解决复杂问题的 GPU kernel 生成方面的改进。这种被称为“测试时缩放（_test-time scaling_）”的技术在推理过程中分配更多计算资源，从而提高模型性能。
   - 这使得 AI 能够有效地制定**策略**，通过在选择最佳方案前评估多种结果，来模拟人类解决问题的过程。
- **tinygrad 中的图重写（Graph rewrite）Bug 挑战**：成员们讨论了 CI 失败的原因，有人认为**缩进错误**导致 `bottom_up_rewrite` 从 `RewriteContext` 中被移除。其他人则指出图处理可能存在更深层次的问题，例如错误的重写规则或顺序。
   - 一位成员表示他们正在学习代码库，并鼓励尝试各种方法，而不是局限于单一解决方案。
- **AI 在代码编写中的角色 —— 警告**：一位成员强调了在提交前彻底审查代码 diff 的重要性，指出任何微小的空格变化都可能导致 PR 被关闭。他们还敦促不要直接提交 AI 生成的代码，建议将 AI 用于头脑风暴和反馈。
   - 这种方法被视为尊重社区时间并鼓励成员进行原创代码开发的一种方式。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.tinygrad.org/developer/speed/">Speed - tinygrad docs</a>: 未找到描述</li><li><a href="https://x.com/__sxsm/status/1889916679167287496">0xSG - */acc (@__sxsm) 的推文</a>: @__tinygrad__ 新的 beam search 刚刚发布。引用 Anne Ouyang (@anneouyang)：来自 Nvidia 的新博文：LLM 生成的 GPU kernel 显示出比 FlexAttention 更快的速度，并实现了 100% 的数值校正...</li><li><a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">使用 DeepSeek-R1 和推理时缩放自动化 GPU Kernel 生成 | NVIDIA 技术博客</a>: 随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为测试时缩放（test-time scaling）或推理时缩放（inference-time scaling）的新缩放定律正在出现。也被称为 AI 推理...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/9047">由 rmtew 提交的 PR #9047：确保 Windows CI 正确测试指定的后端</a>: 确保设置的后端环境变量通过 $GITHUB_ENV 持久化到下一步。除非 shell 显式设置为 bash，否则它在 Windows 上实际上不会持久化。添加了断言...
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1339321094112284683)** (2 messages): 

> `tinygrad vs PyTorch, 性能与成本效率, 理解硬件` 


- **评估 tinygrad 与 PyTorch**：一位用户询问了从 **PyTorch** 切换到 **tinygrad** 的好处，想知道学习 tinygrad 的投入是否值得。
   - 另一位成员建议，如果你在寻找**更便宜的硬件**或者想要理解“底层”发生了什么，那么切换是有意义的。
- **更便宜且更快的模型**：对于那些关注**成本效率**或性能的人来说，迁移到 tinygrad 最终可能会获得比典型 PyTorch 配置**更快的模型**。
   - 对于专注于优化和资源管理的用户来说，这可能代表着显著的优势。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1339295436812587038)** (1 messages): 

> `LLM Agents MOOC Hackathon, Participation statistics, Winning teams, Top represented countries, Top represented companies` 


- **LLM Agents Hackathon 获胜团队公布**：随着 **LLM Agents MOOC Hackathon** 获胜团队的揭晓，社区内充满了激动的情绪，这展示了全球 AI 社区的**惊人参与度**。
   - *Dawn Song 教授* 强调了这一成就，并在 Twitter 上分享了细节，重点提到了全球参与者展现出的*令人振奋的热情*。
- **揭晓令人惊叹的参与统计数据**：本次 Hackathon 吸引了来自 **127 个国家**的 **约 3,000 名参与者**，涉及 **1,100 多所大学**和 **800 多家公司**。
   - 参与人数最多的国家是**美国**、**印度**和**中国**，展示了全球人才的多样化代表性。
- **参与表现突出的顶尖院校**：获得认可的机构包括 **UC Berkeley**、**UIUC**、**Stanford**、**Carnegie Mellon** 和 **Northeastern**，这些学校吸引了大量的参与者。
   - 这些学校是**代表人数最多**的院校，表明了它们在 AI 社区中的强大影响力。
- **参加 Hackathon 的主要公司**：**Amazon**、**Microsoft**、**Samsung** 和 **Salesforce** 等公司在参与者中占有显著比例。
   - 它们的参与凸显了本次 Hackathon 对 AI 领域关键参与者的吸引力，促进了协作与创新。
- **在 Hackathon 网站上探索获胜作品**：可以在官方 Hackathon 网站上探索**获胜团队**及其提交的作品。
   - 鼓励参与者和关注者访问 [Hackathon 网站](https://rdi.berkeley.edu/llm-agents-hackathon/) 以庆祝这些成就。



**提及的链接**：<a href="https://x.com/dawnsongtweets/status/1889686697564315963)">Dawn Song (@dawnsongtweets) 的推文</a>：🎉 很高兴宣布 LLM Agents MOOC Hackathon 的获胜团队！我们对全球 AI 社区展现出的惊人参与度和热情感到非常振奋：🌍 来自 127 个国家的约 3,000 名参与者...

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1339296200913850450)** (1 messages): 

> `Spring 2025 MOOC, Advanced LLM Agents, Live Sessions, AI for Mathematics` 


- **2025 春季 MOOC 正式启动**：**2025 春季 MOOC** 已正式启动，面向更广泛的 AI 社区，并邀请大家转发 [Dawn Song 教授的公告](https://x.com/dawnsongtweets/status/1889355520294944829)。
   - 本学期旨在延续 2024 秋季课程的成功，当时该课程拥有 **1.5 万+ 注册学员**，且 YouTube 上的**课程观看量超过 20 万次**。
- **探索高级主题**：更新后的课程涵盖了诸如 **Reasoning & Planning**（推理与规划）、**Multimodal Agents**（多模态 Agents）以及 **AI for Mathematics and Theorem Proving**（用于数学和定理证明的 AI）等高级主题。
   - 这一举措反映了人们对 **Agent Safety & Security**（Agent 安全与保障）等更复杂的 AI 领域日益增长的兴趣。
- **加入直播课程**：课程将于**每周一太平洋时间 (PT) 下午 4:10 进行直播**，为所有参与者提供参与式的学习体验。
   - 该系列课程邀请从学生到研究人员的所有人共同参与，塑造 **LLM Agents** 的未来。



**提及的链接**：<a href="https://x.com/dawnsongtweets/status/1889355520294944829)">Dawn Song (@dawnsongtweets) 的推文</a>：非常高兴宣布我们的高级 LLM Agents MOOC（2025 春季）！基于我们 2024 秋季 LLM Agents MOOC 的成功（1.5 万+ 注册学员，约 9,000 名 Discord 成员，20 万+ 课程观看量 ...

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1339332225757610034)** (10 messages🔥): 

> `Hackathon 参与, 证书重发请求, Prompt 评估, Ninja 认证, 证书声明表单` 


- **本学期没有 Hackathon**：一位新学生询问本学期是否可以参加 Hackathon，但已确认今年春季没有相关计划。
   - *Prof Song 过去曾举办过多次 Hackathon*，未来可能会继续举办。
- **未收到证书**：多位用户报告未收到证书，其中一名学生请求将其重发至特定邮箱地址。
   - Tara 确认她会将此请求加入任务列表，但提到可能要到周末才能完成。
- **Ninja 认证的 Prompt 评估**：一名学生询问了为 Ninja 认证提交的 Lab 作业评分情况以及最佳 Prompt 的识别。
   - Tara 表示没有正式评分，并建议在指定的频道中针对其他学生的提交内容测试 Prompt。
- **证书声明要求**：另一名学生请求协助查找其 2024 年秋季的证书，并被询问是否填写了必要的声明表单。
   - Tara 对其在系统中的状态表示不确定，指出提交该表单是处理证书的必要步骤。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1339294431488114759)** (2 messages): 

> `发布详情更新, AI/ML 新手指南` 


- **发布详情更新**：一份公告指出**更多详情**将很快发布，并感谢用户的**耐心等待**。
   - *请关注即将发布的更新*，届时可能会提供更多见解。
- **AI/ML 新手指南**：一位新成员表示有兴趣获得关于开启 **AI/ML 领域**学习以及理解模型训练技术的指导。
   - *为了寻求帮助，他们向社区咨询*关于如何有效开启学习之旅的见解。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/)** (1 messages): 

tarande57: we'll release details soon! thank you for your patience!
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1339334636060540971)** (9 messages🔥): 

> `使用 Torchtune 进行分布式推理, Torchtune Docker 镜像, 使用 vLLM 进行模型加载` 


- **Torchtune 支持分布式推理 (Distributed Inference)**：一位成员询问关于在多 GPU 上使用 **Torchtune** 运行分布式推理的问题，另一位成员提供了 [GitHub recipe](https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py) 链接作为指导。
   - 此外，建议使用 **vLLM** 加载保存的模型作为替代方案，这也能带来速度提升。
- **Torchtune 目前没有可用的 Docker 镜像**：一位成员询问是否存在 **Torchtune** 的 Docker 镜像，并表示难以找到。
   - 另一位参与者确认**目前没有 Docker 镜像**，并引导其参考 GitHub 上的 [安装指南 (installation instructions)](https://github.com/pytorch/torchtune?tab=readme-ov-file#installation)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune?tab=readme-ov-file#installation">GitHub - pytorch/torchtune: PyTorch native post-training library</a>: PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/dev/generate_v2_distributed.py">torchtune/recipes/dev/generate_v2_distributed.py at main · pytorch/torchtune</a>: PyTorch 原生训练后处理库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1339362958970589184)** (5 messages): 

> `Checkpointing Branch, Recipe State Functionality, Documentation Improvement, Team Collaboration` 


- **Checkpointing Branch 成功克隆**：一名成员确认他们已成功克隆了 **checkpointing branch**，并表示测试后表现非常出色。
   - 他们表示打算验证 **recipe_state.pt** 是否按预期工作，并可能扩展关于恢复训练（resuming training）的文档。
- **对测试成功的轻松回应**：另一名成员对测试的成功做出了积极回应，并以幽默的口吻承认了之前的某种不确定性。
   - 这种轻松的交流突显了团队在故障排除和项目推进中的战友情谊。
- **关于过去困惑的询问**：出现了一个不确定的时刻，一名成员询问自己是否在某些细节上理解错误。
   - 这表明团队成员之间正在就项目进行持续的讨论和澄清。
- **愿意在 Checkpointing PR 上进行协作**：一名团队成员在 **checkpointing PR** 中留言，表达了对该项目协作的热情。
   - 这展示了团队合作的积极态度以及对改进 checkpointing 流程的承诺。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1339398253610930226)** (10 messages🔥): 

> `Inference-Time Scaling, LangChain vs DSPy, DSPy 2.6 Changes` 


- **NVIDIA 探索推理时缩放 (Inference-Time Scaling)**：NVIDIA 的新实验展示了使用 **DeepSeek-R1** 模型优化 GPU attention kernels 的推理时缩放方法，通过评估多个结果来实现更好的问题解决能力。
   - 值得注意的是，这种技术在推理过程中分配了额外的计算资源，类似于人类解决问题的策略。
- **在 LangChain 和 DSPy 之间选择**：关于何时应该选择 **LangChain** 而非 **DSPy** 的问题被提出，强调了两者都有独特的用例。
   - 一位成员建议，如果学习曲线看起来过于陡峭，那么利用 LangChain 中成熟的方法可能会更好。
- **DSPy 2.6 更新日志查询**：一位用户询问了 **DSPy 2.6** 的更新日志，提到了为 Signatures 引入的 **instructions**，并质疑其与之前版本相比的有效性。
   - 回复澄清说这些 instructions 自 2022 年以来就一直存在，GitHub 上提供了详细的变更日志以供进一步了解。



**提到的链接**：<a href="https://developer.nvidia.com/blog/automating-gpu-kernel-generation-with-deepseek-r1-and-inference-time-scaling/">Automating GPU Kernel Generation with DeepSeek&#x2d;R1 and Inference Time Scaling | NVIDIA Technical Blog</a>：随着 AI 模型扩展其解决更复杂挑战的能力，一种被称为测试时缩放（test-time scaling）或推理时缩放（inference-time scaling）的新缩放定律正在出现。也被称为 AI 推理...

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1339454416151121971)** (10 messages🔥): 

> `GPT4All v3.9.0 与 Deepseek R1 集成、LocalDocs 功能与局限性、NOIMC v2 发布与实现、Nomic 的多语言 MoE 文本嵌入、将英文 Prompt 转换为代码` 


- **GPT4All v3.9.0 接入 Deepseek R1**：成员们澄清 GPT4All v3.9.0 并没有完全集成 **Deepseek R1**，而是允许用户在本地下载并运行该模型，强调了其离线能力。
   - 讨论中提到了在本地运行全量模型的挑战，因为目前似乎仅限于 13B 参数等较小的变体，其表现不如全量版本。
- **LocalDocs 仍需改进**：一位用户分享了对 **LocalDocs** 功能的挫败感，称其功能非常基础，在处理其 TXT 文档时准确率仅有约一半。
   - 有人提出疑问，这种局限性是源于使用了 **Meta-Llama-3-8b instruct** 模型，还是由于设置不当。
- **关于 NOIMC v2 实现的担忧**：成员们质疑为什么 **NOIMC v2** 模型在发布后仍未得到妥善实现。
   - 分享了 [nomic-embed-text-v2-moe 模型](https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe) 的链接，强调了其多语言性能和能力。
- **关于多语言文本嵌入的讨论**：**nomic-embed-text-v2-moe** 模型因其顶尖的多语言能力而受到称赞，它支持约 **100 种语言**，与同等规模的模型相比具有显著的高性能。
   - 强调了灵活的 Embedding 维度及其完全开源的特性，并提供了其 [代码](https://github.com/nomic-ai/contrastors) 链接。
- **寻求将 Prompt 转换为代码的建议**：一位用户寻求关于使用什么工具将 **英文 Prompt** 转换为可运行代码的帮助，表示需要有效的解决方案。
   - 由于缺乏具体的建议，引发了对促进这一过程的合适选项的进一步询问。



**提到的链接**：<a href="https://huggingface.co/nomic-ai/nomic-embed-text-v2-moe">nomic-ai/nomic-embed-text-v2-moe · Hugging Face</a>：未找到描述

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1339597639150211204)** (2 messages): 

> `Rerank 3.5 行为、Cohere 与 Salesforce BYOLLM` 


- **Rerank 3.5 评分不一致**：一位用户报告称，同一份文档在 **Rerank 3.5** 中分不同批次处理时会获得不同的评分，这与他们对确定性行为的预期相悖。
   - *这种变异性似乎不合常理*，因为它是作为 Cross-encoder 运行的。
- **使用 Cohere 与 BYOLLM 的挑战**：一位成员询问是否有人成功将 **Cohere** 作为 LLM 与 Salesforce 的 BYOLLM 开放连接器配合使用，并指出 [api.cohere.ai/v2/chat](https://api.cohere.ai/v2/chat) 的 Chat 端点存在问题。
   - 他们提到正尝试根据 Salesforce 支持人员的建议，创建一个 https REST 服务来调用 Cohere 的 Chat API。


  

---


---


---


{% else %}


> 完整的频道细分内容已为邮件格式进行删减。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}