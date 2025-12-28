---
companies:
- meta-ai-fair
- google-deepmind
- hugging-face
date: '2024-09-27T21:53:11.435082Z'
description: '**Meta** 发布了 **Llama 3.2**，其中包括用于端侧 AI 的轻量级 1B 和 3B 模型，具备摘要和检索增强生成（RAG）等功能。**Molmo**
  是一款新型多模态模型，随之发布的还有一个大型密集字幕数据集。**Google DeepMind** 宣布了 **AlphaChip**，这是一种 AI 驱动的芯片设计方法，旨在改进
  TPU 和 CPU 的设计。**Hugging Face** 的免费公开模型数量突破了 100 万个，凸显了小型专业化模型的价值。


  讨论内容涵盖了扩展 RAG 应用的挑战、运行 ChatGPT 级别模型的端侧 AI 的未来、大型语言模型（LLM）的可靠性问题，以及被 NeurIPS 2024
  接收的新 Elo 基准测试。AI 伦理和监管话题包括言论自由责任以及可能影响开源 AI 的加州 SB-1047 法案。“AlphaChip 改变了计算机芯片设计”，以及“预计一年内移动设备上将出现
  ChatGPT 级别的 AI”。'
id: fd9ea842-feac-4042-b9a3-ffbdf7a49551
models:
- llama-3-2
- llama-3
- molmo
original_slug: ainews-not-much-happened-today-1696
people:
- demis-hassabis
- clementdelangue
- svpino
- awnihannun
- osanseviero
- omarsar0
- sarahookr
- ylecun
title: 今天没发生什么特别的事。
topics:
- on-device-ai
- multimodality
- chip-design
- retrieval-augmented-generation
- rag
- benchmarking
- reliability
- ai-regulation
- free-speech
- pytorch-optimization
---

<!-- buttondown-editor-mode: plaintext -->**平静的一天正是你所需要的**

> 2024/9/26-2024/9/27 的 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord 社区（**224** 个频道，**2635** 条消息）。预计节省阅读时间（按 200wpm 计算）：**288 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来讨论 AINews！

今天有很多非头条新闻：

- [GDM 发布了 AlphaChip](https://deepmind.google/discover/blog/how-alphachip-transformed-computer-chip-design/)
- [FTC 打击欺骗性 AI 宣传](https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes)
- [Copilot 现已在浏览器端的 GitHub.com 上线](https://x.com/ashtom/status/1839393494366138530)
- [关于 OpenAI 闹剧的大量报道](https://x.com/garrisonlovely/status/1839655744850772272?s=46)
- [GGML 开始通过 HuggingFace 变现](https://x.com/ggerganov/status/1839703977073487993)

在浏览下方新闻的同时，你可以收听最新的 [Latent Space，嘉宾是 Shunyu Yao 和 Harrison Chase](https://www.latent.space/p/shunyu)！

如果你在旧金山参加 DevDay，考虑在周一带着你的 demo 和犀利观点来参加[我们的 DevDay pregame 活动](https://lu.ma/devday-pregame)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与进展**

- **Llama 3.2 发布**：Meta 发布了 Llama 3.2，包括用于设备端 AI 应用的轻量级 1B 和 3B 模型。[@AIatMeta](https://twitter.com/AIatMeta/status/1839365639687086308) 指出，这些模型使开发者能够构建个性化的、设备端的 Agentic 应用，具备摘要、工具使用和 RAG 等功能，且数据无需离开设备。[@awnihannun](https://twitter.com/awnihannun/status/1839330067039887622) 展示了 4-bit 量化的 Llama 3.2 1B 在 iPhone 15 Pro 上以约 60 tokens/sec 的速度运行。

- **Molmo 多模态模型**：发布了一个名为 Molmo 的新多模态模型，[@osanseviero](https://twitter.com/osanseviero/status/1839398112701386912) 强调了其数据流水线和训练过程。该模型使用了一个包含 71.2 万张图像/130 万条说明的密集描述数据集，以及用于监督微调的各种数据集。

- **AlphaChip**：Google DeepMind 宣布了 AlphaChip，一种用于芯片设计的 AI 方法。[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1839306984480231852) 表示，它已经改变了他们设计微芯片的方式，从用于 AI 模型的 TPU 到数据中心的 CPU。[@demishassabis](https://twitter.com/demishassabis/status/1839354651206160563) 指出了一个反馈闭环：AlphaChip 被用于设计更好的 AI 芯片，而这些芯片随后又被用于训练更好的模型。

**AI 基础设施与平台**

- **Hugging Face 里程碑**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1839375655688884305) 宣布 Hugging Face 突破了 1,000,000 个免费公开模型的里程碑，强调了针对特定用例的小型化、专业化模型的重要性。

- **RAG 应用**：[@svpino](https://twitter.com/svpino/status/1839364380947054596) 讨论了扩展 RAG 应用的挑战，指出由于向量相似度搜索的局限性，更多的数据可能会让效果变差。他强调的研究表明，随着知识库的增长，准确率会下降。

- **设备端 AI**：几条推文讨论了设备端 AI 的潜力，[@cognitivecompai](https://twitter.com/cognitivecompai/status/1839448460619128962) 预测再过一年，ChatGPT 级别的 AI 将在移动/嵌入式设备上运行。

**AI 研究与基准测试**

- **LLM 的可靠性**：[@omarsar0](https://twitter.com/omarsar0/status/1839332359554163127) 分享了来自《Nature》论文的见解，该论文认为更大且更易受指令引导的 LLM 可能会变得不那么可靠，存在难度一致性、任务规避和 Prompt 稳定性方面的问题。

- **Elo 基准测试**：[@sarahookr](https://twitter.com/sarahookr/status/1839399320048763247) 宣布关于 NLP 中 Elo 基准测试的研究被 NeurIPS 2024 接收，解决了这一广泛使用的评估方法中的可靠性问题。

**AI 伦理与监管**

- **言论自由与 AI**：[@ylecun](https://twitter.com/ylecun/status/1839402554809373144) 强调了言论自由的负责任使用，警告传播有害阴谋论可能带来的法律后果。

- **AI 监管**：几条推文讨论了 SB-1047，这是一项可能影响开源 AI 开发的加州法案。[@ylecun](https://twitter.com/ylecun/status/1839398310899339699) 表示希望州长 Gavin Newsom 会否决它。

**AI 开发工具与技术**

- **PyTorch 优化**：[@cHHillee](https://twitter.com/cHHillee/status/1839421129682997723) 讨论了 PyTorch 在强化学习工作负载中的性能提升，通过使用 CUDA Graphs 和 torch.compile，实现了超过 5 倍的加速。

- **网页抓取**：[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1839348216317505735) 分享了一个 GitHub 仓库，用于轻松抓取网页并以 LLM 友好的格式（如 JSON、清洗后的 HTML 和 Markdown）输出。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Llama 3.2：性能提升与欧盟监管挑战**

- **Llama 3.2 Vision Models 图像像素限制** ([Score: 40, Comments: 3](https://reddit.com//r/LocalLLaMA/comments/1fqawht/llama_32_vision_models_image_pixel_limits/)): 新的 **Llama 3.2 Vision Models** 的 **11B** 和 **90B** 版本最大图像尺寸均为 **1120x1120 像素**，具有 **2048** token 输出限制和 **128k** context length。这些模型支持 **gif, jpeg, png, and webp** 图像文件类型，这些信息在官方文档中并不容易找到，需要通过大量测试才能确定。
  - Llama 3.2 Vision Models 的 **最大图像尺寸** 实际上是 **4 张 560x560 图像**，正如 [Hugging Face 上的 preprocessor config](https://huggingface.co/meta-llama/Llama-3.2-90B-Vision-Instruct/blob/main/preprocessor_config.json) 中所揭示的。该配置指定了 **"max_image_tiles": 4** 且图像尺寸为 **560x560**。
- 用户对提供的模型能力信息表示赞赏，指出其对实际应用非常有用。
- **[通过 ChatterUI 在 Android 上运行 Llama 3.2](https://v.redd.it/gqbakkmtc6rd1)** ([Score: 39, Comments: 9](https://reddit.com//r/LocalLLaMA/comments/1fpze6d/running_llama_32_on_android_via_chatterui/)): 该帖子宣布发布 **ChatterUI v0.8.0-beta3**，现在支持在 **Android** 设备上运行 **Llama 3.2** 模型。使用 **Snapdragon 7 Gen 2** 处理器，该应用在 prompt processing 方面达到 **50 tokens per second**，在文本生成方面达到 **10 tokens per second**，展示了在现代 **Android** 硬件上的良好性能。作者提供了 [Beta 版发布链接](https://github.com/Vali-98/ChatterUI/releases/tag/v0.8.0-beta3) 并征求反馈，特别是关于角色列表和聊天历史记录的更改。
  - 用户对移动设备上的 **更大模型** 表现出兴趣，其中一位用户发现与他们已经在运行的更大模型相比，**Llama 的发布令人失望**。
  - 用户对 **iOS 版本** 的 ChatterUI 感兴趣，但开发者提到 **Mac** 的成本是发布到 **App Store** 的障碍。
  - 注意到该应用在搭载 **Llama 3.2** 模型的 **Android** 设备上的性能，prompt processing 达到 **50 tokens per second**，生成达到 **10 tokens per second**。
- **Llama 3.2 在 EU 被禁止使用吗？** ([Score: 71, Comments: 132](https://reddit.com//r/LocalLLaMA/comments/1fqhjs9/is_llama_32_banned_to_use_in_eu/)): 据报道，Huggingface 上的 **Llama 3.2** 许可证 **限制了居住在 EU 的个人和公司** 使用多模态模型的权利，尽管这种限制在 **GitHub** 许可证中 **并不存在**。这种差异引发了关于新 Llama 多模态版本中潜在的 **数据收集和用户指纹识别** 的疑问，这可能是为了应对 **EU 数据保护法**。
  - **EU AI Act** 和 **GDPR** 被认为是 **Meta** 在 EU 限制 **Llama 3.2** 的原因，同时也存在对未经同意使用个人数据进行训练的担忧。**AI Act 的实施** 将于 **2025 年 2 月** 开始，这引发了关于 Meta 采取预防措施的疑问。
  - 讨论集中在 **EU 法规** 对 AI 模型的影响，特别是关于 **生物识别分类** 和 **版权问题**。一些用户对 EU 法规表示沮丧，而另一些人则捍卫其对数据保护的重要性。
  - 关于 **本地** 运行 AI 模型是否可以豁免于 EU 法规存在争论。提到了 GDPR 中的 **“household exemption”**，但监管机构和法院将如何针对开源 AI 模型解释这些法律仍存在不确定性。


**主题 2. AI 下一代硬件：NVIDIA RTX 5090 规格泄露**

- **[RTX 5090 将配备 32GB GDDR7 (1568 GB/s) 显存](https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked)** ([Score: 87, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1fq2aad/rtx_5090_will_feature_32gb_of_gddr7_1568_gbs/)): 传闻 **RTX 5090** 将配备 **32GB GDDR7 显存**，带宽为 **1568 GB/s**。这代表了相对于当前一代的重大升级，可能为 AI 和图形密集型应用提供实质性的性能提升。
  - 价格讨论占据主导地位，用户推测 **RTX 5090** 的价格可能为 **$3500** 甚至 **$5090**。一些人希望上一代显卡能降价，但 **3090s** 的价格在某些地区保持稳定或有所上涨。
  - 该显卡的 **600W 功耗** 引发了对功率限制的担忧。用户对 **32GB 显存升级** 的意义展开辩论，有人称其为“巨大”，而另一些人则认为在经历了三代 24GB 之后，这仍然不足。
  - 显存带宽计算受到了审视，用户建议正确的数字应该是 **1792 GB/s** 而不是 1568 GB/s。注意到在单张显卡上运行 **70B 模型** 甚至可能是 **90B Llama 3.2** 的潜力。

**主题 3. 大语言模型（LLM）的量化与性能分析**

- **评估性能损失：Qwen2.5 32B Q4_K_M 与 BF16 MMLU PRO 评估结果对比** ([Score: 79, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fps3vh/estimating_performance_loss_qwen25_32b_q4_k_m_vs/))：该帖子通过不完整的 **MMLU PRO** 评估，对比了 **Qwen2.5 32B** 模型在 **Q4_K_M** 量化版本与 **BF16** 版本下的性能。尽管数据集不完整存在局限性，但该对比提供了量化导致性能下降的粗略估计，展示了各学科类别的结果，总体性能从 **66.58%** (BF16) 下降到 **64.23%** (Q4_K_M)。评估使用 **Ollama** 作为后端和 GitHub 托管的评估工具，并提供了具体的配置细节。
  - 讨论了 **Qwen2.5 32B** 模型在 **MMLU-Pro 排行榜**上的表现，用户注意到其性能接近 72B 版本。该排行榜允许通过上传 JSON 文件提交**自报结果**，这引发了关于提交来源可靠性的疑问。
  - 用户表示有兴趣将 **Q4_K_M** 量化与使用合适校准数据的 **IQ4_XS / NL** 等其他格式进行对比。一些人建议创建排序柱状图，以便更好地可视化不同量化版本之间的性能差异。
  - **Q4_K_M** 量化在历史等某些类别中表现出意料之外的提升，这被归因于量化过程中可能的“运气成分（lucky dice rolls）”。用户还讨论了与 **BF16** 相比极小的性能损失，认为这是换取更低资源需求的价值权衡。
- **在 8 核笔记本上使用 Rust 以 21 tok/s 的速度运行新款 Llama 3.2 1B 模型推理** ([Score: 58, Comments: 8](https://reddit.com//r/LocalLLaMA/comments/1fqb0zd/running_inference_on_the_new_llama_32_1b_model_at/))：作者扩展了其**基于 Rust 的项目**，以支持新款 **Llama 3.2 1B 和 3B 模型**的推理，在不使用 ML 库的情况下，在 **8 核笔记本**上达到了 **21 tokens 每秒**的速度。该项目已发布在 [GitHub](https://github.com/samuel-vitorino/lm.rs)，现在包含一个**轻量级 WebUI**，作为本地 CPU 推理终端聊天界面的替代方案。
  - 用户称赞了该项目的**性能**，并将其与 **iPhone** 的处理能力进行了对比。作者强调了从零开始构建的**学习体验**，将其描述为“当你最终搞定它时，痛苦与回报交织的感觉”。
  - 讨论了对 **Windows GUI 聊天可执行文件**的需求。作者承认这是一个被要求的功能，并建议调整后端以兼容支持多操作系统的现有**前端**。
  - 关于使用**浏览器**还是**原生应用**作为 GUI 产生了争论。浏览器因高 RAM 占用以及相比原生应用较低的 CPU/GPU 性能而受到批评。


**主题 4. 创意写作与角色扮演 AI 模型的进展**

- **[这是你们中某些人一直在等待的模型 - Mistral-Small-22B-ArliAI-RPMax-v1.1](https://huggingface.co/ArliAI/Mistral-Small-22B-ArliAI-RPMax-v1.1)** ([Score: 36, Comments: 22](https://reddit.com//r/LocalLLaMA/comments/1fpvj0o/this_is_the_model_some_of_you_have_been_waiting/))：**Mistral-Small-22B-ArliAI-RPMax-v1.1** 是一款用于创意写作和角色扮演的新型 AI 模型。该模型基于 **Mistral 22B 参数**基座，旨在擅长**基于角色的交互**，与之前的版本相比，提供了更好的连贯性和创造力。
  - **Mistral Small 22B ArliAI RPMax v1.1** 模型的训练和评估损失（eval loss）均低于 1.0，超过了 **Llama 3.1 70B** 版本。这种表现表明，尽管该模型参数量较小，但在创意写作和角色扮演任务中可能表现出色。
  - **RPMax 数据集**经过精选，消除了重复和合成生成内容，专注于质量而非数量。训练方法采用单轮训练（single epoch）、低梯度累积和较高的学习率，以防止对特定角色套路或故事产生过拟合。
  - 用户对该模型在短篇小说写作中的表现表示关注，并要求**公开数据集**。一些人询问了 **VRAM 需求**以及在资源有限的系统上运行该模型的 **EXL2 量化**选项。

- **Abliteration 不仅影响模型的行为和响应方式，还影响其虚构文学角色的思维和反应方式** ([Score: 58, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1fqfmu6/abliteration_doesnt_only_effect_how_the_model/))：该帖子讨论了 **"abliteration"** 对 **AI language models** 产生的一个意想不到的后果，指出它不仅影响模型的直接响应，还影响模型创建的 **fictional characters** 的行为。作者观察到，**abliterated models** 倾向于生成在通常会表现出愤怒、反抗或不安的情况下反应更 **positively and agreeably**（积极且随和）的角色，从而有效地从模型及其虚构创作中消除了拒绝行为。
  - 用户使用 **system prompts** 测试了 **abliterated models**，发现它们仍然可以被引导去拒绝请求。一些人认为这些模型更适合作为 **work tools**，特别是在 **healthcare** 等对合规性要求极高的领域。
  - **Abliteration** 的影响因应用程度而异。一些模型，如 **Gemma 2 9b**，即使在 vanilla 状态下也会表现出意想不到的行为（例如 "homicidal bias"）。[EQ Bench creative writing table](https://eqbench.com/creative_writing.html) 表明 **Gemma2 finetunes** 在这一领域表现良好。
  - 一些用户注意到 **abliterated models** 可能仍然存在审查，但通过对请求的误解或重新解释来表达。这种行为可能会延伸到角色扮演场景，影响虚构角色的反应方式。


**Theme 5. Hugging Face 里程碑：100 万个模型**

- **Hugging Face 刚刚突破了 1,000,000 个模型** ([Score: 167, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1fpx9ve/hugging_face_just_passed_1000000_models/))：**Hugging Face** 达成了一个重要的里程碑，其平台上可用的模型数量超过了 **1,000,000** 个。这一成就由 **Julian Bilcke** 在 X（原 Twitter）上宣布，并可在 [Hugging Face models page](https://huggingface.co/models) 上验证，展示了该平台广泛的机器学习模型收藏。
  - **Duplicate models** 在 Hugging Face 上非常普遍，用户注意到同一个模型的多次上传（例如 Llama-3.2-1B-Instruct.Q4_K_M.gguf）以及存疑的微调声明。**SomeOddCodeGuy** 提到对于旧模型能看到 "**5-15 个 q4 或 q5 gguf repos**"。
  - 用户讨论了 **evolutionary AI development** 的潜力，**balcell** 建议将 weights 视为 DNA 并引入遗传算法特性。**involviert** 分享了一个成功的小规模进化模拟示例。
  - 人们对模型的质量和功能表示担忧，**remyxai** 指出在查询 hub APIs 时，“**有一半的时间没有 model card**”。其他人则质疑有多少模型实际上能发挥其预期的功能。

## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 研究与模型进展**

- **Google DeepMind 的 AlphaChip 变革微芯片设计**：[Google DeepMind 宣布](https://www.reddit.com/r/singularity/comments/1fpx6sh/google_deepmind_our_ai_for_chip_design_method/)，其 AI 驱动的芯片设计方法 AlphaChip 显著改进了微芯片的设计流程。这一进步可能会加速 AI 硬件的开发。

- **新的 "blueberry" 图像生成模型出现**：一个[名为 "blueberry" 的神秘新图像生成模型](https://www.reddit.com/r/singularity/comments/1fpwuu7/a_new_mysterious_image_gen_model_called_blueberry/)出现在排行榜上，其表现优于 FLUX.1 等现有模型。其来源尚不清楚，但有人推测它可能来自 OpenAI。

- **Google 的 NotebookLM 增加音频和视频输入**：[Google 的 NotebookLM 工具现在允许用户提交 YouTube 视频和音频文件](https://www.reddit.com/r/singularity/comments/1fq02im/notebooklm_now_allows_submitting_youtube_videos/)作为知识库来源，扩展了其多模态能力。

**AI 行业与公司新闻**

- **OpenAI 领导层变动**：最近 [OpenAI 发生了多起高层离职事件](https://www.reddit.com/r/OpenAI/comments/1fpt5gy/one_left/)，包括 Mira Murati、Bob McGrew 和 Barret Zoph。这引发了关于公司内部潜在问题的讨论。

- **OpenAI 计划建设大规模数据中心**：OpenAI 已[要求美国政府批准 5GW 的数据中心](https://www.reddit.com/r/singularity/comments/1fpx4ml/openai_asked_us_to_approve_energyguzzling_5gw/)，凸显了开发先进 AI 所需的巨大计算能力。

- **Sam Altman 推动快速突破**：报告显示 Sam Altman 正在[向 OpenAI 员工施压，要求迅速将研究突破转化为公开发布的产品](https://www.reddit.com/r/singularity/comments/1fq93b6/sam_altman_says_in_the_next_couple_of_years_we/)，这可能会加速 AI 的进展。

**AI 政策与社会影响**

- **联合国优先考虑 AI 治理**：[联合国呼吁以对待气候变化的紧迫感来对待 AI](https://www.reddit.com/r/singularity/comments/1fq3811/the_united_nations_wants_to_treat_ai_with_the/)，这标志着全球对 AI 社会影响的关注日益增加。

- **美国政府成立 AI 基础设施工作组**：拜登政府[成立了一个工作组来协调 AI 数据中心基础设施的政策](https://www.reddit.com/r/singularity/comments/1fpx4ml/openai_asked_us_to_approve_energyguzzling_5gw/)，展示了政府对 AI 发展的参与度不断提高。

**AI 模型发布与改进**

- **Flux.1 Dev 增加 ControlNet Outpainting**：[Flux.1 Dev 模型现在支持 ComfyUI 中的 ControlNet Outpainting](https://www.reddit.com/r/StableDiffusion/comments/1fq1wfa/flux1_dev_controlnet_outpainting_comfyui/)，扩展了其图像生成能力。

- **Elektroschutz LoRA 发布**：一个新的 [名为 Elektroschutz 的 Stable Diffusion LoRA](https://www.reddit.com/r/StableDiffusion/comments/1fqgo3l/elektroschutz_styled_warnings_nobody_asked_for/) 已发布，展示了开源 AI 模型的持续创新。

---

# AI Discord 内容回顾

> 由 O1-mini 生成的摘要之摘要的总结

**主题 1. 语言模型性能与新发布**

- [**ColQwen2 霸榜 Vidore 排行榜**](https://x.com/manuelfaysse/status/1839657285053788483)：基于 **Qwen2-VL backbone** 的 **ColQwen2** 模型取得了显著的 **+5.1 nDCG@5** 评分，在 Vidore 排行榜上超越了 **colpali-v1.1**。
  
- [**Phi-3.5 的审查引发社区辩论**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**Microsoft** 的 **Phi-3.5** 模型因其广泛的审查制度而受到批评，导致用户开始在 **Hugging Face** 上探索 [**无审查版本**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)。

- [**Llama 3.2 增强视觉与 Token 处理**](https://github.com/ggerganov/llama.cpp/issues/8010)：**Llama 3.2 11B Vision** 模型现在支持高达 **128k tokens**，并引入了改进的视觉功能，尽管性能基准测试显示结果褒贬不一。

**主题 2. 工具、集成与新功能**

- [**Aider 推出架构师/编辑器模式以实现高效编程**](https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-architect.md)：**Aider** 中新的 **architect/editor mode** 简化了编码工作流，能够利用 **o1-preview** 和 **Claude 3.5** 等模型更快地修复 Bug。

- [**OpenInterpreter 首次推出 Electron 前端**](https://x.com/parrotexplore/status/1839721139515302137)：**OpenInterpreter** 发布了 **Electron 前端**，增强了用户体验并促进了更广泛的社区参与。

- [**LangChain 集成 Langfuse 和 PostHog 用于 MistralAI 追踪**](https://t.co/KGxjjoO0vM)：一篇 [**教程**](https://t.co/KGxjjoO0vM) 展示了如何在 **LangChain** 中设置 **Langfuse**，通过 **PostHog** 进行全面的 **LLM 应用监控**和**用户分析**。

**主题 3. AI 工作负载中的硬件与 GPU 性能**

- [**传闻 NVIDIA RTX 5090 将配备 32GB VRAM**](https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs)：推测称即将推出的 **NVIDIA RTX 5090** 将包含 **32GB VRAM** 版本，而 **RTX 5080** 可能会在最初发布 **16GB** 版本后获得 **24GB** 的升级。

- [**TensorWave 向社区提供 MI300X GPU**](https://github.com/NVIDIA/TransformerEngine/pull/1019)：来自 **TensorWave** 的 **Darrick** 宣布向社区成员提供 **MI300X** 单元，旨在提高 **GPU 采用率**和**教育计划**。

- [**AMD GPU 在 AI 基准测试中表现不佳**](https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks)：据报道，**AMD GPU**（如 **5700 XT** 和 **7900 XTX**）在 **Stable Diffusion** 和 **Blender** 等生产力任务中落后于 **NVIDIA 3070**，凸显了性能差异。

**主题 4. 部署更新与 API 增强**

- [**Cohere 发布具有增强聊天能力的 API v2**](https://docs.cohere.com/reference/chat-v2)：**Cohere** 的 **API v2** 引入了新的端点，如 **v2/chat**，其功能包括 `messages` 参数和**系统消息支持**，增强了**聊天交互**。

- [**OpenRouter 针对 Gemini 模型转向基于 Token 的计费**](https://x.com/OpenRouterAI/status/1839738812877918617)：**OpenRouter** 针对 **Gemini** 模型从按字符计费转为按 **tokens** 计费，调整价格后预计为 **Flash** 和 **1.5 Pro** 模型降低约 **50% 的成本**。

- [**Meta 的 Orion AR 眼镜集成到 Perplexity AI**](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g)：**Meta** 的 **Orion AR 眼镜**已接入 **Perplexity AI**，旨在彻底改变**增强现实**环境中的用户交互。

**主题 5. 模型训练与优化技术**

- [**DSPy 与 Langtrace 集成以实现高级实验管理**](https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy)：**DSPy** 现在支持 **Langtrace**，能够自动捕获 **traces**、**checkpoints** 和**评估分数可视化**，显著增强了 **AI 实验工作流**。

- [**微调 Llama 模型引发过拟合担忧**](https://github.com/unslothai/unsloth/issues/1040)：用户报告了在**微调 Llama 3.2-3B** 时遇到的挑战，强调了低训练损失带来的**过拟合**风险，并强调了正确进行**数据处理**和 **tokenizer 调整**的必要性。

- [**LoRA+ 优化提高模型训练效率**](https://github.com/axolotl-ai-cloud/axolotl/pull/1932)：**LoRA+** 优化参数已更新以修复默认学习率问题，增强了**模型训练**过程的效率和稳定性。

---

# 第一部分：高层级 Discord 摘要

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Architect/Editor 模式简化编码流程**：Aider 中新的 **architect/editor 模式** 增强了编码工作流，配合 **o1-preview** 和 **Claude 3.5** 等模型可以更快地修复 Bug。
   - 用户建议在设计任务中利用 **Sonnet 3.5** 以实现效率最大化。
- **鼓励进行模型性能基准测试**：建议用户对 **o1-preview**、**o1-mini** 和 **Sonnet 3.5** 等各种模型组合进行基准测试，以优化性能。
   - 性能可能因项目规模和编辑上下文而异，这表明量身定制的设置能提供最佳效果。
- **提议新增 /copy 命令**：一项关于新增 **/copy 命令** 的提案旨在让用户轻松地将最后的 LLM 输出复制到剪贴板以便后续使用。
   - 该功能增强了工作流，特别是对于那些频繁使用 **/ask 命令** 的用户。
- **讨论 Streamlit 的交互限制**：成员们注意到 **Streamlit** 在 Aider 使用场景中存在局限性，建议为了提高交互性有必要进行重新设计。
   - 虽然重新设计的潜力得到了认可，但目前该小组并未将其视为优先事项。
- **关于 Token 使用情况的观察**：讨论集中在 Aider 的 **token usage** 上，建议将文件数量控制在 1-3 个以避免性能下降。
   - 建议成员使用 `/tokens` 来监控使用情况，因为超过 **30k tokens** 可能会导致不可预测的行为。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Molmo 与 LM Studio 的兼容性**：新的 Vision 模型在短期内不会在 LM Studio 中得到支持，因为它们与 **llama.cpp** 不兼容。
   - 用户注意到 **Llama 3.2 11b** 与 **3.1 8b** 类似，但增加了参数以增强视觉功能。
- **关于 Llama 3.2 文本生成的疑问**：社区对 **Llama 3.2** 的 token 支持提出了疑问，有说法称其可以处理高达 **128k tokens**。
   - 关于该模型的性能以及与集成 Bug 相关的问题，出现了褒贬不一的报告。
- **LM Studio 的升级疑虑**：用户对从版本 **0.2.31** 升级到 **0.3.x** 的模型兼容性和设置保留表示不安。
   - 已确认过渡到 **0.3.x** 不会导致数据丢失，尽管它会替换之前的版本。
- **NVIDIA GPU 传闻升温**：传闻指出即将推出的 **NVIDIA RTX 5090** 可能配备 **32GB VRAM**，而 **RTX 5080** 在发布 **16GB** 版本后可能会推出 **24GB** 变体。
   - 对于 **5080** 的能力存在广泛质疑，用户声称它无法满足当前的编程和 AI 需求。
- **LLM 性能压力测试建议**：为了进行有效的压力测试，用户建议在 **LM Studio** 中采用本地服务器 API 调用，以高效管理多个请求。
   - 一位成员正在制作一个专注于这些压力测试方法的教程，强调使用自定义数据集。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **呼吁组建 Llama 3.2 模型工作组**：如 [此 GitHub issue](https://github.com/ggerganov/llama.cpp/issues/8010) 中所述，有人提议创建一个工作组将 **Llama 3.2 vision models** 集成到 **llama.cpp** 中。
   - 该 issue 指出，一旦相关组件完成重构，多模态支持即可恢复。
- **对优化 Cerebras 芯片代码的兴趣**：关于为 **Cerebras chips** 优化代码的讨论凸显了社区对获取有效使用见解的渴望。
   - 成员们对联系 Cerebras 的相关人员以获取该硬件的额外指导非常感兴趣。
- **寻找适用于 Windows 的最新 Triton Wheel**：一位成员正在寻找适用于 Python 3.10 的最新编译版 [Triton wheel for Windows](https://link.to.triton.windows)，这反映了更广泛的兼容性需求。
   - 围绕安装问题的社区参与继续成为多个平台上 Triton 用户的焦点。
- **分享 M2 Pro 基准测试**：一位成员对他们的 **M2 Pro benchmarks** 表示兴奋，并引用了 [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) 在设备上执行扩散模型的推理。
   - 他们展示了在实际语境中强化 M2 Pro 基准测试能力的视觉效果。
- **TensorWave 提供 MI300X 以促进采用**：来自 TensorWave 的 Darrick 宣布可能向社区成员提供 **MI300X** 单元，旨在加强对其使用的教育。
   - 这一机会引发了积极的参与，成员们对这一提议表示兴奋。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama 模型微调引发困惑**：用户讨论了微调 **Llama models** 的细微差别，指出对 `chatml` 等数据格式以及为 special tokens 调整 tokenizer 设置的必要性存在困惑。
   - 成员们对过拟合表示担忧，警告低训练损失（training losses）可能预示着模型陷入了记忆陷阱。
- **模型 Checkpoint 加载错误显现**：一位用户在尝试加载 `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` 模型时遇到了数据不匹配错误，并指出了具体的异常。
   - 这引发了排查讨论，建议将重点放在尺寸和配置设置上，认为这些可能是导致问题的元凶。
- **围绕新显卡的推测**：社区成员讨论了即将推出的 GPU（如 **5090**）的规格和发布传闻，尽管存在怀疑，但普遍预测会有 **32GB VRAM** 的选项。
   - 观点差异很大，这表明虽然传闻四起，但仍需要实际的基准测试（benchmarks）来平息争议。
- **数据打包（Data Packing）提升训练效率**：成员们强调，通过数据打包，训练框架可以管理不相关的部分，从而简化流程并实现对后续 token 的高效预测。
   - 据指出，这种技术通过对多个样本的有效管理，显著改善了训练动态。
- **Transformers 更新与模型兼容性**：用户确认已安装最新的 **transformers** 版本 (4.45.1)，这表明他们正在持续努力优化模型实现。
   - 围绕量化（quantization）挑战的讨论，特别是针对 **Phi3.5**，展示了由于致命的 **vocab size mismatch** 错误而需要采取替代策略的需求。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **未审查模型（Uncensored Models）的挑战**：用户注意到某些 **Hugging Face models** 受到审查，导致难以使用 12B chat 模型创建游戏机器人，并建议使用 **Venice.ai** 等替代方案。
   - 这一讨论强调了在更广泛的创意应用中对未审查模型的需求。
- **探索 Neuralink 的 CUDA 实现**：一位参与者分享了在 **Neuralink** 中使用 **CUDA** 以增强高级 GPU 编程中模型性能的见解。
   - 这对于提高各种 AI 应用的执行效率具有重要意义。
- **阿里巴巴推出 MIMO 技术**：**Alibaba** 推出了 **MIMO**，这是一种能够通过简单输入创建逼真人物视频的新型 AI，并通过包括 **Interactive Scene Control** 在内的 **10 个演示** 进行了展示。
   - 该技术展示了 AI 生成内容中全新沉浸式体验的潜力。
- **寻求文本转视频（Text-to-Video）模型训练仓库**：有人请求提供专注于 **text-to-video (T2V)** 模型**分布式 GPU 训练**的仓库，表明需要增强训练资源。
   - 为了提供帮助，有人建议查看 [CogVideo SAT finetuning](https://github.com/THUDM/CogVideo/blob/main/sat/README.md) 等资源。
- **专家黑客提供的网络安全服务**：一名自称专家黑客的人员提供了各种**网络安全（cybersecurity）课程和服务**，并邀请在这些领域进行合作。
   - 这凸显了 AI 与网络安全之间有趣的交集，这在当今的技术格局中变得越来越重要。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Token 计数变更**：OpenRouter 将对 **Gemini** 模型从按字符计数转为按 **tokens** 计数，这使得 `/activity` 页面上的 token 数量减少了约四倍。
   - 此次调整导致单价翻倍，但对于 Flash 和 1.5 Pro 模型，预计可实现约 **50%** 的成本降低。
- **Llama 3.2 Vision 参数讨论**：用户询问了 **Llama 3.2 vision** 的参数设置以避免被拒绝，特别是在进行吸引力评估时。
   - 共识认为，侧重安全性的训练可能会阻止模型对此类查询做出充分响应。
- **数据库升级停机计划取消**：原定的数据库升级停机计划已取消，服务将保持正常运行。
   - 升级的后续调度更新将在确定后另行通知。
- **Chatroom UI 重大升级**：OpenRouter 宣布了 Chatroom 的全新 UI，默认折叠显示带有推理过程的模型响应，从而提高清晰度。
   - 官方承诺将进一步增强 UI，旨在提供更好的用户界面体验。
- **OpenRouter 遭遇速率限制**：有用户报告遇到 **429 Resource Exhausted** 错误，表明模型因超出速率限制 (rate limit) 而无法处理请求。
   - 目前正在努力与 Google 协商更高的速率限制，以缓解这些问题。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 频道礼仪澄清**：由于在错误频道发布内容引发了误解，官方对*频道的适用性进行了简要说明*。部分成员仍然乐于看到非 Cohere 相关内容的分享。
   - 一位成员对他们项目的启动表示乐观，并感谢社区提供的发帖指引。
- **Embed-English-v3 模型微调受阻**：关于 **embed-english-v3** 模型微调的咨询表明，目前**没有任何嵌入器 (embedder) 可以进行微调**。
   - 建议对于需要特定调整的用户，可以使用来自 Hugging Face 的**自定义嵌入模型**。
- **API v2 端点正式上线**：新的 API **v2** 端点已发布，显著增强了 **Chat V2**，并引入了 `messages` 参数等新功能。更多信息可在 [API Reference](https://docs.cohere.com/reference/chat-v2) 中找到。
   - 用户讨论了测试密钥速率限制的影响，明确了限制是基于账户的，因此轮换密钥的收益大打折扣。
- **文化多语言 LMM 基准测试势头强劲**：**MBZUAI** 团队正在构建一个涵盖 **100 种语言** 的**文化多语言 LMM 基准测试**，旨在改进其多模态数据集。
   - 协助翻译的志愿者将被邀请作为共同作者参与 **CVPR 2025** 的论文提交，这是一项社区驱动的努力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Tiled Upscale 提供比 ADetailer 更慢的替代方案**：**Tiled Upscale** 可以替代 **ADetailer** 并达到类似效果，但由于它处理整个图像，其速度要慢 **50 倍**左右。
   - 这种较慢的替代方案引发了在需要针对特定区域进行详细放大时效率如何的问题。
- **AMD GPU 在生产力任务中表现不佳**：讨论探讨了 AMD GPU（如 **5700 XT**）在 **Stable Diffusion** 和 **Blender** 任务中表现乏力，证明其更适合游戏。
   - 用户报告称，在生产力基准测试中，**3070** 的表现优于 **7900 XTX**，凸显了 GPU 性能的差异。
- **翻新 GPU 受到青睐**：选择翻新 GPU 而非二手 GPU 的优势引发了热烈辩论，焦点在于通过维修和检查提高的可靠性。
   - 一位用户分享了使用翻新 **3090 TI** 的经验，强调其性能几乎与新显卡一样好。
- **SSD 对加载时间至关重要**：确认的研究结果表明，在 **Stable Diffusion** 中使用 **SSD** 与传统 HDD 相比，可以将模型加载时间缩短 **10 倍或更多**。
   - 成员指出，在 **M.2 SSD** 上运行模型比旧技术能显著提升图像生成速度。
- **物体尺寸的创意提示词 (Prompting)**：参与者分享了在图像生成中设置物体尺寸的有效提示词技巧，建议使用各种描述性词汇。
   - 虽然有人开玩笑地提出了 *'yuge'* 和 *'bigly'* 等幽默短语，但最终大家还是更倾向于使用简单的术语。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity UI 问题困扰用户**：多名用户在 Perplexity 网站上遇到错误，报告称交互导致 `net::ERR_BLOCKED_BY_CLIENT` 错误，而 Android 应用仍可正常运行。
   - 这导致了用户的极大挫败感，特别是该问题在桌面和移动浏览器上持续存在。
- **API 功能引发咨询**：用户表达了希望通过 Perplexity API 获取生成式 AI 最新动态的愿望，并对目前特定 API 功能的限制提出疑问。
   - 用户对现有解决方案的稳健性表示担忧，并提出需要探索改进方案。
- **订阅促销活动造成困惑**：由于一名用户在无法获得访问权限的情况下尝试兑换 Pro 订阅的促销代码，导致挫败感增加，并引发了关于账户转移的进一步咨询。
   - 其他用户参与进来，澄清了转移订阅所涉及的步骤。
- **Meta 的 Orion AR 眼镜提升体验**：Meta 最近发布的 [Orion AR 眼镜](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g) 旨在彻底改变增强现实中的用户交互。
   - 初步反馈表明，这可能会显著改变用户在虚拟环境中的参与方式。
- **OpenAI 转向营利性未来**：OpenAI 的 [营利性转型](https://www.perplexity.ai/search/what-happened-with-wordpress-w-V8a7N3D4QMqBc3vZdzXzVg) 在 AI 竞争压力下可能会重塑其融资策略。
   - 这一转变引发了对其未来运营策略影响的疑问。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GPU 显存容量差异引发辩论**：讨论强调了 **5080** 和 **5070** GPU 之间显存大小的差异，其中 **5080** 型号被认为拥有接近 **20GB** 的显存。
   - 成员们注意到跨代显存容量翻倍的趋势，并参考了 **3080** 和 **3090** 型号。
- **DisTrO 论文发布备受期待**：关于 **DisTrO** 论文发布日期的好奇心与日俱增，成员们渴望获得见解，特别是来自最近的一次演讲。
   - 在有人请求更便捷的访问方式后，分享了完整演讲的有用链接。
- **知识图谱与 Bitcoin Ordinal Theory 融合**：一名成员讨论了他们在 **知识图谱（Knowledge Graphs）** 以及源自 **Bitcoin Ordinal Theory** 的独特 Embedding 方面的工作。
   - 他们提出 LLM 从语义丰富性中形成 **基于图的表示（graph-based representations）**，暗示了涌现智能的可能途径。
- **Claude Sonnet 3.5 推理能力提升**：**Claude Sonnet 3.5** 的推理能力有所进步，这归功于对示例推理轨迹（reasoning traces）的利用。
   - 一个突出的例子展示了改进，指明了进一步探索推理增强的未来方向。
- **Hermes 可在 4090 上本地运行**：一名成员确认 **Hermes** 可以使用支持任何 **GGUF 版本** 的 **LMStudio** 在 **4090 GPU** 上本地运行。
   - 这为用户提供了一种无需 API 访问即可查找和使用 **Hermes** 的简便方法。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Agentic Search 项目面临预算削减**：一位开发者分享了他们的 **Agentic Search** 项目因昂贵的计算和 token 使用而失败的经历，这促使他们考虑微调像 **Llama 3b** 这样的小型模型。
   - 这一转变凸显了大型模型给 AI 领域的开发团队带来的资源限制。
- **AI 在学术界的采用率激增**：讨论显示，超过 **50%** 的硕士生在作业中使用 AI 生成的内容，引发了关于生产力与学术诚信的辩论。
   - 参与者对 AI 深入教育环境后对学习可能产生的长期影响表示担忧。
- **AI 的能源消耗引发辩论**：关于 AI 系统**能源消耗**的问题浮出水面，凸显了人们对其环境影响的日益关注。
   - 成员们讨论了随着 AI 技术在各行各业变得更加普遍，采取可持续实践的必要性。
- **改变开发者游戏规则的工具**：一位成员推荐了 **ChatGPT Toolbox** Chrome 扩展，该扩展具有聊天历史搜索和 prompt 管理功能，可提高使用 ChatGPT 的生产力。
   - 关注点还转向了备受期待的 **Orion 模型**，预计它将引入可能彻底改变开发流程的强大新工具。
- **未来一代面临技能丧失的风险**：人们担心，由于对技术的依赖日益增加，未来一代可能会失去像手写这样的传统技能。
   - 参与者幽默地推测了在技术主导的未来社会对基本技能的看法，提出了关于学习工具演变的问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **探索开源模型的赞助**：一位成员询问 Eleuther 是否为开源模型提供任何赞助计划，表示缺乏资源来完整训练他们的项目。
   - 这引发了关于开源领域内此类倡议的社区支持的讨论。
- **LLM 搜索空间模拟的创新**：提出了一个涉及 LLM 抽象搜索空间的概念，利用蒙特卡洛树搜索（Monte Carlo tree search）通过文本扩散（text diffusion）来模拟连续思考。
   - *该方法旨在对计算过程中最连贯的想法进行排名，* 预示着 LLM 架构的潜在进步。
- **比较 FP6 前后的权重分布**：讨论围绕比较模型在 **FP6** 前后的权重分布展开，并暗示使用 [seaborn](https://seaborn.pydata.org/) 等库进行可视化。
   - 目标是观察是否会出现任何异常，成员们建议尝试多个绘图库。
- **ColQwen2 引起轰动**：新模型 **ColQwen2** 被宣布为顶级的视觉检索器，在 Vidore 排行榜上以 **+5.1 nDCG@5** 的分数超越了 **colpali-v1.1**。
   - 该模型利用 **Qwen2-VL 骨干网络**，承诺在视觉检索任务中表现卓越，如[此贴](https://x.com/manuelfaysse/status/1839657285053788483)所述。
- **在 H100 上测试小型模型**：一位成员表示愿意协助在 **H100** 上测试小型模型，表现出对贡献能力的信心。
   - 这激发了讨论中其他人的热情和赞赏。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Langtrace 增强了 DSPy 实验管理**：Langtrace 现在支持运行 DSPy 实验，并能自动捕获 **traces**、**checkpoints** 和**评估分数可视化**，显著改进了管理工作流。
   - 用户可以为每个 pipeline 模块创建独立项目，从而实现针对性优化并轻松部署 checkpointed prompts。
- **MIPROv2 编译运行遇到问题**：用户报告了在 MIPROv2 编译运行期间跟踪评估数据的挑战，尽管在日志中可以看到 traces，这表明可能存在配置失误。
   - 排查发现，在调用 `compile()` 时需要正确的属性，以确保准确的数据跟踪。
- **DSPy 优化工具引发讨论**：成员们对 DSPy 的优化工具表示好奇，类似于 **Tensorboard**，用于在 AI 工作流中高效跟踪指标。
   - 他们分享了关于 [DSPy Visualizer](https://link.to.visualizer) 等工具的见解，以及通过 Langtrace 提供的额外支持。
- **探索用于 RAG 的 DSPy ReAct Agents**：成员们询问了使用 **DSPy ReAct agents** 的示例，特别是结合 **LlamaIndex retriever** 实现 ReAct RAG。
   - 其他用户指出了 **repo (examples/agents/)** 中现有的示例，并承诺很快会添加更全面的示例。
- **RAG Agents 优化的功能请求**：有请求建议将更多**向量数据库**（如 **Qdrant** 和 **LanceDB**）与 DSPy RAG agents 集成，这体现了混合搜索能力的趋势。
   - 关于多模态 RAG pipeline 优化讨论得到了确认，该领域即将有新进展。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **关于 Mojo MAX 桌面背景的投票**：一名成员发起了一项关于 **Mojo / MAX 品牌桌面背景**的投票，邀请大家为可爱的 Mojo 火焰和 MAX 宇航员投票。
   - 反应不一，一名成员简单地回复了 *'Bruh'*，表示惊讶或不感兴趣。
- **现在发帖需要验证**：现在除了列出的少数特定频道外，在所有频道发帖都必须进行验证，以增强控制。
   - 成员被引导至验证频道，那里有一个演示 GIF 解释了操作流程。
- **Mojo 中的错误处理需求**：成员们讨论了 Mojo 当前的错误消息未引用用户代码的问题，这阻碍了 debugging。
   - 由于现有实现的限制，人们对该领域的改进表示担忧。
- **提议为 Variant 类型增加安全标记联合 (Safe Tagged Union)**：一名成员提议将 **Variant** 类型演进为*安全*标记联合 (safe tagged union)，以增强 pattern matching 能力。
   - 讨论集中在确保与现有模型以及 pattern matching 预期之间的兼容性。
- **呼吁增强 Mojo 文档**：成员们一致认为迫切需要改进 Mojo 和 MLIR dialects 的文档，以澄清用户的疑虑。
   - 对现有结构的混淆阻碍了开发，因此需要更清晰的指南。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **FTC 严厉打击误导性 AI 营销**：FTC 发起了针对 AI 工具相关误导性声明的打击行动，特别影响了 **Do Not Pay** 等公司，详见其 [投诉 PDF](https://www.ftc.gov/system/files/ftc_gov/pdf/DoNotPayInc-Complaint.pdf)。
   - 社区成员对 FTC 关于 AI 的定义表示担忧，担心这可能导致许多初创公司受到审查。
- **生成式 AI 的可持续性受到质疑**：一篇文章讨论了当前生成式 AI 热潮可能不可持续的性质，预测可能会发生影响大型科技公司的重大崩盘，链接见其 [新闻通讯](https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter)。
   - 批评者认为，像 **GitHub Copilot** 这样的工具展示了明确的业务价值，这反驳了不可持续性的说法。
- **Geohot 对 AMD 的不满**：Geohot 表达了对 AMD 的不满，在注意到 RDNA3 之后没有重大产品后，质疑该公司的创新轨迹。
   - 这种沮丧情绪反映了社区对 AMD 技术进步停滞和动力的广泛担忧。
- **ColQwen2 模型发布**：社区对 **ColQwen2** 模型的推出表示欢呼，该模型集成了 **Qwen2-VL** 骨干网络，以提高性能和效率。
   - 此次发布标志着视觉识别能力的重大提升，因其在 Vidore 排行榜上的显著影响而受到赞誉。
- **AI 工程面试引发兴奋**：一位成员分享了获得面试机会并可能转为 AI Engineering 角色的热情。
   - *“参加了一个可能转为 AI Engineering 角色的面试，所以我很高兴。”*

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Paragon 构建功能丰富的聊天机器人**：来自 [useparagon](https://t.co/KEE2LOnGoR) 的博客文章和视频展示了他们如何使用 LlamaIndex 的 create-llama 创建一个聊天机器人，与来自 **Slack**、**Google Drive** 和 **Notion** 的客户数据进行交互。
   - *它能够持续且实时地摄取数据，* 使集成非常高效。
- **Langfuse 和 PostHog 增强 MistralAI**：在 [Jupyter notebook](https://t.co/KGxjjoO0vM) 中分享的教程解释了如何设置 **Langfuse** 以跟踪 LLM 应用程序，并集成 **PostHog** 进行用户分析。
   - 这种设置可以为 AI 应用程序提供全面的 **监控 (monitoring)** 和 **分析 (analytics)**，从而简化开发过程。
- **NLTK 的 punkt 资源缺失**：一位用户报告在使用 **NLTK** 时遇到 *Resource punkt not found* 错误。另一位成员建议检查 **llama-index** 的版本，因为最新版本使用的是 *punkt_tab*。
   - 与 NLTK 的 punkt 相关的 *资源问题* 暗示了潜在的兼容性担忧。
- **加载微调模型的挑战**：一位用户在将本地微调的 **Llama3.1-8B** 加载到 GPU 以执行 Text2SQL 任务时遇到困难。成员建议手动加载模型和分词器（tokenizer），并确保其位于 GPU 上。
   - 共享了一个详细的代码片段，展示了如何使用量化（quantization）设置模型以优化性能。
- **优化客户支持的向量搜索**：一种优化向量搜索的拟议策略涉及将问题存储在向量块（vector chunk）中，同时将答案保留在元数据（metadata）中。该方法旨在通过在搜索过程中关注问题的语义来提高准确性。
   - 用户寻求验证，并欢迎对其方法进行进一步改进的建议。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 在担忧声中匆忙发布 GPT-4o**：高管们旨在 Google 开发者大会之前推出 **GPT-4o**，导致发布过程仓促且 **安全数据不完整**，随后该模型被标记为部署风险过高。据报道，员工为了在管理安全评估的同时赶上这一紧迫的截止日期，经历了 **每天 20 小时** 的工作。
   - [Garrison Lovely](https://x.com/garrisonlovely/status/1839655744850772272?s=46) 的一篇文章揭示了安全团队在这次高风险发布期间面临的巨大压力。
- **OpenAI 应对薪酬诉求**：正如 [The Information](https://www.theinformation.com/articles/behind-openais-staff-churn-turf-wars-burnout-compensation-demands) 所概述的，随着 OpenAI 估值飙升，公司正面临员工对薪酬的持续不满。员工已从利润单位（profit units）中套现 **超过 12 亿美元**，这促使研究人员在激烈的人才竞争中以辞职相威胁。
   - 新任 CFO **Sarah Friar** 正在应对这一动荡局面，在领导层更迭之际，许多研究人员要求大幅加薪以留任。
- **OpenAI 的领导层不稳定性**：核心人物 **Mira, Bob, 和 Barret** 最近的离职加剧了 OpenAI 持续的领导层不稳定性，引发了对其长期发展方向的担忧。团队成员的情绪反应反映了在竞争激烈的环境下留住人才所面临的更广泛挑战。
   - 在提升透明度方面，一名实习生幽默地将他们的辞职比作体验 **珍爱新生儿** 的那种苦乐参半的感觉。
- **Substack 接入 iPhone IAP 订阅**：作为 **Substack 畅销作者**，现在可以获得 **iPhone 应用内购买 (IAP) 订阅** 权限，这标志着向移动设备数字出版的转变。这为内容创作者在流行平台上更有效地变现其作品开辟了渠道。
   - 这对移动市场的内容创作者具有重大意义，为增加互动和收入机会铺平了道路。
- **苹果 App Store 管理挑战揭秘**：成员们分享了对 **Apple App Store** 的深刻见解，开发者通常将其视为一场 **恐怖秀 (horror show)**，并讨论了其管理的复杂性。对话强调了开发者在应对 App Store 政策所造成的挑战性环境时，导航其复杂格局的必要性。
   - 虽然现实可能令人畏缩，但讨论揭示了开发者在管理其应用分发的复杂运作时可以采用的潜在策略。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **开源社区在多模态支持方面滞后**：一位成员指出，在整个行业向该方向转型之际，**开源社区**在采用 **多模态支持 (multimodal support)** 方面表现滞后。
   - 这种观点反映了人们对社区 **创新 (innovation)** 速度日益增长的担忧。
- **理解领域主席 (Area Chair) 的角色**：一位成员解释说，**AC** 指的是被称为 **领域主席 (Area Chair)** 的元评审员（meta reviewer），他们在评审过程中发挥着关键作用。
   - 这一见解强调了组织在学术和协作环境中的重要性。
- **用于训练对话分割的 Python 代码片段**：一位用户展示了一个旨在为训练目的而 **分割对话** 的 Python 代码片段，以确保对话不超过 **最大序列长度 (maximum sequence length)**。
   - 他们强调了其效用，特别是在处理长对话的同时保留训练数据集中的上下文。
- **关于 Flex Attention 优化的讨论**：一位成员强调 **Flex Attention** 是一种新的优化实现，与之前的 Attention 方法相比提供了更大的灵活性。
   - 共享了多个资源，包括详细介绍其设计的 [PyTorch 博客链接](https://pytorch.org/blog/flexattention/)。
- **LoRA+ 优化参数更新**：一位成员请求将 `loraplus_lr_embedding` 设置为特定值，并引用了 [最近 GitHub PR 中的修复](https://github.com/axolotl-ai-cloud/axolotl/pull/1932)。
   - 他们解释说，由于未能为该参数使用默认值，该修复至关重要。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **IOMMU 在 Nvidia P2P 中的作用**：一位用户询问在使用 [tinygrad GPU modules](https://github.com/tinygrad/open-gpu-kernel-modules) 时，为什么必须禁用 **IOMMU** 才能支持 **Nvidia P2P**，这表明需要进一步的技术见解。
   - 这种不确定性凸显了一个值得讨论的领域，因为用户正在寻求澄清关键的硬件交互。
- **GPU 云定价竞争引发讨论**：George Hotz 建议 **GPU 的竞争价格为 $0.50/小时**，引发了与 salad.com 和 vast.ai 等供应商选项的比较。
   - 参与者对该价格是否包含 VAT（增值税）以及是否反映了真实的市场竞争力表示担忧。
- **CLOUD=1 功能引发辩论**：关于 **CLOUD=1** 是否包含 **CPU** 资源展开了辩论；用户对强制性的设备连接表示不安。
   - 他们强调，降低成本需要有稳健的解决方案作为补充，以证明该服务模式的合理性。
- **ML 任务的数据上传挑战**：一位成员强调了在连接和上传大型训练数据集方面的严重问题，希望 **tinygrad** 能够缓解这些挫败感。
   - 讨论指出，**data-compute ratio**（数据计算比）对于效率至关重要，特别是在 **mini LLMs 和 CNNs** 等较小模型中。
- **对持久化存储成本的考虑**：用户对 **persistent storage billing**（持久化存储计费）表示担忧，并询问 tinygrad 是否会处理此类费用，因为许多云供应商都有单独的费用。
   - 这指向了关于云服务架构中成本管理的更广泛讨论。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Llama 3.2 11B Vision 免费可用**：TogetherCompute 与 AIatMeta 合作免费提供 **Llama 3.2 11B Vision**，允许开发者尝试开源多模态 AI。在此处访问此创新工具 [here](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free)。
   - 为了获得更高的性能，还提供了 **Llama 3.2 11B & 90B** 的付费 Turbo 端点。
- **无限制访问激发创意**：成员们讨论了无限制访问 **Llama 3.2** 的影响，幽默地建议它可以为整个 **LAION dataset** 添加字幕。这引发了社区围绕创意应用的轻松互动。
   - 这场有趣的对话强调了大家对突破 AI 工具创意边界的共同热情。
- **对家庭照片生成的关注**：一位成员询问了特定应用在生成 **family photos**（家庭照片）方面的效果，突显了对 AI 驱动的个性化内容的浓厚兴趣。这次讨论强调了在日常生活中推动实际应用的日益增长的需求。
   - 该询问反映了人们对 AI 生成相关图像能力的持续好奇。
- **庆祝版权执法胜利**：一位成员分享了一篇 LinkedIn 帖子，庆祝在版权执法方面取得的成功，强调 **正义的一方赢得了这一局**。这被誉为社区诚信的一次重大胜利。
   - 这种情绪营造了积极的氛围，重申了社区对道德实践的承诺。
- **神经网络中位置信息的讨论**：成员们对位置信息如何整合到 latent pixels 的 feature vector 中表示困惑，并注意到 CLIP text embeddings 中缺乏 positional encoding。他们强调模型中的 self-attention 步骤也有助于这一过程。
   - 这带来了关于 convolution 边缘在为 attention 比较提供 **positional data**（位置数据）方面重要性的建设性见解。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **讲座聚焦于 LLM Safety**：鉴于之前对 **AI safety** 的关注，有人对涉及 **social alignment** 的 LLM agents 讲座表示关注。Dawn Song 教授预计将在 **12 月 2 日** 的演讲中探讨这一话题。
   - 这表明关于在教学内容中平衡安全与对齐的对话正在持续进行。
- **课程报名流程确认**：关于课程注册的说明确认，填写 Google 表单即可获取所有课程资料，作业截止日期定为 **2024 年 12 月 12 日**。参与者对这种清晰的沟通表示感谢。
   - 这突显了行政流程的清晰度对于顺畅学习体验的重要性。
- **作业截止日期的困惑**：一名参与者询问了 Berkeley 学生与 MOOC 学生之间作业截止日期的差异，确认所有作业截止日期均为 **2024 年 12 月 12 日**。统一截止日期的规定提高了课程的可访问性。
   - 对学生来说，拥有清晰的时间表至关重要，因为困惑会影响专注度和表现。
- **Quiz 3 可用性混乱**：参与者在查找 **Quiz 3** 时遇到困难，引发了关于其可访问性的讨论，确认该测验在 **MOOC 学生网站**上仍然有效。这导致了更多关于测验结构的咨询。
   - 确保所有学生都能参加测验对于营造公平的学习环境至关重要。
- **实验作业发布时间表受到询问**：一位用户询问了实验作业的发布时间表，注意到 MOOC 网站上的信息存在空白。持续讨论课程清晰度对于学生跟踪作业进度仍然是关键。
   - 关于作业安排的有效沟通将增强学生的参与度和准备工作。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 展示链上分析实力**：一位成员演示了如何使用 [OpenInterpreter](https://x.com/parrotexplore/status/1839721139515302137) 将**可能可行**的代码转变为用于链上分析的**完全功能代码**，并分享了 [Google Colab](https://t.ly/vBSPe) 链接。
   - 这种方法的转变受到了好评，并引发了社区的进一步**转发**。
- **LLaMA 中的多模态支持问题**：讨论集中在 LLaMA 项目自 **#5882** 以来移除的 **multimodal support**，更新将取决于 **llava** 的重构。
   - 建立了一个跟踪线程，整合了见解和相关问题的链接，以便进行后续跟进。
- **令人兴奋的前端开发热议**：一位成员强调了为 OpenInterpreter 开发 **Electron frontend** 的潜力，引发了热烈讨论。
   - 这种热情反映了对 **OpenInterpreter** 社区持续开发的积极态度。
- **HF 发布最新的 90b Vision 模型**：**HF** 宣布更新，引入了 **90b vision** 模型，现已可用于各种视觉任务。
   - 预计这次更新将显著增强相关任务在现实世界中的应用。
- **OpenInterpreter 暖心的影响力**：一位成员分享了 **OpenInterpreter** 如何改变了他们的生活，让他们建立了深厚的友谊并探索了 AI 领域，表达了对社区的感激之情。
   - 他们引用了一年前的[病毒式演示](https://x.com/MikeBirdTech/status/1839750338179674590)，强调了该项目在他们旅程中的变革潜力。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **优化用于客户支持的向量搜索**：一种优化 **vector search** 的新策略旨在将问题存储在向量块（vector chunk）中，将答案存储在元数据（metadata）中，从而提高问题匹配的**精度（precision）**。
   - 该方法专注于问题的**语义（semantics）**，通过过滤无关信息来简化搜索结果。
- **从 Excel 提取上下文的挑战**：一位成员报告了在从复杂的 Excel 文件中进行**上下文提取（contextual extraction）**以生成有意义的 LLM 输出时遇到的困难。
   - 尽管进行了彻底的搜索，他们仍未找到解决此问题的有效方法。
- **CF Booking Chatbot 简化会议室管理**：新构建的 **CF Booking Chatbot** 通过检查可用性和预订来帮助管理会议室，并附带了[展示其功能的演示视频](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langchain-chainlit-ai-activity-7245291326919872512-O06M)。
   - 目前正在计划集成 **Google Calendar** 以实现自动同步，进一步简化流程。
- **Unize Storage 生成高质量知识图谱**：介绍 **Unize Storage**，这是一个可以从任何输入文本创建准确知识图谱的 AI 系统，在处理较大输入时，其表现优于 **LangChain** 的 **LLMGraphTransformer** 等现有系统，**准确率达到 85%**。
   - 这展示了相比 LangChain **55% 准确率**的重大飞跃，突破了图谱生成的界限。
- **Unize Storage 提供免费 API 访问**：**Unize API** 提供免费额度，让用户有机会实验新的 **Unize Storage** 系统，并允许可视化生成的知识图谱。
   - 感兴趣的用户可以使用[此 Playground](https://api.unize.org/signup)开始与系统交互。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **强制执行 PackedDataset 大小限制**：一位成员建议强制要求打包大小不能超过**数据集最大长度的 2 倍**，以防止处理序列时出现错误。
   - 这一建议是作为防止**运行时不一致性（runtime inconsistencies）**的潜在保障而提出的。
- **揭示最大序列长度失效案例**：事实证明，即使单个输入超过 **max_seq_len**，当前的实现也可能失败，尤其是在配置不匹配的情况下。
   - 建议使用显式的令牌长度门控（gating）进行修复，以防止这些**运行时错误**。
- **GitHub 错误讨论亮点**：对话指向了一个 [GitHub 错误](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130)，表明可能决定允许序列大于 **max_seq_len**。
   - 此链接可能阐明了当前处理**打包数据集大小（packed dataset sizes）**背后的逻辑。
- **协作审查要求**：一位成员建议另一位用户在返回后审查此讨论的内容，并强调了其**重要性**。
   - 这突显了故障排除过程中的**协作性质**。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **用户对函数调用评估的困惑**：一位用户对**function calling evaluation**过程表示困惑，并询问是否可以使用结构为 `<prompt>, <llm_response>, <ideal response>` 的自定义**评估数据集**进行分析。
   - 他们特别感兴趣于一个用于有效的**错误细分分析（error breakdown analysis）**的包。
- **对本地 LLM 部署的兴趣**：提出的另一点是希望支持**本地部署的 LLM** 功能，以便使用个人数据集提取错误指标。
   - 用户请求推荐适用于此背景下**function calling capabilities**的代码库。
- **LLM 在应用中的集成**：对话强调了 **Large Language Models (LLMs)** 在 LangChain 和 AutoGPT 等应用中的集成，提到了 **GPT, Gemini, Llama** 和 **Mistral** 等模型。
   - 它们在驱动软件解决方案方面的先进**function calling**能力被认为是一个日益增长的趋势。
- **宝贵资源：Berkeley Function-Calling Leaderboard**：用户强调了 **[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** 是评估 LLM 函数调用能力的资源。
   - 他们指出，该排行榜是基于以用户为中心的函数调用用例。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **探索在 Jamba 中使用 OpenAI SDK**：一位用户询问了如何将 **OpenAI SDK** 与 **Jamba** 结合使用，并对其可行性提出了疑问。
   - 这一询问凸显了用户对于在 **Jamba** 框架内集成不同 AI 工具以增强功能的兴趣。
- **Jamba 的集成查询不断增加**：围绕 **Jamba** 的讨论非常热烈，特别是关于如何利用 **OpenAI SDK** 简化流程。
   - 这些讨论表明开发者对于连接不同框架并增强项目能力的兴趣日益浓厚。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# PART 2: 按频道详细摘要和链接

{% if medium == 'web' %}

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1288943654467145749)** (328 messages🔥🔥): 

> - `Architect/Editor Mode`
> - `Model Comparisons`
> - `Copy Command Feature`
> - `File Handling in Aider`
> - `Token Usage and Efficiency` 

- **架构师/编辑器模式（Architect/Editor Mode）增强**：Aider 中全新的架构师/编辑器模式改进了编码工作流，能够利用 **o1-preview** 和 **Claude 3.5** 等模型实现更快的 Bug 修复和更好的复杂任务处理。
   - 用户反馈使用架构师模式可以简化编码过程，但建议在处理设计相关任务时改用 **Sonnet 3.5**。
- **性能基准测试模型**：鼓励用户对各种模型组合进行基准测试，例如将 **o1-preview** 与不同的编辑器模型（如 **o1-mini** 和 **Sonnet 3.5**）结合，以确定最高效的配置。
   - 反馈表明，最佳性能可能取决于上下文、项目规模和编辑需求。
- **引入 /copy 命令**：提议增加一个新的 `/copy` 命令，允许用户轻松地将 **LLM** 的最后一次输出复制到剪贴板，以便在其他文档中使用。
   - 该功能旨在提升用户体验，特别是对于那些经常使用 `/ask` 命令获取信息的用户。
- **Aider 中的多行输入处理**：用户在讨论向 Aider 粘贴多行文本时注意到，最近的更新允许无障碍地输入此类文本。
   - 强调了简化输入方法对于改善编码体验的重要性。
- **Token 使用和编辑格式**：用户注意到在切换编辑格式时 **Token** 使用量会有所不同，特别提倡使用 'whole' 编辑格式以减少大型编辑过程中的错误。
   - 强调指出，虽然有些人可能会遇到 **Token** 使用量增加的情况，但根据项目选择合适的格式通常会改善整体结果。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://artifacts.e2b.dev/">AI Artifacts by E2B</a>: 未找到描述</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/faq.html#how-do-i-include-the-git-history-in-the-context">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/usage/commands.html#entering-multi-line-chat-messages">In-chat commands</a>: 使用 /add、/model 等聊天内命令控制 aider。</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>: 使用 chat、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/2024/09/26/architect.html">Separating code reasoning and editing</a>: Architect 模型描述如何解决编程问题，而 Editor 模型将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。</li><li><a href="https://code.fittentech.com">免费好用的AI编程助手 Fitten Code - 支持VS Code、PyCharm、Intellj、Visual Studio</a>: 未找到描述</li><li><a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://github.com/paul-gauthier/aider/blob/main/aider/website/_posts/2024-09-26-architect.md">aider/aider/website/_posts/2024-09-26-architect.md at main · paul-gauthier/aider</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json">litellm/model_prices_and_context_window.json at main · BerriAI/litellm</a>: 用于以 OpenAI 格式调用 100 多个 LLM API 的 Python SDK、代理服务器（LLM 网关） - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm</li><li><a href="https://github.com/paul-gauthier/aider/commit/c2c4dbd2a8319f3eab72939f60e2b199a452ff1d">Merge pull request #1595 from jbellis/paste · paul-gauthier/aider@c2c4dbd</a>: feat: 将 /clipboard 重命名为 /paste</li><li><a href="https://github.com/fry69/aider/tree/copy-command">GitHub - fry69/aider at copy-command</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 fry69/aider 的开发做出贡献。</li><li><a href="https://aider.chat/docs/usage/conventions.html">Specifying coding conventions</a>: 告知 aider 在处理代码时遵循你的编码规范。</li><li><a href="https://github.com/paul-gauthier/aider.git">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 paul-gauthier/aider 的开发做出贡献。</li><li><a href="https://github.com/paul-gauthier/aider/pull/1768/files?short_path=cc1e175#diff-cc1e1755d30fcde78f0ba0eb881bb3418d6e5f6b5e29c54de244eeda17059bbb">Proposed changes to the senior / junior editing modes by cschubiner · Pull Request #1768 · paul-gauthier/aider</a>: • 默认使用 Fast Mode 以提高效率：对于快速更改和简单的编码，aider 将使用标准模型。 • 必要时切换到 Architect Mode：对于更复杂和深思熟虑的编码...</li><li><a href="https://fireworks.ai/blog/cursor">How Cursor built Fast Apply using the Speculative Decoding API </a>: Cursor 作为一个 AI 原生 IDE，利用 Fireworks 推理栈增强了其 Instant Apply、Smart Rewrites 和 Cursor Prediction 等功能。该博文介绍了 Speculative Decoding API...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1288943560896417832)** (22 messages🔥): 

> - `与 Aider 的反馈循环`
> - `Streamlit 的局限性`
> - `文件创建问题`
> - `Claude 3.5 的优势`
> - `Aider 中的 Token 使用` 


- **改进与 Aider 的反馈循环**：成员们讨论了在处理 *Streamlit* 等 GUI 时增强反馈循环的挑战，特别是超出仅描述前端需求的部分。
   - 有人建议改进可能需要重新设计 Aider，因为当前的交互似乎有些受限。
- **承认 Streamlit 的限制**：一位用户观察到 *Streamlit* 在 Aider 等用例中似乎受到限制，建议可能需要重新设计以实现更多交互性。
   - 回复暗示虽然重新设计可以改进功能，但目前这并不是高优先级任务。
- **Aider 在文件创建方面遇到困难**：一位用户对 Aider 有时无法按预期创建或编辑文件表示沮丧，经历了不一致的行为。
   - 另一位用户指出，类似问题可能是由于后端 LLM 变慢导致的，这强化了不同会话间行为不确定的观点。
- **评估 Claude 3.5 的优势**：一位成员询问使用 *Claude 3.5* 是否对弱模型和强模型都有质量提升，暗示可能存在微小的成本权衡。
   - 另一位成员确认了在聊天摘要和 commit 消息方面有细微改进，建议用户尝试一下，如果不满意再切换回来。
- **Token 使用影响 Aider 的性能**：一位参与者指出，Aider 在处理较少文件（理想情况下一次 1-3 个）时效果显著更好，特别是为了避免性能下降。
   - 建议使用 `/tokens` 命令监控 Token 使用情况，因为超过 20k 到 30k Token 可能会导致意外行为。



**相关链接**: <a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。

  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 messages): 

fry69_61685: https://erikbern.com/2024/09/27/its-hard-to-write-code-for-humans.html
  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1288938892900307037)** (77 messages🔥🔥): 

> - `Molmo 与 LM Studio`
> - `Llama 3.2 的能力`
> - `LM Studio 更新问题`
> - `使用 CLI 管理本地模型`
> - `LM Studio 中的对话导出` 


- **Molmo 与 LM Studio 的兼容性**：新的 Vision 模型暂时不会在 LM Studio 中得到支持，因为它们与 llama.cpp 不兼容。
   - Llama 3.2 11b 本质上与 3.1 8b 相同，但增加了用于视觉功能的参数。
- **关于 Llama 3.2 文本生成的查询**：Llama 3.2 模型在 Token 限制方面受到询问，混合报告显示它应该支持高达 128k Token。
   - 社区成员讨论了该模型性能及其处理更新的各种问题，提到了存在漏洞的集成。
- **LM Studio 的升级担忧**：用户担心从 0.2.31 版本过渡到 0.3.x 版本，特别是关于模型兼容性和设置保留。
   - 已确认升级到 0.3.x 将替换之前的版本，但不会导致数据丢失。
- **使用 LMS CLI 进行模型管理**：一些用户报告 LMS CLI 无法识别 LM Studio 服务器正在运行的问题，引发了关于故障排除的讨论。
   - 社区分享了关于通过 WebSocket 访问本地模型的发现，并讨论了对官方文档的需求。
- **LM Studio 中的对话导出**：在 0.3.* 版本中，导出对话的功能被移除，导致了对轻松分享讨论的担忧。
   - 用户获悉该功能可能会在未来的更新中回归，因为当前版本是完全重构的版本。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.10668">Language Modeling Is Compression</a>：长期以来，人们已经确定预测模型可以转化为无损压缩器，反之亦然。顺便提一下，近年来，机器学习社区一直专注于训练 i...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI</a>：LM Studio CLI。通过在 GitHub 上创建帐户来为 lmstudio-ai/lms 开发做出贡献。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1288963697695522848)** (246 messages🔥🔥): 

> - `低 Q 量化下 70B 模型的性能`
> - `关于 NVIDIA RTX 5090 和 5080 的传闻`
> - `不同 AI GPU 选项的对比`
> - `LLM 的负载测试方法`
> - `CPU 散热问题与升级` 

- **70B 模型在低 Q 下的高性能表现**：一位用户报告称，使用 **24GB VRAM GPU** 在 **70B 模型**上达到了 **18 tokens/sec** 的速度，强调了其在 **IQ2** 下的可用性。
   - 另一位用户指出，即使是这个速度，输出也远快于手动输入，强调了其在某些任务中的有效性。
- **关于 NVIDIA GPU 的传闻四起**：传闻称即将推出的 **NVIDIA RTX 5090** 可能配备 **32GB VRAM**，而讨论显示在 **16GB** 版本发布后，可能会推出 **24GB** 版本的 **RTX 5080**。
   - 用户对 **5080** 的规格表示怀疑，认为其不足以满足当前的游和 AI 需求。
- **评估 AI 应用的 GPU 选择**：讨论了几款 GPU，包括价格约 **$650** 的 **RTX 3090**、**$850** 的 **3090 TI** 以及 **$300** 的 **P40**，每款都有其各自的性能影响。
   - 关于 AI 工作负载的最佳选择意见不一，许多用户强调了新款 GPU 的重要性以及旧型号的潜在局限性。
- **LLM 性能负载测试**：对于负载测试，用户建议在 **LM Studio** 中使用本地服务器 API 调用，利用自定义数据集高效地发送多个请求。
   - 一位用户表示，他们正在制作在 LM Studio 中进行模型负载测试的教程。
- **对 CPU 散热解决方案的担忧**：关于 CPU 散热展开了讨论，一位用户指出其 **Corsair AIO** 散热器发出嗡嗡声，可能影响性能。
   - 考虑了替代方案和更换选项，强调了保持充足散热以实现最佳硬件功能的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.amazon.ca/PowerColor-Radeon-7900-XT-Graphics-Card/dp/B0BMWHCGBZ/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/kopite7kimi/status/1839343725727941060">来自 kopite7kimi (@kopite7kimi) 的推文</a>：GeForce RTX 5090 PG144/145-SKU30 GB202-300-A1 21760FP32 512-bit GDDR7 32G 600W</li><li><a href="https://tenor.com/view/you-dont-turn-your-back-on-family-you-cant-walk-away-from-family-you-cant-leave-family-behind-you-cant-ignore-family-you-cant-disregard-family-gif-16058425">你不能背弃家人，你不能离开家人 GIF - You Dont Turn Your Back On Family You Cant Walk Away From Family You Cant Leave Family Behind - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/paulwnos-gif-26909845">Paulwnos GIF - Paulwnos - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://wccftech.com/nvidia-24-gb-geforce-rtx-5080-gpu-after-16-gb-first-gaming-blackwell-shipments-spotted/">传闻 NVIDIA 将在 16 GB 版本后推出 24 GB GeForce RTX 5080 GPU，首批游戏 Blackwell 出货量曝光</a>：据 Chiphell 论坛报道，传闻 NVIDIA 的 GeForce RTX 5080 GPU 将在 16 GB 型号发布后获得 24 GB 的升级。</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k-cores-400w/">NVIDIA GeForce RTX 5090 32 GB 与 RTX 5080 16 GB 规格曝光：5090 核心数超 20K 且功耗 600W，5080 核心数超 10K 且功耗 400W</a>：NVIDIA 为玩家准备的下一代 GPU —— GeForce RTX 5090 和 RTX 5080 的规格已由 Kopite7kimi 揭晓。</li><li><a href="https://youtu.be/bJKj1yIc4sA">60 美元的 AI GPU？？？</a>：对 NVIDIA P102-100 进行基准测试。这是一款旧的加密货币挖矿卡，可重新用于 AI 推理。对于那些...的人来说，它非常便宜且具有极高的性价比。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1fqsafn/nvidia_jetson_agx_thor_will_have_128gb_of_vram_in/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs">入门指南 | LM Studio 文档</a>：在 Mac、Windows 或 Linux 上本地运行开源 LLM</li><li><a href="https://www.overclockers.co.uk/8pack-supernova-mk3-amd-ryzen-threadripper-pro-extreme-pc-sys-8pk-00076.html">8Pack Supernova MK3 - AMD Ryzen Threadripper Pro 极致 PC</a>：立即在线订购 8Pack Supernova MK3 - AMD Ryzen Threadripper Pro 极致 PC，享受快速送达服务。</li><li><a href="https://wccftech.com/nvidia-geforce-rtx-5090-32-gb-rtx-5080-16-gb-specs-5090-20k-cores-600w-5080-10k">NVIDIA GeForce RTX 5090 32 GB 与 RTX 5080 16 GB 规格曝光：5090 核心数超 20K 且功耗 600W，5080 核心数超 10K 且功耗 400W</a>：NVIDIA 为玩家准备的下一代 GPU —— GeForce RTX 5090 和 RTX 5080 的规格已由 Kopite7kimi 揭晓。</li><li><a href="https://videocardz.com/newz/nvidia-geforce-rtx-5090-and-rtx-5080-specs-leaked">NVIDIA GeForce RTX 5090 与 RTX 5080 规格泄露 - VideoCardz.com</a>：GeForce RTX 5090 将配备 21760 个 CUDA 核心、32GB GDDR7 显存和 600W 功耗，RTX 5080 配备 16GB 显存。消息来自 Kopite7kimi 本人。这位最可靠的 NVIDIA 爆料者之一现已确认了规格...</li><li><a href="https://www.canadacomputers.com/index.php?cPath=43_557_559&sf=:3_22&co=&mfr=&pr=">选购 Powered By Nvidia 及更多产品 - Canada Computers</a>：未找到描述
</li>
</ul>

</div>
  

---



### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1288956818726457355)** (4 条消息): 

> - `Llama 3.2 vision models`
> - `Cerebras chip optimization` 


- **呼吁组建 Llama 3.2 模型小组**：有人提议组建一个工作组，将 **Llama 3.2 视觉模型** 集成到 **llama.cpp** 中，详见 [此 GitHub issue](https://github.com/ggerganov/llama.cpp/issues/8010)。
   - 该 issue 指出多模态支持已被移除，在相关组件重构后可以恢复。
- **对 Cerebras 芯片代码优化的兴趣**：一位成员询问了关于为 **Cerebras 芯片** 优化代码的努力，以及购买它们是否是明智的选择。
   - 另一位成员表示有兴趣与来自 Cerebras 的任何人取得联系以获取见解，表明希望获得有关此话题的更多信息。



**提到的链接**：<a href="https://github.com/ggerganov/llama.cpp/issues/8010">server: 恢复多模态支持 · Issue #8010 · ggerganov/llama.cpp</a>：自 #5882 以来多模态已被移除。取决于 llava 的重构，我们将能够恢复支持：#6027。创建此 issue 主要是为了跟踪目的。如果有人想...

  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1288942895545581629)** (26 条消息🔥): 

> - `Triton Windows Wheel`
> - `理解 BLOCK_SIZE`
> - `Triton 中的 Inline Assembly`
> - `Triton Kernel 中的 TMA`
> - `MPS 故障处理` 


- **寻找最新的 Windows 版 Triton Wheel**：一位用户正在寻找适用于 Python 3.10 的最新编译版 [Triton wheel for Windows](https://link.to.triton.windows)。这反映了社区对于确保在不同平台上正确安装的持续关注。
- **澄清 Triton 中的 BLOCK_SIZE**：引发了关于 `BLOCK_SIZE` 的讨论，一位成员指出它与线程数（thread counts）不同，将其描述为并行化的维度大小（例如在数组中）。其他人将 `BLOCK_SIZE` 与 CUDA 的 `blockDim` 进行了比较，强调它定义了一个 block 内操作的元素数量。
- **在 Triton Kernel 中使用 Inline Assembly**：关于在 Triton 中执行特定汇编操作的咨询，引出了关于如何使用 [inline assembly](https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise) 且不需要大量 shared memory 的见解。虽然建议使用 Inline assembly，但也有人对其局限性以及与更优化的 broadcast 方法相比的差异表示担忧。
- **探索 Triton Kernel 开发中的 TMA**：一位成员询问在 Triton 中是否有必要使用完整的 TMA 指令，还是使用像 `tl.make_block_ptr` 这样更简单的方法就能获得类似的性能提升。这突显了社区对 Hopper 架构内高效数据管理的探索，以及 Kernel 优化的细微差别。
- **MPS 故障导致执行停止**：有人担心 MPS 故障无法由 Python 管理，导致问题发生时应用程序完全停止。这种情绪反映了对 Apple MPS 更广泛的不满，引发了关于替代框架及其功能的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/python-api/generated/triton.language.inline_asm_elementwise.html#triton.language.inline_asm_elementwise)">triton.language.inline_asm_elementwise &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://pytorch.org/blog/hopper-tma-unit/">深入探讨用于 FP8 GEMM 的 Hopper TMA 单元</a>：摘要</li><li><a href="https://github.com/triton-lang/triton/blob/1e093fbfff2fb3bd4406d9379f7aa62deaf74965/python/test/unit/hopper/test_gemm.py#L56-L57">triton/python/test/unit/hopper/test_gemm.py (位于 1e093fbfff2fb3bd4406d9379f7aa62deaf74965) · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1289006842688700416)** (17 条消息🔥): 

> - `PyTorch Profiler 性能计数器`
> - `torch.flip HIP 错误`
> - `Swin2SR GitHub 仓库`
> - `PyTorch 基准测试仓库`
> - `在 TorchScript 中更新字典` 


- **PyTorch Profiler 性能计数器查询**：一位用户询问在使用 NVIDIA GPU 时，**PyTorch Profiler** 是否需要启用性能计数器，怀疑它只是简单地测量时间和 VRAM。
   - 根据另一位用户的看法，*应该不需要*，因为它并不追踪 L2 命中率等指标。
- **使用 torch.flip 时遇到 HIP 错误**：一位用户报告在将 **torch.flip** 与 **swin2sr** 结合使用时遇到了 **HIP error: invalid device function**。
   - 他们指出代码在 CPU 上运行正常，但在 GPU 上失败，并请求排查帮助。
- **分享 Swin2SR GitHub 项目**：对话中重点介绍了 **swin2sr** 的 GitHub 仓库，这是一个用于图像超分辨率的高效 Transformer，并分享了参考链接。
   - 项目描述显示其与 **ECCV 2022** 会议相关及其具体应用，并建议尝试其功能。
- **寻找用于基准测试的未优化 PyTorch 代码**：一位用户询问是否存在包含简单、未优化的 **PyTorch code** 以及基准测试指标的在线仓库，以便进行练习。
   - 作为回应，另一位成员提供了一个 **nsys tutorial** 链接，该教程演示了如何诊断关于数据加载和 GPU 的性能瓶颈。
- **在 TorchScript 中高效更新字典**：一位成员提出了关于在 **TorchScript** 中处理大批量输入时，如何高效更新包含 long 到 int 映射的 **dictionary** 的问题。
   - 他们指出在编译到 PyTorch 时使用 for 循环的局限性，并寻求替代方案。



**提到的链接**：<a href="https://github.com/mv-lab/swin2sr">GitHub - mv-lab/swin2sr: [ECCV] Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration.  Advances in Image Manipulation (AIM) workshop ECCV 2022. Try it out! over 3.3M runs https://replicate.com/mv-lab/swin2sr</a>: [ECCV] Swin2SR: SwinV2 Transformer for Compressed Image Super-Resolution and Restoration.  Advances in Image Manipulation (AIM) workshop ECCV 2022. Try it out! over 3.3M runs https://replicate.com/...

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1288993317950062613)** (6 条消息): 

> - `.clangd 配置`
> - `CUDA 路径问题`
> - `剖析 Kernel 函数性能` 


- **更新 .clangd 以修复错误**：一位成员成功更新了他们的 `.clangd` 配置文件，通过移除有问题的标志解决了未知命令错误。
   - 尽管进行了这些更改，他们仍然面临与 **libdevice** 和 CUDA 安装路径相关的问题。
- **对 CUDA 安装版本的困惑**：一位成员对错误提示表示困惑，该提示称 `/usr/local/cuda-12.6` 的安装版本是 11.0，尽管路径是正确的。
   - 他们尝试添加 `--cuda-path=/usr/local/cuda` 但没有发现任何改善，导致需要进一步排查。
- **关于剖析 kernel 函数的咨询**：一位成员就如何剖析 HIP kernel 的性能寻求建议，据报道其性能比压缩 kernel 差 5 倍。
   - 他们探索了使用 `clock()` 和独立的计时线程，并向他人询问其他的调试策略。
- **命令行参数混淆**：一位成员承认混淆了 `clang` 和 `nvcc` 的命令行参数，导致了错误的配置。
   - 他们最终成功配置了带有适当标志的新 `.clangd`，这帮助解决了大部分问题。


  

---

### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1288975422062723092)** (5 条消息): 

> - `C 语言图像处理`
> - `书籍的热门章节`
> - `stb_image 库`
> - `Yann LeCun 的提及` 


- **C 语言读取图像的建议**：一位用户询问了关于在 C 语言中读取图像的建议，特别是针对他们正在阅读的书籍第 3 章的内容。
   - 另一位成员推荐了 [stb_image 库](https://github.com/nothings/stb/blob/master/stb_image.h)，并指出对于大多数用途来说，这是一个“即插即用”的解决方案。
- **Yann LeCun 对 Ocean C++ 的认可**：一位成员强调 **Yann LeCun** 提到在 Meta 使用 **Ocean C++** 图像处理库，这为该建议增加了可信度。
   - 这位 AI 和图像处理领域知名人物的认可受到了社区的好评。
- **用户对 stb_image 库的满意度**：在查看了推荐的 **stb_image 库**后，原用户表达了他们的满意，称其完美地完成了任务。
   - 这一正面反馈反映了该库在 C 语言读取图像方面的有效性。
- **书籍章节的受欢迎程度**：另一位用户分享了他们对该书第 4.7 节的喜爱，参与了关于其受欢迎程度的讨论。
   - 他们的热情表明书中的某些部分引起了读者的共鸣。



**提到的链接**：<a href="https://github.com/nothings/stb/blob/master/stb_image.h">stb/stb_image.h at master · nothings/stb</a>：用于 C/C++ 的 stb 单文件公共领域库。通过在 GitHub 上创建一个账号来为 nothings/stb 的开发做出贡献。

  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1288942346355871807)** (138 条消息🔥🔥): 

> - `GPU 模型的 Windows 支持`
> - `FP8 和 Int8 训练问题`
> - `Torchao 性能分析`
> - `量化训练中的问题`
> - `GPU 编程相关的链接和资源` 


- **Windows 支持被证明是有效的**：一位成员分享到，与“只有 Linux 用户对 GPU 编程感兴趣”的理论相反，bitsandbytes 约有 **10-15%** 的下载来自 Windows 用户，这表明兴趣正在上升。
   - 另一位成员幽默地推测，各种设置中可能存在“僵尸” Windows 机器贡献了这些数据。
- **FP8 和 Int8 训练的挑战**：成员们讨论了在尝试加载 FP8 模型时，NVIDIA 系统出现显存溢出 (OOM) 的错误，而 Int8 则运行正常，这表明权重化（weight quantization）未能有效应用。
   - 对话强调了确保 FP8 高效利用内存的必要性，并提出了 FP8 主要是提升计算速度而非减少内存占用的疑问。
- **分析 Torchao 的 CPUOffloadOptimizer**：一位成员通过联系原作者和社区，寻求对 Torchao CPUOffloadOptimizer 性能分析结果的见解。
   - 另一位成员建议创建一个讨论线程，以便更广泛的受众参与对话。
- **模型量化问题**：关于模型行为的持续讨论指出，虽然权重在 FP8 中似乎没有被量化，但它们在 Int8 中可以工作，这引发了对两种情况下内存使用的担忧。
   - 成员们注意到显著的内存消耗，约为 **24,000 MB**，从而引发了关于优化和 FP8 有效性的疑问。
- **对公共资源和链接的反馈**：有人提到在与 Torchao 相关的博客中发现了细微的拼写错误，一位成员指出一个失效链接重定向不当，导致了清晰度问题。
   - 社区鼓励对这类错误进行沟通，以促进更好的 GPU 编程资源建设。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/PyTorch/status/1839696520527929384">来自 PyTorch (@PyTorch) 的推文</a>：我们很高兴正式发布 torchao，这是一个 PyTorch 原生库，通过利用低比特数据类型、量化和稀疏性，使模型更快、更小。我们的技术以易读的方式编写...</li><li><a href="https://scholar.google.com/citations?user=_2_KAUsAAAAJ">Furkan Gözükara</a>：助理教授，计算机工程师，Toros 大学 - 被引用 20 次 - 数据挖掘 - 情感分析 - 文本分类 - 产品聚类 - 聚类</li><li><a href="https://github.com/pytorch/ao/issues/957">这只适用于 Linux 吗？ · Issue #957 · pytorch/ao</a>：我在 Windows 上安装了，但在执行 from torchao.quantization import quantize_ 时失败。pip freeze Microsoft Windows [版本 10.0.19045.4894] (c) Microsoft Corporation。保留所有权利。R:\CogVideoX_v1\...</li><li><a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/bu2mX.gif">黄仁勋 NVIDIA CEO GIF - Huang Jensen Nvidia Ceo - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/pytorch/ao/blob/63cb7a9857654784f726fec75c0dc36167094d8a/torchao/prototype/quantized_training/int8.py#L124">ao/torchao/prototype/quantized_training/int8.py (位于 63cb7a9) · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/quantized_training#int8-mixed-precision">ao/torchao/prototype/quantized_training (位于 main 分支) · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/benchmarks/quantized_training/pretrain_llama2.py">ao/benchmarks/quantized_training/pretrain_llama2.py (位于 main 分支) · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701df4c68e3bc401dc4ddb7e55f2b5dc7854d6fR103-R116>">(进行中，仅限 int8) torchao: fp8/int8 由 bghira 提交 · Pull Request #986 · bghira/SimpleTuner</a>：未找到描述</li><li><a href="https://github.com/bghira/SimpleTuner/pull/986/files#diff-327015d4d445c4efaaa945a93701dc4c68e3bc401">(进行中，仅限 int8) torchao: fp8/int8 由 bghira 提交 · Pull Request #986 · bghira/SimpleTuner</a>：未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1288949705530802259)** (17 messages🔥): 

> - `Edge LLM Challenge`
> - `Integration of Sonnet and Voice`
> - `Prompting vs Speaking`
> - `Meta AR Glasses`
> - `Code Execution on New Platforms` 


- **Edge LLM Challenge 正在组队**：参与者表示有兴趣组队参加 **Edge LLM Challenge**，该竞赛要求团队开发适用于在智能手机上运行的预训练 LLM 压缩方法。
   - 该挑战涉及创建如 **Phi-2**、**Llama-3-8B** 和 **Qwen-7B** 等模型，并使用 **OpenCompass benchmark** 进行评估。
- **探索 Sonnet 的语音功能**：一名成员询问了 **Sonnet** 与语音输入输出的集成情况，好奇它是否支持 TTS 功能。
   - 尽管对 ChatGPT 等现有工具持怀疑态度，但他们承认了对话式 AI 的潜力。
- **相比音频更倾向于文本回复**：一位成员表示个人更喜欢文本回复而非音频，理由是速度和清晰度，尤其是在处理代码等复杂内容时。
   - 他们指出，虽然说话可以提高输入速度，但语音回复缺乏可编辑性带来了挑战。
- **对 Meta AR 眼镜的发展方向感兴趣**：一位参与者对 **Meta** 在开发 AR 眼镜方面的进展表示热烈期待，特别是其在编程领域的应用。
   - 他们强调希望这款眼镜能成为像 **iPhone** 一样切实可用的平台，以释放编程潜力。



**Link mentioned**: <a href="https://edge-llms-challenge.github.io/edge-llm-challenge.github.io/challenge">no title found</a>: no description found

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1288969786352472065)** (2 messages): 

> - `Meetups in Guatemala`
> - `GPU reading/work groups in London` 


- **寻求在危地马拉的见面会**：一名成员提到他们在 **Guatemala**（危地马拉），并询问附近是否有人可以组织见面会，同时也对 **Belize**（伯利兹）和 **Mexico**（墨西哥）等邻近国家感兴趣。
   - 他们对与该地区的其他人建立联系进行讨论或合作持开放态度。
- **伦敦 GPU 小组推荐**：另一名成员征求 **London**（伦敦）任何可以提供合作机会的 **GPU reading or work groups**（GPU 读书或工作组）的推荐。
   - 这突显了在 GPU 社区内进行本地参与和学习的愿望。


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1288990620446232586)** (33 messages🔥): 

> - `RMSNorm integration`
> - `MLP block backpropagation`
> - `Kernel efficiency concerns`
> - `Attention backward pass issues`
> - `RepKV backward debugging` 


- **RMSNorm Backward Pass 已完全集成**：**RMSNorm** 的 **backward pass** 已成功集成，最终 RMSNorm 后的梯度与预期相符。
   - *下一步是 SwiGLU backward*，以进一步推进向 Transformer 的集成。
- **MLP Block 反向传播成功**：已确认 Transformer 的 **MLP block** 反向传播功能正常，计划下一步处理 Attention 块。
   - 记录了一些细微改动，但集成工作正按计划进行且符合预期。
- **对 Kernel 效率的担忧**：有人担心当前的 Kernel 实现浪费了大量线程，特别是在 replicate factor 设置为 4 时，可能导致 **75% 的线程处于 noop 状态**。
   - 为 backward 创建了一个额外的 **scratch buffer**，以解决 dbias_buffer 大小问题。
- **Attention Backward Pass 进展困难**：**Attention** 的 backward pass 引起了关注，其数值看起来高得可疑，特别是由于与 **PyTorch** 相比采用了不同的 replication 方法。
   - 在进行大量调试后，用户怀疑 **RoPE** 或 **repKV** 的 backward 实现中有一个是损坏的。
- **对调试结果感到困惑**：经过大量调试，用户得出结论，问题可能 **不在 repKV** 的 backward，这导致对问题的真实根源感到困惑。
   - 尽管将 repKV backward 的 CPU 参考实现重写得非常安全，用户仍表示感到挫败，并决定暂时休息一下。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1289312419910389771)** (6 messages): 

> - `TensorWave MI300X Offer`
> - `Community Engagement` 


- **TensorWave 提供 MI300X 以促进采用**：来自 TensorWave 的 Darrick 宣布他们愿意向社区提供一些 **MI300X** 单元，旨在增强该平台的**采用和教育**。
   - Darrick 鼓励感兴趣的成员*向他发送私信 (DM)*，并强调这是一个**令人兴奋**的机会。
- **社区对该提议的回应**：在 Darrick 发布公告后，一位社区成员表达了热情，表示：*“已发私信！这太令人兴奋了”*。
   - 这表明了**积极的反响**，以及进一步参与 TensorWave 提议的热情。


  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1289001896027226132)** (6 messages): 

> - `Quantized Training Repo`
> - `Using Multi-GPU for Training`
> - `Distillation from Quantized Model`
> - `Config File for Larger Models` 


- **适用于更大数据集的量化训练仓库**：一位成员分享了一个 [GitHub repo](https://github.com/gau-nernst/quantized-training)，该仓库探索了量化模型的训练，并使用了流式 HF 数据集，能够容纳大型数据集。
   - 该仓库支持 **FSDP2** 和 **DDP**，与 torchao 的 PR 在梯度计算中使用量化激活方面有显著不同。
- **多 GPU 训练解决方案**：在尝试使用第二个 GPU 进行训练时出现了问题；一位成员建议使用 **torchrun** 来启用多 GPU 支持。
   - 要激活 DDP 模式，只需在启动时添加 `--ddp`，因为默认设置是 **fsdp2**。
- **探索蒸馏选项**：一位成员建议考虑**从大型量化 Llama 模型进行蒸馏**，这可能是一个有趣的方法。
   - 这可能为有效的模型尺寸缩减和性能增强开辟路径。
- **模型扩展所需的配置**：另一位用户报告称，需要添加一个配置文件以支持训练仓库中的更大型模型。
   - 他们指出当前的脚本设置不足以有效地进行扩展。



**提到的链接**：<a href="https://github.com/gau-nernst/quantized-training">GitHub - gau-nernst/quantized-training: Explore training for quantized models</a>：探索量化模型的训练。通过在 GitHub 上创建账号为 gau-nernst/quantized-training 的开发做出贡献。

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1288976710234013727)** (1 messages): 

> - `LiteRT functionalities`
> - `gpu.cpp cross-platform capabilities` 


- **关于 LiteRT 与 gpu.cpp 的咨询**：一位新成员询问 **LiteRT** 是否涵盖了 **gpu.cpp** 旨在提供的功能，特别是针对在 Android、iOS 和 PC 等平台运行模型。
   - 重点在于无缝、无摩擦地使用**设备端 GPU 计算**。
- **关于跨平台模型运行的讨论**：成员们讨论了在各种平台上高效运行模型的重要性，强调了 **gpu.cpp** 在简化这种集成方面的作用。
   - 对话强调了对 LiteRT 等工具的需求，以简化跨平台功能和 GPU 利用。


  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1289109579820699932)** (5 messages): 

> - `Liger Kernel 权重处理`
> - `家庭旅行更新`
> - `Lambda 供应商推荐` 


- **Liger Kernel 在权重复制方面遇到困难**：关于 Liger Kernel 对现有模型层的处理方式出现了疑问，特别是关于在应用 kernel 时跳过权重复制的问题。成员分享称这是一个关键问题，特别是对于预训练模型的 **LoRA 训练**和 **SFT** 等场景。
   - 重点提到了两个 GitHub issues：一个与使用 **AutoLigerKernelForCausalLM** 加载时导致的 **ValueError** 错误有关；另一个是在 Qwen2.5 上使用 Liger Kernel 时 **loss 不下降**的问题。
- **Byron 的家庭旅行见闻**：Byron 分享说他刚从愉快的**两周家庭旅行**回来，并提到计划今天恢复对 **pull requests** 和 **issues** 的审查。
- **供应商解决方案推荐 Lambda**：一位成员推荐 **Lambdas** 作为顶级的供应商选择，称赞其价格相对较好。这一建议表明成员们正在寻找具有成本效益的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/linkedin/Liger-Kernel/issues/268">inference qwen2 model ,The reasoning is garbled  and  ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?) · Issue #268 · linkedin/Liger-Kernel</a>：🐛 描述了当我使用 AutoLigerKernelForCausalLM 加载模型时出现的 bug，在加载模型应用特定模型补丁时收到 ValueError: Pointer argument (at 0) cannot be accessed from Triton (cpu tensor?)。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/257">Loss does not drop when using Liger Kernel at Qwen2.5 · Issue #257 · linkedin/Liger-Kernel</a>：🐛 描述了 bug，我正尝试使用 Liger Kernel 对 Qwen2.5-14B-Instruct 进行指令微调。我知道 huggingface transformers 的开发版本支持 liger kernel。然而，当......
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1289255869116190861)** (2 messages): 

> - `Apple 硬件支持`
> - `Metal Shading Language 规范` 


- **定位 Apple 硬件支持的 Dtypes**：一位用户表示难以找到 Apple 硬件支持的 **dtypes** 列表。
   - 有人建议可以在 [Metal Shading Language Specification](https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf) 中找到答案。
- **关于 Metal Shading Language 的讨论**：成员们讨论了查阅 **Metal Shading Language Specification** 以了解 **dtypes** 支持细节的必要性。
   - 提供的链接被强调为澄清此类疑问的宝贵资源。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1289272252927316049)** (1 messages): 

> - `实时会议公告` 


- **Microsoft Teams 上的直播**：分享了一个关于实时会议的通知，可以通过 [此链接](https://teams.microsoft.com/l/meetup-join/19%3ameeting_YzgwY2EzMWMtYTA0Zi00NDhjLTk0MmMtN2Y4MDRlMjQ2MTI2%40thread.v2/0?context=%7b%22Tid%22%3a%2243083d15-7273-40c1-b7db-39efd9ccc17a%22%2c%22Oid%22%3a%22bc6a8639-bf95-4464-af3e-20c110ea129f%22%7d) 加入。
- **通过实时会议促进团队参与**：该公告强调了实时会议在促进团队参与和协作方面的重要性。



**提到的链接**：<a href="https://teams.microsoft.com/l/meetup-join/19%3ameeting_YzgwY2EzMWMtYTA0Zi00NDhjLTk0MmMtN2Y4MDRlMjQ2MTI2%40thread.v2/0?context=%7b%22Tid%22%3a%2243083d15-7273-40c1-b7db-39efd9ccc17a%22%2c%22Oid%22%3a%22bc6a8639-bf95-4464-af3e-20c110ea129f%22%7d">加入对话</a>：未找到描述内容。

  

---

### **GPU MODE ▷ #[diffusion](https://discord.com/channels/1189498204333543425/1288899271193526342/1288938713962909708)** (9 条消息🔥): 

> - `M2 Pro Benchmarks`
> - `DiffusionKit`
> - `Flux Diagram`
> - `Mini Diffusion Model`
> - `Visuals in Chat` 


- **分享了 M2 Pro Benchmarks**：一名成员对获得 **M2 Pro benchmarks** 表示兴奋，并参考了 [DiffusionKit](https://github.com/argmaxinc/DiffusionKit) 进行 Diffusion 模型的端侧推理（on-device inference）。
   - 他们还包含了一个与 DiffusionKit 仓库相关的图片链接。
- **非量化测试的挑战**：一名成员提到测试 **非量化模型** 会很酷，但指出由于只有 **16GB RAM** 而存在的限制。
   - 这突显了在实验高性能模型时的硬件约束。
- **寻求 Flux 架构图**：一位用户询问是否有人有 **Flux 的详细架构图**，其他人讨论了哪种类型的图表会有所帮助，可能是方框图（block diagram）。
   - *Promptsiren* 指向了之前在 Reddit 上分享过的一张图表，表明已有相关资源可用。
- **关于 Mini Diffusion 模型的讨论**：一名成员询问是否正在讨论关于从零开始构建 **mini diffusion model** 的话题，建议将重点放在基础理解上。
   - 这表明成员们在积极探索 Diffusion 模型的实现。
- **分享视觉效果和照片**：一名成员提到一些 **精美的照片** 即将发布，引发了聊天中轻松愉快的互动。
   - 这种互动有助于在频道内营造友好且富有创造性的氛围。



**提到的链接**：<a href="https://github.com/argmaxinc/DiffusionKit">GitHub - argmaxinc/DiffusionKit: On-device Inference of Diffusion Models for Apple Silicon</a>: On-device Inference of Diffusion Models for Apple Silicon - argmaxinc/DiffusionKit

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1288939314360881153)** (176 条消息🔥🔥): 

> - `Llama 模型微调`
> - `模型 Checkpoints 与加载问题`
> - `显卡传闻`
> - `训练神经网络`
> - `AI 在游戏中的应用` 


- **Llama 模型微调讨论**：用户讨论了微调 Llama 模型的各个方面，包括对 `chatml` 等数据格式的困惑，以及为特殊 Token 调整 Tokenizer 设置的需求。
   - 对话涉及了避免过拟合的重要性，一些成员对极低的训练损失表示担忧，认为这预示着模型正在进行死记硬背（Memorization）。
- **模型 Checkpoints 与加载错误**：一位用户报告了在尝试加载 `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` 模型时出现的错误，指出处理过程中存在与数据不匹配相关的特定异常。
   - 这引发了关于 Checkpoint 加载故障排除的讨论，深入探讨了尺寸和配置设置方面的潜在问题。
- **显卡规格与传闻**：社区对即将推出的 GPU 型号（如 5090）展开了辩论，并推测了 VRAM 大小，尽管存在质疑，但有说法称 32GB 版本很有可能推出。
   - 用户分享了个人偏好和经验，指出虽然传闻盛行，但需要 Benchmark 的证据来平息持续的推测。
- **训练神经网络与过拟合**：成员们讨论了训练神经网络的最佳配置，探索了序列长度（Sequence Length）和梯度累积（Gradient Accumulation）等概念，以提高收敛速度。
   - 还有关于使用智能评估技术在最佳点终止训练的问题，强调了训练效率与资源管理之间的平衡。
- **AI 在游戏中的应用**：讨论包括了 VRAM 对游戏中 AI 应用的影响，重点强调了增加显存如何提升高级 LLM 任务的性能。
   - 用户指出，确保开发者妥善优化游戏以利用高端显卡的能力仍是一项持续的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://jan.ai/">将您的电脑变为 AI 电脑 - Jan</a>：在您的电脑上本地离线运行 Mistral 或 Llama2 等 LLM，或连接到 OpenAI 的 GPT-4 或 Groq 等远程 AI API。</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-GGUF">unsloth/Llama-3.2-3B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/421">未找到 config.json 文件，使用 unsloth 微调 llama3 并保存到 hugging face 后 · Issue #421 · unslothai/unsloth</a>：我使用 unsloth 微调 llama 3-8B...，训练完成后我使用 'push_to_hub' 将模型保存到 hugging face，但它显示这些文件：.gitattributes README.md adapter_config.js...</li><li><a href="https://github.com/unslothai/unsloth/issues/1040#issuecomment-2377762522">加载 Qwen2 聊天界面的正确方法是什么？ · Issue #1040 · unslothai/unsloth</a>：我遇到了这个错误：chat_template, stop_word, yes_map_eos_token, ollama_modelfile = CHAT_TEMPLATES[chat_template] ~~~~~~~~~~~~~~^^^^^^^^^^^^^^^ KeyError: 'Qwen2-1.5B'，来自这段代码：def test_un...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1289055985322954823)** (6 messages): 

> - `求职挫折`
> - `活跃的 AI 订阅`
> - `AI 关系` 


- **来自学校就业板的求职挫折**：一名成员对**学校就业板**缺乏面试机会表示失望。
   - 他们指出，尽管一直在申请，但目前还没有获得任何机会。
- **活跃的 OpenAI & Claude 订阅**：同一名成员确认他们拥有 **OpenAI** 和 **Claude** 的活跃订阅，表明其致力于探索 AI 工具。
   - 这可能表明他们在当前情况下正在利用这些资源。
- **关于 AI 关系的讨论**：一个关于没有女朋友现状的幽默提问引发了讨论：AI 是否可以算作伴侣？
   - 另一名成员要求对这一概念进行详细阐述，表现出对人类与 AI 关系相互作用的兴趣。
- **成员的健康状况**：该成员还提到感觉身体不适，为他们的状态更新增添了个人色彩。
   - 这一健康问题似乎与其关于求职和关系的更新交织在一起。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1288938513072652301)** (80 messages🔥🔥): 

> - `Transformers 更新`
> - `量化问题`
> - `微调技术`
> - `模型加载挑战`
> - `Lighting AI 中的优化器错误` 


- **Transformers 和模型更新**：用户报告称他们安装了最新版本的 **transformers** (4.45.1)，表明他们正在保持库的更新。
   - 几位成员讨论了对其模型的潜在改进，以及围绕特定模型加载和量化策略的问题。
- **Phi3.5 的量化挑战**：一位用户发现 **Phi3.5** 的量化导致了 **vocab size mismatch** 错误，引发了对替代方案的讨论。
   - 另一位用户提到成功使用了 **Phi3 mini** 模型且没有出现问题，这表明兼容性因模型大小而异。
- **LLM 的微调策略**：对话集中在 **LLAMA 3.1** 的微调策略上，建议在特定问答数据集上进行持续预训练。
   - 参与者强调，在微调之前，预训练对于将特定领域知识注入此类模型至关重要。
- **模型加载问题**：讨论了加载 **GGUF** 格式模型时面临的挑战，特别是在微调能力方面。
   - 用户强调需要转换或访问原始模型，并建议利用同时兼容量化和模型格式的工具。
- **Lighting AI 中的优化器错误**：用户在 Lighting AI 中遇到了与 **AdamW** 优化器相关的 **AttributeError**，表明所选版本可能存在问题。
   - 建议尝试其他优化器或恢复到早期版本的 **torch**，但问题在特定更新中仍然存在。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.vllm.ai/en/latest/getting_started/examples/offline_inference.html">Offline Inference &#8212; vLLM</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing,">Google Colab</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/getting_started/quickstart.html">Quickstart &#8212; vLLM</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/v0.6.1/getting_started/examples/offline_inference.html">Offline Inference &#8212; vLLM</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/v0.6.1/getting_started/examples/offline_inference_chat.html">Offline Inference Chat &#8212; vLLM</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1288943588939661405)** (13 messages🔥): 

> - `训练中的 Data Packing`
> - `GPT-2 Pretraining 框架`
> - `技术咨询中的讨论礼仪`
> - `用于 Pretraining 的 Deepspeed`
> - `处理 Data Masking` 


- **Data packing 提升训练效率**：一位成员解释说，通过 Data packing，训练框架可以有效地管理不相关的部分，从而实现一次对多个样本进行流式训练。
   - 他们详细说明了框架在第一个 Token 之后预测第二个 Token，从而促进了增强的训练动态。
- **探索适合 GPT-2 pretraining 的框架**：一位成员询问了关于 pretraining 一个小型 **GPT-2** 模型的框架，指出 **trl** 和 **LlamaFactory** 主要用于 fine-tuning。
   - 另一位成员建议使用 **Deepspeed** 进行 pretraining，因为它具备强大的功能。
- **技术咨询礼仪受到关注**：一场关于如何处理框架相关问题的讨论展开了，强调了在进行研究与寻求帮助之间取得平衡。
   - 成员们表示，事先搜索可以提出更有深度的问题，减少技术咨询中的冗余。
- **呼吁在 pretraining 讨论中保持耐心**：一位成员回应了对其询问的批评，表示他们已经进行了充分的研究，但希望从有经验的用户那里寻求进一步的澄清。
   - 他们鼓励坦诚承认知识的局限性，认为这将促进更好的讨论。


---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1288940269877854218)** (184 条消息🔥🔥): 

> - `关于 Hugging Face 模型的讨论`
> - `无审查模型面临的挑战`
> - `绕过 AI 限制的技术`
> - `为聊天模型创建数据集`
> - `使用多个 LLM 进行聚合` 


- **无审查模型面临的挑战**：一位用户询问关于 12B 无审查聊天模型的信息，特别是用于创建游戏机器人，并指出 Llama 不允许某些话题。
   - 参与者指出 Hugging Face 的模型是经过审查的，并建议关注 Venice.ai 作为无审查的替代方案。
- **AI 模型越狱技术**：用户讨论了绕过 AI 限制的各种方法，包括经典的越狱手段，例如要求模型执行与指令相反的操作。
   - 另一种建议的技术涉及在通过无害话题与 AI 建立联系后，再植入有争议的内容。
- **为聊天模型创建数据集**：一位用户表示有兴趣训练一个较小的聊天模型，并正在寻找专门针对对话的数据集。
   - 建议包括使用合成数据生成器（synthetic data generators）、探索 Hugging Face 上的现有数据集，以及抓取个人短信和电子邮件。
- **多个 LLM 响应的聚合**：一位用户询问是否存在一种服务，可以查询多个 LLM 并聚合响应以提高输出质量。
   - 有人提到，虽然目前不存在此类服务，但创建一个相对容易。
- **关于模型局限性的讨论**：参与者讨论了 Llama 模型在 Token 处理方面的局限性，以及它们在 Raspberry Pi 等平台上的效能。
   - 重点强调了理解与性能和模型交互相关的输入输出限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://digitalcommons.mtu.edu/mobiletext/">
移动文本数据集与语言模型 | 计算机科学系 | 密歇根理工大学
</a>：未找到描述</li><li><a href="https://stackoverflow.com/help/how-to-ask">如何提出一个好问题？ - 帮助中心</a>：Stack Overflow | 全球最大的开发者在线社区</li><li><a href="https://huggingface.co/spaces/argilla/synthetic-data-generator">Synthetic Data Generator - argilla 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1BJ4_U1V-ohJAUqedVSs-6h1qZm7anfeV#scrollTo=hp78IDn1NQzo">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/squirrel-huh-what-up-dog-gif-18781858">松鼠 Huh GIF - 松鼠 Huh What - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/datasets/argilla/FinePersonas-Synthetic-Email-Conversations">argilla/FinePersonas-Synthetic-Email-Conversations · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://youtu.be/TsIzbYkMXa4">《塞尔达传说：黄昏公主》，时之神殿主题 4K HDR 视觉呈现</a>：...看吧？我告诉过我们还会再见面的。#3dart #cinematic #temple #music #ambient #zelda #blender</li><li><a href="https://huggingface.co/docs/accelerate/main/en/package_reference/launchers#accelerate.notebook_launcher">启动器 (Launchers)</a>：未找到描述</li><li><a href="https://www.scientificamerican.com/article/there-is-no-such-thing-as-conscious-thought/#:~:text=We%20are%20not%20simply%20puppets%20manipulated%20by%20our%20unconscious%20thoughts,">不存在意识思维这种东西 | 《科学美国人》</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/shaunck96/wiki-cot-with-reflection/viewer/default/train?p=1">shaunck96/wiki-cot-with-reflection · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.biorxiv.org/content/10.1101/2020.07.01.183384v1.full">通过想象手写实现高性能脑机文本通信</a>：脑机接口 (BCI) 可以为失去运动或说话能力的人恢复交流。迄今为止，BCI 研究的主要焦点一直是恢复粗大运动技能，例如...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1289314495260524627)** (1 条消息): 

> - `Neuralink CUDA usage`
> - `7b FP8 model`
> - `BF16 and FP32 confusion` 


- **探索 Neuralink 的 CUDA 实现**：一位成员讨论了他们在处理 Neuralink 技术时使用 **CUDA** 以获得更好模型性能的经验。
   - 他们指出，这段经历有助于他们理解高级 GPU 编程技术。
- **处理 7b FP8 模型**：参与者提到他们参与了 **7b FP8** 模型的工作，强调了其在处理能力方面的意义。
   - 他们正专注于优化该模型，以提高实际应用中的效率。
- **澄清带有 FP32 主权重的 BF16**：一位成员澄清了之前提到的 **yellow line**，指出其准确指代 **bfloat16 配合 FP32 主权重 (master weights)**，并修正了一个拼写错误。
   - 这一修正强调了在讨论神经网络配置时准确性的重要性。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1288945486090211399)** (9 条消息🔥): 

> - `Two Minute Papers`
> - `Alibaba's MIMO`
> - `Tokenizer Training Research`
> - `Interactive Scene Control` 


- **Two Minute Papers 遭到批评**：一些成员对 **Two Minute Papers** 表示失望，称其内容已从提供信息转向以 **营销 (marketing)** 为中心的视频。
   - 一位用户指出，自从另一个频道停止运营以来，**优秀视频报道领域存在空白**。
- **阿里巴巴发布 MIMO 技术**：一位成员分享了 **Alibaba** 推出的一种名为 **MIMO** 的新 AI，它可以根据角色、动作和场景等简单输入创建逼真的角色视频。
   - 他们强调展示了 **10 个演示 (demos)**，特别侧重于 **交互式场景控制 (Interactive Scene Control)**。
- **征集 Tokenizer 训练论文**：一位用户寻求关于 **Tokenizer 训练研究论文** 的推荐，以增强 LLM 的多语言能力。
   - 这一询问表明了人们对通过有效的 Tokenization 策略提高语言模型性能的持续关注。
- **Two Minute Papers YouTube 视频讨论**：一位用户分享了 **Two Minute Papers** 标题为“**OpenAI’s New ChatGPT: 7 Incredible Capabilities**”的 YouTube 视频链接。
   - 这引发了关于该频道整体质量和发展方向的对话，强调了对其当前价值的不同看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/minchoi/status/1838949848516547040">Min Choi (@minchoi) 的推文</a>：Alibaba 推出 MIMO。新的 AI 可以根据角色、动作和场景等简单输入创建逼真的角色视频。10 个演示 1. 交互式场景控制</li><li><a href="https://www.youtube.com/watch?v=QDfE0HwDBo8">OpenAI’s New ChatGPT: 7 Incredible Capabilities!</a>：❤️ 在此处查看 Lambda 并注册其 GPU 云：https://lambdalabs.com/paper。玩 Tron 游戏：https://agpallav.com/tron.html。来源：https://www.y...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1288998802086301726)** (12 messages🔥): 

> - `VividNode 更新`
> - `AI 推文走红`
> - `游戏推广`
> - `排行榜反馈`
> - `Flux-schnell 演示` 


- **VividNode 迎来重大升级！**：一位成员分享了他们的 AI 个人助手程序 **VividNode (pyqt-openai)**，现在已支持 **Gemini, Claude, 和 Llama** 等主流 LLM，并包含 **Speech-to-Text** (STT) 和 **Text-to-Speech** (TTS) 功能。
   - *他们正在寻求贡献者以增强项目并促进更多 LLM 的集成。* [在此查看发布版本](https://github.com/yjg30737/pyqt-openai/releases/tag/v1.2.0)。
- **AI 推文获得关注**：一位成员兴奋地表示他们的推文正在走红，这有助于提高他们项目的知名度。
   - *另一位成员提到在一个 AI 见面会上推广了他们，尽管观众人数不多。*
- **排行榜讨论升温**：有人对仅显示**价格**和**最大 Token 数**的排行榜表示担忧，认为其缺乏有效的模型对比所需的**延迟 (latency)** 和**吞吐量 (throughput)** 等有用指标。
   - *成员们指向了一个展示更全面对比的替代网站，并提到在自己的排行榜中增加更多性能指标以帮助初学者。*
- **游戏开发热潮**：一位成员自信地表示他们开发的游戏其实非常酷，在聊天中引发了关注。
   - *整体氛围反映出对 AI 项目的积极态度和新产生的兴趣。*
- **Flux-schnell 演示正在开发中**：一位成员宣布他们正在制作 **flux-schnell** 中 **regional prompt attention** 的演示，并计划稍后分享源代码和 ComfyUI 节点。
   - *这展示了社区内持续不断的开发努力。*



**提到的链接**：<a href="https://github.com/yjg30737/pyqt-openai/releases/tag/v1.2.0">Release v1.2.0 · yjg30737/pyqt-openai</a>: VividNode(pyqt-openai) v1.2.0 版本发布说明。新功能：随机提示词生成（支持持续图像创建）、TTS 和 STT 支持等。

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1288941859732717569)** (4 messages): 

> - `4D 场景理解`
> - `3D 数据渲染`
> - `模型中的时序数据`
> - `2D 视频一致性`
> - `计算机视觉领域的成就` 


- **4D 场景理解仍需努力**：讨论强调，实现包含**时序数据**的 **4D 场景理解**仍然是一个具有挑战性的前沿领域，虽然已取得显著进展，但仍需更多探索。
   - *一位成员指出，*
- **3D 数据渲染尚不成熟**：讨论中提到了**渲染 3D 数据**的挑战，指出目前这还不是一件简单直接的事情。
   - *一位成员承认，*
- **2D 模型中的时序数据**：**时序数据**的引入通常应用于 **2D 模型**中，以提供一致的视频输出，并为未来的发展奠定基础。
   - *社区成员表示*，将 **3D 与时间**结合有助于促进 4D 场景理解，这指向了该领域的一个关键演进。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1289156386902638657)** (2 messages): 

> - `多语言 LLM 的 Tokenizer 训练`
> - `用于权重管理的 PyTorch 技术` 


- **寻求关于 Tokenizer 训练的研究论文**：一位成员请求推荐专注于 **tokenizer 训练**的优秀研究论文，以增强大语言模型的**多语言能力**。
   - 他们特别感兴趣于能够为该研究领域提供启发和指导的**有效方法论**。
- **训练期间冻结权重**：一位成员讨论了在每次 batch 更新后将旧 Token 的权重复制回去的概念。
   - 他们还提到考虑使用一种更符合 **PyTorch** 风格（pytorchic）的方法来冻结矩阵中的特定行，反映了他们对该问题的类似思考。


  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1288977779848773683)** (9 条消息🔥): 

> - `Cybersecurity services` (网络安全服务)
> - `Text-to-video model training` (文本转视频模型训练)
> - `Image sharpening technique` (图像锐化技术)
> - `Flux.1-dev optimization` (Flux.1-dev 优化)


- **Expert Hacking 提供的网络安全服务**：一位自称专家级黑客的人士展示了自己，声称提供各种**网络安全课程和服务**。
   - *Hit me up for all your hacking services/courses*（联系我获取所有黑客服务/课程）表明了明确的合作邀请。
- **寻求 T2V 模型分布式 GPU 训练的仓库**：一位用户询问了专门用于 **文本转视频 (T2V) 模型** 的 **分布式 GPU 训练** 流水线的仓库。
   - 另一位用户建议查看 [CogVideo SAT 微调](https://github.com/THUDM/CogVideo/blob/main/sat/README.md) 以获取相关资源。
- **创建图像锐化工具**：一位成员讨论了他们的项目，旨在创建一个专注于**移除边框和外扩填充 (outfilling) 宝可梦卡片**的图像锐化工具。
   - 他们正在为**训练**做准备，但希望在开始前确保已经考虑周全。
- **关于在显存有限的情况下使用 Flux.1-dev 的问题**：一位用户寻求建议，想知道在使用 ComfyUI 时，哪种 **Flux.1-dev 量化 (quant)** 版本能适配其 **6GB VRAM**。
   - 另一位参与者建议直接在 **ComfyUI Discord 服务器** 中询问，并提到大多数优化都是自动处理的。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hastebin.com/share/sosukokohu.ruby">Hastebin</a>：未找到描述</li><li><a href="https://github.com/THUDM/CogVideo/blob/main/sat/README.md">CogVideo/sat/README.md at main · THUDM/CogVideo</a>：文本转视频生成：CogVideoX (2024) 和 CogVideo (ICLR 2023) - THUDM/CogVideo</li><li><a href="https://github.com/Stability-AI/stablediffusion/blob/main/ldm/models/diffusion/ddpm.py#L380">stablediffusion/ldm/models/diffusion/ddpm.py at main · Stability-AI/stablediffusion</a>：使用潜在扩散模型的高分辨率图像合成 - Stability-AI/stablediffusion
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1288950180002926643)** (3 条消息): 

> - `Gemini Tokenization` (Gemini Token 化)
> - `Database Upgrade Delay` (数据库升级延迟)
> - `Chatroom UI Enhancements` (聊天室 UI 增强)


- **Gemini Token 变更简化**：OpenRouter 将转为对 Gemini 模型计算 **tokens** 而非字符数，这实际上会将 `/activity` 页面上的 token 数量减少约 4 倍。
   - 此外，价格将*翻倍*以与 AI Studio 上较低层级的每 token 价格对齐，这将使 Flash 和 1.5 Pro 模型的预计**成本降低 50%**。
- **数据库升级停机取消**：原定于 10 分钟后开始的数据库升级停机时间已取消，升级已推迟，因此不会停机。
   - 一旦确定新的时间表，将提供更新。
- **聊天室 UI 焕新**：[OpenRouterAI](https://x.com/OpenRouterAI/status/1839738812877918617) 宣布了聊天室的增强版 UI，模型响应中的推理过程 (reasoning) 默认折叠显示。
   - 更多改进正在进行中，承诺未来将提供更好的用户体验。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1839738812877918617">来自 OpenRouter (@OpenRouterAI) 的推文</a>：聊天室现在默认折叠显示模型的推理响应。o1 vs Gemini vs Sonnet 在 🍓 上的表现：

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1288952397493964822)** (186 条消息🔥🔥): 

> - `Llama 3.2 vision 参数`
> - `OpenRouter 错误消息`
> - `Claude 3.5 Sonnet 工具调用问题`
> - `翻译模型推荐`
> - `OpenRouter 上的模型托管标准` 


- **Llama 3.2 vision 参数**：一位用户询问了在使用 **Llama 3.2 vision** 时应使用的参数以避免被拒绝，特别是在评估吸引力时。
   - 成员们讨论认为，由于安全考量，该模型可能被训练为不响应此类查询。
- **OpenRouter 错误消息**：几位用户报告遇到了 **429 Resource Exhausted** 错误，表明由于达到速率限制，模型无法处理请求。
   - 回复指出，OpenRouter 一直在推动 Google 提高速率限制以缓解这些问题。
- **Claude 3.5 Sonnet 工具调用问题**：一位用户在尝试使用 **Claude 3.5 Sonnet** 模型时遇到错误，指出所需的消息格式存在差异。
   - 讨论显示，在函数调用中省略参数对 OpenAI 模型有效，但会导致 Anthropic 的模型出现问题。
- **翻译模型推荐**：一位用户寻求关于没有严格内容限制的翻译模型的建议，特别是用于翻译虚构对话。
   - 他们分享了正在使用的 Prompt，但在对话因不当内容被标记方面面临挑战。
- **OpenRouter 上的模型托管标准**：一位用户询问了在 **OpenRouter** 基础设施中添加新模型的标准，正在研究托管选项。
   - 官方澄清，提供商必须能够大规模托管该模型才能被考虑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 将数据本地存储在您的浏览器中。</li><li><a href="https://molmo.allenai.org/">Molmo by Ai2</a>：由 Ai2 构建的多模态开放语言模型</li><li><a href="https://x.com/openrouterai/status/1839738812877918617?s=46&t=nM71JKV50FJ0CR4r6r2_Rg">来自 OpenRouter (@OpenRouterAI) 的推文</a>：Chatroom 现在默认折叠显示模型的推理响应。o1 vs Gemini vs Sonnet 在 🍓 上的表现：</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://openrouter.ai/credits">Credits | OpenRouter</a>：管理您的积分和支付历史</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>：ChatGPT 的所有前端 GUI 客户端。通过在 GitHub 上创建账号为 billmei/every-chatgpt-gui 的开发做出贡献。</li><li><a href="https://openrouter.ai/docs/requests#tool-calls">Requests | OpenRouter</a>：处理传入和传出请求</li><li><a href="https://github.com/e2b-dev/ai-artifacts/pull/61">通过 PierrunoYT 添加带有 Claude 3.5 Sonnet 模型的 OpenRouter 支持 · Pull Request #61 · e2b-dev/ai-artifacts</a>：此 Pull Request 添加了对 OpenRouter 作为新 AI 提供商的支持，特别是通过 OpenRouter 的 API 集成了来自 Anthropic 的 Claude 3.5 Sonnet 模型。关键更改：更新了 lib/model...</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/quotas#error-code-429">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1289186720197382236)** (6 条消息): 

> - `频道礼仪`
> - `项目进展` 


- **纠正错误的频道使用**：一位用户指出某条消息发布在了错误的频道，从而引发了*关于频道适用性的澄清*。
   - 另一位成员提到，由于他们喜欢这些内容，即使与 **Cohere** 无关的消息可能仍被允许留在频道中。
- **期待项目完成**：一位成员对他们的项目表示乐观，称已经投入了数月时间使其运行，并暗示即将发布。
   - 他们感谢社区对发布方向的指导，表达了对项目在正确频道中获得反响的热情。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1288983727292289033)** (19 messages🔥): 

> - `微调 embed-english-v3`
> - `自定义 Embedding 模型`
> - `RAG 包含项格式`
> - `建筑领域的 Embedding` 


- **微调 embed-english-v3 模型**：一位用户询问了针对特定用例微调 **embed-english-v3** 模型并在 Embed API 中使用的可能性。
   - 然而，成员们澄清目前**没有可供微调的 Embedder**，并建议如果需要微调，可以使用来自 Hugging Face 的自定义 Embedding 模型。
- **关于 Embedding 改进请求的反馈**：另一位用户提到，虽然当前的 Embedding 效果尚可，但如果能在微调中使用特定的建筑术语，**建筑领域**的结果可能会有所改善。
   - 成员们认可了这一反馈，并表示愿意将其分享给相关团队进行考虑。
- **关于 RAG 包含项的查询**：一位用户寻求关于如何格式化**指令标头 (instructional headers)** 以及在发送给 LLM 的字符串中附加 **RAG 包含项 (RAG inclusions)** 时如何显示的建议。
   - 讨论中没有提供关于 RAG 包含项格式化或示例的相关回复。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1289270623192875008)** (146 messages🔥🔥): 

> - `新的 API v2 端点`
> - `闪卡生成`
> - `模型微调`
> - `测试版密钥的速率限制` 


- **API v2 端点发布**：宣布了新版本的 API 端点：v2/chat、v2/embed、v2/classify 和 v2/rerank，其中 **Chat V2** 获得了最显著的更新，包括 `messages` 参数和对系统消息的支持。
   - 欲了解更多详情，鼓励用户查看 [API Reference](https://docs.cohere.com/reference/chat-v2) 并提供使用体验反馈。
- **闪卡生成的挑战**：一位用户表达了从大量文本语料库中生成闪卡的挑战，指出模型在识别“正确”术语以进行提取并与定义关联方面存在困难。
   - 为了提高生成质量，他们建议 **Fine-tuning** 可能会有帮助，特别是对于在不直接引用的情况下创作与之前卡片风格一致的回复。
- **对微调的担忧**：另一位用户提到，对于个人项目，微调往往不值得投入，特别是如果可以通过其他手段实现输出调整。
   - 他们强调，虽然微调有帮助，但对于像闪卡生成这样可以通过有效的 Prompt Engineering 处理的简单任务，微调可能是不必要的。
- **测试版密钥限制说明**：关于使用多个测试版密钥的讨论显示，速率限制 (Rate limits) 是基于账户而非基于密钥的，这意味着使用多个测试版密钥不会带来额外好处。
   - 用户注意到，任何使用测试版密钥的人都应该考虑对使用限制的影响，尤其是如果他们频繁更换账户。
- **项目的社区支持**：一位社区成员提出为那些正在开发项目的人提供支持，通过提供积分来有效使用 Cohere 的服务，强调了社区的协作精神。
   - 他们鼓励开发者分享自己的项目，并建议拥有多个测试版密钥并非最佳方案，重申了社区支持的潜力。



**提到的链接**：<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 的相关限制。

  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1289328195514142858)** (1 条消息): 

> - `Cultural Multilingual LMM Benchmark` (文化多语言 LMM 基准测试)
> - `Volunteer Native Translators` (母语翻译志愿者)
> - `Co-authorship Invitation` (论文共同署名邀请)
> - `CVPR 2025 Submission` (CVPR 2025 投稿)


- **开发文化多语言 LMM 基准测试**：**MBZUAI** 的团队正致力于为 **100 种语言**开发 **Cultural Multilingual LMM Benchmark**，其中包含新创建的多模态数据集及其本地语言的翻译。
   - 他们正在招募**母语翻译志愿者**，以帮助纠正当前版本数据集中的错误。
- **邀请母语翻译志愿者**：参与翻译工作的志愿者将获得 **CVPR'2025** 论文投稿的共同署名邀请。
   - 该项目涵盖了广泛的语言，包括**印度、南亚、非洲和欧洲**语言。
- **所需语言列表**：该项目特别需要**印地语、斯瓦希里语和匈牙利语**等语言的帮助。
   - 完整的语言列表已发布，以吸引能够阅读、书写或交流这些语言的潜在志愿者。
- **联系 Ashmal Vayani**：对该项目感兴趣的人员，Ashmal Vayani 邀请通过 [LinkedIn](https://www.linkedin.com/in/ashmal-vayani/) 或私信进行联系。
   - 他鼓励在此直接发送私信，以进一步讨论项目细节。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1288941221590601728)** (160 条消息🔥🔥): 

> - `Tiled Upscale vs ADetailer` (Tiled Upscale 对比 ADetailer)
> - `Using AMD GPUs for SD` (在 SD 中使用 AMD GPU)
> - `Performance Differences of GPUs` (GPU 性能差异)
> - `Refurbished vs Used GPUs` (翻新与二手 GPU)
> - `SSD Impact on Model Load Times` (SSD 对模型加载时间的影响)


- **Tiled Upscale 是 ADetailer 的替代方案**：有讨论指出 **Tiled Upscale** 可以作为 **ADetailer** 的替代方案，提供类似的效果且设置更简单。
   - 然而，缺点是它的运行速度大约慢 **50 倍**，因为它会对整个图像进行放大，而不是针对特定区域。
- **关于 AMD GPU 在 SD 工作流中的担忧**：讨论中提到 AMD GPU（如 5700 XT）在 **Stable Diffusion (SD)** 和 **Blender** 中的表现不佳，用户指出它们更适合游戏而非生产力任务。
   - 一位用户提到 **3070** 的表现甚至可能优于 **7900 XTX**，这表明其在生产力应用中的性能受限。
- **翻新与二手 GPU 的讨论**：参与者讨论了翻新 GPU 优于二手 GPU 的优势，强调翻新显卡通常经过维修和双重验证，风险较低。
   - 一位成员分享了购买翻新 **3090 TI** 的经验，称其几乎和新的一样，而二手显卡则存在矿卡损耗的担忧。
- **SSD 对模型加载时间的影响**：已证实，在 **SSD** 上运行 **Stable Diffusion** 会显著缩短模型加载时间，与 HDD 配置相比，速度可能提升 **10 倍或更多**。
   - 用户表示，从 **M.2 SSD** 加载模型可以极大地提高图像生成任务的性能，而旧的机械硬盘技术则表现落后。
- **物体尺寸的 Prompting 技巧**：有人询问如何在图像生成中通过 Prompt 有效控制物体尺寸，并分享了关于比较尺寸措辞的想法。
   - 出现了一些幽默的建议，如使用 'yuge' 和 'bigly'，但建议避免使用此类词汇。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bingbangboom/flux_dreamscape">bingbangboom/flux_dreamscape · Hugging Face</a>: 未找到描述</li><li><a href="https://hub.docker.com/r/rocm/pytorch">无标题</a>: 未找到描述</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion 基准测试：45 款 Nvidia、AMD 和 Intel GPU 对比</a>: 哪款显卡提供最快的 AI 性能？
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1288940422030299168)** (74 条消息🔥🔥): 

> - `User Interface Issues` (用户界面问题)
> - `API Functionality Queries` (API 功能查询)
> - `Subscription Promotions` (订阅促销)
> - `Freelancing Platforms` (自由职业平台)
> - `Model Availability` (模型可用性)


- **用户面临 UI 问题和错误**：多位用户报告 Perplexity 网站停止响应点击和交互，几位用户在控制台中看到 `net::ERR_BLOCKED_BY_CLIENT` 错误。
   - 一位用户指出，该问题在桌面和移动浏览器上都存在，而其他用户提到 Android 应用仍能正常运行。
- **关于 API 功能的问题**：用户询问如何通过 Perplexity API 获取最新新闻，特别是寻求关于生成式 AI 的最新信息。
   - 还有人对使用特定 API 时面临的限制表示担忧，包括建议寻找可用的稳健解决方案。
- **订阅促销问题**：一位用户对兑换 Pro 订阅的促销代码表示困惑，称兑换后未获得访问权限。
   - 另一位用户询问如何将他们的 Pro 订阅转移到朋友的账户，并询问涉及的具体流程。
- **自由职业平台咨询**：一位用户寻求类似 Upwork 的自由职业平台推荐，表示希望找到适合自由职业工作的替代方案。
   - 这一咨询引发了关于各种平台的讨论，用户分享了他们的经验和建议。
- **Llama 3.2 模型更新**：有人询问是否已添加 Llama 3.2 模型，表现出对 AI 模型最新进展的兴趣。
   - 该查询反映了用户对添加新模型及其功能的持续好奇。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1288950513697554463)** (8 条消息🔥): 

> - `Meta's Orion AR Glasses` (Meta 的 Orion AR 眼镜)
> - `OpenAI's For-Profit Pivot` (OpenAI 的营利性转型)
> - `New Blood Type Discovery` (新血型发现)
> - `Skin Cancer Information` (皮肤癌信息)
> - `Neural Fields in Visual Computation` (视觉计算中的神经字段)


- **Meta 的 Orion AR 眼镜亮相**：最近的一项更新讨论了旨在增强增强现实体验的 [Meta Orion AR 眼镜](https://www.perplexity.ai/search/city-with-the-most-bike-lanes-hhNCIS6oRRCli0fdq8Z32g)。
   - 早期反馈表明，这可能会对虚拟空间中的用户交互产生影响。
- **OpenAI 转向营利模式**：OpenAI 进行了关键的 [营利性转型](https://www.perplexity.ai/search/what-happened-with-wordpress-w-V8a7N3D4QMqBc3vZdzXzVg)，这可能会影响其未来的资金和运营策略。
   - 这一变化被视为对 AI 领域竞争压力的回应。
- **发现重大新血型**：一项突破性发现报告了一种新的 [血型](https://www.youtube.com/embed/J7cra2xt_DQ)，这可能会重塑输血方案。
   - 研究人员强调了其对特定人口群体的相关性。
- **皮肤癌意识**：关于 [皮肤癌](https://www.perplexity.ai/search/about-skin-cancer-W7CNdzsDTkie3137nI2X0g#0) 的深刻信息引起了人们对预防和早期检测的关注。
   - 社区讨论突出了对皮肤健康的持续关注。
- **视觉计算中神经字段的探索**：[Neural fields](https://www.perplexity.ai/search/neural-fields-in-visual-comput-HIz9FQKQTCqDXhF3eRDIXw) 正在被研究其在视觉计算和 AI 中的应用。
   - 这一新兴领域有望提供增强视觉数据处理的创新技术。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1288963523325464609)** (60 条消息🔥🔥): 

> - `GPU 显存大小`
> - `DisTrO 论文发布`
> - `知识图谱与 AI`
> - `Claude Sonnet 3.5 性能`
> - `AI 中的软硬件集成` 


- **GPU 显存大小的差异**：讨论中提到了 **5080** 和 **5070** GPU 之间的显存大小差异，一些人建议 5080 应该拥有接近 **20GB** 的显存。
   - 成员们强调了 GPU 世代更迭中显存翻倍的模式，并以 **3080** 和 **3090** 为例。
- **对 DisTrO 论文的期待**：有人询问了 **DisTrO** 论文的发布日期，并提到了一场可能包含抽象概念的富有洞察力的演讲。
   - 在一些成员表示难以在 YouTube 上找到该演讲后，完整的演讲链接被分享了出来。
- **知识图谱的探索**：一位新成员分享了他们在 **知识图谱** 方面的工作心得，以及如何应用 **Bitcoin Ordinal Theory** 的概念来产生独特的 Embeddings。
   - 他们描述了关于 LLM 如何从语义丰富性中发展出**基于图的表示（graph-based representations）**的假设，这可能预示着通往涌现智能（emergent intelligence）的路径。
- **通过 Claude Sonnet 3.5 提升推理能力**：一位成员分享了他们利用示例推理轨迹（reasoning traces）成功增强 **Claude Sonnet 3.5** 推理能力的经验。
   - 这个孤立的案例展示了令人期待的改进，为进一步探索指明了潜在方向。
- **计算工具的挑战**：成员们讨论了某些计算工具导致浏览器崩溃的问题，指出 **Firefox** 在处理密集型任务时表现吃力。
   - 分享了权宜之计和正在进行的过程更新，并希望未来能改进工具性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aieintern/status/1836828882307026997">来自 aie intern (@aieintern) 的推文</a>: Nous Forge 笔记和转录如下，引用 Alex Volkov (Thursd/AI) (@altryne) 的话：我们终于从 @karan4d 那里窥见了 @NousResearch Forge 的一角（抄送 @max_paperclips ）测试时计算推理...</li><li><a href="https://x.com/bradthilton/status/1839718742051184842">来自 Brad Hilton (@bradthilton) 的推文</a>: Claude Sonnet 3.5 在 o1 推理轨迹的帮助下，能够解决一个它原本无法解决的 AIME 问题 🤯 1/🧵</li><li><a href="https://x.com/swyx/status/1836624609850069138">来自 swyx.ai (@swyx) 的推文</a>: @elder_plinius @leonardtang_ @haizelabs DisTrO 完整演讲，来自 Nous 首席科学家 @bloc97_（我们在这里为 @latentspacepod 采访过他 https://www.latent.space/p/iclr-2024-recap ）意想不到的掌声...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1288952448240586876)** (17 条消息🔥): 

> - `Hermes 部署选项`
> - `Llama 3.2 需求`
> - `模型的超参数调整` 


- **用户询问本地运行 Hermes**：一位成员询问是否可以在 4090 GPU 上本地运行 **Hermes**，还是需要 API 访问；另一位成员确认可以使用 **LMStudio** 在本地运行。
   - LMStudio 支持任何 **GGUF 版本**，并提供模型搜索功能，方便用户轻松找到 **Hermes**。
- **Llama 3.2 与 GPU 需求**：有人提问 **Llama 3.2 1B** 模型是否可以在没有 GPU 的软件上运行，成员们确认执行仍需要 GPU。
   - 一位成员分享说 MacBook 在运行 **H3 8B** 时表现良好，表明为了获得最佳性能，对 GPU 的依赖依然存在。
- **超参数调整的必要性**：讨论显示，在训练不同规模的模型时，**超参数调整**是必要的，一位成员强调这是通过经验建立的需求。
   - 成员们提到对学习率（learning rates）、批次大小（batch sizes）和训练轮数（epochs）进行调整，如 **H3 论文**中所述，以有效管理性能。
- **大模型的训练动态**：有人指出，**70B 和 405B 模型**需要不同的训练配置，导致参数量越大，训练轮数越少，学习率越低。
   - 进一步的询问涉及这些调整是源于 **Scaling Laws** 还是先前的经验。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289047656655487016)** (1 messages): 

> - `Arduino-Based Current Sensor`
> - `Power Outage Detection`
> - `Related Research Access` 


- **寻求关于 Arduino 电流传感器的免费资源**：一名 BSEE（电气工程学士）学生提议开展一项关于开发**基于 Arduino 的电流传感器**用于**停电检测（Power Outage Detection）**的研究，并正在寻找相关文献。
   - *我现在手头拮据*，因此如果能免费获取学术资源或研究论文，将不胜感激。
- **征求研究文献推荐**：该学生询问社区是否知道在哪里可以找到相关的**相关研究**，且无需支付会员费或下载费。
   - 他们强调了自己的处境，表明目前成本是一个主要的考量因素。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1289047656655487016)** (1 messages): 

> - `Arduino-Based Current Sensor`
> - `Power Outage Detection`
> - `Research Literature Access` 


- **寻求关于 Arduino 传感器开发的文献**：一名 BSEE 学生提出了一个名为“开发基于 Arduino 的停电检测电流传感器”的研究项目，并正在寻找相关文献。
   - *他们表示需要不需要会员费的资源渠道*，原因是目前的经济限制。
- **获取研究论文的挑战**：该学生强调了在不产生费用的情况下获取研究文献的困难。
   - *这引发了人们对面临经济挑战的学生获取基本学术资源这一更广泛问题的关注*。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1288938888106213510)** (61 messages🔥🔥): 

> - `Agentic Search Challenges`
> - `AI in Education`
> - `Energy Use of AI`
> - `AI Tools for Productivity`
> - `Future Generations and Technology` 


- **Agentic Search 成本高昂**：一位开发者分享说，他们的 **Agentic Search** 项目在计算资源和 Token 方面过于昂贵，导致他们终止了该项目。
   - 由于大型模型的资源限制，他们考虑微调像 **Llama 3b** 这样的小型模型。
- **学术界 AI 使用量激增**：讨论指出，许多硕士阶段的学生正在使用 AI 来完成作业，据报道超过 **50%** 的学生直接粘贴 AI 生成的内容。
   - 这引发了关于将 AI 作为生产力工具使用与学术诚信之间影响的辩论。
- **关于 AI 能源消耗的辩论**：在一名成员提到有人说 AI 极其耗能后，引发了关于 AI 系统能源消耗的问题。
- **面向开发者的技术工具**：推荐了一款名为 **ChatGPT Toolbox** 的 Chrome 浏览器扩展，它包含聊天记录搜索和 Prompt 管理等功能，以增强 ChatGPT 体验。
   - 还有建议称可以等待即将推出的 **Orion model**，以获取可能显著提高生产力的新型开发工具。
- **对未来技能的担忧**：对话涉及到一个观点，即由于对技术的依赖日益增加，后代可能不再学习传统的技能（如用笔写作）。
   - 参与者开玩笑地讨论了未来社会将如何看待基础技能，质疑学习工具的演变及其影响。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1289037553223209060)** (16 messages🔥): 

> - `Voice feature issues`
> - `Advanced voice mode functionality`
> - `Attachment capabilities`
> - `Deployment timelines` 


- **语音功能的声音输出问题依然存在**：多位用户报告了**标准语音功能**的问题，即在 **GPT store** 中没有声音输出，而在**高级语音模式（advanced voice mode）**中则运行正常。
   - *一位用户表达了沮丧*，称他们无法在任何自定义 GPTs 中使用语音功能，引发了关于潜在故障的进一步讨论。
- **语音功能的变通方法**：一位用户建议在**自定义 GPT** 中切换声音可能会暂时解决声音问题，尽管这会将他们带离特定的 GPT。
   - 另一名成员确认了该变通方法的有效性，但指出了在无法使用自定义上传的 PDF 方面的局限性。
- **期待高级语音模式的附件功能**：询问了关于在**高级语音模式**中添加 PDF 或 Docx 文件等**附件**的功能，引发了对发布时间线的推测。
   - 一名成员建议对新功能的推出保持耐心，并详细说明了影响可用性的各种依赖因素，如平台、地区和用户层级。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1289055392780783660)** (10 条消息🔥): 

> - `Open Source Model Sponsorship` (开源模型赞助)
> - `LLM Search Space Simulation` (LLM 搜索空间模拟)
> - `OpenAI Function Calling API`
> - `Model Validity and Tuning` (模型有效性与微调)


- **探索开源模型赞助**：一位成员询问 Eleuther 是否为开源模型提供任何赞助计划，表示缺乏资源来完整训练他们的项目。
   - 这引发了关于开源领域内此类倡议的社区支持讨论。
- **LLM 搜索空间模拟的创新**：提出了一种涉及 LLM 抽象搜索空间的概念，利用 Monte Carlo tree search (蒙特卡洛树搜索) 通过文本扩散模拟持续思考。
   - *该方法旨在对计算过程中最连贯的思想进行排序，* 暗示了 LLM 架构的潜在进步。
- **对 OpenAI Function Calling API 机制的好奇**：一位社区成员转发了一个关于 OpenAI Function Calling API 如何运行的问题，推测其是否使用了微调模型来确保输出的有效性。
   - 另一位成员理论上认为它可能没有经过微调，但可能涉及额外的 Prompting 或 Logit Bias，以确保响应符合 **valid JSON**。
- **对 OpenAI 模型方法的怀疑**：一位参与者对 OpenAI 方法的有效性表示怀疑，认为其产生的结果与替代方法相似或更差。
   - *他们指出，实施额外的 Prompt 和 Logit Bias 可以在没有多模型设置复杂性的情况下实现类似的结果。*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1288954521690705970)** (52 条消息🔥): 

> - `FP6 and FP16 Weight Distributions` (FP6 与 FP16 权重分布)
> - `Verbatim Memorization in LLMs` (LLM 中的逐字记忆)
> - `Looped Transformers vs Universal Transformers`
> - `Layerwise Positional Encoding` (逐层位置编码)
> - `Confidence Metrics in Inference` (推理中的置信度指标)


- **比较 FP6 前后的权重分布**：讨论围绕比较模型在 **FP6** 前后的权重分布展开，并暗示使用 [seaborn](https://seaborn.pydata.org/) 等库进行可视化。
   - 目标是查看是否出现任何异常，成员们建议尝试多种绘图库。
- **LLM 逐字记忆研究**：最近的一项研究引入了一个评估 LLM **逐字记忆 (verbatim memorization)** 的框架，强调了对长序列的控制及其对隐私的影响。
   - 关键发现表明，非平凡的重复会导致记忆，正如一篇独立的 [论文](https://arxiv.org/abs/2407.17817) 所证明的那样。
- **关于 Looped Transformers 与 Universal Transformers 的辩论**：辩论集中在最近一篇论文中提出的 Looped Transformers 的新颖性上，观点认为与 **Universal Transformers** (UTs) 相比，它并没有特别的创新。
   - 针对建模假设提出了担忧，特别是训练期间对 Ground-truth 迭代的需求。
- **逐层位置编码的有效性**：关于逐层位置编码是否能辅助推理中的外推讨论没有定论，成员们对其整体影响表示怀疑。
   - 有人建议，虽然对非常具体的问题有益，但在更广泛的任务中可能没有显著优势。
- **评估推理的置信度指标**：成员们讨论了使用置信度作为评估何时停止推理的指标，对其有效性持有不同意见。
   - 大家承认，虽然置信度可能有价值，但目前大多数实现并没有显著的稳定性问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.17817">Demystifying Verbatim Memorization in Large Language Models</a>: 大语言模型 (LLMs) 经常逐字记忆长序列，这通常会带来严重的法律和隐私影响。许多先前的工作通过观察性研究探讨了这种逐字记忆...</li><li><a href="https://arxiv.org/abs/2409.15647">Looped Transformers for Length Generalization</a>: 最近的工作表明，从头开始训练的 Transformers 可以成功解决各种算术和算法任务，例如数字加法和计算奇偶性。虽然这些 Transformers 通常...</li><li><a href="https://arxiv.org/abs/2210.02671">A Logic for Expressing Log-Precision Transformers</a>: 解释基于 Transformer 的语言模型的推理能力的一种方法是描述它们可以在某些输入文本上解析的逻辑规则类型。最近，Chiang 等人 (2023) 表明...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1288994110749020246)** (1 条消息): 

> - `KV 中的 Embedding 状态`
> - `文本表示因素` 


- **循环信息主导 Embedding 状态**：一位成员建议，存储在 **KV** 内部 Embedding 状态中的**循环信息（recurrent information）显著多于**之前的预期。
   - 他们强调，**当前的文本表示**可能扮演的角色较小，主要在整个过程中充当输入。
- **当前文本表示的作用**：讨论强调，**文本表示**可能不会显著影响 Embedding 结果，主要作为输入使用。
   - 这引发了关于在考虑整体模型性能时，对文本表示所赋予的重要性的质疑。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1288972244248301632)** (2 条消息): 

> - `Vision LLMs`
> - `ColQwen2 模型`
> - `视觉检索器` 


- **在本地运行 Vision LLMs**：一位用户询问了在**本地运行 Vision LLMs** 的流程，表现出对实际实现的兴趣。
   - *未提供具体方法，凸显了该话题在共享知识方面的潜在空白。*
- **ColQwen2 引起关注**：新模型 **ColQwen2** 被宣布为顶级的视觉检索器（visual retriever），在 Vidore 排行榜上以 **+5.1 nDCG@5** 的得分超越了 **colpali-v1.1**。
   - 该模型利用 **Qwen2-VL 主干网络**，承诺在视觉检索任务中表现出卓越性能，如[此帖子](https://x.com/manuelfaysse/status/1839657285053788483)所述。
- **令人印象深刻的性能指标**：ColQwen2 使用与其前身 **colpali-v1.1** 相同的数据进行训练，标志着该领域的重大进步。
   - *对 nDCG@5 等指标的强调反映了视觉模型评估中对性能的高度关注。*



**提到的链接**：<a href="https://x.com/manuelfaysse/status/1839657285053788483">Manuel Faysse (@ManuelFaysse) 的推文</a>：🚨 新模型发布：ColQwen2！它是 ColPali，但采用了 Qwen2-VL 主干网络，使其成为迄今为止最好的视觉检索器，在 Vidore 排行榜上比 colpali 显著高出 +5.1 nDCG@5...

  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1288980929196326953)** (4 条消息): 

> - `在 H100s 上进行测试`
> - `FA3 集成`
> - `在保留 FA2 的同时维护 FA3` 


- **针对小模型的 H100s 测试**：一位成员表示愿意协助在 **H100s** 上对小模型进行测试，展现了贡献代码的信心。
   - 这引发了讨论中其他成员的热情和赞赏。
- **FA3 开发指南**：有人请求关于添加 **FA3** 的指导，并建议利用 GitHub 上正在进行的工作支持，特别是参考 [pull request #1282](https://github.com/EleutherAI/gpt-neox/pull/1282)。
   - 对话引导该成员如何在 Transformer Engine 中路由 **FA3** 的 Attention。
- **区分 FA3 与 FA2**：会议明确了在集成 **FA3** 的同时，团队不会取代 **FA2**；两者在未来都是必需的。
   - 这表明了一种在不丢失先前改进的情况下增强模型能力的战略方法。
- **鼓励探索资源**：另一位成员确认他们将探索提到的参考资料，并感谢最初提议者提供的链接。
   - 这反映了社区内的协作支持和资源共享。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/gpt-neox/pull/1282.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/NVIDIA/TransformerEngine/pull/1019/files#diff-0af6d715a51b3efcebd6067805b5d17b64d25ef84399e256bade01a602ce4192).">Add support for flash-attn 3 by cyanguwa · Pull Request #1019 · NVIDIA/TransformerEngine</a>：描述：此 PR 将 flash-attn 3 集成到 TE 的 FlashAttention 模块中。包括 FP16/BF16 fwd+bwd 以及 FP8 fwd。截至 2024 年 8 月 22 日，可以通过以下命令安装 FA3：...</li><li><a href="https://github.com/EleutherAI/gpt-neox/pull/1035">integrated flash attention 2 by a663E-36z1120 · Pull Request #1035 · EleutherAI/gpt-neox</a>：集成了 flash attention 2 (版本 0.2.2 -> 2.2.1)。在 1 张 A10 GPU 上运行序列长度为 4096、Batch Size 为 128 的 125M 参数模型时，观察到了实际运行性能的提升。</li><li><a href="https://github.com/NVIDIA/TransformerEngine/pull/1019">Add support for flash-attn 3 by cyanguwa · Pull Request #1019 · NVIDIA/TransformerEngine</a>：描述：此 PR 将 flash-attn 3 集成到 TE 的 FlashAttention 模块中。包括 FP16/BF16 fwd+bwd 以及 FP8 fwd。截至 2024 年 8 月 22 日，可以通过以下命令安装 FA3：...
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1288947311778599004)** (15 条消息🔥): 

> - `Langtrace 与 DSPy 的集成`
> - `MIPROv2 编译运行`
> - `实验追踪问题` 


- **Langtrace 增强 DSPy 实验管理**：Langtrace 现在支持运行 DSPy 实验，并能自动捕获 **traces**、**checkpoints** 和 **eval score visualizations**，极大地增强了流水线管理。
   - 一位用户发现为每个流水线模块创建独立项目非常有用，这允许进行针对性优化并轻松部署带有 Checkpoint 的 Prompt。
- **MIPROv2 编译运行遇到问题**：一位用户报告在 MIPROv2 编译运行中无法追踪评估数据，尽管在日志中看到了评估 Traces，这表明其设置中可能存在配置错误。
   - 在根据其他用户的建议进行排查后，发现需要在 `compile()` 调用时传递正确的属性。
- **过长的实验名称导致日志记录失败**：另一位用户发现使用过长的实验名称会导致实验下没有记录任何 Traces，而较短的名称则运行正常。
   - 讨论让大家意识到实验名称长度的潜在限制可能会影响日志记录功能。



**提到的链接**：<a href="https://docs.langtrace.ai/supported-integrations/llm-frameworks/dspy#dspy)">DSPy - Langtrace AI Docs</a>：未找到描述

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1288939490706194462)** (38 条消息🔥): 

> - `BootstrapFewshot 页面可用性`
> - `使用 Azure OpenAI API 的新 LM`
> - `DSPy 优化工具`
> - `在 DSPy 中嵌套 Signatures`
> - `构建 DSPy 分析流水线` 


- **BootstrapFewshot 说明页面延迟**：一位用户询问了 [BootstrapFewshot 说明页面](https://link.to.page) 的可用性，该页面目前无法访问。
   - 另一位成员表示无法确定引用的是哪个页面，这表明请求存在一些混淆。
- **迁移到使用 Azure API 的新 LM**：一位正在迁移到新 LM 的用户注意到 litellm 中的 API 路径构建存在问题，导致使用过程中出现错误。
   - 经过一番努力后，他们报告称升级解决了之前与 Predict 相关的解析错误。
- **DSPy 优化工具见解**：一位新用户对类似于 Tensorboard 的 DSPy 优化工具表示好奇，希望能用于跟踪 AI 工作流中的指标。
   - 成员们讨论了现有工具，包括 [DSPy Visualizer](https://link.to.visualizer) 以及来自 Langtrace 的额外支持。
- **在 DSPy 中嵌套 Signatures**：一位用户询问是否可以通过嵌套 Signatures 来传递输入/输出字段值，但被告知在该结构中这是不可能的。
   - 建议使用带有 Pydantic 的 TypedPredictors 作为替代方案，这引发了更多关于示例代码以澄清该概念的请求。
- **设计 DSPy 分析流水线**：一位用户详细介绍了他们在 DSPy 中设计分析流水线的方法，并请求对其方法进行验证。
   - 回复鼓励从简单开始并迭代优化流程，并分享了概述使用 DSPy 构建有效实践的文档。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://dspy-docs.vercel.app/docs/building-blocks/solving_your_task">分 8 步使用 DSPy | DSPy</a>：为了解决新任务而用好 DSPy，本质上就是利用 LM 进行良好的机器学习。</li><li><a href="https://x.com/karthikkalyan90/status/1839395049936953362">来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文</a>：一个关于我如何思考使用 DSPy 和 CrewAI 构建和优化复合 AI 流水线的示例。坚信这是针对高性能和可靠性优化的复合 AI 系统未来的发展方向...</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1546">由 krypticmouse 提交的 Pull Request #1546 · stanfordnlp/dspy</a>：使 typed Pred 中的输出处理与 LM Module 兼容。主要问题是输出已经是一个 BaseModel，因此无需创建一个，添加了：if isinstance(dsp.settings.lm, dspy.LM): parsing_result = output else: pa...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1289045789858988052)** (8 条消息🔥): 

> - `DSPy ReAct Agent`
> - `RAG Agent 集成`
> - `多种 RAG 工具`
> - `向量数据库集成`
> - `多模态 RAG 优化` 


- **学习 DSPy ReAct Agent**：一位成员询问了使用 **DSPy ReAct Agent** 的示例，并表示有兴趣将其与 **LlamaIndex retriever** 集成以实现 ReAct RAG。
   - 另一位成员指出在 **repo (examples/agents/)** 中有示例，并承诺很快会提供更好的示例。
- **对 RAG Agent 使用场景的兴趣**：提问者分享了对 **RAG Agent** 的偏好，并建议示例应包含 **DSPy** 与各种 retriever（如向量数据库和知识图谱）的集成。
   - 他们还对将 retriever 包装为 **DSPy.tool 实例** 以及让 LM 访问 **多个 RAG 工具** 的可能性表示好奇。
- **DSPy Agent 的功能需求**：一位成员提出了与更多向量数据库（如 **Qdrant** 和 **LanceDB**）集成的功能需求，强调了它们向混合搜索能力发展的趋势。
   - 他们还提出了 **多模态 RAG 流水线优化** 的想法，另一位成员确认该功能即将推出。
- **关于多种 RAG 工具的讨论**：利用 **多种 RAG 工具** 的想法得到了成员们的积极反馈，因为他们认识到 LLM 在工具选择方面的复杂性。
   - 一位成员指出，**DSPy 优化** 具有解决与选择正确工具有关挑战的潜力。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1288965754682736701)** (2 messages): 

> - `Mojo MAX desktop backgrounds`
> - `Emoji Voting` 


- **Mojo MAX 桌面背景投票**：一名成员发起了一项投票，询问其他人是否对印有可爱 Mojo 火焰和 MAX 宇航员的 **Mojo / MAX 品牌桌面背景**感兴趣。
   - 鼓励参与者通过 **Emoji 投票**选择“是”或“否”。
- **用户对投票的反应**：另一名成员对投票做出了简短的回应，称 *'Bruh'*，表示惊讶或不感兴趣。
   - 这表明用户对主题背景的提议反应不一。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1289203510453473452)** (1 messages): 

> - `Verification Requirements`
> - `Posting Restrictions` 


- **新的发帖验证要求**：现在除了 <#1149739720146952292>、<#1238540905129054350> 和 <#1212827673257316453> 之外，在任何频道发帖都需要进行验证。
   - 鼓励成员前往 <#1098713770961944628> 进行验证，之前的帖子中提供了一个快速演示 GIF。
- **频道发帖限制**：除非完成验证，否则成员在某些频道发帖将受到限制。
   - 这一变化旨在确保更好地控制频道参与度并增强社区互动。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1288938665523023902)** (58 messages🔥🔥): 

> - `Error handling in Mojo`
> - `Improvements to Variant type`
> - `Sum types in programming languages`
> - `Mojo documentation needs`
> - `Pattern matching and exhaustiveness checking` 


- **Mojo 错误信息未引用用户代码**：一位成员指出，他们的代码没有在错误信息中被提及，错误信息反而依赖于标准库的实现。
   - 另一位成员指出，考虑到目前的实现方式，短期内改进这些信息存在局限性。
- **提议将 Variant 演进为安全的 Tagged Union**：一位成员计划提议将 `Variant` 类型演进为“安全”的 Tagged Union，以便更好地进行模式匹配（Pattern Matching）。
   - 讨论围绕确保这一变化能与现有的 Traits 和模式匹配预期良好集成展开。
- **对原生 Sum Types 的渴望**：成员们表达了对类似于 Rust 或 Swift Enums 的原生 Sum Types 的渴望，强调了它们在消息传递系统中的效率。
   - 一位成员指出了使用简单 Variant 的人体工程学优势，并将其与引入复杂性的多态 Variant 进行了对比。
- **呼吁改进 Mojo 文档**：由于现有的用户困惑，成员们一致认为需要更好的关于 Mojo 和 MLIR Dialects 的公开文档。
   - 文档的缺失导致了对构造（Constructs）的不当使用，阻碍了开发工作。
- **穷举性检查（Exhaustiveness checking）和类型推断**：讨论强调了穷举性检查在系统设计中的重要性，它能实现更安全的重构实践。
   - 成员们对仅依赖类型推断（Type Inference）可能导致的意外类型冲突表示担忧。



**提到的链接**：<a href="https://github.com/VitWW/rfcs/blob/partial_types3/text/0000-partial_types.md">rfcs/text/0000-partial_types.md at partial_types3 · VitWW/rfcs</a>：Rust 更改的 RFC。通过在 GitHub 上创建账户为 VitWW/rfcs 的开发做出贡献。

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1288990792093929577)** (20 messages🔥): 

> - `FTC Crackdown on AI Tool Claims`
> - `Concerns About Generative AI Sustainability`
> - `Geohot's Frustration with AMD`
> - `ColPali Model with Qwen2-VL`
> - `Effectiveness of AI in Software Development` 

- **FTC 严厉打击误导性 AI 宣传**：FTC 宣布严厉打击有关 AI 工具的虚假宣传，重点关注了像 **Do Not Pay** 等公司因误导性营销手段而涉及的潜在欺诈案件，详见其 [投诉文件 PDF](https://www.ftc.gov/system/files/ftc_gov/pdf/DoNotPayInc-Complaint.pdf)。
   - 讨论内容包括对 FTC 关于 AI 定义的质疑，以及担心许多初创公司可能会因此类行动而受到审查。
- **对 Generative AI 可持续性的质疑**：一篇文章认为当前 Generative AI 的繁荣是不可持续的，预言可能会发生灾难性的崩溃，从而损害大型科技公司和初创生态系统，详见 [newsletter](https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter)。
   - 批评者指出，该论点在技术理解和用例方面存在不足，并指出 GitHub Copilot 等工具已经展示了明确的商业价值。
- **Geohot 表达对 AMD 的沮丧**：Geohot 分享了对继续与 AMD 合作感到动力不足的情绪，在意识到 RDNA3 之后没有计划推出重大芯片后，他对公司的未来表示怀疑。
   - 这种情绪反映了社区内更广泛的担忧，即由于 AMD 进展停滞而导致的动力缺乏。
- **对新 ColPali 模型发布的兴奋**：社区庆祝新模型 **ColQwen2** 的发布，该模型通过利用 **Qwen2-VL** 骨干网络提高了准确性和效率，从而实现了比之前版本更显著的性能提升。
   - 该模型被视为视觉识别能力的重大进步，在 Vidore Leaderboard 性能上取得了显著提升。
- **关于 AI 商业价值的辩论**：关于 Generative AI 真实商业价值的辩论日益激烈，一些声音声称其缺乏有效性，而另一些人则引用了 GitHub Copilot 等软件工程用例中产生巨大价值的证据。
   - 研究表明，这些工具每年为 Amazon 等公司节省数亿美元，这促使人们呼吁围绕 AI 技术的实际影响和应用进行更细致的讨论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/simonw/status/1839030384949854642">来自 Simon Willison (@simonw) 的推文</a>：看起来 FTC 对 AI 有一个非常有信心的定义。引用 Charlie Dolan (@cdolan92) 的话：FTC 宣布了与 AI 相关的打击行动。辣评：好！有很多诈骗...</li><li><a href="https://x.com/__tinygrad__/status/1839221471182512632?s=46">来自 tiny corp (@__tinygrad__) 的推文</a>：@AMD 在意识到 RDNA3 之后没有大芯片后，我们有点失去了继续研究它的动力。从我们的角度来看，AMD 没有真正的未来。我知道这让我们自食其果...</li><li><a href="https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsletter">次级 AI 危机</a>：我在本通讯中写的任何内容都不是为了播种怀疑或“仇恨”，而是对我们现状以及当前路径可能走向的冷静评估。我相信人工...</li><li><a href="https://www.latent.space/p/mar-jun-2024">AI 寒冬之风</a>：2024 年 3 月至 6 月回顾：人们对 AI 夏天产生怀疑。这就是为什么 AI Engineers 是解决方案。</li><li><a href="https://x.com/ggerganov/status/1839703977073487993">来自 Georgi Gerganov (@ggerganov) 的推文</a>：是的，http://ggml.ai 将从 HF 上所有使用 llama.cpp 驱动的端点中获得收入分成。所以对于任何想支持我们的人，请务必尝试这些端点 ♥️ 引用 swyx.a...</li><li><a href="https://www.wheresyoured.at/subprimeai/?ref=ed-zitrons-wheres-your-ed-at-newsl">次级 AI 危机</a>：我在本通讯中写的任何内容都不是为了播种怀疑或“仇恨”，而是对我们现状以及当前路径可能走向的冷静评估。我相信人工...</li><li><a href="https://greaterdanorequalto.com/ai-code-generation-as-an-agent-of-tech-debt-creation/">AI 代码生成作为技术债产生的代理</a>：我已经在我的 IDE 中禁用了所有基于 LLM 的 AI Assistants/Copilots/无论你怎么称呼它们。</li><li><a href="https://x.com/garrisonlovely/status/1839655744850772272?s=46">来自 Garrison Lovely (@GarrisonLovely) 的推文</a>：这篇文章充满了重磅炸弹。@dseetharaman 的出色报道。最大的一个是：OpenAI 匆忙测试了 GPT-4o（已有报道），发布了模型，随后确定了...</li><li><a href="https://x.com/jobergum/status/1839667559404093658?s=46">来自 Jo Kristian Bergum (@jobergum) 的推文</a>：我们的祈祷得到了回应。一个 ColPali 模型，但采用了 Qwen2-VL 骨干网络！这意义重大，因为 - 它提升了准确性（巨大收益） - 更少的 patch 向量，更高效 - 许可宽松 引用...</li><li><a href="https://www.ftc.gov/news-events/news/press-releases/2024/09/ftc-announces-crackdown-deceptive-ai-claims-schemes">FTC 宣布打击虚假 AI 声明和计划</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1289315495488786433)** (38 条消息🔥): 

> - `AI Engineering 面试`
> - `屏幕共享问题`
> - `使用本地模型`
> - `Braintrust JSON mode`
> - `COT 实验` 


- **对 AI Engineering 面试感到兴奋**：一位成员表达了对面试的喜悦，这可能会带来一个 **AI Engineering** 的职位。
   - *“参加了一个可能转为 AI Engineering 职位的面试，所以我很高兴。”*
- **屏幕共享的困扰**：多位成员遇到了屏幕共享问题，并建议重新加载或切换平台。
   - *“退出再重新进入对我有效……”* 强调了为解决持续加载问题而尝试的不同方法。
- **本地模型讨论**：有人提出了创建一个 **本地模型** 是否会增强功能的问题，并对其潜在益处表示好奇。
   - *“你认为制作一个本地模型来做这些事情会有所改善吗？”*
- **关于 Braintrust 能力的问题**：一位成员询问了 **Braintrust** 是否支持 **JSON mode**，寻求对其集成的澄清。
   - *“我不熟悉 Braintrust，他们允许 JSON mode 吗？”* 引发了关于约束和灵活性的讨论。
- **实验 COT 技术**：成员们分享了关于使用 **chain-of-thought (COT)** 技术的俏皮评论，并讨论了整体实验经验。
   - *“罕见的 COT 失败 😂”* 反映了对模型实验复杂性的轻松看法。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1288995177469444096)** (4 条消息): 

> - `Paragon 集成`
> - `Langfuse 和 PostHog 教程`
> - `LlamaIndex 与 Box`
> - `FinanceAgentToolSpec`
> - `RAG 与 LlamaIndex` 


- **Paragon 构建功能丰富的聊天机器人**：来自 [useparagon](https://t.co/KEE2LOnGoR) 的博客文章和视频展示了他们如何使用 LlamaIndex 的 create-llama 创建一个聊天机器人，该机器人可以与来自 **Slack**、**Google Drive** 和 **Notion** 的客户数据进行交互。
   - *它能够持续且实时地摄取数据*，使集成非常高效。
- **Langfuse 和 PostHog 增强 MistralAI**：在 [Jupyter notebook](https://t.co/KGxjjoO0vM) 中分享的教程解释了如何设置 **Langfuse** 以追踪 LLM 应用，并集成了 **PostHog** 进行用户分析。
   - 这种配置为 AI 应用提供了全面的**监控**和**分析**，简化了开发流程。
- **LlamaIndex 与 Box 的合作公开**：Alex Novotny 和 **@seldo** 在最近的对话中讨论了 **LlamaIndex** 与 **Box** 的集成，重点介绍了 **LlamaParse**、**LlamaCloud** 和 **LlamaHub** 等功能。
   - 他们还深入探讨了 **RAG** 的工作原理以及用户在进行此类集成时应考虑的事项。点击[此处](https://t.co/KL0kkDTY65)查看视频。
- **FinanceAgentToolSpec 解锁金融数据**：**LlamaHub** 上的 **FinanceAgentToolSpec** 软件包允许 Agent 查询来自 **Polygon**、**Finnhub** 和 **Seeking Alpha** 等来源的公开金融数据。
   - Hanane 的文章详细阐述了该软件包在利用 LlamaIndex 进行金融分析时的实用性。阅读更多请点击[此处](https://t.co/7bsEm4Er1m)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1288980334410469429)** (28 条消息🔥): 

> - `NLTK 资源问题`
> - `在 GPU 上加载微调模型`
> - `最佳开源向量数据库`
> - `自托管可观测性工具`
> - `向量搜索优化策略` 


- **NLTK 的 punkt 资源缺失**：一位用户报告在使用 **NLTK** 时遇到 *Resource punkt not found* 错误。另一位成员建议检查 **llama-index** 的版本，因为最新版本使用的是 *punkt_tab*。
   - 提到了与 NLTK punkt 相关的*资源问题*，暗示了潜在的兼容性顾虑。
- **加载微调后的 Llama3.1-8B 遇到挑战**：一位用户在将其为 Text2SQL 任务本地微调的 **Llama3.1-8B** 加载到 GPU 时遇到困难。成员们建议手动加载模型和 tokenizer，并确保在初始化期间将其放置在 GPU 上。
   - 分享了一个详细的代码片段，展示了如何使用量化（quantization）设置模型以优化性能。
- **关于开源向量数据库的讨论**：一位用户询问在没有高级检索机制的应用中，哪种开源向量数据库最好。他们强调需要即使在简单设置下也能良好运行的选项，特别是针对其高级 RAG 应用。
   - 他们分享了一个关于向量数据库的视频链接，可能是想从社区寻求进一步的见解。
- **自托管可观测性工具推荐**：自托管可观测性工具的推荐包括 **Arize Phoenix**，它提供了追踪 LlamaIndex 应用的框架。另一个提到的选项是 **LlamaIndex 内置的插桩（instrumentation）**可观测性功能。
   - 一位用户表达了对能够轻松配合 Docker 使用的解决方案的需求，强调了简化设置过程的必要性。
- **针对客户支持优化向量搜索**：一种提出的向量搜索优化策略涉及将问题存储在向量分块（vector chunk）中，而将答案保留在元数据（metadata）中。该方法旨在通过在搜索过程中专注于问题的语义来提高准确性。
   - 用户寻求对该策略的验证，并欢迎对其方法进行进一步改进的建议。



**提到的链接**：<a href="https://docs.llamaindex.ai/en/stable/examples/cookbooks/oreilly_course_cookbooks/Module-5/Observability/">Observability - LlamaIndex</a>：未找到描述内容

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1289248624173256787)** (11 messages🔥): 

> - `OpenAI 仓促发布 GPT-4o`
> - `安全团队面临的挑战`
> - `员工薪酬诉求`
> - `领导层更迭`
> - `人才招聘工作` 


- **OpenAI 在竞争压力下仓促发布 GPT-4o**：高管们旨在赶在 Google 开发者大会之前推出 **GPT-4o**，尽管存在担忧，仍导致了匆忙发布。据报道，这一决定是基于**不完整的安全数据**做出的，而后续数据表明该模型的部署风险过高。
   - [@dseetharaman](https://x.com/garrisonlovely/status/1839655744850772272?s=46) 的一篇文章强调了安全团队如何每天工作 **20 小时**，几乎没有时间进行彻底检查。
- **OpenAI 面临内部薪酬不满**：[The Information 的文章](https://www.theinformation.com/articles/behind-openais-staff-churn-turf-wars-burnout-compensation-demands) 揭示了随着 OpenAI 估值上升，员工不断提出薪酬要求。据报道，近年来员工通过出售利润单位已套现**超过 12 亿美元**。
   - 包括新任 CFO **Sarah Friar** 在内的领导层正在努力应对研究人员因经济顾虑而威胁离职的问题，而此时人才竞争正日益加剧。
- **领导层更迭影响 OpenAI 的稳定性**：持续的领导层更迭与核心研究人员**要求增加薪酬**有关。在面对来自 **Safe Superintelligence** 等初创公司的竞争性报价时，领导层在留住人才的谈判中产生了紧张局势。
   - 这些挫折和招聘压力加剧了人才危机，促使 OpenAI 领导层提供丰厚的反向报价（counteroffers）以留住研究人员。



**提到的链接**：<a href="https://x.com/garrisonlovely/status/1839655744850772272?s=46">Garrison Lovely (@GarrisonLovely) 的推文</a>：这篇文章充满了重磅炸弹。@dseetharaman 的报道非常出色。最大的一个是：OpenAI 仓促进行了 GPT-4o 的测试（已有报道），发布了模型，随后确定了...

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1288980447895883786)** (15 messages🔥): 

> - `OpenAI 领导层变动`
> - `员工的公开声明`
> - `AI 文化差异`
> - `科技界的行业情绪反应`
> - `游戏玩家文化术语` 


- **OpenAI 面临领导层转型**：一位团队成员感人至深的告别信强调了核心领导者 **Mira, Bob, and Barret** 的离职，并将其比作中世纪父母面对丧亲之痛时的韧性。
   - 尽管存在缺陷，OpenAI 仍因其才华横溢的员工队伍而受到赞扬，离职的领导者也被祝愿在未来的事业中一切顺利。
- **员工公开分享内部纠纷**：成员们开玩笑地注意到 OpenAI 员工如何公开表达内部纠纷（drama），将其比作皇室的 PR 声明。
   - 一名实习生幽默地哀叹自己的辞职，将他们在 OpenAI 的时光比作珍爱新生儿，引发了人们对透明度文化的质疑。
- **AI 文化与其他行业的对比**：讨论中提到了 AI 圈子中独特的文化，并评论了 AI 社区中极高的“特权感”（entitlement）。
   - 这引发了关于 AI 从业者与传统科技公司员工相比，在表达方式上是多么独特的结论。
- **对话中引用了游戏玩家文化**：一位成员幽默地指出了文化差异，用游戏术语“tilted”来形容 OpenAI 员工的公开表达。
   - 这引发了一场关于此类术语在游戏圈之外是否被广泛认可的对话。
- **在 OpenAI 动荡之际，Anthropic 保持稳定**：与 OpenAI 形成鲜明对比的是，据指出 **Anthropic** 的所有联合创始人目前仍留在公司。
   - 这增加了一层稳定性，与 OpenAI 领导层变动中所表现出的情绪动荡形成鲜明对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/woj_zaremba/status/1839696945008582672?s=46">Wojciech Zaremba (@woj_zaremba) 的推文</a>：看到 Mira、Bob 和 Barret 离开很难过——不仅因为他们是优秀的领导者，还因为我会想念每天见到他们的日子。他们是我的朋友。他们的离开让我想到了...</li><li><a href="https://www.urbandictionary.com/define.php?term=tilted>)">Urban Dictionary: tilted&gt;)</a>：当你对某人/某事感到有点生气时。</li><li><a href="https://x.com/sashadem/status/1839728129935540589?s=46">Sasha de Marigny (@sashadem) 的推文</a>：很高兴地报告，Anthropic 的联合创始人目前都还在公司愉快地工作。没有人因为中世纪瘟疫或按摩浴缸而离开。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1289294779179470879)** (4 messages): 

> - `Substack Best Seller`
> - `Apple App Store Management` 


- **Substack 实现 iPhone IAP 订阅**：作为 **Substack best seller**，现在获得了 **iPhone In-App Purchase (IAP) 订阅**的新权限。
   - 这一机会凸显了移动平台上数字出版的增长潜力。
- **Apple App Store 的幕后**：讨论揭示了关于管理 **Apple App Store** 的有趣见解，开发者通常将其视为一场**恐怖秀 (horror show)**。
   - 成员们对应对这些挑战的**幕后实现**表示了浓厚兴趣。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

420gunna: https://x.com/venturetwins/status/1839685317462458650
秒锁这个 (Instalocking this)
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1289000758473265194)** (7 messages): 

> - `Multimodal support`
> - `Area Chair roles`
> - `Conversation splitting in training` 


- **开源社区在 Multimodal 支持方面滞后**：一位成员指出，在整个行业向该方向转型之际，**开源社区**在采用 **Multimodal** 支持方面正处于落后地位。
   - 这种情绪反映了人们对社区**创新 (innovation)** 速度日益增长的担忧。
- **理解 Area Chair 角色**：一位成员解释说，**AC** 指的是被称为 **Area Chair** 的元评审员 (meta reviewer)，他在评审过程中发挥着关键作用。
   - 这一见解强调了在学术和协作环境中组织工作的重要性。
- **用于对话处理的 Python 代码片段**：一位用户展示了一个 Python 代码片段，旨在为训练目的而**拆分对话**，确保对话不会超过**最大序列长度 (maximum sequence length)**。
   - 他们强调了其效用，特别是在处理长对话的同时保留训练数据集中的上下文。
- **对话预处理 (Preprocessing) 的增强**：另一位成员建议在添加到对话分段之前实施**消息长度**检查，以确保数据完整性。
   - 他们强调了该功能在某些数据集的 **preprocessing** 流水线中潜在的实用性。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1288959433853898814)** (19 messages🔥): 

> - `Multi-modal VLM 协助`
> - `YAML 配置问题`
> - `Flex Attention 讨论`
> - `LoRA+ 优化更新`
> - `LoRA+ 中的默认学习率` 


- **寻求 Multi-modal VLM 挑战方面的帮助**：一名成员对目前的 **multi-modal** 设置表示沮丧，称数据 **pre-processing**（预处理）并非最优。
   - 他们邀请其他人参与贡献，建议通过协作解决问题来推动进展。
- **关于学习率 YAML 配置的困惑**：多名成员讨论了其 YAML 配置中的特定参数，包括 `loraplus_lr_ratio` 和 `loraplus_lr_embedding`。
   - 一名成员指出，他们需要一个完整的 YAML 文件来排查其复现（reproduction）问题。
- **引入 Flex Attention 进行优化**：一名成员强调 **Flex Attention** 是一种新的优化实现，与之前的 Attention 方法相比提供了更大的灵活性。
   - 共享了多个资源，包括详细介绍其设计的 [PyTorch 博客链接](https://pytorch.org/blog/flexattention/)。
- **LoRA+ 优化修复更新**：一名成员请求将 `loraplus_lr_embedding` 设置为特定值，并引用了 [最近 GitHub PR 中的修复](https://github.com/axolotl-ai-cloud/axolotl/pull/1932)。
   - 他们解释说，由于未能为该参数使用默认值，该修复至关重要。
- **关于 LoRA+ 默认学习率的讨论**：成员们质疑是坚持使用 `loraplus_lr_embedding` 的默认学习率，还是将其与主学习率匹配。
   - 他们注意到 **LoRA+** 论文在其主要学习率中使用了 `1e-6`，这可能解释了默认设置的原因。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/flexattention/">FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention</a>：</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/8fc300b747aa09e09ab80be0b11ab70726985e26/src/llama_recipes/finetuning.py#L226-L245">llama-recipes/src/llama_recipes/finetuning.py</a>：用于微调 Meta Llama 的脚本，采用可组合的 FSDP 和 PEFT 方法，涵盖单节点/多节点 GPU。支持默认和自定义数据集，适用于摘要和问答等应用。</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1932">winglian 修复了空的 lora+ lr embedding · Pull Request #1932 · axolotl-ai-cloud/axolotl</a>：未找到描述。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/)** (1 messages): 

invisietch: Fp8 在我的 2x 80GB A100 上可以运行，在 2x H100 上应该也没问题。
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1289009015283519590)** (21 条消息🔥): 

> - `Nvidia P2P 支持与 IOMMU`
> - `GPU 云服务的定价与竞争`
> - `CLOUD=1 服务详情`
> - `训练中的数据上传挑战`
> - `持久化存储计费` 


- **IOMMU 在 Nvidia P2P 中的作用**：一位用户询问在使用 [tinygrad GPU modules](https://github.com/tinygrad/open-gpu-kernel-modules) 时，为什么需要关闭 IOMMU 才能支持 Nvidia P2P，寻求对其交互机制的澄清。
   - 目前没有立即的回应，表明需要对该技术交互进行更深入的了解。
- **GPU 云定价竞争引发讨论**：George Hotz 提议 **GPU 费率为 $0.50/小时**，按秒计费，这引发了与 salad.com、vast.ai 和 runpod 等更便宜选项的对比。
   - 讨论中提出了关于增值税（VAT）影响的担忧，以及该定价是否考虑了竞争对手所涵盖的税收成本。
- **CLOUD=1 特性引发辩论**：关于 **CLOUD=1** 是包含 CPU 资源还是仅包含 GPU 展开了讨论，用户对于需要保持设备在线的要求感到不安。
   - 参与者认为，除了节省价格外，还需要更好的解决方案来证明该服务结构的合理性。
- **机器学习任务的数据上传挑战**：一位成员强调，连接和上传大型数据集是一个主要的痛点，希望 tinygrad 能够简化这一过程。
   - 挑战在于数据计算比，这可能会影响 **mini LLMs 和 CNNs** 等较小模型的效率。
- **对持久化存储成本的考量**：一位用户询问是否有计划解决 **持久化存储计费** 问题，并指出许多供应商对此进行单独收费。
   - 这一担忧反映了对云计算服务整体成本结构的广泛顾虑。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1289103130667126835)** (4 条消息): 

> - `Pull Request #6779`
> - `设备加载问题`
> - `PR 对比` 


- **提交 Pull Request #6779**：一位用户提交了他们的第一个 PR 尝试，标题为 [get_available_backends](https://github.com/tinygrad/tinygrad/pull/6779)，并表示这可能不像预期的那样简洁，因为 George 更倾向于单行代码实现。
   - 他们请求针对需要进一步研究改进的领域提供反馈。
- **未察觉的竞争性 PR**：该用户意识到已经提交了另一个他们之前并不知道的 PR，而且那个 PR 似乎比他们自己的尝试更好。
   - 对他们来说很*遗憾*，因为他们已经发现了本可以对自己 PR 进行的改进。
- **George 批评现有 PR**：George 评论说竞争对手的 PR 也不好，表明对质量的持续担忧。
   - 用户推测问题是否源于 `Device.DEFAULT` 加载了每个设备，这预示着当前实现中潜在的问题。



**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/6779">get_available_backends for device by i-jared · Pull Request #6779 · tinygrad/tinygrad</a>：尝试加载每个 backend。返回任何加载成功的 backend。虽然不像 George 要求的 #6689 那样只有 1 行，但它可以工作并遵循代码库中现有的惯例。

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1288940590301446184)** (15 条消息🔥): 

> - `Llama 3.2 11B Vision`
> - `Voice Cloning`
> - `Family Photo Generation`
> - `Copyright Enforcement`
> - `Maintaining Independence` 


- **Llama 3.2 11B Vision 免费开放**: TogetherCompute 与 @AIatMeta 合作免费提供 **Llama 3.2 11B Vision**，以便开发者可以免费尝试开源的 **Multimodal AI**，无限访问地址见[此处](https://api.together.ai/playground/chat/meta-llama/Llama-Vision-Free)。
   - 为了获得更好的性能，他们还为 Llama 3.2 11B 和 90B 提供了付费的 **Turbo endpoints**。
- **关于无限访问的讨论**: 成员们讨论了无限访问 Llama 3.2 的影响，幽默地建议可以用它来为整个 **LAION dataset** 生成 caption。
   - 这引发了一个关于社区参与的轻松建议。
- **Voice Cloning 对话**: 一位成员幽默地提到与自己的 **Voice Cloning** 副本对话，为讨论增添了轻松的氛围。
   - 这引起了成员们的参与和笑声。
- **对照片生成应用的关注**: 一位成员询问了关于生成**家庭照片**的特定应用的有效性，表现出对 AI 驱动解决方案的兴趣。
   - 这表明人们对 AI 在创建个人和用户特定内容方面的能力越来越感兴趣。
- **Copyright Enforcement 的胜利**: 一位成员分享了一篇 LinkedIn 帖子，庆祝最近在 **Copyright Enforcement** 方面的胜利，暗示**正义的一方赢得了这一回合**。
   - 这被强调为社区内诚信和独立性的胜利。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/togethercompute/status/1839071026728333778">来自 Together AI (@togethercompute) 的推文</a>: 🚀 我们与 @AIatMeta 合作免费提供 Llama 3.2 11B Vision，以便开发者可以零成本尝试开源的 Multimodal AI。除了我们的免费额度外，在限定时间内...</li><li><a href="https://smallpdf.com/result#r=a30cd403fcb0a6c119f2933a843cfe07&t=share-document?trk=feed-detail_comments-list_comment-text">Smallpdf.com</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1289070350336000073)** (8 messages🔥): 

> - `Positional Information in CNNs` (CNN 中的位置信息)
> - `Positional Encoding in Transformers` (Transformer 中的位置编码)
> - `Scaling Laws in Machine Learning` (机器学习中的 Scaling Laws)
> - `Fourier Feature Extraction` (傅里叶特征提取)
> - `Trends in Neural Network Architectures` (神经网络架构趋势)


- **对 Latent Pixel 定位的困惑**：一位成员对位置信息如何整合进 Latent Pixel 的特征向量表示困惑，并指出尽管它们在与 CLIP 文本嵌入的 Cross-attention 中发挥作用，但似乎缺乏显式的位置编码。
   - 另一位成员指出，模型中的 Self-attention 步骤也有助于这一过程，并强调卷积边缘为 Attention 比较提供了位置数据。
- **CNN 隐式学习位置信息**：讨论重点提到了一篇论文，该论文探讨了 CNN 如何通过使用局部滤波器实现效率，同时也指出了隐式学习到的绝对位置信息的重要性。
   - 论文表明，包括 Zero-padding 在内的填充技术有助于传递位置信息，从而促进卷积层的学习。
- **Scaling Laws 对机器学习原理的影响**：一位成员推荐阅读一篇根据 Scaling Laws 重新审视机器学习原理的论文，指出研究重点已从最小化泛化误差转向减少近似误差。
   - 该论文挑战了在大语言模型背景下某些正则化原则的有效性，并强调了一种被称为“Scaling Law Crossover”的现象。
- **未来：单一的 Transformer 模型？**：一位成员幽默地总结道，神经网络的未来可能只会通向“一个巨大的 Transformer”。
   - 这一评论反映了随着 Transformer 模型不断进步，架构趋于简化的广泛趋势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.15156">Rethinking Conventional Wisdom in Machine Learning: From Generalization to Scaling</a>：大语言模型预训练的显著成功和 Scaling Laws 的发现标志着机器学习范式的转变。值得注意的是，主要目标已从最小化泛化……演变而来。</li><li><a href="https://arxiv.org/abs/2001.08248">How Much Position Information Do Convolutional Neural Networks Encode?</a>：与全连接网络相比，卷积神经网络 (CNN) 通过学习与具有有限空间范围的局部滤波器相关的权重来实现效率。这带来的一个影响是……</li><li><a href="https://emu.baai.ac.cn/about">Emu3</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1288959843440529544)** (16 messages🔥): 

> - `Lecture Coverage on Social Alignment` (关于社会对齐的课程覆盖)
> - `Course Enrollment Confirmation` (课程注册确认)
> - `Assignment Deadlines and Clarifications` (作业截止日期与澄清)
> - `Qquiz Availability` (Quiz 可用性)
> - `Lab Assignment Release Timing` (实验作业发布时间)


- **预计课程将涵盖社会对齐**：有人担心课程是否会涉及 LLM Agent 的**社会对齐 (Social Alignment)**，因为最后两节课似乎侧重于 **AI Safety**。
   - *Prof Dawn Song 的研究可能会在 12 月 2 日*关于 **LLM Safety** 的演讲中触及这些主题。
- **澄清课程报名流程**：一位用户询问在使用提供的 Google 表单链接报名课程后是否会收到确认。
   - 另一位用户回复称，填写报名表即可获得所有课程材料的访问权限，作业截止日期标注为 2024 年 **12 月 12 日**。
- **作业截止日期引起困惑**：一位参与者寻求关于作业截止日期的澄清，注意到伯克利校内学生和 MOOC 学生之间发布的截止日期可能存在差异。
   - 确认信息显示，MOOC 网站上的**所有作业**截止日期均为 2024 年 **12 月 12 日**，且网站上已确认 Quiz 的可用性。
- **关于 Quiz 3 可用性的困惑**：一位参与者表示难以找到 **Quiz 3**，想知道它是否已被移除。
   - 其他人澄清说，它在 **MOOC 学生网站**上仍然可以访问，而不是在伯克利学生网站上。
- **关于实验作业发布的查询**：一位关注实验作业的参与者在阅读完 MOOC 网站后，寻求有关其发布时间表的信息。
   - 这表明关于课程结构和时间表的讨论正在进行，强调了学生之间明确信息的重要性。



**提到的链接**：<a href="https://llmagents-learning.org/f24">Large Language Model Agents</a>：未找到描述

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1289034521693651035)** (6 messages): 

> - `OpenInterpreter application`
> - `LLaMA 中的多模态支持`
> - `OI 的前端开发`
> - `链上分析演示` 


- **使用 OpenInterpreter 转向最终解决方案**：一位成员演示了如何使用 [OpenInterpreter](https://x.com/parrotexplore/status/1839721139515302137) 进行链上分析，展示了从**可能有效**的代码向**完全功能化代码**的转变。
   - 他们分享了一个用于 Python 代码的 [Google Colab](https://t.ly/vBSPe) 链接，并欢迎社区成员进行**转发**。
- **LLaMA 中的多模态支持问题**：讨论了在 LLaMA 项目中恢复**多模态支持**的问题，指出该功能自 **#5882** 以来已被移除。
   - 更新取决于 **llava** 的重构；该线程主要用于通过相关 issue 的链接来跟踪进度。
- **令人兴奋的 OI 前端开发**：一位成员表达了对为 OpenInterpreter 创建 **Electron 前端**的兴奋之情，强调了其酷炫之处。
   - 这种兴奋情绪表明社区对正在进行的开发工作持积极态度。
- **在 OpenInterpreter 社区分享良好氛围**：一位成员向更广泛的 **Open Interpreter X 社区**分享了 OpenInterpreter 应用帖子，强调了其价值。
   - 这种分享精神突显了 OpenInterpreter 开发参与者之间令人鼓舞的环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.ly/vBSPe">Google Colab</a>: 未找到描述</li><li><a href="https://x.com/parrotexplore/status/1839721139515302137">来自 Parrot Explorator (@parrotexplore) 的推文</a>: 强大的 @OpenInterpreter 可用于从“ChatGPT 给我们提供可能有效的代码”过渡到“获得实现目标的最终解决方案代码”。这里有一个演示...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2345831496">server: 恢复多模态支持 · Issue #8010 · ggerganov/llama.cpp</a>: 多模态功能自 #5882 起已被移除。取决于 llava 的重构，我们将能够恢复该支持：#6027。创建此 issue 主要是为了跟踪目的。如果有人想...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8010#issuecomment-2376339571">server: 恢复多模态支持 · Issue #8010 · ggerganov/llama.cpp</a>: 多模态功能自 #5882 起已被移除。取决于 llava 的重构，我们将能够恢复该支持：#6027。创建此 issue 主要是为了跟踪目的。如果有人想...
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1289301733473914972)** (7 messages): 

> - `解码数据包错误`
> - `服务器连接问题`
> - `请求设置信息` 


- **解码数据包错误通知**：一位用户报告了一条警告消息，指出在处理具有特定进程和作业 ID 的输入时，由于*发现无效数据*而导致**解码数据包错误**。
   - 每当服务器重启或尝试客户端连接时，此问题都会一致出现，且没有其他终端错误。
- **手机卡在“Starting...”屏幕**：另一位用户描述了他们的手机在尝试连接时卡在 **“Starting...”** 页面。
   - 这一持续存在的问题引发了关于系统连接稳定性的进一步疑问。
- **请求详细的设置信息**：一位社区成员建议该用户发布一个帖子，详细说明他们的**设置**（OS、复现步骤），并包含更完整的终端输出打印。
   - 他们建议在指定频道分享这些信息，以便更好地进行协助和故障排除。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1289152581037326452)** (3 条消息): 

> - `HF 90b vision update`
> - `Impact of OpenInterpreter` 


- **HF 推出 90b vision 模型**：一位成员宣布 **HF** 已更新 **90b vision** 模型，现已开放使用。
   - 预计此次更新将显著增强各种 vision 相关的任务。
- **OpenInterpreter 改变生活**：一位成员分享了他们的经历，表示 **OpenInterpreter** 完全改变了他们的生活，让他们结识了了不起的朋友并深入探索了 A.I. 世界。
   - 他们表达了对社区的感谢，并强调了加速 **open source tech** 的使命，同时引用了他们一年前走红的 demo。



**提到的链接**：<a href="https://x.com/MikeBirdTech/status/1839750338179674590">来自 Mike Bird (@MikeBirdTech) 的推文</a>：一年前的今天，我为在网上发现的这个酷炫新工具制作了一个小 demo。只是想展示一下它的功能，结果它就走红了。从那时起，@OpenInterpreter 完全改变了...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1289179389874933853)** (2 条消息): 

> - `Vector search optimization`
> - `Contextual extraction from Excel` 


- **为客户支持优化 vector search**：一种提议的 vector search 优化策略包括在 vector chunk 中存储问题，并在带有 prompt 的 metadata 中存储答案，旨在提高问题匹配的 **precision**。
   - 在这种方法中，重点仍然是问题的 **semantics**，通过减少无关信息，可能会获得更准确的搜索结果。
- **从复杂的 Excel 文件进行 contextual extraction 的挑战**：一位成员表示，很难找到从复杂的 Excel 文件中进行 **contextual extraction** 的有效方法，以便让 LLMs 做出有意义的响应。
   - 尽管进行了广泛的搜索，他们尚未发现能够促进这一过程的可行解决方案。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1289085277553557557)** (2 条消息): 

> - `CF Booking Chatbot`
> - `Unize Storage AI System`
> - `Knowledge Graph Generation`
> - `Google Calendar Integration`
> - `LangChain Performance Comparison` 


- **CF Booking Chatbot 简化会议室管理**：一位成员发布了他们使用 LangChain 构建的新 **CF Booking Chatbot**，它可以简化会议室的可用性检查、预订和管理。该帖子包含一个[展示其功能的 demo 视频](https://www.linkedin.com/posts/ismile-bharmal-3b82241ab_langchain-chainlit-ai-activity-7245291326919872512-O06M)，并提到计划集成 **Google Calendar** 以实现自动同步。
- **Unize Storage 承诺增强 graph generation**：另一位成员介绍了一个名为 **Unize Storage** 的 AI 系统，它可以从任何输入文本生成高质量的 knowledge graphs。他们强调其性能优于现有系统，包括 **LangChain** 的 **LLMGraphTransformer**，在处理较大输入时达到了 **85% accuracy**，而 LangChain 为 **55%**。
- **Unize 提供免费 API 访问以供实验**：**Unize API** 为用户提供了体验新 **Unize Storage** 系统并获取免费 API 额度的机会。他们可以可视化 knowledge graphs，并通过这个专为易于访问而设计的 [Playground](https://api.unize.org/signup) 开始使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://blog.unize.org/p/introducing-unize-storage">Introducing Unize Storage</a>：一个用于大规模生成高质量 knowledge graphs 的 AI 系统。</li><li><a href="https://developers.unize.org/kgstorage.">Introduction - Unize API</a>：未找到描述</li><li><a href="https://api.unize.org/signup">Unize</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1288966136146296832)** (3 条消息): 

> - `PackedDataset constraint`
> - `max_seq_len handling`
> - `RuntimeError in dataset processing`
> - `Error discussion on GitHub` 


- **强制执行 PackedDataset 尺寸的需求**：一名成员建议强制规定打包后的尺寸不能超过 **dataset max length 的 2 倍**，以防止处理序列时出现错误。
   - 这一讨论是作为防止运行时不一致的潜在保护措施而提出的。
- **数据集中的简化失败案例**：据演示，当前的实现在单个输入超过 **max_seq_len** 时可能会失败，特别是在配置导致边界不匹配的情况下。
   - 建议使用显式的 token 长度门控（gating）进行修复，以防止这些运行时错误。
- **GitHub 错误链接讨论**：对话指向了一个 [GitHub 错误](https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130)，该错误表明可能已经做出了允许序列大于 **max_seq_len** 的决定。
   - 此链接可能阐明了当前处理 packed dataset 尺寸背后的原因。
- **提及待进一步审查**：一名成员建议另一名用户在返回后审查此讨论的内容，表明了其重要性。
   - 这突显了故障排除过程中的协作性质。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_packed.py#L130">torchtune/torchtune/datasets/_packed.py at main · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1288953002731765780)** (1 条消息): 

> - `Function Calling Evaluation`
> - `Customization of Evaluation Dataset`
> - `Integration with LLMs`
> - `Berkeley Function-Calling Leaderboard`
> - `Error Breakdown Analysis` 


- **对代码库中 Function Calling 评估的困惑**：一名用户对代码库的评估过程表示困惑，询问是否可以提供自己的**评估数据集**进行分析。
   - 他们特别寻求一个能够使用包含 **<prompt>, <llm_response>, <ideal response>** 的数据集来分解错误的包。
- **对本地 LLM 部署的兴趣**：该用户对允许在自己的数据集上使用**本地部署的 LLM** 以有效提取错误指标的功能感兴趣。
   - 他们请求推荐其他可能处理此类需求的代码库，特别是在 **function calling 能力**方面。
- **探索应用中的 LLM 集成**：对话强调了在 Langchain 和 AutoGPT 等各种应用中集成 **Large Language Models (LLMs)** 的日益增长的趋势。
   - 提到了包括 **GPT, Gemini, Llama,** 和 **Mistral** 在内的模型，因为它们在驱动软件解决方案方面具有出色的 function calling 能力。
- **将 Berkeley Function-Calling Leaderboard 作为资源**：用户引用了 **[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)** 作为评估 LLM function calling 能力的宝贵资源。
   - 他们指出，该排行榜是根据以用户为中心的 function calling 使用案例构建的，涵盖了各种函数调用形式。



**提到的链接**：<a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics">Berkeley Function Calling Leaderboard</a>：未找到描述

  

---



### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/)** (1 条消息): 

azaw: 我们如何为 jamba 使用 openAI sdk？这可能吗？
  


{% else %}


> 各频道的完整详细分析已在邮件中截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}