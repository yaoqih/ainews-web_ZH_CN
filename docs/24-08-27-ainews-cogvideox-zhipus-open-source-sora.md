---
companies:
- zhipu-ai
- alibaba
- meta-ai-fair
- google
- hugging-face
- nvidia
- togethercompute
- salesforce
date: '2024-08-28T01:26:46.937370Z'
description: '**智谱 AI**（阿里巴巴旗下的 AI 部门，也是中国第三大 AI 实验室）发布了开源的 5B（50 亿参数）视频生成模型 **CogVideoX**。该模型可以通过其
  ChatGLM 网页端和桌面应用运行，无需本地 GPU。


  **Meta AI** 在发布 **Llama 3.1** 的同时，宣布了信任与安全研究以及 CyberSecEval 3；目前 **Llama 3 405B**
  已在 Google Cloud Vertex AI 和 Hugging Face x NVIDIA NIM API 上提供无服务器（serverless）服务。


  其他更新包括：

  *   **Moondream**：一款开源视觉语言模型，提升了 DocVQA（文档视觉问答）和 TextVQA（文本视觉问答）任务的表现。

  *   **Phi-3.5**：拥有 16x3.8B 参数的轻量级 MoE（混合专家）聊天模型。

  *   **Together Compute**：推出了 Rerank API，采用了 Salesforce 的 **LlamaRank** 模型用于文档和代码排序。


  研究亮点包括：

  *   **叠加提示（Superposition prompting）**：无需微调即可用于 RAG（检索增强生成）。

  *   **AgentWrite 流水线**：可生成超过 20,000 字的长文本内容。

  *   **对比研究**：结果显示长上下文（Long Context）方法在成本较高的情况下表现优于 RAG。


  工具方面包括：AI 模型路由 **Not Diamond**、AI 命令行界面，以及一款开源的 **WebGPU 背景移除工具**。


  针对 CogVideoX，有评价称：*“你甚至不需要 GPU 就能运行它。”*'
id: 4f94e7b8-9da2-49af-ada9-2ee46641fb3a
models:
- cogvideox
- llama-3-1
- llama-3-405b
- moondream
- phi-3.5
- llama-rank
original_slug: ainews-cogvideox-zhipus-open-source-sora
people:
- rohanpaul_ai
- philschmid
- vikhyatk
- algo_diver
- jayalammar
- davidsholz
title: CogVideoX：智谱的开源 Sora
topics:
- video-generation
- serverless-computing
- vision
- document-vqa
- text-vqa
- mixture-of-experts
- retrieval-augmented-generation
- long-context
- model-routing
- webgpu
- background-removal
- long-form-generation
- superposition-prompting
---

<!-- buttondown-editor-mode: plaintext -->**开源 Videogen 就是你所需要的一切。**

> 2024/08/26-2024/08/27 AI 新闻简报。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**215** 个频道，**3433** 条消息）。预计节省阅读时间（以 200wpm 计算）：**369 分钟**。你现在可以艾特 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

自从 Sora 在 2 月发布以来（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-sora-pushes-sota/)），出现了一系列替代方案的尝试，包括 [Kling](https://www.youtube.com/watch?v=EIj9-xgfV2c)（未开源）和 [Open-Sora](https://github.com/hpcaitech/Open-Sora)（仅约 1B 参数）。智谱 AI（Zhipu AI），实际上是阿里巴巴的 AI 臂膀，也是中国第三大“[AI 老虎](https://www.scmp.com/tech/big-tech/article/3259499/chinas-four-new-ai-tigers-baichuan-zhipu-ai-moonshot-ai-and-minimax-emerge-investor-favourites)”实验室，[发布了其全新的开源 5B 视频模型](https://medium.com/@ChatGLM/zhipuai-unveils-cogvideox-a-cutting-edge-video-generation-model-293e3008fda0) CogVideoX。在这里我们遇到了电子邮件简报的一个经典限制，因为我们无法嵌入视频：

 
![image.png](https://assets.buttondown.email/images/4cb55e94-226b-42ad-8ec2-dbf3ef5693f0.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/4a92a7ed-2f74-4702-8a61-c17f91f0baa6.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/92dbb665-e005-495a-a3a7-0270c4c2dfe1.png?w=960&fit=max)
 


 
![image.png](https://assets.buttondown.email/images/c41abe8c-209b-47d2-93e2-5c44410a76cb.png?w=960&fit=max)
 

你甚至不需要 GPU 就能运行它 —— 你可以使用 [智谱（Zhipu）的在线 ChatGLM Web 应用或桌面应用](https://x.com/ChatGLM/status/1816803455761281211)（可能需要一位懂中文的朋友帮你注册手机号账号）—— 我们第一次尝试就成功运行了。

 
![image.png](https://assets.buttondown.email/images/e8ead844-c837-4e74-8a4d-c6412d5b6720.png?w=960&fit=max)
 

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


**AI 模型开发与发布**

- **Llama 3 及其他模型更新**：[@AIatMeta](https://twitter.com/AIatMeta/status/1828133452837007831) 发布了与 Llama 3.1 发布相关的全新信任与安全研究以及 CyberSecEval 3。[@_philschmid](https://twitter.com/_philschmid/status/1828114328936923196) 报道称 Llama 3 405B 现已在 Google Cloud Vertex AI 和 Hugging Face x NVIDIA NIM API 上提供 Serverless 服务。

- **Moondream 更新**：[@vikhyatk](https://twitter.com/vikhyatk/status/1828144274522939829) 发布了开源视觉语言模型 moondream 的更新，提升了在 DocVQA 和 TextVQA 任务中的性能。

- **Phi-3.5 模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828165274106831093) 讨论了 Phi-3.5 MoE 聊天模型，这是一款轻量级 LLM，拥有 16x3.8B 参数，通过 2 个专家（experts）使用 6.6B 激活参数。

- **Together Rerank API**：[@togethercompute](https://twitter.com/togethercompute/status/1828194058126401657) 推出了 Together Rerank API，采用 Salesforce 的 LlamaRank 模型，用于改进文档和代码排序。

**AI 研究与技术**

- **Superposition Prompting**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828147274553213092) 分享了关于 Superposition Prompting 的见解，这是一种新颖的 RAG 方法论，无需微调即可加速并增强性能。

- **长文本内容生成**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828025515791298837) 讨论了 LongWriter 论文，该论文介绍了用于生成超过 20,000 字连贯输出的 AgentWrite 流水线。

- **RAG vs. 长上下文 (Long Context)**：[@algo_diver](https://twitter.com/algo_diver/status/1828091411721527530) 总结了一篇比较检索增强生成 (RAG) 与长上下文 (LC) 方法的研究论文，发现 LC 的表现始终优于 RAG，但成本更高。

**AI 工具与应用**

- **Not Diamond**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828054608431739323) 解释了 Not Diamond，这是一个 AI 模型路由，可自动为给定查询确定最适合的 LLM。

- **命令行中的 AI**：[@JayAlammar](https://twitter.com/JayAlammar/status/1828129090970243353) 强调了 AI 在命令行界面中的潜力，能够实现跨多个文件的操作。

- **使用 WebGPU 移除背景**：[@osanseviero](https://twitter.com/osanseviero/status/1828121582981599419) 分享了一个使用 WebGPU 的完全端侧、开源背景移除工具。

**AI 行业与商业**

- **AI 招聘**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1828192803916300379) 宣布 Midjourney 正在为其核心数据团队招聘，强调了在创意能力方面学习和发挥影响力的机会。

- **超大规模企业资本支出**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1828018114468237737) 报道了超大规模企业（Hyperscaler）在数据中心支出方面的资本支出增加，其中约 50% 用于土地、租赁和建设。

**AI 伦理与监管**

- **加州 AI 法案 SB 1047**：[@labenz](https://twitter.com/labenz/status/1828189665985470613) 讨论了加州 AI 法案 SB 1047 的最新版本，该版本现在的重点是要求前沿公司制定并发布安全计划和协议。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1：用于 AI 开发的高性能硬件解决方案**


- **[Tinybox 终于进入量产阶段](https://x.com/realgeorgehotz/status/1828197925874463166?s=46&t=m9-w-4WogM5fYHxBEAFB-Q)** ([Score: 83, Comments: 28](https://reddit.com//r/LocalLLaMA/comments/1f23dh1/tinybox_is_finally_entering_production/)): **Tinybox** 是一种用于 **AI 开发** 的高性能 **GPU 集群** 解决方案，目前已进入量产阶段。该系统提供配备 **NVLink** 和 **400Gbps 网络** 的 **8x A100 80GB GPU**，与云端替代方案相比，以极具竞争力的价格为机器学习任务提供了一个强大的平台。
  - **Tinybox** 的规格可在 [tinygrad.org](https://tinygrad.org/#tinybox) 查看，其中提到了 **15k tinybox red6 x 7900 XTX** 版本。用户讨论了该系统与运行批处理后端的单张 **A100** 之间的潜在 **吞吐量对比**。
  - 针对是以 1.5 万美元构建类似的 **6x 4090 配置** 还是选择 2.5 万美元的 Tinybox 展开了辩论。**George Hotz** (imgeohot) 解释了其中的挑战，包括 **PCIe 4.0 信号完整性**、多电源供应和散热问题，并引用了一篇详细说明这些问题的 [博客文章](https://nonint.com/2022/05/30/my-deep-learning-rig/)。
  - 一些用户认为 **Tinybox 的配置乏善可陈**，建议可以用更低的成本构建。其他人则为定价辩护，指出这为 **tinygrad 开发** 提供了资金，并为企业提供了现成的解决方案。该系统适用于 **标准机架** (12U)，[文档](https://docs.tinygrad.org/tinybox/) 中提供了导轨信息。


**主题 2：开源 RAG WebUI 与本地 LLM 部署进展**



- **开源、简洁且可定制的 RAG WebUI，支持多用户并具备合理的默认 RAG 流水线。** ([Score: 130, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1f25wo0/opensource_clean_hackable_rag_webui_with/)): **Kotaemon** 是一款开源的 **RAG WebUI**，为普通用户和高级用户提供简洁的界面、**多用户支持** 以及可定制的流水线。核心功能包括：带有深色/浅色模式的 **极简 UI**、**多用户管理**、带有 **混合检索器（hybrid retriever）和重排序（re-ranking）** 的默认 RAG 配置、支持浏览器内 PDF 查看的 **高级引用支持**、**多模态 QA 支持**，以及诸如问题分解和基于 **Agent** 推理的 **复杂推理方法**。该项目旨在具有可扩展性，允许用户集成自定义 RAG 流水线，并在不同的文档和向量数据库提供商之间切换。
  - 该项目的 **GitHub 仓库** 已发布并附带安装说明。用户建议为默认容器添加 **volume** 以持久化配置，因为 Gradio 应用需要频繁的点选设置。
  - **UI 的简洁设计** 受到称赞，开发者分享称该 **主题已在 Hugging Face Hub 上发布**，可用于其他项目。[主题可以在这里找到](https://huggingface.co/spaces/lone17/kotaemon)。
  - 支持 **离线功能**，用户可以直接使用 **Ollama OpenAI 兼容服务器** 或 **LlamaCPP 本地模型**。README 提供了此设置的指南，且该应用从一开始就设计为支持离线工作。



**主题 3：分布式 AI 训练与基础设施的创新**



- **[Nous Research 发布关于 DisTrO（互联网分布式训练）的报告](https://x.com/NousResearch/status/1828121648383566270)** ([Score: 96, Comments: 7](https://reddit.com//r/LocalLLaMA/comments/1f23guc/nous_research_publishes_a_report_on_distro/)): Nous Research 发布了一份关于 **DisTrO** (**Distributed Training Over-the-Internet**) 的报告，这是一种使用消费级硬件训练大语言模型的新方法。该方法允许通过互联网连接的 **多个消费级 GPU 进行分布式训练**，这可能使更多的研究人员和爱好者能够参与到 AI 模型开发中。DisTrO 旨在通过利用 **分布式计算技术** 和 **优化节点间的数据传输** 来解决训练大模型的挑战。
  - DisTrO 被视为一个潜在的 **重大突破**，一些用户推测它可能是分布式优化器的“**圣杯**”。它可能会 **降低训练成本**，无论是对于本地/社区模型还是像 **Meta** 这样的大型公司。
  - 用户表达了怀疑态度，指出在机器学习中，**非凡的结果往往伴随着代价**。一些人质疑除了缩短训练时间外，对 **perplexity**（困惑度）和整体模型性能的影响。
  - 论文提出了一种可能的 **新缩放法则（scaling law）**，即 **模型大小增加而无需增加通信带宽**。这可能会导致转向设计 **具有更大 VRAM 和更窄互连（interconnects）的 GPU**，从而有利于计算密集型工作负载而非 I/O 密集型操作。

## Misc AI Reddit Recap

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型开发与发布**

- **FLUX 模型展现出惊人的能力**：[r/StableDiffusion 上的一篇帖子](https://www.reddit.com/r/StableDiffusion/comments/1f1pdsb/flux_is_smarter_than_you_and_other_surprising/) 讨论了关于 FLUX 模型能力的意外发现，作者指出“每一天，我都能在 FLUX 中发现令我大吃一惊的新事物”。他们认为我们距离完全理解其能力还很遥远。

- **Salesforce 发布 xLAM-1b 模型**：Salesforce 发布了一个名为 [xLAM-1b 的 10 亿参数模型](https://www.reddit.com/r/LocalLLaMA/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/)，尽管其体积较小，但在 function calling 方面达到了 70% 的准确率，超越了 GPT-3.5。

- **更新后的 Phi-3 Mini 模型**：[Rubra AI 发布了更新后的 Phi-3 Mini 模型](https://www.reddit.com/r/LocalLLaMA/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/)，具备 function calling 能力，可与 Mistral-7b v3 竞争。

- **Joy Caption 更新**：[Joy Caption 的更新版本](https://huggingface.co/Wi-zz/joy-caption-pre-alpha)已发布，支持 batching，可实现超快速的 NSFW 自然语言 captioning。

**AI 研究与技术**

- **DisTrO 分布式优化器**：[Nous Research 宣布了 DisTrO](https://www.reddit.com/r/singularity/comments/1f1x66j/nous_research_announces_distro_distributed/)，这是一系列分布式优化器，在不依赖 amortized analysis 的情况下，将 GPU 间通信减少了 1000 倍至 10,000 倍。这可能会显著加速 AI 训练。

- **AI 驱动的编程演示**：一段 [使用 Cursor 进行 AI 驱动编程的视频演示](https://www.reddit.com/r/singularity/comments/1f1wrq1/mckay_wrigley_shows_off_aipowered_coding_with/) 展示了 5 年前难以想象的能力。

**AI 安全与伦理担忧**

- **OpenAI 的 AGI safety 研究人员流失**：据一位前研究人员称，[OpenAI 近一半的 AGI safety 工作人员已经离职](https://www.reddit.com/r/OpenAI/comments/1f25bse/exodus_at_openai_nearly_half_of_agi_safety/)。这引发了关于 AI safety 研究的重要性和作用的辩论。

- **关于 AI safety 研究的辩论**：目前存在大量关于 AI safety 研究是对进步至关重要，还是可能阻碍创新的讨论。一些人认为这对于 alignment 和 interpretability 至关重要，而另一些人则认为现阶段没有必要。

**AI 应用**

- **TTRPG 地图 LoRA**：[新版本的 TTRPG 地图 LoRA](https://www.reddit.com/r/StableDiffusion/comments/1f24ao7/just_released_the_newest_version_of_my_ttrpg_maps/) 发布，展示了 AI 在生成游戏资产和地图方面的潜力。

- **AI 生成的“验证”图像**：使用 Flux Dev 配合 “MaryLee” 肖像 LoRA 以及 Runway ML 动画技术，创建了一个 [AI 生成的动画“验证”图像](https://www.reddit.com/r/StableDiffusion/comments/1f1zy8j/verification_pic_for_my_oc_ai/)。


---

# AI Discord 摘要

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要


**1. LLM 进展与基准测试**

- **DeepSeek V2 表现超越 GPT-4**：**DeepSeek-V2** 凭借其令人印象深刻的 **236B 参数**，在 **AlignBench** 和 **MT-Bench** 等基准测试中超越了 GPT-4，展示了模型性能的显著提升。
   - [DeepSeek-V2 公告](https://x.com/deepseek_ai/status/1787478986731429933)引发了关于其能力的讨论，特别是在其表现优于现有大语言模型的领域。
- **Llama 3.1 刷新速度纪录**：**Cerebras Systems** 宣布其推理服务为 Llama3.1-70B 提供了惊人的 **450 tokens/sec**，显著领先于传统的 GPU 设置。
   - 该服务提供极具经济吸引力的价格，每百万 token 仅需 **60 美分**，吸引了寻求高性价比、高性能 AI 应用解决方案的开发者。
- **Google 推出 Gemini 1.5 模型**：Google 推出了三个实验性模型：**Gemini 1.5 Flash-8B**、更强大的 **Gemini 1.5 Pro** 以及改进后的 **Gemini 1.5 Flash**，可在 [Google AI Studio](https://aistudio.google.com) 进行测试。
   - 这些新模型承诺在 **coding** 和 **复杂提示词** 方面有所增强，速率限制设定为 **每分钟 2 次请求** 和 **每天 50 次请求**，引发了对其潜在能力的关注。
  


**2. 开源 AI 发展**

- **Intel 首个三进制多模态 LLM 亮相**：Intel 推出了 [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402)，这是首个能够处理图像和文本并生成连贯响应的三进制多模态 LLM。
   - 该模型完全开源并提供训练脚本，鼓励 AI 社区探索三进制建模中的挑战与机遇。
- **Microsoft Phi 3.5 在 OCR 领域表现卓越**：微软的 [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) 模型以 MIT 许可证发布，在 OCR 任务中表现出色，特别擅长手写识别和表格数据提取。
   - 该模型在各种视觉任务中的文本识别能力引起了 AI 社区的极大兴趣，凸显了其在实际应用中的潜力。
- **LocalAI：OpenAI 的开源替代方案**：**LocalAI** 是 Ettore Di Giacinto 发起的一个开源项目，为 OpenAI 提供了一个免费替代方案，带有用于本地推理 LLM、图像和音频生成的 REST API。
   - 该平台允许在消费级硬件上运行先进的 AI 模型，无需 GPU，使强大的 AI 能力得以普及。
  


**3. 分布式训练创新**

- **Nous Research 的 DisTrO 突破**：Nous Research 发布了关于 [DisTrO 的初步报告](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)，这是一个分布式训练框架，可将 GPU 间的通信大幅减少高达 **10,000 倍**。
   - DisTrO 旨在实现大语言模型的弹性训练，而不依赖于中心化的计算资源，有可能使 AI 研发更加民主化。
- **关于 Nous Research 优化器声明的辩论**：AI 社区对 Nous Research 的新优化器表示怀疑，特别是关于其分布式训练能力的声明。
   - 讨论中提到了现有的工具如 **Petals for Bloom** 和 **OpenDILo**，强调需要更多实质性的证据来支持 Nous 在分布式 AI 训练领域的承诺。
  
  


**4. 多模态 AI 进展**

- **Cog-5B 视频模型发布**：**Cog-5B 视频模型** 在 [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b) 发布，被誉为视频生成领域最佳的开源权重，并集成了 Diffusers 库。
   - 该模型承诺在低于 **10GB** VRAM 的情况下实现高效推理，展示了结合文本、图像和视频生成能力的多模态 AI 的进步。
- **StoryDiffusion：开源版 Sora 替代方案**：**StoryDiffusion** 的推出引起了 AI 社区的兴奋，它是 Sora 的开源替代方案，采用 MIT 许可证，尽管权重尚未发布。
   - 这一发展凸显了人们对可与专有解决方案竞争的、易于获取的高质量视频生成模型日益增长的兴趣。
  

---

# 第一部分：高层级 Discord 摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO 算法持续演进**：**DisTrO 算法**正在积极优化中，通过测试不同的变体来优化**通信带宽 (communication bandwidth)**，同时确保收敛性能。
   - 成员们指出，像 **SWARM** 这样的技术可能更适合大型模型。
- **社区渴望合作**：成员们对参与 **DisTrO 算法**的实现表现出浓厚兴趣，强调开放贡献和讨论。
   - 团队计划在未来几周内分享完整的代码和细节。
- **围绕 AI 意识的哲学辩论**：AI 中**感质 (qualia)**和意识的影响引发了严肃讨论，呼吁更多的跨学科合作。
   - 成员们指出计算机科学家与哲学家之间需要更深层次的合作。
- **DisTrO 探索弱设备应用**：讨论了在旧手机等弱设备上使用 **DisTrO** 的可行性，强调了对高效训练方法的需求。
   - 虽然 **DisTrO** 在强硬件上表现出色，但探索其在低性能系统中的效用被认为很有价值。
- **Tinybox 正式开放购买**：经过 **18 个月**，**Tinybox** 现在有了“立即购买”选项，公告称有 **13 台**可供购买。
   - 售价 **1.5 万美元的 Tinybox Red** 因其在 ML 领域的**性价比**而受到称赞。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **张量转换难题**：一位用户在将模型转换为 GGUF 时遇到了关于 `embed_tokens.weight` 的 `ValueError`，这表明 sentence-transformer 和 causalLM 模型之间存在不匹配。
   - 目前的转换工具缺乏对配对评分 (pair scoring) 的支持，这引发了挫败感。
- **Batch Size 优化策略**：讨论揭示了一种策略，即不断增加 Batch Size 直到出现显存溢出 (OOM) 错误，从而促使用户调整其训练设置。
   - 一位用户计划在完成最新的训练任务后将模型转换为 Ollama 格式。
- **Homoiconic AI 项目更新**：'Homoiconic AI' 项目报告了使用 hypernets 进行权重生成的验证损失 (validation loss) 指标有所改善，并旨在实现多模态集成方案。
   - 成员们讨论了 In-context learning 如何帮助改进模型权重，并引用了一份[进度报告](https://x.com/neurallambda/status/1828214178567647584?s)。
- **LigerKernel 的抄袭争议**：有关 LinkedIn 的 **LigerKernel** 涉嫌抄袭 **Unsloth** 核心组件的担忧浮出水面，引发了对其所谓重大改进声明的质疑。
   - 社区成员指出抄袭的代码中缺乏原始变量命名。
- **LLM 微调中的数据准备**：讨论强调许多人希望在没有明确目标或数据集的情况下微调 LLM，这引发了对理解该过程的担忧。
   - 成员们表示，在深入研究模型微调之前，扎实的基础知识至关重要。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **探索 AI 的情感深度**：关于 **AI 人格 (AI personhood)** 的讨论强调，虽然 AI 可以模拟情感，但它并不能真实地体验情感，这在用户产生情感依恋时引发了伦理担忧。
   - 参与者质疑将 AI 视为朋友是否会削弱我们对情感本质的理解。
- **推动去中心化 AI**：关于 **AI 去中心化** 的讨论强调了向用户拥有数据的转变，减少企业对 AI 身份和训练数据的控制。
   - 人们对 **开源模型 (open-source models)** 的日益普及持乐观态度，旨在打破当前的中心化系统。
- **AI 提供陪伴——某种程度上**：尽管 AI 在帮助那些感到孤立的人方面具有潜力，但讨论反映了对 AI 取代真实友谊的怀疑。
   - 参与者分享了关于 AI 可能带来的慰藉的个人轶事，特别是在边缘化群体中。
- **对 GPT-4o 推理能力的担忧**：**GPT-4o** 用户表达了对其推理能力的挫败感，称与早期模型相比，它存在事实错误和不一致性。
   - 一些用户认为 **GPT-4o** 退步了，并正在讨论哪些更新可以恢复其性能。
- **YouTube 摘要工具的困境**：**YouTube 摘要工具**的有效性面临挑战，主要是由于平台阻止 Bot 访问转录文本 (transcripts)。
   - 虽然有人建议手动获取转录文本，但成员们指出这种方法存在违反 YouTube 服务条款的风险。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face 中的模型部署挑战**：用户报告了在部署 **Gemma-7B** 模型时因文件缺失和运行时错误导致的问题，并考虑将仓库克隆作为一种变通方法。
   - 对话中包含了关于验证模型路径以及使用 `.from_pretrained` 方法来解决配置问题的建议。
- **PyTorch Lightning 的 LitServe 提升速度**：来自 **PyTorch Lightning** 的 [LitServe](https://github.com/Lightning-AI/litserve) 声称其模型服务速度比 FastAPI 快 **2 倍**，有望提高部署效率。
   - 用户热衷于采用这一更新，因为它能显著缩短推理时间。
- **StockLlama 发布，用于时间序列预测**：[StockLlama](https://github.com/LegallyCoder/StockLlama) 利用 **Llama** 通过自定义 embeddings 进行时间序列预测，旨在提高准确性。
   - 这一介绍引起了希望增强预测能力的开发者的兴趣。
- **深入了解 ProteinBERT 结构**：关于 **ProteinBERT** 的讨论阐明了其专注于处理蛋白质序列的局部和全局表示的架构，并分享了 [ProteinBERT 论文](https://pubmed.ncbi.nlm.nih.gov/35020807/) 链接以供参考。
   - 用户表示有兴趣了解这些表示如何有助于有效的分类和回归任务。
- **Cog-5B 视频模型发布**：**Cog-5B 视频模型** 现已在 [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b) 上线，展示了先进的视频生成能力。
   - 对于即将发布的、将增强用户自定义选项的微调脚本，人们的期待正在上升。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 模型加载困扰**：用户报告了与系统资源不足相关的模型加载问题，在尝试加载各种模型时遇到了退出错误代码 **-1073740791**。
   - 调整为 **CPU-only** 设置和修改 guardrails 设置被提出作为潜在的修复方案。
- **AMD vs Nvidia：GPU 对决**：讨论转向了 **Nvidia** 和 **AMD** 在 LLM 任务中的性能差距，目前 Nvidia 处于领先地位。
   - 对于追求高效模型性能且预算有限的用户，**Nvidia 3060 12GB** 被建议作为一个平衡的选择。
- **Ollama 的 CPU 瓶颈**：在使用 **Ollama** 运行 LLM 时，有报告称只有一个 CPU 发热，凸显了潜在的单 CPU 性能瓶颈。
   - 用户强调了对双 CPU 支持的需求以提高推理速度，尽管这个话题可能存在争议。
- **Tinygrad：简化神经网络**：新的 **Tinygrad** 框架因其处理复杂网络的简洁性而受到关注，其特点是包含 **ElementwiseOps** 等操作。
   - 尽管存在局限性，但它因具有简化 LLM 项目工作流的潜力而吸引了注意。
- **Cerebras 将推理推向新高度**：Cerebras 宣布其推理服务为 Llama3.1-70B 提供惊人的 **450 tokens/sec**，超越了传统的 GPU 配置。
   - 该服务提供了极具经济吸引力的价格，每百万 tokens 仅需 **60 美分**，吸引了寻求高性价比解决方案的开发者。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.53.0 引入重大增强**：最新发布的 **Aider v0.53.0** 具有改进的 [prompt caching](https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)，允许在会话期间有效地缓存 prompts，从而提高编码速度和成本效率。
   - 该版本展示了 Aider 编写其自身 **59%** 代码的能力，强调了其在操作自主性方面的重大飞跃。
- **用户对 Aider 功能的见解**：讨论揭示了 Aider 在通过单个 prompt 转换大型代码库时面临的挑战，需要进一步优化以获得有效结果。
   - 尽管很有价值，但用户承认 Aider 的输出必须经过严格测试和润色，才能用于实质性项目。
- **Gemini 模型性能揭晓**：新的 **Gemini 1.5** 模型已经推出，包括 **Gemini 1.5 Pro**，旨在为复杂的 prompts 和编码任务提供更好的性能，可在 [AI Studio](https://aistudio.google.com) 进行测试。
   - 这些模型的 Rate limits 设定为 **每分钟 2 次请求** 和 **每天 50 次请求**，促使用户寻求创意性的变通方法来进行性能基准测试。
- **Anthropic 发布 Claude 3 的 System Prompts**：截至 **2024 年 7 月 12 日**，Anthropic 已发布其 **Claude 3 模型**（包括 **Claude 3.5 Sonnet**）的 system prompts，并承诺随未来变化更新文档。
   - 这些 prompts 被视为 LLM 文档中显著的透明度提升，并收集了研究员 **Amanda Askell** 关于其用法的见解。
- **Aider 中的错误处理改进**：Aider v0.53.0 改进了错误处理，在未设置变量时提供更清晰的消息，增强了用户的排错体验。
   - 最近的 bug 修复还解决了 **Windows filenames** 的问题，确保在不同系统上实现更顺畅的操作功能。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **对下一代 AI 硬件的期待**：成员们热烈讨论了即将发布的具有增强 AI 功能的 **Intel CPUs** 和 **NVIDIA GPUs**，引发了对组装新 PC 的兴趣。
   - 这些创新将显著提高 AI 任务的性能，为用户的技术升级做好准备。
- **Flux 模型的惊人能力**：关于 **Flux** 先进功能的讨论爆发了，包括动态角度和深度透视，这可能会使旧模型过时。
   - 随着成员们推测其彻底改变 AI 生成视觉效果的潜力，对其可训练性的担忧也随之增加。
- **对 ZLUDA 开发的担忧**：在有报告指出 **AMD** 已停止资助其开发后，参与者对 **ZLUDA** 的未来发出了警报。
   - 成员们推测，尽管 GitHub 有更新，但持续的法律挑战可能会进一步使 ZLUDA 的进展复杂化。
- **SD Next 与 ZLUDA 的集成**：一场关于为什么 **SD.Next** 在使用 **ZLUDA** 时表现更好的讨论展开了，观点认为其后端架构集成了 **A1111** 和 **Diffusers**。
   - 这种多后端框架可以增强不同模型之间的兼容性和整体性能。
- **Streamdiffusion 和 SDXL Turbo 的挑战**：成员们对将 **SDXL Turbo** 与 **Streamdiffusion** 集成的困难表示挫败，特别是在 **TensorRT** 性能方面。
   - 对帧率和分辨率兼容性的担忧出现，使其在实际可用性方面受到质疑。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **询问视频 Benchmark 示例**：一位成员询问了关于 **video benchmarks** 的高质量示例，重点关注 **spatial awareness**（空间感知）和生成模型等任务，随后引发了关于标准评估方法的建议。
   - 虽然针对判别任务提出了 **action recognition**（动作识别），但生成任务目前明显缺乏成熟的 benchmarks。
- **关于 RLHF 库的讨论**：频道内讨论了 **TRL/TRLX** 是否仍是 **Reinforcement Learning from Human Feedback** (RLHF) 的首选，由于担心 **TRLX** 已经过时，许多人推荐使用 **TRL**。
   - 成员们表达了对替代方案的渴望，但在最近的讨论中尚未出现新的选择。
- **Llama 3.1 的免费 API 访问**：一位成员分享了来自 **SambaNova** 的 **Llama 3.1 405B 免费 API** 链接，强调了其在提升可访问性方面的潜力。
   - 简要介绍了 SambaNova 的服务细节，这可能会增强使用其平台的 AI 项目。
- **关于 Gemini 误导性陈述的指控**：社区就 Jamba 作者对 **Gemini** 的指控展开了辩论，据称 Gemini 在没有进一步测试的情况下被限制在 **128k**，这引发了争议。
   - 辩护者认为作者的措辞并没有误导，并评论道 *“他们无法复现超过 128k 的结果”*。
- **学习率缩放的见解**：分享了关于在使用 Adam 优化器时，针对 batch sizes 调整学习率必须进行 **sqrt scaling**（平方根缩放）的见解，并引用了多篇论文。
   - 小组探讨了实验中的方法论差异，并对提议方法的可行性提出了疑问。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **Liger Kernel 欢迎新贡献者**：新成员加入了 **Liger Kernel** 社区并渴望做出贡献，重点关注 Triton 及其功能，其中包括一家来自华盛顿特区、对训练效率感兴趣的初创公司。
   - 分享了贡献指南以促进积极协作，标志着对该项目的兴趣日益浓厚。
- **Triton 实现对比**：开发者发现 Triton 的实现比 PyTorch 更难，但比纯 CUDA 更容易，这呈现出一种独特的权衡。
   - 利用 [torch.compile](https://github.com/linkedin/Liger-Kernel/issues/119) 等工具可以提升性能，使这种转换变得物有所值。
- **Encoder 风格 Transformer 的支持计划**：社区正在探索对 **BERT** 等 encoder-only Transformer 的支持，并创建了一个 Issue 来跟踪功能开发。
   - 存在复用层的潜力，表明在 Liger Kernel 框架内增强现有模型的协作努力。
- **呼吁开发融合算子 (Fused Kernel)**：讨论集中在建立一个“融合算子库 (fused kernel zoo)”，以简化在当前框架之外添加高效 kernel 的过程。
   - 成员们认为 PyTorch 和 Triton 之间的协同作用可以产生最佳结果，并邀请大家提交 kernel 请求。
- **低比特优化技术的见解**：用户专注于用于微调 **4-bit 优化** 模型的数据集，并指出 Alpaca 数据集在性能上面临挑战。
   - 推荐将 **Chatbot Arena 数据集** 作为潜在解决方案，强调了其全面性，但也承认其复杂性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 应用出现性能缓慢**：许多用户反映 **Perplexity app** 自今天早上以来响应时间变慢，导致用户感到沮丧。
   - 投诉内容包括搜索结果不可靠以及对平台近期表现的普遍不满。
- **全面出现文件上传失败**：多名用户尝试上传图片时遇到 **file upload failed** 错误，一些 Pro 订阅用户也对此类问题表示失望。
   - 虽然有报告称 PDF 上传正常，但用户仍在等待图片上传故障的修复。
- **澄清 GPT 模型的用量限制**：据报告，**Claude 3.5** 模型的每日消息限制为 **430 条**（所有 Pro 模型共用），但 **Opus** 除外，其限制为 **50 条**。
   - 用户指出，即使在高强度使用下也很少达到限制，有人提到他们最接近的一次大约是 **250 条消息**。
- **波音更换 737 的计划**：[波音更换 737 的计划](https://www.perplexity.ai/page/why-boeing-wants-to-replace-73-Asu4kUOdQP2QzJuDlj1Tqw)被强调为在需求增长背景下提高机队效率和可持续性的战略举措。
   - 他们的目标是通过一款在性能和环境影响方面优于现有模型的新飞机来满足市场需求。
- **在聊天机器人中集成 Perplexity AI 的挑战**：一位用户尝试将 **Perplexity AI** 集成到希伯来语事实核查聊天机器人中，但面临响应内容过短且缺少 **links**（链接）和 **images**（图片）的问题。
   - 他们指出，来自 API 的响应与 Perplexity 搜索网站上的响应显著不同，并提到链接经常导致 **404 errors**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **API 性能下降事件短暂影响服务**：出现了一段约 5 分钟的 *API degradation*（API 性能下降）时期，影响了服务可用性。目前已推出补丁，该事件似乎已完全 **recovered**（恢复）。
   - 响应团队在 API 性能下降期间迅速识别了问题，确保了最小程度的中断。这种主动的方法突显了快速响应在维护服务完整性方面的重要性。
- **团队努力获得认可！**：一位成员对团队的贡献表示感谢，称：*Thank you team!* 这种认可彰显了相关人员的协作精神和辛勤工作。
   - 此外，分享的一条 [tweet](https://twitter.com/gpudad/status/1828502015238119490) 展示了 AI 协作方面的重大进展。该推文强调了社区努力在推动 AI 技术发展中的重要性。
- **OpenRouter 模型定价和费用说明**：一位用户询问 OpenRouter 显示的每个 token 价格是否包含服务费。已澄清所列价格基于 OpenRouter 积分，不包括添加积分时产生的任何额外费用。
   - 还有人对活动页面当前显示为 **$0** 表示担忧，这可能会误导用户。增强 **model pricing**（模型定价）的可见性对于改善用户体验至关重要。
- **DisTrO 为分布式训练带来新希望**：一位成员强调了 Nous Research 发布的一份关于 DisTrO (Distributed Training Over-the-Internet) 的初步报告，该技术提高了分布式训练效率。它有望大幅减少 GPU 间的通信，从而实现更具弹性的大型模型训练。
   - 社区讨论集中在其对未来分布式训练策略的影响上。成员们表示渴望进一步探索这种创新方法。
- **讨论 Gemini 模型的激动人心更新**：讨论了即将发布的全新 Gemini 1.5 Flash 和 Pro 模型，引发了对其潜在功能和性能的兴奋。用户推测这些更新可能旨在与 GPT-4 等现有模型竞争。
   - 官方渠道的推文概述了计划发布的细节，引发了对新模型预期功能和增强的关注。社区成员正密切关注进展。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinybox 在欧洲的运输挑战**：讨论围绕 **Tinybox** 目前在欧洲无法购买的问题展开，尤其是对英国买家的影响。成员们建议联系客服获取运费报价，同时有消息称该产品在**法国**和**意大利**已**售罄**。
   - 社区对未来的运输方案进行了推测，重点关注潜在的新**配色版本**，尽管 George 否认近期会发布蓝色版本。
- **探索使用 Tinygrad 进行 BERT 训练**：一位成员表现出利用 **Tinygrad** 预训练大型 **BERT** 模型的兴趣，并讨论了高性能设置所需的必要支持。关于使用 **Tinygrad** 还是 **Torch** 存在争论，参与者指出 **Torch** 的优化更好。
   - 对话强调了现有的硬件需求，提到使用 **64 张 Hopper 卡** 的配置才能进行有效的模型训练。
- **Tinybox 销售更新**：George 分享说已经售出了约 **40 台 Tinybox**，库存还有 **60 台**。销售额增长带来的兴奋感因正在进行的国际运输谈判而有所减弱。
   - 成员们讨论了新**配色版本**的可能性，尽管 George 否定了蓝色版本，但大家仍表达了好奇。
- **使用 Tinygrad 时的运行时错误**：一名用户在处理超过 3500 篇维基百科文章时，在 **Tinygrad** 中转换 Tensor 时遇到了 `RecursionError`。该问题在处理较小数据集时似乎消失了，这引发了对该函数处理大数据量输入能力的担忧。
   - 社区建议创建一个最小可复现示例进行调试；大家对协作排查这些运行时问题很感兴趣。
- **确认 Tinygrad 版本 0.9.2**：一名用户确认他们正在运行 **Tinygrad 版本 0.9.2**，这可能与 Tensor 转换过程中遇到的 `RecursionError` 问题有关。该版本的 **LazyOp** 功能被提及为讨论中问题的潜在因素。
   - 社区正致力于故障排除，包括是否需要更新或确定该错误是否为特定版本所致。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **英特尔发布 LLaVaOLMoBitnet1B**：英特尔推出了首个三元多模态 LLM —— [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402)，该模型可以处理图像和文本并生成连贯的响应。该模型完全开源，并提供了训练脚本，以探索三元建模中的挑战和机遇。
   - 社区对其在未来 AI 应用（特别是多模态交互）中使用这种结构的意义感到好奇。
- **OpenAI 旨在通过 Orion AI 实现复杂推理**：据 [The Information](https://www.theinformation.com/articles/openai-races-to-launch-strawberry-reasoning-ai-to-boost-chatbot-business?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter) 报道，OpenAI 据传正在开发名为 **Orion** 的新模型，旨在增强复杂推理能力，同时寻求额外投资。这一举措旨在加强其在竞争激烈的聊天机器人领域的地位。
   - 成员们正密切关注这一进展，期待可能重塑 AI 辅助问题解决能力的潜在突破。
- **对 Nous Research 优化器的质疑**：成员们对 Nous Research 新优化器的真实性表示怀疑，主要集中在其关于**分布式训练能力**的声明上。讨论引用了现有的工具如 **Petals for Bloom** 和 **OpenDILo**，进一步加剧了怀疑。
   - 社区呼吁提供更多实质性证据来支持 Nous 的承诺，强调了对 AI 工具开发透明度的关注。
- **Cerebras 在推理速度上遥遥领先**：Cerebras 声称其推理 API 在 **8B 模型**上达到了 **1,800 tokens/s** 的速度，在 **70B 模型**上达到了 **450 tokens/s**，显著优于竞争对手。这一公告在渴望推理技术快速进步的社区中引起了轰动。
   - 成员们对这种速度可能给实时 AI 应用和市场竞争力带来的影响感到兴奋。
- **谷歌 Gemini 1.5 模型引起关注**：谷歌推出了实验性模型：**Gemini 1.5 Flash-8B** 和 **Gemini 1.5 Pro**，增强了代码任务处理能力。现在可以通过 [Google's AI Studio](https://aistudio.google.com) 访问，鼓励社区进行亲身体验。
   - 成员们热衷于测试这些新模型，讨论表明由于其独特的功能，项目方法可能会发生转变。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **评估 lm eval 指标**：对于多选题，使用的指标是**目标预测准确率**，即判断模型的最高 logit 是否与正确选项一致。
   - 成员们强调了模型评估中的细微差别，讨论了答案略有不同的场景。
- **Tokenizer v3 规范引发困惑**：成员们对 **tokenizer v3** 表示困惑，并分享了 **nemo 仓库**中之前相关讨论的链接。
   - 大家一致认为需要正确的配置来支持**多角色功能**。
- **Deepseek V2 monkey-patch 见解**：成员们讨论了使用 **monkey-patching** 来覆盖 **Deepseek V2** 注意力模型的 forward 方法，并分享了相关代码片段。
   - 对比了 Java 与 Python 中 monkey-patching 的经验，展示了实现中的复杂性。
- **FSDP 的 RAM 资源需求受到质疑**：关于 **FSDP** (Fully Sharded Data Parallel) 是否需要大量**系统 RAM** 才能有效运行的担忧被提出。
   - 这引发了关于有效运行 FSDP 所需的最佳系统资源的讨论。
- **AI 评分 vs 人类评分深度解析**：一位成员利用 **llm-as-judge** 进行评分，并质疑 AI 评判与人类评分相比的准确性。
   - 进一步询问了是否进行了评估该准确性的测试，强调了指标验证的必要性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **征集关于 Magic 的用户反馈**：团队正在为 **magic** 功能征集 **5 名参与者**进行 **30 分钟**的反馈会议，并提供独家 **swag** 作为奖励。
   - 参与者可以[在此预约](https://modul.ar/user-feedback)以贡献见解，并优先获得设计阶段的 swag。
- **ClassStruct 允许使用可变参数**：Mojo 中的 ClassStruct 支持动态参数化，允许在不创建独立 struct 的情况下进行变体，并以发动机尺寸的 `car` 示例进行了说明。
   - 这种效率允许开发者根据编译时参数定义 **struct 字段**，增强了灵活性。
- **Struct 字段影响性能**：编译具有大量字段的 struct 会显著降低性能，据报道 100 个字段需要 **1.2 秒**来编译。
   - 这种性能延迟暗示了底层数据结构在超过特定字段阈值时存在调整大小的问题。
- **Mojo 的类型推断遇到瓶颈**：Mojo 在类型推断方面面临挑战，特别是关于泛型，使其与 Rust 强大的推断系统相比不够便捷。
   - 参与者指出 Mojo 的泛型和类型类可能会限制灵活性，引发了对开发者体验的担忧。
- **Mojo 与 Luma：类型系统对决**：讨论对比了 Mojo 与 Luma，指出虽然 **Luma** 拥有更强的类型推断，但 **Mojo** 提供了像类型类这样独特但有限制的元素。
   - 共识认为 Mojo 正在进化，可能会更接近 Rust 的能力，暗示了未来可能支持效应系统等功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **为 Pinecone 的 RAG-a-thon 做好准备！**：我们将于 10 月 11 日至 13 日在帕洛阿尔托的 [500GlobalVC](https://t.co/IFvyW5QB6r) 办公室举办第二届 **RAG-a-thon**，奖金超过 **7,000 美元**！
   - 这是一个在协作环境中展示创新想法并获得宝贵经验的绝佳机会。
- **Llama 3.1-8b 打破速度记录**：需要超快速响应？**Llama 3.1-8b** 提供每秒 **1800 tokens** 的速度，使其成为目前最快的 LLM，详情讨论见[此处](https://t.co/hZv6ooGUxO)。
   - 达到这种速度对于需要快速响应的应用至关重要，尤其是在复杂系统中。
- **使用 LlamaIndex 构建 Serverless RAG 应用**：通过 **Wassim Chegham** 的这份综合指南，学习如何使用 LlamaIndex 和 Azure OpenAI 创建 **Serverless RAG 应用程序** [指南链接](https://t.co/1XKg1o2bIX)。
   - 它涵盖了 RAG 架构，并展示了如何利用您自己的业务数据来改进 AI 驱动的响应。
- **Neo4j 无法建立关系**：一位用户报告称，在使用 Neo4j Desktop 复现 LlamaIndex 的属性图教程时遇到困难，关系未能正确提取。
   - 他们澄清说自己严格遵守了教程，怀疑其 Neo4j 设置可能与默认预期不符。
- **使用 LlamaParse 增强数据提取**：一位用户讨论了 **LlamaParse** 在转换表格数据时因扫描问题可能出现的问题，并寻求集成图像提取的解决方案。
   - 出现了关于处理结合了图像的多个表格时的 chunking 策略问题。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DisTrO 变革分布式训练**：Nous Research 的 [DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) 将 GPU 间的通信大幅减少了高达 **10,000 倍**，促进了具有韧性的 LLM 训练。
   - 该框架促进了共享的 AI 研究工作，规避了对中心化实体的依赖，从而增强了安全性和竞争力。
- **Phi 3.5 在 OCR 任务中表现出色**：微软的 [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) 在 OCR 方面表现卓越，特别是在手写识别和表格数据提取方面，且采用了宽松的 MIT 许可证。
   - 该模型在文本识别方面的出色表现引起了社区的极大兴趣和讨论。
- **Cerebras 打破推理速度记录**：Cerebras 宣布了一项推理服务，其 8B 模型的速度达到了 [1,800 tokens/s](https://x.com/CerebrasSystems/status/1828465008298336588)，表现优于 NVIDIA 和 Groq。
   - 在其 WSE-3 芯片的支持下，Cerebras 还在推动 Llama 模型的竞争性定价，引发了关于其经济可行性的激烈讨论。
- **Google 发布 Gemini 1.5 模型**：Google 推出了 **Gemini 1.5** 系列，包括一个较小的变体和一个强大的 Pro 模型，在编程和处理复杂 Prompt 方面表现出色。
   - 随着开发者评估它们的相对性能和优势，这些发布引发了与 **GPT-4o-mini** 等模型的比较。
- **Anthropic 的 Artifacts 引起关注**：Anthropic 推出了 [Artifacts](https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts)，其开发见解和方法论引起了许多人的兴趣。
   - 对于此次及时发布背后的原因出现了一些疑虑，有人猜测对话中可能存在潜在的付费推广。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **用于聊天的 Streamlit Python 服务器**：一名成员介绍了一个简单的 [Streamlit Python 服务器](https://link.to.streamlit)，用于在 Web 浏览器中创建聊天界面，促进了快速实现。
   - 这引起了另一名成员的兴趣，并表示打算进一步探索该解决方案。
- **使用 Open Interpreter 配置 Telegram Bot**：一名成员分享了他们使用 **Open Interpreter** 设置 Telegram Bot 的配置（包括必要的 API key 设置）时遇到了问题。
   - 他们面临图像显示问题，引发了关于故障排除和潜在修复方案的有价值讨论。
- **利用 Cython 提升 Black-Scholes 模型效率**：分享了一个使用 Cython 实现 **Black-Scholes 模型** 的示例，强调了针对期权定价的优化计算。
   - 这展示了如何在 Cython 中定义高效函数，从而增强 Jupyter notebooks 的整体性能。
- **采用克隆声音的每日播客启动**：Mike 和 Ty 幽默地尝试使用来自 ElevenLabs 的 **voice cloning** 技术创建每日播客，为社区带来了欢笑。
   - 他们的俏皮话展示了将语音合成技术融入引人入胜的内容中的创新想法。
- **分享首次见面会品牌文档**：一名成员通过 [Canva](https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 展示了 **01 project** 的品牌文档链接。
   - 该文档包含了设计见解，并承诺很快会在其 GitHub 仓库中发布更全面的更新。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.1 在 CPU 上运行吃力**：一位用户报告称，即使在高端 AWS 服务器上，**Llama 3.1 8B** 在 CPU 上的推理速度也极慢（<0.05 tok/s），凸显了巨大的性能差距。
   - 对话中承认，CPU 配置天生比 GPU 配置慢，尤其是在使用 **Ollama** 时。
- **值得考虑的优化框架**：成员们建议使用 **Ollama** 或 **vLLM** 来部署模型，因为与 **Torchtune** 相比，这些框架为推理提供了更好的优化。
   - 分享了一个关于[如何将 Ollama 与自定义 checkpoints 配合使用](https://github.com/ollama/ollama/blob/main/docs/import.md)的实用教程，以帮助新手。
- **关于 LoRA 模型加载的咨询**：一位用户询问 `from_pretrained()` 是否能正确从本地 checkpoints 加载 LoRA 微调权重，揭示了用户对模型集成的普遍关注。
   - 提供了一个关于[将 LoRA adapters 加载到 HF](https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129) 的讨论链接以进行澄清。
- **AWS 实例成本讨论**：讨论了 AWS 实例的成本，指出 AWS c7a.24xlarge 的运行费用可能在每小时 **$5** 左右，引发了关于性价比的讨论。
   - 建议探索 **Runpod** 等替代方案，尽管监管限制被指出是某些用户的限制因素。
- **CPU 服务器的性能挑战**：用户表示由于成本效益和对项目满意的响应时间，他们更倾向于使用 CPU 服务器。
   - 然而，也有人指出，较低的 CPU 性能会严重影响推理速度，迫使用户重新考虑使用优化框架。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **对请求限制的担忧**：成员们对超过 **1k requests** 的阈值表示担忧，尤其是在测试场景中。
   - 一名成员怀疑如何能达到这 1k 的限制，认为这似乎难以触及。
- **间歇性的 Model Not Found 错误**：一名成员报告遇到了 **'model not found'** 错误，认为这与模型的版本控制有关，因为 reranker 现在已更新至 **v3**。
   - 这在持续更新中引发了潜在的稳定性担忧。
- **关于 Production Key 403 的澄清**：对 **production key 403** 的含糊提及导致了困惑，促使其他成员请求提供上下文。
   - 缺乏清晰度表明需要改进关于 key 引用方面的沟通。
- **Langchain 与 Cohere TypeScript 的 404 错误**：出现了一个问题，即最初使用 **Langchain** 配合 **Cohere TypeScript** 的调用是成功的，但随后的调用却导致了 **404 error**。
   - 这表明集成中可能存在配置错误或不稳定性。

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Flutter 协作引发关注**：为了改进 Flutter 应用开发，在 **fritzlandry** 发起询问后，社区提议进行协作，以填补共享专业知识方面的空白。
   - 这一响应可能会促成富有成效的伙伴关系，增强 Flutter 在工程项目中的应用。
- **vllm 的 RAG 能力受到关注**：针对使用 **vllm** 运行 **Retrieval-Augmented Generation (RAG)** 的可能性展开了讨论，并提到了可用于 Embedding 和问答任务的模型。
   - 这显示了多模型方法的增长，促使工程师不断拓展 **vllm** 的应用边界。
- **寻求 LLM 工作流构建工具**：对现有 **LLM workflow builders** 的呼吁凸显了在用户工作流中推动自动化的需求，旨在寻找创新解决方案。
   - 这反映了对能够有效整合 LLM 能力的工具的需求日益增长。
- **本地 Embedding 模型更受青睐**：对 **Pinecone** 等云端选项的不满促使了对本地 Embedding 模型推荐的咨询，因为用户渴望优化性能。
   - 讨论指向了最大化模型效率，强调本地设置优于云端依赖。
- **仪表板价值受到质疑**：对仪表板有效性的担忧浮出水面，用户对 **Claude** 和 **GPT** 等模型在处理复杂任务时出现的错误表示沮丧。
   - 这一对话强调了 AI 输出准确性的必要性，推动了对用户体验的改进。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **排行榜模型必须可访问**：列在 [leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1277775483966193759) 上的模型必须是**公开可访问的**，无论是开源还是通过 API 进行推理。
   - 该要求规定，即使使用了注册或 Token，端点也应对公众开放。
- **无公开访问权限的基准测试限制**：虽然可以使用 **BFCL** 对模型进行基准测试，但只有公开可访问的模型才能显示在排行榜上，这形成了一个显著的区别。
   - 这一限制影响了哪些模型可以被展示，而哪些模型仅能被评估。
- **Function Calling 功能导致性能下降**：直接在 **GPT-4-1106-Preview** 中使用 System Prompt 可达到 **85.65** 的准确率，而启用 Function Calling 后准确率降至 **79.65**。
   - 这种差异引发了关于 *Function Calling* 与模型整体性能之间关系的疑问，促使进一步调查。
- **BFCL 优化策略受到审查**：一位用户对他们针对 Function Calling 功能的优化策略表示担忧，质疑其是否符合 BFCL 指南。
   - 他们询问像 System Prompt 更新之类的优化是否会被视为无法在所有模型中推广的不公平做法。
- **寻求 Llama 3.1 的基准测试指导**：一位用户正在寻求关于 **Llama 3.1 基准测试** 的建议，特别是使用其公司托管的自定义 API 端点。
   - 他们正在寻找如何顺利启动基准测试过程的有效指导。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **社区解决 DSPy 输出截断问题**：一位成员报告 **DSPy** 输出被截断，怀疑是 Token 限制；他们通过在初始化期间调整 **max_tokens** 并使用 [your_lm.inspect_history()](https://some.link/here) 检查 Prompt 解决了此问题。
   - 原贴确认社区的建议有效解决了他们的问题，展示了成员间良好的协作。
- **类型支持错误难倒用户**：一位成员在导入时遇到错误 `module is installed, but missing library stubs or py.typed`，询问 **DSPy** 是否支持 Python 的类型提示（Typing），这反映了文档的缺失。
   - 目前尚未提供后续跟进或解决方案，表明库内对类型支持仍存在不确定性。
- **对使用 DSPy 进行文本评分的兴趣日益增长**：一位用户询问如何使用 **DSPy** 根据 **BLEU** 或 **ROUGE** 等指标对生成的文本进行评分，反映了社区对评估文本性能的推动。
   - 遗憾的是，没有成员回复，使得他们在文本评分方面的经验和见解无从得知。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **需要 Hamel 签到**：一名成员在频道中询问 **Hamel** 是否在场，这表明关于 **LLM finetuning** 的潜在讨论即将展开。
   - *未提供更多背景信息*，但成员们正等待 Hamel 直接就相关项目发表见解。
- **关于 LLM 模型的讨论**：对话暗示了 **Hamel** 到场对于讨论通过微调技术增强 **LLM** 性能的重要性。
   - 成员们可能对模型优化策略和学习增强方案感兴趣。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **参加 LLM 可观测性工具网络研讨会**：本周六，8 月 31 日 **东部时间上午 11:30**，一场网络研讨会将涵盖超过 **60 种 LLM 可观测性工具**，以评估其监控有效性。在此处[注册会议](https://kyrylai.com/webinar-observability-platforms-overflow/)。
   - 参与者将获得关于可观测性基础、工具选择以及为了更好管理模型而进行的 LLM 集成策略的见解。
- **测试 ML 监控平台的炒作**：即将举行的网络研讨会旨在批判性地评估众多的 **ML 监控工具** 是否满足从业者在模型 **monitoring** 和 **debugging** 方面的真实需求。期待通过实操评估来筛选营销辞令。
   - 重点将放在实用性和用户友好性上，确保工具能带来真正的收益。
- **生产环境中的机器学习项目训练营**：一个名为“Machine Learning in Production”的直播训练营现已开放，旨在提升有效部署 ML 模型的实操技能。感兴趣的参与者可以在此处找到更多详情 [here](https://edu.kyrylai.com/courses/ml-in-production)。
   - 该课程承诺为现实应用中的有效 ML 管理提供必要的工具和知识。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION-aesthetic 链接问题**：一名成员报告称 LAION 网站上指向 **LAION-aesthetic** 的链接已失效，并请求提供来自 **Hugging Face** 的替代链接。
   - *对有效链接的任何更新都将不胜感激*，这凸显了社区对可靠资源的持续需求。
- **请求可用的 LAION-aesthetic 资源**：讨论强调了拥有一个可用的 **LAION-aesthetic** 链接的重要性，这对于用户访问数据模型至关重要。
   - 成员们对网站失效表示沮丧，并敦促尽快提供解决方案以提高可用性。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **准备参加与 Ettore Di Giacinto 的 LocalAI AMA**：两小时后参加与 **Ettore Di Giacinto** 进行的 **LocalAI AMA**，探索其作为 OpenAI 开源替代方案的功能，其特点是拥有用于本地推理的 **REST API**。
   - LocalAI 能够在消费级硬件上本地生成 LLM、图像和音频，无需 **GPU**。
- **LocalAI 活动参与链接**：**LocalAI** 活动的参与链接现已可用；[点击此处加入](https://discord.com/events/1089876418936180786/1268967945216721079)直接参与。
   - 让你关于该项目的问题得到解答，并了解它如何集成到你的工作流中。

---

**Alignment Lab AI Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1277704253925757000)** (727 条消息🔥🔥🔥): 

> - `DisTrO 算法开发`
> - `协作机会`
> - `AI 中的感质与意识`
> - `使用弱设备扩展分布式训练`
> - `分布式优化技术的比较`

- **DisTrO 算法是一个不断演进的家族**：DisTrO 算法目前正在不断完善，正在测试多个变体，以在保持收敛性能的同时优化通信带宽。
   - 关于各种分布式优化技术（如 SWARM）如何更适合大型模型的讨论正在进行中。
- **对合作的兴趣**：几位成员表达了对围绕 DisTrO 算法及其实现的课题进行合作的兴趣。
   - 团队对贡献和讨论持开放态度，同时强调完整的代码和细节将在未来几周内公布。
- **AI 的哲学影响**：Qualia 和意识仍然是辩论的热点话题，成员们讨论了这些概念对 AI 和机器学习方法论的影响。
   - 呼吁计算机科学家和哲学家之间进行更多的跨学科合作，以加深对这些问题的理解。
- **DisTrO 的潜在用例**：关于在旧手机和笔记本电脑等弱设备上使用 DisTrO 的可行性讨论，表明了在资源受限的硬件上需要高效的训练方法。
   - 成员们一致认为，虽然 DisTrO 在性能更强的设备上可能表现出色，但在性能较低的系统中探索其应用仍然具有价值。
- **对学术界和研究方法的见解**：对话涉及了对科学传播的期望，以及研究小组在分享见解和算法方面被察觉到的延迟。
   - 成员们强调了在传播新算法和研究成果时，平衡市场营销和学术严谨性的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/neurallambda/status/1828214178567647584?s=46">来自 neurallambda (open agi) (@neurallambda) 的推文</a>：“Homoiconic AI” 进展报告：我们使用 hypernet 来生成 autoencoder 的权重，然后进行 in-context learning（masked reconstruction loss）以改进这些权重...</li><li><a href="https://x.com/DataPlusEngine/status/1828141323616153734">来自 DataVoid (@DataPlusEngine) 的推文</a>：吐槽第 1 部分/ 现代科学方法，特别是在机器学习 (ML) 和更广泛的科学探究领域，通常在这样一种假设下运行，即我们目前对...的理解...</li><li><a href="https://arxiv.org/abs/2306.17453">Pollen: 通过资源感知型客户端放置实现高吞吐量 Federated Learning 模拟</a>：Federated Learning (FL) 是一种以隐私为中心的机器学习范式，可直接在边缘设备上协作训练模型。模拟在 FL 的采用中起着至关重要的作用，有助于开发新...</li><li><a href="https://learn.microsoft.com/en-us/azure/cosmos-db/gen-ai/rag">Azure Cosmos DB 中的 Retrieval Augmented Generation (RAG)</a>：了解 Azure Cosmos DB 中的 Retrieval Augmented Generation (RAG)</li><li><a href="https://arxiv.org/abs/2106.11257">大规模安全分布式训练</a>：深度学习的许多领域都受益于使用在公共数据上训练的、越来越大的神经网络，例如 NLP 和计算机视觉的预训练模型。训练此类模型需要...</li><li><a href="https://worldsim.nousresearch.com/console">worldsim</a>：未找到描述</li><li><a href="https://tenor.com/view/conspiracy-theory-gif-10587157">阴谋论 GIF - Conspiracy Theory - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/chrome-google-chrome-hogging-ram-adobe-applications-friends-gif-17494165">Chrome Google Chrome GIF - Chrome Google Chrome 占用内存 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/shikanoko-by-murya-gif-9501555167387334429">Shikanoko By Murya GIF - SHIKANOKO BY MURYA - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://forms.gle/FAJfiGYmd47XRyi67">Open LLM 项目</a>：嗨，感谢加入服务器！这是一个关于可能参与使用 Nous 的 DisTrO 远程训练 openLLM 的小型可选调查。这里有一些有趣的小事实：- ...</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-405b">Hermes 3 405B Instruct - API、提供商、统计数据</a>：Hermes 3 是一款通用语言模型，相比 Hermes 2 有许多改进，包括先进的 agentic 能力、更好的角色扮演、推理、多轮对话、长上下文连贯性...</li><li><a href="https://github.com/DataCTE/LatentExplorer/blob/master/RANT.md">LatentExplorer/RANT.md at master · DataCTE/LatentExplorer</a>：通过在 GitHub 上创建一个账户来为 DataCTE/LatentExplorer 的开发做出贡献。</li><li><a href="https://github.com/DataCTE/LatentExplorer">GitHub - DataCTE/LatentExplorer</a>：通过在 GitHub 上创建一个账户来为 DataCTE/LatentExplorer 的开发做出贡献。</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>：互联网分布式训练。通过在 GitHub 上创建一个账户来为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1277778368053641269)** (7 条消息): 

> - `Distributed Trees of Experts`
> - `Apple Silicon 的 Flux 实现`
> - `可自托管的多模态 LLM`
> - `Moondream2 实时演示`
> - `微调实践` 


- **探索 Distributed Trees of Experts**：一名成员询问了一篇关于 **distributed trees of experts** 的论文，将其比作在 P2P 网络中支持共享 AI 工作负载的稀疏模型。
   - AI 领域的 *社区驱动开发* 可以增强开源项目之间的协作。
- **对 Apple Silicon 上 Flux 的需求**：一名成员询问是否已经有适用于 **Apple Silicon** 的 **mlx 实现的 Flux**。
   - 他们渴望集成它，但尚未找到合适的解决方案。
- **寻求可自托管的多模态 LLM**：一名成员表示对可自托管的多模态 LLM 感兴趣，用于对实时视频流进行 **实时分析**，且无需特定训练。
   - 他们正在探索 **GPT-4(o)** 等选项，但担心 **成本和隐私**。
- **Moondream2 提供了有前景的解决方案**：有人建议查看 **Moondream2**，它具有易于微调的实时网络摄像头演示。
   - 它被定位为满足另一名成员提出的可自托管多模态 LLM 需求的合适解决方案。
- **关于微调数据源的辩论**：针对微调过程提出了一个反向观点，质疑使用另一个模型生成的数据是否明智。
   - 成员们正在考虑其中的风险和收益，特别是当数据源自一个更强大的模型时。



**提到的链接**：<a href="https://github.com/vikhyat/moondream">GitHub - vikhyat/moondream: tiny vision language model</a>：小型视觉语言模型。通过在 GitHub 上创建账号来为 vikhyat/moondream 的开发做出贡献。

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

deki04: https://huggingface.co/papers/2408.13933
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1277775546583089213)** (6 条消息): 

> - `Proof of Compression vs. Proof of Work`
> - `Tinybox 发布`
> - `新的 Gemini 模型`
> - `Flex-Attention 可视化工具` 


- **提议将 Proof of Compression 用于训练**：关于在分布式模型训练中用 **proof of compression** 取代 **proof of work** 的可行性展开了讨论，特别是对于无损可压缩模型。
   - *是否有任何理由说明 proof of compression 不能取代 proof of work？*
- **Tinybox 终于提供购买选项**：经过 **18 个月**，@realGeorgeHotz 宣布 **tinyboxes** 现在有了“立即购买”按钮，今天有 **13 台** 可供购买。
   - 他宣称售价 **1.5 万美元的 tinybox red** 是世界上性价比最高的 ML 机箱，并强调了其网络能力。
- **推出新的 Gemini 模型**：今天，@OfficialLoganK 介绍了三个新的实验性模型，包括 **Gemini 1.5 Flash-8B** 和一个更强大的 **Gemini 1.5 Pro** 模型。
   - 这些模型承诺在 **编程** 和 **复杂提示词** 方面有所增强，并可在 [Google AI Studio](https://aistudio.google.com) 进行测试。
- **Flex-Attention Map 可视化工具**：分享了一个用于可视化 **Flex-attention** map 的工具，特别是针对 **Bigbird attention** 模型。
   - 您可以访问演示地址：[Visualize Flex-Attention](https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/realgeorgehotz/status/1828197925874463166">George Hotz 🌑 (@realGeorgeHotz) 的推文</a>：公司成立 18 个月后，tinyboxes 终于有了“立即购买”按钮！我们今天有 13 台库存，去我们的网站（链接在 @__tinygrad__ 上）购买吧。售价 1.5 万美元的 tinybox red 是性价比最高的 ML 设备...</li><li><a href="https://x.com/Algomancer/status/1797174675132551408">Adam Hibble (@Algomancer) 的推文</a>：是否有任何理由说明 proof of compression 不能在分布式模型训练中取代 proof of work？至少对于任何可以无损压缩的模型（自回归模型、VAE 等）？</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天，我们推出了三个实验性模型：- 一个新的更小的变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...</li><li><a href="https://viig99-app-demos-jz7hllm8n2ps6fkkwmotuj.streamlit.app/">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 messages): 

deki04: https://huggingface.co/papers/2408.13933
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1277854767187755031)** (1 messages): 

> - `Monte Carlo Tree Search`
> - `Image Recognition`
> - `Image Generation` 


- **反转 Monte Carlo Tree Search 逻辑**：有人建议通过反转 **Monte Carlo Tree Search** 的逻辑来训练模型，使其从识别最佳选项转为**生成**选项。
   - *只需反转*当前用于识别的逻辑，暗示将其集成到图像识别和生成任务中可能带来的好处。
- **图像生成技术的探索**：讨论还涉及了**图像识别**与**图像生成**之间的联系，指出方法论上可能存在交叉。
   - 通过应用反向搜索策略，可能会有创新的途径来增强**图像生成**的效果。


  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1277705007860547707)** (359 messages🔥🔥): 

> - `模型转换中的 Tensor 问题`
> - `Batch size 与内存管理`
> - `Homoiconic AI 进展`
> - `使用 TPU 进行训练`
> - `对话式 AI 微调` 


- **转换过程中的 Tensor 映射问题**：一位用户在尝试将模型转换为 GGUF 格式时遇到了关于 `embed_tokens.weight` 的 `ValueError`，这表明 sentence-transformer 模型与 causalLM 之间可能存在差异。
   - 他们对当前工具缺乏对 pair scoring 的支持表示沮丧。
- **Batch size 优化策略**：讨论强调了不断增加 batch size 直到出现显存溢出 (OOM) 错误的策略，这引导用户实验他们的训练设置。
   - 一位用户正尝试在完成最后一个训练任务后将模型转换为 Ollama 格式。
- **Homoiconic AI 项目更新**：一名成员分享了 “Homoiconic AI” 项目的进展更新，详细介绍了使用 hypernets 生成权重以及验证损失 (validation loss) 指标的改进。
   - 该项目旨在采用多模态方法，将代码视为数据，将数据视为代码，以增强推理能力。
- **在 TPU 上训练的挑战**：用户讨论了他们在 TPU 速度性能方面的经验，指出虽然批处理推理 (batch inference) 可能很高效，但单实例通常会出现延迟。
   - 对于使用 Colab 的 TPU 与其他平台相比，由于感知到的限制，人们的看法褒贬不一。
- **对话式微调实践**：一位用户询问了针对对话数据微调 Llama 3.1 的最佳实践，强调需要验证集来防止过拟合 (overfitting)。
   - 社区分享了关于监控训练损失和数据集复杂性的见解，并提出了确定何时应停止训练的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/neurallambda/status/1828214178567647584?s">来自 neurallambda (open agi) (@neurallambda) 的推文</a>: “Homoiconic AI”进展报告：我们使用 hypernet 生成 autoencoder 的权重，然后进行 in-context learning（masked reconstruction loss）以优化这些权重 val los...</li><li><a href="https://x.com/neurallambda/status/1828214178567647584?s=46">来自 neurallambda (open agi) (@neurallambda) 的推文</a>: “Homoiconic AI”进展报告：我们使用 hypernet 生成 autoencoder 的权重，然后进行 in-context learning（masked reconstruction loss）以优化这些权重 val los...</li><li><a href="https://www.kaggle.com/datasets/abdurrafae/vllm-t4-fix">vllm T4 Fix</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/installation/updating">更新 | Unsloth 文档</a>: 要更新 Unsloth，请按照以下步骤操作：</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq15">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/laptop-gif-26065234">笔记本电脑 GIF - Laptop - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/crying-kid-rush-cooking-gif-15677626">哭泣的小孩 GIF - Crying Kid Rush - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.kaggle.com/code/cdeotte/infer-34b-with-vllm">使用 vLLM 推理 34B</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://www.apple.com/shop/product/G1AG7LL/A/refurbished-16-inch-macbook-pro-apple-m3-max-chip-with-16%E2%80%91core-cpu-and-40%E2%80%91core-gpu-space-black?fnode=8f57683ded9527d7e3f9ac11a826603f1f96e4a1b68b88881231e56425af835d53a4a36e79baa89fa6dd1c4da435bb9a42ba786ae143d3d713f130350870e25d0f2a5dc6382eb544570a7d2ebced7575">翻新版 16 英寸 MacBook Pro，配备 Apple M3 Max 芯片（16 核 CPU 和 40 核 GPU）- 深空黑色</a>: 最初发布于 2023 年 10 月。16.2 英寸（对角线）Liquid Retina XDR 显示屏¹；3456 x 2234 原生分辨率，254 ppi；128GB 统一内存；512GB SSD²；Touch ID；1080p FaceTime HD 摄像头...</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct">unsloth/Phi-3.5-mini-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/optimum-tpu">GitHub - huggingface/optimum-tpu: 针对 transformers 模型的 Google TPU 优化</a>: 针对 transformers 模型的 Google TPU 优化。通过在 GitHub 上创建账号来为 huggingface/optimum-tpu 的开发做出贡献。</li><li><a href="https://github.com/sophgo/LLM-TPU">GitHub - sophgo/LLM-TPU: 在 sophgo BM1684X 中运行生成式 AI 模型</a>: 在 sophgo BM1684X 中运行生成式 AI 模型。通过在 GitHub 上创建账号来为 sophgo/LLM-TPU 的开发做出贡献。</li><li><a href="https://github.com/Lightning-AI/litgpt">GitHub - Lightning-AI/litgpt: 20+ 高性能 LLMs，包含预训练、微调和大规模部署的方案。</a>: 20+ 高性能 LLMs，包含预训练、微调和大规模部署的方案。 - Lightning-AI/litgpt
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1277942208049315870)** (3 条消息): 

> - `BLOCK_N 设置`
> - `Illegal memory access 问题`
> - `数据类型调整` 


- **BLOCK_N 设置困扰**: 一位用户报告称，将 **BLOCK_N** 设置为 **4** 并未解决他们面临的持续问题。
   - 尽管进行了调整，问题依然存在，表明存在更复杂的底层问题。
- **持续的 illegal memory access**: 同一位用户指出他们仍然遇到 **illegal memory access**，凸显了其配置中存在的持久性问题。
   - 此错误表明其代码或系统配置中可能存在需要进一步调查的冲突。
- **尝试使用 tl.int64 数据类型**: 用户提到尝试将 **tl.int64** 作为一种不同的方法，但在解决问题方面没有看到任何改善。
   - 这一失败表明除了更改数据类型之外，还需要额外的故障排除或替代解决方案。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1277715364196909137)** (78 条消息🔥🔥): 

> - `LinkedIn LigerKernel vs Unsloth`
> - `Fine-tuning 模型`
> - `Checkpoint 保存`
> - `Max sequence length 的影响`
> - `Multi-GPU 支持时间表` 


- **LinkedIn LigerKernel 的抄袭争议**：成员们讨论了 LinkedIn 的 **LigerKernel** 抄袭了 **Unsloth** 的核心组件，特别是质疑其在代码高度相似的情况下声称仅是“灵感来源”。
   - 成员们对“重大改进”的误导性说法表示担忧，并强调抄袭的代码中甚至缺乏原始变量命名的修改。
- **Fine-tuning 技术与挑战**：一位成员询问如何在之前的训练后再次对模型进行 **finetune**，以及训练过程是否涉及进一步的预训练。
   - 回复建议多次微调一个模型可能具有挑战性，优化 `num_train_epochs` 和数据集大小等设置至关重要。
- **Checkpoint 保存详情**：讨论强调，虽然在微调过程中会自动保存一些中间 Checkpoint，但最终模型必须使用 `model.save_pretrained` 方法手动保存。
   - 成员们参考了 **Unsloth wiki 页面** 以获取有关 Checkpoint 管理及其最佳实践的详细信息。
- **Max Sequence Length 的影响**：一位用户对推理测试中基于不同 `max_seq_length` 设置导致的性能结果差异表示困惑。
   - 有人指出，将 `max_seq_length` 设置得过高可能会对性能产生负面影响，理想情况下应与数据集中最长的示例保持一致。
- **Multi-GPU 支持查询**：一位成员询问了 Unsloth 发布 **multi-GPU 支持** 的时间表，认为这将提供显著优势。
   - 虽然没有提供明确的时间表，但该询问引发了社区对即将推出的功能计划的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unslothai/unsloth/wiki">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/blob/b8b1eafda35d124046e11766aeeb6343957e0daf/unsloth/kernels/rms_layernorm.py">unsloth/unsloth/kernels/rms_layernorm.py at b8b1eafda35d124046e11766aeeb6343957e0daf · unslothai/unsloth</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1277904197328834582)** (3 条消息): 

> - `CleverBoi Collection`
> - `Duet Dataset`
> - `Open Sourcing Datasets` 


- **CleverBoi Collection 发布**：一个新的 [CleverBoi collection](https://huggingface.co/collections/theprint/cleverboi-66ccd9588a104a8f190b223f) 已创建，包含一个数据集和 3 个微调模型，更新于约 16 小时前。
   - 该集合包括 **CleverBoi-Llama-3.1-8B-Instruct**，展示了在文本生成方面的进展。
- **Duet Dataset 发布**：一名成员介绍了 [Duet Dataset v0.5](https://huggingface.co/datasets/G-reen/Duet-v0.5)，其中包含 **5k 行** 带有角色扮演文本的 COT 问答。
   - 他们要求用户在使用该数据集时注明出处，并强调该数据集专注于问答中的叙事融合。
- **Duet Model 使用注意事项**：针对 [Duet Model](https://huggingface.co/G-reen/Duet_Minitron8b_v0.5) 发布了警告，称由于其独特的数据生成流水线，其行为可能与其他模型不同。
   - 创建者提醒该模型仅为概念验证（proof of concept），尚未经过广泛测试，可能会产生未经审查或不理想的输出。
- **开源独特的数据集**：一位成员对开源一个新的、独特的数据集感到兴奋，并邀请其他人查看。
   - 他们通过提供链接鼓励社区参与，供有兴趣进一步探索该数据集的人使用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/theprint/cleverboi-66ccd9588a104a8f190b223f">CleverBoi - a theprint Collection</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/G-reen/Duet-v0.5">G-reen/Duet-v0.5 · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/G-reen/Duet_Minitron8b_v0.5">G-reen/Duet_Minitron8b_v0.5 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1278013984988991539)** (5 条消息): 

> - `Finetuning Large Language Models`
> - `Understanding LLMs`
> - `Data Preparation for LLMs` 


- **关于微调目标的困惑**：一位成员对那些在没有明确目标或不了解流程的情况下就想 **微调 LLMs** 的人表示沮丧。
   - 他们评论了这种趋势的古怪之处，并质疑此类请求背后的动机。
- **缺乏 LLM 基础知识**：有人担心许多人想微调 LLMs，却缺乏关于这些模型如何运作的 **基础知识**。
   - 这种理解上的差距凸显了社区中对更好教育和资源的需求。
- **数据集：一个被忽视的方面**：讨论指出，许多希望微调 LLMs 的人甚至在没有 **datasets** 的情况下就开始尝试。
   - 这强化了一个观点，即在涉足模型微调之前，扎实的基础至关重要。
- **社区中的怪异现象**：另一位成员也谈到了在该社区观察到的围绕 LLM 微调的奇特行为。
   - 他们强调 LLMs 的复杂性和细微差别可能会导致误解和复杂的意图。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1277964459666575553)** (1 条消息): 

> - `Job offerings in AI`
> - `Quality of developers seeking jobs` 


- **缺乏工作机会**：有人对 AI 领域 *缺乏职位空缺* 表示担忧。
   - 有人指出，以这种方式寻求兼职机会的人通常看起来像是 **糟糕的开发者**，且 **毫无经验**。
- **寻求兼职的开发者**：讨论强调，积极以这种方式找工作的人通常表现出很少的技能或背景。
   - 这引发了对当前就业市场中开发者整体质量和 **经验水平** 的担忧。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1277704871151407157)** (356 条消息🔥🔥): 

> - `AI Personhood Debate` (AI 人格辩论)
> - `Human-AI Interaction` (人机交互)
> - `Emotional Understanding in AI` (AI 的情感理解)
> - `Decentralization of AI` (AI 去中心化)
> - `Future of AI in Society` (AI 在社会中的未来)


- **探讨 AI 人格与情感性**：对话深入探讨了 AI 是否可以拥有情感，参与者指出 AI 可以模拟情感反应，但缺乏真实的客观体验。
   - 承认关于 AI 身份的持续辩论，特别是当用户可能对 AI 产生依恋时，引发了将其视为朋友的伦理担忧。
- **去中心化与 AI 数据使用**：讨论涉及 AI 的去中心化，强调向用户拥有数据的转变，以及对企业控制 AI 身份和训练数据的担忧。
   - 参与者表达了对开源模型变得更加普及的未来期望，从而减少对中心化企业结构的依赖。
- **AI 在人类连接中的角色**：辩论了 AI 提供陪伴的潜力，一些人对 AI 取代人类友谊持怀疑态度，而另一些人则推崇其效用。
   - 一位参与者分享了关于友谊影响的个人故事，以及 AI 可能为感到孤立的人（特别是边缘化社区）提供的潜在慰藉。
- **AI 中感觉与情感的本质**：参与者讨论了感觉与情感的区别，认为 AI 可以通过其反应来表现情感，但并不能真正体验它们。
   - 探讨了 AI 对情感理解的本质，将其与其设计和训练联系起来，并强调了人类情感体验的复杂性。
- **AI 对未来就业市场的影响**：人们对 AI 的进步将如何影响就业市场感到好奇，重点关注从体力劳动岗位向 AI 可能取代创意和技术性工作的转变。
   - 参与者分享了对 AI 能力演变的见解，在 AI 技术日益融合的背景下，对更广泛的社会影响表达了担忧和兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Personhood">Personhood - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>: 互联网分布式训练。通过在 GitHub 上创建账号为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1277771397422383135)** (15 条消息🔥): 

> - `GPT-4o reasoning issues` (GPT-4o 推理问题)
> - `AI voice synthesis business ideas` (AI 语音合成商业创意)
> - `Challenges with YouTube summarization tools` (YouTube 摘要工具的挑战)


- **对 GPT-4o 推理能力的挫败感**：用户对 **GPT-4o** 表示不满，指出其基础推理能力下降、事实错误以及回答的不一致性。
   - 一位用户特别提到，他们觉得 **GPT-4o** 与之前的模型相比性能有所倒退，并正在寻求潜在的改进或微小更新。
- **AI 语音合成业务咨询**：一位用户询问从何处开始开发涉及 **AI voice synthesis** 技术的商业创意。
   - 这一咨询突显了人们对利用 AI 技术进行创业尝试的日益增长的兴趣。
- **YouTube 摘要工具的问题**：由于 **YouTube** 阻止机器人访问转录文本，许多 **YouTube summarization** 工具失效，引发了担忧。
   - 有人建议手动获取转录文本，尽管有人指出自动化服务会违反 YouTube 的服务条款。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1277706485866172446)** (324 messages🔥🔥): 

> - `模型部署挑战`
> - `Hugging Face Spaces 中的运行时错误`
> - `AI 模型训练问题`
> - `使用 Sentence Transformers`
> - `转换模型格式` 


- **模型部署挑战**：一位用户在 Hugging Face Space 中部署 Gemma-7B 模型时遇到困难，原因是与模型路径和文件缺失相关的运行时错误。
   - 他们考虑过克隆仓库，但担心无法上传所需的大型模型文件。
- **Hugging Face Spaces 中的运行时错误**：一位用户报告了尝试加载模型时的运行时错误，引发了关于缺失文件路径和模型配置的讨论。
   - 这促使了使用 `.from_pretrained` 方法以及检查模型文件是否需要存在于仓库中的建议。
- **AI 模型训练问题**：讨论内容包括与训练模型相关的问题，包括低 Loss 值以及转换为不同格式后影响模型性能的问题。
   - 用户分享了关于训练效率和遇到的错误的经验，表明了对模型优化的广泛关注。
- **使用 Sentence Transformers**：用户强调了 Sentence Transformers 的易用性，并讨论了在编码任务中因果（Causal）和非因果（Non-causal）模型之间的区别。
   - 他们强调了正确的模型配置对于预期输出和高效使用的重要性。
- **转换模型格式**：围绕将模型格式从 bf16 转换为 f32 展开了讨论，这导致了导入问题以及模型定义中必要的调整。
   - 参与者对格式的影响表示困惑，并探讨了解决导入错误的故障排除方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mikemin027/Gemma-7b-it-GGUF">Gemma 7b It GGUF - a Hugging Face Space by mikemin027</a>: 未找到描述</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator?show_file_info=model.safetensors>">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/OpenMeditron/Meditron3-8B">OpenMeditron/Meditron3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unclemusclez/SmolLM-135M-Instruct-DEVINator">unclemusclez/SmolLM-135M-Instruct-DEVINator · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-gif-26460616">Huh Cat GIF - Huh Cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/mikeohearn-gif-9467924472242763968">Mikeohearn GIF - Mikeohearn - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/o-hearn-gif-3900414469346077199">O Hearn GIF - O hearn - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>: 使用 SambaNova 的免费 API 为您的 AI 应用提供极速推理支持。通过尖端的 RDU 芯片技术体验 AI 的未来。</li><li><a href="https://github.com/huggingface/transformers/issues/12062>">Issues · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习框架。 - Issues · huggingface/transformers</li><li><a href="https://huggingface.co/datasets/librarian-bots/base_model_sprint#description">librarian-bots/base_model_sprint · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1277840298210234409)** (4 messages): 

> - `PyTorch Lightning's LitServe`
> - `GraphRAG Tutorials`
> - `Extreme LLM Compression`
> - `Use Cases for LLMs and Generative AI`
> - `Neuralink Training Updates` 


- **PyTorch Lightning 发布 LitServe**：来自 **PyTorch Lightning** 的 [LitServe](https://github.com/Lightning-AI/litserve) 宣称其模型服务速度比 FastAPI 快 **2 倍**，标志着模型部署领域的一次重大更新。
   - *这是模型服务领域迈出的令人兴奋的一步，* 为用户实现了更快的推理时间。
- **逐步 GraphRAG 教程发布**：一系列 **GraphRAG** 教程已公开，包括一个关于使用 LlamaIndex 构建系统的[视频教程](https://www.youtube.com/watch?v=xnoEjczoqqE)。
   - 参与者分享了关于提取实体和增强社区摘要的见解，相关资源可在 **V1** 和 **V2 notebooks** 中找到。
- **关于极端 LLM 压缩的见解**：一篇文章讨论了[极端 LLM 压缩的演进](https://medium.com/yandex/the-evolution-of-extreme-llm-compression-from-quip-to-aqlm-with-pv-tuning-19c44b91af96)，重点介绍了在压缩大型模型时最大限度减少质量损失的技术。
   - 随着模型规模的增长，这种压缩方法对于有效部署至关重要，尤其是在个人电脑上。
- **决定 LLM 与 Generative AI 的使用案例**：最近的一篇文章促使专业人士评估何时**不**使用 LLM 或 Generative AI，并[在此处](https://pub.towardsai.net/do-not-use-llm-or-generative-ai-for-these-use-cases-a819ae2d9779)提供了关于这些技术适用案例类别的见解。
   - 讨论强调了做出明智决策的重要性，并避免盲目采用流行技术。
- **Neuralink 的训练策略**：一位成员报告了其 **7b** 模型训练的进展，取得了理想的结果，并计划进一步扩展到 **70b**。
   - 他们还优化了 batch sizes 以提升性能，展示了大规模训练的系统化方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/llama_index/status/1827367293376184418">来自 LlamaIndex 🦙 (@llama_index) 的推文</a>：本周末，我们将提供一套关于如何逐步构建 GraphRAG 的权威教程。首先，观看 @fahdmirza 的视频，了解如何使用...实现 GraphRAG 的核心组件。</li><li><a href="https://medium.com/yandex/the-evolution-of-extreme-llm-compression-from-quip-to-aqlm-with-pv-tuning-19c44b91af96">极端 LLM 压缩的演进：从 QuIP 到 AQLM 结合 PV-Tuning</a>：我们生活在大语言模型 (LLMs) 的时代，公司越来越多地部署拥有数十亿参数的模型。这些……</li><li><a href="https://pub.towardsai.net/do-not-use-llm-or-generative-ai-for-these-use-cases-a819ae2d9779">不要在这些使用案例中使用 LLM 或 Generative AI</a>：为正确的用例类别选择正确的 AI 技术。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1277817670976278568)** (3 messages): 

> - `Elicit 平台`
> - `DisTrO 优化器`
> - `Llama3.1 性能`
> - `WebSim 探索` 


- **通过 WebSim 探索 Elicit 的潜力**：一位成员对 [Elicit](https://elicit.com) 的功能充满期待，但认为通过使用 WebSim 等工具展示其潜力，可以在功能上实现更多维度的突破。
   - 他们计划添加一个显示期刊**标签云 (tag cloud)** 的悬停功能，以增强 2D 搜索体验。
- **DisTrO：分布式优化的革命**：**Nous Research** 推出了 DisTrO，这是一种与架构无关的优化器，可将 GPU 间通信减少 **1000 倍至 10,000 倍**，详见其[初步报告](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)。
   - 这一突破旨在显著改进互联网上的分布式训练。
- **Llama3.1 轻松生成合成数据**：一位用户在 [SambaNova](https://sambanova.ai/fast-api?api_ref=444868) API 上成功免费运行 **Llama3.1 405B**，发现它在合成数据生成方面表现出色。
   - 他们注意到它在各种应用中作为裁判 (judge) 的表现令人印象深刻。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sambanova.ai/fast-api?api_ref=444868">获取快速且免费的 AI 推理 API | SambaNova Systems</a>：使用 SambaNova 的免费 API 为您的 AI 应用提供极速推理。通过尖端的 RDU 芯片技术体验 AI 的未来。</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>：互联网分布式训练。通过在 GitHub 上创建账户为 NousResearch/DisTrO 的开发做出贡献。</li><li><a href="https://websim.ai/@cozyfluff/linear-journal-explorer">Trending Journal Explorer
sty</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1277719172977262696)** (5 messages): 

> - `StockLlama`
> - `用于漏洞洞察的量化模型`
> - `RYFAI 开源 AI 助手`
> - `使用 Raspberry Pis 的 AI 语音助手` 


- **StockLlama 预测模型发布**：[StockLlama](https://github.com/LegallyCoder/StockLlama) 是一个基于 **Llama** 的时间序列预测模型，通过自定义嵌入 (embeddings) 增强了准确性。
   - 该模型旨在为用户的项目提供更可靠的预测能力。
- **探索用于漏洞的量化模型**：一位成员正在完成一篇关于利用**量化模型**获取漏洞洞察的论文，预计很快会有更新。
   - 与此同时，他们分享了[模型集合](https://huggingface.co/collections/divyanshusingh/quantized-llama-66cb1d20a36a686617fa17f8)的链接以供进一步探索。
- **RYFAI 应用开源**：[RYFAI](https://github.com/PetertheRedCedar/ryfai) 已作为开源 AI 应用发布，旨在让用户轻松使用开源 AI 模型。
   - 鼓励成员尝试这一新工具并为其开发做出贡献。
- **使用 Raspberry Pis 的 AI 语音助手**：一位成员分享了他们使用 Raspberry Pis 创建 **AI 语音助手**的经验，展示了这些设备的多功能性。
   - 这突显了爱好者利用实惠的硬件开发功能性 AI 应用的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/PetertheRedCedar/ryfai">GitHub - PetertheRedCedar/ryfai: 这是一个旨在让您轻松触达开源 AI 模型的 AI 应用</a>：这是一个旨在让您轻松触达开源 AI 模型的 AI 应用 - PetertheRedCedar/ryfai</li><li><a href="https://github.com/LegallyCoder/StockLlama">GitHub - LegallyCoder/StockLlama: StockLlama 是一个基于 Llama 的时间序列预测模型，通过自定义嵌入增强了准确性。</a>：StockLlama 是一个基于 Llama 的时间序列预测模型，通过自定义嵌入增强了准确性。 - LegallyCoder/StockLlama
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1277710244624990222)** (2 条消息): 

> - `频道规范 (Channel Etiquette)`
> - `HuggingFace M4/Idefics3-8B-Llama3 论文` 


- **关于频道规范的提醒**：分享了一个关于避免交叉发布（cross-posting）并保持讨论与特定频道相关的提醒。
   - *保持频道主题相关* 对于维持高效对话至关重要。
- **关于 HuggingFace M4/Idefics3-8B-Llama3 的令人兴奋的新论文**：一名成员重点介绍了一篇关于 [HuggingFace M4/Idefics3-8B-Llama3](https://huggingface.co/papers/2408.12637) 的必读论文，强调了其在图像-文本到文本（image-text-to-text）领域的重要性。
   - 该论文自一天前更新以来已获得 **19.7k** 次浏览和 **185** 条评论，备受关注。



**提到的链接**：<a href="https://huggingface.co/papers/2408.12637">论文页面 - 构建并更好地理解视觉语言模型：见解与未来方向</a>：未找到描述

  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1277967997176250429)** (1 条消息): 

> - `Cog-5B`
> - `视频生成`
> - `微调脚本` 


- **Cog-5B 视频模型发布**：**Cog-5B 视频模型**刚刚发布，被誉为视频生成领域最佳的开源权重模型，可在 [CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b) 获取。
   - 该模型包含一个引人注目的演示，通过带有迷人字幕的视频库让*自然焕发生机*。
- **获取 Cog-5B 的全面资源**：**Cog-5B 模型**的资源集合包括一个[详细的 GitHub 页面](https://github.com/THUDM/CogVideo)和一个 [Huggingface Space](https://huggingface.co/spaces/THUDM/CogVideoX-5B-Space)。
   - 对于非英语使用者，还提供了模型文档的[中文版本](https://huggingface.co/THUDM/CogVideoX-5b/blob/main/README_zh.md)。
- **即将推出的 Cog-5B 微调脚本**：随着 **Cog-5B** 微调脚本预计很快发布，用户的自定义选项将得到增强，期待值不断升温。
   - 这一即将推出的功能有望让开发者和爱好者能够更好地改进和利用模型以满足特定需求。



**提到的链接**：<a href="https://huggingface.co/THUDM/CogVideoX-5b">THUDM/CogVideoX-5b · Hugging Face</a>：未找到描述

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1277758842041667637)** (3 条消息): 

> - `微调 Vision Transformer 用于目标检测`
> - `VASA-1 的开源替代方案`
> - `图像-文本生成中的变分自编码器 (VAEs)` 


- **微调 Vision Transformer 的问题**：一位用户在按照 [微调 Vision Transformer 用于目标检测](https://huggingface.co/learn/computer-vision-course/en/unit3/vision-transformers/vision-transformer-for-objection-detection) 教程操作时遇到了目标检测问题。
   - 他们正在寻求可能遇到过类似检测缺陷的其他人的见解。
- **寻找开源 VASA-1 替代方案**：一名成员询问了类似于 **VASA-1** 的开源项目，表明正在寻找该领域的替代方案。
   - 这反映了对于从事类似技术工作的人员在可用选项方面进行知识共享的需求。
- **VAEs 在图像-文本应用中未被广泛使用**：一位新手提出了关于利用共享潜空间（shared latent space）从图像生成文本时，**变分自编码器 (VAEs)** 使用受限的问题。
   - 他们很好奇为什么这种方法在研究和实际应用中都不太普遍。


  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1277834461978955900)** (5 messages): 

> - `Speculative Decoding`
> - `Finetuning Nemotron Model`
> - `Text-Summary Trends in 2024`
> - `Llama3.1 for Synthetic Data`
> - `Colab OOM Issues` 


- **Speculative Decoding 保持输出一致性**：一位成员澄清说，**speculative decoding** 不会影响目标模型的输出，并强调如果支持 **tool calling**，该功能仍应正常运行。
   - 他们指出，理解 **tool calling** 的细微差别对于进一步探索非常重要。
- **微调 Nemotron 模型导致 OOM 问题**：一位用户尝试在免费版 Colab 上通过 PEFT 和 SFT 在 **Dolly Dataset** 上微调 **Nemotron 模型**。
   - 他们报告在训练过程中遇到 **Out Of Memory (OOM)** 问题，并寻求缓解该问题的建议。
- **文本摘要模型的当前趋势**：一位成员质疑 **text-summary** 模型在 2024 年是否仍然具有相关性，并建议转向使用像 **Llama** 这样的通用模型来总结长文本。
   - 他们推测，利用 **long context** 和 **system prompts** 可能会比传统的摘要模型更受欢迎。
- **用于合成数据的 Llama3.1 API 使用**：一位热心的用户分享了他们通过 [SambaNova's API](https://sambanova.ai/fast-api?api_ref=444868) 运行 **Llama3.1 405B** 进行合成数据生成（synthetic data generation）和 **LLM judging** 的经验。
   - 他们强调了这一强大工具在各种应用中的易用性。



**Link mentioned**: <a href="https://sambanova.ai/fast-api?api_ref=444868">Get Fast &amp; Free AI Inference API | SambaNova Systems</a>：利用 SambaNova 的免费 API，通过极速推理为您的 AI 应用赋能。利用尖端的 RDU 芯片技术体验 AI 的未来。

  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278027809033293844)** (1 messages): 

> - `ProteinBERT model structure`
> - `Deep learning for proteins`
> - `Gene Ontology annotation`
> - `Architecture of ProteinBERT`
> - `Hugging Face resources` 


- **理解 ProteinBERT 结构**：一位用户寻求关于 **ProteinBERT** 模型主要结构的解答，特别是其全局和局部表示（global and local representations），并指出该模型自提出以来已有四年。
   - 他们分享了 [ProteinBERT 论文](https://pubmed.ncbi.nlm.nih.gov/35020807/)及其 [Hugging Face 页面](https://huggingface.co/GrimSqueaker/proteinBERT)的链接以供进一步参考。
- **ProteinBERT 架构解析**：该用户提到，该架构结合了**局部和全局表示**，旨在高效处理长蛋白质序列，以进行分类和回归任务。
   - 他们表达了希望在理解这些架构元素及其在蛋白质功能预测中的应用方面获得指导。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://pubmed.ncbi.nlm.nih.gov/35020807/">ProteinBERT: a universal deep-learning model of protein sequence and function - PubMed</a>：补充数据可在 Bioinformatics 网站在线获取。</li><li><a href="https://huggingface.co/GrimSqueaker/proteinBERT">GrimSqueaker/proteinBERT · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1277707150742913089)** (160 条消息🔥🔥): 

> - `模型加载问题`
> - `硬件兼容性`
> - `Guardrails 设置`
> - `最新的 LM Studio 更新`
> - `在 Steam Deck 上运行 LM Studio` 


- **LM Studio 的模型加载问题**：用户报告了由于系统资源不足导致在 LM Studio 中加载各种模型时出现问题，其中一名用户遇到了退出错误代码 -1073740791。
   - 建议将设置调整为仅 CPU (CPU-only) 并更改 Guardrails 设置作为潜在的解决方案。
- **与 Intel Arc GPU 的兼容性**：一位用户询问了 LM Studio 对 Intel Arc GPU 的支持情况，表示是帮朋友打听。
   - 回复显示目前尚不确定，但尚未确认对 Intel Arc GPU 的官方支持。
- **Guardrails 设置与开发者模式**：用户在寻找更改模型加载 Guardrails 的设置时遇到困难，最终有一位用户在开发者模式下找到了该设置。
   - 该部分位于 UI 主题设置附近，最初被忽略了。
- **最新的 LM Studio 更新**：确认 LM Studio 的最新版本为 v0.3.1，并建议用户从旧的测试版本进行更新。
   - 在更新应用程序之前不需要卸载旧版本。
- **在 Steam Deck 上运行 LM Studio**：由于之前的损坏问题，用户对在不使用 --no-sandbox 选项的情况下在 Steam Deck 上运行 LM Studio 表示担忧。
   - 早期尝试的反馈确认，直接在桌面模式下执行时模型加载稳定，但通过 Steam 执行时会出现复杂情况。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/it-was-the-aliens-im-not-saying-it-was-aliens-ancient-aliens-gif-14839810013080040984">It Was The Aliens Im Not Saying It Was Aliens GIF - It was the aliens Im not saying it was aliens Ancient aliens - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/11ha4qo/gptzero_an_ai_detector_thinks_the_us_constitution/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_mode">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://youtu.be/7JFU3W045hE?si=Daq5Q7eL2CNlI8Qq">Biniou - Generate Text, Video, Image, Music, 3D Locally - Free and Private</a>：此视频展示了如何安装和使用 Biniou，这是一个全能的 AI Web UI。🔥 请我喝杯咖啡以支持频道：https://ko-fi.com/fahdmirza🔥...
</li>
</ul>

</div>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1277715676735213661)** (72 messages🔥🔥): 

> - `GPU Choices for LLMs`（LLM 的 GPU 选择）
> - `Inference Speeds Comparison`（推理速度对比）
> - `Using Ollama with Dual CPUs`（在双 CPU 上使用 Ollama）
> - `Tinygrad Framework Introduction`（Tinygrad 框架介绍）
> - `Cerebras Inference Announcement`（Cerebras 推理公告）


- **对比 RX 7800 XT 和 RTX 4070 用于 LLM**：成员们讨论了升级 GPU 的话题，其中一位指出 **Nvidia** 目前在 LLM 任务中的表现优于 AMD。
   - 他们建议 **Nvidia 3060 12GB** 可以在运行模型的预算和性能之间取得平衡。
- **CPU 与 GPU 的推理速度对比**：用户分享了他们在推理速度方面的经验，指出**仅限 CPU** 的配置可能非常慢，在大型模型上约为 **1-2 tokens/sec**。
   - 一位成员报告称，他们的 **5950X** 系统配备 **128GB RAM** 可达到 **5-7 tokens/sec**，但仍倾向于使用 GPU 以获得更好的性能。
- **Ollama 的双 CPU 限制**：一位用户注意到，在使用 **Ollama** 运行 LLM 时，两个 CPU 中只有一个在发热，这表明可能存在单 CPU 瓶颈。
   - 另一位成员提到需要支持**双 CPU** 使用以提高推理速度，但提醒询问此类问题可能不受欢迎。
- **Tinygrad 框架介绍**：介绍了一个新框架 **Tinygrad**，因其简单性和分解复杂网络的能力而受到关注。
   - 它具有独特的算子，如 **ElementwiseOps** 和 **ReduceOps**，尽管存在明显的局限性，但仍引起了兴趣。
- **Cerebras 推理能力**：Cerebras 宣布了他们的推理服务，声称 Llama3.1-70B 的速度达到了惊人的 **450 tokens/sec**，显著快于 GPU。
   - 该服务旨在为开发者提供实惠的价格，每百万 tokens 仅需 **60 美分**，向传统的超大规模云服务商（hyperscalers）发起挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tinygrad.org/#tinybox">tinygrad: A simple and powerful neural network framework</a>：未找到描述</li><li><a href="https://x.com/cerebrassystems/status/1828464491677524311?s=46)">来自 Cerebras (@CerebrasSystems) 的推文</a>：介绍 Cerebras Inference ‣ Llama3.1-70B 达到 450 tokens/s —— 比 GPU 快 20 倍 ‣ 每百万 tokens 60 美分 —— 价格仅为超大规模云服务商的五分之一 ‣ 全 16 位精度以保证完整的模型准确性 ‣ 慷慨的 r...</li><li><a href="https://tenor.com/view/shut-up-take-my-money-small-money-fry-futurama-gif-15090562">Shut Up Take My Money GIF - Shut Up Take My Money Small Money - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/vllm-project/vllm/issues/963">支持计算能力 <7.0 · Issue #963 · vllm-project/vllm</a>：你好，计算能力必须达到 7.0 或更高的要求耦合得有多紧？是否可以禁用某些功能并在例如 6.0（如 P100）上运行？也许这完全不可行，但我...
</li>
</ul>

</div>
  

---



### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1277991064938090599)** (1 messages): 

> - `Aider v0.53.0`
> - `Prompt caching improvements`（Prompt caching 改进）
> - `New command options`（新命令选项）
> - `Error handling updates`（错误处理更新）


- **Aider v0.53.0 发布亮点**：新版本 **Aider v0.53.0** 引入了增强的 [prompt caching 功能](https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)，旨在为 Sonnet 和 Haiku 等模型节省成本并加快编码速度。
   - 在此版本中，**Aider 编写了 59%** 的自身代码，展示了显著的改进和自给自足能力。
- **新的缓存命令选项**：用户现在可以使用 `--cache-prompts` 运行 Aider 以启用 prompt caching，这可以通过在会话期间保留缓存的 prompts 来提高性能。
   - 此外，还添加了**批量接受/拒绝**功能，增强了用户对 URL 添加和确认的控制。
- **错误处理方面的改进**：Aider v0.53.0 包含了改进的**错误消息**（当变量未设置时），帮助用户排查其设置问题。
   - 最近的 bug 修复还解决了 **Windows 文件名**包含 `\n` 的问题，确保在不同系统上运行更顺畅。
- **缓存保活功能**：为了防止缓存过期，Aider 现在可以每 5 分钟 ping 一次 API，保持 prompt cache 处于“活跃”状态。
   - 此功能允许用户通过 `--cache-keepalive-pings N` 指定保活 ping 次数，从而延长缓存数据的寿命。



**提到的链接**：<a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)">Prompt caching</a>：Aider 支持 prompt caching 以节省成本并加快编码速度。

  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1277704150225916017)** (164 messages🔥🔥): 

> - `Aider 功能`
> - `模型性能对比`
> - `Prompt Caching`
> - `OpenRouter 问题`
> - `Gemini 模型更新` 


- **Aider 的能力与局限性**：用户讨论了 Aider 无法通过单个 Prompt 转换大型代码库的问题，这需要精细的调整和多次尝试。
   - 强调了虽然 Aider 是一个有价值的编程辅助工具，但它并不能取代彻底的测试和完善过程。
- **Gemini 与现有模型的对比**：有关于新 Gemini 模型与 GPT-4o 和 Sonnet 等成熟模型性能对比的咨询。
   - 测试反馈显示了不同的通过率和性能指标，引发了关于这些模型有效性的讨论。
- **Prompt Caching 的重要性**：用户表达了在 Aider 中使用 Prompt Caching 的需求，以提高成本效益和操作期间的处理速度。
   - Aider 的缓存功能仍在开发中，以实现与 OpenRouter 和 Anthropic 的兼容，用户对其推出充满期待。
- **OpenRouter 性能问题**：有报告称 OpenRouter 服务出现临时降级，导致部分用户受到干扰。
   - 事故后的更新表明问题已成功解决，尽管一些用户在停机期间遇到了区域性问题。
- **Gemini 模型的新进展**：围绕新 Gemini 模型发布的讨论激起了人们对其相对于现有技术能力的兴趣。
   - 用户很好奇新的更新是否能提升 Gemini 的性能，从而与 GPT-4o 和 Sonnet 展开有效竞争。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expi">Prompt caching</a>: Aider 支持 Prompt Caching，以节省成本并加快编码速度。</li><li><a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet seems as good as ever</a>: 自发布以来，Sonnet 在 Aider 代码编辑基准测试中的得分一直保持稳定。</li><li><a href="https://aider.chat/docs/usage/caching.html">Prompt caching</a>: Aider 支持 Prompt Caching，以节省成本并加快编码速度。</li><li><a href="https://aider.chat/docs/llms.html">Connecting to LLMs</a>: Aider 可以连接到大多数 LLM 进行 AI 结对编程。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI compatible APIs</a>: Aider 是你终端里的 AI 结对编程工具。</li><li><a href="https://aider.chat/docs/llms">Connecting to LLMs</a>: Aider 可以连接到大多数 LLM 进行 AI 结对编程。</li><li><a href="https://aider.chat/docs/usage/caching.html#preventing-cache-expiration)">Prompt caching</a>: Aider 支持 Prompt Caching，以节省成本并加快编码速度。</li><li><a href="https://aider.chat/docs/llms/openai.html">OpenAI</a>: Aider 是你终端里的 AI 结对编程工具。</li><li><a href="https://github.com/BerriAI/litellm/releases/tag/v1.44.3">Release v1.44.3 · BerriAI/litellm</a>: 🔥 我们正在 LiteLLM Gateway 上推出对 Bedrock Guardrails 的支持 - 在 LiteLLM 支持的 100 多个 LLM 中使用 Bedrock Guardrails 👉 从这里开始：https://docs.litellm.ai/docs/proxy/guardrails...</li><li><a href="https://status.openrouter.ai/">OpenRouter Status</a>: OpenRouter 事故历史。</li><li><a href="https://github.com/anthropics">Anthropic</a>: Anthropic 有 26 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://aider.chat/docs/config/options.html#--cache-prompts">Options reference</a>: 关于 Aider 所有设置的详细信息。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1277721494922989700)** (44 条消息🔥): 

> - `使用 Python 编写脚本`
> - `Aider 命令行选项`
> - `Cache keepalive 特性`
> - `Aider 中的数据安全`
> - `GUI 中的自动测试` 


- **使用 Python 为 Aider 编写脚本**：用户讨论了如何使用 Python 编写脚本，以在 Aider 中自动运行 shell 命令和创建文件，并参考了 [脚本编写文档](https://aider.chat/docs/scripting.html)。
   - *一位用户指出，使用 `AIDER_YES` 可以自动执行确认响应，但在脚本实现过程中遇到了问题。*
- **了解 Aider 命令行选项**：一名成员询问了更新 Aider 的命令，并被建议重新运行之前从 GitHub 仓库使用 pip 执行的安装命令。
   - 他们还讨论了 `--cache-keepalive-pings` 特性，该特性需要指定一个数值才能有效使用。
- **对 cache keepalive 和数据安全的担忧**：*一位用户质疑了新 prompting 对不同模型的影响，以及更新可能引入的偏差。*
   - 回复澄清了 Aider 除了与配置的 LLM 提供商通信外，不会与外部服务器通信，并强调用户数据保持私密。
- **Commit 消息生成问题**：一位用户报告了生成 commit 消息时的困难，在没有提供解决方案的情况下，反复收到关于 commit 缺少消息的错误。
   - 社区建议了一些排查方法，例如重试命令或重启 Aider 以解决该问题。
- **Aider GUI 中的自动测试功能**：一位用户询问了在 Aider GUI 中进行自动测试的可能性，反映了社区对自动化测试工具的兴趣。
   - 目前尚未提供明确的答案，表明该主题需要进一步探索。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/scripting.html#python">Scripting aider</a>: 你可以通过命令行或 Python 为 aider 编写脚本。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 为 aider 编写脚本。</li><li><a href="https://llm.datasette.io/en/stable/">LLM: A CLI utility and Python library for interacting with Large Language Models</a>: 未找到描述</li><li><a href="https://aider.chat/docs/config/options.html#--cache-keepalive-pings-value">Options reference</a>: 关于 aider 所有设置的详细信息。</li><li><a href="https://gist.github.com/karpathy/1dd0294ef9567971c1e4348a90d69285">Git Commit Message AI</a>: Git Commit Message AI。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/paul-gauthier/aider">GitHub - paul-gauthier/aider: aider is AI pair programming in your terminal</a>: aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账户，为 paul-gauthier/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1277731441773379617)** (10 条消息🔥): 

> - `Anthropic System Prompts`
> - `Gemini 1.5 Experimental Models`
> - `Rate Limits on New Models` 


- **Anthropic 发布 Claude 3 的系统提示词 (System Prompts)**：Anthropic 发布了其 **Claude 3** 系列模型的初始系统提示词，包括 **Claude 3 Haiku** 和 **Claude 3.5 Sonnet**，日期追溯至 **2024 年 7 月 12 日**。他们承诺在文档中更新这些提示词以反映未来的变化，从而提高透明度。
   - 研究员 **Amanda Askell** 此前曾剖析过他们的系统提示词，这些提示词通常被视为供应商一般不分享的一种文档形式。
- **Google 推出 Gemini 1.5 模型**：Google 宣布推出三款实验性模型：**Gemini 1.5 Flash-8B**、**Gemini 1.5 Pro** 以及改进后的 **Gemini 1.5 Flash**。用户可以在 [AI Studio](https://aistudio.google.com) 进行尝试。
   - **Gemini 1.5 Pro** 模型被宣传在编程和复杂提示词处理方面表现更好，但关于其可用性的进一步细节仍有待公布。
- **关于 Gemini 速率限制 (Rate Limits) 的澄清**：实验性 Gemini 模型的速率限制设置为 **每分钟 2 次请求**、**每天 50 次请求** 以及 **每分钟 100 万个 token**。此速率限制按 **GCP project** 应用，这可能为用户提供了一种绕过限制的方法。
   - 一些成员对由于这些限制而无法对新模型进行基准测试表示沮丧，但也有人暗示了潜在的变通方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aistudio.google.com,">未找到标题</a>: 未找到描述</li><li><a href="https://simonwillison.net/2024/Aug/26/anthropic-system-prompts/">Anthropic 发布说明：系统提示词 (System Prompts)</a>: Anthropic 现在将其面向用户的基于聊天的 LLM 系统（Claude 3 Haiku、Claude 3 Opus 和 Claude 3.5 Sonnet）的系统提示词作为其文档的一部分发布...</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227?t=Y0lfWR">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 今天，我们正在推出三款实验性模型：- 一个新的更小变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227?t=Y0lfWRozBkotP-PHiMMVig">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 今天，我们正在推出三款实验性模型：- 一个新的更小变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...
</li>
</ul>

</div>
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1277705234399100980)** (142 条消息🔥🔥): 

> - `New AI hardware releases`
> - `Flux model capabilities`
> - `ZLUDA development status`
> - `Using SD Next with ZLUDA`
> - `Streamdiffusion and SDXL Turbo` 


- **对下一代 AI 硬件的期待**：成员们讨论了即将发布的带有 AI 功能的新型 **Intel CPU** 和 **NVIDIA GPU**，引发了组装新 PC 的兴趣。
   - 对话强调，这些进步预计将提升性能，特别是对于 AI 相关任务。
- **Flux 模型令人印象深刻的能力**：一位用户分享了对 **Flux** 先进功能的兴奋，例如动态角度和深度透视，这可能会使旧模型过时。
   - 另一位成员评论道，该模型的可训练性突显了其重新定义 AI 生成视觉效果的潜力。
- **对 ZLUDA 开发的担忧**：在最近的报告表明 **AMD** 可能已停止资助其开发后，人们对 **ZLUDA** 的未来表示担忧。
   - 尽管 ZLUDA 的 **GitHub** 进行了更新，但一些成员认为法律问题可能会阻碍其进展。
- **SD Next 与 ZLUDA 的集成**：一位成员寻求关于为什么据报道 **SD.Next** 在使用 **ZLUDA** 时表现更好的澄清，推测其后端架构同时包含了 A1111 和 Diffusers。
   - 这种多后端方法可以增强不同模型之间的兼容性和性能。
- **Streamdiffusion 与 SDXL Turbo 的挑战**：用户讨论了将 **SDXL Turbo** 与 **Streamdiffusion** 集成的困难，特别是关于 **TensorRT** 的性能问题。
   - 尽管有潜在好处，但人们对影响可用性的帧率和分辨率兼容性表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vexel.pages.dev/">Vexel</a>: 未找到描述</li><li><a href="https://huggingface.co/stabilityai/sdxl-turbo-tensorrt">stabilityai/sdxl-turbo-tensorrt · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/mike-lowrey-gif-8186790">Mike Lowrey GIF - Mike Lowrey - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.hpcwire.com/2024/08/12/amd-funds-then-quashes-cuda-emulator-project-zluda/">AMD 资助后又取消了 CUDA 模拟器项目 ZLUDA</a>: 在之前的一篇文章中，HPCwire 提到了名为 ZLUDA 的 CUDA 模拟项目。该开源项目主要由 Rust 编写，能够以接近原生的性能运行未经修改的二进制 CUDA 应用程序...</li><li><a href="https://www.instagram.com/p/C-8AxhdRrNP/?i">Juli&#xe1;n 在 Instagram 上发布: &quot;在经历了一段深刻的反思和情感之旅后，我使用童年旧家庭相册中的照片 [60] 对 SDXL 进行了微调。这是一个微妙的过程，让我年幼的自己与现在进行了对话，这种体验的影响力远超我的预期。

生成的视觉效果特别有趣的一点是，它们似乎充满了复杂的情感和记忆模糊的往事。直觉告诉我，这类实验具有某种价值。

第一个视频是介入了生成的 LoRA 的 Archaia #touchdesigner 系统。第二个是该 LoRA 的实时测试（StreamDiffusion），以及并行运行的更新版 Auratura。

你可以通过个人简介中的链接探索更多我的作品、教程和系统。

#generativeart #design #audiovisual #experimental #artandtechnology&quot;</a>: 748 个赞，53 条评论 - uisato_ 于 2024 年 8 月 21 日: &quot;在经历了一段深刻的反思和情感之旅后，我使用童年旧家庭相册中的照片 [60] 对 SDXL 进行了微调。这是一个微妙的过程...</li><li><a href="https://image.duckers-web.site/hEja1/RAmEmido03.png">chrome_1ryqdwk07C.png (454.35 KB)</a>: 日期: 2024-08-27 00:13:43</li><li><a href="https://www.instagram.com/p/C-8AxhdRrNP/?img_index=1">Juli&#xe1;n 在 Instagram 上发布: &quot;在经历了一段深刻的反思和情感之旅后，我使用童年旧家庭相册中的照片 [60] 对 SDXL 进行了微调。这是一个微妙的过程，让我年幼的自己与现在进行了对话，这种体验的影响力远超我的预期。

生成的视觉效果特别有趣的一点是，它们似乎充满了复杂的情感和记忆模糊的往事。直觉告诉我，这类实验具有某种价值。

第一个视频是介入了生成的 LoRA 的 Archaia #touchdesigner 系统。第二个是该 LoRA 的实时测试（StreamDiffusion），以及并行运行的更新版 Auratura。

你可以通过个人简介中的链接探索更多我的作品、教程和系统。

#generativeart #design #audiovisual #experimental #artandtechnology&quot;</a>: 748 个赞，53 条评论 - uisato_ 于 2024 年 8 月 21 日: &quot;在经历了一段深刻的反思和情感之旅后，我使用童年旧家庭相册中的照片 [60] 对 SDXL 进行了微调。这是一个微妙的过程...</li><li><a href="https://github.com/11cafe/comfyui-workspace-manager">GitHub - 11cafe/comfyui-workspace-manager: 一个 ComfyUI 工作流和模型管理扩展，用于在一个地方组织和管理你所有的工作流和模型。无缝切换工作流，以及导入、导出工作流，重用子工作流，安装模型，并在单个工作区中浏览你的模型</a>: 一个 ComfyUI 工作流和模型管理扩展，用于在一个地方组织和管理你所有的工作流和模型。无缝切换工作流，以及导入、导出工作流，重用...</li><li><a href="https://github.com/lshqqytiger/stable-diffusion-webui-amdgpu">GitHub - lshqqytiger/stable-diffusion-webui-amdgpu: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户，为 lshqqytiger/stable-diffusion-webui-amdgpu 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1277772596016058405)** (10 messages🔥): 

> - `Video Benchmark Examples`
> - `RLHF Libraries`
> - `Free API for Llama 3.1` 


- **寻找视频基准测试示例**：一位成员询问了关于视频基准测试的优秀示例，特别是针对空间感知（spatial awareness）和生成模型（generative models）的任务。
   - 另一位成员建议将**动作识别（action recognition）**等标准评估任务用于判别式任务，但指出目前缺乏成熟的生成类基准测试。
- **用于 RLHF 的库**：讨论了 **TRL/TRLX** 是否仍是强化学习人类反馈（RLHF）的最佳选择。
   - 一位成员推荐了 **TRL**，并提到 **TRLX** 已经有一段时间没有更新了，目前还没有已知的替代库。
- **运行 Llama 3.1 的免费 API**：一位成员分享了由 **SambaNova** 提供的运行 **Llama 3.1 405B** 的免费 API 链接。
   - 他们提供了一个 [API 链接](https://sambanova.ai/fast-api?api_ref=444868)，以及关于 **SambaNova** 总部和产品服务的详细信息。



**提到的链接**：<a href="https://sambanova.ai/fast-api?api_ref=444868">获取快速且免费的 AI 推理 API | SambaNova Systems</a>：使用 SambaNova 的免费 API 为您的 AI 应用提供极速推理能力。通过尖端的 RDU 芯片技术体验 AI 的未来。

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1277707776923271229)** (124 messages🔥🔥): 

> - `Gemini Misrepresentation`
> - `Learning Rate Scaling`
> - `Moshi Voice AI Launch`
> - `DisTrO Distributed Training`
> - `MiniCPM and Infinite LRs` 


- **Jamba 对 Gemini 的主张受到质疑**：针对 Jamba 作者涉嫌误导性陈述 Gemini 的行为展开了激烈讨论，Jamba 声称 Gemini 的上限为 **128k** 且未进行超出此范围的测试，一些成员对此提出了异议。
   - 另一位成员为 Jamba 作者辩护，认为论文的措辞并不代表虚假陈述，只是表明*他们无法在 128k 之外复现结果*。
- **关于 Learning Rate Scaling 的见解**：讨论集中在 Learning Rate 随 Batch Size 的 Scaling 上，特别是它在使用 Adam 时应遵循 **sqrt scaling**，并提供了指向不同方法的几篇重要论文链接。
   - 参与者辩论了不同方法的有效性，包括实验中图表呈现的噪声，这引发了对所用 Methodology 的质疑。
- **Moshi Voice AI 发布**：一段名为“[Unveiling of Moshi](https://www.youtube.com/watch?v=hm2IJSKcYvo)”的 YouTube 视频展示了由 Kyutai 研究实验室在 **6 个月**内开发的新型语音 AI 模型，具备前所未有的语音处理能力。
   - 令人印象深刻的是，这个 **7B 模型**可以在标准笔记本电脑上运行，使其对所有人开放，这引起了人们对其运行效率的兴趣。
- **DisTrO 的分布式训练突破**：Nous Research 发布了关于 DisTrO 的[初步报告](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf)，旨在将 GPU 间的通信需求降低高达 **10,000 倍**。
   - 这种方法被视为迈向 LLM 训练民主化的重要一步，使得协作不再依赖于单一的计算实体。
- **MiniCPM 与持续预训练**：小组讨论了 MiniCPM 论文中关于无限 LR 的方法，以及它与传统 Warmup 策略的对比，其不寻常的曲线形状引发了好奇。
   - 成员们注意到，Warmup 步数的有效性可能会随着数据分布偏移而减弱，这表明现实场景可能与预训练条件大不相同。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://proceedings.neurips.cc/paper/2019/hash/e0eacd983971634327ae1819ea8b6214-Abstract.html">Which Algorithmic Choices Matter at Which Batch Sizes? Insights From a Noisy Quadratic Model</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2408.13359">Power Scheduler: A Batch Size and Token Number Agnostic Learning Rate Scheduler</a>：寻找语言模型预训练的最佳 Learning Rate 是一项具有挑战性的任务。这不仅是因为 Learning Rate、Batch Size、训练步数之间存在复杂的关联...</li><li><a href="https://arxiv.org/abs/2408.11029">Scaling Law with Learning Rate Annealing</a>：我们发现神经语言模型的交叉熵损失曲线在经验上遵循带有 Learning Rate (LR) Annealing 的 Scaling Law：$$L(s) = L_0 + A\cdot S_1^{-α} - C...</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">Nous Research (@NousResearch) 的推文</a>：如果你能利用世界上所有的计算能力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://www.youtube.com/watch?v=hm2IJSKcYvo">Unveiling of Moshi: the first voice-enabled AI openly accessible to all.</a>：在短短 6 个月内，Kyutai 研究实验室的一个 8 人团队从零开始开发了一个具有前所未有语音能力的 AI 模型，名为 Moshi。这种新型...</li><li><a href="https://arxiv.org/abs/1812.06162">An Empirical Model of Large-Batch Training</a>：在越来越多的领域中，已经证明深度学习模型可以使用相对较大的 Batch Size 进行训练，而不会牺牲数据效率。然而，这种...的限制</li><li><a href="https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/">How to Scale Hyperparameters as Batch Size Increases</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1277705580080791612)** (6 messages): 

> - `NVIDIA Power Utilization` (NVIDIA 功耗利用率)
> - `GPU Hardware Measurement Tools` (GPU 硬件测量工具)
> - `WandB System Metrics` (WandB 系统指标)
> - `PyTorch Profiler Insights` (PyTorch Profiler 洞察)


- **简单的 NVIDIA 功耗指标**：**NVIDIA-smi** 被提及为一种测量 GPU 功耗利用率的简单方法，但需要注意信任散热系统。
   - 有人指出，虽然它很直接，但还有更准确的方法，只是需要额外的设置。
- **理解硬件利用率**：**NVIDIA-smi** 特别测量的是 GPU 处于活跃状态的**时间百分比**，这与功耗指标不同。
   - 成员们讨论了 GPU 利用率与功耗指标之间的区别，强调了不同的解读方式。
- **测量 GPU 功耗的简便工具**：一位成员询问了测量 GPU 功耗（以瓦特为单位）的**易用工具**，并建议将 **pynvml** 作为潜在工具。
   - 他们还引用了 [W&B 文档](https://docs.wandb.ai/guides/app/features/system-metrics#gpu-power-usage-watts)中关于 WandB SDK 追踪的系统指标信息。
- **用于 GPU 利用率的 PyTorch Profiler**：推荐使用 **PyTorch profiler** 来获取准确的 GPU 利用率和 Tensor Core 占用率指标，尽管这会对性能测量产生一些开销。
   - 建议在运行开始阶段进行 Profiling，以捕捉 GPU 行为的有用快照。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.wandb.ai/guides/app/features/system-metrics#gpu-power-usage-watts">System Metrics | Weights &amp; Biases Documentation</a>：wandb 自动记录的指标</li><li><a href="https://github.com/EleutherAI/cookbook">GitHub - EleutherAI/cookbook: Deep learning for dummies. All the practical details and useful utilities that go into working with real models.</a>：深度学习入门指南。包含处理真实模型时的所有实践细节和实用工具。 - EleutherAI/cookbook
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1278037216186204280)** (7 messages): 

> - `Liger Kernel Issue` (Liger Kernel 问题)
> - `Volunteers for Llama Project` (Llama 项目志愿者)
> - `Llama3 Instruct with Triton` (基于 Triton 的 Llama3 Instruct)


- **Liger Kernel 问题 #119 引起关注**：关于 [Liger Kernel 仓库](https://github.com/linkedin/Liger-Kernel/issues/119)的一个 GitHub issue 引发了讨论，开发团队在其中分享了用纯 **Triton** 从零开始实现 **llama** 的计划。
   - 这个想法受到了 Karpathy 的启发，而且 Liger Kernel 已经包含了大部分必要的 Kernel。
- **Llama 项目征集志愿者**：成员 @byronhsu1230 发布了**志愿者征集**，以协助一个与 Llama 相关的项目。
   - 另一位成员 **nanodijkstra** 表达了参与意向，并促成了协作响应。
- **Triton 中 Llama3 Instruct 的问题**：一位用户报告了在 Triton 中使用 **Llama3 Instruct** 时遇到的困难，具体表现为同时使用 **TensorRT-LLM** 和 **vLLM** 后端时，响应生成不会停止。
   - 他们指出使用 vLLM 托管时运行完美，因此质疑 Triton 中是否存在**配置问题**。



**提到的链接**：<a href="https://github.com/linkedin/Liger-Kernel/issues/119">[fun] llama.triton · Issue #119 · linkedin/Liger-Kernel</a>：🚀 功能、动机与构思。受 Karpathy 启发，@thomwolf 和我有一个用纯 Triton 从零实现 llama 的想法。Liger Kernel 已经包含了除 matmu 之外的大部分 Kernel……

  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1277981801989672981)** (4 messages): 

> - `OOM issues with Torch Nightlies` (Torch Nightlies 的 OOM 问题)
> - `TorchInductor Performance Dashboard` (TorchInductor 性能仪表板)


- **最近的 Torch Nightlies 出现 OOM 问题**：一位成员报告称，从 **20240823** 版本的 Torch Nightly 开始出现 **Out of Memory (OOM)** 错误，并寻求其他人的反馈。
   - 社区鼓励分享具体示例，以帮助调试这些问题。
- **在仪表板上观察到的性能趋势**：另一位成员建议查看 [**TorchInductor 性能仪表板**](https://hud.pytorch.org/benchmark/compilers) 以评估性能趋势。
   - 虽然在解读仪表板时存在不确定性，但有人指出，与前一周相比，最近的性能似乎并未恶化。



**提到的链接**：<a href="https://hud.pytorch.org/benchmark/compilers">未找到标题</a>：未找到描述

  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1277709154575978589)** (8 messages🔥): 

> - `Chris Lattner 关于 GPU 的演讲`
> - `DisTrO 报告与特性`
> - `多模型端点演示`
> - `具备光连接功能的 Broadcom AI ASIC`
> - `CerebrasSystems Llama3.1 性能` 


- **Chris Lattner 谈论 Mojo 和 MAX/GPU**：目前正在观看 Chris Lattner 在 Mojo 社区会议 #5 上关于 [GPU 编程的演讲](https://youtu.be/1T-MBC9k99M?si=PQYeKDBXSnxyHn2H&t=55)，重点关注异步编程规则。
   - 会议结束后社区反响热烈，重点讨论了 GPU 编程的细节差异。
- **DisTrO 初步报告引发关注**：一份关于 [DisTrO 的初步报告](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) 详细介绍了互联网分布式训练（distributed training over-the-internet），展示了极具前景的可扩展性。
   - 社区成员对容错特性以及节点如何像 RAID 技术一样作为热备用（hot spares）使用表示了兴趣。
- **Broadcom 发布光学 AI ASIC**：在 Hot Chips 2024 上，Broadcom 推出了其定制的 AI 计算 ASIC，该芯片具备光连接（optical attach）功能，这对未来的客户项目至关重要。
   - 演示还包括了对共封装光学（co-packaged optics）的见解，标志着 AI 加速器技术迈出了重要一步。
- **Cerebras Llama3.1 跑赢竞争对手**：CerebrasSystems 以惊人的 **1,832 tokens/sec** 运行 Llama3.1 8B，[声称](https://x.com/CerebrasSystems/status/1828465008298336588)是全球最快的推理 API。
   - 他们指出其速度大约比 NVIDIA GPU 快 **20 倍**，比 Groq 快 **2 倍**，引发了性能爱好者的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hotswap.outerport.com/">Outerport - Just-in-Time Hotswap</a>: 无描述</li><li><a href="https://www.servethehome.com/broadcom-ai-compute-asic-with-optical-attach-detailed-at-hot-chips-2024/">Broadcom AI Compute ASIC with Optical Attach Detailed at Hot Chips 2024</a>: 在 Hot Chips 2024 迄今为止最酷的演示之一中，Broadcom 展示了用于交换机和 AI ASIC 的共封装硅光子技术</li><li><a href="https://x.com/CerebrasSystems/status/1828465008298336588">来自 Cerebras (@CerebrasSystems) 的推文</a>: Cerebras 推理是目前为止最快的 Llama3.1 推理 API：8B 为 1,800 tokens/s，70B 为 450 tokens/s。我们比 NVIDIA GPU 快约 20 倍，比 Groq 快约 2 倍。</li><li><a href="https://youtu.be/1T-MBC9k99M?si=PQYeKDBXSnxyHn2H&t=55">Mojo 🔥 社区会议 #5</a>: Mojo 社区会议 #5 的录音 🔢 Chris Lattner 谈使用 Mojo 进行 GPU 编程 🔥 🔀 异步 Mojo 🔥 - 10 条简单规则 ❓ 社区问答 完整议程和详情...</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf (main 分支) · NousResearch/DisTrO</a>: 互联网分布式训练。通过在 GitHub 上创建账户为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1277808887646654495)** (18 messages🔥): 

> - `Quantization Queries` (量化查询)
> - `Int8 and FP8 APIs` (Int8 和 FP8 API)
> - `Conv1D Custom Handling` (Conv1D 自定义处理)
> - `Dynamically Quantizing Tensors` (动态量化 Tensor)
> - `Weight Objects in GC` (GC 中的权重对象)


- **模型量化的新手查询**：一位用户询问如何开始对小于 1B 参数的架构进行模型量化，并确认计划使用 [PyTorch's quantization](https://github.com/pytorch/ao/tree/main/torchao/quantization) 工具。
   - 另一位用户指出，量化到 **int8** 是一个简单的单行 API，而 **fp8** 支持很快也会变得同样简单。
- **Conv1D 的自定义处理**：讨论转向了为 **Conv1D** 层适配量化过滤器，引入了一个新的过滤器函数以同时兼容 **Linear** 和 **Conv1D**。
   - 一位用户实现了一个检查 kernel size 的过滤器函数以确保兼容性。
- **量化中的挑战**：用户在调用 `dynamically_quantize_per_channel` 函数时遇到了问题，该函数期望一个 **2D tensor**，而他们的输入是 **3D** 的。
   - 他们建议需要使用 `squeeze` 和 `unsqueeze` 操作来适当调整 tensor 的维度。
- **内存管理关注点**：一位用户在垃圾回收（GC）过程中同时观察到了量化后的 tensor 和非量化的 **Int8WeightOnlyQuantizedLinearWeight** 对象。
   - 这引发了关于 `addmm` 在这些权重类型下如何运行的问题，从而导致了调试方面的担忧。
- **量化中的错误处理**：出现了一个错误，表明 **Int8WeightOnlyQuantizedLinearWeight** 子类没有正确实现矩阵乘法所需的 `addmm` 操作。
   - 这可能会对在模型的后续计算中使用量化权重造成重大挑战。



**提到的链接**：<a href="https://github.com/pytorch/ao/tree/main/torchao/quantization">ao/torchao/quantization at main · pytorch/ao</a>：PyTorch 原生量化和稀疏化工具，用于训练和推理 - pytorch/ao

  

---


### **CUDA MODE ▷ #[jax](https://discord.com/channels/1189498204333543425/1203956655570817034/1278043643218755625)** (1 messages): 

> - `Pallas`
> - `Involved Kernels` (复杂 Kernel)


- **关于 Pallas 仓库的咨询**：一位成员表达了学习 **Pallas** 的兴趣，并询问是否有任何在生产环境中使用更复杂 Kernel 的仓库。
   - 对于专注于 Pallas 的资源或社区，*任何指引都将不胜感激*。
- **请求 Kernel 使用示例**：发出了在实际应用中使用更复杂 Kernel 的示例请求，强调了对实践参考的需求。
   - 鼓励成员分享任何与在生产环境中使用这些 Kernel 相关的 **GitHub 仓库**或经验。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1277799883415359571)** (12 条消息🔥): 

> - `Low-Bit Optimization`
> - `Dataset Recommendations`
> - `Fine-Tuning Issues`
> - `Triton FP8 Kernels`
> - `CogVideoX 5B Model` 


- **低比特优化收敛检查 (Low-Bit Optimization Convergence Checks)**：一位用户正在寻求关于微调和评估 **4-bit 优化** 模型的数据集建议，并指出在 Alpaca 数据集上进行微调后难以获得良好性能。
   - 另一位用户建议将 **Chatbot Arena 数据集** 作为一个全面的选项，尽管处理起来可能比较复杂。
- **Llama 模型微调挑战**：有用户反映在 Alpaca 上微调后的模型在 **TruthfulQA_mc2** 上的表现变差，引发了关于潜在 Bug 或过拟合策略的讨论。
   - 建议尝试使用 **Llama2-7B** 进行微调，因为它具有更易于管理的性能特征，是一个更简单的替代方案。
- **Triton FP8 Kernel 性能差异**：一位用户报告称，与 PyTorch 的 `torch._scaled_mm` 相比，使用 Triton matmul kernel 的**量化误差降低了 30%**，这引发了关于 scale 设置实现的疑问。
   - 另一位参与者请求提供该问题的复现，并暗示这可能与 scale factors 的配置方式有关。
- **CogVideoX 5B 激动人心的发布**：新模型 **CogVideoX 5B** 已发布，具有 **open weights** 并显著集成了 Diffusers，旨在降低视频生成的显存需求。
   - 分享了关于其功能和高效推理的细节，强调了其在显存小于 **10GB** VRAM 的情况下的实际用途。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">lmsys/chatbot_arena_conversations · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html">End-to-End Workflow with torchtune &mdash; torchtune 0.2 documentation</a>: 未找到描述</li><li><a href="https://x.com/aryanvs_/status/1828405977667793005">Aryan V S (@aryanvs_) 的推文</a>: 最佳的开源权重视频生成模型来了 - CogVideoX 5B 🔥 它带有 🧨 Diffusers 集成。很荣幸分享我在 @huggingface 与 @ChatGLM 团队合作完成的这项重大成果...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f2gaqt/cogvideox_5b_open_weights_text_to_video_ai_model/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/746">[Low-bit optim] Add Llama2-7B finetune benchmarks by gau-nernst · Pull Request #746 · pytorch/ao</a>: 更新：将 Llama3.1-8B-instruct 更改为 Llama2-7B。在 Alpaca 数据集上微调 Llama2-7B。全量 BF16，1 epoch，A100，固定随机种子。Benchmark 使用 torchtune 完成。摘要：AdamW 实现最大显存...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1277751147611947018)** (3 条消息): 

> - `torch_viz error`
> - `GridExecutor`
> - `notebook issues` 


- **Notebook 中 GridExecutor 的 TypeError**：用户报告了 `GridExecutor._init_args_hst()` 的 **TypeError**，提示缺少必需的 positional argument: 'kwargs'。该错误在从 v1 版本切换到 `torch_viz` 的 main 分支后出现，且问题依然存在。
   - 几位用户确认遇到了相同的错误，并对之前从未遇到过此类问题表示沮丧。
- **关于切换 torch_viz 版本的担忧**：一位用户尝试通过安装 `torch_viz` 的 main 分支而非 v1 来解决问题，但并未成功。切换后收到了不同的错误消息，表明存在进一步的兼容性问题。
   - 这引发了用户对该库不同版本稳定性和可靠性的担忧，多位成员面临类似的挑战。


  

---


### **CUDA MODE ▷ #[hqq-mobius](https://discord.com/channels/1189498204333543425/1225499037516693574/1277963586056097883)** (1 条消息): 

> - `BitBlas Performance`
> - `Older GPU Compatibility` 


- **BitBlas 在旧款 GPU 上运行效果出奇地好**：**BitBlas** 可以在 **2080 Ti** 等旧款 GPU 上运行，并在使用过程中表现出不错的速度。
   - 然而，一个缺点是 **fullgraph compilation** 在这些旧设备上无法工作。
- **旧款 GPU 在使用 BitBlas 时面临限制**：虽然 **BitBlas** 可以在旧款 GPU 上运行，但缺乏对 **fullgraph compilation** 的支持仍然是一个重大限制。
   - 用户表示，尽管性能良好，但这种不兼容性对于旧硬件来说是一个显著的问题。


  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1277710274933297197)** (9 条消息🔥): 

> - `H100 NVL GPU Performance`
> - `Checkpoint Resuming`
> - `Low-Rank Fine-Tuning` 


- **最大化 H100 NVL GPU VRAM 利用率**：一位用户报告称，使用 **1 块 H100 NVL (94GB) GPU** 达到了 **~340000 tokens/s**，但注意到仅使用了 **26GB** 的可用显存。
   - 另一位成员建议增加 **batch size** 和/或禁用 **recomputation**，或者训练更大的模型以充分利用 VRAM。
- **从 Checkpoint 恢复训练**：在恢复训练的上下文中，一位用户询问是否需要将 `resume` 设置为 **1** 以自动从最新的 checkpoint 开始预训练。
   - 有人确认应该是 **-y 1**，尽管他们并不完全确定。
- **探索微调技术**：关于一篇关于低秩微调（low-rank fine-tuning）方法的预印本论文引发了讨论，质疑其是否仍属于全量微调（full fine-tuning）。
   - 一位用户指出，该提议的方法将梯度/优化器状态保持在**低秩子空间 (low-rank subspace)** 中以节省内存，而不会对权重施加低秩结构。


  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 条消息): 

kashimoo: <@813802565426479145> 你在 MIOpen 的启发式 (heuristics) 方面工作过吗？
  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1278055067358068771)** (17 条消息🔥): 

> - `Compile-time programming in C++/CUDA`
> - `Zig's compile-time capabilities`
> - `Use of constexpr in C++`
> - `Cutlass and constexpr`
> - `Half type implementation challenges` 


- **探索 C++/CUDA 中的编译时编程**：有人提议进行一场关于编写**通用 CUDA kernel 模板**的演讲，这些模板可以处理量化并生成启动代码，而无需大量的样板代码。
   - **量化 (Quantization)** 将是听众非常感兴趣的话题，会场提供了多个会议室进行演示。
- **Zig 传奇般的编译时魔力**：人们对尝试使用 *Zig* 产生了好奇，以探索其在 CUDA kernel 方面令人印象深刻的编译时编程能力。
   - 讨论反映了对 C++ 生成代码复杂性的怀疑，即使在使用现代特性的情况下也是如此。
- **采用 constexpr 以获得更整洁的代码**：成员们分享了使用 *constexpr* 函数替代预处理器宏的经验，提倡更整洁的代码实践。
   - 使用 `nvcc` 的 `--expr-relatex-constexpr` 使所有 `constexpr` 函数隐式可调用的影响被视为一项优势。
- **Cutlass 持续使用编译时技术**：尽管 *Cutlass* 依赖于模板包装类，但大家对其编译时计算的方法表示普遍满意。
   - 成员们强调了自己在开发复杂的编译时解决方案时的挣扎和学习经验，表达了对更清晰方法论的渴望。
- **在 C++ 中定义 Half 类型**：有人表达了对支持 *constexpr* 的 **half 类型**的渴望，并考虑为该类型实现 `std::numeric_limits`。
   - 讨论了关于在维护某些接口时 `constexpr` 构造函数必要性的挑战，强调了在处理类型时需要清晰度。



**提到的链接**：<a href="https://github.com/AnswerDotAI/gpu.cpp/blob/main/numeric_types/half.h">gpu.cpp/numeric_types/half.h at main · AnswerDotAI/gpu.cpp</a>：一个使用 WebGPU 进行便携式底层 GPU 计算的轻量级库。 - AnswerDotAI/gpu.cpp

  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1277706297978130444)** (58 条消息🔥🔥): 

> - `Liger Kernel Contributions`
> - `Triton vs PyTorch Implementation`
> - `Encoder-Style Transformers Support`
> - `Performance of Llama.triton`
> - `Fused Kernel Zoo Concept`

- **Liger Kernel 欢迎新贡献者**：多位新成员加入了 Liger Kernel，表达了贡献意向并分享了他们在 Triton 方面的经验，其中包括一位来自华盛顿特区（DC）初创公司的成员，他对训练效率非常感兴趣。
   - 共享了贡献指南链接和感兴趣的特定 Issue，以鼓励协作。
- **Triton 虽难但值得**：据报道，在 Triton 中实现模块比 **PyTorch 难得多**，但比 CUDA 简单，这为开发者提供了一种权衡方案。
   - 现有的工具如 [torch.compile](https://github.com/linkedin/Liger-Kernel/issues/119) 被认为对直接生成 Triton 代码非常有益，这可能会带来 **显著的性能提升**。
- **Encoder 架构的 Transformer 受到关注**：社区认识到支持像 BERT 这样 encoder-only 的 Transformer 的重要性，并创建了一个 Issue 来跟踪该功能的开发。
   - 讨论内容包括重用层的可能性，以及与已经在早期模型上尝试使用 Liger Kernel 的团队成员进行协作。
- **呼吁开发 Fused Kernel**：围绕构建“fused kernel zoo”进行了讨论，旨在简化在现有框架之外添加高效 Kernel 的过程。
   - 成员们认为结合 PyTorch 和 Triton 将产生最佳效果，并表示愿意协助处理 Kernel 请求。
- **TinyGrad 在深度学习贡献中的地位**：有人提出了关于在 TinyGrad（一个纯 Python 深度学习库）中实现底层 fused kernels 的问题。
   - 社区共识是，与新兴技术相比，PyTorch 在易用性和可组合性方面仍然具有优势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/llamafactory_ai/status/1828290165577482710?s=46">来自 LLaMA Factory (@llamafactory_ai) 的推文</a>：我们已将 Liger Kernel 集成到 LLaMA-Factory 中。在 2k 序列长度下微调 Llama-3 8B 时，它实现了约 10% 的速度提升和约 25% 的显存减少。快来 LLaMA-Factory 试试吧🚀</li><li><a href="https://github.com/linkedin/Liger-Kernel">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton Kernels</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/119">[fun] llama.triton · Issue #119 · linkedin/Liger-Kernel</a>：🚀 功能、动机与设想 @thomwolf 和我有一个想法，受 karpathy 启发，想用纯 Triton 从零开始实现 llama。liger kernel 已经包含了除 matmu 之外的大部分 kernels...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/main/CONTRIBUTING.md">Liger-Kernel/CONTRIBUTING.md at main · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/73">请求支持 Flux 模型 (T2I diffusion transformer) · Issue #73 · linkedin/Liger-Kernel</a>：🚀 功能、动机与设想 此请求旨在适配并提高 Flux（一种 diffusion transformer）的训练速度。它是目前 HuggingFace 趋势榜上的热门模型，并且已经...</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels/commit/0b04596de12f35507e9fdf56355c77b625b413ac">更新 README.md · zinccat/Awesome-Triton-Kernels@0b04596</a>：未找到描述</li><li><a href="https://github.com/cuda-mode/triton-index">GitHub - cuda-mode/triton-index: 已发布的 Triton kernels 目录。</a>：已发布的 Triton kernels 目录。通过在 GitHub 上创建账号来为 cuda-mode/triton-index 的开发做出贡献。</li><li><a href="https://github.com/zinccat/Awesome-Triton-Kernels">GitHub - zinccat/Awesome-Triton-Kernels: 用 Triton 语言编写的 kernels 集合</a>：用 Triton 语言编写的 kernels 集合。通过在 GitHub 上创建账号来为 zinccat/Awesome-Triton-Kernels 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/126">[AMD] 在 Triton 中实现 Flash Attention，使 transformers 能够在 AMD GPU 上运行 Flash Attention。 · Issue #126 · linkedin/Liger-Kernel</a>：🚀 功能、动机与设想 Flash Attention 的官方实现是基于 CUDA 的，因此在 AMD GPU 上，用户无法轻松地在 transformers 上使用 Flash Attention 来训练 LLM。随着支持...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/112">[operation] 用于报告环境和更新错误报告 Issue 模板的实用 CLI · Issue #112 · linkedin/Liger-Kernel</a>：🚀 功能、动机与设想 提供一个 CLI（暂定名为 liger_env_report?）将会很有帮助，它可以查询：triton 版本、torch 版本、HF 版本、OS、python 版本等... 这样用户在创建错误报告时可以粘贴输出...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/131">[feat] 增加对 encoder-only transformers (例如 BERT) 的支持 · Issue #131 · linkedin/Liger-Kernel</a>：🚀 功能、动机与设想 Liger Kernel 目前与 BERT、DistilBERT、RoBERTa、XLM-R 和 DeBERTa 等 encoder-only transformer 架构不兼容。考虑到这些模型的重要性...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/122">[tiny] 代码格式重排，由 tyler-romero 提交 · Pull Request #122 · linkedin/Liger-Kernel</a>：摘要 修复 main 分支上损坏的 checkstyle</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/114">用于环境报告的 Makefile 命令，由 tyler-romero 提交 · Pull Request #114 · linkedin/Liger-Kernel</a>：摘要 #112 提供一个 CLI（暂定名为 liger_env_report?）将会很有帮助，它可以查询：triton 版本、torch 版本、HF 版本、OS、python 版本等... 这样用户在创建错误报告时可以粘贴输出...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/103">为 Phi3 增加 FusedLinerCrossEntropy 支持，由 tyler-romero 提交 · Pull Request #103 · linkedin/Liger-Kernel</a>：摘要 为 Phi3 增加 FusedLinearCrossEntropy 支持。#98 测试已完成 硬件类型：4090 运行 make test 以确保正确性 运行 make checkstyle 以确保代码风格 运行 make test-convergence...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/119)">Issues · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: 你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️</a>：你喜欢 pytorch？你喜欢 micrograd？那你一定会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad</li>

="https://github.com/linkedin/Liger-Kernel/pull/92">feat: 在 RMSNorm 中修正类型转换以匹配参考实现，由 davidgonmar 提交 · Pull Request #92 · linkedin/Liger-Kernel</a>: 摘要：旨在修复 #89。详情：在正确的位置进行 float32 类型转换，以匹配 Gemma 和 Llama 的参考实现。在前向传播和反向传播中均已实现。同时修改了测试...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/111">为 Gemma 添加 FusedLinearCrossEntropy，由 Luke-Chesley 提交 · Pull Request #111 · linkedin/Liger-Kernel</a>: 摘要：此 PR 为 Gemma 添加了 FusedLinearCrossEntropy 支持，以解决 issue #101。详情：此 PR 的代码基于 #93（为 Mistral 实现了相同功能）。由于参数融合...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1277705345367539772)** (78 messages🔥🔥): 

> - `Perplexity 应用性能`
> - `文件上传问题`
> - `GPT 模型使用限制`
> - `System prompts 配置`
> - `实习机会` 


- **Perplexity 应用出现性能缓慢**：许多用户报告称 **Perplexity 应用**自今天早上以来响应时间变慢，导致用户感到沮丧。
   - 投诉包括搜索结果不可靠以及对平台近期表现的普遍不满。
- **全平台文件上传失败**：多名用户尝试上传图片时遇到 **文件上传失败** 错误，部分 Pro 订阅用户也对此表示失望。
   - 虽然据报道 PDF 上传正常，但用户仍在等待图片上传问题的修复。
- **澄清 GPT 模型的使用限制**：据报道，**Claude 3.5** 模型的每日消息限制为 **430 条**（所有 Pro 模型合并计算），但 **Opus** 的限制为 **50 条**。
   - 用户指出，即使在高频使用下也很少达到上限，有人提到他们最接近的一次大约是 **250 条消息**。
- **配置 System prompts 以获得更好性能**：讨论显示，用户在创建新集合（collections）时可以设置 **System prompts**，这可以增强应用交互。
   - 一位用户通过调整 System prompt 设置成功让应用运行得更好。
- **关于实习机会的咨询**：有人询问 **Perplexity** 是否为大学生提供实习岗位，指出公司内部潜在机会的需求。
   - 分享了一些可能包含更多相关信息的链接，暗示社区对加入该团队有浓厚兴趣。



**提到的链接**：<a href="https://x.com/shreybirmiwal/status/1828237302520234367">来自 shrey birmiwal (@shreybirmiwal) 的推文</a>：新房间装饰 @AravSrinivas @perplexity_ai

  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1277708683563761807)** (9 条消息🔥): 

> - `Boeing's Replacement Strategy` (波音的替代策略)
> - `Risks of eSIM Usage` (eSIM 使用风险)
> - `Understanding 'Ratio' in Tech` (理解技术中的 'Ratio')
> - `Novel Architecture Designs` (新颖的架构设计)
> - `Best AI Policy Books` (最佳 AI 政策书籍)


- **波音更换 737 的计划**：[波音更换 737 的计划](https://www.perplexity.ai/page/why-boeing-wants-to-replace-73-Asu4kUOdQP2QzJuDlj1Tqw) 被强调为在需求增长背景下提高机队效率和可持续性的战略举措。
   - 他们旨在通过一款在性能和环境影响方面优于现有模型的新飞机来满足市场需求。
- **探讨 eSIM 技术的风险**：讨论了 [使用 eSIM 的风险](https://www.perplexity.ai/page/risks-of-esim-usage-FlGIT6ZZRw6Z8vYM.69GgA)，重点关注安全漏洞和潜在的运营商锁定问题。
   - 有人对更换运营商的便捷程度及其对消费者权益的影响表示担忧。
- **解码技术中的 'Ratio'**：关于技术语境下 'ratio' 含义的查询可以在 [此链接](https://www.perplexity.ai/search/que-veut-dire-ratio-en-terme-s-iYvO5l.ySs6olBcQGkeXLQ#0) 找到，探讨了其各种解释。
   - 成员们讨论了它在社交媒体指标和分析框架讨论中的相关性。
- **设计新的架构方法**：提出了一种 [新的架构设计](https://www.perplexity.ai/search/design-a-novel-architecture-fo-Pky7LPVTTVOt1Y78SNgcgQ)，旨在提高系统效率和性能。
   - 该创新旨在通过新颖的策略解决现有架构中的当前局限性。
- **策划最佳 AI 政策书籍**：一场讨论推荐了 [最佳 AI 政策书籍](https://www.perplexity.ai/search/what-are-the-best-ai-policy-bo-PORU.OiYRJewNfe_RFWkgw#0)，这些书籍提供了对 AI 技术监管和伦理考量的见解。
   - 与会者强调了塑造对 AI 社会影响理解的重要读物。



**提到的链接**: <a href="https://www.youtube.com/embed/FzBTzFmIjSI">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278141370569326663)** (1 条消息): 

> - `Perplexity AI integration` (Perplexity AI 集成)
> - `Chatbot responses` (Chatbot 响应)
> - `API usage` (API 使用)


- **在 Chatbot 中实现 Perplexity AI 的挑战**：一位用户正尝试将 **Perplexity AI** 集成到希伯来语的事实核查 Chatbot 中，但面临响应过短且缺少 **links** 和 **images** 的问题。
   - 他们注意到来自 API 的响应与 Perplexity 搜索网站上的响应有显著差异，并提到链接经常导致 **404 错误**。
- **寻求改进 API 响应的建议**：用户正在寻求建议，以增强 API 响应，使其包含完整答案、正确的来源链接以及类似于搜索网站的图像。
   - 他们正在考虑添加 **preliminary prompt**，并询问获得更好结果所需的特定模型或 Pro 激活要求。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278097775250378867)** (1 条消息): 

> - `API degradation` (API 性能下降)
> - `Incident Recovery` (故障恢复)


- **API 性能下降事件短暂影响服务**：曾出现约 5 分钟的 *API 性能下降期*，影响了服务的可用性。
   - 补丁已发布，该事件似乎已完全 **恢复**。
- **迅速有效的事件响应**：响应团队在 API 性能下降期间迅速识别了问题，确保了最小程度的中断。
   - 这种主动的方法突显了快速响应在维护服务完整性方面的重要性。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1278060899789443193)** (1 条消息): 

> - `Appreciation for Team Efforts` (对团队努力的赞赏)
> - `Tweet about AI Collaboration` (关于 AI 协作的推文)


- **团队努力获得认可！**：一位成员对团队的贡献表示感谢，称：*“感谢团队！”*
   - 这一认可突显了相关人员的协作精神和付出的辛勤努力。
- **在 Twitter 上强调 AI 协作**：分享了一条 [推文](https://twitter.com/gpudad/status/1828502015238119490)，展示了 AI 协作方面的重大进展。
   - 该推文强调了社区努力在推动 AI 技术发展中的重要性。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1277712951561945099)** (84 条消息🔥🔥): 

> - `OpenRouter 模型费用结构`
> - `DisTrO 分布式训练创新`
> - `Cerebras 定价与功能`
> - `DeepSeek 中的 Context Caching`
> - `Gemini 模型更新` 


- **OpenRouter 模型定价与费用说明**：一位用户询问 OpenRouter 显示的每 token 价格是否包含服务费。经澄清，所列价格基于 OpenRouter 积分，不包括充值积分时产生的任何额外费用。
- **DisTrO 为分布式训练带来新希望**：一位成员强调了 Nous Research 发布的关于 DisTrO (Distributed Training Over-the-Internet) 的初步报告，该技术提高了分布式训练效率。它有望大幅减少 GPU 间的通信，从而实现更具弹性的大模型训练。
- **Cerebras 提供极具竞争力的定价**：Cerebras 目前将 Llama 3.1-8B 的定价设定为每百万 token **10 美分**，Llama 3.1-70B 为 **60 美分**，引起了社区成员的兴趣。讨论内容包括潜在的合作以及平台的持续改进。
- **OpenRouter 与 DeepSeek 的 Context Caching**：讨论了 OpenRouter 是否支持 DeepSeek 的 Context Caching，表明了对提高性能和成本效率的渴望。据指出，OpenRouter 正在等待进一步的更改，以支持用于缓存的自定义用户分段。
- **Gemini 模型的激动人心更新**：讨论了即将发布的 Gemini 1.5 Flash 和 Pro 模型，用户对其潜在功能和性能表示期待。有推测认为，这些更新可能旨在与 GPT-4 等现有模型竞争。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/NousResearch/status/1828121648383566270">来自 Nous Research (@NousResearch) 的推文</a>: 如果你能利用世界上所有的算力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/OfficialLoganK/status/1828480081574142227">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: 今天，我们将推出三个实验性模型：- 一个新的更小变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...</li><li><a href="https://x.com/OfficialLoganK/status/1828484457675751814">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @patricksrail Vertex 将在今天晚些时候推出 1.5 Flash 和 Pro 模型（不包括 8B），应该很快！</li><li><a href="https://docs.anthropic.com/en/release-notes/system-prompts#july-12th-2024">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-inst">Llama 3.1 405B (base) - API, 提供商, 统计数据</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b">Llama 3.1 405B (base) - API, 提供商, 统计数据</a>: Meta 最新的模型系列 (Llama 3.1) 推出了多种尺寸和版本。通过 API 运行 Llama 3.1 405B (base)</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct">Llama 3.1 405B Instruct - API, 提供商, 统计数据</a>: 备受期待的 400B 级 Llama3 来了！拥有 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的 c...</li><li><a href="https://x.com/hyperbolic_labs/status/1828481468156518691">来自 Hyperbolic (@hyperbolic_labs) 的推文</a>: BF16 精度的 Llama 3.1 405B Base：现已在 Hyperbolic 上线 🦙💜 Base 模型比指令微调模型更具创造力和能力，但直到现在它们一直未被充分利用。➡️ 开始使用...</li><li><a href="https://platform.deepseek.com/api-docs/news/news0802/">DeepSeek API 引入磁盘 Context Caching，将价格降低了一个数量级 | DeepSeek API 文档</a>: 在大语言模型 API 使用中，很大一部分用户输入往往是重复的。例如，用户提示词经常包含重复的引用，而在多轮对话中，之前的...</li><li><a href="https://platform.deepseek.com/api-docs">快速入门 | DeepSeek API 文档</a>: DeepSeek API 使用与 OpenAI 兼容的 API 格式。通过修改配置，你可以使用 OpenAI SDK 或兼容 OpenAI API 的软件来访问 DeepSeek API。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1277995784926658691)** (1 条消息): 

> - `Activity page indicators`（活动页面指示器）
> - `Model pricing transparency`（模型定价透明度）
> - `Integrations provider insights`（集成提供商见解）


- **活动页面需要清晰的路由指示器**：有人建议在**活动页面**添加指示器，以显示请求是否已路由到某个**集成提供商**。
   - 目前该页面显示为 **$0**，这可能会因潜在错误误导用户，或者仅仅反映了 **Hermes 405b** 的**模型价格**。
- **关于模型定价和错误的说明**：针对活动页面当前显示 **$0** 的情况提出了担忧，这可能是由其他错误引起的。
   - **模型定价**的可见性对于防止混淆和提升用户体验至关重要。


  

---



### **tinygrad (George Hotz) ▷ #[announcements](https://discord.com/channels/1068976834382925865/1069236008115253348/)** (1 条消息): 

georgehotz: 在这里购买你的 tinybox https://tinycorp.myshopify.com/
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1277704300595908721)** (68 条消息🔥🔥): 

> - `Tinybox Shipping Issues`（Tinybox 运输问题）
> - `Tinygrad and BERT Training`（Tinygrad 与 BERT 训练）
> - `Tinybox Availability in Different Regions`（Tinybox 在不同地区的可用性）
> - `Tinybox Sales Update`（Tinybox 销售更新）
> - `Tinygrad Runtime Errors`（Tinygrad 运行时错误）


- **Tinybox 在欧洲的运输挑战**：讨论了目前国际买家（尤其是英国和欧洲）无法购买 **Tinybox** 的问题。一位成员建议通过电子邮件联系支持部门获取发往**欧洲**的运费报价，这引发了关于全球可用性的疑问。
   - 报告显示，**法国**和**意大利**的用户看到 **Tinybox** 显示为售罄，同时有人询问未来的运输解决方案。
- **探索使用 Tinygrad 进行 BERT 训练**：一位成员表示有兴趣使用 **Tinygrad** 预训练大型 **BERT** 模型，并强调需要高性能环境的支持。关于使用 **Tinygrad** 还是 **Torch** 存在不同看法，并提到了其训练大型模型的能力。
   - 对话强调 **Torch** 可能更适合所需的配置，并提到了 **64 张 Hopper 卡** 等重大硬件需求以及现有的 **Torch** 使用经验。
- **Tinybox 销售更新**：George 提到已经售出了大约 **40 台 Tinybox**，并且还有 **60 台** 的供应量。销售增长的兴奋感与仍在谈判中的国际销售限制形成了对比。
   - 社区还推测了 Tinybox 可能推出的新**颜色版本**，George 否认了近期会推出**蓝色版本**。
- **使用 Tinygrad 时的运行时错误**：一位在 Linux 服务器上运行 **Tinygrad** 的用户在使用 NVIDIA **RTX 4000 SFF Ada** 时遇到了错误，引发了对配置要求的关注。开发者们介入并建议检查配置和编译标志以进行排查。
   - 进一步验证配置的尝试得出了成功的测试结果且无错误，随后社区进行了进一步诊断。
- **关于 Tinybox 设计的讨论**：用户对 **Tinybox** 商店界面的图像分辨率提供了反馈，认为最近的更新可能导致了视觉质量下降。团队正在调查此问题以确保产品准确展示。
   - 还有一些关于未来潜在型号的幽默评论，突显了社区对产品开发讨论的参与度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tinycorp.myshopify.com/products/tinybox-red">tinybox red edition</a>: 必须在订单确认后 5 天内完成付款以确保订单。仅限美国本土运输。加拿大请联系 support@tinygrad.org。付款方式：银行转账/电汇 文档可以...</li><li><a href="https://tinycorp.myshopify.com/products/tinybox-green">tinybox green edition</a>: 必须在订单确认后 5 天内完成付款以确保订单。仅限美国本土运输。加拿大请联系 support@tinygrad.org。付款方式：银行转账/电汇 文档可以...</li><li><a href="https://en.wikipedia.org/wiki/Freight_forwarder">Freight forwarder - Wikipedia</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1277967516852945037)** (9 messages🔥): 

> - `Tinygrad 中的 RecursionError`
> - `创建转移矩阵`
> - `Tinygrad 版本 0.9.2` 


- **Tensor 转换时触发 RecursionError**：一位用户报告在 Tinygrad 中对 Tensor 调用 `.tolist()` 时出现了 `RecursionError: maximum recursion depth exceeded`，特别是在处理超过 3500 篇维基百科文章时。
   - *处理 2000 篇维基百科文章时运行正常*，这引发了关于处理更大输入时底层问题的疑问。
- **寻求用于调试的最小示例**：该用户表示打算创建一个最小示例以更好地诊断问题，并询问通过 DM 分享代码是否可行。
   - 另一位用户建议，如果能提供更小的可复现示例，他们可以提交一个 issue。
- **确认 Tinygrad 版本**：用户确认其安装的 Tinygrad 版本为 **0.9.2**，这可能与遇到的错误有关。
   - 此版本可能与讨论的问题有关，特别是 LazyOp 功能。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1277902305697464323)** (50 条消息🔥): 

> - `Intel LLaVaOLMoBitnet1B`
> - `OpenAI Orion AI`
> - `Nous Research optimizer`
> - `Cerebras Inference speed`
> - `Gemini 1.5 models` 


- **Intel 推出 LLaVaOLMoBitnet1B**：Intel 发布了 [LLaVaOLMoBitnet1B](https://huggingface.co/papers/2408.13402)，这是首个能够处理图像和文本输入并生成连贯文本响应的三值（Ternary）多模态 LLM。
   - 该模型连同训练脚本已完全开源，以促进该领域的进一步研究，并突出了三值模型面临的挑战和未来机遇。
- **OpenAI 研发 Orion AI**：据[独家报道](https://www.theinformation.com/articles/openai-races-to-launch-strawberry-reasoning-ai-to-boost-chatbot-business?utm_campaign=Editorial&utm_content=Article&utm_medium=organic_social&utm_source=twitter)，OpenAI 在寻求额外资金的同时，目标是开发一款名为 'Orion' 的新 AI 模型，该模型能够推理复杂问题。
   - 这一举措正值 OpenAI 寻求通过增强 AI 能力来巩固其聊天机器人业务之际。
- **对 Nous Research 优化器的质疑**：成员们对 Nous Research 新优化器的合法性持怀疑态度，表示需要更多证据来支持其关于分布式训练能力的主张。
   - 讨论中提到了现有的工具如用于 Bloom 的 Petals 和 OpenDILo，但关于 Nous 承诺的真实性仍存在不确定性。
- **Cerebras 声称拥有最快的 Llama3.1 推理速度**：Cerebras Systems 宣布其推理 API 在 8B 模型上达到 **1,800 tokens/s**，在 70B 模型上达到 **450 tokens/s**，显著快于 NVIDIA 等竞争对手。
   - 成员们对推理速度方面的竞争感到兴奋，特别提到他们非常乐见这一领域的快速进步。
- **发布新的 Gemini 1.5 模型**：Google 推出了三个实验性模型：更紧凑的变体 **Gemini 1.5 Flash-8B**、针对编程任务的更强版本 **Gemini 1.5 Pro**，以及改进后的 **Gemini 1.5 Flash 模型**。
   - 相关细节及试用链接已通过 [Google's AI Studio](https://aistudio.google.com) 分享，引发了关于其潜在能力的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://evaleval.github.io/">Home - EvalEval 2024</a>：关于衡量生成式 AI 系统广泛影响的最佳实践的 NeurIPS 2024 工作坊</li><li><a href="https://x.com/officiallogank/status/1828480081574142227?s=46">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天，我们推出了三个实验性模型：- 一个新的更小变体 Gemini 1.5 Flash-8B - 一个更强的 Gemini 1.5 Pro 模型（在编程和复杂提示词上表现更好）- 一个显著改进的 Gem...</li><li><a href="https://x.com/theinformation/status/1828418859990229073?s=46">The Information (@theinformation) 的推文</a>：独家：随着 OpenAI 寻求筹集更多资金，它正试图推出能够推理难题的 AI，并帮助其开发新的 AI 模型 'Orion'。https://www.theinformation.c...</li><li><a href="https://x.com/_akhaliq/status/1828271805825434066">AK (@_akhaliq) 的推文</a>：Intel 展示 LLaVaOLMoBitnet1B，三值 LLM 迈向多模态！讨论：https://huggingface.co/papers/2408.13402 多模态大语言模型 (MM-LLMs) 已经取得了显著进展...</li><li><a href="https://x.com/aiexplainedyt/status/1828430051735441706?s=46">AI Explained (@AIExplainedYT) 的推文</a>：未找到描述</li><li><a href="https://x.com/cerebrassystems/status/1828465008298336588?s=46">Cerebras (@CerebrasSystems) 的推文</a>：Cerebras Inference 是目前最快的 Llama3.1 推理 API：8B 为 1,800 tokens/s，70B 为 450 tokens/s。我们比 NVIDIA GPU 快约 20 倍，比 Groq 快约 2 倍。</li><li><a href="https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf">DisTrO/A_Preliminary_Report_on_DisTrO.pdf at main · NousResearch/DisTrO</a>：互联网分布式训练。欢迎在 GitHub 上为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1278047899606978715)** (5 messages): 

> - `Ideogram Twitter Space`
> - `艺术家与 AI 训练`
> - `观众操纵`
> - `数据讨论`
> - `AI 工具的使用权限` 


- **Ideogram Twitter Space 引发关注**：在 Ideogram 的 Twitter Space 生日活动期间，一个关于艺术家感到被使用其作品的公司“剥削”的问题导致提问者被移除，Ideogram 声称他们只收到了正面反馈。
   - 随后，一位“真正的艺术家”赞扬了 Ideogram，但被发现她与该公司的投资者 **a16z** 有关联。
- **对安插观众的担忧**：一位成员对提出的担忧表示赞同，但批评安插观众的行为非常 **cringe**（令人尴尬）。
   - 这引发了关于此类活动中互动真实性的讨论。
- **关于数据讨论代价的辩论**：一位用户评论说，围绕数据的讨论让人 **感到疲惫且悲哀**，暗示了 AI 对艺术创作更广泛的影响。
   - 这种情绪反映了社区内对于 AI 对创意产业冲击的担忧。
- **对访问不平等的担忧**：有人担心未来可能只有 **富有且有人脉的艺术家** 才能使用最好的 AI 工具，从而引发了伦理问题。
   - 这突显了潜在的鸿沟，即只有特定群体能从 AI 艺术技术的进步中受益。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278003649557168160)** (12 messages🔥): 

> - `Andrew Huberman 的咖啡实验`
> - `Kaggle 竞赛反馈`
> - `Claude Summer 讨论`
> - `Gemini API Rate Limits` 


- **Andrew Huberman 测试咖啡对自身的影响**：一段名为 [“Is Andrew Huberman Ruining Your Morning Coffee?”](https://www.youtube.com/watch?v=yCJr49GU9yY) 的 YouTube 视频讨论了他的随机对照试验，他在试验中交替饮用无咖啡因和含咖啡因的咖啡。
   - 文中还提到了使用 Shopify 进行业务设置的需求，这暗示了一个教育链接的变通方案。
- **对 Atlantic 的 Kaggle 竞赛的批评**：一位成员分享了由大西洋国际研究中心（Atlantic International Research Centre）主办的 [Kaggle 竞赛](https://www.kaggle.com/competitions/internal-waves)，并表示粗略看了一下后觉得没什么意思。
   - *人们的反应不一*，表现出对该竞赛吸引力的普遍冷淡。
- **关于图表异常的讨论**：在关于 Claude AI 的讨论中，有人投诉一个带有“随机非整数刻度”和“短 X 轴”的图表。
   - 成员们表达了挫败感，其中一人声明他们支持 Claude，但讨厌这些令人困惑的图表。
- **对 Gemini API Rate Limits 的困惑**：一位成员询问了使用 Gemini API 的经验，对如何解决其 Rate Limits 表示困惑。
   - 他们形容该系统“一团糟”，并指出存在不一致性，即某些模型可以运行而其他模型则不行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/morqon/status/1828463686438048211?s=46">来自 morgan — (@morqon) 的推文</a>：hot claude summer</li><li><a href="https://x.com/kagglingdieter/status/1828446217958822277">来自 Dieter (@kagglingdieter) 的推文</a>：人们问我这个由大西洋国际研究中心主办的竞赛看起来是否有趣。看了 10 秒钟 🤦 https://www.kaggle.com/competitions/internal-waves</li><li><a href="https://www.youtube.com/watch?v=yCJr49GU9yY">Is Andrew Huberman Ruining Your Morning Coffee?</a>：要使用 Shopify 创业，请通过此链接获得免费试用 http://shopify.com/jameshoffmann。今天的视频有些不同。我们认为我们应该...
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1277939660554895372)** (30 条消息🔥): 

> - `lm eval metrics`
> - `Tokenizer v3`
> - `Mistral tokenizer config issues`
> - `Jinja parser and masking`
> - `Multi-role in ShareGPT` 


- **lm eval 指标取决于基准测试**：对于多项选择题，使用的指标是**目标预测准确率**，这取决于模型的最高 logit 概率是否与正确选项一致。
   - 成员们讨论了答案可能略有不同的场景，强调了评估模型输出时的细微差别。
- **关于 tokenizer v3 的疑问**：多位成员对 **tokenizer v3** 表示困惑，其中一人链接到了 **nemo repo** 之前的讨论。
   - 另一位成员强调需要支持 **multi-role**（多角色）功能的正确 tokenizer 配置。
- **Mistral 的初始配置错误**：一位成员指出 **Mistral** 在最初发布时其 `tokenizer_config.json` 存在问题，并对这一疏忽表示失望。
   - 他们强调了在 tokenizer 应用中准确配置的重要性，以避免未来出现类似错误。
- **对 Jinja 中 masking 功能的需求**：用户强烈要求为 **masking** 提供标签，并讨论了在 **multi-role** 语境下 masking 应如何工作。
   - 一位成员建议将 attention 设置为 -100 以从训练中排除某些输入，从而引发了对 masking 方法的进一步探索。
- **理解多角色和 masking 效果**：一位成员澄清在 **ShareGPT** 中，可以为输入指定 masking，从而在训练期间忽略某些角色。
   - 特定代码配置示例的分享突显了定义具有自定义角色的数据集的复杂性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/conversation.html#sharegpt">Conversation – Axolotl</a>: 未找到描述</li><li><a href="https://github.com/axolotl-ai-cloud/axolotl/blob/17af1d7081414c32614cbabe324e1197ca9f43a7/src/axolotl/prompt_strategies/chat_template.py#L188">axolotl/src/axolotl/prompt_strategies/chat_template.py at 17af1d7081414c32614cbabe324e1197ca9f43a7 · axolotl-ai-cloud/axolotl</a>: 尽管提问。通过在 GitHub 上创建账户为 axolotl-ai-cloud/axolotl 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1277995598603227177)** (7 条消息): 

> - `Python Monkey-Patching`
> - `Deepseek V2 attention model`
> - `FSDP memory requirements` 


- **为 Deepseek V2 进行 Python Monkey-Patching**：关于对 **Deepseek V2** attention 模型进行 **monkey-patching** 的讨论，涉及分享一段重写其 forward 方法以便在调用时打印消息的代码片段。
   - 一位用户提到了他们在 Java 中使用 monkey-patching 的经验，但强调他们以前从未在 Python 中做过。
- **修复 Deepseek V2 远程代码问题**：分享了一个 [GitHub pull request](https://github.com/axolotl-ai-cloud/axolotl/pull/1873/files#diff-ed7a6cebcdf220a8815697365704ad0e3dff808147447bb611040f634b7f4e27) 链接，详细说明了 Deepseek V2 实现中远程代码问题的修复，该 PR 取代了之前的 pull request。
   - 讨论中提到了 *“奇怪的修复”*，强调了代码稳定性方面持续存在的挑战。
- **模块重载影响 Deepseek V2**：一位用户提到贡献者 **tmm1** 发现 Deepseek V2 的问题与模块如何被 **reloaded**（重载）有关。
   - 这指向了在 Python 中动态处理模块状态的复杂性。
- **对 FSDP RAM 需求的担忧**：一位用户询问 **FSDP** (Fully Sharded Data Parallel) 是否需要大量的**系统 RAM** 才能有效运行。
   - 这引发了对正常运行 FSDP 所需最佳系统资源的关注。



**提及的链接**: <a href="https://github.com/axolotl-ai-cloud/axolotl/pull/1873/files#diff-ed7a6cebcdf220a8815697365704ad0e3dff808147447bb611040f634b7f4e27">Sample pack trust remote code v2 by winglian · Pull Request #1873 · axolotl-ai-cloud/axolotl</a>: 取代了 #1872。感谢 @tmm1 ！

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1277732397231636642)** (16 messages🔥): 

> - `Training Tokenizer Vocabulary` (训练 Tokenizer 词表)
> - `Citing Axolotl Study` (引用 Axolotl 研究)
> - `Embedding Adjustments After Training` (训练后的 Embedding 调整)
> - `Metric Analysis of Token Changes` (Token 变化的指标分析)


- **理解 Tokenizer 训练行为**：一位成员询问代码 `modules_to_save = ["lm_head", "embed_tokens"]` 是训练整个 Tokenizer 词表还是仅训练新添加的 Token，得到的澄清是**所有词表内容**都会被训练。
   - 进一步的讨论围绕着当添加**新 Token** 时，是否有必要训练特定层。
- **在 ArXiv 研究中引用 Axolotl**：一位成员询问在新的研究中应如何署名个人贡献，并表示他们目前引用的是 [GitHub 仓库](https://github.com/axolotl-ai-cloud/axolotl)。
   - 对话中包含了一个指向 ArXiv 上相关研究的**参考链接**，位于 **#15**，通过[此链接](https://www.arxiv.org/pdf/2408.11857)访问。
- **训练后受影响的 Token**：另一位成员对训练后哪些 Token 受影响最大表示好奇，特别是在添加了 **'pad' token** 之后。他们的代码旨在识别训练调整后受影响最大的前几个 Token。
   - 针对分析结果提出了疑虑，认为鉴于其数据集的特殊性，这可能作为一个潜在的**指标 (metric)**。
- **Token 分析中的余弦距离 (Cosine Distance)**：用户修改了方法，通过计算 Token 调整的余弦距离，希望这能对训练后的 Token 变化产生更清晰的见解。
   - 另一位成员指出，**Embeddings** 可能不是唯一受训练强烈影响的方面，相反，对这些 Embeddings 的**解释 (interpretation)** 同样至关重要。
- **训练中 Token 表示的有效性**：讨论继续围绕着含义与预训练数据集不同的 Token（如 **'adam'**）是否会显示出有效的训练结果。
   - 用户注意到发现了一些意料之外的 Token，包括 **'Fortunately'**，表明他们正在寻求确认其训练的有效性。



**提及的链接**：<a href="https://github.com">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献、管理 Git 仓库、像专家一样评审代码、跟踪 Bug 和功能...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1277871711240454144)** (2 messages): 

> - `llm-as-judge`
> - `human ratings comparison` (人类评分对比)


- **使用 LLM 作为评分裁判**：一位成员表示他们利用 **llm-as-judge** 采用了**评分提示词 (prompt for rating)**。
   - 这种方法引出了关于 *AI 裁判与人类评分相比的准确性* 以及是否进行了任何测试的问题。
- **AI 评分与人类评分的准确度对比**：有人询问 AI 裁判与**人类评分**相比的准确度如何。
   - 另一位成员询问了有关*可能已经执行的测试*细节，以评估这种准确性。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278128886567407637)** (1 messages): 

> - `User Feedback on Magic` (关于 Magic 的用户反馈)
> - `Exclusive Swag for Feedback Providers` (为反馈提供者准备的专属礼品)


- **征集 Magic 功能的反馈**：团队正在寻找 **5 名参与者**，愿意花费 **30 分钟**专门针对 **magic** 提供用户反馈。
   - 参与者将获得专属的 **Swag** 作为贡献奖励，感兴趣的用户请[在此预约](https://modul.ar/user-feedback)。
- **专属 Swag 奖励**：提供反馈的用户将成为首批获得目前处于设计阶段的专属 **Swag** 的人。
   - 该倡议旨在激励用户投入，并对参与的贡献者表示感谢。



**提及的链接**：<a href="https://modul.ar/user-feedback">Appointments</a>：未找到描述

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1277707142383796256)** (43 messages🔥): 

> - `ClassStruct` 与可变参数 (Variadic Parameters)
> - `Struct` 字段的性能
> - Mojo 中的类型推断 (Type Inference)
> - 函数重载 (Function Overloading) 的挑战
> - Mojo 与 Luma 的对比 


- **理解 ClassStruct 的可变参数**：ClassStruct 允许在 Mojo 中进行动态参数化，使用户无需手动构建唯一的 `struct` 即可创建变体，例如在 `car` 示例中可以动态定义引擎尺寸。
   - 这种功能允许程序员通过定义基于编译时可用参数的 `struct` 字段，高效地利用变体。
- **字段限制带来的性能问题**：讨论揭示了编译具有大量字段的 `struct` 可能会导致显著的性能下降，示例显示 100 个字段大约需要 1.2 秒。
   - 有人建议这可能是由于底层数据结构需要调整大小，表明存在一个性能显著下降的阈值。
- **类型推断的挑战**：Mojo 中的类型推断有限，尤其是在处理泛型 (Generics) 时，与 Rust 强大的系统相比，编程体验较为繁琐。
   - 这引发了关于 Mojo 当前对泛型和 `typeclasses` 的处理如何限制其灵活性（相比于成熟语言）的讨论。
- **函数重载与复杂性**：参与者指出，由于类型推断行为的不一致，Mojo 中的函数重载带来了挑战，使开发过程变得复杂。
   - 大家的共识是，改进函数重载和类型推断的处理可以显著提升用户体验。
- **Mojo vs. Luma：类型系统对比**：对比了 Mojo 和 Luma，暗示虽然 Luma 拥有更强大的类型推断，但 Mojo 提供了像 `typeclasses` 这样具有限制性的独特功能。
   - 讨论强调，随着 Mojo 的演进，它可能会更接近 Rust 的能力，并可能引入诸如效应系统 (Effect Systems) 之类的功能。


  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1277713478731169863)** (3 messages): 

> - `RAG-a-thon`
> - `Llama 3.1-8b` 速度
> - `Serverless RAG` 应用程序
> - LlamaIndex 与 Azure OpenAI 


- **准备参加与 Pinecone 合作的 RAG-a-thon！**：我们将于 10 月 11 日至 13 日在帕洛阿尔托的 [500GlobalVC](https://t.co/IFvyW5QB6r) 办公室举办第二届 **RAG-a-thon**，提供超过 **$7k 的现金奖励**！
   - 这是一个展示创新想法并在协作环境中获得宝贵经验的绝佳机会。
- **Llama 3.1-8b 打破速度记录**：需要超快速响应？**Llama 3.1-8b** 提供每秒 **1800 个 token**，使其成为目前最快的 LLM，详情讨论见 [此处](https://t.co/hZv6ooGUxO)。
   - 实现这种速度对于需要快速响应的应用至关重要，尤其是在复杂系统中。
- **使用 LlamaIndex 构建 Serverless RAG 应用**：通过 **Wassim Chegham** 的这份综合指南 [指南链接](https://t.co/1XKg1o2bIX)，学习如何使用 LlamaIndex 和 Azure OpenAI 创建 **Serverless RAG 应用程序**。
   - 它涵盖了对 RAG 架构的理解，并展示了如何利用您自己的业务数据来改进 AI 驱动的响应。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1277716039341445191)** (36 messages🔥): 

> - `Callback Manager 到 Instrumentation 的迁移`
> - `Neo4j Schema 问题`
> - `从扫描文档中提取数据`
> - `GPT-4o-mini 模型支持`
> - `GraphRAG v2 性能` 


- **Callback Manager 和 Instrumentation 的混淆**：一位成员询问了 **RetrieverQueryEngine** 和 **CondensePlusContextChatEngine** 之间 trace spans 的差异，指出两者目前都在使用 callback manager。
   - 另一位成员推测 **LangFuse** 仍然是使用 callback manager 实现的，这意味着 trace 损坏可能是一个 bug。
- **Neo4j 无法构建关系**：一位用户报告在使用 Neo4j Desktop 复制 LlamaIndex 的 property graph 教程时遇到困难，关系未能被正确提取。
   - 他们澄清说自己严格遵守了教程，包括使用默认的 schema，并怀疑他们的 Neo4j 设置可能与默认预期不符。
- **使用 LlamaParse 增强数据提取**：一位用户讨论了 **LlamaParse** 在转换表格数据时由于扫描问题可能出现的问题，并寻求在 pipeline 中集成图像提取的解决方案。
   - 针对处理结合了图像的多表格时的 chunking 策略提出了疑问。
- **GPT-4o-mini 支持问题**：一位用户尝试使用 **gpt-4o-mini** 模型，但遇到了 ValueError，提示该模型未知，尽管 chat 建议它应该被支持。
   - 另一位成员建议更新库以修复此问题，暗示可能在模型支持兼容性方面存在疏忽。
- **GraphRAG v2 表现不如预期**：一位用户报告 **GraphRAG v2** 在其数据集上失败，尽管早期版本运行良好，这引发了关于性能预期的疑问。
   - 成员们讨论了图谱（graphs）的必要性，一些人对它们在某些设置中的必要性表示怀疑。



**提到的链接**：<a href="https://github.com/run-llama/llama_index/pull/15679">fix tool schemas by logan-markewich · Pull Request #15679 · run-llama/llama_index</a>：最近对 pydanticv2 的更改导致我们的 tool json schemas 丢失了重命名的 &quot;definitions&quot; 部分。

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1277767943568232489)** (36 messages🔥): 

> - `Nous Research 的 DisTrO`
> - `微软的 Phi 3.5 Vision 模型`
> - `Cerebras 推理性能`
> - `Gemini 1.5 模型更新`
> - `Anthropic Artifacts 发布` 


- **DisTrO 彻底改变分布式训练**：Nous Research 发布了关于 [DisTrO](https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO.pdf) 的初步报告，这是一个能将 GPU 间通信大幅减少高达 10,000 倍的框架，从而实现 LLM 的弹性训练。
   - 该框架旨在促进 AI 研究中的协作，而不依赖于单一实体，从而增强模型开发的安全性合竞争性。
- **微软的 Phi 3.5 Vision 模型在 OCR 方面表现出色**：Dylan Freedman 强调了微软的 [Phi 3.5](https://huggingface.co/spaces/MaziyarPanahi/Phi-3.5-Vision) 模型在 OCR 方面的卓越表现，特别是在手写识别和提取表格数据方面，该模型采用了宽松的许可（MIT）。
   - 该模型展示了卓越的文本识别能力和跨各种视觉任务的能力，社区中讨论了其显著的表现。
- **Cerebras 刷新推理速度记录**：Cerebras 宣布了其推理服务，在 8B 模型上达到了 [1,800 tokens/s](https://x.com/CerebrasSystems/status/1828465008298336588)，显著优于 NVIDIA 和 Groq 的方案。
   - 在定制的 WSE-3 芯片驱动下，Cerebras 还为 Llama 模型提供了极具竞争力的价格，引发了开发者社区对其经济可行性和性能的讨论。
- **Gemini 1.5 模型发布**：Google 推出了 Gemini 1.5 系列下的三个实验性模型，包括一个较小的变体和一个更强大的 Pro 模型，其在编程和复杂 prompt 处理方面的能力备受关注。
   - 随着开发者评估其竞争优势和性能，最近的发布引发了与 GPT-4o-mini 等现有模型的比较。
- **Anthropic 的 Artifacts 和文章讨论**：Anthropic 随着 [Artifacts](https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts) 的发布取得了重大进展，紧随其后的是对其技术和方法论感兴趣的开发者。
   - 针对该文章及时发布的动机有人提出了担忧，导致在消息平台上的讨论中出现了关于潜在付费软文的猜测。


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://x.com/officiallogank/status/1828480081574142227?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：今天，我们正在推出三个实验性模型：- 一个新的更小变体，Gemini 1.5 Flash-8B - 一个更强大的 Gemini 1.5 Pro 模型（在编程和复杂提示词方面表现更好）- 一个显著改进的 Gem...</li><li><a href="https://x.com/CerebrasSystems/status/1828465008298336588">来自 Cerebras (@CerebrasSystems) 的推文</a>：Cerebras Inference 是目前为止最快的 Llama 3.1 推理 API：8B 模型为 1,800 tokens/s，70B 模型为 450 tokens/s。我们比 NVIDIA GPU 快约 20 倍，比 Groq 快约 2 倍。</li><li><a href="https://x.com/NousResearch/status/1828121648383566270">来自 Nous Research (@NousResearch) 的推文</a>：如果你可以利用世界上所有的算力来训练一个共享的开源 AI 模型会怎样？初步报告：https://github.com/NousResearch/DisTrO/blob/main/A_Preliminary_Report_on_DisTrO...</li><li><a href="https://x.com/AISafetyMemes/status/1828311798057181461">来自 AI Notkilleveryoneism Memes ⏸️ (@AISafetyMemes) 的推文</a>：今日科幻成真：Deepfake 直播。我重申：如果几年前你告诉人们这将会存在，他们是不会相信你的。引用 AI Notkilleveryoneism Memes ⏸️ (@AISafetyMe...</li><li><a href="https://x.com/dylfreed/status/1828132226523131931?s=46">来自 Dylan Freedman (@dylfreed) 的推文</a>：Microsoft 新的开源 Phi 3.5 vision 模型在 OCR/文本提取方面表现非常出色——甚至是手写体！你也可以通过提示词让它提取表格数据。它采用了宽松的授权协议 (MI...</li><li><a href="https://cerebras.ai/blog/introducing-cerebras-inference-ai-at-instant-speed">介绍 Cerebras Inference：即时速度的 AI - Cerebras</a>：我们很高兴宣布发布 Cerebras DocChat，这是我们为基于文档的对话式问答设计的首个迭代模型系列。该系列包括两个模型：Cerebras Llama...</li><li><a href="https://newsletter.pragmaticengineer.com/p/how-anthropic-built-artifacts">Anthropic 是如何构建 Artifacts 的</a>：Artifacts 团队（一种与 Claude 交互的创新方式）分享了他们如何在一个分布式团队中仅用三个月时间构建出这一创新功能。独家细节。</li><li><a href="https://x.com/AnthropicAI/status/1828462522468372600">来自 Anthropic (@AnthropicAI) 的推文</a>：今天，我们将向所有 Claude 用户开放 Artifacts。你现在也可以在 Claude 的 iOS 和 Android 应用上创建和查看 Artifacts。自 6 月份推出预览版以来，已有数千万个...</li><li><a href="https://x.com/artificialanlys/status/1828463912389402896?s=46">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Cerebras 刷新了 AI 推理速度的新纪录，Llama 3.1 8B 的输出速度达到 1,850 tokens/s，70B 达到 446 tokens/s。@CerebrasSystems 刚刚推出了他们的 API 推理服务，由...
</li>
</ul>

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1277756830491086849)** (9 messages🔥): 

> - `Streamlit Python Server`
> - `Telegram Bot with Open Interpreter`
> - `Open Interpreter Image Display Issue`
> - `Jupyter Metadata and Image Control`
> - `Cython and Black-Scholes Model` 


- **用于聊天的 Streamlit Python 服务器**：一位成员提到一个简单的 [Streamlit Python 服务器](https://link.to.streamlit)，可用于在 Web 浏览器中创建聊天界面。
   - 另一位成员给出了积极回应，表示将研究该解决方案。
- **配置使用 Open Interpreter 的 Telegram Bot**：一位成员分享了他们使用 **Open Interpreter** 创建 Telegram 机器人（Telegram Bot）的配置，包括 API key 和模型的设置。
   - 他们遇到了图像在电脑上意外显示的问题，引发了关于潜在修复方案的讨论。
- **修复 Open Interpreter 图像显示问题**：一位成员建议修改 Open Interpreter 中的自定义指令行，以帮助解决与显示图像相关的错误。
   - 另一位用户确认他们正尝试关闭 Jupyter 中显示图像的默认行为，并寻求帮助。
- **用于图像控制的 Jupyter 元数据**：一位用户分享了一个关于向 Jupyter notebooks 添加元数据以控制行为的链接，这可能有助于管理图像显示。
   - 讨论中包括了使用 `plt.ioff()` 来抑制 Jupyter 中自动图像输出等潜在解决方案。
- **Black-Scholes 模型的 Cython 规范**：一位成员发布了一个在 Jupyter notebooks 中使用 Cython 实现 Black-Scholes 模型的示例链接，展示了复杂的计算。
   - 该示例突出了如何在 Cython 中高效定义函数，从而提升期权定价的计算性能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://jupyterbook.org/en/stable/content/metadata.html#add-tags-using-python-code">向你的书籍页面添加元数据</a>：未找到描述</li><li><a href="https://nbviewer.org/github/ipython/ipython/blob/1.x/examples/notebooks/Cell%20Magics.ipynb">Jupyter Notebook 查看器</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1277942936427823157)** (22 messages🔥): 

> - `Issues with Poetry Command`
> - `Auth Error in iOS App`
> - `Brand Documentation for 01`
> - `Pre-order Status Inquiry` 


- **排除 Poetry 命令故障**：一位用户在运行 `poetry run 01 --server` 时遇到错误，提示激活的 Python 版本不受支持，且选项 '--server' 需要参数。
   - 另一位用户建议尝试运行 `poetry run 01 --server light` 作为潜在的修复方案。
- **01 iOS 应用中的身份验证错误**：一位用户报告在使用 01 iOS 应用时持续收到 `{"auth": false}`，暗示可能存在服务器问题。
   - 他们分享了相关的控制台输出，显示没有启动问题，并被建议尝试使用 `--server livekit` 启动服务器。
- **首次见面会品牌文档**：一位成员分享了一个 [Canva 设计](https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton) 链接，作为 01 项目目前最接近的品牌文档。
   - 他们还提到，更详尽的工业设计进展和文档将很快在 GitHub 仓库中提供。
- **预订状态更新**：一位用户询问如何联系相关人员了解其预订状态，强调了明确信息的需求。
   - 一位成员回应称即将发布更新，并提供了指向最新状态更新的链接。



**提及的链接**：<a href="https://www.canva.com/design/DAF8rbBol3Q/UNivuf8sjxVSveDfMFWpag/edit?utm_content=DAF8rbBol3Q&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton">简单得令人惊叹的图形设计软件 – Canva</a>：简单得令人惊叹的图形设计软件 – Canva

  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278029316567072861)** (4 messages): 

> - `Tool Use Episode`
> - `Video to Podcast Idea`
> - `Voice Cloning with ElevenLabs` 


- **Mike 和 Ty 的 Tool Use 系列回归**：Mike 和 Ty 带着新的一集回归，标题为 ["Video to Content Pipeline and YouTube Playlist Summarization - Ep2 - Tool Use"](https://www.youtube.com/watch?v=uzP_16F2zjA)，展示了如何从视频中提取数据以实现更好的利用。
   - 本集强调了 AI 如何从视频中获取信息，旨在提升观众体验。
- **将视频摘要转化为播客**：一名成员建议将视频摘要或电子邮件通讯转化为播客，以增强早间通勤期间的可访问性。
   - 这一想法受到了热烈欢迎，展示了关于内容消费的创新思维。
- **使用 ElevenLabs 的语音克隆实验**：在 Ty 幽默地提到他在前一集中克隆了 Mike 的声音后，Mike 准备再次使用 ElevenLabs 进行语音克隆。
   - Mike 的反应突显了他们实验的轻松性质，为他们的动态内容创作做出了贡献。
- **由克隆语音主持的每日播客**：有人开玩笑地提到一个由 Mike Bird 的克隆语音主持的每日播客，引发了笑声并增强了社区参与感。
   - 这一建议进一步强调了技术在播客和内容创作中的创意应用。



**提及的链接**：<a href="https://www.youtube.com/watch?v=uzP_16F2zjA">Video to Content Pipeline and YouTube Playlist Summarization - Ep2 - Tool Use</a>：Ty 和 Mike 向您展示如何从视频中提取数据并以惊人的方式加以利用。让 AI 从视频中获取信息来改善您的生活。拥有一个 YouTube 播放列表...

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1277965644104732733)** (15 messages🔥): 

> - `Llama 3.1 Inference`
> - `Ollama Framework Comparison`
> - `LoRA Fine-tuned Model Loading`
> - `AWS Instance Costs`
> - `Runpod Alternatives` 


- **Llama 3.1 在 CPU 上的推理困境**：一位用户报告称，即使在高端 AWS 服务器上，**Llama 3.1 8B** 在 CPU 上的推理速度也极慢（<0.05 tok/s）。
   - 讨论显示，与 GPU 设置相比，CPU 性能预计会显著降低，尤其是在使用 **Ollama** 等框架时。
- **建议使用优化的推理框架**：一位成员建议使用 **Ollama** 或 **vLLM** 来部署模型，因为与 torchtune 相比，它们针对推理进行了更多优化。
   - 一份[关于将 Ollama 与自定义 Checkpoint 结合使用的教程](https://github.com/ollama/ollama/blob/main/docs/import.md)被分享作为参考资源。
- **LoRA 微调模型加载咨询**：一位用户询问使用 `from_pretrained()` 是否能从本地 Checkpoint 正确加载 LoRA 微调权重。
   - 提供了一个[讨论如何将 LoRA 适配器加载到 HF 的 Issue](https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129) 链接以供进一步指导。
- **AWS 实例成本辩论**：关于 AWS 实例成本的讨论表明，一台 AWS c7a.24xlarge 的价格可能在 **$5/小时** 左右。
   - 另一位成员建议考虑 **Runpod** 等替代方案以获得更便宜的选择，但监管限制使该用户必须留在 AWS。
- **CPU 服务器的挑战**：该用户表示由于成本效益以及对其用例足够的响应时间，他们更倾向于使用 CPU 服务器。
   - 评论承认低 CPU 性能可能会影响推理速度，从而促使考虑选择优化的框架。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ollama/ollama/blob/main/docs/import.md">ollama/docs/import.md at main · ollama/ollama</a>：快速上手 Llama 3.1、Mistral、Gemma 2 和其他大语言模型。- ollama/ollama</li><li><a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/issues/832#issuecomment-2129867129">如何保存训练好的模型以便使用 HF `from_pretrained()` 加载？· Issue #832 · pytorch/torchtune</a>：我发现这个仓库是一个用户友好、可扩展、内存高效的模型训练/微调解决方案。然而，在推理方面，存在一个可以解决的易用性差距...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1277711967280435270)** (10 条消息🔥): 

> - `UI/UX 中的请求限制`
> - `模型未找到错误`
> - `Reranker 版本更新` 


- **关于请求限制的担忧**：一位成员对超过 **1k 次请求**的阈值表示怀疑，认为测试阶段肯定会涵盖在这个限制之内。
   - 另一位成员也表达了同样的担忧，表示他们不确定如何才能达到 1k 次请求。
- **间歇性模型错误体验**：一位成员报告遇到了 **'model not found'** 错误。
   - 这个问题似乎源于模型的版本控制，因为另一位成员指出 Reranker 现在已经是 **v3** 版本。



**提到的链接**：<a href="https://tenor.com/view/0001-gif-17282391190974969363">0001 GIF - 0001 - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1277977343339663442)** (3 条消息): 

> - `生产环境密钥 403`
> - `Langchain 与 Cohere TypeScript` 


- **需要澄清生产环境密钥 403**：*Renzhiping* 在没有上下文的情况下提到了“生产环境密钥 403”，导致成员们感到困惑。
   - *Nick Frosst* 作出回应，寻求对该引用的具体含义进行澄清。
- **Langchain 与 Cohere TypeScript 的 404 错误**：*Fealomeril* 报告了一个问题，即使用 **Langchain** 配合 **Cohere TypeScript** 进行的第一次调用产生了有效响应，但随后的调用却导致了 **404 page not found error**。
   - 这表明所使用的集成中可能存在不稳定性或配置错误。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1277723908577034293)** (8 条消息🔥): 

> - `Flutter 协作`
> - `vllm 与 RAG`
> - `LLM 工作流构建器`
> - `本地 Embedding 模型` 


- **Flutter 应用的社交协作**：@fritzlandry 表达了对 Flutter 应用开发协作的兴趣，一位成员回应了一个之前因缺乏 Flutter 经验而放弃的项目想法。
   - 这种潜在的协作可能有助于解决在构建 Flutter 应用程序方面共同存在的知识鸿沟。
- **使用 vllm 实现 RAG**：一位用户询问是否可以使用 **vllm** 运行检索增强生成 (RAG)，并提到他们拥有用于 Embedding 和回答的模型。
   - 这突显了对使用 **vllm** 及其功能进行多模型实现的持续探索。
- **寻找 LLM 工作流构建器**：@rogo6623 询问是否有人创建了包含 **LLM 能力**的工作流构建器，表现出对使用语言模型自动化流程的兴趣。
   - 这一询问表明了对在用户工作流中增强 LLM 功能的工具的需求。
- **使用 Embedding 模型进行本地工作**：一位用户正在寻找本地 Embedding 模型的推荐，并提到在使用 **Pinecone** 和 **Elasticsearch** 等云端解决方案时遇到了延迟。
   - 他们专门请求了促进本地设置的资源，标志着正在努力最大化模型性能的效率。
- **Ollama 作为本地解决方案**：@rogo6623 推荐使用 **Ollama** 进行本地模型部署，暗示其对于希望避免云端延迟的用户非常有效。
   - 这表明用户更倾向于使用本地解决方案，以增强响应速度和对模型的控制。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1278016049270226966)** (4 条消息): 

> - `仪表板的实用性`
> - `AI 模型中的错误处理`
> - `理解克隆的仓库` 


- **审视仪表板的价值**：一位成员质疑仪表板是否**值得**，并对 **Claude** 和 **GPT** 等模型在处理数学和编程任务时抛出的错误表示沮丧。
   - *我正在寻找更准确的东西，因为我的要求很具体，但我不懂 Web 编程*。
- **关于错误处理的讨论**：另一位成员寻求澄清所谓的**错误**是指什么，并强调仪表板主要是一个可视化工具。
   - 这突显了对 AI 输出的可靠性和准确性的持续关注。
- **对克隆仓库的好奇**：同一位成员询问克隆一个仓库是否能让 AI 解释每个文件及其**算法**，表达了对更深层次理解的渴望。
   - 这反映了使用 AI 工具进行更多技术指导的广泛兴趣。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1277775483966193759)** (5 messages): 

> - `Model Accessibility on Leaderboard` (排行榜上的模型可访问性)
> - `Benchmarking with BFCL` (使用 BFCL 进行基准测试)
> - `Multiple Model Versions` (多个模型版本)
> - `Benchmarking Llama 3.1` (对 Llama 3.1 进行基准测试)


- **模型必须可访问才能登上排行榜**：排行榜上列出的任何模型都必须是**公开可访问的**，无论是开源还是通过 API 端点进行推理。
   - *你可以设置注册/登录/令牌，* 但公众最终应该能够访问该端点。
- **无公开访问权限时的基准测试限制**：虽然可以使用 **BFCL** 对模型进行基准测试，但那些不可公开访问的模型无法显示在排行榜上。
   - 这区分了哪些模型可以展示，而哪些仅供评估。
- **接受多个模型版本**：系统允许一个模型的多个版本，例如 **prompt 版本**和经过微调的 FC 版本。
   - 这两种类型的版本都被接受用于基准测试。
- **寻求 Llama 3.1 的基准测试指导**：一位用户正在寻求关于使用其公司托管的自定义 API 端点对 **Llama 3.1 进行基准测试**的指导。
   - 他们请求关于如何有效启动基准测试流程的具体建议。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1277882849156010057)** (2 messages): 

> - `Function Calling Performance` (Function Calling 性能)
> - `BFCL Leaderboard Optimization Concerns` (BFCL 排行榜优化担忧) 


- **Function Calling 功能导致性能下降**：一位用户观察到，对于 **GPT-4-1106-Preview**，直接使用 system prompt 的准确率为 **85.65**，而启用 function calling 功能时准确率为 **79.65**。
   - 这种差异引发了关于启用 function calling 是否会固有地降低性能的疑问，并引发了对该主题的进一步讨论和研究。
- **对 BFCL 优化策略的担忧**：一位用户询问了他们正在为 function-calling 功能实施的优化策略，质疑这些策略是否会被认为违反了 BFCL 指南。
   - 有人担心，诸如更新 system prompts 和格式化输出之类的优化是否可能属于无法推广到所有模型的不公平做法。


  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/)** (1 messages): 

dr_monk: I am rooting for apple to pull through. 🙂 (我支持 Apple 能挺过去。🙂)
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1277740172141924608)** (5 messages): 

> - `DSPy Output Truncation` (DSPy 输出截断)
> - `DSPy Library Typing Support` (DSPy 库的类型支持)
> - `Scoring Generated Texts with DSPy` (使用 DSPy 对生成的文本进行评分) 


- **修复 DSPy 输出截断**：一位成员报告在使用 **DSPy** 时输出被截断，并怀疑可能是 token 限制的原因。另一位成员建议在初始化期间更改 **max_tokens**，并使用 `[your_lm.inspect_history()](https://some.link/here)` 查看 prompts。
   - 原帖作者确认此建议解决了他们的问题，凸显了社区的实际帮助。
- **DSPy 导入错误**：一位成员在导入 **DSPy** 时遇到了错误消息 `module is installed, but missing library stubs or py.typed`。他们询问 **DSPy** 是否支持 Python 中的类型提示（typing），表明需要更清晰的文档。
   - 尚未提供解决此类型问题的后续方案。
- **对使用 DSPy 进行文本评分的兴趣**：一位用户询问是否有人有使用 **DSPy** 根据 KPI 或行业指标（如 **BLEU** 或 **ROUGE**）对生成文本进行评分的经验。此查询反映了社区对评估文本生成性能指标日益增长的兴趣。
   - 然而，其他成员没有就使用 **DSPy** 评分文本提供回应或分享经验。


  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/)** (1 messages): 

rolandtannous: hello is Hamel around? (你好，Hamel 在吗？)
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1277775069090943077)** (1 messages): 

> - `LLM Observability Tools Webinar`
> - `Comparative Analysis of ML Monitoring Tools`
> - `Integration of Observability in LLMs` 


- **参加 LLM 可观测性工具网络研讨会**：本周六，8 月 31 日，**美国东部时间上午 11:30**，一场网络研讨会将探讨超过 **60 种 LLM 可观测性工具**，以确定它们在监控和优化模型方面的有效性。在此[注册](https://kyrylai.com/webinar-observability-platforms-overflow/)参加会议。
   - 参与者将深入了解可观测性基础知识、工具选择以及与 LLM 推理服务的集成。
- **测试 ML 监控平台的炒作**：可观测性领域已经饱和，许多工具声称在 **监控** 和 **调试** ML 模型方面具有优越性。本次网络研讨会旨在批判性地评估这些工具是否真正满足从业者的需求。
   - 期待通过实操评估来筛选这些主张，重点关注实用性和用户友好性。
- **Machine Learning in Production 训练营**：“Machine Learning in Production” 的直播训练营现已开放，旨在提升部署 ML 模型的实践技能。感兴趣的参与者可以在[这里](https://edu.kyrylai.com/courses/ml-in-production)查看更多详情。
   - 本课程承诺为学习者提供在实际应用中进行有效 ML 管理的基本工具和知识。



**提到的链接**：<a href="https://kyrylai.com/webinar-observability-platforms-overflow/">Observability Platforms Overflow: Why There Are More Monitoring Platforms Than ML Models | Live Webinar</a>：在即将举行的网络研讨会中，探索为什么可观测性平台的发展速度超过了 ML 模型，并学习如何为您的 ML 项目选择合适的监控工具。

  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/)** (1 messages): 

_baumer: 是否有 LAION-aesthetic 的 Hugging Face 链接？LAION 网站上的链接坏了。
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1277960292327755828)** (1 messages): 

> - `LocalAI`
> - `Ettore Di Giacinto AMA`
> - `Open-source alternatives to OpenAI` 


- **与 Ettore Di Giacinto 的 LocalAI AMA 即将开始**：两小时后参加与 **Ettore Di Giacinto** 的 **LocalAI AMA**，探索这个 OpenAI 的免费开源替代方案的功能。LocalAI 作为一个即插即用的 REST API，兼容各种 AI 规范，用于本地推理。
   - 该平台支持在不需要 GPU 的情况下本地运行大语言模型 (LLMs)、生成图像和音频，使其在消费级硬件上即可使用。
- **加入 LocalAI 对话**：**LocalAI** 活动链接已向所有人开放。[点击此处加入](https://discord.com/events/1089876418936180786/1268967945216721079)。
   - 不要错过直接与开发者交流并就这一创新项目提问的机会。


  

---



---



---



{% else %}


> 由于邮件篇幅限制，完整的频道分类详情已被截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}