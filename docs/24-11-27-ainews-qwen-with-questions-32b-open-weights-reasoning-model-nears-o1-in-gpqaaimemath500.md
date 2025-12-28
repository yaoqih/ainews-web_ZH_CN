---
companies:
- deepseek
- sambanova
- hugging-face
- dair-ai
date: '2024-11-28T01:23:25.425054Z'
description: '**DeepSeek r1** 在“开源 o1”模型的竞争中处于领先地位，但尚未发布权重；与此同时，**Justin Lin** 发布了
  **QwQ**，这是一个 **32B 开源权重模型**，在基准测试中表现优于 **GPT-4o** 和 **Claude 3.5 Sonnet**。QwQ 似乎是
  **Qwen 2.5** 的微调版本，强调通过顺序搜索和反思来解决复杂问题。


  **SambaNova** 宣传其 RDU 在推理任务上优于 GPU，突显了 AI 系统重心从训练向推理的转变。在 Twitter 上，**Hugging Face**
  宣布了对 llama.cpp 实例的 CPU 部署支持；**Marker v1** 作为一种更快速、更准确的部署工具正式发布；**Agentic RAG**（代理式检索增强生成）的发展则专注于集成外部工具和高级
  LLM 链，以提升响应准确性。


  开源 AI 社区的势头日益强劲，随着 **Flux** 等模型的普及，反映出 AI 正在向涵盖图像、视频、音频和生物学在内的多模态模型转变。'
id: fe1629b8-c778-490b-98d4-1570ba86fd12
models:
- deepseek-r1
- qwq
- gpt-4o
- claude-3.5-sonnet
- qwen-2.5
- llama-cpp
original_slug: ainews-qwen-with-questions-32b-open-weights
people:
- justin-lin
- clementdelangue
- ggerganov
- vikparuchuri
title: Qwen with Questions：32B 开源权重推理模型在 GPQA/AIME/Math500 表现上逼近 o1。
topics:
- model-releases
- benchmarking
- fine-tuning
- sequential-search
- inference
- model-deployment
- agentic-rag
- external-tools
- multi-modal-models
---

<!-- buttondown-editor-mode: plaintext -->**Think different.**

> 2024年11月27日至11月28日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**198** 个频道和 **2864** 条消息）。预计节省阅读时间（按 200wpm 计算）：**341 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在“开源 o1”的竞赛中，DeepSeek r1（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-deepseek-r1-claims-to-beat-o1-preview-and/)）仍然拥有最好的结果，但尚未发布权重。听起来疲惫不堪的 Justin Lin [今天突然发布了](https://x.com/Yuchenj_UW/status/1861858852855320803) QwQ，**包括权重、Demo 等一切**：


![image.png](https://assets.buttondown.email/images/f5da457a-5f23-4b1a-a24b-e395f826ab10.png?w=960&fit=max)


值得注意的是，这个 32B 的开源权重模型在每个基准测试中都完全击败了 GPT-4o 和 Claude 3.5 Sonnet。

对 QwQ 进行分类是一项尴尬的任务：它在采样时间扩展（sampling time scaling）方面做了一些模糊的尝试，足以[让 /r/localLlama 感到兴奋](https://www.reddit.com/r/LocalLLaMA/comments/1h1c691/qwq_reflect_deeply_on_the_boundaries_of_the/)：


![image.png](https://assets.buttondown.email/images/edb7d8eb-6402-42e0-b064-2a4f7d495254.png?w=960&fit=max)


但[模型权重本身](https://x.com/gazorp5/status/1861883506055606567)显示它看起来像是一个 Qwen 32B 模型（可能是 Qwen 2.5，[我们的报道在此](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)），所以也许它只是经过微调，学会了“花时间思考、质疑和反思”，“仔细检查工作并从错误中学习”。“这种仔细反思和自我质疑的过程在解决复杂问题方面取得了显著突破”。所有这些都是模糊的 ChatGPT 式描述，并不构成技术报告，但该模型是真实、在线且可下载的，这说明了很多问题。开源的“推理轨迹”（reasoning traces）展示了它是如何被调整以进行顺序搜索（sequential search）的：


![image.png](https://assets.buttondown.email/images/6e367ece-86e3-4c03-b18f-144a9320d46e.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/5375c269-58a3-4403-ab5b-24f8a7916f6e.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/172ef6bc-a32f-4029-87fc-68be2d57ab67.png?w=960&fit=max)


更完整的技术报告即将发布，但如果结果属实，这确实令人印象深刻……也许真正的讽刺在于 Reflection 70B（[我们的报道在此](https://x.com/swyx/status/1832234771973583220)）并没有错，只是生不逢时……

---

**[由 SambaNova 赞助]** 推理（Inference）正迅速成为 AI 系统的主导功能，取代了模型训练。是时候开始使用专为该任务构建的处理器了。SambaNova 的 RDU 在速度和灵活性方面比 GPU 具有[一些独特的优势](https://shortclick.link/lk96sw)。

> **swyx 的评论**：[RDU](https://shortclick.link/lk96sw) 回归了！如果像 QwQ 这样简单的 32B 自回归（autoregressive）LLM 就能击败 4o 和 3.5 Sonnet，那对于替代计算供应商来说是非常好的消息，他们可以对这种标准模型架构进行极致优化，实现快得惊人且廉价的推理。

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

**主题 1. Hugging Face 与模型部署**

- **Hugging Face 在 CPU 上的推理端点**：[@ggerganov](https://twitter.com/ggerganov/status/1861813652208107597) 宣布 Hugging Face 现在支持在 CPU 服务器上部署由 llama.cpp 驱动的实例，这标志着**向更广泛的低成本云端 LLM 可用性迈进了一步**。
- [@VikParuchuri](https://twitter.com/VikParuchuri/status/1861840948369453256) 分享了 **Marker v1** 的发布，该工具速度**快了 2 倍**且准确度更高，预示着 AI 模型部署基础设施的进步。
- **Agentic RAG 进展**：[@dair_ai](https://twitter.com/dair_ai/status/1861788821970362396) 讨论了 **Agentic RAG**，强调了其在构建利用**外部工具**增强响应准确性的稳健 RAG 系统中的效用。
  - 讨论重点介绍了将更**先进的 LLM 链和向量库 (vector stores)** 集成到系统中的策略，旨在提高 AI 响应的精度。

**主题 2. 开源 AI 势头**

- **热门 AI 模型讨论**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1861841502579859499) 指出 @bfl_ml 的 **Flux** 成为 Hugging Face 上最受欢迎的模型，表明使用重点正从 LLM 转向多模态模型。
  - 对话包含了关于**图像、视频、音频和生物模型使用量增加**的见解，显示出企业环境中对多样化 AI 模型更广泛的接受和集成。

- **SmolLM 招聘活动**：[@LoubnaBenAllal1](https://twitter.com/LoubnaBenAllal1/status/1861698912437821750) 宣布了 SmolLM 的实习机会，专注于 **LLM 训练**和**数据集策划**，突显了对开发小型、高效模型日益增长的需求。

**主题 3. NVIDIA 与 CUDA 进展**

- **CUDA Graphs 与 PyTorch 增强**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1861615852094710095) 赞扬了 **CUDA graphs** 的效率，主张通过**单进程多 GPU (single-process-multi-GPU)** 程序简化 PyTorch 中的流程，以提高计算速度和生产力。
  - 对话涉及了优化 PyTorch 中 **DataParallel** 以利用**单进程效率**的建议。

- **Torch Distributed 与 NVLink 讨论**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1861610288899822023) 质疑了 **torch.distributed** 在 NVIDIA GB200 NVL72 方面的能力，强调了处理多 GPU 时的复杂性和注意事项。

**主题 4. 风投惯例的影响与 AI 行业洞察**

- **风险投资批评与机遇**：[@saranormous](https://twitter.com/saranormous/status/1861645773152366880) 批评了一些风投的不理性决策，主张建立更强大的伙伴关系以保护创始人。
- [@marktenenholtz](https://twitter.com/marktenenholtz/status/1861843799246299410) 分享了对应届毕业生**高期望**以及管理初级程序员挑战的看法，暗示了全行业对人才可持续增长的担忧。

- **AI 中的 Stripe 与 POS 系统**：[@marktenenholtz](https://twitter.com/marktenenholtz/status/1861578678926290974) 强调了 **Stripe** 在提高业务数据质量方面的作用，指出 **POS 系统** 提供高投资回报率 (ROI)，对于利用 AI 进行数据采集的企业至关重要。

**主题 5. 多模态模型开发**

- **ShowUI 发布**：[@_akhaliq](https://twitter.com/_akhaliq/status/1861654585296769473) 讨论了 ShowUI，这是一个设计为 GUI 视觉 Agent 的视觉-语言-动作模型，标志着将 AI 集成到交互式应用中的趋势。
  - [@multimodalart](https://twitter.com/multimodalart/status/1861811807813308926) 赞扬了 **QwenVL-Flux** 的功能，它增加了图像变体和风格迁移等功能，增强了多模态 AI 应用。

**主题 6. 梗与幽默**

- **幽默与 AI 文化**：[@swyx](https://twitter.com/swyx/status/1861627477774475561) 以幽默的方式看待 AI 发展，并提到了 **Neuralink** 和 AI 爱好者文化。
  - [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1861766885773279718) 分享了一个关于 AI 的机智虚构场景，探讨了过去的玩具在当今蓬勃发展的 AI 景观中可能如何被重新构思。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. QwQ-32B：Qwen 的新推理模型媲美 O1-Preview**

- **[QwQ: "深思未知的边界" - 似乎是带有 Test-Time Scaling 的 Qwen](https://qwenlm.github.io/blog/qwq-32b-preview/)** ([Score: 216, Comments: 84](https://reddit.com/r/LocalLLaMA/comments/1h1c691/qwq_reflect_deeply_on_the_boundaries_of_the/)): **Qwen** 宣布预览发布其名为 **QwQ** 的新 **32B** 语言模型，该模型似乎实现了 **test-time scaling**。仅从标题来看，该模型的重点似乎在于推理能力和探索知识边界，尽管在没有额外上下文的情况下无法确定具体功能。
  - 在 [HuggingFace](https://huggingface.co/spaces/Qwen/QwQ-32B-preview) 上的初步测试表明，**QwQ-32B** 的表现与 **OpenAI 的 O1 preview** 相当，用户注意到其推理过程非常冗长。目前已提供多个**量化版本**，包括适用于 **24GB VRAM** 的 **Q4_K_M** 和适用于 **16GB VRAM** 的 **Q3_K_S**。
  - 用户报告该模型在推理时显著地“**话多**”，在 temperature=0 时，对于简单问题使用了多达 **3,846 tokens**（相比之下 O1 为 **1,472**）。该模型需要特定的 **system prompt**，其中提到 "You are Qwen developed by Alibaba" 以获得最佳性能。
  - 技术测试显示，该模型在处理诸如“**爱丽丝的姐妹**”等复杂推理问题时表现强劲，但在遵循严格输出格式方面存在一些局限。用户注意到某些内容可能存在**审查问题**，且与其他推理模型相比性能表现不一。


- **Qwen Reasoning Model????? QwQ??** ([Score: 52, Comments: 9](https://reddit.com/r/LocalLLaMA/comments/1h1dlrw/qwen_reasoning_model_qwq/)): 该帖子似乎在询问 **QwQ** 的发布情况，这似乎与 **Qwen Reasoning Model** 有关，但未提供具体细节或背景。帖子包含一张截图，但没有关于模型功能或发布时间的额外背景信息。
  - **QwQ-32B-Preview** 模型已在 [Hugging Face](https://huggingface.co/Qwen/QwQ-32B-Preview) 上发布，并在其[博客文章](https://qwenlm.github.io/blog/qwq-32b-preview/)中进行了详细说明。初步测试显示其与其他模型相比具有强劲的性能。
  - 用户报告称，这款 **32B 参数开源模型** 的表现与 **O1-preview** 相当，表明开源语言模型取得了重大进展。该模型展示了特别强大的推理能力。
  - 该模型由 **Qwen** 发布，可立即进行测试，早期用户在与现有模型的对比测试中报告了积极的结果。


**主题 2. Qwen2.5-Coder-32B AWQ 量化优于其他方法**

- **Qwen2.5-Coder-32B-Instruct-AWQ: 使用 OptiLLM 和 Aider 进行基准测试** ([Score: 58, Comments: 16](https://reddit.com/r/LocalLLaMA/comments/1h0sf96/qwen25coder32binstructawq_benchmarking_with/)): **Qwen2.5-Coder-32B-Instruct** 模型在 **2x3090 GPU** 上使用 **AWQ** 量化进行了基准测试，测试了包括 **Best of N Sampling**、**Chain of Code** 以及通过 **Aider** 和 **OptiLLM** 实现的不同编辑格式在内的各种配置。在使用 "whole" 编辑格式和 temperature **0.2** 时，达到了 **74.6%** 的峰值 **pass@2** 分数。测试显示 **AWQ_marlin** 量化优于普通 AWQ，而 "whole" 编辑格式的表现始终优于 "diff" 格式，且较低的 temperature 设置（**0.2** 对比 **0.7**）产生了更高的成功率，尽管 **chain-of-code** 和 **best-of-n** 技术虽然减少了错误，但对整体成功率的影响微乎其微。
  - **VRAM 使用情况**和 **temperature=0** 的测试是社区关注的重点，用户请求在 temperature 为 **0**、**0.05** 和 **0.1** 以及各种 **topk** 设置（**20**、**50**、**100**、**200**、**1000**）下进行额外基准测试。
  - **AWQ_marlin** 量化模型可在 [Huggingface](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct-AWQ) 获取，并可以使用 **SgLang** 运行时通过参数 *"--quantization awq_marlin"* 启用。
  - 用户建议测试 **min_p sampling**，推荐参数为 **temperature=0.2**、**top_p=1**、**min_p=0.9** 和 **num_keep=256**，并引用了关于其优势的相关 [Hugging Face 讨论](https://github.com/huggingface/transformers/issues/27670)。

- **Qwen2.5-Coder-32B-Instruct - 使用几天后的评测** ([Score: 86, Comments: 87](https://reddit.com/r/LocalLLaMA/comments/1h0w3te/qwen25coder32binstruct_a_review_after_several/)): **Qwen2.5-Coder-32B-Instruct** 模型在搭载 **Oobabooga WebUI** 的 **3090 GPU** 上进行了测试，结果显示出明显的局限性，包括在缺乏信息时编造回答、提供错误的代码审查建议，以及在处理 protobuf 实现或跨会话维持上下文等复杂任务时失败。尽管有这些缺点，该模型在编写 **Doxygen** 注释方面表现出色，在用户验证的前提下能提供有价值的代码审查反馈，并可作为改进代码的有效交流对象，对于能够批判性评估其输出的开发者来说是一个有用的工具。
  - 用户强调原帖使用的是 **4-bit 量化模型**，这会显著损害性能。多位专家建议至少使用 **6-bit** 或 **8-bit 精度** 的 **GGUF** 格式，并配合适当的 **GPU offloading** 以获得最佳效果。
  - **Mistral Large** 被称赞在编程任务上优于 **GPT/Claude**，用户提到了其 **10 亿 token 限制**、免费访问以及更好的代码生成能力。据观察，与竞争对手相比，该模型能生成更准确、可编译的代码。
  - 几位用户强调了正确的 **system prompts** 和 **sampling parameters** 对获得最佳结果的重要性，并分享了诸如[此指南](https://www.reddit.com/r/LocalLLaMA/comments/1gpwrq1/how_to_use_qwen25coderinstruct_without/)之类的资源。**Unsloth "fixed" 128K 模型** 配合 **Q5_K_M 量化** 被推荐作为替代方案。


**Theme 3. 32B 模型推理的性价比硬件配置**

- **运行 32B 模型最便宜的硬件** ([Score: 63, Comments: 107](https://reddit.com/r/LocalLLaMA/comments/1h12cmq/cheapest_hardware_go_run_32b_models/)): 讨论了 **32B 语言模型** 的运行需求，重点在于如何将模型完全放入 GPU RAM 中并实现 **>20 tokens/second** 的性能。帖子对比了 **NVIDIA GPU**，指出单块 **RTX 3090** 只能处理 **Q4 量化**，同时探索了双 **RTX 3060** 显卡与价格约 **1200 欧元** 的二手双 **3090** 方案的优劣。
  - 用户讨论了替代硬件方案，包括售价 **$90-300** 的 **Tesla P40** GPU（运行 **72B 模型** 速度为 **6-7 tokens/second**），以及双 **RTX 3060** 在运行 **Qwen 2.5 32B** 模型时达到 **13-14 tokens/second**。
  - 一个使用 **exllama v2** 的显著配置展示了在单块 **RTX 3090** 上运行 **32B 模型**，采用 **5 bits per weight**、**32K context** 和 **Q6 cache**，并开启了 flash attention 和 cache 量化。
  - 性能对比显示，**RTX 4090** 处理 Prompt 的速度比 **M3 Max** 快 **15.74 倍**，而 **Intel Arc A770s** 被建议作为具有更高显存带宽的预算选项，但存在软件兼容性问题。


**Theme 4. NVIDIA Star-Attention：长序列处理速度提升 11 倍**

- **[GitHub - NVIDIA/Star-Attention: Efficient LLM Inference over Long Sequences](https://github.com/NVIDIA/Star-Attention)** ([Score: 51, Comments: 4](https://reddit.com/r/LocalLLaMA/comments/1h0vjg6/github_nvidiastarattention_efficient_llm/)): **NVIDIA** 在 [GitHub](https://github.com/NVIDIA/Star-Attention) 上发布了 **Star-Attention**，这是一种用于处理 **Large Language Models** 长序列的新型注意力机制。该项目旨在提高处理超长文本序列时 **LLM 推理** 的效率。
  - **Star-Attention** 通过将注意力计算分散到多台机器上，在保持 **95-100%** 准确度的同时，实现了内存占用和推理时间高达 **11 倍** 的缩减。该机制采用了包含 blockwise-local 和 sequence-global 注意力阶段的**两阶段分块稀疏近似 (two-phase block-sparse approximation)**。
  - **SageAttention2** 被建议作为单机注意力优化的更好替代方案，而 **Star-Attention** 主要对大型计算集群和分布式系统有益。
  - 用户指出，这是 **NVIDIA** 的一种策略，通过支持跨多台机器的分布式计算来处理注意力机制，从而鼓励用户购买更多显卡。


## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**Theme 1. Claude 获得 Model Context Protocol 以实现直接系统访问**

- **MCP 对我来说感觉像是 Level 3** ([Score: 32, Comments: 23](https://reddit.com/r/ClaudeAI/comments/1h0uzfo/mcp_feels_like_level_3_to_me/))：**Claude** 展示了自主编码能力，通过独立编写和执行 **Python** 代码，绕过传统的 API 集成方法，利用环境文件夹访问 **Bing API**。这一发现表明，通过直接代码执行，扩展工具使用和 **agent swarms** 具有潜力，使 **AI 系统** 能够通过编写和运行自己的程序以更高的自主性运行。
  - **Claude desktop app** 中的 **MCP** 功能实现了无需传统 API 集成的自主代码执行。可以通过桌面应用配置菜单中的 **developer settings** 进行访问。
  - 用户强调了 **Claude** 如何独立创建和执行工具来完成它无法立即完成的任务，包括抓取文档和通过自定义解决方案获取信息。
  - 虽然自主编码能力很强大，但人们也对未来可能的限制以及为了防止环境/数据问题而进行适当 **sandboxing** 的需求提出了担忧。目前有一个 **Python library** 可用于实现自定义服务器和工具。


- **[Model Context Protocol 是我想要的一切](https://i.redd.it/4328d93nqg3e1.png)** ([Score: 28, Comments: 12](https://reddit.com/r/ClaudeAI/comments/1h1701e/model_context_protocol_is_everything_ive_wanted/))：**Model Context Protocol (MCP)** 使 **Claude** 能够直接与计算机系统和 API 交互，代表了 AI 系统集成方面的重大进步。
  - 用户 **cyanheads** 创建了一个 [实现 MCP 工具的指南](https://github.com/cyanheads/ModelContextProtocol-Tools/blob/main/guides/creating_mcp_tools.md)，展示了仅需 **$3** API 成本的协议实现。
  - 带有 **MCP** 的 **Claude desktop app** 通过注入系统提示词（system prompt）的工具函数，实现了与互联网访问、计算机系统和智能家居设备的原生集成。该功能通过一个 [天气工具示例](https://ibb.co/QcYK6QP) 得到了演示。
  - 实际演示包括 **Claude** 将 **Pac-Man 游戏** 写入本地磁盘，并利用互联网研究能力创建了一篇关于 **quantum computing breakthroughs** 的博客文章，展示了该系统在文件系统交互和网络研究方面的多功能性。


**主题 2. ChatGPT Voice 实现 15% 的冷启动电话转化率**

- **我让 ChatGPT 帮我打销售电话……它真的在成交** ([Score: 144, Comments: 67](https://reddit.com/r/ChatGPT/comments/1h1f9vp/i_got_chatgpt_to_make_sales_calls_for_me_and_its/))：作者尝试将 **ChatGPT's voice mode** 用于房地产冷启动电话（cold calls），在 **100 次通话** 中实现了 **12-15% 的有意义对话率** 和 **2-3% 的会议预订率**，显著优于其 **3-4% 对话率** 的手动尝试。成功源于 **AI 的预先披露** 和新奇因素，潜在客户出于对技术的好奇而在通话中停留更长时间，其中一个案例促成了 **签署合同**，同时 AI 保持了一贯的专业精神并能有效处理拒绝。
  - 该系统每月约需 **$840** 的 **OpenAI API** 费用，在目前的 **Tier 3** 访问权限下可处理 **21 路并发通话**，并有潜力在 **Tier 5** 扩展到 **500 路并发通话**。后端实现使用 **Twilio** 和 **websockets**，代码量约为 **5000 行**。
  - 针对 **TCPA 合规性** 提出了法律担忧，截至 **2024 年**，违反 **Do Not Call Registry** 的规定可能面临 **每路违规 $50,120 的罚款**。用户指出，**AI 通话需要事先获得明确的书面同意**，以避免集体诉讼。
  - **新奇因素** 被强调为一个关键但暂时的优势，预计 **AI 冷启动电话** 将在几个月内广泛应用于销售领域。一些用户建议创建 **AI receptionists** 来应对预期中增加的 AI 冷启动电话。


**主题 3. OpenAI 获得软银 15 亿美元投资并推动军事合同**

- **[OpenAI 获得来自 SoftBank 的 15 亿美元新投资，允许员工通过要约收购出售股份](https://www.cnbc.com/2024/11/26/openai-gets-1point5-billion-investment-from-softbank-in-tender-offer.html)** ([Score: 108, Comments: 2](https://reddit.com/r/OpenAI/comments/1h1951w/openai_gets_new_15_billion_investment_from/)): **OpenAI** 通过要约收购获得了来自 **SoftBank** 的 **15 亿美元投资**，使员工能够出售其股份。继 **2023** 年早些时候与 **Microsoft** 达成重大交易后，此次投资延续了 **OpenAI** 强劲的融资势头。
  - **SoftBank** 的参与引起了评论者的担忧，因为其**投资记录充满争议**，特别是最近一些面临重大挑战的高知名度科技投资。


- **[AI 公司（从 Meta 到 OpenAI）的新一轮“圈地运动”是军事合同](https://fortune.com/2024/11/27/ai-companies-meta-llama-openai-google-us-defense-military-contracts/)** ([Score: 111, Comments: 6](https://reddit.com/r/OpenAI/comments/1h1bxno/the_new_land_grab_for_ai_companies_from_meta_to/)): **Meta**、**OpenAI** 和其他 **AI 公司**正将 **美国军事和国防合同** 视为新的收入来源和市场扩张机会。帖子正文未提供更多背景或具体细节。
  - **政府支出**和**军事合同**被视为科技公司可靠的收入来源，评论者指出这遵循了将业务增长与国防资金挂钩的历史模式。
  - 社区对 **AI 公司**与**军事应用**合作表示担忧和怀疑，并提到了 **Skynet** 以及对安全影响的质疑。
  - 讨论强调这些合同是由**纳税人的钱**资助的，暗示公众在这些 AI 发展中拥有利益。


**主题 4. Blender 的本地 LLaMa-Mesh 集成发布**

- **[Blender 的本地 LLaMa-Mesh 集成刚刚发布！](https://i.redd.it/ic1zaxexgi3e1.gif)** ([Score: 81, Comments: 7](https://reddit.com/r/StableDiffusion/comments/1h1fb5r/local_integration_of_llamamesh_in_blender_just/)): **Blender** 的 **LLaMa-Mesh AI** 集成已发布，实现了本地处理能力。帖子缺少关于功能、实现细节或下载信息的额外详情。
  - **LLaMa-Mesh AI** 项目已在 [HuggingFace](https://github.com/huggingface/meshgen) 上发布，供初步发布和测试。
  - 用户对该工具的潜力表示乐观，特别是与现有的**扩散模型 (diffusion model)** 网格生成方法相比。


---

# AI Discord 摘要

> 由 O1-preview 提供的摘要之摘要的总结

**主题 1. AI 模型在效率和性能上取得新突破**

- [**Deepseek 在推理基准测试中超越 OpenAI**](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas): **Deepseek 的 R1 模型**在多个推理基准测试中超越了 **OpenAI 的 o1**，标志着 AI 领导地位的转变。在 **High-Flyer** 雄厚的计算资源（包括估计 **5 万张 Hopper GPU**）支持下，Deepseek 计划开源模型并提供具有竞争力的 API 定价。
- [**OLMo 2 在 AI 对决中超越开源模型**](https://allenai.org/blog/olmo2): **OLMo 2** 推出了新的 **7B 和 13B 模型**，其表现优于其他开源模型，特别是在**递归推理**和复杂的 AI 应用中。**高管们**对 OLMo 2 的能力议论纷纷，反映了 AI 技术的尖端进步。
- [**MH-MoE 模型在不增加体积的情况下提升 AI 效率**](https://arxiv.org/abs/2411.16205): **MH-MoE** 论文提出了一种多头机制，在匹配稀疏 MoE 模型性能的同时超越了标准实现。令人印象深刻的是，它与 **BitNet** 等 **1-bit LLMs** 兼容，扩展了其在低精度 AI 环境中的实用性。

**主题 2. AI 工具和基础设施升级**

- [**TinyCloud 携 GPU 大军发布——54x 7900XTXs**](https://x.com/__tinygrad__/status/1861645755452363011)：**Tinygrad** 宣布即将推出 **TinyCloud**，计划在年底前通过 **9 台 tinybox reds** 为贡献者提供 **54 个 GPU** 的访问权限。凭借确保稳定性的定制驱动程序，用户只需使用 API key 即可轻松调用这些算力——无需复杂操作，只有纯粹的 GPU 动力。
- [**MAX Engine 凭借全新 Graph APIs 展现实力**](https://www.modular.com/blog/max-24-3-introducing-max-engine-extensibility)：**MAX 24.3** 版本引入了可扩展性，允许通过 **MAX Graph APIs** 创建自定义模型。旨在实现**低延迟、高吞吐量的推理**，MAX 优化了基于 ONNX 等格式的实时 AI 工作负载——这是对 AI 基础设施领域的一次有力冲击。
- [**Sonnet 现在支持读取 PDF 了——没开玩笑！**](https://aider.chat/docs/faq.html#why-is-the-llm-speaking-to-me-in-an-unexpected-language)：**Aider** 中的 **Sonnet** 现在支持读取 PDF，使其成为开发者更全能的助手。用户反馈该功能运行顺畅，称其“能有效解析 PDF 文件”并提升了工作流——这绝对是生产力的巨大飞跃。

**主题 3：AI 社区应对伦理困境**

- [**Bsky 数据集风波让研究人员陷入困境**](https://clearsky.app/)：一名 Hugging Face 员工创建的 **Bluesky 帖子**数据集在遭到强烈抵制后被撤下，尽管其符合服务条款。此次撤除阻碍了**社交媒体研究**，对小型研究人员打击最大，而大型实验室则未受影响——这是典型的“小人物吃亏”案例。
- [**Sora 视频生成器泄露引发 AI 界骚动**](https://youtu.be/qh5Eis0sBl4?si=OgB66SVZyB6sPMCi)：OpenAI 的 **Sora Video Generator** 泄露引发了热烈讨论，一段 [YouTube 视频](https://youtu.be/qh5Eis0sBl4?si=OgB66SVZyB6sPMCi) 爆出了内幕。社区成员剖析了该工具的性能，并批评了迅速撤销公共访问权限的行为——AI 圈总是少不了戏剧性事件。

**主题 4：用户应对 AI 工具的成长烦恼**

- [**Cursor Agent 陷入无尽文件夹乱象**](https://forum.cursor.com/t/cursor-composer-window-in-latest-version/30611)：用户报告称 **Cursor Agent** 正在生成无尽的文件夹，而不是进行合理的组织——这简直是目录界的“衔尾蛇”。许多人建议提供更清晰的指令来纠正 Agent 的行为——没人喜欢混乱的工作空间。
- [**Jamba 1.5 Mini 在函数调用上出差错**](https://openrouter.ai/docs)：通过 **OpenRouter** 使用的 **Jamba 1.5 mini 模型**在处理 function calls 时返回空响应。虽然在不使用函数调用时表现正常，但用户们正困惑地检查代码——function calling 不应该是黑盒。
- **Prompting 研究让工程师陷入两难**：参与者对缺乏一致的**实证性 prompting 研究**表示沮丧，相互矛盾的研究让情况变得扑朔迷离。社区正在寻求有效的 prompting 技术的明确指导——标准化刻不容缓。

**主题 5：AI 行业的巨额资金与动态**

- [**PlayAI 融资 2100 万美元助力语音 AI**](https://blog.play.ai/blog/21m-funding)：**PlayAI** 从 **Kindred Ventures** 和 **Y Combinator** 获得了 **2100 万美元**，用于开发直观的语音 AI 界面。这笔资金注入旨在增强**语音优先界面**，让机人交互像爵士乐独奏一样流畅。
- [**生成式 AI 支出高达 138 亿美元**](https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/)：**生成式 AI** 支出在 2024 年飙升至 **138 亿美元**，标志着从 AI 幻想向现实应用的转变。企业正在慷慨解囊，但决策者仍在努力思考如何获得最大的投资回报。
- [**SmolVLM 以小身躯掀起大波澜**](https://x.com/andi_marafioti/status/1861437314351632662?s=46)：**SmolVLM** 作为一个 **2B VLM**，为**端侧推理（on-device inference）**树立了新标准，在 GPU RAM 占用和 token 吞吐量方面优于竞争对手。它可以在 **Google Colab** 上进行微调，让强大的 AI 在消费级硬件上触手可及——小模型，大能量。

---

# 第一部分：Discord 高层级摘要

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX Engine 在 AI 推理方面优于 ONNX**：**MAX Engine** 被定义为一个旨在利用模型的 AI 推理强力引擎，而 **ONNX** 作为一个模型格式，支持包括 [ONNX](https://onnx.ai)、[TorchScript](https://pytorch.org/docs/stable/jit.html) 和原生 Mojo 图在内的多种格式。
  
  - **MAX** 旨在跨多种硬件提供**低延迟、高吞吐量的推理**，相比 ONNX 侧重于模型传输，它更优化了实时模型推理。
- **内存压力导致 Mojo 测试执行变慢**：在 Mojo 目录中运行所有测试的时间明显长于运行单个测试，这表明多个 [mojo-test-executors](https://github.com/modularml/mojo/issues/3815) 带来了潜在的内存压力。
  
  - 逐步注释掉测试函数可以改善运行时间，这表明**内存泄漏**或**高内存占用**导致了速度变慢。
- **MAX 24.3 版本增强了引擎功能**：最近发布的 [MAX 24.3 版本](https://www.modular.com/blog/max-24-3-introducing-max-engine-extensibility) 强调了其可扩展性，允许用户通过 **MAX Graph APIs** 创建自定义模型。
  
  - Graph API 促进了 Mojo 中**高性能符号计算图**的实现，使 MAX 的定位超越了单纯的模型格式。
- **Mojo 增强内存管理策略**：讨论显示，**Chrome 的内存占用**导致了 Mojo 测试执行期间的性能问题，凸显了整体系统的内存压力。
  
  - 用户承认需要更好的硬件来处理工作负载，并强调了有效**资源管理**的必要性。
- **Mojo 的 Origin 追踪提升了内存安全性**：关于 Mojo 中 Origin 追踪的讨论透露，目前 `vec` 及其元素共享相同的 Origin，但新的实现将引入独立的 Origin，以实现更精确的 **aliasing**（别名）控制。
  
  - 引入两个 Origin（向量的 Origin 和其元素的 Origin）将通过准确追踪潜在的 **aliasing** 场景来确保**内存安全**。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Agent 的文件夹故障**：用户报告称 **Cursor Agent** 正在创建无穷无尽的文件夹，而不是正确地组织它们，这表明存在潜在的 Bug。他们建议提供更清晰的命令可以增强 Agent 功能，并防止在管理项目结构时产生混淆。
  
  - 这些问题在 [Cursor 论坛](https://forum.cursor.com/t/cursor-composer-window-in-latest-version/30611)上得到了广泛讨论，强调了提高命令清晰度以简化文件夹组织的必要性。
- **Cursor v0.43.5 功能变更**：最新的 **Cursor 0.43.5 版本**更新引发了关于功能缺失的讨论，包括 @web 功能的更改以及标签页（tabs）系统的移除。尽管有这些变化，一些用户仍然赞赏 Cursor 对其生产力的持续贡献。
  
  - 详细的反馈可以在 [Cursor Changelog](https://changelog.cursor.com/) 中找到，用户在其中表达了对新功能集的担忧和赞赏。
- **Model Context Protocol 集成**：用户对在 Cursor 中实现 **Model Context Protocol (MCP)** 表现出浓厚兴趣，以便创建自定义工具和上下文提供者。用户认为 MCP 可以通过提供更多定制化功能来极大提升整体用户体验。
  
  - 关于 MCP 潜力的讨论在 [Cursor 文档](https://docs.codeium.com/windsurf/cascade)中得到了强调，突出了其在扩展 Cursor 能力方面的作用。
- **Cursor 增强开发者工作流**：许多用户分享了使用 **Cursor** 的积极体验，强调了它如何改变了他们的编码和项目执行方式。得益于 Cursor，即使是非专业开发者在处理雄心勃勃的项目时也感到更有信心。
  
  - 这些令人振奋的体验经常在 [Cursor 论坛](https://forum.cursor.com/t/woah-what-happened-to-cursor-with-this-new-update/30407)中被提及，展示了 Cursor 对开发工作流的影响。
- **Cursor 中的 Markdown 格式挑战**：用户讨论了 Cursor 中的 **Markdown 格式问题**，特别是代码块无法正确应用的问题。他们正在寻求解决方案或快捷键绑定，以增强工作流并在最新更新中更好地管理文件。
  
  - 针对这些 Markdown 问题的解决方案和变通方法在多个[论坛帖子](https://forum.cursor.com/t/cursor-composer-window-in-latest-version/30611)中进行了探讨，表明对改进 Markdown 支持的持续需求。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora Video Generator 泄露引发讨论**：最近 OpenAI 的 **Sora Video Generator** 泄露事件引发了热议，并提到了一个详细介绍该事件的 [YouTube 视频](https://youtu.be/qh5Eis0sBl4?si=OgB66SVZyB6sPMCi)。
  
  - 社区成员讨论了该工具的性能以及泄露后立即撤销公共访问权限的问题。
- **ChatGPT 的图像分析显示出不同的结果**：用户实验了 **ChatGPT 的图像分析**能力，注意到基于 Prompt 结构和图像呈现方式的不一致性。
  
  - 反馈强调，与无引导的交互相比，提供图像上下文时的交互效果更好。
- **经验性 Prompting 研究缺乏共识**：参与者对**经验性 Prompting 研究**的稀缺表示沮丧，理由是研究结果相互矛盾且缺乏标准化方法。
  
  - 在众多相互矛盾的论文中，社区寻求关于有效 Prompting 技术的明确指导。
- **AI Phone Agents 旨在实现类人交互**：讨论强调 **AI Phone Agents** 应该模拟人类交互，而不是模仿脚本化的 IVR 响应。
  
  - 成员们强调了 AI 在通话过程中理解上下文并验证关键信息的重要性。
- **开发全面的模型测试框架**：工程师们讨论了如何创建强大的**模型测试框架**，能够大规模评估模型并跟踪 Prompt 的更改。
  
  - 建议包括实施验证系统，以确保在不同用例中的响应准确性和一致性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **MH-MoE 提高模型效率**：[MH-MoE](https://arxiv.org/abs/2411.16205) 论文详细介绍了一种多头机制，该机制聚合来自不同专家空间的信息，实现了与稀疏 MoE 模型相当的性能，同时超越了标准实现。
  
  - 此外，它与 BitNet 等 **1-bit LLMs** 兼容，扩展了其在低精度环境下的适用性。
- **Star Attention 增强 LLM 推理**：[Star Attention](https://arxiv.org/abs/2411.17116) 引入了一种块稀疏注意力机制，可将长序列的推理时间和内存使用量减少高达 **11 倍**，同时保持 **95-100% 的准确率**。
  
  - 该机制与基于 Transformer 的 LLMs 无缝集成，促进了长序列任务的高效处理。
- **DALL-E 的变分界争议**：关于 [DALL-E 论文](https://arxiv.org/abs/2102.12092)中提出的变分界（variational bound）存在持续争议，有人断言由于对条件独立性的错误假设，该变分界可能存在缺陷。
  
  - 参与者正在审查这些潜在的疏忽是否会影响模型中提出的不等式的有效性。
- **Qwen 的开源权重推理模型**：新发布的 **Qwen 推理模型**被认为是第一个能够执行高级推理任务的大型开源权重模型，可通过量化至 4 bits 来实现。
  
  - 然而，一些参与者对之前被归类为推理模型的模型表示怀疑，并引用了其他展示推理能力的方法。
- **OLMo 的增长与性能**：自 2024 年 2 月以来，**OLMo-0424** 在下游性能方面表现出显著改进，特别是通过比其前代产品[提升性能](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d)。
  
  - 增长生态系统也见证了来自 LLM360 的 Amber 和 M-A-P 的 Neo 模型等项目的贡献，增强了模型开发的开放性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RWKV-7 实现 soft AGI 的潜力**：开发者表示希望推迟 **RWKV-8** 的开发，认为 **RWKV-7** 的能力足以作为 soft AGI 的候选者，并承认仍需进一步改进。
  
  - 他们强调虽然 **RWKV-7** 很稳定，但在过渡到 **RWKV-8** 之前仍有可以增强的领域。
- **Mamba 2 架构的效率**：社区对新的 **Mamba 2 架构**感到好奇，特别是其效率方面以及与现有模型的对比。
  
  - 重点关注 **Mamba 2** 是否允许更多的 tensor parallelism，或者是否比传统架构更具优势。
- **SSMs 与图曲率的关系**：参与者讨论了 **SSMs (State Space Models)** 在消息传递（message passing）方面的潜力，强调了图中**顶点度 (vertex degree)** 与流形上**曲率 (curvature)** 之间的类比。
  
  - 他们指出，较高的顶点度与负曲率相关，展示了高斯曲率（Gaussian curvature）的离散版本。
- **RunPod 服务器的 OpenAI 端点配置**：一位用户发现 **RunPod 教程**没有明确提到 **OpenAI completions** 的正确端点，而这是成功通信所必需的。
  
  - 为了使其工作，请使用 [此端点](https://%7BPOD_ID%7D-11434.proxy.runpod.net/v1/chat/completions) 并在请求中提供模型。

 

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash 1.5 容量提升**：OpenRouter 对 [Gemini Flash 1.5](https://openrouter.ai/docs/provider-routing) 的容量进行了**重大提升**，解决了用户反馈的 rate limiting 问题。鼓励遇到问题的用户重试请求。
  
  - 这一增强预计将通过增加整体系统容量，显著改善高流量期间的用户体验。
- **提供商路由优化**：该平台现在正将**呈指数级增长的流量**路由到成本最低的提供商，确保用户平均能享受到更低的价格。更多细节可以在 [提供商路由文档](https://openrouter.ai/docs/provider-routing) 中找到。
  
  - 该策略通过在必要时回退到其他提供商的机制来维持性能，从而优化成本效益。
- **Grok Vision Beta 发布**：OpenRouter 正在增加 **Grok Vision Beta** 的容量，鼓励用户在 [Grok Vision Beta](https://openrouter.ai/x-ai/grok-vision-beta) 进行测试。
  
  - 此次发布为用户提供了在服务扩展过程中探索增强型视觉能力的机会。
- **EVA Qwen2.5 价格翻倍**：用户观察到 **EVA Qwen2.5 72B** 的价格翻了一倍，引发了关于这一变化是促销性质还是标准涨价的疑问。
  
  - 推测认为，价格调整可能是由竞争加剧和不断变化的业务策略驱动的。
- **Jamba 1.5 模型问题**：用户报告了来自 AI21 Labs 的 **Jamba 1.5 mini 模型**的问题，具体表现为在调用函数时收到空响应。
  
  - 尽管尝试了不同版本，问题依然存在，导致用户推测这可能与消息准备或后端挑战有关。

 

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **QwQ-32B-Preview 推动 AI 推理**：[QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) 是一款增强 **AI 推理能力** 的实验性模型，尽管它在 **语言混杂** 和 **递归推理循环** 方面仍存在挑战。
  
  - 尽管存在这些问题，多位成员对其在 **数学** 和 **编程** 任务中的表现表示兴奋。
- **Olmo 在一致性上超越 Llama**：根据社区讨论，**Olmo 模型** 在各项任务中表现出一致的性能，使其与 **Llama** 区别开来。
  
  - 此外，据称 **Tülu** 在特定提示词下的表现优于 Olmo 2，引发了关于领先模型的辩论。
- **低比特量化优化训练不足的 LLM**：一项关于 [Low-Bit Quantization](https://arxiv.org/abs/2411.17691) 的研究显示，**低比特量化** 有利于训练不足的大语言模型，且模型规模越大，性能退化越小。
  
  - 预测表明，使用超过 **100 万亿 token** 训练的模型可能会遭遇量化性能问题。
- **Deepseek 在推理基准测试中超越 OpenAI**：**Deepseek 的 R1 模型** 在多个推理基准测试中超越了 **OpenAI 的 o1**，凸显了这家初创公司在 AI 领域的潜力。
  
  - 在 **High-Flyer** 雄厚的算力资源支持下（包括估计 **5 万张 Hopper GPU**），Deepseek 准备开源模型并在中国启动极具竞争力的 API 定价。
- **Bsky 数据集风波干扰研究**：一名 HF 员工创建的 *Bluesky 帖子数据集* 在遭到强烈抵制后被删除，尽管该数据集符合 **ToS**。
  
  - 此次删除使 **社交媒体研究** 受到挫折，讨论集中在对小型研究人员的影响以及即将发布的 [数据集](https://clearsky.app/) 上。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 引擎弃用**：关于 **Perplexity 引擎弃用** 的讨论出现，用户对未来的支持和功能更新表示担忧。
  
  - 参与者建议转向 **Exa** 或 **Brave** 等替代方案，以维持项目的稳定性和连续性。
- **Perplexity Pro 中的图像生成**：**Perplexity Pro** 现在支持图像生成，但用户注意到对输出的控制有限，使其更适合偶尔使用而非精细项目。
  
  - 目前没有专门的图像生成页面，这是提升用户体验的一个反馈点。
- **增强的模型选择优势**：Perplexity 订阅者可以使用 **Sonnet**、**4o** 和 **Grok 2** 等先进模型，这些模型显著提升了编程和数学计算等复杂任务的性能。
  
  - 虽然免费版本能满足基本需求，但订阅版为需要更强大功能的用户提供了实质性优势。
- **Perplexity API 金融数据源**：有关于 Perplexity API 所使用的 **金融数据源** 及其为内部项目集成 **股票代码数据** 能力的咨询。
  
  - 引用了一张截图来展示可用的金融信息类型，强调了该 API 在实时数据应用中的实用性。
- **Perplexity API 中的 Reddit 引用支持**：用户注意到 **Perplexity API 不再支持 Reddit 引用**，并质疑这一变化的深层原因。
  
  - 这一改动影响了那些依赖 Reddit 作为数据源的用户，引发了关于替代引用方法的讨论。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonnet 引入 PDF 支持**：Sonnet 现在支持读取 PDF，通过命令 `aider --install-main-branch` 启用。用户反馈 **Sonnet** 能够通过新功能有效解析 PDF 文件。
  
  - 成功使用 PDF 读取功能的成员分享了正面反馈，这增强了他们在 Aider 中的工作流。
- **QwQ 模型面临性能障碍**：多位用户在 [glhf.chat](https://glhf.chat) 上对 QwQ 模型进行基准测试时遇到了 **网关超时错误**。
  
  - 正如社区所讨论的，**QwQ** 模型的加载延迟正在影响其在性能测试中的响应速度。
- **为隐私实现本地 Whisper API**：一位成员分享了他们在 **Apple M4 Mac mini** 上托管本地 **Whisper API** 进行语音转录的配置，重点关注隐私保护。
  
  - 他们提供了 `curl` 命令示例，并将 API 托管在 [Whisper.cpp Server](https://api.ailocal.org) 上供社区测试。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Axolotl 与 Unsloth 框架对比**：一位成员对比了 **Axolotl** 和 **Unsloth**，强调 **Axolotl** 可能显得臃肿，而 **Unsloth** 提供了更精简的代码库。[Instruction Tuning – Axolotl](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/inst_tune.html)。
  
  - **数据集质量**被强调为优于框架选择，一位用户指出性能很大程度上取决于数据集的质量，而非底层框架。
- **RTX 3090 价格波动**：讨论显示 **RTX 3090** 的价格差异巨大，平均价格在 **1500 美元**左右，但有些挂牌价低至 **550 美元**。
  
  - 这种价格差异表明市场波动剧烈，成员们对如此广泛的价格区间感到惊讶。
- **GPU 托管方案与 Docker**：成员们表示更倾向于选择拥有可扩展 GPU（可按分钟开启）的 **24GB GPU 托管商**，并强调运行 **Docker 容器**以减少 SSH 依赖。
  
  - 对 **Docker 容器**的关注反映了通过容器化实现高效、灵活的 GPU 资源管理的趋势。
- **使用多 GPU 微调本地模型**：用户寻求关于使用 JSON 数据文件微调本地 **Llama 3.2** 模型的建议，并对支持 **multi-GPU** 以增强性能表现出浓厚兴趣。
  
  - 虽然 **multi-GPU** 支持仍在开发中，但已向有兴趣参与的成员开放了限量 Beta 测试。
- **理解等式中的运算顺序**：一位成员复习了**运算顺序**以纠正一个等式，确认乘法优先于加法，从而得出表达式 **1 + 2*3 + 4*5 + 6*7 + 8*9 = 141**（而非 479）。
  
  - 该修正遵循 PEMDAS 规则，指出了原始等式结构中的误解。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **通配符（Wildcard）定义之争**：参与者讨论了编程中“Wildcards”的不同解释，对比了 [Civitai](https://civitai.com) 和 Python 环境下的语境，并引用了 [Merriam-Webster 定义](https://www.merriam-webster.com/dictionary/wild%20card)。
  
  - 术语使用的差异受到**编程历史**的影响，导致在不同场景下如何正确应用通配符存在多种观点。
- **图像生成工作流挑战**：一位用户报告称，尽管使用了合适的 Prompt 和风格指南，但在生成一致的角色图像方面仍存在困难，正在寻求工作流优化方案。
  
  - 建议包括尝试 [image-to-image generation](https://civitai.com) 方法，并利用 **Civitai** 等平台上的现有工作流来增强一致性。
- **ControlNet 在 Large Turbo 模型上的性能**：一位成员确认了 **ControlNet** 功能在 3.5 **large turbo** 模型上的有效性，引发了关于其与新模型兼容性的讨论。
  
  - 这激发了成员们对使用最新模型版本时的性能指标和潜在集成挑战的兴趣。
- **创建高质量图像**：讨论强调了**时间**、**探索**和 **Prompt 实验**在创作高质量角色肖像中的重要性。
  
  - 建议用户参考成功的 [workflows](https://civitai.com)，并考虑使用多种技术来提升图像输出质量。
- **Stable Diffusion 插件问题**：过时的 Stable Diffusion 扩展导致了 Checkpoint 兼容性问题，促使一位用户寻求插件更新。
  
  - 社区建议检查插件仓库以获取最新更新，并提到一些用户在尝试排障后仍然遇到问题。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的讽刺性测试**：一位成员正在 [NotebookLM](https://www.youtube.com/watch?v=e-OipxzsqtU&t=2s) 中**尝试讽刺性文章**，并指出该模型大约有一半时间能识别出笑话，而另一半时间则失败。
  
  - 他们专门指示模型*质疑作者的人性*，从而产生了 [YouTube 视频](https://www.youtube.com/watch?v=e-OipxzsqtU&t=2s)中展示的幽默结果。
- **AI 语音与视频展示**：**AI 在一段新[视频](https://youtu.be/ttDOBb5NYiQ?feature=shared)中展示了其语音和视频能力**，幽默地强调了手指的缺失。
  
  - 该成员鼓励爱好者、怀疑者和好奇者探索[英文版](https://youtu.be/ttDOBb5NYiQ?feature=shared)和[德文版](https://youtu.be/iZci0WpAmGY?feature=shared)视频中呈现的令人兴奋的可能性。
- **Gemini 模型评估**：一位成员在 [Google AI Studio](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2216h7Ioo6fmgSMtCWoSWBx8vK-kHEjsGHo%22%5D,%22action%22:%22open%22,%22userId%22:%22105185943804239990679%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing) 中进行了**两个 Gemini 模型的对比**，并在 NotebookLM 中总结了发现。
  
  - 分享了来自 NotebookLM 的配套音频概述，希望这些见解能传达给 Google 的开发团队。
- **NotebookLM 功能问题**：用户报告了 NotebookLM 的**功能问题**，例如表示访问困难的黄色渐变警告以及 AI 性能不稳定的情况。
  
  - 用户担心现有的聊天会话可能会丢失，以及 NotebookLM 当前聊天系统的临时性。
- **播客时长挑战**：一位用户观察到，尽管有缩短长度的自定义指令，生成的播客时长始终在 **20 分钟**左右。
  
  - 建议指示 AI 为忙碌的受众创建更简洁的内容，以便更好地管理播客时长。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 与 LiteLLM 的集成**：用户在将 **Cohere API** 与 **LiteLLM** 集成时遇到了困难，特别是关于**引用（citations）功能**无法按预期工作的问题。
  
  - 他们强调 LiteLLM 作为一个元库（meta library）与多个 LLM 供应商对接，并请求 Cohere 团队协助改进此集成。
- **增强 LiteLLM 中的引用支持**：目前的 **LiteLLM** 实现缺乏对 Cohere 聊天端点返回的引用的支持，限制了其可用性。
  
  - 用户建议在 LiteLLM 代码中添加新参数来处理引用，并表示愿意贡献代码或等待维护者的回复。
- **全栈 AI 工程专业知识**：一位成员强调了其作为**全栈 AI 工程师**的角色，拥有超过 **6 年**设计和部署可扩展 Web 应用程序及 AI 驱动解决方案的经验，使用的技术包括 **React**、**Angular**、**Django** 和 **FastAPI**。
  
  - 他们详细介绍了自己在 **Docker**、**Kubernetes** 以及跨 **AWS**、**GCP** 和 **Azure** 等云平台的 CI/CD 流水线方面的技能，并分享了他们的 GitHub 仓库：[AIXerum](https://github.com/AIXerum)。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **PlayAI 融资 2100 万美元以增强语音 AI**：PlayAI 已从 [Kindred Ventures](https://blog.play.ai/blog/21m-funding) 和 [Y Combinator](https://x.com/dps/status/1861413927856546187?s=46) 获得了 **2100 万美元** 的资金，用于为开发者和企业开发直观的语音 AI 接口。
  
  - 此次资本注入旨在改进**无缝的语音优先接口**，重点是增强人机交互，将其作为一种自然的通信媒介。
- **OLMo 2 性能超越开源替代方案**：[OLMo 2](https://x.com/allen_ai/status/1861511421064028646?s=46) 推出了全新的 **7B 和 13B 模型**，其性能超越了其他开源模型，特别是在**递归推理**和各种 AI 应用方面。
  
  - **高管们**对 OLMo 2 的潜力感到兴奋，理由是它在处理复杂的 **AI 场景**方面表现出色，反映了 AI 技术的最新进展。
- **SmolVLM 为端侧 VLM 设定新标准**：**SmolVLM**（一个 **2B VLM**）已经发布，旨在实现**端侧推理**，在 **GPU RAM 占用**和 **token 吞吐量**方面均优于竞争对手。
  
  - 该模型支持在 [Google Colab](https://x.com/andi_marafioti/status/1861437314351632662?s=46) 上进行微调，并专门针对需要在消费级硬件上进行高效处理的使用场景进行了优化。
- **Deepseek 模型在推理方面超越 OpenAI**：**Deepseek** 最近的 AI 模型在推理基准测试中表现优于 **OpenAI**，引起了 AI 社区的关注。
  
  - 在**中国对冲基金幻方量化（High-Flyer）**的支持下，Deepseek 致力于构建基础技术并提供**价格亲民的 API**。
- **2024 年企业生成式 AI 支出达到 138 亿美元**：**生成式 AI** 支出在 2024 年飙升至 **138 亿美元**，表明企业内部正从实验阶段转向执行阶段。
  
  - 尽管对更广泛的采用持乐观态度，但决策者在定义有效的**实施策略**方面仍面临挑战。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LoLCATs 线性 LLM 吞吐量减半**：[LoLCATs 论文](https://arxiv.org/abs/2410.10254)探讨了在不进行全模型微调的情况下线性化 LLM，结果显示在小 batch size 下，其**吞吐量**比 **FA2** 降低了 50%。
  
  - 尽管**节省了内存**，但线性化模型并未超越此前预期的**二次复杂度注意力模型**，这引发了对其整体效率的质疑。
- **ThunderKittens 发布 FP8 支持**：正如其[博客文章](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)所述，**ThunderKittens** 引入了 **FP8 支持**和 **fp8 kernel**，仅用 **95 行代码**就实现了 **1500 TFLOPS**。
  
  - 团队强调通过简化 kernel 编写来推进新架构的研究，实现细节可在其 [GitHub 仓库](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/FP8)中找到。
- **FLOPS 计数工具提高准确性**：统计 **FLOPS** 面临挑战，因为现有脚本会遗漏许多操作，从而影响研究的可靠性；建议使用 [fvcore](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) 和 [torch_flops](https://github.com/zugexiaodui/torch_flops) 等工具进行精确测量。
  
  - 准确的 FLOPS 计数对于验证模型性能至关重要，社区倡导采用这些增强工具以减少差异。
- **cublaslt 加速大矩阵运算**：**cublaslt** 被认为是管理**低精度**大矩阵的最快选择，在矩阵运算中表现出令人印象深刻的速度。
  
  - 这种优化对于旨在提升大规模矩阵计算性能的 AI 工程师特别有益。
- **LLM Coder 增强 Claude 集成**：[LLM Coder 项目](https://github.com/msaroufim/llm_coder)旨在通过在 prompt 中提供主要 API 并将其集成到 **VS Code** 中，来提高 **Claude** 对库的理解。
  
  - 该倡议寻求提供更准确的代码建议，并邀请开发者表达兴趣并为其开发做出贡献。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 集成 Azure OpenAI 端点**：LlamaParse 现在支持 [Azure OpenAI 端点](https://twitter.com/llama_index/status/1861550505761349761)，在确保企业级安全性的同时，增强了其解析复杂文档格式的能力。
  
  - 这一集成允许用户通过定制的 API 端点，在应用程序中有效地管理敏感数据。
- **CXL 内存提升 RAG 流水线性能**：来自 [MemVerge](https://twitter.com/llama_index/status/1861825056621600995) 的研究表明，利用 **CXL 内存**可以显著扩展 RAG 应用程序的可用内存，从而实现全内存运行。
  
  - 这一进展预计将提升检索增强生成（RAG）系统的性能和可扩展性。
- **使用 LlamaIndex 构建质量感知型文档聊天机器人**：通过结合用于文档摄取与检索的 **LlamaIndex** 和用于监控的 **aimon_ai**，开发者可以构建一个具有质量感知能力的文档聊天机器人，主动检查幻觉等问题。
  
  - LlamaIndex 利用 **milvus** 作为向量存储，确保聊天机器人能够进行高效且有力的数据检索。
- **MSIgnite 展示 LlamaParse 和 LlamaCloud 功能**：在 **#MSIgnite** 期间，Farzad Sunavala 和 @seldo 在分组会议中发布了关于 [LlamaParse 和 LlamaCloud](https://twitter.com/llama_index/status/1861887478602592715) 的重大公告。
  
  - 演示重点展示了**多模态解析**，突出了 LlamaParse 处理各种文档格式的能力。
- **BM25 检索器与 Postgres 的兼容性**：一位成员询问关于构建 **BM25 检索器**并将其存储在 **Postgres 数据库**中的问题，得到的回复建议需要一个 **BM25 扩展**来实现此功能。
  
  - 这种集成将在 Postgres 管理的环境中实现更高效的检索过程。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyCloud 发布，搭载 54 块 7900XTX GPU**：即将发布的 **TinyCloud** 令人期待，到今年年底，它将包含 **9 台 tinybox red**，共配备 **54 块 7900XTX GPU**，供在 tinygrad 中使用 CLOUD=1 的贡献者使用。
  
  - 来自 [tiny corp 的推文](https://x.com/__tinygrad__/status/1861645755452363011)确认，由于使用了**定制驱动程序**，该设置将非常稳定，确保用户只需一个 API key 即可简便使用。
- **招聘云基础设施和 FPGA 专家**：**GeorgeHotz** 强调需要聘请一名**全职云基础设施开发人员**来推进 TinyCloud 项目，并征集有意贡献的 **FPGA 后端**专家。
  
  - 他强调这是支持开发并有效维护项目基础设施的必要条件。
- **流片就绪：支持 Qualcomm DSP 和 Google TPU**：根据 GeorgeHotz 的概述，**流片（tapeout）就绪的前置条件**包括从 tinygrad 中移除 **LLVM**，并添加对 **Qualcomm DSP** 和 **Google TPU** 的支持。
  
  - 此外，目前的重点还包括开发 **tinybox FPGA 版本**，目标是实现**自主 AMD 技术栈**。
- **tinygrad 中的 GPU 基数排序优化**：用户讨论了通过支持 **64 位**、**负数**和**浮点数**来增强 **GPU 基数排序（radix sort）**，并强调通过分块（chunking）来提高大数组的性能。
  
  - 一个相关的[示例](https://github.com/tinygrad/tinygrad/blob/84f96e48a1bb8826d868ad19ea34ce2deb019ce1/examples/stunning_mnist.py#L29-L31)展示了使用 **UOp.range()** 来优化 Python 循环。
- **排序算法的向量化技术**：参与者探索了潜在的**向量化技术**，以优化排序算法中迭代数字并更新排序输出的部分。
  
  - 其中一个建议包括使用直方图为每个位置预填充一个常量张量，以实现更高效的赋值。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 提交数突破 1000 次**：一名成员祝贺团队在 Torchtune 主仓库完成了**第 1000 次 commit**，这体现了项目团队的专注与投入。
  
  - 附带的图片展示了这一里程碑，突显了团队的辛勤工作和承诺。
- **由 Torchtune 驱动的教育聊天机器人**：一位用户正在利用 OpenAI assistants 开发一款专注于问答（QA）和网络安全的教育聊天机器人，并寻求关于 **Torchtune 兼容性**和微调流程的指导。
  
  - **Torchtune** 强调开源模型，需要获取模型权重（model weights）才能进行有效的微调。
- **提升 LoRA 训练性能**：成员们讨论了 **LoRA 单设备 recipes** 的性能，询问了训练速度和收敛时间。
  
  - 一位成员指出，*将学习率提高 10 倍可以改善训练性能*，这暗示了一个潜在的优化路径。
- **通过 Activation Offloading 提升内存效率**：对话涉及了 DPO 中的 **activation offloading**，成员们表示并未观察到显著的内存收益。
  
  - 一位成员幽默地表达了困惑，同时在寻求一个可能澄清这些问题的公开 PR。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter OS 模式 vs 普通模式**：**Open Interpreter** 现在通过 CLI 提供**普通模式**，并通过具有 GUI 功能的 **OS 模式**提供服务，后者需要多个模型进行控制。
  
  - 一位用户强调，他们打算将 OS 模式用作通用的网页爬虫，并专注于 CLI 应用。
- **Open Interpreter Point API 问题**：用户报告称 **Open Interpreter Point API** 似乎已宕机，并持续出现错误。
  
  - 这些困难引起了社区对该 API 可靠性的担忧。
- **围绕 MCP 工具的兴奋**：新的 **MCP tool** 引起了巨大的反响，成员们将其描述为“疯狂的”且**非常巨大（really HUGE）**。
  
  - 这反映了社区对探索其功能的兴趣日益增长。
- **MCP 工具集成与速查表**：成员们分享了已安装的 MCP 服务器和工具列表，包括 **Filesystem**、**Brave Search**、**SQLite** 和 **PostgreSQL**。
  
  - 此外，还提供了**速查表（cheatsheets）**以帮助最大限度地利用 MCP，强调了社区共享资源的重要性。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **SmolLM2-1.7B 发布**：社区对 [SmolLM2-1.7B](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU) 的发布表现出极大的热情，强调了其对 **LLM 任务前端开发**的影响。
  
  - “这太疯狂了！”一位成员评论道，强调了开发者在**可访问性和能力**方面的转变。
- **Transformers.js v3 发布**：Hugging Face 宣布发布 **Transformers.js v3**，引入了 [WebGPU 支持](https://huggingface.co/blog/transformersjs-v3)，其性能比 **WASM 快达 100 倍**。
  
  - 该更新包括 **25 个新的示例项目**和 **120 种支持的架构**，为开发者提供了丰富的资源。
- **前端 LLM 集成**：一位成员强调了 **LLM 任务在前端的集成**，这标志着应用开发的一个重要演进。
  
  - 这一进步展示了当今开发者可以利用的日益增强的**能力**。
- **Qwen 2.5 微调配置**：一位用户寻求在 Axolotl 中配置 **Qwen 2.5 模型全量微调**的指导，重点关注影响训练效果的参数。
  
  - 存在关于特定 **unfrozen_parameters** 必要性的疑问，表明需要对模型配置有更深入的理解。
- **模型配置指导**：用户对微调各种模型的**配置设置**表现出浓厚兴趣。
  
  - 一位用户询问如何获取未来模型类似设置过程的指导，突显了持续的**社区学习**。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Feature Store 网络研讨会助力 ML 流水线**：参加由创始人 **Simba Khadder** 主持的 [12 月 3 日太平洋时间上午 8 点的 Feature Store 网络研讨会](https://buff.ly/3OqcG1V)。了解 **Featureform** 和 **Databricks** 如何通过简化 **ML** 生态系统中的 **Feature Store** 类型，来促进大规模数据流水线的管理。
  
  - 本次会议将深入探讨处理 **PB 级（petabyte-level）** 数据以及使用 **Apache Iceberg** 实现版本控制，为增强你的 **ML** 项目提供实战见解。
- **GitHub 总部举办 Multi-Agent 训练营**：注册参加 [12 月 4 日](https://lu.ma/multi-agent-meetup)在 **GitHub 总部**举行的 **Multi-Agent Framework Bootcamp**。参与专注于 **Multi-Agent** 系统的专家演讲和工作坊，并获得与行业领袖交流的机会。
  
  - 议程包括由 **Lorenze Jay** 和 **John Gilhuly** 带来的《使用 CrewAI 自动化繁琐事务》以及《通过评估构建生产就绪的 Agents》等环节。
- **为 AI 工程师分享 LLMOps 资源**：一位成员分享了一个 [LLMOps 资源](https://dub.sh/everything-about-llmops)，强调了其三部分结构，并敦促同行将其加入书签以进行全面的 **LLMOps** 学习。
  
  - **Large language models** 正在推动一场变革浪潮，使 **LLMOps** 成为 **AI** 工程中新兴的运营框架。
- **LLMs 彻底改变技术交互**：**Large Language Models (LLMs)** 正在重塑与技术的交互方式，为 **chatbots**、**virtual assistants** 和**高级搜索引擎**提供动力。
  
  - 它们在开发**个性化推荐系统**中的作用标志着行业内运营方法的重大转变。

 

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **使用 Whisper 进行高效音频字幕生成**：一位用户寻求使用 **Whisper** 加快音频数据集字幕生成的方法，但在对短音频文件进行批处理时遇到了问题。
  
  - 他们强调，**Batching**（批处理）将处理时间从 **13 分钟**缩短到了 **1 分钟**，并进一步优化至 **17 秒**。
- **Whisper 批处理优化**：用户讨论了在 **Whisper** 中对短音频文件进行批处理的挑战，旨在提高效率。
  
  - 强调了通过批处理减少处理时间的效果，展示了从 **13 分钟**到 **17 秒**的飞跃。
- **字幕脚本限制**：用户分享了用于 **Whisper** 音频处理的 [字幕脚本](https://cdn.discordapp.com/attachments/823813160075132991/1311340795458224199/test1.py)。
  
  - 他们指出，该脚本在批处理机制中难以高效处理短音频文件。
- **Faster Whisper 集成**：提到了 GitHub 上的 [Faster Whisper](https://github.com/SYSTRAN/faster-whisper)，以利用 **CTranslate2** 提升转录速度。
  
  - 该用户表示打算利用此工具更快速地处理其音频数据集。

 

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL 中的 Llama 3.2 默认提示词**：据用户观察，**Llama 3.2** 模型使用了来自 **BFCL** 的默认系统提示词。
  
  - 这种对标准配置的依赖表明在评估模型性能时具有一致的基准。
- **多轮对话类别影响准确率**：于 **9 月中旬**引入的多轮对话（multi turn）类别导致 **v3** 版本发布后整体准确率明显下降。
  
  - 这些具有挑战性的类别对各种模型的平均得分产生了负面影响，而 **v1** 和 **v2** 基本上未受影响。
- **新指标导致排行榜分数变化**：**10/21** 和 **11/17** 的最新排行榜更新显示，由于引入了针对多轮对话类别的新评估指标，分数出现了显著波动。
  
  - 之前正确的条目现在可能被标记为错误，这突显了旧版状态检查器（state checker）的局限性以及新指标 [PR #733](https://github.com/ShishirPatil/gorilla/pull/733) 的改进。
- **公开发布生成结果**：已宣布计划上传并公开分享用于排行榜检查点的所有生成结果，以方便查看错误日志。
  
  - 这一举措旨在为观察到的不同模型间的 **Agentic** 行为差异提供更深入的见解。
- **多轮对话类别中 Prompting 模型 vs FC**：在多轮对话类别中，**Prompting 模型**的表现往往优于其 **FC (Function Calling)** 对应模型。
  
  - 这一观察结果引发了关于在具有挑战性的评估场景中 Prompting 有效性的讨论。

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **测验分数确认邮件缺失**：一名成员报告在提交后未收到测验分数的**确认邮件**，这引发了对通知系统可靠性的担忧。
  - 作为回应，另一名成员建议检查**垃圾邮件文件夹**或使用不同的电子邮件地址，以确保提交已被正确记录。
- **电子邮件提交故障排除**：为了解决电子邮件问题，一名成员建议验证是否发送了来自 [Google Forms](https://www.google.com/forms/about/) 的**确认邮件**。
  - 他们强调，如果没有收到电子邮件，很可能表明提交未成功，并建议进一步调查表单的配置。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Hidden States Unconference 邀请 AI 创新者**：欢迎参加在旧金山举行的 **Hidden States Unconference**，这是一场研究人员和工程师探索 AI 界面和隐藏状态的聚会，时间为 [12月5日](https://discord.com/events/1089876418936180786/1304175188581290094)。
  - 这一为期一天的活动旨在通过协作讨论推向 AI 方法的边界。
- **宣布构建超轻量级 RAG 应用工作坊**：在 [12月10日](https://discord.com/events/1089876418936180786/1293281470642651269) 即将举行的工作坊中，学习如何使用 sqlite-vec 和 llamafile 配合 Python 创建**检索增强生成 (RAG)** 应用程序。
  - 参与者将体验在没有任何额外依赖项的情况下构建应用程序。
- **ESM-1 生物表征学习启动**：**Paper Reading Club** 将于 [12月12日](https://discord.com/events/1089876418936180786/1305611638979694623) 讨论 Meta AI 的 **ESM-1 蛋白质语言模型**，作为生物表征学习系列的一部分。
  - 本次会议旨在吸引参与者关注 AI 在生物研究中的创新应用。
- **Demo Night 展示湾区创新**：参加 [12月15日](https://discord.com/events/1089876418936180786/1305577025918074890) 的**旧金山 Demo Night**，见证来自 AI 领域当地创作者的突破性演示。
  - 该活动由 **Gen AI Collective** 主办，重点展示技术与创意的交汇。
- **与 Linda Dounia Rebeiz 一起解决数据偏差**：[12月20日](https://discord.com/events/1089876418936180786/1298252973373128725) 加入《时代》周刊百大人物 **Linda Dounia Rebeiz**，了解她使用策划数据集训练无偏见 AI 的方法。
  - 她将讨论使 AI 能够反映现实而非强化偏见的策略。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba 1.5 Mini 集成至 OpenRouter**：一名成员尝试通过 **OpenRouter** 使用 **Jamba 1.5 Mini Model**，并配置了位置和用户名等参数。
  - 然而，与不使用函数调用的成功输出相比，**function calling** 在 JSON 响应的 **content** 字段中返回了空输出。
- **OpenRouter 上的 Function Calling 产生空输出**：当通过 **OpenRouter** 使用 **function calling** 时，**Jamba 1.5 Mini Model** 在 JSON 响应中返回了空的 **content** 字段。
  - 这一问题在不调用 **function calling** 的情况下并未出现，表明该设置存在特定问题。
- **OpenRouter 使用的密码更改请求**：一名成员请求更改与通过 **OpenRouter** 使用 **Jamba 1.5 Mini Model** 相关的**密码**。
  - 他们详细说明了为用户数据管理设置的参数，如位置和用户名。

---

**HuggingFace Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第二部分：按频道详细摘要与链接

{% if medium == 'web' %}

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1311095112129974333) (435 条消息🔥🔥🔥):

> `Mojo 性能问题，Mojo 中的内存管理，可变别名（Mutable aliasing）与 Origin，函数签名与 Origin`

- **Mojo 测试的性能问题**：在 Mojo 目录中运行所有测试比单独运行它们耗时显著更长，这指向了由于生成了大量 `mojo-test-executors` 而导致的潜在内存压力。
  
  - 一位用户注意到，在注释掉测试函数后，运行时间逐渐改善，这表明内存泄漏或高内存占用可能是导致变慢的原因。
- **内存与进程管理**：讨论强调了 Chrome 的内存占用对测试执行期间性能问题的影响，表明了整体系统的内存压力。
  
  - 用户承认需要更好的硬件来处理工作负载，强调了有效资源管理的必要性。
- **理解 Mojo 中的 Origin**：关于 Origin 如何在 Mojo 中跟踪引用的讨论展开，揭示了在当前模型中，`vec` 及其元素共享相同的 Origin，但这将在新实现中发生变化。
  
  - 将引入两个 Origin：Vector 的 Origin 和其元素的 Origin，从而允许更精确地跟踪潜在的别名（aliasing）并确保内存安全。
- **函数签名中的可变别名**：小组讨论了如何通过参数化 Origin 值在函数签名中安全地实现可变别名，这允许不同的参数具有相同的 Origin。
  
  - 这允许函数在保持内存安全的同时修改 Vector 元素，遵循与 Rust 别名规则相同的逻辑。
- **Mojo 类型系统的未来方向**：Nick 提出了一个处理引用和 Origin 的模型，该模型强调局部推理（local reasoning）同时允许可变别名，这需要未来的研究和安全性的形式化证明。
  
  - 该模型旨在表征并防止可能导致悬空指针（dangling pointers）的情况，确保 Mojo 设计中稳健的内存安全。

**提到的链接**：

- [Subtyping and Variance - The Rust Reference](https://doc.rust-lang.org/reference/subtyping.html#variance)：未找到描述
- [Subtyping and Variance - The Rustonomicon](https://doc.rust-lang.org/nomicon/subtyping.html#variance)：未找到描述
- [Issues · modularml/mojo](https://github.com/modularml/mojo/issues/3815)：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 做出贡献。
- [0738-variance - The Rust RFC Book](https://rust-lang.github.io/rfcs/0738-variance.html#why-variance-is-good)：未找到描述
- [Compiler Explorer - C++ (Circle) (Latest)](https://godbolt.org/z/czPPhsKj7)：关于安全性的 #feature。`template<typename T>void assign(T^ input, T val) { *input = val;}int main() { std2::string_view hello("hello"); { std2::string world("world");...`
- [[BUG] Can't use returned reference from function · Issue #3813 · modularml/mojo](https://github.com/modularml/mojo/issues/3813#issuecomment-2501980044)：Bug 描述 `struct Foo: var a: Int32 fn __del__(owned elf): print("Destroyed Foo") fn __init__(inout self): self.a = 1 print("Created Foo") fn min(ref [_] a: Foo, ref [_] b: Fo...`
- [Compiler Explorer - C++ (Circle) (Latest)](https://godbolt.org/z/Mox694js9)：关于安全性的 #feature。`int main() safe { std2::vector<std2::string_view> vec { }; { std2::string s("Hello world"); mut vec.push_back(s); } // println(vec[0]);}`
- [Compiler Explorer - C++ (x86-64 clang (trunk))](https://godbolt.org/z/79rzof1E9)：`/*fn assign<T>(input: &mut T, val: T) { *input = val;}fn main() { let mut hello: &'static str = "hello"; { let world = String::from("world&...`
- [[docs] Stdlib insider documentation by owenhilyard · Pull Request #3793 · modularml/mojo](https://github.com/modularml/mojo/pull/3793)：开发标准库的人员需要更多关于 API 契约以及 Runtime 和 Compiler builtins 行为的信息，以便能够编写正确且高性能的代码...

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1311169515907846195) (4 messages):

> `MAX Engine vs ONNX, MAX Graph API, AI Model Inference`

- **MAX Engine 被定义为 AI 推理的强大引擎**：MAX 被描述为旨在利用模型的引擎，而 ONNX 仅作为模型格式，支持 ONNX、TorchScript 和原生 Mojo graphs 等多种格式。
  
  - 它的目标是在不同硬件上提供 **低延迟、高吞吐量的推理**，以简化 AI 工作负载。
- **MAX 24.3 版本增强了引擎能力**：最近发布的 [MAX 24.3 版本](https://www.modular.com/blog/max-24-3-introducing-max-engine-extensibility) 强调了其可扩展性，使用户能够通过 MAX Graph APIs 创建自定义模型。
  
  - Graph API 促进了 Mojo 中高性能符号计算图的实现，使 MAX 不仅仅是一个模型格式。
- **ONNX 专注于模型传输，MAX 优先考虑性能**：与 ONNX 旨在传输模型的目标不同。
  
  - 这一区别表明了 MAX 优化实时模型推理的目标。

 

**提到的链接**：[Get started with MAX Graph | Modular Docs](https://docs.modular.com/max/tutorials/get-started-with-max-graph)：了解如何使用我们的 Mojo API 构建模型图，以便使用 MAX Engine 进行推理。

 

---

### **Cursor IDE ▷ #**[**general**](https://discord.com/channels/1074847526655643750/1074847527708393565/1311074052638511124) (332 messages🔥🔥):

> `Cursor Agent Performance, Cursor Version Updates, User Experiences with Cursor, Model Context Protocol (MCP), Markdown Issues and Bug Fixes`

- **用户对 Cursor Agent 的不满**：一些用户报告了 Cursor Agent 的问题，例如它会创建无穷无尽的文件夹而不是正确地组织它们，这表明可能存在 Bug。
  
  - 用户建议提供更清晰的指令可以改进 Agent 功能，并防止在与项目结构交互时产生混淆。
- **最新版本发布与功能**：Cursor 的最新更新（版本 0.43.5）引发了关于缺失功能的讨论，包括 @web 功能的变化以及标签页系统的移除。
  
  - 尽管有这些更新，一些用户仍对 Cursor 在提高生产力和高效管理编码任务方面的贡献表示赞赏。
- **实现自定义工具和协议**：用户对 Model Context Protocol (MCP) 及其在 Cursor 内创建自定义工具和上下文提供者的潜力表示了兴趣。
  
  - 讨论表明，此类实现可以极大地增强用户体验并提供更量身定制的功能。
- **Cursor 对开发的积极影响**：许多用户分享了他们使用 Cursor 的愉快经历，强调了它如何改变了他们的编码和项目执行方式。
  
  - 用户强调，尽管他们不是专业开发人员，但 Cursor 使他们能够更有信心地应对雄心勃勃的项目。
- **Markdown 问题与解决方法**：有关于 Cursor 中 Markdown 格式问题的讨论，特别是关于代码块无法正确应用的问题。
  
  - 用户正在寻求解决方案或快捷键绑定，以在最新更新中增强工作流并更好地管理文件。

**提到的链接**：

- [Cascade - Codeium Docs](https://docs.codeium.com/windsurf/cascade)：无描述
- [Cursor - The IDE designed to pair-program with AI.](https://changelog.cursor.com/)：无描述
- [Tweet from echo.hive (@hive_echo)](https://x.com/hive_echo/status/1861533909827400120)：2 个 Cursor Composer Agent 在同一个项目上协作。一个构建项目，另一个在第一个 Agent 完成后进行审查并编写报告。通过 comment 中的 cursor rules 文件来实现...
- [Woah what happened to cursor with this new update?](https://forum.cursor.com/t/woah-what-happened-to-cursor-with-this-new-update/30407)：我再也找不到 200k 上下文选项了，不知为何我所做的所有更改都被撤销了？AI 聊天中的滚动很奇怪——如果我在代码文本中按 shift+up，整个聊天窗口会滚动...
- [Shopify Sub App Intergration](https://docs.google.com/document/d/1WH54-xFc-qYxD3vcJM9J9demCBiXVTRaQo_tVqxWmnY/edit?tab=t.smx6oz4yhusn#heading=h.q5mnh4vhlbs6)：计划将我们的订阅应用与 Seal subscriptions（订阅应用）集成。我们自定义构建了自己的订阅应用（每月收费、更改产品、跳过一个月等）...
- [Cursor Composer Window in latest version](https://forum.cursor.com/t/cursor-composer-window-in-latest-version/30611)：嗨！在最新版本 0.43.4 中，CMD + SHIFT + I 无法在我喜欢的那个新 Composer 窗口中打开 Composer。这是故意的吗？它确实可以在聊天窗口中通过 CMD + I 打开...

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1311130112288231495) (95 条消息🔥🔥):

> `Sora Video Generator Leak, ChatGPT Image Analysis Capabilities, Discord Community Engagement, Translation Challenges, Usage of AI in Content Creation`

- **Sora 视频生成器泄露引发热议**：成员们讨论了最近 OpenAI 的 **Sora Video Generator** 泄露事件，并引用了一段详细介绍该事件的 [YouTube 视频](https://youtu.be/qh5Eis0sBl4?si=OgB66SVZyB6sPMCi)。
  - 讨论内容包括对该工具效能的看法以及未达预期的遗憾，一些人指出公共访问权限已经被撤销。
- **测试 ChatGPT 的图像分析功能**：用户测试了 ChatGPT 分析图像的能力，注意到根据 Prompt 的构建方式和图像呈现方式的不同，回答存在不一致性。
  - 参与者强调了过去的某些功能，即与没有任何引导的新对话相比，提供图像上下文可以带来更好的交互效果。
- **Discord 中的互动与幽默**：频道内展示了用户之间关于每日主题的轻松互动，一些人幽默地建议为经常获胜的人提供奖励。
  - 诸如“ChatGPT 是一种猫系 AI”之类的评论展示了社区内讨论的趣味性。
- **翻译请求中的挑战**：一位用户展示了他们将一段**加拿大英语文本**翻译成乌尔都语的结果，并分享了公共访问链接。
  - 这体现了频道用户热衷于分享教育资源的协作精神。
- **关于 AI 能力的反馈**：参与者分享了使用 ChatGPT 识别角色并根据视觉解读创建回复的经验。
  - 一位使用 Free Plan 的用户报告了有效的识别效果，分享了关于不同订阅计划下能力差异的对比观察。

**提到的链接**：[Public Access to Open AI's Sora Video Generator just Leaked...](https://youtu.be/qh5Eis0sBl4?si=OgB66SVZyB6sPMCi)：在这段视频中，我讨论了 OpenAI 的 Sora（一种先进的 AI 视频生成工具）意外泄露的事件。据报道，此次泄露是由抗议的艺术家发起的……

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1311204625323655212) (85 条消息🔥🔥):

> `Accessing ChatGPT for Free, User Experience with ChatGPT, ChatGPT's Reliability and Validity, Using Files with GPT, Community Interaction and Humor`

- **用户寻求免费访问 ChatGPT Plus 的方法**：多位用户讨论了无需支付升级费用即可访问 **ChatGPT 4** 的潜在方法，包括使用多个账号或通过朋友获取订阅的建议。
  - 然而，结论是无限制地使用该平台通常需要付费计划，这让许多人觉得不值得进一步探索。
- **社区幽默与支持**：在整个讨论过程中，成员们建立了一种幽默的关系，开着关于如何使用 ChatGPT 的玩笑，同时鼓励喜剧天赋。
  - 评论范围从关于鬼魂的轻松调侃到嘲讽那些关于不付钱就想糊弄过去的严肃询问。
- **ChatGPT 对选民欺诈讨论的处理方式**：一位用户担心，尽管缺乏关于该话题的实证同行评审研究，但 ChatGPT 传达了关于选民欺诈的偏见观点。
  - 另一位成员警告不要将 ChatGPT 视为此类问题的权威，并提醒社区不要讨论政治。
- **GPT 文件交互的最佳实践**：一位用户就如何让 GPT 参考提供的文件而不是依赖其通用知识寻求建议。
  - 建议包括在 Prompt 中保持具体，并直接指示 GPT 分析给定的报告。
- **关于 ChatGPT 能力的讨论**：社区成员强调，虽然像 ChatGPT 这样的 LLM 能力强大，但应将其视为文本生成器，而非确定的真理来源。
  - 强调了核实事实的重要性以及 AI 在提供可靠信息方面的局限性，确保用户对输出保持批判性态度。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1311193849691115571) (62 条消息🔥🔥):

> `Empirical Prompting Research, AI Phone Calls, Model Testing Frameworks, Error Identification in AI, General Problem-Solving Strategies`

- **缺乏经验性 Prompting 研究**：一位成员对 Prompting 技术缺乏经验性研究表示沮丧，指出许多现有研究似乎相互矛盾，未能建立黄金标准。
  
  - 另一名成员建议，虽然存在大量论文，但它们往往相互矛盾，使得社区在寻求有效 Prompting 的清晰度方面感到困惑。
- **AI 电话 Agent vs IVR 系统**：讨论强调了 AI Agent 与传统 IVR 系统之间的区别，强调 AI 应该模拟人类交互，而不仅仅是模仿脚本化的响应。
  
  - 成员们一致认为，向用户确认关键信息以确保准确性非常重要，就像人工操作员所做的那样。
- **构建稳健的模型测试框架**：一位成员对创建一个能够大规模评估模型的测试框架表示担忧，特别是在跟踪 Prompt 更改及其对响应的影响方面。
  
  - 建议包括根据现有信息验证提取的数据，并在交互过程中实施检查以向用户确认准确性。
- **模型行为与准确性测试**：成员们讨论了明确 Prompt 的必要性，同时也理解某些指令可能被模型正确推断而无需详细解释。
  
  - 反馈循环机制和基于观察到的行为迭代改进指令被视为增强模型性能的关键策略。
- **利用检索增强生成 (RAG)**：一位成员询问了鼓励 GPT 引用所提供文件的最佳实践，从而引出了使用检索增强生成 (RAG) 进行上下文检索的建议。
  
  - 这种方法将允许模型根据查询利用文件中的特定数据，从而增强其响应的相关性。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1311193849691115571) (62 条消息🔥🔥):

> `Empirical Prompting Research, AI Phone Call Agents, Testing and Consistency in AI, IVR vs AI Interaction, RAG for File Referencing`

- **Prompting 缺乏经验性研究**：人们对目前缺乏经验性 Prompting 研究表示担忧，现有论文往往呈现矛盾的信息，且对有效 Prompting 的指导有限。
  
  - 一位成员指出，由于交互的复杂性以及 Prompt 被解释的不同方式，为 AI Agent 创建单元测试 (unit tests) 非常困难。
- **AI 电话 Agent 应该模仿人类**：讨论强调 AI 电话 Agent 不应被视为传统的 IVR，而是旨在实现更无缝的人机交互，让 AI 对对话做出动态反应。
  
  - 成员们强调了引导 AI 理解上下文并提取关键信息的重要性，而不是依赖它正确推断每个细节。
- **构建稳健的测试框架**：创建能够监控 Prompt 更改如何影响模型性能的测试框架，对于确保 AI 在众多用例中的可靠性至关重要。
  
  - 建议设计允许轻松测试的系统，并保持对模型正确处理各种信息类型能力的检查。
- **通过引导赋能 AI**：成员们讨论了在提供明确指令的同时允许模型推断更简单任务的必要性，确保其专注于容易出错的领域。
  
  - 这种量身定制的方法在将模型适应特定用例时允许更个性化的体验，因此通过测试进行观察和学习非常重要。
- **使用 RAG 进行上下文引用**：一个建议是利用 RAG (Retrieval-Augmented Generation) 来引导 AI 引用特定文件，从而提高响应的准确性。
  
  - 这种方法鼓励模型在回答查询时从指定的知识库中提取信息，有效地将提供的文件集成到其响应机制中。

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1311082931170574406) (85 条消息🔥🔥):

> `OLMo 模型更新、GPU 租赁经验、Nous Hermes 模型对比、Qwen 推理模型发布、加密货币诈骗问题`

- **OLMo 的增长与性能**：自 2024 年 2 月首次发布以来，OLMo-0424 在下游任务性能上有了显著提升，特别是与早期模型相比，[性能大幅增强](https://allenai.org/blog/olmo-1-7-7b-a-24-point-improvement-on-mmlu-92b43f7d269d)。
  
  - 模型开发生态系统的开放性不断提高，包括来自 LLM360 的 Amber 和 M-A-P 的 Neo 模型等项目的显著贡献。
- **GPU 租赁的挑战**：一位用户正在调研用于 AI/ML 项目的 GPU 租赁经验，寻求关于选择服务时的痛点和决策因素的见解。
  
  - 感兴趣的主题包括价格、可用性、性能，以及本地 GPU 与云端租赁对比的直接经验。
- **Nous Hermes 模型性能讨论**：成员们讨论了 **Hermes 3** 在多个 Benchmark 上表现不如 **Llama 3.1** 的性能问题，并强调了潜在的 Overfitting（过拟合）担忧。
  
  - 对话强调了不同模型如何根据模型训练配置表现出不同的性能。
- **Qwen 的开源权重推理模型**：新发布的 Qwen 推理模型被认为是第一个能够执行高级推理任务的重要开源权重（Open Weight）模型，可通过 4 bits 量化实现。
  
  - 参与者对之前模型被归类为推理模型表示怀疑，指出早期模型通过各种方法也展示了一定的推理能力。
- **社区中的加密货币诈骗**：关于个人冒用 Nous Research 之名推广可疑加密货币项目的讨论，引发了对社区内潜在诈骗的担忧。
  
  - 成员们强调了核实身份隶属关系的重要性，以保护社区成员免受欺诈活动侵害。

**提到的链接**：

- [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/)：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD。注：QwQ 的发音为 /kwju:/，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...
- [OLMo 2: The best fully open language model to date | Ai2](https://allenai.org/blog/olmo2)：我们的下一代全开源 Base 和 Instruct 模型处于性能和训练效率的帕累托前沿（Pareto frontier）。

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1311083269948833944) (9 条消息🔥):

> `Test Time Training, ARC Prize, 表格问答`

- **处于停滞状态的 Test Time Training 项目**：一位成员询问是否有任何活跃的 **Test Time Training** 项目，得到的回复是目前这些项目正在**私下**进行。
  
  - 另一位成员建议，随着未来 **r1 paper** 的发布，这种情况可能会发生改变。
- **联系 ARC Prize 参与者**：一位用户表达了交流意愿，并提到他们参与了 **ARC Prize**。
  
  - 有人建议通过私信联系特定用户以进行进一步讨论。
- **表格问答（Table Question Answering）咨询**：一位新用户寻求关于**表格问答**的建议，询问是否有人在该领域有经验。
  
  - 目前没有关于该咨询的回复记录，该问题仍处于开放讨论状态。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1311219957111193670) (93 条消息🔥🔥):

> `MH-MoE 模型效率、Star Attention 机制、DALL-E 变分界问题、金融经济学与贝叶斯统计、算法交易经验`

- **MH-MoE 提升模型效能**：关于 **MH-MoE** 的论文强调，通过使用多头机制聚合多样化的专家信息，可以提高模型性能，同时达到与稀疏 MoE 模型相当的效率。
  
  - 正如最近的讨论所述，它的表现优于标准实现，同时兼容 **1-bit LLMs**（如 BitNet）。
- **Star Attention 降低 LLM 推理成本**：**Star Attention** 利用分块稀疏注意力机制（block-sparse attention mechanism），在保持 **95-100% 准确率** 的前提下，将 LLM 在长序列上的推理时间和显存占用降低了高达 **11 倍**。
  
  - 该机制可与基于 Transformer 的 LLMs 无缝集成，促进长序列任务的高效处理。
- **DALL-E 的变分界受到质疑**：围绕 DALL-E 论文的变分界（variational bound）展开了辩论，特别是对某个不等式的有效性和条件独立性的假设提出了质疑。
  
  - 对话集中在联合分布的某些属性是否成立，并推测作者可能忽略了关键假设。
- **贝叶斯统计在金融研究中的应用**：一位成员分享了他们在金融经济学方面的背景，特别是博士期间专注于**用于资产定价的贝叶斯统计**，并强调了该学科的趣味性。
  
  - 对话还涉及了算法交易的实践经验，包括所面临的挑战。
- **对算法交易的反思**：参与者讨论了算法交易的个人经验，其中一位成员提到仅在 **2017 年** 的加密货币热潮期间获得了一些利润。
  
  - 分享的反思突显了算法交易创业中所涉及的复杂性和学习曲线。

**提到的链接**：

- [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)：文本到图像生成传统上侧重于为固定数据集上的训练寻找更好的建模假设。这些假设可能涉及复杂的架构、辅助损失或 s...
- [𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文](https://x.com/gm8xx8/status/1861616780310929450)：Star Attention：长序列上的高效 LLM 推理🔗：https://github.com/NVIDIA/Star-Attention 论文：https://arxiv.org/abs/2411.17116 Star Attention 是一种分块稀疏注意力机制...
- [𝚐𝔪𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文](https://x.com/gm8xx8/status/1861282008082599967?s=46)：MH-MoE：多头混合专家模型 论文：https://arxiv.org/abs/2411.16205 MH-MoE 通过使用多头机制聚合来自不同专家空间的信息来提高模型性能。它匹配...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1311140808501624944) (4 条消息):

> `Karpathy 效应、TL;DR 时事通讯、Priompt 的 Python 移植、提示词设计工具`

- **源于 Karpathy 效应的博客文章热度**：一位成员注意到他们老板 Will 写的一篇博客文章正受到关注，这很可能是由于 **Karpathy 效应（Karpathy bump）**。
  
  - 这篇文章一直在流传，表明兴趣显著增加。
- **AI TL;DR 时事通讯作为主要来源**：另一位成员分享说，他们在 **TL;DR 时事通讯**（特别是专注于 AI 的那一期）中收到了这篇博客文章。
  - 他们提到每天都会阅读这两者，突显了该时事通讯的专业化定位。
- **使用 Python 进行提示词设计的尝试**：一位用户分享了他们花费 **9 小时**将 **priompt** 移植到 **Python** 的经验，并将成果提供给社区使用。
  
  - 他们链接了 [GitHub - zenbase-ai/py-priompt](https://github.com/zenbase-ai/py-priompt) 作为管理**提示词设计（prompt designs）**和**上下文窗口（context windows）**的资源。

**提到的链接**：[GitHub - zenbase-ai/py-priompt: Prompt design in Python](https://github.com/zenbase-ai/py-priompt)：Python 中的提示词设计。通过在 GitHub 上创建账户为 zenbase-ai/py-priompt 的开发做出贡献。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1311219957111193670) (93 messages🔥🔥):

> `MH-MoE paper, Star Attention paper, DALL-E variational bounds, Conditional independence in ML, Bayesian stats in finance`

- **MH-MoE 提升模型效率**：关于 [MH-MoE](https://arxiv.org/abs/2411.16205) 的论文描述了一种使用多头机制聚合信息的方法，在达到稀疏模型效率的同时，性能优于标准实现。
  
  - 值得注意的是，它保持了与 **BitNet** 等 1-bit LLM 的兼容性。
- **Star Attention 增强 LLM 推理**：[Star Attention](https://arxiv.org/abs/2411.17116) 提出了一种块稀疏注意力机制，可将长序列的推理时间和内存占用显著降低高达 **11倍**，同时保持高精度。
  
  - 它采用集成了局部和全局注意力的两阶段过程，兼容大多数基于 Transformer 的 LLM。
- **DALL-E 的变分界争议**：讨论围绕 DALL-E 论文中提出的变分界（variational bound）潜在问题展开，有观点认为由于对假设的误解，该变分界*可能是错误的*。
  
  - 参与者思考了假设条件独立性（conditional independence）的影响，这可能会影响所涉及不等式的有效性。
- **质疑机器学习推导中的条件独立性**：一位成员指出，如果推导错误地假设了独立性，可能会导致 KL divergence 计算结果错误。
  
  - 对话探讨了处理分布中依赖关系的含义，并指出了围绕证明结构的困惑。
- **关于金融和交易的个人见解**：成员们分享了在金融经济学中使用贝叶斯统计的经验，其中一位成员提到短暂的算法交易经历几乎没有产生利润。
  
  - 向金融领域的转型，结合对 Black-Scholes 模型入门的见解，引发了关于交易策略和市场行为的进一步讨论。

**提到的链接**：

- [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)：文本到图像生成传统上侧重于为固定数据集上的训练寻找更好的建模假设。这些假设可能涉及复杂的架构、辅助损失或 s...
- [来自 𝚐𝚖𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文](https://x.com/gm8xx8/status/1861616780310929450)：Star Attention: Efficient LLM Inference over Long Sequences🔗: https://github.com/NVIDIA/Star-Attention 论文: https://arxiv.org/abs/2411.17116 Star Attention 是一种块稀疏注意力机制...
- [来自 𝚐𝚖𝟾𝚡𝚡𝟾 (@gm8xx8) 的推文](https://x.com/gm8xx8/status/1861282008082599967?s=46)：MH-MoE: Multi-Head Mixture-of-Experts 论文: https://arxiv.org/abs/2411.16205 MH-MoE 通过使用多头机制聚合来自不同专家空间的信息来提高模型性能。它匹配...

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1311074668379115621) (6 messages):

> `FSDP Spaghetti Internals, Dynamic Structs, Knowledge Conflict`

- **FSDP 内部结构打破假设**：讨论强调 FSDP 混乱的内部结构（spaghetti internals）**拆解了所有张量和模块状态假设**，仅在最后重新组装它们。
  
  - 这一过程揭示了复杂的**状态管理**（state management）技术，这些技术需要仔细考虑。
- **Setattr 引起警觉**：一位成员开玩笑地提到，在处理代码时，一点点 **setattr** 就能让开发者保持**清醒**。
  
  - 这暗示了动态修改属性所带来的复杂性和潜在的意外情况。
- **根据模型名称定制结构体**：一位成员分享了他们根据输入的模型名称**决定结构体**（struct）的方法，确保所有内容都经过**完全类型检查**（type checked）。
  
  - 这种方法强调了模型结构中强大的类型安全性和适应性。
- **承认知识冲突**：简要提到的**知识冲突**（knowledge conflict）表明正在进行的关于信息差异的讨论。
  
  - 随着成员们在共享知识中导航，这促使了对清晰度和解决方案的需求。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1311074230820933694) (276 messages🔥🔥):

> `RWKV-7 developments, SSM and graphs curvature, Mamba 2 architecture, Gradient descent optimization, Curvature and vertex degree analogy`

- **RWKV-7 展示了实现 soft AGI 的潜力**：开发者表达了推迟 RWKV-8 开发的希望，认为 RWKV-7 的能力足以作为 soft AGI 的候选方案，同时也承认需要进一步改进。
  
  - 他们强调，虽然 RWKV-7 相当稳定，但在过渡到 RWKV-8 之前仍有可以增强的领域。
- **探索 SSMs 及其与图（graphs）的关系**：参与者讨论了 SSMs (State Space Models) 在消息传递（message passing）方面的潜力，强调了图中顶点度（vertex degree）与流形曲率（curvature on manifolds）之间的类比。
  
  - 他们指出，较高的顶点度与负曲率相关，展示了高斯曲率（Gaussian curvature）的离散版本。
- **关于学习中迭代更新的见解**：讨论围绕以批处理格式而非按 token 顺序对状态矩阵进行更新展开，同时考虑了 L2 loss 及其对学习的影响。
  
  - 这包括建议引入权重衰减（weight decay），以在多个梯度步骤中优化更新过程。
- **离散结构中的曲率特征**：对话提出了这样一个观点：图中额外的边角（edge angle）数据可以帮助推导不同类型的曲率，从而增强对其属性的理解。
  
  - 参与者被鼓励将这些概念与三角网格（triangle meshes）联系起来，将其作为将曲率原理应用于图的垫脚石。
- **Mamba 2 及其效率**：社区对新的 Mamba 2 架构感到好奇，特别是其效率方面以及与现有模型的对比。
  
  - 重点在于 Mamba 2 是否允许更多的张量并行（tensor parallelism），或者是否比传统架构更具优势。

**提到的链接**：

- [Predicting Emergent Capabilities by Finetuning](https://arxiv.org/abs/2411.16035)：现代 LLM 扩展中的一个基本开放挑战是对涌现能力（emergent capabilities）缺乏理解。特别是，语言模型预训练损失被认为是高度可预测的函数...
- [no title found](https://chinmayhegde.github.io/introml-notes-sp2020/pages/lecture3_notes.html)：未找到描述
- [Tweet from BlinkDL (@BlinkDL_AI)](https://x.com/BlinkDL_AI/status/1861753903886561649)：RWKV-7 "Goose" 🪿 是一个元上下文学习器（meta-in-context learner），通过在每个 token 处进行上下文梯度下降（in-context gradient descent），在上下文中对其状态进行测试时训练（test-time-training）。它就像一个不断适应外部环境的世界模型...
- [State Space Duality (Mamba-2) Part I - The Model | Tri Dao](https://tridao.me/blog/2024/mamba2-part1-model/) ：未找到描述
- [AI as Humanity's Salieri: Quantifying Linguistic Creativity of Language Models via Systematic Attribution of Machine Text against Web Text](https://arxiv.org/abs/2410.04265)：长期以来，创造力一直被认为是 AI 最难模仿的人类智能方面之一。然而，随着 ChatGPT 等 Large Language Models (LLMs) 的兴起，人们开始质疑...
- [RWKV-LM/RWKV-v7/rwkv_v7_demo.py at main · BlinkDL/RWKV-LM](https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v7/rwkv_v7_demo.py)：RWKV 是一种具有 Transformer 级 LLM 性能的 RNN。它可以像 GPT 一样直接训练（可并行化）。因此它结合了 RNN 和 Transformer 的优点——出色的性能、快速的推理...
- [GitHub - BlinkDL/modded-nanogpt-rwkv: RWKV-7: Surpassing GPT](https://github.com/BlinkDL/modded-nanogpt-rwkv)：RWKV-7：超越 GPT。通过在 GitHub 上创建账户为 BlinkDL/modded-nanogpt-rwkv 的开发做出贡献。
- [Single-unit activations confer inductive biases for emergent circuit solutions to cognitive tasks](https://www.biorxiv.org/content/10.1101/2024.11.23.625012v1)：训练后的循环神经网络（RNNs）已成为模拟大脑神经动力学的主要框架，因为它们能够模仿群体水平的计算如何从内部产生...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1311217106162941952) (1 messages):

> `Research updates`

- **分享了新的研究见解**：一名成员在频道中分享了一个**有趣的研究更新**。
  
  - 您可以在 [Discord 链接](https://discord.com/channels/729741769192767510/1266757996923588721/1311211171898064947)中找到更多详情。
- **更新亮点讨论**：该更新引发了成员们关于其在当前研究趋势中影响的讨论。
  
  - 成员们表达了他们的兴奋之情，指出该研究为进一步探索铺平了道路。

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1311192975132332175) (1 messages):

> `RunPod server configurations, OpenAI completions endpoint, Llama 3.2 model usage`

- **RunPod 教程缺少 OpenAI completions 详情**：一位用户发现 RunPod 教程没有明确提到 OpenAI completions 的正确端点（endpoint），而这是成功通信所必需的。
  
  - 为了使其正常工作，请使用 [此端点](https://%7BPOD_ID%7D-11434.proxy.runpod.net/v1/chat/completions) 并在请求中提供模型。
- **成功推送到外部 Ollama 服务器**：用户报告称将一个 generateuntil 任务推送到外部 Ollama 服务器，并指出在运行 **Llama 3.2** 的服务器上可以看到该请求。
  
  - 这表明与部署的模型交互成功，突显了服务器设置的功能性。

**提到的链接**：

- [未找到标题](https://{POD_ID}-11434.proxy.runpod.net/v1/chat/completions)：未找到描述
- [未找到标题](https://{POD_ID}-11434.proxy.runpod.net/v1/chat/completions')：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1311381561937100831) (1 messages):

> `Gemini Flash 1.5, Provider Routing, Load Balancing, Grok Vision Beta`

- **Gemini Flash 1.5 容量提升**：OpenRouter 对 **Gemini Flash 1.5** 的容量进行了**重大提升**。之前遇到 rate limiting 的用户可以重新尝试请求。
  
  - 这一改进应能在高流量期间显著提升用户体验。
- **Provider 价格优化**：平台现在正将**呈指数级增长的流量**路由到成本最低的 provider，确保用户平均能从更低的价格中受益。更多信息可以在 [Provider Routing 文档](https://openrouter.ai/docs/provider-routing) 中找到。
  
  - 该策略通过在必要时 fallback 到其他 provider 来维持性能。
- **Load Balancing 策略详解**：OpenRouter 的负载均衡优先考虑运行时间稳定的 provider，并根据**成本效益**路由请求。例如，请求会进行**加权**，以偏向过去 10 秒内成本最低且故障最少的 provider。
  
  - 这确保了在高需求情况下资源得到高效且有效的利用。
- **Grok Vision Beta 发布**：OpenRouter 正在提升 **Grok Vision Beta** 的容量，鼓励用户通过 [此链接](https://openrouter.ai/x-ai/grok-vision-beta) 进行尝试。
  
  - 这是用户在服务扩展能力时对其进行测试的机会。

**提到的链接**：

- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing)：跨多个 provider 路由请求
- [Grok Vision Beta - API, Providers, Stats](https://openrouter.ai/x-ai/grok-vision-beta)：Grok Vision Beta 是 xAI 具有视觉能力的实验性语言模型。通过 API 运行 Grok Vision Beta

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1311096024680632361) (136 条消息🔥🔥):

> `Jamba 1.5 model, AI21 Labs support, EVA Qwen2.5 pricing, Claude API issues, OpenRouter functionality`

- **Jamba 1.5 模型问题**：用户报告了来自 AI21 Labs 的 **Jamba 1.5 mini 模型** 的问题，特别是在调用函数时收到空响应。
  
  - 尽管尝试了不同版本，问题依然存在，用户推测这可能是由于消息准备或后端问题导致的。
- **EVA Qwen2.5 定价翻倍**：一些用户注意到 **EVA Qwen2.5 72B** 的价格翻了一倍，并询问这是促销价格结束还是标准涨价。
  
  - 有推测认为定价变化可能是由于竞争加剧和商业策略调整。
- **Claude API 错误**：一位用户在使用 **Claude API** 时遇到错误，收到关于函数被屏蔽和后端问题的消息。
  
  - 错误归因于 Anthropic 状态面板上最近记录的一个事件，但许多用户报告 Claude 模型的行为不一致。
- **OpenRouter 模型上下文长度关注**：一位用户对 **qwen-2.5-coder** 模型仅允许最高 32k 上下文长度（而非预期的 128k）表示担忧。
  
  - 澄清说明对 128k 的支持取决于提供商，特别指出目前只有 **Hyperbolic** 提供完整的上下文窗口。
- **OpenRouter 聊天流式传输问题**：用户对 OpenRouter 聊天室中的聊天流式传输表示沮丧，因为屏幕不断滚动导致难以阅读消息。
  
  - 用户请求增加一个禁用流式传输的选项，以提升用户体验和可读性。

**提到的链接**：

- [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/)：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 注：这是 QwQ 的发音：/kwju:/，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...
- [Q\*: Improving Multi-step Reasoning for LLMs with Deliberative Planning](https://arxiv.org/abs/2406.14283)：大语言模型 (LLMs) 在许多自然语言任务中展示了令人印象深刻的能力。然而，自回归生成过程使 LLMs 容易产生错误、幻觉...
- [{"model": "anthropic/claude-3.5-sonnet", "messages": [{"role": "system", "conten - Pastebin.com](https://pastebin.com/UjrGQZj6)：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以让你在线存储文本一段时间的网站。
- [The capabilities of multimodal AI | Gemini Demo](https://www.youtube.com/watch?v=UIZAiXYceBI&list=PL590L5WQmH8cSyqzo1PwQVUrZYgLcGZcG&index=12&t=200s&ab_channel=Google)：我们的原生多模态 AI 模型 Gemini 能够跨文本、图像、音频、视频和代码进行推理。以下是 Gemini 的精彩瞬间，了解更多...
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing#quantization)：跨多个提供商路由请求
- [Elevated errors for requests to Claude 3.5 Sonnet](https://status.anthropic.com/incidents/nwn20mzh9v1k)：未找到描述
- [Llama 3.1 Euryale 70B v2.2 - API, Providers, Stats](https://openrouter.ai/sao10k/l3.1-euryale-70b)：Euryale L3.1 70B v2。通过 API 运行 Llama 3.1 Euryale 70B v2.2
- [Dubesor LLM Benchmark table](https://dubesor.de/benchtable)：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1311146860697747488) (4 条消息):

> `Custom API Keys Access Requests`

- **开发者渴望自定义 API Keys**：多位开发者表示有兴趣获得 **自定义 API key 功能** 的访问权限，强调了他们对该平台能力的极大热情。
  
  - 一位用户感谢团队的 **出色工作**，而另一位用户则询问如何获取该功能的访问权限。
- **更多自定义测试密钥请求**：另一位开发者加入对话，表示他们也希望申请自定义 **beta keys** 的访问权限。
  
  - 这表明 **社区成员** 对于探索平台高级功能的兴趣日益增长。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1311074930477236276) (61 messages🔥🔥):

> `QwQ-32B-Preview, Olmo 模型差异, PRM 在 Scaling 中的效用, Tülu 与 Olmo 性能对比, Demo 可用性`

- **QwQ-32B-Preview 模型见解**：[QwQ-32B-Preview](https://huggingface.co/Qwen/QwQ-32B-Preview) 是一款实验性模型，突出了 **AI 推理能力**，但面临**语言混杂**和**递归推理循环**等问题。
  
  - 尽管存在局限性，但它在**数学和编程**方面展现出潜力，多位成员对此表达了兴趣。
- **Olmo 模型表现出差异化性能**：成员们讨论了 **Olmo 模型** 的行为与 **Llama** 截然不同，特别指出其在不同任务中的性能一致性。
  
  - 值得注意的是，有人提到 **Tülu** 在特定提示词上的表现优于 Olmo 2，引发了关于哪个模型领先的进一步讨论。
- **质疑 PRM 的有效性**：围绕**过程奖励模型 (PRM)** 对于扩展 AI 模型是否真正有效展开了讨论，因为它们在初始阶段的效用虽然明确但很有限。
  
  - 一位成员幽默地评论某个发布版本感觉太模糊，缺乏实质性细节，对其在现实世界中的适用性表示怀疑。
- **探索 Demo 选项**：关于所讨论模型的 Demo 可用性出现了疑问，一些成员确认之前已经分享过 Demo。
  
  - Natolambert 表示打算在撰写文章之前搭建一个 Demo，表明将通过亲身实践来理解模型功能。
- **澄清 Pass@1 与 Greedy Generation**：对话转向技术层面，成员们澄清 **pass@1** 展示的结果可能属于 **greedy generation**，并对其影响展开了辩论。
  
  - 结果的轻微偏差让一些成员感到困惑，特别是关于与 **AIME 分数**相关的特定指标。

**提到的链接**：

- [QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD。注：QwQ 的发音为 /kwju:/，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...
- [Qwen/QwQ-32B-Preview · Hugging Face](https://huggingface.co/Qwen/QwQ-32B-Preview): 未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1311225131082846269) (49 条消息🔥):

> `Bsky 数据集争议，对社交媒体研究的影响，用户屏蔽趋势，Bluesky 上的数据集发布，在线帖子的公开可访问性`

- **Bsky 数据集引发愤怒**：一名 *HF 员工创建了一个 Bsky 帖子数据集*，这些帖子是公开的且根据 ToS 允许使用，但这引发了强烈的抵制，并导致随后*该数据集被删除*。
  
  - 一位用户指出，*“他只是在伤害小人物”*，因为这些行为反映了研究社区内更大的问题。
- **研究领域面临挫折**：数据集的删除使*整个社交媒体研究领域陷入倒退*，特别是在被切断 Twitter API 访问权限之后。
  
  - 用户讨论了*大型实验室将如何不受影响*，而小型研究人员则因担心诉讼而受苦。
- **Bluesky 上被屏蔽最多的用户**：报告显示 *Daniel 已成为 Bluesky 上被屏蔽次数排名第一的用户*，引发了公众褒贬不一的反应。
  
  - 虽然有些人表示同情，但其他人表示，作为*反 AI 恐惧的象征感觉肯定不好*。
- **数据集发布狂潮**：在争议中，一位用户宣布*发布了一个包含 200 万条 Bluesky 帖子的数据集*，激起了更多贡献的呼声。
  
  - 这表明一种上升趋势，即*研究人员鼓励彼此分享庞大的数据集*，尽管可能面临抵制。
- **社区对互联网可访问性的反应**：像 Natolambert 这样的用户讨论道，*互联网本质上使内容对大型科技公司和公众开放*，这引发了关于隐私的分歧。
  
  - 对话包括对*屏蔽者以及由此产生的社交媒体 Feed 政治性质*的不同观点，暗示了对更精简讨论的渴望。

**提到的链接**：

- [ClearSky](https://clearsky.app/): ClearSky
- [来自 Alpin (@AlpinDale) 的推文](https://x.com/AlpinDale/status/1861819574259192082): 发布：一个包含 200 万条 Bluesky 帖子的数据集。该数据集是使用 Bluesky 的 API 收集的，我希望它对所有的研究人员都有用！
- [Sid Losh (@losh.bsky.social)](https://bsky.app/profile/losh.bsky.social/post/3lbx5257qlk24): 我不确定加密在量子末日前夕是否有帮助。我认为更安全的方法是简单地认为你曾在任何地方发布的任何内容最终都会暴露在公众审视之下...
- [Bluesky](https://bskye.app/profile/ind3fatigable.xyz/post/3lbxeknnrvs26): 未找到描述
- [Jade (@not.euclaise.xyz)](https://bsky.app/profile/not.euclaise.xyz/post/3lbxgbhkws22a): 这里有 2.35 亿条帖子 https://zenodo.org/records/11082879
- [Nathan Lambert (@natolambert.bsky.social)](https://bsky.app/profile/natolambert.bsky.social/post/3lbxgb7lrqs2v): 一个勇敢的灵魂。赢得了我的关注
- [Xeophon (@xeophon.bsky.social)](https://bsky.app/profile/xeophon.bsky.social/post/3lbvxycwoh22s): 我有研究社交媒体仇恨言论的同事，他们一直受困于非常陈旧的 Twitter 数据。Bluesky 本可以成为一个替代方案，现在他们被你的行为吓到了。干得好。
- [Daniel van Strien (@danielvanstrien.bsky.social)](https://bsky.app/profile/danielvanstrien.bsky.social/post/3lbu6l4fxdc2e): 新的 @huggingface.bsky.social @bsky.app 社区组织的第一个数据集：one-million-bluesky-posts 🦋📊 100 万条来自 Bluesky firehose API 的公开帖子🔍 包括文本、元数据和语言...
- [@alpindale.bsky.social](https://bsky.app/profile/alpindale.bsky.social/post/3lbxb3zmfys25): 发布：一个包含 200 万条 Bluesky 帖子的数据集。该数据集是使用 Bluesky 的 API 收集的，我希望它对所有的研究人员都有用！

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1311147667102896178) (14 messages🔥):

> `模型见解, O1 LLM 影响, 即将发布的内容, DesuAnon 的新存档格式, 2024 AI 插图`

- **对 'O1' LLM 的期待**：'O1' 模型被强调为重新思考 LLM 各个方面的**极好礼物**，引发了成员们的兴奋。
  
  - *一位成员将其与 TempleOS 进行了比较*，增加了讨论的热度。
- **即将进行的录音采访**：一场关于 RL 的采访计划于**下周**录制，引起了群组内的期待。
  
  - 成员们渴望看到这次采访可能提供的**见解**。
- **DesuAnon 的存档更新**：关于新存档格式和 Prompts 的更新信息已添加到 desuAnon 的 [Hugging Face](https://huggingface.co/desuAnon/SoraVids/tree/main) 页面。
  
  - 这一新更新在大约 **21 小时前**得到了验证，表明社区正在持续发展。
- **2024 AI 的视觉洞察**：成员们对一张名为 **'2024 Alan D. Thompson AI Bubbles & Planets'** 的极具视觉冲击力的图片感到兴奋，认为其非常迷人。
  
  - 一位成员表示这张*图片非常出色 (kind of owns)*，强调了它的吸引力。
- **订阅者互动计划**：有人建议与订阅者分享内容，反映了保持社区知情权的承诺。
  
  - 据提到，**内容计划于下周发布**，以确保持续的互动和参与。

**提到的链接**：

- [desuAnon/SoraVids at main](https://huggingface.co/desuAnon/SoraVids/tree/main)：未找到描述
- [Nathan Lambert (@natolambert.bsky.social)](https://bsky.app/profile/natolambert.bsky.social/post/3lbvktl2d3223)：提问 —— 谁提出了 “post-training” 这个术语？它在过去 12-18 个月中出现，但我不知道出处，我需要了解一下。🙇
- [Models Table](https://lifearchitect.ai/models-table/)：在新标签页中打开 Models Table | 返回 LifeArchitect.ai。数据字典 Model (Text)，大语言模型的名称。

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1311200493749669928) (2 messages):

> `LLM 中的低比特量化, Deepseek 的 AI 进展, CEO 梁文锋的背景, High-Flyer 在 Deepseek 中的角色, AI 开发中的算力资源`

- **低比特量化有利于训练不足的 LLM**：研究表明，**低比特量化 (Low-bit quantization)** 更有利于训练不足的大语言模型，且模型规模越大，量化诱导的性能下降 (QiD) 越小。该研究揭示了 Scaling Laws，有助于通过从 1500 多个量化 Checkpoints 中得出的数据，预测 LLM 的训练水平以及不同模型大小所需的 Token 数量。
  
  - 深入研究发现，预测显示未来使用超过 **100 万亿 Tokens** 训练的模型可能会面临不理想的低比特量化性能。
- **Deepseek 在推理基准测试中击败 OpenAI**：据报道，**Deepseek** R1 模型在多个推理基准测试中**击败了 OpenAI 的 o1**，凸显了这家初创公司尽管低调但潜力巨大。
  
  - 在 **High-Flyer** 的全额资助下，Deepseek 专注于基础技术和开源其模型，同时通过提供低廉的 API 费率在中国引发价格战。
- **梁文锋 (Liang Wenfeng) 令人印象深刻的背景**：CEO **梁文锋**在创立 Deepseek 之前曾领导 **High-Flyer**（幻方量化），这是一家估值达 **80 亿美元** 的顶尖量化对冲基金。他的领导力预示着 Deepseek 在 AI 领域创新方法背后的雄厚专业实力。
  
  - 在他以往经验的指导下，Deepseek 旨在取得重大进展，且目前没有立即的融资计划。
- **High-Flyer 的算力优势**：Deepseek 的扩展能力得到了 **High-Flyer 计算集群** 的支持，为这家 AI 初创公司提供了充足的计算资源。专家估计他们可能拥有约 **5 万张 Hopper GPU**，远超公开披露的 **1 万张 A100**。
  
  - 这种对尖端技术的获取使 Deepseek 在开发创新模型时，在 AI 领域具有极强的竞争力。

**提到的链接**：

- [Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens](https://arxiv.org/abs/2411.17691)：我们发现低比特量化有利于训练不足的大语言模型 (LLM)，观察到规模更大或训练 Token 较少的模型经历的量化诱导性能下降较少...
- [Deepseek: The Quiet Giant Leading China’s AI Race](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas)：其 CEO 最深度采访的带注释翻译版。

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1311079959321772063) (2 条消息):

> ``

- **SnailBot News 警报**：已向 <@&1216534966205284433> 角色发送了关于 SnailBot News 的通知。
- **重复的 SnailBot News 通知**：再次向 <@&1216534966205284433> 角色发送了 SnailBot News 通知。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1311081656588828756) (102 条消息🔥🔥):

> `Perplexity 支持问题、折扣优惠、图像生成能力、模型选择优势、搜索/聊天混合领域的竞争对手`

- **Perplexity 的支持服务通常让人感觉不足**：许多用户对 Perplexity 的支持表示不满，称其“枯燥”且“匮乏”。几位成员甚至提到由于这些问题取消了订阅。
- **新用户可使用折扣码**：成员们分享了一个月免费 Pro 订阅的促销折扣码信息，鼓励他人在过期前使用。该代码为寻求优惠的新用户提供了潜在价值。
- **Perplexity Pro 中的图像生成功能**：虽然 Perplexity Pro 允许生成图像，但用户指出对输出的控制有限，使其不太适合详细的需求。常规使用对于偶尔的图像需求是不错的，尽管没有专门的图像生成页面。
- **订阅中模型选择的优势**：订阅者可以访问更好的模型，如 Sonnet、4o 和 Grok 2，这些模型增强了处理编程和数学等复杂任务的性能。许多用户觉得免费版已经足够，但订阅为频繁和高级用户增加了可观的价值。
- **与 You.com 的对比**：会员讨论强调了 Perplexity 与 You.com 作为搜索/聊天混合领域竞争平台的对比。用户表示更倾向于 Perplexity，但也承认两个平台各有优劣。

**提到的链接**：

- [Mike Smith (@mikesmith187) 的推文](https://x.com/mikesmith187/status/1861824606790676915?s=46)：很少有公司能像 @perplexity_ai 这样，更少有人能像 @apostraphi 那样设计。与 Phi 合作的 dive club 这一集绝对精彩。值得一听，因为...
- [floguo (@floguo) 的推文](https://x.com/floguo/status/1861586996700872978?s=61)：与 @perplexity_ai 保持好奇心

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1311213320593735681) (5 条消息):

> `AI 音景、认知行为疗法、最古老的字母文字、Bluesky 对阵扎克伯格、黑色星期五辩论`

- **探索 AI 自然音景**：分享了一个关于 Bjork 创作的迷人 [AI 生成自然音景](https://www.perplexity.ai/page/bjork-s-ai-nature-soundscape-LCJ46dmFQE21FJN6NfmNDw) 的链接，该音景增强了对自然声音的欣赏。
  
  - 这个项目可能反映了艺术与技术的融合，对音响发烧友和 AI 爱好者都很有吸引力。
- **认知行为疗法见解**：一位成员链接到了一个讨论 [认知行为疗法](https://www.perplexity.ai/search/cognitive-behavioral-therapy-i-3FO_vuPqQUW5s6NlAFiy4w) 的页面，重点介绍了其技术和应用。
  
  - 这可能会引发围绕心理健康策略及其在各种场景下有效性的讨论。
- **文字史上的历史性发现**：揭示了发现的 [最古老字母文字](https://www.youtube.com/embed/Ho0g_ri8Rj4)，这可能会重塑我们对人类交流历史的理解。
  
  - 这一发现可能会引发关于古代语言及其随时间演变的精彩辩论。
- **扎克伯格对 Bluesky 的担忧**：分享了一篇讨论 [扎克伯格对 Bluesky 影响社交媒体动态的担忧](https://www.perplexity.ai/page/zuckerberg-fears-bluesky-s-ris-W9NsfXd_S1eoT.8tZJ.ESg) 的文章。
  
  - 这突显了社交媒体平台之间的竞争格局以及用户参与度所涉及的利益。
- **黑色星期五是骗局吗？一场讨论**：提出了一个质疑 [黑色星期五是否真的是骗局](https://www.perplexity.ai/search/is-black-friday-a-scam-cds47yL8TSK2rGIvtUHb_w) 的查询，邀请大家对促销期间的消费行为发表看法。
  
  - 这个话题很可能会引发关于消费主义和营销策略的热烈讨论。

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1311128380091142164) (4 条消息):

> `Perplexity API financial data sources, Reddit citation issues, GitHub projects, Perplexity engine depreciation`

- **Perplexity API 财务数据来源**：一位成员询问 **Perplexity engine** 从何处获取其财务数据，以及 API 是否可以将这些数据提供给其他项目供内部使用。
  
  - 他们表示对显示公司收益的 **stock ticker data** 感兴趣，并参考了相关的截图。
- **API 不再支持 Reddit 引用**：一位用户询问了 **API 不再支持 Reddit 引用** 背后的原因。
  
  - 这可能表明功能发生了变化，影响了依赖这些数据源获取数据的用户。
- **GitHub 项目展示**：有人分享了他们的第一个 GitHub 项目 **perplexity-cli**，邀请其他人查看并可能给一个 star。
  
  - 该项目为 **Perplexity API** 提供了一个简单的命令行客户端，允许用户直接从终端提问并接收回答。
- **关于工具弃用警告的讨论**：一位用户建议其他人不要为他们认为已 **deprecated** 或仅处于 beta 阶段的工具付费，并推荐了 **Exa** 或 **Brave** 等替代方案。
  
  - 此类反馈强调了在进行财务投入之前评估工具可靠性的重要性。

 

**提到的链接**：[GitHub - dawid-szewc/perplexity-cli: 🧠 A simple command-line client for the Perplexity API. Ask questions and receive answers directly from the terminal! 🚀🚀🚀](https://github.com/dawid-szewc/perplexity-cli.git): 🧠 一个简单的 Perplexity API 命令行客户端。直接从终端提问并接收回答！ 🚀🚀🚀 - dawid-szewc/perplexity-cli

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1311093017922502718) (84 条消息🔥🔥):

> `Scripting Aider with Python, Sonnet in Aider Development, Benchmark Results for Aider, Cedarscript Discussion, QwQ Model Performance`

- **使用 Python 编写 Aider 脚本**：许多用户表达了使用 Python 编写 Aider 脚本的兴趣，讨论强调这在技术上是不受支持的，但自发布以来在实践中证明是稳定的。
  
  - 主要担忧仍然是非官方 API 可能会在没有向后兼容性的情况下发生变化，因此建议用户固定（pin）其依赖项。
- **Sonnet 主导 Aider 开发**：Aider 开发中使用的主要模型是 **Sonnet**，这是大多数用户的首选。
  
  - 用户讨论了将 Sonnet 与各种模型配置结合使用的有效性及其对性能的影响。
- **Cedarscript 结果引发质疑**：Cedarscript 尚未展示其承诺的能力，导致用户对其有效性表示怀疑。
  
  - 讨论指出，该工具仅在有限的上下文中进行了 benchmark，因此很难评估其整体潜力。
- **QwQ 模型性能问题**：多位用户报告在尝试对 QwQ 模型进行 benchmark 时出现 **gateway timeout errors**。
  
  - 提到了 glhf.chat 平台上模型加载的延迟，影响了其在 benchmark 期间的响应速度。
- **Aider vs. Cursor AI IDE**：用户将 Aider 与 Cursor AI IDE 进行了比较，强调尽管有一些重叠的功能（如 autocomplete），但两者用途不同。
  
  - 虽然 Aider 被认为是一个集成在 Emacs 中的多功能工具，但 Cursor 提供了功能更全的 IDE 体验，并计划进行增强。

 

**提到的链接**：[QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 注意：这是 QwQ 的发音：/kwju:/ ，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...

 

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1311101741445419088) (26 条消息🔥):

> `Sonnet 的 PDF 支持、使用 Aider 进行重构、Aider 命令与特性、Next.js 文件夹结构问题、用于转录的 Whisper API`

- **Sonnet 获得 PDF 阅读能力**：一位成员报告称，通过执行 `aider --install-main-branch` 命令，Sonnet 刚刚增加了 PDF 支持。
  
  - 另一位用户分享了他们的积极体验，指出 Sonnet 能够有效地理解新的 PDF 读取功能。
- **使用 Aider 重构项目**：一位用户讨论了他们在项目中重构代码的方法，并表达了在管理多个聊天（chats）时遇到的挑战。
  
  - 另一位成员提到，他们正在使用 Aider 的 `/read` 命令逐步重写现有项目，以遵循新的编码模式。
- **理解 Aider 的 editor 命令**：关于新的 `/editor` 命令引发了讨论，该命令允许用户在编辑器消息中添加命令前缀。
  
  - 用户在各种环境中对此进行了测试，导致在 PowerShell 中出现问题，但在 Neovim 中取得了成功。
- **Next.js 文件夹结构问题**：一位成员对他们的 Next.js 设置表示沮丧，因为现有的文件夹结构被意外地重新创建。
  
  - 这引发了关于在保持代码库整洁的同时如何有效管理项目路径的担忧。
- **用于隐私保护的本地 Whisper API**：一位用户分享了他们用于转录服务的本地 Whisper API 实现，强调了其设置中的隐私性。
  
  - 他们为感兴趣的社区成员提供了示例 curl 命令，用于测试托管在 Apple M4 Mac mini 上的 API。

**提到的链接**：

- [Whisper.cpp Server](https://api.ailocal.org): 未找到描述
- [FAQ](https://aider.chat/docs/faq.html#how-are-the-aider-wrote-xx-of-code-stats-computed): 关于 aider 的常见问题解答。
- [FAQ](https://aider.chat/docs/faq.html#why-is-the-llm-speaking-to-me-in-an-unexpected-language): 关于 aider 的常见问题解答。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1311110330352402502) (46 条消息🔥):

> `Axolotl vs Unsloth, Fine-tuning 预训练模型, 多模型 Inference, Fine-tuning 数据集格式化, Embeddings 模型 Fine-tuning`

- **讨论 Axolotl vs Unsloth**：一位成员强调，在 **Axolotl** 和 **Unsloth** 之间的选择取决于用户偏好，并指出 **Axolotl** 可能会让人感觉臃肿，而 **Unsloth** 提供了更精简的代码库。
  
  - *Mrdragonfox* 指出，性能的很大一部分取决于数据集质量，而不仅仅是框架。
- **基于先前调优进行模型 Fine-tuning**：*Fjefo* 警告说，对已经调优过的模型进行 **Fine-tuning** 可能会导致**灾难性遗忘 (catastrophic forgetting)**，从而影响输出质量。
  
  - 一位成员澄清说，虽然可以进一步进行 **Fine-tuning**，但可能不会产生有益的结果。
- **在单个 GPU 上进行多个模型的 Inference**：讨论围绕在单个 **A100** GPU 上运行多个 **Llamas 3.2 8B** 模型展开，成员们建议 **Batch inference** 可能会更有效率。
  
  - *Lee0099* 提到可以利用 **LoRa adapters** 在单个基础模型上为不同任务管理不同的模型。
- **为 Fine-tuning 格式化数据集**：一位用户询问了 **Fine-tuning** 数据集所需的格式，旨在为他们的模型提供输入和指令。
  
  - *Lee0099* 回复说，使用 JSONL 文件的 **Alpaca format** 是一种典型方法，**Unsloth** 的 Notebooks 中有相关示例。
- **对 Fine-tuning embeddings 模型的兴趣**：一位成员表示有兴趣使用 **Unsloth** 来 **Fine-tuning** 一个 **Embeddings** 模型，但没有找到关于该主题的现有 Notebooks。
  
  - *Theyruinedelise* 表示，预计很快将支持 **Embeddings** 模型的 **Fine-tuning**。

**提到的链接**：

- [Instruction Tuning – Axolotl](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/inst_tune.html)：未找到描述
- [mesolitica/malaysian-Llama-3.2-3B-Instruct · Hugging Face](https://huggingface.co/mesolitica/malaysian-Llama-3.2-3B-Instruct)：未找到描述
- [FAQ — NVIDIA Triton Inference Server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/faq.html#how-can-i-fully-utilize-the-gpu-with-triton-inference-server.)：未找到描述
- [unsloth/Phi-3.5-mini-instruct-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/Phi-3.5-mini-instruct-bnb-4bit)：未找到描述
- [Added Support for Apple Silicon by shashikanth-a · Pull Request #1289 · unslothai/unsloth](https://github.com/unslothai/unsloth/pull/1289)：未优化。目前尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes。用于构建 bitsandbytes 的命令：`cmake -DCOMPUTE_BACKEND=mps -S .`。安装命令：`pip install unsloth-zoo==2024.11.4`，`pip install xformers==0.0.25`。
- [How to Finetune Llama-3 and Export to Ollama | Unsloth Documentation](https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama#id-6.-alpaca-dataset)：为在 Ollama 上本地运行而创建定制化个人助手（如 ChatGPT）的入门指南。
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks)：查看下方列表以获取我们所有的 Notebooks：

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1311075099721334836) (17 条消息🔥):

> `RTX 3090 定价, GPU 托管方案, 托管中的 Docker 容器, 对高 VRAM GPU 的需求`

- **RTX 3090 定价惊喜**：一位成员报告称 **RTX 3090** 的平均价格仍约为 **1500 美元**，这让讨论中的几个人感到惊讶。
  
  - 其他人指出看到过低至 **550 美元** 的价格，表明市场存在波动。
- **GPU 托管偏好**：用户希望有一种 **24GB 显存层级的托管服务**，并带有可以按分钟开启的额外 GPU，反映了对 GPU 资源可扩展性的需求。
  
  - 成员们讨论了托管应主要运行 **Docker containers**，从而最大限度地减少对 **SSH** 的依赖。
- **对高 VRAM GPU 的高需求**：讨论强调了与 **A5000** 等 **20GB** 选项相比，对 **24GB** GPU 的**更高需求**，暗示行业更倾向于更强大的显存配置。
  
  - 成员们强调，在行业中，由于 **HBM** 的优势，规模的标准单位通常瞄准 **80GB** 配置。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1311077621257211995) (40 messages🔥):

> `Finetuning Local Models, Multi-GPU Support, Evaluating Models for EM and F1-Score, Model Quantization, Using Prompt Styles for Mistral-Nemo`

- **本地 Ollama 模型微调技巧**：一位成员询问如何在 Unsloth 中声明本地 **Llama 3.2 model**，以便使用 JSON 数据文件进行微调，但在查找正确的模型路径时遇到了困难。
  
  - *另一位用户建议改用 Unsloth 模型，称其对初学者非常友好，并分享了相关文档链接。*
- **期待 Multi-GPU 支持**：用户询问了 **multi-GPU** 微调的预期支持时间以及是否会有 Beta 测试，并表达了参与意向。
  
  - 一位成员指出 multi-GPU 支持尚未发布，另一位确认存在 Beta 测试，但名额有限。
- **模型评估咨询**：一位成员寻求关于如何在 Unsloth 中评估模型的 **EM 和 F1-score** 的建议，但收到的关于具体方法的回复并不明确。
  
  - 另一位成员继续询问类似话题，显示出对更清晰指导的需求。
- **理解量化优势**：一位成员询问为什么将 **4-bit quantized** 且微调后的模型保存为 **fp16** 格式进行推理会更好，另一位用户强调 16-bit 可以提高推理质量。
  
  - 这突显了关于量化技术及其对模型性能影响的持续讨论。
- **为 Mistral-Nemo 使用 GPT 样式的 Prompt**：有提问关于 **GPT-style prompts** 是否适用于微调 **Mistral-Nemo-12b-4bit** 模型，另一位用户确认如果是 Instruct 模型则具有兼容性。
  
  - 这反映了用户对不同模型架构的有效 Prompt 样式的广泛好奇。

**提及的链接**：

- [Blog](https://unsloth.ai/blog): 无描述
- [text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts](https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb): 使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): 查看下方列表以获取我们所有的 Notebook：

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1311280388999741460) (2 messages):

> `Equation Correction, Order of Operations`

- **添加括号以修正等式**：一位成员提供了一个错误的等式 **1 + 2 imes 3 + 4 imes 5 + 6 imes 7 + 8 imes 9 = 479**，并寻求通过添加括号使其成立。
  
  - 他们详细列出了计算步骤，得出结论：如果不加括号，左侧总和为 **141**，与预期的 **479** 相差甚远。
- **理解运算顺序 (PEMDAS)**：作为等式修正过程的一部分，该成员回顾了 **运算顺序**，确认乘法优先于加法。
  
  - 执行的计算包括 **2 imes 3 = 6**、**4 imes 5 = 20**、**6 imes 7 = 42** 以及 **8 imes 9 = 72**，最终得出总和 **141**。

 

**提及的链接**：[QwQ: Reflect Deeply on the Boundaries of the Unknown](https://qwenlm.github.io/blog/qwq-32b-preview/): GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD。注：这是 QwQ 的发音：/kwju:/，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1311086276732260473) (99 条消息🔥🔥):

> `Wildcard 定义, 图像生成工作流, ControlNet 功能, 高质量图像创作, SD 插件更新`

- **关于 Wildcard 定义的辩论**：展开了一场关于编程中 “wildcards” 定义及其在不同语境下应用的对话，特别是与 Civitai 和 Python 术语相关的部分。
  
  - 参与者注意到在理解和使用上的差异，词典定义和编程历史的参考影响了各自的观点。
- **图像生成工作流中的挑战**：一位用户表示，尽管使用了正确的 prompt 和风格指南，但在生成跨角色的一致性图像时仍面临困难，并寻求潜在解决方案的建议。
  
  - 建议包括尝试 image-to-image 生成方法，以及利用 Civitai 等平台上提供的工作流。
- **ControlNet 在 Large Turbo 上的表现**：成员们讨论了 ControlNet 功能在与 3.5 large turbo 模型配合使用时的有效性，其中一人确认了其性能表现。
  
  - 这场对话引发了人们对使用新模型时的兼容性和所达成效果的兴趣与好奇。
- **创作高质量图像**：围绕制作高质量角色肖像的讨论强调了时间、探索以及尝试不同 prompt 和风格的重要性。
  
  - 鼓励用户查看成功图像的工作流，并考虑使用不同的技术来提升输出质量。
- **Stable Diffusion 插件问题**：一场关于过时的 SD 扩展导致 checkpoint 兼容性问题的讨论展开，促使一位用户寻求更新。
  
  - 社区强调检查插件的 readme 文件以获取更新和最佳实践，而一位用户分享了尽管进行了故障排除，某些 checkpoint 仍存在持续性问题。

 

**提到的链接**：[WILD CARD 的定义](https://www.merriam-webster.com/dictionary/wild%20card)：一个未知或不可预测的因素；在常规预选赛选手全部确定后，被选中填补剩余季后赛或锦标赛名额的人……查看完整定义

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1311084979857522778) (9 条消息🔥):

> `NotebookLM 实验、AI 视频展示、模型对比、讽刺文学写作、AI 交互见解`

- **在 NotebookLM 中实验讽刺文章**：一位成员正在尝试将讽刺文章输入 NotebookLM，并注意到它大约有一半的时间能识别出笑话，而另一半时间则不能。
  
  - 他们专门指示模型去质疑作者的人性，从而产生了有趣的结果，并在一段 [YouTube 视频](https://www.youtube.com/watch?v=e-OipxzsqtU&t=2s)中进行了展示。
- **AI 在创意展示中登台**：在一段新视频中，AI 展示了其在语音和视频方面的能力，同时幽默地分享了它的缺陷，特别是没有手指这一点。
  
  - 该成员鼓励所有观众——无论是爱好者、怀疑论者还是好奇者——去探索视频中呈现的令人兴奋的可能性，视频提供 [英文](https://youtu.be/ttDOBb5NYiQ?feature=shared) 和 [德文](https://youtu.be/iZci0WpAmGY?feature=shared) 版本。
- **Google AI Studio 中 Gemini 模型的对比**：一位成员对两个 Gemini 模型进行了对比，并在 NotebookLM 中总结了对话，希望这些见解能传达给 Google 团队。
  
  - 他们提供了 [AI Studio 对话链接](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2216h7Ioo6fmgSMtCWoSWBx8vK-kHEjsGHo%22%5D,%22action%22:%22open%22,%22userId%22:%22105185943804239990679%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing) 并分享了来自 NotebookLM 的音频概览。
- **关于 AI 递归困境的讨论**：一位成员指出，最新版本的 NotebookLM 在反复处理相似材料时（特别是在根据之前的指令创建文档时）难以处理递归问题。
  
  - 他们分享了一段强调此问题的音频剪辑，以及[相关音频的链接](https://cdn.discordapp.com/attachments/1124403655819415592/1311084979484360724/VPH03.mp3?ex=6748e409&is=67479289&hm=d1335b8b3e0c163172776781b55707f3dda695e39fbf9e25015560570536c9a8&)。
- **对 AI 模型描述的担忧**：一位成员质疑将 4o-mini 归类为“先进模型”的合理性，并将其与在解决渡河谜题时表现挣扎的 o1-preview 进行了对比。
  
  - 他们分享了一个音频文件，详细描述了这些模型被察觉到的局限性，详见附带的讨论。

**提到的链接**：

- [Deep Dive AI 阅读 Hugh Mann 的费米悖论论文](https://www.youtube.com/watch?v=e-OipxzsqtU&t=2s)：我将 Hugh Mann 的费米悖论论文以及他的 UFO/纳斯卡木乃伊揭秘论文输入到 Google 的 NotebookLM Deep Dive AI 播客生成器中，它们...
- [未找到标题](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%2216h7Ioo6fmgSMtCWoSWBx8vK-kHEjsGHo%22%5D,%22action%22:%22open%22,%22userId%22:%22105185943804239990679%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)：未找到描述
- [Complex stuff_ (1).wav](https://drive.google.com/file/d/1353DFDDSgslEcEYqc1VZi1apWdRnkdw_/view?usp=sharing)：未找到描述

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1311076062733668394) (72 条消息🔥🔥):

> `Notebook LM 支持问题, Notebook 共享限制, AI 内容生成, Podcast 长度自定义, 网络连接问题`

- **Notebook LM 功能问题**：用户报告了使用 Notebook LM 时的困难，例如笔记本上出现的黄色渐变警告似乎暗示了访问问题，以及对 AI 性能变得不稳定的投诉。
  
  - 用户担心现有的聊天会话是否会丢失，以及 Notebook LM 当前聊天系统的临时性（ephemeral nature）。
- **共享笔记本的挑战**：一位用户尝试将笔记本从个人 Gmail 共享到工作邮箱，但遇到了影响部分参与者的限制，尽管其中一人被成功添加。
  
  - 其他讨论包括如何更有效地利用教育资源，以及共享信息时的数隐私问题。
- **AI 内容生成的复杂性**：有用户在尝试让 Notebook LM 分析历年试卷以识别题型和答案模式时遇到困难，指出该模型经常提供过于宽泛的回答。
  
  - 有建议认为可能需要更多试卷才能开发出用于评分流程的可靠分析模型。
- **对服务定价的担忧**：关于 Notebook LM 是否会转为付费服务的猜测开始出现，因为官方承认未来可能会推出针对企业用户的特性。
  
  - 用户对免费服务的可持续性以及最终出现付费墙（paywalls）的风险表示怀疑。
- **Podcast 长度管理**：一位用户注意到生成的 Podcast 长度始终在 20 分钟左右，尽管设置了缩短长度的自定义指令，这引发了关于长度控制的询问。
  
  - 有建议提出，指示 AI 为忙碌的受众创建简洁内容，作为管理 Podcast 时长的一种潜在解决方案。

**提到的链接**：

- [Pola's Revenge by @sharednonlocality666 | Suno](https://suno.com/song/a3997712-f7c5-4da5-bc27-c5ed9c3179d4)：诡异的 techno，女性合成人声，女性合成唱腔，诡异的小提琴，快速，精致，pystrance，诡异的歌曲。使用 Suno 聆听并创作你自己的作品。
- [Breaking News: Scrooge's SECRET REVEALED! Ghostly Encounter Changes EVERYTHING](https://youtu.be/YMr9KB7-_uc?si=_oHcJXbqjWUdmhcn)：来自 Cratchit 县的突发新闻让全镇都在议论！以吝啬闻名的 Ebenezer Scrooge 似乎经历了一场戏剧性的转变...
- [One billion gagillion fafillion shabadabalo shabadamillion shabaling shabalomillion yen](https://youtu.be/ngKT3MIfwpo?si=ls5ElHZa_SHPZDbQ)：未找到描述
- [NotebookLM | Note Taking & Research Assistant Powered by AI](https://notebooklm.google/business)：利用 AI 的力量进行快速总结和笔记，Notebook LM 是你强大的虚拟研究助手，植根于你可以信赖的信息。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1311146096168538194) (7 条消息):

> `Cohere 社区, 欢迎新成员`

- **对 Cohere 学习的热情**：成员们表达了加入 **Cohere community** 的热情，以及渴望了解更多其产品服务的愿望。
  
  - *Enisda* 特别表示他们迫不及待地想要开始，强调了这里热情的氛围。
- **社区欢迎新人**：成员们向包括 *Subarna_b* 和 *Enisda* 在内的新参与者表示了热烈欢迎，营造了友好的环境。
  
  - *xvarunx* 和其他人热情地问候了新人，培养了归属感。

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1311084150979166310) (68 messages🔥🔥):

> `Cohere API 与 LiteLLM 的使用，Cohere 模型集成挑战`

- **用户在 LiteLLM 集成方面面临挑战**：用户表示在将 Cohere API 与 LiteLLM 结合使用时遇到困难，主要是因为 citations 功能无法按预期工作。
  
  - 他们强调 LiteLLM 作为一个 meta library，用于连接多个 LLM 提供商，并请求 Cohere 团队提供支持以改进集成。
- **引用支持的潜在增强方案**：有人指出，目前的 LiteLLM 实现不支持 Cohere chat endpoint 返回的 citations，这限制了可用性。
  
  - 建议在 LiteLLM 代码中添加新参数以允许处理 citation，用户愿意为此做出贡献或等待维护者的回复。
- **社区对模型使用见解的兴趣**：一位 Cohere 团队成员对了解社区如何使用他们的模型表示兴奋，并对用户的参与表示感谢。
  
  - 这种互动展示了协助优化功能实现和改善用户体验的意愿。

**提到的链接**：

- [[Feature]: Add citations support to completion for Cohere models · Issue #6814 · BerriAI/litellm](https://github.com/BerriAI/litellm/issues/6814)：Cohere 支持的 Feature Models 在 completion 请求中添加 documents 变量时会返回 citations（无论 stream=True 还是 stream=False）。作为响应，chat completion 会返回 citations...
- [Quick Start | OpenRouter](https://openrouter.ai/docs#models.)：开始使用 OpenRouter 进行构建
- [litellm/litellm/llms/cohere/chat.py at main · BerriAI/litellm](https://github.com/BerriAI/litellm/blob/main/litellm/llms/cohere/chat.py)：Python SDK，以 OpenAI 格式调用 100 多个 LLM API 的 Proxy Server (LLM Gateway) - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1311410094558416920) (1 messages):

> `全栈 AI 工程，Web 应用程序开发，AI 驱动的解决方案，容器化与部署，模型训练与部署`

- **拥有多样化技能的全栈 AI 工程师**：一位成员分享了他们作为 **Full Stack AI Engineer** 的专业经验，拥有超过 **6 年** 设计和部署可扩展 Web 应用程序及 AI 驱动解决方案的经验，涵盖前端和后端技术。
  
  - 他们强调了在 **React**、**Angular** 以及 **Django** 和 **FastAPI** 等各种后端框架方面的熟练程度。
- **容器化和微服务专业知识**：他们强调了在 **Docker** 容器化和 **Kubernetes** 编排方面的经验，以及在 **AWS**、**GCP** 和 **Azure** 等云平台上进行无缝部署的 CI/CD 流水线经验。
  
  - 他们的方案确保了微服务和 serverless 架构中**安全**、**可扩展**且**易于维护**的解决方案。
- **利用知识图谱和向量数据库进行 AI 开发**：在 AI 专业领域，他们使用**知识图谱 (knowledge graphs)** 以及 **Pinecone** 和 **Weaviate** 等向量数据库开发**自主 AI Agent**。
  
  - 他们利用 **TensorFlow** 和 **PyTorch** 执行 **NLP**、**计算机视觉**和推荐系统等任务。
- **高级 NLP 技术**：该成员精通部署 **LLMs (Large Language Models)**，并利用 **Hugging Face Transformers** 和 **spaCy** 等库执行高级 NLP 任务。
  
  - 他们的专业知识包括 embedding 模型和向量搜索，这对于在 AI 应用中解释复杂输入至关重要。
- **在 GitHub 上展示项目**：为了探索他们的工作，该成员引导他人访问他们的 GitHub 仓库：[AIXerum](https://github.com/AIXerum)。
  
  - 他们欢迎关于 AI、软件工程和解决技术挑战的讨论。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1311085812980383779) (55 messages🔥🔥):

> `PlayAI 融资，OLMo 2 发布，SmolVLM 介绍，Deepseek AI 进展，企业级生成式 AI`

- **PlayAI 获得 2100 万美元融资**：PlayAI 已从包括 Kindred Ventures 和 Y Combinator 在内的知名投资者处筹集了 **2100 万美元**资金，用于为开发者和企业开发直观的语音 AI 接口。
  
  - 此次融资旨在增强人机交互，专注于将无缝的语音优先接口作为一种自然的通信媒介。
- **OLMo 2 超越竞争对手**：**OLMo 2** 的发布引入了新的模型（7B 和 13B），其性能超越了其他开源模型，这让高管们对其在 AI 应用中的潜力感到兴奋。

- 这些模型据称在递归推理（recursive reasoning）等功能方面表现出色，并将用于各种 AI 场景，反映了 AI 技术的最新进展。
- **SmolVLM 发布，专为设备端使用设计**：SmolVLM 是一款 **2B VLM**，旨在实现设备端推理，在 GPU RAM 占用和 Token 吞吐量方面优于竞争对手。
  
  - 该模型可以在 Google Colab 上进行微调，并针对需要在消费级硬件上进行高效处理的用例进行了定制。
- **Deepseek 成为关键参与者**：据报道，Deepseek 最近的 AI 模型在推理基准测试中超越了 OpenAI，引起了 AI 社区的关注。
  
  - 在中国对冲基金幻方量化（High-Flyer）的支持下，Deepseek 致力于构建基础技术并提供价格实惠的 API。
- **2024 年生成式 AI 支出激增**：2024 年生成式 AI 支出飙升至 **138 亿美元**，表明企业正从实验阶段转向执行阶段。
  
  - 尽管对生成式 AI 的广泛采用持乐观态度，但决策者在定义有效的实施策略方面仍面临挑战。

**提到的链接**：

- [Ai2 (@allen_ai) 的推文](https://x.com/allen_ai/status/1861511421064028646?s=46)：认识 OLMo 2，迄今为止最好的全开放语言模型，包括经过多达 5T Tokens 训练的 7B 和 13B 模型系列。OLMo 2 的表现优于其他全开放模型，并可与开放权重模型竞争...
- [David Singleton (@dps) 的推文](https://x.com/dps/status/1861413927856546187?s=46)：🚀 很高兴宣布我的新公司：/dev/agents。我正在为 AI agents 构建下一代操作系统，与前同事 @hbarra、@alcor 和 @ficus 共同创立。我们非常激动...
- [Pedro Cuenca (@pcuenq) 的推文](https://x.com/pcuenq/status/1861445806206939446?s=46)：SmolVLM 刚刚发布 🚀 这是一个非常棒、小巧且完全开放的 VLM，我对其在微调和设备端用例中的表现感到非常兴奋 💻 它还通过 mlx-vlm 提供了 0-day MLX 支持...
- [Chain of Code: Reasoning with a Language Model-Augmented Code Emulator](https://chain-of-code.github.io/)：Chain of Code：使用语言模型增强的代码模拟器进行推理的项目页面。
- [PlayAI 筹集 2100 万美元资金并发布新的语音模型](https://blog.play.ai/blog/21m-funding)：PlayAI 是一家语音 AI 公司，致力于为实时对话构建令人愉悦且功能强大的语音 Agents 和语音界面，目前已筹集 2100 万美元种子轮融资。
- [Ben (e/treats) (@andersonbcdefg) 的推文](https://x.com/andersonbcdefg/status/1861516338948198470)：在单位资源产出方面，没有哪家前沿模型公司比 Allen AI 做得更好。这些人简直强得离谱。随着大型实验室变得僵化和更具 Google 风格，有理由认为这可能会...
- [2024：企业生成式 AI 现状 - Menlo Ventures](https://menlovc.com/2024-the-state-of-generative-ai-in-the-enterprise/)：企业 AI 格局正在实时重写。我们调查了 600 名美国企业 IT 决策者，以揭示新兴的赢家和输家。
- [Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1861802181587771627)：在 Ai2 发布多个最先进 AI 模型的疯狂几周。过去一个月里发布了 Tulu 3 后训练、OLMo2、Molmo。我学到了很多关于如何打造优秀模型训练团队的知识...
- [Mike Bird (@MikeBirdTech) 的推文](https://x.com/MikeBirdTech/status/1861880365058584588)：QWQReasoning + 歧义 != 答案 的快速测试。
- [我这周读的论文：视觉语言模型](https://www.artfintel.com/p/papers-ive-read-this-week-vision)：他们一直在发布 VLM，所以我一直在写...
- [Matt Shumer (@mattshumer_) 的推文](https://x.com/mattshumer_/status/1861511649934614906?s=46)：介绍 OpenReasoningEngine，一个开源的测试时计算（test-time-compute）引擎，可用于任何兼容 OpenAI 的模型。支持图像输入、函数调用、基础持续学习等。这是一个早期的...
- [Hugo Barra (@hbarra) 的推文](https://x.com/hbarra/status/1861414956023062850?s=46)：我正与一些我合作过的最优秀的人一起创办一家新公司，感到无比激动。我们称之为 /dev/agents。回到我们的 Android 根源，构建一个新的操作系统...
- [清晰与模糊任务](https://aligned.substack.com/p/crisp-and-fuzzy-tasks)：为什么模糊任务很重要，以及如何让模型与其对齐。
- [Andi Marafioti (@andi_marafioti) 的推文](https://x.com/andi_marafioti/status/1861437314351632662?s=46)：走起！我们正在发布 SmolVLM，一个专为设备端推理构建的小型 2B VLM，在类似的 GPU RAM 占用和 Token 吞吐量下优于所有模型。SmolVLM 可以在 Google Colab 上进行微调...

- [来自 Kyle Lo (@kylelostat) 的推文](https://x.com/kylelostat/status/1861514222842323295?s=46)：很高兴分享 OLMo 2！🐟 7B 和 13B 权重，高达 4-5T tokens，完全开源的数据、代码等 🐠 更好的架构和训练稳定性方案 🐡 阶段性训练，采用了新的数据混合方案 Dolmino🍕durin...
- [来自 Zihan Wang (@wzihanw) 的推文](https://x.com/wzihanw/status/1861263524242042923)：@EMostaque @deepseek_ai 我协助将这段采访翻译成了英文，以帮助大家更多地了解 DeepSeek 创始人——梁文锋 🐋。点击此处查看 ↓ https://drive.google.com/file/d/1DW5ohZWxoC...
- [Qwen/QwQ-32B-Preview · Hugging Face](https://huggingface.co/Qwen/QwQ-32B-Preview)：未找到描述
- [QwQ：深入反思未知的边界](https://qwenlm.github.io/blog/qwq-32b-preview/)：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD。注：这是 QwQ 的发音：/kwju:/ ，类似于单词 “quill”。思考、质疑、理解意味着什么？这些是...
- [Deepseek：引领中国 AI 竞赛的沉默巨人](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas)：其 CEO 最深度采访的带注释翻译版
- [Ask HN：为寻求跟上 AI 最新进展的 SWE 提供的建议 | Hacker News](https://news.ycombinator.com/item?id=42256093)：未找到描述
- [来自 Eugene Yan (@eugeneyan) 的推文](https://x.com/eugeneyan/status/1861822326087753922)：很高兴在 HN 上被提及，为那些学习 AI 的工程师们提供参考 🥰 帮助他人是我写作的一大动力：## 构建 AI 系统 • 构建基于 LLM 系统的方法模式：https://eugeneyan.com/writing/llm...

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1311432703698276423) (4 条消息):

> `投票改进建议, 课程通用设置, 个人使用的 GPU 获取途径`

- **投票改进建议**：一位成员征求关于投票的反馈，表示愿意改进其格式或将其转换为 Google Form 以提高易用性。
  
  - *如果大家更倾向于那样，我可以删除这个投票，做一个 Google Form 并把结果发在这里*。
- **分享课程设置经验**：另一位成员询问有经验的人士在开始一门课程时通常会采用什么样的通用设置。
  
  - 该询问旨在为新学习者收集优化配置的见解。
- **GPU 获取途径见解**：一位参与者指出，许多人可以通过其公司或大学获取 GPU，这些资源可以用于个人项目。
  
  - 这强调了需要计算能力的学习者可能获得的潜在资源。

 

---

### **GPU MODE ▷ #**[**cuda**](https://discord.com/channels/1189498204333543425/1189607726595194971/1311103013787664444) (3 条消息):

> `Kernel Fusion, Model Traces, 减少 CUDA Kernel 启动开销`

- **关于 Kernel Fusion 技术的见解**：一场关于 [Kernel Fusion 技术](https://www.kapilsharma.dev/posts/cuda-mode-fusing-kernels-talk/) 的过往演讲强调了其在 Triton 和 CUDA 环境中的相关性。
  
  - 对 Kernel Fusion 的关注源于其通过优化操作执行方式来提升性能的潜力。
- **Model Traces 作为资源管理工具**：文章强调了广泛使用 **Model Traces** 来分析操作调度、计时和资源使用。
  
  - 这种方法可以精确定位有利于资源重用的操作，从而避免将数据不必要地重新加载到寄存器中。
- **关于减少 Kernel 启动开销的讨论**：[Twitter](https://x.com/mike64_t/status/1861829940167188902) 上的一条长推文探讨了最小化 **CUDA Kernel 启动开销** 的策略，其中包含来自 librecuda 作者的见解。
  
  - 对话显示，修改 launch bounds 可能会显著影响与 Kernel 启动相关的性能增强。

 

**提到的链接**：[来自 mike64_t (@mike64_t) 的推文](https://x.com/mike64_t/status/1861829940167188902)：@memorypaladin @ID_AA_Carmack @ezyang 如果这个理论成立，更改 launch bounds 应该会起到类似的作用。

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1311116183084335226) (9 messages🔥):

> `FSDP multiple ranks, FLOPS counting challenges, GPU performance discrepancies, Math library workspace issues, Efficient use of FLOP counters`

- **如何在 FSDP 中加载预训练模型？**: 为了在 FSDP2 的多个 rank 之间加载预训练模型，必须在加载前**手动分片 (manually shard)** 权重，如[此示例](https://github.com/pytorch/torchtune/blob/b5d2e6372017c163914b13b2514f29914e5dbb84/torchtune/training/_distributed.py#L150-L223)所示。该过程涉及在每个 rank 上初始化完整模型，并逐步对权重进行分片，而无需在任何单个 GPU 上实例化完整的模型权重。
- **FLOPS 计数的挑战**: 统计 **FLOPS** 可能会有问题，因为许多操作会逃避现有脚本的检测，导致错误的 flop 计数，从而影响研究的可靠性。建议的解决方案包括使用 [fvcore](https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md) 和 [torch_flops](https://github.com/zugexiaodui/torch_flops) 等工具进行更准确的测量。
- **GPGPU 性能差异**: 一位用户指出，他们的计算表明 GPT-2 Small 在 3080Ti 上的性能应达到 **>3k tok/s**，但在实践中并未达到。人们对数学库中不同的 **kernels** 和 **temporary workspaces** 等潜在因素表示担忧，这些因素可能会导致意外的内存使用。
- **Workspace 问题可能导致 OOM**: 数学库中临时 workspace 的大小因硬件而异，可能导致显存溢出 (OOM)。用户可以覆盖 workspace 设置（例如 `CUBLAS_WORKSPACE_CONFIG`）以更好地管理内存，尽管硬件更改仍可能使问题复杂化。
- **FLOPS 计数的细微差别**: 准确统计 **FLOPS** 是非常微妙的，因为即使检测到了操作，仍存在可能影响结果的细微之处。一位用户对他们的 **GPT-2** 模型性能表示沮丧，怀疑缺失的 FLOPS 统计可能导致了性能差异。

 

**Link mentioned**: [torchtune/torchtune/training/_distributed.py at b5d2e6372017c163914b13b2514f29914e5dbb84 · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/b5d2e6372017c163914b13b2514f29914e5dbb84/torchtune/training/_distributed.py#L150-L223): PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**algorithms**](https://discord.com/channels/1189498204333543425/1189861061151690822/1311399646958059621) (4 messages):

> `LoLCATs paper, ThunderKittens kernel, Model Throughput Issues, Linearized Attention Performance`

- **LoLCATs 论文见解**: 围绕 [LoLCATs 论文](https://arxiv.org/abs/2410.10254) 展开了讨论，该论文专注于在不进行全模型微调的情况下线性化 LLM，并指出在小 batch size 下，其性能下降到 **FA2** 的**一半吞吐量**。
  
  - 有趣的是，线性化模型显示出内存节省，但未能超越之前预期的**二次复杂度 Attention 模型 (quadratic attention model)**。
- **质疑 Kernel 效率**: LoLCATs 实现中使用了哪些 kernel 尚不明确，特别是是否使用了 **ThunderKittens** kernel 或效率较低的 kernel。
  
  - 成员们根据附录推断结果可能表明使用了 ThunderKittens kernel，但这引发了关于整体效率的疑问。
- **MLP 大小 vs. Context Window**: 讨论强调了对 **4k context window** 可能太小而无法在考虑 MLP 权重矩阵大小时显著影响性能的担忧。
  
  - 权重矩阵属性显示 MLP 大小超过了 KV 矩阵维度，可能影响吞吐量。
- **算法增益 vs. 性能**: 参与者指出，在 **4k sequences** 下，Attention 算子带来的算法增益可能会被线性投影和完整的 FFN 块所掩盖。
  
  - 尽管有这些预期，但如论文初步结果所示，吞吐量仍存在令人惊讶的差距。

 

**Link mentioned**: [Linearizing LLMs with LoLCATs](https://hazyresearch.stanford.edu/blog/2024-10-14-lolcats-p1): 未找到描述

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1311144192034213938) (11 条消息🔥):

> `#pragma unroll 的用法，GPU 工作的机器规格，NVIDIA vs AMD GPU，近期 GPU 架构的重要性`

- **理解 #pragma unroll**：成员们讨论了在 CUDA kernel 中使用 `#pragma unroll` 的直觉，强调当为了性能需要完全展开（complete unrolling）时经常使用它，特别是为了将数组保留在 registers 中而不是 local memory 中。
  
  - 一位成员指出，展开大循环可以增强 pipeline parallelism，尽管这可能会增加 register pressure 和代码体积。
- **机器学习任务的规格**：一位成员寻求关于 GPU 工作的机器规格建议，促使其他人澄清 RAM 和 GPU 是最重要的组件。
  
  - 另一位成员强调，虽然没有“万能”规格，但与 AMD 相比，使用 NVIDIA GPU 通常提供更好的性能。
- **NVIDIA GPU 提供更好的性能**：成员们一致认为选择 NVIDIA GPU 是明智的，**3060** 因其 **12GB VRAM** 被提及为一个稳妥的选择。
  
  - 对话指出，对于 kernel 开发，新架构优先于单纯的原始速度，因为它们支持 `bf16` 和 `fp8` 等高级特性。

 

---

### **GPU MODE ▷ #**[**llmdotc**](https://discord.com/channels/1189498204333543425/1227345713348870156/1311151329279873114) (2 条消息):

> `freeCodeCamp 上的 CUDA 课程，cublaslt 性能`

- **查看 YouTube 上的 CUDA 课程**：一位成员建议查看 YouTube 上的 [freeCodeCamp CUDA 课程](https://www.freecodecamp.org) 以获取学习资源。
  
  - 他们强调了该课程对于深入了解 CUDA 编程的相关性。
- **cublaslt 处理大矩阵最快**：一位成员指出 **cublaslt** 是处理 **low precision** 大矩阵的最佳选择，展示了其令人印象深刻的速度。
  
  - 他们为那些热衷于优化矩阵运算性能的人推荐了这一部分。

 

---

### **GPU MODE ▷ #**[**intel**](https://discord.com/channels/1189498204333543425/1233802893786746880/) (1 条消息):

binarysoloist: 关于 Meteor Lake NPU 的第一个 Google 搜索结果显示它是 shared memory

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1311102699164532817) (5 条消息):

> `主板更换，GPU 性能，CMOS 电池检查，Pull Request 评论`

- **烧毁的主板需要更换**：一位成员报告说他们的主板**烧毁**了，在几乎完成一项任务后**需要更换**。
  
  - 另一位成员对此评论道：*这下真的让 GPU 飞起来了（go brrrrr）*。
- **检查 CMOS 电池**：鉴于主板问题，一位成员建议检查 **CMOS 电池** 是否需要更换。
  
  - 这一建议为排查主板故障提供了一个技术角度。
- **Pull Request 预期**：该成员表示他们提交了一个 **pull request**，但预计会有很多评论，参考编号为 **#410**。
  
  - 他们宣称：*它还活着……不知怎么地*，暗示了尽管硬件出现故障，他们仍在坚持。
- **GPU 疯狂运转**：另一位成员对这种情况评论道：*这家伙干得太卖力了（cooking so hard）*。
  
  - 这一轻松的评论反映了尽管存在技术问题，但围绕正在进行的 GPU 活动的兴奋感。

 

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1311406596768792708) (2 messages):

> `CUDA bot usage, VS Code extension for AI coding, GitHub Copilot customization`

- **衡量 CUDA Bot 的受欢迎程度**：一位成员建议，衡量 **CUDA bot** 成功与否的一种方法是跟踪用户参与度，并提议开发一个 **VS Code 扩展**来探索其利用率。
  
  - 他们提到目前该工具利用了 **Claude** 和来自 **Thunderkittens** 的随机 prompts，但在支持开源模型和增强 UX 方面仍有改进空间。
- **介绍 LLM Coder 项目**：一个名为 [llm_coder](https://github.com/msaroufim/llm_coder) 的新项目旨在通过在 prompts 中提供主要 API 并将其集成到 **VS Code** 中，帮助 **Claude** 更好地理解库。
  
  - 项目描述强调了将 **Claude** 与库连接起来以改进编码建议，该成员邀请其他人表达兴趣。
- **简化 Copilot 的自定义指令**：一位成员指出了一种通过 [自定义指令](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot) 增强 **GitHub Copilot** 的更直接的方法。
  
  - 该资源旨在简化自定义过程，使开发者更容易定制 Copilot 的功能。

**提到的链接**：

- [Adding custom instructions for GitHub Copilot - GitHub Docs](https://docs.github.com/en/copilot/customizing-copilot/adding-custom-instructions-for-github-copilot)：未找到描述
- [GitHub - msaroufim/llm_coder: Help Claude know about your library by giving it the main APIs in a prompt and integrate it into VS Code](https://github.com/msaroufim/llm_coder)：通过在 prompt 中提供主要 API 并将其集成到 VS Code 中，帮助 Claude 了解你的库 - msaroufim/llm_coder

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1311431034029473794) (3 messages):

> `FP8 support, ThunderKittens kernels`

- **ThunderKittens 引入 FP8 支持**：今天 **ThunderKittens** 正式发布了 **FP8 支持**和 **fp8 kernels**，响应了社区对量化数据类型的需求。查看 [博客文章](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8) 了解完整的特性详情。
  
  - 该实现在仅 **95 行代码**中就达到了 **1500 TFLOPS**，展示了其效率和有效性。
- **分享 Kernel 实现细节**：**ThunderKittens** 的重点一直是简化 kernel 编写并促进对新架构的研究。用户可以 [在此](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/FP8) 进一步探索 kernel 的实现。
  
  - 项目背后的团队包括 [Benjamin Spector](https://benjaminfspector.com/) 和 [Chris Ré](https://cs.stanford.edu/people/chrismre/) 等知名成员，他们为项目的开发做出了贡献。

**提到的链接**：

- [ThunderKittens: Bringing fp8 to theaters near you](https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8)：未找到描述
- [ThunderKittens/kernels/matmul/FP8 at main · HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/matmul/FP8)：用于快速 kernel 的 Tile primitives。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1311109592771596408) (4 条消息):

> `Azure OpenAI 终端节点, 用于 RAG 的 CXL 内存, 质量感知文档聊天机器人, MSIgnite 发布会, LlamaParse 功能`

- **LlamaParse 支持 Azure OpenAI 终端节点！**: LlamaParse 现在支持 [Azure OpenAI 终端节点](https://twitter.com/llama_index/status/1861550505761349761)，在确保企业级安全性的同时，增强了其作为复杂文档格式解析器的能力。
  
  - 此次集成允许用户通过定制的 API 终端节点，在应用程序中有效地管理敏感数据。
- **利用 CXL 内存提升 RAG 流水线！**: 来自 [MemVerge](https://twitter.com/llama_index/status/1861825056621600995) 的最新研究强调了使用 CXL 内存如何显著扩展 RAG 应用程序的可用内存。
  
  - 这一进展有望通过全内存操作来提高性能。
- **使用 LlamaIndex 构建文档聊天机器人！**: 通过结合用于文档摄取和检索的 LlamaIndex 以及用于监控的 @aimon_ai，你可以构建一个具有质量感知能力的文档聊天机器人，主动检查幻觉等问题。
  
  - LlamaIndex 使用 @milvusio 作为向量存储，确保聊天机器人能够高效且有效地检索数据。
- **来自 #MSIgnite 的激动人心的发布！**: 在 #MSIgnite 期间，Farzad Sunavala 和 @seldo 在分论坛会议中展示了关于 [LlamaParse 和 LlamaCloud](https://twitter.com/llama_index/status/1861887478602592715) 的重大发布。
  
  - 演示包含了 **多模态解析 (multimodal parsing)** 等功能，展示了 LlamaParse 在各种格式下的处理能力。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1311081507540045876) (18 条消息🔥):

> `基于 Postgres 的 BM25 检索器, 在 Milvus 中加载索引, Ollama API 独立性, 使用 o1 模型提取 Pydantic 模型, 文档哈希对比`

- **BM25 检索器与 Postgres 兼容性**: 一位成员询问关于构建 **BM25 检索器** 并将其存储在 **Postgres 数据库** 中的问题。另一位成员回复称，需要一个 **BM25 扩展** 才能实现此功能。
- **在 ChromaDB 中从 Milvus 加载索引**: 一位成员询问如何将 **索引从 'milvus_local.db'** 加载到 Milvus 中，并提供了一个 ChromaDB 代码示例作为参考。建议指出，当指向数据库时，**MilvusVectorStore** 会自动检查数据库是否存在。
- **确保 Ollama API 调用的无状态性**: 一位成员询问如何确保对 Ollama 的 API 调用中 **没有共享上下文**，另一位成员澄清说，除非在调用中共享了特定数据，否则它基本上是无状态的。调用者的输入可以决定上下文的保留。
- **使用 o1 模型提取 Pydantic 模型**: 一位成员分享了使用 **o1-preview** 和 **o1-mini** 模型提取 Pydantic 模型时遇到的挑战，提到由于参数不支持导致调用失败。回复强调 o1 模型缺乏对结构化输出的支持，并建议将请求与 **gpt-4o-mini** 串联，但这种方法遭到了质疑。
- **设置文档哈希进行对比**: 一位成员询问是否有办法为 **文档设置哈希** 以在不使用元数据的情况下对比文本。回复指出哈希方法是硬编码的，建议通过子类化来修改对比逻辑。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1311204720634761248) (10 messages🔥):

> `TinyCloud Infrastructure, FPGA Backend Development, Cloud Access for Tinygrad Contributors, Intel/ARC Support Concerns, Tinybox Performance Focus`

- **TinyCloud 即将发布**：令人兴奋的 **TinyCloud** 即将到来，年底前将提供 **9 台 tinybox red (54x 7900XTX)** 供贡献者在 tinygrad 中通过 CLOUD=1 免费使用。
  
  - 由于采用了自定义驱动程序，该设置将非常稳定，用户只需使用 API key 即可享受简洁体验。
- **需要全职云基础设施开发人员**：GeorgeHotz 提到需要有人**全职负责云基础设施**，以进一步开发 TinyCloud 项目。
  
  - 除了这一需求，他还呼吁招募 **FPGA 后端**专家，欢迎感兴趣的候选人。
- **Tapeout 进度的先决条件**：列出了实现 Tapeout 就绪的关键**先决条件**，包括从 tinygrad 中移除 LLVM，以及支持 Qualcomm DSP 和 Google TPU。
  
  - 他还提到了对 **tinybox FPGA 版本**的需求，以及建立**主权 AMD 技术栈**的总目标。
- **提到 Intel/ARC 但缺乏关注**：一名成员对缺乏 **Intel/ARC** 支持讨论表示惊讶，询问是否已决定降低其优先级。
  
  - GeorgeHotz 回应称，这并没有什么“新意”，并将其称为“非常弱的 GPU”。
- **专注于 Tinybox 性能**：大家达成共识，不应支持每一种消费级配置，而应集中精力提高 **tinybox 的性能和可靠性**，特别是云端访问。
  
  - 这一战略重点被认为有利于项目的演进。

**相关链接**：[来自 tiny corp (@**tinygrad**) 的推文](https://x.com/__tinygrad__/status/1861645755452363011)：我们将在年底前在测试云中部署 9 台 tinybox red (54x 7900XTX)。（配合我们的自定义驱动程序运行稳定）将免费供 tinygrad 贡献者使用。使用它就像运行...一样简单。

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1311086002160402566) (11 messages🔥):

> `GPU Radix Sort Optimization, Handling Edge Cases, Sorting Algorithm Selection, Vectorization Techniques, Support for Bitonic Sort`

- **为各种数据类型优化 GPU 基数排序 (Radix Sort)**：用户分享了通过支持 **64-bit**、**负数**和**浮点数**来增强 GPU 基数排序的见解，同时强调了针对大数组性能的切块 (chunking) 处理。
  
  - 他们还提到了一个潜在的 **UOp.range()**，用于优化 Python 循环，并附带了[示例](https://github.com/tinygrad/tinygrad/blob/84f96e48a1bb8826d868ad19ea34ce2deb019ce1/examples/stunning_mnist.py#L29-L31)。
- **基数排序实现的边缘情况**：用户提供了一系列用于测试 GPU 基数排序的边缘情况，包括已排序、反序以及包含重复项或大数字的列表。
  
  - 他们还指出在排序算法中处理单元素和空 Tensor 的重要性。
- **根据上下文选择排序算法**：一名成员讨论了根据可用资源选择排序算法，建议如果没有 GPU，应优先考虑 **基数排序 (radix sort)** 的先前需求，而非 **双调排序 (bitonic sort)**。
  
  - 他们解释说，双调排序的优势源于其在 GPU 上的并行化能力。
- **探索排序的向量化**：用户询问了潜在的向量化技术，以优化排序算法中迭代数字并更新排序输出的部分。
  
  - 他们建议使用直方图 (histogram) 可能允许预填充每个位置的常量 Tensor，以实现高效赋值。
- **扩展排序中的 Swap 功能**：一名成员观察到，支持 Swap 函数可以增强排序能力，使其包括 **双调排序 (bitonic sorts)** 或 **合并冒泡排序 (merged bubble sorts)**。
  
  - 他们强调这些实现可以符号化地完成，或者在实际 Buffer 上就地 (in place) 完成。

**相关链接**：[tinygrad/examples/stunning_mnist.py](https://github.com/tinygrad/tinygrad/blob/84f96e48a1bb8826d868ad19ea34ce2deb019ce1/examples/stunning_mnist.py#L29-L31)：你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1311182479767638066) (15 条消息🔥):

> `教育聊天机器人开发, Torchtune 兼容性, LoRA 单设备 Recipe, Torchtune Commit 里程碑, Activation Offloading 与内存效率`

- **探索使用 Torchtune 开发教育聊天机器人**：一位用户正在使用 OpenAI assistants 开发专注于问答（QA）和网络安全等特定领域的教育聊天机器人，并寻求关于 Torchtune 兼容性和微调流程的指导。
  
  - *Torchtune 专注于开源模型*，这需要获取模型权重才能进行有效的微调。
- **庆祝 Torchtune 的第 1000 次 Commit**：一位成员祝贺团队在 Torchtune 主仓库实现了 **第 1000 次 Commit**，这体现了对项目的投入。
  
  - 附带的图片强调了这一里程碑，展示了团队的辛勤工作和奉献精神。
- **关于 LoRA 训练速度的讨论**：成员们讨论了 **LoRA 单设备 Recipe** 的性能，并询问了训练速度和收敛时间。
  
  - 一位成员指出，*将学习率提高 10 倍可以改善训练性能*，这表明了一个潜在的优化路径。
- **通过 Activation Offloading 提高内存效率**：对话涉及了 DPO 的 Activation Offloading，透露成员们并未观察到显著的内存增益。
  
  - 一位成员幽默地表达了他们的困惑，同时寻求关于一个可能揭示所遇问题的公开 PR 的澄清。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1311137492040286248) (8 条消息🔥):

> `Normal Mode vs OS Mode, OS Mode 需求, CLI vs GUI 功能, Open Interpreter Point API`

- **Normal Mode 限制与 OS Mode 的能力**：**Normal Mode** 通过 CLI 运行，而 **OS Mode** 提供 GUI 功能，需要多个模型进行控制。
  
  - 一位用户表示，他们打算将 OS Mode 用作通用网页爬虫，主要依赖 CLI 应用程序。
- **旧版 OS Mode 终端交互功能**：一位用户询问旧版 OS Mode 终端交互在不需要 GUI 的情况下是否仍然有效。
  
  - 前一条确认其运行状态的消息迅速解决了他们的问题。
- **Open Interpreter Point API 的问题**：一位用户报告了访问 **Open Interpreter Point API** 的困难，指出它似乎已宕机。
  
  - 在尝试使用该 API 时遇到了 **持续性错误**，引发了用户的担忧。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1311125813290078248) (6 条消息):

> `MCP 工具反馈, 已安装的服务器与工具, MCP Cheatsheet`

- **对 MCP 的兴奋**：一位成员对新的 **MCP 工具** 表示热烈欢迎，称其为 *疯狂* 且 **非常巨大（HUGE）**。
  
  - 这反映了社区内探索其功能的兴趣日益增长。
- **已安装 MCP Server 列表**：一位成员分享了他们为 MCP 安装的工具，包括 **Filesystem**、**Brave Search** 以及 **SQLite** 和 **PostgreSQL** 等多个服务器选项。
  
  - 他们计划在即将到来的开发项目中使用这些工具进行实验。
- **Cheatsheet 增强 MCP 使用**：另一位成员提供了两个 MCP Server 的 **Cheatsheet** 链接，其中包含文档中未提及的选项和设置。
  
  - 这些资源旨在帮助用户最大限度地利用 MCP，突显了社区间的见解分享。
- **寻求关于 Toolhouse.ai 工具的反馈**：一位成员询问了将来自 Toolhouse.ai 的 **30 多个工具** 转换为 MCP Server 以增加实用性的反馈。
  
  - 这表明用户正在共同努力增强工具集成并改进 MCP 框架内的功能。

**提到的链接**：

- [MCP_SERVERS_CHEATSHEET.md](https://gist.github.com/davidteren/f78f9a184488f8732f98938e88cdea2d)：GitHub Gist：立即分享代码、笔记和片段。
- [MCP_SERVER_CONFIG.md](https://gist.github.com/davidteren/80c1b3e1ee092113f6d3fe8a1b7572d7)：GitHub Gist：立即分享代码、笔记和片段。

---

### **Axolotl AI ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1311381732880154738) (5 条消息):

> `SmolLM2-1.7B, Transformers.js v3 发布, 前端 LLM 任务, Axolotl 全量微调, Qwen 2.5 模型配置`

- **SmolLM2-1.7B 发布引发热议**：社区对 [SmolLM2-1.7B](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU) 反应热烈，指出其在前端开发 LLM 任务中的重要意义。
  
  - 一位成员表示 *“这太疯狂了！”*，这表明了开发者在可访问性和能力方面的转变。
- **Transformers.js v3 引入重大增强**：Hugging Face 宣布发布 🤗 Transformers.js v3，其特点是支持 [WebGPU](https://huggingface.co/blog/transformersjs-v3)，性能比 **WASM 快达 100 倍**。
  
  - 新更新包括 **25 个新示例项目**和 **120 种支持的架构**，为开发者提供了丰富的资源。
- **前端 LLM 任务成为现实**：一位成员强调了将 LLM 任务集成到前端的转变，反映了应用开发的一次重大演进。
  
  - 这一进步凸显了当今开发者所拥有的前所未有的能力。
- **关于 Qwen 2.5 全量微调的咨询**：一位用户寻求在 Axolotl 中为 **Qwen 2.5** 模型配置全量微调（full fine tuning）的指导，并引用了影响训练效果的参数。
  
  - 针对特定 **unfrozen_parameters** 必要性的疑问随之产生，表明需要更好地理解模型配置。
- **寻求模型配置知识**：在各种模型的微调背景下，用户表示有兴趣学习如何有效地掌握配置设置。
  
  - 一位用户提出了关于如何获取未来类似模型咨询指导的问题，突显了社区内持续的学习氛围。

**提到的链接**：

- [SmolLM2 1.7B Instruct WebGPU - HuggingFaceTB 的 Hugging Face Space](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU)：未找到描述
- [Transformers.js v3: WebGPU 支持、新模型与任务等……](https://huggingface.co/blog/transformersjs-v3)：未找到描述

---

### **MLOps @Chipro ▷ #**[**events**](https://discord.com/channels/814557108065534033/869270934773727272/1311351166608867430) (2 条消息):

> `Feature Store 网络研讨会, Multi-Agent 框架训练营`

- **参加我们的免费 Feature Store 网络研讨会**：参加我们今年最后一场网络研讨会，时间为 **太平洋时间 12 月 3 日上午 8 点**，主题为 *使用 Featureform 和 Databricks 构建企业级 Feature Store*，由创始人 **Simba Khadder** 主讲。深入了解 Feature Store 如何通过简化 ML 生态系统中的三种主要类型，成为管理大规模数据流水线的关键。
  
  - 本次会议将涵盖处理 **PB 级**数据和使用 **Apache Iceberg** 进行版本控制等主题，同时提供增强 ML 项目的可操作见解；[在此注册](https://buff.ly/3OqcG1V)！
- **GitHub 总部的 Multi-Agent 框架训练营**：1 月 **12 月 4 日**加入我们，参加专注于 Multi-Agent 框架的训练营，活动将在 **GitHub 总部**举行，包含专家演讲和工作坊。在享受免费食物和饮料的同时，与行业领袖建立联系；[在此注册](https://lu.ma/multi-agent-meetup)。
  
  - 议程包括 *使用 CrewAI 自动化繁琐任务* 和 *通过评估构建生产级 Agent* 等环节，由 **Lorenze Jay** 和 **John Gilhuly** 等专业人士主讲。

**提到的链接**：

- [开发者训练营：掌握 Multi-Agent 框架与评估技术 · Luma](https://lu.ma/multi-agent-meetup)：加入我们在 GitHub 办公室举办的活动，享受一个充满专家演讲、动手工作坊以及与 Multi-Agent 系统领先专业人士交流机会的夜晚……
- [ML Feature Lakehouse：使用 Iceberg 构建 PB 级数据流水线](https://buff.ly/3OqcG1V)：加入我们 1 小时的网络研讨会，我们将深入探讨不同类型的 Feature Store 及其用例！

---

### **MLOps @Chipro ▷ #**[**general-ml**](https://discord.com/channels/814557108065534033/828325357102432327/1311311632126971944) (2 messages):

> `LLMOps 资源，大语言模型的影响`

- **LLMOps 必读资源**：一位成员分享了[学习 LLMOps 的极佳资源](https://dub.sh/everything-about-llmops)，强调了其三个组成部分，并鼓励他人收藏。
  
  - 他们强调了由**大语言模型 (LLM)** 驱动的变革浪潮，以及作为 **LLMOps** 出现的运营框架。
- **LLM 正在革新技术**：讨论详细说明了 **LLM** 如何改变与技术的交互，为聊天机器人、虚拟助手和高级搜索引擎等应用提供动力。
  
  - 提到了它们在创建**个性化推荐系统**中的作用，这标志着运营方法的重大转变。

 

**提到的链接**：[LLMOps 第一部分：简介](https://dub.sh/everything-about-llmops)：世界正在经历一场由大语言模型 (LLM) 驱动的变革浪潮。这些先进的 AI 模型能够理解和生成人类质量的文本，正在改变交互方式...

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1311340598866874429) (4 messages):

> `音频数据集字幕生成，Faster Whisper 批处理，字幕生成脚本，音频合并策略`

- **寻求使用 Whisper 进行高效音频字幕生成**：一位用户表示有兴趣使用 **Whisper** 快速为音频数据集生成字幕，但面临短音频文件不符合批处理要求的问题。
  
  - 他们指出，批处理显著缩短了处理时间，将 **13 分钟**的音频处理时间从 **1 分钟**缩短到了更高效的 **17 秒**。
- **分享当前的字幕生成脚本**：用户分享了他们当前的音频字幕生成脚本链接 ([test1.py](https://cdn.discordapp.com/attachments/823813160075132991/1311340795458224199/test1.py?ex=674880c9&is=67472f49&hm=79ecab3679dd73ee14a376b3ce78ca44b1399b47891b9a283b1d15ce0e002c66&))。
  
  - 他们指出了该脚本在高效处理短音频文件进行批处理方面的局限性。
- **合并短音频文件的策略**：用户提出了一个想法，即将多个短音频文件合并在一起，为 **Whisper** 处理创建一个更长的输入。
  
  - 他们建议，尽管最初担心质量问题，但这种方法可能允许捕获带有时间戳的文本。
- **Faster Whisper GitHub 资源**：用户引用了 GitHub 上的 [Faster Whisper](https://github.com/SYSTRAN/faster-whisper) 作为使用 CTranslate2 提高转录速度的资源。
  
  - 他们表达了利用该工具加快音频文件处理速度的动力。

 

**提到的链接**：[GitHub - SYSTRAN/faster-whisper: 使用 CTranslate2 的 Faster Whisper 转录](https://github.com/SYSTRAN/faster-whisper)：使用 CTranslate2 的 Faster Whisper 转录。通过在 GitHub 上创建账号为 SYSTRAN/faster-whisper 做出贡献。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1311188418453307402) (3 messages):

> `Llama 3.2 System Prompt, Multi Turn Categories Evaluation, Leaderboard Score Changes, Error Logs Observations`

- **Llama 3.2 使用 BFCL 默认提示词**：用户注意到 **Llama 3.2** 所使用的提示词是来自 **BFCL** 的默认系统提示词。
  
  - 这暗示了在评估模型性能时对标准配置的依赖。
- **多轮对话类别影响准确率**：于 **9 月中旬**引入的多轮对话类别由于其挑战性，导致 **v3** 版本发布后整体准确率出现明显下降。
  
  - 这一变化影响了平均分数，特别是对各种模型产生了影响，而 v1 和 v2 的变化极小。
- **排行榜分数显著波动**：**10/21** 和 **11/17** 的两次最新排行榜更新显示出显著的分数变化，这是由于为多轮对话类别引入了新的评估指标。
  
  - 之前正确的条目现在可能被标记为错误，这突显了旧版状态检查器的局限性以及 [PR #733](https://github.com/ShishirPatil/gorilla/pull/733) 中详述的新指标的改进。
- **即将公开生成结果**：宣布计划上传并公开分享用于排行榜检查点的所有生成结果，以便审查错误日志。
  
  - 这将为观察到的不同模型间的 Agent 行为差异提供见解。
- **提示词模型在多轮对话类别中表现挣扎**：观察发现，在多轮对话类别中，仅靠 Prompting 的模型表现往往优于其对应的 FC（Function Calling）模型。
  
  - 这引发了关于在挑战性评估场景中 Prompting 有效性的疑问。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1311217066736619571) (2 messages):

> `Quiz Score Notifications, Confirmation Emails`

- **缺失测验分数邮件**：一位成员表示担心在提交后没有收到关于测验分数的确认邮件。
  
  - 另一位成员建议检查**垃圾邮件文件夹**，或者尝试使用不同的电子邮件，以防提交未被记录。
- **排除邮件问题**：针对该询问，一位成员建议核实是否发送了来自 Google Forms 的确认邮件。
  
  - 他们强调，如果找不到邮件，很可能意味着提交没有成功。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1311366440997355654) (1 messages):

> `Hidden States Unconference, Local RAG Application Workshop, ESM-1 Protein Language Model Discussion, San Francisco Demo Night, Data Bias Seminar`

- **Hidden States Unconference 邀请 AI 创新者**：欢迎参加在旧金山举行的 **Hidden States Unconference**，这是一个研究人员和工程师的聚会，旨在 [12 月 5 日](https://discord.com/events/1089876418936180786/1304175188581290094) 探讨 AI 界面和隐藏状态。
  
  - 这一为期一天的活动旨在通过协作讨论推向 AI 方法的边界。
- **构建超轻量级 RAG 应用**：在 [12 月 10 日](https://discord.com/events/1089876418936180786/1293281470642651269) 的即将举行的工作坊中，学习如何使用 sqlite-vec、llamafile 和 Python 创建 **检索增强生成 (RAG)** 应用程序。
  
  - 参与者将体验在没有任何额外依赖的情况下构建应用。
- **以 ESM-1 开启生物表征学习**：**论文阅读俱乐部** 将在 [12 月 12 日](https://discord.com/events/1089876418936180786/1305611638979694623) 讨论 Meta AI 的 **ESM-1 蛋白质语言模型**，作为生物表征学习系列的一部分。
  
  - 本次会议旨在吸引参与者关注 AI 在生物研究中的创新应用。
- **Demo 之夜展示湾区创新**：参加 [12 月 15 日](https://discord.com/events/1089876418936180786/1305577025918074890) 的 **旧金山 Demo 之夜**，见证本地 AI 领域创作者的突破性演示。
  
  - 该活动由 **Gen AI Collective** 主办，重点展示技术与创意的交汇。
- **与 Mozilla 的 Linda Dounia Rebeiz 一起应对数据偏差**：在 [12 月 20 日](https://discord.com/events/1089876418936180786/1298252973373128725) 加入 TIME100 获奖者 **Linda Dounia Rebeiz**，了解她使用策划数据集训练无偏差 AI 的方法。
  
  - 她将讨论使 AI 能够反映现实而非强化偏见的策略。

 

---

### **AI21 Labs (Jamba) ▷ #**[**general-chat**](https://discord.com/channels/874538902696914944/874538902696914947/1311419467619766384) (1 条消息):

> `Jamba 1.5 Mini Model, Function Calling Issues, OpenRouter Performance, Password Change Request`

- **通过 OpenRouter 使用 Jamba 1.5 Mini Model**：一位成员提到尝试通过 **OpenRouter** 使用 AI21 Labs 的 **Jamba 1.5 Mini Model**，并处理特定的用户数据和密码更改请求。
  
  - 这涉及在发送给模型的 messages 中设置 location 和 username 等参数。
- **Function Calling 输出为空**：尽管进行了设置，但在调用 **Function Calling** 时，模型返回了空输出，显示的 JSON 响应中 **content** 字段为空。
  
  - 这表明存在潜在问题，因为同一位成员报告称，在不使用 **Function Calling** 时可以获得合理的输出。
- **寻求社区关于 Function Calling 的见解**：该成员询问社区中是否有人成功在 OpenRouter 上使用过 **Function Calling**，并对空响应表示担忧。
  
  - 他们强调，在不使用 tools 的情况下尝试非常顺利，这引发了关于此配置具体问题的疑问。

 

---

{% else %}

> 完整的逐频道详情已针对电子邮件进行截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}