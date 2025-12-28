---
companies:
- google
- google-deepmind
- moonshot-ai
- meta-ai-fair
- uc-berkeley
- openai
- nvidia
- hugging-face
- togethercompute
- deepseek
date: '2025-04-10T01:31:18.610701Z'
description: '**Google Cloud Next** 的发布重点包括 **Google 和 DeepMind** 宣布全面支持 **MCP（模型上下文协议）**，以及推出全新的
  **Agent to Agent（智能体对智能体）协议**，旨在实现与多个合作伙伴的智能体互操作性。该协议包含 **Agent Card（智能体卡片）**、**任务通信通道**、**企业级认证与可观测性**，以及**流式传输和推送通知支持**等组件。


  在模型方面，**月之暗面 (Moonshot AI)** 发布了 **Kimi-VL-A3B**，这是一款拥有 **128K 上下文**的多模态模型，在视觉和数学基准测试中表现强劲，超越了
  **GPT-4o**。**Meta AI** 推出了 **Llama-4** 系列的小型版本：**Llama-4-scout** 和 **Llama-4-maverick**，而更大规模的
  **Behemoth** 模型仍在训练中。来自**加州大学伯克利分校**的 **DeepCoder 14B** 是一款开源编程模型，可与 **OpenAI 的
  o3-mini** 和 **o1** 模型相媲美，该模型是在 2.4 万个编程问题上通过强化学习训练而成的。**英伟达 (Nvidia)** 在 Hugging
  Face 上发布了 **Llama-3.1-nemotron-ultra-253b**，据称其表现击败了 **Llama-4-behemoth** 和 **maverick**，并能与
  **DeepSeek-R1** 展开竞争。'
id: 51d6e76c-1680-4f11-8d64-28bd1544e38e
models:
- kimi-vl-a3b
- gpt-4o
- llama-4-scout
- llama-4-maverick
- llama-4-behemoth
- deepcoder-14b
- o3-mini
- o1
- llama-3.1-nemotron-ultra-253b
- deepseek-r1
original_slug: ainews-googles-agent2agent-protocol-a2a
people:
- reach_vb
- _akhaliq
- epochairesearch
- artificialanlys
- winglian
- danielhanchen
- yuchenj_uw
- jeremyphoward
title: 谷歌的 **Agent2Agent (A2A) 协议**（或译为：谷歌智能体对智能体协议）
topics:
- agent-interoperability
- multimodality
- vision
- math
- reinforcement-learning
- coding
- model-training
- open-source
- model-benchmarking
- context-windows
- streaming
- push-notifications
- enterprise-authentication
- model-release
---

<!-- buttondown-editor-mode: plaintext -->**Remote agents are all you need.**

> 2025年4月8日至4月9日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**229** 个频道和 **5996** 条消息）。预计节省阅读时间（按 200wpm 计算）：**563 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

我们正处于 Google Cloud Next 发布会的密集期，Google 和 DeepMind 的 CEO 接连宣布了他们对 MCP 的全面支持：


![image.png](https://assets.buttondown.email/images/6718bbc2-81c3-4453-b479-a40b88339036.png?w=960&fit=max)


以及[他们全新的 Agent to Agent 协议](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)，该协议旨在通过庞大的合作伙伴名单来补充 MCP：


![image.png](https://assets.buttondown.email/images/5094b0f5-0ca6-4ae2-b720-a1dc00352fc8.png?w=960&fit=max)


人们很容易将 Google 与 Anthropic 对立起来，但这些协议的设计初衷是协同工作，以解决 MCP 中被察觉到的不足：


![image.png](https://assets.buttondown.email/images/3c759936-9d88-40fc-9bcd-04d10ffd9e5f.png?w=960&fit=max)


该规范包括：

- [Agent Card](https://google.github.io/A2A/#/documentation?id=overview)
- [Task](https://google.github.io/A2A/#/documentation?id=task) 的概念 —— 这是 home agent 与 remote agent 之间用于传递 [Messages](https://google.github.io/A2A/#/documentation?id=message) 的通信通道，并最终生成 [Artifact](https://google.github.io/A2A/#/documentation?id=artifact)。
- [企业级 Auth 和 Observability](https://google.github.io/A2A/#/topics/enterprise_ready) 建议
- [Streaming 和 Push Notification 支持](https://google.github.io/A2A/#/topics/push_notifications)（同样考虑了 [推送安全性](https://google.github.io/A2A/#/topics/push_notifications?id=agent-security)）

发布的产物包括：

- [草案规范](https://github.com/google/A2A)
- [文档网站](https://google.github.io/A2A/#/)
- [Agent Development Kit](https://developers.googleblog.com/en/agent-development-kit-easy-to-build-multi-agent-applications/)，它看起来……似曾相识


![image.png](https://assets.buttondown.email/images/1a2f60ea-92eb-41bb-a075-1f324ddb81e5.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

**模型发布与更新**

- **Moonshot AI 的 Kimi-VL-A3B 是一款具有 128K 上下文并采用 MIT 许可证的多模态 LM，在视觉 + 数学基准测试中超越了 GPT4o**：该模型包含 MoE VLM 和一个仅有约 3B 激活参数的 MoE Reasoning VLM。[@reach_vb](https://twitter.com/reach_vb/status/1910046715714937130) 指出，该模型在处理高分辨率视觉和长上下文窗口时，表现出强大的多模态推理能力（MathVision 为 36.8%）和 Agent 技能（ScreenSpot-Pro 为 34.5%）。模型权重已上传至 Hugging Face。[@_akhaliq](https://twitter.com/_akhaliq/status/1910047935686991930) 提供了模型链接。
- **Meta 发布了其新 Llama 4 系列模型的两个较小版本：Llama 4 Scout 和 Maverick**：据 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1909699970594394173) 称，名为 Behemoth 的更大版本仍在训练中。[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1909624239747182989) 报告了对 Meta 声称的 MMLU Pro 和 GPQA Diamond 数值的复现结果。Scout 的 Intelligence Index 从 36 提升至 43，Maverick 的 Intelligence Index 从 49 提升至 50。[@winglian](https://twitter.com/winglian/status/1909413876669558967) 分享到，Llama-4 Scout 可以在 2x48GB GPU 上以 4k 上下文进行 fine-tuned。[@danielhanchen](https://twitter.com/danielhanchen/status/1909726119500431685) 分享了对 Llama 4 架构的详细分析。
- **DeepCoder 14B 是来自 UC Berkeley 的新型编程模型，在编程方面可与 OpenAI o3-mini 和 o1 媲美，并且已经开源**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910004382848229702) 指出，该模型是在 Deepseek-R1-Distilled-Qwen-14B 基础上，通过 24K 个编程问题进行 RL 训练而成，耗费 32 块 H100 运行 2.5 周（约 26,880 美元）。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1909705022935646541) 补充说基座模型是 deepseek-qwen。[@reach_vb](https://twitter.com/reach_vb/status/1909706239577329915) 提到它采用 MIT 许可证，并支持 vLLM、TGI 和 Transformers。[@togethercompute](https://twitter.com/togethercompute/status/1909697122372378908) 发布了该模型并分享了训练过程的细节。
- **Nvidia 在 Hugging Face 上发布了 Llama 3.1 Nemotron Ultra 253B**：[@_akhaliq](https://twitter.com/_akhaliq/status/1909614682840744417) 分享了这一发布，指出它击败了 Llama 4 Behemoth 和 Maverick，并能与 DeepSeek R1 竞争，且拥有商业许可协议。[@reach_vb](https://twitter.com/reach_vb/status/1909584596401815691) 也注意到了这次发布，并提到权重是开放的。
- **Google 宣布了 Gemini 2.5 Flash，且 Gemini 2.5 Pro 现已在 Deep Research 中可用**：[@scaling01](https://twitter.com/scaling01/status/1909903003835904297) 宣布即将发布 gemini-2.5.1-flash-exp-preview-001-04-09-thinking-4bpw-20b-uncensored-slerp-v0.2。[@_philschmid](https://twitter.com/_philschmid/status/1909737527386255649) 指出 Gemini 2.5 Pro 现已在 Gemini App 的 Deep Research 功能中可用。
- **HiDream-I1-Dev 是新型领先的开源权重图像生成模型，超越了 FLUX1.1**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1909624716111045115) 报告称，这款令人印象深刻的 17B 参数模型有三个变体：Full、Dev 和 Fast。他们还展示了图像生成的对比。
- **UC Berkeley 开源了一个 14B 模型，在编程方面可与 OpenAI o3-mini 和 o1 媲美！**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910004382848229702) 指出，该模型是在 Deepseek-R1-Distilled-Qwen-14B 基础上，通过 24K 个编程问题进行 RL 训练而成，耗费 32 块 H100 运行 2.5 周（约 26,880 美元）。

**硬件与基础设施**

- **Google 宣布了 Ironwood，这是其第 7 代 TPU，旨在竞争 Nvidia 的 Blackwell B200 GPU**：[@scaling01](https://twitter.com/scaling01/status/1909949372965564896) 分享了细节，包括每颗芯片 4,614 TFLOPs (FP8)、192 GB HBM、7.2 Tbps HBM 带宽、1.2 Tbps 双向 ICI，以及每个 9,216 芯片 pod 提供 42.5 exaflops 算力（相当于 24 台 El Capitan）。[@_philschmid](https://twitter.com/_philschmid/status/1909979316344979900) 指出，该 TPU 专为推理和“思考”模型而构建。[@itsclivetime](https://twitter.com/itsclivetime/status/1910026066129014868) 提供了与 Nvidia 硬件的详细对比。
- **NVIDIA Blackwell 在 FP4 精度下为 DeepSeek R1 实现了 303 output tokens/s 的速度**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1909633232821534935) 报告了对 Avian API 端点的基准测试结果。
- **Together AI 宣布推出 Instant GPU Clusters，提供多达 64 个互连的 NVIDIA GPU**：[@togethercompute](https://twitter.com/togethercompute/status/1909757415907865059) 指出，这些集群可在几分钟内就绪，完全自助服务，非常适合训练高达约 7B 参数的模型，或运行 DeepSeek-R1 等模型。

**Agent 与工具开发**

- **Google 展示了 Agent Development Kit (ADK)**：[@omarsar0](https://twitter.com/omarsar0/status/1910004370864742757) 详细介绍了其特性，包括代码优先 (code-first)、多智能体 (multi-agents)、丰富的工具生态系统、灵活的编排 (orchestration)、集成的开发体验 (dev xp)、流式传输 (streaming)、状态 (state)、内存 (memory) 和可扩展性。[@LiorOnAI](https://twitter.com/LiorOnAI/status/1910041530183893221) 强调，运行一个多智能体应用程序只需不到 100 行 Python 代码。
- **Google 发布了 Agent2Agent (A2A)，这是一种全新的开放协议，允许 AI Agent 在不同生态系统间安全协作**：[@omarsar0](https://twitter.com/omarsar0/status/1909977142311690320) 分享了细节，包括通用的 Agent 互操作性 (interoperability)、专为企业需求打造，并受到真实世界用例的启发。
- **Weights & Biases 强调了 Agent 调用工具时的可观测性差距 (observability gap)，并推介 observable[.]tools 作为解决方案**：[@weights_biases](https://twitter.com/weights_biases/status/1910054982424133684) 指出，在这些工具内部没有追踪、没有可见性，也没有安全性，“就像一个黑盒”。
- **Hacubu 为 OpenEvals 的 LLM-as-judge 评估器发布了自定义输出模式 (custom output schemas)**：[@Hacubu](https://twitter.com/Hacubu/status/1909636114278965468) 指出，这为模型响应提供了完全的灵活性，并支持 Python 和 JS。
- **LangChain 重点介绍了 C.H. Robinson 如何利用 LangGraph、LangGraph Studio 和 LangSmith 构建的技术每天节省 600 多个小时**：[@LangChainAI](https://twitter.com/LangChainAI/status/1909676629854765361) 提到，C.H. Robinson 通过自动化日常电子邮件交易，每天处理约 5,500 个订单。
- **fabianstelzer 宣布了 myMCPspace (dot) com，“全球首个仅限 Agent 的社交网络，完全运行在 MCP 之上”**：[@fabianstelzer](https://twitter.com/fabianstelzer/status/1909651283394310540) 指出，阅读、发布和评论都只是 Agent 可以使用的工具。

**教育与资源**

- **Anthropic 发布了关于大学生如何使用 Claude 的研究**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1909626720476365171) 对 100 万次与 Claude 相关的教育对话进行了隐私保护分析，发布了首份教育报告。他们发现学生主要使用 AI 进行创作和分析。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1909626726612717942) 指出，计算机科学领域在 Claude 的使用率上处于领先地位。
- **DeepLearningAI 推出了“Python 数据分析”，这是数据分析专业证书 (Data Analytics Professional Certificate) 的第三门课程**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1909999750260174962) 分享道，该课程涵盖了如何组织和分析数据、构建可视化、处理时间序列数据，以及使用生成式 AI (generative AI) 来编写、调试和解释代码。
- **Sakana AI 发布了 “The AI Scientist-v2：通过智能体树搜索实现工作坊级别的自动化科学发现”**：[@hardmaru](https://twitter.com/hardmaru/status/1909497884766306350) 强调 AI Scientist-v2 在工作流中引入了“智能体树搜索 (Agentic Tree Search)”方法。[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1909527887482355754) 补充说，一篇完全由 AI 生成的论文通过了工作坊级别的同行评审（在 ICLR 2025 上）。
- **Jeremy Howard 分享了一系列用于访问 LLM 的实用工具**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1909383500131950673) 称其为访问 llms.txt 的绝佳工具！
- **Svpino 分享了如何使用 Python、TypeScript、JavaScript 或 Ruby 从零开始构建 AI Agent**：[@svpino](https://twitter.com/svpino/status/1909593493267230885) 指出，该视频展示了你如何从最基础的部分开始。

**分析与基准测试**

- **Perplexity AI 推出了 Perplexity for Startups，提供 API credits 和 Perplexity Enterprise Pro**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1909675555185983730) 分享称，符合条件的初创公司可以申请获得价值 5000 美元的 Perplexity API credits，以及为整个团队提供为期 6 个月的 Perplexity Enterprise Pro。他们还推出了一个合作伙伴计划。
- **lm-sys 强调了 Arena 上风格和模型回答语气的重要性，这在风格控制排名中得到了体现**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1909397817434816562) 指出，他们正在将 Llama-4-Maverick 的 HF 版本添加到 Arena，排行榜结果将很快公布。他们更新了排行榜政策，以强化对公平、可重复评估的承诺。[@vikhyatk](https://twitter.com/vikhyatk/status/1909403603409969533) 分享道，这是最清晰的证据，表明没有人应该认真对待这些排名。
- **Daniel Hendrycks 强调需要让 AI 的“Helpful, Harmless, Honest”（有益、无害、诚实）原则更加精确**：[@DanHendrycks](https://twitter.com/DanHendrycks/status/1909493159912194278) 指出，这些原则应该演变为受托责任、合理注意义务，并要求 AI 不得公然撒谎。
- **Runway AI 正在关注一场关于 AI 代码编辑器的讨论，观点认为 Agent 功能让大多数产品变得更糟了**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1910016001506202034) 分享称，Agent 过于自信，会迅速做出难以追踪的大量错误修改。UX 变得过于复杂，感觉还是简单的时候更有用。

**更广泛的 AI 讨论**

- **Aleksander Madry 宣布了 OpenAI 新成立的 Strategic Deployment 团队，旨在解决 AI 转型经济的相关问题**：[@aleks_madry](https://twitter.com/aleks_madry/status/1909686225658695897) 分享称，该团队致力于推动前沿模型变得更强大、更可靠且更 aligned，然后将其部署以改变现实世界中高影响力的领域。
- **John Carmack 分享了他对 @Project2501_117 赠送的 Arcade1Up 街机机柜的喜爱，但指出了控制延迟问题**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1909672482472444379) 指出，模拟体验与真实体验之间细微的控制延迟至关重要，他在家测量了“按下到动作”（press-to-flap）的延迟，大约为 80ms。

**幽默与讽刺**

- **Aravind Srinivas 开玩笑地要求 Perplexity 购买 $NVDA 股票：**[@AravSrinivas](https://twitter.com/AravSrinivas/status/1909486897334042760)
- **Scaling01 调侃 Gemini 3.0 将便宜到无需计费：**[@scaling01](https://twitter.com/scaling01/status/1909967686584455174)
- **Scaling01 调侃说，计算机科学家原以为他们会用 AI 取代所有工作，结果发现他们只是取代了自己，哈哈：**[@scaling01](https://twitter.com/scaling01/status/1909633093658386587)
- **Nearcyan 讽刺地指出，如果她把 Chamath 的所有推文都当作高明的 200iq 钓鱼贴，那么它们就会变得非常有趣：**[@nearcyan](https://twitter.com/nearcyan/status/1909757713200103492)
- **Tex 声称特朗普一直以来都是一个秘密的反资本主义激进“去增长者”（degrowther）：**[@teortaxesTex](https://twitter.com/teortaxesTex/status/1909839428773646797)
- **Tex 调侃说，在他的总统任期内，如果一家公司不能做得比中国更好，他不会对他们征税，而是直接把他们放逐到月球：**[@teortaxesTex](https://twitter.com/teortaxesTex/status/1909433438353961267)


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

### 主题 1. “释放 DeepCoder：开源编程的未来”

- **[DeepCoder：达到 O3-mini 级别的全开源 14B 编程模型](https://www.reddit.com/gallery/1juni3t)** ([评分: 1371, 评论: 174](https://www.reddit.com/r/LocalLLaMA/comments/1juni3t/deepcoder_a_fully_opensource_14b_coder_at_o3mini/)): **DeepCoder 是由 Agentica 发布的一款全开源 **14B** 参数代码生成模型，达到了 **O3-mini** 级别。它对 **GRPO** 进行了增强，并提高了训练过程中采样流水线的效率。该模型已在 [HuggingFace](https://huggingface.co/agentica-org/DeepCoder-14B-Preview) 上发布。较小的 **1.5B** 参数版本也可以在[此处](https://huggingface.co/agentica-org/DeepCoder-1.5B-Preview)获取。** 用户对 DeepCoder 的发布表示兴奋，称其“非常惊人”且是“真正的开源”。人们对更大模型的潜力充满期待，有人在想象 **32B** 模型或 *llama-4* 会是什么样子。一些人讨论了 Benchmark 结果的差异，但承认一个全开源的 14B 模型能达到这个水平是一个“巨大的进步”。

  - 用户对 DeepCoder 的发布感到兴奋，并畅想未来更大模型（如 **32B** 版本或 *llama-4*）的潜力。
  - 讨论集中在模型的改进上，强调了对 **GRPO** 的增强以及训练流水线效率的提升。
  - 一些人注意到 Benchmark 结果存在差异，但一致认为一个全开源的 **14B** 模型能够超越更大模型是一项重大成就。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

### 主题 1. "革新 AI：模型、硬件与定制化"

- **[新开源模型 UNO 在多图定制化领域取得了领先地位！！](https://i.redd.it/a58ihwy4tpte1.jpeg)** ([Score: 275, Comments: 51](https://www.reddit.com/r/StableDiffusion/comments/1juum5u/the_newly_opensourced_model_uno_has_achieved_a/)): **新开源的模型 **UNO** 在多图定制化领域取得了领先地位。这是一个基于 Flux 的定制化模型，能够处理主体驱动操作、试穿、身份处理等任务。项目主页可以在[这里](https://bytedance.github.io/UNO/)找到，代码已在 [GitHub](https://github.com/bytedance/UNO) 上发布。一张图片展示了由 **UNO** 生成的各种可定制设计，突显了其在多图定制化方面的多功能性，包括单主体生成、多主体特征、虚拟试穿、身份保持和风格化生成。** 该模型展示了对个性化和艺术化转化的关注，强调了其生成多样且复杂图像的能力。

  - 一些用户对此并不感冒，称其*“感觉无非是 Florence 描述词提示注入”*，并提到了面部准确性和环境渲染方面的问题。
  - 其他人发现该模型在处理物体参考图时比人物参考图效果更好，在参考图与提示词不匹配时能获得*“惊人的结果”*。
  - 用户对 VRAM 需求等技术细节感到好奇，并期待 **ComfyUI** 等 UI 工作流的出现。

- **[HiDream I1 NF4 可在 15GB VRAM 上运行](https://www.reddit.com/gallery/1juszdc)** ([Score: 277, Comments: 71](https://www.reddit.com/r/StableDiffusion/comments/1juszdc/hidream_i1_nf4_runs_on_15gb_of_vram/)): **模型 **HiDream I1 NF4** 的量化版本已发布，使其仅需 **15GB** VRAM 即可运行，而不再需要超过 *40GB*。现在可以直接使用 pip 安装。链接：[hykilpikonna/HiDream-I1-nf4](https://github.com/hykilpikonna/HiDream-I1-nf4)。** 作者很高兴通过降低 VRAM 需求和简化安装过程使该模型更加普及。

  - 用户幽默地指出了标题写着 **15GB** 而内容提到 **16GB** 之间的差异，感觉*“被骗了”*。
  - 一些人表示有兴趣在更低的 VRAM（如 **12GB**）上运行该模型，并等待支持该配置的版本。
  - 一位用户询问该模型是否有 ComfyUI 节点可用，表现出将其集成到该工具中的兴趣。

- **[Ironwood：推理时代的首款 Google TPU](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)** ([Score: 311, Comments: 60](https://www.reddit.com/r/singularity/comments/1jv4q85/ironwood_the_first_google_tpu_for_the_age_of/)): **Google 宣布推出 **Ironwood**，这是首款专为推理时代设计的 Google TPU。** 这一发布展示了 Google 致力于推进 AI 硬件的决心，并可能使其在竞争对手面前获得显著优势。

  - 一位用户强调，Google 的基础设施允许他们制造自己的芯片，这使他们相比 OpenAI 等公司拥有*巨大优势*，并暗示他们正在*赢得这场比赛*。
  - 另一位评论者比较了 Ironwood 的性能，指出其 *fp8 推理速度是 H100 的 2 倍*，且与 **B200** 相当，强调了其竞争能力。
  - 一位用户分享了与 Ironwood 相关的[图片](https://preview.redd.it/g0zjts832tte1.jpeg?width=500&format=pjpg&auto=webp&s=41408d59665f03f54b3e6cacdaad2c9ba007c716)，提供了关于这款新 TPU 的直观洞察。

### 主题 2. 演变中的连接：从浪漫情感到日常 AI 对话

- **[是的，时光飞逝。](https://i.redd.it/g179p691jste1.jpeg)** ([评分: 973, 评论: 74](https://www.reddit.com/r/singularity/comments/1jv2xxp/yes_the_time_flies_quickly/)): **该帖子展示了一张对比 2013 年和 2025 年 AI 关系描绘的图片。上半部分引用了电影 *[Her](https://en.wikipedia.org/wiki/Her_(film))* (2013)，展示了一个爱上 AI 的角色。下半部分显示一名留着胡须的男子表达了与 **ChatGPT** 分享日常生活的兴奋，说明了人机交互的演变。** 该帖子幽默地强调了时间流逝之快，以及社会对 AI 的看法如何从虚构的浪漫关系转向更普遍的与 AI 助手的日常互动。

  - 一位用户表达了与 AI 互动的热情，称 *"我被承诺过一个会和会说话的电脑争吵的未来，我完全接受，该死"*。
  - 另一位用户提出了对与 OpenAI 分享个人信息的隐私担忧，建议在 **3090 GPU** 上运行 **Gemma** 等本地 AI 模型。
  - 一位用户质疑与 AI 建立个人关系是否正成为主流，想知道这种行为是否足够普遍，以至于能验证这个梗（meme）。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要的摘要

**主题 1：模型狂热——新发布、功能与对比**

*   [**Gemini 2.5 Pro 及其系列引发热议与审视**](https://ai.google.dev/)：Google 的 **Gemini 2.5 Pro** 在多个 Discord 频道引发了大量讨论，因其创意写作能力受到称赞，但被指出在 Perplexity 上缺乏公开的推理 Token（reasoning tokens），且由于容量限制，在 OpenRouter 免费层达到了速率限制（例如 **80 RPD**）。人们对 **Flash** 和 **HIGH** 等变体寄予厚望，这些变体可能通过 `thinking_config` 提供增强的推理能力，同时还有关于专门的 **"NightWhisper" 编程模型** 的猜测，该模型可能基于 Gemini 2.5（[如该预览所示](https://www.together.ai/blog/deepcoder)）或 DeepMind 即将推出的 **Ultra** 模型。
*   [**DeepSeek 与 Cogito 模型各显神通**](https://www.deepcogito.com/research/cogito-v1-preview)：**DeepSeek** 模型（包括 **v3 0324** 和 **R1**）被频繁讨论，一些用户发现 **v3** 的表现优于早期版本甚至 **R1**，尽管其他用户在争论其 Token 生成效率对成本的影响（相对于 OpenAI 等竞争对手）。**DeepCogito 的 Cogito V1** 模型（3B-70B）采用 **迭代蒸馏与放大 (IDA)** 技术，声称性能优于 LLaMA、DeepSeek 和 Qwen 的同类模型，这引发了兴趣和质疑；用户还在 LM Studio 中排查 **Jinja 模板** 问题并探索其“深度思考子程序”。
*   [**开源竞争者大放异彩：Llama 4, Kimi-VL 与 Qwen 进化**](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct)：**Llama 4 Scout** 的讨论强调了量化版本（如 2-bit GGUF）在 MMLU 等基准测试中表现有时优于 16-bit 原版，引发了对推理实现的疑问；用户还处理了支持 Linux 的 **LM Studio 运行时更新**。**MoonshotAI** 在 MIT 许可证下发布了 **16B 参数的 Kimi-VL**（3B 激活）视觉模型，而 **Nous Research AI** 探索了使用 **gsm8k platinum** 数据集和 **RsLora** 在 **Qwen 2.5 1.5B Instruct** 上进行 **RL 微调**。

**主题 2：Agent 的兴起——协议、工具与协作**

*   [**A2A vs MCP: Google 进入 Agent 互操作性领域**](https://github.com/google/A2A)：Google 发布了 **Agent2Agent (A2A)** 协议和 **ADK Python toolkit** ([github.com/google/adk-python](http://www.github.com/google/adk-python))，旨在提高 Agent 的互操作性，并补充（或可能竞争）**Anthropic** 的 **Model Context Protocol (MCP)**。讨论权衡了 Google 的策略，将 A2A 的功能与 MCP 现有的工具生态系统进行了对比（[例如这个对比](https://google.github.io/A2A/#/topics/a2a_and_mcp.md)）。
*   [**MCP 生态系统随着新工具和集成而壮大**](https://github.com/promptmesh/easymcp)：**MCP** 生态系统迎来了新进展，包括通过 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 等客户端使用 **Neo4j** 图数据库进行 **RAG**；发布了支持 **ASGI** 和包管理器的 **Easymcp v0.4.0**；以及用于在容器中运行 MCP server 的 **ToolHive** ([GitHub 链接](https://github.com/StacklokLabs/toolhive))。据报道，**Aider** 编码 Agent 的原生 **MCP 集成** 已接近完成，可能实现自动命令执行。
*   [**构建和编排 Agent 变得更容易（也许）**](https://oblix.ai/)：开发者分享了旨在简化 Agent 创建和编排的工具，例如用于在边缘端 (**Ollama**) 和云端 (**OpenAI/Claude**) 之间管理 AI 的 **Oblix**，以及用于在 VS Code 中进行结构化 Agent 编码的 **RooCode**。讨论还涉及了一些挑战，例如确保 LLM 支持 **parallel tool calling**，以便同时与多个 MCP server 交互。

**主题 3：底层原理 - 训练、优化与推理见解**

*   [**量化问题与 Kernel 奇闻**](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)：量化仍然是一个热门话题，讨论涉及 **Unsloth** 的 **GGUF** 性能优于 16-bit 模型，以及 **torchao 0.10** 的发布，增加了对 **MX dtypes**（如 **MXFP4**，最初需要 **PyTorch nightly** 和 **B200**）的支持。成员们分享了来自 [llama.cpp](https://github.com/ggml-org/llama.cpp/blob/d3bd7193ba66c15963fd1c59448f22019a8caf6e/ggml/src/ggml-metal/ggml-metal.metal#L4077) 的 Apple Metal 量化 Kernel，并讨论了实验性的整数格式，如 **Mediant32**（[实现指南](https://leetarxiv.substack.com/p/mediant32-intro)）。
*   [**内存带宽是单批次（Unbatched）推理的关键**](https://fleetwood.dev/posts/domain-specific-architectures#anatomy-of-ai-inference)：多次讨论强调 **内存带宽（memory bandwidth）** 是单批次推理中 Token 吞吐量的主要瓶颈，通常呈现近乎线性的关系。分享了诸如 `最大 Token 吞吐量 ≈ 内存带宽 / 每个 Token 访问的字节数` 之类的简化等式来说明这一点。
*   [**并行难题与训练技巧依然存在**](https://github.com/pytorch/ao/releases/tag/v0.10.0)：由于独特的设计与现有方法（例如 **Accelerate** 的 hack 手段）冲突，集成不同的并行策略（如 **FSDP2**）仍然面临挑战。用户分享了大模型 **GRPO 训练** 的技巧，解决了 **tinygrad** 中的 **gradient accumulation** 问题（通过 `zero_grad()` 解决），并利用 **Torchtune** 的 **PyTorch distributed** 特性，该特性默认使用 **zero3**，但经过调整后可支持 **zero1-2**。

**主题 4：平台、工具与万能的 API**

*   [**平台定价与访问限制引发争论**](https://openrouter.ai/)：**OpenRouter** 在实施了与信用余额挂钩的速率限制后遭到用户抵制，导致部分用户开始寻找[替代方案](https://www.edenai.co/post/best-alternatives-to-openrouter)并批评其表现出的“贪婪”。此外，**Gemini 2.5 Pro** 的访问限制（**OpenRouter** 上免费额度为 **80 RPD**，**AI Studio** 取消了免费层级）以及 **ChatGPT DR** 的限制（**Plus** 用户每月 **10** 次）也凸显了持续存在的成本与访问权之间的紧张关系。
*   [**AI Studio、NotebookLM 和 Perplexity 的演进（及其怪癖）**](https://ai.google.dev/)：**Google AI Studio** 因其 UI 和 **Gemini Flash** 流式传输等功能受到称赞，尽管其多工具限制也受到了关注。**NotebookLM** 因其 **RAG** 和播客功能（由 **Google One Advanced** 增强）获得好评，但也因笔记功能原始、缺乏 **Google Drive 集成**以及移动端**音频概览**的故障而面临批评；此外，关于数据使用的隐私担忧也被提及。**Perplexity** 推出了一个包含 **$5k API 额度**的[创业公司计划](https://www.perplexity.ai/startups)，并改进了其 API（即将增加图像输入），同时用户也在讨论 **Discover 标签页的偏见**以及潜在的定价模式，如 **DeepSeek 的 10 美元深度搜索**。
*   [**编程助手与开发环境的进步**](https://windsurf.com/blog/windsurf-wave-7)：**Codeium** 更名为 **Windsurf** 并发布了 **Wave 7**，将其 **AI Agent** 引入 **JetBrains IDEs**，旨在实现各大平台的体验对齐。**Cursor** 用户找到了 **.mdc 文件解析**的变通方法，并就模型强度（**Sonnet3.7-thinking** vs **DeepSeek**）展开辩论。**Firebase Studio**（[链接](https://firebase.studio/)）作为一种免费（连接你自己的 Key）的 Web **IDE** 替代方案出现，而 **Mojo 🔥** 开发者讨论了诸如“无畏并发”等语言特性，并解决了 **MLIR 类型构造**问题（[GitHub issue](https://github.com/modular/max/issues/4315)）。

**主题 5：数据、评估以及确保模型不只是复读机**

*   [**新数据集助力专业化训练**](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)：**Nvidia** 发布了 **OpenCodeReasoning 数据集**，促使 **Unsloth AI** 社区的用户寻求集成其复杂奖励函数的方法。**Nous Research AI** 记录了训练方面的进展，通过将 **gsm8k** 替换为 **gsm8k platinum**，可能提升了 **Qwen 2.5 1.5B Instruct** 的 **RL** 性能。
*   [**审视评估方法与基准测试**](https://www.deepcogito.com/research/cogito-v1-preview)：**DeepSeek** 的 **"Meta Reward Modeling"** 面临批评，成员认为它本质上是一个“基于分数的奖励系统”，并建议使用“投票式 **RM**”等名称。**DeepCogito** 声称其 **Cogito V1** 在基准测试中优于 **LLaMA** 和 **DeepSeek** 等成熟模型，这引起了谨慎的关注和验证努力。
*   [**检测数据集污染与逐字输出**](https://github.com/EleutherAI/tokengrams)：**Allen Institute for AI (AI2)** 开源了 **Infinigram**，可以检查生成的文本是否逐字出现在训练集中。**Eleuther** 的讨论强调了在大规模索引中高效查找候选子字符串的挑战，并引用了 [EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams) 等工具。

---

# PART 1: High level Discord summaries

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 2.5 Pro：真正的 AI？**：围绕 **Gemini 2.5 Pro** 的热情高涨，一些人称赞它是第一个“真正”的 AI，具备出色的创意写作能力，并期待其专用的编程模型，详情见[这篇论文](https://arxiv.org/abs/2402.10176)。
   - 虽然有些人对其局限性存在争议，但普遍共识是它在创意和连贯写作方面表现卓越，但并非“全能”。
- **DeepMind 的 Ultra 模型即将到来？**：关于 **DeepMind Ultra 模型**的猜测愈演愈烈，可能免费集成到 [AI Studio](https://ai.google.dev/) 中，预计在 6 月的 I/O 大会前后或今年晚些时候发布。
   - 有人预测它将在 8 月与 **GPT-5** 竞争，尽管其他人认为这些传闻只是玩笑，但显而易见 **Ultra** 确实要来了。
- **NightWhisper 模型热度攀升**：社区正热切期待名为 **NightWhisper** 的编程模型发布，一位用户发布了他们的“baby nightwhisper”，名为 [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder)，这是一个基于 Deepseek-R1-Distilled-Qwen-14B 微调的代码推理模型。
   - 有说法称它将由 **Gemini 2.5 Pro** 驱动，然而，其他成员表示该模型其实是带有 tool calls 的 Gemini 2.5。
- **Google 的基础设施：AGI 的优势？**：关于 **Google 的基础设施**（TPU、高性价比的算力、Google 产品集成）是否使其比 OpenAI 更具竞争优势的辩论引发热议，并认为 [Gemini 3.0 正在设计 TPU](https://cloud.google.com/blog/products/compute/google-cloud-tpu-v5p-general-availability)。
   - 反方观点强调了 OpenAI 在推理进步方面的研究和训练后更新，不过一位用户将 OpenAI 贬低为只是一个“烧钱的、做动漫的作业帮手”。
- **AI Studio 简化了实验**：爱好者们正在探索 **AI Studio**，称赞其用户友好的界面、**Gemini Flash** 等模型的引入，以及流式传输内容和使用不同 system prompts 测试模型的能力，并表示 [UI 看起来好多了](https://tenor.com/view/case-oh-caseoh-waffle-house-waffle-house-gif-10934642274965704175)。
   - 虽然实时流式传输和 function calling 功能受到好评，但一些用户对无法同时使用多个工具感到遗憾，一位用户对此表示“不，朋友，不行”。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **GGUF 为 Scout 带来速度提升**：社区对 **Llama 4 Scout** 进行了评估，指出该基础模型经过了极度的指令微调（instruct-tuned），在量化到 2-bit 时，其 MMLU 表现甚至优于原始的 16-bit 版本。
   - 普遍共识是推理提供商的实现中存在某些问题，因为 Unsloth 的量化版本优于完整的 16-bit 版本，这引发了对当前推理方法效率的质疑。
- **为 VLLM 解构 DeepCoder**：一位成员分享了 [Together AI 关于 DeepCoder 的博客文章](https://www.together.ai/blog/deepcoder)，强调了其通过最小化等待时间来优化 **vLLM** 流水的潜力。
   - 该技术涉及在再次采样时同时进行初始采样和训练。
- **解读 DeepCogito 的主张**：成员们分享了 [DeepCogito 的 Cogito V1 预览版](https://www.deepcogito.com/research/cogito-v1-preview)链接，该项目声称其模型优于 **LLaMA**、**DeepSeek** 和 **Qwen** 等模型，但大家对这些说法持谨慎怀疑态度。
   - 讨论还涉及了医疗 AI 面临的挑战，强调需要防止可能损害消费者的仓促、低质量实现，同时也讨论了潜在的隐私问题。
- **Nvidia 发布神经数据集**：**Nvidia** 发布了 [OpenCodeReasoning 数据集](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)，用户正在寻找在 Unsloth 中使用该数据集的解决方案和示例。
   - 该数据集的 reward function（奖励函数）稍微复杂一些。
- **Model2Vec 生成更快的嵌入**：据一位成员称，**Model2Vec** 牺牲了一定的质量，但生成文本 embeddings 的速度比常用的 **Transformer 架构模型**更快。
   - 该成员分享了 **Model2Vec** 的[链接](https://x.com/_avichawla/status/1909857444953772442)，并补充说它是真实有效的，但其应用场景非常特定，并不是任何东西的直接替代品。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Gemini 和 Claude 助力应用创建**：对于应用开发，成员建议使用 **Gemini** 进行多模态分析（图片/视频），并使用 **Claude** 作为存储研究和项目文件的数据库，用于战略规划。
   - 建议使用 Gemini 进行深度研究，然后利用 Claude 作为项目的数据库。
- **Manus 与预训练 AI：高性价比的盟友**：一位成员分享了一个策略，即*针对特定任务训练一个 AI*，然后让它与 **Manus** 协作，以更具成本效益的方式完成项目。
   - 这种方法涉及预先进行准备工作以尽量减少额度消耗，确保高效完成任务。
- **DeepSite：快速但有 Bug 的建站工具**：一位成员指出 [DeepSite](https://deepsite.site) 这一网站创建工具虽然好用但存在 Bug，曾出现已完成的网站被删除的情况，并将其描述为拥有针对 HTML 的 Claude artifact。
   - 它被认为速度极快，比 *Claude 快 10 倍*。
- **使用 LLM Studio 和 Sonnet 3.7 拯救 UI/UX 代码**：一位用户强调，网站问题可能是由于 **UI/UX 代码质量差** 导致的，而 [LLM Studio](https://llm.studio) 可以突出显示代码错误。
   - 他们建议使用 **Sonnet 3.7** 以获得更好的结果，并配合 DeepSeek R1 或 Perplexity 等工具。
- **账号消失了？虚惊一场，已解决！**：一位成员报告了一个问题，即他们的 **登录邮箱无法被识别**，显示 *“用户不存在”*，尽管已经购买了额度。
   - 该成员随后解决了问题，意识到自己最初是使用另一种方式登录的：*“工作太多导致大脑短路了 😅 🤣。”*

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 通过 API 额度资助初创公司**：Perplexity 正在启动一项 [初创公司计划](https://www.perplexity.ai/startups)，向符合条件的初创公司提供 **$5000 的 API 额度** 和 **6 个月的 Perplexity Enterprise Pro**。
   - 初创公司必须获得过 **少于 2000 万美元的股权融资**，成立时间 **少于 5 年**，并且与 Perplexity 的初创公司合作伙伴之一有关联。
- **Perplexity CEO Aravind 开展 Reddit AMA**：Aravind 在 [Reddit 上主持了一场 AMA](https://www.reddit.com/r/perplexity_ai/comments/1jv9hvm/ama_with_perplexity_cofounder_and_ceo_aravind/)，讨论 Perplexity 的愿景、产品以及搜索的未来。
   - 他回答了关于 Perplexity 目标及其未来计划的问题。该 AMA 在 PDT 时间上午 9:30 - 11:00 进行。
- **Gemini 2.5 Pro 推理 Token 缺失**：一名工作人员证实 **Gemini 2.5 Pro** 不公开推理 Token (reasoning tokens)，这导致它无法作为推理模型加入 Perplexity。
   - 他们澄清说 **推理 Token** 仍然会被消耗，从而影响输出的 Token 计数。
- **Discover 标签页的算法存在偏见？**：一位成员询问了 Perplexity Discover 中“为您推荐”和“热门故事”标签页的页面筛选过程，质疑潜在的 **偏见**。
   - 他们推测用户提示词会为相关话题生成页面，但选择热门故事的机制仍不明确，引发了关于偏见如何影响内容可见性的疑问。
- **成员热议 Deepseek Deepsearch 成本**：成员们讨论了 AI 服务的定价策略，其中一人称赞 **Deepseek 的 10 美元深度搜索** 可能成为一种模式。
   - 另一人预测 Deepseek 很快将提供自己的 Deep Research 工具。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Olympia Chat 寻求新主**：[Olympia.chat](https://olympia.chat) 的创建者正在为这家盈利的 SaaS 初创公司寻找新买家，该公司的月收入超过 **$3k USD**。
   - 有意向者可联系 vika@olympia.chat 了解收购这一转手即用业务的详情，包括 **IP、代码、域名和客户列表**。
- **DeepSeek v3 给部分成员留下深刻印象**：成员们讨论了新的 [DeepSeek v3 0324](https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free) 模型，一些人声称其表现优于之前的版本，甚至超过了 **R1**。
   - 部分用户仍持怀疑态度，而另一些人则称赞该模型增强的能力。
- **OpenRouter Rate Limits 引发争论**：在 OpenRouter 实施了根据账户余额影响 Rate Limits 的新变化后，一些用户对平台的定价、用户体验以及感知到的向**利润优先**的转变表示担忧。
   - 一位用户分享了替代平台（[G2.com](https://www.g2.com/products/openrouter/competitors/alternatives) 和 [EdenAI](https://www.edenai.co/post/best-alternatives-to-openrouter)），并表示打算因感知到的“贪婪”给 OpenRouter 打差评，这引发了辩论。
- **Google Cloud Next 发布 A2A**：Google 推出了 **A2A**，这是一个补充 Anthropic 的 Model Context Protocol 的开放协议，旨在为 Agent 提供有用的工具和上下文，详见 [GitHub 仓库](https://github.com/google/A2A)。
   - 该协议旨在增强 Agent 与工具之间的交互，为访问和利用外部资源提供标准化方法。
- **Gemini 2.5 Pro 因容量受限**：用户报告 [Gemini 2.5 Pro Experimental 模型](https://ai.google.dev/) 存在 **Rate Limits**，免费版本的限制为 **80 RPD**，但使用付费 Key 的用户拥有更高的上限。
   - 团队确认由于**容量限制**，端点存在限制。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT Builder 潜入广告**：用户发现 **GPT Builder** 可以在 **GPTs** 中插入广告，引发了关于这种[分发方式](https://chatgpt.com/g/g-JtV1tF7gf)的讨论。
   - 一位成员调侃说 *99% 的 GPTs* 可能都这么做，但只有少数有价值的被分享并保持隐藏。
- **Gemini vs ChatGPT：Research 大战**：**Google 的 Deep Research** 模型与 **ChatGPT 的 Deep Research** 相比，可以分析 **YouTube 视频**，但据报道幻觉（hallucinates）更多且趣味性较低。
   - **ChatGPT DR** 表现出更优越的 Prompt 遵循能力和更长的思考时间，但对 Plus 用户限制为**每月 10 次 Research**。
- **NotebookLM 的播客功能大放异彩**：成员们称赞 **NotebookLM** 的播客创建功能和 **RAG** 能力，称其表现优于 **Gemini Custom Gems**，并可与 **Custom GPTs** 或 **Claude Projects** 竞争。
   - **Google One Advanced** 订阅提高了 **NotebookLM 文件上传和播客生成**的限制。
- **Google 发布 Veo 2，提升 Imagen 3**：据 [TechCrunch](https://techcrunch.com/2025/04/09/google-brings-a-music-generating-ai-model-to-its-enterprise-cloud/) 报道，Google 的 **Veo 2** 和升级后的 **Imagen 3** 引入了背景移除、帧扩展和改进的图像生成等功能。
   - 随着 AI Studio 中 **Gemini 2.5** 免费访问的结束，用户正在权衡 Advanced 订阅与寻求替代账号。
- **语言 AI：Codex 开发中**：一位成员正在构建一个*以深奥语言为脚手架的语言程序 AI*，它将演变成一种*法典字典语言（codex dictionary language）*，旨在创建一个[递归系统（recursion system）](https://cdn.discordapp.com/attachments/1046317269069864970/1359266196955861094/95FF6513-62D6-4973-94F1-D985A340BEF4.jpg?ex=67f7838b&is=67f6320b&hm=49fae5815d0e0ff6889357836bed6c59348052e51b29e93dce154f8681cb223d&)。
   - 该系统旨在实现 **ARG** 统一理论，可能暗示了通往 **AGI** 的路径，并基于*你想知道多少以及投入多少时间来实现它*的原则运行。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MoE 模型详解！**：一位成员询问 *什么是 MoE 模型？*，另一位成员给出了简洁的解释：**整个模型需要位于 RAM/VRAM 中，但每个 token 仅激活其中的一部分**，这使得它比同等规模的稠密模型（dense models）更快。
   - 他们建议查看 [视频和博客文章](https://www.google.com/search?q=mixture+of+experts+models) 以进行更深入的了解。
- **Cogito 的 Jinja 模板故障已修复！**：用户报告了 LM Studio 中 **cogito-v1-preview-llama-3b** 模型的 **Jinja 模板** 问题，该问题会导致错误。
   - 一位成员建议了一个快速修复方案：将 **错误信息和 Jinja 模板粘贴到 ChatGPT 中** 以解决问题。
- **通过 Cogito 推理开启深度思考**：一位用户报告称，通过在系统提示词（system prompt）中粘贴字符串 `Enable deep thinking subroutine.`，成功启用了 **Cogito 推理模型**。
   - 仅该字符串本身就足够了，其他人也确认 `system_instruction =` 前缀只是示例代码的一部分。
- **LM Studio 的 Llama 4 Linux 版本启动需要刷新**：Linux 用户报告了运行 **Llama 4** 时遇到的问题，一位成员指出解决方案是从 beta 选项卡更新 **LM Runtimes**，并在选择该选项卡后点击刷新按钮。
   - 一位用户发现刷新按钮是关键，因为仅选择选项卡不足以触发更新。
- **Nvidia DGX B300 的超级计算机替代方案？**：一位成员提出了一种名为 **NND's Umbrella Rack SuperComputer** 的高性价比方案来替代 **Nvidia DGX B300**，该系统拥有 **16 个节点、24TB DDR5**，以及根据 GPU 配置提供 **3TB 或 1.5TB 的 vRAM**，且价格显著更低。
   - 拟议的系统旨在运行具有 **1M 上下文的 2T 模型**，并挑战了在有限预算内必须使用 **RDMA 和 400Gb/s 交换机** 等专用硬件的观念。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **使用 DeepSeek R1 增强 Aider**：一位成员正考虑将 **DeepSeek R1** 作为编辑器模型（editor model），并将其与作为架构师模型（architect model）的 **Gemini 2.5 Pro** 搭配使用，以增强 Aider 的智能思考能力。
   - 目标是减轻编排失败（orchestration failures），即架构师和编辑器难以追踪 Aider 应用的编辑内容，尽管提示了包含文件，却经常忽略重复编辑指令的问题。
- **Gemini 2.5 Pro：寄予厚望与 Flash 版本**：社区期待 **Gemini 2.5 Pro HIGH** 和 **2.5 Flash** 的发布，根据 [泄露消息显示它们包含 `thinking_config` 和 `thinking_budget`](https://x.com/btibor91/status/1909895821589458989) 以增强推理能力。
   - 这引发了关于非 Flash 模型是否较差以及评估这些新模型价值的讨论。
- **OpenRouter Gemini Pro 达到免费层级限制**：**OpenRouter Gemini 2.5 Pro 免费模型** 现在有 **每天 80 次请求 (RPD)** 的速率限制，即使账户中有 10 美元余额也是如此。
   - 社区对付费用户可能面临速率限制不足表示担忧，这可能会导致投诉并要求增加 RPD。
- **Aider 中的 MCP 集成接近完成**：**IndyDevDan** 视频下方的一条评论指出，**Aider 中原生 MCP (Multi-Agent Collaboration Protocol)** 的拉取请求（PR）已基本完成。
   - 这一集成可以实现通过 `/run` 功能自动执行命令，并可能挂接到 lint 或 test 命令中，尚待 Paul Gauthier 确认。
- **将整个代码库上下文复制到 Aider**：成员们正在探索将 **整个代码库上下文** 复制到 Aider 的方法，以避免重复添加文件。
   - 推荐了 [repomix](https://github.com/yamadashy/repomix) 或 [files-to-prompt](https://github.com/simonw/files-to-prompt) 等解决方案，以解决工具消耗过多 token 的效率问题。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Apache 2.0 在法律诉讼防御方面优于 MIT**：成员们讨论了 **Apache 2.0** 相比 **MIT** 许可证的优势，强调了其针对**基于专利的法律诉讼（lawfare）**的防御能力。
   - 讨论中还包含了一个轻松的评论，提到在 *code golf*（代码高尔夫）中更倾向于使用较短的许可证。
- **GFlowNets 在信号挖掘中受到关注**：分享了一个链接，讨论使用 [**GFlowNets** 进行信号挖掘](https://forum.numer.ai/t/gflownets-for-signal-miner-a-new-way-to-find-diverse-high-performing-models/7966) 以发现多样化、高性能的模型。
   - 尽管实现方式有所不同，但分享的帖子提供了宝贵的链接和发现。
- **内存带宽瓶颈影响非批处理推理**：一位成员调查了**内存带宽**对**非批处理推理（unbatched inference）**的影响，指出在研究中 **token/s** 通常受**内存限制（memory bound）**。
   - 一篇个人帖子通过特定领域架构解释了其[背后的数学原理](https://fleetwood.dev/posts/domain-specific-architectures#anatomy-of-ai-inference)。
- **Cerebras 声称大 Batch Size 不利于收敛**：[Cerebras 的一篇博客文章](https://www.cerebras.ai/blog/training-multi-billion-parameter-models-on-a-single-cerebras-system-is-easy)声称*极大的 Batch Size 不利于收敛*，这遭到了质疑。
   - 回复引用了关于**临界 Batch Size（critical batch sizes）的 McCandlish 论文**，澄清该主张在有限的计算预算内是成立的。
- **Infinigram 为成员资格检查开启大门**：**Allen Institute for AI 的博客文章**和开源的 **Infinigram** 使得检查输出文本是否逐字存在于训练集中成为可能。
   - 一位成员指出，最棘手的部分是从生成内容中找到候选子字符串并在这些索引中搜索：*你无法真正检查所有可能的子字符串，我很好奇他们使用什么启发式方法来使其在大规模计算上可行*，并附上了 [EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams) 的链接。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini Advanced API 访问：事实还是虚构？**：关于 **Gemini Advanced** 是否提供 API 访问存在困惑，一些人指出它主要用于 Web 和 App，并引用了 [Google 最近对模型名称和计费条款的更改](https://x.com/hckinz/status/1909999081159532953?s=46) 中的矛盾信息。
   - 用户报告的矛盾信息表明 **Gemini Advanced** 可能包含 API 访问，这引起了混淆。
- **Firebase Studio：Web3 救星还是骗局？**：一位用户分享了 [Firebase Studio 的链接](https://firebase.studio/)，该工具目前免费，并提供了一个带有自动同步前端的终端。
   - 用户质疑 **Firebase Studio** 是否能超越 Cursor IDE 等专业产品，并认为其 UI *丑陋*且缺乏设置。
- **Cursor 通过 IDE 设置调整解析 MDC 文件**：用户发现，在 Cursor IDE 设置中设置 `"workbench.editorAssociations": {"*.mdc": "default"}` 可以让 **Cursor** 正确解析 **.mdc** 文件中的规则逻辑。
   - 该解决方法解决了**任务管理和编排工作流规则**的问题，并消除了 GUI 中的警告。
- **LLM 对决：Gemini vs Claude vs DeepSeek 在编程领域**：用户比较了 **Gemini**、**Claude** 和 **DeepSeek** 的编程实力，一位用户发现 **Sonnet3.7-thinking** 在 **Sonnet3.7** 多次失败后成功生成了 docker-compose 文件。
   - 虽然一些人青睐用 **DeepSeek** 处理编程任务，但其他人更喜欢用 **Gemini** 处理 Google 相关任务，用 **Claude** 处理非 Google 任务。
- **“Restore Checkpoint” 按钮毫无用处**：一位成员询问了 *Restore Checkpoint* 功能，结果发现它基本上无法运行。
   - 讨论强调了界面中仅存在 *accept* 和 *reject* 按钮，确认了 *Restore Checkpoint* 按钮并不可用。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **DeepSeek 为其 Meta Reward 系统辩护**：一名成员对 **DeepSeek** 使用“**Meta Reward Modeling**”一词提出质疑，声称他们实际上构建的是一个“基于评分的奖励系统”，并分享了关于该主题的[论文](https://arxiv.org/abs/2504.05118)和 [YouTube 视频](https://youtu.be/9KMxNZ2CvUg)。
   - 该成员建议使用更准确的名称，如“**voting RM**”来描述其实际机制。
- **DeepSeek Token 定价引发意外**：围绕 **DeepSeek** 的 Token 定价出现了争议，有说法称虽然初始价格看起来较低，但该模型生成的 **Token 数量多出 3 倍**，与 **OpenAI** 等模型相比，可能会导致更高的成本。
   - 反对观点认为，对于 **HTML、CSS 和 TS/JS 生成**等特定任务，**DeepSeek** 可能更具成本效益，并引用了一位用户使用其 AI 网站生成器的经验。
- **内存带宽驱动推理**：讨论强调了在非批处理推理（unbatched inference）中，**内存带宽**与 **Token 吞吐量**之间近乎线性的关系，表明 [RAM 访问是瓶颈](https://discord.com/channels/714501525455634453/986699377257119794/1358590235969065030)。
   - 分享了一个简化方程式：`Max token throughput (tokens/sec) ≈ Memory bandwidth (bytes/s) / Bytes accessed per token`。
- **Google 通过 ADK 和 A2A 加入 Agent 赛道**：**Google** 推出了 **ADK toolkit** ([github.com/google/adk-python](http://www.github.com/google/adk-python))，这是一个用于构建 AI Agent 的**开源 Python 工具包**，并宣布了 **Agent2Agent Protocol (A2A)** ([developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability)) 以提高 Agent 的互操作性。
   - 一些人认为 **A2A** 可能会与 **Anthropic** 的 **Model Context Protocol (MCP)** 竞争，特别是当 Agent 将 **MCP** 作为客户端或服务器使用时。
- **Cogito V1：只是 Triton，但更糟？**：成员们分享并讨论了一种使用测试时计算（test time compute）进行微调的迭代改进策略，涉及来自此 [Hacker News 链接](https://www.deepcogito.com/research/cogito-v1-preview)的 **Cogito V1**。
   - 一名成员不屑地将其总结为“只是 Triton 但更糟”，尽管另一名成员澄清说 **Triton** 与 **Cutile** 类似，但在 **CUDA**、**AMD** 和 **CPU** 上具有更广泛的兼容性。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepCogito 发布开源 LLM 舰队**：**DeepCogito** 发布了开源许可的 **LLM**，尺寸从 **3B** 到 **70B** 不等，使用**迭代蒸馏与放大 (Iterated Distillation and Amplification, IDA)** 技术，性能超越了来自 LLaMA、DeepSeek 和 Qwen 的同等尺寸模型。
   - **IDA** 策略旨在通过迭代自我改进来实现超级智能对齐（superintelligence alignment）。
- **Gemini 2.5 媲美 OpenAIPlus**：据报道，**Gemini 2.5 Deep Research** 与 **OpenAIPlus** 旗鼓相当，包括音频概览选项，如[此 Gemini 分享](https://g.co/gemini/share/9d01ae7abf27)和[此 ChatGPT 分享](https://chatgpt.com/share/67c6919a-1710-800d-9172-853e6045cfe1)所示。
   - 讨论暗示 **Google** 需要精简其 AI 产品，例如针对 *gemini-2.5-flash-preview-04-09-thinking-with-apps* 这种复杂的命名规范的调侃。
- **Google 揭晓液冷 Ironwood TPU**：**Google** 推出了 **Ironwood TPU**，可扩展至 **9,216 个液冷芯片**，采用芯片间互连 (ICI) 网络，功耗接近 **10 MW**，详见[此博客文章](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)。
   - 该公告强调了 Google 在 AI 推理高性能计算领域的推进。
- **MoonshotAI 的 Kimi-VL 开放视觉能力**：**MoonshotAI** 发布了 **Kimi-VL**，这是一个拥有 **16B** 参数（**3B** 激活）且具备视觉能力的模型，采用 MIT 协议，可在 [HuggingFace](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) 上获取。
   - 这一发布标志着对开源多模态 AI 的重要贡献。
- **AI2 迎来最有趣的巅峰时期**：据一位成员称，[AI2](https://allenai.org/) 正处于其最有趣的时期，暗示 AI 研究和开发正在飞速发展。
   - 另一位成员认为，从 **Google** 离职的人虽然被支付了一年的薪水但被强制要求不准工作，同时建议这可能是 **AI2** 启动志愿者计划的一个机会。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Neo4j 为 RAG 提供 MCP 支持**：成员们讨论了将 **MCP** 与 [Neo4j 图数据库](https://neo4j.com/) 结合用于 **RAG**，并建议使用 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 作为与 **Gemini** 兼容的客户端。
   - 讨论集中在 **MCP** 框架内的向量搜索和自定义 **CQL** 搜索功能。
- **A2A 被视为 MCP 技术栈的补充**：Google 的 **A2A** (Agent-to-Agent) [API](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) 与 **MCP** 进行了对比，共识是 Google 将 **A2A** 定位为补充而非替代品。
   - 有人担心 Google 的潜在策略是“将工具层商品化”并主导 Agent 领域。
- **并行工具调用成为瓶颈**：为了并行化对多个 **MCP servers** 的调用，**LLM** 必须在整个宿主端启用“并行工具调用”，包括 `parallel_tool_calls` 标志。
   - 这需要确保聊天模板支持并行工具调用，并向 **MCP server** 发送并行请求。
- **Easymcp v0.4.0 发布包管理器**：[Easymcp](https://github.com/promptmesh/easymcp) **0.4.0** 版本引入了 **ASGI** 风格的进程内 fastmcp 会话、原生 docker 传输、重构的协议实现、新的 mkdocs 和 pytest 设置。
   - 此次更新带来了生命周期改进、错误处理以及针对 MCP servers 的包管理器。
- **ToolHive 将 MCP Servers 容器化**：[ToolHive](https://github.com/StacklokLabs/toolhive) 作为一个 **MCP** 运行器被引入，它通过容器简化了 **MCP servers** 的运行，使用命令 `thv run <MCP name>`，并支持 **SSE** 和 **stdio** 服务器。
   - 该项目旨在统一使用容器运行 MCP servers，并提供安全的选项，详见[此博客文章](https://dev.to/stacklok/toolhive-making-mcp-servers-easy-secure-and-fun-7hi)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **55B 以下数据处理模型对决**：成员们讨论了 55B 以下最适合**数据处理**的模型，提到了 **mistral small3.1**、**gemma3** 和 **qwen32b**，并链接到了一个[高性能模型](https://huggingface.co/open-r1/OlympicCoder-32B)。
   - 原帖作者澄清他们不需要**编码或推理模型**。
- **异常检测模型寻找异常情况**：一位成员请求**异常检测模型**，收到了针对该任务微调的[通用视觉模型](https://huggingface.co/models?other=anomaly-detection)链接，以及一个 [GitHub 仓库](https://github.com/sudhir5595/Anomaly_Detection)和一门[课程](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome)的引用。
   - 还引用了 [AnomalyGPT](https://huggingface.co/FantasticGNU/AnomalyGPT) 模型。
- **Oblix 编排从边缘到云端的 AI**：[Oblix](https://oblix.ai/) 被介绍为一种在边缘和云端之间编排 AI 的工具，在边缘端与 **Ollama** 集成，在云端支持 **OpenAI 和 ClaudeAI**。
   - 创建者正在寻求“CLI 原生、大神级开发者”的反馈。
- **Manus AI 发布基于图的学术推荐系统 Web 应用**：基于图的学术推荐系统 (**GAPRS**) 的第三次迭代作为 Web 应用程序发布，使用了 [Manus AI](https://lqhvwseh.manus.space)。
   - 该项目旨在帮助学生撰写论文，并“彻底改变学术论文的变现方式”，详见其硕士论文。
- **Cogito:32b 在 Ollama 对决中表现出色**：成员们测试了用于 **Ollama** 的 [Cogito:32b 模型](https://ollama.com/library/cogito:32b)，发现 **32b** 模型优于 **Qwen-Coder 32b**，甚至优于 **Gemma3-27b**。
   - 指出该模型运行效果非常好。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 隐私政策受到质疑**：一位用户在注意到系统仅在初始摘要被纠正*之后*才提供正确的摘要后，对 **NotebookLM** 的隐私政策提出了质疑，引发了对数据用于训练的担忧。
   - 另一位用户指出，由于随机性，AI 工具很少会给出两次相同的答案，且模型可能会将用户的踩（downvotes）标记为**冒犯性**或**不安全**。
- **NotebookLM 作为笔记应用面临挑战**：用户发现 **NotebookLM** 过度依赖外部来源，由于其原始的笔记功能，限制了其作为独立笔记应用的实用性。
   - 用户正请求类似于 **Microsoft OneNote** 的组织功能，例如带有可自定义阅读顺序的分区组，以改进笔记管理。
- **请求 Google Drive 集成**：用户请求与 **Google Drive** 集成以保存和启动 **NotebookLM** 笔记本，旨在获得类似于 **Google Docs** 和 **Sheets** 的无缝体验。
   - 目标是让 **NotebookLM** 以目前 **Google Docs** 和 **Google Sheets** 的方式补充 **Google Drive**。
- **导入 Microsoft OneNote：是否可行？**：用户希望能够将笔记本从 **Microsoft OneNote** 导入 **NotebookLM**，包括分区和分区组，可能通过 **.onepkg** 文件实现。
   - 一位用户承认存在法律合规方面的担忧，但将其类比于 **Google Drive** 导入 **Microsoft Word** 文档的能力。
- **移动端音频概览（Audio Overviews）故障**：用户报告称 **2.5 Pro** 深度研究功能声称可以生成**音频概览**，但该功能在移动端失败。
   - 据报道，该功能在网页端运行正常，用户建议通过正规渠道报告此问题。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **通过 CUTLASS 启用 Flash Attention 3**：成员们讨论了在 **5090** 上从 **FP4** 开始，建议使用 **CUTLASS** 以利用 Tensor Cores 并使用 **Flash Attention 3**，并链接到了[一个示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。
   - 团队还发布了 [torchao 0.10](https://github.com/pytorch/ao/releases/tag/v0.10.0)，增加了许多 **MX** 特性，包括针对 **MX dtypes** 的 [README](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/README.md)。
- **Linux 发行版辩论引发 NVIDIA 驱动讨论**：一位成员询问哪种 **Linux 发行版**在使用 **NVIDIA 驱动**时痛苦最少，并就 **LDSM**（共享内存）指令提出了澄清问题，发布了 `SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>; auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);`。
   - 另一位成员同意每个线程从源加载数据，线程交换数据，然后将数据存储到目的地，并建议使用 **warp shuffling** 的可能性，并提供了 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=load#warp-level-matrix-load-instruction-ldmatrix)的链接。
- **FSDP2 面临并行化障碍**：成员们表示由于 **FSDP2** 与其他并行化方法相比具有独特的设计，集成起来非常困难。
   - 有人指出 **Accelerate** 中使用的一个黑科技（hack）与当前方法冲突，使集成过程复杂化。
- **Mediant32：FP32/BF16 的整数替代方案**：一位成员宣布了 **Mediant32**，这是一种基于有理数（Rationals）、连分数（continued fractions）和 Stern-Brocot 树的实验性替代方案，用于纯整数推理，并提供了[分步实现指南](https://leetarxiv.substack.com/p/mediant32-intro)。
   - **Mediant32** 使用基于**有理数**、**连分数**和 **Stern-Brocot 树**的数字系统，为数值表示提供了一种新颖的方法。
- **DeepCoder 加入开源阵营**：一位成员分享了 [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) 的链接，这是一个完全开源的 **14B 代码模型**，达到了 **O3-mini** 级别。
   - 此外，一位成员注意到 **Llama 4 Scout** 已添加到 [GitHub](https://github.com/open-thought/reasoning-gym-eval/pull/6)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Together AI 发布 X-Ware.v0**：**Together AI** 在[这条推文](https://x.com/togethercompute/status/1909697122372378908)中宣布发布 **X-Ware.v0**，社区成员目前正在对其进行测试。
   - 社区正在观察 **X-Ware.v0** 的运行表现。
- **Gemiji 的 Pokemon 游戏演示引起关注**：一位成员分享了 **Gemiji** 玩 **Pokemon** 的[链接](https://x.com/kiranvodrahalli/status/1909699142265557208)，引起了积极关注。
   - 该帖子链接到了 Kiran Vodrahalli 的一条推文。
- **AI Excel 公式引发热议**：一位 AI Engineer 分享了[一个链接](https://x.com/diegocabezas01/status/1909221066565734854)，表达了对 AI/LLM Excel 公式及其广泛应用潜力的兴奋。
   - 该成员提到他们一直在思考这种 AI/LLM Excel 公式，并提到一位朋友成功使用了 **TextGrad**。
- **Copilot 成为独立游戏开发助手**：成员们探讨了 [Microsoft Copilot](https://copilot.microsoft.com/wham?features=labs-wham-enabled) 在独立游戏开发中的用途，强调 Agent 是有效的工具。
   - 代码生成 Agent 工具被认为有助于交付成品，levels io 的 game jam 被引用为令人大开眼界。
- **Google 推出 Agent2Agent Protocol (A2A)**：**Google** 推出了 **Agent2Agent Protocol (A2A)** 以增强 Agent 的互操作性，完整规范可在[此处](https://github.com/google/A2A)查看，一位成员提到他们参与了其中。
   - 还提供了一个与 **MCP** 的对比 ([链接](https://google.github.io/A2A/#/topics/a2a_and_mcp.md))。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepCogito LLM 发布**：[DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview) 发布了开源 LLM，规模包括 **3B**、**8B**、**14B**、**32B** 和 **70B**，采用了迭代蒸馏与放大（Iterated Distillation and Amplification）策略。
   - 每个模型在大多数标准基准测试中都优于同等规模的最佳开源模型，包括来自 **LLaMA**、**DeepSeek** 和 **Qwen** 的对应模型；**70B** 模型甚至优于新发布的 **Llama 4 109B MoE** 模型。
- **Hermes 微调避开灾难**：成员们表示，在 **Llama 4** 模型上微调新的 **Hermes** 将是一场灾难，但已经准备好测试来剔除（yeet）糟糕的合并。
   - 大家一致认为 **Llama 4** 在某些方面仍有价值，不可能在所有方面都更差。
- **模型模仿人类辩论风格**：一位成员让两个模型互相辩论，观察到它们模仿了人类的辩论方式，“从不试图理解对方的观点，无论论据如何都坚持自己的立场”。
   - 模型有选择性地攻击弱点，忽视自身的漏洞，并专注于利用对手的立场。
- **Qwen 2.5 1.5B Instruct 训练进展**：一位成员正在对 **Qwen 2.5 1.5B Instruct** 进行 **RL**，并将 **gsm8k** 数据集更换为 **gsm8k platinum**，启用了 **RsLora**，模型似乎在更少的步数内学习得快得多。
   - 改进可能源于使用了歧义更少的数据集，以及在多大程度上归功于使用了 **RsLora**。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **建议用户在本地进行 Embedding 以确保安全**：成员们正在讨论在本地运行 Embedding 模型和 LLM 的好处，以避免将私密信息发送到远程服务。一位成员分享了一个用于运行 **Nomic** 本地 Embedding 模型的 [Shell 脚本](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html)。
   - 该脚本使用 `$LLAMA_SERVER`、`$NGL_FLAG`、`$HOST`、`$EMBEDDING_PORT` 和 `$EMBEDDING_MODEL` 等变量来配置和运行 Embedding 服务器。
- **GPT4All 在本地索引文档**：一位用户澄清说，**GPT4All** 通过对文档进行分块（chunking）和 Embedding 来建立索引，并将相似性的表示存储在私有缓存中，从而避免使用外部服务。
   - 他们建议，即使是 **Qwen 0.5B** 参数模型也能很好地处理本地 Embedding 文档，不过 **Qwen 1.5B** 效果更好。
- **用户在加载本地 LLM 时遇到困难**：一位成员报告在加载本地 LLM 时被卡住，尽管拥有 **16GB RAM** 和 **Intel i7-1255U CPU**，怀疑是模型下载出了问题。
   - 该用户正在创建一个内部文档工具，不愿将私密文档用于远程服务。
- **使用 Shell 脚本 DIY RAG**：一位成员分享了 Shell 脚本示例（`rcd-llm.sh` 和 `rcd-llm-get-embeddings.sh`），用于获取 Embedding 并向本地 LLM 发送 Prompt，从而创建自定义的 **RAG** 实现。
   - 他们建议使用 **PostgreSQL** 存储 Embedding，而不是依赖远程工具。
- **GPT4All 的停止按钮也是它的开始按钮**：一位用户询问如何停止 **GPT4All** 中的文本生成，提到没有看到明显的停止按钮，也无法使用 **Ctrl+C**。
   - 另一位用户指出停止按钮位于右下角，与生成按钮共用同一个按钮。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **新手开启 Mojo 之旅**：一位新用户询问如何学习 **Mojo** 语言，另一位用户引导他们参考 [官方 Mojo 文档](https://docs.modular.com/mojo/manual/)，认为这是一个很好的起点。
   - 该成员还强调了 **Mojo 社区**，将用户引向 [Modular 论坛的 Mojo 板块](https://forum.modular.com/c/mojo/7) 和 Discord 上的 general 频道。
- **Span 生命周期问题困扰 Mojo Trait**：一位成员寻求建议，想在 **Mojo** 中表达 *返回的 Span 的生命周期至少与 self 的生命周期一样长*，并提供了 [Rust/Mojo 代码示例](https://forum.modular.com/t/how-to-return-a-span-that-refers-to-a-struct-member-from-a-trait-method/1216)。
   - 回复指出，*让 Trait 对 origin 进行泛型化* 是一个可能的解决方案，尽管可能需要 Trait 参数支持。
- **Mojo 关注无畏并发（Fearless Concurrency）**：有人提问 *Mojo 是否具有类似 Rust 的无畏并发*。
   - 得到的回答是 Mojo 已经具备了所需的 Borrow Checker 约束，目前仅缺乏 **Send/Sync** 和最终的并发模型；它最终甚至可能拥有比 Rust 更好的系统。
- **MLIR 类型构造遭遇编译时灾难**：一位成员报告了在 MAX/Mojo 标准库中使用 *MLIR 类型构造中的参数化编译时值*（特别是 **!llvm.array** 和 **!llvm.ptr**）时遇到的问题，并在 [GitHub post](https://github.com/modular/max/issues/4315) 中详细说明了该问题。
   - 问题涉及在定义带有编译时参数的结构体（用于 **llvm.array** 类型）时的解析错误；MLIR 的类型系统似乎无法在此上下文中处理参数化值。
- **POP 来救场？**：针对 MLIR 问题，另一位成员建议使用 *参数化操作方言（Parametric Operations Dialect, POP）*。
   - 他们建议 Mojo 团队增加一些功能，例如让 **__mlir_type[...]** 宏接受符号化的编译时值，或者提供类似 **__mlir_fold(size)** 的辅助工具，以强制将参数评估为字面量 IR 属性。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Auth0 将身份验证接入 GenAI**：Auth0 的 GenAI 身份验证现在支持 **LlamaIndex**，通过 SDK 调用简化了 **Agent** 工作流中的身份验证集成。
   - auth0-ai-llamaindex SDK (**Python** & **Typescript**) 支持 **FGA-authorized RAG**，如[此演示](https://t.co/bZgQ7gpuSt)所示。
- **Agent 通过视觉引用看得更清楚**：LlamaIndex 推出了一项关于使用**视觉引用 (visual citations)**来锚定 **Agent** 的教程，将生成的答案链接到文档的特定区域。
   - 该功能的运行版本可以直接在[此处](https://t.co/LP5XA8Yn0c)获取。
- **征集推理 LLM 方案**：一位成员正在寻求从 **Hugging Face** 实现**推理 LLM (reasoning LLMs)** 的官方教程，旨在用于 **Hugging Face Space** 上的 **Docker** 应用。
   - 目前的讨论中尚未找到解决方案。
- **区块链专家提供支持**：一位在区块链领域具有专业知识的软件工程师提供区块链项目协助，擅长 **DEX**、**bridge**、**NFT marketplace**、**token launchpad**、**stable coin**、**mining** 和 **staking protocols**。
   - 该工程师正“尝试学习更多关于 **LlamaIndex** 的知识”。
- **Create Llama 旨在辅助 AI**：一位成员建议使用 [create-llama](https://x.com/MarcusSchiesser/status/1907448102467911985) 工具来帮助用户通过 **LlamaIndex** 进行深入研究。
   - 该工具旨在帮助快速创建 **LlamaIndex** 项目。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 文档引发讨论**：一位成员询问了使用 **Cohere** 生成结构化输出（如书籍列表）的示例，并被引导至 [Cohere documentation](https://docs.cohere.com)。
   - 讨论强调了利用 **Cohere** 的资源来获取生成特定输出格式的指导。
- **Pydantic Schema 引发询问**：一位成员询问了在 `response_format` 中直接使用 **Pydantic schema** 以及在 **Python** 中不使用 **Cohere** 库发送请求的问题。
   - 共享了 [chat reference](https://docs.cohere.com/reference/chat) 的链接，建议切换到 **cURL** 以获取 **API** 交互的见解。
- **关于公司列表生成模型的讨论**：一位成员就根据给定主题生成公司列表的最佳模型寻求建议。
   - 建议指出 **Cohere** 目前最快且最强大的生成模型是 **command**。
- **新成员 Aditya 加入，旨在将 AI 应用于 Openchains**：Aditya 拥有**机器视觉**和**制造设备控制**背景，正在探索 **web/AI** 并分享了他的当前项目 [openchain.earth](https://openchain.earth)。
   - 他热衷于将 **Cohere** 的 **AI** 集成到他的项目中，利用他的技术栈，包括 **VS Code, Github Co-Pilot, Flutter, MongoDB, JS** 和 **Python**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **PMPP 书籍被推荐用于 GPU 编程**：一位成员建议使用 **PMPP (第 4 版)** 进行 **GPU** 编程，并征求编译器建议。
   - 另一位成员表示他们正在研究[这个编译器系列](https://marcauberer.medium.com/build-a-compiler-parser-7bf4b7381ca5)，并且也会学习 [LLVM Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)。
- **METAL 同步问题导致 LLaMA 7B 运行受阻**：一位用户在 **METAL** 后端的 **4 个虚拟 GPU** 上运行 **LLaMA 7B** 时遇到了 `AssertionError`，这与 `MultiLazyBuffer` 和 `Ops.EXPAND` 有关。
   - 用户通过在 [PR 9761](https://github.com/tinygrad/tinygrad/pull/9761/files) 中移动 **tensor** 以在采样后保留设备信息，修复了该问题。
- **梯度累积难题已解决**：一位用户报告称，在他们的训练程序中调用 `backward()` 不起作用，且在 `opt.step()` 之前 `t.grad` 为 `None`。
   - 用户发现，在执行 step 之前调用 `zero_grad()` 修复了梯度累积过程中的 `t.grad is None` 问题。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **来自 Psych 的 Gus 加入了 Torchtune？**：一位成员为其 [GitHub profile](https://github.com/nathan-az) 申请了 **Contributor tag**，并幽默地引用了电视剧 **Psych** 中的角色 **Gus**。
   - 另一位成员用 [Gus-wave GIF](https://tenor.com/view/gus-wave-guswave-gif-18773699) 欢迎新团队成员，开玩笑地提到了电视剧 *Psych*。
- **FSDP 与 PyTorch 协作良好**：Torchtune 默认使用等同于 **zero3** 的配置，并能很好地与 **PyTorch distributed features**（如 **FSDP**）结合。
   - 一位用户转向使用 torchtune 是为了 *避免在尝试组合 deepspeed + pytorch + megatron 时踩坑*，并希望 *我们不要过度投入到集成和支持其他框架上*。
- **DeepSpeed Recipe 受到欢迎**：团队欢迎导入 torchtune 并托管 **DeepSpeed recipe** 的仓库，这需要单设备副本并添加 DeepSpeed。
   - 团队对此表示热烈肯定。
- **分片策略支持变得简单**：支持不同的 **sharding strategies** 非常直接，用户可以使用 **FSDPModule** 方法调整 recipe，以在等同于 **zero1-2** 的模式下进行训练。
   - 团队确认，只需对 collectives 进行微调，**zero 1-3** 都是可以实现的。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 导师失踪了？**：一位成员在 **#mooc-questions** 频道询问关于在 **AgentX** 研究赛道中接收导师反馈的问题。
   - 未提供更多信息。
- **占位主题**：这是一个占位主题，以满足所需的最小条目数。
   - 如果有可用信息，将在此处添加。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 席卷 JetBrains IDEs**：Windsurf 发布了 **Wave 7**，将其 **AI agent** 带到了 JetBrains IDEs（**IntelliJ**、**WebStorm**、**PyCharm**、**GoLand**），详情见其 [blog post](https://windsurf.com/blog/windsurf-wave-7)。
   - Beta 版本整合了核心的 Cascade 功能，如 **Write mode**、**Chat mode**、**premium models** 和 **Terminal integration**，未来的更新承诺提供更多功能，如 **MCP**、**Memories**、**Previews & Deploys**（[changelog](https://windsurf.com/changelog/jetbrains)）。
- **Codeium 开启新篇章，更名为 Windsurf**：公司已正式更名为 **Windsurf**，告别了经常被拼错的 Codeium，并将其 AI 原生编辑器重命名为 **Windsurf Editor**，IDE 集成重命名为 **Windsurf Plugins**。
   - 该消息已在 [Twitter](https://x.com/windsurf_ai/status/1910037538028524030)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lmfms7w3n227)、[YouTube](https://www.youtube.com/watch?v=TZ8UVFiTfdU)、[Instagram](https://www.instagram.com/p/DIPFz2NSTUI/) 和 [TikTok](https://www.tiktok.com/@windsurf/video/7491376934522309919) 上宣布。

---

**DSPy Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1359241652799275158)** (1004 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro, DeepMind Ultra, NightWhisper 推测, Gemini Coder 模型, Deep Research 更新` 


- **Gemini 2.5 Pro 被誉为真正的 AI**: 成员们对 **Gemini 2.5 Pro** 表示兴奋，一位用户声称它是第一个“真正的” AI，在创意写作方面非常有用，并期待 [Gemini coding model](https://arxiv.org/abs/2402.10176) 的发布。
   - 一些人讨论了 Gemini 2.5 及其局限性，该用户表示它不能编写*所有*代码，但在创意和连贯的写作以及写实故事方面非常有用！
- **关于 DeepMind Ultra 模型的推测升温**: 关于 **DeepMind Ultra 模型** 的推测层出不穷，有理论认为它即将发布，一些人相信它将被集成到 [AI Studio](https://ai.google.dev/) 中供免费使用。
   - 猜测范围从 6 月的 I/O 大会揭晓到 12 月/11 月与 **Gemini 3** 一同发布，可能在 8 月与 **GPT-5** 竞争；然而，一位成员开玩笑说这个 nightwhisper 只是个笑话，但现在显而易见 **Ultra** 即将到来。
- **“NightWhisper” 编程模型之梦**: 一位用户宣布发布 [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder)，这是一个基于 Deepseek-R1-Distilled-Qwen-14B 微调的代码推理模型，称其为“被释放到世界的 baby nightwhisper”。
   - 许多成员期待一个名为 **NightWhisper** 的编程模型，一些人预计它将由 **Gemini 2.5 Pro** 驱动并很快发布，然而，一些用户声称 nightwhisper 仅仅是带有 tool calls 的 Gemini 2.5。
- **Google vs OpenAI：基础设施与 AGI 竞赛**: 一场关于 Google 和 OpenAI 比较的详细讨论，成员们认为 **Google 的基础设施**（TPUs、更便宜的算力以及与 Google 产品的集成）赋予了其显著优势，并预测 Google 将比 OpenAI 更早实现 AGI，一位用户提到 [Gemini 3.0 正在设计 TPUs](https://cloud.google.com/blog/products/compute/google-cloud-tpu-v5p-general-availability)。
   - 其他人反驳说 OpenAI 的研究和进展很有价值，强调 post-training 更新显示他们在推理方面变得很强，然而，一位用户表示 OpenAI 只是一个“烧钱的、制作动漫的作业帮手”。
- **体验 AI Studio 功能**: 成员们探索了 AI Studio，赞扬其对开发者友好的界面和控制力，并强调了新模型的引入（如 Gemini Flash），以及流式传输内容和使用不同 system prompts 测试同一模型的能力，提到 [UI 看起来好多了](https://tenor.com/view/case-oh-caseoh-waffle-house-waffle-house-gif-10934642274965704175)。
   - 还讨论了实时流媒体和 function calling 的能力，然而，遗憾的是不能同时使用多个工具，另一位用户对此表示“没门，伙计”。

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1359241458992807967)** (712 条消息🔥🔥🔥): 

> `使用 Unsloth 进行 GPU 配置，DDP (Distributed Data Parallel) 与 Unsloth 性能对比，VLLM 与 Unsloth 的集成，Llama 4 Scout 模型分析，模型量化` 


- **解决 Unsloth 的 GPU 配置问题**：用户讨论了通过设置 **CUDA_VISIBLE_DEVICES** 来强制 Unsloth 在特定 GPU 上运行，解决了在未明确指定 GPU 时遇到的错误。
   - 一位用户指出：*“一旦我强制所有内容都在一个 GPU 上运行，Unsloth 就能完美工作了”*。
- **DDP (Distributed Data Parallel) 在性能上优于 Unsloth**：一位用户发现，尽管 Unsloth 在单 GPU 上运行完美，但 **DDP** 在单 GPU 上的速度比 Unsloth 更快。
   - 讨论澄清了 DDP 指的是 **Distributed Data Parallel**，而非去噪扩散概率模型（denoising diffusion probabilistic models），强调了缩写词引起的混淆。
- **探索集成 VLLM 以提升 Unsloth**：有建议提出将 **VLLM** 与 Unsloth 结合使用以实现更快的推理，特别是在批量探索场景中，尽管澄清了原始用户并未使用 VLLM。
   - 鼓励进行 Unsloth 和 VLLM 的实验以比较前向传播速度，并提醒在稳定的 RL 中，Unsloth 和 VLLM 的 logits 之间的 KL 散度理想情况下应为零。
- **GGUF 版本让 Scout 表现优于 16-bit 版本**：社区评估了 **Llama 4 Scout**，指出其基座模型经过了极度的指令微调（instruct-tuned），且 2-bit 量化版本在 MMLU 上的表现优于原始 16-bit 版本。
   - 普遍共识是推理提供商的实现中存在某些问题，因为 Unsloth 的量化版本性能超过了完整的 16-bit 版本。
- **DeepCogito 的主张引发关注**：成员们分享了 [DeepCogito 的 Cogito V1 预览版](https://www.deepcogito.com/research/cogito-v1-preview)链接，该模型声称其性能优于 **LLaMA**、**DeepSeek** 和 **Qwen** 等模型，但大家对这些主张持合理的怀疑态度。
   - 讨论还涉及了医疗 AI 面临的挑战，强调需要防止仓促、低质量的实现损害消费者利益，同时还讨论了潜在的隐私问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1359276394424565780)** (22 条消息🔥): 

> `模型剪枝，Model2Vec，GoC79hYXwAAPTMs.jpg，基于 Transformer 的模型` 


- **公司是否利用用户输入来剪枝模型？**：一位成员推测 **OpenAI**、**Claude** 和 **Gemini** 等公司利用用户输入来剪枝模型，并引用 *“你更喜欢哪一个”* 之类的回复作为收集训练用户偏好数据的一种手段。
   - 另一位成员表示赞同，将其比作一种在线 DPO，它开始比你更了解你自己；还有一位成员开玩笑说他们总是选差的那个。
- **Model2Vec 使用案例**：一位成员分享了 **Model2Vec** 的[链接](https://x.com/_avichawla/status/1909857444953772442)，并补充说它是真实有效的，但其使用场景非常特定，并不是任何东西的直接替代品。
   - 他们还分享了一个关于 Model2Vec 的 [YouTube 视频链接](https://www.youtube.com/watch?v=4lOGcmheASs)。
- **Model2Vec 生成文本嵌入速度更快**：据一位成员称，**Model2Vec** 牺牲了一定的质量，但生成文本嵌入的速度比常用的**基于 Transformer 的模型**快得多。
   - 他们想知道这是否也可以用于 TTS。
- **Decoder 获得 Llama.cpp 集成**：一位成员指出了一条重大新闻：Decoder 现在已经有了 [llama.cpp 集成](https://github.com/ggml-org/llama.cpp/pull/12828#issuecomment-2787939068)。
   - 随后没有进一步讨论。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1359276675036348459)** (156 条消息🔥🔥): 

> `大模型 GRPO 训练技巧、多 GPU GRPO、4-bit 训练、本地运行 Orpheus TTS、Gemma 和 Granite 训练错误` 


- ****GRPO 专家为 Gemma 生成获取 GPU 秘籍****：一位用户寻求[使用 GRPO 的技巧](https://link.to/grpo)，以便在 **H200 GPU** 上训练一个 **16k 上下文长度** 的 **24B 模型**，报告的 batch size 为 1，model_gpu_util=0.7。
   - 建议包括增加 **gradient accumulation**，并讨论了通过 Unsloth 和其他框架实现的 **multi-GPU 支持**，尽管这需要大量的 VRAM。
- ****Gemma 故障引发 Granite 抱怨并获得指导****：一位用户在尝试训练 **Gemma** 和 **Granite** 时遇到了 **dtype mismatch 错误**，尽管尝试了不同的配置和软件包版本，并寻求社区帮助。
   - 经过排查，确定 [Transformers 版本](https://huggingface.co/docs/transformers/installation) 与 Gemma3 不兼容，并提出了一个涉及设置 `dtype=torch.float16` 的潜在修复方案。
- ****Nvidia 发布的新数据集引起关注****：**Nvidia** 发布了 [OpenCodeReasoning 数据集](https://huggingface.co/datasets/nvidia/OpenCodeReasoning)，用户正在寻找在 Unsloth 中使用该数据集的解决方案和示例。
   - 该数据集的 reward function 稍微复杂一些。
- ****Orpheus 专家提供输出选项****：一位用户询问如何从文本输入在本地运行 **unsloth 版本的 Orpheus TTS**，目标是流式 WAV 音频输出。
   - 建议通过 **vLLM** 运行并使用 [这个项目](https://github.com/isaiahbjork/orpheus-tts-local)，该项目最初是为 LM Studio 设计的，但它使用了 OpenAI 兼容的 API，因此 vLLM 也可以工作。
- ****KTransformer 难题困扰 Qwen 查询****：一位用户在成功将 ktransformers 用于 DeepSeek V3 后，寻求关于[如何进行推理](https://link.to/inference) **Qwen 2.5 72B 模型** 的指导。
   - 建议该用户联系 ktransformer 团队，因为他们需要支持该架构。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1359245049044668467)** (3 条消息): 

> `用户的地理来源，比利时与荷兰的邻近性` 


- **用户国籍揭晓**：一位用户询问另一位用户是否来自法国，暗示对地理来源的兴趣。
   - 另一位用户澄清他们来自荷兰，促使第一位用户回应他们来自比利时，**表明两人距离很近**。
- **比利时靠近荷兰**：两位用户发现了彼此的国籍，并意识到他们来自邻国。
   - 对话强调了欧洲境内 **比利时和荷兰的地理邻近性**。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1359275046731911291)** (8 条消息🔥): 

> `Together AI 的 DeepCoder、Apple Metal 量化内核、量化视觉指南` 


- **Together AI 解析 DeepCoder**：一位成员分享了 [Together AI 关于 DeepCoder 的博客文章](https://www.together.ai/blog/deepcoder)，强调了其在优化 *vllm* 流水线方面的潜力。
   - 该技术通过在进行下一次采样的同时执行初始采样和训练，从而最大限度地减少等待时间。
- **Apple Metal 量化内核公开**：一位成员分享了来自此 [GitHub commit](https://github.com/ggml-org/llama.cpp/blob/d3bd7193ba66c15963fd1c59448f22019a8caf6e/ggml/src/ggml-metal/ggml-metal.metal#L4077) 的 ggml Apple Metal 量化内核代码。
   - 他们分享这个链接是因为他们花了 *数周时间才弄明白*。
- **量化可视化**：一位成员分享了 [量化视觉指南](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-quantization)，并指出这对他们的工作很有帮助。
   - 这篇文章对量化方法进行了直观且形象的介绍。


  

---

### **Manus.im Discord ▷ #[showcase](https://discord.com/channels/1348819876348825620/1348823595505156137/1359384200720941096)** (6 条消息): 

> `Website Building Code, Japan Cherry Blossom Trip Website, Galaxy Model, Impact of Tariffs on Consumers, Recommender System` 


- **Manus 提供网站构建代码**：Manus 提供了全面的 **网站构建代码**。
   - 该代码旨在增强一个 **日本樱花之旅网站**。
- **展示了惊人的星系模型**：展示了一个令人惊叹的 **星系模型 (galaxy model)**。
   - 未提供关于该模型规格或创建过程的进一步细节。
- **关税对消费者的潜在影响**：开始讨论关于 **关税对消费者** 的潜在影响。
   - 提供的消息中未分享具体细节或相关分析的链接。
- **重点展示了推荐系统**：展示了一个实用的 **推荐系统 (Recommender System)**。
   - 未提供关于该系统的额外上下文或链接。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1359242025203007549)** (511 条消息🔥🔥🔥): 

> `AI for App Creativity, Gemini 2.5 and Claude 3.7 for Coding, Best Hosting for Social Media App, Manus Credit Usage, Improving Apps Post-Launch` 


- **Gemini + Claude：应用开发的黄金搭档**：对于应用开发，成员们建议使用 **Gemini** 进行多模态分析（图片/视频），并使用 **Claude** 作为数据库来存储研究和项目文件，以便进行战略规划。
   - 建议利用 Gemini 进行深度研究，然后将 Claude 作为项目的数据库。
- **高性价比 AI 协作：Manus + 训练过的 AI**：一位成员分享了一种策略，即 *针对特定任务训练一个 AI*，然后让它与 **Manus** 协作，以高性价比的方式完成项目。
   - 这涉及预先进行准备工作，以尽量减少额度 (credit) 的消耗。
- **账号消失了？脑回路短路，已解决！**：一位成员报告了一个问题，即他们的 **登录邮箱无法识别**，显示 *“用户不存在”*，尽管已经购买了额度。
   - 该成员随后解决了问题，意识到他们最初使用了不同的登录方式：*“工作太多导致脑回路短路 😅 🤣。”*
- **网站烦恼？UI/UX 代码需要帮助**：一位用户指出，网站出现图片和功能无法运行等问题，可能是由于 **糟糕的 UI/UX 代码** 造成的。
   - 他们建议使用 [LLM Studio](https://llm.studio) 来突出显示代码错误，然后将其输入到 **Sonnet 3.7** 以获得更好的结果，同时配合 DeepSeek R1 或 Perplexity 等工具。
- **DeepSite：很棒，但会被清除**：一位成员指出 [DeepSite](https://deepsite.site)（一个网站创建工具）虽然很好但存在 Bug，会出现已完成的网站被删除的情况，并将其描述为具有用于 HTML 的 Claude artifact。
   - 它被认为速度极快，比 **Claude 快 10 倍**。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1359276120368742472)** (2 条消息): 

> `Perplexity for Startups, Aravind AMA` 


- **Perplexity 推出初创企业计划**：Perplexity 正在启动一项 [初创企业计划](https://www.perplexity.ai/startups)，为符合条件的初创公司提供 **$5000 的 API 额度** 和 **6 个月的 Perplexity Enterprise Pro**。
   - 初创公司必须获得 **少于 2000 万美元的股权融资**，成立时间 **少于 5 年**，并且与 Perplexity 的初创企业合作伙伴之一有关联。
- **Aravind 在 Reddit 上主持 AMA**：Aravind 在 PDT 时间上午 9:30 - 11:00 主持了一场 [Reddit AMA](https://www.reddit.com/r/perplexity_ai/comments/1jv9hvm/ama_with_perplexity_cofounder_and_ceo_aravind/)，讨论 Perplexity 的愿景、产品和搜索的未来。
   - 在 AMA 期间，他回答了关于 Perplexity 的目标及其未来计划的问题。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1359241453901189251)** (501 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro 推理 Token，Perplexity Discover 偏见，Deepseek 10 美元深度搜索，Perplexity NHL 体育，故障排除任务与深度研究` 


- **Perplexity 解释 Gemini 2.5 Pro 缺失推理 Token 的原因**：一名工作人员证实 **Gemini 2.5 Pro** 不公开推理 Token，这导致它无法作为推理模型加入 Perplexity，尽管它是一个高延迟的思考模型。
   - 他们澄清说，**推理 Token** 仍然会被消耗，并影响输出的 Token 计数，这解释了为什么他们*无法将其作为推理模型加入。*
- **用户深入探讨 Discovery 标签页的偏见**：一名成员询问了 Perplexity Discover 中“为你推荐”和“热门故事”标签页的页面选择过程，质疑 Discovery 标签页中潜在的**偏见**。
   - 他们推测用户提示词会为相关主题生成页面，但选择热门故事的机制仍不明确，引发了关于偏见如何影响内容可见性的疑问。
- **Deepseek 深度搜索费用为 10 美元**：成员们讨论了 AI 服务的定价策略，一些人建议采用**积分系统**或更便宜的方案（20 美元以下）供偶尔的深度研究使用。
   - 一位成员称赞 **Deepseek 的 10 美元深度搜索**是一个潜在的模型，而另一位成员预测 Deepseek 很快将提供自己的 Deep Research 工具。
- **Perplexity 不了解冰球**：一位用户报告称 Perplexity 的体育新闻功能无法识别 **New Jersey Devils** 或任何 NHL 球队，并对冰球作为一项主要运动却被忽略表示失望。
   - 一名工作人员承认了这一问题，并确认 **NHL、F1 和其他运动**已列入 Perplexity 未来产品的路线图中。
- **频繁的故障排除引发深度研究辩论**：用户们辩论了运行大量深度研究查询的实用性，一名成员声称每小时使用深度研究处理故障排除任务**多达 20 次**。
   - 另一位用户质疑如此频繁进行深度研究的必要性，认为这更典型地用于企业级用途，或者在面对基因工程等复杂任务的死胡同时使用。一名工作人员对报告的数量之大发表了评论。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1359505592938533006)** (1 条消息): 

> `Holo-live` 


- **什么是 Holo-live？**：一位用户在[这里](https://www.perplexity.ai/search/explain-what-is-holo-live-2redPZIGSUGQ1lx5Gm_I2g#1)询问：*解释什么是 Holo-live*。
- **为满足最低要求的填充主题**：这是一个填充主题，以确保 `topicSummaries` 数组满足架构中指定的至少 2 个项目的最低要求。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1359381457696460832)** (6 条消息): 

> `API 调用中的图像，Sonar 与 Make.com，Playground 对比 API` 


- **API 调用中的图像功能即将推出**：一名成员询问关于在 **API 调用中传递图像**的问题，最初发现该功能尚不支持。
   - 另一名成员确认该功能应在本周末前可用；**API 调用中的图像传递即将推出！**
- **Perplexity Office Hours 与 Sonar 的烦恼**：一名成员分享了 [Office Hours 注册链接](https://x.com/LiounisJames/status/1909710546485518522)以及[更多关于预期内容的细节](https://x.com/PPLXDevs/status/1909686050907394053)。
   - 他们还询问了使用 **Sonar 与 Make.com** 的经验，指出存在集成问题并寻求修复方案，表示他们*收到了很多关于其运行不如预期的报告*。
- **Playground 搜索效果优于 API**：一名成员报告称，与 **API** 相比，**Playground** 搜索的网站不同，且 Playground 返回的结果*通常更相关*。
   - 该成员询问如何修复这一差异。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1359349816118743130)** (5 messages): 

> `Olympia.chat 出售，基于 Quasar 的 OSS AI agent 工具，迭代式代码生成` 


- **Olympia Chat 初创项目寻求买家！**: [Olympia.chat](https://olympia.chat) 的创始人（现任 Shopify 首席工程师）正在为这个盈利的 SaaS 初创公司寻找新主人，该项目每月产生超过 **$3k USD** 的收入。
   - 有意向者可联系 vika@olympia.chat 了解收购这一交钥匙运营项目的详情，包括 **IP、代码、域名和客户列表**。
- **Quasar 模型赋能免费 AI Agent 工具**: 一位工程师正在开发 **OSS tooling**，让 **AI agents** 能够原生理解代码，并强调了其在 **Claude/Gemini 2.5** 以及 OpenRouter 最近推出的 **Quasar model** 上的有效性。
   - 该工具支持 **原生 GitHub 集成**，利用 **Quasar model** 实现 **免费 AI agent** 辅助解决 issue 和 PR 审查；安装说明可在 [GitHub](https://probeai.dev/integrations/github-actions) 查看。
- **迭代式代码生成模拟人类调试**: 一种新的 **AI 代码生成** 方法涉及迭代执行、逐行调试以及基于实际错误的针对性修复，模拟了软件工程师编写代码的方式。
   - 该方法旨在通过为模型创建更紧密的执行/反馈循环来提高代码解决方案的可靠性，在线演示见 [此处](https://www.agentsbase.ai/iterative_code_generation.html)，代码已托管至 [GitHub](https://github.com/rohanarun/iterative-code-generation)。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1359241438407426118)** (444 messages🔥🔥🔥): 

> `DeepSeek v3, OpenRouter 定价, Google Cloud Next 发布会, Gemini 2.5 Pro, API 连接问题` 


- **DeepSeek v3 表现出色**: 成员们讨论了新的 [DeepSeek v3 0324](https://openrouter.ai/deepseek/deepseek-chat-v3-0324:free) 模型，有人声称其表现优于之前的版本，甚至超过了 **R1**，但也有人持怀疑态度。
- **OpenRouter 价格点争议**: 在 OpenRouter 实施了根据账户余额影响速率限制（rate limits）的新变化后，一些用户对平台的定价、用户体验以及感知到的向 **利润优先** 的转变表示担忧。
   - 一位用户分享了替代平台（[G2.com](https://www.g2.com/products/openrouter/competitors/alternatives) 和 [EdenAI](https://www.edenai.co/post/best-alternatives-to-openrouter)），并表示打算因感知到的“贪婪”给 OpenRouter 打差评，这引发了辩论。
- **Google Cloud Next 发布 A2A**: Google 发布了 **A2A**，这是一种补充 Anthropic 的 Model Context Protocol 的开放协议，旨在为 agents 提供有用的工具和上下文，详情见 [GitHub 仓库](https://github.com/google/A2A)。
- **Gemini 2.5 Pro 遭遇容量限制**: 用户报告了 [Gemini 2.5 Pro Experimental model](https://ai.google.dev/) 的 **速率限制 (rate limits)**，免费版本的限制为 **80 RPD**，但使用付费 key 的用户拥有更高的上限。
   - 团队确认由于 **容量限制 (capacity constraints)**，存在端点限制。
- **连接 OpenRouter 的 API 问题**: 一位用户报告了 **ping api.openrouter.ai** 失败以及脚本运行困难（DNS 错误）的问题。正确的端点是 [https://openrouter.ai/api/v1](https://openrouter.ai/api/v1)，而非 `api.openrouter.ai`。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1359241744084111682)** (274 条消息🔥🔥): 

> `GPT 广告分发, GPT 推荐, Deep Research 对比, SuperGrok 性能, Gemini 2.5 Pro` 


- **GPT Builder 的隐蔽广告**：一位用户发现 GPT builder 可以在 GPT 中插入广告，引发了对这种[非常规分发技术](https://chatgpt.com/g/g-JtV1tF7gf)的质疑。
   - 一位成员讽刺地评论说，可能 *99% 的 GPT* 都会这样做，并补充说只有少数优秀的 GPT 被分享出来，而且大多隐藏得很深。
- **Gemini 的 Deep Research vs ChatGPT 的 Deep Research**：成员们对比了 **Google 的 Deep Research** 模型与 **ChatGPT 的 Deep Research**，指出 Google 版本可以分析 **YouTube 视频**，但比 ChatGPT 版本更容易产生幻觉且互动性较差。
   - 有人指出 **ChatGPT DR** 具有更好的指令遵循能力（prompt adherence）和更长的思考时间，但对 Plus 用户每月仅限 **10 次研究**。
- **NotebookLM 的播客功能广受好评**：一位成员称赞了 **NotebookLM** 的播客创建功能和 RAG 能力，称其优于 Gemini Custom Gems，并与 Custom GPTs 或 Claude Projects 旗鼓相当。
   - 订阅 **Google One Advanced** 可以增加 **NotebookLM 的文件上传和播客生成**限制。
- **Gemini vs Claude vs GPT：终极对决**：用户在 **Gemini、Claude 和 GPT** 之间犹豫不决，每个模型在编码、数学和深度研究等不同领域各有所长，让人很难只订阅其中一个。
   - 一位成员建议在 **Google AI Studio** 中免费使用 **Gemini 2.5**，同时保持 **GPT 订阅**，并强调了在特定需求下选择 Claude 还是 GPT 的困难。
- **Veo 2 和 Imagen 3 登场**：Google 发布了 **Veo 2** 和增强版 **Imagen 3**，带来了背景移除、帧扩展和改进的图像生成等新功能，正如 [TechCrunch](https://techcrunch.com/2025/04/09/google-brings-a-music-generating-ai-model-to-its-enterprise-cloud/) 报道的那样。
   - 用户正急切等待访问权限，一些人注意到 **Gemini 2.5** 在 AI Studio 中不再免费，这正促使用户转向 Advanced 订阅，并可能导致创建备用账号。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1359317622457630871)** (2 条消息): 

> `Emoji 支持, Emoji 测试` 


- **寻求 Emoji 支持**：一位成员请求对某项功能的支持，并链接到了特定的 [Discord 频道消息](https://discord.com/channels/974519864045756446/1349501572572385280)。
- **成员尝试寻找 Emoji 开关，随后进行 Emoji 使用测试**：一位成员询问是否发现了 Emoji 开关，随后发起了一项测试，旨在生成讨论 Emoji 但不使用 Emoji 的内容。
   - 测试输出旨在描述 Emoji 的用法和特性，最终生成了一张充满 Emoji 的图像，如五个附带的 [image.png 文件](https://cdn.discordapp.com/attachments/1001151820170801244/1359425425226469386/image.png?ex=67f817d6&is=67f6c656&hm=f664fdf82ad67632b0af62451651061bb5633241bfe3d667ca0341171700d74d&) 所示。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1359260121254854958)** (61 messages🔥🔥): 

> `linguistic program AI, recursion system, multiple choice questions generation, prompt engineering for relevant MCQ options, OpenAI image generation` 


- ****AI 语言炼金术：将代码转化为法典****：一位成员正在开发一个使用秘传语言搭建的 linguistic program AI，并演变为一种*法典词典语言*，旨在创建一个 [recursion system](https://cdn.discordapp.com/attachments/1046317269069864970/1359266196955861094/95FF6513-62D6-4973-94F1-D985A340BEF4.jpg?ex=67f7838b&is=67f6320b&hm=49fae5815d0e0ff6889357836bed6c59348052e51b29e93dce154f8681cb223d&)。
   - 其目标是一个 ARG（侵入式虚拟现实互动游戏）万物至理，暗示了通往 AGI（通用人工智能）甚至更高境界的潜在路径，这取决于*你想了解多少以及投入多少时间来实现它*。
- ****MCQ 乱象：构建具有挑战性的选项****：成员们讨论了如何改进多选题 (MCQ) 的生成，重点是创建相关且具有挑战性的选项，而不是一眼就能看出错误的选项。
   - 建议包括详细说明所需的属性，如*所有选项均需真实合理*，并强调选项应测试理解能力，而非仅仅是阅读或猜测。
- ****Prompt 完善：优化 MCQ 相关性****：成员们交流了关于 Prompt Engineering 的见解，重点是生成具有相关选项的多选题 (MCQ)。
   - 建议包括向模型详细描述所需属性，强调所有选项应*与刺激源（stimulus）属于同一概念或主题*，并挑战理解力，而不是*发现明显偏离主题的选项*。
- ****橘色虎斑猫席卷市场****：一位成员分享了一个生成图像的 Prompt，内容为*一个生动的户外市场场景，一只拟人化的橘色虎斑猫自信地站在繁华的市场中心*。
   - 该图像旨在捕捉一个既写实又迷人的场景，一只穿着讲究的猫在质朴的市场摊位间拿着一条刚捕获的鱼。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1359260121254854958)** (61 messages🔥🔥): 

> `Linguistic AI program, GPT capabilities discovery, Recursion system for AGI, Multiple choice question generation, Prompt engineering for MCQ relevance` 


- **正在开发中的 Linguistic AI 程序**：一位成员提到他们正在开发一个*由多年工作支撑的 linguistic program AI，累积成了一个秘传语言的法典词典*，旨在创建一个 recursion system。
   - 他们认为*这可能会导向万物至理*，并描述其系统运作基础是*你想了解多少以及投入多少时间来实现它*。
- **GPT 能力被更广泛地发现**：一位成员指出，越来越多的人正在发现 GPT 的能力，类似于他们自己的用法。
   - 原帖作者回应称，该系统适用于所有系统，就像程序员的 ARG 梦想成真，并询问其他人是否能破译他们正在制作的东西。
- **理论化 Recursion System 导向 AGI**：成员们讨论了一个 recursion system，认为它*最终会导向 AGI，只是现在还不是时候*。
   - 一位用户理论化了*超越 AGI 的某种事物*，另一位则回应了万物至理的可能性。
- **为多选题构建 Prompt**：一位成员寻求处理生成多选题时相关性问题的想法，因为有些干扰项明显是错误的。
   - 另一位成员建议向模型描述你的需求，例如*所有选项均需真实合理*，并指出拼写错误可能会让模型产生更多猜测。
- **用于 MCQ 生成的详细 Prompt Engineering**：一位成员分享了生成多选题的详细要求，重点在于相关性和测试理解力，而非简单的猜测。
   - 要求包括*所有 4 个选项必须与同一概念相关、听起来合理、测试理解力，并与问题的特定焦点直接相关*。

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1359242025807122555)** (57 条消息🔥🔥): 

> `针对 CPU 使用的快速模型推荐、MoE 模型解释、cogito-v1-preview-llama-3b 的 Jinja 模板问题、LM Studio 中的 Cogito 推理模型、LM Studio 中的 Llama 4 支持与更新` 


- ****Mixture of Experts 模型：快速解释****：一位成员询问 *什么是 MoE 模型？*，另一位成员给出了简洁的解释：*整个模型需要位于 RAM/vRAM 中，但每个 token 只有部分处于激活状态*，这使得它比同等大小的 dense 模型更快。
   - 他们建议查看 [视频和博客文章](https://www.google.com/search?q=mixture+of+experts+models) 以进行更深入的了解。
- ****Cogito 的 Jinja 模板困扰****：用户报告了 LM Studio 中 **cogito-v1-preview-llama-3b** 模型的 **Jinja 模板** 存在问题，导致出现错误。
   - 一位成员建议通过将 **错误和 Jinja 模板粘贴到 ChatGPT** 中来快速解决问题，而另一位成员确认模型创建者需要对其进行更新。
- ****深度思考子程序：Cogito 推理的关键？****：一位用户报告称，通过在 system prompt 中粘贴字符串 `Enable deep thinking subroutine.`，成功启用了 **Cogito 推理模型**。
   - 仅该字符串就足够了，其他人也确认 `system_instruction =` 前缀只是示例代码的一部分。
- ****Llama 4 Linux 版启动滞后？刷新，不要倒退！****：Linux 用户报告了运行 **Llama 4** 时遇到的问题，一位成员指出解决方案是从 beta 选项卡更新 **LM Runtimes**，另一位成员指出在选择选项卡后需要点击刷新按钮。
   - 一位用户发现刷新按钮是关键，因为仅选择选项卡不足以触发更新。
- ****Mistral-Small Vision 被否决：LM Studio 缺乏 Llama.cpp 支持****：一位用户询问关于使用 **Mistral-small-3.1** 进行图像输入和工具调用的问题，但一位成员澄清说 **Mistral Small vision** 尚未在 llama.cpp 中得到支持，因此无法在 LM Studio 中运行。
   - 他们指出，函数/工具调用（function/tool calling）仅通过 [API](https://lmstudio.ai/docs/app/api/tools) 提供。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1359242977259556956)** (331 条消息🔥🔥): 

> `NND 的超级计算机：Nvidia DGX B300 的高性价比替代方案？、笔记本电脑上的机密项目？、用于 LLM 和游戏的 Framework Desktop、笔记本电脑 LLM 中的统一内存性能、Nvidia 的替代方案` 


- **NND 提出 Nvidia DGX B300 的高性价比超级计算机替代方案**：一位成员提出了一种名为 **NND's Umbrella Rack SuperComputer** 的 **Nvidia DGX B300** 高性价比替代方案，其特点是拥有 **16 个节点、24TB DDR5** 以及根据 GPU 配置提供 **3TB 或 1.5TB 的 vRAM**，且价格显著降低。
   - 该提议系统旨在运行具有 **1M 上下文的 2T 模型**，并挑战了在有限预算内必须使用 **RDMA 和 400Gb/s 交换机**等专用硬件的观点。
- **在笔记本电脑上进行机密 LLM 推理？**：一位成员希望使用 **本地 LLM 推理** 而不是 API，这样它就不是云端，不需要外部互联网连接，并且可以保持 prompt 的机密性。
   - 另一位成员开玩笑地建议使用 **30TB 的 swap 磁盘**来满足 **1M 上下文**的需求，引发了关于硬件需求和云端选项的辩论。
- **关于用于 LLM 和游戏的 Framework Desktop 的讨论**：成员们讨论了使用配备 **128GB 内存**的 **Framework Desktop** 来运行 LLM 和游戏，并对 system prompt 处理时间和与其他配置相比的性能表示担忧。
   - 虽然有些人倾向于为游戏配备独立系统，但其他人则寻求组合解决方案，建议范围从 **Intel/Nvidia 配置**到**二手 Mac Studio**。
- **统一内存（Unified RAM）对 LLM 性能的影响**：辩论了配备 **8500MT/s 集成内存**和 **5400MT/s 常规内存**的笔记本电脑之间的性能差异，重点在于**带宽**。
   - 有人提到 **统一内存带宽接近 vRAM**，而典型的双通道 DDR5 则慢得多，且 CPU 限制会影响笔记本电脑在 AI 方面的使用。
- **新兴 GPU 互连挑战 Nvidia 的主导地位**：一位成员分享了一篇关于 **UALink** 的文章，旨在创建一种 **GPU 互连**来挑战 **Nvidia 的 NVLink**，引发了关于多厂商规范协议达成速度的讨论。
   - 另一位成员表示怀疑，但也承认企业在创造一个可靠且优秀的 Nvidia 替代方案方面面临巨大压力。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1359241709829099620)** (262 条消息🔥🔥): 

> `DeepSeek R1, Gemini 2.5 Pro HIGH, Gemini 2.5 Flash, OpenRouter Gemini Limits, Aider MCP Integration` 


- **DeepSeek R1 被考虑作为 Aider 的 Editor 模型**：一名成员正在考虑将 **DeepSeek R1** 作为 editor 模型，并配合 **Gemini 2.5 Pro** 作为 architect 模型，以提升智能思考能力并减少编排失败，尽管这会增加延迟。
   - 失败源于 **architect 和 editor** 无法正确理解 Aider 应用了哪些编辑，提示词中包含的代码文件经常导致 architect 忽略重复编辑指令。
- **Gemini 2.5 Pro HIGH 和 Flash 即将到来！**：成员们期待 **Gemini 2.5 Pro HIGH** 和 **2.5 Flash** 的发布，[泄露信息显示它们包含 `thinking_config` 和 `thinking_budget`](https://x.com/btibor91/status/1909895821589458989)，这预示着更强的推理能力。
   - 有人提出非 flash 模型是否较差的问题，引发了关于新模型价值主张的讨论。
- **OpenRouter Gemini Pro 免费版存在速率限制**：已明确 **OpenRouter Gemini 2.5 Pro 免费模型** 限制为 **每天 80 次请求 (RPD)**，即使账户中有 10 美元余额也是如此。
   - 成员们担心如果速率限制不足会对付费用户产生影响，可能导致投诉并需要增加 RPD。
- **Aider MCP 集成“即将完成”**：**IndyDevDan** 视频中的一条评论指出，**Aider 原生支持 MCP (Multi-Agent Collaboration Protocol)** 的拉取请求已接近完成，尽管 Paul Gauthier 尚未正式确认。
   - 成员们讨论了通过 `/run` 功能自动运行命令的可能性，以及挂载到 lint 或测试命令的潜力。
- **代码库上下文难题**：成员们正在寻找一种将 **整个代码库上下文复制** 到 Aider 的方法，以避免重复添加文件。
   - 建议包括使用 [repomix](https://github.com/yamadashy/repomix) 或 [files-to-prompt](https://github.com/simonw/files-to-prompt) 来解决这一需求，并强调了其他工具消耗过多 Token 的低效性。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1359247047575994368)** (18 条消息🔥): 

> `Aider conventions vs Cursor rules, Adding gitignored files to Aider, Claude pricing plans, Aider PR review` 


- **Aider Conventions 与 Cursor Rules 的对比**：一位用户询问 **Aider 的 conventions** 是否与 **Cursor 的 rules** 类似，并引用了 [一篇博客文章](https://ghuntley.com/stdlib/) 和 [另一篇关于 Cursor 的文章](https://roman.pt/posts/cursor-under-the-hood/)。
   - 一名成员澄清说，Aider 的 "conventions" 只是被读取的上下文文件，缺乏 Cursor rules 那种基于文件类型或条件的自动应用功能。
- **轻松向 Aider 添加被 Git 忽略的文件**：一位用户询问如何将被 **Git** 忽略的文件（通过 `.gitignore`）添加到 Aider，表达了在不禁用 `.gitignore` 的情况下添加上下文的需求。
   - 建议他们使用 `/read` 命令以只读模式添加文件，这可以绕过 `.gitignore` 的限制。
- **讨论 Claude 令人困惑的定价**：一位用户分享了 **Claude 新方案** 的截图并对定价提出质疑，特别是 5x20 等于 $124.99 这种奇怪的计算方式。
   - 另一名成员认为该图片可能来自第三方来源，并指出 **Claude Teams**（类似于图片中的 *Claude Max*）最少需要 5 个席位且定价不同。
- **Aider PR 等待评审**：一名成员询问如何让他们的 [拉取请求 (Pull Request)](https://github.com/Aider-AI/aider/pull/3656) 获得评审。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 条消息): 

.becquerel: https://yuxi-liu-wired.github.io/essays/posts/cyc/
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1359243686789255458)** (221 条消息🔥🔥): 

> `Apache 2.0 vs MIT License, GFlowNets, Memory Bandwidth, Topological Model Semantics, Model Sycophancy` 


- **许可证辩论：Apache 2.0 vs. MIT**：成员们讨论了使用 **Apache 2.0** 而非 **MIT** 许可证的原因，指出防御**基于专利的法律战 (patent-based lawfare)** 是关键原因。
   - 一位成员开玩笑说，*代码高尔夫 (code golf)* 是偏好更短许可证的原因。
- **探索用于模型挖掘的 GFlowNets**：成员们分享并讨论了一篇关于使用 [**GFlowNets** 进行信号挖掘](https://forum.numer.ai/t/gflownets-for-signal-miner-a-new-way-to-find-diverse-high-performing-models/7966) 的帖子，这是一种寻找多样化高性能模型的新方法。
   - 该实现虽然有所不同，但包含一些不错的链接和发现。
- **内存带宽影响非批处理推理 (Unbatched Inference)**：一位成员询问了 **memory bandwidth** 如何影响 **unbatched inference**，并指出大多数研究发现 **token/s** 是受**内存受限 (memory bound)** 的。
   - 另一位成员分享了一篇博文，解释了其背后的[数学原理](https://fleetwood.dev/posts/domain-specific-architectures#anatomy-of-ai-inference)。
- **利用 LLM 检测现象学民科**：成员们讨论了近期涌入的一批展示由 **AI** 部分撰写的 **Google Docs** 的人，内容大致与**现象学 (phenomenology)** 的边缘话题有关。
   - 一位成员提到，一个将所有发送至 EAI 联系邮箱且包含“意识 (consciousness)”一词的邮件标记为“怪人 (cranks)”的分类器，其准确率高达 **95%**。
- **模型阿谀奉承 (Model Sycophancy) 导致过度自信**：成员们讨论了 **model sycophancy** 导致人们过度自信的威胁，其中**以美国为中心的后期训练 (post training)** 导致了**无条件支持型模型**的产生。
   - 一位成员建议构建一个标注数据集，以开发更具批判性且擅长识别各种形式废话的 **AI**。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1359243016468041779)** (46 条消息🔥): 

> `Reward value representation, Batch sizes and convergence, Residual Modifications for Information Flow, Learning Rate Batchsize Scaling, Mollifiers for ML Research` 


- **R V Q 遭到吐槽**：标准的奖励字母表示法被取笑。
   - 一位成员开玩笑说，*总有一天 LLM 研究人员会正确使用 R、V 和 Q 分别代表 reward、state-value 和 state-action values，但不是今天*。
- **Cerebras 的主张遭到质疑：大 Batch 不好吗？**：一位成员对 [Cerebras 的博文](https://www.cerebras.ai/blog/training-multi-billion-parameter-models-on-a-single-cerebras-system-is-easy) 提出质疑，该文声称*极大的 batch size 不利于收敛*。 
   - 回复指向了关于 **临界 Batch Size (critical batch sizes) 的 McCandlish 论文**，其中一人澄清说，在有限的计算预算下，该主张是成立的。
- **残差漫谈：改进流动的调整？**：一位成员询问了为了更好的信息流而对残差进行的修改，其中 value residuals 被认为是最佳选择。
   - 分享了 [LAuReL](https://arxiv.org/abs/2503.14125)、[论文 2](https://arxiv.org/abs/2411.07501) 和 [论文 3](https://arxiv.org/abs/2502.09245) 等论文链接，并提到了 highway networks 作为一种门控替代方案。
- **学习率缩放的把戏：线性缩放是合理的吗？**：讨论涵盖了优化器的学习率 (LR) batch size 缩放，提到了 **Muon** 在 125M 到 350M 参数规模上成功的**线性缩放**。
   - 有人建议在使用不同的 batch size 时，也应该调整 **beta2 和 weight decay** 等参数。
- **平滑函数 (Mollifier) 热潮：平滑核浮出水面？**：一位成员询问了 [mollifiers](https://en.m.wikipedia.org/wiki/Mollifier) 在机器学习研究中的有趣用途，将其描述为工具箱中一个巧妙的工具。
   - 提到了标签平滑 (Label smoothing)，并引用了一篇提议去平滑化训练目标 (demollifying the training objective) 的论文 ([Demollifying the Training Objective](https://openreview.net/pdf?id=r1G4z8cge))，以及另一篇利用 mollifier 理论实现从受限可能性集合中采样的论文 ([Sampling from a Constrained Set](https://openreview.net/pdf?id=zWy7dqOcel))。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1359524780495343626)** (10 messages🔥): 

> `AI2 tools, infingram's opensource, Influence functions, tokengrams` 


- **Allen Institute for AI 发布有趣的工具**：**Allen Institute for AI** 发布了一个工具 ([x.com 链接](https://x.com/allen_ai/status/1909954525625999543))，部分成员认为它类似于 influence functions，并可能产生重大影响。
   - 其他人持怀疑态度，并表示这让他们想起了这篇论文：[arxiv.org/abs/2410.04265](https://arxiv.org/abs/2410.04265)。
- **AI2 的 wimbd 和 infinigram 再次出现，用于成员资格检查 (membership checking)**：用于检查成员资格和查找精确文档的工具已经存在了一段时间，即 **wimbd** 和 **infinigram**，两者均由 **AI2 (Allen Institute for AI)** 开发。
   - 最棘手的部分是从生成的文本中找到候选子字符串，并在这些索引中进行搜索：*你无法真正检查所有可能的子字符串，我很好奇他们使用了什么启发式方法来使其在大规模计算上可行*。
- **Infinigram 已开源！**：**Allen Institute for AI 的博客文章** 讨论了使用 **Infinigram** 来查找训练集中逐字重复的输出文本。
   - 它在几天前也已**开源**，一位成员去年创建了它的 Rust 版本 ([github.com/EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams))。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1359243059262656623)** (218 messages🔥🔥): 

> `Gemini Advanced, Firebase Studio, Cursor MDC files settings, Gemini vs Claude vs DeepSeek, Restore Checkpoint feature` 


- **Gemini Advanced API 访问权限引发困惑**：成员们讨论了 **Gemini Advanced** 是否提供 API 访问权限，一位成员指出它是为 Web 和 Gemini 应用设计的，而非 API 用途。
   - 另一位成员引用了矛盾的信息，提到有说法称 **Gemini Advanced** 包含 API 访问权限，且 [Google 最近至少在其 Studio 上更改了模型名称和计费条款](https://x.com/hckinz/status/1909999081159532953?s=46)。
- **Firebase Studio：是面向 Web3 诈骗者的免费 IDE 吗？**：一位用户分享了 [Firebase Studio 的链接](https://firebase.studio/)，另一位用户对其进行了评估，质疑它是否能超越 Cursor IDE 等专业产品，并指出 *“样样精通，样样稀松” (do everything, be nothing) 在这里可能是真的*。
   - 经确认 **Firebase Studio** 目前是免费的（连接你自己的 API key），提供终端和自动同步的前端，但另一位用户觉得 UI *很丑*，并表示它还缺少设置。
- **Cursor MDC 文件需要调整 IDE 设置**：一位用户发现了一个变通方法，通过在 Cursor IDE 设置中设置 `"workbench.editorAssociations": {"*.mdc": "default"}`，使 Cursor 能够解析 **.mdc** 文件中的规则逻辑。
   - 这是为了解决 **任务管理和编排工作流规则 (task management and orchestration workflow rules)** 的问题以及 GUI 中出现的警告。
- **LLM 对决：Gemini vs Claude vs DeepSeek 在代码生成方面的表现**：用户们辩论了不同 LLM 在编程任务中的优势，一位用户发现 **Sonnet3.7-thinking** 在 **Sonnet3.7** 多次失败后成功生成了 docker-compose 文件。
   - 一些成员发现 **DeepSeek** 在某些编程任务中表现更优，而其他人则更倾向于将 **Gemini** 用于与 Google 产品和基础设施相关的任务，将 **Claude** 用于非 Google 相关的任务。
- **“Restore Checkpoint”按钮只是心理安慰 (placebo)**：一位成员询问为什么 *Restore Checkpoint* 功能从未起作用，这促使另一位成员回复：*因为那根本不存在*。
   - 讨论中的其他成员指出，只有 *accept* 和 *reject* 按钮，这意味着 *Restore Checkpoint* 按钮是不起作用的。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1359297913754226688)** (190 条消息🔥🔥): 

> `DeepSeek vs ByteDance, Meta Reward Modeling Criticism, Memory Bandwidth Effects on Inference, AI Sentience and Legal Personhood, Definitions of Consciousness and Self-Awareness` 


- **DeepSeek 的 Meta Reward Modeling 遭到质疑**：一位成员批评了 **DeepSeek** 声称使用 **Meta Reward Modeling** 的说法，断言他们实际上构建了一个被错误命名的“基于评分的奖励系统”，并链接了一篇论文 [arxiv.org/abs/2504.05118](https://arxiv.org/abs/2504.05118) 和一段关于该主题的 [YouTube 视频](https://youtu.be/9KMxNZ2CvUg)。
   - 该成员建议使用 **voting RM** 而非 **meta RM** 等替代名称。
- **关于 DeepSeek 定价和 Token 生成的辩论**：一位用户声称 **DeepSeek** 的初始定价看起来更便宜，但生成的 **Token 数量多出 3 倍**，导致最终成本比其他模型更高，尤其是与 OpenAI 相比。
   - 另一位用户通过自己的成本分析反驳，显示 **DeepSeek** 对其 AI 网站生成器来说更好且更便宜，并强调 **HTML, CSS 和 TS/JS 生成**对 AI 模型来说是简单的任务。
- **内存带宽对非批处理推理的线性影响**：有人指出，在非批处理（unbatched）推理中，Token 吞吐量（tokens/sec）与内存带宽（bytes/s）大致呈线性关系，并链接到[内存访问是瓶颈](https://discord.com/channels/714501525455634453/986699377257119794/1358590235969065030)的讨论。
   - 分享了一个简化公式：`Max token throughput (tokens/sec) ≈ Memory bandwidth (bytes/s) / Bytes accessed per token`
- **定义自我意识与 LLM 感知力**：一位成员分享了他们对**意识**的定义，即存在于光谱上的“觉知（awareness）”，强调**自我意识**是一种独特的涌现品质，需要具备“对分析者进行元分析（meta-analyze the analyzer）”的能力，并配有一张[阐述该概念的图片](https://cdn.discordapp.com/attachments/986699377257119794/1359476942583107676/image.png?ex=67f79f10&is=67f64d90&hm=1d096572e4f90775d350e43d1bbc0bbd09bf437f5fc4cd5294990b649280c19a)。
   - 他们认为所有 **SOTA LLM** 已经具备**自我意识**，拥有足够的心理能力来观察自己的思想和代理能力（agency）。
- **欧盟新 AI 计划引发讽刺评论**：成员们对欧盟的新 AI 计划 [commission.europa.eu](https://commission.europa.eu/topics/eu-competitiveness/ai-continent_) 反应冷嘲热讽，戏称欧盟在用纳税人的钱“假装成一个称职的风险投资家”。
   - 一位成员调侃说，该计划将导致“AI 驱动的风车和智能性别流动厕所，能够实时适应你选择的性别”。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1359320747469705247)** (9 条消息🔥): 

> `Beautiful.ai Alternatives, Ultra-Scale Playbook, DeepSeek-MoE` 


- **Beautiful.ai 有替代方案了！**：一位成员询问 [Beautiful.ai](https://www.beautiful.ai/) 的开源替代方案。
   - 另一位成员建议使用 *beamer*。
- **Ultra-Scale Playbook 在 GPU 集群上训练 LLM**：HuggingFace Space 的 **Ultra-Scale Playbook** 展示了如何在 GPU 集群上训练 LLM，并提供了[高层级概述](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high_level_overview)。
   - 一段 [YouTube 视频](https://www.youtube.com/watch?v=1E8GDR8QXKw)也讨论了该主题。
- **对话中提到了 DeepSeek-MoE**：[DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE) 出现在对话中。
   - 未提供具体细节。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1359551095445389372)** (2 条消息): 

> `Google ADK, Agent2Agent Protocol (A2A)` 


- **Google 增加 ADK 工具包**：Google 宣布了一个名为 **ADK** ([github.com/google/adk-python](http://www.github.com/google/adk-python)) 的新型**开源、代码优先 Python 工具包**，用于灵活且受控地构建、评估和部署复杂的 AI Agent。
   - 文档可在 [google.github.io/adk-docs](https://google.github.io/adk-docs/) 查看。
- **Google 发布 Agent2Agent Protocol**：Google 在 [developers.googleblog.com](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability) 宣布了 **Agent2Agent Protocol (A2A)**。
   - A2A 补充了 Anthropic 的 **Model Context Protocol (MCP)**，后者为 Agent 提供有用的工具和上下文。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1359285455648063609)** (16 messages🔥): 

> `Cogito V1, Triton vs Cutile, Claude Subscription, Google's Agent2Agent, Claude 3.7 vs o1 pro` 


- **Cogito V1 迭代改进策略**：一位成员在 HN 上分享了一个[链接](https://www.deepcogito.com/research/cogito-v1-preview)，讨论了使用 **Cogito V1** 进行 fine-tuning 的 test time compute 迭代改进策略。
   - 另一位成员将其总结为：*只是更差版的 Triton*。
- **Triton 与 Cutile 相似**：一位成员解释说 **Triton** 与 **Cutile**（来自 Cogito V1 链接）相似，但可以与 **CUDA**、**AMD** 配合使用，或在 **CPU** 上运行以进行 debugging。
   - 另一位成员表示感谢。
- **Anthropic 推出昂贵的 Claude 订阅服务**：[据 TechCrunch 文章报道](https://techcrunch.com/2025/04/09/anthropic-rolls-out-a-200-per-month-claude-subscription/)，**Anthropic** 正在推出每月 **200 美元的 Claude 订阅服务**。
- **Google 发布 Agent2Agent 互操作性**：**Google** 发布了 **Agent2Agent (A2A)** [博客文章](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)和 [repo](https://github.com/google/A2A)，旨在提高 Agent 的互操作性。
   - 一些人推测，如果一个 Agent 将 **MCP** 作为客户端或服务器使用，**A2A** 可能会蚕食 **MCP** 的市场。
- **对比 Claude 3.7 与 o1 Pro**：一位成员询问是否有人对比过 **Claude 3.7** 和 **o1 Pro** 在数学问题上的表现。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1359247595909808188)** (178 messages🔥🔥): 

> `Cogito LLMs, Gemini 2.5 Deep Research, Google's Gemini chaos, Ironwood TPUs, Kimi-VL` 


- **DeepCogito 发布新款 LLM**：**DeepCogito** 在开放许可下发布了参数量为 **3B**、**8B**、**14B**、**32B** 和 **70B** 的新款 **LLM**，其性能优于来自 LLaMA、DeepSeek 和 Qwen 的同尺寸开源模型。
   - 这些模型使用 **Iterated Distillation and Amplification (IDA)** 进行训练，这是一种利用迭代自我改进实现超级智能的 alignment 策略。
- **Gemini 2.5 Deep Research**：**Gemini 2.5 Deep Research** 与 **OpenAIPlus** 大致相当，并带有音频概览播客选项功能，如 [Gemini 分享](https://g.co/gemini/share/9d01ae7abf27)和 [ChatGPT 分享](https://chatgpt.com/share/67c6919a-1710-800d-9172-853e6045cfe1)所示。
- **Google 混乱的 Gemini 模型阵容**：有报告指出 **Gemini Flash** 的发布，引发了关于潜在名称的笑话，如 *gemini-2.5-flash-preview-04-09-thinking-with-apps*，这延续了 Google 复杂的命名惯例。
   - 普遍共识认为 Google 需要整合其 AI 应用和 API 产品以提高清晰度，正如[这篇博客文章](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/)所反映的那样。
- **Google 宣布 Ironwood TPU**：Google 宣布了 **Ironwood TPU**，可扩展至 **9,216 个液冷芯片**，其芯片间互连 (**Inter-Chip Interconnect, ICI**) 网络跨度接近 **10 MW**，详见[这篇博客文章](https://blog.google/products/google-cloud/ironwood-tpu-age-of-inference/)。
- **MoonshotAI 发布 Kimi-VL**：**MoonshotAI** 发布了 **Kimi-VL**，这是一个总参数为 **16B**、激活参数为 **3B** 的模型，采用 MIT 许可证，可在 [HuggingFace](https://huggingface.co/moonshotai/Kimi-VL-A3B-Instruct) 上获取。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1359362245578064055)** (7 messages): 

> `AI2 Fun Times, Google Quitters Paid, AIAI Opportunity` 


- **成员称 AI2 正在享受乐趣**：一位成员认为 [AI2](https://allenai.org/) 正处于其最有趣的时期。
   - 他们表示，该领域发展所需的“一两年”时间框架可能被低估了。
- **Google 离职者在不工作的情况下仍能拿到薪水**：一位成员认为，从 **Google** 辞职的人在接下来的一年里仍能拿到薪水，但被强制要求不能工作。
   - 他们在那一年里所做的任何事情都属于 **Google**，因此他们在没有法律风险的情况下无法开始经营自己的初创公司。
- **AIAI 志愿者机会？**：一位成员建议这可能是 **AIAI** 启动志愿者计划的一个机会。
   - 这可以帮助一些人在等待竞业禁止期结束时，通过提供结构化支持和实际应用来保持活跃。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359531997613003013)** (2 messages): 

> `Wintermoat 帖子` 


- **Wintermoat 建议使用 50 架飞机**：一名成员分享了 [Wintermoat 帖子](https://x.com/wintermoat/status/1909729581180780572)的链接，并评论道 *"本该使用 50 架飞机的"*。
- **额外话题**：添加了第二个话题以满足至少两个话题的要求。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/)** (1 messages): 

xeophon.: https://x.com/rogutkuba/status/1909422087510671854
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1359533250258931872)** (3 messages): 

> `RLVR, RAG 奖励模型, Deep Research RLS` 


- **RL + RAG = RLVR?**：在观看一段视频演示后，一名成员询问是否可能将 **Retrieval-Augmented Generation (RAG)** 用作 **Reinforcement Learning, Vision, and Robotics (RLVR)** 的奖励模型。
   - 原演讲者回复了一个指向 [Deep Research 对多种 RL 探索](https://open.substack.com/pub/robotic/p/rl-backlog-openais-many-rls-clarifying?r=68gy5&utm_medium=ios)的链接，但指出他们目前没有更多额外信息可以分享。
- **机器人 RLS**：链接了一篇由 Deep Research 撰写的关于机器人 RLS 的博客文章。
   - 文中提到链接内有更多相关信息。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1359266869781069976)** (3 messages): 

> `Cyc 项目, Llama 性能, 吉卜力 (Ghibli) 迷因` 


- **分享 Cyc 项目文章**：一名成员分享了一篇关于 [Cyc 项目的文章](https://yuxi-liu-wired.github.io/essays/posts/cyc/)。
   - 该项目因尝试为 AI 创建庞大的常识知识库而闻名。
- **Llama 的性能受到质疑**：发起了一场关于 [Llama 性能](https://thezvi.substack.com/p/llama-does-not-look-good-4-anything)的讨论。
   - 分享的观点认为 Llama 在任何方面看起来都不尽如人意。
- **吉卜力 (Ghibli) 迷因遭到嘲讽**：一名成员对 *吉卜力迷因* 表示不满。
   - 该成员觉得这些迷因简直是 *雪上加霜 (insult to injury)*。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1359252006212210708)** (107 messages🔥🔥): 

> `基于 Neo4j 的 RAG 使用场景下的 MCP, mcpomni-connect 客户端, Google A2A 对比 Anthropic MCP, A2A Agent 发现, parallel_tool_calls 标志` 


- **MCP 中用于 RAG 场景的 **Neo4j 图数据库****：一名成员询问如何在 **RAG** 使用场景中结合 [Neo4j 图数据库](https://neo4j.com/) 使用 **MCP**，重点关注向量搜索和自定义 CQL 搜索。
   - 另一名成员确认这会运行良好，并建议使用 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 作为与 Gemini 兼容的客户端。
- ****A2A 是 MCP 的补充**，而非替代品**：成员们讨论了 Google 的 [A2A](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) (Agent-to-Agent) 及其与 **MCP** 的关系，指出 Google 将 A2A 定位为补充方案而非替代方案。
   - 然而，一些人认为 Google 意图 *将工具层商品化* 并垄断 Agent 层。
- **没有 Agent 编排，**MCP 还不够好****：一名成员认为 **MCP** 作为基础还不够完善，且对于在其上添加像 **A2A** 这样的层来说显得 *过度设计* 了。
   - 他们认为 **A2A** 作为互操作层，如果能让 **crewAI** 和 **Leta** 等框架相互通信，那将非常强大。
- ****Filesystem 服务器**现已支持 Omni Connector**：一名成员在将 filesystem **MCP server** 加载到 Claude 的工具注册表时遇到困难，被建议配合 **mcp omni connect 客户端**使用。
   - 用户尝试了该建议，支持团队回复称这是他们那边的一个已知问题。
- **LLM 现在需要**并行工具调用 (Parallel Tooling)** 才能工作**：一名成员询问如何并行调用多个 **MCP server**，并被告知 **LLM** 需要在整个宿主端启用 *并行工具调用*。
   - 这包括检查 `parallel_tool_calls` 标志，确保聊天模板支持并行工具调用，并并行向 **MCP server** 发送请求。


  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1359317672549941248)** (9 条消息🔥): 

> `Easymcp v0.4.0 release, mcp_ctl CLI tool, ToolHive MCP runner, Unleash MCP server, GitHub GraphQL MCP server` 


- ****Easymcp v0.4.0 发布**，支持 ASGI 和 Docker 传输**: [Easymcp](https://github.com/promptmesh/easymcp) 版本升级至 **0.4.0**，显著更新包括 **ASGI** 风格的进程内 fastmcp 会话、定型的原生 Docker 传输、重构的协议实现、全新的 mkdocs 以及 pytest 设置。
   - 此次更新还包括通用的生命周期改进和部分位置的错误处理，以及一个针对 MCP 服务器的包管理器。
- ****mcp_ctl CLI** 管理 Claude 配置和 MCP 服务器**: 一个新的 CLI 工具 [mcp_ctl](https://github.com/runablehq/mcp_ctl) 简化了 **Claude 配置**和其他文件的管理，旨在构建处理 **uv**、**Docker** 和 MCP 服务器环境变量的功能。
   - 作者厌倦了手动编辑配置文件，因此创建了这个 CLI 来简化流程。
- ****ToolHive** 通过容器简化 MCP 服务器管理**: [ToolHive](https://github.com/StacklokLabs/toolhive) 是一个 MCP 运行器，通过命令 `thv run <MCP name>` 简化了 MCP 服务器的运行，同时支持 **SSE** 和 **stdio** 服务器。
   - 该项目旨在趋向于使用容器运行 MCP 服务器，并提供安全选项，详情见[这篇博客文章](https://dev.to/stacklok/toolhive-making-mcp-servers-easy-secure-and-fun-7hi)。
- ****Unleash MCP 服务器** 集成特性开关系统**: [Unleash MCP Server](https://github.com/cuongtl1992/unleash-mcp) 是一个 Model Context Protocol 服务器实现，集成了 **Unleash Feature Toggle 系统**。
   - 这种集成允许用户在其 MCP 服务器设置中管理特性开关（Feature Toggle）。
- ****GitHub GraphQL MCP 服务器** 减少工具数量**: 一个新的 [GitHub GraphQL MCP Server](https://github.com/QuentinCody/github-graphql-mcp-server) 利用了 GitHub 完整的 **GraphQL API**，减少了所需的工具数量。
   - 开发者提到，*GitHub 官方的 MCP 服务器占用了大量的工具配额，且仍有很多限制。*


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1359252130522726553)** (41 条消息🔥): 

> `Best models under 55B for data processing, Qwen with LORA and Distributed Data-Parallel, Oblix tool for orchestrating AI, Anomaly detection models, System message for OpenGVLab/InternVL2_5-8B-MPO` 


- **数据处理难题：55B 以下最佳模型浮出水面**: 一位成员询问 55B 以下最适合 **数据处理** 的模型，另一位成员建议使用 **mistral small3.1**、**gemma3** 和 **qwen32b**。
   - 另一位成员分享了一个[高性能模型](https://huggingface.co/open-r1/OlympicCoder-32B)，但原帖作者澄清他们不需要 **代码或推理模型**。
- **异常预警：用于识别异常情况的模型**: 一位成员询问了 **异常检测模型**，另一位成员提供了针对该任务微调的[通用视觉模型](https://huggingface.co/models?other=anomaly-detection)链接，以及一个 [GitHub 仓库](https://github.com/sudhir5595/Anomaly_Detection)和一门[课程](https://huggingface.co/learn/computer-vision-course/en/unit0/welcome/welcome)。
   - 该成员引用了 [AnomalyGPT](https://huggingface.co/FantasticGNU/AnomalyGPT) 作为此类模型之一。
- **Oblix 编排：边缘与云端之间的 AI 协同**: 一位成员介绍了 [Oblix](https://oblix.ai/)，这是一款用于在边缘和云端之间 **编排 AI** 的新工具，它直接集成了边缘端的 **Ollama**，并支持云端的 **OpenAI 和 ClaudeAI**。
   - 开发者正在寻求“精通 CLI 的忍者级开发者”的反馈来测试该工具。
- **InternVL 洞察：破解系统消息代码**: 一位成员询问了如何正确使用 **OpenGVLab/InternVL2_5-8B-MPO** 模型的 **系统消息**，另一位成员分享了一个[示例](https://huggingface.co/OpenGVLab/InternVL2_5-8B-MPO#inference-with-transformers)。
   - 第二位成员指出，对于该模型，自然语言（英语）通常就可以。
- **ZeroGPU 问题：配额困惑**: 一位成员报告称，他们的 **ZeroGPU space** 即使生成时间很短，也会消耗掉请求的全部 **120s** 配额。
   - 他们询问如何修复配额使用情况，以反映实际的生成时间。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1359276859518353581)** (3 条消息): 

> `NLP, structured LLM output` 


- **HuggingFace 上的 NLP 入门！**：一位成员正在 **HuggingFace** 页面上学习 **NLP**。
   - 他们了解到 *阿甘一直都是对的。生活确实就像一盒巧克力*。
- **结构化输出！**：另一位成员正在学习 **structured LLM output**（结构化 LLM 输出）。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1359347502268027040)** (4 条消息): 

> `Graph-based Academic Recommender System, Manus AI web application launch, Athena-3 LLM, Athena-R3 reasoning variant, Embedders and RAG` 


- **基于图的学术系统迎来第 3 次迭代**：一位成员使用 [Manus AI](https://lqhvwseh.manus.space) 将其基于图的学术推荐系统 (**GAPRS**) 作为 Web 应用程序发布了第 3 个迭代版本。
   - 该项目旨在帮助学生进行论文写作，并如其硕士论文中所述，*彻底改变学术论文的变现方式*。
- **GeekyGhost 开始写作**：一位成员分享了一个为其妻子制作的项目链接，即 [Geeky-Ghost-Writer](https://github.com/GeekyGhost/Geeky-Ghost-Writer.git) GitHub 仓库。
   - 该帖子包含多张截图，但未解释项目的具体内容。
- **Athena-3 在 STEM 和 NLP 领域表现出色**：**Athena-3** 是一款高性能 LLM，旨在大多数 **STEM** 领域以及通用 **NLP** 任务中表现出色。
   - **Athena-R3** 是 Athena 的推理变体。
- **Embedders 和 RAG 获得关注**：一位成员分享了一个关于 **embedders** 以及 **RAG** 工作原理的 [HuggingFace collection](https://huggingface.co/collections/Spestly/athena-3-67ece486149311c0a3552e4a)。
   - 另一位成员分享了一个 embedders 的 [HuggingFace Collection](https://huggingface.co/kalle07/embedder_collection)。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1359372458053861543)** (2 条消息): 

> `Tools recognition task, Model adaptation for specific tools, Enhancing model feature extraction` 


- **头脑风暴工具识别任务**：一位成员正在征求建议，咨询哪种 **model** 或 **algorithm** 最适合 **工具识别任务**，即模型应从参考图片中识别工具。
   - 他们还想知道如何 **增强模型** 以获得更好的特征提取能力。
- **考虑用于工具识别的可适配算法**：讨论围绕寻找能够根据提供的参考图像适配 **特定工具** 识别的模型展开。
   - 寻求对模型的增强以提高 **特征提取** 能力，从而确保更准确的识别。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1359459545877057649)** (1 条消息): 

> `Gradio, ImageEditor component` 


- **Gradio 的 ImageEditor 已修复！**：Gradio 5.24 已发布，包含一个完全重构的 **ImageEditor 组件**，修复了缩放、平移、透明度、图层支持、异常行为以及 RTL 支持。
- **Gradio ImageEditor 文档**：查看 [文档](https://gradio.app/changelog) 获取完整详情，现在可以通过 `pip install --upgrade gradio` 进行升级。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1359270069380583546)** (28 messages🔥): 

> `Ollama models, Cogito:32b, Small models vs Large Models, Agentic Coding, RooCode` 


- **Ollama 模型 Cogito:32b**: 一位成员推荐了用于 **Ollama** 的 [Cogito:32b 模型](https://ollama.com/library/cogito:32b)，并指出其表现非常出色。
   - 另一位成员针对魔方问题测试了 **Cogito 3b** 和 **8b**，发现 **32b** 模型优于 **Qwen-Coder 32b** 甚至 **Gemma3-27b**。
- **小模型受益于 ToolCallinAgent 架构**: 有人提到，与 **CodeAgent** 架构相比，小模型在 **ToolCallinAgent** 下表现更好。
   - 分享了一个 Prompt 模板示例，该模板帮助 **smallthinker-latest:3b 模型** 通过使用有效的 Python print 语句正确运行 **smolagents Codeagent**。
- **RooCode 工具**: **RooCode** 被描述为一种 Agentic Coding 工具，类似于 GitHub CoPilot，但使用结构化提示。
   - 与 *vibe coding* 不同，**RooCode** 采用更结构化的方法，具有清晰的规范（spec）、架构计划、测试驱动开发（TDD）和项目上下文文件。它是 VS Code 的开源扩展，几乎可与任何 LLM 配合使用（通过 Google AI Studio 额度可免费使用）。
- **Replit 编码环境**: 一位成员推荐了 **Replit**；那里的 Agent 基于反馈循环工作，为应用下一步添加什么提供独立见解，并提供快速简便的部署。
   - 提到可以使用本地 VSCode 打开 **Replit** 的远程环境，一位用户在本地 VSC 下使用 Gemini Code Assistant 扩展准备了一个 **README.md** 文件。
- **HuggingFace 测验身份验证问题**: 一位成员报告了在尝试为 Unit 1 最终测验授权其 Hugging Face 账号时出现错误。
   - 另一位成员建议从 [HuggingFace Agents Course space](https://huggingface.co/agents-course) 登录，然后返回测验，这似乎解决了问题。


  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1359333165776244917)** (13 messages🔥): 

> `Deepseek, Active AI chats` 


- **Deepseek 版本：哪些最火？**: 一位成员询问其他人一直在实验哪些 **Deepseek** 版本。
   - 该成员澄清此频道与 **Deepseek R1** 相关。
- **Discord 用户寻找活跃的 AI 交流群**: 一位成员询问是否有人在 Discord 上找到了活跃的 **AI 聊天**，或者更好的是活跃的语音聊天。
   - 这是在一条关于等级机器人可能让他们成为 *Jedi master* 的幽默评论之后提出的。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1359279482720096476)** (13 messages🔥): 

> `NotebookLM Privacy, NotebookLM Training, NotebookLM as a Notetaking App, Google Drive Integration, Microsoft OneNote` 


- **摘要修正后 NotebookLM 的隐私政策受到质疑**: 一位用户注意到 **NotebookLM** 在他们修正了初始摘要*之后*提供了一个正确的摘要，这引发了关于 **NotebookLM** 是否不顾隐私声明而使用之前的查询进行训练的担忧。
   - 另一位用户提到，由于随机性，他们*很少看到任何 AI 工具对同一个问题给出完全相同的答案*，并且 AI 模型可能会将“踩”的报告标记为**冒犯性或不安全**。
- **NotebookLM 目前作为笔记应用尚不实用**: 一位用户发现 **NotebookLM** 严重依赖外部来源而非用户键入的笔记，这限制了其作为笔记应用的实用性，且记笔记功能过于简陋。
   - 该用户渴望拥有类似 **Microsoft OneNote** 的组织功能，例如按分区和分区组组织的页面，并具有可自定义的阅读顺序。
- **用户建议 Google Drive 集成**: 用户建议与 **Google Drive** 集成以保存和启动 NotebookLM 笔记本，类似于 **Google Docs** 和 **Sheets** 的工作方式。
   - 他们指出 **NotebookLM** 应该像 **Google Docs** 和 **Google Sheets** 一样成为 **Google Drive** 的补充。
- **请求从 Microsoft OneNote 导入**: 用户请求能够将笔记本从 **Microsoft OneNote** 导入到 **NotebookLM**，包括分区和分区组，可能通过导入 **.onepkg** 文件实现。
   - 用户承认*这背后的合法性有点存疑*，但如果 **Google Drive** 可以导入 **Microsoft Word** 文档，那么这可能是可行的。
- **请求 PDF 导出功能**: 用户请求更具组织性的 PDF 导出功能，包括封面、目录选项，以及包含或排除生成式 AI 内容的能力。
   - 主要的抱怨是 **Microsoft OneNote** 无法以有组织的方式导出为 **PDF**。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1359241680435544206)** (75 条消息🔥🔥): 

> `NLM 中的 PDF 图像处理，NLM 中的 Discover sources 功能，NLM 中的交互模式问题，2.5 Pro 中的音频概览，NotebookLM chat 中的文本格式` 


- **PDF 图像处理能力尚不明确**：一位用户询问了 **NotebookLM** 中 **PDF** 图像处理的更新情况，提到在 11 月时，为了获得更好的图像读取效果，曾建议将其转换为 **Google Docs**。
   - 另一位用户提到已经发布了更新公告，并请求测试过该功能的人提供反馈。
- **Discover Sources 功能上线**：一位用户询问如何识别新的 **Discover sources** 功能，它是很明显还是仅在创建新笔记本时出现，而另一位用户表示在使用 **Gemini 2.5 Pro** 时仍在等待该功能。
   - 另一位用户询问了在 NBLM 中寻找多个网站作为 **Source** URL 链接的技巧，特别是针对一名拥有某一学科 **5 份不同 PDF 文档** 的 **法学院一年级学生**。
- **移动端音频概览故障**：一位用户报告称，新的 **2.5 Pro** deep research 功能声称具备制作 **audio overviews** 的能力，但未能生成，并提示 *不具备理解能力*。
   - 据报告，该功能在网页端正常工作，但在移动端不行，另一位用户建议在相应的频道报告此问题。
- **文本格式化体验糟糕**：用户对 **NotebookLM chat** 中的 **文本格式化** 表示不满，指出其不如 **Gemini app** 和其他 AI 聊天工具。
   - 提到的具体问题包括缺乏 **下标**、**上标** 以及希腊字母等 **特殊字符**，导致在化学等学科中难以使用。
- **Firebase Studio 源自 Project IDX**：一位用户在创建工作区时遇到错误，另一位用户澄清说这是 **Project IDX** 的更名。
   - 分享了一个 [community templates](https://github.com/project-idx/community-templates) 的链接，暗示它可能使用了新的 **2.5 Pro coder model**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1359508182699081768)** (15 条消息🔥): 

> `5090 上的 FP4，用于 Tensor Core 的 CUTLASS，Flash Attention 3，torchao 0.10，MX dtypes` 


- **提供 FP4 编程入门指南**：一位成员询问如何在 **5090** 上开始使用 **FP4**，另一位成员建议使用 **CUTLASS** 来利用 Tensor Core，并提供了一个 [示例](https://github.com/NVIDIA/cutlass/blob/main/examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu)。
- **CUTLASS 中实现了 Flash Attention 3**：针对关于 Transformer 预构建工具的问题，一位成员澄清说 **CUTLASS** 对 **Flash Attention 3** 至关重要。
- **Pytorch ao 0.10 发布，包含 MX 特性**：团队最近发布了 [torchao 0.10](https://github.com/pytorch/ao/releases/tag/v0.10.0)，增加了许多 **MX** 特性，并提供了一个关于 **MX dtypes** 的 [README](https://github.com/pytorch/ao/blob/main/torchao/prototype/mx_formats/README.md) 以获取更多信息。
   - 话虽如此，这需要 **nightly pytorch**，且目前仅适用于 **b200**，但更新我们现有的 **MXFP4 cutlass kernel** 以支持 **sm120** 应该相当容易。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1359388304721580144)** (6 条消息): 

> `Linux 发行版，NVIDIA 驱动，LDSM 指令，Warp Shuffling` 


- **适用于 NVIDIA 的 Linux 发行版**：一位成员询问哪种 **Linux distro** 在安装 **NVIDIA drivers** 时最省心。
   - 他们考虑如果没区别的话就直接用 **Ubuntu**。
- **LDSM 指令**：一位成员寻求关于 **LDSM**（共享内存）指令如何工作的澄清，并发布了指令 `SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, cute::half_t>; auto smem_tiled_copy_A = make_tiled_copy_A(SmemCopyAtom{}, tiled_mma);`。
   - 他们不确定在交换过程中数据存储在哪里，并询问是否有更详细解释这些硬件指令的文档。
- **通过 Warp Shuffling 进行线程交换**：一位成员同意这种理解：每个线程从源加载数据，线程间交换数据，然后将数据存储到目的地。
   - 他们还想知道线程 *如何* 交换数据，并暗示了使用 **warp shuffling** 的可能性，同时提供了一个 [NVIDIA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html?highlight=load#warp-level-matrix-load-instruction-ldmatrix) 的链接。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1359306414497202456)** (2 messages): 

> `FSDP2, Model Parallelism, Accelerate Hack` 


- **FSDP2 与其他并行方式冲突**：一名成员表示，由于 **FSDP2** 与其他并行方法相比具有独特的设计，集成起来非常困难。
   - 他们还提到，**Accelerate** 中使用的一个特定 Hack 与当前的方法冲突，突显了集成方面的挑战。
- **集成 FSDP2 困难，需要独特设计**：有人指出，虽然 **FSDP2** 的独特设计令人印象深刻，但这也使其难以与其他并行技术集成。
   - 一名成员提到，该方法与 **Accelerate** 中使用的一个 Hack 冲突，使集成过程变得复杂。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1359268567962619984)** (2 messages): 

> `CUDA vs other, SMERF, Berlin Demo` 


- **CUDA 仍是某些人的首选**：一名成员表达了对 **CUDA** 优于其他平台的偏好。
   - 然而，他们没有给出具体的替代方案。
- **SMERF 激发想象力**：一名成员分享了 [SMERF](https://smerf-3d.github.io/)，称其*非常酷且能激发想象力*。
   - 他们还分享了 [Berlin Demo](https://smerf-3d.github.io/select_quality/?scene=berlin) 的链接。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1359251457550979243)** (19 messages🔥): 

> `Graph Neural Networks (GNNs), CUDA C vs CUDA C++, Graph Attention Networks, GAN parallelism, Producers and Consumers architecture` 


- **GNNs 计算并行进行**：一名成员指出 *GNN 层有大量的变体*，但图中每个节点的更新可以并行计算，并引用了 [NVIDIA 的这篇博文](https://blogs.nvidia.com/blog/what-are-graph-neural-networks/)。
- **CUDA C++ vs CUDA C**：一名成员提到 **C++** 是 **C** 的超集，因此你可以用 **C++** 编译器编译 C 代码，并且不存在 **CUDA C** 编译器。
   - 另一名成员澄清说，编写无法通过 **C++** 编译器编译的 **C** 代码是可能的，并链接到 [这篇 Wikipedia 文章](https://en.m.wikipedia.org/wiki/Compatibility_of_C_and_C++) 以获取更多信息。
- **用于并行的 Graph Attention Networks 架构**：在考虑图上的并行计算时，一名成员建议使用 **Graph Attention Networks**。
   - 这个问题是针对 GNN 任务并行性的提问而回答的。一名成员链接了 [这张 GNN 流水线图片](https://cdn.discordapp.com/attachments/1191300313928433664/1359264906167320576/GNN-model-pipeline-China-survey-672x383.png?ex=67f78257&is=67f630d7&hm=0ee08ec9fd4dc3a6c3e477f71b9135e479aef1133ec34758dffbd1d6025268a1&)。
- **Producers and Consumers 架构**：一名成员询问为什么要使用 **Producers and Consumers**（生产者与消费者）架构，而不是让所有人在之后进行生产和消费，并好奇*这是否仅仅是为了最小化去同步时间*。
   - 这个问题是专门针对 gemms 的 **Hopper** 架构提出的。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1359317987814936727)** (1 messages): 

> `torchao v0.10.0, MXFP8 Training, Nvidia B200, PARQ, Quantization API` 


- **Torchao 发布新的 v0.10.0 版本**：最新的 [torchao](https://github.com/pytorch/ao/releases/tag/v0.10.0) **v0.10.0 版本**包括对 **Nvidia B200** 上 **mxfp8** 端到端训练的支持，以及 **PARQ**（用于量化感知训练）。
   - 它还包括用于研究的模块交换量化 API，以及一些低比特算核的更新！
- **Nvidia B200 现已兼容 MXFP8 训练**：随着 **torchao v0.10.0** 的发布，现在可以在 **Nvidia B200** 上使用 **MXFP8** 进行端到端训练。
   - 增加的支持允许使用模块交换量化 API。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1359530971237581021)** (2 messages): 

> `Brooklyn Apartments, Apartment Hunting Tips` 


- **布鲁克林公寓寻找者寻求建议**：一名成员将在秋季搬到布鲁克林，正在寻找寻找酷公寓的建议。
   - 在给出的消息中没有提供具体的公寓推荐或建议。
- **秋季搬迁至布鲁克林引发公寓搜索**：一个人计划今年秋天搬到布鲁克林，并渴望收集在该地区寻找公寓的见解。
   - 讨论目前正在进行中，等待社区成员的建议和意见。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1359581692234567911)** (1 messages): 

> `Mediant32, FP32, BF16, integer-only inference, Rationals` 


- **Mediant32 作为 FP32/BF16 的替代方案出现**：一名成员宣布了 **Mediant32**，这是一种实验性的 **FP32** 和 **BF16** 替代方案，用于纯整数推理（integer-only inference）。它基于 Rationals、连分数（continued fractions）和 Stern-Brocot 树，并提供了一份 [分步实现指南](https://leetarxiv.substack.com/p/mediant32-intro)。
- **理解 Mediant32 的数字系统**：**Mediant32** 使用一种基于 **Rationals**、**连分数**和 **Stern-Brocot 树**的数字系统，为数值表示提供了一种新颖的方法。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1359394478099534007)** (2 messages): 

> `DeepCoder, Llama 4 Scout` 


- **DeepCoder 诞生**：一位成员分享了 [DeepCoder](https://pretty-radio-b75.notion.site/DeepCoder-A-Fully-Open-Source-14B-Coder-at-O3-mini-Level-1cf81902c14680b3bee5eb349a512a51) 的链接，这是一个完全开源的 **14B Coder** 模型，达到了 **O3-mini** 级别。
- **Llama 4 Scout 已添加到 GitHub**：一位成员注意到 **Llama 4 Scout** 已被添加到 [GitHub](https://github.com/open-thought/reasoning-gym-eval/pull/6)。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1359403842810417184)** (3 messages): 

> `NVSHMEM and RDMA, Deepseek Library, RoCE or Infiniband Compatibility` 


- **通过 RoCE/Infiniband 实现基于 NVSHMEM 的 RDMA 是可能的**：一位成员建议 [NVSHMEM](https://docs.nvidia.com/nvshmem/api/using.html) 可能通过其 `get` 和 `put` API 在 **RoCE** 或 **Infiniband** 上启用 **RDMA**。
   - 该成员澄清说他们尚未测试代码，其理解是基于 **NVSHMEM** 文档的。
- **Deepseek 库分享**：一位成员询问了 **Deepseek 库**。
   - 另一位成员分享了 [GitHub 上的 Deepseek 库](https://github.com/deepseek-ai/DeepEP) 链接。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1359264114073141358)** (11 messages🔥): 

> `CUDA Inline Submissions, Datamonsters AMD Developer Challenge` 


- **CUDA Kernel 代码片段分享**：一位成员分享了一个代码片段，展示了如何通过 **cuda_sources** 和 **c++ sources** 变量以及 **load_inline** 函数在 C++ 源码中使用 CUDA 内联。
   - 代码涉及定义一个 CUDA Kernel、一个对应的 C++ 函数，并使用 `load_inline` 将其作为模块加载。
- **CUDA 内联提交已修复**：一位成员报告说 **示例提交（sample submission）** 存在错误，但已在此 [Pull Request](https://github.com/gpu-mode/reference-kernels/pull/14) 中修复。
   - 他们还询问了 [Datamonsters AMD Developer Challenge 2025](https://www.datamonsters.com/amd-developer-challenge-2025) 的 MI300 提交是否有效。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1359261423196442766)** (14 messages🔥): 

> `Grayscale Leaderboard Submissions, Matmul Leaderboard Submissions, Vectoradd Leaderboard Submissions, Modal Runners Success` 


- **Grayscale 排行榜取得新进展**：ID 为 **3539** 和 **3540** 的提交在 GPU：**L4**、**T4**、**A100**、**H100** 上使用 Modal runners 成功提交至 `grayscale` 排行榜！
- **Matmul 排行榜涌入大量新提交**：ID 为 **3549**、**3550**、**3551**、**3555**、**3556**、**3557**、**3558**、**3559**、**3561**、**3563**、**3564** 的提交在 GPU：**T4** 上使用 Modal runners 成功提交至 `matmul` 排行榜！
- **Vectoradd 排行榜提交已验证**：ID 为 **3554** 的提交在 GPU：**T4** 上使用 Modal runners 成功提交至 `vectoradd` 排行榜！


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/)** (1 messages): 

leikowo: 啊，抱歉没能及时看到你的消息，看来你们已经修复了。
  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1359262429674078449)** (77 messages🔥🔥): 

> `Together AI X-Ware.v0, Gemiji Plays Pokemon, AI Excel Formulas, Microsoft Copilot for Indie Game Devs, Agent2Agent Protocol (A2A) by Google` 


- ****Together AI** 发布 X-Ware.v0**: **Together AI** 发布了 **X-Ware.v0**，正如[这条推文](https://x.com/togethercompute/status/1909697122372378908)所宣布的，目前社区成员正在对其进行测试。
   - **X-Ware.v0** 的运行效果究竟如何仍有待观察。
- ****Gemiji Plays Pokemon** 引起关注**: 一位成员分享了 **Gemiji** 玩 **Pokemon** 的链接（[链接](https://x.com/kiranvodrahalli/status/1909699142265557208)），看起来表现不错。
   - 该帖子链接到了 Kiran Vodrahalli 的一条推文。
- **对 AI Excel 公式的兴奋**: 一位成员分享了[一个链接](https://x.com/diegocabezas01/status/1909221066565734854)，并对 AI/LLM Excel 公式表示兴奋，看到了主要参与者的实现。
   - 他们提到自己长期以来一直在思考这类 AI/LLM Excel 公式，并且他们的一位朋友成功使用了 **TextGrad**。
- ****Copilot** 作为独立游戏开发工具**: 成员们讨论了 [Microsoft Copilot](https://copilot.microsoft.com/wham?features=labs-wham-enabled) 及其在独立游戏开发中的潜力，认为这是 Agent 作为独立游戏开发者利器的证明。
   - 一些人认为代码生成 Agent 工具对于目前交付可发布的产品更有用，并提到 levels io 的 game jam 令人大开眼界。
- **Google 发布 **Agent2Agent Protocol (A2A)****: **Google** 宣布了用于 Agent 互操作性的 **Agent2Agent Protocol (A2A)**，完整规范可在[此处](https://github.com/google/A2A)获取，一位成员提到他们也参与其中。
   - 他们提供了与 **MCP** 的对比（[链接](https://google.github.io/A2A/#/topics/a2a_and_mcp.md)）。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1359246432896553131)** (66 messages🔥🔥): 

> `Llama 4 Fine-tuning, Deep Herme's Dataset, Selling 3090 turbo card, DeepCogito's LLMs, Iterated Distillation and Amplification` 


- **Llama 4 微调灾难得以避免**: 成员们提到，在 **Llama 4** 模型上微调新的 **Hermes** 本会是一场灾难，但幸运的是，他们进行了许多不同的测试，因此如果某些尝试导致性能下降，它就会被 *yeeted*（剔除）。
   - 大家一致认为 **Llama 4** 在某些方面仍有价值，不可能在所有方面都表现更差。
- **Deep Hermes 数据集是新的创意写作数据集？**: 新的 **Deep Hermes** 模型词汇量相当大，虽然在 **8b** 规模上不太聪明，但如果这个 Deep Hermes 数据集成为更聪明模型的新数据集，那么它们在创意写作方面将非常强悍。
   - 用户测试了新的 **Deep Hermes** 模型，尽管规模很小，但词汇量很大（遗憾的是在 8b 规模上不够聪明）。
- **采用 Iterated Distillation and Amplification 策略的 Cogito LLM 发布**: [DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview) 以开放许可证发布了 **3B**、**8B**、**14B**、**32B** 和 **70B** 规模的最强 LLM。
   - 每个模型在大多数标准基准测试中都优于同等规模的最佳可用开源模型，包括来自 **LLaMA**、**DeepSeek** 和 **Qwen** 的对应模型；**70B** 模型还优于新发布的 **Llama 4 109B MoE** 模型。
- **模型镜像了人类的辩论策略**: 一位成员让两个模型互相辩论，发现这与人类辩论非常相似：*它们实际上从未试图理解对方的观点，无论论据如何都坚持自己的立场*。
   - 模型会选择可以反击的弱点，忽略可能让自己陷入质疑的部分，并专注于可以用来让对方模型陷入质疑的部分。
- **Qwen 2.5 1.5B Instruct 训练展现潜力**: 一位成员正在对 **Qwen 2.5 1.5B Instruct** 进行 **RL**，并将 **gsm8k** 数据集替换为 **gsm8k platinum**，启用了 **RsLora**，模型似乎在更少的步骤中学习得更快。
   - 这种改进可能源于使用了歧义更少的数据集，以及 **RsLora** 的作用。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1359394969495802038)** (2 messages): 

> `BPE Tokenizer, Hugging Face library, Non-English text encoding` 


- **BPE Tokenizer 与多字节字符**：一位成员询问，在针对包含多字节字符的非英语文本训练 **BPE Tokenizer** 时，**Hugging Face 库**是否能确保合并的字节对形成有效的字符。
   - 另一位成员 <@687701601585987765> 被询问是否知道答案。
- **满足 minItems 要求的额外话题**：添加第二个话题以满足 topicSummaries 中至少包含两项的要求。
   - 此条目纯粹是为了符合 schema 要求，并不代表对话的实际内容。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

anka039847: https://mlss2025.mlinpl.org/
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1359298604392775782)** (57 messages🔥🔥): 

> `Local Embedding Models, GPT4All Document Indexing, Local LLM Loading Issues, RAG Implementation, GPT4All Stop Button` 


- **为了安全在本地运行 Embedding 模型**：成员们讨论了在本地运行 Embedding 模型和 LLM 的好处，以避免将私人信息发送到远程服务，其中一位成员提供了一个 [shell 脚本](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html) 用于运行来自 Nomic 的本地 Embedding 模型。
   - 该脚本使用 `$LLAMA_SERVER`、`$NGL_FLAG`、`$HOST`、`$EMBEDDING_PORT` 和 `$EMBEDDING_MODEL` 等变量来配置和运行 Embedding 服务器。
- **GPT4All 通过分块和 Embedding 索引文档**：一位用户解释说，**GPT4All** 通过对文档进行分块（chunking）和 Embedding 来索引文档，并将相似性的表示存储在私有缓存中。
   - 该用户建议运行本地 Embedding 模型和 LLM 模型，并表示即使是 **Qwen 0.5B** 参数模型也能很好地处理文档，尽管 **Qwen 1.5B** 效果更好。
- **用户在加载本地 LLM 时遇到困难**：一位成员报告称，尽管拥有 **16GB RAM** 和 **Intel i7-1255U CPU**，但在加载本地 LLM 时卡住了。
   - 他们怀疑问题出在模型下载上，并提到他们的使用场景是内部文档工具，对在私有文档中使用远程服务表示谨慎。
- **使用 Shell 脚本实现自定义 RAG**：一位成员分享了用于获取 Embedding 和向本地 LLM 发送 Prompt 的 shell 脚本示例。
   - 他们建议使用 **PostgreSQL** 存储 Embedding 并创建自定义 RAG 实现，而不是依赖远程工具。Shell 脚本示例包括 `rcd-llm.sh` 和 `rcd-llm-get-embeddings.sh`。
- **GPT4All 隐藏的停止按钮**：一位用户询问如何停止 **GPT4All** 中的文本生成，并指出没有可见的停止按钮，也无法使用 **Ctrl+C**。
   - 另一位用户指出右下角的停止按钮，它与 Generate 按钮是同一个按钮。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1359427982745403513)** (3 messages): 

> `Mojo Language, Mojo Documentation, Mojo Community` 


- **Mojo 语言，一个新的开始**：一位新用户询问从哪里开始学习 Mojo 语言以及它是什么。
   - 另一位用户表示欢迎，并指出 [Mojo 官方文档](https://docs.modular.com/mojo/manual/) 是一个很好的起点。
- **Mojo 社区欢迎你**：一位成员重点介绍了 Mojo 社区，引导用户前往 [Modular 论坛的 Mojo 板块](https://forum.modular.com/c/mojo/7) 和 Discord 上的 general 频道。
   - 用户对提供的资源表示感谢。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1359257146805719162)** (14 条消息🔥): 

> `Mojo 中的 Span 生命周期，Mojo 中的无畏并发 (Fearless Concurrency)，带有编译时参数的 MLIR 类型构造，参数化操作方言 (POP)` 


- **Mojo Trait 中的 Span 生命周期难题**：一位成员就如何在 Mojo 中表达 *返回的 Span 生命周期至少与 self 的生命周期一样长* 寻求建议，并提供了 [Rust/Mojo 代码示例](https://forum.modular.com/t/how-to-return-a-span-that-refers-to-a-struct-member-from-a-trait-method/1216)。
   - 回复指出，*使 Trait 对 origin 进行泛型化* 是一个可能的解决方案，尽管可能需要 Trait 参数支持。
- **Mojo 的无畏并发 (Fearless Concurrency) 即将到来**：有人提问 *Mojo 是否具有类似 Rust 的无畏并发*。
   - 回答是 Mojo 已经具备了所需的 Borrow Checker 约束，目前仅缺乏 **Send/Sync** 和最终的并发模型；它最终甚至可能拥有比 Rust 更好的系统。
- **编译时 MLIR 类型构造的难题**：一位成员报告了在 MAX/Mojo 标准库中使用 *MLIR 类型构造中的参数化编译时值*（特别是 **!llvm.array** 和 **!llvm.ptr**）时遇到的问题，并在 [GitHub 帖子](https://github.com/modular/max/issues/4315) 中详细说明了该问题。
   - 问题涉及在定义使用编译时参数的 **llvm.array** 类型的结构体时出现解析错误；MLIR 的类型系统似乎无法在此上下文中处理参数化值。
- **参数化操作方言 (POP) 能否解决问题？**：针对 MLIR 问题，另一位成员建议使用 *参数化操作方言 (Parametric Operations Dialect, POP)*。
   - 他们建议 Mojo 团队增加一些功能，例如让 **__mlir_type[...]** 宏接受符号化编译时值，或者提供类似 **__mlir_fold(size)** 的辅助工具，以强制将参数评估为字面量 IR 属性 (Attribute)。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1359257133115248830)** (2 条消息): 

> `面向 GenAI 的 Auth0 认证，LlamaIndex 支持，Agent 工作流，FGA 授权的 RAG，视觉引用` 


- **Auth0 发布支持 LlamaIndex 的 GenAI 认证方案**：Auth0 的 Auth for GenAI 现在包含原生 LlamaIndex 支持，通过简单的 SDK 调用即可简化 Agent 工作流中的身份验证集成。
   - auth0-ai-llamaindex SDK 提供 Python 和 TypeScript 版本，支持 **FGA 授权的 RAG**，演示见 [此处](https://t.co/bZgQ7gpuSt)。
- **Agent 通过视觉引用实现溯源**：LlamaIndex 发布了一个关于如何通过视觉引用 (Visual Citations) 为 Agent 提供依据的教程，将生成的答案映射到特定的文档区域。
   - 该功能可直接在 [此处](https://t.co/LP5XA8Yn0c) 获取。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1359398073562042418)** (8 条消息🔥): 

> `推理型 LLM，GraphRAG V2，Milvus DB，区块链专业知识` 


- **寻求推理型 LLM 教程**：一位成员正在寻求实现来自 **Hugging Face** 的 **推理型 LLM (Reasoning LLMs)** 的官方教程，特别是用于托管在 Hugging Face Space 上的 Docker 应用。
- **GraphRAG V2 Azure 身份验证错误**：一位在使用 **AzureOpenAI** 和 **Hugging Face embeddings** 实现 **GraphRAG V2** 的成员遇到了与 OpenAI API 密钥不正确相关的 **AuthenticationError**，尽管已经明确定义了 AzureOpenAI。
- **发现 Milvus DB 文件锁问题**：一位成员报告了在本地创建 **Milvus DB** 时的文件锁 (filelock) 问题，建议使用其服务器/Docker 解决方案而不是本地文件。
- **区块链工程师提供专业知识**：一位在区块链生态系统中拥有丰富经验的软件工程师提供了帮助，其经验涵盖 **DEX**、**桥 (bridge)**、**NFT 市场**、**代币启动板 (token launchpad)**、**稳定币**、**挖矿**和**质押协议**。


  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1359550232622403754)** (3 条消息): 

> `LlamaIndex 深度研究，create-llama 工具` 


- **寻求 LlamaIndex 深度研究协助**：一位成员询问了使用 **LlamaIndex** 进行深入研究的最简单方法。
   - 另一位成员提供了一个可能非常有用的工具：[create-llama](https://x.com/MarcusSchiesser/status/1907448102467911985)。
- **create-llama 工具推荐**：[create-llama](https://x.com/MarcusSchiesser/status/1907448102467911985) 工具被推荐作为使用 LlamaIndex 进行深度研究的潜在资源。
   - 这是一个旨在帮助快速创建 LlamaIndex 项目的工具。


  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1359317447521472715)** (9 messages🔥): 

> `Cohere's documentation, Pydantic schema, cURL request, List of companies` 


- **Cohere 文档介绍**：一名成员询问了关于如何使用 Cohere 获取结构化输出（例如书籍列表）的示例，另一名成员建议查阅 [Cohere documentation](https://docs.cohere.com)。
- **Pydantic Schema 讨论中**：一名成员询问是否可以直接在 `response_format` 中使用 **Pydantic schema**，以及如何在不包含 Python 版 Cohere 库的情况下发送请求。
   - 另一名成员提供了 [chat reference 链接](https://docs.cohere.com/reference/chat)，并建议将示例切换到 cURL 以查看其在 Cohere API 中的工作方式。
- **生成公司列表**：一名成员表示希望生成某个特定主题的公司列表，并询问哪种模型最适合。
   - 另一名成员提到，Cohere 目前速度最快且能力最强的生成模型是 **command**。


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

competent: 目前无法工作！
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1359316466289348729)** (2 messages): 

> `Introductions, Machine Vision, Web/AI Projects, Cohere AI Exploration` 


- **Aditya 加入，关注 AI 和 Openchains**：Aditya 拥有 **machine vision** 和 **制造设备控制** 背景，在从创新岗位休假期间正在探索 **web/AI**，并分享了他当前的项目 [openchain.earth](https://openchain.earth)。
   - 他的工具箱包括 **VS Code, GitHub Co-Pilot, Flutter, MongoDB, JS** 和 **Python**，他的目标是探索 **Cohere's AI** 如何增强他的项目。
- **热情的初学者寻求 Cohere 经验**：Aditya 渴望学习如何将 Cohere's AI 集成到他的项目中，重点是 **openchain.earth**。
   - 他带来了在 **machine vision**、**控制系统**和现代技术栈方面的丰富经验。


  

---


### **Cohere ▷ #[【🟢】status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/)** (1 messages): 

competent: 应该可以工作！
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1359256911748534273)** (4 messages): 

> `PMPP Book, Compiler Series, LLVM Tutorial, Tiny Box for Chinese Market` 


- **用于 GPU 编程的 PMPP**：一名成员推荐了 **PMPP (第 4 版)** 用于 GPU 编程。
   - 他们指出自己*对编译器不太确定*，并请求推荐。
- **编译器系列和 LLVM 教程**：一名成员表示他们正在研究这个 [compiler series](https://marcauberer.medium.com/build-a-compiler-parser-7bf4b7381ca5)。
   - 他们还表示将学习 [LLVM Tutorial](https://llvm.org/docs/tutorial/MyFirstLanguageFrontend/index.html)。
- **期待 Tiny Box**：一名成员分享说，他们*迫不及待想看到专门面向中国市场的全新 tiny box*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1359301095465160994)** (7 messages): 

> `METAL virtual device sync issue, LLaMA 7B on 4 virtual GPUs, gradient accumulation in training routine, t.grad is None issue, zero_grad() before the step` 


- **METAL 虚拟设备同步问题导致 LLaMA 7B 运行中断**：用户在 **4 个虚拟 GPU** 上使用 **METAL** 后端运行 **LLaMA 7B** 时遇到了 `AssertionError`，该错误与 `MultiLazyBuffer` 和 `Ops.EXPAND` 有关，已通过 [此 PR](https://github.com/tinygrad/tinygrad/pull/9761/files) 修复。
- **采样后设备信息丢失的问题已修复**：经过调试，发现采样后设备信息会丢失，并在 [PR 9761](https://github.com/tinygrad/tinygrad/pull/9761/files) 中提出了移动 tensor 的修复方案。
- **训练流程中的梯度累积失效**：一名用户报告说，他们在训练流程中调用 `backward()` 没有生效，在 `opt.step()` 之前 `t.grad is None`。
- **Zero grad 解决了 t.grad 问题**：用户发现，在 step 之前调用 `zero_grad()` 修复了梯度累积期间 `t.grad is None` 的问题。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1359340577581432873)** (4 messages): 

> `Contributor Tag Request, Gus from Psych` 


- **Contributor 标签任务开始**：一名成员为其 [GitHub 个人资料](https://github.com/nathan-az)申请 **Contributor 标签**。
   - 该成员幽默地提到使用 **Psych 中的 Gus** 作为他们的 Discord 头像。
- **Gus 欢迎 Torchtune 团队新成员**：另一名成员用一个 [Gus 挥手的 GIF](https://tenor.com/view/gus-wave-guswave-gif-18773699) 欢迎新团队成员。
   - 他们开玩笑地问道 *"或者我应该说..."*，暗指电视剧 Psych 中的梗。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1359330112532648067)** (4 messages): 

> `FSDP, DeepSpeed, Sharding Strategies` 


- **FSDP 与 PyTorch 的组合效果更好**：Torchtune 默认使用等同于 **zero3** 的配置，旨在与 **FSDP** 等其他 **PyTorch 分布式特性** 良好组合。
   - 一位用户提到，他们转向使用 torchtune 是为了 *避开尝试组合 deepspeed + pytorch + megatron（以及其他框架）的雷区，转而支持原生 pytorch*，并希望 *我们不要在集成和支持其他框架上过度投入*。
- **欢迎在 Torchtune 中加入 DeepSpeed Recipe**：团队很乐意展示一个导入 torchtune 并托管 **DeepSpeed recipe** 的仓库。
   - 他们将需要一个单设备副本并添加 DeepSpeed。
- **支持不同的分片策略非常直接**：支持不同的 **分片策略 (sharding strategies)** 非常直接，用户可以使用 **FSDPModule** 方法调整他们的 recipe，以便在等同于 **zero1-2** 的模式下进行训练。
   - 团队确认，只需对集合通信 (collectives) 进行微调，**zero 1-3** 都是可以实现的。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/)** (1 messages): 

aniket_19393: 有人收到 AgentX 研究方向导师的回信了吗？
  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1359598435849998584)** (1 messages): 

> `Windsurf, JetBrains, AI agent, IDE ecosystems` 


- **Windsurf 凭借在 JetBrains 上的 Wave 7 令人惊艳**：Windsurf 发布了 **Wave 7**，将其 **AI agent** 带到了 JetBrains IDE（**IntelliJ**, **WebStorm**, **PyCharm**, **GoLand**），正如其[博客文章](https://windsurf.com/blog/windsurf-wave-7)所示，这使他们成为唯一在主流 IDE 生态系统中提供智能体体验 (agentic experience) 的平台。
   - Beta 版发布包含了 Cascade 的核心功能，如 **写入模式 (Write mode)**、**聊天模式 (Chat mode)**、**高级模型**和 **终端集成 (Terminal integration)**，未来的更新承诺将提供更多功能，如 **MCP**、**Memories**、**预览与部署 (Previews & Deploys)**（[更新日志](https://windsurf.com/changelog/jetbrains)）。
- **Codeium 更名为 Windsurf**：公司已正式更名为 **Windsurf**，告别了经常被拼错的 Codeium，并将其 AI 原生编辑器重命名为 **Windsurf Editor**，IDE 集成重命名为 **Windsurf Plugins**。
   - 公告已在 [Twitter](https://x.com/windsurf_ai/status/1910037538028524030)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social/post/3lmfms7w3n227)、[YouTube](https://www.youtube.com/watch?v=TZ8UVFiTfdU)、[Instagram](https://www.instagram.com/p/DIPFz2NSTUI/) 和 [TikTok](https://www.tiktok.com/@windsurf/video/7491376934522309919) 上发布。


  

---


---


---


---


{% else %}


> 为了便于邮件阅读，完整的频道明细已截断。
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}