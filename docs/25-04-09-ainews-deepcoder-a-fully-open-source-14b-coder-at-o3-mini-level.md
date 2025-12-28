---
companies:
- together-ai
- agentica
- opena
- bytedance
- google-deepmind
- moonshot-ai
- meta-ai-fair
- runway
date: '2025-04-09T19:51:30.081055Z'
description: '以下是为您翻译的中文内容：


  **Together AI 与 Agentica** 发布了 **DeepCoder-14B**，这是一款开源的 140 亿（14B）参数编程模型，在编程基准测试中可与
  OpenAI 的 **o3-mini** 和 **o1** 媲美。该模型采用字节跳动的开源强化学习（RL）框架训练，成本约为 **26,880 美元**。**Google
  DeepMind** 推出了 **Gemini 2.5 Pro**，并向订阅者提供实验性的 “Flash” 版本。**月之暗面（Moonshot AI）** 推出了
  **Kimi-VL-A3B**，这是一款拥有 **128K 上下文**窗口的多模态模型，在视觉和数学基准测试中表现优于 **gpt-4o**。**Meta AI**
  发布了 **Llama 4 Scout** 和 **Maverick**，另有一款更大的 **Behemoth** 模型正在训练中，这些模型采用了混合专家（MoE）和
  L2 范数技术。**Runway** 推出了 **Gen-4 Turbo**，在成本不变的情况下，其效果比 Gen-3 提升了 10 倍。**谷歌**宣布 **Imagen
  3**（一款高质量文本生成图像模型）现已登陆 Vertex AI，可更轻松地实现物体移除。该报告重点介绍了开源贡献、强化学习训练优化，以及在编程、多模态和图像生成领域显著的模型性能提升。'
id: 4dcb087d-c5a7-4dae-a9c0-dd1cf43afe20
models:
- deepcoder-14b
- o3-mini
- o1
- gemini-2.5-pro
- kimi-vl-a3b
- gpt-4o
- llama-4-scout
- maverick
- behemoth
- gen-4-turbo
- imagen-3
original_slug: ainews-deepcoder-a-fully-open-source-14b-coder-at
people:
- philschmid
- lepikhin
- reach_vb
- akhaliq
- yuchenj_uw
- epochairesearch
- danielhanchen
- c_valenzuelab
title: DeepCoder：达到 O3-mini 级别的完全开源 14B 编程模型
topics:
- open-source
- reinforcement-learning
- code-generation
- multimodality
- model-training
- mixture-of-experts
- l2-normalization
- image-generation
- model-performance
- context-windows
---

<!-- buttondown-editor-mode: plaintext -->**GPRO+ 就够了。**

> 2025年4月7日至4月8日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**229** 个频道，**7279** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**692 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在 DeepSeek R1 发布之后（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-deepseek-r1-o1-level-open-weights-model/)），出现了一大批“比 R1 更开源”的克隆尝试，如果不算[蒸馏工作](https://www.youtube.com/watch?v=jrf76uNs77k&t=1036s)，目前似乎[只有 HuggingFace 的 OpenR1](https://github.com/huggingface/open-r1) 仍在发布活跃更新。然而，今天 Together 和 [Agentica Project](https://agentica-project.com/)（此前曾开展 [DeepScaleR 工作](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)）推出了一款专注于代码的 14B 推理模型，其评分达到了 O3-mini 级别：


![image.png](https://assets.buttondown.email/images/2bd31332-ba53-4690-b8b0-05b9d310f013.png?w=960&fit=max)


通常这类项目很容易刷榜，因此并不出众，但该项目的独特之处在于它是完全开源的——包括数据集、代码、配方（recipe）等，这意味着其教育价值很高，尤其是考虑到其合作者之前的成果。

专门针对 RL 训练，他们指出了采样器瓶颈：


![image.png](https://assets.buttondown.email/images/d2d565ad-746c-451a-806a-2cf7f74f1488.png?w=960&fit=max)


因此，他们对流水线化（pipelining）有非常独到的见解：


![image.png](https://assets.buttondown.email/images/1c7ec18e-12c3-4305-841d-0aedd82e15d5.png?w=960&fit=max)


并且他们还提出了对 DeepSeek GRPO 的更新：


![image.png](https://assets.buttondown.email/images/0d839059-e7f3-48ea-8ee1-07b5108182af.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

**模型发布与更新**

- **Gemini 2.5 Pro（包括其 "Flash" 实验版本）现已面向订阅用户开放**，消息来自 [@Google](https://twitter.com/Google/status/1909747273149395425) 和 [@_philschmid](https://twitter.com/_philschmid/status/1909737527386255649)。正如 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1909943627218129004) 所指出的，可以通过 Gemini 应用中的 Deep Research 功能进行访问。[@lepikhin](https://twitter.com/lepikhin/status/1909748715340152967) 提到团队正在努力应对所有流量。
- **Moonshot AI 发布了 Kimi-VL-A3B**，这是一款具有 128K 上下文、采用 MIT 许可证的多模态 LM。根据 [@reach_vb](https://twitter.com/reach_vb/status/1910046715714937130) 的说法，它在视觉 + 数学基准测试中表现优于 GPT4o。模型已在 [Hugging Face](https://twitter.com/reach_vb/status/1909706444028670311) 上可用并集成了 Transformers。[@_akhaliq](https://twitter.com/_akhaliq/status/1910047935686991904) 也关注到了此次发布。
- **Together AI 与 Agentica 合作发布了 DeepCoder-14B**，这是一个开源编程模型，在编程任务上可与 OpenAI 的 o3-mini 和 o1 媲美。据 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910004382848229702) 称，其训练成本约为 26,880 美元。[@togethercompute](https://twitter.com/togethercompute/status/1909697122372378908) 指出，该模型、训练代码、数据集和详细博客均已发布。根据 [@togethercompute](https://twitter.com/togethercompute/status/1909697131645903065) 的数据，它在 **LiveCodeBench 上获得了 60.6% 的分数，在 CodeForces 上获得了 1936 分**，在竞赛级编程任务中与 o3-mini (low) 和 o1 表现相当。[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1910008307202548074) 提到，它是使用来自 ByteDance 的开源 RL 框架训练的。
- **Meta AI 发布了 Llama 4 Scout 和 Maverick**，正如 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1909699970594394173) 所提到的，一个名为 Behemoth 的更大版本正在训练中。根据 [@danielhanchen](https://twitter.com/danielhanchen/status/1909726119500431685) 的说法，Maverick 混合了 MoE 层与密集层，而 Scout 在 QK 上使用了 L2 Norm。
- **Runway 发布了 Gen-4 Turbo**，据 [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1909976566987161785) 称，在相同价格点下，其效果比 Gen-3 提升了 10 倍。
- **Google 宣布了 Imagen 3**，这是他们最高质量的文本生成图像模型，现已集成在 Vertex AI 中。据 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1910009261075357902) 称，该模型可以更轻松地移除不需要的对象。
- **Google 宣布了 Veo 2**，据 [@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1910009257405133179) 称，它允许用户在 Vertex AI 中精炼和增强现有素材，并指导镜头构图。

**评估与基准测试**

- **OpenAI 发布了全新的 Evals API**，用于以编程方式定义测试、自动化评估运行以及迭代 Prompt，并可将其集成到任何工作流中，正如 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1909721613853139353) 所述。[@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1909721618676695270) 指出，良好的评估有助于系统地提高模型响应的质量。
- **Epoch AI Research 对 Llama 4 进行了评估**，发现 Maverick 和 Scout 在 GPQA Diamond 上的得分分别为 67% 和 52%，与 Meta 报告的分数相似，消息来自 [@EpochAIResearch](https://twitter.com/EpochAIResearch/status/1909700016249479506)。
- **ZeroBench 测试显示当前的视觉语言模型表现不佳**，据 [@LiorOnAI](https://twitter.com/LiorOnAI/status/1910022443453800746) 称，GPT-4V 和 Gemini 在 100 个高难度视觉推理问题上的 pass@1 和 5/5 可靠性得分均为 0%。

**Agent 系统与工具**

- **Auth0 的 Auth for GenAI 现在提供原生 LlamaIndex 支持**，使得在 Agent 工作流中构建身份验证变得更加容易，消息由 [@llama_index](https://twitter.com/llama_index/status/1909697035365961954) 发布。
- **MongoDB 发布了一个包含 100 多个关于 AI Agents 和 RAG 的分步 Notebook 仓库**，涵盖了从聊天机器人构建到 Airbnb Agent 的内容，消息来自 [@LiorOnAI](https://twitter.com/LiorOnAI/status/1909695352497910232)。

**行业分析**

- **Swyx 认为推特圈对个人开发者工具的评价很准确，但对 AI 如何改进 SDLC 的各个方面（这可能更具影响力）缺乏认识**，这使得 Sourcegraph 作为一个 AI 开发者工具公司处于有利地位，根据 [@swyx](https://twitter.com/swyx/status/1909695963498946903) 的说法。
- **Nearcyan 认为消费者不会通过 prompting 生成自己的完整应用**，因为大多数优秀的应用都需要数据，而消费者并没有真正的数据可移植性，根据 [@nearcyan](https://twitter.com/nearcyan/status/1909730703388115132) 的说法。
- **Svpino 认为学习如何在自己的手艺中应用 AI 至关重要**，正如 Shopify 所理解的那样，那些洞察先机的人正在要求人们去学习和研究，根据 [@svpino](https://twitter.com/svpino/status/1909699728545349689) 的说法。

**幽默/梗 (Humor/Memes)**

- **Vikhyatk 调侃西雅图市中心的午餐花费 16-20 个 H100-hours**，自从将美元转换为 H100-hours 后，热量消耗下降了 10 倍，根据 [@vikhyatk](https://twitter.com/vikhyatk/status/1909752681742422383) 的说法。
- **Scaling01 调侃 Gemini 3.0 将便宜到无法计费**，根据 [@scaling01](https://twitter.com/scaling01/status/1909967686584455174) 的说法。
- **Andrew Carr 注意到 Gemini 在玩《宝可梦》时的表现**，引用了 Gemini 的话：“我不敢相信花了六次尝试，现在游戏居然问我是否想通过给这东西起个昵称来进一步羞辱自己。没门。我不想给这个象征我失败的符号命名。我会按 B 键拒绝”，根据 [@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1909707900240773444) 的说法。


---

# AI Reddit 摘要

> 我们的流水线昨天发生了故障。抱歉！

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要的摘要的摘要

**主题 1：模型狂热：Gemini 称霸，Llama 4 遇挫，新竞争者涌现**

*   **Gemini 2.5 Pro 夺冠，但缺乏推理透明度**：在多个 Discord（LMArena, OpenRouter, Perplexity AI, Nous Research AI, aider）中，**Gemini 2.5 Pro** 因其通用能力、创意写作甚至从复杂 prompt 生成功能性代码而获得高度评价，通常被认为优于 **GPT-4.5** 和 **Claude 3.5 Sonnet** 等竞争对手。然而，用户注意到其推理 token 没有通过 Perplexity API 暴露，阻碍了其作为推理模型的使用，而且即使具备深度研究能力，除非在 **AI Studio** 中进行特定接地（grounded），否则仍会出现幻觉。
*   **Llama 4 发布引发用户哀叹**：**Llama 4** (Scout, Maverick) 的发布让用户普遍感到失望（LM Studio, Manus.im, Yannick Kilcher, Nomic.ai），用户称其“糟糕”、“过度炒作”，尽管在日语表现上尚可，但可能是退步。担忧集中在“草率的后期训练”、可能由于过拟合或“刷榜”导致的基准测试有效性存疑，以及比预期性能水平更高的 **VRAM** 要求，导致许多人等待大修或坚持使用 **Qwen 14B** 等替代方案。
*   **Cogito & Nvidia 模型挑战现状**：新模型正在掀起波澜，包括 **DeepCogito 的 v1 Preview** 模型（3B-70B），通过**迭代蒸馏和放大 (IDA)** 训练，声称优于 **Llama, DeepSeek 和 Qwen** 的同类模型，甚至优于 **Llama 4 109B MoE**，提供直接回答和自我反思模式 ([DeepCogito Research](https://www.deepcogito.com/research/cogito-v1-preview))。**Nvidia** 也悄悄发布了一个 SOTA 级别的推理模型 [Llama-3.1-Nemotron-Ultra-253B-v1](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1)，具有开启或关闭推理能力的开关 ([Nvidia Blog Post](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/))。

**主题 2：训练与微调前沿**

*   **Unsloth 微调修复与 FP4 发现**：**Unsloth AI** 解决了 3 个以上 GPU 上的 **DDP** 训练问题，建议进行特定的 CUDA 设备可见性设置，同时由于数据效率原因，提倡在 QLoRA 训练中使用 **bitsandbytes (bnb)** 而非 GGUF。用户探索了通过 [Unsloth](https://github.com/unslothai/unsloth) 等工具使用 **FP4** 对量化模型进行微调以实现更快的训练，并澄清虽然直接微调量化模型不可行，但 **LoRA** 提供了一条可行的路径。
*   **分布式训练辩论：DeepSpeed vs. FSDP 与不可信计算**：在 **Torchtune** 中，关于集成 **DeepSpeed** 的优点展开了辩论，维护者更倾向于原生 PyTorch **FSDP** 以获得更好的可组合性，尽管也提供对社区 **DeepSpeed** recipe 的支持。与此同时，受 **Nous DeMo** 论文启发的 **Panthalia** 平台（[X.com 等候名单](https://x.com/panthaliaxyz/status/1909342585505669228)）旨在通过梯度压缩（[算法文档](https://docs.panthalia.com/gradient-compression-algorithm)）验证用于**分布式数据并行 (DDP)** 训练的不可信、低成本算力。
*   **讨论的新技术与研究方向**：研究人员讨论了 [Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html) 的 **Hierarchical Perceiver** 专利，这可能与 Gemini 中的长上下文（long context）有关，并辩论了 **QKNorm** 的进展（[论文 1](https://arxiv.org/abs/2503.05453)，[论文 2](https://arxiv.org/abs/2502.00919)）。其他讨论包括用于在复杂任务中扩展自动化提示工程（Prompt Engineering）的 **MIPRO** 算法（[TensorZero 博客](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks)），以及助力 **DAPO** 研究以获得更好 RLHF 回答的 **OLMo**（[DAPO 论文](https://arxiv.org/abs/2504.05118)，[OLMo 论文](https://arxiv.org/abs/2504.04022)）。

**主题 3：工具与平台：更新、Bug 与博弈**

*   **平台更新：新 UI、速率限制与品牌重塑**：**LMArena** 推出了用于测试的 [Alpha UI](https://alpha.lmarena.ai/)，而 **OpenRouter** 发布了精美的新前端，但将免费模型的速率限制收紧至 **50 RPD**（除非用户拥有 10 美元以上的额度），引发了用户的不满。**Codeium** 在其编辑器取得成功后，正式更名为 **Windsurf**（[品牌重塑公告](https://windsurf.com/blog/windsurf-rebrand-announcement)），并开设了新的 [SubReddit](https://www.reddit.com/r/windsurf)。
*   **工具故障：Bug 困扰 Cursor、Aider 和 API**：**Cursor** 用户报告了 **C/C++ 扩展**的问题，需要回滚版本（[论坛帖子](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902)），自动选择功能选择了较差的模型，以及因绕过试用限制而可能面临的封禁。**Aider** 用户面临 **/architect 模式**编辑被截断的问题，并寻求禁用自动提交的方法（[Aider 配置文档](https://aider.chat/docs/config/options.html)），而 **Perplexity API** 用户注意到与 Web UI 相比存在差异，以及 **Sonar** 提示词过于关注系统提示词的问题（[提示词指南](https://docs.perplexity.ai/guides/prompt-guide)）。
*   **框架挫折与修复：Mojo、MAX、Granite**：**Mojo** 开发者讨论了其借用（borrowing）范式（[Mojo vs Rust 博客](https://www.modular.com/blog/mojo-vs-rust)）、`__moveinit__` 与 `__copyinit__`（[示例代码](https://github.com/sstadick/mojo-demo/tree/main/examples)）以及管理 `Span` 的生命周期。用户对比了 **MLX** 和 **MAX**，指出 **MAX** 目前无法调用 Apple Silicon GPU，而 **Unsloth AI** 用户发现了一个在 Colab 中修复 **GraniteModel** Bug 的快速方法，涉及编辑 `config.json`。

**主题 4：AI 生态系统：研究、传闻与现实应用**

*   **研究动态：专利、审计与遗忘学习**：**Google DeepMind** 尝试为 **Hierarchical Perceiver** 申请专利（[专利链接](https://www.freepatentsonline.com/y2025/0103856.html)，[论文链接](https://arxiv.org/abs/2202.10890)），引发了关于防御性专利申请和长上下文 Gemini 的讨论。研究人员正在为一项基于伦理的审计调查寻求 AI 专业人士的参与（[调查链接](https://link.webropolsurveys.com/S/AF3FA6F02B26C642)），同时 **ICML** 宣布举办机器学习遗忘学习（machine unlearning）研讨会（[研讨会网站](https://mugenworkshop.github.io/)）。
*   **行业洞察与内幕：Google 的薪酬、关税与网络犯罪**：一篇 [TechCrunch 文章](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-)声称 **Google** 据传向部分离职的 AI 员工支付一年薪水以防止其加入竞争对手，这引发了关于合法性和影响的质疑。有担忧指出，可能对 **NVDA GPU** 征收的**关税**可能会减缓 AI 的进展，而另一些人则注意到网络罪犯对 AI 的采用似乎比预期要慢，尽管未来的“冲击”仍有可能发生。
*   **应用与集成：MCP、数学、身份验证与 Agent**：**Model Context Protocol (MCP)** 的使用案例得到讨论，包括使用 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 等客户端将 **Neo4j** 图数据库集成到 RAG 中；**Semgrep** 使用 SSE 重写了其 MCP 服务器（[Cursor 演示](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df)）。**AI4Math** 的讨论强调了将 LLM 与 **Lean** 等形式化系统结合用于定理证明（[Kaiyu Yang 讲座](https://www.youtube.com/live/cLhWEyMQ4mQ)），同时 **Auth0** 的 **Auth for GenAI** 集成了原生 **LlamaIndex** 支持（[推文](https://twitter.com/llama_index/status/1909697035365961954)）。Mozilla AI 发布了 `any-agent` 以简化 Agent 框架评估（[GitHub 仓库](https://github.com/mozilla-ai/any-agent)）。

**主题 5：GPU 与硬件动态**

*   **硬件难题：ROCm 困扰与 METAL 同步故障**：由于缺乏官方支持（[AMD 文档](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)）和 WSL 透传问题，用户仍难以在 **AMD 7800XT** GPU 上通过 **WSL** 运行 **ROCm**。在 **tinygrad** 中，一名调试 **METAL 同步问题**悬赏的用户发现，LLaMA 中的分片问题可能源于 **COPY** 操作在 **XFER** 命令完成之前执行，导致数据读取错误。
*   **性能难题与优化**：**tinygrad** 用户报告称，在 **AMD** 硬件上使用 **BEAM=2** 可获得显著加速，性能超越 **Torch**。在 **GPU MODE** 中，讨论集中在 Triton 的 `tl.make_block_ptr` 配合 **`boundary_check`** 以安全处理越界内存（会有轻微性能代价），以及 **TorchTitan** 独特的预编译策略，该策略可能规避 `torch.compile` 的 bug（[TorchTitan 代码](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313)），尽管 `torch.compile` 和 **FSDP** 的数值问题依然存在。
*   **GPU 专家的新发布与资源**：**Nvidia** 的 **PhysX** CUDA 物理模拟内核现已[开源](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384)，欢迎社区进行移植（如 ROCm）。**TorchAO v0.10.0** 已发布（[发布说明](https://github.com/pytorch/ao/releases/tag/v0.10.0)），增加了针对 **Nvidia B200** 的 **MXFP8** 训练支持和模块交换量化 API。学习资源方面，推荐了 [geohotarchive YouTube 频道](https://www.youtube.com/@geohotarchive/videos)和 **《Programming Massively Parallel Processors (PMPP)》**一书（第 4 版）。

---

# 第一部分：Discord 高层级摘要

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Gemini 2.5 Pro 被宣布为 AI 至尊**：成员们称 [Gemini 2.5 Pro](https://ai.google.com/models/gemini) 为第一个“真正的” AI，强调其在创意写作和一致性方面优于之前的模型。
   - 虽然 **Gemini 2.5 Pro** 在通用任务中表现出色，但有人指出尚未发布的 **Nightwhisper** 模型在 coding 能力上更胜一筹。
- **OpenAI 的 Deep Research 受到质疑**：尽管有人声称 **OpenAI 的 Deep Research** [项目](https://openai.com/research/deep-research)是“用于网页搜索的最佳 Agent”，但对其仍存疑虑，有人表示“带有工具的 2.5 简直是另一个层级的存在”。
   - 普遍观点认为 **Deep Research** 仅仅是 **OpenAI** 现有 **o3 model** 的更名版本。
- **DeepCoder-14B 亮相，反响平平**：**Together AI** 和 **Agentica** 推出了 [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder)，这是一个代码推理模型，通过分布式 RL 从 **Deepseek-R1-Distilled-Qwen-14B** 微调而来。
   - 然而，这次发布遭到了批评，一位用户嘲讽其营销是“有史以来最愚蠢、最可耻的营销”，称考虑到这只是 o3-mini，其提升并不令人印象深刻。
- **NightWhisper 的编程能力引发期待**：尽管 **NightWhisper** 在 webdev 和 lmarena 上的可用时间很短，但其在竞技场中展示的 coding 能力让人们对其潜在的发布充满热情。
   - 有推测认为 **NightWhisper** 可能与即将推出的 **Google Ultra model** 一致。
- **Alpha UI 开启众测**：**Alpha UI** 现在可[在此](https://alpha.lmarena.ai/)进行测试，**无需密码**。
   - 用户被要求通过提供的 [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) 和 [Airtable](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 链接提供反馈和 Bug 报告，预计 **Desktop & Mobile** 端都将频繁更新。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 修复 DDP 训练补丁**：用户报告了 **HF Trainer** 和 **DDP** 在 3 个或更多 GPU 上无法工作的问题，建议确保 **CUDA** 可见设备设置为特定 GPU，但 [Unsloth 支持 DDP](https://docs.unsloth.ai/)。
   - 经过测试，它抛出了 ValueError，因此成员建议确保 **CUDA** 可见设备设置为特定的 GPU。
- **LoRA 训练首选 bnb**：建议在 **QLoRA** 训练中使用 **bnb** (bitsandbytes) 而非 **GGUF**，因为这样可以节省 4 倍的数据下载量，并且可以保存 adapter 并将其与 **bnb** 模型合并，以便稍后导出为 **GGUF**。
   - 用户在为微型模型选择使用 **bnb 4-bit** 还是 **GGUF** 进行 **LoRA** 训练时，共识倾向于前者。
- **Llama 4 模型获得“草率”评价**：测试 **Llama 4** (Scout 和 Maverick) 的成员发现，尽管 post-training 显得有些草率，但它在日语方面表现良好，且是能力出众的 **base models**。
   - 普遍情绪是等待即将到来的 post-training 彻底翻新。
- **DeepCogito v1 声称在 LLM 性能上领先**：DeepCogito 声称其 [v1 Preview models](https://www.deepcogito.com/research/cogito-v1-preview) 优于同尺寸的最佳开源模型，包括来自 **LLaMA**、**DeepSeek** 和 **Qwen** 的对应模型。
   - 这些模型提供了直接回答（标准 **LLM**）或在回答前进行自我反思（类似推理模型）的能力。
- **GraniteModel Bug 影响 Colab**：用户在使用 **GraniteModel** 的 Colab notebook 时遇到了 Bug，并提出了一个快速修复方案：编辑 `granite_based/config.json`，将 **GraniteModel** 替换为 **GraniteForCausalLM** 并重新运行单元格。
   - 在 Colab 上编辑该文件的推荐方法是下载后在本地编辑，然后将修改后的版本重新上传到 Colab。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 免费模型限制收紧**：OpenRouter 将免费模型的 token 限制降至 **50**，引发了用户的负面反应。用户对降低限制表示沮丧，一些人认为这就像是设置了“付费墙”。
   - 拥有至少 **10 美元额度（credits）**的账户，其每日请求数（**RPD**）将提升至 **1000**，而额度**少于 10 美元**的账户，其 **RPD** 将从 **200** 降至 **50**。
- **Quasar 即将推出基于额度的速率限制**：更新说明指出，**Quasar** 很快将实施依赖于额度的速率限制，虽然没有每小时限制，但速率限制为 **每分钟 20 次请求**。
   - 成员们开启了一个[反馈线程](https://discord.com/channels/994043905957435544/1243614384297644072)，供用户发布对这些变化的看法。
- **OpenRouter 推出精美的新前端**：OpenRouter 推出了非常酷炫的新前端，非常感谢 [clinemay](https://discord.com/channels/1091220969173028894/1195014798837043240/1358883684609953812)！
   - 一位用户开玩笑说，这看起来像是 *gpt-3.5 用了大约 4 分钟做出来的网站*。
- **Gemini 获封模型之王**：**Gemini 2.5 Pro** 与其他模型相比完全处于另一个层次，使其成为迄今为止最强大的模型。
   - 一位用户指出，它的评分排名是：*1. gemini 2.5 pro ... 10. 其他所有人*。
- **Nvidia 悄然发布推理模型**：[Nvidia](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) 默默发布了一个 SOTA 级别的推理模型。
   - 这个新模型随手展现出的性能就优于 **Behemoth**。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Daniel Mac 使用 GraphDB 进行代码图谱化**：一位成员分享了 [Daniel Mac 的推文](https://x.com/daniel_mac8/status/1908332949251948808)，内容关于使用**图数据库（graph database）**进行代码查询。
   - 这引发了关于使用图数据库进行代码分析以及理解代码库中复杂关系的潜在益处的讨论。
- **Manus.im 吞噬额度**：一位用户报告称 [Manus.im](https://manus.im) 未能正确回答问题，并且在单次 prompt 中消耗了其 **1000 个免费额度**中的 **984** 个。
   - 用户建议将 [Smithery.ai](https://smithery.ai/) 和 [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers) 作为潜在的替代方案。
- **C/C++ 扩展错误频发**：一位用户报告称，自 2023 年 3 月开始使用 Cursor 以来，遇到了与 **C/C++ 扩展**相关的错误，并指出该扩展可能仅限于 Microsoft 产品。
   - 建议的解决方法包括[回滚到以前的版本](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902)，用户还分享了讨论该问题的[其他论坛线程](https://forum.cursor.com/t/c-c-extension-broken/75182)。
- **Auto-Select 模型被指责为骗局**：有用户报告称 **auto-select** 模型选项会选择低质量的模型，一位用户声称它“搞砸了我的代码库”。
   - 另一位用户认为这种行为可能是故意的，引发了对 **auto-select** 功能可靠性的担忧。
- **Cursor 对绕过免费层级的用户祭出封号大棒**：一位成员报告称，绕过 Cursor 的试用版本可能会导致被完全禁止使用该工具，并警告说“你很快就会完全无法使用它”。
   - 这引发了关于 Cursor 试用版限制的公平性以及尝试规避限制所带来后果的辩论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Llama 4 令用户失望**：用户对 **Llama 4** 的表现表示失望，一些人将其描述为一种退步，并质疑基准测试（benchmark）的有效性。
   - 虽然 **Llama 4** 提供了与 **17B** 模型相似的速度/成本以及与 **24-27B** 模型相似的结果，但它需要更多的 **VRAM**，这使得它对普通用户来说毫无意义，而 **Qwen** 的 **14B** 模型则受到了称赞。
- **WSL 上的 ROCm 在 7800XT 上仍无法工作**：一名用户报告称，由于缺乏官方支持，通过 **WSL** 运行的 **ROCm** 无法在 **7800XT** 上运行（[AMD 文档](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)）。
   - 另一名用户建议它*可能*可以工作，因为两款显卡都是 **RDNA3** 架构，但第一名用户确认由于 **WSL passthrough** 问题，根本*不可能*运行成功。
- **快速修复 Cogito Jinja 错误**：用户报告了在使用 **cogito-v1-preview-llama-3b** 时出现的 **Jinja templates** 错误，并被建议使用 **ChatGPT** 来快速修复模板。
   - 社区模型维护者已收到关于模板异常的通知，预计很快会更新模型。
- **Docker 遭到吐槽**：在一名成员表示想与任何说 **Docker** 坏话的人成为“好朋友”后，另一名成员开玩笑地问：“**Docker** 是对你的家人做了什么吗？”
   - 第一名成员幽默地回答道：“我的心理医生说我不应该谈论这件事。”
- **辩论构建经济型超级计算机**：一名用户提议使用 **RTX 4090 D GPU** 或性能稍弱的方案构建一个 **16 节点超级计算机**，目标是运行具有 **1M 上下文** 的 **2T 模型**。
   - 怀疑者质疑其可行性，强调了对 **RDMA**、高速互连和专业工程师的需求。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **初创公司通过 Perplexity 节省开支**：Perplexity AI 推出了一个[初创公司计划](https://www.perplexity.ai/startups)，为符合条件的初创公司提供价值 **$5000** 的 Perplexity API 额度以及 **6 个月** 的 Perplexity Enterprise Pro。
   - 申请资格要求融资额少于 **$20M**，成立时间少于 **5 年**，并与初创公司合作伙伴有关联。
- **Gemini 2.5 推理功能引发争议**：成员们注意到 **Gemini 2.5 Pro** 没有通过 API 开放其推理 Token（reasoning tokens），因此无法作为推理模型包含在 Perplexity 中，尽管它是一个*高延迟思考模型*。
   - 因此，与 **AI Studio** 不同，其推理过程不会通过 API 显示。
- **Deep Research High 备受期待但进展受阻**：用户正在等待 **Deep Research High** 的推出，该功能旨在平均使用 **150-200 个来源**，但一名用户报告称 *Perplexity 的深度研究获取了 23 个来源，而免费的 Gemini 深度研究获取了超过 500 个*。
   - 一些成员对发布时间表缺乏沟通以及当前版本的输出只是摘要而非真正的深度研究感到沮丧；可以查看 [DeepSeek Subreddit](https://www.rxddit.com/r/DeepSeek/s/zFUYlP8NeV)。
- **Llama 4 面临基准测试造假的指责**：针对一个质疑 **Llama 4** 是否在基准测试中造假的 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/does-llama-4-fake-benchmarks-pw9wkBJ4TCOUtdZu8fmTdg#0)，引发了广泛关注。
   - 这是关于模型基准测试透明度以及用于评估 **Llama 4** 方法论的更广泛讨论的一部分。
- **Perplexity API：提示词问题依然存在**：一名用户报告称 **Sonar** 的响应侧重于系统提示词（system prompt）而非用户查询，而一名团队成员澄清说系统提示词在搜索阶段并不使用，建议用户参考 [Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide) 优化**用户提示词**。
   - 此外，一些成员讨论了在总结网页时 **Perplexity API** 与 **Web UI** 之间的差异，在使用 **sonar-reasoning-pro** 时，**API sandbox** 的结果甚至比实际的 API 好得多。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **本地版 Manus 指日可待**：成员们推测未来可能会推出本地版本的 **Manus**，类似于其他 **AI models**。
   - 这将允许用户在自己的硬件上运行 **Manus**，解决额度消耗和数据隐私方面的担忧。
- **MCP 服务器已在 Claude 上部署**：据一名成员报告，截至 2024 年 11 月 25 日，**MCP servers** 已在 **Claude** 上可用，并可与 Claude 代码配合使用。
   - 这种集成使用户能够在 Claude 环境中利用 **MCP servers** 来增强功能。
- **Llama 4 炒作降温**：在 **Openrouter.AI** 上进行测试后，用户报告称 **Llama 4** 因回复质量不佳而被过度炒作。
   - 批评还指向了 **Zucks**，他被指责在 **benchmarks** 上造假，导致性能预期被夸大。
- **Octopus 网页爬虫大放异彩**：一位成员报告称，免费网站爬虫 [Octopus](https://octoparse.com/) 在 Zillow 和 Realtor 上运行效果良好，是每月 130 美元的 Bardeen 的高性价比替代方案。
   - Bardeen 的高昂成本促使人们建议使用 **Manus** 构建自定义爬虫，作为一种更经济的解决方案。
- **Manus 额度紧缺引发用户不满**：用户对 [Manus credits](https://www.manus.im/pricing) 的高昂成本表示不满，报告称即使是简单的任务也会消耗大量额度，一名用户在单个标准复杂度任务上就耗尽了 1000 个免费额度。
   - 为了减少额度消耗，用户建议将任务拆分为更小的对话窗口，并考虑将 **Proxy** 作为更便宜的替代方案，同时等待 **Manus** 定价和额度计划的更新。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.5 与 Sonnet 的提示词能力对比**：用户发现 **Gemini 2.5** 的逻辑很强，但指令遵循能力较差，与之形成对比的是 **Sonnet** 功能丰富的编码能力，但需要更多的提示词引导。
   - 一位用户报告称，使用 Gemini 2.5 只需要 1 次提示，而 Sonnet 需要 3 次提示，尽管 Sonnet 拥有*多文件输入方法和批处理*等高级功能。
- **Aider 的自动提交功能引发混乱？**：由于 Aider 会提交未经测试的代码，一位用户寻求禁用 **Aider's auto-committing** 的方法，并参考了 [Aider configuration options](https://aider.chat/docs/config/options.html)。
   - 另一位用户建议提供 [model and key](https://aider.chat/docs/troubleshooting/models-and-keys.html)，否则 Aider 将根据可用密钥进行猜测。
- **OpenRouter 缺失 Sonar Pro 引用**：一位用户质疑通过 **OpenRouter** 使用 **Perplexity Sonar Pro** 时缺失引用链接，并在此提供了[视觉参考](https://cdn.discordapp.com/attachments/1131200896827654144/1358926629170319490/image.png?ex=67f798cc&is=67f6474c&hm=fe2c340b866bec81e485bbed3c2d1fe17071b540d6ea5c803306211e3d9f2ceb&)。
   - 讨论暗示通过 OpenRouter 使用某些模型时，引用链接的可靠性可能存在问题。
- **软件工程师的间隔年是职业生涯杀手？**：一篇文章认为，对于软件工程师来说，休间隔年或长假是一个糟糕的决定，文章引用了对当前技术格局的见解，详见[这篇文章](https://ghuntley.com/dothings/)。
   - 作者认为，技术快速演进的本质使得长时间的休息不利于保持竞争力。
- **架构模式编辑被中断**：用户报告在 Aider 的 **/architect mode** 编辑过程中添加新文件会导致编辑被切断，从而可能丢失编辑器状态。
   - 避免在编辑过程中添加新文件似乎可以让过程不间断地继续。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **AgentSpace 为企业开启 NotebookLM**：Google 的 **AgentSpace** 文档显示，**NotebookLM Enterprise** 现在可以设置[客户管理的加密密钥 (CMEK)](https://cloud.google.com/agentspace/notebooklm-enterprise/docs/set-up-notebooklm#cmek)，以实现更好的数据加密控制。
   - 一位用户询问了商业规模的 **NotebookLM**，另一位成员指出了这一新产品。
- **NotebookLM 的隐私保证得到确认**：据一名成员称，**NotebookLM** 的 **Enterprise** 和 **Plus** 版本均确保用户数据保持私密，绝不会进入公共领域。
   - 这一澄清解决了对 **Google 隐私政策**和条款的误解，并指出其内置了防止 Prompt Injection（提示词注入）的机制。
- **用户纠正改进了 NotebookLM 的摘要**：一位用户报告称，**NotebookLM** 最初误读了一篇学术文章，但在提供引用和解释后自行进行了纠正。
   - 从头开始在不同的 **Google 账号**中重复相同的 Prompt 得到了正确的结果，这引发了关于训练和隐私的疑问。
- **Discovery Mode 推出仍在进行中**：用户仍在等待 **NotebookLM** 中的新功能 **Discovery Mode**，预计从发布日期起需要长达**两周**的时间完成推送。
   - 一位用户幽默地要求作为 *Google 铁粉获得特殊待遇* 以尽早获得访问权限。
- **Gemini 在深度研究中仍会产生幻觉**：用户报告称，即使有互联网访问权限，**Gemini** 在进行 **Deep Research** 时仍会产生“幻觉”。
   - 一名成员澄清说，**Gemini** 可以连接到 Google Search，但需要在 **AI Studio** 中设置特定的 Grounding 指令。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek R2 准备在 LlamaCon 发布**：成员们敦促 **DeepSeek** 在 **LlamaCon** 当天发布 **R2** 以利用热度，并指出 **MoE** 的训练数据与基础模型不同，引用了[这篇论文](https://arxiv.org/abs/2410.19034)。
   - 此次发布可能会挑战其他模型，并在活动期间吸引大量关注。
- **Together AI 进入训练领域**：**Together AI** 正在进入模型训练业务，[这一案例研究](https://www.together.ai/models/cogito-v1-preview-llama-70b)展示了 **Cogito-v1-preview-llama-70B** 模型。
   - 此举标志着其向提供包括训练基础设施和服务在内的全面 AI 解决方案转变。
- **传闻 Google 支付 AI 员工薪水让其闲置**：根据 [TechCrunch 的这篇文章](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-)，**Google** 据称支付部分 **AI 员工**一年的薪水让他们无所事事，而不是允许他们加入竞争对手。
   - 一名成员批评这是一种*具有极其糟糕的二阶效应的基础管理思路*，另一名成员指出，这可能会因为限制员工在合同期内的行为或开发工作而产生法律风险。
- **关税威胁 NVDA GPU 可用性**：成员们推测，如果**关税**持续存在，由于 **NVDA GPU** 成本增加，AI 领域可能会放缓。
   - 这可能会影响开发和研究，因为获取必要硬件的财务压力会变大。
- **OLMo 助力 DAPO 研究**：成员们讨论了一篇 [DAPO 论文](https://arxiv.org/abs/2504.05118)，认为其提供了“极端价值”，并引用了[另一篇基于 OLMo 构建的论文](https://arxiv.org/abs/2504.04022)。
   - 研究人员指出了一种新型计算方法，可以在 **RLHF** 任务中获得更好的答案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **DeepMind 的分层专利追求**：[Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html) 正在尝试为 **Hierarchical Perceiver** 申请专利，人们将专利图表与原始 [研究论文](https://arxiv.org/abs/2202.10890) 中的图表进行了对比。
   - 推测认为，这项专利可能与 DeepMind 在 Gemini 中实现的 **超长上下文长度（ultra-long context lengths）** 工作有关，可能是一种防御性措施。
- **调查寻求 AI 审计专家**：一位研究人员正在寻求 AI 专业人士参与一项关于生成式 AI 系统基于伦理审计的调查。
   - 该 [调查](https://link.webropolsurveys.com/S/AF3FA6F02B26C642) 旨在收集关于审计或评估 AI 系统（尤其是生成模型）的见解。
- **关于 QKNorm 可疑进展的辩论**：成员们辩论认为 **QKNorm 的进展** 并非正确的方向，并引用了 [这篇论文](https://arxiv.org/abs/2503.05453)。
   - 一位成员推荐了一篇 [更好/更早的论文](https://arxiv.org/abs/2502.00919)。
- **ICML 邀请对机器遗忘（Unlearning）进行研究**：一位成员分享了 [ICML](https://icml.cc/Conferences/2024) 将举办 **机器遗忘研讨会（machine unlearning workshop）** 的消息。
   - 研讨会的网站可以在 [这里](https://mugenworkshop.github.io/) 找到。
- **寻求 LM Harness 实施指导**：一位成员询问关于 **HotpotQA 的 LM harness 实现**，以评估 **Llama** 和 **GPT models**。
   - 请求关于针对 **HotpotQA** 运行评估的指导。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama-4-Scout-17B 已适配 llama.cpp**：[Llama-4-Scout-17B text-to-text](https://github.com/ggml-org/llama.cpp/pull/12791) 支持已添加到 *llama.cpp*，成员们正在对该模型进行转换和量化。
   - 这一预发布版本引起了用户的兴奋，大家渴望测试其能力。
- **Gemini 2.5 Pro 生成功能性代码片段**：**Gemini 2.5 Pro** 因能根据复杂提示词生成功能性代码片段而受到称赞，请在 [此消息](https://cdn.discordapp.com/attachments/1149866623109439599/1358975415426879589/message.txt?ex=67f7c63b&is=67f674bb&hm=1c655347ddb71efc0e03a079e62d8e26286724363242370cf6f19b9e50cc1980&) 中查看提示词和响应。
   - 一位用户报告使用 **aider-chat** 结合 **Gemini 2.5 Pro**，从 **300k token 上下文** 中编辑或创建了 15 个文件，包括他们的前端、API 和微服务。
- **HiDream-I1 生成高质量图像**：**HiDream-I1** 是一款新型开源图像生成基础模型，拥有 **17B 参数**，使用 **Llama 3.1 8B** 作为文本编码器，采用 [MIT 许可证](https://huggingface.co/HiDream-ai/HiDream-I1-Full) 发布。
   - 它 *在包括写实、卡通、艺术等多种风格中产生了卓越的效果，实现了最先进的 HPS v2.1 分数，符合人类偏好*。
- **Cogito 模型使用迭代蒸馏**：一套全新的 **Cogito** 模型（**3B-70B**）表现优于 **Llama, DeepSeek, 和 Qwen** 等模型，这些模型使用 **迭代蒸馏与放大（Iterated Distillation and Amplification, IDA）** 进行训练，该方法可以迭代地提高模型的能力。
   - 值得注意的是，据 [此项研究](https://www.deepcogito.com/research/cogito-v1-preview) 概述，**70B 模型** 据称超越了新发布的 **Llama 4 109B MoE 模型**。
- **Panthalia 平台旨在通过 DDP 验证低成本算力**：受 **Nous DeMo** 论文启发，一个旨在验证用于通过互联网进行模型训练的不可信、低成本算力的平台已经开发完成，该平台使用分布式数据并行（DDP），可通过 [X.com](https://x.com/panthaliaxyz/status/1909342585505669228) 加入等待名单。
   - 该平台使用了一种梯度压缩算法，记录在 [此处](https://docs.panthalia.com/gradient-compression-algorithm)，代码可在 [GitHub](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py) 上获取。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPUMODE 的数据集需要 PyTorch 2.5**：用于 [Inductor Created Data](https://huggingface.co/datasets/GPUMODE/Inductor_Created_Data_Permissive) 的 **GPUMODE** "triton" 数据集是使用 **PyTorch 2.5** 创建的，创建者承诺将更新 readme。
   - 用户在 **PyTorch 2.6+** 上运行该数据集时可能会遇到问题。
- **Triton 获得边界检查功能**：一名成员建议使用带有 **`boundary_check`** 和 **`padding_option="zero"`** 的 `tl.make_block_ptr` 来创建指针，以便在越界内存访问时填充零。
   - 对方澄清说，省略 `boundary_check` 可以提高速度，但由于潜在的缓冲区溢出，存在触发 *"device-side assert triggered"* 等错误的风险。
- **TorchTitan 在操作前进行编译**：**TorchTitan** 在操作前会进行独特的逐块编译，这可能是为了规避某些 **torch compile bugs**；详见 [torchtitan/parallelize_llama.py#L313](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313)。
   - 同时使用 `torch.compile` 和 **FSDP** 时，可能仍然存在数值问题。
- **PhysX 现已开源**：NVIDIA 的 **CUDA 物理模拟内核**现已[开源](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384)，并且已经有人在开发 **ROCm** 版本。
   - **Triton-Distributed** 学习笔记详细介绍了将 Triton 与 **NVSHMEM/ROC-SHMEM** 融合以实现多 GPU 执行的方法。
- **LiveDocs 提供可靠的文档管理**：**LiveDocs** 的创建者邀请用户使用其升级后的服务来*编写代码文档*，现在通过在 [www.asvatthi.com](http://www.asvatthi.com) 注册即可使用更多功能。
   - 其中包含一张界面截图，展示了各种代码文档页面。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **FP4 微调加速任务完成**：用户正在探索使用 [Unsloth](https://github.com/unslothai/unsloth) 等工具通过 **FP4** 微调量化模型，该工具允许加载低精度模型进行训练和量化。
   - 虽然可以通过 **LoRA** 对量化模型进行微调，但直接对量化模型本身进行微调是不可能的。
- **Parasail 提供卓越性能**：新型推理提供商 **Parasail** 在结束隐身模式后，正寻求与 Hugging Face 合作。据 [The Next Platform](https://www.nextplatform.com/2025/04/03/parasail-brokers-between-ai-compute-demand-and-supply/) 报道，该公司已在 Open Router 上每天处理 **30 亿 token**，并为私有公司每天处理超过 **50 亿 token**。
   - The Next Platform 报道称，Parasail 在 AI 算力需求和供应之间充当经纪人。
- **Llama.cpp 跨越至 Llama 4**：根据 [GitHub releases](https://github.com/ggml-org/llama.cpp/releases)，后端 **Llama.cpp** 已更新以支持 **Llama 4**。
   - 此次更新增强了与最新 Llama 模型的兼容性和性能。
- **AI Runner 桌面 GUI 正式发布**：一名成员发布了 **AI Runner**，这是一个使用 HuggingFace 库在本地运行 AI 模型的桌面 GUI，如[此 YouTube 视频](https://youtu.be/IPn3TcQr7e0)所述。
   - 该工具允许用户创建和管理具有自定义声音、性格和情绪的聊天机器人。这些机器人是使用 llama-index 和 ReAct 工具构建的 Agent，能够通过 **Stable Diffusion** 生成图像并进行实时语音对话（使用 espeak、speecht5 或 openvoice）。
- **any-agent 库简化 Agent 框架评估**：Mozilla AI 团队发布了 `any-agent`，这是一个旨在简化尝试不同 Agent 框架的库，[GitHub 仓库](https://github.com/mozilla-ai/any-agent)已开放供用户尝试和贡献。
   - 该库支持 **smolagents**、**OpenAI**、**Langchain** 和 **Llama Index** 等框架。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Semgrep MCP Server 获得 Docker 助力**：一位成员报告称已运行 [Semgrep MCP server](https://mcp.semgrep.ai/sse) 超过一个月，该服务器通过 **Docker** 和 **AWS EC2** 托管。
   - 这一配置是 MCP 在云端环境部署的实际演示，鉴于其易用性，具有广泛采用的潜力。
- **Semgrep MCP Server 修复 CORS 错误**：在连接 [Cloudflare Playground](https://playground.ai.cloudflare.com/) 时报告的 **CORS error** 已被迅速解决。
   - 该工具正配合 **Cursor** 进行测试，表明了实际应用和集成的需求。
- **MCP 为企业客户提供 HTTP 请求-响应支持**：针对企业客户对 MCP 中 **HTTP request-response** 支持的需求展开了讨论，详见[此 Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206#issuecomment-2766559523)。
   - 对该功能的需求凸显了 MCP 在企业机构中日益增长的采用率。
- **MCP 集成图数据库用于 RAG**：一位成员询问了在 **RAG** 场景中使用 MCP 配合 **Neo4j graph database** 的情况，重点关注向量搜索和自定义 **CQL search**。
   - 另一位成员确认这是一个很好的用例，并链接到了 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 作为可行的 MCP 客户端，展示了 MCP 的多功能性。
- **Semgrep 使用 SSE 重写 MCP Server**：一位成员重写了 [Semgrep's MCP server](https://github.com/semgrep/mcp)，并分享了在 [Cursor](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df) 和 [Claude](https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411) 中使用 **SSE** 的演示视频。
   - 该服务器使用 **SSE** 是因为 [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/416) 尚不支持 HTTP streaming。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Shopify 的 AI 探索势头强劲**：Shopify 的 AI 战略正受到关注，如[此推文](https://fxtwitter.com/tobi/status/1909251946235437514)所述。
   - 该公司正推动其平台全线集成 AI，内部讨论集中在实际应用和战略影响上。
- **Anthropic API 额度设有有效期**：Anthropic API 额度在一年后过期，这可能是为了简化会计处理，并考虑到快速发展的 AI 领域。
   - 成员们认为，这一政策有助于在快速变化的领域中管理预期，为资源分配和未来规划提供框架。
- **NVIDIA 推理模型支持开关切换**：NVIDIA 发布了一个新模型，具备开启或关闭推理的能力，详见[此博客文章](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)，该模型已在 [Hugging Face](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) 上线。
   - 该功能允许开发者尝试不同的推理方法，并针对特定任务微调其 AI 应用。
- **网络犯罪对 AI 的采用慢于预期**：尽管出现了 FraudGPT 等基础 AI 应用，但网络犯罪分子大规模采用 AI 的速度出奇地慢，有人推测当他们更广泛地采用 AI 时，可能会发生“网络犯罪 AI 冲击”。
   - 一位成员指出，LLM 可能直到最近才足够成熟到可以用于网络犯罪，这表明该技术在这一背景下仍在发展中。
- **Gemini 直播宝可梦游戏**：Gemini AI 正在玩《宝可梦》，引起了关注，如[此推文](https://fxtwitter.com/kiranvodrahalli/status/1909699142265557208)所示。
   - 这展示了 AI 在游戏和互动娱乐方面的潜力，证明了其在虚拟环境中处理复杂任务的能力。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Llama 4 基准测试缺陷曝光**：一位成员断言 **Llama 4** 在非博弈、非过拟合的基准测试中**表现不佳 (flops)**，引发了对论文 [arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220) 和相关 [YouTube 演讲](https://www.youtube.com/watch?v=klW65MWJ1PY) 的关注。
   - 根据[此 fxtwitter 链接](https://fxtwitter.com/lmarena_ai/status/1909397817434816562?t=Gdzbf-abkahHSxqhEeqAkw&s=19)，人们担心 *Meta 应该澄清 “Llama-4-Maverick-03-26-Experimental” 是一个为了优化人类偏好而定制的模型*。
- **解码 Bayesian Structural EM 的秘密**：一位成员强调 **Bayesian inference**（贝叶斯推理）结合权重和架构已有约一个世纪的历史，并引用 [Bayesian Structural EM](https://arxiv.org/pdf/1301.7373) 作为例子。
   - 他们认为，*同时更新架构和权重并不能获得仅靠权重无法获得的表达能力 (expressivity)*，并引用 [DARTS](https://arxiv.org/pdf/1806.09055) 或 [ES-ENAS](https://arxiv.org/pdf/2101.07415) 作为进一步的例子。
- **模型的 DNA：程序化模型表示 (Procedural Model Representation)**：一位成员介绍了**程序化模型表示**，即通过一个小种子生成一个大型模型（架构 + 权重），设想下载一个 10MB 的模型来生成一个 100TB 的模型。
   - 该成员将其描述为*通过下载 DNA 来生成人类*，通过更换种子来生成不同的模型。
- **Cogito 14b 采用高效工具模板**：**14b 模型**出人意料地开始使用比初始指令中提供的更高效的工具调用 (tool calling) 模板，参见 [Cogito 模型](https://ollama.com/library/cogito)。
   - 这表明该模型可能自主优化了其工具使用，为进一步研究提供了潜在领域。
- **DeepCogito 迭代改进**：一位成员分享了来自 **Hacker News** 的链接，关于使用测试时计算 (test time compute) 进行微调的**迭代改进策略**，出自 [DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview)。
   - 另一位成员指出了[这篇论文](https://arxiv.org/pdf/2408.04220)，并分享了一个关于调整 **pre-training text**（预训练文本）的 [精彩演讲](https://www.youtube.com/watch?v=klW65MWJ1PY)。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Granite 8B 的 RAG 能力令人印象深刻**：成员们报告称 [IBM Granite 8B](https://www.ibm.com/blogs/research/granite-foundation-models/) 在 **RAG** 任务中非常有效，特别是在提供引用 (references) 方面。
   - 其他成员表示赞同，也发现 **Granite** 非常有效。
- **Docling 精细处理 OCR**：一位成员推荐使用 **docling** 进行**图像 OCR**，特别是针对扫描件等非文本 PDF，以便运行 embeddings。
   - 他们强调了其在生成 embedding 方面的持续运行，以及集成到带有索引文档的数据库中，从而通过交集实现 **RAG**。
- **语义分块 (Semantic Chunking) 对上下文进行分块**：一位成员分享了一个语义分块服务器，展示了其在 [剪贴板示例](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html) 中的应用。
   - 他们注意到它与音频和图像处理的兼容性，建议使用 **ComfyUI** 来结合所有模态。
- **Llama 第 4 代遭到猛烈抨击**：一位成员痛批 **Llama 第 4 代模型**，称其*与较小的模型相比表现糟糕*。
   - 其他人表示同意，并指出 [Reddit 评论](https://www.reddit.com/r/LocalLLaMA/) 推测它可能在较小的“高质量”数据集上过拟合了，尽管某些基准测试显示出前景。
- **GPT4All：本地运行！**：一位成员建议主要在本地使用 **GPT4All**，以确保隐私并避免将私密信息发送到远程 API。
   - 他们详细说明了如何在本地运行 embedding 模型，并通过分块 (chunking) 和嵌入 (embedding) 对文件进行索引，并参考了一个 [shell 脚本示例](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html)。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **MAX 在 Apple Silicon 部署方面遇到困难**：一位成员对比了 **MLX** 和 **MAX**，指出 **MAX** 目前无法像 **MLX** 那样以 Apple Silicon GPU 为目标，这给直接对比和部署带来了挑战。
   - 他们建议，虽然 **MLX** 对于初始实验很方便，但在服务器设置中部署 Apple 生态系统的实际限制使得有必要重写为 **MAX**、**JAX** 或 **PyTorch** 等框架。
- **Mojo 借用范式获得好评**：一位新人分享了一篇[对比 Mojo 和 Rust 的博客文章](https://www.modular.com/blog/mojo-vs-rust)，观察到 Mojo 的 *默认借用 (borrow by default)* 感觉更直观，并想知道 Mojo 如何处理函数的返回值。
   - 随后讨论了 Mojo 如何处理从函数返回值的机制。
- **Moveinit vs Copyinit 深度探讨**：一位成员澄清说，在 Mojo 中返回对象时，`__moveinit__` 的存在决定了对象是否被移动，否则将使用 `__copyinit__`，并提供了一个 [GitHub 上的示例](https://github.com/sstadick/mojo-demo/tree/main/examples)。
   - 该成员还指向了 [Mojo 官方文档](https://docs.modular.com/) 以获取完整信息。
- **Span 生命周期让你困扰？使用 Rebind！**：一位成员询问如何在 Mojo 中指定 *“返回值的生命周期至少与 self 的生命周期一样长”*，特别是针对 `Span`。
   - 另一位成员建议使用 `rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))` 或使 trait 对 origin 进行泛型化，但指出目前尚不支持 trait 参数。
- **自我推广规则触发管理员干预！**：一位成员举报了 Discord 频道中的一条帖子违反了自我推广规则。
   - 管理员表示同意，确认该帖子确实违反了社区的自我推广指南。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **寻求优雅的 Tensor 命名方式**：一位成员正在寻求一种更优雅的方式来命名 Tensor，以便在打印模型参数时更容易跟踪，而不是手动在 Tensor 类中添加 *name* 属性。
   - 该成员正在寻求简化 Tensor 命名约定的技术，以增强代码的可读性。
- **GPU 编程和编译器开发资源**：一位成员表示有兴趣参与 **GPU 编程** 和 **编译器开发**（针对 tinygrad 等项目），并请求学习资源或博客文章。
   - 该成员计划阅读 [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/)，并征求关于 GPU 编译器开发的图书或博客推荐。另一位成员推荐了 [geohotarchive YouTube 频道](https://www.youtube.com/@geohotarchive/videos) 作为学习 tinygrad 的资源，以及 **PMPP (第 4 版)** 用于 GPU 编程。
- **METAL 同步故障导致 LLaMA 分片出错**：一位成员在复现悬赏任务中关于 **METAL 同步问题** 的最小示例时发现了分片（sharding）中的异常行为，怀疑从 **METAL:1** 到 **CPU** 的 **COPY** 操作在从 **METAL** 到 **METAL:1** 的 **XFER** 结束之前就执行了。
   - 用户认为这导致 CPU 在 **LLaMA** 推理期间读取的是零，而不是正确的分片。
- **AMD BEAM=2 为 Tinygrad 提速**：一位用户报告称，使用 **AMD** 配合 **BEAM=2** 获得了令人印象深刻的速度提升，达到了 **64 it/s**，超过了之前使用 Torch 达到的 **55+ it/s** 的最佳纪录。
   - 成员们指出 *BEAM=2 通常优于 Torch*。
- **LLaMA 分片丢失设备信息**：一位用户在运行带有 `--shard 4` 参数的 **llama.py** 时遇到了 **AssertionError**，表明采样后设备信息丢失。
   - [GitHub](https://github.com/tinygrad/tinygrad/pull/9761/files) 上提出了一个潜在的修复方案，即移动 Tensor。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Llama 4 助力全新 RAG 工作流**：一个快速入门教程演示了如何使用 **Llama 4** 从零开始构建 **RAG 工作流**，展示了如何利用 LlamaIndex 工作流设置围绕数据摄取 (ingestion)、检索 (retrieval) 和生成 (generation) 的核心步骤，详见[此推文](https://twitter.com/llama_index/status/1909635186079453494)。
   - 该教程专注于围绕数据摄取、检索和生成的核心步骤。
- **Auth0 与 LlamaIndex 联手推出 GenAI 身份验证**：**Auth0 的 Auth for GenAI** 现在提供原生 LlamaIndex 支持，使得在 Agent 工作流中构建身份验证变得更加容易，正如[此推文](https://twitter.com/llama_index/status/1909697035365961954)中所宣布的。
   - 这一集成简化了在基于 Agent 的应用程序中加入身份验证的过程。
- **Gemini 2.5 Pro 停用，转向统一 SDK**：成员们发现 **Gemini 2.5 Pro** 已被弃用，建议改用 **Google 最新的统一 SDK**，如 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/)中所述。
   - 有人提到 **Google SDK** 不会验证模型名称，而是假设提供的名称是有效的，因此仔细检查可能很重要。
- **StructuredPlannerAgent 被移除**：`StructuredPlannerAgent` 的文档已被删除，因为在 Agent 文档清理过程中它不再被维护，并提供了一个回链供历史参考：[StructuredPlannerAgent](https://docs.llamaindex.ai/en/v0.12.15/examples/agent/structured_planner/)。
   - 建议不要使用 `StructuredPlannerAgent`，而是使用带有 **planning tool**（规划工具）的 Agent 来进行一些 **Chain of Thought (CoT)** 推理，或者在调用 Agent 之前使用 **LLM** 本身来创建计划。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **成员询问活动录音**：一名成员询问了无法参加现场活动的人是否可以获得活动录音，但未得到回应。
   - 该成员表达了兴趣，因此在未来，**发布活动录音**将使缺席的成员受益。
- **新手寻求结构化输出指导**：一名新成员请求提供如何使用 **Cohere** 获取结构化输出（例如书籍列表）的示例，并被引导至 [Cohere 文档](https://docs.cohere.com)。
   - 用户承认对 **Cohere** 缺乏经验，官方文档中可能需要更多关于 **structured output** 的示例。
- **通过 cURL 集成 Pydantic Schema**：一名成员寻求在 Cohere 的 `response_format` 中直接使用 **Pydantic schemas** 且不使用 Cohere Python 包的方法。
   - 他们收到了 [Cohere Chat API 参考链接](https://docs.cohere.com/reference/chat) 以及一个向 `https://api.cohere.com/v2/chat` 发送请求的 **cURL** 示例，模仿了 **OpenAI SDK** 的方法。
- **Cohere 回避向量数据库推荐**：历史上一直避免对 **vector DBs** 做出明确推荐，因为 Cohere 的模型旨在与 *所有* **vector DBs** 有效配合。
   - 这种方法确保了广泛的兼容性以及对 **vector database 生态系统** 的中立立场，这意味着不需要针对任何特定的 **vector DB** 进行特殊优化。
- **Aditya 加入 Cohere 社区**：拥有 **machine vision and control** 背景的 Aditya 在休假期间介绍了自己，并正在通过 [openchain.earth](https://openchain.earth) 项目探索 Web/AI。
   - Aditya 正在使用 **VS Code**、**GitHub Copilot**、**Flutter**、**MongoDB**、**JS** 和 **Python**（评估中），希望了解更多关于将 **Cohere AI** 集成到其项目中的信息。



---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **寻求贡献者标签**：一名成员在 Discord 上申请 Contributor 标签，并分享了他们的 [GitHub 用户名](https://github.com/nathan-az)。
   - 该用户风趣地提到他们的 Discord 头像使用的是美剧《灵异妙探》（*Psych*）中的角色 Gus。
- **关于 TorchTune 集成 DeepSpeed 的辩论**：一名成员询问是否可以将 [DeepSpeed](https://www.deepspeed.ai/) 作为后端集成到 TorchTune 中，并创建了一个 [Issue](https://github.com/pytorch/torchtune/issues/2569) 来讨论这种可能性。
   - 一位维护者询问了更多背景信息，并指出 **FSDP 支持 DeepSpeed 的所有分片（sharding）选项**。
- **TorchTune 倾向于 FSDP 而非 DeepSpeed**：TorchTune 更倾向于使用 **FSDP**，因为它能更好地与 PyTorch 的其他分布式特性组合，并认为*同时支持好两个版本是不可行的*。
   - 为了避免 DeepSpeed、PyTorch 和 Megatron 组合时的复杂性而迁移到 TorchTune 的用户，更倾向于坚持使用原生 PyTorch。
- **TorchTune 的 DeepSpeed Recipe？**：一位维护者建议创建一个社区 Recipe，通过导入 TorchTune 并托管一个 DeepSpeed Recipe，并表示如果建立了代码库，愿意对其进行推荐。
   - 这使得对 **DeepSpeed** 感兴趣的用户可以在 TorchTune 中使用它，同时保持核心框架专注于原生 PyTorch。
- **为 ZeRO-1/2 训练调整 FSDPModule**：由于 TorchTune 默认使用相当于 **ZeRO-3** 的配置，因此关于如何使用 **FSDPModule** 方法调整 Recipe 以进行 **ZeRO-1/2** 训练的文档或更多 Recipe 将会很有帮助。
   - 据信，只需对集合通信（collectives）进行非常微小的调整，即可实现 **ZeRO 1-3**。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **MIPRO 算法在复杂任务上的扩展**：一篇[文章](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks)测试了 **MIPRO 自动提示词工程算法**在不同复杂程度任务中的表现，从命名实体识别到基于文本的游戏导航。
   - 该研究利用了 **CoNLL++、HoVer、BabyAI** 和 **τ-bench**（涉及 Agent 工具使用的客户支持）等任务。
- **大型模型更能发挥 MIPRO 的优势**：研究发现，在复杂设置下，**大型模型从 MIPRO 优化中获益更多**，这可能是因为它们能更有效地处理较长的多轮示例（demonstrations）。
   - 反馈的质量显著影响 MIPRO 的优化过程，即使是来自**带有噪声的 AI 生成反馈**也能看到明显的改进。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Kaiyu Yang 探索形式化数学推理**：客座讲师 **Kaiyu Yang** 在直播中发表了题为 *“用于自动形式化和定理证明的语言模型”* 的演讲，视频可在[此链接](https://www.youtube.com/live/cLhWEyMQ4mQ)观看。
   - 讲座涵盖了使用 LLM 进行形式化数学推理的内容，包括**定理证明**和**自动形式化**（autoformalization）。
- **AI4Math 对 AI 系统变得至关重要**：**数学人工智能 (AI4Math)** 对于 AI 驱动的系统设计和验证至关重要，它借鉴了 NLP 技术，特别是针对精选数学数据集训练 LLM。
   - 一种补充方法涉及基于 **Lean** 等系统的形式化数学推理，这些系统可以验证推理的正确性并提供反馈。
- **Yang 博士增强数学领域的 AI 能力**：Meta FAIR 的研究科学家 **Kaiyu Yang 博士**专注于通过集成 **Lean** 等形式化系统来增强 AI 的数学推理能力。
   - 他的工作探索了使用 LLM 执行定理证明（生成形式化证明）和自动形式化（将非形式化语言翻译为形式化语言）等任务。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Manifold Research 深度探讨**：**Manifold Research Group** 将于本周六（太平洋标准时间 4/12 上午 9 点）举办他们的 [第 4 次社区研究电话会议](https://lu.ma/wlne416w)，展示他们的最新项目。
   - 讨论内容将包括**多模态 AI**、**自组装空间机器人**以及**机器人元认知**，并邀请在垂直科学领域进行协作。
- **群集空间机器人技术起飞**：Manifold Research Group 的一名专注于空间机器人群的研究生发出了参加此次研究电话会议的邀请。
   - 该研究会议旨在鼓励协作并探索空间机器人领域的前沿科学。

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Codeium 在编辑器取得成功后更名为 Windsurf**：在 2024 年 11 月成功推出 **Windsurf Editor** 后，Codeium 更名为 **Windsurf**，其[品牌重塑公告](https://windsurf.com/blog/windsurf-rebrand-announcement)中对此进行了说明。
   - 新名称代表了人类与机器能力的融合，旨在创造强大的体验。
- **Windsurf 启动新的 SubReddit**：Windsurf 推出了新的 [SubReddit](https://www.reddit.com/r/windsurf) 以建立社区，同时也对其 Discord 服务器进行了调整。
   - 这些变化包括更新页面和重命名频道，以反映新的 **Windsurf** 品牌。
- **Codeium Extensions 获得新的 Plugin**：随着品牌重塑，**Codeium Extensions** 现在正式更名为 **Windsurf Plugins**，并承诺会有更多创新。
   - 公司重申了他们持续增强 **Windsurf Editor** 的决心。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区沉寂时间过长，请告知我们，我们将将其移除。

---

# PART 2: 频道详细摘要与链接

{% if medium == 'web' %}

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1358879398237044846)** (1134 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, OpenAI's Deep Research, Google's AI Strategy, DeepCoder-14B Preview Model, NightWhisper Model` 

- **Gemini 2.5 Pro 被誉为卓越模型**：成员们称 [Gemini 2.5 Pro](https://ai.google.com/models/gemini) 为第一个“真正”的 AI，并指出它在创意写作和一致性方面优于其他模型。
   - 一些用户观察到，虽然 **Gemini 2.5 Pro** 在通用任务中表现出色，但 **Nightwhisper** 在编程方面更胜一筹。
- **OpenAI 的 Deep Research 受到审视**：用户对 OpenAI 的 [Deep Research](https://openai.com/research/deep-research) 提出疑问，指出其作为网页搜索 Agent 的潜力，有人表示“带工具的 2.5 简直处于另一个水平”。
   - 然而，普遍共识认为 Deep Research 只是 OpenAI 现有的 o3 模型。
- **Together AI 发布 DeepCoder-14B 预览模型**：**Together AI** 和 **Agentica** 联合发布了 [DeepCoder-14B-Preview](https://www.together.ai/blog/deepcoder)，这是一个代码推理模型，*通过分布式 RL 基于 Deepseek-R1-Distilled-Qwen-14B 进行微调*。
   - 一位用户指出这是“有史以来最愚蠢、最可耻的营销”，称考虑到这只是 o3-mini，其提升并不令人印象深刻。
- **NightWhisper 模型的编程实力受到称赞**：用户们正热切期待 **NightWhisper** 的潜在发布，强调了它在竞技场（arena）中展示的编程能力，尽管它在 webdev 和 lmarena 上仅短暂可用。
   - 有人推测它与即将推出的 Google Ultra 模型相同。
- **O3 模型变体评价褒贬不一**：成员们对比了 OpenAI 的 **O3 Mini** 和 **O3** 模型，其中一人指出 **O1** 在决定思考时长方面比 **O3 mini** 更熟练。
   - 一位拥有 O3 medium 访问权限的用户形容它在语言相关问题上比 O1 更好，但在代码方面仍弱于 Gemini 2.5 Pro。

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1359224745370976438)** (1 messages): 

> `Alpha UI, Desktop & Mobile, Bugs, Leaderboard` 

- **Alpha UI 开放测试**：**Alpha UI** 现在已开放测试，**无需密码**，访问地址为 [https://alpha.lmarena.ai/](https://alpha.lmarena.ai/)。
   - 鼓励用户通过提供的 [Google Forms](https://forms.gle/8cngRN1Jw4AmCHDn7) 和 [Airtable](https://airtable.com/appK9qvchEdD9OPC7/pagxcQmbyJgyNgzPx/form) 链接提交反馈和 Bug 报告。
- **Alpha UI 更新迅速**：公告提到 **Alpha UI** 是一个功能有限的早期版本，但 **Desktop & Mobile** 端的更新正在快速推进。
   - 对于最新的模型和排行榜数据，用户应参考主站，这表明 Alpha 版本可能尚未完全同步最新数据。

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1358881886088204319)** (586 messages🔥🔥🔥): 

> `Unsloth DDP 支持, GGUF vs bnb LoRA 训练, Llama 4 分析, cogito-v1 预览版 LLMs` 


- **Unsloth 解决 DDP 训练问题**：一位用户报告了 **HF Trainer 和 DDP** 在使用 3 个或更多 GPU 时无法工作，但在 2 个 GPU 上运行正常的问题，但 [Unsloth 支持 DDP](https://docs.unsloth.ai/)。
   - 经过测试后，系统抛出了 ValueError，一名成员建议确保将 CUDA 可见设备设置为特定的 GPU。
- **bnb 是首选方案**：一位用户询问是针对微型模型在 **bnb 4-bit** 还是 GGUF 上训练 LoRA，得到的建议是使用 **bnb** (bitsandbytes) 进行 QLoRA 训练，因为这样可以节省 4 倍的数据下载量。
   - 一旦 Adapter 训练完成，可以将其保存并与 bnb 模型合并，然后导出为 GGUF。
- **Llama 4 模型获得“草率”的评价**：一位成员测试了 **Llama 4** (Scout 和 Maverick)，并提到它在日语方面表现良好，似乎是能力很强的 Base 模型，但 Post-training（后训练）做得比较草率。
   - 另一位成员评论说，他们将等待 Post-training 的彻底翻新。
- **DeepCogito 的 v1 预览版 LLMs 提出强力主张**：一位用户分享了 [DeepCogito 的 v1 Preview 模型](https://www.deepcogito.com/research/cogito-v1-preview)，声称其模型性能优于同尺寸的最佳开源模型，包括来自 LLaMA、DeepSeek 和 Qwen 的对应模型。
   - 他们声称每个模型都可以直接回答（标准 LLM），或者在回答前进行自我反思（类似 Reasoning 模型）。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1358950021415964873)** (21 messages🔥): 

> `iMatrix 动态上传, Apple BFloat, 模型剪枝, Online DPO` 


- **iMatrix 动态上传登陆 HF**：成员们将 [Llama-4-Scout-17B-16E-Instruct-GGUF](https://huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct-GGUF) 的 iMatrix 动态版本上传到了 HuggingFace。
- **BFloat 中的 B 代表 Brain**：根据 [Apple 的文档](https://developer.apple.com/documentation/metal/mtldatatype/bfloat?changes=_5_5&language=objc)，**bfloat** 中的 "B" 代表 "Brain"，该数据类型是由 Google Brain 开发的。
- **狂想理论 (Schizo Theory)**：一位成员分享了他的 *“狂想理论：像 OpenAI / Claude / Gemini 这样的公司会利用用户输入来剪枝（prune）他们的模型”*。
   - 他认为 *“你更喜欢哪一个”* 之类的响应是为了 *收集用户偏好数据以训练他们的模型*。
- **Online DPO 比你更了解你自己**：一位成员指出，Online DPO 开始比你自己更了解你。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1358885784073011441)** (175 messages🔥🔥): 

> `GraniteModel bug, Unsloth 在 MacOS 上运行, 多 GPU 支持, Gemma 3 12b 问题, GRPO 训练` 


- ****GraniteModel Bug 困扰 Colab 用户！****：用户在 Colab notebook 中使用 **GraniteModel** 时遇到了 Bug，快速修复方法是编辑 `granite_based/config.json`，将 **GraniteModel** 替换为 **GraniteForCausalLM** 并重新运行单元格。
   - 在 Colab 上编辑文件的推荐方法是下载文件、在本地编辑，然后将修改后的版本重新上传到 Colab。
- ****MacOS 错失 Unsloth 的 GPU 优势****：Unsloth 目前 **仅支持 GPU**，这导致没有 NVIDIA GPU 的 MacOS 用户会遇到 `NotImplementedError`。
   - 不过，[这个 Pull Request](https://github.com/unslothai/unsloth/pull/1289) 提供了一个潜在的解决方案，旨在解决 MacOS 的兼容性问题。
- ****多 GPU 支持即将推出！****：用户们正热切期待 Unsloth 微调的多 GPU 支持。
   - 团队给出的答复是 *“很快 (tm)”*。
- ****Gemma 3 12b 面临加载失败****：用户报告 `push_to_hub_merged` 没有将所有必要文件上传到 HF，因此无法使用 `AutoModelForCausalLM.from_pretrained("modelname/here")`，并报错 `OSError: modelname/here does not appear to have a file named pytorch_model.bin`。
   - 一位成员建议，如果你使用的是大于 1B 的 Gemma，它在技术上属于 Vision Language Model，因此某些地方略有不同。建议用户针对 Gemma 3 尝试 `FastModel` 而非 `FastLanguageModel`。
- ****寻求超大模型的 GRPO 训练技巧****：一位用户寻求关于使用 GRPO 训练具有 **16k** 上下文长度的 **24B** 模型的建议，在拥有 141GB VRAM 的 H200 上仅能维持 1 的 Batch Size，并询问了 Unsloth Pro 计划的多 GPU 支持情况。
   - 建议包括增加梯度累积（Gradient Accumulation），以及通过其他框架实现多 GRPO 支持的可能性，并讨论了关于采样效率的分布式 GRPO 概念。


  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1359245049044668467)** (2 条消息): 

> `地点澄清` 


- **地点并非法国**：一位成员询问另一位成员是否来自法国。
   - 该成员回应澄清他们来自 **荷兰（Dutch/Holland）**。
- **地点已确认**：该成员确认他们来自荷兰。
   - 这回应了最初的问题，明确了他们的原籍。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1358921030390911268)** (36 条消息🔥): 

> `LLMs 知识存储替代方案，用于内存卸载的 RAG，Vector DBs 与隐私，检索增强训练，DeepSeek-V3` 


- **LLMs 权衡知识存储替代方案**：成员们讨论了 LLMs **将知识检索卸载到 RAG 流水线**以减小模型体积并提高速度的潜力，以及训练注意力头（attention heads）与向量数据库（vector database）协同学习的可能性。
   - 有建议称 OpenAI 可以针对私有数据集提供**通用的向量数据库知识查找**，开源 LLM 内核可以接入这些数据以获取额外的上下文。
- **重新构想 RAG：检索部分的演进**：讨论围绕将 LLMs 拆分为**知识模型**和**对话模型**展开，其中对话模型专注于智能、推理以及对知识模型的工具调用。
   - 虽然这被类比为 RAG，但重点在于一个能与专家系统或基于相同 embeddings 构建的专用向量数据库协作的内核，从某种意义上有效地增加了词表大小。
- **向量数据库探索：隐私优势初现**：一位成员指出，OpenAI 可能会从免费提供开源内核中获益：“看看接入我们的注意力向量查找之前的基准测试。再看看接入之后的。”
   - 这也可能通过**仅卸载静态知识内存查找**来带来隐私方面的优势。
- **奖励式再训练：忘记那些能被高效记忆的内容**：一位参与者建议进行“检索增强训练”，**奖励模型去忘记**那些可以通过向量搜索高效记住的内容。
   - 这种方法可以通过在训练期间利用外部知识源来构建更高效的模型。
- **DeepCoder 优化详情**：一位成员分享了关于 [Together AI 博客文章](https://www.together.ai/blog/deepcoder) 的链接，内容涉及 **DeepCoder 优化**，强调了其在优化 vLLM 流水线方面的潜力。
   - 该优化通过在进行初始采样和训练的同时再次进行采样，从而最大限度地减少了采样等待时间。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1358909049588154488)** (5 条消息): 

> `速率限制，额度，Quasar 速率限制，关于速率限制的反馈` 


- **OpenRouter 调整免费模型速率限制**：拥有至少 **$10 额度**的账户，其每日请求数（**RPD**）将提升至 **1000**，而额度**少于 $10** 的账户，RPD 将从 **200** 降至 **50**。
- **Quasar 将获得基于额度的速率限制**：更新还提到 **Quasar** 很快将实施取决于额度的速率限制。
- **关于免费模型速率限制的反馈**：一位成员开启了一个[反馈贴](https://discord.com/channels/994043905957435544/1243614384297644072)供用户发表对这些变化的看法。
- **不提供每小时速率限制**：目前没有每小时速率限制，但速率限制为 **每分钟 20 次请求**。


  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1359349816118743130)** (2 条消息): 

> `Olympia.chat, Shopify, SaaS 营销, 交付即用业务` 


- **Olympia.chat 寻求新领导层**：[Olympia.chat](https://olympia.chat) 的创始人已就任 **Shopify** 的首席工程师（Principal Engineer），公司正在寻找一位经验丰富的站点运营者来接管技术维护和 **SaaS 营销**。
   - 该网站目前处于**盈利状态**，每月产生超过 **$3k 美元**的收入，创始人对于潜在接管的条款持灵活态度，提供包含所有知识产权（IP）在内的**交付即用业务（turnkey operation）**。
- **Olympia.chat 的财务表现**：尽管去年峰值接近 **$8k**，Olympia.chat 目前仍能稳定每月产生超过 **$3k 美元**的收入。
   - 资金缺乏导致营销活动停止，从而影响了客户流失率。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1358879753461043202)** (758 messages🔥🔥🔥): 

> `OpenRouter Frontend, Quasar Open Sourced, Free Model Rate Limits, API Keys Please, Gemini` 


- **OpenRouter 发布惊艳新前端**：OpenRouter 推出了一个非常酷的新前端，向 [clinemay](https://discord.com/channels/1091220969173028894/1195014798837043240/1358883684609953812) 致敬！
   - 一位用户开玩笑说，这看起来像是 *gpt-3.5 用大约 4 分钟做出来的网站*。
- **Gemini 模型处于顶级水平**：**Gemini 2.5 Pro** 与其他模型相比完全处于另一个层次，使其成为目前最强大的模型。
   - 一位用户指出，它的排名是 **1. gemini 2.5 pro** ... **10. 其他所有人**。
- **免费模型限制收紧，社区反应强烈**：OpenRouter 将免费模型的 Token 限制降低至 **50**，引发了用户的复杂反应，一些人对限制降低表示沮丧。
   - 一些用户觉得这就像是一个*付费墙*。
- **API Keys 获取更便捷**：用户现在在创建账户并充值后，可以轻松获取 **API key**，只需在右上角的下拉菜单中进入 keys 选项即可创建。
   - 一位社区成员表示：*我之前在询问关于 App 的事，以便帮你把 key 放在正确的位置，但不确定 Godot 是如何处理这个问题的*。
- **Nvidia 悄然发布 SOTA 级推理模型 Llama 3.1**：[Nvidia](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) 悄然发布了一个 SOTA 级的推理模型。
   - 这个新模型随手展示了它比 **Behemoth** 更出色。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1358879030627143810)** (762 messages🔥🔥🔥): 

> `Augment, Vector DB vs graph DB, Manus.im, Cursor C/C++ extension error, Model selection` 


- **Daniel Mac 转向使用图数据库（Graph DB）处理代码**：一位成员分享了 [Daniel Mac 的推文](https://x.com/daniel_mac8/status/1908332949251948808)链接，内容是关于使用 **图数据库** 进行代码查询。
- **Manus.im 耗尽额度**：一位用户报告称 [Manus.im](https://manus.im) 未能正确回答问题，并在单次提示词中耗尽了 **1000 个免费额度** 中的 **984 个**。
   - 另一位成员建议探索其他替代方案，如 [Smithery.ai](https://smithery.ai/) 或 [Awesome MCP Servers](https://github.com/punkpeye/awesome-mcp-servers)。
- **C/C++ 扩展错误**：一位用户报告称，自 2023 年 3 月 Cursor 发布以来一直使用它，但最近收到了与 **C/C++ 扩展** 相关的错误，该扩展可能仅限于在 Microsoft 产品中使用。
   - 解决方法包括 [回退到之前的版本](https://forum.cursor.com/t/c-c-extension-usage-restriction-message-appears-in-cursor/75902)，用户还分享了讨论该问题的[其他论坛帖子](https://forum.cursor.com/t/c-c-extension-broken/75182)。
- **自动选择（Auto-Select）是个骗局**：用户报告称 **auto-select** 模型选项正在选择一些垃圾模型。
   - 一位用户声称它*搞砸了我的代码库*，而另一位用户则认为这是故意这样设计的。
- **Cursor 免费层级引发争议**：一位成员报告称，绕过 Cursor 的试用版可能会导致用户被完全禁止使用 Cursor。
   - 一位用户指出：*现在要封禁你了，但只是想让你知道，我希望你之前并不喜欢用 Cursor，因为你很快就完全无法使用它了。*


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1358908572959900011)** (158 条消息🔥🔥): 

> `Llama 4 令人失望、GPU 需求与模型大小、LM Studio 与 Ollama、Jinja 模板` 


- **Llama 4 性能令用户失望**：用户对 **Llama 4** 的性能表示失望，称其表现*糟糕*且*倒退了 10 步*，并质疑基准测试的有效性。
   - 其他人认为，由于**随机数据**、**连接过多**或**数据集污染**，较大的模型可能存在质量控制问题，而 Qwen 的 **14B** 模型则受到了称赞。
- **LLM 尺寸与硬件影响**：讨论涉及 **VRAM 消耗**与模型稀释之间的关系，一些人指出，消耗较少 VRAM 的模型通常看起来经过了更多的蒸馏或稀释以减小尺寸。
   - 一位用户澄清说，**Llama 4** 的结果与 **24-27B** 模型相似，但具有 **17B** 模型的速度和成本，然而它需要更多的 VRAM，这使得它对普通用户来说毫无意义。
- **LM Studio 的远程 GPU 兼容性引发讨论**：用户讨论了将 **LM Studio** 连接到 **Ollama** 远程实例的可能性，但已确认 **LM Studio 与 Ollama 不兼容**。
   - 此外，还提出了将 LM Studio 与远程 GPU 集群连接的可能性，并讨论了 **Snapdragon X Series** NPU 的使用及其在 LM Studio 和 llama.cpp 中的（缺乏）支持。
- **Cogito 模型的 Jinja 错误通过 ChatGPT 修复**：用户报告了在使用 **cogito-v1-preview-llama-3b** 时遇到的 **Jinja 模板**错误，并被建议使用 **ChatGPT** 快速修复模板。
   - 社区模型维护者已收到关于模板异常的通知，预计将更新模型。
- **为小白解析 MoE 模型**：一位用户询问：*什么是 MoE 模型？*
   - 一位热心成员解释说，**Mixture of Experts (MoE)** 模型可以比稠密模型更快，因为每个 token 只有部分模型处于激活状态，尽管整个模型必须加载到 VRAM 中。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1358890575981449236)** (398 条消息🔥🔥): 

> `Docker 差评、AMD ROCm WSL 困境、内存限制与主板、Umbrella Rack 超级计算机、快速阅读技巧` 


- **Docker 遭到吐槽**：在一名成员表示想与任何说 **Docker** 坏话的人成为*“好朋友”*后，另一名成员开玩笑地问：*“Docker 是对你的家人做了什么吗？”*。
   - 第一位成员幽默地回答道：*“我的心理医生说我不应该谈论这件事。”*
- **ROCm 在 WSL 上对 7800XT 仍有问题**：一位用户报告称，由于缺乏 [AMD 官方文档](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/compatibility/wsl/wsl_compatibility.html)中所示的支持，通过 WSL 运行的 ROCm 无法在 **7800XT** 上工作。
   - 尽管如此，另一位用户建议它*可能*可以工作，因为两张显卡都是 **RDNA3** 架构，而第一位用户确认半年前由于 WSL 透传问题，根本*无法*运行。
- **内存限制辩论**：在关于 **RAM** 限制的讨论中，一位用户指出 **Ryzen 7000** 的内存控制器较弱，消费级硬件的 **BIOS** 限制为 **192GB**，而主板可以容纳 **256GB**。
   - 另一位用户指出 **AMD 官网**标注的限制是 **128GB**，对此第一位用户回应称人们运行 **192GB** 已经很多年了，并将这种差异归因于服务器硬件具有不同的质量目标。
- **组装 NND Umbrella Rack 超级计算机**：一位用户提议使用 **RTX 4090 D GPU**（总计 3TB VRAM）或较低性能选项（1.5TB VRAM）构建一台 **16 节点超级计算机**，目标是在比 **Nvidia DGX B300** 更便宜的预算内运行具有 **1M 上下文的 2T 模型**。
   - 怀疑者质疑其可行性，一位用户直言不讳地表示：*“这不是这么搞的……”*，并强调需要 **RDMA**、快速互连和资深工程师，强调该用户的目标在目前的硬件上是不可能实现的。
- **语言模型微调教学项目**：一位成员询问关于涉及高性能硬件（**2 台 RTX ADA 6000**，**512GB RAM**）的有趣且具有教育意义的项目，并询问学习微调像 **phi4** 这样的小型实例是否是个好主意。
   - 另一位成员建议从头开始**预训练 LLM** 或**微调 LLM**，并指向了来自 Nvidia 的编码数据集 ([huggingface.co](https://huggingface.co/datasets/nvidia/OpenCodeReasoning))，并建议微调基础模型（Base Model）而非指令模型（Instruct Model）会更好。


  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1359276120368742472)** (1 条消息): 

> `Perplexity for Startups program, API Credits, Enterprise Pro` 


- **Perplexity 推出创业公司计划**：Perplexity AI 正在推出一项 [创业公司计划](https://www.perplexity.ai/startups)，提供资源帮助创业公司减少研究时间并专注于构建。
   - 该计划为整个团队提供价值 **$5000** 的 Perplexity API 积分和 **6 个月** 的 Perplexity Enterprise Pro；申请资格要求融资额少于 **$20M**，成立时间少于 **5 年**，且与 Startup Partner 有关联。
- **创业公司计划详情**：Perplexity for Startups 计划旨在为符合条件的创业公司提供加速发展所需的资源。
   - 符合条件的创业公司可以获得 **$5000** 的 API 积分和为期 **6 个月** 的 Perplexity Enterprise Pro 订阅，使其整个团队能够访问先进的 AI 能力。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1358880405138178180)** (453 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro performance, Deep Research High rollout, Perplexity Discover tab, Manus Invites are still needed, AI image generation on Android` 


- **Gemini 2.5 Pro 缺失推理输出**：成员们讨论了 **Gemini 2.5 Pro** 不公开其推理 token，因此无法作为推理模型包含在 Perplexity 中，尽管它是一个*高延迟思考模型*。
   - 由于 **Gemini 2.5 Pro** 的推理 token 未被发送，通过 API 使用 Perplexity 时不会像在 AI Studio 中那样显示推理过程，但它仍然是一个*高延迟思考模型*。
- **Deep Research High 推出缓慢**：成员们正热切期待 **Deep Research High** 的推出，预计该功能平均将使用 **150-200 个来源**，然而，一位用户报告称 *Perplexity 的深度研究获取了 23 个来源，而免费的 Gemini 深度研究获取了超过 500 个*。
   - 一些成员对发布时间表缺乏沟通以及当前版本输出摘要而非进行真正的深度研究表示沮丧。请查看 [DeepSeek Subreddit](https://www.rxddit.com/r/DeepSeek/s/zFUYlP8NeV)。
- **Gemini 2.5 Pro 在 Perplexity 上表现出色**：用户注意到 Perplexity 中新增了 **Gemini 2.5 Pro**，一位用户发现 **Gemini 2.5 Pro** 创作的一个故事击败了 **GPT 4.5** 创作的其他 3 个故事，另一位用户表示它现在正在驱动深度研究，并交付了像 [这份报告](https://cdn.discordapp.com/attachments/1047649527299055688/1359301884208480266/DR_2.5_Pro.pdf?ex=67f7a4c7&is=67f65347&hm=400f4c8d943d0887565453ecb42690e59499500547b64395af83ad45cadd3916&) 一样的详细报告。
   - 然而，一位用户指出，尽管模型生成了 **16,098 个 token** 的详细报告，但答案经常在 **500-800 个 token** 处被截断。
- **用户报告 Perplexity 自动启用 Pro 模式**：几位用户报告称 Perplexity 正在对免费用户自动启用 Pro 模式，以消耗其每日额度。
   - 一位用户表示非 Pro 模型似乎很*烂*。
- **上传 PDF 文件时报告的问题**：Pro 用户尝试上传 8 个 .pdf 文件，加载 5 分钟后，要么只上传了一两个，然后立即消失并弹出错误提示 *文件上传失败*。
   - 文件大小范围从 **114kb 到 9,502kb**


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1359158847511597308)** (1 条消息): 

> `Llama 4, Benchmark Faking` 


- **针对 Llama 4 的基准测试造假指控**：一位用户分享了一个 [Perplexity AI 搜索结果](https://www.perplexity.ai/search/does-llama-4-fake-benchmarks-pw9wkBJ4TCOUtdZu8fmTdg#0)，质疑 **Llama 4** 是否在基准测试中造假。
   - 分享的链接提供了关于 **Llama 4** 涉嫌操纵基准测试的相关讨论和潜在证据。
- **关于模型基准测试的持续辩论**：对话突显了 AI 模型基准测试中透明度和可靠性的更广泛问题，这是 AI 社区中一个反复出现的主题。
   - 人们对用于评估 **Llama 4** 的方法论以及产生误导性结果的可能性表示担忧。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1358887341116100780)** (29 条消息🔥): 

> `Perplexity API 新闻抓取, Perplexity API Sonar 提示词编写, Perplexity API 搜索差异, Perplexity API 引用, Perplexity API 沙盒` 


- ****Perplexity API** 新闻抓取**：一位用户请求增加新闻 API 功能，以便根据查询或主题抓取新闻，类似于 [particle.news](https://particle.news)，团队回应称他们已经建立了合作伙伴关系，通过其 API 提供新闻。
   - 一名团队成员建议利用 **Sonar** 现有的功能构建新闻 API 特性，并将其添加到 [API cookbook](https://github.com/ppl-ai/api-cookbook) 中。
- ****Perplexity API** Sonar 提示词编写**：一位用户报告了 **Sonar** 的问题，即响应内容过度集中在系统提示词上，而不是动态处理用户查询。
   - 一名团队成员澄清说，在搜索阶段不会使用系统提示词，建议用户参考 [Prompt Guide](https://docs.perplexity.ai/guides/prompt-guide) 优化 **用户提示词 (user prompt)**。
- ****Perplexity API** 搜索差异**：一位用户报告了 **Perplexity API** 与 **网页端 UI (web UI)** 在总结网页时的差异，API 无法检索某些链接，且结果的结构化程度较低。
   - 该用户正在寻求帮助以解决这些问题，因为 **Sonar-pro** 模型在 **API** 和 **网页端 UI** 之间产生的结果不一致。
- ****Perplexity API** 沙盒更优？**：一位用户报告称，在使用 **sonar-reasoning-pro** 时，**API 沙盒 (sandbox)** 给出的结果比实际 API 好得多。
   - 用户正在寻求建议，如何让 **API** 提供与 **沙盒** 相同的结果。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1358886363457388676)** (463 条消息🔥🔥🔥): 

> `高强度模式, Manus 本地版本, Genspark vs Manus, Llama 4 炒作, Manus 额度使用` 


- **本地版 Manus 即将推出**：成员们讨论了未来将可能推出 **Manus** 的本地版本，就像大多数其他 **AI 模型** 一样。
- **MCP 服务器可用**：成员们注意到自 2024 年 11 月 25 日起，**Claude** 已提供 **MCP 服务器**，并可与 Claude code 配合使用。
   - 一些成员表示怀疑，并引用了过去在他们所谓的“诅咒模型合并 (cursed model merging)”方面的成功尝试。
- **Llama 4 被过度炒作**：用户分享称，在 **Openrouter.AI** 上测试并收到欠佳的响应后，认为 **Llama 4** 被过度炒作了。
   - 其他人声称 **Zucks** 受到批评是因为他据称在 **基准测试 (benchmarks)** 中作弊。
- **Octopus 网页抓取工具效果良好**：一名成员报告称，免费网站抓取工具 [Octopus](https://octoparse.com/) 在 Zillow 和 Realtor 上运行良好，而 Bardeen 每月收费 130 美元。
   - 另一名成员表示，既然可以用 **Manus** 构建自己的工具，每月 130 美元似乎太贵了。
- **Manus 额度太贵**：几位用户抱怨 [Manus 额度](https://www.manus.im/pricing) 太贵，一名用户报告称一个“标准复杂度”的任务就用完了所有 1000 个免费额度，希望定价和额度方案能得到更新。
   - 一些用户分享说，最好将其拆分为带有新对话窗口的小任务，并推荐 **Proxy** 作为更便宜的替代方案。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1358879728995668009)** (237 messages🔥🔥): 

> `Gemini 2.5 vs Sonnet Thinking, Aider's auto-testing, Gemini 2.5 Pro vs exp, OpenRouter citation links, AI resume builder` 


- **Gemini 2.5 与 Sonnet 对决**：成员们讨论了 **Gemini 2.5** 强大的逻辑但较差的指令遵循能力，对比 **Sonnet** 功能丰富但代码准确性略低的表现，最终后者在生成可用程序时所需的 prompt 更少。
   - 一位用户报告称 **Gemini 2.5** 只需要 1 个 prompt，而 **Sonnet** 需要 3 个 prompt，尽管 Sonnet 实现了*多种文件输入方法、可选的拖放、批量处理、文件队列管理、显式转换启动、显式取消、可调节窗口等*。
- **Aider 自动测试（Auto-Testing）困扰**：一位用户正寻求启用 **Aider 的 auto-testing** 并可能禁用 **auto-committing**，因为存在提交未测试代码的问题，并指向了 [Aider 配置选项](https://aider.chat/docs/config/options.html)。
   - 另一位用户建议提供 [model 和 key](https://aider.chat/docs/troubleshooting/models-and-keys.html)，否则 Aider 会根据它能找到的任何 key 来猜测你想要的内容。
- **Gemini 2.5 Pro exp 速率限制**：用户对比了 **Gemini 2.5 Pro exp** 和 **Gemini 2.5 Pro preview**，注意到不同的速率限制（rate limits），并且有人报告看似免费的 `pro-exp` 模型产生了费用。
   - 尽管一位用户觉得 exp 版本较弱，但另一位用户在试用 **Sonnet** 一小时内就取消了订阅，还有用户在两者上都遇到了速率限制问题，尤其是在通过 OpenRouter 使用时。
- **OpenRouter 缺失引用链接**：一位用户询问在使用 **OpenRouter** 上的 **Perplexity Sonar Pro** 等服务时，缺失引用链接（citation links）是否正常，并附带了一张[图片](https://cdn.discordapp.com/attachments/1131200896827654149/1358926629170319490/image.png?ex=67f798cc&is=67f6474c&hm=fe2c340b866bec81e485bbed3c2d1fe17071b540d6ea5c803306211e3d9f2ceb&)。
- **DIY AI 简历生成器创意**：一位用户正在寻找一种 **LLM 驱动的工具**，用于根据职位列表分析简历并建议措辞修改，另一位用户建议自己构建一个工具。
   - 一位用户建议，如果具备一定的编程经验，就可以构建这个工具，并且还可以用它来测试 Gemini 2.5 pro。


  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1358969628058976366)** (8 messages🔥): 

> `Architect mode interruptions, Aider Response Time, Aider Cursor Rules` 


- **Architect 模式编辑被截断？**：用户报告在被要求添加新文件时，**/architect 模式**的编辑会被中断，可能会丢失编辑器状态。
   - 拒绝添加新文件似乎可以让编辑继续进行。
- **Aider 响应时间受到质疑**：用户报告在 **Aider v0.81.1** 中使用 `openrouter/deepseek/deepseek-r1` 和 `openrouter/anthropic/claude-3.5-sonnet` 的速度和 ChatGPT 一样慢。
   - 一位用户为了创建一个 schema 文件等待了“整整 5 分钟”，结果却因为连接问题收到了 `litellm.APIError`。
- **比较 Aider 约定与 Cursor 规则**：用户询问 Aider 的约定（conventions）是否与 [Cursor rules](https://roman.pt/posts/cursor-under-the-hood/) 类似。
   - 一位成员澄清说 *aider 的 "conventions" 并不是一个特定的功能*，而只是发送给 LLM 的附加上下文。它们可以通过手动添加或使用 `--read CONVENTIONS.md` 来实现。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1359034687086923989)** (8 messages🔥): 

> `Software Engineer Gap Year, LLMs as AI Coworkers, Programming LLMs for Successful Outcomes` 


- **软件工程师休间隔年不是个好主意？**：一篇文章指出，对于软件工程师来说，现在休间隔年/度假是一个极其糟糕的决定/时机，并指出了对当前技术格局的见解，详见[这篇文章](https://ghuntley.com/dothings/)。
- **LLM 作为 1000 个 AI 同事**：来自 Anthropic 的 Anni Betts 建议，软件工程师不应只考虑拥有“一个 AI 同事”，而应考虑拥有“1000 个同时疯狂处理你整个待办事项列表的 AI 同事”。
   - 根据作者的说法，这可以通过[对 LLM 进行编程](https://ghuntley.com/stdlib/)并构建一个能够产生成功 LLM 结果的 “stdlib”（标准库）来实现。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1358936328233943241)** (10 messages🔥): 

> `NotebookLM 商业选项、NotebookLM 隐私保证、NotebookLM 误读学术文章` 


- **Google 的 AgentSpace 为企业解锁 NotebookLM**：一位用户询问了具有数据隐私和特定编程能力的商业规模版 **NotebookLM**，另一位成员链接到了 [Google 的 AgentSpace NotebookLM Enterprise 文档](https://cloud.google.com/agentspace/notebooklm-enterprise/docs/set-up-notebooklm#cmek)，该文档支持 **CMEK**。
   - 文档概述了如何使用 **Customer-Managed Encryption Keys (CMEK)** 设置 NotebookLM，从而对数据加密提供更大的控制权。
- **NotebookLM 提供的隐私保证**：一位成员解释说，**Enterprise** 和 **Plus** 版本的 **NotebookLM** 都提供隐私保证，强调无论使用哪个版本，用户数据都不会进入公共领域。
   - 他们澄清这一点是为了解决对 **Google 隐私政策**和**服务条款**的根本性误解，并进一步暗示该平台具有防止 prompt injection 尝试的机制。
- **用户更正后 NotebookLM 改进了摘要**：一位用户注意到 **NotebookLM** 最初误读了学术文章摘要中的一个关键点，但在用户提供引用和解释后进行了自我修正。
   - 据该用户称，在不同的 **Google 账号**中使用相同的文章重复相同的 prompt，从一开始就得到了正确的摘要，这引发了关于模型是否使用之前的查询进行训练以及隐私声明是否准确的疑问。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1358885964021498016)** (204 messages🔥🔥): 

> `Discovery Mode 推出、Google Cloud Next 和 Google I/O、NotebookLM 法律用例、Gemini 新功能与 deep research、Podcast Audio Overviews` 


- **Discovery Mode 仍处于缓慢推出阶段**：用户报告正在等待新的 **Discovery Mode** 功能，预计从发布之日起，推出过程将长达**两周**。
   - 一位用户开玩笑地要求作为 *Google 铁粉获得特殊待遇*，请求成为 alpha 测试员。
- **Google Cloud Next 和 Google I/O 承诺带来惊喜**：即将举行的 **Google Cloud Next** 和 **Google I/O** 活动预计将揭晓新功能，尽管细节仍处于严格保密状态。
   - 一位用户幽默地将 Cloud Next 比作*圣诞节*，而 Google 扮演圣诞老人的角色。
- **NLM 用于法律用例及打印问题**：一位用户寻求关于使用 NotebookLM 从法律文件中提取特定信息的建议，旨在获取条款编号和相关文本，并寻求关于打印包含所有链接的完整答案的帮助。
   - 另一位成员建议将内容拆分为 **10-20 个 notebooks**，每个 notebook 包含特定内容，以便在每个 notebook 中立即询问相同的问题。
- **Gemini 的 deep research 仍存在幻觉**：一些用户报告称，尽管 **Gemini** 的 **deep research** 可以访问互联网，但仍会出现*幻觉 (hallucinations)*。
   - 一位成员澄清说，**Gemini** 可以连接到 Google Search，但如果你不指定要对其进行 grounding，它就不会这样做，并建议在 **AI Studio** 中进行测试。
- **Podcast Audio Overviews 即将登陆 NotebookLM**：据报道，新的 **2.5 Pro deep research** 版本将具备制作 **audio overviews** 的能力，但并非对所有用户都可用。
   - 一位 Google 员工澄清说，如果来源中围绕一个中心话题涵盖了多个不同的角度，复杂的话题会导致生成更长的 podcast。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1358880373685354668)** (92 messages🔥🔥): 

> `DeepSeek R2 Release, LlamaCon, Llama-4-Maverick, Style Control Ranking, HF version of Llama-4-Maverick` 


- **DeepSeek R2 必须在 LlamaCon 期间发布**：成员们鼓励与 **DeepSeek** 有联系的人在 **LlamaCon** 当天发布 **R2** 以利用热度，并引用了 [arxiv.org](https://arxiv.org/abs/2410.19034) 的研究，指出 **MoE** 所需的训练数据与基础模型不同。
- **LM Arena 政策更新**：早期分析显示，**风格和模型回复语调**是一个重要因素（在风格控制排名中有所体现），**Llama-4-Maverick** 的 HF 版本正被添加到 Arena。但 **Meta** 应该更清楚地说明 *Llama-4-Maverick-03-26-Experimental* 是一个为优化人类偏好而定制的模型，因此排行榜政策正在更新，以强化对**公平、可重复评估**的承诺。
   - 成员们反应说这是一个*废话连篇且表情符号泛滥的垃圾内容 (slopfest)*。
- **Cogito 模型以开源许可证发布**：强大的 **LLM（尺寸包括 3B, 8B, 14B, 32B 和 70B）** 正以开源许可证发布，每个模型在大多数标准基准测试中都优于 **LLaMA, DeepSeek, 和 Qwen** 同尺寸的最佳开源模型，且 **70B 模型** 优于新发布的 **Llama 4 109B MoE 模型**。
   - 这些 **LLM** 使用 **Iterated Distillation and Amplification (IDA)** 进行训练，这是一种可扩展且高效的超人工智能对齐策略，利用来自 [DeepCogito](https://huggingface.co/collections/deepcogito/cogito-v1-preview-67eb105721081abe4ce2ee53) 的迭代自我提升。
- **Together AI 进军训练业务**：**Together AI** 正在进入训练业务，如该 [案例研究](https://www.together.ai/models/cogito-v1-preview-llama-70b) 所示。
- **Google Gemini 2.5 Pro Deep Research 发布**：根据 [9to5Google](https://9to5google.com/2025/04/08/gemini-2-5-pro-deep-research/)，**Google Gemini 2.5 Pro Deep Research** 已发布，一名成员报告称 Gemini 2.5 deep research 与 **OpenAI Plus** 大致相当，并带有一个音频概览播客选项之类的功能。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1359001640761294910)** (30 messages🔥): 

> `OpenAI Image Gen Capabilities, Logprob Reward, Arxiv Publishing, Arxiv Moderation, Phi-CTNL` 


- **OpenAI 的新图像生成能力**：一位成员询问了描述 **OpenAI** 如何解锁新图像生成能力的报告，暗示这并非新模型，而是*潜在能力 (latent capabilities)*。
   - 另一位成员建议这是通过使用类似于 [这个](https://discord.com/channels/1179127597926469703/1208183216843005962/1358810240627376259) 的目标函数并结合 [这篇论文](https://arxiv.org/abs/2503.19618) 中提到的 *logprob reward* 实现的。
- **Arxiv 发布流程揭秘**：成员们讨论了在 **Arxiv** 上发布的流程，指出需要*少量的背书 (vouch)*，这与过去物理学界直接倾倒内容的做法不同。
   - 他们补充说存在*极小的随机审核概率*，即使是胡言乱语的论文也可能被拒绝，但大多数人还是照发不误。
- **冠军 Phi-CTNL 论文有 20 次引用**：一位成员分享了 [这篇论文](https://arxiv.org/abs/2309.08632) 的链接，描述了一个拥有 20 次引用的冠军模型，惊叹道：*神级 (Godlike)。*
   - 另一位成员注意到了这个*残暴的*模型名称 **phi-CTNL**，并推测如果 **Meta** 在未来的 **Llama 4** 论文中引用它会引发什么反应。


  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1359173566054142232)** (15 messages🔥): 

> `Google AI Staff, AI Sabbatical, NVDA Tariffs, ASI, Google's management vibes` 


- **Google 据称向 AI 员工支付薪酬让他们无所事事**：一篇 [TechCrunch 文章](https://techcrunch.com/2025/04/07/google-is-allegedly-paying-some-ai-staff-to-do-nothing-for-a-year-) 讨论了 **Google** 据称如何向部分 **AI staff** 支付为期一年的薪水让他们无所事事，而不是让他们加入竞争对手。
   - 一位成员将其描述为 *管理层最基本的想法，但其所有二阶效应都极其糟糕*。
- **AI 工程师渴望休假去“看树”**：一位成员表示，在 AI 领域一两年内稳定下来后，他们会很乐意休个 **sabbatical**（学术休假）并写一本书。
   - 另一位成员表示，*从企业端（如 McKinsey 等）来看，他们会给研究人员很长的假期以确保他们保持投入，否则你会发现随着时间的推移，人才都会流失*。
- **关税可能加速 AI 降温**：一位成员建议，如果 **tariffs**（关税）持续存在，AI 领域将在一个月内冷静下来，因为人们负担不起 **NVDA GPUs**。
   - 他们指出 *如果关税持续，一个月内就会冷静下来，因为你买不起 NVDA GPUs*。
- **Google 向离职者支付薪水引发法律风险**：一位成员澄清说，**Google** 正在向已经离职的人支付额外一年的薪水，但强迫他们不准工作。
   - 理论上，他们在这一年里做的任何事情都属于 **Google**，因此他们在不面临法律风险的情况下，无法开始经营自己的 Startup 或进行其他项目。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1359020372875284723)** (12 messages🔥): 

> `Google Cloud Next, Qwen 3 Launch, GPT 4.5 preferences, Claude Code Credits, Tim Apple` 


- ****Google Cloud Next** 将发布新模型**：一位成员分享道，根据 [这条 X 帖子](https://x.com/OfficialLoganK/status/1909443890366890200)，Google 将在周三开始的 **Cloud Next** 上发布新模型。
   - 这可能意味着 **Qwen 3** 的发布。
- ****GPT 4.5** 偏好测试正在进行中**：一位成员暗示 OpenAI 正在收集 **GPT 4.5** 的偏好，并链接到了 [这条 X 帖子](https://x.com/phill__1/status/1909623249563959551)。
   - 他们正在寻找 *High Taste Tester LMarena* 来发表意见。
- ****Anthropic** 提供 **Claude Code Credits****：一位成员分享了一个链接，**Anthropic** 向 1,000 名试用 **Claude Code** 的用户提供 [$50 的 Claude Code credits](https://www.anthropic.com/contact-sales/claude-code-credits)。
   - 根据帖子内容，这些额度可能只够 *修改一个变量名*。


  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1358978645632352450)** (5 messages): 

> `Jiankui He's X ad revenue` 


- ****Jiankui He** 的 AdSense 财富**：一位用户开玩笑说 **Jiankui He** 从 [X 创作者广告分成](https://x.com/Jiankui_He/status/1909417417396437200) 中赚了大钱。
   - 另一位用户调侃说他从中赚了 *$20*，或者如果 **Elon Musk** 想“搞他”的话，能赚 *$20K*。
- **AdSense 收益推测**：关于 **Jiankui He** 在 X 上可能获得的广告收入引发了猜测。
   - 估计范围从微薄的 *$20* 到更可观的 *$20,000*，具体取决于 **Elon Musk** 的干预。


  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1358999578468417588)** (24 条消息🔥): 

> `DAPO 论文, OLMo, Tulu 3, BoN Sampling` 


- **DAPO 论文提供“极端价值”**：频道成员讨论了 [一篇 DAPO 论文](https://arxiv.org/abs/2504.05118)，认为其提供了“极端价值”。
   - 他们还引用了 [另一篇基于 OLMo 的论文](https://arxiv.org/abs/2504.04022)。
- **Tulu 3 的工作被另一篇论文采用**：提到并链接了一篇使用 **Tulu 3 工作** 的论文：[https://arxiv.org/abs/2504.03790](https://arxiv.org/abs/2504.03790)。
- **研究中的 Alpha = 与其他研究人员交流**：一位成员表示，“研究中最大的 Alpha 就是与其他研究人员交流”，并分享了一篇论文的见解，指出该论文“使用了一种非常不同的推理时计算（inference time compute）方法”。
   - 他们还指出，该论文“表明 **BoN sampling** 实际上只是改变了 **RLHF** 中的 beta 因子（降低了 KL 惩罚）”，并且“你可以以不同的方式设计推理时计算，这样你就不是在 hack RL（在这种情况下使用 RM 作为引导），从而获得更好的答案”。
- **BoN sampling 会在未来的工作中替代吗？**：一位成员询问 **BoN sampling** 是否可以在未来的工作中作为替代。
   - 另一位成员回答说，“虽然实现起来更复杂，但如果 FLOPs 等效的话，当然没问题”。
- **Ash Vaswani 关于本科技术报告的推文**：一位用户分享了 [Ash Vaswani 的推文](https://x.com/ashVaswani/status/1909642828554387675)，并表示链接的论文“感觉不太好”。
   - 该成员表示这篇论文“感觉非常像本科生的技术报告”，但拒绝在推特上发表负面评论。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1358891406877266041)** (3 条消息): 

> `Karan Dalal 的帖子, Yuxi Liu 的文章` 


- **Karan Dalal 的帖子走红**：一位成员分享了 [Karan Dalal 在 fxtwitter 上的帖子](https://fxtwitter.com/karansdalal/status/1909312851795411093?s=61) 链接，引发了热烈讨论。
   - 原帖作者的反应仅仅是“WTF”。
- **Yuxi Liu 的文章引起关注**：一位成员发布了 [Yuxi Liu 的文章](https://yuxi-liu-wired.github.io/essays/posts/cyc/) 链接，立即引发了讨论。
   - 没有提到关于该文章的具体细节。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 条消息): 

natolambert: 我的帖子在 Marcus 的对比下显得很慷慨，天哪
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1358920108726030640)** (106 messages🔥🔥): 

> `Adam second-moment estimate buffers, Google DeepMind Patents, Hierarchical Perceiver, AI Auditing Survey, GFlowNets` 


- **Adam 缓存重建讨论浮现**：成员们讨论了 **Adam second-moment estimate buffers**（Adam 二阶矩估计缓存）的效用，以及如何为开源模型高效地重建这些缓存，在准确性与计算成本之间取得平衡，以寻求潜在的方法改进。
   - 有人指出，将 **beta2** 设置为极高值（例如 0.999999）并将学习率设为零可以提高准确性，尽管预训练的最后一个 epoch 仍存在挑战。
- **DeepMind 为 Hierarchical Perceiver 申请专利**：成员们注意到 [Google DeepMind](https://www.freepatentsonline.com/y2025/0103856.html) 正试图为 **Hierarchical Perceiver** 申请专利，并将专利图示与原始 [研究论文](https://arxiv.org/abs/2202.10890) 中的图示进行了对比。
   - 一些人推测该专利可能与 DeepMind 在 Gemini 中实现的 **ultra-long context lengths**（超长上下文长度）有关，并讨论了这究竟是一种防御性措施，还是该技术在最初未被广泛采用后目前实际应用情况的体现。
- **许可证对决：Apache 2.0 胜过 MIT**：对话中提到相比 MIT，人们更倾向于 **Apache 2.0 许可证**，理由是它能防御机器学习领域的专利诉讼。
   - 讨论强调，机构惯性和 GitHub 组织设置更青睐 Apache 2.0，观点认为 *除了 GPLv2 的怪癖或想要进行法律诉讼纠纷外，没有理由支持 MIT 而非 Apache 2.0*。
- **传闻 DeepMind 在模型发布中保留实力 (Sandbagging)**：成员们讨论了一个来自 [Reddit 帖子](https://old.reddit.com/r/LocalLLaMA/comments/1jp1555/deepmind_will_delay_sharing_research_to_remain/) 的传闻，称 **DeepMind** 可能会延迟发布研究成果以保持竞争优势。
   - 一位参与者澄清说，**sandbagging** 指的是 *在能力上有所保留*，而不是故意发布糟糕的模型版本来误导他人。
- **调查征集 AI 审计专家**：芬兰图尔库大学的一位研究人员正在进行一项关于生成式 AI 系统伦理审计的调查，寻求具有 AI 审计、模型评估、风险/合规或 AI 原则伦理对齐实际经验的专业人士参与。
   - 该 [调查](https://link.webropolsurveys.com/S/AF3FA6F02B26C642) 旨在收集关于审计或评估 AI 系统（尤其是生成式模型）的见解。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1358891732053528637)** (35 messages🔥): 

> `QKNorm, Soft RL, Llama 4 Memorization, Critical Batch Size, Reward, Value, Q-value letters` 


- **QKNorm 进展被认为存疑**：一位成员推荐了一篇 [更好/更早的论文](https://arxiv.org/abs/2502.00919)，并表示 **QKNorm 进展** 并非正确的发展方向，参考了 [这篇论文](https://arxiv.org/abs/2503.05453)。
- **Soft RL 目标总结**：一位成员总结道，**Soft RL** 的目标 *是学习一种策略，该策略不仅知道对每个查询的良好响应，而且理想情况下知道对每个查询的所有良好响应。*
   - 他们链接到了 [test-time-training.github.io/video-dit/](https://test-time-training.github.io/video-dit/) 和 [这条推文](https://x.com/karansdalal/status/1909393559981375574)，同时提到了线程块集群（thread block clusters）。
- **Llama 4 在 MATH-Perturb 上表现不佳**：在关于衡量 **Llama 4** 模型对测试集记忆化（memorization）的讨论中，一位成员表示 *它在 MATH-Perturb 数据集上表现相当糟糕*，并链接到了 [这条推文](https://x.com/KaixuanHuang1/status/1909387970773234088)。
- **临界批量大小 (Critical Batch Size) 受到评议**：针对 *极大的批量大小不利于收敛* 的观点，一位成员引用了关于临界批量大小的标准 **McCandlish 论文** 来支持该论点，并链接到了 [这篇论文](https://www.cerebras.ai/blog/training-multi-billion-parameter-models-on-a-single-cerebras-system-is-easy)。
- **R 代表 Return，Redditor 如是说**：一位成员开玩笑说 *总有一天 LLM 研究人员会从 R、V 和 Q 中使用正确的字母分别代表 Reward、state-value 和 state-action values，但不是今天*，同时链接到了 [这篇论文](https://web3.arxiv.org/abs/2503.19037)。
   - 另一位成员回应道 *脑筋急转弯，R 代表 Return*，并附上了 [这篇论文](https://arxiv.org/abs/2504.01928v1) 的链接。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1358894123033170021)** (9 messages🔥): 

> `Baranuik and Balestriero's works, ReLU networks, Boris Hanin's ReLU networks paper, ICML machine unlearning workshop` 


- **超平面神经网络规避过拟合**：有人指出，由于 **ReLU neural nets** 通过沿超平面切割输入空间来工作，它们具有一种在低维空间中表现更好、在高维空间中能抑制过拟合的隐式偏置（implicit bias）。
   - 至少需要 *d+1* 个超平面才能围成一个有界集（bounded set），因此一个完美过拟合的模型如果要将每个数据点围在单独的有界集中，至少需要 *n*(d+1) 个神经元。
- **Hanin 关于超平面处理的有益提示**：一名成员分享了 [Boris Hanin 论文的链接](https://arxiv.org/abs/1906.00904)，该论文展示了 **ReLU networks** 的一些数学特性，特别是研究了它们常数区域（constant regions）的几何结构。
   - 另一名成员表达了对论文中某个特定图表的喜爱。
- **ICML 邀请对机器遗忘进行深入研究**：一名成员分享了 [ICML](https://icml.cc/Conferences/2024) 将举办 **machine unlearning workshop**（机器遗忘研讨会）。
   - 研讨会的网站可以在[这里](https://mugenworkshop.github.io/)找到。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1359005284605100172)** (1 messages): 

> `LM Harness, HotpotQA, Llama Eval, GPT Models` 


- **寻求指导：针对 HotpotQA 的 LM Harness**：一名成员询问关于 **HotpotQA 的 LM harness 实现**，以便评估 **Llama** 和 **GPT models**。
   - 他们请求关于针对 **HotpotQA** 运行评估的指导。
- **评估中的 Llama 和 GPT 模型**：成员们正在评估 **Llama** 和 **GPT models**。
   - 他们需要一个 **HotpotQA 的 LM harness 实现**来完成这项工作。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1358927244734500874)** (127 messages🔥🔥): 

> `Llama-4-Scout-17B, Gemini 2.5 Pro Code Generation, aider-chat & Gemini 2.5 Pro, HiDream-I1 Image Model, DeepCogito LLMs & IDA` 


- **Llama-4-Scout-17B 准备接入 llama.cpp**：[Llama-4-Scout-17B text-to-text](https://github.com/ggml-org/llama.cpp/pull/12791) 支持已添加到 *llama.cpp*，成员们正在努力进行模型的转换和量化。
   - 这一预发布版本让急于测试其能力的内核用户感到兴奋。
- **Gemini 2.5 Pro 生成高质量代码片段**：**Gemini 2.5 Pro** 因其能够根据复杂提示词生成功能性代码片段而受到赞赏。可以在[此消息](https://cdn.discordapp.com/attachments/1149866623109439599/1358975415426879589/message.txt?ex=67f7c63b&is=67f674bb&hm=1c655347ddb71efc0e03a079e62d8e26286724363242370cf6f19b9e50cc1980&)中查看提示词和回复。
- **aider-chat 结合 Gemini 2.5 Pro 创建 AGI 原型**：一位用户报告称，使用 **aider-chat** 结合 **Gemini 2.5 Pro**，在 **300k token context** 下编辑或创建了 15 个文件，包括他们的前端、API 和微服务。
   - 该用户感觉他们现在已经拥有了部署生产级 AGI 原型所需的所有文件。
- **HiDream-I1 图像模型生成高质量图像**：**HiDream-I1** 是一个新的开源图像生成基础模型，拥有 **17B parameters**，使用 **Llama 3.1 8B** 作为文本编码器，根据 [MIT license](https://huggingface.co/HiDream-ai/HiDream-I1-Full) 发布，能在数秒内达到最先进的图像生成质量。
   - 它*在包括写实、卡通、艺术等多种风格上产生了卓越的效果，达到了最先进的 HPS v2.1 分数，符合人类偏好*。
- **DeepCogito 模型使用迭代蒸馏与放大（IDA）**：一套新的 **Cogito** 模型（**3B-70B**）声称优于同尺寸的 **Llama, DeepSeek, 和 Qwen** 模型。它们使用 **Iterated Distillation and Amplification (IDA)** 进行训练，该技术通过[这项研究](https://www.deepcogito.com/research/cogito-v1-preview)中概述的放大和蒸馏循环迭代地提高模型能力。
   - 值得注意的是，据称其 **70B model** 超越了新发布的 **Llama 4 109B MoE model**。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1358919204622831677)** (4 messages): 

> `LayerNorm 实现, Llama4 上下文窗口, H100 使用` 


- **LayerNorm 统计数据按样本计算**：一位成员实现了 **LayerNorm**，并指出其与 **BatchNorm** 的关键区别在于按样本（**axis=1**）而不是按批次计算统计数据，并使用 **keepdims=True** 以避免操作数问题。
   - 他们还移除了 running averages，因为均值和方差取决于特征数量而非批次大小，并附带了[一张图片](https://cdn.discordapp.com/attachments/1154120232051408927/1358919204270641342/image.png?ex=67f791e1&is=67f64061&hm=3aeae5b8f48b37b2e22dacbdc7c0fe25279704c08caf5ff4cdbf3df8a01acf2a&)进行展示。
- **Llama4 需要 H100 吗？**：一位成员询问了在单张 **H100** 上测试具有 **10M 上下文窗口**的 **Llama4** 的相关事宜。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1358913098085302273)** (18 messages🔥): 

> `分布式数据并行训练, 不可信低成本算力, Nous DeMo 论文, 梯度压缩算法, P2P 可中断算力` 


- **Panthalia 平台通过 DDP 验证低成本算力**：一个旨在验证用于互联网模型训练的不可信、低成本算力的平台已经开发完成，该平台使用分布式数据并行（DDP），灵感源自关于压缩的 **Nous DeMo** 论文，可通过 [X.com](https://x.com/panthaliaxyz/status/1909342585505669228) 加入等待名单。
   - 该平台使用了一种梯度压缩算法，文档详见[此处](https://docs.panthalia.com/gradient-compression-algorithm)，代码可在 [GitHub](https://github.com/ritser-labs/panthalia-worker/blob/main/spl/util/demo.py) 获取。
- **Panthalia 旨在以 $0.60/小时转售 H100 算力**：在早期阶段，**Panthalia** 旨在以可中断价格转售低成本供应商的算力，例如 **H100 为 $0.60/小时**，**4090 为 $0.13/小时**，并利用 **DDP** 和 **DeMo 风格压缩**。
   - 权重存储在可靠的服务器上，从而为初始用户提供扩展能力，长期计划包括构建 **P2P 可中断算力**的供应。
- **Panthalia 支持用户自定义训练和插件**：该平台支持的模型大小仅受设备容量限制，使用 **DeMo 压缩**可显著减小体积，并提供[插件系统](https://docs.panthalia.com/buying-compute/create-a-plugin)允许用户定义自己的模型、训练方法（**QLoRA**）和分布式训练算法（**DeMo vs. DiLoCo**）。
   - 用户可以下载权重，算力单元可以在子网内标准化以确保验证，支持 Stripe（信用卡）支付和加密货币提现。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1358953132406739005)** (10 messages🔥): 

> `GPUMODE triton 数据集, 用于 triton kernel 的 PyTorch 版本, GPUMODE 网站改进, GPUMODE 招聘门户` 


- **GPUMODE Triton 数据集：在 PyTorch 2.5 上生成**：用于 [Inductor Created Data](https://huggingface.co/datasets/GPUMODE/Inductor_Created_Data_Permissive) 的 **GPUMODE** "triton" 数据集是使用 **PyTorch 2.5** 创建的。
   - 创建者承诺更新 readme 以反映这一关键细节，因为用户在 **PyTorch 2.6+** 上运行它时可能会遇到问题。
- **GPUMODE 网站：建议在新标签页打开导航**：一位用户建议 **GPUMODE** 网站上的 "Lectures" 和 "Resources" 标签页应在新标签页中打开，因为它们是指向 YouTube/GitHub 的超链接。
   - 这将防止用户在同一标签页中离开 **GPUMODE** 网站，从而*提升用户体验*。
- **GPUMODE：招聘门户想法**：一位成员提议在 **GPUMODE** 网站上增加一个招聘门户，通过抓取特定频道的帖子来创建新职位。
   - 他们还建议为招聘发布者提供一个**静态模板**（JSON 或 YAML），以确保格式一致并简化条目创建，GPUMODE 工作人员已确认了该建议。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1359114922361094144)** (14 条消息🔥): 

> `block_ptr 用法, tl.load 和 boundary_check, 边界检查与性能` 


- **使用 Block Pointers 填充越界访问**：一位成员建议使用 `tl.make_block_ptr` 创建指针，以便在越界内存访问时填充零，特别强调了 **`boundary_check`** 和 **`padding_option="zero"`** 的用法。
   - 提供的用法示例利用 **`tl.make_block_ptr`** 及其参数（如 **`shape`**, **`strides`**, **`offsets`**, **`block_shape`**, 和 **`order`**）来创建指针，然后使用带有边界检查的 **`tl.load`** 加载数据。
- **`tl.make_block_ptr` 深入探讨**：一位成员询问了关于 **`tl.make_block_ptr`** 的问题，包括是否可以在循环中大量使用、如何使用 offset 参数以及 order 参数的含义。
   - 另一位成员澄清说，应该调用 **`tl.advance`** 来递增指针以便在循环中加载数据，offset 代表起始元素索引，而 order 参数定义了内存布局（例如：列优先矩阵）。
- **`boundary_check` 的顺序无关紧要，但为了正确性是必需的**：一位成员询问了 **`tl.load`** 中 `boundary_check` 的含义和行为，特别是其顺序以及省略它的后果。
   - 解释指出 `boundary_check` 的顺序并不重要，省略它可以提高速度，但会面临诸如 *"device-side assert triggered"* 之类的错误风险，这是由于潜在的缓冲区溢出导致的，特别是当数组维度不是 block size 的倍数时。
- **能否填充零或 NaN 以外的其他值？**：一位成员询问在使用 block pointers 时是否可以**填充零或 NaN 以外的值**。
   - 另一位成员回答说这很难实现，但你可以通过使用 **`tl.where(x == x, x, another)`** 将 NaN 替换为另一个值，因为 `nan != nan`。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1358978121948463234)** (4 条消息): 

> `Deepseek 通信库, NVSHMEM 与 Unified Virtual Addressing (UVA), LDSM (Local Data Share Memory), 优化的 smem 加载` 


- **基于 NVIDIA NVSHMEM 构建的 Deepseek 库**：Deepseek 通信库是基于 NVIDIA 的 **NVSHMEM** 库构建的。
- **探讨用于 UVA 节点内通信的 NVSHMEM**：一位成员询问 **NVSHMEM** 是否在节点内通信中使用 **Unified Virtual Addressing (UVA)**，从而允许通过 NVlink 对远程 GPU 进行点对点（P2P）加载/存储。
- **关于 LDSM 复制的讨论**：一位用户询问定义 `make_tilded_copy` 的代码，并表示当前的图像看起来不像是来自 `tiled_mma`，而是像 **LDSM** 复制。
   - 有人解释说，使用 **LDSM** 时，warp 中的 32 个线程协同工作，将数据从 smem 复制到 rmem；如果 smem 是行优先的，T0 从 smem 加载 0-7，并将 0,1,8,9,128,129,136,137 存储到其自身的寄存器内存中。
- **优化的 smem 加载**：一位成员分享了一段代码片段 `tCsA = thr_mma.partition_A(sA); tCrA = thr_mma.make_fragment_A(tCsA); copy(tCsA, tCrA);`，用于根据 `tiled_mma` 对 `sA` 进行分区。
   - 他们补充说，为了实现优化的 **smem load**，应该使用 **LDSM**。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1358940322679361676)** (9 条消息🔥): 

> `TorchTitan 的编译策略, FSDP 数值问题, FSDP2 模型提取` 


- **TorchTitan 的预编译策略**：标准做法通常是在操作后进行编译，但 **TorchTitan** 在操作前进行独特的逐块（per-block）编译，可能是为了规避一些 **torch compile bug**；参见 [torchtitan/parallelize_llama.py#L313](https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/llama3/parallelize_llama.py#L313)。
   - 这种 block-wrapping 方法旨在利用 Dynamo 的缓存来跳过 **Triton 的 LLVM** 编译（该过程较慢），然而，在同时使用 `torch.compile` 和 FSDP 时，仍可能存在数值问题。
- **FSDP 和 torch compile 导致数值问题**：一家研究实验室在使用 **FSDP** 配合 `torch.compile` 时遇到了数值问题，导致训练不稳定，reward 会突然暴跌。
   - 他们发现禁用 `torch.compile` 解决了这些问题，并警告要*谨慎使用 torch compile*，强调这些问题是在使用 **HF qwen2.5** 以及自定义的 **GRPO+entropy loss** 时观察到的。
- **从 FSDP2 提取封装模型仍具挑战性**：一位成员询问如何从 **FSDP2** 封装的模型中获取原始模型，因为修改是就地（in place）进行的，且 **FSDPModule** 中未实现 `copy.deepcopy`。
   - 另一位成员建议，由于 **FSDP** 通过封装模块就地修改模型，建议在应用 **FSDP** 之前保留原始模型。


  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1359155079055282206)** (5 条消息): 

> `CUDA physics simulation kernels go open source, Triton-Distributed, SMERF 3D` 


- **PhysX 公开！**: NVIDIA 的 **CUDA 物理模拟内核**现已[开源](https://github.com/NVIDIA-Omniverse/PhysX/discussions/384)；一些开发者已经在着手开发 **ROCm** 版本。
- **Triton 获得分布式超能力！**: 一份学习笔记详细介绍了 **Triton-Distributed**，它将 Triton 与 **NVSHMEM/ROC-SHMEM** 融合，以实现多 GPU 执行，为分布式任务添加 IR，并支持计算与通信重叠（[链接](https://x.com/thatperfguy/status/1909360454465433831)）。
- **SMERF 的柏林 Demo 依然很酷**: **SMERF** (**Scalable Modelling of Explicit Radiance Fields**) 项目的[柏林 Demo](https://smerf-3d.github.io/select_quality/?scene=berlin) 在 **3D 场景重建**能力方面依然令人印象深刻；项目页面在[这里](https://smerf-3d.github.io/)。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1359031721986883726)** (2 条消息): 

> `Krea hiring, ML engineers, GPU cluster, diffusion models, interns` 


- ****Krea** 寻求 ML 工程师以发挥 **GPU** 卓越性能！**: **Krea** 正在[招聘 ML 工程师](https://jobs.ashbyhq.com/krea)，负责优化其 **GPU** 集群的训练/推理流水线，寻找对加速图像生成模型充满热情的人才。
- ****Krea** 需要扩散模型研究员**: **Krea** 还在寻找有兴趣增强扩散模型可控性和美学效果的研究员。
- **实习岗位咨询**: 一位成员询问了 **Krea** 是否有潜在的实习机会。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1358916933272994056)** (15 条消息🔥): 

> `Graph Neural Networks (GNNs), Graph Attention Networks (GATs), CUDA compilation of C code, NVIDIA Streaming Multiprocessors, Thread cooperation in CUDA` 


- **GNN 计算是彻底并行的**: 成员们讨论了**图神经网络 (GNNs)** 的并行特性，指出图中每个节点的更新通常可以并行计算。
   - 一位成员提到，**图注意力网络 (GATs)** 架构就是其中一个能想到的例子。
- **C++ 编译器可能无法编译有效的 C 代码**: 成员们讨论了 **C++** 编译器可以编译所有 **C** 代码的说法，并引用了服务器的 FAQ。
   - 一位成员指出，编写无法用 **C++** 编译器编译的 **C** 代码是可能的，并引用了 [Wikipedia 文章](https://en.m.wikipedia.org/wiki/Compatibility_of_C_and_C++)。
- **CUDA 术语表更新**: 一位成员建议加入一张来自 Hennesy & Patterson 的截图，将术语分解为通用概念（例如：**NVIDIA Streaming Multiprocessors = Cores**）。
   - 另一位成员建议将其作为建议添加到[术语表](https://docs.google.com/document/d/1xNRvBJS1CPurxGESSRljGCS0fetpIOafd22JV8D4Ufg/edit?tab=t.0)中，或者发布在频道里。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1359317987814936727)** (1 条消息): 

> `torchao 0.10.0 release, MXFP8 training, PARQ, Module Swap Quantization API, Low Bit Kernels` 


- **TorchAO 发布新版本：v0.10.0**: **torchao** 的 **0.10.0 版本**引入了对 **Nvidia B200** 上 **mxfp8** 的端到端训练支持，以及用于量化感知训练的 **PARQ**。
   - 此版本还包括一个用于研究的 **module swap quantization API** 以及**低比特内核**的更新，详情请参阅[发布说明](https://github.com/pytorch/ao/releases/tag/v0.10.0)。
- **Nvidia B200 可以使用 MXFP8 训练**: 得益于 **torchao 0.10.0 版本**的更新，**Nvidia B200** 现在支持 **MXFP8** 的端到端训练。
   - 这些训练能力将实现更好、更快的量化感知训练和新的研究。
- **TorchAO 为研究发布 Module Swap Quantization API**: 新的 **Module Swap Quantization API** 将使研究人员能够有效地对模型中的自定义模块应用量化。
   - **torchao 0.10.0 版本**允许研究人员通过将标准模块替换为量化版本，更灵活地实验量化策略。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

twzy: 今天遇到了 Yann LeCun，他看起来很生气
  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1358899608889327776)** (9 条消息🔥): 

> `Tom and Jerry Diffusion Transformers, Nvidia Hopper Distributed Shared Memory, Verifying Untrusted Low-Cost Compute, LiveDocs Code Documentation` 


- **动画时间：团队凭借 Tom & Jerry Transformer 取得成功**：一个团队通过微调 **Diffusion Transformer** 完成了一个制作 1 分钟长 **Tom and Jerry** 动画的项目，其工作已被 **CVPR 2025** 接收，并在 [GitHub 上发布了微调代码](https://github.com/test-time-training/ttt-video-dit)。
   - 他们还发布了一个由 Diffusion Transformer 完全生成的未经编辑的输出 [示例视频](https://cdn.discordapp.com/attachments/1358899608889327776/1358907723433115749/homeless.mp4?ex=67f78730&is=67f635b0&hm=3b64ef6ea758875651b8faacd7f4e0ad769cfbe8488c5ea07b1511958c608660&)。
- **Hopper 的隐藏硬件助力高性能 RNN**：一位成员提到，在 **Nvidia Hopper** 架构上有一个非常有趣的特性，即可以使用 **Distributed Shared Memory**（分布式共享内存）在 SM 的 SRAM 之间直接传输数据。
   - 他们利用这一特性在单张 GPU 上跨 SM 运行其 RNN 隐藏状态的 **Tensor Parallelism**（张量并行），从而消除了写回 HBM 的需求。
- **Panthalia 平台提供可信并行计算证明**：一位成员一直致力于开发一个验证不可信低成本计算的平台，以便通过互联网使用 **Distributed Data Parallel**（分布式数据并行）来训练模型，[详情见此](https://x.com/panthaliaxyz/status/1909342585505669228)。
   - 他们使用了一种深受 **DeMo 论文** 启发的压缩算法（[文档](https://docs.panthalia.com/gradient-compression-algorithm)）。
- **LiveDocs 发布专业级代码文档服务**：**LiveDocs** 的创建者邀请用户使用其升级后的服务来“记录你的代码”，现在通过在 [www.asvatthi.com](http://www.asvatthi.com) 注册即可使用更多功能。
   - 随附了一张界面截图，展示了各种代码文档页面。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1359233590935949443)** (1 条消息): 

> `AlphaGeometry, KernelBench, GPU kernel generation` 


- **KernelBench 展示引导 GPU Kernel 生成能力**：一位成员提到了之前关于使用验证器通过 Test-time Compute Scaling 来引导 **GPU Kernel 生成** 能力的工作，并引用了 [KernelBench](https://arxiv.org/abs/2502.10517) 中的实验。
   - 该方法并不完全是 **AlphaGeometry 风格**，但涉及一小组用于应用角度追逐解题器（Angle Chasing Solvers）的动作。
- **几何与验证器讨论**：讨论涉及与 **AlphaGeometry 风格** 技术和验证器相关的方法。
   - 提到的一种方法本质上涉及一组相当有限的可能动作，用于应用角度追逐解题器。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1358956210988257401)** (6 条消息): 

> `Quasar Alpha, Reasoning Gym Levels, Curricula Tasks` 


- **Quasar Alpha：Open Router 测试模型**：一位用户分享了 **Quasar Alpha**（Open Router 测试模型）的性能，并附带了 [图片](https://cdn.discordapp.com/attachments/1316377974672588850/1358956210753114293/Figure_1.png?ex=67f7b458&is=67f662d8&hm=47ec1cb8ad8ce7db302a1d483baf3aeca52da523c45675b7e468a14ac8e5b740&)。
   - 另一位用户询问了原始输出，以便可能在 [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval) 的 PR 中添加。
- **Reasoning Gym 任务等级需要定义**：一位用户提到他们正在为 reasoning-gym 中目前缺乏定义的 **15 个任务** 定义等级，计划在晚上完成。
   - 该用户询问是否可以提交 PR 以使这些定义在主分支上可用，并获得了批准。
- **Reasoning Gym 课程任务 PR**：一位用户询问为没有课程（Curricula）的任务添加课程是否适合提交 PR 到 [reasoning-gym-eval](https://github.com/open-thought/reasoning-gym-eval)。
   - 这种行为受到了鼓励。


  

---


### **GPU MODE ▷ #[gpu模式](https://discord.com/channels/1189498204333543425/1342364798058500148/1358939166250893482)** (3 条消息): 

> `DeepSeek Communication Library, NVSHMEM and UVA, Intra-node communication` 


- **DeepSeek 利用 NVSHMEM**：一位成员询问 **DeepSeek 通信库** 是否是基于 NVIDIA 的 **NVSHMEM 库** 构建的。
- **NVSHMEM 对 UVA 的使用受到质疑**：一位成员质疑 **NVSHMEM** 是否在节点内通信中使用 **Unified Virtual Addressing (UVA)**。
- **通过 NVLink 进行 Peer-to-Peer 加载/存储**：该成员补充说，使用 **UVA**，可以对存储在远程 GPU（通过 **NVLink** 等连接）中的数据执行 Peer-to-Peer 的加载/存储。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1359236223498977391)** (11 条消息🔥): 

> `提交带有内联 CUDA 的 .py 文件、CUDA Kernels、Grayscale CUDA 示例、torch::extension` 


- **内联 CUDA 提交问题**：一位用户报告了在提交带有内联 **CUDA** 的 **.py 文件**时遇到困难，并对参考脚本的有效性提出了质疑。
   - 管理员确认了该问题，并请求提供失败任务的链接以协助调试。另一位用户建议示例提交可能不正确，而其他内联 **CUDA** 实现可能有效。
- **CUDA 内联脚本解决方案**：一位用户请求内联 **CUDA** 提交的示例脚本，另一位用户提供了一个使用 **C++** 和 **CUDA** 的代码模板。
   - 该代码模板包括 **CUDA 源码**（一个 `grayscale_kernel` 函数）、**C++ 源码**（包括 `<torch/extension.h>`）以及一个通过 `load_inline` 加载的 **Python 模块**。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1358888800029249841)** (17 条消息🔥): 

> `vectoradd 基准测试、grayscale 基准测试、Modal runners` 


- **VectorAdd 基准测试大爆发**：使用 **Modal runners** 在 **L4 GPU** 上进行的多个 **vectoradd** 基准测试提交已成功，ID 范围从 **3500** 到 **3532**。
- **Grayscale 排行榜受到关注**：使用 **Modal runners** 在 **L4, T4, A100 和 H100 GPU** 上进行的 **grayscale** 排行榜提交已成功，包括 ID **3503, 3536, 3539 和 3540**。
- **Modal Runners 交付结果**：使用 **Modal runners** 提交到 **T4** 和 **A100 GPU** 的 **vectoradd** 排行榜（ID 分别为 **3537** 和 **3538**）已成功。


  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1358880301543329923)** (5 条消息): 

> `排行榜差异、CUDA 提交失败` 


- **排行榜时间单位引起混淆**：一位用户注意到网页版 ([https://gpu-mode.github.io/discord-cluster-manager/](https://gpu-mode.github.io/discord-cluster-manager/)) 和 Discord 排行榜之间的时间单位存在差异，前者显示为**纳秒 (nanos)**，而后者显示为**毫秒 (millis)**。
   - 一个新的排行榜网站正在准备中，时间单位将被转换以提高清晰度。
- **CUDA 提交受阻**：一位用户报告称，来自 ([https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py](https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py)) 的示例 **CUDA** 提交在作为测试运行时失败。
   - 这被认为是出乎意料的，用户被要求提供具体的错误信息。


  

---


### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1358932410410205254)** (3 条消息): 

> `A100 vs L40, FP8 支持, 4bit 权重, 开源 w4a8 kernels, GPU Fryer 工具` 


- **A100 在带宽和 Tensor 操作上压倒 L40**：据报告，**A100** 在 **DRAM 带宽**和 **Tensor 操作**方面几乎比 **L40** 快两倍。
   - 尽管 **L40** 具有 **FP8 支持**和更大的 **L2 缓存**，但 *vLLM 在其常规发行版中并未包含针对 Lovelace 架构优化的 kernel*。
- **4-bit 权重下 8-bit 浮点数的收益有限**：在使用 **4-bit 权重**的情况下，**Hopper/Lovelace** 架构中 **8-bit 浮点数支持**带来的收益有限。
   - 目前没有可用的开源 **w4a8 kernels**。
- **用于排查问题的 GPU Fryer 工具**：运行 [GPU Fryer 工具](https://github.com/huggingface/gpu-fryer) 可以帮助识别问题。
   - 该工具由 Hugging Face 维护，对于压力测试和调试 **GPU 配置**非常有用。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1358883324382416936)** (52 messages🔥): 

> `FP4 Fine-tuning, Parasail Inference Provider, Llama.cpp Llama 4 Support, Mobile SQL Generation Models, Multi-Agent AI Deployment` 


- **FP4 微调热潮加速任务完成**：用户正在探索使用 [Unsloth](https://github.com/unslothai/unsloth) 等工具对 **FP4** 量化模型进行微调，该工具允许加载低精度模型进行训练和量化。
   - 虽然可以通过 **LoRA** 微调量化模型，但直接微调量化模型本身是不可能的。
- **Parasail 旨在提供卓越性能**：**Parasail** 是一家新的推理提供商（Inference Provider），在结束隐身模式后正寻求与 Hugging Face 合作。据 [The Next Platform](https://www.nextplatform.com/2025/04/03/parasail-brokers-between-ai-compute-demand-and-supply/) 报道，它已经在 Open Router 上每天处理 **30亿 token**，并为私有公司每天处理超过 **50亿 token**。
- **Llama.cpp 跨越至 Llama 4**：根据 [GitHub releases](https://github.com/ggml-org/llama.cpp/releases)，后端 **Llama.cpp** 已更新以支持 Llama 4。
- **手机端推崇小型 Transformer**：对于在移动设备上根据数据描述生成 **SQL 查询**，推荐使用 **Qwen 2.5 0.5B** 以及来自 [SmollM2 Intermediate Checkpoints 集合](https://huggingface.co/collections/HuggingFaceTB/smollm2-intermediate-checkpoints-67c079ca030f714c30ce49a1)和 [TinyLlama 集合](https://huggingface.co/collections/TinyLlama/tinyllama-11b-v11-660bb405bf46efd55c2094fc)的模型。
   - 建议通过 **ONNX** 将模型转换为 **TensorRT** 格式，以利用旧架构。
- **编排选项开启新机遇**：**Oblix** ([https://oblix.ai/](https://oblix.ai/)) 是一款用于编排边缘与云端之间 AI 的新工具，它在边缘端集成了 Ollama，在云端支持 OpenAI 和 ClaudeAI，旨在创建低延迟、注重隐私的工作流。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1359043209065402530)** (3 messages): 

> `Ollama local deployment, NLP in HuggingFace` 


- **新手使用 Ollama 进行本地部署**：一位成员开始学习使用 **Ollama** 结合 **Python** 和 **OpenAI** 进行本地部署。
   - 他们使用 **Ollama** 本地部署是为了避免支付 **OpenAI API keys** 的费用。
- **新手在 HuggingFace 学习 NLP**：一位成员提到他们正在 **HuggingFace** 页面学习 **NLP**。
   - 他们希望在截止日期前完成课程。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1359180244493271211)** (1 messages): 

> `Daily Papers Podcast, Takara TLDR` 


- **Takara TLDR 激发了 Daily Papers Podcast！**：一位用户借鉴了 **Takara TLDR** 的概念，创建了一个 [每日论文播客（Daily Papers Podcast）](https://huggingface.co/spaces/eswardivi/Daily_Papers_Podcast)。
   - 该播客似乎托管在 HuggingFace 平台上。
- **每日 AI 论文摘要，现在以播客形式呈现**：一位用户将 **Takara TLDR** 的概念重新混剪成了 [每日论文播客](https://huggingface.co/spaces/eswardivi/Daily_Papers_Podcast)。
   - 这可能是紧跟最新 AI 研究的宝贵资源。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1358965715436044460)** (3 messages): 

> `AI Runner, GAPRS` 


- **AI Runner 桌面 GUI 正式发布！**：一位成员发布了 **AI Runner**，这是一个使用 HuggingFace 库在本地运行 AI 模型的桌面 GUI，详见 [此 YouTube 视频](https://youtu.be/IPn3TcQr7e0)。
   - 该工具允许用户创建和管理具有自定义声音、性格和情绪的聊天机器人，这些机器人是使用 llama-index 和 ReAct 工具构建的 Agent，可以生成 **Stable Diffusion** 图像并进行实时语音对话（使用 espeak、speecht5 或 openvoice）。
- **GAPRS 3.0 起航！**：一位成员发布了他们硕士论文项目的第三次迭代，这是一个名为 **GAPRS**（基于图的学术推荐系统）的 Web 应用程序，网址为 [lqhvwseh.manus.space](https://lqhvwseh.manus.space)。
   - **GAPRS** 的目标是帮助学生了解从哪里开始撰写论文，简化学术研究流程，并彻底改变学术论文的变现方式；更多细节可以在该成员的硕士论文中找到。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1359061388873437254)** (3 messages): 

> `Monocular Depth Models, Segmentation Problem, Tools Recognition Task` 


- **探索单目深度模型 (Monocular Depth Models)**：一位成员询问另一位成员是否找到了问题的解决方案，并提到他们已经尝试过 **monocular depth models**。
- **提出分割解决方案**：一位成员针对涉及垂直杆的 **segmentation problem** 提出了解决方案，建议检查具有相同标签的不同分割块的边界框（bounding boxes）在 x 坐标上的重叠情况。
   - 该用户建议 *取 min(x_lefts)->max(x_rights)*；他们还建议通过使用 (x_mid +/- 0.5*width_pole) 来修整较厚的框。
- **工具识别任务**：一位成员询问关于 **tool recognition task** 的最佳模型/算法建议，并说明该模型应通过提供参考图片来识别工具。
   - 该成员询问是否应该增强模型以获得更好的特征提取效果。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1358967599786102866)** (4 messages): 

> `Dataset forms, Unit 1 Quiz failing to load, Agents Build Errors, Chat templating exercises` 


- **数据集形式引发困惑**：一位成员指出有人对 *两个数据集执行相同的操作，但它们的起始形式（forms）不同*。
   - 另一位成员请求更多细节，询问 *你说的形式（forms）是什么意思？*
- **Unit 1 测验加载失败，重定向次数过多**：一位成员报告 **Unit 1 quiz** 无法加载并陷入重定向循环：*agents-course-unit-1-quiz.hf.space 重定向次数过多*。
   - 他们提到自己是编程新手，不确定如何解决该问题，正在寻求支持。
- **Agents 出现构建错误**：一位成员报告在尝试获取错误日志时遇到 *Build error*，并陷入循环。
   - 最初，他们在与 Agent 聊天时无法获得任何响应，甚至在 *复制并粘贴工具名称* 时也遇到了问题。
- **寻找 Chat Template 练习伙伴**：一位成员正在寻找可以一起讨论 **chat templating exercises** 的人。
   - 未提供其他细节，他们只是在寻找学习伙伴。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1358906149986107702)** (26 messages🔥): 

> `Code Agents Ch. 2 Notebook Issues, Gemini Models as Alternatives, Course FAQ Request, any-agent library release, RAG with smart glasses challenge` 


- **Code Agents 第 2 章 Notebook 需要付费**：一位成员报告运行 Code Agents 第 2 章的 Notebook 需要付费，收到了关于凭据无效以及推荐模型需要付费的错误。
   - 他们寻求关于正确登录或使用替代 Token 来运行这些本应免费的课程示例的建议。
- **推荐使用 Gemini 模型以绕过付费墙**：一位成员建议使用 **Gemini models** 作为许多国家的免费替代方案，并链接到了包含说明的 [课程笔记](https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6#using-other-llm-providers-outside-hugging-face)。
   - 其他成员强调了像 **Ollama** 和其他提供商（**OpenAI**, **Grok**）提供的慷慨的免费 Token 资源，以绕过 Hugging Face 的付费墙。
- **Agent 课程需要 FAQ 章节**：多位成员请求在 Agent 课程中加入 **FAQ section**，因为许多用户面临相同的初始问题，并发现 Discord 导航很困难。
   - 讨论明确了虽然没有官方的 FAQ 页面，但 Discord 频道包含许多可以搜索的常见问题。
- **`any-agent` 库简化了 Agent 框架评估**：Mozilla AI 团队发布了 `any-agent`，这是一个旨在简化尝试不同 Agent 框架的库。
   - 该库支持 **smolagents**, **OpenAI**, **Langchain** 和 **Llama Index** 等框架，并提供 [GitHub 仓库](https://github.com/mozilla-ai/any-agent) 供用户尝试和贡献。
- **Meta CRAG 多模态挑战赛发布**：社区分享了来自 **Meta** 的一个有趣的挑战：与智能眼镜 RAG 相关的 [CRAG Multi-Modal Challenge 2025](https://www.aicrowd.com/challenges/meta-crag-mm-challenge-2025)。
   - 这被推荐作为一个知识练习，以巩固在课程中所学的知识。


  

---

### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1359333165776244917)** (13 messages🔥): 

> `Deepseek R1, 活跃的 AI Discord 聊天` 


- **Deepseek R1 聊天室闲聊开始**：一位成员询问另一位成员一直在使用哪些 **Deepseek** 版本。
   - 另一位成员打趣回应说他们根本没在工作，但假设这个房间是关于 **Deepseek R1** 的，所以 **AI** 正在为“他们”工作。
- **闲聊者寻找活跃的 AI 社区**：一位成员询问 **Discord** 上活跃的 **AI** 聊天频道，或者更好的，活跃的语音聊天频道。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1358883312361279630)** (75 messages🔥🔥): 

> `Semgrep MCP server, MCP HTTP Streaming, MCP 与 CORS 错误, MCP Github server 问题, 用于 Graph API 应用的 MCP` 


- **Semgrep 的 MCP Server 引起关注**：一位成员已经运行 [mcp.semgrep.ai/sse](https://mcp.semgrep.ai/sse) 超过一个月，通过 **Docker** 和 **AWS EC2** 托管。
- **Semgrep MCP Server 解决了 CORS 错误**：一位成员报告了连接 [Cloudflare Playground](https://playground.ai.cloudflare.com/) 时的 **CORS 错误**，该错误已被迅速修复。
   - 报告者指出该工具正在使用 **Cursor** 进行测试，也需要在那里修复 CORS。
- **MCP HTTP 请求-响应支持到来**：讨论中提到了企业客户对 MCP 中 **HTTP 请求-响应** 支持的需求，如 [此 Pull Request](https://github.com/modelcontextprotocol/modelcontextprotocol/pull/206#issuecomment-2766559523) 所示。
   - 成员指出许多企业组织正在使用 MCP，预计该功能将进一步提高其采用率。
- **MCP 增强了基于图数据库的 RAG**：一位成员询问在 **Neo4j 图数据库** 的 **RAG** 场景中使用 MCP，重点是向量搜索和自定义 **CQL 搜索**。
   - 另一位成员确认这是一个很好的用例，并推荐了 [mcpomni-connect](https://pypi.org/project/mcpomni-connect/) 作为可行的 MCP 客户端。
- **Cloudflare 提供远程 MCP Server 教程**：对于那些寻求更简单的远程 MCP Server 入门教程的人，一位成员推荐了 [Cloudflare Agents 指南](https://developers.cloudflare.com/agents/guides/remote-mcp-server/)。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1358879016286949799)** (15 messages🔥): 

> `Semgrep 重写 MCP, C# MCP SDK, ASGI 风格的进程内 fastmcp 会话` 


- **Semgrep 重写 MCP Server**：一位成员重写了 [Semgrep 的 MCP server](https://github.com/semgrep/mcp)，并分享了在 [Cursor](https://www.loom.com/share/8535d72e4cfc4e1eb1e03ea223a702df) 和 [Claude](https://www.loom.com/share/f4440cbbb5a24149ac17cc7ddcd95cfa?sid=f190a5d6-176f-4ceb-86a2-35e98e701411) 中的演示视频。
   - 托管服务器使用 **SSE**，而非 HTTP streaming，因为 [Python SDK](https://github.com/modelcontextprotocol/python-sdk/pull/416) 尚不支持。
- **MCP SDK 利用 Sqlite 存储 LLM 记忆**：一位成员尝试使用 [C# MCP SDK](https://github.com/mbcrawfo/KnowledgeBaseServer) 利用 **sqlite** 存储 **LLM 记忆**。
   - 新版本已发布，为记忆提供**重要性排序**以优化搜索结果，并旨在扩展到更大的记忆图谱。
- **ASGI 风格的 Fastmcp 会话定稿**：一位成员将 [easymcp 版本提升至 0.4.0](https://github.com/promptmesh/easymcp)，显著变化包括 **ASGI** 风格的进程内 **fastmcp 会话**。
   - 其他更新包括定稿的**原生 Docker 传输**、重构的协议实现、新的 mkdocs 以及完整的 pytest 设置。
- **与 MCP Server 的终端聊天**：一位成员创建了一个 [与 MCP server 聊天的终端工具](https://github.com/GeLi2001/mcp-terminal)。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1358895968614879354)** (62 messages🔥🔥): 

> `Shopify AI Mandate, Anthropic API Credits, API Latency Benchmarking, Cybercriminals and AI, LLM Automated Exploitation` 


- **Shopify 的 AI 愿景备受关注**：Shopify 的 AI 指令（mandate）正获得关注，正如[这条推文](https://fxtwitter.com/tobi/status/1909251946235437514)所强调的。
- **Anthropic 的 API 额度有有效期**：Anthropic API 额度在一年后过期，这可能是为了简化会计处理，并考虑到快速发展的 AI 领域。
   - 正如一位成员建议的，这项政策有助于在快速变化的领域中管理预测。
- **NVIDIA 的推理模型具有开关功能**：NVIDIA 发布了一个可以开启或关闭推理功能的新模型，详见[这篇博客文章](https://developer.nvidia.com/blog/build-enterprise-ai-agents-with-advanced-open-nvidia-llama-nemotron-reasoning-models/)，并可在 [Hugging Face](https://huggingface.co/nvidia/Llama-3_1-Nemotron-Ultra-253B-v1) 上获取。
- **网络犯罪的 AI 冲击可能延迟**：尽管出现了 FraudGPT 等基础 AI 应用，但网络罪犯对 AI 的大规模采用速度出奇地慢，有人推测，当他们更广泛地采用 AI 时，可能会发生“网络犯罪 AI 冲击”。
   - 一位成员指出，LLM 可能直到最近才变得足够好，足以用于网络犯罪。
- **Gemini 玩宝可梦并直播疯狂过程**：Gemini AI 正在玩宝可梦，引起了广泛关注，如[这条推文](https://fxtwitter.com/kiranvodrahalli/status/1909699142265557208)所示。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1358957032652148937)** (14 messages🔥): 

> `Llama 4 flops on benchmarks, Bayesian Structural EM, Procedural model representation DNA, Meta should have clarified, Disrupt Science Hackathon Details` 


- **Llama 4 在非针对性优化的基准测试中表现不佳**：一位成员声称 **Llama 4** 在非针对性优化、非过拟合的基准测试中**表现惨淡**。
   - Daily Paper Discussion 频道将讨论[这篇论文](https://arxiv.org/abs/2408.04220)，主作者最近的一次演讲（[YouTube 链接](https://www.youtube.com/watch?v=klW65MWJ1PY)）也讨论了这篇论文。
- **Meta 应该澄清 Llama 4**：有观点认为 *Meta 应该更清楚地说明 “Llama-4-Maverick-03-26-Experimental” 是一个为了优化人类偏好而定制的模型。*
   - 这一讨论基于[这个 fxtwitter 链接](https://fxtwitter.com/lmarena_ai/status/1909397817434816562?t=Gdzbf-abkahHSxqhEeqAkw&s=19)。
- **贝叶斯推理见解**：一位成员指出，贝叶斯推理（Bayesian inference）结合权重和架构已有约 100 年历史，并引用了 [Bayesian Structural EM](https://arxiv.org/pdf/1301.7373) 作为高级示例。
   - 论点是，虽然结合权重和架构是标准做法（例如在 [DARTS](https://arxiv.org/pdf/1806.09055) 或 [ES-ENAS](https://arxiv.org/pdf/2101.07415) 中），但*同时更新架构和权重并不能获得仅靠权重无法获得的表达能力（expressivity）*。
- **程序化模型表示：模型的 DNA**：一位成员引入了**程序化模型表示（procedural model representation）**的概念，即一个小种子可以生成一个大型模型（架构 + 权重）。
   - 他们设想下载一个 10MB 的模型来生成一个 100TB 的模型，或者通过更换种子来生成不同的模型，类似于*下载 DNA 来生成一个人*。
- **Disrupt Science 黑客松详情发布**：**Disrupt Science Hackathon** 的详细信息已发布。
   - 详情可以在[这个 Discord 链接](https://discord.com/channels/714501525455634453/796137754508656641/1359164013304615143)中找到。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1358900565114884207)** (12 messages🔥): 

> `Fast.ai Diffusion Methods, F_A_E_S_I_k=2 Discussion, Open Source beautiful.ai Alternatives` 


- **Fast.ai 探索扩散方法**：一位成员分享了 [Fast.ai 扩散方法课程](https://course.fast.ai/Lessons/part2.html)的链接。
   - 另一位成员询问了课程第二部分的发布时间。
- **解读 F_A_E_S_I_k=2 的启示**：一位成员开玩笑说拥有 `F_A_E_S_I_k=2` 导致获得了 40 小时的视频，这与关于 [arxiv.org/abs/2408.04220](https://arxiv.org/abs/2408.04220) 的论文讨论有关。
   - 他们推测它*可能在构建时就内置了这一点，但可能不擅长处理“大海捞针”式的问题，尤其是当这些“针”彼此之间存在依赖关系时*。
- **寻找 Beautiful.ai 的开源替代品**：一位成员询问是否有 [Beautiful.ai](https://www.beautiful.ai/) 的开源替代方案。


  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455534453/1269724655405498429/1359159357559804036)** (1 messages): 

> `Efficient Tool Calling Templates, Cogito 14b` 


- **Cogito 14b 的高效 Tool 模板**：**14b 模型**突然开始使用一种比最初指令中提供的更高效的 Tool Calling 模板。
   - 建议查看 [Cogito model](https://ollama.com/library/cogito) 以获取示例和灵感。
- **新的高效 Tool Calling 模板实现**：一位用户报告称，一个 **14b 模型**意外地采用了一种更高效的 Tool Calling 模板。
   - 这表明该模型可能已经自主优化了其 Tool 使用，为进一步研究提供了一个潜在领域。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455534453/853983317044756510/1358885648978936050)** (9 messages🔥): 

> `Adapting Pre-training Text, Diffusion Modeling to Control LLMs, Llama 4 Release Issues, Iterative Improvement Strategy` 


- **预训练适配演讲**：一位成员分享了一个[精彩演讲](https://www.youtube.com/watch?v=klW65MWJ1PY)，内容关于如何适配**预训练文本**以包含相关事实的数据库查询，从而训练 **LLM** 在生成过程中进行查找。
- **Diffusion Models 引导 LLM**：一位成员提到使用 **Diffusion Modeling** 来控制 **LLM**，并指出 [这篇论文](https://arxiv.org/pdf/2408.04220) 是相关的资源。
- **Llama 4 失败原因解析**：**Llama 4** 的糟糕发布被归因于拙劣的实现。
- **DeepCogito 迭代改进策略预览**：一位成员分享了来自 **Hacker News** 的链接，内容是关于使用 Test Time Compute 进行微调的迭代改进策略，来自 [DeepCogito](https://www.deepcogito.com/research/cogito-v1-preview)。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1358879938358411466)** (28 messages🔥): 

> `IBM Granite 8B, RAG references, docling OCR, semantic chunking server, ComfyUI image generation` 


- **Granite 8B 在 RAG 任务中表现出色**：一位成员报告称 [IBM Granite 8B](https://www.ibm.com/blogs/research/granite-foundation-models/) 在 **RAG** 方面表现良好，特别是在 **LLM** 提供引用参考方面。
   - 另一位成员表示赞同，也发现 **Granite** 非常有效。
- **Docling 用于非文本 PDF OCR**：一位成员推荐使用 **docling** 进行出色的**图像 OCR**，特别是针对扫描件等非文本 PDF。
   - 他们强调了其在 Embedding 方面的持续运行，以及通过索引文档集成到数据库中，从而通过交集实现 **RAG**。
- **用于上下文文本的语义分块服务器**：一位成员分享了一个语义分块（Semantic Chunking）服务器，并展示了其在 [剪贴板示例](https://gnu.support/files/tmp/clipboard-2025-04-07-22-49-36.html) 中的应用。
   - 他们注意到它与音频和图像处理的兼容性，并建议使用 **ComfyUI** 来结合所有模态。
- **Llama 4 因表现糟糕而遭到抨击**：一位成员抨击 **Llama 第 4 代模型**，称其*与较小的模型相比非常糟糕*。
   - 其他人表示同意，并指出 [Reddit 评论](https://www.reddit.com/r/LocalLLaMA/) 推测它可能在较小的“高质量”数据集上过拟合了，尽管一些基准测试显示出前景。
- **GPT4All：保持本地运行以确保安全**：一位成员建议主要将 **GPT4All** 用于本地操作，以确保隐私并避免将私密信息发送到远程 API。
   - 他们详细说明了如何在本地运行 Embedding 模型，并通过分块和 Embedding 对文件进行索引，并参考了一个 [Shell 脚本示例](https://gnu.support/files/tmp/clipboard-2025-04-09-01-48-48.html)。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1358887711406297212)** (4 messages): 

> `MLX vs MAX, Apple Silicon GPU limitations, MAX capabilities` 


- **MLX vs MAX：深度对比**：一位成员对比了 **MLX**（一个类似于 JAX 的数组编程框架）和 **MAX**，指出虽然 **MLX** 是为 Apple Silicon GPU 定制的，但 **MAX** 目前无法以它们为目标，这给直接对比带来了挑战。
   - 该成员强调，用于 AMD GPU 的 **MAX** 最终将实现在 MI300A 和 AMD 消费级 CPU 上的共享内存优势，类似于 **MLX**，这预示着未来功能的融合。
- **Apple Silicon 的部署缺陷**：该成员警告不要在大型项目中完全依赖 **MLX**，理由是在服务器环境中部署 Apple Silicon 非常困难，这可能导致在部署时需要将代码重写为 **MAX**、**JAX** 或 **PyTorch** 等框架。
   - 他们强调，虽然 **MLX** 在初始实验阶段可能很方便，但在服务器设置中 Apple 生态系统的实际局限性应该是一个关键考虑因素。
- **MAX 的手动向量化和多设备支持**：该成员详细说明，尽管 **MAX** 的 API 层级较低，但它提供了与 NumPy 相当的功能，并利用 Mojo 进行自动和手动向量化，使其对程序员非常友好。
   - 他们承认 **MAX** 在 autodiff 方面存在局限性，但强调了其多设备支持（以 Llama pipeline 为例）以及避免 Tensor 形状问题，将其定位为一个尽管存在挑战但依然强大的替代方案。
- **违反 Discord 自我推广规则**：一位成员指出，某篇特定的帖子似乎不适合该 Discord 频道，认为其可能违反了自我推广规则。
   - 一名管理员表示同意，确认该帖子确实违反了社区的自我推广指南。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1359089138992808017)** (16 messages🔥): 

> `Mojo vs Rust, __moveinit__ and __copyinit__ in Mojo, Returning values in Mojo, Span lifetime in Mojo` 


- **Mojo 与 Rust 的借用机制：新视角**：一位 Mojo 新人分享了一篇[对比 Mojo 和 Rust 的博客文章](https://www.modular.com/blog/mojo-vs-rust)，并指出 Mojo 的 *"默认借用 (borrow by default)"* 感觉更加直观。
   - 该成员随后询问了 Mojo 如何处理函数返回值的。
- **Moveinit vs Copyinit：深入探讨 Mojo 对象返回**：一位成员澄清说，在 Mojo 中返回对象时，`__moveinit__` 的存在决定了对象是否被移动，否则将使用 `__copyinit__`，并提供了一个 [GitHub 上的示例](https://github.com/sstadick/mojo-demo/tree/main/examples)。
   - 该成员还指向了 [Mojo 官方文档](https://docs.modular.com/) 以获取完整信息。
- **通过 rebinding 解锁 Mojo 中的 Span 生命周期！**：一位成员询问如何在 Mojo 中指定 *"返回值的生命周期至少与 self 的生命周期一致"*，特别是针对 `Span`。
   - 另一位成员建议使用 `rebind[Span[UInt8, __origin_of(self)]](Span(self.seq))` 或使 trait 在 origin 上泛型化，但也指出目前尚不支持 trait 参数。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1359015949952618607)** (5 messages): 

> `Tensor Naming, GPU Programming, Compiler Development, Tinygrad Contribution Resources, PMPP 4th ed` 


- **寻求优雅的 Tensor 命名技巧**：一位成员询问是否有更优雅的方法来为 Tensor 命名，以便在打印模型参数时更容易追踪，并提到他们目前是在 Tensor 类中手动添加 *name* 属性。
- **请求 GPU 编程和编译器开发资源**：一位成员表示有兴趣参与 tinygrad 等项目的 **GPU 编程** 和 **编译器开发**，并请求学习资源或博客文章。
   - 他们计划阅读 [tinygrad-notes](https://mesozoic-egg.github.io/tinygrad-notes/)，并征求关于 GPU 编译器开发的图书或博客推荐。
- **推荐 geohotarchive YouTube 频道**：一位成员推荐了 [geohotarchive YouTube 频道](https://www.youtube.com/@geohotarchive/videos) 作为学习 tinygrad 的资源。
- **推荐《PMPP》第 4 版用于 GPU 编程**：一位成员推荐了 **PMPP (第 4 版)** 用于学习 GPU 编程，并建议如果发现任何优秀的编译器资源请进行分享。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1358887432199602199)** (12 messages🔥): 

> `METAL 同步问题，AMD 在 BEAM=2 下的性能，ContextVar 类型，LLaMA 分片问题，采样后设备信息丢失` 


- **METAL 同步故障导致分片异常**：一位成员在复现悬赏任务中的 **METAL 同步问题**最小示例时，发现了分片中的异常行为。
   - 该用户怀疑从 **METAL:1** 到 **CPU** 的 **COPY** 操作在从 **METAL** 到 **METAL:1** 的 **XFER** 结束之前就开始执行了，导致 CPU 读取到的是零而不是正确的分片。
- **AMD BEAM=2 为 Tinygrad 提速**：一位用户报告称，使用 **AMD** 配合 **BEAM=2** 获得了显著的速度提升，达到了 **64 it/s**，超过了之前使用 Torch 达到的 **55+ it/s** 的最佳成绩。
   - 成员们注意到 **BEAM=2** 通常比 Torch 更快。
- **LLaMA 分片混乱：设备信息在转换中丢失**：一位用户在运行带有 `--shard 4` 参数的 **llama.py** 时遇到了 **AssertionError**，表明采样后设备信息丢失了。
   - [GitHub](https://github.com/tinygrad/tinygrad/pull/9761/files) 上提出了一个移动 Tensor 的潜在修复方案，但这与 **METAL** 或同步问题没有直接关系。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1359195100206338160)** (2 messages): 

> `RAG 工作流教程，Auth0 为 LlamaIndex 提供的 GenAI 身份验证` 


- **使用 Llama 4 的 RAG 工作流**：一个快速入门教程演示了如何使用 **Llama 4** 从零开始构建 **RAG 工作流**，展示了如何使用 LlamaIndex workflows 设置摄取、检索和生成等核心步骤，如[这条推文](https://twitter.com/llama_index/status/1909635186079453494)所示。
- **Auth0 的 Auth for GenAI 发布并支持 LlamaIndex**：**Auth0 的 Auth for GenAI** 现在提供原生 LlamaIndex 支持，使得在 Agent 工作流中构建身份验证变得更加容易，如[这条推文](https://twitter.com/llama_index/status/1909697035365961954)所宣布。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1358991517569519756)** (13 messages🔥): 

> `Gemini 2.5 Pro，Google 最新的统一 SDK，StructuredPlannerAgent 文档，Agent 规划工具` 


- **Gemini 2.5 Pro 不可用**：一位成员询问 **Gemini 2.5 Pro** 是否可用，但发现了一条弃用消息，建议使用 **Google 最新的统一 SDK** 而不是 Gemini 2.5 Pro，如 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/examples/llm/google_genai/)所述。
- **Google SDK 模型名称未经验证**：一位成员指出 **Google SDK** 不验证模型名称，而是假设提供的名称是有效的，并建议手动设置 `context_window` 值，因为 **Gemini 2.5** 的上下文窗口非常大。
- **`StructuredPlannerAgent` 文档已移除**：由于 Agent 文档清理以及重复实现，`StructuredPlannerAgent` 的文档已被移除，因为它不再维护。
   - 提供了旧文档的回链：[StructuredPlannerAgent](https://docs.llamaindex.ai/en/v0.12.15/examples/agent/structured_planner/)。
- **推荐使用 Agent 规划工具**：建议使用带有**规划工具**（可进行一些**思维链 CoT** 推理）的 Agent，或者在调用 Agent 之前使用 **LLM** 本身创建计划，以替代 `StructuredPlannerAgent`。


  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1358932250565283982)** (8 messages🔥): 

> `Events Recording Availability, Structured Output Examples, Pydantic Schema Integration, API Requests without Cohere Package, Model Recommendation for Company List Generation` 


- **活动录制：是否有回放？**：一位成员询问是否提供活动录制，以便那些无法实时参加的人观看，因为有些活动听起来很有趣。
   - 未给出回复。
- **成员寻求结构化输出示例**：一位新成员询问如何使用 Cohere 获取结构化输出（例如书籍列表），并表示自己在该领域缺乏经验。
   - 该成员被引导至 [Cohere documentation](https://docs.cohere.com) 作为起点。
- **在 Cohere 中使用 Pydantic Schema**：一位成员寻求直接在 Cohere 的 `response_format` 中使用 **Pydantic schemas** 的方法，以及如何在不使用 Cohere Python 软件包的情况下发送请求，旨在避免引入依赖。
   - 他们收到了 [Cohere Chat API 参考链接](https://docs.cohere.com/reference/chat)，并获知了如何使用 **cURL** 向 `https://api.cohere.com/v2/chat` 发送请求。
- **带有 Response Format 的 OpenAI SDK 示例**：一位成员发现 **cURL** 示例非常有用，并注意到它也出现在带有 `response_format` 参数的 **OpenAI SDK** 示例中。
   - 该成员随后请求推荐最适合生成特定主题公司列表的模型。


  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1358890160380575934)** (1 messages): 

> `Vector Databases, Model Compatibility, Explicit Recommendations` 


- **历史上一直避免对 Vector DB 进行明确推荐**：从历史上看，官方一直避免对 **vector DBs** 做出明确推荐，因为我们的模型与所有这些数据库都能很好地配合。
   - 这是因为模型旨在与*所有* **vector DBs** 有效协作，而无需针对任何特定数据库进行专门优化。
- **跨 Vector DB 的模型兼容性**：模型设计具有广泛的兼容性，确保它们在各种 **vector database** 解决方案中表现良好。
   - 这种方法避免了偏袒特定的 **vector DBs**，并对生态系统保持中立立场。


  

---


### **Cohere ▷ #[「🤖」bot-cmd](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

competent: 目前无法工作！
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1359316466289348729)** (2 messages): 

> `Introduction to Aditya, Machine vision and control, Innovation accelerator, Openchain.earth project, Tools used by Aditya` 


- **Aditya 加入 Cohere 社区**：Aditya 介绍了自己，他拥有用于制造设备（半导体/电子）的 **machine vision and control** 背景。
   - 目前，他们正从 **innovation accelerator/matchmaking/assessment role** 离职休假，以探索 Web/AI，并正在开展 [openchain.earth](https://openchain.earth) 项目。
- **Aditya 的技术栈公开**：Aditya 在项目中使用 **VS Code**、**Github Co-Pilot**、**Flutter**、**MongoDB**、**JS** 和 **Python**（评估中）。
   - 他们来到这里是为了进一步了解 **Cohere's AI** 及其在项目中的应用。


  

---


### **Cohere ▷ #[【🟢】status-updates](https://discord.com/channels/954421988141711382/1346652044181897307/)** (1 messages): 

competent: 应该可以工作！
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1359340577581432873)** (1 messages): 

> `Contributor Tag Request, Discord Roles` 


- **贡献者标签处理中**：一位成员在 Discord 上申请 Contributor 标签，并分享了他们的 [GitHub username](https://github.com/nathan-az)。
   - 他们还风趣地提到自己的 Discord 头像使用了《Psych》中的角色 Gus。
- **申请 Discord 角色**：一位用户正在寻求提升在 Discord 服务器中的角色，特别是 Contributor 标签。
   - 他们链接了自己的 GitHub 个人资料进行验证，并拿自己的头像开了个玩笑。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1358908107149021255)** (6 messages): 

> `DeepSpeed Integration, FSDP vs DeepSpeed, FSDP Sharding, zero1-3 training` 


- **TorchTune 的 DeepSpeed 集成讨论**：一位成员询问了将 [DeepSpeed](https://www.deepspeed.ai/) 作为后端集成到 TorchTune 的事宜，并创建了 [一个 issue](https://github.com/pytorch/torchtune/issues/2569) 来讨论其可能性。
   - 一位维护者询问了更多背景信息，并指出 **FSDP 支持 DeepSpeed 的所有分片（sharding）选项**；集成 DeepSpeed 的潜在原因包括：*作为 FSDP bug 的备选方案、不同的硬件/加速器支持以及速度优势*。
- **TorchTune 倾向于 FSDP 而非 DeepSpeed**：TorchTune 更倾向于使用 **FSDP**，因为它能更好地与 PyTorch 的其他分布式特性组合，且官方认为*同时维护好两个版本是不可行的*。
   - 为了避免 DeepSpeed、PyTorch 和 Megatron 组合带来的复杂性而迁移到 TorchTune 的用户更倾向于坚持使用原生 PyTorch，因此没有必要过度投入于集成和支持其他框架。
- **社区 Recipe 构想：DeepSpeed 结合 TorchTune**：一位维护者建议创建一个社区 Recipe，通过导入 TorchTune 来托管 DeepSpeed Recipe，并表示如果建立了相关仓库，可以将其作为特色推荐。
   - 这允许对 **DeepSpeed** 感兴趣的用户在 TorchTune 中使用它，同时保持核心框架专注于原生 PyTorch。
- **为 zero1-2 训练调整 FSDPModule**：由于 TorchTune 默认使用相当于 **zero3** 的配置，因此关于如何通过 **FSDPModule** 方法调整 Recipe 以进行 **zero1-2** 训练的文档或更多 Recipe 将会很有价值。
   - 据信，只需对集合通信（collectives）进行极小的调整，即可实现 **zero 1-3**。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1359213591215210626)** (1 messages): 

> `MIPRO, Automated Prompt Engineering, Task Complexity Scaling` 


- **MIPRO 算法在扩展复杂任务上的测试**：一篇文章 [article](https://tensorzero.com/blog/from-ner-to-agents-does-automated-prompt-engineering-scale-to-complex-tasks) 测试了 **MIPRO 自动提示词工程算法**在不同复杂度任务中的表现，涵盖了从命名实体识别到基于文本的游戏导航。
   - 该研究利用了 **CoNLL++、HoVer、BabyAI** 和 **τ-bench**（涉及 Agent 工具使用的客户支持）等任务。
- **模型大小对 MIPRO 优化至关重要**：研究发现，在复杂设置下，**大型模型从 MIPRO 优化中获益更多**，这可能是因为它们能更有效地处理更长的多轮示例（demonstrations）。
   - 反馈的质量显著影响 MIPRO 的优化过程，即使是来自 **带有噪声的 AI 生成反馈**，也能观察到明显的改进。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1358886256062500924)** (1 messages): 

> `Kaiyu Yang, AI4Math, Theorem Proving, Autoformalization` 


- **Kaiyu Yang 关于形式化数学推理的讲座**：特邀演讲者 **Kaiyu Yang** 今天在直播中发表了题为 *“用于自动形式化和定理证明的语言模型”* 的演讲；[链接在此](https://www.youtube.com/live/cLhWEyMQ4mQ)。
   - 他的讲座涵盖了使用 LLM 进行形式化数学推理，包括 **定理证明（theorem proving）** 和 **自动形式化（autoformalization）**。
- **AI4Math 对 AI 驱动系统至关重要**：**数学人工智能 (AI4Math)** 对于 AI 驱动系统的设计和验证至关重要，其技术与 NLP 类似，特别是通过在精选的数学数据集上训练 LLM。
   - 一种补充方法是基于 **Lean** 等系统的形式化数学推理，这些系统可以验证推理的正确性并提供反馈。
- **Meta 的 Yang 博士增强了数学领域的 AI 能力**：Meta FAIR 的研究科学家 **Kaiyu Yang 博士** 专注于通过集成 **Lean** 等形式化系统来增强 AI 的数学推理能力。
   - 他的工作探索了使用 LLM 执行诸如定理证明（生成形式化证明）和自动形式化（将非形式化语言翻译为形式化语言）等任务。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1358918518338355270)** (1 条消息): 

> `Manifold Research Group, Multimodal AI, self-assembling space robotics, robotic metacognition, Community Research Call #4` 


- **Manifold Research Group 主办 Research Call #4**：Manifold Research Group 将于本周六（4/12 @ 9 AM PST）举办 [Community Research Call #4](https://lu.ma/wlne416w)。
   - 本次会议将涵盖他们在 **Multimodal AI**、**self-assembling space robotics** 以及 **robotic metacognition** 方面的最新工作。
- **太空机器人研究起飞**：一位来自 Manifold Research Group、专注于太空机器人集群（robotic swarms）的博士生发出了研究会议邀请。
   - 该会议旨在促进合作并探索太空机器人领域的前沿科学。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1358969916283162864)** (1 条消息): 

> `Codeium rename, Windsurf Reddit, Windsurf Plugins` 


- **Codeium 更名为 Windsurf**：继 2024 年 11 月 **Windsurf Editor** 发布并获得*惊人的采用率*后，Codeium 已正式更名为 **Windsurf**。
   - 根据其 [更名公告](https://windsurf.com/blog/windsurf-rebrand-announcement)，新名称更好地体现了他们*结合人类与机器，创造毫不费力的强大体验*的愿景。
- **Windsurf 推出新的 SubReddit**：该公司为社区推出了一个新的 [SubReddit](https://www.reddit.com/r/windsurf)。
   - 该公告是随着 Discord 服务器的更改一同发布的，包括页面更新和频道重命名。
- **Codeium Extensions 现更名为 Windsurf Plugins**：随着品牌重塑，**Codeium Extensions** 现已正式更名为 **Windsurf Plugins**。
   - 公司承诺将以同样的创新精神，一波接一波地持续改进 **Windsurf Editor**。


  

---


---


{% else %}


> 完整的各频道详细解析已针对电子邮件进行缩减。 
> 
> 如果您想查看完整解析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}