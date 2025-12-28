---
companies:
- deepseek
- nvidia
- google-deepmind
date: '2024-11-21T02:41:02.660253Z'
description: '**DeepSeek** 发布了 **DeepSeek-R1-Lite-Preview**，这是一款开源推理模型，在数学基准测试中达到了
  **o1-preview 级别的性能**。该模型具有透明的思维过程，在实时问题解决方面展现出巨大潜力。


  **英伟达 (NVIDIA)** 报告第三季度营收达到创纪录的 **351 亿美元**，其中数据中心业务同比增长 **112%**，这主要受到 **Hopper**
  和 **Blackwell 架构** 的推动，后者提供了 **2.2 倍的性能提升**。


  **Google DeepMind** 推出了 **AlphaQubit**，这是一个量子计算系统，旨在改进纠错能力，其表现优于领先的解码器，尽管在扩展性和速度方面仍面临挑战。AI
  社区将继续关注 **推理模型**、**基准测试** 以及 **量子纠错** 领域的进展。'
id: cf470da6-b5f6-4356-ac97-101745d71f29
models:
- deepseek-r1-lite-preview
- o1-preview
- hopper
- blackwell
- alphaqubit
original_slug: ainews-deepseek-r1-claims-to-beat-o1-preview-and
people:
- yann-lecun
title: DeepSeek-R1 声称超越了 o1-preview，并且将会开源。
topics:
- reasoning
- benchmarking
- quantum-error-correction
- quantum-computing
- model-performance
- model-release
---

<!-- buttondown-editor-mode: plaintext -->**Whalebros 就够了。**

> 2024年11月20日至11月21日的 AI 新闻。我们为您检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 30 个 Discord（217 个频道和 1837 条消息）。预计节省阅读时间（按 200wpm 计算）：**197 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

自从 o1 发布以来（我们的报道见[此处](https://buttondown.com/ainews/archive/ainews-o1-openais-new-general-reasoning-models/)、[此处](https://buttondown.com/ainews/archive/ainews-learnings-from-o1-ama/)和[此处](https://buttondown.com/ainews/archive/ainews-o1-destroys-lmsys-arena-qwen-25-kyutai/)），“开源”复现的竞赛就已拉开帷幕。两个月后，在提及 [Nous Forge Reasoning API](https://x.com/NousResearch/status/1856417883934601246) 和 [Fireworks f1](https://fireworks.ai/blog/fireworks-f1) 的同时，[DeepSeek 似乎](https://x.com/deepseek_ai/status/1859200141355536422)已经做出了第一次令人信服的尝试，它 1) 拥有比 o1-preview *更好* 的基准测试结果，并且 2) 拥有公开可用的 Demo 而不是等待名单。


![image.png](https://assets.buttondown.email/images/7225b5d6-d19f-4bda-b827-417771eab45b.png?w=960&fit=max)


在基准测试方面，它并没有全面超越 o1，但在重要的数学基准测试上表现出色，并且除了 GPQA Diamond 之外，在所有其他测试上至少比同类产品更好。


![image.png](https://assets.buttondown.email/images/62d1286c-683b-4083-a07a-cbb5cc21d1e0.png?w=960&fit=max)


同样重要的是，他们似乎复现了 OpenAI 提到的类似的推理时间扩展（inference-time-scaling）性能提升，但这次带有了实际的 x 轴：


![image.png](https://assets.buttondown.email/images/be08c1f6-c666-4032-bafc-38bd05fa5da9.png?w=960&fit=max)


至于 “R1-Lite” 的命名，[传闻](https://x.com/nrehiew_/status/1859265550767067518)（基于[微信公告](https://x.com/phill__1/status/1859263165000729024)）它是基于 DeepSeek 现有的 V2-Lite 模型，该模型仅是一个具有 2.4B 激活参数的 16B MoE —— 这意味着如果他们能够成功扩大规模，“R1-full” 将是一个绝对的怪物。

一个值得注意的结果是，它在 [Yann LeCun 钟爱的 7 档齿轮问题](https://x.com/nrehiew_/status/1859268539770900923)上表现（虽然不一致）良好。

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

**1. NVIDIA 财务更新与市场洞察**

- **NVIDIA 报告第三季度创纪录营收**：[@perplexity_ai 讨论了](https://twitter.com/perplexity_ai/status/1859371790826701288)来自 NVIDIA 第三季度财报电话会议的见解，强调了 **351 亿美元的创纪录营收**，较上一季度**增长 17%**。主要增长驱动力包括**强劲的数据中心销售**以及对 NVIDIA **Hopper 和 Blackwell 架构**的需求。该公司预计将继续增长，第四季度预测营收为 375 亿美元。
  
- **财报电话会议中的详细表现**：[来自 @perplexity_ai 的另一份更新](https://twitter.com/perplexity_ai/status/1859361535577268698)进一步指出，**数据中心营收达到 308 亿美元**，同比增长 **112%**。据报道，Blackwell 架构比 Hopper 提供了 **2.2 倍的性能提升**。

**2. DeepSeek-R1-Lite-Preview：新型推理模型进展**

- **DeepSeek-R1-Lite-Preview 发布**：[@deepseek_ai 对发布 **DeepSeek-R1-Lite-Preview** 感到兴奋](https://twitter.com/deepseek_ai/status/1859200141355536422)，该模型在 MATH 基准测试上提供了 **o1-preview 级别性能**，并具有透明的思考过程。该模型的目标是尽快推出开源版本。

- **DeepSeek-R1-Lite-Preview 的评估**：[多位用户（如 @omarsar0）](https://twitter.com/omarsar0/status/1859373413439066590)讨论了它的能力，包括**数学推理能力的提升**以及在代码任务中的挑战。尽管存在一些小瑕疵，该模型在实时问题解决和推理方面展现出了潜力。

**3. AlphaQubit 在量子计算方面的进展**

- **AlphaQubit 与 Google 的合作**：[@GoogleDeepMind 介绍了 AlphaQubit](https://twitter.com/GoogleDeepMind/status/1859273133234192598)，这是一个旨在提高量子计算纠错能力的系统。该系统表现优于领先的算法解码器，并在规模化场景中显示出潜力。

- **量子纠错中的挑战**：尽管取得了这些进展，[来自 Google DeepMind 的额外见解](https://twitter.com/GoogleDeepMind/status/1859273153534681383)指出，在扩展和速度方面仍存在持续性问题，强调了使**量子计算机更加可靠**的目标。

**4. GPT-4o 的发展与 AI 创意增强**

- **GPT-4o 增强的创意写作**：[@OpenAI 指出了](https://twitter.com/OpenAI/status/1859296125947347164) GPT-4o 在生成更自然、更具吸引力的内容方面的更新。[用户评论（如来自 @gdb）](https://twitter.com/gdb/status/1859329768707195161)强调了在处理文件和提供更深层见解方面的改进。

- **Chatbot Arena 排名更新**：[@lmarena_ai 分享了](https://twitter.com/lmarena_ai/status/1859307979184689269)对 ChatGPT-4o 登上榜首的兴奋，它在创意写作和技术性能方面有显著提升，超越了 Gemini 和 Claude 模型。

**5. AI 实现与工具**

- **LangChain 和 LlamaIndex 系统**：[@LangChainAI 宣布了](https://twitter.com/LangChainAI/status/1859250598698422392)平台的更新，重点关注可观测性、评估和 Prompt Engineering。他们强调无缝集成，为开发者提供完善 LLM 应用的全面工具。

- **AI 游戏开发课程**：[@togethercompute 与行业领导者合作推出了](https://twitter.com/togethercompute/status/1859315685974999413)一门关于构建 AI 驱动游戏的课程。它专注于集成 LLM 以创建沉浸式游戏。

**6. 迷因/幽默**

- **高中 AI 怀旧**：[@aidan_mclau 幽默地反思了](https://twitter.com/aidan_mclau/status/1859378924771254734)使用 AI 完成哲学作业的经历，展现了对 AI 教育用途的轻松调侃。

- **国际象棋迷因**：[@BorisMPower 参与了一个国际象棋迷因话题](https://twitter.com/BorisMPower/status/1859070338111599005)，思考游戏背景下的战略举措和决策。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek R1-Lite 在数学基准测试中追平 o1-preview，开源版即将推出**

- **DeepSeek-R1-Lite 预览版正式发布** ([Score: 189, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1gvnhob/deepseekr1lite_preview_version_officially_released/)): **DeepSeek** 发布了其全新的 **R1 系列推理模型**，该模型采用 **reinforcement learning**（强化学习）训练，具备强大的反思和验证能力，其 **chain of thought**（思维链）推理长度可达**数万字**。该模型在**数学**、**代码**和**复杂逻辑推理**任务中的表现与 **o1-preview** 相当，并在 [chat.deepseek.com](http://chat.deepseek.com) 提供了透明的推理过程展示。
  - **DeepSeek-R1-Lite** 目前仍处于开发阶段，官方公告确认其目前仅限网页端使用，暂不提供 API 访问。根据其 [推文](https://x.com/deepseek_ai/status/1859200141355536422) 透露，公司计划**开源**完整的 **DeepSeek-R1 模型**，发布技术报告，并部署 API 服务。
  - 初步的用户测试显示，该模型在**数学**方面表现出色，具有详尽的推理步骤，尽管部分用户注意到其响应时间比 **o1-preview** 更长。根据 DeepSeek 之前的发布记录，该模型被推测拥有 **15B 参数**。
  - 社区反应凸显了**中国 AI 实验室**在 GPU 受限的情况下取得的快速进展，用户指出该模型透明的思考过程可能惠及开源社区的发展。多位用户证实了其在 **AIME 和 MATH 基准测试**中的强劲表现。

- **[中国 AI 初创公司阶跃星辰 (StepFun) 凭借其全新的 1 万亿参数 MOE 模型在 livebench 排名靠前](https://i.redd.it/tqgyvi01ky1e1.jpeg)** ([Score: 264, Comments: 74](https://reddit.com/r/LocalLLaMA/comments/1gvdnzi/chinese_ai_startup_stepfun_up_near_the_top_on/)): **StepFun**（阶跃星辰）开发了一款 **1 万亿参数的 Mixture-of-Experts (MOE) 模型**，在实时 AI 模型排行榜 **livebench** 上取得了极具竞争力的分数。原始资料中未披露该模型的具体性能指标和技术细节。
  - **Livebench 评分**显示，相对于其庞大的体量，该模型目前的**表现不及预期**。用户注意到它被更小的模型如 **o1 mini**（估计为 **70-120B 参数**）击败，且**数学得分**尤其低。
  - 该模型似乎处于早期训练阶段，讨论中提到的“**Step 2**”可能暗示其正处于第二阶段训练。用户推测其表现平平是由于**严重训练不足**，而非架构限制。
  - 讨论集中在该模型的 **MoE 架构**和部署策略上，专家指出每个 **transformer 层**都需要自己的一套专家系统，这导致在推理和训练期间产生大量的 **GPU-to-GPU 通信**需求。

**主题 2：复杂的开源 LLM 工具：研究助手与记忆框架**

- **我创建了一个真正能做研究的 AI 研究助手！给它任何主题，它会搜索网页、抓取内容、保存来源，并为你提供完整的研究文档 + 摘要。使用 Ollama (免费) - 只需提问，让它开始工作！无 API 成本，开源，本地运行！** ([Score: 487, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/)): **Automated-AI-Web-Researcher** 是一款基于 **Python** 的工具，它利用 **Ollama** 和**本地 LLM** 进行全面的网络研究。它可以根据单个查询自动生成多达 **5 个特定的研究重点**，持续搜索和抓取内容并保存来源，最后创建包含摘要的详细研究文档。该项目已在 [GitHub](https://github.com/TheBlewish/Automated-AI-Web-Researcher-Ollama) 上开源，完全在本地运行，支持 **phi3:3.8b-mini-128k-instruct** 或 **phi3:14b-medium-128k-instruct** 等模型，具备暂停/恢复功能，并允许用户针对收集的研究内容进行追问。
  - 用户反馈在不同 **LLM** 上的成功率各异——虽然有些人在使用 **Llama3.2-vision:11b** 和 **Qwen2.5:14b** 时遇到了生成空摘要的问题，但另一些人成功使用了 **mistral-nemo 12B**，在 **16GB VRAM** 下实现了 **38000 上下文长度**，CPU 占用率为 3%，GPU 占用率为 97%。
  - 社区提出了一些技术建议，包括忽略 **robots.txt**、增加对 **OpenAI API** 兼容性的支持（后来通过 PR 实现），以及使用 "lib" 文件夹重构代码库，并利用 **pydantic** 或 **omegaconf** 等工具进行规范的配置管理。
  - 关于该工具用途的讨论强调了其在寻找和总结真实研究方面的价值，而非仅仅生成内容，同时也对来源验证和网页抓取信息的真实准确性提出了担忧。

- **Agent Memory** ([Score: 64, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1gvhpjj/agent_memory/)): 多个 GitHub 项目对 **LLM Agent 内存框架**进行了对比，关键实现包括 **Letta**（基于 MemGPT 论文）、**Memoripy**（支持 Ollama 和 OpenAI）以及 **Zep**（维护时序知识图谱）。多个框架通过 **Ollama** 和 **vLLM** 支持**本地模型**，尽管许多框架默认假设具有 **GPT 访问权限**，且对开源替代方案的兼容程度各不相同。

  - 对比涵盖了活跃项目如 **cognee**（用于文档摄取）和 **MemoryScope**（具有内存整合功能），以及开发资源如 **LangGraph Memory Service** 模板和用于 RAG 实现的 **txtai**，大多数框架通过 **LiteLLM** 等工具提供 **OpenAI 兼容 API** 支持。
  - **基于向量的内存系统**使用邻近度和重排序（reranking）来确定相关性，这与 **Kobold** 或 **NovelAI** 中使用的简单**关键词激活**系统形成对比。向量方法在空间上映射概念（例如，“汉堡王”比“英格兰国王”更接近食物相关的术语），并利用小型神经网络或直接通过 AI 评估进行重排序。
  - 内存框架的主要区别在于它们处理上下文注入的方式——从自动化到手动方法不等——更复杂的系统结合了**知识图谱**和**决策树**。内存处理可能会变得资源密集，有时消耗的 Token 甚至超过了实际对话。
  - **LLM 内存系统**领域仍处于实验阶段，尚未建立最佳实践，涵盖了从基础的设定集（lorebook）风格实现到复杂的上下文感知解决方案。简单的系统需要更多的人工监督来纠正错误，而复杂的系统在应对上下文错误方面表现出更强的鲁棒性。


**主题 3. 硬件与浏览器优化：Pi GPU 加速与 WebGPU 实现**

- **[LLM 硬件加速——在 Raspberry Pi 上（使用低成本 Pi 作为基础计算机的高端 AMD GPU）](https://www.youtube.com/watch?v=AyR7iCS7gNI)** ([Score: 53, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1gvdrvj/llm_hardware_accelerationon_a_raspberry_pi_topend/)): **Raspberry Pi** 配置可以通过 **Vulkan** 图形处理运行具有 **AMD GPU 加速**的 **Large Language Models (LLMs)**。这种硬件设置将 **Raspberry Pi** 的性价比与高端 **AMD GPU** 的处理能力相结合。
  - 使用 **6700XT** GPU 配合 **Vulkan** 后端实现了 **40 t/s** 的 **Token 速率**，而使用 **RTX 3060** 配合 **CUDA** 的速率为 **55 t/s**。**ARM** 平台上缺乏 **ROCm** 支持显著限制了性能潜力。
  - 一套完整的 **Raspberry Pi** 设置成本约为 **383 美元**（不含 GPU），而同类的 **x86** 系统（如 **ASRock N100M**）成本为 **260-300 美元**。**Intel N100** 系统仅多消耗 **5W** 功率，同时提供更好的兼容性和性能。
  - 用户指出，**AMD** 可能会开发一种专用产品，在**类 NUC** 的外形尺寸中结合基础 **APU** 和高 **VRAM GPU**。即将发布的 **Strix Halo** 可能会测试市场需求，尽管像双 **P40**（500 美元）这样的替代方案仍具竞争力。


- **由 Qwen2.5-Coder 驱动的浏览器内网站生成器** ([Score: 55, Comments: 8](https://reddit.com/r/LocalLLaMA/comments/1gv73fn/inbrowser_site_builder_powered_by_qwen25coder/)): 一个在浏览器中运行的 **AI 网站生成器**，使用 **WebGPU**、**OnnxRuntime-Web**、**Qwen2.5-Coder** 和 **Qwen2-VL** 从文本、图像和语音输入生成代码，尽管由于性能限制，目前仅上线了文本转代码功能。该项目实现了 **Moonshine** 用于语音转文本，并在 [GitHub](https://github.com/pdufour/llm-coder/blob/main/src/hooks/useSpeech.js) 和 [Huggingface](https://huggingface.co/spaces/pdufour/Qwen2-VL-2B-Instruct-ONNX-Q4-F16/blob/main/index.js) 上提供了集成代码示例，目前性能受限于 GPU 能力，主要在 **Mac** 系统上进行了测试。
  - 开发者详细介绍了**模型转换**中的挑战，通过[导出文档](https://huggingface.co/pdufour/Qwen2-VL-2B-Instruct-ONNX-Q4-F16/blob/main/EXPORT.md)和自定义 **Makefile** 分享了他们的流程，并指出**混合数据类型**和**内存管理**问题使该项目变得尤为困难。
  - 社区反馈显示了在 **Linux** 和 **NVIDIA RTX** 硬件上测试该系统的兴趣，同时也有用户反映由于背景颜色相近，在 **iPhone** 设备上存在 **UI 对比度问题**。


**主题 4. 模型架构：GPT-4、Gemini 及其他闭源模型分析**

- **闭源模型参数量推测** ([Score: 52, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1gve7sk/closed_source_model_size_speculation/)): 该帖子分析了**闭源 LLM** 的参数量，认为 **GPT-4 Original** 拥有 **280B 激活参数**和 **1.8T 总参数**，而 **GPT-4 Turbo** 和 **GPT-4o** 等更新版本的激活参数量逐渐减少（分别约为 **~93-94B** 和 **~28-32B**）。分析将模型架构与定价联系起来，将**微软的 Grin MoE** [论文](https://arxiv.org/pdf/2409.12136)与 **GPT-4o Mini**（**6.6B-8B** 激活参数）挂钩，并将 **Gemini Flash** 版本（**8B**、**32B** 和 **16B** 稠密）与 **Qwen** 等模型以及 **Huyuan** 和 **Yi Lightning** 的架构进行了对比。
  - **Qwen 2.5** 的性能参数比支持了现代模型激活参数更小的理论，特别是在 **MoE architecture** 和闭源研究取得进展的情况下。讨论表明 **Claude** 的效率可能低于 **OpenAI** 和 **Google** 的模型。
  - **Gemini Flash** 的 **8B** 参数量可能包含了视觉模型，使得核心语言模型约为 **7B parameters**。该模型在此规模下的性能被认为非常出色。
  - 社区估计 **GPT-4 Turbo** 拥有 **~1T** 参数（**100B** 激活），**GPT-4o** 拥有 **~500B**（**50B** 激活），而 **Yi-Lightning** 根据其低廉的定价和推理能力，规模可能更小。**Step-2** 由于定价较高（输入 **$6/M**，输出 **$20/M**），估计规模更大。
- **[Judge Arena 排行榜：将 LLM 作为评估者的基准测试](https://i.redd.it/rcrq5uh6r02e1.png)** ([Score: 33, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1gvl5x5/judge_arena_leaderboard_benchmarking_llms_as/)): **Judge Arena Leaderboard** 旨在测试 **LLM** 评估和评判其他 AI 输出的能力。由于帖子正文缺乏背景信息，本摘要无法包含有关方法论、指标或参与模型的具体细节。
  - **Claude 3.5 Sonnet** 最初在 **Judge Arena** 排行榜上领先，但随后的更新显示出显著的波动，**7B models** 在开源条目中升至顶级位置。在 **1197 votes** 后，排名显示 **ELO spread** 从约 400 分压缩至约 250 分。
  - 社区成员质疑结果的有效性，特别是关于 **Mistral 7B (v0.1)** 表现优于 **GPT-4**、**GPT-3.5** 和 **Claude 3 Haiku** 的情况，高误差范围（约 100 ELO points）被认为是可能的解释。
  - 批评者指出了 **judgment prompt** 的局限性，认为其缺乏具体的评估标准和深度，而要求忽略回复长度的指令可能会通过“粉红大象效应”反过来影响评估者。


## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. 实时 AI 面部识别演示引发隐私警报**

- **[这位荷兰记者演示了实时 AI 面部识别技术，识别出了正在与之交谈的人。](https://v.redd.it/lb0h1g0st12e1)** ([Score: 2523, Comments: 304](https://reddit.com/r/ChatGPT/comments/1gvo603/this_dutch_journalist_demonstrates_realtime_ai/)): **荷兰记者**通过在现场对话中识别个人，展示了**实时面部识别 AI** 的能力。没有提供关于所使用的具体技术或实现的额外背景或技术细节。
  - 获赞最高的评论强调了**隐私担忧**，建议“**不要在网上任何地方发布附带真实姓名的照片**”，获得了 **457** 个赞。多位用户讨论了继续佩戴**口罩**以及规避面部识别的方法。
  - 讨论显示这可能使用了 **Pimeyes** 或类似技术，用户指出 **Clearview AI** 拥有更先进的能力，可以“*在音乐会的人群中找到你的脸*”。几位用户指出，演示可能涉及第二个人进行手动搜索。
  - 用户辩论了社会影响，一些人称其为“**对民主和自由的威胁**”，而另一些人则讨论了诸如**汽车销售**之类的实际应用。对话包括对政府监控和数据隐私的担忧，特别是提到**中国**和其他国家。


**主题 2. CogVideoX 1.5 图生视频：质量与性能的权衡**

- **[CogvideoX 1.5 img2vid 对比 - BF16 vs FP8](https://v.redd.it/v531xs8o412e1)** ([Score: 165, Comments: 49](https://reddit.com/r/StableDiffusion/comments/1gvm302/comparison_of_cogvideox_15_img2vid_bf16_vs_fp8/)): **CogVideoX 1.5** 帖子缺乏足够的上下文或内容，无法生成关于 **BF16** 和 **FP8** 实现对比的有意义技术摘要。帖子正文中未提供分析这些数值格式之间质量差异的细节。
  - **性能指标**显示出显著差异：在 **RTX 3060 12GB** 上，生成 **1360x768** 分辨率的 **24 帧**，**BF16** 耗时 **12分57秒**，而 **FP8** 仅需 **7分57秒**。由于 **OOM 错误**，**BF16** 需要 **CPU offload**，但能提供更稳定的结果。
  - **CogVideoX 1.5** 面临量化挑战，无法在 **FP16** 下运行。在可用选项中，**TorchAO FP6** 提供了最佳质量结果，而 **FP8DQ** 和 **FP8DQrow** 由于支持 **FP8 scaled matmul**，在 **RTX 4090** 上表现出更快的性能。
  - 在 **Windows** 上安装需要使用 [TorchAO v0.6.1](https://github.com/pytorch/ao/releases/tag/v0.6.1) 进行特定设置，并修改 `base.h` 文件中的代码，将 `FragM` 定义更改为 `Vec<unsigned int, 1>`。


**Theme 3. 10 个 AI Agent 实时协作创作小说**

- **[10 个自主 AI Agent 实时创作小说](https://i.redd.it/dfxwtvmpg12e1.png)** ([Score: 277, Comments: 153](https://reddit.com/r/ChatGPT/comments/1gvn049/a_novel_being_written_in_realtime_by_10/)): **10 个自主 AI Agent** 实时协作创作一部小说，尽管帖子正文中未提供有关过程、实现或结果的更多细节。该概念暗示了一项多 Agent 创意写作和 AI 协作的实验，但由于缺乏进一步背景，无法总结具体的技术细节。
  - 用户对 **AI 生成的长篇内容**表示出明显的**怀疑**，许多人指出 **ChatGPT 在超过几页后就难以保持连贯性**，并且经常**遗忘情节要点和角色**。获 **178 个赞** 的热门评论强调了这一局限性。
  - 作者解释了他们维持叙事连贯性的解决方案：通过一个**基于文件的协作系统**，多个 Agent 访问**全局地图**、**内容摘要**和**运行变更日志**，而不是依赖于单个上下文窗口。该系统目前正处于使用 **Qwen 2.5** 进行准备和结构化的阶段。
  - 几位用户辩论了 AI 生成小说的**艺术价值**和**目的**，认为文学从根本上是关于表达人类经验和建立人类联系。批评者指出，像 **ChatGPT** 和 **Claude** 这样的 AI 模型可能会避开让小说变得有趣的争议性话题。


**Theme 4. StepFun 的 1T 参数模型在 LiveBench 排名上升**

- **[中国 AI 初创公司 StepFun 的新型 1 万亿参数 MOE 模型在 LiveBench 排名靠前](https://i.redd.it/p01x5ci4j02e1.png)** ([Score: 29, Comments: 0](https://reddit.com/r/OpenAI/comments/1gvkjib/chinese_ai_startup_stepfun_up_near_the_top_on/)): **StepFun**（一家**中国 AI 初创公司**）开发了一个 **1 万亿参数的混合专家 (MOE) 模型**，在 **LiveBench** 上名列前茅。该模型的表现展示了中国公司在大规模 AI 模型开发领域日益增强的竞争力。

- **[Microsoft CEO 表示，与其说 AI Scaling Laws 撞墙了，不如说我们正看到推理时 (inference) 计算的新 Scaling Law 出现](https://v.redd.it/c8tfecx1y22e1)** ([Score: 99, Comments: 40](https://reddit.com/r/OpenAI/comments/1gvsw59/microsoft_ceo_says_that_rather_than_seeing_ai/)): **Microsoft CEO** 讨论了关于 **AI Scaling Laws** 的观察，指出与其说遇到了计算限制，不如说有证据表明专门针对**推理时 (test-time inference) 计算**出现了新模式。帖子正文缺乏具体细节或引用，限制了对该观察的声明或支持证据的进一步分析。
  - 讨论表明，**推理时计算**涉及允许模型“思考”更长时间并对输出进行迭代，而不是接受第一反应，其准确率随思考时间呈**对数级**增长。这代表了除传统训练计算缩放之外的第二个缩放因子。
  - 包括 **Pitiful-Taste9403** 在内的几位用户将此解释为**参数缩放 (parameter scaling)** 已达极限的证据，导致公司将推理优化作为 AI 进步的替代路径。
  - “**Scaling Law**”一词引发了辩论，用户将其与**摩尔定律 (Moore's Law)** 进行比较，认为它更多是一种趋势而非基本定律。一些人对这些发展对普通人的经济影响表示怀疑。


---

# AI Discord Recap

> O1-mini 对“摘要之摘要”的总结

**主题 1. 定制化模型部署占据核心地位**

- [**在 Hugging Face 上部署定制化 AI 模型**](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler)：开发者现在可以使用 `handler.py` 文件在 Hugging Face 上部署量身定制的 **AI 模型**，从而实现自定义的前处理和后处理。
  
  - 这一进展利用 **Hugging Face endpoints** 增强了模型的灵活性，并使其能够更好地集成到各种应用程序中。
- [**DeepSeek-R1-Lite-Preview 性能比肩 OpenAI 的 o1-Preview**](https://x.com/deepseek_ai/status/1859200141355536422)：**DeepSeek** 发布了 **R1-Lite-Preview**，在 **AIME** 和 **MATH** 基准测试中达到了 **o1-preview 级别的性能**。
  
  - 该模型不仅反映了 **OpenAI** 的进步，还引入了可实时访问的**透明推理过程**。
- [**腾讯混元模型微调现已开放**](https://huggingface.co/spaces/tencent/Hunyuan-Large)：用户可以通过 [GitHub 仓库](https://github.com/Tencent/Tencent-Hunyuan-Large)和[官方 Demo](https://huggingface.co/spaces/tencent/Hunyuan-Large) 等资源微调 [**腾讯混元模型 (Tencent Hunyuan model)**](https://huggingface.co/tencent/Tencent-Hunyuan-Large)。
  
  - 这为各种 NLP 任务提供了更强的定制化能力，扩展了模型的适用性。

**主题 2. AI 模型性能与优化飞速发展**

- [**SageAttention2 使推理速度翻倍**](https://arxiv.org/html/2411.10958v1)：[**SageAttention2 技术报告**](https://arxiv.org/html/2411.10958v1) 揭示了一种 **4-bit 矩阵乘法**方法，在 **RTX40/3090** GPU 上实现了比 FlashAttention2 快 **2 倍的加速**。
  
  - 这项创新可以作为无缝替换方案，在不牺牲准确性的情况下显著加速**推理 (inference)**。
- [**GPT-4o 获得创意提升和文件处理增强**](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：**OpenAI** 更新了 **GPT-4o**，提升了其**创意写作**能力，并改进了**文件处理**以获得更深刻的见解。
  
  - 改进后的模型在聊天机器人竞赛的**编程 (coding)**和**创意写作**等类别中重返榜首。
- [**关于通过模型量化提升性能的讨论**](https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation)：用户对**量化 (quantization)**对模型性能产生的负面影响表示担忧，相比量化版本更倾向于原始模型。
  
  - 建议包括要求 **OpenRouter** 等提供商进行更清晰的信息披露，并探索修改评估库以适配剪枝模型。

**主题 3. 创新 AI 研究开辟新路径**

- [**ACE 方法增强模型控制**](https://arxiv.org/abs/2411.09003)：**EleutherAI** 引入了 **ACE (Affine Concept Editing)** 方法，将概念视为仿射函数，以更好地控制**模型响应**。
  
  - 在 **Llama 3 70B** 等模型上的测试表明，ACE 在处理有害和无害提示词的**拒绝行为**方面优于以往技术。
- [**Scaling Laws 揭示低维能力空间**](https://arxiv.org/abs/2405.10938)：一篇新论文指出，**语言模型性能**受**低维能力空间**的影响比单纯的多维扩展更大。
  
  - 来自 **Apollo** 的 **Marius Hobbhahn** 倡导推进评估科学，强调严谨的**模型评估**实践。
- [**Generative Agents 模拟超过 1,000 名真实个体**](https://arxiv.org/abs/2411.10109)：一种新型架构有效地模拟了 **1,052 名真实人类**的态度和行为，在综合社会调查 (General Social Survey) 中达到了 **85% 的准确率**。
  
  - 这减少了不同**种族和意识形态群体**之间的准确性偏差，为探索社会科学中的个人和集体行为提供了强大的工具。

**主题 4. AI 工具集成与社区支持蓬勃发展**

- [**Aider 的安装难题通过强制重装解决**](https://aider.chat/docs/troubleshooting/edit-errors.html)：遇到 **Aider** 安装问题（特别是 API keys 和环境变量问题）的用户发现，通过执行**强制重装**可以成功解决。
  
  - 该方案简化了设置流程，使 **DeepSeek-R1-Lite-Preview** 和其他模型的集成更加顺畅。
- [**LM Studio 结合云端方案应对硬件限制**](https://github.com/unslothai/unsloth/wiki#-we-are-hiring)：成员们讨论了在有限硬件上运行 **DeepSeek v2.5 Lite** 的问题，强调需要至少 **24GB VRAM** 的 GPU。

- 云端硬件租赁被视为具有成本效益的替代方案，提供高速模型访问，且不受本地硬件限制。
- [**Torchtune 的自适应批处理优化 GPU 利用率**](https://github.com/pytorch/torchtune/pull/2035)：在 **Torchtune** 中实现 **adaptive batching** 旨在通过动态调整 batch sizes 来防止 **OOM errors**，从而最大化 **GPU utilization**。
  
  - 建议将此功能作为 flag 集成到未来的 recipes 中，以增强训练效率和资源管理。

**主题 5. 前沿 AI 发展应对多样化挑战**

- [**LLMs 在没有显式提示的情况下表现出内在推理能力**](https://arxiv.org/abs/2402.10200)：研究表明，通过调整解码过程，**large language models (LLMs)** 可以在没有显式 prompting 的情况下展示出类似于 **chain-of-thought (CoT)** 的推理路径。
  
  - 通过调整以考虑 **top-$k$ alternative tokens**，揭示了 LLMs 固有的推理能力，减少了对手动 prompt engineering 的依赖。
- [**Perplexity AI 在 API 挑战中推出购物功能**](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A)：**Perplexity AI** 推出了新的 **Shopping** 功能，引发了关于其仅限美国市场的讨论，同时用户正面临 **API response consistency** 的问题。
  
  - 尽管是 **Pro** 用户，一些成员仍对限制表示沮丧，导致对 **ChatGPT** 等替代方案的依赖增加。
- [**OpenRouter 解决模型描述和缓存澄清问题**](https://x.com/natolambert/status/1859255627882664034?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：用户发现 **OpenRouter** 上 **GPT-4o** 模型描述存在差异，促使官方快速修复了 model cards。
  
  - 用户寻求关于不同供应商之间 **prompt caching** 策略的澄清，并对 **Anthropic** 和 **OpenAI** 的协议进行了比较。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **现在可以部署自定义 AI 模型**：一位成员发现可以使用 `handler.py` 文件在 Hugging Face 上部署自定义 **AI models**，从而实现量身定制的模型前后处理。
  
  - 该过程涉及指定请求和响应的处理方法，通过 [Hugging Face endpoints](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler) 增强自定义能力。
- **发布关于 AI 安全见解的新论文**：**Redhat/IBM** 的 AI 研究人员发表了一篇论文，探讨了公开可用 AI 模型的安全影响，重点关注风险和生命周期管理。
  
  - 该论文概述了提高开发者和用户安全性的策略，旨在 AI 社区内建立更标准化的实践。[查看论文](https://huggingface.co/papers/2411.12275)。
- **自动化 AI 研究助手问世**：一个使用本地 LLMs 构建的 **Automated AI Researcher** 被创建，用于根据用户查询生成研究文档。
  
  - 该系统利用网页抓取来汇编信息并生成与主题相关的摘要和链接，使研究更加便捷。
- **LangGraph 学习倡议**：用户 `richieghost` 发起了围绕 **LangGraph** 的学习，讨论了其在社区中的应用和发展。
  
  - 这突显了人们对在 AI 模型中集成基于图的技术的持续兴趣。
- **Ada 002 的语义搜索挑战**：使用 OpenAI **Ada 002** 的 **Semantic search** 会优先考虑主导话题，导致不太突出但相关的句子排名较低。
  
  - 用户正在寻求 **semantic search** 的替代方案，以提高信息提取的有效性。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **o1 发布传闻**：有关 **OpenAI o1 模型** 即将发布的传言甚嚣尘上，可能会与 **DevDay Singapore** 同步，尽管这些传闻尚未得到证实。
  
  - 一位成员指出，*“周三发布会很奇怪”*，这凸显了尽管存在不确定性，社区仍保持着高度的期待。
- **DeepSeek 的 RL 驱动**：围绕 **DeepSeek Prover** 的讨论揭示了人们对其 **reinforcement learning**（强化学习）应用的兴趣，尽管在模型大小和性能方面存在挑战，成员们仍期待可能发布的论文。
  
  - 社区正在考虑由于这些性能障碍而导致完整发布推迟的可能性。
- **GPT-4o 取得进展**：**OpenAI** 宣布了 **GPT-4o** 的更新，增强了其创意写作能力和文件处理能力，这使其在聊天机器人竞赛的创意写作和编程等性能类别中重返巅峰。
  
  - 此次更新强调了 **GPT-4o** 在相关性和可读性方面的提升，详见 [OpenAI 官方推文](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)。
- **LLM 学习循环**：最近的研究见解表明，**LLM 记忆训练样本**的方式会显著影响其泛化能力，模型在记忆之前先理解概念有助于更好地预测测试准确率。
  
  - **Katie Kang** 分享说，这种方法允许仅根据训练动态来预测测试结果。
- **NeurIPS NLP 被否决**：有人对 **NeurIPS D&B 审稿人** 驳回专注于 **韩语 LLM 评估** 的项目表示担忧，理由是中文领域已经存在类似的工作。
  
  - 社区成员认为每种语言都需要定制化的模型，并强调了 **NLP 发展** 中包容性的重要性，正如 [Stella Biderman 的推文](https://x.com/blancheminerva/status/1859271409429795083?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ) 所强调的那样。

 

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLM 微调技巧**：一位用户利用 **16-bit 版本** 成功将 **Llama 3.1** 导出到 [Hugging Face](https://github.com/unslothai/unsloth) 用于 **本地项目**，以增强构建 RAG 应用的性能。
  
  - 成员们建议使用 **16-bit 版本** 来优化 **fine-tuning** 能力，确保模型开发过程中的资源管理更加高效。
- **SageAttention2 加速推理**：[SageAttention2 技术报告](https://arxiv.org/html/2411.10958v1) 介绍了一种 **4-bit 矩阵乘法** 方法，其速度比 FlashAttention2 快 **2 倍**。
  
  - 凭借对 **RTX40/3090** 硬件的支持，SageAttention2 可作为 **FlashAttention2** 的直接替代方案，在不损失指标保真度的情况下增强 **推理加速**。
- **训练 Llama 模型**：多位成员分享了训练不同 **Llama 模型** 的经验，并指出根据 **模型参数** 和 **数据集大小**，成功程度各不相同。
  
  - 建议包括从基础模型开始，并调整 **训练步数** 以获得最佳 **性能**。
- **通过多 GPU 训练增强性能**：用户正在探索 Unsloth 的 **多 GPU 训练** 功能，目前该功能尚未推出，但预计很快发布。
  
  - 讨论了利用 **Llama Factory** 管理多个 GPU 等策略，以为即将推出的功能做准备。

 

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 安装挑战已解决**：用户在 **Aider** 的安装过程中遇到了问题，特别是关于 [API keys](https://aider.chat/docs/troubleshooting/edit-errors.html) 和环境变量的设置，导致部分用户尝试重新安装组件。
  
  - 一位用户报告称，执行 **force reinstall**（强制重装）成功解决了安装难题。
- **DeepSeek 性能令人印象深刻**：**DeepSeek-R1-Lite-Preview** 在 AIME 和 MATH 基准测试中的表现与 **o1-preview** 持平，且与之前的模型相比，响应速度更快。
  
  - 该模型透明的推理过程增强了其在编程任务中的有效性，允许用户实时观察其思考过程。
- **对 OpenRouter 模型质量的担忧**：用户对 **OpenRouter** 使用开源模型的量化版本表示不满，对其在 **Aider Leaderboard** 上的表现提出质疑。
  
  - 用户呼吁在排行榜上发布更清晰的警告，说明使用 **OpenRouter** 的量化模型时可能出现的性能差异。
- **讨论模型量化的影响**：量化会对模型性能产生负面影响，导致用户更倾向于选择原始模型而非量化版本。
  
  - 用户建议 **OpenRouter** 应当公开具体的模型版本，以便准确设定性能预期。
- **了解 Aider 的聊天模式**：成员们讨论了各种 **Aider** 聊天模式的效果，强调将 **o1-preview** 作为 Architect，并配合 **DeepSeek** 或 **o1-mini** 作为 Editor 使用可以获得最佳效果。
  
  - 一位用户指出，**Sonnet** 在处理日常任务时表现异常出色，且不需要复杂的配置。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **通过仿射编辑实现 ACE 模型控制**：作者介绍了一种新的 **ACE (Affine Concept Editing)** 方法，将概念视为仿射函数，以增强对模型响应的控制。ACE 能够将激活值投影到超平面上，在 **Gemma** 的测试中展示了在管理模型行为方面更高的精确度。
  
  - **ACE** 在包括 **Llama 3 70B** 在内的十个模型上进行了评估，在处理有害和无害提示词的拒绝行为控制方面取得了优异效果。该方法超越了以往的技术，为引导模型行为提供了更可靠的策略。
- **潜动作推动逆动力学**：一位用户询问了关于 **latent actions**（潜动作）和 **inverse dynamics models**（逆动力学模型）的顶级论文，表示对这些领域内最前沿的研究感兴趣。讨论强调了相关文献对于推进当前 AI 方法论的重要性。
  
  - 虽然没有引用具体的论文，但对话强调了探索 **latent actions** 和 **inverse dynamics models** 以突破现有 AI 框架界限的重要性。
- **缩放定律揭示能力维度**：一篇新发表的论文 [*Understanding How Language Model Performance Varies with Scale*](https://arxiv.org/abs/2405.10938) 提出了一种基于约 100 个公开模型的缩放定律（Scaling Laws）观察方法。作者认为，语言模型的性能更多地受到**低维能力空间**的影响，而不仅仅是在多个尺度上进行训练。
  
  - **Apollo 的 Marius Hobbhahn** 被公认为 AI 社区内推动评估方法科学化的领先倡导者，这突显了 AI 模型开发中对严谨评估实践日益增长的关注。
- **WANDA 剪枝增强模型效率**：一位成员询问 **lm-eval** 是否支持剪枝模型的 zero-shot 基准测试，并提到了使用 **WANDA** 剪枝方法。用户对从 zero-shot 评估中获得的**可疑结果**表示担忧。
  
  - 讨论内容包括对 **lm_eval** 进行修改以兼容剪枝模型，以及使用 **vllm** 在 **ADVBench** 上进行评估，并分享了具体的代码片段来演示模型加载和推理方法。
- **Forgetting Transformer 集成遗忘门**：**Forgetting Transformer** 论文介绍了一种在 softmax attention 机制中加入遗忘门的方法，解决了传统位置嵌入（position embeddings）的局限性。这种方法通过将遗忘门自然地集成到 **Transformer** 架构中，为循环序列模型提供了一种替代方案。
  
  - 社区讨论引用了诸如 [**Contextual Position Encoding (CoPE)**](https://arxiv.org/abs/2405.18719) 等相关工作，并分析了不同的位置嵌入策略，评估了像 **ALiBi** 或 **RoPE** 这样更简单的方法是否比最近复杂的方案集成效果更好。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 被 ChatGPT 超越**：用户将 [Perplexity](https://perplexity.supply/) 与 **ChatGPT** 进行了对比，强调了 **ChatGPT** 的多功能性和卓越的对话能力。
  
  - 尽管是 Perplexity 的 **Pro** 用户，一些人仍对其局限性表示沮丧，导致对 **ChatGPT** 的依赖增加。
- **推出 Perplexity Shopping 功能**：新的 **Perplexity Shopping** 功能引发了讨论，用户询问其是否为美国市场独有。
  
  - 社区对了解该购物功能的潜在访问限制表现出浓厚兴趣。
- **报告 API 功能问题**：用户报告称，尽管切换了模型，**API** 响应仍保持不变，造成了困惑和沮丧。
  
  - 社区讨论了平台的灵活性，并对响应的多样性提出了质疑。
- **Next.js 全栈开发见解**：分享了一个关于 [全栈 Next.js 开发](https://www.perplexity.ai/search/webapp-fullstack-nextjs-hono-b-W1pCGRCUSJmPBFklgsg.7w) 的资源，提供了对现代 Web 框架的见解。
  
  - *探索使用 Hono 进行服务端路由！*
- **NVIDIA AI 芯片过热担忧**：正如[这份报告](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A)中所详述，人们对 **NVIDIA AI 芯片过热**表示担忧。
  
  - 讨论强调了与长时间使用这些芯片相关的风险。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 1114 在处理输入方面存在困难**：用户报告称，**Gemini 1114** 在对话过程中经常忽略图像输入，导致产生幻觉响应，这与 **Grok vision Beta** 等模型不同。
  
  - 成员们希望得到确认和修复，对该模型反复出现的问题表示沮丧。
- **DeepSeek 发布新的推理模型**：新模型 **DeepSeek-R1-Lite-Preview** 正式发布，该模型拥有增强的推理能力，并在 **AIME** 和 **MATH** 基准测试中表现出色。
  
  - 然而，一些用户注意到该模型的运行速度较慢，引发了关于 **DeepInfra** 是否可能是更快替代方案的讨论。
- **关于 Prompt Caching 的澄清**：**Prompt Caching** 适用于 **DeepSeek** 等特定模型，用户对其他提供商的缓存策略提出了疑问。
  
  - 一些成员讨论了不同系统之间缓存工作方式的差异，特别提到了 **Anthropic** 和 **OpenAI** 的协议。
- **GPT-4o 模型描述问题**：用户发现新发布的 **GPT-4o** 存在差异，指出该模型错误地列出了 **8k context** 以及与 **GPT-4** 相关的错误描述。
  
  - 在指出错误后，成员们看到模型卡片得到了快速更新和修复，恢复了准确信息。
- **RP 模型对比**：成员们讨论了用于故事叙述和角色扮演（RP）的 **Claude** 替代方案，由于 **Hermes** 的质量和性价比，有人建议使用它。
  
  - 用户表示对这些模型的体验各异，有些人觉得 **Hermes** 更合适，而另一些人则继续忠于 **Claude**。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **受限硬件上的模型加载**：一位用户在 **36GB RAM M3 MacBook** 上的 LM Studio 中遇到了 **模型加载问题**，并强调了关于系统资源限制的错误消息。
  - 另一位成员建议在这种配置下避免使用 **32B 模型**，建议最高使用 **14B** 以防止过载。
- **LLM 的 GPU 和 RAM 需求**：讨论强调运行 **DeepSeek v2.5 Lite** 的 Q4_K_M 变体至少需要 **24GB VRAM**，而完整的 Q8 则需要 **48GB VRAM**。
  - 成员们更倾向于选择 **NVIDIA GPU** 而非 AMD，原因是驱动稳定性问题会影响性能。
- **云端解决方案 vs 本地硬件**：用户探讨了将 **云端硬件租赁** 作为本地配置的经济型替代方案，每月费用在 **$25 到 $50** 之间。
  - 这种方法可以在不受本地硬件限制的情况下访问高速模型。
- **针对 AI 工作负载的工作站设计**：一位成员寻求关于在 **$30,000 到 $40,000** 预算内构建用于微调 LLM 的工作站建议，考虑了选择多个 NVIDIA **A6000s** 还是较少的 **H100s**。
  - 讨论强调了显存和硬件灵活性在应对预算限制时的重要性。
- **模型推荐与偏好**：用户根据性能和写作质量推荐了各种模型，包括 **Hermes 3**、**Lexi Uncensored V2** 和 **Goliath 120B**。
  - 鼓励用户随着新选项的出现尝试不同的模型，以找到最适合个人用例的选择。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **海量游戏 PC 指南**：一位用户正在寻求预算在 **$2500** 以内的 **游戏 PC** 推荐，询问关于组件选择和购买渠道的建议。
  - 他们鼓励其他人发送私信以获取个性化建议。
- **角色一致性挑战**：一位成员询问如何在整个绘本中保持一致的 **角色设计**，因为在生成多张图像时难以应对变化。
  - 建议包括使用 **FLUX** 或图像转换技术来提高一致性。
- **AI 模型 vs. Substance Designer**：关于 **AI 模型** 是否能有效替代 **Substance Designer** 引发了讨论，强调了在该领域进一步探索的必要性。
  - 成员们分享了对不同 AI 模型能力及其表现的看法。
- **视频生成的 GPU 优化**：用户讨论了在显存（VRAM）受限的 GPU 上进行 **AI 视频生成** 的困难，并指出处理时间可能会很慢。
  - 建议的操作流程包括清理 VRAM 并使用更高效的模型，如 [CogVideoX](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)。
- **快速 AI 绘图技术**：一位成员询问了屏幕上快速更新的 **AI 绘图** 表现背后的技术，想知道其具体实现方式。
  - 回复指出，这通常依赖于强大的 GPU 和一致性模型（Consistency Models）来实现快速更新。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的音频生成增强**：一位成员展示了他们[由 AI 角色主持的播客](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255)，利用 **NotebookLM** 来编排复杂的角色对话。
  
  - 他们详细介绍了其中的多步流程，包括各种 AI 工具的集成以及 **NotebookLM** 在促进动态对话中的作用。
- **NotebookLM 中的播客创建工作流**：一位成员分享了他们使用 **NotebookLM** 进行音频生成来创建 [Spotify 上的德语播客](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde)的经验。
  
  - 他们强调了 **NotebookLM** 出色的音频功能，并寻求**自定义建议**以增强其播客制作效果。
- **音频文件的转录功能**：成员们讨论了将生成的音频文件上传到 **NotebookLM** 进行自动转录的选项。
  
  - 另外，一位成员建议利用 MS Word 的 *Dictate...Transcribe* 功能将音频转换为文本。
- **合并笔记功能评估**：成员们审议了 **NotebookLM** 中的“合并为笔记”（Combine to note）功能，评估其将多条笔记合并为单个文档的功能性。
  
  - 一位成员质疑其必要性，因为现有的合并笔记能力已经存在，并寻求对其效用的进一步说明。
- **共享笔记本功能**：一位用户询问了与同行共享笔记本的流程，并在过程中遇到了困难。
  
  - 另一位成员澄清说，在 **NotebookLM** 界面的右上角有一个“分享笔记”按钮，可以方便地进行共享。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek-R1-Lite-Preview 发布**：DeepSeek 宣布推出 [DeepSeek-R1-Lite-Preview](http://chat.deepseek.com)，展示了在 **AIME** 和 **MATH** 基准测试中增强的性能，并具有透明的推理过程。
  
  - 用户对其潜在应用感到兴奋，并指出推理能力的提升会随着长度的增加而有效扩展。
- **GPT-4o 更新增强功能**：OpenAI 发布了新的 [GPT-4o 快照](https://platform.openai.com/docs/models#gpt-4o) `gpt-4o-2024-11-20`，该版本提升了创意写作能力，并改进了文件处理以获得更深刻的洞察。
  
  - 最近的性能测试显示，GPT-4o 在多个类别中重新夺回了榜首位置，突显了显著的进步。
- **Truffles 硬件设备准备用于 LLM 托管**：**Truffles** 硬件设备被确定为一种用于在家自托管 LLM 的半透明解决方案，被幽默地称为“发光的乳房植入物”。
  
  - 这个绰号反映了围绕创新的家用 LLM 部署方案的轻松讨论。
- **Vercel 收购 Grep 以助力代码搜索**：Vercel 宣布收购 [Grep](https://grep.app/)，使开发者能够高效地搜索超过 500,000 个公共仓库。
  
  - 创始人 Dan Fox 将加入 Vercel 的 AI 团队，以增强代码搜索功能并改进开发工作流。
- **Claude 经历可用性波动**：用户报告了 **Claude** 的间歇性可用性问题，在不同实例中经历了零星的停机。
  
  - 这些可靠性问题引发了积极讨论，用户通过社交媒体平台寻求更新。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Triton 在 Softmax 上胜过 Torch**：一位成员在 RTX 3060 上对比了 [Triton 的 fused softmax](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 与 PyTorch 的原生实现，强调了 Triton 更平滑的性能表现。
  
  - 虽然 Triton 的表现通常优于 PyTorch，但在某些情况下 PyTorch 的性能与 Triton 持平甚至更优。
- **Metal GEMM 取得进展**：展示了 [Philip Turner 的 Metal GEMM 实现](https://github.com/philipturner/metal-flash-attention)，一位成员指出他们自己的实现达到了理论最大速度的 **85-90%**，与 Turner 的结果相似。
  
  - 进一步的讨论涉及了优化 Metal 编译器以及从性能关键循环中移除寻址计算（addressing computations）的挑战。
- **Dawn 的渲染性能退化**：人们对 Dawn 最新版本的性能退化表示担忧，特别是在 Chrome **130** 之后的 wgsl-to-Metal 工作流中，尽管 Chrome **131** 有所改进。
  
  - 与未定义行为（Undefined Behavior, UB）检查代码放置位置相关的问题被认为是导致落后于 Chrome **129** 的潜在原因。
- **FLUX 通过 CPU Offload 提速**：一位成员报告称，通过在 **4070Ti SUPER** 上实现逐层 CPU offload，FLUX 推理速度提升了 **200%**，推理时间从 **3.72 s/it** 降至 **1.23 s/it**。
  
  - 讨论强调了在高性能机器上使用 pinned memory 和 CUDA streams 的有效性，尽管在共享实例上性能提升有限。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DeepSeek-R1-Lite-Preview 发布**：[DeepSeek-R1-Lite-Preview](https://x.com/deepseek_ai/status/1859200141355536422) 现已上线，在 AIME 和 MATH 基准测试中表现出 **o1-preview 级别的性能**。
  
  - 它还包含实时的**透明思维链（thought process）**，并计划很快发布开源模型和 API。
- **用于写书的 AI Agents**：[Venture Twins](https://x.com/venturetwins/status/1859298925930479998) 展示了一个项目，其中十个 AI agents 协作编写一本完全自主生成的书，每个 agent 被分配不同的角色，如设定叙事和保持一致性。
  
  - 随着项目的实时开发，可以通过 [GitHub commits](https://github.com/Lesterpaintstheworld/terminal-velocity) 监控进度。
- **无需提示的 LLMs 推理**：研究表明，通过调整解码过程以考虑前 $k$ 个备选 token，**大语言模型（LLMs）** 可以在没有显式提示的情况下展现出类似于 **chain-of-thought (CoT)** 的推理路径。
  
  - 这种方法强调了 LLMs 的**内在推理能力**，表明 CoT 机制可能固有地存在于它们的 token 序列中。
- **生成式 Agent 行为模拟**：一种新架构有效地模拟了 **1,052 名真实个体**的态度和行为，生成式 agents 在综合社会调查（General Social Survey）的回答中达到了 **85% 的准确率**。
  
  - 该架构显著减少了跨**种族和意识形态群体**的准确性偏差，为社会科学中探索个人和集体行为提供了工具。
- **Soft Prompts 咨询**：一位成员询问了关于 [post](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f) 中提到的 LLMs **soft prompts** 的研究，强调了它们在将系统提示优化到嵌入空间（embedding space）方面的潜力。
  
  - 另一位成员回应称 soft prompts 的概念**非常有趣**，表明了社区内对此的一定兴趣。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **API 使用挑战**：一位成员报告在搜索 **API** 或工具时发现这两个选项都不尽如人意，表达了挫败感。
  
  - 这个问题反映了社区内对寻找高效资源的广泛兴趣。
- **模型选项澄清**：关于 **4o model** 以及它是否使用了 **o1 mini** 或 **o1 preview** 进行了讨论，确认倾向于 **o1 mini**。
  
  - 一位成员建议检查设置以验证选项，提倡动手排查问题。
- **高 Temperature 性能**：一位成员询问在 **较高 Temperature** 下性能的提升是否与其 Prompt 风格有关，暗示可能存在过多的引导规则或约束。
  
  - 这引发了对优化 Prompt 设计以增强 AI 响应能力的思考。
- **o1 的 Beta 访问权限**：一位成员对 NH 授予他们 **o1 的 Beta 访问权限** 表示兴奋和感谢，这点亮了他们的早晨。
  
  - *Woo! Thank you NH for making this morning even brighter* 反映了对新更新的兴奋之情。
- **在 Prompt 中部署分隔符**：一位成员分享了 OpenAI 关于使用三引号或 XML 标签等分隔符的建议，以帮助模型清晰地理解输入的不同部分。
  
  - 这种方法有助于更好地构建 Prompt 以改进模型响应，使输入解析更容易。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **API Key 问题阻碍访问**：多位成员报告遇到 **403 错误**，表明在尝试访问某些功能时 **API keys** 无效或使用了过时的 Endpoint。
  
  - 一位成员分享了在验证其 API keys 后仍遇到 **fetch 错误** 和使用 sandbox 功能困难的经历。
- **CORS 错误中断 API 调用**：一位 **free tier** 用户在控制台中遇到了多个 **CORS 错误**，尽管使用了标准设置且未添加额外插件。
  
  - 尝试升级到 production key 以解决这些问题的努力未能成功，凸显了 free tier 的局限性。
- **探索高级模型 Tuning 技术**：讨论深入探讨了是否可以仅使用 preamble 和可能的聊天历史来实现模型 Tuning。
  
  - 提出了关于模型对各种训练输入的适应性问题，表明需要更有效的 Tuning 方法。
- **Cohere 推出多模态 Embeddings**：一位成员称赞了新的图像 **multi-modal embeddings**，指出其在应用中有了 **显著改进**。
  
  - 然而，有人对 **每分钟 40 次请求的 rate limit** 表示担忧，这阻碍了他们的预期使用场景，导致他们寻求替代方案。
- **Harmony 项目简化问卷协调**：**Harmony** 项目旨在利用 LLM 协调问卷项目和元数据，为研究人员提供更好的数据兼容性。
  
  - 正在举办一场竞赛以增强 Harmony 的 **LLM 匹配算法**，参与者可以在 [DOXA AI](https://harmonydata.ac.uk/doxa/) 上注册并为使 Harmony 更加健壮做出贡献。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **自适应批处理优化 GPU 使用**：[adaptive batching](https://github.com/pytorch/torchtune/pull/2035) 的实现旨在通过动态调整批次大小来最大化 **GPU utilization**，从而防止训练过程中的 **OOM errors**。
  
  - 有建议将此功能作为未来 recipes 中的一个 flag 集成，理想情况下在 `packed=true` 时激活以保持效率。
- **增强 DPO Loss 结构**：关于当前 **TRL** 代码结构中是否包含近期关于 DPO 改进的论文（如 [Pull Request #2035](https://github.com/pytorch/torchtune/pull/2035) 所示）存在疑虑。
  
  - 有人请求澄清是否应移除 **SimPO** 及任何独立类，以保持 DPO recipe 的简洁直接。
- **SageAttention 加速推理**：[SageAttention](https://github.com/thu-ml/SageAttention) 相比 **FlashAttention2** 和 **xformers** 分别实现了 **2.1x** 和 **2.7x** 的加速，同时在各种模型中保持了端到端指标。
  
  - *“这里的推理增益非常酷！”* 表达了对 SageAttention 带来的性能提升的兴奋。
- **对比 sdpa 与 Naive sdpa 的基准测试**：成员建议针对提议的 **sdpa/flex** 方法与 **naive sdpa** 方法进行基准测试，以识别性能差异。
  
  - 分数中的数值误差可能会根据所使用的 **sdpa backend** 和 **data type** 而有所不同。
- **Nitro 订阅影响服务器提升（Server Boosts）**：一名成员强调，如果用户取消其 **免费 Nitro** 订阅，**server boosts** 将被移除，从而影响服务器管理。
  
  - 这强调了维持 **Nitro** 订阅以确保服务器福利不中断的重要性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 处理 Triton 集成**：一位用户询问了 **Tinygrad** 与 **Triton** 的原生集成情况，并引用了早期的讨论。**George Hotz** 指引他们查阅问题文档以获取进一步说明。
  
  - 进一步的讨论澄清了集成步骤，强调了 **Tinygrad** 与 **Triton** 之间的兼容性以提升性能。
- **SASS 汇编器寻求替代 PTXAS**：成员们讨论了 **SASS assembler** 的未来，询问其是否旨在取代 **ptxas**。**George Hotz** 建议参考问题文档了解更多细节。
  
  - 这引发了人们对 **SASS assembler** 相比 **ptxas** 可能带来的改进的兴趣，尽管关于该汇编器的长期角色仍存在一些不确定性。
- **FOSDEM AI DevRoom 征集 Tinygrad 演讲者**：一位社区成员分享了在 2025 年 2 月 2 日举行的 [FOSDEM AI DevRoom](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room) 上进行演讲的机会，强调了 **Tinygrad** 在 AI 行业中的作用。鼓励感兴趣的演讲者进行联系。
  
  - 此次演讲旨在展示 **Tinygrad** 的最新进展，并促进 AI 工程师之间的协作。
- **Tinybox 黑客松期待动手实践**：一位成员提议组织一场 FOSDEM 前的 **hackathon**，建议在现场带上一台 **Tinybox** 以提供动手体验。他们表达了在活动期间边喝比利时啤酒边与社区互动的热情。
  
  - 黑客松旨在促进 **Tinygrad** 开发者之间的实际讨论和协作项目。
- **在 Tinygrad 中探索 Int64 索引**：一位成员质疑在不涉及 **huge tensors** 的场景下使用 **int64 indexing** 的必要性，寻求了解其优势。讨论旨在澄清 **int64 indexing** 在大规模张量操作之外的使用场景。
  
  - 通过探索各种 **indexing techniques**，社区正在评估在较小张量语境下 **int64** 与 **int32** 索引对性能和效率的影响。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 同步函数中可 await 异步函数**：一位成员对在 **Mojo** 同步函数内部能够 *await 异步函数* 感到困惑，这与 Python 的限制形成对比，并寻求关于这种异步功能处理差异的澄清或解释。
- **关于 Mojo 库仓库的查询**：另一位成员对是否存在类似于 **pip** 的 **Mojo** 库仓库感到好奇，正在寻找可以访问 Mojo 库的资源或链接。
- **使用 Max 测试 Moonshine ASR 模型**：一位用户分别使用 **Max** 的 Python API 和原生 **Mojo** 版本测试了 **Moonshine ASR** 模型的性能，注意到两者都比直接使用 **onnxruntime** 的 Python 版本慢了约 **1.8 倍**。
  
  - **Mojo** 和 Python **Max** 版本转录 10 秒语音大约需要 **82ms**，而原生 **onnxruntime** 仅需 **46ms**。相关链接：[moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1) 和 [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862)。
- **因 TensorMap 导致 Mojo Model.execute 崩溃**：在分享的 **mojo** 文件顶部的注释中提供了运行 **Moonshine ASR** 模型的说明。
  
  - 用户的实践表明，将 **TensorMap** 传入 **Model.execute** 会导致崩溃，由于 **Mojo** 的限制，必须手动解包 **26 个参数**。相关链接：[moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1)。
- **寻求 Mojo 性能优化**：该用户表示这是他们最初的几个 **Mojo** 程序之一，并承认代码可能不够地道（idiomatic）。
  
  - 他们请求协助以实现更好的性能，并强调渴望提高自己的 **Mojo** 和 **Max** 技能。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **腾讯混元 (Hunyuan) 模型微调**：一位成员询问了关于微调 [腾讯混元模型](https://huggingface.co/tencent/Tencent-Hunyuan-Large) 的事宜，并分享了 [GitHub 仓库](https://github.com/Tencent/Tencent-Hunyuan-Large) 和 [官方网站](https://llm.hunyuan.tencent.com/) 的链接。
  
  - 提供的其他参考资源包括 [技术报告](https://arxiv.org/abs/2411.02265) 和 [Demo](https://huggingface.co/spaces/tencent/Hunyuan-Large)。
- **在 MI300X 上使用 Bits and Bytes**：一位成员分享了在 MI300X 系统上使用 [Bits and Bytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 的经验，强调了其易用性。
  
  - 他们强调在更新时必须使用 `--no-deps` 标志，并提供了一个强制重新安装该包的单行命令。
- **用于 LLaMA 持续预训练的 Axolotl Colab Notebook**：一位用户询问 Axolotl 是否提供任何用于 **LLaMA 持续预训练** 的 **Colab Notebook**。
  
  - Phorm 回复称搜索结果为 **undefined**，表示目前没有可用的 Notebook，并鼓励用户稍后再次检查更新。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Juan 寻求多模态挑战的帮助**：Juan 询问了在处理 **多模态问题** 时如何使用对 **视觉语言模型 (vision language models)** 的实验性支持。
  
  - 另一位成员提供了进一步的帮助，表示 *“如果有任何问题请告诉我！”*。
- **Juan 发现了 mmmu notebook**：Juan 随后自己找到了 **mmmu notebook**，这为他的项目提供了所需的持。
  
  - 他感谢社区的 *“出色工作”*，对现有资源表示赞赏。
- **将 Semantic Router 作为基准**：一位成员建议将 [Semantic Router](https://github.com/aurelio-labs/semantic-router) 作为 **分类任务** 的性能基准，强调其 **极速 AI 决策** 能力。
  
  - 该项目专注于 **多模态数据的智能处理**，它可能提供我们旨在超越的竞争性基准。
- **专注于性能提升**：有人断言需要超越现有 **分类工具** 的性能，并以 **Semantic Router** 作为参考点。
  
  - 讨论围绕确定指标和策略展开，以实现比该工具设定的基准更好的结果。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LLM-Native 简历匹配功能上线**：感谢 [@ravithejads](https://twitter.com/ravithejads)，开发了一种 **LLM-native 解决方案**用于简历匹配，增强了传统的筛选方法。
  
  - 这种创新方法解决了招聘中人工筛选**缓慢且乏味的过程**，提供了一个更高效的替代方案。
- **12 月 12 日构建 AI Agents 网络研讨会**：加入 [@Redisinc](https://twitter.com/Redisinc) 和 LlamaIndex 参加 **12 月 12 日**的研讨会，重点关注构建**数据驱动的 AI Agents**。
  
  - 该会议将涵盖架构 Agent 系统以及**降低成本**和优化**延迟**的最佳实践。
- **PDF 表格数据提取方法**：**#general** 频道的一位成员询问了从包含文本和图像的 PDF 文件中提取**表格数据**的方法。
  
  - 他们表示有兴趣了解是否存在任何可以简化此过程的现有应用程序。
- **PDF 数据提取应用**：另一位成员寻求推荐专门用于从 PDF 中提取数据的应用程序。
  
  - 这突显了社区内对能够处理各种复杂 PDF 工具的需求。

 

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **新 UI 引发褒贬不一的评价**：一些用户觉得新 UI 有点**令人应接不暇**，且注意力引导不清晰，有人将其比作《异形》（*Alien*）中的计算机。
  
  - 然而，其他人开始欣赏其受 UNIX 启发的设计，认为它适合 **1.0 版本功能**。
- **需要配置 Rate Limit**：一位用户对受到 **Anthropic** 的 Rate Limit（速率限制）表示沮丧，并指出 Interpreter 当前的错误处理会导致在超过限制时退出会话。
  
  - 他们强调了在未来更新中加入更好的 Rate Limit 管理的重要性。
- **用户呼吁 UI 增强**：有呼声要求提供信息更丰富的 UI，显示当前工具、模型和工作目录，以增强可用性。
  
  - 用户还倡导建立潜在的**插件生态系统**，以便在未来版本中实现可定制功能。
- **提议分离计算工作负载**：一位成员建议将 LLM 工作负载分配到本地和云端计算之间，以优化性能。
  
  - 这反映了对当前 Interpreter 设计局限性的担忧，该设计主要针对一次运行一个 LLM。

 

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **明天 Intel AMA 环节**：一场与 **Intel** 合作的 **Hackathon AMA** 定于**明天（11/21）太平洋时间下午 3 点**举行，为参与者提供来自 Intel 专家的直接见解。别忘了[在这里观看直播](https://www.youtube.com/watch?v=_Wm5guUXt54)并设置提醒！
  
  - 鼓励参与者准备好问题，以最大限度地从会议中获益。
- **参与者注册困惑**：一位用户报告在加入三个不同的群组并使用多个电子邮件地址注册后未收到电子邮件，这让他们对注册是否成功产生了不确定性。
- **活动类型澄清**：一位成员寻求澄清注册问题是关于 **Hackathon** 还是 **MOOC**，这突显了参与者对不同注册类型可能存在的混淆。

 

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Refact.AI 实时演示重点展示 Autonomous Agents**：**Refact.AI** 正在举办一场实时演示，展示其 **Autonomous Agent** 和 [tooling](https://github.com/smallcloudai)。
  
  - 加入 [实时演示和对话](https://discord.com/events/1089876418936180786/1300459081181429810) 以探索他们的最新进展。
- **Refact.AI 发布新 Tooling**：**Refact.AI** 团队发布了新的 [tooling](https://github.com/smallcloudai) 以支持其 **Autonomous Agent** 项目。
  
  - 鼓励参与者在实时演示活动期间体验这些工具。

 

---

**Alignment Lab AI Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

**LAION Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道（Guild）沉寂太久，请告知我们，我们将移除它。

---

# 第 2 部分：按频道细分的详细摘要和链接

{% if medium == 'web' %}

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1308644010251915346) (263 条消息🔥🔥):

> - `Hugging Face Discord Community`
> - `AI and Machine Learning Projects`
> - `Gradio and Streamlit Integration`
> - `LangChain and RAG`
> - `General Discussion and Support Requests`

- **社区参与和支持**：成员们分享了他们在 Hugging Face 社区的经验，讨论了与模型训练和技术问题（如 RuntimeErrors）相关的支持请求。
  
  - 社区提供了排错技巧并鼓励资源共享，促进了协作式的问题解决。
- **将 AI 模型集成到项目中**：用户探索了将 AI 模型集成到应用中的方法，并建议使用 Gradio，因为它比 LangChain 更简单高效。
  
  - 讨论内容包括为各种 AI 模型构建界面和工作流的实践方法，强调动手学习。
- **探索 RAG 和 AI Agents**：辩论了 Retrieval-Augmented Generation (RAG) 的概念和创建 AI Agents 的方法，并建议通过现有博客进行学习。
  
  - 成员们强调了通过项目实践来巩固理解并探索潜在创意应用的重要性。
- **新项目与协作机会**：介绍了一个名为 Open/acc 的新社区倡议，专注于 Open Science 和 Machine Learning 领域的协作。
  
  - 鼓励参与者在这个新空间分享活动和想法，以促进创新。
- **常规讨论与幽默**：经常出现关于烹饪、共同兴趣以及对 Discord 内“类邪教”社区的幽默调侃等轻松话题。
  
  - 成员们还分享了有趣的 GIF 并进行友好的打趣，营造了积极的社区氛围。

**提到的链接**：

- [O'Reilly Media - Technology and Business Training](https://www.oreilly.com)：未找到描述
- [Tweet from undefined](https://x.com/jadechoghari)：未找到描述
- [Hamster Cry GIF - Hamster Cry Tears - Discover & Share GIFs](https://tenor.com/view/hamster-cry-tears-funny-wiping-tears-gif-23475965)：点击查看 GIF
- [Simpsons Homer GIF - Simpsons Homer Bart - Discover & Share GIFs](https://tenor.com/view/simpsons-homer-bart-lisa-join-us-gif-17846376318791889140)：点击查看 GIF
- [Lemon Demon Sundial GIF - Lemon Demon Sundial View-Monster - Discover & Share GIFs](https://tenor.com/view/lemon-demon-sundial-view-monster-view-monster-viewmonster-gif-12281888211472007989)：点击查看 GIF
- [open-acc (open/ acc)](https://huggingface.co/open-acc)：未找到描述
- [Spaces - Hugging Face](https://huggingface.co/spaces?sort=trending&search=inpaint)：未找到描述
- [HeyGen - AI Video Generator](https://www.heygen.com/)：未找到描述
- [Argil AI - Get ai short videos with AI clones in 2 minutes.](https://www.argil.ai/)：使用 Argil AI 快速轻松地创建包含 AI 克隆的 AI 驱动短视频。
- [Large Language Models explained briefly](https://m.youtube.com/watch?v=LPZh9BOjkQs)：在此深入了解：https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi 技术细节演讲：https://youtu.be/KJtZARuO3JY
- [Rabbit Bunny GIF - Rabbit Bunny Toilet - Discover & Share GIFs](https://tenor.com/view/rabbit-bunny-toilet-yes-come-gif-4686108)：点击查看 GIF
- [Sunday Cult Of The Lamb GIF - Sunday Cult of the lamb Cult - Discover & Share GIFs](https://tenor.com/view/sunday-cult-of-the-lamb-cult-happy-sunday-god-gif-422811577611096801)：点击查看 GIF

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/) (1 条消息):

richieghost: 今天我在学习 LangGraph

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1308679098402476082) (10 条消息🔥):

> - `3D Printing 设计`
> - `生成式设计工具`
> - `自定义 AI 模型部署`
> - `AI 安全研究`
> - `自动化 AI 研究员`

- **3D Printing 实现复杂设计**：以令人印象深刻的 **Bugatti 刹车卡钳**为代表，许多组件展示了 **3D Printing** 实现复杂设计的能力。通过机器学习模型优化组件移除，工程师可以提升包括汽车和建筑在内的各行业性能。
  
  - *利用向量计算，* 该过程不仅提高了汽车制造的效率，还扩展到了更广泛的工程应用中。
- **生成式设计工具可免费使用**：生成式设计工具因其创意和创新能力而受到赞誉，通过 **Fusion 360** 的教育许可可免费获取。这使得学生和爱好者都能接触到先进的设计技术。
  
  - 这些工具引发的关注源于它们彻底改变设计思维和实施方案的潜力。
- **现在可以部署自定义 AI 模型**：一位成员分享了在 Hugging Face 上使用 **handler 文件**部署自定义 AI 模型的发现，这允许对模型进行量身定制的预处理和后处理。该过程涉及创建一个 `handler.py` 文件，其中指定了处理请求和响应的方法。
  
  - 这种多功能设置通过 [Hugging Face endpoints](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler) 增强了 AI 项目的定制化能力。
- **发布关于 AI 安全见解的新论文**：来自 **Redhat/IBM** 的 AI 研究人员最近发表了一篇论文，讨论了公开 AI 模型的安全影响，涉及风险和生命周期管理。论文提出了全面的策略，以增强开发者和用户的安全性。
  
  - 该论文旨在促进 AI 社区内更标准化的实践，为关于安全性和透明度的讨论做出了重大贡献。[查看论文](https://huggingface.co/papers/2411.12275)。
- **自动化 AI 研究助手起步**：有人利用本地 LLM 创建了一个**自动化 AI 研究员**，它可以根据用户查询生成研究文档。该系统利用网页抓取（web scraping）来汇编信息，并生成与主题相关的摘要和链接。
  
  - 这一创新强调了 AI 在简化研究和信息收集方面的潜力，使其触手可及。

**提到的链接**：

- [FreeAL: Towards Human-Free Active Learning in the Era of Large Language Models](https://arxiv.org/abs/2311.15614)：在各种 NLP 任务中，为模型训练收集高质量的标注数据是众所周知的耗时且耗力。虽然已有大量解决方案，例如针对小型语言模型的自动学习...
- [Paper page - Building Trust: Foundations of Security, Safety and Transparency in AI](https://huggingface.co/papers/2411.12275)：未找到描述
- [Create custom Inference Handler](https://huggingface.co/docs/inference-endpoints/main/en/guides/custom_handler#create-custom-inference-handler)：未找到描述
- [no title found](https://amfg.ai/2019/07/24/7-complex-designs-achieved-with-3d-printing/)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gvlzug/i_created_an_ai_research_assistant_that_actually/)：未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1308803911905181778) (4 条消息):

> - `分形森林生物`
> - `音乐和动画中的 AI`
> - `有效的 Prompting 技巧`
> - `结合音乐的迷幻体验`
> - `Neo 的 60 年代之旅`

- **60 年代的神经航行**：标题为 [A.I. The Matrix Red Pill Scene Psychedelic Trip](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3) 的 YouTube 视频探索了当 Neo 的红药丸将他送回 60 年代时会发生什么，将标志性音乐与 AI 动画融合在一起。
  
  - 该视频融合了 **The Beatles**、**The Doors** 和 **Jimi Hendrix** 的曲目，创造了充满活力的视听体验。
- **AI 中的 Prompting 艺术**：围绕视频 [BAD vs GOOD prompting](https://youtu.be/m3Izr0wNfQc) 展开了讨论，该视频探讨了在当今 AI 环境中有效 Prompting 技巧的必要性。
  
  - 鼓励成员发表评论，思考 Prompting 不断变化的动态及其对 AI 输出的影响。
- **对酷内容的赞赏**：一位成员对这部黑客帝国主题的 AI 视频表达了热情，说道：*“非常酷的分享，谢谢！”*
  
  - 这些反应突显了社区对融合了艺术风格的创新 AI 应用的兴趣。

**提到的链接**：

- [BAD vs GOOD prompting](https://youtu.be/m3Izr0wNfQc)：让我们在这个视频中看看现在是否仍然需要进行好的 Prompting，以及是否存在差异，差异在什么时候产生。欢迎留言...
- [A.I. The Matrix Red Pill Scene Psychedelic Trip with The Beatles, The Doors, Nirvana, Jimi Hendrix🎧🔈](https://youtu.be/JugL1okFCqI?si=zn7wpJFaQJnQcJx3)：耳机必备 #4K #ai #animation #thematrix #redpill #bluepill #johnlennon #jimmorrison #jimihendrix #kurtcobain #nirvana #thebeatles #thedoors #psych...

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1308719052771819581) (4 条消息):

> - `3080 GPU 定价`
> - `VRAM 利用率`
> - `频道讨论礼仪`

- **廉价 3090 GPU 价格触底**：成员们讨论了二手 **3090 GPU** 现在售价为 **400-500€**，被认为非常划算。
  
  - 另一位成员建议 **400-450€** 可以被视为这些显卡的一个不错成交价。
- **GPU 使用率问题被提出**：有成员对 GPU 是否被充分利用表示担忧；一位成员觉得只有 **VRAM** 在被积极使用。
  
  - 这引发了关于在任务期间实际发挥了多少性能的疑问。
- **请求将无关讨论移出**：一位成员请求将无关话题的讨论重定向到另一个频道，以保持阅读小组的专注。
  
  - 他们鼓励其他人使用相关频道进行进一步讨论，为小组活动营造更好的环境。

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1308714238583246859) (5 条消息):

> - `语义搜索挑战`
> - `Evaluate 库的问题`
> - `Pandas 的替代方案`

- **语义搜索在重点话题上遇到困难**：一位用户在使用 OpenAI 的 **Ada 002** 进行 **semantic search** 时面临挑战，其中 embeddings 优先考虑主导话题，导致不太突出但相关的句子排名较低。
  
  - 他们正在寻求 **semantic search** 的替代方案，以有效地提取所需信息。
- **对 Evaluate 库感到沮丧**：一位用户表达了对 **Evaluate Library** 的不满，称他们必须为演示手动计算 **lift** 指标，效率低下。
  
  - 他们分享了一种恼火的情绪，表示当库不能按预期工作时非常令人烦恼。
- **需要比 Pandas 更快的替代方案**：另一位用户分享了他们在处理大型数据集时使用 **Pandas** 的困扰，发现其速度较慢，并寻求更快速库的建议。
  
  - 这突显了社区内对更高效数据处理工具的持续需求。

 

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1308757287942225920) (6 条消息):

> - `Diffusers 版本问题`
> - `CogVideoX1.5-5B-I2V 仓库更新`
> - `Colab 会话崩溃`
> - `FP16 模型加载`
> - `过采样与欠采样咨询`

- **Diffusers 版本运行失败**：一名成员报告称，尝试在较新版本的 **diffusers** 上运行 **i2v** 未能成功。
  
  - 该问题可能与代码库中反映的最近更新有关。
- **CogVideoX1.5-5B-I2V 仓库需要更新**：另一名成员指出 **CogVideoX1.5-5B-I2V** 仓库**需要修正**，并强调了两小时前的一次最近提交。
  
  - 他们引用了 [CogVideoX1.5-5B-I2V 讨论区](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/discussions?status=closed) 以获取更多细节。
- **加载模型时 Colab 会话崩溃**：一名成员分享了一个 [Colab 链接](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing)，在尝试加载 Transformer 模型时会话发生了崩溃。
  
  - 他们推测崩溃可能是由于尝试加载 **fp16 模型** 导致的。
- **请求最小可复现代码片段**：一名成员建议在报告问题时应提供**最小可复现代码片段 (minimal reproducible snippet)**，以便于排查故障。
  
  - 这种方法将有助于明确用户面临的具体问题。
- **欠采样与过采样咨询**：一名成员询问在所讨论的上下文中是否可以进行**过采样 (oversampling) 或欠采样 (downsampling)**。
  
  - 这反映了人们对改进模型训练技术的持续关注。

**提到的链接**：

- [THUDM/CogVideoX1.5-5B-I2V · Discussions](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V/discussions?status=closed)：未找到描述
- [Google Colab](https://colab.research.google.com/drive/17CqYCqSwz39nZAX2YyonDxosVKUZGzcX?usp=sharing)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1308755202949513266) (175 条消息🔥🔥):

> - `DeepSeek Prover`
> - `OpenAI o1 release` (OpenAI o1 发布)
> - `GPT-4o update` (GPT-4o 更新)
> - `Model performance comparison` (模型性能对比)
> - `Community discussions on AI models` (社区关于 AI 模型的讨论)

- **DeepSeek Prover 工作讨论**：成员们对 DeepSeek 在其模型中使用 reinforcement learning（强化学习）的工作表示关注，并推测可能会发布相关论文。
  
  - 讨论中提到了模型规模和性能方面的挑战，暗示完整版本的发布可能会推迟。
- **OpenAI o1 即将发布**：关于 OpenAI 即将发布 o1 模型的猜测四起，一些成员对社区中流传的时间线传闻表示怀疑。
  
  - 讨论暗示 OpenAI 需要展示其 o1 模型，以应对行业内日益激烈的竞争。
- **GPT-4o 获得更新**：OpenAI 宣布了 GPT-4o 的更新，提升了其创意写作能力和文件处理能力。
  
  - 该模型在聊天机器人竞赛的创意写作和 coding（编程）等多个性能类别中重新回到了榜首。
- **模型性能对比**：成员们对比了包括 OpenAI 和 DeepSeek 在内的各种 AI 模型的性能，指出了创意和技术技能改进的重要性。
  
  - 讨论中反映了用户对模型的使用体验，强调了在不同任务中的优缺点。
- **社区参与和反应**：社区围绕最新的 AI 模型更新和性能指标展开了热烈讨论，经常分享一些幽默的见解。
  
  - 几位用户对 AI 发展的方向表达了同样程度的兴奋和怀疑。

**提到的链接**：

- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1859307979184689269)：来自 Chatbot Arena 的激动人心消息❤️‍🔥 在过去的一周里，最新的 @OpenAI ChatGPT-4o (20241120) 以 "anonymous-chatbot" 的身份匿名参赛，收集了 8,000 多张社区投票。结果是？...
- [来自 OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1859296125947347164?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：GPT-4o 获得了更新 🎉 该模型的创意写作能力得到了提升——更加自然、引人入胜且量身定制，以提高相关性和可读性。它在处理上传的文件方面也表现得更好...
- [来自 DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1859200149844803724)：🌟 DeepSeek-R1-Lite-Preview 的推理 Scaling Laws。更长的推理，更好的性能。DeepSeek-R1-Lite-Preview 在 AIME 上的得分随着思考长度的增加而稳步提升。
- [到 2024 年底是否会发布一个利用像 o1 这样 chain-of-thought 推理的“优秀”（fine-tuned）开源模型？](https://manifold.markets/Soli/will-a-finetuned-opensource-model-u)：90% 的可能性。该模型应该能够根据问题的复杂性决定需要思考多长时间。理想情况下，它在 LMSYS 上的排名应该高于“普通”模型，但这并不是一个 ...
- [哪家主要的 AI 实验室将率先发布像 OpenAI 的 o1 那样“在回答前思考”的模型？](https://manifold.markets/NeuralBets/which-major-ai-lab-will-be-the-firs)：OpenAI o1 博客文章称：我们正在推出 OpenAI o1，这是一个通过 reinforcement learning 训练以执行复杂推理的新型大语言模型。o1 在回答之前会进行思考——它可以产生很长的 ...
- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1859241005465432540)：这很有趣，也很有前景。“DeepSeek-R1-Lite 还使用了一个较小的 base model，无法充分发挥长思考链的潜力。”我想知道是否也有类似的...
- [未找到标题](https://mp.weixin.qq.com/s/e1YnTxZlzFvjcmrLLTA8fw?poc_token=HI7bPWejXDRRW5OqohHtuuqRtJ4F_UgfMxhXIhnk)：未找到描述
- [GitHub - deepseek-ai/DeepSeek-Prover-V1.5](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5)：通过在 GitHub 上创建一个账户来为 deepseek-ai/DeepSeek-Prover-V1.5 的开发做出贡献。
- [来自 FxTwitter / FixupX 的推文](https://x.com/search?q=stay%20in%20line%20vote&src=typed_query))：抱歉，该用户不存在 :(

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1308771688439087107) (6 messages):

> - `Francois Fleuret 提及`
> - `韩国 LLM 评估问题`
> - `日本 LLM 排行榜`

- **Francois Fleuret 的争议言论**：围绕 **Francois Fleuret** 是否在抨击 **lbdl** 展开了讨论，突显了社区中持续存在的紧张关系和观点冲突。
  
  - 一位用户对这种情况表示难以置信，称其为 *unbelievable*。
- **对 NeurIPS 审稿人标准的批评**：NeurIPS D&B 审稿人以“中文中已存在类似项目”为由驳回一个 **韩国 LLM 评估** 项目，这一做法引发了担忧。
  
  - 评论者认为每种语言都值得拥有定制化的模型，强调了 **NLP 开发** 中包容性的必要性。
- **强调日本 LLM 性能测试**：一位用户赞扬了由 **@llm_jp** 创建的 **日本 LLM 排行榜**，该榜单测试了跨多种 NLP 任务的性能。
  
  - 他们指出，**日语**书写需要多种字符集，这增加了评估工作的复杂性。

**提到的链接**：

- [Stella Biderman (@BlancheMinerva) 的推文](https://x.com/blancheminerva/status/1859271409429795083?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：NeurIPS D&B 审稿人认为“这在中文中已经存在”是驳回韩国 LLM 评估项目价值的理由，但我所看重的人们对此有更清晰的认识。Eve...
- [François Chollet (@fchollet.bsky.social)](https://bsky.app/profile/fchollet.bsky.social/post/3lbew74c7is2k)：几乎每周我都会听到有人买了一本深度学习书，作者的名字和我的很像，因为他们以为是我写的。大约有一半的读者认为...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1308652899479453736) (25 条消息🔥):

> - `o1 发布推测`
> - `LLM 中的训练动态`
> - `Reinforcement Learning 趋势`
> - `模型评估瓶颈`
> - `发布疲劳与发布后计划`

- **o1 发布推测引发关注**：有传言称 **o1 release** 可能会在明天进行，恰逢 **DevDay Singapore**，但这基于未经证实的传闻。
  
  - *一位成员指出*，*周三发布会很奇怪*，但社区仍在密切关注更新。
- **探索 LLM 训练动态**：最近的研究结果表明，模型如何 **memorize**（记忆）训练样本会影响其泛化能力，特别是如果它们在记忆之前先理解了概念。
  
  - 这突出了一种仅根据训练动态来预测模型测试准确率的方法，正如 **Katie Kang** 在讨论中所分享的那样。
- **Reinforcement Learning 的意外回归**：一位成员对 **Reinforcement Learning (RL)** 的复兴表示兴奋，尽管之前存在疑虑，现在感觉可以再次拥抱他们的 RL 背景。
  
  - *他们评论道*，“*我可以重新做一个 RL 人了*”，这反映了社区内更广泛的乐观情绪。
- **模型评估瓶颈**：人们对 **evaluation bottleneck**（评估瓶颈）提出了担忧，指出评估 MMLU 仅需几个小时，但仍可能拖慢进度。
  
  - 随后讨论了如何决定何时停止训练，尽管感到疲惫，但仍有关于坚持努力的观点。
- **发布疲劳与后续计划**：随着发布临近，评论建议在发布后需要恢复，并考虑迎接一个轻松的 12 月。
  
  - *在闲聊中*，*一位成员幽默地提到*，“*我累死了*”，这表明了发布过程对开发者的消耗。

**提到的链接**：

- [Nathan Lambert (@natolambert) 的推文](https://x.com/natolambert/status/1859255627882664034?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ)：我很高兴看到 RL 继续其宏大的接管，我曾多次对此表示怀疑。我可以重新做一个 RL 人，甚至不需要伪装成 “RLHF” 或 “NLP”……
- [Lucas Beyer (bl16) (@giffmana.bsky.social)](https://bsky.app/profile/giffmana.bsky.social/post/3lbfok33dt22g)：https://www.astralcodexten.com/p/how-did-you-do-on-the-ai-art-turing
- [Sergey Levine (@svlevine) 的推文](https://x.com/svlevine/status/1859118047304602061)：来自 @katie_kang_ 的一个有趣的新结果：在训练足够长时间后，LLM 将精确地重现训练样本（这并不奇怪）。但它们如何达到这一点的过程很重要：如果它们首先得到了正确的答案……
- [Jimmy Apples 🍎/acc (@apples_jimmy) 的推文](https://x.com/apples_jimmy/status/1859062064410751266)：显然 o1 明天发布？周三发布会很奇怪，但我猜是因为 DevDay Singapore。听小道消息说的，但我无法证实。
- [Jimmy Apples 🍎/acc (@apples_jimmy) 的推文](https://x.com/apples_jimmy/status/1859121134777843765)：这是在一个小时前发出的。Mini 和 preview API 访问权限，让我们看看是否还有更多内容。

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1308645677320114268) (176 messages🔥🔥):

> - `Vision Support`
> - `Multi-GPU Training`
> - `Internship Opportunities`
> - `Data Quality in NLP`
> - `Training Llama Models`

- **Vision Support 即将到来**：一位成员询问了 **Vision Support** 的进展，另一位成员确认该功能确实很快就会推出。
  
  - 此功能将随其他更新一起发布。
- **讨论 Multi-GPU Training**：新用户正在探索 Unsloth 的 **Multi-GPU Training** 能力，目前注意到该功能尚未上线，但预计很快推出。
  
  - 成员们分享了使用 **Llama Factory** 管理多 GPU 的策略。
- **提供实习岗位**：讨论了 Unsloth 提供的 **Internship Roles**，引发了对所需经验的好奇。
  
  - 成员们被引导至一个详细说明机会和当前需求的链接。
- **数据质量对 NLP 的重要性**：一位用户寻求关于其 **NLP task** 数据集清洗的指导，强调了其对成功的关键性。
  
  - 对话强调了数据集质量的重要性，并建议从较小的数据集开始，以便在训练期间更好地控制。
- **训练 Llama 模型**：几位成员分享了他们训练不同 **Llama Models** 的经验，发现成功程度随参数的不同而变化。
  
  - 建议包括在扩展规模之前先从 Base Models 开始，权衡数据集大小，并调整训练步数以获得最佳性能。

**提到的链接**：

- [Machine Learning Projects by huq02 using Weights & Biases](https://wandb.ai/authorize)：Weights & Biases，机器学习开发者工具。
- [Nice Smack GIF - Nice Smack Delicious - Discover & Share GIFs](https://tenor.com/view/nice-smack-delicious-meme-gif-8375212)：点击查看 GIF。
- [Llama 3.1 | Model Cards and Prompt formats](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_1/)：Llama 3.1 - 最强大的开源模型。
- [Home](https://github.com/unslothai/unsloth/wiki#-we-are-hiring)：微调 Llama 3.2, Mistral, Phi, Qwen 2.5 & Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth。
- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)：未找到描述。
- [Fine-tuning A Tiny-Llama Model with Unsloth](https://www.analyticsvidhya.com/blog/2024/02/fine-tuning-a-tiny-llama-model-with-unsloth/)：使用 Unsloth 的用户友好工具和高级功能微调您的 Tiny-Llama 模型以实现巅峰性能。
- [Google Colab](https://colab.research.google.com/drive/15OyFkGoCImV9dSsewU1wa2JuKB4-mDE_?usp=sharing,)：未找到描述。
- [unsloth/unsloth/chat_templates.py at 5078a870c04e60b2491cd4f2974cf78521961179 · unslothai/unsloth](https://github.com/unslothai/unsloth/blob/5078a870c04e60b2491cd4f2974cf78521961179/unsloth/chat_templates.py#L583))：微调 Llama 3.2, Mistral, Phi, Qwen 2.5 & Gemma LLMs，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth。

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1308881292326211656) (9 messages🔥):

> - `NixOS Installation`
> - `Fedora KDE Experience`
> - `Windows-like Linux Distritos`
> - `Checkpoint Selection in AI Training`

- **朋友们推荐 NixOS**：一位成员提到他们的朋友正敦促他们安装 **NixOS**，显示出这在用户中的流行趋势。
  
  - 这引发了群体中对各种 Linux 发行版的好奇和评论。
- **Fedora KDE 大受欢迎**：一位成员热情地推广 **Fedora KDE**，惊呼 “Fedora KDE ftw”。
  
  - 讨论包括了关于其与其他操作系统相比的优势的轻松玩笑。
- **Linux 发行版美学**：另一位成员评论说 Fedora KDE 看起来“有点像 **Windows**”，并对其外观表示兴奋。
  
  - 他们对该发行版界面的幽默看法引起了频道内其他人的共鸣。
- **AI 训练 Checkpoint 的选择困境**：一位成员询问其他人选择 AI 训练 Checkpoint 时的偏好，问道：“你们会选择哪个 Checkpoint？”
  
  - 他们分享说自己毫不犹豫地选择了训练 Checkpoint **200**，并邀请大家对不同的方法发表意见。

 

**提到的链接**：[Dsa GIF - Dsa - Discover & Share GIFs](https://tenor.com/view/dsa-gif-22912899)：点击查看 GIF。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1308881736322650202) (10 messages🔥):

> - `Fine-tuning LLMs`
> - `Model Export to Hugging Face`
> - `Pre-tokenization and Continued Pretraining`
> - `Inference with VLLM`
> - `Checkpoint Callback for Saving Models`

- **导出 Llama 3.1 供本地使用**：一位新用户成功运行了 **2x Llama 3.1 8b**，并寻求关于如何导出模型以便在本地项目中使用 Hugging Face 的指导。
  
  - 成员们建议使用 **16-bit 版本**，以在构建 RAG 应用时获得更好的性能和能力。
- **预分词数据集查询**：关于**已分词数据集（tokenized dataset）**在持续预训练（Continued Pretraining）中的兼容性引发了讨论，一位用户对此表示不确定。
  
  - 另一位成员建议，传递**未分词的数据集（untokenized dataset）**可能对训练效果更好。
- **模型加载中的警告**：一位用户报告了从 Hugging Face 加载模型时出现与无效转义序列相关的 **SyntaxWarning**，并提供了代码片段中显示的特定警告文本。
  
  - 尽管有这些警告，控制台输出确认模型和 Tokenizer 已成功加载。
- **微调的 Checkpoint 管理**：一位成员寻求关于 Fine-tuning 的建议，同时确保 Checkpoint 能保存到存储解决方案中，如 **Google Drive** 或 **Kaggle datasets**。
  
  - 另一位用户确认了使用 **callbacks** 的建议，并提到可以参考 **Weights & Biases (WandB)** 进行实验追踪。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1308701305472024626) (1 messages):

> - `SageAttention2`
> - `Quantized Attention`
> - `Inference Acceleration`

- **SageAttention2 拥有精确的 4-bit 注意力机制**：[SageAttention2 技术报告](https://arxiv.org/html/2411.10958v1)介绍了一种 **4-bit 矩阵乘法**方法，可加速 Attention 过程，与 FlashAttention2 相比实现了 **2 倍加速**。
  
  - SageAttention2 旨在优化 Attention 计算的同时保持精度，标志着在**推理加速（Inference Acceleration）**方面的重大增强。
- **SageAttention 的 GitHub 仓库**：[SageAttention GitHub 仓库](https://github.com/thu-ml/SageAttention/tree/main)声称在不损失端到端指标的情况下，分别比 FlashAttention2 和 xformers 提升了 **2.1 倍和 2.7 倍的速度**。
  
  - 该实现表明 SageAttention2 是 **FlashAttention2** 的即插即用替代方案，针对 **RTX40/3090** 硬件进行了优化，但目前仅用于推理。
- **SageAttention2 的局限性**：值得注意的是，SageAttention2 仅支持**推理**而不支持训练，这明确了其预期的使用场景。
  
  - 开发重点在于优化性能特性，同时确保与现有模型的兼容性。

**提到的链接**：

- [SageAttention2 Technical Report: Accurate 4 Bit Attention for Plug-and-play Inference Acceleration](https://arxiv.org/html/2411.10958v1)：未找到描述
- [GitHub - thu-ml/SageAttention: Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models.](https://github.com/thu-ml/SageAttention/tree/main)：量化注意力机制，在不损失各种模型端到端指标的情况下，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速。—— thu-ml/SageAttention

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1308649606825119784) (148 messages🔥🔥):

> - `Aider 安装挑战`
> - `DeepSeek 性能表现`
> - `OpenRouter 相关担忧`
> - `模型量化影响`
> - `编程工具对比`

- **Aider 安装挑战已解决**：用户在 Aider 的安装过程中遇到了困难，特别是 API keys 和环境变量的配置，导致一些人考虑通过重新安装组件来解决问题。
  
  - 经过排查，一位用户确认强制重新安装有助于成功启动并运行 Aider。
- **DeepSeek 性能令人印象深刻**：DeepSeek-R1-Lite-Preview 在 AIME 和 MATH 基准测试中表现出与 o1-preview 相当的性能，且响应速度比之前的模型更快。
  
  - 该模型的透明推理过程允许用户实时查看其思考过程，提升了其在编程任务中的有效性。
- **对 OpenRouter 模型质量的担忧**：用户对 OpenRouter 使用开源模型的量化版本表示不满，这引发了对其在 Aider Leaderboard 上表现的质疑。
  
  - 有人呼吁在 Aider Leaderboard 上发布更清晰的警告，说明使用 OpenRouter 量化模型时可能出现的性能差异。
- **模型量化影响的讨论**：量化对模型性能的负面影响引起了用户的关注，许多人更倾向于使用原始模型而非量化版本。
  
  - 用户建议 OpenRouter 应公开具体的模型版本，以准确反映性能预期。
- **编程工具的对比使用**：用户对比了 Aider、Cursor 和 Sonnet 等多种编程工具，分享了它们在文件创建和编辑任务中的有效性见解。
  
  - 许多参与者指出，他们发现 Aider 在编辑方面特别有优势，而像 Cline 这样的替代方案对于大规模使用来说成本太高。

**提到的链接**：

- [Paul Gauthier (@paulgauthier) 的推文](https://x.com/paulgauthier/status/1859320459634016553?s=46&t=AkDCTtZVFFazuKDknG6fLA)：新的 gpt-4o-2024-11-20 在 Aider 的代码编辑基准测试中得分与 08-06 版本相同，落后于 05-13 版本。这可能是 OpenAI 家族模型中第一次更新后没有……
- [文件编辑问题](https://aider.chat/docs/troubleshooting/edit-errors.html)：Aider 是你终端里的 AI 配对编程工具
- [主页](https://aider.chat/)：Aider 是你终端里的 AI 配对编程工具
- [DeepSeek](https://chat.deepseek.com/)：与 DeepSeek AI 聊天。
- [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards/)：LLM 代码编辑能力的量化基准测试。
- [Models | OpenRouter](https://openrouter.ai/models)：在 OpenRouter 上浏览模型
- [Qwen2.5 Coder 32B Instruct - API, Providers, Stats](https://openrouter.ai/qwen/qwen-2.5-coder-32b-instruct)：Qwen2.5-Coder 是最新的代码专用 Qwen 大语言模型系列（原名 CodeQwen）。通过 API 运行 Qwen2.5 Coder 32B Instruct
- [DeepSeek V2.5 - API, Providers, Stats](https://openrouter.ai/deepseek/deepseek-chat)：DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5
- [Andrew Curran (@AndrewCurran_) 的推文](https://x.com/andrewcurran_/status/1859235248632123763?s=46&t=LoeRx5EgmzbDflKGl42Euw)：在 o1-preview 发布两个月后，其 Chain-of-Thought 推理已被复现。DeepSeek 表示官方版本的 DeepSeek-R1 将完全……
- [OpenAI (@OpenAI) 的推文](https://x.com/OpenAI/status/1859296125947347164)：GPT-4o 更新了 🎉 模型的创意写作能力得到了提升——更自然、更具吸引力，且量身定制的写作提高了相关性和可读性。它在处理上传文件方面也表现更好……
- [🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！ | DeepSeek API 文档](https://api-docs.deepseek.com/news/news1120)：🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级性能。
- [Meta: Llama 3.1 70B Instruct – Provider Status](https://openrouter.ai/meta-llama/llama-3.1-70b-instruct/providers)：查看提供商状态并向 Meta: Llama 3.1 70B Instruct 发起负载均衡请求 - Meta 最新的 Llama 3.1 系列模型推出了多种尺寸和版本。这个 70B instruct-t...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1308719778260717598) (29 messages🔥):

> - `Aider usage challenges` (Aider 使用挑战)
> - `Chat modes best practices` (聊天模式最佳实践)
> - `Token limit concerns` (Token 限制担忧)
> - `Language support in Aider` (Aider 中的语言支持)
> - `Context extension mechanisms` (上下文扩展机制)

- **理解 Aider 的聊天模式**：成员们讨论了各种 Aider 聊天模式的效果，强调使用 **o1-preview** 作为 Architect（架构师）配合 **DeepSeek** 或 **o1-mini** 作为 Editor（编辑器）能提供最佳效果。
  
  - 一位用户指出 **Sonnet** 在处理日常任务时表现异常出色，无需复杂配置。
- **Aider 的 Token 限制消耗**：用户表达了对生成代码时快速消耗 Token 的担忧，并建议引入缓存机制以有效存储上下文。
  
  - 一位成员获知，修改 `/read` 文件列表会破坏缓存，但修改 `/add` 文件则不会。
- **意外的语言响应问题**：一位用户报告在使用 **o1-mini** 时收到了西班牙语回复，尽管其约定文件（conventions file）中明确规定使用英语。
  
  - 提供的解决方案是使用 `--language` 选项显式指定语言。
- **自定义上下文扩展功能请求**：一位用户询问如何向 Aider 添加个人自动上下文扩展机制，寻求为自定义代码创建扩展点。
  
  - 然而，官方澄清目前版本的 Aider 尚无法实现此类功能集成。
- **在 Aider 中运行脚本**：关于使用 Aider 自动化清单任务的讨论建议使用 `--message` 参数或 `aider -m` 模式进行脚本编写。
  
  - 讨论中提供了如何使用 Shell 脚本运行循环并将指令应用于多个文件的示例。

**提到的链接**：

- [Separating code reasoning and editing](https://aider.chat/2024/09/26/architect.html)：Architect 模型负责描述如何解决编程问题，而 Editor 模型负责将其转化为文件编辑。这种 Architect/Editor 方法产生了 SOTA 基准测试结果。
- [Scripting aider](https://aider.chat/docs/scripting.html)：你可以通过命令行或 Python 对 Aider 进行脚本化操作。
- [Chat modes](https://aider.chat/docs/usage/modes.html)：使用 chat、ask 和 help 聊天模式。
- [Supported languages](https://aider.chat/docs/languages.html#how-to-add-support-for-another-language)：Aider 几乎支持所有流行的编程语言。

---

### **Eleuther ▷ #**[**announcements**](https://discord.com/channels/729741769192767510/794042109048651818/1308867389319942154) (1 messages):

> - `Linear vs Affine Representation`
> - `ACE Method for Control in Language Models`
> - `Refusal Behavior in Language Models`

- **澄清线性表示假设 (Linear Representation Hypothesis)**：论文强调了定义**线性表示**时的歧义：它质疑应该将其视为保持原点的*线性*函数，还是不保持原点的*仿射 (affine)*函数。
  
  - 这一区别非常重要，因为之前的发现（特别是 **Arditi et al.** 的研究）严重依赖于这种解释，导致在 **RWKV** 等模型中产生了误导性的结果。
- **引入用于仿射控制的 ACE**：作者提出了新的 **ACE (Affine Concept Editing)** 方法，将概念视为仿射函数，以增强对模型响应的控制。
  
  - ACE 允许将激活值投影到超平面上，通过在 **Gemma** 上的测试证明，该方法在管理模型行为方面具有更高的精度。
- **对拒绝行为的可靠控制**：ACE 在包括 **Llama 3 70B** 在内的十个模型上进行了测试，在处理有害和无害提示词时，它实现了对拒绝行为更好的控制。
  
  - 该方法改进了以往的技术，表明这是一种更可靠的引导模型行为的策略。
- **研究贡献邀请**：为了继续改进 ACE 方法，作者邀请感兴趣的人士在特定的研究频道中进行自我介绍。
  
  - 作者对贡献者的努力表示感谢，并强调了社区协作在推动这项研究中的重要性。
- **访问研究材料**：该项目的 GitHub 仓库可以在 [steering-llama3](https://github.com/EleutherAI/steering-llama3) 找到，相关论文已发表在 [arXiv](https://arxiv.org/abs/2411.09003) 上。
  
  - 更多见解和讨论可以通过作者的 [Twitter thread](https://x.com/norabelrose/status/1859307287112007896) 关注。

**提及的链接**：

- [GitHub - EleutherAI/steering-llama3](https://github.com/EleutherAI/steering-llama3)：通过在 GitHub 上创建账户，为 EleutherAI/steering-llama3 的开发做出贡献。
- [Refusal in LLMs is an Affine Function](https://arxiv.org/abs/2411.09003)：我们提出了仿射概念编辑 (ACE) 作为一种通过直接干预激活值来引导语言模型行为的方法。我们首先从模型激活向量的仿射分解开始……
- [Tweet from Nora Belrose (@norabelrose)](https://x.com/norabelrose/status/1859307287112007896)：在这篇论文中，我们指出了先前关于线性表示假设研究中的一个歧义：线性表示是一个线性函数——即保持原点的函数——还是一个仿射函数……

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1308771416170037249) (20 messages🔥):

> - `GPGPU Performance`
> - `PyTorch Optimization Techniques`
> - `Data Loading Strategies`
> - `GPU Memory Management`

- **GPGPU 的变革时代**：在 GPGPU 的早期，在 **G200** 上编写代码比高性能 CPU 快 **30倍**，这导致了它在超级计算机中的广泛采用。
  
  - Nvidia 凭借其消费级产业根基实现的扩展能力，使其 GPU 成为高性能计算的**明智选择**。
- **PyTorch 性能优化捷径**：为了提升 **PyTorch 性能**，将数据存储在 GPU 内存中比使用数据加载器 (data loader) 能显著加快处理速度。
  
  - 社区分享了使用 **HLB-CIFAR10** 或 **Airbench** 的建议，以在卷积网络中获得更好的性能。
- **切换数据格式以提升速度**：将数据保持为 **UINT8** 格式直到最后一刻，可以使**内存带宽需求减半**，从而提高传输速度。
  
  - 在使用前才在 GPU 上转换数据，可以确保高效的内存传输和处理。
- **平衡 CPU 和 GPU 数据加载**：在使用 CPU 和 GPU 数据加载流水线时，避免被 **CPU worker 变成瓶颈**至关重要。
  
  - 确保 CPU 效率可以支持更顺畅、更快速地向 GPU 传输数据，从而提升整体模型训练性能。
- **参考优化实践**：分享了关于优化训练策略的细节，包括指向 [David Page 的技巧包 (bag of tricks)](https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb) 等资源的链接。
  
  - 社区贡献者指出了过去的努力和已确立的优化深度学习模型训练的实践。

**提及的链接**：[cifar10-fast/bag_of_tricks.ipynb at master · davidcpage/cifar10-fast](https://github.com/davidcpage/cifar10-fast/blob/master/bag_of_tricks.ipynb)：通过在 GitHub 上创建账户，为 davidcpage/cifar10-fast 的开发做出贡献。

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1308805502054039592) (125 条消息🔥🔥):

> - `Latent Actions and Inverse Dynamics Models`
> - `nGPT Baseline Bugs`
> - `Use of Position Embeddings`
> - `Document Masking Impact on Training`
> - `Forgetting Transformer`

- **Latent Actions 论文探讨**：一位用户询问了关于 **latent actions** 和 **inverse dynamics models** 的最佳论文，暗示对该领域的最新（SOTA）成果感兴趣。
  
  - 虽然没有提供具体的论文，但对话暗示了这些领域相关文献的重要性。
- **nGPT 的 Baseline 评估复杂化**：成员们讨论了 nGPT 已发布代码和 Baseline 评估中的不一致性，特别是影响对比指标的 Bug。
  
  - 有人指出，内部和外部代码的差异使得有效的评估几乎不可能，从而导致对结果的怀疑。
- **Position Embedding 创新**：讨论围绕 Position Embeddings 的新方法展开，特别是涉及 Attention 机制中累积宽度计算的方法。
  
  - 提到了 **Forgetting Transformer** 和新的 **Contextual Position Encoding (CoPE)** 等相关论文，因为它们解决了传统 Position Embeddings 中的问题。
- **Document Masking 对模型性能的影响**：小组辩论了 Document Masking 技术的影响，发现它能在不显著损害训练性能的情况下减少 Token 需求。
  
  - 同时也引发了对评估公平性的担忧，因为数据传输方式的改变（如文档边界带来的偏差）可能会产生潜在优势。
- **关于有效定位策略的问题**：提出了使用 Position Embeddings 解决 Attention 问题的不同策略，包括简单方法是否可能优于复杂映射。
  
  - 成员们分析了像 **ALiBi** 或 **RoPE** 这样的方法是否比近期研究中提出的替代方案能更好地集成。

**提到的链接**：

- [Contextual Position Encoding: Learning to Count What's Important](https://arxiv.org/abs/2405.18719)：Attention 机制是 Large Language Models (LLMs) 的核心组件，允许序列中的 Token 相互交互，但它是顺序无关的。引入 Position Encoding (P...
- [YouJiacheng (@YouJiacheng) 的推文](https://x.com/YouJiacheng/status/1859353724713566290)：@hi_tysam 这是一个滑动窗口，如果层数 >1，信息仍然可以传播。
- [Forgetting Transformer: Softmax Attention with a Forget Gate](https://openreview.net/forum?id=q2Lnyegkr8)：现代循环序列模型的一个重要组成部分是遗忘门（forget gate）。虽然 Transformer 没有显式的循环形式，但我们展示了可以自然地融入遗忘门...
- [koszarskyb - 概览](https://github.com/koszarskyb)：GitHub 是 koszarskyb 构建软件的地方。
- [GPT baseline block computation error · Issue #1 · NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/issues/1#issuecomment-2484596258)：你好，非常感谢开源 nGPT。我在 GPT (use_nGPT=0) baseline 的 block 计算中发现了一个错误。正在进行的计算是：x = norm(x) + attn(norm(x)) x = n...
- [modded-nanogpt/records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt at master · KellerJordan/modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/111924_FlexAttention/8384493d-dba9-4991-b16b-8696953f5e6d.txt)：在 7.8 分钟内使用 8xH100 达到 NanoGPT (124M) 质量。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 的开发做出贡献。

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1308914897366552656) (2 messages):

> - `Scaling Laws in Language Models`
> - `Evaluation Science Advocacy`
> - `Marius Hobbhahn`

- **新的 Scaling Laws 论文出现**：最近发表的一篇题为 *Understanding How Language Model Performance Varies with Scale* 的论文可以在[这里](https://arxiv.org/abs/2405.10938)查阅，该论文详细介绍了一种基于约 100 个公开可用模型的 Scaling Laws 观察方法。
  
  - 作者提出，Language Model 的性能更多地是**低维能力空间**的函数，而不仅仅是在多个规模上进行训练的结果。
- **Marius Hobbhahn 倡导评估科学**：一位成员指出，**Apollo 的 Marius Hobbhahn** 是 AI 社区中推动评估方法科学化最突出的倡导者之一。
  
  - 这似乎凸显了人们对在 AI 模型开发中加强严谨评估实践的兴趣日益浓厚。

 

**提到的链接**：[Observational Scaling Laws and the Predictability of Language Model Performance](https://arxiv.org/abs/2405.10938)：理解 Language Model 性能如何随规模变化对于基准测试和算法开发至关重要。Scaling Laws 是建立这种理解的一种方法，但其要求...

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1308927065319538769) (15 messages🔥):

> - `Zero-shot benchmarking for pruned models`
> - `WANDA pruning method`
> - `lm_eval library compatibility`
> - `Model evaluation on ADVBench`
> - `vllm model usage`

- **关于剪枝模型 Zero-shot 基准测试的问题**：一位成员询问 **lm-eval** 是否支持剪枝模型的 Zero-shot 基准测试，并提到他们正在使用一种名为 **WANDA** 的方法。
  
  - 他们对从 Zero-shot 评估中获得的**可疑结果**表示了一些担忧。
- **lm_eval 与剪枝模型的兼容性**：一位用户讨论了为适配剪枝模型而对 **lm_eval** 进行的修改，并指出他们拥有的版本非常旧。
  
  - 他们询问**当前版本**是否同时支持量化模型和剪枝模型。
- **使用 vllm 在 ADVBench 上进行评估**：对话透露一位成员正使用 **vllm** 在 **ADVBench** 上评估他们的模型，并分享了他们运行推理的方法。
  
  - 他们提供了用于生成输出的代码行：`vllm_outputs = model.generate(dialogs, sampling_params)`。
- **加载剪枝模型**：分享的加载模型的方法为 `from vllm import LLM; vllm_model = LLM(hf_model_path, tokenizer, dtype = 'bfloat16', swap_space = 128)`。
  
  - **hf_model_path** 表示剪枝模型的路径，而 **swap_space** 的作用被明确为稍后讨论的重点。
- **积极排查模型推理问题**：一位成员正积极寻求澄清他们在推理中使用剪枝模型的情况，并询问关于 **swap_space** 参数的问题。
  
  - 他们提到稍后会重新审视这个问题，以进一步了解他们的疑虑。

 

**提到的链接**：[GitHub - locuslab/wanda: A simple and effective LLM pruning approach.](https://github.com/locuslab/wanda?tab=readme-ov-file#zero-shot-evaluation)：一种简单且有效的 LLM 剪枝方法。通过在 GitHub 上创建账号为 locuslab/wanda 的开发做出贡献。

 

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1308668094805311499) (132 条消息🔥🔥):

> - `Perplexity vs. ChatGPT`
> - `推荐码使用`
> - `Perplexity 购物功能`
> - `API 功能性`
> - `iOS 上的图像生成`

- **用户对比 Perplexity 与 ChatGPT**：用户分享了使用 Perplexity 与 ChatGPT 相比的体验，指出 **ChatGPT** 通常因其多功能性和卓越的对话能力而更受青睐。
  
  - 一些用户尽管是 Perplexity 的 **Pro** 用户，但对其感知的局限性表示沮丧，并指出使用 ChatGPT 的频率更高。
- **推荐码说明**：有用户询问在使用第一个推荐码后，是否可以在同一账户上使用多个推荐码。得到的澄清是推荐码仅适用于新订阅者。
  
  - 这意味着一旦用户获得过 **Pro** 状态，就无法再使用另一个推荐码来获取折扣。
- **关于购物功能的讨论**：针对新的 **Perplexity Shopping** 功能出现了一些疑问，特别是它是否为美国市场专属。
  
  - 用户表达了兴趣，并寻求关于购物功能潜在访问限制的澄清。
- **对 API 功能的担忧**：几位用户报告了即使切换模型，**API** 响应也保持不变的问题，导致了困惑和沮丧。
  
  - 这引发了关于该平台在灵活性和响应多样性方面感知缺陷的讨论。
- **iOS 上受限的图像生成**：一位用户询问是否可以在 **iOS app** 上创建图像，结果显示此功能仅限于 iPad 用户。
  
  - 这一限制引发了关于该应用在不同设备上功能的进一步讨论。

**提到的链接**：

- [Supported Models - Perplexity](https://docs.perplexity.ai/guides/model-cards)：未找到描述
- [Perplexity Supply](https://perplexity.supply/)：好奇心与品质的碰撞。我们的高端系列为好奇者提供精心设计的服饰。从重磅棉质基础款到刺绣单品，每一件都体现了我们的……
- [无标题](https://docs.perplexity.ai/guides/getting-started)：未找到描述
- [Crypto Meets Real Estate: South Africans Can Now Buy Property With Bitcoin](https://techfinancials.co.za/2024/11/18/crypto-meets-real-estate-south-africans-can-now-buy-property-with-bitcoin/)：南非房地产行业首创，买家现在可以通过安全且完全合规的交易，使用加密货币购买房地产。
- [How Many Stars Are in the Universe](https://cosmicvoage.com/how-many-stars-are-in-the-universe/)：宇宙中有多少颗恒星。宇宙浩瀚而壮丽，人类正是在这种美丽中继续推测。
- [The Milky Way's 100 Billion Planets - NASA](https://www.nasa.gov/image-article/milky-ways-100-billion-planets/)：这张艺术家的插图展示了银河系恒星周围行星的普遍程度。与实际相比，行星、它们的轨道及其宿主恒星都被大大放大了……
- [How Many Stars in the Milky Way ?](https://www.youtube.com/watch?v=Fpgfd6FHQxg)：有没有想过在浩瀚的银河系中有多少颗星星在闪烁？🌌 在这段迷人的 120 秒动画之旅中，我们深入探索了迷人的……

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1308646352560980049) (9 条消息🔥):

> - `Web App Fullstack with Next.js`
> - `Chicken or Egg Paradox Solved`
> - `Michelin Star Cities`
> - `NVIDIA Chips Overheat`
> - `Stock Monitoring for Qubit`

- **Web App 全栈开发示例**：分享了一个讨论 [全栈 Next.js 开发](https://www.perplexity.ai/search/webapp-fullstack-nextjs-hono-b-W1pCGRCUSJmPBFklgsg.7w) 的资源，提供了对现代 Web 应用程序框架的见解。
  
  - *探索使用 Hono 进行服务端路由！*
- **先有鸡还是先有蛋悖论的最终定论**：古老的 **先有鸡还是先有蛋悖论** 已得到解决，详细见解请参见 [此处](https://www.perplexity.ai/page/chicken-or-egg-paradox-solved-i_BYFB5DQ6W8XeXOr_4eXQ)。
  
  - *深入探讨进化生物学，本文阐明了起源！*
- **拥有最多米其林星级的城市揭晓**：关于哪个城市拥有最多 **米其林星级** 的讨论可以在 [此处](https://www.perplexity.ai/search/city-with-the-most-michelin-st-NgnS7MxURLOQO4cReb.j6A) 找到。
  
  - *回顾全球知名城市的烹饪排名！*
- **NVIDIA 芯片遭遇过热问题**：本报告 [此处](https://www.perplexity.ai/page/nvidia-ai-chips-overheat-SRXQJH9yQ8ebTG_KeAT46A) 提出了对 **NVIDIA AI 芯片过热** 的担忧。
  
  - *讨论强调了与长时间使用相关的风险！*
- **持续追踪 Qubit 股票**：分享了监控 **Qubit 股票** 的行动呼吁以及 [此处可见](https://www.perplexity.ai/page/qubit-stock-N9_yIkN5RbGoYzs2L___Lg) 的见解。
  
  - *建议投资者保持警惕！*

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1308808204880318615) (1 条消息):

> - `Perplexity API`
> - `Domain Filtering`

- **Perplexity API 的过滤问题**：一位用户强调了在使用 **Perplexity API** 通过黑名单过滤特定域名来筛选搜索结果时遇到的困难。
  
  - 他们表达了挫败感，因为本应排除的域名继续出现在结果中，并询问是否遗漏了某些格式要求。
- **排查过滤域名结果的问题**：讨论集中在 **Perplexity API** 中 **域名过滤** 的有效性。
  
  - 寻求关于可能影响黑名单域名可见性的格式设置细节的澄清。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1308661803219423253) (120 条消息🔥🔥):

> - `Gemini 1114 性能`
> - `DeepSeek 更新`
> - `Prompt caching`
> - `GPT-4o 模型问题`
> - `RP 模型对比`

- **Gemini 1114 在输入处理方面表现不佳**：用户反馈 **Gemini 1114** 在对话中经常忽略图像输入，导致产生幻觉响应，而 **Grok vision Beta** 等模型则没有这类问题。
  
  - 成员们希望得到确认和修复，对该模型反复出现的问题表示沮丧。
- **DeepSeek 发布新推理模型**：新模型 **DeepSeek-R1-Lite-Preview** 已发布，宣称具有增强的推理能力，并在 AIME 和 MATH 基准测试中表现出色。
  
  - 然而，一些用户注意到该模型的运行速度较慢，引发了关于 **DeepInfra** 是否是更快替代方案的讨论。
- **关于 Prompt caching 的说明**：**DeepSeek** 等特定模型已支持 Prompt caching，用户对其他提供商的缓存策略提出了疑问。
  
  - 一些成员讨论了不同系统之间缓存工作方式的差异，特别提到了 **Anthropic** 和 **OpenAI** 的协议。
- **GPT-4o 模型描述问题**：用户发现新发布的 **GPT-4o** 存在差异，指出该模型错误地列出了 **8k context**，并且描述信息错误地关联到了 **GPT-4**。
  
  - 在指出错误后，成员们看到模型卡片（model card）得到了快速更新和修复，恢复了准确信息。
- **RP 模型对比**：成员们讨论了用于故事创作和 Role-playing (RP) 的 **Claude** 替代方案，由于 **Hermes** 的质量和性价比，有人建议使用它。
  
  - 用户对这些模型的体验各不相同，有些人觉得 **Hermes** 更好，而另一些人则继续忠于 **Claude**。

**提到的链接**：

- [DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1859200141355536422?s=46&t=2a7uDiV3mox9o-E5jIFbLQ)：🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级别表现。💡 实时透明的思考过程。🛠️ 开源...
- [Yi Large - API, Providers, Stats](https://openrouter.ai/01-ai/yi-large)：Yi Large 模型由 01.AI 设计，针对以下场景：知识搜索、数据分类、类人聊天机器人和客户服务。通过 API 运行 Yi Large。
- [anthropic-cookbook/misc/prompt_caching.ipynb at main · anthropics/anthropic-cookbook](https://github.com/anthropics/anthropic-cookbook/blob/main/misc/prompt_caching.ipynb)：展示使用 Claude 的一些有趣且有效方法的 Notebooks/Recipes 集合。
- [GPT-4o (2024-11-20) - API, Providers, Stats](https://openrouter.ai/openai/gpt-4o-2024-11-20)：2024-11-20 版本的 GPT-4o 提供了更高级的创意写作能力，写作风格更自然、更具吸引力且更具针对性，提升了相关性和可读性。它还擅长处理...
- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching#deepseek)：优化 LLM 成本高达 90%。
- [Provider Routing | OpenRouter](https://openrouter.ai/docs/provider-routing)：在多个提供商之间路由请求。

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1308662164961103902) (6 条消息):

> - `自定义 Provider keys`
> - `Key 集成访问权限`
> - `Anthropic Claude 3.5 Sonnet`
> - `x-ai/grok-beta`
> - `xai`

- **多人请求自定义 Provider keys**：几位成员表达了申请 **自定义 Provider key** 的愿望，包括针对 **x-ai/grok-beta** 和 **Anthropic Claude 3.5 Sonnet** 的 key。
  
  - *一位用户提到，他们已经拥有一个带有余额的账户*，如果能与 OpenRouter 配合使用将非常有益。
- **询问 Key 集成访问权限**：一位成员询问了获取 **Key 集成访问权限** 的流程，并表达了测试该功能的积极性。
  
  - 这表明用户对探索可用功能和工具持续保持兴趣。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1308690107003437068) (58 条消息🔥🔥):

> - `Model Loading Issues` (模型加载问题)
> - `System Requirements for Models` (模型的系统要求)
> - `Optimizing Performance with Limited Hardware` (在有限硬件下优化性能)
> - `Exploring Cloud-Based Solutions` (探索云端解决方案)
> - `Model Recommendations and Preferences` (模型推荐与偏好)

- **MacBook 上的模型加载挑战**：一位用户在 LM Studio 中加载本地模型时遇到困难，原因是系统资源不足，错误提示指出其 **36GB RAM M3 MacBook** 存在过载风险。
  
  - 另一位成员建议 **32B 模型** 对于该配置来说太大了，建议最高使用 **14B**。
- **了解模型系统要求**：讨论显示，可以通过在模型文件大小的基础上增加 **10%** 来估算所需的 RAM，不过为了获得更好的性能，建议选择更小的模型。
  
  - 有人指出，较大的模型会占用大量可用内存，从而降低机器处理其他任务的功能。
- **探索性能优化**：为了提高 **1050 Ti GPU** 的性能，建议包括使用更小的模型尺寸、减小 context size 以及确保高效的代码实践。
  
  - 一位用户提到，当本地硬件不足时，租用云端硬件可能是一种具有成本效益的解决方案。
- **云端模型使用**：一位成员分享了他们转向租用云服务器来使用模型的经历，发现与购买硬件相比，这在经济上更有利，成本在 **每月 25 到 50 美元** 之间。
  
  - 这种方法允许在不受本地硬件限制的情况下访问高速模型。
- **模型推荐与用户偏好**：用户推荐了各种模型，包括 **Hermes 3**、**Lexi Uncensored V2** 和 **Goliath 120B**，并根据性能和写作质量指出了个人偏好。
  
  - 建议尝试不同的模型以找到最适合个人用例的选择，并鼓励在有新选项可用时进行尝试。

**提到的链接**：

- [NousResearch/Hermes-3-Llama-3.1-70B · Hugging Face](https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-70B)：未找到描述
- [Qwen/Qwen2.5-Coder-32B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)：未找到描述

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1308675045199183912) (64 条消息🔥🔥):

> - `VM Performance with Qwen Models` (Qwen 模型在 VM 中的性能)
> - `Hardware Requirements for DeepSeek v2.5 Lite` (DeepSeek v2.5 Lite 的硬件要求)
> - `Workstation Design for LLMs` (LLM 工作站设计)
> - `GPU Selection for AI Workloads` (AI 工作负载的 GPU 选择)
> - `Fine-tuning vs. Running Models` (微调 vs. 运行模型)

- **无 GPU 的 VM 在运行 Qwen 模型时表现吃力**：一位成员报告称，在没有 GPU 的虚拟机上运行 Qwen 2.5 模型导致性能严重受限，仅能达到约 **1 token/秒**。
  
  - 另一位用户澄清说，仅靠 CPU 的推理速度可能非常缓慢，而 GPU 将显著改善这种情况。
- **DeepSeek v2.5 Lite 的 RAM 和 GPU 要求**：对于运行 **DeepSeek v2.5 Lite**，建议 Q4_K_M 变体至少需要 **24GB VRAM**；完整的 Q8 则需要约 **48GB VRAM**。
  
  - 成员们讨论认为 NVIDIA 显卡是首选，因为 AMD 的驱动不稳定会影响性能。
- **LLM 工作站构建指南**：一位用户正寻求关于构建用于微调 LLM 的工作站的建议，预算为 **30,000 到 40,000 美元**，并正在权衡选择多块 NVIDIA **A6000s** 还是较少的高端选项（如 **H100**）。
  
  - 讨论强调了显存的重要性，以及在预算限制下使用二手硬件的灵活性。
- **优化性能的 GPU 选择**：有人指出，使用 **多块 24GB 3090** 可以作为昂贵新机型的可行替代方案，尽管缺乏 NVLink 性能。
  
  - 一位成员分享了一个资源，对比了各种 GPU 在 LLM 推理中的 token/秒性能基准测试。
- **理解微调与运行模型的区别**：与运行模型相比，微调模型消耗的资源要多得多，需要更高的内存和计算能力。
  
  - 成员们思考了专用 AI 芯片作为解决运行大模型相关硬件挑战的潜在方案。

 

**提到的链接**：[GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：多块 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1308671878063460472) (102 条消息🔥🔥):

> - `游戏 PC 推荐`
> - `一致性角色创建`
> - `用于 Substance Designer 的 AI 模型`
> - `用于视频生成的 GPU 利用率`
> - `绘画 AI 演示`

- **组装游戏 PC 的建议**：一位用户正在寻求预算在 **$2500** 以内的游戏 PC 推荐，询问关于组件选择和购买渠道的建议。
  
  - 他们鼓励其他人发送私信以获取个性化建议。
- **绘本角色一致性的挑战**：一位成员询问如何在整本绘本中保持一致的角色设计，因为在生成多张图像时经常遇到差异问题。
  
  - 建议包括使用 **FLUX** 或图像转换技术来提高一致性。
- **纹理创建 AI 模型的探索**：讨论了 AI 模型是否可以有效替代 **Substance Designer**，并强调了在该领域进一步探索的必要性。
  
  - 成员们分享了他们对不同 AI 模型能力及其性能的看法。
- **优化 AI 视频生成的 GPU 使用**：用户讨论了在显存（VRAM）有限的 GPU 上进行 AI 视频生成的困难，并指出处理速度可能较慢。
  
  - 建议的操作包括清理 VRAM 以及使用更高效的模型，如 **CogVideoX**。
- **了解快速 AI 绘画技术**：一位成员询问了屏幕上快速更新的 AI 绘画表现背后的技术，想知道其具体实现方式。
  
  - 回复指出，这通常依赖于强大的 GPU 和一致性模型（consistency models）来实现快速更新。

**提到的链接**：

- [THUDM/CogVideoX1.5-5B-I2V · Hugging Face](https://huggingface.co/THUDM/CogVideoX1.5-5B-I2V)：未找到描述
- [thud (Adam M)](https://huggingface.co/THUD)：未找到描述
- [ByteDance/Hyper-SD · Hugging Face](https://huggingface.co/ByteDance/Hyper-SD)：未找到描述
- [GitHub - kijai/ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper)：通过在 GitHub 上创建账号，为 kijai/ComfyUI-CogVideoXWrapper 的开发做出贡献。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1308644641016647732) (17 条消息🔥):

> - `NotebookLM 中的音频生成`
> - `Notebooks 的外部访问`
> - `播客创建`
> - `转录功能`
> - `定制建议`

- **使用 NotebookLM 生成播客**：一位成员展示了他们的播客，由 AI 角色讨论**气候创新**，并强调了涉及的多个步骤，包括使用 AI 工具和 NotebookLM。
  
  - 他们分享了[播客链接](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255)，并详细介绍了创建不同角色之间对话的过程。
- **外部操作员访问 Notebook 功能**：围绕简化外部操作员访问 NotebookLM 的讨论展开，一位成员透露他们利用商业版 Gmail 来简化流程。
  
  - 他们提到已经创建了各种参考指南，这些指南已在他们的快速入门手册文件夹中提供。
- **音频文件的转录选项**：建议新用户将生成的音频文件重新上传到 NotebookLM，它会为他们转录内容。
  
  - 另外，一位成员建议对音频文件使用 MS Word 的 Dictate...Transcribe 功能。
- **播客音频创建的反馈**：成员们分享了使用 NotebookLM 创建播客的经验，强调了有效的音频生成功能。
  
  - 一位用户在 [Spotify](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde) 上分享了他们的德语播客，并表示对定制建议感兴趣。
- **Notebook 音频的直接链接**：多位成员分享了与 NotebookLM 生成的音频内容相关的链接，包括个人播客和讨论独特话题的特定剧集。
  
  - 其中一个值得注意的剧集提到了微重力环境下的葡萄酒陈酿，引用了具体的科学实验和结果。

**提到的链接**：

- [未找到标题](https://notebooklm.google.com/notebook/c92bf58d-3a48-4462-9801-964d86829a1c/audio)：未找到描述
- [Wein & Wahnsinn: Verrückte Fakten in 5 Minuten](https://open.spotify.com/show/5OapAqDLaWMxAzqXgywBZH?si=2e422be55d784fde)：播客 · bbrocher · 欢迎来到 Wein & Wahnsinn：5 分钟疯狂事实，这个播客将带你进入离奇、荒诞且往往出人意料的葡萄酒世界。在这里，你可以期待...
- [未找到标题](https://notebooklm.google.com/notebook/c92bf58d-3a48-4462-9801-964)：未找到描述
- [Anti Schwerkraft Weine](https://open.spotify.com/episode/35ahZlYN43acsu8rNTJyYD?si=16d8b1e6740a4b99)：Wein & Wahnsinn: Verrückte Fakten in 5 Minuten · 剧集
- [Subly - Story - Delhi Air Pollution](https://preview.getsubly.com/preview/89c122b6-cc30-458a-9d2b-d3728098b255)：需要为您的视频添加字幕吗？免费试用 Subly！在几分钟内，我们自动转录并翻译您的视频或音频。您可以进行样式设计，添加品牌 Logo 和标题，准备好分享...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1308679850931912775) (35 messages🔥):

> - `Combining Notes Feature` (合并笔记功能)
> - `Reliability of Uploaded Sources` (上传源的可靠性)
> - `Sharing Notebooks` (共享 Notebooks)
> - `Deep Dive Document Generation` (深度挖掘文档生成)
> - `Limitations on Uploading Large Files` (上传大文件的限制)

- **合并笔记功能正在讨论中**：成员们正在讨论现有的“Combine to note”功能，该功能允许用户将多个笔记合并为一个文档。
  
  - 一位成员对将笔记转换为实际源表示困惑，质疑其效用，因为合并笔记已经可以实现。
- **关于上传源可靠性的看法**：成员们分享了在使用上传源时遇到幻觉（hallucinations）的混合体验，有些人认为它很可靠，而另一些人则指出了差异。
  
  - 一位成员指出，引用（citations）通常质量很高，不会陷入典型的幻觉陷阱。
- **共享 Notebooks 的挑战**：一位用户询问了与朋友共享 Notebooks 的流程，在成功执行方面遇到了困难。
  
  - 另一位成员确认，界面右上角有一个“share note”按钮可用于此目的。
- **深度挖掘文档生成引发关注**：将笔记创建并总结为单个文档的潜力引发了成员间的对话，尽管有些人认为这有些多余。
  
  - 一位成员提到能够汇编摘要，并表示如果可以下载源文件，该功能会更有用。
- **探索上传大文件的限制**：一位成员在尝试上传包含超过 444,000 行的大型 CSV 文件时遇到错误，发现限制在 10,000 行左右。
  
  - 他们向其他人寻求确认平台内是否存在任何强制的文件大小限制。

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1308767374060687360) (48 messages🔥):

> - `DeepSeek-R1-Lite-Preview`
> - `GPT-4o Update`
> - `Truffles Hardware Device`
> - `Vercel Acquires Grep`
> - `Claude Availability Issues`

- **DeepSeek-R1-Lite-Preview 发布**：DeepSeek 宣布 [DeepSeek-R1-Lite-Preview](http://chat.deepseek.com) 上线，其在 AIME 和 MATH 基准测试中表现出色，并具有透明的推理过程。
  
  - 该系统显示出随着推理长度增加性能提升的趋势，用户对其在各种任务中的应用潜力感到兴奋。
- **新的 GPT-4o 更新发布**：OpenAI 发布了新的 GPT-4o 快照 `gpt-4o-2024-11-20`，增强了创意写作能力，并能更高效地处理上传的文件以获取洞察。
  
  - 在最近的性能测试中，GPT-4o 在多个类别中重夺榜首，展示了显著的进步。
- **Truffles 硬件设备受到关注**：用户识别出 “Truffles” 是一款半透明硬件设备，旨在用于在家中自托管 LLM，被幽默地称为“发光的乳房植入物”。
  
  - 这种奇特的描述反映了围绕创新家庭 LLM 解决方案的轻松讨论。
- **Vercel 收购 Grep**：Vercel 宣布收购 [Grep](https://grep.app/)，该工具允许开发者在超过 500,000 个公共仓库中搜索代码。
  
  - 创始人 Dan Fox 将加入 Vercel 的 AI 团队以增强代码搜索能力，旨在提高开发效率。
- **Claude 面临可用性问题**：用户报告了 Claude 间歇性的可用性问题，一些人经历了停机，而另一些人发现它可以正常运行。
  
  - 随后展开了关于该服务可靠性的讨论，导致一些用户在社交媒体上查看更新。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/itsalltruffles)：未找到描述
- [来自 Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1847277918243754156?s=46)：来自 NVIDIA 研究人员的新 Transformer 架构修改。nGPT：一种基于超球面的 Transformer，可实现 4-20 倍的训练加速并提高 LLM 的稳定性。**本论文中的提案**...
- [来自 Teortaxes▶️ (@teortaxesTex) 的推文](https://x.com/teortaxesTex/status/1859295840768229880)：顺便说一句：通过 pivot tokens 扩展测试时计算（scaling test time compute）将与工具使用产生巨大的协同效应。看：它已经渴望存在，乞求一个容器。正如 @gwern 和 @xenoco 所预言的那样...
- [来自 jack morris (@jxmnop) 的推文](https://x.com/jxmnop/status/1858627599981048211?s=46)：更多有趣的开源研究新闻 - 新论文发布 (nGPT) - 声称训练速度比 GPT 快 4-20 倍 - 令人震惊 - 非常酷 - 非常有价值 - 社区尝试复现 - 结果站不住脚 - 转向...

- [来自 xjdr (@_xjdr) 的推文](https://x.com/_xjdr/status/1859272181844422813)：whalebros 表现出色。它不仅似乎复现了 o1-preview 的结果，还非常有效地复现了（至少部分）过程。我猜它使用了非常相似的……
- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1859265550767067518)：传闻 DeepSeek R1-Lite 是一个拥有 16B 参数的 MoE 模型，其中激活参数为 2.4B。如果属实，他们的 MATH 分数从 17.1 提升到了 91.6。引用 Phil (@phill__1) @nrehiew_ 来自他们的微信公告：
- [来自 DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1859200141355536422)：🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级别的性能。💡 实时透明的思考过程。🛠️ 开源……
- [来自 Akshay Agrawal (@akshaykagrawal) 的推文](https://x.com/akshaykagrawal/status/1858933658025160719?s=46)：我的联合创始人 @themylesfiles 和我创立了 Marimo Inc.，以继续构建 @marimo_io notebook 和其他 Python 数据工具。我们筹集了由 @antgoldbloom 和 @shyammani 领投的 500 万美元种子轮融资……
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1859296408131731592)：新的 GPT-4o 快照现已在 API 中提供，版本号为 `gpt-4o-2024-11-20`：https://platform.openai.com/docs/models#gpt-4o。引用 OpenAI (@OpenAI) GPT-4o 获得了更新 🎉 模型的创意写作……
- [来自 Phil (@phill__1) 的推文](https://x.com/phill__1/status/1859263165000729024)：@nrehiew_ 来自他们的微信公告：
- [比特币亿万富翁 Barry Silbert 谈论他的下一个大赌注——“去中心化 AI”](https://fortune.com/crypto/2024/11/20/decentralized-ai-yuma-bittensor-bcg-barry-silbert/)：Silbert 将担任 Yuma 的 CEO，这是 DCG 的一家新子公司，专注于与 Bittensor 区块链相关的 AI 生态系统。
- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1859268539770900923)：它解决了 Yann Lecun 的 7 齿轮问题。
- [Vercel 收购 Grep 以加速代码搜索 - Vercel](https://vercel.com/blog/vercel-acquires-grep)：宣布收购 Grep，以进一步实现我们帮助开发者更快工作和交付的使命。
- [来自 Yaroslav Bulatov (@yaroslavvb) 的推文](https://x.com/yaroslavvb/status/1859032271208223191?s=46)：有几个独立的团队正在尝试复现 https://github.com/NVIDIA/ngpt。目前还存在一些 bug，所以可能需要一些时间，但我看好其核心理念，因为它代表了归一化（normalization）……
- [来自 Akari Asai (@AkariAsai) 的推文](https://x.com/akariasai/status/1858876162467881015?s=46)：3/ 🔍 什么是 OpenScholar？它是一个检索增强型 LM，具有 1️⃣ 一个包含 4500 万+ 开放获取论文的数据存储库 2️⃣ 一个专门的检索器（retriever）和重排序器（reranker）用于搜索数据存储库 3️⃣ 一个经过微调的 8B Llama……
- [来自 Rohan Paul (@rohanpaul_ai) 的推文](https://x.com/rohanpaul_ai/status/1847277918243754156?s=4)：来自 NVIDIA 研究人员的新 Transformer 架构改进。nGPT：一种基于超球面的 Transformer，可实现 4-20 倍的训练加速并提高 LLMs 的稳定性。\*\*本文中的提案\*\*……
- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1859228088292213007)：嗯，这很有趣，我尝试过的大多数模型在生成前 10 个单词时都不会失败，而这个模型没有意识到它只生成了 7 个单词而不是 10 个。
- [来自 Teortaxes▶️ (@teortaxesTex) 的推文](https://x.com/teortaxesTex/status/1859224352731828303/photo/1)：这个人的研究重点挺有意思的。引用 Zhihong Shao (@zhs05232838) 我们的 DeepSeek 推理模型在代码和数学方面表现出色。快来试试吧！
- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1859218213915001157?s=46)：关于 DeepSeek 发布最有趣的一点是，他们基本上复现了 o1 的扩展定律（scaling laws）。引用 DeepSeek (@deepseek_ai) 🌟 DeepSeek-R1-Lite-Preview 的推理扩展定律：更长的 R...
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1859307979184689269)：来自 Chatbot Arena 的激动人心的消息❤️‍🔥 在过去的一周里，最新的 @OpenAI ChatGPT-4o (20241120) 以“anonymous-chatbot”的身份匿名参加了测试，收集了 8,000 多张社区投票。结果是？……
- [来自 Tim Shi (@timshi_ai) 的推文](https://x.com/timshi_ai/status/1858937064647258326?s=46)：激动人心的更新！🌊 Cresta 已筹集 1.25 亿美元 D 轮融资，以加速构建用于客户体验的 Agent！
- [来自 Ryan Sun (@sun_hanchi) 的推文](https://x.com/sun_hanchi/status/1859243238986588166?s=46)：等等，Lite 版使用的是 16B MoE 基础模型😱😱😱 所以从技术上讲它对应的是 o1-mini。想象一下完整版……顺便说一下，DeepSeek 可能没有足够的 GPU 来对完整模型进行 RL，所以我们会看到反向……
- [来自 Teortaxes▶️ (@teortaxesTex) 的推文](https://x.com/teortaxestex/status/1859259359630356955?s=46)：真的，我紧张得一直在啃指甲。10/10 会再次尝试提示。失败、转折、最后的决战——以及辉煌的胜利。DeepSeek-r1-lite 通过将 LLM 推理转变为一种……

- [来自 jack morris (@jxmnop) 的推文](https://x.com/jxmnop/status/1858895357209403510?s=46): nGPT 的 Transformer 实现中的错误非常容易犯，残差流（residual stream）被传播为 `x = norm(x) + attn(norm(x))` 而不是 `x = x + attn(norm(x))`。简而言之，这破坏了...

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/) (1 条消息):

cappuchinoraro: 谢谢

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1308694276040949790) (5 条消息):

> - `Triton 教程性能`
> - `GPU 对比`
> - `Softmax Kernel 性能分析`

- **Triton vs Torch Softmax 性能**: 一位成员在 RTX 3060 上使用 [Triton 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html) 对比了 Triton 的融合 Softmax 操作与 PyTorch 原生实现的性能，指出 Triton 的表现更平滑。
  
  - 他们强调虽然 Triton 总体表现更好，但在某些情况下 PyTorch 的性能与 Triton 持平甚至更优。
- **吞吐量观察不一致**: 另一位成员评论了 Triton 教程结果与他们自己观察到的吞吐量差异，认为 GPU 硬件的不同可能会影响结果。
  
  - 他们推测性能对比在不同 GPU 上可能不可靠，并建议在 A100 上进行测试以查看结果是否稳定。
- **在 4090 上分析 Softmax Kernel**: 一位成员补充说他们正在 4090 上分析 Softmax Kernel，记录了固定 Batch Size 为 128 的性能指标，并将结果与 Triton 教程进行了对比。
  
  - 他们表示其发现与教程中详述的结果更加一致，尽管他们关注的是 ops/sec 而不是 GB/sec。

 

**提到的链接**: [Fused Softmax — Triton 文档](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html): 未找到描述

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/1308721932853706803) (3 条消息):

> - `Readme 更新`
> - `Torchchat 与 Torchtune 的关联`

- **Readme 需要小幅更新**: 一位成员建议 Readme 应该提到 **torchchat** 也与 **torchtune** 相关联。
  
  - 这得到了另一位成员的认同，并提供了一个处理该变更的 [相关 GitHub Pull Request 链接](https://github.com/pytorch/ao/pull/1319)。
- **用于更新的 GitHub Pull Request**: 由 **drisspg** 提交的上述 GitHub Pull Request 旨在用必要信息更新 **README.md**。
  
  - 提到该 Pull Request 是相关的，并对该主题提供了全面的更新，反映在 GitHub [链接](https://github.com/pytorch/ao/pull/1319) 中。

 

**提到的链接**: [Update README.md by drisspg · Pull Request #1319 · pytorch/ao](https://github.com/pytorch/ao/pull/1319): 未找到描述

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1308874151301353504) (2 条消息):

> - `票价变动`
> - `尽早购票`

- **票价变动频率**: 一位成员询问票价变动的频率，以及现在购买还是推迟购买更明智。
  
  - 另一位成员回答说**通常越早买越便宜**，并强调如果你距离旅行还有几个月，价格提醒会更有用。
- **关于早买票的建议**: 有人建议早买票通常会获得更低的价格。
  
  - *价格变动提醒*对于那些提前几个月规划行程的人来说比临近购买更有用。

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/1308711923084427305) (11 条消息🔥):

> - `Metal GEMM Implementations` (Metal GEMM 实现)
> - `WebGPU and Metal Compatibility` (WebGPU 与 Metal 兼容性)
> - `Register Optimization Techniques` (寄存器优化技术)
> - `Performance Regressions in Dawn` (Dawn 中的性能回归)
> - `AGX Machine Code Disassembly Tools` (AGX 机器码反汇编工具)

- **Philip Turner 的 Metal GEMM 实现见解**：一名成员重点介绍了 [Philip Turner 的仓库](https://github.com/philipturner/metal-flash-attention)，其中包含 Metal GEMM 实现，但提到代码过去的可读性更高。
  
  - 他们还指出，自己的 Metal GEMM 达到了理论最大速度的 **85-90%**，与 Turner 的实现类似。
- **WebGPU 在 MMA Intrinsics 方面的挑战**：一位参与者回忆起关于 Shared Memory 的问题，并询问 WebGPU 是否暴露了可以通过 Metal 访问的 **MMA intrinsics**。
  
  - 他们承认不确定编译器是否针对此功能进行了改进。
- **优化寄存器利用率**：一名成员分享了一种技术，通过将数组访问替换为指针递增（从 *a[i]* 改为 *a++*），节省了 **25 个寄存器**。
  
  - 他们提醒，Metal 编译器需要进行重度优化，特别是将寻址计算移出热循环（hot loops）。
- **Dawn 的性能回归问题**：针对最新版本 Dawn 的性能回归提出了担忧，特别是在 Chrome **130** 之后的 WGSL 到 Metal 工作流中。
  
  - 有建议称 Chrome **131** 相比 **130** 提高了性能，但仍落后于 **129**，可能存在与 UB 检查代码位置相关的问题。
- **AGX 机器码反汇编工具**：分享了一个由 Freedesktop 开发者维护的用于反汇编 **AGX 机器码** 的工具：[applegpu 仓库](https://github.com/dougallj/applegpu)。
  
  - 该资源在衡量编译代码中的寄存器利用率时被引用。

 

**提到的链接**：[Chromium](https://issues.chromium.org/issues/41486305)：未找到描述

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1308703908868395058) (17 条消息🔥):

> - `Debugging Assistance` (调试协助)
> - `CUDA Device Mapping` (CUDA 设备映射)
> - `Model Distribution across GPUs` (跨 GPU 的模型分布)
> - `Tensor Parallelism` (张量并行)
> - `Hugging Face Sharding Strategy` (Hugging Face 分片策略)

- **提供调试协助**：一名成员提出帮助调试，并建议通过逐一关闭优化来隔离问题。
  
  - 另一名成员确认了这种方法，并对快速提供的帮助表示感谢。
- **CUDA 设备映射成功**：在 L4 上使用 `cuda` 作为设备映射运行良好，引发了关于快速验证解决方案的幽默交流。
  
  - 该解决方案归功于另一名成员，预计他很快会提供更多见解。
- **讨论模型分布问题**：一名成员表示担心，尽管使用了 'auto'，模型在执行期间可能只利用了一个 GPU。
  
  - 这引发了关于 Tensor Parallelism 以及无法将模型分布到所有四个 GPU 的局限性的讨论。
- **对模型使用的观察**：成员们讨论了关于 'auto' 设置如何跨 GPU 分布模型的观察结果。
  
  - 在检查使用统计数据后，对 Hugging Face 采用的默认分片（sharding）策略表示不确定。
- **来自 0x000ff4 的测试更新**：一名成员简要更新了某个项目的测试工作进展。
  
  - 未分享关于测试过程或其目标的更多细节。

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1308729659390955521) (4 messages):

> - `FLUX 推理优化`
> - `CPU offloading 技术`
> - `不同机器上的 GPU 性能`

- **通过 CPU Offload 优化 FLUX 速度**：一位成员报告称，通过在 **4070Ti SUPER** 上实施逐层 CPU offload 技术，实现了 **200% 的速度提升**，推理时间低至 **1.23 s/it**。
  
  - 他们的结果表明，相比于基准方法 `.enable_sequential_cpu_offload()`（耗时 **3.72 s/it**）有显著改进。
- **Mobicham 关于并行 Offloading 的见解**：另一位成员分享了使用 pinned memory 和 CUDA streams 进行 scale 和 zero offloading 的经验，指出这在高性能机器上表现良好，但在共享实例上速度较慢。
  
  - *他们推测了在 runpod/vastai 等资源受限环境下的效率。*
- **关于 LLM 解码瓶颈的讨论**：作为回应，一位成员评论说，对于 LLM 解码，尽管有重叠数据传输和计算的方法，但 CPU 到 CUDA 的传输仍可能成为瓶颈。
  
  - 然而，在使用 **FLUX** 进行图像生成时，由于其较高的算术强度 (arithmetic intensity)，缓慢的数据传输影响较小。
- **分享视频资源以获取更多见解**：一位成员分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=9Q7jMiXayXE)，该链接可能针对所讨论的优化提供额外的见解或相关内容。
  
  - 对于那些正在探索类似性能增强的人来说，这个视频可能会有所帮助。

**提到的链接**：[Thien Tran (@gaunernst) 的推文](https://x.com/gaunernst/status/1859168533554565325)：将 FLUX CPU offloading 速度提升 200%。在 4070Ti SUPER (16GB) 上，基准测试 (.enable_sequential_cpu_offload())：3.72 s/it + pin memory：2.09 s/it (+78%) + CUDA stream (显式同步)：1.32 s/it ...

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1308647313366978560) (27 messages🔥):

> - `DeepSeek-R1-Lite-Preview`
> - `用于写书的 AI Agent`
> - `LLM 知识评估`

- **DeepSeek-R1-Lite-Preview 发布**：[DeepSeek-R1-Lite-Preview](https://x.com/deepseek_ai/status/1859200141355536422) 现已上线，在 AIME 和 MATH 基准测试中展现出 **o1-preview 级别的性能**，具备强大的推理能力。
  
  - 它具有实时的**透明思维过程**，开源模型和 API 即将推出。
- **AI 团队合作写书**：[Venture Twins](https://x.com/venturetwins/status/1859298925930479998) 展示的一个项目涉及十个 AI Agent 协作编写一本完全自主的小说，每个 Agent 扮演不同的角色，如设定叙事、保持连贯性和研究情节。
  
  - 随着它们实时工作，可以通过 [GitHub commits](https://github.com/Lesterpaintstheworld/terminal-velocity) 跟踪进度。
- **创新的 LLM 基准测试提案**：一位成员提议了一种基准测试，用于测试 LLM 对“不知道的内容”的了解程度，其中正确的回答不计分。
  
  - 该评估侧重于模型如何回答错误的问题，结合了知识和推理。

**提到的链接**：

- [DeepSeek (@deepseek_ai) 的推文](https://x.com/deepseek_ai/status/1859200141355536422)：🚀 DeepSeek-R1-Lite-Preview 现已上线：释放超强推理能力！🔍 在 AIME 和 MATH 基准测试中达到 o1-preview 级别性能。💡 实时的透明思维过程。🛠️ 开源...
- [Justine Moore (@venturetwins) 的推文](https://x.com/venturetwins/status/1859298925930479998)：有人正在使用一个由 10 个 AI Agent 组成的团队编写一本完全自主的小说。它们各有分工——设定叙事、保持连贯性、研究情节……你可以关注……
- [GitHub - Lesterpaintstheworld/terminal-velocity: 由 10 个团队、共 100 个 AI Agent 自主创作的小说](https://github.com/Lesterpaintstheworld/terminal-velocity)：由 10 个团队、共 100 个 AI Agent 自主创作的小说 - Lesterpaintstheworld/terminal-velocity

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1308676993440550983) (8 条消息🔥):

> - `Learning Rate Scheduling`
> - `Warmup and Decay Strategies`
> - `LLMs 的 Test Time Scaling`
> - `Cyclic Learning Rate Schedulers`

- **理解跨 Epoch 的 Learning Rate 行为**：讨论了**特定 Step 的 Learning Rate** 是否应该在不同 Epoch 之间保持一致。会议明确了 Learning Rate 通常在 **Warmup 期间上升**，然后随时间 **Decay**，这导致不同 Epoch 中对应 Step 的值不同。
- **探索 Learning Rate Scheduler 配置**：一位成员提到，以前 Learning Rate 是在每个 Epoch 进行调整的，但目前通常是基于所有 Epoch 的 **Total Steps** 进行配置，并在每个 Gradient Step 进行调整。他们鼓励研究 **Cyclic Learning Rate Schedulers** 这种现代方法。
- **关于 LLMs Test Time Scaling 的咨询**：在另一个咨询中，一位成员询问是否有人正在研究 Large Language Models 的 **Test Time Scaling**，并征求相关想法。这引发了好奇心，并暗示了关于 Scaling 策略的持续讨论。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308839061980708874) (2 条消息):

> - `LLMs 推理能力`
> - `Generative Agent 模拟`

- **LLMs 无需 Prompt 即可推理**：研究表明，通过改变 Decoding 过程以检查前 $k$ 个备选 Token，**Large Language Models (LLMs)** 可以在没有显式 Prompt 的情况下展示出类似于 **Chain-of-Thought (CoT)** 的推理路径。
  
  - 该方法突出了 LLMs 的**内在推理能力**，并表明 CoT 路径可能固有地存在于其序列中。
- **超过 1,000 人的行为模拟**：一种新型架构模拟了 **1,052 个真实个体**的态度和行为，证明了 **Generative Agents** 在综合社会调查（General Social Survey）中能以 **85% 的准确率**复制人类反应。
  
  - 该架构减少了**不同种族和意识形态群体**之间的准确性偏差，为研究个人和集体行为的工具铺平了道路。

**提到的链接**：

- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)：在增强 Large Language Models (LLMs) 的推理能力方面，先前的研究主要集中在特定的 Prompting 技术，如 Few-shot 或 Zero-shot Chain-of-Thought (CoT) Prompting...
- [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109)：人类行为模拟的愿景——即在各个领域复制人类行为的通用计算 Agent——可以在政策制定和社会科学中实现广泛的应用。我们提出...

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1308670981618929746) (3 条消息):

> - `Soft Prompts`
> - `LLM 优化`

- **关于 Soft Prompts 研究的咨询**：一位成员询问是否有人研究过某篇 [帖子](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f) 中提到的 LLMs **Soft Prompts** 概念。*他们强调了 Soft Prompts 在将 System Prompts 优化到 Embedding 空间方面的潜力。*
  
  - 另一位成员回应称，Soft Prompts 的想法**非常有趣**，表明了对该话题的一定关注。
- **关于 Soft Prompts 价值的讨论**：对话表明了对 LLM 社区中 Soft Prompts **尚未开发的潜力**的好奇。*成员们似乎认为，对这一领域的更多探索可能会为进一步的发展带来成果。*

**提到的链接**：[@saganite.bsky.social](https://bsky.app/profile/saganite.bsky.social/post/3lbeajzg3ms2f)：真心想弄清楚为什么 "Soft Prompts" 在 LLMs 中不更常用。对于那些不熟悉的人来说，Soft Prompts 是已经转换为 Embedding 的 System Prompts ...

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308839061980708874) (2 条消息):

> - `LLMs Reasoning without Prompting`
> - `Generative Agent Behavioral Simulations`

- **LLMs 在无需 Prompt 的情况下进行有效推理**：一项研究调查了大型语言模型 (LLMs) 是否可以通过改变解码过程，在不使用 Prompt 的情况下进行有效推理，结果表明 CoT 推理路径往往是自然产生的。这种新颖的方法允许在没有复杂的各种手动 Prompt Engineering 的情况下，评估 LLMs 的**内在推理能力 (intrinsic reasoning abilities)**。
  
  - 研究结果表明，通过利用 **top-k** 替代 Token，研究人员可以从预训练模型中激发有效的推理，从而深入了解其固有能力。
- **针对 1,052 名个体的突破性行为模拟**：一项新研究提出了一种创新的 Agent 架构，利用大型语言模型根据定性访谈来模拟 **1,052** 名真实个体的态度和行为。这些 Generative Agents 在综合社会调查 (General Social Survey) 中准确地复制了受访者的回答，准确率高达 **85%**，与自我报告的答案相匹配。
  
  - 值得注意的是，与仅依赖人口统计学描述的 Agent 相比，该架构减少了不同种族和意识形态群体的准确性偏见，为在社会科学中探索个人和集体行为的工具奠定了基础。

**提到的链接**：

- [Generative Agent Simulations of 1,000 People](https://arxiv.org/abs/2411.10109)：人类行为模拟的前景——即在各个领域复制人类行为的通用计算 Agent——可以在政策制定和社会科学中实现广泛的应用。我们提...
- [Chain-of-Thought Reasoning Without Prompting](https://arxiv.org/abs/2402.10200)：在增强大型语言模型 (LLMs) 的推理能力方面，先前的研究主要集中在特定的 Prompt 技术上，例如 few-shot 或 zero-shot Chain-of-Thought (CoT) Prompting...

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1308645095439994921) (18 条消息🔥):

> - `Daily Theme Winner`
> - `API Usage Discussion`
> - `Model Options and Performance`

- **庆祝每日主题获胜**：一位成员在首次赢得**每日主题 (daily theme)** 挑战后表达了喜悦，称他们“非常开心”。
  
  - 这引起了社区的积极反应，突显了成员对持续活动的参与度。
- **寻求 API 解决方案**：一位成员提到正在寻找 **API** 或工具，但发现两个选项都不尽如人意，表达了挫败感。
  
  - 这反映了社区内对寻找有用资源的广泛兴趣。
- **澄清模型选项**：讨论围绕 **4o 模型** 以及它使用的是 **o1 mini** 还是 **o1 preview** 展开，确认其可能使用了 **o1 mini**。
  
  - 另一位成员建议检查设置以验证选项，鼓励动手解决问题。
- **频道切换困惑**：有询问关于 **ai-discussions** 频道是否与另一个特定频道进行了交换，表明可能存在沟通误解。
  
  - 一位成员对混淆表示歉意，并提到打算将评论移至正确的 **off-topic** 空间。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1308733699793092608) (3 条消息):

> - `High Temperature Performance`
> - `Beta Access to o1`
> - `Gaming Character Genshin Impact`

- **高 Temperature 性能咨询**：一位成员询问在**较高 Temperature** 下性能的提升是否与他们的 Prompt 风格有关，并怀疑是否设置了过多的引导规则或约束。
  
  - 这为优化 Prompt 设计以获得更好的 AI 响应能力提供了有趣的思考。
- **感谢 o1 的 Beta 访问权限**：一位成员表达了对 NH 授予他们 **o1 Beta 访问权限** 的兴奋和感激，这让他们的早晨变得更加美好。
  
  - *“Woo! 感谢 NH 让今天早上变得更加灿烂”* 反映了对新更新的兴奋之情。
- **《原神》(Genshin Impact) 角色嘉明 (Gaming) 的困惑**：一位成员提出了关于 ChatGPT 无法检索到 **Gaming**（《原神》中的一个角色）信息的问题。
  
  - 这突显了 AI 在涉及热门游戏角色及其背景知识方面可能存在的差距。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1308659715097497631) (8 messages🔥):

> - `Using Delimiters in Prompts`（在 Prompt 中使用分隔符）
> - `Markdown for Clarity`（使用 Markdown 提高清晰度）
> - `Game Mechanics Understanding`（游戏机制理解）
> - `Model Context Expectations`（模型上下文预期）

- **使用分隔符提高清晰度**：一位成员分享了 OpenAI 关于使用三引号或 XML 标签等分隔符的建议，以帮助模型清晰地解析输入的不同部分。
  
  - 这种方法有助于更好地构建 Prompt 结构，从而改善模型响应，使输入解析更容易。
- **Markdown 作为格式化工具**：另一位成员建议使用 Markdown 语法创建结构化的标题和列表，以提高 Prompt 的清晰度。
  
  - 示例包括用于主标题的 `# Heading 1` 和各种列表格式，展示了结构化文本如何增强模型的理解。
- **通用模型的适应性**：讨论指出，由于 GPT 是通用模型，它可能不会严格遵守像 Tic Tac Toe 这样的游戏机制。
  
  - 这强调了在处理特定场景时，清晰的上下文和预期对于引导模型输出的重要性。
- **为模型上下文直接标记**：一位成员提议提供显式标签，如 'Section head: This Topic'，以帮助模型正确推断上下文。
  
  - 该技术强调了模型依赖标记和上下文来生成更相关的响应。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1308659715097497631) (8 messages🔥):

> - `Using Delimiters for Clarity`（使用分隔符提高清晰度）
> - `Markdown Formatting`（Markdown 格式化）
> - `Improving GPT's Understanding`（提高 GPT 的理解力）
> - `Game Mechanics in GPT`（GPT 中的游戏机制）
> - `Model Context and Labeling`（模型上下文与标记）

- **在 Prompt 中使用分隔符提高清晰度**：根据 OpenAI 的建议，使用三引号或章节标题等分隔符可以帮助模型理清输入的各个部分。
  
  - 这种做法有助于模型有效地解释不同章节，增强整体理解。
- **Markdown 格式化技巧**：Markdown 可用于创建标题和格式化文本，讨论中分享了 `# This is Heading 1` 以及 **加粗** 或 *斜体* 样式的示例。
  
  - 许多成员强调了反引号行和列表在清晰组织内容方面的实用性。
- **游戏机制与 GPT 响应**：一位成员指出，由于 GPT 的通用性质，它可能会偏离 **Tic Tac Toe** 等简单游戏机制。
  
  - 幽默的是，他们提到在讨论这个话题时获得了 **25 经验值**。
- **模型上下文与标记以实现更好的交互**：参与者建议直接标记章节，例如使用 'Section head: This Topic'，以引导模型的理解。
  
  - 强调提供上下文有助于 LLM 进行推测和模式匹配，从而丰富其响应。

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1308645952181112925) (12 messages🔥):

> - `API Key Issues`（API Key 问题）
> - `CORS Errors`（CORS 错误）
> - `Python Learning Projects`（Python 学习项目）

- **403 错误表示 API 问题**：多位成员讨论了遇到 **403 错误** 的情况，这表明在尝试访问某些功能时，要么是 API Key 无效，要么是调用了旧的 Endpoint。
  
  - 一位成员分享说，在检查其 API Key 后，他们遇到了 **fetch 错误** 以及使用 Sandbox 功能的困难。
- **关于 Free Tier API 限制的咨询**：一位成员确认他们仍处于 **Free Tier**，并尝试升级到 Production Key 以解决持续存在的问题，但面临进一步挑战。
  
  - 他们报告了控制台中的几个 **CORS 错误**，并指出其设置是标准的，没有安装额外的插件。
- **Python 共同学习**：一位成员提到参加了 **30 Days of Python**，并与小组分享了他们的学习项目。
  
  - 这引发了关于其他成员正在进行的项目的普遍询问，培养了社区感和协作感。

**提及的链接**：

- [Cohere Dashboard 上的 Fetch 错误](https://imgur.com/a/zMBZQfz)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的迷因、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频等来振奋你的精神……
- [imgur.com](https://imgur.com/uP2tgwp)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的迷因、有趣的 GIF、鼓舞人心的故事、病毒式传播的视频等来振奋你的精神……

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1308709449829187584) (6 messages):

> - `基于账户的设置`
> - `模型训练提示词`
> - `保加利亚语数据集`
> - `模型微调技术`
> - `贡献流程`

- **账户特定配置**：提到将基于每个账户进行调整，强调了定制化设置的必要性。
  
  - 这种方法确保了针对个人用户需求的定制化。
- **引导 command-r**：有人建议允许 command-r 在用户监督下起草系统提示词，以增强性能。
  
  - 这可以简化提示词的创建过程。
- **保加利亚语训练数据集**：讨论指出，特定于**保加利亚语**的额外训练数据对于模型微调至关重要。
  
  - 用户提出收集数据集，并请求在消息线程中分享发现，以便团队审查。
- **模型微调能力**：有人询问是否可以仅使用 Preamble 和可能的聊天历史来微调模型。
  
  - 这提出了关于模型对各种训练输入适应性的重要问题。
- **寻求贡献方面的帮助**：一位用户表示不确定如何开始贡献，并请求帮助以了解流程。
  
  - 这表明需要更清晰的贡献路径指南。

 

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1308829039607418920) (4 messages):

> - `RAG 聊天机器人问题`
> - `Cohere 多模态 Embeddings`
> - `速率限制问题`

- **RAG 聊天机器人遇到 API Token 错误**：一位用户报告称，尽管使用了来自 [dashboard](https://dashboard.cohere.com/api-keys) 的有效 Cohere API Key，但在执行其 RAG 聊天机器人代码时仍出现 `invalid api token` 错误。
  
  - 他们提供了代码片段，并请求协助识别错误来源。
- **对多模态 Embeddings 的赞赏以及对速率限制的担忧**：一位成员对新的图像多模态 Embeddings 表示兴奋，并指出在其应用中观察到了**极佳的改进**。
  
  - 然而，他们强调了**每分钟 40 次请求的速率限制**这一重大问题，这阻碍了他们的使用场景，并寻求潜在替代方案的建议。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1308766287882620989) (4 条消息):

> - `Harmony 开源项目`
> - `LLM 匹配算法竞赛`
> - `Harmony 的数据可用性`
> - `Harmony 中的 Natural Language Processing`
> - `Harmony 的 Discord 社区`

- **用于问卷协调的 Harmony 项目**：**Harmony** 项目旨在利用 LLM 对问卷项目和元数据进行回顾性协调，研究人员利用它来实现更好的数据兼容性。
  
  - 它通过 [此工具](https://harmonydata.ac.uk/) 促进了工具之间的比较，并探索了跨版本和语言的潜在兼容性。
- **加入竞赛以改进 LLM 算法**：Harmony 正在举办一场竞赛，以增强其 LLM 匹配算法，并为参与者提供奖品，无需具备 LLM 训练的先验经验。
  
  - 参与者可以在 [DOXA AI 上注册](https://harmonydata.ac.uk/doxa/) 参加竞赛，并协助使 **Harmony** 变得更加健壮。
- **Harmony 的数据可访问性**：成员们询问了开源 **Harmony** 项目中使用的数据，并得到了关于其可用性的回复。
  
  - 该项目的代码和数据可以在其 [GitHub 页面](https://github.com/harmonydata/harmony) 上找到。
- **利用 Natural Language Processing**：Harmony 项目利用 **Natural Language Processing** 来改进跨不同研究和语言的问卷项目的匹配。
  
  - 有关 Harmony 算法性能的更多见解，可以在详细的 [博客文章](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/) 中探索。
- **参与 Harmony Discord 社区**：该项目鼓励用户加入 **Harmony Discord 服务器**，参与讨论并为匹配挑战做出贡献。
  
  - 成员可以访问 **「matching-challenge」** 频道获取更新和协作。

**提到的链接**：

- [GitHub - harmonydata/harmony: The Harmony Python library: a research tool for psychologists to harmonise data and questionnaire items. Open source.](https://github.com/harmonydata/harmony)：Harmony Python 库：心理学家协调数据和问卷项目的研究工具。开源。- harmonydata/harmony
- [Harmony | A global platform for contextual data harmonisation](https://harmonydata.ac.uk/)：上下文数据协调的全球平台
- [Competition to train a Large Language Model for Harmony on DOXA AI | Harmony](https://harmonydata.ac.uk/doxa/)：在 DOXA AI 上为 Harmony 训练大语言模型的竞赛 | Harmony

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1308760060159201320) (7 条消息):

> - `使用 sdpa/flex 获取 Post-softmax 分数`
> - `Attention 分数计算`
> - `Flex Attention 更新`
> - `sdpa 性能基准测试`

- **使用 sdpa/flex 获取 Post-softmax 分数**：要使用 sdpa/flex 获取 **post-softmax 分数**，可以输入一个初始化为单位矩阵且形状为 `[..., seqlen, seqlen]` 的虚拟 Tensor。这种方法可能需要两次调用，尽管一些成员认为这并非必要。
  
  - 讨论的实现细节参考了 2.5.1 版本中 **flex 行为** 的潜在变化，这可能会影响该方法的可行性。
- **自行计算 Attention 分数**：一位成员建议直接使用 `torch.softmax(q @ k.transpose(-1, -2), dim=-1)` 计算 **attention 分数**，从而对存储选项提供更多控制。他们指出，由于 **F.sdpa/flex** 实现了 flash-attn 算法，因此需要进行一些分数重计算（recomputation）才能保存它们。
  
  - 另一位成员表示赞同，指出除非有特定原因需要避免，否则这将是一个直接的初步尝试。
- **对 sdpa 方法进行基准测试**：建议将提议的方法与 **naive sdpa** 方法进行基准测试，以识别性能差异。分数的数值误差可能会根据所使用的 **sdpa 后端** 和 **数据类型** 而有所不同。

**提到的链接**：[pytorch/torch/nn/attention/flex_attention.py at release/2.5 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/release/2.5/torch/nn/attention/flex_attention.py#L909)：Python 中具有强大 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1308808453610930196) (14 条消息🔥):

> - `Adaptive Batching 实现`
> - `改进 DPO 损失函数`
> - `标准方法与新研究方法的对比`
> - `服务器提升 (Server Boosts) 与 Nitro 订阅`
> - `代码结构与模块化担忧`

- **旨在优化 GPU 利用率的 Adaptive Batching**: 讨论集中在实现 [adaptive batching](https://github.com/pytorch/torchtune/pull/2035)，通过调整 batch size 来最大化 GPU 利用率，并避免训练过程中的 OOM 错误。
  
  - 有建议认为该功能可以作为未来 recipes 中的一个 flag 添加，理想情况下在 `packed=True` 时激活以保持效率。
- **评估 DPO 损失结构的变更**: 有人对当前 TRL 代码的结构表示担忧，并讨论了是否应包含近期关于 DPO 修改的论文，因为这些修改可能不够显著。
  
  - 有人呼吁明确是否移除 **SimPO** 及任何独立的类，以保持 DPO recipe 的整洁和直观。
- **倾向于标准方法而非新方法**: 大家达成共识，通常的做法是实现标准方法，同时为领域内其他人的创新策略保留灵活性。
  
  - 成员们讨论了尝试新的研究预印本 (preprints) 与坚持使用成熟技术之间的潜在权衡。
- **取消 Nitro 订阅的影响**: 一位成员提到，如果用户取消免费的 **Nitro** 订阅，**server boosts** 将被移除，强调了这对服务器管理的影响。
  
  - 这一评论引起了人们对维持订阅以获得不间断服务器福利的价值的关注。
- **深入探讨 TRL 代码挑战**: 针对 **TRL** 代码的复杂性和模块化提供了反馈，特别是针对不同参数进行多次检查所导致的问题。
  
  - 小组讨论了简化 DPO recipe 的必要性，以确保其更具可扩展性 (hackable)，从而增强未来的开发。

**提到的链接**: [由 krammnic 提交的添加 RPO、DPOP 损失，并在基础 DPO 损失中添加 lambda_dpop · Pull Request #2035 · pytorch/torchtune](https://github.com/pytorch/torchtune/pull/2035): Context What is the purpose of this PR? Is it to add a new feature fix a bug update tests and/or documentation other (please add here) Please link to any issues this PR addresses. Changelog W...

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1308760351541694537) (2 条消息):

> - `SageAttention`
> - `推理增益`

- **SageAttention 实现显著加速**: [SageAttention](https://github.com/thu-ml/SageAttention) 项目采用了量化注意力机制，与 **FlashAttention2** 和 **xformers** 相比，分别实现了 **2.1x** 和 **2.7x** 的加速，同时在各种模型中保持了端到端指标。
  
  - *Pretty cool inference gains here!* 暗示了对 SageAttention 所带来的性能提升的兴奋。
- **关于 SageAttention 推理增益的讨论**: 一位成员表达了对通过实现 SageAttention 获得推理增益 (inference gains) 的看法，指出其性能提升非常强劲。
  
  - 该话题引起了其他人的兴趣，可能引发关于其在各种 AI 模型中应用的进一步讨论。

**提到的链接**: [GitHub - thu-ml/SageAttention: 量化注意力机制，与 FlashAttention2 和 xformers 相比，分别实现了 2.1 倍和 2.7 倍的加速，且不会损失各种模型的端到端指标。](https://github.com/thu-ml/SageAttention/): Quantized Attention that achieves speedups of 2.1x and 2.7x compared to FlashAttention2 and xformers, respectively, without lossing end-to-end metrics across various models. - thu-ml/SageAttention

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1308670171505758260) (6 messages):

> - `Tinygrad and Triton Integration`
> - `SASS Assembler Questions`
> - `FOSDEM AI DevRoom Presentation`
> - `Tinybox Hackathon Proposal`

- **关于 Tinygrad 的 Triton 集成查询**：一位用户询问 **Tinygrad** 现在是否已经与 **Triton** 实现了原生集成，并引用了早期的讨论。
  
  - George Hotz 提示用户参考问题文档以获取澄清。
- **SASS Assembler 的意图**：讨论围绕即将编写的 **SASS assembler** 是否旨在取代 **ptxas** 展开。
  
  - 一位用户对其问题的相关性表示不确定，George Hotz 建议他们查阅问题文档。
- **FOSDEM AI DevRoom 演讲者征集**：一位社区成员分享了 **Tinygrad** 开发者在 2025 年 2 月 2 日举行的 **FOSDEM AI DevRoom** 上进行演讲的机会。
  
  - 他们强调了 **Tinygrad** 在 AI 行业中的重要性，并鼓励感兴趣的人员联系以进行协作。
- **Tinybox 黑客松构想**：同一位成员提议组织一场 FOSDEM 之前的黑客松，并邀请有人能携带 **Tinybox** 到现场进行实操体验。
  
  - 他们表达了与社区成员在比利时啤酒的陪伴下进行讨论的热情，希望这能增强活动氛围。

 

**提到的链接**：[FOSDEM 2025 - Low-Level AI Engineering & Hacking Dev Room](https://aifoundry.org/fosdem-2025-low-level-ai-engineering-hacking-dev-room)：探索 FOSDEM 新设立的 “Low-Level AI Hacking & Engineering” Dev Room，展示驱动 AI 行业的开源项目。提交会议申请或成为这一创新项目的赞助商...

 

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1308923419408207952) (1 messages):

> - `int64 indexing`
> - `huge tensors`

- **关于 int64 索引实用性的问题**：一位成员询问在不使用 **huge tensors** 的情况下，**int64 indexing** 是否有必要。
  
  - 讨论旨在澄清在没有大型 Tensor 应用的情况下，使用 **int64** 索引的场景或潜在优势。
- **索引技术探索**：社区正在深入研究 Tensor 操作中使用的各种 **indexing techniques**，其中可能包括 **int64**、**int32** 等。
  
  - 他们正在考虑这些索引方法对小型 Tensor 操作性能和效率的影响。

 

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1308838379256807444) (2 messages):

> - `Async functions in Mojo`
> - `Mojo library repository`

- **Mojo 同步函数中可 await 异步函数**：一位成员对于在 **Mojo** 的同步函数中能够 await 一个异步函数感到困惑，这与 Python 的限制形成了对比。
  
  - 他们正在寻求对这种异步功能处理差异的澄清或解释。
- **关于 Mojo 库仓库的查询**：另一位成员对是否存在类似于 **pip** 的 **Mojo** 库仓库感到好奇。
  
  - 他们正在寻找提供 **Mojo** 库访问权限的资源或链接。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1308916166185455768) (5 messages):

> - `Moonshine ASR Model Performance`
> - `Mojo Program Observations`
> - `Max API vs ONNX Performance`

- **使用 Max 测试 Moonshine ASR 模型**：一位用户使用 **Max** 的 Python API 和原生 **Mojo** 版本测试了 **Moonshine** ASR 模型的性能，指出两者的速度都比直接使用 **onnxruntime** Python 版本慢了约 **1.8 倍**。
  
  - **Mojo** 和 Python Max 版本转录 10 秒语音大约需要 **82ms**，而原生的 **onnxruntime** 仅需 **46ms**。
- **分享运行指令与观察结果**：在分享的 **mojo** 文件顶部的注释中提供了运行 **Moonshine** ASR 模型的指令。
  
  - 该用户的经验表明，将 **TensorMap** 传入 **Model.execute** 会导致崩溃，由于 **Mojo** 的限制，必须手动解包 **26 个参数**。
- **寻求 Mojo 性能优化**：该用户表示这是他们最初编写的 **Mojo** 程序之一，并承认可能不够地道（idiomatic）。
  
  - 他们请求协助以获得更好的性能，并强调渴望提高自己的 Mojo 和 Max 技能。

**提到的链接**：

- [moonshine.mojo](https://gist.github.com/keveman/ea167957fb6364470cb265c5d9aa9da1)：moonshine.mojo。GitHub Gist：即时分享代码、笔记和代码片段。
- [moonshine.py](https://gist.github.com/keveman/d2aea1a059c9a14972783ede2d6b6862)：moonshine.py。GitHub Gist：即时分享代码、笔记和代码片段。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1308703559465959444) (2 messages):

> - `Tencent Hunyuan Model`
> - `Bits and Bytes on MI300X`

- **关于腾讯混元模型微调的讨论**：一名成员询问了微调 [Tencent Hunyuan model](https://huggingface.co/tencent/Tencent-Hunyuan-Large) 的经验，并分享了包括 [GitHub](https://github.com/Tencent/Tencent-Hunyuan-Large) 和[官网](https://llm.hunyuan.tencent.com/)在内的多个有用链接。
  
  - 他们还提供了 [Technical Report](https://arxiv.org/abs/2411.02265) 和 [Demo](https://huggingface.co/spaces/tencent/Hunyuan-Large) 供参考。
- **在 MI300X 上使用 Bits and Bytes**：一名成员分享了在 MI300X 系统上使用 [Bits and Bytes](https://github.com/bitsandbytes-foundation/bitsandbytes) 的成功经验，强调了其易用性。
  
  - 他们强调在更新时要记住使用 `--no-deps` 标志，并分享了一个用于强制重新安装该包的单行命令。

**提到的链接**：[tencent/Tencent-Hunyuan-Large · Hugging Face](https://huggingface.co/tencent/Tencent-Hunyuan-Large)：未找到描述

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**community-showcase**](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 messages):

volko76：我们还需要正确地编写 Prompt 吗？
[https://youtu.be/m3Izr0wNfQc](https://youtu.be/m3Izr0wNfQc)

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1308662697252098079) (4 messages):

> - `Axolotl Collab Notebooks`
> - `Continual Pretraining of LLaMA`

- **咨询 Axolotl Collab Notebooks**：一位用户询问 Axolotl 是否提供任何可用于 **LLaMA 持续预训练（Continual Pretraining）**的 **collab notebooks**。
  
  - Phorm 作出回应，表示将搜索 **OpenAccess-AI-Collective/axolotl** 以获取相关信息。
- **Notebook 查询结果为未定义**：Phorm 的搜索结果返回为 **undefined**，表明目前没有可用于所述目的的 notebook。
  
  - 鼓励用户稍后再次查看这些资源可用性的更新。

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined)：更快地理解代码。

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1308822739083202641) (5 messages):

> - `multimodal problems`
> - `vision language models`
> - `mmmu notebook`

- **Juan 寻求多模态挑战的帮助**：Juan 询问在处理**多模态问题**时，如何使用对**视觉语言模型（Vision Language Models）**的实验性支持。
  
  - 另一名成员提供了进一步协助，并表示：*如果有任何问题请告诉我！*
- **Juan 发现了 mmmu notebook**：Juan 随后自己找到了 **mmmu notebook**，这为他的项目提供了所需的持。
  
  - 他感谢社区的*出色工作*，对现有资源表示赞赏。

---

### **DSPy ▷ #**[**examples**](https://discord.com/channels/1161519468141355160/1161519685616025600/1308754985961525308) (1 条消息):

> - `Semantic Router`
> - `Classification Tasks`

- **Semantic Router 作为基准**：一位成员建议将 [Semantic Router](https://github.com/aurelio-labs/semantic-router) 作为分类任务性能的基准，并强调了其**超快速 AI 决策**的能力。
  
  - 该项目专注于**多模态数据的智能处理**，它可能提供我们旨在超越的竞争性基准。
- **专注于性能提升**：有人断言现有分类工具的性能需要被超越，并以 **Semantic Router** 作为参考点。
  
  - 讨论围绕着确定指标和策略，以实现比该工具设定的基准更好的结果。

 

**提到的链接**：[GitHub - aurelio-labs/semantic-router: Superfast AI decision making and intelligent processing of multi-modal data.](https://github.com/aurelio-labs/semantic-router)：超快速 AI 决策和多模态数据的智能处理。- aurelio-labs/semantic-router

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1308839925042642965) (2 条消息):

> - `LLM-Native Resume Matching`
> - `Building AI Agents with LlamaIndex`
> - `Webinar on December 12`

- **LLM-Native 简历匹配解决方案发布**：感谢 [@ravithejads](https://twitter.com/ravithejads)，一种用于简历匹配的 **LLM-native 解决方案**已经开发完成，增强了传统的筛选方法。
  
  - 这种创新方法解决了招聘中手动筛选**缓慢且乏味的过程**，提供了一个更高效的替代方案。
- **加入我们的构建 AI Agents 网络研讨会**：在 **12 月 12 日**即将举行的网络研讨会中，学习如何使用 LlamaIndex 和 [@Redisinc](https://twitter.com/Redisinc) 构建**数据驱动的 AI Agents**。
  
  - 本次会议将涵盖构建 Agentic 系统架构以及**降低成本**和优化**延迟（latency）**的最佳实践。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1308790453147144193) (2 条消息):

> - `Extracting table data from PDFs`
> - `Applications for PDF data extraction`

- **从 PDF 中提取表格数据**：一位成员询问了从包含文本和图像等各种元素的 PDF 文件中提取**表格数据**的方法。
  
  - 他们表示有兴趣了解是否有任何现有的应用程序可以促进这一过程。
- **关于 PDF 数据提取应用的咨询**：另一位成员寻求关于可以专门从 PDF 中提取数据的任何可用应用程序的建议。
  
  - 这突显了社区内对能够处理各种复杂 PDF 工具的需求。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1308644726441906226) (4 条消息):

> - `New UI Feedback`
> - `Rate Limit Issues`
> - `Interpreter Design`
> - `Future UI Configurations`

- **新 UI 引发复杂感受**：一些用户觉得新 UI 略显繁杂，注意力引导不清晰，有人将其比作电影《异形》中的计算机。然而，其他人开始欣赏其受 UNIX 启发的设计，认为它适合 1.0 版本的功能。
- **需要 Token 和速率限制（Rate Limit）配置**：一位用户对受到 Anthropic 的速率限制表示沮丧，指出目前 Interpreter 中的错误处理会导致在超过限制时退出会话。他们强调了在未来更新中加入更好的速率限制管理的重要性。
- **对 UI 改进的建议**：有呼声要求建立一个信息更丰富的 UI，显示当前的工具、模型和工作目录，以增强可用性。用户还提倡建立一个潜在的“插件生态系统”，以便在未来版本中允许自定义功能。
- **建议分离计算工作负载**：一位成员建议将 LLM 工作负载分配在本地和云端计算之间，以优化性能。这反映了对当前 Interpreter 设计局限性的关注，该设计主要针对一次运行一个 LLM。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1308953617528656003) (1 messages):

> - `Intel AMA`
> - `Hackathon Insights`

- **明天 Intel AMA 会议**：一场与 **Intel** 合作的 **Hackathon AMA** 定于 **明天（11/21）太平洋时间下午 3 点** 举行，这是一个直接从 Intel 专家那里获取见解的机会。
  
  - 别忘了[在此观看直播](https://www.youtube.com/watch?v=_Wm5guUXt54)并设置提醒！
- **即将举行活动的提醒**：提醒 @everyone 即将举行的 Intel AMA，强调其对获取知识的重要性。
  
  - 鼓励参与者准备好问题，以充分利用本次会议。

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1308807297564282880) (2 messages):

> - `Registration Issues`
> - `Hackathon vs MOOC Registration`

- **用户面临注册困惑**：一位用户表示，在加入三个不同的群组并使用多个电子邮件地址注册后，没有收到任何电子邮件。
  
  - 他们不确定自己的注册是否成功。
- **活动类型澄清**：另一位成员要求澄清，想知道该用户是指 **hackathon** 还是 **MOOC 注册**。
  
  - 这突显了参与者对不同注册类型之间可能存在的混淆。

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1308958351589113877) (1 messages):

> - `Refact.AI`
> - `Autonomous Agents`
> - `Live Demo`
> - `Tooling`

- **Refact.AI 团队的精彩演示**：**Refact.AI** 团队成员 <@326360689453039618> 和 <@1291640853462253652> 正在主持一场现场演示，展示他们的 **autonomous agent** 和 [tooling](https://github.com/smallcloudai)。
  
  - 加入[此处的现场演示与对话](https://discord.com/events/1089876418936180786/1300459081181429810)，深入了解他们的开发进展！
- **现场活动公告**：已宣布一项活动，由 **Refact.AI** 成员讨论他们最新的技术和工具。
  
  - 鼓励参与者参与有关 **autonomous agent** 的**现场演示与对话**。

 

---

---

---

---

---

---

{% else %}

> 逐个频道的详细分解内容已因邮件篇幅而截断。
> 
> 如果您想查看完整分解，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}