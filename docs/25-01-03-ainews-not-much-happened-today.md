---
companies:
- olmo
- openai
- qwen
- cerebras-systems
- langchain
- vercel
- swaggo
- gin
- echo
date: '2025-01-04T07:58:51.225259Z'
description: '以下是该文本的中文翻译：


  **Olmo 2** 发布了一份详细的技术报告，展示了一个前沿完全开源模型在预训练、中期训练及后期训练阶段的全部细节。开源推理解决方案 **PRIME** 实现了
  **26.7% 的 pass@1**，在基准测试中超越了 **GPT-4o**。性能提升方面，**Qwen 32B (4-bit)** 在 **M4 Max**
  芯片上的生成速度超过了 **40 tokens/秒**，而 **libvips** 在图像缩放处理上比 **Pillow** 快 **25 倍**。文中还介绍了一些新工具，包括用于
  Swagger 2.0 文档生成的 **Swaggo/swag**、与 Git 兼容的版本控制系统 **Jujutsu (jj)** 以及安全工具 **Portspoof**。机器人领域的进展包括一个具有数米宽视野和更高帧率的武器检测系统。硬件基准测试对比了
  **H100** 和 **MI300x** 加速器。应用场景涵盖了利用 PRIME 进行医疗错误检测，以及集成 **LangChainAI** 和 **Vercel
  AI SDK** 的金融 AI 智能体。架构方面的见解指出，目前需要类似于 **SSM（状态空间模型）** 或 **RNN（循环神经网络）** 的技术突破。'
id: 55642e7a-1407-494d-bac5-6053afc28810
models:
- prime
- gpt-4o
- qwen-32b
original_slug: ainews-not-much-happened-today-4979
people:
- akhaliq
- jason-wei
- vikhyatk
- awnihannun
- arohan
- tom-doerr
- hendrikbgr
- jerryjliu0
- adcock-brett
- shuchaobi
- stasbekman
- reach-vb
- virattt
- andrew-n-carr
title: 今天没发生什么事。
topics:
- reasoning
- chain-of-thought
- math
- coding
- optimization
- performance
- image-processing
- software-development
- agent-frameworks
- version-control
- security
- robotics
- hardware-optimization
- medical-ai
- financial-ai
- architecture
---

<!-- buttondown-editor-mode: plaintext -->**开年平淡的一周**

> 2025年1月2日至1月3日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**217** 个频道，**2120** 条消息）。预计节省阅读时间（以 200wpm 计算）：**236 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

许多“开源 o1”模仿者引发了不少关注，但大多未能建立足够的信心，与此同时 o1 继续给人留下深刻印象。[Olmo 2 发布了他们的技术报告](https://x.com/soldni/status/1875266934943649808?s=46)（[我们之前的报道见此](https://buttondown.com/ainews/archive/ainews-olmo-2-new-sota-fully-open-model/)），作为少数仅存的前沿完全开源模型之一，该报告提供了典型的完整 {pre|mid|post}-training 细节。


![image.png](https://assets.buttondown.email/images/987e58a5-32e2-404e-9fd6-f9209e187d48.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与性能**

- **模型开发与基准测试**：[@_akhaliq](https://twitter.com/_akhaliq/status/1875039314771660989) 介绍了 **PRIME**，这是一种提升语言模型推理能力的开源解决方案，实现了 **26.7% pass@1**，超越了 **GPT-4o**。此外，[@_jasonwei](https://twitter.com/_jasonwei/status/1875268874859344349) 讨论了在评估 **Chain-of-Thought** 方法时数据集选择的重要性，强调了它们在**数学和编程**任务中的有效性。

- **优化技术**：[@vikhyatk](https://twitter.com/vikhyatk/status/1875200315966005513) 对 **libvips** 进行了基准测试，发现其在调整图像大小时比 **Pillow** 快 **25 倍**。此外，[@awnihannun](https://twitter.com/awnihannun/status/1874930431969394875) 报告称，**Qwen 32B (4-bit)** 在 **M4 Max** 上的生成速度超过 **40 toks/sec**，突显了性能的提升。

- **架构见解**：[@_arohan_](https://twitter.com/_arohan_/status/1875041433620815874) 批评了尽管计算量呈指数级增长，但架构突破却停滞不前，并建议可能需要类似于 **SSMs** 或 **RNNs** 的架构突破。

**AI 工具与框架**

- **开发工具**：[@tom_doerr](https://twitter.com/tom_doerr/status/1875080307881263387) 分享了 **Swaggo/swag**，这是一个从 **Go** 代码注释生成 **Swagger 2.0** 文档的工具，支持 **Gin** 和 **Echo** 等框架。此外，[@hendrikbgr](https://twitter.com/Hacubu/status/1875230158162174222) 宣布了 **Cerebras Systems** 与 **LangChain.js** 的集成，为 **JavaScript/TypeScript** 应用启用**流式传输 (streaming)**、**工具调用 (tool calling)** 和**结构化输出 (structured output)**。

- **Agent 框架**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1874930168739017149) 预览了即将到来的 2025 年 **Agent 架构**，重点关注**报告生成**和**客户支持**等领域的**可定制性**。

- **版本控制与安全工具**：[@tom_doerr](https://twitter.com/tom_doerr/status/1875072709568106775) 介绍了 **Jujutsu (jj)**，这是一个兼容 **Git** 的 VCS，使用 **changesets** 实现更简单的版本控制；以及 **Portspoof**，这是一种安全工具，可使所有 **TCP 端口** 看起来都处于开放状态以威慑攻击者。

**机器人与硬件**

- **机器人进展**：[@adcock_brett](https://twitter.com/adcock_brett/status/1874960476565815473) 展示了其武器检测系统的 **Gen 2** 版本，具有**数米宽的视野**和**更快的图像帧率**。此外，[@shuchaobi](https://twitter.com/shuchaobi/status/1874992397060592021) 推广了由其最新硬件设计驱动的视频语音模型。

- **硬件优化**：[@StasBekman](https://twitter.com/StasBekman/status/1874981298290430130) 添加了**高端加速器缓存大小**章节，比较了不同制造商的**缓存架构**；[@StasBekman](https://twitter.com/StasBekman/status/1874979658112086234) 分享了 **H100 vs MI300x** 的基准测试，指出**不同的用例有不同的胜出者**。

**AI 应用与用例**

- **医疗与金融应用**：[@reach_vb](https://twitter.com/reach_vb/status/1875225903346909256) 讨论了通过 **Process Reinforcement through Implicit Rewards (PRIME)** 增强临床笔记中的**医疗错误检测**。[@virattt](https://twitter.com/virattt/status/1874984637346324555) 发布了一个集成了 **LangChainAI** 和 **Vercel AI SDK** 的 **AI 金融 Agent** 生产级应用。

- **创意与教育工具**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1874933780114514166) 展示了如何使用 **Gemini** 和 **Imagen 3.0** 等工具将 **文本转换为 3D 打印物体**。[@virattt](https://twitter.com/virattt/status/1874984639754080597) 还重点介绍了 **Aguvis**，这是一个适用于多个平台的 **视觉 GUI Agent**。

- **工作流与自动化**：[@bindureddy](https://twitter.com/bindureddy/status/1875003427488772334) 详细介绍了 **Agents** 如何管理 **工作流**、**数据转换**和**可视化组件**，而 [@llama_index](https://twitter.com/llama_index/status/1875225903346909256) 则提供了在**发票处理**中构建 **Agentic Workflows** 的资源。

**行业动态与新闻**

- **公司成长与投资**：[@sophiamyang](https://twitter.com/sophiamyang/status/1875219788407980237) 庆祝在 **MistralAI** 工作满一周年，并指出团队已从 **20 人增长到 100 多人**。[@Technium1](https://twitter.com/Teknium1/status/1875261716361281699) 报道称，今年各大数据中心（Datacenters）的支出已达 **800 亿美元**。

- **监管与市场趋势**：[@tom_doerr](https://twitter.com/tom_doerr/status/1875259944481779981) 批评了**欧盟快速推进的监管政策**，[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1875282332875419878) 则谈到了对 **H-1B 签证持有者**的担忧以及**技术投资下降**的问题。

- **AI 领导力与会议**：[@swyx](https://twitter.com/swyx/status/1875253083737055299) 宣布 **AIEWF 的 AI 领导力分论坛**现已在 **YouTube** 上线，其中包括来自 **@MarkMoyou (NVIDIA)** 和 **@prathle (Neo4j)** 等领导者的见解。

**社区与个人感悟**

- **悼念与个人故事**：[@DrJimFan](https://twitter.com/DrJimFan/status/1874959979553427815) 分享了对 **Felix Hill** 的由衷悼念，对其逝世表示哀悼，并反思了 AI 社区内部面临的巨大压力。

- **生产力与学习**：[@swyx](https://twitter.com/swyx/status/1875258588635320381) 强调了**自我驱动项目**对个人成长的重要性，[@RichardMCNgo](https://twitter.com/RichardMCNgo/status/1875104827900129666) 则提倡**记录工作流程**以增强学习效果和数据可用性。

**梗与幽默**

- **轻松视角**：[@Scaling01](https://twitter.com/scaling01/status/1875151612693647714) 调侃了**架构见解的无关紧要性**，而 [@HamelHusain](https://twitter.com/HamelHusain/status/1875235369970737207) 则用多个 **🤣 表情符号**分享了幽默的反应。

- **幽默轶事**：[@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1875210891022754103) 发布了关于**弄丢电饭煲**的推文，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1875243857056821477) 对有趣的内容回复了 **"🤣🤣🤣🤣"**。

- **有趣的观察**：[@nearcyan](https://twitter.com/nearcyan/status/1875026913590386715) 幽默地对比了**推特想法的数量与质量**，[@thinkzarak](https://twitter.com/thinkzarak/status/1874950373980400128) 则分享了关于 **AI 在社会中角色**的机智见解。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1：LLM 性能飞跃催生对新 Benchmark 的需求**

- **Killed by LLM – I collected data on AI benchmarks we thought would last years** ([Score: 98, Comments: 18](https://reddit.com/r/LocalLLaMA/comments/1hs6ftc/killed_by_llm_i_collected_data_on_ai_benchmarks/)): **GPT-4** 在 **2023** 年彻底改变了 AI 基准测试，不仅超越了最先进的分数，还使其趋于饱和，标志着一个类似于通过 **Turing Test** 的重要里程碑。到 **2024** 年，其他模型如 **O1/O3** 和 **Sonnet 3.5/4o** 纷纷赶上，使数学、推理和视觉基准测试达到饱和，而 **Llama 3/Qwen 2.5** 则让开源权重模型具备了竞争力。作者主张在 **2025** 年改进基准测试，以更好地衡量现实世界的可靠性，因为当前的基准测试无法评估预期在 **2030** 年前解决的任务，并邀请大家为其 [GitHub 仓库](https://github.com/R0bk/killedbyllm) 贡献力量以进行进一步开发。
  - 评论者讨论了 **GPT-4** 和 **O1/O3** 等 AI 模型在处理复杂任务时的局限性，指出它们擅长生成初始代码或样板代码，但在集成、安全和利基问题上表现挣扎。他们强调，虽然这些模型可以提供令人印象深刻的概览和解决方案，但在处理更大、更复杂的应用时往往会失败。
  - 对话强调了编码范式转变的潜在需求，建议采用为 AI 理解而优化代码的框架。**Robk001** 和 **Gremlation** 讨论了将代码分解为小的、易于管理的块如何提高 AI 性能，后者指出高质量的输入会带来更好的 AI 输出。
  - 像 **Grouchy-Course2092** 和 **butteryspoink** 这样的用户分享了在向 AI 模型提供详细输入时生产力提高的经验。他们指出，结构化的方法（如使用 **SDS+Kanban boards**）可以显著提高 AI 生成代码的质量，这表明用户输入质量在 AI 有效性中起着至关重要作用。


- **LLM as survival knowledge base** ([Score: 83, Comments: 88](https://reddit.com/r/LocalLLaMA/comments/1hsm57o/llm_as_survival_knowledge_base/)): **Large Language Models (LLMs)** 作为动态知识库，提供针对特定场景和可用资源的即时建议，超越了书籍或电视节目等传统媒体。作者针对假设情景实验了流行的本地模型，发现它们通常有效，并寻求其他进行过类似研究并确定了“末日”场景首选模型的人的见解。
  - **电力和资源担忧**：由于高功耗，在生存场景中使用 **LLMs** 的实用性存在争议。一些人认为，小型模型（如 **7-9B**）配合便携式太阳能装置可能会有用，而另一些人则强调将宝贵资源用于可能不可靠的 AI 输出是低效的。**ForceBru** 强调了 LLM 输出的随机性，而其他人建议将 LLM 与书籍等传统资源结合使用，以获得更可靠的指导。
  - **可信度与幻觉**：包括 **Azuras33** 和 **Calcidiol** 在内的许多评论者对 LLM 的幻觉表示担忧，建议将 **Retrieval-Augmented Generation (RAG)** 与维基百科导出等落地数据源集成，以提高可靠性。**AppearanceHeavy6724** 等人讨论了多次询问同一问题以识别一致答案并降低幻觉风险的技术。
  - **模型微调与实际应用**：**Lolzinventor** 和 **benutzername1337** 讨论了针对生存特定知识微调较小模型（如 **Llama 3.2 3B**）的潜力，并指出从生存和 DIY 资源中策划数据集的重要性。**Benutzername1337** 分享了在生存旅行中使用 8B 模型的个人经验，强调了其效用以及受电力限制的局限性。


**Theme 2. Deepseek V3 Hosted on Fireworks, Privacy and Pricing**

- **Deepseek V3 托管于 Fireworks (不收集数据, $0.9/m, 25t/s)** ([Score: 119, Comments: 65](https://reddit.com/r/LocalLLaMA/comments/1hselkx/deepseek_v3_hosted_on_fireworks_no_data/)): **Deepseek V3** 现已托管在 **Fireworks** 上，与 Deepseek API 不同，它通过不收集或销售数据来提供增强的隐私保护。该模型支持完整的 **128k context size**，成本为 **$0.9/m**，运行速度为 **25t/s**；然而，其服务条款引发了隐私方面的担忧。**OpenRouter** 可以代理到 Fireworks，且正如其 [Twitter thread](https://x.com/FireworksAI_HQ/status/1874231432203337849) 中讨论的那样，目前已有支持 fine-tuning 的计划。
  - **隐私担忧与可信度**：用户对 **Fireworks** 的隐私声明表示怀疑，指出公司通常拥有广泛的服务条款，允许其广泛使用提交的内容。文中强调了对数据收集和潜在滥用的担忧，一些用户对 Fireworks 的可信度提出了质疑。
  - **性能与成本问题**：用户报告了通过 **OpenRouter** 访问 **Fireworks** 时的不满，理由是与其他替代方案相比，响应时间更慢且成本更高。有提到 **Deepseek V3** 是一个 **MoE model**，在 **671B** 参数中仅激活 **37B**，这使得它在大规模运行时更便宜，但用户仍对低廉的定价持怀疑态度。
  - **技术实现与基础设施**：讨论涉及了 **Deepseek V3** 性能所需的技术基础设施，认为其成本效益可能源于对内存的高效利用和基础设施设计。引用了 **Exolabs' blog** 以深入了解在 **Mac Minis** 等替代硬件上运行此类模型的见解。


- **Deepseek-V3 GGUF 版** ([Score: 63, Comments: 26](https://reddit.com/r/LocalLLaMA/comments/1hsort6/deepseekv3_ggufs/)): **u/fairydreaming** 和 **u/bullerwins** 已将 **DeepSeek-V3 GGUF** 量化版上传至 [Hugging Face](https://huggingface.co/bullerwins/DeepSeek-V3-GGUF/tree/main)。有人请求上传在 **512GB DDR4 RAM** 和 **单张 3090 GPU** 环境下的 t/s 表现。
  - **内存需求**：讨论强调 **q4km** 需要大约 **380 GB RAM** 加上额外的上下文空间，总计接近 **500 GB**，这使得它不适合内存较小的系统，如 **搭载 m4 芯片的 Macbook Pro**。提到 **Q2** 量化的 RAM 需求较低，为 **200 GB**，但被认为效果不佳。
  - **硬件考量**：用户正在讨论硬件升级，其中一人计划订购额外的 **256GB DDR5 RAM** 来测试该配置，而其他人则表示受限于主板限制。**bullerwins** 提供了性能基准测试，指出在其配置下使用 **Q4_K_M** 时，**prompt processing** 为 **14t/s**，**text generation** 为 **4t/s**，并提到使用了拥有 **8 个 DDR4 内存通道** 的 **EPYC 7402 CPU**。
  - **性能对比**：关于 CPU 与 **4x3090 GPU** 的性能存在争论，**bullerwins** 指出，与 GPU 相比，使用 CPU 时 prompt 处理性能损失 **28%**，推理性能损失 **12%**。GPU 只能加载 **61 层中的 7 层**，凸显了在这种情况下 GPU 显存的局限性。


**主题 3. 清华 Eurus-2：新型 RL 方法超越 Qwen2.5**

- **训练一个超越 GPT-4o 的 7B 模型？** ([Score: 74, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1hsk8h8/train_a_7b_model_that_outperforms_gpt4o/)): **清华团队**推出了 **PRIME (Process Reinforcement through Implicit Rewards)** 和 **Eurus-2**，通过仅使用 1/10 的数据，使 **7B 模型** 实现了超越 **Qwen2.5-Math-Instruct** 的高级推理能力。他们的方法通过实施 **implicit process reward modeling**（隐式过程奖励建模）来解决强化学习 (RL) 中的挑战，以应对精确且可扩展的密集奖励以及 RL 算法效率的问题。[GitHub 链接](https://github.com/PRIME-RL/PRIME)
  - **GPU 需求**：**David202023** 询问了硬件需求，寻求训练 **7B 模型** 所需硬件的详细信息，表明了对技术规范的兴趣。
  - **图片显示问题**：**tehnic** 提出了无法查看图片的问题，暗示项目资源可能存在访问性或托管问题。
  - **模型测试**：**ozzie123** 表达了下载和评估模型的计划，展示了社区的参与度以及对项目成果的实际兴趣。


**主题 4. OLMo 2.0：竞争性开源模型发布**

- **[2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656)** ([Score: 117, Comments: 29](https://reddit.com/r/LocalLLaMA/comments/1hsdrpg/2_olmo_2_furious/)): **OLMo 2** 旨在性能上超越 **Llama 3.1** 和 **Qwen 2.5**，预示着 AI 模型开发领域竞争激烈。帖子标题暗示了对速度和强度的关注，可能参考了《速度与激情》（Fast and Furious）系列电影。
  - **OLMo 2 的性能与数据策略**：**OLMo 2** 模型位于计算性能的 **Pareto frontier**（帕累托前沿），经常超越 **Llama 3.1** 和 **Qwen 2.5** 等模型。团队采用了**自下而上的数据策展策略**，通过合成数据专注于数学等特定能力，同时通过高质量的预训练数据保持通用模型能力。
  - **社区与开源参与**：**OLMo 2** 的发布因其开放性而受到赞誉，包括 **7B 和 13B 规模**在内的所有模型均可在 [Hugging Face](https://huggingface.co/allenai/OLMo-2-1124-7B) 上获取。社区赞赏该项目的**完全开源性质**，认可其透明的训练数据、代码和方案（recipes）。
  - **未来发展与社区互动**：OLMo 团队正积极与社区互动，讨论潜在的更大模型（32B 或 70B），并正在进行将 **Molmo recipe** 应用于 **OLMo 2** 权重的实验。团队还在 [Hugging Face](https://huggingface.co/collections/allenai/pixmo-674746ea613028006285687b) 上分享了 Molmo 的训练数据链接。


## Other AI Subreddit Recap

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. 视频生成工具对比：Sora vs Veo2 vs Minimax**

- **[Pov trying to use the $200 version of Sora...](https://v.redd.it/cr0zeu3ihnae1)** ([Score: 164, Comments: 47](https://reddit.com/r/OpenAI/comments/1hs5bzr/pov_trying_to_use_the_200_version_of_sora/)): 该帖子缺乏具体内容或上下文来进行详细总结，因为标题仅提到了 **200 美元版本的 Sora**，未做进一步阐述。
  - 讨论重点关注 AI 模型中的**内容过滤**问题，用户对非性内容触发的**政策违规**表示沮丧。一些人认为过滤系统过于严格，特别是在描绘女性方面，并质疑当前内容审核方法的有效性。
  - 用户讨论了 **hailuoai.video** 和 **minimax** 等**视频生成替代方案**，并比较了它们的功能和效果。**Veo2** 因其卓越的效果而受到关注，尽管由于候补名单限制，访问权限有限；**hunyuan video** 也被提及为一个强有力的竞争对手。
  - 用户讨论了 AI 训练中的**版权悖论**，指出允许模型使用受版权保护的材料进行训练，却限制生成与其相似的输出，这存在不一致性。此外，还提出了对内容拒绝率高以及为了避免负面宣传而可能增加限制的担忧。


**主题 2. GPT-4o：优于 GPT-3.5 的高级推理能力**

- **[Clear example of GPT-4o showing actual reasoning and self-awareness. GPT-3.5 could not do this](https://www.reddit.com/gallery/1hs5ffs)** ([Score: 112, Comments: 74](https://reddit.com/r/OpenAI/comments/1hs5ffs/clear_example_of_gpt4o_showing_actual_reasoning/)): 该帖子讨论了 **GPT-4o** 在高级推理和自我意识方面的能力，指出这些功能是对 **GPT-3.5** 的改进。帖子正文未提供具体示例或上下文。
  - 讨论指出，**GPT-4o** 识别和解释模式的能力并不代表推理，而是增强了的**模式识别**。评论者强调，模型识别“HELLO”等模式的能力归功于其训练和 **tokenization** 过程，而非任何形式的自我意识或推理。
  - 包括 **Roquentin** 和 **BarniclesBarn** 在内的几位评论者解释说，模型的表现归功于 **tokenization** 和 **embeddings**，这使其能够在没有明确指令的情况下识别模式。这与模型基于先前上下文预测下一个 token 的设计相一致，而不是展示真正的推理或内省。
  - 对话还涉及了将 **“HELLO”模式作为测试的局限性**，建议使用不明显的模式可以更好地展示推理能力。**ThreeKiloZero** 等人认为，模型庞大的训练数据集和多参数结构使其能够匹配模式而非推理，这表明上下文和训练数据在其回答中具有重要性。


---

# AI Discord 摘要回顾

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. 性能风云与降速风波**

- [**DeepSeek 性能下滑**](https://status.deepseek.com/)：用户抱怨 DeepSeek v3 的 TPS 降至 0.6，引发了对服务器扩容的呼声。他们密切关注状态页面寻找缓解迹象，但许多人仍渴望更快的模型。
- [**Windsurf 与 Cascade 额度冲突**](https://codeium.com/plan)：用户发现 Windsurf 中的内部错误消耗了额度，导致混乱。尽管有自动化保证，扣费仍在继续，引发了大量关于退款的沮丧帖子。
- **ComfyUI 幕后机制**：SwarmUI 运行在 ComfyUI 的后端以实现用户友好的渲染，而像 Omnigen 或 SANA 这样的竞争方案则相对落后。粉丝们称赞 LTXVideo 和 HunyuanVideo 能以极小的质量损失实现极速的视频生成。

**主题 2. 额度紧缩与成本困惑**

- [**60 万美元模型训练的账单冲击**](https://news.ycombinator.com/item?id=39224534)：工程师们分享了训练大模型时令人咋舌的 GPU 账单，其中 7B 参数模型的成本约为 8.5 万美元。他们讨论了更便宜的托管方案，如 [RunPod](https://runpod.io) 以及来自 [LoQT 论文](https://arxiv.org/abs/2405.16528) 的低秩适配器。
- [**支付问题困扰 API 用户**](https://openrouter.ai/api/v1)：OpenRouter 的信用卡被拒以及 Perplexity 的订阅异常引发了混乱。一些人通过更换卡片或清除缓存解决了问题，但烦恼情绪仍在蔓延。
- **Flex 与 Premium 额度**：多个社区抨击了使用上限和额度无法结转的问题。为未使用的 Token 付费或处理“内部错误”会话，促使人们呼吁更透明的方案。

**主题 3. 模型首秀与微调热潮**

- [**Sonus-1 大放异彩**](https://sonus.ai/blog/sonus-1)：其 Mini、Air、Pro 和 Reasoning 变体因 2025 年先进的文本生成能力引发热议。一则 [推文](https://x.com/RubiksAI/status/1874682159379972325) 展示了结合 Aider 与 Sonus mini 模型实现的快速代码输出。
- [**Swarm 库加入 NPM**](https://www.npmjs.com/package/agentswarm)：这个 TypeScript 多 Agent AI 库宣称拥有超越 OpenAI Swarm 的协同效应，因其模块化设计赢得赞誉。其他人则将希望寄托在 ICML 2025 的 PersonaNLP 上，专注于基于人格的 NLP 任务。
- **Qwen2-VL 与 Llama 3.1 引发微调热**：社区正在努力解决视觉适配器部分缺失或失效的问题，而 Llama 3.x 和 SmallThinker-3B 性能飙升。人们还在使用 Unsloth、Axolotl 或 Hugging Face Transformers/PEFT 进行自定义任务微调。

**主题 4. 工具的突破与摩擦**

- [**GraphRAG 与 Graphrag 占据头条**](https://github.com/microsoft/)：新的检索增强生成策略激发了人们对代码和文本任务的兴趣。讨论涵盖了多检索器设置和加权向量以改进查询结果。
- [**发票 Agent 与 K-Summary 测试版**](https://twitter.com/llama_index/status/1875225903346909256)：LlamaIndex 展示了自动对发票进行支出类别和成本中心分类的功能，而韩国市场的新摘要工具也吸引了测试者。用户对 Torchtune 中的分块交叉熵和内存友好方法赞不绝口。
- **AI 编程热潮**：Codeium、Cursor 和 Aider 社区在应对随机代码更改、Linting 狂热和受限的套餐等级。尽管感到沮丧，许多人仍称赞更快的开发周期和更一致的代码建议。

**主题 5. 硬件、VRAM 与 HPC 探索**

- [**RTX 50xx VRAM 争议**](https://discord.com/channels/729741769192767510) 让工程师们怀疑 NVIDIA 人为限制了显存。他们思考更大的 VRAM 还是巧妙的内存卸载（offload）方案才是真正的出路。
- [**Torch.compile 令人头疼**](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html)：用户在 Inductor 缓存、Flash Attention 中的动态分支以及棘手的 Triton 内核调用方面遇到了严重的降速。他们测试了环境黑客手段以规避段错误（segfaults），希望官方补丁能解决编译混乱。
- [**1-bit LLM 热潮**](https://arxiv.org/abs/2402.17764)：BitNet 关于三值权重和大幅削减资源的讨论令 HPC 爱好者感到兴奋。一些人赌这些低比特突破将在不牺牲模型准确性的情况下大幅削减训练账单。

---

# 第一部分：高层级 Discord 摘要

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 的动荡困扰**：许多成员报告了 **Windsurf** 的**重大性能问题**，理由是频繁的内部错误和运行缓慢，并引用了[这条关于将 Figma 设计变为现实的推文](https://x.com/windsurf_ai/status/1874948790928687194)。
   - 他们还观察到 **Claude 3.5 Sonnet** 在会话中途性能下降，尽管 [Plan Settings](https://codeium.com/plan) 中有官方免责声明，但仍导致了意外的额度消耗。
- **Cascade 额度争议**：社区讨论集中在 **Cascade** 甚至在操作失败时也会扣除额度，重复的“内部错误”消息导致了困惑。
   - 几位用户声称，尽管有自动保证，这些扣费仍然存在，促使一些人通过 [Support | Windsurf Editor and Codeium extensions](https://codeium.com/support) 进行申诉。
- **DeepSeek v3 与 Sonnet 3.6 的对决**：尽管基准测试声称如此，一些人认为 **DeepSeek v3** 不如 **Sonnet 3.6**，更倾向于使用 **Gemini** 等免费替代方案。
   - 他们对 DeepSeek 的真实优势表示怀疑，而其他人则引用 [Things we learned about LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/) 以获取更多数据。
- **Windsurf 中的代码编辑混乱**：用户提到了随机的代码更改和未完成的任务，要求提供更清晰的解决方案以保持 AI 工作流的连续性。
   - 许多人诉诸于将指令保存在外部文件中，然后重新加载它们以保持对话不偏离轨道。
- **额度系统怨言**：成员批评了 **Premium** 和 **Flex** 额度结构，抱怨使用上限和未能结转的问题。
   - 他们敦促建立更公平的分配模型，据报道通过电子邮件和 [Support | Windsurf Editor and Codeium extensions](https://codeium.com/support) 取得的效果参差不齐。



---



## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Sonus-1 登台亮相**：新推出的 **Sonus-1 系列**（Mini, Air, Pro, Reasoning）在[一篇博客文章](https://sonus.ai/blog/sonus-1)中发布，重点关注 2025 年的高级文本生成能力。
   - [Rubik's AI 的一条推文](https://x.com/RubiksAI/status/1874682159379972325)强调了 mini 模型中快速的代码生成，引发了关于与 Aider 协同作用的讨论。
- **Deepseek 在重载下步履蹒跚**：社区成员观察到 **Deepseek** 降至 **1.2 TPS**，引发了对服务器容量和可靠性的投诉。
   - 其他人证实 [Deepseek Chat v3](https://openrouter.ai/deepseek/deepseek-chat-v3) 仍可通过 `--model openrouter/deepseek/deepseek-chat` 访问，但质疑是否需要更多服务器。
- **OpenRouter 的 API Key 困惑**：一些人在使用 OpenRouter API 时遇到了**身份验证难题**，怀疑配置文件中的密钥放置不正确。
   - 一位用户通过仔细检查模型设置确认成功，建议社区注意 YAML 中隐藏的空格。
- **Tailwind 和 Graphrag 备受关注**：成员们探索将 Tailwind CSS 文档上下文添加到 Aider 中，建议复制或索引相关信息以便快速参考。
   - 微软的 **Graphrag** 工具也作为 RAG 替代方案出现，激发了对更高效 CLI 实现的兴趣。
- **Aider 愿望清单扩大**：用户请求切换类定义和先验上下文以优化代码编辑，旨在减少无关建议。
   - 他们还设想了对命令提示符更好的控制，将高级上下文管理视为首要的下一步。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **OpenWebUI 导出扩展了数据集视野**：成员们讨论了从 **OpenWebUI** 导出聊天 JSON 以创建数据集，并参考了所有者的格式建议。
   - 他们强调了与 [vLLM](https://docs.vllm.ai/en/latest/) 等本地推理设置的潜在协同效应，指出将结构良好的数据与先进的推理相结合可以改善训练结果。
- **Ollama 量化困惑**：围绕 **Ollama** 的模型量化出现了挑战，用户注意到默认的 GGUF 文件以 FP16 运行。
   - 与会者建议进行手动调整，并指向 [mesosan/lora_model config](https://ollama.com/mesosan/lora_model) 以寻求潜在解决方案。
- **分类任务的微调热潮**：社区成员推荐将 **Llama 3.1 8B** 和 **Llama 3.2 3B** 用于中等复杂度的任务，并称其在分类任务中表现良好。
   - 他们强调使用 **RTX 4090** 等 GPU 硬件，并指出 [Unsloth 的文档](https://docs.unsloth.ai/get-started/beginner-start-here) 中关于高效微调的技巧。
- **复旦大学专注于 O1 复现**：最近的一份 **复旦报告** 深入报道了 **O1 复现工作**，详见[这篇论文](https://arxiv.org/pdf/2412.14135)。
   - 一位成员称赞其为迄今为止最详尽的资源，引发了对 O1 项目后续步骤的兴趣。
- **Process Reinforcement 暗示代码发布**：**Process Reinforcement** 论文因其关于隐式奖励的思想而受到关注，尽管许多人对缺乏代码表示遗憾。
   - 社区成员对代码即将发布保持乐观，将其描述为值得关注的 *进行中工作 (work in progress)*。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek V3 模型的隐私谜题**：成员们对 **DeepSeek V3** 可能存储代码并在私有数据上进行训练表示担忧，强调了隐私问题以及在用户项目中的不确定收益。
   - 他们质疑个人和企业的风险，辩论该模型的优势是否足以证明其可能的数据保留方式是合理的。
- **Cursor 大幅缩减开发时间**：一位用户分享了一个成功的案例，使用 **Cursor** 配合 **SignalR** 在远短于预期的时间内完成了一个项目。
   - 其他人也纷纷给出正面反馈，指出 AI 驱动的工具如何帮助他们更自信地应对复杂的开发任务。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Swarm 入侵 NPM**：新发布的用于 **multi-agent AI** 的 [Swarm library](https://www.npmjs.com/package/agentswarm) 已上线 NPM，提供了超越 **OpenAI Swarm** 的协作系统高级模式。它采用 TypeScript 构建，具有模型无关 (model-agnostic) 的设计，且在一天前刚刚更新。
   - 社区成员赞扬了其模块化结构，称其为迈向 **灵活多智能体 (flexible multi-agent)** 协同的大胆一步，在性能上可能超越旧框架。
- **PersonaNLP 为 ICML 2025 做准备**：计划中的 ICML 2025 **PersonaNLP** 工作坊正在征集论文和共享任务，重点关注语言建模中以用户为中心的方法。组织者正公开与有兴趣改进基于角色的 NLP 方法的研究人员进行协调。
   - 参与者建议开设专门频道进行更深入的协作，并表达了增强工作坊范围的热情。
- **巨型模型成本飙升**：最近的讨论显示，**模型训练** 账单已达到 **$600,000**，[Hacker News 帖子](https://news.ycombinator.com/item?id=39224534)和 [Moin Nadeem 的推文](https://x.com/moinnadeem/status/1681371166999707648)也强调了这一点。成员们指出，仅一个 7B 模型在商业 GPU 上的成本就约为 **$85,000**。
   - 一些工程师指向了 [RunPod](https://runpod.io) 等服务以获取更便宜的配置，并探讨了 [LoQT 论文](https://arxiv.org/abs/2405.16528) 中的 **低秩适配器 (low-rank adapters)** 是否能减少支出。
- **Hermes 数据困境**：社区成员发现 Hermes 中**没有针对某些成人场景的显式训练数据**，推测这种遗漏可能会限制模型的更广泛能力。他们质疑缺乏此类数据是否会限制知识的广度。
   - 有人声称跳过这些数据点会移除潜在的关键细微差别，而其他人则认为这是为了获得更简单的模型输出而进行的合理折中。
- **Llama 权重分析**：分析师在 **Llama2** 的 **K** 和 **Q** 权重中发现了意想不到的振幅模式，暗示 Token 的重要性不一致。他们分享的图像表明权重在表示关键特征时存在部分冗余。
   - 成员们讨论了专门的微调或 Token 级门控 (token-level gating) 作为可能的补救措施，强调了改进 Llama2 架构的新角度。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Extractor.io 的消失之谜**：一位用户发现 **Extractor.io** 似乎已不复存在，尽管有[这个精选 LLM 列表](https://llm.extractum.io/list/)作为替代方案，但仍引发了困惑。
   - 其他人对这种突然消失表示质疑，一些人认为它可能合并到了不同的域名或进行了品牌重塑。
- **LM Studio 仅限文本**：社区成员确认 **LM Studio** 专注于大语言模型（LLM），无法生成图像。
   - 他们建议使用 **Pixtral** 处理图片任务，并指出它依赖于 **MLX Engine**，且仅在特定硬件上运行。
- **没有视觉能力的 Qwen2-VL**：爱好者们观察到 **Qwen2-VL-7B-Instruct-abliterated** 由于缺少 vision adapter（视觉适配器），无法处理图像。
   - 他们强调，对基础模型进行 **proper quantization**（正确量化）对于充分发挥其文本能力的优势至关重要。
- **用整个互联网进行训练？**：一位用户提出了将所有互联网数据喂给 AI 的想法，但许多人指出，庞大的规模和低劣的数据质量是潜在的陷阱。
   - 他们强调 **bad data**（坏数据）会损害性能，因此 **quality**（质量）必须胜过单纯的数量。
- **通过 GPU Offload 加速任务生成**：本地 LLM 粉丝利用 **GPU** 支持来加速使用 **Llama 3.1** 生成任务的过程，并获得了更好的响应效果。
   - 他们建议选择启用 GPU 的模型并观察 **Task Manager**（任务管理器）指标，并提到了在 **4070 Ti Super** 配置下的成功案例。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **AI 中的多语言尝试**：爱好者们通过编写创意 Prompt，在非英语环境下测试了 **audio overviews**（音频概览），在语言扩展方面取得了部分成功。
   - 他们报告称翻译质量参差不齐，并建议使用更好的 **language-specific models**（特定语言模型）来解决这些差距。
- **K-Summary Beta 引起关注**：一位用户推广了一款在韩国市场迅速崛起的全新 **AI summarization**（AI 摘要）产品，为 Beta 测试人员提供了体验精简摘要的机会。
   - 几位社区成员表示渴望将其与现有的摘要工具进行对比，以实现更快的文本处理。
- **自定义功能引发辩论**：成员们担心 **adjusting system prompts**（调整系统提示词）可能会暴露绕过常规 AI 限制的方法。
   - 他们辩论了创作自由与 AI 安全使用之间的界限，权衡了潜在收益与滥用风险。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 为 Web 应用引入 AI 逻辑**：一位成员分享了在他们的 Bolt Web 应用中集成 **logic/AI** 的计划，在赞扬视觉效果的同时表示需要更强的功能，并参考了 [BoltStudio.ai](https://boltstudio.ai/)。
   - 他们征求了将代码驱动的工作流与 AI 模块合并的策略，并提到增量升级和本地测试是可行的前进方向。
- **Supabase 简化 Bolt 中的邮箱登录**：开发者们称赞了 **Supabase** 邮箱身份验证在基于 Bolt 的应用中的表现，强调了使用 local storage 来管理用户角色。
   - 他们指向 [StackBlitz Labs](https://github.com/stackblitz-labs) 以实现前端与灵活后端的桥接，同时也承认关于 Bolt 的 token 使用情况仍存在争议。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 对阵 ChatGPT：Perplexity 的强力搜索**：**Perplexity O1** 引发了褒贬不一的反应，一些人抱怨**每日 10 次搜索**的限制并称其为麻烦，而另一些人则认为它在以搜索为中心的任务中大有可为。
   - 与 **ChatGPT** 的对比中，**Opus** 因其无限使用和超长上下文而受到称赞，如[这条推文](https://x.com/pplxsupply/status/1875019712268263658)所述。
- **Grok 的收获或抱怨**：尽管成本较低，一些人仍称 **Grok** 为他们用过的“最差模型”，引发了关于模型可靠性的辩论。
   - 另一些人则吹捧 **3.5 Sonnet model** 表现更强，暗示用户忠诚度正在发生转移。
- **Perplexity 的 UI 大改与订阅**：最近的 **UI 更改**增加了股票和天气信息，促使一位用户清理缓存以避开恼人的首页元素。
   - 成员们在 [AravSrinivas 的推文](https://x.com/aravsrinivas/status/1874943854849425780)中讨论了无限查询和省钱方案，展示了多样化的订阅选择。
- **令人印象深刻的 2025 年 AI 面试题**：一份分享的指南概述了应对 **2025** 年棘手 **AI** 问题的方法，详情见[此链接](https://www.perplexity.ai/search/the-job-interview-question-of-geScATofQC.NYw5MqWsyiA)。
   - 参与者认为，充分的准备对于在招聘环境中保持竞争力至关重要。
- **基于欧洲的 API 和 1.5B-Token 聊天机器人的愿景**：一位用户期待与 **pro search** 速度相匹配的**欧洲服务器**，以支持性能更佳的 **1.5B-token** 聊天机器人。
   - 他们认为这种集成将增强聊天机器人的实用性，特别是对于大规模 token 的使用。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 的身份验证困扰**：多位用户在尝试请求时遇到了 **'Unauthorized'** 错误，即使在 [OpenRouter](https://openrouter.ai/api/v1) 账户有余额且 API key 正确的情况下也是如此。他们报告称更改了 HTTPS 地址并调整了证书，但仍未解决。
   - 一些人推测问题可能涉及 **n8n** 配置不匹配或连接问题，并指出手动调整 URL 设置仍然失败。
- **DeepSeek 令人恐惧的拖累**：社区成员抱怨 [DeepSeek v3](https://app.hyperbolic.xyz/models/deepseek-v3) 的速度仅为 **0.6 TPS**，导致响应缓慢。[DeepSeek 服务状态页面](https://status.deepseek.com/)显示需求量巨大且可能存在扩容不足。
   - 他们担心使用量已超出目前的预测，呼吁在更广泛推广之前提升容量。
- **结构化输出寻求救星**：一位用户希望寻找 **gpt-4-mini** 的替代方案来处理 JSON 格式的回复，但在目前的阵容中发现选择有限。其他人建议使用 **Gemini Flash**，并指向 [LiteLLM](https://github.com/BerriAI/litellm) 以在统一界面中处理多个 API。
   - 他们注意到了潜在的速率限制约束，并建议监控使用指标，同时参考了 [RouteLLM](https://github.com/lm-sys/RouteLLM) 作为跨模型路由请求的另一种解决方案。
- **Janitor AI 加入 OpenRouter**：成员们讨论了如何将 Janitor AI 与 **OpenRouter** 关联，重点关注 API 端点的高级设置。他们概述了切换某些身份验证字段并匹配代理 URL 以实现协同使用。
   - 分享了各种配置，结论是正确的 URL 对齐和 token 处理使集成变得无缝。
- **信用卡被拒，付款失败**：一些用户在尝试通过 OpenRouter 付款时发现**信用卡支付失败**，尽管不同的卡有时可以正常工作。一位用户指出 Capital One 卡存在持续问题，而第二张卡则处理成功。
   - 他们考虑了特定银行的规则或 OpenRouter 支付网关的特性，建议受影响的用户尝试多种计费方式。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SwarmUI 凭借 ComfyUI 飞速发展**：成员们解释了 **SwarmUI** 如何利用 **ComfyUI** 的后端在保持相同性能的同时提供更简洁的 UI，强调了其易用性和强大的功能。
   - 他们还重点介绍了一个 [Stable Diffusion Webui Extension](https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper)，该扩展可以简化模型管理，引发了关于提升工作流效率的前端讨论。
- **SANA 与 Omnigen 的空间对决**：社区测试了 **SANA** 在小型硬件上的快速推理能力，并将其与速度较慢且图像质量有时逊于 **SDXL** 的 **Omnigen** 进行了对比。
   - 爱好者们质疑 SANA 是否值得占用 HDD 空间，尤其是当 **Flux** 可能提供更好的模型性能时。
- **LTXVideo 与 HunyuanVideo 全速前进**：**LTXVideo** 因在新 GPU 上渲染速度更快且质量几乎无损而获得赞誉，超越了旧的视频流水线。
   - 与此同时，**HunyuanVideo** 引入了更快的步数（steps）和更好的压缩技术，激发了人们对近期视频生成领域进展的热情。
- **Flux Dev 在图像生成文本需求中表现出色**：成员们认为 **Flux Dev** 是在图像中嵌入文本的顶级开源模型，足以与 **Ideogramv2** 和 **DALL-E 3** 等闭源方案竞争。
   - 他们还提到 **Flux 1.1 Ultra** 是输出清晰文本的“最佳闭源模型”，并引用了用户测试和并排对比结果。
- **GPU 收益与内存必备要求**：爱好者们建议在 AI 任务中使用 **RTX-series** 显卡，并建议等待可能进一步降价的新品发布。
   - 他们强调至少需要 **32GB RAM** 和充足的 **VRAM** 才能保证图像生成的流畅性，并突出了稳定性方面的优势。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **RTX 50xx VRAM 限制引发热议**：工程师们讨论了 **RTX 50xx** 系列中传闻的 VRAM 限制，怀疑 **NVIDIA** 人为限制显存以避免产品线重叠。
   - 一些人质疑增加的 GB 容量对 AI 任务是否重要，表达了对潜在性能瓶颈的沮丧。
- **VRAM vs. RAM：内存大混战**：多位参与者主张将 **VRAM** 重新分类为 **L3 Cache**，并指出在某些场景下，普通 RAM 的速度可能比 VRAM 慢 4 倍。
   - 其他人思考了在 VRAM 和 RAM 之间进行指令流水线化的可能性，并警告称任何不匹配都可能阻碍大规模模型推理的吞吐量。
- **高阶 Attention 带来新变革**：研究人员探索了 **attention on attention** 技术，参考了 [Quartic Transformer](https://github.com/lucidrains/quartic-transformer) 中的扩展，并将其与 **Mamba** 或 **SSM** 风格的卷积联系起来。
   - 他们将这些想法与 **ring attention** 联系在一起，引用了第二篇论文中更大的上下文窗口，并指出了可能的线图（line-graph）或超图（hypergraph）平行关系。
- **HYMBA 阻击 SWA**：社区成员认为 **HYMBA** 在某些层混合全注意力（full attention）可能会削弱 **SWA** 或 SSM 背后的效率收益。
   - 他们权衡了更强大的跨窗口表示与额外开销之间的利弊，指出实际的性能提升仍需进一步测试。
- **Pytorch Flex Attention Bug 依然存在**：一些用户报告了 **Pytorch** 的 **Flex Attention** 持续存在问题，这阻碍了实现复杂注意力模式的尝试。
   - 他们发现 `torch.compile` 经常与一些较少使用的模型特性发生冲突，迫使他们在修复方案出台前退回到标准的 attention 层。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **2024 LLMs & Image Generation Gains**：即将发布的文章 [LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/) 聚焦了 **Large Language Models** 在 2024 年的重大飞跃，包括 **multimodal**（多模态）扩展和激烈的价格竞争。社区还注意到生成式图像带来的 **meme culture**（迷因文化）趋势，这些图像将普通人变成了喜剧性的“bro”，并赋予了 **Santa**（圣诞老人）一个严肃的形象。
   - 成员们认为这些跨领域的突破促进了更广泛的创意应用，并强调了新指标如何推动 LLM 性能边界。他们还观察到成本降低和易于访问的 API 加速了小型项目的采用。
- **Text Extraction Throwdown**：一项 [基准研究](https://cdn.discordapp.com/attachments/1075282825051385876/1324793970726928537/A_Comparative_Benchmarking_Evaluation_of_Text_Extraction_Tools.pdf) 测试了使用各种库解析 **regulatory document**（监管文档）的效果。贡献者们特别指出 **pdfium** + **tesseract** 组合在处理棘手的数据提取任务中表现出色。
   - 他们强调，这些解决方案比单机 OCR 或 PDF 解析工具能更好地处理现实世界的复杂性。一些人认为工作流集成是构建稳健文本流水线的下一个重要步骤。
- **SmallThinker-3B's Surging Stats**：[Hugging Face](https://huggingface.co/PowerInfer/SmallThinker-3B-Preview) 上新的 **SmallThinker-3B-preview** 在多项评估中超越了 **Qwen2.5-3B-Instruct**。这款紧凑型模型针对资源受限的场景，但在基准测试分数上显示出显著飞跃。
   - 它对边缘友好型占用的强调拓宽了实际应用场景，在较小的占用空间与强大的性能之间架起了桥梁。一些参与者怀疑这些改进源于专门的微调和数据策化。
- **OLMo2's Outstanding Outline**：[OLMo2 技术报告](https://x.com/soldni/status/1875266934943649808) 长达 50 多页，详细介绍了 **LLM development** 流水线中的四个关键组件。它对数据处理、模型架构、评估和部署策略进行了深入分解。
   - 读者称赞其直接揭示现实世界经验的做法，突出了可重复和可扩展训练的最佳实践。该报告鼓励开发者以更深的技术清晰度来优化现有工作流。
- **Summit & Transformers Tactics**：仅限受邀参加的 [AI Engineer Summit](https://www.latent.space/p/2025-summit) 在 **10:1** 的申请比例下回归，旨在展示 **AI engineering** 的新突破。组织者回顾了 **3000 座** 世博会规模的成功，吸引了超过 **100 万** 在线观看。
   - 与此同时，来自 [此资源](https://x.com/sannykimchi/status/1176517584319127553) 的 **Understanding Transformers** 综述为学习 **self-attention** 和现代架构变体提供了一条结构化路径。峰会策划者鼓励为推动社区知识进步的高级讲解内容提供客座发布机会。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU GEMM Gains**：一位用户指出 GPU 上实际与理论 **GEMM** 性能之间存在差异，引用了 [Notion](https://yywangcs.notion.site/Inconsistency-in-GEMM-Performance-16efc9f5d80580838090dded05493014) 上的文章和相关的 [Twitter 帖子](https://x.com/YyWangCS17122/status/1874856334845489191)。
   - 他们暗示，宣布“最佳性能”可能会激发社区提供更先进的解决方案。
- **Triton Tuning Troubles**：据报道，移除 **TRITON_INTERPRET** 可以大幅提升 Triton kernel 的性能，尤其是在矩阵乘法任务中。
   - 其他人证实，将 batch size 设置为 **16** 或更多，并针对大输入调整浮点容差，可以缓解 kernel 调用问题。
- **The Flash br/bc Dilemma**：一位用户询问在 Flash Attention 中使用动态 **br/bc** 以获得更好的适应性，但其他人坚持认为固定尺寸要“快 10 万亿倍”。
   - 他们建议像 **Flash Attention** 那样编译多个版本，旨在平衡速度与更灵活的参数。
- **Torch Inductor Cache Letdown**：一次讨论涉及 **Inductor** 缓存加载时间过长的问题，即使使用基于 Redis 的远程缓存，加载时间也达到了 **5 分钟**。
   - 他们怀疑编译后的 kernel 加载仍然会导致延迟，这促使人们对内存使用和激活需求进行额外审查。
- **P-1 AI’s Radical AGI Push**：P-1 AI 正在招聘 **artificial general engineering**（通用人工智能工程）计划的人员，[开放职位见此](https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6)。
   - 他们的核心团队成员来自前 DeepMind、Microsoft、DARPA 等，旨在利用 **multimodal LLMs** 和 GNN 增强物理系统设计，处理曾经被认为不可行的任务。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LoRA 库之争：TRL 胜过 Unsloth**：#ml-questions 频道对 LLM 微调工具的优劣展开了辩论，称赞 **TRL** 详尽的文档，同时认为 **Unsloth** 过于难用，并引用了 [LLM 微调库对比](https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit)。
   - 尽管 **Unsloth** 拥有 2 万个 GitHub Star，但成员们更推荐使用 Hugging Face 的 **Transformers/PEFT**，以及 **Axolotl** 和 **Llama Factory** 来进行更简单的 LoRA 微调。
- **门控游戏：MoE 与 OLMoE**：#ml-questions 的成员询问了关于 Mixture of Experts 的门控网络，特别是 **Deepseek v3** 中使用的路由机制。
   - 一位用户推荐了 **OLMoE** 论文，强调其较少的专家数量使复杂性保持在可控范围内。
- **2 万人规模的预录教程之争**：在 #random 频道，社区讨论了是否应向 2 万名观众分享预录教程，并强调这些演讲已获得好评。
   - 另一位用户开玩笑地将 **UK AI Safety Institute** 称为“情报特工”，而其他人则注意到 **LinkedIn** 在 AI 圈子中的竞争性。
- **苦涩的教训与 Felix 的遗产**：在 #reads 频道，成员们哀悼 **Felix** 的逝世，他是 [The Bittersweet Lesson](https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing) 的作者，给社区留下了深刻印象。
   - 他们讨论了通过 PDF 备份来保护他的作品，担心注销的 Google 账号可能会中断未来的访问。
- **SnailBot 的迟缓恶作剧**：在 #posts 频道，**SnailBot** 引发了笑声，用户称其为“一只慢吞吞的蜗牛”，并对其性能惊呼“天哪”。
   - 它喜剧般的节奏娱乐了许多人，大家觉得它那蜗牛般的一致性名副其实。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **FLUX.1 [dev] 助力新鲜图像生成**：成员们重点介绍了 **FLUX.1 [dev]**，这是一个用于文本到图像合成的 **12B 参数 rectified flow transformer**，引用了 [Black Forest Labs 的公告](https://blackforestlabs.ai/announcing-black-forest-labs/)。
   - 他们指出其质量仅次于 **FLUX.1 [pro]**，并包含用于科学研究的 **open weights**（开放权重），反映出社区对实验的热情。
- **ChatGPT 搜索可靠性备受关注**：一位用户询问 **ChatGPT** 是否能处理实时网络结果，并将其与 **Perplexity** 等专业工具进行了对比。
   - 社区反馈表明，该模型可能会受到数据更新的**限制**，一些人更倾向于使用外部搜索解决方案来弥补**信息缺失**。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 闪亮发布 Rerank-3.5**：成员们以热情的祝福迎接新的一年，期待 **rerank-3.5** 很快在 Azure 上部署，为高级文本处理提供下一代排序器。
   - 对话包括了对潜在用例的询问，有人问 *“到目前为止你觉得它怎么样？”*，突显了社区对提升性能的**见解**的渴望。
- **Embedding 速率限制提升与最佳实践**：用户探索了通过联系 [Cohere support](mailto:support@cohere.com) 来申请提高 **embedding rate limits** 的流程，旨在处理更繁重的工作负载。
   - 社区成员概述了现有的 **API 限制**：试用版每分钟 100 次请求，生产版每分钟 2,000 次请求，强调应高效使用以避免超限。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 势头强劲**：社区成员称赞 **Torchtune** 在多个 AI 模型中得到了更广泛的应用，并强调了一种衡量性能的方法。
   - 一位用户建议探索 [transformer module](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transformer.py#L482) 的替代评估方法，以获得更好的见解。
- **Chunked Cross Entropy 提升内存效率**：**Chunked cross entropy** 通过拆分计算来减少内存使用，如 [PR #1390](https://github.com/pytorch/torchtune/pull/1390) 所示。
   - 其中一个变体使用了 **log_softmax** 而不是 `F.cross_entropy`，引发了关于性能和内存优化的讨论。
- **A6000 上的 Flex Attention 寻求变通方案**：成员们在 **A6000** 上遇到了 **PyTorch Torchtune** 的 bug，发现通过 `torch.compile()` 设置 `flex_attention_compiled` 可以实现内核变通。
   - 他们提议使用环境变量的方法，并警告说 **2.6.0** 版本中是否会有永久修复尚不确定，参考 [Issue #2218](https://github.com/pytorch/torchtune/issues/2218)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **使用 LlamaParse 的发票智能**：最近的一个 [Notebook 演示](https://twitter.com/llama_index/status/1875225903346909256)展示了一个自定义的**发票处理 Agent**，它利用 LlamaParse 实现流畅的工作流，自动对**支出类别**和**成本中心**进行分类。
   - 成员们强调了**自动化**在减少人工错误和加速财务相关任务方面的作用，并参考了该 Agent 更高效地处理发票流水线的方法。
- **数据存储的简单 JSON 方案**：社区成员讨论了将 LLM 评估数据集存储在 **S3**、**Git LFS** 或本地 **JSON** 文件中的方案，强调了极低的开销和简单的结构。
   - 他们建议对大型 JSON 数据进行压缩，并推荐使用 **Pydantic** 进行快速集成，同时指出选择 **SQL** 还是 **NoSQL** 取决于数据集的大小。
- **多检索器融合难题**：一位用户将 **2 个向量嵌入（vector embedding）检索器**与 **2 个 BM25** 检索器结合使用，但报告称这种融合设置的查询结果较差。
   - 讨论指向了通过调整权重、索引或重排序（re-ranking）策略来提升混合检索方法**响应质量**的方向。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 表现不佳但获得认可**：用户批评 **Open Interpreter 1.0** 的表现不如经典版本，缺少代码执行和网页浏览功能，并分享了 [GitHub 上的 OpenInterpreter](https://github.com/KillianLucas/open-interpreter) 作为参考。他们还强调了来自 **OpenProject**、**Rasa** 和 **Kotaemon** 的重要开源贡献。
   - 参与者强调了文本格式损坏和搜索工具缺失的问题，但仍赞扬开源社区推动了新功能的开发。
- **安装步骤简化设置**：出现了针对 Mac、Windows 和 Linux 的 **Open Interpreter** 单行安装流程，可实现快速的 Web UI 体验。好友们证实该方法简化了安装后的命令执行。
   - 好奇的用户在 #general 频道测试了该设置，确认这免去了他们手动配置环境的麻烦。
- **WhatsApp 笑话与对全天候交易工具的需求**：一位用户尝试了 **网页版 WhatsApp 消息发送**，开玩笑说这给乏味的文字聊天注入了活力。这次交流促使其他人分享了个人技术驱动的日常生活经验。
   - 另一场讨论集中在对**全天候交易点击器**的需求上，暗示需要一种永不休眠的操作系统级解决方案来持续执行命令。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **对 Mojo 链表的渴望**：有人请求一个能在 nightly 版本上运行的**链表（linked list）**代码库，凸显了在 **Mojo** 生态系统中对快速访问数据结构的渴望。
   - 贡献者们就高效的原型设计发表了意见，并建议将极小的开销作为精简探索的关键动力。
- **使用 Mojo 打造 CLI 和 TUI 工具**：一位开发者展示了他们在 **Mojo** 中构建 **CLI** 和 **TUI** 库的学习过程，为命令行爱好者开发了新的实用工具。
   - 其他人开玩笑说要打造一个新的 **Mojo** Shell，模仿 **bash** 和 **zsh**，进一步增强了社区对深度终端集成的热情。
- **AST 探索与调试风波**：成员们分享了他们在 `RootAstNode` 和 `DisplayAstNode` 等**索引风格树**上的成功经验，但在使用 **Mojo** 调试器时遇到了段错误（segmentation faults）。
   - [GitHub issue #3917](https://github.com/modularml/mojo/issues/3917) 记录了在 **--debug-level full** 下的这些崩溃，引发了关于复杂递归结构的激烈交流。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **1 月份 LLM Agents 证书的喜讯**：成员们根据最新更新确认，LLM Agents MOOC 的**证书**将于 **1 月底**颁发。
   - 他们建议大家*保持关注*以获取更多详情，并引导感兴趣的学习者访问 [2025 春季注册页面](https://llmagents-learning.org/sp25)。
- **2024 秋季落幕，2025 春季招手**：**2024 秋季**的入学现已关闭，失去了获得该学期证书的机会。
   - 成员们鼓励通过提供的表格加入 **2025 春季**课程，并提到课程大纲即将迎来改进。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini 的 GraphRAG 进展**：一位用户询问是否使用了特定的 **GraphRAG** 方法，结果发现 **Gemini** 调整了针对代码相关实体的默认提示词，以增强提取效果。
   - 他们指出，这种方法可以提高实体提取步骤的清晰度，重点在于优化 **DSPy** 的功能。
- **赠与者博弈（Donor's Game）的深入探索**：一名成员使用 **DSPy** 对博弈论中的 **Donor's Game** 进行了模拟，以复制多代之间重复的策略升级。
   - 他们引用了 [一个 GitHub 仓库](https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120)，该仓库实现了《LLM Agents 间的合作文化演进》（*Cultural Evolution of Cooperation among LLM Agents*）中的方法，探索鼓励 **LLM agents** 之间合作行为的方式。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 对 Windows 的支持**：一名成员询问 **tinygrad** 是否接受 Windows Bug 修复的 Pull Request，并强调了在 Windows 并非主要关注点的情况下支持该系统的挑战。
   - 另一名成员推测，如果这些修复能保持一致性和稳定性，将会受到欢迎，这表明了对跨平台扩展持谨慎但开放的态度。
- **Shapetracker 深度解析**：一名成员赞扬了 **tinygrad** 文档的详尽性，并引用了 [tinygrad-notes 博客](https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html) 以获取更深入的见解。
   - 他们寻求关于基于 `Shapetracker` 的矩阵内存中索引和步长（stride）计算的细节，并请求相关参考资料以澄清底层原理。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **WandB 与 MLflow 的长跑**：许多人注意到 **Weights & Biases** 提供托管服务，而 **MLflow** 可以自行托管以获得更多控制权。
   - 两个平台都能有效地追踪机器学习实验，团队的选择取决于成本和对工作流所有权的期望。
- **数据日志的妙用**：一些人提到将实验结果存储在 **Postgres** 或 **Clickhouse** 中，作为基础版本控制的备选方案。
   - 他们一致认为，在无法使用专门平台时，这是一种务实的路线。
- **经典机器学习（Classical ML）的现状**：一位用户质疑在 LLM 时代，**经典机器学习**（如推荐系统和时间序列）是否正在淡出。
   - 其他人持反对意见，认为尽管 LLM 备受关注，但这些领域仍然至关重要。
- **BitNet 进军 1-bit 领域**：关于 **BitNet** 的近期研究展示了 1-bit LLMs 在降低资源需求的同时，性能可媲美全精度模型。
   - 研究人员引用了 [这篇论文](https://arxiv.org/abs/2402.17764)，描述了三值权重（ternary weights）如何实现更廉价且高效的硬件支持。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **AI 阅读器：低成本笔记本电脑的 PDF 问答**：一位在非洲构建低成本系统的用户希望开发一种 **AI 阅读器**，在访问 PDF 时打开聊天 GUI，以帮助学生进行测试和内容理解，并正在探索 [Nomic embed 的用法](https://nomic.ai)。
   - 他们计划在本地硬件上处理内容 embeddings，并询问如何提供实时的模拟考试反馈，同时强调要最小化重新索引（re-indexing）的开销。
- **动态领域的内权威性排名**：一位参与者建议以一种随 **计算机科学** 演进而变化的方式来衡量教育材料的权威性。
   - 他们担心频繁重新索引带来的性能开销，提出了一种更灵活的方法来保持数据的时效性。
- **在搜索中提升学生成绩单的权重**：贡献者建议给 **学生成绩单** 增加额外权重，以在内容检索中反映个人的学术成长。
   - 他们认为更个性化的索引方法是下一个转变方向，让个人能够更精确地追踪学习成就。
- **按学科索引以增强资源控制**：用户提出了一种专注于 **学科** 而非单一参考资料（如书籍）的索引系统，旨在包含补充文章和笔记。
   - 他们认为这种方法能更好地覆盖知识盲点，并为备考提供更直接的资源选择。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Blender 与 AI 在 3D 领域的结合**：在 **#general** 频道中，有人询问关于利用 **AI** 驱动与 **Blender** 协作进行高级 3D 标注的问题，并提到了社区中的协同效应。
   - 他们正在寻找合作伙伴来扩展 **Blender** 的功能，旨在将 **AI** 更深层地集成到基于几何的任务中。
- **脑电波与动物趣谈**：一位参与者提到了利用 **Animals and EEG** 进行 **语言/动作映射 (language/action mapping)**，寻找探索 **AI** 和神经科学在动物研究中应用的团队。
   - 他们希望通过 EEG 数据解码 **动物行为**，这预示着生物启发式实验可能迎来新浪潮。
- **YoavhaCohen 的推文出现**：**#research** 频道中出现了一个指向 [这条推文](https://x.com/yoavhacohen/status/1875148348489113891) 的链接，但未提供更多细节。
   - 虽然内容神秘，但暗示了对 **YoavhaCohen** 未来动态的关注，引发了好奇。

---

**Axolotl AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# PART 2: 各频道详细摘要与链接

{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1324484170390437989)** (188 条消息🔥🔥): 

> `Windsurf 性能问题, Cascade 额度消耗, Codeium 插件建议, 用户支持体验, 学习与编程工具` 

- **Windsurf 性能问题**：许多用户报告在使用 **Windsurf** 时遇到**严重的性能问题**，包括大量的内部错误和响应时间缓慢。
   - 一些成员对错误导致额度消耗表示沮丧，对服务感到不满。
- **Cascade 额度消耗**：讨论了即使在发生内部错误时 **Cascade** 也会消耗额度的问题，用户认为这不公平。
   - 尽管自动回复声称并非如此，但用户注意到在操作失败期间仍会产生费用。
- **Codeium 插件建议**：用户分享了各种用于学习和提高编程能力的工具和库，推荐了 **React**、**Svelte** 和 **Next.js** 等对初学者友好的选项。
   - 一些用户选择主要使用 Codeium 插件进行项目初始化，而选择聊天功能进行学习。
- **用户支持体验**：用户对 Codeium 的客户支持评价褒贬不一，通常会收到自动回复，但缺乏有效的技术解决方案。
   - 几位用户建议通过电子邮件联系，描述问题并分享截图以寻求帮助。
- **学习与编程工具**：许多用户讨论了他们的编程历程，比较了各种编程辅助工具和 IDE。
   - 鼓励初学者利用社区资源和导师指导来提高编程技能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/@riandoris">Rian Doris</a>：Rían Doris 是 Flow Research Collective 的联合创始人兼 CEO，该机构是全球领先的巅峰表现研究和培训机构，专注于解码心流状态的神经科学并帮助...</li><li><a href="https://apps.apple.com/us/app/gold-fisher/id6739973000?l=zh-Hans-CN">‎Gold Fisher</a>：‎Gold Fisher - 终极宝藏捕鱼冒险！潜入令人上瘾的街机捕鱼体验，在这里寻金与深海冒险相遇！像熟练的勘探者一样挥动你的鱼钩...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1324471413544845438)** (194 条消息🔥🔥): 

> `Windsurf 性能问题，DeepSeek v3 vs. Sonnet 3.6，代码编辑错误，Prompt 与配置管理，额度系统反馈` 


- **Windsurf 面临性能挑战**：用户报告“Cascade has encountered an internal error”等错误增多，并对 Claude 3.5 Sonnet 的性能下降表示沮丧。
   - 多位成员说明了级联响应的问题，导致在没有有效解决问题的情况下增加了额度消耗。
- **DeepSeek v3 vs. Sonnet 3.6 之争**：讨论围绕 DeepSeek v3 与 Sonnet 3.6 的感知性能展开，一些人声称尽管有基准测试数据支持，但 DeepSeek 的表现仍不尽如人意。
   - 成员们对 DeepSeek 的价值表示怀疑，更倾向于使用 Gemini 等具有竞争力的免费替代方案。
- **代码编辑错误依然存在**：用户在 Windsurf 中进行代码编辑时遇到随机的破坏性更改，并要求更好地处理未完成的任务。
   - 一些人寻求关于如何 Prompt AI 在不丢失上下文的情况下继续之前任务的建议。
- **Prompt 与配置文件管理**：成员们分享了保存和引用 Prompt 配置的策略，以避免在使用 Windsurf 时进行重复解释。
   - 建议包括使用文本文件存储解释或向 AI 传递知识以备后用。
- **额度系统反馈**：用户对额度系统持批评态度，特别是 Premium 用户 Prompt 额度和 Flex 额度，主张根据使用情况更好地分配额度。
   - 人们对没有 Premium 额度后的持续使用情况表示担忧，并强调了未使用的额度结转（rollover）的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/#llms-need-better-criticism">2024 年我们学到的关于 LLMs 的事</a>：在 2024 年，Large Language Models 领域发生了很多事情。以下是对过去一年我们在该领域发现的事情的回顾……</li><li><a href="https://x.com/windsurf_ai/status/1874948790928687194">来自 Windsurf (@windsurf_ai) 的推文</a>：让你的 Figma 设计栩栩如生</li><li><a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/">2024 年我们学到的关于 LLMs 的事</a>：在 2024 年，Large Language Models 领域发生了很多事情。以下是对过去一年我们在该领域发现的事情的回顾……</li><li><a href="https://codeium.com/plan">方案设置</a>：未来的编辑器，就在今天。Windsurf Editor 是首个由 AI Agent 驱动的、能让开发者保持专注的 IDE。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://codeium.com/support,">页面未找到 | Windsurf Editor 和 Codeium 扩展</a>：Codeium 是开发者喜爱且企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor 和 Codeium 扩展</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://codeium.com/blog/codeium-better-chat">比 ChatGPT 更好的 Chat</a>：Codeium Chat 技术栈在模型、推理和 UX 方面取得了突破，使其成为一流的 AI Chat 开发者体验。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1324470956122570905)** (198 条消息🔥🔥): 

> `Linting 困惑、Deepseek 性能、Aider 中的 Architect 模式、新 AI 模型发布、在 Aider 中使用 OpenRouter` 


- **关于 Linting 请求的困惑**：一名成员对 AI 工具频繁发出的 Linting 请求表示困惑，质疑其为何专注于 Linting 而不是允许使用 `prettier` 等工具进行手动格式化。
   - 另一名成员澄清说，Aider 会自动对代码进行 Linting 以识别并修复问题，并提到了针对流行语言的内置 Linter。
- **Deepseek 的性能问题**：用户报告了 Deepseek 的性能缓慢，有些用户的速度低至 **1.2 TPS**，导致对其可靠性的不满。
   - 有建议认为需求增加可能影响了 Deepseek 的速度，并呼吁增加更多服务器以满足用户需求。
- **从 Architect 模式切换回常规模式**：一位用户寻求帮助，想知道如何从 Aider 的 Architect 模式切换回常规模式，同时仍能按需使用 Architect 功能。
   - 建议使用 `/chat-mode` 命令来切换模式，或者直接通过命令行实现。
- **新 AI 模型 Sonus-1 发布**：讨论围绕 **Sonus-1 系列** AI 模型的发布展开，强调了它们的能力和特定用例。
   - 该发布详情见于一篇 [博客文章](https://sonus.ai/blog/sonus-1)，其中概述了可用的不同版本：Mini、Air 和 Pro。
- **通过 OpenRouter 使用 Deepseek**：成员们讨论了将新的 **Deepseek Chat v3** 模型与 Aider 集成，探索使用命令 `--model openrouter/deepseek/deepseek-chat` 来访问它。
   - 确认了该设置确实会运行最新的 v3 模型，从而增强用户在 Aider 中的体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://sonus.ai/blog/sonus-1">介绍 Sonus-1：LLM 的新时代 - Goran Babarogic 产品设计师</a>：介绍 Sonus，来自 Sonus AI 的最先进语言模型。体验先进的推理、自然语言理解和强大的文本生成能力。</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 与测试</a>：自动修复 Linting 和测试错误。</li><li><a href="https://aider.chat/docs/troubleshooting/support.html">使用 /help</a>：使用 “/help” 询问有关使用 Aider、自定义设置、故障排除、使用 LLM 等方面的帮助。</li><li><a href="https://aider.chat/docs/usage/copypaste.html">通过 Web Chat 进行复制/粘贴</a>：Aider 可与 LLM Web Chat UI 配合使用</li><li><a href="https://x.com/RubiksAI/status/1874682159379972325">来自 Rubik's AI (@RubiksAI) 的推文</a>：🎉 新年快乐！🚀 介绍 Sonus-1 系列：Mini, Air, Pro 和 Reasoning！一套旨在满足您 2025 年及以后多样化需求的新模型！🧵(1/7)</li><li><a href="https://aider.chat/docs/usage/modes.html">聊天模式</a>：使用 code, architect, ask 和 help 聊天模式。</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat-v3">DeepSeek V3 - API, 提供商, 统计数据</a>：DeepSeek-V3 是来自 DeepSeek 团队的最新模型，建立在先前版本的指令遵循和代码编写能力之上。在近 15 万亿 Token 上进行了预训练，报告的评估结果...</li><li><a href="https://github.com/mufeedvh/code2prompt">GitHub - mufeedvh/code2prompt: 一个将代码库转换为包含源码树、提示词模板和 Token 计数的单个 LLM 提示词的 CLI 工具。</a>：一个将代码库转换为包含源码树、提示词模板和 Token 计数的单个 LLM 提示词的 CLI 工具。 - mufeedvh/code2prompt</li><li><a href="https://github.com/Aider-AI/aider/issues/166)">Issues · Aider-AI/aider</a>：Aider 是您终端中的 AI 配对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1324469322982363199)** (47 条消息🔥): 

> `OpenRouter API 问题，Aider 配置与上下文管理，Tailwind CSS 文档集成，用于 RAG 的 Graphrag 工具，Aider 的功能需求` 


- **OpenRouter API 故障**：用户报告了使用 OpenRouter API 时的问题，特别是关于身份验证问题以及配置文件中错误的 API key 设置。
   - 一位用户确认了他们的配置可以工作，强调了检查正确模型设置的重要性。
- **配置 Aider 以实现有效的上下文管理**：成员们讨论了如何更好地配置 Aider 的上下文管理，包括使用外部文件上下文以及理解模型的 metadata。
   - 针对使用命令自动集成相关文档以实现更流畅的工作流，提出了具体建议。
- **集成 Tailwind CSS 文档**：Aider 用户表达了直接集成 Tailwind CSS 文档上下文的兴趣，建议包括复制相关信息或使用其他服务的索引文档。
   - 有人提议通过命令提示符自动查阅 Tailwind 文档。
- **关于 Graphrag 工具的讨论**：一位用户提到正在使用 Microsoft 的 Graphrag 工具，并指出它似乎比传统的 RAG 方法更高效。
   - 社区表达了寻找与 Graphrag 兼容的有效 CLI 工具的兴趣。
- **Aider 的未来功能需求**：用户分享了对 Aider 的功能需求，强调改进其在进行代码更改前管理类定义和先前上下文的方式。
   - 有建议提出通过开关功能来提升用户体验和编码任务的效率。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.jetbrains.com/ai/ai-assistant-features/">AI Assistant Features</a>: 探索 JetBrains AI Assistant 的功能：上下文感知代码生成、高级代码补全、自动测试创建、AI 聊天等。无缝集成到您的 JetBrains IDE 中...</li><li><a href="https://aider.chat/docs/faq.html#how-can-i-add-all-the-files-to-the-chat">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/languages.html">Supported languages</a>: Aider 支持几乎所有流行的编程语言。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>: 为 LLM 配置高级设置。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML config file</a>: 如何使用 YAML 配置文件配置 aider。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1324473294740787342)** (188 messages🔥🔥): 

> `OpenWebUI dataset export, Inference methods in Unsloth, Model quantization issues, VLLM for LLM inference, Fine-tuning choices for text classification` 


- **从 OpenWebUI 导出聊天记录**：成员们讨论了从 OpenWebUI 导出聊天 JSON 的问题，并提到了将其格式化为数据集的方法。
   - 建议向 OpenWebUI 的所有者咨询关于数据集导出格式的信息。
- **本地运行 Unsloth 推理**：出现了关于在微调模型后本地运行 Unsloth 推理的问题，强调了将基础模型与 LoRa 结合使用。
   - 成员们被引导至 vLLM 文档，以了解如何在各种模型中实现推理。
- **Ollama 的量化挑战**：讨论集中在运行 Ollama 量化模型相关的问题上，用户正在排查错误和配置。
   - 有人指出 Ollama 中的 GGUF 文件默认为 FP16，并建议手动调整 Modelfile 以进行其他量化。
- **选择微调模型**：社区成员根据分类复杂度分享了适合微调的模型建议，推荐了 Llama 3.1 8B 和 Llama 3.2 3B。
   - 讨论强调了使用 RTX 4090 等硬件以在模型训练和推理中获得最佳性能。
- **分类任务的经验**：用户分享了 LLM 在分类任务上的表现经验，报告了 Llama 和 BERT 模型面临的挑战和不同的结果。
   - 一位参与者提到，使用 Llama 3.2 3B 对不平衡数据集进行分类，达到了约 74% 的准确率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/">Welcome to vLLM! &#8212; vLLM</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>: 未找到描述</li><li><a href="https://ollama.com/mesosan/lora_model">mesosan/lora_model</a>: cmc server abbie sir clone-ish</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1h1GYAGGXMhkPHr4QhRz5TLt6ZP-W8Rr5#scrollTo=J7lk6l0CuPXS">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">Tutorial: How to Finetune Llama-3 and Use In Ollama | Unsloth Documentation</a>: 为创建可在 Ollama 本地运行的定制个人助手（如 ChatGPT）提供的初学者指南</li><li><a href="https://ollama.com/mesosan/lora_model/blobs/97f36c95b3fd">mesosan/lora_model/model</a>: cmc server abbie sir clone-ish</li><li><a href="https://github.com/unslothai/unsloth/issues/689">Does unsloth have a script for full parameter fine-tuning? · Issue #689 · unslothai/unsloth</a>: 未找到描述</li><li><a href="https://dub.sh/bit.ch">bit.chan</a>: 查看 bit.chan 的个人资料和 3D 模型
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1324475954734567597)** (16 条消息🔥): 

> `Unsloth 库安装问题，Granite 训练错误，针对特定任务微调模型，在 Colab 中使用自定义数据集，理解 Embedding 与微调的区别` 


- **Unsloth 库出现 ModuleNotFoundError**：一名用户报告称，尽管尝试了 `pip install` 等多个安装步骤，但在 Kaggle notebook 中尝试导入 **unsloth** 时仍持续出现 **ModuleNotFoundError**。另一名成员建议查看[此处](https://docs.unsloth.ai/get-started/unsloth-notebooks)关于在 Kaggle notebook 中使用 Unsloth 的教程。
   - 提到的另一个可能解决方案是 **chat template** 的问题，这可能会影响不同平台的性能，并提供了[错误修复链接](https://docs.unsloth.ai/basics/errors)。
- **Granite 训练代码错误报告**：成员们讨论了使用最新版本进行 **granite 训练**时遇到的错误，但不确定问题是否出在用户端。一名用户一直尝试解决各种安装问题，但不确定错误的根本原因。
   - 这引发了关于其他人是否也面临类似困难的疑问，暗示该功能需要更清晰的文档或支持。
- **模型微调建议**：一名用户正在为针对 SQL 生成和技术文档查询的微调寻找 **base model** 建议。建议包括将 **Qwen 2.5** 作为一个可靠的选择，同时推荐了用于数据去重和训练格式化的工具。
   - 随后展开了关于在特定应用中使用 **RAG Agent** 的讨论，展示了实现有效模型训练的多样化方法。
- **如何在 Colab 中使用 CSV 数据**：一名用户询问如何将他们的 **CSV 文件** 集成到 Colab 的教程中，寻求在提供的示例中使用自有数据的步骤。一名成员建议了诸如将数据上传到 **Hugging Face** 或直接上传到 Colab 以方便访问的选项。
   - 提供了关于在 Colab 中加载和管理数据的最佳方法的说明，帮助新手熟悉环境。
- **寻找 Unsloth 学习资源**：一名用户表示有兴趣获取推荐的 **视频或教程**，以开始使用 **Unsloth** 并安装训练模型所需的先决条件。他们目前依赖 GitHub 文档，但渴望看到更具吸引力的分步指南。
   - 重点被放在深入理解工具和使用场景上，以此作为提高学习和实施效率的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=vITh0">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：查看下方列表获取我们所有的 notebook：</li><li><a href="https://colab.research.google.com/drive/1ef-tab5bhkvWmBOObepl1WgJvfvSzn5Q?usp=sharing#scrollTo=vITh0KVJ10qX">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/errors">Errors | Unsloth 文档</a>：要修复设置中的任何错误，请参阅下文：
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1324672299445784598)** (8 条消息🔥): 

> `通过隐式奖励进行过程强化，O1 复现工作，复旦报告，强化学习代码` 


- **复旦关于 O1 复现工作的报告引起关注**：复旦的报告被认为是目前为止关于 **O1 复现工作** 最详尽的资源，可在[此处](https://arxiv.org/pdf/2412.14135)查阅。
   - 一名成员评论道：*这几乎是我目前发现的关于 O1 复现工作最全面的一份报告*。
- **过程强化论文缺少代码**：尽管 **Process Reinforcement through Implicit Rewards** 论文非常值得一读，但成员们对其缺少配套的强化学习 **代码** 表示遗憾。
   - 一名成员表达了失望，称 *遗憾的是没有关于 RL 的代码*，而另一名成员提到关于代码的更新即将发布。
- **鼓励持续开发**：一名成员指出，关于 **Process Reinforcement 论文** 的信息中包含了代码即将可用的承诺，并称其为 *进行中的工作 (Work in Progress)*。
   - 对代码发布的期待反映了社区对源自该研究的实际应用的兴趣。


  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1324470663842627584)** (170 条消息🔥🔥): 

> `DeepSeek V3 模型讨论、电子邮件欺骗事件、使用 Cursor 进行项目开发、网站设计灵感、市场开发语言选择` 


- **DeepSeek V3 模型讨论**：用户对 **DeepSeek V3 模型** 存储代码并使用用户数据进行训练表示担忧，这引发了隐私问题。
   - 一些成员强调了它的能力，同时也质疑在项目中使用此类模型的益处和影响。
- **电子邮件欺骗事件及社区反应**：一名用户收到了一封来自伪造地址的幽默且带有侮辱性的邮件，引发了大家的笑声以及对欺骗手段有效性的讨论。
   - 成员们开玩笑地鼓励伪造邮件的制作者围绕发送此类邮件开展业务，并指出了这种情况的荒谬性。
- **利用 Cursor 进行快速项目开发**：一位用户分享了使用 **Cursor AI** 快速完成一个涉及 **SignalR** 项目的正面经验，声称开发时间显著减少。
   - 这引发了关于 **Cursor** 等 AI 工具如何改变开发格局、使复杂任务变得可控的讨论。
- **寻找网站设计灵感**：一位用户寻求提供设计灵感的网站推荐，并收到了社区的多个建议。
   - 成员们分享了 **land-book.com** 和 **godly.website** 等网站，并对这些有用的资源表示感谢。
- **为市场选择合适的语言**：成员们讨论了构建市场（Marketplace）最合适的编程语言，建议倾向于使用 **JavaScript**，因为它易于使用。
   - 他们强调，虽然许多语言都可能适用，但 **JavaScript** 允许在前端和后端进行高效开发。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://microsoft.github.io/monaco-editor/">Monaco Editor</a>: 未找到描述</li><li><a href="https://x.com/liamesp/status/1869319333954089218?s=46">来自 liam (@liamesp) 的推文</a>: 现在，@krea_ai 编辑器中支持对象 + 画笔选择</li><li><a href="https://tenor.com/view/drinks-gif-8100232030695928923">Drinks GIF - Drinks - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://forum.cursor.com/t/error-request-type-deprecated-when-using-docs-in-composer-agent-mode/38610/14">[已解决] 错误：在 Composer Agent 模式下使用 @docs 时请求类型已弃用</a>: 现在可以工作了吗？我们大约一小时前发布了修复程序。</li><li><a href="https://www.youtube.com/watch?v=NCaRixtXNIo"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1324472338280087613)** (139 条消息🔥🔥): 

> `Swarm 库发布，研究提案协助，ICML 的 NLP 工作坊，微调模型，AI 模型的训练成本` 


- **Swarm 库现已在 NPM 上线**：用于创建多 Agent AI 系统的新库 **Swarm** 刚刚在 [NPM](https://www.npmjs.com/package/agentswarm) 上发布。它具有灵活性和模型无关性（model agnosticism），适用于各种 AI 协作。
   - 作者指出，与现有的 **OpenAI Swarm** 相比，该包提供了更好的模式和改进。
- **需要研究提案支持**：一位成员询问了关于研究提案的支持，促使其他人建议在特定频道发布以寻求社区帮助。该服务器似乎对研究相关查询的协作努力持开放态度。
   - 这表明社区非常活跃，愿意协助学术抱负，特别是在 AI 研究领域。
- **征集 NLP 工作坊参与者**：介绍了关于 ICML 2025 **PersonaNLP workshop** 的提案，邀请大家对论文提交和共享任务表示关注。这反映了在 NLP 领域促进协作的持续努力。
   - 鼓励感兴趣的成员在共享频道中加入有关该提案的讨论。
- **在个人硬件上微调模型**：成员们讨论了在 **4090** 等个人 GPU 上微调模型的可行性，并建议使用 **Unsloth**、**Axolotl** 或 **Llama Factory** 等工具。这反映了在本地实验 AI 模型日益增长的兴趣。
   - 虽然像 Amazon Bedrock 这样的云选项因隐私性而受到关注，但本地选项为个人开发者提供了灵活性。
- **AI 模型昂贵的训练成本**：一位成员强调了训练大型 AI 模型相关的高昂成本，提到大型模型的价格高达 **600,000 美元**。讨论集中在没有大规模资源的个人进行此类尝试的可及性上。
   - 提出了各种高效训练的方法，反映了使 AI 研究更具可行性的浓厚兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2405.16528">LoQT: Low-Rank Adapters for Quantized Pretraining</a>: 尽管在使用低秩适配器和量化方面取得了进展，但在没有模型分片（sharding）、训练期间卸载（offloading）或逐层梯度...的情况下，在消费级硬件上预训练大型模型仍然是不可能的。</li><li><a href="https://runpod.io?ref=jgbvgh5q">RunPod - 为 AI 构建的云</a>: 在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://www.youtube.com/watch?v=XwL_cRuXM2E"> - YouTube</a>: 未找到描述</li><li><a href="https://news.ycombinator.com/item?id=39224534">如果你到处读读，训练一个 7B 模型的成本大约在 85,000 美元左右；1.4 s... | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/moinnade">来自 FxTwitter / FixupX 的推文</a>: 抱歉，该用户不存在 :(</li><li><a href="https://x.com/moinnadeem/status/1681371166999707648">来自 Moin Nadeem (@moinnadeem) 的推文</a>: 这是 LLaMa 2 论文中一张重要的图表。它直接概述了该模型的预训练时长！以下是成本，假设来自 @LambdaAPI 的 A100 为每小时 1.50 美元：- 7B 模型成本为 276,480 美元！- 13B 模型...</li><li><a href="https://www.npmjs.com/package/agentswarm">agentswarm</a>: 使用 Vercel AI SDK 创建 OpenAI 风格 Swarm Agent 的 LLM 无关 TypeScript 框架。最新版本：0.0.4，最后发布：一天前。通过运行 `np...` 开始在你的项目中使用 agentswarm。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1324503207468077056)** (24 条消息🔥): 

> `Hermes 训练数据, VLM 使用案例, Fine-Tuning 模型, 模型权重分布, Mergoo 介绍` 


- **关于 Hermes 训练数据的争议**：一位成员指出 Hermes 中**没有明确的训练数据**来模拟性爱场景，引发了关于这是否会影响模型性能的推测。
   - 另一位成员评论说，这种缺失实际上可能会阻碍模型利用某些知识的能力，而这些知识本可以增强其更广泛的性能。
- **探索 VLM 使用案例**：讨论集中在 Vision Language Models (VLMs) 的**实际应用**上，建议包括验证 Web 开发和机器人技术。
   - 一位成员对识别计算机元素表示好奇，并提议这可能是特殊 Fine-Tuning 过程的任务。
- **当前用于 Fine-Tuning 的模型**：成员们分享了 **Llama**、**DeepSeek** 和 **Qwen** 是最常用于 Post-training 和 Fine-Tuning 的模型。
   - 出现了关于它们的商业使用 License 以及潜在 Tokenization 考虑的问题。
- **模型权重分布的低效性**：分享了关于 Llama2 模型中 **K 和 Q 权重分布**的见解，表明这暗示了编码中的某些低效性。
   - 附带的图像引发了对权重幅度的进一步分析，暗示并非所有 Token 在输出中都具有同等的重要性。
- **Mergoo 介绍**：一位新人询问了 **Mergoo**，将其社区动态比作两年前的 r/localllama。
   - 回复显示了对 r/localllama 现状的复杂情绪，表明最近的变化导致了一些用户的不满。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1324501544191660195)** (119 条消息🔥🔥): 

> `Extractor.io 网站, LM Studio 图像生成, Qwen2-VL 模型限制, 使用互联网数据进行 AI 训练, 模型性能问题` 


- **Extractor.io 网站困惑**：一位用户报告发现了 **Extractor.io** 网站，却发现它已不存在，这引起了一些关注。
   - 另一位用户建议它可能已被[此链接](https://llm.extractum.io/list/)取代，该链接提供了一个精选的语言模型列表。
- **LM Studio 不支持图像生成**：聊天参与者确认 **LM Studio** 无法直接生成图像，因为它仅处理 LLMs，而不处理视觉模型。
   - 提到了像 **Pixtral** 这样的替代方案，但它们仅在 Mac 上通过 **MLX Engine** 可用。
- **Qwen2-VL 模型缺乏视觉能力**：讨论指出 **Qwen2-VL-7B-Instruct-abliterated** 模型没有视觉适配器，因此无法处理图像。
   - 参与者强调需要对原始模型进行适当的 Quantization（量化），以充分利用其功能。
- **使用全互联网数据进行 AI 训练的挑战**：一位用户表达了对在整个互联网上训练 AI 的兴趣，但其他人指出，由于数据质量和可用性问题，这是不切实际的。
   - 几位参与者强调，**质量比数量更重要**，糟糕的数据会损害模型性能。
- **模型性能和参数讨论**：用户讨论了他们使用各种模型的经验，指出了重复性等性能问题，特别是在 **Mistral Nemo Instruct** 中。
   - 他们还辩论了所需的模型大小和参数，认为更好的模型即使参数较少也可能产生更优的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llm.extractum.io/list/">所有大型语言模型</a>：大型和小型语言模型（开源 LLMs 和 SLMs）的精选列表。具有动态排序和过滤功能的所有大型语言模型。</li><li><a href="https://www.shepbryan.com/blog/what-is-gguf">什么是 GGUF？初学者指南 — Shep Bryan</a>：在寻找运行本地 AI 模型时，你可能见过 GGUF 这个术语。但它是什么？它又是如何让你在自己的设备上使用尖端 LLMs 的？</li><li><a href="https://huggingface.co/mradermacher/Qwen2-VL-7B-Instruct-abliterated-GGUF/tree/main">mradermacher/Qwen2-VL-7B-Instruct-abliterated-GGUF at main</a>：未找到描述</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio - Beta 版本</a>：LM Studio 的 Beta 和发行候选版本</li><li><a href="https://github.com/ggerganov/ggml/blob/master/docs/gguf.md">ggml/docs/gguf.md at master · ggerganov/ggml</a>：用于机器学习的张量库。通过在 GitHub 上创建账户来为 ggerganov/ggml 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus: 专注于 Prompting 和生成</a>：专注于 Prompting 和生成。通过在 GitHub 上创建账户来为 lllyasviel/Fooocus 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1324510979618504788)** (38 条消息🔥): 

> `Local LLM Usage, API Concerns, GPU vs CPU Utilization, Quest Generation, Hardware Recommendations` 


- **倾向于本地 LLM 而非 API**：@octaviagoetia 表达了对使用 **GPU** 而非 **在线 API** 的偏好，理由是对 **TOS 限制** 的担忧以及潜在的延迟问题。
   - @heychazza 表示赞同，并强调了使用高性能笔记本电脑进行 **LLM** 尝试的好处，无需担心意外费用。
- **使用 Llama 3 成功生成任务**：@octaviagoetia 目前正在使用 **Llama 3.1** 生成任务线，并计划升级到 **Llama 3.2 1b**，认为它更具创意且效率更高。
   - 这种方法允许 **动态 NPC 交互**，减少了手动创建任务所需的工作量。
- **确认 LLM 中的 GPU 使用情况**：@antonyj0666 询问了如何在使用 **4070 Ti Super** 显卡的本地 LLM 设置中启用 **GPU** 利用率。
   - @heyitsyorkie 确认，选择具有明确 **GPU offload capabilities** 的模型对于正常运行至关重要。
- **在推理过程中识别 GPU 使用情况**：为了确认程序是否正在利用 GPU，@christianazinn 建议在模型推理期间检查 **Task Manager**（任务管理器）。
   - 如果运行模型时 GPU 性能指标出现峰值，则表明 **GPU** 已成功参与工作。
- **不同的 GPU Offload 能力**：@mrhoopers 分享了关于各种具有 **GPU offload capabilities** 模型的见解，并强调绿色指示器代表完全支持。
   - 他澄清说，虽然绿色表示潜在的 GPU 使用，但并不保证 **模型当前正运行在 GPU 上**。



**相关链接**：<a href="https://model.lmstudio.ai/download/NousResearch/Hermes-3-Llama-3.1-8B-GGUF">在 LM Studio 中下载并运行 NousResearch/Hermes-3-Llama-3.1-8B-GGUF</a>：在你的 LM Studio 本地使用 NousResearch/Hermes-3-Llama-3.1-8B-GGUF

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1324478524706263071)** (15 条消息🔥): 

> `Various media references, Conflict of interest in studies, Translation checks, Long-term mycelium storage techniques, Turing Test discussions` 


- **分享的媒体链接**：用户分享了各种媒体链接，包括[这篇文章](https://www.akashq.com/post/23a45b75-75ed-4741-94a6-0252f493748a)和[一个 Spotify 节目](https://open.spotify.com/show/5X89wBkhOVCYJJR9NsntVJ?si=f87a4ab74070424e)，涵盖了多种话题。
   - 这些链接引发了关于其内容和相关性的额外讨论。
- **对研究偏见的担忧**：*一位用户提到在审查研究时利益冲突披露的重要性*，强调了关于谁从研究中获益的透明度。
   - *Retraction Watch* 被指出是一个识别已被撤回但仍被广泛引用的研究的重要资源。
- **真菌学存储技术**：*一位用户分享了关于使用斜面培养基（slants）进行长期菌丝体培养存储的见解*，并附带了关于该技术的音频资源链接。
   - 讨论还包括了利用不同浏览器有效保存音频文件的技巧。
- **图灵测试讲座见解**：*围绕一场关于如何通过图灵测试（Turing Test）的讲座展开了讨论*，并辅以讨论的音频文件链接。
   - 这引发了关于提高此类测试表现的方法和策略的交流。
- **转换音频文件**：*用户讨论了将音频文件从 WAV 转换为 MP3 的方法*，提到了 Audacity 和 Adobe Audition 等工具。
   - 一位用户对这些技巧表示感谢，表明了音频文件管理的实际应用。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://open.spotify.com/show/5X89wBkhOVCYJJR9NsntVJ?si=f87a4ab74070424e">WeTalk Business</a>：播客 · Bilal Rmaili · 买地吧，因为上帝不再造地了。咨询：@3amfxct</li><li><a href="https://www.akashq.com/post/f5a0eff9-6ad7-4137-b35c-186f243ce3c7">1 月 3 日的历史上发生了什么？</a>：由 This Day in History 发布的 1 月 3 日历史事件</li><li><a href="https://www.akashq.com/post/23a45b75-75ed-4741-94a6-0252f493748a">1 月 2 日发生了什么？</a>：由 This Day in History 发布的 1 月 2 日历史事件
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1324475851038789714)** (101 条消息🔥🔥): 

> `笔记本共享问题、Beta 测试体验、多语言功能、AI 摘要产品、自定义功能使用` 


- **笔记本共享面临问题**：用户在尝试通过电子邮件共享笔记本时遇到问题，收件人界面中未显示通知或共享的笔记本。
   - 多位尝试过两种共享方式的用户已确认此问题，表明可能存在系统故障。
- **Beta 功能的挑战**：多名用户在加入 Beta 交互模式时遇到超长等待时间，经常出现无限加载界面。
   - 问题可能源于麦克风访问权限，导致用户对输入电平和音频设备设置产生疑虑。
- **AI 的多语言能力**：虽然音频概览（Audio Overview）功能官方仅支持英文，但用户已尝试通过创意 Prompting 生成其他语言的内容。
   - 反馈显示，虽然翻译功能可行，但质量差异显著，尤其是非欧洲语言。
- **AI 摘要产品受到关注**：一位用户正在推广一款新 AI 摘要产品的 Beta 测试机会，该产品在韩国市场取得了显著成功。
   - 社区成员对测试该产品表现出兴趣，该产品承诺提供精简的摘要功能。
- **对自定义功能的担忧**：有人对自定义功能的潜在滥用表示担忧，担心其被用于修改 System Prompts 并以此不公平地利用 AI 能力。
   - 用户讨论了在创作自由与 AI 伦理实践之间保持平衡的重要性，并将其与 NSFW 内容管理进行了类比。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://akashq.com">Akas: home to AI podcasts</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/05adb4e6-7905-4f0d-abe5-1886faf6e4f1/audio">未找到标题</a>: 未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?sjid=9595575708425202127-NC&visit_id=638697544924377600-2451215312&p=plus&rd=1">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://youtu.be/aG0ixD3OY80"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1324779928939659396)** (3 条消息): 

> `Bolt 代码处理问题、UI 实现挑战、Web 应用开发技巧` 


- **Bolt 的“创意”代码行为令用户沮丧**：成员们表示，尽管有明确指令，**Bolt** 有时仍会自行注释掉或删除代码。
   - *“即使尝试明确说明，它有时仍会这样做”*，这呼应了多位用户的担忧。
- **JSX Grid 代码管理不善**：一位用户报告称，当为 UI 提供 **JSX grid code** 时，**Bolt** 完全重新设计了它，忽略了原始输入。
   - 他们寻求关于如何防止 **Bolt** 在处理请求时过于“有创意”的建议。
- **新年新应用：寻求 Logic/AI 方面的帮助**：一位成员分享了使用 **Bolt** 开发 Web 应用的兴奋之情，但承认在实现 **Logic/AI** 功能方面遇到了困难。
   - 他们指出，视觉效果非常出色，但功能需要增强，并征求相关技巧。


  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1324468876477595648)** (110 条消息🔥🔥): 

> `Bolt 的计费问题，在 Bolt 中进行调试，在 Bolt 中集成 API，使用 mock 数据构建前端，在 Bolt 中使用 Supabase` 


- **用户反馈 Bolt 的 Token 消耗过快**：多位用户对 Bolt 中 Token 余额迅速耗尽表示担忧，对消耗速率表示不满。
   - 一位用户建议在前端开发中使用 mock 数据可能有助于缓解 Token 的快速消耗。
- **预览屏幕空白问题**：多位用户在使用 Bolt 时遇到空白屏幕，建议检查 Bolt 终端中的错误或关闭未使用的标签页。
   - 一位用户通过重启电脑解决了该问题，这表明系统性能可能会影响预览功能。
- **Netlify 回调 URL 返回 404**：一位用户在从 Bolt 构建的表单向 Netlify 回调 URL 发送 POST 请求时遇到 404 错误，质疑这是否是功能限制。
   - 建议用户改用 Netlify Functions，因为 Netlify 的静态托管不直接接受 POST 请求。
- **Bolt 的客户支持**：用户询问了针对计费和技术问题的客户支持可用性，回复指出支持选项有限。
   - 据指出，客户支持主要处理计费问题，而不提供技术协助。
- **对使用 Supabase 进行邮箱登录的兴趣**：用户分享了在基于 Bolt 构建的应用中集成 Supabase 进行邮箱身份验证的经验。
   - 建议使用 local storage 测试前端功能，作为一种在不立即部署的情况下有效管理用户角色的策略。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boltstudio.ai/">BoltStudio.ai | Full Stack Prompt Engineering</a>: 未找到描述</li><li><a href="https://answers.netlify.com/t/receiving-post-requests/86669">Receiving POST requests</a>: 你好！我在处理 POST 请求时遇到问题。我有一个独立的 node.js 服务器，偶尔会向我的 netlify 站点发送 POST 请求。然而，我已经研究了几个小时，仍然...</li><li><a href="https://github.com/stackblitz-labs">StackBlitz Labs</a>: StackBlitz Labs 有 2 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/stackblitz">StackBlitz</a>: StackBlitz 有 45 个可用的代码库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1324480026220953600)** (103 条消息🔥🔥): 

> `Perplexity O1 Feature, ChatGPT vs. Perplexity, Grok Model Opinions, UI Changes in Perplexity, Using AI Subscriptions` 


- **对 Perplexity O1 特性的评价褒贬不一**：用户在使用 Perplexity 的 O1 特性时遇到了问题，有报告称格式错误以及每日搜索次数受限。*一位用户表达了挫败感，称：“兄弟，太麻烦了，而且每天只有 10 次搜索。”*
   - 另一位用户提到他们一直在 X 上使用免费版的 Grok，这引发了对其与其他模型相比能力的关注。
- **AI 工具对比：Perplexity 与 ChatGPT**：用户讨论了 Perplexity 和 ChatGPT 之间的区别，指出 Perplexity 在搜索能力方面更胜一筹，而 ChatGPT 由于其更大的 Context，在非搜索任务中可能表现更好。一位用户评论道：“对我来说绝对值得，因为 Opus 是无限的，而且他们所有模型的 Context 记忆力都非常高。”
- **Grok 模型收到褒贬不一的反馈**：Grok 收到了一些批评性反馈；一位用户声称尽管它具有性价比，但却是“我用过最差的模型”。其他人则更倾向于 3.5 Sonnet 模型，因为它在任务执行中表现强劲。
- **Perplexity 最近的 UI 变更引发讨论**：用户注意到了 Perplexity 最近的 UI 变化，例如在首页增加了股票和天气信息，并询问如何禁用这些功能。一位用户通过清除缓存解决了不想要的显示元素问题，并表示：“这让我想起了那些糟糕的浏览器主页！”
- **关于订阅方案和用户反馈的关注**：对话包括用户讨论他们对各种 AI 工具的订阅，以及对无限查询等功能的体验。一些用户对由于订阅中意外出现的故障而无需支付某些服务费用感到庆幸。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/pplxsupply/status/1875019712268263658?s=46">来自 Perplexity Supply (@PPLXsupply) 的推文</a>：ask.learn.do.</li><li><a href="https://x.com/aravsrinivas/status/1874943854849425780?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：如果你是 Perplexity 的 Android 用户，请私信获取即将推出的酷炫功能的早期测试资格。</li><li><a href="https://tenor.com/view/new-year-happy-new-year-2025-happynewyear-new-years-gif-11684221509638957422">新年快乐 GIF - 2025 新年快乐 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://chrisbora.substack.com/p/wbs-framework">超越 Prompting：What-Boundaries-Success 框架</a>：粒子物理学和搜索系统如何导致 AI Control 的根本性突破</li><li><a href="https://github.com/cbora/aispec">GitHub - cbora/aispec: 一种面向 AI 优先开发的规范语言，通过结构化的解空间缩减，将重点从实现转向意图</a>：一种面向 AI 优先开发的规范语言，通过结构化的解空间缩减，将重点从实现转向意图 - cbora/aispec
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1324589625037029497)** (7 条消息): 

> `马斯克诉讼支持, AI 面试准备, 海洋与二氧化碳吸收, 2025 年预测, Perplexity AI 起源` 


- **AI 教父支持马斯克诉讼**：*AI 教父*公开支持马斯克的诉讼，引发了关于其对行业影响的讨论。最近的一段 [YouTube 视频](https://www.youtube.com/embed/tRAGov9VRxQ) 重点讨论了这些持续存在的争议。
   - 该视频还涵盖了数据中心对**电网**的扭曲以及对 2025 年的预测。
- **2025 年攻克 AI 面试问题的技巧**：一名成员分享了关于如何在 2025 年求职面试中有效应对 AI 相关问题的指南，强调了准备策略。更多详情可以在[此处](https://www.perplexity.ai/search/the-job-interview-question-of-geScATofQC.NYw5MqWsyiA)找到。
   - 随着 AI 岗位的竞争日益激烈，该指南显得尤为重要。
- **海洋吸收了人类排放的 1/3 二氧化碳**：一篇文章讨论了海洋如何吸收大约**三分之一**的人类二氧化碳排放量，强调了它们在气候调节中的关键作用。你可以在[此处](https://www.perplexity.ai/page/oceans-absorb-1-3-of-human-co2-S586TEA4QN.ngghjoWC0nQ)阅读更多相关内容。
   - 这进一步强化了海洋保护在气候行动努力中的重要性。
- **Perplexity AI 的概念构思**：讨论涉及了 Perplexity AI 背后想法的起源，成员们分享了相关见解。如需了解更深层的背景，请查看[此处](https://www.perplexity.ai/search/when-was-the-idea-behind-perpl-.JlEn.TrRl.zj824uT_xwQ)的链接。
   - 了解其起源对于理解该平台的持续发展至关重要。



**提到的链接**：<a href="https://www.youtube.com/embed/tRAGov9VRxQ">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1324678621754818657)** (1 条消息): 

> `API 服务器位置, Chatbot 集成, Token 利用率` 


- **欧洲 API 以获得更好性能**：一名成员表示乐观，认为如果 **API** 的服务器位于**欧洲**且性能与 **pro search** 匹配，它将被集成到他们的 **15 亿 Token Chatbot** 中。
   - 他们表示希望在这一潜在集成后，Chatbot 的功能能够得到改进。
- **对 Token 利用的预期**：该成员强调了通过新的 API 集成有效利用 **15 亿 Token** 的潜力。
   - 他们期待通过这种方式增强 Chatbot 的能力。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1324467807093522544)** (86 条消息🔥🔥): 

> `OpenRouter 身份验证问题, DeepSeek 性能, 结构化输出的模型推荐, Janitor AI 与 OpenRouter 的集成, 支付处理问题` 


- **OpenRouter 身份验证问题**：多名用户报告了在 n8n 上使用 OpenRouter 时出现的问题，消息显示尽管已加载额度，但仍出现 **'Unauthorized'** 错误。
   - *Matt070655* 提到更改了 HTTPS 地址并添加了 API key，但仍然遇到连接被拒绝的情况。
- **DeepSeek 性能困扰**：用户对 **DeepSeek 的缓慢性能** 表示沮丧，其中一位指出其 **TPS** 低至 **0.6**。
   - 有人担心当前的需求可能未得到充分预测，导致体验下降。
- **结构化输出的模型推荐**：一位用户正在寻找 **gpt-4-mini** 的替代方案用于结构化 JSON 输出，发现选择有限。
   - 其他人建议使用 **Gemini Flash** 等模型，并讨论了版本的有效性以及预期的速率限制（rate limitations）。
- **Janitor AI 与 OpenRouter 的集成**：提供了关于如何设置 OpenRouter 与 Janitor AI 的帮助，强调了在设置中对 URL 和 API 兼容性的调整。
   - 指南强调了在高级设置中切换选项以实现更好的集成。
- **支付处理问题**：用户报告在 OpenRouter 上处理支付时遇到困难，特别是某些信用卡失败而其他卡可以正常使用。
   - 一位用户指出 Capital One 卡一直失败，而另一张备选卡则成功处理了支付。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://app.hyperbolic.xyz/models/deepseek-v3">Hyperbolic AI Dashboard</a>: 未找到描述</li><li><a href="https://www.litellm.ai/">LiteLLM</a>: LiteLLM 处理 100 多个 LLM 的负载均衡、故障转移和支出跟踪。全部采用 OpenAI 格式。</li><li><a href="https://openrouter.ai/api/v1">OpenRouter</a>: LLM 的统一接口。为您的 Prompt 寻找最佳模型和价格。</li><li><a href="https://status.deepseek.com/">DeepSeek Service Status</a>: 未找到描述</li><li><a href="https://github.com/lm-sys/RouteLLM">GitHub - lm-sys/RouteLLM: 一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！</a>: 一个用于服务和评估 LLM 路由器的框架 - 在不牺牲质量的情况下节省 LLM 成本！ - lm-sys/RouteLLM</li><li><a href="https://github.com/BerriAI/litellm">GitHub - BerriAI/litellm: Python SDK, 代理服务器 (LLM Gateway)，以 OpenAI 格式调用 100 多个 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq]</a>: Python SDK, 代理服务器 (LLM Gateway)，以 OpenAI 格式调用 100 多个 LLM API - [Bedrock, Azure, OpenAI, VertexAI, Cohere, Anthropic, Sagemaker, HuggingFace, Replicate, Groq] - BerriAI/litellm
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1324467935972163704)** (67 messages🔥🔥): 

> `SwarmUI vs Forge, SANA and Omnigen Models, Video Generation Models, Text Output in Images, GPU Recommendations for AI` 


- **SwarmUI 在提供 ComfyUI 性能的同时兼具简洁性**：成员们讨论了 SwarmUI 利用 ComfyUI 作为后端，在保持与 ComfyUI 相当的性能的同时，提供了一个用户友好的图形界面。
- **SANA 很快但并不总是值得占用空间**：用户讨论了 SANA 和 Omnigen 的有效性，指出虽然 SANA *非常快*，但在性能上落后于 Flux，可能不值得占用 HDD 空间。
   - 关于 Omnigen 的意见表明它*相当慢*，且生成的图像质量可能不如 SDXL。
- **视频生成的飞速进展**：LTXVideo 等模型引起了热议，据报道它在新型 GPU 上的渲染速度显著提高且没有质量损失。
   - HunyuanVideo 的改进也受到了赞扬，与之前的版本相比，它可以用更少的步数实现高效处理。
- **目前图像生成文本的最佳模型**：对于在图像中生成文本，Flux Dev 被提及为领先的开源模型，可与 Ideogramv2 和 DALL-E 3 等顶尖闭源模型相媲美。
   - 推荐的*最佳闭源模型*是 Flux 1.1 Ultra。
- **AI 工作负载的 GPU 建议**：在讨论硬件时，建议投资 RTX 系列等 GPU 以获得 AI 任务的最佳性能，特别强调等待即将推出的型号以可能降低价格。
   - 成员们建议至少配备 32GB RAM 以进行有效的图像生成，并强调了 VRAM 与 RAM 的重要性。



**Link mentioned**: <a href="https://github.com/butaixianran/Stable-Diffusion-Webui-Civitai-Helper">GitHub - butaixianran/Stable-Diffusion-Webui-Civitai-Helper: Stable Diffusion Webui Extension for Civitai, to manage your model much more easily.</a>: Stable Diffusion Webui 的 Civitai 扩展，让你更轻松地管理模型。 - butaixianran/Stable-Diffusion-Webui-Civitai-Helper

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1324528711680266331)** (21 messages🔥): 

> `RTX 50xx series VRAM limitations, Integration of SSDs with GPUs, Memory performance between VRAM and RAM, Cache hierarchy in GPUs` 


- **RTX 50xx 系列 VRAM 容量争论**：用户讨论了 **VRAM 容量限制** 可能是为了防止蚕食 NVIDIA 的产品线，暗示定价背后可能存在隐秘动机。
   - 一位用户指出，“撇开梗不谈，为什么我不能拥有这个，或者哪怕是这么一点延迟也会让它在 AI 领域变得毫无用处吗？” 突显了用户对限制的沮丧。
- **探讨 SSD 与 GPU 的集成**：一位用户回忆起 **某些 AMD 显卡** 可能集成了 SSD，但被提醒这可能使用的是 **未使用的 PCIe 通道** 而非完全集成。
   - 这引发了关于该技术与 GPU 关联的未来讨论。
- **理解 VRAM 与普通 RAM 的性能差异**：一位用户质疑在 **gh200 形式参数** 的背景下，普通 RAM 延迟慢 4 倍的说法是否正确。
   - 另一位用户解释说，虽然所有 **RTX GPU** 的 **VRAM 带宽** 峰值都在 **1TB** 左右，但卸载到 RAM 仍然面临挑战。
- **关于将 VRAM 重新分类为缓存的思考**：一位用户建议将 **20+ GB VRAM** 重命名为 **L3 Cache**，并加入更大的内存组件以优化性能。
   - 然而，有人担心如果权重仅为读取模式，可能会出现潜在的性能问题。
- **GPU 中的流水线和内存控制**：一位用户推测，有效的流水线需要控制器处理 **VRAM 与 RAM 插槽** 之间的 R/W 命令，而不像集成缓存那样独立运行。
   - 这引发了关于模型推理在 **最坏情况性能** 影响下的疑问。

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1324497407261675532)** (35 条消息🔥): 

> `Attention on Attention Weights, Quartic Transformer, Ring Attention, Higher-Order Attention, HYMBA and SWA` 


- **探索权重矩阵上的注意力 (Attention on Weight Matrices)**：成员们讨论了在注意力权重矩阵上执行注意力操作的想法，一些人提到了类似于 Higher-Order Attention 的概念，这可能会改善表示连接。
   - 一位参与者指出，在 Mamba 或 SSM Conv 论文中的应用表明，两次卷积可能对应于 Quad Attention。
- **Quartic Transformer 研究**：提到了指向 [Quartic Transformer](https://github.com/lucidrains/quartic-transformer) 的链接，该项目探索了在不考虑效率的情况下，注意力机制中各节点之间的性能。
   - 这种方法引发了关于在边图 (edge graphs)、线图 (line graphs) 和超图 (hypergraphs) 中可能存在的类似概念的讨论。
- **重温 Ring Attention 论文**：一位成员澄清说，在讨论 “Ring Attention” 时，大多数人指的是使其概念普及的第二篇论文，该论文展示了令人印象深刻的 Context Length。
   - 这引发了关于从最初提案到新模型中 Context Length 能力演进的见解。
- **HYMBA 的效率担忧**：有推测认为，在少数层中使用 Full Attention 的 HYMBA 可能与 SWA 和 SSM 等混合模型的初衷相悖。
   - 参与者辩论了通过在较少的 Token 上将 Full Attention 权衡为 Sliding Window Attention 以增强跨窗口表示的效率。
- **Pytorch Flex Attention 的挑战**：对话以对 Pytorch Flex Attention 中 Bug 的挫败感结束，这些 Bug 阻碍了一些成员对其实现的测试。
   - 尽管 torch.compile 具有潜在优势，但据观察，在与不够成熟的模型一起使用时，它经常会遇到 Bug。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.15371">DiSHA: Dimension-Sharding Adaptation with Fast Convergence and Fast Computation</a>: Low-Rank Adaptation (LoRA) 利用了 Large Language Models (LLMs) 权重更新的低内在秩，建立了 Parameter-Efficient Fine-Tuning (PEFT) 范式。然而，LoRA 仍然面临...</li><li><a href="https://en.m.wikipedia.org/wiki/Line_graph">Line graph - Wikipedia</a>: 未找到描述</li><li><a href="https://github.com/lucidrains/quartic-transformer">GitHub - lucidrains/quartic-transformer: Exploring an idea where one forgets about efficiency and carries out attention across each edge of the nodes (tokens)</a>: 探索一种不考虑效率并在节点（Token）的每条边上执行注意力的想法 - lucidrains/quartic-transformer</li><li><a href="https://huggingface.co/datasets/lennart-finke/SimpleStories">lennart-finke/SimpleStories · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2112.00578">Systematic Generalization with Edge Transformers</a>: 最近的研究表明，自然语言理解中的系统泛化对于 Transformer 和 Graph Neural Networks 等最先进的神经模型来说仍然是一个挑战。为了解决...
</li>
</ul>

</div>

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1324572958328361112)** (15 条消息🔥): 

> `2024 年 LLM 发展、文本提取工具评估、图像生成趋势、SmallThinker-3B 模型性能、OLMo2 技术报告发布` 


- **2024 年 LLM 发展继续加速**：随着即将发布的文章 [LLMs in 2024](https://simonwillison.net/2024/Dec/31/llms-in-2024/)，文中分享了对 **Large Language Models** 领域重大进展的回顾，重点介绍了过去一年中突破的障碍和新兴的指标。
   - 关键主题包括多模态能力的进步，以及由于竞争加剧导致的显著价格下降。
- **社区讨论数据提取工具**：一位成员发布了一项 [基准研究](https://cdn.discordapp.com/attachments/1075282825051385876/1324793970726928537/A_Comparative_Benchmarking_Evaluation_of_Text_Extraction_Tools.pdf)，评估了针对监管文件的文本提取工具，引发了对从复杂表格中进行有效数据提取方法的关注。
   - 社区成员提供了见解，讨论了用于文档处理的 **pdfium** 和 **tesseract** 的成功组合。
- **图像生成的模因 (Meme) 文化趋势**：2023 年 11 月出现了一种模因趋势，用户提示 ChatGPT 生成不断修改的图像，展示了这种参与式内容创作形式的有趣结果。
   - 突出的例子传达了图像的幽默演变，例如将一个普通人变成一个“兄弟 (bro)”，或者描绘神情严肃的圣诞老人。
- **推出 SmallThinker-3B 模型**：**SmallThinker-3B-preview** 模型作为一种新的微调模型推出，在各种评估中展示了相比 **Qwen2.5-3B-Instruct** 显著的基准性能提升。
   - 该模型由于其紧凑的体积，特别针对边缘部署，使其适用于资源受限的环境。
- **发布 OLMo2 技术报告**：[OLMo 2 技术报告](https://x.com/soldni/status/1875266934943649808?s=46) 已发布，在 50 多页的详细内容中深入探讨了 LLM 开发流水线的四个关键组成部分。
   - 该报告旨在提供对 LLM 开发领域内基本方法论和实践的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.val.town/blog/fast-follow/">我们在复制所有最佳代码助手中学到的东西</a>：从 GitHub Copilot 到 ChatGPT 再到 Claude Artifacts，Val Town 如何借鉴了所有代码生成工具的精华</li><li><a href="https://simonwillison.net/2024/Dec/31/llms-in-2024/">2024 年我们学到的关于 LLM 的事</a>：2024 年 Large Language Models 领域发生了很多事情。以下是对过去一年该领域发现的回顾……</li><li><a href="https://minimaxir.com/2025/01/write-better-code/">如果你一直要求 LLM “写更好的代码”，它们能写出更好的代码吗？</a>：大多数程序员希望 AI 写代码更快：我希望 AI 写出更快的代码。</li><li><a href="http://openlayer.com?">Openlayer：企业级 AI 质量、评估和监控</a>：Openlayer 帮助您测试和监控高质量的 AI 系统。</li><li><a href="https://huggingface.co/PowerInfer/SmallThinker-3B-Preview">PowerInfer/SmallThinker-3B-Preview · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1874868847754580431">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：chat，这是真的吗？smol QwQ? https://huggingface.co/PowerInfer/SmallThinker-3B-Preview</li><li><a href="https://x.com/WolframRvnwlf/status/1874889165919384057">Wolfram Ravenwolf 🐺🐦‍⬛ (@WolframRvnwlf) 的推文</a>：新的一年，新的基准测试！测试了一些在我最新报告之后发布的新模型（DeepSeek-V3, QVQ-72B-Preview, Falcon3 10B），以及一些“老”模型（Llama 3.3 70B Instruct, Llama 3.1 Nemo...</li><li><a href="https://x.com/soldni/status/1875266934943649808?s=46">Luca Soldaini 🎀 (@soldni) 的推文</a>：OLMo 2 技术报告已发布。我们在这份报告中深入细节，用 50 多页篇幅介绍了 LLM 开发流水线的 4 个关键组件：
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1324525366550728715)** (4 messages): 

> `AI Engineer Summit, AI Engineer World's Fair, Understanding Transformers` 


- **AI Engineer Summit 官宣！**: 为我们下一场仅限受邀参加的 [AI Engineer Summit](https://www.latent.space/p/2025-summit) 做好准备！第一届峰会以 **10:1 的申请比例** 售罄，我们很高兴能再次举办。
   - 我们之前的 **多轨道 World's Fair** 售出了 **3000 个席位**，并在 YouTube 上获得了 **超过 100 万次观看**，展示了 AI Engineering 领域的顶级大咖。
- **Latent Space 特约投稿机会**: 发布了合作征集；任何有兴趣撰写优秀的 Transformers 详解文章的人都可以投稿！这是一个在社区内深度参与的绝佳机会。
   - 欲了解更多信息，请查看[此处](https://discord.com/channels/822583790773862470/1323930993786228858/1324157846031700061)的讨论链接。
- **在 Latent Space 理解 Transformers**: [Understanding Transformers](https://x.com/sannykimchi/status/1176517584319127553) 提供了一份极具洞察力的资源列表，用于学习 Transformers 的运作机制，从 **Self-attention** 到 **Positional encodings**。最近的进展如 **BERT**、**XLNet** 和 **GPT-2** 都深受这种架构的影响。
   - *Sannykimchi* 分享了一个简洁的概念学习路线图，以便有效地深入研究 Transformers，满足那些渴望掌握这一重要主题的人。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/2025-summit">官宣纽约 AI Engineer Summit：全力投入 Agent Engineering + Leadership</a>: 官宣第二届 AI Engineer Summit 的主题。立即申请！</li><li><a href="https://x.com/sannykimchi/status/1176517584319127553">来自 sanny (@sannykimchi) 的推文</a>: Transformers 引领了 #NLProc 最近的一波进展，如 BERT、XLNet 和 GPT-2，所以这里有一份我认为对学习 Transformers 工作原理很有帮助的资源列表💻，从 Self-attention 到...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1324844996934766783)** (31 messages🔥): 

> `Discord Bot Building, Obsidian Tool Discussions, Screen Sharing Issues, Webhook Limitations` 


- **Discord Bot 创意引发关注**: 成员们讨论了构建 **Discord bot** 的可能性，表达了协作完成项目或从零开始的共同兴趣。一位成员提到了他们自己的 **yolo Discord** 以及为其生成 Endpoint 的能力。
- **Obsidian 工具交流**: 一位成员表示自己是 **Obsidian** 初学者，并表示有兴趣与任何愿意引导对话的人交流。这表明了在掌握该工具方面寻求指导和协作的愿望。
- **屏幕共享时的回声问题**: 参与者在屏幕共享期间遇到了 **Echo**（回声）问题，有人建议静音屏幕共享以缓解该问题。据观察，屏幕共享正在分享 **计算机音频**，这加剧了回声效果。
- **Discord 中 Webhook 的局限性**: 一位成员指出了 **Webhook** 的局限性，强调虽然它们可以发布消息，但无法读取数据。这引起了人们对使用 Webhook 实现更具交互性的 Bot 功能所面临挑战的关注。
- **构建新 App 时的挫败感**: 一位成员对启动新项目的复杂性感到沮丧，表示在 App 构建过程中感到大脑疲劳。这种情绪引起了其他人的共鸣，表明这是参与者面临的共同挑战。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1324775554041380916)** (2 messages): 

> `GEMM performance on GPU, Identifying inefficiencies in GEMM, Optimizing GEMM computations, Suggestions for improvements in GEMM` 


- **分享 GEMM 性能见解**：一名成员强调了 GPU 上实际 **GEMM 性能** 与理论性能之间的差距，这影响了对最佳性能的评估。
   - 他们还引用了关于检测 GEMM 计算中低效问题的文章，可在此处 [阅读](https://yywangcs.notion.site/Inconsistency-in-GEMM-Performance-16efc9f5d80580838090dded05493014)，并在 [Twitter](https://x.com/YyWangCS17122/status/1874856334845489191) 上进行了总结。
- **通过社区反馈优化 GEMM**：有人建议，在论坛上宣布已达到的**最佳性能**可能会促使社区成员提供更优化的实现方案。
   - 这种方法有助于发现之前可能未曾考虑过的解决方案或改进措施。



**提到的链接**：<a href="https://x.com/YyWangCS17122/status/1874856334845489191).">来自 YyWangCS (@YyWangCS17122) 的推文</a>：矩阵乘法 (GEMM) 性能对深度学习至关重要。我写了一篇关于如何自动检测 GPU 上 GEMM 计算低效问题的文章。https://yywangcs.n...

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1324559433686192158)** (20 messages🔥): 

> `Triton kernel performance, Matrix multiplication issues, Testing kernel equivalence, Pointer loading in Triton, Floating point operation ordering` 


- **Triton kernel 性能调试**：一名成员发现设置 `os.environ['TRITON_INTERPRET'] = '1'` 会导致性能变慢，而移除该设置则能提高 kernel 执行速度。
   - 另一位用户确认了移除 `triton_interpret` 的好处，从而解决了实现矩阵乘法时的问题。
- **矩阵乘法 kernel 调用问题**：一位用户在尝试在一个 kernel 函数中调用另一个时遇到挑战，在调整安装和设置后解决了该问题。
   - 建议将 batch size 增加到至少 **16**，并对较小的矩阵使用 padding。
- **在大输入下测试 kernel 等效性**：关于 Triton 在使用较大输入值时的 kernel 等效性出现了疑问，因为结果与使用 `torch.randn` 创建的较小随机值有显著差异。
   - 另一位用户建议调整 `torch.allclose` 中的 `atol` 参数，以处理浮点数差异。
- **Triton 中的指针加载差异**：一名成员注意到在使用偏移量加载数据时出现了意外的结束索引，由于可视化结果不同而导致困惑。
   - 他们分享了输出结果和图像以供深入分析，突显了指针操作结果的复杂性。
- **浮点运算顺序的影响**：强调了运算顺序会显著影响浮点计算的结果，因为加法和乘法在浮点运算中不满足交换律。
   - 这指出了在比较 Triton 和 Torch 实现的输出时，潜在的差异来源。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1uTsmN2-U3TRHi0mhJ2Dp0nZh-yM_gt4_?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1uTsmN2-U3TRHi0mhJ2Dp0nZh-yM_gt4_?usp=shar">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1324687255633395733)** (3 messages): 

> `Dynamic br/bc values in Flash Attention, Fixed br/bc values performance, Compiling versions for selection` 


- **探索 Flash Attention 中的动态 br/bc 值**：一名成员询问了动态确定 Flash Attention 中 **br/bc** 值的方法，而不是使用固定值。
   - *这种方法可能会提高灵活性，但可能需要更复杂的实现。*
- **固定 br/bc 值提供更高速度**：另一名成员指出，固定的 **br/bc** 值速度明显更快，称其快了 **10 万亿倍**。
   - *这强调了性能调优中速度与动态适应性之间的权衡。*
- **动态选择的编译器解决方案**：建议如果必须进行动态选择，可以编译不同的版本并从中选择，类似于 **Flash Attention** 的方法。
   - *这可以在性能和灵活性之间提供平衡，允许根据特定需求量身定制方案。*


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1324493313260388362)** (11 条消息🔥): 

> `模型级编译 (Model Level Compile)、内存分析问题、Inductor 缓存性能、Flex Attention 复杂性、梯度与激活管理` 


- **带有选择性禁用的模型级编译**：一位用户考虑使用模型级编译，同时根据 [文档](https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html) 详述，使用 `torch.compiler.disable` 标记跳过部分内容以提升性能。另一位用户指出，这是目前管理模型编译的一种合理方法。
   - 他们讨论了对通用操作使用编译装饰器，而对成本较高的操作禁用编译，并提到对常用形状（popular shapes）使用全模型编译。
- **Inductor 缓存性能仍然缓慢**：一位用户分享说，通过 Redis 使用远程 Inductor 缓存有助于减少预热时间，但仍然存在较长的延迟，即使在缓存命中的情况下也长达 **5 分钟**。这种延迟归因于加载已编译 Kernel 的过程，引发了对性能的担忧。
- **调查线性层的内存分配**：一位用户观察到其模型中的线性层正在调用 `cudaFree`，尽管预期 Torch 的缓存分配器（caching allocator）会进行重用，但这引发了关于其内存分配行为的疑问。他们提供的分析追踪（profiling traces）表明输出分配未按预期重用，并观察到输出始终为 **128 MiB**。
   - 另一位参与者询问这是否是在启用梯度的情况下发生的，并指出反向传播可能需要保存激活值（activations），而原用户确认 `tensor.requires_grad` 已设置为 **False**，且没有进行任何反向存储。



**提及的链接**：<a href="https://pytorch.org/docs/stable/torch.compiler_fine_grain_apis.html">TorchDynamo APIs for fine-grained tracing &mdash; PyTorch 2.5 documentation</a>：未找到描述

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1324488328804040716)** (1 条消息): 

> `P-1 AI、研究工程师职位、通用工程人工智能 (Artificial General Engineering)` 


- **P-1 AI 为隐身模式初创公司寻求人才**：P-1 AI 正在积极招聘多个职位，重点是开发一种**通用工程（超级）智能 (artificial general engineering (super)intelligence)**，以提高物理系统设计的效率。更多详情请见其 [招聘列表](https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6)。
   - 团队成员来自 **ex-Google DeepMind、Microsoft、Airbus 和 DARPA** 的研究人员，并拥有硅谷投资者的强大支持。
- **研究工程师职位需要高级技能**：作为**研究工程师**，候选人将使用具有定量推理能力的**多模态 (multimodal)** LLM 和 GNN 构建并部署 AI 系统。该职位旨在物理系统设计方面实现**前所未有的性能**。
   - 该职位强调应对**此前无法完成的任务**，展示了 P-1 AI 团队的雄心目标。



**提及的链接**：<a href="https://jobs.lever.co/P-1AI/84ae5c01-9160-44a5-a7c8-a107e645f0a6">P-1 AI - Staff Research Engineer</a>：未找到描述

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1324726229256900648)** (5 messages): 

> `GPU Upgrade Considerations, Federated/Gossip Learning Resources, CUDA Learning, VRAM and Global Memory Importance, Upcoming Event on Turing Tensor Cores` 


- **考虑 GPU 升级**：一位成员正在考虑从 **AMD RX 480** 升级到 **二手 RTX 2080 Ti** 或 **3090**，优先考虑 CUDA 学习和本地模型。
   - 有人指出，虽然 **RTX 3090** 提供了更多优势，但它可能需要升级 PSU 和主板，这使得决策变得复杂。
- **寻求 Federated Learning 资源**：一位成员表示难以找到关于 **Federated** 和 **Gossip Learning** 的学习资源，称他们找到了 **DeepMind 关于 Federated Learning 的课程**，但没有找到关于 Gossip Learning 的可靠来源。
   - 他们正在寻求建议，以帮助建立对这两个主题的基础理解。
- **CUDA 学习对升级 GPU 的重要性**：另一位成员建议，对于 **CUDA** 学习，**2080 GPU** 的 Turing 架构已经足够，特别是考虑到其 **7.5** 的 compute capability。
   - 他们建议，对于其他用例，更大的 **VRAM** 和 global memory 可能使 **3090 更合适**，尽管其成本更高。
- **即将举行的聚焦 Turing Tensor Cores 的活动**：一位成员提到 **2 月 15 日** 将举行一场活动，届时服务器的一位顶尖高手将讲解 **Turing Tensor Cores** 的使用。
   - 这可能为那些有兴趣了解 GPU 技术进步的人提供重要的见解。
- **近期入职与社区贡献**：有一条关于某人入职 **Together AI** 的记录，表明了社区的成长。
   - 此外，还分享了一个 **Karpathy** 讨论 **llm.c** 的视频，可能为小组提供宝贵的见解。



**Link mentioned**: <a href="https://youtu.be/aR6CzM0x-g0?feature=shared&t=630)"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1324778323070419014)** (3 messages): 

> `Learning Cuda, Mental Health Awareness, Felix Hill Tribute` 


- **学习 CUDA 是必须的！**：一位成员表达了对*学习 CUDA* 必要性的兴奋，并分享了一个强调其重要性的 [YouTube 视频](https://www.youtube.com/shorts/gFva8uNJPNg)。
   - 该视频对于那些对 GPU 编程感兴趣的人来说，可能是一个引人入胜的 CUDA 入门介绍。
- **悼念 Felix Hill**：一位成员分享了 **Felix Hill 去世** 的噩耗，对这一损失表示深切哀悼。
   - 这凸显了社区人物的影响力以及成员们共同的悲痛感。
- **优先考虑心理健康**：一位成员强调 *没有什么比你自己的心理健康更重要*，敦促他人将其放在首位。
   - 这一评论在快节奏的技术世界中提醒了自我关怀的重要性。



**Link mentioned**: <a href="https://www.youtube.com/shorts/gFva8uNJPNg"> - YouTube</a>: no description found

  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1324810302067314718)** (2 messages): 

> `TorchTitan, MFU vs HFU, Lecture 39 on YouTube` 


- **TorchTitan 博客文章获得好评**：一位成员阅读了一篇讨论 **TorchTitan** 的博客文章，并认为其非常出色。
   - 他们强调了该内容在理解 **Model FLOP/S Utilization** 方面的重要性。
- **Lecture 39 中解释了 MFU vs HFU**：讨论强调了 **MFU** (Model FLOP/S Utilization) 有时是如何与 **HFU** (Hardware FLOP/S Utilization) 进行对比计算的。
   - [Lecture 39](https://youtu.be/VYWRjcUqW6w?feature=shared&t=2606) 的相关部分可以在 **43:26-47:29** 之间找到，提供了有用的见解。



**Link mentioned**: <a href="https://youtu.be/VYWRjcUqW6w?feature=shared&t=2606)"> - YouTube</a>: no description found

  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1324770409815736350)** (1 messages): 

> `Transduction Goals, Prompt Optimization, Training Procedure` 


- **追求 Transduction 的清晰度**：Transduction 的目标是将 Prompt 从 `[examples] | test input` 转换为 `转换过程的自然语言描述 | test output`。
   - 这种方法旨在阐明输入与输出之间的关系，同时简化处理过程。
- **关于 Prompt 优化的咨询**：有人提出了关于 **prompt optimization** 现有工作的问题，特别是针对 Transduction 任务。
   - 鼓励成员分享该领域任何已知的进展。
- **为 ARC 问题构建数据集**：该提案涉及创建一个数据集，格式为 `example 1 | example 2 | example 3 inp | trainable prompt | example 3 output`，涵盖每个 ARC 问题。
   - 目标是利用从最后一个示例的输出计算出的 Loss，对 `trainable prompt` 进行反向传播 (backpropagation)。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1324539530874851410)** (14 messages🔥): 

> `Fine-tuning LLMs with LoRA, HuggingFace TRL vs. alternatives, MoE routing techniques, Documentation availability` 


- **LLM 微调工具的不同体验**：用户分享了使用 LLM 微调库的混合体验，其中 **Unsloth** 被认为较难，而 **TRL** 因其详尽的文档和易用性受到称赞。
   - 一位成员指出 **Axolotl** 很受欢迎，而 **Llama Factory** 可能缺乏足够的英文文档。
- **GitHub Stars 并不代表一切**：讨论围绕 **GitHub stars** 对 LLM 工具的重要性展开，有评论认为 Stars 可能无法准确反映可用性或支持情况。
   - 尽管 **Unsloth** 以 **20k stars** 领先，用户仍建议利用 **HuggingFace 的 Transformers/PEFT** 来进行流线化的 LoRA 微调。
- **微调资源推荐**：一位用户分享了一个名为 [LLM Fine-Tuning Library Comparison](https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit) 的链接，该链接对比了各种 LLM 微调库并提供了见解。
   - 对话强调了某些工具缺乏全面的文档，突显了寻找可靠资源的重要性。
- **Mixture of Experts (MoE) 的门控网络 (Gating Networks)**：一位用户寻求关于讨论 MoE **gating networks** 论文的建议，旨在了解 **Deepseek v3** 中使用的路由方法。
   - 另一位成员推荐了 **OLMoE** 论文，该论文以其对 MoE 的研究而闻名，尽管其专家数量较少。



**提到的链接**：<a href="https://docs.google.com/document/d/1k0E2XCuqJDGiD6IsP2rb6s3D1Iu96gro9WRCSajl0zs/edit">LLM Fine-Tuning Library Comparison</a>：使用 LoRA 微调 LLM：五种流行库的对比分析。大语言模型 (LLMs) 的兴起开启了自然语言处理的新纪元，实现了显著的进步...

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1324519239574355999)** (18 messages🔥): 

> `Post-training 教程内容, 录制演讲的质量, AI Safety Institute 的活动, LinkedIn 动态, Chatbotarena 图表维护` 


- **关于分享教程内容的辩论**：一场关于是否通过 YouTube 和 Interconnects 分享 Post-training 教程的讨论展开了，人们担心这会向拥有 **2 万** 名成员的庞大群体发送垃圾信息。
   - 另一位成员指出，**预录制的演讲**一直受到高度评价，建议将其分享给更广泛的受众可能会大有裨益。
- **预录制演讲的质量感知**：成员们对高质量内容表示赞赏，称录制的演讲已成为教育产品的亮点。
   - 一位成员提到，只要内容对增长有积极贡献，就应该被分享。
- **英国 AI Safety Institute 被称为情报人员**：有人发表了一个幽默的评论，质疑为什么英国 **AI Safety Institute** 的成员不被视为情报人员，因为他们经常与 AI 研究人员进行社交活动。
   - 该帖子揭示了这种非正式社交网络对于向英国政府共享信息所具有的潜在影响。
- **AI 圈的 LinkedIn 动态**：一位成员调侃了 **LinkedIn 竞赛**的竞争本质，暗示了 AI 专业人士中盛行的社交策略。
   - 讨论附带了一张截图，为 AI 社区职业行为的对话增加了视觉重点。
- **Chatbotarena 图表维护担忧**：一位成员对 **Chatbotarena** 图表未得到维护表示惊讶，认为它具有重大价值。
   - 该评论附带一张图片，进一步说明了讨论的背景。



**提到的链接**：<a href="https://x.com/typewriters/status/1874924700398436450">来自 Lauren Wagner (@typewriters) 的推文</a>：OH：我不明白为什么英国 AI Safety Institute 的人不被视为情报人员。他们从伦敦搬到了旧金山，举办派对，让喝醉的研究人员开口说话，然后发送...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1324684692041109536)** (4 messages): 

> `The Bittersweet Lesson, Felix 的贡献, Google 账户担忧` 


- **反思《苦涩的教训》（The Bittersweet Lesson）**：一位成员分享了一个 [Google 文档](https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing) 链接，讨论了他们对 **The Bittersweet Lesson** 的独到见解。
   - 遗憾的是，该文章的作者最近去世了，引发了人们对其影响力作品的反思。
- **保留 Felix 的遗产**：成员们对 **Felix** 最近的离世表示哀悼，称赞他对社区做出的精彩贡献。
   - 一位成员提到存有作品的 **PDF 备份**，强调了保持这些贡献可访问性的重要性。
- **对 Google 账户寿命的担忧**：有人指出，访问文档可能会有问题，因为 Google 账户最终可能会被注销。
   - 成员们意识到，如果账户没有得到妥善保存，可能会面临失去重要文档访问权限的风险。



**提到的链接**：<a href="https://docs.google.com/document/d/1MPqtT_1vQ-73j796tf7sXIZKCRcIfUD0cVU_UbPXnUU/edit?usp=sharing">The Bittersweet Lesson</a>：The Bittersweet Lesson 😆 Transformer 中归纳偏置（Inductive bias）的奇特案例，Felix Hill，2024 年 10 月 21 日。你还记得几年前归纳偏置的概念在机器学习中处于核心地位的时候吗...

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1324599532914343987)** (6 messages): 

> `SnailBot 新闻, SnailBot 的娱乐价值` 


- **SnailBot 的性能遭到幽默批评**：一位成员幽默地评论说 SnailBot 是 *一只缓慢的蜗牛*，强调了它的性能。
   - 评论中包含了对其速度的怀疑，一位成员惊呼：*天哪（good lord）*。
- **SnailBot 的娱乐因素**：成员们对 SnailBot 的滑稽动作表示好笑，指出 *它太有娱乐性了*。
   - 另一位成员评论说它的名字非常贴切，名副其实。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1324470121040838798)** (19 messages🔥): 

> `FLUX.1 [dev], Minecraft 图像过滤器, 社区动态, 群组使用, AGI 讨论` 


- **FLUX.1 [dev] 能力揭秘**：成员们讨论了 **FLUX.1 [dev]** 的特性，这是一个拥有 **120 亿参数** 的 rectified flow transformer，可以根据文本生成图像。更多详情请访问 [博客文章](https://blackforestlabs.ai/announcing-black-forest-labs/)。
   - 关键特性包括仅次于 **FLUX.1 [pro]** 的输出质量，以及用于科学研究的开放权重。
- **Minecraft 图像过滤器故障**：讨论中提到 **Minecraft** 图像提示词因“Goofy”或“Minecraft”等关键词而无法通过。一个富有创意的提示词奏效了，生成了一个包含橡皮鸡和香蕉等元素的搞怪服务器图标。
   - 成员们建议寻找潜在问题词汇的替代拼写，以绕过过滤器。
- **社区氛围评论**：一位成员提到，服务器感觉被 **AGI 崇拜者** 占领了，而那些需要 OpenAI 支持的人变成了少数。他们注意到**孤独的个体**和救世主情结混合在一起，主导了话语权。
   - 这指向了社区内部不断变化的动态，一些成员对此感到担忧。
- **群组使用说明**：一位用户询问如何有效利用该群组，促使另一位成员建议阅读群组说明。分享了一张图片，可能详细说明了使用指南。
   - 这反映了为帮助成员适应群组互动而进行的持续努力。
- **GPT 生成描述的幽默**：有人幽默地建议使用“Delve”一词来与群组互动，引发了笑声。成员们注意到，群组描述据称是由 **GPT-3** 编写的，这充满了讽刺意味。
   - 这种轻松的评论强调了 AI 与人类互动的融合。



**提到的链接**：<a href="https://huggingface.co/black-forest-labs/FLUX.1-dev">black-forest-labs/FLUX.1-dev · Hugging Face</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1324534437953929216)** (5 messages): 

> `ChatGPT 搜索结果, YouTube GPTs 功能, 跨频道发布礼仪` 


- **ChatGPT 搜索的可靠性**：一位用户询问是否可以依赖 **ChatGPT** 获取搜索和网络相关结果。
   - 这引发了关于与其他搜索工具（如 **Perplexity**）进行**比较**的讨论。
- **YouTube GPTs 在检索方面表现不佳**：用户对 **YouTube GPTs** 无法分析或检索有用信息表示担忧。
   - 这种情绪表明用户对这些模型的功能感到沮丧。
- **跨频道发布（Cross posting）的顾虑**：一位成员警告不要在多个频道重复发布内容，指出这可能被视为**垃圾信息（spamming）**。
   - 这对于维护聊天环境的礼仪非常重要。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

castilla99_87524: 在 Sora 中制作不同场景时，如何保持角色的一致性？
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/)** (1 messages): 

castilla99_87524: 在 Sora 中制作不同场景时，如何保持角色的一致性？
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1324545258616389653)** (7 messages): 

> `新年祝福, Azure 上的 Rerank` 


- **新年快乐！**：成员们对新的一年感到兴奋，表达了对未来**幸福**、**成功**和**美好回忆**的希望。
   - *Cohere* 已准备好迎接各种可能性，一位成员宣称：*“这就是属于我们的一年！”*
- **对 Rerank-3.5 的期待**：在询问 **Azure** 上 **Rerank** 的消息时，一位成员提到了即将推出的 **rerank-3.5**，它应该很快就会发布。
   - 另一位成员邀请大家讨论使用案例，问道：*“到目前为止你觉得它怎么样？”*


  

---

### **Cohere ▷ #[rules](https://discord.com/channels/954421988141711382/954422415016996864/1324618481936891958)** (1 条消息): 

> `服务器指南、推广规则、垃圾信息政策、商业活动限制` 


- **在交流中保持 PG 标准**：所有成员必须确保消息尊重他人，且不含任何有害或不当内容，因为这是一个 PG 级别的服务器。
   - 不遵守规定可能会导致警告或被移出服务器。
- **鼓励使用英语**：要求成员主要使用英语进行交流，以便用户之间更好地理解。
   - 该指南有助于保持清晰度并增强社区内的参与度。
- **允许有限的推广**：广告只能在指定频道发布，例如在专注于 Cohere 相关内容的 <#1218409701339828245> 中分享项目。
   - 这有助于减少混乱，并使频道内容与社区利益保持相关。
- **严格的垃圾信息规定**：禁止在多个频道发布重复消息，过度发送垃圾信息将导致消息被删除。
   - 也不鼓励成员在不必要的情况下 ping 他人，以维持良好的沟通环境。
- **禁止招聘活动**：__本服务器严禁发布职位信息、招聘活动以及任何与就业机会相关的广告。__
   - 禁止通过 DM 向团队成员询问工作机会，以保持对社区讨论的关注。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1324508757509672981)** (2 条消息): 

> `Command-R 功能、Command-R 的问题、恢复进程` 


- **对 Command-R 方法论的困惑**：一位成员解释说，**Command-R** 的工作方式涉及在以恢复格式重写之前详尽地写出内容。
   - 他们指出，虽然命令的初始部分是可以理解的，但 **'-R'** 似乎引起了混乱并中断了进程。
- **使用 Command-R 遇到的问题**：成员们讨论了使用 **Command-R** 时面临的问题，强调了执行 **'-R'** 功能时的困难。
   - 普遍观点认为需要进行改进，以增强该命令的清晰度和有效性。


  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1324493791180230756)** (6 条消息): 

> `提高 Embedding 速率限制、Cohere API 速率限制` 


- **申请提高 Embedding 速率限制**：一位用户询问如何申请提高其 **embeddings 速率限制**。
   - 针对此类咨询，可以联系 **Cohere 支持团队**，邮箱为 support@cohere.com。
- **Cohere API 速率限制概览**：Cohere API 为各种端点定义了速率限制，其中 **Embed** 端点在测试期间允许 **100/min**，在生产环境中允许 **2,000/min**。
   - 此外，使用测试密钥时，所有端点的每月调用上限为 **1,000 次**，确保开发者能够高效管理其使用情况。


  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1324731503170551869)** (10 messages🔥): 

> `Torchtune Benchmarking, Chunked Cross Entropy Implementation, Memory Gains during Compilation` 


- **Torchtune 基准测试获得关注**：越来越多的 **papers 和 models** 正在采用 Torchtune 进行 Benchmarking，强调了其作为 AI 模型性能评估工具的实用性。
   - 一位成员指出，展示 Torchtune 提供的 **alternative evaluation methods**（替代评估方法）可能会大有裨益。
- **Chunked Cross Entropy PR 讨论**：讨论了由 *felipemello1* 提交的 [Chunked cross entropy](https://github.com/pytorch/torchtune/pull/1390)，强调其目的是通过优化计算过程来减少内存占用。
   - 成员们分享了在输出投影（output projection）后应用分块（chunking）进行编译时，**memory gains**（内存收益）减少的相关结果。
- **实现变体与比较**：一位成员分享了一个使用 *log_softmax* 代替 `F.cross_entropy` 的变体，在内存占用可控的情况下获得了良好的性能。
   - 这种方法允许完整的 Compilation 而不会中断，并促使进一步探索 **Torchtune 内部的 Benchmarking**。
- **Wandb Profiling 与内存优化**：成员们讨论了使用 **Wandb** 对实现进行 Profiling 的可能性，以评估相比标准方法的性能提升。
   - 分享了关于 **Torch** 在 Cross-entropy 计算期间最小化内存占用所做努力的见解。
- **性能改进机会**：一位成员指出 Torchtune 的 Transformer 实现中存在潜在的性能优化空间，并引用了代码链接作为可探索的领域。
   - 社区正在考虑如何将 **chunked_nll** 集成到 Compilation 过程中以提高效率。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/blob/main/torchtune/modules/transformer.py#L482.">torchtune/torchtune/modules/transformer.py at main · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/1390">chunked cross entropy by felipemello1 · Pull Request #1390 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档还是其他（请在此处添加）机制：Chunked cross entropy 通过处理...来减少内存。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1324473445672550505)** (1 messages): 

> `PyTorch Torchtune Bug, Flex Attention Compilation, Kernel Finding` 


- **A6000 上 PyTorch Torchtune Bug 的解决方法**：一位用户分享了在使用 **A6000** 时遇到 PyTorch Torchtune 库 Bug 的经验，并成功通过 `torch.compile()` 设置 `flex_attention_compiled` 来找到可用的 Kernel。
   - 他们建议在 Attention 调度（dispatch）重构之前，使用 `mode=os.getenv("TORCHTUNE_FLEX_MODE", None)` 可能提供临时解决方案，并建议在 **A6000** 上进行测试。
- **潜在的环境变量解决方案**：为了避免在 `packed=True` 时自动默认为 Flex，一位成员提议允许用户设置环境变量来决定 **Torchtune** 库中的模式。
   - 如果用户希望避免从源码安装，这种方法可能会有所帮助，但需要在 **A6000** 等硬件上进行验证。
- **2.6.0 版本中 Bug 修复的不确定性**：关于上述 Bug 是否已在 PyTorch Torchtune **2.6.0** 版本中解决存在一些不确定性，因为关于分支切割（branch cutting）的讨论已经发生。
   - 成员们对修复状态表示担忧，暗示在得出结论前需要进一步测试。



**提及的链接**：<a href="https://github.com/pytorch/torchtune/issues/2218.">Issues · pytorch/torchtune</a>：PyTorch 原生训练后库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。

  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1324784792096608318)** (1 messages): 

> `Invoice processing agent, LlamaParse, Agentic workflow, Spend categories, Cost centers` 


- **构建实用的发票处理 Agent**：@ravithejads 最近发布的一个 notebook 展示了如何从零开始构建一个**发票处理 Agent**，该 Agent 可以自动提取发票行项目，并使用**支出类别**和**成本中心**对其进行丰富。
   - 该 Agent 在其工作流中使用了 [LlamaParse](https://twitter.com/llama_index/status/1875225903346909256)，并提供了关于自动化发票处理流程的第一手见解。
- **使用 LlamaParse 实现发票自动化**：该 notebook 展示了 **LlamaParse** 在创建简化**发票处理**工作流方面的实用性。
   - 通过利用自动化，用户可以显著减少手动错误并提高处理财务文档的效率。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1324800166875762758)** (9 messages🔥): 

> `Dataset storage options, Query fusion for retrievers, JSON advantages, Compression techniques for data, Using SQL or NoSQL for datasets` 


- **探索数据集存储选项**：成员们讨论了他们首选的 LLM 评估数据集存储解决方案，提到了 **S3**、**Git LFS** 和 **Hugging Face datasets**。
   - 有人指出，其他选项还包括任何 **SQL** 或 **NoSQL** 数据库，具体取决于数据集的内容。
- **JSON 作为简单的存储格式**：一位成员表示，如果使用 **S3** 或 **LFS**，他们通常将数据存储为 **JSON blob**，如果数据量很大则可以进行压缩。
   - 他们还提到，将数据存储在 JSON 中有助于降低复杂性，并允许与 **Pydantic models** 轻松集成。
- **对查询融合（Query fusion）结果的担忧**：一位成员分享了在使用查询融合设置 **2 个 vector embedding retrievers** 和 **2 个 BM25 retrievers** 时难以获得理想结果的问题。
   - 他们寻求关于如何提高这种检索器组合**响应质量**的建议。
- **轻量级数据集建议使用 JSON + Git**：在针对数据集存储的讨论中，一位成员表示，对于较小的数据集，他们会坚持使用 **JSON 和 Git**，直到需要更强大的解决方案为止。
   - 这强调了在数据集存储和管理中从简单开始的实用性。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1324553118117138494)** (8 messages🔥): 

> `Open Interpreter functionality, Open-source contributions, Installation Steps for Open Interpreter, Web WhatsApp messaging, Trading clicker execution mode` 


- **Open Interpreter 似乎有所欠缺**：用户表示担心 Open Interpreter 1.0 的表现似乎比经典版本更差，特别指出其无法运行代码以及文本格式损坏的问题。
   - 他们还指出缺少网页浏览和搜索工具，令人感到失望。
- **开源贡献获得认可**：scalenow AI 社区认可了开源社区的关键作用，强调了像 [OpenInterpreter](https://github.com/KillianLucas/open-interpreter) 这样的工具对功能增强的贡献。
   - 他们赞扬了 **OpenProject**、**Rasa** 和 **Kotaemon** 等项目，展示了它们在提升服务能力方面的重要性。
- **精简的安装流程**：分享了 Open Interpreter 的安装步骤，包括适用于 Mac、Windows 和 Linux 的单行命令，可无缝自动化设置过程。
   - 用户可以在安装后访问 Web UI，从而更轻松地进行交互和命令执行。
- **Web WhatsApp 消息传递的乐趣**：一位用户幽默地记录了他们通过 Web WhatsApp 进行消息传递的交互，并对这种体验表示有趣。
   - 这引发了关于在日常沟通中使用该技术的轻松交流。
- **寻找常驻的交易点击器**：一位用户正在寻找一种交易点击器，该点击器需要“始终激活的 OS 执行模式”才能有效运行。
   - 这一需求表明需要一种能够持续执行命令而不中断的解决方案。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/)** (1 messages): 

rd4com: 🥳 新年快乐！！
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1324494784525500477)** (6 条消息): 

> `Linked List 实现, 构建 CLI 和 TUI 工具, AST 和 Index-Style 树, Mojo 调试问题` 


- **寻求 Linked List 实现**：一名成员请求一个适用于 nightly 版本的 Linked List 实现，表示希望直接进行实验，而不想费力亲自动手编写。
   - 这突显了社区对现成实现方案的共同需求，以便进行快速原型开发。
- **Toasty 的 CLI 和 TUI 项目**：Toasty 幽默地分享了他们正在通过开发专用库来学习构建 CLI 和 TUI 工具。
   - 另一名成员建议，这可能会促成一个为 Mojo 打造的新 Shell，类似于现有的 bash 和 zsh。
- **AST 和 Index-Style 树的开发**：一名成员报告了在实现 Index-Style 树和图方面的进展，并提到了使用的具体文件和结构，如 `RootAstNode` 和 `DisplayAstNode`。
   - 他们在调试过程中遇到了 Segmentation faults，特别是在处理 `DisplayAstNode` 时，这表明递归类型（recursive types）可能存在潜在问题。
- **Mojo 调试问题的详情**：讨论涉及一个 [GitHub issue](https://github.com/modularml/mojo/issues/3917)，其中一名成员在使用 Mojo 调试器时遇到了 Seg faults，而正常运行脚本时则没有问题。
   - 该问题强调了开发递归类型时面临的挑战，反映了在 Mojo 中调试复杂数据结构的复杂性。



**提到的链接**：<a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>：Bug 描述：使用调试器运行 Mojo 脚本会发生 Seg faults，而常规运行 Mojo 脚本则可以运行完毕（尽管作者注意到常规脚本中也存在奇怪的行为...）

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1324467766148857936)** (5 条消息): 

> `证书发放, 2024 秋季课程报名, 2025 春季课程注册` 


- **证书预计于 1 月下旬发放**：根据成员更新的消息，证书将大约在 **1 月底** 发放。
   - 一名成员提到，在所有证书发送完毕后，请 *保持关注* 进一步的公告。
- **2024 秋季课程报名已截止**：经成员确认，获得 **Fall 2024** 课程证书的机会已经结束。
   - 现在鼓励有意向的学生参加 **Spring 2025** 课程，聊天中已提供注册表单。



**提到的链接**：<a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>：MOOC，2025 春季

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1324497467181498460)** (4 条消息): 

> `GraphRAG 实现, Donor's Game 模拟, DSPy 策略更新` 


- **使用 GraphRAG 进行实体提取**：一位用户询问是否使用了特定的 GraphRAG 实现，随后得知 **Gemini** 修改了默认的 Prompts，以便提取与代码库相关的实体。
   - *这表明通过定制化方法可以增强提取过程。*
- **使用 DSPy 模拟 Donor's Game**：一名成员讨论了模拟博弈论中的 **Donor's Game**，其中 Agents 根据之前的获胜策略进行策略升级，并为每一代进行递归运行。
   - *他们提议利用 DSPy 作为处理这些策略更新的潜在有效工具。*
- **共享文化进化实现的代码**：共享了 Donor's Game 模拟的代码，链接指向一个 [GitHub repository](https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120)，该仓库实现了一篇名为 *Cultural Evolution of Cooperation among LLM Agents* 论文中的方法论。
   - 该论文研究了 **LLM Agents** 社会是否可以通过进化方法发展出合作行为。



**提到的链接**：<a href="https://github.com/CakeCrusher/cultural_evolution/blob/main/donors_game/game/orchestrator.py#L120">cultural_evolution/donors_game/game/orchestrator.py at main · CakeCrusher/cultural_evolution</a>：实现了论文 *Cultural Evolution of Cooperation among LLM Agents* 中概述的方法论。该论文探讨了大语言模型 (LLM) Agents 社会是否可以发展出合作...

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1324818750834475078)** (2 条消息): 

> `Tinygrad Windows Support, Pull Requests for Windows Bugs` 


- **tinygrad 对 Windows 支持的态度**：一名成员询问 **tinygrad** 是否会接受专门修复 Windows 平台 Bug 的 Pull Requests。
   - 这突显了支持 Windows 的持续挑战，尽管它不是 **tinygrad** 开发的主要重点。
- **接受针对 Windows 的简单修复**：另一名成员推测，如果修复方案**简单且稳定**，此类 Pull Requests 可能会被接受。
   - 这表明社区愿意努力提高兼容性，尽管标准比较谨慎。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1324475911436767394)** (2 条消息): 

> `Shapetracker Documentation, Matrix Memory Layout, Memory Index Calculation, Stride in Matrix Access` 


- **对库文档的赞赏**：一名成员对库文档的质量表示赞赏，希望更多库能提供类似的资源。
   - *These blogs are very nice* —— 强调了对改进各类库文档的渴望。
- **寻求 Shapetracker 数学原理的解答**：一名成员询问了 `Shapetracker` 对象中计算索引背后的数学原理，并引用了一篇特定的 [tinygrad-notes 博客](https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html)。
   - 他们阐明了在访问以线性数组存储的矩阵时 Stride 的概念，并请求获取相关资源以更好地理解底层原理。



**提及的链接**：<a href="https://mesozoic-egg.github.io/tinygrad-notes/20241217_st.html">Shapetracker</a>：tinygrad 教程

  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1324626602868478015)** (4 条消息): 

> `Weights and Biases vs MLflow, Recording Experimentation Results, State of Classical Machine Learning, 1-bit Large Language Models` 


- **Weights and Biases 是云端的，MLflow 是自托管的**：Weights and Biases (WandB) 作为托管服务运行，这使得它对某些用户来说不太合适，而 **MLflow** 提供了自托管选项。
   - 这两种工具在管理机器学习实验方面都非常有效。
- **存储在数据库中的实验结果**：在最坏的情况下，实验结果可以记录到 **Postgres** 或 **Clickhouse** 数据库中。
   - 这为跟踪实验数据提供了一种备选方案。
- **关于经典机器学习现状的讨论**：一名成员对经典机器学习是否正被 LLM 应用所掩盖表示担忧。
   - 在这种不断演变的格局中，关于**推荐系统**、**时间序列**和**聚类**问题的未来出现了一些疑问。
- **BitNet：开创性的 1-bit LLMs**：最近的发展（如 **BitNet**）引入了 **1-bit Large Language Models** (LLMs)，它们在性能指标上能有效匹配全精度模型，同时具有极高的成本效益。
   - 这些模型使用三元权重，并为针对低比特表示量身定制的硬件优化创造了新机会。



**提及的链接**：<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit Large Language Models (LLMs) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1324789662065365135)** (4 条消息): 

> `AI Reader 工具，内容中的 Embedding 权重，按主题索引` 


- **面向学生的 AI Reader 工具**：一位用户正在开发一种低成本笔记本电脑，配备 “AI Reader” 工具，每当打开 PDF 时都会启动一个聊天 GUI 进行辅导，旨在帮助非洲的学生。
   - 他们正在探索使用 Nomic embed 来处理内容 Embedding 并提供模拟考试反馈，需要确保在用户查询期间实现有效的本地 Embedding。
- **按相关性对内容权威性进行排名**：一位成员建议需要一个能够随时间演变的教育内容权威性排名系统，特别是在计算机科学等快速变化的领域。
   - 他们强调了在无需每次重新索引的情况下动态维护此排名的挑战。
- **在学习材料中优先考虑学生成绩单**：讨论继续探讨了学生成绩单应在内容排名中占据更高权重，以反映个人学业表现。
   - 这提出了在教育工具中采用更个性化索引方法的想法。
- **按主题索引以获得更广泛的上下文**：一位用户建议考虑基于“主题”或“话题”而非仅仅基于“书籍”的索引方法，从而允许整合补充文章和笔记。
   - 这种方法可以为选择相关资源和有效填补信息空白提供更强的控制力。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1324539058566729768)** (2 条消息): 

> `AI 与 Blender 3D 建模，动物与 EEG，动物的语言/动作映射` 


- **寻求 AI 与 Blender 协作**：一位成员询问是否有任何团队正在研究专门用于**注释**的 **AI 与 Blender 3D 建模**，以支持 Blender 社区。
   - 该请求突出了对通过 AI 增强 **Blender 能力**的协作项目的兴趣。
- **探索动物语言映射**：另一位成员对研究 **动物与 EEG** 结合 AI 进行**语言/动作映射**及动物理解的社区表现出浓厚兴趣。
   - 这表明人们对神经科学与 AI 在**动物行为研究**交叉领域的兴趣日益浓厚。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 条消息): 

yoavhacohen: https://x.com/yoavhacohen/status/1875148348489113891
  

---


---


---


---


---


{% else %}


> 完整的频道逐条分析已针对电子邮件进行了截断。
> 
> 如果您想查看完整的分析，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}