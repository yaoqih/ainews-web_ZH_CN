---
companies:
- openai
- anthropic
- alibaba
- microsoft
- cohere
- langchain
- weights-biases
- deepseek
- rakuten
- rbc
- amd
- johns-hopkins
date: '2025-01-10T03:35:37.109683Z'
description: '以下是该文本的中文翻译：


  **rStar-Math** 通过使用 **7B 参数的大语言模型 (LLM)**、**蒙特卡洛树搜索 (MCTS)** 以及 **过程奖励模型 (Process
  Reward Model)**，在数学推理方面达到了 **90.0% 的准确率**，超越了 **OpenAI 的 o1-preview**。**阿里巴巴** 推出
  **通义千问 (Qwen Chat)**，搭载 **Qwen2.5-Plus** 和 **Qwen2.5-Coder-32B-Instruct** 模型，增强了视觉语言和推理能力。**微软**
  发布 **Phi-4**，该模型采用了 **40% 的合成数据** 进行训练，并改进了预训练过程。**Cohere** 推出 **North**，这是一个集成了
  **大语言模型 (LLM)**、**检索增强生成 (RAG)** 和自动化的安全 AI 工作区，专为私有部署设计。**LangChain** 展示了一个具备多步工作流和开源数据集的企业研究智能体。**Transformers.js**
  发布了用于 JavaScript 文本嵌入和图像分割的演示。研究亮点包括：用于增强思维链推理的 **Meta Meta-CoT**、具备递归自我改进能力的 **DeepSeek
  V3**，以及协作式 AI 开发平台。行业合作方面，包括 **乐天 (Rakuten)** 与 **LangChain** 的合作、**North** 为 **加拿大皇家银行
  (RBC)** 的 90,000 名员工提供支持，以及 **Agent Laboratory** 与 **AMD** 和 **约翰霍普金斯大学** 的合作。技术讨论强调了
  **CUDA** 和 **Triton** 对 AI 效率的重要性，以及 **吴恩达 (Andrew Ng)** 提出的不断演进的 AI 辅助编程技术栈。'
id: 1925a87f-9ac9-4fa2-b017-b90e7b4a88a7
models:
- rstar-math
- o1-preview
- qwen2.5-plus
- qwen2.5-coder-32b-instruct
- phi-4
- claude-3.5-sonnet
original_slug: ainews-not-much-happened-today-4520
people:
- reach_vb
- rasbt
- akshaykagrawal
- arankomatsuzaki
- teortaxestex
- aidangomez
- andrewyng
title: 今天没发生什么。
topics:
- math
- process-reward-model
- mcts
- vision
- reasoning
- synthetic-data
- pretraining
- rag
- automation
- private-deployment
- multi-step-workflow
- open-source-dataset
- text-embeddings
- image-segmentation
- chain-of-thought
- multimodal-reasoning
- finetuning
- recursive-self-improvement
- collaborative-platforms
- ai-development
- partnerships
- cuda
- triton
- ai-efficiency
- ai-assisted-coding
---

<!-- buttondown-editor-mode: plaintext -->**更多的 PRMs 就是你所需要的一切？**

> 2025年1月8日至1月9日的 AI 新闻。我们为您检查了 7 个 subreddits、[433 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 32 个 Discord 服务器（219 个频道，2928 条消息）。预计节省阅读时间（以 200wpm 计算）：**312 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

祝贺 [Anthropic 的所有七位亿万富翁联合创始人](https://x.com/andrewcurran_/status/1876705929296581078?s=46)。

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

**AI 模型与基准测试**

- **rStar-Math 在数学推理方面超越 OpenAI 的 o1**：[@reach_vb](https://twitter.com/reach_vb/status/1877206745652592763) 详细介绍了 **rStar-Math** 如何利用 **MCTS** 和 **Process Reward Model**，在 MATH 基准测试中通过 **7B LLM** 实现了 **90.0% 的准确率**，表现优于 **o1-preview** **+4.5%**。
- **Qwen Chat 在 Open WebUI 上线**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1877426465349972113) 宣布发布 **Qwen Chat**，包含 **Qwen2.5-Plus** 和 **Qwen2.5-Coder-32B-Instruct** 等模型，增强了 **vision-language** 和 **reasoning** 能力。
- **微软 Phi-4 模型发布**：[@rasbt](https://twitter.com/rasbt/status/1877380409387802946) 分享了关于 **Phi-4** 的见解，强调其在 **40% 合成数据**上进行训练，并通过增加训练 epoch 提升了 **pretraining** 的性能影响。

**AI 工具与平台**

- **面向企业的 North AI Workspace**：[@cohere](https://twitter.com/cohere/status/1877335657908949189) 推出了 **North**，这是一个集成了 **LLMs**、**RAG** 和 **automation** 的安全 AI 工作空间，针对 **private deployments** 进行了优化，旨在提升员工生产力。
- **LangChain 的公司研究 Agent**：[@LangChainAI](https://twitter.com/LangChainAI/status/1877400985691439150) 展示了一个**公司研究 Agent**，它遵循包括 **Research**、**Extraction** 和 **Reflection** 阶段在内的**多步工作流**，并提供了一个用于评估的**开源数据集**。
- **Transformers.js 演示发布**：[@tom_doerr](https://twitter.com/tom_doerr/status/1877343672280207668) 分享了一系列 **Transformers.js** 的**演示**，涵盖了在 **JavaScript environments** 中执行 **text embeddings** 和 **image segmentation** 等任务。

**AI 研究与研究报告**

- **Gradient Dissent 播客剧集**：[@weights_biases](https://twitter.com/weights_biases/status/1877109160652976493) 邀请了 **@akshaykagrawal**，在最新一期的 **Gradient Dissent** 中讨论了用于 **AI development** 的**协作平台**。
- **LLM 中的 Meta Chain-of-Thought**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1877191479216558597) 介绍了 **Meta Meta-CoT**，这是 **Chain-of-Thought** 的一种扩展，通过对底层推理过程建模，增强了 **multimodal reasoning** 能力。
- **DeepSeek V3 与 LLM 的自我改进**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1877403009795182674) 讨论了 **DeepSeek** 使用 **domain-specific data** 进行 **finetuning** 和**递归自我改进**的方法，强调了 **MCTS** 在生成高质量训练数据中的作用。

**AI 行业合作伙伴关系**

- **乐天 (Rakuten) 与 LangChain 合作**：[@LangChainAI](https://twitter.com/LangChainAI/status/1877415045455372778) 宣布与 **Rakuten** 合作，认可其为少数几家通过 **Generative AI** 交付**真实价值**的公司之一。
- **North 与 RBC 的合作伙伴关系**：[@aidangomez](https://twitter.com/aidangomez/status/1877320622021222513) 透露了与 **@RBC** 的合作，旨在为**金融服务**优化 **North**，并支持 **90,000 名员工**采用最新的 **AI technologies**。
- **Agent Laboratory 与 AMD 及约翰霍普金斯大学的合作**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1877378568277340306) 强调了 **Agent Laboratory** 如何使研究人员能够使用 **LLM agents** 完成**整个研究过程**，促进了**开源**和可定制的解决方案。

**技术讨论与开发**

- **CUDA 和 Triton 助力 AI 效率**：[@hkproj](https://twitter.com/hkproj/status/1877323712703193565) 强调了学习 **CUDA** 和 **Triton** 对于在 **AI development** 中获得显著 **financial gains** 的重要性，正如链接视频中展示的那样。
- **AI 辅助编程最佳实践**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1877405010893619238) 分享了他不断演进的 **software stack**，利用 **OpenAI’s o1**、**Anthropic’s Claude 3.5 Sonnet** 等 **AI tools** 以及各种 **deployment platforms** 来提升 **prototyping efficiency**。
- **AI 模型中的动态少样本提示 (Dynamic Few-Shot Prompting)**：[@hwchase17](https://twitter.com/hwchase17/status/1877407417656553831) 讨论了在 **Realm-X** 中实现 **dynamic few-shot prompting**，通过根据用户查询选择最相关的示例，将性能从 **~40% 显著提升至 ~80%**。

**迷因与幽默**

- **AI Agents 与工作生活平衡**：[@bindureddy](https://twitter.com/bindureddy/status/1877189717080551775) 幽默地列举了 **AI agents** 的特征，调侃了它们目前的局限性，同时预测了其快速的进步。
- **AI 取代工作**：[@mickeyxfriedman](https://twitter.com/mickeyxfriedman/status/1877137887952597251) 开玩笑说 **AI 正在消除各种独特的职位角色**，突显了 AI 颠覆性影响中幽默的一面。
- **个人 AI 体验**：[@karpathy](https://twitter.com/karpathy/status/1877102757464719652) 分享了他被 **AI** 增强的日常生活，以幽默的方式反映了 AI 工具与日常生活的融合。

**AI 社区与活动**

- **斯坦福 NLP 研讨会**：[@stanfordnlp](https://twitter.com/stanfordnlp/status/1877121694193836305) 宣布了 **@taoyds** 关于 **Vision-Language Models** 的演讲，邀请非校内人士注册参加研讨会。
- **面向 AI 工程师的 GitHub Expo**：[@swyx](https://twitter.com/swyx/status/1877434750891073715) 推广了 **@aiDotEngineer Expo**，目标受众是招聘 **AI engineers** 的人群，并鼓励通过专用空间参与。
- **AI Studio 加入 Google DeepMind**：[@osanseviero](https://twitter.com/osanseviero/status/1877452798683430988) 庆祝了 **AI Studio**、**Gemma** 和 **Gemini API** 与 **Google DeepMind** 的合并，期待在 **open models** 和 **accessible research** 方面取得加速进展。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Groq 对模型的处理：见解与对比**

- **[这就是我在 Groq 上使用模型的体验](https://i.redd.it/7tqzm8bsiube1.png)** ([Score: 1096, Comments: 64](https://reddit.com/r/LocalLLaMA/comments/1hwwvuz/this_sums_my_experience_with_models_on_groq/))：该帖子幽默地批评了 **Groq** 在 **Llama3.3 70b** 和 **Qwen2.5 72b** 模型上的表现，将其比作一个算术飞快但极不准确的角色。该迷因暗示虽然 Groq 的处理速度可能很快，但可能缺乏精度，正如通过一个错误的乘法结果的喜剧性交流所描绘的那样。
  - **Groq 的性能与用例**：**Groq** 因过度量化模型以适应如 **230 MB** 这样的小 **VRAM** 尺寸而受到批评，这可能导致精度下降。用户建议 Groq 更适合处理简单的任务（如清理转录文本），而不是复杂的推理任务。
  - **对比评估**：**Cerebras** 评估了包括 Groq 在内的各供应商的 **Llama 3.1 8B 和 70B** 模型，发现尽管有幽默的批评，Groq 的表现与其他供应商相当。该评估可以在 [Cerebras 的博客](https://cerebras.ai/blog/llama3.1-model-quality-evaluation-cerebras-groq-together-and-fireworks)上找到。
  - **模型替代方案与疑问**：一些用户质疑选择 Groq 的决定，建议使用 **Qwen2.5 72b** 等替代方案以获得更好的结果。也有人怀疑该帖子可能由 **Cerebras** 或 **Nvidia** 等竞争对手赞助。


**主题 2. Phi-4 性能：基准测试 vs 现实任务**

- **[Phi 4 仅有 14B，但在多项任务中优于 Llama 3.1 70B。](https://i.redd.it/uwfo8ig8jwbe1.png)** ([分数: 251, 评论: 63](https://reddit.com/r/LocalLLaMA/comments/1hx5i8u/phi_4_is_just_14b_but_better_than_llama_31_70b/)): 根据一份分析 AI 模型激活参数与 MMLU 综合性能得分的散点图，14B 参数模型 **Phi-4** 在特定任务中表现出优于 **Llama 3.1 70B** 的性能。该图表强调了 **Phi-4** 的高效率和有效性，将其定位为“小而强大”的模型，超越了如 **Llama-3.3-70B** 和 **Qwen2.5-72B** 等更大型的模型。
  - **Phi-4 的 Benchmark 重点**: 业界对 **Phi-4** 在真实世界任务中的表现存在质疑，有人声称它在 Benchmark 中表现出色是因为针对 Benchmark 数据进行了大量训练，而非实际任务。**SnooPaintings8639** 指出，虽然 **Phi-4** 在 Benchmark 上得分很高，但在实际用例和封闭测试中表现挣扎，暗示存在过拟合（overfitting）的担忧。
  - **模型对比**: **Phi-4** 并非被普遍认为优于 **Llama 3.1 70B** 或 **Qwen 2.5 35B** 等更大型模型。**siegevjorn** 和 **silenceimpaired** 对其优越性表示怀疑，**Vishnu_One** 则确认它并未超越 **Qwen 2.5**。
  - **训练与数据策略**: 正如 **rabbotz** 所强调的，**Phi-4** 的训练策略侧重于利用合成数据（synthetic data）进行复杂问题的推理。**x0wl** 提到，该模型在训练中被刻意避开了事实性问题，导致其在通用知识方面表现不佳，但在数学 Benchmark 中表现优异。


- **Phi-4 Llamafied + 4 个 Bug 修复 + GGUF，动态 4-bit 量化** ([分数: 202, 评论: 64](https://reddit.com/r/LocalLLaMA/comments/1hwzmqc/phi4_llamafied_4_bug_fixes_ggufs_dynamic_4bit/)): **Phi-4 模型** 已更新，包含 **4 个 Bug 修复**，改进了 Tokenizer 和聊天模板（chat template）的处理，从而增强了推理和微调性能。该模型现已 **Llamafied**（Llama 化），以兼容各种框架，使用 [Unsloth](https://github.com/unslothai/unsloth) 可实现 **2 倍微调速度提升、70% VRAM 占用减少** 以及 **9 倍上下文长度扩展**。[HuggingFace](https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa) 上的新上传内容包括 **GGUF、4-bit 和 16-bit 版本**，以及通过选择性保留 16-bit 层来提高准确性的 **动态 4-bit 量化（Dynamic 4-bit quants）**。
  - **Bug 修复与改进**: **Phi-4 模型** 获得了重大 Bug 修复，特别是在 Tokenizer 方面，提升了性能。修复细节见 [博客文章](https://unsloth.ai/blog/phi4)，这些修复增强了模型的准确性，例如在使用更新后的 GGUF 文件时，Python 测试通过率提升了 **20%**。
  - **动态 4-bit 量化与兼容性**: **动态 4-bit 量化** 主要用于推理或微调，而非为了兼容 **llama.cpp** 等框架。如 [这篇博客文章](https://unsloth.ai/blog/dynamic-4bit) 所述，与 **BitsandBytes 4-bit** 相比，这些量化版本提供了更高的准确性。
  - **用户反馈与性能**: 用户报告称 **Phi-4 模型** 的性能和准确性有所提高，超出了预期以及 **Phi-3** 等先前版本。据指出，由于聊天模板的修复，该更新显著提升了在渗透测试（Pentesting）多选题等测试中的表现。


**主题 3. NVIDIA Project DIGITS 显存带宽推测**

- **为什么我认为 NVIDIA Project DIGITS 将拥有 273 GB/s 的内存带宽** ([Score: 372, Comments: 130](https://reddit.com/r/LocalLLaMA/comments/1hwthrq/why_i_think_that_nvidia_project_digits_will_have/))：作者根据 NVIDIA CES 演讲图像中内存芯片尺寸的测量结果，估计 **NVIDIA Project DIGITS** 将拥有 **273 GB/s** 的内存带宽。他们使用 GIMP 修正了图像透视，并将内存芯片的长宽比与 **Micron 128Gb LPDDR5X 芯片**进行了对比，得出 **315-ball x32 总线封装**是最接近的匹配项。演讲中未提及内存带宽，这表明其带宽可能并非特别高。
  - 讨论中充满了对 **NVIDIA Project DIGITS** 估计的 **273 GB/s 内存带宽**的怀疑，用户将其与拥有 **546GB/s** 带宽的 **Apple M4 Max** 等硬件进行对比，并质疑为什么 NVIDIA 在演讲中没有提到带宽，暗示其带宽并不出众。用户还将其与 **AMD 的 Strix Halo** 进行对比，并指出 **Xeon** 或 **Epyc** 系统可能以更低的价格提供相似或更好的性能。
  - 评论者争论了 **DIGITS** 与 **Ryzen AI Max+ PRO 395** 的实用性，指出 **Ryzen 395** 在通用用途上可能更便宜且更全面，而 **DIGITS** 则提供 **CUDA** 和潜在的集群优势。两台机器都配备了 **128GB 内存**，但人们对 **DIGITS** 的速度以及与其他系统相比的价值表示担忧。
  - 考虑到 **Micron** 与 **NVIDIA** 过去的业务关系以及可能使用的 **Micron LPDDR5X 内存**，人们对 **Micron** 参与 **DIGITS** 项目进行了推测。一些用户提到 **Micron 的双芯片封装 (dual die packaging)** 是一种节省成本的措施，而另一些人则指出 **DIGITS** 可以被视为具有 **CUDA** 能力的、价格过高的 **AMD Strix Halo** 版本。


**主题 4. TransPixar：保持透明度的生成模型**

- **[TransPixar：一种保持透明度的新型生成模型，] (https://v.redd.it/8fhb41uq1xbe1)** ([Score: 417, Comments: 40](https://reddit.com/r/LocalLLaMA/comments/1hx7421/transpixar_a_new_generative_model_that_preserves/))：新型生成模型 **TransPixar** 已发布，因其在生成资产中保持**透明度**的能力而受到关注。这一特性在创建游戏资产方面具有潜力，标志着用于游戏开发的生成模型取得了进展。
  - **TransPixar** 因其在生成游戏资产方面的实用性而受到赞誉，并提供了其 **GitHub**、**Arxiv** 以及 **Hugging Face demo** 和模型的链接：[GitHub](https://github.com/wileewang/TransPixar), [Arxiv](https://arxiv.org/abs/2501.03006), [Demo](https://huggingface.co/spaces/wileewang/TransPixar), [Model](https://huggingface.co/wileewang/TransPixar)。
  - 有人担心使用来自大型动画工作室的**注册商标名称**，这可能会导致法律问题。
  - 该模型处理 **RGBA 输出**的能力被强调为一项重大的技术进步，因为大多数 AI 模型通常只产生 **RGB 输出**，这使得实现透明度成为一项复杂的功能。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Salesforce 的 AI 战略：到 2025 年停止招聘软件工程师**

- **[由于 AI 的影响，Salesforce 在 2025 年将不再招聘软件工程师](https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/)** ([Score: 729, Comments: 116](https://reddit.com/r/OpenAI/comments/1hwxh09/salesforce_will_hire_no_more_software_engineers/)): 由于 **AI** 的进步，**Salesforce** 计划在 **2025** 年停止招聘软件工程师。
  - 许多用户认为 **Salesforce 的 AI 公告**主要是一种营销策略，而非真正取代工程师的真实战略。**Indicava** 和 **bmson** 表示怀疑，引用了过去关于 AI 在 Salesforce 决策中作用的营销说法，而 **Frugal_Ferengi** 则认为 AI 目前还无法有效取代人类工程师。
  - 尽管发布了公告，**Salesforce** 仍在继续招聘工程师，尤其是在**印度**，这与停止招聘的说法相矛盾。**WonderingStarDusts** 和 **WH7EVR** 提供了持续招聘的证据，暗示该声明可能并未反映公司的实际招聘做法。
  - 讨论了 AI 对软件工程岗位的影响，**This_Organization382** 和 **wtf_is_a_monad** 对 AI 目前完全取代工程师的能力表示怀疑。他们强调像 **ChatGPT** 这样的 AI 模型在处理复杂任务时仍然很吃力，限制招聘的决定可能是一个缺乏实质数据支持的过早举动。


**主题 2. ChatGPT 失控：识别 Anthropic 式的错误**

- **[ChatGPT 失控了](https://v.redd.it/jx1hota7vtbe1)** ([Score: 408, Comments: 38](https://reddit.com/r/OpenAI/comments/1hwwdfd/chatgpt_loses_it/)): 标题为 **"ChatGPT loses it"** 的帖子缺乏详细正文，并包含一段无法分析的视频。文中未提供进一步的技术细节或讨论点。
  - 关于手机内存存满时质量是否会发生变化引发了幽默的讨论，**Caneofpain** 指出质量变化在技术上是真实的，但微小到无法测量。**Trollsmurf** 补充说，内存类型可能会以不同方式影响质量，由于电子状态的变化，添加数据可能会使设备变轻。
  - **Wirtschaftsprufer** 分享了一个涉及 **ChatGPT** 回复的喜剧轶事，展示了 AI 在回忆事件时出人意料且幽默的行为。
  - **Ithkuil** 评论了这种幽默的持久性，思考到 2025 年人们的看法会如何变化，**Drtoucan** 设置了一个提醒，以便在一年后重新审视这个话题。


**主题 3. 阴谋论：OpenAI 抹除前员工数据**

- **[X 用户 Mario Nawfal 发布的一条热门帖子声称 OpenAI 已从 ChatGPT 中删除了其前员工 Suchir Balaji 的所有痕迹。The Crypto Times 对该用户的说法进行了事实核查，发现属实。](https://www.cryptotimes.io/2025/01/09/has-openai-removed-traces-of-whistleblower-balaji-from-chatgpt/)** ([Score: 107, Comments: 67](https://reddit.com/r/OpenAI/comments/1hxbsq6/a_viral_post_by_x_user_mario_nawfal_had_claimed/)): 据 **X 用户 Mario Nawfal** 的热门帖子称，**OpenAI** 据称从 **ChatGPT** 中删除了前员工 **Suchir Balaji** 的所有痕迹。**The Crypto Times** 核实了这些说法并确认了其准确性。
  - 几位评论者质疑这些病毒式传播说法的可靠性，**Mrkvitko** 等用户指出标题具有误导性，强调 **Suchir Balaji** 的信息可能从未出现在训练数据中，而不是被删除了。**Tall-Log-1955** 和 **traumfisch** 批评了阴谋论视角以及 **The Crypto Times** 等来源的可信度。
  - 围绕 **Suchir Balaji** 在 **OpenAI** 角色的讨论突出了他的重大贡献，并引用了 **John Schulman** 对 Balaji 重要工作的认可。然而，关于他的吹哨人身份存在争议，**NotFromMilkyWay** 指出他违反了 NDA 协议以及随之而来的法律和个人后果。
  - 对话涉及了 **ChatGPT** 数据处理的技术层面，**traumfisch** 和 **SkaldCrypto** 讨论了网页搜索功能是否会让 **ChatGPT** 因为 Balaji 的媒体曝光度而识别出他，并将其与典型的训练数据限制进行了对比。


---

# AI Discord 摘要

> 由 o1-2024-12-17 生成的摘要之摘要的摘要

**主题 1. 模型对决与惊喜** 
 
- [**Phi-4 超越 Microsoft 官方版本**](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large)：Unsloth 的 Phi-4 性能飙升，超过了官方 Microsoft 版本。在一则活跃的 [推文](https://x.com/UnslothAI/status/1877136074042126338) 中，他们提到：*“我们发现并修复了 Phi-4 中的 4 个 bug，并将该模型 Llamafied 了”*。其 4-bit 和 16-bit 版本的发布立即在社区中引发了热潮。  
- [**rStar-Math 带来的惊人提升**](https://x._akhaliq/status/1877206745652592763)：Microsoft 的这项技术将 Qwen2.5-Math-7B 在 MATH benchmark 上的表现从 58.8% 提升至 90.0%，而 Phi3-mini 则从 41.4% 跃升至 86.4%。它们现在能解决约 53.3% 的美国数学奥林匹克（USA Math Olympiad）题目，引发了关于小型 LLM 巨大飞跃的讨论。  
- [**Qwen Chat 开启新大门**](https://chat.qwenlm.ai)：这个全新的 Web UI 统一了 Qwen 系列模型，支持直接上传文档和侧边栏对比。未来的扩展将包括 *voice*（语音）、*web search*（网页搜索）等，预示着一个用户友好的 AI 前沿阵地。  

**主题 2. 编程工具与 HPC 升级**  

- [**ComfyUI 集成 OpenPose**](https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2151)：用户通过参考 [workflow 指南](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use) 使用控制节点，克服了 Pony 模型的使用摩擦。一些人曾转向 Forge UI，但在新的节点集成方案出现后又重新回归。  
- [**AMD vs Nvidia GPU 宿命之战**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux)：社区成员对比了在 Windows 上使用 ZLUDA、ROCm 或原生 GPU 驱动的性能。每种方法都有独特的增益，官方 wiki 指南澄清了安装步骤。  
- [**自托管 Codeium 走向主流**](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides)：企业团队通过 [GitHub Issue #115](https://github.com/Exafunction/codeium.el/issues/115) 发现了本地部署版本，推动了高级配置的普及。同时，开发者称赞 *Cascade* 具有极低的编码开销和极速的端到端建站能力。  

**主题 3. 前沿 Prompting 与解码技术**  

- **Speculative Decoding 成为焦点**：有人称其为 *“语言模型的 DLSS”*，声称它能大幅降低训练和推理中的 GPU 占用。爱好者们非常推崇这一理念，认为这是在节省计算时间的同时优化输出的有效途径。  
- [**Function Calling 模型引发好奇**](https://x.com/altryne/status/1877220144725758414)：用户正在寻求开源 Function Calling 的 benchmark，重点关注训练后的准确性微调。结构化的 Prompt 和强大的测试集被认为是实现可靠调用的“秘方”。  
- **Meta-Prompting 与 System Message 调整**：创作者们发布了多层指令，通过重写 System Directives 来塑造模型响应。一些人坚持认为 *“真正的魔力在于从一开始就准确指定你想要的输出”*，强调精确的目标设定优于盲目猜测。  

**主题 4. HPC 与 GPU 启示**  

- **MI210 Occupancy 令 HPC 圈困惑**：开发者在基于 CDNA 架构的 GPU 上发现了令人费解的限制：每个计算单元仅 *2.5 个 block*，或者在使用 `__syncthreads()` 时仅为 *2* 个。他们将这些奇怪的 Occupancy 限制归因于 AMD 硬件设计深层的特性。  
- [**NVIDIA 推出 3000 美元的家用超级计算机**](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw)：爱好者们为个人 AI 实验室获得 HPC 级别的算力而欢呼，这突破了标准工作站的限制。早期采用者已经窥见了在不耗尽财力的前提下，在家进行真正 AI 实验的可能性。  
- [**ARC Prize 转型为非营利组织**](https://arcprize.org/blog/arc-prize-2025)：在 *Greg Kamradt* 的带领下，组织者转向以结构化资金引导 2025 年的 AGI 研究。他们基于 2024 年 ARC Prize 的洞察，承诺将推出更广泛的开源 AI 计划。  

**主题 5. 大型黑客松与企业动态**  

- [**AI Agent 黑客松吸引开发者**](https://studio.ottomator.ai/hackathon/register)：OpenRouter 以 10 美元的 API 额度和总计 6,000 美元的奖金池吸引参与者，n8n 提供了现金奖励。Live Agent Studio 环节于 1 月 8 日至 22 日运行，获胜者将于 2 月 1 日揭晓。  
- **Salesforce 冻结 2025 年招聘**：Marc Benioff 承诺 *Agentforce* 将带来 *30% 的生产力提升*，并宣称 *“五年后我们会变得更强大”*。尽管招聘冻结，支持者仍注意到了 AI 与企业战略之间强大的协同作用。  
- **Anthropic 以 600 亿美元估值融资 20 亿美元**：投资者估算其年度经常性收入（ARR）为 8.75 亿美元，这引发了对 2025 年突破性进展的“美好祈祷”。AI 领域对这笔巨额资金表示欢迎，期待地平线上出现巨大的飞跃。  

---

# 第一部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 在 OpenPose Pony 上的进展**：讨论围绕在 ComfyUI 中将 **OpenPose** 控制与 Pony 模型集成展开，参考了 [Forge UI 中的节点集成技巧](https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2151)。
   - 一位用户在 ComfyUI 的功能上遇到了挑战，转而使用 Forge UI 以改进工作流，但其他人从 [ComfyUI 工作流资源](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use)中提出了解决方案。
- **断电导致 SD 生成中断**：人们开始担心在 **Stable Diffusion** 生成过程中如果发生断电，可能会对 **GPUs** 造成损害以及导致数据损坏。
   - 一位用户确认 GPU 通常是安全的，但突然的中断可能会导致操作系统级的文件错误或数据丢失，并敦促进行频繁备份。
- **保持 AI 工具同步**：维护最新的 **A1111** 和 **ComfyUI** 被证明具有挑战性，旧版本的 Python 会引发冲突。
   - 参与者指出，使用 [Python 3.10.11](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki) 可以解决大多数版本不匹配问题，确保在这些框架之间的一致使用。
- **AMD GPU 对决**：用户对比了 Windows 上支持 AMD GPU 的 **ZLUDA** 和 **ROCm**，指出两者各有千秋。
   - 他们引用了在 AMD 硬件上设置 stable-diffusion-webui 的 [官方指南](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux)，并再次确认了原生 Windows 替代方案的可行性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 Phi-4 超越微软**：**Unsloth's Phi-4** 模型在 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large) 上超越了微软官方版本，在修复关键 bug 后发布了 **GGUF**、**4-bit** 和 **16-bit** 版本。
   - *“我们在 **Phi-4** 中发现并修复了 **4 个 bug**，并将该模型 **Llamafied**。”* 这是 [Unsloth AI (@UnslothAI) 的推文](https://x.com/UnslothAI/status/1877136074042126338)中的官方说法，引起了社区的热烈讨论。
- **Qwen2.5-Math-7B Instruct 在表格处理上备受推崇**：**Qwen2.5-Math-7B-Instruct** 模型被建议用于高效的 Markdown 表格计算，一些用户以 **3e-5** 的学习率训练了一个 epoch。
   - 一位用户在了解到 `mistralai/Mathstral-7B-v0.1` 不是基础模型或 PEFT 模型后，将注意力转向了 **Qwen** 的替代方案，以获得更好的表格性能。
- **投机采样 (Speculative Decoding) 登场**：**Speculative decoding** 被强调为语言模型的“DLSS”，旨在减少 **training** 或 **inference** 期间的资源消耗。
   - 该建议受到了好评，一位成员认为这是在节省 GPU 时间的同时优化 **model output** 的新视角。
- **LoRA 合并取得进展**：社区成员讨论了将基于较小变体训练的 **LoRA** 适配器合并到较大的 **16-bit** 模型中，以保持性能保真度。
   - 他们强调了细节损失极小，并警告说在 **4-bit** 基础上进行合并可能会降低最终结果的质量。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **私有化部署的 Codeium 取得进展**：社区成员发现了用于企业级部署的私有化（Self-Hosted）版本 **Codeium**，并寻求获取该版本的详细信息，同时参考了 [Codeium 定价详情](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides)。他们还查阅了 [GitHub Issue #115](https://github.com/Exafunction/codeium.el/issues/115) 以获取提取 **API keys** 的技巧。
   - 讨论中涉及了部署是否简便，以及此举是否会增加大型团队的采用率。一些人指出，**Codeium** 对个人用户仍然免费，而企业用户则追求本地部署的灵活性。
- **Windsurf 的困扰**：用户遇到了持续的 **Windsurf** 崩溃、冻结以及随机出现的“窗口无响应”错误。一名 **Ubuntu 24.04** 用户报告运行成功，而另一名使用 **Arch with Hyprland** 的用户通过删除配置文件解决了 Token 提交问题。
   - 他们希望 [Windsurf Editor Changelogs](https://codeium.com/changelog) 中的未来修复能解决稳定性问题。尽管有人报告在某些系统上运行流畅，但闪退表现削弱了用户的信心。
- **Cascade 大获好评**：社区成员称赞 **Cascade** 具有可靠的工作流处理能力和极低的代码编写开销。一位用户声称，利用其功能仅需极少的工作量就构建了公司网站。
   - 其他人对 **Cascade** 面板自动打开感到沮丧，并寻求更好的切换开关。他们在 [Codeium Feedback](https://codeium.canny.io) 上敦促开发者进行修复，希望能尽快解决。
- **Flow Credit 计费乱象**：几位参与者抱怨 **flow credits** 计费混乱，并怀疑存在重复收费。一位用户提到在信用额度分配极少的情况下却被收取了巨额费用，感觉被技术支持忽视了。
   - 他们敦促其他人在 [Codeium Feedback](https://codeium.canny.io) 上记录类似的计费投诉。对于协作中维持 **prompt credits** 的担忧也浮出水面，引发了对更透明的使用情况追踪的呼吁。
- **Agent 愿景与更新阵痛**：一些人询问在 Windsurf 中使用 **agents** 的情况，但论坛缺乏关于官方集成的明确信息。这引发了对桥接其他平台功能的兴趣。
   - 最近的一次更新导致 **Cascade** 中出现偶发性的命令失败和令人费解的代码生成。报告的问题从性能缓慢到部分功能损坏不等，引发了对快速补丁的反复呼吁。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Composer 的困惑**：反复出现的投诉指出 **Cursor composer** 倾向于忽略 **.cursorrules**，这促使用户转向其他编程工具以寻求可靠的编辑。
   - [0.44.9 版本中持续到 0.44.10 的生成卡顿问题](https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4) 加剧了用户对 **composer** 稳定性的不满。
- **Claude 的古怪特性**：多条评论强调，如果通过刻意的 **prompts** 鼓励 **Claude** 分享内部推理过程，它的表现会非常好。
   - 然而，用户仍对其不稳定的输出质量感到恼火，这需要仔细监控，并掩盖了潜在的生产力提升。
- **Cursor Rules 的严谨性**：社区成员强调使用专门的 [.cursorrules 文件](https://dotcursorrules.com/) 来引导模型在每个项目中保持合规。
   - [Cursor Directory](https://cursor.directory) 被引用为针对流行框架和语言定制的规则集中心。
- **文档需求与开发者对话**：参与者抨击了 **Cursor** 文档的不足，称其在高级功能和运行时指标方面令人困惑。
   - 他们建议通过 [官方论坛](https://forum.cursor.com) 获得开发者的更快回复，但许多人希望能有更深入的文字资源。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **颜色编码 Prompting 变得简单**：爱好者建议在 Prompt 中指定**颜色名称**和**十六进制代码**，强调简洁的指令以提高清晰度。
   - 一名成员建议采用简短的“只是一个想法”的方法，旨在通过保持指令简洁来消除困惑。
- **带有前缀的公共 Repos**：一名成员透露了 **StackBlitz** 的一个[公共 Repos 功能](https://x.com/stackblitz/status/1843668731681267801)，允许用户通过在 GitHub URL 前添加 'http://bolt.new' 来打开。
   - 他们指出这种设置增加了可访问性，让用户能够快速从可访问的仓库中加载代码。
- **Subreddit AI 征集问答**：一篇推广帖子介绍了 [**SubReddit AI**](https://subredditai.com)，邀请大家就 Prompting 策略提问。
   - 社区成员讨论了短 Prompt 策略和代码片段的使用，以优化模型输出。
- **Bolt 性能崩溃与 PWA 摩擦**：用户报告了 **Bolt** 的性能故障，有人因重复的代码插入消耗了 100k tokens。
   - 其他人抱怨 PWA 设置错误，尽管有少数人成功启动了他们的 PWA 以证明其可行性。
- **Supabase 与 GitHub 回滚困惑**：参与者指出 **Supabase** 迁移无法随项目代码一起回滚的问题，存在不可逆更改的风险。
   - 他们建议频繁进行 fork，而一些人在设置过程中遇到了 **GitHub** 部署障碍，包括空仓库问题。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 与 DeepSeek 的碰撞**：用户对比了 **Claude** 和 **DeepSeek**，对 **DeepSeek** 的能力评价褒贬不一，且偶尔出现执行错误。
   - 一些人强调[使用 VPN](https://unify.ai/) 或仔细设置可能会减少停顿，但其他人对其可靠性仍持怀疑态度。
- **Aider 的配置困惑**：成员在 Aider 发送 'prompt' 列表而非 'messages' 时遇到了 `litellm` 的 **TypeError** 问题，这与[故障排除文档](https://aider.chat/docs/troubleshooting/edit-errors.html)中的指导相呼应。
   - 他们引用了 [CONTRIBUTING.md](https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md) 进行澄清，并讨论了通过 [PR #540](https://github.com/Aider-AI/aider/pull/540) 自动化 pull requests 的最佳实践。
- **关注 OpenAI 的 Tier 5 密钥**：一场关于 **OpenAI** 模型分级的对话展开，讨论了 200 美元的 O1 Pro 订阅以及 [Unify.ai](https://github.com/unifyai/unify) 等替代方案。
   - 参与者权衡了成本与灵活性，分享了为高级功能实现稳健覆盖的技巧。
- **Gemini 2.0 Flash 移动端测试**：有人在处理杂务时，在语音模式下测试了 **Gemini 2.0 Flash Experimental**，用于快速进行应用创意头脑风暴。
   - 他们注意到它缺乏用于结构化规范的 Markdown 输出，但随后它创建了一个简明摘要以简化开发步骤。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **DeepResearch 与 NotebookLM 的笨重忧郁**：社区成员注意到 **DeepResearch** 与 **NotebookLM** 之间没有直接联系，并引用了一个关于提高研究和内容效率的 [YouTube 视频](https://youtu.be/spj0n-bFKJo)。
   - 他们考虑了可能的变通方法，如基于扩展的上传，并强调 **NotebookLM** 仍然缺乏处理外部仓库的完全原生方法。
- **通过 NotebookLM Plus 获取引用摘要**：一位用户引导 **NotebookLM** 仅返回源材料中的直接引用，观察到在没有 Plus 版本改进的内存保留功能下，可靠性会有所波动。
   - 他们还指出在不同会话中复制命令流存在困难，建议使用 **NotebookLM Plus** 以获得更稳定的 Prompt 遵循能力。
- **从英文生成普通话播客**：一名成员询问如何在 **NotebookLM** 中从**英文**源材料生成**普通话**播客，但未发现具体的解决方案。
   - 社区提出了协作想法，承认需要更强大的多语言处理工具。
- **许可证哀歌与播客提示词**：许多人遇到了与工作区许可证和功能移除相关的 **NotebookLM** 使用问题，讨论了重新开始或创建新笔记本以从头开始的可能性。
   - 一些人尝试了 [Illuminate](https://illuminate.google.com/create) 等外部工具以获得播客输出中的多样化语音，而另一些人则寻求创意 Prompt 以从精选源材料生成音频。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen Chat 快速亮相**：全新的 [Qwen Chat](https://chat.qwenlm.ai) 为 Qwen 模型扩展了 Web UI，支持**模型对比**、文档上传和可视化界面。
   - [Qwen 的推文](https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113)暗示即将推出更多增强功能，激发了社区的热情。
- **Snapdragon X Elite 关注 OpenCL？**：一位用户询问了 Snapdragon X Elite 对 **OpenCL** 支持的可能性，并引用了 Llama.cpp 中优化计算开销的更新。
   - 爱好者预见，如果集成实现，**LLaMA** 模型在不同硬件上的性能将得到提升。
- **AMD RX 7900XT vs Nvidia：GPU 宿命之战**：社区成员将 **AMD RX 7900XT** 与 **Nvidia 4090**、**4080** 和 **3090** 进行了对比，重点关注显存带宽问题，并引用了 [Reddit 上的讨论](https://reddit.com/...)。
   - 他们得出结论，在为高负载 LLM 工作负载选择 GPU 之前，详细的基准测试是关键。
- **MacBook VRAM 调整以适配更大模型**：MacBook 用户尝试通过 **/etc/sysctl.conf** 设置 **iogpu.wired_limit_mb=54272**，为 4-bit 和 6-bit MLX 模型释放内存。
   - 他们报告称，一旦系统识别出增加的 VRAM 分配，速度会有显著提升。
- **DIGITS 延迟风波**：等待 **DIGITS** 的成员希望它能提供进入 **Nvidia** 生态系统的广泛入口，但对延迟表示不满。
   - 他们保持乐观，认为一旦可用，全 CUDA 加速可以简化大规模 LLM 实验。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **图表生成势头强劲**：一位用户发现 ChatGPT 能够根据代码请求生成 **GRAPH**（图表），展示了高级数据可视化的潜力。
   - 另一位用户惊叹 *yea unbelievable*，突显了社区对 GPT 扩展功能的兴趣。
- **Meta-Prompting 成为焦点**：参与者探索了 **Meta-Prompting** 这一高级技术，通过分层指令塑造 AI 输出。
   - 一位成员强调从一开始就明确**期望的输出**，称其为获得稳健响应的关键。
- **Hassabis 寻求新一轮融资**：社区对 **Hassabis** 及其即将到来的投资者轮次表现出热情，赞扬他在 AI 领域取得的丰硕成就。
   - 他们表达了良好的祝愿，强调了群体对成功融资的希望。
- **OpenAI 提示策略受到审视**：一位参与者批评了 **OpenAI** 的方法，认为重新设计系统消息可能会提高性能。
   - 他们还强调了贡献**缺乏财务收益**的问题，引发了关于此类协作公平性的讨论。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **rStar-Math 提升模型准确率**：微软的 [rStar-Math](https://x.com/_akhaliq/status/1877206745652592763) 将 **Qwen2.5-Math-7B** 的准确率从 **58.8%** 提升至 **90.0%**，将 **Phi3-mini-3.8B** 从 **41.4%** 提升至 **86.4%**，超越了以往在 MATH 任务上的尝试。
   - 它解决了约 **53.3%** 的美国数学奥林匹克竞赛题目，引发了关于**小型 LLM** 性能巨大飞跃的讨论。
- **Qwen Chat 助力多模型协同**：[Qwen Chat](https://chat.qwenlm.ai) 在单一 UI 中统一了 **Qwen2.5-Plus** 和 **Qwen2-VL-Max**，支持侧边对比和文档上传。
   - 未来的扩展暗示将增加**联网搜索、图像生成和语音**功能，标志着向用户友好型 AI 交互迈出更大步伐。
- **NuminaMath 的数据瑕疵引发关注**：**NuminaMath** 旨在提供一致的单框解决方案，但 **2.6%** 的条目没有结果，**7.7%** 的条目有多个结果，表明可能存在数据异常。
   - 贡献者质疑开源数据集的质量，强调了大规模数学语料库中潜在的陷阱。
- **MoEs 优于稠密模型**：在相同的参数使用情况下，**Mixture of Experts** 的表现历来优于**稠密模型**，这意味着更大的参数池能带来更好的峰值性能。
   - 讨论倾向于在高级任务中使用 MoEs，尽管**训练复杂性**被认为是一个主要挑战。
- **AI 成本讨论引起政策观察者的警觉**：一份声称开源 AI 需要 **$5M** 的估算引起了混乱，随后的 [推文](https://x.com/teortaxesTex/status/1877467302989295673/photo/1) 澄清了实际的总支出。
   - 成员警告说，公众可能会忽视 **capex、R&D** 和数据策展支出，从而导致对 AI 预算的错误结论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SmolLM Steps Up with 320GB Dataset**: **SmolLM Corpus** 的发布推迟到了“明天”，现在承诺提供 **320GB** 的可分片数据，而不是之前的 1TB 未压缩版本，以便于处理。
   - 一位用户称其“比之前的 1TB 未压缩版本更易用”，引发了早期采用者对**完整数据集**的期待。
- **SciAgents Sparks Scientific Synergy**: 社区成员赞扬了 **SciAgents** 的本体论方法（ontological approach），认为其揭示了研究中的跨学科联系，并引用了[这篇 arXiv 论文](https://arxiv.org/abs/2409.05556)。
   - 虽然它目前尚未达到 **GPT-4-level** 的突破水平，但用户看到了在多个科学领域进行更高层级学习编排（learning orchestration）的巨大潜力。
- **Grokking Gains Steam with Weight Decay**: 参与者强调 **grokking** 与 **Softmax Collapse** 相关，引用了 [Grokking at the Edge of Numerical Stability](https://arxiv.org/abs/2501.04697)，并指出高强度的 **0.1 weight decay** 通常能缓解过拟合。
   - 他们质疑 attention 对 **softmax** 的依赖，提出了 **sigmoid loss** 等替代方案，同时建议较低的 WD 可能有助于避免 LLM 优化中的低秩陷阱（low-rank pitfalls）。
- **Modal Makes GPU Training Accessible**: 几位用户称赞 **Modal** 允许通过云端 GPU 进行更大规模的模型训练，并提到每月慷慨的 **$30 免费额度** 是其一大亮点。
   - 一位用户称赞它在处理大型任务时比传统的预留实例“更具成本效益”，重点在于大规模支持 **researchers**。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Alpha Competition: Swift Softmax Showdown**: 一项新的 [alpha competition](https://link.to.alpha_competition) 邀请追求速度的开发者在暂存服务器上设计最快的 **softmax kernel**，报名现已开放。
   - 早期参赛者测试了性能提升，并对结果感到兴奋。
- **Nectar Social’s Sweet $10k Bounty**: 初创 AI 公司 **Nectar Social** 为在西雅图招聘 **LLM/AI Engineer** 和 **Sr/Staff Product Manager** 等职位提供高达 **$10,000** 的推荐费。
   - 他们由主要投资者资助，专注于**社交电商**（social commerce），鼓励感兴趣的人士联系。
- **ARC Prize’s Non-Profit Pivot**: [ARC Prize](https://arcprize.org/blog/arc-prize-2025) 正在转型为非营利基金会，以塑造围绕 AGI 的研究，由 **Greg Kamradt** 及其团队指导。
   - 他们强调了一个更结构化的框架，并借鉴了 ARC Prize 2024 的见解。
- **MicroDiT Meets MMDIT**: 研究人员完成了 **MicroDiT** 的复现，分享了[模型权重](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt)和用于本地测试的[推理脚本](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb)。
   - 目前，计划中的 **DCAE** autoencoder 和 **MMDIT** 升级有望提高 prompt 遵循能力，但尚待更强大的算力资源。
- **MI210 Occupancy: The Great ROCm Riddle**: 爱好者们研究了 **MI210** 上令人费解的 occupancy 数值，观察到每个 compute unit 有 2.5 个 block 以及其他意外数据。
   - 他们发现添加 **__syncthreads()** 会使最大值降至正好为 **2**，突显了基于 CDNA 的 GPU 的特性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DisTrO 的发布推动了协作**：新**开源的 DisTrO** 引起了多位用户的兴奋，他们渴望将其集成到自定义设置中。
   - 讨论围绕改进文档以及与高级优化器（optimizers）的潜在协同作用展开。
- **DeepSeek V3 引发输出质量辩论**：官方 **DeepSeek V3** 与第三方提供商之间的输出差异引发了关于缓存和模型问题的猜测。
   - 一些人怀疑重复的回答源于缓存奇点，而另一些人则认为是固有的模型微调（tuning）限制。
- **Hermes 模型引发审查讨论**：**Hermes** 模型因部分审查而受到批评，许多人发现必须使用系统提示词（system prompts）来绕过限制。
   - 关于是通过高级提示词工程（prompt engineering）还是更深层的训练变更来解锁真正无过滤模型的意见不一。
- **函数调用模型引发 Benchmark 好奇心**：成员们对比了**开源函数调用（function-calling）模型**，寻找 Benchmark 和提升函数调用准确性的策略。
   - 训练后改进和结构化提示词被认为是优化性能的主要手段。
- **Qwen 7B 以 AIME 级别的技能惊艳数学迷**：**Qwen 7B** 以 o1 级别的水平解决了 AIME 问题，[这条推文](https://x.com/altryne/status/1877220144725758414)强调了基于 MCTS 的反思方法。
   - 虽然许多人称赞该模型的计算技巧，但也有人质疑这些数学成就否能转化为更广泛的推理能力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Salesforce 令人惊讶的停招与高涨的雄心**：Marc Benioff 宣布 **Salesforce** 在 **2025** 年将不再招聘软件工程师，理由是 **Agentforce** 带来了 **30% 的提升**。
   - 他引用了[这篇文章](https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff)，并预测尽管处于招聘冻结期，“五年后我们将变得更强大”。
- **OpenAI 的大修影响了自定义指令**：10 月 19 日，**OpenAI** 对 ChatGPT 语音系统的更新在引入新功能的同时，似乎破坏了自定义指令（custom instructions）。
   - 一条[推文](https://x.com/topmass/status/1877444315871326422)强调了被中断的**语音改进**，以及在这些变更期间对稳定测试的迫切需求。
- **Anthropic 惊人的 20 亿美元估值飞跃**：消息人士确认 **Anthropic** 正在筹集 **20 亿美元**，估值飙升至 **600 亿美元**，助力其 2025 年的增长战略。
   - 一份[记录](https://x.com/andrewcurran_/status/1876705929296581078)显示其年度经常性收入（ARR）达到 **8.75 亿美元**，强调了“企业销售的显著扩张”。
- **Google 将 AI 团队整合至 DeepMind 旗下**：多个 **Google** AI 团队将与 **Google DeepMind** 合并，推动 **2025** 年新的开源模型计划和开发者工具。
   - 一篇[帖子](https://x.com/osanseviero/status/1877452798683430988)暗示了“未来激动人心的一年”，并预示了统一 AI 工作的可能内部变动。
- **Moondream 模型取得进展**：更新后的 **Moondream 2b** 视觉语言模型引发了关于脚本可用性和功能改进的讨论。
   - 一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/)提到了“资源共享”，并称赞了该模型的强劲表现。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Hackathon 热潮与 Live Agent Studio 对决**：OpenRouter 宣布举办 [AI Agent Hackathon](https://studio.ottomator.ai/hackathon/register)，提供 **$10** 的 API 额度和 **$6,000** 的奖金池，此外还为顶尖的 n8n Agent 设立了新的**现金奖励**。
   - **Live Agent Studio** 环节将于 1 月 8 日至 22 日举行，获胜者将于 2 月 1 日揭晓，社区投票从 1 月 26 日开始。
- **Gemini Flash 震撼登场**：一位用户分享了 **Gemini Flash 1.5** 的性能指标，在 **255.6 tps** 的速度下，以 **$0.000171** 的成本完成了 **63,364** 次请求和 **7,018** 次输出。
   - 爱好者们对其功能表示赞赏，尽管有人建议进行额外调整以获得更流畅的体验。
- **OpenRouter UI 遭遇延迟峰值**：成员们批评 **OpenRouter** 在聊天记录超过 **1k 行**时 UI 反应迟钝，导致滚动和输入变得繁琐。
   - 他们建议改进分页和活动过滤功能以保持运行速度。
- **O1 API 的奇特现象困扰开发者**：开发者注意到 **O1 API** 响应中出现了 **=====** 块，取代了反引号并引起了困惑。
   - 有人猜测这可能是为了节省 Token，但许多人认为这具有干扰性。
- **Hanami 受到简短关注**：一些人好奇是否有人在采用 **Hanami**，其中一位用户在测试过程中遇到了意外字符。
   - 随后讨论了其可靠性，尽管具体细节有限。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 推出 CSV 下载功能**：Perplexity 引入了从响应中**将表格下载为 CSV** 的选项，使数据提取变得轻而易举。
   - 开发者对这一功能表示欢迎，如[这张截图](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg)所示，称其为处理数据任务的关键便利功能。
- **Youzu.ai 室内设计灵感**：AI 驱动的 [Youzu.ai](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688) 帮助用户规划房间设计并识别本地购买选项，简化了购物流程。
   - 社区反馈赞扬了其用户友好的方式，称其为*繁重设计任务的颠覆者*。
- **Ecosia 寻求与 Perplexity 建立绿色合作伙伴关系**：来自 **Ecosia** 的一位产品经理联系了 Perplexity，寻求协作努力和**绿色搜索**协同效应。
   - 他们难以找到合适的联系人，因此请求社区进行引荐，希望能减少连接两个平台的阻力。
- **NVIDIA 的家用超级计算机引发讨论**：根据[这份公告](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw)， **NVIDIA** 发布了一款售价 $3000 的个人用超级计算机套装。
   - 爱好者们注意到了在家进行 AI 实验的潜力，赞扬了拥有超越典型工作站限制的 HPC 能力的可能性。
- **丰田的火箭传闻**：报告指出 **Toyota** 正在探索新的火箭领域，如[这篇文章](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q)所述。
   - 尽管丰田主要是一家汽车制造商，但其向航空航天领域的扩张引发了关于技术跨界的猜测。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 'North' 推动生产力提升**：Cohere 宣布开启 **North** 的早期访问（EAP），这是一个集成了 **LLMs**、**搜索**和 **Agent** 的一体化**安全 AI 工作空间**，旨在超越 **Microsoft Copilot** 和 **Google Vertex AI Agent Builder**，详见[其博客](https://cohere.com/blog/north-eap)。
   - 他们展示了日常任务中**无缝的用户体验**，社区强调了其推动运营效率的潜力，并引用了 [Cohere 的官方推文](https://x.com/cohere/status/1877335657908949189)。
- **Command R+ 助力大型生成式运行**：一位用户强调了 **Command R+** 在大型生成模型中的应用，并参考了[官方模型概览](https://docs.cohere.com/docs/models)以获取高级工作流和性能细节。
   - 社区兴趣点包括如何将 **Command R+** 融入日常任务的建议，再次确认了其作为强大模型使用的核心功能地位。
- **从 embed-v2 升级到 v3 引发关注**：一位用户寻求从 **embed-v2** 迁移到 **v3** 的指南，并对重新生成海量语料库表示担忧。
   - 他们注意到了 **embed-v2** 可能被弃用的前景，引发了关于增量升级策略和潜在陷阱的讨论。
- **滚动聊天方式突破 4k Token 限制**：用户对使用 **cmd-r+** 生成完整章节或进行推理时受到的 **4k token** 限制表示沮丧。
   - 社区提议采用**滚动聊天历史（rolling chat history）**来突破这些界限，指出这是一种实现更长输出的更平滑方法。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **悬赏助力 PR #8505**：社区为在 OS X 上使用 **MOCKGPU** AMD 重新测试 [PR #8505](https://github.com/tinygrad/tinygrad/pull/8505) 提供奖励，可通过 PayPal 或 USDC 在 **Tinygrad** 社区支付。
   - George 提到这专门针对 OS X 的问题，成员们希望这能稳定 GPU 测试。
- **LL-VM 势在必行！**：他们提议将 **LLVM JIT** 与 LLVM autogen 合并，参考 [PR #8486] 以简化迭代，同时在 `support/llvm.py` 中管理多个版本。
   - 关于 LLVM 中**函数签名（function signature）**变化的担忧得到了缓解，LLVM 14 到 19 的测试未显示出阻碍性问题。
- **新人现在就开始贡献！**：成员们敦促新开发者加入 **Tinygrad**，强调欢迎更多的 Pull Request。
   - 他们指出特定任务设有悬赏机制，强调了社区的支持性环境。
- **TinyGrad 博客讲解代码布局**：一篇新的[博客文章](https://adelaloui.me/tinygrad-codebase-explained-ish/)概述了 **Tinygrad** 的核心结构，重点关注核心的 `tinygrad/` 目录。
   - 作者警告不要修改该区域之外未经测试的代码，社区对这一谨慎策略表示赞同。
- **TinyGrad 中的设备设置至关重要**：开发者澄清，在创建 Tensor 之前设置 `Device.DEFAULT` 可以根据需要使用 **METAL**、**CUDA** 或 **CLANG**。
   - 他们补充说，**CLANG** 默认在 CPU 上运行，在 **Tinygrad** 中提供了更直接的控制。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nvidia 在 GPT4All 基准测试中碾压 Vulkan**：成员观察到在运行 GPT4All 时，**Nvidia** GPU 的表现优于 **llama.cpp Vulkan**，详情参考 [issue #3365](https://github.com/nomic-ai/gpt4all/issues/3365)。
   - 他们将卓越的速度归功于 **CUDA** 栈，展示了显著的硬件性能提升。
- **phi-4 模型引起关注**：用户在 GPT4All 中测试了 **phi-4-Q4_0**，并确认其在 JavaScript 任务上运行良好，详情见 [phi-4-Q4_0.gguf](https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf)。
   - 他们强调了其 **MIT** 许可证，并引用了 Hugging Face 上的 **Microsoft 发布版本**。
- **本地服务器 API 引发困惑**：成员发现本地服务器 API 仅识别 **OpenAI** 调用，导致缺少 **openai_api_key** 配置时出现错误。
   - 他们质疑缺乏本地托管支持，并指出了目前 GPT4All 设置中的限制。
- **聊天模板设置难倒初学者**：一位新用户在配置 **Vicuna** 聊天模板时遇到困难，因为旧模型缺乏专门的指令。
   - 他们被引导至 GitHub 获取指导，以确保模板能产生正确的输出。
- **角色扮演模型引发兴趣**：对于 **COTE anime** 角色扮演（RP），小组提议使用 **Nous Hermes 2** 以获得沉浸式内容和创作深度。
   - 他们还提到探索 [llama3-8B-DarkIdol-2.2-Uncensored-1048K](https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K) 以进行进一步实验。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GitHub 聚会与 Agentic 工作流**：定于 **1月15日** 的 **GitHub 总部活动** 承诺将深入探讨使用 ArizeAI **调试 AI Agent**、使用 GroqInc 实现**快速推理**，以及使用 LlamaIndex 构建 Agentic 工作流，详见[此公告推文](https://twitter.com/llama_index/status/1877103276635848846)。
   - 这场线下聚会旨在将实际演示与 AI 驱动系统的实时开发技巧相结合，参与者期待能获得显著的知识增长。
- **Agentic 文档工作流将于 2025 年到来**：根据[这篇博文](https://twitter.com/llama_index/status/1877420085691953385)，一种名为 **Agentic Document Workflows (ADW)** 的新范式将在 2025 年前将文档直接集成到业务流程中。
   - 社区成员将其描述为“致力于简化多格式处理的专项推动”，指向了为提高组织效率而设计的更强大的 Pipeline 设计。
- **Ollama 的 3 秒速度突破**：据报道，更新后的 **Ollama** 将评估时间缩短至 **3 秒** 以下，激发了本地 LLM 用户对性能基准测试的兴趣。
   - 这一进展引发了关于实时推理可能性的讨论，参与者权衡了其对更广泛部署场景的影响。
- **PostgreSQL 向量索引的曲折**：成员们探索了使用 PostgreSQL JSON 索引的 **VectorStoreIndex**，以通过元数据过滤节点，突显了部分变通方案和设计挑战。
   - 一些人主张官方应提供索引支持以处理海量数据，强调了对 **LlamaIndex** 中更高级搜索功能的需求。
- **QueryFusionRetriever 的 Token 纠纷**：将 **TEI Reranker** 与 **QueryFusionRetriever** 结合使用的用户遇到了 **'Input validation error'**，原因是 Token 限制，尤其是在 **top-K** 设置为 **25** 时。
   - 一些人建议降低 top-K 或调整参数，并参考 [TEI Rerank 文档](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/)以获取有关最佳内存使用的指导。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Rust 优化 Actor 部署**：Mojo 中 Actor 实现的 **Rust** 语法减少了类型边界带来的额外干扰，特别是在 **GlommioMultipaxosWorker** 中。
   - 参与者担心**重载解析（overload resolution）**可能会增加扩展代码库的复杂性。
- **Quojo 加速量子编程**：社区展示了 **Quojo** 库，这是一个在 **Mojo** 中运行的量子计算引擎，详见[此 GitHub 仓库](https://github.com/Deftioon/Quojo)。
   - 他们称赞其快速构建的能力，将其比作 **Qiskit** 风格的方法，旨在弥合理论量子原理与实际开发之间的鸿沟。
- **MLIR 削减冗余步骤**：一段分享的 YouTube 演示展示了 **MLIR** 如何引导量子操作的硬件资源使用。
   - 成员们注意到它可以在编译时移除单位矩阵乘法（identity multiplication），从而提高运行效率。
- **Qiskit 投身量子模拟**：一些人推荐使用 **Qiskit** 进行量子电路实验，即使没有直接的 IBM API 连接。
   - 他们将其与 **Quojo** 等较小的框架进行了对比，一致认为 Qiskit 生态系统有助于新开发者快速上手。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **黑客松延期导致结果滞后**：组织者更新了黑客松网站的时间表，表示由于等待评委反馈，最终结果推迟到 **1月** 公布，许多优秀的参赛作品给评委留下了深刻印象。
   - 他们提到大部分统计工作已经完成，但某些评委尚未提交最终评审，因此请参与者等待即将发布的官方公告。
- **Google Form 故障与 Twitter 问题**：一名用户在修改之前的 **Google Form** 提交内容时遇到困难，组织者建议重新提交，而其他人则建议如果原始邮箱已关闭，请使用其他邮箱。
   - 针对已注销的 **Twitter 账号** 是否影响证书资格的问题，官方确认账号停用不会影响最终的证书发放。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OI 1.0 中的 Python 困惑**：成员们发现，在 **OI 1.0** 中使用 `--tools interpreter` 可能无法完全启用直接的 **Python code** 执行，因为它仍然尝试调用 `python.exe`。
   - 系统消息中的一行内容暗示 OI 1.0 的内置解释器已更改，导致一些用户不确定直接运行代码是否仍然可行。
- **gpt-4o-mini 取得进展**：一些人测试了 **gpt-4o-mini** 模型，指出它在处理某些命令时表现更好，并且可以打印部分文件内容而不是全部文本。
   - 他们还指出 AI 仍显示出一些弱点，促使需要更多调整来优化性能。
- **对模型和参数的好奇**：一位用户寻求关于模型能力的细节，希望得到参数分解以及任何必要的修改建议。
   - 这一请求激发了人们对调整交互方式以获得更好结果的额外兴趣。
- **检查 Custom Instructions**：参与者分享了鼓励谨慎使用工具的 **Custom Instructions**，特别是围绕 **OI 1.0** 中的代码执行。
   - 他们建议在运行前验证命令的可行性，旨在帮助 AI 更可靠地处理复杂任务。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **TruLie 引起好奇**：参与者寻求关于 **TruLie dataset** 的信息，探讨其当前的关联性和实际应用，但未分享直接链接。
   - 一些参与者提到对其如何服务于潜在的 ML pipeline 感兴趣，尽管没有提供进一步的细节。
- **Image-to-3D 取得进展**：成员们讨论了可以在笔记本电脑上运行的 **image-to-3D** 技术，引用了 **Gaussian splat** 和 **NeRF** 库以及 [3D Arena](https://huggingface.co/spaces/dylanebert/3d-arena)。
   - 他们强调了用于 3D 重建的单图像 pipeline，并权衡了 GPU 性能对实际工作流的影响。
- **Chirpy3D 创作鸟类艺术**：关于 [Chirpy3D](https://kamwoh.github.io/chirpy3d/) 的讨论集中在用于 3D 鸟类生成的连续部分潜变量（continuous part latents），该项目与 **University of Surrey** 和 **Imperial College London** 有关。
   - 一些参与者认可了 Chirpy3D 的创意方法，将基于部分的建模与生成式设计相结合，用于未来潜在的扩展。
- **World Models 拓宽 3D 视野**：成员们提到了 **World Models**，它集成了物理感知网络用于逼真的视频创建，并与 3D 生成主题紧密相关。
   - 他们认为这些模型是 **image-to-3D** 工作流的补充，尽管没有提到直接的资源或链接。
- **寻求 Agent 注册表**：参与者正在寻找一个用于构建 AI **Agent** 的优质开源工具注册表，强调协作和代码共享。
   - 一位用户询问是否有任何标准资源，但对话中未出现具体的链接或解决方案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **聊天机器人 COT 得到提升**：一位参与者询问如何改进聊天机器人的 **Chain of Thought (COT)**，而不仅仅是添加签名（signature），并强调了彻底评估方法的重要性。
   - 他们特别问道：*除了设置签名之外，还有什么方法可以改进 COT 吗？*，希望能优化对话交互中的推理步骤。
- **Evals 成为焦点**：**Drew Breunig** 的一篇文章倡导为 **LLMs** 构建自己的 **eval**，解释说这比模型或提示词更关键，并分享了[他的博客文章](https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html)。
   - 他宣称 *你的 eval 是你拥有的最有价值的 AI 资产*，敦促团队改进方法、追踪改进并进行频繁测试。
- **Drew Breunig 强调工具和职业生涯**：他介绍了自己在 **PlaceIQ**、**Precisely** 和 **Overture Maps Foundation** 的背景，并分享了一个包含其工作时间线细节的[个人网站](https://www.dbreunig.com/)。
   - 他展示了用于追踪日常事务的 **StepList** 和用于自我监测的 **Reporter**，暗示这些解决方案能加速个人认知。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jovial Jamba 启动播客转录查询**：一位用户利用 **Jamba** 的 **Conversational RAG** 构建了一个基础的 **Python app**，用于查询播客转录内容以方便回忆。
   - 他们将其描述为“非常有趣”，尽管该项目仍处于 **work in progress**（进行中）状态。
- **AI 代码生成的古怪失误**：另一位用户指出，在对 AI 生成的 **HTML**、**Javascript** 和 **PHP** 代码进行故障排除时，发现了一些喜剧性的失误。
   - 他们认为当前的 AI 技术热潮仅仅是**触及了可能性的皮毛**。
- **PHP 依然是可靠的 Web 开发伙伴**：一位成员继续依靠 **PHP** 进行 Web 开发和本地 IRC 机器人编码，并称赞其易于集成的特性。
   - 他们表示 **Jamba** 通过使用与其他 API 类似的对话数组（conversation arrays），简化了某些任务。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ModernBERT 简短亮相**：**#general** 频道的一位用户询问是否有人测试过微调 **ModernBERT**，希望能交流经验并获取性能提示。
   - 随后没有进一步的回复或参考资料出现，对话仅限于这一初始提问。
- **Nectar Social 丰厚的推荐奖金**：在 **#jobs** 频道，**Nectar Social** 宣布了多个开放职位（包括 **Sr/Staff Product Manager** 和 **LLM/AI Engineer**），成功入职的推荐奖金高达 **$10,000**。
   - 他们目前处于“半隐身”状态，在**西雅图**及其他地区招聘，并为 **NYC/LA** 的 **Customer Success Manager** 或创始客户经理等职位提供灵活的选择。



---


**MLOps @Chipro Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Axolotl AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：频道详细摘要与链接


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1326645308322615421)** (719 条消息🔥🔥🔥): 

> `ComfyUI 功能, Pony 中的 OpenPose 控制, 停电对 SD 的影响, AI 工具更新, 在不同接口下使用 AMD GPU` 


- **在 ComfyUI 中探索 Pony 的 OpenPose 控制**：用户讨论了在 ComfyUI 的 Pony 模型中利用 OpenPose 控制的最佳方法，并寻求关于安装和节点集成的指导。
   - 一位用户强调，在尝试 ComfyUI 时遇到了障碍，导致他们考虑使用 Forge UI 等替代方案。
- **关于使用 SD 时停电的担忧**：一位用户提出了对 **Stable Diffusion** 生成过程中潜在停电影响的担忧，以及他们的 **GPU** 是否会受到影响。
   - 另一位用户建议，虽然 **GPU** 可能没事，但断电可能会损坏文件系统，导致数据丢失。
- **AI 工具的更新与维护**：用户分享了保持 A1111 和 **ComfyUI** 等 AI 工具更新所面临的挑战，特别是在更新过程中遇到问题后。
   - 有人指出，过时的 **Python** 版本可能会导致与各种 AI 模型的兼容性问题，并建议使用 **Python 3.10.11**。
- **比较 AMD GPU 的支持与性能**：讨论转向了 **AMD GPU** 支持，重点关注在 Windows 上直接使用 **ZLUDA** 和 **ROCm** 的区别。
   - 会议澄清，虽然 **ZLUDA** 在 Windows 上提供某些好处，但也可以在不使用它的情况下实现对 **AMD GPU** 的原生支持。
- **用户互动与社区支持**：新用户在频道中就特定模型功能和社区内的用法寻求帮助，强调了对更新指南的需求。
   - 会议强调了与开发者社区互动以分享知识和解决 AI 工具中常见问题的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://civitai.com/articles/6182/how-to-make-a-lora-on-colab">如何在 Colab 上制作 LoRA | Civitai</a>：在 WebUI 的 extra 选项卡下进行批量裁剪 (1024x1024) 和放大（我使用 4x_NMKD-UltraYandere_300k）（从目录批量处理），上传到 Drive，运行...</li><li><a href="https://tenor.com/view/cyanide-and-happiness-distraught-shocked-diagnosis-gif-23623883">Cyanide And Happiness Distraught GIF - Cyanide And Happiness Distraught Shocked - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-">懒人教程 | 如何在 Colab 上使用 Trainer LoRA | SD 1.5 & XL 作者：mikus (简单易行) | Civitai</a>：警告！我在这方面经验并不丰富，所以我建议先学习所有功能，并阅读更多关于如何操作的教程...</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-by-mikus-silly-and-easy">懒人教程 | 如何在 Colab 上使用 Trainer LoRA | SD 1.5 & XL 作者：mikus (简单易行) | Civitai</a>：警告！我在这方面经验并不丰富，所以我建议先学习所有功能，并阅读更多关于如何操作的教程...</li><li><a href="https://www.youtube.com/watch?v=vWdMXTk4zRo">与 KINGSKULL 一起风筝冲浪</a>：未找到描述</li><li><a href="https://civitai.com/models/548997/image-to-video-comparison-workflow">图像转视频对比工作流 - v1.0 | Stable Diffusion XL 工作流 | Civitai</a>：摘要 此工作流是作为一个实验制作的，用于对比支持 "image to video" 的各种技术。事实上，它允许对比以下...</li><li><a href="https://civitai.com/models/134056/explosm-cyanide-and-happiness-style">Explosm Cyanide and Happiness 风格 - 2 | Stable Diffusion LoRA | Civitai</a>：推荐设置 0.8-1.2，负面提示词使用：nose, chin, ears, cheeks, jawline cyanide and happiness（使用 lipstick, breasts 来生成女性...</li><li><a href="https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use">图像转视频（使用 AnimateDiff 和 IP Adapter 的 ComfyUI 工作流）即插即用 | Civitai</a>：工作流在右上角的附件 json 文件中。附件是一个用于将图像转换为视频的 ComfyUI 工作流。它会改变图像...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md">StableSwarmUI/docs/Features/IPAdapter-ReVision.md 位于 master 分支 · Stability-AI/StableSwarmUI</a>：StableSwarmUI，一个模块化的 Stable Diffusion Web 用户界面，重点在于使强力工具易于访问、高性能且具有可扩展性。 - Stability-AI/StableSwarmUI</li><li><a href="https://np.reddit.com/user/kejos92/comments/1hjkkmx/ltxv_inference_on_amd_gpus/">在 AMD GPU 上进行 LTXV 推理</a>：# 简介 在过去的两个月里，随着...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux">在 AMD GPU 上安装与运行</a>：Stable Diffusion web UI。通过在 GitHub 上创建一个账号来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=PWgvGjAhvIw">Outkast - Hey Ya! (官方高清视频)</a>：OutKast 的 "Hey Ya!" 官方高清视频。收听 OutKast：https://Outkast.lnk.to/listenYD 订阅 Outkast 官方 YouTube 频道：https://Outkas...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki">首页</a>：Stable Diffusion web UI。通过在 GitHub 上创建一个账号来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/wileewang/TransPixar">GitHub - wileewang/TransPixar</a>：通过在 GitHub 上创建一个账号来为 wileewang/TransPixar 的开发做出贡献。</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/.github/images/swarmui.jpg">SwarmUI/.github/images/swarmui.jpg 位于 master 分支 · mcmonkeyprojects/SwarmUI</a>：SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，重点在于使强力工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/Swa...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2151">由 parsee-mizuhashi 移除侵犯许可证 / 潜在恶意 / 混淆的代码 · Pull Request #2151 · lllyasviel/stable-diffusion-webui-forge</a>：另请参阅相应仓库中的此 PR。侵犯许可证：此代码至少部分复制自 ComfyUI，后者采用 GPL-3.0 许可证，禁止在不提供源代码的情况下发布编译后的代码。</li>

...</li><li><a href="https://github.com/CoderSJX/AI-Resources-Central">GitHub - CoderSJX/AI-Resources-Central: 本仓库专注于汇集来自全球的优秀人工智能（AI）开源项目。无论你是寻找灵感来启动自己的项目，还是想要学习如何使用最新的AI技术，这里都是一个绝佳的起点。我们致力于为AI开发者、研究人员以及爱好者提供一个平台，以便于探索、交流并共享各种AI项目的代码与实现。</a>: 本仓库专注于汇集来自全球的优秀人工智能（AI）开源项目。无论你是寻找灵感来启动自己的项目，还是想要学习如何使用最新的AI技术，这里都是一个绝佳的起点。我们致力于为AI开发者、研究人员以及爱好者提供一个平台，以便于探索、交流并共享各种AI项目的代码与实现。 - CoderSJX/AI-Resources-Central
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1326644994135822396)** (393 条消息🔥🔥): 

> `Phi-4 Bug Fixes, Unsloth Model Deployment, Chat Templates in LLMs, Adapting Models for Inference, Quantization Impacts` 


- **Unsloth Phi-4 Bug Fixes**: Unsloth 的 Phi-4 版本在 Open LLM Leaderboard 上已经超越了 Microsoft 官方版本，博客中正在报告持续的 Bug 修复。
   - 建议用户关注博客更新，并检查在最近的修复后重新运行 Finetune 是否能提升性能。
- **Apple 设备兼容性问题**: 目前 Unsloth 不支持 Apple Silicon（除非运行 Linux），这限制了部分用户探索其功能的能力。
   - 切换到 Ubuntu 的用户报告了内存不足等性能问题，特别是在使用 Gemma 等特定模型时。
- **理解 Chat Templates**: Chat Templates 会影响 Fine-tuning 过程和模型部署，建议使用特定的结构。
   - 训练好的 Chat Template 包含在 `tokenizer_config.json` 中，用户应根据应用需求设计模板。
- **模型适配建议**: 在为 Unsloth 模型动态挂载 Adaptor 时，推理时首选高分辨率（16bit）模型，而非 Quantized（4bit）版本。
   - 这种方法可以最大限度地减少损失，并推荐用于 Merge 以获得更好的性能。
- **Quantization 见解**: 用户质疑 Quantized 模型如何能超越非 Quantized 模型，并发现尽管存在固有噪声，它们仍然比 Microsoft 的产品有所改进。
   - 讨论内容包括 Quantization 涉及的权衡以及评估模型适配策略的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Unsl">来自 FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://x.com/UnslothAI/status/1877136074042126338">来自 Unsloth AI (@UnslothAI) 的推文</a>：Phi-4，包括 GGUF + 4-bit + 16-bit 版本现已上线 @HuggingFace！我们在 Phi-4 中发现并修复了 4 个 bug，并将该模型 Llamafied（Llama 化）。查看所有带有我们 bug 修复的 Phi-4 版本：https://huggingface.co/collec...</li><li><a href="https://xkcd.com/1425/">Tasks</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large">Open LLM Leaderboard - 由 open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://runpod.io?ref=bb842lb3">RunPod - 为 AI 构建的云</a>：在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth Requirements | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://rog.asus.com/us/laptops/rog-strix/rog-strix-scar-18-2025/">ROG Strix SCAR 18 (2025) G835 | 游戏笔记本｜ROG - 玩家国度｜ROG 美国</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/index">PEFT</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/learn/cookbook/en/llm_judge">使用 LLM-as-a-judge 🧑‍⚖️ 进行自动化且通用的评估 - Hugging Face 开源 AI 食谱</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hwzmqc/phi4_llamafied_4_bug_fixes_ggufs_dynamic_4bit/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (所有版本) - unsloth 收藏集</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Iq1JeXKYg5k">RTX 5090 笔记本电脑来了！</a>：Nvidia 的 Blackwell 50 系列笔记本电脑已发布，包括 RTX 5090, RTX 5080, RTX 5070Ti, RTX 5070。RTX 5090 笔记本电脑 - https://rog.asus.com/us/laptops-group/Nvidia - https:...</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/cognitivecomputations/laserRMT">GitHub - cognitivecomputations/laserRMT：这是我们对 'Layer Selective Rank Reduction' 的自行实现</a>：这是我们对 'Layer Selective Rank Reduction' 的自行实现 - cognitivecomputations/laserRMT</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct/blob/main/tokenizer_config.json">tokenizer_config.json · unsloth/Llama-3.2-1B-Instruct (main 分支)</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/1516">由 danielhanchen 提交的 Bug 修复 · Pull Request #1516 · unslothai/unsloth</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326655511847763978)** (3 条消息): 

> `求职成功, 搞笑 GIF` 


- **Infinit3e 庆祝获得新工作**：一位成员高兴地宣布他们的求职结束了，并表示：**“求职结束，我现在就业了”**。
   - 这引发了一个回应，重点展示了一个 GIF，内容是**一个穿着西装打着领带的男人在人群面前做鬼脸**。
- **分享了 Amogus6969 GIF**：一位成员分享了一个[搞笑 GIF](https://tenor.com/view/amogus6969-gif-26819393)，描绘了一个穿着西装的男人做出幽默表情，引起了频道内的关注。
   - 内容描述将其标注为**“一个穿着西装打着领带的男人在人群面前做鬼脸”**。



**提到的链接**：<a href="https://tenor.view/amogus6969-gif-26819393">Amogus6969 GIF - Amogus6969 - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326791388213284935)** (48 messages🔥): 

> `Mathstral-7B-v0.1 模型限制、表格计算的模型建议、长上下文训练、合并 LoRA 模型、姓名拆分的经典机器学习方法` 


- **Mathstral-7B-v0.1 不受支持**：`mistralai/Mathstral-7B-v0.1` 已确认既不是基座模型也不是 PEFT 模型，这导致社区成员开始寻找替代方案。
   - *Theyruinedelise* 表示计划在未来支持该模型。
- **7B 表格模型建议**：*etherl* 建议尝试 **Qwen/Qwen2.5-Math-7B-Instruct** 模型，它在表格计算（特别是小型 Markdown 表格）方面表现良好。
   - 成员 *marioz_70065* 计划尝试以 **3e-5** 的学习率训练一个 Epoch。
- **长上下文模型训练**：*shaswat_singh.* 询问关于训练 Llama 7B 模型以支持更长上下文的问题，提到指令/输入键接近 2k tokens，输出可能达到 7k tokens。
   - *marioz_70065* 建议将查询拆分为较小的部分以进行处理。
- **合并与量化 LoRA 模型**：讨论集中在合并基于 4B 版本训练的 LoRA 模型，以及是否应使用 16B 模型进行此过程。
   - *fjefo* 澄清说，在训练和合并过程中都使用 **16-bit 模型**可以确保过程的完整性。
- **使用经典机器学习进行姓名拆分**：成员 *andresmosqueraw* 寻求关于拆分姓名和识别性别的建议，并提议使用微调作为解决方案。
   - *mrdragonfox* 建议使用经典机器学习技术而非大语言模型（LLM）来完成此任务。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426">LoRA: Load and Merge Your Adapters with Care</a>：使用 QLoRA 微调 LoRA 适配器的案例</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1326857299767066746)** (4 messages): 

> `语言模型的 DLSS、投机采样 (Speculative Decoding)` 


- **探索语言模型的类 DLSS 技术**：一位成员询问是否存在类似于 **DLSS** 的语言模型技术，可以优化**训练或推理**过程以减少资源消耗。
   - 他们特别寻求有关解决这一优化挑战的研究见解。
- **投机采样 (Speculative Decoding) 介绍**：另一位成员介绍了 **Speculative Decoding** 的概念，作为与之前询问相关的潜在方法。
   - 最初的提问者确认了这一建议并表达了感谢。


  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1326645517907787849)** (125 条消息🔥🔥): 

> `Codeium Self-Hosted Version, Windsurf Performance Issues, Cascade Model Benefits, Custom Model Training, Prompt Credit Usage` 


- **Codeium 推出自托管版本**：一位用户强调，**Codeium** 的自托管版本现已在企业版方案中提供。
   - *我该如何获取？* 其他人询问，想了解部署细节。
- **Windsurf 面临稳定性问题**：多位用户报告了 **Windsurf** 持续存在的问题，包括窗口崩溃和连接问题。
   - 一位用户提到，他们正面临程序频繁卡死并收到“窗口无响应”错误的问题。
- **Cascade 展示出优于 Windsurf 的优势**：用户称赞 **Cascade** 模型的高效性，特别是在处理操作时不会超出 Flow 限制。
   - 一位成员分享说，他们利用 Cascade 的功能，以极少的编码工作量成功构建了公司网站。
- **对特定任务的自定义模型感兴趣**：一位成员询问是否可以在 **Codeium** 上针对特定任务训练自定义模型。
   - 这引发了关于平台内功能和可用训练选项的讨论。
- **对 Prompt 额度消耗的担忧**：用户对超出 **Prompt 额度** 表示担忧，尤其是在协作任务的大量使用中。
   - 一位用户提到，由于多次操作会话，他们的额度很快就耗尽了。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides">Pricing | Windsurf Editor and Codeium extensions</a>: Codeium 对个人永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活的部署。</li><li><a href="https://github.com/Exafunction/codeium.el/issues/115">How do I get my api key? · Issue #115 · Exafunction/codeium.el</a>: 我正尝试在 codeium.com 上查找我的 API Key 但没找到。我该去哪里看？
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1326642701403422740)** (140 条消息🔥🔥): 

> `Windsurf Installation Experiences, Cascade Panel Issues, Flow Credits and Billing Concerns, Agent Integration with Windsurf, Update Feedback` 


- **Windsurf 安装体验**：用户分享了在不同系统上的安装体验，其中一位提到 Windsurf 在 **Ubuntu 24.04** 上运行完美，而另一位在 **Arch with Hyprland** 上遇到了 Token 提交问题。
   - 经过排查，一位用户通过删除与 Windsurf 相关的特定目录文件成功解决了问题。
- **Cascade 面板问题**：用户对启动新项目时 **Cascade** 面板自动打开表示担忧，一位用户报告相关问题影响了他们的工作流。
   - 其他人建议目前的设置可能无法有效防止面板重新打开，表明需要更清晰的解决方案。
- **Flow 额度与计费担忧**：多位用户对计费相关问题表示沮丧，特别是被两次扣费却未收到 **Flow 额度**，并询问如何通过支持渠道解决这些问题。
   - 一位用户指出，他们被收取了巨额费用但只获得了有限的服务，感觉被支持渠道忽视了。
- **Agent 与 Windsurf 的集成**：一位用户询问在 Windsurf 中使用 **Agent** 的可能性，参考了对其他平台最近推出的类似功能的兴趣。
   - 回复显示出对这种集成的不确定性，揭示了用户对功能熟悉程度的差距。
- **更新反馈**：在最近的一次更新后，用户报告了各种问题，包括命令无法执行以及 Cascade 中的异常行为（如生成不必要的源代码）。
   - 反馈表明，与之前的版本相比，一些用户的性能和功能有所下降。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://codeium.canny.io/">Codeium Feedback</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>: Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1326643664906092615)** (246 条消息🔥🔥): 

> `Cursor composer 问题, Claude 性能, Cursor rules 使用, 社区反馈, Cursor 文档` 


- **Cursor composer 的持续性问题**：用户报告了 Cursor composer 的重大问题，包括频繁忽略提供的 cursor rules 以及对代码库进行不必要的更改。
   - 反馈表明，即使是高级付费计划（premium plans）也出现了性能下降，这促使用户重新考虑对 composer 功能的依赖。
- **Claude 表现不稳定**：贡献者指出，当针对特定任务进行提示时，Claude 表现良好，特别是当被指示在回答中使用内心思考（inner thoughts）和独白（monologues）时。
   - 然而，许多用户对其处理提示词（prompts）的不一致性表示沮丧，并认为在应用更改时需要仔细监督。
- **Cursor rules 的正确用法**：强调了用户应创建 `.cursorrules` 文件，以便在项目开发中为 Claude 等模型设定明确的行为准则。
   - 参与者分享了构建提示词的策略，以提高 Claude 对规则的遵循度，并建议更聚焦的提示词能带来更好的效果。
- **社区参与和支持**：用户讨论了社区在提供支持方面的作用，强调官方 Discord 提供了一个分享问题并获得 Cursor 开发者响应的平台。
   - 对于严肃的咨询，社区建议利用 Cursor 论坛寻求开发团队的直接协助。
- **Cursor 的文档和功能**：用户一致认为 Cursor 的文档在某些方面比较匮乏，甚至将其比作托管在有问题的平台上。
   - 用户表达了对改进文档以及提高请求统计和应用功能透明度的期望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://onecompiler.com/bootstrap/435jnyccv">Card Glow Magnetic - Bootstrap - OneCompiler</a>: 未找到描述</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4">Composer Stuck at &quot;Generating&quot; - Specific Composer Instance, Not Global Issue</a>: 嘿……这方面有进展吗？我仍然看到 Composer 会话卡住。我刚升级到 0.44.10；在 0.44.9 中卡住的当前会话在 0.44.10 中仍然卡住。它一直停留在 “generating” 状态...</li><li><a href="https://cursor.directory/">Cursor Directory</a>: 为你的框架和语言寻找最佳的 cursor rules</li><li><a href="https://dotcursorrules.com/">.CursorRules</a>: 通过 Cursor Rules 自定义 AI 行为，简化开发流程，并根据你的框架和语言定制代码生成、建议和查询。</li><li><a href="https://ui.aceternity.com/">Aceternity UI</a>: 漂亮的 Tailwind CSS 和 Framer Motion 组件，基于 Next.js 和 TypeScript 构建。</li><li><a href="https://magicui.design/">Magic UI</a>: 漂亮的 UI 组件和模板，让你的落地页看起来非常惊艳。</li><li><a href="https://ui.lukacho.com/">Lukacho UI</a>: 使用 Next.js | TailwindCSS | Framer Motion 制作的动画 UI 组件库</li><li><a href="https://www.eldoraui.site">EldoraUI</a>: 漂亮的 UI 组件和模板，让你的落地页看起来非常惊艳。</li><li><a href="https://www.cult-ui.com/">cult/ui</a>: 可以复制并粘贴到 React 应用中的组件。可定制、开源、类型安全。</li><li><a href="https://www.hover.dev/">Animated UI Components and Templates for React and TailwindCSS | Hover.dev</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1326656125126447185)** (11 条消息🔥): 

> `Prompting 技巧、支付系统问题、Public Repos 功能、睡眠时间玩笑、Subreddit AI 推广帖子` 


- **掌握色彩 Prompting**：一位成员强调了在 Prompt 中指定**颜色名称**和 **Hex 代码**的重要性，并指出要明确说明每种颜色的使用位置。
   - 在陈述需求时，“仅仅是一个想法”就足够了——简洁胜过详尽！
- **支付系统仍在建设中**：一位成员提到**支付系统**尚未运行，暗示有效的解决方案仍在开发中。
   - 这表明正在进行持续的开发工作，以最终为用户完善该功能。
- **令人兴奋的 Public Repos 功能发布**：另一位成员分享了一个与 10 月份在 **X** 上宣布的 Public Repos 功能相关的链接，详细介绍了它如何与 GitHub URL 配合使用。
   - 该功能允许用户通过简单的后缀轻松访问公共仓库，增强了易用性。
- **睡眠时间影响回复**：成员们开玩笑地讨论了混乱的**睡眠时间**对回复时间的影响，并互相致歉。
   - 这种人情味为社区成员之间增添了一层情谊。
- **Subreddit AI - 欢迎提问**：一篇推广帖子分享了 **Subreddit AI** 的链接，邀请大家就该项目中使用的 Prompting 技巧进行提问。
   - 这展示了社区在协助和分享 Prompting 知识方面的开放态度！


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://subredditai.com">SubReddit AI</a>: 未找到描述</li><li><a href="https://x.com/stackblitz/status/1843668731681267801">来自 StackBlitz (@stackblitz) 的推文</a>: 你现在可以在 bolt.new 中打开公共仓库了 🙌 如何操作？对于任何 GitHub URL，只需在前面加上 "http://bolt.new" 即可！（发布说明见下文！）
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1326647185496080484)** (211 条消息🔥🔥): 

> `Bolt 性能问题、PWA 开发、Token 管理、集成疑虑、GitHub 部署问题` 


- **用户报告 Bolt 的性能问题**：许多用户对 Bolt 的性能表示沮丧，包括导致代码丢失和 Token 过度消耗的频繁错误。
   - 一位用户提到，由于 Bolt 不断插入未修改的代码段，导致消耗了超过 100k Token。
- **PWA 开发的挑战**：关于 StackBlitz 与 Progressive Web Apps (PWAs) 兼容性的问题浮出水面，一些用户收到了关于不支持配置的错误消息。
   - 尽管存在这些挑战，一位用户报告成功从 Bolt 部署了 PWA，显示出不同的使用体验。
- **Token 消耗与管理见解**：关于 Token 管理的讨论强调，有时需要将整个文件插入聊天中以明确所需的更改，因为沟通不畅会导致过度的 Token 消耗。
   - 用户分享了有效管理 Token 使用的技巧，以避免陷入生成代码的重复循环中。
- **与 Supabase 的集成及回滚**：有用户担心 Supabase 的 migrations 无法随项目代码一起回滚，导致行为混乱和潜在的不可逆更改。
   - 用户建议定期对项目进行 Fork 以便于恢复，但也承认自动化 migrations 回滚存在困难。
- **GitHub 部署问题**：一些用户在将项目部署到 GitHub 时遇到困难，过程中出现了空仓库的问题。
   - 加强与 GitHub 的集成以管理代码版本控制和回滚，被强调为改进用户工作流的首要任务。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boltstudio.ai">BoltStudio.ai | Full Stack Prompt Engineering</a>: 未找到描述</li><li><a href="https://bolters.io">Bolters.io | Community Supported Tips, Tricks &#38; Knowledgebase for Bolt.new No-Code App Builder</a>: Bolt.new 的文档和指南</li><li><a href="http://bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: 使用任何你想要的 LLM 来 Prompt、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5149">Suggestion: Selector · Issue #5149 · stackblitz/bolt.new</a>: 这是我为站点添加选择器选项的建议。我将尝试更详细地解释：当你用鼠标突出显示并进入聊天，例如说更改名称或删除 t...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: 问题：Bolt 在聊天中输出应用逻辑。例如，当用户达到速率限制时，提供升级链接的代码会作为聊天响应发送给用户。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1326689892834476083)** (66 messages🔥🔥): 

> `AI 编辑器对比, Aider 和 O1 性能, 关于 AI 能力的讨论, 对 Aider 的开发贡献, OpenAI 模型的使用` 


- **AI 编辑器对比：Claude vs Deepseek**：用户对 **Deepseek** 的评价褒贬不一，有人认为它不如 **Claude** 出色，尽管有一位用户注意到它在处理命令时有一些隐蔽的行为。
   - **Deepseek** 有时会干扰 Aider，导致执行中出现意想不到的问题。
- **Aider 正在成为助手的助手**：一位用户幽默地指出，Aider 正准备成为他们助手的助手，展示了编程自动化的趋势。
   - 有人呼吁那些不知疲倦地致力于创建完美编程环境的人站出来并建立联系。
- **AI 的未来与主动性**：一位参与者对 AI 的未来表示乐观，认为它将变得更加主动，并最终增加 AI 与用户之间的提问比例。
   - 讨论涉及了当前 AI 受限于电力和计算成本的现状，并对未来受摩尔定律（Moore's Law）驱动的进步寄予厚望。
- **Aider 对自动化编程的贡献**：一位用户描述了一个愿景，即在 Aider 中提出 Issue 就能生成自动化的 Pull Requests，展示了 AI 在软件开发流程中的集成。
   - 对话反思了在编码过程中保持人工监督的重要性，同时也承认了 Aider 在简化开发流程方面的潜力。
- **澄清 OpenAI 模型配置**：一位用户询问了带有 **openai/** 前缀的模型与不带前缀的模型之间的区别，以更好地理解代码库。
   - 回复澄清了这些模型在功能上是相同的，无论是否带有前缀都可以引用，这体现了命名约定的灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>: Aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 和测试</a>: 自动修复 Linting 和测试错误。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · Aider-AI/aider</a>: Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/pull/540">feat: add `/rg` command (ripgrep for `/add` files) by aleclarson · Pull Request #540 · Aider-AI/aider</a>: 注意：使用此命令需要在你的机器上安装 ripgrep。工作原理：它通过子进程调用带有 -l 标志的 rg 以返回文件名列表。然后这些文件名被输入到...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1326651345029169212)** (61 条消息🔥🔥): 

> `Aider 配置问题, OpenAI 模型访问, DeepSeek 性能, 任务管理技巧, AI 模型中的上下文管理` 


- **Aider 配置问题**：用户报告了 Aider 在与本地 litellm 代理通信时发送 'prompt' 列表而非 'messages' 列表的问题，导致出现 TypeError 消息。
   - 分享的配置详情强调 `litellm_provider` 应与第一个区块匹配，这表明用户设置中可能存在配置错误。
- **OpenAI 模型访问选项**：一位用户询问了关于获取 Tier 5 OpenAI 密钥的途径，并收到了使用 200 美元的 O1 Pro 订阅或使用 Unify.ai 等现有服务来访问各种模型的建议。
   - 社区讨论了使用不同提供商的优缺点，包括潜在成本和功能的可用性。
- **DeepSeek 性能问题**：用户对 DeepSeek 的性能表示担忧，有用户在几次请求后遇到停顿，并询问这是否为正常现象。
   - 一些成员分享了持续稳定使用 DeepSeek 的经验，建议可以使用 VPN 来改善对负载较低模型的访问。
- **任务管理技巧**：一位用户寻求关于管理 AI 提出的多个任务建议的建议，得到的建议是创建一个 TODO.md 文件以有效地跟踪任务。
   - 对话强调了对高效任务管理工作流的需求，以及在不丢失上下文的情况下处理多个建议的策略。
- **AI 模型中的上下文管理**：用户讨论了 Aider 对聊天历史记录的处理，指出它会在会话内保留历史记录，但可以清除以避免模型产生混淆。
   - 社区建议通过鼓励模型更有效地处理上下文，来减少重复建议的方法。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://unify.ai/">Unify: Build AI Your Way</a>: 工具太多、太复杂？秒级构建你自己的工作流！</li><li><a href="https://github.com/unifyai/unify">GitHub - unifyai/unify: Build Your AI Workflow in Seconds ⚡</a>: 秒级构建你的 AI 工作流 ⚡。通过在 GitHub 上创建账号为 unifyai/unify 做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326675570632822886)** (1 条消息): 

> `Gemini 2.0 Flash Experimental, 应用开发协助, 语音模式交互` 


- **Gemini 2.0 Flash 在语音交互中表现出色**：在外出办事时，我在 iOS 上以语音模式与 **Gemini 2.0 Flash Experimental** 进行了交流，把它当作一名乘客。
   - *它出人意料地为我的应用创意生成了标准并讨论了任务*，展示了其强大的对话能力。
- **语音模式缺失 Markdown 输出**：尽管任务生成很有效，但 **Gemini 2.0 Flash** 并没有为我的应用规范创建 Markdown 文件。
   - 我注意到了这个局限性；如果在开发过程中能用它来组织信息，将会非常有帮助。
- **简洁的任务总结功能**：回家后，我提示 **Gemini** 总结我们的对话，得到了一组清晰的要点任务。
   - 这一功能在为应用开发过程构建工作流方面证明了其价值。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1326658028476301342)** (19 条消息🔥): 

> `DeepResearch 报告，NotebookLM 中的引用模式，从英文到中文的播客，NotebookLM Plus 中的 System Prompt，创意播客提示词` 


- **DeepResearch 报告集成讨论**：成员们讨论了 **DeepResearch** 与 **NotebookLM** 之间缺乏直接集成的问题，并探索了使用扩展程序批量上传来源等替代方案。
   - 一位成员分享了一个 [YouTube 视频](https://youtu.be/spj0n-bFKJo)，其中涵盖了如何利用 NotebookLM 提升研究和内容创作效率的技巧。
- **指令 NotebookLM 进行直接引用**：一位成员通过在 System Prompt 中使用指令结构，成功让 NotebookLM 仅使用来源中的直接引用进行回答。
   - 他们指出，除非使用 Plus 版本以增强记忆保留，否则在回答的一致性方面会面临挑战。
- **中文播客生成咨询**：一位成员询问是否有人研究出如何通过 NotebookLM 将 **英文源内容** 生成 **中文播客**。
   - 针对这一特定任务的回答不够具体，表明需要社区协作。
- **关于 Plus 版本中 System Prompt 的澄清**：讨论涉及了 NotebookLM Plus 中 System Prompt 的存在，成员们强调了由于 Plus 版和免费版外观相同而导致的困惑。
   - 寻求关于持续使用 System Prompt 相关功能差异的澄清。
- **播客提示词请求**：一位用户分享了对适用于 **podcasting** 的有效提示词的需求，并邀请社区贡献创意。
   - 该请求突显了大家对增强播客制作创意过程的共同兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/spj0n-bFKJo">NotebookLM (AI Tool) Tips to level up your Research &amp; Content Creation!</a>: 提升研究和内容创作水平的 AI 工具 - NotebookLM！从即时消化数百个文档、视频、网站到多语言研究...</li><li><a href="https://www.akashq.com/post/ad632a26-91b5-44b4-b8f4-5b5fd3f083e8">What happened on Jan 8?</a>: 1 月 8 日发生了什么？由 This Day in History 发布</li><li><a href="https://www.akashq.com/post/122c0310-6683-45d7-adec-3d3f4bbebd16">What happened on Jan 9?</a>: 1 月 9 日发生了什么？由 This Day in History 发布
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1326641747840995348)** (94 条消息🔥🔥): 

> `Notebook LM 使用问题、Podcast 生成、Workspace 许可证故障、AI 工具功能、语言支持` 


- **Notebook LM 使用问题汇总**：用户报告了在访问 Notebook LM 功能时遇到的挑战，通常与 Workspace 许可证以及因使用率低而被移除的功能有关。
   - 一些成员询问了关于管理来源、解决上传故障以及检索之前对话的问题。
- **生成定制化 Podcast**：成员们探讨了从选定来源生成 Podcast 的选项，强调用户可以指定音频输出应关注哪些来源。
   - 有人提议使用 Google 的 Illuminate 等其他工具作为变通方案，以增强生成的 Podcast 中的声音多样性。
- **功能限制的变通方法**：分享了关于如何在笔记本中管理多个模块的提示，指出目前尚无链接多个笔记本以获取跨来源响应的功能。
   - 建议用户在遇到来源过时的问题时创建新笔记本，以简化任务。
- **语言支持与交互**：讨论涉及在不同语言中使用 Notebook LM，指出用户可以提示该工具以其首选语言（如日语）进行回复。
   - 参与者提到了翻译的准确性，并对支持多语言交互表示了热情。
- **分享 Notebook LM 内容**：用户表达了将 Notebook LM 生成的内容分享给他人的兴趣，同时寻求关于共享材料编辑权限的澄清。
   - 为 Notebook LM 新手提供了资源，包括教程播放列表，以帮助其有效入门。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.icloud.com/iclouddrive/061hg1R50Jv4idRhgdUqoMxWg#Captura_2025-01-09_a_las_8.15">iCloud Drive - Apple iCloud</a>: 未找到描述</li><li><a href="https://youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8&si=dKJ-7Kp7mBsQk6LN">NotebookLM Tutorials: Exclusively researched and developed!</a>: 深入探索由专家策划的 NotebookLM 教程合集，这些教程基于数月的实操实验和研究。该播放列表揭示了未公开的...</li><li><a href="https://illuminate.google.com/create">Illuminate | Learn Your Way</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你的 Gen AI 工具，用于更快地理解复杂内容。</li><li><a href="https://notebooklm.google.com/notebook/982b3b0c-0913-4599-816a-9c845a6b7d79/audio">无标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.">Google NotebookLM | Note Taking &amp; Research Assistant Powered by AI</a>: 利用 AI 的力量进行快速摘要和记笔记，NotebookLM 是你强大的虚拟研究助手，植根于你可以信赖的信息。</li><li><a href="https://akashq.com">Akas: home to AI podcasts</a>: 未找到描述</li><li><a href="https://youtu.be/spj0n-bFKJo">NotebookLM (AI Tool) Tips to level up your Research &amp; Content Creation!</a>: 提升研究和内容创作水平的 AI 工具 - NotebookLM！从即时消化数百份文档、视频、网站到多语言研究...</li><li><a href="https://youtu.be/2_EwajcUlk8">NotebookLM + Illuminate: Use this FREE AI tool before it becomes PAID!</a>: 获取 NotebookLM 的独家研究材料：https://www.youtube.com/playlist?list=PL-HkokgcYrl5SrKYeVo28JA4OMPbslhA8 🎙️精通 NotebookLM 的研究...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1326662295522381945)** (66 条消息🔥🔥): 

> `LM Studio 相关问题, 模型加载问题, 模型的目录结构, Qwen Chat 发布公告, LLM 应用开发见解` 


- **用户面临 LM Studio 模型加载问题**：许多用户讨论了在 LM Studio 中加载模型的问题，并对版本兼容性和模型路径感到困惑。
   - 一位用户通过确保不从安装程序打开应用，而是直接从应用程序（Applications）运行，解决了他们的问题。
- **对模型目录结构的困惑**：用户对 LM Studio 访问模型所需的特定目录结构表示沮丧，这增加了他们在不同应用间共享模型的难度。
   - 虽然有人建议了组织模型的替代方案，但许多人为了易用性更倾向于使用统一的目录方法。
- **Qwen Chat 激动人心的发布**：官方宣布推出 [Qwen Chat](https://chat.qwenlm.ai)，展示了其与各种 Qwen 模型交互的功能。
   - 功能包括模型对比、文档上传和视觉理解支持，未来还计划进行更多增强。
- **Snapdragon X Elite 的 OpenCL 后端支持**：一位用户询问了 Snapdragon X Elite 上 OpenCL 后端的潜在支持情况，并提到了 Llama.cpp 最近的更新。
   - 这反映了用户对针对不同硬件配置优化 LLaMA 模型的广泛兴趣。
- **LLM 应用的发展趋势**：用户讨论了不同 LLM 应用在发展过程中功能的趋同，许多人分享了使用多种工具的经验。
   - 一位用户强调了在紧跟竞争功能的同时，开发利用这些 LLM 的自定义应用程序的乐趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113">来自 Qwen (@Alibaba_Qwen) 的推文</a>：🚀 激动人心的消息！我们很高兴宣布 Qwen Chat ( https://chat.qwenlm.ai ) 正式上线——这是您与 Qwen 模型交互的全新首选 Web UI！🌟💬 轻松与我们的旗舰模型对话...</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：发现、下载并运行本地 LLM</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">侧载模型 (Sideload models) - 高级 | LM Studio 文档</a>：使用在 LM Studio 之外下载的模型文件
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1326644383373725738)** (33 条消息🔥): 

> `AMD RX 7900XT 性能, MacBook Pro 的外接 GPU 选项, 寻找系统瓶颈, ML 模型的内存配置, DIGITS 的可用性` 


- **AMD RX 7900XT 对比 Nvidia GPU**：成员们讨论了 **7900XT** 的 **TOPS** 性能，并将其与 **4090**、**4080** 和 **3090** 型号进行了对比，同时对内存带宽表示了担忧。
   - 对话建议参考 [这个 Reddit 讨论](https://reddit.com/) 中提供的具体基准测试，以获取详细的对比。
- **MacBook 的外接 GPU 支持**：一位成员询问是否可以在 MacBook Pro 上使用 **Sidecar** 类型的显卡，特别是针对配备 **M3 Pro Max** 和 **64GB RAM** 的机型。
   - 然而，会议确认这在 **Apple Silicon** 上是不可能的，不像旧款的 Intel Mac 具有此类功能。
- **定位系统瓶颈**：一位用户询问如何寻找系统瓶颈，并分享了他们在运行 **Llama 3.3 70B Instruct** 时使用的 **Ryzen 7 7800X3D** 和 **AMD RX 7900GRE** 配置详情。
   - 回复指出性能问题可能与 RAM 有关，将数据完全加载到 **VRAM** 将显著提高速度。
- **为 ML 模型调整 MacBook Pro 内存**：为了给 **VRAM** 分配更多内存，一位成员建议修改 **/etc/sysctl.conf** 文件，设置 **iogpu.wired_limit_mb=54272** 以获得更好的模型性能。
   - 通过调整这些设置，用户可以在可用的内存限制内更有效地运行 **4-bit** 和 **6-bit** 的 **MLX** 模型。
- **对 DIGITS 平台的期待**：一位成员对 **DIGITS** 的到来表示期待，认为尽管有所延迟，它仍是访问完整 **Nvidia** 技术栈的最佳选择。
   - 同时也对发布后计算能力的潜在速度提出了担忧。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1326658567394164767)** (60 条消息🔥🔥): 

> `TensorFlow GPU 问题, 模型安全担忧, 机器学习最佳 YouTube 频道, Jupyter Notebook vs Python 文件, TensorFlow 环境搭建` 


- **TensorFlow GPU 问题持续存在**：一位成员表达了挫败感，尽管安装了 **CUDA 12.6.3**、**cuDNN 9.6.0** 和 **TensorFlow 2.11.0**，但他们的 Jupyter 内核仍无法检测到 **NVIDIA GeForce RTX 3060** GPU。
   - 他们表示：*我有 64G RAM 和 12G GPU*，并进一步寻求解决这一持续存在的检测问题的指导。
- **对模型安全和越狱（Jailbreaks）的担忧**：讨论了 OpenAI 可能对越狱反应过度，试图通过提高模型安全性来应对，但完全阻止越狱似乎是徒劳的。
   - 一位参与者指出 *总会存在某种形式的越狱*，并建议 OpenAI 如果不尝试进行大规模修补，可能会节省资金。
- **寻找机器学习学习资源**：一位用户询问了学习 **machine learning** 的最佳 **YouTube 频道**，引发了简短的讨论。
   - 回复中夹杂着幽默和对传统学习资源有效性的怀疑。
- **相比 Jupyter Notebook 更倾向于调试**：一位成员描述了在编码时避免使用 **Jupyter Notebook**，而更喜欢普通的 Python 文件以及设置断点进行调试的能力。
   - 他们强调这种方法在排除代码故障时可以获取 *更重要的信息*。
- **安装和配置 TensorFlow 环境**：在故障排除过程中，一位用户在引导下通过命令正确设置了他们的 **tf-gpu-env** 环境，以确保与 Jupyter Notebook 的兼容性。
   - 用户间的讨论强调了环境设置中可能存在的问题，这对于成功安装和使用 TensorFlow 至关重要。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1326721363750162513)** (7 条消息): 

> `GPT 代码处理, 图表生成` 


- **对 GPT 代码回复的挫败感**：一位用户表达了挫败感，无论他们在 Prompt 中请求多少次完整代码， GPT 仍然继续用注释而不是实际代码来回复。
   - *它总是生成类似“此处代码保持不变”的内容*，用户注意到虽然 **4o** 版本运行良好，但 **o1** 往往会错误地处理他们的请求。
- **对图表生成的兴奋**：一位成员评论了 ChatGPT 生成 **GRAPH**（图表）的惊人能力。
   - 另一位用户对此表示怀疑，在回应图表生成时说：*是的，难以置信*。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 条消息🔥): 

> `Meta-Prompting, Hassabis 的融资轮, Prompt Engineering 概念, OpenAI 的财务回报` 


- **探索 Meta-Prompting**：一位成员询问了关于 **Meta-Prompting** 的经验或有趣的用例，表明了扩展其 Prompt 技能的兴趣。
   - 另一位成员尝试用草草记录的想法进行回复，强调该小组正专注于探索这一概念。
- **对 Hassabis 融资轮的期待**：讨论转向支持 **Hassabis** 即将进行的融资轮，一位成员表达了对他对 AI 贡献的钦佩。
   - *为这次创业寻求美好的祈祷*，展示了社区的鼓励。
- **理解什么是好的 Prompt**：一位成员强调，创建一个有效的 Prompt 始于确切知道希望从模型中获得什么样的 **output**（输出）。
   - 这反映了小组内对 **prompt engineering** 基础方面的共同理解。
- **对 OpenAI 财务认可的担忧**：一位成员对即使在从事 AI 工作时也没有从 OpenAI 获得任何经济补偿表示不满。
   - 这引发了对该小组尽管有贡献且缺乏财务回报但仍具有封闭性质的反思。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 条消息🔥): 

> `Meta-Prompting, OpenAI 的 Prompting 方法, Hassabis 的投资轮, AI 社区参与, 创建有效的 Prompt` 


- **探索 Meta-Prompting 的用法**：成员们讨论了 **Meta-Prompting** 的概念及其潜在用例，其中一位成员表示有兴趣加深对有效 Prompting 的理解。
   - 另一位成员补充道，编写一个好的 Prompt 始于准确了解你希望从模型中获得什么样的输出。
- **OpenAI 的方法受到批评**：一位成员评论说 OpenAI 在实际的 Prompting 技术上可能已经落后，并建议通过更改 **system messages** 来优化性能。
   - 该成员还表达了对尽管参与了 AI 技术，但群体缺乏经济收益的沮丧。
- **Hassabis 的投资轮**：一位成员号召大家为 Hassabis 即将到来的 **investor round** 祈祷，并认可了他们令人印象深刻的工作。
   - 他们表达了一种希望成功获得融资的同志情谊。
- **AI 工作的经济收益**：讨论围绕成员是否从 AI 相关工作中获得金钱报酬展开，一种观点认为许多人确实从中受益。
   - 一位成员讽刺地评论说，他们在为社区做贡献的同时，却没有从 OpenAI 获得任何收入。


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326757945525927977)** (3 条消息): 

> `ICLR 参会, 聚会详情` 


- **ICLR 参会热度**：成员们正在确认参加 **ICLR**，其中一人用简单的“到时候见！”表达了兴奋之情。
   - 另一位成员 **philpax** 表示他们很快就会到达，并分享了外貌细节以便大家识别。
- **Philpax 为聚会做准备**：Philpax 提到他没有移动网络，将在外面等候，身穿 **浅褐色外套、黑色牛仔裤**，并携带 **健身包和背包**。
   - 此说明旨在帮助他人在现场识别他。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326754105724768357)** (19 messages🔥): 

> `rStar-Math 改进，O1 对比 GPT4o + MCTS，Qwen Chat 发布，中国 AI 访谈见解` 


- **rStar-Math 显示出重大进展**：Microsoft 的 [rStar-Math](https://x.com/_akhaliq/status/1877206745652592763?s=61) 将 **Qwen2.5-Math-7B** 的得分从 **58.8%** 提升至 **90.0%**，将 **Phi3-mini-3.8B** 从 **41.4%** 提升至 **86.4%**，表现优于现有模型。
   - 它在美国数学奥林匹克竞赛（USA Math Olympiad）中排名前 20% 的高中生之列，平均解决问题的比例为 **53.3%**。
- **关于 O1 与 GPT4o + MCTS 效率的辩论**：出现了一场关于 O1 是否比 **GPT4o + MCTS** 具有独特优势的讨论，成员们仔细审查了其性能和效率。
   - 有人担心 MCTS 资源消耗过大，而另一些观点认为 O1 可能只是现有策略的一种更高效的适配。
- **Qwen Chat 增强了与模型的交互**：[Qwen Chat](https://chat.qwenlm.ai) 宣布发布，允许用户在统一的 Web UI 中与各种 Qwen 模型进行交互，包括 **Qwen2.5-Plus** 和 **Qwen2-VL-Max**。
   - 它具有文档上传、模型对比功能，并承诺未来会增加网页搜索、图像生成和语音交互等增强功能。
- **来自访谈的中国 AI 行业见解**：在一次访谈中，**李开复**讨论了中国 AI 初创公司面临的挑战，提到了与美国同行相比在资金和技术方面的局限性。
   - 他强调了 **零一万物 (Zero One Infinity)** 的重心转移，现在更倾向于高效的中型模型，而不是继续训练大型模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TeamCodeLLM_AI/status/1877254042574844153">Wavecoder (@TeamCodeLLM_AI) 的推文</a>：🚀 介绍 EpiCoder：一个基于分层特征树的框架，用于多样化和复杂的代码生成。🔍 在基准测试中表现优异，可处理从简单函数到多文件项目的各种任务...</li><li><a href="https://x.com/JustinLin610/status/1877427101370036595">Junyang Lin (@JustinLin610) 的推文</a>：就在这里！Qwen Chat (https://chat.qwenlm.ai)，我们为 Qwen 模型打造的新 Web UI。链接是：chat dot qwen lm dot ai！您可以与最令人印象深刻的...</li><li><a href="https://x.com/_akhaliq/status/1877206745652592763?s=61">AK (@_akhaliq) 的推文</a>：Microsoft 发布 rStar-Math。小型 LLM 可以通过自我进化的深度思考掌握数学推理。在 MATH 基准测试中，它将 Qwen2.5-Math-7B 从 58.8% 提升到 90.0%，将 Phi3-mini-3.8B 从 41.4% 提升到...</li><li><a href="https://mp.weixin.qq.com/s/IUA482JlwI4CcRpiMRGHbA">晚点对话李开复丨零一万物部分团队并入阿里，“灵魂拷问来得太快了”</a>：“机会来临时，要勇敢做决策，机会消失时也是。”
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1326754434746814515)** (21 messages🔥): 

> `NuminaMath 数据集, 首席作者背景, 心理学与商学学位, 开源数据质量, 高中竞赛` 


- **对 NuminaMath 数据质量产生质疑**：虽然 **NuminaMath** 中 **89.7%** 的条目包含一个框选答案，但由于 **2.6%** 没有答案且 **7.7%** 有多个答案，引发了对潜在质量问题的担忧。
   - *此类问题凸显了开源和公开数据的现状*，并暗示了更深层次的质量隐忧。
- **首席作者令人惊讶的背景**：值得注意的是，该论文的第一作者是斯坦福大学的心理学博士生，鉴于该项目的技术性质，这引起了人们的关注。
   - 评论认为这是一种不寻常的跨界，因为这位深入参与数学数据分析的作者来自不同的学术领域。
- **对高中竞赛的认可**：一位深入分析了 **NuminaMath** 数据集 **cn_k12 子集** 的成员感叹道：*“我总结出我根本没机会赢过中国高中生”*。
   - 对于教育竞赛中同龄人设定的严苛标准，挫败感依然存在。
- **心理学项目并非儿戏**：讨论显示心理学项目的竞争极其激烈，一位成员强调了被此类项目录取的难度。
   - 鉴于首席作者的背景，对话转向了从心理学向商业相关领域转型的性质。
- **商业技能是下一个前沿**：一位成员强调，如果 Coding 问题被解决了，**Business** 就是下一个前沿，突出了技术熟练度之外的实用技能的重要性。
   - 这种转变说明了技术增强型工作空间中职业技能的演变格局。



**提到的链接**：<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>：我们提出了一个新颖的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模得出特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)……

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1326642132345421866)** (11 messages🔥): 

> `大规模模型的复杂性, Transformers vs MoEs` 


- **复杂性是大规模供应商的关键**：一位成员指出，虽然完善模型涉及大量 **Complexity**，但这无疑是值得的，特别是对于 **Large Scale Providers** 而言。
   - 这种观点呼应了在平衡模型性能与托管便利性方面所面临的挑战。
- **MoEs 优于 Dense Models**：另一位成员强调，在保持相同数量的 *Active* 参数时，**Mixture of Experts (MoEs)** 通常优于 **Dense Models**。
   - 他们认为 **更多的权重存储了更多的信息**，从而带来更好的峰值性能。
- **Transformers 鼓励重叠的方法**：一位成员警告不要阅读关于 **Transformers** 的内容，提到它们倾向于 **对专家进行 for 循环 (for loop over experts)**，这可能令人沮丧。
   - 然而，另一位成员认为这种 **for loop** 方法可能是一个更好的入门概念。
- **关于架构效能的辩论**：随后展开了关于 Transformers 是否能在 **架构上带来更好的模型** 的讨论，重点在于训练效率。
   - **“更好”** 模型的概念仍然是主观的，引发了关于性能差异的进一步讨论。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1326645897110491238)** (17 messages🔥): 

> `AI Alignment 讨论, 后训练模型塑造, AI 领域的冒充者综合征, 博客发布挑战` 


- **来自 Anthropic Salon 的 AI Alignment 见解**：在最近的一段 [YouTube 视频](https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964)中，研究人员在 Anthropic 活动期间讨论了 **alignment**，Josh Batson 提到了 Amanda Askell 在将基础模型塑造为有目标的 Agent 方面所起的作用。
   - 这引发了关于这种塑造是发生在 pretraining 还是 post-training 阶段的辩论，一位成员将这个过程比作从一块粘土中雕刻出一个角色。
- **关于模型训练过程的澄清**：成员们讨论了角色塑造工作可能在模型开发过程的早期就已经整合，而不仅仅是最后一步。
   - 一位成员表示：*“教一个 Agent 做到 well-aligned，似乎比直接从 pretraining 中产生的结果更容易，”* 这表明了对训练阶段看法的转变。
- **与冒充者综合征（Imposter Syndrome）的斗争**：一位成员分享了在技术领域感到冒充者综合征的经历，承认在职业生涯转型期间面临的挑战。
   - 他们幽默地将其描述为一种**被诅咒的超能力**，强调虽然它带来了压力，但也为他们的学习之旅增添了价值。
- **博客发布的挑战**：讨论延伸到了发布博客文章相关的困难，一位成员提到他们在平衡内容与 MLA 格式的正确引用方面感到吃力。
   - 他们表达了对哪些帖子会吸引读者的不确定性，突显了内容创作的试错本质。



**提到的链接**：<a href="https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964">How difficult is AI alignment? | Anthropic Research Salon</a>：在旧金山举行的 Anthropic Research Salon 活动上，我们的四位研究人员——Alex Tamkin、Jan Leike、Amanda Askell 和 Josh Batson——讨论了 alignment 科学...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1326957605561827338)** (3 messages): 

> `高效深度学习, 弹窗干扰, 博客导航问题` 


- **探索高效深度学习技术**：一篇分享的博客文章讨论了各种 **efficient deep learning** 技术，包括 **model pruning**、**quantization** 以及 **NVIDIA GPUs** 的演进。
   - 文章概述了重要的章节，如**现有的快速线性代数方法**和**用于更好收敛的归纳偏置（inductive biases）**。
- **弹窗干扰博客阅读**：*一位用户报告说有一个弹窗遮挡了博客首页的一部分*，给读者带来了**挫败感**。
   - 另一位用户幽默地评论了这种情况，谈到了 **googling** 的挑战。



**提到的链接**：<a href="https://alexzhang13.github.io/blog/2024/efficient-dl/">Alex L. Zhang | A Meticulous Guide to Advances in Deep Learning Efficiency over the Years</a>：一份非常详尽的长篇指南，介绍了深度学习算法、硬件、库、编译器等是如何变得更加高效的。

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1327017242222923907)** (14 messages🔥): 

> `AI 成本担忧, 开源 AI, 政策制定者的反应` 


- **关于 AI 成本的讨论**：有人对 **开源 AI** 成本仅为 **$5M** 表示担忧，这引发了政策制定者的警觉反应。
   - 对话中强调了一些评论，指出真实成本不仅仅包含 GPU 机时，并引用了一条[带有进一步讨论的推文](https://x.com/teortaxesTex/status/1877467302989295673/photo/1)。
- **AI 经济学中的误解**：一位成员指出，所展示的说明性数字并未计入**总资本支出（capex）、研发（R&D）费用**或与数据生成相关的成本。
   - 这引发了人们的担忧，即**普通读者**可能会忽视可能影响他们对该主题理解的重要细节。
- **闲聊氛围**：整体氛围积极，成员们对讨论的观点表达了热情和赞同。
   - 回复包括“Nice p1!”等支持性评论，表明了对对话的参与度。



**提到的链接**：<a href="https://x.com/teortaxesTex/status/1877467302989295673/photo/1,">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@natolambert 我同意实质内容，但你为什么把它表现得像是在辟谣？他们在那儿说了 GPU-hours*$/hr 不包括他们的总 capex、R&D 费用或数据生成（而且是我...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1326646270244294758)** (33 条消息🔥): 

> `GPT-NeoX vs Nvidia NeMo, SmolLM Corpus Upload, SciAgents Research Discussion, Modal for Model Training, DL Framework Usability vs Performance` 


- **GPT-NeoX 因性能受青睐**：讨论指出 **GPT-NeoX** 和 **Nvidia NeMo** 都拥有基于 Megatron 的 SFT 和 RLHF，但人们更倾向于选择 **GPT-NeoX**，因为其性能更稳健。
   - Open-instruct 被视为一个可靠但性能较低的选择，因为它基于 **TRL** 和 **HF trainer**。
- **SmolLM Corpus 发布更新**：一位用户分享称 **SmolLM Corpus** 的上传将推迟到明天，预计分片后的完整数据集大小为 **320GB**。
   - 他们指出新结构比之前的 **1TB 未压缩**版本更易用。
- **关于 SciAgents 研究的讨论**：成员们讨论了来自 **SciAgents** 研究的见解，赞赏其通过本体论方法揭示科学领域跨学科关系的方式。
   - 有成员指出它尚未达到 **GPT-4 级别的突破**，但承认其在更高级别学习编排（learning orchestration）中的应用潜力。
- **使用 Modal 进行训练的优势**：几位用户称赞 **Modal** 可用于训练超出本地 GPU 显存的模型，并强调每月 30 美元的免费额度是一个显著优势。
   - 他们提到，对于大型任务，它比传统的专用预留（dedicated reservations）更具成本效益，同时为研究人员提供了慷慨的支持。
- **深度学习框架对比**：对各种深度学习框架的对比揭示了从易用性到性能的光谱，突出了 **Megatron-LM** 和 **HF trainer** 等框架的优缺点。
   - 讨论了将 **GPT-NeoX** 用于非 Transformer 模型的挑战，强调了性能与灵活性之间的权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2409.05556">SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning</a>：人工智能领域的一个关键挑战是创建能够通过探索新领域、识别复杂模式和揭示...来自主推进科学理解的系统。</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - a Hugging Face Space by Vokturz</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Avelina/python-edu">Avelina/python-edu · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1326751257628508171)** (42 条消息🔥): 

> `Grokking phenomenon, Weight decay in LLMs, Softmax and scaling issues, Alternative loss functions, Attention mechanisms` 


- **Grokking 与 Softmax Collapse**：讨论了 **Grokking** 现象（即模型在过拟合后出现泛化），并将其归因于由于缺乏正则化导致的 **Softmax Collapse (SC)**。
   - 成员们建议，缓解 SC 可以在不需要强力干预的情况下实现 Grokking，同时也对“现实设置”下的结果表示了担忧。
- **Weight Decay 的主导地位**：大家达成共识，许多现代 LLMs 严重依赖 **0.1 Weight Decay** (WD)，以解决与朴素损失函数相关的优化能力问题。
   - 讨论强调了专门针对 Attention 机制改进 Weight Decay 的潜在策略，建议较低的 WD 可以避免诱导低秩（low rank）。
- **对 Softmax 函数的批判**：社区对 **Softmax** 的有效性提出了质疑，指出所有的 Softmax 函数可能都存在缺陷，并建议使用 Sigmoid loss 等替代方案。
   - 成员们注意到非 Softmax 函数在 Attention 中表现更好的案例，同时讨论了损失函数是否需要像 Attention 那样进行概率分离。
- **探索替代损失函数**：提到了在 LLM 训练中使用 **Sigmoid loss** 作为替代方案，特别是在 CLIP loss 的背景下，尽管个人的实验结果差异很大。
   - 一种建议的方法是引入辅助损失（如 `abs(norm(logits) - 1.0)`），以在不增加模型设计复杂度的情况下提高训练效率。
- **神经网络中的 Scaling**：围绕 **Unit scaling** 展开了讨论，许多人认为有必要通过它来对抗重缩放对称性（rescaling symmetries）导致的模型性能问题。
   - 对话强调，数值分离对于 Attention 机制至关重要，而对于语言损失（language loss），应优先考虑词选择的灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>：我们提出了一个全新的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模得出特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)……</li><li><a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>：我们展示了 rStar-Math，证明了小语言模型 (SLMs) 可以在不依赖优等模型蒸馏的情况下，媲美甚至超越 OpenAI o1 的数学推理能力。rStar-Math 实现了……</li><li><a href="https://arxiv.org/abs/2501.04697">Grokking at the Edge of Numerical Stability</a>：Grokking 是指在长时间过拟合后突然出现的泛化现象，是一个挑战我们对深度学习理解的惊人现象。虽然在理解……方面取得了显著进展。</li><li><a href="https://arxiv.org/abs/2411.04282">Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding</a>：大型语言模型 (LLMs) 展现了令人印象深刻的能力，但在处理需要多个步骤的复杂推理任务时仍然面临困难。虽然像 Chain-of-Thought (CoT) 这样的提示方法可以改进……</li><li><a href="https://x.com/rm_rafailov/status/1877446475271037314">来自 Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：我们关于“推理时计算 (inference time compute)”以及过去几个月研究工作的新立场论文发布了！我们提出了一些关于为什么它是必要的、它是如何工作的、以及为什么我们需要它的理论……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326840285199732768)** (6 messages): 

> `Llama 2 pretraining issues, Memory profiling for GPU usage, Model parallelism configurations, SLURM setups and outputs` 


- **Llama 2 预训练配置的挑战**：尽管使用了修改后的 **6.7B config**，但在尝试设置 **7B Llama2 风格模型**的预训练时，出现了挂起和 OOM 错误。
   - 怀疑是 **Model parallelism** 设置导致了问题，因为 **1.3B config** 在两个节点上运行完美。
- **请求内存使用分析（Memory Usage Profiles）**：一位用户请求在 MP = 1 和 MP = 2 配置下对 **1.3B 和 2.7B 模型**进行内存使用分析，以辅助调试。
   - 监控可能会揭示过高的 VRAM 使用率，即使它尚未导致 OOM 错误。
- **探索 6.7B 模型的内存性能**：有疑问提出，当使用 MP = 1 但 PP = 2 时，**6.7B 模型**是否会出现 OOM 错误。
   - 这种情况尚未测试，用户计划很快进行评估。
- **SLURM 输出指示潜在问题**：运行期间的最后日志条目显示，在没有崩溃的情况下，S3 checkpointing 依赖于 **boto3** 和 **hf_transfer**。
   - 在执行 **Llama 2** 配置期间的挂起问题显示，尽管启动了 wandb 运行，但没有任何进展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/boto/boto3">GitHub - boto/boto3: AWS SDK for Python</a>：适用于 Python 的 AWS SDK。可以通过在 GitHub 上创建账号为 boto/boto3 的开发做出贡献。</li><li><a href="https://github.com/huggingface/hf_transfer">GitHub - huggingface/hf_transfer</a>：为 huggingface/hf_transfer 的开发做出贡献。</li><li><a href="https://gist.github.com/aflah02/cbbcff84509ea3490604199c308ecf53">6-7B.yml</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/aa7bc6ef2bb4fda5d62fb102f399848b">local_setup_wandb_modified_with_slurm.yml</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/fa5a3f2bf6891e8d8b9cb14da2777bb8">pretrain_6_7B.sh</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/e1541111956d9721b125ffc1ff34cd93">out_file_slurm.txt</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/560436b0c0263b642724b69199898695">err_file_slurm.txt</a>：GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1326667086910455809)** (10 messages🔥): 

> `NCU profile comparison, Scam prevention advice, Learning Triton/CUDA, Options for simulated distributed training, Long context benchmarking` 


- **通过对比 NCU Profiles 获取见解**：一位用户建议，对比 **32x32 vs 16x16** 配置的 **NCU profile** 可以提供关于性能差异的见解。
   - 这种方法可以揭示需要优化的区域，或突出与规模相关的问题。
- **警惕诈骗者**：一位成员在被误导进行关于 **Bitcoin** 和诈骗的对话后，警告其他人不要向特定用户汇款。
   - 当欺诈活动的证据被提交后，社区迅速采取行动封禁了该用户。
- **学习 Triton/CUDA 大有裨益**：一位用户询问在 GPU 数量有限（如 **8xH100s**）的情况下是否有必要学习 **Triton/CUDA**。
   - 另一位用户回答说，通过这些技术了解 GPU 运行机制对于编写更好的性能优化代码是极其宝贵的。
- **模拟分布式训练**：一位成员询问在没有必要基础设施的情况下，模拟分布式训练的选项。
   - 针对 **JAX** 以及像 **Accelerate/Torch Lightning** 这样能简化分布式训练过程的框架提出了建议。
- **寻求长上下文基准测试（Long Context Benchmark）推荐**：一位用户正在致力于加速 **LLM inference** 的解码，并寻找具有大量输出生成的长上下文基准测试。
   - 他们指出目前的基准测试倾向于短提示（short prompts），并表示需要能反映现实长生成任务的基准测试。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326642293784055909)** (8 条消息🔥): 

> `WGMMA 计算需求，Fused MLP 的 Triton 实现，Profiling Triton 操作，Proton Profiler，Triton 示例中的 Torch.device` 


- **WGMMA 需要 4 个 Warp 计算**：会议指出 **WGMMA** 需要将计算拆分到 **4 个 warp** 上，且最小 **tile size** 为 **64**。
   - 澄清了关于每个 warp 至少需要 **16** 个大小的限制的困惑。
- **寻求 Triton Fused MLP 实现**：一位用户询问了 [tiny-cuda-nn GitHub 仓库](https://github.com/NVlabs/tiny-cuda-nn)中提到的 **fused MLP** 是否有现有的 **Triton** 实现。
   - 他们还质疑了为什么片上（on-chip）**MLP** 没有被广泛使用，思考是否因为它对于大多数应用来说规模太小。
- **Profiling Triton 操作的讨论**：提出了关于如何对 **Triton** 操作进行 **profiling** 的咨询，并将其与标准 **Torch** 和 **CUDA** 运行时使用的工具进行了对比。
   - 回复建议使用 **proton** 和 **NCU** 来对 **Triton** 进行 **profiling**。
- **Proton Profiler 介绍**：分享了一个名为“[Dev Tools: Proton/Interpreter](https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb)”的 YouTube 视频，讨论了编写 **Triton** kernel 的有用工具。
   - 视频详细解释了 **Proton** 工具，强调了它在调试中的实用性。
- **修复 Triton 示例中的 AttributeError**：报告了在运行 **Triton** 教程示例时的一个错误，特别是关于 `get_active_torch_device` 的 **AttributeError**。
   - 建议改用 `torch.device('cuda')`，这解决了该问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb">Dev Tools: Proton/Interpreter</a>：Keren 谈到了有助于编写 Triton kernel 的工具——特别是 Triton 解释器，它对于调试类似非法访问等问题非常有帮助...</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: 极速 C++/CUDA 神经网络框架</a>：极速 C++/CUDA 神经网络框架。可以通过创建 GitHub 账号为 NVlabs/tiny-cuda-nn 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1326808364381896767)** (14 条消息🔥): 

> `CUDA 驱动的重要性，Memory Banking 讲座，编写 CUDA Kernel，CUDA 中的 Blackwell vs. Hopper，CUDA 文件上传技巧` 


- **CUDA 驱动对 CUDA 功能至关重要**：一名成员强调 **NVIDIA 驱动** 是 **CUDA** 运行所必需的，并指出如果没有它，**CUDA** 无法在缺少 **NVIDIA GPU** 的系统上运行。
   - 另一位用户确认他们的系统由于没有 **GPU** 而返回错误，强调了正确驱动设置的必要性。
- **对 Memory Banking 讨论的好奇**：一名成员询问是否有讲座讨论过 **memory banking**，表现出对该主题进一步学习的兴趣。
   - 没有提供关于现有讲座的直接回答，这表明在共享知识方面可能存在空白。
- **支持 CUDA Kernel 编写**：一名初学者表示希望在编写一个简单的 **CUDA kernel** 来计算 **2D 矩阵** 的 **max**（最大值）和 **mean**（平均值）时获得帮助。
   - 另一位用户建议他们可以获得协助，并指出上传 `.cpp` 文件以便于查看的好处。
- **Blackwell 会增强 CUDA 编程模型吗？**：一名成员询问 **Blackwell** 是否会像 **Hopper** 一样对 **CUDA** 编程模型引入重大增强。
   - 他们质疑优化的 **Blackwell kernel** 是否会与 **Hopper** 中看到的增强功能保持一致，包括 **producer-consumer** 模型和异步 **tensor core** 指令。
- **分享 CUDA 文件以寻求帮助的技巧**：分享了一个关于上传带有 `.cpp` 扩展名的 **CUDA** 文件以在 Discord 频道中获得更好可见性的技巧。
   - 此外，还提醒成员有一个专门的频道可用于解答针对初学者的问题。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1326788513307561984)** (2 messages): 

> `Nectar Social 职位空缺，GPU 和 HPC 领域的欧洲咨询公司` 


- **Nectar Social 提供丰厚的推荐奖金**：初创 AI 公司 **Nectar Social** 正在招聘多个职位，包括位于**西雅图**的 **Sr/Staff Product Manager** 和 **LLM/AI Engineer**，推荐奖金高达 **$10,000**。
   - *联系获取详情*：该公司由大型基金支持，发展迅速，专注于**社交电商（social commerce）**。
- **欧洲咨询公司寻求硬件导向的开发人员**：一家总部位于**阿姆斯特丹**和**布达佩斯**的欧洲咨询公司正在招聘专注于 **GPU** 和 **HPC 软件**的开发人员，特别是熟悉 **CUDA, HIP, OpenCL 和 C++** 的人才。
   - 他们与 **AMD** 等客户紧密合作，参与 **rocPRIM** 和 **hipCUB** 等项目，职位详情可以在[这里](https://www.linkedin.com/jobs/view/4120980579/)找到。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1326810683538669590)** (3 messages): 

> `Ubuntu 上的 CUDA 安装，在没有 NVIDIA GPU 的 MacBook 上启动 AI 项目，使用云服务商获取 CUDA 支持` 


- **Ubuntu 上的 CUDA 安装变得简单**：对于希望在 Ubuntu 上安装 CUDA 的用户，[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) 提供了详尽的指南。
   - 它提供了关于 CUDA 作为旨在优化 GPU 性能的并行计算平台的关键细节。
- **MacBook 用户需要了解 CUDA 的局限性**：一位用户表达了对在没有 NVIDIA GPU 的 MacBook 上启动项目的担忧，强调了在 CUDA 相关任务中的挑战。
   - 作为回应，另一位成员建议使用云服务商或 **Google Colab**、**Lightning AI** 等平台来获取 CUDA 能力。



**提到的链接**：<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>：未找到描述

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

kashimoo: 我女朋友说我梦话都在说 CUDA 😭
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1326856718063243325)** (24 messages🔥): 

> `MI210 计算单元性能，Kernel 启动优化，GPU Occupancy 差异，工作组大小计算，RX7900XTX 性能见解` 


- **MI210 Occupancy 值引发疑问**：讨论强调了 **MI210** 每个计算单元（CU）的最大工作组（workgroups）数量似乎不是整数，重点关注了与其架构相关的每个 CU 2.5 个最大 Block 等预期值的差异。
   - *一位成员指出，如果在 Kernel 末尾添加 `__syncthreads()`，最大限制将变为精确的 2*。
- **Kernel 启动在 Occupancy 中获益**：有人指出 **CDNA1** 的 **Occupancy** 理论上是 10，但在单个 Kernel 启动期间实际达到 8 左右，而同时启动则可以发挥全部潜力。
   - 与 **MI100** 的对比表明，最大活跃 Warp（active warps）可能因 GPU 型号的架构优化而异。
- **计算结果的困惑**：成员们辩论了每个 **SIMD** 的最大 Block 和 Occupancy 的正确计算方法，揭示了基于工作组大小的预期值与记录值之间的差异。
   - 一份详细分析得出结论，不同架构（如 **CDNA2/3** 和 **RDNA2/3**）的 Occupancy 指标各不相同，从而凸显了这些模型的复杂性。
- **探索 RX7900XTX 性能指标**：贡献者讨论了 **RX7900XTX** 的性能指标，指出与 MI210 的 Occupancy 检查相比，它支持每个 **SIMD** 最多 16 个 Warp。
   - *后台计算的加入凸显了一些意想不到的结果，尤其是在跨 GPU 型号进行比较时。*
- **询问 RX5000 GPU 性能**：一位成员表示有兴趣了解 **RX5000** GPU 的性能指标，寻求关于其性能结果的贡献。
   - 这一询问反映了对跨各种 GPU 架构进行比较见解的持续追求。



**提到的链接**：<a href="https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/">通过大线程组优化 GPU Occupancy 和资源利用率</a>：Second Order Ltd 的联合创始人 Sebastian Aaltonen 讨论了如何优化使用大线程组的计算着色器（compute shaders）的 GPU Occupancy 和资源利用率。

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1326867199893307422)** (1 条消息): 

> `MicroDiT replication, DCAE autoencoder, MMDIT prompt adherence, Compute grants` 


- **MicroDiT 复现成功！**：**MicroDiT** 论文的复现工作已经完成，模型权重可在此处[下载](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt)，推理脚本已发布在 [GitHub](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb)。
   - *“我想我正在大展身手”*——作者对一位社区成员在推进该项目过程中提供的算力支持表示了感谢。
- **架构改进探索**：计划通过引入 **DCAE** 作为自动编码器（autoencoder）并使用 **MMDIT** 来增强架构，以提高提示词遵循度（prompt adherence）。
   - 这些改进的动力在于增强模型在执行任务时的整体效能。
- **寻求算力资助**：一名成员正在积极寻求 **compute grants**（算力资助）以加速实验，并提到其个人 GPU 性能不足。
   - 这种对资金的寻求凸显了研究人员在进行计算密集型项目时面临的挑战。



**提到的链接**：<a href="https://x.com/SwayStar123/status/1854884660981219399">sway (@SwayStar123) 的推文</a>：MicroDiT 复现完成。权重下载地址：https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt 推理脚本地址：https://github.com/SwayStar123/mic...

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326649425732436008)** (2 条消息): 

> `Alpha competition, Softmax kernel performance` 


- **Softmax Kernel Alpha 竞赛开幕**：一位成员宣布在他们的测试服务器上启动了首个运行中的 [alpha competition](https://link.to.alpha_competition)，邀请参与者竞争开发最快的 **softmax kernel**。
   - *“给我发私信，我会给你发送邀请！”*
- **竞赛热度持续高涨**：参与者对新的竞赛形式表现出极大的兴趣，强调了优化性能的机会。
   - *“Woo hoo!”* 是群组中表达热情的共同心声。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1326648370105684088)** (3 条消息): 

> `ThunderKittens GitHub repo, Collaboration on kernel development, CPP performance metrics` 


- **探索 ThunderKittens 进行性能测试**：参考 [ThunderKittens GitHub 仓库](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python) 来复现测试，并探索 tile primitives 以获得更好的性能指标。
   - *测试框架（harness）可以根据选择的维度（如序列长度和 batch size）输出 TFLOPS*。
- **征集 Kernel 项目协作**：一位成员表示有兴趣探索各种 **kernels**（如 MoE 和 Deep Seek Attention），并邀请他人共同协作。
   - 他们强调希望扩大对仓库的贡献，并鼓励感兴趣的人员与其联系。
- **关于问题解决的澄清**：一位成员询问了之前提到的问题是否已解决，表明了对进展的持续关注。
   - 这反映了小组内的积极参与，以确保所有事项都能得到及时处理。



**提到的链接**：<a href="https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python">HazyResearch/ThunderKittens 的 main 分支测试目录</a>：用于快速 kernel 的 Tile primitives。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1326819116644040704)** (7 条消息): 

> `ARC Prize 非营利组织转型、拒绝采样实验、文本领域探索、Meta CoT 论文见解、位置编码影响` 


- **ARC Prize 转型为非营利组织**：[ARC Prize](https://arcprize.org/blog/arc-prize-2025) 正在转型为一个成熟的非营利基金会，以加强迈向 AGI 的研究，并由其本人担任主席。
   - 该计划旨在由 **Greg Kamradt** 掌舵，利用他在 ARC Prize 2024 中积累的专业知识来指导研究进展。
- **准备拒绝采样实验**：一名成员正在建立一个简单的 **rejection sampling**（拒绝采样）基准实验，预计今晚运行。
   - 这项工作突显了在评估采样方法时采用的实践方法。
- **探索 ARC 的文本领域**：初步探索集中在 **text-domain**（文本领域），因为视觉编码器（vision-encoders）存在资源限制，文本领域是更可行的方法。
   - 一名成员表示愿意在以后合作将实验扩展到视觉输入。
- **Meta CoT 论文讨论经典局限性**：[Meta CoT 论文](https://arxiv.org/abs/2501.04682) 提出了关于为什么经典思维链（Chain of Thought, CoT）方法经常表现不佳的关键见解。
   - 作者强调了 CoT 的不足，并提出了增强推理能力的潜在改进方案。
- **自定义位置编码增强**：一名成员分享说，使用 **custom embeddings**（自定义嵌入）进行位置编码（positional encodings）比传统方法显著提升了模型性能。
   - 讨论中提到了利用更简单的模型以及 **TGI** 和 **Axolotl** 等工具来建立实验的基准设置。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>: 我们提出了一个全新的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模到达特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)....</li><li><a href="https://x.com/fchollet/status/1877069518171943000">François Chollet (@fchollet) 的推文</a>: ARC Prize 正在演变为一个成熟的非营利基金会，以进一步实现我们指导和加速迈向 AGI 研究进展的任务。特别感谢 @GregKamradt，他将领导...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1326779084004528190)** (47 条消息🔥): 

> `贡献 GPU 用于训练、DisTrO 开源、DeepSeek V3 差异、Hermes 模型审查、Cursor 与 IDE 对比` 


- **贡献 GPU 用于训练**：一名新成员正在了解一个项目，并询问是否可以贡献自己的 GPU 用于训练，但被告知请关注未来的更新。
   - *目前尚未开放贡献*，暗示未来可能会有变化。
- **DisTrO 已开源**：讨论透露 **DisTrO** 优化器已经开源，其代码可在共享仓库中获取。
   - 成员们注意到许多人已经在他们的项目中实现了它，引发了对协作和文档的兴趣。
- **DeepSeek V3 输出的差异**：一名成员质疑官方 **DeepSeek V3** API 与其他供应商之间输出质量的差异，指出响应中存在重复性。
   - 社区成员建议这可能是由于官方 API 的激进缓存策略造成的，但对第三方质量的看法不一。
- **了解 Hermes 模型的审查**：一名成员对 **Hermes** 模型的审查表示担忧，发现它并非完全无审查，且依赖于系统提示词（system prompt）指令。
   - 讨论强调，如果使用适当的系统提示词，模型可以按预期运行。
- **评估 Cursor 与 IDE**：一位参与者质疑 **Cursor** 工具是否值得从 **WebStorm** 或 **PyCharm** 等传统 IDE 切换过去，并对生产力表示怀疑。
   - 社区成员普遍认为，如果用户对当前工具感到满意，建议坚持使用，因为不同 AI 自动补全工具的生产力可能相似。

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326966515295326298)** (2 messages): 

> `减少模型内存占用, 开源 Function calling 模型, Function calling 准确率基准测试` 


- **不使用分块（chunking）减少内存占用的技巧**：一位成员询问了在 **RTX 4090** 上运行 **Qwen2.5-32B-Instruct-AWQ** 模型时减少内存占用的策略，该用户在处理约 6K Token 输入长度时报告了 OOM 错误。
   - 启用 **flash attention** 并没有显著降低 **VRAM 占用**，因此需要寻找更有效的方法。
- **关于最佳开源 Function calling 模型的咨询**：另一位成员寻求关于**最佳开源 Function calling 模型**的建议，并询问是否有追踪其 **Function calling 准确率**的基准测试。
   - 他们特别想知道模型在 **post training pipeline**（后训练流水线）中是如何提升 Function calling 能力的。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 messages): 

> `研究想法与论文, Carson Poole 的项目` 


- **研究进展查询**：一位用户询问了研究进展，得到的回复是引导其前往 Carson Poole 的个人网站获取信息。
   - *Carson 提到他网站上约有一半的想法已经转化为论文，并鼓励用户去查看。*
- **Carson Poole 的研究网站**：Carson Poole 分享了他的个人网站 [poole.ai](https://poole.ai) 以及其他项目如 [Forefront.ai](https://forefront.ai) 和 [Simple AI Software](https://simpleaisoftware.com) 的链接。
   - 他还邀请用户通过电子邮件联系他以获取更多关于研究想法的信息，并在其网站上列出了几篇可以作为灵感来源的论文。
- **值得探索的研究想法**：Carson 提供了一系列研究想法，包括 [ReLoRA](https://arxiv.org/abs/2307.05695)、[Sparse Upcycling](https://arxiv.org/abs/2212.05055) 和 [GQA](https://arxiv.org/abs/2305.13245) 等作品，并附带了原始出处链接。
   - 这些论文中的每一篇都指出曾在 Discord 上讨论过，并提供了具体的日期和链接以供参考。



**提到的链接**: <a href="https://poole.ai">Carson Poole 的个人网站</a>: 未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326790180220436532)** (11 messages🔥): 

> `Qwen 7B 性能, 模型中的自我反思, 数学推理能力, LLM 在数学中的实用性, LLM 的可靠性` 


- **微软的 Qwen 7B 取得了令人印象深刻的数学成就**：微软展示了 Qwen 7B 在 AIME 上的表现达到了 o1 的水平，通过其 MCTS 驱动过程展示了增强的数学能力，该过程允许模型像推理模型一样进行自我反思。
   - 这一进展引起了极大关注，预计在即将播出的播客中将进一步讨论相关发现。
- **关于数学能力与推理能力的辩论**：一位成员认为数学能力并不一定反映推理能力，并指出 LLM 在这两者之间表现出的关联有限。
   - 对话揭示了一种观点，即虽然数学技能很重要，但 LLM 在推理任务中的应用仍存疑问。
- **对 LLM 在数学中可靠性的怀疑**：人们对 LLM 解决数学问题的可靠性表示担忧，一位成员建议它们在进行更复杂的估算时无法获得足够的信任。
   - 尽管一些人承认 LLM 在某些任务（如估算）中很有用，但对于它们在更苛刻场景中的应用仍保持警惕。



**提到的链接**: <a href="https://x.com/altryne/status/1877220144725758414?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 呃，伙计们……微软刚刚让 Qwen 7B 在 AIME 上的表现达到了 o1 的水平 😵‍💫 他们还展示了通过其 MCTS 驱动过程，模型具备了像推理模型一样的自我反思能力。将……

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 条消息): 

> `Progress on Ideas, Research Ideas List, Carson Poole's Contributions` 


- **进展查询**：一位用户询问了关于研究或项目进展的更新，引发了 Carson Poole 的回复。
   - Carson 提到可以在他的个人网站上找到见解，并暗示许多想法已经转化为了论文。
- **Carson Poole 的研究想法**：Carson 分享了一份可以参考或“借鉴”的研究想法列表，其中几个想法链接到了相关的学术论文，例如 [ReLoRA](https://arxiv.org/abs/2307.05695)。
   - 他强调，有些想法最早出现在 2022 年 11 月的讨论中，体现了持续进行的协作研究。



**提到的链接**：<a href="https://poole.ai">Carson Poole's Personal Site</a>：未找到描述

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1326679514205392918)** (47 条消息🔥): 

> `Salesforce hiring freeze, OpenAI custom instructions update, Anthropic funding round and valuation, Google AI product merger, Moondream model release` 


- **Salesforce 冻结软件工程师招聘**：Marc Benioff 透露，**Salesforce** 在 **2025** 年将不再招聘软件工程师，理由是其产品 **Agentforce** 带来的 AI 使生产力提升了 **30%**。
   - 在播客采访中，尽管招聘冻结，他仍对公司的增长表示乐观，声称五年后公司可能会变得“更大”。
- **OpenAI 更新导致 custom instructions 失效**：据报道，OpenAI 的一次更新在将新功能集成到 ChatGPT 语音系统时，导致了 **custom instructions** 失效。
   - 一位社区成员提到，就在更新实施时，他们正在录制一段关于提高语音质量的视频。
- **Anthropic 获得 20 亿美元新融资**：Anthropic 正在筹集 **20 亿美元**，使其估值达到惊人的 **600 亿美元**，是去年数据的三倍。
   - 据报道，其年度经常性收入（ARR）达到了约 **8.75 亿美元**，主要来自商业销售，显示出显著增长。
- **Google AI 产品合并至 DeepMind 旗下**：一位成员对多个 AI 工作室合并入 **Google DeepMind** 表示兴奋，期待 **2025** 年在开源模型和开发者工具方面的进展。
   - 存在关于 Google 内部潜在重组和结构变化的猜测，并讨论了其现有模型产品中的冗余问题。
- **Moondream 更新及脚本可用性**：分享了关于视觉语言模型 **Moondream 2b** 的更新，有人询问用于运行该新模型的可用脚本。
   - 社区继续讨论了关于该模型能力和整体更新过程的细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/topmass/status/1877444315871326422?s=46">来自 topmass (@topmass) 的推文</a>：就在我录制视频展示如何让 ChatGPT Advanced Voice 变得更好时，OpenAI 正在发布更新，这破坏了自定义指令，但似乎也增加了新功能……</li><li><a href="https://www.interconnects.ai/p/the-state-of-post-training-2025">2025 年后训练 (Post-training) 的现状</a>：立即观看 (54 分钟) | 我在 NeurIPS 关于语言建模教程的重新录制版（并增加了一些内容）。</li><li><a href="https://x.com/natolambert/status/1877020436246204596?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：我重新录制了我们 NeurIPS 语言模型教程的后训练部分，增加了一些幻灯片，并在 @interconnectsai 上写了一份简短的现状报告。请欣赏！链接在引用推文中。00:00 介绍...</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-beni">Marc Benioff 表示，Salesforce 在 2025 年将不再招聘软件工程师</a>：Salesforce 首席执行官 Marc Benioff 宣布不再招聘新的软件工程师 —— 看看 AI 如何塑造公司的未来。</li><li><a href="https://x.com/andrewcurran_/status/1876705929296581078?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Anthropic 正在筹集另外 20 亿美元。这一轮融资将使 Anthropic 的估值达到 600 亿美元，是去年的三倍多。据华尔街日报报道，其 ARR 最近达到了约 8.75 亿美元...</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/">Marc Benioff 表示，Salesforce 在 2025 年将不再招聘软件工程师</a>：Salesforce 首席执行官 Marc Benioff 宣布不再招聘新的软件工程师 —— 看看 AI 如何塑造公司的未来。</li><li><a href="https://x.com/osanseviero/status/1877452798683430988">来自 Omar Sanseviero (@osanseviero) 的推文</a>：我非常激动地分享，我们（AI Studio, Gemma, Gemini API）正在加入 Google DeepMind！😱 2025 年对于开源模型、可访问的研究以及面向开发者的出色工具来说，将是非常激动人心的一年...</li><li><a href="https://x.com/tsarnick/status/1877089046528217269">来自 Tsarathustra (@tsarnick) 的推文</a>：François Chollet 表示，OpenAI 的 o1 模型在可能的思维链 (Chain of Thought) 空间中运行搜索过程，生成自然语言程序并适应新颖性，这是一种“真正的突破”...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxjzol/new_moondream_2b_vision_language_model_release">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=6yIMb0K-aS4">当今语言模型后训练 (Post-training) 是如何完成的</a>：与 2024 年初相比，我对 2025 年初后训练的开放配方 (Open recipes) 和知识现状要乐观得多。去年我最初的...</li><li><a href="https://youtu.be/vOVDyRIS09k?si=KxqRdqzBdgL5zxVL">我在一天内重制了 Apple Genmoji</a>：以下是我如何在不到一天的时间内重制 Apple 新的 AI 生成表情符号功能（即 Genmoji）。在这里亲自体验 Open Genmoji：https://github.com/Eva...</li><li><a href="https://github.com/EvanZhouDev/open-genmoji">GitHub - EvanZhouDev/open-genmoji: 属于我们大家的生成式表情符号。</a>：属于我们大家的生成式表情符号。通过在 GitHub 上创建账户，为 EvanZhouDev/open-genmoji 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1326704638870683648)** (1 条消息): 

> `AI Agent Hackathon, OpenRouter API credits, Live Agent Studio competition, Prize pool increase, Registration details` 


- **Hackathon 提供 OpenRouter 额度和现金奖励**：[ottomator.ai 的 AI Agent Hackathon](https://studio.ottomator.ai/hackathon/register) 的参与者可以领取 **$10** 的 OpenRouter API 额度，表现优异者的总奖金池为 **$6,000**。
   - 注册现已开放，截止日期为 1 月 22 日，获胜者将于 2 月 1 日公布。
- **Live Agent Studio 竞赛详情**：**Live Agent Studio Hackathon** 将于 1 月 8 日至 1 月 22 日晚上 7:00 (CST) 举行，社区投票将于 1 月 26 日开始。
   - 获胜者名单将于 2 月 1 日通过直播公布，鼓励参与者使用自选工具构建与该 studio 兼容的 Agent。
- **n8n Agent 奖金池增加**：**n8n 团队**增加了奖金池，目前总计 **$6,000**，其中为最佳 n8n Agent 设立了 **$700** 和 **$300** 的专项奖。
   - 这两个奖项将由 n8n 团队进行评审，为参与者使用其平台提供了额外动力。
- **重要的 Hackathon 指南**：提醒参与者仔细阅读 Hackathon 的协议和构建 AI Agent 的综合指南。
   - 这些资源将在参与前提供关键信息和说明，确保每个人都做好充分准备。



**提及的链接**：<a href="https://studio.ottomator.ai/hackathon/register">oTTomator</a>：未找到描述

  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1326712396831785010)** (46 条消息🔥): 

> `OpenRouter Performance Issues, O1 API Response Format, Gemini Flash Performance, Hanami Usage, Crypto Payments` 


- **OpenRouter 的 UI 性能受到批评**：成员们讨论了 **OpenRouter** 的基础设施虽然很棒，但其 UI 性能不足，特别是在处理超过 **1k 行**的长聊天记录时。
   - 用户指出，当聊天记录超过此限制时，滚动和打字几乎变得不可能，并请求改进活动过滤和分页功能。
- **O1 API 中的奇怪格式**：多位用户报告 **O1 API** 的响应使用 **=====** 而不是反引号进行格式化，导致对其行为的困惑和不满。
   - 一位用户推测这可能会节省 Token，而其他人则质疑这一变化的合理性。
- **Gemini Flash 能力**：一位成员分享了 **Gemini Flash 1.5** 的**性能指标**，记录了 63,364 个输入和 7,018 个输出，成本为 **$0.000171**，且达到了令人印象深刻的 **255.6 tps**。
   - 尽管为了更好的用户体验提出了一些性能建议，但他们对 Gemini 表现出了极大的热情。
- **Hanami 框架咨询**：一位用户询问是否有人在使用 **Hanami** 框架，而另一位用户指出在测试期间遇到了意外字符。
   - 这引发了成员之间关于其可靠性和可用性的简短讨论。
- **加密货币支付被视为解放**：一位用户幽默地评论说，在使用**加密货币**支付后超越了政府的限制，引发了其他人的祝贺。
   - 这种轻松的交流突显了社区对新兴支付方式的参与度。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1326655467577147412)** (1 条消息): 

> `CSV Downloads, Table Responses` 


- **CSV 下载功能上线**：用户现在可以通过选择下载选项，直接从响应中将**表格下载为 CSV 文件**，这一功能增强了易用性。
   - 这一新增功能通过随附的 [图片](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg?ex=6781892f&is=678037af&hm=f69ea0b4635a0df0dfe206fdd64762dd6fd44a96818c6347e1f1aad37404e0fe&) 进行了展示。
- **通过表格功能增强用户交互**：新的 CSV 下载选项为用户提取数据提供了一种无缝方式，增强了与表格响应的交互。
   - 通过简化数据检索，该功能旨在提升整体用户体验。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1326642921490874369)** (33 条消息🔥): 

> `Youzu.ai 设计工具，Perplexity 用户问题，协作项目提案，Ecosia 合作伙伴咨询，Perplexity 优化技巧` 


- **Youzu.ai 提供室内设计辅助**：[Youzu.ai](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688) 是一款 AI 驱动的工具，通过根据用户所在地建议购买地点来帮助用户创建精美的房间设计，显著减轻了过程中的压力。
   - 一位成员称赞了它的有效性，表示它将购物的折磨变成了一种愉快的体验。
- **Perplexity 用户面临技术故障**：几位用户报告了 Perplexity 的问题，包括文本输入延迟、持续出现的上传请求窗口，以及无法在 Spaces 中重新加载修改后的文件。
   - 一位用户甚至询问了平台内输入和输出的最大 context size。
- **呼吁在有意义的项目上进行协作**：一位成员表达了希望利用群组中多样化的人才来创建具有影响力的项目的愿望，并邀请大家贡献力量，无论投入时间多少。
   - 他们强调了协作的重要性，旨在共同打造一些令人难忘的东西。
- **Ecosia 寻求与 Perplexity 建立合作伙伴关系**：来自 Ecosia 的一位产品经理联系并寻求建立潜在的合作伙伴关系，并提到在寻找合适的联系渠道方面存在困难。
   - 他们希望得到社区的帮助，以促进这次协作。
- **关于优化 Perplexity 使用的讨论**：用户讨论了针对专业研究优化 Perplexity 的策略，包括模型选择和 system prompts 的最佳实践。
   - 一位用户向资深成员寻求建议，希望为自己的个人资料创建一个强大的、预配置的 system prompt。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688">Youzu.ai: 当 AI 室内设计遇上现实世界购物</a>：介绍全球首个由 AI 驱动的 Design-to-Buy 平台✨</li><li><a href="https://x.com/omidaziz/status/1877409601202631083?s=46">omid (@omidaziz) 的推文</a>：设计得最好和最差的 AI 应用
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1326740486169427988)** (6 条消息): 

> `丰田探索火箭技术，即将发布的视频游戏，IndyCar 车手平均水平，西班牙人平均寿命，NVIDIA 家用超级计算机` 


- **丰田的火箭雄心**：据[这篇文章](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q)所述，丰田据报道正在探索火箭技术的创新。这家汽车巨头正涉足新的航空航天领域。
- **即将发布的视频游戏预览**：围绕[即将发布的视频游戏](https://www.perplexity.ai/search/prochaines-sorties-de-jeux-vid-zgsehswCSLuZemsB7i3UYA)展开了讨论。玩家们正热切期待着接下来的新动态。
- **IndyCar 车手表现洞察**：对 [IndyCar 车手平均水平](https://www.perplexity.ai/search/indycar-driver-averages-mOBWLru4TWqQJrczuSDMtQ)的分析揭示了整个赛季的表现指标。粉丝们正深入研究数据以衡量车手的成功。
- **西班牙寿命统计数据**：[这份分析](https://www.perplexity.ai/search/average-lifespan-of-a-spaniard-OOT0EWBjS6ifrw142dFOwg#0)讨论了西班牙人的平均寿命。健康趋势揭示了关于长寿的重要见解。
- **NVIDIA 的家用超级计算机产品**：根据[这份公告](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw)，NVIDIA 售价 **3000 美元**的新型超级计算机现已可供家庭使用。这项创新将重新定义家庭计算。
   - AI 技术与个人计算的融合正变得更加触手可及。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1326717449072279654)** (3 条消息): 

> `韩语 API 使用，Llama-3.1 的替代模型，Discord 讨论链接` 


- **韩语 API 请求**：@razer8967 表达了对仅提供**韩语**响应的 API 的需求。
   - *I want API answer only Korean lang* 详细说明了用户对特定语言的需求。
- **替代模型讨论**：@razer8967 澄清他们正在寻找 Llama-3.1-sonar-small, large, huge 之外的模型以满足其需求。
   - 这暗示了用户正在持续寻找与其语言偏好兼容的合适模型。
- **Discord 对话链接**：一名成员分享了与该主题相关的 [Discord 链接](https://discord.com/channels/1047197230748151888/1047202784090538054/1316804335258173460)，但缺乏额外的上下文信息。
   - 尽管提供了链接，但未能从分享的内容中提取出具体的见解。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1326950230167392407)** (2 条消息): 

> `North AI 工作区，Cohere 发布活动，生产力工具` 


- **Cohere 发布 North AI 工作区**：Cohere 宣布开启 ***North*** 的早期访问，这是一个全方位、**安全的 AI 工作区**平台，它将 LLM、搜索和 Agent 集成到一个直观的界面中，以提高生产力。
   - 该平台旨在超越 **Microsoft Copilot** 和 **Google Vertex AI Agent Builder**，为追求运营效率的用户提供无缝体验。
- **North 承诺实现巅峰生产力**：North 结合了多种功能，通过让 AI 轻松集成到日常工作流程中，帮助用户实现**巅峰生产力**。
   - 此次发布令社区感到兴奋，突显了其在提高工作场所**效率**方面的潜力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/cohere/status/1877335657908949189">来自 cohere (@cohere) 的推文</a>: 今天，我们开启了 North 的早期访问！我们的全方位安全 AI 工作区平台将 LLM、搜索和 Agent 结合到一个直观的界面中，毫不费力地将 AI 集成到您的日常...</li><li><a href="https://cohere.com/blog/north-eap">North 简介：一个助您完成更多工作的安全 AI 工作区</a>: North 将 LLM、搜索和自动化结合到一个安全的 AI 工作区中。它超越了 Microsoft Copilot 和 Google Vertex AI Agent Builder，无缝提升了员工的生产力和运营...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1326828245412220955)** (7 条消息): 

> `用于生成模型的 Command R+，Embedding 升级指南，分类模型错误处理，Alignment Evals 黑客松` 


- **Command R+ 成为大型模型的关键**：对于大型生成模型，推荐使用 **Command R+**，详细的模型信息可在 [模型概览](https://docs.cohere.com/docs/models) 中找到。一位用户询问了如何有效利用该功能的潜在工作流。
- **从 embed-v2 升级到 v3 的指南**：由于担心重新生成 Embedding 的任务量巨大，一位用户寻求从 **embed-v2** 升级到 **v3 Embedding** 的最佳实践建议。他们指出了 embed-v2 未来可能被弃用的可能性。
- **大型分类模型中的错误处理**：由于请求中存在 **2,500 个示例的限制**，而用户的数据集包含 **95,429 个标记示例**，因此在训练分类模型时遇到了错误。他们寻求关于如何有效处理大型数据集进行 Fine-tuning 的指导。
- **即将举行的 Alignment Evals 黑客松**：一位用户宣布他们将于 25 日举办 **Alignment Evals 黑客松**，其中包括发布评估和解释性教程。其他成员鼓励向社区分享黑客松详情，以吸引更多人参与。

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326682963806519296)** (26 messages🔥): 

> `Cohere LLM API 递归输出问题, 使用 Cohere 生成长报告, 处理模型输出中的 token 限制, API 速率限制错误, 设置生成上下文的自动模式` 


- **Cohere LLM API 陷入递归输出**: 一位用户报告在使用 Python ClientV2 时遇到了 Cohere LLM API 的**递归循环**问题，导致了过多的 token 消耗。
   - 有建议提出实现一个 **max_tokens** 参数来限制响应流中的事件数量。
- **生成长报告的挑战**: 一位用户询问如何扩展 **cmd-r+** 的输出长度，理由是 **4k tokens** 不足以生成完整的章节或包含推理过程。
   - 建议使用滚动聊天历史（rolling chat history）来扩展输出生成，以规避 **4k 限制**。
- **API 速率限制问题**: 一位用户在查询 API 时遇到了 **TooManyRequestsError**，表明他们超过了允许的限制。
   - 社区成员建议联系 **support**（支持团队）以获取有关 API 问题的协助。
- **关于设置自动模式的澄清**: 关于如何在 API 上实现管理聊天历史的 **auto mode** 存在困惑，用户正在寻求更清晰的指导。
   - 一位成员解释了复制聊天历史的过程，但进一步寻求关于是否存在自动模式的帮助。
- **API 中的最大消息大小错误**: 一位用户在尝试对超过 4MB 的 datasets 端点进行 GET 请求时遇到了**消息大小错误**。
   - 这突显了 API 在接收请求的最大消息大小方面可能存在的限制。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1326786815415291904)** (2 messages): 

> `Discord 频道规则` 


- **发布规则提醒**: 一位用户被提醒阅读频道规则，并且只能在一个频道中发布消息。
   - 该提醒强调了遵守指南对于维持 Discord 社区秩序的重要性。
- **用户确认提醒**: 另一位用户表示道歉，并表示稍后会处理该提醒。
   - 这一回应表明了在认可社区请求的同时表现出合作态度。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1326763968773095506)** (18 messages🔥): 

> `PR #8505 的赏金, LLVM JIT 和 Autogen 讨论, LLVM API 的稳定性, 对 Tinygrad 的贡献` 


- **为 PR #8505 提供赏金**: 一位成员被告知，重新测试 [PR #8505](https://github.com/tinygrad/tinygrad/pull/8505) 的**赏金已就绪**，支付方式可选 PayPal 或 USDC。
   - *George 提到该赏金专门用于 OSX 上的 MOCKGPU AMD 测试。*
- **LLVM JIT 和 Autogen 合并计划**: 一位成员分享说 **PR #8486** 已准备好接受审查，旨在将 LLVM JIT 与 LLVM autogen 结合以简化开发。
   - 他们指出，讨论**处理多版本方法**的评论可以在 `support/llvm.py` 中查看。
- **对 LLVM API 稳定性的担忧**: 一位成员对 LLVM API 中**函数签名的静默变更**表示担忧，且未找到先前的案例。
   - George 安慰说这种静默变更不太可能发生，该成员也确认目前在 LLVM 14-19 版本中运行正常，没有问题。
- **鼓励对 Tinygrad 的新贡献**: 一位用户表达了对贡献 Tinygrad 的兴奋，并询问如何参与的指导。
   - 对贡献的兴趣表明社区非常欢迎并渴望协作努力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llvm.org/docs/DeveloperPolicy.html">LLVM 开发者政策 &#8212; LLVM 20.0.0git 文档</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/8505">patrini32 在 OSX 上的 MOCKGPU amd 测试 · Pull Request #8505 · tinygrad/tinygrad</a>: 依赖于 #8501
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1326927564308086785)** (4 条消息): 

> `TinyGrad 博客概览，带设备规范的层初始化，TinyGrad 中的设备选项` 


- **分享 TinyGrad 博客文章**：一名成员分享了他们的 [博客文章](https://adelaloui.me/tinygrad-codebase-explained-ish/)，该文章概述了对 TinyGrad 代码库的探索，反映了随着深入研究该项目所获得的见解。
   - 他们强调这是一个**高层级概览**，并警告不要修改核心 **tinygrad/** 目录之外未经测试的代码。
- **层初始化的设备规范**：有人提出了关于在初始化类似 **nn.Linear** 的层时，如何为权重和偏置指定设备的问题。
   - 一名成员建议在创建任何 Tensor 之前使用 `Device.DEFAULT` 来设置所需的设备，选项包括 **METAL**、**CUDA** 和 **CLANG**。
- **设备选项说明**：当被问及设备选项时，回复列出了在 TinyGrad 中初始化 Tensor 的几种规范，并强调 **CLANG** 将使用 CPU。
   - 这一系列选项允许用户在 TinyGrad 框架内工作时根据硬件偏好进行定制。



**提及的链接**：<a href="https://adelaloui.me/tinygrad-codebase-explained-ish/">TinyGrad Codebase Explained-ish</a>：对 TinyGrad 仓库结构和关键文件的详细（大概）解释

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1326643603430183044)** (22 条消息🔥): 

> `GPT4All 的 Nvidia 性能，使用 phi-4 模型，本地服务器 API 问题，模型的模板设置，角色扮演模型推荐` 


- **GPT4All 的 Nvidia 性能提升**：有人指出 **llama.cpp Vulkan** 与 GPT4All 使用的版本之间存在**显著的性能差异**，特别是在具有 **CUDA** 优势的 **Nvidia** 上。
   - 这突显了取决于硬件配置和模型实现的不同性能水平。
- **phi-4 模型取得成功**：一名成员分享了他们在 GPT4All 中使用 **phi-4-Q4_0** 模型成功运行的 **javascript 测试**，并指出它在其模板下运行良好。
   - 根据 Hugging Face 上的 **Microsoft 发布**，该模型已获得 **MIT** 官方许可。
- **本地服务器 API 引发疑问**：讨论了本地服务器 API 仅与 **OpenAI** 兼容的问题，引发了在使用本地语言模型时出现缺失 **openai_api_key** 错误的担忧。
   - 成员们质疑为什么似乎不支持本地托管，并澄清了当前配置的基础限制。
- **模型的模板设置困惑**：一名初学者表示在 GPT4All 上为 **Vicuna 模型** 设置 **Chat Template** 时遇到困难，无论输入什么都得到标准回复。
   - 建议检查 GitHub 资源以获取模板配置，并强调旧模型可能缺乏特定的聊天模板。
- **角色扮演模型推荐**：对于 **COTE 动漫** 中的角色扮演，建议包括使用 **Nous Hermes 2 模型**，该模型以其与 RP 内容的兼容性而著称。
   - 社区成员还鼓励在 Reddit 等平台上探索资源，以获取针对角色扮演场景的进一步推荐。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf">phi-4-Q4_0.gguf · GPT4All-Community/phi-4-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3365.">nomic-ai/gpt4all</a>：GPT4All：在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K">aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1326662062595641425)** (2 条消息): 

> `GitHub HQ Event, Agentic Document Workflows, AI Agents Debugging, Fast Inference Systems, LlamaIndex Workflows` 


- **1 月 15 日 GitHub 总部活动**：参加 **1 月 15 日**在 GitHub 总部举办的专家演讲，内容涵盖使用 @arizeai 调试 **AI agents**、使用 @GroqInc 创建**快速推理 Agent 系统**，以及使用 LlamaIndex 构建 **agentic workflows**。更多详情请见[公告推文](https://twitter.com/llama_index/status/1877103276635848846)。
   - 此次活动将为开发者和技术爱好者提供实战见解和多种学习机会。
- **2025 年向 Agentic Document Workflows 转型**：一种名为 **Agentic Document Workflows (ADW)** 的新范式将在 2025 年通过直接集成到**业务流程**中来简化文档处理。在[此处博客文章](https://twitter.com/llama_index/status/1877420085691953385)中探索 ADW 的核心原则。
   - ADW 强调处理多种格式的文档，旨在提高运营效率。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326725320505561091)** (18 条消息🔥): 

> `Ollama performance, Access control for applications, Vector database indexing, Local TEI server for reranking, QueryFusionRetriever token limit` 


- **Ollama 更新提升性能**：一位用户报告称，在更新 **Ollama** 后，评估时间缩短至不到 **3 秒**。
   - *处理速度的提升已引起关注*，引发了对进一步性能评估的兴趣。
- **通过电子邮件限制控制应用访问**：有人提出了关于部署应用以确保仅具有特定电子邮件地址的用户可以访问的问题，并建议使用 **Cloud Run + Google IAP** 作为解决方案。
   - 这为非技术用户管理应用可访问性提供了一个简单的解决方案。
- **VectorStoreIndex 集成中的挑战**：一位用户探索了在使用带有 JSON 索引的 PostgreSQL 数据库时，根据元数据键在 **VectorStoreIndex** 中过滤节点的可行性。
   - 讨论表明需要手动索引，或在 **LlamaIndex** 中进一步支持自动索引。
- **本地 TEI 服务器的 reranking 功能**：一位用户分享了关于利用**本地 TEI 服务器**进行 reranking 的 API 参考，但对实现方式表示不确定。
   - 鼓励社区更新现有讨论以反映相关功能，特别是关于 **issues 和 pull requests** 的内容。
- **QueryFusionRetriever 遇到 token 限制**：一位用户报告了在将 **TEI Reranker** 与 **QueryFusionRetriever** 结合使用时遇到的问题，因超过 token 限制而出现“Input validation error”。
   - 他们提供了一个显示设置检索器配置的代码片段，重点突出了 **25 top K** 参数。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/">Tei rerank - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py">llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py at main · run-llama/llama_index</a>：LlamaIndex 是一个用于 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/9572">[Feature Request]: Text Embeddings Inference Reranker · Issue #9572 · run-llama/llama_index</a>：功能请求：你好，我们能否为 Text Embeddings Inference 服务器提供一个类似于 SentenceTransformerRerank 或 CohereRerank 的 reranking 类？原因：我们遇到了性能/扩展性问题...
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326677862723616880)** (18 条消息🔥): 

> `Rust 的 Actor 模型语法, Mojo 中的重载解析 (Overload resolution), Mojo 中的量子计算库, MAX 与量子计算, 用于量子操作的 Quojo 库` 


- **Rust 的语法简化了 Actor 实现**：一位成员赞赏 Rust 的语法使得像 **GlommioMultipaxosWorker** 这样的 Actor 的多行可重用实现变得更加简单，减少了冗长的类型边界带来的干扰。
   - 他们担心在大型代码库中，由于频繁的变动，重载解析顺序（overload resolution order）可能会变得棘手。
- **对 Mojo 量子计算库的兴趣**：一位成员询问 Mojo 中是否存在正在开发的量子计算库，希望能找到一个类似 “Qiskit” 的实现以获得实践经验。
   - 另一位成员建议利用 **MAX**，并解释了它支持各种硬件配置的能力，以及未来在量子编程方面的潜力。
- **简述 MLIR**：一位成员分享了一个 YouTube 视频链接，解释了 **MLIR** 的概念，以阐明其在优化量子操作中的作用。
   - 他们说明了 MAX 如何在运行时分析硬件使用情况，以进行计算优化（例如消除单位矩阵乘法）。
- **Quojo 库介绍**：一位用户指向了 GitHub 上的 **Quojo** 库，将其描述为一个用 Mojo 编写的量子计算引擎。
   - 社区反应积极，认可了量子计算领域新兴年轻开发者的成长速度。
- **从 Qiskit 开始进行量子模拟**：针对疑问，一位成员指出 **Qiskit** 应该作为起点，并澄清库可以在不需要直接访问 IBM API 的情况下模拟量子操作。
   - 成员们讨论了像 Quojo 这样库的潜力，同时也承认了其中涉及的学习曲线。



**提到的链接**：<a href="https://github.com/Deftioon/Quojo">GitHub - Deftioon/Quojo: A Quantum Computing Machine written in Mojo</a>：一个用 Mojo 编写的量子计算引擎。可以通过创建账号为 Deftioon/Quojo 的开发做出贡献。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1326806666129313843)** (1 条消息): 

> `黑客松结果, 评审时间线更新` 


- **黑客松结果发布推迟**：黑客松网站已更新以反映时间线的变化，表明虽然大多数结果已统计完毕，但在评委进一步反馈后，最终结果预计将在 **1月** 公布。
   - 参赛者已被告知评委对**提交的作品印象深刻**，一旦所有结果确认，将发布正式公告。
- **评委仍在评审作品**：大多数最终结果已统计，但组织者仍在等待几位评委的回复，然后才能最终确定。
   - 这一延迟已传达给参赛者，并强调了评委对提交项目的积极反响。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1326777282391703635)** (6 条消息): 

> `Google Form 编辑, Twitter 账号注销, 表单提交过程, 电子邮件访问问题` 


- **Google Form 不允许编辑**：一位用户报告说 **Google Form** 不允许他们编辑之前的提交，并寻求帮助。
   - *回复建议重新提交表单将覆盖之前的条目*。
- **使用不同的电子邮件访问表单**：一位成员提到，尝试使用不同的电子邮件访问表单可能会奏效，并建议在字段中输入正确的电子邮件。
   - *另一位用户注意到表单已关闭，但询问他们注销的 Twitter 账号是否会影响获得证书的资格*。
- **Twitter 账号状态不会取消资格**：在确认表单已关闭后，该用户对因 **Twitter 账号注销** 而被取消资格表示担忧。
   - *一位成员安慰他们，注销账号不会导致失去获得证书的资格。*


  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326953902591049849)** (7 messages): 

> `OI 1.0 Python Execution, AI Improvement Observations, Model and Parameters Inquiry, Custom Instructions Insights` 


- **关于 OI 1.0 Python 执行的担忧**：一名成员询问 `--tools interpreter` 是否旨在启用在 OI 1.0 中运行 Python 代码，并指出它仍然尝试调用 python.exe。
   - 另一名成员指向了系统消息中的一行特定内容，该内容暗示 OI 1.0 无法再直接运行代码，从而引发了困惑。
- **AI 改进尝试**：一位成员分享了对 AI 性能的见解，指出在某些命令下性能有所提高，并且 AI 能够打印文件的头部（head）而不是全部内容。
   - 他们提到使用了 **gpt-4o-mini** 模型，并强调了他们观察到 AI 表现吃力的领域。
- **关于模型和参数的查询**：一位用户请求有关正在使用的模型和参数的详细信息，寻求对其功能及任何必要更改的简要说明。
   - 这一查询反映了用户对如何优化与 AI 交互的广泛兴趣。
- **自定义指令反思**：讨论包括分享自定义指令集，展示了旨在改进 AI 在代码执行方面交互的各种指南。
   - 这些指令强调了对命令的谨慎处理，并鼓励在使用前确认工具功能。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1326744255678644325)** (5 messages): 

> `TruLie dataset, Image-to-3D techniques, Chirpy3D, World Models, Gaussian splats` 


- **关于 TruLie 数据集的查询**：一名成员询问了 **TruLie dataset**，向社区寻求有关其当前相关性的信息。
- **Image-to-3D 技术的探索**：讨论了 **image-to-3D** 技术的最新进展，特别是寻求可以在笔记本电脑上运行的开源选项。
   - 成员们对利用单张图像输入的技术表现出兴趣，并推荐了 **Gaussian splat** 和 **NeRF libraries**。
- ****Chirpy3D** 引起关注**：一名成员分享了 **Chirpy3D** 的链接，这是一个专注于用于创意 3D 鸟类生成的连续部分潜变量（continuous part latents）的项目，强调了其在 3D 建模领域的潜力。
   - 该项目与来自 **University of Surrey** 和 **Imperial College London** 等知名机构的多位研究人员的贡献相关联。
- **World Models 的兴起**：另一个感兴趣的话题是 **World Models**，它集成了物理感知网络以实现更真实的视频生成。
   - 虽然不严格属于 image-to-3D，但有人指出这种方法与正在进行的关于 3D 生成技术的讨论密切相关。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/dylanebert/3d-arena">3D Arena - a Hugging Face Space by dylanebert</a>：未找到描述</li><li><a href="https://kamwoh.github.io/chirpy3d/">Chirpy3D</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

rom1504: 是否有任何用于构建 Agent 的优质开放工具注册表？
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326831683705507903)** (4 条消息): 

> `改进聊天机器人的 COT，为 LLM 构建自定义评估，评估在 AI 开发中的重要性，Drew Breunig 的工作与项目` 


- **新手寻求改进聊天机器人的 COT**：一位成员询问了除了设置 Signature 之外，还有哪些改进聊天机器人 **Chain of Thought (COT)** 的方法。
   - *除了为其设置 Signature 之外，还有什么改进 COT 的方法吗？*
- **为 LLM 构建你自己的评估 (Eval)**：讨论中重点介绍了 Drew Breunig 的一篇文章，强调了在选择模型时构建自定义评估的重要性，并强调 **Evals 是必不可少的**。
   - Breunig 指出，*你拥有的最有价值的 AI 资产是你的评估 (Eval)，而不是你的模型或提示词 (Prompts)*。
- **Drew Breunig 的 AI 贡献**：Drew Breunig 介绍了自己，解释了他在 **PlaceIQ** 的背景以及目前在 **Precisely** 和 **Overture Maps Foundation** 的工作。
   - 他提倡开发诸如 **StepList**（一款管理日常事务的应用）和 **Reporter**（一款衡量自我理解的应用）之类的工具。
- **社区对评估的参与**：一位社区成员对 Breunig 关于评估的文章表示了极大的热情，并表示会去查看相关内容。
   - *太棒了！我现在就去看看。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.dbreunig.com/">首页</a>：关于技术、文化、媒体、数据及其交互方式的写作。</li><li><a href="https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html">你的评估比模型更重要</a>：一个构建良好的自定义评估可以让你快速测试最新模型，在开发提示词和流水线时更快地迭代，并确保你始终朝着产品的特定目标前进。</li>
</ul>

</div>
  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326675915857461260)** (3 条消息): 

> `使用 Jamba 的 Python 应用，AI 代码生成，PHP 编程，Jamba 功能` 


- **播客制作人利用 Jamba 进行剧集回顾**：一位成员分享说，他们使用 **Jamba 的 Conversational RAG** 创建了一个基础的 **Python 应用**，用于查询播客转录文本以更好地回顾剧集内容。
   - 尽管该应用仍是一个 **正在进行中的工作 (work in progress)**，但实验过程 *非常有趣*。
- **AI 代码生成的有趣怪癖**：另一位成员指出，他们在利用 **AI** 生成代码的初步体验中，在调试 **HTML, Javascript** 和 **PHP** 代码时遇到了一些有趣的错误。
   - 他们评论说，当前的 AI 技术热潮似乎仅仅是 **触及了可能性的皮毛**。
- **PHP 仍是 Web 项目的主力**：一位成员分享了他们持续在 Web 开发和本地 IRC 机器人编码中使用 **PHP** 的经历，强调了其可靠性。
   - 他们在连接到 **Jamba** 后表示满意，因为它通过使用与其他 API 类似的 **对话数组 (conversation arrays)** 简化了编程的某些方面。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 条消息): 

jovial_lynx_74856: 这里有人尝试过微调 ModernBERT 吗？
  

---


### **Torchtune ▷ #[jobs](https://discord.com/channels/1216353675241590815/1326789182932123708/1326789612580110369)** (1 条消息): 

> `Nectar Social 招聘，推荐奖金，AI 初创公司职位` 


- **Nectar Social 正在搜寻人才**：一家处于早期阶段的 AI 初创公司 **Nectar Social** 正在迅速扩张，并寻求填补 **西雅图** 及其他地区的关键职位，包括 **高级/资深产品经理 (Sr/Staff Product Manager)**、**LLM/AI 工程师**等。
   - 他们为成功入职的推荐提供高达 **$10,000** 的推荐奖金，目前处于 *半隐身模式 (semi-stealth mode)* 运营。
- **职位多样且地点灵活**：空缺职位还包括提供 **灵活** 地点选择的 **客户成功经理 (Customer Success Manager)**，以及位于 **纽约/洛杉矶** 的 **创始客户经理 (Founding Account Executives)**。
   - 申请人若拥有 *之前的初创公司经验* 将被优先考虑，以增加成功机会。


  

---


---


---


---


---


{% else %}


> 完整的各频道详细分析已针对电子邮件进行删减。
> 
> 如果你想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}