---
companies:
- xai
- google
- anthropic
- openai
- cognition
- ai21-labs
- nvidia
- langchain
date: '2024-09-04T01:35:37.812399Z'
description: '以下是为您翻译的中文内容：


  **xAI** 宣布推出 **Colossus 10万卡 H100 集群**，该集群能够在 4 天内训练出一个 FP8 精度的 GPT-4 级别模型。**Google**
  为 **Gemini** 引入了**结构化输出**功能。**Anthropic** 讨论了 **Claude** 的性能问题，认为这可能与 API 提示词的修改有关。**OpenAI**
  增强了其 Assistants API 中文件搜索（File Search）的控制功能。**Cognition** 和 **Anthropic** 的负责人现身播客节目。走红的
  **快手-Kolors (Kwai-Kolors)** 虚拟试穿模型以及开源实时语音对话模型 **Mini-Omni**（类似于 **gpt-4o-voice**）正式发布。


  此外，文中还重点介绍了关于使用 LoRA 和 QLoRA 进行参数高效微调、长文本嵌入挑战以及 Claude 的 LaTeX 渲染功能的教程。**AI21 Labs**
  发布了 **Jamba 1.5** 模型，具备 256K 上下文窗口，并提升了长文本处理速度。**NVIDIA** 的 **Mistral-Nemo-Minitron-8B**
  在开源大模型排行榜（Open LLM Leaderboard）上首次亮相。**LangChain** 推出了用于工作区组织的资源标签，**svpino** 分享了一个低代码
  AI 应用工具包。法律 AI 智能体以及使用 LangSmith 进行的金融智能体评估也备受关注。'
id: a00a923a-5c1d-4a39-af65-baa5017358b8
models:
- gpt-4o-voice
- gemini
- claude
- jamba-1.5
- mistral-nemo-minitron-8b
original_slug: ainews-everybody-shipped-small-things-this
people:
- dario-amodei
- scott-wu
- fchollet
- svpino
title: 这个假期周末，大家都在发布一些小东西。
topics:
- fine-tuning
- long-context
- parameter-efficient-fine-tuning
- latex-rendering
- real-time-audio
- virtual-try-on
- resource-tags
- low-code
- ai-agents
- workspace-organization
- model-benchmarking
---

<!-- buttondown-editor-mode: plaintext -->**smol updates 是你唯一需要的。**

> 2024年9月2日至9月3日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**214** 个频道和 **2424** 条消息）。预计节省阅读时间（以 200wpm 计算）：**281 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

让我们来看看：

- 来自 xAI：[Colossus 100k H100 集群已上线](https://x.com/elonmusk/status/1830650370336473253)。[根据 Semianalysis 的说法](https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true)，该集群可以在 4 天内训练出一个 FP8 GPT-4 级别（2e25 FLOPs）的模型。
- 来自 Google：[Gemini 获得了 Structured Output](https://x.com/OfficialLoganK/status/1829678117054792160)
- 来自 Anthropic：[Dario 参加了一个播客](https://youtu.be/7xij6SoCClI?feature=shared)
  - 许多人指出 [Claude 变得越来越差](https://x.com/nearcyan/status/1829674215492161569?s=46)，这或许是因为 [API 中 prompt 的修改](https://x.com/_philschmid/status/1830559304241287611?s=46)。目前尚无官方回应。
- 来自 OpenAI：[增强了 Assistants API 中 File Search 的控制功能](https://x.com/openaidevs/status/1829259020437475771?s=46)
- 来自 Cognition：[Scott Wu 参加了一个播客](https://share.snipd.com/episode/faaed93f-9297-4926-aa03-78643ea68d65)
- [Kwai-Kolors 虚拟试穿模型走红](https://x.com/basedjensen/status/1829763446763896903?s=46)
- [Mini-Omni](https://x.com/osanseviero/status/1830875530209513587?s=46) 发布，这是一个开源的实时语音对话模型。类似于 GPT4o Voice。

既然今天是平静的一天，你可以思考一下来自你友好的邻居 AI Engineering 播客关于[智能商品化（commoditization of intelligence）的更广泛趋势](https://x.com/latentspacepod/status/1831020483967701260)。

---

{% if medium == 'web' %}

**目录**

[TOC]

{% else %}

**目录**和**频道摘要**已移至此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

---

**AI 生产力提升与微调**

- **参数高效微调**：[@fchollet](https://twitter.com/fchollet/status/1826674137089409377) 分享了一个关于使用 LoRA 和 QLoRA 对 LLM 进行参数高效微调的教程，重点介绍了如何通过简单的脚本启用 QLoRA。**"gemma_lm.quantize('int8')"**
- **长上下文 Embedding 挑战**：[@JinaAI_](https://twitter.com/JinaAI_/status/1826649449919369726) 讨论了 RAG 系统中朴素分块嵌入流水线（chunking-embedding pipelines）的**“丢失上下文问题” (Lost Context Problem)**，并引入了 “Late Chunking” 方法。
- **Claude 增强功能**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1826667671364272301) 宣布在 Claude 的功能预览中添加了 **LaTeX 渲染**，以改善数学方程式的显示效果。

**高性能模型发布**

- **Jamba 1.5 模型**：[@AI21Labs](https://twitter.com/AI21Labs/status/1826607933167469024) 发布了 Jamba 1.5 Mini 和 Large，具有 **256K 上下文窗口**、**2.5 倍更快**的长上下文性能以及 **JSON 输出**等工具。[@Yampeleg](https://twitter.com/Yampeleg/status/1826642273669544363) 指出：**“这是第一个能够与顶级模型竞争的 Mamba 混合模型”**。
- **Mistral-NeMo-Minitron-8B**：[@NVIDIA](https://twitter.com/clefourrier/status/1826672319970115887) 作为首个登上 Open LLM Leaderboard 的 Nvidia 模型亮相，在各项基准测试中表现显著优于其他模型。

**增强型协作工具与框架**

- **LangSmith 工作空间组织**：[@LangChainAI](https://twitter.com/LangChainAI/status/1826643491130421476) 引入了**资源标签 (resource tags)**，以高效管理项目、数据集和 Prompt。**“使用资源标签在 LangSmith 中组织你的工作空间。”**
- **AI 应用低代码工具包**：[@svpino](https://twitter.com/svpino/status/1826590311948452035) 提供了一个开源、自托管的 AI 入门套件，包括用于**工作流自动化的 n8n**、用于**本地模型托管的 Ollama** 以及用于**向量存储的 Qdrant**。**“引导一个全功能的低代码开发环境来构建 AI 应用程序。”**

**AI 在法律与金融领域的应用**

- **AI 法律 Agent**：[@SpellbookLegal](https://twitter.com/scottastevenson/status/1826611628852609551) 推出了 Spellbook Associate，这是一个 AI Agent，可以**将法律项目分解为计划**、执行任务并审查工作。**“律师的电动自行车。”**
- **LangSmith 评估**：[@virattt](https://twitter.com/virattt/status/1826621769371021564) 为一个沃伦·巴菲特金融 Agent 添加了评估功能，使用 LangSmith 高效地设置和可视化评估。

**性能优化与现实应用**

- **Phi-3.5 Vision**：[@Microsoft](https://twitter.com/mervenoyann/status/1826640879995813925) 推出了 Phi-3.5 视觉模型，超越了现有基准。**“4.2B 模型，128k Token 上下文长度”**
- **Neuralink 游戏**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1826619574651171148) 分享了 Neuralink 测试的进展，参与者可以用大脑控制游戏元素，暗示了在游戏和其他领域的近期应用前景。**“思想将是唯一的限制。”**

**迷因/幽默**

- [@swyx](https://twitter.com/swyx/status/1826673468223688956): "转发 [@latentspacepod](https://twitter.com/latentspacepod): 微调 GPT-4o 值得吗？"
- [@rez0__](https://twitter.com/rez0__/status/1826671312330523118): "好吧，我放弃了。我现在相信了。这就像是‘我妻子的丑闻教会了我关于 B2B 销售的道理’那种 LinkedIn 恶搞帖，但它是真的。"
- [@goodside](https://twitter.com/goodside/status/1826651729443827805): "那是个旅游的好地方，但你不会想住在那里。"

---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. Star Command R 32B v1：TheDrummer 发布的新作**

- **[Drummer 的 Coo- ... *咳咳* Star Command R 32B v1！来自 Theia 和 Rocinante 的创作者！](https://huggingface.co/TheDrummer/Star-Command-R-32B-v1)** ([得分: 47, 评论: 14](https://reddit.com//r/LocalLLaMA/comments/1f71b1j/drummers_coo_ahem_star_command_r_32b_v1_from_the/)): **Star Command R 32B v1** 已发布，这是一款由 **TheDrummer**（**Theia** 和 **Rocinante** 的开发者）创建的新 AI 模型。该模型被描述为拥有 **320 亿参数** 的 AI，定位为该领域其他大语言模型的竞争对手，尽管发布公告中未提供具体的性能指标或对比。
  - 用户们调侃 **TheDrummer** 这次相对温和的模型命名，有人将其比作“*色情明星转型主流，或者摔跤手步入政坛*”。开发者以一个幽默的 GIF 进行了回应。
  - 该模型的 **GGUF 版本** 已在 [Hugging Face](https://huggingface.co/TheDrummer/Star-Command-R-32B-v1-GGUF) 上线。一些用户对未来可能的模型表示期待，包括假想中的 **104B Command-R-Sutra**。
  - 讨论涉及了该模型生成显式内容的潜力，用户根据 **TheDrummer** 以往创建此类功能模型的声誉，对其能力进行了推测。

**主题 2. 使用 Ollama 的社区驱动免费 AI 服务器**

- **我制作了自己的本地 AI，你可以免费使用，** ([得分: 37, 评论: 52](https://reddit.com//r/LocalLLaMA/comments/1f711c3/i_made_my_own_local_ai_u_can_use_it_for_free/)): 该用户使用 **Ollama** 创建了一个**本地 AI 服务器**，其特点是包含用于获取当前信息的 **Llama 3.1**、用于无限制 AI 体验的 **Llama 3 (dolphin)** 以及用于图像识别的 **LLava**。该服务器在 [evaai.ngrok.app](http://evaai.ngrok.app/) 免费向公众开放，创作者正在寻求关于微调、提高可访问性以及通过捐赠维持服务器运行方面的帮助。
  - 创作者表示有兴趣为服务器添加**工具**，如**图像生成**，可能会使用 **Stable Diffusion**。用户可以在 **open-webui** 的 **Workspace** 面板中找到工具和函数。
  - 有人建议加入 **The Horde**，这是一个为没有 GPU 的用户提供 LLM/SD 使用的**众包计算网络**。创作者表现出兴趣，但也表达了对资源管理和限制的担忧。
  - 关于**隐私**，该服务器不验证电子邮件，允许使用虚假邮箱注册，并提供删除聊天记录和用户数据的选项。系统运行在 **3070 GPU** 上，速度达到 **75 tokens/秒**。

**主题 3. 比较用于 OCR 和复杂布局理解的小型视觉 LLM**

- **用于 OCR 的最佳小型视觉 LLM？** ([得分: 31, 评论: 17](https://reddit.com//r/LocalLLaMA/comments/1f71k60/best_small_vision_llm_for_ocr/)): 该帖子讨论了**小型视觉语言模型 (LLM)** 在**光学字符识别 (OCR)** 方面的表现，特别是针对简历和发票等**复杂文档结构**。作者发现 **InternVL 1.5** 非常有效且速度相对较快，而 **Phi Vision** 功能更强大但速度较慢，并提到在简单情况下使用 **PaddleOCR**。他们还指出 **Florence-2** 擅长目标检测和图像描述，并提供了一个 [开源 VLM 排行榜链接](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard) 作为参考。
  - 对于纯 OCR 任务，推荐使用 **Surya OCR**，用户报告其在手写文本识别方面优于 **PaddleOCR**。[Surya GitHub 仓库](https://github.com/VikParuchuri/surya/tree/master) 可供部署使用。
  - **Qwen2-vl**（尤其是 7B 模型）的 OCR 能力受到称赞，在某些测试中甚至优于 **internvl2-8b** 等更大的模型。用户指出，虽然 OCR 模型提取文本速度更快，但 **VLM** 可以更有效地提取结构化数据。
  - 微软的 **Kosmos-1.5** 因其 OCR 能力和以 **Markdown 格式**输出的能力而受到关注。然而，一些用户更倾向于使用 **Marker**（由 **VikPachuri** 开发的另一个开源工具）来进行 Markdown 输出和整体 OCR 性能处理。

## AI Reddit 内容汇总

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型开发与基础设施**

- **xAI 的 Colossus 训练集群**：xAI 已上线名为 Colossus 的 **100,000 H100 GPU 训练集群**，并计划在[未来几个月内翻倍至 200,000 GPU](https://x.com/elonmusk/status/1830650370336473253?s=46)。

- **OpenAI 的自研芯片开发**：OpenAI 正在[与 TSMC 合作开发其首款自研芯片](https://wccftech.com/openai-developing-custom-chip-on-tsmc-a16-angstrom-process/)，采用 A16 Angstrom 工艺，专门用于 Sora 视频应用。

- **Google DeepMind 的多模态学习**：[Google DeepMind 的一篇论文](https://arxiv.org/html/2406.17711v1)展示了如何通过联合样本选择进行数据策展（data curation），从而加速多模态学习。

- **Microsoft 的 MInference**：[Microsoft 的 MInference 技术](https://arxiv.org/abs/2407.02490)可在保持准确性的同时，实现长上下文任务中高达数百万 token 的推理，大幅提升了支持模型的运行速度。

**AI 模型发布与改进**

- **Salesforce 的 xLAM-1b**：Salesforce 发布了 xLAM-1b，这是一个拥有 10 亿参数的模型，在[函数调用（function calling）方面实现了 70% 的准确率，超越了 GPT 3.5](https://www.reddit.com/r/LocalLLaMA/comments/1dz8g10/salesforce_tiny_giant_xlam1b_model_surpasses_gpt/)。

- **Phi-3 Mini 更新**：Rubra AI 发布了更新后的 Phi-3 Mini 模型，[具备函数调用（function calling）能力](https://www.reddit.com/r/LocalLLaMA/comments/1dzhe38/phi3_mini_june_with_function_calling/)，可与 Mistral-7b v3 竞争，并优于基础版 Phi-3 Mini。

**AI 研究与应用**

- **合成数据生成**：一篇关于[扩展合成数据生成](https://www.reddit.com/r/MachineLearning/comments/1dzergu/r_scaling_synthetic_data_creation_with_personas/)的论文利用 LLM 内部的多样化视角，通过从网络数据中策展出的 10 亿个角色（personas）来生成数据。

- **Anthropic 的 AI 群体智能**：Anthropic 的 CEO 报告称，[大模型现在正在衍生出更小的模型](https://v.redd.it/oju63dvc0emd1)来完成任务并汇报结果，形成了一种减少人类干预需求的群体智能（swarm intelligence）。

**AI 行业与社区讨论**

- **OpenAI 订阅价值**：OpenAI 的应用研究负责人[承认对他们的订阅服务感到失望](https://www.reddit.com/r/singularity/comments/1f7o9zg/head_of_applied_research_at_openai_im_sorry_we/)，并承诺将进行改进以提升其价值。

- **Stable Diffusion 子版块管理**：[Stable Diffusion 子版块正面临管理问题](https://www.reddit.com/r/StableDiffusion/comments/1f7194f/we_need_to_talk_about_a_new_mod_who_has_a_history/)，用户对新版主的行为以及社区规则的变更表示担忧。

**迷因与幽默**

- 一篇标题为[“然后这就发生了”](https://i.redd.it/ob5wg9rxvemd1.jpeg)的帖子在 r/singularity 引起了广泛关注。

---

# AI Discord 摘要回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要

**1\. LLM 进展与基准测试**

- **Mistral-Nemo 价格大洗牌**：[Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) 的价格下降了 **23%**，这可能预示着 LLM 供应商竞争格局的变化。
  - 这一显著的价格变动可能表明市场动态正在演变，分析师们正密切观察竞争对手将如何应对 Mistral 激进的价格策略。
- **GPT-4o 表现优于 Turbo 版本**：**GPT-4o** 目前比 GPT-4 Turbo 便宜 **50%**（**每百万输入 token 5 美元，每百万输出 token 15 美元**），同时拥有 **2 倍的速度**和高达每分钟 1000 万 token 的 **5 倍速率限制**。
  - 凭借 **128k 上下文窗口**和增强的 **vision（视觉）能力**，GPT-4o 为寻求语言模型效率和高级功能的用户提供了强有力的选择。

**2\. 优化 LLM 推理与训练**

- **Apple Silicon 的内存带宽难题**：虽然 **Apple Silicon** 拥有令人印象深刻的内存带宽，但与 GPU 相比，其在 CPU 推理方面的效用有限，**M1 Max** 宣称的 **400GB/s** 带宽在实际效果上引发了质疑。
  - 讨论表明，尽管理论带宽很高，但 Apple Silicon 上 LLM 推理的实际性能可能差异巨大，这促使人们进一步研究如何针对 AI 工作负载优化这些架构。
- **Triton 加载顺序影响性能**：**Triton** 用户发现改变加载（load）顺序会导致显著的性能差异，在一个案例中，性能从 **1.89506** 提升到了 **2.440731**。
  - 这一观察引发了关于编译器处理加载停顿（load stalls）和指令调度方式的疑问，暗示了 LLM 训练和推理流水线中潜在的优化空间。
- **Activation Checkpointing 的成功实践**：一名成员成功用极少的代码实现了 **activation checkpointing**，并展示了在使用 **124M BF16** 时基于 batch size 的不同内存需求。
  - 该实现显示，在不复用的情况下内存占用为 **1211 MiB**，而在 100% 重新计算层时仅为 **176 MiB**，突显了 LLM 训练中巨大的内存优化潜力。

**3\. 开源 AI 框架与社区努力**

- **Mini-Omni 语音模型开源**：能够同时生成文本和音频的 [Mini-Omni](https://hf.co/gpt-omni/mini-omni) 开源模型已发布，可用于实时语音对话，其 [代码库](https://github.com/gpt-omni/mini-omni) 和研究论文详细介绍了流式音频输出功能。
  - 此次在 Twitter 上的发布引发了关于该模型潜在应用及其对未来 AI 交互影响的讨论，展示了社区对多模态 AI 开源进展的热情。
- **Toolio 0.5.0 增强 LLM 控制**：**Toolio 0.5.0**（被称为“文本的胜利”）为这款专为 **Apple Silicon** 设计的 Python 工具包引入了改进的文档和更好的 prompt 构建功能，包括符合 [JSON schema](https://json-schema.org/) 的结构化 LLM 响应生成。
  - 此次更新旨在为开发者提供对文本生成的精细控制，将 Toolio 定位为那些不仅需要常规文本生成、尤其是需要 tool-calling 功能的开发者的关键工具。
- **Mojo 标准库开放贡献**：**Mojo Standard Library**（标准库）现已部分开放贡献，尽管某些部分仍与编译器紧密绑定。目前已提供稳定版本，但强大的稳定性保证仍在建立中。
  - 社区成员对贡献机会表示兴奋，同时也指出由于该库的全部潜力和生产就绪性仍在实现中，需要保持谨慎。

**4\. AI 硬件与基础设施**

- **10 万张 H100 集群分析引发辩论**：一项对 **100,000 H100 集群** 的全面考察讨论了能效、网络拓扑以及 Ethernet 和 InfiniBand 方案之间的权衡，强调了这些集群如何反映出 **GPT-4** 之后 AI 进展感官上的放缓。
  - 该分析引发了对集群可靠性和故障恢复的担忧，表明尽管维持了与前几代相似的计算指标，但在有效扩展当前模型方面仍面临挑战。
- **H200 与 H100 价格动态**：**H200** GPU 目前 8 连装版本的价格为 **18 万美元**，而据报道 **H100** 价格**大幅上涨**，这可能与 **Tesla** 在市场上的活动有关。
  - 这些价格趋势引发了关于大型科技公司的高需求对 AI 硬件生态系统影响的讨论，社区正密切关注持续的需求将如何改变未来的定价和供应策略。

---

# 第 1 部分：Discord 高层摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 微调引发讨论**：用户报告了在微调 **Gemma 2B** 模型时遇到的障碍，特别是在调整训练参数后生成了随机输出。
  - 讨论强调了需要一致的微调模板来优化 token 使用，并警告不要更改模板。
- **Numpy vs. Cupy：Gemma 2 实现**：一名成员使用 [Numpy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) 从零开始成功实现了 **Gemma 2**，随后过渡到了 [Cupy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb)。
  - **Cupy** 版本需要具有 **24GB** 显存的 GPU 才能进行有效计算，同时还提供了一个适用于低显存 GPU 的 [f16 版本](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb)。
- **llama.cpp 的 RPC 内存难题**：成员们分享了关于 **llama.cpp** 与 RPC 服务器集成的挫折，其中一人表示它无法在服务器机器上保留内存。
  - 这种挫败感体现了实现复杂的 AI 模型和基础设施要求所带来的挑战。
- **关于文本转语音（Text-to-Speech）微调的咨询**：一位用户寻求使用 Unsloth 微调文本转语音模型的帮助，但得到的澄清是该工具缺乏此功能。
  - 对话中提到了 Whisper 训练指南，该指南需要更大的数据集才能进行有效训练。
- **API 订阅成本受到关注**：由于未能充分利用完整的 **$20** token 配额，对成本的担忧促使了关于从订阅服务转向仅使用 **API** 的讨论。
  - 这一趋势反映了用户在更好地管理 AI 相关费用和访问权限方面的广泛举措。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Phi-3.5-mini 在浏览器中表现出色**：**Phi-3.5-mini (3.8B)** 模型使用 WebGPU 在浏览器中以约 90 tokens/second 的速度运行，确保了完全本地化处理以增强**隐私**。查看演示和[源代码](https://x.com/xenovacom/status/1826992922509595068)。
  - *用户报告称，与基于服务器的模型相比，本地处理输入时的延迟显著降低。*
- **强化学习（Reinforcement Learning）仓库发布**：一名成员分享了一个用于实现**强化学习算法**的 GitHub 仓库，该仓库受 Sutton 和 Barto 的书籍启发，旨在涵盖讨论的各种算法。访问该项目[请点击这里](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch)。
  - *社区成员对协作贡献以增强算法实现表现出了兴趣。*
- **AOE2 的动态游戏状态策略**：一名成员提议了一个针对《帝国时代 II》（**Age of Empire II**）的 **CV 项目**，旨在通过使用 **SAM** 和 **YOLO** 等计算机视觉工具映射游戏资产，创建专注于决策策略的 AI 助手。他们的方法涉及高效检测游戏元素。
  - *讨论还涉及了在游戏过程中进行本地动态更新以获得有意义见解的可行性。*
- **需要训练视觉语言模型（Vision Language Models）**：有人对当前 LLM（如 ChatGPT-4）在有效计数和定位图像内物体方面的局限性表示担忧。建议考虑训练**视觉语言模型 (VLM)**，以利用先进的图像处理技术。
  - *视觉和语言模型不断发展的交集为 AI 开发中的工程师带来了新的挑战和机遇。*
- **用于医疗保险申诉的 AI 工具**：推出了一种用于申诉医疗保险拒赔的新工具，利用 **OCR** 扫描信件并生成 AI 驱动的申诉书，可通过 [fighthealthinsurance.com](https://www.fighthealthinsurance.com/) 访问。
  - *重点放在了确保该工具的操作和数据管理符合 **HIPAA** 法律。*

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 模型加载技巧**：**LM Studio** 用户发现保存在不同文件夹中的模型无法直接加载。为了使用模型，需要将它们组织在 **LM Studio** 内部特定的目录结构中。
  - 可以通过 “My Models” 视图更改模型文件夹，从而简化模型管理流程。
- **LM Studio 中的 GPU 故障排除**：有用户报告 **LM Studio** 无法识别其 GPU，引发了关于故障排除步骤的讨论。建议包括检查 **Developer Tab** 中的 LM Runtimes 作为诊断措施。
  - 这突显了兼容硬件在确保软件平稳运行中的重要性。
- **质量测试的 Temperature 设置**：用户讨论了 **LM Studio** 中 **temperature settings** 在评估模型输出中的关键作用，特别是用于质量评估的低设置。敦促初学者查阅资源以了解温度对 **LLMs** 的影响。
  - 这强调了通过精细的参数调整来增强模型性能的必要性。
- **Apple Silicon 的内存带宽限制**：虽然 **Apple Silicon** 提供了极高的内存带宽，但与 GPU 相比，其在 CPU 推理方面的效用有限，引发了性能担忧。**M1 Max** 宣传的 **400GB/s** 在有效性方面仍受到质疑。
  - 讨论表明，实际性能差异显著，值得进一步调查。
- **OpenWebUI 的 RAM 缓存问题**：有报告称 **OpenWebUI** 由于预加载行为消耗了过多的 **RAM**，据称在 **192GB** 中占用了 **150GB**。用户推测缓存管理方式中可能存在软件 Bug 或配置错误。
  - 这强调了在 Web UI 框架中采用稳健资源管理策略的必要性。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **应对技术领域职业倦怠的策略**：成员们讨论了在要求苛刻的技术环境中管理 **burnout** 的各种方法，预计稍后会分享更多见解。
  - *保持动力*被强调为当前环境下开发者面临的主要障碍。
- **CUDA 职位依然稀缺**：有人对 **CUDA 职位稀缺** 表示担忧，公司通常寻找许多合格候选人所缺乏的经验。
  - *这种准入门槛*已成为社区内的一个争议点，影响着新人。
- **Triton 的加载顺序影响性能**：更改 **Triton** 中的加载顺序导致了显著的速度差异，一位用户的速度从 **1.89506** 提升到了 **2.440731**。
  - 这引发了关于编译器在处理加载停顿（load stalls）和指令调度方面性能的疑问。
- **FP8 的 CUDA Kernel 需求**：为了支持 **FP8**，Kernel 需要 **SM_89** 或更高版本，这影响了与 **A100** 等特定 GPU 的兼容性。
  - 在 **4090** 上的测试显示，性能比 torch 提高了 **1.3 倍**，表明了新架构的优势。
- **Activation Checkpointing 的高效使用**：使用极少的代码成功实现了 **Activation checkpointing**，根据处理的 Batch Size 影响内存使用。
  - 配置显示，在不重用的情况下内存需求为 **1211 MiB**，而在重新计算层时为 **176 MiB**。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **注意网络钓鱼！**：参与者对一个可疑网站表示担忧，由于其使用了**不安全的 HTTP 协议**和*未加密的数据传输*，该网站很可能是一个网络钓鱼中心。
  - 他们敦促用户*避免在这些网站上分享个人信息*，以降低安全风险。
- **ComfyUI 面临配置困扰**：用户详细说明了 ComfyUI 的问题，特别是与缺少配置文件相关的错误以及对模型安装的困惑。
  - 有人建议利用 *Save Text File* 节点来跟踪 ComfyUI 中的提示词和工作流。
- **获得更好结果的提示词技巧**：对于 Stable Diffusion，由逗号分隔的属性构成的提示词结构可以产生更好的效果，尤其是对于像 **SD 1.5** 这样的旧模型。
  - 然而，得益于增强的文本编码能力，新模型更适合使用*自然语言提示词*。
- **关于 Stable Diffusion 3.1 的推测**：参与者推测了 **Stable Diffusion 3.1** 的发布，并指出目前信息有限，且大多来自非官方渠道。
  - 在社区等待 **Stable AI** 官方公告之际，他们呼吁大家保持*耐心*。
- **对模型训练资源的需求**：用户表示需要针对特定角色和艺术风格训练 **LoRA** 模型的指导，强调了更新资源方面的空白。
  - 社区分享了一个 [Flux 的 GitHub 仓库](https://github.com/black-forest-labs/flux)，这可能有助于深入了解新模型的功能。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 标准库开放贡献**：**Mojo Standard Library** 已部分开放贡献，尽管某些部分仍与编译器紧密绑定。尽管已有稳定版本可用，但对其生产就绪性的担忧依然存在，**稳健的稳定性保证**仍需建立。
  - 成员表示鼓励更新和贡献，但该库的全部潜力仍有待实现。
- **Modular CLI 逐步接近最终版本**：**Modular CLI** 的更新表明它已接近完成，随后将引入 **Magic**，这将把包管理功能推向前端。目前的开发重点主要是 GPU 支持，标志着纯 CPU 版本的发布即将结束。
  - 开发者们对类似于 **Rust 的 Cargo** 那样更流畅的包管理体验充满期待，旨在提升可用性。
- **MLIR 指向语言互操作性的进展**：关于 **MLIR** 集成的讨论强调了其桥接不同编程语言间通信的潜力，尽管翻译挑战依然存在。值得注意的是，成员们评论说 MLIR 可能会简化某些方面，但同时也会使其他方面复杂化。
  - 讨论中提出了有关**向后兼容性**以及适应现有 C 预处理器依赖项的担忧。
- **OSDI '21 主题演讲赞扬 MAX**：来自 [OSDI '21](https://www.youtube.com/watch?v=36myc8wQhLo) 的主题演讲强调，MAX 可以增强 AI 和 HPC 之外的计算能力，并提到其优化硬件交互的潜力。**Mojo + MAX** 的结合可以促进对各种处理器的更好利用。
  - 预期这种集成将显著提升各种系统的计算能力。
- **内存域可视化为图节点**：讨论建议将内存域（Memory Domains）表示为图节点，以增强理解它们之间延迟和带宽等关系的能力。这种方法可以允许硬件感知编译器（hardware-aware compilers）就数据移动做出明智的决策。
  - 成员们承认现有通道存在摩擦，表示打算开发一个基于 DPDK 的通道，以在管理可变计算时间的同时缓解这些复杂性。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 内容质量辩论升级**：参与者认为 AI 工具的兴起可能会导致更多**低质量、标题党内容**，从而可能降低在线信息的整体质量。
  - 然而，一些人断言 AI 生成内容之间的竞争将推动更高的标准，并**提高相关性和准确性**。
- **AI 辅助求职申请但引发担忧**：讨论透露，个人正在使用 AI 为求职申请创建量身定制的简历，然后 AI 工具会对其进行效率评估。
  - 这引发了对潜在的 **no human in the loop**（无人工参与）场景影响招聘标准的担忧。
- **LAION 数据集恢复访问**：LAION 数据集在之前因内容担忧被移除后，现在已可以再次访问，即将进行的更新将使其与 **Clip retrieval API** 集成。
  - 参与者分享了访问该数据集的资源，以增强 AI 训练。
- **基于 LLM 的 Agent 发布深度论文**：Manifold Research Group 发布了一篇题为 *Intelligent Digital Agents in the Era of Large Language Models* 的立场论文，强调了**基于 LLM 的 AI Agent** 的进展。
  - 该论文探讨了突破和局限性，并邀请在其 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 上进行进一步讨论。
- **发布新的 MultiNet 评估指标**：Manifold 定义了用于基准测试多个 **Vision-Language Models (VLMs)** 及应用的新评估指标，可在其 [GitHub repository](https://github.com/ManifoldRG/MultiNet?ref=manifoldrg.com) 中获取。
  - 该倡议旨在提供详细的数据集覆盖范围并改进 AI 指标中的质量评估。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Manifold Research Group 发布立场论文**：Manifold Research Group 分享了他们最近关于**基于 LLM 的自主 Agent** 的[立场论文](https://www.manifoldrg.com/llm-agents/)，展示了自主系统的进展。
  - 他们邀请感兴趣的人士加入其 [Discord 社区](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com)进行更多讨论。
- **Manifold 的算力可用性挑战**：确认了来自 Manifold 的有限算力（compute）选项，这依赖于学术和行业合作伙伴关系，具体细节因项目而异。
  - 有关可用算力资源的查询已转交给 **Harsh** 或 **Sidh** 以获得针对性指导。
- **ICLR 会议声望高于 NIPS workshop**：讨论强调，在 **ICLR** 主会上发表论文对简历的影响力显著高于在 **NIPS workshop** 发表，因为 workshop 的录取率较低。
  - ICLR 作为 **tier 1 conference** 的认可度得到了强调，为其论文增加了分量。
- **探索 LLM 与抽象-结晶步骤**：有提议建议 LLM 可以通过引入**抽象-结晶（abstraction-crystallization）**步骤来改进，以评估多个抽象短语，从而增强输出的创造力。
  - 这可能涉及通过向量相似度对短语进行排名，引导输出远离对最高概率（top-probability）的依赖。
- **关于 Diffusion Models 学习物理规律的讨论**：人们对 Diffusion Models 在准确学习物理定律与仅在现有数据集上**过拟合（overfitting）**之间的有效性表示担忧。
  - 有人指出，强制执行物理结构可能会限制这些模型的表达能力，值得进一步研究。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **学生可免费获得一个月 Perplexity Pro**：学生在 **9 月 15 日**前使用 .edu 邮箱注册，即可领取 **一个月免费的 Perplexity Pro**。该服务在为学术研究提供快速、精准的回答方面表现出色。
  - 其功能涵盖了从剖析复杂话题到制定饮食计划等多个方面，是学习者的多功能工具。
- **达到 500 人注册，全校即可赢取免费访问权限**：如果一个校区达到 **500 人注册**，全校学生都将免费获得 **一年的 Perplexity Pro**，这激发了竞争精神。
  - 该挑战活动持续至 **9 月 15 日**，用户可以在[此处](https://www.perplexity.ai/backtoschool)查看注册进度。
- **Perplexity API 的使用引起了兴趣**：一名成员探讨了结合使用 API 与 [Make.com](https://make.com) 创建 Perplexity 页面的潜力，反映了用户对集成的兴趣。
  - 目前的文档对此缺乏清晰说明，因此有人建议咨询官方 [Perplexity 文档](https://docs.perplexity.ai)以获取进一步指导。
- **Pro API 的文件上传功能**：有疑问指出 Pro API 在通过 CLI 界面进行搜索查询时，是否具备接受 .txt 和 .pdf 等文件上传的能力。
  - 用户希望获得类似于 Web 界面的功能，表明了对增强分析能力的渴求。
- **Perplexity Xfinity 优惠引发热议**：关于 [Perplexity Xfinity 优惠](https://www.perplexity.ai/search/perplexity-xfinity-deal-QCK.FX71SZCO6kSpE0YtYQ)的分享链接暗示了将为用户提供令人兴奋的优惠，可能会提升用户体验。
  - 细节尚不明确，但人们对这一合作伙伴关系可能带来的内容充满期待。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral-Nemo 价格大幅下调**：[Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) 的价格下降了 **23%**，反映了市场动态的变化。
  - 这一显著的价格变动可能预示着 **Mistral** 模型需求或供应的转变，促使分析师关注竞争对手的反应。
- **Mume AI 应用惊艳亮相**：使用 OpenRouter 作为供应商推出的 **Mume AI** 应用，为用户提供了超过 **100 个模型**用于文本和图像生成。
  - 开发者正积极寻求社区 **反馈**，以便在该应用进入早期阶段时进行优化，从而促进用户参与。
- **Google 和 Claude 模型的缓存功能**：讨论透露，通过 OpenRouter 实现 **Google** 和 **Claude** 模型的缓存功能可能即将落地。
  - 用户对缓存路由表示了担忧，特别是考虑到这两个端点并不共享同一个缓存。
- **关于多轮对话支持的澄清**：针对 OpenRouter 中 **多轮对话 (multi-turn conversations)** 的咨询澄清了用户必须重新发送整个聊天历史记录以保持连续性。
  - 回复指出，由于 LLM 本质上是无状态的 (stateless)，用户需要自行管理这一环节。
- **AI 中保持角色一致性的最佳模型**：一位用户寻求关于保持角色一致性的最佳模型建议，并表示对 **Midjourney** 不太满意。
  - 讨论旨在创建一个可靠的 Instagram AI 影响力者 (influencer)，期间推荐了 **Segmind** 等替代方案。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NousCon 活动宣布于 9 月 18 日举行**：**NousCon** 活动定于 **9 月 18 日**在**旧金山**举行，紧接在 **PyTorch Conference** 之后。
  - 鉴于**名额有限**，鼓励热情的参与者查看[官方公告](https://x.com/NousResearch/status/1831032559477866754)并点击[此处](https://lu.ma/zlgp0ljd)的注册链接预订席位。
- **Hermes-3 以闪电般的速度完成训练**：**Hermes-3** 的训练过程现在仅需 **4 分钟**即可完成，这引发了人们对训练技术效率的关注。
  - 这种极快的训练速度引发了社区成员关于“训练速通（speedrunning training）”的调侃。
- **质疑 LLM 推理框架**：成员们注意到目前缺乏解决 **LLM Reasoning and Planning** 的显著**框架**，凸显了有效解决方案的空白。
  - 讨论中包含了对 **LLM-Modulo 概念**的怀疑，一些成员主张关注 **Yann LeCun** 建议的实际应用。
- **介绍 Gemma 2：从 Numpy 到 CuPy 的迁移**：一位成员正在尝试使用 **Numpy** 从头开始实现 **Gemma 2**，并计划将其迁移到 **CuPy** 以提升性能。
  - 他们分享了 [Numpy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) 和 [CuPy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb) 的链接，以及有效运行所需的 **GPU** 显存建议。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **SearchGPT 发布猜测升温**：用户猜测 **SearchGPT** 即将发布，一些用户在加入等候名单后短暂看到了显示“You're in”的弹窗，尽管访问权限很快就消失了。
  - 另一位用户指出 **Perplexity** 的表现优于 **SearchGPT**，特别是由于 **Arc** 集成了 **Perplexity**，使其目前成为更受欢迎的选择。
- **AI 探索游戏内容的趣味性**：一位成员提出了制作 **AI 玩 UNO** 视频的想法，引发了关于 **AI** 在参与性内容创作中潜力的讨论。
  - 这一概念反映了人们对利用 **AI** 在游戏中实现互动体验日益增长的兴趣。
- **GPT-4o 提供比 Turbo 更具吸引力的特性**：**GPT-4o** 被宣传为比 **GPT-4 Turbo** 便宜 **50%**，成本为 **每百万输入 token 5 美元，每百万输出 token 15 美元**，同时拥有 **2 倍的速度**和高达 **每分钟 1000 万 token** 的 **5 倍速率限制**。
  - 凭借 **128k 上下文窗口**和增强的**视觉能力**，**GPT-4o** 将自己定位为寻求效率的用户的强力竞争者。
- **社区对 ChatGPT 政策感到沮丧**：用户对 **ChatGPT** 处理敏感话题的方式表示担忧，注意到响应模式的变化和消息删除的增加，这可能会阻碍用户使用。
  - 用户呼吁 **AI** 开发者提高透明度和响应速度，以解决这些持续存在的问题。
- **通过清晰度提升 AI 写作**：成员们强调需要更清晰的指令来减少 **AI** 回复中不需要的短语，主张转向提供所需语言的正向示例。
  - 通过强调模型“应该做什么”而不是“避免做什么”，参与者注意到这可以产生更符合行为技术预期的有效结果。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **自动文档检索提升效率**：最近的一个 notebook 展示了将 **RAG (Retrieval-Augmented Generation)** 与结构化查询相结合，增强了针对大型数据集的文档检索能力，详见[相关帖子](https://t.co/nfXnpvM93D)。
  - *如何检索正确的文档？* 该方法有效地解决了这一挑战。
- **LLM 轻松制作 PowerPoint 幻灯片**：一个创新的 TypeScript 应用可以将笔记转换为 **PowerPoint 幻灯片**，让用户摆脱繁琐的任务，专注于创意，该[演示链接](https://t.co/ItJ3edWmXF)展示了其功能。
  - 该应用不仅能总结笔记，还能生成额外内容，展示了 **LLM** 的强大能力。
- **关于 Jina AI Late Embeddings 类的提案**：一名成员提议利用新的“late embeddings”方法为 **Jina** 开发一个 embeddings 类，参考见 [HF 代码](https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling)。
  - 另一名成员建议，通过使用 BaseNodeParser 类，大部分代码可能适用于 node parser 软件包。
- **Gemini LLM 在初始化时遇到困难**：一位用户在重启内核后遇到了 **Gemini** LLM 的 **AttributeError**，并指出在更改之前它是可以正常工作的。
  - 建议更新依赖项，以解决由于最近 **pydantic** 升级引起的问题。
- **聊天引擎消息过滤咨询**：一名成员寻求一种从 LLM 查询的消息历史记录中过滤答案的方法，旨在仅将问题发送给聊天引擎。
  - 另一名成员提出，通过子类化 memory 并重写 `get()` 方法可能是一个潜在的解决方案。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **H200 价格维持在 18 万美元的高位**：目前，**H200** 的 **8** 卡规格价格为 **18 万美元**，这引发了关于高需求影响市场定价的讨论。
  - 成员们正在关注这一价格如何影响 AI 硬件生态系统的可及性。
- **H100 价格飙升与 Tesla 有关**：近期 **H100** 价格的**巨大涨幅**被认为与 **Tesla** 的活动有关。
  - 社区很好奇此类行业的持续需求将如何改变未来的定价策略。
- **聊天模板 PR 助力设置**：**聊天模板 PR** 被强调为自动加载 tokenizer 模板的关键，显著简化了设置过程。
  - 这一进展预计将为使用 AI 聊天界面的新用户简化入门流程。
- **SFTT 中的交叉熵损失说明**：一位用户询问 **SFTT** 是否计算 **cross entropy loss**（交叉熵损失），另一位用户引导其查看 GitHub 上 **LLaMA** 的建模代码进行确认。
  - 这突显了清晰列出代码库引用对于理解损失计算的重要性。
- **探索用于微调的多人对话**：一位成员讨论了在没有 Agent 的情况下，利用**多人对话**对模型进行微调，重点在于如何格式化此类数据。
  - 讨论涉及了通过聊天历史提示词训练模型，以使其更好地掌握对话流。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Playground 中的新工具引发关注**：成员们确认 Playground 中的新模型现已**启用工具 (tools)**，促进了探索和创意。
  - 在此公告发布后，一位团队成员发出了热情的鼓励：“祝开发愉快！”
- **LLM 是否能辅助报告生成？**：有人询问是否可以使用 **LLM** 根据之前的写作风格和会议记录为内部审计团队生成报告。
  - 成员们受邀分享利用这些模型进行高效报告生成的经验。
- **模型卡片差异被指出**：一名成员指出 [model card](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) 错误地将模型大小标注为 **35B**，而非 **32B**。
  - 团队承认了这一疏忽，并承诺很快会进行修正。
- **Cohere 支持 Server Side Events！**：已确认向聊天 API 发送 `Accept: text/event-stream` 请求头将允许用户接收 **SSE 事件**。
  - 文档更新正在进行中，以包含这一此前未公开的功能。
- **功能请求流程已明确**：一名成员询问如何提交关于服务器端事件的功能请求，引发了团队成员之间的对话。
  - 反馈已被采纳，并计划与产品团队进行进一步讨论。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **编排你的 Multi-Agent 对话助手**：一位成员寻求建立 **Multi-Agent 对话助手**的帮助，特别是对 **Supervisor 架构**及其固有的复杂性感兴趣。
  - 讨论强调了不同的架构方法，并呼吁分享经验和见解。
- **Hybrid Retriever 是未来**：一位用户提出了 **Hybrid Retriever** 的概念，它结合了**两个或多个检索器**以增强搜索性能。
  - 这个想法激发了热情，成员们对其潜在应用表示兴奋。
- **揭秘 Hugging Face Embeddings**：一位成员讨论了将 **encode_kwargs** 传递给 **Hugging Face embedding endpoint**，并分享了一段代码片段以供参考。
  - 他们确认 **TEI** 会自动处理 embedding 归一化，简化了他们的实现。
- **Toolio 0.5.0 带来令人兴奋的特性**：**Toolio 0.5.0** 的发布引入了改进的文档，并支持符合 [JSON schema](https://json-schema.org/) 的 LLM 响应生成。
  - 开发者可以通过针对其需求定制的结构化输出，实现对文本生成的更多控制。
- **Generative AI 项目需要你的 Star**：一位成员在 GitHub 上分享了他们今年的 **Generative AI 项目**，鼓励其他人[查看他们的作品](https://www.linkedin.com/posts/isham-rashik-5a547711b_github-di37dubai-tech-talk-intelligent-chatbot-activity-7236606074173222912-Fp2U)并为仓库点亮 Star。
  - 对项目参与度的推动强调了社区反馈对于项目曝光和协作的关键作用。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Python PATH 引起困惑**：一位成员在虚拟环境中使用 `pip install open-interpreter` 多次安装后，在让其 **Open Interpreter** 的 Python 脚本识别模块时遇到挑战。
  - 这引发了社区关于*环境设置最佳实践*的持续讨论。
- **House Party 活动公告**：宣布了一场令人兴奋的 **House Party** 活动，承诺将带来迄今为止最具影响力的重大新闻和演示。
  - 活动将进行**直播**和录制，但鼓励参加者亲临现场，以免错过体验。
- **每周 Tool Use 推荐**：本周的 **Tool Use** 节目邀请了一位嘉宾，重点介绍了他们的见解和讨论。你可以点击[这里](https://www.youtube.com/watch?v=UUUWt8GaS64)查看该剧集。
  - *感谢社区的支持*——经验分享继续活跃着围绕工具使用的讨论。
- **与嘉宾的愉快交流**：成员们表达了在 Tool Use 环节中与新嘉宾交流的喜悦。
  - *一位成员分享了他们在对话中的快乐*，为共同学习创造了一个包容的环境。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **同行数据影响结果**：一位成员确认，当数据源自同一个 **sample** 时，来自同一行的所有数据点都会影响**最终结果**。
  - 他们进一步询问了正在分析的**特定数据集**，强调了明确数据交互的必要性。
- **LoRA Checkpoints 引发疑问**：尽管设置了 `adapter_weights_only`，但在 checkpoint 字典中使用完整的合并 adapter 权重引起了关注。
  - 澄清说明该过程在 [Llama 405B PR](https://github.com/pytorch/torchtune/pull/1449) 中已被**完全移除**，尽管所有 recipe 的更新仍在进行中。
- **更多 Adapter 权重支持的空间**：有人建议增强在微调配置中支持 `adapter_weights_only` 的灵活性。
  - 这与旨在提高 AI 模型训练当前用户易用性的普遍共识相一致。
- **Max Sequence Length 解决方案即将到来**：围绕新一代更新的兴奋感与日俱增，讨论中涉及了对 **max_seq_len** 问题的潜在修复。
  - 对解决这些挑战的协作努力充满信心，表明社区正在采取积极主动的方法。
- **Max Sequence Length 重构草案正在评审中**：分享了 **max_seq_len** 实现重构的草案，表明 GitHub 上的开发正在进行中。
  - 该成员承诺在明天的讨论后更新文档，展示了致力于改进的决心。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **排行榜中缺失模型的致歉**：团队承认在重新生成排行榜结果时疏忽了一个 **model**，并承诺在下次更新中予以纠正。
  - 这一承诺旨在提高排行榜上模型展示的准确性。
- **新数据集优先于 Hermes 模型**：重心已转移到新的 **dataset release**，导致新模型请求的处理推迟到本周晚些时候或下周。
  - 鼓励成员在等待更新期间为他们心仪的模型提交 PR。
- **Chat 模式增加了解码的复杂性**：模型现在同时在 **chat mode** 和 **FC mode** 下运行；后者有助于结构化输出，提高解码效率。
  - chat mode 中的 **DEFAULT_SYSTEM_PROMPT** 旨在更系统地引导回复。
- **澄清排行榜数据来源**：`leaderboard_live.html` 使用的是 **BFCL V2-Live dataset**，而主页面 `leaderboard.html` 汇总了所有 **BFCL V2 datasets**（包括 Live 和非 Live）。
  - 理解这一区别对于准确解读排行榜结果至关重要。
- **在 GitHub 上提出关于排行榜差异的问题**：一名成员报告称在 GitHub 上提交了一个关于排行榜差异的 issue，并提供了 [issue 链接](https://github.com/ShishirPatil/gorilla/issues/620)。
  - 他们还表示，如果其解决方案能匹配所述问题，愿意提交 PR。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mini-Omni 语音模型开源**：[Mini-Omni](https://hf.co/gpt-omni/mini-omni) 已发布，这是一个能够同时生成文本和音频的开源模型，适用于实时语音对话。其 [代码库](https://github.com/gpt-omni/mini-omni) 和随附的研究论文详细介绍了该模型令人印象深刻的流式音频输出能力。
  - Twitter 上的讨论强调了该对话模型的潜在应用和令人兴奋的前景，以及它对未来 AI 交互的影响。
- **对 10万块 H100 集群的深刻分析**：对 **100,000 H100 clusters** 的全面考察涉及了能效、网络拓扑以及 Ethernet 和 InfiniBand 方案之间的权衡。报告指出，尽管维持了相似的计算指标，但这些集群反映出 **GPT-4** 之后 AI 进展的放缓。
  - 这份详细分析引发了对集群可靠性和故障恢复的担忧，表明在有效扩展当前模型方面存在挑战，如[此报告](https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true)所示。
- **新版 Latent Space Podcast 启动**：[Latent Space](https://x.com/latentspacepod/status/1831020483967701260) 宣布了新的播客剧集，重点关注 AI 工程的最新趋势。旨在应对不断变化的格局，并分享该领域领先专家的见解。
  - 听众可以期待深入探讨核心 AI 话题和社区驱动知识分享的启发性讨论。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **探索 WeaviateRM 集成**：一名成员对 **WeaviateRM integration** 表现出兴趣，并请求创建一个关于 **text2vec-ollama** 的论坛议题。他们分享了 [Weaviate 论坛](https://forum.weaviate.io/latest) 的链接以供进一步讨论。
  - 另一名成员确认愿意提供帮助，同意发起论坛议题，并以感谢结束了对话。
- **探索使用 COPRO 进行长度管理**：一名成员询问如何使用 **COPRO** 或类似模型来有效优化指令长度，并建议调整 **max_tokens**。
  - 他们提议实施一个指标返回系统，作为管理指令长度的一种方式。
- **Zero-shot 指令优化器技术**：讨论围绕采用 **zero-shot instruction optimizer** 来控制模型中的指令长度展开。
  - 成员们辩论是仅通过限制 **max_tokens** 来设置长度约束，还是为指令和输入长度创建复杂的指标。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM 增强报告生成**：一位成员询问如何利用 **LLMs** 根据以往的写作风格和会议记录生成报告，旨在协助内部审计团队进行报告创作。
  - 此次讨论强调了自动化报告生成在提高效率方面的潜力。
- **会议记录的多样化定义**：围绕“会议记录”一词进行了澄清，建议其可能包括带有与会者姓名的完整转录文本。
  - 这引发了关于什么是完整的会议文档的不同解读的深入对话。
- **合成会议初具规模**：一位用户分享了他们使用 [persona-hub](https://github.com/tencent-ailab/persona-hub) 创建合成会议格式并促进模拟对话的工作。
  - 他们注意到这些模拟中的 token 使用量很高，但赞赏它为训练 LLMs 带来的丰富多样性。
- **会议摘要的文本转语音计划**：计划实施 Text-to-Speech，利用 LLMs 进行摘要提取，并从会议摘要中生成音频。
  - 此外，重点在于训练一个 **whisper model** 用于说话人识别，以增强会议期间的来源归属。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 亮点**：George Hotz 的项目 **tinygrad** 展示了一种极简主义的深度学习方法，为大型框架提供了一个有趣的替代方案。
  - 尽管聊天中的细节较少，但围绕 **tinygrad** 的兴奋情绪表明 AI 工程师对轻量级解决方案的兴趣日益浓厚。
- **社区参与**：该频道进行了简短的互动，th.blitz 热情地向成员打招呼，这突显了社区的活跃参与。
  - 这个简单的问候表明，即使是微小的互动也能在技术讨论中培养归属感。

---

**Alignment Lab AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**Interconnects (Nathan Lambert) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1280248522909683823)** (592 条消息🔥🔥🔥):

> - `Unsloth fine-tuning`
> - `Gemma 2B model`
> - `Chat templates`
> - `Dataset quality`
> - `LLM training parameters`

- **Unsloth 微调的挑战**：用户讨论了微调 Gemma 2B 模型时遇到的问题，特别是训练后模型生成随机内容的挑战。
  - 据观察，更改训练参数或数据集可能会导致模型输出出现意外结果。
- **模板一致性的重要性**：对话强调，在对 instruct-tuned 模型进行微调时，为了获得最佳效果，使用其原始微调时所用的相同模板至关重要。
  - 用户认为更改模板可能会导致 Token 使用效率降低和推理（inference）挑战。
- **数据集质量重于数量**：参与者一致认为，对于有效的微调，数据集的质量比数量更重要。
  - 为了获得最佳结果，建议使用高质量的数据集进行微调。
- **微调中的实验**：在保持传统方法的同时，参与者表示愿意尝试各种微调参数，如 rank 和 alpha。
  - 大家认识到，即使打破常规，实验也能产生有价值的见解。
- **协作与学习**：在整个讨论过程中，用户分享了见解和经验，营造了学习 LLM 微调的协作氛围。
  - 成员们对社区的帮助和交流的丰富知识表示感谢。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct">unsloth/Meta-Llama-3.1-8B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1vIrqH5uYDQwsJ4-OO3DErvuv4pBgVwk4?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://tenor.com/view/jambajew-steve-brule-stare-gif-18457155">Jambajew Steve Brule GIF - Jambajew Steve Brule Stare - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://lu.ma/xd0zzk0h">Continued Pretraining and Fine-Tuning with Unsloth · Luma</a>：持续预训练（Continued pretraining）与有监督微调（SFT）在业界的小语言模型（SLMs）中正变得越来越流行。寻找更快的……</li><li><a href="https://ollama.com/library/llama3.1:8b-instruct-fp16">llama3.1:8b-instruct-fp16</a>：Llama 3.1 是来自 Meta 的新型 SOTA 模型，提供 8B、70B 和 405B 参数版本。</li><li><a href="https://huggingface.co/collections/HuggingFaceTB/smollm-6695016cad7167254ce15966">🪐 SmolLM - a HuggingFaceTB Collection</a>：未找到描述</li><li><a href="https://tenor.com/view/jizz-adult-swim-john-reilly-blink-gif-14841420">Jizz Adult Swim GIF - Jizz Adult Swim John Reilly - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B">unsloth/Meta-Llama-3.1-8B · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d05x6v/llamacpp_runs_18_times_faster_than_o">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1d05x6v/llamacpp_runs_18_times_faster_than_ollama/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: Python bindings for llama.cpp</a>：llama.cpp 的 Python 绑定。通过在 GitHub 上创建一个账户来为 abetlen/llama-cpp-python 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/975#issuecomment-2323009118">Single GPU training in Multi-GPU system doesn't work. · Issue #975 · unslothai/unsloth</a>：在多 GPU 系统中进行单 GPU 训练不起作用，即使在导入 unsloth 之前通过 os.environ CUDA_VISIBLE_DEVICES 限制为 1 个 GPU。原因：check_nvidia 函数会产生新进程来检查……</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/include/llama.h">llama.cpp/include/llama.h at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建一个账户来为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://tenor.com/view/o-hearn-gif-3900414469346077199">O Hearn GIF - O hearn - Discover &amp; Share GIFs</a>：点击查看 GIF</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1280640529142382602)** (3 条消息):

> - `llama.cpp 与 RPC 的集成`
> - `API 订阅考量`

- **llama.cpp 与 RPC 服务器的挑战**：一位成员表示在将 **llama.cpp** 与 RPC 服务器配合使用时遇到困难，称其无法在服务器机器上保留任何内存。
  - *“我不知道为什么它不保留任何内存”* 表达了对集成过程的挫败感。
- **因成本原因切换 API 使用方式**：另一位成员提到考虑从订阅服务切换为仅使用 **API**，因为他们每月消耗的 Token 总额不足 **$20**。
  - 这反映了用户在优化 AI 访问相关成本方面的潜在趋势。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1280281516232015944)** (19 条消息🔥):

> - `DPO Notebook 推理`
> - `Unsloth 安装问题`
> - `Xformers 相关的 TypeError`
> - `文本转语音 (Text-to-Speech) 模型微调`
> - `购买 Unsloth 的联系方式`

- **DPO Notebook 缺少推理代码**：一位用户参考了一个用于微调 Llama 模型的 DPO notebook，但注意到其中没有提供推理代码。
  - 另一位成员建议从现有的推理 notebook 中复制推理代码作为解决方案。
- **Unsloth 的安装问题**：一位用户在 Docker 容器中安装 Unsloth 时遇到问题，并报告了过程中出现的一个奇怪错误。
  - 另一位成员建议创建一个 Python 版本为 3.9 或 3.10 的新环境作为潜在的修复方案。
- **与 Xformers 相关的 TypeError**：成员们讨论了在运行模型生成命令时遇到的 TypeError，特别是提到了 'Multiple dispatch failed' 错误。
  - 一位用户找到了解决方案，尽管他们不确定自己采取了哪些步骤来解决它。
- **文本转语音模型微调的资源**：一位 AI 微调初学者询问 Unsloth 是否可以协助微调文本转语音 (Text-to-Speech) 模型，但被告知该功能目前不支持。
  - 他们寻求相关资源推荐，并提到了一份 Whisper 训练指南，该指南可能需要更大的数据集才能进行有效训练。
- **关于购买 Unsloth 的咨询**：一位用户表达了购买 Unsloth 的兴趣，并询问该交易的合适联系人。
  - 另一位成员建议联系项目团队或 Unsloth Pro 以获取帮助。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://colab.research.google.com/drive/15vttTpzzVXv_tJwEk-hIcQ0S9FcEWvwP?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1aqlNQi7MMJbynFDyOQteD2t0yVfjb9Zh?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：查看下方列表获取我们所有的 notebook：</li><li><a href="https://download.pytorch.org/whl/cu121">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/blog/fine-tune-whisper">使用 🤗 Transformers 为多语言 ASR 微调 Whisper</a>：未找到描述</li></ul></div>

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1280490048659390617)** (1 条消息):

> - `Gemma 2 实现`
> - `Numpy vs Cupy`
> - `GPU 需求`

- **从零开始实现 Gemma 2**：在过去的 3 天里，一位成员成功使用 [Numpy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) 从零开始实现了 **Gemma 2**，随后将其移植到了 [Cupy](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb)。
  - 该实现展示了在 GPU 和 CPU 上运行 Gemma 2 的能力，使其适用于不同的硬件配置。
- **Cupy GPU 需求**：为了获得最佳性能，**Cupy** 版本需要一块拥有 **24GB** 显存的 GPU，这对于高效处理计算至关重要。
  - 另外，对于显存小于 **16GB** 的 GPU，用户可以运行 [Cupy f16 版本](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb) 以在执行计算时节省内存。
- **使用 Numpy 在 CPU 上运行**：用户仍然可以使用 [Numpy notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) 在 CPU 上运行该实现，为没有高性能 GPU 的用户提供了更广泛的选择。
  - 这一选项对于测试和不需要大量硬件资源的规模较小的计算非常有用。

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1280623804481142857)** (1 条消息):

> - `Phi-3.5-mini`
> - `关于视觉语言模型的新论文`
> - `构建你自己的机器人`
> - `TRL v0.10.1 发布`
> - `碳排放追踪`

- **Phi-3.5-mini 在浏览器中运行**：**Phi-3.5-mini (3.8B)** 模型现在可以使用 WebGPU、Transformers.js 和 ONNX Runtime Web 在浏览器中以约 90 tokens/秒的速度运行，实现了完全本地化处理以增强**隐私**。
  - 演示和源码可通过[此链接](https://x.com/xenovacom/status/1826992922509595068)获取。
- **发布了极具洞察力的新论文**：Hugging Face 的一篇新论文提供了关于**最先进 (state-of-the-art)** 视觉语言模型及其当前局限性的见解，适合初学者和专家。
  - 如果你正在寻找该领域的新视角，这篇论文值得一读；请在[这里](https://x.com/HugoLaurencon/status/1827986085097402553)查看。
- **创建你的自主机器人**：发布了一份关于如何**构建你自己的机器人**的深入教程，允许用户仅用一台笔记本电脑就能教它新技能。
  - 这种交互式方法让你的自制机器人能够自主行动；教程可以在[这里](https://x.com/RemiCadene/status/1825455895561859185)找到。
- **TRL v0.10.1 包含大量新功能**：**TRL v0.10.1** 的发布包括 DeepMind 的 Online DPO 等增强功能，以改进 LLM 对齐，并集成了 Liger kernel 以加速 SFT。
  - 在 [GitHub](https://github.com/huggingface/trl/releases/tag/v0.10.1) 上探索各种新功能，包括用于视觉语言模型的 DPO。
- **模型卡片新增碳排放追踪功能**：Hub 上推出了一项新功能，直接在模型卡片上显示模型训练期间的碳排放量。
  - 该倡议旨在鼓励模型作者分享他们的**碳排放**数据；更多详情请见[这里](https://x.com/AymericRoucher/status/1830621688163127417)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://x.com/xenovacom/status/1826992922509595068)">来自 Xenova (@xenovacom) 的推文</a>：简直不敢相信... Phi-3.5-mini (3.8B) 在浏览器中通过 WebGPU 配合 Transformers.js 和 ONNX Runtime Web 以约 90 tokens/秒的速度运行！🤯 由于一切都在 100% 本地运行，不会发送任何消息...</li><li><a href="https://x.com/HugoLaurencon/status/1827986085097402553)">来自 Hugo Laurençon (@HugoLaurencon) 的推文</a>：无论你是：• 想要对 SOTA VLM 方法及其局限性有高层次了解的完全初学者 • 正在寻找该领域新方向的专家，我们的新论文可能...</li><li><a href="https://x.com/RemiCadene/status/1825455895561859185)">来自 Remi Cadene (@RemiCadene) 的推文</a>：等待终于结束了！！！😁 我们刚刚发布了一份关于如何构建你自己的机器人的深度教程！只需一台笔记本电脑，通过演示几个动作就能教它新技能。然后看着你自制的机器人...</li><li><a href="https://x.com/NielsRogge/status/1828010283530424684)">来自 Niels Rogge (@NielsRogge) 的推文</a>：好了，终于可以用 Flux 免费给自己进行 Dreambooth 了！请注意，这实际上是 @levelsio 或类似 @FAL 或 @replicate 的服务正在变现的内容。以下是操作方法（简短线程）：</li><li><a href="https://x.com/_marcsun/status/1828824017593385276)">来自 Marc Sun (@_marcsun) 的推文</a>：`transformers` + `torchao` 量化 + `torch.compile` 实现更快的推理速度和更少的内存占用 🔥 "meta-llama/Meta-Llama-3.1-8B-Instruct" 的 4-bit 仅权重（weight-only）量化演示：</li><li><a href="https://x.com/_lewtun/status/1829184370390777980)">来自 Lewis Tunstall (@_lewtun) 的推文</a>：TRL v0.10.1 发布了，内容非常充实 💪 🔁 来自 @GoogleDeepMind 的 Online DPO 用于对齐更好的 LLM 🐯 来自 @LinkedIn 的 Liger kernel 集成以增强 SFT 🖼️ 针对 VLM 的 DPO：🌋 LLaVa, ✨ PaliGem...</li><li><a href="https://x.com/abhi1thakur/status/1828049871967846897)">来自 abhishek (@abhi1thakur) 的推文</a>：🚨 新竞赛警报 🚨 来自 ROAM 挑战赛的真实世界对抗性攻击（Real-world Adversarial Attack）解决了在图像可能被故意对抗的环境中部署深度学习系统的关键问题...</li><li><a href="https://x.com/AymericRoucher/status/1830621688163127417)!">来自 Aymeric (@AymericRoucher) 的推文</a>：Hub 上的新功能！☁️ 训练期间产生的碳排放现在会显示在模型卡片上！（需要模型作者先填写该信息）希望这能促使更多人展示碳排放...</li><li><a href="https://x.com/abhi1thakur/status/1830963506252067277)">来自 abhishek (@abhi1thakur) 的推文</a>：如何在 Hugging Face 上训练你自己的 Flux LoRA：Twitter 上最简单的 LoRA 训练指南。在这个线程中，我将向你展示如何在 Hugging Face 上为各种...训练你自己的 Flux LoRA。</li><li><a href="https://x.com/_philschmid/status/1828441244558618944)">来自 Philipp Schmid (@_philschmid) 的推文</a>：宣布“Cloud AI Tuesdays”。🚀 每周二，我们将分享如何在云端（@googlecloud, @awscloud, @microsoft Azure…）使用开放模型构建 AI 的详细示例 ☁️ 今天，我们开始...</li><li><a href="https://x.com/mervenoyann/status/1826005697924050966)">来自 merve (@mervenoyann) 的推文</a>：微软发布了一系列 Phi-3 模型，包括一个视觉模型！🤏🏻 4.2B 模型，128k token 上下文长度 🥹 MMMU 得分 43.0（对于这个尺寸来说非常好）🎥 支持单张/多张图像和视频...</li><li><a href="https://x.com/mervenoyann/status/1829144958101561681)">来自 merve (@mervenoyann) 的推文</a>：NVIDIA 刚刚发布了 NVEagle 🦅 令人印象深刻的视觉语言模型，提供 7B、13B 以及针对聊天微调的 13B 版本，通过 MoE 视觉编码器提升了视觉感知能力 💬 继续阅读了解详情...</li><li><a href="https://x.com/huggingface/status/1829549834652483983)">来自 Hugging Face (@huggingface) 的推文</a>：几位 Hugging Face 团队成员将前往旧金山参加 PyTorch Conference，我们将以独特的方式庆祝。欢迎参加 9 月 19 日在 @PyTorch Conference 举办的 🌟Hugging Face Party🌟！M...</li><li><a href="https://huggingface.co/organizations/HF-Party/share/LmPGIYKDiiYvPOUoAPXUjdIAXskWeRSKMk)">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li></ul></div>

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1280241580917264464)** (243 条消息🔥🔥):

> - `Hugging Face API 与模型使用`
> - `模型性能与训练`
> - `社区问题与调试`
> - `ChatGPT 发展与更新`
> - `内容创作与 AI 工具`

- **Hugging Face API 用于商业用途**：一位用户询问，如果模型允许，切换到 Hugging Face 的 Pro 计划是否允许在商业应用中使用推理 API。
  - 澄清指出，对于他们的需求，Inference Endpoints 可能更高效且更具成本效益，同时也提到了对免费使用和速率限制（rate limits）的担忧。
- **T4 与 L4 GPU 上的模型性能**：讨论了在 T4 GPU 上运行 FLUX 模型的问题，用户担心是否需要切换到 L4 以获得最佳性能。
  - 社区成员的见解指出，该模型的大小（12B）可能会导致高资源需求，暗示了潜在的成本影响。
- **调试 Token Embeddings 问题**：一名成员报告修复了 Token Embeddings 尺寸不正确的问题，该问题最初导致了性能故障。
  - 这反映了社区在调试和改进模型配置方面的持续参与。
- **用户与 AI 系统的交互**：进行了一场关于创建一个 AI 支持的视频网站的轻松讨论，该网站将根据内容的“露骨程度”进行过滤，表达了对更好内容管理的需求。
  - 这个想法激发了利用 AI 技术改进社交媒体和内容审核的思考。
- **Hugging Face 的更新与观察**：用户对 Hugging Face 生态系统最近的更新或变化表示好奇，强调了新闻和开发公告方面的空白。
  - 推测包括由于近期缺乏沟通，用户参与度和社区动态可能发生的转变。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="http://localhost:8080" )```"="">未找到标题</a>：未找到描述</li><li><a href="https://blog.adnansiddiqi.me/">Adnan's Random bytes</a>：编程、生产力、创业和生活黑客</li><li><a href="https://distill.pub/2018/building-blocks/">The Building Blocks of Interpretability</a>：可解释性技术通常是孤立研究的。我们探索了当你将它们结合时产生的强大界面——以及这种组合空间的丰富结构。</li><li><a href="https://huggingface.co/docs/transformers/en/perf_infer_gpu_one">GPU inference</a>：未找到描述</li><li><a href="https://tenor.com/view/vine-so-no-head-angry-mad-throw-phone-gif-16162067">Vine So No Head GIF - Vine So No Head Angry - Discover &amp; Share GIFs</a>：点击查看 GIF</li></ul></div>

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1280263056957640755)** (5 条消息):

> - `FP8 与混合精度`
> - `使用 Meta Humans 创建 AI 化身`
> - `面向学生的 Perplexity AI Pro`
> - `发布 RAG 聊天机器人`
> - `FST-NLP`

- **FP8 基准测试成就**：成功训练了一个改进的 **FP8 - Bfloat16** **混合精度 (mixed precision)** 基准；在识别出导致 FP8 在梯度累加（gradient accumulation）期间出现 **NaN** 的问题后，解决了 **Loss** 匹配问题。
  - 这一改进在管理计算复杂性的同时，提高了训练模型的**效率**。
- **使用 Meta Humans 创建 AI 化身**：学习了通过 Epic Games 的 **Meta Humans** 构建 **AI 化身 (AI Avatars)**，方法是在 x86_64 上设置 **Unreal Engine 5.4**，关联 GitHub 账号后即可免费使用。
  - 这一资源为数字角色和沉浸式体验的进一步创意开发开辟了道路。
- **面向学生的免费 Perplexity AI Pro**：发现拥有 **.edu 邮箱** 的学生可以通过访问 [此链接](https://www.perplexity.ai/backtoschool) 注册免费一个月的 **Perplexity AI Pro**。
  - 此优惠**仅限两周**，是学生探索先进 AI 工具的绝佳机会。
- **准备部署 RAG 聊天机器人**：讨论了为部署 RAG 聊天机器人而自信地说出 **'ship it!'** 的准备工作，考虑了 **Docker/容器化** 以及可能的 **Google Cloud Run**。
  - 重点是在部署策略中平衡成本和创新架构。
- **探索 FST-NLP**：提到了 **FST-NLP**，表明了对自然语言处理进展的兴趣。
  - 这反映了对 NLP 技术及其影响的持续关注。

 

**提到的链接**：[Perplexity - Race to Infinity](https://www.perplexity.ai/backtoschool)：欢迎回到学校！仅限两周，领取由我们提供的一个月免费 Perplexity Pro。推荐你的朋友，如果你的学校达到 500 人注册，我们将把免费月份升级为一整年免费...

 

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1280288160927977596)** (7 messages):

> - `Negative Probabilities` (负概率)
> - `Hugging Face Blog Explorers`
> - `Firefox Tab Manager` (Firefox 标签页管理器)
> - `GitHub Contributions` (GitHub 贡献)

- **探索负概率 (Exploring Negative Probabilities)**：一位成员分享了一篇题为 [Negative Probability](https://arxiv.org/abs/2405.03043) 的论文，讨论了负概率在量子理论和 Bayesian 建模中的应用。
  - 文中提到，*当利率为负时*，某些分布中的负值存在相关性。
- **加入 Hugging Face Blog Explorers**：一位成员请求协助加入 [Hugging Face Blog Explorers](https://huggingface.co/blog-explorers)，并分享了他们最近关于 #autotrain 教程的 [GitHub PR](https://github.com/argilla-io/argilla/pull/5375)。
  - 另一位成员在审查请求后，鼓励他们*随时再次申请*以获得批准。
- **Firefox 标签页管理器增强**：一位成员介绍了一个 [Firefox 标签页管理插件](https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/)，支持垂直标签页并集成了历史记录和书签。
  - 该插件需要标签页、历史记录和书签权限，并强调书签不会在扩展程序内部被删除。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://arxiv.org/abs/2405.03043">Negative Probability</a>：负概率主要出现在量子理论和计算中。Bartlett 基于特征函数和非常规随机变量提供了一个定义。正如 Bartlett 所观察到的，负...</li><li><a href="https://addons.mozilla.org/en-US/firefox/addon/grasshopper-urls/">Grasshopper – 获取此 🦊 Firefox 扩展 (en-US)</a>：下载适用于 Firefox 的 Grasshopper。强大的标签页管理器。</li><li><a href="https://github.com/argilla-io/argilla/pull/5375.">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 发现、fork 并为超过 4.2 亿个项目做出贡献。</li></ul></div>

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1280289020189999206)** (14 messages🔥):

> - `Reinforcement Learning Algorithms Repository` (强化学习算法仓库)
> - `Health Insurance Appeal Bot` (医疗保险申诉机器人)
> - `Basalt Project Launch` (Basalt 项目发布)
> - `Data Transformation Tool` (数据转换工具)
> - `RAG System on Macbook` (Macbook 上的 RAG 系统)

- **Khashayar 的强化学习仓库备受关注**：一位成员分享了他们的 GitHub 仓库，用于实现基于 Sutton 和 Barto 经典著作的 **Reinforcement Learning Algorithms**，希望对他人有所帮助。仓库地址见 [此处](https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch)。
  - 该成员表达了对该项目的热情，旨在覆盖书中讨论的各种算法。
- **医疗保险申诉机器人现已上线！**：一位成员介绍了一个新工具，旨在帮助用户申诉医疗保险拒赔，访问地址为 [fighthealthinsurance.com](https://www.fighthealthinsurance.com/)。该工具使用 **OCR** 扫描拒赔信，并通过生成式 AI 生成潜在的申诉内容。
  - 反馈强调了遵守 **HIPAA** 法规以及确保数据使用透明度的重要性。
- **介绍 Basalt：下一代功能创建**：**Basalt** 项目正式发布，旨在为产品经理简化 AI 功能的创建和部署。感兴趣的用户可以通过链接的 [Typeform](https://www.linkedin.com/posts/marquis-guillaume_im-pleased-to-announce-the-launch-today-activity-7236642886371405825-Fm8x?utm_source=share&utm_medium=member_desktop) 访问并试用该项目。
  - 公告鼓励社区提供反馈，以提高参与度并完善工具。
- **使用 Cyyrus 转换您的数据**：一位成员分享了他们的项目 **Cyyrus**，这是一个将非结构化数据转换为适用于 Hugging Face 的可用数据集的工具。他们希望协助用户构建用于评估和微调等各种应用的数据集。
  - 该工具仍处于开发阶段，欢迎对其实用性提供反馈。
- **寻求本地 RAG 系统资源**：一位成员询问是否有人在 Macbook 上使用开源模型和资源本地创建了 **RAG system**。他们收到了一个指向 [LlamaIndex](https://pyimagesearch.com/2024/09/02/llamaindex-building-a-smarter-rag-based-chatbot/#:~:text=LlamaIndex%20provides%20a%20very%20seamless%20way) 的有用链接。
  - 随后讨论了新版 Mac 上的 CUDA 兼容性，反映了对优化本地设置性能的好奇。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.fighthealthinsurance.com/">挑战您的健康保险拒绝 —— 使用 AI 生成健康保险申诉</a>：未找到描述</li><li><a href="https://huggingface.co/blog/anakin87/spectrum">使用 Spectrum 对语言模型进行选择性微调</a>：未找到描述</li><li><a href="https://pyimagesearch.com/2024/09/02/llamaindex-building-a-smarter-rag-based-chatbot/#:~:text=LlamaIndex%20provides%20a%20very%20seamless%20way">LlamaIndex：构建更智能的基于 RAG 的聊天机器人 - PyImageSearch</a>：探索 LlamaIndex 如何通过更智能的索引和检索技术增强基于 RAG 的聊天机器人，从而获得更准确、更高效的响应。</li><li><a href="https://github.com/U-C4N/ImageWizard">GitHub - U-C4N/ImageWizard：ImageWizard 是一款现代 Web 应用程序，提供高级图像处理功能，如格式转换、压缩、像素化、ASCII 艺术生成和背景移除。基于 Next.js、React 和 TypeScript 构建，为各种图像处理任务提供用户友好的界面。</a>：ImageWizard 是一款现代 Web 应用程序，提供高级图像处理功能，如格式转换、压缩、像素化、ASCII 艺术生成和背景移除。基于 Next...</li><li><a href="https://github.com/KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch">GitHub - KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch：强化学习算法的实现（源自 Sutton &amp; Barto 的《强化学习导论》）</a>：强化学习算法的实现（源自 Sutton &amp; Barto 的《强化学习导论》）- KhashayarRahimi/Reinforcement-Learning-Algorithms-From-Scratch</li><li><a href="https://github.com/mdabir1203/Modular-Rust-Learning">GitHub - mdabir1203/Modular-Rust-Learning：通过模块化项目学习 Rust 和 OOP</a>：通过模块化项目学习 Rust 和 OOP。通过在 GitHub 上创建账户为 mdabir1203/Modular-Rust-Learning 做出贡献。</li><li><a href="https://github.com/NotTheStallion/Re-shard_Safetensors">GitHub - NotTheStallion/Re-shard_Safetensors：此仓库帮助您了解 safetensors 的结构如何存储 LLM 的不同层，并重新分片/重新块化 safetensors 文件，即使它们无法放入 GPU 显存中。（无 Autoclass）</a>：此仓库帮助您了解 safetensors 的结构如何存储 LLM 的不同层，并重新分片/重新块化 safetensors 文件，即使它们无法放入 GPU...</li><li><a href="https://github.com/wizenheimer/cyyrus">GitHub - wizenheimer/cyyrus：将非结构化数据转换为可用数据集</a>：将非结构化数据转换为可用数据集。通过在 GitHub 上创建账户为 wizenheimer/cyyrus 做出贡献。</li></ul></div>

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1280241746948919317)** (4 条消息):

> - `Age of Empire II 的 CV 项目`
> - `LLM 在视觉任务中的局限性`
> - `游戏资产映射策略`
> - `动态游戏状态更新`

- **针对 AOE2 的创新 CV 项目**：一位成员提出了一个 **CV 项目**，旨在为 **Age of Empire II** 创建 AI 助手，重点关注长期和短期决策策略。
  - 他们的方案涉及将游戏资产映射到文本矩阵，使用 **SAM** 和 **YOLO** 等计算机视觉工具来检测游戏元素。
- **LLM 在视觉对象识别方面面临困难**：有人对 **最先进的 LLM**（如 ChatGPT-4）的局限性提出了担忧，这些模型通常无法对图像中的物体进行计数和定位。
  - 据指出，这些模型主要描述图像，而不是在坐标级别进行精确观察。
- **将游戏资产映射到文本矩阵**：拟议的策略涉及创建一个 **text_map**，在缩小游戏屏幕比例的同时，表示关键游戏资产及其移动。
  - 目标是通过为 LLM 使用基于文本的输入来增强计数和定位能力。
- **对单张快照游戏分析的担忧**：一位成员对仅从单张游戏快照中能推断出多少策略表示怀疑，因为地图非常庞大。
  - 他们建议捕获动态状态可能会提供更有意义的见解。
- **需要动态更新或游戏注入**：有人建议在游戏移动时保持文本矩阵的动态更新，或者直接向游戏注入信息。
  - 这突显了需要更全面的数据捕获，而不仅仅是依赖计算机视觉。

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1280245637773922437)** (12 messages🔥):

> - `Multi-shot vs Many-shot learning`
> - `使用 nomic-embed-text-v1.5 训练自定义模型`
> - `Hugging Face 推理端点错误`

- **澄清 Multi-shot 和 Many-shot 学习的区别**：讨论了 **few-shot**、**multi-shot** 和 **many-shot** 学习的定义，后两个术语存在混淆。
  - *一位参与者指出，通常术语包括 zero-shot、one-shot 和 few-shot，且都不涉及在训练期间更新权重。*
- **寻求自定义模型训练指导**：一位用户询问如何以 **nomic-embed-text-v1.5** 为基础，针对特定用例训练自定义模型。
  - *他们请求在训练流程方面获得指引，特别是通过私信方式。*
- **Hugging Face 推理端点遇到问题**：另一位用户报告了一个错误。
  - *他们不确定问题是源自 Hugging Face 还是 AWS，并寻求协助解决。*

 

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1280394652062715999)** (1 messages):

> - `Yolo Diffusion`
> - `图像掩码技术 (Image Masking Techniques)`
> - `计算机视觉 (Computer Vision)`
> - `VLM 训练`

- **Yolo Diffusion 已过时**：一位成员指出 **Yolo Diffusion** 是一种主要用于掩码和带掩码重绘（inpainting with masks）的老技术，并建议现在已有更好的方法。
  - 他们建议在 **computer vision** 频道咨询以获取最新的方法。
- **库存水平测量误区**：澄清了关于 Yolo Diffusion 的讨论与测量 **库存水平 (stock levels)** 无关。
  - 该成员强调需要对 **computer vision** 进行更专业的咨询。
- **训练 VLM 变得必要**：为了利用改进的图像处理技术，可能需要考虑 **训练视觉语言模型 (VLM)**。
  - 这一建议源于图像分析及其应用领域的不断演进。

 

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1280254461637300345)** (95 messages🔥🔥):

> - `LM Studio 模型管理`
> - `使用特定 GPU`
> - `测试的 Temperature 设置`
> - `访问多模型功能`
> - `文本转图像模型支持`

- **LM Studio 模型管理技巧**：告知用户，保存在不同文件夹中的模型无法直接加载，但可以在 LM Studio 的“My Models”视图中更改模型文件夹。
  - 要从另一个文件夹加载模型，需要按照 LM Studio 内特定的目录结构进行组织。
- **在 LM Studio 中使用特定 GPU**：一位用户遇到 LM Studio 无法识别其 GPU 的问题，引发了关于潜在故障排除步骤的询问。
  - 另一位用户建议检查 Developer Tab 中的 LM Runtimes 作为诊断步骤。
- **质量测试的 Temperature 设置**：用户正在讨论 LM Studio 中 Temperature 设置对于评估模型输出的重要性，特别强调了使用低设置进行质量评估。
  - 建议用户参考关于 LLM 中 Temperature 的初学者指南以进一步了解。
- **LM Studio 中的多模型功能**：围绕在独立的本地服务器端口上运行多个模型展开了讨论，对于如何通过当前 LM Studio 功能实现这一点，反馈不一。
  - 大多数用户确认在单个实例中加载多个模型是可行的，尽管自动更新端口可能会使运行独立实例变得复杂。
- **文本转图像模型支持咨询**：一位用户询问 LM Studio 是否支持文本生成图像，得到的确认是目前不支持。
  - 替代建议包括使用外部工具，如 ComfyUI 中支持的 Flux 1。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://lmstudio.ai/docs/text-embeddings">Text Embeddings | LM Studio</a>：文本嵌入是将文本表示为数值向量的一种方式。</li><li><a href="https://fastflux.ai/,">FastFLUX | 免费即时生成 FLUX 图像</a>：使用 FastFLUX 在毫秒内创建精美的 FLUX 图像。免费、快速且无需注册。图像生成由 Runware 提供支持。</li><li><a href="https://github.com/lmstudio-ai/lmstudio.js">GitHub - lmstudio-ai/lmstudio.js: LM Studio TypeScript SDK (公测 Alpha 版)</a>：LM Studio TypeScript SDK (公测 Alpha 版) - lmstudio-ai/lmstudio.js</li></ul></div>

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1280240839645200479)** (142 条消息🔥🔥):

> - `Apple Silicon Memory Bandwidth` (Apple Silicon 内存带宽)
> - `Needing Multiple GPUs for LLMs` (LLM 对多 GPU 的需求)
> - `Using Unsloth for Fine-tuning` (使用 Unsloth 进行微调)
> - `Performance of Older GPUs for LLMs` (旧款 GPU 在 LLM 上的性能)
> - `Cache Issues with OpenWebUI` (OpenWebUI 的缓存问题)

- **Apple Silicon 及其内存带宽限制**：虽然 Apple Silicon 拥有惊人的内存带宽，但由于与 GPU 相比访问受限，其在 CPU 推理方面仍存在局限，且在性能表现上存在显著的功耗差异。
  - M1 Max 宣称拥有 **400GB/s** 的内存带宽，但关于如何有效利用这一带宽的细节仍不明确。
- **LLM 的 GPU 资源意识**：一位成员计划将一台 **2015 Xeon 服务器**搭配多块 **1070 GPU** 用于 LLM，但讨论中对其因年代久远和规格导致的性能限制表示担忧。
  - 使用像 1070 这样的旧款 GPU 可能会扩展显存容量，但会牺牲速度；专家建议使用更新的型号以获得可用的性能。
- **使用 Unsloth 进行微调**：讨论转向使用 **Unsloth** 工具来微调 LLM，有迹象表明它可以在当前配置下工作，而无需彻底更换硬件。
  - 成员们指出，微调方法的进步可能使其在不购买高端设备的情况下变得可行，并引用了社区中的案例。
- **旧款 GPU 的性能预期**：成员们辩论了 **GT 1030** 和 **1070** 等旧款 GPU 在推理任务中的有效性，对 Token 生成速度的预期较低。
  - 虽然 GPU 具有优势，但相对于 CPU 推理的性能提升似乎很微小，且受到模型架构的影响。
- **OpenWebUI 的缓存问题**：一位用户报告 **OpenWebUI** 将过多数据预加载到缓存中，消耗了过量的 RAM，在 **192GB** 总量中占用了 **150GB**。
  - 这种意外行为引发了对 RAM 管理策略中潜在软件 Bug 或配置错误的担忧和讨论。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.anandtech.com/show/17024/apple-m1-max-performance-review/2">Apple's M1 Pro, M1 Max SoCs Investigated: New Performance and Efficiency Heights</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3.1, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/qnguyen3/chat-with-mlx">GitHub - qnguyen3/chat-with-mlx: An all-in-one LLMs Chat UI for Apple Silicon Mac using MLX Framework.</a>：基于 MLX 框架的 Apple Silicon Mac 全功能 LLM 聊天 UI - qnguyen3/chat-with-mlx</li><li><a href="https://github.com/mlx-chat/mlx-chat-app">GitHub - mlx-chat/mlx-chat-app: Chat with MLX is a high-performance macOS application that connects your local documents to a personalized large language model (LLM).</a>：Chat with MLX 是一款高性能 macOS 应用程序，可将您的本地文档连接到个性化大语言模型 (LLM) - mlx-chat/mlx-chat-app</li><li><a href="https://github.com/preternatural-explore/mlx-swift-chat?tab=readme-ov-file">GitHub - preternatural-explore/mlx-swift-chat: A multi-platform SwiftUI frontend for running local LLMs with Apple's MLX framework.</a>：一个多平台 SwiftUI 前端，用于通过 Apple 的 MLX 框架运行本地 LLM - preternatural-explore/mlx-swift-chat</li></ul></div>

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1280508786024382498)** (14 条消息🔥):

> - `LLM.int8() 论文`
> - `量化技术`
> - `涌现的离群特征 (Emergent outlier features)`
> - `动态 vs 静态量化`
> - `模型在量化上的性能表现`

- **量化中的涌现离群特征 (Emergent Outlier Features)**：对话围绕 [LLM.int8() 论文](https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/autograd/_functions.py#L318-L321) 中讨论的 **涌现离群特征** 展开，并引发了关于这些特征在 Llama 2 和 3 等较新 LLM 中相关性的疑问。
  - *有推测认为，训练改进和架构变化可能会减轻这些离群值的影响。*
- **量化方法的差异**：**Mobicham** 指出，激活值的静态量化可能会产生问题，而动态量化往往表现更好，并引用了 SmoothQuant 论文在大模型上的结果。
  - 他们提到，虽然静态量化会影响准确性，但大模型的权重通常更容易量化且不会产生显著损失。
- **离群值对权重量化的影响**：**Mobicham** 进行的测试表明，**激活值中的离群值**对 `W8A8` 性能有很大影响，而仅权重量化（weight-only quantization）受到的影响极小。
  - 他们认为，由于较旧的训练方案和架构，OPT/BLOOM 等模型可能受影响更大。
- **Hopper 支持与限制**：用户 **theultrainstinct** 注意到 bitsandbytes 中的 int8 在 **Hopper** 架构上不受支持，并对某些说法提出了质疑。
  - 他们参考了关于量化能力和阈值的[更多细节](https://arxiv.org/pdf/2405.20835v1)。
- **模型量化中的阈值选项**：**Theultrainstinct** 提到可以在量化中设置离群值阈值，从而跳过分解（decomposition）步骤，但警告说某些模型（如 OPT）对这种调整非常敏感。
  - 相比之下，**Llama 2/3** 和 **Mistral** 等模型在这些条件下的表现明显更好。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://gist.github.com/mobicham/d08684728660f1cafbce94e4e69f7576">outliers_impact_W8A8.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/bitsandbyte">BitsAndByte - 概览</a>：GitHub 是 BitsAndByte 构建软件的地方。</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/autograd/_functions.py#L318-L321">bitsandbytes/bitsandbytes/autograd/_functions.py</a>：通过 PyTorch 的 k-bit 量化实现可访问的大语言模型。- bitsandbytes-foundation/bitsandbytes</li></ul></div>

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1280517490597957694)** (34 条消息🔥):

> - `Triton Load Ordering` (Triton 加载顺序)
> - `Compiler Optimizations` (编译器优化)
> - `Performance Tweaks in Triton` (Triton 中的性能微调)
> - `Dummy Conditions in Loops` (循环中的哑条件)
> - `Lecture References` (课程参考)

- **Triton 加载顺序影响速度**：用户注意到在 **Triton** 中更改加载顺序会导致不同的加速效果，其中一位用户的加速比从 **1.89506** 提升到了 **2.440731**，具体取决于加载顺序。
  - 显著的速度差异引发了关于编译器如何处理加载停顿（load stalls）和指令调度（instruction scheduling）的问题。
- **编译器在重新排列加载方面的局限性**：讨论强调虽然 **Triton** 编译器可以移除不必要的加载，但它缺乏进行大规模指令重排的能力。
  - 这意味着开发者可能需要手动调整加载顺序以优化性能，这与典型的编译器预期相反。
- **循环中的哑条件可绕过错误**：观察到在循环中插入类似 `if(k < bound)` 的哑条件可以规避某些 **Triton** 错误。
  - 这引发了对 **Triton** 在循环结构中错误处理行为的进一步探究。
- **对 Triton 文档的兴趣**：一位用户提到了 **CUDA Mode** 系列中的 **Lecture 14**，以获取关于 **Triton** 的更多背景信息。
  - 尽管指导尚不明确，用户表示它仍然是理解 **Triton** 功能的有用资源。
- **调查加载顺序微调**：用户鼓励在 **Triton** 中手动尝试加载顺序，并指出这是确定性能差异的快速方法。
  - 这种实践方法可能有助于微调未来的 **Triton** kernel 以获得更好的效率。

**提及的链接**：[lectures/lecture_014/A_Practitioners_Guide_to_Triton.ipynb at main · cuda-mode/lectures](https://github.com/cuda-mode/lectures/blob/main/lecture_014/A_Practitioners_Guide_to_Triton.ipynb)：cuda-mode 课程材料。通过在 GitHub 上创建账户来为 cuda-mode/lectures 的开发做出贡献。

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1280291847360610335)** (1 条消息):

> - `App Development Efficiency` (应用开发效率)
> - `Performance Optimization` (性能优化)
> - `Torch Scaling Techniques` (Torch 缩放技术)

- **开发者优先考虑构建而非运行应用**：一位成员表示，他们通常在**构建**和**调试**应用上花费的时间比在生产环境中运行它们的时间更多。
  - *“我仍然希望我测试的模型运行得快”*，以避免在代码更改期间等待结果，这表明这是开发者中的普遍优先级。
- **直接使用 torch._scaled_mm 以提高速度**：为了提高测试效率，该成员认为使用 **torch._scaled_mm** 是在代码更改期间快速运行模型的最佳选择。
  - 他们假设其他以类似方式编码的人可能会同意这种性能优化策略。

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息):

iron_bound: [https://m.youtube.com/watch?v=RIkse0tJ0hE&t=1s](https://m.youtube.com/watch?v=RIkse0tJ0hE&t=1s)

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1280622783013060701)** (6 条消息):

> - `PMPP 与 Synchronization`
> - `Volta 中的 Independent Thread Scheduling`
> - `Warp-Synchronous Programming 的弃用`

- **理解 Thread Warps 中的 Synchronization**：一位用户对 PMPP 中关于 Barrier Synchronization 的矛盾表述提出疑问，指出关于每 block 32 个线程是否需要 `__syncthreads()` 的说法*不可能同时正确*。
  - 澄清指出 **PMPP 的表述对于较新的 NVIDIA 硬件是准确的**，而另一种说法反映了旧架构的实践。
- **Volta 对 Thread Scheduling 的改变**：讨论引用了 Robert_Crovella 的回答，解释了 Volta 引入了 **Independent Thread Scheduling**，这使得 **Warp-Synchronous Programming** 被弃用。
  - 这一变化允许开发者实现**细粒度 Synchronization**，而无需依赖早期架构的隐式行为。
- **Warp-Synchronous Programming 的技术转型**：一位用户指出，由于 Volta 的改进，之前依赖 Warp-Synchronous Programming 的方法论现在已经过时。
  - 重点已转向利用新架构能力的**显式 Synchronization 技术**。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://forums.developer.nvidia.com/t/32-thread-block-doesnt-need-syncthreads/1490/18">32 thread block doesn't need _syncthreads()?</a>：有趣的话题，我在 Ampere GPU 上尝试了一个归约求和（reduced sum）应用，每 block 使用 32 个线程，似乎仍然需要 syncthreads() __global__ void global_sum_reduce_kernel(float * arr, f...</li><li><a href="https://developer.nvidia.com/blog/inside-volta/">Inside Volta: The World’s Most Advanced Data Center GPU | NVIDIA Technical Blog</a>：今天在圣何塞举行的 2017 GPU 技术大会上，NVIDIA CEO 黄仁勋宣布了全新的 NVIDIA Tesla V100，这是有史以来最先进的加速器。从语音识别到训练...</li></ul></div>

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1280352896050925580)** (13 条消息🔥):

> - `TorchAO 中的 RuntimeError`
> - `AWQ w4a16 CUDA kernel 移植`
> - `MXLinear 类错误实现`

- **TorchAO 中 quant_llm_linear 的 RuntimeError**：一位用户报告了一个 `RuntimeError`，指出算子 `torchao::quant_llm_linear` 在特定文件位置已经注册了一个 fake 实现。
  - 另一位成员建议*重新安装 torchao*，提到他们在当天早些时候也遇到了类似的错误。
- **关于移植 AWQ w4a16 CUDA kernel 的讨论**：关于是否移植 AWQ w4a16 CUDA kernel 出现了疑问，一位成员不确定是否已有他人在处理。
  - 有成员建议考虑使用现有的 tinygemm kernel，但有人指出 *tinygemm kernel 使用浮点零（floating point zeros）*，这无法与 AWQ 配合使用。
- **MXLinear 类实现困惑**：一位寻求在 `MXLinear` 中实现方法的用户注意到实现中类型检查（特别是围绕 MXTensor 类型）可能存在混淆。
  - 他们后来意识到*权重和输入 Tensor 在调用 linear 函数之前都被转换为高精度*，从而解决了部分困惑。

 

---

### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1280670710221635585)** (1 条消息):

> - `Tensor Model Parallelism`
> - `GPU Memory 利用率`

- **用于生产级工作的 Tensor Model Parallelism**：一场关于 **Tensor Model Parallelism** 是否适用于生产级实现的讨论兴起，建议使用 **8 个 GPU** 可能是理想的选择。
  - 这种划分有助于达到合适的 **Shared Memory** (smem) 需求，以获得最佳性能。
- **GPU Memory 划分见解**：将模型计算划分到 **8 个 GPU** 上的想法被强调为实现合适 **Shared Memory** 大小的一种手段。
  - 这种方法可能在性能和资源分配方面带来优势，确保生产模型高效运行。

 

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1280549168363995206)** (8 messages🔥):

> - `Burnout management`
> - `CUDA job scarcity`
> - `Niche job dynamics`
> - `Triton and CUDA trends`
> - `OpenGL relevance`

- **应对倦怠陷阱**：成员们讨论了应对 **burnout**（职业倦怠）的策略，强调了在持续低迷的市场中保持动力的挑战。
  - *一位成员提到*，他们稍后会分享自己的见解，希望能引发富有成效的讨论。
- **CUDA 就业市场显得冷清**：对 **CUDA 职位稀缺** 的挫败感浮出水面，有评论指出一些面试虽然承诺提供学习机会，却排除了缺乏经验的候选人。
  - *另一位成员指出*，这为许多合格的人才设置了不公平的准入门槛。
- **小众职位的双刃剑**：一位成员评论道，**niche jobs**（小众职位）的申请者较少，但也显著限制了整体机会，在就业市场中形成了一种平衡博弈。
  - 这种观点引起了其他人的共鸣，引发了关于追求这些专业化角色的影响的讨论。
- **Triton 和 CUDA 引领潮流**：讨论转向了 **Triton** 和 **CUDA**，它们在当前技术趋势中表现突出，特别是在机器学习应用中。
  - *一位成员分享了一个 [Reddit 帖子](https://redd.it/1f7wumb) 的链接*，强调了它们在行业中的相关性。
- **OpenGL 意外的流行度**：**OpenGL** 框架在对话中出现，其流行程度令人惊讶，引发了关于其在当前机器学习项目中适用性的疑问。
  - 这一评论促使人们进一步探究其在开发者中保持热度的原因。

 

**提到的链接**：[Reddit - Dive into anything](https://redd.it/1f7wumb)：未找到描述

 

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1280282039177842740)** (4 messages):

> - `Activation Checkpointing`
> - `Memory Optimization`
> - `GELU/Layernorm Backward Pass`
> - `Pipeline Parallelism`
> - `FP8 Implementation`

- **Activation Checkpointing 的突破**：一位成员成功用极少的代码实现了 **activation checkpointing**，展示了基于 **124M BF16** 模型在不同 batch size 下的内存需求。
  - 不同配置的内存占用包括：无内存复用时为 **1211 MiB**，100% 层重计算时为 **176 MiB**。
- **使用 GELU/Layernorm 节省内存**：在反向传播中重新计算 **GELU/Layernorm** 可有效减少内存需求，该操作在每层执行 **3 次**。
  - 这种方法导致了更低的 **内存占用**，在不显著增加复杂性的情况下提高了效率。
- **残差内存管理建议**：目前的实现总是为每一层保存 **residual3**，但优化这一点可以在增加复杂性的代价下获得更大的内存节省。
  - *一位成员建议*，将精细的残差管理与 **Pipeline Parallelism** 结合，可以更有效地利用 GPU 存储。
- **Pipeline Parallelism 的可行性**：该成员表示有信心实现 **Pipeline Parallelism**，虽然比 checkpointing 要求更高，但可能并不过于复杂。
  - 计划在完善现有功能后，优先实现 **FP8**。

 

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 messages):

anthonix_tm: 是的，我试过了。

---

### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1280626274947039256)** (2 messages):

> - `Second Wave of Responses`
> - `Third Wave of Responses`

- **第二波回复已发布**：**第二波回复**现已发布，表明在收集参与者反馈方面取得了进展。
  - *期待感增加*，成员们正在等待关于出席确认的更多细节。
- **潜在的第三波回复**：将根据确认出席的人数发布第三波回复。
  - 这种方法旨在确保反馈保持相关性，并能代表参与者的兴趣。

 

---

### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1280243121955012669)** (87 messages🔥🔥):

> - `CUDA kernel requirements`
> - `FP8 support`
> - `Model training issues`
> - `Liger-Kernel PR updates`
> - `CI/CD fixes`

- **CUDA Kernel 需要 SM_89 以支持 FP8**：讨论强调该 Kernel 需要 **SM_89** 或更高版本才能获得原生 **FP8** 支持，这影响了与 **A100** 等特定 GPU 的兼容性。
  - 成员指出，在 **4090** 上的测试显示性能比 **torch** 提升了高达 **1.3x**。
- **训练模型性能关注**：有人提出了关于使用 **Liger kernel** 配合 **DeepSpeed Zero3** 训练 **Qwen2 72B** 模型的问题，并指出了在显存占用和训练损失（loss）方面的挑战。
  - 建议包括通过禁用 **Liger** 功能来进行故障排除，以识别性能问题。
- **Liger-Kernel PR 更新**：最近的 PR 解决了冲突并引入了更新，包括一个向仓库添加 **pyproject.toml** 的拉取请求。
  - 有人呼吁解决 **CI** 冲突，成员之间进行了协作努力以确保顺利合并。
- **CI/CD 修复与改进**：成员们讨论了对 **CI/CD** 配置的必要更改，包括更新贡献指南以反映新的构建系统。
  - 分享了旨在修复 **CI** 问题的 PR，并鼓励合并和验证这些更改。
- **实验性功能与性能测试**：分享了对 **conv2d** kernel 和 **rms_norm** 中部分聚合（partial aggregation）的改进，表明这在性能上有益。
  - 参与者表示打算包含额外的基准测试，并进一步优化功能，重点关注 **flux** 模型。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/.github/workflows/publish-release.yml#L27">Liger-Kernel/.github/workflows/publish-release.yml at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/.github/workflows/publish-nightly.yml#L38">Liger-Kernel/.github/workflows/publish-nightly.yml at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/185">gemm fp8 e4m3 by AndreSlavescu · Pull Request #185 · linkedin/Liger-Kernel</a>：摘要：为 FP8 实现了具有 E4M3 表示的 FP8 gemm。Issue #65 已完成测试：测试了不同大小的方阵 (64, 256, 512, 1024, 2048) + 不同大小的非方阵 ...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/202/files">ci fix by AndreSlavescu · Pull Request #202 · linkedin/Liger-Kernel</a>：摘要：CI 修复。已完成测试：N/A。硬件类型：RTX 4090。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格，运行 make test-convergence 以确保收敛性。</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/e249eee723978bf8610ff1ea2297d048a2417e20/test/transformers/test_cross_entropy.py#L315">Liger-Kernel/test/transformers/test_cross_entropy.py at e249eee723978bf8610ff1ea2297d048a2417e20 · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/175">Monkeypatch for Qwen2-VL by tyler-romero · Pull Request #175 · linkedin/Liger-Kernel</a>：摘要：针对最近发布的 Qwen2-VL 的 Monkeypatch。HF transformers 建模代码：https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py F...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/182/files">Feat/faster rms norm by S1ro1 · Pull Request #182 · linkedin/Liger-Kernel</a>：摘要：在 rms_norm 中实现了部分聚合，类似于 #179 中描述的 layer_norm。已完成测试。硬件类型：运行 make test 以确保正确性，运行 make checkstyle ...</li><li><a href="https://github.com/linkedin/Liger-Kernel/blob/d338f4b9923e452baecff6d36775242a5319df4c/test/convergence/test_mini_models.py#L337">Liger-Kernel/test/convergence/test_mini_models.py at d338f4b9923e452baecff6d36775242a5319df4c · linkedin/Liger-Kernel</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/150">[BUILD] Add pyproject.toml by AndreSlavescu · Pull Request #150 · linkedin/Liger-Kernel</a>：摘要：添加了 pyproject.toml。已完成测试：运行 pip install -e . 并成功构建。硬件类型：RTX 3090。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格 ...</li></ul></div>

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1280244417885634680)** (145 条消息🔥🔥):

> - `Phishing concerns about a website`（关于某个网站的钓鱼风险担忧）
> - `Issues with ComfyUI and Stable Diffusion`（ComfyUI 与 Stable Diffusion 的问题）
> - `Usage of prompts in Stable Diffusion`（Stable Diffusion 中的 Prompt 使用）
> - `Stable Diffusion 3.1 updates`（Stable Diffusion 3.1 更新）
> - `Resources for training models and workflows`（模型训练与工作流资源）

- **钓鱼网站警示**：参与者对一个可疑网站表示担忧，指出由于其使用不安全的 HTTP 协议且数据传输未加密，该网站很可能是一个钓鱼中心。
  - *它看起来完全是非法的*，用户应避免在这些网站上分享个人信息。
- **ComfyUI 错误与模型混淆**：用户讨论了 ComfyUI 的问题，特别是关于缺少配置文件的错误，以及对某些模型是否已安装的误解。
  - 成员建议使用 *Save Text File* 节点来在 ComfyUI 中追踪 Prompt 和工作流。
- **Stable Diffusion 的 Prompt 结构**：在使用 Stable Diffusion 时，有人指出使用逗号分隔属性的 Prompt 结构通常效果更好，特别是对于 SD 1.5 等旧模型。
  - 然而，由于文本编码能力的提升，新模型更受益于使用自然语言 Prompt。
- **围绕 Stable Diffusion 3.1 的不确定性**：参与者推测了 Stable Diffusion 3.1 可能的发布，指出目前信息匮乏，且大多源自非官方渠道。
  - 社区呼吁大家保持耐心，等待 Stability AI 的官方公告。
- **模型训练资源**：用户表示需要针对特定角色和艺术风格训练 LoRA 模型的指导，表明对更新资源有需求。
  - 社区分享了一个 Flux 的 GitHub 仓库，这可能有助于理解与新模型功能相关的更新和工作流。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI (formerly StableSwarmUI), A Modular Stable Diffusion Web-User-Interface, with an emphasis on making powertools easily accessible, high performance, and extensibility.</a>：SwarmUI（原 StableSwarmUI），一个模块化的 Stable Diffusion Web-User-Interface，强调易用的高级工具、高性能和可扩展性。</li><li><a href="https://github.com/black-forest-labs/flux">GitHub - black-forest-labs/flux: Official inference repo for FLUX.1 models</a>：FLUX.1 模型的官方推理仓库。通过在 GitHub 上创建账号来为 black-forest-labs/flux 的开发做出贡献。</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1280625383015710721)** (104 messages🔥🔥):

> - `Mojo Standard Library`
> - `Modular CLI Updates`
> - `Magic CLI Introduction`
> - `MLIR and LLVM Integration`
> - `C++ and Haskell Interop Challenges`

- **Mojo 标准库部分开放贡献**：几位成员讨论了 **Mojo Standard Library**，指出部分内容已开放贡献，而其他部分仍与编译器紧密绑定。
  - 然而，**生产就绪版本尚未发布**，虽然存在稳定版本，但目前缺乏健壮的稳定性保证。
- **Modular CLI 接近最终更新**：**Modular CLI** 的更新动态表明，在过渡到 **Magic**（一个将集成包管理功能的新工具）之前，它已接近最后一个版本。
  - 团队目前正专注于 GPU 开发，这意味着仅限 CPU 的版本发布很快将告一段落。
- **Magic CLI 的打包方式类似于 Rust 的 Cargo**：**Magic CLI** 被提议利用 conda 封装器，旨在提供类似于 **Rust Cargo** 的更流线型包管理体验。
  - 成员们对能够避免像 **pip** 那样的环境管理陷阱表示兴奋，同时确保 C/C++ 依赖项更易于获取。
- **MLIR 作为提升语言互操作性的桥梁**：讨论集中在 **Clang 的 MLIR 后端** 在改善编程语言间互操作性方面的潜力，尽管在准确转换构造（constructs）方面存在挑战。
  - 共识是，虽然它简化了某些方面，但也引入了复杂性，特别是在向后兼容性和 C 预处理器方面。
- **Rust 在性能和 FFI 方面的优势**：Rust 被强调为处理需要速度的任务的有效内核语言，特别是在像 Haskell 这样的纯语言可能感到吃力的地方。
  - 对话指出，Haskell 库可以通过与 Rust 链接来获得性能提升，同时也承认了在不同语言之间建立共识的困难。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://github.com/llvm/clangir">GitHub - llvm/clangir: A new (MLIR based) high-level IR for clang.</a>：一个新的（基于 MLIR 的）Clang 高级 IR。通过在 GitHub 上创建账号来为 llvm/clangir 的开发做出贡献。</li><li><a href="https://docs.google.com/document/d/1zzZC6Kl7Le3Pd124aRb9uXr48APaWKhKBZeISm7s-qs/edit#heading=h.jyh6j2yblt83)">Magic🪄 + Conda Alpha Release Documentation</a>：Magic🪄 + Conda Alpha 发布文档介绍。我们很高兴地宣布在 Conda 上发布 MAX 的 Alpha 版本，以及我们名为 Magic 🪄 的新包管理器，它将取代 Modular CLI...</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1280241180394655876)** (24 条消息🔥):

> - `Passing Environment Arguments to Mojo Scripts` (向 Mojo 脚本传递环境变量)
> - `Destructor Automatic Calls in Mojo` (Mojo 中的析构函数自动调用)
> - `InlineFixedVector Usage and Lifecycle` (InlineFixedVector 的用法与生命周期)
> - `Weak Reference for Arc` (Arc 的 Weak 引用)
> - `MaybeUninit Alternatives` (MaybeUninit 的替代方案)

- **向 Mojo 脚本传递环境变量**：根据 [Mojo CLI 文档](https://docs.modular.com/mojo/cli/run)，在执行脚本期间传递环境变量，请使用 `mojo run mojoScript.mojo '~/.config/' 2`。成员们讨论了 `sys.argv` 如何涵盖此用例的细节。
  - 一位成员建议尝试不同的命令格式，以观察参数是如何被处理的。
- **理解 Mojo 中的析构函数调用**：Mojo 采用了 **ASAP 销毁策略**，一旦对象不再需要就立即销毁，并立即调用 `__del__()` 析构函数。参考了 [Mojo 生命周期](https://docs.modular.com/mojo/manual/lifecycle/death) 文档以澄清此行为。
  - 成员们讨论了某些函数（如 `!pop.array`）是否需要手动销毁，这引发了不同的意见。
- **关于 InlineFixedVector 设计的疑虑**：讨论了 **InlineFixedVector** 选择使用内联方法而不是将该决定留给内联器（inliner）的设计选择；旧的编程实践被认为是可能的原因。一位成员推测，随着即将到来的变化，**InlineFixedVector** 可能会很快被逐步淘汰，取而代之的是更简单的数据结构。
  - 另一位成员提到，一旦阻塞性的编译器工作完成并允许在 List 中进行优化，改进就会到来。
- **关于 Arc 的 Weak 引用查询**：一位成员询问为 **Arc** 添加 `Weak` 引用是否有益，或者目前是否处于搁置状态。这一询问表明了在管理其需求的同时增强 **Arc** 功能的兴趣。
  - 还有关于 `kgen.variant` 以及它是否暗示在初始化或销毁时具有自动行为的讨论。
- **探索 MaybeUninit 的替代方案**：一位成员询问了在不使用字节切片转换（byte slice punning）等不安全方法的情况下，**MaybeUninit** 的替代表示。探索了在处理未初始化数据时保持安全性的建议。
  - 讨论反映了要避免对 **Arc** 中使用的类型提出过于宽泛的要求。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.modular.com/mojo/stdlib/sys/arg/argv">argv | Modular Docs</a>：argv() -&gt; VariadicList[StringRef]</li><li><a href="https://docs.modular.com/mojo/manual/lifecycle/death">Death of a value | Modular Docs</a>：关于 Mojo 何时以及如何销毁值的解释。</li></ul></div>

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1280522271597002864)** (9 条消息🔥):

> - `OSDI '21 Keynote`
> - `Generality of MAX`
> - `Memory Domain Communication`
> - `Compiler Enhancements for Hardware`
> - `Heterogeneous Compute`

- **OSDI '21 对 MAX 潜力的关注**：来自 [OSDI '21](https://www.youtube.com/watch?v=36myc8wQhLo) 的一场富有洞察力的主题演讲解释了 MAX 如何在 AI 和 HPC 之外增强计算，强调了其优化硬件交互的能力。
  - 这表明 **Mojo + MAX** 可能能够有效利用各种处理器，从而在整个系统中最大化计算能力。
- **MAX 通用性的论据**：一位成员肯定了需要统一软件来解决现代异构计算（heterogeneous computing）的复杂性，并对 **Mojo + MAX** 的潜力表示信心。
  - 他们强调了防止供应商锁定（vendor lock-in）以及实现语言灵活性以更好利用现代硬件进步的必要性。
- **探索高级通信原语**：讨论了内存域（memory domains）之间雄心勃勃的通信原语的潜力，建议对传统通道进行改进。
  - 有人担心现有通道存在摩擦，质疑当前工作通信机制的效率。
- **内存域作为图节点**：建议将内存域表示为图节点，详细说明它们之间的各种链接及其特性（如延迟和带宽）。
  - 这种方法可以使硬件感知（hardware-aware）编译器比人工操作更有效地做出明智的数据移动和计算决策。
- **通道设计的未来**：一位成员表示打算开发基于 DPDK 的通道，因为其性能声誉，同时也承认通道带来的摩擦。
  - 然而，他们仍然认为通道在计算时间可变的环境中对于管理工作非常有价值。

**提到的链接**：[ASPLOS 2021 - Golden Age of Compilers](https://docs.google.com/presentation/d/1ZMtzT6nmfvNOlIaHRzdaXpFeaAklcT7DvfGjhgpzcxk/edit#slide=id.p)：硬件/软件协同设计时代的编译器黄金时代，Chris Lattner，SiFive Inc，2021年4月19日，编程语言和操作系统的架构支持国际会议...

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1280250187327078430)** (108 条消息🔥🔥):

> - `AI and Content Quality`
> - `Job Applications and AI`
> - `LAION Dataset Availability`
> - `AI as a Creativity Tool`
> - `Concerns about AI-generated Content`

- **关于 AI 对内容质量影响的辩论**：有观点认为，AI 工具的兴起可能会导致低质量、标题党内容的增加，一些人认为这降低了互联网的整体质量。
  - 相反，其他人断言 AI 生成内容之间的质量竞争将推动更高的标准，从而提高内容的相关性和准确性。
- **AI 在求职申请中的使用**：讨论强调了个人如何利用 AI 制作定制化的求职简历，而招聘人员则使用 AI 工具进行高效评估。
  - 这引发了对可能出现“无人工参与（no human in the loop）”场景的担忧，以及对招聘过程中评估质量的影响。
- **LAION 数据集的状态**：讨论了 LAION 数据集在之前因内容担忧被移除后的可用性。
  - 参与者确认该数据集已可以再次访问，并提到与 Clip 检索 API 等工具的集成将很快更新。
- **AI 作为创意增强器**：一位成员提出 AI 可以充当“创意倍增器”，熟练用户可以通过 AI 工具显著提高生产力。
  - 然而，其他人担心 AI 的滥用，导致创意领域低价值内容的泛滥。
- **对 AI 和虚假信息的担忧**：注意到大规模 AI 生成虚假信息的潜力，并担心其对选举等重大社会结果的影响。
  - 参与者讨论了技术进步的必要性，以过滤和评估 AI 生成内容的质量，从而减轻压倒性的错误信息。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://rom1504.github.io/clip-retrieval/">Clip front</a>：未找到描述</li><li><a href="https://aws.amazon.com/ec2/instance-types/inf2/">Compute – Amazon EC2 Inf2 instances – AWS</a>：未找到描述</li></ul></div>

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1280250504454209597)** (1 条消息):

> - `LLM-Based Autonomous Agents`
> - `Manifold Research Group`
> - `Research Log Updates`
> - `MultiNet Evaluation Metrics`
> - `Research Opportunities`

- **探索基于 LLM 的自主 Agent**：Manifold Research Group 发布了一篇题为 *Intelligent Digital Agents in the Era of Large Language Models* 的立场论文，提供了关于基于 LLM 的 AI Agent 进展及其类人决策能力的见解。感兴趣的参与者受邀加入 [Discord](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com) 参与讨论，并访问其 [官方网站](https://www.manifoldrg.com/llm-agents/) 进一步探索。
  - 该论文讨论了该研究领域的突破与局限性，并指出了未来的合作机会。
- **Research Log #042 亮点**：Manifold 最新的 [Research Log](https://www.manifoldrg.com/research-log-042/) 详细介绍了他们在 AI 项目上的每周进展以及 AI 社区的显著突破。这份持续更新的文档体现了该团队对开源 AI 透明度和创新的承诺。
  - 参与者可以查看分享的亮点，并加入与这些重要进展相关的持续讨论。
- **MultiNet 评估指标已定义**：Manifold 团队已成功定义了评估指标，计划用于对几种最先进的 Vision-Language Models (VLMs) 和 Vision-Language Applications (VLAs) 进行基准测试。相关细节可以在其 [GitHub 仓库](https://github.com/ManifoldRG/MultiNet?ref=manifoldrg.com) 中找到。
  - 关于详细的数据集覆盖范围，团队通过此 [链接](https://github.com/ManifoldRG/MultiNet/issues/19?ref=manifoldrg.com) 提供了深入见解。
- **开源团队机会**：Manifold Research Group 正在寻求能够通过各种研究项目和运营角色做出有意义贡献的个人，强调了他们对开源协作的承诺。感兴趣的候选人可以在其 [机会页面](https://www.manifoldrg.com/opportunities/) 了解更多信息。
  - OS 团队正在寻找充满热情的志愿者，建议申请人在申请前审阅 [OS Research Team Expectations](https://docs.google.com/document/d/e/2PACX-1vQgq32ChlP_e26mRPgfC31lZJCcAHAgbJ_Tn1nfzq8pfysoPAUqAWnel87Qc26h2Q/pub)。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>：B Faught, H Lu, T Marshall, H Sikka, P Guruprasad, B Gauri (2024)</li><li><a href="https://www.manifoldrg.com/research-log-042/">Research Log #042</a>：欢迎阅读 Research Log #042！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为来自更广泛研究社区的突破性成果...</li><li><a href="https://www.manifoldrg.com/opportunities/">Manifold Research Group (第 1 页)</a>：未找到描述</li></ul></div>

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1280250026802675796)** (12 条消息🔥):

> - `Manifold Research Group 的立场论文`
> - `Manifold 的算力可用性`
> - `ICLR 与 NIPS Workshop 发表的影响力对比`
> - `代码领域的 TinyStories 类比`

- **Manifold Research Group 分享近期论文**：来自 Manifold Research Group 的 Luke 介绍了他们关于 [LLM Based Autonomous Agents](https://www.manifoldrg.com/llm-agents/) 的立场论文，强调了该领域的关键进展。
  - 他们鼓励感兴趣的人加入其 [Discord 社区](https://discord.gg/MfYZmYEGaa?ref=manifoldrg.com)并查看其 [GitHub](https://github.com/ManifoldRG?ref=manifoldrg.com)。
- **Manifold 提供有限的算力**：Luke 确认 Manifold 作为各种学术和行业合作伙伴关系的一部分提供有限的算力，但具体情况取决于项目和团队。
  - 对于可用算力资源的详细咨询，建议直接联系 **Harsh** 或 **Sidh**。
- **ICLR 对简历的影响力高于 NIPS Workshop**：一位成员提到，在 **ICLR 主会**发表论文对简历的提升显著优于 **NIPS Workshop**，因为 Workshop 的录取标准较低。
  - ICLR 被公认为 **Tier 1 会议**，因此更具声望。
- **Linux 内核代码库作为代码界的 “TinyStories”**：针对关于类似 **TinyStories** 的代码资源提问，一位成员幽默地提到了 **Linux 内核代码库**。
  - 另一位成员建议了 **K&R**（Kernighan 和 Ritchie），可能指的是那本经典的计算机科学书籍，它也是编程知识的基石。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://www.manifoldrg.com/llm-agents/">Intelligent Digital Agents in the Era of Large Language Models</a>：B Faught, H Lu, T Marshall, H Sikka, P Guruprasad, B Gauri (2024)</li><li><a href="https://www.manifoldrg.com/research-log-042/">Research Log #042</a>：欢迎阅读 Research Log #042！我们记录了 Manifold Research Group 各项计划的每周研究进展，并重点介绍了我们认为来自更广泛研究社区的突破...</li><li><a href="https://www.manifoldrg.com/opportunities/">Manifold Research Group (第 1 页)</a>：未找到描述</li></ul></div>

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1280372624442134619)** (34 条消息🔥):

> - `对新概念的反馈`
> - `LLM 抽象-结晶 (Abstraction-Crystallization)`
> - `扩散模型与物理学`
> - `扩散模型中的时间步修改`
> - `使用 H100 GPU 进行 MoE 训练`

- **寻求对新概念的反馈**：一位成员表示有兴趣就他们一直在开发的一个新概念获取反馈，担心是否会令人反感。
  - 另一位成员鼓励他们分享，并安慰说这没有坏处，而且他们可能会了解到相关的现有工作。
- **LLM 缺乏抽象-结晶步骤**：有人提议 LLM 可以从一个允许它们评估多个抽象短语的步骤中受益，从而增强其输出潜力。
  - 该想法包括根据与 Prompt 的向量相似度对相关短语进行排序，这可能会产生更具创造性的回答，而不是仅仅依赖于概率最高的输出。
- **关于扩散模型理解物理学的担忧**：讨论围绕扩散模型是否能真正学习物理定律，还是仅仅是对数据集的过拟合展开。
  - 一位成员强调，强加物理结构可能会降低模型的表达能力 (Expressivity)，从而引发了对学习此类约束的担忧。
- **在扩散模型中随时间步修改权重**：有人推测是否存在使用时间步 (Timesteps) 修改扩散 U-Net 权重的工作，而不仅仅是调整输入。
  - 一位成员指出，扩散模型中典型的自适应归一化 (Adaptive Norms) 会根据时间步改变其缩放 (Scales) 和偏置 (Biases)。
- **MoE 训练与 H100 GPU 性能**：关于如何准确评估 H100 GPU 上 MoE 训练效率的问题出现了，特别是与稀疏操作 (Sparse Operations) 相关的问题。
  - 一位成员澄清说，H100 中的稀疏 Tensor Core 与 MoE 的稀疏性是不同的，并暗示营销宣传可能与实际收益不符。

 

**提到的链接**：[机器能学习数学结构吗？](https://pr4-kp.github.io/posts/machine-learning-sl2z/)：关于我上学期使用机器学习回答代数问题的研究工作的讨论

 

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1280240845853036585)** (31 条消息🔥):

> - `Transformers and Token Embeddings`
> - `MLP Layers in Transformers`
> - `Interpretability Across Training Checkpoints`
> - `Transformers as Graph Neural Networks`

- **理解 Transformers 中的 Token Embeddings**：成员们讨论了 Transformer 如何学习一个维度为 **VocabSize x EmbeddingDimension** 的向量，并断言每个 token 都有一个对应的 embedding。
  - **Attention heads** 是允许每个 token 通过在输入上生成 **QK softmax** 并将其与 token embeddings 相乘来影响其他 token 的关键。
- **MLP 层在组合 Token 信息中的作用**：Transformer 中的 MLP 扩展然后缩小 embedding 维度，但值得注意的是，它不会在不同 token 之间混合信息。
  - MLP 中的权重在 **token 之间共享**，这使得按 token 有效追踪神经元激活成为可能。
- **训练过程中神经元的可解释性**：有人提出了一个问题，即模型神经元的可解释性是否会在训练过程中发生变化，特别是对于 **Pythia** 模型中的不同 checkpoint。
  - 该假设认为可解释性可能会波动，由于叠加效应（superposition effects），可能呈现先低、后高、再降低的趋势。
- **Transformers 与图神经网络 (Graph Neural Networks) 的联系**：一位成员将 Transformer 块比作**图神经网络**，认为它在运行过程中会创建针对每句话的**邻接矩阵 (adjacency matrices)**。
  - 注意到 Attention 机制与节点-边图连接的相似性，特别是在 Attention 模式如何捕捉多跳关系（multi-hop relationships）方面。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/PonceLab/circuit_toolkit">GitHub - PonceLab/circuit_toolkit: Common utility functions for insilico experiments for visual neuroscience</a>：视觉神经科学计算机实验的常用工具函数 - PonceLab/circuit_toolkit</li><li><a href="https://transformer-circuits.pub/2021/framework/index.html">A Mathematical Framework for Transformer Circuits</a>：未找到描述</li></ul></div>

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1280442007663411215)** (2 条消息):

> - `lm-evaluation-harness issue`
> - `Maintainer response`

- **请求对 lm-evaluation-harness 问题的反馈**：一位成员请求维护者对 [此 issue](https://github.com/EleutherAI/lm-evaluation-harness/issues/2268) 提供反馈，以帮助推动项目进展。
  - 他们还表示如果适用，愿意进一步贡献代码。
- **维护者确认问题请求**：一位维护者做出了回应，感谢该成员提交 issue，并确认他们会进行查看。
  - 这表明了来自维护团队的积极参与和支持。

 

**提到的链接**：[Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/2268)：一个用于语言模型 few-shot 评估的框架。- Issues · EleutherAI/lm-evaluation-harness

 

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1280612868722528399)** (22 条消息🔥):

> - `PyTorch 与 CUDA 兼容性`
> - `Deepspeed 问题`
> - `模型代码库对比`
> - `训练配置`
> - `测试与合并功能`

- **排除 PyTorch 和 CUDA 故障**：成员们讨论了解决与 **PyTorch** **2.4** 版本和 **CUDA** 兼容性相关的问题，特别是专注于降级 **PyTorch** 以避免 **flash attention** 的安装问题。
  - 有建议称，安装与本地 **CUDA** 版本兼容的 **PyTorch wheel** 将解决安装问题，并分享了具体的参考链接。
- **确认 Deepspeed bug**：强调了一个与 **Deepspeed** 相关的已知 bug，包括一个 **GitHub** 链接，该链接提供了一个小的修复方案，以解决由 **Torch** 更改引起的 **import error**。
  - 一位成员确认解决了之前的 **import error**，但预计设置方面会有进一步的复杂性，表明合并可能会引入新问题。
- **预训练实现中的困难**：针对 **Nanotron** 和 **OLMO** 等各种预训练代码库提出了担忧，指出它们通常缺乏与替代 **Transformer** 和并行方案的兼容性。
  - 成员们表示，某些仓库仅支持基础实现，这激发了对具有不同位置编码的 **GPT-2 变体**的兴趣，并强调 **Neox** 是其中的佼佼者。
- **寻求训练配置方面的见解**：社区成员现在热衷于利用新获得的 **H100 GPU** 进行增强，而不仅仅是微调现有模型。
  - 讨论了新配置带来突破的可能性，并邀请成员分享他们的经验见解。
- **协作合并工作**：一位成员提出协助合并和测试新功能，强调了开发中协作的必要性。
  - 成员们鼓励分享发现以进行潜在的推广，展示了代码库创新和改进的互助氛围。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://pytorch.org/get-started/previous-versions/">Previous PyTorch Versions</a>：安装以前版本的 PyTorch</li><li><a href="https://github.com/microsoft/">Microsoft</a>：来自 Microsoft 的开源项目和示例。Microsoft 拥有 6357 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/5346">logger update with torch master changes by rogerxfeng8 · Pull Request #5346 · microsoft/DeepSpeed</a>：修复由 torch 上游清理 pytorch/pytorch@b6201a6 引起的 logger 导入问题的小修复。log 变量在 torch master 中被重命名。使用公共 API 创建 logger 以避免...</li><li><a href="https://github.com/microsoft/DeepSpeed/pull/5346/files">logger update with torch master changes by rogerxfeng8 · Pull Request #5346 · microsoft/DeepSpeed</a>：修复由 torch 上游清理 pytorch/pytorch@b6201a6 引起的 logger 导入问题的小修复。log 变量在 torch master 中被重命名。使用公共 API 创建 logger 以避免...</li></ul></div>

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1280625431774761010)** (2 条消息):

> - `面向学生的免费 Perplexity Pro`
> - `校园注册挑战赛`
> - `排行榜与激励措施`

- **学生可获赠一个月免费 Perplexity Pro**：学生在 **9 月 15 日**之前使用 .edu 邮箱注册，即可获得 **一个月免费 Perplexity Pro**。该服务提供快速、准确的回答，非常适合应对学术挑战。
  - Perplexity 提供的解决方案涵盖了从解释复杂话题到根据现有食材制定饮食计划等各个方面。
- **达到 500 人注册，全校即可获赠免费访问权限**：如果一个校区达到 **500 人注册**，全校学生都将免费获得 **一年的 Perplexity Pro**。鼓励参与者广而告之并邀请朋友加入，以实现这一目标。
  - 此活动截止至 **9 月 15 日**，当前的注册详情可以在[此处](https://www.perplexity.ai/backtoschool)进行追踪。
- **支持注册活动的视觉素材**：公告中包含了一些吸引人的视觉素材，用于推广免费试用月和注册挑战。这种创意方式旨在提高用户的兴趣和参与度。
  - 这些视觉素材强调了兴奋感和竞争感，旨在激励学生利用这一优惠。

 

**提到的链接**：[Perplexity - Race to Infinity](https://www.perplexity.ai/backtoschool)：欢迎回到学校！在短短两周内，即可兑换我们赠送的一个月免费 Perplexity Pro。推荐你的朋友，因为如果你的学校达到 500 人注册，我们将把那个免费月升级为一整年的免费...

 

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1280257905169272954)** (87 条消息🔥🔥):

> - `Perplexity Pro 共享选项`
> - `Copilot 品牌重塑`
> - `Xfinity Pro 订阅`
> - `学生折扣`
> - `Pro 的使用问题`

- **Perplexity Pro 共享选项**：成员们询问了是否可以与家人共享 Perplexity Pro 订阅，但目前尚无可用的共享选项。
  - *考虑在社区频道中建议改进*，因为目前还没有现成的家庭共享选项。
- **Copilot 转型**：一位用户对启用 Copilot 感到困惑，该功能已更名为“Pro”，这导致了一些误解。
  - 针对名称变更进行了澄清，目前没有特定的对应激活选项。
- **Xfinity Pro 订阅福利**：提到通过 Xfinity 注册 Pro 的用户可以使用代码获得额外的使用次数，这暗示了促销优惠。
  - 一位用户确认他们能够多次使用促销代码，从而获得了更大的灵活性。
- **学生折扣的差异**：多位用户对学生折扣的有限可用性表示沮丧，质疑为什么它主要适用于美国学校。
  - 参与者分享了因地区电子邮件域名而未收到优惠或不符合资格的经历，并主张包容性。
- **Pro 的使用问题**：一位成员报告在进行有限次数的搜索后遇到了付费墙，对免费访问和 Pro 访问之间的区别感到困惑。
  - 其他人也纷纷加入讨论，分享了类似的经历，并建议了诸如重新加入频道之类的故障排除方法。

 

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1280319517070725213)** (8 条消息🔥):

> - `Perplexity Xfinity 优惠`
> - `早晨习惯`
> - `DNA 计算开发领导者`
> - `Claude 驱动亚马逊 Alexa`
> - `后端之间的代理`

- **Perplexity Xfinity 优惠浮出水面**：分享了一个关于 [Perplexity Xfinity 优惠](https://www.perplexity.ai/search/perplexity-xfinity-deal-QCK.FX71SZCO6kSpE0YtYQ)的链接。相关细节可能会为用户揭示令人兴奋的产品或合作伙伴关系。
- **拆解早晨习惯**：一篇[新文章](https://www.perplexity.ai/search/what-is-the-morning-routine-fo-W691GG9LQEuBaar1MqxNHw)探讨了*什么是良好的早晨习惯*。这可以为高效的开启一天提供见解。
- **关于 DNA 计算开发的见解**：有人提出了关于谁在领导 *DNA 计算* 开发的问题，链接[在此](https://www.perplexity.ai/search/who-leads-development-in-dna-c-EKWUUF.BTUKLyifMrQcY9w)显示了详细的见解。这有助于加深对这一前沿领域的理解。
- **Claude 驱动亚马逊 Alexa**：一段引人入胜的视频显示 *亚马逊 Alexa 由 Claude 驱动*，在一个框架中同时探索了神经科学和计算 [链接](https://www.youtube.com/embed/oTWCM4aIA5g)。这揭示了 AI 与认知科学交汇处的进展。
- **后端之间使用代理**：[分享的链接](https://www.perplexity.ai/search/why-use-proxy-between-backend-iUaKITv2QsCuUI4V7It.JA)中提供了一场关于*为什么在后端之间使用代理*的讨论。这种理解对于构建高效的后端架构至关重要。

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1280500001511837759)** (3 条消息):

> - `Perplexity API 使用`
> - `文件上传功能`
> - `Make.com 集成`

- **通过 API 创建 Perplexity 页面**：一位用户询问了是否可以通过 API 创建 Perplexity 页面，特别是关于与 [Make.com](https://make.com) 集成的问题。
  - 另一位成员给出了否定回答，建议查看官方 [Perplexity 文档](https://docs.perplexity.ai)以获取更多信息。
- **pplx-api 中的文件上传支持**：一位用户询问 Pro API 在使用 CLI 界面时，是否允许在搜索查询的 payload 中上传文件（例如 .txt, .pdf）。
  - 该询问强调希望获得与 Web 界面相同的文件上传功能，以便进行更好的分析。

**提到的链接**：[未找到标题](https://docs.perplexity.ai)：未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1280547914141401108)** (1 条消息):

> - `Mistral 降价`

- **Mistral-Nemo 价格下调**：[Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) 的价格下降了 **23%**，反映了市场动态的变化。
  - 这一显著的价格变化可能预示着 **Mistral** 模型供需关系的转变。
- **市场对 Mistral-Nemo 降价的反应**：行业分析师正密切关注 [Mistral-Nemo](https://openrouter.ai/models/mistralai/mistral-nemo) **23%** 的降价，以了解其对竞争对手的影响。
  - 一些交易者认为，这可能会导致大量用户开始探索替代方案。

**提到的链接**：[Mistral Nemo - API、提供商、统计数据](https://openrouter.ai/models/mistralai/mistral-nemo)：由 Mistral 与 NVIDIA 合作构建的 12B 参数模型，具有 128k token 上下文长度。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1280553288378355722)** (2 条消息):

> - `Mume AI 应用发布`
> - `反馈请求`
> - `免费层级可用性`

- **Mume AI 应用兴奋登场**：**Mume AI** 应用（**Muse Mesh** 的缩写）已使用 OpenRouter 作为提供商正式发布，这标志着开发者在这个新兴领域取得了一个令人兴奋的里程碑。
  - 用户可以探索超过 **100 个模型**，这些模型提供文本和图像生成能力，以及 **Vision-enabled 模型**。
- **开发者鼓励社区反馈**：开发者表达了接收社区**反馈**以改进 Mume AI 的热情，并强调这只是众多里程碑的开始。
  - 开发者强调，由于该应用仍处于开发的**早期阶段**，每一条反馈都弥足珍贵。
- **免费层级提供每日 Token**：Mume AI 设有**免费层级**，每天为用户提供 Token，类似于开发者最初使用 OpenRouter 免费层级的体验。
  - 这一特性鼓励用户尝试该应用，同时也让更广泛的受众能够使用它。
- **跨平台可用性**：Mume AI 可在 **App Store** 和 **Play Store** 上获取，允许用户无缝下载和使用。
  - 该应用支持一系列功能，包括**多模态学习**以及通过各种模型类别生成创意内容。
- **用户友好的界面特性**：该应用拥有简洁的界面，支持根据用户系统主题定制的**亮色和暗色模式**，有助于保持对任务的专注。
  - 其组织结构允许用户按 **Marketing**（营销）、**Science**（科学）和 **Technology**（技术）等类别**探索**模型。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://apps.apple.com/us/app/mume-ai/id6523427150">‎Mume AI</a>：‎~ 通过聊天界面访问 100 多个模型，进行头脑风暴，获取创意灵感 ~ 通过支持图像识别的各种多模态模型从图像中学习 ~ 生成精美的图像...</li><li><a href="https://play.google.com/store/apps/details?id=ai.musemesh.mume">Mume AI - Google Play 上的应用</a>：未找到描述</li></ul></div>

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1280260741106106489)** (83 messages🔥🔥):

> - `Google 和 Claude 模型的 Caching 功能`
> - `OpenRouter 中的多轮对话 (Multi-turn conversations)`
> - `AI 模型中的角色一致性 (Character consistency)`
> - `在 Cursor 和 ContinueDev 中使用 OpenRouter`
> - `误扣费的退款申请`

- **Google 和 Claude 模型的 Caching 能力**：成员们讨论了通过 OpenRouter 为 **Google** 和 **Claude** 模型提供 **Caching** 的可能性，有迹象表明该功能即将实现。
  - 然而，由于两个 Endpoint 不共享同一个 Cache，引发了关于 Cache 路由的担忧。
- **关于多轮对话支持的澄清**：一位用户询问了 OpenRouter 对 **多轮对话 (multi-turn conversations)** 的支持情况，这引发了关于必须重新发送整个聊天历史以维持连续性的讨论。
  - 回复指出，由于 LLM 是无状态的 (stateless)，用户需要在自己端处理这一环节。
- **AI 角色一致性的最佳模型**：一位用户寻求关于维持角色一致性的最佳模型建议，提到 **Midjourney** 的效果并不理想，而另一位用户建议将 **Segmind** 作为潜在解决方案。
  - 对话强调了创建 Instagram AI influencer 的愿望以及实现更可靠输出的方法。
- **在其他提供商中使用 OpenRouter 的挑战**：一位成员表达了在 **Cursor** 中使用 OpenRouter 的问题，指出 **Cursor** 出于隐私考虑要求所有请求都必须经过他们。
  - 其他咨询涉及在尝试将 **ContinueDev** 与 OpenRouter 结合使用时遇到的困难，相关文档提供了解决方案。
- **误扣费的退款申请**：一位用户在误充值 **$174** 后请求退款，表达了对该情况的焦虑。
  - 该请求凸显了在计费问题上提供清晰用户支持的必要性。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://docs.continue.dev/reference/Model%20Providers/openrouter">OpenRouter | Continue</a>：OpenRouter 是商业和开源模型的统一接口，让您可以以最优价格访问最佳模型。您可以在此处注册，在 keys 页面创建您的 API key，然后...</li><li><a href="https://openrouter.ai/docs/frameworks#using-openai-sdk">Frameworks | OpenRouter</a>：支持模型集成的 Frameworks</li><li><a href="https://www.langchain.com/">LangChain</a>：LangChain 的系列产品在开发者的开发旅程中的每一步提供支持。</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>：每一个 ChatGPT 的前端 GUI 客户端。欢迎在 GitHub 上通过创建账号为 billmei/every-chatgpt-gui 的开发做出贡献。</li></ul></div>

---

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1280594847895523411)** (1 messages):

> - `NousCon 活动`
> - `PyTorch Conference`
> - `旧金山 (San Francisco)`

- **NousCon 活动定于 9 月 18 日举行**：我们将在 **PyTorch Conference** 之后，于 **9 月 18 日**在**旧金山**举办 **NousCon** 活动。
  - 名额有限，更多详情请参阅[官方公告](https://x.com/NousResearch/status/1831032559477866754)和注册链接[此处](https://lu.ma/zlgp0ljd)。
- **NousCon 名额有限**：建议参与者注意 **NousCon** 活动**名额有限**，强调了尽早注册的必要性。
  - 参与者可以通过提供的[注册链接](https://lu.ma/zlgp0ljd)锁定席位。

 

**提到的链接**：[来自 Nous Research (@NousResearch) 的推文](https://x.com/NousResearch/status/1831032559477866754)：NousCon，9 月 18 日，旧金山，名额有限。[https://lu.ma/zlgp0ljd](https://lu.ma/zlgp0ljd)

 

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1280245978263191572)** (56 条消息🔥🔥):

> - `Hermes-3 训练效率`
> - `创作者中的性别比例`
> - `应对诈骗者的策略`
> - `'Nous' 的发音`
> - `Hermes 的美学`

- **Hermes-3 训练速度极快**：Hermes-3 的训练仅需 **4 分钟** 即可完成，引发了关于当前模型训练技术效率的讨论。
  - 成员们因这种惊人的效率开玩笑说这是在“速通训练”。
- **对 Hermes 创作者性别动态的好奇**：一位成员幽默地询问了 Hermes 创作者中的 **性别比例**，表现出对模型背后多样性的兴趣。
  - 这引发了一场关于 AI 开发中代表性重要性的轻松讨论。
- **使用 Hermes 对抗诈骗者的创新方法**：一位成员提议使用 Hermes 来 **浪费诈骗者的时间**，建议它可以与诈骗者周旋而不暴露用户身份。
  - 这引发了关于对 Hermes 能让诈骗者纠缠多久进行基准测试（benchmarking）潜力的讨论。
- **关于如何发音 'Nous' 的见解**：社区讨论了 **'Nous' 的发音**，揭示了其语言根源的一些有趣细节。
  - 任何困惑都得到了澄清，一些成员还拿默音字母的含义开起了玩笑。
- **对 Hermes 美学的赞赏**：成员们对 Hermes 无与伦比的 **美学（aesthetics）** 表示赞叹，将其视觉效果归功于某位特定的创作者。
  - 这引发了更多对 Hermes 品牌整体设计和吸引力的赞扬和评论。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://tenor.com/view/luh-calm-fit-hazbff-opium-bird-stoon-gif-3957809579245532765">Luh Calm Fit Hazbff GIF - LUH CALM FIT HAZBFF OPIUM BIRD - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://developer.nvidia.com/blog/nvidia-sets-new-generative-ai-performance-and-scale-records-in-mlperf-training-v4-0/">NVIDIA 在 MLPerf Training v4.0 中创下生成式 AI 性能和规模新纪录 | NVIDIA 技术博客</a>：生成式 AI 模型有多种用途，例如帮助编写计算机代码、创作故事、作曲、生成图像、制作视频等。随着这些模型的持续发展...</li><li><a href="https://developer.nvidia.com/blog/nvidia-blackwell-platform-sets-new-llm-inference-records-in-mlperf-inference-v4-1/">NVIDIA Blackwell 平台在 MLPerf Inference v4.1 中创下 LLM 推理新纪录 | NVIDIA 技术博客</a>：大语言模型 (LLM) 推理是一项全栈挑战。强大的 GPU、高带宽的 GPU 到 GPU 互连、高效的加速库以及高度优化的推理...</li></ul></div>

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1280440022104805457)** (1 条消息):

> - `LLM 规划与推理`
> - `Yann LeCun 的概念`
> - `LLM-Modulo 架构`

- **寻求关于 LLM 规划与推理的见解**：一位成员询问了关于 **LLM 规划与推理** 的更新，表示很难找到能从核心解决该领域的卓越框架。
  - 他们指出，像 **Yann LeCun** 提出的概念看起来更现实，但仍缺乏解决基础 LLM 推理和规划挑战的全面方案。
- **对 LLM-Modulo 概念的担忧**：同一位成员评论说，**LLM-Modulo 概念** 在解决 LLM 推理和规划的关键方面似乎并不尽如人意。
  - 他们表达了希望与正在积极讨论或研究从根本上解决这些问题的架构的人士建立联系。

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1280490116829417535)** (2 条消息):

> - `Gemma 2 实现`
> - `Numpy 和 CuPy Notebooks`

- **介绍 Gemma 2：从 Numpy 到 CuPy 的过渡**：一位成员报告称正在尝试从零开始使用 **Numpy** 实现 **Gemma 2**，然后再将其移植到 **CuPy**。
  - 他们提供了两种实现的 Notebook 链接：[Numpy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final.ipynb) 和 [CuPy Notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy.ipynb)。
- **运行 CuPy Notebooks 的指南**：对于 **CuPy notebook**，建议使用具有 **24GB** 显存的 **GPU** 以获得最佳性能。
  - 另外，对于显存小于 **16GB** 的 **GPU**，用户应使用 [CuPy f16 notebook](https://github.com/githubpradeep/notebooks/blob/main/gemma2%20final_cupy_f16.ipynb)，而 **Numpy notebook** 适用于 CPU 运行。

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1280440022104805457)** (1 条消息):

> - `LLM Reasoning Frameworks`
> - `Yann LeCun's Concepts`
> - `LLM-Modulo Approach`
> - `Architecture for LLM Planning`

- **质疑 LLM 推理框架**：成员们表示，目前缺乏能够从核心层面有效解决 LLM **推理与规划（Reasoning and Planning）**问题的**卓越框架**。
  - *一位成员反思了对能够真正解决推理问题的概念的需求*，并引用 Yann LeCun 的观点认为其可能更具实际意义。
- **对 LLM-Modulo 的怀疑**：部分成员对 **LLM-Modulo 概念**持怀疑态度，表示该概念并未给他们留下深刻印象。
  - *人们对其有效性提出了担忧*，并呼吁就如何从根本上解决 LLM 推理和规划挑战展开讨论。
- **寻求 LLM 解决方案的合作**：成员们表达了与其他开发者建立联系的愿望，特别提到了关于 LLM 框架的合作机会。
  - *成员们表现出与该领域关键人物交流的兴趣*，以探索推理和规划方面的创新解决方案。

 

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1280285075535036598)** (31 条消息🔥):

> - `SearchGPT release speculation`
> - `AI in gaming`
> - `Simulation and consciousness`
> - `AI model performance`
> - `Community feedback on ChatGPT`

- **SearchGPT 发布热议**：一位用户推测 SearchGPT 可能很快发布，并指出一些加入候补名单的用户短暂看到了显示“You're in”的弹窗。然而，由于弹窗迅速消失，未能成功访问该服务。
  - 尽管充满期待，另一位用户认为 **Perplexity** 目前的表现优于 SearchGPT，且 **Arc** 已经集成了 Perplexity，使其成为更好的选择。
- **AI 玩 UNO 的视频**：一位成员建议制作一段 AI 玩 UNO 游戏的视频，并征求是否可行的见解。AI 在游戏领域的参与引发了关于创意应用的讨论。
  - 这展示了人们对 AI 主导的内容创作和交互体验的持续兴趣。
- **模拟的重新定义**：一位用户提出了“模拟（simulation）”的重新定义，强调观察者在解释体验中的意识作用。这将焦点从外部条件转向内部过程，尤其是在虚拟现实（Virtual Reality）等语境下。
  - 该用户向社区征求反馈，以评估这一哲学立场的清晰度和有效性。
- **对 ChatGPT 政策的挫败感**：一位成员对 ChatGPT 处理敏感话题的方式表示不满，注意到其响应模式的变化和消息删除现象。他们认为如果这些问题不解决，此类行为可能会导致用户流失。
  - 这一讨论凸显了用户对 AI 交互体验的持续关注，特别是围绕政策执行方面。
- **社区改进建议**：针对这些不满，一位用户建议其他人在专门的反馈频道中表达诉求，以实现有效的改变。这些言论强调了社区要求 AI 开发者提供更透明、更及时的支持。
  - 这表明 AI 提供商与用户之间在政策和服务改进方面迫切需要进行沟通。

 

**提及的链接**：[来自 Boris Power (@BorisMPower) 的推文](https://x.com/BorisMPower/status/1830714579116323004)：@Dr_Singularity 很抱歉我们让你失望了，感谢你的耐心——希望我们能尽快纠正这一点，让订阅变得更有价值。

 

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1280571814765924394)** (4 条消息):

> - `GPT-4o Features`
> - `ChatGPT File Saving Issues`

- **GPT-4o 表现优于 GPT-4 Turbo**：GPT-4o 比 GPT-4 Turbo **便宜 50%**，价格为 **$5/M input tokens 和 $15/M output tokens**，具有 **2 倍的速度**和 **5 倍的速率限制（rate limits）**，最高可达每分钟 **1000 万 tokens**。
  - 它的上下文窗口（context window）为 **128k**，与 GPT-4 Turbo 相比，它具有更卓越的**视觉能力**和**多语言支持**，使其成为用户的极佳选择。
- **ChatGPT 文件保存问题**：一位用户报告在尝试保存 ChatGPT 中的文件时遇到错误，系统提示在检索更新文本段落的下载链接时出现问题。
  - 尽管使用了 **plain txt** 格式，该用户仍面临障碍，这表明当前的文件服务可能存在中断或限制。用户对该功能的失效表示沮丧，因为此前曾成功保存过更长的文本。

 

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1280265229623754832)** (4 messages):

> - `休闲写作指令`
> - `正面与负面示例`
> - `行为主义与正向强化`
> - `处理写作中的禁忌`

- **休闲写作指令**：一位成员表示希望避免过于复杂或幽默的句子，敦促使用简单词汇的更休闲的风格。
  - 他们指出，回复中仍包含不想要的短语，强调了指令清晰度的必要性。
- **正面与负面示例**：另一位成员建议提供要使用的正面语言示例，而不是要避免的负面示例，重点关注所需的短语。
  - 这包括一份可接受术语的列表，以引导模型远离不理想的措辞。
- **行为主义与正向强化**：一位成员支持强调模型应该做什么而不是不应该做什么的想法，将其类比为行为技术。
  - 他们解释说，正向强化（positive reinforcement）比负向强化能带来更好的结果。
- **处理写作中的禁忌**：一位成员评论了撰写禁忌话题的复杂性，将其比作处理像镭（radium）这样的危险材料。
  - 他们强调在这种情况需要格外谨慎和考虑，以确保处理得当。

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1280265229623754832)** (4 messages):

> - `避免不想要的短语`
> - `指令中的正向强化`
> - `引导模型行为`

- **Flaskie 对指令清晰度的反馈**：一位成员对模型在有明确避免指令的情况下仍返回不想要的短语表示沮丧。
  - 他们主张更好地引导模型关注正面示例，而不是专注于负面示例。
- **正向指令的重要性**：另一位成员强调，指示模型做什么比指示它避免什么更有效。
  - 他们提供了行为学视角，建议正向强化能更有效地鼓励预期的结果。
- **对模型行为模式的担忧**：一位成员评论了模型倾向于重复上下文中见过的短语，即使这些短语是在要求避免它们的指令中出现的。
  - 他们通过一个关于模型对命令响应的例子说明了这一点，暗示了模型如何解释指令的一个根本性挑战。
- **对敏感话题的谨慎**：一位成员将处理敏感话题类比为管理像镭（radium）这样的危险材料，表明需要小心谨慎。
  - 他们暗示处理禁忌话题需要谨慎的方法，并承认其中涉及的复杂性。

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1280324241421500431)** (2 messages):

> - `自动文档检索`
> - `用于演示文稿生成的 LLM`

- **自动文档检索增强 RAG 效率**：最近的一个 notebook 展示了如何将 **RAG (Retrieval-Augmented Generation)** 与结构化查询相结合，以实现更好的文档检索，尤其是在处理大型数据集时，正如在[相关帖子](https://t.co/nfXnpvM93D)中所述。
  - *如何检索正确的文档？* 这种方法旨在有效解决该问题。
- **LLM 从笔记生成 PowerPoint 幻灯片**：一个创新的 TypeScript 应用允许用户将笔记转换为 **PowerPoint 幻灯片**，将他们从繁琐的结构化任务中解放出来，专注于创意，展示了 **LLM** 的力量。
  - 该应用不仅能将演讲笔记总结为幻灯片，还能生成额外内容，详见[演示链接](https://t.co/ItJ3edWmXF)。

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1280282216076935188)** (37 条消息🔥):

> - `Jina AI Late Embeddings`
> - `Gemini LLM 问题`
> - `在 ChatEngine 中过滤消息历史`
> - `关于 VectorStoreIndex 的问答`
> - `Tavily 工具的本地替代方案`

- **Jina AI Late Embeddings 类提案**：一位成员建议为 **Jina** 创建一个 Embeddings 类，以便通过 [HF](https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling) 利用新的 “late embeddings” 方法。另一位成员指出，大部分代码可能通过实现 BaseNodeParser 类集成到 node parser 包中。
- **Gemini LLM 面临初始化错误**：一位用户报告了在重启内核后与 **Gemini** LLM 相关的 **AttributeError**，并特别提到之前运行正常。建议更新依赖项，特别是由于最近的 **pydantic** 升级可能会导致与低版本的冲突。
- **为 LLM 查询过滤聊天消息历史**：一位成员询问如何在仅向 Chat Engine 发送问题之前，从消息历史中过滤掉答案。另一位成员建议，对 memory 进行子类化并重写 `get()` 方法可能是一个解决方案。
- **在 VectorStoreIndex 中通过 ID 获取节点文本**：一位成员询问在已知 **VectorStoreIndex** 中节点的文本和 ID 时，如何获取该节点的嵌入向量（embedded vector）。建议包括：如果 Embeddings 是在本地生成的，可以通过索引的内部结构访问嵌入数据。
- **RAG 的 Tavily 本地替代方案**：一位用户在参考 RAG 工作流的示例 Notebook 时，寻求 **Tavily** 工具的本地替代方案。回复明确了 Tavily 是一个网页搜索工具，需要使用 Google 或 Bing 等替代方案。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://llamahub.ai/l/readers/llama-index-readers-twitter?from=">未找到标题</a>：未找到描述</li><li><a href="https://llamahub.ai/l/readers/llama-index-readers-snscrape-twitter?from=">未找到标题</a>：未找到描述</li><li><a href="https://llamahub.ai/l/readers/]">未找到标题</a>：未找到描述</li><li><a href="https://github.com/run-llama/llamacloud-demo/blob/main/examples/advanced_rag/corrective_rag_workflow.ipynb">llamacloud-demo/examples/advanced_rag/corrective_rag_workflow.ipynb at main · run-llama/llamacloud-demo</a>：通过在 GitHub 上创建账号来为 run-llama/llamacloud-demo 的开发做出贡献。</li><li><a href="https://github.com/jina-ai/late-chunking/blob/main/chunked_pooling/chunking.py">late-chunking/chunked_pooling/chunking.py at main · jina-ai/late-chunking</a>：用于解释和评估 late chunking (chunked pooling) 的代码 - jina-ai/late-chunking</li><li><a href="https://github.com/jina-ai/late-chunking/tree/main/chunked_pooling">late-chunking/chunked_pooling at main · jina-ai/late-chunking</a>：用于解释和评估 late chunking (chunked pooling) 的代码 - jina-ai/late-chunking</li><li><a href="https://jina.ai/news/late-chunking-in-long-context-embedding-models/">长上下文 Embedding 模型中的 Late Chunking</a>：在保留上下文信息的同时对长文档进行分块具有挑战性。我们引入了 “Late Chunking”，它利用长上下文 Embedding 模型来生成上下文分块嵌入...</li></ul></div>

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1280461304083251201)** (14 messages🔥):

> - `H200 Pricing`
> - `H100 Demand Surge`
> - `Chat Template PR`
> - `GH200 Offer`
> - `KTO Performance`

- **H200 价格维持在 180k 高位**：一位成员指出，目前 **H200** 的 **8** 卡版本定价为 **180k**。
  - 这一状态引发了关于市场高需求如何影响定价的疑问。
- **H100 价格飙升与 Tesla 相关**：一位成员报告称近期 **H100** 显卡价格**大幅上涨**，暗示这与 **Tesla** 的活动有关。
  - 社区正关注来自 Tesla 等公司的持续需求将如何影响未来的市场趋势。
- **Chat Template PR 辅助设置**：一位成员强调了 **chat template PR** 的重要性，指出它允许自动加载 tokenizer 的模板。
  - 另一位成员表示，这一功能将显著简化设置流程。
- **GH200 报价为 45k**：一位成员提供了以 **45k** 获取 **GH200** 的交易，引发了关于当前定价的讨论。
  - 有趣的是，另一位成员表现出对显卡本身而非交易的偏好，凸显了对特定硬件的持续需求。
- **关于 KTO 性能的疑问**：一位成员询问了 **KTO** 在系统和多轮对话设置下的性能表现。
  - 社区似乎对了解 KTO 在这些条件下的运作方式有浓厚兴趣，并引发了成员们的响应。

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/)** (1 messages):

caseus_: 请为这项改进创建一个 issue

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1280389179724927008)** (22 messages🔥):

> - `Cross Entropy Loss in SFTT`
> - `Fine-tuning Axolotl on Multi-User Dialogues`
> - `Custom Templates for Multi-User Interaction`

- **SFTT 中的 Cross Entropy Loss 详解**：一位用户询问 SFTT 是否计算 **cross entropy loss**，另一位用户引导他们查看 **GitHub** 上 **LLaMA** 的建模代码以进行验证。
  - 此次讨论强调了定位正确代码库作为损失计算参考的重要性。
- **探索用于微调的多用户对话**：一位成员对在没有 **Agent** 的情况下使用**多人对话**微调模型表示好奇，并提出了如何格式化此类数据的问题。
  - *他们考虑是否可以通过将聊天历史作为 prompt 来训练模型理解对话流。*
- **建议使用自定义 Chat Templates**：另一位用户建议为多用户模拟自定义 **chat template**，而不是依赖传统的 user-agent 交互。
  - *这种方法突显了开发更具针对性数据集的潜力，因为目前的方法在处理多用户场景时似乎存在局限。*
- **播客转录面临的挑战**：一位用户指出，目前很难找到处理**播客转录**或涉及两个以上参与者对话的现有方法。
  - *这反映了一种普遍观点，即多用户数据集在当前的 AI 训练方法论中尚未得到充分讨论。*
- **对多用户数据集的兴趣**：几位成员对开发**多用户数据集**表示出兴趣，认识到它们在增强对话模型方面的潜力。
  - *他们承认虽然现有解决方案有限，但探索这些数据集可能会产生宝贵的见解。*

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1280342359208624138)** (12 messages🔥):

> - `Tools in Playground`
> - `LLM for Report Generation`
> - `Model Card Accuracy`

- **Playground 中的新模型已启用 Tools**：一位成员迫切希望在 playground 中尝试新模型的 tools，并得到了 **tools 现已启用**的确认。
  - *“祝开发愉快！”* 是团队成员的回应，鼓励进一步探索。
- **探索用于报告的 LLM**：有人咨询关于使用 **LLM** 根据之前的写作风格和会议记录生成报告的问题，旨在协助内部审计团队。
  - 成员们正在征集利用 LLM 实现这些目标的经验或见解。
- **Model Card 信息有误**：一位成员指出 [model card](https://huggingface.co/CohereForAI/c4ai-command-r-08-2024) 错误地标注模型大小为 **35B**，而实际应为 **32B**。
  - 团队承认了这一疏忽，并保证将进行更新。

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1280414887813582929)** (23 messages🔥):

> - `Server Side Events`
> - `Feature Request Submission`
> - `RAG JSON Output`
> - `Documentation Updates`

- **Cohere 支持 Server Side Events！**: 已确认通过向 Chat API 发送 `Accept: text/event-stream` 请求头，用户将接收到 **SSE events**。
  - *Billy* 正在更新文档以反映这一特性，该特性此前未在文档中说明。
- **Cohere 功能请求流程**: Frank 询问了关于提交支持 Server Side Events 的功能请求。
  - Sssandra 确认了反馈，并提到她将咨询产品团队以采取进一步行动。
- **RAG 的 JSON 输出限制**: 一位成员指出，**RAG** 目前不支持通过 `response_format` 输出 JSON，这一点可能并不明显。
  - Sssandra 对此反馈做出了回应，告知该问题已传达给团队，以便在未来的文档中考虑。

 

**提及的链接**: [Using server-sent events - Web APIs | MDN](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events): 开发使用 Server-Sent Events 的 Web 应用程序非常简单。你需要在服务器上编写少量代码来将事件流传输到前端，但客户端代码的工作方式几乎相同...

 

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1280682299746422916)** (1 messages):

> - `Command-R-Plus 08-2024 Issues`
> - `Web-Search Connector Behavior`

- **Command-R-Plus 08-2024 表现出不稳定性**: 从 **command-r-plus**（6月版本）过渡到 **command-r-plus-08-2024** 导致了不稳定的行为，例如在**极高的 Temperature** 下运行，导致输出充满无关内容。
  - *此问题仅在启用 web-search connector 时出现*，导致应用迅速达到 Max Tokens 限制并破坏了预期功能。
- **Web-Search Connector 加剧了输出问题**: 用户注意到，在使用 08-2024 版本时，**web-search connector** 是导致异常输出行为的关键。
  - 相比之下，6月版本在进行事实核查和在线研究时运行可靠，没有这些问题。

 

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1280303802242830438)** (12 messages🔥):

> - `Asistente Conversacional MultiAgente`
> - `Hybrid Retriever Implementation`
> - `Hugging Face Embedding`
> - `Normalization of Embeddings`
> - `Encode_kwargs Parameter`

- **寻求多 Agent 对话助手的指导**: 一位成员请求帮助编排 **多 Agent 对话助手 (Multi-Agent Conversational Assistant)**，对高级架构方法表现出兴趣。
  - 他们询问了关于 **Supervisor 架构**及其复杂性的经验。
- **Hybrid Retriever 概念**: 一位用户提到了创建 **Hybrid Retriever** 的可能性，即结合使用 **两个或更多 Retriever** 以获得更好性能。
  - 另一位成员表示赞同，简短地回复了“Cool”。
- **在 Hugging Face Embeddings 中传递 Encode_kwargs**: 一位成员讨论了使用 **Hugging Face Embedding 端点**，并就如何传递 **encode_kwargs**（如归一化）寻求建议。
  - 他们提供了一个示例代码片段来展示他们的实现尝试。
- **TEI 中的 Embedding 归一化**: 在得到建议后，一位成员确认 **TEI** 会自动归一化 Embedding，并澄清他们不需要指定 **encode_kwargs**。
  - 他们注意到其 Embedding 归一化检查返回为 **true**，确认 Embedding 已经过归一化。

<div class="linksMentioned"><p><strong>提及的链接</strong>:</p><ul><li><a href="http://localhost:8080" ,"="">未找到标题</a>: 未找到描述</li><li><a href="http://localhost:8080" )```"="">未找到标题</a>: 未找到描述</li></ul></div>

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1280514382836535297)** (2 条消息):

> - `Claude Sonnet 3.5 集成`
> - `Toolio 0.5.0 发布`
> - `LLM 结构化响应生成`
> - `文档聊天应用`
> - `类 OpenAI API`

- **使用 Claude Sonnet 3.5 与文档聊天**：一位开发者介绍了一个允许用户**与文档聊天**的工具，利用 **Claude Sonnet 3.5** 实现无缝交互，包括文件创建和编辑功能。
  - 他们指出，该工具目前仅处理**文本文件**，并存在一些局限性，可以通过 `.repoaiignore` 文件进行优化。
- **Toolio 0.5.0 发布，增强功能**：**Toolio 0.5.0** 版本发布，被称为“文本的胜利”，为这款专为 **Apple Silicon** 设计的 Python 工具包带来了改进的文档和更好的 Prompt 构建。
  - 显著更新包括符合 [JSON schema](https://json-schema.org/) 的结构化 LLM 响应生成，以及对更简便工具集成的支持。
- **大语言模型的结构化控制**：Toolio 旨在通过为开发者提供对文本生成的**细粒度控制**和结构化输出来克服 Large Language Models 带来的挑战。
  - 它的定位是为需要非临时性文本生成的开发者提供的关键工具，重点关注可靠的 Tool Calling 功能。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://repoai.dev">RepoAI</a>：未找到描述</li><li><a href="https://OoriData.github.io/Toolio/">Toolio—Mac 上 LLM 的结构化输出、Schema 控制响应和 Tool-calling</a>：未找到描述</li><li><a href="https://github.com/OoriData/Toolio/releases/tag/v0.5.0">Release 0.5.0 - 文本的胜利（文档、更好的 Prompting 等）· OoriData/Toolio</a>：添加了 llm_helper.debug_model_manager——一种提取原始 Prompt 和 Schema/Tool-call 信息的方法，用于调试底层 LLM 行为；除了 README 之外的文档（doc 文件夹）；测试用例；demo/algebra_tutor...</li><li><a href="https://pypi.org/project/Toolio/">Toolio</a>：类 OpenAI HTTP 服务器 API 实现，支持结构化 LLM 响应生成（例如使其符合 JSON schema）</li></ul></div>

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1280430357052784710)** (1 条消息):

> - `生成式 AI 项目`
> - `聊天机器人开发`

- **值得关注的生成式 AI 项目**：一位成员再次分享了他们今年的**生成式 AI 项目**，重点介绍了他们在 GitHub 上的工作，并发布了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/isham-rashik-5a547711b_github-di37dubai-tech-talk-intelligent-chatbot-activity-7236606074173222912-Fp2U) 敦促其他人探索这些项目。
  - 他们幽默地通过鼓励成员为他们的项目点亮 Star 来寻求支持。
- **推动项目参与**：在社区中，重点在于**参与**分享的项目，成员们表达了提供反馈和支持的兴趣。
  - 这种互动不仅促进了协作，还提高了该领域创新者的知名度。

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1280358622244638813)** (13 条消息🔥):

> - `Python PATH 问题`
> - `Open Interpreter 安装困扰`
> - `即将举行的 House Party 活动`

- **Python PATH 引起困惑**：一位成员在虚拟环境中使用 `pip install open-interpreter` 多次安装后，其 **Open Interpreter** 的 Python 脚本仍无法识别该模块，正面临困扰。
  
- **House Party 活动公告**：宣布了一场令人兴奋的 **House Party** 活动，承诺将发布重大新闻和演示，这可能是迄今为止最具影响力的活动。
  - 该活动将进行**直播**并录制，但鼓励大家参加以避免错过现场体验。

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1280578355992989849)** (2 条消息):

> - `Tool Use`
> - `嘉宾出席`

- **每周 Tool Use 推荐**：本周的 **Tool Use** 节目邀请了一位嘉宾，重点介绍了他们的见解和讨论。你可以在[这里](https://www.youtube.com/watch?v=UUUWt8GaS64)观看该集节目。
  - *感谢社区的支持*——经验的分享继续激发着围绕工具使用的讨论。
- **与嘉宾的愉快交谈**：成员们对在 Tool Use 环节中与新嘉宾聊天表示愉快。与嘉宾的贡献和互动丰富了正在进行的对话。
  - *一位成员分享了他们在对话中的喜悦*，为共同学习创造了一个包容的环境。

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1280306446902493215)** (1 条消息):

> - `Data Impact on Outcomes` (数据对结果的影响)
> - `Specific Dataset Inquiry` (特定数据集查询)

- **同一行数据影响结果**：一位成员确认，如果来自同一个 **sample**（样本），同一行中的所有数据点都会影响**最终结果**。
  - 他们询问是否正在分析特定的 **dataset**，表示对进一步细节感兴趣。
- **请求数据集详情**：该成员询问其他人是否在关注某个特定的 **dataset**，暗示了对数据问题的协作调查。
  - 这一询问强调了在分析语境中理解各种 datasets 如何相互作用的重要性。

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1280494869751533639)** (6 条消息):

> - `LoRA Fine-tuning Checkpoint Dictionary` (LoRA 微调 Checkpoint 字典)
> - `Llama 405B PR Changes` (Llama 405B PR 变更)
> - `Max Sequence Length Refactor` (Max Sequence Length 重构)

- **对 LoRA Checkpoint 字典的困惑**：一位成员对即使在设置了 `adapter_weights_only` 时仍使用完整的合并 adapter 权重构建 checkpoint 字典表示担忧，质疑其必要性。
  - 另一位成员澄清说，这一步骤在 Llama 405B 的 PR 中已被完全移除，但尚未在所有 recipes 中更新。
- **支持仅限 Adapter 权重**：一位成员支持在通用情况下应灵活支持将 `adapter_weights_only` 作为一个选项的观点。
  - 这表明大家达成共识，即改进微调配置选项可以增强易用性。
- **Max Sequence Length 似乎有了潜在解决方案**：一位成员对最近的 generation 更新表示兴奋，并提到了 `max_seq_len` 问题的潜在解决方案。
  - 他们表示有信心找到可行的解决方案，暗示将采取协作方式推进。
- **讨论了 max_seq_len 的重构草案**：分享了 `max_seq_len` 实现的重构草案，表明 GitHub 上正在进行相关开发。
  - 该成员承诺在明天的进一步讨论后，整理 Pull Request 上的文档。

<div class="linksMentioned"><p><strong>提到的链接</strong>：</p><ul><li><a href="https://github.com/pytorch/torchtune/pull/1449.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/70440446a4acf53e05cf7d74988fab21c8fd32e3/recipes/lora_finetune_single_device.py#L548).">torchtune/recipes/lora_finetune_single_device.py at 70440446a4acf53e05cf7d74988fab21c8fd32e3 · pytorch/torchtune</a>：一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li></ul></div>

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1280437614536691753)** (3 条消息):

> - `Leaderboard Updates` (排行榜更新)
> - `New Hermes Model` (新的 Hermes 模型)
> - `Model Requests` (模型请求)

- **对排行榜中遗漏模型的道歉**：团队承认在批量重新生成结果时遗漏了一个 **model**，并承诺在下次排行榜更新中将其重新加入。
  - 此次更新强调了他们致力于在排行榜上准确展示模型的承诺。
- **重心转移到 Hermes 模型的新数据集**：目前注意力集中在新的 **dataset release** 上，这导致处理新模型的请求被推迟到本周晚些时候或下周初。
  - 与此同时，鼓励听众为他们希望包含在排行榜中的模型提交 PR。
- **对澄清表示感谢**：一位成员对有关最近更新和模型管理的解释表示感谢。
  - 这反映了积极的社区参与和对查询的响应。

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1280435941617434634)** (4 messages):

> - `Chat Mode vs FC Mode` (Chat 模式 vs FC 模式)
> - `Leaderboard Differences` (排行榜差异)
> - `Issue Raising on GitHub` (在 GitHub 上提交 Issue)

- **Chat Mode 增加了解码的复杂性**：模型同时具有 **chat mode** 和 **FC mode**，FC mode 以结构化方式输出，便于解码，而 chat mode 输出普通文本消息，给解码带来了挑战。
  - **DEFAULT_SYSTEM_PROMPT** 在 chat mode 中实现，用于引导模型以结构化格式响应，从而辅助解码。
- **排行榜差异说明**：`leaderboard_live.html` 专门针对 **BFCL V2-Live 数据集**，而主排行榜 `leaderboard.html` 则包含了所有 **BFCL V2 数据集**（包括 Live 和非 Live 版本）。
  - 这种区分对于准确解读排行榜结果以及理解数据集的评估方式至关重要。
- **在 GitHub 上提交 Issue**：一名成员确认他们已针对排行榜差异在 GitHub 上提交了 Issue，并提供了 [Issue 链接](https://github.com/ShishirPatil/gorilla/issues/620)。
  - 他们还表示，如果与概述的问题一致，愿意提交 PR，展现了协作解决问题的积极态度。

**提及的链接**：[Issues · ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/issues/620)：Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - Issues · ShishirPatil/gorilla

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1280429307453505599)** (5 messages):

> - `Mini-Omni Voice Model` (Mini-Omni 语音模型)
> - `100k H100 Clusters Analysis` (10 万卡 H100 集群分析)

- **Mini-Omni 语音模型开源**：[Mini-Omni](https://hf.co/gpt-omni/mini-omni) 是一个开源的实时音频对话模型，可以同时生成文本和音频，并支持流式音频输出。
  - 该模型已在 Twitter 上分享，并附带了其 [代码库](https://github.com/gpt-omni/mini-omni) 和详细介绍其功能的研讨论文链接。
- **对 10 万卡 H100 集群的深入分析**：一份关于 **100,000 H100 集群** 的详细解释涵盖了功耗、网络拓扑以及 Ethernet 与 InfiniBand 的权衡等方案。
  - 讨论强调了自 **GPT-4** 以来 AI 能力感知上的停滞，这是由于单个模型的算力投入没有显著增加，尽管其他模型也拥有类似的 FLOP 指标。

<div class="linksMentioned"><p><strong>提及的链接</strong>：</p><ul><li><a href="https://www.semianalysis.com/p/100000-h100-clusters-power-network?triedRedirect=true">100k H100 Clusters: Power, Network Topology, Ethernet vs InfiniBand, Reliability, Failures, Checkpointing</a>：前沿模型扩展的挑战与需求、通过内存重构进行故障恢复、机架布局</li><li><a href="https://x.com/osanseviero/status/1830875530209513587?s=46">Omar Sanseviero (@osanseviero) 的推文</a>：Mini-Omni，一个开源实时音频对话模型 ⚡️ 实时对话式语音转语音 🤯 可同时生成文本和音频 🚀 流式音频输出 模型：https://hf.c...</li><li><a href="https://www.latent.space/p/fb3dd9ec-ec10-4155-876f-4cf0c9faf67a?postPreview=paid&amp;updated=2024-09-03T07%3A28%3A24.287Z&amp;audience=everyone&amp;free_preview=false&amp;freemail=true">Latent Space</a>：AI 工程师简报 + 美国前 10 的科技播客。探索 AI UX、Agents、Devtools、Infra、开源模型。查看 https://latent.space/about 了解 Chris Lattner、Andrej Karpathy 等人的精彩内容...</li></ul></div>

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 messages):

swyxio: 新播客上线！[https://x.com/latentspacepod/status/1831020483967701260](https://x.com/latentspacepod/status/1831020483967701260)

---

### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1280357758667063339)** (3 messages):

> - `WeaviateRM Integration` (WeaviateRM 集成)
> - `text2vec-ollama Discussion` (text2vec-ollama 讨论)

- **探索 WeaviateRM 集成**：一名成员表示有兴趣深入研究 **WeaviateRM 集成**，并请求在论坛上开启一个关于 **text2vec-ollama** 的 Issue。
  - 他们分享了 [Weaviate 论坛](https://forum.weaviate.io/latest) 的链接以供进一步讨论。
- **协作确认**：另一名成员确认愿意提供帮助，并同意开启论坛 Issue。
  - 对话在表达谢意中结束。

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1280292053560983552)** (1 条消息):

> - `COPRO 使用`
> - `Zero-shot 指令优化`

- **探索使用 COPRO 进行长度管理**：一位成员询问了关于使用 **COPRO** 或类似模型来有效优化指令长度的问题。
  - 他们建议检查调整 **max_tokens** 或实现指标返回系统是否能帮助管理指令长度。
- **Zero-shot 指令优化器技术**：讨论集中在使用 Zero-shot 指令优化器来引导模型内的指令长度。
  - 成员们辩论了设置长度限制是仅涉及限制 **max_tokens**，还是需要为指令和输入长度创建更复杂的指标。

 

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1280382689991725147)** (2 条消息):

> - `LLM 报告生成`
> - `会议记录作为输入`
> - `合成会议数据`
> - `用于会议摘要的 Text-to-Speech`
> - `Speaker-Diarization 训练`

- **探索使用 LLM 进行报告生成**：一位成员询问是否有人尝试过使用 **LLM** 根据之前的写作风格和来自不同利益相关者的会议记录来生成报告。
  - 这种方法旨在协助内部审计团队进行报告创建。
- **澄清会议记录**：另一位成员寻求对会议记录定义的澄清，建议它可能指包含与会者姓名的完整转录文本。
  - *你所说的会议记录具体指什么？* 引发了关于不同解读的讨论。
- **合成会议生成的见解**：一位用户讨论了他们使用 [persona-hub](https://github.com/tencent-ailab/persona-hub) 创建合成会议主题并模拟对话的工作。
  - 他们分享说，生成这些模拟涉及大量的 Token 使用，但为训练目的提供了一套多样化的会议数据。
- **音频和摘要技术**：对话内容包括计划使用 Text-to-Speech 模型为每个会议与会者生成音频，并使用 LLM 总结会议。
  - 讨论还涉及训练一个用于 **speaker-diarization** 的 **whisper 模型**，并开发一个与这些会议相关的特定 Text-to-Speech 模型。

 

**提到的链接**：[GitHub - tencent-ailab/persona-hub: Official repo for the paper "Scaling Synthetic Data Creation with 1,000,000,000 Personas"](https://github.com/tencent-ailab/persona-hub)：论文 "Scaling Synthetic Data Creation with 1,000,000,000 Personas" 的官方仓库 - tencent-ailab/persona-hub

 

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 条消息):

th.blitz: 你好 <a:LofiGirlWaveAnimated:927957453847556136>

---

---

---

---

---

---

{% else %}

> 完整的频道逐项分析已因邮件篇幅而截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}