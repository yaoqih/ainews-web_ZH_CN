---
companies:
- safe-superintelligence
- sakana-ai
- you-com
- perplexity-ai
- anthropic
- ai2
date: '2024-09-05T03:25:36.408094Z'
description: '以下是该文本的中文翻译：


  **Safe Superintelligence (SSI)** 以 **50 亿美元**的估值筹集了 **10 亿美元**，重点关注 Ilya Sutskever
  所暗示的安全和搜索方法。**Sakana AI** 完成了 **1 亿美元的 A 轮**融资，强调受自然启发的集体智能。**You.com** 在完成 **5000
  万美元的 B 轮**融资后，转型为类似 ChatGPT 的生产力智能体，而 **Perplexity AI** 在今年夏天筹集了超过 **2.5 亿美元**。**Anthropic**
  推出了企业版 Claude，具备 **5 亿 token 的上下文窗口**。**AI2** 发布了一个名为 OLMo 的 **64 专家混合专家 (MoE) 模型**，其性能超越了
  Llama2-13B-Chat。关键的 AI 研究趋势包括高效的 MoE 架构、AI 对齐的挑战、GPU 成本以及用于自主任务的新兴 AI 智能体。AI 开发的创新功能包括视频生成的命令与控制、检索增强生成
  (RAG) 的效率提升，以及 Anthropic 企业版计划中的 GitHub 集成。


  *“我们的标志旨在让人联想到鱼群聚集在一起，并根据简单的规则形成一个连贯实体的想法，因为我们希望在研究中利用自然界的灵感，如进化和集体智能。”*'
id: 10a98466-e7c6-4648-91ad-4a36d91165f3
models:
- olmo
- llama2-13b-chat
- claude
- claude-3.5-sonnet
original_slug: ainews-1150m-for-ssi-sakana-youcom-claude-500m
people:
- ilya-sutskever
- mervenoyann
- yuchenj_uw
- rohanpaul_ai
- ctojunior
- omarsar0
title: SSI、Sakana、You.com 获 11.5 亿美元融资 + Claude 支持 5 亿上下文。
topics:
- mixture-of-experts
- model-architecture
- model-training
- gpu-costs
- retrieval-augmented-generation
- video-generation
- ai-alignment
- enterprise-ai
- agentic-ai
- command-and-control
---

<!-- buttondown-editor-mode: plaintext -->**10 亿美元就是实现 Safe Superintelligence 的全部所需？**

> 2024年9月3日至9月4日的 AI 新闻。我们为您查阅了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务（**213** 个频道，**3131** 条消息）。预计节省阅读时间（以 200wpm 计算）：**340 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

更多无固定主题的新闻：

- **Safe Superintelligence**（[我们的报道见此](https://buttondown.email/ainews/archive/ainews-theres-ilya/)）宣布以 50 亿美元估值融资 10 亿美元。[Ilya 在路透社的报道中暗示了他们的搜索方法](https://www.reuters.com/technology/artificial-intelligence/openai-co-founder-sutskevers-new-safety-focused-ai-startup-ssi-raises-1-billion-2024-09-04/V)。
- **Sakana AI** [宣布了其 1 亿美元的 A 轮融资](https://sakana.ai/series-a/)，并进一步阐述了他们的方法：*“我们的 Logo 旨在唤起鱼群聚集并根据简单规则形成连贯实体的理念，因为我们希望在研究中利用自然界的思想，如进化和集体智能。”*
- **You.com** [宣布了 5000 万美元的 B 轮融资，并转向 ChatGPT 的产品形态](https://techcrunch.com/2024/09/04/you-com-refocuses-from-ai-search-to-deeper-productivity-agents-with-new-50m-round/) —— 实际上将 AI Search 领域让给了 Perplexity，后者在[今年春天融资 6300 万美元](https://x.com/AravSrinivas/status/1782784338238873769)后，又在[今年夏天融资超过 2.5 亿美元](https://techcrunch.com/2024/04/23/perplexity-is-raising-250m-at-2-point-5-3b-valuation-ai-search-sources-say/)。
- [Anthropic 发布了 Claude for Enterprise](https://x.com/alexalbert__/status/1831349257497895345?s=46)，具备 500m 上下文窗口。
- ChatGPT [从 Next.js 重写为 Remix](https://x.com/ryanflorence/status/1831379475654947233?s=46)
- [AI2 发布了 64 专家的 MoE](https://x.com/natolambert/status/1831353405585195121?s=46) 版 OLMo（[我们的报道见此](https://buttondown.com/ainews/archive/ainews-ai2-releases-olmo-the-4th-open-everything/)）


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

**AI 研发的关键趋势**

- **MoE 模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1831357563620753577) 介绍了 OLMoE，这是一个开源的 **Mixture-of-Experts Language Model**，拥有 1B 激活参数和 7B 总参数，在 5 万亿 token 上训练。据报道，它的表现优于具有相似激活参数的模型，包括 Llama2-13B-Chat。详情突出了涉及每层 64 个专家的创新架构，并专注于高效的训练技术。
- **AI 对齐的挑战**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1831350605063544980) 讨论了在昂贵的 GPU 环境下训练大模型的逻辑，并分享了关于模型上下文需求以实现最佳性能的见解，强调了高级 AI 任务不断增长的资源需求。提到了与 GPU 使用相关的具体成本，为 AI 研究的经济影响提供了实际的缩影。
- **新兴 AI 项目**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1831477875510956290) 描述了 AI Agent 领域的尖端进展，强调了向能够自主执行文档分析和技术图像生成等任务的项目转变。这些 Agent 根据用户定义的任务运行，展示了 AI 在实际企业应用中深度集成的趋势。

---

**AI 开发的创新工具与 API**

- **AI 中的指挥与控制**：[@ctojunior](https://twitter.com/cto_junior/status/1831303273074016384) 详细介绍了一种现代的 AI 驱动视频生成方法，该方法利用常规控制流水线并对其进行调整，以便更好地集成到生成模型中。这反映了在自动化系统中增强人类交互能力的更广泛趋势。
- **RAG 系统**：[@omarsar0](https://twitter.com/omarsar0/status/1831389521839267888) 提供了关于 Retrieval-Augmented Generation (RAG) 的见解，强调了其与长上下文模型相比的相关性。他们指出 RAG 在以更少的 token 产生更优结果方面的运行效率，表明这是未来应用的一个重要研究领域。
- **GitHub 集成**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1831475071250223411) 展示了在新的 Anthropic Enterprise 计划下，AI 应用向 GitHub 集成迈进的趋势，标志着在具有增强安全特性的协作编码环境中迈向运行效率的一步。

---

**AI 部署的行业影响**

- **医疗创新**：[@qdrant_engine](https://twitter.com/qdrant_engine/status/1831240904293728327) 推出了结合文本和图像数据以增强诊断能力的工具，反映了通过 AI 辅助进行的医疗工作流程的持续变革。多模态搜索的集成代表了旨在改善患者护理的关键进展。
- **教育推广**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1831354562000695483) 宣布了专注于 Python 编程的教育计划更新，强调了知识工作者对 AI 素养的需求。该倡议旨在加深在专业环境中对 AI 工具的理解和互动。
- **地缘政治维度**：来自 [@ylecun](https://twitter.com/ylecun/status/1831432018900492298) 的见解讨论了 AI 治理对不同系统中言论自由的更广泛影响，将技术论述与基本民主原则联系起来。这突显了在日益增长的 AI 时代进行深思熟虑的监管的必要性。

---

**AI 讨论中的幽默与迷因**

- **编码哀歌**：[@Aidan_mclau](https://twitter.com/aidan_mclau/status/1831390491456828142) 幽默地指出了程序员面临的常见挣扎，反思了当今软件开发中的荒诞与压力。这捕捉到了开发者在应对现代编码环境复杂性时的共鸣情绪。
- **创始人模式**：[@HamelHusain](https://twitter.com/HamelHusain/status/1831344024059334740) 定义了 "founder mode"，将对创业者的严苛要求与观察到的成功率和陷阱进行对比，对创业生活和预期产生了一种戏谑的氛围。
- **AI 趣事**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1831463052203716653) 对 AI 的现状及其超级智能的承诺进行了讽刺性评论，嘲讽了围绕 AI 能力的言论，同时对 AI 发展中的炒作与现实提供了视角。

---

# AI Reddit 摘要

## /r/LocalLlama 回顾

**主题 1. 新 AI 模型与前代模型的基准测试对比**

- **OLMoE - 一个仅有 10 亿激活参数的全开源稀疏 MoE 模型** ([Score: 161, Comments: 8](https://reddit.com//r/LocalLLaMA/comments/1f8lfb7/olmoe_a_fully_open_source_sparse_moe_with_only_1/)): OLMoE 是一种使用 **sparse Mixture-of-Experts** 的新型**开源语言模型**，拥有 **70 亿参数**，但每个输入 token 仅使用 **10 亿参数**，其性能超越了具有相似激活参数的模型，甚至超过了像 **Llama2-13B-Chat** 这样更大的模型。该模型在 **5 万亿 token** 上进行了预训练，并适配创建了 **OLMoE-1B-7B-Instruct**。这项工作的各个方面，包括模型权重、训练数据、代码和日志，都通过各种平台公开。
  - **OLMoE 的性能**在与 **Deepseek V2 Lite 16B MoE** 等较新模型对比时受到质疑。用户注意到了该模型的**开放性**，但也对微调期间 **MoE 训练速度优势**提出了担忧，理由是 **GPU utilization**（利用率）和 **loss stabilization**（损失稳定）方面的问题。
  - 该模型的 **7B 参数**总量和 **1B 激活参数**量因其作为**本地助手**的潜力而受到称赞。用户预计在**量化**后，**无需 GPU** 即可达到 **30-50 tokens/s**，非常适合笔记本电脑使用。
  - 社区表达了对 **GGUF support** 以及与 **llama.cpp** 集成的兴趣。一些用户正在等待 **GGUF 版本**以及与更现代模型的**基准测试**对比，以便进行公平比较。


**主题 2. Claude-Dev 扩展添加对本地 LLM 的支持**

- **Claude-Dev 现已支持本地 LLM！（Ollama，OpenAI 兼容服务器）** ([Score: 66, Comments: 13](https://reddit.com//r/LocalLLaMA/comments/1f8hfyh/claudedev_now_with_local_llm_support_ollama/)): **Claude-Dev 1.5.19 版本**已经发布，通过 **Ollama** 和 **OpenAI-compatible servers** 增加了对**本地语言模型**的支持。此更新可在 [GitHub](https://github.com/saoudrizwan/claude-dev/releases/tag/v1.5.19) 上获得，解决了社区长期以来的功能请求。
  - 用户对 **Claude-Dev** 与**本地语言模型**的兼容性表示兴奋，特别提到了 **deepseek coder v2** 的经济性和潜在性能。
  - 此次更新广受好评，用户期待尝试包括 **Gemini**、**GPT-4** 以及用于简单任务的免费本地选项在内的各种模型。
  - 社区成员对新的 **API support** 表示赞赏，表明这是一个备受期待的功能。


## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 与自主系统**

- **Minecraft 中的自主 Agent 文明**：一个[开创性的实验](https://www.reddit.com/r/singularity/comments/1f88z58/the_first_ever_agent_civilization_1000_truly/)，展示了 **1000 多个自主 AI Agent** 在 Minecraft 中创建了自己的文化、经济、宗教和政府。

- **特斯拉的 Actually Smart Summon (ASS)**：特斯拉[发布了改进版](https://www.reddit.com/r/singularity/comments/1f7zmno/tesla_launches_ass_actually_smart_summon/)的智能召唤功能，展示了自动驾驶汽车技术的进步。

**AI 图像生成与处理**

- **ComfyUI Advanced Live Portrait**：一个使用 ComfyUI 进行实时 AI 驱动的肖像生成和操作的[演示](https://www.reddit.com/r/StableDiffusion/comments/1f84qs6/comfyui_advanced_live_portrait/)。

- **Stable Diffusion 的改进文本编码器**：针对 Flux.1 的[新型 ViT-L/14 / CLIP-L Text Encoder 微调](https://www.reddit.com/r/StableDiffusion/comments/1f83d0t/new_vitl14_clipl_text_encoder_finetune_for_flux1/)，在图像生成中提供了增强的文本遵循度和细节。

**AI 发展与未来预测**

- **GPT-NEXT 发布公告**：OpenAI Japan [预告了 2024 年的 GPT-NEXT](https://www.reddit.com/r/singularity/comments/1f7y4z9/gptnext_in_2024_from_openai_japan/)，暗示了语言模型的潜在进步。

- **AI 进展视角**：一张[信息图](https://www.reddit.com/r/singularity/comments/1f7xds5/important_to_zoom_out/)强调了考虑长期 AI 进展而非仅仅关注短期发展的重要性。

**迷因与幽默**

- **GPT-Hype 迷因**：一个关于 GPT 模型发布及其相关炒作循环的[幽默解读](https://www.reddit.com/r/singularity/comments/1f7yodc/gpthype/)。

- **AI vs. 机器人推测**：一张[迷因风格的图片](https://www.reddit.com/r/singularity/comments/1f81ia6/what_do_you_think_chat_which_one_is_coming_to/)，对比了人形机器人与超人工智能可能到来的先后顺序。


---

# AI Discord 摘要回顾

> 由 Claude 3.5 Sonnet 生成的摘要之摘要的摘要


**1. LLM 进展与基准测试**

- **Llama 3 模型引发轰动**：Meta 的 **Llama 3 系列**模型（包括庞大的 **405B 参数**版本）已经发布，具备 **128k context windows** 和 **function calling** 等能力。
   - 这些模型已经开始部署，OpenRouter 以 **$2.5/mil tokens** 的竞争性价格[推出了 Llama 3.1-405B-instruct](https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers)。用户渴望在各种基准测试中测试其性能。
- **Command R 模型迎来更新**：Cohere 发布了更新的 **Command R** 和 **R+ 模型**，在推理、代码编写和多语言检索增强生成 (RAG) 任务中表现更佳。
   - 由于 GQA 增强，这些模型拥有高达 **50% 的吞吐量提升**，且[价格大幅下调](https://docs.cohere.com/changelog/command-gets-refreshed) —— Command R 现在的输入/输出 token 价格为 **$0.15/$0.60**，而 R+ 为 **$2.50/$10.00**。
  


**2. LLM 优化技术**

- **用于高效训练的低秩近似**：研究人员正在探索分布式训练设置中梯度传输的 **low-rank approximations**（低秩近似），以减少节点间的通信开销。
   - 这种方法与为大规模 AI 项目开发自适应通信模式的持续努力相一致，正如 [DiLoCo: Distributed Low-Communication Training of Language Models](https://arxiv.org/abs/2311.08105) 等论文中所讨论的那样。
- **动态专家路由增强模型灵活性**：**Dynamic Expert Routing** 的概念正受到关注，它允许模型在训练期间定义自己的专家，而不是使用固定配置。
   - 虽然在增强模型适应性方面很有前景，但成员们注意到该主题缺乏全面的文献，这标志着一个值得进一步研究和开发的领域。
  


**3. 开源 AI 进展**

- **Tinygrad 推出实惠的云服务**：**Tinygrad** 推出了价格仅为 **$60/月** 的云服务，提供 **4090 GPU** 和 **500 GB** 存储空间，定位比 **vast.ai 便宜 3 倍**。
   - 该服务允许用户在本地运行 tinygrad，同时利用更快的云端操作，正如在 [tweet](https://x.com/__tinygrad__/status/1829379908017238210?s=46) 中宣布的那样，每个 'TinyJit' 函数仅需 **一次往返**。
- **Re-LAION 5B 数据集解决安全顾虑**：**Re-LAION-5B 数据集**已[发布](https://laion.ai/blog/relaion-5b/)，更新了 LAION-5B，增加了额外的安全措施，并删除了指向疑似 CSAM 内容的链接。
   - 这一更新是与 **Internet Watch Foundation** 等机构合作开发的，旨在为 AI 研究和开发提供一个更合乎伦理且安全的数据集。
  


**4. AI 应用与行业影响**

- **GameNGen 实时模拟 DOOM**：**GameNGen** 神经模型展示了实时模拟游戏 **DOOM** 的能力，在单个 TPU 上实现了超过 **20 fps** 的高质量交互。
   - 人类评分者难以区分[模拟片段和真实游戏画面](https://gamengen.github.io/)，展示了神经模型在游戏开发和交互式媒体方面的潜力。
- **Meta AI 助手获得关注**：据报道，Meta 的 AI 助手月活跃用户已达 **4 亿**，日活跃用户达 **4000 万**，表明其在市场上被迅速采用。
   - 据 [The Information 报道](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx)，这一增长表明 Meta 在 AI 助手领域正在取得进展，尽管仍落后于 ChatGPT 据称的 **2 亿周活跃用户**。
  

---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **LLM 微调重获信任**：参与者打破了微调无法教授新概念的迷思，展示了依赖于正确参数和强大数据集的成功实现。
   - 挑战依然存在，一些人指出模型容易产生 **hallucinate**（幻觉），因此需要精心设计的微调方法。
- **RAG 与微调之争**：关于 **RAG** 与微调在减少幻觉方面效果的热烈讨论得出结论，混合方法可以实现利益最大化。
   - 参与者承认 **RAG** 在上下文锚定（context-grounding）方面的优势，建议灵活运用方法可能会产生更好的结果。
- **Llama 3.1 在 OpenRouter 上线**：新推出的 **Llama 3.1-405B-instruct** 模型具有显著的 **128k context**，价格极具竞争力，为 **$2.5/mil tokens**。
   - 该模型支持 **function calling**，迅速吸引了渴望利用其先进能力的用户的关注。
- **GPT-4o 降价**：OpenAI 的 **GPT-4o** 模型现在每 1M tokens 的成本为 **$4**，大幅降低了与 token 使用相关的费用，鼓励了更广泛的采用。
   - 凭借其 **Structured Outputs** 功能，**GPT-4o** 可以将响应与 **JSON Schemas** 对齐，将 **LLM** 定价策略转向由性能驱动的经济模式。
- **多 GPU 训练中的挑战**：成员们提出了多 **GPU** 设置中因 **GPU** 检测协议配置错误（特别是在脚本执行中）而遇到错误的问题。
   - 提交了一个 **pull request** 以改进与 **CUDA** 配置的兼容性，强调了优化代码处理 **GPU** 环境的需求。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 模型性能受到关注**：用户对新 **Gemini** 模型的性能表示怀疑，特别是它与 **Aider** 的兼容性。
   - 虽然有些人觉得它令人印象深刻，但对其在各种环境下的有效性仍存在担忧。
- **Sonnet 基准测试显示性能稳定**：[最近的评估](https://aider.chat/2024/08/26/sonnet-seems-fine.html)表明 **Sonnet** 保持了有效的代码编辑能力，基准测试结果稳定。
   - 尽管有传言，但性能统计数据揭示了在不同测试中一致的通过率。
- **Magic Dev 发布长短期记忆模型**：**Magic Dev** 推出了一个具有海量 **100M token context window** 的模型，通过推理增强了编程任务。
   - 这一进展因其在复杂问题解决任务中的潜在应用而引起了关注。
- **Aider 的发展路径与社区参与**：Paul G. 赞扬了社区在 **Aider** 演进中的参与，表示目前没有剧烈变动的计划。
   - 关于未来增长的讨论包括可能推出 **GUI** 版本以促进用户参与。
- **Aider 模型支持方面的困惑**：讨论强调了在使用 **OpenRouter API** 的 **Aider** 模型时，**.env** 文件中设置的混乱。
   - 关于未指定 **LLM Provider** 的错误引发了关于所需环境变量的讨论。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **个性化 LLM 以提升用户体验**：用户强调了**个性化 LLM** 对于创建独特个性并保持交互长期记忆的重要性。
   - 针对维持个性化体验所需的 **API** 调用成本影响，用户提出了担忧。
- **Grok 2 与 Gemini 的对决**：**Grok 2** 和 **Gemini** 之间展开了激烈的对比，突出了 **Grok** 的创造力，但在处理复杂任务时存在不一致性。
   - 用户分享了对 **Grok** 输出的挫败感，指出其结果随 **prompt** 质量的不同而有显著差异。
- **优化职位匹配分数**：解析了简历（CV）与职位描述对比中不平衡的相似度分数，识别出的范围在 **5 到 65** 之间。
   - 反馈建议重新校准 **prompt** 和分类规则，以提高评分的公平性和清晰度。
- **API 调用策略——独立 Prompt vs 单个 Prompt**：围绕针对不同问题使用多次 **API** 调用还是针对文档评估使用单个综合 **prompt** 展开了辩论。
   - 建议倾向于使用独立的调用以减少 **hallucinations**，从而增强响应的清晰度和可靠性。
- **通过 Batch Processing 增强文档分析**：聊天集中于利用 **batch processing**（批处理）作为简化大型文档分析并保持效率的策略。
   - 传阅了 OpenAI 的 **batch processing** 文档链接，激发了对高效数据提取技术的兴趣。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3 模型需要高性能硬件**：一位用户寻求构建 **RAG** 应用的帮助，询问 **LLaMA 3 模型**（包括 **8B**、**70B** 和 **405B 参数**版本）理想的**本地 GPU** 和 **RAM 配置**。
   - 回复建议使用 **Nvidia A100** GPU，运行 **LLaMA 405B** 至少需要 **300GB GPU 显存**，这引发了关于其成本和运营可行性的讨论。
- **Amazon ML Challenge 2024 寻找合作者**：一位成员正在为 **Amazon ML Challenge 2024** 寻找队友，旨在合作开展创新项目。
   - 未提供挑战赛的具体细节，这向更多热情的贡献者发出了加入的公开邀请。
- **探索 CAD 系统中的 AI**：讨论集中在 AI 与 **CAD** 系统的集成，引发了对类似于 **J.A.R.V.I.S** 功能的兴趣。
   - 成员们分享了他们在整合 AI 方面取得的进展，展示了增强交互式应用的现实潜力。
- **文本转语音 ML 的进展**：**Text-to-Speech-ML GitHub** 项目旨在通过社区协作改进文本转语音技术，欢迎用户贡献。
   - 这一举措标志着社区在语音合成相关的机器学习方面的进步，加强了开源贡献。
- **使用 AI 制作火球动画**：成员们讨论了在照片中制作**火球**效果动画的技术，推荐了 **AnimateDiff** 和 **IP Adapter Plus** 等工具。
   - 这种社区驱动的探索反映了通过各种创意技术，利用动画元素增强静态图像的集体努力。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **LTM 架构使用 RNN 处理 attention**：在简短的交流中，一位成员指出 LTM 架构似乎利用 **RNN** 来管理 attention。
- **理解 Triton 的 Atomic Add Scope 设置**：Triton 的 atomic add 设置中，`scope=GPU` 配置将操作限制在单个 GPU 内，而 `scope=system` 允许跨多 GPU 计算，这可能会影响性能。
   - 多 GPU 环境的默认设置是 `scope=GPU`，确保在无需额外配置的情况下实现功能。
- **FX pass 将 aten 操作映射到 Triton**：出现了一个关于创建 FX pass 以将 **aten** 操作直接映射到自定义 **Triton** kernel 的咨询，旨在进行性能优化。
   - 用户确认可以从 **PyTorch** 原生调用 Triton，无缝集成高级 GPU 加速。
- **Attention 层量化引发讨论**：成员们讨论了 attention 层中 **QKV 投影**量化的影响，强调了维持模型准确性的必要性。
   - 默认的 filter_fn 会自动量化 **Linear 层**，这引发了关于其操作假设的疑问。
- **v0.2.0 版本发布增强了 Liger-Kernel**：Liger-Kernel 的新版本 v0.2.0 提高了 API 清晰度并引入了更广泛的模型支持，但一些用户正面临 **Out Of Memory** (OOM) 错误。
   - 集成 LayerNorm 模块显示出良好的性能，尽管 Hugging Face 示例中的 OOM 问题仍然存在。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **优化 SDXL 性能**：为了提升 **SDXL 性能**，用户建议在 `webui-user.bat` 文件中添加 `--xformers`、`--medvram-sdxl` 和 `--no-half-vae`，特别是针对低 VRAM GPU。
   - 这些调整旨在提高速度并减少 VRAM 占用，同时不影响与 VAE 的兼容性。
- **澄清 SEG 实现**：关于工作流中 **SEG** 的讨论揭示了对其必要性和复杂性的困惑，特别是涉及 Impact Pack 等工具时。
   - 参与者质疑 SEG 是标准方法还是针对某些工具的专门功能。
- **AI 模型的高昂训练成本**：据报道，训练 **SD1.5** 或 **SDXL** 等基础模型需要数月时间，成本可能高达数百万美元，这引发了对资源分配的担忧。
   - 用户指出，与大型模型相比，**LORA** 模型的训练资源需求要少得多。
- **RunwayML 撤下 Stable Diffusion 仓库**：RunwayML 从 HuggingFace 等平台移除 **Stable Diffusion 1.5** 仓库的行为在社区内引起了警觉。
   - 此举暗示其重心可能正从早期模型转移，引发了用户对未来发展的猜测。
- **GPU 生成时间的辩论**：使用 **3060** 和 **3060 Ti** GPU 的用户分享了他们在 **SDXL** 和 **Flux** 模型上的生成时间经验，引发了对性能的关注。
   - 用户担心这些 GPU 是否能应对较长的生成时间以及相关的模型存储需求。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 3 在失忆模式下进行交流**：用户发现 **Hermes 3** 的失忆模式 (amnesia mode) 表现出对正式语言的偏好，拒绝使用像 'bruh' 这样被认为不友好的非正式词汇。这表明 AI 模型展现出明确交流风格的可能趋势。
   - 这一观察引发了关于 **AI 性格特征** 将如何塑造未来模型交互的问题。
- **低秩近似优化梯度传输**：讨论围绕使用 **low-rank approximations** 在分布式节点间进行高效梯度传输展开，这可能减轻通信开销。这种方法与训练中对自适应通信模式的需求相契合。
   - 成员们强调了在大规模 AI 项目中优化 **梯度性能** 以提高训练效率的重要性。
- **在多样化数据上训练 LLaMA 3**：一位用户正使用来自 Reddit 和 StackExchange 等来源的合成及真实指令数据训练 **8b LLaMA 3** 模型，旨在减少“AI 味”的行为。这展示了改进模型训练的多样化方法。
   - 这些努力可能会在多样化数据集如何影响 **AI 行为** 和基准测试方面产生重要发现。
- **为 LLM 引入 Word Game Bench**：**Word Game Bench** 是一个针对 LLM 的新型评估框架，专注于 **Wordle** 等互动游戏，以解决典型的评估缺陷。该基准测试标志着通过趣味交互评估 **模型性能** 的创新转变。
   - 成员们对该基准测试在提高语言模型评估准确性方面的潜在见解表示热烈期待。
- **GameNGen 实时呈现 DOOM**：神经模型 **GameNGen** 展示了脱离传统引擎独立模拟 **DOOM** 的能力，实现了超过 **每秒 20 帧** 的现实感。人类评分显示，很难将其模拟效果与实际游戏画面区分开来。
   - 关于该模型如何影响 **Unreal Engine** 等平台的讨论，强调了将此类 **模拟技术 (simulation technologies)** 集成到未来游戏中的前景。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 更新 0.3.2 提升性能**：最新的 LM Studio 更新 (0.3.2) 解决了 **Flash Attention** 的延迟问题，增强了本地推理性能。
   - *用户评价褒贬不一*，在注意到功能改进的同时，也对与早期版本相比的稳定性表示担忧。
- **Flash Attention 模型兼容性**：已确认 **LLaMa-3.1** 和 **Mistral** 支持 **Flash Attention**，讨论还扩展到了 Google 的 Gemma-2。
   - 这反映出用户渴望评估各种支持模型的整体性能水平。
- **M2 Ultra Mac 展示 LLM 潜力**：一位用户成功配置了一台拥有 **192 GB Unified Memory** 的 **M2 Ultra Mac**，目标直指 LLM 开发。
   - *Pydus* 对他在这台新硬件上能有效加载的模型大小表示好奇。
- **多 GPU 配置的电源管理**：一套包含 **4x RTX 4090s** 的配置经计算功率限制为 **3500W**，引发了关于电力分配的讨论。
   - 担忧集中在如何在不使断路器过载的情况下，安全地在多个插座之间分配电力。
- **来自 Llama 3.1 的 LLM 性能洞察**：**Llama 3.1** 的 70B 模型在多 GPU 配置上达到了 **每秒 97 个 tokens**，而之前的记录显示速度较慢。
   - 讨论集中在优化性能上，特别是在跨 GPU 分配模型层并确保高效利用方面。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini Flash 8B 模型发布**：新的 [Gemini Flash 8B](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp) 已与 [Gemini Flash Experiment](https://openrouter.ai/models/google/gemini-flash-1.5-exp) 一同上线，在 AI Studio 最终定价确定前，两者目前均可**免费**使用。
   - 此次发布是 Google 在 Google Vertex 与 AI Studio 分离后，增强其模型产品线和用户导航计划的一部分。
- **daun.ai 庆祝上线**：**daun.ai** 团队因成功上线而收到祝贺，这标志着社区中的一项重大成就。
   - 随着用户对这一里程碑的认可，聊天频道中充满了*欢呼与致意*。
- **Cohere Command 模型的激动人心更新**：**Command R** 模型的更新重构了接入点并更改了模型 ID，以提高运营效率。
   - 用户对这些更新特别热情，并提到了在价格和模型性能方面的益处。
- **Perplexity 模型遇到问题**：有用户报告了 Perplexity 模型的问题，收到了无效模型的错误——这源于之前已迅速处理的 bug。
   - 随着用户寻求了解影响性能的错误范围，对这些问题进行澄清是必要的。
- **基础设施升级导致近期停机**：近期的基础设施升级导致停机时间增加以及系统响应挑战。
   - 团队已承认这些问题，并将其归因于数据库限制以及正在进行的加强 Backend（后端）的项目。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **NaN 权重导致 embedding 训练中断**：一位用户报告称，尽管初始范围正常，但其 embedding 权重在训练仅几步后就变成了 **NaN**。通过检查梯度和损失组件，发现原因可能是 **data-dependent decay term**（数据相关的衰减项）。
   - *Lightning* 的 *detect_anomaly=True* 设置有助于根据梯度分析追踪问题。
- **社区寻求研究思路的反馈**：一位博士生就使用扩散模型进行压缩的研究寻求建议，并询问最佳分享方式。成员建议在 general 频道或特定区域分享，进行低压力的讨论。
   - 强调了对网络输入进行正则化损失以保持稳定性，并突出了在讨论 **Sparse Autoencoders (SAEs)** 时明确假设的重要性。
- **澄清 SAE 中的稀疏编码**：一位成员澄清说，在其 SAE 方法中，重建错误损失项应与关注稀疏性的损失并存，以避免训练期间的偏差。建议增加额外的损失项，以帮助获取来自冻结网络的统计上下文。
   - 指出对 **LLMs** 在 **SAEs** 中作用的误解对于编码过程至关重要。
- **Dynamic Expert Routing 增强灵活性**：在讨论中，有人解释说，允许模型在训练期间定义自己的专家，比固定配置更能提高适应性。对相关论文的请求显示，现有文献中存在空白。
   - 强调了对 **Dynamic Expert Routing** 概念需要更多资源。
- **用于模型评估的新 Word Game Bench**：社区推出了一个名为 **Word Game Bench** 的基准测试，旨在评估语言模型在 **Wordle** 等单词拼图游戏上的表现。值得注意的是，目前没有模型的平均胜率超过 **50%**。
   - 该基准测试鼓励模型进行交互和反馈，而不是依赖静态响应。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Discord 服务器成员突破 10 万**：**Discord server** 目前已达到 **100K 成员** 的里程碑，展示了社区的蓬勃发展。
   - 成员们对社区的支持表示感谢，强调团队渴望共同*继续成长和进化*。
- **Pro 订阅失效问题**：用户对 **Pro subscriptions** 消失表示担忧，这可能是由于误用优惠码或账号差异导致的。
   - 一位用户提到兑换的优惠码失效，引发了关于潜在凭证滥用措施的疑问。
- **模型性能受到审查**：讨论显示，在不同 AI 模型之间切换往往会产生类似的响应，引发了对更新后 **model differentiation**（模型差异化）的怀疑。
   - 一位用户指出，关于模型类型的查询返回的是通用信息，而非关于 *GPT 或 Claude* 的具体细节。
- **PPLX API 额度获取问题**：多位用户报告在购买 Pro 后未收到承诺的 **$5 PPLX API credits**，并寻求协助解决。
   - 支持团队缺乏解决方案，导致用户请求提供账号详情以进行进一步调查。
- **速率限制引发困惑**：一位用户在调用 API 端点时遇到了 **429 Client Error**，尽管其脚本中的函数调用极少。
   - 他们对过早触发速率限制表示担忧，并寻求对底层因素的澄清。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R+ 模型带来显著性能提升**：最近更新的 **Command R** 和 **R+** 模型（包括 `command-r-08-2024`）在推理、编码和多语言 **RAG** 方面表现出更强的性能，吞吐量提升高达 **50%**。
   - 此外，价格也进行了大幅调整：Command R 模型输入为 **$0.15**，输出为 **$0.60**；而 R+ 现在输入为 **$2.50**，输出为 **$10.00**（每百万 tokens）。
- **用户质疑 MMLU 的实际相关性**：Nick Farst 指出 **MMLU** 与现实世界的应用关联有限，因为其大部分内容已经过时。
   - 讨论反映了社区的共识，即优先考虑实际性能指标，而非 **MMLU** 等传统基准测试。
- **C4AI 学者计划引发关注**：有人询问在读研究生是否有资格参加 **C4AI Scholars Program**，特别是针对 1 月份的实习。
   - 成员建议直接联系 **C4AI** 以获取申请流程和后续机会的明确细节。
- **包含 400 万+ 条目的 Maya LLaVA-Pretrain 数据集发布**：**Maya LLaVA-Pretrain** 数据集现在拥有分布在 **8 种语言** 中的 **4,404,776** 个条目，旨在增强大型语言和视觉模型的预训练。
   - 访问该数据集需要同意共享条件，以确保尽管数据集是公开的，但仍符合使用政策。
- **试用 API Key 限制使用**：一位试用 API Key 用户遇到了速率限制（Error 429），每月仅允许 **1000 次 API 调用**，这凸显了升级到生产 Key 的必要性。
   - 用户讨论了在生成输出中对引用进行重排序（reranking）的策略，旨在精简多余的引用以提高清晰度。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Codeium 融资 1.5 亿美元用于扩张**：Codeium 宣布获得 **1.5 亿美元** 的 C 轮融资，公司估值达到 **12.5 亿美元**，总计筹集了 **2.43 亿美元** 用于推动 **R&D**。
   - 利用这些资金，尽管他们尚未动用 1 月份的 **Series B** 资金，但他们旨在加速增长。
- **Meta AI 助手的惊人覆盖范围**：Meta 的 AI 助手月活跃用户已达到 **4 亿**，日活跃用户达到 **4000 万**，凸显了其在市场上的快速普及。
   - 随着平台的增长，这种使用量的激增引发了关于潜在许可需求的讨论。
- **DeepMind 推出可定制的 Gems**：Google DeepMind 推出了 **Gems**，这是为特定角色（如**学习教练**和**编程伙伴**）设计的可定制 AI 聊天机器人。
   - 批评者强调，它们的成功取决于用户友好性以及对这些工具进行的策展（curation）质量。
- **新播客讨论 LLM 基准测试**：最新的 [Latent Space Podcast](https://x.com/latentspacepod/status/1829173832877519152) 中，Google DeepMind 的 Nicholas Carlini 强调了定制 **LLM 基准测试** 的必要性。
   - 他讨论了训练数据提取技术以及由于 OpenAI 丢失 **logprobs** 而带来的挑战。
- **对研究 Agent 效率的担忧**：在讨论中，参与者对研究 Agent 表示担忧，指出平均研究时长为 **2 分钟**，成本约为 **$0.005**，这表明存在效率低下。
   - 关于生成研究论文的 **STORM 方法** 与 one-shot 方法的有效性也展开了辩论，人们更倾向于持续反馈。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 在 Web3 中日益增长的角色**：虽然 **Mojo** 在区块链协议中被探索，但与 **Go**、**Rust** 和 **C++** 相比，它在严肃开发方面仍显不成熟。对 Mojo 的 IO 和网络 API 的持续增强对于匹配现代硬件能力至关重要。
   - *反馈强调需要更强大的开发环境，以减轻程序员对内存管理的担忧。*
- **Mojo 编译器开源的不确定性**：**Mojo** 被宣传为开源，但由于小团队的快速迭代，编译器的源代码目前尚不可用。这种情况何时或是否会改变的时间表仍然模糊。
   - *成员们对项目开发方向缺乏透明度表示担忧。*
- **辩论：编程语言性能**：一场激烈的讨论评估了 **Go** 的性能，特别是与 **C** 的对比，指出 Go 优化器的保守性可能导致在复杂问题上的性能较差。这引发了关于 Go 在未来某些应用中适用性的疑问。
   - *关于 **Go** 在历史上究竟变慢了多少，出现了不同的意见。*
- **MAX SDK 开发策略**：**MAX SDK** 的开发团队正在权衡开发速度、许可和社区参与之间的平衡。寻找同时精通 **MLIR** 和 **Mojo** 的贡献者已被证明具有挑战性。
   - *成员们呼吁扩大团队努力，以填补这些知识空白。*
- **对 OPENSEA 合作的兴奋**：有消息称将与 **OPENSEA** 合作进行新的免费铸造（free mint），引发了成员间的讨论和兴趣。鼓励通过分发的申领链接参与。
   - *虽然兴趣显而易见，但一些成员选择了退出，理由是参与程度各异。*

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 应用在 Docker 中受挫**：一位用户在使用 Docker 容器中的 ChatOllama 对象时遇到了 **LangChain** 应用的问题，而它在 Docker 之外运行正常。根本原因被确定为基础 URL 问题，通过切换到直接的 Ollama 主机 URL 得到解决。
   - *看来 Docker 设置需要特定的配置才能在 LangChain API 中表现良好。*
- **ChatOllama vs Ollama 对决**：**ChatOllama** 专门针对类聊天交互，而 **Ollama** 服务于更广泛的语言模型任务，两者各具独特功能。用户分享了这两个模型的使用示例和详细的 API 参考。
   - *社区赞扬了量身定制的使用案例，明确了为什么会根据项目需求选择 ChatOllama 而非 Ollama。*
- **实时流式输出困惑**：一位用户在使用其 **agent executor** 时面临挑战，该执行器收集了所有输出而不是进行实时流式传输。关于设置 `streamRunnable = False` 对输出行为影响的问题随之而来。
   - *澄清这种行为对于优化模型部署中的实时交互至关重要。*
- **用于增强 LLM 的混合 RAG 模型**：讨论围绕通过反馈和微调技术改进 **LLMs** 展开，尽管它们无法实时学习。参与者探索了传统 RAG 模型和自查询（self-query）技术等替代方案，以提升模型性能。
   - *重点放在了演进 RAG 策略上，以确保竞争性的性能基准。*
- **为 HR 创建自定义 GPT**：一位用户旨在为其 HR 团队构建一个专门的 **GPT 模型**，强调了避免其响应中出现幻觉（hallucinations）的重要性。提出了实施有效 RAG 技术的建议，以优化模型的输出。
   - *社区智慧倾向于根据真实反馈进行迭代调整，以培养一个高效的 HR 工具。*

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GymNation 的数字化转型胜利**：GymNation 显著提升了其会员体验，**数字化线索到销售的转化率提高了 20%**，并实现了 **87% 的数字化线索对话率**，详见其 [成功案例](https://twitter.com/llama_index/status/1829272433687470489)。
   - 他们与 LlamaIndex 的合作推动了 **真实的业务成果**。
- **计划举行 LLMs 生产环境应用讲座**：关注即将于 **9 月 9 日** 举行的关于生产环境中 **large language models** 的讨论，见 [Twitter](https://twitter.com/llama_index/status/1829304873210491392) 上的见解。
   - 本次讲座旨在为有效部署 LLMs 提供关键信息。
- **LlamaIndex 与 MLFlow 集成**：与 **MLFlow** 的新集成增强了 LlamaIndex 应用程序的跟踪和评估能力，正如联合创始人在 [此处](https://twitter.com/llama_index/status/1829569770364227895) 的播客中所分享的。
   - 此次集成有望改进 ML 模型的 **日志记录和性能评估**。
- **加入 LLM x Law 黑客松**：**9 月 8 日** 将迎来 **LLM x Law Hackathon** 的激动人心机会，探索 AI 在法律领域的应用，更多详情请见 [Twitter](https://twitter.com/llama_index/status/1829594190705201570)。
   - 预计将有三个专注于法律领域创新 AI 开发的赛道。
- **通过 MoW 和 RAG 增强财务分析**：一种结合了 **Mixture of Workflows (MoW)** 和 **Corrective RAG** 的新方法，允许使用 **Phi-3** 和 **Qwen-2** 等模型进行高级财务数据分析，如 [此处](https://twitter.com/llama_index/status/1829615136505795064) 所述。
   - 该方法实现了 **对财务报表的上下文感知分析**。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **下周加入 House Party！**：一位成员宣布下周将举行 [House Party](https://discord.gg/open-interpreter-1146610656779440188)，时间定在较早的时段以聚集更多参与者。
   - 邀请函包含了一条诚挚的信息以鼓励参与，为即将到来的活动营造了热烈氛围。
- **需要适用于 KDE 的终端应用**：一位成员报告了 **Konsole**（KDE 当前的终端应用）在滚动时导致屏幕溢出的问题。
   - 围绕替代终端应用程序展开了讨论，以有效处理这些问题。
- **Obsidian OI 插件需要补丁**：一位用户称赞了 **Obsidian OI plugin** 的教程视频，但遇到了安装问题并寻求帮助。
   - 另一位成员敦促在特定频道详细说明这些问题，以便获得针对性的帮助。
- **GameNGen 神经模型驱动实时游戏**：_GameNGen_ 神经模型在单个 TPU 上实现了超过 **20 fps** 的实时 **DOOM** 模拟，展示了令人印象深刻的交互质量。
   - 下一帧预测的 **PSNR 达到 29.4**，测试者发现很难区分真实游戏和模拟游戏。
- **AgentOps 团队让成员们兴奋不已**：对 Adam 和 **AgentOps** 团队的期待与日俱增，最近的讨论强调了令人兴奋的发展。
   - 成员们对这些见解以及围绕未来动向的积极氛围表示感谢。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Google 采购 GPU 引发疑问**：成员们质疑为什么 **Google 在已经拥有 TPU 的情况下仍在购买 NVIDIA GPU**，暗示了潜在的性能考量。
   - *TPU 够用吗？* 在竞争日益激烈的情况下，这引发了对 Google 硬件策略的好奇。
- **RunwayML 清理 Stable Diffusion 仓库**：关于 **RunwayML 删除其在 HuggingFace 和 GitHub 上所有 Stable Diffusion 1.5 仓库**的讨论爆发，这导致了现有项目的混乱。
   - 成员们对 **diffusers 1.5** 功能的影响表示担忧，一位成员指出这破坏了**单文件加载 (single file loading)**。
- **对删除仓库的沮丧**：成员们对 RunwayML 在删除仓库前缺乏归档的远见表示恼火，这影响了各种依赖项。
   - 一位成员推测删除背后可能存在*法律原因*，但未发现引用的具体问题。
- **生成小说封面的挑战**：一位成员分享了为**小说封面**生成合适图像的挑战，寻求实现更偏向**漫画或卡通风格**的方法。
   - 尽管尝试了 **DALL-E**，他们收到的却是 AI 感极重的图片，说明了实现预期风格的困难。
- **Re-LAION-5B 数据集发布**：**Re-LAION-5B 数据集**发布，这是 LAION-5B 的一次重要更新，旨在解决安全问题并删除指向疑似 CSAM 的链接。
   - 与 **Internet Watch Foundation** 等组织的共同努力确保了数据集的完整性，目前提供两个安全版本供下载，详见[公告](https://laion.ai/blog/relaion-5b/)。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **科技巨头关注 OpenAI 的新融资**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round) 报道，Nvidia, Apple 和 Microsoft 正在洽谈投资 **OpenAI** 的新一轮 **1000 亿美元融资**。
   - *很高兴看到一家非营利组织吸引如此大的兴趣*，强调了社区对这一潜在投资的兴奋。
- **ChatGPT 凭借庞大的用户群占据主导地位**：根据 [The Information](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx) 的数据，**ChatGPT** 拥有超过 **2 亿周活跃用户**，而 **Meta AI** 以 **4000 万日活跃用户**紧随其后。
   - 一些成员讨论了 **Meta AI** 可用性受限的影响，特别是在 **EU** 等地区。
- **Tinygrad 推出平价云服务**：**Tinygrad** 推出了每月仅需 **60 美元**的云服务，配备 **4090 GPU** 和 **500 GB** 存储空间，比 **vast ai 便宜 3 倍**。用户可以在本地运行 tinygrad，并通过每个 **'TinyJit' 函数**仅需一次往返来实现更快的云端操作。
   - 该产品旨在为需要本地和云端能力的开发者提供无缝过渡。
- **关于系统提示词 (System Prompts) 与评估的咨询**：一位用户寻求关于 **system prompts** 对评估分数影响的研究，突显了对 Prompt Engineering 日益增长的兴趣。
   - 该咨询表明，人们希望探索如何通过更好的 Prompt 管理来有效改变 AI 模型的性能结果。
- **对聊天机器人竞争的期待**：成员们对正在进行的**聊天机器人大战**表示兴奋，其中一人宣称：*聊天机器人大战已经打响。*
   - 这些论述反映了对 AI 助手生态系统不断演进的信心。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **QLoRA 显存达到极限**：关于 **QLoRA** 显存需求的担忧浮现，用户质疑其是否足以在 **4 个 48GB GPU** 上进行训练。用户指出，在没有 CPU offloading 的情况下，即使是 **较短序列**，其配置也接近显存极限。
   - *成员们讨论了显存性能对训练动态的影响以及潜在的优化方案。*
- **多 GPU 评估咨询**：提出了关于 **TorchTune** 中 **多 GPU 评估** 是否可行的问题，引发了关于最佳实践和配置预期的讨论。
   - *参与者分享了关于性能影响和实现最佳结果配置的看法。*
- **Torch 版本兼容性说明**：一位用户确认他们正在使用 **Torch 版本 2.4.0+cu124**，这引发了与其他配置的兼容性担忧。该版本可能会影响模型在各种配置下的表现。
   - *兼容性讨论强调了软件版本与预期性能结果保持一致的重要性。*
- **排查非法内存访问错误**：一名成员报告在训练期间遇到 **illegal memory access** 错误，建议使用 **CUDA_LAUNCH_BLOCKING=1** 进行有效调试。
   - *他们指出 CUDA 错误可能是异步报告的，这增加了排查过程的复杂性，并建议需要进行更深入的调查。*

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **邀请 DSPy 社区加入这场变革**：一位成员分享了一个 GitHub 仓库，邀请 DSPy 社区加入围绕它的 **变革**，强调社区参与。
   - *他们对协作努力表现出极大的热情，提升了项目的参与度。*
- **LinkedIn 自动职位申请工具走红**：**LinkedIn Auto Jobs Applier** 的 GitHub 仓库备受关注，**每天获得超过 2k 个点赞**，显示出其不断上升的人气。
   - *然而，成员们对其功能表示担忧，指出尚未解决的 GitHub Issue 表明该工具 **仍有待完善**。*
- **与 Michael Ryan 的湾区 AI 见面会**：Michael Ryan 将在 [湾区 AI 见面会](http://Bay.Area.AI) 上讨论 **DSPy** 和 **LM Programs**，涵盖 MIPROv2 优化算法的应用。
   - *他的讨论强调应以与传统软件相同的严谨性来对待 LM Programs，突出了测试和审计的重要性。*
- **AgentOps 平台介绍**：[AgentOps](https://agents.staf.ai/AgentOps) 提供了创建 Agent 的工具，包括图表、监控和回放分析，旨在增强 LLM 的使用。
   - *该开源平台邀请社区贡献，可通过其 [GitHub 仓库](https://github.com/AgentOps-AI/agentops) 获取。*
- **DSPy 疑问与支持**：一位用户寻求关于在哪里发布 **DSPy** 疑问的说明，表现出对故障排除和参与的积极兴趣。
   - *这反映了一个活跃的社区，成员们渴望互相支持并提高对 DSPy 功能的理解。*

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Axolotl GitHub 文档的深色模式请求**：一名成员请求为 Axolotl GitHub 文档添加 **深色模式**，称目前的浅色模式很伤眼。
   - *对于频繁访问配置参数的用户来说，切换到深色模式将显著增强可用性。*
- **Llama 70B 训练的最佳硬件**：关于全量训练 **Llama 70B** 模型所需硬件的问题被提出，特别是关于 **A6000** GPU 是否足够。
   - *确认使用 3x A6000 GPU 足以进行全参数模型的训练。*
- **Transformers 中引入 Assistant Prefill 功能**：一个 Pull Request 提议为 Transformers 中的聊天模板添加 **assistant prefill** 功能，使模型能够自主开始回答。
   - *这一添加旨在满足内部和 GitHub 上表达的广泛需求。*
- **Llama 3.1 特殊 Token 修复**：关于 **Llama 3.1** 基础模型中未初始化特殊 Token 的问题引起了关注，特别是关于分布外（out-of-distribution）嵌入的问题。
   - *作为回应，引入了一个新选项 `fix_untrained_tokens: true` 来帮助解决这些问题。*

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Groq 排行榜添加延迟**：成员指出 **Groq** 尚未添加到排行榜，其 **PRs** 预计将于下周提交。
   - *我们仍在等待* Groq 为评估过程做出贡献。
- **致力于清晰的文档步骤**：一位成员保证他们将记录可复现性所需的必要步骤，以解决之前讨论中的疑虑。
   - 这一举措旨在增强模型 **Documentation** 的**清晰度**。
- **GIS 几何表示测试案例挑战**：一位成员分析了一个 **Java 测试案例**，其模型在 GIS 几何表示的初始化提示词中遇到了困难。
   - 尽管面临挑战，他们得出的结论是，模型的响应在初始化方面优于函数调用 (function calls)。
- **评估温度设置澄清**：成员询问是否如前所述，**所有模型**都在温度为 **0** 的情况下进行评估，以确保公平比较。
   - 一位成员强调，保持参数不变对于获得一致的函数调用输出至关重要。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 的操作限制受到质疑**：一位成员询问 **tinygrad** 是否局限于 **statically scheduled operations**，以及它是否在 **semi-structured sparsity** 和 **weight selection** 方面存在困难。
   - 这一询问引发了围绕该框架整体能力的讨论，并对可能超出 tinygrad 能力范围的操作提出了质疑。
- **George Hotz 寻求 tinygrad 限制的澄清**：George Hotz 请求用户提供在 **tinygrad** 中难以执行的操作的具体示例，旨在评估该框架的通用性和局限性。
   - 这表明他正采取主动方式来了解操作调度可能如何影响 tinygrad 在复杂任务中的可用性。
- **Tensor.cat 在处理 sharded tensors 时面临问题**：一位用户报告在使用 `Tensor.cat` 沿 batch 轴连接 sharded tensors 时遇到 **AssertionError**，表明存在 padding 问题。
   - 虽然可以通过 unsqueeze 增加一个额外维度，但用户在重塑 (reshaping) 结果 Tensor 时仍然遇到困难，这进一步增加了实现的复杂性。
- **澄清 Tensor.cat 错误根源**：用户询问 `Tensor.cat` 的问题是 tinygrad 的根本限制，还是仅仅因为缺乏支持的功能。
   - 他们正在考虑修改代码以处理额外的 batch 维度，或者探索替代方法来规避对 `cat` 的需求。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间没有动态，请告知我们，我们将将其移除。

---

# PART 2: 渠道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1278796860781232260)** (459 条消息🔥🔥🔥): 

> - `Fine-tuning LLMs`
> - `RAG vs. Fine-tuning`
> - `Multi-GPU Training`
> - `Model Ranking and Alpha`
> - `Data Generation for Training`

- **LLM 微调与知识获取**：存在一种普遍的误区，认为微调无法有效地教授新概念，一些研究甚至认为这会导致幻觉（hallucinations）。然而，参与者分享了成功微调的经验，特别是当使用正确的参数和策略时。
   - 讨论指出，微调可以改善响应风格，但拥有稳健的数据集并理解模型训练的复杂性至关重要。
- **RAG 与微调在减少幻觉方面的对比**：参与者讨论了 RAG 与微调在减少幻觉方面的有效性，见解表明，如果操作得当，微调具有其优势。有人指出，RAG 在特定语境下能更好地证实陈述。
   - 最终达成的共识是，结合这两种方法可以产生更好的结果。
- **多 GPU 训练问题的讨论**：用户在尝试同时在不同 GPU 上运行多个 notebook 时遇到了挑战，导致训练出错。经确认，代码对 GPU 检测的处理在根据环境设置验证 GPU 数量时产生了冲突。
   - 已提交一个 Pull Request 来解决这些问题，强调了正确处理 `CUDA_VISIBLE_DEVICES` 环境变量的必要性。
- **微调中 Rank 和 Alpha 的重要性**：模型的 Rank 影响微调期间可训练参数的数量，建议使用更高的 Rank 来有效地教授新概念。参与者讨论了利用更高 Rank 与避免灾难性遗忘（catastrophic forgetting）之间的平衡。
   - 还提到了 RSLora 作为训练期间稳定 Rank 的潜在方法，尽管其有效性在不同用户之间似乎有所差异。
- **有效训练的一般策略**：分享了关于使用特定数据生成和设置配置来增强微调过程的技巧，强调了反复试验的必要性。提供了文章和视频等资源链接，以获取关于有效方法论的进一步见解。
   - 几位社区成员对分享的知识表示感谢，并计划在周末尝试讨论的方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://mistral.ai/news/mathstral/">MathΣtral</a>：为了向阿基米德致敬（今年是我们庆祝他诞辰 2311 周年），我们自豪地发布了首个 Mathstral 模型，这是一个专门为数学推理和科学讨论设计的 7B 模型...</li><li><a href="https://tenor.com/view/fumo-touhou-fumo-touhou-gif-23545090">Fumo Touhou GIF - Fumo Touhou Fumo Touhou - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.kaggle.com/code/mohsenghafari/kaggle-mistral-7b-unsloth">Kaggle Mistral 7b Unsloth محسن کره</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：查看下方列表获取我们所有的 Notebooks：</li><li><a href="https://github.com/mlabonne/llm-autoeval?">GitHub - mlabonne/llm-autoeval：在 Google Colab 中自动评估你的 LLMs</a>：在 Google Colab 中自动评估你的 LLMs。通过在 GitHub 上创建账号来为 mlabonne/llm-autoeval 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2405.05904">在新知识上微调 LLMs 是否会鼓励幻觉？</a>：当大语言模型通过监督微调（SFT）进行对齐时，它们可能会遇到预训练期间未获取的新事实信息。通常推测这可能会教会模型...</li><li><a href="https://github.com/unslothai/unsloth/pull/974">修复单 GPU 环境下的多 GPU 设置训练。由 Sehyo 提交 · Pull Request #974 · unslothai/unsloth</a>：check_nvidia() 最初为 nvidia-smi 派生了一个新进程，从而绕过了 GPU 数量可能受操作系统环境变量限制的情况，因为这不会反映在新进程中。添加...</li><li><a href="https://magic.dev/blog/100m-token-context-windows">1 亿 Token 上下文窗口</a>：关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://huggingface.co/posts/dylanebert/255000504996462">Hugging Face 上的 @dylanebert：“这是一个关于如何微调的 1 分钟视频教程……”</a>：未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit：将算力和书籍转换为指令微调（Instruct-Tuning）数据集（或分类器）！</a>：将算力和书籍转换为指令微调（Instruct-Tuning）数据集（或分类器）！ - e-p-armstrong/augmentoolkit</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>：在为 Open Hands (Open Devin) 准备的 DEVINator 数据上训练的 SmolLM 135M Instruct</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_pos">GitHub - linkedin/Liger-Kernel：用于 LLM 训练的高效 Triton Kernels</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://www.mercity.ai/blog-post/guide-to-fine-tuning-llms-with-lora-and-qlora">使用 LoRA 和 QLoRA 微调 LLMs 的深度指南</a>：在本博客中，我们详细解释了 QLoRA 的工作原理，以及如何在 Hugging Face 中使用它来微调你的模型。</li><li><a href="https://tenor.com/view/orange-cat-smile-cat-smile-orenge-cat-smiling-gif-23133369">橘猫微笑 GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。</a>：LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...</li><li><a href="https://github.com/linkedin/Liger-Kernel?trk=public_post_comment-text">GitHub - linkedin/Liger-Kernel：用于 LLM 训练的高效 Triton Kernels</a>：用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号来为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts">微调是为了形式，而非事实 | Anyscale</a>：微调是特定领域模型精炼（DSMR）的一种方法，但它并不是提高特定领域性能的万灵药。</li><li><a href="https://github.com/unslothai/unsloth/issues/636">将模型存储到 Hugging Face 不起作用 · Issue #636 · unslothai/unsloth</a>：你好，我认为将模型存储到 Hugging Face 的说明不太清楚。Notebook 中的以下行尝试将模型推送到 HF 模型仓库 ("hf/model", tokenizer, quantization_m...</li><li><a href="https://github.com/linkedin/L">GitHub - linkedin/L...</li>

<a href="https://github.com/linkedin/Liger-Kernel/issues/57">与 unsloth 进行基准测试 · Issue #57 · linkedin/Liger-Kernel</a>: 🚀 功能、动机和推介：嘿，你有没有针对使用类似 kernel 的 unsloth 运行过任何基准测试？我猜你的项目可以作为支持多 GPU 的直接替换方案（drop-in replacement）。Alt.....</li><li><a href="https://x.com/BramVanroy/status/1827090122363564251">来自 Bram (@BramVanroy) 的推文</a>: @hsu_byron 这个稳定吗？如果稳定的话，与 @huggingface trainer 的下游集成将非常有价值 :o 我觉得需要通过 accelerate 实现，抄送 @TheZachMueller。</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: 可定制且优化的 Transformers 构建模块，支持组合式构建。</a>: 可定制且优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1278854896661168159)** (12 条消息🔥): 

> - `Training AI scripts` (训练 AI 脚本)
> - `Meta's upcoming models` (Meta 即将推出的模型)
> - `GPT-4o pricing changes` (GPT-4o 价格变动)
> - `LLM provider comparisons` (LLM 提供商对比)
> - `Gemini 2.0 updates` (Gemini 2.0 更新)


- **通过单个脚本简化 AI 训练**：一位用户分享说，他们正在创建两个脚本，以帮助任何人轻松训练自己的 AI，无需复杂的库或设置。
   - 他们强调这只是一个在安装依赖后用 Python 运行的单个代码文件，并提供了 [text generation web ui](https://github.com/oobabooga/text-generation-webui) 的链接作为资源。
- **Meta 将发布新的 Llama 模型**：一篇帖子指出，Meta 很快将发布更新以及下一批 Llama 模型，引发了成员们对这些模型性质的猜测。
   - 讨论包括可能出现类似于 **Chameleon 类型**的新多模态模型，而不是 Llama 4。
- **使用 GPT-4o 降低成本**：一位用户指出，OpenAI 的新 GPT-4o 模型将成本降低至 **每 1M tokens 4 美元**，使得输入和输出 tokens 都显著变便宜。
   - 它还支持 **Structured Outputs**，确保模型输出完全符合指定的 JSON Schemas，从而推动了 LLM 提供商经济模式的转变。
- **LLM 提供商如同现代应用商店**：一位成员评论说，LLM 提供商类似于应用商店，旨在控制开发，从而导致了基于 token 的付费系统，而不是从销售中抽取分成。
   - 另一位成员将其与 **Firebase** 等云服务进行了比较，暗示了货币化策略的更广泛趋势。
- **Gemini 2.0 引起关注**：用户对 **Gemini 2.0** 表达了兴奋，其中一人链接到了讨论其功能和影响的 Reddit 帖子。
   - 有人建议 Gemini 2.0 可能与 AI Studio 中列出的实验性模型有关，引发了对其能力的关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenAIDevs/status/1820987573793386527?utm_campaign=The+Batch&utm_source=hs_email&utm_medium=email">来自 OpenAI Developers (@OpenAIDevs) 的推文</a>: 我们最新的 GPT-4o 模型输入 token 便宜了 50%，输出 token 便宜了 33%。它还支持 Structured Outputs，确保模型输出完全符合您的 JSON Schemas。</li><li><a href="https://www.reddit.com/r/Bard/comments/1f4xamv/wow_gemini_20/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1f43ep8/meta_to_announce_updates_and_the_next_set_of/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: 用于大语言模型的 Gradio Web UI。</a>: 用于大语言模型的 Gradio Web UI。可以通过创建账号为 oobabooga/text-generation-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1278799638295744522)** (67 messages🔥🔥): 

> - `Learning Rate Scheduler`
> - `Fine-Tuning Use Cases`
> - `Tokenizers and Model Loading`
> - `GPU Resources and Configuration`
> - `Memory Optimization Techniques` 


- **Cosine Learning Rate Scheduler 见解**：一位成员询问了在将 Scheduler 设置为带有 Warmup Steps 的 Cosine 模式时，Learning Rate 图表的表现。
   - 另一位成员指出，具体行为可能因特定配置而异。
- **探索 Fine-Tuning 使用案例**：成员们讨论了 Fine-Tuning 的适用场景，并分享了关于其效果的褒贬不一的经验。
   - 一位成员强调，理解 Fine-Tuning 是适用于引入新知识，还是更适合其他用途至关重要。
- **训练后 Tokenizer 使用的差异**：一位成员询问何时有必要在训练模型后推送 Tokenizer。
   - 共识是，只有在添加了新 Token 的情况下才有必要推送 Tokenizer。
- **有效管理 GPU 资源**：讨论集中在租用硬件与购买硬件进行 AI 训练任务的效率对比。
   - 几位成员一致认为，对于偶尔的训练任务，租用计算资源通常更便宜且更实用。
- **训练期间优化 RAM 使用**：一位在训练 DPO 模型时遇到内存不足错误的成员，寻求在 16GB RAM 限制内的优化技术建议。
   - 建议包括减小 Batch Size 以及检查不同训练方法的内存需求，并指出 DPO 通常比标准的 Fine-Tuning 需要更多的 VRAM。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/llama3-1">使用 Unsloth Fine-tune Llama 3.1</a>：通过 Unsloth Fine-tune 并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的 Context 长度！</li><li><a href="https://hastebin.com/share/ilelinosan.python">Hastebin</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1278885321509179485)** (4 messages): 

> - `OpenRouter Launch`
> - `Llama 3.1 Model`
> - `Pricing Strategy` 


- **Llama 3.1 在 OpenRouter 上低调发布**：经过数周的努力，**Llama 3.1-405B-instruct** 现已在 OpenRouter 上线，支持完整的 **128k Context** 和 **Function Calling**。
   - 该模型可在 **avian.io** 使用，为真实用户提供服务，同时保持 **$2.5/mil tokens** 的最低价格。
- **关于从 OpenRouter 获利的评论**：针对一项询问，该成员澄清说，无论链接使用情况如何，他们都会收到报酬，并对在该项目中的工作表示满意。
   - 他们强调自己不会从推荐中赚取额外费用，并对构建该基础设施感到自豪。



**提到的链接**：<a href="https://openrouter.ai/models/meta-llama/llama-3.1-405b-instruct/providers">Meta: Llama 3.1 405B Instruct – 提供商状态</a>：查看提供商状态并向 Meta: Llama 3.1 405B Instruct 发起负载均衡请求 —— 备受期待的 400B 级 Llama 3 已经到来！具备 128k Context 和令人印象深刻的评估分数...

  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/)** (1 messages): 

hamchezz: 我想为了某个不明确的目标 Fine-Tuning 一个 LLM，纯粹是因为好玩 😄
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1278793009789534239)** (285 条消息🔥🔥): 

> - `Gemini Model Performance` (Gemini 模型性能)
> - `Sonnet Benchmarking` (Sonnet 基准测试)
> - `Magic Dev's Long-Term Memory Model` (Magic Dev 的长期记忆模型)
> - `Aider's Growth and Future` (Aider 的增长与未来)
> - `Coding Assistance Tools` (编程辅助工具)


- **Gemini 模型性能备受关注**：关于新 Gemini 模型性能的讨论不断，一些用户表示怀疑，特别是与其在 Aider 中的兼容性有关。
   - 回复显示，虽然许多人认为该模型令人印象深刻，但对其在某些特定场景下的有效性仍存疑虑。
- **Sonnet 基准测试显示性能稳定**：最近的评估表明 Sonnet 依然高效，基准测试结果显示 Aider 的代码编辑能力随时间推移没有出现明显退化。
   - 用户注意到，尽管有传言，但性能统计数据揭示了在各项测试中通过率保持稳定。
- **Magic Dev 发布长期记忆模型**：Magic Dev 推出了一款拥有 1 亿（100M）token 超大上下文窗口的模型，旨在通过推理而非依赖记忆来增强编程任务。
   - 这一进展引发了人们对其在软件开发和其他复杂问题解决任务中潜在应用的兴趣。
- **Aider 的发展路径与社区参与**：Paul G. 对社区参与 Aider 这一开源工具的演进表示热忱，并表示目前没有做出剧烈变动的即时计划。
   - 讨论涵盖了未来的增长方向，包括可能推出 GUI 版本以增加用户参与度以及投资机会。
- **AI 编程工具对比**：用户讨论了 Zed 等编程辅助工具的使用体验以及对 Magic Dev 预期产品的看法，并指出 Magic 目前尚无可用产品。
   - 普遍观点认为 Microsoft 的 GitHub Copilot 是该领域的重要参与者，其不断扩大的用户群和营收能力备受关注。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/2024/08/26/sonnet-seems-fine.html">Sonnet seems as good as ever</a>：Sonnet 在 aider 代码编辑基准测试中的得分自发布以来一直保持稳定。</li><li><a href="https://tenor.com/view/dancing-cat-dance-cat-cat-meme-chinese-cat-gif-12629347036627000898">Dancing Cat Dance GIF - Dancing cat Dance Cat - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/homer-brain-monkey-gif-11098413">Homer Brain GIF - Homer Brain Monkey - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.continue.dev/features/codebase-embeddings">Codebase Retrieval | Continue</a>：与你的代码库对话</li><li><a href="https://x.com/openaidevs/status/1823510395619000525?s=46">OpenAI Developers (@OpenAIDevs) 的推文</a>：该模型现在也可以在 API 中以 `chatgpt-4o-latest` 形式使用。我们建议大多数 API 使用场景采用 `gpt-4o-2024-08-06`，但也很高兴让开发者测试我们最新的改进...</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows</a>：关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。</li><li><a href="https://github.com/nu">Nu Deployment</a>：Nu Deployment 有 3 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/nus-apr/auto-code-rover">GitHub - nus-apr/auto-code-rover</a>：一个具备项目结构感知能力的自主软件工程师，旨在实现自主程序改进。在 SWE-bench lite 中解决了 30.67% 的任务 (pass@1)，在 SWE-bench verified 中解决了 38.40% 的任务 (pass@1)，每项任务成本低于 $0.7。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1278853802145091595)** (69 messages🔥🔥): 

> - `Aider Model Support`
> - `Repo File Detection`
> - `User Experience with Aider Errors`
> - `Handling Large Repos with Aider`
> - `Integration with GitHub Copilot` 


- **Aider Model Support Confusion (Aider 模型支持困惑)**：用户讨论了 `.env` 文件中 Aider 模型的设置，在使用 OpenRouter API 时，关于模型名称的正确格式存在困惑。
   - 一位用户提到收到关于未指定 LLM Provider 的错误，导致对所需环境变量产生困惑。
- **Repo File Detection Issues (仓库文件检测问题)**：一位用户询问如何让 Aider 自动检测最近创建的文件，因为这些文件似乎只能通过 `/add` 命令显现。
   - 社区建议使用 `/drop` 和 `/clean` 等命令，但都没有解决新文件无法自动建议的问题。
- **Improving User Experience with Aider Errors (改进 Aider 错误的初学者体验)**：围绕如何更有效地展示错误消息展开了讨论，以避免在出现问题时让长文本块使用户感到不知所措。
   - 成员们一致认为，简化错误呈现可以改善 UX，特别是当用户无法立即识别关键错误消息时。
- **Handling Large Repos with Aider (使用 Aider 处理大型仓库)**：一位用户询问了 Aider 在处理大型仓库时的效率限制，引发了关于管理复杂性的讨论。
   - 为了保持 Aider 的专注，建议仅添加相关文件，并将任务分解为更小、易于管理的步骤，以便进行更有效的代码编辑。
- **Integration with GitHub Copilot (与 GitHub Copilot 的集成)**：一位成员询问 Aider 理论上是否可以利用 GitHub Copilot API，因为他们的公司已经批准了 Copilot，但尚未批准其他 LLM。
   - 对话强调了由于公司审批流程（特别是围绕法律审查）在集成各种 LLM 时面临的挑战。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/warnings.html">Model warnings</a>: Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/config/dotenv.html">Config with .env</a>: 使用 .env 文件为 Aider 存储 LLM API 密钥。</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: 了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://github.com/ChimeHQ/SwiftTreeSitter">GitHub - ChimeHQ/SwiftTreeSitter: Swift API for the tree-sitter incremental parsing system</a>: 用于 tree-sitter 增量解析系统的 Swift API - ChimeHQ/SwiftTreeSitter</li><li><a href="https://openrouter.ai/settings/keys">Keys | OpenRouter</a>: 管理你的密钥或创建新密钥</li><li><a href="https://aider.chat/docs/languages.html#how-to-add-support-for-another-language">Supported languages</a>: Aider 支持几乎所有流行的编程语言。</li><li><a href="https://github.com/paul-gauthier/grep-ast/issues/7">`py-tree-sitter-languages` is unmaintained · Issue #7 · paul-gauthier/grep-ast</a>: 你好 @paul-gauthier，感谢你在 Aider 上的工作。我使用得很愉快。这个项目使用了 https://github.com/grantjenks/py-tree-sitter-languages，但该项目已停止维护且...</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>: 使用 Aider 进行 AI 结对编程的技巧。</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 Aider 的常见问题解答。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1278937913689899091)** (1 条消息): 

> - `Anthropic Prompt Engineering`
> - `Jupyter Notebooks`
> - `UVX Tool` 


- **Anthropic 提供卓越的 Prompt Engineering 资源**：Anthropic 在提供文档方面持续保持领先，通过[此链接](https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/)可以访问其 **Prompt Engineering 交互式教程**。
   - 该教程通过 Jupyter notebooks 交付，确立了 Anthropic 在 LLM 厂商文档领域的领导地位。
- **Jupyter Notebooks 实操 Prompt Engineering**：教程的前几章虽然基础，但有效地演示了如何使用 **Anthropic API** 进行简单的 prompt 操作，并建议使用安装命令 `%pip install anthropic` 以确保正确的虚拟环境设置。
   - 用户可以使用 `git clone https://github.com/anthropics/courses` 和 `uvx --from jupyter-core jupyter notebook courses` 等命令快速启动 Jupyter 服务器。
- **问题报告与社区贡献**：一位用户提到在学习教程后，在 Anthropic 的 GitHub 仓库提交了一个 issue 和一个 pull request，这表明了活跃的社区参与。
   - 这体现了通过协作改进资源的承诺，增强了整体学习体验。



**提到的链接**：<a href="https://simonwillison.net/2024/Aug/30/anthropic-prompt-engineering-interactive-tutorial/">Anthropic’s Prompt Engineering Interactive Tutorial</a>：Anthropic 延续了其在领先 LLM 厂商中提供最佳文档的趋势。该教程以一组 Jupyter notebooks 的形式提供 —— 我使用了它……

  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1278796810671882260)** (318 条消息🔥🔥): 

> - `Personalization of LLMs`
> - `Comparison of AI Models`
> - `AI for Customer Support`
> - `AI Coding Performance`
> - `Upcoming AI Releases` 


- **LLM 的个性化需求**：用户讨论了 **LLM 个性化**的重要性，建议 AI 拥有独特的性格和对话的长期记忆，以提升用户体验。
   - 同时也提出了关于使用 API 调用来维持个性化交互所带来的**成本**影响的担忧。
- **AI 模型之间的对比**：辩论了各种 AI 模型的性能，特别是 **Gemini** 和 **Grok 2**，一些用户注意到 Grok 很有创意，但在处理复杂的编程任务时并不总是可靠。
   - 一些用户对 **Grok** 等模型的特定输出表示失望，强调其有效性取决于具体的 prompt。
- **AI 在客户支持应用中的表现**：一位用户询问了如何构建用于客户支持的 AI 聊天机器人，提到了使用 **OpenAI API** 的潜力，但也对其复杂性表示担忧。
   - 建议包括从更简单的 AI 项目开始，或者考虑与开发者合作或使用现有的 no-code 解决方案。
- **AI 在编程中的性能**：对话涵盖了 **Grok 2** 和 **Gemini** 等模型的相对编程能力，讨论了 Grok 在某些创意方面表现更好，但在调试（debugging）方面可能会失误。
   - 证言表明，用户需要根据具体的编程任务仔细选择模型，并暗示 Grok 在某些场景下可以提供可行的输出。
- **未来的 AI 发布与发展**：关于 **OpenAI** 和 **Cerebras** 新 AI 模型发布时间的推测不断出现，同时也引发了对市场影响以及在 AGI 领域与其他国家竞争的担忧。
   - 参与者对 AGI 进展的影响表达了谨慎态度，特别是在地缘政治背景下，同时讨论了科技公司之间潜在的合作。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/)** (1 条消息): 

smilebeda: 👍
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 条消息🔥): 

> - `Job Description vs CV Matching`
> - `Prompt Engineering Strategies`
> - `Document Analysis Efficiency`
> - `API Call Structure`
> - `Deep Document Analytics Discussion` 


- **Job Description vs CV Matching 评分**：讨论强调了用于评分 CV 与职位描述（Job Description）相似度的 Prompt 存在的问题，特别是如何评估来自不同背景的候选人。
   - 一位用户提到他们在 Prompt 导致评分不准确方面的困扰，并分享了对所使用的特定评分方法的反馈。
- **寻找更好的匹配分类规则**：建议优化职位评分的分类规则，以避免过度依赖语义相似度。
   - 一位用户表示，增强分类规则可以立即改善评分结果。
- **分离 API 调用 vs 单个 Prompt**：对话探讨了是针对不同问题使用多个 API 调用，还是使用单个 Prompt 进行文档分析。
   - 建议使用独立的请求可以减少幻觉（hallucinations）并保持回复的清晰度。
- **利用 Batch Processing 提升效率**：分享了使用 Batch Processing 的建议，并附带了 OpenAI 文档链接，以增强 Prompt 效率。
   - 该方法可以简化大型文档的分析，且不会使模型过载。
- **探索深度文档分析**：有人询问关于深度文档分析以及使用 ChatGPT 响应进行数据收集的讨论。
   - 用户被引导至特定频道，以获取有关 Fine-Tuning 和模型利用的更多信息。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1279017909661863967)** (16 条消息🔥): 

> - `Prompt Engineering`
> - `API Call Strategies`
> - `Document Analytics`
> - `Batch Processing`
> - `Fine-Tuning` 


- **Job Description 匹配中的挑战**：一位成员分享了使用 OpenAI API 比较职位描述与 CV 的经验，尽管使用了包括特定评分规则在内的多个 Prompt，但收到的相似度评分仍不一致。
   - CV 的评分范围从 5 到 65 不等，且某些比较给出的理由不明确，导致了困惑。
- **通过 Fine-Tuning 改善结果**：一位成员建议对模型进行 Fine-Tuning 可能会解决评分差异，但指出这需要大量数据集。
   - 提出了另一种方案，即强制 API 以 JSON 格式返回响应，以确保结构化输出。
- **优化 Prompt 复杂度**：一位用户询问了针对多个问题使用独立 API 调用与使用单个综合 Prompt 从大型文档中提取信息的效率对比。
   - 建议指出，独立的请求可以减少产生幻觉的机会，因此小型请求是更优的策略。
- **文档分析探索**：一位成员对深度文档分析表示兴趣，并寻求讨论有效数据提取方法的资源。
   - 提到可以先从 ChatGPT 响应开始进行数据收集，然后再转向对其他模型进行 Fine-Tuning。
- **开发者资源共享**：一位用户被引导至关于 Batch Processing 的 OpenAI 文档，以优化其实现策略。
   - 另一位成员对该资源表示感谢，并表示有兴趣获取更多关于文档分析的信息。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1278799856135311360)** (223 messages🔥🔥): 

> - `Llama 3.1 405B 的推理`
> - `GPT-2 本地使用`
> - `Amazon ML Challenge 2024`
> - `测试与代码质量`
> - `AI 与 CAD 的集成` 


- **Llama 3.1 405B 推理端点的挑战**：一位成员询问了成功调用 **Llama 3.1 405B** 模型推理端点的情况，另一位提到使用 **NIM 3.1 405B** 的良好体验。
   - 大家似乎很有兴趣了解这一最新模型与其他模型相比所提供的效率和能力。
- **GPT-2 的使用体验**：一位用户对 **GPT-2** 产生的意外输出表示困惑，称其在运行过程中表现出威胁性且行为异常。
   - 其他人推荐使用 **Llama** 和 **Mistral** 等更新的模型作为聊天和指令任务的更好替代方案。
- **Amazon ML Challenge 2024 寻觅队友**：一位用户正在为 **Amazon ML Challenge 2024** 寻找队友，希望在竞赛项目上进行合作。
   - 该询问缺少关于挑战赛细节的额外信息，为社区内的互动创造了机会。
- **测试在代码质量中的重要性**：一位成员分享了运行多次测试对代码质量重要性的见解，强调了早期错误检测和提高可维护性。
   - 他们列举了各种测试方法，包括单元测试和集成测试框架，以确保整体代码的稳定性。
- **AI 与 CAD 系统的集成**：一场关于类似《钢铁侠》中 **J.A.R.V.I.S** 的 AI 模型能力的对话展开，提到了现有的将 AI 与 **CAD** 集成的项目。
   - 会中指出，多位个人已在 AI 领域取得了显著进展，展示了交互式应用的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to">torch.Tensor.to &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://huggingface.co/docs/hub/repositories-licenses">Licenses</a>：未找到描述</li><li><a href="https://x.com/NM_Reid/status/1825997577151525338">Noah Reid (@NM_Reid) 的推文</a>：呃，Anaconda 刚刚给我们的 HPC 管理员发了消息，说我们违反了他们的服务条款（ToS），现在我们需要支付许可费，或者从我们的系统中移除他们所有的软件？</li><li><a href="https://ollama.com/unclemusclez/smollm-135m-instruct-devinator">unclemusclez/smollm-135m-instruct-devinator</a>：在 DEVINator 数据上训练的 SmolLM 135M Instruct，用于 Open Hands (Open Devin)
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1278900366250872914)** (4 messages): 

> - `模型训练中的人类反馈`
> - `低比特量化 (Low Bit Quantization)`
> - `GPU 对训练的重要性`
> - `用于 AI 学习的 Colab 和 Kaggle` 


- **人类反馈在模型准确性中的作用**：最近的一篇论文讨论了**人类反馈 (human feedback)** 如何作为评估 **Large Language Models** 的事实标准及其对训练目标的影响。研究强调了偏好评分的弱点，表明它们可能低估了事实性等关键方面，详见 [PDF](https://arxiv.org/abs/2309.16349)。
   - 分析显示，输出的**断言性 (assertiveness)** 可能会扭曲感知到的事实性错误，这表明需要更全面的评估指标。
- **学习低比特量化 (Low Bit Quantization)**：一位成员对**低比特量化**表示感兴趣，并链接了一篇关于该技术的基石性[研究论文](https://arxiv.org/pdf/1609.07061)。这表明研究重点在于优化模型性能，同时减少存储需求。
   - 理解这种方法可以显著提高神经网络运行的效率。
- **GPU 对训练的重要性**：针对有效模型训练中 **GPU** 的必要性提出了强烈建议，特别是通过 **Colab** 或 **Kaggle** 等平台。该成员敦促其他人不要在没有 GPU 支持的情况下进行训练。
   - 这反映了对于现代 AI 任务中基于 CPU 训练的局限性已达成共识。



**提到的链接**：<a href="https://arxiv.org/abs/2309.16349">Human Feedback is not Gold Standard</a>：人类反馈已成为评估 Large Language Models 性能的事实标准，并越来越多地被用作训练目标。然而，目前尚不清楚哪些属性...

  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1278820480358420603)** (4 messages): 

> - `LLM Pruning`
> - `Text-to-Speech Machine Learning`
> - `Multi-Party Chat Agents`
> - `Qwen2-VL series`
> - `Vision-Language Models` 


- **高效的 LLM Pruning 策略**：最近的一项研究探讨了针对开源权重预训练 LLMs 的层剪枝（layer pruning）策略，发现直到移除 50% 以上的层之前，性能下降都非常微小。他们使用 **parameter-efficient finetuning** 技术（如 QLoRA）来增强剪枝后的模型。
   - 该方法可以显著减少计算资源，同时提高推理（inference）过程中的内存效率并降低延迟。
- **GitHub 项目助力 Text-to-Speech 增强**：[Text-to-Speech-ML GitHub 仓库](https://github.com/Azymack/Text-to-Speech-ML-)是一个旨在推进文本转语音技术的协作项目。用户可以通过创建 GitHub 账号参与其开发。
   - 该倡议为语音合成领域的机器学习协作改进和创新提供了一个平台。
- **多方对话 Agents 的探索**：一篇新论文讨论了对话系统处理**多方对话（multi-party conversations）**的需求，这与传统的双人对话不同。作者引入了一个名为 MultiLIGHT 的数据集，旨在改进模型在更复杂交互中的训练。
   - 挑战包括决定何时发言，以及根据对话中的多个角色生成连贯的回复。
- **Qwen2-VL 系列：SOTA Vision-Language Models**：Qwen 发布了 **Qwen2-VL 系列**，这是先进的 Vision-Language Models，还结合了视频理解能力。该系列标志着视觉和语言处理集成方面的重大进展。
   - 更多详情可以查看他们在 [Qwen 博客](https://qwenlm.github.io/blog/qwen2-vl/)上的公告。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>：我们对流行系列的开源权重预训练 LLMs 的简单层剪枝策略进行了实证研究，发现在不同问答基准测试中，直到...</li><li><a href="https://arxiv.org/abs/2304.13835">Multi-Party Chat: Conversational Agents in Group Settings with Humans and Models</a>：目前的对话研究主要研究双人（两方）对话，并未涉及两个以上发言者共同交谈的日常场景。在这项工作中，我们收集了...</li><li><a href="https://github.com/Azymack/Text-to-Speech-ML-">GitHub - Azymack/Text-to-Speech-ML-</a>：通过在 GitHub 上创建账号，为 Azymack/Text-to-Speech-ML- 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1278811373656215704)** (12 条消息🔥): 

> - `VividNode`
> - `ToonGPT Launch`
> - `Word Game Bench`
> - `FLUX LoRA Training`
> - `Thoth Bot` 


- **VividNode: 您的个人 AI 聊天机器人**：一位用户发布了名为 **VividNode** 的开源聊天机器人，该项目使用 Python 和 PySide6 开发，并计划在未来增强其功能。作者表达了对提升技能以及在开源项目上与他人合作的兴奋之情。
   - 用户可以直接在桌面上体验类 GPT 的功能和图像生成，聊天记录存储在本地。
- **ToonGPT 在 Product Hunt 上线**：***ToonGPT*** 现已在 [Product Hunt](https://www.producthunt.com/products/toontales-kiddiegpt) 上线，邀请用户支持这款为儿童设计的互动趣味工具。开发者热衷于收集反馈以改进平台。
   - 他们强调渴望社区参与和用户互动，以进一步增强产品功能。
- **推出用于模型评估的 Word Game Bench**：介绍了一个名为 **Word Game Bench** 的新框架，用于在单词拼图游戏中评估语言模型，突出了其挑战性。该基准测试在 **Wordle** 和 **Connections** 等游戏上评估模型，提供了优于典型评估方式的独特优势。
   - 模型以交互方式参与游戏，利用反馈来提高性能，同时正在考虑一些非常规的评估方法。
- **FLUX LoRA 训练教程发布**：分享了使用 **Kohya SS GUI** 进行 FLUX LoRA 训练的**教程指南**，旨在为用户简化训练过程。该教程专为 GPU 资源有限的用户设计，特别是针对运行 Windows 的 8GB GPU。
   - 本指南适合希望通过实用框架快速上手的学习者。
- **AI 驱动的 Thoth Bot 发布**：**Thoth Bot** 是一款 AI 驱动的 CLI 工具，通过 Groq API 利用多个 LLM 执行聊天和 Python 代码生成等任务。它通过自动化代码生成、执行和错误修复等功能简化了编码工作流。
   - 该项目旨在提高编码效率并简化程序员的开发过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>：未找到描述</li><li><a href="https://dev.to/p3ngu1nzz/scaling-up-parallel-training-with-tau-llm-and-unity-ml-agents-53bh">未找到标题</a>：未找到描述</li><li><a href="https://airesearch.wiki/index.html">ai-research-agent</a>：未找到描述</li><li><a href="https://medium.com/@yjg30737/what-is-vividnode-how-to-use-it-4d8a9269a3c0">什么是 VividNode 以及如何使用它</a>：VividNode 是一款允许您在桌面上直接体验 GPT 聊天机器人 (ChatGPT) 和图像生成功能的软件，无需……</li><li><a href="https://www.producthunt.com/products/toontales-kiddiegpt"> ToonTales - KiddieGPT - 产品信息、最新更新和 2024 年评论 | Product Hunt</a>：介绍 ToonGPT：一个为孩子们精心打造的愉快 AI 伴侣！受我女儿 Becky 的启发，ToonGPT 将卡通的魔力与互动乐趣相结合，激发创造力和快乐……</li><li><a href="https://github.com/U-C4N/Thoth-Bot">GitHub - U-C4N/Thoth-Bot: AI 驱动的 CLI 工具，用于聊天、Python 代码生成，并使用通过 Groq API 的多个 LLM 进行改进。通过自动化代码生成、执行和错误修复来简化编码工作流。</a>：AI 驱动的 CLI 工具，用于聊天、Python 代码生成，并使用通过 Groq API 的多个 LLM 进行改进。通过自动化代码生成、执行和错误修复来简化编码工作流。 - U-...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1278904332636389430)** (5 条消息): 

> - `Meta FAIR 的 Transfusion 研究`
> - `对 Transfusion 的印象`
> - `GitHub 更新` 


- **Meta FAIR 的 Transfusion 声称在多模态建模方面取得飞跃**：Meta FAIR 关于 **Transfusion** 的突破性研究引入了一个统一框架，用于在混合模态序列上训练 Transformer，它能同时预测 token 并进行图像扩散（diffusion）。
   - 实验结果展示了**卓越的性能**和可扩展性，达到了与拥有 **70 亿参数**和 **2 万亿多模态 token** 的模型相当的结果。
- **对 Transfusion 潜在影响的兴奋**：成员们对 **Transfusion** 的潜力表示赞同，其中一人指出它是**多模态建模**领域的游戏规则改变者。
   - *“看到这将如何塑造多模态任务中 AI 的未来，令人感到兴奋，”* 这句话捕捉到了讨论中普遍的狂热情绪。
- **关于 Transfusion 实际性能的疑问**：一位成员对 **Transfusion** 的表现表示好奇，并评论说论文中出现了许多 **Gen AI 关键词**。
   - 这种怀疑态度反映出人们希望在炒作之外，能更深入地了解该框架的有效性。
- **为了记录而更新 GitHub**：一位成员更新了他们的 **GitHub** 以进行记录，并请求针对发现的任何问题提供反馈。
   - 这反映了社区在保持透明度和解决潜在疑虑方面的持续努力。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1278908354969997403)** (13 条消息🔥): 

> - `图像处理技术`
> - `迁移学习 (Transfer Learning) 挑战`
> - `噪声文档分类`
> - `项目协作` 


- **利用图像处理评估文档质量**：一位成员建议结合使用图像处理技术和 OpenCV 等预训练模型，从**模糊度**和**暗度**方面评估文档质量。
   - 他们建议采用 **Laplacian variance** 等算法来检测模糊，并参考 **VGG** 或 **ResNet** 等 CNN 来获取通用图像质量特征。
- **文档质量的迁移学习实验**：另一位成员报告了在手动调整亮度和模糊度的数据集上使用**迁移学习**所面临的挑战，表示在实际测试中表现不佳。
   - 他们表示有兴趣探索 OpenCV 技术，并询问是否有相关文章可以指导他们，因为这是组织中普遍存在的问题。
- **关于项目源代码的讨论**：有人请求提供项目源代码，以便为所讨论的文档分类项目提供更具体的解决方案。
   - 成员们分享了他们的 GitHub 仓库链接，包括 [noisy_doc_clf](https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb)，其中记录了增强图像和迁移学习的尝试。
- **深夜交流**：一位成员提到现在是深夜，提议第二天早上继续讨论。
   - 他们还提到发送了**好友请求**，得到了另一位成员的同意和感谢。



**提到的链接**：<a href="https://github.com/ajkdrag/noisy_doc_clf/blob/main/notebooks/train.ipynb">noisy_doc_clf/notebooks/train.ipynb at main · ajkdrag/noisy_doc_clf</a>：通过在 GitHub 上创建账户来为 ajkdrag/noisy_doc_clf 的开发做出贡献。

  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1278803082410721300)** (9 messages🔥): 

> - `LLaMA 3 models`
> - `GPU requirements for LLaMA 3`
> - `Inference configurations for LLaMA 3`
> - `Comments on shared advice`
> - `Cost considerations for LLaMA 3` 


- **关于 LLaMA 3 模型的求助**: 一位成员寻求在使用 **LLaMA 3 models** 构建 RAG 应用方面的帮助，询问针对 **8B**、**70B** 和 **405B 参数**模型的合适 **on-premise GPUs** 和 **RAM 配置**建议。
   - 他们特别询问了有效运行这些大型模型所需的最佳 GPU 和 RAM 设置。
- **Nvidia A100 被推荐为最佳 GPU**: 针对该请求，另一位成员建议将 **Nvidia A100 - large** 作为最佳 GPU 选项。
   - 他们没有明确该建议对应哪个具体模型，从而引发了关于 RAM 需求的进一步追问。
- **LLaMA 405B 对 GPU RAM 需求极高**: 一位成员解释说，运行 **LLaMA 405B** 将需要不止一个 GPU，并且根据精度设置（precision settings），至少需要 **300Gb 的 GPU RAM**。
   - 他们警告说该模型的运行成本极高，并建议考虑 **cloud-based options**（云端方案）。
- **对贡献内容的思考**: 一些成员对提供的建议发表了评论，其中一人断言这是模型生成的回复，没有增加任何有用信息。
   - 另一位成员幽默地推测，前一条消息可能是由 **LLaMA 3 本身**生成的。
- **关于 Audit Log 的询问**: 鉴于对话的动态情况，一位成员建议检查 **audit log**，可能是为了进一步澄清之前的交互。
   - 这一言论表明了回顾讨论参与背景的意愿。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1278919068015005746)** (2 messages): 

> - `Animating Fireballs`
> - `AnimateDiff`
> - `IP Adapter Plus`
> - `SVD` 


- **为照片中的火球制作动画**: 一位用户询问如何仅对其照片中的 **fireball**（火球）制作动画。
   - 另一位成员建议使用 **AnimateDiff** 配合 **IP Adapter Plus** 或 **SVD** 作为潜在的解决方案。
- **火球动画技术的建议**: 社区讨论了为静态图像添加动画效果的各种技术。
   - 强调了 **AnimateDiff** 等特定工具在实现动态视觉增强方面的有效性。


  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 messages): 

iron_bound: 听起来他们的 LTM 架构有一个用于 attention 的 RNN
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1278902836868022282)** (1 messages): 

> - `Triton Atomic Add Scope`
> - `Multi-GPU Configurations`
> - `GPU vs System Scope` 


- **理解 Triton 的 Atomic Add 作用域设置**: `scope=GPU` 配置指定操作仅在 GPU 上执行，而 `scope=system` 则涵盖了跨多个 GPU 和 host 的计算。
   - 这种区别会根据跨多 GPU 设置的执行上下文影响性能优化。
- **默认 GPU 作用域与多 GPU 兼容性**: 默认的 `scope=GPU` 旨在多 GPU 环境中无需额外配置即可开箱即用地高效运行。
   - 用户在使用默认设置时，应能获得预期的多 GPU 功能，而无需特殊调整。
- **澄清作用域选项的含义**: 在 Triton 的 atomic add 实现中，`scope=system` 意味着能够同时利用 GPU 和 host 资源进行操作。
   - 它本质上表示比仅 GPU 更广泛的操作上下文，允许集成处理。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1279199308410781848)** (3 条消息): 

> - `自定义 Triton kernel 的 FX pass`
> - `从 PyTorch 调用 Triton`
> - `FX pass 示例` 


- **关于 FX pass 映射的好奇心**：一位用户对是否可以通过 FX pass 将 **aten** 操作映射到自定义 **Triton** kernel 表示好奇。
   - 这表明了通过自定义实现来优化特定操作性能的兴趣。
- **PyTorch 中的原生 Triton 调用**：另一位用户澄清说，你可以从 **PyTorch** 程序中**原生调用 Triton 代码**，并通过 **torch.compile** 启用相关功能。
   - 这为直接在 PyTorch 工作流中集成高级 GPU 加速开辟了途径。
- **FX pass 示例资源**：讨论中提到，浏览 **Triton** 代码对于研究 FX pass 很有帮助，并提供了一个具体的 [GitHub 资源链接](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py)。
   - 提供的链接指向一个文件，该文件可以作为理解 PyTorch 上下文中 FX pass 的参考。



**提到的链接**：<a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/fx_passes/pre_grad.py">pytorch/torch/_inductor/fx_passes/pre_grad.py at main · pytorch/pytorch</a>: Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch

  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1278803556308353197)** (25 条消息🔥): 

> - `Quantization Techniques` (量化技术)
> - `AWQ Implementation Challenges` (AWQ 实现挑战)
> - `Low-Bit Optimizer Adjustments` (低比特优化器调整)
> - `Model Fixes in AO` (AO 中的模型修复)
> - `Mixed Precision Quantization` (混合精度量化)


- **对 Attention 层进行量化引发辩论**：讨论了关于在 Attention 层中有意对 **QKV 投影**进行量化的问题，并就如何保持准确性提出了见解。
   - 成员们指出，**默认 filter_fn** 在特定假设下运行，会自动量化具有 2D 形状的 **Linear 层**。
- **AWQ 的整数预期引起困惑**：一名成员强调了 AWQ 的一个**实现困境**，强调与整数存储相比，浮点零点会恶化困惑度 (perplexity)。
   - 据透露，之前尝试交换量化/反量化（quant/dequant）函数的尝试受到了**舍入逻辑**差异的影响，导致实现不兼容。
- **低比特优化器代码正在审查中**：参与者审查了**低比特优化器代码**，对一行关于非符号位的代码提出质疑，指出这可能是一个疏忽。
   - 发现这段代码是从外部库复制的，这引发了关于**保持正确逻辑**的进一步深入讨论。
- **AO 的 ML 模型修复看起来很有前景**：指出了 AO 的 **llama 模型**的一个修复方案，其中特定训练条件下的一个实现细节被标记为需要注意。
   - 审查人员已参与其中，以确保模型行为符合预期功能。
- **混合精度工作引起关注**：随着成员分享方法和参考资料，对话倾向于探索**混合精度量化**技术。
   - 提供了一个相关项目链接，展示了一名实习生根据不同模型大小进行的**混合精度量化**工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/blob/main/torchao/prototype">ao/torchao/prototype at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化 - pytorch/ao</li><li><a href="https://gist.github.com/mobicham/8b3147742beb3b302064453a15ced428#file-awq_hqq_test-py-L52">awq_hqq_test.py</a>：awq_hqq_test.py。GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/quantization/prototype/mixed_precision">ao/torchao/quantization/prototype/mixed_precision at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L28C5-L28C54).">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorc">pytorc - 概览</a>：pytorc 有 2 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/prototype/low_bit_optim/quant_utils.py#L69-L106">ao/torchao/prototype/low_bit_optim/quant_utils.py at main · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化与稀疏化 - pytorch/ao</li><li><a href="https://pytorch.org/docs/stable/generated/torch.searchsorted.html">torch.searchsorted &mdash; PyTorch 2.4 文档</a>：未找到描述</li><li><a href="https://github.com/pytorch/ao/pull/769">由 yiliu30 修复 llama 模型 · Pull Request #769 · pytorch/ao</a>：在训练模式下 (model.setup_caches(..., training=True)) 且 input_pos 为 None 时，freq_cis 被 L208 覆盖。ao/torchao/_models/llama/model.py 第 19 行...</li><li><a href="https://github.com/bitsandbytes-foundation/bitsandbytes/blob/e4674531dd54874c0abbc786ad5635c92c34dc3e/bitsandbytes/functional.py#L360">bitsandbytes/bitsandbytes/functional.py at e4674531dd54874c0abbc786ad5635c92c34dc3e · bitsandbytes-foundation/bitsandbytes</a>：通过针对 PyTorch 的 k-bit 量化实现可访问的大语言模型。- bitsandbytes-foundation/bitsandbytes
</li>
</ul>

</div>

### **CUDA MODE ▷ #[sequence-parallel](https://discord.com/channels/1189498204333543425/1208496482005549086/1279136959381639301)** (2 messages): 

> - `Flash Attention Kernel Challenges`
> - `NVIDIA GeForce RTX 3090 Compatibility`
> - `Attention Head Dimensionality` 


- **Flash Attention Kernel 共享内存大小的难题**：讨论围绕编写 Flash Attention Kernel 时遇到的共享内存大小挑战展开，特别是 Q、K 和 V 块显示出巨大的内存需求，例如一个 Q 块需要 **131,072 字节**。
   - 用户觉得他们可能忽略了 Flash Attention 如何在 SRAM 尺寸较小的非 Hopper GPU 上运行。
- **NVIDIA GeForce RTX 3090 上的 Flash Attention 问题**：有人提出了在两块 **NVIDIA GeForce RTX 3090** GPU 上运行 flash_attn 包时遇到的问题，原因是这些 GPU 的算力（Compute Capability）为 **8.6**。
   - 该情况在一个 [GitHub issue](https://github.com/Dao-AILab/flash-attention/issues/190) 中被强调，用户正在寻求这些兼容性问题的解决方案。
- **Attention Heads 对模型维度的影响**：有人提出了一个问题，即巨大的模型维度是否被划分到了各个 Attention Head 中，并建议 Flash Attention 中的**每个 Head** 仅管理约 **64 或 128** 的较小内部维度。
   - 这引发了关于在保持性能的同时优化 Flash Attention 模型资源使用的思考。



**提到的链接**：<a href="https://github.com/Dao-AILab/flash-attention/issues/190">Support for NVIDIA GeForce RTX 3090 with Compute Capability 8.6 · Issue #190 · Dao-AILab/flash-attention</a>：Issue 描述：你好，我正在一个拥有两块 NVIDIA GeForce RTX 3090 GPU 的系统上使用 flash_attn 包，两者的算力均为 8.6。尝试运行该包时，我遇到了...

  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1278869677178884159)** (15 messages🔥): 

> - `Twitter recommendations`
> - `Benefits of Twitter`
> - `Summer ‘24 Twitter Poll`
> - `Logistics for CUDA Mode event` 


- **Twitter 账号推荐**：一名成员寻求 Twitter 上值得关注的账号推荐，引发了多次回应。
   - 一位用户分享了一个 [关注列表](https://x.com/marksaroufim/following)，而另一位用户则建议重新考虑加入 Twitter 的决定。
- **关于 Twitter 效用的讨论**：成员们讨论了 Twitter 在紧跟最新新闻和关注进行 SOTA 研究的知名实验室方面的益处。
   - 有人提到它对于分享工作很有用，其他人也同意通过精选关注账号来改善体验。
- **关于 24 年夏季 Twitter 价值的投票**：发起了一项投票，询问成员在 2024 年夏季花在 Twitter 上的时间是净正面还是净负面的。
   - 对话探讨了 Twitter 是否最终值得投入时间，回复中情感交织。
- **CUDA Mode 活动的后勤问题**：一名新成员询问了即将举行的 CUDA Mode 活动的后勤情况，包括酒店住宿和食物供应。
   - 他们表示由于周末要从州外赶来，因此需要这些信息。


  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1278873486789447681)** (6 messages): 

> - `L2 Side Aware Performance`
> - `FP8 Transition`
> - `Loss Landscape Insights`
> - `Training Sample Drop Impact` 


- **L2 Side Aware 代码提升性能**：在修复了一些 bug 后，L2 Side Aware 代码在 GELU forward 中稳定达到了 **~1823GB/s**，高于使用 x128 的 Kernel 3 的 **1791GB/s**。
   - 这些改进还带来了**更低的功耗**，尽管仍有必要进行进一步的简化和优化。
- **计划切换到 FP8**：开发者计划明天切回 **FP8** 编码，以刷新对该部分的记忆。
   - 目前 L2 Side Aware 代码的进展被认为暂时足够了。
- **关于 Loss Landscape 的见解**：一名成员讨论了某些优化如何导致 Loss Landscape 中的驻点（stationary point），表明这可能与 **regular AdamW** 方法有相似之处。
   - 这些见解表明，理解所达到的极小值的质量至关重要，强调了进行一些 Vanilla 微调验证的必要性。
- **丢弃训练样本的影响**：有人担心丢弃部分训练样本可能会如何影响优化结果，但有人建议其影响可能不会比其他策略更“不确定”。
   - 对话强调了通过实现来对比传统方法结果的重要性。


  

---

### **CUDA MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/)** (1 条消息): 

mobicham: https://x.com/JamesLiuID/status/1829554782287413513
  

---


### **CUDA MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1278791518769119324)** (140 条消息🔥🔥): 

> - `Liger-Kernel Release v0.2.0`
> - `LayerNorm Kernel 更新`
> - `Hugging Face 示例中的内存问题`
> - `原子操作 (Atomic Operations) 研究`
> - `文档增强` 


- **Release v0.2.0 带来的变化**：宣布了 Liger-Kernel 的新版本发布，解决了之前的发布问题，亮点包括更整洁的 API 和更多的模型支持。
   - 最近的测试显示一些用户遇到了显存溢出 (OOM) 错误，引发了关于新版本与 0.1.1 相比内存效率的讨论。
- **LayerNorm Kernel 集成**：合并了一个 PR 以集成自定义 Kernel 和 LigerLayerNorm 模块，提升了归一化操作的性能。
   - 关于性能的讨论指出，在多 GPU 操作中使用 sys scope 值得进一步研究。
- **Hugging Face 示例中的内存问题**：用户报告称，在运行 Hugging Face 示例进行模型训练时，该框架的 0.2.0 版本导致了意外的 OOM 错误。
   - 一位用户确认回滚到 0.1.1 版本解决了该问题，而其他人则推测了与使用 Liger 相关的潜在原因。
- **调查原子操作 (Atomic Operation) 不匹配问题**：一位用户对 RMS 归一化 Kernel 在重写以支持部分聚合时出现的测试反复失败表示担忧。
   - 尽管在测试中设置了手动种子以实现确定性行为，但结果中仍然存在不匹配，这表明 Kernel 实现中存在底层问题。
- **Liger-Kernel 的文档更新**：README 中新增了描述 LayerNorm Kernel 的章节，以增强文档并指导用户使用。
   - 聊天参与者讨论了对 Liger-Kernel 中集成自定义操作的更好文档和教程的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://hidet.org/docs/stable/gallery/developer-guides/add-operator-resolve-rule.html">添加算子解析规则 &#8212; Hidet 文档</a>: 未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/179)">Issues · linkedin/Liger-Kernel</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)`">CUDA 语义 &mdash; PyTorch 2.4 文档</a>: 未找到描述</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/174">在使用 LigerKernel 时 torch.compile() 抛出异常 · Issue #174 · linkedin/Liger-Kernel</a>: 🐛 描述该 bug ... 文件 &quot;/home/tromero/workspace/seahorse/.venv/lib/python3.11/site-packages/torch/_inductor/async_compile.py&quot;, 第 173 行, 在 triton kernel = TritonCodeCache.load(kernel_...</li><li><a href="https://github.com/linkedin/Liger-Kernel/releases/tag/v0.2.0">Release v0.2.0 版本发布说明 · linkedin/Liger-Kernel</a>: 开场感言 🫶 谢谢！我们想借此机会向社区表达由衷的感谢！2500+ ⭐ , 10+ 新贡献者, 50+ PRs, 以及与 Hugging Face 🤗 的集成, 还有...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/180">[文档] AndreSlavescu 将 LayerNorm 添加到 README · Pull Request #180 · linkedin/Liger-Kernel</a>: 摘要：在 README 中添加了 LayerNorm 描述。测试完成：不适用。硬件类型：RTX 3090。运行 make test 以确保正确性，运行 make checkstyle 以确保代码风格，运行 make test-convergenc...</li><li><a href="https://github.com/linkedin/Liger-Kernel.git">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton Kernels</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/169">[算子] AndreSlavescu 提交 LayerNorm Kernels + LigerLayerNorm · Pull Request #169 · linkedin/Liger-Kernel</a>: 摘要：集成了 layernorm 自定义 kernels + LigerLayerNorm 模块。测试完成：测试了 layernorm kernels 的正确性。硬件类型：RTX 3090。运行 make test 以确保正确性，运行 make...</li><li><a href="https://github.com/linkedin/Liger-Kernel/pull/135/files#diff-0b31d056b2cdb59db1baaba4c4e7e0a79ed70b445ca67ff928ec57ffa89c6d0fR71">AndreSlavescu 提交自定义 Embedding kernel · Pull Request #135 · linkedin/Liger-Kernel</a>: 摘要：添加了 Embedding 前向/反向 kernels + 映射到 nn.Embedding 的 LigerEmbedding 类。nn.Embedding 对于像 BERT 这样的 encoder-only 模型非常有用。参考：#131。测试完成：测试了...</li><li><a href="https://github.com/linkedin/Liger-Kernel?tab=readme-ov-file#kernels">GitHub - linkedin/Liger-Kernel: 用于 LLM 训练的高效 Triton Kernels</a>: 用于 LLM 训练的高效 Triton Kernels。通过在 GitHub 上创建账号，为 linkedin/Liger-Kernel 的开发做出贡献。
</li>
</ul>

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1278792548558835713)** (187 条消息🔥🔥): 

> - `Model Compatibility and Optimizations` (模型兼容性与优化)
> - `SGE Usage in AI Tools` (AI 工具中的 SGE 使用)
> - `Training Model Costs` (模型训练成本)
> - `Stable Diffusion Model Updates` (Stable Diffusion 模型更新)
> - `Generation Times with Different GPUs` (不同 GPU 的生成时间)


- **SDXL 性能优化**：用户建议在 `webui-user.bat` 中添加 `--xformers`、`--medvram-sdxl` 和 `--no-half-vae`，以提升 SDXL 在低显存（VRAM）GPU 上的表现。
   - 这些命令旨在共同提高速度并减少 VRAM 占用，同时保持与 VAE 的兼容性。
- **深入理解 SEG**：关于 SEG 在工作流中的实现存在困惑，用户质疑其在 Impact Pack 等工具中的必要性和复杂性。
   - SEG 的集成引发了疑问，即它是一种成熟的方法，还是为特定工具创建的新概念。
- **AI 模型训练成本**：训练 SD1.5 或 SDXL 等基础模型被指出需要数月时间和巨额资金投入，成本可能高达数百万美元。
   - 用户讨论道，虽然大型模型的训练成本高昂，但较小的 Checkpoint 或 LoRA 模型可以使用较少的资源进行训练。
- **RunwayML 删除 Stable Diffusion 仓库**：RunwayML 从 HuggingFace 等平台移除其 Stable Diffusion 1.5 仓库的行为引起了用户的担忧。
   - 围绕这一决定的猜测暗示其重心可能正在从旧模型转移。
- **3060/3060 Ti 的生成速度**：使用 3060 和 3060 Ti GPU 的用户讨论了他们在运行 SDXL 和 Flux 模型时的生成时间体验和预期。
   - 用户对硬件是否能妥善处理漫长的生成时间和模型存储需求表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://imgur.com/ygD5YMm">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行迷因、娱乐 GIF、感人故事、病毒视频等来振奋精神...</li><li><a href="https://imgur.com/Xr44AHl">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行迷因、娱乐 GIF、感人故事、病毒视频等来振奋精神...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings">命令行参数与设置</a>: Stable Diffusion web UI。通过在 GitHub 上创建账号为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2408.16232">通过可解释的潜空间操作增强条件图像生成</a>: 在图像合成领域，在遵循条件提示的同时保持对参考图像的忠实度仍然是一个重大挑战。本文提出了一种集成 d... 的新方法。</li><li><a href="https://github.com/kshitij79/CS-7476-Improvements-in-Diffusion-Model">GitHub - kshitij79/CS-7476-Improvements-in-Diffusion-Model</a>: 通过在 GitHub 上创建账号为 kshitij79/CS-7476-Improvements-in-Diffusion-Model 的开发做出贡献。</li><li><a href="https://www.amazon.de/Fantastische-Fabelwesen-Stressabbau-Entspannung-Fantasie-Kreaturen/dp/B0CN5B8WTG/ref=sr_1_1?crid=3IBODT2J8X6H6&dib=eyJ2IjoiMSJ9.-3XggVW3uObjvvXQqObf-g-EWf_V6QDcBkrHerEySuY2P3W0J8JG92mAOXoFt2DWOwZHT1w0m6M4IrDxhUwXVi523Affpx6n5y5TI3Pal5iMGXUuSJEje7x1BSRxDuAhRJqcESyU0awWBpc07xA90cucn7Z_uETG34wev0if1-ON4ICntYnPnlLPGVH6WUk532dqEr89fXftuzS4TrhIrYMCKNik-WVzuMj3aU2Vvr8.d_Vd1P3m4memC-Dd8Agtfsyxu8CgD6J3vjQdJ--SaDo&dib_tag=se&keywords=fabelwesen+malbuch&qid=1724956770&sprefix=Fabelwesen+%2Caps%2C126&sr=8-1">奇幻生物：成人减压与创意放松涂色书 - 包含精灵、仙女、龙和许多其他神秘幻想生物 : Press, Flying Colours, Gehrke, Nora: Amazon.de: Bücher</a>: 未找到描述</li><li><a href="https://mp.weixin.qq.com/s/ZKJieSzqISyzCB8Iz9tY8A">【AI行业报告】Top 100 AI 产品 (第3期)</a>: AI 行业报告第三期，来看看哪些 AI 产品上榜了？
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1278805467065352263)** (118 条消息🔥🔥): 

> - `AI 中的失忆模式 (Amnesia Mode)`
> - `梯度的低秩近似 (Low-Rank Approximations)`
> - `训练 LLaMA 3`
> - `针对 LLM 的 Word Game Bench`
> - `Hermes 3 梯度行为` 


- **AI 在沟通风格上表现出偏好**：用户在测试 **Hermes 3** 的失忆模式（amnesia mode）时注意到，它更倾向于使用专业语言，甚至会拒绝像 'bruh' 这样被认为不适合家庭场景的非正式词汇。
   - 这突显了一个潜在趋势，即 AI 模型表现出预定义的性格特征或沟通准则。
- **关于优化中低秩近似的讨论**：一位用户建议在梯度传输中使用**低秩近似 (low-rank approximations)**，这可以减少分布式训练中节点间的通信开销。
   - 这与目前关于自适应通信模式和梯度对性能影响的持续讨论相一致。
- **在多样化指令数据上训练 LLaMA 3**：一名成员分享了他们正在使用来自 Reddit 和 StackExchange 等平台的合成及真实指令数据训练一个 **8b LLaMA 3 模型**。
   - 他们的目标是观察在这些真实数据上训练是否能减少“AI 味”的行为，展示了模型精调（refinement）的不同方法。
- **用于评估语言模型的 Word Game Bench**：介绍了 **Word Game Bench**，这是一个专注于 **Wordle** 和 **Connections** 等文字游戏的 LLM 测试框架，展示了一种新颖的评估策略。
   - 该基准测试强调通过交互式游戏体验来衡量模型性能，而非静态响应，解决了 LLM 评估中的常见痛点。
- **模型微调中的意外行为**：用户观察到模型微调结果异常，有报告称在训练过程中出现损失值（loss）波动以及潜在的梯度爆炸（exploding gradients）。
   - 这些见解反映了训练大型 AI 模型固有的复杂性和不可预测性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2311.08105">DiLoCo: Distributed Low-Communication Training of Language Models</a>: 大型语言模型 (LLM) 已成为机器学习许多应用中的关键组件。然而，训练 LLM 的标准方法需要大量紧密互连的加速器...</li><li><a href="https://x.com/wingsoverheaven/status/1829024789693968628">wings (@wingsoverheaven) 的推文</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/nousresearch/hermes-3-llama-3.1-70b">Hermes 3 70B Instruct - API, Providers, Stats</a>: Hermes 3 是一个通用语言模型，相比 [Hermes 2](/models/nousresearch/nous-hermes-2-mistral-7b-dpo) 有许多改进，包括先进的 Agent 能力、更好的角色扮演、推理...</li><li><a href="https://wordgamebench.github.io">Word Game Bench</a>: 未找到描述</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">zafir (@zafstojano) 的推文</a>: 很高兴分享 "Word Game Bench" —— 一个评估语言模型在文字拼图游戏中表现的有趣基准！这是一个相对困难的基准，目前没有模型的平均得分超过 50%...</li><li><a href="https://fxtwitter.com/sama/status/1829205847731515676?s=19">Sam Altman (@sama) 的推文</a>: 我们很高兴与美国 AI Safety Institute 达成协议，对其未来的模型进行发布前测试。出于多种原因，我们认为这在国家层面发生非常重要...</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: A Gradio web UI for Large Language Models.</a>: 一个用于 Large Language Models 的 Gradio web UI。可以通过在 GitHub 上创建账号来为 oobabooga/text-generation-webui 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1278815792594686032)** (43 messages🔥): 

> - `Instruct Tuning`
> - `Model Hosting and Precision`
> - `Full Precision 与 Quantization 的性能对比`
> - `Lambda Cloud API 使用`
> - `1 亿 Token Context Window` 


- **Instruct Tuning 与用户端训练**：*许多成员讨论了在用户输入与输出上训练模型的差异，* 结论是仅在输出（outputs）上进行训练能获得更好的 Benchmark 结果。
   - 一位成员提到他们在 Loss 计算中 Mask 掉了输入数据，专注于从输出开始进行预测。
- **对 Full Precision 模型托管的需求**：*几位成员对缺乏全精度 Hermes 3 模型的托管选项表示沮丧，* 尤其是考虑到该模型仅权重就占用 **810GB** 的巨大资源需求。
   - 有人指出，像 Anthropic 和 OpenAI 这样的大型供应商拥有专门用于服务模型的优化硬件，这暗示了其他供应商可能面临的需求挑战。
- **性能对比：8-bit vs Full Precision**：*强调了 8-bit Quantization 与 16-bit Full Precision 之间存在几个百分点的差异，* 且越大的模型表现出越强的 Quantization 抗性。
   - 讨论揭示了较小模型在低比特率下可能出现连贯性问题，表明模型大小显著影响性能。
- **适用于 Hermes 3 的 Lambda Cloud API**：*成员们讨论了将 Lambda Cloud API 作为访问 Hermes 3 的可行方法，* 但担心它仅提供 8-bit 量化选项，且不允许设置 System Prompt。
   - 然而，一位成员指出该 API 实际上支持 System Prompt，表明了其在特定应用中的可用性。
- **Context Window 技术的突破**：*一位成员提到了关于 1 亿 Token Context Window 的说法，并将其称为“魔法”，* 预示着 LLM 能力的提升。
   - 另一位成员补充道，此类进步可能类似于此前公认的 AI 领域突破，暗示了该领域持续的发展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/NousResearch/Hermes-3-Llama-3.1-405B">NousResearch/Hermes-3-Llama-3.1-405B · Hugging Face</a>：未找到描述</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lambda-chat-completions-api">Using the Lambda Chat Completions API | Lambda Docs</a>：未找到描述</li><li><a href="https://docs.lambdalabs.com/on-demand-cloud/using-the-lam">Lambda Docs</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 messages🔥): 

> - `GameNGen`
> - `DOOM 模拟`
> - `实时游戏引擎`
> - `独特的游戏设计`
> - `恐怖游戏潜力` 


- **GameNGen 模拟 DOOM**：游戏 **DOOM** 完全由 _GameNGen_ 神经模型模拟，这标志着实时游戏引擎领域的一项重大技术成就。
   - *人类评分者很难* 区分模拟片段与实际游戏画面，展示了该模型的有效性。
- **概念验证备受推崇**：成员们公认 _GameNGen_ 是一个**伟大的概念验证（Proof of Concept）**，并引发了关于 **Unreal** 等主流引擎如何集成类似技术的兴趣。
   - 讨论强调了对这种创新方法感兴趣的个人进行*复制（Replication）的潜力*。
- **迷幻的游戏体验**：_GameNGen_ 的游戏画面被描述为**迷幻且梦幻（trippy and dreamlike）**，引发了人们对未来实现写实游戏的科技能力的关注。
   - 一位成员表示有兴趣利用*独特的幻觉效果（hallucination effects）*来打造原创游戏体验，尤其是在恐怖游戏领域。
- **恐怖游戏设计的潜力**：利用 _GameNGen_ 技术开发**独特且令人耳目一新的恐怖 IP** 的前景令人兴奋，重点在于其独特的氛围特质。
   - 然而，有人指出，此类设计可能需要对模型进行大量的*人工引导（hand-holding）*才能达到预期效果。



**提到的链接**：<a href="https://gamengen.github.io/">GameNGen</a>：Diffusion Models Are Real-Time Game Engines

  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1278890189103566868)** (8 messages🔥): 

> - `GameNGen`
> - `Neural Models in Gaming`（游戏中的神经模型）
> - `DOOM Simulation`（DOOM 模拟）
> - `Innovative Game Engines`（创新游戏引擎）
> - `Dreamlike Gameplay`（梦幻般的玩法）


- **GameNGen 使用神经模型模拟 DOOM**：创新的 _GameNGen_ 神经模型展示了在不使用任何传统游戏引擎的情况下模拟经典游戏 **DOOM** 的能力，帧率超过每秒 **20 帧**。
   - 人类评分者难以区分模拟片段与真实游戏画面，展示了其作为概念验证（proof of concept）的有效性。
- **关于与 Unreal Engine 整合的讨论**：成员们对 **Unreal Engine** 等平台未来可能如何整合像 _GameNGen_ 这样的神经模拟技术表示好奇。
   - 社区渴望看到这项技术的复制实现，强调了其对游戏开发的潜在影响。
- **独特的幻觉特质引发关注**：_GameNGen_ 的游戏画面被描述为**迷幻（trippy）**且梦幻，引发了关于其在原创游戏创意方面潜力的讨论。
   - 一位成员建议，可以利用该模型的独特特质来打造一个**令人耳目一新的恐怖 IP**，尽管这可能需要大量的引导。
- **未来技术对游戏的意义**：社区成员对在游戏中使用神经模型的未来影响感到兴奋，特别是对于创建更具沉浸感的体验。
   - 他们设想将这些独特的玩法愿景与传统游戏机制相结合，以实现更深层次的互动。



**相关链接**：<a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---



### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1278791887024689195)** (93 messages🔥🔥): 

> - `LM Studio Updates`
> - `Flash Attention Support`
> - `Model Performance Issues`
> - `API Security`
> - `Text Generation Models` 


- **LM Studio 更新 (0.3.2) 性能改进**：最新的 LM Studio 更新 (0.3.2) 解决了之前 Flash Attention 的延迟问题，提升了本地推理环境下的用户性能。
   - 一位用户对改进的功能表示感谢，同时也提到了与早期版本相比在稳定性方面的担忧。
- **Flash Attention 模型讨论**：成员们讨论了目前哪些模型支持 Flash Attention，其中 LLaMa-3.1 和 Mistral 被强调为兼容选项。
   - Google 的 Gemma-2 模型也被指出支持 Flash Attention，引发了关于整体模型性能和兼容性的讨论。
- **有限 VRAM 运行大模型的挑战**：拥有 8GB VRAM 的用户报告了加载较大模型（特别是 xLAM 7b）时的困难，表明了与上下文和 VRAM 利用率相关的性能问题。
   - 为了排查性能缓慢的问题，建议调整设置以优化 VRAM 使用，并尝试使用较小的上下文数值进行测试。
- **API 安全担忧**：一位用户询问如何在端口转发期间为 LM Studio API 添加身份验证以确保安全。
   - 建议根据需要实施自定义安全措施，一位成员分享了创建反向代理（reverse proxy）以实现更好控制的计划。
- **文本和语音生成功能**：有关于 LM Studio 内部是否提供文本转图像或文本转语音功能的咨询。
   - 用户被告知此类功能目前在 LM Studio 框架内不可用或不受支持。


<div class="linksMentioned">

<strong>相关链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/sophosympatheia/Midnight-Miqu-70B-v1.5/blob/main/tokenizer_config.json#L31">tokenizer_config.json · sophosympatheia/Midnight-Miqu-70B-v1.5 at main</a>: 未找到描述</li><li><a href="https://github.com/THUDM/CogVideo">GitHub - THUDM/CogVideo: Text-to-video generation: CogVideoX (2024) and CogVideo (ICLR 2023)</a>: 文本转视频生成：CogVideoX (2024) 和 CogVideo (ICLR 2023) - THUDM/CogVideo</li><li><a href="https://huggingface.co/lmstudio-community/xLAM-7b-r-GGUF/tree/main">lmstudio-community/xLAM-7b-r-GGUF at main</a>: 未找到描述</li><li><a href="https://github.com/YorkieDev/LMStudioWebUI">GitHub - YorkieDev/LMStudioWebUI: A wip version of a simple Web UI to use with LM Studio</a>: 一个用于 LM Studio 的简单 Web UI 的开发中版本 - YorkieDev/LMStudioWebUI
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1278858932751695872)** (82 messages🔥🔥): 

> - `M2 Ultra Mac 配置`
> - `多 GPU 的电源分配`
> - `LLM 性能基准测试`
> - `使用 NVLink 进行显存共享`
> - `Llama 3.1 模型速度测试` 


- **M2 Ultra Mac 到货，LLM 潜力巨大**：一位用户配置了拥有 **192 GB Unified Memory** 和 **2 TB 硬盘** 的新 **M2 Ultra Mac**，旨在探索 LLM 之前先建立开发环境。
   - *Pydus* 渴望了解他的新机器能加载多大的模型。
- **讨论 GPU 的功耗限制**：针对拥有 **96核 EPYC** 和 **4x RTX 4090s** 的配置，一位用户指出计算显示功耗限制为 **3500W**，强调了在多个插座上进行谨慎电源分配的必要性。
   - 对话涉及配置多个 **PSU**，并确保它们能够承受负载而不导致断路器跳闸。
- **LLM Token 速率的实证测试**：**Mylez_96150** 分享了 **Llama 3.1** 70B 模型在多 GPU 设置下运行速度达到 **每秒 97 个 token**，而另一位用户表示早前记录的速度可能仅为 **每秒 1 个 token**。
   - 讨论探索了各种设置，包括如何在跨 GPU 拆分模型层时优化性能。
- **多 GPU 推理的挑战**：关于如何高效地在多个 GPU 上运行 **LLM** 提出了疑虑，特别是使用 **NVLink** 驱动程序时性能是否会有所提升，以及显存共享如何影响速度。
   - 交流中强调，正确的模型加载和配置有可能显著提高吞吐量。
- **辩论 PCIe 配置的影响**：一位用户询问将 **RTX 4090** 的设置从 **Gen4 x16** 切换到 **Gen4 x8** 在处理 **70B** 或 **405B** 模型时会如何影响性能。
   - 另一位用户解释说，只有在多个 GPU 运行稠密模型时，推理速度才可能受到显著影响，这意味着某些配置可能会看到剧烈的性能变化。



**提到的链接**：<a href="https://tenor.com/view/power-usage-auxiliary-nuclear-gif-22138997">Power Usage Auxiliary Nuclear GIF - Power Usage Auxiliary Nuclear - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1278938553975308310)** (2 messages): 

> - `Gemini Flash 8B`
> - `Gemini Flash Experiment`
> - `价格更新`
> - `数据库停机`
> - `供应商分离` 


- **Gemini Flash 8B 模型发布**：新的 [Gemini Flash 8B (EXP)](https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp) 已与 [Gemini Flash Experiment](https://openrouter.ai/models/google/gemini-flash-1.5-exp) 一同上线。
   - 在 AI Studio 的定价最终确定前，这两个模型都将保持 **免费**。
- **Google Vertex 与 Google AI Studio 分离**：Google Vertex 已正式从 Google AI Studio 中分离，现在被识别为两个独立的供应商。
   - 此举旨在明确产品方案并改善用户导航。
- **Gemini 实验性模型价格调整**：最新更新确认，目前所有 **Gemini Experimental 模型** 都是 **免费** 的。
   - 此次调整旨在提供可访问的资源，直到未来的定价模型建立。
- **因数据库问题导致的停机记录**：由于数据库错误，经历了 **15 分钟** 的停机，目前已恢复。
   - 问题已迅速解决，将对服务的影响降至最低。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/google/gemini-flash-8b-1.5-exp">Gemini Flash 8B 1.5 Experimental - API, Providers, Stats</a>：Gemini 1.5 Flash 8B Experimental 是 [Gemini 1. 的 8B 参数实验版本。通过 API 运行 Gemini Flash 8B 1.5 Experimental</li><li><a href="https://openrouter.ai/models/google/gemini-flash-1.5-exp>">Gemini Flash 1.5 - API, Providers, Stats</a>：Gemini 1.5 Flash 是一个基础模型，在各种多模态任务中表现良好，如视觉理解、分类、摘要以及从图像、音频和视频创建内容...
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1278792284514549924)** (2 条消息): 

> - `daun.ai launch`
> - `AI CLI tool` 


- **daun.ai 庆祝发布成功**：成员们向 **daun.ai** 背后的团队表示祝贺，庆祝其最近的成功发布。
   - *聊天频道中充满了对这一重大里程碑的欢呼和认可。*
- **探索全能型 AI CLI 工具**：一位用户对 **sigoden** 开发的 [AI CLI 工具](https://github.com/sigoden/aichat) 表达了极大的热情，将其描述为集 Chat-REPL、Shell Assistant 等功能于一体的全能解决方案。
   - 该工具支持访问包括 **OpenAI**、**Claude** 和 **Gemini** 在内的多种模型，凸显了其多功能性。



**提到的链接**：<a href="https://github.com/sigoden/aichat">GitHub - sigoden/aichat: All-in-one AI CLI tool featuring Chat-REPL, Shell Assistant, RAG, AI tools &amp; agents, with access to OpenAI, Claude, Gemini, Ollama, Groq, and more.</a>：集成了 Chat-REPL、Shell Assistant、RAG、AI tools &amp; agents 的全能型 AI CLI 工具，支持访问 OpenAI、Claude、Gemini、Ollama、Groq 等。 - sigoden/aichat

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1278791605289095269)** (146 条消息🔥🔥): 

> - `Sonnet 和 DeepSeek 的缓存功能`
> - `Perplexity 模型的问题`
> - `Cohere Command 模型更新`
> - `对 Qwen 模型提供商的担忧`
> - `停机时间与基础设施升级` 


- **Sonnet 和 DeepSeek 的缓存功能即将推出**：一位成员询问了 **Sonnet** 和 **DeepSeek** 缓存功能的上线时间，另一位成员表示可能很快就会推出，甚至可能是明天。
   - 讨论强调了数据库事件导致了优先级的转移，从而影响了发布时间表。
- **Perplexity 模型的问题**：一位用户报告了 Perplexity 模型出现的错误，收到了一条关于无效模型的提示，并要求澄清该问题。
   - 该问题被确认是由于之前的 Bug 引入的，目前已得到快速处理。
- **Cohere Command 模型更新已发布**：**Command R** 模型进行了一次显著更新，简化了其访问点并更改了模型 ID，以确保运行更加顺畅。
   - 用户对更新带来的好处表示兴奋，特别是关于定价和改进的模型性能。
- **对 Qwen 模型提供商的担忧**：用户对 **Qwen** 的提供商 **DashScope** 表示担忧，因为该提供商在用户中知名度不高，尽管它产出了极具前景的 Benchmark 数据。
   - 尽管对提供商存在不确定性，用户仍表现出通过现有平台探索和测试该模型的渴望。
- **停机时间与基础设施升级**：报告的停机时间有所增加，引发了在基础设施升级过程中对系统健康状况和响应能力的担忧。
   - 团队承认了源于数据库限制的问题，并正在进行旨在加强后端实力的项目。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://api.together.ai/models/Qwen/Qwen1.5-4B-Chat">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/qwen/qwen-2-7b-instruct)">Qwen 2 7B Instruct - API, Providers, Stats</a>: Qwen2 7B 是一个基于 Transformer 的模型，在语言理解、多语言能力、编程、数学和推理方面表现出色。它具有 SwiGLU 激活、Attention QKV 偏置和 Grou...</li><li><a href="https://cohereforai-c4ai-command.hf.space/">Cohere Command Models</a>: Command R 模型针对多种用例进行了优化，包括推理、摘要和问答。由 Cohere 和 Cohere For AI 开发。</li><li><a href="https://x.com/alibaba_qwen/status/1829187292038115413?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 要访问 Qwen2-VL-72B，暂时您应该按以下方式使用我们的官方 API：</li><li><a href="https://x.com/OfficialLoganK/">来自 GitHub - FixTweet/FxTwitter 的推文</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://x.com/OfficialLoganK/status/1828922199425548486)">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @DaveManouchehri 在 AI Studio 中免费。我凭记忆不知道 Vertex 的实验性端点是否免费</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/gemini-experimental#pricing)">未找到标题</a>: 未找到描述</li><li><a href="https://github.com/Pythagora-io/gpt-pilot/issues">Issues · Pythagora-io/gpt-pilot</a>: 第一个真正的 AI 开发者。通过在 GitHub 上创建账号为 Pythagora-io/gpt-pilot 的开发做出贡献。</li><li><a href="https://huggingface.co/CohereForAI">CohereForAI (Cohere For AI)</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/docs/model-cards.","type":"invalid_model","code":400}}">未找到标题</a>: 未找到描述</li><li><a href="https://marketplace.visualstudio.com/items?itemName=PythagoraTechnologies.gpt-pilot-vs-code&ssr=false#review-details">Pythagora (GPT Pilot) Beta - Visual Studio Marketplace</a>: Visual Studio Code 扩展 - 第一个真正的 AI 开发者。</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据各应用的使用情况对语言模型进行排名和分析
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1278825342605201611)** (56 messages🔥🔥): 

> - `Embedding 训练中的 NaN 权重`
> - `研究合作反馈`
> - `SAE 中的稀疏编码`
> - `Vision embedding 与 vision token 对比`
> - `训练输入统计调整` 


- **NaN 权重导致 Embedding 训练中断**：一位用户报告称，尽管初始范围正常，但其 Embedding 权重在训练开始几步后就变成了 **NaN**。多位成员建议检查梯度和可能四舍五入为零的损失组件，最终指向 **数据依赖的衰减项 (data-dependent decay term)** 可能是主要原因。
   - *Lightning 的 detect_anomaly=True 设置在调试中非常有用*，通过梯度分析将问题追踪到了衰减项。
- **寻求社区的研究反馈**：一名博士生就使用扩散模型 (diffusion models) 进行压缩的研究项目想法寻求反馈，并询问发布位置。成员建议发布在 general 频道或指定的低压力讨论空间。
   - 另一位成员建议对网络输入进行正则化损失以维持稳定性，同时强调在讨论 SAE 时需要明确假设。
- **关于 SAE 中稀疏编码的讨论**：一位成员澄清说，在他们的 SAE 方法中，重建误差损失项应与侧重稀疏性的损失并存。即使输入是固定的，额外的损失也可以防止训练期间偏离预期分布。
   - 参与者强调了对 LLM 在 SAE 中作用的潜在误解，并强调来自冻结网络 (frozen networks) 的统计数据可以为编码过程提供必要的上下文。
- **探索 Vision Embedding 与 Token 的对比**：一位用户提出了关于在建模中使用 **vision embedding** 和 **vision token** 方法之间权衡的问题。社区承认了两者之间的差异，但指出目前尚不清楚每种技术的具体优势。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1278803733127495800)** (88 条消息🔥🔥): 

> - `Dynamic Expert Routing` (动态专家路由)
> - `Tokenization Challenges` (Tokenization 挑战)
> - `Multi-Token Prediction` (多 Token 预测)
> - `Finite Scalar Quantization` (有限标量量化)
> - `Symmetry in Neural Networks` (神经网络中的对称性)


- **Dynamic Expert Routing 讨论**：一位成员解释说，允许模型在训练期间定义自己的专家（Experts），相比于配置文件中固定的专家数量，可以增强灵活性。
   - 有人请求提供与此概念相关的论文，指出该领域需要更多的文献支持。
- **Tokenization 面临批评**：有观点认为 Tokenization 掩盖了重要的数据特征，并可能随着 Multi-Token Prediction 和并行 Transformer 的进步而变得过时。
   - 一位成员指出，当前的 Tokenization 方法可能会阻碍模型训练效率和整体模型性能。
- **探索 Multi-Token Prediction**：讨论了 Multi-Token Prediction (MTP) 在推理和规划任务中的有效性，一些成员对其在较小模型中的效用表示怀疑。
   - 有人提到，已发表的研究尚未减轻关于 MTP 在较小架构上性能表现的担忧。
- **引入 Finite Scalar Quantization**：FSQ 被强调为 VQ-VAE 中 Vector Quantization (VQ) 的一种有前景的替代方案，可能提高 Codebook 利用率和模型效率。
   - 成员们注意到 FSQ 在保持竞争力的指标的同时具有简洁性，使其成为进一步探索的极具吸引力的选择。
- **神经网络中的对称性探索**：一篇新论文讨论了损失函数中对称性的负面影响，这可能导致低容量状态，并提出了一种减轻该问题的方法。
   - 有人对在不同模型中应用这种新损失函数的实现成本和有效性提出了担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.00417">Rethinking Tokenization: Crafting Better Tokenizers for Large Language Models</a>：Tokenization 显著影响语言模型 (LM) 的性能。本文追溯了分词器从词级到子词级的演变，分析了它们如何平衡 Token 和类型...</li><li><a href="https://arxiv.org/abs/2408.15495">Remove Symmetries to Control Model Expressivity</a>：当损失函数中存在对称性时，模型很可能陷入有时被称为“坍缩”的低容量状态。陷入这些低容量状态会导致...</li><li><a href="https://arxiv.org/abs/2408.16532">WavTokenizer: an Efficient Acoustic Discrete Codec Tokenizer for Audio Language Modeling</a>：语言模型已有效地应用于自然信号建模，如图像、视频、语音和音频。这些模型的一个关键组件是编解码分词器，它负责压缩高维...</li><li><a href="https://medium.com/@techsachin/layerskip-faster-llm-inference-with-early-exit-and-self-speculative-decoding-3110cb93c94e">LayerSkip: faster LLM Inference with Early Exit and Self-speculative decoding</a>：简介</li><li><a href="https://arxiv.org/abs/2309.15505">Finite Scalar Quantization: VQ-VAE Made Simple</a>：我们建议在 VQ-VAE 的潜变量表示中，用一种称为有限标量量化 (FSQ) 的简单方案取代矢量量化 (VQ)，我们将 VAE 表示投影到一个固定的...</li><li><a href="https://arxiv.org/abs/2310.05737">Language Model Beats Diffusion -- Tokenizer is Key to Visual Generation</a>：虽然 LLM 是语言生成任务的主导模型，但它们在图像和视频生成方面的表现不如扩散模型。为了有效地将 LLM 用于...</li><li><a href="https://arxiv.org/abs/2406.07548">Image and Video Tokenization with Binary Spherical Quantization</a>：我们提出了一种基于 Transformer 的新型图像和视频分词器，采用二进制球面量化 (BSQ)。BSQ 将高维视觉嵌入投影到低维超球面上，然后应用...
</li>
</ul>

</div>

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1279061226571169802)** (5 messages): 

> - `Word Game Bench`
> - `衡量多项选择题的一致性` 


- **介绍用于语言模型的 Word Game Bench**：开发了一个名为 **Word Game Bench** 的新基准测试，用于评估语言模型在文字拼图游戏（特别是 **Wordle** 和 **Connections**）上的表现。
   - 目前没有模型的平均胜率超过 **50%**，该基准测试旨在通过交互和反馈而非静态响应来测试模型。
- **衡量一致性的挑战**：一位成员正试图比较多项选择题的回答，以衡量当 Prompt 略有变化时的一致性，并针对同一问题在不同 Prompt 下使用扩展数据集。
   - 建议包括使用 `doc_to_target` 或 `doc_to_text` 等函数创建代表所需比较的数据集，尽管这需要为每个模型投入相应的努力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://wordgamebench.github.io">Word Game Bench</a>: 未找到描述</li><li><a href="https://x.com/zafstojano/status/1829398835585520076">来自 zafir (@zafstojano) 的推文</a>: 很高兴分享 “Word Game Bench” —— 一个用于评估语言模型在文字拼图游戏上表现的有趣基准测试！这是一个相对困难的基准测试，目前没有模型的平均得分超过 50%...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1278814192404664381)** (1 messages): 

> - `Discord 服务器增长` 


- **Discord 服务器成员突破 10 万**：Discord 服务器现已达到 **10 万成员** 的惊人里程碑，展示了社区的蓬勃发展。
   - 向 **@everyone** 表示由衷的感谢，感谢大家的支持和反馈，并强调了团队对继续共同进化的兴奋之情。
- **感谢社区的支持和反馈**：团队对成员们提供的所有 **支持和反馈** 表示感谢，进一步强调了社区参与的重要性。
   - 他们强调渴望与每一位参与者一起 *继续成长和进化*。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1278814360260710411)** (120 条消息🔥🔥): 

> - `Perplexity Pro 订阅问题`
> - `AI 模型性能`
> - `Perplexity 宣传材料`
> - `AI 竞赛与 Hackathons`
> - `图片上传的用户体验` 


- **关于 Pro 订阅消失的讨论**：多位用户对他们的 Pro 订阅消失表示担忧，潜在原因包括促销代码（promo codes）的误用和账户问题。支持团队建议针对未解决的订阅访问问题通过电子邮件进行联系。
   - 一位用户特别提到兑换了促销代码后发现其失效，从而引发了关于现行代金券滥用防范措施的疑问。
- **关于模型选择与性能的讨论**：用户注意到在不同 AI 模型之间切换有时会得到相似的答案，这引发了对模型差异化的怀疑。有人建议所使用的模型可能没有进行有效的区分，这可能是由于最近的更新导致的。
   - 一位参与者特别指出，询问模型类型（GPT 或 Claude）时，返回的是关于 Perplexity 模型的通用回答，而不是所选模型的具体信息。
- **在 AI 展览中展示 Perplexity**：一位正在法国组织 AI 展览的用户请求提供宣传材料和视频资源，以便在活动中有效地展示 Perplexity AI。他们明确表示需要 YouTube 内容以外的材料来丰富演示。
   - 这表明用户有兴趣通过协调良好的营销努力来增强用户参与度以及对 Perplexity AI 的理解。
- **对帖子（Thread）删除的担忧**：用户对帖子意外消失表示担忧，一些人坚持认为他们从未删除过自己的帖子。这引发了关于平台可能自动删除帖子的讨论，引起了成员们的不满。
   - 一位用户强调刷新浏览器导致他们的帖子丢失，要求澄清删除政策。
- **关于图片上传功能的咨询**：多位用户报告了上传图片的问题，引发了对使用该功能时遇到的技术困难的讨论。解决方案表明，禁用某些浏览器扩展程序有助于恢复功能。
   - 这反映了为维持对平台功能的全面访问而持续进行的技术排障和用户调整。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rohansai22.github.io/resume/">Maragoni Rohan Sai - Portfolio</a>：未找到描述</li><li><a href="https://tenor.com/view/griffith-berserk-eclipse-guts-berserk-anime-meme-gif-10622855093064880455">Griffith Berserk GIF - Griffith Berserk Eclipse - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1278946013813542966)** (10 条消息🔥): 

> - `MrBeast 新闻`
> - `C++ 编程`
> - `维京人对现代文化的影响`
> - `OpenAI 的 DALL-E`
> - `肌肉结节` 


- **MrBeast 怎么了？**：成员们通过这个[链接](https://www.perplexity.ai/search/what-happened-to-mrbeast-S0hJBJ01TSKV6CqiLDXnvw)讨论了围绕 **MrBeast** 的最新动态。
   - 讨论暗示了人们对他目前活动和状态的极大兴趣。
- **对 C++ 程序的需求**：一位用户在[C++ 编程任务](https://www.perplexity.ai/search/write-a-c-plus-plus-program-fo-aJscZujqQZGLq2_8THGP5A)上寻求帮助。
   - 这反映了用户对学习和提高编程技能的持续兴趣。
- **维京人正流行**：一位成员在这次[讨论](https://www.perplexity.ai/search/what-have-vikings-done-for-mod-Cb_PHCx7Ty2cDQZVa14iJA)中探索了**维京人**对现代文化的影响。
   - 这种热情表明人们对历史影响的迷恋日益加深。
- **探索 DALL-E 的功能**：分享了一个关于 **OpenAI DALL-E** 及其功能的链接，详见[此处](https://www.perplexity.ai/search/openai-s-dall-e-0eZkD0GfRliPUTnsBpKBIQ)。
   - 成员们正积极关注并理解 AI 的创造性输出。
- **了解肌肉结节**：一位用户通过这个[链接](https://www.perplexity.ai/search/what-are-muscle-knots-also-kno-.GsfiArjRTW.5wmBcUIYtA)发布了关于“肌肉结节”的查询。
   - 这表明了学习更多关于健康和身体机能知识的愿望。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1278811459739979888)** (9 条消息🔥): 

> - `PPLX API Credits`
> - `Perplexity Pro Searches Availability`
> - `Rate Limiting Issue` 


- **用户未收到 PPLX API 额度**：多位用户反馈在购买 Pro 后，未收到承诺的 **$5 PPLX API 额度**。
   - *一名用户请求提供账户详情以进一步调查该问题*，但目前尚未提供解决方案。
- **API 不支持 Pro Searches**：一名用户对在调用 API 时 **Pro Searches** 的运作方式表示不确定。
   - 另一位成员澄清说 **Pro 无法通过 API 使用**，并询问该功能未来是否会推出。
- **速率限制（Rate limiting）担忧**：一名用户在多次调用 API 端点时遇到了 **429 Client Error: Too Many Requests**。
   - 他们询问为何在脚本中仅使用了三个函数的情况下就立即触发了 **insta rate limited**。


  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1278792607710969916)** (70 条消息🔥🔥): 

> - `Command R+ Model Updates`
> - `Throughput and GQA Impact`
> - `Cohere Scholars Discord Access`
> - `MMLU Optimization Discussion`
> - `Open Weights and Licensing` 


- **Command R+ 模型表现出重大改进**：包括 `command-r-08-2024` 在内的更新模型的推出，在多语言检索增强生成 (RAG)、数学和推理任务中表现出更强的性能。
   - 用户注意到，由于 GQA 的增强，吞吐量显著增加，效率比之前的模型提高了多达 50%。
- **关于 MMLU 相关性的讨论**：Nick Farst 表示 MMLU 与实际应用的相关性不强，并指出其很大一部分内容已经过时且无关紧要。
   - 成员们对 MMLU 表现出冷淡态度，强调需要关注模型性能中更实际的方面。
- **用户期待更新和功能**：成员们渴望了解 Cohere 将带来的新更新，特别是 GQA 对模型性能的影响以及潜在的量化方法。
   - 讨论显示出社区对性能指标的兴趣，以及对新旧模型进行定量比较的愿望。
- **访问 Cohere Scholars Discord**：一名用户询问如何加入 Cohere Scholars Discord，并得到指引访问 [Cohere For AI 网站](https://cohere.com/research) 获取加入链接。
   - 这反映了持续的社区参与以及对参与平台讨论的兴趣。
- **开放权重讨论**：针对模型权重开放的性质进行了澄清，指出虽然大多数权重可用，但最新的更新尚未完全开源。
   - 成员们分享了访问 Hugging Face 开放权重的链接，并思考了这对学术和研究友好型使用的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>：未找到描述</li><li><a href="https://docs.cohere.com/">Cohere Documentation — Cohere</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/safety-modes">Safety Modes — Cohere</a>：安全模式文档描述了如何使用默认模式和严格模式，以便对模型输出进行额外控制。</li><li><a href="https://cohere.com/blog/command-series-0824">Updates to the Command R Series</a>：Command R 模型系列的最新版本在编码、数学、推理和延迟方面都有所改进。</li><li><a href="https://huggingface.co/datasets/joey234/mmlu-human_sexuality-original-neg">joey234/mmlu-human_sexuality-original-neg · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://cohere.com/blog/">The Cohere Blog</a>：探索我们收集的富有洞察力的博客文章，涵盖各种生成式 AI 主题。我们的文章提供深入分析、专家意见和实用建议，以提供信息和启发。 </li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus-08-2024">CohereForAI/c4ai-command-r-plus-08-2024 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1279078308243439656)** (6 messages): 

> - `Command R models`
> - `Pricing updates`
> - `Hugging Face availability`
> - `Fine-tuning defaults`
> - `Ollama deployment` 


- **Command R 和 R+ 模型刚刚发布！**: 更新后的 **Command R** 和 **R+** 模型现在在推理、编程、工具使用和多语言 RAG 方面都有性能提升，并具有更低的延迟和新的安全模式。
   - 可以使用别名 `command-r-08-2024` 和 `command-r-plus-08-2024` 访问更新后的模型。
- **公布新价格结构**: **Command R** 的 Token 价格现在为输入 **$0.15**，输出 **$0.60**，而 **R+** 已降至输入 **$2.50**，输出 **$10.00**，使其更具性价比。
   - 这些显著的降价包括 R 的输入价格降低了 **3 倍**，R+ 的输入价格降低了 **30%**。
- **模型现已在 Hugging Face 上可用**: 用户可以在 **Hugging Face** 上找到更新后的模型，并期待它们被转换并上传到 **Ollama**。
   - *Vincentbosch* 指出，这些模型在 **Ollama** 上完全准备就绪还需要一些时间。
- **关于微调默认设置的问题**: 讨论了新模型是否将作为 **微调的新默认设置**，暗示了标准做法的转变。
   - 目前尚未提供确切答案，留给成员们猜测的空间。
- **基准测试请求**: 一位用户询问了新模型 **benchmarks** 的可用性，以便更好地了解其性能指标。
   - 截至目前，这些基准测试尚未发布，引发了对其结果的进一步关注。



**提及的链接**: <a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: 未找到描述

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1278833174037925909)** (10 messages🔥): 

> - `C4AI Scholars Program`
> - `Command R+ Release`
> - `GDPR Compliance`
> - `Cohere's Trust Center` 


- **面向研究生的 C4AI Scholars Program**: 一位成员询问 **C4AI Scholars Program** 是否接受在读研究生，可能通过 1 月份的实习安排。
   - 另一位成员建议直接联系 **C4AI** 以澄清此事。
- **关于 Command R+ 发布的咨询**: 一位用户询问最新版本的 **Command R+** 是否即将发布，表达了对新进展的兴趣。
   - 尚未提供任何回复来澄清发布时间表。
- **Cohere API 的 GDPR 合规性**: 有人询问 **Cohere** API 是否符合 **GDPR** 法规，特别是关于 Command R+ 的数据使用。
   - 后续回复包括来自 **Cohere Trust Center** 的帖子，其中可能包含有用信息。
- **Cohere Trust Center 资源**: 一位成员分享了 **Cohere Trust Center** 的链接，强调其对数据机密性、完整性和可用性的重视。
   - 对于担心合规性和数据处理的用户来说，这似乎是一个非常有用的资源。



**提及的链接**: <a href="https://cohere-inc.secureframetrust.com/">  Cohere Inc | Trust Center
</a>: 未找到描述

  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1279021162814505012)** (46 条消息🔥): 

> - `试用版 API Key 的速率限制问题`
> - `重排序引用 (Reranking Citations)`
> - `安全模式 (Safety Mode) 与 Preamble 的交互`
> - `金融数据分析中的引用` 


- **试用版 API Key 超出速率限制**: 一位用户在使用试用版 API Key 时遇到了速率限制错误（Error 429），该 Key 每月仅允许 **1000 次 API 调用**。
   - 另一位用户解释说，若要提升容量，需要升级到生产环境 Key，这需要提供信用卡详情。
- **通过重排序引用来减少冗余**: 一位用户就如何限制生成文本中的引用数量寻求建议，因为在其 **180 字的输出**中引用数量过多。
   - 建议包括对引用进行重排序以仅显示**前 3 个**，或者专注于显示参考文档而非行内引用。
- **安全模式与 Preamble 之间的交互**: 有人提问在 API 使用中 **safety_mode** 是否会覆盖自定义的 **preamble** 设置。
   - 澄清说明安全模式是独立运行的，执行“安全指令”，并使用结构化格式在 **STRICT** 或 **CONTEXTUAL** 模式下呈现 Prompt。
- **金融数据分析应用的开发**: 一位用户正在开发一款金融数据分析应用，并提到引用对于其输出的重要性。
   - 该应用仍处于早期阶段但展现出潜力，其他用户表示愿意为其开发提供帮助。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: 此页面描述了 Cohere API 的相关限制。</li><li><a href="https://docs.cohere.com/reference/rerank">Rerank — Cohere</a>: 该端点接收一个查询和一个文本列表，并生成一个为每个文本分配了相关性得分的有序数组。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1279088423038222451)** (1 条消息): 

> - `Maya LLaVA-Pretrain 数据集`
> - `多语言数据集特性`
> - `翻译质量结果`
> - `API 支持与批处理` 


- **Maya LLaVA-Pretrain 已上线！**: **Maya LLaVA-Pretrain** 数据集已发布，包含跨 **8 种语言**的 **4,404,776** 条条目，专为预训练大型语言与视觉模型而设计。
   - 该数据集通过机器翻译和毒性过滤，从原始的 [LLaVA-pretrain 英语数据集](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain) 扩展而来，适用于图像描述 (image-captioning) 和视觉问答 (visual question-answering) 任务。
- **访问数据集需要同意协议**: 要访问 **Maya LLaVA-Pretrain 数据集**，用户必须登录或注册以同意该仓库要求的共享条件。
   - 尽管数据集是公开访问的，但这一步是必要的，以确保符合使用政策。
- **对 API 支持的感谢**: 一位用户对团队成员在数据集准备期间提供的**批处理 (batch processing)**和 **API 支持**表示感谢。
   - 他们强调了与 **c4ai-aya-35B** 模型 API 以及 **command-r-plus** API 在优化毒性过滤方面的合作。
- **即将发布的翻译质量结果**: 团队计划近期在数据集卡片 (dataset card) 上展示**翻译质量结果**，以增强数据集的可信度。
   - 这是继数据集准备工作之后的后续步骤，重点展示在机器翻译工作中所取得的进展。



**提及的链接**: <a href="https://huggingface.co/datasets/kkr5155/Maya-llava-pretrain">kkr5155/Maya-llava-pretrain · Datasets at Hugging Face</a>: 未找到描述内容内容

  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1278795696421732435)** (31 条消息🔥): 

> - `Codeium 融资更新`
> - `Meta AI 助手增长`
> - `DeepMind 的可定制 Gems`
> - `代码生成工具的演进`
> - `Tome 转向企业级 AI 辅助` 


- **Codeium 获得 1.5 亿美元 Series C 融资**：Codeium 宣布完成 **1.5 亿美元** 的 Series C 融资，公司估值达到 **12.5 亿美元**，且尚未动用今年 1 月份的 **Series B** 资金。
   - 随着总融资额达到 **2.43 亿美元**，联合创始人旨在利用这笔资金加速 **R&D** 和增长计划。
- **Meta AI 助手用户数达到惊人规模**：Meta 的 AI 助手已拥有 **4 亿月活跃用户** 和 **4000 万日活跃用户**，显示出极快的普及速度。
   - 讨论表明，他们可能很快需要寻求许可，这反映了该平台日益增长的关注度和使用量。
- **Google DeepMind 推出可定制的 Gems**：DeepMind 推出了 **Gems**，这是一种可定制的 AI 聊天机器人，可以在各种场景中充当领域专家，包括 **Learning Coach**（学习教练）和 **Coding Partner**（编程伙伴）。
   - 批评者指出，能否获得青睐将很大程度上取决于这些工具的易用性和内容策展。
- **关于代码生成工具现状的讨论**：代码生成工具的最新进展（如 **Claude 3.5 Sonnet** 和 Townie 的重新设计）正在开启通过与 AI 对话构建软件的新方法。
   - 尽管热度很高，但人们仍然担心这些工具主要充当华丽的演示（demos），而缺乏与现有代码库的深度集成。
- **Tome 转型专注于企业级 AI 辅助**：Tome 正在进行品牌重塑，致力于成为专门的 AI 助手，旨在帮助企业开拓新的企业级客户。
   - 这一转型标志着公司在 AI 领域重新定位时的重大战略转变。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/changelog/command-gets-refreshed">Command models get an August refresh — Cohere</a>: 未找到描述</li><li><a href="https://blog.val.town/blog/codegen/">How we built Townie – an app that generates fullstack apps</a>: 类似 Claude Artifacts，但带有后端和数据库</li><li><a href="https://x.com/GoogleDeepMind/status/1828855383131074997">Tweet from Google DeepMind (@GoogleDeepMind)</a>: 在接下来的几天里，开始创建并与 Gems 聊天：Gemini 的可定制版本，充当特定领域的专家。🤝 我们还为不同场景推出了预设的 Gems - 包括 Learni...</li><li><a href="https://x.com/1x_tech/status/1829567690681307284?s=46">Tweet from 1X (@1x_tech)</a>: 介绍 NEO Beta。为人而设计。为家而造。</li><li><a href="https://x.com/AravSrinivas/status/1829261003164696703">Tweet from Aravind Srinivas (@AravSrinivas)</a>: 令人印象深刻的数据</li><li><a href="https://x.com/hliriani/status/1829284172470620613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Henri Liriani (@hliriani)</a>: 我们正在重启 Tome，使其成为一家不同的公司。@magicaltome 现在是一个用于开拓新企业账户的 AI 助手。这里有关于我们历程的一点分享……</li><li><a href="https://www.1x.tech/androids">Our Androids | 1X Technologies</a>: 受人类本性启发。认识 EVE 和 NEO，了解它们如何利用 embodied learning 来解决问题，从满足劳动力需求到日常辅助。</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1829541138736509102">Tweet from Ahmad Al-Dahle (@Ahmad_Al_Dahle)</a>: 紧随我昨天分享的 Llama 更新之后，我们也看到 Meta AI 的使用量增长飞快，周活跃用户达到 1.85 亿！🚀</li><li><a href="https://techcrunch.com/2024/08/29/github-copilot-competitor-codeium-raises-150m-at-a-1-25b-valuation/">GitHub Copilot competitor Codeium raises $150M at a $1.25B valuation | TechCrunch</a>: Codeium，一家开发 AI 驱动工具以对抗 GitHub Copilot 的初创公司，以 12.5 亿美元的估值融资 1.5 亿美元。</li><li><a href="https://techcrunch.com/2024/0">2024 | TechCrunch</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1278805134695989333)** (1 条消息): 

> - `Latent Space Podcast`
> - `LLM Benchmarks`
> - `Meetup 公告` 


- **关于 LLM Benchmarks 的新剧集**：[Latent Space Podcast](https://x.com/latentspacepod/status/1829173832877519152) 的最新一期邀请了来自 **Google DeepMind** 的 Nicholas Carlini，讨论编写你自己的 **LLM benchmarks** 的重要性。
   - 他涵盖了诸如*他如何使用 AI*、他的大语言模型 benchmark 以及*从 LLM 中提取训练数据*等话题，并强调了 OpenAI 缺失 **logprobs** 的问题。
- **下个月的 Meetup 预告**：提到了由 **@213041230177763329** 组织的即将于下个月举行的 Meetup。
   - 提到了 Meetup 的细节，鼓励听众参与。



**提到的链接**：<a href="https://x.com/latentspacepod/status/1829173832877519152">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 为什么你应该编写自己的 LLM benchmarks，嘉宾：@GoogleDeepMind 的 Nicholas Carlini。涵盖了他的热门内容：- 我如何使用 AI - 我的大语言模型 benchmark - 提取训练数据...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1279169268168265750)** (57 条消息🔥🔥): 

> - `STORM 方法对比 one-shot 研究论文生成`
> - `屏幕共享查看问题`
> - `Research Agent 的有效性`
> - `CogVLM 讨论`
> - `基于语言的学习策略` 


- **关于 STORM 方法或 One-shot 生成的辩论**：一位成员表示更倾向于 **STORM 方法**，认为 one-shot 生成研究论文感觉太脆弱，且过度依赖繁琐的人工验证。
   - 另一位成员反驳道，将生成过程拆解以获取反馈，可能比不间断地运行整个过程产生更好的结果。
- **屏幕共享查看困惑**：成员们遇到了屏幕共享的问题，有些人能看到 Yikes 的屏幕，而其他人则面临加载问题。
   - 这引发了关于查看差异的讨论，这种一个人能看而其他人不能看的情况并不常见。
- **对 Research Agent 局限性的担忧**：参与者讨论了 Research Agent 潜在的缺点，提到了之前的经验以及对其能力的担忧。
   - 一位成员指出，平均研究耗时约 **2 分钟**，成本约为 **$0.005**，这表明可能存在效率低下的问题。
- **CogVLM 引起未来讨论的兴趣**：社区发现 **CogVLM** 非常有趣，在强调了其 GitHub 上的特性后，提到需要有人就此进行演讲。
   - 一位成员将其在生成的论文中的表现比作“LLM barf”（LLM 呕吐物），暗示技术与其应用之间存在脱节。
- **探索使用 LLM 学习新代码库**：一位成员表示他们有能力展示 **Langflow** 的实际应用，重点是如何利用 **LLM** 来学习新的代码库。
   - 他们建议这个元话题（meta topic）可以促进参与者之间更深入的讨论和理解。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://storm.genie.stanford.edu/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/jimmykoppel/status/1828077206204981423">来自 Jimmy Koppel (@jimmykoppel) 的推文</a>：但所有这些都是为了阻止你仔细观察他们到底在做什么。因为我不认为那里有什么实质内容。</li><li><a href="https://github.com/THUDM/CogVLM">GitHub - THUDM/CogVLM: a state-of-the-art-level open visual language model | 多模态预训练模型</a>：一个达到 state-of-the-art 级别的开源视觉语言模型 | 多模态预训练模型 - THUDM/CogVLM</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1278890172225683557)** (34 条消息🔥): 

> - `Mojo 与 Web3 应用`
> - `Mojo 的开源状态`
> - `编程语言性能对比`
> - `MAX SDK 与许可协议`
> - `与 OPENSEA 的合作` 


- **Mojo 在 Web3 领域仍处于成熟期**：讨论显示，虽然区块链协议正在考虑使用 Mojo，但与 Go、Rust 和 C++ 相比，它在严肃开发方面仍显稚嫩。
   - 成员们注意到 Mojo 的 IO 和网络 API 正在开发中，需要进行调整以匹配现代硬件。
- **Mojo 编译器的开源状态**：虽然据报道 Mojo 是一种开源语言，但其编译器的源代码目前尚未公开，因为一个小团队正负责其快速迭代。
   - 关于编译器未来何时或是否会开源仍存在不确定性。
- **编程语言性能辩论**：成员们辩论了 Go 与 C 的性能，关于 Go 变慢了多少有不同的报告，这影响了它在某些应用中的适用性。
   - Darkmatter 强调，Go 优化器的保守方法在处理复杂问题时可能导致显著的性能缺陷。
- **MAX SDK 开发见解**：Modular 团队正在平衡开发速度、产品许可和社区参与，重点是开源贡献。
   - 虽然承认需要扩大团队，但寻找同时熟悉 MLIR 和 Mojo 的人才仍然是一个挑战。
- **与 OPENSEA 的 Free Mint 合作**：成员们获悉了与 OPENSEA 合作进行的新 Free Mint 活动，鼓励通过申领链接参与。
   - 虽然一些用户表示感兴趣，但其他人选择了不参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/legal/max-mojo-license">Modular: MAX &amp; Mojo 社区许可协议</a>：MAX SDK ("MAX") &amp; Mojo 社区许可协议规定了我们允许如何使用我们的软件，以及你如何利用它改变世界。</li><li><a href="https://www.modular.com/company/career-post?4419827005&gh_jid=4419827005)">Modular: 招聘职位</a>：在 Modular，我们相信优秀的文化是创建伟大公司的关键。我们的三大支柱是：打造用户喜爱的产品、赋能人才、成为一支不可议的团队。</li><li><a href="https://docs.modular.com/max/faq#will-it-be-open-sourced">MAX 常见问题解答 | Modular 文档</a>：关于 MAX Engine 预期问题的解答。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1278792965090967552)** (15 条消息🔥): 

> - `内存管理与架构`
> - `软件设计中的错误`
> - `Mojo 中的查找表 (Lookup Tables)`
> - `元组索引的错误处理`
> - `编程中的类型感知` 


- **架构师在内存管理中的角色**：一位成员表示，如果程序员需要担心特定内存是否被释放，那么这反映了架构师在系统/环境设计上的失败。
   - 他们强调开发团队需要创建一个充分的开发环境，以避免此类担忧。
- **拥抱软件开发中的错误**：一位成员指出，软件设计中的错误是不可避免的，并强调了架构灵活性的重要性。
   - 他们总结道，团队不应追求完美无瑕的设计，而应专注于适应失败。
- **为 Mojo 生成查找表**：一位成员分享了他们创建一个脚本来生成包含自定义查找表的 `.mojopkg` 文件的兴奋之情。
   - 他们对添加的功能表示喜悦，展示了对该项目的热情。
- **错误与元组边界感知**：有人担心元组的越界错误可能会因为每个索引处的类型感知而导致编辑器中的混淆。
   - 讨论暗示这些情况缺乏清晰的错误消息，同时也承认其在大多数情况下是有效的。
- **错误消息中对 InvalidType 的需求**：有人建议引入 `InvalidType` 以改进类型不匹配时的错误处理机制。
   - 这被认为在处理 `Type != Type` 错误消息时可能非常有益。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1279140100516741121)** (2 messages): 

> - `fastai model export`
> - `Modular framework agnostic model format` 


- **Fastai 的模型导出建议**：讨论强调了 **fastai** 允许用户使用 `Learner.export` 命令导出训练好的 **PyTorch model** 以用于生产环境部署。
   - 一位成员建议，如果能重写 `Learner.export` 以便在导出模型的同时生成用于输入流水线（input pipeline）的 **Mojo code**，那将会非常酷。
- **Modular 解决 Pickle 问题的雄心**：一位成员指出，**Modular** 似乎准备通过创建一种**跨平台、框架无关的模型格式**来解决 **pickle problem**。
   - 这暗示了在不同平台间标准化模型部署的趋势。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1278800835765338175)** (46 messages🔥): 

> - `LangChain with Docker`
> - `ChatOllama vs Ollama`
> - `Real-time streaming in LangChain`
> - `Using Hybrid RAG models`
> - `Building a competent GPT for HR` 


- **LangChain 应用在 Docker 中失败**：一位用户报告了他们的 LangChain 应用在容器化时调用 ChatOllama 对象出现问题，但在非 Docker 环境下运行正常。
   - 该问题被确定与基础 URL（base URL）有关，通过改用直接的 Ollama 主机 URL 得到了解决。
- **ChatOllama 与 Ollama 的区别**：ChatOllama 专为聊天类交互设计，而 Ollama 则用于涉及语言模型的通用任务，并提供针对其用例定制的特定功能。
   - 提供了如何使用这两种模型的详细信息，以及展示它们各自 API 参考的示例。
- **实时流式输出问题**：一位用户在使用其 Agent 执行器时遇到困难，该执行器汇总了所有输出而不是实时进行流式传输。
   - 另一位成员询问了 `streamRunnable = False` 的行为，以了解其对流式传输能力的影响。
- **用于 LLM 训练的混合 RAG 模型**：讨论强调了尽管 LLM 无法进行实时学习，但通过反馈和微调实现持续改进的可能性。
   - 还提到了传统的 RAG 模型和自查询（self-query）技术等替代方案，作为增强性能的可行选项。
- **为人力资源部门构建胜任的 GPT**：一位用户表示希望为其 HR 团队庞大的手册创建一个专门定制的 GPT 模型，并强调该模型不能产生幻觉（hallucinate）。
   - 建议包括使用优秀的 RAG 技术，并通过基于反馈的迭代调整来增强模型性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://v02.api.js.langchain.com/classes/langchain.agents.AgentExecutor.html">AgentExecutor | LangChain.js</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/25022>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 开发做出贡献。</li><li><a href="http://ollama:11434">)">未找到标题</a>: 未找到描述</li><li><a href="https://api.python.langchain.com/en/latest/agents/langchain.agents.agent.AgentExecutor.html">langchain.agents.agent.AgentExecutor &mdash; 🦜🔗 LangChain 0.2.16</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/6398">通过 Docker 运行 Ollama 时，它不会响应任何 API 调用或 Python 客户端库的请求 · Issue #6398 · ollama/ollama</a>: 问题是什么？我在装有 RTX-4000 的 Ubuntu 22 机器上成功安装了 nvidia docker toolkit，并将 ollama 作为 docker 容器启动，暴露端口 11434：docker run -d --gpus=all --en...
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/)** (1 messages): 

来源发现: https://www.getaiphone.app/
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1278831898885357568)** (5 messages): 

> - `GymNation Success Story`
> - `LLMs in Production Talk`
> - `LlamaIndex & MLFlow Integration`
> - `LLM x Law Hackathon`
> - `Enhanced Financial Data Analysis` 


- **GymNation 的数字化转型胜利**：GymNation 显著提升了其会员体验，将**数字化线索到销售的转化率提高了 20%**，并实现了 **87% 的数字化线索对话率**。
   - 他们与 LlamaIndex 的合作推动了**真实的业务成果**，详见其最新的 [成功案例](https://twitter.com/llama_index/status/1829272433687470489)。
- **9 月 9 日探索生产环境中的 LLMs**：在 **9 月 9 日** 即将举行的活动中，听取 @seldo 讨论生产环境中的 **Large Language Models**。
   - 详情请见 [Twitter](https://twitter.com/llama_index/status/1829304873210491392) 上的公告。
- **LlamaIndex 与 MLFlow 集成**：联合创始人 @jerryjliu0 在播客中分享了关于我们**与 MLFlow 全新集成**的见解，增强了用户记录和评估 LlamaIndex 应用的方式。
   - 此次集成可以更好地对 ML 模型进行**追踪、评估和部署**，完整的演示可在 [线上](https://twitter.com/llama_index/status/1829569770364227895) 查看。
- **参加 9 月 8 日的 LLM x Law 黑客松**：由 @hexapode 组织的 **LLM x Law 黑客松** 备受期待，将于 **9 月 8 日** 探索 AI 与法律的交汇点。
   - 参与者可以期待专注于 AI 开发和法律应用的三个赛道，更多信息分享在 [Twitter](https://twitter.com/llama_index/status/1829594190705201570) 上。
- **通过 MoW 和 RAG 增强财务分析**：这种新方法结合了 **Mixture of Workflows (MoW)** 和 **Corrective RAG**，用于彻底的财务数据分析，使用了 Phi-3 和 Qwen-2 等模型。
   - 该方法实现了**财务报表的上下文感知分析**，如分享的 [详情](https://twitter.com/llama_index/status/1829615136505795064) 中所述。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1278804321340756092)** (28 messages🔥): 

> - `LlamaIndex Warning`
> - `Query Engines Deprecation`
> - `Llama3 LLM Usage`
> - `Handling JSON in LLM`
> - `Azure OpenAI Integration Issues` 


- **关于有效配置键的 LlamaIndex 警告**：一位用户报告收到了关于 LlamaIndex V2 中配置键更改的警告，特别提到了 'allow_population_by_field_name' 和 'smart_union'。
   - 另一位用户建议该警告可能与正在使用的 SQLAlchemy 版本有关。
- **关于 Query Engines 弃用的担忧**：一位用户根据文档对 QueryEngines 可能被弃用表示担忧，文档中提到了 RAG 工作流的弃用方法。
   - 其他人澄清说，只有特定的结构化输出方法被弃用，而核心查询引擎（Query Engines）不受影响。
- **使用 Llama3 LLM 进行 API 调用**：一位用户询问如何配合 OpenAI 使用 Llama3 来执行 `generate_qa_embedding_pairs`，寻求明确的指导。
   - 另一位用户建议全局定义 LLM 以保持使用一致性，或者在函数调用期间将其作为关键字参数传递。
- **在 LLM 工作流中处理 JSON 数据**：一位用户分享了他们在将外部 API 的 JSON 输出成功集成到 LLM 时遇到的困难。
   - 他们得到的建议是在将 JSON 响应传递给 LLM 之前对其进行适当的格式化，以避免复杂化。
- **Azure OpenAI 集成问题**：一位用户对 LlamaIndex 和 Azure AI 的集成表示沮丧，称搜索结果中存在引用（citation）不匹配的问题。
   - 另一位用户反驳了这一说法，表示他们在生产环境中没有观察到类似问题，并鼓励贡献代码以修复任何潜在的 Bug。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/structured_outputs/query_engine/">(已弃用) Query Engines + Pydantic Outputs - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Function Calling Agent 的工作流 - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1278950929756065876)** (1 messages): 

> - `LitServe`
> - `LlamaIndex`
> - `AI Model Deployment` 


- **使用 LitServe 实现闪电般的 AI 模型服务**：文章介绍了 **LitServe**，这是一个高性能的推理引擎，允许开发者高效地部署和管理各种 AI 模型。
   - 它强调了 LitServe 与 **LlamaIndex** 的结合，增强了构建智能应用的多功能性。
- **LitServe 与 LlamaIndex 的集成**：当与 **LlamaIndex** 配合使用时，LitServe 释放了构建稳健 AI 应用的新潜力。
   - 这种组合为开发者提供了改进的工具和资源，以进行有效的 AI 模型管理。



**提到链接**: <a href="https://medium.com/ai-artistry/serving-ai-models-at-lightning-speed-with-litserve-and-llamaindex-4e7decdb5ae1">Serving AI Models at Lightning Speed with LitServe and LlamaIndex</a>: Ankush k Singal

  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1278797119464935457)** (11 messages🔥): 

> - `House Party Announcement`
> - `Terminal Applications for KDE`
> - `Obsidian OI Plugin Issues`
> - `GPT-4o Memory Concerns` 


- **参加下周的 House Party！**：一名成员宣布下周将举办 [House Party](https://discord.gg/open-interpreter-1146610656779440188)，并坚持使用较早的时间以聚集更多参与者。
   - 邀请函包含了一条由衷的信息以鼓励大家参与。
- **征求 KDE 终端应用建议**：一名成员询问适用于 KDE 的终端应用，并指出当前的 **Konsole** 在滚动时会出现屏幕溢色（screen bleeding）问题。
   - 讨论涉及了各种终端应用的有效性，其中一人确认在标准终端中也存在类似问题。
- **Obsidian OI 插件问题解决**：一位用户称赞了关于 **Obsidian OI plugin** 的视频，但在安装时遇到了问题，正在寻求解决建议。
   - 另一名成员建议将详细的安装问题发布到特定频道，以便获得相关帮助。
- **GPT-4o 的遗忘问题**：一位用户对 **GPT-4o** 无法记住过去的交互表示沮丧，并询问如何利用它进行 Web 开发。
   - 他们幽默地建议向模型请教如何提高记忆能力的技巧。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1279126210349236278)** (2 messages): 

> - `Potential applications discussion`
> - `House party meetup` 


- **对潜在发展的期待**：一名成员对即将到来的发展表示渴望，并愿意参与其中。
   - 他们提到对潜在应用有一些想法，并希望有机会进一步讨论。
- **将 House Party 作为讨论场所**：另一名成员建议下周四的 House Party 将是讨论潜在应用的绝佳时机。
   - 这次非正式聚会旨在为更深入的交流提供机会。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1278795160880156793)** (3 messages): 

> - `GameNGen Neural Model`
> - `DOOM Simulation`
> - `Shout-out to AgentOps`
> - `YouTube Video Discussion` 


- **GameNGen 神经模型驱动实时游戏**：_GameNGen_ 神经模型完全实时地模拟了经典游戏 **DOOM**，在单个 TPU 上实现了超过 **20 fps** 的帧率，并产生了高质量的交互。
   - 下一帧预测的 **PSNR 达到 29.4**，人类评分者难以区分真实游戏画面与模拟画面。
- **对 AgentOps 团队未来的期待**：最近的讨论中强调了对 Adam 和 **AgentOps** 团队接下来的成就充满期待。
   - 一名成员对分享的见解以及围绕团队发展的积极热情表示感谢。
- **YouTube 视频致谢**：一段带有时间戳的 **YouTube 视频** 提到了社区成员，引起了聊天室的关注。
   - 贡献者强调该视频内容引人入胜，值得一看，并特别强调了其中的致谢环节。



**提到链接**: <a href="https://gamengen.github.io/">GameNGen</a>: Diffusion Models Are Real-Time Game Engines

  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1278886572242239559)** (14 messages🔥): 

> - `Google buying NVIDIA GPUs` (Google 购买 NVIDIA GPU)
> - `RunwayML deletes repos` (RunwayML 删除仓库)
> - `Effects on diffusers` (对 diffusers 的影响)
> - `Realistic image generation for novels` (小说的写实图像生成)
> - `Re-LAION-5B dataset update` (Re-LAION-5B 数据集更新)


- **Google 采购 GPU 引发疑问**：成员们质疑为什么 **Google 在已经拥有 TPU 的情况下仍在购买 NVIDIA GPU**，这暗示了潜在的性能考量。
   - *TPU 够用吗？* 在竞争日益激烈的环境下，这引发了对 Google 硬件策略的好奇。
- **RunwayML 清除 Stable Diffusion 仓库**：关于 **RunwayML 删除了他们在 HuggingFace 和 GitHub 上所有 Stable Diffusion 1.5 仓库**的讨论爆发了，这导致了现有项目的混乱。
   - 成员们表达了对 **diffusers 1.5** 功能受影响的担忧，一位成员指出这破坏了 **single file loading**（单文件加载）。
- **对删除仓库的沮丧**：成员们对 RunwayML 缺乏远见、在没有存档的情况下删除仓库感到恼火，这影响了各种依赖项。
   - 一位成员推测删除背后可能有*法律原因*，但没有发现引用的具体问题。
- **寻求写实的漫画封面**：一位成员分享了为**小说封面**生成合适图像的挑战，寻求实现更多**漫画或卡通风格**的方法。
   - 尽管尝试了 **DALL-E**，但他们得到的却是 AI 生成感极强的图片，说明了实现预期风格的困难。
- **发布 Re-LAION-5B 数据集**：**Re-LAION-5B 数据集**发布，这是 LAION-5B 的重要更新，解决了安全问题并删除了指向疑似 CSAM 的链接。
   - 与 **Internet Watch Foundation** 等组织的共同努力确保了数据集的完整性，现在有两个安全版本可供下载。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://laion.ai/blog/relaion-5b/">Releasing Re-LAION 5B: transparent iteration on LAION-5B with additional safety fixes | LAION</a>: &lt;p&gt;今天，在遵循&lt;a href=&quot;https://laion.ai/notes/laion-maintenance/&quot;&gt;安全修订程序&lt;/a&gt;后，我们发布了 Re-LAION-5B，这是 LAION 的更新版本...</li><li><a href="https://huggingface.co/runwayml">runwayml (Runway)</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[announcements](https://discord.com/channels/823813159592001537/826154622644649985/)** (1 messages): 

mega_b: https://laion.ai/blog/relaion-5b/
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1278793158058180700)** (10 messages🔥): 

> - `OpenAI Funding Round` (OpenAI 融资轮)
> - `Chatbot Wars` (聊天机器人大战)
> - `Meta AI Usage` (Meta AI 使用情况)


- **科技巨头关注 OpenAI 的新融资**：据 [Bloomberg](https://www.bloomberg.com/news/articles/2024-08-29/nvidia-has-held-discussions-about-joining-openai-s-funding-round) 报道，**三家市值最高的科技公司** Nvidia、Apple 和 Microsoft 正在讨论投资 OpenAI 的新一轮 **1000 亿美元融资**。
   - *很高兴看到一个非营利组织吸引如此大的兴趣，* 另一位成员评论道，强调了这一潜在投资的重要性。
- **ChatGPT 以庞大的用户群占据主导地位**：ChatGPT 拥有超过 **2 亿周活跃用户**，而据 [The Information](https://www.theinformation.com/articles/metas-ai-assistant-wins-millions-of-users-in-challenge-to-chatgpt?utm_source=ti_app&rc=c48ukx) 引用，Meta AI 据称正在取得进展，拥有 **4000 万日活跃用户**。
   - 讨论围绕 Meta AI 的使用方式是否与 ChatGPT 类似展开，一些人指出它在 **EU** 等地区不可用。
- **对聊天机器人竞争的期待**：成员们对 **Chatbot** 领域持续的竞争表示兴奋，引用了“聊天机器人大战已经开始”等评论。
   - 一位参与者指出，*在我看来这是预料之中的*，展示了对 AI 助手行业增长的信心。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/markgurman/status/1829233740704559182">Mark Gurman (@markgurman) 的推文</a>: Nvidia, Apple 和 Microsoft —— 三家市值最高的科技公司 —— 正在洽谈投资 OpenAI，作为该公司新一轮 1000 亿美元融资的一部分。https://www.bloomberg.com/news/articles...</li><li><a href="https://x.com/amir/status/1829248019910537470?s=46">Amir Efrati (@amir) 的推文</a>: 聊天机器人大战已经开始。ChatGPT：2 亿+ 周活。Meta AI 可能紧随其后（尽管尚不清楚人们的使用方式是相同的还是误打误撞！）https://www.theinformation.com/articles/m...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1278967107492778015)** (3 messages): 

> - `Tinygrad Cloud Service`
> - `System Prompts Impact` 


- **Tinygrad 推出高性价比云服务**：Tinygrad 宣布了一项新的云服务方案，每月仅需 **$60**，包含一块 **4090 GPU** 和 **500 GB** 云存储，比 **vast ai** 便宜 **3 倍**。
   - 用户可以继续在本地使用 tinygrad，同时受益于云端更快的操作，并承诺每个 **'TinyJit' 函数**只需 **一次往返（one roundtrip）**。
- **关于 System Prompts 和评估的咨询**：一位用户询问是否有研究论文探讨 **System Prompts** 对评估分数的影响，以及分数是否可以产生有意义的偏移。
   - 这表明人们对于理解 Prompt Engineering 如何影响 AI 模型性能结果的兴趣日益浓厚。



**提到的链接**：<a href="https://x.com/__tinygrad__/status/1829379908017238210?s=46">来自 tiny corp (@__tinygrad__) 的推文</a>：即将推出：CLOUD=1。只需 $60/月（比 vast ai 便宜 3 倍），我们将为您提供一块 4090 和 500 GB 的云存储。像往常一样在开发机上使用 tinygrad，但它在云端运行速度极快……

  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1278800875212636261)** (11 messages🔥): 

> - `QLoRA Memory Issues`
> - `Multi GPU Evaluation`
> - `Torch Version Compatibility`
> - `Illegal Memory Access Errors` 


- **QLoRA 触及显存限制**：一位成员对 **QLoRA** 的显存占用表示怀疑，认为 4 张 48GB GPU 应该足以支持训练。
   - 他们指出，在不使用 CPU offloading 的情况下，他们的配置在处理较短序列时就已接近显存极限。
- **关于多 GPU 评估支持的讨论**：一位成员询问 **TorchTune** 是否支持 **多 GPU 评估**。
   - 这引发了关于配置和性能预期的进一步讨论。
- **Torch 版本至关重要**：另一位成员澄清他们使用的是 **Torch 版本 2.4.0+cu124**，这引发了关于配置兼容性的疑问。
   - 他们指出，该版本可能会影响模型在不同配置下的表现。
- **训练期间出现非法内存访问**：一位成员报告在运行训练脚本时遇到 **illegal memory access** 错误，建议通过传递 **CUDA_LAUNCH_BLOCKING=1** 进行调试。
   - 他们还强调 CUDA 错误可能是异步报告的，这增加了排查问题的难度。


  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1278791464586969140)** (5 messages): 

> - `LinkedIn Auto Jobs Applier`
> - `DSPy community engagement`
> - `GitHub repo discussion` 


- **邀请 DSPy 社区加入变革**：一位成员兴奋地分享了一个 GitHub 仓库，表示他们想邀请 DSPy 社区加入围绕该项目的**变革**。
   - *笑，* 他们强调了社区参与对其项目的重要性。
- **LinkedIn Auto Jobs Applier 走红**：据报道，**LinkedIn Auto Jobs Applier** 的 GitHub 仓库正受到关注，**每天获得超过 2k 个点赞**。
   - 然而，人们对其功能性表示担忧，GitHub issues 显示该项目**仍有待完善**。
- **对项目测试的担忧**：一位成员询问该项目是否经过测试，暗示用户反馈表明其可能仍缺乏稳定性。
   - 这暗示随着关于项目可行性讨论的继续，仍需要进一步的改进。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1278853571194130434)** (5 messages): 

> - `DSPy: Prompt Optimization`
> - `Bay Area AI Meetup`
> - `AgentOps platform`
> - `Michael Ryan's Talk` 


- **与 Michael Ryan 共同参加 Bay Area AI Meetup**：Michael Ryan 将在 @AIconference 之后的 Bay Area AI 聚会上（地点在 [GitHub HQ](http://Bay.Area.AI)）发表演讲，讨论 **DSPy** 和 **LM Programs**，重点介绍如何构建可靠的 LLM 应用程序。
   - 他的演讲将涵盖 MIPROv2 优化算法，以及像对待传统软件系统一样严谨对待 LM Programs 的重要性，并强调测试和审计。
- **用于 LM Programs 的 MIPROv2**：Michael Ryan 在 DSPy 中引入了 **LM Programs** 的概念，探索了对 LLM 的可组合调用，从而增强可靠性并优化性能。
   - 他因其贡献获得了 ACL 2024 的 **Best Social Impact award**（最佳社会影响奖），说明了他在语言模型领域工作的意义。
- **AgentOps 平台介绍**：[AgentOps](https://agents.staf.ai/AgentOps) 提供了一套强大的工具，用于创建具有图表、监控和重放分析功能的 Agent，旨在消除 LLM 使用中的盲目性。
   - 该平台是开源的，通过其 [GitHub repository](https://github.com/AgentOps-AI/agentops) 促进协作改进。
- **DSPy 疑问与支持**：一位用户询问了发布有关 **DSPy** 使用疑问的合适频道，寻求社区支持的方向。
   - 这表明人们对参与 DSPy 功能以及可能为社区内的故障排除做出贡献有着浓厚的兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/ChiefScientist/status/1829231009344434400?t=wow3U2BluHEv16-MI2YcaQ&s=19">来自 Alexy 🤍💙🤍 (@ChiefScientist) 的推文</a>: 非常激动能在由 @github HQ 在旧金山 SOMA 举办的 @AIconference 后的 http://Bay.Area.AI 聚会上接待 Michael Ryan！DSPy: Prompt Optimization for LM Programs Michael Ryan, @Stanford...</li><li><a href="https://docs.google.com/spreadsheets/d/1VnOv_C0v_FgDeKuQBaGuMNsWgoWOpLkGbE_XS_2Vb3Q/edit?gid=0#gid=0">由 AgentOps.ai &amp; Agen.cy 提供的 Agent 数据库</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1279021509838639264)** (5 messages): 

> - `Axolotl GitHub Documentation`
> - `Training Hardware for Llama 70B`
> - `A6000 GPUs` 


- **请求 Axolotl GitHub 文档支持深色模式**：一位成员表示希望 Axolotl GitHub 文档提供 **dark mode** 选项，称浅色模式对眼睛不友好。
   - *这一更改将提高那些经常参考配置参数的用户的可用性。*
- **寻求 Llama 70B 训练的最佳硬件**：一位成员询问了目前全量训练 **Llama 70B** 模型的硬件要求，质疑几块 **A6000** GPU 是否足够。
   - 另一位成员给出了肯定的回答，建议 *3x A6000 GPU* 应该足以胜任该任务。
- **考虑使用 A6000 进行全权重训练**：一位成员对使用 A6000 GPU 训练 **70B full weight** 模型的影响表示惊讶。
   - *对话体现了对 LM 训练硬件能力的希望与怀疑的交织。*


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1278894713742561300)** (1 messages): 

> - `Assistant Prefill Feature`
> - `GitHub Contributions` 


- **Transformers 新增 Assistant Prefill 功能**：一个新的 [pull request](https://github.com/huggingface/transformers/pull/33198) 为聊天模板和 TextGenerationPipeline 提议了 **assistant prefill** 功能，允许模型预填充初始回复。
   - 该功能在内部和 GitHub 上已被**多次请求**，证明了社区对其的需求。
- **新 pull request 带来的增强**：由 **Rocketknight1** 主导的 pull request 增加了内部和外部 **GitHub** 上都渴望的功能。
   - 通过实现此功能，Transformers 库继续演进，解决了**用户需求**并增强了可用性。



**提及的链接**：<a href="https://github.com/huggingface/transformers/pull/33198">由 Rocketknight1 提交的为聊天模板和 TextGenerationPipeline 添加 assistant prefill 的 Pull Request #33198 · huggingface/transformers</a>：在内部和 Github 上都被多次请求的功能是 assistant prefill：即为模型开始部分响应并让其继续的能力。我们使用了一个稍微...

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1278976543678533692)** (3 messages): 

> - `Llama 3.1 special tokens`
> - `Fixing untrained tokens` 


- **Llama 3.1 仍面临特殊 token 问题？**：一位成员询问 **Llama 3.1 base** 是否仍存在未初始化的特殊 tokens 问题，特别是关于 embeddings 超出分布范围（out of distribution）的问题。
   - 另一位参与者确认了该问题，并提供了一个涉及添加新选项来解决该问题的方案。
- **引入 `fix_untrained_tokens` 选项**：针对提出的担忧，一位成员宣布引入了 `fix_untrained_tokens: true` 选项，现在可以进行设置。
   - 该选项预计将解决未初始化特殊 tokens 所面临的问题。


  

---



### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1278839619231682632)** (6 messages): 

> - `Groq Leaderboard Updates`
> - `Documentation of Model Steps`
> - `GIS Geometry Presentation Test Case Issues`
> - `Model Evaluation Temperature Settings` 


- **Groq 等待加入排行榜**：成员们注意到 **Groq** 尚未被添加到排行榜中，但预计他们的 **PRs** 将在下周左右提交。
   - *我们仍在等待*他们为评估过程做出贡献。
- **步骤文档化保证**：一位成员保证他们将记录可复现性所需的必要步骤，以解决之前讨论中提出的疑虑。
   - 这种主动的方法旨在增强模型文档的清晰度。
- **GIS 几何展示的问题**：一位成员分析了一个 **Java 测试用例**，其中他们的模型在 GIS 几何展示的初始化提示词上遇到了困难。
   - 他们得出结论，由于请求是关于初始化的，模型的回答仍然比执行 function calls 更好。
- **关于评估 Temperature 的澄清**：成员们询问是否如讨论中所述，**所有模型**都在 **0** 的 Temperature 下进行严格评估以确保公平比较。
   - 一位成员表达了他们对设置的理解，并强调保持某些参数不变对于获得一致的 function call 输出至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard#model-specific-optimization">gorilla/berkeley-function-call-leaderboard at main · ShishirPatil/gorilla</a>: Gorilla: Training and Evaluating LLMs for Function Calls (Tool Calls) - ShishirPatil/gorilla</li><li><a href="https://github.com/ShishirPatil/gorilla/discussions/562">Set Model Temperature to 0 for Consistent Leaderboard Results · ShishirPatil/gorilla · Discussion #562</a>: 当前的模型生成脚本 (model_handlers) 在推理时使用默认的 0.7 Temperature。这给模型输出生成引入了一定程度的随机性，导致潜在的...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1278838600510996593)** (2 messages): 

> - `tinygrad capabilities`
> - `sparsity handling` 


- **关于 tinygrad 操作处理的查询**：一位成员询问 **tinygrad** 是否仅限于**静态调度的操作**，以及它在处理**半结构化稀疏性（semi-structured sparsity）**或**权重选择**方面是否存在问题。
   - 这一查询引发了关于其整体能力的进一步讨论，质疑 tinygrad 是否有无法执行的操作。
- **George Hotz 寻求限制示例**：George Hotz 作出回应，要求提供用户认为在 **tinygrad** 中无法执行的操作的具体示例。
   - 这个问题表明他希望澄清该框架在**操作调度**方面的通用性和局限性。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1279210301232906393)** (2 messages): 

> - `Tensor.cat 功能`
> - `Sharded tensors`
> - `Batch dimension 处理`
> - `错误排查` 


- **Tensor.cat 在处理 sharded tensors 时遇到困难**：一位用户在尝试使用 `Tensor.cat` 沿 batch 轴连接两个 sharded tensors 时遇到错误，导致了与 padding 相关的 `AssertionError`。
   - 他们注意到可以 unsqueeze 一个额外的维度，但在尝试对生成的张量进行 reshape 时遇到了进一步的问题。
- **澄清问题的根源**：用户针对该问题的本质提出了两个疑问——这究竟是一个根本性的限制，还是仅仅是不支持的功能。
   - 他们正在探索修改代码以处理额外 batch dimension 的方法，或者利用不同的方法来避免对 `cat` 的需求。


  

---



---



---



---



---



---



{% else %}


> 完整的频道明细已针对邮件进行截断。 
> 
> 如果您想查看完整明细，请访问此邮件的网页版： [{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请 [分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}