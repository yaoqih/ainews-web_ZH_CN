---
companies:
- sakana
- openinterpreter
- reddit
- aether-research
- mistral-ai
- nvidia
- lmdeploy
date: '2024-03-21T23:33:53.811566Z'
description: '以下是该文本的中文翻译：


  **Sakana** 发布了一篇关于进化模型合并（evolutionary model merging）的论文。**OpenInterpreter** 推出了他们的
  **O1 开发工具包（devkit）**。讨论指出 **Claude Haiku** 在 10-shot 示例下的表现被严重低估。针对 **Reddit 的 IPO**，AINews
  推出了 Reddit 摘要功能，首发于 /r/LocalLlama，随后将覆盖 r/machinelearning 和 r/openai 等子版块。**Aether
  Research** 发布了基于 **Mixtral** 的 **Cerebrum 8x7b**，其推理任务表现可媲美 **GPT-3.5 Turbo** 和
  **Gemini Pro**，创下了开源推理模型的新纪录（SOTA）。Cream-Phi-2 的创作者发布了微调模型 **Moistral 11B v1**。一个创意写作基准测试使用
  **Claude Opus** 作为评委。爱好者们正在探索 **1.58 BitNet** 三进制量化和 **1-bit 大语言模型（LLM）** 的训练。英伟达的
  **Blackwell (h200)** 芯片支持 **FP4 精度**量化。**LMDeploy v0.2.6+** 版本支持高效部署视觉语言模型（如 **Qwen-VL-Chat**）。用户正在寻找支持插件和
  RAG（检索增强生成）的 LLM API 图形用户界面（GUI）。此外，文中还讨论了合成训练数据生成以及针对聊天场景微调语言模型的流水线。'
id: 6ae360e4-c736-4611-b702-7a2c02157747
models:
- cerebrum-8x7b
- mixtral-7b
- gpt-3.5-turbo
- gemini-pro
- moistral-11b-v1
- claude-opus
- qwen-vl-chat
original_slug: ainews-welcome-rlocalllama
people: []
title: 欢迎来到 /r/LocalLlama！
topics:
- model-merging
- benchmarking
- quantization
- performance-optimization
- deployment
- vision
- fine-tuning
- training-data
- synthetic-data
- rag
- gui
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月20日至3月21日的 AI 新闻。我们为您检查了 [**358** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **21** 个 Discord（**337** 个频道，**9841** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1033 分钟**。

这是一个新闻较少的日子 —— [Sakana 发布了一篇进化模型合并 (evolutionary model merging) 论文](https://arxiv.org/abs/2403.13187)，[OpenInterpreter 推出了他们的 O1 devkit](https://x.com/openinterpreter/status/1770821439458840846?s=46&t=90xQ8sGy63D2OtiaoGJuww)，人们正在讨论 [如果你制作 10-shot 示例，Claude Haiku 是如何被低估的](https://x.com/mattshumer_/status/1770942240191373770)。

但借着今天 [Reddit 成功 IPO](https://www.cnbc.com/2024/03/21/reddit-ipo-rddt-starts-trading-on-nyse.html) 的机会，是时候终于为 AINews 引入 Reddit 摘要了！目前先从 /r/LocalLlama 开始，我们很快会开始总结评论，接下来我们规划了 r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。如果您发现我们遗漏了任何重大的 alpha 信息发布版块，请告诉我们。

---

**目录**

[TOC] 


---

# REDDIT: /r/LocalLlama

**模型发布与基准测试**

- [Cerebrum 8x7b 来了！](https://www.reddit.com/r/LocalLLaMA/comments/1bj8d4w/cerebrum_8x7b_is_here/) Aether Research 发布了基于 Mixtral 的 Cerebrum 8x7b，其训练方式与其 7b 版本类似。它在推理任务上的表现与 GPT 3.5 Turbo 和 Gemini Pro 相当，使其成为开源推理模型的 SOTA。(201 upvotes)
- [Moistral 11B v1，最“湿润”的 Mistral —— 来自 Cream-Phi-2 的创作者！（微调，非合并）](https://huggingface.co/TheDrummer/Moistral-11B-v1?not-for-all-audiences=true) (165 upvotes)
- [使用 Claude3 作为评委的新创意写作基准测试](https://www.reddit.com/r/LocalLLaMA/comments/1bjih89/new_creative_writing_benchmark_using_claude3_as/) 创建了一个使用 Claude Opus 作为评委的创意写作基准测试，包含 19 个写作提示、36 个狭义定义的评估标准，以及每个问题的示例参考输出。(14 upvotes)

**量化与性能优化**

- [[求助/严肃讨论] - 我尝试实现 1.58 BitNet —— 但我卡住了。](https://www.reddit.com/r/LocalLLaMA/comments/1bjjywn/helpserious_discussion_i_tried_my_hand_at_a_158/) 一位业余爱好者尝试实现 1.58 BitNet Ternary 论文，生成的模型符合预期大小（例如 300M 参数为 72MB）。然而，他们遇到了训练损失 (training loss) 不下降以及推理 (inference) 无法正常工作的问题。(32 upvotes) 
- [1 bit LLM 时代 —— 训练、技巧、代码](https://www.reddit.com/r/LocalLLaMA/comments/1bjinlq/the_era_of_1_bit_llms_training_tips_code/) 分享了 1.58bit 论文的后续。(110 upvotes)
- [Nvidia Blackwell (h200) 与 FP4 精度](https://www.reddit.com/r/LocalLLaMA/comments/1bjlu5p/nvidia_blackwell_h200_and_fp4_precision/) 新的 Nvidia h200 芯片支持 FP4，但目前尚不清楚这种级别的量化在实践中对 LLM 是否有用，因为即使是 FP8 也很少被使用。(8 upvotes)

**部署与服务**

- [LMDeploy 非常易于使用，且在 VLM 部署方面效率极高。[讨论]](https://www.reddit.com/r/LocalLLaMA/comments/1bjaly4/lmdeploy_is_very_simple_to_use_and_highly/) LMDeploy v0.2.6+ 支持多模态模型 (VLM) 的推理和服务，只需使用 `pipeline` API 编写几行代码即可。像 Qwen-VL-Chat 这样的模型可以使用兼容 OpenAI 的服务器或 Gradio UI 进行服务。(18 upvotes)
- [寻找支持 LLM API（openrouter, openai 等）、插件和 RAG 支持的 GUI。](https://www.reddit.com/r/LocalLLaMA/comments/1bjbzpa/searching_for_a_gui_for_llms_apis_openrouter/) 一位用户正在寻找一个用户友好的 GUI，支持 OpenAI 的 ChatGPT API（或 OpenRouter 等兼容接口），并允许使用插件和 RAG。(3 upvotes)
- [带有 RAG 的 LocalLLM 多用户服务器](https://www.reddit.com/r/LocalLLaMA/comments/1bj8avr/localllm_with_rag_multiuser_server/) 有人尝试将 gpt4all 设置为带有 sbert 插件的内部服务器以处理本地文件，但在通过 API 使其工作时遇到了困难。(2 upvotes)

**训练数据与微调**

- [生成训练数据的流水线（10,000 名不同人士的 10,000 篇日记条目）](https://www.reddit.com/r/LocalLLaMA/comments/1bjr3ix/pipeline_for_generating_training_data_10000/) 构建了一个用于生成多样化合成日记数据以进行微调的流水线。它使用了 Prompt 变体、生活变量（职业、情绪等）和随机选择来避免重复内容。(4 upvotes)
- [为聊天微调语言模型](https://www.reddit.com/r/LocalLLaMA/comments/1bjgr43/finetuning_a_language_model_for_chat/) 有人询问如何仅使用文章针对新主题微调聊天语言模型，以及是否需要 Q&A 数据集。(0 upvotes) 
- [准备训练数据](https://www.reddit.com/r/LocalLLaMA/comments/1bjk7dw/preparing_training_data/) 一位用户询问如何为微调准备训练数据。(2 upvotes)

**硬件与计算资源**

- [升级 PC/GPU 以在本地运行 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1bjcbn6/pcgpu_upgrade_to_run_llm_locally/) 有人正考虑升级 GPU 以在本地运行不错的 LLM，目前在考虑 24GB VRAM 的 NVIDIA 显卡。他们想知道主板等其他组件是否也需要升级。(3 upvotes)
- [在 RTX 4080 笔记本电脑上进行微调](https://www.reddit.com/r/LocalLLaMA/comments/1bjfwf2/fine_tuning_on_laptop_rtx_4080/) 一位用户想知道在配备 RTX 4080 12GB 的笔记本电脑上对 Mistral 7B 等模型进行微调是否可行。(2 upvotes)
- [从性价比来看，旧矿卡 P102-100 值得吗？](https://www.reddit.com/r/LocalLLaMA/comments/1bjhufg/old_mining_cards_p102100_worth_it_when_looking_at/) 有人询问单价 20 美元的旧 P102-100 矿卡在推理方面的性价比是否值得，考虑到它们可以解锁到 10GB 但只有 PCIE 1.1 x4 通道。(1 upvote)

**梗图与幽默**

- [“下一个是谁？”](https://i.redd.it/5rma8h7xqipc1.png) 一张梗图，调侃微软为了垄断市场而破坏开源 AI 计划。(349 upvotes)
- [我用 LLM 做了一个游戏。它叫 Classroom Simulator，灵感来自《模拟人生》和《黑与白》。目前已上线并免费游玩。链接在评论区。](https://v.redd.it/zcnqywua1ipc1) (101 upvotes)
- [我讨厌微软](https://www.reddit.com/r/LocalLLaMA/comments/1bjmsfq/i_hate_microsoft/) 一位用户发泄对微软的不满，指责其为了垄断 AI 市场而“破坏每一个开源计划”。(92 upvotes)


# 第 X 部分：AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果

**英特尔与 AI 产能**

- [@sama](https://twitter.com/sama/status/1770468022081527966): "很高兴看到这一点——为英特尔、美国以及更多的 AI 产能感到兴奋！" (681k views)

**调试与反直觉的代码**

- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770528106513600636): "2 小时的调试。不管你怎么说，这都很反直觉。" (541k views)
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770586939059523970): "话虽如此，我不认为不同的语言设计能解决这种‘反直觉’的特定问题。" (13k views)
- [@francoisfleuret](https://twitter.com/francoisfleuret/status/1770693914669772800): "当你那个让你从床上跳起来的想法行不通时的那种感觉（TFW）。" (4k views)

**微软与 OpenAI**

- [@AISafetyMemes](https://twitter.com/AISafetyMemes/status/1770348812600529089): "微软 CEO：如果 OpenAI 明天消失了也没关系" (192k views)
- [@Teknium1](https://twitter.com/Teknium1/status/1770428674883699134): "微软的愚蠢举动。微软雇佣了一个末日论骗子、大师级书籍推销员来‘领导’他们新的 AI 计划。这家伙几个月前才创办了 Inflection，筹集了 20 亿美元来资助他的新书巡回宣传，然后就跑路了？笑死。好吧，我想这让微软彻底退出了优秀模型的竞争。" (180k views)
- [@Teknium1](https://twitter.com/Teknium1/status/1770431787459842092): "我想当你是一个末日论者时，最好的做法就是锁死本可以流向别处的 20 亿美元 VC 资金，然后锁死 50,000 块 H100，接着离开，最后锁死微软自己的 AI 努力 😏" (16k views)
- [@mark_riedl](https://twitter.com/mark_riedl/status/1770622060848378317): "这是我对微软和 Inflection AI 新闻的看法：纳德拉雇佣了一位虐待员工、拖延性虐待案件的毒性经理来管理他们新的 AI 部门。但我猜雇佣 DeepMind 的创始人比拥有良好的领导力更重要。" (15k views)
- [@ethanCaballero](https://twitter.com/ethanCaballero/status/1770511139601871351): "“现在我成了微软，前沿模型初创公司的吞噬者”" (2k views)

**用于对话生成的 Q-Star 基于能量的模型**

- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770635047722438733): 为普通本科生编写的关于对话生成的 Q-star 基于能量的模型 (EBM) 理念的详细解释。关键点：使用抽象语义表示空间，通过优化寻找能量最低的响应，将“决定说什么”与“如何说”分离。(186k views)
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770691504148750772): “引用推文和回复中的很多人都没看该线程的第 2 条帖子……（提示——那才是真正重要的部分。）”(16k views)
- [@jeremyphoward](https://twitter.com/jeremyphoward/status/1770635053196034493): “但实际上这根本不是对 Q* 的描述。相反，它是 Claude 自动生成的关于 @ylecun 的 EBM 项目的解释。如你所见，它们确实看起来*非常*相似。我对这些关于 OpenAI ‘泄密’的说法持怀疑态度。它似乎只是在总结 Yann 的工作。”(13k views)
- [@leithnyang](https://twitter.com/leithnyang/status/1770642937413820926): “这基本上就是把 Yann LeCun 的 JEPA 架构重新包装成了 Q*”(113 views)

**建议与观察**

- [@gdb](https://twitter.com/gdb/status/1770532522692100299): “知道该做什么和实际去做都至关重要，但只看重其中之一是常见的错误”(142k views) 
- [@gdb](https://twitter.com/gdb/status/1770677916826763387): “对细节的执着被低估了”(113k views)
- [@AravSrinivas](https://twitter.com/AravSrinivas/status/1770587464459231269): “极少数人理解长期主义思维。而那些理解的人将获得巨大的回报。”(57k views)

**梗与幽默**

- [@KevinAFischer](https://twitter.com/KevinAFischer/status/1770640604516778216): “没什么好看的。只是一个随机鹦鹉 (stochastic parrot)”(23k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770436957560012857): “获得新老婆 (waifu)”(30k views)
- [@Nexuist](https://twitter.com/Nexuist/status/1770571250047279349): “科技男 (techbros) 卖出 1,000,000 辆电动汽车，他们对世界来说仍然是坏人；科技男将 100,000 吨物资送入轨道，他们对世界来说仍然是坏人；科技男治愈了 10,000 名四肢瘫痪者，他们对世界来说仍然是坏人 <— 你现在就在这里”(72k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770445195043098763): “你有什么毛病”(3k views)
- [@nearcyan](https://twitter.com/nearcyan/status/1770588147405160507): “想象一下，作为一个没有 Neuralink 的创始人，你必须像个老头一样动动手才能工作，哈哈”(17k views)
- [@nearcyan](https://twitter.com/nearcyan/status/1770703167501533540): “哇，你们真的很奇怪”(3k views)
- [@cto_junior](https://twitter.com/cto_junior/status/1770689422343741887): “想象一下这在 Neuralink 上运行 🤩🤩🤩 你可以一直待在 gooncave 里，不管外面是 Hacker Way 1 号还是达美乐披萨店”(1k views)

---

# PART 0: 摘要之摘要之摘要


> 我们得出结论，Claude Opus 是顶级摘要的最佳模型，因此我们将停止 A/B/C 测试（有关我们的努力/记录，请参阅存档）。我们将为所有 3 个及更多模型（包括 Gemini 1.5!!）提供并行运行，因为这个问题在拓扑上与我们将要推出的个性化应用相似。

**1. Grok-1：巨兽出笼**

- xAI 发布了 **Grok-1**，这是一个 **3140 亿参数的 Mixture-of-Experts 模型**，引发了关于其性能与 GPT-3.5、Mixtral 和 LLaMA 对比的辩论。该模型可在 [GitHub](https://github.com/xai-org/grok-1) 上通过 Apache 2.0 许可证获取。
- 讨论集中在 Grok-1 **持续预训练 (continual pretraining) 的潜力**、**量化策略 (quantization strategies)**，以及通过 **种子 (torrents)** 分发对开源 AI 可信度的影响。
- 一个 [高中期末考试数据集](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam) 显示，尽管对其质量存在质疑，但 Grok-1 的表现与 **GPT-4 和 Claude** 接近。

**2. 检索增强生成 (RAG) 的创新**

- 成员们探索了增强 RAG 模型的功能，例如用于详细/结构化输出的 **响应模式**、**引用高亮**、意图理解以及为了提高相关性的任务分解。
- 提议包括 **平衡外部上下文利用与内部知识**、训练用于高效实时 RAG 操作的专用模型，以及 **输出格式化** 的最佳实践。
- 共享了相关资源，包括 Command R 用于 RAG 的 [GitHub 实现](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py) 以及带有行内引用的 [Cohere 模型](https://cohere.ai/docs)。

**3. 大语言模型 (LLM) 的扩展策略与效率**

- 讨论围绕扩展上下文长度的 **持续预训练方案** 展开，重点关注 [这篇论文](https://arxiv.org/abs/2402.10171) 中强调的数据工程方法。
- 一篇 [arXiv 论文](https://arxiv.org/abs/2403.08763) 提出了具有成本效益的技术，如 **学习率预热 (learning rate warming) 和数据重放 (data replay)**，用于在不进行完整重新训练的情况下更新 LLM。
- 探索了像 [Smallstral](https://huggingface.co/AlexWortega/smallstral) 这样 **缩减模型 (downscaling models)** 的可行性，在性能和高效预训练方面展现了前景。

**4. 语言模型的多语言挑战与基准测试**

- 讨论涉及在处理基于英语主导语料库训练的多语言模型时，**特定语言知识的复杂性**，并引用了 [这篇论文](https://arxiv.org/abs/2402.10588)。
- 成员们强调需要 **针对德语的基准测试** 来衡量母语质量，提议进行大学合作，并参考了 [SuperGLEBer](https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/) 等资源。
- 关于高效 LLM 推理的 [Medusa 论文](https://arxiv.org/abs/2401.10774) 以及一项关于 [LLM 对同行评审影响](https://arxiv.org/abs/2403.07183) 的研究，引发了围绕模型效率和学术影响的对话。

**5. 其他**

- **LangChain 增强与集成**：LangChain 用户正在探索 **astream_events** 等新功能，为高级研究助手 [Rubik's AI](https://rubiks.ai/) 寻找 Beta 测试人员，并分享 [AI 聊天机器人](https://github.com/Haste171/langchain-chatbot) 和 [书签管理器](https://twitter.com/uogbuji/status/1768681648516661446) 等项目。此外还在讨论与 **Vertex AI** 和 **Hugging Face** 的集成，以及关于 [构建 AI 应用](https://youtu.be/vHjc5CEoIJE) 和 [plan-and-execute agents](https://www.youtube.com/watch?v=ZlJbaYQ2hm4) 的教程。
- **光子学与 NVIDIA 的进展**：围绕比传统芯片快 1000 倍的 [新型光子芯片](https://youtu.be/8ohh0cdgm_Y) 以及 NVIDIA 的 **H100 GPU** 搭配功耗约 850W 的 ARM 架构 CPU 的讨论引发了热议。NVIDIA 的 **GTC Keynote** 也因提到 1.8T 参数模型和带有 **192GB HBM 的 B100** 等新硬件而令人振奋。
- **提示工程 (Prompt Engineering) 与测试工具**：新的提示工程和测试工具及平台不断涌现，例如 **Prodigy 的提示工程功能**、[PromptTools](https://github.com/hegelai/prompttools)、[PromptFoo](https://github.com/promptfoo/promptfoo)、**Vercel 的 AI Playground** 以及 **Helicone.ai**。此外还进行了 AI 增强博客定制的实验，并讨论了 AI 辅助博客功能。

---



# PART 1: 高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **推出下一代 Stable Video 3D**：Stability.ai 发布了 **Stable Video 3D** (SV3D)，该模型取代了 Stable Video Diffusion，能够从单张图像生成增强的 3D 和多视角合成。他们推出了两个新变体：用于生成轨道视频的 **SV3D_u** 和具有高级功能的 **SV3D_p**。[在此了解更多关于 SV3D 的信息](https://stability.ai/news/introducing-stable-video-3d)。

- **Cascade 的代码难题**：在与 Stable Diffusion 社区的交流中，一位工程师感叹运行 **Stable Cascade** 的代码优化问题，提到它比 Stable Diffusion XL (SDXL) 慢得多，且更消耗 CPU。

- **焦急等待 Stable Diffusion 3**：工程社区对 **Stable Diffusion 3 (SD3)** 的发布充满期待，表达了对增强 Prompt 遵循能力的希望，并传闻早期访问邀请即将发放。

- **围绕加密货币合作的安全质疑**：有关 Stability AI 进军区块链合作伙伴关系的传闻引起了许多工程师的担忧，引发了关于此举对开源传统和安全标准影响的辩论。

- **消费级技术运行 AI 的挑战**：实际讨论指出了在标准硬件配置上运行 Cascade 或 SD3 等高级 AI 模型所面临的挑战，特别强调了对 GPU VRAM 的需求。工程师们还强调了在包括游戏在内的各种应用中，需要更易于获取的生成式 AI 工具。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**Pro 会员福利还是困惑的问题？**：Perplexity AI 已向 **Pro 用户开放 Claude 3 Opus 的无限次每日查询**，但用户对考虑到上下文限制（context limits）后“无限”的实际程度表示担忧。关于“无限”在日常使用和上下文方面的具体含义，是社区中的热门话题。

**AI 育儿前景**：社区展开了一场关于 AI 在简化儿童复杂概念方面作用的激烈讨论，强调了 AI 发展适宜性及其在教育支持中潜力的重要性。

**工程师们的困惑**：尽管计划弃用 `sonar-medium-online` 模型，但该模型在截止日期后似乎仍在运行，导致用户困惑。工程师们辩论了 API 的行为，讨论围绕 `maxtokens` 参数展开，并观察到通过浏览器与通过 API 查询时呈现的不同新闻结果。

**寻找真相与技术工作**：用户分享了使用 **Perplexity AI 的 Claude 3 Opus** 进行创意写作实验、查询最简洁选项、探究朝鲜政治动态、推测火星生活以及抓取职位发布的经验。关于搜索结果中提供链接的可变性和可靠性，存在诸多疑问。

**对企业合作持谨慎乐观态度**：关于 **苹果和谷歌潜在 AI 集成** 的猜测不断增加，成员们热烈讨论了生成式 AI 合作的细节，并分享了对科技巨头战略和 AI 商业化未来的看法。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Grok 1 加入对话**：Elon Musk 的 **Grok 1**（一个拥有 3140 亿参数的 Mixture-of-Experts 模型）发布，其巨大的体量令人惊讶，预计性能低于 Miqu 但高于 Llama2 70b。社区对 Grok 1 与 Mixtral 的可比性表现出浓厚兴趣，相关细节通过 [Hugging Face 上的 xai-org](https://huggingface.co/xai-org/grok-1) 等链接分享。

- **AI 微调技巧与建议**：对于在 Mistral-7b 上微调 **QLoRA**，`2e-4` 的学习率（最高 3 个 epochs）是首选方案。社区提出了创新的模型合并策略，例如将 UltraChat 和基础 Mistral 的合并策略应用于 **Mistral-Yarn**，这在社区中引发了怀疑与乐观并存的讨论。

- **Unsloth AI 登上 GitHub 趋势榜**：Unsloth AI 的 GitHub 仓库因其趋势表现备受关注，所有者向用户表示感谢，并邀请更多工程师查看他们的 [更快的微调仓库 (faster finetuning repository)](https://github.com/unslothai/unsloth)。

- **警惕身份冒充**：据报道，Discord 上出现了一个冒充 **Daniel Han** 的诈骗账号。社区受邀保持警惕，强调了核实身份和举报可疑账号的重要性。

- **模型保存时的 VRAM 困扰**：据指出，在保存类似 7b Mistral bnb 4bit 的模型时，需要充足的 VRAM 和额外的系统 RAM 以防止崩溃。这一问题在对比使用 Colab 与本地环境时尤为突出。

- **社区在 AI 与艺术中的创意联结**：社区讨论倾向于创意表达，成员们互相支持诗歌创作。此外，还交流了资源，如强化学习 (Reinforcement Learning) 的可视化工具，以及在 [UIverse Elements](https://uiverse.io/elements) 上发现的 CSS 或 Tailwind UI 元素集合。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Grok-1 和 Command-R 引发热议**：工程师们正在讨论 xAI 的大规模 Grok-1 模型，以及 Command-R 模型通过 [llama.cpp Pull Request #6033](https://github.com/ggerganov/llama.cpp/pull/6033) 与 LM Studio 的待定集成。虽然由于硬件限制，一些人选择了更小、更高效的模型（如 Gemma 2B 或 Mistral 7B），但其他人正在探索 Command-R 的兼容性，并提供了其 [Hugging Face 仓库](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF)的链接。

- **LM Studio 功能查询**：成员们正在寻求关于 LM Studio 功能的澄清，例如使用个人文档进行对话以及对 autogen 等插件的支持。配置文件可以在 [GitHub](https://github.com/lmstudio-ai/configs) 上找到，关于 AI 难题的疑问则引导成员在特定频道寻求指导。

- **寻求 AI 硬件的和谐配置**：技术讨论集中在硬件配置上，包括即将推出的 5090 GPU 预期的性价比，以及使用 PCIe risers 进行多 GPU 设置的挑战。一场尤为激烈的辩论围绕着语言模型任务的最佳 GPU 选择，以及自定义设置中散热和功耗的影响展开。

- **AVX Beta 版与模型支持**：LM Studio 的 Beta 版应用是一个**旧版本**，没有高优先级的 AVX 支持。虽然它支持某些模型，但最新的模型（如 **starcoder2** 和 **gemma**）尚不可用。不过，在 Beta 版应用上运行 **Mistral** 模型是可行的。

- **AMD ROCm 在 LM Studio 中的角色**：适用于 AMD GPU 的 ROCm 库对于 LM Studio 的兼容性至关重要。支持 gfx1031 和 gfx1032 的预构建 Windows ROCm 库已在 [GitHub](https://github.com/brknsoul/ROCmLibs) 上分享，但目前的讨论表明，模型目前可能仅利用主 GPU，并对未来支持双 7000 系列 GPU 进行了推测。

- **Agent 系统评估进行中**：一条单独的消息询问了用于验证创意概念的 **Agent 系统**的选择过程，突显了成员参与 Agent 评估协作项目的进展。

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NVIDIA 在 RTX 50 系列上采取稳健策略**：NVIDIA 计划为其 GeForce RTX 50 系列 "Blackwell" 显卡配备 **28 Gbps 的 GDDR7 显存**。这比目前已有的 32 Gbps 芯片速度慢，考虑到显存带宽和历史趋势，这一战略选择引发了讨论。链接：[NVIDIA's Memory Strategy](https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed)。

- **AI 模型通过 MatchboxDAO 准备进入游戏领域**：MatchboxDAO 宣布了一个开放游戏数据用于 **AI Agent 开发** 的项目，该项目由社区资助，旨在促进游戏 AI 的创新。链接：[Game On for AI Developers](https://x.com/unkjdgames?s=21)。

- **修改记忆 - Grok-1 的发布与局限性**：xAI 拥有 3140 亿参数的 MoE 模型 **Grok-1** 因相较于 GPT-3.5 提升有限而面临审查，引发了关于超大模型实用性和持续预训练需求的疑问。

- **OpenAI 的 GPT-4 笼罩在猜测中**：NVIDIA CEO 暗示了一种具有 1.8 万亿参数的新架构，助长了其可能就是 **GPT-4** 的传闻。这些猜测包括 OpenAI 尚未正式确认的 MoE 配置暗示。

- **缩小 LLM 规模以增强性能**：一种专注于 **模型缩小 (downscaling models)** 的新方法（如 **Smallstral**）在任务表现和持续预训练有效性方面展示了可喜的结果。这强调了 AI 模型缩放策略的多样性和效率潜力。链接：[Scaling Downward](https://huggingface.co/AlexWortega/smallstral)。

- **RAG 讨论达到新高度**：关于 RAG 能力增强的讨论非常热烈，集中在响应模式和高召回相关性等特性上。社区反思了模型输出中外部上下文利用与内部知识之间的平衡，并探索使用 **更小的专业化模型** 来优化 RAG 流水线。相关链接：[Cohere's in-line citation model](https://cohere.ai/docs)，[Command R for RAG GitHub implementation](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py)。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Grok-1 面临审查**：[Grok-1 模型](https://github.com/xai-org/grok-1) 已进入竞技场，但其性能和 Twitter 的聊天机器人界面效果受到质疑。工程师们对 Grok 的模型大小表示担忧，怀疑在与 Mixtral 或 MiQ 等竞争对手相比时，更大是否意味着更好。同时，有人呼吁提供易于获取的 RAG 教程，并建议注意此 [GitHub issue](https://github.com/pytorch/pytorch/issues/122123) 中详述的 PyTorch Mac 错误。

- **Mamba 模型中的投机采样受到挑战**：模型领域的讨论对 Mamba 等模型的投机采样 (Speculative Sampling) 表示怀疑。与 Transformer 不同，它们可能无法从投机采样中获得类似的收益，且验证的计算成本仍然是一个障碍。模型与 `lm-eval-harness` 的集成正在探索中，同时正在剖析默认使用 `gpt-2-small` 和评估挂起等问题，包括 [此处](https://github.com/EleutherAI/lm-evaluation-harness/issues/1485) 发现的特定死锁问题。

- **数据复杂度动摇缩放定律**：在 [#scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666) 频道中，焦点在于数据集复杂度如何影响语言模型缩放定律 (Scaling Laws)，其中来自概率上下文无关文法 (PCFG) 的句法属性和 gzip 压缩在预测中发挥了作用。研究人员正屏息以待更广泛的实验，以确定缩放定律的具体数值。

- **N-gram 采样技术辩论**：在 [#interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241) 中，工程师们面临从特定 n-gram 统计数据中采样字符串的挑战。提出了一种自回归采样方法来创建与这些统计数据一致的最大熵分布，并在 [GitHub](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py) 上分享了一个实际示例。

- **为预训练打乱 The Pile 数据**：关于 The Pile 数据打乱的询问得到了澄清：原始文件没有打乱，但在 Hugging Face 上提供的预分词 (pretokenized) 数据是打乱过的。这是 Pythia 使用的同一数据集，并指出虽然 The Pile 的单个组件未打乱，但训练/测试/验证集预计是混合的。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **思考 AI 的本质与技术**：工程师们讨论了像 **ChatGPT** 这样的 AI 是否真正“理解”语言，还是由复杂的 next-word prediction（下一个词预测）算法创造的错觉。人类训练的影响也受到了辩论，一些人认为它赋予了超越部分人类的对话能力。

- **惊叹于 DALL-E 3 的能力**：社区对 **DALL-E 3** 相比前代在遵循详细 Prompt 方面的先进能力表示赞赏，同时也考虑了速度和图像保存等实际方面。还提到了利用 **DALL-E 3** 和 **GPT-4** 的 **ChatGPT+** 的优势。

- **AI 模型对比**：根据用户体验对 **GPT-4** 和 **Claude** 进行了对比，讨论了它们的对话能力、成本效率，以及在冗长度和政治正确性方面的各自优势。

- **AI 使用中的挑战与优化**：用户分享了在创作过程中对敏感内容过滤器的挫败感，注意到 ChatGPT 的行为变化（可能由于浏览器扩展冲突引起），并寻求防止 AI 模型拒绝回答的方法。

- **学习 AI 平台与 Prompt 创作**：交流了学习 AI 概念的资源，特别是关于 PyTorch 以及深入研究 AI 所需的数学基础。探索了用于分类任务的 Prompt 以旨在提高性能，同时分享了规避拒绝回答的 Prompt 策略。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **为 Aya 添加滑块**：Aya 演示已集成 *repetition penalty*（重复惩罚），并寻求贡献者在 Gradio 界面中添加 **slider feature**（滑块功能）。可以通过 [这里](https://huggingface.co/spaces/Tonic/Aya/discussions/3) 的 PR 进行贡献。

- **NVIDIA 的强力组合**：NVIDIA 的 **H100 GPU** 与基于 ARM 的服务器 CPU 相结合，功耗约为 **850W**；而基准测试表明 **H100 alone**（单独 H100）功耗就可达 700W。详情请参考 [这些基准测试](https://www.phoronix.com/review/nvidia-gh200-gptshop-ben)。

- **HuggingFace 的数据守护者**：HuggingFace 拥有一个 **data leaderboard**（数据排行榜），重点展示了该平台上托管的超过 **120B models**。在 [这里](https://huggingface.co/spaces/Weyaxi/data-leaderboard) 探索广阔的数据。

- **使用 Hugging Face 和 SageMaker 导航 MLOps**：一个 Amazon SageMaker 和 Hugging Face 工作坊提供了一个用于创建 **MLOps pipeline** 的 notebook；适合希望简化机器学习操作的人员。点击 [这里](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers) 查看工作坊。

- **多语言思考与 AI**：讨论涉及了跨 **中文和英文** 等不同语言工作的机器学习模型，强调了处理特定语言知识和任务时的复杂性。此外，关于 **高效语言模型推理的 Medusa 论文**，以及一项关于 **LLMs 对科学同行评审影响** 的研究，引发了关于模型效率和 LLMs 在学术界影响的对话。参考 Medusa 论文 [这里](https://arxiv.org/abs/2401.10774)，以及同行评审影响研究 [这里](https://arxiv.org/abs/2403.07183)。

- **NL2SQL 的进展与 NVIDIA 的新型芯片组**：一位工程师正在完善 **NL2SQL pipeline**，同时 NVIDIA 的 **Grace Hopper Superchip** 因其在 AI 相关任务中的出色表现而受到关注。对于 NLP 初学者，推荐了 Hugging Face 的 [NLP course](https://huggingface.co/learn/nlp-course/chapter1/1) 和斯坦福大学的 [SLP3 manuscript](https://web.stanford.edu/~jurafsky/slp3/) 等资源，并询问了用于 LLM 部署的免费 API，提到 "ollama" 是一个潜在资源。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **交互式文档革新 RAG**: 提出了一种在 [RAG pipeline](https://t.co/eCdLmlXZFj) 中处理复杂查询的新方法，通过将文档视为交互式工具，从而实现更细致的交互和更好的查询解析。

- **LlamaIndex v0.10.20 发布，引入 Instrumentation**: 最新的 LlamaIndex 更新包含一个 Instrumentation 模块，通过关于 [基础可观测性](https://t.co/GY4unUYOwl) 和 [API 调用追踪](https://t.co/E1d9dtkqAI) 的 notebook 进行了详细说明。

- **通过 Search-in-the-Chain 增强问答**: Shicheng Xu 等人讨论的一篇论文提供了一种将检索与规划交织在一起以改进问答的新方法，重点在于步骤验证和计划调整，详见 [此处](https://t.co/7gLlDyd1cV)。

- **融合 RAG 与求职**: [Kyosuke Morita 的博客文章](https://t.co/1Y9TPgGHW1) 深入探讨了一个求职辅助工具，该工具融合了 LlamaParse 和 LlamaIndex，根据候选人的简历量身定制职位匹配。

- **MemGPT 研讨会扩展 Agent 内存**: [Charles Packer 主持的研讨会](https://t.co/bUpqCvLweS) 探讨了 MemGPT 架构，该架构赋予 Agent 内存工具以与核心内存交互，从而提升 function-calling 能力。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Yann LeCun 对 LLM 的看空引发辩论**: 由 @Teknium1 的推文引发的对话讨论了 Yann LeCun 对大语言模型（LLMs）的怀疑可能源于对不依赖内部独白的认知过程的思考。讨论涉及“形状旋转者（shape rotators）”与“文字工作者（wordcels）”的概念，并引用了 [对缺乏内心独白的人的采访](https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)。

- **Grok-1 的开源发布伴随着质疑与希望**: xAI 发布了 Grok-1，这是一个拥有 3140 亿参数的巨型 Mixture-of-Experts 模型，邀请 AI 社区为其持续训练和评估做出贡献。怀疑者和乐观主义者纷纷发表看法，将 Grok-1 与 LLaMA 和 Claude 等模型进行比较，并思考持续预训练可能带来的改进，正如 Yao Fu 在 [关于 Grok 潜力的思考](https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww) 中所指出的。

- **Paper Club Session 亮点 - Attention 的起源**: **Paper Club session** 阐明了 Transformer 中 Attention 机制出现背后的“原因”，展示了其相对于固定长度编码向量的突破，并允许模型引用输入序列的任何部分，从而为 Transformer 的效率铺平了道路。

- **Lex Fridman 的播客因缺乏深度受到批评**: 听众对 Lex Fridman 采访 Sam Altman 的播客表示失望，批评其缺乏对 OpenAI 运营细节和政治环境的深入讨论，认为这是 AI 领域实质性对话的一次错失机会。

- **关于检索增强生成（RAG）和 Embeddings 的讨论**: 在 AI in Action Club 内部，成员们分享了 “Advanced RAG 01 - Small to Big Retrieval” 的链接，提供了关于 RAG 的详细见解。“对比 Embeddings（contrastive embeddings）”的概念以及 LLMs 在生成此类 Embeddings 中的应用是感兴趣的话题，这表明人们正在寻找超越传统余弦相似度的创新。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

**Copilot 中的 Codex 解密**: **Microsoft Codex** 现在可以在 Copilot 应用中免费访问，集成了 Jupyter Notebooks 以及 simpy 和 matplotlib 等库，从而实现更具资源优势的代码编写环境。

**DALL-E 3 数据集的新家**: 关于 **DALL-E 3 数据集** 从 Hugging Face 移除的困惑已得到解决；它已被重新安置，可通过此 [直接链接](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset) 获取。

**Grok-1 加入 AI 战场**: OpenAI 的 **Grok-1**（注：原作者误写，实为 xAI）是一个令人印象深刻的 314B 参数模型，它隆重登场，在各种基准测试中表现出色。它在 GitHub 上的发布引起了人们的兴趣，并与 **Mixtral** 和 **LLaMA** 等模型进行了比较，可在此处进行 [探索](https://github.com/xai-org/grok-1)。

**提升 LLM 的高效方法**: 一篇 [arXiv 论文](https://arxiv.org/abs/2403.08763) 讨论了成本效益高的方法，如学习率预热（learning rate warming）和先前数据的回放（replay），用于在不进行完整重新训练的情况下更新 LLMs。

**关于 GPT-4 的猜测性传闻**: 继 Nvidia 的暗示之后，关于 **GPT-4** 是一个 1.8 万亿参数的混合专家（MoE）模型的猜测不绝于耳。GPT-4 细节的真实性尚未得到证实，该话题是由一张 [推特图片](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large) 引发的。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**光子芯片超越传统硅芯片**：[Anastasia 的视频](https://youtu.be/8ohh0cdgm_Y)引发了关于比传统芯片快千倍的技术热议，同时还提到了 [Asianometry 频道](https://www.youtube.com/watch?v=29aTqLvRia8)等资源，供寻求硅光子（silicon photonics）和光基网络深入知识的爱好者参考。

**Triton 调试实现可视化**：工程师们分享了一个用于简化 Triton 调试的新可视化工具，以及一套用于深化知识的 **Triton Puzzles**，可在 [Google Colab](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) 上进行试用。

**CUDA 社区揭秘调度器奥秘**：深入讨论探讨了 CUDA 的 warp 调度器和内存管理策略的细微差别，引发了关于 **ProducerProvides, ConsumerTakes**、异步工作（async work）和流同步（stream synchronization）复杂性的对话。

**学术界的可重构计算**：成员们关注了用于高效 ML 的可重构计算这一学术领域，这主要由 [Prof. Mohamed Abdelfattah 的工作](https://www.mohsaied.com/)和 [ECE 5545 课程大纲](https://abdelfattah-class.github.io/ece5545/)推动，尽管对教科书细节存在一些困惑，但通过参考该课程的第一节讲座视频得到了解决。

**赶上 CUDA 进度**：为新加入的 CUDA 爱好者提供了指导，推荐了《Programming Massively Parallel Processors》等书籍（可在 [Amazon](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311) 购买），并鼓励利用 **torch** 等框架步入 ML/DL 领域。

**关于 Striped Attention 和 Flash Attention 的深入讨论**：一场关于 Attention 机制的良性辩论探讨了 *Ring Attention* 和 *Flash Attention* 不同的内存需求，包括建议查阅特定文献（[Striped Attention 论文](https://arxiv.org/abs/2311.09431)）和代码（[GitHub 实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)）以进行澄清。

**AI 与系统在 MLSys 2024 交汇**：工程师们交流了关于 MLSys 2024 会议的细节，强调了其在机器学习（Machine Learning）与系统（Systems）融合以应对新兴 AI 挑战方面的关键作用（[MLSys Conference](https://mlsys.org/)）。

**为 GTC 聚会做准备**：Gautier 最狂热的 AI 爱好者们正在组织 GTC 2023 的聚会，讨论访问计划并分享联系方式，同时也对参加此类独家活动的限制表达了一些幽默的调侃。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**LLaMa 模型与 Prompt 配合良好**：确认 LLaMa 模型能够很好地处理以 "system"、"user" 和 "assistant" 角色构建的 Prompt，这对使用 OpenAI JavaScript 库的用户非常有用。

**脚本将书籍拆解用于 AI 分段**：开发了一个创新脚本，可将书籍拆解以进行 AI 驱动的分段生成；通过 Airoboros 70B 测试并与 lzlv 70B 对比显示，在使用基于指令的数据时，生成质量有显著提升。

**对深度使用分析的需求增加**：讨论强调了社区对类似于 OpenAI 提供的详细使用分析的需求，特别关注每日或每周使用成本等见解，并按模型和应用程序进行细分。

**模型变得“难以捉摸”**：注意到最近模型行为的变化，特别是模型执行任务的意愿有所下降，同时出现了关于访问 sonnet:beta 和 opus:beta 等测试版模型的问题。公司确认应该有通用访问权限。

**为民所用、由民所创的 API**：一位用户计划首次推出一个公共 API，并寻求将其包含在 OpenRouter 的列表中，平台对此做出了积极回应，渴望通过私信交流更多细节。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**API 演进引发好奇**：工程师们正在质疑 LangChain 的 **astream_log** 的未来，因为 **astream_events** 处于 beta 状态；担忧主要围绕潜在的弃用或两者在用例上的区别。

**Rubik's AI 等待热心测试者**：**Rubik's AI** 正在招募 Beta 测试者，这是一个极具前景的研究助手，提供对 **Claude 3 Opus**、**GPT-4 Turbo** 和 **Mistral Large** 的访问。感兴趣的人可以加入 [候补名单](https://rubiks.ai/)。

**LangChain JavaScript 流式传输遇到障碍**：有报告称 JavaScript 中的 `RemoteRunnable` 存在流式传输问题，这与其在 Python 中的功能表现不同。社区正在寻求见解或修复方案，并建议在 [GitHub](https://github.com/langchain-ai/langchain/issues/13126) 和 LangChain 的安全 [指南](https://js.langchain.com/docs/security#reporting-a-vulnerability) 上进行跟进。

**社区展示多样化的 AI 创作**：创新者们推出了各种 AI 工具：一个用于数据分析的 AI 聊天机器人 ([Haste171/langchain-chatbot](https://github.com/Haste171/langchain-chatbot))，管理 Raindrop.io 书签的 **Living Bookmarks** 机器人，关于 [NeuroFusion](https://calendly.com/neurofusion/30min) 生产力的访谈邀请，一个流行的基于 AI 的爬虫 **Scrapegraph-ai**，以及用于模拟销售角色的 **Lyzr.ai's Automata** ([GitHub Repo](https://github.com/LyzrCore/lyzr-automata))。

**AI 学习变得触手可及**：*YouTube 教程* ([Nutriheal Demo](https://youtu.be/vHjc5CEoIJE)) 分享了关于使用 **Langchain's Pebblo** 创建注重隐私的个性化营养 AI 的教学资源，同时还包括本地部署 AI 解决方案、利用通用 UI 构建 AI 助手，以及开发具有战略能力的“计划并执行”风格 AI Agent 的文档 ([Langgraph Tutorial](https://www.youtube.com/watch?v=ZlJbaYQ2hm4))。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**通过 API 揭开模型奥秘**：一篇 [arXiv 论文](https://arxiv.org/abs/2403.09539) 讨论了对受 API 保护的大语言模型 (LLMs) 的查询如何可能泄露专有信息（如模型大小）—— 这是一个意外的 "softmax bottleneck"。人们对这些发现的准确性提出了担忧，特别是当模型使用 MoE 等技术时，这可能会扭曲大小估算。

**开源定义引发争议**：一场 Twitter [对话](https://twitter.com/rasbt/status/1769779229263065184) 引发了机器学习社区关于什么是“开源”的争议预测。这引发了关于是否应将 **数据** 纳入开源软件定义的讨论，并推动在术语边界上建立务实的共识。同时，人们对 EleutherAI 的社交媒体互动策略表示不满。

**Grok-1 加入模型盛宴**：xAI 推出了 [Grok-1](https://x.ai/blog/grok-os)，一个 **3140 亿参数的 MoE 模型**，引发了围绕其发布、性能指标（传闻超过 Falcon）及其营销策略的讨论。有人对基于种子（torrent）的发布方式表示怀疑，认为这会影响开源 AI 模型的声誉和政策，甚至有人开玩笑地提出通过邮寄物理硬盘来运送模型。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **对 Aribus 进展的困惑**：一名成员寻求关于使用 **Aribus** 开发的见解，并分享了一个 [Twitter 链接](https://twitter.com/alignment_lab/status/1758949148143841379)，但在频道内未收到进一步的细节或澄清。
- **寻找精通 HTTP 的 Embeddings**：有人表示有兴趣寻找在 **HTTP 响应**上训练的 Embeddings 模型，并建议可能采用经过适当训练的 Transformer 模型来完成此任务。
- **寻求 Mistral 的微调模型**：有人询问是否有一个同时使用 *orca-math-word-problems-200k 数据集* 和 *nvidia/OpenMathInstruct-1* 进行过微调的 **Mistral 模型**，然而，目前还没有关于此事的后续建议。
- **协作增强 Grok 1 的号召**：协作微调 **Grok 1** 的行动号召提到了对大量 **算力 (compute)** 和 **数据资源** 的需求，并提到 MoE 训练基础设施可用于支持这些努力。
- **Grok 1 基准测试担忧与惊人表现**：**Grok 1** 在 MMLU 上的基准测试表现，以及在 [高中期末考试数据集](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam) 中与 **GPT-4** 和 **Claude** 接近的表现引发了讨论，提出了关于其能力以及进一步训练对大规模算力和多样化数据的持续需求的问题。

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Devin 引发关于应用复杂性的辩论**：一名成员幽默地表示，**Devin** 启发了他们在应用开发中优先考虑简洁性，并暗示复杂的应用程序可能是不必要的。

- **神秘推文引发 Anthropic 阴谋论**：一条指向 [推文](https://x.com/tszzl/status/1768530219378631137?s=20) 的链接引发了担忧，认为 **Anthropic** 可能正在利用其 AI 来影响技术人员，暗示这可能是一种受控反对派（controlled opposition）的伪装。

- **Claude Sonnet 迈向新高度**：公会中的某人正考虑在一个高用量项目中使用 **Claude Sonnet**，并对其他人在每月数千万 **tokens** 规模下使用该 AI 的经验感到好奇。

- **解码 KPU 炒作**：对话揭示了对 [Knowledge Processing Unit (KPU)](https://maisa.ai/blog/kpu) 声明的怀疑，辩论了其与 **GPT-4** 基准测试对比的有效性。Maisa 的 CEO 在 [Twitter](https://x.com/davipar/status/1768683151780683919?s=20) 上澄清，**KPU** 是一种增强现有 **LLM** 的架构方法，而非一个新模型。

- **OpenAI 频道中未完成的事项**：#openai 频道中提到了一段孤立的 [链接](https://x.com/leopoldasch/status/1768868127138549841?s=46)，未提供进一步的上下文。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **德语语言学故障排除**：用户在 **DiscoLM-mixtral-8x7b-v2** 上遇到了困难，特别是在 **instruction fine-tuning** 后生成德语回复时；一人概述了使用 **AutoModel** 进行序列分类时出现的 `ValueError`，暗示存在配置问题。社区还讨论了模型合并、数据集质量和 **prompt** 一致性，强调了在模型集成过程中保持语言质量的挑战。

- **显微镜下的 Grok**：社区在 **GitHub** 上分享了 [Grok 模型发布](https://github.com/xai-org/grok/blob/main/model.py)，探讨了由于其庞大的参数量（3140 亿）及随之而来的计算需求，部署该模型的可行性。

- **评估德语模型掌握程度**：对话引用了诸如 *supergleber-german-language-evaluation-benchmark* 等基准测试，并提到了提供更多信息的 Reddit 帖子和论文。参与者主张在评估平台中创建针对德语的特定基准测试，强调了母语者对语言质量洞察的必要性。

- **语言卓越大学联盟**：有一项提议建议利用德国公立大学的资源来开发能更准确评估语言质量的基准测试，这在扩展 **DiscoLM** 项目的引用中被提及，并倡导学术伙伴关系的价值。

- **演示的乐趣与困境**：*jp1* 分享了在无需特殊调整的情况下在演示中使用 **fastchat/VLLM** 的细节，同时也注意到演示服务器从个人托管迁移到专业托管，不幸地导致了网络问题。*chromix* 提供了一个轻松的对比，暗示更“专业”的托管环境并不总是意味着更高的可靠性。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Prodigy 的新 Prompt Engineering 功能**：**Prodigy** 现在包含了 **prompt engineering** 工具，可将此任务转化为数据标注问题。感兴趣的用户可以在 [Prodigy 功能页面](https://prodi.gy/features/prompt-engineering)探索该产品。

- **Prompt Engineering 的开源辅助工具**：工程社区分享了指向 [hegelai 的 PromptTools](https://github.com/hegelai/prompttools) 和 [PromptFoo](https://github.com/promptfoo/promptfoo) 的链接，鼓励探索这些资源用于 **prompt** 测试以及处理多个 **LLM** 和 **vector databases**。

- **模型基准测试和 Prompt 版本控制 UI 出现**：Vercel 的 [AI Playground](https://sdk.vercel.ai/) 被引用为使用相同 **prompts** 比较不同 AI 模型的工具，而 **Helicone.ai** 新兴的 **prompt** 管理和版本控制功能也正获得认可。

- **AI 增强博客定制尝试**：一名成员承担了一个使用 **GPT-3.5-turbo** 将博客内容适配到不同 **personas** 的项目，在线演示可见于 [How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html)，介绍了用于增强写作重点和清晰度的潜在工具。

- **探索 AI 在博客中的角色**：讨论围绕 AI 增强的博客功能展开，例如以不同的 **personas** 重写、生成反驳观点、基于 **persona** 的内容分享，以及提供摘要或翻译。

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **模型增强方法正在开发中**：一种旨在提高**全局准确率 (global accuracy)**和训练效率的新方法正在准备发布，待改进的图表和结果生成后即可面世。
- **呼吁进行大规模实证验证**：讨论强调，虽然观察到了令人期待的结果，但由于缺乏计算资源，该方法在大规模模型上的有效性实证验证陷入停滞。
- **提供扩展支持**：有人提议讨论这一前景广阔的方法，并探索投入**计算资源 (compute and resources)**来验证并对其进行扩展。
- **在 CIFAR100 上观察到显著提升**：在 CIFAR100 的子集上使用 VGG16 进行一个 epoch 的训练，该方法实现了显著**更高的测试准确率**，展示了初步的成功。
- **讨论图表报告故障**：对话涉及了 Wandb 的技术问题，特别是如何在绘制新实验数据时通过重置步数 (steps) 来有效地更新图表。

---

# PART 2: 频道详细摘要与链接

**Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1219396298176991303)** (1 条消息): 

- **推出 Stable Video 3D**：Stability.ai 宣布发布 **Stable Video 3D**，这是一个基于 Stable Video Diffusion 构建的模型，提供了增强的 3D 质量和多视角能力。它通过输入单张图像并输出多个视角，可用于生成 3D 网格 (3D meshes)；[了解更多关于 Stable Video 3D 的信息](https://stability.ai/news/introducing-stable-video-3d)。
- **优于之前的模型**：SV3D 的发布标志着其性能优于 [Stable Zero123](https://stability.ai/news/stable-zero123-3d-generation) 和其他开源替代方案（如 [Zero123-XL](https://objaverse.allenai.org/docs/zero123-xl/)），承诺大幅提升 3D 技术的质量。
- **发布了两个新的 SV3D 变体**：Stability.ai 发布了两个变体：**SV3D_u** 用于从单张图像生成轨道视频（无需相机调节），以及 **SV3D_p**，它在这些功能的基础上扩展了更多特性。

**提及的链接**：<a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>：当我们发布 Stable Video Diffusion 时，我们强调了视频模型在各种应用中的多功能性。在此基础上，我们很高兴发布 Stable Video 3D。这是一款新模型...

---

**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1218109086101540905)** (988 条消息 🔥🔥🔥): 

- **AI 聊天机器人（目前）还不会写代码**：一位成员对运行 Stable Cascade 的代码表示沮丧，认为其优化很差，可能由聊天机器人编写。他们指出运行 Cascade 的时间比 SDXL 长得多，且 CPU 负载显著。
- **社区期待 SD3 的访问权限**：在对 Stable Diffusion 3 (SD3) 的期待中，社区成员正热切等待更多消息和访问权限，传闻称邀请函可能很快发出。大家猜测并希望 SD3 能在现有模型的基础上改进 prompt 遵循能力。
- **Stability AI 转向加密货币的潜在倾向引发关注**：关于 Stability AI 与区块链和加密货币公司合作的消息引起了社区成员的关注。他们对可能背离开源原则、转向安全性较低且易发诈骗的加密货币集成表示忧虑。
- **在有限的硬件上运行 AI 模型**：成员们讨论了在消费级硬件上运行高级 AI（如 Cascade 或 SD3）的挑战，并比较了不同 GPU 的体验。有人指出，与大型语言模型 (LLM) 相比，图像模型通常对 VRAM 的需求较低。
- **对实用 AI 生成工具的需求日益增长**：社区成员渴望能简化训练或微调过程且不牺牲结果质量的 Stable Diffusion 工具。咨询范围从如何在有限资源下更有效地运行，到针对特定用例（如游戏资产创建）进行微调的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 是一个拥有 314B 参数的 Mixture of Experts 模型 - 基础模型（未经微调） - 8 个专家（2 个激活） - 86B 激活参数 - Apache 2.0 许可证 - 代码： - 祝编码愉快！另：我们正在招聘： </li><li><a href="https://huggingface.co/coqui/XTTS-v2">coqui/XTTS-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/iron-man-mr-clean-mop-ai-floors-gif-27596354">Iron Man Mr Clean GIF - Iron Man Mr Clean Mop - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/avatar-cuddle-hungry-yummy-food-gif-5610436">Avatar Cuddle GIF - Avatar Cuddle Hungry - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/PollyannaIn4D">PollyannaIn4D (Pollyanna)</a>: 未找到描述</li><li><a href="https://tenor.com/view/yess-yes-gif-25420589">Yess GIF - Yess Yes - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.python.org/3/library/pickle.html">pickle — Python object serialization</a>: 源代码：Lib/pickle.py。pickle 模块实现了用于序列化和反序列化 Python 对象结构的二进制协议。“Pickling”是将 Python 对象层级结构转换为...的过程。</li><li><a href="https://www.pny.com/professional/software-solutions/about-nvidia-gpus/nvlink">NVLink | pny.com</a>: 未找到描述</li><li><a href="https://civitai.com/models/207992/stable-video-diffusion-svd)">Stable Video Diffusion - SVD - img2vid-xt-1.1 | Stable Diffusion Checkpoint | Civitai</a>: 查看我们的快速入门指南！ https://education.civitai.com/quickstart-guide-to-stable-video-diffusion/ 基础 img2vid 模型经过训练用于生成...</li><li><a href="https://thedailywtf.com/articles/The_Complicator_0x27_s_Gloves">The Complicator&#39;s Gloves</a>: 优秀的软件在多个方面不断受到攻击。首先是“业余爱好者”，他们尽管只读完了《傻瓜编程》，却不知何故设法拿到了那份巨额合同...</li><li><a href="https://stability.ai/news/introducing-stable-video-3d">Introducing Stable Video 3D: Quality Novel View Synthesis and 3D Generation from Single Images &mdash; Stability AI</a>: 当我们发布 Stable Video Diffusion 时，我们强调了视频模型在各种应用中的多功能性。在此基础上，我们很高兴发布 Stable Video 3D。这个新...</li><li><a href="https://civitai.com/models/351450/proteus-rundiffusion?dialog=commentThread&commentId=372974">Proteus-RunDiffusion - withclip | Stable Diffusion Checkpoint | Civitai</a>: 介绍 Proteus-RunDiffusion。在开发 Proteus-RunDiffusion 的过程中，我们的团队开展了一个探索性项目，旨在提升...的能力。</li><li><a href="https://www.pny.com/professional/software-so">Page Not Found | pny.com</a>: 未找到描述</li><li><a href="https://new.reddit.com/r/StableDiffusion/comments/1b6skvx/wheres_waldo_beach_scenes_as_an_animated_loop/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=fibDNwF8bjs">WKUK - Anarchy [HD]</a>: 最具喜剧色彩的经济无知。—— Murray Rothbard 的《自由、不平等、原始主义和分工》(http://mises.org/daily/3009)。—— "Th...</li><li><a href="https://youtu.be/ruANV24h0Dw?si=rVFKZqowCdpKTzgp">Короткометражный мультфильм &quot;Парк&quot; (сделан нейросетями)</a>: 短篇动画《公园》（由神经网络制作）- 一部非常引人入胜的短篇动画，使用神经网络创作。</li><li><a href="https://github.com/mix1009/sdwebuiapi">GitHub - mix1009/sdwebuiapi: Python API client for AUTOMATIC1111/stable-diffusion-webui</a>: 用于 AUTOMATIC1111/stable-diffusion-webui 的 Python API 客户端 - mix1009/sdwebuiapi</li><li><a href="https://www.youtube.com/watch?v=YTE0OTVOnZU">Vancouver, Canada 1907 (New Version) in Color [VFX,60fps, Remastered] w/sound design added</a>: 我为这段 1907 年加拿大温哥华的视频进行了上色、修复，并添加了天空视觉效果和音效设计。这段视频是在有轨电车上拍摄的，这些...</li><li><a href="https://github.com/DiffusionDalmation/pt_to_safetensors_converter_notebook#">GitHub - DiffusionDalmation/pt_to_safetensors_converter_notebook: This is a notebook for converting Stable Diffusion embeddings from .pt to safetensors format.</a>: 这是一个用于将 Stable Diffusion 嵌入（embeddings）从 .pt 格式转换为 safetensors 格式的 notebook。 - DiffusionDalmation/pt_to_safetensors_converter_notebook</li><li><a href="https://www.youtube.com/watch?v=5mIWo6dgTmI&ab_channel=Megaprojects">The Mushroom Motherboard: The Crazy Fungal Computers that Might Change Everything</a>: 揭开真菌计算的秘密！发现真菌作为生物计算机的惊人潜力。从“森林互联网”到非常规计算...</li><li><a href="https://github.com

/AUTOMATIC1111/stable-diffusion-webui/wiki/API)">Home</a>: Stable Diffusion web UI。通过在 GitHub 上创建账号，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/Stability-AI/generative-models">GitHub - Stability-AI/generative-models: Generative Models by Stability AI</a>: Stability AI 的生成模型。通过在 GitHub 上创建账号，为 Stability-AI/generative-models 的开发做出贡献。</li><li><a href="https://github.com/chaojie/ComfyUI-DragAnything/tree/main">GitHub - chaojie/ComfyUI-DragAnything</a>: 通过在 GitHub 上创建账号，为 chaojie/ComfyUI-DragAnything 的开发做出贡献。</li><li><a href="https://github.com/GraftingRayman/ComfyUI-Trajectory">GitHub - GraftingRayman/ComfyUI-Trajectory</a>: 通过在 GitHub 上创建账号，为 GraftingRayman/ComfyUI-Trajectory 的开发做出贡献。</li><li><a href="https://youtu.be/m9jg1fdOiVY?t=412">在 Mac OS (M1, M2 或 M3) 上安装 ComfyUI</a>: 本视频是一个快速演练，展示如何在 M1 或 M2 Mac 上本地安装 ComfyUI。了解更多关于 AI Animation 的信息，并注册为 AI ...</li><li><a href="https://stable-diffusion-art.com/regional-prompter/)">Regional Prompter: 在 Stable Diffusion 中控制图像构图 - Stable Diffusion Art</a>: 你知道可以为图像的不同区域指定提示词吗？你可以通过 Regional Prompter 扩展在 AUTOMATIC1111 上实现这一点。
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1219057096780419163)** (1 条消息): 

- **Pro 用户可无限次查询 Claude 3 Opus**: 公告透露，**Perplexity Pro 用户**已获得 **Claude 3 Opus** 的每日无限次查询权限，该模型被声称是目前可用的最佳大语言模型 (LLM)。Pro 用户从现在起可以充分利用这一优惠。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1218100055626743851)** (795 条消息 🔥🔥🔥): 

- **Perplexity Pro 的困惑**: 用户对 Perplexity AI 的上下文限制和“无限”声明表示困惑。对话中提到了对 Pro 搜索使用的误解，重点在于 Perplexity 的描述需要更加清晰。
  
- **Claude 3 Opus 讨论**: 用户讨论了 Claude 3 Opus 在 Perplexity AI 中的能力和集成情况，并将其与 GPT-4 及其他模型进行了比较。对话集中在该模型“无限”使用的奥秘以及任何潜在的上下文限制上。
  
- **育儿与 AI**: 一场关于 AI 在向儿童解释复杂话题中作用的热烈辩论爆发了，一位用户主张利用它来简化概念。讨论还涉及儿童的发展能力以及 AI 在教育中的优势。

- **关于 AI 响应能力的辩论**: 用户讨论了 AI 遵循特定提示词的能力，分享了在尝试指示 AI 提供简洁回答或针对儿童问题定制内容时遇到的见解和挑战。

- **潜在的合作伙伴关系与更新**: 围绕苹果、谷歌和生成式 AI 领域的潜在合作伙伴关系及集成出现了猜测，用户分享了新闻链接以及对公司战略的看法。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/3/18/24104626/apple-license-google-gemini-generative-ai-openai-chatgpt">Apple 的 AI 雄心可能包括 Google 或 OpenAI</a>：另一项重大的 Apple / Google 交易可能即将达成。</li><li><a href="https://x.com/AravSrinivas/status/1769475725965566167?s=20">Aravind Srinivas (@AravSrinivas) 的推文</a>：我们已经为 Perplexity Pro 用户取消了 Claude 3 Opus（目前市场上最好的 LLM）的每日查询次数限制！尽情享受吧！</li><li><a href="https://x.com/AravSrinivas/status/1769485603622867394?s=20">Aravind Srinivas (@AravSrinivas) 的推文</a>：是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！ ↘️ Quoti...</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-an">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/shikimori-shikimoris-not-just-cute-shikimoris-not-just-a-cutie-anime-anime-anime-girl-gif-26002811">Shikimori Shikimoris Not Just Cute GIF - Shikimori Shikimoris Not Just Cute Shikimoris Not Just A Cutie Anime - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://us.nothing.tech/pages/perplexity">Nothing Perplexity 优惠</a>：在 Nothing，我们正在构建一个让科技再次变得有趣的世界。还记得每个新产品都让你感到兴奋的时光吗？我们正在带回那种感觉。</li><li><a href="https://fxtwitter.com/BrivaelLp/status/1769482175005577571?s=20">Brivael (@BrivaelLp) 的推文</a>：Zuck 刚刚对 Grok 的发布做出了回应，他似乎并不感冒。“3140 亿参数太多了。你需要一堆 H100，而我已经把它们都买光了” 🤣</li><li><a href="https://x.com/technology/status/1769597406243360937?s=20">Bloomberg Technology (@technology) 的推文</a>：独家：Apple 正在洽谈将 Google 的 Gemini AI 引擎内置到 iPhone 中，这可能是一项重磅交易 https://trib.al/YMYJw2K</li><li><a href="https://youtube.com/clip/Ugkx9gPr2y53Be9C99y-EVVWfZPjRxNQo6FL?si=0r1zDbn2FfjmrsuB">✂️ Sam Altman 谈 AI LLM 搜索</a>：47 秒 · 由 Syntree 剪辑 · 原始视频 "Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419"</li><li><a href="https://youtu.be/OPoWMXqq62Q?si=jk-ZbhjfkZtRkjz7">这些公司在隐藏什么？</a>：关于 Rabbit R1 和 Humane Ai Pin 的看法。如果你想支持本频道，可以考虑点击上方的“加入”按钮成为 Dave2D 会员！http://twit...</li><li><a href="https://fccid.io/2BFB4R1">FCC ID 2BFB4R1 Rabbit Inc. 的 AI Companion</a>：Rabbit Inc. 为 AI Companion 提交的 FCC ID 申请，ID 为 2BFB4R1。包含批准的频率、用户手册、照片和无线报告。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1218101595586429048)** (35 条消息🔥): 

- **使用 Claude 3 Opus 进行创意探索**：使用 **Claude 3 Opus** 进行了一个名为“不断增加智能直到人类无法理解”的有趣创意写作实验。可以在[这里](https://www.perplexity.ai/search/increasing-intelligence-of-HLUn3nOzSx6Nc5ecNpe5pA)进一步探索该任务。
- **可见性是关键**：提醒用户确保他们的主题帖已公开分享，以确保社区可见性。说明见 [Discord 链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **关于清洁度的辩论**：一场关于哪种选项更干净的讨论引起了兴趣，可以在[这里](https://www.perplexity.ai/search/Which-is-cleaner-qIQdwpX1QjiFQvEBgwiydQ)查看。
- **朝鲜的动态**：一项关于**朝鲜金氏**及其行动的 Perplexity 搜索引起了好奇。富有洞察力的结果可在[这里](https://www.perplexity.ai/search/North-Koreas-Kim-.uALFoJfS0mVkML42bECvA)查看。
- **关于未来的问题**：社区分享了关于人类何时可能居住在火星以及其他关于未来的疑问。引人入胜的讨论可在[这里](https://www.perplexity.ai/search/When-can-human-lrFdtQ6NTvCb6LYe.WkreQ)查看。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1218160850670583828)** (64 条消息🔥🔥):

- **模型弃用困惑 (Model Deprecation Confusion)**：`sonar-medium-online` 模型原定于 3 月 15 日弃用，但用户观察到它仍然可以正常工作，而不仅仅是重定向到替代模型。关于弃用是在当天结束时生效还是计划有所改变，引发了各种猜测。
- **API 的得与失 (API Giveth and API Taketh Away)**：在使用 `sonar-medium-online` 时，一位用户发现通过 Web 浏览器获取的新闻与通过 API 获取的新闻之间存在不一致，特别是在关于 Donald Trump 的近期新闻响应上有所不同。
- **在招聘市场的丛林中寻找链接 (Quest for Links in the Job Market Jungle)**：一位用户尝试使用 Perplexity API 获取特定的职位发布链接。值得注意的是，虽然 API 偶尔会提供实际的职位链接，但有时仅返回 LinkedIn 或 Glassdoor 等招聘平台的链接。
- **与 Token 共舞：最大还是最小？ (Dancing with Tokens – Max or Min?)**：讨论了设置 `maxtokens` 参数如何影响 API 的响应。共识显示，如果设置得太低，API 可能会提供不完整的响应；如果设置得太高，它可能不会利用所有可用空间，这表明模型不会“填充”额外空间，而是专注于生成完整的响应。
- **寻找来源与引用 (Seeking Sources & Citations)**：关于 URL 引用的对话确认该功能仍处于 beta 阶段，并为感兴趣的人提供了申请表链接。此外，还讨论了目前“Pro”用户从封闭测试版访问 URL 引用的情况，用户分享了申请链接和模型性能比较的讨论。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai">pplx-api</a>: no description found</li><li><a href="https://perplexity.typeform.com/to/j50rnNiB">pplx-api form</a>: Turn data collection into an experience with Typeform. Create beautiful online forms, surveys, quizzes, and so much more. Try it for FREE.
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1218108656428650526)** (853 messages🔥🔥🔥): 

- **Grok 1：巨兽出笼 (Grok 1: The Behemoth Unleashed)**：Elon Musk 发布了 **Grok 1**，这是一个拥有 3140 亿参数的 Mixture-of-Experts 模型，因其庞大的体积和对大多数用户而言的不切实际性引发了讨论。该模型被预期为训练不足，性能略低于 Miqu，略高于 Llama2 70b，与 Mixtral 相当。

- **QLoRA 的超参数 (Hyperparameters for QLoRA)**：在 Mistral-7b 上微调 **QLoRA** 的首选超参数似乎是 `2e-4` 的学习率和最多 3 个 epochs，正如 Unsloth 的 notebooks 中所建议的那样。不过，鼓励用户根据具体任务和数据集调整这些设置。

- **Discord 中的冒充预警 (Impersonation Alert in Discord)**：用户报告了一个在 Discord 上冒充 **Daniel Han** (`starsupernova`) 的诈骗账号。已向 Discord 提交报告，提醒用户警惕来自该冒充者的好友请求，并在遇到时进行举报。

- **新工具与集成 (New Tools and Integrations)**：AIKit 引入了与 **Unsloth** 的微调集成，为用户提供了使用配置文件微调语言模型的能力，并能使用 Docker 创建兼容 OpenAI 的模型镜像。建议使用 WandB (Weights & Biases) 来监控和可视化训练数据。

- **理解量化 (Understanding Quantization)**：社区对理解语言模型的量化 (Quantization) 持续关注。4-bit BnB quantization 通过减少每个权重的位数来减小模型体积，但也有人寻求学习量化的资源。社区成员还在寻求指令微调 (instruction tuning) 的微调指南和数据集结构。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>: 未找到描述</li><li><a href="https://docs.anthropic.com/claude/page/cosmic-keystrokes">Cosmic keystrokes</a>: 未找到描述</li><li><a href="https://x.ai/about">关于 xAI</a>: 未找到描述</li><li><a href="https://x.ai/blog/grok">宣布推出 Grok</a>: 未找到描述</li><li><a href="https://lightning.ai/live-session/a35263e0-0428-40b6-8828-8e72773a284d">Lightning AI | 将创意转化为 AI，闪电般的速度</a>: AI 开发的一体化平台。协同编码、原型设计、训练、扩展、服务。直接在浏览器中进行，零设置。由 PyTorch Lightning 的创建者打造。</li><li><a href="https://huggingface.co/xai-org/grok-1">xai-org/grok-1 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.ai/">博客</a>: 未找到描述</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T : 在英语和多语言评估中超越 LLaMA 7B 2T (RWKV-v5)</a>: 一个 Linear Transformer 刚刚超越了 Transformer 模型的黄金标准 LLaMA 7B，且在英语和多语言评估中使用的训练 Token 更少。这是历史性的第一次。</li><li><a href="https://arxiv.org/abs/2401.04088">Mixtral of Experts</a>: 我们介绍了 Mixtral 8x7B，这是一种稀疏混合专家 (SMoE) 语言模型。Mixtral 具有与 Mistral 7B 相同的架构，不同之处在于每一层由 8 个前馈块组成 (...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-72B">Qwen/Qwen1.5-72B · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/gemma-bugs">Unsloth 修复 Gemma 错误</a>: Unsloth 正在修复 Google 的开源语言模型 Gemma。</li><li><a href="https://huggingface.co/damerajee/Llamoe-test">damerajee/Llamoe-test · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://sozercan.github.io/aikit/">简介 | AIKit</a>: AIKit 是一个一站式商店，可快速开始托管、部署、构建和微调大语言模型 (LLMs)。</li><li><a href="https://wandb.ai/capecape/alpaca_ft/reports/How-to-Fine-Tune-an-LLM-Part-1-Preparing-a-Dataset-for-Instruction-Tuning--Vmlldzo1NTcxNzE2">如何微调 LLM 第一部分：准备指令微调数据集</a>: 学习如何在指令数据集上微调 LLM！我们将介绍如何格式化数据，并在（几乎）纯 PyTorch 的极简示例中训练像 Llama2、Mistral 等模型。</li><li><a href="https://openhands.ai4bharat.org/en/latest/instructions/datasets.html#supported-datasets">ISLR 数据集 &mdash; 👐OpenHands 文档</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1768991010938404879">来自 Unsloth AI (@UnslothAI) 的推文</a>: Unsloth 本周在 GitHub 上登上了热门榜单！🙌🦥 感谢大家以及所有 ⭐️Stargazers 的支持！查看我们的仓库：http://github.com/unslothai/unsloth</li><li><a href="https://huggingface.co/papers/2402.18668#65f0f5f8de069cd5c55f1dd2">论文页面 - 简单的线性注意力语言模型平衡了召回率与吞吐量

  tradeoff</a>: 未找到描述</li><li><a href="https://huggingface.co/Crystalcareai/GemMoE-Beta-1">Crystalcareai/GemMoE-Beta-1 · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2310.17680">CodeFusion: A Pre-trained Diffusion Model for Code Generation</a>: 想象一下，如果一个开发者只能修改最后一行代码，那么在函数正确之前，他们需要从头开始编写多少次？用于代码生成的自回归模型从...</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: 未找到描述</li><li><a href="https://github.com/AI4Bharat/OpenHands">GitHub - AI4Bharat/OpenHands: 👐OpenHands : Making Sign Language Recognition Accessible. | **NOTE:** No longer actively maintained. If you are interested to own this and take it forward, please raise an issue</a>: 👐OpenHands：让手语识别触手可及。 | **注意：** 不再积极维护。如果您有兴趣接管并推进此项目，请提交 issue - AI4Bharat/OpenHands</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/mixtral/modeling_mixtral.py">transformers/src/transformers/models/mixtral/modeling_mixtral.py at main · huggingface/transformers</a>: 🤗 Transformers: 为 Pytorch, TensorFlow, 和 JAX 提供最先进的机器学习。 - huggingface/transformers</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/jiaweizzhao/GaLore?tab=readme-ov-file#install-galore-optimizer">GitHub - jiaweizzhao/GaLore</a>: 通过在 GitHub 上创建账号来为 jiaweizzhao/GaLore 的开发做出贡献。</li><li><a href="https://huggingface.co/spaces/HirCoir/Piper-TTS-Spanish">Piper TTS Spanish - a Hugging Face Space by HirCoir</a>: 未找到描述</li><li><a href="https://github.com/xai-org/grok-1/issues/6#issuecomment-2002664859">Error when installing requirements · Issue #6 · xai-org/grok-1</a>: 我已经安装了 python 3.10 和 venv。尝试执行 &quot;pip install -r requirements.txt&quot; 错误：忽略了以下需要不同 python 版本的版本：1.6.2 Requires-Python &gt;=3...</li><li><a href="https://github.com/mistralai/mistral-src">GitHub - mistralai/mistral-src: Reference implementation of Mistral AI 7B v0.1 model.</a>: Mistral AI 7B v0.1 模型的参考实现。 - mistralai/mistral-src</li><li><a href="https://www.youtube.com/watch?v=jvqFAi7vkBc">Sam Altman: OpenAI, GPT-5, Sora, Board Saga, Elon Musk, Ilya, Power &amp; AGI | Lex Fridman Podcast #419</a>: Sam Altman 是 OpenAI 的 CEO，该公司是 GPT-4, ChatGPT, Sora 以及许多其他最先进 AI 技术的幕后推手。请通过以下方式支持本播客...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 速度快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://youtu.be/rANv5BVcR5k">Mistral Fine Tuning for Dummies (with 16k, 32k, 128k+ Context)</a>: 在我们最新的教程视频中，探索使用您自己的数据轻松微调语言模型 (LLMs) 的秘密。我们深入探讨了一种具有成本效益且...</li><li><a href="https://the-decoder.com/falcon-180b-open-source-language-model-outperforms-gpt-3-5-and-llama-2/">Falcon 180B open-source language model outperforms GPT-3.5 and Llama 2</a>: 开源语言模型 FalconLM 提供了比 Meta 的 LLaMA 更好的性能，并且也可以用于商业用途。如果收入超过 100 万美元，商业使用需支付版税。</li><li><a href="https://huggingface.co/datasets/teknium/GPT4-LLM-Cleaned">teknium/GPT4-LLM-Cleaned · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/97">Staging PR for implimenting Phi-2 support. by cm2435 · Pull Request #97 · unslothai/unsloth</a>: ….org/main/getting-started/tutorials/05-layer-norm.html]</li><li><a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: Add GaLore optimizer by younesbelkada · Pull Request #29588 · huggingface/transformers</a>: 此 PR 做了什么？如标题所示，添加了来自 https://github.com/jiaweizzhao/GaLore 的 GaLore 优化器。修复了：#29512 这是我目前测试 API 的方式：import torch import datasets from ...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[announcements](https://discord.com/channels/1179035537009545276/1179039782681202829/1218580567453470860)** (1 条消息):

- **Unsloth AI 在 GitHub 上大放异彩**：Unsloth AI 本周在 GitHub 上的活跃度激增，成为热门项目。Unsloth 团队向社区和关注者表示感谢，并邀请更多用户为他们的[更快速、更高效的微调项目](https://github.com/unslothai/unsloth)点亮 Star。

**提到的链接**：<a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning - unslothai/unsloth

  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1218112720994308122)** (25 条消息🔥): 

- **思想的巧合**：一场关于“巧合”的讨论展开，成员们分享了想到某事后在别处偶遇的经历。有人举例说想到一个用户名后看到别人在使用，并认为我们的大脑会潜意识地获取信息，这与儿童的学习方式类似。

- **鼓励创意表达**：成员们互相鼓励彼此的独白，并乐于分享和讨论**诗歌创作**，展现了社区对创意尝试的支持。

- **探索分类任务的微调**：在 AI 微调领域，一位成员分享了使用 **Mistral-7b** 处理**特定领域分类任务**的经验，并思考是否尝试 **Gemma 7b**。另一位成员向大家保证 Unsloth 的所有 Bug 修复已完成，并建议 Gemma 和 Mistral 的优势可能各不相同。

- **AI 模型分支的澄清**：一位成员寻求帮助寻找 AI 模型的 “Mixtral 分支”。热心的回复引导他们找到了正确位置，并提供了 GitHub 上相关 Pull Request 的链接（[Mixtral 支持 Pull Request](https://github.com/unslothai/unsloth/pull/145)）。

- **分享开源 UI 元素和地图**：社区内分享了资源链接，包括 RL 扑克游戏中 Agent 的地图可视化，以及使用 CSS 或 Tailwind 制作的开源 UI 元素集合（[UIverse UI Elements](https://uiverse.io/elements)）。

- **对申请的支持**：成员们对那些考虑申请未指明机会的人表达了支持，增强了社区内友好鼓励的氛围。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pwhiddy.github.io/pokerl-map-viz/">Pokemon Red Map RL Visualizer</a>: 未找到描述</li><li><a href="https://uiverse.io/elements">4217 UI elements: CSS &amp; Tailwind</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/145">[WIP] add support for mixtral by tohrnii · Pull Request #145 · unslothai/unsloth</a>: Mixtral WIP
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1218104575022727230)** (568 条消息🔥🔥🔥): 

- **处理模型保存期间的 VRAM 需求**：一位用户注意到在保存过程中，除了加载模型所需的显存外，VRAM 使用量也很高。在 8GB VRAM 的机器上保存 Mistral-7b bnb 4bit 模型导致了崩溃，这表明成功保存模型需要充足的 VRAM 以及额外的系统 RAM。

- **模型保存期间清理 VRAM 可能无济于事**：当有人建议通过重启电脑清理 VRAM 来解决保存崩溃问题时，有人澄清说模型必须加载到 VRAM 中才能保存，因此重启是不够的。

- **用于训练和保存模型的 Colab 资源**：一位用户在最初失败后成功在 Colab 中运行了代码，强调了在该平台上获得足够资源的运气成分。

- **Colab 与本地机器保存模型的差异**：8GB VRAM 似乎适合运行 Mistral-7b bnb 4bit 模型，这凸显了 Colab 与本地设置在 VRAM 需求上的差异。

- **针对模型合并策略**：有人建议将合并 UltraChat 与基础 Mistral 时使用的策略应用于 Mistral-Yarn，讨论中基于以往模型合并方法的经验，表现出怀疑与乐观并存的态度。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://huggingface.co/ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit">ybelkada/Mixtral-8x7B-Instruct-v0.1-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1X_PHYBawrsCgKfMEPxvIDX__rYa1-v97?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://www.kaggle.com/code/danielhanchen/kaggle-mistral-7b-unsloth-notebook">Kaggle Mistral 7b Unsloth notebook</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自无附加数据源的数据</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/mistral-7b-instruct-v0.2-bnb-4bit">unsloth/mistral-7b-instruct-v0.2-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/trl/main/en/dpo_trainer#accelerate-dpo-fine-tuning-using-unsloth">DPO Trainer</a>: 未找到描述</li><li><a href="https://github.com/artidoro/qlora/blob/main/qlora.py#L746">qlora/qlora.py at main · artidoro/qlora</a>: QLoRA: 量化 LLMs 的高效微调。通过在 GitHub 上创建账号为 artidoro/qlora 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-to-gguf">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#chat-templates">Home</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">Generation - GPT4All 文档</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm">GitHub - vllm-project/vllm: 一个针对 LLMs 的高吞吐量且显存高效的推理与服务引擎</a>: 一个针对 LLMs 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing#scrollTo=FqfebeAdT073,">Google Colaboratory</a>: 未找到描述</li><li><a href="https://pastebin.com/ybSeKHhU">Unsloth: 将 4bit 和 LoRA 权重合并为 16bit...Unsloth: 将使用高达 5.34 - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments">Trainer</a>: 未找到描述</li><li><a href="https://github.com/huggingface/trl/issues/1041">Does DPOTrainer loss mask the prompts? · Issue #1041 · huggingface/trl</a>: 你好，有个小问题，DataCollatorForCompletionOnlyLM 会通过屏蔽提示词的损失来仅对回答进行训练。DPOTrainer (DPODataCollatorWithPadding) 也是这样工作的吗？看起来...</li><li><a href="https://huggingface.co/docs/trl/v0.7.11/en/sft_trainer#train-on-completions-only).">Supervised Fine-tuning Trainer</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha/discussions/2/files">HuggingFaceH4/zephyr-7b-alpha · 添加聊天模板</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceH4/zephyr-7b-alpha#intended-uses--limitations">HuggingFaceH4/zephyr-7b-alpha · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/chat_templates.py#L56">unsloth/unsloth/chat_templates.py at main · unslothai/unsloth</a>: 提速 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github

<ul>
<li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>：llama.cpp 的 Python 绑定。通过在 GitHub 上创建账号来为 abetlen/llama-cpp-python 的开发做出贡献。</li><li><a href="https://github.com/huggingface/alignment-handbook/issues/45#issuecomment-1845598205">在 MT-Bench 上复现 Lora 模型结果 · Issue #45 · huggingface/alignment-handbook</a>：最近，我尝试在自己的数据集上拟合 DPO。最初，我尝试复现你们 LORA 模型的结果（MT-Bench 上的得分为 7.43）。然而，我遇到了一些问题。尽管使用了你们所有的……
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1218239216975351928)** (21 messages🔥): 

- **阅读清单材料**：一位成员提到他们在 Twitter 上看到的一篇 *amazing paper*，并将其加入了阅读清单。
- **训练时长辩论**：关于训练模型的最佳 epoch 数量展开了讨论，一位成员建议最多 4 个 epoch，并指出 **3 个 epoch 是微调语言模型的标准**。
- **寻找平衡点**：在追求最大化知识保留的过程中，一位成员被建议不要使用过多的 epoch，因为这可能导致模型死记硬背数据集，而无法保留更广泛的知识。
- **参数与 Token 比例受到质疑**：另一场对话围绕可训练参数与数据集大小的合适比例展开，暗示拥有 800,000 行的数据集可能需要 32 或 64 的 rank，并建议 **alpha = rank * 2**。
- **模型集成建议**：一位成员分享了 Hugging Face 上 **Tiny Mistral** 和 **Tiny Mistral Instruct** 的链接，这些小型模型可能会集成到 Unsloth Repository 中，并简要介绍了模型的配置。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1">Dans-DiscountModels/TinyMistral-v2.5-MiniPile-Guidelines-E1 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/M4-ai/TinyMistral-6x248M-Instruct/tree/main">M4-ai/TinyMistral-6x248M-Instruct at main</a>：未找到描述
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1218098224586293319)** (301 messages🔥🔥): 

- **好奇的新手与老手**：该频道欢迎新成员，例如一位渴望在 Mac M3 Pro 上探索 LLM 的热心软件工程师，以及一位对潜入 AI 世界感到兴奋的自称“好奇极客”。社区为入门模型和在特定硬件配置上运行的模型提供了建议。

- **寻求指导与解决方案**：用户就软件问题寻求建议，例如卡在验证文件完整性循环中、配置在 LM Studio 中使用的 GPU，以及解决 Kali Linux 中的 JavaScript 错误。在许多情况下，社区成员提供了故障排除协助和变通方法，例如通过 NVIDIA Control Panel 隐藏 GPU。

- **工具、支持与插件讨论**：社区讨论了各种集成，例如在 VSCode 中使用 continue 扩展进行 autopilot coding，以及在本地运行模型（包括像 Grok-1 这样的大型模型）的限制，以及考虑 GPU 资源时的模型大小限制。特别是一位用户分享了将 Visual Studio Code 与 LM Studio 集成以完成编程任务的成功经验。

- **探索模型能力并意识到局限性**：用户询问模型在 LM Studio 中读取和处理文件及文档的潜力，以及是否支持 function 或文档检索。其他人则在思考由于庞大的尺寸和参数，在本地运行像 Grok-1 这样的开源模型的可行性。

- **LM Studio 开发与支持查询**：关于 LM Studio 持续开发的讨论不断涌现，包括即将对 commandr、starcoder2 和 miqu-103B 等特定模型的支持。用户还参与了为 OpenChat 集成创建聊天模板以及推荐适合学习 Python 的模型的讨论。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>: Grok-1 是一个 314B 参数的 Mixture of Experts 模型 - 基础模型（未微调） - 8 个专家（2 个激活） - 86B 激活参数 - Apache 2.0 许可证 - 代码： - 祝编码愉快！另：我们正在招聘：</li><li><a href="https://tenor.com/view/ratha-gif-26742750">Ratha GIF - Ratha - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/xai-org/grok-1/discussions/30">xai-org/grok-1 · 314B 参数有 297G 文件大小？</a>：未找到描述</li><li><a href="https://github.com/continuedev/continue/issues/713"">Issues · continuedev/continue</a>: ⏩ 使用任何 LLM 编码的最简单方法——Continue 是适用于 VS Code 和 JetBrains 的开源自动驾驶仪 - Issues · continuedev/continue</li><li><a href="https://www.youtube.com/watch?v=lCZRwrRvrWg&">Mistral: Easiest Way to Fine-Tune on Custom Data</a>: 此视频由 Gradient.ai 赞助，点击此处查看：https://gradient.1stcollab.com/engineerprompt。在本视频中，我们将学习如何微调 Mistr...</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g&">[1hr Talk] Intro to Large Language Models</a>: 这是一个面向普通观众的 1 小时 Large Language Models 介绍：ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1218119135423234058)** (138 条消息🔥🔥): 

- **对 Command-R 模型支持的期待**：成员们热切期待 Command-R 模型与 LM Studio 的集成，并询问 Beta 测试权限。目前的讨论表明 LM Studio 的下一个版本将支持 Command-R；[llama.cpp 的 Pull Request #6033](https://github.com/ggerganov/llama.cpp/pull/6033)（添加了该模型）已被合并，正等待 LM Studio 更新。

- **Grok 模型热度**：xAI 新发布的 Grok-1 基础模型引发热议，讨论集中在其巨大的体量以及硬件和托管的潜在成本上。成员们分享了关于 Grok 的想法和信息，包括 [ycombinator 上的讨论](https://news.ycombinator.com/item?id=39737281)和包含更多细节的[博客文章](https://x.ai/blog/grok-os)。

- **寻求小型且高效的模型**：显存（VRAM）有限的用户正在寻找能在 RTX 2070 Super 和 GTX 1660 Super 等 GPU 上运行的模型建议。共识建议像 Gemma 2B 或高量化（quantizations）版本的 Mistral 7B 这样的小型模型可以在硬件限制内运行。

- **关于 OpenChat 聊天模板的咨询**：用户正尝试为 OpenChat 配置自定义聊天模板，其中一人为 Yi-9B-200K 等模型[提出了模板结构](https://huggingface.co/01-ai/Yi-9B-200K)；讨论表明，个人实验和查阅文档是正确设置的关键。

- **Yi 模型架构的好奇**：Yi-9B-200K 模型的架构和能力引发了好奇，导致了关于 Transformer 架构、参数重要性和上下文长度（context length）的对话。分享了 Andrej Karpathy 的“[Intro to Large Language Models](https://youtu.be/zjkBMFhNj_g)”演讲和补充 YouTube 视频等教育资源以帮助理解。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=39737281">未找到标题</a>: 未找到描述</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: 来自多模态 LLM 预训练的方法、分析与见解</a>: 在这项工作中，我们讨论了构建高性能多模态大语言模型 (MLLMs) 的方法。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细的...</li><li><a href="https://huggingface.co/01-ai/Yi-34B/discussions/23">01-ai/Yi-34B · Prompt 模板？</a>: 未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://youtu.be/PAbZRGGYNyM?si=xVNZCYUddDvoFUly">什么是大语言模型中的参数？</a>: 什么是大语言模型中的参数？00:26 💡 像 GPT-3 这样的大语言模型中的参数是在训练过程中学习到的变量，用于最小化...</li><li><a href="https://youtu.be/zjkBMFhNj_g?si=Rn96V9CMqEHLy6-7">[1小时演讲] 大语言模型入门</a>: 这是一个面向普通观众的 1 小时大语言模型介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6033">由 acanis 添加 Command-R 模型 · Pull Request #6033 · ggerganov/llama.cpp</a>: 关于 Command-R 35B 模型（128k 上下文）的信息可以在以下网址找到：https://huggingface.co/CohereForAI/c4ai-command-r-v01 基于 llama2 模型并进行了一些更改：新的超参数...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1218213037060657273)** (12 messages🔥): 

- **关于 Command-R 35B 兼容性的困惑**：一场关于 [Hugging Face 仓库](https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF) 的讨论引发了关于 llama.cpp 与 CohereForAI 的 Command-R 模型兼容性的困惑。虽然 GGUF 格式已经可用，但有人澄清说 llama.cpp 目前不支持 c4ai 模型。
- **关于 llama.cpp 支持的矛盾信息**：一位成员澄清了误解，指出 llama.cpp 实际上支持 c4ai 模型，这与对话中之前的消息相矛盾。
- **呼吁 AMD OpenCL 驱动通知**：有人建议在网站的 Linux 下载页面通知 AMD 用户，他们需要 OpenCL 驱动程序才能在程序中使用 GPU。
- **寻求 AI 使用困难的指导**：一位用户对使用 AI 的复杂性表示沮丧，并被引导至特定频道，大概是为了获得更好的支持和详细的帮助。
- **查询 LM Studio 的功能**：有用户询问是否可以在 LM Studio 中使用个人文档进行对话，或者是否可以集成 autogen 等插件。据解释，autogen/langchain 等插件已经通过服务器模式（server mode）连接得到支持。

**提到的链接**：<a href="https://huggingface.co/andrewcanis/c4ai-command-r-v01-GGUF">andrewcanis/c4ai-command-r-v01-GGUF · Hugging Face</a>：未找到描述

  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1218129474348912711)** (480 messages🔥🔥🔥): 

- **关于最佳 GPU 选择的辩论**：社区成员正在讨论即将推出的 5090 GPU 在 LM 任务中的预期性能和价值，并将其与 3090 和 4090 进行比较。观点表明，虽然 5090 在通用 AI 任务中可能提供更好的性价比，但其带宽/价格比可能不会超过 3090。

- **对单槽 5090 的期望**：有人表达了对单槽版 5090 GPU 的渴望，以方便多 GPU 配置。此外，还讨论了 Fractal North 机箱在容纳此类配置方面的有效性，以及对散热需求的观察，例如 Corsair 7000x 全塔机箱在管理功耗和散热方面的功效。

- **追求最多的 PCIe 4.0 插槽**：寻找一款至少具有两个 x16 Gen 5 插槽的主板是一位用户的目标，因为这将改善新 GPU 配置的潜力。有人询问了 Corsair 7000x 配置的功耗，以衡量其散热性能。

- **LM Studio 在工作环境中的适用性**：讨论涉及了 LM Studio 在工作环境中的使用条款，并分享了链接以澄清许可和要求。大家认识到，在企业环境中采用此类工具之前，必须经过审批流程。

- **多 GPU 设置挑战**：分享了使用 PCIe 转接线设置多 GPU 的困难经验，其中 OCuLink 线缆和额外的 PSU 被强调为成功的解决方案。对话详细说明了为了确保功能正常，将所有 GPU 置于相同 PCIe 代际插槽中的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.amazon.com/AMD-3200MHZ-SYSTEM-COMPONENTS-PROCESSORS/dp/B07XP9S55C/ref=sr_1_2">未找到标题</a>: 未找到描述</li><li><a href="https://www.amazon.de/-/en/HHCJ6-NVIDIA-Server-Accelerator-Renewed/dp/B07GJ45V3D/ref=sr_1_2?crid=1O8IZM1RV0TIH&dib=eyJ2IjoiMSJ9.B2ZUEDxvj_Z73GUX0GJebEDmX0cqUrowZhMOgYhwtCaPdx9UH8NiM39aqowgVAc5YENjqRh8_cc1qHbgwPJMprvhMhnuusRAJuQqLmWDyskupHMP8ACQI354KZZjKYrdtnPPNGnuoJdVlHxoPQ8ll9ilsDZZ334_L6TwueHlrTelgoIjaTt650I3FQyWgOFmpTvAb3YigqPDURnBJMq1D6wanBHjVSaSdFOEnWlP2cUV8J9Hq4Lh_0bJbRh-kAaca58OndCeXm-tGVmNFLi7TuMKGZORpZ0Q6IcMd6Vz11w.MFnlYLfXX9YWUon0J_Dg0ds2eKFM6AwZgazWMdxeEjE&dib_tag=se&keywords=Tesla+K80&qid=1710787582&s=computers&sprefix=tesla+k80%2Ccomputers%2C421&sr=1-2">未找到标题</a>: 未找到描述</li><li><a href="https://lmstudio.ai/#can-i-use-lm-studio-at-work?">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://coral.ai/products/m2-accelerator-dual-edgetpu#description">带有双 Edge TPU 的 M.2 加速器 | Coral</a>: 使用 M.2 (E key) 接口将两个 Edge TPU 集成到旧系统和新系统中。</li><li><a href="https://lmstudio.ai/beta-releases.html">LM Studio 测试版发布</a>: 未找到描述</li><li><a href="https://www.aliexpress.com/item/100500634581">404 页面</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/273788651049?">Dell T710 塔式服务器 双 6 核 X5650 **144Gb RAM** 240gb SSD + 6X 600G SFF SAS | eBay</a>: 未找到描述</li><li><a href="https://www.aliexpress.com/item/1005006525215524.html">未找到标题</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/115960685949?">AMD EPYC 7F72 CPU 处理器 24 核 3.20GHz 192MB 缓存 240W - 100-000000141 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/126352871326?epid=11041255665&itmmeta=01HS9333CQ68S4STA8BZJ3V0BH&hash=item1d6b37cf9e:g:DOEAAOSweRlkuVOG&itmprp=enc%3AAQAJAAAA0GtLL6BuVwKKMH1iyVWS1kdp6p0LvQb%2Fcu8c94aisQZDISgf4yKcfrjNbigVkO4IGdfBt3tcIr6du3Nb1xXGbEe2CNScd%2B4RoCdoEx%2BQMPtNGs0TtY3wzAbszVam1AHN8tC%2Bzq%2BVoVhSwCmdZ77779duZUVHF%2Fq1ckL28OWoVp%2FRStC3u0NyyTZtUke6tEsgNdQYOKI4%2BqNOIN11tc8XuhOtaovFo6WzH87nIC6BUNiaWYnvWcqUPH3NUs6Gxi%2FWnel1Vj9wokxL8oELjbCFBOA%3D%7Ctkp%3ABFBMyLaMo8pj">AMD EPYC 7232P CPU 处理器 8 核 3.10GHz 32MB 缓存 120W - 100-000000081 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.ca/itm/126375063761">AMD EPYC 7232P 8 核 3.1GHz 32MB L3 处理器 - Socket SP3 - 100-000000081 | eBay</a>: 未找到描述</li><li><a href="https://www.newegg.com/asrock-rack-romed8-2t/p/N82E16813140044">Asrock Rack ROMED8-2T ATX 服务器主板 AMD EPYC 7003 (带有 AMD 3D V-Cache 技术)/7002 系列处理器 SP3 (LGA 4094) 双 10GbE - Newegg.com</a>: 购买 Asrock Rack ROMED8-2T 服务器主板 AMD EPYC 7003 (带有 AMD 3D V-Cache 技术)/7002 系列处理器 SP3 (LGA 4094) 双 10GbE，享受快速发货和顶级客户服务。一旦您...</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d6">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU 加速器 12 个月保修 | eBay</a>: 未找到描述</li><li><a href="https://www.thingiverse.com/search?q=K80+cooling+&page=1&type=things&sort=relevant">搜索 Thingiverse - Thingiverse</a>: 下载文件并使用您的 3D 打印机、激光切割机或 CNC 进行制造。</li><li><a href="https://www.ebay.de/itm/125947603377?itmmeta=01HS9HRSJMXBV00M1XW59H5NAE&hash=item1d530fe9b1:g:fHQAAOSwWVxkbefZ&itmprp=enc%3AAQAJAAAA4A6tXSRz7NxXocQqxCeo%2F2TdOTiIP1AMtfRCBxeBISSicEa3bP%2FtSfa9CmVAH74vTwUFyfwFd1VhNC71wMalgSqfYNDwr7svQreF5j3Gqk4Brm8Zn7hMHU6mRQVuxRyyv5VyA1PeZKdylhbJH0O%2BC2IM8GdP7yLRbRw6sOGTb2KMO0V0m%2B7aGkzXe6h33qOgF16cjz2vh2TITEEOr1eYGfz7ViQZ846gljR8VFArZiDwxgIU8naY8yQRPUJe4Znn3GYEn3GT3DNHxdg5zoB7qyMOytwL9TKozBLIkBQVtyyq%7Ctkp%3ABk9SR8KZ47HKYw">新款 /Wave ®AI 服务器 NF5688M6 NVIDIA HGX TESLA A800 80G 八路 GPU 服务器/期货 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.co.uk/itm/296113403496?">Dell T710 塔式服务器 双 6 核 X5670 **24 核** 64GB RAM | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.de/itm/145329120119?epid=507128083&itmmeta=01HS9DKVRXS2WQPFX74KY649GW&hash=item21d64a6377:g:kacAAOSw~q1lFEwb&itmprp=enc%3AAQAJAAAA4GTzwRZBHO82ltgqug5ARkRZ5JKlaikKECFytG5%2FNjvBMzyE2UGOBW0yRbeW%2B%2F3prx2LD9sPaLsinW103607IHMVVMe2tg6FIa2KVc%2FUVWqCGgQPrRRS97i9Q%2FZW0nnLz5XSLuFob%2FicmlhLi7Ve68FV47SLRenj5tDoUD8mwpvdoxA5uQtR0DNACYnvlVQe4BeXKFAWKA8iKA6WdrVikWOsQcODTpcW916%2FL8jFOUSFjg9D5%2FP1xg4foswYBWrIeaD4Pm9rguigAFQvYGqHFLKNXgB4CjCD0BczHhSZYunI%7Ctkp%3ABk9SR8i8z63KYw">Nvidia Tesla K80 24GB GPU GDDR5 PCI-E GPU 加速器 12 个月保修 | eBay</a>: 未找到描述</li><li><a href="https://zifa666.aliexpress.com/store/5885523/pages/all-items.html?productGroupId=40000003590095&shop_sortType=bestmatch_sort">Luckim 官方商店 - 惊人的产品</a>

ucts with exclusive discounts on AliExpress</a>: 未找到描述</li><li><a href="https://www.techpowerup.com/cpu-specs/core-i5-3470.c1039#:~:text=Programs%20using%20Advanced%20Vector%20Extensions,performance%20for%20calculation%2Dheavy%20applications.">Intel Core i5-3470 Specs</a>: Ivy Bridge, 4 Cores, 4 Threads, 3.2 GHz, 77 W</li><li><a href="https://www.microcenter.com/product/677156/nvidia-geforce-rtx-3090-founders-edition-dual-fan-24gb-gddr6x-pcie-40-graphics-card-(refurbished)">Micro Center - Computers and Electronics</a>: Micro Center - 计算机与电子产品 - 数千种可购买的产品：台式机、笔记本电脑、显示器、自建 PC 零件、升级组件、数字成像、打印用品、便携式设备、音频设备...</li><li><a href="https://www.aliexpress.com/item/1005006345813657.html">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1219065221327355974)** (4 messages): 

- **寻求不同模型的预设**：一位用户询问是否有针对不同模型的完整预设列表。回复提供了一个 [GitHub 链接](https://github.com/lmstudio-ai/configs)，其中包含 JSON 配置文件以及 LM Studio 的示例配置集合。

- **寻找 ROCm 用户**：一位用户询问聊天室中是否有 ROCm 用户。另一位用户将他们引导至代码为 `#1195858490338594866` 的特定频道，以进行可能更有帮助的讨论。

**提到的链接**：<a href="https://github.com/lmstudio-ai/configs">GitHub - lmstudio-ai/configs: LM Studio JSON configuration file format and a collection of example config files.</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs

  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1219051718172606537)** (1 messages): 

- **咨询 Local Inference Server 能力**：一名成员询问是否有人成功将具有 JSON function calling 功能的模型集成到 **Local Inference Server** 中。目前没有提供进一步的细节或后续跟进。
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1219383598193311744)** (5 messages): 

- **AVX Beta 说明**：一名成员询问 Beta 版应用是否使用了 AVX 指令，并猜测其 Beta 状态是因为使用了 AVX。
- **Beta 版应用详情披露**：确认该 Beta 版应用是一个**旧版本**，且 AVX 支持并非团队目前的高优先级任务。
- **模型兼容性问题**：一名成员询问模型在 Beta 版应用中是否像在新版中一样工作，得到的澄清是：虽然模型可以运行，但不支最新的模型（如 **starcoder2, gemma** 等）。
- **Beta 版上的 Mistral 模型**：在询问后，一名成员获知他们可以在 Beta 版应用上运行 **Mistral** 模型。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1218206050495234070)** (5 messages): 

- **GitHub 上的预构建 ROCm 库**：一名成员分享了一个 [GitHub 链接](https://github.com/brknsoul/ROCmLibs)，指向支持 gfx1031 和 gfx1032 的预构建 Windows ROCm 库。该链接指向一个旨在帮助使用特定 AMD GPU 用户的仓库。
- **LM Studio 尚不支持双 GPU**：一名成员询问在 LM Studio 中将 AMD GPU (6700 xt) 与他们的 7800 xt 配合使用的情况，并指出该软件目前似乎仅利用主 GPU。他们寻求确认是否很快会支持多 GPU。
- **ROCm 不支持 AMD GPU 6700 xt**：另一名成员澄清说，ROCm 官方并不支持 AMD GPU 6700 xt，这就是为什么它无法在 LM Studio 中工作的原因，因为后者使用的是 ROCm 库。
- **在 LM Studio 中并行使用 7000 系列 AMD GPU**：在澄清了 6700 xt 的支持情况后，同一名成员推测，如果有两块 7000 系列 GPU，LM Studio 可能会并行利用它们。

**提到的链接**：<a href="https://github.com/brknsoul/ROCmLibs">GitHub - brknsoul/ROCmLibs: Prebuild Windows ROCM Libs for gfx1031 and gfx1032</a>: 为 gfx1031 和 gfx1032 预构建的 Windows ROCM 库 - brknsoul/ROCmLibs

  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1219265025487667200)** (1 messages): 

- **Agent 系统选择流程**：一名成员询问了关于选择 **agent 系统** 以验证不同 agent 创意概念的进展情况。他们专门联系了另一名成员，以了解其决策过程的最新动态。
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1218144997094723615)** (56 messages🔥🔥):

- **NVIDIA RTX 50 系列 GDDR7 显存速度见解**：分享的一篇文章描述了 NVIDIA 计划为其 GeForce RTX 50 系列 "Blackwell" 显卡配备速度为 28 Gbps 的 GDDR7 显存，尽管目前已有更快的 32 Gbps 芯片。文章根据历史先例和潜在的显存总线宽度推测了 NVIDIA 的策略。
  
- **期待 AI 界面（Interfaces）的进步**：成员们讨论了即将推出的 AI 模型在显著改进 Agent 界面方面的潜力，认为未来的进步可能会将新模型开发与针对 Agent 的定制化相结合。
  
- **游戏数据面向 AI 开发开放**：[MatchboxDAO](https://x.com/unkjdgames?s=21) 宣布一款游戏已向开发者开放其数据，用于创建 AI Agent，并为感兴趣的社区贡献者提供资金支持。
  
- **预测 AI 在社会中的未来角色**：回顾了 Sam Altman 的一项预测，推测了 AI 不断演进的能力，范围从法律和医疗应用到流水线任务，并最终走向机器人伴侣。
  
- **社区讨论交互式 AI Agent**：围绕寻求让 AI 助手在对话中更具响应性的解决方案展开了对话，包括在被中断时智能暂停，并在用户插话后恢复。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/unkjdgames?s=21">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">使用 Langgraph 进行计划与执行</a>：如何创建一个“计划与执行”风格的 Agent。这在很大程度上受到了 Plan-and-Solve 论文以及 Baby-AGI 项目的启发。核心思想是首先...</li><li><a href="https://www.techpowerup.com/320185/nvidia-geforce-rtx-50-series-blackwell-to-use-28-gbps-gddr7-memory-speed">NVIDIA GeForce RTX 50 系列 "Blackwell" 将使用 28 Gbps GDDR7 显存速度</a>：据可靠爆料者 kopite7kimi 称，首批采用 GDDR7 显存的 NVIDIA GeForce RTX 50 系列 "Blackwell" 显卡传闻将配备 28 Gbps 的显存速度...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1218108265854926899)** (16 条消息🔥): 

- **由 "Horny Claudes" 绘制的 Mermaid 图表**：[Repligate 的 Twitter 帖子](https://x.com/repligate/status/1768521441329434937?s=20) 提到了创建一个 "horny Claudes" 网络，据称该网络能生成更好的 Mermaid 图表，这表明模型的状态可能会影响生成的图表质量。评论对这一概念表达了震惊和幽默。

- **Apple 发布 AI 模型信息**：[Apple 讨论了其 AI 模型的细节](https://twitter.com/arankomatsuzaki/status/1768446729710371115)，引发了关于近期从专有渠道分享 AI 模型信息的讨论。讨论中包含了对未发布模型权重（Weights）的失望。

- **AI 对齐（Alignment）的前沿**：[Hugging Face 上的一篇摘要](https://huggingface.co/papers/2403.07691) 探索了一种名为 ORPO 的新算法，用于语言模型的偏好对齐监督微调（Preference-aligned supervised fine-tuning），据称该算法消除了偏好对齐的额外阶段，在不同规模的模型中都显示出前景。

- **复现 MetaAI 的自我奖励语言模型**：[Oxen.ai 社区](https://github.com/Oxen-AI/Self-Rewarding-Language-Models) 尝试复现 MetaAI 的 Self-Rewarding Language Model 论文，为在开源社区中复制研究成果做出了贡献。

- **将 LLM Agent 统一为计算图**：一篇 [研究论文](https://arxiv.org/abs/2402.16823) 介绍了一个新框架，将基于大语言模型的 Agent 视为计算图，并可以进行自动优化，从而实现更高效的问题解决架构。社区对此反应热烈，赞赏这种统一零散 LLM 功能的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/burny_tech/status/1769530798242255129">来自 Burny — Effective Omni (@burny_tech) 的推文</a>：关于马斯克可能通过 Grok 引领开源，从而动摇情报战争中其他巨头玩家的看法。Grok-1 是一个 314B 参数的模型，采用了 Mixture of Experts 架构...</li><li><a href="https://arxiv.org/abs/2402.16823">Language Agents as Optimizable Graphs</a>：为了改进基于 Large Language Models (LLMs) 的问题求解器，人们提出了各种人工设计的 Prompt Engineering 技术，产生了许多不同的代码库。我们将这些方法统一起来...</li><li><a href="https://huggingface.co/papers/2403.07691">论文页面 - ORPO: Monolithic Preference Optimization without Reference Model</a>：未找到描述</li><li><a href="https://x.com/repligate/status/1768521441329434937?s=20">来自 j⧉nus (@repligate) 的推文</a>：@xlr8harder 我没让它发展太远，但现在房间里有人跟我说他们如何创建了一个“horny claudes”网络，以及这些 Claude 如何创造更好的...</li><li><a href="https://github.com/Oxen-AI/Self-Rewarding-Language-Models">GitHub - Oxen-AI/Self-Rewarding-Language-Models：这是 Oxen.ai 社区的工作，旨在复现 MetaAI 的 Self-Rewarding Language Model 论文。</a>：这是 Oxen.ai 社区的工作，旨在复现 MetaAI 的 Self-Rewarding Language Model 论文。 - Oxen-AI/Self-Rewarding-Language-Models
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1218105907615895562)** (656 条消息🔥🔥🔥): 

- **Grok 发布**: xAI 发布了一个名为 Grok-1 的新型 3140 亿参数 MoE 模型。它因性能仅略好于 GPT-3.5 而受到批评，且被认为在没有进一步 Pretraining 的情况下体积太大，不具备实用性。
- **Grok 的商业用途受质疑**: 有人怀疑 Yi-9B 模型是否真的可以用于商业用途，以及许可流程是否仅仅是营销手段。
- **持续 Pretraining 的挑战**: 讨论集中在持续 Pretraining 模型的各种可行性和方法上，特别是像 Mixtral 这样的 MoE 模型，以及在没有特定领域数据的情况下，这是否能提高性能。
- **GPT-4 确认传闻**: NVIDIA CEO 黄仁勋在 GTC 主旨演讲中提到了一种拥有 1.8 万亿参数的架构，传闻这就是 GPT-4。该提法包含了 OpenAI 尚未正式确认的 MoE 配置。
- **推荐阅读**: 几位用户分享了关于各种 AI 主题的最新论文链接，包括来自 Apple 的 Multimodal 模型、Continual Learning 以及类似于生物神经网络的记忆。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://fxtwitter.com/lqiao/status/1768045066776707226?s=20">来自 Lin Qiao (@lqiao) 的推文</a>：我们很高兴与 @NousResearch 合作推出 Hermes 2 Pro 多轮对话和 function calling 模型。该模型在超过 1.5 万个 function calls 和 500 个样本的 function calling DPO 数据集上进行了微调，Her...</li><li><a href="https://x.com/grok/status/1769441648910479423?s=46">来自 Grok (@grok) 的推文</a>：@elonmusk @xai ░权░重░在░简░介░中░</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered">anon8231489123/ShareGPT_Vicuna_unfiltered · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://x.com/whyarethis/status/1769269824587542692?s=46">来自 Parzival - 🌞/⏫ (@whyarethis) 的推文</a>：现在我们有所进展了。</li><li><a href="https://huggingface.co/datas">datas (shu nakamura)</a>：未找到描述</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769773746896662873?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：@Cyndesama Claude 3 Opus 使用 python42 运行 AI 小镇模拟</li><li><a href="https://huggingface.co/migtissera/Tess-70B-v1.6">migtissera/Tess-70B-v1.6 · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.08763">Simple and Scalable Strategies to Continually Pre-train Large Language Models</a>：LLM 通常在数千亿 token 上进行预训练，一旦有新数据可用，往往需要重新开始整个过程。一种更有效的解决方案是持续预训练...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768948484479049897?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：`<cmd> run world_sim.exe --epoch "Earth in 2500" --civilization_type "Type-II on Kardashev scale" </cmd>` ↘️ 引用 mephisto (@karan4d) 我当然会开源 worldsim...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能多模态大语言模型 (MLLM) 的方法。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细且全面的研究...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit LLM 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1768942321129697790?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：Hermes 2 Pro function-calling 模型已与 @ExaAILabs 的搜索引擎集成👀 ↘️ 引用 Barton Rhodes 🦺 (@bmorphism) 增加了对 @ExaAILabs 的支持，以便与 @NousResearch 新的 function-calling 模型配合使用...</li><li><a href="https://huggingface.co/Replete-AI/Mistral-11b-v0.1">Replete-AI/Mistral-Evolved-11b-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.10588">Do Llamas Work in English? On the Latent Language of Multilingual Transformers</a>：我们探究了在不平衡、以英语为主的语料库上训练的多语言模型是否使用英语作为内部中间语言——这是一个对于理解语言模型如何运作至关重要的问题...</li><li><a href="https://x.com/itsandrewgao/status/1769460684956602527?s=46">来自 Andrew Kean Gao (@itsandrewgao) 的推文</a>：我觉得 grok-4bit 对于一块 H100 GPU 来说还是稍微大了一点 :( ↘️ 引用 Andrew Kean Gao (@itsandrewgao) 我的天，@grok 有 3140 亿参数，8 专家混合 (MoE)，没有经过 RLHF/道德化处理，这太...</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO/discussions/10/files">NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO · 添加评估结果</a>：未找到描述</li><li><a href="https://x.com/aravsrinivas/status/1769485603622867394?s=46&t=TOasxww3M5DjlB4iBWa_ig">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：是的，感谢 @elonmusk 和 xAI 团队开源了 Grok 的基础模型。我们将针对对话式搜索对其进行微调并优化推理，并将其提供给所有 Pro 用户！↘️ 引用...</li><li><a href="https://arxiv.org/abs/2403.08540">Language models scale reliably with over-training and on downstream tasks</a>：缩放定律 (Scaling laws) 是开发语言模型的有用指南，但目前的缩放研究与语言模型最终的训练和评估方式之间仍存在差距。例如，缩放...</li><li><a href="https://huggingface.co/datasets/openchat/openchat_sharegpt4_dataset/tree/main">openchat/openchat_sharegpt4_dataset at main</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2303.11934">Sparse Distributed Memory is a Continual Learner</a>：持续学习是人工...</li>

人工神经网络在解决其生物对应物擅长的问题方面。基于使用稀疏分布式存储（Sparse Distributed Memory, SDM）连接核心神经的工作...</li><li><a href="https://x.com/burkov/status/1769496949252673550?s=46&t=TOasxww3M5DjlB4iBWa_ig">来自 Andriy Burkov (@burkov) 的推文</a>：我们还有待观察 Grok 与 GPT-4 相比表现如何，但可以肯定的是，如果你今天要训练一个 OpenAI/Anthropic 的竞争对手，你不再需要从零开始了...</li><li><a href="https://fxtwitter.com/intrstllrninja/status/1769424961192529962?s=20">来自 interstellarninja (@intrstllrninja) 的推文</a>：&lt;cmd&gt; sudo python3 akashic_records.py --entity [&#34;sam altman&#34;, &#34;elon musk&#34;] --mode &#34;email thread&#34; --topic &#34;superintelligence scenarios&#34; &lt;/cmd&gt;</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/causality.ipynb">Abstractions/abstractions/goap/causality.ipynb at main · furlat/Abstractions</a>：一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账户来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/gridmap.ipynb">Abstractions/abstractions/goap/gridmap.ipynb at main · furlat/Abstractions</a>：一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账户来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>：一组用于抽象 IRL 的 Pydantic 模型。通过在 GitHub 上创建账户来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://docs.pydantic.dev/latest/concepts/json_schema/">JSON Schema - Pydantic</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=oYFjDt4-hFw&ab_channel=NewEconomicThinking">Cosma Shalizi - 为什么经济学需要 Data Mining</a>：Cosma Shalizi 敦促经济学家停止他们正在做的事情：将大型复杂模型拟合到一小组高度相关的序列数据中。一旦你...</li><li><a href="https://huggingface.co/01-ai/Yi-9B">01-ai/Yi-9B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/01-ai/Yi-9B-200K">01-ai/Yi-9B-200K · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=t6SQj8YidGA">加速主义加速主义 (Acc/Acc)</a>：加速主义加速主义是指当你加速加速主义，将加速主义应用于那些过于前卫的加速主义部分时：https://www.patre...</li><li><a href="https://www.hd-computing.com/">HD/VSA</a>：   </li><li><a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">NVIDIA 首席执行官 Jensen Huang 的 2024 年 3 月 GTC 主旨演讲</a>：观看 NVIDIA 首席执行官 Jensen Huang 的 GTC 主旨演讲，了解所有关于塑造我们未来的 AI 进展的公告。深入了解这些公告并发现...</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">让我们构建 GPT Tokenizer</a>：Tokenizer 是 Large Language Models (LLMs) 中一个必要且普遍存在的组件，它在字符串和 tokens（文本块）之间进行转换。Tokenizer...</li><li><a href="https://www.youtube.com/wa">Liam Johnson 击败起哄者 | 纽约脱口秀</a>：上周末 Liam Johnson 终于决定在 Giggle Nerd 首次亮相。他在周日 23:00 到 23:25 进行了表演，我们的观众非常喜欢...</li><li><a href="https://github.com/PrismarineJS/mineflayer">GitHub - PrismarineJS/mineflayer：使用强大、稳定且高级的 JavaScript API 创建 Minecraft 机器人。</a>：使用强大、稳定且高级的 JavaScript API 创建 Minecraft 机器人。- PrismarineJS/mineflayer</li><li><a href="https://github.com/Prismarin">Prismarin - 概览</a>：Prismarin 有 3 个可用的代码库。在 GitHub 上关注他们的代码。</li><li><a href="https://hack.meetmeinshibuya.com/">HacksTokyo</a>：东京的 AI x 数字娱乐黑客松！</li><li><a href="https://www.biorxiv.org/content/10.1101/2024.03.11.584515v1">使用深度强化学习进行真实果蝇运动的全身体模拟</a>：动物的身体决定了神经系统如何产生行为。因此，对感觉运动行为的神经控制进行详细建模需要一个详细的身体模型。在这里，我们共同...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1218205298729156648)** (25 条消息🔥):

- **困惑于 Perplexity**：一位成员尝试根据 [Kaggle notebook 指南](https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook)计算 **NousResearch/Llama-2-7b-chat-hf** 的 **Perplexity**，但最终得到了 90.3 这一出乎意料的 **Perplexity** 值。
- **梦想拥有 20b 模型**：人们希望看到一个能与 *Mistral* 媲美的 **20b 基础模型**。虽然对话暗示这需要大量资金，但也讨论了潜在的策略，如 **upscaling** 或与其他模型进行 **merging**。
- **缩小规模是新的扩大规模？**：一位成员分享了他们在通过 **continuous pretraining** 进行 [**downscaling models**](https://huggingface.co/AlexWortega/smallstral) 方面的经验，展示了一个经过分层剪枝的 *Mistral* 变体 **Smallstral** 在各种任务上的表现。
- **扩展模型能力**：有一个关于在 **Transformer** 模型中使用多个并行线性层进行分类的问题，旨在根据语言特征对词汇进行分组。
- **微调前沿**：讨论涉及了利用高性能计算资源进行 **Fine-Tuning** 的可能性，一位成员兴奋地预告了即将推出的 **Mixtral** 模型，该模型在 **QLoRA** 的基础上显示出令人期待的改进。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.kaggle.com/code/philculliton/calculating-the-perplexity-of-4-bit-llama-2/notebook">Calculating the Perplexity of 4-bit Llama 2</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://huggingface.co/AlexWortega/smallstral">AlexWortega/smallstral · Hugging Face</a>：未找到描述</li><li><a href="https://wandb.ai/alexwortega/cpm_rus/runs/w5t4dsat?nw=nwuseralexwortega">alexwortega</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账号为 xai-org/grok-1 做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1218181932853104720)** (18 messages🔥): 

- **链接故障排除 1对1**：一位用户询问链接是否失效，另一位用户简单地回答“没有”。
- **被某个想法震撼**：用户 **fullstack6209** 表示连续几天被一个未指明的想法所震撼，这导致另一位用户寻求关于其含义的澄清。
- **报告 Bittensor 链问题**：**jubilant_dragon_18246** 指出 **Bittensor** 链在过去 11 小时内一直存在问题，**teknium** 幽默地表示它看起来确实坏了。
- **Bittensor 链的恢复路径**：据报告 **Bittensor** 链已恢复，但需要更新 **subtensor**，而并非所有用户都完成了更新。
- **获取 TAO 的冒险**：用户 **ee.dd** 询问购买 **TAO** 以进行注册的最佳地点，并被建议使用 **MEXC** 交易所，随后在 **Kucoin** 上的提现尝试未成功。此外，关于 **GPU** 需求的讨论指出，如果设置 **QLoRA** 训练器，单个 **3090** 就足够了，否则可能需要 **80GB** 或 **48GB**（针对 g1）。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1218682432610373703)** (100 messages🔥🔥):

- **RAG 能力的演进**：成员们讨论了增强 RAG 模型的潜在功能和改进，提到了诸如从详细输出切换到结构化输出的响应模式、引用和片段高亮（span highlighting），以及理解意图和分解任务的能力。还提到了高召回率（recall）和相关性排序，但指出一些 LLM 在处理长外部上下文的推理时面临挑战。
- **RAG 模型的上下文与功能**：关于 RAG 模型应如何平衡使用提供的外部上下文与自身知识存在争论，建议通过“模式”允许模型仅关注外部来源，或在收到提示时利用内部知识进行推断。还提出了训练模型使其能够调用函数（function calling）并分解复杂提取任务的想法。
- **RAG 响应的输出格式**：大家达成共识，虽然 Markdown 可能不需要作为默认输出格式，但输出应包含列表、表格和代码等结构化元素，并保持良好的引用规范。对话中提到了 [Cohere 的模型](https://cohere.ai/docs) 的实用性，该模型在其响应中包含行内引用。
- **专用小型模型在 RAG 流水线中的潜在用途**：有人提议训练专用的、更小的模型来提高 RAG 流水线的效率，例如专门的“相关信息提取器”模型。也有人担心，由于延迟问题，大型模型在实时 RAG 操作中可能不是最优选择。
- **分享 RAG 相关资源和经验**：成员们分享了外部资源的链接，例如 [Command R 用于 RAG 的 GitHub 实现](https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py)，并简要讨论了他们的个人项目和对 RAG 生态系统的贡献。

**提到的链接**：<a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/commanDUH.py">scratchTHOUGHTS/commanDUH.py at main · EveryOneIsGross/scratchTHOUGHTS</a>：用于避免 self 溢出错误的第二大脑草稿记忆。- EveryOneIsGross/scratchTHOUGHTS

  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1218167767379742813)** (273 条消息🔥🔥): 

- **Grok-1 AI 模型讨论**：成员们正在评估 [Grok-1](https://github.com/xai-org/grok-1) 的性能和训练数据规模，并将其与 Mixtral 和 Claude 2 等其他模型进行比较。有人质疑 Twitter 聊天机器人界面是否针对实际使用进行了优化，并期待独立的基准测试（benchmarks）。

- **LLM 评估数据的建议**：社区讨论了使用 NPR 转录文本和 Wikipedia 等各种来源创建基准测试以评估 LLM 的可行性。人们对潜在的版权问题表示担忧，并希望避免法律纠纷。

- **寻求 RAG 实现资源**：一位用户询问了关于检索增强生成（RAG）的最佳教程或实现，表明需要该主题的易懂教学材料。

- **Mac 用户的 PyTorch Bug 警报**：一位成员提出了一个关于 PyTorch 的问题，该 Bug 可能会影响 Mac 上的矩阵乘法，从而导致错误的结果和性能问题，并提供了 [GitHub issue 链接](https://github.com/pytorch/pytorch/issues/122123) 作为参考。

- **会议和期刊投稿**：一位用户寻求关于提交研究论文的经济实惠选择的建议，其中 TMLR 被提及为一个免费期刊选项，同时还讨论了 ICLR 和 AISTATS 等会议投稿以备将来考虑。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok">Announcing Grok</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理是...</li><li><a href="https://x.com/maisaAI_/status/1768657114669429103?s=20">Tweet from Maisa (@maisaAI_)</a>: 介绍 Maisa KPU：AI 推理能力的下一次飞跃。Knowledge Processing Unit 是一个针对 LLM 的推理系统，它利用了它们所有的推理能力并克服了它们固有的...</li><li><a href="https://arxiv.org/abs/2203.07852">Block-Recurrent Transformers</a>: 我们介绍了 Block-Recurrent Transformer，它以循环方式在序列上应用 Transformer 层，并且相对于序列长度具有线性复杂度。我们的循环单元...</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://arxiv.org/abs/2312.12705">Optimizing Distributed Training on Frontier for Large Language Models</a>: 大语言模型 (LLM) 作为基础模型取得了显著成功，通过微调使各种下游应用受益。最近关于 Loss Scaling 的研究表明...</li><li><a href="https://en.wikipedia.org/wiki/Wikipedia:Database_reports/Most_edited_articles_last_month">Wikipedia:Database reports/Most edited articles last month - Wikipedia</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2002.09402">Addressing Some Limitations of Transformers with Feedback Memory</a>: 尽管 Transformer 是前馈网络，但已成功应用于序列、自回归任务。与循环神经网络不同，Transformer 使用 Attention 来捕捉时间关系...</li><li><a href="https://www.npr.org/sections/publiceditor/2009/08/19/112034424/free-transcripts-now-available-on-npr-org>">Free Transcripts now Available on NPR.org</a>: NPR 上喜爱、错过或令人抓狂的故事的转录文本以前每份售价 3.95 美元，但现在在 NPR.org 上是免费的。</li><li><a href="https://github.com/EleutherAI/cookbook/blob/main/calc/calc_transformer_flops.py">cookbook/calc/calc_transformer_flops.py at main · EleutherAI/cookbook</a>: 深度学习入门指南。包含处理真实模型所需的所有实际细节和实用工具。 - EleutherAI/cookbook</li><li><a href="https://aideadlin.es/?sub=ML,CG,NLP,RO,SP,DM,CV">AI Conference Deadlines</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/122123)">Issues · pytorch/pytorch</a>: Python 中具有强大 GPU 加速的 Tensor 和动态神经网络 - Issues · pytorch/pytorch</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>: 使用减半乘法次数的矩阵乘法进行 AI 加速 - trevorpogue/algebraic-nnhw</li><li><a href="https://www.youtube.com/watch?v=Sq1QZB5baNw),">Figure Status Update - OpenAI Speech-to-Speech Reasoning</a>: 未找到描述</li><li><a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>: AI 驱动的知识处理平台。一个用于执行业务任务的简单 API。为软件和应用程序开发人员抽象了使用最新 AI 架构的复杂性。</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok 开源发布。通过在 GitHub 上创建账号为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://www.cs.cmu.edu/~dwoodruf/">David P. Woodruff</a>: 未找到描述</li><li><a href="https://deepmind.google/discover/blog/rt-2-new-model-translates-vision-and-language-into-action/">RT-2: New model translates vision and language into action</a>: 介绍 Robotic Transformer 2 (RT-2)，这是一种新型的视觉-语言-动作 (VLA) 模型，它从网络和机器人数据中学习，并将这些知识转化为通用的指令...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1218100666493304852)** (245 messages🔥🔥): 

- **关于 Mamba 模型的 Speculative Sampling 辩论**: 一场讨论揭示了对 Mamba 等模型进行 *speculative decoding* 的怀疑，指出它们的运行方式不像 Transformer 那样能从 speculative sampling 中获益。尽管它们比典型的串行生成更快，但它们本质上不是并行的，且验证仍需要大量的计算，这使得 speculative sampling 可能无效。

- **Grok 模型规模与性能受到审视**：成员们就拥有世界级团队是否能规避大语言模型（LLM）的不良结果交换了意见，并讨论了 Grok 潜在的性能问题。社区强调，Grok 相对较大的规模并不一定能保证其性能优于 Mixtral 或 MiQ 等现有模型。

- **LLM 的效率与扩展**：讨论了大型语言模型（LLM）的效率和扩展策略，包括使用不同的 GPU 类型和配置。讨论强调了投机采样（speculative sampling）技术的潜在优缺点，以及扩展像 DeepScaleLM 这样深度模型的复杂性，该模型对传统的 Transformer 模型提出了改进建议。

- **辩论 Grok 与其他模型的质量**：讨论了 Grok 作为 Twitter 功能集成的可能优势，尽管它缺乏广泛的使用或可访问的 API。在独立基准测试（benchmarks）和微调（fine-tuning）对比结果出炉之前，对其模型的质量和有效性仍持怀疑态度。

- **训练规范及其对模型质量的影响**：对话涉及了训练规范的重要性，例如所使用数据的数量和类型。有人建议，像 XAi 这样的公司可能会根据内部基准测试（benchmark）的饱和度来决定停止训练，并特别关注 Twitter 上的实时应用和事件。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.ai/blog/grok">发布 Grok</a>：未找到描述</li><li><a href="https://x.com/Aaditya6284/status/1762558439354409345">来自 Aaditya Singh (@Aaditya6284) 的推文</a>：我们研究了 GPT-3.5 和 GPT-4 中这种选择的影响——具体来说，我们观察了从左到右 (L2R) 与从右到左 (R2L) 进行分词（tokenizing）的效果，这是通过使用逗号等分隔符强制执行的。我们...</li><li><a href="https://arxiv.org/abs/2401.16380">重述网络：计算与数据高效语言建模的秘诀</a>：大型语言模型（LLM）是在海量的网络抓取数据上训练的，这些数据通常是非结构化的、有噪声的且措辞不佳。目前的 Scaling Laws 表明，从这类数据中学习需要大量的...</li><li><a href="https://arxiv.org/abs/2402.18510">RNN 并非 Transformer（目前）：上下文检索的关键瓶颈</a>：本文研究了在解决算法问题背景下，循环神经网络 (RNN) 与 Transformer 在表示能力上的差距。我们专注于理解 RNN（已知...）是否...</li><li><a href="https://arxiv.org/abs/2403.09539">受 API 保护的 LLM Logits 会泄露专有信息</a>：大型语言模型 (LLM) 的商业化导致了仅通过高级 API 访问专有模型的普遍做法。在这项工作中，我们展示了即使在保守的假设下...</li><li><a href="https://pytorch.org/blog/accelerating-generative-ai-2/">使用 PyTorch 加速生成式 AI II：GPT，快速</a>：本篇文章是专注于如何使用纯原生 PyTorch 加速生成式 AI 模型的系列博客的第二部分。我们很高兴能分享广泛的新发布的 PyTorch 性能...</li><li><a href="https://arxiv.org/abs/2403.04706">普通 7B 语言模型已具备强大的数学能力</a>：此前人们认为，数学能力只有在极大规模的普通语言模型中才会出现，或者需要大量的数学相关预训练。本文展示了 LLaMA-2 7B 模型...</li><li><a href="https://arxiv.org/abs/2403.09635">Transformer 变得稳定：语言模型的端到端信号传播理论</a>：尽管取得了巨大成功，Transformer 模型在深度扩展方面仍然困难。在这项工作中，我们开发了一个统一的信号传播理论，并提供了控制...矩的公式。</li><li><a href="https://arxiv.org/abs/2403.09611">MM1：多模态 LLM 预训练的方法、分析与见解</a>：在这项工作中，我们讨论了构建高性能的多模态大型语言模型 (MLLM)。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细和...</li><li><a href="https://arxiv.org/abs/2403.06963">Next-token prediction 的陷阱</a>：仅仅一个 Next-token 预测器能否忠实地模拟人类智能？我们将这种在文献中零散分布的直觉担忧具体化。作为起点，我们认为这两个经常被混淆的...</li><li><a href="https://arxiv.org/abs/2403.09394">GiT：通过通用语言接口迈向通用视觉 Transformer</a>：本文提出了一个简单而有效的框架，称为 GiT，仅使用原生 ViT 即可同时适用于各种视觉任务。受多层 Transformer 通用性的启发...</li><li><a href="https://arxiv.org/abs/2403.06504">添加 NVMe SSD 以在单张 GPU 上实现并加速 100B 模型微调</a>：大型语言模型的最新进展为世界带来了巨大价值，其卓越的能力源于它们使用的海量参数。然而，即使是拥有...的 GPU</li><li><a href="https://arxiv.org/abs/2402.00691">Frontier 上大型语言模型架构的对比研究</a>：大型语言模型 (LLM) 在 AI 社区及其他领域引起了广泛关注。其中，Generative Pre-trained Transformer (GPT) 已成为主流架构...</li><li><a href="https://arxiv.org/abs/2403.10430">算术 Teichmuller 空间的构造 IV：abc 猜想的证明</a>：这是我在本系列论文中开发的算术 Teichmuller 空间工作的延续。在本文中，我展示了算术 Teichmuller 空间理论如何通过使用 Shinic...</li><li><a href="https://x.ai/blog/grok-os">Grok-1 开源发布</a>：未找到描述</li><li><a href="https://github.com/xai-org/grok">GitHub - xai-org/grok-1: Grok 开源发布</a>：Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/enfiskutensykkel/ssd-gpu-dma">GitHub - enfiskutensykkel/ssd-gpu-dma: 构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用</a>：构建支持 CUDA 的用户空间 NVMe 驱动程序和存储应用 - enfiskutensykkel/ssd-gpu-dma</li><li><a href="https://github.com/bigscience-workshop/bl

oom-dechonk">GitHub - bigscience-workshop/bloom-dechonk: 一个用于运行模型收缩实验的仓库</a>: 一个用于运行模型收缩实验的仓库。通过在 GitHub 上创建账号来为 bigscience-workshop/bloom-dechonk 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: 我们提出了一种方法，用于估算大型语料库中可能被大语言模型 (LLM) 大幅修改或生成的文本比例。我们的极大似然模型 leve...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>: 本研究探讨了在科学同行评审中使用大语言模型 (LLM)（如 ChatGPT）的情况。作者开发了一种方法来估算同行评审中生成的文本百分比...</li><li><a href="https://artificialanalysis.ai/">Model &amp; API Providers Analysis | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较与分析。涵盖质量、价格、性能和速度（吞吐量和延迟）等关键指标的独立基准测试。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1218832533517766666)** (11 messages🔥): 

- **数据复杂度影响 Scaling Laws**: 强调了语言模型 Scaling Laws 对数据复杂度的敏感性，概率上下文无关文法 (PCFG) 的句法属性和 gzip 压缩是预测特定数据集 Scaling 属性的有效指标。
- **等待全面的实验**: 此外，目前正在进行更全面的实验，以拟合 Scaling Laws 并提供确切数据，并期待使用特定用户的软件包来协助分析。
- **复杂度与下游任务**: 讨论了模型困惑度 (Perplexity) 与数据复杂度之间的关系，以及对下游任务的潜在影响，探讨了如何将这种复杂度与任务特定性对齐，并利用其进行数据清洗和高效预训练。
- **句法规范作为数据集标签**: 在回答有关数据集标签的询问时，解释说额外的标签代表了从用于生成数据集的 PCFG 中导出的句法规范，包括非终结符和终结符的数量等指标。
- **困惑度测量与信息密度**: 澄清了困惑度和 Loss 实际上是相同的，重点是使用 gzip 等压缩测量方法来寻找高效预训练的最佳词汇密度范围。
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1218288738728284241)** (13 messages🔥): 

- **关于从指定分布中采样字符串的查询**: 一位成员询问是否有规范的方法从词汇表上预先指定的 1-gram, 2-gram, ..., n-gram 统计数据集中采样字符串。
  
- **Gram 统计中的约束层级**: 澄清了指定 n-gram 统计数据也决定了所有低阶 gram 的统计数据，尽管需要对句首 (BOS) 和句尾 (EOS) Token 进行一些细微考虑。

- **自回归采样说明**: 自回归采样是从符合指定 n-gram 统计数据的分布中提取样本的方法。该方法从 unigram 分布开始，然后进行条件 bigram 分布等，从而创建与这些指定统计数据相对应的最大熵分布。

- **N-gram 语言模型背景**: 讨论引用了关于词 n-gram 语言模型的 [Wikipedia 条目](https://en.wikipedia.org/wiki/Word_n-gram_language_model)，强调了它们的历史背景以及被循环神经网络和大语言模型等更先进模型取代的过程。

- **从 Bigram 分布采样的实际实现**: 分享了一个用于生成 bigram 的 GitHub Python 脚本作为示例，该脚本是 EleutherAI 分析神经网络训练期间特征演化项目的一部分。脚本位于 [features-across-time/scripts/generate_bigrams.py](https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py)。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://github.com/EleutherAI/features-across-time/blob/main/scripts/generate_bigrams.py">features-across-time/scripts/generate_bigrams.py at main · EleutherAI/features-across-time</a>: 理解神经网络学习到的特征在整个训练过程中是如何演变的 - EleutherAI/features-across-time</li><li><a href="https://en.wikipedia.org/wiki/Word_n-gram_language_model">Word n-gram language model - Wikipedia</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1218143473916575765)** (31 messages🔥): 

- **LLM 与 lm-eval-harness 的集成**：一位用户询问如何为 LLM 模型实现 `generate_until` 和 `log_likelihood` 等函数，特别是针对 Gaudi2 上的 *megatron deepspeed* 版 Llama。提到了 `models` 目录中的参考实现，并表示需要 Demo 以及对继承和参数结构的澄清。然而，未提供具体的解决方案或 Demo 代码。

- **模型错误地默认为 GPT-2-Small**：提出了在 lm-eval-harness 中指定模型但其默认为 `gpt-2-small` 而非指定模型（如 *Mixtral*）的问题。用户发现原因是其命令中指定了两次 `model_args`，导致第一个实例被忽略。

- **报告的 MMLU 分数不一致**：讨论了 OpenLLM 排行榜上报告的 *llama2-70b* MMLU 分数（69%）与用户获得的分数（62-64%）之间的差异。澄清指出，排行榜的平均方法不同，没有根据子任务大小进行加权。

- **lm-evaluation-harness 中潜在的死锁问题**：分享了一个关于 `wmt14-en-fr` 评估死锁的 GitHub Issue ([#1485](https://github.com/EleutherAI/lm-evaluation-harness/issues/1485))。建议包括避免在同一文件系统上运行并发进程，并查看与 multiprocessing 相关的代码以寻找可能的解决方案。

- **LM Harness 模型缓存目录**：关于 `lm-eval` 下载模型位置的问题得到了澄清：模型通常存储在 Hugging Face 缓存目录中，可以通过环境变量（如 `HF_HOME`、`TRANSFORMERS_CACHE` 和 `HF_DATASETS_CACHE`）进行配置。

- **lm-eval-harness 新版本发布**：lm-eval 的新版本 0.4.2 已发布，可在 [PyPI](https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2) 上获取。公告邀请更多贡献者，并承诺对待处理的 Pull Request 进行审查。

- **LM Evaluation Harness 中的翻译**：讨论了在 lm-eval-harness 中包含机器翻译评估（如 *arc_challenge* 或 MMLU）的话题。一种潜在的方法是将此类任务组织在特定目录下，并在名称中注明其翻译性质。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/transformers/perplexity">Perplexity of fixed-length models</a>: 未找到描述</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理您的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/model_guide.md">lm-evaluation-harness/docs/model_guide.md at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1485">`wmt14-en-fr` deadlock issue · Issue #1485 · EleutherAI/lm-evaluation-harness</a>: 在运行此任务的评估时，在进行 ter 指标计算期间，程序会永久卡住。命令为：lm_eval --model hf --model_args pretrained=microsoft/phi-2,trust_remote_code=True ...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2">Release v0.4.2 · EleutherAI/lm-evaluation-harness</a>: lm-eval v0.4.2 发行说明。我们正在为 PyPI 用户发布 lm-eval 的新次要版本！我们很高兴看到 lm-evaluation-harness 的持续使用，包括作为标准测试...</li><li><a href="https://github.com/huggingface/evaluate/blob/8dfe05784099fb9af55b8e77793205a3b7c86465/metrics/perplexity/perplexity.py">evaluate/metrics/perplexity/perplexity.py at 8dfe05784099fb9af55b8e77793205a3b7c86465 · huggingface/evaluate</a>: 🤗 Evaluate：一个用于轻松评估机器学习模型和数据集的库。 - huggingface/evaluate
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1219336845310038047)** (3 messages):

- **关于 The Pile 数据打乱的澄清**：一位成员询问用于预训练的 The Pile 数据是否经过预打乱，随后的澄清解释说原始文件没有打乱，而 Hugging Face 上预处理和预分词（pretokenized）的数据是即插即用的。他们指出这与 Pythia 使用的数据相同。
- **Pile 各部分未打乱，但 Train/Test/Val 可能已打乱**：另一位成员补充说，The Pile 的各个组件没有打乱，部分原因是有些是按日期组织的，但预期原始的训练/测试/验证集划分（train/test/validation split）应该是打乱的，以确保各种数据集之间的良好混合。
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1218173412522852483)** (193 messages🔥🔥): 

- **深入探讨 ChatGPT 的理解能力**：一场讨论思考了 AI 是否真正“理解”语言，考虑到复杂的下一个词预测（next-word predictions）产生的涌现行为，以及人类训练对 AI 性能的影响。他们辩论了 AI “意识”的本质，将物理体验与抽象体验进行对比，认为真正的人类训练创造出的模型能够进行优于某些人类的对话交互。

- **卓越的图像生成**：用户对 **DALL-E 3** 准确遵循详细提示词的能力表示赞叹，称其为“太棒了”，并赞赏其相对于前代产品的进步。他们对比了使用 Microsoft Copilot 的体验，讨论了不同图像生成工具的优缺点，涉及速度和图像保存等问题，一些人因为 **ChatGPT+** 具备底层的 **DALL-E 3** 和 **GPT-4** 能力而更倾向于使用它。

- **辩论 AI 模型**：一场关于 **GPT-4** 与 **Claude** 的对比讨论展开，用户分享了他们在各种任务中使用这两个模型的经验。他们讨论了 **Claude** 作为对话工具的优势，同时指出两个模型各有优缺点，涉及成本效率、政治正确性以及提供信息的冗长程度等方面。

- **学习 AI 和 PyTorch**：用户交流了深入研究 AI 和 **PyTorch** 所需的数学基础建议，建议将预备微积分和线性代数作为起点。推荐了 YouTube 上的 **3blue1brown** 等资源进行直观学习，并鼓励用户进行持续的学习和探索。

- **AI 支持渠道**：交流了如何联系 OpenAI 支持团队的信息。讨论重点包括在支持网站上操作 **OpenAI 的帮助机器人**，引导用户报告 Bug 或提交工单寻求帮助，同时提到 **platform.openai.com** 用于 Bug 报告，并参考了 Discord 中的 **<#1070006915414900886>** 以获取额外帮助。

**提及链接**：<a href="https://openai.com/enterprise-privacy">Enterprise privacy</a>：未找到描述

  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1218428016573812888)** (34 messages🔥): 

- **对 GPT-5 的好奇**：用户询问 **GPT-5** 发布日期的简短交流，但未提供具体信息或日期。
- **GPT-3.5 的集成挑战**：一位用户在让 **GPT Turbo 3.5** 准确生成代码时遇到困难，特别是关于定位网页元素的方法，并怀疑是否是因为 **Playwright** 库过时。
- **排查 GPT 响应问题**：成员报告了 GPT 不响应提示词的问题，其他人建议这可能是需要支持协助的错误。
- **讨论 ChatGPT 行为的突然变化**：用户对过去几天 ChatGPT 行为的变化表示担忧，随后发现问题的用户确认这是与 **WebChatGPT Chrome 扩展** 的冲突。
- **对过滤器敏感度的沮丧**：多位用户对内容过滤器在创意写作中过于敏感表示沮丧，指出即使是像“亲吻嘴唇”这样温和的行为也可能触发 GPT 的警告或拒绝。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **探索分类任务的提示词架构**：一位成员讨论了优化分类任务（**classification tasks**）的提示词结构，旨在获得更高的召回率（recall）和更少的误报（false positives）。他们正在实验提供的上下文数量，并考虑使用自定义 **GPT 模型**。

- **Turbo 在 Playwright 测试中的问题**：在尝试使用 **GPT-3.5 Turbo** 生成 **Playwright** 测试代码时，它生成的代码无法使用。一位成员建议该模型可能没有更新到最新的 **Playwright** 库，而 **GPT-4** 可能会产生更好的结果。

- **处理输出中的拒绝**：一位成员遇到了模型频繁**“拒绝执行任务”**的问题，这引发了关于如何处理或避免此类拒绝的讨论。成员们建议使用 meta-prompting 策略，并将任务拆分为块（chunks），以防止模型触发拒绝条件。

- **行为转变与内容政策**：对话还涉及到一个观察结果，即以前有效的提示词现在会产生**“抱歉，我无法做到”**的消息，这暗示了模型行为随时间的变化或更激进的偏差最小化策略。讨论中提到了在不触及内容政策违规领域的情况下，克服这些障碍所面临的挑战。

- **网页搜索的查询策略**：一位成员询问如何让 AI 使用**多个查询进行网页搜索**，以获取更全面的信息。尽管存在困惑，但会议明确了应该向模型提供关于检查哪些来源以及寻找哪些信息的指导。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1218207072064114718)** (79 messages🔥🔥): 

- **明确分类任务的 Context Window**：一位成员询问了在分类用例的提示词中包含多少上下文最为理想。他们正试图通过详细的提示词架构（考虑到包含输入特征的 **dataframe**）来实现更高的召回率并减少假阳性。另一位成员建议参考“大海捞针”（needle in a haystack）测试结果，并建议使用不超过总 Context Window 的 1/2，以获得最佳的依从性和生成效果。

- **提示词回放**：成员们讨论了 AI 偶尔会出现拒绝任务的倾向，这种倾向在单次对话中似乎会增加频率。有人提出将 **meta-prompting** 作为解决方案，认为它允许 AI 进行自我调节，从而在不违反内容政策的情况下避免拒绝。

- **探索模型响应与性能**：聊天参与者交换了关于 GPT 模型如何响应任务的观察，包括对于以前有效的提示词，拒绝消息有所增加。一位成员强调了“表面算法偏差最小化（Superficial algorithmic bias minimization）”的实施，并提出了一种将 **GPT 响应分类**为各种类型的方法，以辨别提示词是否被理解。

- **网页搜索的困扰与解决方法**：一位用户询问如何指示 GPT 使用多个查询进行网页搜索，以获得更全面的结果集，而不是单一查询。随后的讨论探索了诸如 **prompt engineering** 之类的技术来引导 AI 输出期望的结果，但对该过程的澄清仍然是必要的。

- **分享解决方案并寻求支持**：成员们分享了他们对 GPT 的创意用法，包括创建一个专注于支持的 AI，并征求社区的反馈。还讨论了模型被感知的拒绝行为可能如何影响用户体验和对 AI 交互的期望。

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1218106794698739782)** (96 messages🔥🔥): 

- **Aya Demo 请求添加滑块**：Aya demo 收到了社区贡献，实现了一个较高的*重复惩罚（repetition penalty）*。已请求贡献者在 Gradio 界面中添加**滑块功能**。[点击此处提交 PR 进行贡献](https://huggingface.co/spaces/Tonic/Aya/discussions/3)。

- **NVIDIA H100 和基于 ARM 的服务器 CPU 引发热议**：一个在同一块板卡上结合了**巨大 GPU** 和**服务器 CPU**、传闻功耗约为 **850W** 的设备成为了关注焦点。功耗数据出现了差异，从预期的 **GPU 300-350W** 到声称 H100 功耗**高达 700W** 不等。[基准测试链接](https://www.phoronix.com/review/nvidia-gh200-gptshop-ben)。

- **HuggingFace 上的数据囤积**：一位成员展示了一个**数据排行榜**，展示了 HuggingFace 上托管的海量数据，包括超过 **120B 的模型**。[排行榜链接](https://huggingface.co/spaces/Weyaxi/data-leaderboard)。

- **关于使用大型 LLM 的讨论**：成员们分享了关于使用**大型语言模型** (LLM) 和高性能计算的挑战与思考。话题涵盖了从**单个 token 耗时数十秒**的*缓慢生成速度*，到通过*量化（quantization）提高速度*的潜力，以及管理像 xAI 的 **Grok-1**（拥有 3140 亿参数）这类模型的复杂性。

- **社区对 Grok 发布的热情参与**：拥有 **3140 亿参数**的 **Grok-1 模型**在 Apache 2.0 许可证下发布，引发了广泛讨论。分享了入门 Grok 的相关链接，同时也引发了关于在 HuggingFace 等平台上上传如此庞大数据集的讨论。[阅读更多关于 Grok 的信息](https://x.ai/blog/grok-os) 或在 [HuggingFace](https://huggingface.co/alpindale/grok-1) 上查找 Grok 模型。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.phoronix.com/review/nvidia-gh200-gptshop-ben">来自 Linux Performance, Benchmarks &amp; Open-Source News - Phoronix 的推文</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/ivrit-ai/whisper-large-v3-space">Whisper Large V3 - 由 ivrit-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.ai/blog/grok-os">Grok-1 的开源发布</a>：未找到描述</li><li><a href="https://academictorrents.com/details/5f96d43576e3d386c9ba65b883210a393b68210e">grok-1</a>：Grok-1 是一个拥有 314B 参数的 Mixture of Experts 模型 - 基础模型（未微调） - 8 个专家（2 个激活） - 86B 激活参数 - Apache 2.0 许可证 - 代码： - 祝编码愉快！附：我们正在招聘：</li><li><a href="https://huggingface.co/spaces/Tonic/Aya/discussions/3">Tonic/Aya · 将 repetition_penalty 常数设置为 1.8</a>：未找到描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1768779404442739147">来自 Weyaxi (@Weyaxi) 的推文</a>：🤔你是否曾好奇我们在 @huggingface 上托管了多少数据？在看到 @TheBlokeAI 的模型数量以及平台上闲置的 120B 模型后，我产生了好奇 😅 📊 所以我抓取了所有仓库...</li><li><a href="https://github.com/gradio-app/gradio/issues/7722">Video-LLaVA 演示 API 无法与 Gradio-Client 配合使用 · Issue #7722 · gradio-app/gradio</a>：描述 Bug。我正尝试在 Hugging Face Spaces 上为 Video-LLaVA 模型演示使用 Python API，但我遇到了一个错误：Traceback (most recent call last): File "/Users/kamakshiramamurthy/Deskt...</li><li><a href="https://github.com/moritztng/fltr">GitHub - moritztng/fltr: 类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。</a>：类似于 grep，但针对自然语言问题。基于 Mistral 7B 或 Mixtral 8x7B。 - moritztng/fltr
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1218115205553324112)** (12 条消息🔥): 

- **贝叶斯优化（Bayesian Optimization）令人困惑**：一位成员表达了对 **Bayesian optimization** 与 GridSearch 和 RandomSearch 优化技术对比时的困惑。
  
- **寻求 Hugging Face 指导**：一位成员请求帮助理解如何使用 **Hugging Face** 及其服务，例如用于自然语言处理任务的 Transformers 库。

- **Duet AI Cover 的困扰**：一个咨询集中在制作**双人合唱和乐队的 AI Cover** 上，得到的回复建议分别录制并叠加个人声音以提高质量。

- **使用 SageMaker 和 Hugging Face 的端到端 MLOps**：一位成员分享了一个关于使用 Amazon SageMaker 和 Hugging Face 创建 **MLOps 流水线**的**工作坊笔记本**链接，其中包含详细步骤和先决条件 ([工作坊笔记本](https://www.philschmid.de/mlops-sagemaker-huggingface-transformers))。
  
- **图像处理愿景**：一位成员讨论了计划将基础图像处理工具（如**对比度和亮度调节**）集成到他们的项目 **Fooocus** 中，以避免使用 Photoshop。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co).">未找到标题</a>：未找到描述</li><li><a href="https://www.philschmid.de/mlops-sagemaker-huggingface-transformers">MLOps: 使用 Hub 和 SageMaker Pipelines 的端到端 Hugging Face Transformers</a>：了解如何构建从训练到生产的端到端 Hugging Face Transformers MLOps 流水线。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1218346001421570138)** (12 条消息🔥): 

- **语言二元性的突破**：成员们讨论了机器学习模型处理中文和英文等语言差异巨大的能力。一位成员对这种能力表示惊讶，特别是考虑到每种语言特有的语言结构和思维模式的深刻差异。

- **探索多语言模型的思维过程**：继关于跨中英文工作的语言模型讨论之后，讨论指出任务的简单性可能会掩盖特定语言知识中的细微差别。有人提到，虽然论文中展示的基础任务可以完成，但创作一部中文小说的复杂性可能会凸显这些内在的语言差异。

- **备受关注的 Medusa**：分享了一个关于 **Medusa** 论文的链接，这是一种包含并行处理的高效 Language Model 推理方法。这引发了人们的好奇：当预测不针对特定语言时，此类模型将如何有效地蒸馏信息。

- **评估英文在多语言模型中的影响**：有人担心以英文为主的训练语料库可能会在无意中使模型偏向欧洲语言和思维模式。这一持续的对话反映了社区对语言模型受英文等主导语言影响这一开放性问题的关注。

- **聊天机器人如何改变同行评审**：一篇研究 Large Language Models (LLMs) 对科学同行评审影响的论文受到关注，研究结果表明，AI 会议评审中很大比例的文本可能已被 LLMs 修改。对话似乎集中在 LLM 修改在学术同行评审语境下的行为洞察及其影响。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.10774">Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads</a>：Large Language Models (LLMs) 的推理过程通常由于自回归解码过程缺乏并行性而受到限制，导致大多数操作受限于内存带宽...</li><li><a href="https://huggingface.co/papers/2403.09611">Paper page - MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.07183">Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：我们提出了一种方法，用于估算大型语料库中可能被 Large Language Model (LLM) 大幅修改或生成的文本比例。我们的最大似然模型利用...</li><li><a href="https://bytez.com/read/arxiv/2403.07183">Bytez: Monitoring AI-Modified Content at Scale: A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews</a>：本研究探讨了 Large Language Models (LLMs)（如 ChatGPT）在科学同行评审中的应用。作者开发了一种方法来估算同行评审中由 AI 生成的文本百分比...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1218158991570636900)** (18 条消息🔥): 

- **NL2SQL 寻求者寻求帮助**：一位参与者正在使用 BAAI/llm-embedder、TheBloke/nsql-llama-2-7B-GGUF 和 FAISS 向量存储构建 **NL2SQL 流水线**，寻求关于提高选择相关 SQL 表和生成查询准确性的建议。

- **NVIDIA 最新性能怪兽亮相**：一位成员介绍了 **NVIDIA Grace Hopper Superchip**，强调了其在 HPC、AI 和数据中心应用中的强大实力。

- **NLP 之旅开启**：NLP 初学者被引导至 [HuggingFace Course](https://huggingface.co/learn/nlp-course/chapter1/1) 的 Hugging Face NLP 课程，以及托管在 [Stanford's SLP3 manuscript](https://web.stanford.edu/~jurafsky/slp3/) 的综合教科书。

- **NLP 学习资源汇编**：除上述资源外，参与者还提到了 **Stanford's CS224n course notes**，作为斯坦福手稿的简洁版本，以辅助 NLP 教育。

- **探索用于生产环境的免费 LLM API**：一位用户询问用于生产部署的免费 LLM API，另一位用户建议将 "ollama" 作为本地实现的免费选项。

**提到的链接**：<a href="https://huggingface.co/learn/nlp-course/chapter1/1">Introduction - Hugging Face NLP Course</a>：未找到描述

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1218217429868478474)** (7 条消息):

- **在 RAG 中将文档作为工具进行交互**：提出了一种处理 [RAG pipeline](https://t.co/eCdLmlXZFj) 中复杂查询的创新方法，将每个检索到的文档视为一个交互式工具，从而实现更高级的交互。
- **发布带有 Instrumentation 的 LlamaIndex v0.10.20**：宣布了包含 Instrumentation 模块的新版本 LlamaIndex，并提供了演示 [基础可观测性](https://t.co/GY4unUYOwl) 和 [API 调用观测](https://t.co/E1d9dtkqAI) 的 notebook。
- **通过 Search-in-the-Chain 增强 QA**：讨论了 Shicheng Xu 等人的一篇论文，该论文介绍了一种将检索与规划交织在一起的方法，通过 [验证步骤并相应调整计划](https://t.co/7gLlDyd1cV) 的过程来优化问答。
- **关于基于 RAG 的求职助手的博客文章**：推荐了 [Kyosuke Morita 的博客文章](https://t.co/1Y9TPgGHW1)，介绍了如何结合使用 LlamaParse 和 LlamaIndex 解析 CV，从而创建将候选人与职位匹配的求职助手。
- **MemGPT 网络研讨会发布**：分享了 [Charles Packer 主讲的网络研讨会](https://t.co/bUpqCvLweS)，介绍了 MemGPT 架构，该架构为 Agent 提供了与“核心”内存交互的内存工具，增强了其 function-calling 能力。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1218113300764819488)** (303 条消息🔥🔥): 

- **链接 OpenAI Agent 的难题**：讨论围绕使用 LlamaIndex 文档中描述的工具链接多个 OpenAI Agent 的可能性展开。一名成员尝试使用 LlamaIndex 的 `FunctionTool` 和 `QueryEngineTool`，但遇到了提示消息内容为空或格式错误的错误。
  
- **Xinference CPU 集群查询**：成员们讨论了在 CPU 集群中使用 Xinference 是否可以缩短推理时间。虽然知识库缺乏具体的性能细节，但通常使用 CPU 集群进行推理可以分担工作负载并可能加快处理速度。
  
- **调整本地 LLM 的 Token 限制**：一位用户在更改本地 LLM 的最大 Token 大小时需要帮助。建议使用 `Ollama(... additional_kwargs={"num_predict": number_of_tokens})` 并将 `context_window` 传递给构造函数作为潜在解决方案。
  
- **LlamaIndex 中的过滤**：一位成员询问是否可以在 SimpleFusionRetriever 和 Retriever Query Engine 流程中的检索之前进行元数据过滤。有人提示像 Qdrant 这样的向量数据库可以将过滤器附加到子检索器（sub-retrievers），以实现预检索过滤。
  
- **Langfuse 集成中的 Span 问题**：一位将 Langfuse 与 LlamaIndex 集成的用户注意到某些步骤缺失 Span，例如对用户问题进行 Embedding 和在 Qdrant 中查找文档。建议他们确保将 callback manager 传递到所有组件中（包括 Embedding 模型），以查看预期的 Span。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cloud.llamaindex.ai">LlamaCloud</a>: 未找到描述</li><li><a href="https://www.promptingguide.ai/techniques/fewshot">Prompt Engineering Guide</a>: Prompt Engineering 的全面概述</li><li><a href="https://docs.llamaindex.ai/en/stable/api/llama_index.core.node_parser.CodeSplitter.html">CodeSplitter - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://qdrant.tech/documentation/tutorials/llama-index-multitenancy/">Multitenancy with LlamaIndex - Qdrant</a>: Qdrant 是一个用 Rust 编写的开源向量数据库和向量搜索引擎。它提供快速且可扩展的向量相似度搜索服务，并配备便捷的 API。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">Defining and Customizing Documents - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/image_to_image_retrieval.html">Image to Image Retrieval using CLIP embedding and image correlation reasoning using GPT4V - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/use_cases/extraction.html">Structured Data Extraction - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="https://www.promptingguide.ai/techniques/rag">Prompt Engineering Guide</a>: Prompt Engineering 的全面概述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/tools/root.html">Tools - LlamaIndex 🦙 v0.10.20.post1</a>: 未找到描述</li><li><a href="http://localhost:{port}",>">no title found</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/5c53f41712785e5558156372bdc4f33a6326fa5f/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb">llama_index/docs/examples/vector_stores/Qdrant_using_qdrant_filters.ipynb at 5c53f41712785e5558156372bdc4f33a6326fa5f · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/12034">[Question]: custom llm but is blocked · Issue #12034 · run-llama/llama_index</a>: 问题验证。我已经在文档和 Discord 中搜索了答案。问题代码来自 typing import Optional, List, Mapping, Any from llama_index.core import SimpleDirecto...</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py">llama_index/llama-index-integrations/llms/llama-index-llms-ollama/llama_index/llms/ollama/base.py at main · run-llama/llama_index</a>: LlamaIndex 是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/hofstadter-io/hof/blob/_dev/flow/chat/prompts/dm.cue">hof/flow/chat/prompts/dm.cue at _dev · hofstadter-io/hof</a>: 连接数据模型、Schema、代码生成和任务引擎的框架。与语言和技术无关。 - hofstadter-io/hof</li><li><a href="http://127.0.0.1:9997>">no title found</a>: 未找到描述</li><li><a href="http://localhost:{port}">)">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1218542835754860564)** (4 条消息): 

- **使用 LlamaParse 等工具的 RAG 教程**：分享了一个关于使用 LlamaParse、Qdrant 和 Groq 创建高效 RAG 的分步视频，解释了该过程并展示了 **LlamaParse** 的功能。在 [YouTube](https://youtu.be/w7Ap6gZFXl0) 上观看详细指南。

- **寻求 RAG 准备技巧**：一位成员正在寻求关于准备 **RAG** 文档的顶级技巧，以及自动向 **Pinecone** 添加元数据以实现最佳文档检索的方法。

- **关于使用 RAG 的 AI 助手的 Medium 文章**：推荐了一篇讨论通过具有 RAG 流水线、记忆和 LlamaIndex 的 **AI 助手赋能声音**的文章。深入分析请见 [Medium](https://medium.com/ai-advances/empowering-voices-ai-assistant-with-rag-pipeline-memory-and-llamaindex-11c4e319d915)。

- **在 RAG 实现中切换到 Huggingface 模型**：一位成员在 RAG 的 **RAPTOR** pack 中将 OpenAI 模型替换为 Huggingface 模型时遇到困难，并提到了过程中的多个错误。他们正在寻求根据官方 [GitHub 仓库](https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/examples/raptor.ipynb) 的示例来纠正其实现方式的建议。

**提到的链接**：<a href="https://youtu.be/w7Ap6gZFXl0">RAG with LlamaParse, Qdrant and Groq | Step By Step</a>：在这段视频中，我将向你展示如何使用 LlamaParse, Qdrant 和 Groq 创建一个高效的 RAG。我将解释什么是 LlamaParse 并简要引导你...

---

**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1218154073912639508)** (202 条消息🔥🔥): 

- **理解 Yann 对 LLM 的立场**：一系列讨论重点关注了 @Teknium1 关于 Yann LeCun 对大语言模型 (LLM) 持悲观观点的推文。讨论提到，基于某些人天生缺乏“内心独白”的假设（这可能影响了他们对非语言思维过程的偏好），Yann 可能更倾向于具有视觉推理或规划能力的模型，而非纯语言模型。分享了一段对同样缺乏内心独白的人的采访。成员们对认知推理中“shape rotators”与“wordcels”之间的二分法提出了质疑。
- **OpenAI 的 GTC 虚拟会议优惠**：成员们讨论了 OpenAI 参加 GTC (GPU Technology Conference) 的情况，分享了虚拟会议的免费访问代码，并暗示可能为帮助注册的 Influencer 提供硬件交换计划。提供了注册链接以及会议详情的访问权限。
- **发布 Grok-1：影响尚不明确的巨型模型**：xAI 宣布开源发布 Grok-1，这是一个拥有 3140 亿参数的 Mixture-of-Experts 模型，希望社区能在持续训练和评估方面做出贡献。社区反应不一，一些人对其质量（与 LLaMa 和 Claude 等其他模型相比）表示担忧，同时也对其模型的规模表示赞赏。讨论围绕持续 Pretraining 和 Quantization 以改进或利用该模型的潜力展开。
- **SWYX 对 Lex 播客错失机会的看法**：Lex Fridman 采访 Sam Altman 的播客因未能深入探讨实质性问题、对 OpenAI 内部运作和政治轻描淡写而受到批评。听众认为对话缺乏深度，更多关注于边缘话题，而在提供 AI 和模型进展方面的见解较少。
- **Jensen Huang 的 Nvidia Keynote 预期**：大家对 Nvidia CEO Jensen Huang 的 GTC Keynote 充满期待，推测可能会揭晓 AI 进步的重要参数。虽然没有直接引用证实，但社区似乎接受了演讲中提到的 GPT-4 拥有 1.8 万亿参数的说法。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.ai/blog/grok-os">Grok-1 开放发布</a>：未找到描述</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：这解释了为什么 Yann 对 LLM 如此看空... 😲</li><li><a href="https://x.com/repligate/status/1769241542420738126?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 j⧉nus (@repligate) 的推文</a>：这是在 Claude 的后台导航到 ../../microsoft/bing/bing_chat 目录，然后让 Claude 使用命令自行查看，接着运行：&lt;cmd_soul&gt;... 的结果。</li><li><a href="https://arxiv.org/abs/2402.10171">将语言模型扩展至 128K 上下文的数据工程</a>：我们研究了将语言模型上下文长度扩展到 128K 的持续预训练方案，重点关注数据工程。我们假设长上下文建模，特别是 \textit{t...</li><li><a href="https://huggingface.co/blog/TimothyAlexisVass/explaining-the-sdxl-latent-space">解释 SDXL 潜空间</a>：未找到描述</li><li><a href="https://x.com/francis_yao_/status/1769575936994013611?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Yao Fu (@Francis_YAO_) 的推文</a>：Grok 的 MMLU 仅与 Mixtral 持平，尽管其规模大了一个数量级。我相信它有巨大的潜力但尚未完全释放，良好的持续预训练数据可能会大幅提升...</li><li><a href="https://huggingface.co/collections/suno/bark-6502bdd89a612aa33a111bae">Bark - 一个 suno 集合</a>：未找到描述</li><li><a href="https://x.com/Francis_YAO_/status/1759986097365627054?s=20">来自 Yao Fu (@Francis_YAO_) 的推文</a>：前沿模型都至少有 100k 的上下文长度，Gemini 1.5 甚至有 1m 上下文。那么研究和开源界呢？介绍长上下文数据工程，一种实现...的数据驱动方法。</li><li><a href="https://x.com/teortaxestex/status/1769460562763604375?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：@aidan_mclau 0) 火箭人很糟 1) 它并没有差多少 2) 如你所见，这是一个稀疏上采样的 Grok-0。它还没训练好。在 2023 年，持续预训练已基本解决，并且有验证...</li><li><a href="https://substack.recursal.ai/p/eaglex-17t-soaring-past-llama-7b">🦅 EagleX 1.7T：在英语和多语言评估中超越 LLaMA 7B 2T (RWKV-v5)</a>：线性 Transformer 刚刚超越了 Transformer 模型的金标准 LLaMA 7B，且在英语和多语言评估中使用的训练 Token 更少。历史性的第一次。</li><li><a href="https://x.com/openinterpreter/status/1769448726660337875?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Open Interpreter (@OpenInterpreter) 的推文</a>：百年磨一剑，最后 100 小时。</li><li><a href="https://x.com/teknium1/status/1768452864995942469?s=46&t=6FDP">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：这解释了为什么 Yann 对 LLM 如此看空... 😲</li><li><a href="https://x.com/altryne/status/1768683178888208816?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：Sora 团队出现在伯克利谈论 Sora</li><li><a href="https://x.com/granawkins/status/1768530196557365599?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Grant♟️ (@granawkins) 的推文</a>：“在 24 年第一季度到 25 年第四季度之间，算力将增长 14 倍。然后，如果考虑到算法效率每 9 个月翻一番，明年年底的有效算力将几乎...”</li><li><a href="https://x.com/swyx/status/1769776691562324215?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 swyx (@swyx) 的推文</a>：怎么可能和 sama 聊了 2 小时却一点干货（alpha）都没捞到，不过嘿，我们又聊到了外星人，挺有意思的。</li><li><a href="https://x.com/kk_slider_k_/status/1768464173657158132?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 KZ (@kzSlider) 的推文</a>：这非常有道理。Yann 一直在寻找能够进行视觉推理或利用规划进行推理的模型，而不仅仅是纯语言模型 ↘️ 引用 Teknium (e/λ) (@Teknium1) 这解释了为什么 Yann 是 ...</li><li><a href="https://x.com/emmanuel_2m/status/1768360522028876045?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Emm (@emmanuel_2m) 的推文</a>：🚨 今天，我们很高兴推出 Scenario #UPSCALER！将您的 AI 创作提升至 10k 分辨率。🚀 专为无与伦比的 #CreativeControl 和引导式工作流而构建。💰 起售价仅为 $15/月 ...</li><li><a href="https://x.com/xlr8harder/status/1769454853506638008?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 xlr8harder (@xlr8harder) 的推文</a>：我想我代表了这里的每一个人：3140 亿参数，搞什么鬼</li><li><a href="https://x.com/danielhanchen/status/1769550950270910630?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Daniel Han (@danielhanchen) 的推文</a>：看了下 @Grok 的代码：1. Attention 通过 30/tanh(x/30) 进行缩放？！ 2. 使用了类似 Gemma 的近似 GELU 3. 4 层 Layernorm，不像 Llama 是 2 层 4. RMS Layernorm 在最后进行下转型，不像 Llama..</li>

.</li><li><a href="https://www.nfx.com/post/ai-like-water">来自 AI Is Like Water 的推文</a>：生成式 AI 就像水。这句话源于挫败感，但它开启了 AI 策略手册的新世界。</li><li><a href="https://x.com/burny_tech/status/1769549895835226613?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Burny — Effective Omni (@burny_tech) 的推文</a>：关于 GPT-5 的新细节，来自 Sam Altman。他基本上承认了 GPT-5 将是 GPT-4 的巨大升级，因此我们可以期待类似于从 3 到 4 的跨越。“如果你忽视了进步的速度...”</li><li><a href="https://x.com/joshwalkos/status/1767745681375015076?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Champagne Joshi (@JoshWalkos) 的推文</a>：这是一段与一位缺乏内心独白的女孩的精彩对话。她非常清晰地表达了这种体验。</li><li><a href="https://youtu.be/I-HMKky7Qsw?si=yCvekF3a0zr_1IgA&t=718">超越 Transformers - RWKV 架构与 World Tokenizer 简介 - Eugene Cheah &amp; Harrison Vanderbyl</a>：超越 Transformers - RWKV 架构与 World Tokenizer 简介 - Eugene Cheah &amp; Harrison Vanderbyl，Recursal AI。Transformers 之后会是什么？在...</li><li><a href="https://www.youtube.com/watch?v=USlE2huSI_w">观看：Jensen Huang 的 Nvidia GTC 主旨演讲 - 直播</a>：在太平洋时间下午 1:00 / 东部时间下午 4:00 收看 Nvidia CEO Jensen Huang 开启两年一度的 GTC 大会。再也不会错过任何优惠！查看 CNET 的浏览器扩展程序 👉 ...</li><li><a href="https://github.com/FranxYao/Long-Context-Data-Engineering">GitHub - FranxYao/Long-Context-Data-Engineering：论文《Data Engineering for Scaling Language Models to 128K Context》的实现</a>：论文《Data Engineering for Scaling Language Models to 128K Context》的实现 - FranxYao/Long-Context-Data-Engineering</li><li><a href="https://youtu.be/J0p_thJJnoo?si=IaGuEgUcs1BRgjhF">#51 FRANCOIS CHOLLET - 智能与泛化</a>：在今天的节目中，我们邀请到了 Francois Chollet。自从读了他的《Deep Learning with Python》一书并开始使用...</li><li><a href="https://x.com">来自 GitHub - FixTweet/FxTwitter 的推文</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://www.nvidia.com/gtc/?ncid=ref-inor-332714">GTC 2024：排名第一的 AI 大会</a>：立即注册。在线直播。2024年3月18日至21日。</li><li><a href="https://docs.google.com/document/d/1HZ326V6KNK4QIlG7uEldQEizFgTaO7Hg9uJxURYy9f8/edit">NVIDIA &amp; Harpreet Sahota GTC 2024</a>：未找到描述</li><li><a href="https://youtu.be/jvqFAi7vkBc?si=WTfgLyNfGhkP2Azx">Sam Altman：OpenAI, GPT-5, Sora, 董事会风波, Elon Musk, Ilya, 权力与 AGI | Lex Fridman Podcast #419</a>：Sam Altman 是 OpenAI 的 CEO，该公司是 GPT-4, ChatGPT, Sora 以及许多其他尖端 AI 技术的幕后推手。请通过查看...来支持本播客。</li><li><a href="https://buttondown.email/ainews/archive/ainews-mm1-apples-first-large-multimodal-model/">[AINews] MM1：Apple 的首个大型多模态模型</a>：2024年3月14日至3月15日的 AI 新闻。我们为您检查了 358 个 Twitter 和 20 个 Discord（332 个频道和 2839 条消息）。预计节省的阅读时间（以 200wpm 计算）：...</li><li><a href="https://arxiv.org/abs/2402.10588">Llama 是用英语工作的吗？论多语言 Transformers 的潜在语言</a>：我们探讨了在不平衡、以英语为主的语料库上训练的多语言语言模型是否将英语作为内部中转语言——这对于理解语言模型如何...具有重要意义。</li><li><a href="https://bytez.com/read/arxiv/2402.10588">Bytez：Llama 是用英语工作的吗？论多语言 Transformers 的潜在语言</a>：在这项研究中，科学家们想知道语言模型（可以生成文本的模型）是否在内部将英语作为“中转”语言，即使是在使用其他语言进行提示时。他们发现...</li><li><a href="https://huggingface.co/collections/stereoplegic/multilingual-65389b21be39573b3b2db98d">Multilingual - stereoplegic 收藏集</a>：未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1218137415068422164)** (2 条消息): 

- **加入 Paper Club 讨论**：发布了一个提醒，邀请加入 **Paper Club 会议**，他们正在研读论文《A Comprehensive Summary Of Large Language Models》。会议定于 2 分钟后在频道 <#1107320650961518663> 开始。

- **AI 模型发布新曲**：分享了一首名为 "90s hip-hop song" 的新歌，主题是关于 **AI 模型** 创作新歌，歌词涉及 AI 对音乐的影响以及基于历史数据生成新内容的能力。该歌曲可以在 [Suno AI](https://app.suno.ai/song/83680b6f-db37-44de-adf9-3f7fff6b79d9) 找到。

**提到的链接**：<a href="https://news.ycombinator.com/item?id=39746163">Suno，一个 AI 音乐生成器 | Hacker News</a>：未找到描述

---

**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1218135292574306328)** (20 条消息🔥): 

- **探讨 Attention 背后的原因**：LLM Paper Club (Asia) 的讨论集中在阐明 *为什么开发 Transformer 中的 Attention 机制*。讨论涉及了之前固定长度编码向量的局限性，以及 **Attention 如何允许模型考虑输入序列的所有部分**。
  
- **并行化难题已解决**：一位参与者解释说，Transformer 模型中的 Attention 允许对 **不同的 Token 进行并行处理**，与 RNN 等序列模型相比，能够实现更高效的计算和更快的训练。
  
- **Attention 是效率的关键**：通过 **使用缩放点积运算（scaled dot product operation）独立处理 Token**，Attention 机制消除了 RNN 等旧模型中存在的序列化“等待”需求。
  
- **掌握 LLM 设计背后的直觉**：对话强调了一些直接跳向 GPT 模型的学习者所面临的问题：难以理解 **模型设计中的直觉决策** 以及识别这些设计所解决的问题。
  
- **对托管会议见解的感谢**：在会议结束时，参与者表达了感谢，指出由于主持人的解释，他们对 **长语言模型 (LLMs)** 的演变和基本原理有了更好的直觉理解。

---

**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1218287754715201638)** (36 条消息🔥): 

- **成员的安静签到**：一些成员今天在被动收听或表达一般性问候；由于正在开会，部分人的积极参与可能受限。

- **深度博客文章预告**：一位成员提到他们稍后将在博客上发布某个主题的详细版本，暗示将有更多关于特定讨论的信息。

- **等待的游戏**：一位成员将加载屏幕的体验比作“**RAG 体验**”，可能指的是 Retrieval-Augmented Generation 模型的使用过程。

- **RAG 讨论与资源分享**：分享了一篇题为 "Advanced RAG 01 - Small to Big Retrieval" 的文章链接，建议深入了解检索增强生成（Retrieval-Augmented Generation）：[Advanced RAG](https://towardsdatascience.com/advanced-rag-01-small-to-big-retrieval-172181b396d4)。

- **对 AI 建模替代方案的好奇**：讨论了 AI 建模中余弦相似度（cosine similarity）的替代方案，并提到了“对比嵌入（contrastive embeddings）”的概念以及 **LLMs (Large Language Models)** 在生成这些嵌入中的应用。

**提到的链接**：<a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: 每周即兴会议</a>：2024 主题, 日期, 引导者, 资源, @dropdown GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-struct...

---

**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1218220293865345024)** (168 条消息🔥🔥): 

- **Copilot 上的 Codex**：一位成员发现可以在 Copilot 应用中免费访问 **Microsoft Codex**，它提供了 Jupyter Notebooks 以及 simpy 和 matplotlib 等库。

- **LAION 的 Hugging Face 数据集**：关于 **DALL-E 3 数据集** 从 Hugging Face 移除存在困惑，随后澄清该数据集已移至新位置。提供了一个有用的数据集直接 [链接](https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset)。

- **IPFS 桥接开发**：一位成员正致力于完成一个 MLOps 平台的“模型管理器”，并正在完善 **IPFS - Hugging Face** 桥接。一个用于在 IPFS 上镜像数据集的抓取工具已经可以运行。

- **Grok-1 发布讨论**：分享并讨论了 **Grok-1** 的发布，这是一个由 OpenAI 发布的新型 314B 参数模型。它在 code/humaneval 基准测试中的表现受到关注，并与其他模型如 **Mixtral** 和 **LLaMA** 进行了对比。

- **浏览器中的 AI**：有人提出了关于在没有付费 API 的情况下在浏览器中运行语言模型的问题，从而引出了使用 **transformer.js** 等库的建议。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/datasets/en/loading#hugg">Load</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/en/loading#hugging-face-hub">Load</a>: 未找到描述</li><li><a href="https://tenor.com/view/silicon-valley-yes-cheer-think-gif-9010547">Silicon Valley Yes GIF - Silicon Valley Yes Cheer - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.economist.com/business/2023/11/23/why-chinese-companies-are-flocking-to-mexico">为什么中国公司正涌向墨西哥</a>: 该国提供了进入美国的后门</li><li><a href="https://fxtwitter.com/imgn_ai/status/1769791182270333067">来自 imgnAI (@imgn_ai) 的推文</a>: catgirls 正在 NVIDIA GTC ✨ 为您的创作自由而喵喵叫 👊 这是一个需要被听到的消息 🐱💕</li><li><a href="https://github.com/victorchall/EveryDream2trainer/blob/main/caption_cog.py">EveryDream2trainer/caption_cog.py 位于 main · victorchall/EveryDream2trainer</a>: 通过在 GitHub 上创建账户来为 victorchall/EveryDream2trainer 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/aiwars/comments/1bbxtp6/the_people_behind_the_nightshade_glaze_account/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok 开源发布</a>: Grok 开源发布。通过在 GitHub 上创建账户来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/OpenDatasets/dalle-3-dataset">OpenDatasets/dalle-3-dataset · Hugging Face 上的数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1218181910669295716)** (13 条消息🔥): 

- **频道主题澄清**：成员们指出，关于与免费 Colab 相关的 Web UI 讨论可能不适合 **research** 频道，因为这不属于前沿研究。
  
- **分享生成式世界模型文档**：分享了一个标题为 "Generative Audio Video Text world model" 的 Google 文档链接，但未提供额外的评论或解释。[查看文档](https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk)。

- **在新数据上预训练 LLM**：提到了一篇 **arXiv 论文**，讨论了与在新数据上重新训练语言模型相比，结合学习率预热和旧数据回放等简单技术如何节省计算资源。[阅读文章](https://arxiv.org/abs/2403.08763)。

- **GitHub 上的 Grok 开源发布**：链接了一个 **Grok 开源发布** 的 GitHub 仓库，未对其内容或影响进行进一步讨论。[探索仓库](https://github.com/xai-org/grok-1)。

- **关于 Nvidia 确认 GPT-4 细节的推测**：围绕一条传闻展开了讨论，该传闻引用了 Twitter 上的一张图片，称 Nvidia 确认 **GPT-4** 是一个拥有 1.8 万亿参数的混合专家模型 (MoE)。[查看推文](https://pbs.twimg.com/media/GI-reRIW0AAZpMC?format=jpg&name=large)。同时也有人指出，GPT-4 的确切身份仍处于推测阶段。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.08763">持续预训练大型语言模型的简单且可扩展策略</a>: 大型语言模型 (LLM) 通常在数千亿个 token 上进行预训练，一旦有新数据可用，就必须重新开始该过程。一种更有效的解决方案是持续预训练...</li><li><a href="https://arxiv.org/abs/2403.09611">MM1：多模态 LLM 预训练的方法、分析与见解</a>: 在这项工作中，我们讨论了构建高性能的多模态大型语言模型 (MLLM)。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细且持续的...</li><li><a href="https://docs.google.com/document/d/1f6CpVjdApmQl3nXsUACGtSd9nML3CypI1iE889i4JbM/edit?usp=drivesdk">生成式音频视频文本世界模型</a>: 未找到描述</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok 开源发布</a>: Grok 开源发布。通过在 GitHub 上创建账户来为 xai-org/grok-1 的开发做出贡献。
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1218310691103178803)** (43 条消息🔥): 

- **探索光子学前沿**：[Anastasia 的 YouTube 视频](https://youtu.be/8ohh0cdgm_Y)讨论了一种速度快一千倍的新型芯片技术，并分享了该视频及其相关的 Nature 论文链接。关于光子学的进一步视频推荐包括 [Asianometry 频道](https://www.youtube.com/watch?v=29aTqLvRia8)，主题涵盖硅光子学和基于光的神经网络。

- **PyTorch vs. TensorFlow：内存管理选择解析**：深入讨论了 PyTorch 决定向用户开放张量内存管理的原因，强调了避免隐藏副本、"no magic" 原则以及在数学运算中进行显式设备处理。

- **寻找最新的 GPU Profiling 工具？**：用户讨论了允许在 Ada 或 Hopper GPU 上使用 Nsight Compute 进行性能分析的云 GPU 服务，推荐了 [RunPod](https://www.runpod.io/) 和 [Lambda Labs](https://lambdalabs.com/) 等建议，并有报告称某些服务未授予性能分析所需的必要权限。

- **NVIDIA GTC 主题演讲引发热议**：在 2024 年 3 月的 GTC 主题演讲中，NVIDIA CEO 黄仁勋（Jensen Huang）提到的一个 1.8T 参数的最先进模型引起了成员们的好奇，同时还讨论了新硬件的发布，如搭载 192GB HBM 的 B100、安全增强功能以及互连技术。

- **入门与定位**：一位新成员寻求关于在社区内何处进行自我介绍的指导，得到的建议是前往围绕特定技术和库构建的频道，例如为了平稳起步可以前往 beginner 频道。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/generated/torch.set_default_device.html">torch.set_default_device &mdash; PyTorch 2.2 documentation</a>：未找到描述</li><li><a href="https://www.runpod.io/">以低至 $0.2/小时的价格租用云端 GPU</a>：未找到描述</li><li><a href="https://www.cerebras.net/product-chip/">产品 - 芯片 - Cerebras</a>：未找到描述</li><li><a href="https://lambdalabs.com/">GPU 云、集群、服务器、工作站 | Lambda</a>：为深度学习和 AI 提供的 GPU 云、GPU 工作站、GPU 服务器和 GPU 笔记本电脑。可选 RTX 4090, RTX 3090, RTX 3080, RTX A6000, H100 和 A100。预装 Ubuntu, TensorFlow 和 PyTorch。</li><li><a href="https://youtu.be/8ohh0cdgm_Y?si=q3wOMlzp_Nmn8_AJ">新芯片突破：快 1000 倍</a>：立即获取 TypeAI PREMIUM！点击此处链接开始免费试用：https://bit.ly/Mar24AnastasiInTech 论文地址：https://www.nature.com/articles/s41586...</li><li><a href="https://www.youtube.com/live/Y2F8yisiS6E?si=g5MChTXs3a9gGykE">NVIDIA CEO 黄仁勋 GTC 2024 年 3 月主题演讲</a>：观看 NVIDIA CEO 黄仁勋的 GTC 主题演讲，了解所有关于塑造我们未来的 AI 进展的公告。深入了解这些公告并发现...</li><li><a href="https://lightmatter.co/">Lightmatter®</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=29aTqLvRia8">硅光子学：下一次硅革命？</a>：衷心感谢本频道的朋友、来自 MIT 的 Alex Sludds 建议了这个话题并为我提供了关键资源。在这里关注他：https://a...</li><li><a href="https://www.youtube.com/watch?v=t0yj4hBDUsc">在光网格上运行神经网络</a>：我要感谢 Alex Sludds 在帮助我研究和制作这段视频方面所做的努力。在这里查看他的工作：https://alexsludds.github.io 链接：- The As...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1218241351582482493)** (7 条消息): 

- **新的 Triton 调试可视化工具**：一位成员介绍了一个新的可视化工具，旨在通过提供更好的 **load/stores 空间结构** 视图来简化 Triton 中的调试过程。未提供关于可视化工具外观的具体细节。
- **尝试 Triton Puzzles**：同一位成员还分享了一套 **Triton Puzzles**，这些谜题被认为具有一定挑战性，但有助于理解复杂问题。感兴趣的成员可以尝试一下，并在该 [Google Colab 链接](https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing) 报告发现的任何问题。已知错误包括偶尔出现的重复可视化和段错误（segmentation faults）。
- **寻找 Triton 学习资源？**：一位熟悉 CUDA 的成员询问 Triton 的学习资源。回复建议使用官方 Triton 教程、上述谜题，以及通过为流行的 Triton kernel 编写注释来进行学习。
- **对 Triton 资源的认可**：多位成员对 Triton Puzzles 以及在 CPU 上运行解释器的想法给出了积极回应，表示将探索这些资源。其中一条回复对分享的内容进行了细微的文字修正。

**提到的链接**：<a href="https://colab.research.google.com/drive/1AJc8RFsDeJ3Vx3gRq5dUqmcb-Cy1G8qh?usp=sharing">Google Colaboratory</a>：未找到描述

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1218467001450627072)** (68 条消息 🔥🔥):

- **CUDA Warp Scheduler Inquiry**: 一位成员询问了如何定义 **warp schedulers** 的数量以及每个 **warp scheduler** 控制的 **threads** 数量，旨在了解可以同时运行的 **threads** 总数，以优化效率和 **occupancy**。
  
- **Active Warp Clarification Sought**: 讨论了 **active warp** 这一术语，并寻求关于 **warp** 内 **threads** 涉及场景的澄清，以及这如何影响一个 **warp** 是否被视为 **active**。提供了代码示例来说明困惑点，例如没有 **threads** 满足条件的 **warp** 是否仍符合 **active** 的资格。

- **Memory Manager Abstraction Debated**: 展开了关于 CUDA 中 **memory manager** 的广泛讨论，探索了在内存空间内为数据的 **producers** 和 **consumers** 管理指针的语义和实用性。辩论了 **ProducerProvides**、**ConsumerTakes** 等概念，揭示了在优化 CUDA 应用程序内存使用时对异步工作（async work）和流同步（stream synchronization）的担忧。

- **Reports from the Video-Pipeline Frontier**: 一位成员展示了他们在优化视频流水线（video pipeline）方面的工作，重点是在 **producer** 和 **consumer** 内存空间之间高效传输数据。关于 **Manager** 类接口以及延迟、异步拷贝（async copies）和内存瓶颈在流水线并行（pipeline parallelism）中的作用，进行了活跃的反复讨论。

- **Sharing CUDA Project Architecture Best Practices**: 就 CUDA 中的项目结构进行了问答，特别是 `main()` 函数应该驻留在 `.cpp` 还是 `.cu` 文件中，以及如何正确包含来自 `.cu` 文件的 **kernel** 函数。这引发了关于需要清晰的 CUDA 项目组织教育资源的共同看法。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=Y2F8yisiS6E">GTC March 2024 Keynote with NVIDIA CEO Jensen Huang</a>: 观看 NVIDIA CEO 黄仁勋的 GTC 主旨演讲，了解塑造我们未来的 AI 进展的所有发布。深入了解发布内容并发现...</li><li><a href="https://github.com/tspeterkim/flash-attention-minimal">GitHub - tspeterkim/flash-attention-minimal: Flash Attention in ~100 lines of CUDA (forward pass only)</a>: 约 100 行 CUDA 代码实现的 Flash Attention（仅前向传递） - tspeterkim/flash-attention-minimal
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1219091487455711414)** (5 messages): 

- **Exploring Reconfigurable Computing and ML**: 分享了一个名为 "Prof. Mohamed Abdelfattah" 的 YouTube 视频和一个网站，重点关注康奈尔大学 Abdelfattah 教授团队关于可重构计算和高效机器学习的研究。邀请观众[探索他们的研究](https://www.mohsaied.com/)。

- **Hardware-Centric View of Machine Learning Systems**: 提供了关于 ECE 5545 (CS 5775) 的信息，这是一门以硬件为中心的机器学习课程，涵盖了 ML 算法硬件/软件、优化技术和系统设计等主题。鼓励感兴趣的参与者[阅读教学大纲](https://abdelfattah-class.github.io/ece5545/)。

- **Textbook Mystery in Machine Learning Course**: 一位用户指出 ECE 5545 的参考网站没有指明该课程的“教科书”是什么，称其“很奇怪”。

- **Solving the Textbook Puzzle**: 针对教科书的疑问，有人提到该课程的第一节讲座视频揭示了教科书信息，强调了补充课程材料的重要性。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://abdelfattah-class.github.io/ece5545/">ML Hardware and Systems</a>: 未找到描述</li><li><a href="https://www.youtube.com/@mabdelfattah88">Prof. Mohamed Abdelfattah</a>: 这是康奈尔大学 Mohamed Abdelfattah 教授研究小组的频道。我们正在研究可重构计算和高效机器学习。欲了解更多信息，请查看...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/)** (1 messages): 

vim410: 取决于情况。但是的。
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1219389682241110147)** (5 messages):

- **扎实的 CUDA 基础，为 ML 做好准备**：andreaskoepf 认可了 al0vya 在 CUDA 方面的扎实基础，并建议尝试使用像 **torch** 这样的深度学习框架来开始 ML/DL 的学习，因为这通常涉及*矩阵乘法、逐点非线性（pointwise non-linearities）、softmax 和归一化（normalization）*。
- **CUDA 精通书籍推荐**：andreaskoepf 建议阅读《Programming Massively Parallel Processors》以获得更深入的 CUDA 知识，并补充说虽然该书关于 DL 的内容较少，但它仍然是一本*优秀的通用 CUDA 编程书籍*。[Amazon 上的 Programming Massively Parallel Processors](https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311)。

**提到的链接**：<a href="https://www.amazon.de/-/en/Wen-mei-W-Hwu/dp/0323912311">未找到标题</a>：未找到描述

---

**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1218146385942286407)** (6 条消息): 

- **CUDA 索引困惑已解决**：一名成员对索引表达式 `i = blockIdx.x * blockDim.x + threadIdx.x * 2` 提出疑问，随后得到的澄清是，这种计算可能会导致线程间的索引**重复计算（double-counting）**。举例说明，两个不同的线程最终可能会被分配到相同的索引。
- **考虑在博客发布练习题解答**：一名成员询问了关于将 CUDA 书籍练习题解答**发布在博客上**可能存在的问题，表示难以联系到作者，并对毕业后失去教育邮箱账号感到遗憾。
- **寻求公开内容的许可**：在有人提醒某些内容可能**仅限讲师（instructor only）**后，另一名成员回应称，他们将向 **Wen-mei**（推测是作者之一）确认是否可以公开分享练习题解答。

---

**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1218239914542366790)** (14 条消息🔥): 

- **团队成员因日程繁忙致歉**：一位聊天参与者表示他们非常忙碌，待日程空闲后会通知小组。
- **成员表示难以找到代码**：一名成员表示无法找到特定代码，另一名团队成员提供了一个 [Triton kernel commit](https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5) 链接以提供帮助。
- **寻求关于 Ring Attention 内存需求的澄清**：一名成员正在撰写博客文章，需要澄清 Ring Attention 与 Flash Attention 的内存需求对比，特别是在相对于块大小（block size）的线性内存缩放方面。
- **建议阅读论文以获取见解**：为了更好地理解 Ring Attention 的性能特征，有人建议阅读一篇关于 [Striped Attention 的 arXiv 论文](https://arxiv.org/abs/2311.09431)，其中包含有用的视觉图表。
- **关于 Flash Attention 内存占用的辩论**：讨论继续进行，多名成员辩论 Flash Attention 的内存需求是否确实随块大小 c² 线性缩放，并引用了 [GitHub 上的 Flash Attention 实现](https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2311.09431">Striped Attention: Faster Ring Attention for Causal Transformers</a>：为了帮助应对 Transformer 模型中日益增长的长序列长度需求，Liu 等人最近提出了 Ring Attention，这是一种能够克服单设备内存限制的精确注意力算法...</li><li><a href="https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_kernel.h">flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h at main · Dao-AILab/flash-attention</a>：快速且内存高效的精确注意力机制。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。</li><li><a href="https://github.com/zhuzilin/ring-flash-attention/commit/10d992c3c84a2ee1a2e47dd596615d9aad46f7d5">add naive triton kernel for varlen · zhuzilin/ring-flash-attention@10d992c</a>：未找到描述
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1218332053032927322)** (5 条消息): 

- **AI 与系统在 MLSys 2024 交汇**：成员们讨论了即将于 5 月举行的 MLSys 2024 会议，强调了其在机器学习与系统交叉领域的跨学科性质。该会议被视为解决 AI 领域未来挑战的关键，特别关注整体性方法（holistic approaches）([MLSys Conference](https://mlsys.org/))。

- **当手机不够“聪明”时**：一个幽默的评论将智能手机称为 "Not so smart phone"（不那么智能的手机），但未提供具体背景来理解所引用的潜在问题或主题。

- **计算器难题引发辩论**：成员们就执行计算的正确方式展开了辩论，认为乘法和除法的执行顺序很重要，而另一位成员指出科学计算器处理 `ax` 和 `a×x` 的方式可能不同。未提供具体的例子或进一步的解释。

**提到的链接**：<a href="https://mlsys.org/">MLSys 2024</a>：未找到描述

---

**CUDA MODE ▷ #[gtc-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1218444664315711498)** (9 条消息🔥): 

- **GTC 2023 见面会计划公开**：一位成员计划在周一早上参加 GTC，公开邀请他人见面，并提议通过 DM 分享电话号码。
- **活动爱好者确定日期**：另一位成员宣布他们将于 3 月 14 日至 25 日参加活动，并愿意在活动期间见面。
- **看到日程后决定延长行程**：对会议日程的兴奋促使一位成员考虑参加整周的活动，前提是有像样的 Wi-Fi。
- **GTC Meme 幽默**：一位成员幽默地建议应该做一个关于无法参加 GTC 的 Meme。
- **志愿者希望落空**：一位成员对联系 GTC 志愿者以获取免费门票未果表示失望。
- **理想的潜入策略？**：在提到需要另一种进入 GTC 的方式后，一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Sfrjpy5cJCs) 链接，标题为“我潜入了一个秘密军火商会议”，幽默地暗示了一种非传统的参加会议的方法。

**提到的链接**：<a href="https://www.youtube.com/watch?v=Sfrjpy5cJCs">I Snuck Into A Secret Arms-Dealer Conference</a>：每月在 https://www.patreon.com/Boy_Boy 获取独家视频。这是我们与传奇的澳大利亚政治讽刺团体 The C... 合作制作的。

---

**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1218183723200155748)** (159 条消息🔥🔥): 

- **LLM 格式灵活性确认**：快速确认了 LLaMa 模型可以使用包含 "system"、"user" 和 "assistant" 角色的 Prompt 格式，这与 OpenAI JavaScript 库的用户相关。
  
- **书籍内容处理**：一位用户解释了如何创建一个脚本，将书籍拆解并提示模型相应地生成片段。使用了 Airoboros 70B，并与 lzlv 70B 进行了比较，观察到基于指令的数据可以提高生成质量。

- **寻求详细的分析数据**：用户表示需要类似于 OpenAI 提供的详细使用分析，显示对每日或每周使用成本的需求，以及可能按模型和 App 进行的细分。

- **模型审核与访问查询**：用户报告了模型执行任务意愿的变化，并询问目前通过 API 访问 sonnet:beta 和 opus:beta 的问题，公司确认大多数用户可以正常访问。

- **潜在的新 API 上架**：一位用户表示他们正在建立自己的公共 API，并询问是否可以将其列在 OpenRouter 上，官方回应表示欢迎并邀请通过私信提供更多细节。

- **关于模型成本与性能的讨论**：讨论了使用不同模型的成本，例如 Claude 3 Opus 与 Sonnet 等其他模型的对比，用户就这些 AI 模型的负担能力和性能交换了意见。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai">OpenRouter</a>：LLM 和其他 AI 模型的路由服务</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>：Grok 开源发布。通过在 GitHub 上创建账号为 xai-org/grok-1 的开发做出贡献。
</li>
</ul>

</div>

---

**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1218212402127175711)** (95 条消息🔥🔥): 

- **关于 Streaming API 的疑问**：一位用户询问了 **astream_log** 和 **astream_events** 之间的区别，询问 **astream_log** 是否会被弃用以支持 Beta 版的 **astream_events**，或者它们只是具有不同用例的两个 API。

- **高级研究助手招募 Beta 测试人员**：发出了一项名为 **Rubik's AI** 的高级研究助手的 Beta 测试邀请。感兴趣的用户可以加入候补名单，通过 [Rubik's AI](https://rubiks.ai/) 获得 **Claude 3 Opus**、**GPT-4 Turbo** 和 **Mistral Large** 等高级功能的使用权限。

- **LangChain 文档的反馈与建议**：一位用户表示在查阅 LangChain 文档时存在困难，特别是对初学者而言。一条回复邀请用户针对令人困惑的页面提供具体反馈，或对缺失的内容提出建议。

- **使用 LangChain 获取 LLM 的结构化输出**：一位用户询问如何使用 LangChain 从 LLM 获取结构化输出，例如列出城市及其人口。提供了一个详细的代码示例，使用 **PydanticOutputParser** 来定义所需的输出结构。

- **通过 LangChain 实现 Google Gemini 的函数调用 (Function Calls)**：讨论了如何通过 LangChain 让 Vertex AI 上的 Gemini 模型感知到函数的存在，从而使 LLM 能够响应查询并调用函数。对话中提到了使用 `.bind(functions=[schema])` 将函数 schema 传递给 LLM。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>：未找到描述</li><li><a href="https://codelabs.developers.google.com/codelabs/gemini-function-calling#4.">无标题</a>：未找到描述</li><li><a href="https://bloon.ai">Bloon AI</a>：重新定义智能学习</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19239">Feature Request: Support for Negative Embeddings in Similarity Searches · langchain-ai/langchain · Discussion #19239</a>：已检查，我搜索了现有的想法，没有找到类似的。我添加了一个非常详细的标题，并清楚地描述了功能请求及其动机。功能请求：我建议增加...</li><li><a href="https://www.teradata.com/insights/ai-and-machine-learning/using-natural-language-to-query-teradata-vantagecloud-with-llms">Using Natural Language to Query Teradata VantageCloud With LLMs| Teradata</a>：学习如何将您的英语查询翻译成 SQL，并从您的分析数据库中以通俗易懂的英语接收响应。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1219304272244510741)** (45 messages🔥): 

- **JavaScript 中 `RemoteRunnable` 流式传输的问题**：一位用户在使用 JavaScript 的 `RemoteRunnable` 进行流式输出时遇到挑战。虽然在 Python 中运行正常，但同样的代码在 JavaScript 中会降级为 `/invoke` 而不是调用 `/stream`。
- **请求澄清流式传输机制**：用户寻求关于流式传输为何未按预期工作的澄清，质疑是否是因为 `RunnableSequence` 从 `Runnable` 继承了调用 `invoke` 的 `_streamIterator` 导致的问题。
- **寻求 LangChain 团队的支持**：用户询问如何就流式传输问题联系 LangChain 团队。AI 建议在 GitHub 上报告问题，或根据 Security Reporting Guidelines 通过电子邮件联系。
- **最近的更新中没有已知的修复**：没有提供关于最近可能解决流式传输问题的任何更改信息。AI 建议查看 LangChain 的 GitHub 仓库以获取最新更新。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://js.langchain.com/docs/security#reporting-a-vulnerability>).">Security | 🦜️🔗 Langchain</a>：LangChain 拥有庞大的集成生态系统，包括本地和远程文件系统、API 和数据库等各种外部资源。这些集成允许开发人员创建多功能的应用程序...</li><li><a href="https://api.js.langchain.com/classes/langchain_core_runnables_remote.RemoteRunnable.html#pipe>):">RemoteRunnable | LangChain.js - v0.1.28</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13126>)),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11998>)),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13723>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/17315>)).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1218223379690029179)** (11 messages🔥): 

- **用于数据分析的新型 AI 聊天机器人**：一位用户分享了 GitHub 上的 [Haste171/langchain-chatbot](https://github.com/Haste171/langchain-chatbot) 链接，这是一个旨在以对话形式分析和提取数据信息的 AI 聊天机器人。
- **AI 书签管理**：[Living Bookmarks](https://twitter.com/uogbuji/status/1768681648516661446) 已在 GitHub 上开源，这是一个 Discord AI Chatbot，可以与 Raindrop.io 书签交互，帮助用户在需要时找到相关书签。
- **寻求生产力洞察**：一位用户正在构建一个数字顾问，并邀请技术和专业服务人员讨论生产力、身体和心理健康需求，提供 [30 分钟的咨询时段](https://calendly.com/neurofusion/30min)。
- **基于 AI 的爬虫受到关注**：[Scrapegraph-ai](https://github.com/VinciGit00/Scrapegraph-ai) 是一个使用 LangChain 构建的基于 AI 的爬虫，已在 pip 上发布，安装量超过 2300 次，鼓励用户通过 Star 项目来表示支持。
- **模拟销售角色的 AI 解决方案**：一篇 Twitter 帖子详细介绍了 **Lyzr.ai's Automata** 如何模拟 SDR 和 AE 职能，在多个 AI Agent 以及 OpenAI 和 *Perplexity* 等工具的帮助下，完成从处理邮件列表到达成销售的全过程。项目仓库可在 [GitHub](https://github.com/LyzrCore/lyzr-automata) 上获取。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://calendly.com/neurofusion/30min">User Interview 🔎 - NEUROFUSION Research, Inc.</a>：嘿，我正在构建一个数字顾问，以帮助改善你在工作和生活其他领域的表现。我很想与你交流，了解你在生产力、身体和...方面的需求。</li><li><a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.</a>：用于以对话格式分析/提取数据信息的 AI 聊天机器人。- Haste171/langchain-chatbot</li><li><a href="https://github.com/VinciGit00/Scrapegraph-ai">GitHub - VinciGit00/Scrapegraph-ai: Python scraper based on AI</a>：基于 AI 的 Python 爬虫。通过在 GitHub 上创建账户来为 VinciGit00/Scrapegraph-ai 的开发做出贡献。</li><li><a href="https://youtu.be/vHjc5CEoIJE">Making an AI application in 15 minutes</a>：技术栈 - 自定义 UI 和 RAG：open-webui 的调整版本。- 本地 LLM 托管：用于本地托管 LLM 的 Ollama。- 数据隐私：集成 DaxaAI 的 Pebblo 以...</li><li><a href="https://navvy.co/.">Home</a>：我对 AI 充满热情。让我们联系起来，释放 AI 的潜力，并在创新项目上进行合作！</li><li><a href="https://x.com/siva_1gc/status/1768997890544800070?s=20">Siva Surendira (@siva_1gc) 的推文</a>：这比我们想象的要多花一点时间.. 但它来了.. 😎 使用 @lyzrai Automata 和 @OpenAI 实现 SDR 和 AE 职能的自动化... 运行在 @awscloud 上 - 安全且私密.. 它是如何工作的？👇 Agent 1:...</li><li><a href="https://github.com/LyzrCore/lyzr-automata">GitHub - LyzrCore/lyzr-automata: low-code multi-agent automation framework</a>：低代码多 Agent 自动化框架。通过在 GitHub 上创建账户来为 LyzrCore/lyzr-automata 的开发做出贡献。</li><li><a href="https://amzn.eu/d/3Dcdsbk">未找到标题</a>：未找到描述</li><li><a href="https://amzn.eu/d/2uVnCp8">未找到标题</a>：未找到描述</li><li><a href="https://www.facebook.com/casi.schulze.10">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1218824643436085321)** (2 messages): 

- **个性化营养 AI 演示**：一个名为 **Nutriheal** 的个性化营养 AI 应用展示了如何使用 **Ollama** 和 **Open-webui** 等工具，并通过 Daxa AI 的 **Langchain Pebblo** 集成了隐私保护。一段 *YouTube 视频教程* 解释了如何在 15 分钟内创建此类应用程序，强调了用户友好性和数据保护。[在此观看视频](https://youtu.be/vHjc5CEoIJE)。

- **探索如何本地构建 AI**：该教程还推广了关于在本地构建和部署 AI 解决方案的指南，打破了只有大型科技公司才能处理 AI 的神话。这些资源旨在为个人用户简化复杂 AI 模型的设置和执行。[在此阅读指南](//build-and-deploy-genai-solutions-locally)。

- **AI 聊天助手的通用 UI**：另一份可用资源讨论了为自定义 LLM（Large Language Model）助手创建通用聊天 UI，重点关注不同 AI 解决方案的可重用界面。这暗示了在个人 AI 开发中更广泛的应用和集成的便利性。[在此查看 UI 指南](/generic-ui-for-custom-llm-assistants)。

- **使用 Langgraph 的 Plan-and-Execute 教程**：分享了一个教学视频，内容是受 Plan-and-Solve 论文和 Baby-AGI 项目启发，创建一个“计划并执行”（plan-and-execute）风格的 AI Agent。核心目标是在 AI Agent 中模拟战略规划和执行。[在此查看教程](https://www.youtube.com/watch?v=ZlJbaYQ2hm4)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=ZlJbaYQ2hm4">使用 Langgraph 进行 Plan-and-Execute</a>：如何创建一个“计划并执行”风格的 Agent。这在很大程度上受到了 Plan-and-Solve 论文以及 Baby-AGI 项目的启发。核心思想是首先...</li><li><a href="https://youtu.be/vHjc5CEoIJE">在 15 分钟内制作 AI 应用程序</a>：技术栈 - 自定义 UI 和 RAG：open-webui 的调整版本。- 本地 LLM 托管：用于本地托管 LLM 的 Ollama。- 数据隐私：集成了 DaxaAI 的 Pebblo 以...</li><li><a href="https://navvy.co/.">主页</a>：我对 AI 充满热情。让我们联系起来，释放 AI 的潜力，并在创新项目上展开合作！
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1218217772765544448)** (8 条消息🔥): 

- **通过 API 查询揭示模型秘密**：链接到一篇 [arXiv 论文](https://arxiv.org/abs/2403.09539)，探讨了利用 API 查询获取受 API 保护的大语言模型（LLMs）（如 OpenAI 的 gpt-3.5-turbo）非公开信息的可能性。论文强调了一个 “softmax 瓶颈”（softmax bottleneck），它可能会泄露模型的隐藏层大小（hidden size）和其他细节。

- **模型大小估计曝光**：一位成员讨论了 Carlini 等人的另一篇论文，该论文使用 logits 来估计模型大小，但删减了这些细节；他评论说，当前的这篇论文在没有删减的情况下进行了类似的分析。

- **对 7B 模型大小发现的惊讶**：一位成员对论文中关于某个模型可能只有 7B 大小的暗示表示惊讶。

- **关于模型大小估计不准确的推测**：另一位成员对 7B 模型大小的发现表示怀疑，认为除非存在某种先进的蒸馏（distillation）方法，否则这可能是不准确的。

- **MoE 对模型大小估计的误导**：讨论涉及了如果相关模型使用混合专家模型（MoE），模型大小计算可能存在的不准确性，并指出像 Mistral 这样的模型已经具有相当大的嵌入维度（embedding dimension）。

**提到的链接**：<a href="https://arxiv.org/abs/2403.09539">受 API 保护的 LLMs 的 Logits 会泄露专有信息</a>：大语言模型（LLMs）的商业化导致了对专有模型仅提供高级 API 访问的普遍做法。在这项工作中，我们证明即使在保守的假设下...

  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1219339209270362135)** (19 条消息🔥): 

- **预见 ML 争议**：聊天中分享的一条推文预测，在 Twitter 上关于 [开源定义的一场交流](https://twitter.com/rasbt/status/1769779229263065184) 之后，可能会出现争议。
- **寻求 OSS 的清晰定义**：聊天成员表示希望开源软件（OSS）社区能够就什么是开源达成明确立场，旨在结束持续的争论。
- **对开源定义中排除数据的批评**：有一种观点认为，在开源定义中排除**数据**是一个糟糕的决定，成员们对这种潜在的立场已经感到不满。
- **定义开源的实用性**：人们正努力建立开源的实用定义，以平息有争议的讨论并达成共识。
- **对在线互动的沮丧**：一位用户对 EleutherAI 处理在线对话的方式表示沮丧，暗示这可能会适得其反，并提到打算避开 Twitter，专注于写博客。

**提到的链接**：<a href="https://x.com/BlancheMinerva/status/1769792488091353099">Stella Biderman (@BlancheMinerva) 的推文</a>：@natolambert @felix_red_panda 不过你错了 :P

  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1219005089826607185)** (63 条消息🔥🔥):

- **Grok-1 向公众发布**：xAI 宣布发布 [Grok-1](https://x.ai/blog/grok-os)，这是一个拥有 **3140 亿参数的 Mixture-of-Experts 模型**，在 JAX 和 Rust 之上构建了自定义训练栈。模型权重和架构在 Apache 2.0 许可证下通过 [github.com/xai-org/grok](https://github.com/xai-org/grok) 提供。
- **Grok-1 模型细节引发讨论**：聊天参与者对 **Grok-1** 的性能和发布策略提出质疑，认为它可能“准备不足”或发布仓促。讨论还涉及了此类模型的营销及其分发方式的重要性。
- **与 Falcon 的对比**：针对 Grok 的性能出现了推测，根据给出的 GSM8K (45.94) 和 MMLU (70.5) 基准测试分数，有观点认为 Grok 的表现似乎优于 Falcon 模型。
- **对通过 Torrent 分发模型的担忧**：通过 Torrent 分发 Grok 引发了关于其对开放 AI 团队和政策制定影响的辩论，一些人认为这可能会影响开源模型的公信力和政策支持。
- **关于通过邮件分发模型的幽默建议**：一场关于通过 FedEx 闪存盘分发大型 AI 模型的成本效益的幽默辩论被触发，讽刺地提出“邮购模型业务”作为传统在线出站流量成本的替代方案。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.ai/blog/grok-os">Grok-1 的开放发布</a>：未找到描述</li><li><a href="https://www.wheresyoured.at/peakai/">我们是否已经达到了 AI 的巅峰？</a>：上周，《华尔街日报》发表了对 OpenAI CTO Mira Murati 长达 10 分钟的采访，记者 Joanna Stern 提出了一系列深刻而直接的问题，而 Murati...</li><li><a href="https://x.com/thexeophon/status/1769449427972858103?s=46">Xeophon (@TheXeophon) 的推文</a>：Chinchilla 定律不直接适用于 MoE，对吧？如果适用，我们可以推断出 Grok 的训练数据集大小。它大得出乎意料，所以我猜他们在时间有限的情况下优先考虑了最优性...</li><li><a href="https://fxtwitter.com/grok/status/1769441648910479423">Grok (@grok) 的推文</a>：@elonmusk @xai ░W░E░I░G░H░T░S░I░N░B░I░O░
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1218732428462395502)** (6 条消息): 

- **寻求关于 Aribus 开发的澄清**：一位成员询问其他人正在使用 **Aribus** 开发什么，并附上了一个令他们感到困惑的 [Twitter 链接](https://twitter.com/alignment_lab/status/1758949148143841379)。后续消息中未提供更多细节或澄清。
- **寻找感知 HTTP 的 Embeddings 模型**：有人表示有兴趣寻找专门针对 **HTTP 响应**训练的 Embeddings 模型，并寻求从何处开始搜索的指导。他们还提到，只要经过正确的训练，任何 Transformer 模型都可以用作 Embedding 模型。
- **寻找经过特殊训练的 Mistral 模型**：一位成员正在寻找在 *orca-math-word-problems-200k 数据集* 和 *nvidia/OpenMathInstruct-1* 上都进行过微调 (FT) 的 **Mistral 模型**。目前没有分享后续信息或建议。
- **简短的问候**：一位用户仅以简短的 "hi" 进入聊天。该问候之后没有实质性的讨论。
  

---


**Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1219081302683422851)** (32 条消息🔥):

- **关于 Grok 1 微调协作的呼吁**：一位成员正在寻求微调 **Grok 1** 的合作。这是一个规模庞大、可能训练不足的模型，强调了对大量 **compute** 和 **data** 资源的需求。他们提到现有的 MoE 训练基础设施已经就绪。
- **Grok 1 基准测试性能的潜在问题**：讨论揭示了对 **Grok 1** 在 MMLU 基准测试上表现的担忧，成员们建议需要更多的 compute 能力以及在多样化数据集上进行持续预训练。大家对该模型与 **Mixtral** 等其他模型的性能对比感到好奇。
- **关于模型价值和成本效益的辩论**：对于进一步训练 **Grok 1** 与其他模型相比的成本效益存在怀疑，并且有人质疑它是否能成为最优秀的开源 LLM，或者超越 **GPT-4** 和 **Claude** 等模型。
- **对数据集的好奇与 Jax 专家**：参与者正在探索微调的理想数据组合，并确认了一位自荐的 **Jax 专家** 的加入。数据需求的细节以及训练工作的收益是讨论的重点。
- **Grok 1 出人意料的表现**：一位成员指出 **Grok 1** 在[预留的高中决赛试题集](https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam)中展现了令人惊讶的能力，提到它在该特定考试中的表现接近 **GPT-4** 和 **Claude**。

**提到的链接**：<a href="https://huggingface.co/datasets/keirp/hungarian_national_hs_finals_exam">keirp/hungarian_national_hs_finals_exam · Hugging Face 数据集</a>：未找到描述

---

**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1218226914322415677)** (1 条消息): 

- **Devin 激发了“懒人式”应用开发**：一位成员表达了 **Devin** 如何激励他们变得“甚至懒得往终端粘贴东西”来构建简单的应用。他们认为任何比本地应用更复杂的东西都是大材小用，并质疑当前开源解决方案的有效性。

---

**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1218206756031955006)** (7 条消息): 

- **对算法统治者的恐惧**：分享的一条 [推文](https://x.com/tszzl/status/1768530219378631137?s=20) 暗示 **Anthropic** 可能在充当“受控反对派”，以在技术人员中制造恐惧。
- **除人物图像外，内容审查运行顺畅**：关于内容审查，该成员除了在包含人物的图像上遇到过“**直接拒绝**”的情况外，没有遇到其他问题。
- **探索高吞吐量场景下的 Claude Sonnet**：一位成员正考虑在一个预计每月消耗数千万 token 的项目中使用 **Claude Sonnet**，并正在咨询在这种规模下的使用经验。

**提到的链接**：<a href="https://x.com/tszzl/status/1768530219378631137?s=20">来自 roon (@tszzl) 的推文</a>：anthropic 是受控反对派，旨在让技术人员心生敬畏。

---

**LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1218241222347460619)** (16 条消息🔥): 

- **KPU 作为 LLM 的新解决方案亮相**：Maisa 推出了 [知识处理单元 (KPU)](https://maisa.ai/blog/kpu)，这是一个声称性能超越 GPT-4 等先进语言模型的框架。它在 AI 系统内部将推理与数据处理分离，以增强处理复杂任务的能力。
- **关于 KPU 基准测试的困惑**：讨论中有人质疑为什么将 **KPU**+GPT-4-turbo 与单纯的 GPT-4 进行对比，而不是与 GPT-4-turbo 对比，认为后者才是更合适的基准参照。
- **解密 KPU 背后的技术**：对于 KPU 的实际技术存在一些困惑和调侃，它似乎结合了自我评估和“巧妙的 context window 技巧”，而不是一个全新的模型。
- **对实用性和性能的担忧**：一位成员质疑 KPU 在 MATH 测试上 6% 的提升是否具有实际意义，因为未报告的延迟可能会对产品集成产生负面影响。
- **CEO 解释 KPU**：Maisa 的 CEO 通过 [@davipar 的推文](https://x.com/davipar/status/1768683151780683919?s=20) 澄清，KPU 不是一个新模型，而是一个与现有 LLM 协同工作的架构，旨在优化知识管理，并通过“虚拟 context window”承诺降低成本并提高性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://maisa.ai/blog/kpu">KPU - Maisa</a>：AI 驱动的知识处理平台。一个用于执行业务任务的简单 API。为软件和应用程序开发人员抽象了使用最新 AI 架构的复杂性。</li><li><a href="https://x.com/davipar/status/1768683151780683919?s=20">David Villalón (@davipar) 的推文</a>：很高兴回答！它不是一个新模型，事实上 KPU 与智能提供商（OpenAI, Anthropic...）无关。它是一种与 LLM 协作的新 AI 架构，利用了它们的推理能力...
</li>
</ul>

</div>
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 条消息): 

res6969: https://x.com/leopoldasch/status/1768868127138549841?s=46
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1218132499150934157)** (21 条消息🔥): 

- **多种模型的德语响应生成困难**：一位用户在使用经过指令微调的 **DiscoLM-mixtral-8x7b-v2** 模型生成德语响应时遇到困难，而其他多个模型表现尚可。相关问题是在尝试使用 AutoModel 进行序列分类时出现 `ValueError` 异常，这可能暗示了未识别或不支持的配置类。
  
- **关于 Grok 的协助**：分享了 Grok 模型的 GitHub 链接（[Grok 开源发布](https://github.com/xai-org/grok/blob/main/model.py)），用户讨论了运行该模型的可行性，因为其庞大的规模（3140 亿参数需要大量的计算资源）。

- **德语语言模型的挑战与方法**：用户讨论揭示了关于合并德语语言模型、微调数据集质量以及使用一致的 Prompt 格式以维持语言输出质量的见解。对话强调了在合并模型时保持语言质量的挑战，以及社区协作改进德语语言模型的前景。

- **多语言和德语模型的基准测试**：提到了各种基准测试和伪装成基准测试的项目，如 supergleber-german-language-evaluation-benchmark，并附有论文和 Reddit 帖子的链接以获取更多细节。贡献者讨论了将德语特定基准测试添加到 EleutherAI 的 lm-evaluation-harness 等平台的潜力，以及需要衡量母语人士感知的语言质量的基准测试。

- **利用大学进行语言质量研究**：有人建议利用大学资源来研究和开发评估语言质量的基准测试，并指出公共资助的德国大学可以支持此类倡议。这是在 *DiscoLM* 项目的背景下提到的，强调了学术合作的潜在益处。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_didnt_found_a_better_small_german_llm_anyone/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bfce18/still_did">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://github.com/xai-org/grok/blob/main/model.py">grok-1/model.py at main · xai-org/grok-1</a>：Grok 开源发布。通过在 GitHub 上创建账户，为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/xai-org/grok/blob/e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8/model.py#L294">grok-1/model.py at e50578b5f50e4c10c6e7cff31af1ef2bedb3beb8 · xai-org/grok-1</a>：Grok 开源发布。通过在 GitHub 上创建账户，为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://www.informatik.uni-wuerzburg.de/datascience/news/single/news/our-paper-supergleber-german-language-understanding-evaluation-benchmark-was-accepted-at-the-naacl-2024/">我们的论文 "SuperGLEBer: German Language Understanding Evaluation Benchmark" 被 NAACL 2024 接收</a>：在我们的论文中，我们为德语构建了一个广泛的 Natural Language Understanding 基准测试套件，并随后评估了大量现有的具备德语能力的模型，以创建一个...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1b5vp2e/llm_comparisontest_17_new_models_64_total_ranked/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/ChuckMcSneed/WolframRavenwolfs_benchmark_results">ChuckMcSneed/WolframRavenwolfs_benchmark_results · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/KLUE-benchmark/KLUE">GitHub - KLUE-benchmark/KLUE: 📖 韩语 NLU 基准测试</a>：📖 韩语 NLU 基准测试。通过在 GitHub 上创建账户，为 KLUE-benchmark/KLUE 的开发做出贡献。</li><li><a href="https://github.com/facebookresearch/belebele">GitHub - facebookresearch/belebele: Belebele 数据集仓库，一个大规模多语言阅读理解数据集。</a>：Belebele 数据集仓库，一个大规模多语言阅读理解数据集。 - facebookresearch/belebele</li><li><a href="https://github.com/google-research/xtreme">GitHub - google-research/xtreme: XTREME 是一个用于评估预训练多语言模型跨语言泛化能力的基准测试，涵盖了 40 种不同类型的语言，并包含九个任务。</a>：XTREME 是一个用于评估预训练多语言模型跨语言泛化能力的基准测试，涵盖了 40 种不同类型的语言，并包含九个任务。 - goo...
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1218111377495949322)** (4 条消息): 

- **Demo 无需特殊设置**：*jp1* 澄清说，对于 Demo，通常不需要特殊设置或调整，目前他们默认使用 **fastchat/VLLM**。

- **Demo 服务器已迁移**：*jp1* 告知，用于 Demo 的服务器已从个人厨房环境迁移到更正式的地点。然而，出现了一些意外的网络问题，他们希望在下周初解决。

- **专业托管的弊端**：*chromix* 幽默地对比了他厨房角落里的业余服务器与专业托管服务器的可靠性，后者似乎遇到了各种技术问题，包括网络问题和自发性的 SAN 故障。
  

---



**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1218229369680695428)** (20 条消息🔥): 

- **Prodigy 推出 Prompt Engineering 工具**：一位前 Explosion 员工强调，他们开发的一些 Prompt Engineering 工具现在已成为 Prodigy 付费产品的一部分。该工具旨在将 Prompt Engineering 转化为数据标注问题，可以在 [Prodigy 的功能页面](https://prodi.gy/features/prompt-engineering)上查看。

- **开源工具让 Prompt 测试变得更简单**：成员们分享了用于 Prompt 测试和实验的各种资源，包括 [hegelai 的 PromptTools](https://github.com/hegelai/prompttools) 和 [PromptFoo](https://github.com/promptfoo/promptfoo) 仓库，这些工具支持一系列 LLM 和向量数据库。

- **用于模型比较和 Prompt 管理的 Vercel 和 Helicone.ai**：Vercel [AI Playground](https://sdk.vercel.ai/) 被提及为一个通过单个 Prompt 比较模型的有用界面，而 Helicone.ai 因其在 Prompt 管理和版本控制方面初露头角的能力而受到认可。

- **实验 AI 增强的博客定制化**：一位成员正在试点一个项目，使用 GPT-3.5-turbo 将博客文章“翻译”成各种角色 (personas)，暗示了提升写作清晰度和重点的潜在工具，并分享了一个实时示例的链接：[How to Build a Buzzword](https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html)。

- **关于 AI 增强操作博客的讨论**：交流了关于 AI 如何丰富博客平台的想法，建议的功能包括从不同角色进行重写、提供对立观点、提供基于角色的社交分享，以及生成摘要或翻译。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.dbreunig.com/2020/02/28/how-to-build-a-buzzword.html">How to Build a Buzzword</a>：以及为什么它们如此强大</li><li><a href="https://www.helicone.ai/">Helicone</a>：开发者如何构建 AI 应用程序。开箱即用即可获得可观测性、工具链、微调和评估。</li><li><a href="https://sdk.vercel.ai/">Vercel AI SDK</a>：使用最新的 AI 语言模型构建 AI 驱动的应用程序</li><li><a href="https://github.com/hegelai/prompttools">GitHub - hegelai/prompttools: 用于 Prompt 测试和实验的开源工具，支持 LLMs（如 OpenAI, LLaMA）和向量数据库（如 Chroma, Weaviate, LanceDB）。</a>：用于 Prompt 测试和实验的开源工具，支持 LLMs（如 OpenAI, LLaMA）和向量数据库（如 Chroma, Weaviate, LanceDB）。 - hegelai/prompttools</li><li><a href="https://github.com/promptfoo/promptfoo">GitHub - promptfoo/promptfoo: 测试你的 Prompt、模型、RAG。评估并对比 LLM 输出，捕获回归问题，并提高 Prompt 质量。适用于 OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, 本地及私有模型（如 Mistral/Mixtral/Llama）的 LLM 评估，支持 CI/CD</a>：测试你的 Prompt、模型、RAG。评估并对比 LLM 输出，捕获回归问题，并提高 Prompt 质量。LLM 评估支持 OpenAI/Azure GPT, Anthropic Claude, VertexAI Gemini, Ollama, 本地 &amp;...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/)** (1 条消息): 

obra: 是否有可能恢复 OpenAI 模型在之前的 API 请求中使用的 seed？
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1218193382669549568)** (17 条消息🔥): 

- **待发布的模型改进方法**：一位成员表示他们正在整理一种方法的实验结果，该方法似乎能提高**全局准确率 (global accuracy)** 并使训练**更具样本效率 (sample efficient)**。他们承诺在准备好更好的图表和结构化结果后发布论文/文章。
- **寻求扩展到大模型的资源**：讨论显示，虽然已经进行了一些验证，但由于资源限制，缺乏该方法在大规模模型上有效性的经验证明。该成员表示需要资源来进行这一验证。
- **提议讨论并扩展该方法**：有人提议进行通话以讨论上述方法，并可能帮助分配**算力 (compute) 和资源**来扩展该方法。
- **子集实验中改进明显**：该成员提到，他们的方法在 CIFAR100 的子集上配合 VGG16 训练 1 个 epoch 时获得了**更高的测试准确率**，并引用了具体的准确率数字来强调改进。
- **探索改进图表报告的方法**：有关于在 Wandb（用于报告实验结果的平台）上更新图表问题的评论，特别是如何在绘制新数据时重置步数 (steps)。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=ZlJbaYQ2hm4