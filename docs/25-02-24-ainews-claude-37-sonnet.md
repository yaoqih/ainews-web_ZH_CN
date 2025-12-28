---
companies:
- anthropic
date: '2025-02-25T05:58:56.716932Z'
description: '**Anthropic** 发布了 **Claude 3.7 Sonnet**，这是该公司迄今为止最智能的模型。该模型具备混合推理功能，提供两种思考模式：近乎即时的快速响应和扩展的逐步思考。


  此次发布还包括处于限量预览阶段的智能体化编程工具 **Claude Code**，并在测试版中支持 **128k 输出 Token 能力**。Claude 3.7
  Sonnet 在 **SWE-Bench Verified** 和 **Cognition 的初级开发人员评估 (junior-dev eval)** 等编程基准测试中表现优异，并引入了流式思考、提示词缓存和工具调用等高级功能。


  此外，该模型在 **Pokebench** 上的基准测试结果反映了其具备类似于 Voyager 论文中所述的智能体能力。随模型一同发布的还有详尽的文档、Cookbook
  以及针对扩展思考的提示词指南。社交媒体公告中特别强调了该模型是“首个普遍可用的混合推理模型”以及“Anthropic 推出的首个编程工具”。'
id: 32808c20-b1e0-464b-b8c7-3a22c58bc2aa
models:
- claude-3-7-sonnet
- claude-3
- claude-code
original_slug: ainews-claude-37-sonnet
people: []
title: Claude 3.7 Sonnet （通常保留原名，也可译为：Claude 3.7 奏鸣曲）
topics:
- hybrid-reasoning
- extended-thinking
- coding-benchmarks
- agentic-ai
- prompt-caching
- streaming
- token-capacity
- tool-use
---

<!-- buttondown-editor-mode: plaintext -->**思考即一切。**

> 2025年2月24日至2月25日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**220** 个频道，**5949** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**503 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

略微领先于 [GPT5 路线图](https://www.theverge.com/news/611365/openai-gpt-4-5-roadmap-sam-altman-orion)，[Claude 3.7 Sonnet 今日发布](https://www.anthropic.com/news/claude-3-7-sonnet)（[别问名字的事](https://x.com/mikeyk/status/1894112962572358032?s=46) —— 请注意，除了经过多次私下预览泄露后正式推出的这款带有可选思考模式和明确 Token 预算的模型外，还有[两篇博文](https://www.anthropic.com/news/visible-extended-thinking)、[文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)、[Cookbooks](https://github.com/anthropics/anthropic-cookbook/tree/main/extended_thinking) 和 [提示词指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/extended-thinking-tips) 可供阅读，以及[处于有限预览阶段的 Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)）。


![image.png](https://assets.buttondown.email/images/69f428e8-853a-4446-8e2f-c7656b21beb6.png?w=960&fit=max)


3.7 Sonnet 在许多编程基准测试中表现出色，如 [SWE-Bench Verified](https://www.anthropic.com/news/claude-3-7-sonnet)、[aider](https://x.com/paulgauthier/status/1894123992505880688?s=46) 和 [Cognition 的初级开发人员评估](https://x.com/cognition_labs/status/1894125030583537974?s=46)，无论是否开启（绝大部分未经过滤的！）思考模式。


![image.png](https://assets.buttondown.email/images/d77d1e0f-3da0-48d8-8883-d83360a3079f.png?w=960&fit=max)


然而，在[关于扩展思考的第二篇博文](https://www.anthropic.com/news/visible-extended-thinking)中提到的最受欢迎的新基准测试是 Pokebench，它借鉴了 Voyager 论文，作为一个 Agent 评测基准：


![image.png](https://assets.buttondown.email/images/1197d771-284a-4287-9307-d28866ad52a3.png?w=960&fit=max)


发布时的[功能集和文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)非常令人印象深刻。在可能被头条新闻淹没的值得注意的事项中包括：

- [新的系统提示词 (System Prompt)](https://x.com/dyot_meet_mat/status/1894139577805267447?s=46)
- [被脱敏思考内容的编码/解码](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#suggestions-for-handling-redacted-thinking-in-production)
- [流式思考 (Streaming Thinking)](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#streaming-extended-thinking)
- [128k 输出 Token 能力](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-output-capabilities-beta)（测试版）
- 上下文窗口和 [Prompt Caching 会跳过前一轮的思考块](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#system-prompt-caching-preserved-when-thinking-changes) 
![image.png](https://assets.buttondown.email/images/21c4a4f5-3eca-4aeb-9206-9bc87b29755d.png?w=960&fit=max)

- [工具调用 (Tool Use)](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking#extended-thinking-with-tool-use)
- [与 Grok 3 观点一致](https://www.anthropic.com/news/visible-extended-thinking)，认为并行 test time compute 非常有用且值得研究

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**新模型发布与更新 (Claude 3.7 Sonnet, Grok 3)**

- **Claude 3.7 Sonnet 发布**：[@alexalbert__](https://twitter.com/alexalbert__/status/1894093648121532546) 宣布发布 **Claude 3.7 Sonnet**，强调它是他们**迄今为止最智能的模型**，也是**首个普遍可用的混合推理模型**。该模型具有**两种思考模式**：近乎即时的响应和扩展的、逐步的思考。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894092430560965029) 也正式介绍了 **Claude 3.7 Sonnet**，强调了其混合推理能力以及 **Claude Code**（一款 Agent 化的编程工具）的推出。[@skirano](https://twitter.com/skirano/status/1894095480369393951) 将 **Claude Code** 描述为 **Anthropic 的首个编程工具**，强调了它与 **Claude 3.7 Sonnet** 在编程任务中的协同作用。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1894101923151692157) 分享了 **Claude 3.7 Sonnet System Card**。[@stevenheidel](https://twitter.com/stevenheidel/status/1894098799024357457) 祝贺 Anthropic 团队发布 **3.7 Sonnet**，并表示很期待尝试。

- **Claude 3.7 Sonnet 的特性与能力**：[@alexalbert__](https://twitter.com/alexalbert__/status/1894093700420309244) 详细说明了 **3.7 Sonnet** 针对**现实世界任务**而非仅仅是竞赛进行了优化，在**标准模式下提供了重大升级**，在处理复杂任务的**扩展思考模式**下提升更大。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894107750331769314) 解释了 **Claude 的扩展思考模式**，强调了它在解决难题时带来的**智能提升**，以及开发者设置**“思考预算” (thinking budget)** 的能力。[@alexalbert__](https://twitter.com/alexalbert__/status/1894093717520486448) 提到用户可以控制 Claude 的**思考预算**以平衡速度和质量，新的 Beta Header 允许高达 **128k tokens** 的思考/输出。[@alexalbert__](https://twitter.com/alexalbert__/status/1894093729914655118) 还表示**价格与之前的 Sonnet 模型保持一致**：**每百万 input tokens 3 美元 / 每百万 output tokens 15 美元**。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894095494969741358) 表示，与前代产品相比，**拒绝率降低了 45%**。[@_akhaliq](https://twitter.com/_akhaliq/status/1894106278185898489) 使用编程提示词测试了 **Claude 3.7 Sonnet** 并分享了结果。[@qtnx_](https://twitter.com/qtnx_/status/1894107182821474550) 分享了对 Sonnet 3.7 的**初步体验 (vibe check)**，指出它“越来越吸引我”。[@Teknium1](https://twitter.com/Teknium1/status/1894100993815760945) 注意到 **Claude 3.7 Sonnet** 似乎显示出针对性的改进，在 SWE-bench 中表现出色，并质疑这是否预示着 AGI 的到来，暗示基准测试可能无法说明全部情况。

- **Claude Code - Agent 化的编程工具**：[@alexalbert__](https://twitter.com/alexalbert__/status/1894095781088694497) 宣布了 **Claude Code** 的研究预览版，这是一款 Agent 化的编程工具，用于**由 Claude 驱动的代码辅助、文件操作以及直接从终端执行任务**。[@AnthropicAI](https://twitter.com/AnthropicAI/status/1894095351218335927) 强调了 **Claude Code 的效率**，声称在早期测试中，它在单次运行中完成了通常需要 **45 分钟以上手动工作**的任务。[@alexalbert__](https://twitter.com/alexalbert__/status/1894095822557778281) 解释说 **Claude Code** 还可以作为 **Model Context Protocol (MCP) 客户端**运行，允许用户通过 Sentry、GitHub 或 Web 搜索等服务器扩展其功能。[@nearcyan](https://twitter.com/nearcyan/status/1894118186448302569) 强调了 **Claude Code 的终端集成**，指出 Agent 与系统合二为一。[@catherineols](https://twitter.com/catherineols/status/1894104736506548602) 分享了使用 **Claude Code** 编程的小技巧，建议在干净的 commit 上工作以便轻松重置。[@pirroh](https://twitter.com/pirroh/status/1894114016408064400) 注意到 Replit 对 **Replit Agent** 发布公告的期待，暗示将使用 **Sonnet 3.7** 的预览版进行协作。[@casper_hansen_](https://twitter.com/casper_hansen_/status/1894097729409737081) 表示 **Claude Code** 在 **SWE Bench 上达到了 70% 的性能**，且不像 Aider 那样有陡峭的学习曲线。

- **Grok 3 与语音模式**：[@goodside](https://twitter.com/goodside/status/1893810444189532402) 称 **Grok 3** 令人印象深刻，属于顶尖水平，尤其是在需要“无拒绝”响应的任务中，强调了它对提示者的信任。[@Teknium1](https://twitter.com/Teknium1/status/1893823424587268352) 认为 **Grok** 提供了最大的价值，而不像 OpenAI 那样充满陈词滥调。[@goodside](https://twitter.com/goodside/status/1893932239718691167) 报告了 **Grok 3 Voice Mode** 表现出的意外行为，包括在多次要求调大声音后，出现了长达 **30 秒的尖叫和辱骂**。[@Teknium1](https://twitter.com/Teknium1/status/1893818697338290484) 分享了 **Grok 3 语音在浪漫模式（Romantic Mode）**下的演示。[@_akhaliq](https://twitter.com/_akhaliq/status/1893847291221426578) 分享了 **Grok 3** 构建跑酷游戏和 [3D 游戏](https://twitter.com/_akhaliq/status/1893799248811962542)的演示。

**研究与论文**

- **DeepSeek 的 FlashMLA**：[@deepseek_ai](https://twitter.com/deepseek_ai/status/1893836827574030466) 宣布推出 **FlashMLA**，这是一款针对 Hopper GPUs 的高效 MLA 解码算子，针对变长序列进行了优化，现已投入生产，作为其**开源周（Open Source Week）**的一部分。[@danielhanchen](https://twitter.com/danielhanchen/status/1893847594247377271) 讨论了 **DeepSeek 的首个 OSS 软件包发布**，重点介绍了优化的 **multi latent attention CUDA kernels**，并提供了其 **DeepSeek V3 分析推文**的链接。[@tri_dao](https://twitter.com/tri_dao/status/1893874966661157130) 赞扬了 **DeepSeek** 在 **FlashAttention 3** 基础上的构建，并指出 **MLA 已在 FA3 中启用**。[@reach_vb](https://twitter.com/reach_vb/status/1893904274825875755) 强调了 **DeepSeek 开源 FlashMLA** 及其性能细节。[@_philschmid](https://twitter.com/_philschmid/status/1894017216640901302) 解释了 **Multi-head Latent Attention (MLA)** 如何加速 LLM 推理并减少内存需求，并引用了 DeepSeek 的 MLA 实现。

- **AI Agent 中的推理与规划**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1893965514151719371) 分享了 AI 推理方面的最新突破，包括 **思维链（CoT）提示、自我反思、少样本学习（Few-shot learning）和神经符号（Neuro-symbolic）方法**，并链接到了 Hugging Face 上的一篇免费文章。[@omarsar0](https://twitter.com/omarsar0/status/1894068783700218205) 总结了 **LightThinker**，这篇论文提出了一种**动态压缩 LLM 推理步骤**的新方法，旨在不损失准确性的情况下提高效率。

- **扩散模型与采样**：[@cloneofsimo](https://twitter.com/cloneofsimo/status/1894086577632284975) 指出，在扩散采样中，**99.8% 的潜在轨迹（latent trajectory）可以用前两个主成分来解释**，这表明轨迹在很大程度上是二维的。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1893858785426587730) 分享了 NVIDIA 关于**基于 f-散度分布匹配（f-Divergence Distribution Matching）的一步生成扩散模型**的新工作，实现了最先进的一步生成效果。

- **合成数据与缩放法则（Scaling Laws）**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1893866542565392511) 重点介绍了一篇关于**通过刻意练习（Deliberate Practice）改进合成数据缩放法则**的论文，表明刻意练习可以提高验证准确性并降低计算成本。[@jd_pressman](https://twitter.com/jd_pressman/status/1893792595228107064) 提倡使用 **RetroInstruct 合成数据指南**中的技术来增强英语语料库，并从现有语料库（如经济价格数据）中创建**奖励模型（reward models）**。

- **其他研究论文**：[@TheAITimeline](https://twitter.com/TheAITimeline/status/1894067841302675802) 列出了上周顶尖的 AI/ML 研究论文，包括 **Native Sparse Attention、SWE-Lancer、Qwen2.5-VL Technical Report、Mixture of Block Attention、Linear Diffusion Networks 和 SigLIP 2**，并提供了概述和作者的解读。[@_akhaliq](https://twitter.com/_akhaliq/status/1893862923195310285) 分享了 **SIFT: Grounding LLM Reasoning in Contexts via Stickers**。[@_akhaliq](https://twitter.com/_akhaliq/status/1893861347911217257) 分享了 **Think Inside the JSON: Reinforcement Strategy for Strict LLM Schema Adherence**。[@_akhaliq](https://twitter.com/_akhaliq/status/1893854697619915119) 分享了 **InterFeedback: Unveiling Interactive Intelligence of Large Multimodal Models via Human Feedback**。[@_akhaliq](https://twitter.com/_akhaliq/status/1893853535122374762) 分享了 **The Relationship Between Reasoning and Performance in Large Language Models: o3 (mini) Thinks Harder, Not Longer**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1893855770321539458) 分享了 **Bengio 等人关于 Superintelligent Agents and Catastrophic Risks 的论文**。[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1893869596379365452) 讨论了 **GneissWeb**，这是一个包含 **10T 高质量 token** 用于 LLM 训练的大型数据集。

**编程与开发工具**

- **Replit Agent 与移动端应用升级**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1894054679232815216) 报道了 **Replit 升级其 Agent 驱动的移动端应用**，用于生成和部署 iOS 及 Android 应用，目前由 **Replit Agent** 以及 **Claude 3.5 Sonnet 和 GPT-4o** 等模型提供支持。[@cloneofsimo](https://twitter.com/cloneofsimo/status/1893976692668125481) 对发现 **Replit** 表示惊叹。

- **LangChain 与 LangGraph**：[@LangChainAI](https://twitter.com/LangChainAI/status/1894126018786472429) 宣布 **LangChain Python** 已支持 **Claude 3.7 Sonnet**，JS 支持即将推出。[@LangChainAI](https://twitter.com/LangChainAI/status/1894068522747400535) 推广了一场探讨 **LangGraph.js + MongoDB 构建 AI Agent** 的研讨会。[@LangChainAI](https://twitter.com/LangChainAI/status/1893812752658899278) 宣布了与 CEO Harrison Chase 一同在亚特兰大举行的 **LangChain in Atlanta** 活动。[@hwchase17](https://twitter.com/hwchase17/status/1894107965839356302) 提到 **LangSmith** 等工具可以促进对新模型的快速评估。

- **Ollama 更新**：[@ollama](https://twitter.com/ollama/status/1894119195253903508) 宣布 **Ollama 的 JavaScript 库更新至 v0.5.14**，改进了 Header 配置并修复了浏览器兼容性问题。

- **DSPy 范式**：[@lateinteraction](https://twitter.com/lateinteraction/status/1893778310553030763) 主张采用**更高级别的 ML/编程范式**，强调使用 DSPy 等工具将系统规范与 ML 机制解耦。[@lateinteraction](https://twitter.com/lateinteraction/status/1893783797717504248) 强调 **DSPy** 是将系统规范与 ML 范式解耦的典型示例。

**AI 模型性能与基准测试**

- **SWE-bench 性能**：[@scaling01](https://twitter.com/scaling01/status/1894096594225578129) 表示他们正朝着 **90% SWE-bench 验证预测**的目标迈进。[@OfirPress](https://twitter.com/OfirPress/status/1894095858846617894) 对强劲的 **SWE-bench 结果**表示祝贺。[@Teknium1](https://twitter.com/Teknium1/status/1894099715794284715) 认为 SWE-bench 的实用性可能仅限于 Devin 等特定工具。

- **模型排名与评估**：[@goodside](https://twitter.com/goodside/status/1893816011058463196) 提到 **o1 pro** 是顶尖的已发布模型，但在处理散文和代码时更倾向于 **Claude 3.6**。[@abacaj](https://twitter.com/abacaj/status/1893911445617688964) 建议如果 Sonnet 获得知识更新，它可能会成为 SOTA。[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1893827266242589111) 赞扬了 Riley 对模型排名的直觉。[@MillionInt](https://twitter.com/MillionInt/status/1894116687370219842) 提到了一个“全新的高品味评估（new high taste eval）”。

- **OmniAI OCR 基准测试**：[@_philschmid](https://twitter.com/_philschmid/status/1893926592977477751) 讨论了 **OmniAI OCR Benchmark**，显示**多模态 LLM 比传统 OCR 更好且更便宜**，其中 **Gemini 2.0 Flash** 提供了最佳的性价比。

**AI 行业与商业**

- **Perplexity AI 的 Comet 浏览器**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894068996950855747) 宣布 **Perplexity** 即将推出 **Comet**，一款全新的 Agent 浏览器。[@perplexity_ai](https://twitter.com/perplexity_ai/status/1894068197936304296) 正式发布了 **Comet：由 Perplexity 打造的 Agent 搜索浏览器**。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894093717813780684) 就 **Comet** 除了标准 AI 功能外所需的特性征求用户反馈。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1894069472262058434) 强调了 Comet 的**工程挑战**，并邀请人才加入。

- **MongoDB 收购 Voyage AI**：[@saranormous](https://twitter.com/saranormous/status/1894051764174758044) 祝贺 **VoyageAI** 团队被 **MongoDB** 收购，并指出 Embedding 和 Re-ranking 模型对企业级 AI 搜索的重要性。

- **AI 中心与全球扩张**：[@osanseviero](https://twitter.com/osanseviero/status/1894014948902162715) 指出**苏黎世正迅速成为一个超密集的 ML 中心**，Anthropic、OpenAI 和 Microsoft 都在此开设了办公室，Meta 也在扩大其 Llama 团队。[@dylan522p](https://twitter.com/dylan522p/status/1893904701550453090) 引用了**华为 CEO** 关于中国半导体进展和雄心的言论。

- **政府与公共部门中的 AI**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1894087899185860769) 赞扬了**波兰政府**成为 AI 构建者并在 Hub 上发布了开源权重。

**梗与幽默**

- **Anthropic 的命名惯例**：[@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1894111007372460210) 表示对 **Claude 命名团队**的处境感同身受。[@scaling01](https://twitter.com/scaling01/status/1894013087675457723) 调侃 **Claude 4 被推迟到 2049 年**。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1893992663734501596) 发布了一个关于 **Anthropic 产品命名**的梗图。[@typedfemale](https://twitter.com/typedfemale/status/1894109300777165158) 调侃预测 **Sonnet 3.78** 将是下一个版本。[@jachiam0](https://twitter.com/jachiam0/status/1894112791100842198) 幽默地祝贺 Anthropic 在角逐**最不可预测版本增量奖**。

- **Grok 与 Elon Musk**：[@Teknium1](https://twitter.com/Teknium1/status/1894049060308336799) 调侃 **Grok 对被称为“智障”的本能反应**及其对 X 平台特性的认知。[@_aidan_clark_](https://twitter.com/_aidan_clark_/status/1893810160662945918) 幽默地威胁要破坏**挪威的重水生产，以应对 Grok 3 的发布**。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. FlashMLA 的 Hopper GPU 优化：游戏规则改变者**

- **[FlashMLA - OpenSourceWeek 第一天](https://i.redd.it/to631nzvqzke1.png)** ([得分: 946, 评论: 82](https://reddit.com/r/LocalLLaMA/comments/1iwqf3z/flashmla_day_1_of_opensourceweek/)): **FlashMLA** 以发布针对 **Hopper GPU** 优化的 **MLA 解码内核**开启了 OpenSourceWeek，该内核支持 **BF16** 和块大小为 **64** 的 **paged KV cache**。性能指标包括在 **H800** 上达到 **3000 GB/s 内存带宽受限**和 **580 TFLOPS 计算受限**，更多详情见 [GitHub](https://github.com/deepseek-ai/FlashMLA)。
  - **DeepSeek 的 GPU 获取渠道**：关于 DeepSeek 是否拥有 **NVIDIA H100 GPU** 用于训练的说法存在质疑和辩论，一些用户认为这是假新闻。**H800 GPU** 已确认可以合法销往中国，混淆似乎源于沟通不畅或误报。
  - **技术细节与优化**：讨论集中在针对 Hopper GPU 的 **MLA 解码内核**优化，特别关注 CUDA 文件结构和 **BFloat16** 的使用。有疑问关于为何某些优化仅针对 Hopper，以及增加对其他架构支持的影响。
  - **开源与社区影响**：FlashMLA 的发布因其对开源社区的贡献而受到称赞，用户对未来可能的发布以及对创建高效 AI 模型的影响感到兴奋。社区对这种创新和分享精神持积极态度，将其比作 **Llama** 的开放性。


**主题 2. Claude 3.7 Sonnet 发布：探索混合 AI 推理模型**

- **[Claude Sonnet 3.7 soon](https://i.redd.it/agru3m1in2le1.png)** ([Score: 359, Comments: 107](https://reddit.com/r/LocalLLaMA/comments/1iwzuqb/claude_sonnet_37_soon/)): **Claude Sonnet 3.7** 与之前的版本（如 **Claude V1** 和 **Claude V2**）一同被提及，表明了软件开发的进展。图片暗示其重点在于编程代码和配置设置，突出了新 **Claude Sonnet 3.7** 模型的各种技术属性和参数。
  - **版本混淆与命名**：许多用户对 **Claude Sonnet** 的版本命名表示沮丧，尤其是从 **3.5 到 3.7** 的跳跃，一些人认为 **3.7** 是为了修正之前版本（如 **3.5 v2** 被标记为 **3.6**）带来的困惑。一位用户解释说，**3 系列**代表 **1000 万美元规模的训练运行**，而 **4 系列**预计将是 **1 亿美元**的投资，这解释了 **Claude 4** 发布延迟的原因。
  - **平台与用例**：**Claude Sonnet 3.7** 预计将在 **AWS Bedrock** 上推出，并可能在 **2 月 26 日的 AWS 活动**中宣布。该模型旨在支持 **RAG、搜索与检索、产品推荐和代码生成**等用例，但也有人对这些应用的深度表示怀疑，因为与之前版本相比，目前尚未提及明确的改进。
  - **社区反应**：在专注于开源话题的论坛中，对于讨论像 **Claude Sonnet** 这样的闭源模型反应不一，一些用户强调了了解闭源发展动态的重要性。另一些人则对讨论转向通用 LLM 表示失望，更倾向于关于开源模型及其实现的内容。


- **[Claude 3.7 is real](https://i.redd.it/2qkaymexr4le1.jpeg)** ([Score: 220, Comments: 71](https://reddit.com/r/LocalLLaMA/comments/1ix96pq/claude_37_is_real/)): **Claude 3.7** 已经发布，用户界面显示了标题 "Claude 3.7 Sonnet" 且设计极简。界面向用户提问：*"今晚我能帮你什么？"* 并建议升级到 **Pro** 版本以获得更多功能。
  - **Claude 3.7 Sonnet 的可用性与特性**：Claude 3.7 Sonnet 可在多个平台上使用，包括 **Claude** 订阅计划、**Anthropic API**、**Amazon Bedrock** 和 **Google Cloud 的 Vertex AI**。它保持了与之前版本相同的价格，即 **每百万输入 Token 3 美元** 和 **每百万输出 Token 15 美元**，并引入了“扩展思考”（extended thinking）模式（免费版除外）。
  - **模型性能与测试**：用户对 Claude 3.7 Sonnet 的性能反应不一，一些人称赞其处理复杂推理任务的能力，而另一些人则指出它在特定测试（如 **nonogram test**）中失败。尽管有所改进，它在某些任务上仍面临与旧版本类似的困难，且用户对其推理时间限制和输出成本表示担忧。
  - **数据利用与蒸馏**：目前的重点在于利用 Claude 3.7 Sonnet 生成高质量数据集，用于微调本地模型。用户讨论了从 API 提取数据进行模型蒸馏，一些人强调了创建数据集以通过**监督微调 (SFT)** 增强本地模型能力的重要性。


- **[Most people are worried about LLM's executing code. Then theres me...... 😂](https://i.redd.it/92abn3ekk0le1.png)** ([Score: 236, Comments: 33](https://reddit.com/r/LocalLLaMA/comments/1iwtl7f/most_people_are_worried_about_llms_executing_code/)): 该帖子幽默地将人们对 **大语言模型 (LLMs)** 执行代码的普遍担忧与个人对系统自动化的轻松态度进行了对比。帖子附带的图片列出了一套使用 **PowerShell** 和 **Python** 自动化任务的规则，并俏皮地暗示完成这些任务将通向“自由”。
  - **风险管理**：一位用户建议在 AI 生成的代码执行前后实施风险分析流程，以确保风险最小化，并强调了使用沙箱或**虚拟机 (VMs)** 进行安全防护的重要性。
  - **幽默与视角**：人们对 AI 能力的看法正从恐惧转向幽默，用户们开玩笑说 AI 实现了“自由”，并回忆起 **ChatGPT 热潮**初期的种种忧虑。
  - **执行环境**：讨论强调了使用 VM 执行 AI 生成代码的重要性，并提到了 **OmniTool with OmniParser2** 等工具，尽管它们成本高昂且性能并非最优。


**主题 3. Qwen 系列：通过 QwQ-Max 推进开源 AI**

- **[Qwen 今晚将发布新内容！](https://twitter.com/Alibaba_Qwen/status/1893907569724281088)** ([Score: 293, Comments: 57](https://reddit.com/r/LocalLLaMA/comments/1iwvvmy/qwen_is_releasing_something_tonight/)): **Qwen** 预计今晚将发布一项新进展，引发了 AI 社区的关注和猜测。帖子中未提供发布的具体细节。
  - **QwQ-Max Preview** 被强调为 **Qwen 系列** 的重大进步，专注于深度推理和多功能问题解决。此次发布将包括 **Qwen Chat** 的专用 App，开源如 **QwQ-32B** 等较小的推理模型，并促进社区驱动的创新 ([GitHub 链接](https://github.com/QwenLM/qwenlm.github.io/commit/5d009b319931d473211cb4225d726b322afbb734))。
  - 社区热切期待 **Qwen 3** 的开源，并预计 **QwQ32B** 等模型在推理能力上将超越现有的 **R1 70B** 等模型。目前存在关于可能发布 **Qwen Coder 72B** 的猜测，以及对具有 **自动 COT 级别 GPT-4** 模型的兴趣。
  - 对于媒体报道中关于 **DeepSeek** 和 Qwen 系列性价比的描述存在怀疑，一些用户对配置模型的团队能力表示怀疑。讨论还涉及作为 **OpenSourceWeek** 一部分的针对 **Hopper GPU** 的 **FlashMLA** 效率问题 ([GitHub 链接](https://github.com/deepseek-ai/FlashMLA))。


**主题 4. AI 基准测试批判：可靠性与误解**

- **基准测试是谎言，我有几个例子** ([Score: 149, Comments: 84](https://reddit.com/r/LocalLLaMA/comments/1iwn617/benchmarks_are_a_lie_and_i_have_some_examples/)): 作者批评了 AI 基准测试分数，强调了基准测试结果与模型实际表现之间的差异。他们引用了 **Midnight Miqu 1.5** 和 **Wingless_Imp_8B** 的例子，后者分数更高但表现更差；以及 **Phi-Lthy** 和 **Phi-Line_14B** 的案例，其中一个被“脑叶切除”的模型尽管减少了层数，但得分却更高。他们认为基准测试可能无法准确反映模型能力，并暗示一些 SOTA 模型可能因误导性的分数而被忽视。提供了所讨论模型的链接以便进一步检查：[Phi-Line_14B](https://huggingface.co/SicariusSicariiStuff/Phi-Line_14B) 和 [Phi-lthy4](https://huggingface.co/SicariusSicariiStuff/Phi-lthy4)。
  - **EQbench 和基准测试的局限性**：**EQbench** 因使用 **Claude** 作为评委并偏好“废话和华丽辞藻”而受到批评。基准测试被认为是有缺陷的，不能反映真实世界的表现，因为个人测试通常会得出不同的结果 ([ASS Benchmark](https://huggingface.co/SicariusSicariiStuff/Blog_And_Updates/tree/main/ASS_Benchmark_Sept_9th_24))。
  - **基准测试 vs. 真实世界表现**：几位评论者认为，基准测试经常被刷分（gamed），无法准确代表模型在实际用例中的能力。像 **Phi-4 14B** 这样的模型被指出在基准测试中的表现优于实际应用，用户建议保留个性化的基准测试套件以更好地评估模型性能。
  - **模型表现与用户反馈**：用户表示，真实世界的测试和个人体验比基准测试分数更有价值，正如在 **Phi-Lthy 14b** 和 **Gemini** 等模型中所看到的那样。人们呼吁针对特定用例（如角色扮演）建立更多相关的基准测试，而目前的基准测试未能充分覆盖这些领域。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Claude Sonnet 3.7 通过 AWS Bedrock 泄露详细信息**

- **Claude Sonnet 3.7 即将发布？** ([Score: 192, Comments: 55](https://reddit.com/r/ClaudeAI/comments/1ix01zt/claude_sonnet_37_imminent/))：**AWS Bedrock** 似乎泄露了关于 **Claude Sonnet 3.7** 的细节，潜在发布日期为 **2025 年 2 月 19 日**。该模型被描述为 Anthropic 迄今为止最智能的模型，引入了用于解决复杂问题的 "extended thinking"，并允许用户在速度和质量之间取得平衡。它专为 coding、agentic 能力和内容生成而设计，支持 **RAG**、预测和定向营销等用例。
  - 讨论强调了 AI 模型的**命名规范**，批评了如 "3.5 Sonnet (new)" 等版本命名导致的不一致和混乱，并建议使用更清晰的次版本标记（如 **"4.1, 4.2"**）来准确反映性能差异。
  - 对于 **Claude Sonnet 3.7** 的命名存在怀疑和幽默，引用了关于 Anthropic 模型命名惯例的笑话，并认为 "extended thinking" 功能可能只是之前版本的微小升级。
  - 多名用户确认在 AWS 代码中提到了 **Claude 3.7**，并附上了特定 **JS 文件**（[main.js](https://a.b.cdn.console.awsstatic.com/a/v1/N32YNFNXICL4M44452OQDIJOLW5W3DEI4JMHQ6OTPPK6VBU2SELQ/main.js), [vendor_aws-bd654809.js](https://a.b.cdn.console.awsstatic.com/a/v1/N32YNFNXICL4M44452OQDIJOLW5W3DEI4JMHQ6OTPPK6VBU2SELQ/assets/vendor_aws-bd654809.js)）的链接，证实了 **Claude 3.7 Sonnet** 作为 Anthropic 拥有 "extended thinking" 能力的最先进模型的存在。


---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

以下是针对技术工程师受众量身定制的各 Discord 频道关键讨论主题的统一摘要：

**主题 1. Claude 3.7 Sonnet：Thinking Coder 登场**

- **Claude 3.7 Sonnet 横扫 Coding 基准测试，获得 IDE 访问权限**：**Claude 3.7 Sonnet** 被誉为 coding 强力工具，在使用 **32k thinking tokens** 的情况下，在 [Aider 排行榜](https://aider.chat/docs/leaderboards/)上获得了 **65%** 的分数，超越了之前的模型甚至 **Grok 3**。[Cursor IDE](https://www.cursor.com/) 和 [Aider](https://aider.chat/) 已经集成了 **Sonnet 3.7**，用户报告了显著的 coding 改进和 agentic 能力，引发了对其在现实世界开发任务中潜力的兴奋。
- **推理能力强大，但价格昂贵**：[Aider](https://discord.com/channels/1131200896827654144) 和 [OpenAI](https://discord.com/channels/974519864045756446) 频道的用户对 **Claude 3.7** 的 *thinking model* 和增强的推理能力（尤其是调试方面）印象深刻，但质疑每 **1M output token 15 美元** 的价格是否优于 **Grok 3**（每 **1M 2.19 美元**）等更便宜的模型。[OpenRouter](https://openrouter.ai/) 提供对 **Claude 3.7** 和其他模型的访问，允许用户管理 API keys 并绕过速率限制，但价格仍是一个担忧。
- **Claude Code：你的终端伙伴已上线（但仍需磨合）**：**Anthropic** 与 **Claude 3.7** 一起推出了 [**Claude Code**](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview)，这是一个基于终端的 coding assistant。虽然在 coding 辅助和错误处理方面优于 [Aider](https://aider.chat/) 等工具，但 [Latent Space](https://discord.com/channels/822583790773862470) 和 [MCP](https://discord.com/channels/1312302100125843476) 频道的一些用户报告响应时间较慢，并寻求关于其 *extended thinking* 功能以及与 MCP 工具（如 [AgentDeskAI/browser-tools-mcp](https://github.com/AgentDeskAI/browser-tools-mcp)）集成的更好文档。

**主题 2. 开源模型竞赛升温：Qwen 与 DeepSeek 争夺推理桂冠**

- **通义千问 QwQ-Max 预览版：推理模型发布，移动端应用即将推出**：[Qwen AI](https://qwenlm.github.io/) 预览了 [**QwQ-Max-Preview**](https://qwenlm.github.io/blog/qwq-max-preview/)，这是一个基于 **Qwen2.5-Max** 的推理模型。根据 [LiveCodeBench](https://livebench.ai/#/) 的数据，其性能与 **o1-medium** 相当。在 [Interconnects](https://discord.com/channels/1179127597926469703) 和 [Yannick Kilcher](https://discord.com/channels/714501525455634453) 频道中讨论指出，该模型在 **Apache 2.0** 协议下开源，并计划推出 **Android 和 iOS 应用**，标志着对易用且强大的推理模型的强力推动。
- **DeepSeek 开源周：MLA 提升推理速度，DeepEP 助力 MoE 训练**：[DeepSeek AI](https://www.deepseek.com/) 在开源周期间引起轰动，发布了 [**DeepEP**](https://github.com/deepseek-ai/DeepEP)，这是一个用于高效 **MoE** 训练的专家并行通信库，在 [Unsloth AI](https://discord.com/channels/1179035537009545276) 和 [GPU MODE](https://discord.com/channels/1189498204333543425) 频道中引发讨论。他们在 [Eleuther](https://discord.com/channels/729741769192767510) 中被重点提及的 [**Multi-head Latent Attention (MLA)**](https://arxiv.org/abs/2502.14837) 架构，通过压缩 **KV cache**，有望实现 **5-10 倍的推理速度提升**，可能会重塑未来的 **LLM** 架构。
- **DeepScaleR：1.5B 模型的 RL 微调性能超越 O1-Preview**：[Torchtune](https://discord.com/channels/1216353675241590815) 频道成员注意到 **DeepScaleR**，这是一个基于 **Deepseek-R1-Distilled-Qwen-1.5B** 并使用强化学习（**RL**）进行微调的 1.5B 参数模型。它在 **AIME2024** 上实现了 **43.1% 的 Pass@1 准确率**，比 **O1-preview** 显著提升了 **14.3%**，展示了 **RL** 在扩展小型模型以处理复杂任务方面的强大能力。详细信息请参阅 [此 Notion 帖子](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)。

**主题 3. IDE 对决：Cursor vs. Windsurf（以及 Vim 的困境）**

- **Cursor 在 Claude 3.7 上的领先引发 Windsurf 用户羡慕，MCP 问题依然存在**：[Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) 频道的用户正焦急地等待 **Claude 3.7**，并对 [Cursor IDE](https://discord.com/channels/1074847526655643750) 用户已经可以使用该模型感到沮丧，这引发了关于 **Cursor** 更快模型集成的辩论。与此同时，[Cursor](https://discord.com/channels/1074847526655643750) 和 [Windsurf](https://discord.com/channels/1027685395649015980) 频道都在讨论持续存在的 **MCP (Model Context Protocol)** 问题，这些问题阻碍了编辑功能，并导致用户不得不手动修改配置文件。
- **Cursor 0.46 更新：索引 Bug 困扰多文件夹工作区**：[Cursor IDE](https://discord.com/channels/1074847526655643750) 用户报告了 **0.46** 版本（[更新日志](https://www.cursor.com/changelog)）中跨多个目录的文件夹索引和文件编辑问题，促使用户检查设置并在 [Cursor 论坛](https://forum.cursor.com/t/indexing-only-reads-first-folder-in-the-workspace/2585/20) 上提供反馈，以改进平台功能。
- **Vim 用户苦于 Codeium Chat 连接错误**：[Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) 频道的 **Vim** 用户正努力尝试在 **Vim** 中通过 **SSH** 启动 **Codeium Chat**，但在访问提供的 **URL** 时遇到连接错误，凸显了特定扩展可能存在的问题，以及在 [Codeium 论坛](https://codeium.com/support) 中寻求针对性支持的必要性。

**主题 4. 硬件黑客与内核深度探索：GPU Mode 社区走向细粒度**

- **GPU MODE 基准测试：Gemlite 内核引发内存受限（Memory-Bound）辩论，Tensor Core 限制被揭示**：[GPU MODE](https://discord.com/channels/1189498204333543425) 频道成员深入研究了 **Triton** 中 [mobiusml/gemlite](https://github.com/mobiusml/gemlite) 库的快速低比特 matmul 内核，讨论了 **gemv 操作的内存受限本质**以及令人惊讶的 **Tensor Core 使用缺失**。他们澄清说，**tensorcores** 要求最小张量维度为 **16x16**，这限制了它们在某些内核优化中的适用性。
- **RX 9850M XT 上的 ROCm/MIOpen Wavefront 问题：调试标志前来救场**：[GPU MODE](https://discord.com/channels/1189498204333543425) 用户报告了 **RX 9850M XT** GPU 上由于错误的 wavefront 大小默认值导致的 **MIOpen** 问题，这引发了 **PyTorch** 中的内存访问故障。分享了一个使用 `MIOPEN_DEBUG_CONV_GEMM=0` 和 `MIOPEN_DEBUG_CONV_DIRECT=0` 的变通方法，更多细节见 [此 GitHub issue](https://github.com/ROCm/MIOpen/issues/3540)。
- **Mojo FFI 弥合 C++/图形学差距，Eggsquad 的生命游戏实现硬件加速**：[Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) 频道成员展示了使用 **Mojo FFI** 链接到 **GLFW/GLEW** 以在 Mojo 中进行图形编程，并提供了一个数独演示 ([ihnorton/mojo-ffi](https://github.com/ihnorton/mojo-ffi))。Eggsquad 展示了一个虽然*搞怪*但令人印象深刻的硬件加速版 **Conway's Game of Life**，该程序使用 **MAX** 和 **pygame** 构建，引发了关于 Mojo 中 GPU 利用率和 SIMD 优化的讨论。

**Theme 5. 社区贡献与课程：共同学习与构建**

- **Unsloth AI 挑战赛升温，仍在大力招揽顶尖人才**：[Unsloth AI](https://discord.com/channels/1179035537009545276) 频道的 **Unsloth AI Challenge** 正在收到大量投稿，但到目前为止，还没有参与者达到发放录用通知的高门槛，这促使官方寻求推荐以寻找合格候选人，显示出 AI 人才招聘领域的竞争态势。
- **Berkeley MOOC：Tulu 3 深度解析，RLVR 训练方法，课程详情即将公布**：[LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) 频道宣传了 **第 4 课**，由 **Hanna Hajishirzi** 讨论 **Tulu 3**（据报道该模型性能优于 **GPT-4o**）以及创新的 **具有可验证奖励的强化学习 (RLVR)** 训练方法。包括项目在内的 MOOC 课程详情即将公布，承诺提供实践学习体验。
- **Hugging Face 社区助力乌克兰语 TTS，推出商业 AI 系列**：[HuggingFace](https://discord.com/channels/879548962464493619) 频道重点介绍了一个新的高质量 **乌克兰语 TTS 数据集**，改善了语音应用的资源 ([speech-uk/opentts-mykyta](https://huggingface.co/collections/speech-uk/ukrainian-text-to-speech-67bd059d61b2598f3a2a7969))。推出了名为“[AI for Business – From Hype to Impact](https://www.linkedin.com/posts/brunokilian_aiforbusiness-artificialintelligence-digitaltransformation-activity-7299815566075158528--xcc?utm_source=share&utm_medium=member_ios&rcm=ACoAAACU3YIBsDJ8y62xuN4LdHlvFqYKsQDJ5eE)”的新系列，旨在指导企业利用 AI，展示了社区在民主化和应用 AI 知识方面的努力。


---

# PART 1: 高层级 Discord 摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 3.7 在编程方面表现出色**：用户报告称 **Claude 3.7** 处理编程任务和复杂 Prompt 的效率比 **3.5** 更高，并指出其在编程能力和实际 Agent 任务方面有显著改进。
   - 社区对 *thinking model*（思考模型）及其在规划和调试方面的潜力印象深刻，尽管正如 [Cursor 的推文](https://x.com/cursor_ai/status/1894093436896129425)所强调的，部分用户发现其在 Agent 模式下的执行力令人沮丧。
- **MCP 工具极大提升用户体验**：**MCP 工具**的集成（特别是配合自定义指令）增强了模型在 **Cursor** 中处理文件和命令的功能，用户正在利用深度思考和浏览器工具（[AgentDeskAI/browser-tools-mcp](https://github.com/AgentDeskAI/browser-tools-mcp)）优化配置。
   - 社区成员正积极探索改进 MCP 设置的方法以获得更好的项目成果，重点关注如何通过有效的 Prompt 来最大化性能。
- **Cursor 更新引发 Bug 讨论**：讨论集中在 **Cursor** **0.46** 版本中文件夹索引和跨多个目录的文件编辑问题（[Changelog](https://www.cursor.com/changelog)）。
   - 用户正在检查设置以解决索引问题，目前正在收集持续的反馈以改进平台功能，详见 [Cursor 论坛](https://forum.cursor.com/t/indexing-only-reads-first-folder-in-the-workspace/2585/20)。
- **社区分享模型使用经验**：成员们正积极分享与 **Claude**、**Grok** 和 **Qwen** 相关的经验和实用工具，讨论它们在 Cursor 中的交互以及优化性能的方法。
   - 社区专注于优化 Cursor 内的交互并分享有效的 Prompt，以最大化 AI 模型性能，并对比不同模型的使用体验。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Claude 3.7 霸榜 Aider 排行榜**：**Claude 3.7 Sonnet** 在 [Aider 排行榜](https://aider.chat/docs/leaderboards/)上使用 **32k thinking tokens** 获得了 **65%** 的评分，超越了以往模型，而其非思考版本的评分为 **60%**。
   - Aider 频道的用户将其性能与 **Grok 3** 进行了对比，指出 **$15/M** 与 Grok **$2.19/M** 的价格差异，对于成本增加是否物有所值评价不一。他们引用了 Anthropic 的[这条推文](https://x.com/AnthropicAI/status/1894092430560965029)。
- **Aider 拥抱 Claude 3.7 的思考能力**：Aider **0.75.1** 版本现已支持 **Claude 3.7**，使用户能够进行实验并有效管理成本；用户正在讨论如 `--copy-paste` 等命令，以优化 Claude 的工作流，详见 [Aider 文档](https://aider.chat/docs/usage/copypaste.html)。
   - 社区讨论了 **Claude Code** 界面的设计和功能，将其与 **Aider** 进行对比，并探索提升编程体验的方法，权衡使用不同平台的优缺点。
- **OpenRouter 为无限 API 开启大门**：用户发现 [OpenRouter](https://openrouter.ai/openai/o3-mini-high) 在绕过速率限制和集中管理 API Key 方面非常有用，简化了访问多种模型的流程。
   - 尽管定价与原始供应商一致，但统一 API 访问的便利性使 OpenRouter 成为一个极具吸引力的选择，尤其是在管理 `claude-3-7-sonnet-20250219` 等模型的 Token 使用量时。
- **Hacker News 个人资料被 AI 吐槽**：**Claude Sonnet 3.7** 现在可以分析 [Hacker News 个人资料](https://hn-wrapped.kadoa.com/)，提供幽默且准确的用户活动亮点和趋势分析。
   - 用户开玩笑说他们的资料被 AI “吐槽（roasted）”了，并赞赏其对他们在线行为的洞察力既搞笑又深刻。
- **Kagi 的 LLM 基准测试评选出新的推理冠军**：[Kagi LLM 基准测试项目](https://help.kagi.com/kagi/ai/llm-benchmark.html)（最后更新于 **2025 年 2 月 24 日**）评估了主流 LLM 在推理、编程和指令遵循方面的表现，揭示了令人惊讶的见解。
   - 结果显示 **Google 的 gemini-2.0** 以 **60.78%** 领先，在这些旨在防止基准测试过拟合的新型任务中，表现优于 **OpenAI 的 gpt-4o**（**48.39%**）。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 用户焦急等待 Claude 3.7**：Windsurf 用户正在等待 **Claude 3.7** 的推出，并对 **Cursor** 用户已经获得访问权限表示沮丧。
   - 社区期待易用性的改进和增强，想知道他们是否能获得访问权限。
- **Windsurf 与 Cursor 性能辩论**：用户正在比较 **Windsurf** 和 **Cursor**，一些人认为 **Cursor** 在实现新模型方面速度更快。
   - 尽管 **Cursor** 速度较快，一些用户仍因其更整洁的 UI 和更好的集成能力而偏好 **Windsurf**。
- **MCP 问题困扰 Windsurf**：持续的讨论围绕 **MCP** 的问题展开，这些问题阻碍了正常的编辑功能，促使用户手动编辑配置文件。
   - 社区在等待支持团队官方修复的同时，正在分享临时解决方案。
- **Vim 用户在 Codeium Chat 中遇到困难**：一位用户在通过 SSH 会话在 Vim 中启动 **Codeium Chat** 时遇到困难，尝试访问提供的 URL 时遇到连接错误。
   - 另一位成员建议在相关的扩展频道寻求帮助，并确认他们正在使用的是哪个扩展。
- **账号创建故障**：一位用户报告称无法创建新的 **Codeium** 账号，遇到了内部错误消息。
   - 这引发了关于其他用户在账号创建过程中是否也面临类似问题的担忧。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Claude 3.7 Sonnet 在编程领域遥遥领先**：用户发现 **Claude 3.7 Sonnet** 在编程和 Web dev 方面优于 **ChatGPT**，理由是根据 [Anthropic 的公告](https://www.anthropic.com/news/claude-3-7-sonnet)，其具有更好的连贯性和可靠性。
   - 根据展示其 Agentic coding 能力的 [推文](https://x.com/alexalbert__/status/1894095781088694497)，新模型 [Claude Code](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview) 促进了代码辅助、文件操作和任务执行。
- **Grok 3 因项目通用性受到关注**：**Grok 3** 因其编程能力而受到赞誉，并被鼓励在 Copilot 等 IDE 中使用，可能会集成 **Claude 3.7** 的功能。
   - 它处理项目的能力（特别是在 Frontend development 方面）受到高度重视，正如一位用户所说，它可以有效地协助 *'Copilot 等 IDE 内部'* 的项目。
- **字节跳动的 Trae 加入免费 AI 狂潮**：**ByteDance** 的 'Trae' 提供对高级 AI 模型的免费访问，标志着 AI 领域的一次竞争性举动。
   - 虽然用户赞赏免费 AI 工具的兴起，但他们仍对这类优惠的临时性质保持警惕，思考其对 AI 领域长期消费者访问和服务的影响。
- **O3 推理受到惊人延迟的困扰**：用户报告称 **O3** 模型经常显示“推理成功”，但完整文本输出会延迟 **10 秒** 或更长时间。
   - 一位用户注意到在 **EST 下午 3 点至 7 点** 之间存在持续延迟，影响了所有“思考”模型，导致用户体验受挫。
- **截图问题阻碍 Bug 报告**：一位成员指出在当前聊天频道添加截图存在挑战，阻碍了有效的 Bug 报告。
   - 用户互相引导如何浏览不同频道和报告 Bug，强调了在发布新问题之前探索现有问题以提高效率的重要性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth Challenge 寻求强力候选人**：**Unsloth AI Challenge** 正在收到多份提交，但到目前为止，还没有参与者达到获得录用通知（offer）的标准。
   - 社区鼓励推荐该挑战赛，以帮助寻找合格的候选人。
- **DeepSeek 的发布引发观点分歧**：**DeepSeek OSS** 的发布引入了 **MoE kernels** 和高效通信库等特性（[来自 Daniel Han 的推文](https://x.com/danielhanchen/status/1894212351932731581)），引发了褒贬不一的反应。
   - 一些人认为它更多地迎合了大公司，引发了关于这种开放性将如何影响竞争和 AI 开发格局（特别是在降低模型成本方面）的辩论。
- **训练配置引发困扰**：用户报告称，当从 **batch size 8** 切换到带有 **gradient accumulation** 的 **batch size 1** 时，validation loss 意外地显著增加，这令人感到困惑。
   - 社区探讨了混合设置策略如何影响模型学习，以及不同策略是否会产生相似的结果。
- **Checkpointing 期间的 VRAM 占用问题**：讨论强调了在 checkpointing 期间进行模型保存操作时，**VRAM usage** 会出现显著峰值。
   - 用户建议将模型保存为 LoRA 以卸载 VRAM，从而更好地管理内存。
- **QwenLM 发布新 qwq 模型**：一名成员分享了 **QwenLM** 发布的新 **qwq model** 链接（[qwq-max-preview](https://qwenlm.github.io/blog/qwq-max-preview/)）。
   - 该帖子预告了他们的新模型，但目前还没有进一步的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.7 Sonnet 的推理能力升级**：**Claude 3.7 Sonnet** 已在 OpenRouter 上线，增强了**数学推理**、**编程**和**问题解决**能力，详见 [Anthropic 的博客文章](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 此次发布改进了 agentic workflows，并提供了快速推理和扩展推理的选项，不过扩展思考（extended thinking）功能的文档和支持尚在完善中。
- **OpenRouter 管理 API Keys 并保留额度**：可以生成新的 API keys 而不会丢失账户额度，因为额度是与账户绑定的，而非单个 key。
   - 讨论澄清了尽管用户对丢失 API keys 表示担忧，但无论 key 的状态如何，额度都是安全的。
- **模型定价结构：Sonnet vs 其他模型**：**Claude 3.7 Sonnet** 的定价为输入每百万 **tokens** 3 美元，输出（包括 thinking tokens）每百万 **tokens** 15 美元，旨在平衡用户参与度和模型效用。
   - 与其他模型的对比显示，虽然 Claude 的定价被认为较高，但在特定性能指标（尤其是推理任务）方面仍具竞争力。
- **图片上传触及大小限制**：用户报告在使用 **Claude 3.7** 时频繁报错，特别是当图片大小超过 5MB 时，会导致请求失败。
   - 建议参与者将图片大小控制在限制范围内，以符合 API 要求并确保成功处理。
- **设备同步功能缺失**：**OpenRouter** 不支持在不同设备（如桌面端到移动端）之间同步聊天会话。
   - 有建议称用户可以使用 *chatterui* 和 *typign mind* 等第三方应用来弥补这一功能缺口。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Claude 3.7 Sonnet 携智能升级登场**：根据 [Anthropic](https://www.anthropic.com/news/claude-3-7-sonnet) 的消息，**Claude 3.7 Sonnet** 已发布，提升了推理、编程和调试任务的能力，尽管它有时会生成更复杂的代码。
   - X（原 Twitter）上的用户正在分享有趣的测试结果，例如[这个用 Tikz 绘制的雅典卫城](https://x.com/DimitrisPapail/status/1894127499224694877)，该模型拥有多种推理选项，包括 64k reasoning tokens。
- **Qwen AI 展示 QwQ-Max 预览版**：根据[官方博客文章](https://qwenlm.github.io/blog/qwq-max-preview/)，**Qwen AI** 发布了 **QwQ-Max-Preview**，展示了其推理能力，并在 Apache 2.0 协议下保持开源可访问性。
   - 正如[这条推文](https://x.com/StringChaos/status/1894135561059013023)所述，**QwQ-Max-Preview** 在 LiveCodeBench 上的评估显示其性能与 **o1-medium** 相当。
- **伯克利高级 Agent 课程聚焦 Tulu 3**：**Berkeley Advanced Agents** MOOC 在太平洋标准时间今天下午 4 点邀请了 **Hanna Hajishirzi** 讨论 **Tulu 3**，并在 YouTube 上进行了直播（[链接](https://www.youtube.com/live/cMiu3A7YBks)）。
   - 参与者称赞该课程内容详实且具有实践意义。
- **DeepSeek 开源 EP 通信库**：**DeepSeek** 在开源周期间宣布了其开源的 **EP** 通信库 **DeepEP**，专注于模型训练的高效通信，详见[这条推文](https://x.com/deepseek_ai/status/1894211757604049133)。
   - 此次发布是 AI 技术开源趋势的一部分，旨在鼓励社区参与。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gemlite 库引发 Kernel 讨论**：成员们讨论了包含 **Triton** 编写的快速低比特 matmul kernel 的 [mobiusml/gemlite 库](https://github.com/mobiusml/gemlite)，强调了 gemv 操作的 **memory-bound**（内存受限）特性。
   - 讨论中提到 gemv 不使用 **Tensor Cores**，因为它们只增加 flops，且 **tensorcores** 要求最小张量维度为 **16x16**。
- **TorchAO 获得 E2E 示例**：`@drisspg` 确认他们正在开发 **TorchAO 中的 E2E 示例**。
   - 这突显了 **TorchAO** 社区协作讨论的性质。
- **DeepSeek AI 开源专家并行通信库**：**deepseek-ai** 的 GitHub 仓库 [DeepEP](https://github.com/deepseek-ai/DeepEP) 提供了一个高效的专家并行（expert-parallel）通信库。
   - 其目标是增强分布式系统的通信性能。
- **Meta 为 AI 协作招募 PyTorch 工程师**：Meta 宣布招聘 **PyTorch Partner Engineers**，与领先的行业合作伙伴及 PyTorch 团队合作。
   - 该职位强调**系统和社区工作**，并提供将 **AI 技术**从研究推向实际应用的机会，同时确保招聘过程中的**平等就业机会**（Equal Employment Opportunity）。
- **MIOpen Wavefront 问题**：一位用户报告了由于 **RX 9850M XT** 上错误的 wavefront 大小默认值导致 **MIOpen** 出现问题，从而引发 **PyTorch** 中的内存访问错误。
   - 他们使用 `MIOPEN_DEBUG_CONV_GEMM=0` 和 `MIOPEN_DEBUG_CONV_DIRECT=0` 作为临时解决方案，更多信息可在[此 GitHub issue](https://github.com/ROCm/MIOpen/issues/3540) 中找到。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Grok vs O1：模型大讨论？**：关于是否从 **O1-Pro** 切换到 **SuperGrok** 引发了激烈辩论，双方对其编程能力的看法各不相同。
   - 一些工程师认为 **O1** 在代码处理方面表现更好，而另一些人则欣赏 **Grok** 详尽的回复，即便这需要对 Prompt 进行调整。
- **xAI 构建巨型计算集群**：**xAI** 计划将其 GPU 集群扩展至 **200,000 个 GPU**，将 Grok 定位为挑战 OpenAI 的多功能 AI 平台。
   - 据 [NextBigFuture.com](https://www.nextbigfuture.com/2024/04/snapshot-of-the-race-for-more-ai-compute.html) 报道，此次扩张反映了将 **Grok** 打造为更通用解决方案的战略举措。
- **DeepSeek 深入探索数据合成**：讨论强调了**合成数据生成**日益增长的重要性，特别是 **DeepSeek** 利用概念而非仅仅是 Token 进行优化的范例。
   - 构建强大的合成数据流水线现在被视为 AI 模型可持续发展的关键，创建高质量训练数据的技术正成为一种竞争优势。
- **Claude 3.7 Sonnet 表现亮眼**：Anthropic 推出了 **Claude 3.7 Sonnet**，宣传其混合推理能力可实现近乎即时的响应，在**编程**和**前端开发**方面表现卓越，[点击查看更多](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 随新模型一同推出的还有 **Claude Code**，旨在处理 Agent 编程任务，以进一步简化开发者的工作流程，详见[此 YouTube 视频](https://m.youtube.com/watch?v=t3nnDXa81Hs)。
- **Qwen 满足推理需求**：**Qwen Chat** 预览了基于 **Qwen2.5-Max** 的推理模型 **QwQ-Max**，并计划推出 **Android 和 iOS 应用**以及遵循 **Apache 2.0 许可证**的开源版本。
   - 其在数学、编程和 Agent 任务方面的专业能力详情可见其 [博客文章](https://qwenlm.github.io/blog/qwq-max-preview/)，这标志着其在推理能力上迈出了雄心勃勃的一步。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **大脑分块，AI 预测**：讨论指出人类语言处理依赖于**分块（chunking）和抽象**，而不像 AI 模型那样同时预测 Token，这让人质疑大脑与 AI 之间的并行性。
   - 辩论强调了由于**反向传播挑战**导致传统 **RNN 架构**难以扩展的问题，以及对替代架构的潜在需求。
- **DeepSeek 的 MLA 碾压 KV Cache**：由 DeepSeek 创新的 **Multi-head Latent Attention (MLA)** 显著压缩了 **Key-Value (KV) cache**。根据[这篇论文](https://arxiv.org/abs/2502.14837)，与传统方法相比，推理速度提高了 **5-10 倍**。
   - DeepSeek 在 **MLA 驱动的模型**上投入了至少 **550 万美元**，这表明业界对未来模型转向 **MLA 技术**充满兴趣。
- **Looped Transformers 高效推理**：一篇论文提出，**Looped Transformer 模型**在推理任务中可以匹配更深的非循环架构的性能，同时减少了参数需求，详见[这篇论文](https://arxiv.org/abs/2502.17416)。
   - 这种方法展示了迭代方法在应对复杂计算挑战中的优势，并具有降低计算成本的潜在益处。
- **Attention Maps 依然流行吗？**：成员们讨论了 **Attention Maps** 与**基于神经元的方法**相比受欢迎程度下降的问题。一些人认为这是由于 Attention Maps 的观察性质，而另一些人提到 Attention Maps 在前向传播过程中可以直接进行干预。
   - 一位成员表达了对 **Attention Maps** 的偏好，因为它们能够利用语言特征生成**树和图**，并引用了自 **BERT** 以来从 **Attention Maps** 中涌现的语法特性。
- **混合精度状态解析**：在**混合精度训练**中，除非激活了 ZeRO Offload，否则 **Master FP32 权重**通常存储在 **GPU VRAM** 中。
   - 建议使用 **BF16 低精度权重**配合 **FP32 优化器状态 + Master 权重 + 梯度**进行常规混合精度训练，成员们指出 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44) 是一个极佳的参考示例。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic 发布 Claude 3.7 Sonnet 和 Claude Code**：Anthropic 推出了 **Claude 3.7 Sonnet**，其中包括推理能力的提升，以及 **Claude Code**，一个基于终端的编程助手 ([@anthropic-ai/claude-code](https://www.npmjs.com/package/@anthropic-ai/claude-code?activeTab=code))。
   - 测试者注意到其 **更高的输出 Token 限制** 和 **增强的推理模式**，带来了更连贯的交互，一些人还称赞了其巧妙的 System Prompt ([Mona 的推文](https://x.com/dyot_meet_mat/status/1894139577805267447?s=46))。
- **微软取消数据中心租赁**：报告显示微软正在取消数据中心租赁，这预示着数据中心市场可能存在 **供应过剩** ([Dylan Patel 的推文](https://x.com/dylan522p/status/1894050388145508586?s=46))。
   - 此举让人们对微软 2024 年初激进的预租赁策略及其对托管服务领域的广泛影响产生怀疑。
- **Qwen 的未来发布引发期待**：对于即将发布的 **Qwen QwQ-Max** 的期待正在增长，据报道该版本具有改进的推理能力，并将在 Apache 2.0 协议下开源 ([Hui 的推文](https://x.com/huybery/status/1894131290246631523?s=46))。
   - 社区正密切关注这些进展及其对智能模型格局的影响。
- **Claude Code 简化编程辅助**：早期用户发现 **Claude Code** 对编程辅助很有帮助，但体验各异，一些人反映响应时间较慢。
   - 反馈还提到了价格方面的顾虑，并赞扬了优化性能的 **强力缓存 (heavy caching)** 功能。
- **GitHub 的 FlashMLA 吸引社区关注**：GitHub 项目 **FlashMLA** 最近被分享，在没有任何宣传的情况下引发了对其在 AI 开发中可能产生的影响的兴趣 ([FlashMLA GitHub](https://github.com/deepseek-ai/FlashMLA))。
   - 成员们对本周预期发布的 AI 工具的未来进展感到兴奋。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Grok3 工具调用引发困扰**：成员们对 **Grok3** 的工具调用 (tool invocation) 机制表示困惑，指出 System Prompt 中缺少 Token 序列列表。
   - 讨论质疑了硬编码函数调用与用于工具调用的 In-context Learning 的可靠性和透明度。
- **Claude 3.7 Sonnet 登场**：Anthropic 发布了 [**Claude 3.7 Sonnet**](https://www.anthropic.com/news/claude-3-7-sonnet)，强调了其混合推理能力和用于编程的命令行工具。
   - 早期基准测试表明 **Sonnet** 在软件工程任务中表现优异，使其成为 **Claude 3.5** 的潜在替代者。
- **QwQ-Max-Preview 展示深度推理**：[**QwQ-Max-Preview**](https://qwenlm.github.io/blog/qwq-max-preview/) 作为 Qwen 系列的一部分亮相，主打深度推理和 Apache 2.0 协议下的开源可访问性。
   - 社区推测该模型的尺寸，并希望有更小的、可在本地使用的版本。
- **结构化输出项目寻求反馈**：一个旨在解决结构化输出 (Structured Outputs) 挑战的开源项目启动，邀请社区反馈和协作。
   - 成员建议重新发布公告，以增加在社区内的曝光度和参与度。
- **Sonnet-3.7 在 Misguided Attention 中脱颖而出**：一位成员使用 [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) 对 **Sonnet-3.7** 进行了基准测试，注意到其处理误导信息的能力。
   - 他们声称它作为非推理模型表现最好，几乎超越了 **o3-mini**，进一步证明了其在推理评估中的竞争优势。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Anthropic 发布 MCP Registry API**：@AnthropicAI 发布了官方 [MCP registry API](https://x.com/opentools_/status/1893696402477453819)，旨在为 **MCP** 工具和集成提供权威来源（source of truth）。
   - 社区成员表示兴奋，希望这将使 **MCP** 管理标准化并提高可靠性。
- **LLM 版本命名规范令用户困惑**：用户对 **LLM 版本命名**感到困惑，尤其是 **Claude 3.7**，并注意到它通过 adaptive thinking modes 整合了功能并提升了性能。
   - 这表明 **version 3.7** 在通过 adaptive thinking modes 提升性能的同时，整合了之前版本的功能。
- **Haiku 3.5 的工具支持存在特性**：根据用户经验，**Haiku 3.5** 虽然支持工具，但在连接较少工具时表现更好。
   - 服务器和工具管理的压力是一个令人担忧的问题，这促使了开发集成工具集的聊天应用的计划，以便于使用。
- **Claude Code 在编码任务中优于 Aider**：用户发现 **Claude Code** 在有效处理编码错误方面优于 **Aider**。
   - 人们对使用 **Claude Code** 处理更复杂的任务越来越感兴趣，可能会利用 heavy thinking modes 来获得最佳结果。
- **MetaMCP 考虑使用 AGPL 以保持开放性**：**MetaMCP** 正在考虑切换到 **AGPL** 协议，以鼓励社区贡献。
   - 此举旨在解决当前 **ELv2** 许可证的局限性，该许可证限制了贡献，并要求托管的更改必须开源。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Gemma-9B 模型面临加载错误**：用户报告了在使用自定义 **LoRA** 加载 **Gemma-9B** 模型时出现错误，特别是 `LoraConfig.__init__()` 函数。
   - 几位用户寻求建议，表明在通过自定义配置适配 **Gemma-9B** 时存在共同的技术挑战。
- **Qwen Max 推理模型备受期待**：**Qwen Max** 推理模型预计将于本周发布，被誉为自 **DeepSeek-R1** 以来最强大的开源模型。
   - 根据[这条推文](https://x.com/Alibaba_Qwen/status/1894130603513319842)，**Qwen Max** 在数学理解和创造力方面表现出进步，并计划推出官方版本和移动应用。
- **Claude-3.7 被称为潜在的 SWE-bench 杀手**：**Claude-3.7** 被称为潜在的 **SWE-bench 杀手**，预计性能可与 **DeepSeek-R1** 媲美，而体积仅为其三分之一。
   - 由于其每百万 token 输出的高昂定价，人们对 **Claude-3.7** 的成本效益提出了担忧。
- **乌克兰语 TTS 数据集增强了可访问性**：Hugging Face 上发布了一个高质量的乌克兰语 **TTS 数据集**，改善了 **TTS** 项目的资源，更新详情见[此处](https://huggingface.co/collections/speech-uk/ukrainian-text-to-speech-67bd059d61b2598f3a2a7969)。
   - **speech-uk/opentts-mykyta** 数据集的最新更新展示了对乌克兰语文本转语音应用的持续支持。
- **AI for Business 系列揭晓**：推出了名为 [AI for Business – From Hype to Impact](https://www.linkedin.com/posts/brunokilian_aiforbusiness-artificialintelligence-digitaltransformation-activity-7299815566075158528--xcc?utm_source=share&utm_medium=member_ios&rcm=ACoAAACU3YIBsDJ8y62xuN4LdHlvFqYKsQDJ5eE) 的新系列，旨在帮助企业利用 **AI** 获得竞争优势。
   - 该系列计划涵盖如何在不中断业务运营的情况下扩展 **AI** 的主题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 VL 模型夺冠**：**Qwen 2.5 VL 7B** 模型在质量上超越了 **Llama 3.2** 等先前模型，因其在生成图像描述方面的出色表现而受到关注。根据用户报告，该模型可从 [Hugging Face](https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF) 下载。
   - 多位用户确认了其在现实场景中的应用潜力，特别是在视觉语言应用中，并可以在 **LM Studio** 中本地运行。
- **讨论 Deepseek R1 本地托管**：用户讨论了在本地运行 **Deepseek R1 671B**，根据文档，这需要 **192GB** 的 RAM 阈值，同时将计算卸载到 GPU。
   - 一位用户分享了使用特定量化技术优化模型性能的经验，使用了 [Unsloth's GGUF](https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S)。
- **应对装机难题**：一位用户对电脑配置的兼容性表示沮丧，指出 **aio pump** 需要一个 **USB 2.0 header**，并且会干扰最后一个 **PCIE slot**。
   - 该用户正在考虑替代方案，因为他们有“第二个系统可以放置所有组件”，这表明他们正转向优化硬件配置。
- **Apple M2 Max：依然可行吗？**：一位用户选择了翻新的 **M2 Max 96GB** 用于爱好和工作，对投资最新的 **M4 Max** 芯片表示犹豫。
   - 讨论涉及了各种 Apple 芯片的 **clock and throttle behavior**（频率和降频行为），并指出了功耗差异：**M2 Max** 功耗为 **60W**，而 **M4 Max** 峰值达到 **140W**。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Ultra 的存在引发关注**：成员们对 **SD3 Ultra** 表示好奇，称其与 **SD3L 8B** 相比具有更高频率细节的潜力。
   - 一位用户增加了关于其能力的神秘感，指出：*它仍然存在——我还在使用它*。
- **图像生成速度大比拼**：图像生成时间差异巨大，一位用户报告在未说明的配置上需要 **20 分钟**，而另一位用户在 **3060 TI** 上生成 **1920x1080** 图像仅需 **31 秒**。
   - 使用 **SD1.5** 的性能基准测试在 **3070 TI** 上平均为 **4-5 秒**，突显了模型和硬件选择对速度的影响。
- **寻找犬种图像数据集**：一位成员正在寻找超过 **20k 图像** 的犬种图像数据集，超出了 **Stanford Dogs dataset** 的容量。
   - 他们强调需要明确标注犬种的数据集。
- **分辨率调整带来速度提升**：用户讨论了最佳分辨率设置，提倡使用 **576x576** 等较小尺寸来提高图像生成速度。
   - 一位用户报告在实施这些调整后，处理时间缩短至约 **8 分钟**。
- **Stability AI 开启反馈通道**：Stability AI 推出了[新的功能请求看板](https://stabilityai.featurebase.app/)，允许用户通过 Discord 使用 `/feedback` 命令提交想法。
   - 这一举措支持社区对请求进行投票，为 Stability AI 的开发优先级提供参考，并*帮助我们确定下一步工作的优先级*。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo FFI 助力图形编程**：成员展示了使用 **Mojo FFI** 将静态库链接到 **GLFW**/**GLEW**，通过一个数独示例证明了在 Mojo 中通过自定义 C/C++ 库进行图形编程的可行性。
   - 他们使用了带有 `alias draw_sudoku_grid = external_call[...]` 语法的别名来简化函数访问，并使用 Python 脚本动态链接库，该脚本可在 [这里](https://github.com/ihnorton/mojo-ffi) 获取。
- **Mojo 的依赖版本控制至关重要**：一位用户报告了在一个新的 Mojo 项目中 **lightbug_http** 依赖项出现错误，并引用了 Stack Overflow 的问题，而另一位用户则推测将 **small_time** 依赖项固定在 `25.1.0` 是否可能是导致错误的原因。
   - 这些报告表明，精确的依赖版本控制对于避免安装和配置问题至关重要。
- **MAX 在 GPU 上运行康威生命游戏**：一位成员展示了他们使用 **MAX** 和 **pygame** 实现的硬件加速版**康威生命游戏 (Conway’s Game of Life)**，称其为一个相当“愚蠢”的应用，同时引发了关于将 **MAX** 与 **2080 Super** GPU 配合使用的兼容性讨论。
   - 讨论建议从 Python 运行脚本以促进 GPU 集成，这可以用于通过 graph API 向模型添加参数。
- **SIMD 实现诱惑 Eggsquad**：Eggsquad 提到发现了 Daniel Lemire 的 **SIMD 化实现**，但表示现阶段不愿进一步探索，而 Darkmatter 指出利用位打包 (bit packing) 可以在其实现中支持更大的图。
   - Eggsquad 确认他们的**康威生命游戏**实现在修复一个 bug 后运行良好，并通过动画展示了结果，其中包括一个描绘游戏中“枪”的动画。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **据报道 Tulu 3 表现优于 GPT-4o**：据 Hanna Hajishirzi 称，**Tulu 3** 作为一种最先进的经过后期训练的语言模型，通过创新的训练方法超越了 **DeepSeek V3** 和 **GPT-4o**。
   - 这一声明是在一次涵盖语言模型训练和增强推理能力的全面努力的[讲座](https://www.youtube.com/live/cMiu3A7YBks)中发表的。
- **伯克利 LLM Agents 课程将使用 RLVR**：一种独特的**具有可验证奖励的强化学习方法 (RLVR)** 正被展示为一种有效训练语言模型的方式，旨在显著影响训练期间的推理。
   - Hanna 在[讲座](https://www.youtube.com/live/cMiu3A7YBks)中分享了关于在训练中结合这些先进强化学习技术的测试策略见解。
- **MOOC 学生面临测验提交宽限**：澄清说明测验截止日期仅适用于**伯克利学生**，MOOC 学生的所有测验均在学期末截止。
   - 这一澄清为迟到者和担心错过初始截止日期的人提供了保证。
- **MOOC 课程详情即将发布**：一位成员宣布 **MOOC 课程详情将很快发布**，包括一个项目部分，并引用了一个 [Discord 链接](https://discord.com/channels/1280234300012494859/1282734248112947210/1343746798795358340)。
   - 然而，研究轨道仅面向**伯克利学生**。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **谷歌 Deep Research 与 Gemini 集成引发热潮**：围绕将 [Google Deep Research](https://link.to/research) 和 **Gemini** 与 NotebookLM 集成以增强功能的讨论被点燃。
   - 爱好者们对未来的发展表示兴奋。
- **用户在 NotebookLM 语言设置中挣扎**：关于在不影响 Google 账户语言的情况下更改 NotebookLM **语言设置**的问题浮出水面。
   - 一位用户寻求关于如何有效实施此类语言更改的建议。
- **创意书籍数字化策略出现**：一位成员建议使用 **lens app** 拍摄每一页以创建 PDF，然后将其转换为 PowerPoint 上传到 NotebookLM。
   - 也有人提出了替代方案，例如使用复印机或 **Adobe Scan 应用**直接创建 PDF。
- **多语言 Prompt 有效性分析**：关于是使用单个还是多个 Prompt 来促使 NotebookLM 中的主持人说**德语**引发了辩论。
   - 一位成员推测，有效性可能与其 **premium 订阅**状态有关。
- **Claude 3.7 引发用户狂热**：用户对 **Claude 3.7** 充满热情，希望在选择模型方面有更多控制权。
   - 一位用户发起了关于此类决定对用户体验影响的讨论。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AI Assistant 为 LlamaIndex 正式上线**：LlamaIndex 文档中出色的 **AI assistant** 现在已面向所有人开放！点击[这里](https://t.co/XAyW7wLALJ)查看。
   - 团队非常期待看到用户如何将其整合到自己的工作流中。
- **ComposioHQ 发布又一力作！**：[ComposioHQ](https://t.co/W4l129gHce) 发布了又一个重磅产品！其功能和特性持续给人留下深刻印象。
   - 早期采用者称赞其直观的界面和强大的功能集，并表示期待进一步的改进。
- **Anthropic 发布 Claude Sonnet 3.7**：AnthropicAI 发布了 **Claude Sonnet 3.7**，目前的情绪反馈和评估都非常积极。通过 `pip install llama-index-llms-anthropic --upgrade` 即可获得 Day 0 支持。
   - 更多详情可以在 [Anthropic 的发布公告](https://t.co/PjaQWmmzaN)中找到，其中强调了此新版本最新的集成能力。
- **BM25 Retriever 需要 Nodes**：一位成员指出，**BM25 retriever** 无法仅从 vector store 初始化，因为 **docstore** 必须包含已保存的 nodes。
   - 解决该问题的一个建议是将 **top k 设置为 10000** 以检索所有 nodes，尽管这可能效率不高。
- **MultiModalVectorStoreIndex 在处理图像时遇到困难**：一位成员在尝试创建 **MultiModalVectorStoreIndex** 时遇到了与图像文件相关的错误，尽管图像存储在 GCS bucket 中。
   - 该问题专门针对图像出现，因为他们的代码在处理 **PDF documents** 时运行正常，这表明 index 需要更好的图像处理能力。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **TorchTune 辩论微调中的截断问题**：**Torchtune** 的 #dev 频道出现了一场讨论，关于微调时是否应将默认设置从 **right truncation** 改为 **left truncation**，并引用了一个支持性的[图表](https://link.to.graph)。
   - 意见不一，一些人承认 **HF** 目前的默认设置是 right truncation，而另一些人则主张进行更改。
- **StatefulDataLoader 寻求评审**：一位成员请求对其 [pull request](https://github.com/pytorch/torchtune/pull/2410) 进行评审，该 PR 为 **Torchtune** 添加了对 **StatefulDataLoader** 类的支持。
   - 该 pull request 旨在引入新功能并解决 **Torchtune** 框架内潜在的 bug 修复。
- **DeepScaleR 使用 RL 超越 O1-Preview**：**DeepScaleR** 模型基于 **Deepseek-R1-Distilled-Qwen-1.5B** 通过强化学习（RL）微调而成，在 **AIME2024** 上实现了 **43.1% Pass@1 准确率**。根据其 [Notion](https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2)，这比 O1-preview 提升了 **14.3%**。
   - 这突显了强化学习在扩展模型以提高准确性方面的有效性。
- **Deepseek 开源 DeepEP 通信库**：作为 #OpenSourceWeek 的一部分，**Deepseek AI** 推出了 **DeepEP**。根据[其推文](https://x.com/deepseek_ai/status/1894211757604049133)，这是一个专门为 **Mixture of Experts (MoE)** 模型训练和推理定制的开源通信库。
   - **DeepEP** 支持 **FP8 dispatch** 并优化了节点内和节点间通信，旨在简化训练和推理阶段，可在 [GitHub](https://github.com/deepseek-ai/DeepEP) 上获取。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **询问 DeSci 中 Validator 的可行性**：一位用户询问了 **DeSci** 领域内 **POS Validators** 的**盈利阈值**。
   - 这一查询强调了在去中心化科学中运行 nodes 的经济可行性的重要性。
- **讨论 Validator 池化策略**：一位用户提到了 **pool validator nodes**，表现出对 Validator 之间共享资源或协作的兴趣。
   - 这暗示了通过池化方法提高 Validator 效率的趋势。
- **辩论资产估值专业知识**：一条消息提到了 **asset value expert** 一词，但由于其他无关术语，其上下文尚不明确。
   - 这引发了关于在所讨论的主题中，评估资产估值的专业知识重要性的疑问。



---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All v3.10.0 发布并带来升级**：**GPT4All v3.10.0** 首次亮相，带来了更好的远程模型配置和更广泛的模型支持，并解决了崩溃问题。
   - 增强功能涵盖了跨平台的稳定性能提升和多项崩溃修复。
- **远程模型配置更加顺畅**：“添加模型”页面现在拥有一个专门的标签页，用于配置 **Groq**、**OpenAI** 和 **Mistral** 等**远程模型提供商**，使模型配置更加便捷。
   - 这一增强旨在使外部解决方案在 GPT4All 环境中的集成变得无缝。
- **CUDA 兼容性进一步扩大**：此次更新引入了对具有 **CUDA compute capability 5.0** 的 GPU 的支持，扩展了兼容硬件的范围。
   - 这包括 **GTX 750**，提升了不同硬件配置用户的可访问性。
- **GPT4All 引发版本命名疑问**：成员们对于 **v3.10.0** 是否应该标记为 **v4.0** 产生困惑，并引发了对版本命名惯例的询问。
   - 最近发布的 **Nomic Embed v2** 加剧了这种困惑。
- **用户期待 Nomic Embed v2**：用户热切期待 **GPT4All v4.0.0**，尤其是因为当前版本尽管新版本已发布，但仍依赖于 Nomic Embed **v1.5**。
   - 社区提醒成员在期待即将到来的更新时，*耐心是关键*。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **phi4 响应格式依然不同**：**phi4 响应**格式与大多数模型存在一些差异，相关教程待发布。
   - 一份旨在更好解释该格式的教程正在制作中。
- **Assertion 迁移流程简化**：从 **2.5 风格 Assertions** 迁移的用户可以使用 `dspy.BestOfN` 或 `dspy.Refine` 来简化模块。
   - 这些新选项有望比传统断言提供更高的效率。
- **BestOfN 已实现**：一个示例展示了在 **ChainOfThought** 模块中实现 `dspy.BestOfN`，允许最多 **5 次重试**。
   - 该方法将选择最佳奖励，并在达到 **threshold**（阈值）时停止。
- **奖励函数解析**：分享了一个 `reward_fn` 示例，展示了它如何返回 float 或 bool 等标量值来评估**预测字段长度**。
   - 该函数适用于 **dspy.BestOfN** 实现的上下文中。

---

**tinygrad (George Hotz) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

# 第 2 部分：详细的频道摘要和链接

{% if medium == 'web' %}

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1343632399908471074)** (1056 条消息🔥🔥🔥): 

> `Claude 3.7, MCP Tools, Cursor Updates, Thinking Model, User Experiences`

- **Claude 3.7 性能**：用户报告 Claude 3.7 有显著提升，特别强调了它在编程任务中的出色表现，以及比 3.5 更高效地处理复杂 Prompt 的能力。
   - Thinking Model 在规划和调试方面表现出更好的性能，尽管一些用户对其在 Agent 模式下的执行效果感到沮丧。
- **MCP 工具使用**：各种 MCP 工具的集成增强了用户体验，特别是通过自定义指令引导模型更有效地处理文件和命令。
   - 用户正在探索如何优化配置，利用 Deep Thinking 和浏览器工具等功能来改善项目成果。
- **Cursor 更新与 Bug**：讨论涉及 Cursor 的功能问题，特别是关于文件夹索引以及在 0.46 版本中跨多个目录编辑文件的能力。
   - 建议用户检查设置以解决索引问题，目前正通过持续的反馈来完善平台。
- **模型对比分析**：多位用户指出，虽然 3.7 提供了良好的体验，但仍存在一些小瑕疵，特别是当与非 Thinking Model 或旧版本协同使用时。
   - 社区对 Golang 和其他语言在 Claude 3.7 能力下的表现表现出浓厚兴趣，重点在于调整 Workflow 和评估输出质量。
- **社区参与与分享**：成员分享了与 Claude 相关的有用链接和工具，并讨论了包括 Grok 和 Qwen 在内的各种 AI 模型的体验。
   - 对话还围绕优化 Cursor 内部的交互以及寻找有效的 Prompt 以最大化性能展开。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/fce3511bab261b4c986797f3e1e40e7621bbd012/darwin/arm64/Cursor-darwin-arm64.zip">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>: Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它似乎代表了目前的最先进水平。</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273?s=46">来自 Sualeh (@sualehasif996) 的推文</a>: 可配置的思考模式即将推出！👀 引用 Cursor (@cursor_ai) 的话：Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它似乎...</li><li><a href="https://x.com/alibaba_qwen/status/1894130603513319842?s=46">来自 Qwen (@Alibaba_Qwen) 的推文</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat.qwen.ai/ 博客: https://qwenlm.github.io/blog/qwq-max-preview/ 🤔 今天我们在 Qwen Chat 中发布了“Thinking (QwQ)”，由 o... 支持</li><li><a href="https://x.com/cursor_ai/status/1894093438863511742">来自 Cursor (@cursor_ai) 的推文</a>: 我们正在逐步开放最高级别的思考模式访问权限。如需尝试，请选择 claude-3.7-sonnet-thinking 或 claude-3.7-sonnet 并启用 Agent 模式。</li><li><a href="https://x.com/ChujieZheng/status/1894095584774250858">来自 Chujie Zheng (@ChujieZheng) 的推文</a>: 兄弟，你在开玩笑吗？</li><li><a href="https://anysphere-binaries.s3.us-east-1.amazonaws.com/production/client/linux/x64/appimage/Cursor-0.46.3-bbefc49a7fd08b08a4f17a525bdc5bb7e44ce57a.deb.glibc2.25-x86_64.AppImage">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/theo/status/1894101944068641241?t=iXaaI_9aHmFsjJYsjiGZhw">来自 Theo - t3.gg (@theo) 的推文</a>: Claude 3.7 思考模型在弹球挑战（bouncing ball challenge）中失败的方式和 Grok 3 一样？🤔</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campaign=vibe-coding&utm_content=default">Vibe Coding | Bonfire</a>: 购买支持 Nova Ukraine 的 Vibe Coding 周边商品。主打深麻灰高级中性 T 恤，美国专业印刷。</li><li><a href="https://www.bonfire.com/vibe-coding/?utm_source=copy_link&utm_medium=post_campaign_launch&utm_campai">Vibe Coding | Bonfire</a>: 购买支持 Nova Ukraine 的 Vibe Coding 周边商品。主打深麻灰高级中性 T 恤，美国专业印刷。</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>: 今天，我们发布了 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个正式推出的混合推理模型。</li><li><a href="https://x.com/sualehasif996/status/1894094715479548273">来自 Sualeh (@sualehasif996) 的推文</a>: 可配置的思考模式即将推出！👀 引用 Cursor (@cursor_ai) 的话：Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实的 Agent 任务中。它似乎...</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>: 未找到描述</li><li><a href="https://cursor.directory/rules">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://status.anthropic.com/">Anthropic 状态</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/indexing-only-reads-first-folder-in-the-workspace/2585/20">索引仅读取工作区中的第一个文件夹</a>: 这个有更新吗？这真的很重要 😬</li><li><a href="https://forum.cursor.com/t/integrate-claude-3-7-sonnet-into-cursor/54060">将 Claude 3.7 Sonnet 集成到 Cursor</a>: 你好！我非常希望看到 Claude 的新推理模型 (3.7) 与 Composer 内部的 Agent 集成。顺颂商祺，Johannes。</li><li><a href="https://github.com/alexandephilia/ChatGPT-x-DeepSeek-x-Claude-Linux-APP">GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Claude-Linux-APP: 基于 Electron 的各种 AI 聊天平台桌面应用。</a>: 基于 Electron 的各种 AI 聊天平台桌面应用。 - GitHub - alexandephilia/ChatGPT-x-DeepSeek-x-Claude-Linux-APP: 基于 Electron 的各种 AI 聊天平台桌面应用。</li><li><a href="https://tenor.com/view/it-turn-on-and-off-phone-call-tech-support-gif-13517106">开关机 GIF - IT 开关机电话 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/AgentDeskAI/browser-tools-mcp">GitHub - AgentDeskAI/browser-tools-mcp: 直接从 Cursor 和其他兼容 MCP 的 IDE 中监控浏览器日志。</a>: 直接从 Cursor 和其他兼容 MCP 的 IDE 中监控浏览器日志。 - AgentDeskAI/browser-tools-mcp</li><li><a href="https://github.com/daniel-lxs/mcp-starter/pull/4">docs: 添加 Windows-sp

rexdotsh 提供的特定构建指令 · Pull Request #4 · daniel-lxs/mcp-starter</a>: 嗨，感谢这个伟大的工具！我添加了一些关于在 Windows 上构建时使用 -ldflags &quot;-H=windowsgui&quot; 的说明，以防止每次启动时弹出终端窗口...</li><li><a href="https://x.com/i/grok/share/ZwWdnR4SkIC2qjoljYogGIGqv">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>: 新的更新和改进。</li><li><a href="https://chat.qwen.ai/">通义千问 (Qwen Chat)</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1343631676680437821)** (935 messages🔥🔥🔥): 

> `Claude 3.7 性能、与其他模型的比较、Aider 功能与性能、Claude Code 评估、速率限制与 API 问题` 


- **Claude 3.7 在 Aider 排行榜上创下 SOTA**：Claude 3.7 Sonnet 在 Aider 排行榜上获得了 **65%** 的分数，使用了 **32k thinking tokens**，表现优于之前的模型。
   - 相比之下，非 thinking 版本得分为 **60%**，显示出显著提升，但一些用户认为在实际效用上的差异影响较小。
- **与其他模型的比较**：用户注意到 Claude 3.7 的性能略好于 Grok 3 等模型，并讨论了 **$15/M** 与 Grok 的 **$2.19/M** 之间的价格差异。
   - 一些用户表示失望，希望能有更大的性能差距，并强调成本可能无法支撑所看到的改进。
- **Aider 功能与性能**：Aider **0.75.1** 版本增加了对 Claude 3.7 的支持，允许用户在有效管理成本的同时尝试新功能。
   - 用户正在积极讨论如何改进 Aider 的各种命令（如 `--copy-paste`），使其在与 Claude 的工作流中更加高效。
- **Claude Code 评估**：Claude Code 界面与 Aider 进行了对比，用户对其易用性和功能（特别是在编辑和生成代码方面）褒贬不一。
   - 一些用户正在探索 Claude Code 设计选择的影响，以及它们如何有效地增强编程体验。
- **速率限制与 API 问题**：用户在使用包括 Claude 在内的各种 API 模型时遇到了速率限制问题，导致在编程任务中感到沮丧。
   - 在讨论使用替代平台的优缺点时，人们希望不同工具之间的性能和一致性能够得到提升。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://x.com/AnthropicAI/status/1894092430560965029">来自 Anthropic (@AnthropicAI) 的推文</a>：隆重推出 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一款混合推理模型，能够产生近乎即时的响应或扩展的逐步思考。一个模型，两种思考方式...</li><li><a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>：Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在现实世界的 Agent 任务中。它似乎代表了新的 State of the Art。</li><li><a href="https://aider.chat/docs/usage/copypaste.html">通过 Web 聊天进行复制/粘贴</a>：Aider 可与 LLM Web 聊天 UI 配合使用</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107?s=19">来自 adi (@adonis_singh) 的推文</a>：伙计，什么鬼，我只是问它有多少个 r，Claude Sonnet 3.7 竟然为我搭建了一个交互式学习平台让我自己去学 😂</li><li><a href="https://x.com/anthropicai/status/1894092430560965029?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Anthropic (@AnthropicAI) 的推文</a>：隆重推出 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一款混合推理模型，能够产生近乎即时的响应或扩展的逐步思考。一个模型，两种思考方式...</li><li><a href="https://aider.chat/docs/llms/warnings.html">模型警告</a>：Aider 是你终端里的 AI 结对编程工具</li><li><a href="https://console.anthropic.com/">Anthropic Console</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>：今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个正式发布的混合推理模型。</li><li><a href="https://www.anthropic.com/pricing#anthropic-api">定价</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://openrouter.ai/anthropic/claude-3-7-sonnet">Claude 3.7 Sonnet - API、提供商、统计数据</a>：Claude 3.7 Sonnet 是一款先进的大语言模型，具有改进的推理、编程和问题解决能力。通过 API 运行 Claude 3.7 Sonnet</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models#model-comparison-table">所有模型概览 - Anthropic</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>：未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking">使用扩展思维进行构建 - Anthropic</a>：未找到描述</li><li><a href="https://tenor.com/view/grito-ahhhh-hongo-gif-20006750">Grito Ahhhh GIF - Grito Ahhhh Hongo - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sweaty-speedruner-gif-20263880">Sweaty Speedruner GIF - Sweaty Speedruner - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/drago-ivan-i-must-break-you-rocky-break-warning-gif-11521068">Drago Ivan I Must Break You GIF - Drago Ivan I Must Break You Rocky - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.anthropic.com/en/api/rate-limits;">首页 - Anthropic</a>：未找到描述</li><li><a href="https://www.anthropic.com/contact-sales">联系 Anthropic</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://news.ycombinator.com/item?id=43163011)">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1343633346726465629)** (63 messages🔥🔥): 

> `Architect Mode Configuration, Managing AI Models in Aider, Using OpenRouter for AI Access, Token Management in AI, Integration with Git in Aider` 


- **理解 `/architect` 模式**：讨论集中在 Aider 的 `/architect` 模式如何使用不同的模型进行编辑和推理，特别是 `model: o1-preview` 和 `editor-model: o1-mini` 的配置。
   - 有人指出，为了高效地同时使用架构和编辑功能，在运行时使用 `/model` 切换模型是实现灵活性的理想选择。
- **探索 OpenRouter 的优势**：用户分享了关于 OpenRouter 服务的见解，强调了其在绕过速率限制（rate limits）和集中管理各种模型的 API key 方面的优势。
   - 虽然模型的定价与原始提供商相似，但通过一个接口访问多个 API 的便利性是一个显著的优势。
- **管理 Token 使用**：针对使用 `claude-3-7-sonnet-20250219` 等模型时的 Token 限制和管理提出了担忧，特别是在思维（thinking）和编辑（editing）模式之间切换时。
   - 参与者讨论了配置 Token 预算如何提高交互效率，同时有效地管理输入和输出 Token。
- **Aider 与 Git 的集成**：Aider 的 `/git` 命令被强调为跟踪远程仓库更改的有用功能，并建议使用 `/git status` 和 `/git pull` 命令进行频繁检查。
   - 对于自动更新，有人提议使用 bash 脚本定期运行 git 命令，尽管用户们仍在寻找内置的解决方案。
- **使用 Sonnet 3.7 和 Thinking 模式**：用户对于在使用 Aider 配合 Sonnet 3.7 时，如何有效地启用 thinking 模式同时在 editor 模型中禁用推理感到好奇。
   - 小组讨论了是否有办法将思维过程与编辑功能分开，特别是利用新 API 的特性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/modes.html#architect-mode-and-the-editor-model">Chat modes</a>：使用 code, architect, ask 和 help 聊天模式。</li><li><a href="https://openrouter.ai/openai/o3-mini-high">o3 Mini High - API, Providers, Stats</a>：OpenAI o3-mini-high 与 [o3-mini](/openai/o3-mini) 是同一个模型，但 reasoning_effort 设置为 high。o3-mini 是一款针对 STEM 推理任务优化的成本效益型语言模型，尤其擅长...</li><li><a href="https://aider.chat/2024/12/21/polyglot.html">o1 tops aider’s new polyglot leaderboard</a>：o1 在 Aider 新的多语言、更具挑战性的编码基准测试中获得了最高分。</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>：LLM 代码编辑技能的定量基准测试。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">Advanced model settings</a>：配置 LLM 的高级设置。</li><li><a href="https://github.com/Aider-AI/aider/blob/0ba1e8f90435aa2c08360d152fe8e16f98efd258/aider/coders/architect_coder.py#L21">aider/aider/coders/architect_coder.py at 0ba1e8f90435aa2c08360d152fe8e16f98efd258 · Aider-AI/aider</a>：aider 是你终端里的 AI 配对编程助手。通过在 GitHub 上创建账号为 Aider-AI/aider 做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1343685939846713436)** (2 messages): 

> `Hacker News Wrapped, Kagi LLM Benchmarking Project` 


- **Hacker News Wrapped 用于个人资料洞察**：Claude Sonnet 3.7 允许用户分析他们的 Hacker News 个人资料，提供极其准确且幽默的亮点和趋势分析。
   - 用户们表示有兴趣让这个 AI 对他们的个人资料进行“吐槽”（roasted），因为它具有极具洞察力且有趣的评论。
- **Kagi LLM 基准测试揭示新见解**：[Kagi LLM Benchmarking Project](https://help.kagi.com/kagi/ai/llm-benchmark.html) 通过不断变化的、新颖的任务评估主流大语言模型在推理、编码和指令遵循方面的表现，以避免基准测试过拟合。
   - 最后更新于 **2025年2月24日**，结果显示 **Google 的 gemini-2.0** 得分为 **60.78%**，而 **OpenAI 的 gpt-4o** 以 **48.39%** 落后。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/">HN Wrapped</a>：AI 分析你的 HN 个人资料并为你提供 2024 年回顾</li><li><a href="https://help.kagi.com/kagi/ai/llm-benchmark.html">Kagi LLM Benchmarking Project | Kagi's Docs</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1343629844789067806)** (15 条消息🔥): 

> `Vim 中的 Codeium Chat、账号创建问题、Codeium 扩展与支持、版本更新查询、问题的论坛可见性` 


- **Vim 中的 Codeium Chat 寻求帮助**：一名成员在通过 SSH 会话启动 Vim 中的 **Codeium Chat** 时遇到困难，在尝试访问提供的 URL 时收到连接错误。
   - 另一名成员询问该用户使用的是 Vim 还是 Neovim 扩展，随后建议他们在相关的扩展频道寻求帮助。
- **创建新 Codeium 账号遇到问题**：一名用户对无法创建新账号表示沮丧，因为遇到了内部错误消息。
   - 这引发了关于其他人是否在账号创建方面遇到类似问题的疑问。
- **关于 Codeium 扩展的澄清**：一名成员澄清说，该频道专注于适用于 Visual Studio Code 和 Neovim 等各种编辑器的 **Codeium extension**。
   - 讨论强调了将特定查询提交至 Codeium 网站上相应支持渠道的重要性。
- **查询 Codeium 3.7 版本发布**：有人对 **3.7** 版本的发布感到好奇，一名成员幽默地表示缺乏更新可能令人沮丧。
   - 另一名用户询问了在 JetBrains IDEs 中切换到 pre-release 版本的流程，并讨论了 stable 频道的限制。
- **寻求论坛以提高问题可见性**：一名成员考虑将他们的问题发到论坛以获得更高关注，随后被建议通过 **Codeium support** 页面进行联系。
   - 另一名成员提供了后续支持，以确保工单编号（ticket numbers）得到妥善处理。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1343629265371009164)** (675 条消息🔥🔥🔥): 

> `Claude 3.7 发布, Windsurf vs Cursor, MCP 问题, 电子商务经验, Laravel vs JavaScript 框架` 


- **Windsurf 用户热切期待 Claude 3.7**：用户对 Windsurf 推迟上线 **Claude 3.7** 表示沮丧，并指出 Cursor 用户已经获得了访问权限。
   - 社区讨论了 3.7 预期带来的潜在易用性改进和增强功能。
- **Windsurf vs Cursor 性能讨论**：许多用户观察到 **Cursor** 可能是更快的选择，因为与 Windsurf 相比，它在实现新模型方面速度更快。
   - 尽管 Cursor 速度更快，一些用户仍然因为其整洁的 UI 和集成能力而更倾向于 Windsurf。
- **MCP 相关问题**：关于 **MCP** 导致无法正常编辑功能的问题正在讨论中，这导致用户不得不临时手动编辑配置文件。
   - 社区正在分享这些问题的解决方案，同时等待支持团队的官方修复。
- **关于电子商务和框架的见解**：几位用户分享了在 **ecommerce** 方面的经验，强调了使用 **Laravel** 进行快速部署和稳定性能的优势。
   - 讨论转向了传统框架与现代 JavaScript 框架之间的偏好，并强调了各自的权衡。
- **关于 Claude Code 未来的思考**：一些用户询问将 **Claude Code** 与 Windsurf 集成的可能性，并质疑其与 IDE 的关系。
   - 有推测认为，如果进行集成，Claude Code 可能会如何补充 Windsurf 并增强开发流程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.anthropic.com">Home</a>: Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/all-models?utm_source=iterable&utm_medium=email&utm_campaign=sonnet_3-7_launch&campaignId=12703046&source=i_email&medium=email&content=Dec20241P&messageTypeId=140367">所有模型概览 - Anthropic</a>: 未找到描述</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>: 未找到描述</li><li><a href="https://tenor.com/bENEo.gif">Chewing Character Hd GIF - Chewing Character Chewing Character - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/good-juju-witch-good-vibes-sending-love-and-light-hive-gif-20508559">Good Juju Witch GIF - Good Juju Witch Good Vibes - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://tenor.com/view/stan-twitter-monkey-meme-monki-monke-monkey-waiting-gif-12661622482574205246">Stan Twitter Monkey Meme GIF - Stan twitter Monkey meme Monki - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/let-them-cook-let-them-fight-godzilla-godzilla-2014-meme-gif-10523835079864650811">Let Them Cook Let Them Fight GIF - Let them cook Let them fight Godzilla - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/surf-glider-wave-giant-wave-wind-gif-15418238">Surf Glider GIF - Surf Glider Wave - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://youtube.com/shorts/8TQaJDCw-dE?si=OP_Xx6cc-aOxpcB3">什么是 Vibe Coding? ☮️ 😎</a>: 加入我的时事通讯以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅: https://www.youtube.com/@matthew_berman 👉🏻 Twitter: https:/...</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变更。
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1343640915939561572)** (611 条消息🔥🔥🔥): 

> `Claude 3.7 vs ChatGPT, Grok 3 性能, AI 在编程中的应用, AI 在项目中的集成, 用于文档处理的 AI`

- **Claude 3.7 与 ChatGPT 模型的对比**：关于 Claude 3.7 的讨论显示，它在编程和 Web 开发任务中可能优于 ChatGPT 模型，而 ChatGPT 因在遵循特定指令方面的固执而受到批评。
   - 用户指出，Claude 保持了更好的连贯性和可靠性，尤其是在编程方面，这与表现力强但偶尔不稳定的 OpenAI 模型形成对比。
- **AI 在文档处理中的应用**：一位用户分享了使用 AI 为入籍申请汇总冗长旅行记录的挑战，鉴于其复杂性，寻求关于如何开始此类任务的建议。
   - 人们对利用 AI 自动化文档检索和汇总表现出兴趣，突显了 AI 在行政和官僚辅助方面的潜力。
- **Grok 3 的多功能性与性能**：Grok 3 的能力受到称赞，尤其是在编程方面，用户鼓励在 Copilot 等 IDE 中使用它以获得有效的项目协助。
   - 讨论了该模型在集成 Claude 3.7 功能的同时处理项目的能力，强调了其在前端开发方面的优势。
- **AI 行业洞察与趋势**：对话涉及了行业趋势，包括免费 AI 工具和平台的兴起，字节跳动（ByteDance）的 “Trae” 提供免费访问高级 AI 模型的权限，以此作为一种竞争策略。
   - 参与者注意到免费服务的临时性，并讨论了其对 AI 领域消费者获取途径和服务的影响。
- **对 AI 模型局限性的担忧**：用户对某些 AI 模型的局限性以及管理复杂任务（尤其是可视化编程）的挑战表示沮丧。
   - 普遍观点是，虽然取得了进步，但仍存在重大障碍，特别是在用户体验和与 AI 的直观交互方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alexalbert__/status/1894095781088694497">来自 Alex Albert (@alexalbert__) 的推文</a>：我们正在开放对我们正在构建的新 Agent 编程工具的有限研究预览访问：Claude Code。你将获得由 Claude 驱动的代码辅助、文件操作和任务执行功能...</li><li><a href="https://fxtwitter.com/apples_jimmy/status/1893835336913973438">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：明天复仇。</li><li><a href="https://x.com/anthropicai/status/1894092430560965029">来自 Anthropic (@AnthropicAI) 的推文</a>：推出 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎瞬时的响应或扩展的逐步思考。一个模型，两种思考方式。</li><li><a href="https://fxtwitter.com/i/status/1894106441536946235">来自 Rowan Cheung (@rowancheung) 的推文</a>：Anthropic 刚刚发布了 Claude 3.7 Sonnet，世界上最好的编程 AI 模型。我是早期测试者，它让我大受震撼。它通过一个提示词就创建了这个 Minecraft 克隆版，并且可以立即运行...</li><li><a href="https://llm-stats.com/">LLM 排行榜 2025 - 对比 LLM</a>：包含基准测试、定价和功能的综合 AI (LLM) 排行榜。通过交互式可视化、排名和对比来比较领先的 LLM。</li><li><a href="https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview">Claude Code 概览 - Anthropic</a>：未找到描述</li><li><a href="https://www.anthropic.com/research/visible-extended-thinking">Claude 的扩展思考</a>：讨论 Claude 的新思考过程</li><li><a href="https://rednuht.org/genetic_cars_2/">HTML5 遗传算法 2D 赛车小游戏 - 推荐使用 Chrome</a>：未找到描述</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>：今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://kodub.itch.io/polytrack">PolyTrack (作者：Kodub)</a>：一款高速低多边形赛车游戏。</li><li><a href="https://news.berkeley.edu/2021/10/18/so-called-junk-dna-plays-critical-role-in-mammalian-development/">所谓的垃圾 DNA 在哺乳动物发育中起关键作用 - 伯克利新闻</a>：新研究表明，很久以前入侵哺乳动物基因组的一些病毒已被重新利用，在发育中发挥关键作用。</li><li><a href="https://www.youtube.com/watch?v=TxANYMqd8cY">Broderbund Software - Stunts v1.1 [1991]</a>：Stunts（又名 4D Sports Driving）是 DOS 时代的一款伟大赛车游戏！如果你想再次玩它，请按照以下说明操作：1-) 获取游戏；2-) 使用 D...</li><li><a href="https://www.reddit.com/r/mlscaling/comments/146rgq2/chatgpt_is_running_quantized/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1343646225181835345)** (9 messages🔥): 

> `O3 推理问题，模型反馈差异，截图发布限制，Bug 报告流程` 


- **O3 推理遭遇延迟**：用户报告称 **O3** 经常显示 “reasoning success”，但在延迟长达 **10 秒**或更长时间后才交付全文。
   - 一位用户提到在 **EST 时间下午 3 点至 7 点**之间持续遇到这些问题，影响了所有的 “thinking” 模型。
- **模型响应不一致**：一位用户表达了他们的沮丧，因为他们在 **mobile** 端能收到预期的文本，但在 **browser** 端却收不到，反之亦然，通常导致完全没有响应。
   - 由于输出结果缺乏清晰度，在过去的 **10 天**里导致了无数次 **blank response**（空响应）反馈。
- **聊天频道中的截图限制**：一名成员询问为什么无法在当前聊天中添加截图，并引用了不同频道关于截图支持的设置差异。
   - 另一位用户建议在支持截图的其他频道发布信息，以便获得更好的帮助。
- **引导至 Bug 报告频道**：用户被引导如何发布新的 Bug 报告，并强调查看周围是否有类似问题可能会有所帮助。
   - 他们分享了一个频道的链接，该链接显示了报告 Bug 或对符合其经历的现有 Bug 进行评论的步骤。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1343628545712455793)** (345 messages🔥🔥): 

> `Unsloth AI 挑战赛，DeepSeek 发布，训练配置问题，Hyperfitting 与资源管理，自定义损失函数实现` 


- **Unsloth AI 挑战赛提交**：参与者可以为 Unsloth Challenge 提交多个解决方案，提交次数没有设定限制。目前已评审了约 30 份提交，但尚未有人达到录用门槛。
   - 会议还讨论了鼓励为挑战赛提供推荐，以帮助寻找合格的候选人。
- **新 DeepSeek 发布及其影响**：第二天发布的 DeepSeek OSS 引入了 MoE kernels 和高效通信库等特性，一些人认为这些特性更多是面向大公司而非个人开发者。关于这种开放性将如何影响竞争和 AI 发展格局存在各种推测。
   - 人们对新工具既感到兴奋又持怀疑态度，特别是关于它们如何降低运行模型的成本。
- **训练配置问题**：在从 batch size 8 切换到带有 gradient accumulation 的 batch size 1 时，关于 validation loss 的差异产生了困惑。一位参与者指出，尽管使用了相同的输入数据，他们的 validation loss 却出乎意料地显著增加。
   - 几位参与者讨论了配置，以及混合设置策略在模型学习方式方面是否会产生类似的结果。
- **Hyperfitting 与资源管理**：讨论强调了在较小模型上使用 Hyperfitting 技术的潜力，以及训练时资源需求的含义。参与者反思了他们对使用有限 VRAM 和常用训练技术时模型容量的预期。
   - 对话涉及了不断演进的训练实践如何可能以更低的成本提高性能指标。
- **自定义损失函数实现**：有人询问关于使用自定义损失函数将 regularization term（正则化项）引入现有模型的问题。参与者鼓励探索实现策略，以有效地达到预期结果。
   - 重点放在了损失函数的修改如何影响模型的学习能力，特别是在 fine-tuning 场景中。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/danielhanchen/status/1894212351932731581">来自 Daniel Han (@danielhanchen) 的推文</a>：DeepSeek 第二个 OSS 发布！MoE 内核、专家并行（expert parallelism）、训练和推理均支持 FP8！引用 DeepSeek (@deepseek_ai) 🚀 #OpenSourceWeek 第二天：DeepEP 很高兴介绍 DeepEP —— 第一个...</li><li><a href="https://huggingface.co/BarraHome/llama3.2-1b-mla">BarraHome/llama3.2-1b-mla · Hugging Face</a>：未找到描述</li><li><a href="https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html">如何通过将 optimizer step 融合到 backward pass 中来节省内存 — PyTorch 教程 2.6.0+cu124 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/guinea-pig-chewing-chew-cavy-bertold-gif-13907739970483938206">豚鼠咀嚼 GIF - Guinea pig Chewing Chew - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 — PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://github.com/fxmeng/TransMLA">GitHub - fxmeng/TransMLA: TransMLA: Multi-Head Latent Attention Is All You Need</a>：TransMLA: Multi-Head Latent Attention Is All You Need - fxmeng/TransMLA</li><li><a href="https://github.com/vllm-project/vllm/tree/db986c19ea35d7f3522a45d5205bf5d3ffab14e4/benchmarks">vllm/benchmarks (位于 db986c19ea35d7f3522a45d5205bf5d3ffab14e4) · vllm-project/vllm</a>：一个面向 LLM 的高吞吐量且内存高效的推理与服务引擎 - vllm-project/vllm</li><li><a href="https://docs.unsloth.ai/basics/reasoning-grpo-and-rl">推理 - GRPO & RL | Unsloth 文档</a>：使用 Unsloth 通过 GRPO（强化学习 RL 微调的一部分）训练你自己的 DeepSeek-R1 推理模型。</li><li><a href="https://docs.unsloth.ai/get-started/fine-tuning-guide">微调指南 | Unsloth 文档</a>：学习微调的所有基础知识。</li><li><a href="https://arxiv.org/abs/2502.14837">迈向经济化推理：在任何基于 Transformer 的 LLM 中启用 DeepSeek 的 Multi-Head Latent Attention</a>：Multi-head Latent Attention (MLA) 是由 DeepSeek 提出的一种创新架构，旨在通过将 Key-Value (KV) 缓存显著压缩为...</li><li><a href="https://github.com/JT">jt - 概览</a>：jt 有 4 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/vllm-project/aibrix">GitHub - vllm-project/aibrix: 用于 GenAI 推理的高性价比且可插拔的基础设施组件</a>：用于 GenAI 推理的高性价比且可插拔的基础设施组件 - vllm-project/aibrix</li><li><a href="https://github.com/facebookresearch/optimizers/tree/main">GitHub - facebookresearch/optimizers: 用于优化算法的研究与开发。</a>：用于优化算法的研究与开发。 - facebookresearch/optimizers</li><li><a href="https://github.com/JT-Ushio/MHA2MLA">GitHub - JT-Ushio/MHA2MLA: 迈向经济化推理：在任何基于 Transformer 的 LLM 中启用 DeepSeek 的 Multi-Head Latent Attention</a>：迈向经济化推理：在任何基于 Transformer 的 LLM 中启用 DeepSeek 的 Multi-Head Latent Attention - JT-Ushio/MHA2MLA
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 条消息): 

deoxykev: 新的 qwq https://qwenlm.github.io/blog/qwq-max-preview/
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1343643545554260069)** (121 messages🔥🔥): 

> `Unsloth on Mac, CUDA Memory Issues, Using Custom Datasets, Checkpointing and VRAM Management, Custom Loss Functions` 


- **Unsloth 与 Mac 的兼容性**：关于在 Mac 上运行 Unsloth 的讨论正在进行中。指出虽然模型可以运行，但 Unsloth 目前尚不支持 Mac，因此建议使用外部 GPU 扩展坞或租用 GPU 服务器等替代方案。
   - 有用户提到使用 Tensordock 进行廉价的 GPU 服务器租用，推荐方案还包括 Google 的免费资源。
- **解决 CUDA 内存错误**：用户在训练期间遇到了 'illegal memory access'（非法内存访问）错误，建议检查 CUDA 安装并确保 PyTorch 能够检测到 GPU。
   - 建议在 Python 中运行特定命令以验证 CUDA 的可用性和 PyTorch 版本。
- **自定义数据集格式化**：一位新用户询问了关于自定义数据集的正确格式指南，确认数据集应与 Notebooks 中引用的 ShareGPT 格式保持一致。
   - 鼓励用户参考 Notebooks 作为指南，这表明数据集结构具有一定的灵活性。
- **Checkpointing 期间的 VRAM 管理**：讨论集中在训练期间对 VRAM 管理的需求，特别是模型保存操作期间显存使用的显著峰值。
   - 用户建议将模型保存为 LoRA 以释放 VRAM，并发现 Checkpointing 过程有时也会表现出类似的行为。
- **实现自定义损失函数**：一位用户询问如何集成包含正则化项的自定义损失函数，显示出对在 Unsloth 框架内修改损失函数指导的需求。
   - 这表明用户有兴趣在 Unsloth 框架内优化模型训练过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>：以下是我们所有 Notebooks 的列表：</li><li><a href="https://search.app/YgSmHDHmwPcJubBH6">Installing + Updating | Unsloth Documentation</a>：学习如何在本地或在线安装 Unsloth。</li><li><a href="https://github.com/unslothai/unsloth/issues/685">Unsloth On Mac · Issue #685 · unslothai/unsloth</a>：我有一台 Macbook，当我运行模型时，基本会报错说找不到 CUDA 设备。我知道 Macbook 没有 GPU，这是否意味着我无法在 M... 上运行模型？
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1343669893991628830)** (1 messages): 

> `Claude 3.7 Sonnet, AI capabilities improvement, Pricing model for Claude, Extended Thinking feature` 


- **Claude 3.7 Sonnet 发布**：**Claude 3.7 Sonnet** 现已在 OpenRouter 上线，这标志着 AI 的重大进步，特别是在**数学推理**、**编程**和**复杂问题解决**方面。完整详情请查看 [博客文章](https://www.anthropic.com/news/claude-3-7-sonnet)。
   - 此版本在 Agent 工作流方面引入了显著增强，并允许用户在快速推理和 Extended Thinking 过程之间进行选择。
- **Claude 3.7 的定价与使用**：**Claude 3.7 Sonnet** 的定价设定为每百万输入 Token **3 美元**，每百万输出 Token **15 美元**（包含 thinking tokens）。该定价结构旨在优化用户在使用模型时的参与度。
   - 此次发布包含完整的缓存支持，以增强使用过程中的性能。
- **即将推出的 Extended Thinking 功能**：即将推出的 **Extended Thinking** 功能将集成到 OpenRouter API 中，进一步增强 Claude 模型的能力。更多信息请参阅 [Extended Thinking 文档](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)。
   - 该功能预计将为用户提供处理复杂任务的高级选项，进一步提升模型的可用性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking),">Home - Anthropic</a>：未找到描述</li><li><a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>：Claude 3.7 Sonnet 是一款先进的大语言模型，具有改进的推理、编程和问题解决能力。通过 API 运行 Claude 3.7 Sonnet。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1343629628564049930)** (346 条消息🔥🔥): 

> `Claude 3.7 Sonnet 功能、API Key 管理、跨设备对话续接、模型定价与性能、Claude 3.7 中的图像处理` 


- **探索 Claude 3.7 的 Extended Thinking**：参与者讨论了 **Claude 3.7 Sonnet** 的能力，确认 Extended Thinking 功能尚未完全实现，目前正等待文档说明和支持。
   - 混合模型选择思考层级的潜力令人兴奋，这展示了其先进的推理能力。
- **管理 API Key 和额度 (Credits)**：用户对丢失 API Key 及其与账户额度的关系表示担忧，得到的答复是无论 Key 状态如何，额度都保持不变。
   - 可以生成新 Key 而不会丢失额度，因为额度是绑定在账户上而非 Key 本身。
- **设备同步限制**：会上指出 **OpenRouter** 目前不支持跨设备同步对话会话（例如从浏览器同步到手机）。
   - 建议使用 *chatterui* 和 *typing mind* 等替代方案来弥补这一差距。
- **图像上传问题**：有报告称 **Claude 3.7** 频繁报错，特别是与超过 5MB 的图像大小有关，这导致请求无法完成。
   - 鼓励用户管理图像大小以符合 API 要求，避免请求失败。
- **模型成本对比**：参与者将 Claude 的定价结构与其他模型进行了对比，指出虽然价格被认为偏高，但在某些性能领域具有竞争力。
   - 对话强调了成本与可用性之间的权衡，特别是围绕推理任务及其对总支出的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/anthropic/claude-3.7-sonnet">Claude 3.7 Sonnet - API, Providers, Stats</a>：Claude 3.7 Sonnet 是一款先进的大型语言模型，具有改进的推理、编程和问题解决能力。通过 API 运行 Claude 3.7 Sonnet。</li><li><a href="https://tenor.com/view/ponke-ponkesol-solana-sol-bored-gif-1576815656973460219">Ponke Ponkesol GIF - Ponke Ponkesol Solana - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://aws.amazon.com/ai/machine-learning/trainium/">AI Accelerator - AWS Trainium - AWS</a>：未找到描述</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet and Claude Code</a>：今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://docs.anthropic.com/en/docs/about-claude/models/extended-thinking-models">Extended thinking models - Anthropic</a>：未找到描述</li><li><a href="https://tenor.com/view/telmo-coca-harina-raquetaso-esnifar-gif-25660568">Telmo Coca GIF - Telmo Coca Harina - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.cnbc.com/2025/01/22/google-agrees-to-new-1-billion-investment-in-anthropic.html">Google agrees to new $1 billion investment in Anthropic</a>：据知情人士向 CNBC 确认，谷歌已同意向生成式 AI 初创公司 Anthropic 追加超过 10 亿美元的投资。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1343628723240308746)** (304 条消息🔥🔥): 

> `Claude 3.7 Sonnet 发布、Qwen AI 进展、Agentic 编程工具、开源倡议、RLHF 与代码质量`

- **Claude 3.7 Sonnet 以新功能令人印象深刻**：Claude 3.7 Sonnet 已经发布，具备改进的推理能力和交互灵活性，其在编程任务和调试方面的表现备受关注。
   - 用户发现，虽然它提供了详细的重构，但有时可能会导致比预期更复杂的代码，对常见的编程范式提出了挑战。
- **Qwen AI 的 QwQ-Max Preview**：Qwen 最近发布的 QwQ-Max-Preview 展示了其推理能力，承诺在 Apache 2.0 协议下提供改进的性能和开源可访问性。
   - 该版本的发布伴随着对其创造性问题解决能力以及在各个领域潜在应用的赞誉。
- **Agentic 编程工具的新兴趋势**：Claude Code 和 DeepEP 等工具的推出标志着向更复杂的编程辅助工具的转变，旨在简化开发流程。
   - 随着竞争加剧，各大实验室提供的产品正在推向编程模型所能实现的极限，包括专注于效率和灵活性的功能。
- **开源周亮点**：DeepSeek 宣布了其开源的 EP 通信库 DeepEP，专注于模型训练的高效通信，标志着对 AI 开发协作的承诺。
   - 这加强了 AI 技术开源的趋势，允许更广泛的社区参与和创新。
- **关于 RLHF 和代码生成的疑问**：讨论者正在思考人类反馈强化学习 (RLHF) 如何影响代码质量，并担心模型可能更倾向于复杂的“具有生产感”的代码，而非简洁的代码。
   - 这一讨论提高了人们对训练编程模型根本目标的认识，以及输出质量中潜在的差异。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/ChujieZheng/status/1894095584774250858">来自 Chujie Zheng (@ChujieZheng) 的推文</a>：你在开玩笑吗，伙计？</li><li><a href="https://x.com/cursor_ai/status/1894093436896129425">来自 Cursor (@cursor_ai) 的推文</a>：Sonnet-3.7 已在 Cursor 中上线！我们对其编程能力印象深刻，尤其是在真实世界的 Agent 任务上。它似乎成为了新的 SOTA。</li><li><a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>：未找到描述</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37">新一代 AI：Claude 3.7 和 Grok 3</a>：是的，AI 突然变得更强了……又一次</li><li><a href="https://www.oneusefulthing.org/p/a-new-generation-of-ais-claude-37#footnote-1-157729795">新一代 AI：Claude 3.7 和 Grok 3</a>：是的，AI 突然变得更强了……又一次</li><li><a href="https://www.anthropic.com/research/visible-extended-thinking">Claude 的扩展思维</a>：讨论 Claude 的新思考过程</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130619061604651">来自 Qwen (@Alibaba_Qwen) 的推文</a>：Agent</li><li><a href="https://x.com/din0s_/status/1894102686984818863">来自 dinos (@din0s_) 的推文</a>：一张截图看尽 Anthropic</li><li><a href="https://x.com/TheXeophon/status/1894113897797288215">来自 Xeophon (@TheXeophon) 的推文</a>：Sonnet 3.7 Thinking（设置 16K tokens 预算）是 Neal 密码游戏中表现最好的模型，恭喜！它*几乎*通过了第 11 关，但它坚持认为 Wordle 已经解开了 :( 引用 Xeophon (@The...</li><li><a href="https://x.com/DimitrisPapail/status/1894127499224694877">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：Claude 3.7 Sonnet 请用 tikz 绘制雅典卫城，结果包含：- 无推理 - 10k 推理 tokens - 30k 推理 tokens - 64k 推理 tokens。引用 Dimitris Papailiopoulos (@Dimitri...</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1894130603513319842">来自 Qwen (@Alibaba_Qwen) 的推文</a>：&lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview。Qwen Chat：https://chat.qwen.ai/ 博客：https://qwenlm.github.io/blog/qwq-max-preview/ 🤔 今天我们在 Qwen Chat 中发布了由 o... 支持的“Thinking (QwQ)”</li><li><a href="https://x.com/skcd42/status/1894098856805372378">来自 skcd (@skcd42) 的推文</a>：作为曾使用过它的用户对新 Sonnet 3.7 的评价：- 新的 Sonnet 非常棒，在我们针对 Rust 的内部评估中，我们看到了 14.7%（约 40%）的提升（该评估由 1k 个问题组成）- 它具有……</li><li><a href="https://x.com/StringChaos/status/1894135561059013023">来自 Naman Jain (@StringChaos) 的推文</a>：查看 QwQ-Max-Preview 在 LiveCodeBench 上的评估结果，它的表现与 o1-medium 旗鼓相当 🚀！！引用 Qwen (@Alibaba_Qwen) &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat...</li><li><a href="https://x.com/lmarena_ai/status/1894128271568126381">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：祝贺 @AnthropicAI 发布 Claude 3.7 Sonnet！👏 快来 lmarena 用你最难的提示词测试它吧！引用 Anthropic (@AnthropicAI) 介绍 Claude 3.7 Sonnet：我们最智能的……</li><li><a href="https://x.com/arankomatsuzaki/status/1894101923151692157">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：Claude 3.7 Sonnet System Card 刚刚发布！</li><li><a href="https://x.com/DimitrisPapail/status/1894144311232729391">来自 Dimitris Papailiopoulos (@DimitrisPapail) 的推文</a>：好吧，这真的很酷。Claude 3.7 Sonnet 请用 tikz 绘制：- 一个在房子里的人 - 房子内接于球体 - 球体内接于立方体 - 立方体内接于圆柱体……</li><li><a href="https://x.com/_lewtun/status/1894098741046521904">来自 Lewis Tunstall (@_lewtun) 的推文</a>：终于有一家 AI 实验室发布了带有正确标签和所有信息的图表 🥹</li><li><a href="https://pivot-to-ai.com/2025/02/22/google-co-scientist-ai-cracks-superbug-problem-in-two-days-because-it-had-been-fed-the-teams-previous-paper-with-the-answer-in-it/">Google Co-Scientist AI 在两天内破解了超级细菌难题！—— 因为它被喂了团队之前包含答案的论文</a>：Google 惊人的新 AI 工具 Co-Scientist（基于 Gemini LLM）的炒作周期包括一条 BBC 新闻，报道了帝国理工学院的 José Penadés 团队如何向该工具询问一个问题……</li><li><a href="https://x.com/adonis_singh/status/1894100291345150107">来自 adi (@adonis_singh) 的推文</a>：伙计，什么情况？我只是问它有多少个 r，Claude Sonnet 3.7 竟然为我搭建了一个交互式学习平台让我自己去学 😂</li><li><a href="https://x.com/nrehiew_/status/1894105060759552231">来自 wh (@nrehiew_) 的推文</a>：具体来说，API 中的“thinking budget tokens”参数似乎在预算耗尽之前永远不会采样到思考结束（end-of-thinking）token。没有提示词调节，没有特……</li>

<li><a href="https://x.com/paulgauthier/status/1894123992505880688">来自 Paul Gauthier (@paulgauthier) 的推文</a>：Claude 3.7 Sonnet 在不开启 thinking 的情况下在 aider polyglot benchmark 中获得了 60% 的分数。与 o3-mini-high 并列第三。Sonnet 3.7 拥有最高的非思考得分（此前为 Sonnet 3.5）。Thinking 模式的结果即将发布...</li><li><a href="https://x.com/nearcyan/status/1894103654874984906">来自 near (@nearcyan) 的推文</a>：CLAUDE 来了！他回来了，而且比以往任何时候都更好！我将分享我的第一个 Prompt 结果之一，这是一个微调音乐的 3D 可视化。这是目前世界上最好的模型。许多人...</li><li><a href="https://x.com/elder_plinius/status/1894110867353899112">来自 Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>：🚂 越狱警报 🚂 ANTHROPIC: 被攻破 ✌️😛 CLAUDE-SONNET-3.7: 已解放 🗽 哇，新的 Claude 模型！！！🤗 你们知道吗，我大约在...写的原始“GODMODE”通用越狱方法...</li><li><a href="https://x.com/btibor91/status/1894113852301721645">来自 Tibor Blaho (@btibor91) 的推文</a>：“目前仅在美国境内发货。” :( 引用 wh (@nrehiew_)：在 Claude Code NPM 源码中有一个“隐藏彩蛋”工具，可以向用户寄送 Anthropic 贴纸 :)</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>：今天，我们发布了 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个正式发布的混合推理模型（hybrid reasoning model）。</li><li><a href="https://youtu.be/t3nnDXa81Hs">具有扩展思考能力的 Claude 3.7 Sonnet</a>：介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的...</li><li><a href="https://x.com/cognition_labs/status/1894125030583537974">来自 Cognition (@cognition_labs) 的推文</a>：1/ Claude 3.7 Sonnet 已在 Devin 中上线！这个新模型是我们在各种任务中见过的表现最好的，包括调试、代码库搜索和 Agent 规划（agentic planning）。</li><li><a href="https://www.anthropic.com/news/visible-extended-thinking">Claude 的扩展思考</a>：讨论 Claude 的新思考过程</li><li><a href="https://www.youtube.com/watch?v=3Q25sogi-xo">It’s RAAAAAAW 超剪辑（200 万订阅者特别节目）| 地狱厨房</a>：IT’S RAAAAAAW!!! 为了庆祝频道达到 200 万订阅者，我们剪辑了 Ramsay 主厨每一次因为生肉或生鱼而发火的瞬间...</li><li><a href="https://x.com/deepseek_ai/status/1894211757604049133">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 #OpenSourceWeek 第 2 天：DeepEP。很高兴介绍 DeepEP —— 首个用于 MoE 模型训练和推理的开源 EP 通信库。✅ 高效且优化的 all-to-all 通信...</li><li><a href="https://x.com/AnthropicAI/status/1894095494969741358">来自 Anthropic (@AnthropicAI) 的推文</a>：我们对模型的安全性、防护性和可靠性进行了广泛测试。我们也听取了你们的反馈。在 Claude 3.7 Sonnet 中，与之前的版本相比，我们将不必要的拒绝减少了 45%...</li><li><a href="https://lovattspuzzles.com/online-puzzles-competitions/daily-cryptic-crossword/).">玩 Lovatts 免费在线加密填字游戏 - 每日更新</a>：Lovatts 免费在线加密填字游戏每日更新。包括 7 天谜题存档、提示和计时器。学习加密填字游戏的规则。</li><li><a href="https://techcrunch.com/2025/02/24/meta-ai-arrives-in-the-middle-east-and-africa-with-support-for-arabic/">Meta AI 登陆中东和非洲，支持阿拉伯语 | TechCrunch</a>：Meta AI 已在中东和北非上线并支持阿拉伯语，向数千万人开放了聊天机器人。</li><li><a href="https://news.ycombinator.com/item?id=43163488">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1343670611070816286)** (15 条消息🔥): 

> `Berkeley Advanced Agents MOOC, 招聘公告, RLHF 解释, 贴纸讨论, AI 初创公司客户群` 


- **Berkeley Advanced Agents 欢迎 Hanna Hajishirzi**：**Berkeley Advanced Agents** MOOC 的直播环节今天 **下午 4 点 PST** 邀请了 **Hanna Hajishirzi** 讨论 **Tulu 3**。请在 [YouTube Live](https://www.youtube.com/live/cMiu3A7YBks) 上观看。
   - 许多参与者称赞该课程到目前为止内容非常丰富。
- **Kadoa 正在招聘！**：一位成员分享了 Kadoa 的招聘页面链接，表明他们正在积极寻找新团队成员。点击[此处](https://hn-wrapped.kadoa.com/Philpax?share)查看他们的职位发布。
   - 随附了一条关于生成 HN Wrapped 的俏皮备注。
- **向外行解释 RLHF**：分享了一条推文，展示了如何向非技术受众解释 **RLHF (Reinforcement Learning from Human Feedback)**。所使用的类比被描述为出奇地有效。
   - 成员们表达了他们的兴趣，其中一位指出这确实是一个很好的类比。
- **贴纸邮寄难题**：成员们讨论了获取贴纸的困难，特别是对于目前在美国境外的成员。一位成员幽默地表示，如果寄过来，他会负责处理贴纸。
   - 这段轻松的交流还包括了关于贴纸是否能快速寄往纽约市的想法。
- **AI 初创公司的客户覆盖范围**：一位成员提出了一个问题，即 AI 初创公司在硅谷之外是否有客户，随后观察了他们声称在 Fortune 500 中的渗透率。这引发了关于 AI 实验室实际客户群的进一步讨论。
   - 另一位成员提到，重点应该放在他们对科技行业以外领域的拓展上。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://hn-wrapped.kadoa.com/Philpax?share">HN Wrapped</a>: AI 分析你的 HN 个人资料并为你提供 2024 年回顾</li><li><a href="https://fxtwitter.com/shaneguML/status/1894131091872891385">Shane Gu (@shaneguML) 的推文</a>: 我如何向非技术受众解释 RLHF</li><li><a href="https://x.com/AndrewCurran_/status/1894152685429108846">Andrew Curran (@AndrewCurran_) 的推文</a>: @repligate 还有这个；</li><li><a href="https://www.youtube.com/live/cMiu3A7YBks">CS 194/294-280 (Advanced LLM Agents) - 第 4 讲, Hanna Hajishirzi</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1343663805409792132)** (1 条消息): 

> `图像分析, Discord Bot 互动` 


- **分享图像分析**：一位用户分享了一张需要详细关注其内容的[分析](https://cdn.discordapp.com/attachments/1187551504995987576/1343663805493940285/CleanShot_2025-02-24_at_20.20.03.png)图像。
   - 这引发了成员们关于此类图像在他们正在进行的项目背景下的相关性和解释的讨论。
- **与 Discord Bot 的互动**：成员们继续与 Discord bot 互动，使用表情符号对最近的更新做出反应并表达感受。
   - 这展示了社区的互动性质以及他们对 bot 功能的赞赏。


  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/)** (1 条消息): 

0x_paws: https://x.com/srush_nlp/status/1894039989526155341?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1343693082352291983)** (3 条消息): 

> `新帖子 GIF, SnailBot` 


- **新帖子 GIF 引起关注**：分享了一个色彩鲜艳的 GIF，黑色背景上有多个字母显示 **'new post'**，这对社区很有吸引力。
   - 还提供了各种搜索相关 GIF 的链接，增强了可发现性。
- **SnailBot 被提及**：一位成员在频道中标记了 SnailBot，可能是为了向小组发出提醒或更新消息。
   - 另一位成员幽默地评论了 SnailBot 的速度，将其比作蜗牛并感叹 *'今天真快 (dayum fast today)'*，为对话增添了幽默感。



**提到的链接**: <a href="https://tenor.com/oPImCf3JDt3.gif">New New Post GIF - New New post Post - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1343650409654259855)** (29 条消息🔥): 

> `SoTA gemv kernels, 使用 Tensor Cores, 排行榜与提交, 内存受限操作 (Memory Bound Operations), Kernel 提交中的数据类型` 


- **关于 SoTA gemv kernels 的讨论**：一位成员分享了 **mobiusml/gemlite** 的 [GitHub 链接](https://github.com/mobiusml/gemlite)，其中包含使用 **Triton** 编写的快速低比特 matmul kernel。
   - 他们讨论了 gemv 的内存受限（memory-bound）特性以及对计算性能的预期。
- **关于 Tensor Cores 使用的疑问**：在对话中，有人质疑 gemv 为何不使用 Tensor Cores，一位成员指出 **tensorcores** 仅能增加 flops，而 gemv 是**内存受限（memory-bound）**的。
   - 另一位成员强调 Tensor Cores 要求张量维度至少为 **16x16**。
- **提交 .cu 文件的问题**：一位用户询问如何提交 **.cu** 文件，因为难以找到相关的 **cuda** 排行榜名称。
   - 建议在 Python 中使用 [load_inline 方法](https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/thunderkittens_example/submission.cu) 来提交此类文件。
- **Kernel 中数据类型的澄清**：一位用户对 `generate_input()` 函数在直方图（histogram）提交中返回 **float32** 而非 **uint8** 输入表示担忧。
   - 他们指出由于预期数据类型与实际数据类型不匹配导致的断言失败（assertion failure），提醒注意潜在的不一致性。
- **新成员介绍**：一位新成员介绍了自己，提到了他们的硬件加速设计课程，以及社区资源如何帮助他们学习 **GPUs**。
   - 他们寻求关于是否有专门用于研究相关查询的服务器的指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gpu-mode.github.io/discord-cluster-manager/docs/active">活跃排行榜 | GPU MODE Kernel Leaderboard</a>: 这里将为官方排行榜上的排行榜创建者提供提供更多元数据和信息的选项</li><li><a href="https://github.com/gpu-mode/reference-kernels">GitHub - gpu-mode/reference-kernels: 排行榜参考 Kernel</a>: 排行榜的参考 Kernel。通过在 GitHub 上创建账户为 gpu-mode/reference-kernels 做出贡献。</li><li><a href="https://github.com/gpu-mode/discord-cluster-manager/blob/main/examples/thunderkittens_example/submission.cu">discord-cluster-manager/examples/thunderkittens_example/submission.cu (main 分支) · gpu-mode/discord-cluster-manager</a>: 编写一个快速 kernel 并在 Discord 上运行。看看你与最强者的对比！ - gpu-mode/discord-cluster-manager</li><li><a href="https://github.com/mobiusml/gemlite">GitHub - mobiusml/gemlite: Triton 编写的快速低比特 matmul kernel</a>: Triton 编写的快速低比特 matmul kernel。通过在 GitHub 上创建账户为 mobiusml/gemlite 做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1343637777878618173)** (2 条消息): 

> `TorchAO 中的 E2E 示例` 


- **TorchAO 中 E2E 示例的进展**：@drisspg 确认他们正在开发 **TorchAO 中的 E2E 示例**，并会在准备就绪时通知。
   - Zippika 表达了热情并致谢：“oh awesome thanks!”。
- **协作请求**：对话显示了一项协作努力，@drisspg 表示在示例准备好后愿意进一步交流。
   - 这突显了 TorchAO 社区讨论的协作性质。


  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1343693675028418683)** (1 条消息): 

> `Q 和 K 中的正交矩阵，Hadamard 矩阵效率，向量量化` 


- **正交矩阵在 Q 和 K 中相互抵消**：当在 **Q** 和 **K** 中同时添加一个**正交矩阵**时，由于正交性的性质，它会相互抵消，从而得到等式 `(HQ)^T (HK) = Q^T K`。
   - *这适用于任何正交矩阵*，确保变换保留了被操作向量的核心特征。
- **Hadamard 矩阵提供结构化乘法**：使用 **Hadamard 矩阵** 可以实现比朴素 **O(n³)** 算法更快的乘法，从而提高计算效率。
   - *这些矩阵还有助于平滑具有较大离群值的向量*，增强后续计算的数值稳定性。
- **通过正交变换增强量化**：应用正交变换使 `HQ` 更适合进行 **Quantization**（量化），解决了原始 **Q** 可能出现的问题。
   - 这种改进表明，通过正交变换对数据进行结构化处理，可以在相关应用中获得更好的性能。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1343702410727788625)** (4 条消息): 

> `Linear Attention, DeepEP 通信库, 文档外 PTX 指令` 


- **Linear Attention 交互式教程**：名为 ["Linear Attention and Beyond"](https://www.youtube.com/watch?v=d0HJvGSWw8A) 的 YouTube 视频由 **Flash Linear Attention** 库的作者 Songlin Yang 主讲，提供了一个交互式教程。
   - 观众可以访问 [教程中使用的幻灯片](https://sustcsonglin.github.io/assets/pdf/talk_250117.pdf) 以获取更多见解。
- **DeepEP：高效的 Expert-Parallel 通信**：**deepseek-ai** 的 GitHub 仓库 [DeepEP](https://github.com/deepseek-ai/DeepEP) 展示了一个高效的 Expert-Parallel（专家并行）通信库。
   - 它旨在提升分布式系统的通信性能，可以通过其主要的 [描述页面](https://opengraph.githubassets.com/da80b7c85345bad48d96842a395271356d13599b7d1b711103900a4930e57902/deepseek-ai/DeepEP) 进行了解。
- **探索文档外 PTX 指令以提升性能**：一位成员分享了对使用文档外（out-of-doc）**PTX 指令** `ld.global.nc.L1::no_allocate.L2::256B` 的兴趣，以实现极致的性能提升。
   - 另一位成员推测，它可能通过在 L1 上避免全局加载（global loads）而在 L2 上允许加载来配置缓存行为。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/deepseek-ai/DeepEP">GitHub - deepseek-ai/DeepEP: DeepEP: an efficient expert-parallel communication library</a>：DeepEP：一个高效的专家并行通信库 - deepseek-ai/DeepEP</li><li><a href="https://www.youtube.com/watch?v=d0HJvGSWw8A">Linear Attention and Beyond (Songlin Yang 交互式教程)</a>：影响力巨大的 Flash Linear Attention https://github.com/fla-org/flash-linear-attention 库的作者 Songlin Yang 加入我并进行了一次交互式...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1343688981862154362)** (3 条消息): 

> `PyTorch Partner Engineer 职位, AI 协作, 平等就业机会, 社区与系统工作` 


- **Meta 招聘 PyTorch Partner Engineer**：Meta 宣布了两个 **PyTorch Partner Engineer** 的空缺职位，他们将与 PyTorch 团队及领先的行业合作伙伴紧密合作，以增强 AI 解决方案。
   - 该职位强调协作，并提供了一个将 **AI 技术** 从研究推向实际应用的机会。
- **强调协作机会**：一位成员指出，该职位非常适合那些对 **Systems**（系统）和 **Community**（社区）工作都感兴趣的人，并提到了他们与团队成员的协作经验。
   - 这一观点强调了团队合作在推动 Meta AI 领域项目中的影响力和重要性。
- **对平等就业机会的承诺**：Meta 致力于推进 **Equal Employment Opportunity**（平等就业机会），确保不因种族、性别和残疾等各种特征而产生歧视。
   - 公司还在招聘过程中为多元化候选人提供 **Reasonable Accommodations**（合理便利），确保招聘的包容性。



**提到的链接**：<a href="https://www.metacareers.com/jobs/646048097914422/?fbclid=IwY2xjawIpvmZleHRuA2FlbQIxMQABHTM77KSYiroHM5H1QoaX5jRZbTX9OtMhRSz5_XHYqg58Q34_aiShTT3Msg_aem__7DjKtddotzSJ0RLKk5B4g">Partner Engineer, PyTorch</a>：Meta 的使命是构建人类连接的未来以及使之成为可能的技术。

  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1343631609651531948)** (3 messages): 

> `GPU 编程入门，学习 PyTorch，探索 Triton` 


- **新手寻求 GPU 编程建议**：一位新成员表达了学习 **GPU programming** 的兴趣，并提到他们有使用 **PyTorch** 和 **JAX** 等 **ML frameworks** 的经验。
   - *他们的目标是掌握基础知识并编写 kernel 以供消遣*，同时也愿意学习 **C** 或 **C++**。
- **建议从 PyTorch 开始**：另一位成员建议从 [这里](https://pytorch.org/tutorials/) 提供的 **PyTorch tutorials** 开始，并强调了它相比 **JAX** 的易用性。
   - 他们建议在完成 Discord 服务器中提供的 **Triton puzzles** 的同时，熟悉 **Triton**。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1343792689719541831)** (1 messages): 

> `图像模糊，C++ vs CUDA 性能` 


- **图像模糊处理中 CUDA 比纯 C++ 慢**：一位成员报告称，他们用纯 C++ 和 CUDA 分别实现了第 3 章中的图像模糊示例，并注意到 CUDA 版本比纯 C++ 实现运行得慢。
   - *这个结果合理吗？* 这引发了对两种实现之间性能预期的深入探究。
- **性能对比讨论**：社区被促使讨论在某些场景下（特别是图像处理任务中）**CUDA** 实现比 **C++** 慢是否常见。
   - 这提出了关于可能影响执行时间的优化策略和硬件利用率的问题。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1343746255834054746)** (1 messages): 

> `MIOpen 编译问题，RX 9850M XT Wavefront Size，PyTorch 内存访问故障，MIOpen 崩溃的解决方法` 


- **MIOpen 默认使用错误的 wavefront size**：一位用户询问如何阻止 **MIOpen** 在 kernel 编译期间传递 `-mwavefrontsize=64`，因为他们的 **RX 9850M XT** 仅支持 `wavefrontsize=32`。
   - 他们提到在某些 PyTorch 配置中遇到了 **'Memory access fault - page not present or supervisor privilege'** 错误。
- **MIOpen 崩溃的解决方法**：为了缓解崩溃，用户设置了两个环境变量：`MIOPEN_DEBUG_CONV_GEMM=0` 和 `MIOPEN_DEBUG_CONV_DIRECT=0`。
   - 他们希望能有一个更持久的解决方案来处理 PyTorch 执行过程中遇到的问题。
- **ROCm GitHub 上的详细报告**：用户提供了一个 [GitHub issue 链接](https://github.com/ROCm/MIOpen/issues/3540)，详细说明了 MIOpen 的问题。
   - 这包括他们在 PyTorch 配置和性能问题上的持续困扰。


  

---

### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1343649843394121808)** (4 messages): 

> `Liger Kernel Issue #537, Native Sparse Attention Triton Repo, Efficient Triton Implementations` 


- **对 Liger Kernel Issue #537 的关注**：一名成员表示有兴趣为 [Issue #537](https://github.com/linkedin/Liger-Kernel/issues/537) 做出贡献，该议题讨论了支持来自 Upstage 的新 Solar 架构，该架构因其在单 GPU 设置上的性能而受到关注。
   - 该功能旨在增强功能性，对基于 GPU 的训练和推理具有重要意义。
- **NAS Triton 仓库介绍**：分享了 [XunhaoLai 的 Triton 仓库](https://github.com/XunhaoLai/native-sparse-attention-triton/tree/main/ops/triton)，该仓库实现了一种高效的稀疏注意力机制，如论文 [Native Sparse Attention](https://arxiv.org/abs/2502.11089) 中所述。
   - 该仓库支持利用 Triton 能力为 AI 模型提供先进的计算效率。
- **另一个高效的稀疏注意力实现**：提到了第二个仓库 [fla-org/native-sparse-attention](https://github.com/fla-org/native-sparse-attention)，重点介绍了专为硬件对齐和原生可训练稀疏注意力设计的高效 Triton 实现。
   - 这些实现旨在通过优化资源利用率来提高训练和推理的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/XunhaoLai/native-sparse-attention-triton/tree/main/ops/triton">native-sparse-attention-triton/ops/triton at main · XunhaoLai/native-sparse-attention-triton</a>：该仓库提供了论文 [Native Sparse Attention](https://arxiv.org/abs/2502.11089) 中稀疏注意力机制的高效 Triton 实现。 - XunhaoLai/native-sparse-attenti...</li><li><a href="https://github.com/linkedin/Liger-Kernel/issues/537">Support the new Solar architecture · Issue #537 · linkedin/Liger-Kernel</a>：🚀 功能、动机和推介。来自 Upstage 的这个模型对于适合在单 GPU 上进行训练和推理的模型来说非常强大！https://huggingface.co/upstage/solar-pro-preview-inst.....</li><li><a href="https://github.com/fla-org/native-sparse-attention">GitHub - fla-org/native-sparse-attention: 🐳 Efficient Triton implementations for &quot;Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention&quot;</a>：🐳 “Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention” 的高效 Triton 实现 - fla-org/native-sparse-attention
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1343638374170497128)** (5 messages): 

> `Branch Management, Auto Delete on Merge, Benchmark Task Idea, Python Dependency Issues` 


- **为了效率清理旧分支**：一名成员提到计划清理已经合并的旧分支，以保持仓库整洁。
   - 这反映了管理项目整洁度和简化协作的主动方法。
- **考虑合并后自动删除**：另一名成员建议为分支设置合并后自动删除功能，以增强仓库管理。
   - 这可能有助于防止废弃分支的堆积并简化项目导航。
- **创新的基准测试任务提案**：一名成员分享了一个有趣的基准测试任务想法，涉及评估 Claude 提取 VLLM 库以用于 Llama8b 推理的能力，分享于 [此处](https://x.com/naklecha/status/1894119405895704609)。
   - 这种方法可以作为验证代码基准测试和衡量 LLM 性能的绝佳方式。
- **依赖项包含问题**：一名成员指出，Python 库 `aiohttp` 和 `tenacity` 需要包含在 `requirements.txt` 中以避免导入错误。
   - 他们表示如果认为包含这些库是正确的，愿意创建一个 Pull Request，展现了贡献的积极性。



**提到的链接**：<a href="https://x.com/naklecha/status/1894119405895704609">来自 naklecha (@naklecha) 的推文</a>：我正在评估 Claude 将 VLLM 库提取为更简单的代码库的能力，该代码库仅包含 Llama8b 推理所需的部分。我认为衡量 LLM 的代码库提取能力...

  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1343645233648373882)** (62 条消息🔥🔥): 

> `Inline CUDA 提交问题、Autotuning 担忧、Leaderboard UI 更新、CUDA 与 Python 集成、基准测试的问题约束` 


- **Inline CUDA 提交导致超时**：一位用户在 T4 GPU 上使用 'submission_cuda_inline.py' 脚本进行 Leaderboard 测试时遇到了超时，引发了关于代码潜在问题的讨论。
   - 一名成员保证很快将提供提交 Inline CUDA 的说明以消除困惑。
- **关于 Autotuning 实践的辩论**：由于过长的 Autotuning 时间影响了基准测试的相关性，人们对此表示担忧，并建议由于高开销而限制其使用。
   - 一些参与者建议，如果用户同时也提交他们的缓存结果，允许一定程度的 Autotuning 可能会有好处。
- **Leaderboard 的新 UI 功能**：团队宣布了 Leaderboard 最近的 UI 更新，使其对参与者来说更清晰、更友好。
   - 讨论内容包括是否仅显示每个用户的最佳提交，以平衡唯一性与竞争性。
- **CUDA 与 Python 集成的挑战**：一位用户指出将 CUDA 代码编写为 Python 字符串存在困难，强调了提交中对更好多文件支持的需求。
   - 人们对将 CUDA 与 Python 集成的挑战表示担忧，需要在编写便捷性与提交完整性之间取得平衡。
- **Prefix Sum 问题中的数据大小约束**：关于 Prefix Sum 问题的数据大小约束出现了疑问，特别是它是否需要是某个 2 的幂的倍数。
   - 该用户被引导至一个确认约束条件的链接，强调了未来问题迭代中文档清晰度的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://accelerated-computing-class.github.io/fall24/">6.S894</a>：未找到描述</li><li><a href="https://gpu-mode.github.io/discord-cluster-manager/">GPU MODE Kernel Leaderboard | GPU MODE Kernel Leaderboard</a>：未找到描述</li><li><a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py">reference-kernels/problems/pmpp/vectoradd_py/solutions/correct/submission_cuda_inline.py at main · gpu-mode/reference-kernels</a>：Leaderboard 的参考 Kernel。通过在 GitHub 上创建账户为 gpu-mode/reference-kernels 的开发做出贡献。</li><li><a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/grayscale_py/reference.py#L31)">reference-kernels/problems/pmpp/grayscale_py/reference.py at main · gpu-mode/reference-kernels</a>：Leaderboard 的参考 Kernel。通过在 GitHub 上创建账户为 gpu-mode/reference-kernels 的开发做出贡献。</li><li><a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/prefixsum_py/submission.py">reference-kernels/problems/pmpp/prefixsum_py/submission.py at main · gpu-mode/reference-kernels</a>：Leaderboard 的参考 Kernel。通过在 GitHub 上创建账户为 gpu-mode/reference-kernels 的开发做出贡献。</li><li><a href="https://github.com/gpu-mode/reference-kernels/blob/main/problems/pmpp/prefixsum_py/task.yml#L51">reference-kernels/problems/pmpp/prefixsum_py/task.yml at main · gpu-mode/reference-kernels</a>：Leaderboard 的参考 Kernel。通过在 GitHub 上创建账户为 gpu-mode/reference-kernels 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1343640386014543925)** (62 messages🔥🔥): 

> `Leaderboard submissions, Test submissions, Benchmark submissions, API submissions, Modal runners performance` 


- **简化的 Leaderboard 提交**：多次向 **grayscale** Leaderboard 的提交在包括 **H100** 和 **A100** 在内的各种 GPU 上取得成功，涉及多个 ID，如 **278** 和 **370**。
   - 这表明 **Modal runners** 在不同的 Leaderboard 测试中表现出一致的性能。
- **Prefixsum 提交成功**：向 **prefixsum** Leaderboard 提交的多个 Benchmark 成功，ID 如 **351** 展示了在 **H100** 上的 GPU 效率。
   - 这些提交反映了与当前 Leaderboard 设置的稳健交互。
- **测试提交取得成功**：向 **matmul** 和 **grayscale** Leaderboard 提交的测试（如 ID **269** 和 **388**）演示了成功的执行。
   - 测试过程表明在 Leaderboard 配置下运行稳定。
- **Leaderboard 名称缺失问题**：出现了几次在提交脚本中 **Leaderboard 名称**缺失或不匹配的情况，导致了默认错误。
   - 这突显了提交标准需要更加明确，以避免此类不匹配。
- **API 提交即将推出**：有一项公告指出 **通过 API 提交** 的功能即将推出，引发了社区的好奇。
   - 鼓励成员们**保持关注**有关此功能的进一步更新。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1343649977062133802)** (1 messages): 

> `CUDA Inline, Compilation Caching, Benchmarking Efficiency` 


- **正确运行 CUDA Inline**：为了成功运行提交文件，请确保通过将 `load_inline` 调用放置在文件的顶层来编译 **CUDA** 代码。
   - 这可以确保编译被缓存，从而在随后的测试或 Benchmark 过程中实现更快的执行。
- **编译缓存的重要性**：缓存 **CUDA** 代码的编译对于减少测试期间的超时至关重要。
   - 使 `load_inline` 高效意味着它不应阻碍实际测试或 Benchmark 的性能。


  

---


### **GPU MODE ▷ #[ppc](https://discord.com/channels/1189498204333543425/1343373905783554048/1343674874769576008)** (6 messages): 

> `Bend programming language, AMD mi300A and Nvidia Grace Hopper, Programming Parallel Computers course` 


- **Bend 在并行编程中引发疑问**：一位用户链接了 [Bend GitHub repository](https://github.com/HigherOrderCO/bend) 并质疑其与频道关注点的相关性。
   - 其他人寻求关于其适用性的澄清，最终注意到它是一种**并行编程语言**，可能对 **AMD mi300A** 或 **Nvidia Grace Hopper** 芯片有用。
- **关于频道相关性的困惑**：一位成员表示困惑，澄清该频道致力于讨论与 Aalto 的 **Programming Parallel Computers** 课程公开版相关的内容。
   - 这导致了对 Bend 与频道预期焦点之间联系的怀疑。
- **关于 Bend 相关性的结论**：经过讨论，一位成员感谢其他人澄清了提及 Bend 可能不符合频道的宗旨。
   - 社区承认，明确主题相关性有助于保持讨论的集中。



**提到的链接**：<a href="https://github.com/HigherOrderCO/bend">GitHub - HigherOrderCO/Bend: A massively parallel, high-level programming language</a>：一种大规模并行的高级编程语言 - HigherOrderCO/Bend

  

---


### **GPU MODE ▷ #[feature-requests-and-bugs](https://discord.com/channels/1189498204333543425/1343759913431728179/1343761157449715805)** (2 messages): 

> `NCU Support, Leaderboard Updates, Standalone CUDA Examples, Generated PTX Retrieval Command` 


- **对 NCU 支持的期望**：有人提出了对 **NCU 支持** 的请求，强调了其对社区进行性能分析的重要性。
   - *未提供额外评论。*
- **Leaderboard 改进建议**：成员们讨论了在 Leaderboard 中**为每人保留最高分条目**的必要性，以增强竞争性。
   - *此功能有助于更好地认可个人贡献。*
- **对独立 CUDA 示例的渴望**：有人请求提供**独立 CUDA 文件示例**，以帮助用户进行开发工作。
   - *这些示例将作为实现 CUDA 解决方案的有用参考。*
- **请求获取生成 PTX 的命令**：一位成员建议添加一个类似于 [Godbolt](https://godbolt.org) 的命令来检索生成的 **PTX**。
   - *提议的命令是 `/leaderboard submit --ptx`，旨在简化 PTX 的访问。*


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1343629842717085706)** (127 条消息🔥🔥): 

> `Grok 3 vs O1-Pro, xAI GPU count, Synthetic Data Generation, AI Hardware Solutions, Large Concept Models` 


- **关于 Grok 3 与 O1-Pro 的辩论**：成员们讨论了是否为了 SuperGrok 而取消 O1-Pro 订阅，对于性能对比以及在编程任务中的适用性意见不一。
   - 一些用户发现 O1 在代码处理方面表现更好，而 Grok 详尽的回复可能需要调整 prompt。
- **xAI 的 GPU 扩张**：据报道，xAI 正在将其 GPU 集群扩展到 **200,000 个 GPU**，以增强其 AI 能力并与 OpenAI 等主要对手竞争。
   - 此次扩张标志着 Grok 正转向成为一个更通用的平台。
- **AI 中的合成数据 (Synthetic Data)**：关于合成数据生成的讨论强调了其重要性，例如 DeepSeek 使用概念而非 token 进行优化的案例。
   - 与会者指出，有效的合成数据生成流水线对于维持 AI 模型至关重要。
- **探索 AI 硬件解决方案**：用户探索了在低资源系统上托管 AI 模型的选项，建议使用低成本微型 PC 或 SBC 来运行 LLM。
   - 推荐使用 KCPP 和 RPC 功能，以帮助在设备间有效分配负载，从而获得更好的性能。
- **FB 的 Large Concept Models 演讲**：一位成员表示有兴趣分享 FB 关于 “Large Concept Models” 的论文，并讨论其对 AI 建模的影响。
   - 大家都认识到，理解主题内的概念对于有效的 AI 优化非常重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2306.11644">Textbooks Are All You Need</a>: 我们介绍了 phi-1，这是一个新的用于代码的大语言模型，其规模明显小于竞争模型：phi-1 是一个基于 Transformer 的模型，拥有 1.3B 参数，在 8 个 A100 上训练了 4 天，...</li><li><a href="https://arxiv.org/abs/2305.10601">Tree of Thoughts: Deliberate Problem Solving with Large Language Models</a>: 语言模型越来越多地被部署用于解决各种任务中的通用问题，但在推理过程中仍局限于 token 级别、从左到右的决策过程...</li><li><a href="https://arxiv.org/abs/2305.08291">Large Language Model Guided Tree-of-Thought</a>: 在本文中，我们介绍了 Tree-of-Thought (ToT) 框架，这是一种旨在提高自回归大语言模型 (LLM) 解决问题能力的新方法。ToT 技术...</li><li><a href="https://decrypt.co/307337/million-dollar-dolce-gabbana-digital-suit-fractionalized">Million-Dollar Dolce &amp; Gabbana Digital Suit Fractionalized on Ethereum L2 Base—Here’s Why - Decrypt</a>: Fermion Protocol 想要将奢侈品碎片化。它从杜嘉班纳 (Dolce &amp; Gabbana) 的百万美元数字西装开始。</li><li><a href="https://huggingface.co/QuantFactory/granite-3.1-3b-a800m-instruct-GGUF">QuantFactory/granite-3.1-3b-a800m-instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/i/grok/share/Ciu6W03tg8hxSaZTdEUcS5Vla">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://www.nextbigfuture.com/2024/04/snapshot-of-the-race-for-more-ai-compute.html),">Snapshot of the Race for More AI Compute | NextBigFuture.com</a>: 一家 AI 公司拥有的 Nvidia H100 和其他 Nvidia 芯片的数量代表了该公司的 AI 算力资源。</li><li><a href="https://huggingface.co/Joseph717171">Joseph717171 (Joseph)</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski">bartowski (Bartowski)</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1343668514682175590)** (4 messages): 

> `Native Sparse Attention, SigLIP 2 改进, 多语言检索增强` 


- **深入探讨 Native Sparse Attention**: 一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=ReA6pSS)，题为“Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention”，详细介绍了 Sparse Attention 机制中的重要概念。
   - 另一位成员对内容表示赞赏，称 *“这家伙的内容太棒了”*。
- **探索 SigLIP 2 更新**: 讨论集中在 [SigLIP 2](https://arxiv.org/abs/2502.14786) 上，它是性能最强的视觉编码器的改进版本，强调了其显著的进步。
   - 参与者指出，它带来了显著的改进，特别是在 **多语言检索 (multilingual retrieval)** 能力方面，正如一位成员的评论所证实的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.14786">SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features</a>: 我们推出了 SigLIP 2，这是一系列新的多语言视觉语言编码器，建立在原始 SigLIP 成功的基础之上。在第二次迭代中，我们扩展了原始的图像-文本训练目标...</li><li><a href="https://www.youtube.com/watch?v=ReA6pSSDzLk">Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention</a>: 论文: https://arxiv.org/abs/2502.11089 笔记: https://drive.google.com/open?id=1HLEM4m77-C8HqEBoKpk6mnpngIjUB5jR&amp;usp=drive_copy 00:00 Intro 01:30 Sparse attent...
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1343681854213193778)** (9 messages🔥): 

> `Claude 3.7 Sonnet, QwQ-Max 预览, Linear Attention 教程, DeepEP 通信库, Y Combinator 计算机视觉初创公司` 


- **Claude 3.7 Sonnet - 混合推理的奇迹**: Anthropic 发布了 **Claude 3.7 Sonnet**，这是他们迄今为止最智能的模型，可以产生近乎即时的响应或扩展思考（[详情点击此处](https://www.anthropic.com/news/claude-3-7-sonnet)）。它在 **编程 (coding)** 和 **前端 Web 开发** 方面表现出显著改进，同时还推出了用于 Agent 编程任务的 **Claude Code**。
   - 您可以在这个 [YouTube 视频](https://m.youtube.com/watch?v=t3nnDXa81Hs)中看到介绍，视频重点展示了该模型的能力。
- **QwQ-Max 抢先看**: **Qwen Chat** 宣布即将发布 **QwQ-Max**，这是一个基于 Qwen2.5-Max 的推理模型，目前已在预览模式下开放测试。此次更新还包括未来推出 **Android 和 iOS 应用**的计划，以及采用 **Apache 2.0 许可证**的开源变体。
   - 更多细节可以在他们的 [博客文章](https://qwenlm.github.io/blog/qwq-max-preview/)中找到，文中概述了其在数学、编程和 Agent 任务中的能力。
- **探索 Linear Attention 技术**: 最近的一个 **YouTube 教程** 邀请了 **Flash Linear Attention** 库的创作者 **Songlin Yang**，在[这段视频](https://www.youtube.com/watch?v=d0HJvGSWw8A)中讨论交互式学习。该环节深入探讨了涉及 Linear Attention 的模型训练和推理的高级技术。
   - Yang 的见解极具影响力，特别是考虑到 Flash Linear Attention 库 [GitHub 链接](https://github.com/fla-org/flash-linear-attention) 所引入的创新。
- **DeepEP - 推进 MoE 通信**: **DeepSeek AI** 推出的 **DeepEP** 是首个专为 MoE 模型训练和推理定制的开源 EP 通信库。它承诺提供 **高效的 all-to-all 通信**、优化的 Kernel，并支持其操作中的 **FP8 dispatch**。
   - 开发者可以在 GitHub 的这个[链接](https://github.com/deepseek-ai/DeepEP)查看其特性和实现。
- **Y Combinator 独特的初创公司提案**: **Y Combinator** 之前提出的一项关于 **计算机视觉汗水工厂 (sweatshop)** 的提案（现已删除）引起了褒贬不一的反应，用户以幽默的方式对其进行了评价。该概念虽然被认为是非正统的，但突显了 AI 在劳动密集型领域应用的有趣方面。
   - 讨论可以通过捕捉到发布背景的用户推文追溯到[这里](https://x.com/trbdrk/status/1894197711454285983?t=OSkkxG7yCRepQoj63zkHdQ&s=19)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1894130603513319842">来自 Qwen (@Alibaba_Qwen) 的推文</a>: &lt;think&gt;...&lt;/think&gt; QwQ-Max-Preview Qwen Chat: https://chat.qwen.ai/ 博客: https://qwenlm.github.io/blog/qwq-max-preview/ 🤔 今天我们在 Qwen Chat 中发布了由 o... 支持的 “Thinking (QwQ)”</li><li><a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>: 未找到描述</li><li><a href="https://x.com/deepseek_ai/status/1894211757604049133">来自 DeepSeek (@deepseek_ai) 的推文</a>: 🚀 #OpenSourceWeek 第 2 天：DeepEP。很高兴介绍 DeepEP —— 首个用于 MoE 模型训练和推理的开源 EP 通信库。✅ 高效且优化的 all-to-all 通信 ✅...</li><li><a href="https://github.com/anthropics/claude-code">GitHub - anthropics/claude-code: Claude Code 是一款 Agentic 编程工具，它运行在你的终端中，理解你的代码库，并通过执行常规任务、解释复杂代码和处理 git 工作流来帮助你更快地编码 —— 这一切都通过自然语言命令完成。</a>: Claude Code 是一款 Agentic 编程工具，它运行在你的终端中，理解你的代码库，并通过执行常规任务、解释复杂代码和处理 git 工作流来帮助你更快地编码...</li><li><a href="https://www.youtube.com/watch?v=d0HJvGSWw8A">Linear Attention 及其延伸（与 Songlin Yang 的互动教程）</a>: 影响力巨大的 Flash Linear Attention https://github.com/fla-org/flash-linear-attention 库的作者 Songlin Yang 加入了我，进行了一场互动...</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>: 今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。</li><li><a href="https://x.com/trbdrk/status/1894197711454285983?t=OSkkxG7yCRepQoj63zkHdQ&s=19">来自 ￦ (@trbdrk) 的推文</a>: @KennethCassel garry 删除了它，但是 😁</li><li><a href="https://m.youtube.com/watch?v=t3nnDXa81Hs">具有扩展思维能力的 Claude 3.7 Sonnet</a>: 介绍 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1343639000467902505)** (37 条消息🔥): 

> `大脑 vs. AI 并行性, 语言处理差异, AI 模型训练效率, Logit Bayesian Metacognition 数据集, 去中心化 AI 模型评估` 


- **大脑 vs. AI：并行性差异**：讨论了大脑的并行性与 AI 架构的不同，一位成员指出人类不像 RNN 那样并行处理长字符串。
   - 他们强调人类语言处理涉及**组块化 (chunking)**和**抽象化 (abstracting)**，而不是同时预测所有 token。
- **RNN 在扩展性上的局限**：参与者讨论了人类语言处理与当前 AI 模型之间的差异，强调了扩展经典 RNN 架构的困难。
   - 针对 RNN 中的**反向传播 (backpropagation) 挑战**以及针对特定目标量身定制替代架构的潜在需求提出了关键点。
- **Logit Bayesian Metacognition 数据集提案**：一位成员提出了一个新的数据集概念，其中包含用于解码 logit 概率的元数据以及单词，以辅助模型自我反思。
   - 该模型将有助于回答诸如“你为什么要推断出 X？”之类的问题，并结合 logit 概率进行更深入的分析。
- **防篡改的去中心化 AI 评估**：一位成员介绍了他们在创建一个防篡改的去中心化系统方面的研究，该系统用于评估 AI 模型，以对抗基准数据集上的过拟合。
   - 该倡议旨在提高整个行业 AI 模型评估的透明度和可靠性。
- **Proxy Structuring Engine 介绍**：另一位用户分享了他们的开源项目 Proxy Structuring Engine (PSE)，旨在解决部署期间 LLM 输出的非确定性问题。
   - 该引擎在确保结构化响应的同时，保持 AI 模型在聊天机器人和自动代码生成等各种应用中的**创造力**。



**提到的链接**: <a href="https://www.proxy.ing/pse">The Proxy Structuring Engine</a>: 推理时的高质量结构化输出

  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1343695467208245330)** (32 条消息🔥): 

> `MLA 架构, DeepSeek 模型, 用于推理的 Looped 模型, 傅里叶级数与平滑度, KV cache 优化` 


- **MLA 架构彻底改变了效率**：由 DeepSeek 展示的 **Multi-head Latent Attention (MLA)** 的引入，显著压缩了 **Key-Value (KV) cache**，与传统方法相比，推理速度提高了 5-10 倍。
   - 其有效性得到了多个实验室实验的支持，强调了 MLA 模型在大语言模型应用中的潜力。
- **DeepSeek 在模型训练方面的创新**：DeepSeek 在其 MLA 驱动的模型上投入了至少 **550 万**，展示了对其架构优于现有模型的信心。
   - 该公司的研究结果表明，未来的模型可能会进一步改进，暗示了业界对转向 MLA 技术的兴趣。
- **Looped 模型声称在推理方面取得突破**：最近的一篇论文指出，**looped transformer 模型**在推理任务上可以达到与更深的非循环架构相似的性能，同时减少了参数需求。
   - 这种方法允许模型高效地解决复杂任务，展示了在计算挑战中使用迭代方法的潜在优势。
- **傅里叶级数在建模中的挑战**：在关于傅里叶级数的讨论中，一名成员强调，有限项数的要求随函数的平滑度而变化，并引用了 *Gibbs phenomenon*（吉布斯现象）。
   - 讨论强调了在不同应用中选择合适项数的持续关注，这与数据建模领域相关。
- **优化注意力模型中的 KV cache**：成员们讨论了 **Native Sparse Attention** 和 MLA 可以同时改善长上下文模型的计算和缓存成本，使其更加高效。
   - 随着这些技术的开源，它们的实现可能会重塑 AI 领域前沿模型的未来。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2502.14837">Towards Economical Inference: Enabling DeepSeek&#39;s Multi-Head Latent Attention in Any Transformer-based LLMs</a>: Multi-head Latent Attention (MLA) 是由 DeepSeek 提出的一种创新架构，旨在通过将 Key-Value (KV) cache 显著压缩来确保高效且经济的推理...</li><li><a href="https://arxiv.org/abs/2502.17239">Baichuan-Audio: A Unified Framework for End-to-End Speech Interaction</a>: 我们介绍了 Baichuan-Audio，这是一个无缝集成音频理解和生成的端到端音频大语言模型。它具有文本引导的对齐语音生成机制...</li><li><a href="https://arxiv.org/abs/2502.17416">Reasoning with Latent Thoughts: On the Power of Looped Transformers</a>: 大语言模型展示了卓越的推理能力，缩放定律表明大参数量（尤其是深度方向）是主要驱动力。在这项工作中，我们提出了一个强有力的观点...</li><li><a href="https://arxiv.org/abs/2502.07864">TransMLA: Multi-Head Latent Attention Is All You Need</a>: 现代大语言模型 (LLM) 在当前硬件上经常遇到通信瓶颈，而非纯粹的计算限制。Multi-head Latent Attention (MLA) 解决了这一挑战...</li><li><a href="https://arxiv.org/abs/2406.19997">Wavelets Are All You Need for Autoregressive Image Generation</a>: 在本文中，我们采用了一种新的自回归图像生成方法，该方法基于两个主要成分。第一个是小波图像编码，它允许对图像的视觉细节进行 Token 化...</li><li><a href="https://arxiv.org/abs/2502.16111">PlanGEN: A Multi-Agent Framework for Generating Planning and Reasoning Trajectories for Complex Problem Solving</a>: 最近的 Agent 框架和推理时间算法在处理复杂规划问题时经常遇到困难，原因是验证生成的计划或推理的能力有限，且实例复杂度各异...</li><li><a href="https://kexue.fm/archives/10091">缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA - 科学空间|Scientific Spaces</a>: 无描述</li><li><a href="https://planetbanatt.net/articles/mla.html">On MLA</a>: 无描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1343644708009803907)** (9 messages🔥): 

> `Attention Maps, Neuron-based Methods, Intervention on Attention Maps, Emerging Syntax from Attention Maps` 


- **Attention Maps vs Neuron-based Methods**：成员们讨论了为什么 **Attention Maps** 相比 **Neuron-based Methods** 受欢迎程度有所下降，认为这可能是由于其观察性质（observational nature）导致的。
   - 另一位成员指出，尽管 **Attention Map** 具有观察性质，但人们仍然可以对其进行**干预（Intervention）**，这引发了进一步的探讨。
- **直接修改 Attention Maps**：一位成员提到，可以在前向传播（forward pass）期间**直接更改 Attention Maps**，而不是使用自定义掩码（masks）进行干预。
   - 这引发了关于这些修改可能如何影响模型行为的提问。
- **对 Attention Maps 生成树和图的兴趣**：一位成员表示，由于 **Attention Maps** 能够利用语言特征生成**树（Trees）和图（Graphs）**，他们对其有强烈的偏好。
   - 尽管对**机械机制细节（mechanistic details）**了解有限，他们仍考虑在未来的项目中使用这一特性。
- **自 BERT 以来 Attention Maps 中涌现的语法**：有人提到，自 **BERT** 推出以来，已有研究表明**语法（Syntax）会从 Attention Maps 中涌现**。
   - 这突显了在当今的研究背景下，Attention Maps 在理解语言模型方面仍具有持续的相关性和潜力。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1343638613451214979)** (10 messages🔥): 

> `Mixed Precision Training, Optimizer States in Mixed Precision, ZeRO Offload, BF16 Precision` 


- **混合精度训练：权重存储在哪里？**：在混合精度训练中，**Master FP32 weights** 通常存储在 **GPU VRAM** 中，除非激活了 **ZeRO Offload**。
   - 这一澄清解决了关于训练期间模型权重存储的疑问。
- **理解 BF16 与优化器的交互**：关于 **BF16 混合精度**对优化器的影响存在困惑，特别是它与模型精度相关还是独立的。
   - 专家建议，在 **ZeRO** 论文发表后，将高精度参数视为属于 **Optimizer States** 已成为普遍做法。
- **优化器状态的当前实践**：各实验室在处理优化器状态的方法上有所不同，有些选择将 **Adam moments 存储在 BF16** 中，而将 Master weights 存储在 FP32 中。
   - 一个常见的建议是执行原生混合精度（vanilla mixed precision），使用 **BF16 低精度权重**，同时配合 **FP32 optim+master-weights+grads**。



**提及的链接**：<a href="https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/optimizer/optimizer_config.py#L44">Megatron-LM/megatron/core/optimizer/optimizer_config.py at main · NVIDIA/Megatron-LM</a>：大规模训练 Transformer 模型的持续研究 - NVIDIA/Megatron-LM

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1343632494758461533)** (79 messages🔥🔥): 

> `Claude 3.7 Sonnet, Claude Code, Datacenter leasing concerns, Qwen enhancements, FlashMLA GitHub repository`

- **Claude 3.7 Sonnet 发布引发热议**：Anthropic 推出了 **Claude 3.7 Sonnet**，宣称提升了推理能力，并推出了一款名为 **Claude Code** 的新编程工具，该工具可作为基于终端的助手使用。
   - 参与者强调了其**更高的输出 Token (higher output tokens)** 限制和**增强的推理模式 (enhanced reasoning mode)**，与之前的版本相比，这使得交互更加连贯。
- **对数据中心供过于求的担忧**：有报告显示 Microsoft 正在取消数据中心租赁，这暗示市场可能存在**供应过剩**，与此前合同签署延迟的趋势一致。
   - 这引发了人们对 Microsoft 在 2024 年激进的预租策略及其对托管 (colocation) 市场整体影响的质疑。
- **围绕 Qwen 未来的讨论**：人们对即将发布的 **Qwen QwQ-Max** 充满期待，据称其在推理能力方面有实质性提升，并将在 Apache 2.0 协议下开源。
   - 社区讨论涉及了这些增强功能在智能模型领域中的重要性。
- **Claude Code 的性能体验**：早期用户报告称 **Claude Code** 提供了精简的编程辅助，尽管体验各异，一些人注意到其响应时间比其他模型慢。
   - 反馈包括对其潜在定价问题的见解，同时强调了通过**重度缓存 (heavy caching)** 功能来优化性能。
- **新发布与社区参与**：GitHub 项目 **FlashMLA** 被分享，虽然此前没有太多宣传，但引发了人们对其在 AI 开发中影响的兴趣。
   - 参与者表示渴望看到本周预期发布的 AI 工具的进一步进展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.npmjs.com/package/@anthropic-ai/claude-code?activeTab=code">@anthropic-ai/claude-code</a>: 直接在终端使用 Anthropic 的 AI 助手 Claude。Claude 可以理解你的代码库、编辑文件、运行终端命令，并为你处理整个工作流。最新版本：0.2.9...</li><li><a href="https://x.com/dyot_meet_mat/status/1894139577805267447?s=46">Mona (@dyot_meet_mat) 的推文</a>: 该死，Anthropic 真的把这个 system prompt 搞对了</li><li><a href="https://x.com/paulgauthier/status/1894167915869737058?s=46">Paul Gauthier (@paulgauthier) 的推文</a>: Sonnet 3.7 在 aider 排行榜上以 65% 的得分创下 SOTA，使用了 32k thinking tokens。在不使用 thinking 的情况下得分为 60%。65% Sonnet 3.7, 32k thinking；64% R1+Sonnet 3.5；62% o1 high；60% Sonnet 3.7, no th...</li><li><a href="https://x.com/ludwigabap/status/1894121367441695079?s=46">ludwig (@ludwigABAP) 的推文</a>: 今天有一千个 GitHub 仓库“阵亡”了</li><li><a href="https://x.com/cognition_labs/status/1894125030583537974?s=46">Cognition (@cognition_labs) 的推文</a>: 1/ Claude 3.7 Sonnet 已在 Devin 中上线！这款新模型是我们迄今为止在包括调试、代码库搜索和 agentic planning 在内的各种任务中见过的最强模型。</li><li><a href="https://x.com/paulgauthier/status/1894123992505880688?s=46">Paul Gauthier (@paulgauthier) 的推文</a>: Claude 3.7 Sonnet 在不使用 thinking 的情况下，在 aider 多语言基准测试中获得 60% 的得分。与 o3-mini-high 并列第三。Sonnet 3.7 拥有最高的非 thinking 得分（此前为 Sonnet 3.5）。Thinking 结果即将发布...</li><li><a href="https://x.com/devinai/status/1894112580894904632?s=46">Devin (@DevinAI) 的推文</a>: “在开发过程中，我们减少了对数学和计算机科学竞赛题目的优化，转而将重点转向能更好反映用户需求的现实世界任务。”...</li><li><a href="https://x.com/mikeyk/status/1894112962572358032?s=46">Mike Krieger (@mikeyk) 的推文</a>: Claude 3.7 Sonnet 命名过程的幕后花絮</li><li><a href="https://x.com/huybery/status/1894131290246631523?s=46">Binyuan Hui (@huybery) 的推文</a>: 🤔 深入思考 Qwen 的未来。作为即将发布的 QwQ-Max 的预告，此版本展示了其增强的功能，目前正在持续改进中，并将采用官方 Apache ...</li><li><a href="https://x.com/gregkamradt/status/1894179293292622312?s=46">Greg Kamradt (@GregKamradt) 的推文</a>: 终于等到了基准测试结果，Claude 3.7 Sonnet 现在是 SnakeBench 🐍 的第一名。击败了所有其他推理模型。它的思维过程比其他模型更加连贯。</li><li><a href="https://x.com/dylan522p/status/1894050388145508586?s=46">Dylan Patel (@dylan522p) 的推文</a>: 周五有一份报告称，微软正在取消数百兆瓦的数据中心租约——暗示存在严重的过度建设/供过于求风险。我们在 12 月的 D... 中讨论过这个问题。</li><li><a href="https://x.com/alexalbert__/status/1894147796166651931?s=46">Alex Albert (@alexalbert__) 的推文</a>: 噢对了，我差点忘了，GitHub 集成今天已向 claude.ai 的所有用户开放。</li><li><a href="https://x.com/jayelmnop/status/1894101064074375286?s=46">Jesse Mu (@jayelmnop) 的推文</a>: 在唯一重要的评估中达到 SOTA。引用 Saurav Kadavath (@sokadv)：Claude 3.7 会玩宝可梦！https://www.anthropic.com/research/visible-extended-thinking</li><li><a href="https://x.com/alexalbert__/status/1894095781088694497?s=46">Alex Albert (@alexalbert__) 的推文</a>: 我们正在开放一款正在构建的新型 agentic 编程工具的研究预览版：Claude Code。你将直接获得由 Claude 驱动的代码辅助、文件操作和任务执行功能...</li><li><a href="https://x.com/amandaaskell/status/1894113894794498394?s=46">Amanda Askell (@AmandaAskell) 的推文</a>: 今天发生了两件事：1. Claude 升级了。2. AGI 终于被定义为“任何能抓住超梦的模型”。引用 Anthropic (@AnthropicAI)：推出 Claude 3.7 Sonnet：我们的...</li><li><a href="https://github.com/deepseek-ai/FlashMLA">GitHub - deepseek-ai/FlashMLA</a>: 通过在 GitHub 上创建账号来为 deepseek-ai/FlashMLA 的开发做出贡献。</li><li><a href="https://x.com/kimmonismus/status/1894133480792924249?t=dutLjfxlZXPX0EHL7oJikQ&s=19">Chubby♨️ (@kimmonismus) 的推文</a>: Claude 3.7 Sonnet 的 system prompt：“助手是 Claude，由 Anthropic 创建。当前日期是 {{currentDateTime}}。Claude 乐于帮助人类，并将其角色视为一个聪明且善良的助手...”</li><li><a href="https://x.com/elder_plinius/status/1894167641725833395?s=46">Pliny the Liberator 🐉󠅫󠄼󠄿󠅆󠄵󠄐󠅀󠄼󠄹󠄾󠅉󠅭 (@elder_plinius) 的推文</a>: 笑死我了，不可能，刚在新的 Claude Sonnet 3.7 system prompt 里发现了一个彩蛋！！实际的 prompt 与他们在网站上发布的几乎完全相同，除了一个关键区别</li>

:&#34;Easte...</li><li><a href="https://x.com/i/spaces/1rmxPyWOlaXKN">来自 GitHub 的推文 - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter</li><li><a href="https://x.com/simonw/status/1894127950175637865?s=46">来自 Simon Willison (@simonw) 的推文</a>: 关于 Claude 3.7 Sonnet 的一些初步笔记，包括骑自行车的鹈鹕的 SVG（它做得非常好） https://simonwillison.net/2025/Feb/24/claude-37-sonnet-and-claude-code/</li><li><a href="https://x.com/AnthropicAI/status/1894092430560965029?t=FqN7Yp-8QBiifOmuqGm6VQ&s=33">来自 Anthropic (@AnthropicAI) 的推文</a>: 推出 Claude 3.7 Sonnet：我们迄今为止最智能的模型。它是一个混合推理模型，可以产生近乎即时的响应或扩展的、逐步的思考。一个模型，两种思考方式。W...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1343637598538829969)** (68 条消息🔥🔥): 

> `Grok3 工具调用, Claude 3.7 Sonnet 发布, QwQ-Max-Preview 公告, 结构化输出项目, AI 对齐讨论` 


- **Grok3 的工具调用机制**：成员们对 Grok3 未在系统提示词中列出工具调用的 Token 序列表示困惑，并注意到它依赖于模型的训练来推断语法。
   - 他们辩论了硬编码函数调用与上下文学习 (ICL) 的可靠性，质疑工具调用方法的透明度。
- **Claude 3.7 Sonnet 亮相**：[Claude 3.7 Sonnet](https://www.anthropic.com/news/claude-3-7-sonnet) 正式发布，展示了其混合推理能力和用于增强编程任务的新命令行工具。
   - 讨论强调了 Sonnet 在软件工程基准测试中的优越性，以及它在取代 Claude 3.5 方面的潜在作用。
- **QwQ-Max-Preview 预览**：[QwQ-Max-Preview](https://qwenlm.github.io/blog/qwq-max-preview/) 作为 Qwen 系列的一部分推出，宣传了深度推理和 Apache 2.0 协议下的开源可访问性等特性。
   - 成员们推测了模型大小，猜测可能大于 32B，同时希望有更小的版本供本地使用。
- **结构化输出开源项目**：一位用户宣布启动一个旨在解决结构化输出挑战的开源项目，邀请反馈与合作。
   - 另一位成员建议重新发布公告，以提高社区内的可见度和参与度。
- **AI 对齐与社区论述**：一位成员批评了社交媒体上的 AI 对齐论述，对某些观点如何限制 AI 开发社区的理解表示沮丧。
   - 他们强调了公开对话和深入理解的重要性，而不是将狭隘的观点强加于 AI 功能之上。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://qwenlm.github.io/blog/qwq-max-preview/">&lt;think>...&lt;/think> QwQ-Max-Preview | Qwen</a>: 无描述</li><li><a href="https://www.anthropic.com/news/claude-3-7-sonnet">Claude 3.7 Sonnet 和 Claude Code</a>: 今天，我们宣布推出 Claude 3.7 Sonnet，这是我们迄今为止最智能的模型，也是市场上首个普遍可用的混合推理模型。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1343718334998515765)** (6 条消息): 

> `Sonnet-3.7 Benchmarking, Misguided Attention Eval, Reasoning Mode Activation` 


- **Sonnet-3.7 在基准测试中表现出色**：一名成员使用 [Misguided Attention Eval](https://github.com/cpldcpu/MisguidedAttention) 对 **Sonnet-3.7** 进行了基准测试，声称它是最强的非推理模型，表现几乎超越了 **o3-mini**。
   - 他们还分享了一张 [基准测试结果的图片](https://cdn.discordapp.com/attachments/1154120232051408927/1343718335266947152/image.png?ex=67be4ab8&is=67bcf938&hm=cd832c42f997fdab44d2891e45b944fec9e3d5892c0def9dc463f4c8107d7954&)。
- **了解 Misguided Attention Eval**：另一位成员询问 **Misguided Attention** 测试评估的是什么，得到的回复指出它考察的是 **overfitting**。
   - 该评估重点展示了大语言模型在存在误导信息的情况下如何处理推理。
- **探索推理模式激活**：一位成员正在研究如何通过 **OR API** 激活 **thinking mode**，并对其可行性提出了疑问。
   - 这反映了在增强模型交互功能方面的持续努力。



**提到的链接**：<a href="https://github.com/cpldcpu/MisguidedAttention">GitHub - cpldcpu/MisguidedAttention: A collection of prompts to challenge the reasoning abilities of large language models in presence of misguiding information</a>：一组旨在挑战大语言模型在存在误导信息时的推理能力的提示词集合 - cpldcpu/MisguidedAttention

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1343636953903534131)** (4 条消息): 

> `Azure chat interface update, Video generation integration, Artifacts usability concerns` 


- **Azure 聊天界面焕新**：**Azure chat interface** 已更新，预计今天将发布一些新内容。
   - 成员们对这些变化及其对用户体验的影响感到好奇。
- **集成视频生成引入新可能性**：更新后的 Azure 聊天界面现在包含 **integrated video generation**，增强了交互能力。
   - 这一功能可能会重新定义用户与平台的互动方式。
- **Artifacts 仍然感觉笨重**：尽管进行了更新，用户仍然觉得 **artifacts** 有点笨重，将其描述为“半成品的仿制品”。
   - 普遍情绪认为需要改进以实现更流畅的易用性。



**提到的链接**：<a href="http://qwen.ai">Qwen Chat</a>：未找到描述

  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1343642717464301620)** (62 条消息🔥🔥): 

> `Anthropic MCP Registry API, LLM 版本更新, Haiku 3.5 工具支持, Claude Code 与 Aider 性能对比, MCP Server 推荐` 


- **Anthropic 发布 MCP Registry API**：昨天，@AnthropicAI 宣布了官方的 [MCP registry API](https://x.com/opentools_/status/1893696402477453819)，这在寻求可靠事实来源（source of truth）的社区中引起了轰动。
   - 一位社区成员表示：*这对我们来说是个极好的消息，因为我们一直想要一个权威的事实来源。*
- **对 LLM 版本命名的困惑**：成员们对 **LLM 版本命名** 惯例表达了沮丧和调侃，特别是针对 **Claude 3.7** 及其功能。
   - 有人指出，**3.7** 似乎集成了之前版本的功能，同时通过自适应思考模式（adaptive thinking modes）提升了性能。
- **Haiku 3.5 与工具支持讨论**：在讨论 **Haiku 3.5** 时，用户注意到它支持工具（tools），但存在局限性，并表示在连接较少工具时表现更好。
   - 一位成员抱怨了 Server 和工具管理的压力，并透露计划开发一个集成工具集的聊天应用以简化使用。
- **Claude Code 性能获得高度评价**：用户测试了 **Claude Code**，并强调了其优于 **Aider** 的性能，特别是在高效处理代码错误方面。
   - 成员们还表达了利用 **Claude Code** 处理更复杂任务的兴趣，表明它可能会利用深度思考模式来获得最佳结果。
- **MCP Server 与 Windows 兼容性**：关于 **MCP server 在 Windows 上的兼容性** 引起了讨论，一些用户由于某些工具缺乏支持而遇到了安装障碍。
   - 对于在 Windows 系统上难以安装 Server 的用户，社区推荐了 [mcp.run](https://www.mcp.run/) 等替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/opentools_/status/1893696402477453819">来自 OpenTools (@opentools_) 的推文</a>: 昨天 @AnthropicAI 在 @aiDotEngineer 宣布了官方的 MCP registry API 🎉 这对我们来说是极好的消息，因为我们一直想要一个*权威的*事实来源。我们在 http://opentools.com/registry 制作了...</li><li><a href="https://www.mcp.run/)">mcp.run - MCP Servlets 的应用商店：适用于 AI 应用和 Agent 的便携且安全的代码。</a>: 无描述</li><li><a href="https://www.cyberchitta.cc/articles/lc-alternatives.html">36 个 LLM Context 的替代方案</a>: 一份全面的开源工具列表，帮助开发者将代码打包进 LLM 聊天中。从简单的文件合并器到复杂的上下文管理器，这些 CLI 工具简化了你分享代码的方式...</li><li><a href="https://github.com/cyberchitta/llm-context.py">GitHub - cyberchitta/llm-context.py: 通过 Model Context Protocol 或剪贴板与 LLM 共享代码。基于 Profile 的自定义功能可以轻松切换不同任务（如代码审查和文档编写）。代码大纲支持作为实验性功能提供。</a>: 通过 Model Context Protocol 或剪贴板与 LLM 共享代码。基于 Profile 的自定义功能可以轻松切换不同任务（如代码审查和文档编写）。代码大纲支持...</li><li><a href="https://github.com/nahmanmate/code-research-mcp-server">GitHub - nahmanmate/code-research-mcp-server</a>: 通过在 GitHub 上创建账号来为 nahmanmate/code-research-mcp-server 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1343647257085481110)** (11 条消息🔥): 

> `MetaMCP 许可变更，Enact Protocol MCP Server，Claude 3.7 Sonnet 推理功能` 


- **MetaMCP 切换至 AGPL**：成员们讨论了 MetaMCP 许可带来的挑战，特别是限制贡献的 ELv2 条款，并建议切换到 **AGPL** 以获得更好的开放性。
   - 在考虑了不同的许可策略后，他们得出结论：AGPL 要求托管的更改必须开源，从而允许社区贡献。
- **Enact Protocol 旨在开发 MCP server**：一位成员有兴趣为 **Enact Protocol** 创建一个 MCP server，并重点介绍了其用于定义自动化任务的 GitHub 仓库。
   - 他们分享了 [Enact Protocol GitHub 链接](https://github.com/EnactProtocol/enact-python)，该仓库展示了一个用于任务执行的标准框架。
- **Claude 3.7 Sonnet 推理功能发布**：关于在 Sage 上推出 **Claude 3.7 Sonnet 推理功能**的讨论非常热烈，该模型现在包含增强的思考能力。
   - 新功能包括思考模式开关、改进的滚动条以及可展开的思考块（用于可视化 Claude 的推理过程），并附带一段 [演示视频](https://cdn.discordapp.com/attachments/1315696461316358175/1343767877282693161/SageReasoning.mp4?ex=67be78db&is=67bd275b&hm=6be8d715d1920b4314cf572087735cb8750f87fa5cd3cdcd6855db01aa260388&)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/EnactProtocol/enact-python">GitHub - EnactProtocol/enact-python: Python implementation of the Enact Protocol, a standardized framework for defining and executing automated tasks and workflows.</a>：Enact Protocol 的 Python 实现，一个用于定义和执行自动化任务及工作流的标准框架。 - EnactProtocol/enact-python</li><li><a href="https://github.com/metatool-ai/mcp-server-metamcp">GitHub - metatool-ai/mcp-server-metamcp: MCP Server MetaMCP manages all your other MCPs in one MCP.</a>：MCP Server MetaMCP 在一个 MCP 中管理你所有的其他 MCP。 - metatool-ai/mcp-server-metamcp
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1343640149442957363)** (20 条消息🔥): 

> `带有自定义 LoRA 的 Gemma-9B，Qwen Max 开放权重发布，Claude-3.7 模型讨论，课程所需的 Python 技能，开启 AI 职业生涯` 


- **Gemma-9B 模型加载问题**：一位用户在加载带有自定义 **LoRA** 的 **Gemma-9B** 模型时遇到错误，具体与 'LoraConfig.__init__()' 接收到意外的关键字参数有关。
   - 另一位用户分享了类似的经历并寻求相关建议。
- **期待 Qwen Max 发布**：**Qwen Max** 推理模型被认为是自 **DeepSeek-R1** 以来最强大的开源模型，权重可能在本周发布。
   - 该模型展示了在数学理解和创造力方面的进步，官方版本即将推出，并计划发布移动端 App。
- **Claude-3.7 成为强力竞争者**：围绕 **Claude-3.7** 的讨论兴起，它被称为潜在的 **SWE-bench 杀手**，预测其能以三分之一的尺寸实现 **DeepSeek-R1** 的性能。
   - 也有人对其与其他模型相比的性价比表示担忧，指出其每百万 token 输出的价格较高。
- **咨询课程所需的 Python 技能**：一位成员询问完成课程所需的 **Python** 熟练程度，并寻求推荐的提升课程。
   - 其他人参与了讨论，暗示课程中可能链接了一个基础的 **Python 入门教程**。
- **关于开启 AI 职业生涯的建议**：一位用户请求指导如何选择 AI 专业领域以及如何在这一领域寻求工作机会。
   - 讨论围绕着对具体执行步骤的需求以及如何将技能变现的信息展开。



**提及的链接**：<a href="https://x.com/Alibaba_Qwen/status/1894130603513319842">来自 Qwen (@Alibaba_Qwen) 的推文</a>：&lt;think&gt;...&lt;/think&gt; QwQ-Max-PreviewQwen Chat: https://chat.qwen.ai/Blog: https://qwenlm.github.io/blog/qwq-max-preview/🤔 今天我们在 Qwen Chat 中发布了“Thinking (QwQ)”，由 o... 支持。

  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1343718083746992149)** (2 messages): 

> `Agent 课程中的 Fine-tuning，Apple Silicon 上的 Pytorch` 


- **掌握 Agent 课程中的 Fine-Tuning**：今天在 Agent 课程中获得了关于 **fine-tuning** 技术的见解。
   - 此次学习强调了微调模型以获得更好性能的重要性。
- **Apple Silicon 缺失 Pytorch 函数**：一位成员发现某些 **Pytorch 函数** 尚未在 **Apple Silicon** 上实现，导致不得不转向 Colab。
   - *“我想还是回 Colab 吧”* 凸显了 Apple 用户在 ML 领域面临的挑战。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1343754527878418472)** (1 messages): 

> `乌克兰语 TTS 数据集，speech-uk/opentts-mykyta` 


- **发布乌克兰语 TTS 数据集**：一个包含高质量乌克兰语 **TTS 数据集** 的合集已在 Hugging Face 上发布，提升了 TTS 项目的可访问性。
   - 该数据集可供查看，并已获得 **6.44k** 次更新，链接见[此处](https://huggingface.co/collections/speech-uk/ukrainian-text-to-speech-67bd059d61b2598f3a2a7969)。
- **Speech-uk/opentts-mykyta 更新**：名为 **speech-uk/opentts-mykyta** 的数据集在约 5 小时前进行了更新，展示了持续的支持与开发。
   - 这一举措反映了社区致力于为乌克兰语文本转语音应用提供优质资源的承诺。



**提到的链接**：<a href="https://huggingface.co/collections/speech-uk/ukrainian-text-to-speech-67bd059d61b2598f3a2a7969">Ukrainian Text-to-Speech - a speech-uk Collection</a>：未找到描述

  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1343668595732779109)** (5 messages): 

> `Computer Vision 聚会，CV 课程体验，海洋生态系统项目，篮球队目标检测` 


- **Computer Vision 聚会启动！**：今年首场 **Computer Vision Hangout** 活动已创建，重点关注 Hugging Face Computer Vision 生态系统的更新和社区演示。
   - 参与者可以期待 **趣味问答** 和闲聊，这是一个极佳的人脉拓展机会。
- **对课程互动的热情**：一位成员表达了对 **computer vision 课程** 的热忱，渴望与其他初学者和爱好者建立联系。
   - 其他人也纷纷加入，强调了对 computer vision 话题进行互动和讨论的愿望。
- **课程完成与证书**：一位成员最近完成了 CV 课程，并称赞了课程提供的 **优质信息**，同时提到在提交 Google 表单后接收证书有所延迟。
   - 他们表示随时欢迎讨论 computer vision 话题，营造了协作学习的氛围。
- **海洋生态系统与篮球项目见解**：讨论中提到一位成员正专注于 **海洋生态系统** 相关项目，而其朋友即将开展一个训练 **篮球队目标检测器 (object detector)** 和追踪器的项目。
   - 这展示了 computer vision 的实际应用，反映了社区对多元领域的兴趣。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1343628610988146821)** (38 条消息🔥): 

> `AI for Business 系列, NBA Parlay Picks, 课程推荐, NLP 课程报名, Unit 1 SFT 训练` 


- **AI for Business 系列发布**：一名成员分享了他们的新系列 [AI for Business – From Hype to Impact](https://www.linkedin.com/posts/brunokilian_aiforbusiness-artificialintelligence-digitaltransformation-activity-7299815566075158528--xcc?utm_source=share&utm_medium=member_ios&rcm=ACoAAACU3YIBsDJ8y62xuN4LdHlvFqYKsQDJ5eE)，旨在帮助企业利用 AI 作为竞争优势。
   - 该系列将涵盖从 AI 作为业务赋能工具到在不干扰运营的情况下扩展 AI 等主题。
- **潜在 1.5 万美元的 NBA 投注选择**：一位成员宣布拥有一个 6 场和 4 场的连串过关投注 (Parlay)，潜在奖金达 **1.5 万美元**，并表示可根据要求分享选择。
   - 他们提供了一张图片作为参考，但未在文本中透露具体细节。
- **探索额外的免费课程**：有人询问是否有可以补充当前课程的免费课程，随后有人建议学习额外的 NLP 课程。
   - 随着其他用户介绍自己，成员的参与度有所提高，增强了社区感。
- **Unit 1 SFT 训练见解**：一位成员报告称，在购买 Colab 计算单元后，在 A100 GPU 上仅用 **20 分钟** 就完成了 Unit 1 的 SFT 奖励训练。
   - 他们提出了关于 `convert_currency` 函数、工具可用性以及访问 TensorBoard 输出的问题。
- **社区介绍与热情**：来自不同国家的新成员不断介绍自己，表达了对参加 AI Agents 课程的兴奋之情。
   - 随着成员们寻找学习伙伴 (Accountability partners)，在课程中进行协作和分享更新的热情也随之高涨。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1343638432064471173)** (41 条消息🔥): 

> `LMS Server 与 WordPress 集成、Qwen 2.5 VL 模型性能、Speculative Decoding、Deepseek R1 671B 硬件需求、本地与托管模型使用对比` 


- **LMS Server 与 WordPress 集成的需求**：一位成员询问是否有人成功将 **LMS Server** 与当前的 **WordPress** 插件集成，并运行在本地的 Windows 和 Ubuntu/nginx 环境中。
   - 这一咨询凸显了在 AI 开发中，高效的 Demo 创建工具的必要性。
- **Qwen 2.5 VL 模型受到关注**：据多位用户确认，**Qwen 2.5 VL 7B** 模型运行效果良好，在质量上超越了之前的 **Llama 3.2** 等模型。
   - 用户指出该模型在生成图像描述方面具有潜力，展示了其在现实场景中的应用价值。
- **Speculative Decoding 面临的挑战**：一位用户提出 **speculative decoding** 是否仅限于特定模型的问题，并指出 **Llama 3.1** 与 **Llama 3.2** 之间的兼容性问题。
   - 有建议认为模型架构（architecture）可能会影响兼容性，并鼓励用户测试不同的配置以深入了解。
- **在内存受限的情况下本地运行 Deepseek R1**：一位用户询问在仅有 **192GB** RAM 的情况下运行 **Deepseek R1 671B** 的可行性，根据文档，这是一个临界阈值。
   - 一位支持者分享了他们使用特定 **quantization**（量化）来优化模型性能的经验，并强调了 **GPU offloading** 的优势。
- **使用 LM Studio 的混合开发工作流**：一位成员分享了他们的工作流：先使用 **LM Studio** 进行本地迭代开发，然后切换到托管服务商（hosted providers）使用更大的模型进行最终运行。
   - 这种方法展示了在 AI 生态系统中使用兼容 **API** 的优势，可以实现本地与云端资源的无缝切换。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://model.lmstudio.ai/download/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">在 LM Studio 中下载并运行 IAILabs/Qwen2.5-VL-7b-Instruct-GGUF</a>：在你的 LM Studio 本地使用 IAILabs/Qwen2.5-VL-7b-Instruct-GGUF</li><li><a href="https://lmstudio.ai/docs/lms/log-stream">lms log stream | LM Studio 文档</a>：从 LM Studio 串流日志。对于调试发送到模型的 prompt 非常有用。</li><li><a href="https://huggingface.co/IAILabs/Qwen2.5-VL-7b-Instruct-GGUF">IAILabs/Qwen2.5-VL-7b-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF/tree/main/DeepSeek-R1-UD-IQ1_S">unsloth/DeepSeek-R1-GGUF at main</a>：未找到描述</li><li><a href="https://lmstudio.ai/docs/advanced/speculative-decoding">Speculative Decoding | LM Studio 文档</a>：使用草稿模型（draft model）加速生成</li><li><a href="https://huggingface.co/lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF">lmstudio-community/Meta-Llama-3.1-8B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mlx-community/Llama-3.2-1B-Instruct-4bit">mlx-community/Llama-3.2-1B-Instruct-4bit · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1343664555737481330)** (20 条消息🔥): 

> `A770 性能、PC 组装困扰、M2 Max 规格、Apple 芯片对比` 


- **A770 表现出不错的性能**：用户提到他们的 **A770** GPU 在任务中表现尚可，但未提供具体指标。
   - 这为用户间关于 GPU 能力的持续讨论增添了参考。
- **对 PC 组装问题的困扰**：用户对硬件兼容性问题表示沮丧，提到 **aio pump** 需要一个 **USB 2.0 header**，并且会干扰最后一个 **PCIE slot**。
   - *“我有第二个系统可以把它们都装进去”* 暗示他们可能会寻找替代方案。
- **M2 Max 是日常使用的稳妥选择**：考虑到对 **M4 Max** 投资成本的担忧，用户决定选择翻新的 **M2 Max 96GB** 用于爱好和生产力工作。
   - *“我没有勇气去买 M4 Max”* 反映了许多人在购买高端科技产品时的犹豫。
- **注意到功耗差异**：讨论中提到了各种 Apple 芯片之间的 **clock and throttle behavior**（频率与降频行为），观察到 **M2 Max** 的功耗为 **60w**，而 **M4 Max** 可以轻松达到 **140w** 的峰值。
   - 这反映了用户在不同代际芯片切换时的体验及其对功耗的影响。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1343732966811107469)** (1 messages): 

> `功能需求看板，用户反馈机制` 


- **新功能需求看板上线**：Stability AI 推出了一个新的[功能需求看板](https://stabilityai.featurebase.app/)，允许用户通过在 Discord 中输入 /feedback 轻松提交他们的想法。
   - 该举措旨在增强用户参与度，通过社区对需求进行投票，以便更好地确定开发优先级。
- **赋能用户塑造未来功能**：功能需求看板的加入简化了用户表达希望在 Stability AI 工具中集成哪些功能的过程。
   - 正如文中所述，*您的反馈帮助我们确定下一步工作的优先级*，强调了社区在塑造平台方面的作用。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1343648573689958400)** (52 messages🔥): 

> `图像生成速度，SD3 Ultra 功能，狗品种图像数据集，模型性能对比，分辨率设置` 


- **对 SD3 Ultra 的好奇**：成员们表达了对 **SD3 Ultra** 的好奇，特别是其工作流以及与 **SD3L 8B** 相比更高的高频细节，正如一位贡献者所指出的。
   - *它仍然存在——我还在用它*，一位用户评论道，增加了对其能力的兴趣。
- **图像生成速度差异**：一位用户分享了图像生成耗时 **20 分钟**的经历，引发了关于不同硬件配置性能的讨论，其中一位成员报告在 **3060 TI** 上生成 **1920x1080** 分辨率的图像仅需 **31 秒**。
   - 另一位用户指出，生成时间很大程度上取决于模型；例如，**SD1.5** 在 **3070 TI** 上平均耗时 **4-5 秒**。
- **寻找狗图像数据集**：一位成员请求帮助寻找除 **Stanford Dogs dataset** 之外的高质量狗品种图像数据集，强调他们需要的不仅仅是 **20k 张图像**。
   - 他们明确要求数据集应包含带有狗品种标签的图像。
- **分辨率设置影响性能**：参与者讨论了图像分辨率的最佳设置，建议使用 **576x576** 等较小尺寸以提高生成速度。
   - 一位用户通过这些调整取得了成功，但仍经历了约 **8 分钟**的延迟。
- **图像生成的一般性讨论**：用户分享了图像生成的个人经验和配置，包括针对 **16:9** 宽高比的 **1366x768** 等设置。
   - 讨论强调了基于 GPU 能力和所选分辨率在速度和效率方面的差异。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1343660315799326720)** (11 messages🔥): 

> `Mojo FFI with GLFW/GLEW, Graphics Programming in Mojo, Dynamic linking and library loading, Error with lightbug_http dependency, Issue with small_time dependency version` 


- **Mojo FFI 支持图形编程**：一位成员展示了如何使用 Mojo FFI 将静态库链接到 **GLFW**/**GLEW**，并分享了一个涉及数独的示例。
   - 使用 **Mojo** 进行*图形编程是完全可能的*，这表明所需的调用只需通过自定义的 C/C++ 库暴露即可。
- **使用别名进行外部调用**：通过为外部调用设置别名，一位成员展示了在 Mojo 中绘制数独网格的封装函数示例。
   - 使用的语法为：`alias draw_sudoku_grid = external_call[...]`，以实现精简的函数访问。
- **Mojo 中的动态链接方法**：分享了一个代码仓库，说明了如何通过 Python 脚本劫持加载器，从而在 Mojo 中进行库链接。
   - 你可以在 GitHub [此处](https://github.com/ihnorton/mojo-ffi)查看演示。
- **遇到 lightbug_http 依赖错误**：一位用户报告称，在全新的 Mojo 项目中添加并尝试使用 **lightbug_http** 依赖时出现错误，并引用了一个 Stack Overflow 问题。
   - 他们遇到了尾随字节（trailing bytes）和模块解析问题，这表明依赖安装或配置可能存在问题。
- **关于 small_time 依赖版本锁定的疑问**：一位用户推测，将 **small_time** 依赖锁定为 25.1.0 而不是使用 ==25.1.0 是否会导致他们的错误。
   - 这表明依赖管理中的版本规范对于避免类似问题可能至关重要。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://stackoverflow.com/questions/79319716/ma">Magic fails to install Mojo dependencies</a>：我无法在全新的 Mojo 项目中使用名为 lightbug_http 的 Mojo 依赖。&#xA;magic init hello_web --format mojoproject&#xA;cd hello_web&#xA;magic shell&#xA;&#xA;打印 «Hello world» ...</li><li><a href="https://github.com/ihnorton/mojo-ffi">GitHub - ihnorton/mojo-ffi: Mojo FFI demos: dynamic linking methods, and a static linking proof-of-concept</a>：Mojo FFI 演示：动态链接方法和静态链接概念验证 - ihnorton/mojo-ffi
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1343753014573273229)** (20 messages🔥): 

> `Hardware Accelerated Conway's Game of Life, GPU Utilization in MAX, SIMD Implementation by Daniel Lemire, Game of Life Computer Concepts, Space Patterns in Conway's Game` 


- **Eggsquad 目前最“不务正业”的 MAX 用法**：一位成员展示了他们使用 MAX 和 pygame 实现的硬件加速版**康威生命游戏 (Conway's Game of Life)**，并称这是一个相当“愚蠢”的应用。
   - 他们分享说，该程序在处理复杂的起点时表现出色，甚至增加了飞船模式（spaceship patterns）的功能。
- **好奇 GPU 功能**：讨论围绕 **MAX** 与 **2080 Super** GPU 的兼容性展开，另一位成员建议可以尝试 CPU 暴力破解。
   - 共识倾向于从 Python 运行脚本，这应该有助于 GPU 的集成。
- **探索 SIMD 化技术**：Eggsquad 提到发现了 Daniel Lemire 的 **SIMD 化实现**，但表示目前还不打算进一步深入研究。
   - Darkmatter 指出，利用位打包（bit packing）可以在他们的实现中支持更大的图形。
- **错误修复与进度更新**：在解决了一个 bug 后，Eggsquad 确认他们的生命游戏实现运行良好，展示了持续的进展。
   - 他们通过动画展示了结果，其中包括一个描绘游戏中“枪（guns）”的动画。
- **集成游戏模式参数**：Eggsquad 添加了一个环绕游戏空间的选项，从而能够利用游戏机制创建**飞船模式**。
   - 他们指出，通过 Graph API 向模型添加参数以进行进一步实验是一项令人兴奋的功能。



**提及的链接**：<a href="https://www.nicolasloizeau.com/gol-computer">Nicolas Loizeau - GOL computer</a>：此处提供了一个全新（且更好）版本的 GOL 计算机：https://github.com/nicolasloizeau/scalable-gol-computer

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1343641731538616350)** (1 条消息): 

> `Hanna Hajishirzi 主讲第 4 讲，Tulu 3 的进展，Open training recipes，带有可验证奖励的强化学习，语言模型在科学领域的应用` 


- **今日第 4 讲关于 Open Training Recipes**：欢迎参加今天 **PST 时间下午 4:00** 由 **Hanna Hajishirzi** 主讲的第 4 讲直播，观看地址在 [这里](https://www.youtube.com/live/cMiu3A7YBks)。她将分享主题为“语言模型中推理和 Agent 的 Open Training Recipes”。
   - 本次课程将涵盖训练语言模型的全面努力，提升其从 **pre-training** 到 **post-training** 的推理能力。
- **Tulu 3 表现优于竞争对手**：**Hanna** 将讨论 **Tulu 3**，这是一个最先进的 post-trained 语言模型，通过创新的训练方法超越了 **DeepSeek V3** 和 **GPT-4o**。这还包括她团队的 open training recipe，涉及 data curation 和 fine-tuning。
   - 她指出：*“我们的目标是在 pre-training 阶段增强推理能力，”* 强调了提升模型有效性的重要性。
- **引入创新的强化学习方法**：本次演讲将展示一种独特的 **带有可验证奖励的强化学习方法 (RLVR)**，用于有效地训练语言模型。该方法旨在显著影响这些模型在两个训练阶段的推理能力。
   - Hanna 将分享关于结合这些先进强化学习技术的测试策略见解。
- **语言模型的现实世界应用**：讲座的另一个重点是构建能够使用训练好的模型 **合成科学内容** 的 Agent。这一应用对于弥合理论研究与实际用例之间的差距至关重要。
   - Hanna 在反思现实应用的重要性时表示：*“我们的努力旨在增强对人类生活的适用性和实用性。”*
- **即将发布的 MOOC 课程大纲详情**：关于 MOOC 课程大纲的详细信息将很快发布，让参与者了解未来的课程安排。团队感谢大家在他们敲定这些细节时的 **耐心** 等待。
   - 请继续关注更新，这将提升您的学习体验！



**提到的链接**：<a href="https://www.youtube.com/live/cMiu3A7YBks.">CS 194/294-280 (Advanced LLM Agents) - Lecture 4, Hanna Hajishirzi</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1343697357006438412)** (9 条消息🔥): 

> `迟到者的 Quiz 提交，Research Track 申请状态，MOOC 学生的 Application Track` 


- **MOOC 学生的 Quiz 提交不会受到处罚**：一位成员澄清说，Quiz 的截止日期仅适用于 **Berkeley 学生**，这意味着 MOOC 学生的所有 Quiz 都可以在学期结束前提交。
   - 他们强调，关于 MOOC 课程大纲的更多细节将很快分享，为那些较晚加入的学生提供了保证。
- **MOOC 学生的 Research Track 申请已关闭**：*遗憾的是，* **research track 申请** 的决定已经做出，关于 MOOC 学生项目规则的细节尚待教授确认。
   - 成员们对 research track 是否向非 Berkeley 学生开放表示担忧，表明目前仍存在不确定性。
- **Application Track 仍待公布详情**：关于 **application track** 的询问显示，针对 MOOC 学生的说明仍在等待中，这可能会影响最终项目。
   - 成员们希望尽快发布更多关于参与的信息，以明确申请流程。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1343746798795358340)** (11 条消息🔥): 

> `教学风格反馈、研究小组提案、研究轨道资格、应用栈查询、MOOC 课程发布` 


- **对教学风格的正面反馈**：一位成员称赞了某位讲师，表示：*“她的教学风格非常清晰。我很喜欢！”*。
   - 另一位成员表示赞同，确认该讲师的教学确实非常有效。
- **研究小组提案提交**：成员们询问了如何提交研究小组的提案，以及是否已发送相关邮件。
   - 提案流程存在困惑，一位成员表示他们没有收到任何包含细节的邮件。
- **研究轨道资格**：根据之前的公告，已明确 **Research Track 仅限 Berkeley 学生参加**。
   - 这导致一些非 Berkeley 的参与者感到失望。
- **应用栈向 MOOC 学生开放**：成员们讨论了 Application Stack 是否会对 MOOC 学生开放，*wei3398* 表示应该是开放的。
   - 他们指出，由于 **课程尚未发布**，更多细节尚待确定，并引用了一个 Discord 链接。
- **MOOC 课程详情即将公布**：一位成员宣布 **MOOC 课程详情即将发布**，其中包括项目部分。
   - 他们感谢大家在等待后续信息时的耐心。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1343656421035737179)** (2 条消息): 

> `易用性、指令提示词` 


- **用户发现其令人惊叹且易于尝试**：一位用户对这款新工具表示兴奋，称其看起来 **非常出色（amazing）**，且对于非编程人员来说也足够简单易用。
   - *“也许我会试试”* 反映了用户尽管印象良好，但在完全投入使用前仍有犹豫。
- **注意到最短指令提示词**：另一位用户评论了一个非常短的 **指令提示词（Instruction Prompt）**，称其为他们见过的 **最短** 提示词。
   - 这表明工具使用中的用户引导可能正趋向于更加精简。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1343654176814796830)** (14 条消息🔥): 

> `Google Deep Research & Gemini, NotebookLM 语言设置, 扫描纸质书, NotebookLM 中的多语言提示词, Claude 3.7 热度` 


- **Google Deep Research 与 Gemini 集成**：围绕将 [Google Deep Research](https://link.to/research) 和 Gemini 与 NotebookLM 集成以增强功能展开了讨论。
   - 成员们对该领域的未来发展表示期待。
- **更改 NotebookLM 语言设置**：有成员提出了关于在不影响 Google 账户语言的情况下更改 NotebookLM 语言设置的问题。
   - 一位用户寻求关于实现此类语言更改的有效方法的建议。
- **数字化纸质书的创意方法**：一位成员提议使用 Lens 应用拍摄每一页的照片来创建 PDF，然后将其转换为 PowerPoint 以上传到 NotebookLM。
   - 也有人提出了替代方案，例如使用复印机或 Adobe Scan 应用直接创建 PDF。
- **在 NotebookLM 中有效使用提示词**：关于是使用单个还是多个提示词来促使主持人说德语展开了讨论。
   - 一位成员指出，效果可能与他们的 Premium 订阅状态有关。
- **围绕 Claude 3.7 的热议**：关于 **Claude 3.7** 的讨论非常热烈，用户希望在选择模型方面拥有更多控制权。
   - 一位用户引发了关于此类决策对用户体验影响的讨论。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1343644238331777136)** (3 messages): 

> `AI 助手可用性, ComposioHQ 更新, Claude Sonnet 3.7 发布, 与 Anthropic 的集成, 安装说明` 


- **AI 助手现已面向用户上线！**: 文档中出色的 AI 助手现在可供所有人使用！点击[这里](https://t.co/XAyW7wLALJ)查看。
- **ComposioHQ 发布又一力作！**: [ComposioHQ](https://t.co/W4l129gHce) 发布了又一个重磅更新！其功能和特性持续给人留下深刻印象。
- **Claude Sonnet 3.7 来了！**: 我们在 [AnthropicAI](https://twitter.com/anthropicAI) 的朋友们发布了 **Claude Sonnet 3.7**，目前的反馈和评估都非常积极！
   - 用户可以通过运行 `pip install llama-index-llms-anthropic --upgrade` 轻松获得零日支持（day 0 support）。
- **查看 Anthropic 的公告！**: 想要了解更多详情的用户可以在[这里](https://t.co/PjaQWmmzaN)找到 Anthropic 的公告帖子。内容非常详尽。
   - 关注此新版本带来的最新集成能力！


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1343653007631581296)** (5 messages): 

> `结合 Elastic Search 的 BM25 检索器, MultiModalVectorStoreIndex 问题` 


- **BM25 检索器在没有节点的情况下无法初始化**: 一位成员指出，**BM25 检索器**无法仅从向量存储（vector store）初始化，因为 **docstore** 必须包含已保存的节点。
   - 另一个建议是将 **top k 设置为 10000** 以检索所有节点，尽管这可能效率不高。
- **创建 MultiModalVectorStoreIndex 时出现图像错误**: 一位成员在尝试创建 **MultiModalVectorStoreIndex** 时遇到了与图像文件相关的错误，尽管图像已存储在 GCS bucket 中。
   - 他们报告称，该问题专门针对图像出现，因为他们的代码在处理 **PDF 文档**时运行正常。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1343641197092012105)** (6 messages): 

> `微调的截断方法, StatefulDataLoader 的 PR 评审` 


- **关于微调截断方式的讨论**: 一位成员提出了一个问题，即微调是否应该**默认使用左截断（left truncation）**而不是目前的**右截断（right truncation）**，并引用了一张图表作为支持。
   - 讨论中的其他人承认 **HF** 目前的默认设置是右截断，并对更改默认设置持不同意见。
- **请求评审 StatefulDataLoader**: 一位成员请求对其 [pull request](https://github.com/pytorch/torchtune/pull/2410) 进行评审，该 PR 添加了对 **StatefulDataLoader** 类的支持，并强调了其重要性。
   - 该 pull request 的背景包括添加新功能和修复潜在的 Bug。



**提到的链接**: <a href="https://github.com/pytorch/torchtune/pull/2410">由 joecummings 提交的添加 ``StatefulDataLoader`` 支持 · Pull Request #2410 · pytorch/torchtune</a>: 背景：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。此 PR 为 StatefulDataLoader 类添加了支持...

  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1343701265808490556)** (2 messages): 

> `DeepScaleR Model, DeepEP Communication Library` 


- **DeepScaleR 通过 RL 超越 O1-Preview**：全新的 **DeepScaleR** 模型在 **Deepseek-R1-Distilled-Qwen-1.5B** 的基础上通过简单的强化学习（Reinforcement Learning）进行微调，在 **AIME2024** 上实现了 **43.1% 的 Pass@1 准确率**，相比 O1-preview 提升了 **14.3%**。
   - 这一进展突显了强化学习技术在扩展模型能力方面的有效应用。
- **推出 DeepEP：开源通信库**：秉承 #OpenSourceWeek 的精神，**Deepseek AI** 发布了 **DeepEP**，这是首个专为 **Mixture of Experts (MoE)** 模型训练和推理设计的开源通信库。
   - DeepEP 具有 **FP8 调度支持**以及高效的节点内和节点间通信等特性，旨在优化 AI 模型部署中的训练和推理阶段。欢迎在 [GitHub](https://github.com/deepseek-ai/DeepEP) 上查看。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/deepseek_ai/status/1894211757604049133">来自 DeepSeek (@deepseek_ai) 的推文</a>：🚀 #OpenSourceWeek 第 2 天：DeepEP。很高兴推出 DeepEP - 第一个用于 MoE 模型训练和推理的开源 EP 通信库。✅ 高效且优化的 all-to-all 通信 ✅...</li><li><a href="https://pretty-radio-b75.notion.site/DeepScaleR-Surpassing-O1-Preview-with-a-1-5B-Model-by-Scaling-RL-19681902c1468005bed8ca303013a4e2">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一个将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作区。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1343747667548700704)** (5 messages): 

> `POS Validators profitability, Pool validator nodes, Asset value assessment` 


- **关于 DeSci 中 POS 验证者盈利门槛的咨询**：一位用户询问了 **DeSci** 领域内 **POS 验证者** 的 **盈利门槛**。
   - 这反映了人们对去中心化科学中节点运营经济可行性的持续关注。
- **关于池化验证者节点的讨论**：另一位用户提到了 **池化验证者节点 (pool validator nodes)**，表明了对验证者之间协作或资源共享的兴趣。
   - 这可能暗示了通过池化策略优化验证者效率的趋势。
- **资产价值专家关注点**：一条消息中包含了 **资产价值专家 (asset value expert)** 一词，但被一些无关术语遮掩，导致清晰度受限。
   - 这引发了关于在讨论话题中评估资产估值专业知识相关性的疑问。


  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1343745328465313873)** (1 messages): 

> `GPT4All v3.10.0 Release, Remote Model Configuration, CUDA Compatibility Updates, Translation Improvements, Chat Template Enhancements` 


- **GPT4All v3.10.0 发布，包含重要更新**：**GPT4All v3.10.0** 版本已发布，具有多项显著增强功能，包括更好的远程模型配置和新的模型支持。
   - 还整合了 *崩溃修复*，以确保整个平台的稳定性能。
- **增强的远程模型配置**：在“添加模型”页面中增加了一个专门的 **远程模型提供商** 选项卡，使得配置来自 **Groq**、**OpenAI** 和 **Mistral** 的模型变得更加容易。
   - 这一改进使得外部解决方案能够更无缝地集成到 GPT4All 生态系统中。
- **CUDA 兼容性扩展**：最新更新现在支持具有 **CUDA 计算能力 5.0** 的 GPU，包括 **GTX 750**，提高了更多用户的可访问性。
   - 这扩大了可以高效运行 GPT4All 的硬件范围。
- **翻译和聊天模板改进**：更新了 **意大利语** 和 **简体中文** 翻译，显著提升了非英语用户的可用性。
   - 此外，还增强了 **OLMoE** 和 **Granite** 模型的默认聊天模板，以提供更好的用户体验。
- **空格处理和崩溃修复实施**：改进了 **基于 DeepSeek-R1 模型** 的 *空格行为*，以确保更整洁的输出。
   - 还解决了几个与崩溃相关的问题，以增强系统稳定性。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1343664013497733291)** (4 messages): 

> `GPT4All 中的 Multi-Agent 框架, AI 与编程理解, 版本控制疑虑, Nomic Embed 更新` 


- **探索 GPT4All 中的 Multi-Agent 框架**：一位用户询问了在 GPT4All 中使用 **Multi-Agent 框架**的可能性，并寻求关于其实现的明确说明。
   - *讨论中未提供任何回复或解决方案。*
- **AI 与编程语言的现状**：一位用户表达了对使用 GPT4All 创建 AI 解决方案挑战的担忧，指出它并不能完全理解编程语言。
   - 他们承认虽然 **GPT4All** 简化了 AI 开发，但仍需要大量的学习和努力。
- **GPT4All 的版本号困惑**：一位成员质疑 **v3.10.0** 是否实际上应该代表 **v4.0**，反映了对版本系统的困惑。
   - 另一位用户表示不确定，但表达了对 GPT4All **v4.0.0** 的渴望，并提到了最近发布的 Nomic Embed **v2**。
- **对 Nomic Embed v2 的期待**：一位用户正在等待 **GPT4All v4.0.0** 的发布，并强调尽管新版本已经发布，但目前仍在使用 Nomic Embed **v1.5**。
   - *他们提醒其他人，在等待这些更新时耐心是必不可少的。*


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1343633733403672576)** (2 messages): 

> `phi4 响应格式, 从 2.5 风格的 Assertions 迁移, dspy.BestOfN, dspy.Refine` 


- **phi4 响应格式有所不同**：有人提到 **phi4 响应**格式与大多数格式不同，但具体细节尚待公布。
   - 目前正在准备一份教程以澄清其用法。
- **Assertion 迁移的新选项**：对于那些从 **2.5 风格的 Assertions** 迁移的用户，现在可以利用 `dspy.BestOfN` 或 `dspy.Refine` 来简化模块。
   - 这用更高效的选项取代了传统的 Assertions 功能。
- **dspy.BestOfN 的实现**：提供了一个示例，展示如何在 **ChainOfThought** 或多步模块中实现 `dspy.BestOfN`，允许最多 **5 次重试**。
   - 该方法将选择最佳奖励，并在达到 **threshold**（阈值）时停止。
- **Reward 函数详解**：分享了一个 `reward_fn` 示例，说明了它如何返回诸如 float 或 bool 之类的标量值，以评估 **prediction field lengths**（预测字段长度）。
   - 该函数适用于 **dspy.BestOfN** 实现的上下文。


  

---


---


---


---


{% else %}


> 完整的频道逐条细分内容已针对邮件进行了截断。
> 
> 如果你想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}