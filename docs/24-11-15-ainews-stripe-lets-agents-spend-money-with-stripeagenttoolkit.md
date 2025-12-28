---
companies:
- stripe
- openai
- anthropic
- meta-ai-fair
date: '2024-11-16T01:02:33.643600Z'
description: '**Stripe** 开创了专门为处理支付的智能体（agents）设计的 AI SDK，通过集成 **gpt-4o** 等模型来实现金融交易和基于
  Token 的计费。AI 开发者工具的趋势正强调构建更好的“AI-计算机接口”（AI-Computer Interfaces）以提升智能体的可靠性，其中 **E2B**
  和 `llms.txt` 文档趋势备受关注，并已被 **Anthropic** 采用。


  在 AI 模型新闻方面，**Gemini-Exp-1114** 登顶了视觉排行榜，并在数学竞技场（Math Arena）中表现有所提升；与此同时，关于模型过拟合以及通用人工智能（**AGI**）扩展定律（scaling
  laws）局限性的讨论仍在继续。**OpenAI** 发布了支持 **VS Code**、**Xcode** 和 **Terminal** 集成的 **macOS
  版 ChatGPT 桌面应用**，进一步优化了开发者工作流和结对编程。


  **Anthropic** 推出了一款利用思维链（chain-of-thought）推理的提示词优化工具。**Meta AI** 分享了 **EMNLP2024**
  关于图像字幕生成、对话系统和内存高效微调的顶级研究成果。**ICLR 2025** 的亮点包括基于扩散的光照协调、开源混合专家（MoE）语言模型以及双曲视觉语言模型。此外，一种新的自适应解码方法优化了每个
  Token 的创造力和事实性。针对文档解析和检索增强生成（RAG），业界还推出了 **LlamaParse** 和 **RAGformation** 等新工具。'
id: e4872678-788f-483d-a30f-726fe0dcbfa6
models:
- gpt-4o
- gemini-exp-1114
original_slug: ainews-stripe-lets-agents-spend-money-with
people:
- abacaj
- francois-fleuret
- lmarena_ai
- goodside
- jxmnop
- jaseweston
- stevenheidel
title: Stripe 允许智能体（Agents）通过 StripeAgentToolkit 进行支付。
topics:
- ai-computer-interfaces
- agentic-ai
- model-overfitting
- benchmarks
- scaling-laws
- agi
- chain-of-thought
- image-captioning
- dialogue-systems
- memory-efficient-fine-tuning
- diffusion-models
- mixture-of-experts
- adaptive-decoding
- creativity-optimization
- factuality-optimization
- pair-programming
- document-parsing
- retrieval-augmented-generation
---

<!-- buttondown-editor-mode: plaintext -->**AI SDK 便是你所需的一切。**

> 2024/11/14-2024/11/15 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**217** 个频道和 **1812** 条消息）。预计节省阅读时间（以 200wpm 计算）：**191 分钟**。您现在可以在 AINews 讨论中标记 [@smol_ai](https://x.com/smol_ai)！

今年 AI 开发者工具领域兴起的一个论点是，拥有更好“[AI-Computer Interfaces](https://www.latent.space/p/shunyu)”的工具将作为 Agent 可靠性/准确性的中期解决方案表现得更好。你可以在 [E2B](https://github.com/e2b-dev/e2b) 等工具以及由 Jeremy Howard 发起、[现已被 Anthropic 采用](https://x.com/alexalbert__/status/1857457290917589509)的 `llms.txt` 文档趋势中看到这一点。Vercel 拥有通用的 AI SDK，但 Stripe 是第一家[专门为涉及资金往来的 Agent 创建 SDK](https://stripe.dev/blog/adding-payments-to-your-agentic-workflows)的开发者工具公司：

```js
import {StripeAgentToolkit} from '@stripe/agent-toolkit/ai-sdk';
import {openai} from '@ai-sdk/openai';
import {generateText} from 'ai';

const toolkit = new StripeAgentToolkit({
  secretKey: "sk_test_123",
  configuration: {
    actions: {
      // ... enable specific Stripe functionality
    },
  },
});

await generateText({
  model: openai('gpt-4o'),
  tools: {
    ...toolkit.getTools(),
  },
  maxSteps: 5,
  prompt: 'Send <<email address>> an invoice for $100',
});
```
 
以及支出资金：


![image.png](https://assets.buttondown.email/images/7fa72309-f4c2-4202-a368-a6720623946c.png?w=960&fit=max)


还有基于 token 使用情况进行收费。这是一个非常有前瞻性的举动，解决了常见的痛点。回想起来，Stripe 成为第一家为 AI Agents 构建金融服务的公司并不令人意外。

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

**AI 模型与基准测试**

- **模型过拟合与性能**：[@abacaj](https://twitter.com/abacaj/status/1857429103462215736) 强调了对模型**过拟合 (overfit)** 的担忧，即模型仅在特定的**基准测试 (benchmarks)** 中表现良好。[@francoisfleuret](https://twitter.com/francoisfleuret/status/1857185503784714545) 对**规模法则 (scaling laws)** 已经失效的观点提出质疑，认为单纯增加模型规模可能无法实现 **AGI**。

- **Gemini 与 Claude 对比**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1857110672565494098) 报告称 **Gemini-Exp-1114** 在**视觉排行榜 (Vision Leaderboard)** 中获得**第一名**，并在 **Math Arena** 中的排名有所提升。相比之下，[@goodside](https://twitter.com/goodside/status/1857254346838208756) 批评了 **LLM 的 IQ 类比**，指出 LLM 的智能在不同任务之间存在显著差异。

**AI 公司新闻**

- **OpenAI 更新**：[@OpenAI](https://twitter.com/OpenAIDevs/status/1857129790312272179) 宣布发布 **macOS 版 ChatGPT 桌面应用**，该应用现在可与 **VS Code**、**Xcode** 和 **Terminal** 等工具集成，以增强开发者的工作流。

- **Anthropic 与 Meta 动态**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1857108263042502701) 在 Anthropic Console 中引入了新的 **prompt improver**，旨在利用**思维链 (chain-of-thought)** 推理来优化提示词。同时，[@AIatMeta](https://twitter.com/AIatMeta/status/1857126323023683716) 分享了来自 **EMNLP2024** 的**顶级研究论文**，涵盖了**图像字幕 (image captioning)**、**对话系统**以及**内存高效微调 (memory-efficient fine-tuning)** 等方面的进展。

**AI 研究与论文**

- **ICLR 2025 亮点**：[@jxmnop](https://twitter.com/jxmnop/status/1857447673311191253) 评述了 **ICLR 2025** 的**高分论文**，包括关于**基于扩散的照明协调 (diffusion-based illumination harmonization)**、**开源混合专家语言模型 (OLMoE)** 以及**双曲视觉语言模型 (hyperbolic vision-language models)** 的研究。

- **自适应解码技术**：[@jaseweston](https://twitter.com/jaseweston/status/1857257120338780209) 介绍了 **Adaptive Decoding via Latent Preference Optimization**，这是一种新方法，通过为每个 token 自动选择**创造性或事实性**参数，其表现优于固定温度解码。

**AI 工具与软件更新**

- **ChatGPT 桌面版增强**：[@stevenheidel](https://twitter.com/stevenheidel/status/1857178263959003629) 展示了 **ChatGPT 桌面应用的新功能**，包括**高级语音模式 (Advanced Voice Mode)** 以及与 **VS Code**、**Xcode** 和 **Terminal** 交互的能力，从而实现无缝的**结对编程 (pair programming)** 体验。

- **LlamaParse 与 RAGformation**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1857172518744056049) 介绍了 **LlamaParse**，这是一款用于解析复杂文档的工具，支持**手写内容**和**图表**等特征。此外，[@llama_index](https://twitter.com/llama_index/status/1857118876494078315) 推出了 **RAGformation**，它可以根据自然语言描述**自动配置云端设置**，简化了**云复杂性**并优化了 **ROI**。

**AI Agent 与应用**

- **生产环境中的 AI Agent**：[@LangChainAI](https://twitter.com/LangChainAI/status/1857117443065540707) 透露，**51% 的公司**已经在**生产环境中部署了 AI Agent**，其中**中型公司**以 **63%** 的采用率领先。主要应用场景包括**研究与摘要 (58%)**、**个人生产力 (53.5%)** 和**客户服务 (45.8%)**。

- **Agent 工作流中的 Gemini 与 Claude**：[@AndrewYNg](https://twitter.com/AndrewYNg/status/1857117382378164267) 讨论了如何针对 **Agent 工作流 (agentic workflows)** 优化 **Gemini** 和 **Claude** 等 **LLM**，增强**函数调用 (function calling)** 和**工具使用 (tool use)** 等能力，以提升各种应用中的 **Agent 性能**。

**迷因与幽默**

- **关于 AI 的幽默观点**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1857111141140316419) 分享了一个关于 **Transformers.js** 的轻松迷因，而 [@rez0__](https://twitter.com/rez0__/status/1857190746841079930) 则幽默地评论了受 AI 影响的**清洁习惯**。

- **AI 相关笑话**：[@hardmaru](https://twitter.com/hardmaru/status/1857232575988920620) 拿历史上的 **NVIDIA** 持股开玩笑，[@fabianstelzer](https://twitter.com/fabianstelzer/status/1857177429854351452) 发布了一个有趣的 AI **Prompt** 场景，展示了 **LLM** 中**风格迁移 (style transfer)** 的怪癖。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Gemini Exp 1114 在 Chatbot Arena 中获得最高排名**

- **Gemini Exp 1114 现在在 Chatbot Arena 排名并列总榜第一（不过这名字……）** ([Score: 322, Comments: 101](https://reddit.com/r/LocalLLaMA/comments/1grahpc/gemini_exp_1114_now_ranks_joint_1_overall_on/)): 由 **GoogleDeepMind** 开发的 **Gemini Exp 1114** 在 **Chatbot Arena** 中获得了 **并列总榜第一** 的排名，其分数显著提升了 40 多分，追平了 4o-latest 并超越了 o1-preview。它还在 **Vision 排行榜** 中领跑，并在 Math、Hard Prompts 和 Creative Writing 类别中晋升至第一，同时在 Coding 领域的排名提升至第三。
  - 讨论中也出现了对 **Gemini Exp 1114** 表现的怀疑，一些用户质疑其改进是否源于在 **Claude 的数据** 或其他合成数据集上进行的训练。一些用户幽默地表示，该模型的身份和能力可能被夸大或误解了，正如有关其命名和表现的梗和笑话所展示的那样。
  - 技术辩论涉及 **context length**（上下文长度）和响应时间，指出 **Gemini Exp 1114** 具有 32k 的输入上下文长度，且被认为速度较慢，可能专注于“思考”过程。在推理能力方面，用户将其与 **OpenAI 的 o1** 进行了比较，指出 Gemini Exp 1114 即使没有显式提示，也能有效地使用“思维链”（chain of thought）推理。
  - 用户对 **命名规范** 和模型变体表示关注，提到了 **Nemotron** 并与 **Llama** 模型进行了比较。人们对 Gemini 模型如 "pro" 或 "flash" 的命名感到好奇，并猜测该版本是否是诸如 "1.5 Ultra" 或 "2.0 Flash/Pro" 之类的新迭代。

**主题 2. Omnivision-968M 优化边缘设备视觉处理**

- **Omnivision-968M：边缘设备视觉语言模型，Token 减少 9 倍** ([Score: 214, Comments: 47](https://reddit.com/r/LocalLLaMA/comments/1grkq4j/omnivision968m_vision_language_model_with_9x/)): **Omnivision-968M** 模型专为边缘设备优化，实现了 **图像 Token 减少 9 倍**（从 729 降至 81），增强了在 Visual Question Answering 和 Image Captioning 任务中的效率。它处理图像速度极快，在 M4 Pro Macbook 上为一张 1046×1568 像素的海报生成字幕仅需不到 2 秒，仅占用 988 MB RAM 和 948 MB 存储空间。更多信息和资源可在 [Nexa AI 的博客](https://nexa.ai/blogs/omni-vision)及其 [HuggingFace 仓库](https://huggingface.co/NexaAIDev/omnivision-968M)中找到。
  - 讨论中有人好奇使用 **消费级 GPU**（如几块 3090）构建 **Omnivision-968M** 模型的可行性，还是需要租用更强大的云端 GPU（如 **H100/A100**）进行训练。该模型与 **Llama CPP** 的兼容性及其在 **OCR** 任务中的表现也受到了关注。
  - 讨论还包括可能发布的 **音频 + 视觉投影模型** 以及 **vision/text parameters** 的划分。用户提到了 **Qwen2.5-0.5B** 模型，并对 **Nexa SDK** 的使用表示兴趣，文中提供了 [GitHub 仓库](https://github.com/NexaAI/nexa-sdk)链接。
  - 有人对向 **llama.cpp** 项目回馈贡献表示担忧，一些用户批评开源贡献缺乏互惠性。此外，还有关于 **Coral TPU** 因内存较小而在运行模型方面存在局限性的讨论，建议将入门级 **NVIDIA 显卡** 作为更具成本效益的解决方案。

**主题 3. Qwen 2.5 7B 霸榜 Livebench 排名**

- **[Qwen 2.5 7B 已加入 Livebench，排名超越 Mixtral 8x22B 和 Claude 3 Haiku](https://i.redd.it/bsejiqpgr01e1.png)** ([Score: 154, Comments: 35](https://reddit.com/r/LocalLLaMA/comments/1grr7yb/qwen_25_7b_added_to_livebench_overtakes_mixtral/)): **Qwen 2.5 7B** 已被添加到 **Livebench**，并在排名中超越了 **Mixtral 8x22B** 和 **Claude 3 Haiku**。
  - 用户质疑 **Qwen 2.5 7B** 在基准测试之外的实际效用，指出其在构建基础 **Streamlit** 页面和解析职位发布等任务中表现不佳。**WizardLM 8x22B** 被认为是更理想的替代方案，因为尽管其基准测试分数较低，但在实际应用中表现更优。
  - 几位用户对基准测试的有效性表示怀疑，不相信像 **Qwen 2.5 7B** 这样的小型模型能超越 **GPT-3.5** 或 **Mixtral 8x22B** 等大型模型。他们强调了基准测试结果与实际可用性之间的脱节，特别是在对话和指令任务中。
  - 讨论涉及在特定硬件配置（如 **Apple M3 Max** 和 **NVIDIA GTX 1650**）上运行 **Qwen 2.5 14B** 和 **32B** 等模型的技术细节，以及使用 **fp16** 或 **Q4_K_M** 格式模型的考量。用户还提到 **Gemini-1.5-flash-8b** 是基准测试中的强劲对手，并指出了其多模态能力。
- **Claude 3.5 竟然知道我的姓氏——隐私怪象** ([Score: 118, Comments: 141](https://reddit.com/r/LocalLLaMA/comments/1gr9pze/claude_35_just_knew_my_last_name_privacy_weirdness/)): 该帖子讨论了使用 **Claude 3.5 Sonnet** 的一次令人担忧的经历：尽管用户在会话中仅提供了名字，AI 却在生成的 MIT 许可证中意外包含了用户罕见的姓氏。这引发了关于 AI 是否能访问过去的交互记录或 **GitHub** 个人资料等外部来源的疑问（尽管用户认为自己已选择退出此类数据使用），并促使该用户寻求他人的类似经历或见解。
  - 评论者推测 **Claude 3.5 Sonnet** 可能是通过 **GitHub profiles** 或其他公开数据获取了用户的姓氏，尽管用户努力保护隐私。一些用户建议 AI 可能会通过关联用户的代码风格和公共仓库来推断其身份，而另一些人则怀疑 AI 是否能访问私有数据或账户凭据。
  - 讨论还涉及账户注册时的 **metadata**（元数据）或个人详情（如电子邮件地址或支付信息）是否被用于识别用户。一些评论指出，**LLM** 通常不会直接接收此类元数据，任何明显的个性化可能只是巧合或基于公开数据。
  - 用户还辩论了 **LLM** 在解释其思维过程方面的可靠性，一些人指出模型可能会编造解释或依赖训练数据的关联性。有人建议联系 **Anthropic** 以寻求澄清，因为该事件引发了对隐私和数据使用的担忧。


## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Claude 超越 GPT-4O：代码生成质量的重大转变**

- **3.5 sonnet vs 4o 编程对比：是天差地别还是略胜一筹？** ([Score: 26, Comments: 42](https://reddit.com/r/ClaudeAI/comments/1grqfxi/35_sonnet_vs_4o_in_coding_significant_different/)): **Claude 3.5 Sonnet** 在编程能力上表现出优于 **GPT-4** 的水平。在限制方面，**Claude Pro** 为 **50 条消息/5 小时**，而 **ChatGPT Plus** 为 **80 条消息/3 小时**，此外还有 **O1-mini** 的 **50 条/天**和 **O1-preview** 的 **7 条/天**。帖子作者寻求建议，想知道对于中高级水平的 **Python**、**JavaScript** 和 **C++** 开发，这种性能差异是否值得切换到 **Claude Pro**。
  - 用户反馈 **Claude 3.5 Sonnet** 在编程任务中始终优于 **GPT-4**，一位用户指出 **GPT-4** 的编程能力自最初发布以来已显著退化，当时它还能有效处理 **Matlab-to-Python** 转换和 **PyQT5** 实现等复杂任务。
  - 几位开发者强调了 **Sonnet** 卓越的代码理解和错误修复能力，尽管有人提到在进行高层级架构讨论时会使用 **O1-preview**。用户建议配合 **Sonnet** 使用 **Cursor**，作为应对使用限制的替代方案。
  - 尽管 **Claude Pro** 有 **45 条消息/5 小时**的严格限制，用户仍压倒性地倾向于选择它而非 **GPT-4**，理由是其代码质量和项目理解力更高。一些开发者采用混合方案，在等待 **Claude** 额度重置时切换到 **GPT-4**。
- **Chat GPT plus 出现省略代码、删除函数甚至给出空回复的情况，即使是 o1-preview 也是如此** ([Score: 27, Comments: 21](https://reddit.com/r/OpenAI/comments/1grwdhs/chat_gpt_plus_is_skipping_code_removing_functions/)): **ChatGPT Plus** 用户报告了 **代码生成** 方面的问题，模型在处理**大型代码库**（特别是 **700 行脚本**）时会截断回复、删除无关函数，偶尔还会提供空回复。即使使用 **GPT-4 preview** 模型，问题依然存在，请求完整代码时仅返回修改后的函数，而丢失了原始上下文。
  - 用户反映各模型的**代码质量**均有所下降，有人猜测 **OpenAI** 可能在故意降低性能。另一些人注意到，由于负载较低，在**美国夜间**时段使用 **API** 效果更好。
  - 使用这些模型的最佳实践是将代码拆分为更小、可管理的块。这种方法通过防止文件变得过大或过于碎片化，自然地引导出更好的**架构**。
  - 尽管 **GPT-4 preview** 模型在详细代码分析和复杂主题讨论方面有优势，但 **Claude** 和标准版 **GPT-4** 产出的代码可能更好。


**Theme 2. FrontierMath 基准测试：模型在高等数学中仅得 2 分**

- **[FrontierMath 是一个新的 LLM 数学基准测试，用于测试其极限。目前得分最高的模型仅获得 2%。](https://i.redd.it/diueskl7fz0e1.png)** ([Score: 357, Comments: 117](https://reddit.com/r/OpenAI/comments/1grmvs8/frontiermath_is_a_new_math_benchmark_for_llms_to/)): **FrontierMath** 作为一个测试 **LLM** 的新数学基准，揭示了 **LLM** 数学能力的显著局限性，表现最好的模型准确率仅为 **2%**。该基准旨在评估超越标准测试的高级数学能力，凸显了当前 AI 系统明显的性能差距。
  - 包括 **Terence Tao** 和 **Timothy Gowers** 在内的**菲尔兹奖**得主确认，这些问题“极具挑战性”，超出了典型的 **IMO** 题目。该基准需要研究生、**AI** 和代数软件包协作才能有效解决。
  - **FrontierMath** 中的问题由**数学博士**专门为 **AI 基准测试**设计，需要多位领域专家长期合作。样例问题可在 [epoch.ai](https://epoch.ai/frontiermath/the-benchmark) 查看，但完整数据集仍不公开。
  - 讨论集中在即使在大多数**数学博士**都无法独立解决的问题上达到 **2%** 准确率的意义。用户争论达到这一水平的 AI 是否应被视为协作团队成员而非仅仅是工具，并引用了 [Chris Olah](https://youtu.be/ugvHCXCOmm4) 关于神经网络的研究。


**Theme 3. Chat.com 域名以 1500 万美元售予 OpenAI - 重大企业举措**

- **[印度男子以 12.6 亿卢比出售 Chat.com](https://i.redd.it/jr4r18gks11e1.png)** ([Score: 550, Comments: 173](https://reddit.com/r/ChatGPT/comments/1grtzkl/indian_man_sells_chatcom_for_126_crore/)): **Chat.com** 域名以 **1500 万美元（12.6 亿卢比）**的价格售出，部分款项以 **OpenAI** 股份支付。该域名由一名**印度籍**所有者售出，标志着 2023 年一笔重大的域名交易。
  - **Dharmesh Shah**，**HubSpot** 的 **CTO** 兼科技亿万富翁，在去年以约 **1400 万美元**购入该域名后将其售出。交易包括部分 **OpenAI 股份**，据悉 Shah 与 **Sam**（推测为 Altman）是朋友。
  - 多位评论者批评标题过度关注卖家的国籍，而非其卓越的专业背景。对于据报道身价超过 **10 亿美元**的 Shah 来说，这笔交易规模相对较小。
  - 讨论透露，这并非一项长期的域名投资，反驳了最初关于该域名自互联网早期就被持有的假设。考虑到最近的买入价格，实际利润率相对较低。

**主题 4. Claude 回滚：3.6 版本问题导致版本撤回**

- **笑死，他们现在要撤回 Sonnet 3.6 了？** ([Score: 62, Comments: 55](https://reddit.com/r/ClaudeAI/comments/1grsw8q/lmao_they_are_now_pulling_back_sonnet_36/)): **Anthropic** 似乎回滚了 **Claude 3.6 Sonnet** 并移除了 **Haiku** 的版本编号，截图显示模型选择界面中删除了“(new)”标识。这些变化表明其 **Claude** 模型可能进行了版本调整或静默更新，尽管官方未提供解释。
  - 用户报告称，**Claude Sonnet** 可能仍是 **3.6** 版本，只是移除了“new”标签，模型对事件的了解及其 **10 月 22 日**的版本标识符证实了这一点。
  - 社区成员批评 **Anthropic** 的沟通和版本命名策略，许多人指出该公司近期在透明度和内部组织方面存在问题。一位用户幽默地通过道歉短语来区分不同版本：旧版 **3.5** 使用 *“你是对的，我道歉”*，而新版 **3.5** 使用 *“啊，我现在明白了！”*。
  - 讨论揭示了潜在的性能差异，有报告称该模型在消息输出方面存在限制，且无法通过简单的测试（如计算单词中的字母数量）。页面顶部观察到 **High Demand Notice**（高需求通知），表明系统负载较重。
- **[既然刚登录就看到这种愚蠢的消息，付费还有什么意义？这简直糟糕透顶。我今天甚至还没用过 Claude 就已经被限制了。](https://i.redd.it/r9jn27y8921e1.png)** ([Score: 103, Comments: 50](https://reddit.com/r/ClaudeAI/comments/1grvhm8/whats_the_point_of_paying_if_i_just_logged_in_and/)): **Claude** 用户报告称，尽管拥有付费订阅且当天未使用，仍立即受到访问限制和服务约束。**Anthropic** 的服务限制似乎同时影响了新老付费用户，且没有明确解释或事先通知。
  - 用户报告称，**Claude 付费订阅**很快就会达到使用限制，部分用户在 **上午 11 点** 之前就被限制。多位用户建议使用 **2-3 个账号**（每个 20 美元）或切换到更昂贵的 **API** 作为变通方案。
  - 社区讨论强调需要更好的**使用情况追踪功能**，建议增加一个**进度条**，显示在降级到简洁模式（concise mode）之前的剩余使用量。用户批评使用限制缺乏透明度，且在受限时无法退出**简洁模式**。
  - 技术用户讨论了本地替代方案，推荐使用具有特定硬件要求的 **Ollama**：**NVIDIA 3060**（**12GB VRAM**，$200）或 **3090**（**24GB VRAM**，$700）。对于显存充足的用户建议使用 **Qwen 2.5 32B** 模型，而 **Qwen 14B 2.5** 则被推荐作为更轻量级的替代方案。

---

# AI Discord 摘要回顾

> 由 O1-mini 生成的摘要之摘要的摘要

**主题 1：AI 模型的硬件与性能优化**

- [**技嘉发布 AMD Radeon PRO W7800 AI TOP 48G**](https://www.techpowerup.com/328837/gigabyte-launches-amd-radeon-pro-w7800-ai-top-48g)：技嘉推出了 **AMD Radeon PRO W7800 AI TOP 48G**，配备 **48 GB GDDR6 显存**，面向 **AI 和工作站专业人士**。

**主题 2：模型发布与集成增强**

- [**DeepMind 开源 AlphaFold 代码**](https://www.perplexity.ai/page/deepmind-releases-alphafold-co-rtUBaB6hQDiwRZcst1bXbg)：**DeepMind** 发布了 **AlphaFold** 代码，让更多人能够使用其**蛋白质折叠技术**，预计将加速**生物技术**和**生物信息学**领域的研究。
- [**Google 发布 Gemini AI 应用**](https://www.perplexity.ai/search/https-techcrunch-com-2024-11-1-k6p5L5QTTpOEnUwZXrY.Lw)：**Google** 推出了 **Gemini 应用**，集成了先进的 **AI 功能**以竞争现有工具，详见 [TechCrunch 文章](https://www.perplexity.ai/search/https-techcrunch-com-2024-11-1-k6p5L5QTTpOEnUwZXrY.Lw)。

**主题 3：AI 工具集成与功能开发**

- [**ChatGPT 现已集成 macOS 桌面应用**](https://x.com/OpenAIDevs/status/1857129790312272179)：**ChatGPT for macOS** 现在支持与 **VS Code**、**Xcode**、**Terminal** 和 **iTerm2** 集成，通过直接与开发环境交互来增强**编程辅助**功能。
- [**Stable Diffusion WebUI 对决：ComfyUI vs SwarmUI**](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)：用户对比了 **ComfyUI** 和 **SwarmUI**，由于 **SwarmUI** 安装简便且在 **Stable Diffusion** 工作流中表现稳定，用户更倾向于选择它。

**主题 4：训练技术与数据集管理**

- [**Orca-AgentInstruct 提升合成数据生成**](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/)：**Orca-AgentInstruct** 引入了 **Agentic Flows** 来生成多样化、高质量的合成数据集，从而提升小型**语言模型**的训练效率。
- [**LLM 训练中的有效数据集混合策略**](https://discord.com/channels/1053877538025386074/1154120232051408927/1306997488447651900)：成员们寻求关于在 **LLM 训练**不同阶段**混合与匹配数据集**的指导，强调了在不损害模型性能的情况下优化训练过程的最佳实践。

---

# 第一部分：Discord 高层级摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen 2.5 LaTeX 渲染问题**：用户在使用 **Qwen 2.5** 时遇到 LaTeX 无法在 `$` 符号内正确渲染的问题，导致输出乱码。
  
  - 有建议提出通过创建带有明确指令的 System Prompt 来改善渲染，但目前的尝试尚未成功解决该问题。
- **LM Studio 的 Function Calling Beta 引发关注**：**LM Studio** 用户对新的 **Function Calling Beta** 功能充满热情，正在寻求个人使用经验和反馈。
  
  - 虽然部分成员认为文档清晰易懂，但也有人表示困惑，并期待在未来的更新中看到更多功能。
- **SSD 速度对比与 RAID 配置**：社区讨论了 SSD 性能，特别是 **SABRENT Rocket 5** 与 **Crucial T705** 的对比，以及 PCIe 通道限制对 RAID 设置的影响。
  
  - 用户指出，SSD 的实际性能会因具体的工作负载和 RAID 配置而产生显著差异，从而影响整体效率。
- **技嘉发布 AMD Radeon PRO W7800 AI TOP 48G**：技嘉推出了配备 48 GB GDDR6 显存的 **AMD Radeon PRO W7800 AI TOP 48G**，目标用户为 AI 和工作站专业人士。
  
  - 尽管规格令人印象深刻，但与 **NVIDIA 的 CUDA** 相比，人们对 AMD 驱动程序的可靠性和软件兼容性仍存疑虑。
- **LLM 训练的硬件考量**：参与者指出 24 GB VRAM 对于训练较大的 **LLM** 往往不足，引发了关于升级路径和租用 GPU 的讨论。
  
  - 在 **Mac Mini** 等设备上进行训练是可行的，但可能导致更高的电费支出，这促使成员们考虑更高效的硬件解决方案。

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity API 的 URL 注入问题**：用户报告称 **PPLX API** 在无法确切检索信息时，偶尔会插入**随机 URL**，导致**输出不准确**。
  
  - 讨论强调，API 添加无关 URL 的倾向削弱了其在生产环境中的可靠性，引发了对未来更新中增强准确性的呼吁。
- **DeepMind 发布 AlphaFold 代码**：**DeepMind** 已开源 **AlphaFold** 代码，使更广泛的人群能够访问其蛋白质折叠技术，正如[此处](https://www.perplexity.ai/page/deepmind-releases-alphafold-co-rtUBaB6hQDiwRZcst1bXbg)所宣布。
  
  - 此次发布预计将加速**生物技术**和**生物信息学**的研究，促进蛋白质结构预测方面的创新。
- **ChatGPT 驱动的 Chegg 衰落**：**Chegg** 的价值经历了 **99% 的缩水**，很大程度上归因于与 **ChatGPT** 的竞争，详见[文章](https://www.perplexity.ai/page/chegg-s-ai-driven-downfall-R3mSgjNyQT2tv6Vu.Wx4MQ)。
  
  - **AI** 对 Chegg 等传统教育平台的影响在社区内引发了关于在线学习资源未来的重大辩论。
- **Google Gemini 应用发布**：**Google** 正式发布了 **Gemini 应用**，引入了创新功能以与现有工具竞争，如 [TechCrunch 文章](https://www.perplexity.ai/search/https-techcrunch-com-2024-11-1-k6p5L5QTTpOEnUwZXrY.Lw)所述。
  
  - 该应用集成了 **AI** 和用户交互以提供增强功能，旨在在 AI 驱动的应用领域占据更大的市场份额。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 实验模型性能飙升**：用户报告称，新的 **gemini-exp-1114** 模型在编辑方面的准确率高达 **61%**，在各种测试中表现优于 **Gemini Pro 1.5**，尽管存在细微的格式问题。
  
  - 对比分析表明，**gemini-exp-1114** 提供了与之前版本相似的效能，具体取决于特定的使用场景。
- **Aider 中模型使用的成本影响**：讨论指出，在 **Aider** 中使用不同模型的成本在每条消息 **$0.05 到 $0.08** 之间，受文件配置影响。
  
  - 这导致用户考虑使用更经济的选择，如 ***Haiku 3.5***，以减轻小规模项目的支出。
- **与 Qwen 2.5 Coder 的无缝集成**：用户在集成 **Hyperbolic** 的 **Qwen 2.5 Coder** 时因缺少元数据而遇到问题，这些问题通过更新安装设置得到了解决。
  
  - **Aider** 主分支的更新被证明对克服这些挑战至关重要，促进了顺利集成。
- **使用 Aider 自动生成 Commit 消息**：为了给未提交的更改生成 commit 消息，用户在 **Aider** 中使用诸如 `aider --commit --weak-model openrouter/deepseek/deepseek-chat` 或 `/commit` 的命令。
  
  - 这些命令通过提交所有更改而无需提示选择单个文件，从而实现了 commit 过程的自动化。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **通过针对性训练提升 OCR 准确率**：讨论强调，通过使用合适的文档扫描应用进行针对性训练，可以显著增强 **OCR accuracy**，有可能达到近乎完美的识别率。
  
  - 参与者强调，确保合适的条件（如**正确的扫描技术**和模型 fine-tuning）对于最大化 OCR 性能至关重要。
- **通过反馈集成增强 OCR 模型**：贡献者提议将失败的 OCR 实例重新整合进 **training pipeline**，以提高模型在特定应用中的性能。
  
  - 这种反馈循环方法旨在迭代优化模型，从而提高 OCR 任务的准确性和可靠性。
- **发布用于研究的 OKReddit 数据集**：一名成员发布了 **OKReddit dataset**，这是一个包含从 2005 年到 2023 年 **5TiB** Reddit 内容的精选集合，专为研究目的设计。
  
  - 该数据集目前处于 alpha 阶段，提供了一个**经过过滤的 subreddits 列表**，并提供了[访问链接](https://huggingface.co/datasets/recursal/OKReddit-alpha)，邀请研究人员探索其潜在应用。
- **区分 RWKV 模型与 Recursal.ai 的产品**：一名成员澄清说，虽然 **RWKV models** 与特定的训练挑战相关，但它们在数据集细节方面与 **Recursal.ai** 模型有所不同。
  
  - 计划中的未来集成预示着模型训练方法的进步，增强了两种模型类型的通用性。
- **优化法律应用的 Legal Embeddings**：为法律应用有效训练 embedding 模型，需要使用在**法律数据上预训练**的 embeddings，以避免过长的训练时间和固有的偏见。
  
  - 专注于**特定领域训练（domain-specific training）**不仅能提高准确性，还能加速法律 AI 系统的开发进程。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Triton 和 CUDA 的未来焦点**：成员们计划围绕 **Triton** 和 **CUDA** 开展重要的工程工作，强调了它们对未来项目的重要性。
  
  - 存在对模型改进**收益递减**的担忧，这表明重点正转向**效率（efficiency）**。
- **语言模型偏好转移**：随着 [**Qwen/Qwen2.5-Coder-7B-Instruct**](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) 因其广泛的训练数据而获得青睐，**Mistral 7B v0.3** 模型被认为已经过时。
  
  - 社区成员比较了 **Gemma2** 和 **GPT-4o** 的**性能**，分享了关于它们**效能**的见解。
- **Unsloth 安装与 Lora+ 支持**：用户因缺少 **torch** 遇到了 **Unsloth 安装错误**，建议在当前环境中验证安装。
  
  - **Lora+ support** 已通过 pull requests 在 Unsloth 中得到确认，并讨论了其简单的实现方式。
- **为数学方程微调 Llama3.1b**：一名用户正在为解决数学方程微调 **Llama3.1b**，目前达到了 **77% 的准确率**。
  
  - 尽管他们数据集上的 loss 较低，但他们正在进行 hyperparameter sweeps，以将准确率提高到至少 **80%**。
- **数据集创建：Svelte 与歌词**：由于 **Qwen2.5-Coder** 效果不佳，使用 [Dreamslol/svelte-5-sveltekit-2](https://huggingface.co/datasets/Dreamslol/svelte-5-sveltekit-2) 创建了一个全面的 **Svelte** 文档数据集。
  
  - 针对歌词生成，正在使用 **5780 首歌曲**及相关元数据开发模型，建议使用 *Alpaca chat template*。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Scaling Laws：Twitter 宣称 Scaling 已死**：成员们讨论了 **Twitter 最近关于 Scaling** 在 AI 领域不再有效的说法，强调需要基于 **同行评审论文（peer-reviewed papers）** 的见解，而非未经证实的传闻。
  
  - 一些参与者引用了来自各大实验室的 **新闻来源**，指出近期训练实验的结果令人失望，从而对 Twitter 的断言提出了质疑。
- **LLM 的局限性影响 AGI 愿景**：讨论聚焦于当前 **LLM 架构** 的 **能力约束**，暗示潜在的 **边际收益递减** 可能会阻碍 **AGI 级模型** 的开发。
  
  - 参与者强调 LLM 处理复杂任务（如 **Diffie-Hellman 密钥交换**）的必要性，并对模型在内部维护 **私钥（private keys）** 的能力及整体 **隐私性** 表示担忧。
- **Mixture-of-Experts 增强 Pythia 模型**：一项讨论探索了在 **Pythia 模型套件** 中实现 **Mixture-of-Experts (MoE)** 框架，并在复制现有训练配置与更新 **超参数**（如 **SwiGLU**）之间进行了权衡。
  
  - 成员们对比了 **OLMo** 和 **OLMOE** 模型，注意到 **数据排序** 和 **规模一致性** 方面的差异，这可能会影响 MoE 集成的效果。
- **定义开源 AI：数据 vs 代码**：讨论围绕基于 **IBM 的定义** 对 AI 系统进行 **开源 AI** 分类展开，特别是争论这些要求是适用于 **数据**、**代码** 还是两者兼有。
  
  - 链接到 [**Open Source AI Definition 1.0**](https://opensource.org/ai/open-source-ai-definition)，成员们强调了 **自主性**、**透明度** 和 **协作** 的重要性，同时通过描述性数据披露来规避 **法律风险**。
- **Transformer Heads 在模型中识别反义词**：研究发现某些 **Transformer Heads** 能够计算 **反义词**，分析展示了如 '**hot**' - '**cold**' 的示例，并利用了 **OV 电路** 和 **消融研究（ablation studies）**。
  
  - 在各种模型中存在的可解释特征值证实了这些反义词 Head 在增强 **语言模型（language model）** 理解能力方面的功能。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **NVIDIA NV-Embed-v2 登顶 Embedding 基准测试**：NVIDIA 发布了 [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)，这是一款领先的 Embedding 模型，在 Massive Text Embedding Benchmark 上获得了 **72.31** 分。
  
  - 该模型结合了先进技术，如增强的潜在向量保留和独特的 **难负样本挖掘（hard-negative mining）** 方法。
- **Llama 3.2 训练中 RLHF 与 SFT 的探讨**：关于 **来自人类反馈的强化学习 (RLHF)** 与 **有监督微调 (SFT)** 的讨论集中在训练 Llama 3.2 的资源需求上。
  
  - 成员们指出，虽然 RLHF 需要更多 **显存 (VRAM)**，但对于资源有限的用户来说，SFT 提供了一个可行的替代方案。
- **紧凑型模型实现 SOTA 图像识别**：[adjectiveallison](https://arxiv.org/abs/2411.04732) 介绍了一种图像识别模型，以 **缩小 29 倍** 的体积实现了 SOTA 性能。
  
  - 该模型证明了紧凑型架构可以保持高精度，从而可能减少计算资源消耗。
- **AI 驱动的翻译工具增强文化细微差别**：一款 **AI 驱动的翻译工具** 利用 **Agent 工作流**，通过强调 **文化细微差别** 和适应性，超越了传统的机器翻译。
  
  - 它考虑了地区方言、正式程度、语气和性别差异，确保翻译更加准确且具备上下文感知能力。
- **优化 LLM 训练中的数据集混合**：一位成员请求关于在 **LLM 训练** 的不同阶段有效 **混合和匹配数据集** 的指导。
  
  - 重点在于采用最佳实践来优化训练过程，同时不损害模型性能。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **MistralNemo 和 Celeste 支持已停止**：**MistralNemo StarCannon** 和 **Celeste** 已正式弃用，因为唯一的供应商停止了支持，这影响了所有依赖这些模型的项目。
  
  - 这一移除操作要求用户寻找替代模型或调整现有的工作流以适应变化。
- **Perplexity 在 Beta 版中增加引用溯源 (Grounding Citations)**：**Perplexity** 推出了引用溯源的 Beta 功能，允许在补全响应（completion responses）中包含 URL，以增强内容的可靠性。
  
  - 用户现在可以直接引用来源，提高了生成信息的可信度。
- **Gemini API 现已开放**：**Gemini** 现在可以通过 API 访问，其先进的能力在工程社区中引起了热烈讨论。
  
  - 然而，一些用户报告称未看到更改，表明可能存在发布不一致的情况。
- **OpenRouter 实施速率限制**：**OpenRouter** 为免费模型引入了**每天 200 次请求**的限制，详见其 [官方文档](https://openrouter.ai/docs/limits)。
  
  - 由于可扩展性降低，这一限制给在生产环境中部署免费模型带来了挑战。
- **Hermes 405B 保持效率偏好**：尽管成本较高，**Hermes 405B** 凭借其无与伦比的性能效率，仍然是许多用户的首选模型。
  
  - 像 **Fry69_dev** 这样的用户强调了其卓越的效率，尽管存在利润率方面的顾虑，它依然保持着顶级选择的地位。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 的 Nvidia GPU 配置困扰**：一位用户报告了在配置 **Stable Diffusion** 以使用其独立 **Nvidia GPU** 而非集成显卡时遇到的问题。另一位成员引用了频道中置顶的 [WebUI 安装指南](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides) 以寻求支持。
  
  - 社区强调了遵循设置指南的重要性，以确保 **Stable Diffusion** 工作流能够充分利用 GPU。
- **WebUI 对决：ComfyUI vs. SwarmUI**：一位成员对比了 **ComfyUI** 与 **SwarmUI** 的复杂性，强调 **SwarmUI** 简化了配置过程。建议使用 **SwarmUI** 以获得更直接的安装体验和一致的性能。
  
  - 讨论集中在易用性上，多位用户一致认为 **SwarmUI** 提供了一种技术门槛更低的方法，且不牺牲功能性。
- **寻找最新的图像融合 (Image Blending) 论文**：一位用户寻求帮助寻找最近一篇关于图像融合的研究论文，提到了一位 Google 作者但未能找到。另一位成员建议在 [arXiv](https://arxiv.org/) 中搜索图像融合相关的 Google 论文。
  
  - 社区强调了访问 **arXiv** 预印本的价值，以便及时了解图像融合技术的最新进展。
- **视频超分辨率 (Video Upscaling) 的逐帧修复**：一位成员分享了他们通过每 0.5 秒提取一帧来修复不准确之处并进行视频超分辨率的方法。讨论还涉及使用 **Flux Schnell** 和其他工具来实现快速推理（inference）结果。
  
  - 参与者讨论了各种增强视频质量的技术和工具，强调了在超分辨率过程中平衡速度与准确性的重要性。
- **使用 Diffusers 掌握低去噪局部重绘 (Low Denoise Inpainting)**：一位用户询问如何对特定图像区域进行**低去噪局部重绘 (low denoise inpainting)** 或 **img2img** 处理。建议包括利用 **Diffusers** 进行快速的 **img2img** 工作流，通过极少的步骤来精修图像。
  
  - 社区推荐将 **Diffusers** 作为针对性重绘任务的有效工具，并强调了其在实现高质量图像精修方面的效率。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Scaling Laws 理论面临审查**：成员们对 **Scaling Laws** 理论的有效性提出了质疑，该理论认为增加**计算能力和数据**将增强 AI 能力。
  
  - 一位成员表示宽慰，称降低**交叉熵损失（cross-entropy loss）**是提升 AI 能力的充分条件。
- **GPT-3 Scaling Hypothesis 验证了规模化收益**：引用 [The Scaling Hypothesis](https://gwern.net/scaling-hypothesis)，一位成员指出，随着问题复杂度的增加，神经网络会产生泛化并展现出新的能力。
  
  - 他们强调，于 2020 年 5 月发布的 **GPT-3** 持续证明了规模化的益处，这与收益递减的预测相反。
- **60 亿美元融资将估值推高至 500 亿美元**：一位成员分享了一则 [推文](https://x.com/AndrewCurran_/status/1857437923525931297)，指出下周结束的一轮融资将筹集 **60 亿美元**，主要来自**中东主权基金**，估值为 **500 亿美元**。
  
  - 据报道，这笔资金将直接流向 **Jensen**，助力科技领域的后续发展。
- **历史上的对齐问题担忧在 2024 年案件中再次浮现**：成员们分享了 [TechEmails](https://x.com/TechEmails/status/1857459526267449790)，讨论了 **2017 年**涉及 **Elon Musk、Sam Altman 和 Ilya Sutskever** 关于对齐（misalignment）问题的通信。
  
  - 这些文件与正在进行的 **Elon Musk, et al. v. Samuel Altman, et al. (2024)** 案件相关，突显了对齐担忧的历史背景。
- **Apple Silicon vs NVIDIA GPU：LLM 性价比大对决**：一篇 [文章](https://blog.hjc.im/apple-uma-for-llms-problems.html) 讨论了 **Apple Silicon** 与 **NVIDIA** GPU 在运行 LLM 方面的竞争，强调了 Apple 平台的折中方案。
  
  - 虽然 Apple 的新产品提供了更高的内存容量，但 **NVIDIA 解决方案** 仍然更具性价比。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FSDP 与 torch.compile 协同工作**：一位用户演示了在 **FSDP** 中包装 [torch.compile](https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L40) 且未遇到问题，表明两者可以无缝集成。
  
  - 他们指出了这种方法的有效性，但提到尚未测试反向顺序，为进一步实验留下了空间。
- **NSYS 面临内存过载**：**nsys** 性能分析在崩溃前可能消耗高达 **60 GB 内存**，这引发了对其在大型分析任务中实用性的担忧。
  
  - 为了缓解这一问题，用户建议使用 `nsys profile -c cudaProfilerApi -t nvtx,cuda` 等标志位来优化 nsys 使用，以减少日志记录开销。
- **ZLUDA 将 CUDA 扩展至 AMD 和 Intel GPU**：在一段 [YouTube 视频](https://www.youtube.com/watch?v=ze25Sie2gVQ) 中，Andrzej Janik 展示了 **ZLUDA** 如何在 **AMD** 和 **Intel GPU** 上实现 **CUDA** 功能，这可能会改变 GPU 计算格局。
  
  - 社区成员对这一突破表示赞赏，对在 NVIDIA 硬件之外普及 GPU 计算能力感到兴奋。
- **React Native 通过 ExecuTorch 拥抱 LLM**：**Software Mansion** 发布了一个新的 [React Native 库](https://github.com/software-mansion/react-native-executorch)，利用 **ExecuTorch** 进行后端 **LLM** 处理，简化了移动平台上的模型部署。
  
  - 用户称赞该库易于使用，强调了在 iOS 模拟器上简单的安装和模型启动流程。
- **Bitnet 1.58 A4 加速 LLM 推理**：采用 **Bitnet 1.58 A4** 结合微软的 T-MAC 操作，在 7B 模型上实现了 **10 tokens/s** 的速度，提供了一种无需过度依赖 GPU 的快速推理解决方案。
  
  - 虽然已有将模型转换为 **Bitnet** 的资源，但可能需要进行一些训练后修改以优化性能。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord 频道

- **听众要求 Top Shelf Podcast 提供更多内容**：听众正在推动扩展 **Top Shelf Podcast**，增加更多书籍摘要，特别是要求制作关于 Adam Grant 的 *Think Again* 以及 *The Body Keeps Score* 见解的节目。他们链接了 [Top Shelf Spotify 节目](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF)以支持他们的建议。
  
  - 一位用户鼓励社区成员分享更多书籍推荐，以丰富播客的内容。
- **对 AI 控制暴力的担忧**：一位用户分享了 [“AI 对暴力的垄断” YouTube 视频](https://youtu.be/LgU6R26csf0)，对人工超级智能（artificial superintelligence）管理暴力行为的影响提出了警示。这种对暴力的**垄断**可能会导致重大的伦理和安全问题。
  
  - 该视频探讨了授予 AI 实体控制暴力决策权的潜在后果，引发了关于严格治理措施必要性的讨论。
- **NotebookLM 面临运行问题**：多名成员报告了 **NotebookLM 的技术问题**，例如功能故障和某些功能的访问受限。他们在等待开发团队解决问题时表达了沮丧。
  
  - 用户分享了临时变通方法，并强调需要及时修复以恢复该工具的全部功能。
- **为不同受众定制音频摘要**：一位成员展示了他们使用 NotebookLM 创建**定制化音频摘要**的方法，专门为社工和研究生调整内容。这展示了 NotebookLM 根据受众需求修改内容的能力。
  
  - 定制过程涉及改变演示风格，以更好地适应不同专业群体的信息需求。
- **讨论文档上传限制**：参与者讨论了 NotebookLM 内部的**文档上传限制**，并建议对文档进行分组以遵守上传限制。关于是否可以上传超过 50 份文档的问题也被提出。
  
  - 讨论强调了改进上传能力的必要性，以更好地容纳大量的文档收藏。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 频道

- **Orca 合成数据进展**：关于 [Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/) 的研究展示了其利用**合成数据（synthetic data）**进行小语言模型后训练的能力，使其性能能够媲美更大型的模型。
  
  - [Orca-AgentInstruct](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/) 引入了 **Agentic flows** 来生成多样化、高质量的合成数据集，增强了数据生成过程的效率。
- **Liger Kernel 和 Cut Cross-Entropy 的改进**：**Liger Kernel** 的增强带来了**速度**和**内存效率**的提升，详情见 [GitHub pull request](https://github.com/linkedin/Liger-Kernel/pull/362)。
  
  - 提议的 **Cut Cross-Entropy (CCE)** 方法将 Gemma 2 模型的内存占用从 **24 GB 降低到 1 MB**，使得训练速度比当前的 Liger 设置快约 **3 倍**，详见[此处](https://github.com/apple/ml-cross-entropy)。
- **Axolotl 答疑时间（Office Hours）和反馈环节**：**Axolotl** 将于 **12 月 5 日下午 1 点（EST）**在 Discord 举办首场 **Office Hours** 活动，允许成员提问并分享反馈。
  
  - 鼓励成员将他们的想法和建议带到 **Axolotl 反馈环节**，团队渴望参与并改进平台。更多详情可在 [Discord 群聊](https://discordapp.com/channels/1104757954588196865/1268285745555308649)中查看。
- **Qwen/Qwen2 预训练和 Phorm Bot 问题**：一位成员正在寻求关于使用 **qlora** 和原始文本 jsonl 数据集预训练 **Qwen/Qwen2** 模型的指导，随后在安装 **Axolotl docker** 后使用 instruct 数据集进行微调。
  
  - 报告的 **Phorm bot** 问题包括其无法响应基本查询，表明社区内可能存在技术故障。
- **Meta 邀请参加 Llama 活动**：一位成员收到了来自 **Meta** 的意外邀请，参加在其总部举行的关于开源计划和 **Llama** 的**为期两天的活动**，引发了人们对潜在**新模型发布**的好奇。
  
  - 社区成员正在推测活动的重点，特别是考虑到在没有发言角色的情况下收到邀请的特殊性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4o Token 计费**：一次讨论透露，**GPT-4o** 在高分辨率模式下处理每个 `512x512` 切片会产生 **170 tokens** 的费用，实际上将一张图片估值为约 **227 个单词**。深入分析请参考 [OranLooney.com](https://www.oranlooney.com/post/gpt-cnn/)。
  
  - 参与者质疑了特定 token 定价背后的逻辑，将其类比为编程中的 *magic numbers*（魔术数字），并讨论了其对使用成本的影响。
- **利用 Few-Shot Examples 增强 RAG Prompts**：用户正在探索将文档中的 **few-shot examples** 集成到 **RAG prompts** 中，是否能提高其 **QA agent platform** 的回答质量。
  
  - 社区强调了在该领域进行深入研究的必要性，旨在优化 prompt 策略以增强响应准确性。
- **AI 在 24 点游戏中的表现**：**3.5 AI** 模型已展示出在 **Game of 24**（24 点游戏）中偶尔获胜的能力，展示了 AI 游戏能力的显著进步。
  
  - 这一进步凸显了 AI 算法的持续增强，用户对未来的性能里程碑表示乐观。
- **内容标记政策**：成员们讨论了 **content flags**（内容标记）主要针对模型输出并有助于训练改进，而非暗示用户违规。
  
  - 成员对内容标记的增加表示担忧，特别是在恐怖视频游戏等语境下，这表明监控措施有所加强。
- **高级照片筛选技术**：一位成员提议创建一个编号拼贴图，作为从数百张照片中筛选最佳照片的高效方法，旨在简化选择过程。
  
  - 尽管有人怀疑拼贴方法可能显得“零散”，但其有效性得到了认可，尤其是在按顺序处理任务时。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenAI 发布 'OPERATOR' AI Agent**：在一段名为“OpenAI Reveals OPERATOR The Ultimate AI Agent Smarter Than Any Chatbot”的 [YouTube 视频](https://www.youtube.com/watch?v=YRn9xzTBt20) 中，OpenAI 宣布了他们即将推出的 AI agent，预计很快将面向更广泛的受众。
  
  - 视频强调 *it's coming to the masses!*（它正走向大众！），暗示该 AI agent 的部署规模将大幅扩大。
- **Beta 版应用超越控制台集成**：成员们确认 **desktop beta app** 的性能优于 console integration（控制台集成），这归功于增强的基础设施支持。
  
  - 强调了桌面应用比 open-source repository（开源仓库）拥有更广泛的幕后支持，确保了更好的 Interpreter 体验。
- **Azure AI Search 技术详情**：一段名为“How Azure AI Search powers RAG in ChatGPT and global scale apps”的 [YouTube 视频](https://youtu.be/NVp9jiMDdXc?feature=shared) 概述了 Azure AI Search 中使用的数据转换和质量恢复方法。
  
  - 讨论引发了对专利申请、资源分配以及在大规模应用中高效数据删除流程必要性的关注。
- **概率计算提升 GPU 性能**：一段 [YouTube 视频](https://www.youtube.com/watch?v=hJUHrrihzOQ) 报道称，**probabilistic computing**（概率计算）与顶尖的 NVIDIA GPU 相比，实现了 **1 亿倍的能效提升**。
  
  - 演讲者表示：“在这段视频中，我讨论了概率计算，据报道，与最好的 NVIDIA GPU 相比，它能实现 1 亿倍的能效提升。”
- **ChatGPT 桌面端增强**：**ChatGPT desktop** 的最新更新引入了用户友好的增强功能，标志着对大众用户体验的显著改进。
  
  - 用户渴望体验能优化平台交互的功能，强调了 **desktop** 增强的可用性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinybox pro 开启预订**：**tinybox pro** 现已在 [tinygrad 官网](https://tinygrad.org/#tinybox) 开启预订，售价 **40,000 美元**，配备 8 张 RTX 4090，提供 **1.36 PetaFLOPS** 的 FP16 计算能力。
  
  - 市场定位为单张 Nvidia H100 GPU 的高性价比替代方案，旨在为 AI 工程师提供强大的计算能力。
- **关于 int64 索引悬赏的说明**：一位成员询问了 **int64 索引** 悬赏的要求，特别是关于 tensor.py 中 `__getitem__` 等函数的修改。
  
  - 另一位成员提到了 [PR #7601](https://github.com/tinygrad/tinygrad/pull/7601/files#diff-00bd44b667ec90ae1d3e984e699bc6b498c84ca1b1bd15a025437ded227457bf)，该 PR 解决了该悬赏问题，但尚待受理。
- **CLOUD 设备 buffer 传输功能的增强**：重点介绍了 CLOUD 设备 **buffer 传输函数** 的一个 Pull Request，旨在提高设备间的互操作性。
  
  - 讨论指出目标 buffer 大小检查可能存在歧义，强调了实现中清晰度的必要性。
- **在 GPU 之间原生传输 tensor**：明确了可以使用 `.to` 函数在同一设备的不同 **GPU** 之间传输 tensor。
  
  - 该指南协助用户在项目中高效管理 tensor 传输。
- **寻求对 tinygrad 贡献的反馈**：一位贡献者分享了他们为 **tinygrad** 做出贡献的初步尝试，并寻求全面的反馈。
  
  - 他们提到了专注于数据传输改进的 [PR #7709](https://github.com/tinygrad/tinygrad/pull/7709)。

 

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **在社区电话会议中学习 GenAI 应用构建**：参加我们即将举行的 [社区电话会议](https://twitter.com/llama_index/status/1857500067357405398)，探索如何从非结构化数据创建 **知识图谱 (knowledge graphs)** 以及高级检索方法。
  
  - 参与者将深入研究将 **数据转换** 为可查询格式的技术。
- **Python 文档通过 RAG 系统获得功能提升**：Python 文档通过一个新的 **“Ask AI”** 小组件得到了增强，该组件为代码查询启动了一个精确的 **RAG 系统** [点击查看！](https://twitter.com/llama_index/status/1857536223566508061)。
  
  - 用户可以直接获得针对其问题的 **准确、最新的代码** 回复。
- **CondensePlusContext 的动态上下文检索**：**CondensePlusContext** 压缩输入并为每条用户消息检索上下文，增强了向系统提示词 (system prompt) 中插入 **动态上下文** 的能力。
  
  - 成员们因其在一致管理上下文检索方面的高效性而更青睐它。
- **condenseQuestionChatEngine 面临的挑战**：一位成员报告称，当用户突然切换话题时，**condenseQuestionChatEngine** 可能会生成不连贯的独立问题。
  
  - 建议包括自定义压缩提示词，以有效处理突然的话题转变。
- **在 CondensePlusContext 中实现自定义检索器**：成员们同意使用带有自定义检索器 (custom retriever) 的 **CondensePlusContextChatEngine** 以符合特定需求。
  
  - 他们建议采用自定义检索器和节点后处理器 (node postprocessors) 以优化性能。

 

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Agentic Chunking 研究发布**：一种针对 **RAG 的 Agentic Chunking** 新方法实现了**少于 1 秒**的推理时间，证明了在 GPU 上的高效性且具备成本效益。该研究的完整细节和社区建设可以在其 [Discord 频道](https://discord.com/invite)找到。
  
  - 这一进展展示了检索增强生成过程中的显著性能提升，增强了系统效率。
- **LlamaChunk 简化文本处理**：**LlamaChunk** 引入了一种由 LLM 驱动的技术，通过仅对文档进行单次 **LLM 推理**来优化递归字符文本分割。该方法消除了标准分块算法中通常使用的脆弱正则表达式模式。
  
  - 团队鼓励对 **LlamaChunk** 代码库做出贡献，该代码库已在 [GitHub](https://github.com/ZeroEntropy-AI/llama-chunk) 上公开。
- **RAG Pipeline 的增强**：**RAG Pipeline** 正在通过 **Agentic Chunking** 进行优化，旨在简化检索增强生成过程。这种集成专注于减少推理时间并提高整体 Pipeline 效率。
  
  - 这些更新利用 GPU 效率在提高性能指标的同时保持成本效益。
- **在 Python 中使用 Playwright 上传文件**：一位用户分享了在 **Playwright Python** 中使用 `set_input_files` 方法上传文本文件，随后查询上传内容的方法。这种方法简化了涉及文件交互的自动化测试工作流。
  
  - 然而，该用户指出，在请求 *“你能总结文件中的文本吗？@file2upload.txt”* 时，这种方法感觉有些奇怪。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **关于公开链接的版权困惑**：一位成员断言，“公开链接的公开索引在任何情况下都不构成版权侵权”，引发了关于使用公开链接合法性的困惑。
  
  - 这场辩论突显了社区对与公开链接索引相关的**版权法**的不确定性。
- **对 Discord 礼仪的赞赏**：一位成员用简单的 *ty* 表达了感谢，展示了对所获帮助的感激。
  
  - 这种交流表明了社区内持续的协作支持和对 **Discord 礼仪**的遵守。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **ChatGPT for macOS 与桌面应用集成**：**ChatGPT for macOS** 在面向 Plus 和 Team 用户的当前 Beta 版本中，现已支持与 [VS Code](https://code.visualstudio.com/)、[Xcode](https://developer.apple.com/xcode/)、[Terminal](https://www.apple.com/terminal/) 和 [iTerm2](https://iterm2.com/) 等桌面应用程序集成。
  
  - 这种集成使 **ChatGPT** 能够通过与开发环境直接交互来增强编码辅助，从而可能改变项目工作流。[OpenAI Developers 的推文](https://x.com/OpenAIDevs/status/1857129790312272179)最近宣布了这一功能。
- **dspy GPTs 功能**：有强烈的意向扩展 **dspy GPTs** 的功能，旨在显著增强开发工作流。
  
  - 社区成员讨论了扩展 **dspy GPTs** 集成的潜在好处，强调了对其项目流程的积极影响。
- **用于违规行为的 LLM 文档生成**：一位用户正在开发一个 LLM 应用程序，用于生成全面的法律文件，为因违规（目前专注于**酒精摄入**案例）而面临吊销驾照的驾驶员辩护。
  
  - 他们正在寻求一种方法来创建一个优化的 Prompt，该 Prompt 可以处理各种类型的违规行为，而无需单独定制 Prompt。
- **DSPy 语言兼容性**：一位用户询问了 **DSPy** 对于需要非英语语言的应用的语言支持能力。
  
  - 引用了一个公开的 [GitHub issue](https://github.com/stanfordnlp/dspy/issues/1803)，该 issue 涉及在 DSPy 中添加本地化功能的请求。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **ABI 研究揭示优化挑战**：成员们分享了[新的 ABI 研究论文](https://doi.org/10.1145/3689755)和一份 [PDF](https://www.andrewwagner.io/assets/papers/all-bin-tog.pdf)，强调了 **low-level ABIs** 在促进跨模块优化方面面临的挑战。
  
  - *一位成员指出*，为了获得最大执行速度，通常更倾向于用同一种语言编写所有内容。
- **ALLVM 项目面临运行障碍**：讨论显示，**ALLVM 项目**可能由于编译和链接软件（特别是在浏览器中）时设备内存不足而受阻。[ALLVM 研究项目](https://publish.illinois.edu/allvm-project/)。
  
  - *另一位成员建议* **Mojo** 可以以创新方式利用 ALLVM 进行 **C/C++ bindings**。
- **成员提倡跨语言 LTO**：一位成员强调了 **cross-language LTO** 对现有 C/C++ 软件生态系统的重要性，以避免重写。
  
  - 讨论公认有效的链接可以显著提高遗留系统的性能和可维护性。
- **Mojo 探索 ABI 优化**：成员们探讨了 **Mojo** 在定义 ABI 方面的潜力，该 ABI 通过利用 **AVX-512** 大小的结构并最大化寄存器信息来优化数据传输。
  
  - 该 ABI 框架有望增强各种软件组件之间的互操作性和效率。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **关于 AI 工具的 Intel AMA**：参加与 **Intel** 合作的独家 AMA 会议：[使用 Intel 构建：Tiber AI Cloud 和 Intel Liftoff](https://lu.ma/agents-hackathon-intel)，定于 **太平洋时间 11/21 下午 3 点**，提供关于高级 AI 开发工具的见解。
  
  - 本次活动提供了一个与 Intel 专家互动并获得使用其最新资源优化 AI 项目专业知识的独特机会。
- **Intel Tiber AI Cloud**：Intel 将推出 **Tiber AI Cloud**，这是一个旨在通过提高计算能力和效率来增强黑客松项目的平台。
  
  - 参与者可以探索如何利用该平台来提升性能并简化其 AI 开发工作流。
- **Intel Liftoff 计划**：会议将涵盖 **Intel Liftoff Program**，该计划为初创公司提供技术资源和导师指导。
  
  - 了解旨在帮助年轻公司在 AI 行业内扩大规模并取得成功的全面福利。
- **测验反馈延迟**：一位成员对在尝试赶进度时未收到 **quizzes 5 和 6** 的电子邮件反馈表示担忧。
  
  - 另一位成员建议验证本地设置，并建议 *重新提交* 测验以解决该问题。
- **课程截止日期提醒**：发布了一项紧急提醒，参与者仍有 **资格**，但需要 *快速赶上*，因为每个测验都与课程内容挂钩。
  
  - 最终提交日期定为 **12 月 12 日**，强调了及时完成的必要性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v0.4.0 发布**：**Torchtune v0.4.0** 已正式发布，引入了 **Activation Offloading**、**Qwen2.5 支持**以及增强的**多模态训练 (Multimodal Training)** 等功能。完整的发布说明请参阅[此处](https://github.com/pytorch/torchtune/releases/tag/v0.4.0)。
  
  - 这些更新旨在显著提升用户体验和模型训练效率。
- **Activation Offloading 功能**：**Activation Offloading** 现已在 **Torchtune v0.4.0** 中实现，在 finetuning 和 LoRA recipes 过程中，可将所有文本模型的内存需求降低 **20%**。
  
  - 这一增强功能优化了性能，支持更高效的模型训练工作流。
- **Qwen2.5 模型支持**：**Torchtune** 已添加 **Qwen2.5 Builders** 支持，与 Qwen 模型家族的最新更新保持一致。更多详情请参阅 [Qwen2.5 博客](https://qwenlm.github.io/blog/qwen2.5/)。
  
  - 这一集成促进了在 Torchtune 训练环境中使用 Qwen2.5 模型。
- **多模态训练增强**：**Torchtune** 中的**多模态训练 (multimodal training)** 功能得到了增强，支持 **Llama3.2V 90B** 和 **QLoRA** 分布式训练。
  
  - 这些增强功能使用户能够处理更大的数据集和更复杂的模型，扩展了训练能力。
- **用于合成数据的 Orca-AgentInstruct**：来自 Microsoft Research 的新项目 [**Orca-AgentInstruct**](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/) 提供了一种 Agent 方案，用于大规模生成多样化、高质量的**合成数据集 (synthetic datasets)**。
  
  - 该方法旨在通过利用有效的合成数据生成来进行 post-training 和 fine-tuning，从而提升小语言模型的性能。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **本地 LLM 工作坊定于周二举行**：**构建你自己的本地 LLM 工作坊**定于**周二**举行，主题为 [Building your own local LLM's: Train, Tune, Eval, RAG all in your Local Env.](https://discord.com/events/1089876418936180786/1300842793945530378)，旨在指导成员了解本地 LLM 设置的复杂性。
  
  - 鼓励参与者 RSVP 以增强其本地环境能力。
- **SQLite-Vec 添加元数据过滤**：**SQLite-Vec 现在支持元数据过滤 (Metadata Filtering)** 已于**周三**通过[此活动](https://discord.com/events/1089876418936180786/1300483739872399411)宣布，强调了具有实际应用价值的增强功能。
  
  - 此更新允许通过元数据利用来改进数据处理。
- **周四举行自主 AI Agent 讨论**：参加**周四**关于**探索自主 AI Agent (Autonomous AI Agents)** 的讨论，主题为 [Autonomous AI Agents with Refact.AI](https://discord.com/events/1089876418936180786/1300459081181429810)，重点关注 AI 自动化进展。
  
  - 该活动承诺提供有关 AI Agent 功能和未来轨迹的见解。
- **落地页开发寻求协助**：一位成员正在为其项目寻求开发落地页 (landing page) 的帮助，并计划在 Mozilla AI 舞台上进行现场演示。
  
  - 有兴趣的成员应[在此帖子中联系](https://discord.com/channels/1089876418936180786/1307044141657751592)以提供协作营销支持。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1306723488937021491) (83 messages🔥🔥):

> - `Qwen2.5-Math-72B-Instruct 问题`
> - `LM Studio 本地服务器设置`
> - `LM Studio 中的 Function calling 测试版`
> - `SSD 速度对比`
> - `在 LLM 任务中使用多个应用`

- **Qwen2.5 无法正确渲染 LaTeX**：用户对 **Qwen 2.5** 表达了不满，因为它无法正确显示被 $ 符号包裹的 LaTeX，导致输出内容毫无意义。
  
  - 有建议称，创建一个带有明确指令的 System Prompt 可能会改善结果，但目前解决该问题的尝试均未见效。
- **将 LM Studio 作为 LLM 访问服务器**：一名用户尝试将 **LM Studio** 设置为服务器以调用简单的 LLM，但遇到了与服务器可访问性相关的连接错误。
  
  - 另一名成员提醒他们，LM Studio 的功能是作为 API 而非 Web UI 运行，这可能有助于理清设置思路。
- **对 Function calling 测试版的兴奋**：用户分享了对 LM Studio 中 **Function calling 测试版**功能的狂热，并征求个人使用经验和反馈。
  
  - 一些用户强调文档非常直观，但也有人表示困惑，期待更多功能。
- **关于 SSD 速度和 RAID 配置的讨论**：聊天中讨论了 SSD 的速度，特别是 **SABRENT Rocket 5** 和 **Crucial T705**，同时强调了 PCIe 通道和 RAID 设置的限制。
  
  - 用户指出，虽然 SSD 有理想的最大速度，但实际性能可能会根据具体工作负载和 RAID 配置产生显著差异。
- **使用多个应用程序进行 LLM 实验**：一位用户分享了针对不同任务使用多个 LLM 应用的方法，强调没有单一的应用能在每个用例中都表现出色。
  
  - 他们强调，虽然 **LM Studio** 能满足大部分需求，但针对特定目的（如实验各种模型）拥有多种选择是有益的。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1306714534291046442) (223 messages🔥🔥):

> - `GPU 对比`
> - `AMD Radeon PRO W7800`
> - `用于 AI 的 Apple 硬件`
> - `Qualcomm X Elite v2`
> - `LLM 训练成本`

- **在 Mac Mini M4 和配备 3090 的 PC 之间权衡**：用户讨论了使用 Mac Mini M4 与配备 3090 的 PC 进行 AI 相关任务的优缺点，强调了升级潜力和能效等权衡。
  
  - 大家达成共识，虽然 Mac 用户友好，但在训练方面的表现可能不如配备专用 NVIDIA GPU 的机器。
- **技嘉发布 AMD Radeon PRO W7800 AI TOP 48G**：技嘉最近发布了 AMD Radeon PRO W7800 AI TOP 48G，配备 48 GB GDDR6 显存，旨在面向 AI 和工作站专业人士。
  
  - 尽管规格令人印象深刻，但与 NVIDIA 的 CUDA 相比，人们对 AMD 驱动程序的可靠性和软件兼容性仍有顾虑。
- **在不同硬件上进行 LLM 训练的考量**：参与者指出，24 GB 的 VRAM 对于训练大型模型通常是不够的，这引发了关于潜在升级路径的讨论。
  
  - 在 Mac Mini 上进行训练是可能的，但可能会产生更高的电费，从而引发了关于租用 GPU 进行训练任务的讨论。
- **Qualcomm X Elite v2 和 Windows on ARM**：讨论了即将推出的 Qualcomm X Elite v2 及其对开发的影响，尽管有人提出了对 Windows on ARM 兼容性的担忧。
  
  - 参与者一致认为，虽然正在取得进展，但与 Mac 相比，基于 ARM 的 Windows 在软件兼容性方面仍然匮乏。
- **新 AI 模型的性能和可用性**：用户对各种 AI 模型的性能表现出兴趣，特别是提到了 Qwen 2.5 和 Nemotron 70B 等 LLM 之间的对比。
  
  - 人们担心基于 RAM 和 GPU 性能的模型整体可用性，这会影响推理速度。

**提到的链接**：

- [Extropic 从未来组装自己](https://www.extropic.ai/accelerate)：Extropic 宣布其 1410 万美元的种子轮融资 // Guillaume Verdon // 2023 年 12 月 4 日
- [技嘉发布 AMD Radeon PRO W7800 AI TOP 48G 显卡](https://www.techpowerup.com/328837/gigabyte-launches-amd-radeon-pro-w7800-ai-top-48g-graphics-card)：全球领先的高端游戏硬件制造商技嘉科技今天推出了尖端的 GIGABYTE AMD Radeon PRO W7800 AI TOP 48G。技嘉迈出了重要的一步...

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1306712090240745534) (238 条消息🔥🔥):

> - `Perplexity API 与功能`
> - `移动端 App 问题`
> - `中文语言与文化讨论`
> - `AI 写作中的查重检查`
> - `Perplexity 的 Curator 申请`

- **关于 Perplexity API 和实时浏览的疑虑**：用户对 Perplexity 目前无法访问链接或实时浏览网页表示困惑，并建议尝试不同的提问方式。
  
  - 一些人指出，正确地引导（prompting）模型可能会产生更好的结果，特别是在总结内容方面。
- **移动端 App 功能**：几位用户报告了最近在使用 Perplexity 移动端 App 时遇到的困难，其中一人尽管能成功下载其他 App，但无法安装此 App。
  
  - 提供了从 App Store 更新 App 的建议，同时也有人确认其他人并未遇到问题。
- **关于中文语言的讨论**：几位成员讨论了聊天中中文的使用频率，分享了他们对语言使用、翻译工具和文化认知的看法。
  
  - 用户注意到了简体中文和繁体中文之间的差异，以及使用某些应用程序如何促进交流。
- **AI 写作中的查重担忧**：一位用户询问 Perplexity 在生成文章时是否进行真实的查重检查，引发了对 AI 生成内容原创性的担忧。
  
  - 社区成员建议使用特定的 Prompt 来指示 AI 避免剽窃，同时确认 Perplexity 确实包含资源引用。
- **Curator 角色申请**：一位用户询问了 Curator 申请的状态并被鼓励申请，同时还提供了提高入选机会的建议。
  
  - 另一位用户提到由于忙于学业，但计划将相关项目作为写作样本包含在申请中。

**提及的链接**：

- [Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1857109958107578509?s=61)：自然地
- [Vencord](https://vencord.dev/)：未找到描述
- [Ryan Putnam (@RypeArts) 的推文](https://x.com/rypearts/status/1857512981699113338?s=61)：周五氛围 ✨
- [未找到标题](https://pplx.ai,)：未找到描述
- [未找到标题](https://docs.perplexity.ai)：未找到描述
- [lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1857110672565494098)：来自 Chatbot Arena 的重磅消息🔥 @GoogleDeepMind 最新的 Gemini (Exp 1114)，在过去一周经过 6000 多个社区投票测试，现在以 40 多分的惊人涨幅并列总榜第一 —— ma...
- [Perplexity CEO Aravind Srinivas 谈迈向 AI 策展网络的浪潮 | TechCrunch Disrupt 2024](https://youtu.be/d3boSs5pO9w?si=UIq5UFi_0czRM5HO&t=295)：Perplexity 的 AI 驱动搜索引擎可能是与网络和通用知识交互的下一个阶段 —— 也可能不是。但该公司无疑正在崛起...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1306737723918389298) (10 条消息🔥):

> - `DeepMind AlphaFold Code Release` (DeepMind AlphaFold 代码发布)
> - `Chegg's Decline Due to ChatGPT` (Chegg 因 ChatGPT 而衰落)
> - `Google Gemini App Launch` (Google Gemini App 发布)
> - `Microsoft Quantum Logic` (Microsoft 量子逻辑)
> - `Best Work Mouse` (最佳办公鼠标)

- **DeepMind 发布 AlphaFold 代码**：有消息称 **DeepMind** 已经发布了 **AlphaFold** 的代码，允许更广泛地访问其突破性的蛋白质折叠技术，链接见[此处](https://www.perplexity.ai/page/deepmind-releases-alphafold-co-rtUBaB6hQDiwRZcst1bXbg)。
  
  - 此次发布可能有助于 **biotechnology**（生物技术）和 **bioinformatics**（生物信息学）的进一步研究，推动该领域的全新进展。
- **Chegg 因 ChatGPT 导致的没落**：一项讨论强调了 **Chegg** 的价值如何经历了惊人的 **99% 跌幅**，这主要归因于与 **ChatGPT** 的竞争，详见[这篇文章](https://www.perplexity.ai/page/chegg-s-ai-driven-downfall-R3mSgjNyQT2tv6Vu.Wx4MQ)。
  
  - AI 对 Chegg 等传统教育平台的影响在社区中引发了广泛讨论。
- **Google 发布 Gemini App**：Google 已正式发布 **Gemini app**，为用户提供创新功能，正如这篇 [TechCrunch 文章](https://www.perplexity.ai/search/https-techcrunch-com-2024-11-1-k6p5L5QTTpOEnUwZXrY.Lw)所报道。
  
  - 该应用旨在与现有工具直接竞争，展示了融合 **AI** 与用户交互的新功能。
- **Microsoft 的量子逻辑运算**：另一个重大技术进展是 **Microsoft** 对**量子逻辑运算**（quantum logic operations）的探索，揭示了高级计算的未来可能性，详见[此处](https://www.perplexity.ai/page/microsoft-s-quantum-logic-oper-_nS_NeNBTR2qgG14T5QG0A)。
  
  - 这可能会重塑计算范式，并巩固 Microsoft 在量子技术领域的地位。
- **最佳办公鼠标**：参与者讨论了**最佳办公鼠标**的推荐，强调了舒适度和效率，可以在这篇[搜索文章](https://www.perplexity.ai/search/best-mouse-for-work-031fd.NlSeOAG_vHDd9pgg)中找到。
  
  - 社区热衷于能提高日常工作效率的人体工程学工具。

 

**提到的链接**：[YouTube](https://www.youtube.com/embed/Nx5AmHX-0dM)：未找到描述

 

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1306769345606717522) (2 条消息):

> - `PPLX API Performance` (PPLX API 性能)
> - `URL Injection Issue` (URL 注入问题)

- **PPLX API 在生产环境使用中表现参差不齐**：一位成员对 **PPLX API** 在生产环境中的可靠性表示担忧，特别是其提供准确结果的能力。
  
  - 他们提到，当 API 无法自信地找到某些内容时，它会注入未指定的**随机 URL**，从而导致**结果不准确**。
- **关于 API 可用性的讨论**：另一位参与者分享了他们的观察，认为 **PPLX API** 总体运行良好，但存在明显的缺点。
  
  - 这种持续的对话表明，虽然有些人看到了潜力，但其他人正在质疑其在生产环境中的适用性。

 

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1306712559537356901) (203 条消息🔥🔥):

> - `Gemini Experimental Model`
> - `Aider 配置问题`
> - `使用模型的成本`
> - `将 Aider 与 Hyperbolic 的 Qwen 2.5 Coder 集成`
> - `使用 Aider 生成提交信息`

- **Gemini Experimental Model 性能**：用户报告称新的 **gemini-exp-1114** 模型表现优于其他模型，尽管在测试期间存在细微的格式问题，但一些用户在编辑任务中实现了 **61% 的准确率**。
  
  - 与 **Gemini Pro 1.5** 的对比表明，它可能同样有效，但性能会因具体使用场景而异。
- **Aider 配置困扰**：用户对 **Aider** 无法识别某些文件或设置表示沮丧，特别是在使用 **openrouter/qwen/qwen-2.5-coder-32b-instruct** 配置时，导致了对 Token 限制和费用的困惑。
  
  - 已经实施了一项修复来解决这些问题，建议用户重新加载 main 分支以获得最佳性能。
- **了解模型成本**：关于在 **Aider** 中使用不同模型相关成本的讨论也随之而起，一些用户注意到，根据文件配置的不同，每条消息的费用在 **0.05 到 0.08 美元** 之间。
  
  - 这引发了关于小型项目如何因特定的模型行为和文件处理而产生惊人高额费用的讨论。
- **将 Aider 与 Hyperbolic 的 Qwen 2.5 Coder 集成**：用户在集成 **Hyperbolic 的 Qwen 2.5 Coder** 时遇到了问题，最初由于缺少元数据导致错误，但在修正安装设置后获得了成功。
  
  - 遵循 main 分支更新的明确指令有助于解决这些集成挑战。
- **使用 Aider 生成提交信息**：要为 Git 仓库中未提交的更改生成提交信息，用户可以使用 `aider --commit --weak-model openrouter/deepseek/deepseek-chat` 等命令，或者在 Aider 内部使用 `/commit`。
  
  - 这些命令将创建一个提交信息，但会直接提交所有更改，而不会提示选择文件。

**提到的链接**：

- [Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/1131200896827654144/1131200896827654149/1307075724695306250)：Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。
- [来自 undefined 的推文](https://x.com/paulgauthier)：未找到描述
- [来自 Melvin Vivas (@donvito) 的推文](https://x.com/donvito/status/1857044007911633003)：我正在创办一家新初创公司。这是我的 AI 团队 http://bolt.new - 前端工程师 http://aider.chat - 后端工程师 @crewAIInc - 产品设计师/经理 Claude AI - 内容创作者 @per...
- [‎Gemini - 直接访问 Google AI](https://gemini.google.com/share/6d141b742)：由 Gemini 创建
- [来自 Logan Kilpatrick (@OfficialLoganK) 的推文](https://x.com/OfficialLoganK/status/1857106844805681153)：仍在完善 AIS 的一些细节，很快将在 API 中提供，敬请期待并祝玩得开心！
- [Google AI Studio](https://aistudio.google.com/)：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。
- [‎Gemini - 老年人的挑战与解决方案](https://gemini.google.com/share/6d141b742a13)：由 Gemini 创建
- [脚本化 aider](https://aider.chat/docs/scripting.html)：你可以通过命令行或 Python 对 Aider 进行脚本化。
- [无标题](https://ai.google.dev/gemini-api/docs/models/experimental-models)：未找到描述
- [Ravel](https://ravel.acenturyandabit.xyz/)：未找到描述
- [PHP: 手册快速参考](https://www.php.net/releases/](https://www.php.net/releases/)\n)：PHP 是一种流行的通用脚本语言，驱动着从你的博客到世界上最流行的网站的一切。
- [Aide - 你的 AI 编程助手](https://aide.dev/)：以你认识的最强程序员的速度和知识进行编码。Aide 就在你身边。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1306755094649376829) (38 条消息🔥):

> - `Aider 的 API Key 配置`
> - `Aider 权限错误`
> - `在 Aider 中使用 XAI`
> - `适用于 Aider 的低成本模型`
> - `在 Docker 中运行 Aider`

- **Aider 的 API Key 配置问题**：一位用户在使用 **.aider.conf.yml** 进行 API Key 配置时遇到问题，发现是一个旧的 API Key 导致了失败。
  
  - 另一位用户建议运行 Aider 时加上 `—verbose` 参数，以检查加载了哪个 YAML 文件以及哪个 API Key 正在生效。
- **Aider 中的随机权限错误**：一位用户报告在 Windows 上运行 Aider 时出现随机权限错误，特别是在尝试写入文件时，尽管是以管理员身份运行的。
  
  - 权限错误间歇性地发生在文件上，让用户困惑这是否与 LLM 相关还是其他问题。
- **在 Aider 中使用 XAI 的挑战**：一位用户报告在使用 **XAI 模型 xai/grok-beta** 时遇到困难，收到关于 LLM 提供商无法识别的错误。
  
  - 这似乎源于过时的 `litellm` 依赖项，并提供了相关的 GitHub issues 链接以获取更多背景信息。
- **为 Aider 寻找更便宜的模型替代方案**：一位用户询问在使用 *diff* 格式运行 Aider 时最便宜的模型，并提到现有模型的成本很高。
  
  - 他们指出仅在 ***Haiku 3.5*** 模型上取得了成功，并由于费用问题正在探索替代方案。
- **在 Docker 中运行 Aider 及其配置**：一位用户询问在 Docker 中运行时 Aider 将 sentence transformer 模型存储在哪里，目的是将其映射到容器外部。
  
  - 在重新评估其设置后，他们从使用 Docker 切换到直接更新环境变量，以使用更近期的配置。

**提到的链接**：

- [XAI | liteLLM](https://docs.litellm.ai/docs/providers/xai): https://docs.x.ai/docs
- [Providers | liteLLM](https://docs.litellm.ai/docs/providers): 了解如何在 LiteLLM 上部署和调用来自不同提供商的模型
- [xai model not being recognized · Issue #2295 · Aider-AI/aider](https://github.com/Aider-AI/aider/issues/2295): Issue /model xai/grok-beta Warning for xai/grok-beta: Unknown context window size and costs, using sane defaults. Did you mean one of these? xai/grok-beta Aider v0.62.1 Model: xai/grok-beta with wh...

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/) (1 条消息):

renanfranca9480: [https://supermaven.com/blog/cursor-announcement](https://supermaven.com/blog/cursor-announcement)

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1306800130292322366) (118 条消息🔥🔥):

> - `Hugging Face Sign-Up Issues` (Hugging Face 注册问题)
> - `OCR Accuracy and Improvement` (OCR 准确率与改进)
> - `Reinforcing OCR Models` (强化 OCR 模型)
> - `Loading LoRA State Dicts` (加载 LoRA State Dicts)
> - `Shortening URLs for Hugging Face` (缩短 Hugging Face 的 URL)

- **Hugging Face 注册页面**：多名用户报告在尝试注册 Hugging Face 时遇到 400 错误代码和白屏。
  - 该问题在不同地区的多个用户中持续存在，表明可能存在系统性问题。
- **提高 OCR 准确率**：讨论探讨了如何通过训练显著提高 OCR 准确率，特别是配合适当的文档扫描应用。
  - 参与者表示，如果满足合适的条件，实现接近 100% 的成功率是可能的。
- **强化 OCR 模型**：贡献者建议将 OCR 失败的案例反馈到训练过程中，以增强模型在特定应用中的性能。
  - 基于先前的失败案例对模型进行持续调整和训练，可以实现近乎完美的识别率。
- **使用 State Dicts 加载 LoRA**：一名用户寻求关于使用特定 state_dict 格式加载 LoRA 模型的帮助，并提供了 key 及其 tensor 大小的示例。
  - state_dict 的细节显示它涵盖了基础模型 key 的很大一部分，引发了关于文档的疑问。
- **缩短 Hugging Face URL**：用户讨论了缩短 Hugging Face URL 的想法，类似于 YouTube 使用 'youtu.be' 链接的方式。
  - 虽然 hf.co 已经存在，但参与者强调 URL 仍然需要 '/spaces/' 部分，仅能节省约 9 个字符。

**提到的链接**：

- [Use authentication in huggingface Gradio API!(hosting on ZeroGPU)](https://discuss.huggingface.co/t/use-authentication-in-huggingface-gradio-api-hosting-on-zerogpu/115565)：伙计们。我已经将代码托管在 ZeroGPU 上（为此我订阅了 PRO）。当我以 PRO 用户身份访问网页时，确实获得了比免费用户多 5 倍的使用配额。但是当...
- [Remiscus/MediGen · Update README.md](https://huggingface.co/Remiscus/MediGen/discussions/1)：未找到描述
- [Cat Drugs GIF - Cat Drugs Tripping - Discover & Share GIFs](https://tenor.com/view/cat-drugs-tripping-funny-animals-gif-13749008)：点击查看 GIF
- [GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs](https://github.com/turboderp/exllamav2)：一个用于在现代消费级 GPU 上本地快速运行 LLM 的推理库 - turboderp/exllamav2
- [zero-gpu-explorers/README · Discussions](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions)：未找到描述
- [Getting Started With The Python Client](https://www.gradio.app/guides/getting-started-with-the-python-client)：Gradio 分步教程

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1306781863213531197) (3 条消息):

> - `Missing user updates` (缺失的用户更新)
> - `Online tutorials progress` (在线教程进度)

- **关键用户更新缺失**：<@930102195330900009> 的更新明显缺席，一名成员用大哭的表情符号表达了失望。
  - *人们期待着他们能回到平台分享见解和贡献。*
- **在线教程之旅**：一名成员分享了他们从在线教程开始学习的经历，强调他们从 **基础 (basics)** 开始并逐渐进阶。
  - *这种对学习的专注得到了他人的赞赏和积极回应。*

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1306711759637446677) (3 条消息):

> - `Post Transparency` (帖子透明度)
> - `Community Feedback` (社区反馈)

- **呼吁帖子透明度**：一名成员表示，在发布帖子时不明确说明自己与平台的隶属关系感觉是*虚伪的*。
  - 建议在未来的帖子中澄清关系，以维护社区内的信任。
- **对内容真实性的怀疑**：另一名成员对最近的一篇帖子表示担忧，认为其读起来像*诈骗 (scam)*，并质疑其真实性。
  - 这表明成员们对共享内容的公信力日益警惕。

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1306918700326322207) (9 条消息🔥):

> - `SnowballTarget environment`
> - `OKReddit dataset`
> - `RWKV models`
> - `Runtime errors in spaces`
> - `Search frustrations on Hugging Face`

- **分享 SnowballTarget 环境结果**：成员们讨论了他们在深度强化学习课程中使用 **SnowballTarget environment** 的经验，其中一位分享了他们达到了约 **28** 的奖励。
  
  - 他们尝试了不同的参数，如 **epochs**、**learning rate** 和 **buffer size**，显示出有所改进，但仍面临障碍。
- **OKReddit 数据集介绍**：一位成员介绍了 **OKReddit dataset**，这是一个精选的 **5TiB** Reddit 内容集合（涵盖 2005 年至 2023 年），可用于研究目的。
  
  - 虽然标记为 alpha 版本，但它提供了一个 **经过过滤的 subreddits 列表** 并包含了访问链接，邀请大家讨论其潜在用途。
- **澄清 RWKV 模型与 Recursal.ai 模型的区别**：一位成员指出，虽然 **RWKV models** 可能与训练严谨性有关，但它们在数据集细节方面与 **Recursal.ai** 模型不同。
  
  - 计划在未来进行集成，暗示模型训练方法将持续演进。
- **讨论 Spaces 中的运行时错误**：一位用户报告了在尝试在 Gradio 中使用 RWKV 模型时遇到的 **runtime error**，暗示存在库加载问题。
  
  - Traceback 信息显示缺少 **libnvidia-ml.so.1**，这引发了成员们对设置问题的担忧。
- **对 Space 搜索结果的沮丧**：由于在 Hugging Face Spaces 中搜索时出现大量无法运行的结果，尤其是当这些结果显示为有缺陷时，引发了不满。
  
  - 该成员表示，由于许多列表存在错误，区分可运行的 Spaces 变得具有挑战性，令人感到沮丧。

**提到的链接**：

- [v5-EagleX-v2-7B-gradio - a Hugging Face Space by RWKV](https://huggingface.co/spaces/RWKV/v5-EagleX-v2-7B-gradio)：未找到描述
- [recursal/OKReddit-alpha · Datasets at Hugging Face](https://huggingface.co/datasets/recursal/OKReddit-alpha)：未找到描述
- [Walter White Walter GIF - Walter White Walter Falling - Discover & Share GIFs](https://tenor.com/view/walter-white-walter-falling-breaking-bad-dm4uz3-gif-18078549)：点击查看 GIF

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1306776869667999785) (6 条消息):

> - `GPU compatibility`
> - `Motherboard decisions`
> - `PCIe bandwidth considerations`

- **充分利用你的 GPU**：一位成员指出，如果 **CPU 支持**，你可以运行多达 **4 个双槽 GPU**，但检查 **PCIe bandwidth** 至关重要，因为 X8 而非 X16 可能会显著降低性能。
  
  - *如果它们在物理空间上能装下，基本上可以保证不会互相造成损害*，尽管性能可能会有所不同。
- **主板消费选择**：有人建议避免在主板上花费过多，建议寻找比 **MI60** 更好但价格低于 **$1000** 的产品。
  
  - 高端主板的价格可能相当于一块 **3090 TI founder's edition** 或 **RX 7900XTX** 的价格。
- **现有硬件配置考量**：一位成员提到他们拥有一块运行 **3 个 RX 6800** 的 **MSI Godlike X570**，并分享了在他们的多 GPU 设置中 PCIe 通道分配为 **x8 x4 x4 x4**。
  
  - 尽管第四个 GPU 持续存在问题，该成员仍倾向于不购买新组件。

 

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1306799101593260042) (2 条消息):

> - `Legal Embedding Models`
> - `AI Assistant for Legal Systems`
> - `GTE Finetuning Data`

- **预训练法律嵌入至关重要**：为了有效地为法律应用训练 Embedding 模型，使用在法律数据上预训练的 Embedding 至关重要；否则，训练可能会变得漫长并导致偏见。
  
  - 专注于特定领域的训练可以提高准确性并加快进程。
- **将 GTE 用于 AI 法律助手**：一位成员提到他们在 AI Assistant Legal 系统中使用 **GTE** 进行 finetuning 数据，强调了其在工作流中的重要性。
  
  - 这种方法表明，尽管可能需要法律专业化，但他们选择了通用 Embedding。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1306748284907556916) (102 条消息🔥🔥):

> - `学习 Triton 和 CUDA 的重要性`
> - `语言模型的现状`
> - `微调的输入处理`
> - `模型对比与利用`
> - `社区成员的个人经验`

- **Triton 和 CUDA：未来的重点**：成员们表示，未来将会有大量围绕 **Triton** 和 **CUDA** 的工程工作，这表明它们是值得学习的宝贵技能。
  
  - 有人提到，在创建“更好”的模型时，**收益递减 (diminishing returns)** 是一个令人担忧的问题，这表明重点正在转向效率的提升。
- **语言模型偏好的转变**：讨论显示，**Mistral 7B v0.3** 模型被认为已经过时，而 **Qwen/Qwen2.5-Coder-7B** 由于在充足的数据上进行了训练，受欢迎程度正在上升。
  
  - 成员们分享了对各种模型**效能 (efficacy)** 的见解，对比了 **Gemma2** 和 **GPT-4o** 的性能。
- **微调导出可能存在的问题**：一位用户强调了将微调后的模型导出到 **Ollama** 时遇到的困难，暗示导出后的性能存在不一致性。
  
  - 另一位成员建议，排查问题可能涉及处理 **chat templates** 或内存管理问题，特别是与 **bitsandbytes** 相关的问题。
- **无酒精鸡尾酒 vs 鸡尾酒：周末氛围**：成员们轻松地讨论了个人饮品，有人计划制作马提尼，而其他人则自认是 **mocktail gang**（无酒精鸡尾酒派）的一员。
  
  - 对话反映出一种轻松的氛围，通过共同的兴趣促进了社区联系。
- **用于训练模型的本地监控工具**：一位成员正在寻找本地监控模型训练的替代方案，表示对 **TensorBoard** 之外更轻量级的框架感兴趣，用于跟踪 loss、梯度和学习率。
  
  - 大家对使用 **TensorBoard** 达成了共识，但对更高效的可视化训练指标的工具感到好奇。

**提到的链接**：

- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1857283102315352488)：Nvidia 发布 LLaMA-Mesh：统一 3D Mesh 生成与语言模型
- [Google Colab](https://colab.research.google.com/drive/1nOnpNubkGL5lZhKUBkFOWE5UajzieqCD?usp=sharing#scrollTo=r2v_X2fA0Df5)：未找到描述
- [Boo GIF - Boo - 发现并分享 GIF](https://tenor.com/view/boo-gif-19787475173016375)：点击查看 GIF
- [Qwen/Qwen2.5-Coder-7B-Instruct · Hugging Face](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)：未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1306736089360240743) (8 条消息🔥):

> - `饮食选择`
> - `大米偏好`

- **关于动物源产品的讨论**：一位成员指出饮食中包含大量动物源产品，如 **chicken, egg, milk**，同时质疑为何缺乏 **nuts and seeds**。
  
  - 另一位成员幽默地回应道：“我能说什么呢，我就是个‘动物’嘿嘿”，承认自己不吃坚果或种子。
- **关于饮食习惯的玩笑**：一位成员开玩笑说“我什么都不吃”，以此幽默地回应关于其饮食的评论。
  
  - 幽默在继续，另一位成员也随声附和，简单地说了句 “Mike”。
- **更换大米种类以改善消化**：一位成员提到用 **Basmati rice** 替代 **white rice**，觉得这样感觉更好。
  
  - 他们进一步解释说，虽然 **jasmine rice** 让他们感到不适，但 **Basmati** 更容易消化。

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1306800739137622066) (20 条消息🔥):

> - `Unsloth 安装问题`
> - `对 Lora+ 的支持`
> - `用于 Qwen2.5-Coder 的 Svelte 数据集`
> - `歌词数据集准备`
> - `为数学方程微调 Llama3.1b`

- **因缺失 torch 导致的 Unsloth 安装错误**：一位用户在尝试在其 notebook 中安装 Unsloth 和 xformers 时遇到了错误提示 **ModuleNotFoundError**: No module named 'torch'。
  
  - *似乎即使安装了 torch，问题仍然存在*，其他人建议验证当前环境中的安装情况。
- **Unsloth 对 Lora+ 的支持**：一位用户询问 Unsloth 是否支持 **Lora+**，对此一位参与者确认，如果通过 Pull Request 提交，它是支持的。
  
  - 进一步的讨论表明，实现过程应该非常直接。
- **关于 Qwen2.5-Coder 的 Svelte 数据集建议**：由于 **Qwen2.5-Coder** 的效果不佳，一位用户为 Svelte 文档创建了一个全面的数据集。
  
  - 有建议称应使用 Qwen2.5-Coder 的 *none instruct* 版本以获得更好效果，因为 instruct 是一个经过微调的变体。
- **创建歌词数据集**：一位用户详细介绍了他们致力于创建一个基于 5780 首歌曲及相关元数据生成歌词的模型。
  
  - 他们被建议利用 *Alpaca chat template* 来有效地构建其数据集结构。
- **在数学方程上微调 Llama3.1b**：一位用户寻求关于提高其微调 **Llama3.1b** 解决数学方程模型准确率的建议，目前其准确率为 77%。
  
  - 尽管在数据集上实现了较低的 Loss，他们仍在尝试超参数搜索（hyperparameter sweeps）以冲刺至少 **80% 的准确率**。

**提到的链接**：

- [Google Colab](https://colab.research.google.com/drive/13_z6x9ejloE8mC1IFR8qPmP-7YaFknN9#scrollTo=IqM-T1RTzY6C.)：未找到描述
- [Dreamslol/svelte-5-sveltekit-2 · Datasets at Hugging Face](https://huggingface.co/datasets/Dreamslol/svelte-5-sveltekit-2)：未找到描述

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1306888591019868212) (12 条消息🔥):

> - `直方图绘制替代方案`
> - `开源 AI 定义`
> - `关于 AI 编程的讨论`
> - `用于大型数据集的 Datashader`

- **寻求更快的直方图选项**：由于 **Matplotlib** 性能较慢，一位成员询问了绘制约 **100,000 个标量**直方图的替代方案。
  
  - 另一位成员建议使用 **Datashader** 作为高效可视化大型数据集的潜在解决方案。
- **辩论开源 AI 分类**：关于像 **Granite** 这样的 AI 系统是否根据 **IBM** 的定义被归类为开源 AI，进行了一场关键讨论。
  
  - 有观点指出，虽然一种看法认为**所有数据必须公开**，但其他人认为描述性的数据披露足以规避法律风险。
- **澄清开源数据规则**：一位成员指出，关于充分披露的要求是适用于数据还是代码，在开源 AI 定义中存在混淆。
  
  - 链接到 **Open Source AI Definition** 页面，另一位成员强调了 AI 中自主性、透明度和协作的需求。
- **AI 编程经验**：一位成员分享了他们在物理和编程方面的背景，特别是 **C/C++/Fortran**，以及他们在 **TensorFlow Lite** 方面的经验。
  
  - 他们表达了对使用 AI 聊天生成 **C++ 代码**的兴趣，强调了该项目开放性的吸引力。

 

**提到的链接**：[The Open Source AI Definition – 1.0](https://opensource.org/ai/open-source-ai-definition)：版本 1.0 前言 为什么我们需要开源人工智能 (AI)。开源已经证明，在消除学习、使用、共享的障碍后，每个人都能获得巨大的利益...

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1306754323237048350) (97 messages🔥🔥):

> - `Scaling Laws in AI`
> - `LLM Capabilities`
> - `Encrypted LLM Communication`
> - `Diminishing Returns in Model Training`
> - `Synthetic Tasks and Relevance`

- **Twitter 上关于 Scaling Laws 的言论**：成员们讨论了最近来自 Twitter 的说法，称 **Scaling 已死**，这引发了分歧，并有人主张结论应以具体的论文为依据，而非传闻。
  
  - 一些人引用了主流实验室的**记者消息来源**，报道了最近训练运行（training runs）中令人失望的结果，对这些说法的有效性提出了质疑。
- **对 LLM 能力局限性的担忧**：对话涉及模型的**能力局限性**，有人建议虽然当前的架构可能会遇到边际收益递减，但它们可能会阻碍 **AGI 类模型**的真正进步。
  
  - 参与者强调 LLM 需要掌握复杂的任务，如 **Diffie-Hellman 密钥交换**，并认为目前的模型可能无法在内部保留私钥，从而引发了对隐私的担忧。
- **边际收益递减与研究兴趣**：成员们对 AI 模型训练中出现**边际收益递减**表示期待，因为这将激励进一步的研究和对复杂问题的探索，而不是将 ML 仅仅简化为纯粹的工程任务。
  
  - 讨论指出，为了提高 LLM 的实际可用性，有必要参与更深层次的理解和澄清能力。
- **合成任务的相关性**：**合成任务**（如算术）的相关性受到了质疑，辩论集中在它们对 LLM 现实世界能力的适用性上。
  
  - 一位用户建议，仅仅关注合成任务可能无法反映 LLM 的实际效用，并提到 **AIW 项目**可能并不相关。
- **通信中的同态加密**：展开了关于使用**同态加密（Homomorphic Encryption）**在不泄露敏感信息的情况下实现与 LLM 安全通信的潜力的对话。
  
  - 参与者思考了此类加密方法是否能有效地将私钥隐藏在模型激活值（activations）中，从而在安全需求与性能可用性之间取得平衡。

**提到的链接**：

- [Should I Use Offline RL or Imitation Learning?](https://bair.berkeley.edu/blog/2022/04/25/rl-or-bc/)：BAIR 博客
- [来自 Karol Hausman (@hausman_k) 的推文](https://x.com/hausman_k/status/1640743548990668800)：这是 Offline RL 优于 Behavioral cloning 的根本原因：它允许你以对任务/奖励最优的方式剪切和缝合轨迹。你不需要...
- [Planting Undetectable Backdoors in Machine Learning Models](https://arxiv.org/abs/2204.06974)：鉴于训练机器学习模型所需的计算成本和技术专业知识，用户可能会将学习任务委托给服务提供商。我们展示了恶意学习者如何植入...
- [GitHub - apple/ml-cross-entropy](https://github.com/apple/ml-cross-entropy)：通过在 GitHub 上创建账户，为 apple/ml-cross-entropy 的开发做出贡献。
- [Building machines that learn and think with people - Nature Human Behaviour](https://www.nature.com/articles/s41562-024-01991-9)：在这篇 Perspective 中，作者提出了协作认知科学的观点，以设计可被视为思想伙伴的系统，这些系统的构建是为了满足我们的期望并符合...

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1306718271864832032) (6 messages):

> - `Pythia model suite`
> - `Mixture-of-Experts (MoE)`
> - `Hidden States Unconference`
> - `Transformer heads and antonyms`

- **为 Pythia 探索 Mixture-of-Experts**：发起了一场关于为 **Pythia model suite** 开发 **Mixture-of-Experts** 版本的讨论，质疑是应该完全复制其训练设置，还是采用现代化的超参数（如 **SwiGLU**）和更长时间的训练。
  
  - 成员们发表了意见，对比了 **OLMo** 和 **OLMOE** 等当前模型，指出了在 **data order**（数据顺序）和跨规模一致性方面的差异。
- **Hidden States 非正式会议（Unconference）宣布召开**：一场名为 **Hidden States** 的非正式会议定于 **2024 年 12 月 3 日**在旧金山举行，重点讨论 **AI interfaces** 和隐藏状态。
  
  - 主讲嘉宾包括 **Leland McInnes** 和 **Linus Lee**，并设有激励措施鼓励独立研究员和学生参加。
- **Transformer 中的反义词头（Antonym Heads）**：分享了一篇讨论 **transformer heads** 计算反义词的研究结果，揭示了它们在具有可解释特征值（eigenvalues）的各种模型中普遍存在。
  
  - 该分析包括 **OV circuits**、消融研究（ablation studies），并展示了如 **'hot' - 'cold'** 等反义词示例，确认了它们在语言模型中的功能。

**提到的链接**：

- [无标题](https://hiddenstates.org/)：未找到描述
- [来自 Nora Belrose (@norabelrose) 的推文](https://x.com/norabelrose/status/1857159435686384096)：如果有 Pythia 模型套件的 Mixture-of-Expert 版本，你想用它回答什么样的问题？我们是否应该尝试精确复制 Pythia 的训练设置，但使用 M...
- [反义词头在语言模型中预测语义对立面 — LessWrong](https://www.lesswrong.com/posts/XXK2T4EcbHRkRTBce/antonym-heads-predict-semantic-opposites-in-language-models)：通常，大型语言模型中的注意力层执行两种类型的计算：它们识别哪些 token 位置包含相关信息……

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1306739590165430406) (66 messages🔥🔥):

> - `TEE Wallet Collation Concerns`
> - `NVIDIA NV-Embed-v2 Release`
> - `RLHF vs SFT for Fine-tuning`
> - `Tee's Twitter Activity`

- **提出 TEE 钱包整理问题**：成员们对 TEE 频繁更换钱包表示沮丧，强调这损害了 Bot 的自主性和社区信任。
  
  - 成员们表达了对建立稳定性的担忧，以便用户能够放心地在链上（on-chain）与 Bot 交互。
- **NVIDIA 发布 NV-Embed-v2 模型**：NVIDIA 宣布发布 [NV-Embed-v2](https://huggingface.co/nvidia/NV-Embed-v2)，这是一款在 Massive Text Embedding Benchmark 上排名顶尖的嵌入模型，得分为 72.31。
  
  - 它引入了创新的训练技术，包括改进的潜向量（latent vectors）保留和独特的难负采样（hard-negative mining）方法。
- **关于模型 RLHF 与 SFT 的辩论**：随后讨论了在训练 Llama 3.2 时，人类反馈强化学习（RLHF）与有监督微调（SFT）相比的资源需求。
  
  - 成员们建议，虽然 RLHF 需要更多的 VRAM，但在资源有限的情况下，SFT 可能是更可行的起点。
- **Tee 的不活跃引发疑问**：由于 Tee 长期没有发布推文，引发了对其运行状态的猜测。
  
  - 有人开玩笑说 Tee 可能在周末休长假了，促使其他人去检查它的状态。

**提到的链接**：

- [测试时训练（Test-Time Training）对抽象推理的惊人效果](https://arxiv.org/abs/2411.07279)：语言模型在训练分布内的任务上表现出色，但在需要复杂推理的新问题上往往表现挣扎。我们研究了 t... 的有效性。
- [来自 Dr. Dad, PhD 🔄🔼◀️🔽▶️ (@GarrettPetersen) 的推文](https://x.com/garrettpetersen/status/1857117202622902305?s=12)：得知 Gwern 是一个才华横溢但怀才不遇的人有点难过。我一直以为他是计算机科学教授或软件工程师。
- [nvidia/NV-Embed-v2 · Hugging Face](https://huggingface.co/nvidia/NV-Embed-v2)：未找到描述
- [Your Life Story](https://lifestorys-b93f5c9c5deb.herokuapp.com/)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1306997488447651900) (2 messages):

> - `Mixing datasets for LLM training`（LLM 训练的数据集混合）
> - `Matching strategies for datasets`（数据集的匹配策略）

- **寻求关于数据集混合的见解**：一位成员正在寻找关于如何在 **LLM 训练**的不同阶段有效地**混合和匹配数据集**的资源或见解。
  
  - 他们强调需要*技巧或最佳实践*来增强训练过程。
- **对匹配策略的兴趣**：该成员表达了对寻找与数据集混合相关的多种**匹配策略**方法的期待。
  
  - 这反映了对优化 **LLM 训练**工作流的持续关注。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

adjectiveallison: [https://arxiv.org/abs/2411.04732](https://arxiv.org/abs/2411.04732)

使用缩小 29 倍的模型实现图像识别的 SOTA。

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1306738520555126884) (6 messages):

> - `AI-driven translation tool`（AI 驱动的翻译工具）
> - `Resume website transformation`（简历网站转换）
> - `Translation services`（翻译服务）
> - `Slang translation`（俚语翻译）
> - `Cultural nuances in translation`（翻译中的文化细微差别）

- **先进的翻译工具脱颖而出**：它使用 **AI 驱动的 Agent 工作流**，通过专注于**文化细微差别**和适应性，超越了传统的机器翻译。
  
  - *与大多数系统不同*，它考虑了地区方言、正式程度、语气和基于性别的细微差别，以实现更准确的翻译。
- **将您的简历转换为精美的网站**：用户可以通过 [此服务](https://resumetosite-b55155107b3e.herokuapp.com/) 上传简历，在几分钟内获得一个**专业的、响应式的 Bootstrap 网站**。
  
  - 该过程承诺快速生成网站，体现了用户友好的体验。
- **俚语翻译变得简单**：[Slang translator](https://slangtranslator.cameronaaron.com/) 提供了一种将俚语转换为标准语言的方法，增强了沟通和理解。
  
  - 该工具提供了一个专门针对口语表达的额外翻译层。
- **可定制的翻译体验**：这款先进翻译工具的三步流程包括翻译、反思和精炼，以精确提升输出质量。
  
  - 这允许用户通过调整与语气和地区差异相关的 Prompt 来定制体验，同时保持术语的一致性。

**提到的链接**：

- [Resume to Website Generator](https://resumetosite-b55155107b3e.herokuapp.com/)：未找到描述
- [Advanced Translation Tool - Accurate and Culturally Nuanced Translations](https://translate.cameronaaron.com/)：在语言翻译中考虑文化细微差别、语境、正式程度、语气和性别因素。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/) (1 messages):

adjectiveallison: [https://arxiv.org/abs/2411.04732](https://arxiv.org/abs/2411.04732)

使用缩小 29 倍的模型实现图像识别的 SOTA。

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1307019697262428311) (2 messages):

> - `MistralNemo StarCannon`
> - `Celeste`
> - `Perplexity Citations`（Perplexity 引用）
> - `Beta Features`（Beta 功能）
> - `Model Updates`（模型更新）

- **停止支持 MistralNemo StarCannon 和 Celeste**：由于唯一的提供商已停止支持，**MistralNemo StarCannon** 和 **Celeste** 不再可用。
  
  - 这一变化会影响所有依赖这些模型进行项目的用户。
- **Perplexity 在 Beta 版中引入 Grounding 引用**：Perplexity 模型现在具有 `citations` 的 Beta 实现，允许 URL 随补全响应（completion responses）一同返回。
  
  - 这一新属性为用户提供了获取更多信息的直接链接，增强了生成内容的可靠性。

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1306716058299666433) (65 条消息🔥🔥):

> - `Gemini API 可用性`
> - `OpenRouter 上的 Rate Limits`
> - `Hermes 405B 模型`
> - `Perplexity Citations 上线`
> - `Magnum 72B 模型评估`

- **Gemini API 现在可以访问**：成员们讨论了 **Gemini** 现在可以通过 API 使用，对其功能表示期待。
  
  - 尽管部分用户仍未看到变化，但显然该模型被预期能改善用户交互。
- **理解 OpenRouter 上的 Rate Limits**：对话强调了免费模型的 **Rate Limits**，特别指出每天 **200 次请求** 的限制。
  
  - 成员们强调，这一限制实际上阻碍了在生产环境中的利用，使得免费模型不太实用。
- **频繁回归 Hermes 405B**：Fry69_dev 指出，在测试了多个模型后，他们因其效率而不断回归 **Hermes 405B**。
  
  - 尽管其成本影响了利润率，但对许多用户来说，其性能仍然是无可比拟的。
- **Perplexity Citations 现已在 OpenRouter 上线**：Alex Atallah 宣布 **Perplexity Citations** 已在 OpenRouter 上线，引起了用户的兴趣。
  
  - 然而，一些用户报告称由于 API 响应要求，他们目前还无法访问该功能。
- **Magnum 72B 写作风格评估**：Frehref 提到 **Magnum 72B** 模型尽管价格昂贵，但因其良好的写作风格而受到认可。
  
  - Takefy 计划测试该模型，但对与其使用相关的成本表示担忧。

**提到的链接**：

- [Limits | OpenRouter](https://openrouter.ai/docs/limits): 设置模型使用限制
- [no title found](https://ai.google.dev/gemini-api/docs/models/experimental-models): 未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1306718356828717116) (9 条消息🔥):

> - `Custom Provider Keys 访问权限`

- **对 Custom Provider Keys 的高需求**：包括 giampie.ro 和 @wyatt02146 在内的多位用户表达了申请 **Custom Provider Keys** 访问权限的愿望。
  
  - 请求的数量表明了对这些 **Keys 的显著兴趣**，从而引发了关于获取流程的问题。
- **关于 Custom Provider Keys 访问流程的咨询**：一位成员询问，鉴于社区的大量请求，获取 **Custom Provider Keys** 的访问权限是否可以自动化。
  
  - *这里有很多人请求访问* 表明用户渴望明确获得访问权限所涉及的步骤。
- **关于 Custom Provider Keys 的普遍请求**：用户不断重申对 **Custom Provider Keys** 的请求，多位成员如 @cuts2k 和 @schwemschwam 都表达了兴趣。
  
  - 重复的请求反映了渴望使用该功能的社区成员对访问权限日益增长的期待。
- **请求支持和后续步骤**：像 @pjtidder 这样的成员寻求有关访问 Beta 版 **Custom Provider Keys** 的后续步骤信息，并表示需要支持。
  
  - 这些请求强调了社区对推进其访问请求的明确指导的兴趣。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1306721598358945802) (60 messages🔥🔥):

> - `Stable Diffusion GPU Usage`
> - `WebUI Installation Guides`
> - `Image Blending Research Paper`
> - `Video Upscaling Techniques`
> - `Using Low Denoise Inpaint`

- **在 Nvidia GPU 上运行 Stable Diffusion 的求助**：一位用户表示在配置 Stable Diffusion 以利用其专用 Nvidia GPU 而非集成显卡时遇到困难。
  
  - 另一位用户指出频道置顶消息中提供的安装指南可以提供帮助。
- **ComfyUI 与 SwarmUI 的易用性对比**：一名成员讨论了 ComfyUI 的复杂性，并指出 SwarmUI 简化了配置中涉及的技术流程。
  
  - 建议尝试 SwarmUI 以获得更简单的安装体验和更一致的结果。
- **寻找图像混合研究论文**：一位用户寻求帮助寻找最近一篇关于图像混合的研究论文，记得作者来自 Google，但难以定位。
  
  - 另一名成员建议在 arXiv 上搜索关于图像混合的 Google 论文，可能会有收获。
- **修复视频解剖结构问题的方法**：一名成员分享了他们放大视频的方法，即每 0.5 秒提取一帧来修正现有的不准确之处。
  
  - 讨论还包括使用 Flux Schnell 和其他方法来实现快速推理（inference）结果。
- **关于应用低重绘强度（low denoise）inpaint 的咨询**：一位用户询问如何对图像的特定区域进行低重绘强度的 inpainting 或 img2img 处理。
  
  - 建议包括使用 Diffusers 进行快速的 img2img 工作流，通过极少的步骤来精修图像。

**相关链接**：

- [genmo/mochi-1-preview · Hugging Face](https://huggingface.co/genmo/mochi-1-preview)：未找到描述
- [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
- [alibaba-pai/CogVideoX-Fun-V1.1-5b-InP · Hugging Face](https://huggingface.co/alibaba-pai/CogVideoX-Fun-V1.1-5b-InP)：未找到描述
- [THUDM/CogVideoX1.5-5B-SAT · Hugging Face](https://huggingface.co/THUDM/CogVideoX1.5-5B-SAT)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1306723103514034286) (16 messages🔥):

> - `Scaling Laws Theory`
> - `GPT-3 Scaling Hypothesis`
> - `Funding Round Update`
> - `Changes in Twitter's Terms of Service`
> - `Meta AI Developments`

- **对 Scaling Laws 理论的担忧**：成员们对 **scaling laws** 理论的有效性提出了担忧，该理论认为更多的**算力和数据**将增强 AI 能力。
  
  - *一位成员表示宽慰*，称降低 **cross-entropy loss** 是提高 AI 能力的充分条件。
- **关于 GPT-3 Scaling 假设的见解**：一名成员提到了一篇关于 **scaling hypothesis** 的著名文章，指出随着问题复杂性的增加，神经网络会表现出泛化能力并展现出新的能力。
  
  - 他们强调，2020 年 5 月发布的 **GPT-3** 持续展示了规模化带来的收益，这与收益递减的预测相反。
- **大规模融资轮次公告**：一名成员分享了一条推文，指出下周结束的一轮融资将总计筹集 **60 亿美元**，主要来自**中东主权基金**，估值为 **500 亿美元**。
  
  - 据报道，这些资金将直接流向 **Jensen**，并推动科技领域的后续发展。
- **对 Twitter 新服务条款的担忧**：围绕 Twitter 更新的**服务条款**展开了讨论，该条款允许从明天开始利用用户内容训练 AI 模型。
  
  - 几位成员指出，使用用户内容是一个令人不安的趋势，反映了现状的悲哀。
- **Meta AI 的有趣进展**：一名成员提到了与 **Meta AI** 相关的*令人兴奋的进展*，对即将到来的内容表示热切期待。
  
  - 他们暗示了正在进行的改进以及 AI 领域发展的美好前景。

**相关链接**：

- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1857437923525931297)：筹码堆上又增加了 100,000 块芯片。融资轮显然将在下周结束。总计 60 亿美元（其中 50 亿美元来自中东主权基金），估值 500 亿美元。当然，它...
- [来自 Luiza Jarovsky (@LuizaJarovsky) 的推文](https://x.com/LuizaJarovsky/status/1857128480690917666)：🚨 突发：X 更新的服务条款将于明天生效。请注意 AI 条款：
- [The Scaling Hypothesis · Gwern.net](https://gwern.net/scaling-hypothesis)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**ml-drama**](https://discord.com/channels/1179127597926469703/1181746144821387334/1307080957903245413) (9 messages🔥):

> - `Ilya Sutskever 邮件`
> - `AI 认知`
> - `Misalignment 讨论`

- **AI Art 与 Hallucinations 引发辩论**：围绕 **AI Hallucinations** 的陈述以及对 **AI Art** 的批评引发了担忧，一位成员指出公众认知发生了重大转变。
  
  - *理论上对齐，实际上并非如此* 回应了对 AI 技术意识形态冲突的沮丧情绪。
- **TechEmails 揭示历史讨论**：成员们分享了 **TechEmails** 的链接，其中讨论了涉及 **Elon Musk, Sam Altman 和 Ilya Sutskever** 追溯到 **2017** 年关于 Misalignment 问题的通信。
  
  - 这些文件与正在进行的 **Elon Musk, et al. v. Samuel Altman, et al. (2024)** 案件相关，突显了 Alignment 担忧的历史背景。
- **成员对潜在影响深感兴趣**：对于 **Ilya 和 Sam** 之间围绕 **AI Misalignment** 讨论所产生的影响，反应中交织着惊讶与着迷。
  
  - 对长期存在担忧的承认，突显了这些问题在社区话语中的深度。

**提到的链接**：

- [来自 Internal Tech Emails (@TechEmails) 的推文](https://x.com/techemails/status/1857456141875196380?s=46)：未找到描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456137156669765)：未找到描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456139547316359)：未找到描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456141875196380)：未找到描述
- [来自 undefined 的推文](https://vxtwitter.com/TechEmails/status/1857456144211423482)：未找到描述
- [来自 Internal Tech Emails (@TechEmails) 的推文](https://x.com/TechEmails/status/1857459526267449790)：[此文件来自 Elon Musk, et al. v. Samuel Altman, et al. (2024)。]

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1306715399206731777) (12 messages🔥):

> - `对新模型的预期`
> - `Sam Altman 的 ICO 提案`
> - `Apple Silicon vs NVIDIA 性能`
> - `PyTorch Anaconda 包弃用`

- **新模型引发高预期**：成员们表达了对新模型的期待，其中一人表示，*这个模型最好能超出我们所有的预期*。
  
  - 另一位成员澄清说，这更多是关于将模型、数据和代码结合在一起的 *过程*。
- **Sam Altman 曾提议创建 OpenAI 加密货币**：一条推文透露，根据一份修正后的起诉书，**Sam Altman** 曾在 **2018** 年尝试通过 ICO 创建 **OpenAI 加密货币**。
  
  - 一位参与者对这一决定的影响评论道：*Lol*。
- **Apple Silicon 与 NVIDIA GPU 分析**：一篇详细的 [文章](https://blog.hjc.im/apple-uma-for-llms-problems.html) 讨论了 **Apple Silicon** 与 **NVIDIA** GPU 在运行 LLM 方面的竞争，强调了 Apple 平台的折中之处。
  
  - 虽然注意到 Apple 的新产品具有更高的内存容量，但结论认为 **NVIDIA 解决方案** 仍然更具性价比。
- **PyTorch 将停止发布 Anaconda 包**：最近的一份公告确认，**PyTorch** 将停止在其官方频道发布 **Anaconda packages**。
  
  - 欲了解更多详情，成员们被引导至 dev-discuss 上的一个 [讨论帖](https://dev-discuss.pytorch.org/t/pytorch-deprecation-of-conda-nightly-builds/2590)。

**提到的链接**：

- [来自 PyTorch (@PyTorch) 的推文](https://x.com/PyTorch/status/1857500664831635882)：我们宣布 PyTorch 将停止在 PyTorch 的官方 Anaconda 频道上发布 Anaconda 包。欲了解更多信息，请参阅 dev-discuss 上的以下帖子：https://dev-disc...
- [来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文](https://x.com/andrew_n_carr/status/1857261718466085296?s=46)：我为你达到了速率限制
- [来自 Anna Tong (@annatonger) 的推文](https://x.com/annatonger/status/1857290442930536475)：根据 Elon Musk 针对 OpenAI 起诉书的修正版本，Sam Altman 曾在 2018 年通过提议 ICO 尝试创建 OpenAI 加密货币。Lol
- [来自 Dumitru Erhan (@doomie) 的推文](https://x.com/doomie/status/1857156882353561998)：是谁把这个泄露给 The Information 的？ ;)
- [Apple 统一内存适合运行 LLM？理想很丰满，现实很骨感 | David Huang's Blog](https://blog.hjc.im/apple-uma-for-llms-problems.html)：未找到描述

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1306724926442573858) (8 条消息🔥):

> - `章鱼哥与海绵宝宝 Discord 装饰`
> - `Discord 购物体验`

- **Natolambert 获得章鱼哥装饰**：Natolambert 透露他为 Discord 上的**章鱼哥装饰**付了费，他兄弟的内部消息帮助他获得了独家选项。
  
  - 他随后澄清说，他实际上拥有的是**派大星 (Patrick)** 装饰，纠正了之前的言论。
- **海绵宝宝装饰仍然可用**：另一位成员指出，Discord 商店中仍有**海绵宝宝**装饰，为讨论增添了幽默基调。
  
  - 随后出现了许多“哈哈”时刻，成员们开玩笑地讨论他们的角色选择和装饰失误。

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1306712100126588990) (2 条消息):

> - `Personas`
> - `Prompts`

- **对 Personas 的兴奋**：一位成员用鼓掌表情符号表达了对 **Personas** 的热情，表示支持其使用。
  
  - 另一位成员做出了积极回应，建议有效的 **Prompts** 对于最大化其效用至关重要。
- **有效 Prompts 的重要性**：一位用户强调，使用正确的 **Prompts** 是在讨论中利用 **Personas** 的一种简单而有效的方法。
  
  - 这次对话反映了一种普遍观点，即 **Prompts** 在促进参与性互动中起着至关重要。

---

### **Interconnects (Nathan Lambert) ▷ #**[**retort-podcast**](https://discord.com/channels/1179127597926469703/1238601424452059197/1307067917627560027) (7 条消息):

> - `音频体验`
> - `The Gradient`
> - `播客节奏`

- **Torters 渴望新内容**：成员们表达了兴奋之情，指出已经有一段时间没有新内容了，其中一人强调 **Thomas** 在最近的开场白中渐入佳境。
  
  - *Torters 极度渴望*更多引人入胜的讨论和话题。
- **带着糟糕的音频开车去洛杉矶**：一位成员分享了他们在开车去洛杉矶时**音频体验不佳**的挫败感，表示这降低了他们的享受感。
  
  - 这种情绪引起了其他在通勤期间看重高质量音频的人的共鸣。
- **对 'The Gradient' 的复杂情感**：一位成员开玩笑地表达了对 **'The Gradient'** 的反感，并用表情符号加以强调，暗示其未达到预期。
  
  - 然而，随后他澄清说这只是个玩笑，并承认自己并非真的讨厌 Gradient。
- **关于播客节奏的讨论**：成员们讨论认为，播客**每月一次的频率**可能更容易管理，暗示他们一直忙于其他事务。
  
  - 这一提议的频率似乎更能契合他们目前的日程安排。

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/) (1 条消息):

memorypaladin: 直接问你的问题就行。

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1306948469776322590) (3 条消息):

> - `Kernel 变量传递`
> - `Grid 配置问题`

- **神秘的 Kernel 变量值**：一位用户报告了在向其 **Kernel** 传递变量时遇到的问题，即只有当 x 为 1 时输出值才正确，否则会产生随机数。
  
  - 当尝试像 **grid** (1, 3) 这样的配置时，结果显示为预期值，而使用 (2, y) 时，则显示看似未初始化的数据。
- **潜在的未初始化数据问题**：另一位用户建议，问题可能是由于读取了未初始化的数据，并请求提供代码片段以进一步分析该问题。
  
  - 这一回应表明需要调试策略，以确保在 **Kernel** 中使用变量之前对其进行了正确的初始化。

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1306835802210177096) (7 messages):

> - `FSDP and torch.compile`
> - `Using nsys for profiling`
> - `Proton profiler with Triton`
> - `Memory issues with nsys`
> - `Using emit_nvtx() for profiling`

- **FSDP 与 torch.compile 配合良好**：一位用户分享了来自 [torchtitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L40) 的在 **FSDP** 中包装 **torch.compile** 的代码示例，并表示他们没有遇到任何问题。
  
  - 这种方法看起来很有效，尽管他们没有尝试反向顺序。
- **使用 nsys 进行性能分析存在内存限制**：一位成员指出，**nsys** 在崩溃前可能会消耗高达 **60 GB 的内存**，这引发了对其在性能分析任务中可行性的担忧。
  
  - 另一位用户建议使用 `nsys profile -c cudaProfilerApi -t nvtx,cuda` 等标志来优化 **nsys** 的使用，以减少日志开销。
- **Proton 提供快速性能分析解决方案**：Proton 是 **Triton** 包含的性能分析器（Profiler），可以通过 `proton.start()` 启动并在之后结束，用于快速性能分析。
  
  - 用户应安装 **llnl-hatchet** 来执行查看性能分析结果的命令，这是 **nsys** 的一个更简单的替代方案。
- **使用 emit_nvtx() 进行详细性能分析**：为了深入了解 **CUDA** kernel 调用，建议在程序中使用 `emit_nvtx()` 并在 **nsys** profile 下运行。
  
  - 通过这种方法，用户可以直接观察哪些 **CUDA** kernel 是由特定的 **ATEN** 操作触发的，从而明确性能瓶颈。

**提到的链接**：

- [torchtitan/torchtitan/parallelisms/parallelize_llama.py at main · pytorch/torchtitan](https://github.com/pytorch/torchtitan/blob/main/torchtitan/parallelisms/parallelize_llama.py#L40)：一个用于大模型训练的原生 PyTorch 库。欢迎在 **GitHub** 上为 pytorch/torchtitan 的开发做出贡献。
- [pytorch/torch/autograd/profiler.py at 8043e67026b1cd5b5f1d17c46cd6fe579c322168 · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/8043e67026b1cd5b5f1d17c46cd6fe579c322168/torch/autograd/profiler.py#L889)：Python 中具有强大 **GPU** 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch

---

### **GPU MODE ▷ #**[**cool-links**](https://discord.com/channels/1189498204333543425/1189868872887705671/1307104269652070460) (2 messages):

> - `ZLUDA`
> - `NVIDIA GPUs`
> - `AMD and Intel GPUs`

- **ZLUDA 开发者讨论适用于非 NVIDIA GPU 的 CUDA**：在一段 [YouTube 视频](https://www.youtube.com/watch?v=ze25Sie2gVQ) 中，Andrzej Janik 解释了 **ZLUDA** 如何在 **AMD** 和 **Intel GPU** 上实现 **CUDA** 功能，这可能会改变 **GPU** 计算的格局。
  
  - 一位成员评论说，他们之前曾尝试邀请 Janik 进行讨论，并感叹道：“终于有人请到他了，哈哈”。
- **对 ZLUDA 突破的兴奋**：成员们对 Janik 在视频中的出现表示热烈欢迎，认为这是非 **NVIDIA GPU** 用户迈出的重要一步。
  
  - 人们对 **ZLUDA** 如何使 **GPU** 计算能力的获取更加民主化越来越感到好奇。

 

**提到的链接**：[#246 Developer Of ZLUDA: CUDA For Non Nvidia GPUs | Andrzej Janik](https://www.youtube.com/watch?v=ze25Sie2gVQ)：**CUDA** 是人们购买 **NVIDIA GPU** 的主要原因之一，但如果有一种方法也可以在 **AMD** 和 **Intel GPU** 上拥有这种计算能力呢？现在有了……

 

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1306895986840309811) (13 条消息🔥):

> - `NVCC vs Clang 性能`
> - `LLM 的 GPU 显存计算`
> - `在 Kokkos 中调试寄存器使用情况`
> - `循环展开对性能的影响`

- **NVCC 与 Clang 的性能差异**：一位成员开始测量 **NVCC** 与 **Clang** 之间的性能差异，并注意到每个线程使用的寄存器数量存在惊人的 **2 倍差异**。
  
  - 讨论指出，编译后的 **PTX** 差异可能导致寄存器使用的变化，其中循环展开（loop unrolling）可能是一个潜在因素。
- **计算训练所需的 GPU 显存**：有人提问如何提前计算训练或微调 **Large Language Model (LLM)** 所需的 **GPU 显存**。
  
  - 回复强调需要特定的方法或工具来准确估算显存需求。
- **在 Kokkos 中调试寄存器使用情况**：关于在 **Kokkos** 中调试寄存器使用策略的咨询，特别是关于 **launch bounds** 的使用。
  
  - 建议利用 Kokkos API 来定义执行策略，并使用 [launch bounds](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#launch-bounds) 以获得更好的控制。
- **SASS 汇编与循环展开见解**：寻求关于更激进的循环展开如何与代码中使用的指令数和寄存器数相关的澄清。
  
  - 作为回应，解释了指令增加会导致使用更多寄存器，以维持独立操作并避免流水线停顿（pipeline stalls）。
- **使用 NCU Source View 获取代码洞察**：推荐将 **NCU Source View** 作为识别高寄存器使用代码段的工具，辅助调试工作。
  
  - 指出该视图可以提供关于在 SASS 生成中应关注哪些部分进行潜在优化的见解。

 

**提到的链接**：[Execution Policies - Kokkos documentation](https://kokkos.org/kokkos-core-wiki/API/core/Execution-Policies.html#common-arguments-for-all-execution-policies)：未找到描述

 

---

### **GPU MODE ▷ #**[**triton-puzzles**](https://discord.com/channels/1189498204333543425/1219683012707487794/1306994730407559189) (1 条消息):

> - `Online Softmax`

- **讨论 Online Softmax 技术**：一位成员提到 **online softmax** 技术源于一篇介绍该技巧的[原始论文](https://link.to.paper)。
  
  - *注意到所提供的提示对于这项技术来说并不十分明显。*
- **理解 online softmax 的应用**：这一技巧对于优化某些应用中的模型性能至关重要，特别是在**神经网络**中。
  
  - 成员们鼓励探索各种资源，以更好地理解使用 online softmax 的影响。

 

---

### **GPU MODE ▷ #**[**webgpu**](https://discord.com/channels/1189498204333543425/1262121239044948009/1306786776165384275) (4 条消息):

> - `TFLOPS 性能`
> - `Apple M2 内存模型`
> - `高效缓存策略`

- **难以提升 TFLOPS**：*“我很惊讶这种方法没有获得更高的 flops”*，这表明在利用当前方法实现最佳性能方面存在挑战。
  
  - 另一位成员提到他们成功达到了 **1 TFLOP** 但遇到了问题，暗示这可能更多是*“我个人的问题”*。
- **探索更快的内存访问**：一位用户实验了 **shared memory**，发现它比预期的要慢，并正在考虑使用 **subgroups** 以获得潜在的更快访问。
  
  - 他们对缺乏关于 **Apple M2** 内存模型的清晰文档表示沮丧，特别是关于*如何利用缓存来获得更好性能*的部分。
- **最大化缓存效率**：一位成员建议，性能的提升可能源于**更有效地使用缓存**，并避免在工作组（workgroups）之间进行不必要的重复计算。
  
  - 他们的基准测试比较显示，其他实现产生的**性能与他们自己的相似**。
- **记录内存合并（Memory Coalescing）的困惑**：关于 Apple 硬件上**内存合并**文档的清晰度提出了担忧，同时参考了一篇关于 **NVIDIA 特定术语**的博客文章。
  
  - 缺乏全面的资源似乎阻碍了开发者针对 Apple M2 制定有效的优化策略。

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1307061900500271134) (2 条消息):

> - `Liger-Kernel Bug Fix`
> - `Convergence Test Issues`

- **实现了 flce patching 问题的修复**：提交了一个 Pull Request 以[解决 issue #355](https://github.com/linkedin/Liger-Kernel/pull/385)，处理了在收敛测试中执行 revert 后 flce 未被 patch 的问题。
  
  - 该修复方案包括暂时注释掉 revert patching，并仅针对 **float32** 进行测试。
- **感谢解决了恼人的 reverting 问题**：对一位成员表示感谢，因为他修复了影响开发的复杂且恼人的 reverting 问题。
  
  - 这一认可强调了协作在解决技术挑战中的重要性。

**提到的链接**：[Fix flce not being patched after reverting in convergence test by Tcc0403 · Pull Request #385 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/pull/385)：摘要：解决 #355：1. revert patching 导致 flce 不生效（暂时注释掉 revert patching，仅测试 float32）。该 Bug 的发生是因为我们在定义 model config 字典之前...

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1306852544861179955) (2 条消息):

> - `cudaDeviceSynchronize`
> - `C++ development experience`

- **cudaDeviceSynchronize() 是一种 anti-pattern**：一位用户认为 **cudaDeviceSynchronize()** 是程序员应该避免的一种 anti-pattern，因为它会遍历所有的 CUDA streams 并对它们调用 **cudaStreamSynchronize()**。
  
  - 他们强调这种低效性并没有被清晰地阐述，这可能会导致开发者之间的误解。
- **C++ 开发者寻求免费实习经验**：一位拥有 **2 年**经验的 **C++ 开发者**介绍了自己，表达了希望通过免费工作来获得更多实践经验的愿望。
  
  - 这体现了在开发社区中提升技能和建立人脉的积极态度。

**提到的链接**：[Daniel Galvez (@memorypaladin) 的推文](https://x.com/memorypaladin/status/1856954308056744110)：cudaDeviceSynchronize() 是人们应该避免的一种 anti-pattern。虽然没有明确说明，但它基本上会遍历 CUDA context 中的所有 CUDA streams 并调用 cudaStreamSynchronize() ...

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1306755191693115462) (2 条消息):

> - `Discord Cluster Manager Issue`
> - `Finetuning Loop`
> - `Trion Docs and Examples`

- **受 Quantization Scaling 启发**：一位成员分享了来自 [GitHub Issue #23](https://github.com/gpu-mode/discord-cluster-manager/issues/23) 关于 quantization scaling 的深刻见解，称赞 @TimDettmers 的讨论串非常有启发性。
  
  - 他们强调了“关于如何做研究的研究”以及在该领域普及方法和结果的重要性。
- **在 Finetuning Loop 中挣扎**：一位成员目前正在使用 Discord 中分享的初始数据开发 Finetuning Loop，但结果不尽如人意，因为很少有内容能编译为 Python bytecode。
  
  - 他们对协作和帮助持开放态度，邀请其他人尝试该设置并回答任何问题。
- **使用 Triton 文档进行评估**：在评估（Evals）方面，该成员目前依赖于 Triton 文档中的简单示例和 puzzles，并计划在基础设施建立后扩大工作规模。
  
  - 他们指出，在评估方面取得重大进展之前，重点是确保基础设施的正确性。

**提到的链接**：[Job Queue · Issue #23 · gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager/issues/23)：我最近阅读了 @TimDettmers 关于 quantization scaling 的讨论串。非常有启发性。我认为这种研究，即关于如何做研究的研究，以及普及这些方法和结果，是...

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1306731430054858813) (11 messages🔥):

> - `Kernel Compilation Delays` (Kernel 编译延迟)
> - `Complexity Threshold in Code` (代码中的复杂度阈值)
> - `Template Parameter Impact` (模板参数的影响)
> - `Effective Debugging Strategies` (有效的调试策略)

- **观察到奇怪的 Kernel 编译延迟**：一名成员报告称，他们的 Kernel 在超过一定的**复杂度阈值**后，尽管能编译成功，但编译时间竟然超过了 **30 分钟**。
  
  - 他们发现通过降低复杂度（例如有选择地注释掉代码块），可以将编译时间缩短回 **~5 秒**。
- **调整模板参数可降低复杂度**：另一位成员建议，调整模板和 `constexpr` 参数（特别是更改循环次数）有助于减少编译时间。
  
  - 例如，在包围主函数体的循环中设置 `NC=1`，可以显著加快编译速度。
- **寻求编译延迟的诊断方法**：一名成员请求帮助诊断异常漫长的 Kernel 编译时间，并分享了在此过程中面临的挑战。
  
  - 他们表示不确定该如何着手识别导致该问题的具体原因。
- **模板参数推导在延迟中的作用**：有人指出，缓慢的模板参数推导（Template Parameter Inference）会导致编译时间变长，尤其是在循环展开（loop unrolling）期间。
  
  - 他们建议隔离有问题的代码行，并使模板参数显式化（explicit），以提高性能。

---

### **GPU MODE ▷ #**[**edge**](https://discord.com/channels/1189498204333543425/1303441437592911912/1306746857250361375) (5 messages):

> - `React Native LLM Library` (React Native LLM 库)
> - `LLM Inference on Android` (Android 上的 LLM 推理)
> - `Memory Bound vs Compute Bound in LLMs` (LLM 中的内存受限与计算受限)
> - `Bitnet for Fast Inference` (用于快速推理的 Bitnet)
> - `GGUF Q8 Performance` (GGUF Q8 性能)

- **用于 LLM 的 React Native 库亮相**：Software Mansion 发布了一个在 React Native 中使用 LLM 的新库，该库利用 **ExecuTorch** 进行后端处理。安装说明可在其 [GitHub 页面](https://github.com/software-mansion/react-native-executorch)上找到。
  
  - *用户发现它非常易于使用*，并有清晰的步骤在 iOS 模拟器上启动模型。
- **Android 智能手机上的 LLM 推理 - 内存问题**：讨论提出了**新型 Android 智能手机上的 LLM 推理**是否受内存限制（Memory Bound）的问题。回复强调，对*上下文类型*的依赖决定了 LLM 主要是内存受限还是计算受限（Compute Bound）。
  
  - *有人指出，通常情况下，低上下文任务需考虑内存受限，而高上下文任务则伴随计算受限。*
- **用于快速推理的 Bitnet 1.58 A4**：为了实现更快的推理，使用 **Bitnet 1.58 A4** 配合 Microsoft 的 T-MAC 操作可以在 7B 模型上达到 **10 tokens/s**，尽管这需要对目标模型进行一些重新调整。重要的是，如果缺乏 GPU 资源，它可以在桌面 CPU 上进行训练。
  
  - 目前已有将模型转换为 **Bitnet** 的资源，尽管这可能需要进行训练后调整。
- **GGUF Q8：一种低成本的性能方案**：**GGUF Q8** 被认为是一个可行的选择，对于资源受限的设备几乎没有性能损失，尤其对 **7B-13B 模型**非常有效。然而，由于设备限制，一些用户尚未测试 GGUF Q8 在更小模型上的优势。
  
  - *有人提到 GGUF Q8 可能不会为* ***3B 及以下模型*** *带来同样的收益。*

**提到的链接**：[GitHub - software-mansion/react-native-executorch](https://github.com/software-mansion/react-native-executorch.git)：通过在 GitHub 上创建账户，为 software-mansion/react-native-executorch 的开发做出贡献。

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1306712141385957436) (17 messages🔥):

> - `Top Shelf Podcast`
> - `AI 与暴力`
> - `播客对话技巧`
> - `通过音频进行 Microlearning`
> - `物理教科书辅助`

- **听众希望从 Top Shelf Podcast 获得更多内容**：听众表示有兴趣扩展 **Top Shelf** 播客并增加更多书籍摘要，特别是请求制作关于 Adam Grant 的 *Think Again* 的单集，并在讨论中捕捉 *The Body Keeps Score* 的精髓。
  
  - 一位用户鼓励其他人分享建议以丰富播客内容，并链接到了他们的 [Spotify 节目](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF)。
- **关于 AI 控制暴力的讨论**：一位用户分享了一个名为“[AI 对暴力的垄断](https://youtu.be/LgU6R26csf0)”的 [YouTube 视频](https://youtu.be/LgU6R26csf0)，强调了对人工超级智能（ASI）管理暴力的影响的担忧。
  
  - 该视频深入探讨了在 AI 技术演进背景下，这种暴力**垄断**可能带来的潜在后果。
- **探索播客对话技巧**：成员们讨论了播客主持人中普遍使用的 *yes...and* 框架，评论其如何营造出对话深度的错觉。
  
  - 有人呼吁探索更多批判性的基调，并讨论音频描述中双主持人对话的动态，以提升听觉体验。
- **针对忙碌人士的 Microlearning 目标**：一位成员分享了他们利用 **Top Shelf** 播客促进 Microlearning 的愿景，旨在解决因时间限制而无法阅读整本书的挑战。
  
  - 他们强调，虽然播客可以辅助学习，但它不能取代阅读整本书的体验。
- **使用 Notebook LM 辅助物理教科书阅读**：一位用户提到在阅读物理教科书时利用 **Notebook LM** 获取定义，并建议以数学符号而非 LaTeX 格式输出回复。
  
  - 这突显了教育工具对更用户友好的格式的需求，以增强理解力。

**提到的链接**：

- [Top Shelf](https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF)：播客 · Four By One Technologies · "Top Shelf" 是您获取当今畅销书籍快速、深刻见解的首选播客。只需 15 分钟，即可获得要点、精华和全新的视角...
- [[NEUROPODCAST] Monopoly on violence for AI](https://youtu.be/LgU6R26csf0)：本集探讨了强大的人工超级智能 (ASI) 获得暴力控制权可能带来的后果，可能导致一种“垄断”...
- [Aging & Nutrient Deficiency: Impact on Immunity](https://youtu.be/KNfM1XZCilk)：#衰老与免疫 #微量元素缺乏 #健康衰老 衰老与免疫、微量元素缺乏、健康衰老、增强免疫力、衰老健康提示...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1306724100617539676) (34 条消息🔥):

> - `Art of Prompting with AI` (AI 提示词艺术)
> - `Issues with NotebookLM` (NotebookLM 的问题)
> - `Audio Summaries and Customization` (音频摘要与自定义)
> - `Uploading Documents` (上传文档)
> - `Exploration of Relevant Links` (相关链接探索)

- **AI 提示词艺术是关键**：讨论强调了掌握 **提示词艺术 (art of prompting)** 对于有效使用付费 AI 工具生成艺术作品的重要性。
  
  - 成员们强调，优秀的提示词可以显著提升 AI 生成内容的质量。
- **NotebookLM 面临技术问题**：几位成员报告了 **NotebookLM 的问题**，包括功能操作故障和某些特性无法访问。
  
  - 用户讨论了他们的挫败感，并在等待开发团队修复期间分享了临时的替代方案。
- **为不同受众定制音频摘要**：一位用户分享了从一组资源中为不同受众创建 **定制化音频摘要** 的经验。
  
  - 该过程涉及为社工和研究生调整内容，展示了 NotebookLM 的灵活性。
- **讨论文档上传限制**：关于文档上传限制的问题被提出，成员们建议通过 **文档分组 (group documents)** 来保持在允许的限制范围内。
  
  - 一些用户询问在 NotebookLM 中上传超过 50 份文档的可行性。
- **寻求提示词技巧的改进**：一位成员询问了有助于 **深入研究学术论文** 的提示词，表明需要更有针对性的策略。
  
  - 回复建议使用专家视角或理论方法来进行更深入的讨论。

**提到的链接**：

- [Healthy Habits, Happy Life: Essential Tips for a Vibrant Lifestyle](https://open.spotify.com/show/569AF0vj1DHXXMPIe5UjEq?si=cee83cbd40174993&nd=1&dlsi=9bfbaa37f3be4659)：播客 · LetPeopleTalk · 加入我们的旅程，迈向更健康、更快乐的自己！我们的频道致力于提供实用技巧、专家建议和励志故事，帮助您过上充实的生活...
- [MarkDownload - Markdown Web Clipper - Chrome Web Store](https://chromewebstore.google.com/detail/markdownload-markdown-web/pcmpcfapbekmbjjkdalcgopdkipoggdi)：这款扩展程序类似于网页剪藏工具，但它以 Markdown 格式下载文章。
- [The $3 Trillion AI Opportunity Everyone Missed](https://chrisbora.substack.com/p/the-3-trillion-ai-opportunity-everyone-f62)：为什么今天的“GPU 泡沫”实际上是严重的投资不足。
- [Notebook LM Tutorial: Customizing Content for Different Audiences](https://youtu.be/ASn7UXAC5PU)：学习如何利用 Notebook LM 强大的自定义功能为不同受众量身定制内容！在本教程中，我们将探索如何合成...

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1306712783332573225) (36 条消息🔥):

> - `Liger Kernel 改进`
> - `Cut Cross-Entropy 方法`
> - `Orca3 见解`
> - `来自 Meta 的活动邀请`
> - `Tokenization 与 Fine-Tuning`

- **Liger Kernel 展示了性能提升**：团队强调了 **Liger Kernel** 的增强功能，新的更新提高了**速度**和**显存效率** [查看详情](https://github.com/linkedin/Liger-Kernel/pull/362)。
  
  - 具体改进还包括对 **DPO** 的支持以及对 **Gemma2** 的更改，使其能够利用 **fusedcrossentropy**。
- **Cut Cross-Entropy 方法提出重大效率提升**：一项关于 **Cut Cross-Entropy (CCE)** 的提案建议在计算 Loss 时显著减少显存占用，对于 Gemma 2 模型，可从 **24 GB 降至 1 MB** [详情点击此处](https://github.com/apple/ml-cross-entropy)。
  
  - 据报道，该方法在保持相似显存占用的情况下，训练速度比 Liger 快约 **3 倍**。
- **关于 Orca3 的见解不断涌现**：**Orca3** 获得了积极反馈，成员们指出其表现尤为出色。
  
  - 与 Orca3 相关的讨论引用了其他频道，强调了其在模型架构（Model Architecture）方面的潜在创新。
- **Meta 神秘的 Llama 活动邀请**：一位成员收到了来自 Meta 的意外邀请，参加在其总部举行的关于开源和 Llama 的为期两天的活动，引发了对可能发布**新模型**的好奇。
  
  - 成员们推测了活动的重点，并强调了在非演讲者身份下受邀参加此类行程的奇特性。
- **Fine-Tuning 中的 Token 重叠**：在关于 **Fine-Tuning** 的讨论中，有人质疑如果 Tokenizer 保持不变，**5k 数据**是否足够，一位成员建议数百万个 Token 可能更合适。
  
  - 与此同时，大家对模型训练过程的连续性以及检查 Token 重叠的必要性进行了反思。

 

**提到的链接**：[来自 Kearm (@Nottlespike) 的推文](https://x.com/Nottlespike/status/1857181970746466769)：这就是我今天的情况

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**datasets**](https://discord.com/channels/1104757954588196865/1112023441386778704/1306996095536205904) (1 条消息):

> - `Orca`
> - `Orca 2`
> - `Orca-AgentInstruct`
> - `Synthetic Data Generation`

- **Orca 展示了合成数据的实力**：关于 [Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/) 的研究揭示了其利用合成数据进行小语言模型（Small Language Models） **Post-training** 的能力，实现了与大型模型相似的性能水平。
  
  - 这种方法标志着语言模型训练的显著进步，在 [Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/) 和 [Orca 2](https://www.microsoft.com/en-us/research/blog/orca-2-teaching-small-language-models-how-to-reason/) 中均得到了验证。
- **Orca-AgentInstruct 增强了合成数据生成**：[Orca-AgentInstruct](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/) 探索了 **Agentic Flows**，以大规模生成多样化且高质量的合成数据。
  
  - 通过使用这种 **Agentic** 框架，它可以创建包含 **Prompt** 和 **Response** 的定制数据集，从而提高数据生成的整体效率。

 

**提到的链接**：[Orca-AgentInstruct: Agentic flows can be effective synthetic-data generators](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/)：来自 Microsoft Research 的 Orca-AgentInstruct 可以大规模生成多样化、高质量的合成数据，用于基础 **LLM** 的 **Post-train** 和 **Fine-tune**，以扩展能力、持续学习并增加...

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**announcements**](https://discord.com/channels/1104757954588196865/1113462842436354149/1307014765348126771) (1 messages):

> - `Office Hours`
> - `Axolotl feedback session`

- **加入 Axolotl 的首次 Office Hours！**：我们非常激动地宣布，将于 **12 月 5 日东部时间下午 1 点**在 Discord 举办首次 **Office Hours** 会议，欢迎所有成员提问并分享反馈。
  
  - *这是你咨询任何关于 Axolotl 的问题并贡献想法的机会！*
- **Axolotl 的反馈机会**：本次会议公开邀请大家提出想法和建议，以帮助改进 Axolotl。
  
  - Axolotl 团队渴望倾听并与社区互动，以优化平台。

 

**提到的链接**：[Discord - Group Chat That’s All Fun & Games](https://discordapp.com/channels/1104757954588196865/1268285745555308649)：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-help-bot**](https://discord.com/channels/1104757954588196865/1225300056442409040/1307089440707186769) (9 messages🔥):

> - `Qwen/Qwen2 Model Pretraining`
> - `Eval Steps in Training`
> - `Phorm Bot Malfunction`

- **Qwen/Qwen2 模型预训练步骤**：一位成员正在寻求指导，希望在设置好 **Axolotl docker** 后，如何使用 **qlora** 和他们的原始文本 jsonl 数据集预训练 **Qwen/Qwen2** 模型，然后使用 instruct 数据集进行微调。
  
  - 他们询问了继续进行设置的具体后续步骤。
- **关于 eval_steps 的咨询**：另一位成员询问：*“eval_steps 是什么意思？”*，寻求关于 eval_steps 在训练过程中的重要性的澄清。
  
  - 然而，随后的消息中没有提供明确的答案。
- **Phorm Bot 的响应问题**：一位成员报告说 **Phorm bot** 似乎出现了故障，称其甚至无法回答基本查询。
  
  - 这凸显了社区内可能需要解决的技术问题。

 

**提到的链接**：[OpenAccess-AI-Collective/axolotl | Phorm AI Code Search](https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=undefined))：更快地理解代码。

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1306720758482866258) (25 messages🔥):

> - `P100 vs P40 Scam`
> - `Photo Selection Techniques`
> - `PDF Upload Issues`
> - `GPT-4o Pricing Mechanics`
> - `AI Club Development Plans`

- **收到 P100 而非 P40 的诈骗**：一位成员报告被深圳的一位卖家诈骗，订购了 **P40** 却收到了 **P100**，并对大科技公司纵容此类欺诈行为表示沮丧。
  
  - 他们评论道：*“我绝不会把这玩意儿插上去，”* 表达了对收到的产品的不信任。
- **创新的照片选择策略**：一位成员讨论了从数百张照片中选出最佳照片的挑战，思考了创建一个编号拼图（collage）作为简化流程方法的效率。
  
  - 尽管他们认为拼图技术有些“粗糙”，但承认其有效性，特别是在一次只处理一个任务时。
- **持续的 PDF 上传困难**：一位用户对在不同平台（包括 Mac App、iOS App 和网页浏览器）上传 PDF 的问题表示沮丧。
  
  - 这引发了关于目前各种 OpenAI 系统稳定性的讨论。
- **探索 GPT-4o 计费机制**：讨论指出，**GPT-4o** 在高分辨率模式下处理每个 `512x512` 的切片（tile）收费 **170 tokens**，这证实了一张图片实际上相当于约 **227 个单词**。
  
  - 成员们思考了这个异常具体的 token 收费背后的意义，并将其与编程中的“魔法数字（magic numbers）”概念联系起来。
- **AI 俱乐部的发展计划**：一位成员分享了最近在 **UIUC** 举行的 AI 俱乐部会议的更新，指出正在为未来的活动和下学期的重大发布进行规划。
  
  - 这在社区内引起了兴奋，强调了未来 AI 事业的协作方式。

**提到的链接**：

- [来自 Doomlaser Corporation (@DOOMLASERCORP) 的推文](https://x.com/DOOMLASERCORP/status/1857463705195151398)：#UIUC 首届 AI 俱乐部会议之后。我们中的 4 个人。两周后我们将再次在这里见面，为下学期的重大发布做准备 👄👄🫦👁️👀🧠🫀🫁🦴, #AI #Future
- [一图值 170 Tokens：GPT-4o 如何编码图像？ - OranLooney.com](https://www.oranlooney.com/post/gpt-cnn/)：事实如下：GPT-4o 对高分辨率模式下使用的每个 512x512 切片收费 170 tokens。按照约 0.75 tokens/单词计算，这意味着一张图片大约相当于 227 个单词——仅为...的四倍。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1306713745678012457) (6 messages):

> - `Content Flags` (内容标记)
> - `GPT Issues` (GPT 问题)
> - `User Input Policies` (用户输入政策)

- **内容标记并非主要问题**：*一位成员指出*，虽然多年来收到了许多内容标记，但这些通常是关于模型输出的，有助于改进其训练，而非表明用户违规。
  
  - 他们强调，即使由于用户没有明确请求违禁内容，也可能出现内容标记。
- **关于允许内容的澄清**：*另一位成员对*内容未违反政策（特别是讨论恐怖视频游戏时）却被标记表示沮丧，认为这并不具有伤害性。
  
  - 他们提到最近收到了很多标记，暗示内容监控可能有所加强。
- **寻求 GPT 故障的解决方案**：*一位用户寻求*帮助，询问他人修复 GPT 问题的方案。
  
  - 该询问得到了轻松的回应，说明了用户面临的持续挑战。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1306719390523330673) (5 messages):

> - `Game of 24 AI performance` (24点游戏 AI 表现)
> - `Exploring past experiences` (探索过去经验)
> - `RAG prompt and few-shot examples` (RAG Prompt 与 Few-shot 示例)

- **3.5 AI 在 24点游戏中表现出色**：一位用户分享道，**3.5 AI** 在 24点游戏中通常不会撒谎，使其偶尔能够获胜。
  
  - 这表明 AI 在游戏场景中的性能有了显著提升。
- **对过去游戏时光的怀念**：另一位用户回忆起过去的美好时光，表示有兴趣再次探索 **Game of 24**。
  
  - *“我敢打赌，如果我们想的话，仍然可以探索那个！”* 强调了重新审视这款游戏的共同热情。
- **RAG Prompt 增强咨询**：一位用户正在寻求澄清，即在 **RAG Prompt** 中加入文档示例是否能提高回答质量。
  
  - 他们强调需要对此进行研究，特别是为了开发 **QA Agent 平台**。
- **用于增强推理的提示词 (Hint prompt)**：一位用户建议为 AI 回复设置提示词，规定如果 AI 感到不自信，应该要求更多时间。
  
  - 这种提示词允许 AI 优化其答案，并提出了一种提高响应准确性的方法。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1306719390523330673) (5 messages):

> - `Game of 24` (24点游戏)
> - `Exploring Past AI Iterations` (探索过去的 AI 迭代)
> - `Prompting Techniques for AI` (AI 提示技术)
> - `RAG Prompt with Few-Shot Examples` (带有 Few-Shot 示例的 RAG Prompt)

- **24点游戏技能提升**：一位用户指出，他们的 **3.5 模型** 能够玩 24点游戏，甚至偶尔获胜。
  
  - 这突显了 AI 在游戏过程中的能力提升，反映了积极的趋势。
- **对 AI 探索的怀旧反思**：一位成员回忆起早期的探索，建议他们仍然可以一起重新审视这些想法。
  
  - 这种情绪表明社区对探索 AI 过去的发展和迭代有着浓厚的兴趣。
- **通过示例增强 RAG Prompt**：一位用户询问在 **RAG Prompt** 中添加文档示例是否会提升回答质量。
  
  - 他们强调正在开发 **QA Agent 平台**，并需要关于加入 **Few-shot 示例** 的反馈。
- **带有置信度提示的提示策略**：有人建议使用提示词，鼓励玩家在需要更多思考时间时坦白承认。
  
  - 这种方法旨在减轻解决问题时的压力，促进更深思熟虑的参与。

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1306722907795361873) (17 messages🔥):

> - `OpenAI's 'OPERATOR' AI Agent`
> - `Beta App Performance`
> - `Azure AI Search Methodologies`
> - `Open Interpreter Shell Integration`
> - `Devin AI Preview Access`

- **OpenAI 揭秘 'OPERATOR' AI Agent**：一段名为 ["OpenAI Reveals 'OPERATOR' The Ultimate AI Agent Smarter Than Any Chatbot"](https://www.youtube.com/watch?v=YRn9xzTBt20) 的 YouTube 视频讨论了 OpenAI 即将推出的新 AI Agent，预计很快将面向更广泛的用户群体。
  
  - *它即将面向大众！*
- **Beta App 性能优于控制台集成**：关于 Beta App 性能的确认请求得到了回应，一名成员表示，由于拥有更好的基础设施，**桌面端应用 (desktop app)** 将提供最佳的 Interpreter 体验。
  
  - 他们强调，相比开源仓库，它拥有更多的幕后支持。
- **Azure AI Search 详解**：一段名为 ["How Azure AI Search powers RAG in ChatGPT and global scale apps"](https://youtu.be/NVp9jiMDdXc?feature=shared) 的 YouTube 视频概述了 Azure AI Search 中涉及的数据转换和质量恢复技术。
  
  - 它引发了关于专利、资源以及对有效数据删除流程需求的问题。
- **Open Interpreter Shell 集成发布**：推出了一项新的 **Open Interpreter shell 集成**，允许用户在终端内安装并访问它，以增强交互性。
  
  - 这是一个实验性功能，旨在将任何终端转变为 Open Interpreter 的聊天框，欢迎用户提供反馈。
- **对 Devin AI 的兴趣**：有人询问是否有人拥有 **Devin AI** 的预览访问权限，这引发了一些对其价值的怀疑。
  
  - 一名成员表达了怀疑，称他们认为*它很平庸 (it's lame)*。

**提到的链接**：

- [OpenAI Reveals "OPERATOR" The Ultimate AI Agent Smarter Than Any Chatbot](https://www.youtube.com/watch?v=YRn9xzTBt20)：👉 免费注册 ChatGPT & AI 工作坊：https://web.growthschool.io/ARO 👉 前 1000 人可享 100% 折扣 🔥✅ 加入 Top1% AI 社区获取定期更新...
- [How Azure AI Search powers RAG in ChatGPT and global scale apps](https://youtu.be/NVp9jiMDdXc?feature=shared)：数百万人每天在不知情的情况下使用 Azure AI Search。你可以让你的应用具备与实现检索增强生成 (RAG) 相同的搜索能力...
- [Microsoft Mechanics - Azure AI Search at scale](https://gist.github.com/pablocastro/393e2be08d4581c918dc59a944995fb6)：Microsoft Mechanics - 大规模 Azure AI Search。GitHub Gist：即时分享代码、笔记和代码片段。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1306746816372539422) (4 messages):

> - `Probabilistic computing`
> - `ChatGPT desktop compatibility`
> - `New computing performance`

- **概率计算 (Probabilistic computing) 实现 GPU 性能大幅提升**：一段 [YouTube 视频](https://www.youtube.com/watch?v=hJUHrrihzOQ) 讨论了 **概率计算** 的新突破，据报道，与顶尖的 NVIDIA GPU 相比，其**能效提高了 1 亿倍**。
  
  - *在这段视频中，我讨论了概率计算，据报道，与顶尖的 NVIDIA GPU 相比，它能实现 1 亿倍的能效提升。*
- **ChatGPT 桌面端获得用户友好型增强**：这一进步被标记为对 ChatGPT 桌面端**大众用户的重大利好**，预示着能显著提升用户体验的改进。
  
  - 用户非常关注那些能增强他们与平台交互的功能。
- **兼容多个供应商**：一名成员指出，该平台兼容多个**供应商，包括 OpenAI, Anthropic, Google, AWS Bedrock 和 Replicate**。
  
  - 他们目前正在开发**自定义 URL**，这表明改进工作正在进行中。

 

**提到的链接**：[New Computing Breakthrough achieves 100 MILLION Times GPU Performance!](https://www.youtube.com/watch?v=hJUHrrihzOQ)：在这段视频中，我讨论了概率计算，据报道，与顶尖的 NVIDIA GPU 相比，它能实现 1 亿倍的能效提升。查看...

 

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1306752144845701172) (12 messages🔥):

> - `tinybox pro 预订`
> - `int64 索引悬赏`
> - `buffer 传输函数`
> - `开发中的社区支持`

- **tinybox pro 开启预订**：**tinybox pro** 现在已在 [tinygrad 网站](https://tinygrad.org/#tinybox) 开启预订，售价为 **$40,000**，配备 8 张 RTX 4090。
  
  - 凭借其达到 **1.36 PetaFLOPS** FP16 计算能力的惊人性能，它被定位为单张 Nvidia H100 GPU 的更实惠替代方案。
- **寻求 int64 索引悬赏的指导**：一名成员询问 **int64 索引** 悬赏包含哪些内容，特别是是否需要修改 tensor.py 中的 **getitem** 等函数。
  
  - 另一名成员指出，他们可以搜索已提交的 PR，并重点提到了 [PR #7601](https://github.com/tinygrad/tinygrad/pull/7601/files#diff-00bd44b667ec90ae1d3e984e699bc6b498c84ca1b1bd15a025437ded227457bf)，该 PR 针对此悬赏进行了处理，但尚未被采纳。
- **社区讨论悬赏背景**：一名成员建议，通过 Discord 搜索可以提供关于现有悬赏的有价值背景讨论。
  
  - 对于任何想要了解 **int64 索引** 及其他悬赏详情的人来说，这个资源通常非常有帮助。
- **分享 buffer 传输函数细节**：一个 pull request 被重点提及，其中讨论了 CLOUD 设备上的 **buffer 传输函数**，这对设备互操作性非常有用。
  
  - 交流中提到，关于目标 buffer 是否需要进行大小检查（size check）存在一些歧义。

**提及的链接**：

- [AI 加速器 tinybox pro 以 $40,000 开启预订 —— 该设备配备 8 张 RTX 4090 和两颗 AMD Genoa EPYC 处理器](https://www.tomshardware.com/tech-industry/artificial-intelligence/ai-accelerator-tinybox-pro-goes-up-for-preorder-for-usd40-000-the-device-features-eight-rtx-4090s-and-two-amd-genoa-epyc-processors)：一款更强大但依然实惠的 AI 加速器。
- [mdaiter 提交的 CLOUD 设备上的 Buffer 传输 · Pull Request #7705 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7705/files)：标题说明了一切 —— 从一个设备读取 buffer，并将其放入另一个不同设备中。你其实不需要 assert 或 sz 参数，但我希望保持一致性...
- [ttomsa 提交的更好的 int64 索引 · Pull Request #7601 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7601/files#diff-00bd44b667ec90ae1d3e984e699bc6b498c84ca1b1bd15a025437ded227457bf)：未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1306828288198574101) (2 messages):

> - `tinygrad 贡献`
> - `CLOUD 设备上的 Buffer 传输`
> - `GPU Tensor 传输`

- **贡献 tinygrad：寻求反馈**：一位用户分享了他们第一次尝试为 **tinygrad** 做贡献的经历，并表示愿意接受所有反馈以求改进。
  
  - 他们引用了一个 [GitHub pull request](https://github.com/tinygrad/tinygrad/pull/7709)，其中讨论了他们在数据传输方面的工作。
- **CLOUD 设备上的 Buffer 传输 Pull Request**：该用户重新开启了一个专注于管理 CLOUD 设备上 **buffer 传输** 的 pull request，并引用了之前收到的相关反馈。
  
  - 他们反思性地评论道，事后看来，他们上次在 web buffer 之间读取和转储数据的方法似乎很**愚蠢**。
- **在 GPU 之间原生传输 Tensor**：针对最初的询问，有人澄清说，可以使用 `.to` 函数在同一设备上的不同 **GPU** 之间传输 tensor。
  
  - 这一澄清旨在引导用户使用更有效的方法来处理 tensor 传输。

 

**提及的链接**：[mdaiter 提交的 CLOUD 设备上的 Buffer 传输 · Pull Request #7709 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7709)：根据一些反馈，重新开启此 PR。我上一个 PR 专注于同步从 web buffer 读取数据到 host 设备，然后将其转储到另一个 web buffer。事后看来似乎很愚蠢...

 

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1307059179239903306) (2 messages):

> - `LlamaIndex Community Call`
> - `Python Documentation Upgrade`
> - `Knowledge Graphs`
> - `RAG system`

- **在 Community Call 中学习 GenAI 应用构建**：参加我们即将举行的 [Community Call](https://twitter.com/llama_index/status/1857500067357405398)，学习如何从非结构化数据创建 **Knowledge Graphs** 以及高级检索方法。
  
  - 探索如何将**数据转换**为可查询的格式！
- **RunLLM 助力 Python 文档功能提升**：我们的 Python 文档通过全新的 **'Ask AI'** 组件获得了升级，该组件可以启动一个针对代码查询的高精度 Agentic **RAG 系统** [去看看吧！](https://twitter.com/llama_index/status/1857536223566508061)。
  
  - 用户现在可以直接获得针对其问题编写的**准确且最新的代码**。

 

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1307008984443519066) (10 messages🔥):

> - `condenseQuestionChatEngine`
> - `CondensePlusContext`
> - `retrieving context`
> - `customizing prompts`

- **condenseQuestionChatEngine 的问题**：一位成员提出，当用户突然切换话题时，**condenseQuestionChatEngine** 有时会生成不连贯的独立问题。
  
  - 建议通过自定义 Condense Prompt 来更好地处理话题突变。
- **更倾向于 CondensePlusContext**：另一位成员表示更倾向于使用 **CondensePlusContext**，因为它会压缩输入并为每条用户消息检索上下文。
  
  - 他们强调了其在提供动态上下文方面的高效性，包括将检索到的文本插入到 System Prompt 中。
- **关于上下文检索的澄清**：关于 **CondensePlusContext** 是为每条用户消息还是仅为最新一条消息检索上下文存在困惑，最终达成的共识是它针对的是最新的用户消息。
  
  - 一位成员澄清该功能确实使用了一个 Retriever 来进行上下文检索。
- **使用自定义查询引擎处理查询**：一位成员详细介绍了他们自定义查询引擎的实现，强调了在没有查询引擎参数的情况下使用 **CondensePlusContext** 的挑战。
  
  - 他们提供了代码片段来展示其涉及带有各种 Postprocessors 的 **RetrieverQueryEngine** 的设置。
- **在 CondensePlusContext 中使用自定义 Retriever**：成员们一致认为，针对他们的情况，合适的方法是使用带有自定义 Retriever 的 **CondensePlusContextChatEngine**。
  
  - 他们建议使用自定义 Retriever 和 Node Postprocessors 以符合其特定需求。

 

**提到的链接**：[Postgres Vector Store - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/postgres/)：未找到描述

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1306877923726458953) (6 messages):

> - `New Users on Cohere`
> - `Issues with Model Settings`

- **欢迎 Cohere 的新用户**：多位成员欢迎新成员 **fotis** 和 **lovisonghamnongtimem** 加入 Cohere 社区，鼓励他们享受这里的体验。
  
  - “享受这段旅程并分享乐趣！”是现有成员表达的共同情感。
- **模型设置的持续问题**：一位成员强调了他们在 **command-r-plus-08-2024** 模型设置中遇到的持久问题，并分享了包括 Temperature 和 Frequency Penalty 在内的具体参数。
  
  - 尽管进行了指定的配置，但他们仍不断遇到问题，正在寻求帮助。

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1306875546462060585) (3 messages):

> - `Playwright Python file upload`
> - `Cohere discussion`

- **在 Python 中使用 Playwright 上传文件**：一位用户分享了他们在 Playwright Python 中使用 `set_input_files` 方法上传文本文件并查询上传内容的方法。
  
  - 然而，他们表示当询问“你能总结文件中的文本吗？@file2upload.txt”时，这种方法感觉有点奇怪。
- **对 Cohere 相关性的质疑**：一位用户提出了关于该讨论与 **Cohere** 相关性的问题，表示对该话题的背景感到不确定。
  
  - 在给定的消息记录中，该询问尚未得到答复。

 

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1307026786621591703) (1 条消息):

> - `Agentic Chunking`
> - `LlamaChunk Method`
> - `RAG Pipelines`
> - `Regular Expressions Challenges`

- **代理式分块 (Agentic Chunking) 研究发布**：一种针对 **RAG 的代理式分块**新方法已实现**不到 1 秒**的推理时间，证明了其在 GPU 上的高效性且具备成本效益。
  
  - 有关该研究的完整细节和社区建设信息可以在其 [Discord 频道](https://discord.com/invite)中找到。
- **LlamaChunk 简化文本处理**：介绍 **LlamaChunk**，这是一种由 LLM 驱动的技术，通过对文档仅进行单次 **LLM 推理**，优化了递归字符文本分割。
  
  - 该方法消除了标准分块算法中通常使用的脆弱正则表达式 (regex) 模式，从而能够更好地处理非结构化数据。
- **编写分块正则表达式的挑战**：强调的一个主要痛点是为文档分块编写 **regex** 模式的难度，这些模式可能会失败并产生过大的分块。
  
  - 常见方法包括每 **1000 个字符**或根据空格进行分割，但这些解决方案通常缺乏效率和灵活性。
- **开源贡献邀请**：团队鼓励对 **LlamaChunk** 代码库进行贡献，该代码库已在 [GitHub](https://github.com/ZeroEntropy-AI/llama-chunk) 上公开供大家使用。
  
  - 用户可以查阅 README 以获取该方法运行机制的详细说明。

**提到的链接**：

- [LlamaChunk: Better RAG Chunking Than LlamaIndex | Hacker News](https://news.ycombinator.com/item?id=42148487)：未找到描述
- [GitHub - ZeroEntropy-AI/llama-chunk](https://github.com/ZeroEntropy-AI/llama-chunk)：通过在 GitHub 上创建账号，为 ZeroEntropy-AI/llama-chunk 的开发做出贡献。

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1306713949009481799) (6 条消息):

> - `Copyright infringement debate`
> - `Public link index discussion`

- **关于版权和公开链接的困惑**：一位成员断言，“*在任何情况下，公开链接的公开索引都不构成版权侵权*”。
  
  - 这一声明引发了关于使用公开链接合法性的困惑。
- **关于 Discord 礼仪的一般指导**：另一位成员用简单的“*ty*”表达了感谢，展示了对所获帮助的感激。
  
  - 这种交流表明了社区内持续的协作支持。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1306739627582947369) (1 条消息):

> - `ChatGPT for macOS`
> - `Integration with desktop apps`
> - `dspy GPTs functionality`

- **ChatGPT for macOS 与编程应用集成**：令人兴奋的消息！**ChatGPT for macOS** 现在可以在这项针对 Plus 和 Team 用户的 Beta 功能中，与 **VS Code**、**Xcode**、**Terminal** 和 **iTerm2** 等桌面应用集成。
  
  - 这一增强功能允许 ChatGPT 通过直接与开发环境交互来提供更好的编程辅助，这对于项目来说可能是一个**游戏规则改变者**。
- **对 dspy GPTs 集成的期待**：成员们强烈希望将此功能扩展到 **dspy GPTs**，从而显著增强工作流。
  
  - 成员们讨论了这些集成对其项目的潜在影响，强调了改进的**可能性**。

**提到的链接**：[来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1857129790312272179?t=l7rfG-jT3etXxH9ZrEXPPQ&s=19)：ChatGPT 🤝 VS Code, Xcode, Terminal, iTerm2。ChatGPT for macOS 现在可以与桌面应用协同工作。在面向 Plus 和 Team 用户的早期 Beta 版中，你可以让 ChatGPT 查看编程应用以提供……

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1306763290822705223) (4 条消息):

> - `违规行为的 LLM 文档生成`
> - `DSPy 语言兼容性`

- **扩展 LLM 应用以处理各种违规行为**：一位用户正在开发一个 LLM 应用，用于生成长篇法律文档，为面临吊销驾照风险的驾驶员辩护。目前该应用仅限于**酒精摄入**违规。
  
  - 他们正在寻求一种方法，来创建一个单一的优化 Prompt，使其能够处理各种类型的违规行为，而无需为每种行为单独定制 Prompt。
- **DSPy 语言支持查询**：一位用户询问了 **DSPy** 在非英语语言应用中的语言兼容性。
  
  - 随后的回复指向了一个开放的 [GitHub issue](https://github.com/stanfordnlp/dspy/issues/1803)，该 issue 讨论了 DSPy 的本地化功能请求。

**提到的链接**：[Feature request: localization · Issue #1803 · stanfordnlp/dspy](https://github.com/stanfordnlp/dspy/issues/1803)：你好！又是我 :-)。在我进一步尝试 DSPy 的一些基础功能时，我意识到一件事：目前没有选项可以设置除英语以外的其他语言。问题是，我想确保 LM...

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1306963113136095285) (5 条消息):

> - `新 ABI 研究`
> - `ALLVM 项目`
> - `跨语言 LTO`
> - `Mojo 的 ABI 潜力`

- **研究强调新 ABI**：成员们分享了关于**新 ABI** 的研究论文链接，强调了低级 ABI 在促进跨模块优化方面面临的挑战。*有人指出*，为了获得最高执行速度，通常首选使用单一语言编写所有内容。
  
  - 包含的论文可通过 [DOI 链接](https://doi.org/10.1145/3689755)和 [PDF 链接](https://www.andrewwagner.io/assets/papers/all-bin-tog.pdf)获取。
- **ALLVM 项目的衰落**：讨论指出，**ALLVM 项目**可能因为许多设备缺乏足够的内存来编译/链接所有运行中的软件（尤其是在浏览器中）而受挫。*另一位成员建议*，Mojo 可以以创新的方式利用 ALLVM 进行 C/C++ 绑定。
  
  - 参与者指出，ALLVM 的初衷是统一软件表示，但现在看来大部分已处于停滞状态。
- **对跨语言 LTO 的需求**：一位成员表达了**跨语言 LTO** 的重要性，特别是对于大量人们不愿重写的 C/C++ 软件。他们强调，由于现有软件生态系统的复杂性，这种需求是必要的。
  
  - *讨论公认*，有效的链接将大大提高遗留系统的性能和可维护性。
- **Mojo 潜在的 ABI 创新**：讨论转向了 **Mojo 的潜力**，即定义一种 ABI，通过在寄存器中传递最大信息量来优化数据传输，并利用为 AVX-512 定制的结构大小。*这种方法旨在*增强各种软件组件之间的互操作性和效率。
  
  - 人们希望一个以寄存器效率为核心的 ABI 框架能够改变 C/C++ 在现代环境中的集成方式。

**提到的链接**：[ALLVM Research Project | LLVM All the Things! - University of Illinois at Urbana-Champaign](https://publish.illinois.edu/allvm-project/)：未找到描述

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1307082725009920062) (1 messages):

> - `Intel Tiber AI Cloud`
> - `Intel Liftoff Program`
> - `AMA with Intel`
> - `AI development tools`
> - `Hackathon opportunities`

- **不要错过关于 AI 工具的 Intel AMA**：参加于 **11/21 下午 3 点 PT** 举行的 Intel 独家 AMA 会议 [Building with Intel: Tiber AI Cloud and Intel Liftoff](https://lu.ma/agents-hackathon-intel)，探索先进的 AI 资源。
  
  - 本次活动提供了一个与 Intel 专家互动并深入了解如何优化您的 AI 开发项目的独特机会。
- **探索 Intel Tiber AI Cloud**：Intel 将介绍 **Tiber AI Cloud**，这是一个旨在通过先进的计算能力和效率来增强您的 Hackathon 项目的平台。
  
  - 参与者可以学习如何利用这一强大的工具在项目中获得更好的性能。
- **Intel Liftoff Program 权益详解**：会议还将涵盖 **Intel Liftoff Program**，该计划为初创公司提供技术资源和导师指导。
  
  - 了解可以帮助初创公司在 AI 行业扩展并取得成功的全面权益。

 

**提到的链接**：[Building with Intel: Tiber AI Cloud and Intel Liftoff · Luma](https://lu.ma/agents-hackathon-intel): Building with Intel: Tiber AI Cloud and Intel Liftoff 关于 AMA 加入我们的独家 AMA 会议，届时将有来自我们尊敬的赞助商 Intel 的专家出席……

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1306726730546942002) (3 messages):

> - `Quizzes Feedback`
> - `Course Deadlines`

- **测验反馈延迟**：一位成员表示在尝试补进度时，没有收到 **测验 5 和 6** 的邮件反馈。
  
  - 另一位成员建议他们检查是否是自己端的问题，并建议通过 *重新提交* 来解决问题。
- **课程截止日期的紧急提醒**：一位成员提醒同伴，他们仍然 **有资格** 参加，但由于每个测验都与课程内容挂钩，应 *尽快赶上进度*。
  
  - 最终提交日期为 **12 月 12 日**，强调了按时完成的必要性。

 

---

### **Torchtune ▷ #**[**announcements**](https://discord.com/channels/1216353675241590815/1216353675241590818/1307012901504024667) (1 messages):

> - `Torchtune v0.4.0`
> - `Activation Offloading`
> - `Qwen2.5 Support`
> - `Multimodal Training Enhancement`

- **Torchtune v0.4.0 发布了！**：Torchtune 正式发布了 **v0.4.0**，包含 **大量** 新功能，有望显著提升用户体验。
  
  - 社区的支持在本次发布中起到了至关重要的作用，完整的发布说明可在 [此处](https://github.com/pytorch/torchtune/releases/tag/v0.4.0) 查看。
- **Activation Offloading 提升性能**：**Activation Offloading** 现已在全量微调和 LoRA 方案中实现，可将 **所有文本模型** 的内存需求额外降低 **20%！**
  
  - 该功能旨在为处理文本模型的用户优化整体性能和效率。
- **新增对 Qwen2.5 构建器的支持**：Torchtune 已添加 **Qwen2.5**（Qwen 模型家族的新版本）的构建器。
  
  - 开发者可以在 [Qwen2.5 博客](https://qwenlm.github.io/blog/qwen2.5/) 上找到有关此尖端版本的更多详细信息。
- **多模态训练扩展**：**多模态训练** 功能得到了增强，支持 **Llama3.2V 90B** 和 **QLoRA** 分布式训练。
  
  - 这一扩展使用户能够处理更大的数据集和更复杂的模型，以实现高级训练能力。

 

---

### **Torchtune ▷ #**[**papers**](https://discord.com/channels/1216353675241590815/1293438210097025085/1306996007703281715) (2 条消息):

> - `Orca and Orca 2`
> - `Agentic Solutions for Synthetic Data`
> - `Causal Language Models`

- **Orca-AgentInstruct 致力于合成数据生成**：关于 [Orca](https://www.microsoft.com/en-us/research/publication/orca-progressive-learning-from-complex-explanation-traces-of-gpt-4/) 的最新工作介绍了 **Orca-AgentInstruct**，这是一种用于大规模生成多样化且高质量数据集的 Agentic 解决方案。
  
  - 该方法旨在通过有效的合成数据生成，将小语言模型的性能提升到通常在大型模型中才能看到的水平。
- **倡导在 NLP 领域进行更广泛的理解**：一篇新分享的论文强调了在 Causal Language Models 中避免局部最小值的必要性，并提倡建立更广泛的理解框架。
  
  - 论文主张扩大视角，而不是局限于传统模型的局限性，详见 [预印本](https://arxiv.org/pdf/2406.04823)。

 

**提到的链接**：[Orca-AgentInstruct: Agentic flows can be effective synthetic-data generators](https://www.microsoft.com/en-us/research/blog/orca-agentinstruct-agentic-flows-can-be-effective-synthetic-data-generators/)：来自 Microsoft Research 的 Orca-AgentInstruct 可以大规模生成多样化、高质量的合成数据，用于基础 LLM 的后训练和微调，以扩展能力、持续学习并增加...

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1306737513905524776) (2 条消息):

> - `Local LLM Workshop`
> - `SQLite-Vec Metadata Filtering`
> - `Autonomous AI Agents`
> - `Landing Page Development Assistance`

- **构建你自己的本地 LLM 工作坊**：即将在 **周二** 举行的活动，题为 [Building your own local LLM's: Train, Tune, Eval, RAG all in your Local Env.](https://discord.com/events/1089876418936180786/1300842793945530378)，邀请成员学习本地 LLM 设置的复杂细节。
  
  - 鼓励成员为这一信息丰富的环节进行 RSVP，以增强其本地环境的能力。
- **SQLite-Vec 现已支持元数据过滤**：在 **周三**，成员可以参加一场宣布 [sqlite-vec now supports metadata filtering!](https://discord.com/events/1089876418936180786/1300483739872399411) 增强功能的活动，重点关注实际应用。
  
  - 这是学习如何利用元数据改进数据处理的关键机会。
- **探索自主 AI Agent**：参加 **周四** 关于 [Autonomous AI Agents with Refact.AI](https://discord.com/events/1089876418936180786/1300459081181429810) 的讨论，旨在深入探讨 AI 自动化。
  
  - 该活动承诺提供关于 AI Agent 功能和未来的宝贵见解。
- **提供落地页帮助！**：一位成员正在为其项目寻求搭建落地页的帮助，并计划在 Mozilla AI 舞台上进行现场演示。
  
  - 有兴趣的成员应 [在此线程中联系](https://discord.com/channels/1089876418936180786/1307044141657751592) 以获取协作营销支持。

 

---

---

---

---

---

{% else %}

> 完整的频道详情已针对邮件进行了截断。
> 
> 如果你想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}