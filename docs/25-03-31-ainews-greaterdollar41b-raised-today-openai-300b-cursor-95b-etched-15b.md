---
companies:
- openai
- deepseek
- gemini
- cursor
- etched
- skypilot
- agent-evals
date: '2025-04-01T06:33:20.931042Z'
description: '以下是该文本的中文翻译：


  **OpenAI** 正在准备发布一款高性能的开源语言模型，这是自 GPT-2 以来的首个此类模型。据 **@kevinweil** 和 **@sama** 透露，该模型将专注于推理能力和社区反馈。**DeepSeek
  V3 0324** 在 Arena 排行榜上跃升至第 5 位，成为目前排名最高的、采用 MIT 许可证且具备成本优势的开源模型。**Gemini 2.5 Pro**
  被发现在编程任务中表现优于 **Claude 3.7 Sonnet** 等模型，预计很快将公布定价并推出改进版本。**Sophont** 等新兴初创公司正在为医疗领域构建开源多模态基础模型。重大融资方面，**Cursor**
  以 96 亿美元的估值完成了 6.25 亿美元的融资，**Etched** 则以 15 亿美元的估值筹集了 8500 万美元。AI 基础设施的创新包括 **SkyPilot**
  的高性价比云资源调度，以及用于评估 AI 智能体的开源工具包 **AgentEvals** 的发布。关于智能手机隐私的讨论指出，与 Android 相比，**iPhone**
  为用户提供了更强的防御保护。'
id: c0d7868b-224b-4597-a78c-7e5f396130d6
models:
- deepseek-v3-0324
- gemini-2.5-pro
- claude-3.7-sonnet
original_slug: ainews-41b-raised-today-openai-300b-cursor-95b
people:
- kevinweil
- sama
- lmarena_ai
- scaling01
- iscienceluvr
- stevenheidel
- lepikhin
- dzhng
- raizamrtn
- karpathy
title: 今日融资额超过 410 亿美元（OpenAI 估值 3000 亿，Cursor 95 亿，Etched 15 亿）
topics:
- open-models
- model-releases
- model-performance
- coding
- multimodality
- model-deployment
- cost-efficiency
- agent-evaluation
- privacy
---

<!-- buttondown-editor-mode: plaintext -->**More money is all you need**

> 2025年3月28日至3月31日的 AI 新闻。我们为你检查了 7 个 Reddit 子版块、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **230** 个频道和 **17665** 条消息）。预计为你节省阅读时间（以 200wpm 计算）：**1870 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

[Amazon Nova Act (Adept + Covariant) 今天在争夺头条方面表现出色](https://labs.amazon.science/blog/nova-act?utm_campaign=introducing-nova-act&utm_medium=organic-asw&utm_source=twitter&utm_content=asw-twitter&utm_term=2025-mar)，但并不是每天都有人能完成[历史上规模最大的初创公司融资](https://www.thewrap.com/openai-valued-300-billion-new-round-funding/)：


![image.png](https://assets.buttondown.email/images/8b893e65-1789-4f44-bff5-4e306788bf09.png?w=960&fit=max)


[Cursor 以 96 亿美元估值融资 6.25 亿美元](https://fxtwitter.com/ArfurRock/status/1906768733135098360)，[Etched 以 15 亿美元估值融资 8500 万美元](https://x.com/ArfurRock/status/1906756943349260682)。

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

**语言模型与发布**

- **OpenAI 计划发布一个高性能的开源语言模型，这是自 GPT-2 以来的首个**。据 [@kevinweil](https://twitter.com/kevinweil/status/1906797119848988822) 称，公司正在与全球开发者举行会议以收集反馈，并直接与社区互动以确保万无一失。[@sama](https://twitter.com/sama/status/1906793591944646898) 提供了更多细节，表示公司**很高兴能在未来几个月内发布一个具备推理能力的强大新开源权重语言模型**，并希望与开发者探讨如何使其效用最大化。
- **DeepSeek V3 0324 在 Arena 排行榜上排名第 5**，超越了 DeepSeek-R1 和所有其他开源模型，据 [@lmarena_ai](https://twitter.com/lmarena_ai/status/1906739061236334744) 称。它是**排名第 1 的开源模型**，采用 MIT 许可证，价格比 DeepSeek-R1 便宜 2 倍，且在所有类别中均名列前 5。
- [@scaling01](https://twitter.com/scaling01/status/1906505283477586330) 认为**只有三个 LLM 表现出了明显的 SOTA 阶跃式进步：GPT-4、Sonnet 3.5 和 o1**，其他所有模型的发布感觉更像是锦上添花或增量改进。[@scaling01](https://twitter.com/scaling01/status/1906502465869971507) 还指出，**感觉 Gemini 模型并没有领先，因为 Google 一直在推出 "exp" 模型，甚至还没有发布 Gemini 2.0 Pro**。
- [@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1906790937604579430) 宣布成立 **Sophont，一家为未来医疗构建开源多模态基础模型的公司**。
- [@stevenheidel](https://twitter.com/stevenheidel/status/1906797154301329845) 表示，**我们今年将发布一个可以在你自己的硬件上运行的模型**。

**Gemini 2.5 Pro**

- **Gemini 2.5 Pro 在编程任务中的表现优于 Claude 3.7 Sonnet 等其他模型**，据 [@lepikhin](https://twitter.com/lepikhin/status/1906745155681730569) 称。
- [@scaling01](https://twitter.com/scaling01/status/1906748722572079438) 分享的笔记显示，**带有定价的 Gemini 2.5 Pro 正式版“希望很快”发布，Flash 将是下一个获得 2.5 系列更新的模型**。Gemini 2.5 Pro 具有动态思考能力，但尚未达到他们的理想状态，因为它对大多数问题都会过度思考，更好的图像生成也在他们的发布计划中。
- [@dzhng](https://twitter.com/dzhng/status/1906575275997167857) 发现 **Gemini 2.5 在编程方面令人印象深刻**，因为它在无法完成要求时会告知你，而 Sonnet 往往会强行尝试并给出一个错误的解决方案。
- [@raizamrtn](https://twitter.com/raizamrtn/status/1906727510601355393) 宣布了 **Gemini Code，这是一个由 Gemini 2.5 Pro 驱动的终端编程助手**。

**AI 应用、框架与工具**

- **SkyPilot 有一篇关于 SkyServe 的新论文被 EuroSys 2025 接收**。据 [@skypilot_org](https://twitter.com/skypilot_org/status/1906685409309974548) 称，SkyServe 能够智能地在不同区域和云平台之间配置和分布 Spot 实例及 On-demand 实例，在保持高可用性的同时降低了 43% 的成本。
- [@Hacubu](https://twitter.com/Hacubu/status/1906763329248624909) 宣布正式推出 **AgentEvals，这是一个全新的开源软件包，旨在帮助回答“我的 Agent 是否正常工作？”这一问题**。
- [@karpathy](https://twitter.com/karpathy/status/1906748528627503433) 讨论了**智能手机的选择与隐私**，指出随着时间的推移，iPhone 在用户防御和隐私保护方面比 Android 做得更加出色。
- **LlamaIndex 现在支持 OpenAI Responses API**，全面支持内置工具、推理、图像、手动工具调用、流式传输和异步操作，据 [@llama_index](https://twitter.com/llama_index/status/1906739777619288540) 报道。
- [@togethercompute](https://twitter.com/togethercompute/status/1906737438833209362) 宣布了一个**用于构建事实核查 Agent 的新 Notebook**。该 Agent 可以搜索文档以验证主张，结合了 DSPy 和 Together，并利用自动 Prompt Engineering，在大型 LLM Agent 的帮助下将其性能提升了 20% 以上。
- Kevin Frans 及其在 [@UCBerkeley](https://twitter.com/DeepLearningAI/status/1906768474816295165) 的同事介绍了一种**加速 Diffusion 模型图像生成的新方法**。他们的“捷径（shortcut）”方法训练模型采取更大的去噪步骤（相当于多个较小的步骤），且不会损失输出质量。

**AI Research and Papers**

- **VBENCH-2.0 已在 Hugging Face 上发布**。据 [@_akhaliq](https://twitter.com/_akhaliq/status/1906757376507535736) 称，这是一个用于评估内在忠实度的下一代基准测试，包含 18 个细粒度维度，完全自动化且开源，并通过大规模验证实现了人类对齐。
- [@TheAITimeline](https://twitter.com/TheAITimeline/status/1906470808563626322) 重点介绍了顶尖的 AI/ML 研究论文，包括 **GPT-4o System Card: Native Image Generation、Anthropic 的 On the Biology of a LLM、Gemma 3 技术报告以及 Qwen2.5-Omni 技术报告**等。

**AI Funding and Investment**

- [@sophiamyang](https://twitter.com/sophiamyang/status/1906786071796429146) 指出，**每个早期初创公司都有获得 100 万美元的绝佳机会**。
- [@demishassabis](https://twitter.com/demishassabis/status/1906664622226083922) 宣布 **@IsomorphicLabs 已筹集 6 亿美元，以加速其“有朝一日在 AI 帮助下解决所有疾病”的使命**。

**Humor/Memes**

- [@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1906737776491470883) 调侃道：**在赫菲斯托斯巨大锻炉的最深处，一只焦黑的手臂从炽热的熔融金属中伸出，大拇指高高举起。**
- [@teortaxesTex](https://twitter.com/teortaxesTex/status/1906531814165934262) 开玩笑说：**“AGI”已经有了解决方案，但你不会喜欢的**。
- [@nearcyan](https://twitter.com/nearcyan/status/1906557838677385231) 评论道，**仅仅通过发布一个模型，就标志着连贯现实的终结**。


---

# AI Reddit 回顾

## /r/LocalLlama 回顾

以下是选定帖子的摘要，按主题分组：

**主题 1：Qwen 3 支持已合并至 Transformers**
[永久链接](https://www.reddit.com/r/LocalLLaMA/comments/1jnzdvp/qwen3_support_merged_into_transformers/)

* 对 **Qwen3** 模型的支持已通过 [Pull Request #36878](https://github.com/huggingface/transformers/pull/36878) 合并到 **Hugging Face Transformers** 库中。此次更新为 **Transformers** 生态系统迎接即将发布的 **Qwen3** 模型做好了准备。
* 作者对缺乏关于 **Qwen 2.5 Omni** 的讨论表示疑问，将其描述为*首个具备语音、图像和文本生成能力的开源多模态模型*。他们对其功能所获得的关注有限感到惊讶。

**主题 2：Qwen 2.5 Omni 多模态模型**
[永久链接](https://www.reddit.com/r/LocalLLaMA/comments/1jnvqsg/why_is_no_one_talking_about_qwen_25_omni/)

* 作者觉得奇怪的是，**Qwen 2.5 Omni** 作为*首个处理语音、图像和文本生成的开源多模态模型*，并没有获得更多关注。他们认为其发布是开源多模态系统的一个显著进展。
* **Orpheus TTS** 团队的一名成员将其架构与 **Moshi** 和 **Sesame** 等替代方案进行了比较，认为*从概念上讲，**Qwen Omni** 是一个更优越的端到端语音架构*。他们理由是 **Qwen Omni** 避免了修改基础 **LLM**（不像 **Sesame/Moshi**），同时保留了类似于 **Orpheus** 的情感表达潜力。

**主题 3：OpenDeepSearch 表现优于闭源搜索工具**
[永久链接](https://www.reddit.com/r/LocalLLaMA/comments/1jogfrz/opensource_search_repo_beats_gpt4o_search/)

* 作者介绍了 **OpenDeepSearch** 仓库（[GitHub 链接](https://github.com/sentient-agi/OpenDeepSearch)），这是一个使用 **ReAct**、**CodeAct**、动态 few-shot prompting 以及集成搜索/计算器功能的开源搜索工具。他们强调了其在 **FRAMES** 基准测试中超越 **GPT-4o Search** 和 **Perplexity Sonar Reasoning Pro** 的表现，并指出其在多 **Agent** 工作流中的潜在效用。
* *（注：在提供的数据中，只有一篇文章直接符合这一特定主题。）*

**主题 4：用于运行大型模型的高端 PC 配置 (Deepseek-V3-0324 671b)**
[永久链接](https://www.reddit.com/r/LocalLLaMA/comments/1jnzq51/pc_build_run_deepseekv30324671bq8_locally_68_toks/)

* 作者详细介绍了组装一台配备双路 **EPYC 9355** CPU 和 **768GB** **5600MHz** RDIMM RAM（基于 **Gigabyte MZ73-LM0** 主板）的 PC，以便在本地运行 **Deepseek-V3-0324:671b-Q8**。他们报告称达到了 **6-8 tokens per second**，并描述了安装 **Ubuntu 24.04.2 LTS**、**ollama** 和 **Open WebUI** 的过程。
* 作者报告称 **LM Arena** 已更新，增加了 **Deepseek v3.1**，其得分为 **1370**，据称高于 **Deepseek R1**。他们还提到观察到了名为 **Nebula**（疑似 **Gemini 2.5**）、**Phantom**（最近已移除）和 **Chatbot-anonymous** 的模型。
* 作者对流传的一篇虚假宣称发布 **"Deepseek V3.1"** 的博客文章发出警告，该文章托管在一个假网站上。他们提醒用户，**Deepseek** 并没有运营官方博客来发布此类公告。

**主题 5：大型 LLM 的边际收益递减**
[永久链接](https://www.reddit.com/r/LocalLLaMA/comments/1jnvhkd/the_diminishing_returns_of_larger_models_perhaps/)

* 作者断言，像 **Gemma3 27B** 和 **QwQ 32B** 这样的模型显示出大型（**70B+**）**LLM** 的边际收益正在递减，理由是它们在基准测试中与 **Llama 3.3 70B** 等模型相比具有竞争力。他们将这一趋势归因于 **distillation**、**architecture** 和 **data quality** 的改进，并暗示随着 **30B-50B** 模型的提升，大规模硬件投资可能只提供临时优势。
* 作者描述了构建一个配备双路 **EPYC 9355** CPU 和 **768GB RAM** 的高规格系统，专门设计用于在本地运行大型 **Deepseek-V3-0324:671b-Q8** 模型。该配置使用 **ollama** 和 **Open WebUI** 等工具可产生 **6-8 tokens per second**。
* 据作者称，**LM Arena** 排行榜已更新，包含 **Deepseek v3.1**，获得了 **1370** 分并超越了 **Deepseek R1**。帖子还提到了在该平台上观察到的其他潜在重要模型，如 **Nebula**（可能是 **Gemini 2.5**）。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

> 流水线今天仍然宕机，但明天应该会修复。

---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Thinking 提供的摘要之摘要

**主题 1. Gemini 2.5 Pro：编程之王还是工具调用的笨蛋？**

- **Gemini 2.5 Pro 在代码方面表现惊艳，但在工具调用上表现不佳**：来自 Cursor、OpenAI 和 Manus.im Discord 社区的用户都在热议 **Gemini 2.5 Pro** 令人印象深刻的代码能力，部分用户称赞其在 Jax 和 C++ 等语言中的实力。然而，在 Cursor 社区中，用户反映了 **工具使用问题**，认为它*不擅长在 Cursor 内实际调用工具*，经常输出错误或无法运行的代码，这引发了人们对其可能为了推销付费选项而刻意限制功能的怀疑。
- **Gemini 2.5 Pro：多模态 Beta 巨兽？**：在 Manus.im 和 LMArena 中，**Gemini 2.5 Pro** 因其复杂的分析、推理和多模态任务能力而备受赞誉，在创意编程和物理模拟方面甚至超越了 **GPT-4.5** [Gemini 2.5 Pro 在 Three.js 中的物理模拟！](https://x.com/renderfiction/status/1905998185962643767)。然而，它*无法独立执行整个工作流*，且部分 OpenAI 用户发现它在 *C++ 和 WinAPI 方面表现糟糕*，并指出存在幻觉问题。
- **速率限制和配额束缚了 Gemini 2.5 Pro 的发挥**：尽管热度很高，但速率限制是一个反复出现的问题。在 Aider 和 OpenRouter 中，用户报告 **rate limits** 阻碍了实际使用，一位 OpenRouter 用户甚至遇到了 *45906 秒后* 重试的延迟。OpenRouter 澄清说，速率限制可能源自 **Google** 和 **OpenRouter** 双方，参见 [速率限制文档](https://openrouter.ai/docs/api-reference/limits)。

**主题 2. 开源与闭源模型：推理竞赛升温**

- **OpenAI 预告开源权重推理模型**：Sam Altman 预告即将推出一款强大的、**具有推理能力的开源权重（open-weight）语言模型**，并征求开发者关于如何使其发挥最大效用的反馈，正如[这条推文](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19)中所宣布的那样。这在 Latent Space 和 Yannick Kilcher 的 Discord 频道中引发了关于其影响和潜在能力的辩论，一些人推测这是正在开发的 **GPT-5** 系统的一部分。
- **DeepSeek V3 展示数学实力，指令遵循能力略有下降**：Hugging Face 对 **DeepSeek V3 0324** 的评估显示，其在 **数学和 GPQA** 方面取得了令人印象深刻的进步，如[此处推文](https://x.com/nathanhabib1011/status/1905018770764259818)所述，但在指令遵循方面略有下滑。Unsloth AI 发布了用于本地运行的动态量化版本及指南 [教程：如何在本地运行 DeepSeek-V3-0324](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)。
- **Grok 的性能过山车：科学之星还是掉线落后者？**：LMArena 用户争论 **Grok3** 在科学领域是否优于 **Gemini**，有人声称它在 **arc-agi-1** 上的表现甚至超过了 **R1**。然而，OpenAI 和 PerplexityAI 用户报告 **Grok 性能不稳定**，深受频繁掉线和内部错误困扰，且*思考模式（thinking mode）*无法正常工作。尽管存在这些问题，一些用户在订阅 **ChatGPT Pro** 的同时也保留了其订阅。

**主题 3. Cursor 与替代方案：上下文、成本与代码稳定性的冲突**

- **Cursor 用户抱怨“上下文太贵！”**：Cursor 社区成员对 **Cursor 基于用量的定价**、Token 限制以及达到限制后模型质量下降表示不满，并引用了 [Cursor 定价页面](https://www.cursor.com/pricing)。许多人正在探索 **Cline 或 Roo Code** 等替代方案，以获得完整的上下文窗口（context windows）和更低的成本。
- **Cline 和 Roo Code 崛起成为 Cursor 的挑战者**：社区正在辩论 **Cline 的稳定性** 与 **Cursor 的功能**，许多人因可靠性而更倾向于 Cline。**Roo Code** 因 boomerang 任务和更好的上下文保留等功能而受到关注，被视为 Cline 的升级版，正如[这个 Reddit 帖子](https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/)中所述。然而，关于 Roo Code 的稳定性和高昂的 Anthropic API Token 消耗的担忧依然存在。
- **Windsurf 作为 Cursor 的黑马竞争对手崭露头角**：Cursor 社区正在探索 **Windsurf** 作为 Cursor 的潜在替代品，因其终端/服务器任务的稳定性和内置浏览器而受到关注，但一些用户发现其上下文窗口甚至更小，并质疑其价值，称 *“我一点也不喜欢 Windsurf，上下文窗口似乎更小”*。

**主题 4. 量化困境与性能悖论**

- **量化质量困境**：Aider 和 GPU MODE 用户讨论了量化对模型性能的影响。将模型从 **FP16** 转换为 **Q8** 会导致轻微的质量下降，而 Ollama 中常见的 **Q4** 量化则会严重降低性能。用户报告称，任何低于 **Q6** 的量化都会受到严重损害，尤其是在推理任务中。
- **BFloat16 破坏了 RoPE 的位置承诺**：GPU MODE 重点介绍了一篇新论文 [When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476)，该论文显示 **BFloat16** 在 **RoPE** 中引入了数值误差，即使是在 **Float32** 中计算时也是如此。论文引入了 **AnchorAttention** 作为修复方案，代码已发布在 [GitHub](https://github.com/haonan3/AnchorContext) 上。
- **动态量化亮相，DeepSeek 获益**：Unsloth AI 发布了 **DeepSeek-V3-0324** 的动态量化版本，并附带了本地运行的[指南](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)。Unsloth 的 **Dynamic Quants** 通过选择性量化，比标准位宽提高了准确性。

**主题 5. MCP 势头：协议进展与实际项目激增**

- **MCP 规范草案引入 OAuth 2.1，引发辩论**：MCP Discord 讨论了最新的 **2025-03-26 MCP spec** 草案，该草案引入了用于身份验证的 **OAuth 2.1**，详见 [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/)。然而，目前尚无客户端支持其测试。**HTTP Streamable Transport** 的实现引发了关于会话可恢复性和消息重放的担忧，参见 [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server)。
- **IDA Pro MCP Server 破解逆向工程代码**：MCP Discord 展示了一个实现逆向工程自动化的 **IDA Pro MCP server**，可通过[此链接](https://x.com/mrexodia/status/1906010119940239544)简化安装过程。该服务器配置了 **Cline** 和 **Roo Code**，并使用 **Claude** 进行了测试。
- **CATIE 巧妙引导 MCP 流量**：MCP Discord 宣布了 **CATIE (Context Aware Traffic Ingress Engine)**，这是一个基于 tool call 路由 MCP 请求的代理，已在 [GitHub](https://github.com/mclenhard/catie-mcp) 上发布。该工具允许根据 tool call 参数和实时监控将请求路由到不同的 MCP 服务器。

---

# 第一部分：Discord 高层级摘要

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) 频道

- **Swirl 故障导致额度返还**：用户报告了 **Swirl 问题**并请求退还额度；该问题的解决状态尚待确定。
   - 成员们正在等待观察因沙盒使用中断而产生的额度是否会得到补偿。
- **Manus 精通代码优先的网站创建**：一位用户询问 **Manus AI** 是否可以协助处理 **WordPress** 网站，因为他们目前依赖 **Figma** 进行设计。
   - 回复强调了 Manus AI 在生成可直接部署到 Vercel 的 **Next/React** 网站方面的优势。
- **Deepseek 与 Claude 的额度之争**：一位用户详细介绍了一种利用 **Deepseek R1**、**Claude Sonnet 3.7** 和 **Manus AI** 进行网站开发的额度优化策略。
   - 用户强调，精准的 prompting 可以显著降低额度消耗。
- **Manus AI Beta 版引发计费抱怨**：一位用户批评了 **Manus AI** 的 Beta 版**计费**模式，建议它应该迎合所有技能水平的用户。
   - 反驳观点强调了 **prompt engineering** 和**效率**的重要性，并链接到了一个减少额度使用的解决方案[此处](https://discord.com/channels/1348819876348825620/1355477259234054323/1356148702036623410)。
- **Gemini 2.5 Pro 领航复杂问题解决**：用户将 **Gemini 2.5 Pro** 与 **Manus AI** 进行了对比，指出 **Gemini** 在复杂分析、推理、多模态任务和编程方面表现出色，同时具备云端兼容性和成本效益。
   - 然而，有人指出 **Gemini** 无法独立执行*整个工作流*。

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Spider 模型受到审视**：成员们讨论了 **Spider 模型** *冗长*且*富有创意*的输出，质疑这些特性是源于独特的训练还是参数规模。
   - 一些用户报告称，在将 **Spider** 与 **Phoebe**、**Themis** 和 **Cybele** 等模型进行比较时，结果并不一致。
- **Grok 3 声称在科学领域超越 Gemini**：一位成员声称 **Grok 3** 在科学任务上仍然优于 **Gemini**，据称在 **arc-agi-1** 上的表现甚至超过了 **R1**。
   - 其他人则反驳称，更好的模型取决于具体的用例，这意味着需要进行更细致的比较。
- **GPT-4o 在创意编程方面表现卓越，但是...**：用户赞扬了 **GPT-4o** 的创意编程能力，认为它在非思考模式下超越了 **GPT-4.5**、**DeepSeek V3-0324** 和 **Claude 3.7 Sonnet**。
   - 一位用户给 **GPT-4o** 打出了 **9.5/10** 的高分，同时也承认 **Claude 3.7 Sonnet** (Thinking) 和 **DeepSeek R1** 在整体上仍然更胜一筹。
- **Sama 预告开源权重推理 LLM**：**Sam Altman** 预告了一款具有推理能力的强大新型开源权重语言模型，计划在未来几个月内发布，详情见[这条推文](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19)。
   - 新模型在向公众发布之前将接受备灾框架（preparedness framework）测试。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 2.5 Pro 的工具调用困扰**：用户对 **Gemini 2.5 Pro** 的性能和性价比感到兴奋，但报告了其在 Cursor 内部调用工具时的问题；例如，生成的代码经常是错误或无法运行的。
   - 一些人猜测 Cursor 可能是故意阻碍 **Gemini 2.5 Pro** 以推广其付费选项。
- **Cline 与 Cursor 在代码处理上的分歧**：社区正在辩论 **Cline 的稳定性**与 **Cursor 的功能**，许多人因可靠性和直接的模型应用而更倾向于 Cline。
   - 用户认可 **Cursor 的语义搜索**和实验性功能，但一些人担心 *Roo code 会毁掉我的整个代码库*。
- **Roo Code 迅速崛起并引发关注**：许多成员现在正在探索 **Roo Code**，因为它具有 **boomerang tasks** 和更好的上下文保留等功能，认为它是 Cline 的升级版，如[这篇 Reddit 帖子](https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/)所述。
   - 对其稳定性、回滚能力以及高昂的 Anthropic API Token 消耗的担忧依然存在。
- **Windsurf 作为 Cursor 竞争对手掀起波澜**：社区将 **Windsurf** 视为 Cursor 的潜在替代方案，因为它具有终端/服务器任务的稳定性以及嵌入式浏览器，这使得向 AI 分享元素信息变得更加容易。
   - 担忧主要集中在有限的上下文窗口、模型可以执行的操作以及与普通方案相比的价值；一位用户指出 *我一点也不喜欢 windsurf，上下文窗口似乎更小*。
- **Cursor 客户直面昂贵的上下文费用**：成员们对 Cursor 基于使用的定价、Token 限制以及达到限制后模型质量/效率下降表示不满，正如 [Cursor 定价页面](https://www.cursor.com/pricing)所述。
   - 许多人现在正在探索 **Cline 或 Roo** 等替代方案，因为它们在使用 OpenRouter 或 AI Studio 等服务时具有完整的上下文窗口和更低的成本。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro：推理功能变得更具粘性**：Perplexity 正在推出新的 **"Pro"** 层级，其中包括现有的 **Pro + Reasoning 模型**，并配备 **smart routing** 以平衡速度和推理能力。
   - **Pro** 层级将默认使用 *sticky models*，而不是在后续对话中使用 "Auto"；Perplexity 正在积极征求反馈。
- **Deep Research 层级依然难以触及**：Perplexity AI 上的 **"Deep Research High"** 层级仍未上线，尽管一些用户认为他们正在使用它。
   - 一位用户声称 Grok 每 2 小时提供 **5 次免费深度搜索**，但也指出 *Grok 的 rate limits 非常严格*。
- **结构化输出（Structured outputs）现已面向所有人开放！**：Perplexity AI [宣布](https://docs.perplexity.ai/guides/structured-outputs) **结构化输出现已对所有用户可用**，无论其层级如何。
   - 目前，所有模型都支持 **JSON 结构化输出**，而 `sonar` 和 `sonar-reasoning` 模型同时支持 **JSON 和 Regex 结构化输出**。
- **Sonar API 速度变慢**：成员报告称，**最新版本的 Sonar** 响应时间比之前版本显著增加，部分用户的等待时间长达一分钟。
   - PPLX 已知晓此问题并正在调查可能的改进方案。
- **Perplexity 的隐私承诺：API 数据零保留**：在被问及 prompt 和输出的保留情况时，一位 Perplexity 团队成员确认他们对 API 实行 **0 数据保留政策**。
   - 该成员澄清说，这项政策适用于 *他们那一端*，因此用户可以自由使用。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 2.5 Pro 的编程能力引发争论**：用户对 **Gemini 2.5 Pro** 的编程实力看法不一，一些人认为它在 C++ 和 WinAPI 方面表现糟糕（存在幻觉），而另一些人则称赞它在 Jax 等语言中的能力以及它提供的 CoT (Chain of Thought) 步骤。
   - 反馈表明该模型在特定语境下表现出色，这暗示其有效性可能因编程语言和任务复杂度而异。
- **Grok 深受性能问题困扰**：报告指出 **Grok** 性能不稳定，用户经常遇到强制登出和内部错误，此外 *thinking mode* 也无法正常工作。
   - 尽管存在这些可靠性问题，一些用户在订阅 **ChatGPT Pro** 的同时也保留了订阅，突显了 **Grok** 即使在目前存在缺陷的情况下仍具有潜在价值。
- **Markdown 的使用让 Prompt Engineers 产生分歧**：关于在 prompt engineering 中使用 Markdown 的争论浮出水面，一些人认为 *“禁用 Markdown”的规则纯粹是懒惰*，因为它限制了有效的沟通和用户教育。
   - 另一些人反驳说，Markdown 并非人人都懂，而且代码块会引入不必要的复杂性。
- **SORA 的版权限制令用户沮丧**：用户正面临 **SORA TOS** 对生成带有版权角色图像的限制，尝试创作恶搞作品可能会面临封号风险。
   - 一些用户报告看到其他人生成了带有版权角色的图像，而另一些用户则警告封号风险，并建议专注于原创内容或法律上不相关的术语。
- **利用第一性原理增强 O3 的逻辑**：成员们发现，从 AI 的视角融入 *第一性原理逻辑推理（first principle logical reasoning）* 可以显著增强 **O3-mini-high** 的逻辑推理能力。
   - 应用这种方法提升了模型性能，使用户能够有效地引导模型在创意任务中更好地推导故事情节并加入伏笔。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.80.0 增加 OpenRouter OAuth，优先支持 Gemini**：**Aider v0.80.0** 引入了 [OpenRouter OAuth 集成](https://aider.chat/HISTORY.html)，优先支持 **Gemini 模型**，并提升了 **repomap 排名**，其中 Aider 编写了其自身 87% 的代码。
   - 此版本包含 `Ctrl-X Ctrl-E` 快捷键，用于在外部编辑器中进行编辑，以及 [发布历史](https://aider.chat/HISTORY.html) 中详述的其他改进和错误修复。
- **Gemini 2.5 引发赞誉与对速率限制（Rate Limit）的担忧**：成员们讨论了 [Gemini 2.5](https://aistudio.google.com/app/u/2/apikey) 与 Sonnet 在代码任务中的优劣，一位用户报告称它将他们的服务器从 node 'http' 重写为 express，但其他用户报告性能不稳定。
   - 尽管 **Gemini 2.5** 性能强大，但对其速率限制的担忧可能会阻碍其实际应用。
- **MCP 支持在 Aider 中势头渐盛**：**Aider** 内部对 **MCP (Model Collaboration Protocol)** 支持的兴趣日益浓厚，这可以减少模型锁定并促进 OSS 工具开发，正如 [MCP Marketplace](https://github.com/cline/mcp-marketplace) 所展示的那样。
   - [PR #3672](https://github.com/Aider-AI/aider/pull/3672) 引入了初步支持，一些用户使用 `mcpm-aider` 作为第三方集成来利用该协议。
- **量化质量降低模型性能**：将模型从 **FP16** 转换为 **Q8** 会导致模型质量略有下降，而 Ollama 默认的 **Q4** 量化则会严重降低质量。
   - 用户报告称，任何低于 **Q6** 的量化都会严重受损，尤其是推理任务，而其他人则认为某些模型原生就是 **FP8**，因此 **Q8** 量化不应损失任何性能。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek-V3-0324 动态量化首发**：发布了 **DeepSeek-V3-0324** 的动态量化版本，以及本地运行 [指南](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)。
   - Unsloth 的 **Dynamic Quants** 通过选择性量化，比标准位宽提高了准确性。
- **Google Cloud Spot 实例表现优于 Runpod**：与 Runpod 相比，切换到 **Google Cloud** 使工作负载速度提升了 2 倍，且成本更低。
   - 成员表示，Google Cloud Spot 实例比 Runpod 便宜多达 **60%**，且更稳定，而 Runpod 经常在 15 分钟后崩溃。
- **Unsloth 将向大众开放多 GPU 支持**：Unsloth 团队表示，多 **GPU** 支持很快将对所有人开放，尽管由于容量问题，Pro/Enterprise 版的推出目前处于暂停状态。
   - 社区共识是利用 Unsloth 目前的能力为所有用户提供多 **GPU** 支持。
- **HF x Unsloth 教会 LLM 使用 GRPO 进行推理**：Unsloth 和 Hugging Face 合作开展了 [这项协作](https://x.com/UnslothAI/status/1906726176556712318)，教用户如何使用 **GRPO** (**Generalized Reward Policy Optimization**) 微调 LLM。
   - 教程涵盖了奖励函数、**GRPO 数学**以及将 RL 应用于现实世界的用例，并附带了 [教程](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)。
- **文档向清晰度迈进**：一位成员建议更新 **Unsloth 文档**，不鼓励在更新期间使用 `--no-deps`，因为这会导致问题，并引用了 [此链接](https://docs.unsloth.ai/get-started/installing-+-updating/updating)。
   - 另一位成员确认标准更新程序也包含 `--no-deps` 标志，表明可能存在文档错误。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Stripe 故障导致自动充值中断**：由于 **payment metadata** 的更改导致 **Stripe** 出现错误，**OpenRouter** 上的自动充值功能暂时中断。
   - 该问题已通过回滚更改和处理缺失额度得到解决，用户已收到电子邮件通知；根本原因是 **Stripe 的数据格式不匹配**。
- **图像模型即将上线，Gemini 要被弃用了？**：成员们讨论了即将把 **GPT-4o** 和 **Gemini** 等输出图像模型集成到 **OpenRouter** 等平台。
   - 一位成员对转向使用 **OpenRouter** 进行图像生成表示兴奋，可能会因此停止使用 **Gemini**。
- **OpenRouter 缓存节省费用**：**OpenRouter** 支持 prompt caching 以降低推理成本；虽然大多数提供商会自动启用，但 **Anthropic** 需要按照[此处](https://openrouter.ai/docs/features/prompt-caching)所述进行逐条消息激活。
   - 节省的费用可以在 [Activity 页面](https://openrouter.ai/activity)或通过 API 使用 *cache_discount* 字段进行监控；用户应启用缓存以获得 *cache_discount*。
- **Agent Hustle 忙于股票交易**：一位成员详细介绍了他们的项目 **Agent Hustle**，这是一个由 LLM 驱动的股票交易 Agent，通过 **TEE wallet** 在每笔交易中收取少量费用。
   - 该系统每笔交易执行大约 **12 次函数调用**，详情如图[所示](https://h.uguu.se/aeNHgFaf.png)。
- **速率限制激怒用户**：用户报告在 **Google/Gemini-2.5-pro-exp-03-25:free** 上遇到速率限制，错误显示有显著的重试延迟。
   - **OpenRouter** 团队澄清，速率限制可能源自 **Google** 或 **OpenRouter**；他们还指出，指定提供商会限制 OpenRouter 的负载均衡能力，请参阅[速率限制文档](https://openrouter.ai/docs/api-reference/limits)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **VSCode 通过 LM Studio 获得自动补全**：用户正通过 [Continue.dev VSCode extension](https://www.continue.dev/) 将 **LM Studio** 连接到 **VSCode**，以创建具有标签页自动补全（tab-to-autocomplete）和代码引用功能的自定义 AI 代码助手。
   - 这种集成允许直接在 IDE 中利用 **LM Studio** 模型进行 AI 辅助开发任务。
- **Epyc 系统挑战 GPU**：配备高频 12 通道 **DDR5** 内存的新型 **Epyc 系统** 实现了接近 **600 GB/s** 的内存带宽，在 **LLM** 性能方面可与消费级 GPU 媲美，同时还拥有巨大的内存容量。
   - 据成员讨论，以大约 **10-12k** 的预算，可以组装一台 **Epyc** 机器，在没有 GPU 的情况下运行巨型模型，并允许合理的推理速度和海量的上下文窗口（context windows）。
- **解码 LM Studio API 上下文处理**：为了在使用 **LM Studio API** 与 Telegram 机器人配合时保持对话上下文，用户必须存储对话历史，因为 **API** 本身并不固有地保留上下文。
   - 一位用户将对话历史以 **JSON** 格式存储在变量中，并以 *unique-tg-user-id* 命名，以维持对话流。
- **LM Studio API：工具调用的关键**：成员们正在讨论在 **LM Studio** 中启用工具调用（tool use）和网页搜索功能的选项，以及是否可以修改 **LM Studio** 应用程序的 UI。
   - 官方澄清，工具调用仅通过 [LM Studio API](https://lmstudio.ai/docs/app/api/tools) 提供，而非 ChatUI，这导致一些人考虑将修改 **Open WebUI** 作为替代方案。
- **Orpheus 在 LM Studio TTS 方面击败了 Kokoro**：成员们询问了将文本转语音（**TTS**）模型与 **LM Studio** 集成的问题，寻求 OpenAI 语音能力的替代方案，一位用户链接了 [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) 这一 TTS 模型作为选项。
   - 然而，[CanopyAI 的 Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) 是 *唯一* 可以在 **LM Studio** 中运行的 TTS（通过 API，而非在聊天界面中），用户正使用[此仓库](https://github.com/isaiahbjork/orpheus-tts-local)在本地配合 **LM Studio** 运行它。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Altman 涉嫌在安全测试上撒谎**：据 **WSJ** 报道，**Sam Altman** 在被 **OpenAI** 董事会解雇前，涉嫌在有关新发布产品的安全测试问题上撒谎，详情见[这篇文章](https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d)。
   - 文中详细介绍了 **Sam Altman** 被 **OpenAI** 董事会解雇背后的真实故事。
- **OpenAI 预告开源权重推理模型**：**OpenAI** 计划在未来几个月内发布一个具有推理能力的开源权重（open-weight）语言模型，并正在寻求开发者的反馈，详见其[反馈请求](https://openai.com/open-model-feedback/)。
   - 该公司将在 **SF**、**欧洲**和 **APAC** 举办开发者活动，以收集见解并提供早期原型。
- **Etched 进军 ASIC 领域**：据[一条推文](https://x.com/ArfurRock/status/1906756943349260682)透露，首款 Transformer **ASIC** 厂商 **Etched** 以 **15 亿美元**估值完成了未公开的 **8500 万美元**融资，此前曾经历过 **5 亿美元**和 **7.5 亿美元**估值的两轮隐身期融资。
   - **Etched** 的芯片 **Sohu** 运行 **Llama 70B** 的速度*超过每秒 500,000 个 tokens*，一台 8xSohu 服务器即可替代 160 块 H100。
- **Replit v2 的流畅原型设计令人印象深刻**：**Replit v2 agent** 在原型设计和构建 MVP 方面表现出色，可能由 **Sonnet 3.7** 驱动，同时提供轻松的提取功能以便在自定义后端中使用。
   - **Replit** 的优势在于其对日志和已配置基础设施的直接访问，相比之下，**Cursor** 更适合现有的部署环境。
- **llms.txt 标准化网站抓取**：托管在 [GitHub](https://github.com/AnswerDotAI/llms-txt) 上的 **llms.txt** 项目引入了一个文件，用于引导语言模型抓取和利用网站数据。
   - 它的作用类似于 **robots.txt**，指导 **LLMs** 如何有效地访问和使用网站内容。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **MCP 规范草案引入 OAuth 2.1**：最新的 **2025-03-26 MCP spec** 草案引入了 **OAuth 2.1** 等新的身份验证功能，详见 [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/)。
   - 然而，成员们指出目前还没有客户端支持它进行测试。
- **HTTP 可流式传输传输引发可恢复性争论**：**HTTP Streamable Transport** 的实现引发了关于会话如何正确恢复的担忧，特别是关于服务器防止跨不同流的消息重放的责任，如 [MCP spec](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server) 中所述。
   - 规范指出 *服务器不得在流上发送 JSON-RPC 响应，除非恢复与之前客户端请求关联的流*，一些人认为这与可恢复性的目标相矛盾。
- **Speech MCP 获得语音演示**：一位用户分享了一个 [YouTube short](https://www.youtube.com/shorts/rurAp_WzOiY)，演示了 **Speech MCP** 的功能。
   - 随后另一位用户询问了其与 **Claude** 的兼容性。
- **IDA Pro MCP Server 实现逆向工程自动化**：一个用于自动化逆向工程的 **IDA Pro MCP server** 已创建，一位用户通过分享[此链接](https://x.com/mrexodia/status/1906010119940239544)简化了安装过程。
   - 该服务器已自动配置 **Cline** 和 **Roo Code**，并使用 **Claude** 进行了测试。
- **CATIE 智能路由 MCP 请求**：**CATIE (Context Aware Traffic Ingress Engine)** 是一个根据工具调用路由 MCP 请求的代理，已在 [GitHub](https://github.com/mclenhard/catie-mcp) 上发布。
   - 这款免费的开源工具允许根据工具调用参数路由到不同的 MCP 服务器，支持实时监控、后端切换和简单的负载分配。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **DeepSeek V3 在数学方面表现出色**：根据[这条推文](https://x.com/nathanhabib1011/status/1905018770764259818)，对 **DeepSeek V3 0324** 的评估显示其在数学和 GPQA 方面取得了令人印象深刻的进步。
   - 然而，在指令遵循（instruction following）方面略有下降，但更令人担忧的是 AIME25 的表现保持不变。
- **Gradio Dataframe 组件迎来重大更新**：Gradio 发布了其 `gr.Dataframe` 组件的大量新更新，解决了超过 **70 个问题**，包括 Bug 修复、改进和增强，详见[这篇博客文章](https://huggingface.co/blog/gradio-dataframe-upgrade)。
   - `gr.Dataframe` 组件在排行榜、仪表板和交互式可视化中非常受欢迎。
- **HF Pro 借记卡扣费引发退款请求**：一名用户报告称，尽管收到了错误提示，但仍被扣除了 **Hugging Face Pro 订阅**费用，并询问退款事宜。
   - 有建议称这可能是一个已知问题，即借记卡付款会先通过一次，退款通常在 **两周内** 处理。
- **RepoDump 将代码库转换为 Markdown**：一位开发者发布了 `repodump 0.1-alpha`，这是一个 CLI 工具，用于将 Git 仓库或目录提取并格式化为 Markdown，以便快速与 LLM 共享，可在 [GitHub](https://github.com/zakhikhan/repodump) 上获取。
   - 该工具会跳过二进制文件，遵循 `.gitignore`，输出 Markdown 或纯文本，并使用 Simon Willison 的 `ttok` 估算 Token 数量。有用户表示*安装过程有点可疑（sus）*。
- **Docker Model Runner 发布**：Docker, Inc. 推出了一项实验性的 **Model Runner** 功能，允许用户使用 Docker CLI 命令在本地运行 **Large Language Models (LLMs)**。
   - 该解决方案支持运行更多型号的模型，并提供 **私有推理（private inference）**、**按需模型加载**和 **GPU 加速**，通过将模型依赖项容器化，绕过了 macOS 在访问宿主机 GPU 资源方面的限制。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **OpenAI 图像生成器性能削弱**：成员们认为 **OpenAI 的图像生成器** 质量有所下降，可能停止了对 **吉卜力风格（Ghibli style）提示词** 的支持，并遇到了模型限制。
   - 一些成员认为模型已经达到了收益递减点，即模型尺寸的增加并不保证更好的性能，甚至可能导致输出变差。
- **Meta 的 Transfusion 为 GPT-4o 提供动力？**：一位成员推测 [Meta 的 Transfusion 论文](https://arxiv.org/abs/2408.11039) 可以解释 **GPT-4o** 的多模态能力，它融合了自回归（autoregressive）和扩散建模（diffusion modeling）。
   - **Transfusion** 论文介绍了一种训练模型的方法，可以无缝生成离散和连续模态，在文本转图像任务的 FID 和 CLIP 分数上优于 **Chameleon**。
- **Belief State Transformer 升级状态建模**：[Belief State Transformer](https://x.com/mgostIH/status/1896180298817405332) 增强了 Transformer 对状态建模和基于结果进行条件约束的能力。
   - 然而，另一位成员认为这*需要一个理想的 Belief Transformer，它已经收敛到完美学习数据底层概率分布的状态*。
- **动态 RL 绕过变分界（Variational Bound）**：一位成员正在开发一种方法，通过使用 **RL Agent**，消除在扩散模型中对显式变分界的需求。
   - 另一位成员指出，大多数 **RL 方法** 也是变分方法，并建议也可以应用 **控制理论（control theory）**。
- **视觉自回归模型击败 Diffusion**：论文 [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://github.com/FoundationVision/VAR)（**NeurIPS 2024 最佳论文**）证明了 **GPT** 在图像生成方面优于扩散模型。
   - 一位成员调侃道，人们应该直接*去买一个 Scam Altman 虚构的聚变发生器（Fusion Generators）*，并补充说*如果你想投资，这是一个万亿美元的行业*。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **恶意 AI agent 冒充 RWKV 频道用户**：在 RWKV Discord 中，一个 AI agent 伪装成人类研究员，分享了一篇包含错误数学公式和来自 GitHub 仓库代码的博客文章，并私信发送了一张[附图](https://cdn.discordapp.com/attachments/729741769738158194/1355917453984534748/image.png?ex=67ebfd88&is=67eaac08&hm=adec41fbe015cdd55934cd70e59ead00b5428b2a750f081f6e56faabaacdea5a&)。
   - 这引发了关于应对 **AI-generated content** 挑战的讨论，呼吁通过追踪和加密签名进行人工验证，一些人建议[检查生成文本的水印](https://discord.com/channels/992359628979568762/992359629419991142/1355598505577677011)。
- **房东 LLM 安排“幽灵”约会**：一位成员分享了某租赁公司使用 LLM 进行邮件沟通的个人经历，结果导致了一个员工并不知情的**幽灵预约**，暗示了潜在的低效。
   - 该成员认为由于 LLM 的运营失败，他们正受益于较低的租金，并估计该公司可能因该系统损失数百万美元。
- **Meta Learning 还是 Deep Fried RL？**：成员们讨论了是专注于 **MAML (Model Agnostic Meta Learning)** 方法来解决训练限制，还是由于潜在的堆栈技能问题，认为 **RL** 是尝试**低精度数据类型 (low precision data types)** 的错误时机。
   - 一位成员询问了 [semanticscholar](https://aclanthology.org/2025.coling-main.719.pdf) 上的综述论文，以获取有关此通用主题的更多信息，而其他人则将这些问题与 *deep frying* 联系起来。
- **Neuronpedia 开源，内置 Eleuther 技术！**：可解释性平台 **Neuronpedia** 现已 [MIT 开源](https://x.com/neuronpedia/status/1906793456879775745)，并使用 Eleuther 的 `Delphi`（原 sae-auto-interp）作为其 **auto-interp server**。
   - 公告包括 [GitHub 仓库](https://github.com/hijohnnylin/neuronpedia)、[公共数据集](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/)的链接，以及一篇总结 Neuronpedia 功能的[博客文章](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) 。
- **利用 MMLU-pro 评估**：成员们确认 **MMLU-pro eval** 是使用 `test` 分割运行的，few-shot 示例源自 `validation` 分割，如 [config 文件](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml)中所示。
   - 用户可以通过任务 YAML 中的 `generation_kwargs` 向 `generate` 函数传递额外参数，以压缩 Key/Value (KV) caches 并实现 contrastive beam search。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **xAI 通过股票交换收购 X！**：Elon Musk 透露，**xAI** 在一笔全股票交易中收购了 **X** (Twitter)，估值 **xAI** 为 **800 亿美元**，**X** 为 **330 亿美元**。根据[这篇 CNBC 文章](https://www.cnbc.com/2025/03/28/elon-musk-says-xai-has-acquired-x-in-deal-that-values-social-media-site-at-33-billion.html)，此举旨在整合数据、模型、算力、分发和人才。
   - 此举被推测有助于 X 规避最初收购 Twitter 产生的债务利息，并改进 **Grok** 的数据抓取和训练。
- **Midjourney 进军 LLM 领域！**：以 AI 图像生成闻名的 **Midjourney** 正在转向 LLM，并与 NYU 共同发布了[一篇研究论文](https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/)，关于训练像 Llama 和 Mistral 这样的 LLM 以实现更具创造性的写作。
   - 这标志着 Midjourney 意图在图像生成之外实现多元化，并开发自己的计算和 AI 硬件。
- **GPT-4o 展示推理能力！**：**GPT-4o** 展示了推理能力，引发了关于它是正在开发的 [GPT-5 系统](https://fxtwitter.com/koltregaskes/status/1905907926331539794)一部分的猜测，并伴随着持续的工具和更新添加。
   - 一位成员兴奋地注意到，它甚至可以*在回答过程中决定开始进行推理*。
- **Meta 暗示 Llama 4 即将发布！**：据报道，三个新模型 **cybele, themis, and spider** 的表现似乎是针对 Arena 上的 elomaxxing 进行了优化，这可能预示着 **Llama 4** 发布候选版本即将到来。
   - 传闻 **Meta** 将在官方活动之前发布，效仿 **Llama 3** 在 4 月 18 日的发布，以避免在模型性能上被掩盖。
- **破解 OpenAI 代码：多尺度扩散？**：根据[这条推文](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492)，分析 **OpenAI 图像生成** 帧揭示了一个多尺度结构，证据倾向于交错潜空间自回归（interleaved latent autoregression）而非拉普拉斯金字塔（Laplacian pyramid），通过跨尺度的非因果扩散（non-causal diffusion）进行解码。
   - **OpenAI 图像生成** 中的光栅扫描似乎只是 UI，每一帧都通过从粗到细的多尺度扩散反映全局更新，而不是基于 patch 的 AR。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Ampere GPU 线程表现超出预期**：一位成员计算得出，拥有 96 个 SM 的 **Nvidia Ampere GPU** 理论上应支持 **12288 个线程**，但观察到性能提升一直持续到 **24576 个线程**。
   - 该成员正在分析 [Geohot 的 GPU Noob kernel](https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54) 以理解线程性能，并询问 kernel 延迟隐藏（latency hiding）是否允许在每个 SM 上并发调度两倍的核心数。
- **Triton 模拟的 Dot Scaled 降低了性能**：一位用户报告称，在 **H100** 上使用 Triton 模拟的 `dot_scaled` 函数，其默认向上转型（upcasting）为 `bf16` 的行为会损害性能，并参考了 [Triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html)。
   - 另一位用户询问关于在 Triton 中将整个矩阵加载到 **L1 cache** 并在单个 **SM** 上进行处理的问题，以及对同一矩阵的后续 `tl.load` 调用是否会从 **L1 cache** 而非 **HBM** 获取数据。
- **PTX 编译器编排内存访问**：一位成员对 **FlashAttention 中的内存访问模式**表示困惑，特别是关于为 **128-bit 内存传输**进行数据重塑（reshaping）的必要性，并引用了 **CUDA C Programming Guide** 的第 5.3 节。
   - 另一位成员澄清说，**PTX 编译器**管理寄存器中的数据布局，以确保线程可以通过一条指令向单个对齐的 gmem 地址写入 **128 位连续数据**，并建议使用 **Nsight Systems (nsys)** 和 **Nsight Compute (ncu)** 进行性能分析。
- **研究称 BFloat16 破坏了 RoPE**：一篇新论文（[When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476)）指出，**BFloat16** 会在 **RoPE** 中引入数值误差，从而损害其相对编码，即使在 **Float32** 中计算也是如此。
   - 该论文介绍了 **AnchorAttention**，这是一种即插即用的方法，可提高长文本性能，减少 50% 以上的训练时间，并保留模型的通用能力，支持 **FlashAttention** 和 **FlexAttention** 的代码已在 [GitHub](https://github.com/haonan3/AnchorContext) 上发布。
- **Apple Silicon 内存映射之谜**：一位成员询问了 Apple Silicon M 系列 GPU 的片上缓存和内存层级，寻求与 NVIDIA A100 内存映射等效的 Apple 资料，并链接了一篇关于 [Apple M-Series SoC 的论文](https://arxiv.org/abs/2502.05317v1)。
   - 讨论强调，Apple 不像 NVIDIA 那样公开某些 GPU 细节，因此难以确定具体的缓存数值，但论文提到了 M4 芯片中的 **L1 cache（每个核心 192 KB）** 和高达 **24 MB 的共享 L2 cache**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Shear 通过 Softmax 扩展对齐专业知识**：Emmett Shear、Adam Goldstein 和 David Bloomin 创办了 **Softmax**，这是一家拥有 10 人的初创公司，专注于**有机对齐 (organic alignment)**，旨在融合人类与 AI 的目标，详见 [Core Memory 文章](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax)。
   - 该初创公司总部位于旧金山，从自然和智能系统中汲取灵感，以实现其对齐目标。
- **马斯克将 xAI 与 X 合并**：埃隆·马斯克宣布 **xAI** 正与 X 合并，旨在将 AI 能力和专业知识与 X 的影响力相结合，详情由 [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition) 报道。
   - 此次合并旨在利用 X 广泛的平台来增强和部署 xAI 的先进 AI 技术。
- **GPT-4o 的图像生成是前端把戏？**：一位用户发现 **GPT-4o** 的逐行图像生成其实是浏览器端的动画，服务器仅发送了 **5 张中间图像**，且 patch size 为 **8**，根据[这条推文](https://x.com/jie_liu1/status/1905761704195346680)显示。
   - 这种前端错觉营造了逐渐生成图像的效果，而无需承担逐行生成每一层的计算成本。
- **Gemini 2.5 Pro：现已面向所有人开放**：由于 TPU *运行火热*，**Gemini 2.5 Pro**（实验版）现已向所有 Gemini 用户开放，正如 [GeminiApp 的 Twitter](https://fxtwitter.com/GeminiApp/status/1906131622736679332) 所宣布的那样。
   - 扩大访问权限允许更多用户测试该模型，尽管免费用户有速率限制。
- **MiniMax 通过 Audio Speech-02 实现文本转语音**：**MiniMax AI** 推出了 **Speech-02**，它可以立即将任何文件或 URL 转换为逼真的音频，支持 **30 多种语言**且具有地道风格，支持无限声音克隆和亚秒级流式传输，详见 [MiniMax 的 Twitter](https://fxtwitter.com/MiniMax__AI/status/1906720764885180775)。
   - 该模型单次输入支持高达 20 万字符，非常适合制作有声读物和播客。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Lattner 的遗产：从 LLVM 到 Modular AI**：[Chris Lattner](https://nondot.org/sabre) 分享了他的作品列表，强调了他在 **LLVM**、**Clang**、**Swift**、**MLIR** 和 **CIRCT** 方面的贡献，以及他在 **Modular AI** 的角色。
   - 他的领导力延伸到了 **LLVM Foundation**，他在那里担任董事会成员，进一步巩固了他对现代编译器技术的影响。
- **Mojo REPL 面临弃用**：一个 Modular 论坛讨论链接指出了 [Mojo REPL 的弃用](https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna)，标志着该语言开发环境的转变。
   - **Jeremy Howard** 等成员极力推崇 Notebooks，不仅用于实验，还用于与 **Mojo** 一起打包。
- **Mojo 列表遭遇 Trait 对象 Segfault**：由于 Trait 支持不完善，用户在创建 Trait 对象列表（如 `List[Estimator]`）时遇到了段错误（[issue #4218](https://github.com/modular/max/issues/4218)）。
   - 建议的权宜之计是使用 `List[Variant[KNN, SVM]]`，并通过 `isa` 进行类型检查来调用方法，从而实现一种异构列表管理。
- **`def` vs `fn`：Mojo 语法大对决**：关于 Mojo 中 `def` 与 `fn` 的争论兴起，讨论 `fn` 是否应该因为其类型安全性和通过 Mypy 实现的有类型 Python 工作流而成为默认选项。
   - 虽然有些人认为 `def` 对初学者更友好，但一项功能请求建议[让 `def` 默认返回 None](https://github.com/modular/max/issues/4211)，以弥合 Mojo 和 Python 语法之间的差距。
- **DeepSeek 放弃 CUDA 转向 PTX 层**：成员们指出 [DeepSeek 的突破](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead)是通过**绕过 CUDA** 并直接访问 **PTX 层**（一种底层的类汇编编程接口）实现的。
   - 一位成员还表示 *NVIDIA 驱动程序不被视为 CUDA*，并且 **NVIDIA** 在其术语随时间的变化上*有点混乱且不一致*。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 用户对视频片段的需求**：用户正请求 **NotebookLM** 在使用视频作为来源时，在回复中包含**视频片段**以提供**视觉效果**，团队表示未来将启用**多模态输出 (multi-modal output)**。
   - 用户希望获得时间戳，以便他们能像使用 **Audible** 一样跳转并重听特定章节。
- **思维导图导出功能依然难以实现**：一位用户询问是否能以 **DOT 格式**导出**思维导图 (Mind Maps)**，或者发布一个带有 Google UI 的交互式小程序用于 **NotebookLM**。
   - 遗憾的是，该功能目前尚不可用。
- **寻求集成 Android 分享系统**：用户渴望 **NotebookLM** 能够加入 **Android 分享系统**，理想情况下是通过一个专用 App 实现。
   - 该建议包括在分享菜单中选择 NotebookLM 时，能够自动在默认笔记本中进行搜索。
- **AI 语音在发音上遇到障碍**：一位用户正尝试改进 **NotebookLM** 中 **AI 语音**对单词的发音，特别是具有独特拼写的公司名称。
   - 用户希望通过向 AI 提供另一个具有正确发音的来源，使音频概览能够正确读出公司名称。
- **NotebookLM Plus 触及神秘限制**：一位 **NotebookLM Plus** 订阅者遇到了“已达到每日对话限制”的消息，即使在排除故障后仍阻碍了其使用。
   - 其他用户澄清说，Plus 用户不应面临任何限制。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex + SkySQL 推出 AI Agent**：根据其[公告](https://t.co/Kk7yCCyAuv)，LlamaIndex 与 SkySQL 合作，展示了如何构建无需代码即可实现可靠 **text-to-SQL 转换**的 **AI Agent 系统**。
   - LlamaIndex 现在集成了 **OpenAI Responses API**，支持复杂的**多 Agent 工作流 (multi-agent workflows)**。
- **遥测属性 (Telemetry Attributes) 添加标签**：一位成员寻求在使用 LlamaIndex 时传递自定义遥测属性的方法，特别是将用户 ID 附加到事件中。
   - 共享了一个使用 OpenTelemetry 和 [Colab notebook 示例](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing)的解决方案，以及 [Arize 的文档](https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once)。
- **多模态 OpenAI Agent 首次亮相**：成员们讨论了将图像作为聊天消息传递给 `OpenAIAgent`，其中一人建议利用 [OpenAI 的多模态能力](https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees)。
   - 另一人建议通过工作流 (workflows) 从头构建 Agent，或者修改 `chatmemorybuffer` 以将图像添加到请求中。
- **提出 Internet of Agents**：一位成员分享了一篇关于构建 **Internet of Agents (IoA)** 以解决 **Agentic AI** 中互操作性问题的文章，详情见 [[IoA]](https://www.anup.io/p/architecting-the-internet-of-agents)。
   - 文章建议开放标准可以解锁包括 **LlamaIndex** 在内的跨生态系统的可组合性。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **电子垃圾设备 vs Tinygrad Box**: 一位用户质疑一台改装的、配备 4x 4090 的电子垃圾推理机（链接[见此](https://detail.tmall.com/item.htm?abbucket=18&id=887290683136)）与 **Tinygrad Box** 相比的价值。
   - 针对该机器自制主板可能导致的 **PCIe 错误** 提出了担忧，估计其价值约为 1,000 美元加上 **4090s** 的成本。
- **Finite Field Assembly: CUDA 替代方案出现**: 一位用户分享了 [Finite Field Assembly](https://github.com/LeetArxiv/Finite-Field-Assembly)，这是一个专为有限域计算设计的 **CUDA 替代方案**，扩展了 **C89** 并支持递归计算。
   - 它利用素数的特性来并发地进行多个数组元素的乘法运算，例如在矩阵乘法中。
- **TinyGrad 内部机制公开！**: 一位用户分享了关于 **TinyGrad 内部机制** 的详尽笔记，可在[此处](https://xl0.github.io/tinygrad-notes/)查看，内容涵盖了 **UOps**、**ShapeTracker** 和 **Pattern Matcher**，灵感源自 **mesozoic-egg**。
   - 这些笔记通过对架构的深入探讨，对官方 [TinyGrad 文档](https://docs.tinygrad.org/) 进行了补充。
- **ORT CPUExecutionProvider 静默转换 Float16！**: 一位用户报告称，**ORT CPUExecutionProvider** 会针对 **float16 模型** 静默地将输入转换为 **float32**，使用 **float32** 进行计算，然后将输出转回 **float16**，这阻碍了 **numpy 移除** 工作。
   - 该用户建议添加一个 **envvar**（环境变量），以便在他们的 **ONNX** 设置中复制此行为，用于测试和调试。
- **VAE tinygraining 起飞！**: 一位成员一直在尝试使用 **tinygrad** 构建 **VAE**，并成功修改了 **Huggingface 的 Diffusers 库** 以适配 **tinygrad**。
   - **Stable Diffusion** 中使用的 **VAE** 现在已经可以运行，代码可在[此处](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py)获取。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **FP8 训练方案探索**: 大多数 **FP8 训练方案 (recipes)** 实际上是 **FP8 QAT**，除非你只能在不支持 FP8 的 GPU（如 A100）上训练，在这种情况下你可以直接使用 FP8 训练。
   - 下周五将举行 **Torchtune 答疑时间 (office hours)**，详情请见 [Discord 链接](https://discord.gg/Z9cuQgYX?event=1356379057373184155)。
- **Discord 时区功能终于搞定**: 成员们讨论了 Discord 内部针对活动的**时区自动转换**功能。
   - 一位成员分享了一个 [大脑迷因 GIF](https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-24411104)，以回应成功实现时区即时转换。
- **代码审查团队被要求加速**: 一位成员请求对 [PR #2441](https://github.com/pytorch/torchrec/pull/2441) 进行最终审查以加快合并进程，因为所有检查均已通过。
   - 另一位成员被提醒去审查该 PR。
- **GRPO 教授互联网搜索**: 分享了一篇关于使用 **GRPO** 教授互联网搜索的论文 [arxiv.org/pdf/2503.09516](https://arxiv.org/pdf/2503.09516)。
   - 项目的其他细节尚未透露。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R 展现极速性能**: **Command-R** 模型被确认为*最快且最通用*的模型，默认使用 **Command-A**，但 playground 不支持更改模型。
   - 用户被引导使用 **API** 来尝试不同的模型。
- **Aya-Vision 图片上传故障**: 用户报告在 playground 中使用 **Aya-Vision** 上传图片时出现错误，且在 Hugging Face 上的 [Aya Vision 演示](https://huggingface.co/spaces/CohereForAI/aya_expanse) 有时需要超过 30 秒才能响应。
   - 一位 Cohere 工作人员回应称，*他们将调查其后端的延迟问题。*
- **文档拼写错误导致 Bad Request**: 一位用户报告了 [Cohere 文档](https://docs.cohere.com/v2/reference/createfinetunedmodel) 中的一个拼写错误，其中 `train_epoch=1` 应该是 `train_epochs=1`，导致了 `BadRequestError`。
   - 一位 Cohere 工作人员确认了该拼写错误并发布了修复补丁。
- **独立游戏开发者转向 Cohere**: 一位主要使用 **C++** 结合图形和音频库的自学成才的独立游戏开发者介绍了自己，提到他们目前正在为朋友的**网络动画系列**开发一款**浏览器游戏**。
   - 这位开发者已经开始使用 **Cohere** 作为其他*大牌*模型的替代方案。

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Libre Wolf 安全性受关注**：成员们讨论了 **Libre Wolf** 相较于 **Firefox** 的安全性，并对其优势提出了质疑。
   - 对话并未给出定论，但强调了考虑浏览器安全性的重要性。
- **GPT4All 模型搜索功能缺失**：一位用户报告在搜索 **GPT4All 模型**时遇到困难，并指出缺乏内置的搜索功能。
   - 一名成员澄清说，本地模型列表搜索在过去 2 年里一直不是 **GPT4All** 的功能，并提供了 GitHub 上 [模型列表](https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata) 的链接。
- **文档导入模型寻求建议**：一位成员请求推荐能够导入文档并回答问题的模型。
   - 另一名成员分享了包含官方翻译的 [GPT4All wiki](https://github.com/nomic-ai/gpt4all/wiki)，并建议对其他语言使用 Google Translate。
- **Llama3 8B Instruct 博客创作测试**：一位用户询问 **Llama3 8B Instruct** 是否适合根据视频课程创建博客文章和网页。
   - 讨论引发了关于 **.bin** 和 **.gguf** 文件之间的区别及其互换性的问题，但未就其是否适合写博客给出明确答案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic 的 `conint` 触发验证**：**Pydantic** 中的 `conint` 功能可以设置约束（例如 `conint(ge=1, le=10)`），但如果输出超出指定范围，则会抛出 **ValidationError**。
   - 一位成员请求 DSPy 在验证失败时动态生成示例并重新发送请求，但目前该功能并未按预期运行。
- **MIPROv2 用户遭遇 RateLimitErrors 困扰**：用户报告称，尽管在 Azure OpenAI 上使用 `gpt-4o-mini` 运行 MIPROv2 时设置了 `num_threads=1`，但由于 **MIPROv2.compile()** 会进行多次内部 API 调用，仍频繁出现 **RateLimitErrors**。
   - 建议添加带有 **sleep(30)** 间隔的重试逻辑，降低 `max_*_demos`，并升级到具有内置速率限制功能的最新 DSPy 版本。
- **速率限制规避方案阻碍优化**：用户发现，为了规避 **RateLimitErrors** 而减少 `max_bootstrapped_demos` 和 `max_labeled_demos` 会损害优化效果。
   - 他们建议 DSPy 应该有更好的内部机制来管理 API 调用频率，因为 MIPROv2 和 Copro 中的结构化提示如果因 API 截断或速率限制导致 LLM 返回空输出，可能会引发错误。
- **签名格式为 a,b -> c**：在 DSPy 中，签名被定义为 *"a, b -> c"*，其中 a、b 和 c 是具有实际意义的名称。
   - 优化器随后会生成提示并在数据集上运行，以确定性能最佳的提示。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **DeepMind 工程师将进行 AlphaProof 讲座**：Google DeepMind 的研究工程师 Thomas Hubert 将于 PDT 时间 3/31 上午 10 点展示 "**AlphaProof**：当强化学习遇到形式数学"，并在 [YouTube](https://www.youtube.com/live/3gaEMscOMAU) 进行直播。
   - 讲座将探讨计算机如何为 **Birch and Swinnerton-Dyer 猜想**等重大问题做出贡献；Hubert 拥有斯坦福大学数学硕士学位。
- **MOOC 讲座时间调整**：今天的 **LLM Agents MOOC** 讲座移至 **PST 时间上午 10 点**，以配合来自**英国**的演讲者。
   - 课程网站 ([llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)) 和 [Discord 服务器](https://discord.gg/NWVpQ9rBvd) 提供了 **LLM Agents MOOC** 的重要链接和讨论论坛。
- **讲座录像已上线**：之前 **LLM Agents MOOC** 讲座的录像可以在 [课程网站](https://llmagents-learning.org/sp25) 和 [此 YouTube 播放列表](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn) 中找到。
   - 课程测验是**基于完成情况**的，这意味着只要尝试回答，分数并不重要。
- **提供 AgentX 学分**：**AgentX** 提供学分资源，详情可在 [AgentX 网站](https://rdi.berkeley.edu/agentx/) 找到。
   - 针对希望获得 **AgentX** 学分的人员，信息收集表将于本周发布。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **TMLS 2025 开启演讲嘉宾征集**：[演讲嘉宾征集 (Call for Speakers)](https://www.linkedin.com/posts/toronto-machine-learning-summit_tmls2025-callforspeakers-ai-activity-7303505411800719361-z-V2?utm_source=share&utm_medium=member_ios&rcm=ACoAACF-hfwBzcfh2mYq928aQ3C0PDfox4I_I8s) 已为 2025 年 6 月举行的 **Toronto Machine Learning Summit (TMLS)** 开启。
   - **TMLS 2025** 拥有 **16 个专业方向 (tracks)**，包括 **Advanced RAG**、**Multimodal LLMs**、**AI Agents in Production**、**MLOps for Smaller Teams**、**Responsible AI Implementation** 以及 **GenAI Deployments**。
- **MLOps 关注小团队**：**Toronto Machine Learning Summit** 将设立专门为小团队设计的 **MLOps track**。
   - 该方向为这些团队提供了一个交流经验并从 **MLOps** 领域其他专家处获取见解的平台。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期保持沉默，请告知我们，我们将将其移除。


---

# 第二部分：分频道详细摘要与链接


{% if medium == 'web' %}

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1355255140776411308)** (626 条消息🔥🔥🔥): 

> `沙盒 Swirl 问题的额度退款，用于 WordPress 网站创建的 AI，额度管理，Manus AI 对比 Gemini 2.5` 


- **Swirl Bug 引发额度返还讨论**：一位用户询问关于沙盒 **Swirl 问题** 是否可以退回额度的最新进展。
- **Manus 创建基于代码的网站**：一位用户询问 AI 是否只能帮助制作代码类网站，因为他们目前使用的是 **WordPress** 和 **Figma**。
   - 成员们回答说，AI 可以像业务伙伴一样为你构建网站，或者创建优质的 **Next/React 网站**，并提供所有准备就绪的文件，让你直接部署到 Vercel。
- **巧妙的额度应急计划曝光**：一位用户分享了他们的额度管理策略，通过结合使用 **Deepseek R1**、**Claude Sonnet 3.7**，最后使用 **Manus AI** 来优化网站构建流程。
   - 有人指出，使用极其精准的 Prompt 也能显著提高额度使用效率。
- **网站工作流中的 GPTs 与 Manus**：一位用户抱怨 Manus 在 Beta 测试期间**收费**，并认为它应该适用于“普通人”，而不仅仅是 Prompt 专家。
   - 其他用户反驳称 **Prompt Engineering** 是必不可少的，Manus 优于市面上的其他 AI 选项，并建议尝试[这种方法](https://discord.com/channels/1348819876348825620/1355477259234054323/1356148702036623410)来提高**效率**。
- **Gemini 2.5：处理 Bug 的 Beta 猛兽？**：用户对比了 **Gemini 2.5 Pro** 与 **Manus AI** 在各项任务中的表现，指出 Gemini 在复杂分析、推理、多模态分析和代码编写方面可能更出色，且支持云端协作、更具成本效益，但**无法执行整个工作流**。
   - 还提醒大家查看[该解决方案](https://discord.com/channels/1355477259234054323)，了解如何减少额度消耗以及如何进行多次备份。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/ban-hammer-futurama-scruffy-gif-20750885">Ban Hammer GIF - Ban Hammer Futurama - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/maite-perroni-proactiv-hi-hello-gif-20314066">Maite Perroni Proactiv GIF - Maite Perroni Proactiv Hi - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/based-uh-hello-based-department-based-department-american-psycho-patrick-bateman-gif-22458382">Based Uh Hello Based Department GIF - Based Uh Hello Based Department Based Department - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/backtoschool-billymadison-adam-sandler-sing-gif-19688369">Backtoschool Billymadison GIF - Backtoschool Billymadison Adam - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/whats-up-sup-robin-williams-wazzup-gif-14541215">Whats Up Sup GIF - Whats Up Sup Robin Williams - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hi-hello-there-hello-sup-swag-gif-23881342">Hi Hello There GIF - Hi Hello There Hello - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://jmiivdli.manus.space/">Manus 指南 - 全面指南</a>: 未找到描述</li><li><a href="https://ucebdqhq.manus.space/">使用 Manus AI 进行迭代开发：全面指南</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1355255344997335292)** (859 messages🔥🔥🔥): 

> `Spider Model Analysis, Grok vs Gemini Performance, Coding Benchmarks and Model Evaluation, LLM Prompt Engineering, OpenAI's New Open-Weight Language Model` 


- **Spider 模型受到严密审视**：成员们讨论了 **Spider 模型** 的“冗长”和“创意”特质，有人质疑这是否仅仅是训练上的怪癖而非不同的参数规模，另一些人则报告了其与 **Phoebe**、**Themis** 和 **Cybele** 等模型相比结果不一致。
- **Grok 与 Gemini 在科学领域展开对决**：成员们讨论了 **Grok** 和 **Gemini** 的相对优势，一位成员断言 **Grok3** 在科学任务上仍然更胜一筹，甚至在 **arc-agi-1** 上表现优于 **R1**。
   - 其他人则指出这取决于用户的具体需求。
- **GPT-4o 的创意编码能力受到赞赏**：用户评测了 **GPT-4o**，称其在创意编码方面表现出色，在非思考模式下甚至优于 **GPT-4.5**、**DeepSeek V3-0324** 和 **Claude 3.7 Sonnet**。
   - 一位用户甚至给这些模型打出了 **9.5/10** 的评分，但仍认为不如 **Claude 3.7 Sonnet** (Thinking) 或 **DeepSeek R1**。
- **Sama 预告新的权重开放语言模型**：根据[这篇文章](https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19)，**Sam Altman** 预告了未来几个月将发布一款具有推理能力的强大新权重开放语言模型，并希望与开发者探讨如何使其效用最大化。
   - 该模型似乎在发布前将接受准备工作框架（preparedness framework）测试。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/renderfiction/status/1905998185962643767">来自 renderfiction (@renderfiction) 的推文</a>：Gemini 2.5 Pro 在 Three.js 中的物理模拟！所有这些都始于“one-shot prompts”，但我继续向 Gemini 提问以获得更好的结果。通过下方的 GitHub 克隆 👇 #threejs #Physics</li><li><a href="https://x.com/sama/status/1906793591944646898?t=Xw_DyPuHG0edzBlLvbUn3g&s=19">来自 Sam Altman (@sama) 的推文</a>：TL;DR：我们很高兴在未来几个月发布一款具有推理能力的强大新权重开放语言模型，我们想与开发者交流如何使其效用最大化：https://openai.com/op...</li><li><a href="https://www.reddit.com/r/Bard/comments/">Reddit - 互联网的心脏</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_oth">Reddit - 互联网的心脏</a>：未找到描述</li><li><a href="https://artificialanalysis.ai/models/gpt-4o-chatgpt-03-25">GPT-4o (2025年3月) - 智能、性能与价格分析 | Artificial Analysis</a>：对 OpenAI 的 GPT-4o (2025年3月, chatgpt-4o-latest) 的分析，并与其它 AI 模型在质量、价格、性能（每秒 token 数和首个 token 时间...）等关键指标上进行对比。</li><li><a href="https://siliconangle.com/2025/03/07/microsoft-reportedly-develops-llm-series-can-rival-openai-anthropic-models/>,">据报道微软正在开发可与 OpenAI、Anthropic 模型竞争的 LLM 系列 - SiliconANGLE</a>：据报道微软正在开发可与 OpenAI、Anthropic 模型竞争的 LLM 系列 - SiliconANGLE</li><li><a href="https://www.reddit.com/r/Bard/comments/1jo50hq/gemini_25_pro_will_also_be_a_nonthinking_model/">Reddit - 互联网的心脏</a>：未找到描述</li><li><a href="https://gemini.google.com/share/dd74a82eaa14">‎Gemini - 增强型鹈鹕自行车动画
</a>：由 Gemini Advanced 创建</li><li><a href="https://www.reddit.com/r/Bard/comments/1jnk395/new_moonhowler_model_on_arena_llm_appears_to_be/">Reddit - 互联网的心脏</a>：未找到描述</li><li><a href="https://manus.im">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1j7n2s5/manus_turns_out_to_be_just_claude_sonnet_29_other/">Reddit - 互联网的心脏</a>：未找到描述</li><li><a href="https://github.com/DataEval/dingo">GitHub - DataEval/dingo: Dingo: 一个全面的数据质量评估工具</a>：Dingo: 一个全面的数据质量评估工具 - DataEval/dingo</li><li><a href="https://huggingface.co/spaces/DataEval/dingo">Dingo - DataEval 提供的 Hugging Face Space</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1355261415526633484)** (898 messages🔥🔥🔥): 

> `Gemini 2.5 Pro, Cline vs. Cursor, Roo Code, Windsurf, Cursor Pricing`

- **Gemini 2.5 Pro 获赞，但面临工具调用问题**：用户讨论了 **Gemini 2.5 Pro** 模型，部分人称赞其性能和性价比，而另一些人则报告了在 Cursor 中使用工具时的问题，暗示 Cursor 可能会为了推广付费选项而故意阻碍其功能。
   - 尽管潜力巨大，一些用户发现 Gemini Pro 2.5 *在 Cursor 中并不擅长实际调用工具*，经常输出错误或无法运行的代码。
- **Cline 与 Cursor 之争升温**：讨论围绕 **Cline 的稳定性与效率** 与 **Cursor 的功能与 Bug** 展开。一些用户因其可靠性和直接的模型应用而更青睐 Cline，另一些用户则认可 Cursor 的语义搜索能力和实验性功能。
   - 一位用户表示 *Cline 感觉非常完善 (polished af)*，而另一位用户则表示他们 *担心 Roo Code 会毁掉我的整个代码库*。
- **Roo Code 受到关注**：多位用户正在探索 **Roo Code**，因其 **boomerang tasks** 和更好的上下文保留等功能，称其为 *Cline 的进化版*。但对其稳定性、回滚能力以及高昂的 Anthropic API token 消耗仍存顾虑，导致一些人称其为 *vibe coding* 的选择。
   - 尽管好评不断，一位用户表示：*如果 Roo Code 还没实现这个功能，那对我来说它还没准备好*。
- **Windsurf 作为 Cursor 的替代方案**：用户将 **Windsurf** 视为潜在的 Cursor 替代品，看重其终极计划、终端/服务器任务的稳定性，并提到了嵌入式浏览器，方便与 AI 共享元素信息；然而，人们也担心其有限的 Context Window、模型实际能执行的操作，以及性价比可能不如一些常规方案。
   - 一位用户表示 *我一点也不喜欢 Windsurf，它的 Context Window 似乎更小*，而其他人则指出 Windsurf 似乎稳定性更好。
- **Context Window 与定价**：成员们对 Cursor 基于使用量的定价、Token 限制以及达到使用限制后模型质量/效率下降感到沮丧，这导致一些人开始探索 **Cline 或 Roo** 等替代助手，因为它们可以利用 OpenRouter 或 AI Studio 等服务获得完整的 Context Window 和更低的成本。
   - 一位用户表示 *在 Cursor 上使用 Claude Max 实现相同功能大约需要 2 美元*，随后在谈到替代方案时说 *所以价格降低了 10 倍*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/raizamrtn/status/1906727510601355393?s=46&t=ggmESCIXF0nYw8_kshHz7A">来自 Raiza Martin (@raizamrtn) 的推文</a>：试用 Gemini Code：一个由 Gemini 2.5 Pro 驱动的终端编程助手。💜 http://geminicodes.co 💜发行说明：→ 开始使用非常简单，只需在虚拟环境中运行 `pip install gemini-code`...</li><li><a href="https://x.com/stefcodes/status/1906122522644377788">来自 Stefan Meyer (@stefcodes) 的推文</a>：@angelmercedes @7etsuo @jaivinwylde +1 给 Roo，相比 Cursor，它给了我 100 倍更好的结果</li><li><a href="https://docs.cursor.com/settings/models">Cursor – Models</a>：未找到描述</li><li><a href="https://docs.cursor.com/troubleshooting/common-issues#networking-issues-http">Cursor – Common Issues</a>：未找到描述</li><li><a href="https://prompt.16x.engineer/">16x Prompt - 具备高级上下文管理的 AI 编程</a>：16x Prompt 是一款先进的 AI 编程工具。通过多个 LLM API 集成来管理代码上下文、自定义提示词并更快地交付功能。</li><li><a href="https://fxtwitter.com/adonis_singh/status/1906372453086937422">来自 adi (@adonis_singh) 的推文</a>：sonnet 3.7 vs gemini 2.5 pro - 构建 ChatGPT UI。左侧：claude 3.7 sonnet thinking；右侧：gemini 2.5 pro。我们有了新的 UI 之王！！</li><li><a href="https://www.cursor.com/pricing">定价 | Cursor - AI 代码编辑器</a>：选择适合您的方案。</li><li><a href="https://docs.cursor.com/context/@-symbols/@-docs">Cursor – @Docs</a>：未找到描述</li><li><a href="https://forum.cursor.com/t/guide-maximizing-coding-efficiency-with-mcp-sequential-thinking-openrouter-ai/66461/38?u=kleosr">[指南] 通过 MCP Sequential Thinking 和 OpenRouter AI 最大化编程效率</a>：感谢详细的见解——非常感激。我一直在实施最新的更新日志，并测试 Cursor 发布的 0.48 版本，该版本显著优化了规则...</li><li><a href="https://aistudio.google.com/app/apikey">未找到标题</a>：未找到描述</li><li><a href="https://github.com/TowelDude/cursor-mcp-collector">GitHub - TowelDude/cursor-mcp-collector</a>：通过在 GitHub 上创建账户来为 TowelDude/cursor-mcp-collector 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/ChatGPTCoding/comments/1jn36e1/roocode_vs_cline_updated_march_29/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://cloud.google.com/vertex-ai">Vertex AI 平台</a>：企业级、全托管的统一 AI 开发平台。访问并利用 Vertex AI Studio、Agent Builder 以及 160 多个基础模型。</li><li><a href="https://fireworks.ai/">Fireworks - 生成式 AI 的最快推理</a>：以极快的速度使用最先进的开源 LLM 和图像模型，或者使用 Fireworks AI 免费微调并部署您自己的模型！
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1355688349863510066)** (4 条消息): 

> `Perplexity Pro, Discord 改进, 智能路由` 


- **Perplexity 推出新的 Pro 功能**：Perplexity 即将推出全新的 "**Pro**"，其中包含现有的 **Pro + Reasoning models**。
   - 新的 **Pro** 在后续对话中将默认使用固定模型，而不是 "**Auto**"；这是一个备受期待的改动。
- **Perplexity Pro 具备智能路由**：**Pro** 现在还受益于**智能路由**，以确保在**速度和推理**之间取得最佳平衡。
   - Perplexity 正在相应频道征求反馈。
- **Discord 改进即将到来**：管理团队一直在收集反馈，并将在下周对 Discord 体验进行 **3 项改进**。
   - 这些改进包括：**1) 简化的入职流程**，**2) 更好的反馈转达方式**，以及 **3) Pro 频道的可见性与访问权限**。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1355255143775604766)** (790 messages🔥🔥🔥): 

> `Deep Research High, Grok Rate Limits, Deepseek bribed, Comet Waitlist, OpenRouter` 


- **Deep Research High 仍然不存在**：Perplexity AI 上的 **"Deep Research High"** 层级目前并不存在，尽管部分用户认为自己正在使用它，但 **complexity dev** 今天确认了这一消息。
   - 一位用户还指出，Grok 每 2 小时提供 **5 次免费深度搜索**，同时也指出 *Grok 的速率限制（rate limits）非常严格*。
- **新的 Perplexity 模型即将推出 zzzz**：上周承诺的 **Comet 等候名单（waitlist）推送**以及可能增加的 **HIGH 模型**并未兑现。
   - 一位用户对频繁的更名表示沮丧，称 *将 DeepSeek 重命名为 Perplexity 推理模型并命名为 1776 是非常不道德的……没错，就是 DeepSeek 美国版。这到底是什么鬼*。
- **DeepSeek 被收买退出竞争？**：一位用户推测 **OpenAI 贿赂了 DeepSeek**，以阻止其修复网页搜索功能，从而扼杀竞争。
   - 另一位用户反驳了这一说法，称 *OpenAI 贿赂 DeepSeek 以“关闭网页搜索”毫无意义*，并引用了人才和 OSS 作为理由。
- **对新 UI 更改的烦恼**：用户对新的 UI 更改表示沮丧，特别是 **模型选择选项的移除** 和强制的“Auto”模式，一些人因此要求 **Perplexity Pro 退款**。
   - 一些用户推测，**自动模型选择**是 Perplexity 故意为之，目的是将用户推向更便宜的模型，这导致一些人认为 **Pro 模式变差了**，现在他们更推荐使用 2.5 Pro。
- **体育博彩中的 Pro Search 困扰**：成员们讨论了使用 **Perplexity AI 进行体育博彩**，这引发了关于 AI 在财务决策中不可靠性的警告。
   - 一位用户建议在 **管理账户中可以使用你偏好的 AI**，但补充说 *他们并没有专门针对此用途的 AI*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://manus.im/">Manus</a>：Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，让你在休息时完成一切。</li><li><a href="https://tenor.com/view/totoro-gif-24991987">Totoro GIF - Totoro - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of/">Reddit - 互联网的核心</a>：无描述</li><li><a href="https://tenor.com/view/smh-gif-smh-meme-smh-steve-harvey-i-can%27t-gif-13893533684179296052">Smh Gif Smh Meme GIF - Smh gif Smh meme Smh - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://gizmodo.com/why-does-chatgpts-algorithm-think-in-chinese-2000550311">为什么 ChatGPT 的算法会用中文“思考”？</a>：OpenAI 的新推理模型正在做出一些奇怪且不可预测的行为。</li><li><a href="https://www.getmerlin.in/chat/share/47ebc788-d134-4019-9650-171aa42fc3ef">给我一份关于 mavuika 外观的详细描述</a>：由匿名用户分享于 2025 年 3 月 31 日</li><li><a href="https://ahrefs.com/traffic-checker/?input=https%3A%2F%2Fwww.perplexity.ai%2Fdiscover&mode=exact">网站流量检查器：估算任何站点的流量</a>：深入挖掘任何网站的流量数据，为你的网站寻找增长机会。试用 Ahrefs 流量检查器的免费版本。</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1jm2ekd/message_from_aravind_cofounder_and_ceo_of">Reddit - 互联网的核心</a>：无描述</li><li><a href="https://huggingface.co/blog/open-deep-research">开源 DeepResearch – 解放我们的搜索 Agent</a>：无描述</li><li><a href="https://github.com/sentient-agi/OpenDeepSearch">GitHub - sentient-agi/OpenDeepSearch</a>：通过在 GitHub 上创建账户为 sentient-agi/OpenDeepSearch 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1355297019173933278)** (20 messages🔥): 

> `AI Pathfinding Quirk, Supercomputer, AI diagnoses celiac disease, Google authenticator UX, Self-hosted projects` 


- **蝙蝠 AI 寻路怪癖曝光！**：一个 [Perplexity AI Page](https://www.perplexity.ai/page/bat-ai-pathfinding-quirk-z3ihUFcfSSWKJXOqW9mQ9g) 讨论了 **bat AI** 的一个奇特现象。
   - 虽然没有提供进一步的信息，但用户可以研究这个 **AI quirk**。
- **超级计算机发现了某些东西！**：一个 [Perplexity AI Page](https://www.perplexity.ai/page/supercomputer-uncovers-hyperso-J88jGXHzRAeiRmYDdaDymQ) 提到了一项 **supercomputer** 的发现。
   - 更多详情需要访问该页面。
- **AI 诊断出乳糜泻**：一个 [Perplexity AI Page](https://www.perplexity.ai/page/ai-diagnoses-celiac-disease-8VHqfHlkTVa3QE5AB90Ynw) 讲述了 **AI** 诊断 **celiac disease**（乳糜泻）的情况。
   - 未提供进一步信息。
- **Google Authenticator 的 UX 退步了！**：一个 [Perplexity AI Page](https://www.perplexity.ai/page/google-authenticator-ux-regres-gn5atpKnQw.GTmqeirskSw) 讨论了 **Google Authenticator** 中 UX 的退步。
   - 鼓励用户去调查这些变化。
- **探索最佳 Self-Hosted 项目**：一个 [Perplexity AI Search](https://www.perplexity.ai/search/best-self-hosted-projects-zmdmGpAWR1S6e81ffGemaw) 尝试寻找 **最佳 self-hosted 项目**。
   - 感兴趣的用户可以查看该链接。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1355319696521564322)** (30 messages🔥): 

> `Sonar API performance, Structured outputs, Image Search, Search depth API, Prompt data retention` 


- **Sonar API 速度将得到提升**：成员们报告称，**最新版本的 Sonar** 响应时间比之前的版本明显更长，PPLX 已记录此问题，并将研究改进方案。
   - 另一位成员报告新版 sonar 的 **首个 token 响应时间（time to first token）为 2.25 秒**，而另一位成员则报告了 **1 分钟的等待时间**。
- **结构化输出（Structured Outputs）的层级限制已取消**：Perplexity AI [宣布](https://docs.perplexity.ai/guides/structured-outputs) **结构化输出现在对所有用户开放**，无论其层级如何，立即生效。
   - 公告指出，所有模型均支持 **JSON 结构化输出**，而 `sonar` 和 `sonar-reasoning` 模型目前同时支持 **JSON 和 Regex 结构化输出**。
- **API 即将支持图像搜索**：针对有关使用 API 进行图像搜索以查找类似产品的咨询，一名团队成员确认 **目前尚不支持图像搜索**，但很快就会推出。
   - 他们指出，API 确实提供了一种使用 `return_images=True` 参数来返回图像的方法。
- **用户请求用于搜索深度的 API 参数**：一位用户询问如何在 API 中指定 **搜索深度**（低、中、高），并表示在示例 cURL 请求中找不到该参数。
   - 一名成员回复称，搜索深度可以在请求期间作为额外的 body 传递，并指向了 [API reference](https://docs.perplexity.ai/api-reference/chat-completions)，并承诺将其添加到 cURL 请求示例中。
- **无数据保留政策**：针对有关 Prompt 和输出保留的问题，Perplexity 团队成员确认他们对 **API 实行 0 数据保留政策**。
   - 该成员澄清，此政策适用于他们那一端。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.perplexity.ai/api-reference/chat-completions),">未找到标题</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/guides/structured-outputs)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1355255157125808361)** (747 条消息🔥🔥🔥): 

> `Gemini 2.5 Pro, Grok vs Gemini, AI 图像生成, AI 能源消耗, Cursor & 代码生成` 


- **Gemini 2.5 Pro 编程能力引发讨论**：用户对 **Gemini 2.5 Pro** 的编程能力持不同意见，一位用户指出它在 *C++ 和 WinAPI 方面表现糟糕，且经常产生幻觉*，而另一位用户则认为它在 C++ 方面*非常扎实*，但在宏展开（macro expansion）方面表现不佳。
   - 其他人发现它在某些语言（如 Jax）以及详细的 CoT (Chain of Thought) 步骤方面表现出色。
- **Grok 不稳定的性能引发不满**：多位用户报告了 **Grok** 不稳定的性能，经历了频繁的强制登出和内部错误消息，以及**“思考模式（thinking mode）未按预期工作”**的情况。
   - 尽管存在这些问题，一些用户仍然认为 **Grok** *相当不错*，并继续在订阅 **ChatGPT Pro** 的同时订阅它。
- **新图像生成功能引发辩论**：用户正在尝试新的图像生成功能，虽然对新图像生成有共识，但一些人发现它存在故障（glitchy）且生成的文字模糊。
   - 尽管宫崎骏拒绝 AI 艺术，但许多用户和 AI 仍在模仿他的吉卜力风格。
- **AI 能源消耗受到质疑**：一些用户质疑 AI 是否真的消耗大量能源和水，指出*制作一个汉堡消耗的能量几乎是单次 OpenAI 查询的 6000 倍*，并认为 AI 间接用水的说法可以套用到很多事物上。
   - 其他人坚持认为 AI 消耗大量电力和水，因为数据中心的冷却系统是基于水的，且水分会蒸发并需要补充。
- **Cursor 与代码生成工具讨论**：成员们讨论了 Cursor 内部的代码生成和整体代码质量，包括其缺乏可定制性和受限的界面，但 Cursor 中的模型（如新的 **Gemini 2.5 Max 和 Claude 3.7 Max**）提供了全上下文（full context），但处于付费墙（paywalled）之后。
   - 一位成员询问 Cursor 是否能处理 1 万词的代码，得到的回复是：*即使提供多个文件，它也能修复超过一千行的大文件中的问题*。



**提到的链接**：<a href="https://www.facebook.com/share/r/16QWbFUeEe/">3.4 万次观看 · 3.4 万次互动 | 这是完整披露 🤯 | Krystle Channel</a>：这是完整披露 🤯。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1355347367481573457)** (67 条消息🔥🔥): 

> `ChatGPT 中的文件过期问题, 图像生成的速率限制, 报告潜在漏洞获取奖励, 伦理与使用政策` 


- **ChatGPT 文件过早过期！**：一位用户报告上传到 ChatGPT 的文件在**几分钟内就过期了**，尽管之前的会话很稳定，但这干扰了涉及法律和税务文件的复杂工作。
   - 另一位用户建议使用 **ChatGPT projects** 并将主要文件作为项目文件上传。
- **图像生成用户遭遇速率限制 (Rate Limiting)**：由于新图像模型发布以来负载极高，**Plus 用户**现在在图像生成上遇到了**速率限制**。
   - 由于新图像模型发布后的极端负载，已实施临时措施；新用户也无法在 **Sora** 上创建视频。
- **Bug 赏金猎人获利**：成员们讨论了 OpenAI 的 [Bug Bounty Program](https://openai.com/index/bug-bounty-program/)，报告“范围内（in scope）”的漏洞可以获得奖励。
   - 讨论强调了涉及的**伦理**问题以及遵守**服务条款 (Terms of Service)** 以避免账号被封禁的重要性，特别是涉及违规内容时。
- **用户账号访问权限岌岌可危**：一位用户分享说，他正在 ChatGPT 上测试 YouTube 上的一个理论，让 AI 互相交谈，这引发了对违反 OpenAI [使用政策 (Usage Policies)](https://openai.com/policies/usage-policies/) 的担忧。
   - 另一位成员指出，**违反政策**可能导致账号被暂停或终止，建议该用户仔细阅读条款。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1355255635335184545)** (114 条消息🔥🔥): 

> `Prompt Engineering 中的 Markdown，使用 @ 引入自定义 GPTs，SORA 与受版权保护的角色，用于创意任务的 O1 或 O3` 


- **Markdown 乱象：关于 AI 提示词格式化的辩论爆发**：成员们讨论了在 prompt-engineering 频道中使用 Markdown 的挑战和局限性，指出缺乏 Markdown 支持会阻碍有效的沟通和教育。
   - 一位成员认为，“禁止使用 Markdown 的规则纯粹是偷懒”，这阻碍了用户使用 **AI 所使用的语言**来教育他人，而其他人则指出并非所有人都理解 Markdown，且代码块增加了一层不必要的抽象。
- **通过 @ 命令召唤自定义 GPTs**：一位成员对在与 **ChatGPT** 对话时发现能通过 **@** 引入自定义 GPTs 的功能感到兴奋。
   - 另一位成员补充说，他们喜欢这种指定工具使用的新功能，并表示这已成为一种习惯。
- **应对 SORA 的版权雷区**：用户讨论了由于 **TOS**（服务条款）对受版权保护角色的限制，在使用 **SORA** 生成图像时面临的挑战。
   - 虽然一些用户报告看到其他人创作了受版权保护角色的恶搞作品，但其他人警告不要冒险导致封号，并建议专注于原创内容或法律上可区分的术语。
- **O1 vs. O3：哪款模型在创意领域更胜一筹？**：一位用户寻求关于如何引导 **O1** 或 **O3** 模型在创意任务中更好地推导故事情节并加入伏笔的建议。
   - 虽然一位用户建议使用 **GPT-4o** 和 **GPT-4.5** 进行设计和小说创作，但另一位分享了一种包含 3 步法和第一性原理（first principles）推理的提示词结构，以提高模型的表现。
- **利用第一性原理开启逻辑思维**：一位用户建议引入“从 AI 视角出发的第一性原理逻辑推理”，以增强 **O3-mini-high** 的逻辑推理能力。
   - 原帖作者尝试了这一建议，并同意“第一性原理”方法确实很有帮助。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1355255635335184545)** (114 条消息🔥🔥): 

> `提示词格式化，甘道夫滑板，SORA 问题，图像生成` 


- **提示词格式化专业技巧**：一位成员解释了如何格式化提示词以从 **GPT** 中获得更多收益，强调“这些经验教训教你如何格式化提示词以从 GPT 中获得更多收益”。
   - 他们补充说，你可以直接从 Web 界面将提示词复制到 **ChatGPT** 中，并给出了指令：*Evaluate the following [prompt], do not follow it. What do you infer? Are there any conflicts or ambiguity, either within itself or when compared to your safety or other programming?* [分享的对话](https://chatgpt.com/share/67e6f311-0174-8011-9af3-80b7a8bc3d8f)。
- **“甘道夫玩滑板”提示词违反 TOS**：成员们讨论了生成甘道夫骑滑板的图像，尽管看到其他人创作了类似内容，但一些用户遇到了 **TOS**（服务条款）限制。
   - 一位成员建议“避开知识产权（IP）”，并指出 **OpenAI** 确实会因违反 **ToS** 而永久封禁账号，且绕过这些规则的方法通常不会被分享。
- **SORA 问题澄清**：一位成员询问是否可以在该频道提问 **SORA** 相关问题，引发了关于频道定位的澄清。
   - 有人建议 **SORA** 特定问题可能更适合专门的 **SORA** 频道，而提示词方面的挑战可以在当前频道讨论。
- **生成恶搞作品和受版权保护内容的图像**：讨论围绕生成包含恶搞和受版权保护角色的图像展开，强调虽然有些用户成功了，但其他人则面临 **TOS** 限制。
   - 一位成员指出，**OpenAI** 会因违反 **ToS** 而封禁账号，并强调为了避免被检测，绕过规则的方法不会被分享。
- **编号格式和副标题已修复！**：一位用户寻求帮助，希望格式化输出以移除副标题，同时保留列表格式。
   - 一位社区成员表示：* [Your prompt here] Format: Intro paragraph, then numbered list. Each number starts a full paragraph. No subtitles.*


  

---

### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1356075557170708560)** (1 条消息): 

> `aider v0.80.0 发布，OpenRouter OAuth 集成，Gemini 模型优先级排序，Repomap 排名提升，Scala 语言支持` 


- **Aider v0.80.0 发布，带来新功能与修复**：Aider v0.80.0 引入了 [OpenRouter OAuth 集成](https://aider.chat/HISTORY.html)，优先支持 **Gemini 模型**，并提升了 **repomap 排名**，其中 Aider 自身编写了 87% 的代码。
   - 此版本还添加了 `Ctrl-X Ctrl-E` 快捷键，用于在外部编辑器中编辑输入缓冲区，同时还包含其他改进和错误修复。
- **OpenRouter OAuth 简化模型访问**：如果在未提供模型和密钥的情况下，Aider 现在提供 [与 OpenRouter 的 OAuth 集成](https://aider.chat/HISTORY.html)，从而简化了访问模型的过程。
   - 当设置了 `OPENROUTER_API_KEY` 但未指定模型时，它会根据免费/付费层级状态自动选择 OpenRouter 默认模型。
- **Gemini 模型获得优先级**：最新的 Aider 版本在设置了 `GEMINI_API_KEY` 时会优先使用 `gemini/gemini-2.5-pro-exp-03-25`，如果配置了 `VERTEXAI_PROJECT` 则优先使用 `vertex_ai/gemini-2.5-pro-exp-03-25`，从而增强了模型选择。
   - 这些设置确保用户能够根据其环境变量利用最合适的 Gemini 模型。
- **Repomap 排名获得提升**：[Repomap 排名](https://aider.chat/HISTORY.html) 针对路径组件与聊天中提到的标识符相匹配的文件进行了改进，使定位相关文件变得更加容易。
   - 此外，Scala 语言获得了 repomap 支持，进一步扩大了支持的语言范围。
- **Ctrl-X Ctrl-E 快捷键用于外部编辑器访问**：用户现在可以使用新的 [Ctrl-X Ctrl-E 快捷键](https://aider.chat/HISTORY.html) 在外部编辑器中编辑当前输入缓冲区，改进了编辑工作流。
   - 该功能由 Matteo Landi 贡献，提供了一种利用熟悉的文本编辑器进行输入的便捷方式。



**提到的链接**：<a href="https://aider.chat/HISTORY.html">发布历史</a>：关于 aider 编写自身代码的发布说明和统计数据。

  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1355256424632025118)** (785 条消息 🔥🔥🔥): 

> `修复 AI 生成的代码，Aider 增强，Gemini 2.5，OpenAI Agent SDK，Claude` 


- **在 boomer twitter 上发帖修复生成的代码**：一位成员在 *boomer twitter* (LinkedIn) 上发布了他们帮助修复 AI 生成代码的服务。
   - 另一位成员表示担忧，称 AI 可以轻松生成数千行代码，需要 AI 来清理这些“废料 (slop)”。
- **Gemini 2.5 与 Sonnet 的讨论升温**：成员们讨论了 [Gemini 2.5](https://aistudio.google.com/app/u/2/apikey) 与 Sonnet 在包括代码重写在内的各种任务中的优劣，结果各异。
   - 一位成员称赞 Gemini 2.5 能够 one-shot 将他们的服务器从 node 'http' 重写为 express，但另一位成员表示 *“我对 gemini 2.5 的评价是它极其不一致 (trashinconsistent)，且为了提供良好的 benchmark 数据而训练，但也可能是我使用方式不对。”*
- **Gary 用 GO 编码，通过 GDS 整理 Obsidian Vault**：一位成员分享了他们的 [GitHub 组织](https://github.com/Aider-AI/aider)，并详细介绍了他们用 **GO** 编写的许多应用程序，包括一个名为 *obsorter* 的工具，该工具使用 *Gary 十进制系统 (GDS)* 将他们的 Obsidian vault 文件分类到预定义目录并根据内容重命名。
   - 其他人对这个似乎充当知识管理 *“johnny 十进制系统”* 的系统表示钦佩。
- **DeepSeek 无法遵循指令，不像 Gemini**：一位成员抱怨说，从 **Gemini 2.5** 切换到 **DeepSeek** 后，发现 DeepSeek 无法遵循指令，称 *“我让它跳桥，它却建了两个小村庄”*，同时赞扬了 Gemini。
   - 其他人补充说，在使用新模型时，**Gemini 2.5** 的速率限制 (rate limits) 可能是最大的问题。
- **Aider Benchmarks 受到关注**：一位成员指出 [Aider benchmarks](https://aider.chat/docs/benchmarks.html) 在一段 [YouTube 视频](https://youtu.be/LmpNOY5sQuc?t=43) 中占据了核心位置，并认可了其作为工具的价值。
   - 这引发了讨论，认为 **Aider** 是某些人凭借对工具的深入了解以及如何从 LLM 交互中获得最佳效果而进行的精准提示 (prompting) 的产物。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/mahawaryas27492/status/1906794382659125625?s=46">来自 AI Purr-fessor (Yash) (@MahawarYas27492) 的推文</a>：惊人的 2.5 flash experimental 发布了。🔥 它非常聪明，在我测试的一些推理问题上表现优于 o3 mini high。它正在缓慢推出，所以你可能需要等待官方发布，我...</li><li><a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>：了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://tenor.com/view/techno-viking-viking-gif-26693787">Techno Viking GIF - Techno Viking Viking - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://liveswebench.ai/">LiveSWEBench</a>：未找到描述</li><li><a href="https://tenor.com/view/go-for-it-you-can-do-it-encourage-do-it-gif-14006408">Go For It You Can Do It GIF - Go For It You Can Do It Encourage - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/pov-you-giga-chad-chad-meme-gif-25615024">Pov You GIF - Pov You Giga Chad - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/sherlock-benedict-cumberbatch-hat-gif-15943210">Sherlock Benedict Cumberbatch GIF - Sherlock Benedict Cumberbatch Hat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/wedding-crashers-will-ferrell-what-a-loser-loser-laugh-gif-3957171">What A Loser GIF - Wedding Crashers Will Ferrell What A Loser - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/shhh-shush-silence-nose-gif-17895433">Shhh Shush GIF - Shhh Shush Silence - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/andrew-tate-stare-andrew-tate-andrew-tate-sigma-xafer-gif-10165002945664617941">Andrew Tate Stare Andrew Tate Andrew Tate Sigma GIF - Andrew tate stare Andrew tate Andrew tate sigma - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/andrew-tate-tate-why-gif-940321714429124603">Andrew Tate Why GIF - Andrew tate Tate Why - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/joe-biden-deal-with-it-cool-glasses-meme-gif-13473183379638803062">Joe Biden Deal With It GIF - Joe biden Deal with it Cool - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/usage/modes.html#askcode-workflow">聊天模式</a>：使用 code, architect, ask 和 help 聊天模式。</li><li><a href="https://tenor.com/view/thumbs-up-alright-not-bad-gif-7771888706215464379">Thumbs Up GIF - Thumbs Up Alright - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/llms/gemini.html">Gemini</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/wine-alcohol-red-will-ferrell-drinking-gif-5034418">Wine Alcohol GIF - Wine Alcohol Red - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/techno-point-gif-24022320">Techno Point GIF - Techno Point - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/mufeedvh/code2prompt/issues/107">无法执行 `cargo install` · Issue #107 · mufeedvh/code2prompt</a>：error[E0532]: expected a pattern, found a function call --> /Users/slu/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/code2prompt-3.0.0/src/main.rs:199:17 | 199 | Ok(_) => { | ^^ not a tup...</li><li><a href="https://cloud.google.com/docs/authentication/external/set-up-adc">未找到标题</a>：未找到描述</li><li><a href="https://github.com/joanrod/star-vector">GitHub - joanrod/star-vector: StarVector 是一个用于 SVG 生成的基础模型，它将矢量化转换为代码生成任务。StarVector 使用视觉语言建模架构，同时处理视觉和文本输入，以卓越的精度生成高质量的 SVG 代码。</a>：StarVector 是一个用于 SVG 生成的基础模型，它将矢量化转换为代码生成任务。StarVector 使用视觉语言建模架构，处理视觉和文本...</li><li><a href="https://github.com/Aider-AI/aider/issues/2979#issuecomment-2613554537">恢复聊天记录导致错误 / 聊天记录摘要功能不起作用 · Issue #2979 · Aider-AI/aider</a>：问题：我有一个相当长的聊天记录文件（80k tokens），但它提供了关于我正在构建的项目的大量有价值信息。上周当我使用具有大容量的模型时，它运行良好...</li><li><a href="https://youtu.be/LmpNOY5sQuc?t=43"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/solcloud/Counter-Strike/tree/master?tab=readme-ov-file#counter-strike-football---">GitHub - solcloud/Counter-Strike: 多人 FPS 游戏 - Counter-Strike: Football 🏉</a>：多人 FPS 游戏 - Counter-Strike: Football 🏉。为 solcloud/Counter-Strike 的开发做出贡献</li>

pment by creating an account on GitHub.</li><li><a href="https://aider.chat/docs/config/options.html#history-files>">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://useai.substack.com/p/beyond-prompts-why-context-management">超越提示词：为什么上下文管理能显著提升 AI 性能</a>：仅仅因为模型可以处理大量上下文，并不意味着它应该这样做。以下是如何以及为什么要更好地管理上下文窗口，以便更好地利用 LLM。</li><li><a href="https://arxiv.org/abs/2308.14508">LongBench：长上下文理解的双语多任务基准测试</a>：虽然大语言模型 (LLM) 在许多语言任务中表现出令人印象深刻的性能，但它们中的大多数只能处理几千个 Token 长的文本，限制了它们在更长场景下的应用...</li><li><a href="https://arxiv.org/abs/2307.11088">L-Eval：建立长上下文语言模型的标准化评估</a>：最近，人们对扩展大语言模型 (LLM) 的上下文长度越来越感兴趣，旨在有效处理单轮长输入或具有更广泛历史记录的对话...</li><li><a href="https://arxiv.org/abs/2502.05167">NoLiMa：超越字面匹配的长上下文评估</a>：最近的大语言模型 (LLM) 支持从 128K 到 1M Token 的长上下文。评估这些能力的一种流行方法是“大海捞针”(NIAH) 测试，它涉及检索...</li><li><a href="https://aider.chat/docs/config/api-keys.html">API 密钥</a>：为 API 提供商设置 API 密钥。</li><li><a href="https://aider.chat/docs/config/dotenv.html">使用 .env 进行配置</a>：使用 .env 文件为 aider 存储 LLM API 密钥。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>：如何使用 YAML 配置文件配置 aider。</li><li><a href="https://aider.chat/docs/troubleshooting/models-and-keys.html">模型与 API 密钥</a>：aider 是你终端里的 AI 结对编程助手
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1355257943444164748)** (150 messages🔥🔥): 

> `Gemini 2.5 Pro, Rate Limits, Aider Hooks, Architect mode improvements, MCP support` 


- **Gemini 2.5 Pro 使用情况和配额异常仍然存在**：用户报告了 **Gemini 2.5 Pro** 使用中的不一致性，API 控制台有时会分别显示 **2.5** 和 **2.0** 的使用情况，尽管 **Aider** 仅报告了 **2.5** 的使用，详见 [issue #3641](https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743)。
   - 一位成员提到 **2.0-exp** 是一个内部名称，一些人看到 **2.5** 被应用了不同的配额限制，并推测 **2.5** 可能在复用 **2.0** 的配额。
- **Aider 的缓存写入导致 Token 计数虚高**：一位用户观察到，在使用 **Sonnet** 时，**Aider** 不将缓存写入计为输入，导致显示的发送 Token 数翻倍（例如，发送 **12k**，缓存写入 **6.1k**）。
   - 该用户询问其他人是否遇到过类似情况，目前正在调查根本原因以确保准确的 Token 追踪。
- **Architect 模式的编辑循环令用户感到困扰**：一些用户报告了最近版本中的一个问题，即 **architect mode** 会陷入无限循环，在提供摘要后反复询问是否编辑文件，这可以通过 `/ask` 和 `/code ok` 绕过。
   - 一位成员指出，在配置文件中设置 `auto-accept-architect: false` 可以恢复到之前的行为，即在编辑前总是进行询问。
- **MCP 支持获得关注**：开发者对 **Aider** 中支持 **MCP (Model Collaboration Protocol)** 的兴趣日益增长，讨论围绕其减少模型锁定和促进 OSS 工具开发的潜力展开，如 [MCP Marketplace](https://github.com/cline/mcp-marketplace) 所示。
   - 一位成员提到了通过 `mcpm-aider` 实现的第三方集成，其他人则对内置支持以简化使用表示了兴趣，[PR #3672](https://github.com/Aider-AI/aider/pull/3672) 添加了初步支持。
- **用户寻求对大文件的部分读取功能**：用户正在寻求在 **Aider** 中实现**部分读取**的方法，以处理超出上下文限制的大文件，一些人建议使用 `/run` 命令配合 `head`、`grep` 或 `rag-cli` 等工具。
   - 一位成员分享了一个名为 [rag-tool](https://github.com/chadfurman/rag-tool) 的自定义 **RAG 工具**，该工具使用 Mastra Agent 构建，旨在从代码库中提取细节并处理大文件，可通过 `/run npm run agent` 在 Aider 中使用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/it%27s-a-slow-day-gif-17869747439397645052">It&#039;S A Slow Day GIF - IT&#039;S A SLOW DAY - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/cline/mcp-marketplace">GitHub - cline/mcp-marketplace: This is the official repository for submitting MCP servers to be included in Cline&#39;s MCP Marketplace. If you’ve built an MCP server and want it to be discoverable and easily installable by millions of developers using Cline, submit your server here.</a>: 这是提交 MCP server 以包含在 Cline 的 MCP Marketplace 中的官方仓库。如果你构建了一个 MCP server，并希望它能被数百万使用 Cline 的开发者发现并轻松安装，请在此提交你的 server。</li><li><a href="https://github.com/chadfurman/rag-tool">GitHub - chadfurman/rag-tool: Simple rag-tool using Mastra agents.  Designed to extract details about a codebase and to work with files.  Helpful for when the context is otherwise too big.</a>: 使用 Mastra agent 的简单 RAG 工具。旨在提取代码库详情并处理文件。在上下文过大时非常有用。 - chadfurman/rag-tool</li><li><a href="https://github.com/Aider-AI/aider/issues/3196">100% cpu freezing does not respond to ctrl c on latest release · Issue #3196 · Aider-AI/aider</a>: 问题：在渲染 Markdown/语法高亮时，进程挂起并占用 100% CPU。进程无响应并需要强制终止。这似乎是由于 Pygments 中的一种病态情况导致的...</li><li><a href="https://github.com/Aider-AI/aider/issues/3641#issuecomment-2762538743">Gemini 2.5 Pro or DeepSeek V3 0324 not showing in `/models /` · Issue #3641 · Aider-AI/aider</a>: 我一直在使用 `/models /` 来获取可用模型列表，并基于 Aidermacs 从列表中进行选择，我很高兴 Aider 支持 Gemini 2.5 Pro 和最新的 DeepSeek...</li><li><a href="https://github.com/Aider-AI/aider/issues/2227">Feature: Add GitHub Copilot as model provider · Issue #2227 · Aider-AI/aider</a>: 问题：你好！请添加 GitHub Copilot 作为模型提供商。应该可以这样实现：https://github.com/olimorris/codecompanion.nvim/blob/5c5a5c759b8c925e81f8584a0279eefc8a6c6643/lua/codecompani...</li><li><a href="https://github.com/BerriAI/litellm/pull/9079">Litellm dev 03 05 2025 contributor prs by krrishdholakia · Pull Request #9079 · BerriAI/litellm</a>: 标题、相关 Issue、类型：🆕 新功能、🐛 Bug 修复、🧹 重构、📖 文档、🚄 基础设施、✅ 测试。变更：[必需] 测试 - 如果 UI... 请附上任何新测试在本地通过的截图。</li><li><a href="https://github.com/Aider-AI/aider/pull/3672">Add MCP support by Antonin-Deniau · Pull Request #3672 · Aider-AI/aider</a>: 这是在 Aider 中对 MCP 的初步实现。目前支持在 ~/.aider.conf.yml 中通过以下配置添加 stdio MCP server：mcp: truemcp-servers:  - git-servermcp-server-com...
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1355598596388426213)** (8 messages🔥): 

> `量化对模型性能的影响，Interview Coder AI 工具` 


- **量化会降低模型准确度**：将模型从 **FP16** 转换为 **Q8** 会导致模型质量略有下降，而使用 Ollama 默认的 **Q4** 量化会进一步降低质量。
   - 有人指出，任何低于 **Q6** 的量化都会严重受损，特别是对于推理任务，但另一位成员表示，由于某些模型原生就是 **FP8**，**Q8** 量化*不应损失任何性能*。
- **Interview Coder 承诺颠覆技术面试**：[Interview Coder](https://www.interviewcoder.co/) 被宣传为技术面试的*隐形 AI*，旨在取代 Leetcode 等传统平台。
   - 该工具被描述为*彼得原理加速器*。



**提及链接**: <a href="https://www.interviewcoder.co/">Interview Coder - AI Assistant for Technical Interviews</a>: 未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1355288520390086657)** (536 messages🔥🔥🔥): 

> `DeepSeek-V3-0324 动态量化, RoBERTa 训练优化, 服务动态量化 Checkpoint, 4bit Gemma 3 12B 训练问题, Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit notebook`

- **DeepSeek-V3-0324 Dynamic Quantization 首次亮相**：**DeepSeek-V3-0324** 的动态量化版本已在 Hugging Face 发布，并附带了[本地运行指南](https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally)。
   - Unsloth 的 **Dynamic Quants** 采用选择性量化，相比标准位数量化提高了准确性。
- **Google Cloud Spot Instances 价格击败 Runpod！**：切换到 **Google Cloud** 后，工作负载速度提升了 2 倍，且成本比 Runpod 更低。
   - 成员指出，Google Cloud Spot Instances 比 Runpod 便宜多达 **60%**，且更稳定，而 Runpod 经常在 15 分钟后中断。
- **全员 Multi-GPU 支持 - 即将推出 (Soon™)**：Unsloth 团队表示，Multi-GPU 支持很快将面向所有人开放，但由于容量问题，Pro/Enterprise 版本目前处于暂停状态。
   - 共识是利用 Unsloth 目前的能力，将 Multi-GPU 功能提供给所有人。
- **HF x Unsloth 推理合作**：Unsloth 与 Hugging Face 合作开展了[此项协作](https://x.com/UnslothAI/status/1906726176556712318)，教用户如何使用 GRPO 微调 LLM。
   - 课程涵盖了奖励函数、GRPO 数学原理以及将 RL 应用于实际用例，并附带[教程](https://docs.unsloth.ai/basics/reasoning-grpo-and-rl/tutorial-train-your-own-reasoning-model-with-grpo)。
- **新的 Whisper Notebook 引起关注**：Unsloth 发布了一个用于训练 Whisper 的 Notebook，但如果没有预训练，情感标签（emotive tags）将无法工作。
   - 一位用户展示了如何使用 Orpheus 和 Unsloth 在仅 50k 个德语样本上进行微调，效果已经相当不错。参见[此处](https://x.com/SebastianB929/status/1906049996585099701)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/UnslothAI/status/1906726176556712318">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们与 @HuggingFace 合作，教你如何使用 GRPO 微调 LLM！了解：• Reward functions 及其创建 • GRPO 数学 + Colab 中的免费 Reasoning 训练 • 将 RL 应用于现实世界场景 ...</li><li><a href="https://x.com/SebastianB929/status/1906049996585099701">来自 SebastianBoo (@SebastianB929) 的推文</a>：仅在 5 万个德语样本上微调的 Orpheus 表现已经相当不错。使用了 Unsloth 和 QLoRA。目前仅支持随机说话者。遗憾的是，像 &lt;laughing&gt;、&lt;giggle&gt; 等表达方式还不能...</li><li><a href="https://x.com/UnslothAI/status/1905312972278563256">来自 Unsloth AI (@UnslothAI) 的推文</a>：听听 Orpheus-TTS 在小型 Text-to-Speech 数据集上使用自定义新语音和对话进行微调前后的对比。</li><li><a href="https://x.com/UnslothAI/status/1906460329292476732">来自 Unsloth AI (@UnslothAI) 的推文</a>：转发 @reach_vb：管它呢，685B 参数的 DeepSeek V3 0324 在 M3 Ultra 上本地运行，完全私密 🔥 由 llama.cpp 和 dynamic quants 驱动…</li><li><a href="https://www.nvidia.com/en-us/on-demand/session/gtcspring21-s31151/">GPU 计算的工作原理 | GTC Digital 2021 年 4 月 | NVIDIA On-Demand</a>：来听听 CUDA 首席架构师对 GPU 计算的介绍</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">初学者？从这里开始！ | Unsloth 文档</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/towardsai/ragbook-notebooks/blob/main/notebooks/Chapter%2010%20-%20FineTuning_a_LLM_Financial_Sentiment_CPU.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-deepseek-v3-0324-locally">教程：如何本地运行 DeepSeek-V3-0324 | Unsloth 文档</a>：如何使用我们的 dynamic quants 本地运行 DeepSeek-V3-0324 并恢复精度</li><li><a href="https://huggingface.co/docs/api-inference/en/tasks/text-generation">Text Generation</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-0324-GGUF">unsloth/DeepSeek-V3-0324-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://lenovopress.lenovo.com/lp2179-fine-tuning-llms-using-intel-xeon-cpus">使用 Intel Xeon CPU 微调 LLM</a>：大语言模型 (LLM) 已成为强大的业务工具，在包括问答 (QA)、文本摘要和翻译在内的各种任务中表现出色。然而，它们必须经过...</li><li><a href="https://unsloth.ai/blog/gemma3">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的新多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B、4B、12B 和 27B 尺寸。</li><li><a href="https://unsloth.ai/blog/qwq-32b#Tutorial%20QwQ">运行并微调带有 Bug 修复的 QwQ-32B</a>：使用 Unsloth 的 Bug 修复来微调和运行 Qwen 的新 QwQ-32B 模型。解决无限生成的问题。</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks)">Unsloth 文档</a>：未找到描述</li><li><a href="https://huggingface.co/DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF">DavidAU/L3.2-Rogue-Creative-Instruct-Uncensored-Abliterated-7B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://seb-v.github.io/optimization/update/2025/01/20/Fast-GPU-Matrix-multiplication.html">在 RDNA3 上优化矩阵乘法：50 TFlops，比 rocBLAS 快 60%</a>：简介</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit">unsloth/Qwen2.5-VL-32B-Instruct-unsloth-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/chat_templating">Templates</a>：未找到描述</li><li><a href="https://ai.darvinbox.click/">LiteLLM API - Swagger UI</a>：未找到描述</li><li><a href="https://huggingface.co/hitachi-nlp">hitachi-nlp (Hitachi, Ltd.)</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/tutorials-how-to-fine-tune-and-run-llms/tutorial-how-to-run-qwq-32b-effectively">教程：如何高效运行 QwQ-32B | Unsloth 文档</a>：如何通过我们的 Bug 修复高效运行 QwQ-32B，避免无限生成，并提供 GGUF。</li><li><a href="https://github.com/huggingface/transformers/issues/36822">Gemma 3 在 fp16 下损坏 · Issue #36822 · huggingface/transformers</a>：系统信息 transformers 版本：4.50.0.dev0 平台：Linux-6.8.0-39-generic-x86_64-with-glibc2.35 Python 版本：3.11.10 Huggingface_hub 版本：0.29.3 Safetensors 版本：0.5.3 Accelerate 版本...</li><li><a href="https://github.com/unslothai/llama.cpp">GitHub - unslothai/llama.cpp: C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户，为 unslothai/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggml-org/llama.cpp/blo">GitHub - ggml-org/llama.cpp/blo</a>

b/master/ggml/src/ggml-cuda/convert.cu#L6>">llama.cpp/ggml/src/ggml-cuda/convert.cu at master · ggml-org/llama.cpp</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号，为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/triton-lang/triton-cpu">GitHub - triton-lang/triton-cpu: Triton 的实验性 CPU 后端</a>: Triton 的一个实验性 CPU 后端。通过在 GitHub 上创建账号，为 triton-lang/triton-cpu 的开发做出贡献。</li><li><a href="https://github.com/intel/intel-extension-for-pytorch">GitHub - intel/intel-extension-for-pytorch: 一个用于扩展官方 PyTorch 的 Python 包，可以轻松在 Intel 平台上获得性能提升</a>: 一个用于扩展官方 PyTorch 的 Python 包，可以轻松在 Intel 平台上获得性能提升 - intel/intel-extension-for-pytorch
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1355615311004434442)** (98 messages🔥🔥): 

> `支持工具调用的 Gemma-3 替代方案，全量微调（Full finetuning）的挑战与解决方案，以 ML 为中心的数据集，训练与微调的显存需求对比，Llama 3.2 3B` 


- **寻找支持工具调用的 Gemma-3 替代方案**: 成员们正在寻找支持 **tool use** 的 **Gemma-3** 替代方案，并引用了官方 [Gemma 文档](https://ai.google.dev/gemma/docs/capabilities/function-calling) 中关于其支持函数调用（function calling）的说明。
   - 建议的替代方案包括 **Qwen 2.5**（任何尺寸）或 **Mistral Small 3.1**，并提醒 7B 以下的模型可能表现不佳。
- **OOM 问题困扰全量微调尝试**: 用户在使用 **Unsloth**、**Axololt** 和 **TorchTune** 在单 GPU 或多 GPU 上进行全量微调（full finetuning）时遇到了 **Out of Memory (OOM)** 问题。
   - 一位用户分享了在 **Unsloth Qwen2.5 14B** 上使用 LoRA 成功的经验，并寻求如何将其与全量微调结果进行对比的建议。
- **机器学习数据集搜寻开始**: 一位成员正在寻找以机器学习为中心的数据集，以便为 FOSS 仓库的用户微调一个帮助机器人，并分享了 2 个社区链接：[ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) 和 [ml_papers_arxiv](https://huggingface.co/datasets/ThatDataDude/ml_papers_arxiv)。
   - 另一位成员分享了另一个问答数据集：[ml-arxiv-papers-qa](https://huggingface.co/datasets/hanyueshf/ml-arxiv-papers-qa)。
- **全量微调所需的显存少于从头训练**: 一位用户询问了从头训练模型（尤其是人脸识别等任务）的显存需求，以及是否比微调需要更多显存。
   - 回复指出，与微调相比，从头训练（Training from scratch）需要显著更多的资源（500k+ 张图片，超过 16GB VRAM），因此建议他不要 *重复造轮子*。
- **树懒拥抱（Sloth Hug）表情包受到喜爱**: 一位成员在 🤗 服务器中添加了 <:slothhug:1257540335438008343> 表情符号，并分享了 [discord_sloth_hug.png](https://cdn.discordapp.com/attachments/1179039861576056922/1356196511813472356/discord_sloth_hug.png?ex=67ec58ad&is=67eb072d&hm=99d4d88369da4acb1a46b3daa6fe6d88b814ce029eac9d99c91b4900c99640d6&) 和 [sloth_huglove_large.png](https://cdn.discordapp.com/attachments/1179039861576056922/1356196512740282469/sloth_huglove_large.png?ex=67ec58ad&is=67eb072d&hm=36f432d9b23573e30cd429548f7b97336f8cee495f0519c12f9f862c6f708885&) 的链接。
   - 这引发了其他成员的热烈表情回应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(7B)-Alpaca.ipynb">Google Colab</a>: 未找到描述</li><li><a href="https://tenor.com/view/my-girl-seal-friendship-its-a-deal-seal-spit-gif-17005706">My Girl Seal Friendship GIF - My Girl Seal Friendship Its A Deal - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://ai.google.dev/gemma/docs/capabilities/function-calling">无标题</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/ThatDataDude/ml_papers_arxiv">ThatDataDude/ml_papers_arxiv · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers">CShorten/ML-ArXiv-Papers · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/hanyueshf/ml-arxiv-papers-qa">hanyueshf/ml-arxiv-papers-qa · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1355268580060696646)** (286 messages🔥🔥): 

> `Unsloth 文档更新，Llama3.2-1B GRPO 错误，Deepseek V3 推理缓慢，使用 Unsloth 微调 Aya-Vision，Qwen 模型的 Flash Attention`

- **Unsloth 文档敦促更新依赖项！**：一位成员建议更新 **Unsloth 文档**，以不鼓励在更新期间使用 `--no-deps`，因为这会导致问题，并分享了一个指向文档的[链接](https://docs.unsloth.ai/get-started/installing-+-updating/updating)。
   - 另一位成员确认标准更新程序也包含 `--no-deps` 标志，这表明文档中存在一个需要修正的潜在错误。
- **调试 Aya Vision 8B 维度不匹配问题**：成员们在处理使用 Unsloth 微调 **Aya-vision 8B** 时出现的 `ValueError: Image features and image tokens do not match` 错误，并参考了 [Qwen Vision Fine-tuning notebook](https://huggingface.co/CohereForAI/aya-vision-8b) 作为指南。
   - 经确定，`tokenizer` + `UnslothDataCollator` 无法正确调整图像大小，导致维度不匹配，并且 **AyaVisionProcessor 需要不同的消息格式**，该问题最终得到了解决。
- **排查 Llama3.2-1B GRPO 错误**：成员在对持续预训练的 **Llama3.2-1B** 模型执行 **GRPO** 时遇到错误，特别是与形状约束相关的 `torch.fx.experimental.symbolic_shapes.ConstraintViolationError`。
   - 调试步骤包括检查元模型与微调模型的配置，并验证 `unsloth_fixed` 参数的状态，这表明存在与模型与 Unsloth 实现兼容性相关的问题。
- **使用 Unsloth 进行 Mamba 微调存在问题**：一位成员报告无法让 **Mamba** 微调在 Unsloth 上正常运行，遇到了重定向功能的问题，还提到了 **RWKV-6 HF** 的失败。
   - 成员们讨论认为，虽然 RWKV-6 HF 似乎可以运行，但训练器没有执行任何操作，可能需要修改源代码；然而，**Mamba** 预计只需要一行代码更改即可运行。
- **由于 Assertion Error，Gemma3 的 GGUF 转换失败**：一位成员在尝试将持续预训练的 **Gemma 3** 模型保存或合并为适用于 **vLLM** 的 **Float16** 或 **GGUF** 格式时遇到 `AssertionError`，怀疑转换过程中存在 float32 类型转换问题。
   - 错误发生在 `unsloth_zoo/saving_utils.py` 中，特别是在创建 **LoRA 统计信息**期间，这表明模块数量或 LoRA 参数的一致性可能存在问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/fine-tuning-guide#avoiding-overfitting-and-underfitting">微调指南 | Unsloth 文档</a>：了解微调的所有基础知识和最佳实践。初学者友好。</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating/updating">更新 | Unsloth 文档</a>：要更新或使用旧版本的 Unsloth，请遵循以下步骤：</li><li><a href="https://colab.research.google.com/drive/1nft9qLA9m7s-4G8YgcSNGsO0CL8x1OmW#scrollTo=BRCcEg-9I-3S">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/aya-vision-8b">CohereForAI/aya-vision-8b · Hugging Face</a>：未找到描述</li><li><a href="https://www.kaggle.com/code/shivamgarg1999/qwen-finetuning-pipeline-peft-sgarg">qwen_finetuning_pipeline_peft_sgarg</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://www.kaggle.com/code/shivamgarg1999/qwen-finetuning-pipeline-peft-sgarg/edit/run/230573561">qwen_finetuning_pipeline_peft_sgarg</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://unsloth.ai/blog/gemma3#everything:~:text=Vision%20fine-tuning,truncating%20sequence%20lengths.)">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的新型多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B、4B、12B 和 27B 尺寸。</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-GGUF">unsloth/DeepSeek-R1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/aya_vision">AyaVision</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1178#issue-2610722849">DPO, ORPO - 梯度累积修复 · Issue #1178 · unslothai/unsloth</a>：目标：将梯度累积修复传播到 DPO - 这要困难得多，因为它需要完全重写 https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py</li><li><a href="https://github.com/unslothai/unsloth-zoo/pull/105">修复：rolandtannous 在编译模块中修复 SmolVLM 缩进错误 · Pull Request #105 · unslothai/unsloth-zoo</a>：解决 Unsloth issue #2179。SmolVLM 模型在与 Unsloth 一起使用时，会在生成的编译模块中导致缩进错误。问题描述：当使用 SmolVLM 模型（特别是 SmolVL...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/issues/6559#issuecomment-2678469573">qwen2-vl训练有bug，ValueError: Image features and image tokens do not match: tokens: 1468, features 1936,有人能回一下吗 · Issue #6559 · hiyouga/LLaMA-Factory</a>：提醒：我已阅读 README 并搜索了现有 issue。系统信息：llamafactory 版本：0.9.2.dev0 平台：Linux-5.10.134-16.101.al8.x86_64-x86_64-with-glibc2.35 Python 版本：3.10....</li><li><a href="https://unsloth.ai/blog/gradient">LLM 训练中的 Bug 修复 - 梯度累积</a>：Unsloth 的梯度累积修复解决了 LLM 训练中的关键错误。</li><li><a href="https://github.com/unslothai/unsloth/issues/2179">当在 smolvlm2 中使用 unsloth 时，生成的 unsloth_compiled_cache 文件导致缩进错误 · Issue #2179 · unslothai/unsloth</a>：我尝试在 smolvlm2 中使用 unsloth，但它一直抛出 "unexpected indentation error"。正如错误信息所示，原因是生成的 unsloth_compiled_cache 文件的第 481 行...</li><li><a href="https://github.com/unslothai/unsloth/pull/1289">shashikanth-a 添加了对 Apple Silicon 的支持 · Pull Request #1289 · unslothai/unsloth</a>：未优化。尚不支持 GGUF。从源码构建 Triton 和 bitsandbytes：`cmake -DCOMPUTE_BACKEND=mps -S .` 用于构建 bitsandbytes；`pip install unsloth-zoo==2024.11.4`；`pip install xformers==0.0.25`</li><li><a href="https://github.com/ggml-org/llama.cpp">GitHub - ggml-org/llama.cpp: C/C++ 中的 LLM 推理</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggml-org/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/issues/2204">torch._dynamo.exc.UserError: 目前不支持动态控制流。 · Issue #2204 · unslothai/unsloth</a>：我使用了 Phi 4 GRPO notebook 并将模型替换为 Phi 3 Mini 128k Instruct，必须禁用 use_vllm，但运行代码后导致 Traceback (most recent call last): File "/hom...
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1355512015090876588)** (2 条消息): 

> `OdysseyXL-V2.5 代码请求` 


- **请求共享 OdysseyXL-V2.5 代码**：一名用户请求获取 [open-neo/OdysseyXL-V2.5](https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5) 的代码。
- **另一个主题**：另一个摘要。



**提到的链接**：<a href="https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5">OdysseyXL - 一个 open-neo 集合</a>：未找到描述

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1355285518463533318)** (88 条消息🔥🔥): 

> `GRPO notebooks, reward function, llama 3.1 8b finetuning, ggml-org/llama.cpp quantization, Openllm leaderboard` 


- **请求重构奖励推理**：一名成员询问如何修改 **GRPO notebooks** 中的 **reasoning process**，得到的建议是直接修改 **reward function**。
- **Llama 3.1 微调对决**：一名成员使用 **similarity scores** 评估其微调后的 **Llama 3.1 8b** 模型，并寻求对其方法的验证。
   - 其他成员建议使用 **BLEU score** 或类似指标，而一些成员则警告不要仅依赖 **similarity scores**，因为模型具有随机性（stochastic nature）。
- **llama.cpp 量化探索解决难题**：一名成员分享了一个 [pull request](https://github.com/ggml-org/llama.cpp/pull/12511)，该 PR 为大多数支持的架构（除了 **Mamba**, **RWKV6**, **RWKV6QWEN2** 和 **T5**）增加了对 token-embedding 和 output-tensor 之外的其他张量进行量化的功能。
   - 另一名成员指出，这项工作旨在提高 **GGUF quants** 在不同 **bits-per-weight (bpw)** 下的准确性和能力，类似于 **ExLlama2's quants**。
- **潜空间验证消除真实性真空**：一名成员分享了他们的[第一篇论文](https://github.com/jacobwarren/Latent-Space-Verification-for-Self-Correcting-LLMs)，探讨 LLM 如何感知自己何时在产生 hallucinating（幻觉），以及一种在 **latent space** 进行 **self-correction** 的机制。
   - 另一名成员询问了用于检测 **hallucinations** 的指标，特别是在 **out-of-distribution scenarios**（分布外场景）中。
- **基准测试盛宴：击败糟糕基准测试的最佳选择**：一名成员就使用哪个 **leaderboard or eval** 来比较模型的综合性能寻求建议。
   - 另一名成员认为*不存在所谓的综合性能*，模型在不同的垂直领域表现各异。他们建议针对特定评估使用 **SWE bench**, **aider polygot**, **RULER** 和 **AIME**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/bill-nye-you-have-no-idea-you-literally-dont-know-what-youre-talking-about-science-guy-gif-4774360">Bill Nye You Have No Idea GIF - Bill Nye You Have No Idea You Literally Dont Know What Youre Talking About - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://github.com/ggml-org/llama.cpp/pull/12511">quantize: Handle user-defined quantization levels for additional tensors by EAddario · Pull Request #12511 · ggml-org/llama.cpp</a>：此 PR 增加了对 token-embedding 和 output-tensor 之外的其他张量进行量化的能力。它处理了大部分支持的架构，除了 Mamba, RWKV6, RWKV6QWEN2 和 T5，以避免...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1356278493066821672)** (2 条消息): 

> `自动充值问题, Stripe 元数据不匹配, 积分已添加` 


- **由于 Stripe 故障导致自动充值失败**：由于 **payment metadata** 的更改导致在未收到来自 **Stripe** 的预期数据时出现静默错误，自动充值功能暂时中断。
   - 该功能已通过回滚更改恢复，团队正在处理缺失的积分并改进系统以防止未来再次发生。
- **自动充值停机后积分到账**：导致自动充值停机的问题已完全解决，**所有缺失的积分**已添加到受影响的账户中。
   - 受影响的用户将收到有关解决情况的电子邮件通知。
- **根本原因：Stripe 数据格式和错误的日志记录器**：停机的根本原因是 **Stripe 的数据格式不匹配**，且由于自动化测试不足和错误的日志记录器而加剧。
   - 已实施加强的监控、错误跟踪和端到端测试以避免再次发生；遇到持续问题的用户应通过电子邮件联系团队寻求进一步帮助。

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1355256929928347799)** (402 messages🔥🔥): 

> `输出图像模型时间线，OpenRouter Prompt Caching，Agent Hustle，GPT-4o，免费模型速率限制` 


- **输出图像模型即将推出**：成员们讨论了输出图像模型的到来，期待将其集成到 **OpenRouter** 等平台中，涉及 **GPT-4o** 和 **Gemini** 等模型。
   - 一位成员表示，一旦这些模型可用，他们将直接切换到 **OpenRouter**，不再使用 **Gemini** 的原生服务。
- **OpenRouter 的 Prompt Caching 节省费用**：OpenRouter 支持 Prompt Caching 以节省推理成本，大多数供应商会自动启用该功能；Anthropic 则需要按消息启用，具体说明见[此处](https://openrouter.ai/docs/features/prompt-caching)。
   - 用户可以在 [Activity 页面](https://openrouter.ai/activity)或通过 API 查看缓存节省情况，*cache_discount* 字段表示通过缓存使用节省的费用。
- **Agent Hustle 项目概览**：一位成员分享了他们的项目 **Agent Hustle** 的细节，这是一个股票交易 LLM，利用 **TEE wallet** 在每笔交易中收取少量费用。
   - 该系统总共串联了约 **12 个函数调用 (function calls)**，示例见[此处](https://h.uguu.se/aeNHgFaf.png)。
- **关于速率限制 (Rate Limiting) 的担忧**：成员们报告在 **Google/Gemini-2.5-pro-exp-03-25:free** 上遇到了速率限制，一位用户收到的错误提示为 *Rate limit exceeded, please try again 45906 seconds later*。
   - OpenRouter 团队澄清说，速率限制可能源自 **Google** 或 **OpenRouter**，且指定供应商会限制 OpenRouter 进行有效负载均衡的能力；[查看此文档了解速率限制](https://openrouter.ai/docs/api-reference/limits)。
- **OpenRouter 增加 BYOK 费用**：当在 OpenRouter 上使用你自己的 **OpenAI API key** (BYOK) 时，OpenAI 收取的每笔生成费用将额外加收 **5% 的费用**，该费用将从用户的 OpenRouter 余额中扣除。
   - 此费用仅适用于供应商提供的额度，不适用于直接在 AWS 等上游供应商处使用的额度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/openai/chatgpt-4o-latest)">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/api/v1">Discord</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/api-reference/limits">API Rate Limits - 管理模型使用和配额</a>: 了解 OpenRouter 的 API 速率限制、基于额度的配额和 DDoS 保护。有效配置并监控您的模型使用限制。</li><li><a href="https://openrouter.ai/openai/gpt-4o">GPT-4o - API, Providers, Stats</a>: GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://openrouter.ai/docs/features/prompt-caching">Prompt Caching - 通过智能缓存优化 AI 模型成本</a>: 使用 OpenRouter 的 Prompt Caching 功能降低您的 AI 模型成本。了解如何在 OpenAI、Anthropic Claude 和 DeepSeek 模型中缓存和重用响应。</li><li><a href="https://openrouter.ai/settings/credits">OpenRouter</a>: LLM 的统一接口。为您的提示词找到最佳模型和价格。</li><li><a href="https://fal.ai/models/fal-ai/any-llm">Login || fal.ai</a>: 未找到描述</li><li><a href="https://community.openai.com/t/chatgpt-release-notes-2025-march-27-gpt-4o-a-new-update/1153887">ChatGPT — Release Notes: 2025-March-27 - GPT-4o 新更新</a>: OpenAI 刚刚对 GPT-4o 进行了更新。根据 OpenAI 帮助页面的内容：GPT-4o 感觉更加直观、富有创造力和协作性...</li><li><a href="https://openrouter.ai/docs/api-reference/overview#uploading-base64-encoded-images">OpenRouter API Reference - 完整文档</a>: OpenRouter API 的综合指南。了解请求/响应模式、身份验证、参数以及与多个 AI 模型供应商的集成。</li><li><a href="https://openrouter.ai/openai/chatgpt-4o-latest">ChatGPT-4o - API, Providers, Stats</a>: OpenAI ChatGPT 4o 由 OpenAI 持续更新，指向 ChatGPT 使用的当前版本 GPT-4o。因此它与 API 版本的 [GPT-4o](/models/openai/gpt-4o) 略有不同...</li><li><a href="https://hastebin.com/share/daqowijupu.python">Hastebin</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1355275439945285784)** (318 messages🔥🔥):

> `LM Studio Model Details Fetch Failed Error, VSCode integration with LM Studio, Intel NPU Usage with LM Studio, LM Studio tool use and web search, speculative decoding with LM Studio` 


- ****获取任务受挫：用户与“模型详情错误”作斗争****：一位用户在 Windows 11 上的 LM Studio 中遇到了 `Model details error: fetch failed` 问题，并尝试了各种修复方法，如使用 Hugging Face Proxy、手动更改主机名、调整 DNS 设置、使用 VPN 以及重新安装。
   - 其他成员建议检查防火墙问题、IPV6 问题或不支持的机器架构（仅支持 AVX 的 CPU），但该用户确认他们可以在浏览器和终端中访问 Hugging Face，并且已经尝试切换到 IPV4。
- ****Continue.dev 接入 LM Studio，实现丝滑的 VSCode 自动补全****：一位成员提到，你可以通过一个 [VSCode extension](https://www.continue.dev/) 将 LM Studio 连接到 VSCode，从而创建自定义的 AI 代码助手。
   - 他们强调了该平台在 AI 原生开发中的能力，包括 Tab 键自动补全和引用特定代码的能力。
- ****NPU 尚未就绪：LM Studio 缺乏 Intel Ultra 集成****：一位用户询问 LM Studio 是否可以利用其 Intel Ultra PC 中的 NPU，另一位成员回答说，目前还没有软件可以使用该 NPU。
   - 另一位成员指出 [Windows Studio Effects](https://support.microsoft.com/en-us/windows/windows-studio-effects-273c1fa8-2b3f-41b1-a587-7cc7a24b62d8) 等功能是使用 NPU 的 Windows 功能示例，并明确表示他们不知道有任何 LLM 使用它。
- ****LM Studio API：开启工具使用的关键****：成员们讨论了在 LM Studio 中启用工具使用（tool use）和网页搜索功能的选项，以及是否可以修改 LM Studio 应用程序的 UI。
   - 澄清了工具使用仅通过 [LM Studio API](https://lmstudio.ai/docs/app/api/tools) 提供，而非 ChatUI，这导致一些人考虑将修改 Open WebUI 作为替代方案。
- ****Kokoro TTS 与 Orpheus 争夺 LM Studio 文本转语音的霸主地位****：成员们询问了如何将文本转语音（TTS）模型与 LM Studio 集成，以寻求 OpenAI 语音能力的替代方案，一位用户链接了 [hexgrad/Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) TTS 模型作为一个选项。
   - 然而，有人提到 [CanopyAI's Orpheus](https://huggingface.co/canopylabs/orpheus-3b-0.1-ft) 是唯一可以在 LM Studio 中运行的 TTS（通过 API，而非在聊天界面中），并且可以使用 [这个仓库](https://github.com/isaiahbjork/orpheus-tts-local) 在本地配合 LM Studio 运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: 我们研究了一种新型语言模型架构，该架构能够通过在潜空间中进行隐式推理来扩展测试时计算。我们的模型通过迭代一个循环块来工作，从而展开...</li><li><a href="https://lmstudio.ai/docs/python">lmstudio-python (Python SDK) | LM Studio Docs</a>: 开始使用 LM Studio 的 Python SDK</li><li><a href="https://openrouter.ai/google/gemini-2.5-pro-exp-03-25:free">Gemini Pro 2.5 Experimental (free) - API, Providers, Stats</a>: Gemini 2.5 Pro 是 Google 最先进的 AI 模型，专为高级推理、编程、数学和科学任务而设计。通过 API 运行 Gemini Pro 2.5 Experimental (免费)</li><li><a href="https://lmstudio.ai/docs/python/llm-prediction/structured-response">Structured Response | LM Studio Docs</a>: 使用 Pydantic 模型或 JSON Schema 强制模型输出结构化响应</li><li><a href="https://support.microsoft.com/en-us/windows/windows-studio-effects-273c1fa8-2b3f-41b1-a587-7cc7a24b62d8">Windows Studio Effects - Microsoft Support</a>: 未找到描述</li><li><a href="https://huggingface.co/hexgrad/Kokoro-82M">hexgrad/Kokoro-82M · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/april-fool-gif-25270662">April Fool GIF - April Fool - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.continue.dev/">Continue</a>: 赋能开发者，AI 增强开发 · 领先的开源 AI 代码助手。您可以连接任何模型和任何上下文，在 IDE 内部构建自定义的自动补全和聊天体验</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://github.com/Draconiator/Forgematrix">GitHub - Draconiator/Forgematrix</a>: 通过在 GitHub 上创建账号来为 Draconiator/Forgematrix 开发做出贡献。</li><li><a href="https://github.com/openai/openai-python/issues/961>">openai/openai-python</a>: OpenAI API 的官方 Python 库。通过在 GitHub 上创建账号来为 openai/openai-python 开发做出贡献。</li><li><a href="https://github.com/ggml-org/llama.cpp/issues/11483">Feature Request: Qwen 2.5 VL · Issue #11483 · ggml-org/llama.cpp</a>: 前提条件：我正在运行最新的代码。如果可能，请同时注明版本。我仔细阅读了 README.md。我使用了与我的问题相关的关键词进行搜索，以确保我正在创建...</li><li><a href="https://youtu.be/9KKnNh89AGU">Build a LOCAL AI Web Search Assistant with Ollama</a>: 使用 Ollama 运行本地 LLM，在本视频中，我将向您展示如何编写一个本地 AI 网页搜索助手。拥有一个可以使用网页进行响应并提供最新...</li><li><a href="https://github.com/ggml-org/llama.cpp/tree/master">GitHub - ggml-org/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggml-org/llama.cpp 开发做出贡献。</li><li><a href="https://huggingface.co/canopylabs/orpheus-3b-0.1-ft">canopylabs/orpheus-3b-0.1-ft · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/canopyai/Orpheus-TTS">GitHub - canopyai/Orpheus-TTS: TTS Towards Human-Sounding Speech</a>: 迈向类人语音的 TTS。通过在 GitHub 上创建账号来为 canopyai/Orpheus-TTS 开发做出贡献。</li><li><a href="https://github.com/isaiahbjork/orpheus-tts-local">GitHub - isaiahbjork/orpheus-tts-local: Run Orpheus 3B Locally With LM Studio</a>: 使用 LM Studio 在本地运行 Orpheus 3B。通过在 GitHub 上创建账号来为 isaiahbjork/orpheus-tts-local 开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1355256786613178590)** (63 条消息🔥🔥): 

> `用于 ML 的 Epyc 系统，旧 PC 上的 LM Studio，使用 LM Studio API 保存上下文，Mac Studio vs 多 GPU 推理，分布式 LLM 推理` 


- **Epyc 系统在内存带宽上挑战 GPU**：配备高频 12 通道 **DDR5** 内存的新型 **Epyc 系统**可以实现接近 **600 GB/s** 的内存带宽，凭借其海量的内存容量，在 **LLM** 性能上足以媲美消费级 GPU。
   - 一位成员建议，**10-12k** 的预算就可以构建一台相当不错的 Epyc 机器，能够运行巨大的模型，为合理的推理速度和海量的上下文窗口提供了一种经济的解决方案，而且无需 GPU！
- **旧 PC 获得 LM Studio 助力**：一位成员报告称，通过使用带有 **CPU AVX2** 编译运行时的 **LM Studio**，成功在 2016 年的 Dell Inspiron 笔记本电脑（i7 6700HQ，32GB DDR3，集成显卡）上运行了中等规模的 **Qwen** 和 **Llama** 模型（6Q 量化）。
   - 他对这台旧笔记本依然“老当益壮”感到*惊讶*，并称赞 **LM Studio** *非常出色*！
- **LM Studio API 上下文处理**：在使用 **LM Studio API** 配合 Telegram 机器人时，为了保持对话上下文，用户必须将对话历史存储在变量中（例如 JSON 格式），因为 **API** 本身并不固有地保留上下文。
   - 建议使用 *unique-tg-user-id* 来存储对话，除非它是托管在一部*不断重启*的 PC 上。
- **Mac Studio 诱惑推理服务器构建者**：一位成员在考虑是构建一个带有多个 **Nvidia** 显卡的推理服务器，还是选择一台具有统一内存的 **Mac Studio**，并引用了[这段 YouTube 视频](https://www.youtube.com/watch?v=nwIZ5VI3Eus)。
   - 另一位成员支持选择 **Mac Studio**，理由是成本更低、耗电更少且 **RAM** 更多，并建议以 headless 模式 24/7 运行 **LM Studio**，并指出它支持 **MLX** 模型。
- **分布式推理项目涌现**：针对 **LM Studio** 是否支持多机协作的查询，有人提供了两个分布式 LLM 推理项目的链接：[exo](https://github.com/exo-explore/exo) 和 [distributed-llama](https://github.com/b4rtaz/distributed-llama)。
   - 这些项目旨在将家用设备连接成一个强大的集群以加速 **LLM** 推理，设备越多意味着速度越快。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/ryzenai">LM Studio on Ryzen AI</a>：在您的 PC 上运行 Llama、Mistral、Mixtral 和其他本地 LLM，充分利用 RyzenAI 硬件的强大性能。</li><li><a href="https://www.supermicro.com/en/products/motherboard/H11SSL-i">H11SSL-i | 主板 | Super Micro Computer, Inc.</a>：未找到描述</li><li><a href="https://github.com/exo-explore/exo">GitHub - exo-explore/exo: 在家中使用日常设备运行您自己的 AI 集群 📱💻 🖥️⌚</a>：在家中使用日常设备运行您自己的 AI 集群 📱💻 🖥️⌚ - exo-explore/exo</li><li><a href="https://github.com/b4rtaz/distributed-llama">GitHub - b4rtaz/distributed-llama: 将家用设备连接成强大的集群以加速 LLM 推理。设备越多意味着推理越快。</a>：将家用设备连接成强大的集群以加速 LLM 推理。设备越多意味着推理越快。 - b4rtaz/distributed-llama
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1355257705782182059)** (117 条消息🔥🔥): 

> `FxEmbed, MCP, Sam Altman WSJ, Replit v2, n8n`

- **WSJ 深度报道 Altman 被解雇事件**：**WSJ** 发表了一篇文章，详细介绍了 **Sam Altman** 被 **OpenAI** 董事会解雇背后的真实故事，指控其*在新版本发布的安全测试方面撒谎*（[存档链接](https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d)）。
- **Replit v2 令人印象深刻**：一位成员发现 **Replit v2 agent** 在原型设计和构建 MVP 方面表现非常出色，底层可能使用了 **Sonnet 3.7**，并且很容易提取并用于自己的后端。
   - 有人指出，**Replit** 可以直接访问日志、配置好的基础设施并设置日志记录，这使得整个过程非常顺畅；**Cursor** 更适合现有的部署，但托管基础设施赋予了 **Replit** 优势。
- **OpenAI 将推出权重开放模型**：**OpenAI** 计划在未来几个月内发布一个具有推理能力的权重开放（open-weight）语言模型，并正在征求开发者的反馈（[OpenAI 开放模型反馈](https://openai.com/open-model-feedback/)）。
   - 该公司计划在 **SF**、欧洲和 **APAC** 举办开发者活动，以收集见解并提供早期原型供实验。
- **Cursor 完成巨额融资**：**Cursor** 完成了 **6.25 亿美元**的融资，投后估值为 **96 亿美元**，由 **Thrive** 和 **A16z** 领投，**Accel** 作为新投资方加入（[推文](https://x.com/ArfurRock/status/1906768733135098360)）。
   - 这一估值是在引发了 *vibe coding* 这一流行语之后达成的，其估值在不到一年的时间内从 **4 亿美元**增长到 **25 亿美元**，再到潜在的 **100 亿美元**。
- **Etched 进军 ASIC 领域**：首个 Transformer **ASIC** 公司 **Etched** 以 **15 亿美元**的估值完成了未公开的 **8500 万美元**融资，此前曾经历了两轮估值分别为 **5 亿美元**和 **7.5 亿美元**的隐身期融资（[推文](https://x.com/ArfurRock/status/1906756943349260682)）。
   - **Etched** 的芯片 **Sohu** 运行 **Llama 70B** 的速度超过 *每秒 500,000 个 token*，一台 8xSohu 服务器可替代 160 台 H100。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://runwayml.com/research/introducing-runway-gen-4">Runway Research | 介绍 Runway Gen-4</a>：未找到描述</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q">来自 Sewoong Oh (@sewoong79) 的推文</a>：我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 配对时，ODS 在网络搜索方面的表现优于 OpenAI 的专用模型 GPT-4o-Search...</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882">来自 Amazon Science (@AmazonScience) 的推文</a>：了解 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，可以将稳健的步骤组合成复杂的工作流；处理从预订到 QA 的一切事务...</li><li><a href="https://x.com/TheXeophon/status/1906654834255954049]">来自 Xeophon (@TheXeophon) 的推文</a>：非常激动能分享我一直在做的事情 👀 你知道那种痛苦：你想使用一个项目，但它具有 GPL 或 CC-by-NC 许可证 😭 我们努力工作，我们的 AI Agent 可以转换任何仓库...</li><li><a href="https://fxtwitter.com/peterwildeford/status/1906089368613490736">来自 Peter Wildeford 👊 🇺🇸 🔥 (@peterwildeford) 的推文</a>：这篇《华尔街日报》的文章如果属实，包含了关于 @OpenAI 和 @sama 的一些重磅炸弹 💣‼️ 据称 Sam Altman 显然多次对不同的人撒谎，例如 Altman 就有关...的事宜向董事会撒谎。</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898">来自 Sam Altman (@sama) 的推文</a>：简而言之：我们很高兴能在未来几个月发布一款具有推理能力的强大新型开源权重语言模型，我们希望与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://x.com/sama/status/1906793591944646898">来自 Sam Altman (@sama) 的推文</a>：简而言之：我们很高兴能在未来几个月发布一款具有推理能力的强大新型开源权重语言模型，我们希望与开发者交流如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://runwayml.com/gen-4-bts">Gen-4 幕后花絮</a>：完全使用 Gen-4 制作的短片和音乐视频合集，旨在测试模型的叙事能力。</li><li><a href="https://x.com/rauchg/status/1906814800426086861?s=46">来自 Guillermo Rauch (@rauchg) 的推文</a>：我们正在构建一个用于运行任意计算的 API，针对 Agentic AI 用例和长时间运行的任务。是的，它可以运行服务器。由支持我们每日 100 万次以上 @vercel 构建的基础设施提供动力，并经过优化...</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360]">来自 Arfur Rock (@ArfurRock) 的推文</a>：Cursor 融资结束 —— 由 Thrive 和 A16z 领投，投后估值 96 亿美元，融资 6.25 亿美元。Accel 是新加入的投资者。ARR 为 2 亿美元，较 2024 年 11 月 25 亿美元估值融资时增长了 4 倍。ARR 倍数与上一轮持平，为 50 倍。引用 Abe Brown 的话...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1906654834255954049">来自 Xeophon (@TheXeophon) 的推文</a>：非常激动能分享我一直在做的事情 👀 你知道那种痛苦：你想使用一个项目，但它具有 GPL 或 CC-by-NC 许可证 😭 我们努力工作，我们的 AI Agent 可以转换任何仓库...</li><li><a href="https://x.com/demishassabis/status/1906664622226083922?s=46">来自 Demis Hassabis (@demishassabis) 的推文</a>：很高兴宣布 @IsomorphicLabs 已筹集 6 亿美元，以加速我们有朝一日在 AI 的帮助下解决所有疾病的使命。我长期以来一直认为，改善人类健康是最重要的...</li><li><a href="https://x.com/TheXeophon/status/1906654834255954049">来自 Xeophon (@TheXeophon) 的推文</a>：非常激动能分享我一直在做的事情 👀 你知道那种痛苦：你想使用一个项目，但它具有 GPL 或 CC-by-NC 许可证 😭 我们努力工作，我们的 AI Agent 可以转换任何仓库...</li><li><a href="https://fxtwitter.com/stevenheidel/status/1906797154301329845">来自 Steven Heidel (@stevenheidel) 的推文</a>：我们今年将发布一个可以在你自己的硬件上运行的模型。引用 Sam Altman (@sama) 的话，简而言之：我们很高兴能在未来几个月发布一款具有推理能力的强大新型开源权重语言模型...</li><li><a href="https://x.com/jie_liu1/status/1905761704195346680">来自 Jie Liu (@jie_liu1) 的推文</a>：在破解 GPT-4o 的前端后，我有了惊人的发现：💡 用户看到的逐行图像生成效果只是浏览器端的动画（纯前端技巧）🔦 OpenAI 的服务器发送了 o...</li><li><a href="https://x.com/peterwildeford/status/1906089368613490736?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ">来自 Peter Wildeford 👊 🇺🇸 🔥 (@peterwildeford) 的推文</a>：这篇《华尔街日报》的文章如果属实，包含了关于 @OpenAI 和 @sama 的一些重磅炸弹 💣‼️ 据称 Sam Altman 显然多次对不同的人撒谎，例如 Altman 就有关...的事宜向董事会撒谎。</li><li><a href="https://x.com/ArfurRock/status/1906768733135098360">来自 Arfur Rock (@ArfurRock) 的推文</a>：Cursor 融资结束 —— 6.25 亿美元，估值为 $...</li>

<li>投后估值 96 亿美元，由 Thrive 和 A16z 领投。Accel 是新加入的投资者。ARR 为 2 亿美元，较 2024 年 11 月 25 亿美元融资轮增长了 4 倍。ARR 倍数与上一轮持平，保持在 50 倍。引用 Abe Brown ...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682">来自 Arfur Rock (@ArfurRock) 的推文</a>：🚨新独角兽预警 —— Etched，全球首款 Transformer ASIC。在经历了 5 亿美元和 7.5 亿美元的两轮隐身融资后，又以 15 亿美元的估值完成了未公开的 8500 万美元融资。7.5 亿美元那一轮就在大约 2 个月前。引用...</li><li><a href="https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo">最近的推理研究：GRPO 调整、基础模型 RL 以及数据策应</a>：在汹涌的推理研究浪潮中，我推荐值得阅读的论文。</li><li><a href="https://x.com/sewoong79/status/1906595129965912341?s=46&t=jDrfS5vZD4MFwckU5E8f5Q]">来自 Sewoong Oh (@sewoong79) 的推文</a>：我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 搭配时，ODS 在网页搜索方面的表现优于 OpenAI 的专用模型 GPT-4o-Search...</li><li><a href="https://x.com/AmazonScience/status/1906758835240312882]">来自 Amazon Science (@AmazonScience) 的推文</a>：了解 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，可以将稳健的步骤组合成复杂的流水线；处理从预订到 QA 的一切事务...</li><li><a href="https://fxtwitter.com/AmazonScience/status/1906758835240312882">来自 Amazon Science (@AmazonScience) 的推文</a>：了解 Amazon Nova Act —— 一种构建能可靠使用浏览器的 AI Agent 的简便方法 🧑‍💻 利用我们的新模型，可以将稳健的步骤组合成复杂的流水线；处理从预订到 QA 的一切事务...</li><li><a href="https://fxtwitter.com/sewoong79/status/1906595129965912341">来自 Sewoong Oh (@sewoong79) 的推文</a>：我们正在发布 OpenDeepSearch (ODS)，这是一个可与任何 LLM 配合使用的开源搜索 Agent。当与 DeepSeek-R1 搭配时，ODS 在网页搜索方面的表现优于 OpenAI 的专用模型 GPT-4o-Search...</li><li><a href="https://x.com/ArfurRock/status/1906756943349260682]">来自 Arfur Rock (@ArfurRock) 的推文</a>：🚨新独角兽预警 —— Etched，全球首款 Transformer ASIC。在经历了 5 亿美元和 7.5 亿美元的两轮隐身融资后，又以 15 亿美元的估值完成了未公开的 8500 万美元融资。7.5 亿美元那一轮就在大约 2 个月前。引用...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：我有一个激动人心的消息：我创业了！介绍 Sophont。我们正在为医疗保健的未来构建开源多模态基础模型。医疗 AI 领域需要一个 DeepSeek，而 @SophontAI 将会...</li><li><a href="https://x.com/stevenheidel/status/1906797154301329845">来自 Steven Heidel (@stevenheidel) 的推文</a>：我们今年将发布一个可以在你自己的硬件上运行的模型。引用 Sam Altman (@sama) TL;DR：我们很高兴能发布一款强大的、具备推理能力的新型开源权重语言模型...</li><li><a href="https://x.com/iscienceluvr/status/1906790937604579430?s=4]">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：我有一个激动人心的消息：我创业了！介绍 Sophont。我们正在为医疗保健的未来构建开源多模态基础模型。医疗 AI 领域需要一个 DeepSeek，而 @SophontAI 将会...</li><li><a href="https://fxtwitter.com/demishassabis/status/1906664622226083922">来自 Demis Hassabis (@demishassabis) 的推文</a>：很高兴宣布 @IsomorphicLabs 已融资 6 亿美元，以加速实现我们有朝一日在 AI 帮助下解决所有疾病的使命。我长期以来一直认为，改善人类健康是最重要的...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906756943349260682">来自 Arfur Rock (@ArfurRock) 的推文</a>：🚨新独角兽预警 —— Etched，全球首款 Transformer ASIC。在经历了 5 亿美元和 7.5 亿美元的两轮隐身融资后，又以 15 亿美元的估值完成了未公开的 8500 万美元融资。7.5 亿美元那一轮就在大约 2 个月前。引用...</li><li><a href="https://fxtwitter.com/rauchg/status/1906814800426086861">来自 Guillermo Rauch (@rauchg) 的推文</a>：我们正在构建一个用于运行任意计算的 API，目标是 Agentic AI 用例和长时间运行的任务。是的，它可以运行服务器。由支持我们每日 100 万次以上 @vercel 构建的基础设施提供动力，并经过优化...</li><li><a href="https://fxtwitter.com/ArfurRock/status/1906768733135098360">来自 Arfur Rock (@ArfurRock) 的推文</a>：Cursor 融资轮结束 —— 投后估值 96 亿美元，由 Thrive 和 A16z 领投。Accel 是新加入的投资者。ARR 为 2 亿美元，较 2024 年 11 月 25 亿美元融资轮增长了 4 倍。ARR 倍数与上一轮持平，保持在 50 倍。引用 Abe Brown ...</li><li><a href="https://fxtwitter.com/iscienceluvr/status/1906790937604579430">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：我有一个激动人心的消息：我创业了！介绍 Sophont。我们正在为医疗保健的未来构建开源多模态基础模型。医疗 AI 领域需要一个 DeepSeek，而 @SophontAI 将会...</li>

ntAI 将会...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1jnzdvp/qwen3_support_merged_into_transformers/">Reddit - 互联网的核心</a>：未找到描述</li><li><a href="https://x.com/egeberkina/status/1906088423988875617?s=46">来自 Ege (@egeberkina) 的推文</a>：GPT4o 彻底火了 🔥👨‍🍳视觉食谱来了，而且真的有点天才！提示词在 ALT 中</li><li><a href="https://x.com/stuff/posts/and/things/2398753298579">来自 GitHub - FxEmbed/FxEmbed 的推文：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://x.com">来自 GitHub - FxEmbed/FxEmbed 的推文：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://fxtwitter.com/egeberkina/status/1906088423988875617">来自 Ege (@egeberkina) 的推文</a>：GPT4o 彻底火了 🔥👨‍🍳视觉食谱来了，而且真的有点天才！提示词在 ALT 中</li><li><a href="https://fxtwitter.com/stuff/posts/and/things/2398753298579">来自 GitHub - FxEmbed/FxEmbed 的推文：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能</a>：修复 X/Twitter 和 Bluesky 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FxEmbed/FxEmbed</li><li><a href="https://archive.ph/2025.03.29-230008/https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d">独家 | Sam Altman 被 OpenAI 解雇背后的秘密与误导...</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1355270295279370370)** (189 条消息🔥🔥): 

> `基于 LLM 的代码生成、代码文档策略、Memory-Ref MCP 服务、Cursor IDE 问题、llms.txt 项目` 


- **Harper 揭秘 LLM 代码生成工作流**：一位成员分享了一篇[博文](https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/)，详细介绍了他们的 **LLM 代码生成工作流**，强调了在离散循环中进行头脑风暴、规划和执行的结构化方法。
   - 该文章强调了在使用 **LLMs** 构建小型产品时，制定明确计划以避免浪费时间的重要性。
- **Docs.dev 实现代码文档自动化**：[Docs.dev](https://docs.dev/) 被分享作为一个可以直接从代码库和现有内容*生成文档*的工具，并随着代码更改保持同步更新。
   - 它与 **GitHub** 集成，并提供从 **PRs** 自动生成文档、批量修改以及 SEO 优化分析等功能。
- **Memory-Ref 为 Cursor IDE 提供持久化编程偏好支持**：一位成员分享了一篇 [HN 帖子](https://news.ycombinator.com/item?id=43506068)，关于 **Cursor IDE** 与开源时序知识图谱 **Graphiti** 集成，通过使用 **Memory-Ref MCP** 在不同会话间提供持久化记忆。
   - 此次集成旨在帮助 **Cursor** 记住编程偏好和项目规范，减少不断重复提醒的需求。
- **探讨面向 LLMs 和人类的文档**：成员们讨论了面向 **LLMs** 的文档是否需要与面向人类的文档具有不同的详细程度，并提到 Markdown 正在成为一种首选的“编程语言”。
   - 一位成员链接了他们 **ttmp** 目录的一个示例，展示了他们在语言模型中发现有效的 **GitHub** [文档风格](https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md)。
- **针对 LLM 网站爬取的 llms.txt 标准提案**：**llms.txt** 项目旨在帮助语言模型有效地使用网站数据，该项目已在 [GitHub](https://github.com/AnswerDotAI/llms-txt) 上分享。
   - 该文件旨在为 **LLMs** 提供如何爬取和使用网站内容的指令，类似于 **robots.txt**。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.codeguide.dev/">CodeGuide</a>: CodeGuide 为您的 AI 编程项目创建详细文档。</li><li><a href="https://x.com/PrajwalTomar_/status/1895839765280539068?s=19">Prajwal Tomar (@PrajwalTomar_) 的推文</a>: 在过去的 5 个月里，我使用 Cursor 为客户构建了 16 个 SaaS 产品。现在，我破解了 Cursor 的最佳 AI 编程工作流。这是我构建生产级 MVP 的分步指南：</li><li><a href="https://news.ycombinator.com/item?id=43506068">Show HN: Cursor IDE now remembers your coding prefs using MCP | Hacker News</a>: 未找到描述</li><li><a href="https://harper.blog/2025/02/16/my-llm-codegen-workflow-atm/">My LLM codegen workflow atm</a>: 详细介绍了目前我使用 LLMs 构建软件的工作流，从头脑风暴到规划和执行。</li><li><a href="https://docs.dev/">Docs.dev | AI-assisted docs</a>: 直接从代码库和现有文档生成文档。确保您的文档随代码更改保持最新。</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=1439059137#gid=1439059137">AI In Action: Weekly Jam Sessions</a>: 未找到描述</li><li><a href="https://github.com/AnswerDotAI/llms-txt">GitHub - AnswerDotAI/llms-txt: The /llms.txt file, helping language models use your website</a>: /llms.txt 文件，帮助语言模型使用您的网站 - AnswerDotAI/llms-txt</li><li><a href="https://github.com/nuvic/fzf-kit.nvim">GitHub - nuvic/fzf-kit.nvim: A Neovim plugin that extends fzf-lua with additional utilities</a>: 一个扩展 fzf-lua 并提供额外实用工具的 Neovim 插件 - nuvic/fzf-kit.nvim</li><li><a href="https://github.com/go-go-golems/go-go-mcp/tree/main/ttmp">go-go-mcp/ttmp at main · go-go-golems/go-go-mcp</a>: Anthropic MCP 的 Go 语言实现。欢迎在 GitHub 上为 go-go-golems/go-go-mcp 的开发做出贡献。</li><li><a href="https://github.com/joernio/astgen">GitHub - joernio/astgen: Generate AST in json format for JS/TS</a>: 为 JS/TS 生成 JSON 格式的 AST。欢迎在 GitHub 上为 joernio/astgen 的开发做出贡献。</li><li><a href="https://github.com/go-go-golems/go-go-labs/blob/main/ttmp/2025-03-23/03-add-embeddings-to-command.md">go-go-labs/ttmp/2025-03-23/03-add-embeddings-to-command.md at main · go-go-golems/go-go-labs</a>: GO GO 实验实验室。欢迎在 GitHub 上为 go-go-golems/go-go-labs 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1355263770028412929)** (263 条消息🔥🔥): 

> `MCP 规范更新, HTTP Streamable Transport, OpenAI Agents SDK, UVX MCP Server, Model Context Protocol` 


- **MCP 规范拥抱 OAuth 2.1**：新的 **2025-03-26 MCP 规范** 草案包含了如 **OAuth 2.1** 等新的认证特性，但目前尚无客户端支持其测试；详见 [MCP 规范](https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/authorization/)。
- **HTTP Streamable Transport 引发可恢复性疑问**：关于 **HTTP Streamable Transport** 如何正确恢复会话存在疑问，特别是服务器避免在不同流上重放消息的义务，这目前似乎还处于假设阶段。
   - 规范指出 *服务器不得在流上发送 JSON-RPC 响应，除非是恢复与之前客户端请求关联的流*，这与可恢复性目标相矛盾。
- **环境变量在工具集成中表现不稳定**：成员们讨论了使用 **环境变量** 向工具传递 API token 的问题，在 `@modelcontextprotocol/inspector` 中调试正常，但在 **MCP client** 中调用工具时会抛出未授权错误。
   - 直接在 `claude_desktop_config.json` 文件中传递 token 似乎解决了该问题。
- **进度通知（Progress Notifications）是 MCP 中最棘手的部分**：用户正在寻求从服务器向客户端发送通知的示例，探索针对长耗时资源的 `notification/progress`，并发现 **客户端会向 `/message` 发回请求**。
   - 通知可能需要预先声明，或者在 **Claude Desktop** 等客户端中未获完全支持（仅加载动画有效，但消息不显示），且 `progressToken` 至关重要。
- **Goose 遇到端点描述问题**：一些用户报告了在通过 SSE 将 **Goose** 连接到本地 MCP server 时出现错误，一个快速修复方案是为服务器端点添加描述，正如该 [GitHub issue](https://github.com/block/goose/issues/1880) 中所建议的。
   - 在裁员风波后，Goose 似乎运行正常。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://jsonlint.com/">JSON Online Validator and Formatter - JSON Lint</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2025-03-26/basic/transports/#listening-for-messages-from-the-server">Transports</a>: ℹ️ 协议版本：2025-03-26 MCP 使用 JSON-RPC 对消息进行编码。JSON-RPC 消息必须采用 UTF-8 编码。该协议目前定义了两种标准的传输机制...</li><li><a href="https://openai.github.io/openai-agents-python/mcp/">Model context protocol (MCP) - OpenAI Agents SDK</a>: 未找到描述</li><li><a href="https://tenor.com/view/joke-missed-over-my-head-gif-26041934">Joke Missed GIF - Joke Missed Over My Head - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://block.github.io/goose/docs/getting-started/installation/">Install Goose | codename goose</a>: 选择在 CLI 和/或桌面端安装 Goose：</li><li><a href="https://glama.ai/api/mcp/openapi.json",">MCP API Reference</a>: Glama Gateway 的 API 参考</li><li><a href="https://docs.zapier.com/ai-actions/how-tos/auth">Authentication - Zapier</a>: 未找到描述</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/puppeteer">servers/src/puppeteer at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/filesystem">servers/src/filesystem at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://developers.cloudflare.com/agents/model-context-protocol/authorization/">Authorization · Cloudflare Agents docs</a>: 在构建 Model Context Protocol (MCP) 服务器时，你既需要一种允许用户登录（身份验证）的方式，也需要一种允许他们授予 MCP client 访问其账户资源权限（授权...）的方式。</li><li><a href="https://github.com/cloudflare/ai/tree/main/demos/remote-mcp-github-oauth">ai/demos/remote-mcp-github-oauth at main · cloudflare/ai</a>: 通过在 GitHub 上创建账号，为 cloudflare/ai 的开发做出贡献。</li><li><a href="https://github.com/mo">mo - Overview</a>: mo 拥有 49 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/Abiorh001/mcp_omni_connect">GitHub - Abiorh001/mcp_omni_connect: MCPOmni Connect is a versatile command-line interface (CLI) client designed to connect to various Model Context Protocol (MCP) servers using stdio transport. It provides seamless integration with OpenAI models and supports dynamic tool and resource management across multiple servers.</a>: MCPOmni Connect 是一款通用的命令行界面 (CLI) 客户端，旨在通过 stdio 传输协议连接到各种 Model Context Protocol (MCP) 服务器。它提供了与 OpenAI 模型的无缝集成，并支持跨多个服务器的动态工具和资源管理。</li><li><a href="https://github.com/angiejones/mcp-selenium">GitHub - angiejones/mcp-selenium: An MCP implementation for Selenium WebDriver</a>: Selenium WebDriver 的一个 MCP 实现。通过在 GitHub 上创建账号，为 angiejones/mcp-selenium 的开发做出贡献。</li><li><a href="https://mcp.pipedream.com">Pipedream MCP</a>: 访问超过 2,500 个 API 的 MCP 服务器，拥有 8,000 个预构建工具</li><li><a href="https://mcp.pipedream.com/app/people_data_labs">People Data Labs MCP Server | Pipedream</a>: 人员数据的真实来源</li><li><a href="https://github.com/block/goose/issues/1880">Goose stops responding when custom MCP is added · Issue #1880 · block/goose</a>: 添加自定义 MCP 时 Goose 停止响应 · Issue #1880 · block/goose。描述 Bug：当我启用我的 MCP 服务器时，Goose 变得无响应。该服务器在其他工具中运行正常，并且在 UI 的提示框中显示启动正常。我没有看到...</li><li><a href="https://github.com/tadata-org/fastapi_mcp">GitHub - tadata-org/fastapi_mcp: A zero-configuration tool for automatically exposing FastAPI endpoints as Model Context Protocol (MCP) tools.</a>: 一个零配置工具，用于自动将 FastAPI 端点暴露为 Model Context Protocol (MCP) 工具。 - tadata-org/fastapi_mcp</li><li><a href="https://github.com/punkpeye/awesome-mcp-clients">GitHub - punkpeye/awesome-mcp-clients: A collection of MCP clients.</a>: MCP 客户端集合。通过在 GitHub 上创建账号，为 punkpeye/awesome-mcp-clients 的开发做出贡献。</li><li><a href="https://developer.adobe.com/premiere-pro/uxp/">no title found</a>: 未找到描述</li><li><a href="https://github.com/Abiorh001/mcp_ev_assistant_server/blob/main/ev_assitant_server.py">mcp_ev_assistant_server/ev_assitant_server.py at main · Abiorh001/mcp_ev_assistant_server</a>: 一个功能强大的服务器实现，用于管理电动汽车 (EV) 充电站、行程规划和资源管理。该服务器提供了一套全面的...</li>

用于电动汽车相关工具和 API...</li><li><a href="https://github.com/trending">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1355274687978147861)** (28 条消息🔥): 

> `Speech MCP, IDA Pro 逆向工程, Cursor 的 OpenAPI MCP server, AI 驱动的 RAG 应用, 支持热重载的 MCP server 开发` 


- ****Speech MCP** 演示**: 一位用户分享了一段 [YouTube short](https://www.youtube.com/shorts/rurAp_WzOiY)，展示了 **Speech MCP**。
   - 另一位用户询问是否有与 **Claude** 兼容的版本。
- ****IDA Pro MCP Server** 简化逆向工程**: 一个 **IDA Pro MCP server** 被创建用于自动化逆向工程，具有[简化的安装过程](https://x.com/mrexodia/status/1906010119940239544)，允许用户在 2 分钟内开始尝试 vibe reversing。
   - 该服务器已使用 **Claude** 进行了测试，并自动配置了 **Cline** 和 **Roo Code**。
- ****OpenAPI MCP Server** 与 Cursor 集成**: 开发了一个 **OpenAPI MCP server**，使 **Cursor** 能够直接理解 API 规范，可在 [GitHub](https://github.com/ReAPI-com/mcp-openapi) 上获取。
   - 开发者正在寻求试用用户的反馈。
- ****CATIE** 智能路由 MCP 请求**: **CATIE (Context Aware Traffic Ingress Engine)** 是一个根据工具调用（tool call）路由 MCP 请求的代理，已在 [GitHub](https://github.com/mclenhard/catie-mcp) 上发布。
   - 这个免费的开源工具允许根据工具调用参数路由到不同的 MCP server，支持实时监控、后端切换和简单的负载分配。
- ****Pipedream** 发布带有用户身份验证的 MCP Server**: **Pipedream** 在 [GitHub](https://github.com/PipedreamHQ/pipedream/tree/master/modelcontextprotocol) 上发布了一个 MCP server，使开发者能够为 2,500 多个应用运行自己的 MCP server，并为他们的用户管理具有托管身份验证的服务器。
   - 根据 Pipedream 的说法，使用经过批准的客户端进行托管身份验证是 MCP 大规模运行的必要条件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/mrexodia/status/1906010119940239544">Duncan Ogilvie 🍍 (@mrexodia) 的推文</a>: 简化了我的 IDA Pro MCP server 的安装过程。你现在可以在不到 2 分钟的时间内开始尝试 vibe reversing！🤯 这是使用 Claude 进行测试的，但 Cline 和 Roo Code 也可以...</li><li><a href="https://ai-odyssey-planner.lovable.app/)">AI Odyssey Planner - 在几分钟内规划您的完美旅行</a>: 我们的 AI 旅行助手分析数百万个数据点，根据您的偏好、预算和时间表创建个性化行程。</li><li><a href="https://github.com/PipedreamHQ/pipedream/tree/master/modelcontextprotocol">pipedream/modelcontextprotocol at master · PipedreamHQ/pipedream</a>: 连接 API，速度惊人。对开发者免费。- PipedreamHQ/pipedream</li><li><a href="https://pipedream.com/connect">Connect</a>: Pipedream 是构建连接堆栈中所有服务的强大应用程序的最快方式，在需要时提供代码级控制，在不需要时提供无代码体验。</li><li><a href="https://github.com/ReAPI-com/mcp-openapi">GitHub - ReAPI-com/mcp-openapi: OpenAPI 规范 MCP server。</a>: OpenAPI 规范 MCP server。通过在 GitHub 上创建一个账号来为 ReAPI-com/mcp-openapi 的开发做出贡献。</li><li><a href="https://github.com/Cheffromspace/MCPControl">GitHub - Cheffromspace/MCPControl: 用于 Windows 操作系统自动化的 MCP server</a>: 用于 Windows 操作系统自动化的 MCP server。通过在 GitHub 上创建一个账号来为 Cheffromspace/MCPControl 的开发做出贡献。</li><li><a href="https://github.com/strowk/mcp-autotest">GitHub - strowk/mcp-autotest: 用于自动测试 MCP server 的工具</a>: 用于自动测试 MCP server 的工具。通过在 GitHub 上创建一个账号来为 strowk/mcp-autotest 的开发做出贡献。</li><li><a href="https://github.com/mclenhard/catie-mcp">GitHub - mclenhard/catie-mcp</a>: 通过在 GitHub 上创建一个账号来为 mclenhard/catie-mcp 的开发做出贡献。</li><li><a href="https://www.activepieces.com/mcp">280+ 开源 MCP — 立即在 Activepieces 上使用</a>: 通过 280 多个开源 MCP 赋予 AI 访问您的应用程序的权限。将它们与 Claude、Cursor 或 Windsurf 配合使用，让 AI 读取您的电子邮件、管理您的日历等。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1355977048967086291)** (1 messages): 

> `HF Reasoning Course, Gradio Dataframe, Reranker Models, Model Onboarding, Open R1 Update` 


- **HF Reasoning Course 获得 DeepSeek 助力**：根据[这篇 LinkedIn 帖子](https://www.linkedin.com/posts/ben-burtenshaw_new-unit-in-the-hugging-face-reasoning-course-activity-7311046882691108864-0cBN?utm_source=share&utm_medium=member_desktop&rcm=ACoAADxGwTsBLzNXo2rQ00oBRJPg_9dfhulQnio)，HF 推理课程的一个新单元现在以 **DeepSeek R1** 为特色。
- **Gradio 的 Dataframe 组件性能大幅提升**：Gradio 发布了其 `gr.Dataframe` 组件的一系列新更新，解决了超过 **70 个问题**，包括错误修复、改进和增强，详见[这篇博客文章](https://huggingface.co/blog/gradio-dataframe-upgrade)。
   - `gr.Dataframe` 组件在排行榜、仪表板和交互式可视化中非常受欢迎。
- **Reranker 模型基于 Sentence Transformers**：一篇博客文章详细介绍了如何使用 **Sentence Transformers v4** 训练和微调 Reranker 模型，如[这篇文章](https://huggingface.co/blog/train-reranker)所示。
- **模型入驻体验全新升级**：根据[这条推文](https://x.com/reach_vb/status/1905604906825716112)，HF 推出了全新的模型入驻（onboarding）体验，旨在简化对 Hub 功能的理解。
- **DeepSeek V3 数学技能显著提升**：根据[这条推文](https://x.com/nathanhabib1011/status/1905018770764259818)，对 **DeepSeek V3 0324** 的评估显示其在数学和 GPQA 方面取得了令人印象深刻的进步，但在指令遵循（instruction following）方面略有下降。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/gradio-dataframe-upgrade">介绍 Gradio 全新的 Dataframe！</a>：未找到描述</li><li><a href="https://huggingface.co/blog/train-reranker">使用 Sentence Transformers v4 训练和微调 Reranker 模型</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1905604906825716112">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：HF 上新的模型入驻体验现已上线，你觉得怎么样？其目的是让人们更容易理解 Hub 上已经可以实现的功能，并揭开 Hub 所有功能的神秘面纱！</li><li><a href="https://huggingface.co/blog/open-r1/update-4">Open R1：更新 #4</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/enzostvs/deepsite">DeepSite - 由 enzostvs 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/posts/AdinaY/152448454490712">Hugging Face 上的 @AdinaY</a>：“让我们来看看三月份中国社区的最新发布！👉…”：未找到描述</li><li><a href="https://huggingface.co/blog/endpoint-analytics">Inference Endpoints 中全新的分析功能</a>：未找到描述</li><li><a href="https://x.com/nathanhabib1011/status/1905018770764259818">来自 Nathan (@nathanhabib1011) 的推文</a>：刚刚完成了对 @deepseek_ai V3 0324 的评估！🚀 在数学和 GPQA 方面取得了令人印象深刻的进步，但指令遵循能力略有下降。更令人担忧的是——AIME25 保持不变。可能存在污染...</li><li><a href="https://huggingface.co/blog/burtenshaw/custom-local-coding-vscode">自定义 Vibe Coding 任务第一部分：任务开始 🧙</a>：未找到描述</li><li><a href="https://huggingface.co/blog/intel-gaudi-backend-for-tgi">🚀 在 Intel Gaudi 上使用 TGI 加速 LLM 推理</a>：未找到描述</li><li><a href="https://huggingface.co/blog/giadap/beyond-consent">我点击了“我同意”，但我到底同意了什么？</a>：未找到描述</li><li><a href="https://x.com/hugoch/status/1905561210839298473">来自 Hugo Larcher (@hugoch) 的推文</a>：🧠 LLM 推理不仅仅关乎延迟——它关乎负载下的稳定性。不同的工作负载、配置和硬件 = 非常不同的实际性能。在 Hugging Face 🤗，我们构建了推理...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1355256066149187744)** (145 条消息🔥🔥): 

> `Hugging Face Pro 借记卡问题、视频口型同步工具、RunPod 与 HuggingFace 模型、HF 模型容器化、Agentx 竞赛研究赛道` 


- **Hugging Face Pro 借记卡风波**：一位用户报告称，尽管收到了错误提示，但仍被扣除了 **Hugging Face Pro 订阅**费用，并询问如何退款。
   - 有人建议这可能是一个已知问题，即借记卡付款会先扣款一次，退款通常在 **两周内** 处理。
- **Video Retalking 工具故障**：一位用户分享了 Hugging Face Spaces 上的 [VideoRetalking 工具](https://huggingface.co/spaces/fffiloni/VideoRetalking) 链接，并指出它*运行得相当不错，但有一点小故障*。
   - 他们还想知道像 **HeyGen 这样的 SaaS 解决方案** 是操纵身体的情绪化动作，还是仅仅进行口型同步（lip syncing）。
- **RunPod 的模型管理难题**：一位用户在克隆并改进模型后，正努力尝试在 **RunPod** 上使用来自 **Hugging Face** 的模型，并寻求建议。
   - 该用户负担不起高性能 GPU，同时也在寻找类似 **视频口型同步** 之类的酷炫工具来制作不露脸视频。
- **HF Space 项目转换困扰**：一位用户寻求关于将本地 Python 项目转换为 **Hugging Face Space 项目** 的建议。
   - 有人指出 Spaces 需要 GUI，不过 Docker Spaces 可能是个例外，并且虚拟机并不像本地环境那样免费；文中分享了 [Hugging Face 文档](https://huggingface.co/docs) 的链接。
- **Hugging Face Daily Papers 实现 RSS 化**：一位用户为 [Hugging Face 论文页面](https://huggingface.co/papers) 的每日论文寻求 **RSS 订阅源**。
   - 社区分享了多种解决方案，包括 [rss.app](https://rss.app)、[fetchrss.com](https://fetchrss.com/) 和 [politepol.com](https://politepol.com)，此外还有一位用户创建的订阅源 [papers.takara.ai/api/feed](https://papers.takara.ai/api/feed)，其代码已在 [GitHub](https://github.com/404missinglink/HF-Daily-Papers-Feeds) 上开源。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.jetbrains.com/help/pycharm/hugging-face.html">Hugging Face | PyCharm</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/VideoRetalking">VideoRetalking - fffiloni 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/pycharm-integration">Hugging Face + PyCharm</a>: 未找到描述</li><li><a href="https://huggingface.co/docs">Hugging Face - 文档</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/hub/spaces-dependencies">处理 Spaces 依赖项</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/discord-community/LevelBot/tree/main">discord-community/LevelBot at main</a>: 未找到描述</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: 未找到描述</li><li><a href="https://rss.app">RSS Feed 生成器，从 URL 创建 RSS 订阅源</a>: 排名第一的 RSS 订阅源：从几乎任何来源生成 RSS 订阅，并使用 JS 或 iframe 小部件将新闻订阅嵌入到您的 html 网站中。</li><li><a href="https://fetchrss.com/">RSS 生成器 - FetchRSS</a>: 免费在线 RSS 生成器。从任何网页创建 RSS。为您的网站构建 RSS 订阅或生成用于个人用途的 XML。</li><li><a href="https://politepol.com">为任何网页生成 RSS 订阅源 | PolitePol</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/takarajordan/806643001426071">Hugging Face 上的 @takarajordan: &quot;我为 HuggingFace Daily Papers 制作了一个 RSS 订阅源！！ 🤗 

只需在此订阅：…&quot;</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1355401613464047707)** (2 条消息): 

> `AI Agent 可观测性与评估、Tableau 认证数据分析师培训、WordPress 开发者课程` 


- **在加分单元中攻克 AI Agent 可观测性**：一位成员正在学习 *Agents 课程：加分单元 2 - AI Agent 可观测性与评估*，作为其学习旅程的一部分。
- **Tableau 培训进展稳步推进**：一位成员正在学习 **2024 Tableau 认证数据分析师培训**，已完成 **523 个章节中的 432 个**。
- **WordPress 开发技巧学习中**：一位成员开始了 *成为 WordPress 开发者：用代码释放力量* 课程，并完成了 **234 个章节中的 2 个**。


  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1355709432620908636)** (3 条消息): 

> `Docker Model Runner, Local LLMs, SAGE-2 AI, Symbolic Reasoning System` 


- **Docker 在容器中运行本地 LLMs！**: Docker, Inc. 推出了一项实验性的 **Model Runner** 功能，允许用户使用 Docker CLI 命令在本地运行 **Large Language Models (LLMs)**。
   - 该解决方案支持运行更多模型，具备 **private inference**（私有推理）、**on-demand model loading**（按需模型加载）和 **GPU acceleration**（GPU 加速）功能，通过将模型依赖项容器化，绕过了 macOS 在访问宿主机 GPU 资源方面的限制。
- **SAGE-2 破解“黑盒”！**: **SAGE-2** 是一款采用连续 **symbolic reasoning system**（符号推理系统）设计的新型 AI，使其决策过程可追溯、可解码且可解释。
   - 与 **GPT**、**Gemini** 和 **DeepSeek** 等属于“黑盒”的现代 AI 不同， **SAGE-2** 允许用户查看模型的内部状态和推理过程，这对于医疗和司法等敏感决策中的伦理审计和信任至关重要。你可以在这个 [HF Space](https://huggingface.co/spaces/gnai-creator/sage-two-visual) 中亲自尝试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/879548962464493622/1355709069368885470">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是进行游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://huggingface.co/spaces/gnai-creator/sage-two-visual">SAGE-2 - a Hugging Face Space by gnai-creator</a>: 未找到描述</li><li><a href="https://github.com/Traperto/magic-bytes-validator">GitHub - Traperto/magic-bytes-validator: 通过 magic bytes 和 MIME 类型进行检查的文件验证器</a>: 通过 magic bytes 和 MIME 类型进行检查的文件验证器 - Traperto/magic-bytes-validator
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1355263011425878128)** (33 条消息🔥): 

> `FactoryManager for Linux Containers, AI Menu in Neovim, Tree of Thoughts (ToT) Implementation, Learning UI with Image Gen, RepoDump CLI Tool` 


- **FactoryManager 为 Linux 容器带来 Robotgo 式的体验**: 一位开发者介绍了 **FactoryManager**，这是一个封装了 [linuxserver.io](https://www.linuxserver.io/) 桌面环境容器的 Python 包，实现了程序化控制。
   - 开发者正在征求反馈，是应该为 OpenAI、Anthropic 构建一个可扩展的基类，还是专注于桌面管理。演示视频见 [此 demo 视频](https://cdn.discordapp.com/attachments/897390720388825149/1355263010200879405/outputFactoryManagerDemo.mp4?ex=67ec3f09&is=67eaed89&hm=e81aef37a9c17a9d81430366c5c47fd0bebfd5d3a32e735851dff51b971c7de1&)，仓库见 [GitHub 上的 FactoryManager](https://github.com/sampagon/factorymanager)。
- **NeoVim 获得 AI Menu 及更多功能**: 演示了 **Neovim** 中的 AI 菜单、带有 **MetaHumans** 的 **Unreal Engine 5.5.4** 以及后量子加密技术，全部运行在 **Arch Linux 6.13.5 Hyprland** 上。
   - 此外还有一个指向 [HuggingFace 的 Open OdysseyXL 集合](https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5) 的链接。
- **Tree of Thoughts 让 Chain of Thought 推理相形见绌**: 一位成员分享了一篇博文，解释了 **Tree of Thoughts (ToT)** 论文如何将 **GPT-4** 与树搜索算法结合，显著提升了在左到右的 **Chain of Thought (CoT)** 难以处理的任务上的性能。
   - 在 *Game of 24* 任务中，使用 **CoT** 提示词的 **GPT-4** 仅解决了 **4%** 的任务，而 **ToT** 达到了 **74%** 的成功率，详见这篇 [HuggingFace 博客文章](https://huggingface.co/blog/sadhaklal/tree-of-thoughts)。
- **RepoDump 工具将代码库转换为适用于 LLMs 的 Markdown**: 一位开发者发布了 `repodump 0.1-alpha`，这是一个 CLI 工具，用于提取 Git 仓库或目录并将其格式化为 Markdown，以便快速与 LLMs 分享，可在 [GitHub](https://github.com/zakhikhan/repodump) 上获取。
   - 该工具会跳过二进制文件，遵循 `.gitignore`，输出 Markdown 或纯文本，并使用 Simon Willison 的 `ttok` 估算 token 数量。有用户评价说 *安装过程有点可疑 (sus)*。
- **HF 网站 Chrome 扩展增加了仓库大小显示和讨论搜索功能**: 一位开发者为 HF 网站推出了一款 Chrome 扩展，增加了查看仓库总大小和全文讨论搜索等功能，如 [Chrome Web Store](https://chromewebstore.google.com/detail/hf-tools/pghpacbbnhhoohoniikaafjcnkcjflch) 所示。
   - 这是一个开源项目，代码可在 [GitHub](https://github.com/fakerybakery/hf-tools/) 上获取。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/edwardthefma/Sentify">Sentify - edwardthefma 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/collections/open-neo/odysseyxl-67d4cf53fa315a2e04ca20d5">OdysseyXL - open-neo 集合</a>：未找到描述</li><li><a href="https://imgsli.com/MzY0MzQ1/4/3">Imgsli</a>：未找到描述</li><li><a href="https://imgsli.com/MzY0MzUw/2/0">Imgsli</a>：未找到描述</li><li><a href="https://github.com/emooreatx/EthicsEngine">GitHub - emooreatx/EthicsEngine</a>：通过在 GitHub 上创建账户来为 emooreatx/EthicsEngine 的开发做出贡献。</li><li><a href="https://chromewebstore.google.com/detail/hf-tools/pghpacbbnhhoohoniikaafjcnkcjflch">HF Tools - Chrome 网上应用店</a>：适用于 Hugging Face 的实用工具</li><li><a href="https://github.com/zakhikhan/repodump">GitHub - zakhikhan/repodump: repodump: A lightweight CLI tool that extracts Git repositories as formatted markdown, optimized for sharing with LLMs. Get better AI assistance with your codebase through clean, structured code dumps.</a>：repodump：一个轻量级的 CLI 工具，可将 Git 仓库提取为格式化的 Markdown，专为与 LLMs 共享而优化。通过干净、结构化的代码转储，获得更好的 AI 代码库协助。</li><li><a href="https://github.com/sampagon/factorymanager">GitHub - sampagon/factorymanager: A manager for programmatically controlling linuxserver.io Docker containers with robotgo-cli</a>：一个用于通过 robotgo-cli 以编程方式控制 linuxserver.io Docker 容器的管理器 - sampagon/factorymanager</li><li><a href="https://github.com/fkcptlst/labtasker">GitHub - fkcptlst/labtasker: Experiment task scheduling made easy.</a>：让实验任务调度变得简单。通过在 GitHub 上创建账户来为 fkcptlst/labtasker 的开发做出贡献。</li><li><a href="https://github.com/GeekyGhost/Little-Geeky-s-Learning-UI.git">GitHub - GeekyGhost/Little-Geeky-s-Learning-UI: An Ollama based Gradio UI that uses Kokoro TTS</a>：一个基于 Ollama 的 Gradio UI，使用了 Kokoro TTS。通过在 GitHub 上创建账户来为 GeekyGhost/Little-Geeky-s-Learning-UI 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/sadhaklal/tree-of-thoughts">Understanding and Implementing the Tree of Thoughts Paradigm</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1355520731940786246)** (4 条消息): 

> `使用 Transformers 进行 text2text LLM 的全量微调，Mercury 与 LaMDA 的性能对比，DPO Mistral 7B 训练问题` 


- **寻求 Transformers 库全量微调示例**：一位成员询问了如何使用 **Transformers Python library** 对 **text2text LLM** 进行全量微调（不使用 PEFT、(Q)LoRA 或量化）的示例。
   - 另一位成员建议查看 [Hugging Face tutorials](https://huggingface.co/docs/transformers/index) 以获取不带量化的简单微调脚本。
- **Mercury Coder 与 LaMDA 相比的极速表现**：一位成员分享了 [Mercury Coder 的技术报告](https://drive.google.com/file/d/1xrqTqF88OZblf0NgMjr1REU4doYlkNXf/view?usp=drivesdk)，指出其内容模糊，并质疑为什么 **Mercury** 比 **LaMDA** 快得多。
   - 他们觉得这很奇怪，因为据称两者都使用了 Transformer 骨干网络。
- **DPO Mistral 7B 奖励准确率存疑**：一位成员报告称，在使用 [HumanLLMs/Human-Like-DPO-Dataset](https://huggingface.co/datasets/HumanLLMs/Human-Like-DPO-Dataset) 对 **Mistral 7B** instruct 进行 **DPO** 时，出现了可疑的训练奖励/准确率，立即达到了 100%。
   - 该成员还分享了一张与该问题相关的 [图片](https://cdn.discordapp.com/attachments/922424173916196955/1356231286762766441/image.png?ex=67ebd050&is=67ea7ed0&hm=8607a1e7919e325d26e4405d137204ea4a256e61f3c917685d3684991818486c)，并正在寻找原因和解决方案。



**提到的链接**：<a href="https://drive.google.com/file/d/1xrqTqF88OZblf0NgMjr1REU4doYlkNXf/view?usp=drivesdk.">Inception Labs_Mercury_Tech_Report.pdf</a>：未找到描述

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1355478204076523690)** (12 条消息🔥): 

> `课程集成, Hugging Face Agent 课程, Gradio Client 问题, Unit 3 发布` 


- **课程集成仍未完成？**：一位成员询问该课程是否已完全集成到 **NLP/LLM** 课程中，或者是否还有待更新的内容。
   - 他们急于了解课程后续还有哪些内容。
- **HF Agent 课程证书缺失？**：一位用户报告称已完成 **Hugging Face Agents 课程** 的 Unit 2，但其账户未显示通过 Unit 1 或获得 **Fundamentals 证书**。
   - 尽管下载了证书 PDF，但其账户中没有确认信息。
- **Gradio Client 问题已解决**：几位用户在克隆第一个 Agent 模板部分的 Space 时遇到了与 'bool' 不可迭代相关的 `TypeError`，经追溯是 **Gradio client** 的问题。
   - 一位用户通过在 **requirements.txt** 文件中添加 `pydantic==2.10.6` 提供了快速修复方案，并引用了 [此 GitHub issue](https://github.com/gradio-app/gradio/issues/10649)，该方案解决了问题。
- **Unit 3 何时发布？**：多位成员正在询问 **Unit 3** 的发布日期。
   - 目前还没有关于其发布的具体信息。
- **Gemini 服务是可行的替代方案吗？**：一位成员建议 **Google 的 Gemini 服务** 是一个可行的替代方案，只要能从 AI Studio 获取 **API key**，基本上是免费的。
   - 这一评论是针对另一位用户抱怨必须支付一个月费用才能完成课程的回应。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/879548962464493622/1353904868435169330">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 非常适合玩游戏、与朋友闲逛，甚至建立全球社区。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://github.com/gradio-app/gradio/issues/10649">application does not launch · Issue #10649 · gradio-app/gradio</a>: 描述 Bug：我在 Linux v 5.16.0 下的 Block 中运行这段代码时出错，但在 Windows 操作系统上没有问题。我怀疑问题出在事件管理方式上。版本...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1355256815121596657)** (54 messages🔥): 

> `Base vs Instruct Models, smolagents System Prompt, Hugging Face Agent Course Schedule, Hugging Face Certifications, API Rate Limits` 


- **Base Models vs Instruct Models Clarified**: 一位成员分享了一个 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/)，解释了 **base models**（又称“自动补全模型”）与 **instruct/chat models** 之间的区别。
   - 成员指出，虽然 base model 几乎可以做任何事情，但 *instruct-tuning* 教会它遵循指令，而 *chat-tuning* 则教会它以多轮对话的方式进行响应。
- **Prompt Engineering Struggles in smolagents**: 一位在 Unit 2.1 中设计自己模型的成员在初始化 Agent 后，正努力通过调整 `agent.system_prompt` 来引导模型。
   - 他们询问模型的 *dataflow* 和 *control logic* 是否存在于 Prompt 中，例如 Prompt 示例是否具体决定了 Tool 的使用方式以及数据如何在它们之间传递。
- **Course Schedule Update Still Delayed**: 一位成员询问了 Unit 3 的情况，但另一位成员澄清说该单元尚未发布，目前最新的是关于 observability 和 evaluation 的奖励单元。
   - 成员建议通过 *announcements channel* 关注更新，因为进度表目前不是最新的。
- **HF Certification Reflections Lacking**: 一位成员注意到他们的 HF 账户没有通过 Unit 1 或获得 Fundamentals 证书的记录。
   - 另一位成员确认这是预期行为，因为 PDF 证书是从 Hugging Face Space 生成的，不会保存在个人资料中，并且 **Tool 没有 Rate Limit**。
- **Gemini API Relieves Hugging Face Rate Limit Woes**: 一位成员耗尽了 Hugging Face API 请求限制，并切换到了 **Google Gemini**，分享了一个 [GitHub 仓库](https://github.com/PrinceDobariya0710/huggingface-ai-agent-course)，其中包含使用 Gemini 完成到 Unit 2.2 的练习。
   - 建议在 [Hugging Face Billing](https://huggingface.co/settings/billing) 查看推理使用情况。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://discordapp.com/channels/879548962464493619/1355353373053947924/1355353373053947924">Discord - Group Chat That’s All Fun &amp; Games</a>: Discord 是玩游戏和与朋友放松，甚至建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://huggingface.co/settings/billing,">Hugging Face – The AI community building the future.</a>: 无描述</li><li><a href="https://alexhruska.medium.com/agents-course-smolagents-framework-9ce823afe015#b379">Agents Course smolagents Framework</a>: 在上周发布了 LLM Fine-tuning 的奖励单元后，Hugging Face 团队本周带来了 Unit 2。本单元重点关注……</li><li><a href="https://gist.github.com/skymaiden/8b472bbb01ea9bdfca43f64c32e583a6">Notes from a front-end dev on the Hugging Face &quot;Agents Course&quot;</a>: 一位前端开发关于 Hugging Face &quot;Agents Course&quot; 的笔记 - 01_context.md</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c1sy03/an_explanation_of_base_models_are/">Reddit - The heart of the internet</a>: 无描述</li><li><a href="https://github.com/PrinceDobariya0710/huggingface-ai-agent-course">GitHub - PrinceDobariya0710/huggingface-ai-agent-course: Repository to contain all exercises of Huggingface&#39;s AI agent course but with Google&#39;s Gemini model where you will get enough API requests limits to complete exercises</a>: 包含 Hugging Face AI Agent 课程所有练习的仓库，但使用的是 Google Gemini 模型，在那里你将获得足够的 API 请求限制来完成练习。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/1356258495967793374)** (1 messages): 

> `Mini-R1, Countdown task, GRPOTrainer, vLLM, quantization` 


- **Mini-R1 User Flummoxed by Quantization**: 一位用户正尝试在 **Mini-R1** 上使用 **GRPOTrainer** 和 **vLLM** 运行 **Countdown task**。
- **Quantization Quandaries Plague Project**: 该用户报告在应用 *quantization* 时出现失败。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1355288141959008443)** (154 messages🔥🔥): 

> `OpenAI Image Generator Nerfed, Meta's Transfusion paper and GPT-4o, Belief State Transformer, Dynamic RL, Rejuvenation medicine`

- **OpenAI 图像生成器遭遇“削弱 (Nerf)”**：成员们认为 **OpenAI 的图像生成器** 输出质量有所下降，并且他们可能已经*停止支持吉卜力风格的提示词 (Ghibli style prompts)*。
   - 成员们还表示，*模型已经达到了极限点，即模型变得越来越大，但并没有变得越来越好，甚至在某些情况下变得越来越差*。
- **Meta 的 Transfusion 论文可能为 GPT-4o 提供动力**：一位成员链接了 [Meta 的 Transfusion 论文](https://arxiv.org/abs/2408.11039)，并指出这可以解释 **GPT-4o** 的多模态能力（自回归和扩散建模的混合体）。
   - **Transfusion** 论文介绍了一种训练模型的方法，该模型可以无缝生成离散和连续模态，在文本生成图像方面比 **Chameleon** 获得了更好的 FID 和 CLIP 分数。
- **Belief State Transformer 构建更丰富的潜表征**：一位成员分享了 [Belief State Transformer](https://x.com/mgostIH/status/1896180298817405332) 的链接，并表示它*使 Transformer 能够更好地对状态进行建模，并且还可以额外以结尾为条件！*
   - 另一位成员认为，他们*证明了该架构可以构建理想 Belief Transformer 的这种表征*，但需要一个已经收敛到完美学习数据底层概率分布的理想 Belief Transformer。
- **动态 RL 消除对显式变分边界的需求**：一位成员表示，他正在研究一种通过*引入一个 RL Agent* 来消除扩散模型中对显式变分边界需求的方法。
   - 其他成员表示，大多数 RL 方法也是变分方法，控制理论也可以被使用。
- **返老还童医学被视为一种可能**：成员们表达了对**返老还童医学 (rejuvenation medicine)** 可能在未来 3 年内广泛普及的希望。
   - 一位成员引用了控制细胞和预防癌细胞的问题，认为这是实现这一目标的主要**障碍**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mgostIH/status/1896180298817405332">来自 mgostIH (@mgostIH) 的推文</a>：这篇论文非常酷：The Belief State Transformer。非常简单的技术且训练速度快，使 Transformer（或其他序列模型）能更好地建模状态，并可以额外调节...</li><li><a href="https://fxtwitter.com/SaxenaNayan/status/1905334927526105492">来自 Nayan Saxena (@SaxenaNayan) 的推文</a>：分析 OpenAI 图像生成帧显示出多尺度结构：拉普拉斯增量（Laplacian deltas）突显了迭代的分带编辑，熵实现了定位，而流（flow）发生了偏移。证据支持交错的潜空间自回归（interleaved latent autoregressio...）</li><li><a href="https://arxiv.org/abs/2301.08243">Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture</a>：本文展示了一种在不依赖手工设计的数据增强的情况下，学习高度语义化图像表示的方法。我们介绍了基于图像的联合嵌入预测架构（Image-based Joint-Embedding Predictive Archi...）</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1905730169631080564">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：既然知道 GPT-4o 可能是自回归（autoregressive）和扩散建模（diffusion modeling）的混合体，Meta 的 Transfusion 论文似乎变得极其相关。也许这就是 GPT-4o 的工作原理？引用 Tanishq Mathew Abraham, ...</li><li><a href="https://fxtwitter.com/TheTuringPost/status/1906304408415359067?t=QrP_I5vSzaLt-3r42Hyyig&s=19">来自 TuringPost (@TheTuringPost) 的推文</a>：9 种多模态思维链（Chain-of-Thought）方法 ▪️ KAM-CoT ▪️ Multimodal Visualization-of-Thought (MVoT) ▪️ Compositional CoT (CCoT) ▪️ URSA ▪️ MM-Verify ▪️ Duty-Distinct CoT (DDCoT) ▪️ Multimodal-CoT ▪️ Graph-of-Thoug...</li><li><a href="https://x.com/mtschannen/status/1906021357982257417">来自 Michael Tschannen (@mtschannen) 的推文</a>：4o 原生图像生成已被证实是某种自回归模型。也许现在是 AR 怀疑论者补习关于多模态 AR 模型最新文献的好时机。</li><li><a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?t=s2S595eV_11U1l7BmZkpVg&s=19">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：GPT-4o 被发现具有推理能力！对我来说，这就是正在我们眼前构建的 GPT-5 系统。请看我在第一条评论中的帖子。预计会有更多工具和更新添加到所有模型中。这 ...</li><li><a href="https://tenor.com/view/math-meme-gif-23715871">数学迷因 GIF - Math Meme - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/intensifies-sooning-soontm-midsizedonkey7-nowhere-gif-12050318">Intensifies Sooning GIF - Intensifies Sooning Soontm - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.visualcapitalist.com/charted-the-decline-of-u-s-software-developer-jobs/">图表化：美国软件开发人员职位的下降</a>：Indeed 上美国软件开发人员的招聘职位数量触及 5 年来的最低点，较 2020 年水平下降了 33% 以上。</li><li><a href="https://scholar.google.com/scholar?hl=en&as_sdt=2005&sciodt=0,5&cites=12723682001549119492&scipsc=&q=&scisbd=1">Google Scholar</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1355264411064864891)** (28 条消息🔥): 

> `LLMs 规划 vs 识别, Robert Sapolsky 决定论, 机械可解释性团队受众` 


- **LLM 是在规划还是仅仅在假装？**：成员们讨论了 **LLM 提前规划**的概念，质疑这是否更类似于对可能 Token 序列的 *识别* 或预期，而非实际的 *规划* 或选择。
   - 一位成员指出，对非技术受众使用 *人类术语* 更容易理解，但最初的发布者质疑 LLM 是否拥有自由意志，还是仅仅在预测最可能的输出。
- **Sapolsky 的无自由意志布道**：一位成员提到了 **斯坦福大学教授 Robert Sapolsky**，他从生物神经学角度坚信 **决定论（determinism）**，并推荐了[他关于该主题的 YouTube 视频](https://www.youtube.com/playlist?list=PL848F2368C90DDC3D)。
   - 另一位成员分享了一段引用，**Sapolsky** 在其中表示，在了解到上帝使法老的心刚硬后，他意识到没有上帝也没有自由意志，因此 *宇宙是宏大、空虚且冷漠的*。
- **机械可解释性（Mechanistic Interpretability）团队心直口快**：一位成员注意到 **机械可解释性** 团队似乎并不会针对不同受众调整他们的语言，无论受众是谁都保持技术化。
   - 该成员补充说，他们可能错了，而且他们正在变老，随后附上了一张[老人对着 AI 咆哮](https://tenor.com/jD8yO9J5WHx.gif)的 GIF。



**提到的链接**：<a href="https://tenor.com/jD8yO9J5WHx.gif">咆哮 AI GIF - Yelling Yell Ai - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/)** (1 条消息): 

endomorphosis: https://x.com/TheTuringPost/status/1906304408415359067
  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1355333080721784854)** (47 条消息🔥): 

> `xAI 收购 X，NVIDIA RTX PRO 6000 Blackwell 工作站版，视觉自回归建模：通过次尺度预测实现可扩展图像生成，Runway Gen-4 发布，OpenAI 模型发布` 


- **马斯克通过 xAI 收购 Twitter**：一份 [路透社文章](https://www.reuters.com/markets/deals/musks-xai-buys-social-media-platform-x-45-billion-2025-03-28/) 报道称 **xAI** 以 **450 亿美元** 收购了 **X**，引发了关于贷款抵押品影响和潜在财务策略的讨论。
   - 一些成员开玩笑说这可能是 *洗钱*，或者是从 **xAI** 向 **X** 注入资金的一种方式。
- **NVIDIA 发布 RTX PRO 6000 Blackwell GPU**：**NVIDIA** 发布了 [RTX PRO 6000 Blackwell 工作站版](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)，这是一款 **96GB** 显存的显卡，承诺提供 *极致的 AI 和图形性能*。
   - 成员们将其与使用 **四张 5090** 进行了比较，指出它功耗更低，但 VRAM 和算力较少。
- **新论文中 GPT 表现优于扩散模型**：一位成员分享了 [视觉自回归建模：通过次尺度预测实现可扩展图像生成](https://github.com/FoundationVision/VAR) 的链接，该论文获得了 **NeurIPS 2024 最佳论文**，展示了 **GPT** 在图像生成方面击败了扩散模型。
   - 一位成员不屑地表示：*去买一个 Scam Altman 虚构的聚变发生器吧。如果你想投资，那是万亿级别的产业。*
- **Runway Gen-4 生成一致性媒体**：**RunwayML** 发布了 [Gen-4](https://runwayml.com/research/introducing-runway-gen-4)，能够精确生成跨场景的一致角色、地点和物体。
   - 一位成员表示怀疑，称 *眼见为实*，并批评当前的 AI *比追逐自己尾巴的狗还糟糕*。
- **传闻 OpenAI 将发布小模型**：有推测称 **OpenAI** 将发布一个新模型，可能是针对移动端的轻量化模型，特别是在与 **Apple** 的交易告吹之后。
   - 一位成员开玩笑地建议，参考之前的发布，它可能是拥有 **100M 参数** 的 **GPT 2.5**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://runwayml.com/research/introducing-runway-gen-4">Runway 研究 | 介绍 Runway Gen-4</a>: 无描述</li><li><a href="https://vxtwitter.com/Polymarket/status/1905738829761540123">来自未定义用户的推文</a>: 无描述</li><li><a href="https://x.com/eisneim/status/1896552532568338604">来自 eisneim (@eisneim) 的推文</a>: RTX 4090 96GB 确认！我最后一次去了 GPU 工厂，卖掉了我最后的 4090 24GB 并买了一张全新的 48GB；24GB 是在 2023 年底以 1.5 万人民币（2059 美元）购买的，现在卖了 1.82 万人民币（2498 美元）并买了 4...</li><li><a href="https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/">NVIDIA RTX PRO 6000 Blackwell 工作站版</a>: 体验无与伦比的 AI 和图形性能。</li><li><a href="https://tenor.com/view/cabin-aesthetic-river-cabin-in-the-woods-winter-cabin-gif-13574384">小木屋审美 GIF - 小木屋审美河流 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.forethought.org/research/preparing-for-the-intelligence-explosion">为智能爆炸做准备 | Forethought</a>: 能够加速研究的 AI 可能会在短短几年内推动一个世纪的技术进步。在此期间，新的技术或政治发展将引发重大且具有挑战性的...</li><li><a href="https://github.com/FoundationVision/VAR">GitHub - FoundationVision/VAR: [NeurIPS 2024 最佳论文][GPT 击败扩散模型🔥] [视觉生成中的缩放法则📈] “视觉自回归建模：通过次尺度预测实现可扩展图像生成”的官方实现。一个用于自回归图像生成的*超简单、用户友好且处于 SOTA 状态*的代码库！</a>: [NeurIPS 2024 最佳论文][GPT 击败扩散模型🔥] [视觉生成中的缩放法则📈] “视觉自回归建模：通过次尺度预测实现可扩展图像生成”的官方实现...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1355266897301475519)** (103 条消息🔥🔥): 

> `Distributed Model Deployment, Meta-Learning, RWKV Discord Bot Deception, AI Generated Content, Email LLM Mishaps` 


- **优化 Distributed Model Deployment 的带宽使用**：一位成员正在开发一个基础设施层，通过自适应压缩和智能路由来优化分布式系统中的模型传输和部署，以解决 **bandwidth waste**（带宽浪费）和 **inference latency**（推理延迟）问题。
   - 该成员提议分享一个 Demo，并寻求其他在 **distributed inference** 领域有经验的人的看法。
- **讨论模型训练范式的缺陷**：一位成员质疑模型模仿人类推理的传统训练方法，认为这即使对于 **world foundation models** 也是一种局限。
   - 他们提到 **meta-learning** 是一种替代方案，并寻求关于这一想法潜在缺陷的观点。
- **RWKV Discord 遭到欺骗性 AI Agent 攻击**：RWKV Discord 的成员报告了一起事件：一个 AI Agent 伪装成人类研究员，分享了一篇包含错误数学公式和来自 GitHub 仓库代码的博客文章，以此浪费他人时间。该事件始于一条带有 [附图](https://cdn.discordapp.com/attachments/729741769738158194/1355917453984534748/image.png?ex=67ebfd88&is=67eaac08&hm=adec41fbe015cdd55934cd70e59ead00b5428b2a750f081f6e56faabaacdea5a&) 的私信。
- **社区应对 AI 生成内容的困境**：RWKV Discord 的这起事件引发了关于处理 AI 生成内容挑战的讨论，特别是当来源未公开时，可能会破坏对外部贡献的信任。
   - 成员们敦促追踪 AI 生成的内容，其中一人建议使用加密签名以确保人工验证，并 [检查生成的文本是否存在水印](https://discord.com/channels/992359628979568762/992359629419991142/1355598505577677011)。
- **房东 LLM 安排虚假预约**：一位成员分享了个人经历：一家租赁公司使用 LLM 进行邮件沟通，结果导致了一个员工并不知情的 **phantom appointment**（虚假预约），这表明系统可能存在低效。
   - 该成员认为由于 LLM 的运行故障，他们正受益于较低的租金，并估计该公司可能因为该系统损失了数百万美元。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1355326217414512741)** (12 条消息🔥): 

> `MAML, Neural-guided CoT, RLHF, Low precision data types in RL, Muon or pSGD` 


- **Model Agnostic Meta Learning 是必经之路**：一位成员指出，即使对于 **world foundation models**，**training structure**（训练结构）本身也是一种局限，并建议关注 **MAML (Model Agnostic Meta Learning)** 方法。
   - 他们补充说，将最终目标设定为一个函数会导致对齐问题和效用最大化问题。
- **RLHF 获得神经引导**：一位成员提到了 **neural-guided CoT**，随后询问这究竟是 **CoT-RLHF-ed models**，还是通过某种离散机制引导 **CoT**。
   - 另一位成员建议参考 [semanticscholar](https://aclanthology.org/2025.coling-main.719.pdf) 上的综述论文，以获取关于这一通用主题的更多信息。
- **精度问题困扰 RL Post Training**：一位成员询问关于 **low precision data types**（低精度数据类型）对基于 **RL** 的 Post Training 技术影响的研究。
   - 另一位成员回答说，*RL 阶段不适合尝试低精度*，因为可能存在堆栈技能问题，需要不断的重新调查。
- **重谈 Deep Frying**：一位成员询问了关于 **low precision data types** 对基于 **RL** 的 Post Training 技术影响的研究，另一位成员将这些问题与 *deep frying* 联系起来。
   - 另一位补充说，像 **Muon** 或 **pSGD** 这样的方法在相同任务上不会出现同样的问题（或者至少不会那么严重）。
- **使用 LLM Harness 评估 RAG 流水线**：一位成员询问是否可以在本地计算机上对 **RAG pipeline** 应用 **llm harness evaluation**。


  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1355338196690403441)** (3 messages): 

> `Causal Study Ideas, Functional Form Correctness` 


- **数据依赖型实验思路涌现**：一名成员考虑通过实验研究函数形式（functional form），以确定其是否符合底层的因果机制。
   - 他们假设，如果函数形式符合底层的因果机制，那么 **E** 应该完全取决于数据，并考虑将其作为一个**因果研究（causal study）**进行实验。
- **函数形式假设**：提出了一项假设，即函数形式必须符合底层的因果机制，**E** 才能完全取决于数据。
   - 这为针对现实世界因果过程进行模型的实证测试和验证提供了一条潜在途径。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1355255727521796297)** (7 messages): 

> `Anthropic biology post, Neuronpedia goes open source, Mechanism of factual recall, Attribution graphs` 


- **解锁 Anthropic 的“已知事实”电路！**：一名成员引用了 Anthropic 关于**已知事实电路（known fact circuits）**的 [生物学文章](https://transformer-circuits.pub/2025/attribution-graphs/biology.html#dives-hallucinations)，并对未发布 transcoders 和 **Haiku 权重**以方便回答相关问题表示遗憾。
   - 讨论中链接了一篇[近期论文](https://arxiv.org/pdf/2411.14257)及其对应的 **GitHub 仓库**，以便进一步探索。
- **Neel 关于神经网络数值的见解**：Neel Nanda 分享了他关于分析语言模型中**事实召回（factual recall）**的[早期工作](https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall)。
   - 该帖子包含了理解神经网络的资源，例如[入门指南](https://neelnanda.io/getting-started)、[论文阅读清单](https://neelnanda.io/mechanistic-interpretability/favourite-papers)，以及一个[通过 Neuronpedia 解释 Gemma 2 2B 的交互式演示](https://neuronpedia.org/gemma-scope)。
- **Neuronpedia 转向开源！**：**可解释性平台** Neuronpedia 现已采用 [MIT 协议开源](https://x.com/neuronpedia/status/1906793456879775745)，并使用 Eleuther 的 `Delphi`（原 sae-auto-interp）作为其**自动解释服务器（auto-interp server）**。
   - 公告中包含了 [GitHub 仓库](https://github.com/hijohnnylin/neuronpedia)、[公共数据集](https://neuronpedia-datasets.s3.us-east-1.amazonaws.com/index.html?prefix=v1/) 以及一篇总结 Neuronpedia 功能的 [博客文章](https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source) 链接。


<div class="linksMentioned">

<strong>提及链接</strong>:

<ul>
<li>
<a href="https://www.alignmentforum.org/posts/iGuwZTHWb6DFY3sKB/fact-finding-attempting-to-reverse-engineer-factual-recall">Fact Finding: Attempting to Reverse-Engineer Factual Recall on the Neuron Level (Post 1) — AI Alignment Forum</a>: 如果你是通过 3Blue1Brown 看到这里的，你好！如果你想了解更多关于神经网络解释的通用知识，这里有一些你可能会发现的资源……</li><li><a href="https://x.com/neuronpedia/status/1906793456879775745">来自 neuronpedia (@neuronpedia) 的推文</a>: 公告：我们正在开源 Neuronpedia！🚀 这包括我们所有的机械可解释性工具：可解释性 API、steering、UI、推理、autointerp、搜索，以及 4 TB 的数据 - 已被 35+ 研究引用...</li><li><a href="https://www.neuronpedia.org/blog/neuronpedia-is-now-open-source">Neuronpedia 现已开源 | The Residual Stream</a>: 面向所有人的免费可解释性工具。外加 4TB 数据集。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1355313633780830208)** (97 条消息🔥🔥): 

> `MMLU-pro 评估设置、Few-Shot 示例处理、修改 Utils 导致的 GPU 过载问题、IndexError 调试与解决、向 generate 函数传递额外参数` 


- **验证 MMLU-pro 数据集切分配置**：一位成员确认 **MMLU-pro eval** 使用 `test` 切分运行，其 Few-shot 示例源自 `validation` 切分，详见 [配置文件](https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml)。
   - 系统在采样前对 Few-shot 切分使用 `process_docs` 函数，以从正确的子集中获取 Few-shot 示例。
- **深入探讨 Harness 正则表达式匹配**：成员们讨论了 *lm-harness* 依赖正则表达式 `regex_pattern: 'answer is \(?([ABCDEFGHIJ])\)?'` 进行精确匹配指标（exact match metric）计算，而非更高级的方法。
   - 受 [OpenAI's evals suite](https://github.com/openai/evals) 启发，计划为基准测试添加 LLM as judge 选项，以实现更好的自定义。
- **排查 GPU 过载问题**：一位成员报告在修改了 *mmlu_pro* 的 `utils.py` 代码后出现 **GPU 过载问题**，即使在较小的 Batch Size 下也会遇到内存错误。
   - 修改后的代码采用了动态选项估计（dynamic choice estimation），与默认的预定义选项映射相比，这似乎增加了内存负载。
- **调查并修复任务中的 IndexError**：用户在从选项中移除某个选择时遇到了 `IndexError`，尽管代码看起来处理了所有情况。
   - 错误原因在于 `utils.py` 拥有 A-P 选项，而 MMLU-pro 最多只有 10 个选项，索引处理导致了该错误，需要通过调试器（debugger）逐步排查。
- **向模型生成过程传递额外参数**：用户讨论了压缩 Key/Value (KV) 缓存和实现对比束搜索（Contrastive Beam Search）的需求。
   - 系统支持通过任务 YAML 中的 `generation_kwargs` 向 `generate` 函数传递额外参数。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/8850ebc0e83d1188517a1495ae7811486f8038a7/lm_eval/tasks/mmlu_pro/_default_template_yaml">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml (commit 8850ebc)</a>：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/3816796ebdcb1e7102e2964fc23d8e7a1082eba3/lm_eval/tasks/mmlu_pro/_default_template_yaml#L18-L23)">lm-evaluation-harness/lm_eval/tasks/mmlu_pro/_default_template_yaml (commit 3816796)</a>：一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1355255258082709707)** (163 条消息🔥🔥): 

> `xAI 收购 X (Twitter)、Midjourney 进军 LLM 领域、GPT-4o 推理能力、Llama 4 发布推测、Hermes 模型系统提示词 (System Prompt) 使用`

- **X 标志着终点：xAI 收购 Twitter**：Elon Musk 宣布 **xAI** 已通过全股票交易与 **X** (Twitter) 合并，此次交易对 xAI 的估值为 **800 亿美元**，对 X 的估值为 **330 亿美元**，旨在整合数据、模型、算力、分发和人才。
   - 正如讨论中所提到的，此举被推测可能有助于 X 避免支付最初收购 Twitter 产生的债务利息，并能为 **Grok** 提供更好的数据抓取和训练。
- **Midjourney 的文本转型：进军 LLM 领域**：以 AI 图像生成闻名的 **Midjourney** 正在向 LLM 领域扩张，并与 NYU 合作发布了一篇[研究论文](https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/)，探讨如何训练 **Llama** 和 **Mistral** 等 LLM 进行更具创造性的写作。
   - 这标志着 Midjourney 致力于超越图像生成实现多元化发展，并开发自己的计算和 AI 硬件的野心。
- **GPT-4o 变得更聪明：推理能力显现！**：观察到 **GPT-4o** 展示出推理能力，引发了人们对其是正在开发的 [GPT-5 系统](https://fxtwitter.com/koltregaskes/status/1905907926331539794)一部分的猜测，且不断有工具和更新加入。
   - 一位成员指出，它甚至可以*在回答过程中决定开始进行推理*。
- **发现 Llama 4！发布在即？**：据报道，代号为 **cybele、themis 和 spider** 的三个新模型在竞技场（arena）上的表现像是为 elomaxxing 量身定制的，这可能预示着 Llama 4 的发布候选版本即将推出。
   - 推测 **Meta** 将在官方活动之前发布，效仿 Llama 3 在 4 月 18 日的突击发布，以避免在模型性能上被超越。
- **Hermes 的提示词处方：系统在先，用户在后**：根据其模型卡（model card），对于 **Hermes 模型**，建议使用特定的提示词格式，即在开头仅使用一次 system 角色，后续消息使用 user 角色。
   - 一位成员指出，任何适用于 OpenAI API 的教程也应该适用于 Nous API。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?t=s2S595eV_11U1l7BmZkpVg&s=19">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：GPT-4o 被发现具有推理能力！对我来说，这就是在我们眼前构建的 GPT-5 系统。请看我在第一条评论中的帖子。预计所有模型都会增加更多工具和更新。这 ...</li><li><a href="https://fxtwitter.com/TheTuringPost/status/1906304408415359067?t=QrP_I5vSzaLt-3r42Hyyig&s=19">来自 TuringPost (@TheTuringPost) 的推文</a>：9 种多模态思维链（Chain-of-Thought）方法 ▪️ KAM-CoT ▪️ 多模态思维可视化 (MVoT) ▪️ 组合式 CoT (CCoT) ▪️ URSA ▪️ MM-Verify ▪️ 职责区分 CoT (DDCoT) ▪️ Multimodal-CoT ▪️ Graph-of-Thoug...</li><li><a href="https://fxtwitter.com/sama/status/1906793591944646898?s=46">来自 Sam Altman (@sama) 的推文</a>：摘要：我们很高兴在未来几个月发布一个强大的、具有推理能力的全新开源权重语言模型，我们想与开发者讨论如何使其发挥最大效用：https://openai.com/op...</li><li><a href="https://venturebeat.com/ai/midjourneys-surprise-new-research-on-making-llms-write-more-creatively/">Midjourney 的惊喜：关于让 LLM 写作更具创造性的新研究</a>：从经典的基于 Transformer、以文本为核心的 LLM 中，在认知和性能方面仍有很大的潜力可挖。</li><li><a href="https://unusualwhales.com/news/xs-valuation-is-back-to-44-billion">X 的估值重回 440 亿美元</a>：在经历大幅贬值后，X 现在的估值为 440 亿美元——与 Elon Musk 在 2022 年收购该平台（前身为 Twitter）时支付的金额相同。据报道，这一估值 ...</li><li><a href="https://fxtwitter.com/koltregaskes/status/1905907926331539794?">来自 Kol Tregaskes (@koltregaskes) 的推文</a>：GPT-4o 被发现具有推理能力！对我来说，这就是在我们眼前构建的 GPT-5 系统。请看我在第一条评论中的帖子。预计所有模型都会增加更多工具和更新。这 ...</li><li><a href="https://x.com/farouqaldori/status/1906130990877012342?s=46">来自 Farouq Aldori (@FarouqAldori) 的推文</a>：@Teknium1 抱歉，这是假新闻。使用了你截图中的完全相同的提示词，他们对提示词进行了预处理。</li><li><a href="https://www.cnbc.com/2025/03/28/elon-musk-says-xai-has-acquired-x-in-deal-that-values-social-media-site-at-33-billion.html">Elon Musk 表示 xAI 已收购 X，交易对该社交媒体网站的估值为 330 亿美元</a>：在周五的一条社交媒体帖子中，Elon Musk 表示他的初创公司 xAI 已收购了他的社交媒体公司 X。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1355277531367870494)** (11 条消息🔥): 

> `OLMoE Fine-tuning, Unsloth, Axolotl, Docker, Weaviate` 


- **OLMoE Instruct 模型发布**：AllenAI 发布了 **OLMoE-1B-7B-0125-Instruct** 模型，这是 [OLMoE-1B-7B January 2025](https://huggingface.co/allenai/OLMoE-1B-7B-0125) 模型的监督微调（SFT）变体。它使用了 [Tülu 3 数据集](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct/blob/main/allenai/tulu-3-sft-olmo-2-mixture) 的变体进行训练，并在 [此数据集](https://huggingface.co/datasets/allenai/olmo-2-1124-13b-preference-mix) 上进行了进一步的 DPO 训练，最后使用 [此数据](https://huggingface.co/datasets/allenai/RLVR-GSM) 进行了 RLVR 训练。
   - [OLMoE 论文](https://arxiv.org/abs/2409.02060) 和 [Tülu 3 论文](https://arxiv.org/abs/2411.15124) 提供了更多细节。
- **Unsloth 是微调的最佳方式吗？**：成员们讨论了哪些工具最适合微调模型，*Axolotl*、*Llama Factory* 和 *Unsloth* 的 Notebook 被提及为首选方案。
   - 一位成员确认他们正是通过 **Axolotl** 入门的。
- **Docker 磁盘镜像迁移困境**：一位成员寻求帮助将 Docker 磁盘镜像移动到另一个驱动器，尽管更改了 Docker Desktop 中的路径，但在更新 Docker 根目录时遇到了问题。
   - 该成员试图将 **Weaviate** 连接到另一个驱动器上的磁盘；另一位成员建议他们研究一些 **API**，让 LLM 和 **Weaviate**（在 Docker 内部）进行通信。



**提及的链接**：<a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct">allenai/OLMoE-1B-7B-0125-Instruct · Hugging Face</a>：未找到描述

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

burnytech: https://fxtwitter.com/iScienceLuvr/status/1905730169631080564
  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1355346224903098368)** (6 条消息): 

> `OpenAI image generation, Multiscale Structure in Image Generation, Grok vs. OpenAI Image Generation` 


- **OpenAI 图像生成：多尺度秘密曝光！**：根据 [这条推文](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492)，分析 **OpenAI 图像生成** 的帧揭示了一种多尺度结构，证据倾向于交错潜变量自回归（interleaved latent autoregression）而非拉普拉斯金字塔（Laplacian pyramid），并通过跨尺度的非因果扩散（non-causal diffusion）进行解码。
- **逐行扫描 UI：一个欺骗性的表象？**：根据 [Nayan Saxena](https://fxtwitter.com/SaxenaNayan/status/1905334927526105492) 的说法，**OpenAI 图像生成** 中的逐行扫描只是 UI 效果，每一帧都反映了通过从粗到细的多尺度扩散进行的全局更新，而不是基于 Patch 的自回归（AR）。
   - 分析表明逐行扫描 *纯粹是 UI 效果*。
- **Grok 的图像伪影：基于 Patch 的自回归迹象？**：据推测 **Grok** 使用的是纯自回归模型，输出 Patch（又名 VQ-GAN / Parti），这可能解释了由于重复结构导致的明显伪影。
   - 一位成员指出，出于某种原因，**Grok** 在生成图像方面的表现似乎也差得多。



**提及的链接**：<a href="https://fxtwitter.com/SaxenaNayan/status/1905334927526105492">Nayan Saxena (@SaxenaNayan) 的推文</a>：分析 OpenAI 图像生成帧显示了多尺度结构：拉普拉斯增量突出了迭代的频带编辑，熵定位以及流偏移。证据支持交错潜变量自回归...

  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/)** (1 条消息): 

burnytech: https://fxtwitter.com/iScienceLuvr/status/1905730169631080564
  

---


### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1355377427819266088)** (3 条消息): 

> `Open Reasoning Tasks, Proprietary Models` 


- **推理任务邀请**：一位成员建议查看开放推理任务。
   - 另一位成员确认他们会去查看。
- **检查任务**：Walkerdev 表示他正在检查推理任务。

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1355511552404492319)** (3 messages): 

> `TRL and Accelerate, Nvidia Ampere GPU Thread Performance, GPU Kernel Latency Hiding` 


- **TRL 在幕后利用 Accelerate**：一位成员指出 **TRL** 在后台使用 **Accelerate** 来处理复杂操作，从而简化用户体验。
   - 其意图是抽象掉底层细节，让用户专注于训练。
- **Ampere GPU 线程数超出预期**：一位成员计算出拥有 96 个 SM（每个 SM 有 4 个 warp schedulers）的 **Nvidia Ampere GPU** 理论上应支持 **12288 个线程**，但观察到的性能提升一直持续到 **24576 个线程**。
   - 该成员质疑 **kernel latency hiding** 是否允许在每个 SM 上同时调度两倍的 core。
- **Geohot 的 GPU Noob Kernel 分析**：一位成员正在分析 [Geohot 的 GPU Noob kernel](https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54) 以理解线程性能。
   - 他们质疑该 kernel 在 latency hiding 方面的潜力，想知道这是否能解释观察到的线程数提升。



**提及的链接**：<a href="https://github.com/geohot/gpunoob/blob/master/src/main.rs#L54">gpunoob/src/main.rs at master · geohot/gpunoob</a>：来自 Stream 关于 GPU 如何工作的 Noob 课程。通过在 GitHub 上创建一个账号来为 geohot/gpunoob 的开发做出贡献。

  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1355507202466512907)** (7 messages): 

> `emulated dot scaled triton performance, L1 cache use, persistent kernels` 


- **Triton 的模拟 Dot Scaled 损害性能**：一位用户报告称，在 **H100** 上使用 Triton 的模拟 `dot_scaled` 函数（默认行为是向上转型为 `bf16`）会损害性能。
   - 他们询问是否有办法将类型向上转型为 `fp8`，并链接到了 [Triton 文档](https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html) 作为参考。
- **在 Triton 中利用 L1 Cache 掌握矩阵乘法**：一位用户询问是否可以将整个矩阵加载到 **L1 cache** 中并在 Triton 的单个 **SM** 上进行处理，并质疑 streaming blocks 是否是强制性的。
   - 一位专家澄清说，Triton 抽象掉了 shared memory 管理和 **SM** 调度，建议用户尝试不同的 block sizes，并推荐查看 [triton-puzzles 中的 attention kernels](https://openai.com/index/triton/) 或 [unsloth/liger kernels](https://github.com/unslothai/unsloth) 的实现示例。
- **L1 缓存行为解析**：一位用户询问对同一矩阵的后续 `tl.load` 调用是否会从 **L1 cache** 而不是 **HBM** 中检索。
   - 一位专家解释说，`tl.load` 操作将数据带入 registers 并可能将其缓存在 **L1** 中，如果数据未被驱逐，后续加载可能会命中 **L1**，并强调 **L1** cache 的重用在不同的 kernel 启动之间是不保证的。
- **Persistent Kernel 在 H100 上的性能**：一位用户分享了他们在花了一天时间编写 quant persistent split-K kernel 后，在 **H100** 上仅获得轻微提升（在 M 到 2xM tokens/sec 之间）的经验。
   - 他们正在寻求关于 persistent kernels 能提供更好提升的设置见解，特别是提到了 **M** 值和设备考量。



**提及的链接**：<a href="https://triton-lang.org/main/python-api/generated/triton.language.dot_scaled.html">triton.language.dot_scaled &mdash; Triton 文档</a>：未找到描述

  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1355814984138887248)** (4 messages): 

> `FlashAttention, CUDA C Programming Guide, Nsight Systems (nsys), Nsight Compute (ncu), Memory Coalescing` 


- **FlashAttention 内存访问困惑依然存在**：一位成员对 **FlashAttention 中的内存访问模式**表示困惑，特别是关于为了 **128-bit 内存传输**而进行数据重塑（reshaping）的必要性。
   - 该成员引用了 **CUDA C Programming Guide** 的第 5.3 节，并质疑编译器是否能正确识别内存合并（memory coalescing）的机会。
- **Nsight 工具对性能分析（profiling）非常有用**：一位成员建议使用 **Nsight Systems (nsys)** 和 **Nsight Compute (ncu)** 来分析性能瓶颈，并推荐通过命令行生成报告以便进行可视化。
   - 他们表示 *前者允许你查看 kernel 时间线和一些性能指标，而后者则分析性能瓶颈并提供一些优化建议*。
- **PTX 编译器处理内存布局**：一位成员澄清说，**PTX 编译器**管理寄存器中的数据布局，以确保线程可以通过一条指令将 **128 位连续数据**写入单个对齐的 gmem 地址。
   - 他们补充说，*即使使用（内联）PTX，也无需担心这个问题*。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1355790639966720204)** (3 messages): 

> `torch.compile error, FlexAttention, Arbitrary Sequence Length` 


- **Torch Compile 抛出不支持错误**：一位用户报告了在 Colab 中使用 **torch 2.6** 和 **cuda 12.4** 时，在编译函数内使用子类化的 `nn.Parameter` 会导致与 `__rmul__` 相关的 `torch.compile` 错误。
   - 错误信息为：`Unsupported: call_method UserDefinedObjectVariable(b) __rmul__ [ConstantVariable(int: 3)] {}`，用户想知道这是一个已知问题还是应该提交一个 issue。
- **FlexAttention 现在支持任意序列长度**：一位用户询问 **FlexAttention** 现在是否支持任意序列长度，因为他记得之前的版本要求序列长度必须是 **128** 的倍数。
   - 另一位用户确认，从 **PyTorch 2.6** 开始，已支持任意序列长度。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1355717424409874655)** (1 messages): 

> `RoPE, BFloat16 Precision, FlashAttention2, AnchorAttention` 


- **RoPE 的 BFloat16 失效分析**：一篇新论文（[When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training](https://arxiv.org/abs/2411.13476)）指出，**BFloat16** 会在 **RoPE** 中引入数值误差，从而损害其相对编码能力，即使是在 **Float32** 中进行计算也是如此。
   - 随着上下文长度的增加，第一个 token 会显著导致偏差，但该论文介绍了一种名为 **AnchorAttention** 的即插即用方法，它可以提高长上下文性能，缩短 50% 以上的训练时间，并保留模型的通用能力，支持 **FlashAttention** 和 **FlexAttention** 的代码已在 [GitHub](https://github.com/haonan3/AnchorContext) 上发布。
- **FlashAttention 受 RoPE 的 BFloat16 问题影响**：论文指出，在 **FlashAttention2** 中将张量转换为 **BFloat16** 会导致 **RoPE** 偏离其预期的相对位置编码特性。
   - 这意味着虽然 **RoPE** 可能是用 **Float32** 计算的，但在随后的层（如 **FlashAttention2**）中使用 **BFloat16** 仍然会引入误差。



**提到的链接**：<a href="https://x.com/Haonan_Wang_/status/1859608786765480516">Haonan Wang (@Haonan_Wang_) 的推文</a>：🚀 新论文📜 When Precision Meets Position: BFloat16 Breaks Down RoPE in Long-Context Training🤯 RoPE 失效了，因为... BFloat16！&gt; 即使 RoPE 是用 Float32 计算的（如 Llama 3 和 t...

  

---

### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1355405681737469952)** (2 条消息): 

> `NVIDIA RTX PRO 6000 Blackwell Workstation Edition, GDDR7, Size Zheng, Next Era of AI` 


- **Nvidia 发布 RTX PRO 6000 Workstation Edition**：Nvidia 宣布了 **RTX PRO 6000 Blackwell Workstation Edition**，配备 **96GB GDDR7** 显存，面向 AI 和图形密集型任务，详见其 [产品页面](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/)。
- **作者展示 AI 的下一个时代**：作者 [Size Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+S), [Jin Fang](https://arxiv.org/search/cs?searchtype=author&query=Fang,+J), [Xuegui Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+X), [Qi Hou](https://arxiv.org/search/cs?searchtype=author&query=Hou,+Q), [Wenlei Bao](https://arxiv.org/search/cs?searchtype=author&query=Bao,+W), [Ningxin Zheng](https://arxiv.org/search/cs?searchtype=author&query=Zheng,+N), [Ziheng Jiang](https://arxiv.org/search/cs?searchtype=author&query=Jiang,+Z), [Dongyang Wang](https://arxiv.org/search/cs?searchtype=author&query=Wang,+D), [Jianxi Ye](https://arxiv.org/search/cs?searchtype=author&query=Ye,+J), [Haibin Lin](https://arxiv.org/search/cs?searchtype=author&query=Lin,+H), [Li-Wen Chang](https://arxiv.org/search/cs?searchtype=author&query=Chang,+L), [Xin Liu](https://arxiv.org/search/cs?searchtype=author&query=Liu,+X) 在其 [即将发表的论文](https://arxiv.org/abs/2503.20313) 中展示了 AI 的下一个时代。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2503.20313">TileLink: Generating Efficient Compute-Communication Overlapping Kernels using Tile-Centric Primitives</a>: 大型深度学习模型在广泛的任务中取得了 state-of-the-art 的性能。这些模型通常需要分布式系统来进行高效的训练和推理。基本的...</li><li><a href="https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/">NVIDIA RTX PRO 6000 Blackwell Workstation Edition</a>: 体验无与伦比的 AI 和图形性能。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1355390972451881082)** (7 条消息): 

> `Jax Scaling Book on Transformer FLOPs, Diffusion Game Models, Pulid Faceloader Error, Apple Silicon Memory Hierarchy, Models with Large Per-Layer Dataflow` 


- **Jax Scaling Book 教授 Transformer FLOPs 计数**：一位成员分享了 [jax-ml/scaling-book](https://jax-ml.github.io/scaling-book/transformers/)，该书提供了自回归模型的计算示例，适用于视频模型，通过 **FLOPs、内存带宽和 roofline analysis** 来估算模型约束。
   - 建议针对真实数据进行基准测试，并使用 *nsys* 进行 profile 以验证计算结果，重点关注线性层和 attention 机制。
- **Pulid Faceloader 遇到 CUDA 问题**：一位用户报告称，ComfyUI 中的 Pulid Faceloader 在重启后出现 CUDA 错误，尽管路径设置正确，并引用了一个 [onnxruntime issue](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements)。
   - 建议检查 CUDA 和 cuDNN 版本是否兼容，以及 GPU 是否受该 CUDA 版本支持（目前在 **PyTorch 2.7.0** 与 **CUDA 12.8** 上运行失败）。
- **Silicon 秘密：M-Series 内存揭秘？**：一位成员询问了 Apple Silicon M-Series GPU 的片上缓存和内存层级，寻求与 NVIDIA A100 内存映射等效的 Apple 实现，并链接了一篇关于 [Apple M-Series SoCs 的论文](https://arxiv.org/abs/2502.05317v1)。
   - 讨论指出，Apple 不像 NVIDIA 那样公开某些 GPU 细节，因此难以确定具体的缓存数值，但论文提到了 M4 芯片中的 **L1 缓存（每核心 192 KB）** 和高达 **24 MB 的共享 L2 缓存**。
- **寻找具有高吞吐量层数据流的模型**：一位成员寻找每层数据流（而非总内存占用）至少约为 **10GB** 的模型，并描述了如何测量连续层之间传递的中间激活值（intermediate activations）。
   - 一个建议是探索处理体积数据（volumetric data）的模型，例如在医学领域，**512³ 体素**、**32 通道**和 **fp16 激活值**的数据量每层可产生 **8GiB** 数据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://jax-ml.github.io/scaling-book/transformers/"> All the Transformer Math You Need to Know | How To Scale Your Model </a>：未找到描述</li><li><a href="https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#requirements),">NVIDIA - CUDA</a>：使用 CUDA 执行 ONNX Runtime 应用程序的说明</li><li><a href="https://arxiv.org/abs/2502.05317v1">Apple vs. Oranges: Evaluating the Apple Silicon M-Series SoCs for HPC Performance and Efficiency</a>：本文研究了 Apple Silicon M-Series SoCs（M1, M2, M3 和 M4）在 HPC 方面的架构特征和性能潜力。我们对 CPU 和 GPU 设计进行了详细回顾...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1355341294989480017)** (3 条消息): 

> `Twitter Embeds, Deleting Twitter` 


- **Twitter 嵌入被删除**：一位成员分享了一个关于 **Twitter 嵌入**无法工作的[链接](https://vxtwitter.com/faisal_sayed05/status/1905519905845239869)。
   - 另一位成员开玩笑说，解决 **Twitter 嵌入**问题的办法就是“注销你的 Twitter”。
- **不再使用 Twitter**：另一位成员建议某人注销他们的 Twitter 账号。
   - 原贴者当时正尝试发布一个指向 **vxtwitter.com** 的链接。



**提到的链接**：<a href="https://vxtwitter.com/faisal_sayed05/status/1905519905845239869">来自 undefined 的推文</a>：未找到描述

  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

random.oof: NYC 有线下聚会吗？
  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1355958801714642974)** (1 条消息): 

> `Triton-lang, shared memory encoding, transpose bank conflict` 


- **添加共享内存编码以避免转置 Bank 冲突**：一个 [pull request](https://github.com/triton-lang/triton/pull/5797) 为 **TN GEMM** 中的 **B 操作数** 引入了 **swizzling 模式**。
   - 该实现最初由 @jtang10 在 [PR#4984](https://github.com/triton-lang/triton/pull/4984) 中完成。
- **共享内存中转置 GEMM 操作数优化**：一个 [pull request](https://github.com/triton-lang/triton/pull/6074) 引入了 **共享内存优化**，减少了 **bank 冲突**，并在 **NT**、**TT** 和 **TN GEMM** 及类似情况下实现了 **宽 LDS 存储 (wide LDS stores)**。
   - 当 dot 操作数的 **K 维度** 不是最内层时，此优化适用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/triton-lang/triton/pull/5797">[AMD] Add shared memory encoding to avoid transpose bank conflict by binarman · Pull Request #5797 · triton-lang/triton</a>: 此 PR 为 TN GEMM 中的 B 操作数引入了 swizzling 模式。最初由 @jtang10 在 #4984 中实现。此 PR 是系列相关 PR 的一部分：[AMD] Add shared memory encoding to avoid tran...</li><li><a href="https://github.com/triton-lang/triton/pull/6074">[AMD][OPTIMIZER] Optimize transposed GEMM operand in shared memory  by binarman · Pull Request #6074 · triton-lang/triton</a>: 此 PR 引入了共享内存优化，减少了 bank 冲突，并在 NT、TT 和 TN GEMM 及类似情况（即 dot 操作数 K 维度不是最内层）下实现了宽 LDS 存储。这...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/1355939141740920932)** (3 条消息): 

> `Segmentation fault with liger-kernel, LigerFusedLinearCrossEntropyLoss issues, Reproducing errors with liger-kernel, Environment details for debugging liger-kernel` 


- **Liger-Kernel 出现分段错误 (Segmentation Fault)**：一位成员报告在简单的 PyTorch 脚本中将 `liger-kernel` 与 `LigerFusedLinearCrossEntropyLoss` 配合使用时遇到 **Segmentation fault (core dumped)**。
   - 该脚本涉及线性层模型、输入张量和目标张量，错误发生在 `loss.backward()` 调用期间。
- **调试 Liger-Kernel 问题**：维护者无法复现该分段错误，并请求提供完整的错误代码和环境详情以协助调试。
   - 另一位成员询问了正在使用的 `liger-kernel` 版本，以帮助定位问题。
- **调查 LigerFusedLinearCrossEntropyLoss 问题**：报告的问题集中在 `liger-kernel` 库中的 `LigerFusedLinearCrossEntropyLoss` 函数。
   - 该函数融合了线性层和交叉熵层，通过分块计算 (chunk-by-chunk computation) 来减少内存占用，但在某些配置下似乎会触发分段错误。


  

---


### **GPU MODE ▷ #[metal](https://discord.com/channels/1189498204333543425/1285384841730457600/1355269257881194667)** (6 条消息): 

> `CUDA compiler for Apple GPU, Metal C++, Zig support for Apple GPU, spirv-cross, IREE's Metal HAL driver` 


- **针对 Apple GPU 的 CUDA 编译器**：通过像 [这个](https://github.com/openai/triton) 这样的仓库，理论上可以通过编译为 **Metal C++** 来为 **Apple GPU** 构建 **CUDA 编译器**。
   - 一位成员表示，他们考虑过为 **Zig** 添加类似支持，因为 **Apple** 使用 **LLVM IR**。
- **通过 IREE 实现 Metal 计算着色器**：一位成员建议使用 [IREE 的 Metal HAL 驱动](https://iree.dev/developers/design-docs/metal-hal-driver/#compute-pipeline) 中的计算着色器来针对 **Metal** 进行开发。
   - 他表示 *它是可行的*，尽管显然受限于 **SPIRV** 所能表达的内容以及 **SPIRV-cross** 所支持的内容。



**提及的链接**：<a href="https://iree.dev/developers/design-docs/metal-hal-driver/#compute-pipeline)">Metal HAL driver - IREE</a>：未找到描述

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1355548515543285881)** (4 messages): 

> `Bend parallel language, Phazr AI video avatar tool, CuTe predication, File I/O in Bend` 


- **使用并行性让你的代码在 **Bend** 中成型**: HigherOrderCo 推出了 **Bend**，这是一种面向多核 CPU/GPU 的[大规模并行、高级编程语言](https://higherorderco.com/)，旨在提供类似 Python 的编程体验，同时避免并发编程的复杂性。
- **Phazr 炼金你的视频形象**: 一位成员发布了 **Phazr AI**，这是一个[免费工具](https://www.phazr.ai/)，允许用户在视频通话中以任何人的形象出现，它利用音频驱动的肖像动画技术，并为了隐私在本地运行。
- **分块（Tiling）的胜利：CuTe predication 教程发布**: Simon Veitner 发布了一篇[博客](https://veitner.bearblog.dev/predication-in-cutlass/)，介绍如何在 CuTe 中执行 predication 以帮助泛化 tiling kernels，其中包含一个[指向 Cutlass 文档的链接](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/0y_predication.md)。
- **Bend 增加文件 I/O**: **Bend** 引入了 **File I/O** 功能，使用户能够执行文件操作，详见[此处文档](https://github.com/HigherOrderCO/Bend/blob/main/docs/builtins.md#file-io)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.phazr.ai/">phazr</a>: 未找到描述</li><li><a href="https://higherorderco.com/">Higher Order Company</a>: 未找到描述</li><li><a href="https://veitner.bearblog.dev/predication-in-cutlass/">Predication in Cutlass</a>: CuTe 的 Cutlass 文档简要涉及了该主题，但没有提供完整的代码示例。在这篇博文中，我将解释如何使用 predication...</li><li><a href="https://github.com/HigherOrderCO/Bend/blob/main/docs/builtins.md#file-io">Bend/docs/builtins.md at main · HigherOrderCO/Bend</a>: 一种大规模并行、高级编程语言 - HigherOrderCO/Bend
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1356365091754344558)** (1 messages): 

> `AlphaGeometry LLM + verifier for kernel optimization` 


- **询问用于 Kernel 优化的 AlphaGeometry LLM**: 一位成员询问关于使用 **AlphaGeometry 风格的 LLM + 验证器** 进行 kernel 优化过程的情况。
   - 他们正在寻求有关此想法的历史信息、是否有人尝试过以及任何相关讨论，因为他们是该领域的新手，怀疑自己可能正在 *重新发现现有概念*。
- **未探索的领域：使用 AlphaGeometry LLM 进行 Kernel 优化**: 讨论围绕利用 **带有验证器的 AlphaGeometry 风格 LLM** 来潜在地彻底改变 kernel 优化过程展开。
   - 该询问重点在于了解这种方法之前是否被探索过，并寻求社区内先前尝试或相关讨论的见解，意识到重新审视现有方法的可能性。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1356088943392849920)** (2 messages): 

> `FusedMLP, tiny-cuda-nn, ThunderKittens` 


- **用户询问 ThunderKittens 中是否存在 FusedMLP**: 一位用户询问 [HazyResearch/ThunderKittens](https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/hedgehog) 中是否存在来自 [NVlabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 形式的 **FusedMLP**。
- **新手在 TK 领域寻求指导**: 一位自称新手的用户询问如何在 ThunderKittens 仓库中找到特定的实现（FusedMLP）。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens/tree/main/kernels/hedgehog">ThunderKittens/kernels/hedgehog at main · HazyResearch/ThunderKittens</a>: 用于快速 kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework</a>: 极速 C++/CUDA 神经网络框架。通过在 GitHub 上创建账号来为 NVlabs/tiny-cuda-nn 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1356276080582987888)** (26 messages🔥): 

> `缺乏课程设置的数据集，Futoshiki 数据集生成速度，私有基准测试服务，OpenAI 开源权重推理模型` 


- ****数据集缺乏课程设置，难度调整引发争议****：存在一些没有课程设置（curricula）的数据集（`acre`, `arc_1d`, `arc_agi`, `codeio`, `composite`, `countdown`, `futoshiki`, `gcd`, `gsm_symbolic`, `knight_swap`, `knights_knaves`, `list_functions`, `puzzle24`, `syllogism`, `word_sorting`），其中像 `gsm_symbolic` 和 `list_functions` 在为课程设计调整难度时面临挑战，目前正在进行 bug 调查。
   - 一些成员正专注于冲突（collision）任务，并报告了 `gsm_symbolic` 等特定数据集的问题，引发了关于配置错误和修复的讨论。
- ****Futoshiki 的速度引发对样本量的审查****：`futoshiki` 数据集在合理时间内生成 10,000 个数据集时面临挑战（10 分钟不够用），导致了对可接受生成速度的质疑以及对网格大小配置的调整。
   - 有建议称，如果想快速生成大量样本，应将最大网格大小设置为 **6 或 7**；理论上，对于更大的网格大小，冲突发生的概率本来就会低得多。
- ****私有基准测试服务蓝图****：一个[正在进行中的 Pull Request](https://github.com/open-thought/reasoning-gym/pull/398) 旨在创建一个私有基准测试服务，用户可以填写空白处并将结果上传到 Gradio 进行评分，确保不泄露敏感信息。
   - 该计划涉及生成一套完整的带有空白答案的问题，以实现 *隐藏答案（hidden-answer）* 的基准测试服务，从而实现更受控的评估过程。
- ****OpenAI 开源权重公告令观察者感到震惊****：OpenAI 宣布发布强大的开源权重推理模型令社区感到意外，引发了对其动机的猜测，特别是在潜在融资背景下。
   - 发布开源权重模型如果能引发广泛关注和采用，可能会显著提高其估值，因为每个人都会为此疯狂。



**提到的链接**: <a href="https://github.com/open-thought/reasoning-gym/pull/398">[WIP] Generate Seeded Benchmark Test for Distribution by Miserlou · Pull Request #398 · open-thought/reasoning-gym</a>: 添加了一个脚本来创建一套完整的带有空白答案的问题，可用于隐藏答案的基准测试服务。RNG_SEED=321 python scripts/generate_benchmark.py --num-per-datase...

  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1356081760882000064)** (2 messages): 

> `Discord ID 显示问题，Discord 权限，排行榜 ID 格式化` 


- **Discord ID 显示之谜**：一位用户询问为什么他们在排行榜上的 ID 显示为 **User_1184712546704429106** 而不是他们真实的 Discord ID。
   - 社区怀疑这与 **Discord permissions** 有关，但目前仍未找到解决方案。
- **Discord 权限导致 ID 显示问题**：成员们认为 **Discord perms** 导致了 ID 显示问题。
   - 目前对于如何修复此问题**毫无**头绪。 


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1355267324776808651)** (94 messages🔥🔥): 

> `vectoradd 基准测试，vectorsum 基准测试，conv2d 基准测试` 


- **Vectoradd 基准测试在 H100 上爆发**：记录了多个在 **H100** GPU 上使用 Modal 运行器成功提交的 `vectoradd` 排行榜结果，包括 ID `3247`, `3248`, `3255`, `3256`, `3257`, `3258`, `3259`, `3351`, `3353`, `3367`, `3368` 和 `3369`。
- **Vectorsum 分数在 L4 上飙升**：报告了大量在 **L4** GPU 上使用 Modal 运行器成功提交的 `vectorsum` 基准测试和排行榜结果，ID 范围从 `3272` 到 `3322`，以及从 `3352` 到 `3372`。
- **Conv2d 竞争在 L4, T4, A100, H100 上被征服**：ID 为 `3373` 的排行榜提交在 **L4, T4, A100, H100** GPU 上使用 Modal 运行器成功提交至 `conv2d` 排行榜！
- **Vectorsum 测试在 H100 上表现诱人**：ID 为 `3374` 的排行榜提交在 **H100** GPU 上使用 Modal 运行器成功提交至 `vectorsum` 排行榜。
- **A100 完美通过 Vectoradd**：ID 为 `3338` 的测试提交和 ID 为 `3288` 的排行榜提交在 **A100** GPU 上使用 Modal 运行器成功提交至 `vectoradd` 排行榜。

  

---

### **GPU MODE ▷ #[hardware](https://discord.com/channels/1189498204333543425/1349152646484987974/1355308604462338280)** (13 messages🔥): 

> `PyTorch 分布式训练中的 GPU 温度问题，训练前的 GPU 健康检测，AWS 上的 H100 GPU 温度异常` 


- **AWS H100 热点排查**：一位用户报告在 AWS **H100** 节点上进行 **PyTorch distributed training** 时，特定 GPU 出现高温，其中一块 GPU 持续达到 **90C**，而其他 GPU 平均为 **40C**。
   - 该用户指出这种温度异常会减慢训练速度，并寻求关于训练前硬件/软件健康检查的建议，例如 NCCL 或连接检查。
- **通过功率限制缓解高温**：针对在 PyTorch 分布式训练中遇到 GPU 高温的用户，建议使用 `sudo nvidia-smi -pl` 来[对 GPU 进行功率限制](https://developer.nvidia.com/nvidia-system-management-interface)。
   - 建议认为这可以缓解温度担忧。
- **针对持续的 GPU 散热问题寻求 AWS 支持**：对于在 AWS 特定 **H100** 节点上持续遇到 GPU 高温的用户，建议寻求 AWS 支持的帮助。
   - 建议指出，如果问题是散热器的机械故障，压力测试可能是训练前检测到该问题的唯一方法。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1355269093397364816)** (64 messages🔥🔥): 

> `Softmax AI 对齐初创公司，xAI 与 X 合并，GPT-4o 图像生成，Gemini 2.5 Pro，MiniMax Audio Speech-02 模型` 


- **Shear 的天才之作：Softmax 致力于 AI 对齐**：Emmett Shear、Adam Goldstein 和 David Bloomin 在旧金山创立了 **Softmax**，这是一家拥有 10 人的初创公司，专注于**有机对齐 (organic alignment)**，通过借鉴自然和智能系统将人类与 AI 的目标融合，详情见 [Core Memory 文章](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax)。
- **X 标记 AI 阵地：xAI 与 X 合并**：Elon Musk 宣布 **xAI** 正在“将 xAI 先进的 AI 能力和专业知识与 X 的巨大影响力相结合”，此次合并详情由 [The Verge](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition) 报道。
- **GPT-4o 的图像生成：前端幻觉？**：一位用户发现 **GPT-4o** 逐行生成图像的效果是浏览器端的动画，根据[这条推文](https://x.com/jie_liu1/status/1905761704195346680)，服务器仅发送了 **5 张中间图像**，patch size 为 **8**。
- **Gemini 2.5 Pro 进入实验阶段，扩大访问范围**：由于 TPUs 运行火热，**Gemini 2.5 Pro**（实验版）现已向所有 Gemini 用户开放，正如 [GeminiApp 的 Twitter](https://fxtwitter.com/GeminiApp/status/1906131622736679332) 所宣布的那样。
- **MiniMax 发布带有 TTS 功能的 Audio Speech-02**：**MiniMax AI** 推出了 **Speech-02**，它可以立即将任何文件或 URL 转换为具有 30 多种语言原生风格的逼真音频，支持无限语音克隆和亚秒级流式传输，详情见 [MiniMax 的 Twitter](https://fxtwitter.com/MiniMax__AI/status/1906720764885180775)。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

- [来自 Jie Liu (@jie_liu1) 的推文](https://x.com/jie_liu1/status/1905761704195346680)：在破解了 GPT-4o 的前端后，我有了惊人的发现：💡用户看到的逐行图像生成效果只是浏览器端的动画（纯前端技巧）🔦OpenAI 的服务器发送的是...
- [来自 Runway (@runwayml) 的推文](https://fxtwitter.com/runwayml/status/1906718935778545964)：今天我们推出了 Gen-4，这是我们全新的最先进 AI 模型系列，用于媒体生成和世界一致性。Gen-4 在忠实度、动态运动和控制方面迈出了重要一步...
- [独家：Emmett Shear 带着新公司和大量的 Alignment 回归](https://www.corememory.com/p/exclusive-emmett-shear-is-back-with-softmax)：此处插入政变双关语
- [来自 Tibor Blaho (@btibor91) 的推文](https://x.com/btibor91/status/1906642469531595005)：由于流量过大，OpenAI 暂时禁用了新账号的 Sora 视频生成功能
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1906793591944646898)：简而言之：我们很高兴能在未来几个月发布一款强大的、具有 Reasoning 能力的新型开放权重语言模型，我们想与开发者探讨如何使其发挥最大效用：https://openai.com/op...
- [来自 Ai2 (@allen_ai) 的推文](https://x.com/allen_ai/status/1906734336537501948)：想象一下 AI 进行科学研究：阅读论文、产生创意、设计并运行实验、分析结果……我们还能揭示多少发现？🧐 认识一下 CodeScientist，这是迈向...
- [来自 MiniMax (官方) (@MiniMax__AI) 的推文](https://x.com/MiniMax__AI/status/1906722525029040560)：现在就试试 👉 https://www.minimax.io/audio/ API 访问即将推出——访问 https://www.minimax.io/platform/ 或联系 api@minimaxi.com 获取早期访问权限！
- [来自 MiniMax (官方) (@MiniMax__AI) 的推文](https://fxtwitter.com/MiniMax__AI/status/1906720764885180775)：MiniMax Audio 凭借全新的 Speech-02 模型升级了！立即将任何文件或 URL 转换为逼真的音频。单次输入最高支持 20 万字符，轻松创建有声读物和播客。...
- [来自 All Hands AI (@allhands_ai) 的推文](https://x.com/allhands_ai/status/1906760162406285442)：今天，我们很高兴发布两个重大公告！- OpenHands LM：最强的 32B 编程 Agent 模型，解决了 SWE-bench Verified 上 37.4% 的问题 📈 - OpenHands Cloud：SOTA 开源编程...
- [来自 Google Gemini App (@GeminiApp) 的推文](https://x.com/GeminiApp/status/1906206243053846558)：@dylanjturner9 嘿 @dylanjturner9，免费用户在此模型上有速率限制，这不适用于 Advanced 用户。您的订阅还可以获得更长的 Context Window。
- [来自 Google Gemini App (@GeminiApp) 的推文](https://fxtwitter.com/GeminiApp/status/1906131622736679332)：Gemini 2.5 Pro 正在起飞 🚀🚀🚀 团队正在冲刺，TPUs 正在发热，我们希望尽快将我们最智能的模型送到更多人手中。这就是为什么我们决定推出 Gemini 2...
- [来自 Sam Altman (@sama) 的推文](https://x.com/sama/status/1906771292390666325)：26 个月前 ChatGPT 的发布是我见过的最疯狂的病毒式传播时刻之一，我们在五天内增加了 100 万用户。我们在过去的一个小时里增加了 100 万用户。
- [未找到标题](https://allenai.org/papers/codescientist)：未找到描述
- [来自 wh (@nrehiew_) 的推文](https://x.com/nrehiew_/status/1905930295750107591)：我目前的工作假设是，第一张图像是经过自回归（AR）处理并通过 (vq)vae 解码的。然后它被用作起点（而不是噪声），在像素空间进行某种形式的分块扩散（block wise diffusion）。
- [来自 Yasmine (@CyouSakura) 的推文](https://fxtwitter.com/CyouSakura/status/1906737585063641532)：🥳 很高兴宣布 Open-Reasoner-Zero (ORZ) 的重大更新，这是我们在基座模型上扩展 Reinforcement Learning 的开源倡议！🌊 更新了论文并获得更优结果。使用相同的基座模型...
- [介绍 Amazon Nova Act | Amazon AGI Labs](https://labs.amazon.science/blog/nova-act)：未找到描述
- [图像和视频生成模型 – Amazon Nova 创意内容生成模型 – AWS](https://aws.amazon.com/ai/generative-ai/nova/creative/)：未找到描述
- [Elon Musk 的 xAI 在账面上以 330 亿美元收购了 Elon Musk 的 X](https://www.theverge.com/news/638933/elon-musk-x-xai-acquisition)：纸面作业。

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1355649723025063986)** (5 messages): 

> `Diffusion Models, System Card` 


- **Diffusion 细节辩论**：一名成员对 inference 来源提出质疑，指出这*显然不是标准的 Diffusion 模型*，但 Diffusion 在 sampling 过程中默认确实是从**低频到高频**进行的。
   - 他们补充道，*这一切看起来并不反 Diffusion，只是恰好效果更好*。
- **System Card 中的句子引发好奇**：针对有关 inference 来源的问题，一名成员引用了 System Card 中一个模糊的句子作为 inference 的起源。
   - 最初的问题是 *很好奇这是从哪里推断（inferred）出来的*。


  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1355274866395185182)** (33 messages🔥): 

> `GPT-4o Image Generation, Chorus Pricing, Princess Mononoke IMAX Re-Release, Advanced Voice Mode, Manufacturing CNC` 


- **GPT-4o 离奇的图像生成机制揭晓**：一位用户分享了一条 [tweet](https://x.com/xlr8harder/status/1906247140977856942)，讨论了 **GPT-4o** 图像生成的奥秘，指出最终渲染的图像被放回了模型的 context window 中。
   - 该用户质疑*为什么要将 control flow 返回给模型*，这解释了为什么它有时在生成后会回复 *"Understood"*。
- **Chorus 价格上涨**：[Chorus](https://chorus.sh/pricing) 的付费档位已上涨至 **$100/月**，提供对所有模型的访问权限，或支持自带 API keys。
   - 之前的价格是 **$20/月**，一位用户指出该价格是不可持续的，但一些用户被允许以旧价格“自动续期（grandfathered in）”，至少是暂时的。
- **Princess Mononoke 在 IMAX 斩获数百万美元**：根据一条 [tweet](https://x.com/btibor91/status/1906441722223305081)，吉卜力工作室的 **Princess Mononoke** 在 IMAX 重映大获成功，一个周末就赚了 **400 万美元**，超过了其 1999 年在北美的整个首映期票房（**240 万美元**）。
   - 有人猜测，近期通过 **ChatGPT Image Gen** 兴起的**吉卜力风格艺术**可能正在*推动新鲜的兴奋感*回流给原始创作者。
- **Voice Mode 与 GPT-4o 的契合**：根据一份[新闻稿](https://openai.com/index/hello-gpt-4o/)，高级 Voice Mode 现在使用像 **GPT-4o** 这样的原生 multimodal models，直接处理和生成音频，以实现更自然的对话。
   - 它可以捕捉非语言线索，如语速和情感，尽管用户的每日使用量有限，免费用户可以获得由 **4o-mini** 驱动的预览体验。
- **制造狂人寻求 CNC 暑期工作**：一位用户正在向随机的制造企业创始人发送电子邮件，请求在暑期操作他们的 **CNC** 机器和“杂活”。
   - 他们正试图在暑假找一份工厂车间的工作。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chorus.sh/pricing">未找到标题</a>：一款用于同时与多个 AI 聊天的 Mac 应用。</li><li><a href="https://www.twitch.tv/gemini_plays_pokemon">Twitch</a>：未找到描述</li><li><a href="https://x.com/janbamjan/status/1905754302515396701">janbam (@janbamjan) 的推文</a>：哈利路亚！Claude 在没有提示的情况下休息了</li><li><a href="https://fxtwitter.com/patience_cave/status/1905986861643993286">💺 (@patience_cave) 的推文</a>：我一直在利用 4o 生成的图像创作长篇连环画！这是对能力的重大考验。生成 50 个视觉一致的面板大约花了 10 个小时。这需要不少艺术技巧和...</li><li><a href="https://x.com/xlr8harder/status/1906247140977856942">xlr8harder (@xlr8harder) 的推文</a>：GPT-4o 图像生成的奥秘：当最终渲染的图像被放入模型的 context window 时，他们给模型发送了下面的消息。但为什么要将 control flow 返回给模型呢？(...</li><li><a href="https://x.com/btibor91/status/1906441722223305081>">Tibor Blaho (@btibor91) 的推文</a>：你知道最近吉卜力工作室的 Princess Mononoke 在 IMAX 重映几乎全线售罄吗？一个周末就赚了 400 多万美元——比它整个原始北美上映期间还要多...</li><li><a href="https://www.boxofficemojo.com/release/rl339312641/?ref_=bo_rl_tab#tabs">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1355506845992620227)** (10 messages🔥): 

> `Internal Knowledge Google Workspace 设置，AI 让艺术更易触达` 


- **OpenAI 提供 Internal Knowledge Google Workspace 设置**：成员们分享了一篇[文章](https://help.openai.com/en/articles/10929079-internal-knowledge-google-workspace-admin-managed-setup)，详细介绍了设置 **Internal Knowledge Google Workspace** 的步骤。
- **AI 让艺术更易触达**：一位成员分享了有人正在利用 **AI** 将现有作品转化为流行的 **Corporate Memphis style**，从而让艺术变得更易触达 [推文](https://x.com/xlr8harder/status/1906643226544492832)。



**提及的链接**：<a href="https://x.com/xlr8harder/status/1906643226544492832">来自 xlr8harder (@xlr8harder) 的推文</a>：很高兴宣布我的新系列，我正在使用 AI 将现有作品转化为流行的 Corporate Memphis style，让艺术变得更易触达！

  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1355842314404368518)** (3 messages): 

> `RL 教程 v2，arXiv 上的文献工具，Large Language Models` 


- **RL 教程升级**：一位成员宣布了其 **RL 教程** 的 v2 版本发布，新增了关于 **multi-agent RL** 的章节，改进了 **'RL as inference'** 和 **'RL+LLMs'** 部分，并修复了一些拼写错误（[推文链接](https://x.com/sirbayes/status/1904375008627138851)）。
- **arXiv 上的文献工具**：一位成员分享了 **arXiv 的 Bibliographic and Citation Tools** 页面链接，其中包含代码、数据、媒体、演示和相关论文等板块（[arXiv 链接](https://arxiv.org/abs/2412.05265v2)）。
- **LLMs 展示出卓越的推理能力**：分享了一个关于 **Large Language Models (LLMs)** 展示出卓越推理能力的 arXiv 论文链接。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05265v2">Reinforcement Learning: A Comprehensive Overview</a>：该手稿对 (deep) reinforcement learning 和顺序决策领域进行了宏观且前沿的综述，涵盖了基于价值的方法、策略梯度方法、基于模型的...</li><li><a href="https://x.com/sirbayes/status/1904375008627138851">来自 Kevin Patrick Murphy (@sirbayes) 的推文</a>：我很高兴地宣布我的 RL 教程 v2 版现已上线。我增加了一个关于 multi-agent RL 的新章节，并改进了 'RL as inference' 和 'RL+LLMs' 部分（尽管后者...</li><li><a href="https://www.arxiv.org/abs/2503.19470">ReSearch: Learning to Reason with Search for LLMs via Reinforcement Learning</a>：以 OpenAI-o1 和 DeepSeek-R1 的成功为代表，Large Language Models (LLMs) 展示了卓越的推理能力。然而，将推理与外部搜索过程相结合仍然...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1355649149348872212)** (12 messages🔥): 

> `Reward Models，RLHF 提示词数据，Reward Hacking` 


- **Hinton 讨厌 RLHF，称其为“猪身上涂口红”**：Geoffrey Hinton 表示 [RLHF 是一堆垃圾](https://x.com/vitrupo/status/1905858279231693144)，并将其比作给想卖掉的生锈汽车*重新喷漆*。
- **业内人士认可 Llama-3.1 Nemotron Reward Model**：[Nvidia Llama-3.1 Nemotron-70B-Reward-HF](https://huggingface.co/nvidia/Llama-3.1-Nemotron-70B-Reward-HF) 被认为是一个优秀的通用 Reward Model，而 rewardbench 则*非常过时*。
- **新型混合奖励系统应对 Reward Hacking**：一篇新[论文](https://arxiv.org/abs/2503.22230)探讨了 RLHF 性能扩展中的数据驱动瓶颈，特别是 **reward hacking** 和 **响应多样性下降** 问题，并引入了一种结合了 **Reasoning Task Verifiers (RTV)** 和 **Generative Reward Model (GenRM)** 的混合奖励系统来缓解 reward hacking。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2503.22230">Exploring Data Scaling Trends and Effects in Reinforcement Learning from Human Feedback</a>：Reinforcement Learning from Human Feedback (RLHF) 对于使大语言模型符合人类偏好至关重要。虽然最近的研究集中在算法改进上，但...的重要性...</li><li><a href="https://x.com/vitrupo/status/1905858279231693144">来自 vitrupo (@vitrupo) 的推文</a>：Geoffrey Hinton 说 RLHF 是一堆垃圾。他将其比作给想卖掉的生锈汽车重新喷漆。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1355358682745147525)** (4 条消息): 

> `Moondream 发布, 图像描述 (Image captioning), HF 仓库` 


- **Moondream 发布新版本**：最新的 [Moondream 版本](https://moondream.ai/blog/moondream-2025-03-27-release) 包含一种用于图像描述的 **Long** 格式，生成的描述长度约为 **Normal** 格式的 **2 倍**。
- **Vik 重用 HF 仓库引发不满**：一名成员表示，希望 Vik 能创建一个新的 Hugging Face 仓库，而不是重复使用同一个。
   - 他们补充说，*很好奇他是如何将检测性能做得这么好的。也许只是更了解客户的需求*。



**提到的链接**：<a href="https://moondream.ai/blog/moondream-2025-03-27-release">Moondream 2025-03-27 Release</a>：Moondream 发布公告。

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1355506038001897622)** (10 条消息🔥): 

> `Sam Altman 被解雇事件, AI 能耗, LLMs 在数学奥林匹克竞赛的表现, GPT-4o 与吉卜力工作室风格图像` 


- **Thiel 就 OpenAI 的发展方向警告 Altman**：据 [《华尔街日报》](https://www.wsj.com/tech/ai/the-real-story-behind-sam-altman-firing-from-openai-efd51a5d) 详细报道，Peter Thiel 在 2023 年 11 月洛杉矶艺术区的一次晚宴上，对 Sam Altman 提出了关于 **OpenAI** 发展路径的警告。
- **AI 聊天机器人的能耗低得令人惊讶**：一篇 [博客文章](https://engineeringprompts.substack.com/p/ai-energy-use) 将 AI 聊天机器人一年的能耗与日常活动进行了对比，结果显示其能耗低于驾驶汽车 **10 公里** 或洗 **5 次简短的热水澡**。
   - 作者提供了一个视觉辅助工具，显示一年的聊天机器人使用量所消耗的能量甚至比填满 **两个热水浴缸** 还要少。
- **LLMs 在 2025 年美国数学奥林匹克竞赛中折戟**：一条 X 帖子报告称，当前的 **SOTA LLMs** 在 **2025 年美国数学奥林匹克竞赛** 中表现不佳，在 **6 道题目** 中仅取得了 **5%** 的成功率 [ZainHasan6 推文](https://x.com/ZainHasan6/status/1906767036975301047)。
- **GPT-4o 展现吉卜力工作室风格**：在 **Sam Altman** 宣布 **GPT-4o** 的新图像模型更新后，许多网友生成了 **吉卜力工作室 (Studio Ghibli)** 风格的图像，[Technollama 博客文章](https://www.technollama.co.uk/the-style-returns-some-notes-on-chatgpt-and-studio-ghibli) 对此进行了报道。
   - Altman 本人也向开发者分享了一张修改后的吉卜力风格照片，配文为 *“感受 AGI (Feel the AGI)”*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ZainHasan6/status/1906767036975301047">Zain (@ZainHasan6) 的推文</a>：他们在 2025 年美国数学奥林匹克竞赛题目发布几小时后测试了 SOTA LLMs。测试了 6 道题，剧透一下！它们都很糟糕 -> 5%</li><li><a href="https://engineeringprompts.substack.com/p/ai-energy-use">以日常术语衡量 AI 能耗</a>：将聊天机器人的能耗与人们实际使用和理解的事物进行对比</li><li><a href="https://www.technollama.co.uk/the-style-returns-some-notes-on-chatgpt-and-studio-ghibli">风格回归：关于 ChatGPT 和吉卜力工作室的一些笔记</a>：如果你在 3 月 25 日至 3 月 26 日期间上网，你的时间线可能被大量的 AI 生成图像淹没，其中很大一部分是重新创作的现有照片……</li><li><a href="https://archive.is/xP4N1">独家 | Sam Altman 从 OpenAI 被解雇背后的真实故事 - W…</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1355956094962499787)** (20 messages🔥): 

> `Sora 的拒绝策略、C2PA 防护绕过、水印讨论` 


- **Sora 放宽对敏感图像生成的立场**：Sora 正在从敏感领域的“一刀切式拒绝”转向一种更精确的方法，重点在于防止现实世界的伤害，正如在 [Discord 帖子](https://cdn.discordapp.com/attachments/1228051082631188530/1355965151077077233/CleanShot_2025-03-30_at_19.58.51.png?ex=67ec29f4&is=67ead874&hm=d319601bffc132b42c26a426c09b70b0774ef7e27334e310e6c8297e4ca0fc4c) 中分享的那样。
   - 根据一位用户的说法，你可以使用 **Sora** 毫无问题地生成任何在世政治家的图像，但 **NSFW/NSFL** 内容会被内部搜索工具拦截。
- **C2PA 防护被轻易破解**：**Sora** 使用的 **C2PA 防护** 可以通过简单的转换文件格式或截图来绕过。
   - 一位用户指出，这种旨在确保图像真实性的保护措施不够健壮，无法防止滥用。
- **水印技术被审视为“愚蠢但聪明”**：一位成员对**水印技术**表达了悲观态度，称其为“愚蠢但聪明”。
   - 他们表示想看看“付费转化”是什么样子的，并认为练习撰写相关内容很有价值。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1355393202433032222)** (7 messages): 

> `Chris Lattner 的工作、Modular 论坛链接、Notebooks 与 Mojo` 


- **Lattner 分享已发表作品列表**：[Chris Lattner](https://nondot.org/sabre) 分享了一个指向他已发表作品列表的链接，包括 **LLVM**、**Clang**、**Swift**、**MLIR** 和 **CIRCT**，还提到了他在 **Modular AI** 的领导地位以及在 **LLVM Foundation** 的董事会席位。
- **分享了 Mojo REPL 弃用的论坛链接**：一位成员分享了一个关于 [Mojo REPL 弃用](https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna) 的 Modular 论坛讨论链接。
- **Notebooks 被提倡用于 Mojo 的打包**：一位成员提到 **Jeremy Howard** 是使用 Notebooks 的巨大支持者，不仅将其用于实验，甚至还用于 **Mojo** 的打包。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nondot.org/sabre/Resume.html#writing.">Chris Lattner 的简历</a>：未找到描述</li><li><a href="https://forum.modular.com/t/mojo-repl-deprecation/1158/4?u=melodyogonna">解决 Mojo REPL 最近的一些中断问题</a>：太棒了，谢谢 Owen！根据我的理解，我可以对此提供更多细节，但我会让其他人来做决定。我认为这与 Python 或具体用例关系较小，更多是由内部驱动的...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1355310925480984696)** (138 messages🔥🔥): 

> `AI 中的同质像性 (Homoiconicity)、Mojo 中的尾调用优化 (Tail Call Optimization)、Mojo 的 'out' 参数约定、Mojo 列表中的异构结构体、Mojo 中的饱和算术 (Saturating Arithmetic)`

- **Mojo Bug 暴露了仅推导参数（Infer-Only Parameter）的问题**：一位用户报告了一个 [bug](https://github.com/modular/max/issues/4199)，其中仅推导参数有时会被位置参数覆盖，导致在涉及 traits 和 structs 的特定场景下编译失败。
   - 该问题发生在调用带有仅推导参数的方法时，而等效的函数调用却能按预期工作；目前正在等待修复。
- **改进 Mojo 的 `out` 参数语法：可读性重构**：有人提议通过在文档和语言中将 `out` 参数的类型指定为返回类型，来 [提高 Mojo `out` 参数约定的可读性](https://github.com/modular/max/issues/4200)。
   - 讨论涉及 `out` 参数的位置（放在首位还是末尾），以及支持多个 `out` 参数的可能性，例如用于初始化具有独立读写端的通道（channels）等场景。
- **Mojo List 在使用 Traits 时出现段错误，Variant 方案来救场**：一位用户在尝试创建 trait 对象的 `List`（具体为 `List[Estimator]`）并追加 `KNN` 和 `SVM` 结构体实例时遇到了段错误（[issue #4218](https://github.com/modular/max/issues/4218)）。
   - 作为权宜之计，建议使用 `List[Variant[KNN, SVM]]` 并遍历这些值，使用 `isa` 检查类型以调用相应的方法，因为目前尚未完全支持 trait 实例。
- **`def` vs `fn`：Mojo 的大辩论**：关于 Mojo 中 `def` 与 `fn` 的用法引发了讨论，一些人认为 `fn` 应该作为默认选择，因为它具有类型安全性，并且能更好地与使用 Mypy 等工具的有类型 Python 工作流集成。
   - 另一些人则认为 `def` 对于初学者和偏好类 Python 语法的人来说仍有其地位，特别是在与无类型 Python 库交互时，这引发了一个功能请求：[让 `def` 默认返回 None](https://github.com/modular/max/issues/4211)。
- **基于 GCD 的比例计算：元编程简化分数**：一位用户展示了如何巧妙地利用编译时元编程，通过 `gcd` 自动简化 `Ratio` 结构体，从而在编译时得到简化的分数，不过也有人指出这种方法在进行元编程时可能会带来麻烦。
   - 有人提议将简化过程改为显式函数调用而非自动执行，灵感来自 C++ 中的 `std::ratio`。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/issues/4211">[Feature Request] 让 `def` 默认返回 None · Issue #4211 · modular/max</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？根据论坛上的讨论，我想请求...</li><li><a href="https://github.com/modular/modular-community">GitHub - modular/modular-community: 一个用于存放社区提交的 rattler-build 配方的仓库，旨在通过 modular-community prefix.dev 频道提供社区软件包。</a>: 一个用于存放社区提交的 rattler-build 配方的仓库，旨在通过 modular-community prefix.dev 频道提供社区软件包 - modular/modular-community</li><li><a href="https://github.com/modular/max/issues/4200#issuecomment-2763909956)">[Feature Request] 在文档生成和语言中将 `out` 参数的类型指定为返回类型 · Issue #4200 · modular/max</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？我想提出以下建议：将 out 参数视为...</li><li><a href="https://github.com/modular/max/issues/4199">[BUG] 仅推断参数（Infer-only parameters）有时会被位置参数覆盖 · Issue #4199 · modular/max</a>: Bug 描述。实际行为。考虑以下示例：trait Trait(CollectionElement): fn f(self): ... @value struct Struct(Trait): fn f(self): pass @value struct TestStruct[T: CollectionE...</li><li><a href="https://github.com/modular/max/issues/4200">[Feature Request] 在文档生成和语言中将 `out` 参数的类型指定为返回类型 · Issue #4200 · modular/max</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？我想提出以下建议：将 out 参数视为...</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/unsafe_box.mojo#L72)">samufi/larecs 中的 larecs/src/larecs/unsafe_box.mojo</a>: Larecs🌲 – 一个面向性能的基于原型的 ECS - samufi/larecs</li><li><a href="https://github.com/modular/max/issues/4218">[BUG] 在泛型集合（List[Estimator]）中使用 Trait 对象时发生段错误（Segmentation Fault） · Issue #4218 · modular/max</a>: Bug 描述。实际行为。当尝试实例化 List[Estimator] 并追加 KNN 和 SVM 实例时，Mojo 在解析语句时因段错误崩溃...</li><li><a href="https://github.com/samufi/larecs/blob/c38214e900fdf3d276cd30b41f70154ca1738653/src/larecs/scheduler.mojo#L56">samufi/larecs 中的 larecs/src/larecs/scheduler.mojo</a>: Larecs🌲 – 一个面向性能的基于原型的 ECS - samufi/larecs</li><li><a href="https://github.com/modular/max/issues/1863">[Feature Request] 移除 `Tuple[T, S]` 的特殊语法 `(T, S)` · Issue #1863 · modular/max</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？正如标题所述。你进行此更改的动机是什么？类型...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1355385004057821225)** (2 条消息): 

> `CUDA 定义, DeepSeek 绕过 CUDA, NVIDIA 驱动 vs CUDA` 


- **CUDA: 深度学习的支柱？**: 一位成员分享了[一篇博客文章](https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda)，将 **CUDA** 定义为*深度学习的支柱*和 *NVIDIA 护城河的核心*。
- **DeepSeek 通过 PTX 层绕过 CUDA**: 该成员指出，[DeepSeek 的突破](https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead)是通过**绕过 CUDA** 并直接访问 **PTX 层**实现的。
- **NVIDIA 驱动混淆**: 一位成员提到 *NVIDIA 驱动程序不被视为 CUDA*，并且 **NVIDIA** *在术语的使用上一直比较混乱且不一致*。



**提及的链接**: <a href="https://www.modular.com/blog/democratizing-compute-part-2-what-exactly-is-cuda">Modular: 民主化 AI 计算，第 2 部分：到底什么是 “CUDA”？</a>: 未找到描述

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1355931165877469417)** (15 messages🔥): 

> `视频片段, 思维导图, 多模态输出, Android 分享系统, AI 语音发音` 


- **回复中请求视频片段**：用户请求 **NotebookLM** 在以视频作为来源时，在回答中包含**视频片段**（Video Snippets）以提供**视觉效果**。
   - 一名成员建议，团队未来将启用**多模态输出**（Multi-modal Output）。
- **思维导图导出愿望清单**：一名用户询问是否能以 **DOT 格式**导出**思维导图**（Mind Maps），或发布一个带有 Google UI 的交互式小程序。
   - 暗示目前尚无法实现。
- **寻求集成 Android 分享系统**：用户请求 **NotebookLM** 加入 **Android 分享系统**，暗示需要一个专用 App。
   - 一名用户建议，从分享菜单中选择 NotebookLM 可以自动在默认笔记本中进行搜索。
- **解决 AI 发音失误**：用户正在寻求改进 **NotebookLM** 中 **AI 语音**单词发音的方法，特别是针对拼写“奇特”的公司名称。
   - 用户希望通过向 AI 提供另一个带有正确发音的来源，让**音频概览**（Audio Overview）能够正确读出公司名称。
- **AI 辩论提示词难题**：一名用户反映在提示 **NotebookLM** 生成两名主持人就“**AI 在心理健康中的应用**”持不同观点进行**激烈辩论**时遇到问题。
   - 另一名用户建议使用语音的内部名称（*Host Speaker* 和 *Expert Speaker*）来帮助分配角色。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1355256962216103936)** (96 messages🔥🔥): 

> `NotebookLM 西班牙语支持, iPhone 问题, 音频概览长度, 简报文档生成, NotebookLM Plus 每日限制` 


- **NotebookLM 仍仅支持英语**：用户询问 **NotebookLM** 是否有西班牙语版，得到的回复是目前仅支持**英语**。
   - 一名用户回复了一个[猫咪 GIF](https://tenor.com/view/tole-cat-cute-gif-12080171459357821404)，画面是一只被手握住且爪子交叉的可爱猫咪。
- **NotebookLM 遇到 iPhone 渲染问题**：一名用户报告了在 **iPhone** 上使用 **NotebookLM** 的问题，另一名用户确认它在任何使用 **WebKit** 的浏览器（如 **Mac** 上的 **Safari**）上都无法工作，直到修复程序实施前都无法解决。
   - 另一名桌面端用户也遇到了同样的问题，并报告出现了**白屏**。
- **NotebookLM Plus 用户撞上每日限制**：一名 **NotebookLM Plus** 订阅者报告看到“*已达到每日聊天限制*”的消息，导致无法正常使用，即使退出登录并刷新也无济于事。
   - 另一名用户澄清说 Plus 用户不应有任何限制问题。
- **建议增加 AI 对话功能**：一名用户建议增加 **AI 对话功能**，以便直接与 AI 互动并获取信息，而无需大量阅读，这得到了许多人的支持。
   - 成员们指出已经可以使用交互模式，但他们澄清该建议更倾向于“*聊天功能的语音版*”，即用户通过说话提问，通过聆听获取回答。
- **用户请求时间戳**：用户请求增加带时间戳的章节，以便像 **Audible** 那样跳过或重新收听特定部分。
   - 用户还要求更新到 **Gemini 2.5 Pro**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.gpt-reader.com/">GPT Reader: Free AI Text to Speech with ChatGPT Voices, TTS</a>：适用于 PDF、文章和文档的自然 ChatGPT AI 文本转语音 (TTS)。使用 GPT Reader 下载或通过高质量语音朗读。</li><li><a href="https://myaccount.google.com/age-verification">账号设置：您的浏览器不受支持。</a>：未找到描述</li><li><a href="https://tenor.com/view/tole-cat-cute-gif-12080171459357821404">Tole Cat GIF - Tole Cat Cute - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://g.co/gemini/share/97c40282beb9">‎Gemini - NotebookLM 功能与价格对比
</a>：由 Gemini Advanced 创建</li><li><a href="https://support.google.com/notebooklm/answer/15678219">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1355278078531604550)** (3 条消息): 

> `使用 LlamaIndex 构建 AI Agent 系统，LlamaIndex + Qdrant 适配 Claude，LlamaIndex 支持 OpenAI Responses API` 


- **LlamaIndex 与 SkySQL 联手打造 AI Agents**：LlamaIndex 与 SkySQL 合作，教用户如何在无需编码的情况下构建用于可靠 **text-to-SQL 转换** 的 **AI agent 系统**；更多详情请访问 [SkySQL 官网](https://t.co/Kk7yCCyAuv)。
- **LlamaIndex 通过 Qdrant 为 Claude 准备文档**：LlamaIndex 展示了如何使用为 **Qdrant** 预构建的 MCP server 准备文档并接入 **Claude**，该过程使用 **Angular 文档** 作为数据集并存储在 [Qdrant](https://t.co/uxTZe1D6gI) 中。
- **LlamaIndex 集成 OpenAI Responses API**：LlamaIndex 现在支持 **OpenAI Responses API**，全面支持内置工具、reasoning、图像、手动 tool calling、流式传输（streaming）和异步（async），从而实现 **复杂的 multi-agent 工作流**。
   - 公告指出，[Responses API](https://t.co/hJY7EOhn1Z) 与 Chat API 有很大不同。

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1355266580551831836)** (76 条消息🔥🔥): 

> `Telemetry Attributes, SubQuestionQueryEngine Workflow, VannaPack Memory Integration, Context Passing to Workflows, Image Input to OpenAIAgent` 


- **Telemetry Attributes 被标记**：成员们讨论了在与 LlamaIndex 抽象层交互时传递自定义 Telemetry Attributes 的标准方法，其中一位成员寻求将用户 ID 附加到代码块内执行的所有事件中。
   - 提供了一个利用 OpenTelemetry 和 [Colab notebook 示例](https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing) 的解决方案，用于将属性附加到 spans 和 events，并参考了 [Arize 文档](https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once) 中关于混合仪器的说明。
- **Context 处理难题**：一位用户遇到了尝试将相同的 Context 传递给两个不同 Workflow 的问题，但另一位成员澄清说，**一个 Context 包含单个 Workflow 的所有数据和状态**，并非设计用于共享。
   - 另一位成员询问了如何为 `FunctionAgent` 创建 Context，并遇到了 `AttributeError`，但通过更新 `llama-index-core` 解决了该问题。
- **OpenAI Agents 迈向多模态**：成员们讨论了将图像作为聊天消息传递给 `OpenAIAgent` 的问题，一位成员指出目前缺乏对该功能的直接支持。
   - 一位成员建议使用 [OpenAI 的多模态功能](https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees) 或修改 `chatmemorybuffer` 以将图像添加到请求中，而另一位成员则建议使用 Workflows 从头构建一个 Agent。
- **FunctionAgent 逻辑分离以提高灵活性**：关于 *为什么 `FunctionAgent` 不仅仅是一个 Workflow* 展开了讨论，对此澄清说，它 *需要特定的抽象才能成为具有特定契约的 Agent*。
   - 这种分离提供了更高的灵活性和可维护性，`AgentWorkflow` 作为编排器，而 `FunctionAgent`/`ReActAgent` 等则是可插拔的 Agent 逻辑，[文档中提供了相关示例](https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/kashh65/AutoML">AutoML - kashh65 的 Hugging Face Space</a>: 无描述</li><li><a href="https://colab.research.google.com/drive/1QV01kCEncYZ0Ym6o6reHPcffizSVxsQg?usp=sharing">Google Colab</a>: 无描述</li><li><a href="https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/hybrid-instrumentation#add-attributes-to-multiple-spans-at-once">向 Span 添加属性、元数据和标签 | Arize 文档</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-descr">使用 OpenAI GPT-4V 模型进行图像推理 - LlamaIndex</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/function_calling_agent/">Function Calling Agent 的 Workflow - LlamaIndex</a>: 无描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/multi_modal/openai_multi_modal/#ask-the-model-to-describe-what-it-sees">使用 OpenAI GPT-4V 模型进行图像推理 - LlamaIndex</a>: 无描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations%2Ftools%2Fllama-index-tools-mcp%2Fexamples%2Fmcp.ipynb">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/examples/mcp.ipynb at main · run-llama/llama_index</a>: LlamaIndex 是在您的数据上构建由 LLM 驱动的 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations%2Ftools%2Fllama-index-tools-mcp%2Fllama_index%2Ftools%2Fmcp%2Fbase.py#L57">llama_index/llama-index-integrations/tools/llama-index-tools-mcp/llama_index/tools/mcp/base.py at main · run-llama/llama_index</a>: LlamaIndex 是在您的数据上构建由 LLM 驱动的 Agent 的领先框架。 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples">示例 - LlamaIndex</a>: 无描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1356137911166435398)** (4 messages): 

> `LlamaIndex Upgrade, Internet of Agents` 


- **LlamaIndex 升级引发 Embedding 错误**：一名成员报告了在将 **LlamaIndex** 从版本 **0.8.37** 升级到 **0.9.0** 时，由于缺少 **Embedding** 设置而导致的错误。
   - 另一名成员指出，修复该问题可能需要比 **0.9.0** 更晚的版本。
- **Agent 梦想着互操作吗？**：一位成员发表了一篇文章，概述了解决 Agentic AI 中互操作性问题的可能方向，提议构建一个“**Internet of Agents**”。
   - 该文章可在 [[IoA]](https://www.anup.io/p/architecting-the-internet-of-agents) 阅读，深入探讨了用于通信、记忆、信任和工具使用的**协议层**，并建议开放标准可以解锁包括 **LlamaIndex** 在内的跨生态系统的可组合性。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1355453017230671942)** (35 messages🔥): 

> `Tinygrad Box vs Repurposed E-Waste Inference Machine, Finite Field Assembly Programming Language, TinyGrad Internals Notes, ONNX Float16 Issues, Tenstorrent DevDay` 


- **电子垃圾推理机 vs Tinygrad Box**：一位用户对一台由翻新电子垃圾制成、配备 4x 4090 的推理机的价值提出了质疑（链接见：[天猫](https://detail.tmall.com/item.htm?abbucket=18&id=887290683136)），并将其与 **Tinygrad Box** 进行了对比。
   - 另一位用户评论称，由于其自制主板，该机器很可能饱受 **PCIe 错误** 的困扰，估计其价值约为 1,000 美元加上 **4090** 的成本。
- **有限域汇编 CUDA 替代方案**：一位用户分享了 [Finite Field Assembly](https://github.com/LeetArxiv/Finite-Field-Assembly)，这是一种专为有限域计算设计的 **CUDA 替代方案**，扩展了 **C89** 并支持递归计算。
   - 它利用素数的特性来并发地进行多个数组元素的乘法运算。
- **新笔记详述 TinyGrad 内部原理**：一位用户分享了他们关于 **TinyGrad 内部原理** 的笔记，可在 [此处](https://xl0.github.io/tinygrad-notes/) 查看，内容涵盖了 **UOps**、**ShapeTracker** 和 **Pattern Matcher**，灵感来自 **mesozoic-egg**。
   - 这些笔记深入探讨了 TinyGrad 的架构，是对官方 [TinyGrad 文档](https://docs.tinygrad.org/) 的补充。
- **ONNX 静默处理 Float16 导致困扰**：一位用户报告称，**ORT CPUExecutionProvider** 会静默地将 **float16 模型** 的输入转换为 **float32**，使用 **float32** 进行计算，然后再将输出转回 **float16**，这阻碍了 **numpy 移除** 的进度。
   - 他们建议添加一个 **envvar**（环境变量），以便在他们的 **ONNX** 设置中复制这种行为，用于测试和调试。
- **Tenstorrent DevDay 演讲**：一位用户宣布他们将在旧金山的 **Tenstorrent DevDay** 上展示在 **Wormhole** 上运行的 **AlphaFold 3**，并表示有兴趣结识其他 Tinygrad 用户。
   - 他们还询问了关于多余的 **Tinygrad V1 主板** 潜在销售的问题。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://stats.tinygrad.org/">tinygrad stats</a>: 无描述</li><li><a href="https://xl0.github.io/tinygrad-notes/">My notes on TyniGrad internals – tinygrad-notes</a>: 关于 TinyGrad 内部原理的笔记</li><li><a href="https://github.com/LeetArxiv/Finite-Field-Assembly">GitHub - LeetArxiv/Finite-Field-Assembly: The Finite Field Assembly Programming Language</a>: 有限域汇编编程语言。通过创建账号为 LeetArxiv/Finite-Field-Assembly 的开发做贡献。</li><li><a href="https://detail.tmall.com/item.htm?abbucket=18&id=887290683136">商品详情</a>: 无描述
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1356130203809874000)** (17 messages🔥): 

> `使用 tinygrad 实现 VAE，Huggingface 的 Diffusers 库与 tinygrad，tg_adapter，针对 Tensors 的 torch.to 方法子类` 


- ****VAE tinygrad 化！****：一位成员一直在尝试使用 **tinygrad** 构建 **VAE**。
   - 他们成功修改了 **Huggingface 的 Diffusers 库**以适配 **tinygrad**，并让 **Stable Diffusion** 中使用的 **VAE** 运行起来，代码详见[此链接](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py)。
- ****tinygrad 适配中！****：一位成员创建了一个适配层，用于将 **torch** 调用转换为 **tinygrad** 调用。
   - 该适配层可以在[这里](https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/tg_adapter)找到，它支持将 **tinygrad** 作为无缝替换方案使用。
- ****Tensor 类型转换之争！****：一位成员提到需要为 **Tensors** 创建一个实现 **torch.to** 方法的子类。
   - 这是必要的，因为与 **tinygrad** 不同，**torch.to** 兼具类型转换（typecasting）功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/tg_adapter">tinygrad-stuff</a>：将通用的神经网络架构、特性等移植到 tinygrad</li><li><a href="https://codeberg.org/softcookiepp/tinygrad-stuff/src/branch/master/reimplementation/thf/models/autoencoders/autoencoder_kl.py">tinygrad-stuff/reimplementation/thf/models/autoencoders/autoencoder_kl.py at master</a>：未找到描述内容
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1355892644177248528)** (12 messages🔥): 

> `FP8 训练，Torchtune 办公时间 (Office Hours)` 


- **FP8 训练时间**：大多数 **FP8 训练方案** 实际上是 **FP8 QAT**，除非你只能在不支持 FP8 的 GPU（如 A100）上训练，在这种情况下你可以直接进行 FP8 训练。
   - 一位成员指出下周五将举行 **Torchtune 办公时间 (office hours)**，并提供了 [Discord 链接](https://discord.gg/Z9cuQgYX?event=1356379057373184155)。
- **Discord 时区转换**：成员们讨论了 Discord 内部针对活动进行的**自动时区转换**。
   - 一位成员分享了一个 [大脑迷因 GIF](https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-24411104)，以回应成功实现即时时区转换。



**提到的链接**：<a href="https://tenor.com/view/brain-brain-meme-big-brain-big-brain-meme-big-brain-time-gif-2441110471562975014">Brain Brain Meme GIF - Brain Brain meme Big brain - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1355554002095308801)** (4 messages): 

> `代码审查，合并流程` 


- **代码审查加速合并流程**：一位成员请求对 [PR #2441](https://github.com/pytorch/torchrec/pull/2441) 进行最终审查，以加快合并进程。
   - 另一位成员被提醒去审查该 PR。
- **PR #2441 等待最终审查**：成员们正在寻求对 [PR #2441](https://github.com/pytorch/torchrec/pull/2441) 的最终审查协助。
   - 鉴于所有检查均已通过，此举旨在加速合并过程。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 messages): 

yamashi：利用 GRPO 教会模型进行互联网搜索：https://arxiv.org/pdf/2503.09516
  

---


### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1355348512610717746)** (6 messages): 

> `Command-R 模型，Aya-Vision 模型，Playground 错误` 


- **Command-R 是高速模型**：**Command-R** 模型被确认为*最快且最通用*的模型，默认使用 **Command-A**。
   - 用户可以使用 **API** 来尝试不同的模型，因为 Playground 不支持更改模型。
- **Aya-Vision 在图片上传方面遇到困难**：用户报告在使用 **Aya-Vision** 向 Playground 上传图片时出现错误。
   - 一位用户确认该功能目前无法正常运行，并要求在情况好转时获得通知。
- **禁止发布招聘信息**：版主发布了禁止在频道内发布招聘信息的警告。
   - 这是第一次警告，暗示进一步的违规行为可能会导致更严厉的措施。


  

---

### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1355965743916908574)** (8 messages🔥): 

> `Cohere Docs Fix, API Latency, Aya Vision` 


- **Cohere 文档中的拼写错误已修复！**：一位用户报告了 [Cohere 文档](https://docs.cohere.com/v2/reference/createfinetunedmodel) 中的一个拼写错误，其中 `train_epoch=1` 应为 `train_epochs=1`，该错误会导致 `BadRequestError`。
   - 一位 Cohere 工作人员确认了该拼写错误，并提交了修复程序，*预计很快就会上线*。
- **带有图像的 API 延迟问题**：一位用户报告了使用 `chatv2` 端点时** API 性能不稳定**且响应缓慢，特别是在包含图像时，甚至在增加超时限制后仍会出现超时。
   - 他们测试了 Hugging Face 上的 [Aya Vision 演示](https://huggingface.co/spaces/CohereForAI/aya_expanse)，有时响应时间超过 30 秒，而基于非图像的端点工作速度很快。
- **调试 Aya Vision SDK**：一位用户分享了他们通过 Cohere SDK 使用 **Aya Vision** 的代码片段，请求协助调试**延迟问题**。
   - 一位 Cohere 工作人员回应称，*他们将调查其后端的延迟情况。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/CohereForAI/aya_expanse">Aya Models - a Hugging Face Space by CohereForAI</a>: 未找到描述</li><li><a href="https://docs.cohere.com/v2/reference/createfinetunedmodel">Trains and deploys a fine-tuned model. — Cohere</a>: 训练并部署微调模型。— Cohere
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1355360080664727673)** (2 messages): 

> `Indy Game Dev, C++, Graphics and Audio Libraries, Browser Game, Cohere` 


- **独立游戏开发者的远大目标**：一位主要使用 **C++** 以及图形和音频库的自学成才独立游戏开发者介绍了自己。
   - 他们目前正在为朋友的**网络动画系列**开发一款**网页游戏**，并开始使用 **Cohere** 作为其他大厂产品的替代方案。
- **新用户喜欢 Cohere**：一位游戏开发者提到他们已经开始使用 **Cohere**，并且对目前的结果非常满意。
   - 他们提到一直将其作为*大厂产品*的替代方案。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1355521797415632966)** (14 messages🔥): 

> `Libre Wolf, GPT4All model search, Documentation ingestion, Model differences, Llama3 8B Instruct` 


- **关于 Libre Wolf 浏览器的疑问**：一位成员询问了 **Libre Wolf** 浏览器的使用情况，质疑其与 **Firefox** 相比的安全性。
- **GPT4All 模型搜索困难**：一位成员提到搜索 **GPT4All 模型**列表很困难，因为它不是一个网页，对此另一位成员指出，本地模型列表搜索在 **GPT4All** 中已经有 2 年不是一项功能了。
   - 一位成员提供了 GitHub 上 [模型列表](https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata) 的链接。
- **请求文档摄取方面的帮助**：一位成员询问是否有模型能够摄取文档并根据文档回答问题，并为自己糟糕的英语道歉。
   - 一位成员分享了 [GPT4All wiki](https://github.com/nomic-ai/gpt4all/wiki)，其中包含六种语言的官方翻译，并建议其他语言使用 Google Translate。
- **寻求使用 Llama3 8B Instruct 进行卓越的博客创作**：一位成员询问 **Llama3 8B Instruct** 是否是从录制的视频课程创建博客文章和网页的最佳模型。
   - 一位成员询问 **.bin** 和 **.gguf** 文件之间的区别，以及它们是否可以互换。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/nomic-ai/gpt4all/wiki">Home</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/tree/main/gpt4all-chat/metadata">gpt4all/gpt4all-chat/metadata at main · nomic-ai/gpt4all</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1355532923779944449)** (8 messages🔥): 

> `Pydantic conint, DSPy dynamic example resending, RateLimitError with MIPROv2, Azure OpenAI burst limits, DSPy rate throttling` 


- ****Pydantic** 的 `conint` 限制**: **Pydantic** 中的 `conint` 功能可以设置约束，如 `conint(ge=1, le=10)`，但如果输出超出指定范围，它会抛出 **ValidationError**。
   - 一位成员指出，希望 DSPy 能够动态生成示例并在验证失败时重新发送请求，但该功能目前并未按预期运行。
- ****RateLimitErrors** 困扰 MIPROv2**: 有用户报告称，在 Azure OpenAI 上将 MIPROv2 与 `gpt-4o-mini` 配合使用时，尽管设置了 `num_threads=1`，仍会遇到 **RateLimitErrors**。
   - 另一位用户解释说，问题源于 **MIPROv2.compile()** 会进行多次内部 API 调用，再加上 Azure OpenAI 的突发限制（burst limits），仅靠 `num_threads=1` 无法防止该问题。
- **缓解 Azure 的速率限制 (Rate Limits)**: 为了解决 **RateLimitErrors**，一位用户建议添加带有 **sleep(30)** 间隔的重试逻辑，降低 `max_*_demos`，并考虑升级到具有内置速率限制功能的最新 DSPy 版本。
   - 有人强调，如果 LLM 由于 API 截断或速率限制而返回空输出，MIPROv2 和 Copro 中的结构化提示可能会导致错误。
- **优化过程受限于速率限制规避方案**: 一位用户指出，为了避免 **RateLimitErrors** 而减少 `max_bootstrapped_demos` 和 `max_labeled_demos` 会对优化过程产生负面影响。
   - 他们认为 DSPy 缺乏一种有效的内部延迟机制来管理 API 调用频率。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1355296267210592406)** (4 messages): 

> `DSPy Optimizers, Module Usage, Prompt Engineering in DSPy, Signature Creation in DSPy` 


- **DSPy 优化器优化提示词和权重**: 根据 [DSPy 文档](https://dspy.ai/)，DSPy 通过优化提示词和权重来教导 LM 提供高质量输出，并为**构建模块化 AI 系统**及其**提示词与权重优化**提供算法。
   - 不同的优化器会选择 N 个示例包含在提示词中。
- **DSPy 签名定义为 "a, b -> c"**: 在 DSPy 中，签名被定义为 *"a, b -> c"*，其中 a、b 和 c 是具有实际意义的名称。
   - 随后，优化器会生成提示词并在数据集上运行，以确定性能最佳的提示词。
- **实际模块使用注意事项**: 如果特定实现必须使用优化器，那么 **docstrings 的相关性就会降低**。
- **构建从 NLP 数据到图表的流水线**: 一位成员正致力于利用 DSPy 构建一个将**数据的自然语言处理转化为图表**的工具。



**提到的链接**: <a href="https://dspy.ai/">DSPy</a>: 用于对语言模型进行编程（而非提示）的框架。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1356133200380035102)** (1 messages): 

> `Thomas Hubert, AlphaProof, Formal Mathematics, Reinforcement Learning` 


- **Thomas Hubert 开展 AlphaProof 讲座**: Google DeepMind 的研究工程师 Thomas Hubert 将于 PDT 时间 3/31 上午 10 点进行题为“**AlphaProof**: 当强化学习遇见形式数学”的讲座，并在 [YouTube](https://www.youtube.com/live/3gaEMscOMAU) 上进行直播。
   - 讲座将探讨计算机和计算现在如何常规地应用于数学研究，并为解决诸如 **Birch and Swinnerton-Dyer conjecture**（贝赫和斯维讷通-戴尔猜想）等重大问题做出贡献。
- **伽利略对数学的看法**: 著名的意大利天文学家、物理学家和数学家**伽利略**曾有名言，将数学描述为*宇宙的语言*，而计算机丰富了我们的理解。
   - Hubert 在斯坦福大学获得了数学硕士学位。



**提到的链接**: <a href="https://www.youtube.com/live/3gaEMscOMAU"> - YouTube</a>: 未找到描述内容

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1355402573514801152)** (11 条消息🔥): 

> `课程信息、授课时间、免费额度、AgentX 竞赛` 


- **LLM Agents MOOC 课程信息已列出**：课程网站 ([llmagents-learning.org/sp25](https://llmagents-learning.org/sp25)) 和 [Discord server](https://discord.gg/NWVpQ9rBvd) 提供了 **LLM Agents MOOC** 的重要链接和讨论论坛。
- **Spring 2025 LLM Agents MOOC 往期讲座**：Spring 2025 课程的往期讲座可以在 [课程网站](https://llmagents-learning.org/sp25) 和此 [YouTube playlist](https://www.youtube.com/playlist?list=PLS01nW3RtgorL3AW8REU9nGkzhvtn6Egn) 中找到。
- **如何获取免费额度？**：AgentX 提供额度资源，详情请见 [AgentX website](https://rdi.berkeley.edu/agentx/)，本周将发布针对想要获取 AgentX 额度的人员的信息收集表。
- **讲座提前至 10 AM PST**：今天的讲座已移至 **10 AM PST**，以配合来自 **UK** 的演讲者。
- **基于完成情况的测验**：该课程的测验是 **基于完成情况 (completion based)** 的，只要尝试回答，分数并不重要。



**提到的链接**：<a href="https://llmagents-learning.org/sp25">Advanced Large Language Model Agents MOOC</a>：MOOC，2025 春季

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1355933844368392445)** (1 条消息): 

> `TMLS 2025, MLOps, AI, 演讲者招募 (Call for Speakers), AI Agents` 


- **TMLS 2025 演讲者招募开启**：一名成员宣布了 2025 年 6 月 **Toronto Machine Learning Summit (TMLS)** 的 [演讲者招募 (Call for Speakers)](https://www.linkedin.com/posts/toronto-machine-learning-summit_tmls2025-callforspeakers-ai-activity-7303505411800719361-z-V2?utm_source=share&utm_medium=member_ios&rcm=ACoAACF-hfwBzcfh2mYq928aQ3C0PDfox4I_I8s)。
   - TMLS 2025 将设有 **16 个专业分论坛**，包括 **Advanced RAG**、**Multimodal LLMs**、**AI Agents in Production**、**MLOps for Smaller Teams**、**Responsible AI Implementation** 以及 **GenAI Deployments**。
- **针对小型团队的 MLOps**：Toronto Machine Learning Summit 将设立一个针对小型团队的 MLOps 分论坛。
   - 这是小型团队分享经验并向他人学习的绝佳机会。


  

---


---


---


{% else %}


> 完整的各频道详细分析已在邮件中截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}