---
companies:
- anthropic
- openai
- deepmind
- apple
- zep
- perplexity-ai
- github
date: '2024-10-30T23:17:27.255253Z'
description: '**Anthropic** 发布了关于 Claude 3.5 SWEBench+SWEAgent 的详细信息，与此同时，**OpenAI**
  推出了 SimpleQA，**DeepMind** 则发布了 NotebookLM。**苹果 (Apple)** 宣布了新款 M4 MacBook，同时一个新的
  SOTA（最先进）图像模型 Recraft v3 正式问世。


  Hamel Husain 发表了一篇长达 6,000 字的详细论述，介绍了一种名为“**批判影子 (critique shadowing)**”的方法来构建 LLM
  裁判（LLM judges），旨在使大语言模型与领域专家保持一致，从而解决 AI 团队中数据不可信和未被利用的问题。该工作流程涉及专家评审的数据集和迭代式的提示词优化。


  此外，**Zep** 引入了一个时序知识图谱记忆层，以增强 AI 智能体的记忆能力并减少幻觉。**Anthropic** 还将 Claude 3.5 Sonnet
  集成到了 GitHub Copilot 中，扩大了 Copilot Chat 用户的使用范围。'
id: f2d4419f-c61f-46d6-8045-b1d6334d5125
models:
- claude-3.5-sonnet
- claude-3.5
- notebooklm
- simpleqa
- recraft-v3
original_slug: ainews-creating-a-llm-as-a-judge
people:
- hamel-husain
- swyx
title: 构建 LLM-as-a-Judge（大模型评委）
topics:
- critique-shadowing
- llm-judging
- domain-experts
- dataset-creation
- prompt-engineering
- error-analysis
- temporal-knowledge-graphs
- memory-layer
- ai-agent-memory
- hallucination-reduction
- integration
---

<!-- buttondown-editor-mode: plaintext -->**Critique Shadowing is all you need.**

> 2024年10月29日至10月30日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discords（**231** 个频道和 **2558** 条消息）。预计节省阅读时间（以 200wpm 计算）：**241 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

在 Anthropic（Claude 3.5 SWEBench+SWEAgent 详情）、OpenAI (SimpleQA)、DeepMind (NotebookLM)、Apple (M4 Macbooks) 以及一个神秘的[新 SOTA 图像模型](https://x.com/ArtificialAnlys/status/1851707166744584335) (Recraft v3) 纷纷发布新品的一天，关注一个小众名字的新闻实属罕见，但我们热爱那些实用的新闻。

继热门文章 **Your AI Product Needs Evals**（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-evals-based-ai-engineering/)）之后，Hamel Husain 带着一篇关于[创建驱动业务结果的 LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/) 的 6000 字史诗级论述回归了，文中提出了一个明确的问题陈述：**AI 团队拥有太多他们不信任且不使用的数据。**


![image.png](https://assets.buttondown.email/images/e2f7b1b6-0342-4ae1-82ed-faa407ac0c00.png?w=960&fit=max)


在 [Hamel 的 AI.Engineer 演讲](https://www.youtube.com/watch?v=eLXF0VojuSs)（以及非常有趣的 [Weights & Biases 演讲](https://www.youtube.com/watch?v=IIL2tE4n1Q0)）中呼应了许多标准主题，但这篇文章的显著之处在于它强烈推荐使用 **critique shadowing**，以此创建 few-shot 示例，使 LLM judges 与 **domain experts**（领域专家）保持一致：


![image.png](https://assets.buttondown.email/images/f3353fbf-af1b-4993-8a5d-c9c0303f96f7.png?w=960&fit=max)


**Critique Shadowing TLDR**:

<ol type="1">
<li>寻找首席领域专家 (Principal Domain Expert)</li>
<li>创建数据集
<ul>
<li>生成涵盖您用例的多样化示例</li>
<li>包含真实或合成的用户交互</li>
</ul></li>
<li>领域专家评审数据
<ul>
<li>专家进行通过/失败判定</li>
<li>专家编写详细的 critiques 以解释其推理过程</li>
</ul></li>
<li>修复错误（如果发现）
<ul>
<li>解决评审过程中发现的任何问题</li>
<li>返回专家评审以验证修复效果</li>
<li>如果发现错误，返回第 3 步</li>
</ul></li>
<li>构建 LLM Judge
<ul>
<li>使用专家示例创建 prompt</li>
<li>针对专家判定进行测试</li>
<li>优化 prompt 直到一致性达到满意水平</li>
</ul></li>
<li>执行错误分析
<ul>
<li>计算不同维度的错误率</li>
<li>识别模式和根本原因</li>
<li>修复错误，必要时返回第 3 步</li>
<li>根据需要创建专门的 judges</li>
</ul></li>
</ol>

最终的工作流如下所示：


![image.png](https://assets.buttondown.email/images/f88dde5d-654f-47e6-a904-290b85cb25c7.png?w=960&fit=max)


非常实用，正如 Hamel 在文章中提到的，这也是我们构建 AINews 时所采用的重度依赖 critique 和领域专家的迭代过程！

---

**[由 Zep 赞助]** 为什么 AI agents 到底需要一个记忆层？在 prompts 中包含完整的交互历史会导致幻觉、召回率差以及昂贵的 LLM 调用。此外，大多数 RAG 流水线在处理事实随时间变化的 temporal data 时表现挣扎。Zep 是一项新服务，它使用一种称为 **temporal knowledge graph** 的独特结构来解决这些问题。[通过快速入门指南在几分钟内启动并运行](https://shortclick.link/v157af)。

> swyx 的评论：[Zep 的 4 个 memory APIs 文档](https://help.getzep.com/memory#understanding-zeps-different-memory-apis) 也帮助我更好地理解了 Zep 的功能范围，并提供了一个更好的心理模型，让我了解一个与具体工具无关的 chatbot memory API 应该是什么样子的。值得一读！

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

**GitHub Copilot 与 AI 集成**

- **Claude 集成**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1851297754980761605) 宣布 **Claude 3.5 Sonnet 现已在 GitHub Copilot 中可用**，并将在未来几周内向所有 Copilot Chat 用户和组织开放。[@alexalbert__](https://twitter.com/alexalbert__/status/1851300048711365021) 呼应了这一公告，强调了其在 Visual Studio Code 和 GitHub 中的可用性。

- **Perplexity AI 合作伙伴关系**：[@perplexity_ai](https://twitter.com/perplexity_ai/status/1851315707411337435) 分享了与 GitHub 合作的兴奋之情，详细介绍了在 GitHub Copilot 平台内保持库更新、寻找问题答案以及获取 API 集成协助等功能。

- **多模型支持**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851312007393526094) 指出 **Gemini 1.5 Pro 也已在 GitHub Copilot 中可用**，与 Claude 3.5 Sonnet 和 OpenAI 的 o1-preview 并列。这种多模型支持代表了 GitHub Copilot 产品的一个重大转变。

- **对开发的影响**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851398044551692738) 强调了一项统计数据，“**Google 超过 25% 的新代码现在是由 AI 生成的**”，这表明 AI 对软件开发实践产生了重大影响。

**AI 进展与研究**

- **Layer Skip 技术**：[@AIatMeta](https://twitter.com/AIatMeta/status/1851327605716435011) 宣布发布 Layer Skip 的推理代码和微调 checkpoints。这是一种通过执行部分层并使用后续层进行验证和纠错来加速 LLM 的端到端解决方案。

- **小语言模型 (Small Language Models)**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851390255993417947) 分享了一篇关于 Small Language Models 的综述论文，表明研究界对更高效 AI 模型的持续关注。

- **混合专家模型 (MoE) 研究**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851329455719133572) 讨论了一篇论文，该论文揭示了在 LLM 架构中，MoE 架构以推理能力换取内存效率，更多的专家并不一定让 LLM 更聪明，而是让其更擅长记忆。

**AI 应用与工具**

- **Perplexity Sports**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1851394173821411437) 宣布推出 Perplexity Sports，首先推出用于比赛摘要、统计数据以及球员/球队对比的 NFL 小组件，并计划扩展到其他体育项目。

- **AI 在媒体制作中的应用**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1851242004602114496) 分享了关于 Runway 对 AI 在媒体和娱乐领域愿景的长推文，将 AI 描述为讲故事的工具，并预测将向交互式、生成式和个性化内容转变。

- **开源进展**：[@AIatMeta](https://twitter.com/AIatMeta/status/1851327605716435011) 在 Hugging Face 上发布了 Layer Skip 的推理代码和微调 checkpoints，这是一种 LLM 的加速技术。

**编程语言与工具**

- **Python 的流行度**：[@svpino](https://twitter.com/svpino/status/1851368175192916459) 指出 **Python 现在是 GitHub 上排名第一的编程语言**，超越了 JavaScript。

- **GitHub 统计数据**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1851318725246644625) 分享了 Octoverse 2024 报告的见解，包括 **AI 项目同比增长 98%**，以及 Jupyter Notebook 使用量激增 92%。

**迷因与幽默**

- [@willdepue](https://twitter.com/willdepue/status/1851373942109520040) 开玩笑说 AGI 已经在内部实现了，称“证明 AGI 已在内部实现。我们做到了，Joe”。

- [@Teknium1](https://twitter.com/Teknium1/status/1851383590556483591) 调侃道“日本 AI 公司太残暴了，哈哈”，这可能是指日本某些幽默或激进的 AI 相关进展。

- [@nearcyan](https://twitter.com/nearcyan/status/1851343399863087231) 讲了一个关于投资 NVIDIA 的“80 IQ 骚操作”笑话，因为 ChatGPT 的流行，回想起来这样的投资会产生巨大的回报。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Apple 的 M4 Mac Mini：AI 开发的新竞争者**

- **[Mac Mini 现在看起来很有吸引力……比 5090 更便宜，且拥有近两倍的 VRAM……](https://i.redd.it/juob11y8lqxd1.png)** ([Score: 49, Comments: 18](https://reddit.com//r/LocalLLaMA/comments/1gf1dhf/mac_mini_looks_compelling_now_cheaper_than_a_5090/)): 该帖子指出，与假设的 **5090** 等高端 GPU 相比，搭载 **M4 芯片** 的 **Mac Mini** 对于 **AI 工作负载** 可能是一个更具吸引力的选择。作者强调，Mac Mini 可能更**便宜**，并且提供**近两倍的 VRAM**，使其成为需要大容量内存的 AI 任务的极佳选择。

- **[新款 M4 / Pro Mac Mini 讨论](https://www.apple.com/shop/buy-mac/mac-mini/m4)** ([Score: 40, Comments: 58](https://reddit.com//r/LocalLLaMA/comments/1gezl2e/new_m4_pro_mac_minis_discuss/)): 该帖子讨论了关于针对 **AI 任务** 优化的 **M4 / Pro Mac Mini** 机型的推测。虽然没有提供具体的规格或价格信息，但标题表明了人们对未来 Mac Mini 迭代版本在人工智能应用方面的能力和成本的关注。
  - 关于 **M4 Mac Mini 价格** 的推测：配备 **32GB RAM** 的基础型号估计为 **$1000**，而 **64GB** 版本为 **$2000**。一位用户声称，使用教育优惠，**16GB** 的基础型号可能仅需 **$499**。
  - 关于**内存带宽**和性能的讨论：据估计 M4 拥有 **260 GB/s** 的带宽，在使用 **Qwen 72B 4-bit MLX** 时可能达到 **6-7 tokens/s**。一些用户讨论了在 AI 任务中 Mac Mini 与 **3090s** 等 GPU 之间的权衡。
  - 与 **Nvidia GPU** 的比较：用户讨论了高 RAM 的 Mac Mini 如何与 **4090** 等昂贵 GPU 竞争。然而，也有人指出，虽然 Mac 提供更多 RAM，但 GPU 在 AI 任务的处理速度上仍然明显更快。


**主题 2. Stable Diffusion 3.5 Medium 在 Hugging Face 发布**

- **[Stable Diffusion 3.5 Medium · Hugging Face](https://huggingface.co/stabilityai/stable-diffusion-3.5-medium)** ([Score: 68, Comments: 31](https://reddit.com//r/LocalLLaMA/comments/1gew4mp/stable_diffusion_35_medium_hugging_face/)): **Stable Diffusion 3.5 Medium**，一款新的文本生成图像模型，已在 Hugging Face 上发布。该模型在**文本渲染**、**多主体生成**和**构图理解**方面具有更强的能力，同时与之前的版本相比，图像质量有所提高，伪影有所减少。该模型可根据 **OpenRAIL-M 许可证** 用于商业用途，默认分辨率为 **768x768**，并支持包括 **txt2img**、**img2img** 和 **inpainting** 在内的各种推理方法。
  - 用户询问了自托管该模型的**硬件要求**。根据博客，它需要 **10GB** 的 VRAM，对于“32GB 或更大”的配置，建议使用从 **3090 到 H100** 的 GPU。
  - 出现了关于像 LLM 一样运行**更小量化**版本的可能性的讨论。用户推测社区可能会尝试这样做。
  - 当被问及与 **Flux Dev** 的比较时，一位用户简单地回答“很糟糕”，暗示 Stable Diffusion 3.5 Medium 在某些方面的表现可能不如 Flux Dev。


**主题 3. AI 安全与对齐：辩论与批评**

- **[MacOS 15.1 中的 Apple Intelligence 提示词模板](https://www.reddit.com/gallery/1gepb6t)** ([Score: 293, Comments: 67](https://reddit.com//r/LocalLLaMA/comments/1gepb6t/apple_intelligences_prompt_templates_in_macos_151/)): Apple 的 **MacOS 15.1** 引入了 **AI prompt templates** 和安全措施，作为其 **Apple Intelligence** 功能的一部分。该系统包含用于各种任务（如总结文本、解释概念和生成创意）的 **built-in prompts**，重点在于维护用户隐私和数据安全。Apple 的方法强调 **on-device processing**，并结合了 **content filtering** 以防止生成有害或不当内容。
  - 用户幽默地批评了 Apple 的 **prompt engineering**，开玩笑说它在“乞求”正确的 **JSON output**，并辩论了 **YAML vs. JSON** 的效率。讨论强调了 **minifying JSON** 对节省 token 的重要性。
  - 社区对 Apple 防止 **hallucinations** 和 **factual inaccuracies** 的方法表示怀疑，一位用户分享了一个包含 Apple 资源文件夹中 **metadata.json files** 的 [GitHub gist](https://gist.github.com/dvessel/40a0fae364a3648ac342322aaa758bf4)。
  - 讨论涉及了可能使用的 **30 billion parameter model** (v5.0-30b)，并批评了在活动选项中包含 **diving 和 hiking** 等特定运动的做法，推测这可能受到了管理层的影响。
- **“AI Safety”的危险风险** ([Score: 47, Comments: 62](https://reddit.com//r/LocalLLaMA/comments/1ger1xg/the_dangerous_risks_of_ai_safety/)): 该帖子讨论了 **AI alignment** 工作的潜在风险，链接到一篇文章，该文章认为 **alignment technology** 可能会被滥用，服务于恶意利益而非全人类。作者认为这种情况已经在发生，并指出目前的 **API-based AI** 系统通常执行比西方民主法律更严格的规则，在某些领域有时甚至与 **Taliban** 等极端主义意识形态更加一致，而 **local models** 受影响较小但仍存在问题。
  - 用户注意到 **AI 不一致的内容限制**，一些人指出 **API-based AI** 经常禁止在黄金时段电视节目中随处可见的内容。尽管 AI 公司持有反 NSFW 立场，但对 **NSFW content** 的需求依然巨大。
  - 评论者讨论了 **AI alignment** 被用作 **censorship** 和 **control** 工具的可能性。一些人认为这已经在发生，AI 在意识形态上与其创造者保持一致，并被用来对用户施加权力。
  - 几位用户对 **corporate anxiety** 和 **sensitivity** 驱动 AI 限制表示担忧，这可能导致对言论自由的压制。一些人主张 **widespread AI access**，以平衡公民与政府/企业之间的权力。

## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 研究与技术**

- **LLM 学习概念中的几何结构**：一篇 [在 Twitter 上分享的论文](https://www.reddit.com/r/singularity/comments/1gf8dou/ai_paper_reveals_surprising_geometric_structure/) 揭示了 LLM 学习到的概念中存在令人惊讶的几何结构，包括类脑的 "lobes" 和精确的 "semantic crystals"。

- **Google 的 AI 生成代码**：Google CEO Sundar Pichai 表示，[Google 超过 25% 的新代码现在由 AI 生成](https://www.reddit.com/r/OpenAI/comments/1gfcfbd/google_ceo_says_more_than_a_quarter_of_the/)。一些评论者推测，这可能包括自动补全建议和其他辅助工具。

- **ARC-AGI 基准测试进展**：MindsAI 在 [ARC-AGI 基准测试中取得了 54.5% 的新高分](https://www.reddit.com/r/singularity/comments/1gexvmj/new_arcagi_high_score_by_mindsai_545_prize_goal/)，高于 6 天前的 53%。该奖项的目标是 85%。

**AI 应用与影响**

- **AI 在教育领域**：一项 [研究发现，使用 AI 导师的学生学习到的内容是传统课堂教学的两倍多，且用时更短](https://www.reddit.com/r/singularity/comments/1geyshu/new_article_says_ai_teachers_are_better_than/)。一些评论者指出，AI 可以提供更加个性化、互动式的学习。

- **数字水果复制**：一颗 [李子成为首个在没有人工干预的情况下被完全数字化并重新打印的水果](https://www.reddit.com/r/singularity/comments/1gf6yum/a_fresh_summer_plum_is_the_first_fruit_and_scent/)，且带有其气味。

- **AI 在软件开发中**：从今天开始，开发者可以在 [Visual Studio Code 和 GitHub Copilot 中选择 Claude 3.5 Sonnet](https://www.reddit.com/r/singularity/comments/1gezfd1/starting_today_developers_can_select_claude_35/)。Gemini 也正式加入 GitHub Copilot。

**AI 模型发布与改进**

- **Stable Diffusion 3.5 的改进**：一个 [结合了 SD 3.5 Large、Medium 和 upscaling 技术的流水线](https://www.reddit.com/r/StableDiffusion/comments/1gfdqwq/sd_35_large_medium_upscale_with_attention_shift/) 产生了高质量的图像结果，展示了图像生成能力的进步。

**AI 行业与商业**

- **OpenAI 收入来源**：OpenAI 的 CFO 报告称，[公司 75% 的收入来自付费消费者](https://www.reddit.com/r/OpenAI/comments/1gepbqg/openai_cfo_says_75_of_its_revenue_comes_from/)，而非企业客户。这引发了关于 OpenAI 商业模式和盈利时间表的讨论。

**AI 伦理与社会影响**

- **Linus Torvalds 谈 AI 炒作**：Linux 创始人 Linus Torvalds 表示，[AI 是“90% 的营销和 10% 的现实”](https://www.reddit.com/r/singularity/comments/1gfg7x9/linus_torvalds_reckons_ai_is_90_marketing_and_10/)。这引发了关于 AI 技术现状和未来潜力的辩论。

**迷因与幽默**

- 一张描绘使用动漫风格角色 [为“AI 战争”做准备](https://www.reddit.com/r/StableDiffusion/comments/1gev8xt/im_ready_to_serve_in_the_coming_ai_wars/) 的幽默图片，引发了关于 AI 潜在军事化及其文化影响的讨论。


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的总结

**主题 1：Apple M4 芯片极大提升 AI 性能**

- [**LM Studio 在 Apple 新发布的 M4 MacBook Pro 上大放异彩**](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/): 在最近的 Apple 发布会上，**LM Studio** 展示了其在搭载 **M4 芯片**的新款 **MacBook Pro** 上的能力，突显了其对 AI 应用的影响。
- **传闻称 M4 Ultra 旨在超越 NVIDIA 4090 GPU**：即将推出的 **M4 Ultra** 据传将支持 **256GB** 统一内存，性能可能超越 **M2 Ultra** 并与高端 GPU 匹敌。
- **M3 Max 以每秒 60 Token 的速度给工程师留下深刻印象**：据报道，**M3 Max** 芯片运行 **Phi 3.5 MoE** 等模型的速度约为 **60 tokens per second**，展示了其即使在低配配置下的高效性。

**主题 2：AI 模型更新与争议引发社区热议**

- **Haiku 3.5 发布在即，AI 爱好者兴奋不已**：社区热切期待 **Haiku 3.5** 的发布，有迹象表明它可能很快面世，引发了对其潜在改进的好奇。
- **Gemini 甩开竞争对手，程序员欢呼雀跃**：用户称赞 **Gemini** 在处理数据库编程任务方面的卓越表现，在实际应用中超越了 **Claude** 和 **Aider** 等模型。
- **用户嘲讽微软过度谨慎的 Phi-3.5 模型**：**Phi-3.5** 过度的审查导致了幽默的嘲讽，用户分享了一些讽刺性的回复，突显了该模型拒绝回答简单问题的倾向。

**主题 3：微调与训练障碍挑战 AI 开发者**

- [**Unsloth 团队发现梯度缺陷，动摇训练基础**](https://unsloth.ai/blog/gradient): **Unsloth 团队**揭示了训练框架中**梯度累积 (gradient accumulation)** 的关键问题，这影响了语言模型的一致性。
- **LoRA 微调在 H100 GPU 上碰壁，工程师感到沮丧**：用户在 **H100 GPU** 上进行 **LoRA** 微调时遇到困难，并指出由于 **BitsAndBytes** 问题尚未解决，**QLoRA** 可能是唯一的权宜之计。
- **量化失误导致输出变成乱码**：在 **Llama 3.2 1B QLoRA** 训练期间，用户在应用 **Int8DynActInt4WeightQuantizer** 时遇到了不连贯的输出，突显了量化过程中的挑战。

**主题 4：AI 冲击软件工程，自动化工具蓬勃发展**

- **AI 蚕食软件工程师岗位，开发者感到恐慌**：成员们注意到 **AI** 正在越来越多地接管常规软件工程任务，引发了关于科技就业前景的辩论。
- [**Skyvern 实现浏览器自动化，手动任务迎来对手**](https://www.skyvern.com/): **Skyvern** 推出了一种用于浏览器自动化的无代码解决方案，使用户无需编写代码即可简化工作流程。
- [**ThunderKittens 发布新功能，自带幽默感**](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2): **ThunderKittens 团队**发布了诸如 **exciting kernels** 和语音模型等新功能，并穿插了对可爱小猫的俏皮提及。

**主题 5：OpenAI 应对事实性问题并提升用户体验**

- [**OpenAI 通过新的 SimpleQA 基准测试对抗幻觉**](https://x.com/openai/status/1851680760539025639?s=46): 通过引入 **SimpleQA**，OpenAI 旨在通过 **4,000 个多样化问题**来衡量语言模型的事实准确性，针对的是幻觉问题。
- **ChatGPT 终于支持搜索聊天记录，用户欢呼雀跃**：OpenAI 在 ChatGPT 网页版上推出了**搜索聊天记录**的功能，使用户更容易参考或继续之前的对话。
- **AGI 辩论升温，乐观派与怀疑派交锋**：成员们对实现 **AGI** 的时间表和可行性表达了不同看法，辩论 Google 等公司在面临挑战时能否跟上步伐。

---

# 第一部分：Discord 高层摘要

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **苹果 MacBook Pro 展示 LM Studio**：在最近的苹果发布会上，LM Studio 展示了其在搭载 **M4** 系列芯片的新款 MacBook Pro 上的能力，这标志着其在商业应用中的影响力得到了显著认可。
  
  - 成员们为开发者感到兴奋，并指出这种认可可能会影响未来 AI 工作流的集成。
- **M3 Max 的 Token 速度令人印象深刻**：据报道，**M3 Max** 运行 **Phi 3.5 MoE** 等模型的速度约为 **每秒 60 个 token**，突显了其即使在较低配置下的效率。
  
  - 虽然这令人印象深刻，但一些用户建议，为了追求极致速度，像 **A6000** 这样的专用 GPU 可能会产生更好的结果。
- **H100 GPU 租赁变得经济实惠**：用户提到 **H100** 租赁现在的价格约为 **每小时 1 美元**，使其成为模型推理的一个具有成本效益的选择。
  
  - 尽管价格下降，但关于在各种任务中使用高性能 GPU 与本地模型的实用性讨论也随之出现。
- **M4 Ultra 传闻规格令竞争对手生畏**：传闻即将推出的 **M4 Ultra** 将支持 **256GB** 的统一内存，预计性能将显著超越 **M2 Ultra**。
  
  - 关于 **M4** 与 **4090 GPU** 竞争的猜测层出不穷，用户们对增强的性能指标议论纷纷。
- **Windows vs. Linux 性能之争**：对 **Windows** 的不满情绪浮出水面，强调了其在 AI 任务中相比 **Linux** 的局限性，后者提供了更高的效率和控制力。
  
  - 成员们一致认为 Linux 可以更好地优化 GPU 利用率，尤其是在运行计算密集型应用程序时。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Hugging Face API 支持获取 token 概率**：一位用户确认，可以通过 **Hugging Face** 无服务器推理 API 获取大型语言模型的 token 概率，特别是使用推理客户端时。
  
  - 讨论还涉及了 **rate limits** 和 API 使用情况，并在 [Rate Limits](https://huggingface.co/docs/api-inference/en/rate-limits) 的详细链接中进行了进一步阐述。
- **Ollama 在图像分析中提供隐私保护**：针对 **Ollama** 在图像分析期间访问本地文件的担忧得到了回应，强调其在本地运行，无需服务器交互。
  
  - 这确保了用户在有效分析图像时的隐私。
- **在机器学习中选择正确的道路**：一位参与者强调选择一个涵盖广泛 **data science** 知识的专业，而不仅仅是 AI，并反思了数学和编程技能的重要性。
  
  - 进一步的讨论集中在该领域职业生涯所需的基础方面。
- **Qwen 2 模型遭遇错误的 token 生成问题**：有报告称 **Qwen 2** 基座模型存在问题，特别是由于 **EOS token** 识别错误导致输出末尾出现意外 token。
  
  - 这反映了对模型上下文长度处理能力的更广泛担忧。
- **Langchain SQL agent 在使用 GPT-4 时遇到困难**：从 **GPT-3.5 Turbo** 切换到 **GPT-4** 配合 **Langchain SQL agent** 使用时结果参差不齐，后者表现出一定的困难。
  
  - 对 API 停用的担忧引发了关于替代环境的讨论。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 团队揭示梯度问题**：[Unsloth 团队](https://unsloth.ai/blog/gradient)发布了关于训练框架中梯度累积（Gradient Accumulation）问题的研究结果，该问题会影响语言模型输出的一致性。
  
  - 报告指出，由于对 Loss 计算有显著影响，建议寻找传统 Batch Size 的替代方案。
- **苹果发布紧凑型新款 Mac Mini**：苹果宣布推出新款 [Mac mini](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/)，搭载 M4 和 M4 Pro 芯片，宣称 CPU 性能提升了惊人的 **1.8 倍**。
  
  - 此次发布标志着苹果首款碳中和 Mac 的诞生，是其产品线中的一个重要里程碑。
- **ThunderKittens 带来新功能**：ThunderKittens 团队发表了一篇 [博客文章](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2) 揭晓新功能，重点介绍了 **令人兴奋的 Kernel** 和对话模型。
  
  - 他们还俏皮地提到了社交媒体的反应，并展示了 *超级可爱的猫咪* 来增强社区互动。
- **Instruct 微调挑战**：一位用户在尝试微调 **Meta Llama3.1 8B Instruct** 模型时遇到了 Tensor 形状不匹配的错误。
  
  - 随着用户切换模型但仍受困于合并与加载问题，挫败感倍增，这凸显了兼容性方面的疑虑。
- **Unsloth 在 VRAM 效率方面的努力**：Unsloth 宣布了一种预训练方法，可实现 **2 倍速** 训练并减少 **50% 的 VRAM** 消耗，同时为 Mistral v0.3 7b 提供了一个 [免费 Colab Notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)。
  
  - 建议用户在微调 Embedding 和调整学习率（Learning Rates）时保持谨慎，以稳定训练过程。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **对 AGI 竞赛看法不一**：成员们对实现 **AGI** 的时间表和可行性表达了不同看法，特别是关于谷歌面临监管挑战阻碍其进展的问题。
  
  - 对谷歌困境的 *担忧* 与对新兴算法推动进步的 **乐观** 情绪形成了鲜明对比。
- **模型效率辩论持续升温**：社区讨论了 **大模型** 并不总是更优，指出 **量化（Quantization）** 在不牺牲性能的情况下实现效率的作用，并提到了 **Llama 3.0** 和 **Qwen** 模型。
  
  - 最近的 **量化模型** 被引用为表现优于其更大的对应版本，强调了关注点正转向高效的模型利用。
- **Nvidia GPU 是否足够引发争议**：辩论集中在 **4070 Super GPU** 对于本地 AI 项目是否充足，提醒人们对于高需求应用需要更高 VRAM 的选项。
  
  - 参与者既承认了小模型的性能，也指出了价格亲民的高性能 GPU 在供应上的 **缺口**。
- **Prompt 生成工具需求旺盛**：用户寻求在 OpenAI Playground 中使用 **Prompt 生成工具** 以更好地定制请求，并参考了 [官方 Prompt 生成指南](https://platform.openai.com/docs/guides/prompt-generation)。
  
  - 讨论达成共识，认为该工具对于优化 Prompt 策略至关重要。
- **有序数据对聊天机器人至关重要**：在开发个人聊天机器人时，强调了保持数据有序和简洁的重要性，以避免额外的 API 调用费用，因为无关数据的 **Input Token** 仍会产生费用。
  
  - 一位成员指出，妥善的数据管理不仅是最佳实践，更是 API 使用中关键的财务考量。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Supply 推出新款 Essentials 系列**：[Perplexity Supply](https://perplexity.supply) 推出了一系列为好奇心设计的贴心必需品，让客户通过产品激发对话。
  
  - 全球发货现已覆盖**美国**、**澳大利亚**和**德国**等国家，更新信息可通过[此链接](https://perplexity.supply/sign-up)获取。
- **文件上传问题令用户感到沮丧**：几位用户报告了**文件上传功能**的问题，强调了讨论过程中文件残留的问题。
  
  - *一位用户指出*，与其他平台相比，其文件处理能力较差。
- **探索 Playground 与 API 结果的差异**：一位用户对 [Playground](https://labs.perplexity.ai/) 和 API 之间观察到的结果差异表示担忧，尽管两者使用的是相同的模型。
  
  - 目前尚未对这些不一致背后的原因提供进一步说明。
- **地球的临时新卫星引发讨论**：最近的一项讨论重点介绍了地球的**临时新卫星**，详细说明了其可见性和影响，[点击此处查看详情](https://www.perplexity.ai/page/earth-s-temporary-new-moon-1a.EqH6ARBuNGyHUoOv37A)。
  
  - 这一迷人的发现引发了关于临时天体现象的动态对话。
- **澄清 Perplexity Spaces 的 API 使用**：已澄清目前没有可用于 **Perplexity Spaces** 的 API，网站和 API 作为独立实体运行。
  
  - 一位用户表达了在开发项目中使用 **Perplexity API** 的兴趣，但未收到具体指导。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 通过命令增强文件管理**：Aider 引入了 `/save <fname>` 和 `/load <fname>` 命令以便于上下文管理，简化了批处理和文件处理。
  
  - 该功能消除了手动重建代码上下文的麻烦，使工作流更加高效。
- **备受期待的 Haiku 3.5 发布引发关注**：持续的讨论表明 **Haiku 3.5** 可能很快发布，最早可能就在明天。
  
  - 用户渴望了解其相对于先前版本的增强功能，并希望有显著改进。
- **Qodo AI 与 Cline 引发对比辩论**：关于 **Qodo AI** 如何在可用性和功能方面与 **Cline** 等竞争对手区分开来的讨论浮出水面。
  
  - 尽管起步订阅费用为 **$19/月**，但对功能有限的担忧削弱了人们对 Qodo 市场地位的热情。
- **Skyvern 利用 AI 自动化浏览器任务**：Skyvern 旨在作为一种无代码解决方案简化浏览器自动化，为重复性工作流提供效率。
  
  - 它在网页间的适应性允许用户通过简单的命令执行复杂任务。
- **用户评价 Gemini 的编程效率**：反馈强调了 **Gemini** 在处理数据库相关编程任务方面优于 **Claude** 和 **Aider**。
  
  - 共识显示了 Gemini 在实际编程需求方面的优势，但性能可能会根据上下文而波动。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Oauth 身份验证故障，修复即将发布**：由于 **Oauth 问题**，使用 [openrouter.ai/auth](https://openrouter.ai/auth) 的应用今早面临故障，但在公告发布后不久预计将会有修复方案。
  - 团队确认 **API key 创建** 的停机时间将非常短，以此安抚受影响的用户。
- **macOS 聊天应用招募 Alpha 测试人员**：一位开发者正在为一款灵活的 **macOS** 聊天应用寻求 **alpha 测试人员**，并提供了 [截图](https://imgur.com/a/HI5Py3A) 供查看。
  - 鼓励感兴趣的人员通过 **DM** 获取更多信息，强调了测试期间用户反馈的重要性。
- **围绕 OpenRouter API Key 的安全担忧**：用户对 **OpenRouter API keys** 的脆弱性表示担忧，特别是关于在 **Sonnet 3.5** 等代理设置中的滥用问题。
  - 一位社区成员警告说：*“仅仅因为你认为 key 是安全的，并不意味着它真的安全，”* 强调了 key 管理的重要性。
- **热切期待 Haiku 3.5 发布**：社区对预期发布的 **Haiku 3.5** 议论纷纷，分享的模型标识符 (slug) 为 `claude-3-5-haiku@20241022`。
  - 尽管该模型已在白名单中但尚未正式开放，但迹象表明可能会在一天内发布。
- **请求访问集成功能**：用户纷纷要求访问 **integration feature**（集成功能），强调其对于测试各种能力的重要性。
  - 诸如 *“我想再次申请集成功能！”* 之类的回应表明了对该功能的强烈需求。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU 价格辩论升温**：成员们分析认为，二手 **3090** 显卡的价格低于 **7900 XTX** 型号，强调了预算与性能之间的权衡。
  - *eBay 价格徘徊在 ~$690 左右*，这让注重成本的工程师在选择 GPU 时面临艰难抉择。
- **自定义风格训练**：一位成员询问关于使用 **15-20 张图像** 训练朋友的艺术风格，在选择模型还是 Lora/ti 之间进行讨论。
  - 其他人建议使用 **Lora**，以便根据特定的风格偏好获得更好的角色一致性。
- **Stable Diffusion 中的灰色图像问题**：多位用户报告在 **Stable Diffusion** 中遇到 **灰色图像**，并寻求故障排除建议。
  - 成员们建议尝试不同的 UI 选项，并检查与 **AMD GPUs** 的兼容性以改善输出。
- **UI 大对决：Auto1111 对比 Comfy UI**：**Comfy UI** 因其用户友好性成为热门选择，而一些人仍偏好使用 **Auto1111** 进行自动化。
  - 建议还包括尝试 **SwarmUI**，因为它安装简单且功能齐全。
- **关于即将发布的 AI 模型的传闻**：社区推测 **SD 3.5** 相比 **SDXL** 的潜在普及程度，引发了关于性能的讨论。
  - 对新的 ControlNet 和模型更新的期待与日俱增，这对于跟上 AI 发展步伐至关重要。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **微软降低对 OpenAI 依赖的风险**：讨论涉及微软降低对 OpenAI 依赖的策略，特别是如果 OpenAI 宣布实现 AGI，这可能会为微软提供合同上的退出机制和重新谈判的机会。
  - *“微软永远不会让这种事发生，”* 表达了对 AGI 可能发布的怀疑。
- **AI 延迟问题引发关注**：有报告称某 AI 模型出现了 **20 秒的延迟**，成员们幽默地暗示它可能是 *在土豆上运行的*。
  - 成员将其与 Lambda 的性能进行了对比，后者处理 *10 倍以上的请求，延迟仅为 1 秒*。
- **Hermes 3 性能令用户惊喜**：成员们讨论到 **Hermes 3 8B** 的质量出人意料地可与 **GPT-3.5** 媲美，表现优于其他 10B 以下的模型。
  - 相比之下，**Mistral 7B** 等模型被评价为 *令人遗憾*。
- **寻求西班牙语 Function Calling 数据集**：一位成员寻求构建 **西班牙语的 function calling 数据集**，但在使用开源模型处理来自 **López Obrador** 会议的数据时面临效果不佳的挑战。
  - 他们的目标是处理来自 **一千多个视频** 的信息，旨在实现 *新闻相关性*。
- **Sundar Pichai 强调 AI 在谷歌的作用**：在财报电话会议上，**Sundar Pichai** 表示谷歌超过 **25% 的新代码** 是由 AI 生成的，引发了关于 AI 对编程影响的讨论。
  - 这一被广泛分享的统计数据引发了关于编程实践演变的对话。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **在单张 GPU 上运行多个实例**：成员们讨论了在同一张 GPU 上运行多个 **GPT-NeoX** 实例，旨在通过更大的 Batch Size 最大化显存利用率，尽管 DDP 的收益可能有限。
  
  - 正在进行的对话强调了并行训练的潜在配置和注意事项。
- **使用 CSV 数据的 RAG**：一位成员询问了对 *~3B LLM* 使用原始 CSV 数据进行 RAG 的效果，并表示在遇到案例编号不一致的挑战后，计划将其转换为 JSON。
  
  - 这一举动暗示了预处理的复杂性可能会影响 RAG 的性能。
- **实体提取的 Temperature 调优**：在意识到 **Entity Extraction** 过程中的 Temperature 设置不正确后，一位成员使用修正后的参数重新尝试，以优化结果。
  
  - 这突显了调优模型参数对于获得有效性能的重要性。
- **LLM 中的模块化对偶性与优化**：最近的一篇论文揭示了 **maximal update parameterization** 和 **Shampoo** 等方法可以作为线性层单一对偶映射（Duality Map）的部分近似。
  
  - 这种联系强化了论文中讨论的当代优化技术的理论基础。
- **Diffusion Models 的挑战**：讨论中提到了 **Diffusion Models** 与 **GANs** 及 **autoregressive models** 相比所呈现的独特局限性，特别是在训练和质量指标方面。
  
  - 成员们指出了围绕可控性和表示学习（Representation Learning）的问题，强调了它们对模型适用性的影响。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Elon Musk 洽谈提升 xAI 估值**：据 [WSJ](https://x.com/AndrewCurran_/status/1851310076709224564) 报道，Elon 正在洽谈新一轮融资，旨在将 **xAI** 的估值从 **240 亿美元**提升至 **400 亿美元**。尽管正在讨论，Elon 仍不断否认之前的融资传闻，导致社区对 xAI 的发展方向感到不安。
  
  - 一位成员表示：“*xAI 有点让我害怕*”，反映了社区内更广泛的担忧。
- **揭秘 Claude 3 Tokenizer**：最近的一篇 [文章](https://tokencontributions.substack.com/p/the-mystery-of-the-claude-3-tokenizer) 强调了 **Claude 3 Tokenizer** 的封闭性质，透露可获取的信息非常有限。用户不得不依赖付费服务而非公开文档，这引发了挫败感。
  
  - 该文章强调了开发者在有效利用 Claude 3 时面临的重大障碍。
- **AI2 将搬迁至水边新办公室**：AI2 将于明年 6 月搬迁至新办公室，届时可欣赏 **Pacific Northwest** 的美景。成员们对搬迁表示兴奋，认为宜人的风景是一项福利。
  
  - 这一变动有望为 AI2 团队营造一个更具启发性的工作环境。
- **MacBook Pro 惊人的价格**：成员们对高配 **MacBook Pro** 的昂贵价格做出反应，**128GB RAM + 4TB SSD** 的配置售价约为 **8000 欧元**。讨论突显了对不同地区定价差异的困惑。
  
  - 评论反映了汇率波动和税收如何使寻求尖端硬件的工程师的购买过程变得复杂。
- **Voiceover 增强个人文章体验**：一位成员提倡将 **Voiceover** 作为明天 **Personal Article** 更具吸引力的媒介。他们对 Voiceover 内容表示满意，标志着书面材料交付方式的转变。
  
  - 这表明了集成音频元素以增强用户体验和可访问性的趋势。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 可用性研究邀请**：NotebookLM UXR 正在邀请用户参加一项针对 **Audio Overviews** 的 **30 分钟** 远程可用性研究，入选者将获得 **50 美元礼品**。
  
  - 参与者需要 **高速互联网**、Gmail 账号以及功能正常的音视频设备，研究将持续到 **2024** 年底。
- **Simli 虚拟形象增强播客**：一名成员展示了 **Simli** 如何通过对 .wav 文件进行说话人日志处理（diarization）来同步音频片段，从而叠加实时虚拟形象，为未来的功能集成铺平了道路。
  
  - 这一概念验证为增强播客的用户参与度开辟了令人兴奋的可能性。
- **Pictory 在播客视频制作中的作用**：用户正在探索使用 **Pictory** 将播客转换为视频格式，并讨论了如何有效地整合演讲者的面部。
  
  - 另一位成员提到 **Hedra** 可以通过上传分割的音轨来实现角色可视化，从而促进这一过程。
- **播客生成限制**：用户反映在最初成功后，生成 **西班牙语** 播客时遇到挑战，引发了对该功能状态的疑问。
  
  - 一位用户表达了沮丧，指出：*“它在头两天运行得很好，然后就停止产出西班牙语内容了。”*
- **语音分离技术讨论**：参与者讨论了使用 **Descript** 高效隔离播客中单个说话人的方法，利用了在 **Deep Dive** 期间注意到的自动分割功能。
  
  - 一位用户评论道：*“我注意到 Deep Dive 有时会自行划分为多个章节，”* 展示了该平台简化播客制作的潜力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **AI 挑战软件工程师岗位**：一位成员指出 **AI** 正在越来越多地接管 **常规软件工程师的工作**，表明就业形势正在发生变化。
  
  - 人们对这一趋势对科技行业就业机会的影响表示担忧。
- **对深科技（Deep Tech）的兴趣日益增长**：一位成员表达了参与 **deep tech** 创新的强烈愿望，反映了对先进技术的关注。
  
  - 这突显了技术参与向更深层次（而非表面应用）发展的趋势。
- **FSDP2 API 弃用警告**：一位用户强调了关于 `torch.distributed._composable.fully_shard` 弃用的 **FutureWarning**，敦促切换到 FSDP，详情见 [此 issue](https://github.com/pytorch/pytorch/issues/114299)。
  
  - 在 **torch titan 论文** 发表见解后，这引发了关于 **fully_shard API** 持续相关性的疑问。
- **Rust 应用中的内存分析**：一位成员寻求关于使用 **torchscript** 对 **Rust** 应用进行内存分析的建议，以识别潜在的内存泄漏问题。
  
  - 他们特别希望调试涉及自定义 **CUDA kernels** 的问题。
- **ThunderKittens 演讲安排**：讨论了关于 **ThunderKittens** 的演讲计划、功能和社区反馈，并对协调工作表示感谢。
  
  - 此次参与有望加强围绕该项目的社区联系。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.2 QLoRA 训练问题**：在 **Llama 3.2 1B QLoRA** 训练期间，用户成功实现了 QAT，但在使用 **Int8DynActInt4WeightQuantizer** 时遇到了生成内容不连贯的问题。
  
  - 有人担心 **QAT** 调整可能不足，从而导致量化问题。
- **量化层引发困惑**：量化后生成的文本不连贯归因于 **QAT 训练** 和量化层中的配置错误。
  
  - 用户分享了说明 **torchtune** 和 **torchao** 版本配置错误的示例代码。
- **激活检查点减慢保存速度**：参与者质疑为何默认将 **activation checkpointing** 设置为 false，并指出 **Llama 3.2** 的检查点保存速度大幅下降。
  
  - 对方澄清说，对于较小的模型，这种开销并非必要，因为它会产生额外的成本。
- **动态缓存调整以提高效率**：关于 KV 缓存动态调整功能的提案将根据实际需求高效地分配内存。
  
  - 预计这一变化将通过减少不必要的内存使用来增强性能，特别是在长文本生成期间。
- **多查询注意力在缓存效率中的作用**：正如 **PyTorch 2.5** 增强功能讨论中所述，**multi-query attention** 的实现旨在节省 KV 缓存存储。
  
  - 分组查询注意力（Group query attention）支持被视为一项战略性进展，减轻了未来实现中手动扩展 KV 的需求。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Vyas 博士谈 SOAP Optimizer 方法**：哈佛大学博士后 **Nikhil Vyas 博士**将在即将举行的活动中讨论 **SOAP Optimizer**。欢迎参加 [Discord 活动](https://discord.com/events/954421988141711382/1293256892834910208)以获取见解。
  
  - 这为深入理解与 AI 模型相关的优化技术提供了机会。
- **Command R 模型面临 AI 检测问题**：用户报告称 **Command R 模型**输出的文本始终有 **90-95% 被识别为 AI 生成**，这引起了付费用户的沮丧。
  
  - *创造力是 AI 固有的*，这暗示了与训练数据分布相关的潜在局限性。
- **对邀请和申请回复的担忧**：成员们正在积极询问邀请状态和申请的常见回复时间，对长时间的延迟表示担忧。
  
  - 目前似乎缺乏关于潜在拒绝标准的透明度，表明需要改进沟通。
- **Embed V3 与旧版模型的辩论**：讨论重点是 **Embed V3**、**ColPali** 和 **JINA CLIP** 之间的比较，关注超越旧版 Embedding 的演进比较方法论。
  
  - 成员们对集成 **JSON 结构化输出**如何增强功能（特别是搜索能力）非常感兴趣。
- **寻求账号问题帮助**：对于账号或服务问题，建议用户直接联系 [**support@cohere.com**](mailto:support@cohere.com) 寻求帮助。
  
  - 一位积极的成员表示渴望帮助其他遇到类似问题的用户。

 

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Browserbase 获 2100 万美元融资用于 Web 自动化**：Browserbase 宣布已完成 **2100 万美元 A 轮**融资，由 **Kleiner Perkins** 和 **CRV** 领投，旨在帮助 **AI 初创公司大规模实现 Web 自动化**。在这一篇 [推文](https://x.com/pk_iv/status/1851270308701106383?s=46)中了解更多关于他们宏伟计划的信息。
  
  - *你会构建（🅱️uild）什么？* 强调了他们的目标，即通过未来的开发让初创公司更容易参与 Web 自动化。
- **ChatGPT 终于支持聊天记录搜索**：OpenAI 已在 ChatGPT Web 应用上推出了**搜索聊天记录**的功能，允许用户快速引用或继续过去的对话。此功能通过简化访问先前聊天的流程来提升用户体验。
  
  - OpenAI 在一篇 [推文](https://x.com/openai/status/1851340615344406781?s=46)中宣布了这一更新，强调了与平台之间更流水的交互。
- **Hamel Husain 警告 LLM 评估陷阱**：**Hamel Husain** 的一份指南概述了使用 LLM judges 时的常见错误，例如使用**过多指标**以及忽视领域专家的见解。他强调了经过验证的测量对于更准确评估的重要性。
  
  - 他的指南可以在这篇 [推文](https://x.com/hamelhusain/status/1851645681150382103?s=46)中找到，主张采用聚焦的评估策略。
- **OpenAI 的 Realtime API 推出新功能**：OpenAI 的 Realtime API 现在包含五个**新的表现力语音**，用于改进语音转语音应用，并由于 Prompt Caching 引入了大幅降价。这意味着**缓存文本输入可享受 50% 折扣**，**缓存音频输入可享受 80% 折扣**。
  
  - 新的定价模型促进了 API 更经济的使用，详情见其 [更新推文](https://x.com/OpenAIDevs/status/1851668229938159853?s=46)。
- **SimpleQA 旨在打击 AI 中的幻觉问题**：OpenAI 推出了新的 **SimpleQA** 基准测试，包含 **4000 个多样化问题**，用于衡量语言模型的事实准确性。这一举措直接针对 AI 输出中普遍存在的**幻觉问题**。
  
  - OpenAI 的 [公告](https://x.com/openai/status/1851680760539025639?s=46)强调了在 AI 部署中建立可靠评估标准的必要性。

 

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinycorp 对 Ethos NPU 的立场引发辩论**：成员们讨论了 **Tinycorp 对 Ethos NPU 的非官方立场**，一些人建议询问硬件细节和未来支持。
  
  - 一位用户幽默地指出，详细的问题可能会引发社区对 NPU 性能更丰富的反馈。
- **掌握 Tinybox 上的长时训练任务**：在 **Tinybox** 上管理长时训练任务的策略包括使用 **tmux** 和 **screen** 进行会话持久化。
  
  - 一位成员幽默地抱怨说，尽管有人推荐，但他还是因为懒惰而没有切换到更好的工具。
- **Qwen2 独特的构建模块引发关注**：人们对 **Qwen2** 在 **rotary embedding** 和 **MLP** 等基础元素上的非常规方法感到好奇，并对阿里巴巴的参与进行了推测。
  
  - 一位用户对这种合作表示沮丧，这增加了社区关于依赖关系的激烈讨论。
- **EfficientNet 面临 OpenCL 输出问题**：一位用户报告在用 C++ 实现 **EfficientNet** 时出现 **输出爆炸（exploding outputs）**，这引发了对调试工具的需求，以帮助比较 buffer。
  
  - 建议包括从 **tinygrad** 的实现中访问和转储 buffer 的方法，以便进行更有效的故障排除。
- **将模型导出为 ONNX：一个热门话题**：讨论集中在将 **tinygrad** 模型导出为 **ONNX** 的策略上，并建议使用现有脚本在低端硬件上进行优化。
  
  - 关于直接导出模型与用于芯片部署的替代字节码（bytecode）方法的优劣引发了辩论。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **演进中的 Mojo 惯用法**：成员们讨论了随着该语言获得新功能，**惯用 Mojo（idiomatic Mojo）** 仍在演进中，从而产生了新的最佳实践。
  
  - 与 **Python** 等更成熟的语言相比，这展示了语言惯用法的流动性。
- **学习资源匮乏**：一位成员强调了在寻找 Mojo 中学习 **线性代数（linear algebra）** 资源方面的困难，特别是在 GPU 使用和实现方面。
  
  - 有人建议与 **NuMojo** 和 **Basalt** 的项目负责人直接沟通，可能有助于解决可用资料有限的问题。
- **雄心勃勃的 C++ 兼容性目标**：成员们分享了实现与 **C++** **100% 兼容** 的雄心，讨论集中在 Chris Lattner 的潜在影响上。
  
  - 一位用户认为这将是一个 **彻底的奇迹**，反映了围绕兼容性的高风险和高关注度。
- **语法引发对话**：将 'alias' 重命名为 'static' 的提议引发了关于对 Mojo 语法影响以及可能与 C++ 用法产生混淆的辩论。
  
  - 一些成员对使用 **static** 表示担忧，认为它可能无法像在 C++ 中那样准确地代表其预期功能。
- **探索自定义装饰器**：讨论了在 Mojo 中实现 **自定义装饰器（custom decorators）** 的计划，认为这与编译时执行（compile-time execution）结合可能已经足够。
  
  - 有人指出，像 **SQL 查询验证** 这样的功能可能超出了单纯装饰器的能力范围。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Create-Llama App 发布，助力快速开发**：全新的 **create-llama** 工具允许用户在几分钟内搭建一个 LlamaIndex 应用，支持 **Next.js** 或 **Python FastAPI** 后端全栈支持，并提供多种预配置用例，如 **Agentic RAG**。
  
  - 这一集成促进了多种文件格式的摄取，显著简化了开发流程。
- **来自 ToolhouseAI 的变革性工具**：**ToolhouseAI** 提供了一系列高质量工具，可增强 LlamaIndex Agent 的生产力，在最近的一次黑客松中，这些工具因大幅缩短开发时间而备受关注。
  
  - 这些工具旨在无缝集成到 Agent 中，在加速工作流方面证明了其有效性。
- **增强的多 Agent 查询流水线**：一位成员展示了使用 **LlamaIndex workflows** 构建多 Agent 查询流水线，并推介该方法是实现协作的有效方式。
  
  - 演示材料可以在[此处](https://github.com/run-llama/multi-agent-concierge/tree/main/video_tutorial_materials)获取，以进一步探索实现策略。
- **RAG 与 Text-to-SQL 集成见解**：一篇文章详细阐述了如何使用 [LlamaIndex](https://medium.com/ai-artistry/unleashing-the-power-of-rag-and-text-to-sql-with-llamaindex-5aa27c697ad0) 将 **RAG (Retrieval-Augmented Generation)** 与 **Text-to-SQL** 集成，展示了在查询处理方面的改进。
  
  - 用户报告查询响应时间**减少了 30%**，强调了 LlamaIndex 在提高数据检索效率方面的作用。
- **通过 LlamaIndex 增强用户交互**：LlamaIndex 旨在通过将自然语言输入自动生成 SQL 来简化用户与数据库的交互，从而进一步赋能用户。
  
  - 事实证明，这种方法非常有效，用户表示即使没有深厚的技术知识，也能更有信心地提取数据。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **IReRa 应对多标签分类**：题为 [IReRa: In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178) 的论文提出了 **Infer–Retrieve–Rank**（推理-检索-排序）框架，以提高语言模型在多标签任务中的效率，在 HOUSE、TECH 和 TECHWOLF 基准测试中取得了顶尖成绩。
  
  - 这凸显了缺乏类别先验知识的 LLM 所面临的困境，并提出了一个可以增强整体性能的新框架。
- **与 IReRa 相关的 GitHub 仓库**：成员们注意到了论文摘要中提到的相关 [GitHub repo](https://link.to.repo)，这为讨论的方法论提供了进一步的见解。
  
  - 这将极大地有助于论文研究结果的实现和理解。
- **关于 DSPy 强制结构的辩论**：一位成员质疑，当像 Outlines 这样的库可以更高效地处理结构化生成时，DSPy 是否有必要强制执行结构。
  
  - 另一位贡献者指出，DSPy 自 v0.1 以来的结构强制执行对于从 Signature 到 Prompt 的准确映射至关重要，在有效性与质量之间取得了平衡。
- **质量与结构的对决**：随着对结构化输出可能降低输出质量的怀疑出现，讨论变得激烈起来，有人建议约束实际上可以增强结果，特别是对于较小的模型。
  
  - *这种方法可能会产生很好的效果，特别是对于较小的 LLM，* 这反映了关于质量和格式遵循度的不同观点。
- **将 MIPROv2 与 DSPy 集成**：一位成员分享了利用 Zero-shot **MIPROv2** 配合 Pydantic 优先接口进行结构化输出的见解，主张在 DSPy 的优化过程中进行更多集成。
  
  - *他们表达了对以更集成和原生的方式处理结构化输出的渴望，* 预示着工作流可能得到改进。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **任务自动化预测引发劳动力辩论**：一位用户预测 **virtual beings**（虚拟存在）将导致职位冗余，并将其比作**虚拟天网接管**。
  
  - 这引发了关于 AI 对就业和未来职业格局总体影响的激烈讨论。
- **Open Interpreter 相较于 Claude 的优势**：一名成员询问 **Open Interpreter** 在计算机操作方面与 **Claude** 有何不同。
  
  - **Mikebirdtech** 强调了在 **Claude** 中利用 `interpreter --os` 的功能，并突出了其**开源**（**open-source**）的优势。
- **聊天配置文件恢复引发疑问**：一位用户寻求关于如何使用之前激活的特定 **profile/model**（配置文件/模型）来恢复聊天的建议。
  
  - 尽管使用了 `--conversations`，它仍默认指向**标准模型**，这让用户在寻找解决方案。
- **ChatGPT 聊天历史搜索功能推出**：OpenAI 宣布推出一项功能，允许用户在 **ChatGPT web** 端搜索其**聊天历史**，增强了参考便利性。
  
  - 这一新功能旨在简化用户交互，提升平台上的整体体验。
- **气味数字化取得重大里程碑**：一个团队在没有任何人工干预的情况下成功数字化了**夏李**（summer plum）的气味，实现了重大突破。
  
  - 一名成员表达了对携带**李子香气**的兴奋，并考虑发布一款独家香水来资助科学探索。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Invoke 函数性能之谜**：调用 retriever 的 **.invoke function** 让用户感到困惑，**Llama3.1:70b** 模型的**响应时间**超过 **120 秒**，而本地仅需 **20 秒**。
  
  - 有人怀疑是**安全问题**影响了性能，促使社区协助排查这一异常。
- **FastAPI 路由执行性能**：通过调试日志确认，**FastAPI** 路由表现出令人印象深刻的性能，执行时间始终保持在 **1 秒**以内。
  
  - 用户确认发送的数据是准确的，从而将响应速度问题锁定在 invoke 函数本身。
- **对 Hugging Face 文档的挫败感**：对于旨在设置 **chat/conversational pipeline** 的用户来说，查阅 **Hugging Face Transformers** 的文档一直是一件令人头疼的事。
  
  - 在文档中难以找到核心指导，突显了用户入门体验中需要改进的领域。
- **Knowledge Nexus AI 发起社区倡议**：Knowledge Nexus AI (KNAI) 宣布了旨在连接**人类知识**与 **AI** 的新倡议，重点关注**去中心化**方法。
  
  - 他们的目标是将集体知识转化为**结构化、机器可读的数据**，从而对医疗保健、教育和供应链产生影响。
- **OppyDev 推出插件系统**：**OppyDev 的插件系统**通过创新的 **chain-of-thought**（思维链）推理增强了标准 AI 模型的输出，以提高响应的清晰度。
  
  - 一段教程视频[演示了该插件系统](https://www.youtube.com/watch?v=6JlQwnYn7RY&t=14s)，并展示了 AI 交互中的实际改进。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **LoRA 微调问题仍未解决**：一名成员表示在 **H100 GPUs** 上进行 **LoRA finetuning** 难以找到解决方案，并暗示 **QLoRA** 可能是唯一可行的权宜之计。
  
  - 该问题依然存在，另一名成员确认针对 **Hopper 8bit** 的 **BitsAndBytes issue** 仍处于开启状态且未得到解决。
- **量化挑战依然存在**：讨论强调了量化相关问题的持续挑战，特别是在 **Hopper 8bit** 的 **BitsAndBytes** 背景下。
  
  - 尽管做出了努力，但似乎尚未就这些技术问题建立明确的解决方案。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **图像解码中 Clamping 数值至关重要**：一位成员强调，在将解码后的图像转换为 uint8 之前，如果未能将 **数值限制 (clamp)** 在 [0,1] 范围内，会导致 **超出范围的数值回绕 (wrapping)**，从而影响图像质量。
  
  - *图像外观出现意外结果* 可能是由于在预处理链中忽略了这一关键步骤。
- **解码工作流中可能潜伏的缺陷**：有人对 **解码工作流中可能存在的缺陷** 提出了担忧，这可能会影响整体图像处理的可靠性。
  
  - 需要进一步讨论以彻底识别这些问题并增强工作流的鲁棒性。
- **关于图像处理的新 arXiv 论文**：一位成员分享了一篇名为《Research Paper on Decoding Techniques》的新 arXiv 论文链接，可在此处 [查看](https://arxiv.org/abs/2410.20424)。
  
  - 该论文可能会为当前关于图像解码的讨论提供有价值的见解或方法论。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents 测验位置公布**：一位成员询问了 **LLM Agents** 课程 **每周测验** 的位置，并得到了带有 [测验链接](https://llmagents-learning.org/f24) 的迅速回复，回复称：*“你可以在这里找到所有的测验。”*
  
  - 这些测验对于跟踪课程进度至关重要，可以通过提供的链接访问。
- **准备好参加 LLM Agents 黑客松！**：参与者们了解了即将举行的 **LLM Agents Hackathon**，并获得了 [黑客松详情](https://rdi.berkeley.edu/llm-agents-hackathon/) 链接以报名参加这场编程竞技。
  
  - 此次活动为参与者提供了一个展示技能并就创新项目进行协作的绝佳机会。
- **便捷的课程报名流程**：分享了如何通过 **Google Form** 注册课程的说明，鼓励参与者填写此 [表格](https://forms.gle/svSoNhKcGFjxup989) 加入。
  
  - 这一简单的报名流程旨在提高入学率，让更多工程师参与到该计划中。
- **加入 Discord 上活跃的课程讨论**：提供了加入 [LLM Agents Discord](https://discord.gg/NWVpQ9rBvd) 中 **MOOC 频道** 讨论的详细信息，以促进社区参与。
  
  - 参与者可以在整个课程期间利用该平台提问并分享见解。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Transformer Labs 展示在 LLM 上运行本地 RAG**：**Transformer Labs** 正在举办一场活动，演示如何在可本地安装且具有用户友好 UI 的环境下，在 **LLM** 上进行 **RAG** 的训练、微调和评估。
  
  - 这种无代码（no-code）方法有望让各种技能水平的工程师都能参与此次活动。
- **技术演讲中介绍 Lumigator 工具**：工程师们将深入介绍 **Lumigator**，这是一个开源工具，旨在协助根据特定需求选择最佳的 **LLM**。
  
  - 该工具旨在加快工程师在选择大语言模型时的决策过程。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Llama-3.1-8B-Instruct (FC) 在与 Prompting 模式对比时表现不佳**：一位成员提出 **Llama-3.1-8B-Instruct (FC)** 的表现不如 **Llama-3.1-8B-Instruct (Prompting)**，并对 Function Calling 任务的预期结果表示怀疑。
  
  - *“这种差异的原因是什么？”* 表明了对基于模型预期功能的性能预期的担忧。
- **对 Function Calling 机制的期望**：另一位参与者表达了失望，认为考虑到其设计重点，**FC** 变体应该优于其他变体。
  
  - 这引发了关于当前结果是令人惊讶，还是暗示了模型内部潜在架构问题的讨论。

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1300899341870956635) (161 条消息🔥🔥):

> - `Apple's MacBook Pro Announcement` (Apple MacBook Pro 发布会)
> - `M3 Max Performance` (M3 Max 性能)
> - `Model Access and Inference` (模型访问与推理)
> - `H100 GPU Rental Pricing` (H100 GPU 租赁价格)
> - `Local vs. Remote Model Usage` (本地与远程模型使用)

- **Apple MacBook Pro 展示 LM Studio**: 在最近的 Apple 发布会上，LM Studio 被重点展示，体现了其在搭载 M4 系列芯片的新款 MacBook Pro 上的能力。
  
  - 成员们对开发者在重大商业广告中获得这一重要认可表示兴奋和祝贺。
- **M3 Max 取得令人印象深刻的性能**: 用户报告称，M3 Max 运行 Phi 3.5 MoE 等模型时速度约为每秒 **60 tokens**，强调了其即使在较低配置下的效率。
  
  - 对比讨论建议，对于需要极致速度的用户，像 A6000 专门的 GPU 可能仍然更有利。
- **探索本地模型访问**: 成员们讨论了为本地模型提供互联网访问的各种方法，建议包括端口转发以及设置 Open WebUI 连接 LM Studio。
  
  - 有人指出，某些框架可能允许更轻松地集成以访问 Web 数据，尽管仍期待更直接的选择。
- **H100 GPU 租赁价格下降**: 用户提到 H100 租赁价格现在约为 **每小时 1 美元**，使得模型推理变得更加平易近人。
  
  - 尽管如此，关于使用强大硬件与本地模型的实用性考量仍存在争议。
- **实现 FFN 模块的挑战**: 一位用户分享了他们在模型代码中实现 Feedforward Network (FFN) 模块时的困扰，尽管遵循了指南，但输出仍为乱码。
  
  - 这引发了关于调试和优化模型代码以提高输出质量的讨论。

**提到的链接**:

- ['Let chaos reign': AI inference costs are about to plummet](https://www.businessinsider.com/new-players-startups-ai-inference-driving-prices-down-cheap-workload-2024-10) : 推理正成为 AI 时代的商品。
- [New MacBook Pro features M4 family of chips and Apple Intelligence](https://t.co/8tloKJmq4Q): Apple 今天发布了新款 MacBook Pro，搭载 M4 系列芯片：M4、M4 Pro 和 M4 Max。
- [Mochi Peach GIF - Mochi Peach Cat - Discover & Share GIFs](https://tenor.com/view/mochi-peach-cat-party-props-gif-27063000): 点击查看 GIF
- [$2 H100s: How the GPU Bubble Burst](https://www.latent.space/p/gpu-bubble): H100 曾经的价格是 8 美元/小时（如果你能租到的话）。现在有 7 个不同的二手市场以低于 2 美元的价格出售。发生了什么？
- [ggml/docs/gguf.md at master · ggerganov/ggml](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md): 用于机器学习的 Tensor 库。通过在 GitHub 上创建账户参与 ggml 开发。
- [ggerganov - Overview](https://github.com/ggerganov): 我喜欢大的 .vimrc，我不说谎。ggerganov 有 71 个公开仓库。在 GitHub 上关注他们的代码。
- [magnolia1234/bpc_uploads](https://gitflic.ru/project/magnolia1234/bpc_uploads/blob?file=bypass_paywalls_clean-latest.xpi&branch=main): 通过在 GitFlic 创建账户参与 magnolia1234/bpc_uploads 的开发。

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1300897493730721852) (600 条消息🔥🔥🔥):

> - `M4 Ultra 预期`
> - `Apple Silicon 对比`
> - `GPU 性能讨论`
> - `AI 模型在 GPU 上的适配`
> - `AI 任务中的 Windows vs. Linux`

- **M4 Ultra 预期升温**：传闻即将推出的 M4 Ultra 将支持高达 **256GB** 的统一内存，且性能有望较 M2 Ultra 实现显著提升。
  
  - 预计 M4 Max 将配备 **128GB** 内存，并有推测认为 M4 的性能将直逼目前的 **4090 GPU**。
- **Apple Silicon 性能对比**：用户对 M4 系列与现有 NVIDIA GPU 在运行大语言模型 (LLM) 时的实际表现对比感到好奇。
  
  - 据报道，目前 M2 Ultra 在运行 Mistral 等模型时可达到 **8 - 12 T/S**，而 M4 预计会将这一数值推向更高。
- **GPU 性能讨论**：讨论集中在 M4 的原始 GPU 性能可能比 M2 Ultra 高出 **35-40%**，从而提升整体效率。
  
  - 内存带宽的增加也备受期待，这使得 M4 成为处理密集型计算负载任务的一次重大升级。
- **AI 模型在 GPU 上的适配**：用户指出，在当前硬件配置有限的统一内存限制下，适配如 **60B** 等大型模型面临挑战。
  
  - 在运行 Mistral 等高级 AI 模型时，有效利用内存资源至关重要，需在性能与可用内存之间取得平衡。
- **AI 任务中的 Windows vs. Linux**：用户对 Windows 在 AI 任务和模型运行中受限的性能和控制力表示沮丧。
  
  - 相比之下，Linux 提供了更高的灵活性和效率，特别是在各种 AI 应用的 GPU 利用率方面。

**提及的链接**：

- [AMD Navi 22 GPU Specs](https://www.techpowerup.com/gpu-specs/amd-navi-22.g951)：2560 Cores, 160 TMUs, 64 ROPs
- [Qwen2.5 - a Qwen Collection](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e)：未找到描述
- [EP2098941A1 - Computer apparatus - Google Patents](https://patents.google.com/patent/EP2098941A1/en)：未找到描述
- [Tweet from Patrick Wardle (@patrickwardle)](https://fixupx.com/patrickwardle/status/1318465421796782082)：一个例子，两个 macOS 防火墙：LuLu 和 Little Snitch。尽管尽了最大努力（例如禁用默认规则、创建显式阻止规则、启用“拒绝模式”），Apple 的 App Store 依然...
- [NVIDIA GeForce RTX 4060 Ti 16 GB Specs](https://www.techpowerup.com/gpu-specs/geforce-rtx-4060-ti-16-gb.c4155)：NVIDIA AD106, 2535 MHz, 4352 Cores, 136 TMUs, 48 ROPs, 16384 MB GDDR6, 2250 MHz, 128 bit
- [GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)：使用多个 NVIDIA GPU 还是 Apple Silicon 进行大语言模型推理？- XiongjieDai/GPU-Benchmarks-on-LLM-Inference
- [GitHub - Blaizzy/mlx-vlm: MLX-VLM is a package for running Vision LLMs locally on your Mac using MLX.](https://github.com/Blaizzy/mlx-vlm)：MLX-VLM 是一个用于在 Mac 上使用 MLX 本地运行 Vision LLM 的包。- Blaizzy/mlx-vlm
- [Feature Request: NPU Support · Issue #9181 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/9181)：前提条件：我正在运行最新代码。如果可能请注明版本。我仔细阅读了 README.md。我使用了与问题相关的关键词进行搜索，以确保我创建的是...

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1300905431794974801) (326 条消息🔥🔥):

> - `Hugging Face API 使用`
> - `使用 Ollama 进行图像分析`
> - `机器学习教育`
> - `Transformers 与 Attention 模型`
> - `Docker Spaces 与私有镜像`

- **探索 Hugging Face API 获取 Token 概率**：一位成员询问是否可以通过 Hugging Face 的 serverless inference API 获取大型语言模型（LLM）的 Token 概率，并得到了肯定的答复。
  
  - 讨论中强调了使用 inference client 可以简化流程，随后还讨论了速率限制（rate limits）和 API 的使用方法。
- **使用 Ollama 进行图像分析与本地文件访问**：一位用户对使用 Ollama 进行图像分析时的隐私问题表示担忧，注意到它可以访问本地文件。
  
  - 对方澄清说 Ollama 在本地运行，处理数据时不会发送到任何服务器，从而保证了分析过程的隐私性。
- **选择机器学习与数据科学专业**：一位成员分享了关于选择大学专业的见解，强调了在数据科学领域拥有广泛知识的重要性，而不仅仅是局限于 AI。
  
  - 参与者讨论了扎实的数学基础和相关的编程经验对于机器学习职业生涯的关键意义。
- **Transformers 与 Attention 机制的新发现**：一场关于 Transformers 中是否需要独立 Attention 模型的讨论展开，最终大家意识到 Transformers 可能已经在内部集成了这一功能。
  
  - 鉴于社区成员们的新理解，大家对之前花在研究这一方面的时间表示了关注。
- **Docker Spaces 与私有基础镜像的挑战**：用户讨论了构建 Docker Spaces 的经验，特别是在使用私有基础镜像以及构建过程中出现任务超时的问题。
  
  - 成员们分享了使用公共镜像以避免问题的建议，并提出了故障排除方案，例如利用 factory rebuilds。

**提到的链接**：

- [xxxxxxx (sayaka.M)](https://huggingface.co/xxxxxxx)：未找到描述
- [Rate Limits](https://huggingface.co/docs/api-inference/en/rate-limits)：未找到描述
- [Wan Im Rich GIF - Wan Im Rich Rich - Discover & Share GIFs](https://tenor.com/view/wan-im-rich-rich-gif-18416070)：点击查看 GIF
- [Nervous Hot GIF - Nervous Hot Sweat - Discover & Share GIFs](https://tenor.com/view/nervous-hot-sweat-sweating-perspire-gif-10513221)：点击查看 GIF
- [KoboldAI Lite](https://botlicker.org)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1301088321317113909) (8 条消息🔥):

> - `Llama-3.1 70B 兼容性`
> - `并行计算设置`
> - `Hugging Face 上的微调数据集`
> - `用于问答和情感分析的 LLM 推荐`

- **Llama-3.1 70B 挑战 PC 极限**：一位成员担心他们的 PC 是否能运行 **Llama-3.1 70B**，表示他们已经非常接近运行门槛了。
  
  - *他们分享了当前配置的局限性以及对并行计算的需求。*
- **微调讨论缺乏专用空间**：一位成员询问是否有专门讨论 **fine-tuning**（微调）的频道，另一位成员回答说频道通常是按不同的模态（modalities）划分的。
  
  - *他们还被提醒不要在多个频道重复发帖。*
- **在 Hugging Face 上添加微调数据集**：为了帮助进行 **fine-tuning**，一位成员被引导至 Hugging Face 的 [数据集准备课程](https://huggingface.co/learn/nlp-course/chapter5/2?fw=pt) 以进行快速设置。
  
  - *这为在 Hub 上没有现成数据集时如何开始微调提供了思路。*
- **用于通用问答和情感分析的 LLM 规模选择**：一位用户讨论了在 RTX 3090 配置的限制下，努力寻找能够处理**通用问答**和**情感分析**的 LLM。
  
  - *他们提到尝试加入笔记本电脑的资源来提供额外支持，但在设置并行计算时遇到了挑战。*

 

**提到的链接**：[What if my dataset isn’t on the Hub? - Hugging Face NLP Course](https://huggingface.co/learn/nlp-course/chapter5/2?fw=pt)：未找到描述

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1300948018706645112) (17 条消息🔥):

> - `Latent Space Regularization`
> - `LlamaIndex 中的 Anthropic Agent`
> - `计算建模指南`
> - `图灵的贡献`
> - `Nomic Atlas 洞察`

- **Latent Space Regularization 详解**：一篇文章讨论了 **Latent Space Regularization**，详细介绍了探测行为背后的算法以及探索与计算变量相关的神经相关性的技术。
  
  - 论文强调了应用十条简单规则的重要性，以确保计算建模能产生有意义的见解。
- **在 LlamaIndex 中使用 Anthropic Agent**：[LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/anthropic_agent/) 提供了一个 Notebook，展示了如何利用具有 function calling 能力的 Anthropic agent，特别是针对 claude-3 模型。
  
  - 它引导用户完成初始设置过程，并强调了所需库的安装。
- **计算建模指南**：一份全面的介绍提供了有效计算建模的十条规则，旨在帮助研究人员避免陷阱，并将模型与数据正确关联。
  
  - 这些指南既适用于初学者也适用于高级技术，强调了谨慎应用以避免误导性结论的重要性。
- **重温图灵的原始论文**：一本名为《*Annotated Turing*》的书扩展了 Alan Turing 的基础性工作，通过丰富的生平背景和贡献介绍，使现代读者能够理解他复杂的思想。
  
  - 该书提供的注释阐明了图灵关于可计算性理论（computability theory）的原始论述及其对当代编程的影响。
- **社交媒体上的国会话语**：使用 [Nomic Atlas](https://www.nomic.ai/blog/posts/atlas-story-congressional-tweets) 进行的分析审查了 **320 万条美国国会议员的帖子**，揭示了 2024 年大选前的沟通模式和关键话题。
  
  - 该工具旨在通过将复杂的数据集转化为易于理解的洞察，为政策研究人员和积极参与的公民赋能。

**提到的链接**：

- [来自 Nomic AI (@nomic_ai) 的推文](https://x.com/nomic_ai/status/1851642971575255492)：来自国会的 320 万条 @X 帖子展示了美国立法者如何交流？在进入 2024 年美国总统大选之际，他们发布了什么内容？了解我们在 320 万条国会帖子中发现的内容...
- [Pdf2audio - lamm-mit 开发的 Hugging Face Space](https://huggingface.co/spaces/lamm-mit/PDF2Audio)：未找到描述
- [Function Calling Anthropic Agent - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/agent/anthropic_agent/)：未找到描述
- [美国国会在大选前夕发布了什么内容？](https://www.nomic.ai/blog/posts/atlas-story-congressional-tweets)：探索超过 300 万条来自美国立法者的 X/Twitter 帖子
- [行为数据计算建模的十条简单规则](https://elifesciences.org/articles/49547)：认知和神经科学数据的计算建模是一个富有洞察力且强大的工具，但存在许多潜在的陷阱，可以通过遵循简单的指南来避免。
- [无标题](https://www.amazon.co.uk/Annotated-Turing-Through-Historic-Computability/dp/0470229055)：未找到描述

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1300898898054746195) (3 messages):

> - `Transformer Tokenizer Updates`
> - `GPUs and Docker Integration`
> - `New Blog Post on Docker`
> - `Dstack Task Configurations`

- **Transformer Tokenizer 增强功能即将推出**：一名成员建议将 Tokenizer 推送到 Transformers 中的 **AutoTokenizer** 类，并表示这**完全可行**且**可以实现**。
  
  - 另一名成员确认这确实在**计划**之中，目前正在开发。
- **关于 Docker 与 HF Chat UI 的精彩新博客**：一名成员宣布发布了一篇博客文章，详细介绍了如何使用 Docker 和 Docker Compose 在启用 GPU 的容器中部署 **HF Chat UI**，链接见[此处](https://dstack.ai/blog/docker-inside-containers/)。
  
  - 该文章解释了如何在不直接与 Docker 交互的情况下，通过 **dstack** 使用你自己的 Docker 镜像，同时指出某些现有代码可能仍需要直接交互。
- **用于 Docker 和 Compose 的 Dstack 配置**：最新的 **dstack** 版本允许通过将 `image` 设置为 `dstackai/dind` 并将 `privileged` 设置为 true（包括命令 `start-dockerd`）来在你的配置中使用 Docker。
  
  - 这使得在通过 **dstack** 初始化后可以直接使用 Docker 命令，从而简化了开发环境中的部署流程。

**提到的链接**：[在启用 GPU 的容器中使用 Docker 和 Docker Compose - dstack](https://dstack.ai/blog/docker-inside-containers/)：dstack 的最新版本允许在运行配置中直接使用 Docker 和 Docker Compose。

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1301227310313377872) (2 messages):

> - `User Engagement`
> - `Future Discussions`

- **用户对侧面角色讨论的兴趣**：一名成员表达了参与侧面角色讨论的愿望，显示出对该话题的浓厚兴趣。
  
  - 这反映了社区内对于探索主线叙事之外的角色发展的热情日益增长。
- **对即将到来的对话的期待**：有迹象表明，成员们渴望参与未来围绕侧面角色话题的对话。
  
  - 这种情绪凸显了团队对于丰富讨论内容的持续投入。

---

### **HuggingFace ▷ #**[**NLP**](https://discord.com/channels/879548962464493619/922424173916196955/1301002825735733368) (10 messages🔥):

> - `Qwen 2 model issues`
> - `Langchain SQL agent with GPT-4`
> - `Mini Omni 2 feedback`

- **Qwen 2 模型在 Token 生成方面遇到困难**：一位新用户报告了 **Qwen 2 基础模型**的问题，具体表现为输出末尾出现随机 Token，导致重复。
  
  - 另一名成员澄清说，这可能是由于 **EOS Token** 未被识别或模型达到了其上下文长度限制。
- **在 Langchain SQL Agent 中使用 GPT-4 的挑战**：一名成员成功地将 **Langchain SQL Agent** 与 **GPT-3.5 Turbo** 结合使用，但在切换到 **GPT-4** 时遇到了困难。
  
  - 由于 **GPT-3.5 Turbo** 即将停用，引发了关于替代方案的讨论。
- **关于 Mini Omni 2 模型反馈**：一位用户对新发布的 **Mini Omni 2** 表示感兴趣，但发现其在**语言支持**和**会话历史范围**方面存在不足。
  
  - 他们询问是否有人知道具有类似功能且值得探索的其他替代模型。

---

### **HuggingFace ▷ #**[**diffusion-discussions**](https://discord.com/channels/879548962464493619/1009713274113245215/1301099015429029932) (3 messages):

> - `Diffusion models for non-standard data`
> - `FoldingDiff project`
> - `Consistency Models in AI`

- **探索独特数据上的 Diffusion Models**：研究表明，通过为每个特征通道采用定制的噪声因子，**diffusion models** 可以有效地处理非标准数据类型。
  
  - 这种方法对于具有不同行为的特征特别相关，例如可能需要不同损坏模式的**角向（angular）**和**非角向（non-angular）**特征。
- **FoldingDiff 独特的 Diffusion 方法**：一位成员分享了关于 [FoldingDiff 项目](https://github.com/microsoft/foldingdiff) 的见解，强调其将 diffusion models 用于**蛋白质结构（protein structure）**分析，重点关注三角函数和 attention 机制。
  
  - 该项目提出了创新的方法论，可能通过定制的计算技术增强蛋白质建模。
- **关于 Consistency Models 论文的讨论**：一位成员就题为 [Simplifying, Stabilizing, and Scaling Continuous-Time Consistency Models](https://openai.com/index/simplifying-stabilizing-and-scaling-continuous-time-consistency-models/) 的论文寻求见解，以收集关于该主题的想法。
  
  - 这一询问反映了人们对理解和评估 **consistency models** 在各种 AI 应用中有效性的兴趣日益浓厚。

 

**提到的链接**：[GitHub - microsoft/foldingdiff: Diffusion models of protein structure; trigonometry and attention are all you need!](https://github.com/microsoft/foldingdiff): Diffusion models of protein structure; trigonometry and attention are all you need! - microsoft/foldingdiff

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1300914314693050430) (89 条消息🔥🔥):

> - `Gradient Accumulation Issues` (梯度累积问题)
> - `Apple's New Mac Mini Release` (苹果新款 Mac Mini 发布)
> - `Training Foundation Models` (训练基础模型)
> - `Dataset Preparation in ML` (机器学习中的数据集准备)
> - `Vision Fine-Tuning Delay` (视觉微调延迟)

- **Unsloth 团队调查梯度累积 (Gradient Accumulation)**：[Unsloth 团队](https://unsloth.ai/blog/gradient)发布了一份报告，揭示了训练框架中梯度累积的关键问题，特别是在语言模型生成方面。
  
  - 他们发现使用传统 Batch Size 时输出存在不一致性，从而影响了 Loss 计算。
- **苹果发布新款 Mac Mini**：苹果发布了搭载 M4 和 M4 Pro 芯片的新款 [Mac mini](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/)，并将其标榜为首款碳中和 Mac。
  
  - 其紧凑的设计带来了高达 1.8 倍的 CPU 性能提升，且尺寸小到可以单手掌控。
- **关于基础模型 (Foundation Models) 与微调 (Fine-tuning) 的讨论**：成员们就微调后的 Llama 70B 模型是否应归类为基础模型展开了辩论，并承认了训练中所涉及的复杂性。
  
  - 讨论中提出了对模型准确性和平衡数据集需求的担忧，并建议在特定任务中使用更简单的分类模型。
- **AI 模型微调的准备工作**：讨论围绕数据集准备的最佳实践展开，强调了质量和平衡对于提高模型准确性的重要性。
  
  - 成员们为微调和数据集平衡的新手推荐了来自 Hugging Face 的资源。
- **视觉微调 (Vision Fine-tuning) 延迟**：官方宣布原定于今天进行的视觉微调将推迟，目前预计在本周晚些时候或下周初进行。
  
  - 社区成员表达了期待并对正在进行的工作表示感谢，还幽默地引用了甘道夫 (Gandalf) 的梗图来调侃时间进度。

**提到的链接**：

- [Marcel Binz (@marcel_binz) 的推文](https://x.com/marcel_binz/status/1850806691958313160)：很高兴宣布 Centaur —— 首个关于人类认知的基础模型 (Foundation Model)。Centaur 可以预测并模拟任何可用自然语言表达的实验中的人类行为。你可以随时下载...
- [苹果新款 Mac mini 性能更强、体积更小，专为 Apple Intelligence 打造](https://www.apple.com/newsroom/2024/10/apples-new-mac-mini-is-more-mighty-more-mini-and-built-for-apple-intelligence/)：苹果今天发布了全新的 Mac mini，由 M4 和新款 M4 Pro 芯片驱动，并围绕 Apple Silicon 进行了重新设计，使其体积更加小巧。
- [霍比特人甘道夫 GIF - Hobbit Gandalf Wizard](https://tenor.com/view/hobbit-gandalf-wizard-late-ian-mckellen-gif-12948949)：点击查看 GIF
- [Zach Mueller - PyTorch, Gradient Accumulation, and the dreaded lack of reproducability](https://muellerzr.github.io/blog/gradient_accumulation_part2.html)：关于 PyTorch、梯度累积以及可怕的可复现性缺失。
- [[AMD] Triton Backend for ROCm by micmelesse · Pull Request #1203 · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/pull/1203)：这是一个为 ROCm 上的 Flash Attention 添加 Triton 后端的 PR。我们希望这个 PR 能成为实现该目标的系列 PR 中的第一个。Triton 已经支持 ROCm 一段时间了，而 Flash...

---

### **Unsloth AI (Daniel Han) ▷ #**[**off-topic**](https://discord.com/channels/1179035537009545276/1179039861576056922/1300897569878179912) (41 messages🔥):

> - `提早离校`
> - `衰老的经历`
> - `技术领域的 R&D`
> - `硕士毕业后的求职`
> - `协作工作邀约`

- **提早离校会带来独特的视角**：一位成员分享了他们在 17 岁离校的经历，并指出**在传统教育之外学习**对他们更有利。
  
  - *无论是否得到他人的认可，我的经历都是有价值的*，这彰显了对学术规范之外个人成长的坚定信念。
- **对衰老的反思带来了复杂的情感**：另一位成员表示，步入中年是人生的剧烈变化，称 *随着预期的转变，我的整个生活都发生了翻天覆地的变化*。
  
  - 他们伤感地观察到，许多 20 多岁时的朋友已经离世，强调了**随着年龄增长，人生经历产生的巨大差异**。
- **R&D 角色既令人兴奋又让人不知所措**：一位最近完成硕士学位的成员表示，他们感觉自己像技术世界的菜鸟，仅仅触及了知识的皮毛。
  
  - 他们对新的 R&D 职位表达了热情，说道：*我很幸运能发现这个世界*，并能通过探索它获得报酬。
- **协作工作邀约可能获利颇丰**：一位成员为一个论文项目寻求帮助，为协助录入 200 多页内容提供 **每条 2 sats** 的报酬。
  
  - 他们提到，尽管想使用 AI 寻求帮助，但 *AI 无法理解这些内容*，反映了对当前 AI 局限性的沮丧。
- **对编程测试挑战的幽默看法**：一位成员幽默地分享了他们收到 HackerRank 挑战时的反应，表示这种经历可能令人望而生畏。
  
  - 他们使用了一个表情包来表达这种感觉，传达了许多人在技术测试场景中都会面临的共鸣式挣扎。

 

**提到的链接**：[Brain Dog Brian Dog GIF - Brain dog Brian dog Cooked - Discover & Share GIFs](https://tenor.com/view/brain-dog-brian-dog-cooked-wallahi-im-finished-cooked-dog-gif-1849480349705279416)：点击查看 GIF

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1300907675206291496) (28 messages🔥):

> - `使用 Unsloth 进行持续预训练 (Continued Pretraining)`
> - `Unsloth 的安装问题`
> - `使用自定义数据集微调模型`
> - `使用 Llama 模型进行指令微调 (Instruct Fine-Tuning)`
> - `微调期间的 GPU VRAM 管理`

- **Unsloth 的持续预训练以高效著称**：Unsloth 的最新版本允许对 LLM 进行持续预训练，速度比其他方案快 ***2 倍***，且节省 ***50% 的 VRAM***，并为 Mistral v0.3 7b 提供了一个 [免费 Colab notebook](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing)。
  
  - 关键见解包括关于微调 Embedding 以及使用不同学习率来维持稳定性的建议。
- **Unsloth 软件包的安装困扰**：一位用户在尝试安装 Unsloth 时遇到了安装错误，但在发现软件包名称 'bitsandbytes' 中的拼写错误后解决了问题。
  
  - 另一位成员提醒了隔离 Python 环境的重要性，特别是对于 Linux 系统。
- **关于使用自定义 JSON 数据集进行微调的指导**：一位用户寻求帮助，希望使用嵌套的 JSON 数据集格式微调模型，旨在学习使用 Unsloth 的处理流程。
  
  - 回复强调了通用方法，但提到在模型训练期间需要正确的格式化和兼容性。
- **使用 Meta Llama 模型进行指令微调**：一位用户在尝试使用 Meta Llama3.1 8B Instruct 模板进行微调时遇到了张量形状不匹配 (tensor shape mismatch) 错误，并寻求建议。
  
  - 他们更换了模型，但在合并和加载模型时仍然面临问题，这表明可能存在兼容性缺口。
- **针对 VRAM 限制探索 OLMo 的微调选项**：一位社区成员询问了关于使用 Unsloth 微调 allenai/OLMo-7B-0724-Instruct 模型的问题，特别是关于 VRAM 的占用情况。
  
  - 这一询问反映了许多用户在资源有限的情况下进行模型微调时面临的持续挑战。

**提到的链接**：

- [Miniconda — Anaconda 文档](https://docs.anaconda.com/miniconda/)：未找到描述
- [使用 Unsloth 进行持续 LLM 预训练](https://unsloth.ai/blog/contpretraining)：通过使用 Unsloth 对 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1300921043950633011) (12 messages🔥):

> - `ThunderKittens 更新`
> - `研究中的 Rickroll`
> - `社区反应`
> - `关于 ThunderKittens 的论文`

- **ThunderKittens 团队带着新功能回归**：ThunderKittens 团队发布了一篇 [博客文章](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)，重点介绍了几个新的和改进的功能，包括令人兴奋的 kernels 和对话模型。
  
  - 他们幽默地提到了之前的作品在社交媒体上受到的热烈欢迎，并俏皮地提到了 *额外可爱的猫咪*。
- **社区 Rickroll 引用**：一位成员对被 Rickroll 感到惊讶和有趣，这指的是 ThunderKittens 更新中的一个幽默元素。
  
  - 另一位成员开玩笑说他们不会透露细节，保持 Rickroll 的神秘感。
- **询问 ThunderKittens 论文**：一位成员询问是否有与 ThunderKittens 相关的论文，随后有人分享了 [arXiv 链接](https://arxiv.org/pdf/2410.20399)。
  
  - 讨论暗示了对深入研究论文内容的兴趣。
- **社区喜爱这种轻松的氛围**：几位成员对 ThunderKittens 团队轻松幽默的方式表示赞赏，指出他们的风格非常有趣。
  
  - 一位成员表示他们非常喜欢该团队这种“不严肃”的态度，展示了社区的参与度。

**提到的链接**：

- [Easier, Better, Faster, Cuter](https://hazyresearch.stanford.edu/blog/2024-10-29-tk2)：未找到描述
- [ao/torchao/prototype/low_bit_optim at main · pytorch/ao](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload)：用于训练和推理的 PyTorch 原生 quantization 和 sparsity - pytorch/ao

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1300913767089045647) (129 messages🔥🔥):

> - `AGI 发展`
> - `模型效率`
> - `Quantization 技术`
> - `AI 工具与集成`
> - `用于 AI 的 Nvidia GPU`

- **关于 AGI 和模型的不同观点**：成员们对实现 **AGI** 的时间表和可行性表达了不同看法，争论 Google 等公司是否能在竞赛中跟上步伐。
  
  - 讨论中提到了对 Google 面临的监管挑战的担忧，以及对推动进步的新算法的乐观态度。
- **关于模型效率和尺寸的辩论**：讨论中出现了一种观点，即 **更大的模型** 并不总是更好，强调了通过 **quantization** 在不损失性能的情况下实现效率的潜力。
  
  - 成员们引用了 **Llama 3.0** 和 **Qwen** 模型进行对比，指出最近的 quantized 模型表现优于其体量更大的前代产品。
- **ChatGPT 和 API 开发咨询**：用户寻求关于开始 **API 开发** 和学习 **Python** 的建议，有人建议初学者利用 YouTube 获取基础技能。
  
  - 讨论强调了实际应用的重要性，例如在掌握基础知识后使用 **LeetCode** 来磨练编程技能。
- **关于用于 AI 工作的 Nvidia GPU 辩论**：有一场关于 **4070 Super GPU** 是否足以支撑本地 AI 项目的对话，观点倾向于对于更苛刻的任务，需要更高 VRAM 的选项。
  
  - 成员们共同强调了较小模型效率的不断提高，同时也承认了实惠型 GPU 之间仍然存在的差距。
- **AI 编程工具的演进**：参与者讨论了 **GitHub Copilot** 集成 **Claude** 的转变，指出这一变化如何增强了代码补全过程。
  
  - 评论强调了 AI 工具在节省时间方面的优势，特别是在自动化繁琐的编程任务（如填写函数参数）方面。

**提到的链接**：[Meta 发布 Llama3.2 1B/3B Quantized 模型：加速边缘推理，减少内存占用](https://aidisruptionpub.com/p/meta-releases-llama32-1b3b-quantized)：Meta 推出了 Llama3.2 quantized 模型，推理速度提高 2-4 倍，并减少了内存使用，针对移动设备进行了优化。

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1300950624854081567) (11 messages🔥):

> - `Open Source LLM Tool`
> - `Custom GPT Data Uploads`
> - `Performance of RAG`
> - `Cave Johnson AI`

- **寻找开源 LLM 工具**：一位成员询问是否有支持与 ChatGPT 进行分支对话（branching conversations）的开源 LLM 工具，允许响应从原始线程中分叉并能返回。
  
  - 未提供具体的工具名称，引发了对类似前端建议的征集。
- **自定义 GPT 文件组织**：成员们讨论了是为了获得最佳性能，应该向自定义 GPT 的知识库上传许多较小的文件还是单个大文件。
  
  - 一位成员指出**这取决于具体情况**，并以对科学论文进行 RAG 为例，这种场景下可能更倾向于使用较大的文件。
- **Cave Johnson AI 编译选择**：一位成员分享了他们创建 **Cave Johnson** AI 的方法，将 Portal Wiki 中的所有台词编译成一个单一的文本文件而不进行拆分。
  
  - 他们强调在这种情况下该方法更优，并认为 RAG 实现可以有效地处理这些数据。
- **聊天机器人数据组织的重要性**：在编写个人聊天机器人程序时，一位成员强调了保持数据组织有序且简洁的重要性，以避免在 API 调用中产生与无关 Token 使用相关的成本。
  
  - 使用 API 时，**无关数据的输入 Token 仍会产生费用**，这强化了精细数据管理的必要性。

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1300909797612195903) (2 messages):

> - `Stochasticity`
> - `Prompt Generation Tool`

- **关于 Stochasticity 的讨论**：一位成员提到了 **Stochasticity**，可能指其在 AI 行为或输出中的影响。
  
  - 未提供进一步的细节或背景，该话题保持开放性讨论。
- **寻求 Prompt 生成工具的访问权限**：一位用户询问如何在 Playground 中访问 **prompt generation tool**，以协助为特定任务生成有效的 Prompt。
  
  - 他们引用了 [关于 Prompt 生成的文档](https://platform.openai.com/docs/guides/prompt-generation)，表示不确定其在界面中的具体位置。

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1300909797612195903) (2 messages):

> - `Stochasticity`
> - `Prompt generation`
> - `Playground tools`

- **探索 Stochasticity**：一位成员简要提到了 **Stochasticity** 的概念，似乎在邀请进一步的讨论。
  
  - 该话题未提供更多额外细节。
- **寻找 Prompt 生成工具**：另一位用户询问如何在 OpenAI Playground 中访问 **prompt generation tool**，希望以此更好地为他们的任务定制 Prompt。
  
  - 他们链接到了 [Prompt 生成官方指南](https://platform.openai.com/docs/guides/prompt-generation) 作为参考。

---

### **Perplexity AI ▷ #**[**announcements**](https://discord.com/channels/1047197230748151888/1047204950763122820/1301216704432570430) (1 messages):

> - `Perplexity Supply`
> - `New Shipping Options`

- **Perplexity Supply 发布**：[Perplexity Supply](https://perplexity.supply) 推出了精心设计的必需品，旨在庆祝对知识的追求，目标受众是充满好奇心的人群。
  
  - 客户现在可以通过其各种产品系列购买能够激发对话和好奇心的优质商品。
- **现已支持全球发货**：Perplexity Supply 现在可发往包括**美国**、**澳大利亚**和**德国**在内的多个国家。
  
  - 有兴趣的人士可以通过 [此链接](https://perplexity.supply/sign-up) 注册，以获取未来新品发布和扩展至更多国家的更新信息。

**提到的链接**：

- [Perplexity Supply](https://perplexity.supply)：好奇心与品质的交汇点。我们的高端系列为好奇者提供精心设计的服饰。从重磅棉质基础款到刺绣单品，每一件商品都体现了我们对……的奉献。
- [Perplexity Supply: 即将到来](https://perplexity.supply/sign-up)：Perplexity Supply 旨在通过精心设计的产品探索时尚与理智之间的关系，激发对话并展示你对知识的无限追求。

---

### **Perplexity AI ▷ #**[**general**](https://discord.com/channels/1047197230748151888/1047649527299055688/1300901228074766347) (124 条消息🔥🔥):

> - `File Upload Issues` (文件上传问题)
> - `Pro Subscription Promo Codes` (Pro 订阅优惠券代码)
> - `Spaces and Collections Changes` (Spaces 和 Collections 的变更)
> - `NFL Widgets Introduction` (NFL 小组件介绍)
> - `Comparison of Perplexity and Consensus for Research` (Perplexity 与 Consensus 在研究方面的对比)

- **多位用户面临文件上传问题**：用户报告称 **file upload feature**（文件上传功能）出现故障，导致对话过程中文件残留，令人感到困扰。
  
  - *一位用户指出*，上传文件的处理一直不尽如人意，与其他平台相比表现较差。
- **领取 Pro 订阅优惠券代码时出现问题**：多位用户在领取 **GitHub Universe promo code**（免费一年 Pro 订阅代码）时遇到问题，收到“Invalid promo code”（无效优惠券代码）的提示。
  
  - *一位用户表示担心*，虽然该代码在网页端有效，但在 Android 应用上却失败了。
- **Spaces 和 Collections 功能的变更**：一位用户报告称 **Collections** 部分已重命名为 **Spaces**，并移动到了侧边栏，这引起了混淆。
  
  - *另一位用户确认*，他们在点击自己的 spaces 时遇到了黑屏 bug，证实了其他人的类似经历。
- **NFL 小组件上线**：社区获悉了新的 **NFL widgets**，可提供比赛摘要、统计数据和对比，更多体育功能即将推出。
  
  - *社交媒体上的一则开发公告* 暗示未来体育覆盖范围将扩展到 NFL 之外，如 NBA 和板球。
- **关于 Perplexity 与 Consensus 在医学研究方面的讨论**：一位用户询问 **Perplexity** 与 **Consensus** 的对比，认为 Consensus 可能更适合医学研究。
  
  - *讨论串显示*，一些用户已经对两者进行了测试，以寻求关于各自有效性的见解。

**提到的链接**：

- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1851315709533655116?s=61)：Perplexity 和 GitHub Copilot 用户均可通过 GitHub Marketplace 免费安装集成：https://github.com/marketplace/perplexityai
- [来自 Perplexity (@perplexity_ai) 的推文](https://x.com/perplexity_ai/status/1851341906271408469?s=46)：Perplexity Supply。周边商品，为好奇心而生。明天开售：http://perplexity.supply
- [来自 Pete Lada (@pklada) 的推文](https://x.com/pklada/status/1851411288133681368?s=61)：这是一个很有趣的开发过程——我们加入了很多小细节。Perplexity 的体育功能仍处于早期阶段，告诉我你接下来想看到什么！引用 Aravind Srinivas (@AravSr...

---

### **Perplexity AI ▷ #**[**sharing**](https://discord.com/channels/1047197230748151888/1054944216876331118/1300976168593395722) (9 条消息🔥):

> - `Earth's Temporary New Moon` (地球暂时的“新月亮”)
> - `Python Module Headers` (Python 模块头部)
> - `Search History Viewing` (查看搜索历史)
> - `Jang Gyuri` (张圭悧)
> - `Discord Registration` (Discord 注册)

- **地球获得了一颗暂时的“新月亮”**：一个来源在[此处](https://www.perplexity.ai/page/earth-s-temporary-new-moon-1a.EqH6ARBuNGyHUoOv37A)讨论了地球的**暂时的“新月亮”**。提到了关于其影响和可见性的关键细节。
  
  - *这一发现引发了关于临时天体的精彩讨论。*
- **纠正 Python 头部 (Header) 的误解**：一位用户纠正了关于 Python 模块头部的说法，指出头部应包含**模块的相对路径** [来源在此](https://www.perplexity.ai/search/pythontezuo-cheng-siteiruhuros-45UCvGCZRgeE.u_hNpISEA#0)。
  
  - *准确的模块引用对于成功编码至关重要。*
- **如何查看搜索历史**：一位成员询问如何在平台上**查看搜索历史**，参考了[此处讨论](https://www.perplexity.ai/search/how-to-see-history-of-searches-L8NRTsA.QsyVYbvmJecERg)的过程。
  
  - *获取自己的搜索历史可以简化研究过程。*
- **关于张圭悧 (Jang Gyuri) 的见解**：分享了一个关于**张圭悧**的链接，提供了关于她的工作和成就的有趣见解 [见此处](https://www.perplexity.ai/search/do-you-know-about-jang-gyuri-QOIxaSnhRnOJt9CAseJFPQ)。
  
  - *粉丝和研究人员都渴望了解更多关于她的贡献。*
- **关于 Discord 注册的问题**：有人提出了关于 **Discord 注册流程**的查询，相关讨论链接见[此处](https://www.perplexity.ai/search/why-is-the-discord-registratio-kfj9ciRNRbSj6AZSztMaYw)。
  
  - *澄清注册流程可能会使许多新用户受益。*

---

### **Perplexity AI ▷ #**[**pplx-api**](https://discord.com/channels/1047197230748151888/1161802929053909012/1301054880651415566) (5 messages):

> - `Differences in Playground and API results`（Playground 与 API 结果的差异）
> - `API for Perplexity Spaces`（Perplexity Spaces 的 API）
> - `Perplexity API usage for development`（用于开发的 Perplexity API 使用）

- **Playground vs API 结果差异**：一位用户询问为什么 [Playground](https://labs.perplexity.ai/) 和 API 的结果不同，并指出它们使用的是相同的模型。
  
  - 关于这两个产品之间差异的进一步说明尚未提供。
- **暂无 Perplexity Spaces 的 API**：针对一项咨询，官方表示目前没有 **Perplexity Spaces** 的 API。
  
  - 成员澄清 **Perplexity 网站**和 **API** 是作为两个独立的产品运行的。
- **考虑在项目中使用 Perplexity API**：一位用户表示有兴趣在他们的开发项目中使用 **Perplexity API** 来替代 **OpenAI API**。
  
  - 讨论并未就 Perplexity API 对此类用途的适用性给出具体的指导或建议。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1300903891029721111) (107 messages🔥🔥):

> - `Aider commands`（Aider 命令）
> - `Haiku 3.5 release status`（Haiku 3.5 发布状态）
> - `Qodo AI vs Cline`（Qodo AI 对比 Cline）
> - `Skyvern AI automation`（Skyvern AI 自动化）
> - `Gemini usage in development`（Gemini 在开发中的使用）

- **Aider 实现了 /save 和 /load 命令**：Aider 的一项新功能允许用户通过 `/save <fname>` 保存已添加的文件和只读文件，并通过 `/load <fname>` 加载命令，从而帮助轻松重建上下文。
  
  - 这些命令简化了管理代码上下文的过程，并允许进行批处理。
- **对 Haiku 3.5 发布的期待**：目前正在讨论 Haiku 3.5 的潜在发布，预计它可能很快面世，甚至可能就在明天。
  
  - 用户对其与之前版本相比可能带来的改进感到好奇。
- **Qodo AI 与 Cline 的比较**：虽然 Qodo 为 AI 代码生成提供了多种功能，但用户质疑它与 Cline 等模型相比有何独特之处，特别是在易用性和功能方面。
  
  - Qodo 的订阅起价为每月 19 美元，但缺乏模型选择以及内联补全（inline completion）属于付费功能是用户关注的焦点。
- **用于 AI 浏览器自动化的 Skyvern**：Skyvern 旨在利用 AI 自动化基于浏览器的任务流，为用户处理重复性任务提供无代码/低代码解决方案。
  
  - 该工具能适应网页变化，并具备通过简单命令执行复杂任务的功能。
- **Gemini 与其他工具的对比**：用户正在分享使用 Gemini 的经验，特别是与 Claude 和 Aider 等其他模型相比，它在数据库逻辑查询方面的有效性。
  
  - 尽管 Gemini 的性能会随上下文大小而波动，但对于实际编程需求，大家对其优势已达成普遍共识。

**提到的链接**：

- [Quality-first AI Coding Platform | Qodo (formerly Codium)](https://www.qodo.ai/)：Qodo（原 CodiumAI）为开发者提供质量优先的 AI 工具，以便直接在 IDE 和 Git 中编写、测试和评审代码。
- [Skyvern - Automate Browser-Based Workflows with AI](https://www.skyvern.com/)：Skyvern 帮助公司利用 LLM 和 Computer Vision 自动化基于浏览器的任务流，完全自动化手动流程并取代脆弱或不可靠的脚本。

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1300966084853370901) (12 messages🔥):

> - `Aider Configuration` (Aider 配置)
> - `DeepSeek Coder with Ollama` (结合 Ollama 使用 DeepSeek Coder)
> - `Input Caching Efficiency` (输入缓存效率)
> - `Consistent Code Generation Guidelines` (一致性代码生成指南)
> - `Connection Issues with Copilot` (与 Copilot 的连接问题)

- **针对闭源 LLM 的 Aider 配置**：一位用户询问如何配置 Aider，以便在 architect 模式下将 **Sonnet** 或 **O1** 等闭源 LLM 与 **O1 mini** 或 **Ollama** 结合使用，并寻求指导。
  
  - 另一位成员建议参考 `.aider.conf.yml` 文件和 Aider benchmarks 以进行最佳模型选择。
- **本地运行 DeepSeek Coder 的挑战**：一位用户在本地通过 **Ollama** 运行 **DeepSeek Coder** 时遇到困难，认为可能在设置中遗漏了某些步骤。
  
  - 社区成员提供了协助，以澄清用户的目标并排除故障。
- **输入缓存效果不一**：一位成员分享了他们的经验，认为输入缓存并未如预期般节省成本，并提供了代码库统计数据。
  
  - 他们提供了文件和代码的详细分类，指出了潜在的低效之处。
- **加载规范以实现一致的代码生成**：为了保持一致的代码生成，一位用户想知道是否可以在 Aider 中始终包含一个规范文件。
  
  - 得到的建议是创建一个 Markdown 文件并使用 `/read CONVENTIONS.md` 加载它，或者在 `.aider.conf.yml` 文件中进行配置。
- **Aider 与 Copilot 之间的连接问题**：一位用户报告在使用 Aider 连接 **Copilot** 时出现连接错误，尽管已经设置了必要的环境变量。
  
  - 讨论显示，由于公司政策，该用户只能使用 Copilot，从而引发了关于兼容性的询问。

 

**提到的链接**：[Specifying coding conventions](https://aider.chat/docs/usage/conventions.html)：告知 Aider 在处理代码时遵循你的编码规范。

 

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1300917236680360001) (3 messages):

> - `New Bash Tools from Claude Anthropic` (来自 Claude Anthropic 的新 Bash 工具)
> - `Integration of New Tools with Code Assistants` (新工具与代码助手的集成)

- **探索来自 Claude Anthropic 的新 Bash 工具**：一位成员对最近关于 **Claude Anthropic** 的新 **bash and editor tools** 及其潜在应用的讨论表示感兴趣。
  
  - 对话强调了在 **Aider** 等现有代码助手（code assistants）中实现这些工具的可能性，并强调了它们在提升用户体验方面的重要性。
- **对自动文件处理的期待**：另一位成员表达了对改进的渴望，建议**手动向聊天中添加文件**可能很快就会成为过去。
  
  - 这反映了对聊天功能中自动化和无缝集成的更广泛期望，特别是关于相关文件的处理。

 

**提到的链接**：[GitHub - disler/anthropic-computer-use-bash-and-files](https://github.com/disler/anthropic-computer-use-bash-and-files)：通过在 GitHub 上创建账号，为 disler/anthropic-computer-use-bash-and-files 的开发做出贡献。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1301204362772877333) (1 messages):

> - `Oauth issue` (OAuth 问题)
> - `API key creation` (API key 创建)

- **OAuth 认证今晨出现故障**：使用 [openrouter.ai/auth](https://openrouter.ai/auth) 创建 API key 的应用今晨受到 **OAuth 问题** 的影响。
  
  - 团队已**定位问题**，并确认修复程序将在公告发布后不久上线。
- **OAuth 中断的快速修复**：成员们注意到 **API key 创建** 的中断将很快得到解决，因为修复程序在报告后不久就得到了确认。
  
  - 这一迅速响应确保了依赖 OAuth 系统的应用停机时间降至最低。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**app-showcase**](https://discord.com/channels/1091220969173028894/1092850552192368710/1300916364285841488) (1 messages):

> - `Flexible Chat App for macOS`
> - `Alpha Testing`
> - `User Feedback`

- **为新聊天应用寻找 Alpha 测试人员**：一位开发者正在为其开发的 **macOS** 灵活聊天应用寻找 **alpha 测试人员**，并提供了[截图](https://imgur.com/a/HI5Py3A)链接。
  
  - 鼓励感兴趣的用户 **DM** 开发者以获取更多信息并参与测试阶段。
- **提供截图供预览**：该聊天应用的截图已发布在 [Imgur](https://imgur.com/a/HI5Py3A) 上，展示了其当前的设计和功能。
  
  - 开发者渴望收到潜在测试人员的反馈，以便在公开发布前完善应用。

**提及的链接**：[imgur.com](https://imgur.com/a/HI5Py3A)：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、热门的梗、有趣的 GIF、励志的故事、病毒式传播的视频等来振奋你的精神……

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1300912620160946239) (114 messages🔥🔥):

> - `OpenRouter Key Issues`
> - `Model Selection in OpenRouter`
> - `Haiku 3.5 Release`
> - `Prompt Caching for Models`
> - `OpenRouter Chat Functionality`

- **OpenRouter API Key 抓取风险关注**：一场关于 OpenRouter API Key 安全性的讨论展开，强调了 Key 可能会被他人抓取并滥用，特别是在 Sonnet 3.5 和 Mythomax 等付费代理设置中。
  
  - “仅仅因为你认为 Key 是安全的，并不意味着它就是安全的”是一条值得注意的评论，强调了对敏感信息保持警惕的必要性。
- **模型选择中的差异**：用户对 OpenRouter 自动选择特定模型表示困惑，特别是在使用 `openrouter/auto` 时，尽管期望选择 Claude 3.5 Sonnet 或 GPT-4o，系统却始终选择 Llama 3 70B Instruct。
  
  - 有人请求提供能够触发这些模型选择的 Prompt 示例，这表明需要更清晰地了解系统的行为。
- **对 Haiku 3.5 发布的热切期待**：社区急切等待 Haiku 3.5 的发布，有迹象表明它可能会在一天内发布，尽管该模型尚未在 GCP model garden 中正式上线。
  
  - GCP 的模型标识符 (slug) 被分享为 `claude-3-5-haiku@20241022`，但它仍处于白名单 (allow lists) 阶段，尚未普遍可用。
- **Prompt Caching 的利用**：成员们讨论了 Prompt Caching 在使用 OpenRouter 中的某些模型时降低成本的作用，并建议启用此类缓存以提高效率。
  
  - 讨论中澄清了 Prompt Caching 的运作方式，以及它在特定提供商处的潜在限制，强调了其对整体成本管理的益处。
- **OpenRouter 聊天保存功能**：用户询问了 OpenRouter 内部聊天的保存功能，确认聊天内容存储在浏览器的本地，如果管理不当可能会导致数据丢失。
  
  - 一个分享的链接强调了 OpenRouter 的这一特性，这似乎影响了试图重新访问早期讨论的用户。

**提及的链接**：

- [Prompt Caching | OpenRouter](https://openrouter.ai/docs/prompt-caching)：优化 LLM 成本高达 90%
- [Reddit - Dive into anything](https://www.reddit.com/r/ClaudeAI/comments/1gfuahg/cant_even_fathom_whats_in_the_36_sonnet_training)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/ClaudeAI/comments/1gflwc4/this_seems_to_be_a_new_feature_maybe_it_will_stop)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/ClaudeAI/comments/1gflwc4/this_seems_to_be_a_new_feature_maybe_it_will_stop/)：未找到描述
- [OpenRouter Status](https://status.openrouter.ai/)：OpenRouter 事件历史
- [Models | OpenRouter](https://openrouter.ai/models)：在 OpenRouter 上浏览模型
- [LLM Rankings | OpenRouter](https://openrouter.ai/rankings)：根据各应用的使用情况对语言模型进行排名和分析

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1300906527833718826) (5 messages):

> - `Integration Feature Access`

- **社区对 Integration Feature 访问权限的请求**：多位用户表达了对获取平台内 **integration feature** 访问权限的兴趣，强调了该功能对其需求的重要性。
  
  - 一位成员幽默地提到：*“我想再次申请 integration feature！”*，凸显了对该能力的渴望。
- **重复的 Integration 访问请求**：几位用户表示希望测试 **integration feature**，表明了对其功能的广泛好奇。
  
  - 诸如 *“你好，我想获得 integrations 的访问权限”* 之类的评论很常见，展示了对该功能的需求。

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1300897573426696292) (117 messages🔥🔥):

> - `GPU Comparisons`
> - `Training Models`
> - `Stable Diffusion Issues`
> - `Using Auto1111 vs Comfy UI`
> - `Recent Model Developments`

- **GPU 市场洞察**：成员们讨论了二手 **3090** 显卡的价格低于 **7900 XTX** 型号，并辩论了它们的相对优势，强调了预算考量。
  
  - *eBay 价格徘徊在 ~$690 左右*，让用户在性能指标和价格之间陷入抉择。
- **用于风格适配的训练模型**：一位成员询问了使用 **15-20 张图片** 的小数据集训练朋友艺术风格的最佳方法，并质疑使用模型还是 Lora/ti 更合适。
  
  - 有观点认为，基于风格选择使用 **Lora** 在创建一致性角色方面可能更有效。
- **解决 Stable Diffusion 图像问题**：有多项关于用户在使用 **Stable Diffusion** 时遇到**灰色图像**的咨询，并寻求故障排除支持。
  
  - 成员们建议探索各种 UI 选项并检查与 AMD GPU 的兼容性以提升性能。
- **UI 偏好：Auto1111 vs Comfy UI**：讨论倾向于 **Comfy UI**，因为它更易于使用，而一些用户仍然因为其自动化能力而偏好 **Auto1111**。
  
  - 成员们分享了经验，推荐 **SwarmUI**，因为它具有简单的安装过程和功能。
- **即将到来的 AI 模型进展**：社区推测 **SD 3.5** 是否会像 **SDXL** 一样受欢迎，引发了关于性能对比的讨论。
  
  - 有关于预期发布的全新 control nets 和模型更新的说明，这些更新在不断演进的领域中保持着相关性。

**提到的链接**：

- [no title found](https://rajeevlunkad.substack.com): 未找到描述
- [Anzhc's Face Segmentation (Prototype) | YOLOv8 | Adetailer model - Woman face (real only) | Stable Diffusion Other | Civitai](https://civitai.com/models/293448?modelVersionId=1007485)：新模型，女性和男性面部检测：基本上由 @girlsthatdontexist 赞助，我实际上不知道他们是否想被提及...
- [Create CONSISTENT CHARACTERS for your projects with FLUX! (ComfyUI Tutorial)](https://www.youtube.com/watch?v=MbQv8zoNEfY&list=PLqvJUJ2nkbont6HjW4nXKgRIsF4Aqh0tM&index=14)：通过这个深入的教程，解锁在 AI 艺术中创建完美角色的秘密！如果你喜欢我的作品，请考虑在 Patreon 上支持我：https://www.patr...
- [stabilityai (Stability AI)](https://huggingface.co/stabilityai)：未找到描述

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1300901685664940053) (67 条消息🔥🔥):

> - `微软对 OpenAI 的控制`
> - `AI 模型的延迟问题`
> - `自主 Twitter Agent 的部署`
> - `Flash Attention 与 CUDA 的兼容性`
> - `Hermes 3 与其他模型的性能对比`

- **微软降低对 OpenAI 依赖的风险**：讨论围绕微软降低对 OpenAI 依赖的策略展开，特别是如果 OpenAI 宣布实现 AGI，微软将拥有合同退出的权利并获得重新谈判的机会。
  
  - *“微软绝不会让这种事发生，”* 一位成员表示，对这种发布持怀疑态度。
- **对 AI 延迟的担忧**：人们对某模型报告的 20 秒延迟表示担忧，引发了对其运行硬件的调侃，比如像是*在土豆上运行*。
  
  - 成员们评论了与其他服务相比的性能差异，其中一人指出 Lambda 以 *1s 的延迟处理了 10 倍以上的请求*。
- **部署 Twitter Agent 的指南**：用户对部署自主 Twitter Agent 的指南表现出兴趣，确认存在此类项目的开源仓库。
  
  - 另一位成员鼓励通过 PR 参与贡献，为社区的开源工作出力。
- **在 A6000 上成功运行 Flash Attention**：一位成员分享了在 A6000 上运行 CUDA 12.4 和 PyTorch 2.5.0 环境下成功运行 Flash Attention 2.6.3 的经验，并指出尽管之前存在挑战，但这确实是可行的。
  
  - 他们提到手动构建解决了 pip 安装产生的符号链接问题。
- **AI 模型的性能对比**：成员们讨论了 Hermes 3 8B 与其他 10B 以下模型相比惊人的高性能，断言其质量可与 GPT-3.5 媲美。
  
  - 对话中还批评了像 **Mistral 7B** 这样的模型相比之下显得*令人遗憾*，强调了 Hermes 3 在 Tool Calling 方面的有效性。

**提到的链接**：

- [来自 undefined 的推文](https://vxtwitter.com/DataPlusEngine/status/1851625474327302288)：未找到描述
- [来自 Javi Lopez ⛩️ (@javilopen) 的推文](https://x.com/javilopen/status/1851361418857365974)：我让 AI 展示埃及金字塔是如何建造的，现在我确信我这辈子都会做噩梦了 🤯
- [来自 DataVoid (@DataPlusEngine) 的推文](https://x.com/DataPlusEngine/status/1851632986992632166)：未找到描述
- [fblgit/cybertron-v4-qw7B-MGS · Hugging Face](https://huggingface.co/fblgit/cybertron-v4-qw7B-MGS)：未找到描述

---

### **Nous Research AI ▷ #**[**ask-about-llms**](https://discord.com/channels/1053877538025386074/1154120232051408927/1300920839348289640) (12 messages🔥):

> - `西班牙语 Function Calling 数据集`
> - `Hermes 3 的有效性`
> - `基于 API 的 AI 的数据保留政策`
> - `Apple 的 Private Cloud Compute`
> - `对数据隐私的担忧`

- **对西班牙语 Function Calling 数据集的需求**：一位成员正积极尝试构建 **西班牙语** 的 function calling 数据集，但面临开源模型表现不佳的挑战；提供的一个示例网站强调了正在进行的从 **López Obrador** 晨间会议进行 **数据转换** 的实验。
  
  - 他们引用了从 **一千多个视频** 中处理的海量信息，旨在实现 **新闻相关性**，同时减少数据处理时间。
- **Hermes 3 的自定义有效性**：一位用户报告称，当配置了自定义 system prompt 时，**Hermes 3** 表现得特别有效，且对用户的准入门槛较低。
  
  - 这与 **Llama 3.2** 和 **Qwen** 等其他模型令人失望的表现形成了鲜明对比。
- **深入探讨 AI 数据保留**：人们对基于 API 的 AI 在服务条款（T&Cs）中高水平的数据收集表示担忧，并询问是否会有针对此类做法的抵制。
  
  - 几位参与者指出，许多用户忽略了暗示数据保留的细节，误以为“非训练条款”就意味着“不存储”。
- **Apple 的 Private Cloud Compute 计划**：分享的一篇 Apple 博客文章详细介绍了他们的 **Private Cloud Compute (PCC)** 计划，强调了通过设备端处理用户数据时 **隐私** 和安全的重要性。
  
  - 这表明 Apple 意识到了数据处理问题，尽管有人担心此类解决方案是否主要惠及 Mac 用户。
- **对数据机密性的担忧**：讨论显示，即使是良性的数据处理实践也可能导致偶然的数据泄露，特别是在服务器维护和调试期间的敏感信息。
  
  - 一位参与者强调，许多 AI 的条款通常包含关于记录 prompt 以进行 **滥用监控** 的免责声明，这进一步复杂化了对数据隐私的信任。

**提到的链接**：

- [Blog - Private Cloud Compute: A new frontier for AI privacy in the cloud - Apple Security Research](https://security.apple.com/blog/private-cloud-compute/): 云端安全私密的 AI 处理提出了一个巨大的新挑战。为了支持具有更大基础模型的 Apple Intelligence 的高级功能，我们创建了 Private Cloud Compute (PCC)...
- [Las Mañaneras - Las Mañaneras](https://mananeras.certexai.com/): 未找到描述

---

### **Nous Research AI ▷ #**[**interesting-links**](https://discord.com/channels/1053877538025386074/1132352574750728192/1300928981750644767) (8 messages🔥):

> - `Google 的 AI 生成代码`
> - `NotebookLM`
> - `代码指标`
> - `Continuum 应用`

- **Sundar Pichai 吹捧 AI 对 Google 编程的影响**：在财报电话会议上，**Sundar Pichai** 透露，根据 [Andrew Curran 的推文](https://x.com/AndrewCurran_/status/1851374530998256126)，Google 超过 **25% 的新代码** 是由 AI 生成的。
  
  - 这一统计数据引发了关于 AI 在编程实践中重要性的讨论。
- **NotebookLM 作为代码生成工具**：成员们讨论了 AI 的进步与 **NotebookLM** 有关，这是一个利用各种资源促进代码生成的工具。
  
  - *Mikebirdtech* 确认分享的信息源自 NotebookLM 的功能以及相关链接。
- **质疑代码生产力指标**：*Zachmayer* 评论了“统计代码行数不是衡量生产力的最佳标准”这一观点。
  
  - 他幽默地表示，如果 AI 能删除他 **25% 的代码**，他会感到更印象深刻。
- **Continuum 作为一个潜在平台**：*Felixultimaforeverromanempire* 建议关于 AI 生成代码的对话在 **Continuum** 平台上会很有价值。
  
  - 这表明了探索 AI 编程能力创新应用的兴趣。

 

**提到的链接**：[Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1851374530998256126): Sundar Pichai 在今天的财报电话会议上表示，目前 Google 超过 25% 的新代码是由 AI 生成的。

 

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1301217716983828541) (3 messages):

> - `Stocks`
> - `Meme coin simulation`
> - `Synthetic datasets`

- **股票讨论引发关注**：一名成员询问了关于股票的问题，在频道内引发了简短的讨论。
  
  - *哈哈，我看到了股市模拟* 是随后的轻松回应，展示了对该话题的参与。
- **Meme Coin 模拟正在进行中**：另一名成员提到，他们目前正在开发 *Meme coin 模拟*，用于生成 **合成数据集**。
  
  - 这一努力反映了使用趣味性货币概念进行创新金融建模的趋势。

 

---

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1300942680167026688) (16 messages🔥):

> - `Running multiple instances on a single GPU`
> - `Testing RAG with CSV data`
> - `Entity extraction and temperature settings`
> - `Submissions and rebuttals for COLING`
> - `Variability in harmful instructions across benchmarks`

- **探索在单张 GPU 上运行多个实例**：一名成员询问了如何使用 GPT-NeoX 在同一张 GPU 上运行多个小型 LLM 实例，以便通过更大的 Batch Size 填满 GPU 显存。
  
  - 其他人讨论了可行性，指出了实验并行训练方案的潜力，但提醒不要指望在同一张 GPU 上使用 DDP 能获得显著收益。
- **尝试在自然语言 CSV 上进行 RAG**：一名成员对将带有分隔符的原始 CSV 数据输入本地约 3B 的 LLM 进行 RAG 表示怀疑，并提到在测试中出现了案例编号混淆的问题。
  
  - 他们计划将 CSV 转换为 JSON 以获得更好的性能，这表明可能存在预处理方面的挑战。
- **实体提取设置调整**：在对 qwen2.5 进行测试阶段后，一名成员意识到在实体提取过程中 Temperature 被错误地设置为 0.8 而不是 0，从而影响了结果。
  
  - 他们正在尝试使用正确的 Temperature 设置重新进行提取，以改善结果。
- **COLING 投稿咨询**：一名成员询问是否有人向 COLING 投稿，并注意到针对评审的 Rebuttal 有 500 字的限制。
  
  - 他们寻求澄清这一限制是普遍适用的，还是仅针对他们的具体情况。
- **关于安全 Benchmark 中有害指令的讨论**：一名成员询问是否有论文讨论了由于组织政策不同，导致不同安全 Benchmark 在定义“有害指令”时存在的可变性。
  
  - 他们对 Benchmark 作者的价值观可能如何影响什么被视为“有害”表示了担忧。

 

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1300916558440173638) (44 条消息🔥):

> - `Modular Duality in Optimization` (优化中的模块化对偶性)
> - `Comparison of Optimization Papers` (优化论文对比)
> - `Training Diffusion Models` (训练 Diffusion Models)
> - `Limitations of Diffusion Models` (Diffusion Models 的局限性)
> - `Operator Norms in Neural Networks` (神经网络中的算子范数)

- **Modular Duality 理论的惊喜**：一位成员分享道，正如最近关于 **modular duality** 的论文中所讨论的，**maximal update parameterization** 和 **Shampoo** 等流行方法实际上是对线性层单一对偶映射（duality map）的局部近似。
  - 论文的第 4.1 节详细阐述了这种联系，强调了该理论在当代优化技术中的相关性。
- **对近期论文的批判性评价**：批评者对近期一篇优化理论论文的复杂性和原创性表示怀疑，认为其唯一显著的贡献可能是 **Newton-Schulz iteration**。
  - 一些人评论说，该论文可能使用了复杂的语言来排斥读者而非澄清概念，损害了其易读性。
- **关于微调 Diffusion Models 的对话**：成员们讨论了将 **Dino** 或 **CLiP ViTs** 作为 **diffusion transformers** 进行微调的策略，并对它们是否优于 **RePA** 表示不确定。
  - 提到的一种方法是保留前几层，同时随机初始化其余部分，在进行适配的同时保持标准的 **transformer** 块结构。
- **探讨 Diffusion Models 的局限性**：讨论涉及 **diffusion models** 面临的挑战和局限性，包括在训练时间和质量方面与其他生成模型（如 **GANs** 和 **autoregressive** 模型）的比较。
  - 确定的主要问题包括可控性、潜在空间操作（latent space manipulation）和表示学习（representation learning），并引发了关于这些问题如何影响其适用性的疑问。
- **算子范数与泛函分析**：成员们对 **operator norms** 表现出兴趣，并指出理解它们可以澄清与优化相关的泛函分析（functional analysis）的某些方面。
  - 讨论还包括关于神经网络中不同的张量角色如何证明使用独特的 **operator norms** 是合理的，从而简化优化过程。

**提到的链接**：

- [Jeremy Bernstein (@jxbz) 的推文](https://x.com/jxbz/status/1851328126652960796)：**modular duality** 理论的一个令人惊讶的方面是，**maximal update parameterization** 和 **Shampoo** 等流行方法表现为对线性层单一对偶映射的局部近似……
- [Jeremy Bernstein (@jxbz) 的推文](https://x.com/jxbz/status/1851328119539429487)：在过去的一个月里，我和我的合作者开发的方法被用于刷新 1.5B 规模 **LLMs** 的训练速度记录。我也想帮助科学进步得更快，所以现在准备好……
- [Deep Learning 中的 Modular Duality](https://arxiv.org/abs/2410.21265)：优化理论中的一个老观点认为，由于梯度是一个对偶向量，在将其映射到权重所在的原始空间之前，不能直接从权重中减去它。我们……
- [旧优化器，新范数：选集](https://arxiv.org/abs/2409.20325)：深度学习优化器通常是由凸理论和近似二阶理论混合驱动的。我们选择了三种此类方法——**Adam**、**Shampoo** 和 **Prodigy**——并认为每种方法都可以……

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1300953985418793072) (7 messages):

> - `sae_dashboard`
> - `AI paper on concept geometry`

- **关于 sae_dashboard 的后续工作**：一名成员表示他们尚未尝试运行 **sae_dashboard**，并称这将是其项目的下一部分。
  
  - 他们决定在改进之前先发布目前已有的用于分析 feature 的内容。
- **使用 sae_dashboard 进行以文本为中心的分析**：**sae_dashboard** 允许进行以文本为中心的分析，重点关注哪些 feature 会根据输入文本激活。
  
  - 讨论中阐明了这一点，强调了该工具检查 feature activation 的能力。
- **Tegmark 关于 Feature 结构的新 AI 论文**：一名成员分享了 [Tegmark 推文的链接](https://fxtwitter.com/tegmark/status/1851288315867041903?t=eB9Ft7hF9ocV9s-w3s-O1w&s=19)，宣布了一篇揭示 LLM 学习到的概念中存在**令人惊讶的几何结构**的 AI 论文。
  
  - 该论文讨论了被组织成**类脑“叶” (lobes)**、**语义晶体 (semantic crystals)** 的概念，并展示了一个比以前认为的**更精确**的分形概念云结构。

 

**提到的链接**：[来自 Max Tegmark (@tegmark) 的推文](https://fxtwitter.com/tegmark/status/1851288315867041903?t=eB9Ft7hF9ocV9s-w3s-O1w&s=19)：我们新的 AI 论文揭示了 LLM 学习到的概念中令人惊讶的几何结构：1) 它们形成了类脑的“叶”，2) 它们形成了比最初发现的要精确得多的“语义晶体”...

 

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1300926759235096640) (15 messages🔥):

> - `Freezing Embedding Layer`
> - `Multiple Choice Prompt Format`
> - `Winogrande Context Handling`
> - `Eval Harness API Issues`
> - `Answer Matching Heuristics`

- **我可以冻结 Embedding 层吗？**：一位用户询问了关于冻结 **Megatron 模型**的 **embedding 层**的配置选项，旨在仅训练 Transformer blocks。他们还询问在 **19M.yml** 等配置中是否排除了 embedding 和输出参数。
  
  - 目前没有直接回应，表明讨论仍在进行中。
- **澄清多选题提示词格式**：提供了针对多选题任务的预期提示词格式说明，结构为 `<doc_to_text><target_delimiter><doc_to_choice[i]>`。默认的 **target_delimiter** 是一个空格。
  
  - 样本的答案由产生最高 **logprob** 的选项决定。
- **Winogrande 的独特挑战**：讨论强调 **Winogrande** 的运行方式不同，它会翻转 context，而不是维持一致的 context 进行评估。成员们表示，通常情况下，**conditional loglikelihood** 是在稳定的 context 上计算的。
  
  - Winogrande 结构的独特性被指出是偏离标准做法的。
- **Eval Harness API 的问题**：一位用户在针对兼容 OpenAI 的 API 运行 **eval harness** 时遇到了挑战，并详细说明了发送的请求格式。建议的解决方案是使用 `--apply_chat_template` 来修正格式问题。
  
  - 该修复方案被证实有效，用户对此表示感谢。
- **解释评估结果**：一位用户质疑其评估结果中 **gsm8k** 的 strict-match 值为 **0.0000** 的含义，想知道这是否意味着完全失败。回复指出可能是该任务使用的答案匹配启发式算法（heuristic）出现了故障。
  
  - 建议包括检查生成日志并添加 few-shot 示例以更好地引导模型。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1300934299289255968) (16 条消息🔥):

> - `Elon Musk xAI 融资谈判`
> - `Cursor premium 讨论`
> - `Robonato 具身智能讨论`
> - `Creative Writing Arena 洞察`

- **Elon Musk 正在洽谈提高 xAI 估值**：根据 [WSJ](https://x.com/AndrewCurran_/status/1851310076709224564) 报道，Elon 正在协商新一轮融资，目标是将 xAI 的估值从 **240 亿美元** 提升至 **400 亿美元**。尽管有这些讨论，Elon 一直否认此前的融资传闻。
  
  - *xAI 让我也感到有点害怕*，一位成员分享道，反映了对该公司雄心勃勃的发展方向的看法。
- **Cursor premium 的感知价值**：一位成员思考是否有比 **Cursor premium** 更好的选择，声称它提供了几乎无限的访问权限来使用他们优先考虑的模型。这引发了关于替代方案的评论，包括使用公司的 Claude API key。
  
  - 另一位贡献者幽默地指出，*nato too OP*，强调了所建议方法的感知优势。
- **关于 Robonato 需求的讨论**：一位成员开玩笑说 *robonato 需要被具身化 (embodied)*，强调了对更多物理智能的渴望。他们进一步建议 Robonato 渴望强大的末端执行器 (end effectors) 来进行操作。
  
  - 一位成员构思了通过完全开发的 Robonato 可以实现的博客输出，展示了高级 AI 具身化的潜力。
- **来自 Chatbot Arena 的新洞察**：Chatbot Arena 引入了 [一个新类别](https://x.com/lmarena_ai/status/1851715029621706892) 创意写作 (Creative Writing)，表明关注点明显转向原创性和艺术表达。关键发现显示 **o1-Mini** 已跌出顶级模型行列，而 **Gemini 1.5 Pro** 和 **Flash 002** 则取得了进展。
  
  - 在这些变化中，**ChatGPT-4o-Latest** 继续以显著的增幅保持第一，而 *New Sonnet 3.5* 相比其前代产品也有所提升。

**提到的链接**：

- [来自 Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1851310076709224564)：WSJ 报道称 Elon 正在洽谈新一轮融资，以将 xAI 的估值从 240 亿美元提高到 400 亿美元。Elon 多次否认此前的传闻。
- [来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文](https://x.com/lmarena_ai/status/1851715029621706892)：🚨新 Chatbot Arena 类别：Creative Writing Arena！创意写作（约占投票的 15%）涉及原创性和艺术表达，通常与技术提示词不同。关键发现：- o1-Mini 跌出...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1301098005234974790) (54 条消息🔥):

> - `Claude 3 Tokenizer`
> - `AI2 新办公室`
> - `MacBook Pro 定价`
> - `AI2 尴尬视频`
> - `太平洋西北地区风景`

- **揭秘 Claude 3 Tokenizer**：最近的一篇 [文章](https://tokencontributions.substack.com/p/the-mystery-of-the-claude-3-tokenizer) 分享了关于 **Claude 3 tokenizer** 独特的封闭性质的见解，提到其可用信息有限。
  
  - *必须依赖付费服务* 而不是开放文档被认为是使用过程中的一个令人沮丧的方面。
- **AI2 将搬迁至水边的新办公室**：AI2 的新办公室定于明年 6 月启用，并承诺拥有优美的水景。
  
  - 成员们对这一变化表示热切期待，并提到了 **Pacific Northwest** (太平洋西北地区) 风景的迷人之处。
- **MacBook Pro 惊人的价格**：成员们讨论了高配 MacBook Pro 的天文数字价格，**128GB RAM + 4TB SSD** 的配置售价约为 **8000 欧元**。
  
  - 这种定价让人感到难以置信，大家还讨论了各国汇率和税收的影响。
- **AI2 新视频受到批评**：一位成员分享了一个来自 AI2 的让人尴尬 (cringe-inducing) 的 [YouTube 视频](https://www.youtube.com/watch?v=JSqNIz0uHxQ)，尽管制作质量奇特，但称赞其态度诚恳。
  
  - 其他人也对视频的亮度发表了看法，建议需要进行调整以获得更好的视觉效果。
- **住在风景优美的湖边**：一位成员对 AI2 新办公室的绝佳视野表示羡慕，并将其与自己客厅里看到的小湖美景相提并论。
  
  - 这引发了关于当地风景和太平洋西北地区吸引力的讨论，强调了这两种环境的魅力。

**提到的链接**：

- [The Mystery of the Claude 3 Tokenizer](https://tokencontributions.substack.com/p/the-mystery-of-the-claude-3-tokenizer)：第一部分
- [More than open](https://www.youtube.com/watch?v=JSqNIz0uHxQ)：AI2 相信开放的力量可以构建一个所有人都能接触到 AI 的未来。了解更多关于我们方法的信息：https://allenai.org/more-than-open

---

### **Interconnects (Nathan Lambert) ▷ #**[**memes**](https://discord.com/channels/1179127597926469703/1187551504995987576/1300967019335782454) (3 messages):

> - `420gunna 的统治`
> - `令人印象深刻的财务里程碑`
> - `关于时效性的评论`

- **420gunna 被称为“国王”**：一位成员宣布 **420gunna** 是“国王”，暗示其在讨论中具有显赫地位。
  
  - 另一位成员幽默地评价了这个头衔，暗示这一说法带有轻松的调侃意味。
- **短短 45 分钟内达到惊人的 450 亿！**：一位成员评论称在 **45 分钟内实现了 450 亿**，并配以大笑的表情，指代一个非凡的财务里程碑。
  
  - 这一说法引发了大家的兴趣，突显了该数字夸张的性质。
- **关于入场太晚的评论**：一位成员对某人参与的时机进行了调侃，暗示他们来得“有点太晚了”。
  
  - 这种俏皮的批评为正在进行的互动增添了幽默感，取笑了迟到的行为。

 

---

### **Interconnects (Nathan Lambert) ▷ #**[**posts**](https://discord.com/channels/1179127597926469703/1228051082631188530/1301020601502138380) (8 messages🔥):

> - `语音旁白反馈`
> - `邮件失误`

- **语音旁白增强个人文章体验**：一位成员建议，**语音旁白（voiceover）**是阅读明天**个人文章**更好的互动方式。
  
  - 他们提到自己非常*中意*大部分的**语音旁白内容**。
- **邮件重复发送的失误**：一位成员幽默地提到了不小心发出的**重复邮件**。
  
  - “人们知道直接删除然后继续就行，”暗示用户通常对这类邮件失误持宽容态度。
- **邮件通信的狂野本质**：一位成员将邮件描述为一种**狂野的媒介**，暗示其具有不可预测性。
  
  - 这反映了关于使用邮件进行沟通时各种奇特现象的普遍看法。

 

---

### **Notebook LM Discord ▷ #**[**announcements**](https://discord.com/channels/1124402182171672732/1182376564525113484/1300950734749040725) (1 messages):

> - `NotebookLM 可用性研究`
> - `Audio Overviews 反馈`
> - `参与者奖励`
> - `远程交流机会`

- **NotebookLM 可用性研究报名**：NotebookLM UXR 正在邀请用户分享他们如何创造性地使用 **NotebookLM**，特别是在可用性研究中关于 **Audio Overviews** 的使用。
  
  - 参与者将被安排参加 **30 分钟** 的远程会议，入选者将获得价值 **50 美元** 的等值礼品。
- **通过远程聊天进行互动**：UXR 团队将通过 **Google Meet** 进行远程聊天，以深入了解用户使用 **NotebookLM** 的方式。
  
  - 有兴趣的用户应填写表格，以便在这一协作计划中获得入选机会。
- **分享研究的关键细节**：参与者需要具备**高速互联网连接**、活跃的 Gmail 账号以及功能正常的视听设备。
  
  - 该研究将在 **2024** 年底前的每个**周五**远程进行。

 

**提到的链接**：[参与即将举行的 Google UXR 研究！](https://forms.gle/QVJTJXzaQKzWUPr98)：您好，我正通过一份简短的问卷与您联系，以核实您参加即将举行的 Google 可用性研究的资格。这次研究是一个为您提供反馈的机会...

 

---

### **Notebook LM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1300897860828659814) (38 messages🔥):

> - `Simli 实时化身`
> - `用于播客视频的 Pictory`
> - `语音分离技术`
> - `NotebookLM 播客功能`
> - `Hedra 角色生成`

- **Simli 化身增强播客**：一位成员展示了 **Simli** 如何通过使用 .wav 文件的 diarization 来同步音频片段，从而在播客上叠加实时化身。
  
  - 这一概念验证展示了未来版本中功能集成的潜力。
- **使用 Pictory 制作视频播客**：一位用户询问如何使用 **Pictory** 将播客转换为视频，并表示有兴趣在视频中添加演讲者的面部。
  
  - 另一位成员建议 **Hedra** 可以通过上传分割的音频轨道来实现角色可视化。
- **语音分离变得简单**：对于播客音频中的语音分离，成员们推荐了 **Descript**，并讨论了在转录文本中隔离单个演讲者的有效方法。
  
  - 选项包括从转录文本中裁剪一个声音，或使用多个音轨来优化音频质量。
- **探索 NotebookLM 播客功能**：用户分享了使用 **NotebookLM** 创建引人入胜的播客的经验，强调了从单个单词作为素材开始的简便性。
  
  - 成员们讨论了通过音频和反馈整合的各个阶段，对 **LLMs** 进行迭代过程和测试极限。
- **用于角色生成的 Hedra**：一位成员指出，可以使用 **Hedra** 生成具有表现力的角色，该平台补充了 **Pictory** 在基于角色的视频内容方面的不足。
  
  - 讨论内容包括该平台如何通过 AI 驱动的角色创建和表达来增强叙事。

**提到的链接**：

- [Simli](https://www.simli.com/demo)：未找到描述
- [EverythingSTEVENANDJAMIERICE.wav](https://drive.google.com/file/d/1ILawC-xFr9R2Oh5n9KqDk5gw-m3IQapM/view?usp=drivesdk)：未找到描述
- [Hedra](https://www.hedra.com/)：为每个人提供的视频创作。
- [no title found](https://www.amazon.com/gp/help/customer/display.html?nodeId=GPC35Y68PEZYG3ED)：未找到描述
- [CYBERSECURITY Experts Reveal The Open Jobs Crisis](https://youtu.be/w0tsFTvwfQM)：在这段视频中，网络安全专家揭示了行业内的就业危机。了解对网络安全专业人员的需求以及你如何开始...
- [Notebook LM - Create Video Podcasts in Minutes! With Pictory AI](https://youtu.be/1jgpsGDUXW4)：嘿，大家好！🚀 想使用 AI 将你的内容转化为引人入胜的视频播客吗？在本教程中，我将指导你完成创建视频播客的简单步骤...
- [Impact of Decisions](https://www.notebooklm.video/c/player/2093e575-30b2-4e22-b758-03d5c233be2d)：在这段视频中，我们深入探讨了每一个决定为何都很重要，以及你如何提高做出明智决定的能力。

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1300908030228955136) (40 messages🔥):

> - `Podcast Generation Limitations` (播客生成限制)
> - `Issues with Language Switching` (语言切换问题)
> - `Notebook Features Request` (Notebook 功能请求)
> - `Audio Segmentation Techniques` (音频分段技术)
> - `Interruption Issues in Podcasts` (播客中的打断问题)

- **Podcast Generation Limitations**: 几位用户报告了在生成**西班牙语**播客时遇到困难，尽管之前曾成功生成，这引发了关于该功能何时可能恢复的疑问。
  
  - *一位用户提到*，“它在头两天效果很好。然后，就停止生成西班牙语了。”
- **Issues with Language Switching**: 一位用户分享了一个有趣的实验，模型利用英语源材料成功生成了**芬兰语**播客，展示了其多语言能力。
  
  - 当被问及这种可能性时，他们解释道，\*“我包含了
- **Notebook Features Request**: 有人请求增加 Notebook 中 **Customize** 提示词的长度，因为目前 **500** 的限制被认为有点短。
  
  - 另一位用户建议通过创建一个名为 `Instructions` 的来源作为额外的自定义规避方案。
- **Audio Segmentation Techniques**: 用户一直在探索使用 **Descript** 等工具划分播客剧集的方法，并指出在 **Deep Dive** 期间观察到了自动分段。
  
  - *一位用户评论道*，“我注意到有时 Deep Dive 会自动分成几集。”
- **Interruption Issues in Podcasts**: 有人对播客中打断现象增多表示担忧，这导致了对话流的不连贯。
  
  - *一位用户评论道*，“还有人觉得播客主持人互相打断的情况变多了吗？”

**提到的链接**:

- [How Steve Jobs Foresaw AI’s Power to Preserve Great Minds Like Aristotle & Plato in 1983 🤖🧠](https://youtube.com/shorts/By566GHmA7g?si=sYnaJVIveq5tWkXu): 1983 年，史蒂夫·乔布斯分享了一个富有远见的想法，感觉非常像今天的 AI，比如 ChatGPT 🤖✨。在他于国际设计大会上的演讲中……
- [GitHub - souzatharsis/podcastfy: An Open Source alternative to NotebookLM's podcast feature: Transforming Multimodal Content into Captivating Multilingual Audio Conversations with GenAI](https://www.podcastfy.ai): NotebookLM 播客功能的开源替代方案：利用 GenAI 将多模态内容转化为迷人的多语言音频对话 - souzatharsis/podcastfy

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1301044142234800140) (4 messages):

> - `AI Impact on Jobs` (AI 对就业的影响)
> - `Interest in Deep Tech` (对 Deep Tech 的兴趣)
> - `Advertisement Messages` (广告信息)

- **AI 挑战传统的软件工程职位**: 一位成员指出，**AI** 正在越来越多地取代**普通软件工程师的工作**，这表明就业形势正在发生变化。
  
  - 他们对这一趋势对科技行业就业机会的影响表示担忧。
- **参与 Deep Tech 的愿望**: 一位成员表达了对参与 **deep tech** 创新的强烈兴趣，信号表明其希望在该领域进行更深层次的参与。
  
  - 这反映了人们对先进技术在表面应用之外的潜力的日益好奇。
- **请求删除广告**: 一位成员请求从频道中删除某些**广告信息**，表示对非相关内容感到困扰。
  
  - 这强调了在社区讨论中保持专注和相关性的必要性。

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1300910796423041095) (16 条消息🔥):

> - `FSDP2 API 更新`
> - `在 Rust 中使用 PyTorch 进行内存分析`
> - `torchao 优化器与 SR 支持`
> - `使用 C++ 进行 CUDA Kernel 调试`
> - `Triton Kernels 中的早期剪枝配置`

- **FSDP2 API 弃用通知**：一位用户强调了关于 `torch.distributed._composable.fully_shard` 弃用的 **FutureWarning**，敦促切换到 FSDP，详情可见 [此 issue](https://github.com/pytorch/pytorch/issues/114299)。
  
  - 这引发了关于在 **torch titan 论文** 发表后，**fully_shard API** 是否仍然具有持续相关性的疑问。
- **Rust tch.rs 应用内存分析**：一位成员就使用 **torchscript** 的 **Rust** 应用进行内存分析（memory profiling）以识别潜在内存泄漏问题寻求建议。
  
  - 他们特别感兴趣的是在涉及自定义 **CUDA kernels** 时如何调试潜在问题。
- **torchao 优化器与 SR 支持**：围绕 **torchao 优化器** 展开了讨论，特别是它们是否支持 SR（随机舍入），一位用户提到最近已添加该支持。
  
  - 对话暗示将针对 **bf16 momentum** 寻求 **SR** 支持，尽管有提到之前的测试并未显示出显著改进。
- **在 C++ 中编译 CUDA Kernels**：一位成员表示，为了在使用 `cpp_extension.load` 时更方便地调试，直接在 **C++** 中编译 **CUDA kernels** 存在挑战。
  
  - 得到的建议是设置 `CUDACXX` 环境变量，以选择未与系统路径链接的特定 **CUDA 版本**。
- **Triton kernels 中的早期剪枝配置**：由于 `torch.compile` 缺乏 **prune_configs_by** 支持，一位用户询问了在 **PyTorch/Inductor** 中早期剪枝（early prune）配置的方法。
  
  - 这表明需要替代方法来有效处理 **Triton kernel** 配置。

**提到的链接**：

- [max_autotune_vs_reduce_overhead.py](https://gist.github.com/mobicham/fa4ea2e9d836894d1a67821717aef047)：GitHub Gist：即时分享代码、笔记和代码片段。
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues/114299)：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - Issues · pytorch/pytorch

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1301097319231651840) (12 条消息🔥):

> - `int8 vs fp16 tensor cores`
> - `计算密集型任务的 GPU 选择`
> - `云端 GPU vs 本地 GPU 部署`
> - `张量操作中的性能开销`

- **Int8 Tensor Cores 变慢了？**：讨论者提出了为什么在特定的 GEMM 形状下，**int8 tensor core** 可能比 **fp16 tensor core** 更慢的问题，一些人认为是潜在的 **量化开销 (quantization overhead)** 导致的。
  
  - *量化开销*以及存储结果时不同的输出数据类型可能会显著影响性能对比。
- **为黑客松选择计算设置**：一位参与者讨论了黑客松的最佳设置，权衡了 **M4 Mac mini**、带 **3090 的 PC** 或 **NVIDIA Jetson AGX Orin** 等选项。
  
  - 建议倾向于使用云端 GPU 以获得灵活性，同时也担心在 Apple 硬件上部署的学习曲线。
- **本地 vs 云端 GPU 部署**：分享了关于使用云端 GPU 与拥有本地硬件相比的**开销**和不便之处，强调了快速访问资源的优势。
  
  - 虽然云端 GPU 提供了灵活性，但在产生突发计算需求时，重新连接和潜在的延迟令人沮丧。
- **int8 的高效形状**：有人指出，在大多数硬件配置中，使用 **16 的倍数的 int8 形状**往往更高效。
  
  - 其他人肯定了配置中普遍存在的性能开销，并建议分享代码以进行更深入的分析。

 

---

### **GPU MODE ▷ #**[**torchao**](https://discord.com/channels/1189498204333543425/1205223658021458100/) (1 条消息):

starsupernova: 会去看看！！

---

### **GPU MODE ▷ #**[**rocm**](https://discord.com/channels/1189498204333543425/1233704710389764236/1300982904347693056) (15 条消息🔥):

> - `HIP memory error 问题`
> - `GFX1100 争用问题`
> - `ROCM 版本疑虑`

- **导致系统锁死的 HIP Memory Error**：一位成员报告了一个 **HIP memory error 问题**，该问题会导致机器锁死，在使用 **torch 2.5** 和 **ROCM 6.2** 运行时，journalctl 中会出现数百行垃圾信息。
  
  - 这个问题似乎在 **torch 任务** 与另一个分配显存的进程同时运行时出现，据报告 **headless 模式** 可以避免这些问题。
- **GFX1100 上的争用问题**：讨论了关于 **GFX1100** 的争用问题，指出在执行 **torch** 任务期间，加载一个简单的网页偶尔会导致阻塞行为。
  
  - 一位成员提到，他们的设备经常同时用于 **torch** 和桌面任务，因此很容易受到争用影响。
- **针对 GFX1100 的 ROCM 更新**：对 **ROCM** changelog 中的更新提出了质疑，强调针对 GFX1100 的修复仅在早期版本中提到，特别是从 **6.1** 开始。
  
  - 一位使用 **ROCM 6.2.1** 的成员提到，问题自 **6.0** 以来一直存在，表明这是一个长期存在的问题，且在随后的更新中并未得到解决。

---

### **GPU MODE ▷ #**[**sparsity-pruning**](https://discord.com/channels/1189498204333543425/1247663759434977453/1301178233164795987) (7 条消息):

> - `可变大小块剪枝`
> - `结构化剪枝方法`
> - `非结构化稀疏方法`
> - `Lottery Ticket Hypothesis`
> - `结构化稀疏 winning tickets`

- **探索可变大小块剪枝**：一位成员提出了关于现有工作的问题，即如何将权重矩阵剪枝为具有可变大小块（而非固定大小）的稀疏矩阵。
  
  - 他们询问是否存在能够实现这一点且仍能针对 GPU 性能进行优化的结构化剪枝方法。
- **结构化剪枝需要固定块**：有人指出结构化剪枝通常产生固定大小的块，从而引发了对可变大小块替代方案的查询。
  
  - 尽管如此，一条回复指出结构化剪枝方法通常对 GPU 更友好。
- **非结构化稀疏具有性能提升**：成员们讨论了虽然非结构化稀疏方法可能会带来更好的性能，但要从中获得实际收益可能具有挑战性。
  
  - 非结构化方法通常会导致不规则的稀疏模式，难以在硬件上加速。
- **Lottery Ticket Hypothesis 与结构化剪枝**：引用了一篇讨论 Lottery Ticket Hypothesis 的论文，指出传统剪枝的子网络往往具有非结构化稀疏性，这使得 GPU 加速变得复杂。
  
  - 该论文建议后处理技术可以帮助有效地找到结构化稀疏的 winning tickets，这标志着一个积极的进展。
- **对剪枝技术的不同看法**：虽然剪枝方法多种多样（包括结构化技术），但一位成员对某些尝试的质量发表了评论，称其中一个“有点糟糕”。
  
  - 他们还引用了另一篇展示了有希望结果的论文，尽管对所使用的分类器表示保留。

**提到的链接**：[Coarsening the Granularity: Towards Structurally Sparse Lottery Tickets](https://arxiv.org/abs/2202.04736)：Lottery Ticket Hypothesis (LTH) 表明，稠密模型包含高度稀疏的子网络（即 winning tickets），这些子网络可以独立训练以达到完整的准确度。尽管有许多令人兴奋的……

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1300956838287773768) (7 messages):

> - `ThunderKittens talk schedule`
> - `Livestream on CUDA and ThunderKittens`
> - `TK vs Triton and CUTLASS`
> - `TK library approach`
> - `Mamba-2 kernel complexity`

- **ThunderKittens 演讲即将举行**：一位用户提到计划很快安排一次演讲，以讨论关于 **ThunderKittens** 的功能和反馈。
  
  - 他们感谢了另一位成员的安排，营造了积极参与的社区氛围。
- **CUDA 直播取得成功**：一场名为 **'CUDA + ThunderKittens, but increasingly drunk'** 的直播展示了数小时关于 CUDA 和调试 kernel 的内容，观看地址在 [这里](https://www.youtube.com/watch?v=xcpEl0cGCC4)。
  
  - 观众被告知在会议期间屏幕共享出现了一个小插曲，但可以跳过该部分。
- **TK 作为库对比基于编译器的方法**：成员们讨论了 **ThunderKittens** 旨在提供比 Cutlass 等库更高层级的抽象，同时比 **Triton** 等基于编译器的解决方案更易于使用。
  
  - 对话强调，虽然编译器可能更强大，但库在易用性和灵活性方面更受青睐。
- **ThunderKittens 的可扩展性**：讨论解释了 **ThunderKittens** 如何被设计用于处理复杂任务，同时允许用户根据需要编写自定义 CUDA 代码。
  
  - 提供了一个例子，其中 **Mamba-2 kernel** 利用自定义 CUDA 来满足特定需求，展示了该平台的灵活性。
- **重新审视 Kernel 以提高精度**：演示的 **H100 kernel** 示例完全在 **ThunderKittens** 原语（primitives）内运行，强调了该库的内置功能。
  
  - 相比之下，**Mamba-2 kernel** 利用自定义技术来管理复杂的草作，这些操作不容易表达为简单的 tensor 操作。

 

**提到的链接**：[CUDA + ThunderKittens, but increasingly drunk.](https://www.youtube.com/watch?v=xcpEl0cGCC4)：我的朋友 Quinn (x.com/qamcintyre) 让我教他 CUDA 和 ThunderKittens，我同意了，条件是我们把它拍下来，这样我就可以让新学生……

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1301093839628668999) (27 messages🔥):

> - `Llama 3.2 QLoRA Training`
> - `Quantization Issues`
> - `Activation Checkpointing Impact`
> - `Adapter Weights Saving Time`
> - `Checkpointing Performance Discrepancies`

- **Llama 3.2 QLoRA 训练流程**：一位用户详细介绍了他们复制 **Llama 3.2 1B QLoRA** 训练的过程，指出 QAT 取得了成功，但在使用 **Int8DynActInt4WeightQuantizer** 应用量化时出现了生成内容不连贯的问题。
  
  - 他们认为 **QAT** 可能没有充分调整权重，导致量化过程中出现问题。
- **Checkpoint 的量化挑战**：讨论了量化后生成文本不连贯的问题，并分享了关于 **QAT 训练** 和量化所用配置的见解。
  
  - 一位用户分享的代码片段显示，在使用特定版本的 **torchtune** 和 **torchao** 时，量化层没有被正确准备。
- **Activation Checkpointing 减慢保存速度**：一位用户质疑为什么 **activation checkpointing** 默认设置为 false，并指出这使得保存 checkpoint 的速度显著变慢，尤其是对于 **Llama 3.2**。
  
  - 另一位参与者澄清说，对于较小的模型，checkpointing 并不是必需的，因为它会产生额外的计算开销。
- **仅保存 Adapter 权重以提升速度**：用户确认仅保存 adapter 权重（`save_adapter_weights_only=True`）会导致 checkpoint 保存时间显著缩短，减少到 1 秒。
  
  - 相比之下，保存完整模型要慢得多，保存权重时 **Llama 3.2 1B** 需要 270 秒，而 **Llama 3.1 8B** 需要 30 秒。
- **Checkpointing 的性能差异**：关于 **Llama 3.1** 和 **Llama 3.2** 之间保存时间的差异产生了困惑，观察到后者的 adapter 权重更大。
  
  - 最终，用户发现保存配置极大地影响了性能，并根据模型大小和 adapter 参数得出了意想不到的结果。

 

**提到的链接**：[improve resume from checkpoint · Issue #1551 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/1551)：当前从 checkpoint 恢复的体验可以改进。一些潜在的方法：良好的默认值：从 checkpoint 恢复应该默认使用最后保存的 checkpoint，这样用户就可以……

 

---

### **Torchtune ▷ #**[**dev**](https://discord.com/channels/1216353675241590815/1236040539409879170/1300942946685943858) (20 messages🔥):

> - `kv cache implementation`
> - `dynamic cache resizing`
> - `multi-query attention`
> - `PyTorch 2.5 enhancements`

- **修订 kv Cache 创建策略**：目前，`kv cache` 是使用 **num_heads** 而非 **num_kv_heads** 构建的，这可能导致推理期间不必要的内存占用。提议的更改将确保 `kv cache` 使用 **num_kv_heads** 进行初始化，并在 **expand** 步骤之前移动，从而节省内存。
  
  - 此外，在推理过程中，大家一致认为在 `kv cache` 中存储扩展后的 Tensor 维度副本是冗余的，应当进行优化，这激发了实现这些更改的热情。
- **动态调整大小以提升效率**：实现 `kv cache` 动态调整大小功能具有潜力，允许在特定条件下进行分配。它将根据实际需求而非预定义的最高长度进行重新分配，从而更有效地调整内存使用。
  
  - 该策略可以迎合生成过程持续到满足特定停止条件为止的常见用例，减少空间浪费并提高性能。
- **Multi-query Attention 作为存储解决方案**：讨论指出，实现 **multi-query** attention 的主要原因是为了减少 `kv-cache` 存储。一位成员指出，**PyTorch 2.5** 中的 **grouped query attention** 等特性减轻了手动进行 kv 扩展的需求，证实了这一观点。
  
  - 成员们表示，**flex_attention** 等库中的额外支持与 PyTorch 的进步并驾齐驱，增强了运行效率。
- **关于 PyTorch 版本兼容性的担忧**：有人担心 **enable_gqa** 仅在 **PyTorch 2.5** 或更高版本中可用，因为在不同版本之间保持逻辑一致性可能会使代码管理复杂化。**attention_utils.py** 中概述的方法旨在确保无论具体的 **attention** 实现如何，逻辑都能保持一致。
  
  - 成员们表达了简化版本管理以保持代码清晰的愿望，强调在未来的实现中需要采取务实的方法。
- **代码贡献协作**：一位成员对参与实现表示了极大的热情，并重申愿意尽快提交 Pull Request。其他人鼓励协作，认可了大家对增强项目功能的共同期待。

**提到的链接**：

- [torch.nn.functional.scaled_dot_product_attention — PyTorch 2.5 documentation](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)：未找到描述
- [torchtune/torchtune/modules/attention.py at main · pytorch/torchtune](https://github.com/pytorch/torchtune/blob/main/torchtune/modules/attention.py#L282))：PyTorch 原生微调库。通过在 GitHub 上创建账户为 pytorch/torchtune 的开发做贡献。
- [llama-models/models/llama3/reference_impl/model.py at main · meta-llama/llama-models](https://github.com/meta-llama/llama-models/blob/main/models/llama3/reference_impl/model.py#L185)~~.)：旨在与 Llama 模型配合使用的工具。通过在 GitHub 上创建账户为 meta-llama/llama-models 的开发做贡献。
- [Initialize kv cache w/num_kv_heads instead of num_heads · Issue #38 · pytorch/torchtune](https://github.com/pytorch/torchtune/issues/38)：这将为 GQA / MQA 节省内存，但需要对 attention forward pass 进行一些重构。

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1301167221908701184) (4 messages):

> - `SOAP Optimizer`
> - `Account issues`

- **Nikhil Vyas 博士讨论 SOAP 优化器**：我们将邀请哈佛大学博士后 **Nikhil Vyas 博士** 来分享 **SOAP Optimizer**。
  
  - 欢迎在 [Discord Event](https://discord.com/events/954421988141711382/1293256892834910208) 上收听。
- **账户问题的直接支持**：对于任何**账户或服务相关问题**，请直接发送电子邮件至 [**support@cohere.com**](mailto:support@cohere.com) 以获取帮助。
  
  - 一位成员表达了他们渴望帮助有需要的人。

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1300962010791677952) (17 messages🔥):

> - `Cohere Command R 模型性能`
> - `速率限制问题`
> - `寻求支持协助`
> - `聚焦企业级用例`
> - `预算软件模型应用`

- **Cohere Command R 模型在 AI 检测方面表现不佳**：一位用户表达了挫败感，尽管是付费用户并尝试了多种提示词，**Command R 模型**生成的文本始终具有 **90-95% 的 AI 可检测率**。
  
  - *创造力是 AI 固有的*，因为所有生成的文本都是基于其训练的人工策划数据中的采样分布。
- **处理速率限制错误**：一位用户报告在使用生产环境 API key 时收到 **429 Too Many Requests** 错误，并询问如何申请提高速率限制。
  
  - 另一位成员建议他们发送邮件至 [support@cohere.com](mailto:support@cohere.com) 并附上错误截图以寻求帮助。
- **跟进速率限制提升的支持请求**：在联系支持部门后，该用户确认已发送邮件，并对提供的指导表示感谢。
  
  - 团队确认已收到邮件，并保证同事正在迅速处理该问题。
- **会议论文发表咨询**：一位成员询问 **Cohere 社区**的论文通常发表在哪些会议上。
  
  - 对话中还提到了关于邮件查询的明确时间表或回复。
- **专注于企业级应用**：讨论表明，**Cohere 模型**的核心意图并非生成类人文本，而是为了满足**企业级用例**。
  
  - 成员们强调，该应用主要服务于业务需求，而非休闲或角色扮演场景。

---

### **Cohere ▷ #**[**api-discussions**](https://discord.com/channels/954421988141711382/1168578329423642786/1300953107676659843) (15 messages🔥):

> - `Embed V3 对比`
> - `结构化输出增强`
> - `模型类型显示问题`
> - `微调模型训练错误`
> - `游戏化创意`

- **Embed V3 与 ColPali 架构及 JINA CLIP 的对比**：一位成员询问 **Embed V3** 与 **ColPali** 架构及 **JINA CLIP embeddings** 的对比情况，并认为与旧版 **CLIP 模型**的对比可能已经过时。
  
  - 讨论暗示了 Embedding 架构正朝着超越传统 biencoder 方法的方向演进。
- **JSON 结构化输出增强 Embed V3**：一位成员提出，添加 **JSON 结构化输出数据集**可以增强 **Embed V3** 的搜索能力，并询问其与 **ColPali 多模态架构**的区别。
  
  - 这表明人们对于将结构化输出与现有模型集成以提升功能的兴趣日益浓厚。
- **模型类型显示问题**：一位成员报告称，无论使用哪种模型，**[Model Type]** 始终显示为 **[default]**，并表示需要显示实际的模型名称。
  
  - 另一位成员澄清说，**default 类型**表示其是否为 Fine-tuning 模型，这表明可能存在误解。
- **微调模型训练问题**：一位用户报告了在训练 Fine-tuning 模型时遇到的错误，称其收到消息说支持人员会联系并提供帮助。
  
  - 这引发了对微调过程中可能出现的间歇性问题的担忧，因为该用户前一天还能成功创建模型。
- **Embed V3 的游戏化创意**：一位成员集思广益，讨论了在 Embed V3 中集成结合**图像和文本**组件的想法。
  
  - 还有关于利用等级或参考点来增强用户体验的讨论，这与游戏化的概念相关联。

---

### **Cohere ▷ #**[**projects**](https://discord.com/channels/954421988141711382/1218409701339828245/1301078102444867614) (1 messages):

> - `邀请状态`
> - `申请被拒`

- **咨询邀请状态**：一位成员询问上周申请后的邀请状态，并对未收到任何回复表示担忧。
  
  - 他们特别想知道申请被拒绝是否存在任何依据。
- **咨询申请响应时间**：另一位成员提出了关于申请通常响应时间的问题，以及延迟是否属于正常现象。
  
  - 这凸显了对负责审核申请的组织在沟通方面的潜在担忧。

---

### **Cohere ▷ #**[**cohere-toolkit**](https://discord.com/channels/954421988141711382/1254901651081269268/) (1 messages):

sssandra: <@1132196995361157171> 你好！这是你在使用 toolkit 时遇到的错误吗？

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1300901350078681178) (36 messages🔥):

> - `Browserbase Funding`
> - `ChatGPT Chat History Search`
> - `LLM Evaluation Challenges`
> - `Realtime API Updates`
> - `SimpleQA Benchmark`

- **Browserbase 融资 2100 万美元用于网页自动化**：Browserbase 宣布已完成 **2100 万美元 A 轮融资**，由 **Kleiner Perkins** 和 **CRV** 领投，旨在帮助 **AI 初创公司大规模实现网页自动化**。
  
  - *What will you 🅱️uild?* 体现了他们对未来发展的宏伟目标。
- **ChatGPT 推出聊天记录搜索**：OpenAI 在 ChatGPT 网页版推出了 **搜索聊天记录** 的功能，使用户能够快速引用过去的对话或从上次中断的地方继续。
  
  - 该功能旨在通过提供更便捷的历史对话访问方式来提升用户体验。
- **LLM 评审员的常见错误**：由 **Hamel Husain** 撰写的指南强调了团队在使用 LLM 评审员时面临的陷阱，例如 **指标过多** 以及忽视领域专家。
  
  - 这次讨论强调了需要经过验证的测量方法，以提高评估的准确性。
- **OpenAI Realtime API 的新更新**：OpenAI 的 Realtime API 现在包含五个 **全新的表现力语音**，用于语音对语音体验，并通过 Prompt Caching 提供大幅价格优惠。
  
  - 缓存的文本输入可享受 **50%** 的折扣，而缓存的音频输入可享受 **80%** 的折扣，从而促进 API 的成本效益使用。
- **推出用于事实性评估的 SimpleQA**：OpenAI 推出了 **SimpleQA**，这是一个旨在衡量语言模型事实性的新基准测试，由 **4000 个具有确定答案的多样化问题** 组成。
  
  - 该计划旨在通过确保未来的模型能够产生更值得信赖和可靠的结果，来解决 AI 中的 **幻觉问题 (hallucination problem)**。

**提到的链接**：

- [Hamel Husain (@HamelHusain) 的推文](https://x.com/hamelhusain/status/1851645681150382103?s=46): 我看到团队在使用 LLM judges 时最常犯的错误：• 指标过多 • 评分系统复杂 • 忽略领域专家 • 未经验证的测量。这就是为什么我写了这篇指南，包含详细...
- [Paul Klein IV (@pk_iv) 的推文](https://x.com/pk_iv/status/1851270308701106383?s=46): 下一家估值十亿美元的公司将由 Browserbase 驱动。我们已经帮助数百家 AI 初创公司大规模实现网页自动化。现在，我们完成了 2100 万美元的 A 轮融资，由 Klein 领投...
- [Transformer Explainer: LLM Transformer 模型视觉详解](https://poloclub.github.io/transformer-explainer/): 一个交互式可视化工具，向您展示 Transformer 模型在 GPT 等大语言模型 (LLM) 中是如何工作的。
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1851340615344406781?s=46): 我们开始在 ChatGPT 网页版推出搜索聊天记录的功能。现在，您可以快速轻松地调出对话进行参考，或者从上次中断的地方继续聊天。
- [Julien Chaumond (@julien_c) 的推文](https://x.com/julien_c/status/1850844166755864966): @ollama - @huggingface 的集成已经上线一周了，进展如何？显然，非常棒！我们平均每天有 4500 次 pulls。大约每 20 秒就有一次 pull！...
- [OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/openaidevs/status/1851668229938159853?s=46): 两项 Realtime API 更新：- 您现在可以使用五种新语音构建语音对语音 (speech-to-speech) 体验——这些语音更具表现力和可控性。🤣🤫🤪 - 我们通过使用 prompt caching 降低了价格...
- [Jeff Harris (@jeffintime) 的推文](https://x.com/jeffintime/status/1851674642966286437?s=46): 这些新语音的 prompt-able 能力大大增强了！- 口音 - 情感 - 耳语 - 强调 - 说话速度 - 角色，您可以在 playground 中进行探索 https://platform.openai.com/playground/rea...
- [Coframe (@coframe_ai) 的推文](https://x.com/coframe_ai/status/1851287230746419649?s=46): 网页已死。我们从 @khoslaventures 和 @natfriedman 筹集了 900 万美元，帮助赋予它生命 ⚡
- [zbeyens (@zbeyens) 的推文](https://x.com/zbeyens/status/1851314462155751896?s=46): 介绍 Plate AI，一个由 AI 命令和 Copilot 驱动的富文本编辑器。◆ 可配置插件 ◆ 200+ shadcn/ui 组件 ◆ AI SDK
- [OpenAI (@OpenAI) 的推文](https://x.com/openai/status/1851680760539025639?s=46): 事实性 (Factuality) 是人工智能部署中最大的开放性问题之一。我们正在开源一个名为 SimpleQA 的新基准测试，用于衡量语言模型的事实性。http...
- [Sundar Pichai (@sundarpichai) 的推文](https://x.com/sundarpichai/status/1851366823050297370?s=46): @YouTube 5/ 我们继续投资最先进的基础设施以支持我们的 AI 努力，包括在我们的数据中心内部进行重要工作以提高效率，同时进行重大的硬件...
- [Jason Wei (@_jasonwei) 的推文](https://x.com/_jasonwei/status/1851681730845118799?s=46): 很高兴开源一个名为 SimpleQA 的新幻觉 (hallucinations) 评估工具！有一段时间感觉没有很好的事实性基准测试，因此我们创建了一个简单、可靠且易于...
- [Octoverse: AI 助力 Python 成为顶级语言，全球开发者数量激增](https://github.blog/news-insights/octoverse/octoverse-2024/#the-most-popular-programming-languages): 在今年的 Octoverse 报告中，我们研究了 GitHub 上的公共和开源活动如何显示 AI 随着全球开发者社区规模的激增而扩张。
- [GitHub - langchain-ai/langgraph: 将弹性的语言 Agent 构建为图。](https://github.com/langchain-ai/langgraph): 将弹性的语言 Agent 构建为图。通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。
- [GitHub - langchain-ai/langgraphjs: ⚡ 将语言 Agent 构建为图 ⚡](https://github.com/langchain-ai/langgraphjs): ⚡ 将语言 Agent 构建为图 ⚡。通过在 GitHub 上创建账号来为 langchain-ai/langgraphjs 的开发做出贡献。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1300959293540925573) (11 条消息🔥):

> - `Ethos NPU 的看法`
> - `评估套件持有情况`
> - `Tinygrad 开发问题`
> - `Tinygrad 字体咨询`

- **Tinycorp 对 Ethos NPU 的非正式立场**：一名成员询问了 **Tinycorp 对 Ethos NPU 官方的非正式意见**，引发了对社区见解的好奇。
  
  - 一些参与者建议从更具体的角度切入，指出如果直接询问硬件规格和未来支持相关的问题，咨询效果会更好。
- **寻找评估套件持有者**：一名成员询问是否有人拥有 **评估套件 (evaluation kit)**，暗示了同行间分享经验的可能性。
  
  - 该提议引出了一些建议，即提出更多技术性问题，以获取关于产品实际用途的详细回复。
- **关于 Tinygrad 开发背景的讨论**：有人提醒应遵守社区规则，强调讨论应集中在 **Tinygrad 的开发与使用**上。
  
  - 回复建议成员们明确在何处发布与这些核心主题不直接相关的问题。
- **关于 Tinygrad 网站字体的咨询**：一名用户询问了 **tinygrad 网站顶部图片**所使用的字体，表现出对网站设计的兴趣。
  
  - 该问题引发了关于讨论相关性的社区准则提醒。
- **对 NPU 的普遍看法**：一名成员对 **开源社区** 对 NPU 的热情表示不确定，并反思了之前在 Microsoft 笔记本电脑等产品中对 **NPU 性能** 的体验。
  
  - 对话随后转向探索对 NPU 的潜在支持以及 **TOSA** 等计划的相关性，在随意的询问与严肃的技术讨论之间取得了平衡。

**提到的链接**：

- [tinygrad: 一个简单且强大的神经网络框架](https://tinygrad.org): 未找到描述
- [提问的智慧](http://www.catb.org/~esr/faqs/smart-questions.html): 未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1300926411028303892) (18 messages🔥):

> - `Training Jobs on Tinybox`
> - `Qwen2's Base Building Blocks`
> - `EfficientNet OpenCL Issues`
> - `Exporting Models to ONNX`
> - `Testing Time Training Approaches`

- **在 Tinybox 上运行长时训练任务**：成员们讨论了在远程 Tinybox 上管理长时训练任务的策略，强调使用 **tmux** 和 **screen** 等工具来维持持久会话。
  
  - 有人提到了一种更有效的替代方案建议，但承认因为太懒而没有切换。
- **Qwen2 非常规的基础构建块**：对于 **Qwen2** 从零开始重新实现 **rotary embedding** 和 **MLP** 等基础元件的做法引发了好奇，并引出了关于其与阿里巴巴隶属关系的疑问。
  
  - 一位用户幽默地表达了对阿里巴巴在此情况下的影响力的挫败感。
- **EfficientNet 在 OpenCL 中输出爆炸**：一位用户在通过 C++ 中的自定义 OpenCL 内核实现 **EfficientNet** 时遇到了输出爆炸（数值发散）的问题，并询问用于比较 buffer 的调试工具。
  
  - 针对如何从 **tinygrad** 实现中访问和转储 buffer 以协助排查问题，成员们给出了建议。
- **ONNX 兼容性的模型导出策略**：讨论重点介绍了将 tinygrad 模型导出到 **ONNX** 的潜在方法，建议利用现有脚本在性能较弱的硬件上进行潜在的模型优化。
  
  - 关于是直接导出模型，还是探索用于芯片部署的替代字节码编译方法，存在一些争论。
- **测试时训练（Test Time Training）与格式问题**：成员们考虑了测试时训练对嵌入式模型的影响，特别是内存中保留原始权重格式的重要性。
  
  - 大家的观点倾向于认为，标准化使用 **ONNX** 进行模型导出可以简化跨各种平台的集成过程。

**提到的链接**：

- [Hailo-Application-Code-Examples/runtime/python/streaming/yolox_stream_inference.py at 77441f09b38f4a548fa1bb2f0eaca75701b62fa9 · hailo-ai/Hailo-Application-Code-Examples](https://github.com/hailo-ai/Hailo-Application-Code-Examples/blob/77441f09b38f4a548fa1bb2f0eaca75701b62fa9/runtime/python/streaming/yolox_stream_inference.py#L34)：通过在 GitHub 上创建账号为 hailo-ai/Hailo-Application-Code-Examples 的开发做出贡献。
- [tinygrad/extra/export_model.py at master · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/master/extra/export_model.py)：你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad
- [tinygrad/examples/compile_tensorflow.py at 4c0ee32ef230bdb98f0bc9d0a00f8aaaff4704f1 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/4c0ee32ef230bdb98f0bc9d0a00f8aaaff4704f1/examples/compile_tensorflow.py#L39-L40)：你喜欢 pytorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad

---

### **Modular (Mojo 🔥) ▷ #**[**general**](https://discord.com/channels/1087530497313357884/1098713601386233997/1300993707729752115) (6 messages):

> - `Idiom of Mojo vs Python`
> - `Learning Resources for Mojo`
> - `Contributing to NuMojo and Basalt Projects`
> - `Linear Algebra Implementation in Mojo`
> - `GPU Utilization in Mojo`

- **Mojo 的惯用法（Idiom）仍在演进**：一位成员分享道，**Idiomatic Mojo**（惯用 Mojo）仍在探索中，随着语言获得新功能，新的“最佳实践”也在不断涌现。
  
  - 这表明与 Python 等更成熟的语言相比，该语言的惯用法具有流动性。
- **缺乏 Mojo 学习资源**：另一位成员表示，很难找到学习 **linear algebra**（线性代数）及其在 Mojo 中实现的资源，特别是在 GPU 使用方面。
  
  - 有人指出，鉴于现有资料有限，为 **NuMojo** 和 **Basalt** 等项目做贡献可能会受益于直接与项目负责人沟通。
- **线性代数方法尚未定型**：一位成员指出，对于 Mojo 中线性代数的最佳方法尚未达成共识，目前的实现很大程度上基于其他语言或书籍的翻译。
  
  - 他们强调，虽然现有实践相对狭窄，但方法可能会根据 **speed**（速度）和 **optimization**（优化）等指标而有所不同。

---

### **Modular (Mojo 🔥) ▷ #**[**mojo**](https://discord.com/channels/1087530497313357884/1151418092052815884/1301016112560996453) (17 messages🔥):

> - `Mojo architecture`
> - `C++ compatibility`
> - `Syntax proposal`
> - `C++ macros`
> - `Custom decorators`

- **Mojo 与 C++ 的潜在兼容性**：有一场关于实现与 **C++ 100% 兼容**的讨论，成员们表示如果有人能做到这一点，那一定是 Chris Lattner。
  
  - 一位用户评论说这将是一个**彻底的奇迹**，突显了围绕这一话题的兴趣和担忧。
- **Mojo 的创新语法提案**：一位成员提议将 'alias' 重命名为 'static'，引发了关于在 Mojo 中使用 **static** 术语的影响的讨论，因为这可能会因其在 C++ 中的典型用法而引起混淆。
  
  - 其他人表示，与 C++ 的 **constexpr** 相比，该关键字可能无法准确传达其预期的功能。
- **关于 Mojo 中 C++ 宏的辩论**：由于 **C++ 宏**可能给编译器带来的复杂性，人们对此表示担忧，更倾向于使用**卫生宏 (hygienic macros)**。
  
  - 一位成员强调，Mojo 对编译时执行的**函数**的关注可能会减轻对传统宏的需求。
- **Mojo 中的自定义装饰器**：提到了在 Mojo 中实现**自定义装饰器**的计划，并相信它们与编译时执行相结合可能就足够了。
  
  - 然而，有人指出某些功能（如编译时的 SQL 查询验证）可能超出了装饰器所能完成的范围。
- **替代预处理选项**：一位用户建议探索 [GPP](https://github.com/logological/gpp)（一种通用预处理器），作为在等待 Mojo 中更复杂功能时的替代方案。
  
  - 这反映了人们正在不断寻找能够增强开发体验的工具，同时整合来自 C++ 等成熟语言的功能。

**提及的链接**：

- [GitHub - logological/gpp: GPP, a generic preprocessor](https://github.com/logological/gpp)：GPP，一个通用预处理器。通过在 GitHub 上创建账户，为 logological/gpp 的开发做出贡献。
- [Issues · modularml/mojo](https://github.com/modularml/mojo/issues/3725)：Mojo 编程语言。通过在 GitHub 上创建账户，为 modularml/mojo 的开发做出贡献。

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1300940150364176474) (2 messages):

> - `create-llama app`
> - `ToolhouseAI tools`
> - `hackathon insights`

- **LlamaIndex 的 create-llama 应用发布**：现在可以使用新的 **create-llama** 工具在几分钟内创建一个 LlamaIndex 应用，允许设置完整的 **Next.js** 或 **Python FastAPI** 后端。
  
  - 用户可以从各种预配置的用例中进行选择，如 **Agentic RAG** 或**数据分析**，并旨在轻松摄取多种文件格式。
- **ToolhouseAI 是游戏规则改变者**：@ToolhouseAI 提供了数十种高质量工具，显著减少了 LlamaIndex Agent 的开发时间，最近一次黑客松的参与者证明了这一点。
  
  - 这些现成的工具可以直接插入 Agent，使集成变得顺畅且有效，证实了它们作为大幅节省时间工具的地位。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1300983544180375562) (14 messages🔥):

> - `Multi-agent query pipelines`
> - `LlamaIndex workflows`
> - `RecursiveRetriever class issues`
> - `Parallel agents with memory`

- **多 Agent 查询流水线建议**：[Cheesyfishes](https://www.youtube.com/watch?v=wuuO04j4jPc) 确认使用 LlamaIndex workflows 方法构建多 Agent 查询流水线是一个很好的策略，并分享了一个展示该实现的 demo。
  
  - 视频中的资源可以在[这里](https://github.com/run-llama/multi-agent-concierge/tree/main/video_tutorial_materials)找到。
- **Orchestrator Agent 在多 Agent 系统中的角色**：当一个发言 Agent 无法完全解决问题时，它总是会返回到 Orchestrator Agent 请求转移，这是为了减少 Agent 性能复杂性的设计。
  
  - 这一设计选择旨在通过限制每个 Agent 直接访问其他 Agent 的权限来维护系统的可扩展性。
- **RecursiveRetriever 类的问题**：一位用户提出 RecursiveRetriever 类无法正常工作，尽管节点具有关系，但 `add_nodes` 的输出为空。
  
  - 他们尝试使用特定命令检索所有节点，但未获成功。
- **对 LlamaIndex 多 Agent 指南的需求**：一位用户询问了关于并行使用多个带有不同 LLM 且保持 memory 循环的 Agent 的指南，类似于 CrewAI。
  
  - Cheesyfishes 推荐了一个视频示例，该示例解决了 LlamaIndex 的多 Agent 编排问题，尽管它可能是按顺序运行 Agent 的。
- **多 Agent 框架中的并行工具调用**：在确认分享的视频是按顺序运行 Agent 的同时，Cheesyfishes 指出，工具调用实际上可以相对容易地并发执行。
  
  - 这种灵活性允许 Agent 交互的多样化实现。

 

**提到的链接**：[multi-agent-concierge/video_tutorial_materials at main · run-llama/multi-agent-concierge](https://github.com/run-llama/multi-agent-concierge/tree/main/video_tutorial_materials)：一个使用 LlamaIndex 进行多 Agent 编排的示例 - run-llama/multi-agent-concierge

 

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1301030634533425213) (1 messages):

> - `RAG with LlamaIndex`
> - `Text-to-SQL integration`

- **LlamaIndex 中 RAG 与 Text-to-SQL 的协同作用**：一篇文章讨论了使用 [LlamaIndex](https://medium.com/ai-artistry/unleashing-the-power-of-rag-and-text-to-sql-with-llamaindex-5aa27c697ad0) 将 **RAG (Retrieval-Augmented Generation)** 与 **Text-to-SQL** 集成，展示了其在增强数据检索任务方面的潜力。
  
  - 该文章强调了 LlamaIndex 如何通过有效地将自然语言转换为结构化数据库查询，来优化查询并改善用户体验。
- **探索 RAG 的实际应用**：文章详细介绍了 **RAG** 技术的几种实际应用，特别是在自动化数据检索和提高 SQL 查询准确性方面。
  
  - 文中提到，在使用具有 RAG 功能的 LlamaIndex 时，用户的查询响应时间**减少了 30%**。
- **通过 LlamaIndex 增强用户交互**：LlamaIndex 旨在通过自然语言处理降低 SQL 生成的复杂性，从而简化用户与数据库之间的交互。
  
  - 该实现提高了用户满意度，据报告，用户感到更有能力在没有深厚技术知识的情况下提取数据。

 

---

### **DSPy ▷ #**[**papers**](https://discord.com/channels/1161519468141355160/1203568372667645963/1300986545418928240) (5 messages):

> - `Extreme Multi-Label Classification` (极端多标签分类)
> - `DSPy Programming Model` (DSPy 编程模型)
> - `Online Search for Labels` (标签在线搜索)

- **探索 In-Context Learning 的局限性**：题为 [IReRa: In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178) 的论文讨论了在语言模型（LM）缺乏关于类别的先验知识时，解决多标签分类问题所面临的挑战。
  
  - 该研究提出了一个名为 **Infer–Retrieve–Rank** 的程序，用于高效管理 LM 与检索器（retrievers）之间的交互，在 HOUSE、TECH 和 TECHWOLF 基准测试中取得了最先进（state-of-the-art）的结果。
- **相关 GitHub 仓库的可用性**：一名成员指出，论文摘要中提到了一个与所讨论工作相关的 [GitHub repo](https://link.to.repo)。
  
  - 该仓库可能根据论文的研究结果提供了进一步的见解和实现。
- **对在线搜索标签的兴趣**：一名成员询问在分类过程中使用在线搜索标签代替检索器的可行性。
  
  - 这引发了关于 **DSPy model** 是否可以容纳这种 Agent 风格的搜索功能的疑问。
- **关于标签搜索的澄清**：另一名成员对“在线搜索标签而非使用检索器”的含义表示困惑。
  
  - 这表明需要进一步讨论如何调整或以其他方式实现标签检索方法。

 

**提到的链接**：[In-Context Learning for Extreme Multi-Label Classification](https://arxiv.org/abs/2401.12178)：具有数千个类别的多标签分类问题很难仅靠 In-Context Learning 解决，因为语言模型（LM）可能缺乏关于精确类别或如何……的先验知识。

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1301102055062245396) (4 messages):

> - `DSPy Structure Enforcements` (DSPy 结构强制执行)
> - `Structured Outputs` (结构化输出)
> - `MIPROv2 Integration` (MIPROv2 集成)

- **质疑 DSPy 对结构强制执行的需求**：一名成员质疑，当 Outlines 等库已经存在且 API LM 正在引入结构化生成时，DSPy 强制执行结构/类型的必要性。
  
  - *利用这些现有的库不是更好吗？*
- **结构在 DSPy 中的重要性**：另一名成员澄清说，DSPy 自 v0.1 以来就强制执行结构，以学习从 Signatures 到有效 Prompt 和权重的映射，并适应供应商的能力。
  
  - 他们强调，虽然结构化输出可以作为后端，但 DSPy 的方法在遵循格式与保持质量之间取得了平衡。
- **关于输出质量与结构的讨论**：一名成员对“遵循某些格式可能会降低质量”的观点表示怀疑，认为结构化输出可以从设定的约束和正确的语法使用中受益。
  
  - *这种方法可能会产生很好的效果，特别是对于较小的 LM。*
- **将 MIPROv2 与 DSPy 集成**：另一名成员分享了他们使用 zero-shot MIPROv2 和 Pydantic 优先接口处理结构化输出的经验，目前依赖 MIPROv2 步骤进行构建。
  
  - *他们表示希望在优化过程中有一种更集成、更原生（native）的方式来处理结构化输出。*

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1300942811624902657) (4 messages):

> - `Job Automation Predictions` (工作自动化预测)
> - `Open Interpreter vs Claude` (Open Interpreter 对比 Claude)
> - `Restoring Specific Chat Profiles` (恢复特定聊天配置文件)

- **工作自动化预测引发辩论**：一名用户预测 **virtual beings** 将在不久的将来接管大量 **jobs**，并将其比作 **virtual Skynet takeover**（虚拟天网接管）。
  
  - 这一评论引发了关于 AI 对劳动力影响的对话。
- **Open Interpreter 的独特功能**：一名成员询问 Open Interpreter 与 **Claude** 在电脑操作（computer operations）方面的使用有何不同。
  
  - **Mikebirdtech** 回应称，他们通过 `interpreter --os` 实现了 **Claude's computer use**，同时也强调了他们的 **open-source** 优势。
- **恢复聊天配置文件的挑战**：一名用户询问如何恢复使用了之前讨论中特定 **profile/model** 的聊天。
  
  - 他们提到，虽然可以通过 `--conversations` 进行恢复，但它会默认使用 **standard model**，而不是之前使用的模型。

 

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1300899574629662721) (4 messages):

> - `ChatGPT Chat History Search`
> - `Digitizing Scent`
> - `Scent Teleportation`
> - `Limited Release Fragrance`

- **ChatGPT 现在可以记住聊天记录了**：OpenAI 宣布推出一项功能，允许用户轻松搜索 **ChatGPT web 端的聊天记录**，方便引用或继续之前的对话。
  
  - 这一新功能旨在提升用户体验并简化平台内的交互。
- **突破性的气味数字化**：一个团队成功将**新鲜的夏季李子**数字化，标志着气味数字化领域的一个重大里程碑，且无需人工干预。
  
  - 一位成员表达了兴奋之情，提到他们很享受随身携带**李子香气**的感觉，并考虑生产一款限量版香水以资助科学研究。
- **气味传送里程碑庆典**：Osmo 团队庆祝了他们在气味传送方面取得的成就，并就这一成就对未来创新的意义发表了感言。
  
  - 他们表达了与社区互动的愿望，询问大家是否对支持科学事业的香水发布感兴趣。

**提到的链接**：

- [来自 Alex Wiltschko (@awiltschko) 的推文](https://fxtwitter.com/awiltschko/status/1851327552490733686?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：好吧，我们真的做到了。我们实现了气味数字化。一颗新鲜的夏季李子是第一个在没有人工干预的情况下被完全数字化并重新打印的水果和气味。它闻起来棒极了。天哪，我仍然...
- [来自 OpenAI (@OpenAI) 的推文](https://fxtwitter.com/openai/status/1851340615344406781?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)：我们开始推出在 ChatGPT web 端搜索聊天记录的功能。现在你可以快速轻松地调出聊天记录进行参考，或者从中断的地方继续聊天。

---

### **LangChain AI ▷ #**[**general**](https://discord.com/channels/1038097195422978059/1038097196224086148/1301120030699814974) (2 messages):

> - `invoke function response time`
> - `FastAPI routes efficiency`
> - `Hugging Face Transformers documentation`

- **Invoke 函数引发响应时间谜团**：用户报告了一个问题，在访问 **Llama3.1:70b** 模型的相同推理 URL 时，调用 retriever 的 **.invoke 函数** 返回的响应时间（超过 **120 秒**）明显慢于本地执行（**20 秒**）。
  
  - 用户怀疑存在影响性能的潜在**安全问题**，并寻求社区的帮助。
- **FastAPI 路由性能表现出色**：尽管 invoke 函数存在性能问题，但经调试确认，**FastAPI** 路由表现良好，执行时间不到 **1 秒**。
  
  - 用户保证发送的数据正确且类型无误，将问题锁定在 invoke 函数上。
- **Hugging Face Transformers 文档令人沮丧**：另一位用户表示，由于查阅文档困难，无法使用 **Hugging Face Transformers** 设置 **chat/conversational pipeline**。
  
  - 他们强调在文档中很难找到实现目标所需的必要指导。

---

### **LangChain AI ▷ #**[**share-your-work**](https://discord.com/channels/1038097195422978059/1038097372695236729/1301166756395749526) (1 messages):

> - `Knowledge Nexus AI`
> - `KNAI Discord Community`
> - `KNAI Publication on Medium`
> - `Decentralized Knowledge Systems`
> - `Knowledge Graphs and Semantic Web`

- **Knowledge Nexus AI 启动倡议**：Knowledge Nexus AI (KNAI) 宣布启动社区倡议，旨在桥接**人类知识**与 **AI**，推动一个**去中心化的未来**。
  
  - 他们的使命强调将集体知识转化为**结构化、机器可读的数据**，以驱动医疗、教育和供应链等行业的洞察。
- **加入 KNAI Discord 社区**：KNAI 正在为对 **Knowledge Graphs**、**Semantic Web** 以及 **Web3 & Blockchain** 技术感兴趣的人士创建一个活跃的 Discord 空间。
  
  - 他们邀请创新者、研究人员和爱好者共同塑造**去中心化知识系统**的未来。
- **为 Medium 上的 KNAI 出版物撰稿**：KNAI 正在为其 Medium 出版物征集撰稿人，重点关注 **Knowledge Graph 创新**和 **AI 进展**等话题。
  
  - 鼓励个人开始起草文章，分享他们在**去中心化知识系统**方面的专业知识和实用的技术指南。
- **赋能协作式知识系统**：KNAI 旨在创建一个可访问的协作平台，为多个行业的**洞察与创新**提供动力。
  
  - 他们热衷于通过**社区协作**为机器构建一部全面的百科全书。

 

---

### **LangChain AI ▷ #**[**tutorials**](https://discord.com/channels/1038097195422978059/1077843317657706538/1301212233816084553) (1 messages):

> - `OppyDev Plugin System`
> - `Enhancing AI Output`

- **OppyDev 推出插件系统**：分享了 **OppyDev 插件系统**的简要介绍，展示了它如何通过 **Chain-of-Thought** 推理来增强标准 AI 模型的输出。
  
  - 所解释的过程旨在提供更清晰、更详细的回复，使 AI 交互更加高效。
- **在 OppyDev 上观看教程**：提供了一个演示插件系统及其功能的教程视频，可在[此处](https://www.youtube.com/watch?v=6JlQwnYn7RY&t=14s)观看。
  
  - 该视频强调了通过插件增强 AI 响应的实际案例，促进更深层次的理解。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/) (1 messages):

duh_kola: 没错，但我想要训练的是 Instruction 版本，哈哈。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general-help**](https://discord.com/channels/1104757954588196865/1110594519226925137/1300981796866887730) (2 messages):

> - `LoRA finetuning`
> - `H100 GPUs`
> - `BitsAndBytes issue`

- **LoRA 微调问题仍未解决**：一位成员表示在 H100 GPUs 上寻找 **LoRA finetuning** 解决方案时遇到困难，并暗示 **QLoRA** 可能是目前唯一可行的变通方案。
  
  - 另一位成员确认 **BitsAndBytes 关于 Hopper 8bit 的问题**仍然处于 Open 状态且尚未解决，该问题依然存在。
- **量化挑战持续存在**：讨论强调了量化相关问题的持续挑战，特别是在 **BitsAndBytes for Hopper 8bit** 的背景下。
  
  - 尽管做出了努力，但似乎针对这些技术障碍尚未建立明确的解决方案。

 

---

### **LAION ▷ #**[**general**](https://discord.com/channels/823813159592001537/823813160075132991/1300962200311173123) (2 messages):

> - `Image Decoding Issues`

- **图像解码中的数值截断问题**：一位成员指出，在将解码后的图像转换为 uint8 之前，如果未能将**数值限制 (clamp)** 在 [0,1] 范围内，会导致**超出范围的值发生回绕 (wrap)**。
  
  - 这可能会导致图像质量和外观出现意想不到的结果。
- **对解码工作流的担忧**：另一位参与者建议 **decoding workflow** 中可能存在其他缺陷，潜在地影响整体图像处理。
  
  - 需要进一步讨论以查明这些问题并提高鲁棒性。

 

---

### **LAION ▷ #**[**research**](https://discord.com/channels/823813159592001537/824374369182416994/) (1 messages):

thejonasbrothers: [https://arxiv.org/abs/2410.20424](https://arxiv.org/abs/2410.20424)

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-questions**](https://discord.com/channels/1280234300012494859/1280370030609170494/1301159991826186323) (2 messages):

> - `LLM Agents Quizzes`
> - `LLM Agents Hackathon`
> - `Course Sign Up`
> - `Discord Channel`

- **LLM Agents 测验位置已分享**：一位成员询问了该课程**每周测验 (weekly quizzes)** 的位置。
  
  - 另一位成员迅速回复了[测验](https://llmagents-learning.org/f24)链接，并表示 *'在这里你可以找到所有的测验。'*
- **LLM Agents Hackathon 公告**：参与者收到了关于 **LLM Agents Hackathon** 的通知，并附带了报名了解更多详情的链接。
  
  - 公告中包含了一个指向 [Hackathon 详情](https://rdi.berkeley.edu/llm-agents-hackathon/) 的链接，供感兴趣的人员查看。
- **课程报名指南**：为有意向的学生分享了如何通过 **Google Form** 报名参加课程的说明。
  
  - 鼓励参与者填写此 [表格](https://forms.gle/svSoNhKcGFjxup989) 以注册课程。
- **加入课程讨论**：提供了如何加入 Discord 上的课程讨论的详细信息。
  
  - 参与者可以加入 [LLM Agents Discord](https://discord.gg/NWVpQ9rBvd) 的 **MOOC 频道** 进行提问和讨论。

 

**提到的链接**：[Large Language Model Agents MOOC](https://llmagents-learning.org/f24): MOOC, 2024 年秋季

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1301204930698154097) (1 messages):

> - `Transformer Labs Event`
> - `Lumigator Tech Talk`

- **Transformer Labs 将演示在 LLM 上实现本地 RAG**：来自 **Transformer Labs** 的团队正在举办一场活动，展示如何在本地环境中使用易于安装的 UI 来训练、微调、评估和使用 **LLM** 上的 **RAG**。
  
  - 参与者可以期待一种**无代码 (no-code)** 方案，使其适用于所有技能水平的人员。
- **深入了解 Lumigator 工具**：工程师们将带来关于 **Lumigator** 的详细**技术演讲 (tech talk)**，这是一个开源工具，旨在帮助用户选择最适合其需求的 **LLM**。
  
  - 该工具旨在简化工程师在选择**大语言模型 (large language models)** 时的决策过程。

 

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #**[**leaderboard**](https://discord.com/channels/1111172801899012102/1214705495974092810/1301126927905587301) (1 messages):

> - `Llama-3.1-8B-Instruct (FC)`
> - `Llama-3.1-8B-Instruct (Prompting)`
> - `Function Calling Performance`
> - `Model Comparison`

- **Llama-3.1-8B-Instruct (FC) 表现逊于 Prompting**：一位成员质疑为什么 **Llama-3.1-8B-Instruct (FC)** 的表现比 **Llama-3.1-8B-Instruct (Prompting)** 差，原本预期 FC 模型在 Function Calling 任务中会有更好的结果。
  
  - *这种差异是否有原因？* 强调了对基于模型设计的性能预期的关注。
- **对 Function Calling 性能的预期**：另一位参与者指出，他们预期 **FC** 变体应该由于其专注于 Function Calling 的设计而表现出色。
  
  - 他们思考观察到的性能是令人惊讶，还是预示着模型架构存在潜在问题。

 

---

---

---

---

---

{% else %}

> 完整的频道细分内容已针对邮件进行截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}