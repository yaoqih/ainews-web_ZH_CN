---
companies:
- openai
- langchain
- cognition
- google-deepmind
- vercel
- cloudflare
- openrouter
date: '2025-06-25T05:44:39.731046Z'
description: '**上下文工程 (Context Engineering)** 正成为 AI 领域的一个重要趋势，得到了 **Andrej Karpathy**、**Cognition**
  的 **Walden Yan** 以及 **Tobi Lutke** 等专家的强调。它涉及通过合理组合提示词、检索、工具和状态来管理大语言模型（LLM）的上下文窗口，以优化性能，这已经超越了传统的提示工程。**LangChain**
  及其工具 **LangGraph** 在推动这一方法方面备受关注。此外，**OpenAI** 为 **Google Drive**、**Dropbox**、**SharePoint**
  和 **Box** 等平台推出了 **ChatGPT 连接器**，为 Pro 用户增强了上下文集成能力。其他值得关注的新闻还包括 **Vercel Sandbox**
  和 **Cloudflare Containers** 的发布、**Google DeepMind** 的 **Gemini Code** 泄露及发布，以及 **OpenRouter**
  的融资进展。'
id: MjAyNS0w
models:
- gemini-code
people:
- karpathy
- walden_yan
- tobi_lutke
- hwchase17
- rlancemartin
- kwindla
- dex_horthy
title: '**上下文工程：远不止于提示词**'
topics:
- context-engineering
- retrieval-augmented-generation
- tools
- state-management
- history-management
- prompt-engineering
- software-layer
- chatgpt-connectors
- api-integration
---

**精心构建的上下文就是你所需的一切。**

> 2025年6月24日至6月25日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（220 个频道，5002 条消息）。预计节省阅读时间（以 200wpm 计算）：447 分钟。我们的新网站现已上线，提供完整的元数据搜索和美观的 vibe-coded 历期内容展示。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

今天有很多相关的闻故事可供选择：Vercel Sandbox 和 Cloudflare Containers 巧合得离奇的同步发布；Gemini Code（GDM 对标 Claude Code 的竞争产品）的泄露及随后发布（带有慷慨的额度限制）；或者是 OpenRouter 的融资。

但今天最可能留下深远影响的是“Context Engineering”（上下文工程）作为一个值得关注的趋势得到了确认，这一术语由 [Dex Horthy](https://x.com/dexhorthy/status/1938265310869721478) 或 [Cognition 的 Walden Yan](https://cognition.ai/blog/dont-build-multi-agents) 提出：


![](https://resend-attachments.s3.amazonaws.com/9eE7HoXEVqjxrks)


并由 Tobi Lutke 在上周推广：


![](https://resend-attachments.s3.amazonaws.com/f5f9LBXplycV5dV)


最近几天，许多人都发表了看法：

- [Harrison](https://x.com/hwchase17/status/1937648042985030145)：“我们认为 LangGraph 在实现完全自定义的 context engineering 方面表现出色——但我们希望让它变得更好”
- [Lance Martin](https://rlancemartin.github.io/2025/06/23/context_engineering/)：“上下文通过多种方式进入 LLM，包括提示词（例如用户指令）、检索（例如文档）和工具调用（例如 API）。就像 RAM 一样，LLM 的上下文窗口在处理这些各种上下文来源时具有有限的‘通信带宽’。正如操作系统管理 CPU 的 RAM 内容一样，我们可以将‘context engineering’视为打包和管理 LLM 执行任务所需的上下文。”
- [Kwindla](https://gist.github.com/kwindla/f755284ef2b14730e1075c2ac803edcf)：“如果你的语音 agent 需要可靠地遵循一系列步骤，或者将进行超过几轮的对话，你可能需要考虑进行‘context engineering’，以保持对话上下文简短且集中。
    
    思考 context engineering 的一种有用方式是将对话设计为一系列工作流状态。每个状态对应语音交互过程中一个特定的‘待办任务’。”
    
- [Andrej](https://x.com/karpathy/status/1937902205765607626)：*“在每一个工业级的 LLM 应用中，context engineering 是一门精妙的艺术和科学，即为下一步操作在上下文窗口中填充恰到好处的信息。说是科学，是因为正确完成这项工作涉及任务描述和解释、few-shot 示例、RAG、相关（可能是多模态的）数据、工具、状态和历史、压缩……太少或形式不对，LLM 就没有最佳性能所需的正确上下文。太多或太无关，LLM 的成本可能会上升，性能可能会下降。做好这一点极具挑战性。说是艺术，是因为它涉及到关于‘人类精神’的 LLM 心理学的引导直觉。”*
- [Dex Horthy](https://github.com/humanlayer/12-factor-agents/blob/main/content/factor-03-own-your-context-window.md)：掌控你的上下文窗口（Own your Context Window）
    
    
![](https://resend-attachments.s3.amazonaws.com/YhUo9VWTXzwwmd3)

    

理所当然地，这已立即成为 AI Engineering 中必须掌握的术语和技能。

---

# AI Twitter 综述

**AI 开发、工具与框架**

- **“Context Engineering” 的兴起**：[@karpathy](https://twitter.com/karpathy/status/1937902205765607626) 倡导使用 **“context engineering”** 一词而非 “prompt engineering”，认为它能更好地描述填充 LLM 上下文窗口这一复杂的艺术。他详细说明了这涉及一个复杂的软件层，用于管理 **RAG**、**tools**、**state**、**history** 等，以实现最佳性能。在此基础上，来自 **LangChain** 的 [@hwchase17](https://twitter.com/hwchase17/status/1937648042985030145) 指出这是一个“新的热门话题”，并建议使用 **LangGraph** 来简化上下文管理。[@RLanceMartin](https://twitter.com/hwchase17/status/1937622377267101921) 也撰文介绍了这一过程中的流行模式。
- **ChatGPT 和 OpenAI 发布重大产品更新**：[@OpenAI](https://twitter.com/OpenAI/status/1937681383448539167) 宣布面向 Pro 用户推出 **Google Drive、Dropbox、SharePoint 和 Box** 的 **ChatGPT 连接器**，允许用户为工作任务引入独特的上下文。在被视为对其他代码工具的回应中，[@corbtt](https://twitter.com/corbtt/status/1937757709241311531) 将 **Sam Altman** 的一段话解读为：OpenAI 即将推出的开源模型将达到 **o3-mini** 的水平。
- **Google 发布 Gemini CLI Agent 并提供慷慨的免费额度**：**Google** 发布了 **Gemini CLI**，这是一个开源 (**Apache 2.0**) 的终端 AI Agent。由 [@googleaidevs](https://twitter.com/demishassabis/status/1938023045320335789) 等人分享的公告强调了其强大的免费层级：**每天 1,000 次请求**，速率限制为 **60 RPM**，这被许多人视为推动普及的战略举措。该 CLI 支持 tools 和 **MCP**，[@osanseviero](https://twitter.com/osanseviero/status/1937861590805872801) 分享了 GitHub 和博客链接。社区反应热烈，[@scaling01](https://twitter.com/scaling01/status/1937900212242047384) 等人开玩笑说，如此慷慨的额度一定是某个“读写困难的实习生”写错了。正如 [@qtnx_](https://twitter.com/qtnx_/status/1937976300360155544) 所言，这次发布引发了关于 CLI 编程 Agent（包括 **Claude Code** 等竞争对手）“大逃杀”式竞争的讨论。
- **多 Agent 系统的新课程和协议**：**DeepLearningAI** 和 **Andrew Ng** 宣布了一门关于 **Agent Communication Protocol (ACP)** 的新课程，该协议是与 **IBM Research** 的 **BeeAI** 合作开发的 (@AndrewYNg/status/1937907934094360582)。课程教授如何构建能够使用标准化的 RESTful 接口在不同框架之间进行通信和协作的 Agent。与此同时，围绕 **Model Context Protocol (MCP)** 的生态系统正在壮大，[@lmstudio](https://twitter.com/multimodalart/status/1937917586899144948) 增加了对 MCP 服务器的支持，[@llama_index](https://twitter.com/jerryjliu0/status/1937653599972286873) 发布了用于构建兼容 Claude 的 MCP 服务器的开源模板。
- **DSPy 框架获得关注**：**DSPy** 编程框架正受到显著关注，**Shopify CEO Tobi Lütke** 称其为他 [“首选的 context engineering 工具”](https://twitter.com/lateinteraction/status/1938005712489083252)。框架创建者 [@lateinteraction](https://twitter.com/lateinteraction/status/1937701902000480599) 澄清说，**DSPy** 是一个以 **Signatures** 和 **Modules** 为核心的编程模型，而不仅仅是优化器的集合。**Johns Hopkins University** 关于 DSPy 的新课程也受到了关注 (@DSPyOSS/status/1937698576949518351)。
- **构建和评估 AI Agent 的建议**：在 [**AI.Engineer** World Fair](http://ai.engineer/) 的演讲中，[@jerryjliu0](https://twitter.com/jerryjliu0/status/1937681047875191159) 分享了构建自动化知识工作的 AI Agent 的实际步骤，讨论了 Agent 架构和精心设计的 tools 的重要性。在评估方面，[@HamelHusain](https://twitter.com/HamelHusain/status/1937687931470193102) 推荐了 [@eugeneyan](https://twitter.com/HamelHusain/status/1937687931470193102) 撰写的关于长上下文问答系统评估 (evals) 的指南。

**New Models, Research & Techniques**

- **Google 发布用于 DNA 分析的 AlphaGenome**：**Google DeepMind** 和 **Google AI** 推出了 **AlphaGenome**，这是一款旨在通过预测基因突变影响来帮助科学家更好理解 DNA 的新 AI 模型 (@Google/status/1937897003201044534)。[@IterIntellectus](https://twitter.com/demishassabis/status/1937971182256435323) 将其描述为一种能够读取 **100 万个 DNA 碱基**并仅通过序列预测生物功能的 AI。
- **Anthropic 的 Claude 核心在于数据**：一个反复出现的主题是数据质量的关键作用。[@nrehiew_](https://twitter.com/nrehiew_/status/1937651376013606944) 断言 **Anthropic Claude** 的“灵魂”主要在于其训练数据。这一观点得到了 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1937635148784369828) 的共鸣，他敦促研究人员“停止研究 Subquadratic Attention 论文，去获取更好的数据”。
- **AI 视频和图像生成的进展**：**Kling AI** 宣布了一项 **Motion Control** 功能，可将源视频中的动作捕捉应用到新图像中 (@Kling_ai/status/1937838997730148766)。与此同时，**RunwayML** 宣布其 **Gen-4 References 模型**现已在 API 中可用，提升了一致性和个性化方面的性能 (@c_valenzuelab/status/1937878573852811447)。此外，**OmniGen 2** 以 **Apache 2.0 许可证**发布，被 [@reach_vb](https://twitter.com/reach_vb/status/1937753850259128719) 赞誉为“图像编辑领域的 State of the Art”。
- **推理、生成和训练方面的新研究**：**Sakana AI** 分享了一段视频，解释了他们的 **Reinforcement Learning Teacher**，这是一种利用较小的 Teacher 模型创建推理模型的新方法 (@SakanaAILabs/status/1937743827177206067)。来自**斯坦福大学**和 **Google** 的研究人员推出了 **Weaver**，这是一个旨在弥合“生成-验证差距”（generation-verification gap）的框架，即 LLM 虽然能生成正确答案但无法准确选择它们的问题 (@togethercompute/status/1937653446825435411)。**Snowflake AI Research** 发布了一篇关于 **Arctic Long Sequence Training (ALST)** 的论文，详细介绍了他们在长序列训练方面的方法 (@JayAlammar/status/1937790490092429364)。

**行业新闻与公司战略**

- **Intercom 的“重塑时刻”**：估值 **120 亿美元**的初创公司 **Intercom** 正在经历一个“重塑时刻”，旨在成为一个成熟的 AI App 构建平台，正如 [@swyx](https://twitter.com/swyx/status/1937748319453024527) 所强调的那样。
- **AI 改变医疗和媒体**：据 [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1937909094662463866) 报道，一款能够从常规 CT 扫描中检测**胃癌**的**阿里巴巴** AI 模型已在**中国 20 家医院**部署，筛查了超过 **78,000 名患者**，并在症状出现前数月发现癌症。在媒体领域，[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1937615643731272177) 分享了 **Runway** 的愿景，即 AI 将成为新媒体格局的“底层基础设施”，并将当前时刻比作第一台相机的发明。
- **AI 退出趋势转向人才收购 (Acquihires)**：[@multiply_matrix](https://twitter.com/multiply_matrix/status/1937685737111191835) 的一条推文指出，近期主要的 AI 初创公司退出大多是人才收购，并列举了 **Adept** 被 Amazon 收购、**Inflection** 被 Microsoft 收购以及 **MosaicML** 被 Databricks 收购等例子。
- **行业结盟共同开发 AI Agent**：**Cohere** 宣布其为**斯坦福 DDL 全行业 AI Agent 论坛**的创始成员，与 **Meta**、**Oracle** 和 **PayPal** 联手，共同塑造负责任的发展和跨行业标准 (@cohere/status/1937914623753359378)。

**更广泛的影响与评论**

- **操作系统和浏览器的未来是 AI**：**Perplexity AI CEO** [@AravSrinivas](https://twitter.com/AravSrinivas/status/1937732846543933569) 发表了大胆言论，认为 **“Android 需要为 AI 重构”**，并指出目前的系统是为 Google 的广告业务而非成为一个“真正的 Agentic OS”而优化的。他还表示，**浏览器是 Agent 产生和进化的“原始汤”** (@AravSrinivas/status/1937651271458345028)。
- **工程师进入 AI 领域的路线图**：[@jxmnop](https://twitter.com/jxmnop/status/1937874659980022034) 为想要进入 AI 领域的开发者提供了一份详尽指南。建议包括选择一个特定领域（文本、图像、音频、机器人）和特定方向（训练、推理、数据、安全），像“海绵”一样吸收信息，并执行一个高质量的项目来展示技能。
- **AI 基础设施与电网**：[@dylan522p](https://twitter.com/dylan522p/status/1937943241082437697) 对 **美国电网的脆弱性** 表示担忧，警告称大规模的训练运行可能会引发停电，并导致公众舆论转向反对 AI 基础设施。
- **学术研究现状**：一位 **NeurIPS** 审稿人 [@jxmnop](https://twitter.com/jxmnop/status/1937949143084810625) 分享了在同行评审过程中的挫折经历，描述了一些投稿显然是 **LLM 生成的**、重复的，或者是基于私有的、不可复现的数据。这突显了人们对 AI 领域学术投稿质量和诚信日益增长的担忧。

**法律与政策**

- **美国签证政策要求披露社交媒体信息**：一项新的美国签证政策变动引发了广泛讨论。[@rshereme](https://twitter.com/zacharynado/status/1937923326791295078) 和 [@USAinUKConsular](https://twitter.com/francoisfleuret/status/1937926540769054772) 账号指出，该政策要求所有 **F、M 或 J 类非移民签证申请人** 列出过去五年的社交媒体账号，并公开其个人资料以供审查。
- **Anthropic 在 AI 训练方面赢得关键的“合理使用”裁决**：一名联邦法官裁定 **Anthropic** 训练模型的方法构成 **合理使用 (Fair Use)**，这是 AI 行业的一个重大法律进展。[@JvNixon](https://twitter.com/JvNixon/status/1937654031130010016) 分享了裁决依据，而 [@andykonwinski](https://twitter.com/andykonwinski/status/1937739172263141854) 指出了法庭摘要中的细节，显示 Anthropic 在训练过程中从第三方渠道购买了授权数据集。

**幽默与迷因 (Memes)**

- **Karpathy 的警告**：[@karpathy](https://twitter.com/karpathy/status/1937941695943065640) 发布了一句经典名言：**“愿你的正则项足够强，以免你 RLHF 出一堆 slop。”**
- **讽刺技术怀疑论**：[@giffmana](https://twitter.com/giffmana/status/1937829451670434280) 的一条推文走红，称 **“巨大的金属盒子不能也永远无法在天空中漂浮”**，以此讽刺对技术进步的怀疑论。随后 [@cloneofsimo](https://twitter.com/cloneofsimo/status/1937835663870828716) 也发布了一个类似的讽刺，关于猫进行机器学习研究。
- **Perplexity Logo 投票**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1937953810116448599) 发起了一项投票，邀请社区为 **Perplexity** 的新 Logo 投票，引发了广泛参与。
- **行业讽刺**：[@scaling01](https://twitter.com/scaling01/status/1937900212242047384) 开玩笑说，**Google** 为新推出的 **Gemini CLI** 提供的极其慷慨的免费额度，是因为一名“读写障碍的实习生”把每周 10 次请求误认为是每天 1000 次。

---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. 主要新模型发布与基准测试：Jan-nano-128k & Mistral Small 3.2

- [**Jan-nano-128k：具备超长上下文窗口的 4B 模型（性能依然超越 671B）**](https://v.redd.it/909kwwnbo09f1) ([Score: 755, Comments: 293](https://www.reddit.com/r/LocalLLaMA/comments/1ljyo2p/jannano128k_a_4b_model_with_a_superlong_context/))：**Menlo Research 发布了 Jan-nano-128k，这是一个基于 Qwen3 微调的 4B 参数 LLM，具有 128k token 的上下文窗口，并使用 YaRN scaling 进行了优化。基准测试显示，在极简提示条件下，它的 SimpleQA 评分达到 83.2（配合 MCP），超过了 Deepseek-671B (78.2)，并显著优于 GPT-4o (62.5) 和 Gemini-2.5 Pro (52.9) 等其他领先模型。该模型及其 GGUF 量化版本已在 HuggingFace 上发布（参见 [Jan-nano-128k](https://huggingface.co/Menlo/Jan-nano-128k) 和 [GGUF 转换](https://huggingface.co/Menlo/Jan-nano-128k-gguf)）；其性能取决于是否使用支持 YaRN scaling 的推理引擎（例如 llama.server、Jan app）。技术报告即将发布。** 技术倾向的评论者似乎对 4B 模型的基准测试结果印象深刻，但由于缺乏公开的技术报告，对其参与度指标和基准测试方法仍持怀疑态度。
    - 一位评论者提供了性能背景截图，并提到当使用“重度提示（heavily prompting）”时，Jan-nano-128k 的准确率可达 `83%`。基准测试在有无此技术的情况下均进行了测试，表明该模型的实际表现可能会因 Prompt Engineering 的不同而有很大差异。
    - 讨论中提出了一个关于部署的技术问题，指出虽然 Jan-nano-128k 强调本地运行和隐私，但推荐用法中包含一个依赖项（[mcp-server-serper](https://github.com/marcopesani/mcp-server-serper)），该项需要 Serper API 密钥——这引发了关于完全本地、无 API 部署工作流可行性的讨论。
- [**新款 Mistral Small 3.2 确实给人一种大作的感觉 [非推理模型]**](https://www.reddit.com/r/LocalLLaMA/comments/1lk12th/new_mistral_small_32_actually_feels_like/) ([Score: 242, Comments: 73](https://www.reddit.com/r/LocalLLaMA/comments/1lk12th/new_mistral_small_32_actually_feels_like/))：**Mistral Small 3.2 是一款 24B 参数的 LLM。根据用户测试和对比基准测试，其表现远超其参数级别，特别是在写作和逻辑任务中，超过了 Gemma 3 27B、Llama 3.3 70B 和 Qwen2.5 72B 等模型。提到的技术问题包括在各种量化版本中 Tool calling 损坏和日期输出错误；社区通过 [HuggingFace](https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF) 提供了动态量化的修复版本。为了获得最佳性能，用户建议设置：温度 0.15，重复惩罚 1.0，最小概率采样 0.0，以及 top-p 采样 1.0。** 评论者对即将推出的 Mistral Medium (~70B) 尤其乐观，预计如果它能保持 Small 的性能与规模之比，将超越竞争对手。此外，与大得多的模型相比，用户在逻辑和写作方面更倾向于 Mistral 生成的输出。
    - 据报道，Mistral 3.2 的 Tool calling 和日期检索在许多量化版本中已损坏，但社区已提供修复方案，包括托管在 HuggingFace 上的动态量化模型，详见 https://huggingface.co/unsloth/Mistral-Small-3.2-24B-Instruct-2506-GGUF。
    - 基准测试对比指出，Mistral Small 3.2 在某些测试中优于 Gemma 3 27B，且用户认为它在写作和逻辑等任务中超越了 Llama 3.3 70B 和 Qwen2.5 72B，尽管其参数量显著更小（`24B` 对比 `70B/72B`）。一些人表示期待未来的 "Mistral Medium" 模型如果能保持类似的单位参数性能比，可能会进一步颠覆这一领域。
    - 为了让 Mistral Small 3.2 输出最佳结果，推荐的推理设置是极低温度（`0.15`）、关闭重复惩罚（`1.0`）、关闭最小 P 采样（`0.0`）以及将 top P 采样设为最大值（`1.0`）。尽管通用智能很强，但与被描述为“思考模型”的 Qwen3 30B 等模型相比，其在分步推理方面的局限性仍然明显。

### 2. Gemini CLI 工具免费层级发布与讨论

- [**Gemini 发布了一款开源 CLI 工具，类似于 Claude Code，但提供免费的 1 million token context window，每分钟 60 model requests，每天 1,000 requests，完全免费。**](https://i.redd.it/11rgwmzvv39f1.jpeg) ([Score: 430, Comments: 88](https://www.reddit.com/r/LocalLLaMA/comments/1lkbiva/gemini_released_an_open_source_cli_tool_similar/)): **该图片直观地展示了 Google Gemini 发布的一款开源 CLI 工具，如帖子所述，该工具专注于开发者和代码探索。从技术角度看，该 CLI 提供了高达** `1 million token context window`**，每分钟最高** `60 model requests per minute` **以及每天** `1,000 requests per day` **的免费额度。这使其成为 Claude Code 等工具的高容量、零成本替代方案，但与完全开放的工具不同，它目前需要使用专有的 Gemini API。数据收集用于训练是一个显著话题，官方 [privacy terms](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals) 允许记录 prompt/代码并进行人工审核以改进模型，尽管用户可以选择退出，且据称收集的数据与其 Google 账号分离。** 评论者讨论了 Google 免费提供如此强大工具的影响，普遍认为这是为了收集多样化的训练数据，一位用户指出有退出数据收集的选项。一些人对使用绑定到专有、可能受速率限制的 API 的工具表示保留，并寻求支持本地模型使用的 fork 版本。主要的技术讨论集中在隐私、数据所有权以及实际限制与开源自主权之间的博弈。
    - Google 的 Gemini Code Assist CLI 工具是开源的，并提供 1 million token context window 以及慷慨的免费层级使用限制（`60 requests/minute`，`1,000/day`）。然而，使用该工具需要 Gemini 云端 API，这意味着所有交互都会经过 Google 的基础设施，并受其速率限制和数据收集政策的约束。
    - Gemini Code Assist 的 [privacy notice](https://developers.google.com/gemini-code-assist/resources/privacy-notice-gemini-code-assist-individuals) 规定，Google 会收集 prompt、代码、输出、代码编辑和反馈，以改进服务和机器学习模型，人工审核员可能会对这些数据进行标注。虽然据称数据与你的 Google 账号分离，但除非你选择退出，否则它仍会被用于模型训练。
    - 由于需要使用专有的 Gemini API 以及与模糊或意外计费做法相关的风险，一些用户表示犹豫，并引用了在广告宣传的免费使用期间出现意外高额费用的案例。这导致了对支持本地推理和开放模型兼容性的 fork 版本的呼声，以避免此类供应商锁定和隐私问题。
- [**Gemini CLI：你的开源 AI agent**](https://blog.google/technology/developers/introducing-gemini-cli/) ([Score: 126, Comments: 30](https://www.reddit.com/r/LocalLLaMA/comments/1ljxa2e/gemini_cli_your_opensource_ai_agent/)): **Google 推出了开源的 Gemini CLI，使用户能够通过个人 Google 账号在终端直接与 Gemini AI 模型（包括 Gemini 2.5 Pro）进行交互。该产品提供了丰厚的免费层级：'1 million token context window'、'60 model requests/minute' 和 '1,000 requests/day'，使其适合高强度的开发使用（官方详情存档于 [此处](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/)）。鉴于该 CLI 的开源性质，技术用户对集成本地模型表现出了浓厚兴趣。** 一条评论质疑了如此慷慨的使用限制的可行性和可持续性，在进一步核实前持怀疑态度。另一项咨询询问开源是否能让开发者替换为本地模型，暗示了潜在的可扩展性和本地部署兴趣。
    - 据报道，Google 的 Gemini CLI 提供了对 Gemini 2.5 Pro 的访问，具有异常庞大的 1 million token context window，并在预览期间允许每分钟最多 60 次请求和每天 1,000 次免费请求，用户指出与行业规范相比，这是一个非常慷慨的限制。
    - 讨论提出了技术问题，即 Gemini CLI 的“开源”状态是否允许用户在 Google 自家 Gemini 模型之外插入并运行本地模型，但由于官方帖子和 GitHub 仓库均被删除，目前尚不确定。
    - 官方公告（目前仅能通过存档访问）和 GitHub 仓库的消失引起了关注，暗示发布可能存在问题；一些用户引用了带有截图的存档和替代来源进行记录，而 Google 似乎删除了该项目，可能是为了重新调整或撤回发布。

### 3. MCP 功能集成与创新 LLM 技术 (LM Studio & ThermoAsk)

- [**LM Studio 现在支持 MCP！**](https://www.reddit.com/r/LocalLLaMA/comments/1lkc5mr/lm_studio_now_supports_mcp/) ([Score: 199, Comments: 23](https://www.reddit.com/r/LocalLLaMA/comments/1lkc5mr/lm_studio_now_supports_mcp/)): **LM Studio 宣布支持 MCP (Model Compatibility Protocol)，实现了 LM Studio 与更广泛的本地 LLM 及服务工具之间的无缝互操作性，详见其 [官方博客文章](https://lmstudio.ai/blog/mcp)。此次 MCP 集成旨在简化加载、微调和模型管理工作流，可能主要针对依赖自定义本地模型编排的开发者。** 一位用户报告在搜索模型时遇到错误，表明当前 MCP 实现中的模型发现功能可能存在不稳定性或局限性。另一条评论强调了此次更新对于此前依赖不可靠自定义解决方案的用户的重要性，显示出社区对这种兼容性的强烈需求。
    - 一位用户报告在尝试加载或搜索 LM Studio 内的模型列表时出现错误，表明在新 MCP 支持下，界面或后端处理模型仓库时可能存在问题。
    - 几位用户提到在 beta 频道成功使用了新的 Multimodal Control Protocol (MCP) 支持，这表明该实现在至少某些用例中是稳定的，但仍存在一些 UI/可发现性问题（例如，无法在设置中找到功能或访问模型列表）。
- [**ThermoAsk：让 LLM 自行设置其 Temperature**](https://i.redd.it/t8az5arc1z8f1.png) ([Score: 101, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1ljs95d/thermoask_getting_an_llm_to_set_its_own/)): **该图片是配合文中讨论的技术理念的隐喻插图：使语言模型 (LLM) 能够动态设置其自身的采样 Temperature（“ThermoAsk”）。发光的熔炉和重复的徽标在视觉上强化了 LLM 主动控制其“热度”（采样随机性/创造力）的概念，类似于自然语言生成中的 Temperature。该帖子介绍了一种新技术，并提供了使用 Ollama 的 Python SDK 和 Qwen2.5-7B 的实现，其中 LLM 根据任务需求确定 Temperature。[博客文章](http://amanvir.com/blog/getting-an-llm-to-set-its-own-temperature) 和 [GitHub 仓库](https://github.com/amanvirparhar/thermoask) 详细介绍了该方法。** 评论讨论解决了幻觉的技术挑战，建议使用第二个模型作为质量评估的仲裁者，并提出了关于 random seeds 可复现性的问题。用户强烈鼓励向 LM Studio 及相关工具社区提议此方案，凸显了用户对更广泛采用和实验验证的兴趣。
    - 一位用户提出了关于控制幻觉的重要技术问题，具体询问是否可以使用更高质量的次级 Dense 模型作为独立仲裁者来评估输出。该建议强调了对模型自我评分的担忧，并提议如果评估任务较轻，可以将其卸载到适合 CPU/RAM 环境的模型中。
    - 另一位评论者以兼容 OpenAI 的方式实现了这一想法，使该方法可用于任何 UI/LLM 设置，并提供了[指向其详细介绍实现的 Reddit 帖子的链接](https://www.reddit.com/r/LocalLLaMA/comments/1lkixss/getting_an_llm_to_set_its_own_temperature/)。

## 非技术性 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. Anthropic 图书扫描争议与合理使用裁决

- [**Anthropic 购买了数百万本实体印刷书，并将其数字化扫描以用于 Claude**](https://www.reddit.com/r/singularity/comments/1ljs8np/anthropic_purchased_millions_of_physical_print/) ([Score: 716, Comments: 99](https://www.reddit.com/r/singularity/comments/1ljs8np/anthropic_purchased_millions_of_physical_print/)): **最近的一项法院裁决显示，在 Tom Turvey（曾任职于 Google 图书扫描项目）的指导下，Anthropic 动用了数百万美元的预算购买了数百万本印刷书，旨在为模型训练创建一个专有的数字语料库。这些书被物理拆解并扫描，服务商为每本书生成了高质量的图像 PDF 和 OCR 识别文本，随后丢弃了原件。完整的技术背景和法律证据可在 [32页裁决书 PDF](https://www.documentcloud.org/documents/25982181-authors-v-anthropic-ruling/) 中查看，更多分析见 [Simon Willison 的文章](https://simonwillison.net/2025/Jun/24/anthropic-training/)。** 评论中的技术讨论推测了使用实体书的原因（可能是为了法律/审计追踪或成本效率），并假设这个独特、高保真的语料库可能是 Claude 相比其他模型具有更出色创意写作能力的原因。
    - 一位评论者推测，Anthropic 购买并扫描印刷书的原因可能不仅仅是为了降低版权风险，还包括成本考量，或者是在数字副本授权更为复杂或昂贵的情况下，为法律或透明度目的确保有清晰的获取记录（“纸面证据”）。
    - 另一位用户指出，Claude 在创意写作方面的声誉可能与 Anthropic 直接摄取高质量图书数据有关，这表明大量基于图书的训练语料库可能会增强模型在叙事连贯性和风格多样性方面的表现，使其区别于仅限于互联网文本训练的模型。
    - 讨论还与 Bookshare 进行了技术对比，据报道，Bookshare 通过 3000 万美元的政府资助，以更大的规模和更快的速度实现了类似的图书数字化工作，这表明大规模图书扫描是一个已解决的问题，并突显了 Anthropic 方法中潜在的效率差距。
- [**联邦法官裁定 Anthropic 使用图书训练 Claude 属于合理使用，根据美国版权法是合法的**](https://www.reddit.com/r/ClaudeAI/comments/1ljs3mj/a_federal_judge_has_ruled_that_anthropics_use_of/) ([Score: 158, Comments: 60](https://www.reddit.com/r/ClaudeAI/comments/1ljs3mj/a_federal_judge_has_ruled_that_anthropics_use_of/)): **一位联邦法官裁定，Anthropic 使用合法购买的图书来训练其 Claude 语言模型构成了美国版权法下的合理使用（Fair Use），并强调训练过程具有高度的转换性（参见完整裁决：https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf）。然而，法官认定 Anthropic 在训练后保留 700 万本盗版电子书不符合合理使用标准；这部分侵权的损害赔偿将由陪审团决定。** 评论者辩论了使用合法获取内容与盗版材料之间的伦理和法律区别，一些人强调转换性使用的社会效益，另一些人则批评利用未经授权的副本获利的行为。
    - SeanBannister 强调了裁决中的一个关键细微差别：虽然法官认为出于训练目的购买图书因其转换性质属于合理使用，但在训练后“永久保留 700 万本盗版电子书”则*不*属于合理使用。这开创了一个先例，即 AI 数据使用的合法性取决于获取方式和数据保留政策。接下来的法律步骤涉及通过陪审团确定未购买图书的损害赔偿，这表明依赖侵权数据源的 AI 开发商面临巨大的法律风险。完整裁决：https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_3.pdf

### 2. Google Gemini CLI 和端侧 Gemini 更新

- [**Google 推出 Gemini CLI，这是一个轻量级开源 AI Agent，可将 Gemini 直接引入终端**](https://v.redd.it/hbt1631zp29f1) ([Score: 500, Comments: 75](https://www.reddit.com/r/singularity/comments/1lk5h19/google_introduces_gemini_cli_a_light_opensource/)): **Google 发布了开源的 Gemini CLI，这是一个旨在将 Gemini 的能力直接带入终端的 AI Agent，源代码可在 [GitHub](https://github.com/google-gemini/gemini-cli/) 上获取。该 CLI 支持从终端进行代码库探索和代码修改，但早期用户反馈指出其搜索时间较长且代码导航不一致，部分用户将其与 Anthropic 的 Claude Code 进行比较，并推测其核心差异极小。** 技术评论者辩论了 Gemini CLI 相对于 Claude Code 的有效性，批评 Gemini 的代码搜索性能较慢，且指令遵循/工具调用能力较弱；有人怀疑这两个工具之间的区别仅在于表面变化。
    - 多位用户指出 Gemini CLI 的界面和方法让人强烈联想到 Claude Code，并指责其除了字符串和颜色的更改外几乎没有其他变化；对于与竞争对手相比的真正创新持怀疑态度。
    - 一位测试过 Gemini CLI 的用户报告称，其代码库搜索功能明显比 Claude Code 慢且不可靠，索引一个简单任务需要几分钟，最终还未能找到非注释代码。
    - 存在关于 Gemini 在指令遵循和工具调用能力的性能担忧，一些用户表示，在 Agent 工作流自动化的这些关键领域， Gemini 历来落后于 Anthropic 的 Claude Code。
- [**Gemini CLI：每分钟 60 次模型请求，每天 1,000 次免费请求。100 万 Context Window**](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/) ([Score: 404, Comments: 74](https://www.reddit.com/r/singularity/comments/1ljxou6/gemini_cli_60_model_requests_per_minute_and_1000/)): **Google 推出了开源的 Gemini CLI，支持通过终端访问具有 1M-token Context Window 的 Gemini AI 模型，免费层级为** `60 requests/min` **或** `1,000 requests/day`**，并全面支持可扩展的工作流和集成。该 CLI 旨在提高开发者的生产力和实验效率，其巨大的 Context Window 在 Agent 和子 Agent 场景中可能比竞争模型更具优势（[已存档的公告](https://web.archive.org/web/20250625051706/https://blog.google/technology/developers/introducing-gemini-cli/)）。** 评论者正在将 Gemini 2.5 Pro 与 Claude Opus 进行比较，强调虽然 Opus 可能更“聪明”，但它存在上下文限制和指令失效的问题。Gemini CLI 中庞大的上下文和低成本生成的子 Agent 被认为具有潜在的“Meta Shifting”意义。用户还研究并解决了在 IDE（如 Visual Studio）中的技术使用问题。
    - 一位用户指出，虽然 **Claude Opus** 通常被认为更智能，但它往往会“遗忘、自负、公然违反指令，且上下文很快就会耗尽”，这暗示了其在 Agent 使用场景中的重大实际局限。相比之下，**Gemini 2.5 Pro** 的 `1 million token context` 限制和低成本的子 Agent 生成可能是构建复杂 Agent 系统或代码助手的“Meta Shifting”优势。
    - 讨论强调了 Gemini CLI 大 Context Window（100 万 Token）潜在的 Meta Shifting 影响，特别是与 Claude 等竞争对手相比。增加的上下文大小可以实现更高级的 Agent 工作流，这些工作流需要长期记忆或多个子 Agent 之间的协调，而不会遇到过高的成本或上下文碎片化问题。
    - 开发者对在工作流中集成 Gemini CLI 工具表现出技术兴趣，特别提到了将 Gemini 与 Claude Code 结合，让它们“相互对话”，从而可能同时利用两种模型的独特优势或上下文来进行高级代码生成或自动化流水线。

### 3. AI 模型能力与 Benchmark 进展

- [**Humanity's Last Exam 随时间变化的得分**](https://i.redd.it/pdf5hf5a809f1.png) ([得分: 252, 评论: 41](https://www.reddit.com/r/singularity/comments/1ljwxgy/humanitys_last_exam_scores_over_time/)): **该图像是一张图表，描绘了 AI 模型在 2024 年 4 月至 2025 年 6 月期间在 Benchmark 'Humanity's Last Exam' 上的进展，得分百分比从 0% 到 30% 不等。值得注意的是，Deep Research 在 2 月份引领了改进，但截至发帖时，Moonshot AI 的 Kimi-Researcher 实现了创纪录的 26.9% 的 'pass@1' 得分（高于最初的 8.6%），这归功于其平均 23 个推理步骤并在每个任务中探索 200 多个 URL 的策略。GPT-4o 和 Claude 3.5 Sonnet 等主流模型也被追踪，显示出性能持续上升的趋势。** 评论者表达了对 Kimi-Researcher 的陌生感（“从未听说过 Kimi Researcher”），对其起源的好奇，以及对 GPT-4o 等当前模型得分对比的兴趣。
    - 提出了一个关于 GPT-4o (o3) 在 Humanity's Last Exam 上的校准误差（calibration error）百分比的技术点：其校准误差*显著低于*其他顶级模型，这是理想的，因为校准误差反映了模型对其答案的置信度与其正确性的匹配程度——较低的值表示模型在错误答案上不那么过度自信。
    - 评论者注意到遗漏了某些先进模型，特别是 Claude 4 Opus Research 和 Gemini 2.5 Pro Deep Research，认为这些模型将为 Humanity's Last Exam 等 Benchmark 提供有价值的对比数据。
- [**AlphaGenome：用于更好理解基因组的 AI**](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) ([得分: 336, 评论: 66](https://www.reddit.com/r/singularity/comments/1lk6l28/alphagenome_ai_for_better_understanding_the_genome/)): **来自 DeepMind 的 AlphaGenome 是一款新型 AI 基因组学模型，能够处理高达 1Mbp 的输入 DNA，并针对转录、剪接和变异的调节属性进行 1bp 分辨率的预测。它通过混合卷积+Transformer 架构实现了这一目标，在变异评分和剪接建模 Benchmark 上产生了 state-of-the-art 的性能，且比 Enformer 等之前的模型需要更少的计算量。值得注意的是，AlphaGenome 的架构缓解了平衡长程上下文（例如，距离 >500kb 的调节增强子）与精细单碱基分辨率这一反复出现的痛点，首次整合了这些尺度。核心任务的 R-value 性能约为 0.8-0.85，突显了改进，但也揭示了由于未解决的生物随机性而导致的根本限制。** 线程中的技术讨论赞赏该模型推进了统一远端和局部调节预测的能力，认为它是一个务实的基础科学工具，而非诊断灵丹妙药。关键辩论集中在生物复杂性和数据带来的限制，以及 AlphaGenome 在促进基因组学假设生成和实验设计而非确定性预测方面的效用。
    - AlphaGenome 的主要技术贡献在于其处理整整 1Mbp 基因组序列的能力，同时以单碱基对（1bp）分辨率输出预测，有效地弥合了大规模基因组上下文（如数百 kb 之外的远端增强子）与细粒度调节元件（如单个转录因子结合位点）之间的典型权衡。这被描述为一项重大的工程成就，而非全新的科学发现，它将现有想法整合到一个更统一且有效的框架中。
    - AlphaGenome 的结果虽然强劲，但并未达到完美的预测能力：报告的 R-value 约为 0.8 到 0.85（而非 0.99），反映了基因调节固有的复杂性和随机性，类似于天气预报中的混沌理论。这突显了由于生物和数据复杂性，在全面预测和解释基因组功能方面仍然存在的局限性。
    - 该模型的直接实际用途在于转化研究流程：AlphaGenome 通过过滤噪声并优先处理因果变异和生物机制以进行湿实验（wet-lab）验证（例如，推断非编码变异破坏了特定细胞类型中的特定染色质环（chromatin loop）），帮助研究人员解释来自 GWAS 研究的统计信号。这减少了猜测，并加速了实验后续工作的假设生成。

---

# AI Discord 回顾

> 由 Gemini 2.5 Pro Preview 提供的摘要之摘要的摘要
> 

**主题 1：突破性的 AI 发布和功能增强**

- **Google & Anthropic 释放开发者新动能**：Google 推出了开源的 **Gemini CLI Agent**，由 **Gemini 2.5 Pro** 驱动并支持 MCPs（[Gemini CLI Agent 视频展示](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4)）；与此同时，Anthropic 首次推出了 **Artifacts 和 Artifacts Gallery**，允许用户在 Claude 内部构建 Claude（[Anthropic Artifacts 视频演示](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4)）。这些工具旨在增强开发者与强大 AI 模型的交互。
- **OpenRouter 提升透明度与控制力**：OpenRouter [通过 X.com 发布了新的模型运行时间 API](https://x.com/OpenRouterAI/status/1937869909448441980)，供开发者监控模型可用性，并增强了其 **Bring Your Own Key (BYOK)** 功能，增加了预存密钥测试和使用限制功能（[X.com 上的 BYOK 改进详情](https://x.com/OpenRouterAI/status/1937872903988535400)）。这些更新为开发者提供了对其 AI 模型使用情况更强的可见性和管理能力。
- **MCP 集成随 LM Studio 和 LlamaIndex 扩展**：**LM Studio** 的 [0.3.17 版本发布博客](https://lmstudio.ai/blog/lmstudio-v0.3.17) 宣布了 **MCP Host** 功能（[LM Studio MCP Host 文档](https://lmstudio.ai/docs/app/plugins/mcp)），允许本地 LLM 连接；而 **LlamaIndex** 发布了一个开源模板，用于将 **Claude 兼容的 MCP server** 构建为 Next.js 应用。这些进展拓宽了 **Model Context Protocol** 的生态系统。

**主题 2：模型乱象：性能怪癖、Bug 与基准测试**

- **模型在输出和 Token 限制上遇到障碍**：用户发现 **GPT 4.1 mini** 尽管拥有 **33k token** 的容量，但在 **3800 tokens** 时就会截断输出，并向 JSON 中添加多余字符；同时据报道，**OpenRouter** 供应商存在误报最大输出 tokens 的情况，阻碍了 LLM 的推理任务。这些问题凸显了在不同平台上实现可靠且可预测的模型输出所面临的持续挑战。
- **Cursor 苦于上下文处理，Deepseek 表现下滑**：**Cursor** 用户讨论了针对超过上下文长度的对话进行自动摘要的问题，这可能导致内容丢失；并从 [Cursor 的上下文管理文档](https://docs.cursor.com/context/management) 中注意到 Gemini 处理大上下文的能力优于 Claude。此外还观察到 **Deepseek** 模型在处理上下文时表现不佳，导致 Cursor 将其上下文长度缩减至 **60k tokens** 左右。
- **LLMs 面临逻辑测试，新基准测试涌现**：工程师们探索使用基于逻辑谬误的问题来挑战 LLMs，灵感源自 [维基百科关于哥德尔不完备定理 (Gödel’s incompleteness theorems) 的条目](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems)，以评估其超越模仿的理解能力。另外，[Artificial Analysis MiniMax 基准测试页面](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price) 因其在智能与价格评估中的定位而受到关注，其中涉及了如 [arXiv 上详述的 Multi-head latent attention](https://arxiv.org/pdf/2206.04615) 等技术。

**主题 3：不断演进的开发前沿：GPU、工具与语言**

- **Unsloth 支持 Intel XPU 且廉价 GPU 访问升温**：Unsloth 通过 [Intel XPU 支持的 Unsloth 提交](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a) 宣布支持 Intel XPU，随着 Intel 即将推出 **低于 1000 美元的 48GB GPU**，这一举措被认为意义重大。同时，用户推荐了 [Hyperbolic XYZ GPU 租赁平台](https://app.hyperbolic.xyz/)，其 **H100** 租赁价格低至 **0.99 美元/小时**。
- **MakoGenerate 自动化 Kernel 创建 & GPU Mode 发布 Trimul Benchmark**：**MakoGenerate** 在其 [MakoGenerate 平台网站](https://generate.mako.dev/) 上线，这是一个用于生成可部署到 **H100** 或 **B200** 的 GPU Kernel 的 AI Agent，其 VS Code 扩展正在开发中。与此同时，GPU Mode 为 NVIDIA 和 AMD 硬件引入了 **Triangle Multiplicative Update (Trimul)** Benchmark（[GPU Mode Trimul Benchmark 详情](https://tinyurl.com/gpumode-trimul)），引发了 Kernel 优化的竞争。
- **Mojo 挑战 Rust Async 并引入 Effect Generics**：Mojo 团队旨在通过使用更好的 Async 运行时和 [Modular PR 中详述的 Linear Types](https://github.com/modular/modular/pull/3946) 来简化异步编程，以避免 Rust 中 `Arc<Mutex<T>>` 的复杂性。此外，Mojo 正在 [另一个 Modular PR](https://github.com/modular/modular/pull/4728) 中探索 **Effect Generics**，以解决大多数库的 Function Coloring 问题，允许编译器选择最佳的 I/O API。

**主题 4：AI 的社会镜像：伦理、版权与内容的未来**

- **AI “真相”与审查引发激烈辩论**：成员们对 **Grok AI** 将知识重写为自身“真相”的使命表示担忧，担心政治洗脑，并辩论了 **中国 AI 模型** 中审查制度的有效性。讨论强调了对 AI 系统塑造叙事并遵守（或绕过）内容限制的焦虑，一些人指出 **Yi 模型** 在敏感话题上仍保持未审查状态。
- **版权之争愈演愈烈，Facebook 面临盗版裁决且数据需求增长**：[Adam Eisgrau 关于 Facebook 盗版诉讼的推文](https://x.com/AdamEisgrau/status/1937480346976813454) 表明，尽管在转换性训练（transformative training）方面获得了有利裁决，**Facebook** 可能在图书盗版诉讼的盗版层面败诉，这加剧了关于使用受版权保护材料的持续争论。尽管法律存在不确定性，社区对在训练数据集中利用版权作品的需求依然强劲，一些人主张建立付费系统。
- **Google 的 AI Web 愿景引发“互联网终结”的恐惧**：围绕 Google I/O 发布会的讨论（其中 **AI 将为其他 AI 抓取而编写网站**）引发了“Google 肯定在密谋终结互联网”的笑话。这凸显了人们的担忧，即 AI 生成的内容可能会降低人类创作的 Web 内容的价值，并从根本上改变互联网生态系统。

**主题 5：突破边界：前沿 AI 研究与技术**

- **BitNet 以速度和质量惊艳众人，Cerebras 提供廉价的大规模扩展**：测试 [Azure 上的 BitNet 演示](https://bitnet-demo.azurewebsites.net/)（也在 [Chat-with-Bitnet-b1.58-2B-4T HF Space](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T) 上）的用户表示，其速度和质量令人印象深刻，尤其是对于初始查询。在扩展方面，[Cerebras Cloud 信息页面](https://www.cerebras.ai/cloud) 强调其晶圆级 GPU 在大规模应用时 *极其便宜*，性能可与 Blackwell 媲美，但带宽较低。
- **Anthropic 质疑 RL 研究的严谨性 & Dr. GRPO 出现**：Anthropic 研究人员在 [Dwarkesh 播客](https://youtu.be/64lXQP6cs5M?t=550) 中指出，许多 **RL 论文** 使用较小的模型，这可能会扭曲对前沿模型的见解，并主张在像 DeepSeek 这样的大型模型上进行测试。同时，**Dr. GRPO**（[arXiv 上的 GRPO 论文](https://arxiv.org/abs/2501.12948)，[Discord 上的 Dr. GRPO 讨论](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368)）因在保持性能的同时减少 **GRPO** 的冗余输出（chattiness）而受到关注。
- **“Your Brain on LLMs”研究与 AlphaGenome 揭示新见解**：长达 **206 页** 的 [arXiv 论文 "Your Brain on LLMs"](https://arxiv.org/pdf/2506.08872v1) 尽管最初存在一些校对问题，但其关于人机认知交互的内容受到了称赞。另外，DeepMind 推出了 **AlphaGenome**，这是一个旨在增强基因组理解的 AI 系统，详见 [DeepMind 的 AlphaGenome 博客文章](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/)。

---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Fellowship 奖品仍是谜**：一名成员在达到 **25 名成员**后，寻求关于领取商务 Fellowship 奖励（马克杯、T恤、抽奖券）的明确说明。
   - 目前缺乏领取这些奖励的具体说明和细节。
- **Google 开源 Gemini CLI Agent**：Google 推出了开源的 **Gemini CLI Agent**，由 **Gemini 2.5 Pro** 驱动并支持 **MCPs**；[这段视频](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4)展示了它如何为开发者提供访问 **Gemini 2.5 Pro** 的权限，并支持多字符提示词（multi-character prompts）。
   - 它的目的是让开发者能够访问支持多字符提示词（**MCPs**）的 **Gemini 2.5 Pro**。
- **Anthropic 首次推出 Artifacts 和 Artifacts Gallery**：Anthropic 在 Web 端发布了 **AI 驱动的 Artifacts 和 Artifacts Gallery**，使用户能够在 Claude 内部构建 Claude，如[这段视频](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4)所示。
   - **Artifacts Gallery** 允许在 Claude 环境中进行实时协作和开发。
- **Imagen 4 悄然而至**：**Imagen 4 和 Imagen 4 Ultra** 现已在 AI Studio 和 API 上可用。
   - 一位成员测试了“一个在暴风雪中倒立的小丑”，[分享的图片](https://cdn.discordapp.com/attachments/1047649527299055688/1387374829744685088/wDnjFf5U1YUCQAAAABJRU5ErkJggg.png?ex=685dc5bf&is=685c743f&hm=4f1cc936f04925773b1328ffbcc229e48fa59a5ae4b74754963caf76c527079d&)显示效果还不尽如人意。
- **搜索域名过滤器触发幻觉**：成员们报告称，在 `pplx-api` 频道中，将搜索域名过滤器设置为新闻网站（如 reuters.com）现在会导致**文章幻觉**。
   - 用户反映他们无法获得有效结果或引用数组（citations arrays），这令人感到沮丧。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **上下文危机困扰 Cursor**：成员们讨论了 **Cursor** 自动总结超过上下文长度的聊天窗口，导致内容丢失和用户困惑，并指出 [Gemini 处理大上下文的能力优于 Claude](https://docs.cursor.com/context/management)。
   - 进一步指出，**Deepseek** 模型在理解上下文方面表现最差，导致 **Cursor** 将其上下文长度减少到 **60k tokens** 左右。
- **速率限制传闻困扰 Pro 用户**：一些 **Pro** 用户遇到了不同的速率限制体验，不得不求助于简短的提示词且不附加文件以避免触及限制，而其他人则建议 [Pro+ 计划的潜在价值](https://cursor.com/pricing)。
   - 对于速率限制缺乏透明度的挫败感正在增加，这使得预测使用量和有效规划变得困难，一位用户表示：*Pro 的新现状是付了费，但还得通过深思熟虑的聊天消息来考虑 token 使用情况，笑死*。
- **Gemini CLI 的宏伟尝试折戟**：测试新 **Gemini CLI** 的用户发现它存在 Bug 且尚未准备好发布，报告称在执行 `npm run dev` 时出现冻结以及无法安装 **Nuxt**。
   - 尽管每天提供 **1000 次请求**，但该服务被认为非常缓慢且已损坏，甚至有人开玩笑说 [它可能会在代码库中包含广告](https://blog.google/technology/developers/introducing-gemini-cli/)。
- **Background Agents 的密钥保持私密**：用户现在可以在 **Cursor Settings >> Background Agents >> Secrets** 中配置 **Background Agents** 的密钥，无需将其推送到 `environment.json`。
   - 这使 Agent 能够根据需要使用这些密钥。
- **Git 远程 URL 问题导致 Agent 出错**：一位用户发现，由于存在对有效 `github.com` URL 的检查，带有前导 `www` 的本地仓库 URL 会导致 **Background Agents** 出现问题。
   - Agent 会运行 `git remote get-url origin` 并检查该 URL 是否为 github.com URL。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek 对算力的巨大需求引发了对更小模型的搜索**：成员们正在寻找 [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 的更小版本，因为即使是 **1-bit 版本**对于某些 GPU 来说也太大了，这引发了围绕 **Qwen-3** 等替代方案的讨论。
   - 讨论围绕 **Qwen-3** 与非 Qwen 模型的比较，以及是否应该尝试 **DeepSeek-R1-0528-Qwen3-8B** 展开。
- **确定性之梦：追求 100% 可预测的 AI**：考虑了创建一个能 **输出 100% 确定性结果** 的模型的潜力，建议将 temperature 设置为 0 并进行微调。
   - 一位成员指出，用概率函数实现确定性是有缺陷的，而其他人则强调了随机性的实用性以及实现 **100% 确定性** 的难度。
- **Unsloth 拥抱 Intel XPU：廉价 GPU 的繁荣？**：根据 [此 commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a)，Unsloth 现在支持 Intel XPU。
   - 随着今年晚些时候即将发布的 **价格低于 1000 美元的 48GB GPU**，这被预期将是一个**重大的进展**。
- **MacBook Pro 过热：冷却方案探讨**：讨论围绕 NN 训练期间 MacBook Pro 的冷却展开，建议使用 **铝制支架**、**带风扇的支架** 以及重新涂抹 **导热膏** 等解决方案。
   - 成员们建议将 GPU 温度保持在 **最高 90-95C**，并建议不要使用冰或水进行冷却。
- **Hyperbolic XYZ 上的 H100 廉价租赁**：一位用户推荐了 [Hyperbolic XYZ](https://app.hyperbolic.xyz/)，用于以每小时 **0.99 美元** 租赁 **H100**，以及每小时 **0.28 美元** 租赁 **RTX 4090**，并附带了一个推荐码。
   - 该信息分享在 `#help` 频道中。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT 助力获得 Web 开发工作**：一位成员报告说 **GPT** 协助他们获得了一份 Web 开发人员的工作，引发了关于 **AI 对就业影响** 的讨论。
   - 该成员认为，**AI 对工作** 的最初影响并不是 AI 取代人类，而是 *拥有互联网连接的弱势群体* 获得了机会。
- **O3 和 Pro 解锁云端搜索连接器**：**聊天搜索连接器** 已于 **2025 年 6 月 24 日** 为 Pro 用户推出，集成了 **Dropbox**、**Box**、**Google Drive** 和 **Microsoft OneDrive/SharePoint** 等服务。
   - 此功能仅限于 **EEA**、**瑞士**和**英国**以外的用户，使用户能够使用其同步数据训练 AI 模型。
- **AI 引发艺术辩论**：一位成员发起了关于 AI 在艺术中角色的对话，重点在于使用 AI 进行概念或模型设计，与将纯 AI 生成的物品作为个人原创作品出售之间的区别。
   - 争论的核心在于将 **AI 生成的艺术** 作为个人创作展示的伦理问题，强调了在创作过程中承认工具参与的重要性。
- **逻辑陷阱暴露 LLM 弱点**：成员们探索使用基于逻辑谬误的问题来挑战 LLM，旨在评估它们识别并适当回应荒谬输入的能力，灵感来自 [哥德尔不完备定理 (Gödel’s incompleteness theorem)](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems)。
   - 小组共识认为，未能识别此类陷阱表明其理解力存在缺陷，而不仅仅是模仿，这表明其缺乏对逻辑的真正理解。
- **Minimax 基准测试近在眼前**：一位用户强调了 [MiniMax 基准测试](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price)，它被显眼地放置在智能与价格象限中。
   - 该基准测试包括 [Multi-head latent attention](https://arxiv.org/pdf/2206.04615)，尽管有些人将其斥为 *技术黑话*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **BitNet Demo 演示版质量惊人**：测试 [BitNet demo](https://bitnet-demo.azurewebsites.net/) 的用户反馈其速度和质量令人印象深刻，尤其是在初始查询方面。
   - 该项目作为 HF Space ([Chat-with-Bitnet-b1.58-2B-4T](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T)) 提供，允许通过编程方式访问 **BitNet** 的功能。
- **Cerebras Cloud 提供高性价比的扩展方案**：成员们发现 [Cerebras](https://www.cerebras.ai/cloud) 的晶圆级 GPU 在大规模应用下*非常便宜*，性能可与 **Blackwell** 媲美，但带宽较低。
   - 另一位成员指出 **Groq** 拥有*非常出色的技术*，非常适合大规模推理。
- **LLM 开拓着色器图（Shader Graph）前沿**：成员们探讨了使用 **LLM** 生成**着色器图代码**（可转换为 **HLSL** 或 **GLSL**），以及研究人员如何利用**语言模型**对其进行优化。
   - 一位用户提到 **Nvidia** 正在通过预测像素的小型模型探索**神经材料（neural materials）**。
- **ModernBERT 剖析重点关注输入嵌入**：一位成员关于**令牌输入嵌入（token input embeddings）**和 **ModernBERT** 的*梯度下降*文章被 [LessWrong.com](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert) 采纳。
   - 该 **LessWrong** 文章深入探讨了 **ModernBERT** 架构，重点关注梯度下降如何应用于令牌输入嵌入。
- **RAG 嵌入器集合获得 SmartTaskTool 升级**：一位成员分享了 **RAG 嵌入器集合** ([Hugging Face 链接](https://huggingface.co/kalle07/embedder_collection)) 和一个适用于 Windows 的**小型任务工具栏** ([Hugging Face 链接](https://huggingface.co/kalle07/SmartTaskTool))。
   - **SmartTaskTool** 是一个任务栏图标，现在包含跨语言支持 (en-de-fr-roberta)。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Grok AI 面临洗脑指控**：成员们担心 **Grok AI** 将知识改写为自身“真相”的使命构成了政治洗脑。
   - 普遍观点认为，具有说服力的 LLM 不应致力于强化政治叙事。
- **中国 AI 的审查引发辩论**：关于**中国 AI 模型**审查有效性的辩论兴起，有人声称模型只是使用了外部过滤器，而另一些人则表示模型是自愿遵守法律。
   - 成员们强调，**Yi 模型**（零一万物）在涉及天安门广场等敏感话题时仍未受审查。
- **Gemini 2.5 Pro 被黑马反超？**：**Gemini 2.5 Pro** 在排行榜上表现下滑，归因于 **Blacktooth** 和 **Stonebloom** 等匿名模型的崛起。
   - 虽然有人推测这些模型在 **Gemini 2.5 Pro** 表现不佳的领域表现出色，但也有人认为投票分布的变化才是原因。
- **LM Arena 排行榜易受提示词攻击**：一名用户声称发现了 **4 种方法**可以从 lmarena.ai 提取排行榜数据，引发了伦理和法律讨论。
   - 这些方法包括利用 Hugging Face Space、现有数据转储、网页抓取和浏览器扩展，但一位社区成员表示所给出的 **3/4** 方法是无效的。
- **开源社区对受版权保护内容的渴望**：尽管最近有法院裁决，但社区对在训练数据集利用受版权保护材料的需求依然强烈，尽管其合法性存在争议。
   - 关于近期裁决影响的观点各异，有人认为对持续训练没有影响，而另一些人则希望建立受版权保护数据的付费系统。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 新增 MCP Host 并支持多国语言**：[LM Studio 0.3.17 版本](https://lmstudio.ai/blog/lmstudio-v0.3.17) 现在支持 **MCP Host**，实现了与本地 LLM 的连接；得益于社区本地化者的贡献，目前已支持 **33 种语言**，并引入了 **'Solarized Dark'** 主题。
   - 有关 **MCP Host** 集成和功能的更多信息，请参阅 [LM Studio 文档](https://lmstudio.ai/docs/app/plugins/mcp)。
- **LM Studio 聊天消息消失后又重新出现**：一位用户报告称，在升级 **LM Studio** 后，之前来自“无审查”模型的*隐藏*对话变得可见，其中包括关于“制作虚假身份证件的流程是什么？”的交流。
   - 相关推测从流式传输设置到系统提示词不等，但隐藏对话出现的准确原因尚不清楚。
- **r/LocalLlama Subreddit 迎来新掌门人**：[r/LocalLlama](https://www.reddit.com/r/LocalLlama/) Subreddit 现由新团队管理，引发了关于新版主参与众多其他 Subreddit 的讨论。
   - 一些用户对该版主过广的管理范围表示担忧，而另一些用户则认为目前没有明显的异常迹象。
- **关注用于推理的 Runpod Serverless**：一位用户计划测试 **Runpod serverless**，特别是使用带有网络卷的 *flex workers*，以实现更快的模型加载和冷启动，用于 NVIDIA GPU 的推理任务。
   - 他们还在考虑未来使用 **Predibase** 及其 *turbo lora* 功能。
- **发现 Unsloth AI 的提交 (Commit)**：一位用户分享了 [Unsloth commit 的链接](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a)，这可能表明了对 **Unsloth AI 项目** 的关注或讨论。
   - 目前没有关于该提交内容或重要性的进一步细节。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出模型运行时间 (Uptime) API**：开发者现在可以通过 [OpenRouter API](https://x.com/OpenRouterAI/status/1937869909448441980) 监控模型运行时间，提供了模型可用性的透明度。
   - 此功能有助于更好地规划和管理依赖稳定模型性能的 AI 应用。
- **BYOK 功能增强**：**Bring Your Own Key (BYOK)** 用户现在可以在保存前测试密钥、限制上游使用量并在 API 调用中追踪使用情况，增强了控制力和安全性（[详情点击此处](https://x.com/OpenRouterAI/status/1937872903988535400)）。
   - 这些改进为 API 密钥管理和使用追踪提供了更细粒度的控制。
- **Midjourney 的视频项目让用户感到兴奋**：成员们称赞来自 **Midjourney** 和 **Spellbrush** 的新视频模型是 *i2v（图像转视频）领域的 ChatGPT 时刻*，并希望看到更多能推出 720p 分辨率的基础设施。
   - 虽然提到了 *seedance* 和 *hailuo* 等替代方案，但被认为在质量上明显逊色。
- **GPT 4.1 Mini 表现出输出异常**：**GPT 4.1 mini** 尽管拥有 **33k token** 的容量，但在 **3800 token** 处就会截断输出，并在 JSON 键之前添加 `\xa0`，导致集成问题。
   - 成员建议降低 temperature 并指定 `"response_format": {"type": "json_object" }` 以强制执行正确的 JSON 输出；其他人则发现 **GPT 3.5** 在某些任务上更可靠。
- **Veena 在印度语言语音领域取得成功**：在 OpenRouter 的协助下，一款名为 **Veena** 的新型**印度语言**语音 AI 模型已发布，详情见 [X.com](https://x.com/Dheemanthreddy_/status/1937839083281437021)。
   - 成员们对此次发布表示祝贺，这标志着本地语言 AI 支持迈出了重要一步。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **C++ 构建系统令人困惑**：成员们表达了对 **CMake** 的沮丧，有人表示为了让 CMake 与某些随机库协同工作而浪费的时间实在太多，并称 **Bazel** 是最伟大的（*goat*）构建系统。
   - 建议的替代方案包括 [Meson](https://mesonbuild.com/)、[Buck2](https://buck2.build/)、[xmake](https://xmake.io/#/) 和 [Zig](https://ziglang.org/)。
- **Triton 闭门造车，社区渴望交流**：一位成员指出 [Triton 不再对公众开放](https://discord.com/channels/1189498204333543425/1189607595451895918/1378126514318737541)，并重点介绍了 **Gluon**，这是 Triton 仓库中一个新的更高级别的 DSL，类似于 **Paszke 的 Mosaic**，并指向了 [test_core.py](https://github.com/triton-lang/triton/blob/5c9e54535dfe34d4b60fd13a4a27b6d74f3c8344/python/test/gluon/test_core.py)。
   - 一位成员对 **Triton 团队**停止与那些将 **Triton** 作为项目基础的人交流表示困惑，特别是关于**为什么 Triton 不支持 Windows** 的问题。
- **AMD 云服务选项丰富**：成员们推荐 [DigitalOcean](https://www.digitalocean.com/) 和 [TensorWave](https://www.tensorwave.com/) 作为 **AMD 机器**的优质云供应商，特别是对于小型项目和实验。
   - 其他提到的供应商包括 **Hotaisle** 和 **Runpod**，一位成员指出 Hotaisle *相当不错*。
- **MakoGenerate 进军 VS Code，LLM 的怪癖**：创作者宣布了 **MakoGenerate**，这是一个可以生成可部署到 **H100** 或 **B200** 的 GPU kernel 的 AI agent，并邀请大家在 [generate.mako.dev](https://generate.mako.dev) 提供反馈；同时确认他们已经在开发 **VS Code 扩展**并提供无限免费额度。
   - 用户注意到 **LLM** 有时会在提供的问题和 prompt 之间切换，即使被明确指示忽略示例问题，这使得让 LLM 按预期工作变得更加困难。
- **Trimul 任务在 NVIDIA 和 AMD 上取得成功**：宣布了一个基于 AlphaFold 系列模型中使用的 **Triangle Multiplicative Update** 的新问题，现已在 [GPU Mode](https://tinyurl.com/gpumode-trimul) 上提供，支持 **NVIDIA** 和 **AMD** 硬件。
   - 一位用户以 **7.92 ms** 的成绩获得了 **B200** `trimul` 排行榜的**第一名**，另一位用户以 **20.0 ms** 的成绩获得了 **A100** `trimul` 排行榜的**第一名**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dr. GRPO 减少了 GRPO 的啰嗦**：根据 [这段 Discord 讨论](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368)，成员们报告称 **Dr. GRPO** 在保持性能的同时减少了 **GRPO** 的啰嗦程度。
   - 引用了一个 [YouTube 视频](https://www.youtube.com/watch?v=K34gBCjzni8) 和 [论文](https://arxiv.org/abs/2501.12948) 来实现 **GRPO**，而 **Dr. GRPO** 是在此基础上构建的。
- **深入探讨前向传播（Forward Propagation）细节**：澄清了 **LeCun** 将 **Forward Propagation (FF-prop)** 定义为标准的前向推理过程，即层在训练后运行，没有反向传播。
   - 虽然 **Hinton 的 Forward Forward** 可能无法扩展，但 **Forward Gradients** 作为反向传播的转置可以有效地工作，是寻找导数最基本的方法。
- **AlphaGenome 崛起，照亮基因组**：DeepMind 推出了 **AlphaGenome**，这是一个旨在增强我们对基因组理解的 AI 系统，详情见[最近的博客文章](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/)。
   - 这一公告引发了 **ml-news** 频道成员之间的对话。
- **Brain-on-LLMs 的校对问题**：一位成员注意到了一项名为 *Your Brain on LLMs* 的**研究**，并分享了一张关于深色模式下字体和文本颜色不一致的[截图](https://cdn.discordapp.com/attachments/1045297868136779846/1387359102975610880/Snipaste_2025-06-25_16-09-14.png?ex=685db71a&is=685c659a&hm=9cfa39ed78e88f2ca714cf645ba04ec641289dd15a88077b4b4669245859e86a)。
   - 尽管最初视觉上感到震惊，但在阅读了一部分后，该成员评论说这篇 **206 页的论文** ([https://arxiv.org/pdf/2506.08872v1](https://arxiv.org/pdf/2506.08872v1)) *实际上相当不错*。
- **Google 推出 Gemini CLI**：Google 发布了 [Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/)，这是一个免费的开源 AI agent，将 Gemini 直接带到开发者的终端。
   - 该工具被宣传为向个人提供*无与伦比的访问权限*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Murati 的 Thinking Machines Lab 专注于面向业务的 RL**：根据[这篇文章](https://xcancel.com/steph_palazzolo/status/1937284120062706004)，Mira Murati 的新 AI 初创公司 **Thinking Machines Lab** 正专注于**面向业务的强化学习 (RL)**。
   - 尚未提供关于具体产品或发布日期的更多细节。
- **Warp 2.0 进入 Agentic 开发领域**：**Warp 2.0** 被介绍为一个 Agentic 开发环境，使开发者能够通过 *prompt 编程* 而非手动编码。根据[这条推文](https://xcancel.com/warpdotdev/status/1937525185843752969)，它在 **Terminal-Bench** 上排名第一，并在 **SWE-bench Verified** 上达到了 **71%**。
   - 这代表了向 AI 驱动的代码辅助和自动化的转变。
- **Airtable 的 Omni AI Agent 重塑应用平台**：根据[这条推文](https://xcancel.com/howietl/status/1937577526634987595)，**Airtable** 已重新发布为 **AI 原生应用平台**，通过 **Omni** 转向彻底的“重塑”。Omni 是一个 AI 应用构建 Agent，允许用户通过对话方式构建强大的应用。
   - 这展示了 AI Agent 越来越多地集成到应用开发工作流中。
- **Liquid AI 打造简洁的推理模型**：来自 **Liquid AI** 的 Maxime Labonne 宣布了一个 **10 亿参数的推理模型**，该模型既准确又简洁，结合了**监督微调 (SFT)** 和 **GRPO (Generative Reinforcement Learning from Human Preferences)**，详见[这条推文](https://xcancel.com/maximelabonne/status/1937819336204304692)。
   - 该模型旨在以相对较小的参数规模提供高效的推理能力。
- **OpenRouter 获得 AI 模型市场支持**：Deedy 宣布支持 **OpenRouter**，这是一个 AI 模型市场，通过单一 API 为开发者提供 **400 多个 LLM** 的访问权限，每年处理 **100 万亿个 token**，详见[这条推文](https://xcancel.com/deedydas/status/1937902948920811729)。
   - 该平台的规模表明，市场对通过统一接口访问多样化 AI 模型的需求巨大。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Facebook 在盗版诉讼中受挫**：一则[推文](https://x.com/AdamEisgrau/status/1937480346976813454)指出，尽管在转换性训练（transformative training）方面获得了有利裁决，但 **Facebook** 可能在图书盗版诉讼中败诉。
   - 该裁决对使用受版权保护材料训练 **LLM** 的影响正受到密切关注。
- **Nous 探讨录用 Yacine 的可能性**：成员们讨论了为什么 **Nous Research** 还没有雇佣前 **X** 工程师 **Yacine**，关于他是否适合 ML 职位的意见不一。
   - 一些成员质疑他的技能组合，而另一些成员则考虑他是否合适。
- **OpenRouter 输出 Token 限制引发不满**：用户报告称，许多 **OpenRouter** 提供商虚报了其最大输出 token 数，阻碍了推理 **LLM** 的性能。
   - **16k token** 的硬性限制导致无法运行大多数 **AIMS** 问题，但一些用户正通过选择能兑现承诺 token 限制的特定提供商来解决这一问题。
- **Anthropic 抨击 RL 研究现状**：**Anthropic** 研究员 **Sholto Douglas** 和 **Trenton Bricken** 在 [Dwarkesh 播客](https://youtu.be/64lXQP6cs5M?t=550)中指出，许多 **RL 论文** 使用较小的模型，这可能会扭曲前沿模型的动态。
   - 他们主张在最大的 **DeepSeek** 模型上进行实验以获得更具代表性的结果，并暗示当前的研究可能无法反映现实世界的性能。
- **Hermes 4 的体量与托管期望**：一名成员宣布，基于 **671b** 参数模型的 **Hermes 4** 预计将在下个月左右发布。
   - 另一名成员询问谁将托管 **Hermes 4**，并指出目前 OpenRouter 上 **Deepseek V3** 或 **R1** 的托管商通常速度慢、价格昂贵或不稳定。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 通过 Linear Types 避开 Rust Async 的复杂性**：Mojo 旨在通过使用更好的 async 运行时和 [linear types](https://github.com/modular/modular/pull/3946) 来解决 Rust 的 async 复杂性，从而避免使用 `Arc<Mutex<T>>` 等结构。
   - Mojo 寻求控制线程间的数据移动，并确保数据不会被提前释放，在倾向于 thread-per-core 的同时，可能提供可选的 work stealing。
- **Effect Generics 解决 Mojo 中的函数着色问题**：Mojo 正在探索 Effect generics，以解决大多数库的函数着色（function coloring）问题，详见此 [PR](https://github.com/modular/modular/pull/4728)。
   - 这种方法结合 effect generics，允许编译器/运行时为程序选择“最佳”的 IO API，除非涉及自定义 IO API 绑定。
- **困惑的错误消息困扰 Mojo Dictionaries**：一位 Mojo 新用户报告了在使用 **Dict struct** 时遇到的困惑错误消息，特别是关于在不带括号的情况下使用 `.value`、`.keys` 和 `.items`。
   - 错误消息 *"statements must start at the beginning of a line"* 被认为没有帮助，用户已被要求在 [GitHub](https://github.com/modular/modular) 上提交 issue，建议提供更具描述性的错误消息。
- **InlineArray Moveinit 需进一步检查**：**InlineArray** 在移动操作 (`b = a^`) 期间的行为受到质疑，有人担心元素的 copy 或 move 构造函数都没有被调用，这可能表明存在 bug。
   - 看来 **InlineArray** 在移动初始化期间执行的是位拷贝（bitwise copy），缺乏显式的 moveinit。
- **TorchScript 编译仍需 Torch**：用户意识到使用 **InferenceSession** 编译 **TorchScript** 文件仍需要 **Torch 环境**。
   - 他们对需要 **Torch** 依赖项表示沮丧。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **OpenAI API 出现运行时间问题**：成员报告称，由于 [OpenAI API 的问题](https://platform.openai.com/docs/status)，他们的**应用程序宕机**，收到了 `HTTP/1.1 404 Not Found` 错误。
   - 这表明请求的资源未找到，影响了应用程序的可用性。
- **SIMBA 错误解决方案已解锁**：成员分析了一个关于冻结子模块和预测器清单的 [SIMBA 错误](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/simba.py)。
   - 修复方法是确保从 `name_predictors` 返回的预测器与 `append_rule` 和 `append_demo` 期间迭代的预测器一致，特别是在使用 `._compiled = True` 时。
- **关于 Discord DSPy 标签的提议引发讨论**：一位成员提议创建一个 **Discord DSPy 标签**，以便在用户名旁展示 DSPy 专业知识。
   - 根据 [Discord's Guilds FAQ](https://support.discord.com/hc/en-us/articles/23187611406999-Guilds-FAQ)，实现此类标签至少需要对 Discord 服务器进行 **3 次 boost**。
- **dspy.Prediction 模式探讨**：一位成员询问从模块的 forward 方法返回 **dspy.Prediction** 以外的内容是否属于反模式（anti-pattern）。
   - 共识认为，如果 metric 函数不知道预期的输出是什么，可能会导致问题，从而影响优化。
- **Shopify 创始人助力 DSPy**：Shopify 创始人 [Tobi Lutke 加入 DSPy](https://x.com/tobi/status/1937967281599898005)。
   - 这一出人意料的举动凸显了该项目日益增长的重要性。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 的限制提示滞后导致工作丢失**：用户反馈称，**NotebookLM** 在自定义提示词（customize prompt）*之前*不会告知是否已达到生成限制，导致潜在的工作内容丢失。
   - 成员们想知道，当他们返回时，超长的自定义提示词是否会保留在 notebook 中。
- **Vimeo 在 NLM 中遇到访问障碍**：用户报告了在 **NotebookLM** 中将 **Vimeo** 视频作为来源时遇到的问题，安全功能阻止了内容访问。
   - 一位成员建议使用 [cobalt.tools](https://cobalt.tools/) 下载视频作为变通方案，而另一位成员则询问如果已经上传了转录文本，是否还需要视频本身。
- **AI 音频的 AdSense 获利模糊性**：一位用户询问 **YouTube** 是否允许使用 **AI 生成语音**以及来自 **NotebookLM** 内容的频道进行获利。
   - 另一位成员指出 AI 与版权之间存在*灰色地带*，并建议研究 YouTube 关于 **AI 内容获利**的规则。
- **NLM 高效处理首选 PDF 格式**：在一个消息线程中，用户询问 **PDF** 还是 **MD** 格式更适合 **NotebookLM**。
   - 另一位成员回答说 **PDF 是更好的格式**。
- **PrintFriendly 提供打印就绪页面**：一位用户识别出图片中的扩展程序为 **PrintFriendly**，并在 [Chrome Web Store](https://chromewebstore.google.com/detail/printfriendly-print-pdf-a/ohlencieiipommannpdfcmfdpjjmeolj) 中找到了它。
   - **PrintFriendly** 可将网页转换为适合打印的格式和 **PDF** 格式。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyGrad 重构悬赏吸引关注**：成员们对**重构悬赏（refactor bounties）**表现出浓厚兴趣，将其作为了解 **tinygrad 内部原理**以及通过 **JIT 测试函数**案例进行学习的切入点。
   - 一位成员甚至提交了一个 Pull Request (PR) 来处理数组输入，但其测试用例失败了。
- **调度器启发式算法大幅缩减图大小**：使用 `RING=0` 配合基础的**调度器启发式算法（scheduler heuristic）**，显著将最大的 **graphexec** 大小从 **5k 降至 2k**。
   - 这一改进突显了调度器优化对图执行效率的影响。
- **FUSE_OPTIM 难以奏效**：设置 `FUSE_OPTIM=1` 似乎没有产生预期效果，促使一位成员开始探索非贪婪搜索策略。
   - 这表明当前的 fuse 优化实现可能存在问题，值得进一步调查。
- **NCCL 巧妙处理 CUDA 图**：有人提问 **NCCL** 如何管理 **CUDA graphs**，显然它们运行良好，这与 tinygrad 当前的实现形成对比。
   - 这表明 **NCCL** 可能会提供对 tinygrad 的 CUDA 图集成有益的见解或技术。
- **零维张量困扰梯度计算**：一位用户质疑为什么 `a` 的梯度是一个*任意数字*而不是 **4**；这是由于**零维张量（zero-dimensional tensors）**需要梯度引起的。
   - 建议是禁用这些张量，并推荐将 `a` 更改为 `Tensor([2.0], requires_grad=True)`。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **谷歌用 AI 终结 Web 生态**：成员们讨论了 Google I/O 的发布内容，即 **AI 将编写网站并生成内容**，而这些内容仅供其他 **AI 抓取和总结**。
   - 一位成员开玩笑说 *Google 肯定在亲手终结 Web*，很快 Chrome 就会变成一个聊天界面。
- **MCP 客户端脱离桌面限制**：一位成员澄清说 **MCP 客户端/主机架构** *可以是任何形式，如 Web、CLI*。
   - 该成员有兴趣在云端运行一个**基于守护进程（daemon-based）的 MCP 客户端**，并使用一个**轻量级的基于 REST 的代理**来处理浏览器 UI 通信，将 HTTP 转换为 MCP。
- **浏览器 MCP 客户端想法涌现**：一位成员建议直接在浏览器中构建 **MCP 客户端**，甚至可能也在那里创建 **MCP server**，以避免 SSE 和流传输的复杂性。
   - 他指出他将研究该选项，这可能是一个有趣的想法。
- **Hugging Face MCP 身份验证触发器可用**：成员们讨论了 Hugging Face 对 MCP 的身份验证，可通过 [https://hf.co/mcp?login](https://hf.co/mcp?login) 触发。
   - 他们注意到身份验证默认是匿名的。
- **MCP Cloud 推出托管服务**：[MCP Cloud](https://mcp-cloud.ai) 推出了专门针对 **MCP server** 的托管服务，提供专用实例、JWT 身份验证和实时日志，可在数秒内完成部署。
   - 它支持多工作流和复制/粘贴集成（特别是与 **N8N**），面向需要可靠、安全的 MCP 基础设施的开发人员和团队。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 可靠性在积分损失中受到质疑**：多位用户报告了 **Manus** 的问题，包括卡在 **thinking** 状态并抛出 **internal server errors**，同时对最近的 **credit loss** 表示担忧。
   - 一些用户发表意见认为 *Manus 变笨了且经常出错*。
- **提供邀请码，积分消耗引发讨论**：一位用户提供了 **invitation code**，同时讨论了 **Manus 增加的积分消耗**。
   - 有人声称 *它确实在消耗更多积分*。
- **分配积分受限，是 Bug 吗？**：一位用户报告仅收到 **1k credits**，未提供更多背景信息。
   - 尚不清楚这是 Bug 还是预期行为。
- **Manus 拒绝共享 VS Code 密码**：一位尝试在 **Manus 电脑**上访问 **VS Code** 的用户遇到了要求输入密码的登录提示，而 **Manus** 拒绝提供。
   - 用户被告知 *检查 .../config yaml 配置文件以获取密码*。
- **Quality Agent 模式 vs High Effort 模式**：一位用户询问新的 **quality agent mode** 是否与之前的 **high effort mode** 相同。
   - 未提供确切答案。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **正在寻找负责任 AI 频道**：一位成员询问 Cohere Discord 中是否有专门讨论 **responsible AI**、**AI safety** 和 **fairness** 的频道。
   - 该请求未得到立即回应或指向现有资源。
- **学生在以太坊自动化代码审查**：一位加州大学戴维斯分校的学生兼以太坊基金会实习生正在利用 **Perplexity** 进行研究，自动化代码审查和漏洞检测。
   - 他的工作还探索了 **LLMs** 和 **LLM memory** 的对抗性角度。
- **柏林学生维护机器学习公平性工具包**：一位柏林的计算语言学学生维护着 **fairlearn**，这是一个用于 **ML fairness** 的开源工具包。
   - 她计划在协助 **Aya project** 后，将她的公平性专业知识应用于计算语言学。
- **工程师玩转 Transformer 架构**：一位 AI 工程师/研究员正专注于针对小型用例修改 **Transformer Architecture**。
   - 该工程师发布了一份名为 *Agents: All You Need* 的时事通讯。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 开启兼容 Claude 的 MCP 服务器**：LlamaIndex 发布了一个新的开源模板仓库，用于构建一个兼容 **Claude 的 MCP 服务器**，该服务器作为一个具有完整 **OAuth 2.1 支持** 的 **Next.js app**。
   - 该项目在内部黑客日期间创建，简化了远程 **Model Context Protocol** 服务器的创建，以实现无缝操作。
- **Agent 通过 LlamaIndex 的记忆模块获得记忆**：LlamaIndex 与 **AIMakerspace** 合作，正在为 **LlamaIndex Agents** 开发新的 **memory blocks**。
   - 这些 **memory blocks** 将涵盖持久化聊天历史和 **long-term memory**，详情见[此处](https://t.co/D4ZiBK54Fh)。
- **构建 Zoom 会议记录 Agent**：成员现在可以利用 **Zoom** 的 **RTMS** 获取实时数据，为 **NotionHQ** 构建 **Meeting Notetaker agent**。
   - [此链接](https://t.co/4m2IOcz7Se)提供了一个展示该集成的完整示例。
- **AI 工程师寻求 LLM 时事通讯宝库**：一位成员请求推荐关注真实世界 **LLM** 用例的 **AI newsletters**。
   - 他们寻求重点介绍 **LLMs** 实际应用而非仅仅是模型发布和更新的时事通讯。
- **LlamaCloud API 抛出 Job ID 错误**：一位成员报告在通过 **LlamaCloud API** 获取解析任务结果时出现 *invalid job_id* 错误，参考了[此文档](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get)。
   - 另一位成员建议 API 调用可能需要一个 `/{result_type}` 参数，参考了 [LlamaCloud 文档](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get)和 [SDK 代码](https://github.com/run-llama/llama_cloud_services/blob/98ad550b1ad29d97e566c43e21ad19edaee6d38d/llama_cloud_services/parse/base.py#L49)。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT4All 网站面临技术问题**：一位用户报告了官方 **GPT4All 网站** [nomic.ai/gpt4all](https://www.nomic.ai/gpt4all) 的错误，指出其 **GPU 使用率过高 (60%)**。
   - 该用户还推广了他们的开源项目 [HighNoonLLM](https://versoindustries.github.io/HighNoonLLM/) 并寻求潜在的合作。
- **GPT4All 在 Qt 版本上遇到困难**：一位用户发现 **GPT4All** 的 **CMakeLists.txt** 要求 **Qt 6.7**，而 C++ 代码却使用了 **Qt 6.8** 独有的特性，尽管文档声称 **Qt 6.5+** 就足够了。
   - 他们补充说，**GPT4All** 的 **Qt 模块**不符合 **Qt 6.8** 中更严格的注册方式，仍在使用 [Qt 文档](https://doc.qt.io/qt-6/qml-singleton.html) 中提到的已弃用的命令式单例注册。
- **GPT4All 落后于 LM Studio**：在一位用户询问如何在 **GPT4All** 中使用 **Microsoft 的 1.58B 2B4T 模型**后，另一位用户建议改用 **LM-Studio**。
   - 该用户表示 *GPT4All 的更新不够及时*。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **GenAI 主导 AI 讨论**：成员们观察到 **Generative AI** 已经掩盖了其他 AI 领域，导致需要设立一个“非 GenAI”类别。
   - 有人将这比作将所有医学命名为“非心脏病学”，突显了 AI 在生成模型之外的广度。
- **工程师开始 Tetris 机器人项目**：一位 AI 工程师正在寻求关于构建能够进行实时棋盘检测和游戏操作的 **Tetris 机器人**的建议。
   - 这位工程师是此类项目的新手，正在寻求启动开发流程的指导。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 贡献者意向**：一位 Torchtune 用户表示他们 *将投入一些时间* 为该项目做贡献。
   - 未提供更多细节。
- **Stas 的回顾性推文**：这里分享了来自 Stas 的一条推文 [链接](https://x.com/stasbekman/status/1937563125659893900?s=46&t=b1X88nwMsmZgHkmMFkiG3g)。
   - 关于这条推文没有分享更多细节。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **未讨论任何主题**：在提供的文本中未发现讨论主题。
   - 请提供相关的讨论文本以便总结。
- **未提供链接**：提供的文本中没有讨论任何链接或 URL。
   - 包含相关资源链接的总结将更具参考价值。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Aoun Linkbuilder 加入对话**：Aoun Linkbuilder 介绍了自己，他拥有政府学院大学（Government College University）的**数字受众理学学士学位**，专注于 **SEO 和数字营销**。
   - Aoun 设定的目标 *不仅是提高排名，还要增强可见性、引入有机流量，并最终为客户带来切实的增长*。
- **Aoun 强调 SEO 技能组合**：Aoun 描述了自己在**页面内和页面外 SEO、本地 SEO 以及技术性 SEO** 方面的扎实基础。
   - 他们的数字营销之旅源于*赋能企业和企业家在网络领域蓬勃发展的热情*。
- **Taylor Swift 粉丝分享联系方式**：Aoun 分享说，在数字领域之外，经常可以看到他们与朋友和爱犬在一起，听着 **Taylor Swift 的专辑**，或者通过**工艺美术**探索创意。
   - Aoun 提供了指向其官方账号和服务的各种链接，联系邮箱为 aounlinkbilder@gmail.com，官方网站见[此处](https://aounlinkbuilders.my.canva.site/)。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。

---

您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。

---

# Discord：分频道详细总结与链接

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1387145509046845583)** (1061 条消息🔥🔥🔥): 

> `Business fellowship rewards, Imagen 4, Gemini CLI Agent, Claude artifacts` 


- ****Fellowship 奖励仍不明朗****：一名成员询问在达到 **25 名成员**后如何领取 Business fellowship 奖励（马克杯、T恤、抽奖券），但具体说明仍不明确。
- ****Google 的 Gemini CLI Agent 正式开源****：Google 正在推出一款由 **Gemini 2.5 Pro** 驱动并支持 **MCPs** 的全新开源 **Gemini CLI Agent**。
   - 正如[这段视频](https://video.twimg.com/amplify_video/1937849103657930752/vid/avc1/1280x720/QosBoysN--Q80PTL.mp4)所示，它为开发者提供了访问 **Gemini 2.5 Pro** 的权限，并支持多角色提示词 (MCPs)。
- ****Anthropic 发布 Artifacts 和 Artifacts Gallery****：Anthropic 在 Web 端发布了 **AI 驱动的 Artifacts 和 Artifacts Gallery**，使用户能够“在 Claude 内部构建 Claude”，详见[这段视频](https://video.twimg.com/amplify_video/1937926883707891713/vid/avc1/1920x1080/3Gu7ntwPGQT0j8dX.mp4)。
- ****Google 低调在 AI Studio 和 API 上推出 Imagen 4****：**Imagen 4 和 Imagen 4 Ultra** 现已在 AI Studio 和 API 上可用，尽管根据一位展示了[“一个在暴风雪中倒立的小丑”示例](https://cdn.discordapp.com/attachments/1047649527299055688/1387374829744685088/wDnjFf5U1YUCQAAAABJRU5ErkJggg.png?ex=685dc5bf&is=685c743f&hm=4f1cc936f04925773b1328ffbcc229e48fa59a5ae4b74754963caf76c527079d&)的成员说法，其效果*尚未完全达到预期*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1387202590109864017)** (9 条消息🔥): 

> `Perplexity AI Image Generation, NYC Mayoral Primary, Amtrak Baltimore, Google Accident, China EV Rise` 


- **Perplexity AI Labs 创作梦幻图像**：一位用户使用 **Perplexity AI Labs** 创建了一张图像，并表示：“这比我梦寐以求的还要好。”
- **链接讨论广泛话题**：分享的链接涉及从 [Ubisoft 的《全境封锁 2》补丁](https://www.perplexity.ai/page/ubisoft-s-the-division-2-patch-dWFALCxPQmCDkPmNrKBfUQ)到[纽约市长初选](https://www.perplexity.ai/page/new-york-city-mayoral-primary-JcgCeh9ASOmZS6v5m2ydLw)等各类话题。
- **链接继续包含各类新闻**：更多分享链接涉及从[巴尔的摩的 Amtrak 铁路](https://www.perplexity.ai/page/amtrak-train-stuck-in-baltimor-HW5PP3_ITvSpGWKpriA.Jg)到[绑架未遂指控被撤销](https://www.perplexity.ai/page/prosecutors-drop-attempted-kid-A6aImqvBSzyZ8INw9ijgJw)等主题。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1387390049502494772)** (3 条消息): 

> `search domain filters, hallucinations of articles` 


- **搜索域名过滤器导致幻觉？**：成员们报告称，将搜索域名过滤器设置为新闻网站（如 reuters.com）现在会导致**文章幻觉**。
   - 他们也无法获得有效的结果或引用数组 (citations arrays)，这在频道中引起了一些挫败感；一位成员感叹道：“该死……我真希望我们能得到一个答复”。
- **搜索域名过滤器问题尚无答复**：成员们感到沮丧，因为他们尚未收到关于搜索域名过滤器为何产生幻觉的答复。
   - “该死……我真希望我们能得到一个答复”。


  

---

### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1387146374105137285)** (674 messages🔥🔥🔥): 

> `Cursor 上下文长度, Gemini vs Claude, Cursor Rules, 速率限制, Cursor 新定价` 


- **上下文危机：Cursor 的上下文长度难题**：成员们讨论了 **Cursor** 可能会自动总结超出上下文长度的聊天窗口，导致内容丢失和用户困惑，并且 [Gemini 处理大上下文的能力优于 Claude](https://docs.cursor.com/context/management)。
   - 有人指出 Deepseek 模型在理解上下文方面表现最差，导致 **Cursor** 将其上下文长度缩减至约 **60k tokens**。
- **Gemini 角斗士 vs. Claude 巨像：谁更胜一筹？**：用户对比了 **Gemini** 和 **Claude** 模型，一些人发现 **Gemini 2.5 Pro** 在其他平台上更流畅，而另一些人则认为由于合作伙伴关系，**Claude** 在 **Cursor** 内部表现更好。
   - 一位用户指出 *Gemini 在 Cursor 中处理工具调用（尤其是编辑文档）时存在巨大问题*，但仍然更倾向于使用它而非 **Claude**。
- **速率限制传闻：Pro 用户对定价感到困惑**：Pro 用户遇到了不同的速率限制体验，一些用户为了避免触发限制而采用短提示词且不附加文件的方式，另一些人则呼吁 [推出 Pro+ 计划的潜在价值](https://cursor.com/pricing)。
   - 还有人对速率限制缺乏透明度表示不满，这使得预测使用情况和有效规划变得困难。一位用户指出 *Pro 版的新玩法是付了费还得通过深思熟虑的对话消息来考虑 token 使用，笑死*。
- **Google CLI 宏大的 Gemini 策略出师不利**：用户测试了新的 **Gemini CLI**，发现它存在 Bug 且尚未准备好发布，问题包括在执行 `npm run dev` 时卡死以及无法安装 **Nuxt**。
   - 尽管每天提供 **1000 次请求** 的慷慨优惠，但该服务被认为速度极慢且已损坏，一位用户开玩笑说 [它可能会在代码库中包含广告](https://blog.google/technology/developers/introducing-gemini-cli/)。
- **OpenAI 宕机：Cursor 社区陷入忧郁**：Cursor 用户报告遇到 **API error** 消息，部分用户无法使用 **O3 Pro** 模型。
   - 原因随后被确定为 **OpenAI** 宕机，建议订阅 [Cursor 官方状态页面](https://status.cursor.com/) 以获取更新，并使用论坛寻求支持。


  

---


### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1387145283925966858)** (36 messages🔥): 

> `密钥管理, Git 远程 URL 问题, 远程机器设置, 环境变量, Background Agent 规则` 


- **Background Agents 的密钥管理**：用户可以在 **Cursor Settings >> Background Agents >> Secrets** 中配置 Background Agents 的密钥，避免将其推送到 `environment.json`。
   - 这允许 Agent 根据需要使用这些密钥。
- **Git 远程 URL 问题困扰 Background Agents**：一位用户发现带有前导 `www` 的本地仓库 URL 会导致 Background Agents 出现问题，因为它会检查是否为有效的 `github.com` URL，并指出这**搞乱了 Background Agents**。
   - Agent 会运行 `git remote get-url origin` 并检查该 URL 是否为 github.com URL。
- **远程机器和 Brew 安装故障排除**：一位用户报告了在远程机器上安装 **Brew** 的问题，即使在创建快照后，关闭并重新打开设置时，Brew 及其 PATH 设置也会丢失。
   - 建议他们在做出更改后创建新快照，或者使用 **Dockerfile** 来管理 Brew 安装。
- **Background Agents 自定义规则：用户的诉求**：一位用户对需要反复向 Background Agents 提供指令（例如避免在 Docker 容器外进行后端单元测试以及强制执行 lint 检查）感到沮丧。
   - 他们正在寻求一种设置持久规则的方法，以避免重复的提示。
- **在 Background Agents 上运行 Python 3.11**：一位用户寻求在 Background Agent 上运行 **Python 3.11** 的最简路径，因为他们遇到了环境默认为 Python 3.13 的问题，导致某些包出现兼容性问题。
   - 有人分享了一个 **Dockerfile**，作为指定特定 Python 版本（包括将其设置为默认版本）的方法。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1387147210466001051)** (637 条消息🔥🔥🔥): 

> `DeepSeek-R1-0528, 确定性模型, Mesh Link 成功, Unsloth 支持 Intel XPU, MacBook Pro 散热` 


- **寻求更小的 DeepSeek 模型**：成员们正在寻找 [DeepSeek-R1-0528](https://huggingface.co/deepseek-ai/DeepSeek-R1-0528) 的更小版本，因为即使是 **1-bit 版本** 对某些 GPU 来说也太大了。
   - 一位成员询问了 **Qwen-3** 与非 Qwen 模型的对比，以及是否应该尝试 **DeepSeek-R1-0528-Qwen3-8B**。
- **确定性输出讨论**：成员们讨论了创建一个输出 **100% 确定性** 结果模型的可能性，建议包括将 temperature 设置为 0 以及进行微调。
   - 一位成员指出，用概率函数实现确定性是有缺陷的，而其他人则提到了随机性的用处以及实现 **100% 确定性** 的难度。
- **Unsloth 支持 Intel XPU**：成员们注意到 Unsloth 现在支持 Intel XPU，根据 [此 commit](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a)。
   - 当今年晚些时候发布 **价格低于 $1k 的 48GB GPU** 时，这将是一个**重大突破**。
- **MacBook Pro 过热**：成员们讨论了在 NN 训练期间为 MacBook Pro 降温的方法，建议包括使用**铝制支架**、**带风扇的支架**或重新涂抹**导热膏**。
   - 一位用户建议 **90-95C 作为 GPU 最高温度**，并建议不要使用冰块，而另一位用户提到要避开水。
- **AI 垃圾内容末日即将来临**：一位成员预测 **AI 垃圾内容（AI slop）将在 2026 年之前摧毁互联网**，而其他人认为这已经发生了。
   - 一位成员评论道：*并非所有 AI 都是垃圾邮件.. 但所有垃圾邮件都是 AI*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1387145684209238178)** (49 条消息🔥): 

> `Flux QLoRA, 廉价 GPU 推荐, Hyperbolic XYZ, OpenSloth 上的多 GPU SFT, 带有 Vision 训练的 Chat Templates` 


- **Flux QLoRA 博客文章前景看好**：一位成员分享了 [关于 Flux QLoRA 的 Hugging Face 博客文章](https://huggingface.co/blog/flux-qlora)，认为这是一个潜在的有用资源。
   - 另一位用户表示感谢，称其*前景看好*。
- **在 Hyperbolic XYZ 租用 H100**：一位用户推荐使用 [Hyperbolic XYZ](https://app.hyperbolic.xyz/) 以每小时 **$0.99** 的价格租用 **H100**，以及每小时 **$0.28** 的价格租用 **RTX 4090**。
   - 他们还提供了一个推荐码，可获得额外的 **$6** 额度。
- **Gemma3 Vision Notebook 即将推出**：一位成员表示他们将在今天晚些时候发布 **vision notebook**，但如果你想要参考，可以下载 [此处](https://github.com/unslothai/unsloth/pull/2785) 标记为 vision 的任何 notebook。
   - 他们还提到，仅添加缺失的字段/参数似乎会产生错误。
- **Vision 训练不需要 Chat Templates**：一位成员建议：*如果你使用 **UnslothVisionDatacollator**，请不要在 vision 中使用 chat template。它会自动为你处理*。
   - 另一位成员建议，量化主要影响编程和数学，并推荐使用 **o3/Gemini** 等 **SOTA** 模型。
- **Llama3 输出合并了所有回复**：一位用户报告说，在训练 **llama3.1-8B-Instruct** 时，输出是正确的，但似乎将所有回复合并到了一个输出中。
   - 团队回应称他们已经进行了更新，保存功能应该可以正常工作。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1387265229670449223)** (3 条消息): 

> `OAT Zero, 数据训练` 


- **OAT Zero 揭晓**：一位成员分享了关于 **OAT Zero** 的 [YouTube 链接](https://youtu.be/z3awgfU4yno?si=yyYVWNEYbPiupRrD)，并想知道大家的看法。
   - 另一位成员回应说人们已经预料到了，并链接到了 **OAT Zero Notion 页面**：[oatllm.notion.site/oat-zero](https://oatllm.notion.site/oat-zero)。
- **修复微调数据**：一位成员表示：*无论如何，你都需要修复你的训练数据，并将你的微调模型锚定到该固定的训练数据上*。
   - 对话的截图也被分享作为参考。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1387241094261116979)** (1 条消息): 

> `ChatGPT connectors, Google Drive, Dropbox, SharePoint, Box` 


- **云服务 ChatGPT 连接器发布！**：针对 **Google Drive**、**Dropbox**、**SharePoint** 和 **Box** 的 **ChatGPT connectors** 现已面向 ChatGPT 中的 Pro 用户开放（不包括 EEA、CH、UK 地区）。
   - 这些连接器旨在*为日常工作引入独特的上下文*。
- **关于云连接器的详情**：此功能仅适用于 **ChatGPT Pro 用户**。
   - 当前版本不适用于 **EEA**、**CH** 和 **UK** 地区的用户。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1387155403816829029)** (522 条消息🔥🔥🔥): 

> `GPT helps get web dev job, AI taking jobs, O3 and Pro launched, Selling AI as Art, BS detector benchmark` 


- **GPT 助力获得 Web 开发职位**：一位成员表示 **GPT** 帮助他在今天获得了一份 Web 开发工作。
   - 他们认为第一波 **AI 取代工作**的浪潮并非 AI 本身，而是拥有互联网访问权限的弱势群体。
- **O3 与 Pro 发布解锁搜索连接器**：**聊天搜索连接器**于 **2025 年 6 月 24 日**面向 Pro 用户发布，支持 Dropbox、Box、Google Drive、Microsoft OneDrive/SharePoint 等集成，但仅限于 EEA、瑞士和英国以外的用户。
   - 成员们对 Pro 中新的**搜索连接器**功能感到高兴，该功能允许他们使用大量同步的个人数据来训练 AI。
- **AI 生成内容引发艺术辩论**：一位成员引发了关于 AI 作为艺术工具角色的讨论，区分了将 AI 用于概念/原型设计与将纯 AI 生成的作品作为个人艺术品出售的行为。
   - 该成员认为，*让别人相信是你制作了该作品而实际上并非如此*是不道德的。
- **使用 BS Detector 基准测试 LLM**：成员们讨论了通过创建违背逻辑的问题来诱导 LLM 给出荒谬答案，以此测试它们识别逻辑谬误的能力，并引用了 [哥德尔不完备定理](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems)。
   - 一些人认为，模型无法识别逻辑陷阱表明其缺乏理解能力，而不仅仅是在模仿。
- **MiniMax 基准测试显而易见**：一位成员分享了 [MiniMax benchmark](https://artificialanalysis.ai/models/minimax-m1-80k?models=gpt-4-1%2Co3%2Co4-mini%2Cllama-4-maverick%2Cllama-4-scout%2Cgemini-2-5-pro%2Cgemini-2-5-flash-reasoning%2Cclaude-4-sonnet-thinking%2Cclaude-4-sonnet%2Cmistral-medium-3%2Cdeepseek-r1%2Cdeepseek-v3-0324%2Cgrok-3-mini-reasoning%2Cnova-premier%2Cminimax-m1-80k%2Cllama-3-1-nemotron-ultra-253b-v1-reasoning%2Cqwen3-235b-a22b-instruct-reasoning%2Cgpt-4o#intelligence-vs-price)，指出它隐藏在最具吸引力的智能 vs 价格象限中。
   - 它采用了 [Multi-head latent attention](https://arxiv.org/pdf/2206.04615)，但其他人称之为*技术术语堆砌 (technobabble)*。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1387175216832057454)** (2 条消息): 

> `File Uploading Issues, Project Folder Problems` 


- **文件上传触发加载转轮**：一位用户报告了在项目文件夹中删除或上传文件时遇到的问题，遇到了**加载转轮 (spinning wheel)** 且进程停滞长达 8 小时。
   - 该用户尝试使用 **Android 手机**和安装了 **Safari** 及 **Google Chrome** 的 **Mac**，但问题依然存在。
- **部分用户文件上传正常**：与报告的问题相反，另一位用户表示他们在删除或添加文件时没有遇到任何问题。
   - 这表明该问题可能是孤立的，或者与特定的配置或文件类型有关。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1387324267045060669)** (2 条消息): 

> `Introductions, Channel Welcome` 


- **用户自我介绍**：用户 @coolstaconnormobile 发起对话，询问频道中是否有人，并请求在回复时艾特（ping）他。
   - 这条消息作为开场白，旨在寻求频道其他成员的互动和参与。
- **欢迎新用户**：用户 darthgustav 回复了初始消息，欢迎 @coolstaconnormobile 加入 <#1046317269069864970> 频道。
   - 他们进一步询问了打算讨论的话题，邀请 @coolstaconnormobile 分享其具体的兴趣或问题。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1387324267045060669)** (2 messages): 

> `频道介绍` 


- **新成员向频道问好**：新成员 @coolstaconnormobile 加入了频道并主动联系，寻求与其他成员的互动。
- **频道欢迎新人**：频道成员 darthgustav. 回应了新成员的问候，表示欢迎并邀请他们介绍讨论话题。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1387148735590694993)** (311 messages🔥🔥): 

> `BitNet 量化至 1.58 bits，Gradio 问题，AI agents，Llama 3.1 8B 即时 128k，Model Context Protocol` 


- **BitNet Demo 令用户惊叹**：用户正在尝试 [BitNet demo](https://bitnet-demo.azurewebsites.net/)，并报告称其*结果质量令人震惊*，尤其是其速度和初始查询表现。
   - 一位用户强调该模型已在 HF Spaces 上线 ([Chat-with-Bitnet-b1.58-2B-4T](https://huggingface.co/spaces/suayptalha/Chat-with-Bitnet-b1.58-2B-4T))，并可以通过编程方式使用。
- **Cerebras 云在大规模应用时非常便宜**：成员们讨论了 [Cerebras](https://www.cerebras.ai/cloud)，指出它*几乎已脱离 Beta 阶段*，其晶圆级 GPU 在*大规模应用时非常便宜*。
   - 一位用户指出，虽然 Cerebras 专注于高端 HPC，但它与 **Blackwell** 性能相当，价格稍便宜但带宽略低；另一位用户重申 **Groq** 是非常出色的技术，非常适合大规模推理。
- **开源巅峰：独角兽 CEO 提交 PR**：一位用户开玩笑说，让一个*顶尖独角兽 CEO* 在你的仓库提交 PR 可能是开源生涯的巅峰，并发布了仓库截图。
   - 另一位用户指出 Jina AI 的 **Han Xiao** 博士也会这样做。
- **用户调试 Gradio 加载问题**：一位成员报告 Gradio 应用卡在加载界面，发布了日志并寻求帮助。
   - 另一位成员建议检查堆栈跟踪（stack trace）或者直接重启 Space。
- **针对印度语的新语音 AI 模型发布**：一位成员发布了 [Veena](https://x.com/Dheemanthreddy_/status/1937839083281437021)，这是一个针对印度语的新语音 AI 模型。
   - 他鼓励其他人分享关于语音质量的反馈。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1387173991978107030)** (2 messages): 

> `Linux Bash 脚本编写，Shell 脚本实用程序` 


- **确认 Bash 脚本编写协助**：一位用户对 **Linux Bash 脚本编写**方面的帮助表示感谢。
- **感谢 Shell 脚本专业知识**：用户表达了对 **Shell 脚本**实用性的赞赏。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1387175888969273477)** (45 messages🔥): 

> `LLM Shader Graph Code Generation, Nvidia Neural Materials, Material Dataset using Quixel, gSPLAT with Gaussian+ filtering, Rust Crate for Local LLMs` 


- **LLM 深入研究 Shader Graph 代码生成**：成员们讨论了使用 **LLM** 生成 **Shader Graph 代码**（可转换为 **HLSL** 或 **GLSL**），以及研究人员如何利用**语言模型**优化这些代码。
   - 一位成员指出，现有方法使用基于规则的生成，而 **Nvidia** 正在通过预测像素的小模型探索 **Neural Materials**。
- **Quixel + LLM 赋能 Shader Graph 生成流水线**：一位成员正开始使用 **Quixel** 和 **LLM** 构建**材质数据集**来生成 **Shader Graph**，并将 LLM 光栅化与游戏引擎的程序化生成进行对比。
   - 目前的优化通常不尽如人意，导致需要手工编写 Shader，因此他们正在寻求更好的解决方案。
- **gSPLAT Gaussians 挑战高维材质 Shader**：讨论围绕带有 **Gaussian+ 滤波**的 **gSPLAT** 是否能在 5-10 年内取代材质 Shader 展开，并承认目前的方法本质上是“烘焙一切”。
   - 据分享，某些材质拥有多达 **107 个输入参数**，这表明当前的数据/纹理理念可能不足，使得学习到的权重（learned weights）成为一个很好的替代方案。
- **Rust crate 简化了本地 LLM 的 Tool Calling**：一位成员正在开发一个 **Rust crate** 以简化本地 LLM 的使用，重点是让 **Tool Calling** 变得更容易，并请求 API 反馈，同时分享了一些 Rust 代码。
   - 另一位成员建议使用一个支持重试的宏来添加基本的错误处理或重试机制，该宏可以自动处理瞬时错误。
- **RAG 获得 Embedder 集合与 SmartTaskTool**：一位成员分享了一个 **RAG embedder 集合**（[Hugging Face 链接](https://huggingface.co/kalle07/embedder_collection)）和一个适用于 Windows 的**小任务工具栏**（[Hugging Face 链接](https://huggingface.co/kalle07/SmartTaskTool)）。
   - **SmartTaskTool** 是一个任务栏图标，而非浮动窗口，现在已包含跨语言支持 (en-de-fr-roberta)。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1387178419682279434)** (1 messages): 

> `LessWrong Post, Gradient Descent, Token Input Embeddings, ModernBERT` 


- **关于 Gradient Descent 的 LessWrong 帖子被接收**：一位成员宣布他们关于 **Token Input Embeddings** 和 **ModernBERT** 的 **Gradient Descent** 帖子已被 [LessWrong.com](https://www.lesswrong.com/posts/GK2LSzxjEejzDjzDs/gradient-descent-on-token-input-embeddings-a-modernbert) 接收。
- **ModernBERT 分析**：该 LessWrong 帖子深入探讨了 **ModernBERT** 架构。
   - 它专注于 **Gradient Descent** 如何应用于 **Token Input Embeddings**。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1387533186753364101)** (1 messages): 

> `User Profile embeddings, Thematic analysis, Cosine similarity analysis` 


- **使用 Embeddings 处理用户画像**：一位成员正在进行一个项目，旨在利用**用户画像 Embeddings** 和**余弦相似度（Cosine Similarity）**，识别哪些受访者的观点与文章最一致。
   - 他们计划将每个用户的回答合并为一个单一画像，为每个画像创建 Embeddings，然后将这些 Embeddings 与文章的 Embeddings 进行比较。
- **主题分析仍悬而未决**：该成员提到他们正在考虑**主题分析（Thematic Analysis）**，但不确定在该项目中的具体实现。
   - 他们尝试过摘要生成器（summarizers），但结果不够准确，无法代表输入内容。
- **寻求相似度分析建议**：该成员正在寻求关于进行**相似度分析**的不同方法的建议。
   - 他们指出，进行此类分析的方法似乎有很多，不确定该选择哪一种。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1387288001293586593)** (17 messages🔥): 

> `smolagents 工具使用，DuckDuckGoSearchException 修复` 


- **smolagents 工具使用受到质疑**：一位成员询问了 **smolagents** 中的工具使用问题，注意到在使用本地 **qwen 7b 模型**时，模型有时会在其思考输出（thinking output）中修改工具代码。
   - 另一位成员建议使用 **togetherAI** 配合 **qwen3-235b-A22b-fp8-tput** 模型和 **qwen-agent 库**作为更优的替代方案，理由是与其他供应商相比，其性价比更高。
- **DuckDuckGoSearchException 需要修复**：一位成员报告在访问 `https://lite.duckduckgo.com/lite/` 时遇到了 `DuckDuckGoSearchException`，并伴有 `RuntimeError: operation timed out` 错误。
   - 消息中未提供任何解决方案或建议。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1387145274635321537)** (293 messages🔥🔥): 

> `Grok 的政治灌输，中国模型审查，Gemini 2.5 Pro 的排名下滑，LM Arena 排行榜数据，LLM 的版权材料` 


- **Grok AI 被指控进行政治灌输**：一些成员担心 **Grok** 声称的目标——将其所知重写为他们自己的“真相”——是糟糕且危险的，本质上是政治灌输，而 LLM 极具说服力。
- **中国模型未能得到妥善审查的争议**：成员们就 **中国 AI 模型** 是否得到了有效审查展开争论。一些人认为它们只是通过添加外部过滤器来切断模型或替换响应来遵守法律。
   - 其他人则坚持认为，许多中国实验室与其政府拥有相同的价值观且根基深厚，并指出 **Yi 模型** 在没有 jailbreak 的情况下对 **Tienanmen Square** 相关内容不设限。
- **Gemini 2.5 Pro 被匿名模型取代？**：成员们讨论了 **Gemini 2.5 Pro** 在排行榜上排名下降的原因，将其归因于 **Blacktooth** 和 **Stonebloom** 等匿名模型的崛起。
   - 一些人认为这些匿名模型可能在 **Gemini 2.5 Pro** 较弱的领域表现出色，而另一些人则认为投票者的分布发生了变化。
- **LM Arena 排行榜通过提示工程被暴露**：一位成员声称发现了 **4 种方法** 可以从 lmarena.ai 获取排行榜数据，引发了关于抓取该网站的伦理和法律问题的讨论；一位社区成员指出所给出的 4 种方法中有 **3/4** 是无效的。
   - 随后列出了这 4 种方法，包括利用 Hugging Face space、预先存在的数据转储（data dumps）、网页抓取和浏览器扩展。
- **开源社区渴望版权材料**：成员们讨论了近期关于版权数据集的法院裁决，并强烈呼吁在训练数据集中使用版权材料。
   - 关于近期版权裁决对 LLM 训练的影响存在冲突观点：一些人认为没有影响，因为无论如何训练都会继续；另一些人则希望建立一个必须为版权材料付费的系统。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1387469021489791036)** (1 messages): 

> `MCP Host, LM Studio v0.3.17, 新语言` 


- **LM Studio 成为 MCP Host**：**LM Studio 0.3.17** 引入了 **MCP Host** 功能，使用户能够将喜爱的 MCP 服务器连接到本地 LLM。
- **LM Studio 支持 11 种新语言**：得益于社区本地化贡献者，LM Studio 0.3.17 增加了对 **11 种新语言** 的支持，使总数达到 **33** 种。
   - 伴随语言更新的还有全新的 **'Solarized Dark'** 主题和大量的错误修复。
- **LM Studio v0.3.17 发布！**：最新的 [LM Studio version 0.3.17](https://lmstudio.ai/blog/lmstudio-v0.3.17) 引入了 **MCP Host** 支持，允许连接到本地 LLM，同时新增了 **11 种新语言** 和 **'Solarized Dark'** 主题。
   - 更多信息请参阅 [LM Studio 文档](https://lmstudio.ai/docs/app/plugins/mcp)。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1387147558543167579)** (181 条消息🔥🔥): 

> `LM Studio “隐藏”聊天消息，r/LocalLlama 的新管理层，网络安全 LLM，使用 LLM 进行 300k token 翻译，LM Studio 中的 MCP Host 与 Client` 


- **神秘的模型故障：LM Studio 隐藏聊天记录**：一位用户报告称，在升级 **LM Studio** 并加载新模型后，之前来自“无审查”模型的*隐藏*对话变得可见，其中包括一段关于*“制作假身份证的过程是什么？”*的隐藏交流。
   - 这引发了关于原因的推测，理论范围从流式传输设置到系统提示词（system prompts）不等，但隐藏对话的确切原因尚不清楚。
- **Reddit 救援：r/LocalLlama 迎来新掌门**：[r/LocalLlama](https://www.reddit.com/r/LocalLlama/) 子版块已由新管理层接管，引发了关于新版主参与众多其他子版块的讨论。
   - 一些用户对该版主广泛的管理范围表示担忧，而另一些用户则认为目前没有明显的异常迹象。
- **网络安全 LLM 选择策略**：一位用户请求推荐最适合**网络安全**用途的 **LLM**，并寻求每个模型的具体 USP/优势。
   - 有人指出，在使用 **LLM** 执行网络安全相关任务时，设置良好的系统提示词对于避免不断的警告和免责声明至关重要，此外还提到了 [whiterabbitneo](https://huggingface.co/models) 是一个难以解读的模型。
- **Token 难题：300k 翻译任务**：一位用户询问了处理 **300k tokens** 进行翻译的内存需求，透露了在一次性翻译大块文本时模型崩溃的困扰。
   - 建议采用分块处理并自动化翻译流程，并指出 [llama 4 scout](https://huggingface.co/models) 支持高达 **1000 万 token** 的上下文长度，以及使用 **python** 自动化分块。
- **MCP 机制：Host 与 Client 区别详解**：一位用户寻求关于 **LM Studio** 和 **MCP servers** 背景下 **MCP host** 与 **client** 区别的澄清。
   - 解释称 **LM Studio** 充当 host，而具备工具调用能力的 **LLM** 充当 client，随着 **LM Studio** 集成工具功能，两者的界限变得模糊；且 client 完全依赖于 host。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1387163725328421014)** (22 条消息🔥): 

> `PCIe 通道配置，Unsloth commit，Runpod serverless，GPU 外部扎带安装，内存模块温度报告` 


- **PCIe 通道分配详情公开**：用户 **oldtimer8430** 表示其系统的 PCIe 通道配置为 **16, 4, 4**，并可能可以配置得更均匀。
   - 他提到正在安装驱动程序并进行测试，表明正在进行活跃的系统设置和配置。
- **发现 Unsloth Commit！**：一位用户分享了 [Unsloth commit 的链接](https://github.com/unslothai/unsloth/commit/01c5e1a24935b93d9d3197815ad71751d9dfb37a)，可能表明对 **Unsloth AI 项目** 的关注或讨论。
   - 未提供关于该 commit 内容或重要性的进一步细节。
- **Runpod Serverless 推理评估**：一位用户提到计划尝试 **Runpod serverless**，特别是关注带有网络卷的 *flex workers*，以实现更快的模型加载和冷启动。
   - 他们正考虑将 Runpod 作为使用大型 NVIDIA GPU 进行推理任务的平台，未来也在考虑 **Predibase** 及其 *turbo lora* 功能。
- **GPU 扎带机箱改造**：一位用户幽默地描述了由于空间限制或其他原因，使用扎带将 GPU 安装在机箱*外部*的情况。
   - 其他用户对此表示好笑和难以置信，其中一人开玩笑说这种设置是典型的“美式风格（'Murican'）”。
- **内存模块现已显示温度**：一位用户注意到内存模块现在可以报告温度。
   - 这引发了关于硬件监控功能以及系统内热量管理的讨论。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1387432224076333076)** (3 条消息): 

> `模型运行时间 API、BYOK 改进、平台费用简化、华盛顿州和俄亥俄州销售税、数据库停机` 


- **OpenRouter 通过 API 监控模型运行时间**：开发者现在可以通过 [API](https://x.com/OpenRouterAI/status/1937869909448441980) 追踪模型运行时间。
- **BYOK 用户迎来新改进**：Bring Your Own Key (BYOK) 用户现在可以在保存密钥前进行测试、限制上游用量，并在 API 调用中追踪使用情况（[详情点击此处](https://x.com/OpenRouterAI/status/1937872903988535400)）。
- **平台费用结构精简**：OpenRouter 正在将其平台费用简化为 **5.5%**，最低费用为 **$0.80**，而加密货币支付将为 **5%** 且无最低费用，[此前的公告在此](https://discord.com/channels/1091220969173028894/1092729520181739581/1381645967866204261)。
- **华盛顿州和俄亥俄州将征收销售税**：**华盛顿州**和**俄亥俄州**的用户在结账时将看到适用的销售税，其他对推理征税的 [州](https://stripe.com/guides/introduction-to-saas-taxability-in-the-us) 也将陆续跟进。
   - 小额订单的费用将会增加，OpenRouter 指出 *对于绝大多数订单，总费用与我们之前的定价相比将会下降*。
- **短暂的数据库故障导致 401 错误**：由于 SSL 配置更改，OpenRouter 在 **东部时间下午 4:10** 经历了约 **30 秒** 的意外数据库停机。
   - 停机可能导致部分用户出现 *短暂的 401 错误*。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1387147590981779468)** (168 条消息🔥🔥): 

> `Midjourney 视频模型、GPT 4.1 Mini 问题、OpenRouter 费用变更、Claude Max 对比 OpenRouter、Veena 语音 AI 模型` 


- ****Midjourney 的视频尝试取得成功****：成员们对 **Midjourney** 和 **Spellbrush** 推出的新视频模型赞不绝口，称其为“i2v 的 ChatGPT 时刻”，并希望他们能获得更多基础设施来推出 720p 版本，且更倾向于在 GPU 上托管。
   - 其他成员提到了 *seedance* 和 *hailuo* 等替代方案，但最初的发布者表示它们的质量 *远不及* 前者。
- ****GPT 4.1 Mini 的输出异常****：**GPT 4.1 mini** 表现出不服从指令的情况，尽管容量为 **33k token**，但在 **3800 token** 处截断输出，并在 JSON 键之前添加 `\xa0`。
   - 成员们建议降低 temperature 并指定 `"response_format": {"type": "json_object" }` 以强制执行正确的 JSON 输出，而另一位成员报告称使用 **GPT 3.5** 处理类似任务取得了成功。
- ****OpenRouter 的费用结构面临争议****：OpenRouter 的新费用结构引入了 **$0.80** 的基础费用，引发了褒贬不一的反应，一些用户对小额订单成本增加表示担忧（例如，充值 **$5** 需支付 **$0.80** 费用）。
   - 该变更的支持者指出，它简化了费用计算，并使大多数用户和大额订单受益，而且 *税费也已加入*。OpenRouter 工作人员也介入进行了进一步解释。
- ****Claude Max 挑战 OpenRouter 的便利性****：随着 **Anthropic** 提供 **Claude Max** 和 **Claude Code**，一位成员质疑 OpenRouter 的持续价值，理由是 Claude 的订阅模式可以节省成本。
   - 其他成员表示，OpenRouter 为各种模型提供统一的登录/支付方式以及测试新模型的能力，OpenRouter 工作人员回应称他们可能会发布 *OR max 解决方案*。
- ****Veena 在印度语言语音领域取得进展****：一位成员宣布推出 **Veena**，这是一款针对 **印度语言** 的新语音 AI 模型，并对 OpenRouter 的协助表示感谢。
   - 详情见 [X.com](https://x.com/Dheemanthreddy_/status/1937839083281437021)，成员们对其发布表示祝贺。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1387151274591391795)** (17 messages🔥): 

> `build2, Meson, Buck2, xmake, Zig` 


- **Build2 遭到冷遇**：一位成员询问关于 **build2** 的使用经验，但讨论迅速转向了其他替代方案。
   - 建议的替代方案包括 [Meson](https://mesonbuild.com/)、[Buck2](https://buck2.build/)、[xmake](https://xmake.io/#/) 和 [Zig](https://ziglang.org/)。
- **C++ 构建系统引发辩论**：一位成员感叹*没有好的 C++ 构建系统*，引发了其他人的共鸣。
   - 其他人表达了对 **CMake** 的沮丧，其中一人表示，为了让 CMake 与某些随机库协同工作而浪费的时间实在太多了。
- **澄清 GCC 是编译器而非构建系统**：一位成员询问在大型项目中使用 **GCC** 的情况，随后有人澄清 **GCC** 是编译器，而非构建系统。
   - 解释称，虽然 **GCC** 可以编译单文件项目，但为了依赖管理和多平台支持，仍需要构建系统。
- **Bazel 获得赞赏**：一位成员称 **Bazel** 是史上最强（goat）的构建系统。
   - 他们对该系统可能存在的任何问题表示并不在意。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1387424287123701811)** (6 messages): 

> `Triton no longer open to public, Triton's lack of YouTube uploads, Gluon DSL, Triton support on Windows, Rust-based ML framework using Triton` 


- **Triton 的 Training Wheels 项目已结束**：一位成员指出 [Triton 不再向公众开放](https://discord.com/channels/1189498204333543425/1189607595451895918/1378126514318737541)。
- **Triton 团队停止更新 YouTube 视频**：一位成员提到他们很怀念 **Triton 的 YouTube 上传视频**，并希望了解新合并背后的逻辑，尤其是考虑到目前有大量新内容正在被合并。
   - 另一位成员想知道关于 **Triton 为什么不支持 Windows** 的潜在问题。
- **Gluon：Triton 的新 DSL**：一位成员重点介绍了 **Gluon**，这是 Triton 仓库中一个新的高级 DSL，类似于 **Paszke 的 Mosaic**。
   - 提供了 [test_core.py](https://github.com/triton-lang/triton/blob/5c9e54535dfe34d4b60fd13a4a27b6d74f3c8344/python/test/gluon/test_core.py) 的链接。
- **社区渴望与 Triton 团队沟通**：一位成员对 **Triton 团队** 停止与那些将 **Triton** 作为项目基础的开发者沟通表示困惑。
   - 他还提到正在开发一个名为 [teenygrad](https://github.com/teenygrad/teenygrad) 的 **基于 Rust 的 ML 框架**，该框架使用 **Triton** 作为其核心 DSL。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1387183769433145407)** (2 messages): 

> `cub library, CUDA` 


- **使用 `block_reduce.cuh` 代替完整的 `cub/cub.cuh`**：根据 [这个 PR](https://github.com/pytorch/pytorch/pull/156380)，不建议包含整个 `cub/cub.cuh` 头文件；应改用 `#include <cub/block/block_reduce.cuh>`。
- **新成员准备开始第一个 CUDA 项目**：一位新成员读完了《CUDA by Example》，正准备开始他们的第一个 CUDA 项目。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1387164446488662028)** (8 messages🔥): 

> `cuML toolkit versions, ThreadIdx usage in CUDA, Matrix multiplication` 


- **cuML 工具包版本问题已解决**：一位用户通过卸载其工具包并下载更精确的版本，解决了 **cuML** 的问题。
- **关于 CUDA 中 threadIdx 使用的澄清**：一位用户询问为什么在 CUDA 的基础矩阵乘法中，**threadIdx.y** 用于行，而 **threadIdx.x** 用于列。
   - 另一位用户解释说，`threadIdx.x` 是 **warps 布局** 的维度，这会影响内存访问的 coalesced 方式。
- **使用 X 和 Y 维度进行行列类比**：另一位用户对 **threadIdx.x** 和 **threadIdx.y** 的用法提供了直观的解释，将其与行和列分别在 x 和 y 方向上增加尺寸的方式联系起来。
   - 提问者发现这种构思很有帮助，理解了*在单行优先（row-major）数组中添加一列需要插入数组中的每个 colSize*。


  

---

### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1387443019178836032)** (7 messages): 

> `AMD Cloud Providers, rocprofv3 client vs rocprof` 


- **DigitalOcean 和 TensorWave 成为 AMD 云服务的首选**：成员们推荐 [DigitalOcean](https://www.digitalocean.com/) 和 [TensorWave](https://www.tensorwave.com/) 作为 **AMD 机器** 的优质云供应商，尤其适合小型项目和实验。
   - 提到的其他供应商还包括 **Hotaisle** 和 **Runpod**，其中一位成员指出 Hotaisle *相当不错*。
- **rocprofv3 客户端的范围受到质疑**：一位成员询问 **rocprofv3 client** 的愿景是否旨在未来某个时间点取代 **rocprof-compute** 和 **rocprof-sys** 的全部功能。
   - 另一位成员对响应速度表示惊讶，表明 **rocprofv3** 客户端的表现已经令人印象深刻。


  

---


### **GPU MODE ▷ #[intel](https://discord.com/channels/1189498204333543425/1233802893786746880/1387160283423969301)** (1 messages): 

> `Intel GPU atomic latency, VTune, SYCL device cycle counters, Ponte Vecchio` 


- **测量 Intel GPU Atomic 延迟**：一位成员询问如何使用 **VTune** 或 **SYCL device cycle counters** 测量 **Intel Ponte Vecchio GPU** 上每个线程的 Atomic 延迟。
   - 他们正在寻求关于如何准确测量该指标以进行性能分析和优化的建议。
- **用于 Atomic 延迟测量的工具**：用户正在探索 **VTune** 和 **SYCL device cycle counters** 等选项，以获取详细的延迟指标。
   - 这表明了对高层级 Profiling 工具和底层硬件计数器的双重兴趣。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1387321703041990808)** (28 messages🔥): 

> `MakoGenerate Feedback, VS Code Extension for MakoGenerate, LLM prompting issues, Kernel Tuner integration` 


- ****MakoGenerate** 免费部署至 H100 和 B200**：创作者宣布了 **MakoGenerate**，这是一个可以生成可部署至 **H100** 或 **B200** 的 GPU Kernel 的 AI Agent，并邀请用户在 [generate.mako.dev](https://generate.mako.dev) 提供反馈。
   - 一位用户建议举办一场比赛，看看用户是否真的能通过 *Prompt 调优出更好的 Kernel*。
- **用户强烈要求推出 VS Code 扩展**：一位用户建议开发一个 **VS Code 扩展**，以便在云端测试 Kernel 的编译、正确性和速度，从而实现本地开发和云端验证，因为他们不喜欢目前的聊天界面。
   - 创作者确认他们已经在开发 **VS Code 扩展**，并提供无限量的免费额度。
- **LLM 在处理 Prompt 时遇到困难**：用户注意到 **LLM** 有时会在提供的问题和 Prompt 之间切换，即使被明确指示忽略示例问题，这使得让 LLM 按预期执行任务变得更加困难。
   - 一位用户建议允许用户在 Agent 尝试后与其进行“对话”，以进一步优化输出。
- **Kernel Tuner 可以自动调优 MakoGenerate**：一位用户建议集成 **kernel_tuner** ([https://github.com/KernelTuner/kernel_tuner](https://github.com/KernelTuner/kernel_tuner)) 作为自动调优的扩展。
   - 他们认为如果 **MakoGenerate** *不需要选择特定问题* 会更好，因为 LLM 默认会偏向于选择问题。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1387171490964832369)** (4 messages): 

> `KernelLLM, GPU Kernels, Mirage-project` 


- **KernelLLM 需要 Prompt 格式化**：一位成员分享了如何为 **KernelLLM** 格式化 Prompt 以使模型达到最佳性能的 [指南](https://huggingface.co/facebook/KernelLLM/discussions/5#685b0903b3d048882566b17b)。
   - KernelLLM 期望实现 **Model(nn.Module)** 和 **get_inputs** 函数，对其他类型的输入不够灵活。
- **Mirage 生成 GPU Kernel**：一位成员分享了 **Mirage** 的链接，这是一个无需使用 Triton/CUDA 编程即可自动生成快速 **GPU Kernel** 的项目，并附带了相关的 [Tweet](https://x.com/mako_dev_ai/status/1937873917646897479?s=46&t=Z-_IUEOhekbm7eaIddmkvQ) 链接。
   - 该项目的仓库可以在 Google Share 的 [此处](https://share.google/41nz6vDcGvu45uUIc) 找到。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1343002580531417211/1387187241712881674)** (6 条消息): 

> `Triangle Multiplicative Update, GPU credits, Competition Details` 


- ****Triangle Multiplicative Update** 登陆 GPU Mode**: 一个基于 AlphaFold 系列模型中使用的 **Triangle Multiplicative Update** 的新题目已经发布，可在 [GPU Mode](https://tinyurl.com/gpumode-trimul) 上查看。
   - 该题目同时适用于 **NVIDIA** 和 **AMD** 硬件。
- ****GPU 积分** 竞赛说明**: 一名成员询问了如何获取免费的 **GPU 积分** 以参加 Triangle Multiplicative Update 竞赛。
   - 官方澄清，Discord 上的 [提交界面](https://gpu-mode.github.io/discord-cluster-manager/docs/intro/) 允许用户在所有 GPU 上免费提交、测试和基准测试 Kernel，无需支付任何费用。
- **新挑战引发热潮**: 成员们对新的 **Triangle Multiplicative Update** 挑战赛表示兴奋。
   - 有人感叹道：*"噢，这题目真酷"*。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1387192542738514001)** (39 条消息🔥): 

> `Leaderboard Results, vectorsum benchmark, trimul benchmark, amd-identity benchmark, B200 performance` 


- **B200 基准测试表现出色**: 一名用户在 **B200** 的 `trimul` 排行榜上以 **7.92 ms** 的成绩获得 **第一名**。
   - 另一名用户以 **8.20 ms** 获得 **第二名**。
- **A100 完美完成 trimul 任务**: 一名用户在 **A100** 的 `trimul` 排行榜上以 **20.0 ms** 获得 **第一名**。
   - 另一名不同的用户以 **23.3 ms** 获得 **第二名**。
- **MI300 惊艳表现**: 一名用户在 **MI300** 的 `trimul` 任务中以 **11.0 ms** 夺得 **第一名**，并在 **MI300** 的 `amd-identity` 任务中创下 **24.9 µs** 的个人最佳成绩。
- **vectorsum 在多种 GPU 上取得佳绩**: 在 `vectorsum` 任务中，一名用户在 **H100** (**91.5 µs**) 和 **T4** (**781 µs**) 上获得 **第二名**，并在 **T4** 上获得 **第三名** (**806 µs**)。
   - 该用户还在 **L4** 上排名 **第 5**、**第 6** 并多次成功运行，在 **A100** 上也获得了多个名次 (**151 µs**, **159 µs**)。


  

---


### **GPU MODE ▷ #[status](https://discord.com/channels/1189498204333543425/1343350424253632695/1387187660900012104)** (1 条消息): 

> `AMD, NVIDIA, new leaderboard, hardware optimization` 


- **AMD 和 NVIDIA 新排行榜挑战揭晓！**: 一个新的排行榜题目现已面向 **AMD** 和 **NVIDIA** 硬件开放，详细介绍请见 [此处](https://tinyurl.com/gpumode-trimul)。
- **基准测试之乐：Trimul 成为焦点**: 这项名为 **Trimul** 的新挑战旨在测试 **AMD** 和 **NVIDIA** GPU 的极限，将硬件优化推向最前沿。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1387377674233577512)** (24 条消息🔥): 

> `Lua actions 注释, set_inventory 问题, ModuleNotFoundError: No module named 'agents', LuaSurface vs LuaPlayer API, Factorio 测试失败` 


- **调整 Lua Actions 加载以进行调试**：成员们讨论了是否应该注释掉默认加载到游戏中的 **Lua actions** 以辅助调试，并建议增加一个 *verbose* 标志来控制加载过程中打印的详细程度。
   - 该提案包括在启用 verbose 标志时打印每个加载的项目，否则仅打印 *Start/Finished loading actions*。
- **通过 1-based Indexing 解决 set_inventory 难题**：一位用户对 `set_inventory` 为何没有清空库存感到困惑，即使在调用命令后，检查发现给定的物品（如 `{'iron-chest': 2, ...}`）仍然存在。
   - 问题最终得到解决，原因是 **Factorio Lua 使用的是 1-based indexing（从 1 开始的索引）**，而不是从 0 开始，因此将代码改为 `self.add_command('clear_inventory', agent_idx **+ 1**)` 修复了该问题。
- **缺失 Agents 模块导致执行中断**：运行 `uv run python env/src/gym_env/run_eval.py --run_config eval/open/independent_runs/run_config_example_lab_play.json` 时出现了 `ModuleNotFoundError: No module named 'agents'` 错误。
   - 此错误导致评估脚本无法运行，因为它找不到必要的 `GymAgent` 类。
- **LuaSurface 模拟 LuaPlayer 的手动操作灵活性**：编写了一个脚本来对比 `LuaSurface` 与 `LuaPlayer` API 的行为，发现当使用 `build_check_type = blueprint_ghost | manual` 时，`LuaSurface.can_place_entity` 可以作为 `LuaPlayer` 的直接替代品。
   - 在靠近水边的 3x3 网格上进行的测试确认了这一点，但仍需进一步测试在远离水的地方放置抽水泵（offshore pumps）或在没有资源的地方放置钻头的情况，以确认其行为是否始终一致；目前已知 `build_check_type.manual` 有效，而 `blueprint_ghost` 则无效。
- **Factorio 测试套件因 RCON 连接故障而失败**：许多测试由于 `AttributeError: 'FactorioNamespace' object has no attribute 'set_inventory'` 和 `RCONConnectError: Failed to communicate authentication setup to the server` 而失败。
   - 成员分享称这就是他们之前提到的问题，即一种 *所有测试都开始因相同错误而失败* 的状态。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1387199760993095853)** (1 条消息): 

> `Persistent Ping Pong GEMM Kernel, sm90 的 CuTe DSL, TMA 传输, MMA 启动, Barrier 同步问题` 


- **尝试实现 Persistent Ping Pong Kernel**：一位成员尝试使用 **CuTe DSL** 为 **sm90** 编写一个 persistent ping pong GEMM kernel，其中一个 producer warpgroup 负责启动 **TMA transfers**，两个 consumer warpgroups 负责启动 **MMAs**。
   - 他们在开发过程中遇到了 [barrier 同步问题](https://github.com/NVIDIA/cutlass/issues/2418)。
- **CuTe DSL 的生产力受到赞赏**：尽管面临同步挑战，该成员仍赞扬了 **CuTe DSL** 几乎瞬时的编译时间、易于打印/调试的特性以及 Pythonic 的风格。
   - 他们指出：*这比在 C++ 中做同样的事情体验要好得多*，并强调了开发者生产力的提升。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1387145746339467324)** (96 条消息🔥🔥): 

> `RL 破解, GRPO vs Dr. GRPO, 前向传播 (Forward Propagation), 进化方法, 版权材料的合理使用` 


- **GritLM 占据 NER 阵地**：一名成员建议，对于命名实体识别 (NER)，ContextualAI 的项目 [GritLM](https://github.com/ContextualAI/gritlm) 值得关注。
   - 关于图像着色，他们指出如果涉及 **Gaussian Splatting**，它可能不再是一个探索较少的 ML 领域，其主要用例是 **Digital Twins**（数字孪生）。
- **Dr. GRPO 减少了 GRPO 的“废话”**：在分享了一张图片后，成员们讨论了 **Dr. GRPO** 如何在保持性能的同时减少 **GRPO** 的冗余输出（yapping），并引用了 [此 Discord 链接](https://discordapp.com/channels/714501525455634453/853983317044756510/1387157193656373368)。
   - 一位成员提到了用于实现 **GRPO** 的 [YouTube 视频](https://www.youtube.com/watch?v=K34gBCjzni8) 和 [论文](https://arxiv.org/abs/2501.12948)，并指出 **Dr. GRPO** 是在此基础上构建的。
- **深入探讨前向传播 (Forward Propagation) 细节**：关于 **Forward Propagation (FF-prop)**，会议澄清了 **LeCun** 指的是标准的前向推理过程，即层在训练后运行，不含反向传播。
   - 会议强调 **Hinton** 的 **Forward Forward** 无法扩展，但 **Forward Gradients** 确实有效，它被描述为反向传播的转置，是寻找导数最基本的方法。
- **合理使用 (Fair Use) 面临法律框架界定**：一名成员分享了来自北加州巡回法院的一份 [法庭文件](https://storage.courtlistener.com/recap/gov.uscourts.cand.434709/gov.uscourts.cand.434709.231.0_2.pdf)，概述了在针对 **Anthropic** 的 AI 训练案件中，什么构成了对版权材料的合理使用。
   - 建议在 Discord 上创建一个法律板块，用于追踪相关的立法和裁决。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1387148411945619476)** (36 条消息🔥): 

> `RWKV 重复问题, Brain on LLMs 研究, RLVR 在数学推理上的应用, Qwen 模型代码推理` 


- **定期报告的机器人重复唠叨**：成员们提到 **RWKV** 往往会产生大量重复内容，这引发了关于模型特性及潜在修复方案的讨论。
   - 分享了一个相关的 arXiv 论文链接 ([https://arxiv.org/abs/2506.09278](https://arxiv.org/abs/2506.09278)) 以供进一步探索，尽管最初的关注度似乎不高。
- **Brain-on-LLMs 的明显错误让浏览者困惑**：一位成员发现了一项名为 *Your Brain on LLMs* 的**研究**，并分享了一张 [截图](https://cdn.discordapp.com/attachments/1045297868136779846/1387359102975610880/Snipaste_2025-06-25_16-09-14.png?ex=685db71a&is=685c659a&hm=9cfa39ed78e88f2ca714cf645ba04ec641289dd15a88077b4b4669245859e86a)，突出了**校对问题**，如深色模式下的字体和文本颜色不一致。
   - 尽管存在视觉缺陷，该成员在阅读 30 页后指出该论文“实际上相当不错”，并邀请其他人阅读这篇 **206 页的论文** ([https://arxiv.org/pdf/2506.08872v1](https://arxiv.org/pdf/2506.08872v1))。
- **LLM 杠杆化学习负载减轻**：一位成员讽刺地考虑使用 LLM 来总结关于 LLM 的 **206 页论文**，承认正在将认知负荷转移到 AI 系统上。
   - 他们发现了支持自己倾向于使用 LLM 处理认知任务的证据，并开玩笑说通过误导模型某些函数存在，可以诱导模型写出更好的单元测试。
- **RLVR 揭示了强大的推理能力，但真的可靠吗？**：一位成员分享了一篇关于**带有可验证奖励的强化学习 (RLVR)** 的论文 ([https://arxiv.org/abs/2506.10947](https://arxiv.org/abs/2506.10947))，指出其发现即使在存在伪奖励 (spurious rewards) 的情况下，也能激发模型强大的数学推理能力。
   - 论文强调，虽然 **RLVR** 提高了 **Qwen2.5-Math-7B 的 MATH-500 性能**，但伪奖励对 Qwen 模型有效，但在 **Llama3** 或 **OLMo2** 等其他模型上失败，其确切机制尚不明确。


  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1387302455255240835)** (2 条消息): 

> `Discord 用户否认声明` 


- **用户登录后立即否认关联**：一名用户声明：*我不参与此群组或其中的任何人，我不知道我为什么在这里，可能是被第三方添加的，我不支持此群组分子的任何行为。*
- **用户询问另一名用户的名字**：另一名用户只是简单地询问另一位用户：*lucas?*。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1387156827740962929)** (5 messages): 

> `R1-Zero-Like Training, Reinforcement Learning, AlphaGenome, Gemini CLI, Dwarkesh Podcast` 


- **类 R1-Zero 训练探讨**：讨论了一篇题为 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783) 的论文。
   - 其他讨论的论文包括 [Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model](https://arxiv.org/abs/2504.13837)、[Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711) 以及 [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)。
- **DeepMind 发布 AlphaGenome**：DeepMind 发布了一篇关于 [AlphaGenome](https://deepmind.google/discover/blog/alphagenome-ai-for-better-understanding-the-genome/) 的博客文章，这是一种用于更好理解基因组的 AI。
   - 该内容在频道中被链接并进行了讨论。
- **Google 推出 Gemini CLI**：Google 推出了 [Gemini CLI](https://blog.google/technology/developers/introducing-gemini-cli-open-source-ai-agent/)，这是一个免费且开源的 AI Agent，可将 Gemini 直接引入开发者的终端。
   - 它宣称能为个人提供*无与伦比的访问权限*。
- **Frontier Models 需要更多实验**：Anthropic 研究员 **Sholto Douglas** 和 **Trenton Bricken** 认为，许多论文使用较小的模型和算力，这可能无法反映 Frontier Models 的情况。
   - 引用他们在 [Dwarkesh podcast 9:10 处](https://youtu.be/64lXQP6cs5M) 的发言，他们建议需要在最大的 **DeepSeek model** 上进行实验。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1387178043453472829)** (100 messages🔥🔥): 

> `Thinking Machines Lab, Warp 2.0, NeoBERT, Airtable AI, Long-Context Q&A Systems` 


- **Murati 的 Thinking Machines Lab 专注于业务领域的 RL**：根据 [这篇文章](https://xcancel.com/steph_palazzolo/status/1937284120062706004)，Mira Murati 的新 AI 初创公司 **Thinking Machines Lab** 正专注于面向业务的 **Reinforcement Learning (RL)**。
- **Warp 2.0 进入 Agentic 开发领域**：**Warp 2.0** 被介绍为一个 Agentic 开发环境，使开发者能够“通过 prompt 编码”而非手动编码。根据 [这条推文](https://xcancel.com/warpdotdev/status/1937525185843752969)，它在 **Terminal-Bench** 上排名第一，并在 **SWE-bench Verified** 上达到了 **71%** 的分数。
- **Airtable 的 Omni AI Agent 重塑应用平台**：根据 [这条推文](https://xcancel.com/howietl/status/1937577526634987595)，**Airtable** 已重新发布为 **AI-native 应用平台**，通过 **Omni** 转向完全的“重塑”。Omni 是一个 AI 应用构建 Agent，允许用户通过对话方式构建强大的应用。
- **Liquid AI 打造简洁的推理模型**：来自 **Liquid AI** 的 Maxime Labonne 宣布了一个 **10 亿参数的推理模型**，该模型既准确又简洁，结合了 **Supervised Fine-Tuning (SFT)** 和 **GRPO (Generative Reinforcement Learning from Human Preferences)**，详情见 [这条推文](https://xcancel.com/maximelabonne/status/1937819336204304692)。
- **OpenRouter 获得 AI 模型市场支持**：Deedy 宣布支持 **OpenRouter**，这是一个 AI 模型市场，为开发者提供通过单一 API 访问 **400 多个 LLMs** 的权限，每年处理 **100 万亿个 tokens**，详见 [这条推文](https://xcancel.com/deedydas/status/1937902948920811729)。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1387167669534458047)** (61 messages🔥🔥): 

> `Facebook 书籍盗版诉讼, GPU 额度使用, Yacine 入职状态, 编程协作, Anthropic 竞争` 


- **Facebook 在书籍纠纷中受挫**：一条 [推文](https://x.com/AdamEisgrau/status/1937480346976813454) 指出，尽管法院裁定训练具有转换性（transformative），但 **Facebook** 可能并未在书籍盗版诉讼的盗版部分获胜。
- **免费 GPU 额度激发微调幻想**：一位成员正在寻求在没有编程经验的情况下，如何使用 **50 美元的免费 GPU 额度** 来处理 LLM，考虑使用 **Claude 4 Sonnet** 或 **Gemini 2.5 Pro** 等模型来编写代码。
   - 建议包括使用该额度来 **微调 LLM**，但也建议不要为了花掉而盲目使用。
- **Nous 讨论 Yacine 入职事宜**：成员们讨论了为什么 **Nous Research** 还没有聘用前 **X** 工程师 **Yacine**，对其技能组合以及他是否适合 ML 角色意见不一。
- **Eager Egg 寻求编程伙伴**：成员们讨论了在 Nous VC 一起写代码，一位自称 **egg** 的成员收到了未来在 **Rust** 和其他项目中进行协作的邀请。
   - 另一位成员建议，**如果 8B 模型在网站上无法运行，请尝试将其下载到你的 PC 上**。
- **Anthropic 增强 Artifacts 功能**：**Anthropic** 增加了 *LLM 集成能力*，现在有了一个类似 **Google** 的查找 artifacts 的地方。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1387313789443112961)** (5 messages): 

> `OpenRouter 问题, LLM 推理限制, Token 限制` 


- **OpenRouter 提供商虚标 Token 限制**：一位成员指出，许多 **OpenRouter** 提供商似乎误报了其最大输出 Token，限制了推理型 **LLM** 的效用。
   - **16k tokens** 的硬限制导致无法运行大多数 **AIMS** 问题，且支持团队未予回复。
- **针对问题提供商的新评价系统？**：一位成员提到，可能会有一个评价系统供用户举报特定提供商的问题。
   - 原帖作者通过选择那些确实能提供承诺 Token 限制的特定提供商解决了问题。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387157480722665533)** (12 messages🔥): 

> `类 R1-Zero 训练, RL 激励推理, RL 微调子网络, 伪奖励 (Spurious Rewards), Anthropic 对玩具模型的质疑` 


- **类 R1-Zero 训练的批评出现**：论文 *Understanding R1-Zero-Like Training: A Critical Perspective* [质疑了训练方法论](https://arxiv.org/abs/2503.20783)，这些方法与 R1-Zero 类似。
   - 还提到了其他论文，包括一篇关于虚拟现实强化学习中 *伪奖励 (Spurious Rewards)* 的论文 [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)。
- **Anthropic 对小模型 RL 研究表示怀疑**：Anthropic 研究员 **Sholto Douglas** 和 **Trenton Bricken** 在 [Dwarkesh 播客](https://youtu.be/64lXQP6cs5M?t=550) 中辩称，由于依赖较小的模型和有限的算力，分析 **RL** 的论文可能无法反映现实世界的性能。
   - 他们建议，实验理想情况下应在最大的 **DeepSeek** 模型上进行，以产生更具代表性的结果。
- **Hermes 4 (671b) 即将问世**：一位用户宣布，基于 **671b** 参数模型的 **Hermes 4** 将在下个月左右发布。
   - 针对托管质量的担忧，他们向社区保证托管安排已经落实。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1387157480722665533)** (12 messages🔥): 

> `R1-Zero-Like Training, RL Incentivizing Reasoning Capacity, RL Finetunes Small Subnetworks, Spurious Rewards in RLVR, Dwarkesh Podcast` 


- **R1-Zero 训练遭到质疑！**: 一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=z3awgfU4yno) 以及多篇论文，包括 [Understanding R1-Zero-Like Training: A Critical Perspective](https://arxiv.org/abs/2503.20783)、[Does Reinforcement Learning Really Incentivize Reasoning Capacity in LLMs Beyond the Base Model](https://arxiv.org/abs/2504.13837)、[Reinforcement Learning Finetunes Small Subnetworks in Large Language Models](https://arxiv.org/abs/2505.11711) 以及 [Spurious Rewards: Rethinking Training Signals in RLVR](https://arxiv.org/abs/2506.10947)。
- **Anthropic 批评 RL 论文！**: Anthropic 研究员 Sholto Douglas 和 Trenton Bricken 在 [Dwarkesh 播客](https://youtu.be/64lXQP6cs5M?t=550) 中辩称，许多 **RL 论文** 使用的是较小/玩具模型，这可能无法准确反映前沿模型的动态。
- **基于 671b 的 Hermes 4 即将到来！**: 一名成员宣布，基于 **671b** 参数模型的 **Hermes 4** 预计将在下个月左右发布。
- **DeepSeek 托管困境！**: 一名成员询问谁将托管 **Hermes 4**，并指出目前 OpenRouter 上 **DeepSeek V3** 或 **R1** 的托管服务通常速度慢、价格贵或不稳定。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1387205150644506746)** (5 messages): 

> `Chris Interview` 


- **请求 Chris 访谈链接**: 一名成员询问 Chris 提到的访谈链接。
   - 另一名成员提供了一个 [YouTube 视频](https://www.youtube.com/watch?v=04_gN-C9IAo) 链接作为回应。
- **另一个话题**: 另一名成员询问了其他事情。
   - 另一名成员作出了回应。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1387145474657619999)** (29 messages🔥): 

> `Tokio Arc<Mutex<T>>, Mojo Async Plans, Linear Types, Effect Generics, InlineArray Move Semantics` 


- **Mojo 旨在避开 Rust 的 Async 困境**: Mojo 旨在通过更好的 Async 运行时和 [线性类型 (Linear Types)](https://github.com/modular/modular/pull/3946) 来改进 Rust 的 Async 难题，从而避免对 `Arc<Mutex<T>>` 等结构的需求。
   - 通过控制线程间的数据移动并确保数据不会过早被释放，Mojo 寻求消除与 Rust Async 相关的常见问题，可能在倾向于 Thread-per-core 以简化开发的同时，提供可选的任务窃取 (Work Stealing) 机制。
- **效应泛型解决 Mojo 中的函数着色问题**: Mojo 正在探索 **效应泛型 (Effect Generics)** 以解决大多数库的函数着色 (Function Coloring) 问题，详情见此 [PR](https://github.com/modular/modular/pull/4728)。
   - 这种方法结合效应泛型，允许编译器/运行时为程序选择“最佳”的 IO API，除非涉及自定义 IO API 绑定。
- **Mojo 字典报错信息模糊**: 一位 Mojo 新用户反馈在处理 **Dict 结构体** 时遇到了令人困惑的错误信息，特别是关于在不带括号的情况下使用 `.value`、`.keys` 和 `.items`。
   - 错误信息 *"statements must start at the beginning of a line"* 被认为没有帮助，该用户被要求在 [GitHub 上提交 Issue](https://github.com/modular/modular)，建议提供更具描述性的错误信息。
- **InlineArray 的 Moveinit 行为分析**: **InlineArray** 在移动操作 (`b = a^`) 期间的行为受到质疑，有人担心元素的拷贝构造函数和移动构造函数都没有被调用，这可能暗示存在 Bug。
   - 看起来 **InlineArray** 在移动初始化期间执行的是按位拷贝 (Bitwise Copy)，缺少显式的 moveinit。


  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1387418444139335722)** (5 messages): 

> `TorchScript Compilation, Inference Session, Moving Trained Artifacts, ONNX Loading Issue` 


- **TorchScript 需要 Torch 环境**：一位用户意识到，使用 **InferenceSession** 编译 **TorchScript** 文件需要 **Torch 环境**。
   - 他们对必须依赖 **Torch** 表示沮丧。
- **移动训练产物**：一位成员正尝试将训练好的产物从训练服务器移动并推送到 **inference API server**。
   - 他们询问是否有办法直接保存和加载 max 编译后的模型。
- **尝试 ONNX 但遇到文件格式错误**：有人尝试使用 **ONNX** 以避免在容器中包含 **Torch**，并参考了[这篇博客文章](https://www.modular.com/blog/bring-your-own-pytorch-model)。
   - 然而，对于一个有效的 **.onnx** 模型，他们遇到了 *unknown file format error*（未知文件格式错误），并正在寻求将 **ONNX** 加载到 **inference session** 的帮助。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1387163051639177257)** (36 messages🔥): 

> `OpenAI issues, SIMBA errors, Discord DSPy tag, dspy.Prediction anti-pattern` 


- **OpenAI API 面临运行时间问题**：一位成员报告称，由于 [OpenAI API 的问题](https://platform.openai.com/docs/status)，他们的**应用程序已宕机**。
   - 收到的错误是 `HTTP/1.1 404 Not Found`，这表明请求的资源无法找到。
- **SIMBA 错误调试深度探讨**：成员们讨论了与冻结子模块和 predictor 清单相关的 [SIMBA 错误](https://github.com/stanfordnlp/dspy/blob/main/dspy/teleprompt/simba.py)。
   - 解决方案包括确保从 `name_predictors` 返回的 predictor 与 `append_rule` 和 `append_demo` 期间迭代的 predictor 保持一致，特别是在使用 `._compiled = True` 时。
- **对 Discord DSPy 标签的期望**：一位成员建议创建一个 **Discord DSPy 标签**显示在用户名旁边，以展示 DSPy 专业知识。
   - 有人指出，根据 [Discord's Guilds FAQ](https://support.discord.com/hc/en-us/articles/23187611406999-Guilds-FAQ)，此类标签需要 Discord 服务器至少获得 **3 次 boosts**。
- **探讨 dspy.Prediction 返回模式**：一位成员询问从模块的 forward 方法返回 **dspy.Prediction** 以外的内容是否被视为一种反模式（anti-pattern）。
   - 另一位成员回答说，虽然这可能可行，但可能会导致问题，特别是如果 metric 函数不知道如何处理输出，从而影响优化。
- **Shopify 创始人加入 DSPy**：Shopify 创始人 [Tobi Lutke 加入了 DSPy](https://x.com/tobi/status/1937967281599898005)。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1387272443709620234)** (5 messages): 

> `Deep Dives, Chrome Extension, Time Constraints` 


- **Deep Dives 持续时间最长**：一些用户报告称，最长的 deep dives 可以超过 **110mc**。
   - 目前还没有关于什么是 "deep dive" 或这具体指代什么的明确描述。
- **确认 PrintFriendly Chrome 扩展**：一位用户确认图片中的扩展程序是 **PrintFriendly**，并在 [Chrome Web Store](https://chromewebstore.google.com/detail/printfriendly-print-pdf-a/ohlencieiipommannpdfcmfdpjjmeolj) 中找到了它。
   - PrintFriendly 可以将网页转换为打印机友好格式和 PDF 格式。
- **时间限制大多被忽略**：一位用户询问如何让机器人遵守时间限制，并指出它要么忽略限制，要么将其延长至**最多 18 分钟**。
   - 另一位用户表示这与大量来源有关，并要求它在输出中包含每一个来源。


  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1387179137051000852)** (29 条消息🔥): 

> `NotebookLM Generation Limits, Vimeo Video Sources, Podcast Monetization with AI Voice, PDF vs. MD for NotebookLM, NotebookLM Video Overviews` 


- ****NotebookLM 的限制困扰****：用户对 NotebookLM 在自定义提示词（customize prompt）*之前*不提示已达到生成限制表示沮丧，这可能导致工作丢失。
   - 成员们想知道，当他们返回时，非常长的自定义提示词是否会保留在笔记本中。
- ****Vimeo 在 NLM 中的使用困扰****：用户报告了在 NotebookLM 中使用 Vimeo 视频作为源的问题，安全功能阻止了内容访问。
   - 一位成员建议使用 [cobalt.tools](https://cobalt.tools/) 下载视频作为变通方案，而另一位成员则询问如果已经上传了转录文本（transcripts），是否就不再需要视频本身了。
- ****AI 音频的 AdSense 模糊地带****：一位用户询问 YouTube 是否允许使用 AI 生成语音和来自 NotebookLM 内容的频道进行变现。
   - 另一位成员指出 AI 和版权（copywrite）之间存在*灰色地带*，并建议研究 YouTube 关于 AI 内容变现的规则。
- ****PDF 更利于高效处理****：在一个消息线程中，一位用户询问 PDF 还是 MD 格式更适合 NotebookLM。
   - 另一位成员回答说 **PDF 是更好的格式**。
- ****孟加拉语失误：口音音频之苦****：一位用户报告说 NotebookLM 中的孟加拉语音频概览使用的是 **西孟加拉邦口音，而不是标准孟加拉语口音**。
   - 他们还询问该功能是否终于开始支持其他语言。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1387282124851253309)** (21 条消息🔥): 

> `tinygrad refactor bounties, JIT testing function, RING=0 scheduler heuristic, FUSE_OPTIM, NCCL with cuda graphs` 


- **TinyGrad 重构悬赏引发关注！**：成员们讨论认为，重构悬赏是了解 **tinygrad 内部机制（internals）** 的绝佳方式，其中一个案例涉及 **JIT 测试函数**。
   - 一位成员提交了一个 PR 来处理数组输入，从而导致测试用例失败。
- **调度器启发式大幅缩减图大小！**：使用简单的 **调度器启发式（scheduler heuristic）** 配合 `RING=0`，将最大的 **graphexec 从 5k 降至 2k**。
- **FUSE_OPTIM 未能生效！**：`FUSE_OPTIM=1` 似乎没有任何效果，因此该成员打算尝试非贪婪搜索（non-greedy search）。
- **NCCL 很好地处理了 CUDA Graphs！**：一位成员询问 **NCCL** 是如何处理 CUDA graphs 的，它似乎运行良好，不像 tinygrad 的实现。
- **输入张量引发麻烦！**：一位成员询问他们被关闭的 PR，该 PR 修复了传递列表时输入张量（input tensors）为空的问题。
   - 他们写了一个递归函数来提取它们，但回复明确指出 *该修复是错误的*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1387254788298641408)** (4 条消息): 

> `Gradient Calculation, Zero-Dimensional Tensors, Constant Gradient Issues` 


- **梯度计算之谜**：一位用户质疑为什么在给定场景下 `a` 的梯度是一个*任意数字*而不是 **4**。
   - 另一位成员解释说，问题出在需要梯度的 **零维张量（zero-dimensional tensors）** 上，建议禁用这些张量，并推荐将 `a` 更改为 `Tensor([2.0], requires_grad=True)`。
- **零维张量故障**：问题的出现是因为只有 **标量值（scalar values）** 可以进行反向传播（backwarded），而用户的 `b` 恰好是一个标量，导致了 *垃圾输出*。
   - *垃圾输出* 是由于 **常数梯度（constant gradients）** 具有无关的算术逻辑单元（**ALUs**）造成的；具体数值 **6.7725887** 是按 `4*log(2)+4` 计算得出的。
- **常数梯度导致 UNIQUE 问题**：异常的梯度值可能是由于计算图中的 **UNIQUE 问题** 导致的。
   - 参与计算的常数导致了该问题，常数梯度最终使用了无关的 ALU。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1387304482668413108)** (17 messages🔥): 

> `AI 生成的网站，MCP 客户端架构，基于浏览器的 MCP 客户端，Hugging Face MCP 身份验证，Reddit 版主` 


- **Google 计划让 AI 编写网页**：成员们回想起了 Google I/O 的发布内容，即 **AI 将编写网站并生成内容**，而这些内容仅供其他 **AI 抓取和总结**。
   - 另一位成员开玩笑说 *Google 肯定在酝酿 Web 的终结*，很快 Chrome 将变成一个聊天界面。
- **MCP 客户端不必是桌面应用**：在回答关于 MCP 客户端/主机架构的问题时，一位成员澄清说 *它可以是任何形式，网页、CLI 等*。
   - 该成员有兴趣在云端运行一个 **基于 daemon 的 MCP 客户端**，并使用一个 **轻量级的基于 REST 的代理** 来处理浏览器 UI 通信，将 HTTP 转换为 MCP。
- **基于浏览器的 MCP 客户端想法很有趣**：一位成员建议直接在浏览器中构建 **MCP 客户端**，甚至可能在那里也创建 **MCP server**，以避免 SSE 和流式传输的复杂性。
   - 他指出他将研究该选项，这可能是一个有趣的想法。
- **Hugging Face MCP 身份验证触发器已可用**：成员们讨论了 Hugging Face 的 MCP 身份验证。
   - 他们指出你需要通过 [https://hf.co/mcp?login](https://hf.co/mcp?login) 触发身份验证，并且默认情况下是匿名的。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1387467811835613244)** (2 messages): 

> `MCP 托管服务，mcp-cloud.ai，MCP server 部署` 


- **MCP Cloud 推出托管服务平台**：[MCP Cloud](https://mcp-cloud.ai) 推出了专门针对 **MCP servers** 的托管平台，提供专用实例、JWT 身份验证和实时日志，可在数秒内完成部署。
   - 它支持多工作流和复制/粘贴集成，特别是与 **N8N** 的集成，面向需要可靠、安全 MCP 基础设施的开发者和团队。
- **MCP Cloud 寻求反馈与合作**：该平台正积极寻求反馈以进行改进，并寻找成熟的 **MCP servers** 以通过其平台提供服务。
   - 其特性包括 *专用实例*、*生产级* 基础设施和 *多工作流支持*。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1387167711632560228)** (19 messages🔥): 

> `Manus 宕机，额度丢失，Manus 变笨，邀请码，1k` 


- **额度丢失之际 Manus 的可靠性受到质疑**：多位用户报告了 **Manus** 的问题，包括 **卡在思考阶段** 并抛出 **内部服务器错误**，同时还有对近期 **额度丢失** 的担忧。
   - 一些用户发表意见认为 *Manus 变得更笨了且会犯错*。
- **提供邀请码，额度消耗引发讨论**：在关于 **Manus 增加额度消耗** 的讨论中，一位用户提供了 **邀请码**。
   - 有人声称 *它肯定消耗了更多额度*。
- **分配的额度有限**：一位用户报告仅收到 **1k 额度**，未提供更多上下文。
   - 目前尚不清楚这是 bug 还是预期行为。
- **Manus 拒绝分享 VS Code 密码**：一位尝试在 **Manus 的电脑** 上访问 **VS Code** 的用户遇到了需要密码的登录提示，而 **Manus** 拒绝提供。
   - 该用户被告知 *检查 .../config yaml 处的配置文件以获取密码*。
- **Quality Agent 模式 vs High Effort 模式**：一位用户询问新的 **Quality Agent 模式** 是否与之前的 **High Effort 模式** 相同。
   - 未提供确切答案。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1387360682533584957)** (3 messages): 

> `负责任的 AI，AI 安全，公平性，ML 暑期学校，AI 黑客松` 


- **寻找负责任 AI 的频道**：一位成员询问是否有专门讨论 **负责任的 AI**、**AI 安全** 或 **公平性** 的频道。
   - 给出的消息中未提到相关资源。
- **ML 暑期学校 Google Group 访问权限**：一位成员询问是否有人被 **ML 暑期学校** 录取并能访问其 **Google Group**。
   - 消息中未给出回复。
- **欧洲的 AI 黑客松和暑期学校**：一位成员请求推荐欧洲专注于 **AI** 的优质 **AI 黑客松** 或 **暑期学校**。
   - 消息中未给出回复。


  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1387250392101818539)** (10 messages🔥): 

> `Automated Code Review, ML Fairness OSS Toolkit, Geometric Deep Learning, Transformer Architecture Modification` 


- **Harsh 在 Ethereum 自动化代码审查**：一位加州大学戴维斯分校（UC Davis）的计算机科学系学生，同时也是 **Ethereum Foundation** 的 AI/安全工程实习生，正致力于自动化代码审查和漏洞检测流程。
   - 他使用 **Perplexity** 进行初步的主题调研，并正在研究 **LLMs** 的对抗性角度以及 **LLM** 记忆。
- **Tamara 维护 ML 公平性工具包**：驻扎在柏林的一位计算语言学和 **NLP** 硕士生，负责维护 **fairlearn**（一个 **ML** 公平性开源工具包）。
   - 她旨在将其 **ML** 公平性专业知识应用于 **CL** 领域，并在协助 **Aya project** 后重新加入社区。
- **Aniket 探索 Geometric Deep Learning**：正在攻读 AI 和机器学习硕士学位的 Aniket 正在深入研究 **Geometric Deep Learning** 领域的主题。
   - 他希望与 AI 社区互动并学习。
- **Sam 在巴黎完成 AI 硕士学位**：在巴黎即将完成数据科学和 AI 硕士学位的 Sam 正在从事**基因组学和生物信息学**方面的工作。
   - 他利用 **Hugging Face, Langchain, 和 Colab** 等工具，并期待社区交流。
- **AI 工程师修改 Transformer 架构**：一位 AI 工程师/研究员专注于为小型用例修改 **Transformer Architecture**。
   - 该工程师还发布了一份名为 *Agents: All You Need* 的时事通讯。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1387188407398301738)** (3 messages): 

> `MCP Server, Next.js app, Agent Memory, Meeting Notetaker agent, NotionHQ` 


- **使用 Next.js 发布兼容 Claude 的 MCP Server**：LlamaIndex 宣布了一个新的开源模板仓库，用于将兼容 Claude 的 **MCP server** 构建为具有完整 **OAuth 2.1 支持**的 **Next.js app**。
   - 该项目是在内部黑客日期间创建的，简化了远程 **Model Context Protocol servers** 的创建以实现无缝运行。
- **Agent 通过新的 Memory Blocks 获得记忆**：LlamaIndex 正在与 **AIMakerspace** 讨论为 **LlamaIndex Agents** 提供新的 **memory blocks**。
   - 他们将涵盖持久化聊天历史、**long-term memory** 以及内存的自定义逻辑；更多详情请见[链接](https://t.co/D4ZiBK54Fh)。
- **为 NotionHQ 构建会议记录 Agent**：成员们现在可以为 **NotionHQ** 构建一个**会议记录 Agent**。
   - **Zoom** 宣布了 **RTMS**，允许使用来自 Zoom Meetings 的实时数据；完整示例请见[此处](https://t.co/4m2IOcz7Se)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1387233956323262567)** (5 messages): 

> `AI Newsletters with real-world LLM use cases, LlamaCloud parsing job ID errors, LlamaCloud API bugs` 


- **寻找具有实际 LLM 案例展示的 AI 时事通讯**：一位成员询问是否有关注 **LLMs** 真实世界用例的 **AI newsletters**，而不仅仅是模型发布和更新。
   - 该成员正在寻找能够突出人们如何积极使用 **LLMs** 进行构建的时事通讯。
- **LlamaCloud Job ID 混淆问题**：一位成员报告称，在尝试根据[此文档](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get)使用 **LlamaCloud API** 获取解析任务结果时，遇到了 "invalid job_id" 错误。
   - 他们使用了 **LlamaCloud API key** 进行身份验证，并使用了从解析器的 `load_data()` 方法中获取的 job_id。
- **LlamaCloud API 的参数难题**：一位成员根据 **SDK** 的用法建议，API 调用末尾可能缺少一个 `/{result_type}` 参数（例如 `/json`），并引用了 [LlamaCloud 文档](https://docs.cloud.llamaindex.ai/API/get-job-json-result-api-v-1-parsing-job-job-id-result-json-get)。
   - 他们链接了相关的 [SDK 代码](https://github.com/run-llama/llama_cloud_services/blob/98ad550b1ad29d97e566c43e21ad19edaee6d38d/llama_cloud_services/parse/base.py#L49)作为参考。


  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1387147906045448202)** (7 messages): 

> `gpt4all.io official?, GPT4All Qt requirement issues, 1.58B 2B4T model from Microsoft` 


- **GPT4All 网站存在 Bug**：一位用户询问 * **gpt4all.io** 是官方的吗？* 另一位用户回复了指向官方网站 [nomic.ai/gpt4all](https://www.nomic.ai/gpt4all) 的链接，但反映 *该页面存在 Bug，占用了我 60% 的内置 GPU*。
   - 该用户还指向了他们自己的开源项目 [versoindustries.github.io/HighNoonLLM/](https://versoindustries.github.io/HighNoonLLM/) 并询问 *我的项目是否有机会与你们的项目交叉？很想和你们聊聊潜在的合作。*
- **GPT4All 需要 Qt 升级**：一位用户报告称，文档记录的 **Qt 要求是 6.5+**，但 **CMakeLists.txt 要求 6.7**，而 C++ 代码中使用的某个特性仅在 **6.8** 中可用。
   - 该用户还表示，它无法找到自己的 **Qt 模块**，因为它不符合 **6.8** 中更严格/新的注册方式，而是继续按照 [Qt 文档](https://doc.qt.io/qt-6/qml-singleton.html) 使用已弃用的命令式单例注册。
- **GPT4All 已过时，建议使用 LM-Studio**：一位用户询问关于在 **GPT4All** 中运行 **Microsoft 的 1.58B 2B4T 模型** 的问题。
   - 另一位用户建议改用 **LM-Studio**，并指出 *GPT4All 已经跟不上更新了*。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1387385048277319852)** (5 messages): 

> `GenAI vs Traditional AI, Building a Tetris Bot` 


- **GenAI 抢占了 AI 的风头**：成员们讨论了 **Generative AI**（生成式 AI）已成为焦点，以至于 *“非 GenAI”* 现在成了一个分类。
   - 一位成员指出，AI 是一个完整的领域，所以说 *“非 GenAI”* 就像把整个医学领域命名为 *“非心脏病学”* 一样。
- **AI 工程师寻求构建俄罗斯方块机器人的建议**：一位成员正尝试构建一个 **Tetris bot**（俄罗斯方块机器人），能够实时检测棋盘和下落的方块，并使用 AI 进行游戏。
   - 他们以前没有做过此类项目，正在寻求如何开始的建议。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

dizzy7948: 好的，会做的，希望能抽时间做些贡献。
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1387396205587206175)** (2 messages): 

> `Stas tweet` 


- **Stas 追溯性推文**：这里分享了一条来自 Stas 的推文 [链接](https://x.com/stasbekman/status/1937563125659893900?s=46&t=b1X88nwMsmZgHkmMFkiG3g)，尽管用户给出的名字是 *Stassssss*。
   - 关于这条推文没有分享更多细节。
- **指向旧推文的链接**：一位用户链接到了一条旧推文。
   - 未提供进一步信息。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1387156751630991561)** (2 messages): 

> `` 


- **未讨论任何主题**：提供的文本中未发现讨论主题。
   - 请提供相关的讨论文本以进行总结。
- **未提供链接**：提供的文本中未讨论任何链接或 URL。
   - 包含相关资源链接的总结将更具信息量。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1387312674248986696)** (1 messages): 

> `简介、SEO 专业知识、个人兴趣、联系信息` 


- **Aoun Linkbuilder 自我介绍**：Aoun Linkbuilder 介绍了自己，他拥有 Government College University 的 **数字受众理学学士学位 (Bachelor of Science degree in Digital Audiences)**，专注于 **SEO 和 Digital Marketing**。
   - Aoun 表示，他在 Digital Marketing 领域的旅程源于一种*致力于赋能企业和企业家在网络领域蓬勃发展的热情*。
- **SEO 专业知识亮点**：Aoun 描述了自己在 **on-page 和 off-page SEO、local SEO 以及 technical SEO** 方面拥有深厚的基础。
   - 他表示，他的*目标不仅是提升排名，还要增强可见性、驱动有机流量，并最终为客户带来切实的增长*。
- **Aoun Linkbuilder 分享个人兴趣**：Aoun 分享说，在数字领域之外，你经常会发现他与朋友和他们的爱犬在一起，欣赏 **Taylor Swift 专辑**，或者通过 **艺术与手工** 探索创意。
   - Aoun 邀请他人建立联系，共同探讨如何提升数字存在感并将商业梦想变为现实。
- **Aoun Linkbuilder 分享联系信息**：Aoun 提供了指向其官方账号和服务的各种链接，联系邮箱为 aounlinkbilder@gmail.com，官方网站在[这里](https://aounlinkbuilders.my.canva.site/)，以及 Facebook [个人主页](https://www.facebook.com/profile.php?id=61552225973148)。
   - Aoun 的信息中还包括 Instagram [页面](https://www.instagram.com/aounlinkbilder/)、Linkedin [个人主页](https://www.linkedin.com/in/aoun-linkbuilder-30652b237/)、Twitter [账号](https://twitter.com/aounlinkbilder)、Discord 账号 (**aounlinkbilder-96582**)、GitHub [仓库](https://github.com/AounLinkBuilder-96582)、Reddit [个人主页](https://www.reddit.com/user/Awkward-Regret5585/) 以及 Linktr.ee [页面](https://linktr.ee/aounlinkbuilder96582)。