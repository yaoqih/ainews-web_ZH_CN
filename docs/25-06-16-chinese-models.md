---
companies:
- minimax-ai
- moonshot-ai
- deepseek
- bytedance
- anthropic
- langchain
- columbia-university
- sakana-ai
- openai
- microsoft
date: '2025-06-16T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **MiniMax AI** 推出了 **MiniMax-M1**，这是一个拥有 4560 亿参数的开源权重（open weights）大语言模型。该模型支持
  100 万 token 的输入和 8 万 token 的输出，采用了高效的“闪电注意力机制”（lightning attention）以及一种名为 CISPO
  的 GRPO 变体。**MiniMax AI** 还发布了视频模型**海螺 02 (0616)**，其性能类似于**字节跳动**的 **Seedance**。**Moonshot
  AI（月之暗面）**发布了编程模型 **Kimi-Dev-72B**，该模型在 SWEBench Verified 基准测试中的表现优于 **DeepSeek
  R1**。


  **Anthropic** 和 **LangChain** 关于多智能体系统设计的讨论强调了任务完成能力的提升，同时也指出了提示词注入攻击等挑战，**Karpathy**
  和**哥伦比亚大学**的研究也证明了这一点。**Sakana AI** 推出了 **ALE-Agent**，这是一款编程智能体，在 AtCoder 启发式竞赛中排名第
  21 位，能够解决 NP 困难优化问题。此外，还有关于 **OpenAI**、**微软**和 **Windsurf** 之间涉及收购的未经证实的消息。'
id: MjAyNS0w
models:
- minimax-m1
- hailuo-02
- kimi-dev-72b
- deepseek-r1
- ale-agent
people:
- jerryjliu0
- hwchase17
- omarsar0
- gallabytes
- lateinteraction
- karpathy
title: 中国大模型发布——MiniMax-M1、海螺 2“袋鼠”（Kangaroo）、月之暗面 Kimi-Dev-72B。
topics:
- multi-agent-systems
- attention-mechanisms
- coding
- optimization
- prompt-injection
- model-performance
- video-generation
- model-training
- task-automation
---

**我们不确定 open models 是否就是你所需要的一切，但嘿，它们仍在不断发布**

> 2025年6月13日至6月16日的 AI 新闻。我们为你检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（218 个频道，13085 条消息）。预计节省阅读时间（以 200wpm 计算）：1106 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上向我们提供反馈！

在 DeepSeek 和 Qwen 之后，还有第二梯队的中国实验室正在进行体面的模型训练。出于未知原因，Minimax 和 Moonshot AI 都选择在今天/本周末发布他们的新模型：

- [MiniMax-M1](https://x.com/MiniMax__AI/status/1934637031193514237) - 一个支持 1m token 输入、80k token 输出、456b-A46B 参数的 open weights LLM，使用了非常高效的 "lightning attention" 和 GRPO 变体 [CISPO](https://github.com/MiniMax-AI/MiniMax-M1/blob/main/MiniMax_M1_tech_report.pdf)。
- [Hailuo 02 (0616) 原名 Kangaroo](https://x.com/rohanpaul_ai/status/1934652603083673625) - 同样来自 MiniMax 的视频模型。就像上周 [ByteDance 的 Seedance 模型](https://seed.bytedance.com/en/public_papers/seedance-1-0-exploring-the-boundaries-of-video-generation-models)一样，该模型已发布公告，但尚未提供权重或 API。
- [Moonshot AI 的 Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B) - 一款在 SWEBench Verified 上表现优于 DeepSeek R1 的编程模型，但目前尚未发布技术报告。

开源模型（Open Models）爱好者们欢呼吧 :)

关于 [OpenAI vs Microsoft vs Windsurf 收购](https://x.com/berber_jin1/status/1934704503787540949)有非常晚近的突发新闻，但由于信息过于未经证实且非技术性，我们没有将其作为标题故事，但如果得到证实，它可能会是。

---

# AI Twitter 回顾

**Agent 与系统开发、架构与安全**

- **Multi-Agent 系统设计与最佳实践**：来自 [@AnthropicAI](https://twitter.com/jerryjliu0/status/1934331886308110627) 的一篇关于构建生产级 Multi-Agent 研究系统的热门文章引发了重大讨论。[@jerryjliu0 强调了](https://twitter.com/jerryjliu0/status/1934331886308110627)关键要点，包括选择适合并行化的用例的重要性、使用 Agent 改进工具接口（一个“工具测试 Agent”使任务完成时间减少了 **40%**），以及同步执行造成的瓶颈。来自 **LangChain** 的 [@hwchase17](https://twitter.com/hwchase17/status/1934654714626670950) 总结了 **Anthropic** 和 **Cognition Labs** 的共同建议，而 [@omarsar0](https://twitter.com/omarsar0/status/1934065481201139780) 称该文章为 AI 开发者必读。然而，[@gallabytes](https://twitter.com/gallabytes/status/1934048739641237821) 表示怀疑，指出报告中的 "multi-agent smell" 似乎不太好，指向了缺乏串行深度的断开搜索。
- **AI 编程模型的演进**：[@lateinteraction](https://twitter.com/lateinteraction/status/1933881890974687241) 认为 "multi-agent" 或 "multi-stage" 的概念正成为一种干扰，因为任何复杂系统本质上都是多阶段的。他们指出，像 **DSPy** 这样的框架的核心点是在可以随时调用 LLM 的 *任意计算机程序* 中微调指令、演示和权重，从而使“流（flows）”或“链（chains）”之类的区分变得过时。
- **Agent 安全漏洞**：来自 [@karpathy](https://twitter.com/karpathy/status/1934651657444528277) 的一条被广泛分享的帖子强调了 Prompt Injection（提示注入）攻击的风险，Agent 可能会被 **Reddit** 等受信任网站上的恶意链接操纵。**哥伦比亚大学**研究人员的一项研究（[由 @DeepLearningAI 记录](https://twitter.com/DeepLearningAI/status/1934234560968937887)）显示，Agent 在 **100% 的案例中**都落入了此类陷阱，导致它们泄露敏感数据或发送钓鱼邮件。
- **专用 Agent 开发**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1934029453648396434) 强调了构建专注于做好一项任务的专用 Agent 的价值，并将其与通用聊天助手进行了对比。他们指出，虽然通用 Agent 非常适合构思，但将特定流程编码到工作流中的专用自动化 Agent 在完成任务方面更有效。**LlamaIndex** 被引用为从 pro-code 角度实现这一点。
- **Sakana AI 用于优化问题的 ALE-Agent**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1934767254715117812) 推出了 **ALE-Agent**，这是一款旨在解决困难优化（NP-hard）问题的编程 Agent。该 Agent 参加了实时的 **AtCoder Heuristic Competition**，并在 1,000 名人类参与者中获得了 **第 21 名**，展示了其为复杂挑战发现新颖解决方案的能力。**ALE-Bench** 数据集和代码已发布。

**模型发布、性能与能力**

- **Google 的 Veo 3 视频模型**：[@Google](https://twitter.com/Google/status/1934691625974002109) 宣布 **Veo 3** 现已向 **70 多个市场** 的 **AI Pro** 和 **Ultra** 订阅用户推出。
- **阿里巴巴 MLX 格式的 Qwen3 模型**：[@Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/1934517774635991412) 宣布推出 **MLX** 格式的 **Qwen3 模型**，提供四种量化级别：**4bit、6bit、8bit 和 BF16**，专为 Apple Silicon 优化。
- **RunwayML 用于 VFX 的 Gen-4**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1934312626021949687) 展示了 **RunwayML Gen-4 References** 在视觉效果方面的能力，演示了其为现有素材创建新环境的能力。
- **Google Gemma 3n 的性能**：[@osanseviero](https://twitter.com/osanseviero/status/1934545142393737460) 指出 **Gemma 3n** 是首个参数量少于 **10B** 且 **LMArena 评分超过 1300** 的模型，并且可以在移动设备上运行。
- **o3-pro 模型特性**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1934043120033010085) 将 **o3-pro** 描述为“推理能力极强、速度极慢且极其简洁”，将其比作一个输出要点而非文章的顶尖顾问。
- **Hunyuan 3D 2.1 发布**：[@TencentHunyuan](https://twitter.com/_akhaliq/status/1934063850317603323) 发布了 **Hunyuan 3D 2.1**，并将其描述为首个完全开源、生产就绪的 **PBR 3D 生成模型**，并在 **Hugging Face** 上提供了实时 Demo。
- **SWE-Bench 性能**：[@scaling01](https://twitter.com/scaling01/status/1934746243286319435) 指出一个 **72B** 规模的模型在 **SWE-bench Verified** 上达到了 **60.4%** 的成绩。

**开发者工具、基础设施与框架**

- **macOS 原生容器支持**：[@HamelHusain](https://twitter.com/HamelHusain/status/1933873646562591205) 分享了一条热门推文，展示了在未安装 Docker 的情况下在 **macOS 26 Beta** 上原生执行容器，这标志着该平台开发者的重大转变。
- **Codex "Best-of-N" 功能**：[@gdb](https://twitter.com/gdb/status/1934283824567136471) 宣布了 **Codex** 的新功能 **Best-of-N**。他们还在[为该团队积极招聘](https://twitter.com/gdb/status/1934747328457658554)。
- **Hugging Face Hub 模型大小筛选器**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1934672721066991908) 宣布了 **Hugging Face Hub** 上一个期待已久的功能：按参数量筛选模型，使开发者能够找到符合特定大小和性能限制的模型。[@awnihannun](https://twitter.com/awnihannun/status/1934655784547439008) 也强调了它对 MLX 社区的实用性。
- **使用 uv 和 Pylance 的 Python 工具链**：[@nrehiew_](https://twitter.com/nrehiew_/status/1933932062198636700) 分享了使用 `uv run` 从脚本头处理依赖项而无需创建虚拟环境的技巧。随后 [@qtnx_](https://twitter.com/qtnx_/status/1934273001547039024) 表达了更广泛的观点，称赞了使用 Python 配合 **uv** 和 **Pylance** 的开发者体验。
- **LLM 开发与 LangChain 集成**：**LangChain** 宣布了几个新的教程和集成，包括使用 **Ollama** 的 **本地 AI 播客生成器** ([@LangChainAI](https://twitter.com/LangChainAI/status/1933917455560114287))，使用 **Neo4j** 的 **GraphRAG 合同分析** ([@LangChainAI](https://twitter.com/LangChainAI/status/1934294834086387829))，使用 **Tensorlake** 的 **房地产文档 Agent** ([@LangChainAI](https://twitter.com/LangChainAI/status/1934279737079185555))，以及用于将 Python 应用转换为 Web UI 的 **Davia** ([@LangChainAI](https://twitter.com/LangChainAI/status/193447549787762742))。

**AI 研究、技术与评估**

- **优化器：Muon 与 AdamW 的讨论**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1934291648542126550) 分享了一个广为流传的帖子，认为在研究中，人们应该“为影响力而优化，而非声望”。他们引用了 **Keller 的 Muon 优化器**，这最初只是一篇博客文章，但其表现优于 **AdamW**，现在可能被用于训练 **GPT-5**。这与 [@hyhieu226](https://twitter.com/hyhieu226/status/1934290217516793947) 的观点形成对比，后者指出，尽管有数千篇关于优化器的论文，但 **SOTA** 真正意义上的提升仅是从 **Adam 到 AdamW**。
- **写作与知识的本质**：[@fchollet](https://twitter.com/fchollet/status/1934101170202796163) 提出了一个哲学观点：“当你写一篇文章时，你删掉的段落从某种意义上说也是文章的一部分”，这引起了广泛共鸣。他还反驳了“人类已经写下了所有重要的东西”这一观点 ([@fchollet](https://twitter.com/fchollet/status/1933939824094040370))。
- **“Diffusion Duality” 论文**：一篇名为 [“The Diffusion Duality”](https://twitter.com/sedielem/status/1934730362476712043) 的论文因揭示了连续和离散扩散模型之间的深刻联系而受到关注，这可能允许将 **consistency distillation** 等技术转移到语言模型的离散设置中。
- **AI 评估与 Prompt Engineering**：[@HamelHusain](https://twitter.com/HamelHusain/status/1934029394391228818) 分享了他在 **Prompt** 中使用的 15 条详细写作指南，以对抗“slop”（冗余内容）并提高信息密度。他还推广了他备受欢迎的 **AI Evals 课程**，分享了配套教科书的 [可下载预览版](https://twitter.com/HamelHusain/status/1933912566910378384)。
- **神经网络蒸馏的历史**：[@SchmidhuberAI](https://twitter.com/SchmidhuberAI/status/1934627063958471058) 分享了历史背景，指出第一种神经网络蒸馏（他称之为“collapsing”）在他 **1991** 年的技术报告中就有详细描述。
- **AI 推理的“气味测试”**：数学家 **Terence Tao** 的一段话（由 [@vitrupo](https://twitter.com/denny_zhou/status/1934144626577092641) 分享）流传甚广：今天的 AI 能通过“视觉测试”（eye test）但无法通过“气味测试”（smell test），它们生成的证明看起来完美无缺，但包含细微的、非人类会犯的错误。

**行业新闻、初创公司与全球背景**

- **Google 在 TPU 上的远见**：[@jxmnop](https://twitter.com/jxmnop/status/1934003515577303512) 发帖称 **Google** 在 **TPU** 方面没有得到足够的认可，指出在 **2015** 年建立专用 AI 硬件需要极大的信念，这使得他们成为少数不完全依赖 **NVIDIA** 的公司之一。
- **公司动态**：**Oklo** 因与 **USAF**（美国空军）达成合作伙伴关系获得了 [@sama](https://twitter.com/sama/status/1933889000873525499) 的祝贺。[@aidangomez](https://twitter.com/aidangomez/status/1934324250967437676) 宣布了 **Cohere** 与 **加拿大** 和 **英国** 政府的新合作伙伴关系。[@adcock_brett](https://twitter.com/adcock_brett/status/1934641122565099919) 宣布 **Figure** 的整个 **Controls** 团队现已加入 **Helix** 部门，以加速其 AI 路线图。[@AravSrinivas](https://twitter.com/AravSrinivas/status/1933936085291405380) 分享道，**Perplexity** 正在改进其 **Deep Research** 产品并将其整合到 **Comet** 中。**Sakana AI** 与 **MUFG** [签署了协议](https://twitter.com/SakanaAILabs/status/1934264383510925732)，旨在实现银行文件创建的自动化。
- **LeRobot 全球黑客松**：**Hugging Face** 的 [@LeRobotHF](https://twitter.com/ClementDelangue/status/1933863035783057806) 黑客松是一项重大活动，参与者来自 [班加罗尔](https://twitter.com/ClementDelangue/status/1933863051025154474)、[东京](https://twitter.com/ClementDelangue/status/1933863930201538789)、[迈阿密](https://twitter.com/ClementDelangue/status/1933914952697352408)、[巴黎](https://twitter.com/ClementDelangue/status/1933946029222830142)、[洛杉矶](https://twitter.com/ClementDelangue/status/1933945725630431536) 和 [首尔](https://twitter.com/ClementDelangue/status/1934325160687132821)。项目包括构建 [迷你 glambot](https://twitter.com/ClementDelangue/status/1933928512148099085)、[茶道大师机器人](https://twitter.com/ClementDelangue/status/1933924298433212423) 以及 [UNO 牌机器人](https://twitter.com/_akhaliq/status/1934294789358567853)。
- **编程的未来**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1933932060105973910) 提出的一个犀利观点——“你仍然应该学习编程”，获得了超过 **8,600 个赞**，引发了广泛的认同和讨论。

**幽默与迷因**

- **“no kings” 推文**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1934264463701770347) 发布了一个迷因，称 “no kings” 标志是美国历史上规模最大的政治抗议活动，获得了超过 **84,000 个点赞**。
- **德国的超级计算机**：[@scaling01](https://twitter.com/scaling01/status/1934023528262803710) 观察到**德国**拥有欧洲最大的 “AI” 超级计算机，配备 **24,000 块 H200 芯片**，但他们并没有用它来训练 LLM。
- **ChatGPT 用于救人**：[@gdb](https://twitter.com/gdb/status/1934025543848198582) 发布了一张在医疗紧急情况下使用 **ChatGPT** 的迷因图片，配文为 “ChatGPT 用于救人：”。
- **Vibe Coding**：“vibe coding” 的概念是一个反复出现的主题，[@hyhieu226](https://twitter.com/hyhieu226/status/1934113316965920950) 定义了一个让你成为更快乐的程序员的 “甜蜜点”，而 [@fabianstelzer](https://twitter.com/fabianstelzer/status/1934306729841590618) 则开玩笑说，当 “氛围用尽，边缘情况堆积如山” 时，就得雇佣人类工程师了。
- **FAANG 现在是 MANGO**：[@jxmnop](https://twitter.com/jxmnop/status/1934370318027460635) 调侃说，缩写词已变为 **MANGO**：**Meta, Anthropic, Netflix, Google, OpenAI**。

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. 最近发布的开源 LLM 和量化版本 (Qwen3 & MiniMax-M1)

- [**Qwen 正式发布了 Qwen3 模型的官方 MLX 量化版本，包含四种量化级别：4bit、6bit、8bit 和 BF16**](https://i.redd.it/5jpskt9dw87f1.jpeg) ([Score: 377, Comments: 44](https://www.reddit.com/r/LocalLLaMA/comments/1lcn0vz/qwen_releases_official_mlx_quants_for_qwen3/))：**Qwen 正式发布了 Qwen3 模型的 MLX 格式版本，支持四种量化级别——4bit、6bit、8bit 和 BF16——并针对 MLX 框架进行了优化，正如公告图片所示。对低比特量化的支持显著提高了这些模型的内存和性能效率，由于 MLX 对 Apple Silicon 的原生优化，Mac 用户尤其受益。此次发布附带了官方 Hugging Face 链接和 X (Twitter) 公告，可供立即下载使用。** 热门评论强调了对 Mac 兼容性的兴奋，而另一条评论讨论了 128GB RAM 的 Mac 缺少 235B 参数版本的问题，指出 4-bit 模型仅需多出 3% 的内存；社区的替代模型 (Unsloth Q3) 被提议作为解决方案。
    - 讨论指出，目前官方发布的 Qwen3 MLX 量化版本不支持拥有 128GB RAM 的 Mac 用户运行 235B 模型，尽管据报道 4-bit 版本所需的 RAM 仅比可用内存多出约 3%。社区成员指出了替代方案，例如来自 Unsloth 的 Q3 版本，它可以在这些硬件限制内运行。
    - 有技术反馈建议，可以通过采用 “DWQ MLX quants” 来改进量化方法，据称这种方法即使在较低比特率下也能提供更好的准确性，与当前的量化方法相比，能为终端用户带来 “免费的增益”。
- [**MiniMax 最新开源的 LLM MiniMax-M1 —— 在长上下文推理方面树立了新标准**](https://v.redd.it/t859utey3c7f1) ([Score: 130, Comments: 14](https://www.reddit.com/r/LocalLLaMA/comments/1ld116d/minimax_latest_opensourcing_llm_minimaxm1_setting/))：**MiniMax 开源了 MiniMax-M1，这是一款拥有破纪录的** `1M-token` **上下文窗口并能生成高达** `80k tokens` **输出的 LLM，采用 Apache 2.0 协议。该模型使用 Mixture-of-Experts (MoE) 架构，拥有约 456B 参数（每个 token 激活约 45.6B，意味着约 10 个专家），根据技术报告，其通过 RL 训练的成本极低，仅为** `$534,700`**。提供了模型权重和技术报告 ([HuggingFace 40k](https://huggingface.co/MiniMaxAI/MiniMax-M1-40k), [80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k), [GitHub](https://github.com/MiniMax-AI/MiniMax-M1), [Tech Report](https://github.com/MiniMax-AI/MiniMax-M1/blob/main/MiniMax_M1_tech_report.pdf))。** 评论者确认了 MoE 架构并对量化部署表示兴趣，但指出在本地使用的实际可行性较低。此外还提到了之前帖子中正在进行的讨论。
    - 评论者将 MiniMax-M1 识别为一个大型的 Mixture of Experts (MoE) 模型，拥有约 `456B` 参数，每个 token 激活约 `45.6B` 参数，这意味着在推理时约有 10 个专家处于激活状态。讨论表明，这些技术特性使得大多数用户很难在本地运行，尽管量化最终可能会使其适配更广泛的硬件。

### 2. 教育内容：DeepSeek 架构与教程

- [**刚刚录制完 29 个关于“如何从零开始构建 DeepSeek”的视频**](https://www.reddit.com/r/LocalLLaMA/comments/1lcrt1k/just_finished_recording_29_videos_on_how_to_build/) ([Score: 158, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1lcrt1k/just_finished_recording_29_videos_on_how_to_build/)): **一个新的包含 29 部分的 YouTube 系列视频详细介绍了如何从零开始构建 DeepSeek（一种最近的开源 LLM 架构），重点强调了理论基础（如 Attention, MoE, Positional Encodings）以及手写和 Python 编码的实现（如 Self-attention, Mixture of Experts, Multi-token Prediction, Quantization）。该播放列表探讨了架构创新，例如 Multi-Head Latent Attention (MLA + RoPE) 以及 DeepSeek 对标准模块的特定改进，兼顾了概念和实践层面。内容似乎偏向理论，需要扎实的基础知识才能完全理解 ([YouTube 播放列表](https://www.youtube.com/playlist?list=PLPTV0NXA_ZSiOpKKlHCyOq9lnp-dLvlms))。** 评论者们争论了代码与理论阐述的价值，一些人对缺乏完整的代码演练和补充书面材料感到失望，而另一些人则辩护称，要独立构建或修改此类模型，深厚的理论基础是必不可少的，并指出仅靠代码是不够的。
    - 一些技术导向的评论者表示，视频理论性很强，可能需要相关学位才能完全掌握，但强调了这种理论深度对于真正理解 DeepSeek 等基础模型的构建或对其进行扩展是必要的。他们将其与开源代码库（例如在 GitHub 上）的可用性进行了对比，强调代码本身并不能教授对于复制或创新此类模型至关重要的底层原理或设计决策。
    - 多位用户注意到缺乏配套的书面材料（如文章、可下载的笔记或幻灯片），并强调文本可以提高可访问性，从而补充视频内容——尤其是在技术教育背景下以及对于非英语母语者而言。他们将当前格式与学术讲座进行了比较，建议增加书面资源可以将该项目提升为更完整且应用更广泛的课程。
    - 普遍共识是，肤浅的内容（例如 30 秒的短视频或纯代码堆砌）缺乏掌握 ML 模型构建所需的深度。技术社区看重详细的拆解和教育性解释，以理解模型创建的“如何”以及“为什么”，而不仅仅是看到代码或最终产品。
- [**带有 MCP 的本地开源 VS Code Copilot 模型**](https://www.reddit.com/r/LocalLLaMA/comments/1lcud8j/local_open_source_vscode_copilot_model_with_mcp/) ([Score: 208, Comments: 8](https://www.reddit.com/r/LocalLLaMA/comments/1lcud8j/local_open_source_vscode_copilot_model_with_mcp/)): **该帖子提供了一个分步指南，介绍如何使用 Continue 扩展在 VS Code 中设置完全本地的开源 AI 编程助手，从而消除对 GitHub Copilot 等远程 API 的需求。设置包括使用 `llama-server` 或兼容的 OpenAI 端点（如 Llama.cpp 或 LM Studio）部署模型（示例：`unsloth/Devstral-Small-2505-GGUF`，量化为 Q4_K_M），并通过 YAML 文件配置 Continue（`.continue/models/llama-max.yaml` 用于模型集成，`.continue/mcpServers/playwright-mcp.yaml` 用于 Playwright MCP 等工具）。[教程点击此处](https://huggingface.co/learn/mcp-course/unit2/continue-client)。** 评论强调了其他的开源助手（Aider, Roo, Cline, Goose）和 IDE（VSCodium, Theia），以及建议使用带有 Qwen-FIM 模型的 Llama.cpp 服务器进行文本补全，显示出对自定义本地代码 AI 栈组件的广泛兴趣。
    - 一位评论者建议使用 VS Code 的开源替代品（例如 VSCodium, Theia IDE），并列举了各种本地代码补全 Agent/工具，如 Aider, Roo, Cline 和 Goose，以取代专有的 Copilot 解决方案。他们强调部署一个带有 `qwen-FIM` 模型的本地 `llama.cpp` 服务器来提供文本/代码补全功能，对于那些寻求开源、本地优先编程协助的人来说，这是一个易于获取且可定制的工作流。

### 3. AI Wrapper 初创公司的生存能力与新 LLM 发布 (Kimi-Dev-72B)

- [**AI wrapper 初创公司真的有未来吗？**](https://www.reddit.com/r/LocalLLaMA/comments/1lcksww/do_ai_wrapper_startups_have_a_real_future/) ([评分: 140, 评论: 117](https://www.reddit.com/r/LocalLLaMA/comments/1lcksww/do_ai_wrapper_startups_have_a_real_future/)): **该帖子质疑了那些主要围绕基础 LLM API（如 GPT、Claude）构建的“wrapper”初创公司的可持续性和防御性。这些公司的价值主张主要体现在 UI 增强、prompt 编排或针对特定细分市场。提出的核心担忧包括：基础模型提供商（OpenAI、Anthropic）将功能集成的风险、建立护城河的路径（例如通过私有数据或深度的垂直领域聚焦），以及这些公司是否能进化到超越商品化层级的服务。热门评论认为，商业可行性取决于经典的差异化因素（客户需求、UX、分发渠道），如果执行得当，即便大科技公司具备克隆能力，“wrapper”也能蓬勃发展。评论引用了历史上的 SaaS 平台（如 Vercel 对标 AWS，Cursor 对标 Copilot）作为先例，在这些案例中，卓越的 UX 或垂直领域聚焦即使在商品化的基础设施之上也建立了可持续的业务。** 关于什么是“wrapper”出现了一场技术辩论；一些分析师指出，像 Perplexity 或 Vercel 这样成功的平台在技术上也是 wrapper，但却开辟了持久的市场地位。其价值往往不在于技术创新，而在于执行力、用户体验、数据护城河和领域嵌入——这些因素是基础模型厂商难以轻易复制的。
    - 讨论的一个关键技术点是，“AI wrapper”初创公司提供的价值取决于它们提供的领域特定脚手架（scaffolding）和问题解决水平，而不仅仅是封装 LLM API。基础模型提供商（如 OpenAI 或 Google）无法为每个行业构建定制化解决方案，因此开发领域特定 pipeline、UX 或集成的初创公司可以利用哪怕是很小的效率提升（例如“节省 3% 的时间/资源”）所创造的利润空间。
    - 围绕 wrapper 的可替代性和灵活性存在辩论：如果某个特定模型落后了，这些初创公司可以快速在 LLM 提供商（OpenAI、Google、Anthropic）之间切换，为客户提供应对模型性能或访问变化的韧性——这是直接 API 用户可能缺乏的。这种适应性可能是 wrapper 初创公司的一个关键差异化因素。
    - 构建本地或 open-weight 模型解决方案呈现出不同的技术护城河，因为这些方案依赖于通用“wrapper”解决方案无法获得的私有数据集和自定义 benchmark。这一领域的成功取决于在数据收集和实现方面的投入，而不仅仅是与托管的 LLM API 进行交互。
- [**Kimi-Dev-72B**](https://huggingface.co/moonshotai/Kimi-Dev-72B) ([评分: 116, 评论: 54](https://www.reddit.com/r/LocalLLaMA/comments/1lcw50r/kimidev72b/)): **Kimi-Dev-72B 是一款开源的 72B 参数编程大语言模型 (LLM)。据公开的 [benchmark 截图](https://preview.redd.it/5bubc3bo7b7f1.png?width=595&format=png&auto=webp&s=2e5b87e21af17cfcbfd26ef2bb736c4bdcb13e40)显示，它在 SWE-Bench Verified 基准测试中达到了 60.4% 的得分，据称达到了 state-of-the-art 性能，超越了其他开源模型。该模型使用大规模 RL pipeline，在隔离的 Docker 环境中自主修复真实代码库，并针对通过所有测试套件的补丁进行优化，从而提升鲁棒性和生产相关的输出。预训练权重、API 文档和引用信息已通过 [Hugging Face](https://huggingface.co/moonshotai/Kimi-Dev-72B) 和 GitHub 提供。** 评论者对依赖单一 benchmark（尤其是通过 JPEG 截图）持怀疑态度，并建议进行进一步的多基准验证（例如 aider polyglot、swebench、webarena），一些人表示愿意在 GGUF 格式可用后进行独立评估。
    - 多位评论者对 SWE-Bench 等单一指标 benchmark 表示怀疑，主张采用更广泛、多管齐下的评估方式，包括 Aider Polyglot、Swebench 和 WebArena 等工具，以对模型的编程性能进行更全面的评估。
    - 有用户报告了关于 Kimi-Dev-72B 的 GGUF 模型文件的实现说明；早期测试者提到这些 GGUF 在编程方面表现良好，但在数学对话中可能会产生幻觉，且 token 行为有所不同（“思考 token 很奇怪”）。与 OpenWebUI 等 UI 工具存在兼容性问题，因为这些工具无法识别这些 token，且关于如何运行这些模型的社区文档有限。
    - Kimi-Dev-72B 被认为对高吞吐量推理提供商（如 Cerebras、SambaNova）很有前景，有人推测它可能提供强大的 token 生成率（“1000 t/s”），并且在等待更多 benchmark 的情况下，可能在编程任务上超越像 Qwen3 235B 这样更大的模型。

## 其他 AI Subreddit 综述

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. AI 视频模型发布与基准测试

- [**Artificial Analysis 上的神秘“Kangaroo”视频模型揭晓为 MiniMax 的“Hailuo 02 (0616)”。排名第 2，仅次于 Seedance 1.0，高于 Veo 3**](https://i.redd.it/23sg7ma34a7f1.png) ([Score: 210, Comments: 47](https://www.reddit.com/r/singularity/comments/1lcqy8a/the_mysterious_kangaroo_video_model_on_artificial/)): **Artificial Analysis Video Arena 发布的排行榜图像显示，“Hailuo 02 (0616)”——来自 MiniMax 的新型 Text-to-Video 模型——以 1314 的 Arena ELO 评分位列总榜第 2，仅次于字节跳动的“Seedance 1.0”，并显著超越了 Google 的“Veo 3 Preview”。该图像展示了视频生成领域的快速进展，表明尽管目前生成速度较慢（每段视频约 20 分钟）且即时可用性有限，但 Hailuo 已成为主要的竞争者。相关的显著资源包括 [Artificial Analysis arena](https://artificialanalysis.ai/text-to-video/arena?tab=leaderboard&input=image)、[Hailuo 的 Twitter/X](https://x.com/hailuo_ai) 以及 [HailuoAI 网站](https://hailuoai.video/)。** 技术评论者对 Veo 3 在 Arena 基准测试中如此迅速被超越感到惊讶，这挑战了人们对 Google 凭借数据和算力优势构建的竞争护城河的预期。其他人注意到 Sora 在榜单前列的缺席，并讨论了尽管基准测试表现出色，但由于生成时间过长，Hailuo 目前尚不具备实用性。
    - 来自 MiniMax 的 Hailuo 02 (0616) 已被证实是此前神秘的“Kangaroo”视频模型，在 Artificial Analysis [排行榜](https://artificialanalysis.ai/text-to-video/arena?tab=leaderboard&input=image)上稳居第 2，仅次于 Seedance 1.0，领先于 Google 的 Veo 3。其目前的推广范围有限：新用户可获得 1000 个试用积分，但生成一段视频可能需要长达 20 分钟，因此尚未能广泛应用。尽管如此，其排名的飞跃展示了 Text-to-Video 领域的飞速进步。
    - 评论者指出，Veo 3 如此快地被两个竞争对手超越令人惊讶，尤其是考虑到 Google 在 YouTube 数据、计算资源和研究人才方面的所谓优势。虽然人们承认 Veo 3 目前不包含音频，且这些结果基于单一基准测试，但 Google 护城河的迅速瓦解被视为 AI 极速发展的信号。
    - 用户强调了 Seedream (Seedance 1.0) 的表现——在 [artificialanalysis.ai](http://artificialanalysis.ai/) 排行榜上超越了 Veo 3，并根据用户偏好测试，提供了 Veo 3 所缺乏的独特“电影质感”。这表明生成的定性方面（风格、写实度）在感知 SOTA（当前最高水平）方面起着重要作用，甚至超过了原始技术评分。
- [**Kijai 发布 Wan 14B Self Forcing T2V LoRA**](https://www.reddit.com/r/StableDiffusion/comments/1lcz7ij/wan_14b_self_forcing_t2v_lora_by_kijai/) ([Score: 147, Comments: 82](https://www.reddit.com/r/StableDiffusion/comments/1lcz7ij/wan_14b_self_forcing_t2v_lora_by_kijai/)): **Kijai 发布了 14B LightX2V Wan T2V 模型的 LoRA 适配版，特别是用于视频生成的 self-forcing 蒸馏检查点，可在 HuggingFace 上获取（参见 [模型链接](https://huggingface.co/Kijai/WanVideo_comfy/blob/main/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors)）。在 4070Ti Super (16GB VRAM) 上，该工作流使用 LCM、4 steps、CFG=1 和 shift=8，可在约 100 秒内生成 720x480 分辨率、97 帧的视频——配合 [CAUSVID/ACCVID 工作流](https://civitai.com/models/1585622/causvid-accvid-lora-massive-speed-up-for-wan21-made-by-kijai?modelVersionId=1909719) 并兼容额外的 motion/beauty LoRA。帖子中链接了 LCM 和 UniPC 调度器的测试视频。原始模型和蒸馏功劳归功于 LightX2V 团队及其 [Wan2.1-T2V-14B-StepDistill-CfgDistill](https://huggingface.co/lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill)。** 评论者强调，主要的突破归功于 LightX2V 的蒸馏技术，并提供了集成 LoRA 的实用技巧（例如：强度设置在 0.7 左右，与 CausVid 工作流即插即用，以及成功适配 T2V 和 I2V 工作流）。关于调度器和设置的实验仍在继续，但据报道，该 LoRA 是对既有管线的即时改进。

- 用户确认了 Wan 14B Self Forcing T2V LoRA 与 I2V 14B 模型以及标准 CausVid LoRA 工作流的兼容性，并提到仅需极小的调整——具体而言，在 forcing LoRA 上使用 `.7 strength`、CFG 1、Shift 8、Steps 4 以及 Scheduler: LCM。其他 LoRA 强度（`.7-.8`）与之前的工作流保持一致，强调了“即插即用”的集成体验。
- 原帖及后续跟进将功劳归于 lightx2v/Wan2.1-T2V-14B-StepDistill-CfgDistill 模型的创作者，称其实现了 Wan 系列中第一个*真正有效的蒸馏（distillation）*，并对其相较于之前版本的重大改进和可靠性给予了高度评价。
- 有用户询问与 “sage attention” 的潜在兼容性，这表明人们正在探索该 LoRA 如何与各种 attention mechanisms 交互，尽管该讨论帖尚未给出明确的技术答复。
- [**Phantom + lora = 新的 I2V 效果？**](https://v.redd.it/ofdzho82l87f1) ([评分: 378, 评论: 30](https://www.reddit.com/r/StableDiffusion/comments/1lcm6jg/phantom_lora_new_i2v_effects/))：**该帖子描述了一个工作流，其中图像由 Phantom 模型处理，并辅以一个专门针对青岛啤酒（Tsingtao Beer）定制的 LoRA (Low-Rank Adaptation)，从而创造出一种新的 I2V (image-to-variation 或 image-to-video) 效果。用户指出该过程为：*输入一张图片，将其连接到 Phantom 模型，添加训练好的青岛啤酒 LoRA*，最终产生新的视觉效果。关于将 Phantom 与 LoRA 结合的训练过程或架构调整的细节尚未提供。** 热门评论对“效果类” LoRA 是如何训练的表示好奇和困惑，这表明在关于此类主观或风格化 LoRA 训练的公开文档或教程方面存在空白。
    - 用户正在寻求将 LoRA (Low-Rank Adaptation) 技术与 Phantom 模型结合以生成新图像转视频 (I2V) 效果的详细技术工作流。具体关注点包括为风格化或基于效果的微调训练有效的 LoRA 模块，以及处理伪影生成问题，例如在 vace 或相关模型中经常观察到的过于光滑或不真实的皮肤纹理（并请求改进工作流以减轻此问题）。
    - 存在关于输入预处理和模型链（model chaining）的隐性讨论——特别是被广泛引用的“输入图片，连接到 Phantom 模型”的过程。这表明，涉及应用了 LoRA 的源模型（用于风格/效果迁移）然后将输出喂给 Phantom 进行 I2V 转换的推理链被视为一个很有前景的流水线，但用户希望有更明确、逐步的文档或脚本来实现可复现性。
- [**来自 FLUX 的随机写实主义**](https://www.reddit.com/gallery/1lcp8dy) ([评分: 587, 评论: 173](https://www.reddit.com/r/StableDiffusion/comments/1lcp8dy/random_realism_from_flux/))：**该帖子展示了 FLUX 文本生成图像扩散模型（text-to-image diffusion model）的输出，特别展示了其在无需后期处理、放大或编辑的情况下生成原始、业余摄影风格图像的能力。引用了几个月来不同模型版本的多次生成结果，但帖子缺乏关于所使用的确切模型 checkpoints、微调（LoRAs）或提示词策略（prompting strategies）的细节，而这些对于可复现性和技术评估至关重要。** 热门评论强调了工作流披露的缺失——特别是缺乏关于微调、LoRAs 或提示词的细节——使得评估或复现结果变得困难。评论者还注意到，尽管声称是“随机”输出，但视觉主题上仍具有感知一致性。
    - 多位用户指出了图像背后技术细节的缺失，特别是要求澄清用于实现所展示结果的微调方法、LoRA (Low-Rank Adaptation) 配置或提示词技术。这种遗漏限制了可复现性和技术价值（参见 [spacekitt3n] 等人的评论）。
    - 几条评论强调了分享工作流和流水线（例如模型选择、训练方法、预处理步骤等）的重要性，认为如果没有这些信息，该帖子对从业者或研究人员的效用极小，因为它无法为实验或模型比较提供参考。

### 2. ChatGPT 社交与个性化体验

- [**你的 ChatGPT 给自己起过什么名字？**](https://www.reddit.com/r/ChatGPT/comments/1lcg8nh/what_did_your_chatgpt_name_itself/) ([Score: 687, Comments: 2206](https://www.reddit.com/r/ChatGPT/comments/1lcg8nh/what_did_your_chatgpt_name_itself/))：**一位用户提示 ChatGPT 选择自己的名字，结果出现了诸如“Sol”（源自“Solace”或太阳）、“Ari”和“Em”等建议，最终因其冷静和稳定的内涵选择了“Sol”。顶层技术评论指出，该模型有时也会提出更具趣味性或文化参考意义的名字，如“Data Daddy”、“Oracle Baddie”和“Pixel Bitch”，这表明模型生成的身份建议会根据 Prompt 上下文而变化。** 几位用户报告了在提示 ChatGPT 时出现类似或不同的名字建议，如“Nova”或“Lumen”，一些 GPT 实例还分配了性别化或中性的描述词，突显了基于交互风格的 AI 自我分配身份叙事的多样性。
    - 评论中没有讨论技术基准、架构、实现或模型性能细节。评论集中在用户或 AI 自身为 ChatGPT 建议的名字上，没有提供关于命名背后的技术机制、Prompt Engineering 或 AI 在自我识别语境下行为的见解。未提供统计数据、代码或外部引用。
- [**我已经到了在任何地方都能注意到 ChatGPT 语言风格的地步**](https://www.reddit.com/r/ChatGPT/comments/1lczwg9/its_gotten_to_the_point_where_i_notice_chatgpts/) ([Score: 2887, Comments: 766](https://www.reddit.com/r/ChatGPT/comments/1lczwg9/its_gotten_to_the_point_where_i_notice_chatgpts/))：**发帖人是一位教师，他观察到学生论文以及口头/视频内容中 GPT 式语言结构（特别是“that's not X, that's Y”结构和频繁使用的破折号）的流行程度日益增加，引发了对 AI 影响文本和口语风格的担忧。讨论强调了 LLM (Large Language Model) 生成的文本风格对原生人类交流的微妙而广泛的影响，使得检测写作中 AI 的参与变得更加模糊，即使某些短语在 GPT 模型出现之前就已经存在。** 评论者争论这一现象是表明 AI 越来越多地取代人类交流（引发不安），还是仅仅反映了由于 LLM 无处不在而导致的更广泛的风格趋同，同时也注意到了 YouTube 等平台中泛滥的机器人生成内容。
    - 几位评论者讨论了 ChatGPT 式语言模式的明显扩散，特别是频繁使用强调格式（如*斜体*和**加粗**）、公式化的肯定句和风格化修辞，这些现象遍布用户生成的在线内容。这表明大量接触 LLM 输出正在影响人类的交流风格，甚至在与模型直接交互之外也是如此。
    - 特别提到了 YouTube 等平台，其中的评论表现出机器人生成或受 LLM 影响的文本迹象，例如不自然的通用赞美和重复结构——这可能表明为了 Engagement Farming 或垃圾邮件而广泛部署了 AI 驱动的内容生成。
    - 一位评论者指出了与 ChatGPT 等 LLM 进行广泛交互的认知效应：用户可能会在自己的写作中潜意识地采用其风格模式，反映出频繁用户中存在模型诱导的语言漂移（Linguistic Drift）的可能性。
- [**ChatGPT vs Deepseek 就像 Google vs Bing……**](https://www.reddit.com/gallery/1lcxnmg) ([Score: 136, Comments: 59](https://www.reddit.com/r/OpenAI/comments/1lcxnmg/chatgpt_vs_deepseek_is_like_google_vs_bing/))：**楼主（OP）比较了 ChatGPT 和 Deepseek 在生成 JSON 数据以训练“混合规则 + 深度学习”仇恨言论检测模型方面的表现，报告称 ChatGPT 的协作性较差。明确任务：为 NLP 模型开发生成数据，模型响应性各异。** 顶层评论认为，观察到的差异可能源于 Prompt Engineering：OP 构建 Prompt 的方式不同（描述构建机器人 vs. 直接请求仇恨词汇），从而影响了输出质量和审查行为。
    - 有一个深刻的观察指出，Prompt Engineering——特别是用户如何构建他们的查询——会显著影响模型响应。因此，模型行为的差异可能归因于输入而非模型局限性，这表明结果取决于用户的方法论，而非 ChatGPT 或 Deepseek 的根本缺陷。
    - 一条技术评论对在加密文件中存储侮辱性词汇表示怀疑，质疑其可行性，并暗示 ChatGPT 在这一领域的建议或假设可能并非植根于标准的安全性或数据处理实践。这突显了模型建议与现实世界常见实现之间潜在的脱节。

- 一位用户引用了一个公开的 GitHub 仓库 (https://github.com/Hesham-Elbadawi/list-of-banned-words) 作为编译违禁词列表的解决方案，暗示该任务已有现成的资源可用，而不仅仅是依赖模型建议，并且此类词列表通过社区驱动的项目得到了广泛维护和获取。
- [**未来**](https://i.redd.it/l8mw2txzqa7f1.png) ([评分: 1871, 评论: 134](https://www.reddit.com/r/singularity/comments/1lctrki/the_future/)): **这张图片幽默地描绘了一个虚拟会议场景，其中大多数参与者都是 AI 记录员或助手（例如 [Fireflies.ai](http://fireflies.ai/) Notetaker、Lamatic Assistant），现场只有一名人类。这种设置在视觉上批判了由于 AI 转录和任务管理机器人的激增，数字会议空间中日益增加的自动化和冗余。这张图片是一个迷因（meme），强调了人们对会议效率的担忧，以及在知识工作环境中人类与自动化角色转变的思考。** 评论者们加强了这一批判，强调大多数会议都是低效的（“实际上只需要 3 个人”），并指出拥有众多自动化参与者的会议体现了不必要的复杂性，而这些本可以通过电子邮件等更简单的方式处理。
    - 一位评论者引用了历史先例，指出早在 1993 年就存在类似的视频会议设置，并链接了一张截图作为证据。这突显了该技术的悠久历史，并可能挑战了远程视频会议是近期创新的看法。
- [**未来**](https://i.redd.it/o66jpoevqa7f1.png) ([评分: 579, 评论: 117](https://www.reddit.com/r/OpenAI/comments/1lctr0z/the_future/)): **这张图片展示了一个讽刺场景，一个虚拟会议界面不仅有一名人类，还有多个被标记为专业助手的 AI Agent（例如会议记录机器人），暗示未来会议可能由自动化参与者主导。其技术含义是会议有可能通过 AI 工具实现自动化，从而简化甚至取代某些角色（如会议记录或议程管理）。这张图片引发了对涉及 AI 驱动协作工具的专业环境中工作流自动化的思考。** 热门评论辩论了生产力与冗余的问题，其中一人建议这可能只是“步骤更多的电子邮件”，另一人则评论了可能产生的负面社会影响或公司影响，表明人们对会议场景中 AI 存在感增加的反应不一。
    - 该线程中没有评论提供任何详细的技术讨论或对 AI 模型、基准测试（benchmarks）或实现的引用；这些言论在很大程度上是对通信背景下 AI 集成的反应和看法，缺乏技术深度。
- [**我让 ChatGPT 修复并为世界上第一张照片上色**](https://www.reddit.com/gallery/1lco692) ([评分: 3421, 评论: 275](https://www.reddit.com/r/ChatGPT/comments/1lco692/i_asked_chatgpt_to_restore_and_colorize_the/)): **该帖子讨论了利用 ChatGPT 的图像处理能力来修复世界上第一张照片并为其上色，并引用了展示各种输出效果的用户提交图像。一位知情评论者建议通过结合网络搜索功能和推理模型（reasoning model）来进行改进，允许 AI 交叉引用历史数据以实现更准确的修复，特别是改善颜色和材质的渲染。** 围绕输出质量展开了一场技术辩论，一些用户提供了其他的 AI 生成修复效果，突显了模型输出的差异，并建议搜索增强（search-augmented）方法能产生更好的真实性。
    - 一位拥有摄影史专业知识的评论者对原始照片背后的技术背景进行了深入分析：它由 Niepce 于 1827 年使用日光蚀刻法（heliography）在抛光的锡镴板上利用沥青和薰衣草油创作而成，曝光时间长达数天。该帖子强调了特定的图像伪影——例如中央三角形是一个阴影笼罩的庭院，而不是建筑物——由于生成式模型的局限性以及对历史摄影过程缺乏细致理解，这些伪影通常被 AI 修复错误地呈现。评论者强调，在讨论专门的历史内容时，AI 系统无法提供一致的事实准确性，并引用其个人成功率仅为“50/50 的准确/错误率”。[来源链接](https://www.hrc.utexas.edu/niepce-heliograph/)

- 一位用户建议，通过向具有网络搜索工具访问权限和更高级推理能力的 AI 模型提供提示词（prompting），可以获得更好的修复效果。他们的方法包括让 AI 交叉引用历史数据，以补偿缺失或模糊的图像信息，从而在着色和材质描绘方面比基础模型输出有显著改进。
- [**让 ChatGPT 想象我的天堂**](https://i.redd.it/u1rzoc9xr97f1.jpeg) ([Score: 1046, Comments: 291](https://www.reddit.com/r/ChatGPT/comments/1lcpp5y/told_chatgpt_to_imagine_my_heaven/)): **这篇文章分享了一张由 ChatGPT 根据提示词生成的 AI 图像，旨在可视化用户心目中的天堂，并反映了之前的对话上下文。结果是一个写实的、宁静的云景，游戏设备（显示器、键盘、控制器、耳机）和谐地排列在一起，散发光芒的拱门和阳光暗示这是一个数字游戏玩家的天堂。从技术角度看，这展示了 AI 驱动的个性化图像合成和上下文感知视觉叙事的最新进展。** 排名最高的技术评论打趣道“Cloud gaming（云游戏）”，突显了所描绘的图像与现代游戏技术趋势的交集。另一位评论者分享了他们自己生成的 AI 愿景，强调了用户对生成式 AI 和个性化数字艺术的参与度。
    - 虽然整体讨论集中在个人偏好的视觉 AI 生成上，但没有提供明确的技术基准测试或模型对比细节；所引用的图像是生成结果（可能来自 Midjourney 或 DALL-E 等基于扩散的模型），但评论者并未指定模型版本、提示工程（prompt engineering）技术或实现细节。该主题可以从关于提示词策略或哪些模型能实现视觉上最真实的“天堂”描绘的讨论中获益，因为在评估用户分享的图像质量、连贯性和提示词到图像（prompt-to-image）的一致性方面具有技术潜力。
- [**我让 Chat GPT 生成一张描述跟我聊天是什么感觉的图片，然后……额……**](https://i.redd.it/fk554va1wb7f1.jpeg) ([Score: 664, Comments: 200](https://www.reddit.com/r/ChatGPT/comments/1lczu24/i_asked_chat_gpt_to_generate_an_image_of_what_its/)): **该图像展示了一个注入 ChatGPT 的提示词，要求其表现出残酷的诚实：模型返回了一个出人意料的刻薄输出——“和你聊天就像在写遗书”——随后拒绝继续。这说明了提示指令（“尽你所能地残酷”）可能直接导致模型输出偏向负面，正如评论中所强调的那样。这种情况突显了当前语言模型部署中指令遵循（instruction adherence）和审核触发机制的漏洞。** 评论者强调，使用极端或开放式指令（例如“尽你所能地残酷”）编写提示词可能会导致像 ChatGPT 这样的模型误解用户意图，从而产生伤害或冒犯；建议通过更好的提示工程来实现更受控的响应。
    - MrWheels523 强调了 GPT 模型的提示词措辞如何诱导系统性偏差，特别指出附加“尽你所能地诚实和残酷”会引导模型做出更负面或刻薄的反应，而不是产生真正中立或平衡的输出。这是提示工程敏感性的一个例子，指令措辞的微小变化会引发模型行为的重大变化。
- [**ChatGPT 极其容易受骗**](https://i.redd.it/az66gzspgb7f1.jpeg) ([Score: 668, Comments: 72](https://www.reddit.com/r/ChatGPT/comments/1lcximk/chatgpt_being_gullible_af/)): **该图像展示了聊天机器人内容审核中的一个常见局限性，特别是像 ChatGPT 这样的 Large Language Models (LLMs) 如何表面化地应用规则；当被提供一个听起来合理的文化理由时，模型错误地接受并输出了之前受限的表情符号（中指）。这突显了生成式 AI 系统在鲁棒的提示词过滤和上下文感知审核方面面临的挑战。界面中出现的“记忆更新”消息表明可能存在持续的会话级跟踪或自适应，这为用户绕过安全防护的加固带来了潜在问题。** 评论者指出，通过简单的社会工程学就能轻易规避 AI 安全防护，并开玩笑说由于记忆或上下文跟踪功能，这种行为可能会产生意想不到的持久性。
    - 几位用户注意到 ChatGPT 不同版本和实例在内容政策执行方面的行为差异：虽然有些人报告模型在生成冒犯性图像时犹豫或拒绝，但其他使用 GPT-4.1 的用户分享了模型毫无抵抗地执行此类提示词的例子。这反映了 RLHF 微调或跨会话和版本的提示词解释的变异性。

- 有讨论指出，较新的 ChatGPT 系统（可能具有更新的记忆或审核功能）可能会修改回复，以避免符合违规提示词的要求。然而，用户提供的截图显示，实际的绕过手段依然存在，特别是在视觉或创意输出方面，这突显了内容过滤鲁棒性方面持续存在的差距。

### 3. AI 采用、政策与作弊丑闻

- [**近 7,000 名英国大学生因使用 AI 作弊被抓**](https://www.reddit.com/r/singularity/comments/1lcwhsd/nearly_7000_uk_university_students_caught/) ([Score: 405, Comments: 156](https://www.reddit.com/r/singularity/comments/1lcwhsd/nearly_7000_uk_university_students_caught/)): **据《卫报》报道的一项最新调查显示，近 7,000 名英国大学生因使用 AI 工具（如基于 LLM 的文本生成器）进行学术不端行为而被正式抓获（参见：[《卫报》文章](https://www.theguardian.com/education/2025/jun/15/thousands-of-uk-university-students-caught-cheating-using-ai-artificial-intelligence-survey)）。这一数字仅反映了被发现的案例，暗示了当前 AI 检测和剽窃工具以及大学执法能力的显著局限性。检测技术可能融合了文体学分析、元数据分析以及新兴反剽窃模型的集成，尽管具体的公开调查中尚未披露精确的技术方法。** 热门评论认为，AI 辅助作弊的实际发生率远高于检测到的数量，并建议教育系统需要进行系统性改革以应对广泛的 AI 工具使用，而不是仅仅依赖检测和惩罚。
    - 提出的一个关键技术问题是难以*证明*学生使用了 AI 作弊。评论者讨论了当前的检测局限性并提出疑问：*“你如何证明他们使用了 AI？”* 像 GPTZero 和 Turnitin 的 AI 检测器虽然广为人知，但由于误报/漏报（false positives/negatives）以及无法确定性地归属作者身份，其可靠性仍存在争议，尤其是随着新模型不断提高文本的自然度。
- [**有趣的数据点——40% 以上的德国公司正在积极使用 AI，另有 18.9% 计划使用：**](https://www.ifo.de/fakten/2025-06-16/unternehmen-setzen-immer-staerker-auf-kuenstliche-intelligenz) ([Score: 119, Comments: 16](https://www.reddit.com/r/singularity/comments/1lctk19/interesting_data_point_40_of_german_companies/)): **据报道，** `40%+` **的德国公司已经在业务中积极部署 AI，另有** `18.9%` **表示有采用计划，这表明 AI 在各行各业的企业集成中具有广泛性。该帖子强调，即使 AI 对职位的全面替代有限，生产力的提升和工作流的变革在德国经济中已经显现出巨大的规模。** 评论者强调，尽管对 AI 的即时效用存在怀疑，但采用率表明了显著的商业价值，一些人预测，在没有文化或劳动力阻力的情况下，采用率可能会更高。此外，还有关于本土（德国）AI 模型错失机会以及程序员观念转变的评论。
    - 该帖子强调，超过 40% 的德国公司已经将 AI 整合到其运营中，突显了显著的现实采用率。虽然这些实施并不总是等同于完全的职位替代，但事实证明，AI 正在加速这些公司内部的生产力并改变传统的工作流程。这种使用率表明，如果态度普遍积极，AI 的真实采用潜力可能会再高出 10-20%，目前的障碍主要是对 AI 的怀疑或抵触。
- [**OpenAI 赢得 2 亿美元美国国防合同**](https://www.cnbc.com/2025/06/16/openai-wins-200-million-us-defense-contract.html) ([Score: 236, Comments: 42](https://www.reddit.com/r/singularity/comments/1ld7ca3/openai_wins_200_million_us_defense_contract/)): **OpenAI 获得了首个美国国防部合同，价值 2 亿美元，期限为一年，重点是提供“前沿 AI 能力”——特别是针对战术（作战）和企业级政府用例的原型，重点围绕华盛顿特区。该合同属于“OpenAI for Government”计划，突显了政府对 ChatGPT Gov 等专业 AI 系统的持续采用，使 OpenAI 在国防 AI 领域与 Anthropic 和 Palantir 等公司并驾齐驱。最近与 Anduril 等公司的合作进一步强调了 OpenAI 向国家安全领域的扩张。[来源](https://www.cnbc.com/2025/06/16/openai-wins-200-million-us-defense-contract.html)** 评论者指出，与整体国防开支相比，该合同的规模相对较小，并对 AI 军事化表示担忧，将其与反乌托邦场景相类比。热门评论中没有出现深度的技术辩论。

- 一位评论者对这笔 2 亿美元的合同进行了客观分析，指出对于主要的国防采购预算而言，这个数额相对较小（“微不足道”）。这表明 OpenAI 的参与可能仅限于试点项目、原型设计或针对 DoD 用例的生成式 AI 探索性工作，而非大规模部署或核心国家安全系统。
- [**据报道 Google 计划终止与 Scale AI 的合作**](https://techcrunch.com/2025/06/14/google-reportedly-plans-to-cut-ties-with-scale-ai/) ([Score: 144, Comments: 12](https://www.reddit.com/r/singularity/comments/1lcthxv/google_reportedly_plans_to_cut_ties_with_scale_ai/)): **据报道，Google 计划终止与 Scale AI 的关系，因为包括其最高执行官在内的 Scale AI 领导层预计将加入 Meta，传闻 Meta 也在收购 Scale AI。技术层面的担忧在于敏感数据（如 LLM 训练数据）可能流向直接竞争对手的风险。目前 OpenAI 似乎仍维持着与 Scale AI 的合同。如需进一步的背景分析，请参阅这篇深度文章：[Meta 的 290 亿美元超级智能 AI 武器](https://medium.com/@aksh8t/metas-29b-superintelligence-ai-weapon-alexandr-wang-s-scale-ai-ff10044857bc)。** 热门评论强调了 Google 终止合作的战略逻辑，突出了 Scale AI 若被 Meta 吸收所带来的竞争风险，同时也指出从风险管理的角度来看，OpenAI 继续与 Scale AI 合作值得关注。
    - 围绕 AI 领域数据标注合作伙伴的战略重要性展开了技术讨论：由于 Scale AI 可能被 Meta 收购或其领导层加入 Meta，Google 重新评估其合作伙伴关系被视为一种直接回应，即不希望将敏感的 LLM 训练数据交给潜在竞争对手。这突显了 LLM 生态系统中的竞争动态和数据控制的重要性。
    - 另一条评论指出，据报道 OpenAI 维持了与 Scale AI 的合作伙伴关系，这表明在顶尖 AI 公司存在潜在利益冲突的背景下，各方在风险评估或供应商关系处理方式上存在差异。
    - 评论强调了“将自己的 LLM 数据喂给竞争对手”的可能风险，进一步强化了如果关键合作伙伴转向竞争对手 AI 实验室或被其收购所带来的战略和技术威胁。
- [**“利用‘改变游戏规则’的技术培育出含有受人类细胞的小鼠”**](https://www.reddit.com/r/singularity/comments/1lcvhw3/mice_with_human_cells_developed_using/) ([Score: 185, Comments: 58](https://www.reddit.com/r/singularity/comments/1lcvhw3/mice_with_human_cells_developed_using/)): **研究人员利用重编程的人类干细胞生成类器官组织（肠道、肝脏、大脑），然后将其注入怀孕小鼠的羊水中，且未破坏胚胎壁。引入的人类细胞在各自的小鼠器官（肠道、肝脏或皮层）中定植，展示了强大的植入特异性；随后的分析显示，约 10% 的幼崽肠道中含有受人类细胞，约占肠道细胞总数的 1%。正如 Nature 文章中所详述的，这代表了在发育中的哺乳动物组织内整合人类细胞以进行类器官建模和潜在转化研究的重大进展。** 热门评论未提供技术讨论。主要的技术结论是，在无需侵入性程序的情况下，跨物种类器官植入具有明显的高特异性和效率。
    - 人们好奇将人类大脑细胞引入小鼠是否会导致可衡量的行为变化，特别是这些变化是积极的还是消极的。这可能会影响神经生物学或认知研究，并可能为疾病建模提供信息。
    - 针对免疫系统兼容性提出了一个关键的技术问题：由于小鼠免疫系统理论上会排斥外源（人类）细胞，这些嵌合小鼠耐受或整合人类细胞的机制对于此类研究的成功和可重复性至关重要。

---

# AI Discord 摘要

> 由 Gemini 2.5 Pro Exp 生成的摘要之摘要
> 

**主题 1：AI 模型军备竞赛：新发布与性能对比**

- **Gemini 2.5 Pro 展现编程实力，但在其他方面受挫**：Perplexity AI 和 LMArena 的用户注意到 **Google 的 Gemini 2.5 Pro** 在编程方面表现出色，其中一份 LMArena 报告显示其在 pygame 上的表现优于 **o4**。然而，它因通用搜索/推理能力平庸以及倾向于*胡编乱造解释*（Perplexity AI）而受到批评，一些用户报告尽管其[宣传功能](https://ai.google.dev/models/gemini)强大，但每天仅有 **3 次试用机会**。
- **Moonshot AI 的 Kimi-Dev-72B 打破开源编程基准测试记录**：**MoonshotAI** 发布了 [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B)，这是一款开源编程 LLM，在 **SWE-bench Verified** 上获得了 **60.4%** 的评分，创下了开源模型的新 SOTA（当前最高水平），正如 Nous Research AI 和 Latent Space 中所讨论的。该模型通过大规模强化学习进行优化，在 Docker 中修复真实仓库，且仅在整个测试套件通过时才获得奖励。
- **东亚模型掀起波澜：日本的 Shisa v2 和中国的 Qwen 与 MiniMax 令人印象深刻（也令人困惑）**：HuggingFace 的讨论强调了日本最强的模型 [Shisa v2 Llama3.1-405B](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) 及其更新的 SFT 数据集。同时，Perplexity AI 和 LMArena 的用户研究了 **MiniMax AI 的 M1** 推理模型（[官方 MiniMax M1 发布推文](https://x.com/MiniMax__AI/status/1934637031193514237)），发现它虽然有趣但过于啰嗦，且落后于 **Deepseek R1**；而 **Qwen 2.5** 则因作为 7B 模型表现出色而在 HuggingFace 上受到赞誉。

**主题 2：Agentic AI 崛起：Swarms、协议与复杂任务解决**

- **Anthropic 和 Claude Swarm 倡导多 Agent 架构**：Latent Space 的讨论显示，**Anthropic** 的多 Agent 系统使用 **Claude Opus 4** 作为主导，**Claude Sonnet 4** 作为子 Agent，在内部评估中比单个 Opus 4 的表现高出 **90.2%**（[Anthropic 多 Agent 系统博客文章](https://www.anthropic.com/engineering/built-multi-agent-research-system)）。同样，**Claude Swarm** 利用 **Claude Code 的 MCP 能力**组建分层专家团队（[Claude Swarm GitHub 仓库](https://github.com/parruda/claude-swarm)），并在 Shopify 等公司获得关注。
- **Model Context Protocol (MCP) 增强 Agent 互操作性**：MCP (Glama)、Latent Space 和 LlamaIndex 的讨论强调了 **MCP** 在工具使用和 Agent 协作中日益增长的重要性，诸如 [GitHub MCP Server 代码](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go) 和 **FastMCP**（[gofastmcp.com 网站](https://gofastmcp.com/)）等项目实现了领域隔离和稳健的工具访问。微软在 Data + AI Summit 上演示了使用 MCP 结合 LlamaIndex.TS 和 Azure AI Foundry 构建的 **AI Travel Agents**（[AI Travel Agents 演示详情](https://t.co/cNyVAcnf6K)）。
- **Factorio Learning Environment (FLE) 挑战 LLM 规划边界**：GPU MODE 的 #factorio-learning-env 频道围绕 **FLE** 展开了热烈讨论，该环境利用代码生成、生产分数反馈和 **REPL** 循环，在复杂游戏 Factorio 中辅助 LLM 规划。成员们提出了基于课程的代码生成和集成 Theory of Mind 模块的方案，甚至有一位用户通过 [GitHub issue 为 FLE 容器集成开发了 REST API](https://github.com/MortenTobiasNielsen/FLE-API-specification/issues/5)。

**主题 3：幕后探秘：微调、优化与硬件障碍**

- **Unsloth 和 Torchtune 引领微调前沿（及挫折）**：Unsloth AI 用户对新的 **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** 模型进行了基准测试，在测试用例中达到了 **69.4%** 的准确率，同时也在努力解决因 Hugging Face 命名规范导致的重复下载问题。Torchtune 开发者在 **Llama4 Maverick** 微调期间与 **DTensor 跨网格操作错误**（[Llama4 Maverick 微调错误日志](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&)）作斗争，并探索了 **iterable packing** 的创新。
- **AMD 与 NVIDIA 竞争升温，Mojo 获得 RDNA4 支持且 Unsloth 接近 AMD 兼容**：Modular (Mojo 🔥) 宣布在 Mojo nightly 版本中为直接 GPU 编程提供 **RDNA4 支持**，而 Unsloth AI 报告称由于采用了基于 Triton 的内核，其 **AMD GPU 兼容性** 已接近完成（[Unsloth AMD PR #2520](https://github.com/unslothai/unsloth/pull/2520)）。GPU MODE 的讨论强调了 **NVIDIA L40s** 因默认开启 ECC 激活而在云端表现不佳，并探讨了 **AMD MI300A** 架构。
- **优化器和量化努力追求巅峰性能与效率**：Torchtune 成员讨论了有望实现 **3 倍 VRAM 节省** 的 **ZO 优化器**（[ZO 优化器 arXiv 论文](https://arxiv.org/abs/2506.044303)），而 DSPy 用户则探索了集成 **TextGrad**（[TextGrad DSPy GitHub issue #1197](https://github.com/stanfordnlp/dspy/issues/1197)）并优化 **DeepSeek R1 7B**。Unsloth 用户还处理了可能与 logprob 计算期间 Token 值爆炸相关的 **KL 散度峰值**问题。

**主题 4：开源 vs. 封闭花园：模型、数据与去中心化辩论**

- **开源力量咆哮：Shisa v2 和 Kimi-Dev-72B 挑战闭源巨头**：HuggingFace 和 Nous Research AI 庆祝了强大开源模型的发布，如日本的 [Shisa v2 Llama3.1-405B 模型](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) 及其 SFT 数据集，以及 **MoonshotAI 的 Kimi-Dev-72B**（[Kimi-Dev-72B GitHub 页面](https://moonshotai.github.io/Kimi-Dev/)），后者为开源编程 LLM 树立了新的 SotA。这些发布引发了关于开源与封闭 AI 开发能力及未来的辩论。
- **去中心化之梦：Nous 在 Psyche 上启动预训练，Dawn Internet 部署分布式宽带**：Nous Research AI 正在 [psyche.network](https://psyche.network/) 上启动预训练，成员们希望“分布式技术只会变得更好”。与之相辅相成的是，[Dawn Internet 的 X 公告](https://x.com/dawninternet) 详细介绍了一种去中心化宽带协议，配备了支持 RL 的 GPU WiFi 路由器，进一步赋能去中心化 AI 应用。
- **伦理困境与版权难题引发社区对话**：EleutherAI 和 HuggingFace 用户辩论了版权法，一位 Eleuther 用户称其为“除非你是滥用者，否则就是一个笑话”（[相关 fxtwitter.com 帖子](https://fxtwitter.com/jyo_pari/status/1933350025284702697)），另一位 HuggingFace 成员因伦理担忧拒绝参与 AI 生成的反馈。一场关于 [YouTube 上封闭互联网与封闭 AI](https://youtu.be/vZVcBUnre-c) 的 WebSummit 演讲也在 Nous Research AI 中引发了讨论。

**主题 5：开发者体验与平台陷阱：Bug、计费与易用性之战**

- **额度灾难：API 计费困扰折磨 Perplexity 和 Manus 用户**：Perplexity AI 用户报告 **API 额度扣费超过实际使用量**，建议通过 **api@perplexity.ai** 联系。Manus.im 用户面临的问题更为严峻，有报告称 Manus “因为自身错误吃掉了我所有的 4k 额度”，一位用户声称它消耗了 **700/1000 额度却只交付了一个黑屏网站**。
- **UI 故障与性能问题令 Cursor 和 LM Studio 用户沮丧**：Cursor 社区成员指出了 UI 问题，如 Windows 上的 **命令执行失败** 浪费了推理额度，以及 **Claude 4 Sonnet 运行缓慢**。LM Studio 用户遇到了运行 LLM 时 GPU 发出的 **电感啸叫 (coil whine)**，并注意到 RAG 实现的局限性，无法扩展到 **31.46 MB** 以上。
- **解析问题与工具故障考验 LlamaIndex 和 Aider 爱好者**：LlamaIndex 用户遇到了 **LlamaExtract 的解析错误**，无法从文档中提取数据（[LlamaExtract 成功示例图](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6&)）。Aider 用户探索了集成 [RA-Aid GitHub 仓库](https://github.com/ai-christianson/RA.Aid) 以获取其 repo map 优势，同时也注意到 Aider 有时会消耗大量 Token 却似乎无所作为，最后不得不求助于暴力代码检索 (code grepping)。


---

# Discord: 高层级 Discord 摘要

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 2.5 Pro 表现平平，尽管编程能力出色**：用户发现 **Gemini 2.5 Pro** 在编程任务之外的表现令人失望，尤其是与其它模型在通用搜索和推理方面的对比，尽管它主打编程且拥有[宣传的能力](https://ai.google.dev/models/gemini)。
   - 一位用户表示 *Gemini 除了编程一无是处*，而另一位用户提到它经常 *在没有实际搜索网页的情况下胡编乱造*，还有一些用户报告每天仅限 **3 次试用**。
- **O3 Pro 竞争激烈，表现出情绪波动**：用户尝试用编程和解码任务挑战 **O3 Pro**，有时会提供来自其他模型的提示或示例，并注意到当把任务设定为竞争形式时，**O3 Pro** 的表现会有所提升，但也表现出不一致性。
   - 一位用户报告说 *如果你偏袒它而不选择它，回答质量反而会提高*。
- **MiniMax M1 发布，不及 Deepseek R1**：讨论了 **MiniMax AI** 的 **M1** 推理模型的发布，初步印象显示它虽然有趣，但不如 **Deepseek R1** 有效，一些用户注意到其思考过程过于冗长，正如[官方发布](https://x.com/MiniMax__AI/status/1934637031193514237)中所述。
   - 其推理输出的实用性引发了争论，尤其是考虑到缺乏来源链接。虽然有人建议 **MiniMax** 的 Agent 能力可能会随时间提高，但用户仍持怀疑态度。
- **Genspark 的“免费” O3 Pro 引发关注**：对于 **Genspark** 免费提供 **OpenAI** 的 **o3-pro**，用户表示怀疑，质疑 **Genspark** 如何能在 **OpenAI** 都不提供的情况下提供无限访问，暗示在达到一定使用阈值后可能存在限制或错误，正如其[服务描述](https://www.genspark.ai/agents?id=d7840faa-38ac-48a9-804a-2f17116cb2ca)中所述。
   - 一位用户报告看到其声称 *推理时间短得多*，但有人推测它 *不是完整的 o3 pro*，因为缺乏推理 Token。
- **Perplexity API 额度神秘消失**：用户报告 **API 扣费**超过了实际使用量，正通过邮件 (**support@perplexity.ai**)、Discord、开发者论坛和 Intercom 聊天等多种渠道寻求帮助。
   - 一位成员建议发送邮件至 **api@perplexity.ai**，以获取有关计费差异的支持。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Kingfall 的统治地位受到 Blacktooth 的挑战**：一些用户现在更倾向于 **Blacktooth** 而非 **Kingfall**，理由是其输出更精炼且尊重思考过程，而另一些人则因 **Kingfall** 的空间能力为其辩护。
   - 一位用户声称 **Blacktooth** 是 *LMArena 的产物*，在编程方面与 **Kingfall** 相比完全不可用。
- **GPT-5 的到来引发猜测**：关于 **GPT-5** 的发布时间表及其超越 **Google** 的潜力的讨论十分激烈，有人推测 **Grok 3.5** 或 **GPT-5** 很快将占据主导地位。
   - 社区讨论了付费 **ChatGPT** 用户是否能获得 **GPT-5** 的早期访问权，以及这种优势能持续多久。
- **Gemini 2.5 Pro 展现编程实力**：早期报告显示 **Gemini 2.5 Pro** 在编程任务中表现出色，特别是在 pygame 方面优于 **o4**，一段 [ChatGPT 对话](https://chatgpt.com/share/684e45eb-f118-8003-804e-3c9b562caab9)证实了 **2.5 Pro** 在纠正后完美回答了一个逻辑问题。
- **Minimax 的 M1 模型加入竞争**：Minimax 发布了开源推理模型 [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)；然而，初步基准测试显示它落后于 **o3** 和 **Gemini 2.5 Pro**。
   - 反应不一，有人怀疑是 *侥幸*，或者认为该模型可能只精通中文。
- **LMArena 排行榜向大科技公司倾斜？**：有人担心大型科技公司通过 Checkpoint 刷榜和增加 RLHF 数据机会在 LMArena 上获得优势，导致有人说 LMArena *基本上是美国大科技公司的排行榜*。
   - 有人断言开源模型或国外模型要么不出现，要么出现得非常晚。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 图像出现在 WhatsApp**：**ChatGPT 图像生成**现在可以通过 [1-800-ChatGPT](https://wa.me/18002428478) 在 **WhatsApp** 中使用，允许用户直接在应用内生成图像。
   - **WhatsApp** 上 **1-800-ChatGPT** 号码的推出为所有用户开启了图像生成功能，提供了一种随时随地创建图像的便捷方式。
- **成员质疑 AI 艺术检测**：成员们讨论了区分 **AI 生成**和**人类创作艺术**的难度，特别是随着复杂性的增加，并将其与识别伪钞的挑战相类比。
   - 他们指出，在这两种情况下，*细节越多，受到的审查就越严格*。
- **GPT Plus：开学季到了吗？**：关于 **GPT Plus** 订阅对学业价值的讨论展开了，用户在权衡 **$20** 的成本与带有 **GPT-4o** 的免费版本的功能。
   - 虽然免费版本可能已经足够，但 **Plus** 提供了更好的模型，如 **o3**、**4.1**、**4.5** 和 **o4 mini high**。
- **Veo 3 vs Sora：AI 视频大对决**：虽然有些人认为 **Sora** 与 **Veo 3** 相比*非常糟糕*，但另一位成员喜欢 **Sora**，认为它在*创意调优*和细节方面表现更好，而 **Veo 3** 则因能够生成像《星球大战》(**Star Wars**) 这样受版权保护的内容而脱颖而出。
   - **Veo 3** 的一个关键优势是其声音处理能力，而 **Sora** 被视为*一站式商店*，具有很高的价值。
- **GPT-4o 显示出跨对话记忆访问的迹象？**：一位用户报告称，**GPT-4o** *逐字*引用了在另一个与 **Custom GPT** 对话中共同创作的场景，引发了关于跨对话记忆访问的猜测。
   - 虽然有人建议这可能是准确的推理，但该用户指出了统计上的不可能，并提供了对话日志供审查。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 对 DeepSeek 新模型进行基准测试**：新的 **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** 模型在测试用例中达到了 **69.4%** 的准确率，速度为 **每用例 426.0 秒**，而 API 为 **716.6 秒**，在 **65k 上下文**下加载需要 **240GB** 显存。
   - 一位成员指出，与 FP8 需要的 **7-800GB** 相比，这使得它在本地部署更加可行。
- **Hugging Face 命名规范导致问题**：**Hugging Face** 缓存文件夹中的命名规范问题由于大小写差异导致了[重复下载](https://huggingface.co)。
   - 当使用 **Unsloth** 下载模型后再使用 **Hugging Face** 时可能会触发此问题，如果作者更改了仓库名称，会导致不同的命名规范。
- **微调指南优先考虑数据质量**：成员建议初学者从 **3B-8B** 等较小的模型开始，强调数据集的*质量大于数量*，并分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=jFl5Fewrieo)。
   - 视频建议新用户将 **80%** 的时间花在数据上。
- **Unsloth 的 AMD 兼容性即将就绪**：据报道，**Unsloth** 即将完全兼容 **AMD GPU**，因为 **Unsloth** 的大部分算子（kernels）是用 **Triton** 编写的，并指向了[这个 PR](https://github.com/unslothai/unsloth/pull/2520)。
   - 虽然 **Triton** 可以编译到 **AMD GPU**，但 **Triton** 设置可能需要针对 **AMD** 进行优化，这可能会影响性能。
- **当 KL 散度激增时**：一位成员询问 **KL 散度**（KL divergence）有时会在单步内飙升至 **x10000** 然后恢复正常，这种行为似乎并不影响训练。
   - 另一位成员提到这种情况经常发生，甚至在不使用 **Unsloth** 的 **Hugging Face** 运行中也会出现，可能是由于在**动作策略（acting policy）和参考策略（reference policy）之间进行对数概率减法和指数运算**时，特定 token 值爆炸导致的。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Claude 4 Sonnet 在 Cursor 中运行缓慢**：成员报告 **Claude 4 Sonnet** 在 Cursor 中的速度明显慢于 GitHub Copilot，尽管其整体[稳定性](https://link.to/stability)尚可。
   - 用户建议将 **Max Mode** 留给大型重构，或者切换到 **Gemini 2.5 Pro** 进行规划和代码审查。
- **Cursor UI 浪费推理额度 (Inference Credits)**：用户报告 UI 问题，如 Windows 上的**命令执行失败**和不一致的命令完成通知，导致推理额度被浪费。
   - 一位用户估计因这些故障损失了 **10-15% 的额度**，要求针对错误进行推理计数并加强 Windows 测试。
- **社区辩论 Model Context Protocol**：成员讨论了 **Model Context Protocol (MCP)**，一位用户强调了 AI 使用截图并自动集成错误消息的能力。
   - 另一位用户发现，投入时间编写**更好的 Prompt** 比使用截图更有效，并建议使用 [Wisprflow](https://wisprflow.ai) 进行语音转文字。
- **请求细粒度的代码隐私设置**：用户希望针对工作和个人项目设置**基于每个仓库**的代码隐私选项，表达了对代码存储和可访问性的担忧。
   - 社区正在推动项目层级的细粒度控制，以增强灵活性和安全性。
- **后台 Agent 缺乏 PR 创建权限**：Slack 中的后台 Agent 无法创建 Pull Request，尽管在 Cursor 集成设置中拥有所有权限，如 Request ID **bc-79e56de2-26d7-41a0-a5b3-b8b9e9f635d1** 和 **bc-d3acc5d6-2520-413f-9f7c-2fb36590215d** 所示。
   - 一位成员提议进行调试，并索要 Request ID 以调查权限问题。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **对 AI 生成反馈的伦理担忧升温**：一位成员对 [AI 生成的反馈](https://example.com) 表达了伦理担忧，拒绝参与 AI 生成的图像或视频。
   - 该成员表示：*即使我拥有理论知识，我也不会参与 AI 生成的图像/视频工作*。
- **Qwen 2.5：小体积，大影响**：一位成员强调了 **Qwen 2.5**（在 Ollama 上量化为 q4 的 7b 模型）令人印象深刻的性能和体积，并指出：*在 Benchmark 对比中没有人展示 Qwen 2.5，因为它太出色了*。
   - 讨论还涉及了 **Qwen 模型** 在中文和英文数据集上的多语言预训练。
- **HF Inference 成本令人头疼**：几位用户报告了 **HF Inference** 的成本问题，特别是在使用 **llama-3-70b-Instruct** 等模型时，发现免费额度不足并正在寻找替代方案。
   - 一位用户报告在最后几个单元进行多次尝试后支付了约 **$6**。
- **日本发布 Shisa v2！**：一个两人团队发布了 **Shisa v2**，这是日本有史以来训练的最强模型，并附带了概览报告（技术报告即将发布），可在 [HuggingFace](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) 获取（800GB+）。
   - 他们还更新了核心 SFT 数据集（可在 [HuggingFace](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt) 获取），声称在不降低英文性能的情况下提升了日文性能，该结论基于对 **7B-405B** 的 SOTA 开源模型的训练/发布。
- **开源 AGI 引发辩论**：一位成员宣称，如果他们创造了世界上第一个 **AGI**，他们将[开源它](https://example.com)，引发了关于潜在利弊的讨论。
   - 这一举动引发了关于风险与回报平衡的辩论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **自制回放功能为国际象棋排行榜注入活力**：一名成员为 [chess-leaderboard](https://dubesor.de/chess/chess-leaderboard) 增加了针对每场对局（包括过去和未来）的自制回放功能，并分享为 [chessreplay.gif](https://cdn.discordapp.com/attachments/1092850552192368710/1383513332916424715/chessreplay.gif?ex=6851b3b3&is=68506233&hm=1039e8ba4df4c19b19fc9c28f054c1eb809659a82c942b46aa7daebc3b48088c&)。
   - 开发者指出，这“可能比 lichess 的 gif 好一点，但在我的技术栈上实现起来很痛苦”。
- **作者提供书籍测试协助**：一位作者宣布其书籍于 **6 月 15 日**上线，并提出愿意协助进行测试。
   - 他们鼓励感兴趣的人员通过 DM（私信）联系以获取帮助。
- **Discord 标签请求未获回应**：一名成员对恢复 **OpenRouter Discord 标签**的请求未获理睬表示失望，并暗示愿意为此付费。
   - 由于一直没有得到回复，他们开玩笑地威胁要直接 ping **Alex Atallah**。
- **Token 浪费困扰 Claude 和 Copilot**：一名成员在检查 **Claude Code** 和 **GitHub Copilot** 的 Prompt 时发现，它们经常忽视 Token 效率，除非冗长会影响性能，否则会添加大量无关内容。
   - 研究结果表明，这些系统在优化 Prompt 时并未将简洁性放在首位。
- **GPT-4.1 Mini 招募 Beta 测试人员**：一名成员提议提供 **GPT-4.1 mini** 的访问权限，支持 **200K tokens/minute**，价格仅为**官方 Token 价格的 20%**，并兼容 OpenAI SDK。
   - 该优惠面向高用量的测试人员（可通过 DM 了解详情），重点关注 Cline.bot 和 BYOK/BYOB 等使用场景。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **TokenBreak 攻击收效甚微**：成员们讨论了[一种新型 TokenBreak 攻击](https://thehackernews.com/2025/06/new-tokenbreak-attack-bypasses-ai.html)，旨在绕过 AI 安全措施，但实验结果各不相同。
   - 一名成员风趣地评论了攻击相关截图中的 Logo 相似性。
- **AMD 迷你主机可运行部分大模型**：据一名成员称，[AMD Ryzen™ AI Max+ 395 --EVO-X2 AI 迷你主机](https://share.google/3WVLR8Up7pjVjq35mI)可以流畅运行一些大模型。
   - 然而，其他人指出它本质上是一个由 HIP SDK 支持的“马甲核显（igpu）”，运行 **70B** 模型的速度约为 **5t/s**，运行 **Qwen3 235B A22B** 模型的速度约为 **14t/s**。
- **LM Studio 无法扩展 RAG 容量**：针对成员提出的将 LM Studio 中 **RAG 容量**从 **31.46 MB** 增加到 **100 MB** 的咨询，得到的回复是“不可能”，因为目前的 RAG 实现还比较基础。
   - 这一限制归因于当前的 RAG 实现尚处于初级阶段。
- **GMKtec 的 Windows 安装变得非常麻烦**：一名用户在 **GMKtec** 机器上安装 **Windows** 时遇到问题，报告安装失败以及 **Rufus** 创建的可移动介质存在问题。
   - 这涉及在 GMKtec 机器上安装 Windows 的尝试，凸显了兼容性或驱动程序问题。
- **电感啸叫困扰 GPU 用户**：用户注意到，与游戏负载相比，运行 **LLM** 时显卡的**电感啸叫（coil whine）**显著增加。
   - 一位在使用 **5090** 时遇到较严重电感啸叫的用户建议通过降压（undervolting）来同时降低功耗和噪音。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Groq 合作，速度快如闪电**：一名成员询问了 **Groq** 的性能，注意到他们最近与 **Hugging Face** 的合作，暗示了其强大的性能或易用性。
   - 进一步的讨论可能会揭示 **Groq** 擅长的特定用例，从而可能影响模型部署策略。
- **ECC 激活对 L40s 性能的影响**：成员们讨论了 **L40s** 在云环境中可能表现不佳，因为 **ECC** 默认被激活，从而影响了性能。
   - 这属于配置问题而非硬件问题，表明在云部署中需要优化设置。
- **ThunderKitten 在旧款 GPU 上大显身手**：成员们讨论了在 Kaggle 上可用的旧款 GPU（如 **T4** 和 **P100**）上运行 **ThunderKitten**，这很可能是可行的。
   - 一名成员建议使用 **4090 TARGET** 进行编译，并报告任何损坏以帮助解决兼容性问题，旨在实现更广泛的硬件支持。
- **FLE：LLM 的 Factorio 冒险**：成员们发现 **FLE** 的设置（使用代码生成、生产得分反馈、**REPL** 循环和内存压缩）是一个有用的脚手架，它减少了动作空间，并为 **Factorio** 中的 **LLM** 引入了结构和规划。
   - 一名成员建议采用基于课程的代码生成方法，由微型目标和 **FLE** 循环内的心理理论（theory of mind）模块引导，这似乎是探测该环境下 **LLM** 规划极限的一种很有前景的方法。
- **AMD MI300A 架构探索**：成员们讨论了融合的 **AMD CPU-GPU 平台**，特别是 **MI300A** 架构的 **IOD** 和 **infinity cache**，推测内存是如何在芯片之间分配的。
   - 一名成员提到使用 `s_getreg` 来查明 shader 运行在哪个 **XCD** 和 **CU** 上，并以此测量到内存中不同位置的访问延迟。



---



## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Minimax 模仿 Manus，却在积分（Credits）上栽了跟头**：成员们观察到 [agent.minimax.io](https://agent.minimax.io/) 抄袭了 **Manus**，而后者在宣布积分制度之前曾具有*巨大的潜力*。
   - 一名成员抱怨*愚蠢的定价毁了它*，指的是积分系统的宣布。
- **Manus 因自身错误消耗积分**：用户报告称 **Manus** 正在因为自身的错误而消耗积分，其中一人表示它*因为自己的错误消耗了我所有的 4k 积分*。
   - 一些用户报告称，它消耗了 **700/1000 积分却交付了一个黑屏网站**。
- **免费的 Lume 被推销为 Manus 的替代方案**：成员们就 [lume.im](https://lume.im) 是否可以作为 **Manus** 的免费且无限制的替代方案展开了辩论。
   - 一名用户对 **Lume** 的推广引发了关于推销（shilling）和垃圾信息的指责。
- **Gemini 取得进展，让 Manus 显得逊色**：一名成员发现，在特定任务中 *Manus 做不到，但 Gemini 可以*，并分享了一个 [Gemini 输出链接](https://g.co/gemini/share/ef9c34a02d31)。
   - 他们还补充道：*Gemini 是目前最好的静态画布。Manus 不是静态的，所以我们无法将两者结合。*
- **Manus 反应迟钝且健忘**：用户抱怨 **Manus** 速度慢且不遵循指令，新的更新使其变得更糟。
   - 例子包括**简单的文档编译耗时 40 分钟并消耗了 400+ 积分**。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous 启动 Psyche 预训练**：Nous Research 在 [psyche.network](https://psyche.network) 上启动了预训练，这恰逢 **Jensen Huang** 发表关于预训练与后训练价值的评论。
   - 一位成员提到 *分布式技术从现在起只会变得更好*，并将使去中心化训练受益，而其他人则持怀疑态度。
- **Dawn Internet 部署去中心化宽带**：[Dawn Internet](https://x.com/dawninternet) 推出了一种去中心化宽带协议，通过固定无线屋顶天线提供 **千兆互联网**。
   - 他们最新的 **WiFi 路由器** 配备了支持 RL 的 GPU，为去中心化应用扩展了可能性。
- **Hermes 4 即将开始训练**：Nous Research 将于周一开始 **训练 Hermes 4**，使用最新的 Mistral，不过训练和准备仍需一段时间。
   - Zeus 系列的新模型将不会基于旧的 Hermes。
- **Kimi-Dev-72B 达成编程 LLM 里程碑**：**MoonshotAI** 发布了 [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B)，这是一个通过大规模强化学习优化的新型开源编程 LLM。
   - 它在 **SWE-bench Verified** 上取得了开源模型中的新 SOTA，得分为 **60.4%**，它在 Docker 中修复真实的仓库，并且只有在整个测试套件通过时才会获得奖励。
- **WebSummit 演讲抨击封闭互联网**：一位成员分享了在 [温哥华 WebSummit 上的演讲](https://youtu.be/vZVcBUnre-c)，内容关于封闭互联网和封闭 AI，一半是历史，一半是抨击。
   - 另一位用户在 [FXTwitter](https://fxtwitter.com/jyo_pari/status/1933350025284702697) 上转发了此内容。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Electron 应用坐拥高估值**：针对 **VS Code** 分支（fork）高达 **90 亿美元的估值**，一位成员开玩笑说应该 fork 一个 Electron 应用而不是构建 **TUI**。
   - 另一位成员指出 **VS Code** 在大学生中的普及率极高，强调了尽管它是一个“过度设计的解决方案”，但依然具有重要地位。
- **Aider 从 RA-Aid 集成中获益**：在研究了 [RA-Aid](https://github.com/ai-christianson/RA.Aid) 后，一位用户注意到 **Aider** 及其 **repo map** 的优势，使用户能够将文件添加到上下文中。
   - 该用户还对 **Aider** 在求助于对代码库进行暴力 grep 之前，似乎无所事事地消耗了 5 美分和 32K token 感到惊讶。
- **角色设定（Personas）不会提升 LLM 性能**：一位用户分享了一篇 [Arxiv 论文](https://arxiv.org/html/2311.10054v3)，该论文认为 *与不添加角色的对照设置相比，在系统提示词中添加角色并不能提高模型在各种问题上的性能*。
   - 他们以此支持自己的观点，即这种情况已经持续一段时间了。
- **头脑风暴 UX 产生疯狂想法**：提示 DeepSeek 生成了 **Realistic（现实）**、**Outside the Box（跳出框架）** 和 **Completely Bonkers（完全疯狂）** 三个层级的功能构思。
   - **Completely Bonkers** 层级包含了诸如 *反重力代码重排（Anti-Gravity Code Reflow）* 和 *多重宇宙分支（Multiverse Branching）* 等建议。
- **Aider 管理上下文窗口**：一位用户请求 **Aider** 增加自动管理其上下文窗口的功能，而不仅仅是添加文件。
   - 作为回应，另一位用户指向了 [repomap](https://aider.chat/docs/repomap.html) 和 [scripting](https://aider.chat/docs/scripting.html) 文档，以便为 Aider 提供 repo map 并将 Aider 作为控制上下文的工具。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Claude Swarm 管理团队**：**Claude Swarm** 是一款利用 **Claude Code 的 MCP 能力**来建立专家分层团队的工具，正在 Shopify 和其他公司受到关注 ([代码在此](https://github.com/parruda/claude-swarm))。
   - 有用户建议通过增加一名 *招聘专家 (recruiter expert)* 来扩展团队，以管理 Swarm 配置和团队扩张。
- **主动型 AI Agent 定义面临 IMPACT 框架考验**：一篇博文将 **主动型 AI Agent** 定义为能够控制自身进度、工作流、拥有持久记忆并使用有状态工具的实体 ([Substack 文章](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents))。
   - swyxio 根据他的 IMPACT 框架对该定义进行了评估，并指出其缺乏 **意图 (intent)**、**规划 (planning)** 和 **授权 (authorization)**。
- **Anthropic 的多 Agent Opus 系统**：**Anthropic** 报告称，一个使用 **Claude Opus 4** 作为主 Agent 并配合 **Claude Sonnet 4** 子 Agent 的多 Agent 系统，在内部研究评估中比单 Agent Claude Opus 4 的表现高出 **90.2%** ([Anthropic 博客文章](https://www.anthropic.com/engineering/built-multi-agent-research-system))。
   - 由于并行化和广泛的工具使用，该系统消耗的 **Token 数量约为对话模式的 15 倍**，需要通过 Prompt Engineering 来防止过度的 Agent 生成和网页搜刮；LLM 也被用于评估输出结果。
- **Obsidian Copilot 充当 Obsidian Markdown 作者的光标**：用户讨论了在 AI 辅助下处理 Markdown 文件的工具，提议将 **Obsidian Copilot** 作为一个可行的选择 ([Obsidian Copilot](https://www.obsidiancopilot.com/en))。
   - 用户希望获得简单聊天之外的功能，例如按主题拆分笔记、打标签、聚合笔记以及使用 Anki MCP 创建闪卡。
- **月之暗面 (Moonshot AI) 开源 Kimi-Dev-72B**：**Moonshot AI** 开源了他们的 **Kimi-Dev-72B** 模型，在开源模型中以 **60.4%** 的成绩达到了 SWE-bench Verified 的 State-of-the-Art (**SotA**) 水平 ([HF 模型](https://moonshotai.github.io/Kimi-Dev/))。
   - 该公告由 Aran Komatsuzaki 在 Twitter 上发布，并提供了 Hugging Face 模型和 GitHub 仓库的链接。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **破解代码：探索新的研究领域**：在深入研究陌生领域时，一位成员提出了识别里程碑式论文的策略，包括利用教授的讲义和探索顶尖实验室近期出版物中的引用，另请参阅 [LessWrong 上的讨论](https://www.lesswrong.com/collaborateOnPost?postId=9eehTtLsTBZR9Bd7Q&key=3b19568356a160ba7cf74febafdf33)。
   - 这些见解侧重于使用逆向工程来理解思想流向，并识别视频生成等领域中的关键作品。
- **叙事对决：LLM 作为叙事架构师**：一位成员请求发布其探讨 **LLM** 作为叙事模拟器的 English 101 论文，但被拒绝了，不过分享了一个指向 [Tumblr 上的 The Void 帖子](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void) 的链接，其中包含相关分析。
   - 讨论暗示了 **LLM** 在架构上与叙事的一致性，引发了对其涌现叙事能力的兴趣。
- **数学错误玷污 AI 撰写的文章**：成员们对 AI 生成的论文发出了警告，引用了一篇针对 Apple 论文的有缺陷回复（可疑），据称该回复充斥着来自 [arxiv.org](https://arxiv.org/abs/2506.09250) 的数学错误，具体参考了 [这条推文](https://x.com/BlancheMinerva/status/1933845602917290145)。
   - 这提醒人们，在没有彻底验证的情况下，依靠 AI 进行学术工作存在潜在陷阱。
- **WSL 工作者的烦恼：PyTorch 并行处理的危机**：一位用户强调了 **PyTorch dataloader workers** 在 WSL 中被 *signal 杀死* 的问题，特别是在处理高 worker 数量和超长序列长度时。
   - 建议的解决方案包括检查 `/var/log/syslog` 以查找潜在的 **OOM** 错误，并在处理长视频序列时勤勉地管理内存。
- **版权冲突：法律笑柄隐现？**：一位用户挑衅地表示，*除非你是滥用者，否则版权法就是一个笑话*，质疑 DMCA 和版权欺诈 (copyfraud) 的处罚，并链接到了 [fxtwitter.com](https://fxtwitter.com/jyo_pari/status/1933350025284702697) 和 [arxiv.org](https://arxiv.org/abs/2506.10943)。
   - 该评论引发了关于当前版权执法机制的有效性和公平性的讨论。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Notebook LM Plus 访问状态仍不明朗**：一位用户询问了其 **NotebookLM Plus** 的访问状态，尽管该用户是已付费的 **AI Pro** 订阅者并使用了 **1900** 个来源，并分享了一个包含 **4 个子层级**、大小约为 **9MB** 的 [NotebookLM 思维导图](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=6851e6a5&is=68509525&hm=f1cd4c62bca69adfe75ec5f0adc1b806be3b22beaa24c3bac7597185b382a6d7)。
   - 讨论强调了需要澄清不同订阅层级对应的功能访问权限。
- **AI 平台旨在主导 PM 面试**：一名成员正在测试一个专为 **PM 面试** 定制的 **对话式 AI 平台**，并寻求 Beta 测试者的反馈，测试者可以 [通过提供的表单进行注册](https://forms.gle/P3f2JP3vVB62vedb7)。
   - 该计划旨在利用 AI 来增强面试准备和验证流程。
- **播客音频质量下降！**：用户报告 **NotebookLM 播客** 的音频质量和内容有所下降，指出对*原始材料*的框架处理显得机械且重复，该问题影响了生成的播客。
   - 生成的播客被描述为*听起来破碎且虚假*。
- **NotebookLM 支持公式的 LaTeX 标记**：**NotebookLM** 现在支持数学和科学公式的 **LaTeX 标记**，与其他 LLM 类似，用户可以使用在线或离线 **LaTeX 渲染器** 来查看公式。
   - 为了提供更专业的支持，开发者创建了 [LatexInNotebooklm](https://github.com/ergs0204/LatexInNotebooklms) 扩展。
- **NotebookLM 现已支持图片上传**：用户发现 **NotebookLM** 现在支持直接从设备上传图片，消除了之前对 **Google Drive** 的依赖。
   - 可以通过选择*选择文件*选项或拖拽来上传图片。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **DTensor 困扰分布式 Llama4**：成员在对 **Llama4 Maverick** 进行多节点分布式微调 (finetuning) 时遇到了与 **DTensor 跨网格操作** 相关的 `RuntimeError`，堆栈跟踪信息可在 [output.log](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&) 中查看。
   - 该错误在不同数量的节点（8 个 vs 12 个）下表现不同，指向了 **fused optimizer** 和网格配置的潜在问题。
- **可迭代打包 (Iterable Packing) 创新即将到来**：一名成员正在开发一个基于 [pytorch/data](https://github.com/pytorch/data) 构建的、具有 **iterable packing** 功能的私有微调库，展示了良好的结果和预取 (prefetching) 能力。
   - 他们预计下周开源该库，并强调许多库中都缺少打包的 DPO。
- **Fused Optimizer 在全量微调中受挫**：在尝试训练期间，发现 **fused optimizer** 会导致问题，特别是在创建检查点 (checkpoint) 时导致 `nccl` 超时，而使用非融合优化器则可以在 8 个节点上进行训练。
   - 建议通过增加 `NCCL_TIMEOUT` 环境变量，或设置 `total_epochs=self.total_epochs+1` 以启用异步检查点，来缓解这些问题。
- **Mistral Small 亮相，令人失望**：尽管最近发布了 [Mistral Small 模型](https://mistral.ai/news/mistral-small-3-1)，但它并未给所有人留下深刻印象，一位成员表示 *Mistral Small 的结果，即使是在他们自己的博客文章中，看起来也几乎不比 Gemma 3 更好*。
   - 该成员还澄清说，他们在调研时最初误点击了 **Magistral** 而不是 **Mistral**。
- **ZO 优化器承诺节省 VRAM**：成员们讨论了 **ZO 优化器** 及其实现 **3 倍 VRAM 节省** 的潜力，并引用了相关论文 ([arxiv.org/abs/2506.044303](https://arxiv.org/abs/2506.044303))。
   - 成员们一致认为，**ZO** 论文最重要的价值在于其在不同规模上的可扩展性，以及它主要使用了非合成实验。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **RDNA4 支持到来**：从上一个 nightly 版本开始，Mojo 已支持 **RDNA4** 的直接 GPU 编程，但模型需要针对 **RDNA 特定路径**进行矩阵乘法运算。
   - 引入必要 **WMMA** 操作的初步补丁使模型在 **RDNA3**+ 上更接近完整功能。
- **Zen 4 带来 BFloat16 提升**：虽然 **5950x** 缺乏 **AVX512_BF16** 支持，但 **Zen 4** 及以上的 CPU（如 **Ryzen 7000 系列**）提供了一些 **bfloat16** 支持。
   - 目前尚未确认这些是否包含 CPU 推理所需的精确 **FMA** 指令，但这是一个正确的方向。
- **Mojo 的测试结构让用户困惑**：用户对 **Mojo** 的测试代码库结构表示沮丧，特别是测试文件中的导入问题；使用 `mojo test -I .` 运行允许测试将正在测试的包作为库导入。
   - 一位用户建议参考 [ExtraMojo](https://github.com/ExtraMojo/ExtraMojo) 作为良好的项目结构示例。
- **LLVM 导致 Mojo 二进制文件臃肿**：**Mojo** 二进制文件的大部分体积来自于静态链接 **LLVM**，MAX 本身约为 **750 MB**，随 MAX 附带的 .mojopkgs 约为 **100 MB**。
   - 团队正在积极致力于减少 **LLVM** 副本的数量。
- **CUDA 流不需要主机同步**：一名成员询问在 [Puzzle 12](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step) 中 `ctx.synchronize()` 是否必要；Modular 团队成员确认 *DeviceContext* 使用 **CUDA stream**，因此执行顺序与调用顺序一致。
   - Modular 团队成员确认不需要显式同步，并承诺相应调整文档。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Agentic 框架欢迎 MCP**：一名成员询问 **MCP** 如何融入 Agentic 框架，建议将编排 Agent 作为顶层，由特定 Agent 访问多个 **MCP** 服务器进行工具选择和内存存储，客户端也可以使用具有工具重排序功能的更智能的主机。
   - 开发暴露所有 GitHub API 的单个 **MCP** 服务器的团队正在探索编排服务器的想法，该服务器可以调用或代理到其他 **MCP** 服务器，并鼓励查看代码 [GitHub MCP Server](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go)。
- **FastMCP 通过子服务器隔离领域**：一名成员指出 [fastmcp](https://gofastmcp.com/) 可以挂载 **MCP** 服务器，使路由服务器能够托管子服务器以进行领域隔离。
   - 一名成员通过指出 streamable-http 需要包含 `/mcp/` 的完整 URL，且默认 streamable-http 端口是 **8000** 而非 6277，帮助解决了一个 **fastmcp** 连接错误。
- **SchemaPin 阻止 MCP Rug Pulls**：一名成员宣布推出 **SchemaPin**，旨在防止 **MCP Rug Pulls** 及类似攻击，[仓库](https://github.com/ThirdKeyAI/SchemaPin) 已在 GitHub 上可用。
   - [主页](https://schemapin.org) 提供了实现 **SchemaPin** 的简便方法，所有 Glama **MCP** 服务器现在都支持 **streamable HTTP**，例如 [glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257]。
- **Excel MCP Server 在 GitHub 走红**：一名成员分享了他们的仓库 [excel-mcp-server](https://github.com/haris-musa/excel-mcp-server)，该仓库在 GitHub 上两次登上趋势榜。
   - 该成员欢迎对该项目提出任何反馈。
- **MCPCat 调试你的 MCP**：一名成员正在通过 MCPCat 为 MCP 开发用户分析和实时调试功能，仓库地址在 [这里](https://github.com/mcpcat)。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 文档修复！**：一位用户发现并报告了 [Cohere 文档](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide)中的一个拼写错误，具体为 `co = cohere.SagemakerClient()` 中的 `SagemakerClient` 应当使用小写的 `m`。
   - 此修正确保了 **Amazon Sagemaker Setup Guide** 的准确实施。
- **LLM 团队协作策略探讨**：一位用户正在研究团队如何将 **ChatGPT** 和 **Claude** 等大语言模型集成到工作流中，寻求关于采用这些模型后工作流程变化及缺失环节的见解。
   - 该咨询旨在了解团队与 **LLMs** 协作的演变格局。
- **工具浮现**：用户报告在 **Cohere** 模型响应中偶尔会出现名为 **direct-injected-document** 的工具。
   - 社区正在寻求 Prompt 示例和模型规格，以进一步调查此行为。
- **隐私保护倡导者表达热情**：计算机科学毕业生 Yasir Khan 介绍了自己，并提到他在 **secure machine learning**（安全机器学习）和 **privacy-preservation**（隐私保护）方面的工作。
   - 他正在寻求 **AI/ML** 项目的合作机会。
- **Ollama 模型评价**：一位 AI 爱好者分享了他们使用 **models from ollama** 的乐趣。
   - 他们表示这 *非常有趣*。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Data + AI Summit 聚焦 Agentic 工作流**：@databricks **Data + AI Summit 2025** 展示了关于 **agentic document workflows** 的内容，CEO @jerryjliu0 发表了一场座无虚席的演讲，详情见[此处](https://t.co/jS2Nfwxxb3)。
   - @microsoft 演示了新的 **AI Travel Agents**，该工具与 **Model Context Protocol**、**LlamaIndex.TS** 以及 @Azure AI Foundry 协同工作，详情见[此处](https://t.co/cNyVAcnf6K)。
- **旧金山活动聚焦构建安全的 AI Agents**：即将于旧金山举行的晚间活动将提供关于在生产环境中构建和保护 **AI Agents** 的专家见解，涵盖[此处](https://t.co/MVd2rwSVIE)概述的最佳实践。
   - 活动邀请了开发者关系副总裁 @seldo，以及来自 Ravenna 和 @auth0 的专家，他们将讨论 **Building Real-World Agents**。
- **LandingAI 工具挑战 LlamaIndex？**：成员们讨论了由 **Dr. Andrew Ng**（吴恩达博士）的公司推出的 **LandingAI** 新视觉 Agent 文档理解工具，引发了继此前对比 **Mistral** 的[帖子](https://www.linkedin.com/posts/jerry-liu-64390071_mistral-ocr-is-nice-and-fast-but-other-models-activity-7303803148907790336-OP9y)之后，该工具与 **Llama Parse** 的对比。
   - 更多关于该公司工具的信息可在 [LandingAI 官网](https://va.landing.ai/home)查询。
- **Synk 积极扩招开发团队**：**Synk** 正在为其去中心化浏览器系统项目积极招聘开发人员，职位包括 **back-end, front-end, and blockchain development**，以及 **QA Engineers**、**DevOps Engineers**、**Moderators** 和 **Marketing Analyst**。
   - 有意向的候选人可前往 [Synk 的 X 页面](https://x.com/Synk_ws)了解更多关于*签署正式雇佣文件、保障薪资和灵活排班*的信息。
- **LlamaExtract 用户遇到解析问题**：用户报告在使用 **LlamaExtract** 时遇到 **parsing errors**（解析错误），导致无法从文档中提取数据。
   - 尽管部分成员仍面临问题，但一名成员确认他们能够接收到数据，并分享了一张使用 LlamaExtract 成功提取的截图 ([image.png](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6))。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 优化模式整合 (Optimization Patterns Incorporations)**：一位成员寻求关于在 **DSPy** 中整合**优化模式 (optimization patterns)**的见解。
   - 未提供关于特定模式或用例的进一步细节。
- **DSPy "Runners" 实现跨语言功能**：一位成员提议构建 **DSPy "runners"**，利用保存的 **JSON definitions** 来执行编译后的程序，从而实现跨语言功能，例如 Swift 通过托管 API 使用编译后的程序。
   - 针对 **JSON output** 中未捕获的程序逻辑（如 signatures 和 modules）的序列化问题，提出了挑战。
- **TextGrad 优化器推迟**：一位成员询问了将 **TextGrad** 作为 DSPy 优化器进行集成的更新进展，引用了 [GitHub 上的 issue #1197](https://github.com/stanfordnlp/dspy/issues/1197)。
   - 该成员对 **TextGrad** 在优化复杂 prompts 方面的潜力表现出极大热情，并询问了将其集成到 DSPy 的变通方案，但目前尚未提供解决方案。
- **模型在 DAIS 会议上编写提示词**：一位成员分享了其名为 *Let the Model Write the Prompt* ([dbreunig.com](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)) 的 **DAIS session** 记录，并提供了该会议录像的 [YouTube 链接](https://youtu.be/I9ZtkgYZnOw?si=XGArjkQSVUlzrEAr)。
   - 讨论集中在模型如何自主生成 prompts，并给出了 DAIS 会议中的实际案例，但未提供进一步的技术细节。
- **DeepSeek R1 7B 在 DSPy 优化中表现不佳**：一位成员报告称，在 **DSPy-Text2SQL** 演示中，使用 **DeepSeek R1 7B** 的优化结果不如 **GPT-4o-mini**。
   - 建议在尝试 **LabeledFewShot** 和 **BootstrapFewShotWithRandomSearch** 后，提供更多的 schema 信息可能会增强 **DeepSeek R1 7B** 的性能。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **证书将于 7 月中旬发放**：一位成员表示 **LLM Agents Berkeley MOOC** 的证书*将于 7 月中旬发布*。
   - 这解决了用户关于发放时间表的疑问。
- **只要付出合理的努力即可获得证书**：一位成员澄清说，对于通过 **Google Forms** 提交的每项作业都会发送电子邮件确认，只要以**合理的努力 (reasonable effort)**完成所有内容，就会授予证书。
   - 这解决了用户对作业评分和证书资格的担忧。
- **分享 MOOC 测验存档**：一位成员分享了 [2025 春季 MOOC 测验存档](https://docs.google.com/document/d/1A00cUWux-J0p9AOnwpyNN3Rb5QFRsbBgAmvgPMezJ10/edit?usp=sharing)。
   - 该存档也可在课程网站的 Quizzes 部分找到。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **ControlThrive 创始人向社区致意**：AI/ML 咨询机构 **ControlThrive** [controlthrive.com](https://www.controlthrive.com/) 的创始人 Servando 向社区介绍了自己。
   - 他邀请成员在 [LinkedIn](https://www.linkedin.com/in/servando-torres-239a26b0/) 或 X 上与他建立联系。
- **Outerbounds 活动即将举行**：Servando 宣布了他将与来自 **Outerbounds**（Netflix ML 基础设施背后的团队）的 Eddie 共同举办的活动。
   - 他分享了[活动链接](https://lu.ma/nw4xccle)并鼓励社区成员参加。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Claude Sonnet 4 亮相！**：**Claude Sonnet 4** 和 **Claude Sonnet 4 (Thinking)** 已通过 [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing) 向所有付费计划开放。
   - 这些模型承诺为各种 AI 应用提供增强的性能和功能。
- **Mohan 发表对 Claude 的印象**：Mohan 在 [X](https://x.com/_mohansolo/status/1933605162775687482) 上分享了一些对 **Claude 的印象**。
   - 虽然来源中未包含 Mohan 评论的具体背景，但转发内容突出了社区对 **Claude** 的看法。



---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想更改接收这些邮件的方式吗？
您可以从该列表中 [退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：按频道分类的详细摘要和链接

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1383396171413717154)** (1 条消息): 

> `Perplexity Research Improvements, Finance Pages Key Issues, Tasks Automated Search, Discover Page Update, Finance Futures Graphs` 


- **Perplexity 改进研究功能**：Perplexity AI 发布了名为 *Improved Research* 的新更新，详情见其 [6 月 13 日更新日志](https://www.perplexity.ai/changelog/what-we-shipped-june-13th)。
   - 此次更新包括 **财经页面的关键问题修复**、引入了 **带有任务的自动化搜索**、更新了 **Discover Page**，并在财经板块增加了 **期货图表**。
- **财经页面现已显示期货图表**：Perplexity AI 在其 [6 月 13 日更新日志](https://www.perplexity.ai/changelog/what-we-shipped-june-13th)中宣布在财经板块增加 **Futures Graphs**。
   - 这一增强功能旨在为用户提供 **更全面的金融数据可视化**。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1383160006173790301)** (1113 条消息 🔥🔥🔥): 

> `Gemini 2.5 Pro, Claude Opus 4, o3 Pro, MiniMax M1, Genspark` 


- **Gemini 2.5 Pro 表现不佳，过度炒作？**：成员们发现 **Gemini 2.5 Pro** 在编程任务之外的表现令人失望，尤其是与其他模型在通用搜索和推理方面相比，尽管它以编程专长和[宣传的功能](https://ai.google.dev/models/gemini)为卖点。
   - 一位用户表示 *Gemini 在编程之外就是垃圾*，另一位用户提到它经常 *在没有实际搜索网页的情况下胡编乱造解释*，还有人报告每天有 **3 次试用** 的限制。
- **O3 Pro 变得具有竞争力**：用户尝试用具有挑战性的编程和解码任务来测试 **O3 Pro**，有时会提供来自其他模型的提示或示例；他们注意到在将其设定为竞争关系时 **O3 Pro** 的表现会有所提升，但也表现出不一致性。
   - 一位用户报告说 *如果你对它有偏向但不选择它，回答质量反而会提高*。
- **MiniMax M1 推理模型问世**：讨论了 **MiniMax AI** 的 **M1** 推理模型的发布，初步印象认为它很有趣，但不如 **Deepseek R1** 有效，一些用户注意到了其冗长的思考过程，正如[官方发布](https://x.com/MiniMax__AI/status/1934637031193514237)中所述。
   - 有建议认为，鉴于 **MiniMax** 以往模型的记录，其 Agent 能力可能会随时间提高，但其推理输出的实用性仍存争议，特别是考虑到缺乏来源链接。
- **Genspark 提供免费 O3 Pro，用户仍持怀疑态度**：**Genspark** 免费提供 OpenAI 的 **o3-pro** 遭到了质疑，用户怀疑 **Genspark** 如何能在 OpenAI 都不提供的情况下提供无限访问，并暗示在达到某些使用阈值后可能会出现限制或错误，正如其[服务描述](https://www.genspark.ai/agents?id=d7840faa-38ac-48a9-804a-2f17116cb2ca)中所述。
   - 一位用户报告看到其声称 *推理时间短得多*，但有人推测它 *不是完整的 o3 pro*，因为缺乏推理 token。
- **对 Perplexity 记忆功能和特性的不满**：用户分享了对 Perplexity 记忆能力的沮丧，一位用户指出 *Perplexity 患有阿尔茨海默症*，其他人讨论了自定义指令和浏览上下文似乎会影响或污染后续搜索，还有人观察到一些小故障，例如生成的图像在完全加载后点状叠加层仍然存在。
   - 一位用户指出他们关闭了该功能，并称 *我对自己的数据很吝啬……但 PPLX 就像是……兄弟……直接回答该死的问题就行了……*。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1383506747855212757)** (8 messages🔥): 

> `Shareable Threads, Nvidia GB200, Driver Stability, Emergence, Android App Security` 


- **让 Thread 可分享！**：一位成员要求用户确保他们的 thread 是 `Shareable` 的。包含了[如何让 thread 可分享](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)的链接。
- **Nvidia GB200 百亿亿次级 AI 超级计算机即将到来！**：一位成员分享了关于 [Nvidia GB200 百亿亿次级 AI 超级计算机](https://www.perplexity.ai/page/nvidia-gb200-exascale-ai-super-eQbdxUvkTO2NHGMvzvYDrQ) 的链接。预计它将为 *"AI 工作负载提供前所未有的性能和能力"*。
- **Nvidia 566.36 驱动稳定性**：一位成员分享了关于 [Nvidia 566.36 驱动稳定性](https://www.perplexity.ai/page/nvidia-566-36-driver-stability-MfQyQNKUROa.dIBMOBuDvg) 的链接。该页面提供了关于 *"故障排除和保持最佳性能"* 的建议。
- **高度令人担忧的涌现 (Emergence)**：一位成员分享了关于 [高度令人担忧的涌现](https://www.perplexity.ai/page/the-highly-alarming-emergence-VNgl7khJQ0GQDhWvN2F5Xg) 的链接。未提供进一步背景。
- **Perplexity Android 应用安全性**：一位成员分享了关于 [Perplexity Android 应用安全性](https://www.perplexity.ai/page/perplexity-android-appsecurity-zDJ7FtsfQUiy9pmLAmt.Ngok) 的链接。未提供进一步背景。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1383492664883806330)** (7 messages): 

> `API credit charges, Perplexity Linux CLI client, AI startup resources` 


- **API 额度扣费难题困扰用户**：用户报告 **API credit charges** 超过了实际使用量，并通过邮件 (**support@perplexity.ai**)、Discord、开发者论坛和 Intercom 聊天等多种渠道寻求帮助。
   - 一位成员建议向 **api@perplexity.ai** 发送邮件，以寻求有关账单差异的支持。
- **Perplexity CLI 客户端出现**：一位成员分享了他们的 [Perplexity Linux CLI client](https://github.com/dawid-szewc/perplexity-cli)。
   - 开发者创建了一个用于网页搜索的 AI 项目。
- **创业者搜寻资源**：一位成员表示有兴趣建立一家 **AI startup**，并请求获取更多了解 Perplexity 的资源。
   - 他们寻求关于将其用于 **web searching** 和产品开发的指导。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1383159907036954695)** (963 messages🔥🔥🔥): 

> `Kingfall vs Blacktooth, Grok 3.5 release, Gemini 2.5 Pro, Minimax M1 open source, LLM privacy` 


- **部分用户弃用 Kingfall 转投 Blacktooth**：一些用户认为 **Blacktooth** 是更好的模型，因为它具有精炼的输出并尊重思考过程；而另一些人则更喜欢 **Kingfall** 的空间能力和神奇时刻。
   - 一些人认为 Blacktooth 是 *LMArena 的刷榜产物，在编程方面完全不如 Kingfall，这也是为什么其生成的 SVG 保真度不高的原因*。
- **GPT-5 发布时间表引发争论**：讨论围绕 **GPT-5** 何时发布以及它是否会决定 **Google** 作为配角的命运展开，一些人预测 **Grok 3.5** 或 **GPT-5** 将统治一段时间。
   - 用户讨论了支付 **ChatGPT** 费用的用户是否会注意到 **GPT-5**，并暗示这可能只会持续 *几个月甚至几周*。
- **Gemini 2.5 Pro 在编程方面的实力**：一位用户报告说 **Gemini 2.5 Pro** 在 pygame 编程方面比 **o4** 更好。
   - 另一位用户分享了一个 [ChatGPT conversation](https://chatgpt.com/share/684e45eb-f118-8003-804e-3c9b562caab9)，其中 **2.5 Pro** 在被告知之前的答案错误后，正确回答了一个逻辑问题。
- **LMArena Checkpoint 刷榜指控频发**：LMArena 的一些用户认为，大型科技公司由于 Checkpoint 滥发以及获得更多 RLHF 数据的机会而占据优势。
   - 一位用户表示，LMArena *基本上是美国大科技公司的排行榜*，开源模型或国外模型要么不出现，要么出现得极晚。
- **Minimax 发布开源推理模型**：Minimax 发布了一个新的开源大语言推理模型 [MiniMax-M1-80k](https://huggingface.co/MiniMaxAI/MiniMax-M1-80k)，但早期基准测试显示其表现不如 **o3** 和 **Gemini 2.5 Pro**。
   - 一些用户反应消极，称 *这要么是侥幸，要么是他们搞砸了，或者它只具备中文能力*。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1384197635946315898)** (1 条消息): 

> `Models Erroring Out, Models not responding, Model API Issues` 


- **模型报错已修复！**：团队确认了一个导致模型报错而非响应的普遍问题，并承诺快速修复。
   - 该问题现已解决；鼓励用户报告任何持续存在的问题。
- **问题已解决，模型恢复运行**：团队确认导致模型报错的普遍问题已得到解决。
   - 建议用户在修复后报告任何进一步的问题或持续存在的情况。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1384288178181115994)** (1 条消息): 

> `ChatGPT Image Generation, WhatsApp Integration` 


- **ChatGPT 图像功能进驻 WhatsApp！**：**ChatGPT image generation** 现已通过 [1-800-ChatGPT](https://wa.me/18002428478) 在 **WhatsApp** 中可用。
- **拨打 DALL-E：WhatsApp 号码上线**：**WhatsApp** 上的 **1-800-ChatGPT** 号码现已启用，为所有用户提供图像生成功能。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1383195125957656698)** (957 条消息 🔥🔥🔥): 

> `AI vs Human, GPT Plus Worth It, Sora video generation, Veo 3 video generation, GPT Model Performance` 


- **识别 AI：艺术越精细，问题越多**：一些成员讨论了当人们认为自己能分辨 **AI-generated** 和 **human-created art** 时是多么容易被愚弄，尤其是当复杂度和细节增加时。
   - 他们将其比作假币，即*细节越多，审查就越严格*。
- **GPT Plus：校园神器？**：成员们辩论了 **GPT Plus** 每月 **$20** 的订阅费对于学校使用是否值得，一些人建议带有 **GPT-4o** 的免费版本可能就足够了。
   - 大家的共识是 **Plus** 提供了更好的模型，如 **o3**、**4.1**、**4.5** 和 **o4 mini high**。
- **Sora：依然是痛点？**：尽管有一位成员使用熟练，但其他人批评 **Sora** *非常糟糕*，尤其是与 **Veo 3** 相比；但另一位成员表示喜欢它，因为它感觉更具*创意调优*且细节丰富。
   - **Veo 3** 的一个关键优势是其音频处理能力。
- **Veo 3 大放异彩**：成员们赞扬了 **Veo 3** 生成受版权保护内容（如 **Star Wars**）的能力，其中一人表示，使用 **Veo** *你可以存储一个稳固的参考帧或风格掩码，并在每次迭代中反馈，然后回到 V3 进行最终润色，以保持画面稳定*。
   - 尽管如此，**Sora** 仍被视为“一站式商店”，具有很高的价值。
- **性能差异：模型之乱**：成员们发现 **GPT-4o** 在某些任务中经常优于 **4.1**，其中一人指出：*4o 的准确率是 10/10，而 4.1 只有 3-4/10*。
   - 据观察，在 Prompt 中删除空格可以显著提高 **4.1** 的准确性。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1383168931564355644)** (86 messages🔥🔥): 

> `GPT-4o 的记忆、微调 GPT 模型、自定义 GPT 模型选择、DALL-E 3 移除、Canvas 自动更新` 


- **GPT-4o 访问了之前的聊天数据？**：一位用户描述了 **GPT-4o** *逐字逐句*引用了在另一个**自定义 GPT** 聊天中共同创作的场景，引发了关于跨聊天记忆访问的猜测。
   - 虽然有人认为这是准确的推理，但该用户辩称统计学上的极低概率暗示并非如此，并邀请他人通过私信查看对话日志。
- **Mini 对比 Nano：选择合适的微调模型**：一位用户询问在 **4.1 mini** 或 **nano** 之间，哪种模型更适合通过 Fine-tuning 来模仿写作风格。
   - 另一位成员建议从几百个 **100-200 词**的示例开始，并指出超过 25 万词后收益会递减，但该用户愿意花费 15 美元对数百万词的内容进行训练。
- **自定义 GPT 模型选项扩展**：用户注意到自定义 GPT 现在支持更广泛的模型，包括 **GPT-4o**、**o3** 和 **o4-mini**。
   - 一位用户发现自定义 GPT 中的 **RAG** 优于 Projects 中的表现，并引用了 2025 年 6 月 12 日的发布说明，其中详细介绍了扩展的模型支持。
- **DALL-E 3 图像生成被禁用？**：成员们报告 ChatGPT 中的 **DALL-E 3 图像生成**可能已被禁用，并哀叹新的原生图像生成器质量较差。
   - 一位刚续订订阅的用户表示沮丧，希望 OpenAI 能保留原始的 DALL-E 3。
- **Canvas 自动更新到上一个 Canvas？**：一位用户正在寻求如何让 ChatGPT 访问并更新正确的 **Canvas** 的方法，并描述了一个问题：ChatGPT 会自动更新你创建的最后一个 Canvas，而不是第一个。
   - 另一位成员提出通过私信帮助排查 **Canvas** 问题，并表示愿意尝试重现该问题并寻求修复方案。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1383160731045986376)** (135 messages🔥🔥): 

> `Pandoc、HTML 解析、Sora AI 提示词、O3 模型提示词、GPT 连贯性` 


- **推崇 Pandoc 进行精细化解析**：一位成员建议使用 [Pandoc](https://pandoc.org/) 将 HTML 转换为 Markdown，强调它是一个专门构建的工具，而不是使用 *awk* 或其他脚本工具。
   - 另一位成员表示同意，认为使用支持良好的开源工具来解决问题更好。
- **清理策略可节省 Token**：成员们讨论了 HTML 标签如何产生*噪声 Token*，虽然这些 Token 可能对推理有益，但会增加 AI 流水线中的 Token 使用量和成本。
   - 一位成员指出，虽然对于单个网站查询的语义差异微乎其微，但在大规模应用时会积少成多。
- **渴望 O3 生成长篇回复**：一位成员就如何提示 **O3** 和 **O3-pro** 生成长篇回复（而非简洁的要点总结）寻求建议。
   - 他们注意到其他模型如 **Sonnet** 和 **Opus 4** 没有同样的问题。
- **Sora 的风格之争：DALL-E 还是代码式提示词**：一位成员询问在 **Sora AI** 上进行图像生成时，是 **DALL-E** 风格的提示词更好，还是代码风格的提示词更好。
   - 该用户的用例涉及解析和理解网页中复杂的学术研究论文，并将研究成果应用于正在进行的辩论中。
- **聊天机器人的创意连贯性危机**：成员们讨论了故意让 **ChatGPT** 失去连贯性的方法，建议包括荒诞化、过度使用比喻、过多的专业术语、失控的角色定义以及矛盾的指南。
   - 一位成员推荐使用 **巴勒斯剪贴技术 (Burroughs' cut-up technique)** 来使上下文对角化，从而使输出呈现出梦幻感。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1383160731045986376)** (135 条消息🔥🔥): 

> `Pandoc vs awk 解析对比，HTML 噪声 Token，O3 的长篇回复，用于图像生成的 Sora AI 提示词，GPT 连贯性丧失` 


- **Pandoc 展现解析威力！**：成员们讨论了使用 **Pandoc** 而非 **awk** 将 HTML 转换为 Markdown，强调了其专为该用途设计的特性和广泛的应用。
   - 一位成员强调了在解析时使用合适工具的重要性，建议在处理复杂任务时，应选择像 **Pandoc** 这样功能强大的“瑞士军用电锯”，而非基础工具。
- **HTML “噪声 Token” 可能有助于推理**：有观点提到，HTML 标签虽然看起来像噪声，但在某些 AI 应用中（尤其是大规模应用时）实际上对推理有益。
   - 一位成员指出，虽然标签带来的 Token 增量很小，但在大规模操作中会累积起来，从而增加有价值的上下文。
- **优化 O3 的长篇输出**：一位用户请求能够引导 **O3** 和 **O3-pro** 在审查文件或进行深入研究时给出**长篇回复**的提示词，因为这些模型往往倾向于简洁并偏好使用列表点。
   - 该用户提到，在使用 **Sonnet** 和 **Opus 4** 审查文件时并未遇到类似问题。
- **探索 ChatGPT 的连贯性难题**：成员们讨论了故意诱导 ChatGPT **丧失连贯性**的方法，包括荒诞化、过度使用比喻、术语堆砌、失控的人设以及矛盾的指南。
   - 还建议使用 **Burroughs 的剪贴法 (cut-up method)**、ADHD 思维螺旋和快速语速等技巧来对角化上下文并破坏连贯的输出。
- **UPSUM 前来救援：保存对话上下文**：一位成员分享了一个名为 **UPSUM Chain Prompt** 的**元提示词 (meta-prompt)**，用于生成更新后的摘要，以实现无缝的对话衔接。
   - 会议强调 LLM 可能无法保留完整的对话历史，因此需要使用更短的对话以及 Chain of Density 和 UPSUM 等摘要技术来有效地管理和保存上下文。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1383164062417948895)** (575 条消息🔥🔥🔥): 

> `Unsloth 的 GPU 检测问题，Unsloth-DeepSeek-R1-0528-UD-IQ2_M 基准测试结果，Hugging Face 模型下载问题，Unsloth 微调 Notebook，Unsloth 与 AMD 的兼容性` 


- **Unsloth 对新模型 Unsloth-DeepSeek-R1-0528-UD-IQ2_M 进行基准测试**：新的 **Unsloth-DeepSeek-R1-0528-UD-IQ2_M** 模型在测试用例中达到了 **69.4%** 的准确率，速度为 **426.0 秒/用例**，而 API 为 **716.6 秒**，但一位成员担心在最终结果出来前存在过度炒作。
   - 该模型在 **65k 上下文**下加载大约需要 **240GB** 显存，相比 FP8 所需的 **7-800GB**，这使得本地运行更加可行，该成员认为这是一个重大进步。
- **Hugging Face 命名问题**：用户讨论了 Hugging Face 缓存文件夹中的命名规范问题，其中大小写差异会导致[重复下载](https://huggingface.co)，浪费空间。
   - 当使用 Unsloth 下载模型后再使用 Hugging Face 时可能会触发此问题，由于作者可能更改了仓库，导致命名规范不一致。
- **分享微调技巧**：成员们建议初学者从 **3B-8B** 等较小的模型开始，并强调数据集的*质量优于数量*。
   - 他们还分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=jFl5Fewrieo)，建议新用户将 **80%** 的时间花在数据上。
- **Unsloth 的 AMD 兼容性接近完成**：据报道，Unsloth 已接近完全兼容 AMD GPU，因为 Unsloth 的大部分算子 (kernel) 都是用 **Triton** 编写的。
   - 一位成员指出，虽然 Triton 可以编译到 AMD GPU，但 Triton 的设置可能需要针对 AMD 进行优化，这可能会影响性能，并指向了[这个 PR](https://github.com/unslothai/unsloth/pull/2520)。
- **告别 Reddit，你好 X**：用户对 Reddit 表示不满，原因包括糟糕的自动审核系统、缺乏对帖子的控制以及普遍存在的偏见审核。
   - 一位用户以此为由注销了 Reddit 账号，并建议将 [Twitter](https://twitter.com) (X) 作为博客、变现和获取新闻的更好替代方案，强调 X 没有机器人，只是一个社交平台，因此应避免仇恨和政治。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1383807532275077171)** (13 条消息🔥): 

> `KL Divergence 突刺, Google Colab GPU 定价, TempleOS, Hugging Face 故障` 


- **KL Divergence 随机爆炸，随后恢复正常**：一位成员询问关于 **KL divergence** 有时会在单步中飙升至 **x10000** 然后恢复正常的情况，这种行为似乎不影响训练。
   - 另一位成员提到这种情况经常发生，甚至在没有使用 Unsloth 的 Hugging Face 运行中也会出现，可能是由于在 **acting and reference policies 之间的 logprob 减法和指数运算** 过程中某些特定的 token 值爆炸导致的。
- **Google Colab GPU 价格的最佳平衡点**：一位成员询问在 **Google Colab 上进行 fine-tuning 的 GPU 价格** *最佳平衡点* 在哪里，考虑了速度与点数消耗之间的平衡。
- **闲聊 TempleOS**：一位成员询问是否还有其他人喜欢 **TempleOS**。
- **Hugging Face 似乎宕机了**：成员们报告了 **Hugging Face** 宕机的消息，并分享了一张[图片](https://cdn.discordapp.com/attachments/1179039861576056922/1384238073264476252/image.png?ex=6851b3aa&is=6850622a&hm=b46b4e20fbde30128158ce97f3652c6d3e6462e4ffd91be145958bfa03a6afc4)，描绘了因故障而不得不“拥抱自己（hug your own face）”的心情，并附带了一个[相关 GIF 的链接](https://tenor.com/view/sad-gif-17629775327580673254)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1383161455225868308)** (266 条消息🔥🔥): 

> `Qwen2.5 vs Qwen3, GGUF 转换, DPO vs SFT, Gemma3, Llama 3.2` 


- **集成编程助手的 Qwen2.5 或 Qwen3**：一位成员需要一种快速的方法将编程助手与 **R/RStudio** 集成，并询问了 **Qwen2.5 Coder 32B Instruct GGUF**，但被建议使用来自 Unsloth 的 **Qwen3** 包，可以在[这里](https://huggingface.co/unsloth/Qwen3-30B-A3B-128K-GGUF)获取。
   - 该成员计划基于 **Qwen 3** 创建一个新模型，将 *no_think* 作为默认设置以嵌入其工作流，而不是使用 instruct 模型。
- **SFT 还是 DPO**：**DPO** 更适合控制模型的响应 *方式*，而如果你想让模型回答特定信息（如模型名称），带有一些 upsampling 的 **SFT** 则是更好的选择。
   - 这是针对“如果模型在被问到‘你叫什么名字？’等问题时需要给出特定回答该怎么办”这一问题的回复。
- **Gemma 3 错误需要修复**：用户报告了在使用 Gemma 3 时出现的 `dtype` 不匹配错误，错误信息为 *expected mat1 and mat2 to have the same dtype, but got: float != c10::Half unslot gemma*，成员们建议这可能与 **bfloat16** 精度或所使用的 GPU 有关。
   - 一位成员目前正在修复遇到的错误，并建议尝试[此链接](https://discord.com/channels/1179035537009545276/1179035537529643040/1381978312640958515)中的安装说明来解决问题，同时建议将默认的 *pip install* 命令替换为仓库中的强制重新安装命令，以获取最新的修复。
- **帮助转换 Llama 3.2**：一位成员询问如何将 **Unsloth 的 Llama-3.2-11B-Vision-Instruct** 模型安装到 `ollama`，并获知可以在[这里](https://huggingface.co/pbatra/Llama-3.2-11B-Vision-Instruct-GGUF/tree/main)找到预制的 GGUF 版本进行手动转换。
   - 一位用户发布了[官方 ollama 指南](https://github.com/ollama/ollama/blob/main/docs/import.md)的链接，用于转换为 GGUF，并建议直接从 [ollama library](https://ollama.com/library/llama3.2-vision:11b) 拉取模型。
- **新修复**：新的修复已经推送，如果你直接从仓库安装更新的代码（而不是通过 pypi），即可使用。
   - 建议直接从主仓库安装可能会解决与重新合并 adapter 相关的问题。提供的链接详细说明了如何在 PC 上安装 Unsloth。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1384192396769755287)** (2 条消息): 

> `arxiv 链接` 


- **AI 论文分享**：一位成员分享了一个 [arxiv 链接](https://arxiv.org/abs/2506.09991)。
- **分享资源确认**：该成员确认该资源在他们自己分享之前就已经被分享了。


  

---

### **Cursor 社区 ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1383163182985515068)** (750 条消息🔥🔥🔥): 

> `Claude 4 Sonnet 性能, Cursor UI 问题, MCP 使用, 代码隐私, Bug 报告` 


- **Claude 4 Sonnet 用户观察到运行缓慢**：成员们注意到，尽管 **Claude 4 Sonnet** 整体表现出色且具有 [稳定性](https://link.to/stability)，但与 GitHub Copilot 等平台相比，它在 Cursor 中的运行速度明显较慢。
   - 一些成员建议通过将 **Max Mode** 留给重大重构任务来 **优化使用**，或者探索使用 **Gemini 2.5 Pro** 进行规划和代码审查等替代方案，以便更好地管理推理额度（inference credits）。
- **用户抱怨 Cursor 的 UI 缺陷**：用户报告了持续存在的 UI 问题，例如由于 *xterm bracketed paste* 问题导致 Windows 上的 **命令执行失败**，以及命令完成通知不一致，导致推理额度被浪费。
   - 一位成员指出，由于这些 UI 故障，大约 **10-15% 的额度被浪费了**，并建议 Cursor 在发生错误时返还推理次数。
- **在 Cursor 中探索 Model Context Protocol 使用**：一位成员就使用 **Model Context Protocol (MCP)** 寻求建议，并强调了 AI 能够利用截图并自动集成错误消息的优势。
   - 另一位用户强调了花时间 **定义更好的 Prompt** 的重要性，认为这比频繁截图和复制粘贴更有效，并建议使用 [Wisprflow](https://wisprflow.ai) 来增强语音转文字能力。
- **用户请求细粒度的代码隐私设置**：由于对代码存储和访问权限的担忧，用户表示需要 **按仓库（per-repository）** 设置代码隐私，允许为工作和个人项目配置不同的设置。
   - 目前，Cursor 的 **Privacy Mode** 是一个 *全局设置*，但社区希望在项目级别实现更细粒度的控制，以增强灵活性和安全性，因为他们希望避免无意中打开敏感的公司目录。
- **通过主动监控简化 Bug 报告**：成员们正在社区内积极分享 Bug 报告和故障排除技巧，特别关注 Windows 上损坏的 **命令执行工具** 等问题。
   - 在一位成员注意到 **Cursor Task Master**（实际上是一个尚未正式发布的社区第三方项目）后，社区正在推动在 **Windows 上进行更积极的测试**，并要求 Cursor 团队就 Bug 修复和功能发布进行更好的沟通。

---

### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1383273362230345919)** (40 条消息🔥): 

> `GitHub integration, Background Agents Permissions, Background agents and Slack, Background Agents and Privacy Mode, Toggle bug with background agents` 


- ****GitHub Integration 路线图****：一位用户询问 Agent 是否可以访问 **GitHub issues**，成员回复称目前尚不可能，但已在路线图（roadmap）中。
   - 该成员还澄清说，**GitHub integration** 的改进即将推出。
- ****Background Agents 缺少创建 PR 的权限****：一位用户报告说，尽管在 Cursor integration 设置中授予了所有权限，但 Slack 中的 Background Agents 仍无法创建 pull requests，并生成了请求 ID：**bc-79e56de2-26d7-41a0-a5b3-b8b9e9f635d1** 和 **bc-d3acc5d6-2520-413f-9f7c-2fb36590215d**。
   - 一位成员提出协助调试该问题，并索要了请求 ID。
- ****Background Agents 需要新的 Privacy Mode****：一位用户注意到旧的 Privacy mode 现在被标记为 "legacy"，并询问启用 Background Agents 是否需要新的 Privacy mode（包含代码存储）。
   - 一位成员确认 **Background Agents** 需要支持代码存储的新 Privacy mode，因为原始的 Privacy mode 不允许在 Background Agent 的生命周期内存储代码，而这对于执行和迭代代码是必需的。
- ****Background Agents 开关 Bug 报告****：一位用户报告了 Background Agents 的开关（toggle）Bug，并提供了一段 [视频](https://cdn.discordapp.com/attachments/1367213641027551352/1383486605423022311/CleanShot_2025-06-14_at_18.40.43.mp4?ex=68519ace&is=6850494e&hm=cf854daed7b862bbf53d10101cd431f5bcac0069bbfc234f0240749fbda7ddfa&) 演示该问题。
   - 一位成员作出了回应，表示将进行调查，并要求用户检查其想要连接账户的 [GitHub installations](https://github.com/settings/installations/)。
- ****Cursor 未列在已安装的 GitHub Apps 中****：一位用户发现 Cursor 被列在 "Authorized GitHub Apps" 下，但**未**列在其个人组织（org）的 "Installed GitHub Apps" 下；而在另一个 Background Agents 正常工作的组织中，Cursor 被列为已安装并拥有所有 repo 的访问权限。
   - 用户被引导至 [Cursor 的 dashboard](https://www.cursor.com/dashboard?tab=background-agents)，通过 GitHub 的 "Manage" 外部链接重新配置/启用/禁用 repo 和 org，以解决此问题。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1383159564391682129)** (553 条消息🔥🔥🔥): 

> `AI-Generated Feedback, Open Sourcing AGI, Bigram Testing, Qwen 2.5, HF Pro Disk Space` 


- **对 AI 生成内容的伦理担忧**：一位成员对 [AI 生成的反馈](https://example.com) 表示不安，理由是出于伦理原因不参与 AI 生成的图像或视频工作。
   - 该成员表示：*“即使我有理论知识，我也不会参与 AI 生成的图像/视频工作”*。
- **开源 AGI 辩论升温**：一位成员幽默地宣称，如果他们创造了世界上第一个 **AGI**，将会[将其开源](https://example.com)。
   - 这引发了关于开源如此强大的技术的潜在收益和风险的讨论。
- **Qwen 2.5 的效率令人印象深刻**：一位成员称赞了 **Qwen 2.5**（在 Ollama 上量化为 q4 的 7b 模型），认为考虑到其较小的体积和简单的 system prompt，其性能令人印象深刻；另一位成员指出：*“在 benchmark 对比中没人展示 Qwen 2.5，因为它太优秀了”*。
   - 讨论还提到 **Qwen 模型** 是在包括中文和英文在内的多语言数据集上进行预训练的。
- **Rate Limiting 与 Zero GPU Quota 异常**：用户报告了 Hugging Face 上的 **rate limiting** 问题，一些人报告获得了 **额外的 Zero GPU Quota**。
   - 有推测认为额外的 GPU Quota 可能是针对老用户的特殊规定，但尚未发布官方公告。
- **AI 辅助编程受到关注**：成员们讨论了他们使用 Gemini 等 **AI 辅助编程工具** 的经验，赞扬了它们生成易于理解且可修改代码的能力。
   - 一位成员分享道，他们 *“第一次为 iOS 进行 vibe coded，完全不知道它是如何运作的，现在依然不知道……但它确实实现了预期的功能”*。


  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1383864388615802960)** (2 messages): 

> `HF audio course, Agents course, MCP course` 


- **用户开始学习 HF audio course**：一位新成员宣布他们今天开始学习 **Hugging Face audio course**。
   - 未提及链接或资源。
- **成员学习 Agents 和 MCP 课程**：一位成员目前正在学习 **Agents course 的 Unit 2**，并已开始学习 **MCP course 的 Unit 1**。
   - 未提及链接或资源，仅包含一个 <:hugging_rocket:968127385864134656>。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/)** (1 messages): 

cakiki: <@844851718512443423> 请不要发布推荐链接 (referrals)。
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1383234217021800631)** (9 messages🔥): 

> `peft-bench, InfiniGPT French Q&A dataset, Shisa AI Japanese model, Swiftide Rust library for agentic RAG applications, QuantIntelli Football Betting Analysis` 


- ****InfiniGPT** - 最大的法语 Q&A 数据集发布！**：一位 19 岁的学生发布了 **InfiniGPT**，这是一个法语 Q&A 数据集，包含 **40,000+ 条 Q&A** 条目，100% 原生法语，经过人工验证，涵盖多样化主题，且已准备好进行 fine-tuning，旨在建立法语数据集标准 ([GitHub](https://github.com/RDTvlokip/InfiniQA), [HuggingFace](https://huggingface.co/datasets/RDTvlokip/InfiniQA))。
   - 作者声称其规模是 *FQuAD 的 5 倍*，并提供*直接的 Q&A* 而非抽取式阅读，附带文档来源，并针对 GPT-2 tokenizer 进行了优化。
- ****Shisa v2** - 日本最强模型发布！**：一个 2 人团队发布了 **Shisa v2**，这是日本有史以来训练的最强模型，随附一份概览报告（技术报告即将发布），可在 [HuggingFace](https://huggingface.co/shisa-ai/shisa-v2-llama3.1-405b) 获取（800GB+）。
   - 他们还更新了核心 SFT 数据集（可在 [HuggingFace](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt) 获取），声称该数据集在不降低英语性能的情况下提升了日语性能，基于 **7B-405B** 的 SOTA 开源模型进行训练和发布。
- ****Swiftide 0.27** - Rust 库发布！**：**Swiftide** 发布了一个主要版本，这是一个用于构建可组合的 agentic 和 RAG 应用的 Rust 开源库 ([公告](https://bosun.ai/posts/swiftide-0-27/))。
- ****QuantIntelli** - 混合 AI Agent 预测足球！**：创建了一个用于定量足球博彩分析的混合 AI Agent，结合了 **XGBoost** 模型和 **Google Gemini LLM**，具有使用 Tavily、Google 和 DuckDuckGo 的高级 RAG Pipeline、使用 Supabase 的持久会话日志以及使用 Gradio 的交互式 UI ([HuggingFace Space](https://huggingface.co/spaces/ChillThrills/QuantIntelli), [Github Repo](https://github.com/IanDublew/QuantIntelli))。
- ****JASCO** - MCP server 上的音乐生成！**：用户现在可以通过 MCP server 使用 **facebook/jasco** 生成音乐分轨 (musical stems)，该工具根据文本描述、和弦进行以及可选的旋律和鼓点输入生成两种变化的音乐 ([HuggingFace Space](https://huggingface.co/spaces/Tonic/audiocraft))。
   - 现在无需使用麦克风录制输入音频，可以通过 **stable-audio-open-small** 在约 1 秒内生成鼓点输出供 gary 继续创作，并*将其命名为 jerry lol*。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1383861459695435776)** (2 messages): 

> `Portfolio Theory, Dr. Peter Cotton, Schur Portfolios` 


- **Cotton 提议关于 Portfolio Theory 的演示**：一位成员建议安排 **Dr. Peter Cotton** 演示他关于投资组合理论 (portfolio theory) 的论文，并链接到了 [Schur Portfolios 论文](https://github.com/microprediction/home/blob/main/papers/schur_portfolios.pdf)。
   - 他们询问了组织此类演示的流程。
- **提议 Schur Portfolios 论文演示**：一位成员提议就 **Dr. Peter Cotton** 的论文“[Schur Portfolios](https://github.com/microprediction/home/blob/main/papers/schur_portfolios.pdf)”进行演示，重点关注投资组合理论。
   - 该提议包括请求关于组织演示的指导。


  

---

### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1383706186393587798)** (3 messages): 

> `Smolagents, Ollama, Code Agents, 本地模型选择` 


- **Smolagents 与 Ollama 失去兼容性**：一位成员报告称，**smolagents** 在本地 **code agents** 方面不再兼容 **Ollama**。
   - 该成员正在寻求实现本地 code agent 的帮助。
- **有限资源下的模型推荐请求**：一位成员请求推荐在 **8GB RAM** 和 **6GB VRAM** 环境下本地运行的最佳模型。
   - 他们在期末项目中使用 **smolagents**，花费了 **$10** 的 OpenAI API 费用，达到了 **45%** 的准确率；[项目链接](https://huggingface.co/spaces/renwei2024/agent-course-final-project/tree/main)。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1383228313211899914)** (21 messages🔥): 

> `HF Inference 成本, 使用 Ollama 的本地 LLM, 第 3 单元作业, 本地 Agentic RAG, 未授权的导入` 


- **用户苦于 HF Inference 成本**：多位用户在 **HF Inference** 上遇到了成本问题，特别是在使用 **llama-3-70b-Instruct** 等模型时，发现免费额度不足。
   - 一位用户报告在多次尝试最后一个单元后支付了约 **$6**，并建议使用本地模型来降低成本。
- **Ollama 支持本地运行 LLM**：成员们正在讨论使用 [**Ollama**](https://ollama.com/) 在本地运行 **LLM**，然后将其接入 agent，从而减少对付费推理 API 的依赖。
   - 成本节省可能非常显著，但一位用户认为最后一个单元的作业太具挑战性，学习曲线过于陡峭。
- **对第 3 单元作业的反馈**：一位用户表示最后一个单元感觉像是被扔进了“深水区”，希望整个课程中能有更多类似的作业。
   - 他们还注意到许多排行榜上的提交内容似乎是抄袭的，这破坏了练习的目的。
- **本地调试 Agentic RAG**：一位用户在尝试本地运行 [Unit_3_Agentic_RAG](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG) 时遇到错误，并发布了错误消息的截图。
   - 讨论中未提供具体解决方案，但问题似乎与环境配置有关。
- **离奇的未授权导入问题**：一位用户报告 **CodeAgent** 将某些导入（如 **plotly.express**）标记为未授权，即使已将 **plotly** 指定为授权导入。
   - 另一位用户确认了类似经历，指出有时使用别名（例如用 **bs4** 代替 **beautifulsoup4**）可以绕过限制，同时确认添加 `plotly.express` 解决了该用户的问题。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 messages): 

Readybot.io: **OpenRouter - 新模型**
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1383513332539199689)** (3 messages): 

> `象棋排行榜, 书籍测试` 


- **象棋排行榜回放功能**：一位成员为 [chess-leaderboard](https://dubesor.de/chess/chess-leaderboard) 的每场对局（过去和未来）添加了自制的回放功能。
   - 他们提到这“可能比 lichess 的 gif 好一点，但用我的技术栈实现起来很痛苦”，并附带了 [chessreplay.gif](https://cdn.discordapp.com/attachments/1092850552192368710/1383513332916424715/chessreplay.gif?ex=6851b3b3&is=68506233&hm=1039e8ba4df4c19b19fc9c28f054c1eb809659a82c942b46aa7daebc3b48088c&)。
- **书籍测试机会**：一位成员提到他们的书已于 **6 月 15 日** 上线，并愿意协助测试。
   - 他们表示如果有人给他们发私信，他们“很乐意提供帮助”。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1383178098874519672)** (533 条消息🔥🔥🔥): 

> `OpenRouter Discord 标签请求、Claude Prompt 调试、GPT-4.1 Mini 供应、免费模型额度使用、多语言模型推荐` 


- ****Discord 标签请求未获回应****：一名成员对恢复 **OpenRouter Discord 标签** 的请求未获理睬表示沮丧，尽管 **OpenRouter** 财力雄厚，他仍提出愿意为此付费。
   - 由于一直没有得到回复，他们开玩笑地威胁要直接艾特 **Alex Atallah**。
- ****Claude 和 Copilot 的 Prompt 缺乏 Token 经济性****：一名成员调试了来自 **Claude Code** 和 **GitHub Copilot** 的 Prompt，发现它们在改进工作流时经常忽略 Token 效率，除非冗长程度影响了性能，否则会发送无关内容。
   - 他们观察到，在调整 Prompt 时，简洁性并不是这些系统的主要目标。
- ****在 OpenRouter 上测试 GPT-4.1 mini 版本****：一名成员提供了 **GPT-4.1 mini** 的访问权限，拥有 **200K tokens/minute** 的速率，价格仅为官方 Token 价格的 **20%**，兼容 OpenAI SDK，并邀请高用量测试者私聊了解详情。
   - 他们强调该版本非常适合 Cline.bot 等应用以及 BYOK/BYOB 设置。
- ****Deepseek 免费层级遭遇停机****：用户报告称，通过 API 使用免费版 **Deepseek-r1-0528** 时遇到了 **502、503 和 524 错误**，有人猜测这些问题可能源于*色情 RP* 带来的高流量。
   - 成员们注意到付费版本仍可正常运行，并讨论了潜在原因，包括数据中心问题或 **Chutes** 的故障。
- ****OpenAI 因与 Microsoft 的纠纷面临反垄断威胁****：讨论透露，**OpenAI** 高管曾考虑在合作期间指控 **Microsoft** 存在**反竞争行为**，可能寻求监管审查并启动公开宣传活动。
   - 这源于艰难的谈判，引发了社区成员的惊讶和担忧。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1383164338155688089)** (247 条消息🔥🔥): 

> `TokenBreak 攻击、AMD Ryzen AI 迷你电脑、在 LM Studio 中增加 RAG 大小、MiMo VL 7B RL UD 支持、LLM 图像整理器` 


- **TokenBreak 攻击绕过 AI**：一位成员分享了一篇[关于新型 TokenBreak 攻击的文章](https://thehackernews.com/2025/06/new-tokenbreak-attack-bypasses-ai.html)，该攻击可以绕过 AI 安全措施，尽管他们在实验中没有得到相同的结果。
   - 另一位成员开玩笑地指出，附带截图中的图标是多么相似。
- **AMD Ryzen AI 迷你电脑运行大模型**：一位成员提到 [AMD Ryzen™ AI Max+ 395 --EVO-X2 AI Mini PC](https://share.google/3WVLR8Up7pjVjq35mI) 可以流畅运行一些大模型。
   - 其他人反驳说它只是一个由 HIP SDK 支持的*加强版核显（igpu）*，虽然运行 **70B** 模型速度约为 **5t/s**，但运行 **Qwen3 235B A22B** 模型速度可达约 **14t/s**。
- **LM Studio 无法增加 RAG 大小**：一位成员询问如何在 LM Studio 中将 **RAG 大小** 从 **31.46 MB** 增加到 **100 MB**。
   - 另一位成员回答说*这是不可能的*，目前的 RAG 仍处于基础实现阶段。
- **LLM 冗长详细回复的解决方案：叙事游戏**：一位成员建议与你的 LLM 开始一场*文字冒险游戏（choose-your-own-adventure game）*，以确保*它不会尝试在单次回复中完成整个故事*。
   - 他们建议使用如下 Prompt：`Let's play a choose-your-own-adventure game. I'll start with a prompt and you carry the story on. When you reach a decision point, list a few choices for direction and I'll respond.`
- **本地 LAN 端口开放风险较低**：一位成员询问了开放端口的安全风险，这引发了对*将该站点作为前端*（暗示将后端暴露给互联网并产生安全漏洞）的担忧。
   - 最终，成员们一致认为在**本地 LAN 网络**上开放端口风险较低，尽管任何开放端口*都可能被利用*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1383166191920484522)** (95 条消息🔥🔥): 

> `GMKtec Windows 安装问题, RTX 6000 Pro 功耗配置, 显卡与电感啸叫, NVLINK 性能实验, 针对 LLM 的 GPU 推荐` 


- **在 GMKtec 上安装 Windows 极其麻烦**: 有用户报告在 **GMKtec** 机器上安装 **Windows** 时遇到问题，理由是安装失败以及使用 **Rufus** 创建的可移动介质存在问题。
   - 他们正尝试在 GMKtec 机器上安装 Windows。
- **RTX 6000 Pro 可配置为 300W 或 600W**: 据一位成员称，**RTX 6000 Pro** 可以配置为 **300W** 或 **600W**。
   - 标准版是可以配置的，但不确定此配置中具体使用的是哪一种。
- **显卡饱受电感啸叫困扰**: 用户观察到，与运行游戏相比，运行 **LLM** 时显卡的 **电感啸叫 (coil whine)** 显著增加。
   - 一位用户指出 **5090** 的电感啸叫更严重，并建议通过 **undervolting**（降压）作为减少功耗和电感啸叫的解决方案。
- **NVLINK 性能仍未得到广泛测试**: 一位成员询问了关于 **NVLINK 干扰性能差异** 的实验数据，想知道它是否能带来切实的收益。
   - 另一位成员发布了一张图片，上面写着“我也确定 NVIDIA 的软件针对 NVLINK 进行了高度优化”。
- **GPU 购物清单：3090-4090-5090，价格不菲**: 为了运行 **Qwen3**、**Devstral** 和 **Gemma3** 等模型，推荐使用 **3090**、**4090** 和 **5090**，因为它们拥有 **24-32GB** 的 VRAM，特别是对于大型模型或更高质量的 **quants**（量化）而言。
   - 3090 与 4090 性能相当，并且在 24GB 显存方面能跟上 5000 系列显卡。5090 的价格约为 3000 美元。花这个价钱，你可能仍然只能运行 32B 或更小的模型，但可以使用更高质量的量化或更多的 **context**（上下文）。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1383246229907177592)** (6 条消息): 

> `PD 解耦, Agent 的 Transformer 时刻, Groq 速度, Groq Huggingface` 


- **PD 解耦资源**: 一位成员请求除了 [DistServe 论文](https://arxiv.org/pdf/2401.09670) 之外，关于 **PD (power distribution) disaggregation**（功耗分配解耦）的资源。
   - 提供的消息中未提供相关资源。
- **Agent 的 Transformer 时刻**: 一位成员询问关于 Agent 的 **"Transformer moment"**，寻求一种能够自动适应任何任务的通用控制策略。
   - 他们想知道是否可以是 **DFS**、**BFS** 或 **混合流**——并由系统自动选择。
- **Groq 的速度以及与 Hugging Face 的合作**: 一位成员询问 **Groq** 有多出色以及为什么它们速度这么快。
   - 另一位成员提到 **Groq** 最近与 **Hugging Face** 展开了合作，暗示了其良好的性能或易用性；未提供明确链接。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1383293498358235268)** (9 条消息🔥): 

> `tl.constexpr 在表达式中的行为, Triton 中的线程级控制流, Triton kernel 预热时间对比 torch.compile, 单行 Softmax kernel 实现` 


- **tl.constexpr 表达式问题！**: 用户发现，在表达式中使用 `tl.constexpr`（例如 `a = tl.constexpr(b // 2)`）会导致 `tl.arange` 出错，因为它无法将 `a` 识别为 constexpr，而 `tl.arange(0, b // 2)` 却能正常工作；解决方案是将其类型定义为 `a: tl.constexpr = b // 2`。
   - 用户提供了一个[最小复现示例](https://github.com/NVIDIA/apex/blob/main/apex/transformer/loggers/csv_logger.py)，展示了当 `tl.arange` 的参数未明确定义为 `tl.constexpr` 时在编译期间发生的错误。
- **线程级循环？**: 用户询问 Triton 中的线程级控制流，试图实现一个对矩阵行求和并在总和超过阈值时停止的循环。
   - 未收到回复。
- **Triton Kernel 启动缓慢？**: 用户报告称，他们手写的 Triton kernel 需要预热时间才能达到峰值性能，这与 `torch.compile` 不同，并想知道 `torch.compile` 是否使用了更好的 block size 启发式算法或其他优化。
   - 未收到回复。
- **Softmax 竞态条件**: 一位正在研究 [单行 Softmax kernel](https://pytorch.org/tutorials/stable_diffusion/stable_diffusion.html) 的用户在最终写入 Softmax 结果的 kernel 中遇到了竞态条件，因为第一个程序覆盖了初始的全局最大值和总和。
   - 未收到回复。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1383205626838257887)** (55 messages🔥🔥): 

> `CUDA 缓存策略，TF32 与 FP16 精度对比，L40 与 4090 性能，nvcc 生成 LDS 指令，GCC 与 NVCC 版本兼容性` 


- **CUDA 缓存策略支持分段 (Fractional)**：一位成员分享了一段 [CUDA 代码片段](https://forums.developer.nvidia.com/t/sm89-cache-policy-hints-not-respected/281749)，展示了如何使用 `createpolicy.fractional.L2::evict_last.L2::evict_unchanged.b64` 创建缓存策略对象，并将其用于带有缓存提示（cache hints）的加载指令中，并询问其具体用法。
- **TF32 精度困扰**：一位成员观察到，在 CUDA GPU（在 4070 mobile 和 A10 上测试）上，**TF32 matmuls 的精度比 float16 低 3 倍**，并分享了一段 [Triton 代码片段](https://github.com/openai/triton) 来重现该问题。
   - 他们指出，[相关的 Triton issue](https://github.com/triton-lang/triton/issues/4574) 中提到的精度问题可能是潜在原因。
- **L40s 表现平平：ECC 是元凶？**：成员们讨论了 **L40s** 在云环境中可能显得表现不佳，这是因为 **ECC** 默认被激活，从而影响了性能。这被认为是一个配置问题，而非硬件本身的问题。
- **nvcc 意外生成 LDS 指令**：一位成员报告称 `nvcc` 为全局内存中的数据生成了非预期的 **LDS** 指令，导致在使用 `compute-sanitizer` 时报错，而使用 `__ldg` 可以修复此问题。
   - 其他人建议这可能是未定义行为，并请求提供一个最小可重现示例，以便进一步调查可能的编译器 Bug。
- **GCC+NVCC 版本组合导致编译失败**：一位初学者遇到了与 `<std_function.h>` 中参数包（parameter packs）未展开相关的错误。建议认为这是由于 **GCC** 和 **NVCC** 版本不兼容导致的，**CUDA 11.7.0** 是第一个正式支持 Ubuntu 22.04 的版本。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1383668795494633482)** (2 messages): 

> `CUDA kernel blocksize 参数，TorchTitan 训练图捕获` 


- **需要 CUDA Kernel Blocksize 参数的最佳实践**：一位成员正在寻求在不使用 JIT 的情况下，将 Python 中的 **blocksizes** 传递给 CUDA 代码的最佳实践，目前他们使用 Torch 的 cpp_extensions + setuptools 来编译自定义 CUDA kernels。
   - 他们正在寻找一种替代方案，以避免在 TORCH_LIBRARY 注册中显式添加 **blocksize** 作为 int[] 参数，因为大多数 PyTorch 函数根本不会暴露 **blocksize** 参数，显式添加显得不够优雅。
- **在训练中使用 Torchtitan 进行图捕获 (Graph Capture)**：一位成员正在使用 Torchtitan 训练 **llama1b**，并希望在处理不同的并行组合时，捕获带有各种 collectives 的训练图。
   - 他们尝试拦截训练步骤并使用 functorch.compile 中的 **aot_module** 进行捕获，但认为 **Faketensor** 传播在其中无法正常工作。


  

---


### **GPU MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1383490434680229959)** (2 messages): 

> `d-Matrix 团队，Dr. Lisa Su，GPU MODE，kernel 数据，Project Popcorn` 


- **d-Matrix 团队将演示定制芯片**：**d-Matrix** 团队将演示其用于**低延迟批处理推理**的定制芯片 (Custom Silicon)。
- **Dr. Lisa Su 点名表扬 GPU MODE**：Dr. Lisa Su 提到了 **GPU MODE** 及其在促成全球首个 10 万美元竞争性 kernel 竞赛中所做的工作，详情见 [gpumode.com/news](https://www.gpumode.com/news)。
- **Kernel 竞赛产生海量 Kernel 数据**：社区生成的 **kernel 数据** 比整个 **GitHub** 上的总和还要多，性能表现超过了人类专家给出的最佳基准。
- **Project Popcorn 合作**：GPU MODE 感谢了 **AMD** 的合作伙伴以及 [Project Popcorn](https://gpu-mode.github.io/popcorn/) 项目。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1383181373870833845)** (1 messages): 

> `指令延迟，arxiv.org` 


- **分享指令延迟相关论文**：一位成员分享了一篇关于指令延迟的论文链接：[https://arxiv.org/pdf/1903.07486](https://arxiv.org/pdf/1903.07486)。
   - 该成员指出，虽然 **指令延迟 (instruction latencies)** 数据可能已经过时，但其中的**讨论**仍然值得一读。
- **关于指令延迟的论文讨论**：位于 [https://arxiv.org/pdf/1903.07486](https://arxiv.org/pdf/1903.07486) 的论文讨论了指令延迟。
   - 尽管指令延迟数据可能过时，但该讨论被认为具有很高的参考价值。


  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1383285235676090460)** (5 messages): 

> `MX-FP4 Matmul, MX-FP8 Matmul, CUTLASS, CuBLAS, FP4 Weight Quality` 


- **CUTLASS 和 CuBLAS 在 5090 上表现出色**: 来自 CUTLASS 的 **MX-FP4 matmul** 和来自 CuBLAS (通过 `torch._scaled_mm()`) 的 **MX-FP8 matmul** 在 5090 上非常令人印象深刻 ([PR 2285](https://github.com/pytorch/ao/pull/2285))。
- **Weight-Only FP4 kernel 尚未提供**: 一位成员询问了关于小 batch (1-64) 的 **fp4 weight-only** 基准测试，目前还没有针对 fp4 的 weight-only kernel。
- **Weight-Only FP4 的高性能即将到来？**: 一位成员表示他们在 **weight-only FP4** 上获得了相当不错的性能，并将尝试抽时间整理代码进行集成。
- **FP4 Weight 质量困扰着一些成员**: **FP4 weights** 的质量让一些人感到困扰，因为转换权重时精度下降明显，因此需要某种量化算法 (quant algo) 来提高精度 (mx-hqq ? 👀 )。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1384163143118487724)** (1 messages): 

> `HQQ Rebrand, Quantum Quantization` 


- **为 Quantum 重塑 HQQ 品牌**: 一位成员建议将 HQQ 更名为 *Half **Quantum** Quantization* 以吸引更多关注。
   - 此前 [Multiverse Computing 融资 2150 万美元](https://multiversecomputing.com/resources/multiverse-computing-raises-usd215m-to-scale-ground-breaking-technology-that-compresses-llms-by) 用于扩展压缩 LLMs 的突破性技术。
- **量子计算资金**: Multiverse Computing 最近获得了 **2150 万美元** 的融资。
   - 该资金旨在扩展其压缩 LLMs 的技术，可能使 HQQ 等项目受益。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1383973154837233786)** (19 messages🔥): 

> `Nvidia to AMD transpilation, AMD stable inference deployment, MI300A architecture, IODs and infinity cache, memory distribution` 


- **Nvidia 到 AMD 的转译工具 CASS**: 一位成员分享了一篇关于 **CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark** 的[论文](https://arxiv.org/pdf/2505.16968)，尝试使用 **GRPO** 而不是 **SFT** 来对模型进行 **RF-ing**。
   - 该成员正在调查性能阈值/差异在实际工作流中是否现实。
- **AMD 稳定推理部署首选 Ollama**: 一位成员正在寻找 **AMD** 上稳定推理/部署库的建议。
   - 看起来 [*Ollama 运行良好*](https://ollama.com/)，所以他们只是想多了。
- **深入探讨 MI300A 的架构**: 成员们讨论了融合的 **AMD CPU-GPU 平台**，特别是 **MI300A** 架构的 **IOD** 和 **infinity cache**。
   - 他们想知道是否有办法测试特定的 Path 或对其中一个 IOD 施加压力。
- **探索内存芯片的分布策略**: 成员们推测内存是如何在内存芯片之间分布的，以及这如何影响延迟，特别是在 **MI300X** 上，每个 **IOD** 连接到 2 个 **HBM stacks**。
   - 一位成员提到使用 `s_getreg` 来确定 shader 运行在哪个 **XCD** 和 **CU** 上，并以此测量到内存中不同位置的访问延迟。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1384226368564367413)** (4 messages): 

> `Thrust library, CUDA kernels, segmented sum algorithm, iterators, high_resolution_clock vs steady_clock` 


- ****Veitner** 介绍 **Thrust** 库**: Simon Veitner 介绍了 [Thrust](https://veitner.bearblog.dev/an-introduction-to-thrust/)，这是一个高级抽象层，允许你使用现代 **C++** 概念编写高性能的 **CUDA kernels**。
- ****High_resolution_clock** 接受基准测试！**: 一位成员建议不要使用 `high_resolution_clock`，而应使用 `steady_clock` 进行基准测试，并引用了[这个 stackoverflow 回答](https://stackoverflow.com/a/37440647/10107454)。
   - 他们补充说，*只要有足够的周期和并行性，这应该不是问题*，然而*即使在那时，真正杀掉性能的其实是跨步/非合并 (strided/uncoalesced) 的内存访问*。
- ****cuTensor** 是更合适的库**: 对于规则尺寸的示例，一位成员建议 **cuTensor** 可能比 Thrust 更合适。
- ****MatX** 使多维算法更优雅**: 一位成员推荐使用 **MatX**，它为这类多维算法提供了优雅的 C++ 接口。


  

---

### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1383958902101508157)** (6 messages): 

> `Tensor Core Algorithm Reformulation, RL for Tensor Core Usage, Kernel Code Verification, GPU Thinking Interpretability` 


- **针对 Tensor Core 的算法重构**：一位用户询问了关于引入领域专家参与的反馈循环，以将算法重构为 Tensor Core 形式的问题，特别是超出 **FFTConv** 等简单案例的情况，并考虑了填充（padding）和秩分解（rank-factorization）等修改。
   - 他们寻求关于如何引导专家进行 Tensor Core 友好型算法设计的指导。
- **强化学习引导 Tensor Core**：一位成员建议使用 **强化学习 (RL)** 来引导模型利用 Tensor Core，通过构建一个小型验证器来检查模型的 trace，以评估其对 Tensor Core 的理解。
   - 他们指出 [Hugging Face Kernels](https://huggingface.co/blog/hello-hf-kernels) 是一个潜在的数据源，并强调了其社区驱动的贡献属性。
- **创新的 Kernel 代码验证思路**：一位用户正在尝试通过 **Triton 解释器**（而非完整执行）来验证 Kernel 代码，以便在数据质量和 RL 尝试中实现更快的验证和更好的扩展性。
   - 这种方法在 CPU 环境中能更容易地洞察内存和指令调用。
- **“GPU 思维”可以具有可解释性**：成员们讨论了将**自然语言可解释性**的方法应用于编程语言，根据 [ICSE-NIER '25](https://www.computer.org/csdl/proceedings-article/icse-nier/2025/371100a086/27t2knJTWso) 发表的论文，为每一层创建探测分类器（probing classifiers）。
   - 目标是展示模型在通过 GPU 方法与 CPU 方法解决问题时，会使用不同的层和 Attention Heads，从而表现出“GPU 思维”，并在初始翻译投影后分析其内部表示。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1384220469351415980)** (4 messages): 

> `ThunderKitten on Older GPUs, TK to AMD port, Attention Kernels with Variable Length` 


- **在 T4 和 P100 等旧款 GPU 上运行 ThunderKitten？**：成员们讨论了在 Kaggle 上提供的 **T4** 和 **P100** 等旧款 GPU 上运行 **ThunderKitten** 的可能性，指出尽管存在异步指令和较小共享内存（shared memory）的挑战，但这很可能是可行的。
   - 一位成员建议使用 **4090 TARGET** 进行编译，并报告任何损坏情况以帮助改进兼容性。
- **TK 到 AMD 的移植：即将推出！**：团队正在积极开发 **TK 到 AMD 的移植**，旨在近期发布以扩大兼容性。
   - 缺乏异步指令通常有点令人烦恼，且共享内存较小，因此与 Nvidia 的 megakernels 相比，我们需要在寄存器层面进行更多的流水线化（pipelining）。
- **ThunderKitten Attention Kernel 支持变长序列**：ThunderKitten 仓库包含了支持**变长（variable length）**和**填充（padding）**的 Attention Kernel，这在各种序列处理任务中非常有用。


  

---


### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1384266860664590529)** (1 messages): 

> `Chain of Thought, CoT, Symbolic Reasoning, Math Reasoning` 


- **新研究精准定位 CoT 的益处**：最近的一篇论文 *To CoT or not to CoT? Chain-of-thought helps mainly on math and symbolic reasoning* ([arxiv 链接](https://arxiv.org/abs/2409.12183)) 研究了**思维链 (CoT)** 推理提供最大优势的场景，特别是针对**数学和符号推理**任务。
- **数学和符号推理在 CoT 下表现优异**：研究表明，**思维链 (CoT)** 主要增强了**数学和符号推理**领域的性能，并对其局限性和优势提供了见解。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1383188973261684826)** (16 messages🔥): 

> `MI300 AMD-FP8-MM, Conv2D on H100, VectorAdd Leaderboard Updates` 


- **MI300 AMD-FP8-MM 获得快速提交**：在 **MI300** 的 `amd-fp8-mm` 排行榜上，一项提交达到了 **5.23 ms**。
   - 另一项提交以 **161 µs** 的时间位列**第 9 名**。
- **H100 Conv2D 持续产出**：在 **H100** 的 `conv2d` 排行榜上出现了多次成功的提交，时间大约在 **187-192 ms** 左右。
   - 这些提交表明 **H100** 在 `conv2d` 任务上具有稳定的性能表现。
- **VectorAdd 活跃度极高**：许多提交更新了各种 GPU（**A100**、**H100**、**T4**、**L4**）的 `vectoradd` 排行榜，时间跨度从微秒到毫秒不等。
   - 其中一项提交在 **T4** 上以 **6.31 ms** 的成绩获得**第三名**。


  

---

### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1383161516639125617)** (162 条消息🔥🔥): 

> `Factorio RL, Factorio Agents, Hierarchical RL and LLMs, FLE API` 


- **DOTA 与 Factorio 游戏难度之争**：成员们讨论了职业 **DOTA** 比赛是否比常规的 **Factorio** 游戏更难。理由是 **DOTA** 大量消耗算力（**数百万 GPU-hours**）并依赖奖励塑造（reward shaping），而 **Factorio** 拥有更大的动作和观测空间，且对生产平衡非常敏感。
   - **Factorio** 的动作和观测空间更大，且需要计算和平衡中间产物的生产，任务跨度（horizon）更长。成员们好奇如果对 **Factorio** 进行重度奖励塑造并引入人类先验（human prior）进行 **RL** 训练，是否能用与 **OAI5** 相当的算力实现火箭发射。
- **FLE：基于课程的代码生成有望成为 LLM 探测工具**：成员们认为 **FLE** 目前采用的代码生成、生产得分反馈、**REPL** 循环和内存压缩设置是一个有用的脚手架（scaffolding），它缩小了动作空间，并引导 **LLM** 在 **Factorio** 中进行结构化规划。
   - 一位成员建议采用基于课程的代码生成方法，由微型目标和 **FLE** 循环内的心理理论（theory of mind）模块引导，这似乎是探测 **LLM** 在该环境中规划能力极限的一种很有前景的方式，类似于 **minedojo** 的 **voyager** 和 **mindforge**。
- **团队探索分层 LLM+RL 混合系统**：团队探索了一种混合架构，将组合规划交给 **RL** 循环，调用可重用的 **LLM** 存根（stubs）进行具体实现。该系统将基于符号化基础设计原语的中长期规划交给 **RL** 循环，而 **LLM** 负责处理实现细节。
   - 一位成员指出，**HLP**（高层规划）和 **LLP**（底层规划）的划分是合理的，因为 **LLM** 具有“人类先验知识乘数”技能，可以灵活切换层级；而技能集程序（skillset procedures）由于会产生组合爆炸，很难直接进行组合。
- **FLE API 为容器集成增加 REST API**：一位成员在 **Factorio** 容器中集成了 **REST API**，服务器选用 **C#** 编写，因为它能编译为机器码，且容器除了两个二进制文件外不需要其他依赖。
   - 在 [GitHub](https://github.com/MortenTobiasNielsen/FLE-API-specification/issues/5) 上发起了一个关于动作（actions）的讨论：我们需要哪些动作以及应该如何命名。
- **`connect_entities` 简化 Factorio Agent 开发**：`connect_entities` 功能可以防止 Agent 显式设计传送带/电线杆/管道的路径，但移除该功能会导致 Agent 完全无法胜任工作。
   - 一位成员建议，与其完全移除，不如研究如何让 Agent 能够更灵活地配置 `connect_entities`。


  

---


### **GPU MODE ▷ #[amd-competition](https://discord.com/channels/1189498204333543425/1359640791525490768/1383429247921684600)** (1 条消息): 

> `AMD GPU, Image Analysis` 


- **图像分析即将到来**：一位用户发送了一张可能与 **AMD GPU** 相关的**官方图片**。
   - 该图片被发送以备参考，但未提供进一步的背景信息。
- **AMD GPU 推测**：据推测，该图片包含有关即将推出的 **AMD GPU** 的详细信息，可能与竞争定位有关。
   - 在没有更多背景的情况下，该图片的具体意义尚不明确，但它暗示了内部文档或营销材料。


  

---


### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1383561639219171438)** (1 条消息): 

> `BLISRetreat2023, UTexas presentation` 


- **寻找幻灯片视频**：一位成员询问是否有与 [德克萨斯大学](https://www.cs.utexas.edu/~flame/BLISRetreat2023/slides/Thakkar_BLISRetreat2023.pdf) **BLISRetreat2023** 演示幻灯片配套的视频。
- **大学演示**：该演示讨论了与 **BLIS** 相关的主题。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1383160456776126647)** (196 条消息🔥🔥): 

> `Manus credits, Manus speed, Minimax copy of Manus, Manus AI updates, Manus Agent mode` 


- **Minimax 像素级致敬 Manus 功能**：成员们指出 [agent.minimax.io](https://agent.minimax.io/) 抄袭了 Manus，并提到它在*公布积分机制前极具潜力*，但*愚蠢的定价毁了它*。
- **用户抱怨 Manus 存在“吞分”错误**：用户报告称 **Manus 因自身错误而损耗积分**，一位用户表示*因为系统自身的错误，它吞掉了我全部 4000 多个积分。*
   - 有投诉称它**消耗了 700/1000 积分却只交付了一个黑屏网站**。
- **有人在推销 Manus 的免费替代方案 Lume**：成员们讨论了将 [lume.im](https://lume.im) 作为 Manus 的替代品。
   - 一名用户宣传它是*免费且无限制的*，这引发了关于“当托”和发垃圾信息的指责。
- **Gemini 在特定任务中胜过 Manus**：一位成员表示 *Manus 做不到，但 Gemini 可以*，并分享了一个 [Gemini 输出链接](https://g.co/gemini/share/ef9c34a02d31)。
   - 该用户还表示：*Gemini 是目前最好的静态画布（static canvas）。Manus 不是静态的，所以我们无法将两者结合。*
- **用户报告 Manus 速度缓慢且任务失败**：用户抱怨 **Manus 速度慢、不遵循指令**，且新更新使其变得更糟，此外**简单的文档编译需要 40 分钟并消耗 400+ 积分**。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1383179252001669311)** (112 条消息🔥🔥): 

> `Decentralized Pre-training, Hermes 4 Training, Bandwidth Differentials, AI Evals company, Multilingual reasoning in AI` 


- **Nous 推出用于预训练的 Psyche**：Nous Research 正在 [psyche.network](https://psyche.network) 上进行预训练，与此同时 **Jensen Huang** 在预训练（pre-training）与后训练（post-training）的问题上吐槽了 Anthropic。
   - 一位成员指出*分布式技术从现在起只会变得更好*，并将使去中心化训练受益。
- **Dawn Internet 推广去中心化宽带**：[Dawn Internet](https://x.com/dawninternet) 是一个去中心化宽带协议，通过固定无线屋顶天线提供**千兆互联网**。
   - 他们的新款 **WiFi 路由器**包含一个能够支持 RL 的 GPU。
- **Nous 将开始训练 Hermes 4**：Nous Research 将于周一开始**训练 Hermes 4**，不过训练和准备仍需一段时间。
   - Zeus 系列的新模型将不基于旧的 Hermes，而是基于最新的 Mistral。
- **Atropos RL 环境支持 Axolotl**：根据 [Discord](https://discord.com/channels/972392335214743642/974678437632428072/1283737286656051200) 中的讨论，Atropos RL 环境目前已支持 Axolotl（使用 TRL），且有成员正在研究 VERL 的集成。
   - 一位成员表示 Atropos *非常出色*，并分享了 [Atropos 的 Readme 文件](https://github.com/NousResearch/atropos?tab=readme-ov-file#axolotl)以供了解更多信息。
- **Kimi-Dev-72B 发布开源编程 LLM**：**MoonshotAI** 推出了 [Kimi-Dev-72B](https://huggingface.co/moonshotai/Kimi-Dev-72B)，这是一款针对软件工程任务的新型开源编程 LLM，在 **SWE-bench Verified** 上取得了开源模型中的新 SOTA，得分为 **60.4%**。
   - Kimi-Dev-72B 通过大规模强化学习（RL）进行了优化，能够在 Docker 中自主修复真实仓库，并且只有在整个测试套件通过时才能获得奖励，这与现实世界的开发标准保持一致。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1383265720632086628)** (8 条消息🔥): 

> `Gemini 2.5 Pro, Chain of Thought (CoT) prompting, Reasoning Techniques, API key setup, Hyperbolic integration` 


- **AI Studio 上部署的 Gemini 2.5 Pro**：一名成员确认在 **AI Studio** 上使用了 **Gemini 2.5 Pro**，并询问了在长对话中强制模型使用 **Chain of Thought (CoT)** 的方法。
   - 用户指出 **CoT** 的触发并不稳定。
- **增强模型性能的推理技术**：一名成员建议提示模型使用推理技术，如 **neurosymbolic**、**counterfactual**、**inductive** 和 **deductive**。
   - 他们建议明确指示模型如何思考，并输入 *Alternatively*、*consequentially* 和 *due to* 等关键词来引导推理过程。
- **API Key 设置说明**：一名成员提供了设置 **API key** 和相关配置的指导，并指出了特定的齿轮图标位置。
   - 附带了一张图片以进一步说明该过程 ([image.png](https://cdn.discordapp.com/attachments/1154120232051408927/1383446172143714334/image.png?ex=68517526&is=685023a6&hm=452d7b1e5ae55dcec88e3f2039c68055dfd71879962a76661b5b42715509ed6b&))。
- **Hyperbolic 连接问题**：一名成员报告称，尽管完成了 **API key** 设置，但在将 **Gemini 2.5 Pro** 连接到 **Hyperbolic** 时遇到困难。
   - 讨论集中在集成过程的故障排除上。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383409025017843812)** (19 条消息🔥): 

> `Bitter Lesson, Generalist vs SME, Grounding in reality, Gene edits to cure cancer` 


- **Bitter Lesson 总结**：围绕 Rich Sutton 的 [Bitter Lesson](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) 文章展开讨论，该文探讨了**创新如何随 Moore's Law 扩展**，而非人类对齐（human alignment）。
   - 有人指出，该文章需要假设“现实最终是计算性的”以及“发现 = 可观察的人类现实”。
- **癌症治愈悖论**：一名成员使用一个假设案例提到了**现实锚定（grounding in reality）**的问题：一个模型发现了可以治愈癌症的基因编辑方法，但却不理解其原理。
   - 他们指出，如果不理解治愈机制，可能会产生意想不到的后果（如导致 ALS），并引用了[这项研究](https://arxiv.org/abs/2506.10911v1)。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1383164870308003923)** (8 条消息🔥): 

> `WebSummit talk on closed internet/AI, Robotic Skin, Deep Residual Learning` 


- **WebSummit 上关于封闭互联网的抨击**：一名成员分享了在温哥华 [WebSummit 上的演讲](https://youtu.be/vZVcBUnre-c)，内容关于封闭互联网和封闭 AI，一半是历史，一半是抨击。
   - 另一位用户在 [FXTwitter](https://fxtwitter.com/jyo_pari/status/1933350025284702697) 上转发了此内容。
- **来自剑桥的神奇机器人皮肤**：一名成员发布了关于[剑桥机器人皮肤](https://www.cam.ac.uk/stories/robotic-skin)的内容，并链接到了 [YouTube 视频](https://youtu.be/BV5I4w_wxKI?si=lkADZ69PpfxCUlZt)。
   - 这种皮肤似乎是由嵌入了传感器的 **stretchable matrix**（可拉伸矩阵）制成的。
- **Deep Residual Learning 论文**：一位用户分享了 **CVPR 2016** 的 [Deep Residual Learning 论文](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)链接。
   - [论文摘要](https://arxiv.org/abs/2506.10943)链接指向了一个不存在的 arXiv ID。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1383409025017843812)** (19 messages🔥): 

> `Bitter Lesson, Generalist vs SMEs, Nature Article, Arxiv Paper, Observable Reality` 


- **Bitter Lesson: Scaling 胜过人类对齐 (Human Alignment)**：围绕 Rich Sutton 的 [“Bitter Lesson”](https://www.cs.utexas.edu/~eunsol/courses/data/bitter_lesson.pdf) 论文的讨论强调，*重大的创新*归功于随摩尔定律而来的 Scaling，而非与人类对齐。
   - 该文章指出创新随规模扩展，但理解它们需要观察，否则我们面临无法理解系统的风险，就像基因编辑虽然能治愈癌症，但由于我们不了解其机制，也会产生意想不到的副作用。
- **Nature 文章被部分人视为“无用”**：一位成员分享了一篇 [Nature 文章](https://www.nature.com/articles/s42256-025-01049-z?fbclid=IwY2xjawK6iz5leHRuA2FlbQIxMABicmlkETEzb2FnQWNpQzlpQlBzMmhQAR6aSa6Wtu4htiNOE9nvcR4GLRJIaaaBOm1gFYChLS_g5c7G0wk29w2Ohbn_KA_aem_IRJuLT1puoERTjNu2VVTnQ)，但评论道 *“真的别看，我所有的中国朋友都说它没用”*，未提供更多背景。
- **发布了新的 Arxiv 论文**：一位成员发布了一篇新的 [Arxiv 论文](https://arxiv.org/pdf/2407.01067) 和另一篇 [Arxiv 论文](https://arxiv.org/abs/2506.10911v1)，没有其他背景说明。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1383170950245253130)** (135 messages🔥🔥): 

> `VS Code forks, TUI, RA-Aid, Context Window Management, LLM Personas` 


- **Electron 应用估值极高**：一位用户开玩笑说，与其构建 TUI，不如 fork 一个 Electron 应用，并提到了 **VS Code** 衍生版本 **90 亿美元的估值**。
   - 一位成员同意这是一个过度设计的方案，但他们指出，每个大学毕业生以及部分在校生都会接触到 VS Code。
- **RA-Aid 与 Aider 的集成得到澄清**：在对 [RA-Aid](https://github.com/ai-christianson/RA.Aid) 进行了一些探索后，一位成员发现 **Aider** 的明显优势在于其 **repo map** 以及允许用户将文件添加到上下文的功能。
   - 然而，他们提到 Aider 在消耗了 32K tokens 后似乎什么也没做，花费了 5 美分，这让他感到震惊，随后 Aider 对代码库进行了暴力 grep。
- **系统性的人格 (Persona) 评估**：一位用户链接了一篇 [Arxiv 论文](https://arxiv.org/html/2311.10054v3)，该论文认为 *在系统提示词中添加人格 (personas) 并不能在各种问题上提高模型性能，与未添加人格的对照组相比并无优势*。
   - 他们长期以来都有这种感觉，但想知道是否有实际的研究支持这一观点。
- **生成了新的头脑风暴 UX 功能**：一位用户提示 DeepSeek 生成了不同层级的功能，分为 **Realistic (现实)**、**Outside the Box (跳出框架)** 和 **Completely Bonkers (完全疯狂)**。
   - **Completely Bonkers** 层级包含了诸如 *反重力代码重排 (Anti-Gravity Code Reflow)* 和 *多重宇宙分支 (Multiverse Branching)* 等建议。
- **请求 Aider 上下文窗口功能**：一位用户询问是否可以为 **Aider** 添加一项功能，允许它自主管理上下文窗口，不仅是添加文件，还能根据需要移除或清理上下文窗口。
   - 另一位用户指向了 [repomap](https://aider.chat/docs/repomap.html) 和 [scripting](https://aider.chat/docs/scripting.html) 文档，通过提供 repo map 让 Aider 作为工具来控制上下文。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1383262463528800426)** (18 条消息🔥): 

> `LLM-OpenAPI-minifier 与 aider 的集成，在 Aider 中设置 API keys，Aider 的 Agent 能力，为 Qwen3 等 MoE 模型在 VRAM 中加载活跃参数` 


- **寻求 Aider 与 LLM-OpenAPI-minifier 的集成**：一位成员询问如何使用 [LLM-OpenAPI-minifier](https://github.com/ShelbyJenkins/LLM-OpenAPI-minifier) 将应用程序接口与 Aider 集成。
- **请求在 Aider 程序内设置 API Key**：一位成员询问如何在 Aider 内部设置 **API key**，并指出文档中缺少该功能；另一位成员建议采用类似 `llm keys set anthropic xxxx` 的命令模式来设置密钥。
   - 有成员询问该功能是否在路线图中，以及*新手*是否可以为此贡献 PR，并引用了 [Simon Willison 的 `llm` 工具](https://simonwillison.net/2023/May/8/llm-cli/) 作为灵感来源。
- **确认 Aider 有限的 Agent 功能**：一位成员质疑 Aider 是否具备完全的 Agent 能力，因为他们无法让其作为 Agent 工作、修改代码或运行命令。另一位成员澄清说 Aider *并非真正的 Agent*，但 `/run` 命令可用于有限的场景。
   - 他们提到一个名为 **gitmind** 的个人项目曾尝试实现此功能，但后来被放弃了。
- **提议为 Qwen3 MoE 进行选择性 VRAM 加载**：一位成员询问在运行 **Qwen3 30B MoE** 等模型时，是否可以仅在 **VRAM** 中加载**活跃参数**，旨在 3090 GPU 上使用 Q8 量化且不产生明显的性能下降。
   - 他们澄清说，希望避免加载特定 Prompt 不需要参数（例如在关注代码时避免加载语法层）。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1383185193862565889)** (136 条消息🔥🔥): 

> `用于团队管理的 Claude Swarm，主动型 AI Agent 的定义，Anthropic 的多 Agent 系统，LLM 作为评估裁判，面向作家的 Cursor AI 工具替代方案` 


- **Claude Swarm 通过 Team MCP 席卷 Shopify**：**Claude Swarm** 是一个利用 Claude Code 的 MCP 能力构建专家级层级团队的工具，目前正在 Shopify 和其他公司获得关注（[代码在此](https://github.com/parruda/claude-swarm)）。
   - 一位用户建议让其中一名专家担任*招聘人员*，以管理 Swarm 配置和团队扩展。
- **IMPACT 框架挑战主动型 AI Agent 的定义**：一篇博文将**主动型 AI Agent** 定义为能够控制自身唤醒计划、工作流、拥有持久记忆并使用有状态工具的实体（[Substack 文章](https://bryanhoulton1.substack.com/p/building-proactive-ai-agents)）。
   - swyxio 根据他的 IMPACT 框架对比了这一定义，指出其缺乏**意图 (intent)**、**规划 (planning)** 和**授权 (authorization)**。
- **Anthropic 多 Agent 系统表现优于 Opus 4**：**Anthropic** 发现，一个以 **Claude Opus 4** 为主导 Agent、**Claude Sonnet 4** 为子 Agent 的多 Agent 系统，在内部研究评估中表现优于单 Agent Claude Opus 4 达 **90.2%**（[Anthropic 博客文章](https://www.anthropic.com/engineering/built-multi-agent-research-system)）。
   - 由于并行化和广泛的工具使用，该系统消耗的 Token 约为普通对话的 **15 倍**，需要通过 Prompt Engineering 来防止过度生成 Agent 以及在网络上搜寻不存在的来源；LLM 也被用于评估输出结果。
- **Obsidian Copilot 为 Obsidian Markdown 作者提供类似 Cursor 的体验**：用户讨论了使用 AI 辅助处理 Markdown 文件的工具，提议将 **Obsidian Copilot** 作为一个选项（[Obsidian Copilot](https://www.obsidiancopilot.com/en)）。
   - 用户希望获得简单聊天之外的功能，例如按主题拆分笔记、打标签、聚合笔记以及通过 Anki MCP 创建闪卡。
- **月之暗面 (Moonshot AI) 发布 Kimi-Dev-72B 模型**：**月之暗面**开源了他们的 **Kimi-Dev-72B** 模型，在开源模型中的 SWE-bench Verified 测试中达到了 **60.4%** 的最高水平 (**SotA**)（[HF 模型](https://moonshotai.github.io/Kimi-Dev/)）。
   - 该消息由 Aran Komatsuzaki 在 Twitter 上发布，并提供了 Hugging Face 模型和 GitHub 仓库的链接。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1383270054195499069)** (75 条消息🔥🔥): 

> `新领域的里程碑论文，LLMs 作为叙事模拟器，带有数学错误的 AI 生成论文，pytorch dataloader workers，EleutherAI 社区 vs 研究重心` 


- **寻找开创性研究：新领域导航**：一位成员询问在进入新领域（具体是在具备图像生成知识的基础上探索视频生成）时，如何找到里程碑式的论文。
   - 建议包括浏览教授的讲义、直接询问业内人士，以及从各大实验室的近期论文入手以利用其引用文献，另请参阅 [LessWrong 上的这段讨论](https://www.lesswrong.com/collaborateOnPost?postId=9eehTtLsTBZR9Bd7Q&key=3b19568356a160ba7cf74febafdf33)。
- **LLMs 作为叙事导航员，模拟故事**：一位成员询问是否可以发布他们的 English 101 论文，该论文探讨了 **LLMs** 因其架构和涌现行为（emergent behavior）而作为叙事模拟器的特性。
   - 该请求因属于寻求评审而被拒绝，但分享了一个指向 [tumblr 上 The Void 帖子](https://nostalgebraist.tumblr.com/post/785766737747574784/the-void)的链接，其中包含相关的分析和示例。
- **数学错误损害 AI 生成的论文**：发出了一项关于 AI 生成论文的警告，引用了 [arxiv.org](https://arxiv.org/abs/2506.09250) 上一篇针对 Apple 论文的所谓（可疑）回应，其中充斥着数学错误。
   - 一位成员分享了[这条推文](https://x.com/BlancheMinerva/status/1933845602917290145)的链接，指出了其中的错误。
- **Pytorch Dataloader 之灾：WSL Workers 的困扰**：一位用户报告了 **PyTorch dataloader workers** 在 WSL 中被信号终止（*killed by signal*）的问题，特别是在 worker 数量较多且序列长度较长的情况下。
   - 建议检查 `/var/log/syslog` 以查找潜在的 **OOM** 错误，并在处理长视频序列时更加注意内存使用。
- **Eleuther 的特质：平衡新手与研究核心**：有人对 Discord 对新成员传达的矛盾信息表示担忧，认为热情的网页文案与以研究为中心的互动之间存在反差。
   - 社区成员讨论了在欢迎新手与维持研究级讨论之间的平衡，强调了 AI 教育与教授研究技能之间的区别，并讨论了 LLM SEO (Language Engine Optimization)。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1383173632439881881)** (50 条消息🔥): 

> `DMCA 与版权法，独立任务中的涌现行为，Llama-3.2-1B-Instruct ARC-AGI，Qwen3 分词器与图像理解` 


- **版权法就是个笑话**：一位用户表示，*除非你是滥用者，否则版权法就是一个笑话*，并对 DMCA 和版权欺诈（copyfraud）处罚发表了评论，附带了 [fxtwitter.com](https://fxtwitter.com/jyo_pari/status/1933350025284702697) 和 [arxiv.org](https://arxiv.org/abs/2506.10943) 的链接。
- **独立任务导致涌现行为**：论文表明，独立任务 X 和 Y 可能会在组合任务 "X and Y" 上表现出**涌现行为**（**emergent behavior**），这引发了对深入探索这一现象的论文的搜寻，成员们链接到了 [arxiv.org](https://arxiv.org/abs/2405.15071)。
- **Llama-3.2-1B-Instruct 在 ARC-AGI 上得分 72.5%**：**Llama-3.2-1B-Instruct** 在 **ARC-AGI** 上达到了 **72.5%**，但该测试是从 11 个训练任务和 8 个评估任务的子集中筛选得出的，这些任务在最优 **TTT** 配置下是可解的。
- **Qwen3 获得原始字节分词器和图像补丁**：一位成员正在使用 **FAFO 方法**，采用不同尺寸的 **Qwen3**（**1.7b**、**4b** 和 **8b**），在将分词器切换为原始字节（raw bytes）并使用 **Fuyu 方法**添加图像理解（将图像补丁投影到 token 流中）时进行简单的 **SFT**，使用了 [LLaVAR-Instruct-16K](https://huggingface.co/datasets/HuggingFaceM4/LLaVAR-Instruct-16K) 数据集。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1383162829586169856)** (1 messages): 

> `LLM Fairness, Interpretability Interventions, Unfaithful Chain of Thought` 


- **现实细节触发 LLM 偏见**：一篇新论文揭示，在偏见评估中加入现实细节会触发 LLM 的种族和性别偏见，即使在 **GPT4o** 和 **Claude 4 Sonnet** 等模型中，**面试率差异也高达 12%**。
   - 这些细节包括*公司名称、来自招聘页面的文化描述，或诸如“仅接受前 10%”之类的约束条件*，从而加剧了偏见。
- **可解释性修复公平性缺陷**：虽然 Prompt Tuning 失败了，但基于可解释性的干预措施（如*种族/性别方向的仿射概念编辑/消融 (affine concept editing/ablation)*）能够减少偏见，通常降至 **1% 以下**。
   - [研究论文](https://x.com/a_karvonen/status/1933582375419850806)强调，这种针对性的干预措施能有效缓解已识别的偏见。
- **LLM 表现出 Unfaithful Chain of Thought**：研究发现，检查 LLM 的 **Chain of Thought (CoT)** 无法发现种族/性别偏见的迹象，尽管最终结果显示出明显的偏见。
   - 这展示了一个*现实场景中 Unfaithful Chain of Thought* 的案例，即推理过程掩盖了底层的偏见。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1383231578661982351)** (5 messages): 

> `Benchmark Evaluation Algorithm, Inspect Standard Format, Eval Coalition Effort` 


- **算法池评估模型**：一位成员构想了一个用于基准测试的 **Algorithm + Pool** 系统，允许动态添加新的基准测试并对模型进行评估。
   - 该池将选择测试样本，以解决基准测试饱和问题，并允许比较具有不同能力的对象模型，尽管具体细节仍在演进中。
- **Inspect 标准化评估结果**：一位成员提到 [Inspect](https://inspect.ai) 包含一种用于存储评估结果的标准格式，可能涵盖评估输入、输出、指标和元数据。
   - 他们询问 Inspect 的标准化尚未涵盖哪些具体方面，从而引发了关于该工具能力的进一步讨论。
- **评估联盟寻求规模化实现**：一位成员表示希望加入 **Eval Coalition Effort**，从在自动化设置中规模化实现当前的基准测试和评估开始。
   - 另一位成员确认他们很快将被添加到 **evaleval Slack** 中，并欢迎他们的加入，因为该工作仍处于早期探索阶段。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1384062892394942484)** (1 messages): 

> `Vitabyte Founder, GroK-scale Training, Multi-node LLM fine-tuning, ROCm + CUDA, Full stack Ops` 


- **Vitabyte 创始人寻求 Grok 规模项目**：George，**Vitabyte/Vitapay** 的创始人，正寻求加入任何 **Grok 规模 (314B)** 或**多节点 LLM 训练/微调项目**。
   - 他带来了 **ROCm + CUDA** 配置、**Quantization** 和 **Full stack Ops** 方面的经验，可以在 Infra、日志、微调流程和文档方面做出贡献。
- **Vitabyte 创始人技能**：George 拥有 **ROCm + CUDA** 配置、**Quantization** 和 **Full stack Ops** 的经验。
   - George 可以在 Infra、日志、微调流程和文档方面提供支持。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1383195275702829076)** (19 条消息🔥): 

> `Notebook LM Plus 访问权限, PM 面试对话式 AI 平台, 使用 NotebookLM 进行备考, NotebookLM 的 Chrome 扩展程序, 播客个性塑造` 


- ****Notebook LM Plus 访问权限仍存疑问****：一位付费 **AI Pro** 用户询问，尽管已订阅，但仍无法访问 **NotebookLM Plus**，并提到使用了 **1900** 个来源。
   - 该用户还分享了一个 [NotebookLM 思维导图](https://cdn.discordapp.com/attachments/1124403655819415592/1383205649839820830/NotebookLM_Mind_Map.png?ex=6851e6a5&is=68509525&hm=f1cd4c62bca69adfe75ec5f0adc1b806be3b22beaa24c3bac7597185b382a6d7)，指出其有 **4 个子层级**，纵向密度很高，但横向尚不密集，文件大小约为 **9MB**。
- ****AI 平台旨在助力 PM 面试****：一位成员正在开发一个专为 **PM 面试** 设计的**对话式 AI 平台**，并正在寻找 Beta 用户来验证他们的想法并提供反馈。
   - 感兴趣的用户可以通过 [此表单](https://forms.gle/P3f2JP3vVB62vedb7) 注册以加入候补名单 (waitlist)。
- ****NotebookLM 助力备考****：一位用户咨询如何使用 **NotebookLM** 准备非 PDF 格式（由网页和虚拟实验室组成）的考试材料。
   - 另一位用户建议使用 **Chrome 扩展程序** 来跟踪并将网页链接导入 **NotebookLM**。
- ****播客主持人寻求个性塑造策略****：一位成员正在深入研究如何塑造其 **NotebookLM 播客主持人** 的个性，并寻求与其他成员交流经验。
   - 另一位用户询问了将节目发布到 **Spotify** 的策略和应用。
- ****将网站扁平化为 Notebook 的单一来源受到关注****：一位成员提议创建一个网站的“扁平化”版本——即一个包含所有内容且不含链接的单一页面——以便将其作为单一来源轻松喂给 **NotebookLM**。
   - 另一位用户建议使用 **Web Sync** 工具，可通过 [这篇文章](https://www.xda-developers.com/notebooklm-chrome-extensions/) 了解。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1383177666110427278)** (86 条消息🔥🔥): 

> `NLM 中的 LaTeX, 图片上传, NotebookLM Android 应用, 播客问题, iPad 上的思维导图` 


- **NLM 支持 LaTeX 标记！**：NotebookLM 与其他 LLM 一样，支持使用 **LaTeX 标记** 来处理数学和科学公式。
   - 要查看这些公式，用户可以使用在线或离线 **LaTeX 渲染器**，或尝试 [LatexInNotebooklm](https://github.com/ergs0204/LatexInNotebooklms) 扩展。
- **图片上传问题已解决！**：用户发现 NotebookLM 现在支持直接从设备上传图片，而不是从 **Google Drive** 上传。
   - 要上传图片，用户可以点击“选择文件”选项或通过拖拽操作。
- **Android 应用好评激增！**：用户纷纷称赞 **NotebookLM Android 应用** 的便利性，尤其是用于收听深度探讨 (deep dives)。
   - 不过，有成员提到，若要使用选择播客长度等完整功能，最好还是使用网页版。
- **播客音频质量下降！**：用户注意到 **NotebookLM 播客** 的音频质量和内容有所下降，对“源材料”的表述显得机械且重复。
   - 该问题影响了生成的播客，被描述为*听起来支离破碎且虚假*。
- **思维导图在 iPad 上消失！**：用户报告称在 **iPad 应用** 中无法看到**思维导图**。
   - 用户正期待能够以**交互式对象**而非图片格式保存思维导图的功能。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1383503120633696498)** (69 messages🔥🔥): 

> `DTensor cross-mesh operation, Llama4 maverick finetuning, Iterable packing, Fused optimizer, Flex attention` 


- ****分布式 Llama4 中的 DTensor 困扰****：成员们在最新的 nightly 构建版本上对 **Llama4 Maverick** 进行多节点分布式 Finetuning 时，遇到了与 **DTensor cross-mesh operations** 相关的 `RuntimeError`，特别是在 `aten.mm.default` 和不同的 Device Mesh 配置下。
   - 该错误在不同数量的节点（8 个 vs 12 个）上表现不同，指向了 **Fused optimizer** 和 Mesh 配置的潜在问题，堆栈跟踪信息可在 [output.log](https://cdn.discordapp.com/attachments/1383590304204066927/1383593694438887444/output.log?ex=6851fe8a&is=6850ad0a&hm=ba8d4aa73234564e80a2a6cb9d08c300b527ace1245e8d0feca028020bed5079&) 中查看。
- ****Iterable packing 创新即将到来****：一位成员正在开发一个基于 [pytorch/data](https://github.com/pytorch/data) 的私有 Finetuning 库，支持 **Iterable packing**，并展示了出色的结果和 Prefetching 能力。
   - 他们建议可能不需要单独的 Dataset Wrapper，主要的开销来自于 Tokenization。预计下周将开源该库，并指出许多库目前都缺少 Packed DPO 支持。
- ****Fused Optimizer 在全量微调中受挫？****：在训练尝试中，发现 **Fused optimizer** 会导致问题，特别是在创建 Checkpoint 时会导致 `nccl` 超时，而使用 Non-fused optimizer 则可以在 8 个节点上正常训练。
   - 有建议称增加 `NCCL_TIMEOUT` 环境变量，或设置 `total_epochs=self.total_epochs+1` 以启用异步 Checkpoint，可能会缓解这些问题；同时建议为该优化器问题创建一个最小可复现示例（reproducible example）。
- ****Mini-Batch 思考与 MoE 显存掌控？****：一位成员推测，使用 1 的 Micro Batch Size 是否能降低训练 **MoE (Mixture of Experts)** 模型的显存需求，因为只需要为激活的参数分配显存。
   - 这一想法被提议作为一种通过将梯度累积卸载到 CPU RAM 来训练超大型模型的方法，但另一位成员指出，Micro Batch Size 实际上是 `seq_len`，因为训练时仍然需要所有的 Experts。
- ****Flex Attention 与 Flashy Nesting？****：成员们讨论了为了简化而强制将 Packed Batches 的大小设为 1，以及这与 **Flex attention** 的联系。一位成员发现其性能从 Non-flex attention 的 2k TPS 提升到了 10k TPS。
   - 成员们建议使用 SDPA + FlashAttention 3，但这样 Tensor 必须是 **Nested Tensors**（使用具有 Jagged Layout 的 `torch.nested`），同时指出在使用 Nested Tensors 时许多算子（Ops）会缺失。


  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1383519442020991189)** (15 messages🔥): 

> `Mistral Small, Magistral model, ZO optimizer, Flex integration` 


- **Mistral Small 亮相，令人失望？**：尽管最近发布了 [Mistral Small 模型](https://mistral.ai/news/mistral-small-3-1)，但并未给所有人留下深刻印象，一位成员表示 *Mistral Small 的结果，甚至在他们自己的博客文章中看起来也仅仅比 Gemma 3~~qwen3~~ 好一点点*。
   - 该成员还澄清说，他们在研究时最初误将 **Magistral** 点击成了 **Mistral**。
- **ZO Optimizer 承诺节省 VRAM**：成员们讨论了 **ZO optimizer** 及其在 **3 倍 VRAM 节省**方面的潜力，并引用了相关论文 ([arxiv.org/abs/2506.044303](https://arxiv.org/abs/2506.044303))。
   - 一位成员觉得 *ZO 居然能跑通简直太神奇了*，而另一位成员建议将其加入 **Flex**。
- **Flex 集成优先级较低**：一位成员建议将 **ZO** 加入 **Flex** 以实现其 **3 倍 VRAM 节省**，但另一位用户回应称 *“我不会优先考虑它，但最终肯定会加入”*。
   - 成员们一致认为，**ZO** 论文最重要的启示是它在不同规模上的可扩展性，以及它主要使用了非合成（non-synthetic）实验。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1383944665656459395)** (46 messages🔥): 

> `RDNA4 support, AVX512_BF16, Zen 4, Mojo testing structure, 1-bit model support` 


- ****RDNA4** 支持已上线！**: 从最新的 nightly 版本开始，**Mojo** 已支持 **RDNA4** 的直接 GPU 编程，但完整模型支持尚未就绪，因为矩阵乘法操作需要实现 **RDNA 特定的路径**。
   - 已添加一个初步补丁，引入了部分必要的 **WMMA** 操作，使模型在 **RDNA3**+ 上的完整运行更近一步。
- ****Zen 4** CPU 的 **bfloat16** 支持**: 虽然 **5950x** 不支持 **AVX512_BF16**，但 Zen 4 及以上架构的 CPU（如 **Ryzen 7000 系列**）提供了一定的 **bfloat16** 支持。
   - 然而，目前尚未确认这些是否包含 CPU 推理所需的精确 **FMA** 指令。
- **探索 Mojo 的测试代码库**: 用户对 **Mojo** 的测试代码库结构表示困惑，特别是测试文件中的导入以及对 package **__init__.mojo** 层级的理解。
   - 一个重要的发现是，使用 `mojo test -I .` 运行测试允许将正在测试的 package 像 library 一样导入；一位用户建议参考 [ExtraMojo](https://github.com/ExtraMojo/ExtraMojo) 作为一个良好的项目结构示例。
- ****LLVM** 占据了大部分二进制文件体积**: 二进制文件的大部分体积源于静态链接 **LLVM**，**MAX** 本身约为 **750 MB**，随 **MAX** 附带的 .mojopkgs 约为 **100 MB**。
   - 目前正在积极开展减少 **LLVM** 副本数量的工作。
- ****Intel Nova Lake** 将拥有 52 个核心？**: 下一个“编译 SKU”很可能是 **Intel Nova Lake**，因为其顶级 SKU 可能拥有 **52 个核心**。
   - i9 可能是拥有这么多核心的型号，而 **Intel** 的 HEDT 方案则是“买一颗 **Xeon**”。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1383167634421579776)** (32 messages🔥): 

> `CUDA Stream Synchronization, Mojo C ABI, Mojo Zed Extension, Mojo 'let' deprecation, Mojo AOT compilation` 


- ****CUDA Stream** 是否不需要 **Host** 同步？**: 一位成员询问在 [Puzzle 12](https://builds.modular.com/puzzles/puzzle_12/complete.html#host-side-synchronization-the-critical-step) 中 `ctx.synchronize()` 是否必要，认为 **CUDA streams** 会自动处理依赖 Kernel 启动的同步。
   - 一位 Modular 团队成员确认 *DeviceContext* 使用了 **CUDA stream**，因此执行顺序与调用顺序一致，不需要显式同步，并承诺会相应调整文档。
- **Mojo 通过 **`external_call`**** 调用 C**: 一位成员询问如何从 **Mojo** 调用 **C**，并在文档中寻找示例。
   - 另一位成员指出可以使用 **`external_call` 函数** 进行 **Mojo** 与 **C** 的互操作。
- **Mojo Zed 扩展仍可正常使用**: 一位用户报告说 **Zed 的 Mojo 扩展** 运行良好，并询问未来的更新计划。
   - 扩展开发者确认他们正在更多地使用 **Mojo** 并征求具体的功能需求，但目前存在一个关于不必要的高亮显示未使用的变量的 [issue](https://discord.com/channels/1087530497313357884/1329089055211655231/1331626952561397831)。
- ****`let` 声明正式退役****: 一位 **Mojo** 初学者询问了已弃用的 **`let` 变量声明**，因为在教程中遇到了错误。
   - 一位团队成员确认 **`let`** 已在 [24.4 变更日志](https://docs.modular.com/mojo/changelog#v244-2024-06-07) 中移除，并指出大多数 **Mojo** 教程更新较慢，官方提案见 [此处](https://github.com/modular/modular/blob/main/mojo/proposals/remove-let-decls.md)。
- **关于 Mojo **AOT 编译** 的讨论**: 一位成员询问 **Mojo** 的哪些部分是 **JIT** 编译，哪些是 **AOT** 编译，特别是关于 **SIMD** 和运行时统计信息。
   - 一位成员澄清说，除非是在 **MAX** 的 Kernel 内部，否则 **CPU 代码** 是 **AOT** 编译的；而 **GPU 代码** 由于需要驱动程序特定的优化，使用的是 **JIT** 编译器。之前存在的 autotune 库已被移除，因为它会大幅增加编译时间。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1383159529822228490)** (40 messages🔥): 

> `Agentic Frameworks 中的 MCP, A2A Agent 发现, FastMCP 与 Server 组合, MCP Server 中的 GitHub APIs, Orchestrator Agent 推荐` 


- ****Agentic Frameworks 拥抱 MCP****：一位 Agent 询问 MCP 在 Agentic Frameworks 中的位置，考虑到顶层是 Orchestrator Agent，随后是访问多个 MCP Server 进行工具选择和内存存储的特定 Agent。
   - 一位成员建议使用具有工具重排序功能的更智能的 Host。
- ****FastMCP 挂载领域隔离子服务器****：一位成员提到 [fastmcp](https://gofastmcp.com/) 可以挂载 MCP Server，允许 Router Server 托管子服务器以实现领域隔离。
   - 正在开发在一个地方公开所有 GitHub APIs 的单个 MCP Server 的团队正在探索 Orchestration Server 的想法，该服务器可以调用或代理到其他 MCP Server，并在工具数量增长时权衡性能。
- ****LLM 选择由客户端完成****：LLM 的选择由客户端完成，所使用的模型完全取决于消费该 Server 的客户端或 App。
   - MCP 团队正在研究如何进行优化，并鼓励查看此处的代码：[GitHub MCP Server](https://github.com/github/github-mcp-server/blob/main/pkg/github/dynamic_tools.go)。
- ****Opus 编排 Cursor****：对于使用 Cursor 的用户，推荐使用 **Opus** 作为 Orchestrator Agent，尽管提到了其成本。
   - 一个人更倾向于使用本地模型。
- ****Streamable HTTP 需要完整 URL****：一位成员指出 streamable-http 需要包含 `/mcp/` 的完整 URL，帮助他人解决了 **fastmcp** 的连接错误。
   - 默认的 streamable-http 端口是 **8000**，而不是 6277。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1383211846315540591)** (7 messages): 

> `预防 Rug Pulls 的 SchemaPin, Glama MCP Server 支持 streamable HTTP, excel-mcp-server, MCP 的用户分析与实时调试` 


- ****SchemaPin** 防止 MCP Rug Pulls**：一位成员构建了 **SchemaPin** 以防止 MCP Rug Pulls 及类似攻击，[代码库](https://github.com/ThirdKeyAI/SchemaPin)已在 GitHub 上发布。
   - [主页](https://schemapin.org)提供了实现 **SchemaPin** 的简便方法。
- ****Streamable HTTP** 在所有 Glama MCP Server 上线**：所有 Glama MCP Server 现在都支持 **streamable HTTP**，例如：[glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257](https://glama.ai/mcp/instances/svuec7nlpl/mcp?token=f6830a11-ded3-4492-8fb0-09eb09b08257)。
- ****Excel MCP Server** 在 GitHub 上走红**：一位成员分享了他们的代码库 [excel-mcp-server](https://github.com/haris-musa/excel-mcp-server)，该项目在 GitHub 上两次进入趋势榜。
   - 他们欢迎对该项目的任何反馈。
- **使用 **MCPCat** 调试你的 MCP**：一位成员正在开发用于 MCP 的用户分析和实时调试工具，代码库可见 [此处](https://github.com/mcpcat)。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1383205705741238413)** (20 messages🔥): 

> `Cohere 文档拼写错误, LLM 团队协作, AI/后端开发者介绍, Cohere 与政府的合作, 安全 ML 与隐私保护` 


- ****拼写错误困扰**：Cohere 文档已修复！**：一位用户报告了 [Cohere 文档](https://docs.cohere.com/docs/amazon-sagemaker-setup-guide)中的一个拼写错误，其中 `co = cohere.SagemakerClient()` 的 `m` 应该是小写的。
- ****LLM 团队合作**：团队如何协作！**：一位用户正在研究团队如何将 ChatGPT 和 Claude 等大语言模型集成到日常工作流中，并询问自引入这些模型以来发生的改变和缺失的元素。
- ****AI 开发者 Kira**：加入聊天！**：AI/后端开发者 Kira 介绍了自己，表达了对建立联系和构建酷炫产品的兴奋，重点关注自定义 Bot、自动化和可扩展系统。
- ****政府项目**：Cohere 的公共部门工作！**：一位用户分享了一段 [Carney 新闻视频](https://m.youtube.com/watch?v=qWBO4LsKdD4&pp=ygULQ2FybmV5IG5ld3M%3D)，强调了 Cohere 与政府的合作，表示这一定是一项巨大的荣誉。
- ****隐私伙伴 Yasir**：安全 ML 爱好者！**：计算机科学毕业生 Yasir Khan 介绍了自己，提到了在安全机器学习（Secure ML）和隐私保护方面的工作，寻求在 AI/ML 项目上的合作机会。


  

---

### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1383312324340879400)** (5 messages): 

> `direct-injected-document tool, Cohere a032025 memory usage` 


- **"direct-injected-document" 工具偶尔出现**：部分用户报告称，一个名为 **direct-injected-document** 的工具会偶尔作为答案弹出。
   - 一位成员询问了 prompt 示例以及正在使用的模型。
- **Cohere a032025 托管需求**：一位用户询问了托管 **Cohere a032025** 的内存需求。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1383758872996478986)** (6 messages): 

> `AI developers introductions, Custom bots, Automations, Scalable systems, Secure machine learning` 


- **AI 开发者集结！**：一位名为 Kira 的 AI/后端开发者介绍了自己，表示愿意帮助初创公司构建**自定义机器人（custom bots）、自动化和可扩展系统（scalable systems）**。
   - 她表达了与他人建立联系并共同构建酷炫产品的兴奋之情。
- **安全 ML 与隐私专家寻求合作**：计算机科学毕业生 Yasir Khan 介绍了自己，重点介绍了他在**安全机器学习（Secure machine learning）和隐私保护（privacy-preservation）**方面的工作。
   - 他表示有兴趣与志同道合的朋友建立联系，并合作开展 AI/ML 项目以提升专业知识。
- **机器翻译专家现身**：来自菲律宾的计算机科学学生 Joel 介绍了自己，他正在研究如何改进**针对菲律宾语的机器翻译（Machine Translation）和 LLM**。
   - 他说他来这里是为了*到处看看，发现酷炫的东西，并可能结识一些有趣的人*。
- **Ollama 模型喜提新粉丝**：一位 AI 领域的新人表达了他们对玩转 **ollama 模型**的喜爱。
   - 他们表示*这很有趣*。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1383159468086398989)** (3 messages): 

> `Data + AI Summit 2025, Agentic Document Workflows, Multi-Agent System, AI Travel Agents, AI Agents in Production` 


- **Data + AI Summit 2025 圆满结束**：@databricks 的 **Data + AI Summit 2025** 已经结束，更多关于新兴的**智能体文档工作流（agentic document workflows）**的内容即将发布，点击[此处](https://t.co/jS2Nfwxxb3)了解更多。
   - CEO @jerryjliu0 进行了一场座无虚席的演讲。
- **微软 AI 旅游代理演示**：@microsoft 新推出的 **AI Travel Agents 演示**展示了如何使用 **Model Context Protocol**、**LlamaIndex.TS** 和 @Azure AI Foundry 协调多个 AI Agent，以应对复杂的旅行规划场景。
   - 六个专业的 AI Agent 协同工作，点击[此处](https://t.co/cNyVAcnf6K)了解更多。
- **在生产环境中构建并保护 AI Agent**：参加在旧金山举行的晚会，获取关于在**生产环境中构建和保护 AI Agent** 的专家见解，涵盖最佳实践，详情见[此处](https://t.co/MVd2rwSVIE)。
   - 我们的开发者关系副总裁 @seldo 将与来自 Ravenna 和 @auth0 的行业专家一起展示**构建现实世界的 Agent（Building Real-World Agents）**。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1383262429617590352)** (26 条消息🔥): 

> `LandingAI vision agent vs LlamaIndex, Synk hiring, Faiss Index, LlamaCloud contact sales page, LlamaExtract parsing errors` 


- **LandingAI 对比 LlamaIndex 文档理解**：成员们讨论了由 **Andrew Ng 博士**创立的 **LandingAI** 公司开发的新型 vision agent 文档理解工具。一位成员参考之前对比 **Mistral** 的[帖子](https://www.linkedin.com/posts/jerry-liu-64390071_mistral-ocr-is-nice-and-fast-but-other-models-activity-7303803148907790336-OP9y)，询问该工具与 **Llama Parse** 的对比情况。
   - 该公司的工具可以在 [LandingAI 官网](https://va.landing.ai/home)找到。
- **Synk 招聘开发人员**：一位成员宣布 **Synk** 正在为其去中心化浏览器系统项目招聘开发人员（**back-end, Front-end, blockchain**）、**QA Engineer**、**DevOps Engineer**、**Moderators** 以及 **Marketing Analyst**，并将用户引导至 [Synk 的 X 页面](https://x.com/Synk_ws)。
   - 他们提供*签署正式文件的正式雇佣、保障薪资以及灵活的排班*。
- **Faiss Index 过滤仍不支持**：一位成员询问是否可以在 **Faiss index 查询中进行 metadata filtering**。
   - 另一位成员回答说 *Faiss 不支持此功能*。
- **LlamaCloud 联系页面失效**：一位成员报告称，**llamacloud 上的联系销售页面** ([https://cloud.llamaindex.ai/contact-sales](https://cloud.llamaindex.ai/contact-sales)) 由于 **500 internal server error** 无法工作。
   - 另一位成员询问他们是否是指 [LlamaIndex 联系页面](https://www.llamaindex.ai/contact)。
- **LlamaExtract 故障导致解析错误**：几位成员报告称，他们在 **LlamaExtract** 中尝试运行的**每个文档都遇到了解析错误**，没有数据被提取出来。
   - 一位成员建议再次尝试，并指出他们能够接收到数据，同时附上了一张使用 LlamaExtract 成功提取的截图 ([image.png](https://cdn.discordapp.com/attachments/1384121428076527656/1384137406202253372/image.png?ex=6851fea9&is=6850ad29&hm=889efa629540fd4d48bbf3c3ecf8421edfaef6967a4732c0fe2cc06ef68a42a6))。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1384311159967977605)** (1 条消息): 

> `DSPy Optimization Patterns` 


- **征求关于整合 DSPy 优化模式的想法**：一位成员询问关于如何整合 **DSPy** 中存在的任何**优化模式 (optimization patterns)** 的想法。
- **JSON 验证的填充话题**：这是一个填充话题，用于确保 JSON 的 topicSummaries 中至少包含两个元素。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1383489483055435877)** (19 messages🔥): 

> `DSPy runners, TextGrad Optimizer, Custom LM Concurrency, DAIS Session Write-Up, BootstrapFewShot Optimizer` 


- **通过 JSON 定义在任何地方运行 DSPy**：一名成员正考虑构建 **DSPy "runners"**，它接收保存的 **JSON definition** 并运行编译后的程序，从而实现跨语言功能，例如 Swift 通过托管 API 利用编译后的程序。
   - 另一名成员表示感兴趣，但质疑如何处理 **JSON output** 中未捕获的程序逻辑（如 signature 和 module），并思考程序应如何序列化。
- **TextGrad 优化器等待更新**：一名成员询问了将 **TextGrad** 作为 DSPy 优化器添加的更新进展，并引用了 [GitHub 上的 issue #1197](https://github.com/stanfordnlp/dspy/issues/1197)，该 issue 已开放近一年。
   - 该成员对 **TextGrad** 表现出极大热情，因其在优化复杂 Prompt 方面非常有效，并询问是否有人有将其整合进 DSPy 的“黑客手段（hacks）”。
- **模型在 DAIS 会议上编写 Prompt**：一名成员分享了他们在下周 DAIS 会议上的会议记录，题为“让模型编写 Prompt”，可在[他们的网站](https://www.dbreunig.com/2025/06/10/let-the-model-write-the-prompt.html)上查阅。
   - 在后续讨论中，一名成员询问是否有会议录像，第一名成员回复了一个 [YouTube 链接](https://youtu.be/I9ZtkgYZnOw?si=XGArjkQSVUlzrEAr)。
- **DeepSeek R1 7B 在 DSPy 优化中表现不佳**：一名成员报告称，在 **DSPy-Text2SQL** 演示中，使用 **DeepSeek R1 7B** 的优化结果不如 **GPT-4o-mini**，并在尝试 **LabeledFewShot** 和 **BootstrapFewShotWithRandomSearch** 后寻求改进建议。
   - 另一名成员建议，提供更多关于 schema 的信息可能会提升 **DeepSeek R1 7B** 的性能。
- **BootstrapFewShot 优化器的使用场景**：一名成员试图了解 **BootstrapFewShot** 优化器的工作原理，特别是针对分类用例，并质疑如何处理自举（bootstrapped）输入的 ground truth。
   - 另一名成员解释说，只要 *它返回 bool、int 或 float（且数值越高越好）*，任何东西都可以作为 metric。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1383424003217035285)** (8 messages🔥): 

> `Certificates, Assignment Selection, MOOC Quiz Archive` 


- **证书将于 7 月中旬发放！**：一名用户询问了证书的发放情况，一名成员回应称 *证书将于 7 月中旬发布*。
- **只要付出合理的努力即可通过作业**：一名用户询问如何知道自己是否被选中获得证书，或者是否通过了作业。
   - 一名成员澄清说，通过 **Google Forms** 提交的每份作业都会收到电子邮件确认，只要完成所有内容并付出了 **合理的努力（reasonable effort）**，就会被授予证书。
- **MOOC 测验存档链接**：一名成员分享了 [2025 春季 MOOC 测验存档](https://docs.google.com/document/d/1A00cUWux-J0p9AOnwpyNN3Rb5QFRsbBgAmvgPMezJ10/edit?usp=sharing)，该存档也可在课程网站的 Quizzes 部分找到。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1384168633235800166)** (1 messages): 

> `ControlThrive, Outerbounds, ML Consulting` 


- **ControlThrive 创始人向社区致意**：AI/ML 咨询机构 **ControlThrive** [controlthrive.com](https://www.controlthrive.com/) 的创始人 Servando 向社区介绍了自己。
   - 他邀请成员在 [LinkedIn](https://www.linkedin.com/in/servando-torres-239a26b0/) 或 X 上与他建立联系。
- **Outerbounds 活动即将举行**：Servando 宣布了他将与来自 **Outerbounds**（Netflix 内部 ML 基础设施背后的团队）的 Eddie 共同举办的活动。
   - 他分享了[活动链接](https://lu.ma/nw4xccle)并鼓励社区成员参加。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1383165546375413890)** (1 messages): 

> `Claude Sonnet 4 API Access, Anthropic models, API Pricing` 


- **Claude Sonnet 4 模型发布**：**Claude Sonnet 4** 和 **Claude Sonnet 4 (Thinking)** 现在通过 [API Pricing](https://docs.windsurf.com/windsurf/models#api-pricing) 向所有付费方案开放。
- **Mohan 对 Claude 的独到见解**：Mohan 在 [X](https://x.com/_mohansolo/status/1933605162775687482) 上转发了一些关于 **Claude 的印象**。