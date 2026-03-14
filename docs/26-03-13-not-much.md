---
companies:
- anthropic
- ibm
- perplexity-ai
- llamaindex
- deepseek
- google-chrome
date: '2026-03-10T05:44:39.731046Z'
description: 尽管在易用性（ergonomics）方面受到批评，但 **MCP 工具** 对确定性 API 而言依然具有重要意义；**Chrome v146
  中新增的 Web MCP 支持**让持续浏览智能体（continuous browsing agents）成为可能。**持久化记忆（Persistent memory）**正逐渐成为智能体的核心竞争优势，IBM
  借此提升了任务完成率，而多智能体记忆则被视为一项计算机架构层面的挑战。智能体的用户体验（UX）正朝着**全天候、跨设备运行**的方向演进，iOS 端的 **Perplexity
  Computer** 和 **Claude Code** 的会话管理便是典型案例。**Anthropic** 推出了默认支持 **100 万（1M）上下文的 Opus
  4.6**，且不额外收取长上下文 API 费用，该模型在 100 万 token 下的 **MRCR v2 测试中达到了 78.3%** 的准确率。诸如 **DeepSeek
  稀疏注意力**中的 **IndexCache** 等优化技术，在仅需极少代码改动的情况下，显著提升了大模型的运行速度。
id: MjAyNS0x
models:
- opus-4.6
- glm-5
people:
- pamelafox
- tadasayy
- llama_index
- bromann
- dair_ai
- omarsar0
- abxxai
- teknuim
- bcherny
- kimmonismus
- _catwu
- alexalbert__
- realyushibai
title: 今天没发生什么特别的事。
topics:
- persistent-memory
- agent-infrastructure
- cross-device-synchronization
- long-context
- sparse-attention
- inference-optimization
- computer-architecture
- task-completion
- systems-performance
---

**平静的一天。**

> 2026年3月12日至3月13日的 AI 新闻。我们检查了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有发现更多 Discord 动态。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。友情提示，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以 [选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack) 邮件接收频率！

---

# AI Twitter 回顾

**Agent 基础设施、MCP 摩擦与持久化内存**

- **对 MCP 的抵制主要集中在易用性而非需求上**：feed 流中很大一部分是工程师在争论 **MCP** 是已经“凉了”还是仅仅过度曝光。[@pamelafox](https://x.com/pamelafox/status/2032315760530665895) 开玩笑说，“在大量接触 curl 后，Twitter 宣布 MCP 死亡”，而 [@tadasayy](https://x.com/tadasayy/status/2032327227472589282) 则反驳称使用量仍在激增。更具实质性的观点来自 [@llama_index](https://x.com/llama_index/status/2032487366129233950)：当你需要确定性的、中心化维护的 API 和快速变化的事实真相时，**MCP tools** 表现强劲；而 **skills** 则是更轻量级的本地自然语言程序，但更容易出错。与此相关，[@bromann](https://x.com/bromann/status/2032554703863820325) 指出了 **Chrome v146 对 Web MCP 的新支持**，并展示了一个 LangChain Deep Agent，它可以持续浏览 X 并编译每日摘要。

- **内存（Memory）正成为 Agent 的差异化因素**：最具技术趣味的 Agent 讨论围绕 **persistent memory**（持久化内存）和自我改进展开。[@dair_ai](https://x.com/dair_ai/status/2032459951306866714) 强调了 IBM 的工作，即从 Agent 轨迹中提取可重用的策略/恢复/优化技巧，将 AppWorld 的**任务完成率从 69.6% 提高到 73.2%**，并将**场景目标从 50.0% 提高到 64.3%**，其中在困难任务上的收益最大。与此同时，[@omarsar0](https://x.com/omarsar0/status/2032465974159618452) 总结了一篇论文，将多 Agent 内存重新定义为一个**计算机体系结构问题**，涉及缓存/内存层级、一致性和访问控制问题，而不仅仅是“更多的上下文”。这直接映射到 **Hermes Agent** 等产品工作上，多条推文将其描述为一种可自托管的 Agent，能够随时间保留技能和特定于用户的记忆（[由 @abxxai 概述](https://x.com/abxxai/status/2032463531627663540)，[由 @Teknium 演示](https://x.com/Teknium/status/2032435764588646839)）。

- **Agent UX 正在向全天候、跨设备运行演进**：几次发布将 Agent 推向了更接近“作为协调器的个人电脑”的地位。**Perplexity Computer** 已推向 iOS 并支持跨设备同步，允许用户从手机或桌面启动或管理浏览器计算机任务（[公告](https://x.com/perplexity_ai/status/2032494752642568417)，[Arav 的后续](https://x.com/AravSrinivas/status/2032495364088238147)）。[@bcherny](https://x.com/bcherny/status/2032578639276159438) 展示了 **Claude Code** 的类似流程，即从手机启动笔记本电脑上的会话。Genspark 的 **Claw** 也被定位为拥有持久云端计算机的“AI 员工”（[由 @kimmonismus 总结](https://x.com/kimmonismus/status/2032501165154332711)）。共同的模式是：持久的会话状态、远程执行以及跨多个模型/工具的编排。

**推理、长上下文与系统性能**

- **Anthropic 悄然发布了本周最重要的基础设施相关更新之一**：**Opus 4.6 1M context** 已成为 Max/Team/Enterprise 用户的默认版本（[来自 @_catwu](https://x.com/_catwu/status/2032515975556509827)），并且 Anthropic 取消了 API 对长上下文的额外收费，同时去除了 beta header 的要求，并将媒体限制扩大到**每请求 600 张图片/PDF 页面**（[来自 @alexalbert__ 的详情](https://x.com/alexalbert__/status/2032522722551689363)）。最引人注目的指标是其在 1M token 下的 **MRCR v2 达到 78.3%**，多位观察者称其为长上下文的新高度（例如 [@kimmonismus](https://x.com/kimmonismus/status/2032531949571477517)）。

- **稀疏注意力（Sparse attention）优化仍在产生显著收益**：来自 [@realYushiBai](https://x.com/realYushiBai/status/2032299919999189107) 的出色系统讨论介绍了 **IndexCache**，它在 **DeepSeek Sparse Attention** 中跨层重用稀疏注意力索引信息。据报告，在质量匹配的情况下，**GLM-5 (744B)** 实现了约 **1.2 倍端到端加速**；在 **200K 上下文**的 30B 规模实验模型上，移除 **75% 的索引器**后，**prefill 加速 1.82 倍**，**decode 加速 1.48 倍**。这一点值得关注，因为它针对的是生产规模的稀疏注意力栈，且“代码改动极小”，这正是目前各大实验室所关注的实际优化。

- **KV/cache 和推理优化正在向自回归 LLM 之外的领域扩展**：[@RisingSayak](https://x.com/RisingSayak/status/2032427185345273928) 重点介绍了 **Black Forest Labs 的 Klein KV**，它将缓存的参考图像 KV 注入到后续的 DiT 去噪步骤中用于多参考编辑，声称可实现高达 **2.5 倍的加速**。在基础设施方面，[@satyanadella](https://x.com/satyanadella/status/2032515189086761005) 表示微软是首个验证 **NVIDIA Vera Rubin NVL72** 系统的云服务商，而 [@LambdaAPI](https://x.com/LambdaAPI/status/2032427317696602575) 则针对 Rubin 时代的集群推崇“裸金属优于 Hypervisor”的观点。[@__tinygrad__](https://x.com/__tinygrad__/status/2032429289443053705) 提出了一个更激进的终局方案：在 2027 年将 “exabox” 呈现为一个由 Python 驱动的巨型单一 GPU。

**Post-Training、RL 替代方案及评估研究**

- **一个极具启发性的 Post-Training 结果：随机高斯搜索（random Gaussian search）可以媲美 RL 微调**：讨论度最高的研究结论来自 MIT 相关作者分享的 **RandOpt / Neural Thickets**，由 [@yule_gan](https://x.com/yule_gan/status/2032482266773926281) 和 [@phillip_isola](https://x.com/phillip_isola/status/2032483868603822402) 转发。该研究声称：通过在预训练模型权重中加入高斯噪声并进行集成（ensembling），可以在推理、编程、写作、化学和 VLM 任务上达到与 **GRPO/PPO 相当甚至更好的性能**。他们的解释是，大型预训练模型存在于密集的局部邻域中，其中充满了有用的任务专家——即“**Neural Thickets**”——这使得 Post-Training 比标准优化直觉所暗示的要容易得多。

- **通用数据回放（Generic-data replay）和 Pre-pre-training 重新获得关注**：[@TheTuringPost](https://x.com/TheTuringPost/status/2032441644143055316) 总结了斯坦福大学关于 **Generic-data replay** 的工作，报告称其在 **Fine-tuning 期间提升了 1.87 倍**，在 **Mid-training 期间提升了 2.06 倍**，并带来了具体的下游收益，如 Agent 网页导航能力提升 **+4.5%**，巴斯克语 QA 提升 **+2%**。围绕 “Pre-pre-training” 的独立讨论表明，社区正在重新审视训练管道早期的阶段划分/混合设计，而不仅仅是 Post-Training 技巧（来自 [@teortaxesTex](https://x.com/teortaxesTex/status/2032611773308641493) 的评论）。

- **评估（Evaluation）仍是瓶颈，尤其是在真实性和搜索策略方面**：[@i](https://x.com/i/status/2032458037823483953) 分享了 **BrokenArXiv**，即使是 **GPT-5.4** 也仅能拒绝近期论文中 **40%** 被扰动的虚假数学陈述。[@paul_cal](https://x.com/paul_cal/status/2032526200766103944) 认为，这使得 GPT-5.4 在证明验证类的“胡言乱语检测（bullshit detection）”上优于 Claude，即使其他真实性基准测试结果并不一致。对于检索/搜索，**MADQA** 发现 Agent 通过暴力搜索而非文档间的策略性导航，其回答准确率已接近人类，但距离 Oracle 表现仍有约 **20% 的差距**（通过 [@HuggingPapers](https://x.com/HuggingPapers/status/2032490352502792228)）。

**开源发布、数据集与可复现性**

- **OpenFold3 的新预览版在尖端生物学标准下具有异常的完整性**：[@MoAlQuraishi](https://x.com/MoAlQuraishi/status/2032471033760903511) 发布了 **OpenFold3 preview 2**，称其在多种模态下大幅缩小了与 AlphaFold3 的差距，且不仅发布了权重，还发布了 **训练集和配置（configs）**，使其成为“目前唯一能在功能上实现从头训练和复现的 AF3 系列模型”。这种“可复现性（reproducibility）”是核心点：许多“开源”生物学模型发布在端到端可重训练方面仍有很大欠缺。

- **弱势语言的语音数据获得了实质性提升**：[@osanseviero](https://x.com/osanseviero/status/2032452729059045881) 发布了 **WAXAL**，这是一个开放的多语言语音数据集，涵盖了 **17 种非洲语言的 TTS** 和 **19 种语言的 ASR**。随后 [@GoogleResearch](https://x.com/GoogleResearch/status/2032482132619387348) 描述其包含 **2,400+ 小时** 的音频，涵盖 **27 种撒哈拉以南语言** 和 **超过 1 亿使用者**。虽然不同帖子中的具体语言/任务数量有所不同，但两者都将 WAXAL 定位为非洲语音 AI 领域罕见的、植根于社区的资源。

- **开源社区对训练数据的态度正向“支持许可性重用”倾斜**：最强硬的表态来自 [@ID_AA_Carmack](https://x.com/ID_AA_Carmack/status/2032460578669691171)，他认为开源代码是一份礼物，其价值会被 **AI 训练放大**，而不是被削弱。[@giffmana](https://x.com/giffmana/status/2032528855215276282) 和 [@perrymetzger](https://x.com/perrymetzger/status/2032543203795284218) 呼应了这一观点。最细致的反向观点来自 [@wightmanr](https://x.com/wightmanr/status/2032555294296084755)，他认为编码 Agent 可能会以某种方式绕过署名和许可期望，从而挫伤维护者的积极性，并建议制定 Agent 合规协议可能变得非常重要。

**Developer Tooling, Coding Agents, and Research Automation**

- **Coding-agent 工作流正变得更加自主且更具主见 (opinionated)**：有许多工程师从 “Copilot” 转向**多 Agent 软件工厂**的例子。[@matvelloso](https://x.com/matvelloso/status/2032502379694932178) 描述了一种配置，其中包含 **5 个 Agent** 负责代码评审/测试/安全/性能工作，另有 **2 个 Agent** 负责合并 PR 并运行回归检查。[@swyx](https://x.com/swyx/status/2032464562214293776) 将这一趋势总结为 “**Your Code is your Infra**”，而 [@gokulr](https://x.com/gokulr/status/2032304707398746584) 和 [@matanSF](https://x.com/matanSF/status/2032561391408918797) 指出 **FactoryAI** 正成为一种日益普遍的“软件工厂”层。

- **自主研究 (Autonomous research) 正在成为一个产品类别，但并非新想法**：Karpathy 的 **autoresearch** 及相关黑客松吸引了大量关注，但多条推文指出其在概念上与 **DSPy**、**GEPA** 和贝叶斯优化流水线等旧系统存在重叠。最实用的建议是 [@dbreunig](https://x.com/dbreunig/status/2032313870233321956) 为对这种迭代式自我改进风格感兴趣的人推荐了 **optimize_anything**。Together AI 也发布了 **Open Deep Research v2**，开源了其应用、评估数据集、代码和博客 ([发布链接](https://x.com/togethercompute/status/2032524281461223614))。

**热门推文 (按参与度排序)**

- **xAI 招聘重启**：[@elonmusk](https://x.com/elonmusk/status/2032341856944865487) 表示，在承认漏掉了许多优秀人才后，xAI 正在重新审查历史面试流程，并重新联系此前被拒绝的优秀候选人。
- **Claude 的图表 UI**：[@crystalsssup](https://x.com/crystalsssup/status/2032334906517536969) 发布了一条关于 Claude 新型**交互式图表** UX 的推文，获得了极高关注。
- **移动端 Perplexity Computer**：[@perplexity_ai](https://x.com/perplexity_ai/status/2032494752642568417) 在 iOS 上推出了跨设备 **Computer** 访问功能，这是本周远程 Agent 执行最清晰的产品化案例之一。
- **微软验证 Rubin NVL72**：[@satyanadella](https://x.com/satyanadella/status/2032515189086761005) 宣布 Azure 成为首个验证 **NVIDIA Vera Rubin NVL72** 的云服务商。
- **Nous / Hermes 的势头**：Hermes Agent 及其以记忆为中心的框架通过 [@Teknium](https://x.com/Teknium/status/2032435764588646839) 等人引发了广泛讨论，反映出人们对自托管、可进化的 Agent 治理工具的浓厚兴趣。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. OmniCoder-9B 模型发布与性能

  - **[OmniCoder-9B | 基于 425K Agent 轨迹微调的 9B 编程 Agent](https://www.reddit.com/r/LocalLLaMA/comments/1rs6td4/omnicoder9b_9b_coding_agent_finetuned_on_425k/)** (热度: 781): **OmniCoder-9B** 是由 [Tesslate](https://tesslate.com/) 开发的一个拥有 90 亿参数的编程 Agent，基于 [Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) 架构进行微调，该架构将 Gated Delta Networks 与标准 Attention 交替使用。它在超过 `425,000` 条精心策划的 Agent 编程轨迹上进行了训练，其中包括来自 Claude Opus 4.6 和 GPT-5.4 等模型的数据，专注于真实世界的软件工程任务。该模型具有 `262,144` token 的上下文窗口，可扩展至 `1M+`，并展示了强大的错误恢复和推理能力，例如响应 LSP 诊断和使用 minimal edit diffs。它在 Apache 2.0 许可下发布，完全开放权重。评论者强调了 Qwen3.5-9B 架构令人印象深刻的能力，指出它能够执行通常需要大得多的模型才能完成的任务。人们强烈认为像 Qwen3.5-9B 这样的小型模型代表了本地模型的未来，一些用户对未来可能推出的更大版本（如 27B 模型）表示兴奋。

    - **Qwen 3.5 9B** 正在与大得多的模型进行比较，一些用户认为它在某些任务中的表现与 100B+ 模型相当。这突显了小型模型与中型模型竞争的潜力，特别是在资源受限的本地环境中。模型以较少资源处理复杂任务的能力被视为一项重大进步。
    - 像 **Qwen 3.5 9B** 这样的小型模型面临的一个关键技术挑战是，它们倾向于在不检查的情况下重写现有代码，这在 Agent 循环中是一个问题。然而，当用作文件浏览和代码编辑的后台 Agent 时，与 70B 等大型模型的性能差距比预期的要小。主要问题仍然在于多步错误恢复，小型模型可以修复眼前的错误，但往往会忽略上游原因。
    - 训练集的分布对于像 **OmniCoder-9B** 这样的模型至关重要。虽然 425K 条轨迹看起来很广泛，但如果数据偏向 Python Web 开发等常见任务，模型在基础设施代码或不太常见的语言上的表现可能会受到限制。这突显了多样化训练数据对于确保在各种编码任务中保持强劲性能的重要性。

  - **[Omnicoder-9b 在 Opencode 中表现出色](https://www.reddit.com/r/LocalLLaMA/comments/1rsa8wd/omnicoder9b_slaps_in_opencode/)** (热度: 351): 该帖子讨论了 **OmniCoder-9B** 的性能，这是一个基于 Opus 轨迹对 `qwen3.5-9b` 进行深度微调的版本，可在 [Hugging Face](https://huggingface.co/Tesslate/OmniCoder-9B) 上获取。用户报告称，即使在仅有 `8GB VRAM` 的系统上，使用 `ik_llama` 配合 `Q4_km gguf` 格式，在 `100k` 上下文长度下也能达到 `40tps` 的惊人速度。设置涉及特定参数，如 `-ngl 999`、`-fa 1`、`-b 2048`、`-ub 512`、`-t 8`、`-c 100000`、`--temp 0.4`、`--top-p 0.95` 以及 `--top-k 20`。不过，文中也提到了一个导致全量 Prompt 重新处理的 Bug，并建议通过调整 `ctx-checkpoints` 来解决。一位评论者质疑了 OmniCoder-9B 与常规 Qwen 3.5 9B 和 35B MOE 模型之间的性能对比，特别是在 Opencode 中的工具调用方面。另一位建议将 `ctx-checkpoints > 0` 以解决全量 Prompt 重新处理的问题。

    - **OmniCoder-9B** 因其能够在消费级硬件上运行且不产生重大资源压力而受到关注，使其成为本地部署的一个极具竞争力的选择。这在 Copilot 等主要供应商日益增加配额限制（已从无限使用转为每日限制）的背景下尤为重要，凸显了避免此类限制的本地模型的价值。
    - 一位用户报告了 **Qwen 3.5 模型** 在 Opencode 中的问题，特别是它们无法利用可用工具进行 `grep`、`read` 和 `write` 等操作，而是默认使用 `cat` 和 `ls` 等基础 Shell 命令。这表明该模型在此环境下的工具调用能力可能存在局限性或 Bug。
    - 另一位用户在 TypeScript 前端测试 OmniCoder-9B 时遇到了严重问题，一个简单的格式修改导致了整个前端崩溃。这引发了人们对 OmniCoder-9B 等小型本地模型在实际应用中可靠性的担忧，特别是与提供 `1200次调用/天` 的 Qwen 3.5 Plus 更宽松的使用限制相比。

### 2. Qwen 模型系列与性能

  - **[Qwen3.5-9B 对于 agentic coding 表现相当不错](https://www.reddit.com/r/LocalLLaMA/comments/1rrw8df/qwen359b_is_actually_quite_good_for_agentic_coding/)** (活跃度: 606): **该帖子讨论了 Qwen 3.5-9B 模型在消费级 Nvidia Geforce RTX 3060（12 GB VRAM）上执行 agentic coding 任务的性能。用户尝试了各种模型，包括 Qwen 2.5 Coder 以及基于 Qwen 3 Coder 的 Unsloth 量化版本，但发现 Qwen 3.5-9B 的效果出奇地好，能够持续运行一个多小时而没有出现问题。用户强调 Unsloth-Qwen3 Coder 30B UD-TQ1_0 在代码补全方面也很有效，尽管像 `2-bit quants` 这样的大型模型速度较慢且稳定性较差。帖子建议，在有限的硬件资源下，较小的、非编程优化的 Qwen 模型版本可能表现更好。** 一位评论者指出，Qwen3.5-9B 的表现与 `gpt120b` 等大型模型相当，考虑到其规模，这一点令人印象深刻。另一位用户则报告了褒贬不一的结果，模型有时会出现严重故障，例如破坏构建系统，这表明其性能存在波动性。

    - sleepingsysadmin 强调 Qwen3.5-9B 的表现令人印象深刻，其基准测试结果大约达到 GPT-3 120B 模型的水平，尽管其体积较小。这表明它在处理任务时具有极高的效率和能力，对于考虑资源限制的开发者来说非常值得关注。
    - linuxid10t 分享了使用 Qwen3.5-9B 的毁誉参半的经历，指出虽然它表现不错，但也存在严重失败的情况，比如破坏构建系统和删除项目。这表明在处理关键编程任务时可能存在可靠性问题，尤其是与 RTX 4060 上的 LM Studio 和 Claude Code 等其他工具相比时。
    - -dysangel- 对关于低量化模型效用的持续争论发表了评论，认为尽管存在怀疑，Qwen3.5-9B 证明了此类模型确实可以执行有用的任务。这反映了 AI 社区中关于模型大小、量化与实际效用之间权衡的更广泛讨论。

## 非技术类 AI Subreddit 汇总

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 技术与政治影响

  - **[Palantir CEO 宣称 AI 技术将削弱受过高等教育、多为民主党选民的力量](https://www.reddit.com/r/singularity/comments/1rsnwhl/palantir_ceo_boasts_that_ai_technology_will/)** (活跃度: 2076): ****Palantir CEO Alex Karp** 发表了有争议的言论，暗示 AI 技术将减少“受过高等教育、通常是女性且大多投票给民主党的选民”的影响力，同时增强接受过职业训练的工人阶级男性的力量。在一次 [CNBC 采访](https://newrepublic.com/post/207693/palantir-ceo-karp-disrupting-democratic-power)中，Karp 认为 AI 瓦解了受过人文教育的选民（主要是民主党人）的经济实力，并将其转向工人阶级选民。这一声明引发了人们对 AI 被视为政治操纵工具的担忧，到 2028 年可能会使选民转向激进的反 AI 立场。讨论还涉及 AI 可能在体力劳动之前先实现智力工作（如软件工程）的自动化，强调了需要像 Universal Basic Income（全民基本收入）这样的政策来减轻经济流失。** 评论者对 Palantir 的政治中立性表示怀疑，并指出了 Karp 本人的人文背景具有讽刺意味。讨论反映了公众对 AI 社会影响及其领导者动机的广泛担忧。

    - CombustibleLemon_13 强调了 AI 对职业和白领工作的双重影响，指出自动化同样威胁着工厂和贸易类工作。随着白领职位的减少，失业工人可能会进入蓝领行业，这可能导致劳动力过剩并削弱工人的话语权。这一观点挑战了只有受过高等教育的工作才面临风险的看法。
    - mightbearobot_ 对强大 AI 技术的可及性表示担忧，认为这些技术更有可能被当前的领导层用作控制工具，而不是被广泛普及。这一评论反映了对 AI 民主化的怀疑，并暗示它可能会加剧现有的权力失衡，而不是缓解。

- **[Bernie Sanders 正式提出立法禁止建设所有新的 AI 数据中心，理由是对人类构成生存威胁。](https://www.reddit.com/r/singularity/comments/1rrjcon/bernie_sanders_officially_introduces_legislation/)** (Activity: 4564): ****Bernie Sanders** 提出了一项旨在禁止建设新 AI 数据中心的立法，称其是对人类的生存威胁。这一提案反映了重大的政策立场，可能会影响有关 AI 监管的更广泛政治讨论。该立法可能会影响 AI 技术的发展和部署，因为数据中心对于 AI 模型的训练和运行至关重要。[Source](https://youtu.be/qu2m7ePTsqY?si=zdl_cuRg22Nv_Df5)。** 评论者认为，在美国禁止数据中心可能无法阻止其在全球范围内的建设，因为像中国这样的国家可能会继续发展。其他人建议进行监管而非禁止，提议数据中心应拥有独立的电网，且不应加剧当地环境问题。还有一种观点是建立类似于 CERN 的国际 AI 研究设施。


  - **[SAM ALTMAN：“我们预见未来智能将成为一种像电或水一样的公共事业，人们按表向我们购买。”](https://www.reddit.com/r/singularity/comments/1rro8ej/sam_altman_we_see_a_future_where_intelligence_is/)** (Activity: 9032): ****Sam Altman** 设想未来智能将像电或水等公共事业一样商品化，暗示人们将按量计费购买。这一表态与 OpenAI 在 2015 年提出的“为了全人类的利益推进数字智能”的使命相一致。然而，将智能类比为电力的说法遭到了批评，因为这可能暗示 AI 可能成为一种价格受限且受政府监管的公共事业，这与 OpenAI 当前的高估值和投资者预期相悖。** 评论者辩论了智能商品化的影响，一些人认为这可能会导致类似于公共事业的监管，而这可能不符合 OpenAI 的商业模式或投资者利益。

    - No-Understanding2406 指出了 Sam Altman 将智能类比为电力等公共事业的一个关键缺陷。他们指出，电力是一种商品化服务，通常受到监管，价格封顶，并由公共所有或控制。这意味着如果智能走上同样的道路，OpenAI 可能会成为一家利润空间有限的受监管公共事业公司，这与公司的高估值和投资者预期相矛盾。该评论者认为，Altman 的真实意图可能是将 OpenAI 定位得更像一家石油公司，受到的监管较少且利润更高，尽管这种说法对公众而言较难接受。


### 2. Gemini 和 Nano Banana Pro 用户体验

  - **[Gemini 的任务自动化已经到来，这太疯狂了 | The Verge](https://www.reddit.com/r/singularity/comments/1rs1r4j/geminis_task_automation_is_here_and_its_wild_the/)** (Activity: 722): ****Gemini** 作为一个任务自动化系统，已经推出了处理复杂任务的功能，如叫 Uber 或从菜单中选择商品。它通过提出澄清性问题并做出具有上下文意识的选择（例如跳过不必要的步骤或正确指定加热糕点等偏好），展示了先进的决策能力。这展示了 AI 任务自动化的重大进展，已从简单的命令执行转向细致入微的交互和决策。** 针对航空公司等企业是否会因为定价透明度问题而抵制此类自动化，存在相关争论。一位用户分享了开发用于价格比较的 Chrome 插件的经历，强调了个体可能面临来自大公司的法律挑战。

- Recoil42 描述了 Gemini 的任务自动化能力，强调了其在最少用户输入下处理复杂任务的能力。例如，在叫 Uber 时，Gemini 智能地询问了目的地详情，并通过跳过不必要的步骤简化了流程。同样，在订购咖啡和牛角面包时，它自主做出了诸如加热糕点之类的决定，展示了其相比早期版本更先进的决策能力。
- mckirkus 讨论了航空公司等企业对可能破坏其定价策略的自动化技术的潜在抵制。他分享了一个他开发的 Chrome 插件示例，该插件按每盎司价格对超市产品进行排名，并指出虽然公司可能会尝试采取法律行动来保护自身利益，但在针对个人强制执行此类行动时面临挑战。
- MarcusSurealius 批评了 Gemini 当前能力的实际效用，认为所提供的示例（如订机票）对大多数用户来说并不是日常任务。他建议了更多实际应用，如自动财务管理、报税和创建高效的购物清单，这些将为用户展示更多的日常实用性和相关性。

- **[Nano Banana Pro 的平庸化 (Enshittification)](https://www.reddit.com/r/GeminiAI/comments/1rs58vz/enshittification_of_nano_banana_pro/)** (热度: 1069)：**该帖子讨论了 Gemini 生态系统中的 Nano Banana Pro 图像生成器在 3 月 10 日后感知到的质量下降。用户报告称，该工具之前能生成清晰的 2K 图像，现在生成的输出却充满了像素点且模糊不清。这种变化被视为一种“诱导转向”（bait-and-switch）策略，即最初的高质量结果吸引了用户，随后质量却下降了。帖子配图是一个 Meme，通过香蕉的“诱导转向”对比图展示了这种衰退，象征着服务质量的下降。** 评论者认为，这种下降是更广泛商业模式的一部分，即 AI 服务最初提供高质量输出以吸引用户和媒体关注，随后为了降低成本而降低质量。他们认为这反映了对开源模型（open models）和协作项目的需求，因为专有模型可能会将利润置于用户满意度之上。

    - 讨论强调了 AI 领域一种常见的商业模式：公司最初提供强大的模型来吸引用户和媒体关注，但随后会降低功能以降低成本并最大化利润。这被视为在最初的热度消退后维持财务可持续性的战略举措。
    - 有一种观点认为，OpenAI 和 Claude 等公司实施了诸如频率限制（rate limits）之类的限制性措施来管理成本，这表明当前的定价模型是不可持续的。这反映了更广泛的行业趋势，即 AI 服务最初是易于获取的，但随着时间的推移会变得更加受限，以确保盈利。
    - 对话建议探索本地 AI 模型选项，如 'flux 2 Klein 9b'，如果用户的硬件能够支持，这些模型可以提供无限次使用。这些模型在分辨率和通过可下载的修复补丁进行定制方面提供了灵活性，为可能变得受限或昂贵的商业模型提供了替代方案。

- **[全新的 Gemini UI/UX 2.0 升级已上线！](https://www.reddit.com/r/Bard/comments/1rsnwx6/new_gemini_uiux_20_upgrade_is_here/)** (热度: 730)：**图片展示了全新的 Gemini UI/UX 2.0 升级，强调了个性化和交互式用户界面，重点在于升级到 “Google AI Ultra”。这次升级似乎强调了用户增强其 AI 能力的简化流程，尽管它带来了高昂的成本，用户提到订阅费用为 250 美元。评论反映了关于 Ultra 订阅价值的辩论，一些用户认为 Pro 版本足以满足他们的需求，特别是考虑到成本节约以及与 ChatGPT Pro 和 Claude Opus 4.6 等其他 AI 服务相当的性能。** 用户对 Ultra 订阅的价值表示怀疑，指出 Pro 版本以更低的成本为大多数任务提供了足够的功能。此外，人们还担心 Gemini Pro 版本中可能出现的广告，这可能会降低其效用。

- **IfNightThen** 讨论了从 Ultra 降级到 Pro 的性价比，强调每月可节省 `$220`。他们指出，唯一的重大损失是无法访问 'Deep Think' 和 'agents mode'，而这些在其他地方正逐渐成为标准功能。他们建议对于媒体生成，按月计算使用 VertexAI 可能更经济，并发现 Pro 的存储空间足以满足其需求。
- **Appropriate-Heat-977** 对 Gemini 的定价策略表示担忧，指出即使订阅了 Pro，用户仍被提示升级到 $250 的 Ultra 订阅以访问 Gemini，并可能面临速率限制。这表明访问高级功能的成本结构可能过高，可能不符合用户预期。
- **IfNightThen** 还将 'Deep Think' 的性能与 ChatGPT Pro 和 Claude Opus 4.6 等竞争对手进行了对比，结果并不理想，指出尽管定价昂贵，但它通常需要多次重试才能有效运行。这突显了 Gemini 的高级功能与其他 AI 模型相比可能存在的性能问题。

### 3. AI 模型与基础设施讨论

- **[显著增强：Qwen 3.5 40B dense, Claude Opus](https://www.reddit.com/r/Qwen_AI/comments/1rsa7h0/drastically_stronger_qwen_35_40b_dense_claude_opus/)** (活跃度: 273): **该帖子介绍了 **Qwen 3.5 40B Claude Opus**，这是一个定制构建和调优的模型，是 33 个微调 Qwen 3.5 模型合集的一部分。该模型专注于高推理能力，使用了一个获得超过 `325 个赞` 的数据集。Repository 已更新以包含所使用的数据集。该模型属于一个包含各种尺寸和配置的系列，例如 27B dense 模型，已被用户定制。**Architect series** 使用 XML 工具描述，而 **Holodeck** 模型配置为 Instruct mode，并带有星际迷航（Star Trek）主题。**Qwen3.5-27B-Engineer-Deckard-Claude** 模型的基准测试包括 `arc: 0.668`、`perplexity: 3.674 ± 0.022` 以及其他不同任务的指标。qx86-hi 量化公式采用 8/6 bit 混合精度，表现优于直接的 q8。** 评论者对模型定制的技术细节感兴趣，例如层复制（layer duplication）和量化策略。此外，人们还对这些模型在特定任务中的潜在应用和性能，以及它们对 r/LocalLlama 等社区的吸引力感兴趣。

    - StateSame5557 讨论了从 Qwen 3.5 40B 和 Claude Opus 系列创建定制模型的过程，强调了 Claude 似乎更偏好的 XML 工具描述的使用。他们提供了 Hugging Face 上各种模型的链接，例如 Qwen3.5-40B-Holodeck-Claude，这是一个带有独特 Prompt 设置的 Instruct mode Architect 模型。评论还包括了不同模型的基准测试结果，例如 Qwen3.5-27B-Engineer-Deckard-Claude，它是基于 Philip K. Dick 作品训练的模型与 Claude 模型的合并（merge），使用了混合 8/6 bit 设置的 qx86-hi 量化公式。
    - StateSame5557 提供了多个详细的基准测试结果，包括 Qwen3.5-27B-Architect-Deckard-Heretic 和 Qwen3.5-27B-Text，指标涵盖 arc, boolq, hswag 等。Qwen3.5-27B-Engineer-Deckard-Claude 模型因其独特的量化方法而备受关注，该方法使用混合 8/6 bit 公式，据称性能优于直接的 q8。qx86-hi 模型的 perplexity 为 3.674 ± 0.022，表明其在处理语言任务方面的高效性。
    - Charming_Support726 询问这些模型是否适用于网络安全应用，特别是蓝队（Blue Team）和红队（Red Team）测试。他们对可以针对此类目的进行 “abliterated” 的模型表现出兴趣，表明在网络安全背景下需要强大且适应性强的 AI 模型。

# AI Discords

遗憾的是，Discord 今天关闭了我们的访问权限。我们将不会以这种形式恢复它，但我们会很快发布新的 AINews。感谢阅读到这里，这是一段美好的历程。