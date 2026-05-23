---
companies:
- openai
- claude
- deepseek
- gemini
- qwen
date: '2026-05-04T05:44:39.731046Z'
description: '**2026年5月4日至5月5日的AI新闻要点**显示，AI产品开发重点正在发生转向，目前更加强调**“模型 + 配套工具 (harness)
  + 工作流 + UI + 记忆 + 经济性”**的综合体，而不再仅仅追求模型质量。


  **OpenAI Codex** 和 **Claude** 发布了显著更新，包括 **Appshots**、**自动模式 (auto mode)** 以及 **Sonnet
  4.6** 等新功能。**DeepSeek** 对市场产生了重大影响，将其 **DeepSeek-V4-Pro** 永久降价 75%，与 **Gemini 3.1
  Pro**、**GPT-5.5** 和 **Claude Opus 4.7** 相比，极大地提升了性价比。


  与此同时，**Gemini 3.5 Flash** 在基准测试中表现有所提升，但在实际应用价值方面收到的反馈褒贬不一。随着**通义千问 (Qwen)** 和其他中国前沿模型的发力，竞争格局正进一步加剧。'
id: MjAyNS0x
models:
- codex
- deepseek-v4-pro
- gemini-3.5-flash
- gemini-3.1-pro
- gpt-5.5
- claude-opus-4.7
people:
- gdb
- dzhng
- signulll
- teortaxestex
- ajambrosino
- reach_vb
- theo
- claudedevs
- _mohansolo
- artificialanlys
- scaling01
- yuchenj_uw
- kimmonismus
- officiallogank
- designarena
- alezander907
- giffmana
- jeremyphoward
- hamelhusain
title: 今天没发生什么特别的事。
topics:
- model-performance
- cost-curves
- agent-products
- workflow-optimization
- product-differentiation
- benchmarking
- model-optimization
---

**平静的一天。**

> 2026年5月4日至5月5日的 AI 新闻。我们检查了 12 个 subreddits，[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有新增的 Discord。[AINews 网站](https://news.smol.ai/) 允许你搜索所有往期内容。提醒一下，[AINews 现在是 Latent Space 的一个板块](https://www.latent.space/p/2026)。你可以[选择加入/退出](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)邮件接收频率！

---

# AI Twitter 回顾


**Agent 产品、Harness 以及超越“纯模型”的转变**

- **产品层面正在向栈上层移动**：一个反复出现的主题是，单靠模型质量已不再是护城河；获胜的产品越来越多地是 **model + harness + workflow + UI + memory + economics**。[@gdb](https://x.com/gdb/status/2057670776803996110) 直截了当地说：“模型本身已不再是产品，” 而 [@dzhng](https://x.com/dzhng/status/2057748510947082539) 认为顶尖产品需要 **model <> harness <> product 共生**。同样的模式也出现在实践中：[@signulll](https://x.com/signulll/status/2057850735048458639) 将 ambient AI 和 agentic AI 描述为计算接口的新接缝，而 [@teortaxesTex](https://x.com/teortaxesTex/status/2057770692112798209) 指出，harness 研究仍面临收敛于“复制 Claude Code”而非探索更广泛接口的风险。
- **编程 Agent 的产品差异化正变得具体化**：OpenAI 通过 [“codex thursday no. 6”](https://x.com/ajambrosino/status/2057716220963803577) 发布了另一个实质性的 Codex 更新，包含了 **appshots、/goal 改进、锁定时的远程电脑使用、注释模式、插件共享和分析功能**。[@gdb](https://x.com/gdb/status/2057802037757157838) 另外强调了 **Appshots**，而用户报告了显著的工作流转变：[@gdb](https://x.com/gdb/status/2057704270531903811) 表示很难想象在没有 Codex 之前是如何编程的，[@reach_vb](https://x.com/reach_vb/status/2057830243201622368) 则称他们已经一个多月没打开过 IDE 了。但产品打磨仍有欠缺：[@theo](https://x.com/theo/status/2057960907997876412) 称赞 **T3 Code 的远程功能**领先于其他替代方案，随后在另一份[帖子](https://x.com/theo/status/2057961165175873930)中将其与 Codex 中存在漏洞的远程工作流进行了对比。在 Claude 方面，[@ClaudeDevs](https://x.com/ClaudeDevs/status/2057946803685974482) 将 **auto mode** 扩展到了 Pro 订阅计划，并增加了 **Sonnet 4.6** 支持；[@_mohansolo](https://x.com/_mohansolo/status/2057910616153882949) 在用户抗议后，也不得不对 **Antigravity 2.0** 中的 IDE 支持进行澄清和补丁。

**模型性能、成本曲线与前沿竞争**

- **DeepSeek 的调价举措是最大的市场信号**：[@deepseek_ai](https://x.com/deepseek_ai/status/2057854261699195173) 将 **DeepSeek-V4-Pro 的 75% 折扣优惠永久化**，引发了强烈反响，因为它实质性地改变了 **成本/性能前沿 (cost/performance frontier)**。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2058021452465799403) 量化了其官方定价：**输入 $0.435/M，输出 $0.87/M，缓存输入 $0.0036/M**，估算混合成本约为 **~$0.18/M**，使 V4 Pro 处于智能与运行成本的帕累托前沿。他们估计在 V4 Pro 上运行其 Intelligence Index 的成本比 **Gemini 3.1 Pro Preview 低约 3 倍，比 GPT-5.5 低约 12 倍，比 Claude Opus 4.7 低约 19 倍**。社区反应集中在 DeepSeek 推动“**智能廉价到无需计量 (intelligence too cheap to meter)**”这一趋势上，正如 [@scaling01](https://x.com/scaling01/status/2057835507858518178) 所言。[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2057855546460676410) 和 [@kimmonismus](https://x.com/kimmonismus/status/2057868472965640194) 都强调了此次降价的幅度之大。
- **Gemini Flash 有所提升，但使用反馈褒贬不一**：[@OfficialLoganK](https://x.com/OfficialLoganK/status/2057682092583227881) 报告称 **Gemini 3.5 Flash** 在 **GDPval** 上相比 **3.1 Pro** 取得了重大进展，声称 Flash 现在“正在前沿领域竞争”，[@Designarena](https://x.com/Designarena/status/2057885688125968660) 将其在 Design Arena 上的排名列为 **第 16 位**，相比 Gemini 3 Flash Preview **提升了 16 个名次**。但一些开发者对其有用性与基准测试增益的关系提出了质疑：[@Alezander907](https://x.com/Alezander907/status/2057686331380359566) 发现其在成本更高的情况下，browser-agent 仅有轻微改进；[@giffmana](https://x.com/giffmana/status/2057714729762627950) 认为如果品牌依然暗示廉价，这就称不上是 “Flash 的进步”；[@jeremyphoward](https://x.com/jeremyphoward/status/2057923197639840033) 表示该模型给人的感觉是**为了刷高评测分数而优化，而非为了与人类协作**。这与 [@HamelHusain](https://x.com/HamelHusain/status/2057875320011882923) 对评测的广泛怀疑相一致，他认为当前的工具过轻看待定性的、HITL (人机回环) 的判断。
- **Qwen 和中国前沿模型不断缩小竞争差距**：官方 [@Alibaba_Qwen](https://x.com/Alibaba_Qwen/status/2057767604048240987) 的预热和来自 [@ZhihuFrontier](https://x.com/ZhihuFrontier/status/2057772126162354660) 的详尽第三方评论将 **Qwen3.7-Max** 描述为一个有意义的进步，特别是在 **指令遵循、上下文可靠性和稳定性** 方面，尽管仍存在 **冗长和 Token 使用量高** 的问题。此外，[@scaling01](https://x.com/scaling01/status/2057937081070944709) 声称最近的 ALE-Bench 运行显示，中国的模型如 **Kimi-K2.6, DeepSeek-V4, GLM-5.1** 在该场景下的表现优于多个西方发布的模型。[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057914437156409577) 还报告称，在 Coding Agent 基准测试中，**Cursor Composer 2.5** 的成本比 **Opus 4.7 便宜 3-18 倍**，比 **GPT-5.5 便宜 5-32 倍**，且 Token 使用量显著更低。

**协议、基础设施和 Agent 运行时工具**

- **MCP 的新发布候选版本是对协议的重大简化**：[@dsp_](https://x.com/dsp_/status/2057780712187580924) 宣布了 **MCP 2026-07-28 release candidate**，其核心变化是协议现在变为**无状态（stateless）**：**无需握手，没有会话 ID，且任何请求都可以命中任何服务器实例**。该 RC 版还引入了 **MCP Apps** 和 **Tasks** 等**一等扩展（first-class extensions）**，并加强了身份验证以及制定了更清晰的弃用政策。对于基础设施团队来说，无状态是一个重大的运维转变：更容易扩展、更简单的负载均衡，以及更少的粘性会话（sticky-session）担忧。
- **沙箱和托管执行正成为一等原语（first-class primitives）**：[@_philschmid](https://x.com/_philschmid/status/2057833963633418426) 演示了 **Gemini Managed Agents + Interactions API**，为 Agent 提供了一个具有内存和代码执行能力的托管安全 Linux 沙箱。[@CoreWeave](https://x.com/CoreWeave/status/2057852737073942634) 推出了 **CoreWeave Sandboxes** 的公开预览版，用于 **RL、Agent 工具使用和模型评估**，而 [@cnakazawa](https://x.com/cnakazawa/status/2057823910574588238) 发布了 **Cloudsail**，可为每个任务提供 Cloudflare 沙箱，支持 Shell、Codex 和 GitHub 访问且无需暴露 Token。在编排层，[@skypilot_org](https://x.com/skypilot_org/status/2057854003648598312) 认为 **RL 无法在 Slurm 上运行**，因为现代 RL 是一个具有异构硬件和恢复需求的多服务系统。
- **开源测试框架（harnesses）和内存层正在激增**：[@NVIDIAAI](https://x.com/NVIDIAAI/status/2057855521193881773) 开源了 **AI-Q agent skills**，用于可移植的深度研究流水线，可以接入任意测试框架。[@Teknium](https://x.com/Teknium/status/2057880570160701852) 为 Hermes 增加了 **Bitwarden 支持**以进行密钥管理，随后在 [此处](https://x.com/Teknium/status/2057930638632812642) 为 Hermes 中的 **Grok Build v0.1** 恢复了 **256K 上下文**。[@shannholmberg](https://x.com/shannholmberg/status/2057821004676956586) 描述了 Hermes Agent 下的一个**共享内存“gBrain”层**，具有类型化文件夹和专为专家级 Agent 准备的读取优先访问权限。[@aakashadesara](https://x.com/aakashadesara/status/2057809590616461399) 更新了 **CTOP** 以支持 **Devin**，并提供了一个用于列出、搜索和杀死 Agent 会话的 CLI。

**研究：RL、蒸馏、架构与评估**

- **RL 后训练与奖励设计正处于积极的反思中**：[@RyanBoldi](https://x.com/RyanBoldi/status/2057847412819906658) 介绍了 **Vector Policy Optimization (VPO)**，认为 RL 过程中的标量奖励崩溃会破坏推理时扩展（test-time scaling）。VPO 转而优化 **向量值奖励（vector-valued rewards）**，即便在原始标量目标上也能提升搜索性能。[@lateinteraction](https://x.com/lateinteraction/status/2057854814395019623) 将其视为一种为更多元的环境和目标训练 LLM 的方法，而 [@FeiziSoheil](https://x.com/FeiziSoheil/status/2057889865362993561) 则将其与转向 **结构化反馈（structured feedback）** 而非单一奖励数值的大趋势联系起来。此外，[@jsuarez](https://x.com/jsuarez/status/2057828106023703037) 预告了一个针对涉及极端稀疏性的长期 RL 问题的解决方案，初步测试显示其在某个内部环境上达到了 SOTA。
- **Agent 编译/蒸馏正在成为一个严肃的经济学命题**：[@dair_ai](https://x.com/dair_ai/status/2057846601843146760) 重点介绍了一篇论文，展示了 **全智能体工作流（full agentic workflow）**——包括多步调用、工具使用、草稿本（scratchpads）、决策结构——可以被 **蒸馏进权重（distilled into weights）**，并以 **约低 100 倍的推理成本（inference cost）** 运行，同时保持接近前沿水平的质量。这是迄今为止关于将昂贵的运行时 Agent 循环编译为更便宜的可部署模型的最清晰技术论据之一。
- **架构研究在 vanilla Transformer 之外依然活跃**：[@ChunyuanDeng](https://x.com/ChunyuanDeng/status/2057826955236462715) 介绍了 **LT2**，这是一种 **线性时间循环 Transformer（linear-time looped transformer）**，结合了稀疏注意力和线性注意力使循环变得实用，并推出了蒸馏后的 **Ouro-hybrid-1.4B**。[@ZyphraAI](https://x.com/ZyphraAI/status/2057854519732847029) 分享了将 **Equilibrium Propagation** 从基于能量的模型扩展到生物学真实神经元的研究。在 MoE 方面，[@Jianlin_S](https://x.com/Jianlin_S/status/2057719868917793221) 提出了 **Moving Quantile Balancing**，用于实现 **无损失惩罚的序列级负载均衡（sequence-level load balancing）**。同时，[@allen_ai](https://x.com/allen_ai/status/2057838486204326078) 发布了 **ArtifactLinker**，它可以在运行测试之前预测模型可能在哪些 Benchmark 上达到 SOTA——这在测试集泛滥的背景下是一个有用的元评估（meta-eval）工具。
- **数学与推理能力的讨论再次发生转向**：[@cozyblaze265065](https://x.com/cozyblaze265065/status/2057739317649588558) 报告称，使用具有中等推理能力的 **gpt-5.5** 且不借助工具，在多位数乘法实验中达到了 **99.46%** 的准确率；[@teortaxesTex](https://x.com/teortaxesTex/status/2057826903721951273) 指出现代 LLM 现在可以在不借助工具的情况下进行 **100 位数乘法**。这虽然不是一套完整的推理理论，但进一步削弱了“自回归（autoregression）无法做算术”的陈旧论点。

**多模态系统：视频、语音、世界模型（World Models）与成像**

- **Google 的 I/O 技术栈向持久化 Agent 和世界模拟器演进**：[@Google](https://x.com/Google/status/2057841803550683336) 推出了 **Gemini Spark**，一个用于处理重复性任务、技能和工作流的 **24/7 全天候个人 AI Agent**。[@GoogleDeepMind](https://x.com/GoogleDeepMind/status/2057842131142590512) 还发布了 **Project Genie + Street View**，允许用户将真实的美国地理位置转化为交互式世界；后续帖子确认该功能将通过 Google Labs 向 **Google AI Ultra** 订阅用户推出。在多模态方面，[@Google](https://x.com/Google/status/2057881884219035752) 宣布了用于对话式视频创建/编辑和自定义数字人的 **Gemini Omni**，而 [@emollick](https://x.com/emollick/status/2057874739817808223) 则强调了能够原生编辑视频的**全多模态（fully multimodal）**系统的重要性。
- **Runway 和图像/视频工具持续提升可编辑性**：[@runwayml](https://x.com/runwayml/status/2057826728769134599) 发布了 **Aleph 2.0**，支持**长达 30 秒、1080p 的多镜头序列**，并能进行保留场景其余部分的针对性编辑。[@CuriousRefuge](https://x.com/CuriousRefuge/status/2057920807389806699) 重点介绍了 **SeeDance 2 Stitcher**，用于使用 Omni 生成的连续内容无缝扩展 AI 生成的电影剪辑。
- **语音和图像生成领域出现显著跨越**：[@ArtificialAnlys](https://x.com/ArtificialAnlys/status/2057878247782908109) 将 **Cartesia Sonic-3.5** 评为其 Speech Arena 中新的 **#1 TTS 模型**，理由是其 **Elo 评分为 1218**、支持 **42 种语言**，并具有极强的自然度和文本遵循能力。Cartesia 在[此处](https://x.com/cartesia/status/2057880195403800633)宣称其生产环境的**端到端首字音频延迟仅为 82ms**。在图像生成方面，[@wildmindai](https://x.com/wildmindai/status/2057797994242523317) 关注了腾讯的 **Z-Image 6B**，这是一个**无 VAE** 的**像素空间生成器**，具有 **1K 分辨率**和用于转换 Flux/SD 模型的迁移框架；相关的生态工作包括来自 [@victormustar](https://x.com/victormustar/status/2057752615396557225) 的 Pixal3D 演示，以及来自 [@ostrisai](https://x.com/ostrisai/status/2057931161889095928) 的 AI Toolkit 对 **Z-Image L2P 1k** 的训练支持。

**安全、网络与政策压力**

- **网络安全正迅速成为高级 Agent 的试验场**：[@AnthropicAI](https://x.com/AnthropicAI/status/2057909102542549503) 表示 **Project Glasswing** 及其合作伙伴在一个月内发现了关键软件中 **10,000 多个高级或关键严重程度的漏洞**，并明确警告行业需要适应像 **Claude Mythos Preview** 这样的模型所能发现的漏洞规模。安全产品化紧随其后：[@perplexity_ai](https://x.com/perplexity_ai/status/2057869990536360334) 开源了 **Bumblebee**，这是一个用于 macOS/Linux 的只读扫描器，用于检测风险包、扩展和 AI 工具配置；[@AravSrinivas](https://x.com/AravSrinivas/status/2057873563156402448) 表示企业级部署将需要 **Agentic 沙箱**以及持续的安全工程。
- **美国移民政策的变化引发 AI 领袖的强烈抵制**：几篇高互动帖子认为，一项强制绿卡申请者必须在美国境外申请的拟议规则将直接损害 AI 人才储备。参见 [@Nick_Davidov](https://x.com/Nick_Davidov/status/2057842593850118286)、[@AndrewYNg](https://x.com/AndrewYNg/status/2057907324380217821)、[@theo](https://x.com/theo/status/2057911377151582437)、[@garrytan](https://x.com/garrytan/status/2057958284410380793) 和 [@togelius](https://x.com/togelius/status/2057912236262453607)。共同观点是：该规则惩罚了**合法的海外高技能移民**，削弱了初创企业和研究机构，并损害了美国在 AI 领域的竞争力。

**热门推文（按参与度排序）**

- [@deepseek_ai 关于将 V4-Pro 折扣永久化](https://x.com/deepseek_ai/status/2057854261699195173) —— 本批次中关于 **LLM 推理经济学**最清晰的单一市场信号。
- [@gdb 关于“模型本身不再是产品”](https://x.com/gdb/status/2057670776803996110) —— 对当前 **Agent/外壳产品论题**的简洁阐述。
- [@AnthropicAI 关于 Glasswing 发现 10,000+ 关键漏洞](https://x.com/AnthropicAI/status/2057909102542549503) —— **AI 驱动的网络能力**投入实际应用的最强有力数据点之一。
- [@dsp_ 关于 MCP 2026-07-28 RC](https://x.com/dsp_/status/2057780712187580924) —— 重要的协议更新：**无状态 MCP** 以及一等公民扩展（first-class extensions）。
- [@GoogleDeepMind 关于 Project Genie + Street View](https://x.com/GoogleDeepMind/status/2057842131142590512) —— 向**面向消费者的世界模型**迈出的显著一步。
- [@cursor_ai 关于开放 Cursor SDK 以支持自定义 Agent](https://x.com/cursor_ai/status/2057913121558413770) —— 对在编程 Agent 基础设施之上构建的团队非常重要。

---

# AI Reddit 内容回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Qwen 3.7 发布与 Qwen 3.6 本地性能

  - **[等待 Qwen 3.7 开源权重... 新王降临...](https://www.reddit.com/r/LocalLLaMA/comments/1tjvz6l/waiting_for_qwen_37_open_weight_the_new_king_has/)** (热度: 1217): **该[图片](https://i.redd.it/j8qkty82qj2h1.png)是来自 [Qwen 3.7 博客](https://qwen.ai/blog?id=qwen3.7)的一项基准测试/营销对比，将 **Qwen3.7-Max** 定位为在 Agent 代码编写、软件工程、MCP/工具调用、推理和知识评估方面领先的前沿模型，对比对象包括 **Qwen3.6-Plus**、**DS-V4-Pro Max**、**GLM-5.1**、**Kimi K2.6** 以及 **Claude Opus-4.6 Max**。其技术意义在于，该幻灯片将 Qwen3.7-Max 描绘成在许多基准测试上与 Claude 级别模型具有高度竞争力甚至处于领先地位的模型，尽管 **Claude Opus-4.6 Max** 在 `ClawEval` 和 `CoWorkBench` 等某些任务上似乎仍保持领先。评论者指出，这展示的是 **Max** 模型，并不一定代表更小规模或开源权重的版本，并猜测可能会有一个针对 **Strix Halo** 等本地硬件的 `3.7-122B-A17B` `MXFP4` 模型，拥有 `512k` 上下文。** 主要争论点在于对开源权重的怀疑：评论者指出 **Qwen 历史上从未开源过 Max 系列**，因此标题中“等待开源权重”的说法可能并不现实。其他人则警告说，不要指望假设的 `27B` 模型能达到图中所示的 Max 级别基准测试结果。

    - 几位评论者将 **Qwen Max** 与可能的开源版本区分开来，指出 *“Qwen 从未开源过 Max 系列”*，并警告不要指望较小的 `27B` 变体能匹配 Max 级别的基准测试性能。隐含的技术结论是，任何公开/开源权重的 Qwen 3.7 版本可能与参与基准测试的旗舰模型采用不同的架构或规模。
    - 一个技术愿望清单集中在假设的 **Qwen 3.7 `122B-A17B` MTP MXFP4** 模型上，拥有 `512k` 上下文，评论者认为这将非常适合 **Strix Halo** 级别的本地硬件。另一位用户提到了 **Qwen 3.5 `397B-A17B` NVFP4**，声称它可以在 `4x RTX 6000 Pro` GPU 上运行，并有足够的显存余量支持大约 `10` 个并发的 `200k` token 会话，如果 Qwen 3.7 能达到报道的基准测试水平，它将被定位为潜在的“家用版 Opus”。
    - 一位评论者认为，前沿模型的开源权重发布可能性可能变小，因为能力极强的本地模型会削弱供应商的变现能力。他们声称 Qwen 的策略已从颠覆转向货币化的前沿竞争，这可能会影响像 `397B-A17B` 这样的大型 MoE 模型是否会公开开源。

  - **[Qwen3.6 35Ba3 改变了我的工作流，甚至改变了我使用电脑的方式](https://www.reddit.com/r/LocalLLaMA/comments/1tjwrp7/qwen36_35ba3_has_changed_my_workflows_and_even/)** (热度: 567): **这篇文章描述了一个使用 **Qwen3.6 35B a3** 通过 `pi` 实现的本地 Agent 工作流，用户将可重复的过程转换为由 Codex 生成/记录的“技能（skills）”，然后将其重用于 VPS DevOps、`docling` PDF→EPUB 转换、Playwright 测试、代码工单以及 OS 级的 shell 任务。一个具体的例子：WhatsApp 音频 → AnythingLLM 转录 → `content.md` → 本地生成的落地页，然后由一个“管理” `pi` 进程执行 `plan.md` 工单队列，该进程通过 `pi -p @plan.md "Check the first Ticket with Status UNDONE and do it"` 派生出新鲜上下文的子 Agent，标记工单为 `DONE`，通过 git 提交，最后通过 VPS 技能进行部署。** 评论者关注的是操作层面的问题：什么样的硬件可以运行这种配置，Agent 在拥有 OS 访问权限时是否经过沙箱处理/值得信赖，以及与 Hermes 等其他 Agent 工具相比，`pi` 的采用难度如何。

    - 一位用户报告说，通过 **Unsloth Studio** 在一台配备 **24GB RTX Pro 4000 Blackwell SFF GPU** 的 **MS-02** 上运行 `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`，可以稳定达到 **`>100 tokens/s`**。他们将其性能与 **Mac Studio M2** 上“未优化的 GGUF”进行了比较，将 MS-02 用作 Mac 工作站的小型远程 GPU 服务器，并指出 **Unsloth 未来对 MLX 的支持** 可能会提升 Mac 端的性能。截图：[preview.redd.it](https://preview.redd.it/exwng3d4ik2h1.png?width=3966&format=png&auto=webp&s=03bf5de53b529f1b26f669c21834d9f1d69d16e0)。

- **[在 Qwen3.6 35B A3B 和 ik_llama.cpp 上利用 12GB VRAM 实现 110 tok/s](https://www.reddit.com/r/LocalLLaMA/comments/1tjh7az/110_toks_with_12gb_vram_on_qwen36_35b_a3b_and_ik/)** (热度: 565): **该帖子在 RTX 4070 Super 12GB + Ryzen 7 9700X 上使用 byteshape 的 [`IQ4_XS` `4.19 bpw` GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF) 对 Qwen3.6-35B-A3B MTP 进行了基准测试**。测试对比了上游 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 与 [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp)，设置参数为 `--ctx-size 131072`、`q8_0` KV cache、MTP draft max `3` 以及 `p_min=0.75`。使用相同的 [`mtp-bench.py`](https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090/) 工作负载，上游 `llama.cpp` 的平均速度为 **`89.76 tok/s`**，综合 MTP 接受率为 **`0.9393`**；而 `ik_llama.cpp` 在 `16.64s` 内平均速度为 **`110.24 tok/s`**，宣称实现了 **`23%` 的吞吐量提升**，尽管其在更新后的结果中综合接受率较低（**`0.8749`**）。原作者将实际的适配归功于 `ik_llama.cpp` 的 `--fit`/`--fit-margin 1664` 参数，并通过将 `--fit-margin` 提高到 `1792` 或 `2048` 来缓解 OOM。此外还指出，在 iGPU 上运行显示输出可以腾出几乎全部 `12GB` VRAM 用于推理。评论者主要关注可复现性：他们请求提供完整的上游 `llama.cpp` 命令，并指出最近合并了多个 MTP 相关的 PR，因此基准测试结果可能很大程度上取决于构建日期。针对单显卡 CachyOS/KDE 用户，有人建议了一个技术权宜之计：使用 `LIBGL_ALWAYS_SOFTWARE=1` 和 `GALLIUM_DRIVER=llvmpipe` 启动软件渲染的 Plasma Wayland 会话，将闲置 VRAM 从约 `>1024MB` 降低到 `126MB`，代价是合成器效果变慢或被禁用。

    - 一位 CachyOS/KDE Wayland 用户描述了一个针对单显卡系统的 VRAM 节省方案：创建一个自定义 SDDM 会话，通过使用 `LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe` 和 `KWIN_COMPOSE=Q` 强制 KDE Plasma 通过 CPU 渲染。据报告，KDE Wayland 的闲置 VRAM 从 **> `1024 MB`** 下降到约 **`126 MB`**，为运行 35B 模型腾出了近 1GB 的 VRAM，代价是合成器动画被禁用或运行非常缓慢。
    - 几位评论者关注报告的 `110 tok/s` 是否源于 **ik_llama.cpp** 拥有比上游 **llama.cpp** 更好的 MTP/推测性解码（speculative decoding）行为。其中一位指出，据报道 ik_llama.cpp 的接受率**从未低于 `0.790`**，而 llama.cpp 则低至 **`0.477`**，并要求提供确切的 llama.cpp 命令/设置，同时指出在过去 24 小时内有多个 MTP 相关的 PR 已合并到 llama.cpp。
    - 一位评论者询问了用于 **Qwen3.6 35B A3B** 的 `IQ4_XS` 量化，指出它似乎是内存占用最低的 Q4 量化版本，并请求了解其对模型质量/智能的影响以及最终的 VRAM/RAM 分配细节。这突显了 12GB VRAM 运行时的关键权衡：是通过激进的量化来适配模型，还是保持推理质量并避免过度的 CPU/RAM 卸载（offload）瓶颈。

### 2. 开源 AI 融资与法律压力

  - **[Heretic 已收到 Meta, Inc. 的法律通知](https://www.reddit.com/r/LocalLLaMA/comments/1tjmvx6/heretic_has_been_served_a_legal_notice_by_meta_inc/)** (热度: 2705): **Heretic Free Software Project** 声称收到了代表 **Meta Platforms, Inc.** 的供应商发来的电子邮件法律通知，并已从 Heretic 控制的仓库中删除了 Meta **Llama** 模型权重的衍生品。该项目还宣布了一个正式的德国托管 [Codeberg 镜像](https://codeberg.org/p-e-w/heretic)，并表示正在研究“技术措施”，以便在不依赖单一托管提供商的情况下保留对 Heretic 创建的模型的访问；该帖子讽刺地引用 Llama 为“前 200 名最佳”模型之一，在 [LM Arena](https://lmarena.ai/) 排行榜上“仅落后于 `168` 个其他模型”。热门评论集中在该帖子的讽刺意味上，特别是“落后 `168` 个其他模型”的排行榜梗，并批评了 Meta 的执行行为，理由是 Meta 在模型训练中涉嫌使用了盗版书籍或受版权保护的材料。

    - 一位评论者强调了法律答复的措辞，该措辞将 **Meta 的 Llama 系列** 置于当前的开源/模型竞争背景下：它被描述为在 **LM Arena** 排行榜中排名前 `200`，但落后于来自 `23` 个竞争对手的 `168` 个模型。提出的技术含义是，Meta 的命名强制执行姿态（naming-enforcement posture）与 Llama 相对的基准测试排名以及近期模型发布速度的感知放缓形成了对比。

  - **[DeepSeek 正在推进 102.9 亿美元的融资轮，梁文锋承诺将继续开发开源 AI 模型而非追求短期商业化目标](https://www.reddit.com/r/LocalLLaMA/comments/1tkfvvj/deepseek_is_pushing_forward_with_1029_billion/)** (热度: 797): 据 [Bloomberg](https://www.bloomberg.com/news/articles/2026-05-22/deepseek-founder-declares-agi-goal-as-10-billion-round-advances) 报道，**DeepSeek** 据传正在推进一轮 **`$10.29B` 的融资**，创始人**梁文锋**重申了**以 AGI 为导向的路线图**，并承诺继续发布/开源 AI 模型，而不是优先考虑短期商业化。评论者认为这是一场战略赌注，即模型优势的半衰期很短，开源研究可以比封闭的人才/模型护城河更快地加速迭代。热门评论认为，本地推理用户只是极少数，因此发布权重不会实质性损害 OpenAI、Anthropic、Google 或 Mistral 等实验室的 SaaS/API 收入；任何架构领先优势的保质期估计约为 `~1 年`。另一位评论者表示，开源模型在编程辅助方面已经“足够好”，达到了 **GLM 5.1** 级别的能力，下一个前沿是将类似的能力压缩到更小、更快、更高效的模型中。

    - 评论者认为模型权重的技术/商业保质期很短：架构优势可能仅维持 ~`1 年`，而本地推理用户与托管 API 用户相比只是极少数。其观点是 **OpenAI, Anthropic, Google, Mistral 等** 可以发布权重而不会实质性损害收入，因为大多数用户缺乏硬件或兴趣在本地运行哪怕是一个 `9B` 模型。
    - 一个技术讨论串将当前的开源模型定性为在编程辅助方面达到了“足够好”的能力，并引用 **GLM 5.1** 作为门槛模型。根据该评论，剩下的重点不是原始智能，而是蒸馏/压缩：在更小、更快、更高效的可部署模型中保留这种编程能力。
    - 一位评论者指出了 DeepSeek 自己的报告，称他们正在努力增加多模态能力：[DeepSeek_V4.pdf](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf)。值得注意的技术角度是，尽管面临 GPU/出口制裁限制，DeepSeek 仍继续扩大模型规模，这表明在有限的硬件访问下仍在持续取得进展。



## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 工作流与 Anthropic Agent 培训

  - **[Claude Code 发布 /workflows](https://www.reddit.com/r/ClaudeCode/comments/1tkjy4u/claude_code_dropped_workflows/)** (热度: 1074): **该图片是一个简单的 Claude 品牌公告图，展示了 Claude Code 中的 **`/workflows`**。该帖子声称 Anthropic 在 `Claude Code 2.1.147` 中短暂展示了一个新的工作流系统，随后又将其从变更日志中删除。其技术意义在于将基于 LLM 的编排器替换为 `workflow.js` 代码驱动的控制器：支持结构化阶段、并行分发、条件/循环/预算、重试、后台执行，并通过在阶段之间传递子 Agent 输出（而非通过主聊天上下文）来减少上下文窗口的 “Token 税”。图片链接：[https://i.redd.it/6tuq1a2i3p2h1.png](https://i.redd.it/6tuq1a2i3p2h1.png)。** 评论者们对这是否是一种根本性的多 Agent 模式持怀疑态度，并指向了现有的 Claude Code [Agent 团队](https://code.claude.com/docs/en/agent-teams)。其他人则认为与追求更强的新模型（如 “Opus 4.5”）相比，该功能的优先级较低。

    - 一位评论者链接了 **Anthropic 现有的 Claude Code “Agent 团队”文档** (https://code.claude.com/docs/en/agent-teams)，并指出所描述的 `/workflows` 模式——*“一个主 Agent (一个 LLM) 决定派生哪些子 Agent，保存每个中间结果并规划下一步”*——与已记录的多 Agent 编排概念重合。
    - 据报道，`/workflows` 功能似乎是昙花一现：一位评论者表示它早前在变更日志中可见，但 **Anthropic 随后已将其撤下**，并提供了一张已删除变更日志条目的截图镜像 (https://preview.redd.it/720w663mcp2h1.png?width=2056&format=png&auto=webp&s=d7afca73806dd159eff3141db0f61de5a37526a8)。
    - 一位用户将该功能与他们围绕 **skills + YAML + JavaScript CLI** 构建的自定义编排栈进行了比较，暗示 `/workflows` 可能会将开发者已经在为可重复的 Claude Code 任务流水线手动实现的模式正式化。

  - **[Anthropic 正式发布 13+ 门带证书的免费 AI 课程（包括 Agentic AI 和 Claude Code！）](https://www.reddit.com/r/ClaudeAI/comments/1tjpfh8/anthropic_officially_launched_13_free_ai_courses/)** (热度: 2547): **Anthropic** 通过其基于 Skilljar 的学院提供免费的官方培训目录，可通过 [Anthropic Learn](https://www.anthropic.com/learn) 访问。课程涵盖 **Claude**、**Claude Code**、**Claude API**、**MCP / Agentic 工作流**，以及 **Amazon Bedrock** 和 **Google Cloud Vertex AI** 的部署路径，并提供证书。技术上值得关注的内容包括 MCP 材料，涵盖了关于 `STDIO` 和 `StreamableHTTP` 传输的高级主题，以及用于代码库编辑、测试执行和 “Plan Mode” 的 Claude Code 模块。此外还提到了一个单独的免费 [CodeSignal](https://codesignal.com/) 路径 “Developing Claude Agents”，用于交互式 Python/TypeScript 实验和证书。评论者确认 Skilljar 课程是真实的，因为它们链接自 Anthropic 的官方网站，一位完成了 `10/15` 门课程的用户特别推荐了 MCP 和高级 MCP 模块，称其 *“非常值得一试”*。

    - 几位评论者确认 Skilljar 课程是官方的 **Anthropic** 培训材料，指出课程门户链接自 [anthropic.com/learn](https://www.anthropic.com/learn)，而非第三方诈骗或转发。
    - 一位完成了 `10/15` 门课程的用户特别强调 **MCP** 和 **MCP 高级主题** 模块非常有用，称其对 Model Context Protocol 集成的 `STDIO` 和 `StreamableHTTP` 传输协议进行了实用的覆盖。
    - 一些用户指出该目录并非新发布，已经开放了数月；一位完成了两门课程的评论者形容它们 *“相当基础”*，暗示这些材料对于经验丰富的 AI 开发者来说可能更偏向入门级而非高级。


### 2. Z-Image 6B, Gemini 3.5 Flash 与 OpenAI Math 更新

- **[腾讯发布了支持像素空间生成的 Z-Image 6B。无 VAE 且达到 1k 分辨率。](https://www.reddit.com/r/StableDiffusion/comments/1tkipk6/tencent_released_zimage_6b_with_pixel_space_gen/)** (热度: 899): **这张[图片](https://i.redd.it/69r8ttxmvo2h1.jpeg)是 Tencent/Z-Image 6B / L2P 的示例拼图，展示了在人像、动物、幻想场景、车辆和风格化构图中的 `1024px` 级像素空间图像生成 (pixel-space image generation)，其核心技术主张是无需 VAE 即可进行生成。** 该帖子链接了项目页面 [nju-pcalab.github.io/projects/L2P](https://nju-pcalab.github.io/projects/L2P/)，一名评论者指出了 Hugging Face 上的模型文件：[zhen-nan/L2P](https://huggingface.co/zhen-nan/L2P/tree/main)。评论者主要关注架构趋势——*“看来现在大家都在追求无 VAE 了”*——并对实际质量提出质疑 *“这真的好用吗？”*，而不是提供基准测试或详细评估。

    - 一名评论者指出了 Hugging Face 上的模型文件：**zhen-nan/L2P**，地址为 [https://huggingface.co/zhen-nan/L2P/tree/main](https://huggingface.co/zhen-nan/L2P/tree/main)，这对于想要检查/下载腾讯发布的 **Z-Image 6B** 及其声称的**像素空间生成 / 无 VAE** 设置的读者非常有用。
    - 几条评论强调了向 **No-VAE / 像素空间图像生成** 转变的更广泛技术趋势，一位用户指出 *“看来现在大家都在追求无 VAE 了”*。这一点值得注意，因为避开 VAE 改变了压缩/潜空间瓶颈（latent bottleneck）的权衡，并可能影响重建忠实度、显存成本以及原生高分辨率生成（如帖子声称的 `1k` 分辨率）。
    - 一位评论者提出了与 **Lodestone** 的比较，询问腾讯的方法是否借鉴了 Lodestone 的无潜空间/低潜空间方向，或者 Lodestone 是否可以从 Z-Image 中学习。该线程没有提供基准测试数据，但技术比较显示出人们对融合开源架构以实现直接像素空间扩散/流生成（pixel-space diffusion/flow generation）的兴趣。

  - **[Google 的最新作：Gemini 3.5 Flash 对决全场](https://www.reddit.com/r/singularity/comments/1tjoarz/googles_latest_creation_gemini_35_flash_vs_all/)** (热度: 1503): **该帖子报告了 Google Gemini 3.5 Flash 在 Gemini App 中的一个简单算术错误：对于 Prompt `300+140=460` / “这正确吗？详细拆解一下？”，共享的 Gemini 运行记录据称接受了这个错误的求和结果，同时链接了 [Claude](https://claude.ai/share/8383747a-aaf1-4f6c-a516-0e839f46a698)、[Grok](https://grok.com/share/bGVnYWN5_3c63e371-eb9d-46c3-8ba2-0c745c6795a2) 和 [ChatGPT](https://chatgpt.com/share/6a0f1e13-a0c8-8328-b989-1ac51b92e81c) 的对比运行记录。评论者复现了该问题，并将其归因于 Gemini App 的推理设置：**“标准（Standard）”/默认思考模式表现得像最小化或无推理**，而据报道，**扩展思考（Extended thinking）**或具有更高思考设置的 AI Studio 会返回正确的 `300 + 140 = 440`。** 主要争论点在于，这与其说是基础模型能力的证据，不如说是产品级服务配置的问题：评论者认为 **Gemini App 相对于 AI Studio 被“削弱（nerfed）”了**，尤其是在默认/最小思考设置下。楼主（OP）认为这一结果令其标榜的 SOTA / 金融 Agent 排名感到尴尬，而其他人则认为基准测试性能可能无法反映低投入的 App 默认设置。

    - 用户报告称，这种明显的失败很大程度上取决于 Gemini 的**思考层级（thinking level）**：切换到 **Extended thinking** 可以修正答案，而 **Standard** 被描述为实际上 *“根本不思考”*。另一位评论者通过截图复现了同样的输出（[预览图](https://preview.redd.it/whzg30z8hi2h1.png?width=1557&format=png&auto=webp&s=192481783e75626c47648f50954c4c8fe8fb60a7)），并称 Gemini App 默认为类似**最小思考**的设置，而 **AI Studio** 即使在 **Low** 思考设置下也能避免错误。
    - 围绕**工具调用行为（tool-calling behavior）**提出了技术对比：一位评论者认为 Gemini 的弱点不一定是原始推理能力，而是**工具路由逻辑（tool-routing logic）**，并指出 ChatGPT 可能会将任务委托给 **Python** 处理，而不是纯粹在模型内部解决。这意味着基准测试结果可能取决于是否允许模型调用工具，以及它决定使用工具的可靠程度。

- **[数学系研究生朋友说我们完蛋了](https://www.reddit.com/r/OpenAI/comments/1tkcxxi/math_grad_student_friend_says_were_cooked/)** (Activity: 825): **这张 [图片](https://i.redd.it/l7gd5lx9in2h1.png) 是一个推文截图**，转述了一位数学系研究生对最近声称的 **Erdős 证明**的震惊反应，帖子标题为 *“数学系研究生朋友说我们完蛋了。”* 该内容**没有提供关于证明、定理陈述、模型、Benchmark 或验证过程的技术细节**；其意义在于背景和社会层面：一位数学家将该结果描述为以前“完全无法触及”，并表示 OpenAI 的公告“极其俗气且品味低劣”。** 评论区的讨论大多是非技术的，且由梗（meme）驱动，转向了关于 “OnlyFans 但针对宅男” 的笑话。一位评论者询问“极其俗气且品味低劣”是什么意思，但对于数学或 AI 能力声明并没有实质性的辩论。

    - 一位评论者认为，随着 AI 系统开始在**数学、定理证明和研究级推理（research-level reasoning）**方面展现出能力，人们认为“创造性和智力”工作是安全的观念已经动摇。技术要点在于，自动化风险可能与任务是否具有重复性没有直接关联；相反，高级推理 Benchmark 和形式化证明系统在评估 AI 影响时正变得越来越相关。

# AI Discords

不幸的是，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但我们很快就会发布全新的 AINews。感谢读到这里，这是一段美好的历程。