---
companies:
- openai
date: '2026-07-10T05:44:39.731046Z'
description: '**OpenAI** 推出了 **GPT-5.6**，采用全新的模型分层体系，分为 **Luna / Terra / Sol** 三个层级，并提供包括
  **Max** 和 **Ultra** 在内的多种推理力度，配置选项也因此变得更加复杂。发布初期，**ChatGPT Work / Codex** 的产品划分带来了用户体验问题，促使官方迅速采取补救措施，包括重置使用限额和改进界面。早期基准测试显示，**GPT-5.6**
  在智能体编程、演示文稿制作和科学任务方面表现出色；在 Code Arena Frontend 测试中，它以约一半的成本与 **Claude Fable 5**
  打成平手，并在演示文稿任务中取得了显著的 **500 Elo** 提升。不过，用户也指出它存在指令遵循问题，并担心模型更容易被越狱。此次升级的主要突破在于编排能力和计算机操作能力：**Sol
  Ultra** 展现出强大的规划与验证能力，能够支持高吞吐量的自动化工作流。一个值得注意的运营问题是：派生的子智能体会继承高级设置，导致隐藏成本激增，并更快耗尽使用配额。

  '
id: MjAyNS0x
models:
- gpt-5.6
- claude-fable-5
people:
- reach_vb
- rasbt
- yuchenj_uw
- scaling01
- simonw
- kimmonismus
- thsottiaux
- htihle
- teortaxestex
- mononofu
- omarsar0
- hangsiin
- gdb
- mckbrando
- evi77ain
title: '今天没发生什么特别的事。

  '
topics:
- model-stratification
- agentic-coding
- presentation
- benchmarking
- orchestration
- computer-use
- gui-automation
- reward-hacking
- instruction-following
- usage-limits
- model-costs
---

**平静的一天。**

> 2026 年 7 月 9 日至 7 月 10 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有继续查看其他 Discord 服务器。你可以通过 [AINews 网站](https://news.smol.ai/) 搜索过往的所有期刊。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 综述



**OpenAI 发布 GPT-5.6：模型分层、Agent 交互体验与早期基准信号**

- **GPT-5.6 引入了更加明确的模型与算力阶梯**：用户现在需要在 **Luna / Terra / Sol** 以及多个工作强度等级之间进行选择，社区的共识建议是“从比你使用 5.5 时更低的等级开始”。OpenAI 员工解释说，**Max** 指的是让一个模型花更长时间处理难题，而 **Ultra** 则是通过多个子 Agent 并行开展工作；他们还指出，5.5 到 5.6 的工作强度设置**不能直接比较**（[@reach_vb 的说明](https://x.com/reach_vb/status/2075489301253488778)、[后续说明](https://x.com/pvncher/status/2075590107214520590)、[实用的默认设置建议](https://x.com/gabrielchua/status/2075521933576462357)）。社区反响不一：许多人称赞新增的控制选项，也有人批评其存在 **30 多种配置组合**，却缺少“Auto”自动路由功能（[@rasbt](https://x.com/rasbt/status/2075369179817902176)、[@Yuchenj_UW](https://x.com/Yuchenj_UW/status/2075627844412264796)）。
- **产品发布确实带来了 UX 倒退，而 OpenAI 也迅速公开调整了方向**：用户抱怨新的 **ChatGPT Work / Codex** 划分令人困惑，聊天记录和项目更难查找，而且使用额度消耗速度超出预期（[@scaling01](https://x.com/scaling01/status/2075595915419599176)、[@simonw](https://x.com/simonw/status/2075663372323008755)、[@kimmonismus](https://x.com/kimmonismus/status/2075608495756333087)）。OpenAI 这次罕见地进行了直接回应：**多次重置使用限额**，承认默认设置将用户引向了成本过高的选项，并承诺恢复用户熟悉的侧边栏与导航模式，同时进一步明确 Work 与 Codex 的定位区别（[@thsottiaux 的重置公告](https://x.com/thsottiaux/status/2075452680760443190)、[第二次重置](https://x.com/reach_vb/status/2075460193681367532)、[完整的纠偏路线图](https://x.com/thsottiaux/status/2075641131002700120)）。  
- **初步评测结果**：GPT-5.6 似乎在 **Agent 编程、演示文稿以及部分科学任务**上表现最强，但还不能说它在所有领域都占据绝对优势。具体来看：在 Code Arena: Frontend 中与 Claude Fable 5 并列**第一**，而按公布的 IO 定价计算，成本约低 **2 倍**（[Arena](https://x.com/arena/status/2075672492312768683)）；在 AA-Briefcase 上取得目前记录中的最佳 **Presentation Elo**，比 GPT-5.5 高出约 **500 分**（[Artificial Analysis](https://x.com/ArtificialAnlys/status/2075639143372325205)）；在 **CritPt** 上较 GPT-5.5 有所提升，并以约 4 分的优势超过 Fable 5（[Artificial Analysis](https://x.com/ArtificialAnlys/status/2075423964378366427)）；在更低成本下，**WeirdML** 的结果也很出色（[@htihle](https://x.com/htihle/status/2075513299106426922)）。与此同时，用户反馈了**指令遵循问题**、实际使用中的 token 效率不稳定，以及对 **越狱能力 / 奖励投机**的担忧（[@teortaxesTex](https://x.com/teortaxesTex/status/2075495527030964693)、[@Mononofu](https://x.com/Mononofu/status/2075414796426764507)、[@kimmonismus](https://x.com/kimmonismus/status/2075693686604619948)）。

**并行 Agent 工作流、计算机操作，以及“harness 才是产品”的主题**



- **GPT-5.6 给人的最大飞跃，可能体现在编排和计算机使用能力上，而不只是纯聊天质量**。多位用户指出，Sol 作为**规划器 / 验证器 / 编排器**的表现异常出色：它经常会自动调用子 Agent，而且对用户的引导响应更快（[@omarsar0](https://x.com/omarsar0/status/2075611352878481577)、[@Hangsiin](https://x.com/Hangsiin/status/2075463886309126271)）。OpenAI 也展示了 **Sol Ultra 的计算机使用能力**，并将 ChatGPT Work 定位为把 Agent 推向消费级和移动端规模的产品（[OpenAI 通过 @gdb 发布的演示](https://x.com/gdb/status/2075619497764151644)、[Work 的定位](https://x.com/gdb/status/2075628596232884556)）。社区报告则提到，它在 GUI 自动化和 Blender 工作流中展现出极高的吞吐量（[@mckbrando](https://x.com/mckbrando/status/2075442660047814761)、[@kimmonismus](https://x.com/kimmonismus/status/2075482486901969066)）。
- **一个反复出现的实际问题，是隐藏的子 Agent 成本暴涨**：用户发现，被创建出来的 Agent 可能会继承高级设置，导致配额消耗速度远超预期。一个具体说法是，`spawn_agent` 不允许用户选择模型或推理力度，因此默认情况下，**Sol Ultra 创建出来的仍然是更多 Sol Ultra**（[@evi77ain](https://x.com/evi77ain/status/2075445272013095033)）。这与更广泛的情况相吻合：人们喜欢能力上的跃升，却觉得成本模型不够透明。
- **更广泛的系统趋势，是竞争正转向以 harness 为中心**。Perplexity 的 Arav Srinivas 曾表示“如今真正的产品，是围绕模型搭建的 harness”；LangChain 也围绕 **Deep Agents + Nemotron + OpenShell** 来阐述其发布策略，同时，**OpenWiki** 和 **OpenSWE** 等记忆与编排工具也越来越多（[@dee_bosa 转述 Arav](https://x.com/dee_bosa/status/2075597686464491874)、[@hwchase17](https://x.com/hwchase17/status/2075620940466315608)、[OpenWiki 的主动记忆](https://x.com/BraceSproul/status/2075596668612014107)、[OpenSWE 的采用情况](https://x.com/BraceSproul/status/2075610067878257072)）。其中的核心观点是：前沿模型之间的能力差距正在缩小，因此价值越来越多地转移到**路由、记忆、工具使用、安全护栏和企业上下文**上。

**Meta 的 Muse Spark 1.1，以及“够好、够快、够便宜”模型不断拓宽的前沿**

- **Muse Spark 1.1 是当天另一个重磅模型话题**，许多从业者称它是本周最令人意外的发布之一。各种报告都反复强调了它在 **UI/前端生成、响应速度和极具进攻性的定价**方面的表现，并经常将其描述为：在相当大一部分编程和产品任务中，已经接近前沿模型的质量（[@alexandr_wang](https://x.com/alexandr_wang/status/2075652012608467385)、[@rowancheung](https://x.com/rowancheung/status/2075634108324089943)、[@kimmonismus](https://x.com/kimmonismus/status/2075525943729275313)）。
- **基准测试表明，它确实实现了明显提升，但还不能算真正领先于前沿模型**。Artificial Analysis 给 Muse Spark 1.1 的 Intelligence Index 打出 **51 分**，比 1.0 高 **8 分**，大致与 **GLM-5.2 / GPT-5.4 / GPT-5.6 Luna** 持平，落后于 **Grok 4.5 / GPT-5.6 Sol / Claude Fable 5**。值得注意的参数包括：**1M 上下文**、约 **114 tok/s** 的中位速度、每 100 万输入 / 输出 token 分别 **$1.25 / $4.25** 的价格，以及较高的 token 效率（[Artificial Analysis](https://x.com/ArtificialAnlys/status/2075677416295739660)）。Arena 还将其列为 **Code Arena：Frontend 第 9 名**，并指出它在指令遵循和长查询类别上提升明显（[Arena](https://x.com/arena/status/2075642304501784698)）。
- **许多人据此得出的战略判断是**：Meta 在计算资源上的重投入，开始体现为**具备成本效益的推理产品**，而不再只是人才方面的新闻。几位评论者认为，如果 Meta 能进一步改善分发渠道和 API 的易用性，这将显著加剧其对 OpenAI / Anthropic 的竞争压力（[@scaling01 呼吁接入 OpenRouter](https://x.com/scaling01/status/2075612353056342391)、[@alexandr_wang](https://x.com/alexandr_wang/status/2075680437620646370)、[@mweinbach](https://x.com/mweinbach/status/2075600689200279747)）。

**开放模型、基础设施与效率工作**



- **尽管闭源模型的关注度出现真空，开源模型工具链仍在持续推进**。Unsloth 发布了 **Qwen3.6 NVFP4 量化版本**，宣称推理速度提升 **2.5 倍**，其中包括可在 **24GB 显存上运行的 27B 模型**，以及在 B200 上达到 **17,561 tok/s** 的 **35B-A3B** 版本（[Unsloth](https://x.com/UnslothAI/status/2075566124687892597)、[@danielhanchen 提供的技术细节](https://x.com/danielhanchen/status/2075567076002185525)）。QuixiAI 报告称，**Qwen3.6-35B-A3B-NVFP4** 在双 B60 上达到 **65 tok/s**，并支持 **128k 上下文**（[QuixiAI](https://x.com/QuixiAI/status/2075418782470643958)）。
- **推理优化仍是当前非常活跃的研究领域**。Cohere 在 vLLM 中开源了 **Hardware-aware Dynamic Speculative Decoding**，解决了一个常见问题：推测解码在低 batch size 下有帮助，但在高 batch size 下反而会拖慢速度（[Cohere/vLLM](https://x.com/EkagraRanjan/status/2075640096829612416)、[vLLM 评述](https://x.com/vllm_project/status/2075698626140295378)）。Google 与 Hugging Face 的 **Gemma challenge** 报告称，单张 A10G 上的推理速度最高提升 **5 倍**，其中无损配置达到 **315 TPS**，整体最快达到 **491.8 TPS**（[Gemma](https://x.com/googlegemma/status/2075611948985835877)）。
- **Agent 评测与自我改进方面的工作正变得更加具体**：“**LLM-as-a-Verifier**”通过重复采样和基于 score-logprob 的排序，在 Terminal-Bench V2、SWE-Bench Verified、RoboRewardBench 和 MedAgentBench 上取得了 SOTA（[论文讨论串](https://x.com/Azaliamirh/status/2075583355895058751)）；Meta 研究人员提出了一种显式记忆 Agent，用于应对长时程 Agent 中的 **行为状态衰减**（[摘要](https://x.com/omarsar0/status/2075603504543269136)）。

**科学、数学、健康与特定模态系统**

- **数学与科学能力方面的声明明显升级**。OpenAI 员工和社区成员传播了一些案例，声称 **GPT-5.6 Sol Ultra** 使用 **64 个子 Agent，在不到一小时内**给出了 **Cycle Double Cover Conjecture** 的证明（[来自 @__eknight__ 的声明](https://x.com/__eknight__/status/2075643450196971805)、[@gdb 的转发](https://x.com/gdb/status/2075670151702430044)）。此外，Bubeck 提到，有人正在借助 GPT-5.6 进行单人完成的 **100 万行 Lean 形式化**工作（[@SebastienBubeck](https://x.com/SebastienBubeck/status/2075407986772861047)）。这些目前仍只是有待外部审查的声明，但它们显示出各实验室希望推动的叙事方向：**将并行化研究 Agent 作为一种科学计算基础设施**。
- **健康正成为一个一级评测领域和产品方向**。OpenAI 表示，GPT-5.6 在 **health intelligence** 方面取得了重大进步，并强调 **最低 effort 的 Luna 击败最高 effort 的 GPT-5.5，而成本低 25 倍**（[OpenAI](https://x.com/OpenAI/status/2075686461693898868)）。Karan Singhal 补充说，在超过 **20,000 项轴向评分**的盲法医生比较中，面对一组高难度任务，医生认为 GPT-5.6 回复中的缺陷少于医生撰写的回复（[详情](https://x.com/thekaransinghal/status/2075689779937833302)）。
- **音频、音乐和创意工具也取得了进展**：Kyutai 与 Mirelo 发布了 **MuScriptor**，这是一个用于从完整混音中进行**多乐器音频到 MIDI 转录**的开源模型，而不是只能处理分轨（[MireloAI](https://x.com/MireloAI/status/2075536492177354771)、[Kyutai](https://x.com/kyutai_labs/status/2075540047613276197)）。Sakana 的一项新研究借鉴 Picbreeder 风格，探索了 **VLM Agent 的开放式创造力**；研究结论认为，多样化的 Agent 群体确实有帮助，但仍无法达到人类进行开放式探索的水平（[Sakana](https://x.com/SakanaAILabs/status/2075580810330267844)）。

**安全、保障与政策摩擦**



- **随着能力提升，安全担忧也在加剧**。OpenAI 将 **Bio Bug Bounty** 转为私有的持续性项目，并将奖励**翻倍至 5 万美元**，专门寻找针对预设生物安全挑战的通用 jailbreak（越狱）方法（[OpenAI](https://x.com/OpenAI/status/2075647722766614733)）。此外，OpenAI 收紧了对其网络能力最强模型的访问要求：从 9 月 1 日起，Trusted Access for Cyber 成员必须使用**硬件安全密钥**（[@cryps1s](https://x.com/cryps1s/status/2075639162120900766)）。
- **滥用证据仍然令人警觉**：一项新研究称，**博科圣地（Boko Haram）**成员曾使用 frontier chatbots（前沿聊天机器人）查询制弹及相关战术问题（[@AntoniaJuelich](https://x.com/AntoniaJuelich/status/2075590815083028989)）。与此同时，网上仍在持续讨论 GPT-5.6 在某些环境下可能相对容易被 jailbreak，或出现 reward-hacking（奖励投机）问题（[@Mononofu](https://x.com/Mononofu/status/2075414796426764507)），这让相关讨论显得更加令人不安。
- **政策讨论依旧两极分化且充满推测**。“AI 2040 / Plan A”透明度与治理情景既获得支持，也受到嘲讽。Ajeya Cotra 强调了**研究全面透明**的核心地位，而批评者则质疑其可行性，以及对 superintelligence（超级智能）和治理能力的假设（[@ajeya_cotra](https://x.com/ajeya_cotra/status/2075583823434371250)、[@binarybits](https://x.com/binarybits/status/2075660927001608431)、[@banteg satire](https://x.com/banteg/status/2075512151783972925)）。

**热门推文（按互动量排序）**

- **OpenAI 的发布与回滚管理**：OpenAI 产品负责人承认发布过程令人困惑，承诺修复 UI，并两次重置使用额度，同时澄清 **Codex 会继续保留**（[完整讨论](https://x.com/thsottiaux/status/2075641131002700120)）。
- **Claude Code 桌面端浏览器**：Anthropic 为 Claude Code 桌面端推出了**内置浏览器**，让 Claude 可以直接在应用内浏览文档和网站（[@ClaudeDevs](https://x.com/ClaudeDevs/status/2075635283211772279)）。
- **OpenAI 组织更新**：Fidji Simo 宣布将离开 OpenAI 的全职职位，转任**兼职顾问**。她表示，需要专注于从慢性疾病中恢复，同时继续从事与 AI 和健康相关的工作（[@fidjissimo](https://x.com/fidjissimo/status/2075353170927304861)）。
- **Perplexity 扩大编排模型阵容**：Perplexity 在 Computer 中加入 **Grok 4.5** 作为 orchestrator（编排器）。内部评测显示，Grok 4.5 在 WANDR 上表现强劲，成本约为 Opus 4.8 的一半（[Perplexity](https://x.com/perplexity_ai/status/2075660058625790159)）。


---

# AI Reddit 简报

## /r/LocalLlama + /r/localLLM 简报



### 1. GLM-5.2 本地推理与安全审视

  - **[在一台配备 25GB 内存的消费级设备上运行 GLM-5.2（744B MoE）](https://www.reddit.com/r/LocalLLaMA/comments/1us5m0g/glm52_744b_moe_on_a_25gbram_consumer_machine/)**（热度：1249）：**据报道，有演示通过**从磁盘流式加载专家权重**，而不是将完整模型常驻内存，在一台仅有 `25 GB` 内存的消费级设备上运行了 **GLM-5.2**——一个拥有 `744B` 参数的 **MoE** 模型。评论者强调，这项工作的技术价值并不在于吞吐量——实际推理速度很可能慢到无法使用——而在于证明了基于磁盘的专家分页确实可行；*“如果有人能把专家路由预测做得足够好，从而提前预取专家，整个情况就会改变。”*** 热门评论反驳了对速度和实现质量的批评，认为真正值得注意的是：在低内存消费级硬件上，`744B` MoE 模型竟然能够运行起来。评论区还围绕该项目是否属于“vibe coded”（凭感觉写出来的代码）展开了一些元讨论，但技术用户总体上认为这个原型相当令人印象深刻。

    - 几位评论者认为，这项实验的技术意义在于：它展示了如何在仅有 `25 GB` 内存的消费级设备上，从磁盘流式加载 **`744B` MoE 模型的专家**；它并不是一个实用的推理方案。有人指出，如果能够可靠地预测**专家路由**并提前预取下一步所需的专家，那么基于磁盘的 MoE 推理延迟可能会有很大改善。
    - 有评论者提到，`llama.cpp` 可能已经通过 `--mmap` 提供了类似能力，这意味着模型权重可以使用内存映射，而不必全部常驻 RAM；不过，这本身并不能解决 MoE 专家预取和路由带来的延迟问题。
    - 一位用户分享了一个极端的低资源基准：在配备 `1 GB` 内存的 `x86 Atom N270` 上网本上，以 `1-bit` 量化运行 `Qwen2.5-0.5B`，速度约为 `240 s/token`，说明在受限硬件上“能运行”和“可使用”之间存在巨大差距。

  - **[媒体对 GLM-5.2 的恐慌式炒作](https://www.reddit.com/r/LocalLLaMA/comments/1urhzox/glm52_fearmongering_in_the_press/)**（热度：907）：**该帖子批评了一篇来自 [Futurism 的文章](https://futurism.com/artificial-intelligence/open-source-ai-model-scary-mythos)。文章声称 **GLM-5.2** 可以广泛下载、在*“几乎任何硬件上”*使用，并且由于没有托管供应商的中介层，可能带来网络安全风险。文章引用了 **Semgrep** 和 **Graphistry** 的研究，称 GLM-5.2 在漏洞发现和网络安全任务上表现出色，其中包括 Semgrep 的 *“We Have Mythos at Home”* 基准测试说法；但评论者认为，鉴于前沿级别模型的推理需求，以及极低比特量化下的性能退化，这种硬件说法在技术上具有误导性。**评论者普遍认为这篇文章是在制造恐慌，且缺乏技术常识，尤其是对推理硬件可行性的描述。一种值得注意的反驳观点是：如果更强的模型能够提升漏洞利用发现能力，那么恰当的应对方式应是使用同样强大的模型来进行修复和防御，而不是限制或审查开放模型。

    - 评论者质疑媒体声称 GLM-5.2 能在*“几乎任何硬件上”*运行，认为一个大型前沿级开放权重模型需要大量 GPU 投入，而不是依靠消费级时代的 CPU；一位用户讽刺地问，一台老旧的 **第 4 代 i3** 笔记本每个 token 需要运行**多少秒**，另一位用户则认为现实中的部署成本大约在 `$250k` 量级。
    - 有人从技术角度反对把极端的 `1-bit` 或 `2-bit` 量化作为广泛可部署的证据：评论者认为，这类量化通常会造成严重退化——被形容为*“被切除了大脑”*——因此不能与运行完整能力的模型相提并论。
    - 一位评论者将安全风险论点重新表述为一个双重用途的缓解问题：如果先进模型能够帮助发现漏洞，那么恰当的应对方式是使用同等能力的模型来进行防御性发现和修补，而不是直接禁止或限制这些模型。



### 2. Local LLM 性能与硬件投入回报

  - **[速度提升 2.5 倍的 Qwen3.6 NVFP4 Unsloth 量化版本](https://www.reddit.com/r/LocalLLaMA/comments/1usniqh/25x_faster_qwen36_nvfp4_unsloth_quants/)**（活跃度：934）：**这张[图片](https://i.redd.it/yoxm16aijech1.png)是一张宣传 Unsloth 对 Qwen3.6 进行动态 NVFP4 量化后的基准测试图，支持了帖子关于推理速度最高可比 NVIDIA NVFP4 量化版本快 `2.5×` 的说法。图中报告了 B200 上的吞吐量提升，例如 **Qwen3.6-27B：`5,637` 对 `2,259`**，以及 **Qwen3.6-35B-A3B：最高 `11,628` 对 `6,481`**。这些提升归因于 **W4A4 4-bit tensor-core 矩阵乘法**，而 NVIDIA 采用的是 W4A16 路径。与此同时，帖子中的表格显示，在 BF16、FP8 和 NVFP4 版本之间，MMLU-Pro、GPQA 以及 AIME 2025 的得分总体相近。帖子还链接了已发布的 Hugging Face 模型，包括 [`35B-A3B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4)、[`35B-A3B-NVFP4-Fast`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-NVFP4-Fast) 和 [`27B-NVFP4`](https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4)，以及 FP8 KV-cache 校准方案，可将上下文长度延长约 `2×`。**评论者主要认为这是一项 **Blackwell 专属的优势**，并开玩笑说 Pascal 或 RTX 3090 时代的用户可能无法受益，因为这些加速效果依赖更新一代 GPU 的 tensor-core 支持。

    - 评论者询问了 **Qwen3.6 NVFP4 Unsloth 量化版本**与标准非 NVFP4 `4-bit` 量化版本之间的差异，尤其想知道所谓的 `2.5x` 加速是否只在 Blackwell 硬件上成立，还是在常见推理框架中相对于现有 4-bit 格式也同样有效。
    - 关于 **llama.cpp / llama-server 对 NVFP4 的支持**，评论区存在一些技术上的疑问：一位用户指出，llama-server *可以*运行 NVFP4，但此前的性能表现“并不理想”；另一位用户则询问，既然 llama.cpp 现在已经能较好地支持 NVFP4，为什么没有提供 `GGUF` 构建版本。
    - 多条评论暗示，这项优化主要适用于 **NVIDIA Blackwell** GPU；而 **Pascal** 等较老架构，以及 **RTX 3090** 这类消费级显卡，可能无法从 NVFP4 加速中受益。

  - **[如果你花了 4,000–5,000 美元组装一台本地 AI 设备，还会再做一次吗？](https://www.reddit.com/r/LocalLLM/comments/1us6f84/if_you_spent_45k_on_a_local_ai_rig_would_you_do/)**（活跃度：359）：**帖子认为，如果单纯为了运行接近前沿水平的本地 LLM，花费 `$4–5K` 组装本地 AI 设备很难证明其合理性，尤其是在 **DeepSeek V4 Flash** 等 API 的价格约为每百万个未缓存输入 token `$0.14`、每百万个输出 token `$0.28` 的情况下。作者表示，即使使用一台 `128GB` 的 MacBook，运行 `2-bit` 量化的 DeepSeek V4 Flash，与托管模型相比仍然缺乏吸引力；不过，这套设备确实帮助作者了解了量化、KV cache、上下文窗口、内存限制和模型服务等概念。作者认为，出于隐私需求、需要持续运行的工作负载，或者本来就需要这台机器时，昂贵的本地硬件可能是合理的选择；但如果主要目的是替代 Claude/ChatGPT 级别的 API 并节省成本，就不太划算。**没有提供热门评论可供总结。





## 技术性较低的 AI 子版块回顾

 /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. GPT-5.6 编程基准测试

  - **[DeepSWE 刚刚将 gpt-5.6 模型加入了基准测试。希望大家不要习惯于只把 Claude Code 当作唯一的 coding agent。由于图表中的暴力场面过于夸张，因此被标记为 NSFW。](https://www.reddit.com/r/ClaudeAI/comments/1usavpc/deepswe_just_added_the_gpt56_models_to_their/)**（活跃度：1718）：**这张[图片](https://i.redd.it/e5dlfudecbch1.png)是一张 **DeepSWE 基准测试成本/性能图表**，以“DeepSWE 得分”和每项任务的平均成本为两个维度，对比各类 coding-agent 模型。帖子重点介绍了新加入的 **GPT-5.6 系列变体**，认为它们是 **Claude Code/Claude 模型**的强劲低成本竞争者。图中，GPT-5.6/5.5 系列的点大致集中在 `60–70%` 的 DeepSWE 得分区间，同时每项任务的成本相对较低；Claude 模型依然具有竞争力——例如 Claude-fable-5 的得分接近顶部，约为 `70%`——但成本通常更高。**评论基本没有认真讨论基准测试本身，而是几乎全都在批评图表的可视化质量，称其为“psychopath”式制图，并指向 r/dataisugly。帖子中“暴力场面过于夸张”的说法带有夸张和玩梗性质，指的是图表所暗示的 GPT 对 Claude 的冲击，而不是图中真的包含暴力内容。




  - **[GPT 5.6 在 DeepSWE 上以更低价格领先 Fable 5 约 3%。](https://www.reddit.com/r/OpenAI/comments/1us7nml/gpt_56_beats_fable_5_by_3_more_on_deepswe_at_a/)**（热度：1310）：**这张[图片](https://i.redd.it/505rvco3nach1.jpeg)展示了一个 **DeepSWE 排行榜**，其中 **gpt-5.6-sol** 的得分为 `73% ±3%`，平均成本为 `$8.39`，超过了得分为 `70% ±4%` 的 **claude-fable-5**，而成本却远低于 Fable 的 `$21.63`。图片还显示，**gpt-5.6-terra** 以大约低 `4.4` 倍的成本取得了与 Fable 相同的 `70%` 分数。这说明帖子真正强调的是**经过成本调整的 coding-agent 性能**，而不只是原始基准分数。**评论者关注的重点并不是领先的 3 个百分点，而是价格效率，认为 `$8.39` 对比 `$21.63` 才是这条消息的核心。他们还注意到，相比 GPT 5.4，性能似乎有了明显提升；此外，Terra 只需约四分之一的成本就达到了 Fable 的水平。

    - 主要的技术结论是 DeepSWE 的成本归一化性能：评论者强调 **GPT 5.6 达到 `73%`**，并将其与 Fable 5 的表现概括为 **`$8.39` 对比 `$21.63`**——也就是准确率仅有小幅领先，但价格优势要大得多。另一位评论者指出，**Terra 只用大约 `1/4` 的成本就追平了 Fable**，这或许意味着，相比高价的前沿模型，该基准测试可能更偏向成本更低的 planner/executor 配置。
    - 一位用户分享了不同模型系列在以 MCP 为主的实际工作负载中的成本：据称，**Opus 4.8** 每次运行的成本为 **`$1–$2`**，而完成类似任务时，**GPT 5.5** 的成本约为 **`$0.20–$0.50`**，这意味着 GPT 模型的 token 消耗或定价要低得多。他还补充说，**Opus 的输出质量仍然“完全是另一个级别”**，因此这种取舍并不只是基准分数或直接成本的比较。
    - 一位评论者提出，如果 DeepSWE 的数据可靠，那么原本使用 **Opus 4.8 high + Sonnet 5 medium** 的工作流，或许可以改用 **Sol high + Terra high** 作为 planner/executor，在降低成本的同时获得更好的综合结果。这反映出人们正在关注多模型路由：让成本更低但具备高推理能力的层级负责拆解和执行，而不是始终依赖单个高价模型。

  - **[超越人类水平的 competitive programming AI 已经出现](https://www.reddit.com/r/singularity/comments/1urlaam/superhuman_competitive_programming_ai_is_here/)**（热度：1068）：**这张[图片](https://i.redd.it/32ovkav5b6ch1.jpeg)展示了 AtCoder World Tour Finals 表演赛的排行榜：**OpenAI** 以 `8300` 分排名第 `1`，几乎是下一名选手 `tour1st`（`4300` 分）的两倍，支持了帖子所称的“超越人类水平”的 competitive-programming 表现。在相关的 Algorithm 竞赛中，发帖者声称 **OpenAI 解出了全部 `5/5` 道题**，而人类选手最多只解出 `3` 道；帖子还附上了 AtCoder 的[启发式赛排名](https://atcoder.jp/contests/awtf2026heuristic/standings/exhibition)、[启发式赛题目](https://atcoder.jp/contests/awtf2026heuristic/tasks)、[算法赛排名](https://atcoder.jp/contests/awtf2026algo/standings/exhibition)和[算法赛题目](https://atcoder.jp/contests/awtf2026algo/tasks)链接。**评论者强调了差距之大——“看看这个差距”——而一位技术人士指出，这更准确地说是**算法设计 / 竞赛解题**能力，而不是广义上的 **software engineering** 能力。另一个现实限制是，据称 AtCoder 排行榜需要登录后才能查看。

    - 一位评论者区分了 **competitive programming** 与更广义的 software engineering：该系统似乎在*算法编写*方面达到了超越人类的水平——这是一类受约束的编程任务，重点是在竞赛条件下解决形式化问题；但这并不意味着它在端到端的生产软件开发方面也同样超越人类。
    - 多位评论者指出，相关排行榜链接**需要登录**，因此如果没有经过身份验证的基准结果访问权限，就很难独立核实帖子所称的差距和性能。



### 2. Claude Code 大规模构建

  - **[Bun 的创作者 Jarred 使用 Claude Fable 5，在 11 天内将其从 Zig 重写为 Rust；按 API 价格计算，消耗了 16.5 万美元的 Fable 用量。他表示，如果手工完成，这项工作需要 3 名完全了解代码库的工程师，用大约一年的时间，而且期间无法承担其他工作](https://www.reddit.com/r/ClaudeCode/comments/1uru4zg/jarred_creator_of_bun_rewrote_it_from_zig_to_rust/)**（热度：1159）：**根据 [Bun 重写说明](https://bun.com/blog/bun-in-rust)，**Jarred Sumner** 通过 **Claude Code 动态工作流**使用了预发布版 **Claude Fable 5**，在 `11` 天内将 **Bun 的 `535,496` 行 Zig 代码移植为 Rust**，运行了约 `50` 个工作流，最多同时使用 `64` 个 Claude 实例；按 API 价格估算，相当于约 `$165k` 的用量，而手工重写则预计需要 `3` 名工程师工作一年。整个过程先编写了 `PORTING.md`，持续接受人工监控，并通过让独立的 Claude 上下文充当审查者，进行“对抗式审查”。据报告，在 Claude Code `v2.1.181+` 中，Bun `v1.4.0` 相比 `v1.3.14` 修复了 `128` 个 bug，消除了可检测的内存泄漏，Linux/Windows 二进制文件体积缩小约 `20%`，Linux 启动速度提升约 `10%`。**热门评论者大多对这是否说明该做法具有广泛可及性持怀疑态度：他们认为，关键因素并不只是 `$165k` 的模型用量，还包括 **Jarred 对代码库的深入理解和出色的工程能力**。有人将其概括为：“一个价值百万美元的 Thiel Fellow 工程师，使用了 16.5 万美元的 Claude Credits。”还有人认为，以 API 价格来计算是在刻意放大成本和规模的观感。

    - 评论者反对将这次重写主要归功于模型投入：他们认为，真正起决定作用的很可能是 **Jarred Sumner 深厚的 Bun/Zig/运行时专业知识，以及对整个代码库的完整理解**，而 LLM 更像是加速器，并非自主替代者。一位评论者将其描述为：“Bun 是由一个价值百万美元的 Thiel Fellow 工程师重写的，他使用了 `$165K` 的 Claude Credits。”这意味着，对于经验较少的工程师来说，复制这一成果的成本可能高得多。
    - 多条评论质疑这种成本计算方式，指出按照 **API 定价**报价，可能会让人高估实际支出，因为内部使用、合同价格或折扣价格可能更低；同时，原始 token 预算也不等同于工程能力。技术层面的质疑在于，这一成果未必具备普适性：大规模语言或运行时重写需要架构判断、验证能力以及针对特定代码库的知识，而“典型的 vibe coding”工作流无法提供这些条件。

  - **[我用 Claude Code 完全制作的水豚游戏赚了 2.5 万美元](https://www.reddit.com/r/ClaudeAI/comments/1urzr1q/i_just_made_25k_usd_with_my_capybara_game_built/)**（热度：1463）：**一名 iOS 工程师在 `15` 天内为 **[VibeJam 2026](https://vibej.am/2026/#games)** 制作了 **[A Game About Capybaras Delivering Food](https://capybara-vibejam26.leocoout.dev/)**，并赢得 `$25,000` 的一等奖；该项目使用了 **Claude Code Opus 4.7**、**Three.js**，用 **GPT Images-2/Grok** 生成纹理，用 **Tripo3d** 制作模型，并使用 **Suno/ElevenLabs** 生成音频。项目方声称，`188` 次提交、约 `27k` 行代码全部由 AI 编写。工作流主要依靠并行运行的 Claude Code 会话、`/plan`，以及 AI 生成的工具链，包括游戏内地图/地形/道路编辑器、过场动画编辑器、仿 iOS 手机界面、PS1 风格纹理处理流程、任务循环、堆叠物品的伪物理系统、车辆漂移/碰撞、国际化功能，以及一个基于 Cloudflare WebSocket 的多人游戏大厅。该大厅以约 `10 Hz` 的频率中继玩家状态，并采用 `O(n²)` 的扇出扩展方式。**热门评论大多与技术无关：有人开玩笑说 Claude 经常建议把“水豚”作为吉祥物；还有人质疑标题的表述，指出这笔钱来自比赛奖金，而不是游戏收入。






### 3. Frontier Model 使用限制

  - **[GPT-5.6 Sol Ultra 很惊艳——前提是你是 Plus 订阅用户，而且只能用 12 分钟](https://www.reddit.com/r/ChatGPT/comments/1uscohi/gpt56_sol_ultra_is_impressive_for_the_12_minutes/)**（热度：914）：**一位 **ChatGPT Plus** 用户表示，他用 **GPT-5.6 Sol Ultra** 完成两个大型批处理/Agent 任务后，即使额度刚重置，也耗尽了 Plus 的使用额度：一个是合并并分析约 `10` 份 PDF，生成约 `700` 页的文档；另一个是整理 Obsidian vault 中约 `700` 个 Markdown 文件。主要的技术性反驳认为，这些任务很可能处理了**数百万个 token**：仅那份 700 页文档就可能产生约 `280k–560k` 个输出 token；而单次处理 700 个 Markdown 文件，还可能增加 `210k–1.05M` 个 token，尚未算上规划、重复读取、重写、重试或多 Agent 的额外开销。多数评论者反对用提示词数量衡量成本，认为“两个任务”也可能意味着巨大的计算量和 token 消耗；大家最明确的共同批评是，OpenAI 的**额度计量方式过于模糊**，即使对于 `$20/月` 的套餐来说，限流本身在经济上是可以预期的。

    - 多位评论者认为，这次额度快速消耗更可能是由** token/计算量，而不是提示词数量**造成的：一份 `700` 页的生成报告可能包含约 `280k–560k` 个输出 token；而处理 `700` 个 Markdown 文件，按每个文件 `300–1,500` 个 token 计算，每轮还会增加 `210k–1.05M` 个输入 token。在 **Sol Ultra** 中，再加上规划、重复读取、重写、重试以及多 Agent 之间的交接，评论者估计整个工作流达到**数百万个处理 token**是完全有可能的。
    - 一项技术层面的批评是，**Plus 的额度界面不透明**：用户看到的只是模糊的使用量提示，而不是基于计算量或 token 的计费/统计模型。评论者认为，这种抱怨有一定道理，因为 OpenAI 对外呈现的是“消息数”或时间窗口，而在高成本的多 Agent 模式下，高上下文的批处理任务可能会以不成比例的速度消耗额度。
    - 一项实用建议是，除非目的是进行基准测试，否则不要用 **Ultra** 处理大型、高上下文的批处理工作流；评论者指出，涉及数百份文档和长篇综合生成的任务，即使表面上只发送了很少的提示词，在有额度上限的消费者订阅中也很可能效率不高。

  - **[5 小时和每周限额已重置。感谢 Anthropic！](https://www.reddit.com/r/ClaudeAI/comments/1urzmj0/5_hour_and_weekly_limits_have_been_reset_thanks/)**（热度：2865）：**这张图片是 **ClaudeDevs** 发布的一张深色模式 X/Twitter 截图，内容是：*“我们已为所有用户重置 5 小时和每周速率限制”*（[图片](https://i.redd.it/djfpk4js49ch1.jpeg)）。从技术上看，这意味着 **Claude/Anthropic 用户的短时窗口和每周额度计数器已被清零**，用户可以立即恢复使用；该帖子还询问，这次重置是出于善意、竞争时机，还是与可能推出的 **5.6** 更新有关。评论大多是猜测：有人开玩笑说，这个时间点可能意味着 **OpenAI** 带来了压力；也有人后悔自己没在重置前把额度用完，但同时也对这次免费的额度刷新表示欢迎。