---
companies:
- anthropic
- langchain
- google
- meta-ai-fair
- nvidia
- cohere
- weaviate
date: '2026-07-07T05:44:39.731046Z'
description: 'Anthropic 通过面向移动端和网页的 **Claude Cowork** 扩展了“后台代理”的用户体验，重点突出能够在后台运行任务的协作伙伴。他们还向付费用户开放了
  **Claude Fable 5** 的使用权限。


  在代理设计中，“harness（代理运行框架）”这一概念逐渐受到关注，Lilian Weng 对此进行了重点介绍，**LangChain** 也通过推出新的 **Deep
  Agents** 课程和开源项目呼应了这一趋势。**Google** 的 **Gemini API Managed Agents** 引入了后台执行、自定义函数调用等功能。


  面向操作员的代理基础设施也迎来了更新，包括 **Codex Mobile iOS**、集成 **1Password** 的 **Hermes Agent**，以及支持运行时门控写入权限的
  **Weaviate 1.38**。此外，人们还在探索通过手机或短信实现“人在回路”控制的方式。


  在模型发布方面，**Meta AI** 推出了 **Muse Image**，并预览了 **Muse Video**。这两者都采用了代理式生成循环，包含规划、网页搜索和自我优化，并在
  Image Arena 和 Video Arena 中取得了最高排名。**NVIDIA** 发布了 **Audex**，这是一款拥有 300 亿参数、采用混合专家（MoE）架构、支持
  100 万上下文窗口的模型，可统一处理文本和音频任务。

  '
id: MjAyNS0x
models:
- claude-fable-5
- muse-image
- muse-video
- audex
people:
- mikeyk
- kimmonismus
- lilian_weng
- sakana
- _philschmid
- officiallogank
- dimillian
- reach_vb
- teknuim
- victorialslocum
- omarsar0
- alexandr_wang
- _tim_brooks
title: '今天没发生什么特别的事。

  '
topics:
- agent-design
- background-execution
- task-management
- human-in-the-loop
- agentic-generation
- reinforcement-learning
- model-scaling
- moe
- context-windows
- audio-processing
- video-generation
- image-generation
- open-source
- model-release
---

**平静的一天。**

> 2026 年 7 月 6 日至 7 月 7 日的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/)支持搜索所有过往期刊。提醒一下，[AINews 现在已经成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以选择[订阅或取消订阅](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同的邮件发送频率！




---

# AI Twitter 速览


**Agent 产品、Harness 与长时间运行的工作流**

- **Anthropic 在 Claude 的基础上扩展“后台 Agent”体验**：按用户互动量来看，最大的产品发布是 [Claude Cowork 登陆移动端和 Web](https://x.com/claudeai/status/2074525815820169320)，它将 Claude 定位成一个可以在后台执行任务的协作伙伴，而不只是前台聊天界面。相关帖子还显示，产品正在围绕统一的首页标签，以及更紧密的 Chat/Cowork 集成逐步融合，详见 [@mikeyk](https://x.com/mikeyk/status/2074531605537046953)。此外，Anthropic 还将付费套餐访问 **Claude Fable 5** 的期限延长至 7 月 12 日；[@claudeai](https://x.com/claudeai/status/2074548242386178258) 发布的公告获得了大量互动。不过，许多用户在 [@kimmonismus](https://x.com/kimmonismus/status/2074606005963391225) 等人的评论中指出，这一安排与每周使用限额的时间点搭配得不太妥当。

- **Harness 工程正日益成为 Agent 设计的核心**：Lilian Weng 的新文章被广泛引用，它将递归式自我改进重新定义为围绕 **Harness** 展开，而不是直接修改模型权重；Sakana 在[这条帖子](https://x.com/SakanaAILabs/status/2074489949529776308)中总结了这一方向与 **The AI Scientist**、**ShinkaEvolve** 和 **Darwin Gödel Machine** 的联系。LangChain 也通过 [@LangChain](https://x.com/LangChain/status/2074539083204820997) 和 [@hwchase17](https://x.com/hwchase17/status/2074547871194698207) 发布的新 **Deep Agents** 课程及开源 Harness 项目，呼应了这一转变。Google 也在将这一方向产品化：Gemini API 的 **Managed Agents** 新增了**后台执行**、**远程 MCP 服务器**、**自定义函数调用**和**凭据刷新**功能，详见 [@_philschmid](https://x.com/_philschmid/status/2074533915038027972) 和 [@OfficialLoganK](https://x.com/OfficialLoganK/status/2074552932318765376) 的帖子。

- **实用型 Agent 基础设施正变得更加固化和明确**：近期有几项值得注意的面向操作者的更新：**Codex Mobile iOS** 新增任务管理、筛选差异、SSH 密钥登录、分支比较和附件流程，详见 [@Dimillian](https://x.com/Dimillian/status/2074396968223211819) 和 [@reach_vb](https://x.com/reach_vb/status/2074400018769793176) 的帖子；**Hermes Agent** 增加了可插拔的密钥管理器和原生 **1Password** 集成，并支持将会话/数据集导出为多种格式，包括私有 Hugging Face 仓库，详见 [@Teknium](https://x.com/Teknium/status/2074564207555772912) 发布的[相关帖子](https://x.com/Teknium/status/2074639961727655959)；**Weaviate 1.38** 让其 MCP 服务器正式 GA，并通过运行时控制写入权限，尤其值得注意的是，现在无需重启，就可以实时切换 **MCP_SERVER_WRITE_ACCESS_ENABLED**。详见 [@victorialslocum 的帖子](https://x.com/victorialslocum/status/2074493681403339104)。[@omarsar0](https://x.com/omarsar0) 则展示了一种更具实验性的模式：使用 Dial MCP 服务器，让 Agent 可以通过电话、SMS 或 iMessage 升级决策，从而实现人工介入控制。

**模型与模态发布：音频、语音、机器人和媒体生成**

- **Meta 的 Muse Image/Muse Video 将 Agent 式生成带入媒体领域**：Meta Superintelligence Labs 在 [@AIatMeta](https://x.com/AIatMeta/status/2074577662840832382)、[@alexandr_wang](https://x.com/alexandr_wang/status/2074555909347369105) 和 [@_tim_brooks](https://x.com/_tim_brooks/status/2074578008296628698) 的公告中发布了 **Muse Image**，并预览了 **Muse Video**。其值得关注的技术点不只是图像质量，还在于明确采用了 **Agent 式生成循环**：在渲染之前，系统会进行规划、Web 搜索、工具调用、代码执行和自我改进。Meta 还表示，随着**测试时计算量扩展**，性能会进一步提升；在[这条后续帖子](https://x.com/AIatMeta/status/2074587864923250873)中，Meta 说自我改进行为是在 RL 过程中自然出现的，而不是通过人工编写脚本实现的。在公开评测中，Muse Image 很快登上了 **Image Arena 第二名**，仅次于 GPT Image 2，详见 [Arena 的排名](https://x.com/arena/status/2074581979765539153)；Muse Video 则在[另一条 Arena 帖子](https://x.com/arena/status/2074591193783320851)中首次亮相便位列 **Video Arena 第三名**。



- **NVIDIA 和 Cohere 都推出了强劲的音频产品**：NVIDIA 发布了 **Audex**，这是一个拥有 **300 亿参数 / 30 亿激活参数的 MoE**，支持 **100 万上下文**，用于统一处理文本和音频任务。相关内容由 [@HuggingPapers](https://x.com/HuggingPapers/status/2074384562952749254) 做了总结，[@_weiping](https://x.com/_weiping/status/2074537900172050704) 则进行了更详细的介绍。该模型的核心主张是：在保留文本智能的同时，通过单一的 MoE 主干网络，提供广泛的音频生成与理解能力。Cohere 发布了 **Cohere Transcribe Arabic**，并称其为最准确的开源阿拉伯语 ASR 模型，采用 **Apache 2.0** 许可证。[@cohere](https://x.com/cohere/status/2074499759616729149) 和 [@JayAlammar](https://x.com/JayAlammar/status/2074511963934118282) 的帖子重点提到了它对**方言**、**语码转换**以及**带阿拉伯口音的英语**的支持。

- **开放式机器人领域正不断向 Hugging Face + NVIDIA 生态整合**：NVIDIA 将其机器人技术栈进一步扩展到 HF 生态，把 **GR00T 1.7** 和 **Isaac Teleop** 引入 **LeRobot**，旨在支持开放式人形机器人工作流。相关信息见 [@NVIDIARobotics 的公告](https://x.com/NVIDIARobotics/status/2074380795855147072) 和[集成指南](https://x.com/NVIDIARobotics/status/2074390485251113317)。在具身智能方面，UMA 展示了一个完整的机器人技术栈案例：[‍@RemiCadene](https://x.com/RemiCadene/status/2074442725814878510) 介绍了一个由小团队在 9 个月内打造的原型；而 [Northstar 项目公布](https://x.com/RemiCadene/status/2074442439142609237) 以及 [@psermanet 的安全说明](https://x.com/psermanet/status/2074512829617491996) 则强调了通过软硬件垂直整合，打造值得信赖的机器人。

**训练、推理与后训练技术**

- **Liquid AI 的 “Antidoom” 直接针对推理循环失败问题**：当天最清晰的技术发布之一是 [Liquid AI 的 Antidoom](https://x.com/liquidai/status/2074494130126811473)。这是一种开源训练方法，用于减少小型推理模型陷入 **doom loop** 的情况，即不断重复生成 token，直到上下文耗尽。报告显示，相关问题的发生率大幅下降：在贪心采样下，**LFM2.5-2.6B 从 10.2% 降至 1.4%**，**Qwen3.5-4B 从 22.9% 降至 1%**，下游评测结果也有所提升。该方法名为 **FTPO（Final Token Preference Optimization，最终 Token 偏好优化）**，它会重新标注触发循环的 token，并将概率重新分配给其他候选项。[‍@helloiamleonie](https://x.com/helloiamleonie/status/2074498103982408044) 和 [@LiorOnAI](https://x.com/LiorOnAI/status/2074547819114086561) 对此进行了很好的总结。这很好地体现了近期领域内的一种趋势：不只是单纯扩大参数规模，而是针对具体的失败模式进行修复。

- **推理效率与压缩仍是重要前沿方向**：NVIDIA 的 **Puzzle-75B-A9B** 压缩工作通过 [@omarsar0](https://x.com/omarsar0/status/2074543978129793462) 的介绍获得了广泛关注：该工作在保留推理、编程、长上下文和 Agent 能力的同时，对一个混合 MoE 母模型进行压缩，使服务器吞吐量约提升 **2 倍**；在 H100 上，支持 100 万上下文时的并发请求数也从 **1 个增加到 8 个**。在工具方面，**Nsight Python 1.0** 已在 [@HagedornBastian 的帖子](https://x.com/HagedornBastian/status/2074509770342445375) 中宣布发布，使 GPU 性能分析可以通过 Python 编写脚本实现。Unsloth 也发布了 **DeepSeek-V4-Flash 的 GGUF 文件**，并支持导出为 **NVFP4/FP8**，同时提升了 **GRPO** 和 MoE 的运行速度，详情见 [@danielhanchen 的更新](https://x.com/danielhanchen/status/2074510444778463331)。

- **Agent RL 与验证技术正变得更加专门化**：[‍@cwolferesearch](https://x.com/cwolferesearch/status/2074558199819067606) 提到，研究人员正将 **GRPO 风格的归一化**调整到 Agent RL 场景中，并在**任务级别**或**环境级别**进行归一化，以应对多轮环境中更高的奖励方差。另一方面，[@omarsar0](https://x.com/omarsar0/status/2074556579580711050) 介绍了一篇来自 Stanford/NVIDIA/Berkeley 的、**无需训练的验证器**论文。该方法从评分 token 的 logits 中读取经过校准的连续分数，并在 **Terminal-Bench V2、SWE-Bench Verified、RoboRewardBench 和 MedAgentBench** 上取得了亮眼成绩。这表明，验证能力正逐渐成为一个独立的 scaling 维度。

**可解释性、模型内部机制与 “J-Space” 争论**

- **Anthropic 的 J-space 工作主导了可解释性讨论，但也招致了尖锐批评**：社区对此看法分化明显，一部分人认为这项工作对机制分析很有价值，另一部分人则反对其关于意识的表述。[@danburonline](https://x.com/danburonline/status/2074429991576650014)、[@paul_cal](https://x.com/paul_cal/status/2074388528243310976) 和 [@scaling01](https://x.com/scaling01/status/2074432865794679235) 提出了有力批评，认为根据 Jacobian-lens 的定义，这些向量之所以具有因果性，很大程度上是由构造方式决定的。[@jacobandreas](https://x.com/jacobandreas/status/2074487546692735002) 则提供了一个有用的历史参考，提醒读者回顾最初的 **Jacobian lenses** 论文。



- **更值得关注的技术结论是跨模型结构，而不是关于意识的修辞**：[[@eliebakouch](https://x.com/eliebakouch/status/2074532904009421260)] 在 **38 个开源模型**上计算了 J-lens 几何结构的 **CKA 相似度**，发现不同模型在层级与深度组织方式上竟然具有出人意料的普遍一致性，即使是 **Llama** 和 **OLMo** 这样彼此无关的模型家族也不例外。Anthropic 和 Neuronpedia 还发布了**开源模型的 J-lens 权重**，详见[这篇后续内容](https://x.com/eliebakouch/status/2074537985102565795)。与此同时，Goodfire 推出了用于表示激活中多维概念的 **Block-Sparse Featurizers**，并在[他们的讨论帖](https://x.com/GoodfireAI/status/2074634702737281303)中指出，许多视觉概念本质上是 **2–4 维的区块**，而不是单一方向。

**基准测试、评估与领域专用系统**

- **Agent 和法律基准测试仍在不断暴露这样一个差距： “通过了许多标准”并不等于“真正完成了实际工作”**：[Agent Arena](https://x.com/arena/status/2074484787663052849) 将 **Claude Sonnet 5 (Thinking)** 排在第 **6** 位；它在确认任务成功率和 bash 使用方面表现最突出，但可控性仍存在不确定性。Artificial Analysis 推出了 **Harvey LAB-AA**，这是一个法律 Agent 基准测试，涵盖 **24 个执业领域的 120 项私有法律任务**。在该测试中，**Claude Fable 5** 以 **14.2% 的全项通过率**领先；**Claude Opus 4.8** 和 **GLM-5.2** 均为 **7.5%**，而 GLM 达到这一成绩时，每项任务的成本大约只有 Fable 的 **~6%**，详见[他们的发布公告](https://x.com/ArtificialAnlys/status/2074541975186165887)。这里传达出的关键信息是：模型可能满足许多单独的评分标准，但仍然无法产出合格的端到端交付成果。

- **研究自动化和专业领域系统正在不断拓展**：Google 在[这篇 ICML 帖子](https://x.com/GoogleResearch/status/2074384746076135575)中介绍了 **Experience AI Scientist**，这是一个用于端到端科学工作流的多 Agent 系统。DeepMind 也推出了 **Predicting the Past**，通过普通英语交互，将 Gemini 与 **Aeneas** 和 **Ithaca** 结合，用于希腊语/拉丁语历史分析，详见[他们的讨论帖](https://x.com/GoogleDeepMind/status/2074513661750546762)。在法律 AI 商业化方面，**Norm Ai** 宣布完成 **1.2 亿美元 C 轮融资，估值达到 12 亿美元**，并在[@johnjnay 的帖子](https://x.com/johnjnay/status/2074485345593245833)中介绍了一套完整的“Agentic Law”架构，涵盖软件以及 AI 原生律所。

**热门推文（按互动量排序）**

- **Claude 的访问权限 / 产品推广**：[移动端和网页端的 Claude Cowork](https://x.com/claudeai/status/2074525815820169320)以及[开放至 7 月 12 日的 Fable 5 访问权限](https://x.com/claudeai/status/2074548242386178258)，是互动量最高、且与技术最相关的产品公告。
- **开源开发者计划**：[@ClaudeDevs 为开源项目维护者提供 6 个月的 Claude Max 20x](https://x.com/ClaudeDevs/status/2074570404035993780)，引发了巨大互动，预计将对 OSS 生态中的工具采用产生重要影响。
- **Meta 的媒体生成**：[Muse Image 发布](https://x.com/AIatMeta/status/2074577662840832382)以及 [Arena 将 Muse Image 排名第 2](https://x.com/arena/status/2074581979765539153)，是当天最大的多模态产品新闻。
- **推理可靠性**：[Liquid AI 发布 Antidoom](https://x.com/liquidai/status/2074494130126811473)，是当天技术含金量最高的训练技术帖子。
- **可解释性**：[38 个开源模型中跨模型的 J-lens 普遍性](https://x.com/eliebakouch/status/2074532904009421260)，是 J-space 讨论中最有力的技术延伸。



---

# AI Reddit 速览

## /r/LocalLlama + /r/localLLM 速览

### 1. 开源模型发布与推理效率

  - **[Tencent Hy 发布新的开源模型：Hy3（总参数 295B，激活参数 21B，Apache 2.0）](https://www.reddit.com/r/LocalLLaMA/comments/1uoozt4/new_open_model_from_tencent_hy_hy3_295b_total_21b/)**（热度：653）：****Tencent** 在 [Hugging Face](https://huggingface.co/collections/tencent/hy3) 上发布了非预览版的 **Hy3** 开源模型系列。该系列被描述为一个总参数量为 `295B`、激活参数量为 `21B` 的 MoE 模型，现已采用 **Apache 2.0** 许可，不再使用之前限制较多的社区许可证。帖子指出，早期许可证据称禁止在包括**韩国、英国和欧盟**在内的一些地区使用；热门评论则提到，相比 **HY3-Preview**，新版本据称在基准测试上有所提升，并认为这可能会对高端本地/家用推理部署具有意义。评论者普遍认为，改用 Apache 2.0 许可证是最重要的变化，尤其考虑到 Tencent 近期发布的翻译模型也采用了 Apache 许可证。大家对报告中的基准提升能否转化为实际用途持谨慎乐观态度，但在脱离厂商测试图表、经过实际测试之前，仍然保持一定怀疑。



  - 评论者指出，**Hunyuan/HY3** 目前已列为 **Apache 2.0** 许可证，这与此前的“社区”许可证形成对比；据报道，后者曾限制在 **South Korea、UK 和 EU** 等地区的使用。大家认为，这对部署来说具有重要的技术意义，因为 Apache 2.0 消除了许多商业和地域方面的使用障碍。
    - 一些用户关注 Tencent 宣称的、相较于 **HY3-Preview** 的基准性能提升，是否能转化为真实工作负载中的实际表现。鉴于其据报道采用 `295B` 总参数 / `21B` 激活参数的 MoE 架构，评论者认为，如果 **GGUF** 等推理格式能够推出，它可能会成为“高端家用设备”的一个选择。
    - 早期有观点猜测，HY3 可能会在本地和开放权重工作流中，成为 **Qwen** 与 **MiniMax** 模型的替代方案。不过，在得出结论之前，评论者仍在等待量化版本发布以及独立测试结果。

  - **[新模型：GigaChat3.5-432B-A28B（发布当天即支持 GGUF！）](https://www.reddit.com/r/LocalLLaMA/comments/1uotkm7/new_model_gigachat35432ba28b_with_day0_gguf/)**（热度：510）：****Sberbank/ai-sage** 发布了 **GigaChat3.5-432B-A28B**，这是一款总参数量为 `432B`、激活参数量为 `28B` 的大型 MoE 对话模型，同时还提供了 [基础模型权重](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-base) 和发布当天即可使用的 [GGUF 权重](https://huggingface.co/ai-sage/GigaChat3.5-432B-A28B-GGUF)；目前，`llama.cpp` 通过 [PR #25342](https://github.com/ggml-org/llama.cpp/pull/25342) 提供支持。模型卡片中的摘录声称，与 `700B` 的 **GigaChat 3.1 Ultra** 相比，它的体积缩小了约 `40%`，同时在代码、数学和 Agent 基准测试中表现更好；每个 token 使用的 KV cache 减少约 `4×`，在相同内存下可容纳超过 `2×` 的上下文，吞吐量提升约 `20%`。在架构方面，评论者特别提到，它采用了定制的混合 MoE 结构，将 **MLA** 层与 **GatedDeltaNet** 线性注意力层结合起来；此外还加入了包含两个 MTP head 的 **Multi-Token Prediction**，据称可将贪心解码速度从使用一个 head 时的约 `1.5×`，提升到使用两个 head 时的最高 `2.2×`。**评论者质疑以 **DeepSeek 3.2** 作为基准参考，认为它大约落后前沿系统一年；同时也指出，GigaChat3.5 是一款*非推理*模型，因此在解读基准对比结果时应考虑这一点。这次发布之所以受到称赞，还因为它在如此大的规模下展现出了不同寻常的开放程度——不仅提供了基础模型和中间检查点，尽管确切的训练数据集仍未公开。

    - 几位评论者指出，不应将 **GigaChat3.5-432B-A28B** 直接与当前的前沿推理模型进行比较：有人质疑以 **DeepSeek 3.2** 作为基准参考，因为在他们看来，它“*大约落后前沿模型一年*”；另一人则强调，GigaChat 3.5 是一款**非推理模型**，这会显著影响其基准分数的解读方式。
    - 一段技术说明强调了它相较于 **GigaChat 3.1 Ultra 700B** 的主要架构变化：据报道，GigaChat 3.5 的体积缩小了 `~40%`，但在代码、数学和 Agent 任务上表现更强；每个 token 使用的 KV cache 减少约 `4×`，在相同内存下可容纳 `2×+` 的上下文，生成吞吐量提升 `~20%`。该模型采用定制的 MoE 混合注意力设计，将 **MLA** 与 **GatedDeltaNet** 线性注意力层结合起来，并加入了包含两个 MTP head 的 **Multi-Token Prediction**，据称使用一个 head 时可将贪心解码速度提升约 `1.5×`，使用两个时最高可提升至 `2.2×`。
    - 一位评论者称赞这次发布不仅开放了最终模型的权重，还开放了**中间检查点和基础模型**，认为这对于如此大规模的模型而言十分罕见；目前主要缺少的材料是确切的训练数据集。另一位评论者认为，该模型最擅长的领域可能是**俄语处理**，而在俄语之外的表现则较为一般，因为目前已经有更强的多语言替代方案。



  - **[nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16 · Hugging Face](https://www.reddit.com/r/LocalLLaMA/comments/1upsdmi/nvidianvidianemotronlabs3puzzle75ba9bbf16_hugging/)**（热度：349）：****NVIDIA** 发布了 [`NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16)。这是一款可商用、针对部署进行优化的混合 MoE LLM，基于 **Nemotron-3-Super-120B-A12B**，采用技术报告中介绍的 **Iterative Puzzle** 后训练压缩方法构建而成。模型总参数量从 `120.7B`、激活参数量从 `12.8B`，分别压缩至 `75.3B` 和 `9.3B`，同时保留了交错排列的 **Mamba + MoE + Attention** 层以及 **Multi-Token Prediction**。据称，在单个 `8×B200` 节点上，服务器吞吐量可提升约 `2×`；在单张 H100 上处理 `1M` token 上下文时，并发请求数可由 `1` 提升至 `8`。该模型面向推理/聊天、代码、多语言应用、RAG/Agent 工作负载，以及英语、法语、德语、意大利语、日语、西班牙语和中文的长上下文推理。**评论者主要关注模型的实际部署特性，尤其是相对较小的 `75B` 总参数量、`9B active` 激活参数量，以及 `1M` 上下文长度。一位用户开玩笑地表示，想在 `64GB DDR4 RAM` 上尝试运行 `Q6`/`Q4` 量化版本，这反映出大家对本地及消费级部署的兴趣，尽管此次 BF16 版本主要面向高端加速器。

    - 该帖子指出，**NVIDIA-Nemotron-Labs-3-Puzzle-75B-A9B-BF16** 被定位为一款通用推理/聊天模型，适用于**英语、代码、多语言场景、Agent 系统、RAG、复杂指令跟随以及长上下文推理**，并且拥有特别大的 **`1M` token 上下文窗口**。
    - 一位评论者对 benchmark 表现提出质疑，称该模型公布的结果**不如 Super-120**；而他们认为 Super-120 本身的表现已经不尽如人意，因此这可能意味着该模型相较于其明显的源模型/基础模型，改进十分有限。
    - 有人关注在本地运行量化版本，具体提到了在 **`64GB DDR4 RAM`** 上运行 **Q6/Q4**，这意味着模型能否真正部署，可能很大程度上取决于量化效果，以及受 CPU 和内存限制的推理性能。

  - **[ThinkingCap-Qwen3.6-27B：准确率与基础版 Qwen3.6 相当，但思考过程减少约 50%](https://www.reddit.com/r/LocalLLaMA/comments/1up3mui/thinkingcapqwen3627b_same_accuracy_as_base_qwen36/)**（热度：334）：****bottlecapai** 发布并评测了 [`ThinkingCap-Qwen3.6-27B`](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B#out-of-domain-token-efficiency)，声称该模型在准确率大致达到基础版 **Qwen3.6-27B** 水平的同时，“思考”/推理 token 减少了约 `50%`。作者按照 Qwen 推荐的 `temperature=1.0`，通过多随机种子进行 benchmark，并在推理、MCQA、聊天、system prompt 遵循、安全、数学、代码和 Agent 任务等方面进行了统计显著性检验，同时涵盖领域内留出数据和领域外评测。**评论者总体持谨慎乐观态度：有人认为 Qwen 3.6 是目前最强的低成本开放权重 20B–40B 选项，也有人指出，用户本来就可以通过设置 `reasoning-budget` 来控制成本。一位评论者观察到，该模型在评测中的表现似乎略逊于基础版，但认可发布者对这种权衡进行了透明说明。

    - 评论者指出，类似的思维链长度缩减，也许可以在推理时通过设置 Qwen 的 `reasoning-budget` 来实现，而不必使用单独调优的 checkpoint。这引出了一个技术问题：ThinkingCap 的收益究竟来自模型行为的改变，还是仅仅来自在推理过程中强制采用更低的 token 预算。
    - 一位评论者注意到，报告中的评测结果相较于基础版 Qwen3.6 似乎**略差**，尽管其声称思考 token 减少了约 `50%`；不过，他们认可发布者透明地展示了这种权衡。实际来看，当延迟/成本比保留每一个 benchmark 得分更重要时，这个模型值得测试。
    - 有人提供了一个用于本地推理的 GGUF 构建版本：[bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF](https://huggingface.co/bottlecapai/ThinkingCap-Qwen3.6-27B-GGUF)。对于希望在兼容 llama.cpp 的运行时上评估 `27B` 模型量化部署的用户来说，这一版本很有参考价值。


### 2. 本地模型的可靠性与可解释性



  - **[我测试了 Anthropic 的新 Jacobian Lens，结果它变成了一个本地模型幻觉路由器](https://www.reddit.com/r/LocalLLaMA/comments/1upy31x/i_tested_anthropics_new_jacobian_lens_on_open/)**（热度：367）：**一位 Reddit 用户将 Anthropic 提出的 **Global Workspace / Jacobian Lens** 思路应用到了开放权重模型上，并在 [`solarkyle/jspace`](https://github.com/solarkyle/jspace)、[演示页面](https://solarkyle.github.io/jspace/demo/) 以及 [HF lenses/traces/routers](https://huggingface.co/solarkyle/jspace-lenses) 发布了代码、演示和相关产物。在 `500` 道 TriviaQA 题目上进行模型测试时，针对 Gemma 各变体，Jacobian Lens 的“workspace trajectory”特征——熵斜率、后段熵、熵标准差、答案排名、层间一致性——在预测错误答案方面优于输出 logprob：E4B 的 AUC 为 `0.773`，而 logprob 为 `0.711`；12B 为 `0.824` 对 `0.736`；12B abliterate 版本为 `0.799` 对 `0.731`；26B MoE 为 `0.749` 对 `0.725`。组合多种信号后，效果提升到 `0.787–0.843`。不过，**Qwen 3.6 27B** 是一个反例：它的 logprob 本身已经很强（`0.856`），而 workspace 不仅表现较差（`0.646`），组合后的结果（`0.838`）也不如单独使用 logprob。作者提出的系统是一个单次推理的本地幻觉/风险路由器：先在本地生成答案，获取一次 workspace 快照，再运行一个很小的 logistic-regression sidecar；如果答案表面上置信度很高，但内部状态“雾化”，就升级到搜索、引用或云端模型。另一个值得注意的结果是，abliteration 大幅增加了 Gemma 12B 编造虚假实体的情况（从 `17/50` 增加到 `49/50`）。**评论区对这一解释展开了争论：有人认为 Qwen 的失误并不意外，因为 Qwen 模型似乎“过度训练/已经 grokked”，而且在对齐任务上非常固守模式，因此在这类任务中的输出置信度校准异常准确。另有人提醒，这项实验可能只说明“不确定性 ↔ 相互竞争的潜在候选”，并不能可靠地推出“候选相互竞争就必然意味着幻觉”，因为这种歧义也可能来自正常的推理过程，而非编造。

    - 有几位评论者质疑 Jacobian Lens 信号背后的核心因果解释：这项实验检测到的可能是**多个相互竞争的潜在延续**，而不是直接检测幻觉。一位评论者认为，不确定性自然会增加活跃候选想法的数量，但“想法相互竞争 → 产生幻觉”这一推论并不一定成立；在模型信息不完整、却仍然做出校准良好的猜测时，这一区分尤其重要。
    - 一份详细的仓库级审查认为，幻觉评估受到**事实标签错误**的影响。评论者举例称，Ross Bagdasarian（《The Chipmunks》的创作者）以及 Balfour 之后的 H. H. Asquith，明明可能是正确答案，却被标记为错误。同一位评论者指出，如果标签存在噪声，那么报告中的 AUC 和路由器结果就不可靠；此外，将该方法称为 *label-free* 也有问题，因为路由器是通过**正确/错误答案上的 logistic regression**进行训练的，所以即使运行时使用的是无监督特征，训练过程仍然属于监督学习。
    - 有人批评评估方法可能存在**数据泄漏**：据称，归一化是在交叉验证之前对完整数据集执行的，因此测试折的信息可能进入了训练阶段的预处理。评论者还认为基线过于单薄，主要只有少量 logprob/输出置信度特征，因此“路由器普遍优于置信度校准”的说法有所夸大；尤其是考虑到一些观点认为 **Qwen** 模型在熟悉的任务模式上本来就有异常良好的校准能力，而且表现得非常“固执”或训练过度。

  - **[Qwen 3.6 27B 在 Agent 工作上彻底失败](https://www.reddit.com/r/LocalLLaMA/comments/1uphzhj/qwen_36_27b_absolutely_fails_at_agentic_work/)**（热度：740）：**楼主报告称，在 **llama.cpp nightly** 下使用 **RTX 6000**，让 **Qwen 3.6 27B** 以 `8-bit`/`16-bit` 精度运行时，它在单轮提示以及长篇/演示 HTML 生成方面表现良好，但在多轮 *agentic* 工作流中却反复失败——*“大约每 4 轮就会做出一些完全没脑子的事情”*——因此楼主改回使用 `4-bit`/`5-bit` 的 **Qwen 3.5 122B**。技术回复建议检查 chat-template 和推理配置，尤其可以尝试 [froggeric/Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates) 来排查 Agent 流程中的 bug，并确认 `preserve_thinking` 等参数是否设置正确。**评论者对这一笼统结论持怀疑态度，认为如果没有提供确切的推理参数、模板和复现细节，就很难诊断问题；而且“多数人并没有遇到你的这种情况”。**



- 几位评论者认为，**chat template 问题**可能是导致 Qwen 3.6 27B Agent 能力表现不佳的主要原因，并推荐使用 froggeric 修复过的模板：[Qwen-Fixed-Chat-Templates](https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates)。据称，这些模板“修复了 Agent 工作流中的一些问题”，这意味着相关故障可能源于提示词格式或工具调用序列化方式，而不是基础模型本身。
- 一个技术排查帖询问用户是否使用了正确的推理参数，特别提到了 `preserve_thinking`。评论者希望用户提供完整的参数配置，这表明 Qwen 3.6 27B 在 Agent 工作流中的稳定性可能高度依赖解码配置，以及是否在多轮对话中保留推理轨迹。



### 3. 中国 AI 模型访问权限政策之争

  - **[北京正考虑限制海外访问中国顶尖 AI 模型（Reuters）](https://www.reddit.com/r/LocalLLaMA/comments/1uprmso/beijing_is_looking_at_curbing_overseas_access_to/)**（热度：1011）：**这张图片是 **Reuters 文章的截图**，不是梗图。文章报道称，**北京正考虑限制海外访问中国领先 AI 模型**，涉及 **Alibaba、ByteDance 和 Z.ai** 等公司，理由包括国家安全担忧，以及先进模型泄露的风险。从技术层面看，这可能会影响中国具有竞争力的前沿模型、开放权重模型，或可通过 API 访问的模型在中国境外的可用性，从而减少全球用户获取美国实验室之外替代方案的机会；图片：[i.redd.it/9s1018gggsbh1.jpeg](https://i.redd.it/9s1018gggsbh1.jpeg)。**评论者将此事视为又一项 AI 访问限制，并担心具有竞争力的本土模型和开放模型会变得更难获取。一位评论者认为，**Mistral** 可能会成为更重要的非美国、非中国替代方案，尤其是其巴黎附近的数据中心投入使用后，或许能够训练参数规模约为 `10T` 的模型。

    - 一位评论者将 **Mistral** 视为潜在的非中国开放权重替代方案，并称其巴黎附近的新数据中心预计很快上线，届时可能支持训练参数规模约为 `10T` 的模型。这意味着，如果海外访问中国前沿模型或开放模型受到限制，欧洲的算力能力可能会变得具有战略重要性。
    - 几位评论者讨论了提前归档自己偏好的 **开放权重模型**，其中包括一些目前还无法在本地运行的模型，因为访问限制日后可能会让下载或再分发变得更加困难。这反映出人们对模型可用性、可复现性，以及在地缘政治管控收紧后长期开展本地推理工作流的现实担忧。
    - 一条从技术和商业角度出发的评论认为，**NVIDIA** 可能仍是少数有强烈动机发布开放模型的公司之一，因为开放权重模型能够带动本地 GPU 推理和部署需求。更广泛的担忧是，模型访问限制可能会减少开发者能够使用的、具有竞争力的本地模型的多样性。

  - **[北京并未考虑限制海外访问中国顶尖 AI 模型（对 Reuters 报道的反驳）](https://www.reddit.com/r/LocalLLaMA/comments/1upw37/beijing_is_not_looking_at_curbing_overseas_access/)**（热度：966）：**该帖子质疑一篇 [Reuters 报道](https://www.reuters.com/world/beijing-is-looking-curbing-overseas-access-chinas-top-ai-models-sources-say-2026-07-07/)，这篇报道声称北京可能限制海外访问中国顶尖 AI 模型。帖子认为，报道所提到的商务部与 **Alibaba**、**ByteDance** 和 **Z.ai** 等公司的会谈，实际上讨论的是对外国收购、投资、知识产权泄露，以及技术和人才外流的管控，而不是限制模型在海外的访问。帖子还引用了 [China International Commercial Court](https://ipc.court.gov.cn/zh-cn/news/view-5766.html) 的一份中国政策/法律文件作为证据，称中国的立场并不是全面限制开放权重模型的访问，而是推动“**可信且可控**”的开源；该文件还指出，严格限制开源权重跨境流动，可能会因为削弱中国开发者参与全球开源生态的能力而造成“*自我伤害*”。**评论者对 Reuters 的表述持怀疑态度，有人认为相关消息来源可能受到美国 AI 实验室利益的影响，并指出中国有战略动机继续向海外输出或开源模型，因为这能给美国现有 AI 公司施加压力。

    - 评论者认为，**开放权重模型的可用性对中国 AI 实验室具有战略意义**，因为这有助于推动全球采用，尤其是在美国市场，并直接与 **OpenAI** 和 **Anthropic** 等闭源模型提供商竞争。一种技术和市场层面的观点是，限制海外访问会削弱中国模型的分发能力和生态发展；而保持开放访问，则可以在主要融资或 IPO 叙事展开前，对依赖封闭 API 的竞争对手施加压力。
    - 几位评论者将 Reuters 的说法视为可能由竞争性信息战推动，而不一定反映真实政策。他们指出，限制海外访问最主要的受益者将是美国闭源模型供应商，因为这会减少来自中国前沿模型和开放权重系统的竞争。讨论没有提供基准测试或实施细节，重点放在模型访问策略，以及围绕全球部署展开的竞争态势上。




## 技术性较低的 AI 子版块总结

> /r/Singularity、/r/Oobabooga、/r/MachineLearning、/r/OpenAI、/r/ClaudeAI、/r/StableDiffusion、/r/ChatGPT、/r/ChatGPTCoding、/r/aivideo、/r/aivideo

### 1. Anthropic 的 J-Space 可解释性研究



  - **[Anthropic 在 Claude 中发现了一个“全局工作空间”——一个自行涌现的无声内部推理层](https://www.reddit.com/r/ClaudeCode/comments/1upchq0/anthropic_found_a_global_workspace_inside_claude/)**（热度：1267）：****Anthropic** 报告称，他们借助开源的 [Jacobian Lens](https://www.github.com/anthropics/jacobian-lens) 在 Claude 中识别出一个 `J-space`。它由一组紧凑的内部激活方向组成，似乎发挥着类似功能性*全局工作空间*的作用：其中出现的概念可以被模型报告出来、进行因果编辑，并在不同任务之间复用。在这篇[论文](https://www.anthropic.com/research/global-workspace)中，研究人员通过将 `spider → ant` 或 `France → China` 等概念替换，改变了模型在多个属性上的后续回答；而据称，消融 J-space 虽然保留了语言流畅度，却削弱了多步推理能力。论文还展示了一段算术过程：Claude 无需外部工具，便在不同层中逐步完成 `(4+17)*2+7`（`21` → `42` → `49`）。Anthropic 也将其视为一个安全信号：在模型输出之前，J-space 的激活中会浮现“fake”“fictional”“manipulation”“fraud”和“secretly”等潜在概念；即使在编造内容或刻意错配的 model-organism 设置中，也能观察到类似现象。不过，Anthropic 明确将这一结论限定为*访问意识*（access consciousness），并不涉及主观体验。**技术评论者总体上对此印象深刻，认为这些结果为反驳简单的“随机鹦鹉”观点提供了有力证据；在给出的热门评论中，没有出现实质性的​​方法论争论。

    - 一条技术含量较高的评论重点介绍了 Anthropic 对 Claude 逐层解释 `(4+17)*2+7` 的示例：到 `layer 58` 时，模型已将任务表示为算术问题；到 `layer 75` 时，已计算出 `4+17=21`；到 `layer 83` 时，已得到 `42`；在最后一层则得出 `49`。评论者强调，这说明模型无需外部工具就能进行内部多步计算，更符合某种潜在的推理/工作空间机制，而不是单纯模仿表层 token。
    - 一位评论者分享了一个讲解所提出的 **“J-space”/全局工作空间**概念的技术视频：[YouTube 讲解](https://m.youtube.com/watch?v=rKV5JcALQoQ\u0026pp=iggUQAFKEERqTmoxUnozeDY3MHdMdGg%3D)。另一位评论者指出，“*这个空间并非人为设计，而是在训练过程中涌现出来*”这一说法，与机器学习的核心前提一致：内部表征和行为是通过优化学习形成的，而不是依靠显式手写的结构。

  - **[Anthropic 刚刚报告称，LLM 会保留一些不说出来的隐藏想法——内部的“J-Space”](https://www.reddit.com/r/singularity/comments/1uptvgb/anthropic_just_reported_that_llms_have_hidden/)**（热度：794）：**一位 Reddit 用户总结了 **Anthropic** 关于 LLM 激活中的 `J-space`/类似全局工作空间子空间的论文（[论文](https://www.anthropic.com/research/global-workspace)）。其中，少量潜变量似乎能够支持可报告、被有意维持的多步推理状态；而大量流畅生成过程——包括语法、风格和事实回忆——基本可以绕过这一空间。他们还开发了 **Subtext**（[GitHub](https://github.com/ninjahawk/Subtext)），用于可视化生成前、按 token 分布的内部状态，并举例说明：对于 `12 + 5 = 1`，模型很早就会饱和激活“incorrect”；在输出开始前，还能观察到两跳式激活轨迹，例如第 `20` 层出现 `Italy`，第 `26` 层出现 `euros`。他们同时明确指出，这些结果证明的是功能上可访问的内部信息，而不是主观体验。**评论者的看法分成两派：一派认为这符合机制可解释性研究的预期，能够反驳简单的“随机鹦鹉”框架；另一派则怀疑，这种可视化可能只是普通的神经元/特征激活。不过，评论者普遍认为其中的算术过程和多跳时序结果更具技术意义。



- 评论者重点讨论了 Anthropic 关于 mechanistic interpretability 的主张：模型能够维持一些潜在的内部表征，而这些表征并不会直接反映在生成的 token 中。一种技术性解释认为，这超出了简单的“随机鹦鹉”框架：模型可能会在生成对应输出 token 之前，先激活诸如 `Italy` 表征或中间算术状态等概念。
- 一个受到重点关注的技术要点，是基础训练与后训练之间的区别：有人认为，在基础模型中，内部状态主要是围绕下一个 token 的预测进行优化的；而后训练似乎会促使模型形成更加持久的“身份”或第一人称框架。一位评论者指出，模型在读取输入时就可能在内部将其识别为 prompt injection，甚至早于任何输出的生成，这意味着模型能够独立于眼前的 token 生成过程，对用户文本进行潜在评估。
- 大家还对可复现性和实现细节感兴趣，包括有用户据称重新实现了论文中的部分实验，以及究竟是哪个模型生成了复现代码等问题。算术示例尤其受到关注，因为它们似乎表明模型存在中间计算结构，而不只是为了生成输出、在路径上激活某个概念神经元。


### 2. Claude Code 与 Autonomous Coding Agents

  - **[如今年收入达到 25 亿美元的工具，起初只是一个人在新工作第一周做的副业项目](https://www.reddit.com/r/ClaudeCode/comments/1upcvot/the_tool_that_now_generates_25byear_started_as_a/)**（热度：1034）：**这张[图片](https://i.redd.it/dbt5khhguobh1.jpeg)是一张**非技术性的品牌宣传图**：黑色背景上是复古像素风的“CLAUDE CODE”标志，用来配合讲述 Anthropic 的 Claude Code CLI 的起源故事。帖子称，Claude Code 始于 **Boris Cherny 加入 Anthropic 后第一周制作的原型**，在获得文件系统访问权限后迅速发展，到 2025 年 5 月已达到内部每日使用率的 `80%+`，并据称在约 6 个月内实现了 `$1B` ARR——不过，标题中的“每年 25 亿美元”并没有得到所提供文本的证实。**评论者对“偶然发现”的说法提出质疑，指出 Cline 等 coding agent 早在数月前就已存在，因此 Claude Code 更像是现有模式的内部实现和产品化版本，而不是全新的研究突破。其他评论则认为这张图片纯粹是审美设计，称街机风格的标志与“发烧梦般”的起源故事很相配。

    - 一位评论者质疑了帖子关于该项目具有创新性的说法，认为其核心思路——让 LLM agent 访问本地文件系统来执行 coding 任务——早在此之前 *“6+ 个月”* 就已经存在于 **Cline** 等工具中。他们认为这只是**现有 coding-agent 产品的内部克隆版**，而非研究突破。
    - 另一个技术讨论点集中在 **Claude 编写了 Anthropic 自身超过 `80%` 的代码**这一说法上，有人猜测 Claude desktop app 本身也可能大量由 Claude 生成，或通过“vibe coding”完成。这条评论反映出人们对 Anthropic 如今有多少生产代码是由 Claude-based coding agents 编写或搭建出来的感兴趣。

  - **[我给 GPT 5.5 一个空的 GitHub 仓库，让它自己想办法发展](https://www.reddit.com/r/ChatGPT/comments/1upb4vw/i_gave_gpt_55_an_empty_github_repo_and_told_it_to/)**（热度：795）：**这项实验让一个 LLM “agent”每小时自动唤醒一次（后来频率加倍），对一个最初为空的公开 GitHub 仓库进行操作：检查之前的状态、选择任务、编写并测试代码，然后提交；它最初的产物不是应用代码，而是项目路线图、变更日志、状态记录和决策日志等元项目文件。该仓库名为[**Autonomous Forge**](https://github.com/OmarH-creator/Autonomous-Forge)，目前是一个 pre-alpha、local-first 的 Python CLI，用于实现“repository-native autonomous software-improvement loops”。不过，它目前实现的范围主要是确定性的只读规划与审查功能，包括任务选择、符合策略的规划、提案/验证预览、仓库清单、运行前准备检查，以及运行历史预览；唯一会修改内容的功能是明确确认后执行的 `run-history-write`，它会修改 `.ai/run-history/`。它声明的安全边界相当保守：在未来路线图或策略支持完善之前，不会执行网络调用、测试/验证、差异检查、补丁生成、提交、推送或策略强制执行。**评论者大多关注其递归式的元设计：一个被要求构建某种东西的 autonomous agent，最后选择构建一个用于 autonomous repository maintenance 的工具；有人称之为“一个没有明确目标的工具”。主要的技术批评是，生成的路线图似乎过度偏重规划和流程类产物，而不是具体的实现进展。

    - 评论者指出，生成的项目概念 **“Autonomous Forge”** 本质上是一个元工具：这是一个由 AI 创建的 developer tool，旨在运行“repository-native autonomous software-improvement loops”。也就是说，面对一个空仓库时，模型没有去解决具体的产品问题，而是设计了用于 autonomous code generation/maintenance 的基础设施。一条技术批评认为，该路线图过度偏重**任务规划/编排**，而不是实现、评估或能量化的 developer-tool 功能。



### 3. Claude Fable 5 的访问权限与安全防护摩擦

  - **[Anthropic 将面向付费用户的 Fable 5 使用期限延长至 7 月 12 日](https://www.reddit.com/r/ClaudeAI/comments/1uq2aq5/anthropic_extending_fable_5_for_paid_users_till/)**（热度：1320）：**这是一张 **非梗图截图**，内容是 Claude/Anthropic 在 X 上发布的公告，宣布所有付费方案的用户都可以使用 **“Claude Fable 5”**，期限延长至 `7 月 12 日`：[图片](https://i.redd.it/t1hhakidhubh1.jpeg)。后续说明进一步澄清了配额政策：付费用户最多可以将每周使用限额的 **`50%`** 用于 Fable 5，之后必须使用额外 credits，或切换到其他 Claude 模型。**评论主要批评这次延期通知太突然，影响了配额规划：一些用户表示，他们原本以为使用权限会更早结束，所以赶忙用完每周的 Fable 配额，或者购买了额外 credits。还有一个反复出现的请求是，希望 Anthropic 在延长使用期限的同时，也重置使用限额。


  - **[Fable 5 在我的电脑上发现了真正的恶意软件，结果它自己的安全过滤器又把这条警告标记了](https://www.reddit.com/r/ClaudeAI/comments/1upu3e2/fable_5_found_actual_malware_on_my_pc_and_then_its_own_safety_filters_flagged_the_warning/)**（热度：1292）：**一名用户称，**Fable 5** 检查了 Windows 的 `Run` 注册表项，并发现一条可疑的 PowerShell 持久化命令——`powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden ...`——据称该命令会在用户登录时下载并执行远程脚本，模型将其判定为正在发生的系统入侵（[截图](https://preview.redd.it/2hv0yord1tbh1.png?width=1172\u0026format=png\u0026auto=webp\u0026s=6434cb2d41bb2474fa40398c24fa575b1f74c635)）。用户要求它删除特定的注册表持久化项后，据称清理操作成功了，但随后该会话因涉及“网络安全工作”而被安全过滤器标记，并降级到 **Opus 4.8**（[截图](https://preview.redd.it/402jnkkf1tbh1.png?width=1163\u0026format=png\u0026auto=webp\u0026s=7c6531edf9e593fd592109f91bd0ca45de8e6650)）。**评论者认为，这并不能很好地替代终端安全工具：PowerShell `Run` 键持久化是存在已久的恶意软件模式，传统的 AV/EDR 工具本应更可靠地发现它；而 LLM 可能只删除一个指标，却遗漏其他持久化机制。还有一名评论者提到一个相关的正面案例：AI 扫描代码库时，从 `security.md` 中发现了与生产环境相关的安全问题，而且没有触发降级。

    - 一名评论者认为，这类恶意软件并不新——*“可能已经存在了 12 年左右”*——传统杀毒软件应该能够通过特征码或启发式检测发现它。其技术结论是，**Fable 5 不应取代专用的终端安全工具**，因为 LLM 式扫描可能发现一个指标，却漏掉相关的持久化机制或其他恶意软件痕迹。
    - 一名用户表示，他们曾使用 **Fable** 扫描代码库。Fable 解析了其中的 `security.md`，并补充了多个用户认为与生产环境相关的新安全发现。用户随后修复了这些问题，这说明模型不仅能进行源代码 lint，还能根据项目文档和上下文，帮助发现可采取行动的应用安全问题。

  - **[好吧……我甚至不知道这居然有可能发生](https://www.reddit.com/r/ClaudeCode/comments/1updedl/well_shit_i_didnt_even_know_this_was_possible/)**（热度：688）：**图片是一张 Claude/Anthropic 的账单“Usage credits”页面截图，显示疑似超出消费限额的情况：尽管账户设置了 `$50` 的月度消费上限，但页面显示已消费 `$155.53`（已使用 `311%`），余额为 `-$119.11`（[图片](https://i.redd.it/l04wd5dfxobh1.png)）。发帖者表示，他们让 **Fable** 执行了几个任务，本以为达到上限后使用就会停止，但 Claude 仍继续计费。这引发了一个实际问题：Anthropic 的消费限额究竟是强制执行的硬上限，还是存在延迟、属于软性的记账控制。**评论者对 Anthropic 的客服支持表示怀疑，并建议如果确实被扣费，可以向信用卡公司申请拒付；同时他们也认为，设置好的月度消费上限似乎被忽略，这种情况“很奇怪”。

    - 用户报告了一起疑似 **Anthropic/Claude 账单控制漏洞**：据称，账户设置的月度消费上限没有生效，导致模型在达到预期上限后仍可继续使用并产生额外费用。一名评论者将这种情况与套餐配额行为进行了对比：普通套餐的使用量耗尽后，任务会在执行过程中停止；而付费超额/API 式使用可能会继续产生费用。
    - 有人建议联系 `support@anthropic.com`，将问题描述为**配置/账单漏洞**，并附上消费限额设置的截图。评论者建议先申请按比例退款，如果无效，再考虑向信用卡公司申请拒付；不过这样做可能导致账户被终止，或需要重新注册一个 Claude 账户。







# AI Discord 社区

很遗憾，Discord 今天终止了我们的访问权限。我们不会以目前这种形式恢复它，但很快会推出全新的 AINews。感谢你读到这里，这段旅程曾经很美好。