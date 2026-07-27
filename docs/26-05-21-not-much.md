---
companies:
- nvidia
- openai
- nous-research
date: '2026-05-21T05:44:39.731046Z'
description: "**RAEv2** 推进了以表征为先的词元化方法，收敛速度提升了 **10 倍以上**，生成效果也有所改善；相关测试涵盖**文本生成图像**和**世界模型**。**英伟达的
  Gated DeltaNet-2** 通过按通道设置门控机制，推动了线性注意力的发展；在 **13 亿参数**规模下的语言建模和推理任务中，其表现优于 **KDA**
  和 **Mamba-3**。关于**子词词元化**的研究表明，它的部分优势只有在大规模条件下才会显现；而数据过滤研究则指出，当计算量足够大时，在约 **1e30
  FLOPs** 的规模下，**完全不进行过滤**可能是最优选择。机械可解释性领域的最新进展提出，可以根据特征共同激活的模式对特征进行聚类，从而更好地理解其几何结构。OpenAI
  借助人工智能解决 Erdős 单位距离数学问题的突破，引发了人们对人工智能在数学研究中所扮演角色的讨论。在智能体基础设施中，Harness 仍然是提升能力的关键。
  \n"
id: MjAyNS0x
models:
- raev2
- gated-deltanet-2
- kda
- mamba-3
- dclm
people:
- 1jaskiratsingh
- recatm
- sainingxie
- ahatamiz1
- rasbt
- nousresearch
- tatsu_hashimoto
- goodfireai
- markchen90
- wtgowers
- memecrashes
- cloneofsimo
- lvwerra
title: '今天没发生什么特别的事。

  '
topics:
- representation-learning
- tokenization
- linear-attention
- long-context
- mechanistic-interpretability
- math
- data-filtering
- agent-infrastructure
- language-modeling
- commonsense-reasoning
---

**平静的一天。**

> 2026/5/20—2026/5/21 的 AI 新闻。我们查看了 12 个 subreddit、[544 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216)，没有进一步查看 Discord。[AINews 网站](https://news.smol.ai/)支持搜索过去的所有期刊内容。提醒一下，[AINews 现在已成为 Latent Space 的一个栏目](https://www.latent.space/p/2026)。你可以[选择接收或取消接收](https://support.substack.com/hc/en-us/articles/8914938285204-How-do-I-subscribe-to-or-unsubscribe-from-a-section-on-Substack)不同频率的邮件！




---

# AI Twitter 速览

**模型、基准测试与研究动态：RAEv2、Gated DeltaNet-2、数据过滤与开放数学**

- **RAEv2 与表示优先的 tokenization**：多位研究人员重点介绍了 **RAEv2**，认为它是 Representation Autoencoders 的重要后续工作，可用于统一的视觉理解与生成。[[@1jaskiratsingh](https://x.com/1jaskiratsingh/status/2057568174590304421)]表示，这一更新带来了**超过 10 倍的收敛速度提升**、更好的重建效果和生成效果，测试范围还扩展到了**文本生成图像和 world models**。[[@recatm](https://x.com/recatm/status/2057456332861567359)]发布的中文总结很好地提炼了三项主要发现：与只使用最后一层相比，将最后 **K 个 encoder 层**的输出相加，能够同时改善重建和生成效果，且不会增加推理成本；**RAE 与 REPA 在语义和空间结构方面具有互补性**；REPA 还可以重新表述为一种内部自引导机制，从而无需额外执行弱模型引导过程。[[@sainingxie](https://x.com/sainingxie/status/2057595509519311077)]还指出，除了 FID 之外，现在也出现了新的评估视角，这表明表示驱动的像素解码器仍有大量尚未挖掘的提升空间。

- **标准 attention 与 tokenizer 假设的替代方案**：NVIDIA 的 [**Gated DeltaNet-2**](https://x.com/ahatamiz1/status/2057586630450610673) 在线性 attention 中通过按通道设置的 gates，将 **erase** 和 **write** 操作解耦。在 **1.3B** 参数规模下，它在语言建模和常识推理方面超过了 **KDA** 和 **Mamba-3**，并且在 **RULER** 上展现出显著的长上下文检索优势；[@rasbt](https://x.com/rasbt/status/2057599925878169761)称其为混合 attention 方向中较为有趣的一项工作。关于 tokenization，[@NousResearch](https://x.com/NousResearch/status/2057610978934546805)发布了一项受控研究，探讨**子词 tokenization**为何有效：他们在一个 **1.7B byte-level** 流程中模拟了七种假设中的收益，但在这一规模下，只有**七项干预中的三项**改善了验证损失。另一方面，[@tatsu_hashimoto](https://x.com/tatsu_hashimoto/status/2057489411768803526)报告了 DCLM 上一个出人意料的 scaling 结果：当计算量足够大时，最佳的数据过滤方式可能是**完全不过滤**。预测显示，对于互联网规模的数据池，这一转折点大约会出现在 **1e30 FLOPs**；下游评估结果似乎存在噪声，但总体方向一致（[后续讨论](https://x.com/tatsu_hashimoto/status/2057489440273322447)）。

- **机制可解释性与几何结构**：[@GoodfireAI](https://x.com/GoodfireAI/status/2057487848258101551)认为，目前“模型以曲面流形进行思考，而 SAE 使用直线特征”的主流批评只说对了一部分。他们提出的修正方案是根据 **joint firing patterns** 对 SAE 特征进行聚类，通过**特征组**而不是孤立的原子特征来恢复几何结构（[讨论串续篇](https://x.com/GoodfireAI/status/2057487927089954962)、[文章](https://x.com/GoodfireAI/status/2057487939836502461)）。这是对当前 SAE 讨论的一项有益补充：它并不是要否定稀疏特征，而是提醒我们，解释工作应当从单个特征转向结构化的特征集合。

- **数学作为 AI 研究领域**：本期最受关注的科学讨论，围绕 OpenAI 据报道在 Erdős 单位距离问题上取得的结果展开。[[@markchen90](https://x.com/markchen90/status/2057517045575774598)]认为，这说明数学目前是最适合 AI 辅助实现研究突破的领域；而 [@wtgowers](https://x.com/wtgowers/status/2057536069218742518)指出，如果报道中所称的人类交互程度确实很低，那么这一结果确实很有意思。随后，质疑和关于基准测试可操纵性的担忧迅速主导了讨论：[@memecrashes](https://x.com/memecrashes/status/2057478155246440929)开玩笑说，这项结果“甚至没过 3 小时就被人类淘汰了”；[@cloneofsimo](https://x.com/cloneofsimo/status/2057486750004756524)则指出，围绕什么才算合法的 AI 数学，出现“不断抬高标准”的情况其实早在预料之中。这里有一个有意思的技术层面观察：数学仍然是 AI 协同研究相对清晰的前沿领域，因为其产出可以被验证、讨论并进一步拓展。

**Agents、Harnesses 与开发者工具：Codex、Gemini、Devin 及 Agent 基础设施**



- **Harness 仍是能力提升的重要来源**：[[@lvwerra](https://x.com/lvwerra/status/2057476832664953225)] 发布了 **physics-intern**，这是一个用于科学问题的 harness，可将 **Gemini 3.1 Pro** 的得分从 **17.7 提升到 31.4**，并在该测试设置下超过 **GPT 5.5 Pro**。值得注意的是，GPT 5.5 Pro 本身并没有从这个 harness 中受益，这说明不同模型对脚手架技巧的吸收方式可能存在差异。同样，[[@KLieret](https://x.com/KLieret/status/2057471442066030795)] 让 **mini-swe-agent** 可以运行在 **ProgramBench** 上，明确希望推动软件工程 Agent 领域的 harness 创新。

- **Agent 设计模式正从“优先使用单 Agent”逐渐发展为显式的子 Agent 编排**：[[@cwolferesearch](https://x.com/cwolferesearch/status/2057486293882282293)] 给出了一个实用总结：先从 **single-agent systems** 开始，只有当工具数量过多或 prompt 变得难以管理时，才转向 **manager/sub-agent** 或去中心化的多 Agent 拓扑。这一建议也与子 Agent 用户分享的更多实际观察相吻合：[[@andrew_locke](https://x.com/andrew_locke/status/2057537633555993058)] 形容 Cognition 的 sub-Devin 工作流是一次重大跃升：它把过去看起来需要 **2+ 个工程师周** 的工作压缩到了几小时内。

- **Codex 在模型之上构建了相当完整的产品层**：OpenAI 的“Codex Thursday”更新，与其说是一些孤立的功能，不如说体现了 coding Agent 的发展方向。[[@OpenAIDevs](https://x.com/OpenAIDevs/status/2057530207976989179)] 发布了 **Appshots**，能够同时捕获 Mac 应用窗口的截图和文本，从而提供更丰富的工作上下文；他们还新增了 **团队插件共享**（[链接](https://x.com/OpenAIDevs/status/2057530212339097994)）和更详细的 **组织分析**（[链接](https://x.com/OpenAIDevs/status/2057530213974814844)）。更重要的系统层变化是远程使用电脑：[[@OpenAIDevs](https://x.com/OpenAIDevs/status/2057536706778378692)] 表示，即使 Mac 处于锁定状态，Codex 现在也能让你通过手机安全地使用 Mac 上的应用。这是一个强烈信号，表明 Agent 产品形态正从聊天式 IDE 转向持久化、跨设备的操作员工作流。

- **Gemini 的 Agent 和工具能力正在迅速扩展**：[[@OfficialLoganK](https://x.com/OfficialLoganK/status/2057460544643404125)] 提到，**Gemini 3.5 Flash** 在 **APEX-Agents-AA** 上排名 **#1**，超过了更大的模型。在实际应用方面，[[@_philschmid](https://x.com/_philschmid/status/2057513254856151339)] 展示了一个 GitHub issue 分流 Agent：它只通过 **一次 Gemini API 调用** 构建完成，且不依赖 orchestration framework；[[@skalskip92](https://x.com/skalskip92/status/2057502215506473121)] 则展示了 Gemini 3.5 Flash 如何通过一次多模态 API 调用，取代用于车道线和车辆推理的定制 vision pipeline。Google 也在扩展可执行操作的范围：**Daily Brief**（[公告](https://x.com/GeminiApp/status/2057500470147698936)）以及与 **OpenTable、Canva 和 Instacart** 连接后的操作（[公告](https://x.com/GeminiApp/status/2057550225863246236)），本质上都是面向消费者的 Agent 工作流。

- **开发者基础设施正围绕检索、流式处理、沙箱和安全边界逐渐整合**：Weaviate 在数据库内部内置了 **MCP server**，这样 coding Agent 无需额外进程，就能导入代码仓库并使用 **BM25 + 向量混合检索**（[公告](https://x.com/weaviate_io/status/2057476556449010024)）。LangChain 同时推出了用于控制 Agent 与外部世界边界的 **sandbox Auth Proxy**（[公告](https://x.com/LangChain/status/2057508777759236401)），以及一种新的 **typed streaming protocol**：它将工具、子 Agent、媒体和中断作为一等投影进行渲染，而不再只是 token 流（[概览](https://x.com/bromann/status/2057507753191518602)）。vLLM 的 **Elastic Expert Parallelism** 也是一项值得关注的系统工作：[[@vllm_project](https://x.com/vllm_project/status/2057602243860574463)] 介绍了如何在无需完整重启的情况下，动态调整 MoE 的 **DP/EP 拓扑**，并通过 **NVLink/RDMA** 进行 GPU 之间的直接传输——这不仅对扩展能力很重要，也为未来实现容错 serving 打下了基础。

**基础设施、算力与 AI 商业信号：Modal、Turbopuffer、Hark 与算力竞赛**



- **基础设施层迎来了最清晰的“钱就在那里”的时刻之一**：[‍@Sirupsen](https://x.com/Sirupsen/status/2057470756070781400) 表示，**turbopuffer** 在 3 月的年化营收跑到了 **1 亿美元**，而就在 **19 个月前**，这一数字还只有 **100 万美元**；与此同时，公司仍然**盈利**，且融资金额 **不到 100 万美元**。公司的定位直接而且正逢其时：顶尖团队都知道，“AI 真正施展魔法的时刻，是它能够吸收恰到好处的上下文”，这让许多产品差异化最终都变成了一个**搜索/检索问题**（[后续说明](https://x.com/Sirupsen/status/2057470791516844188)）。这也与 [@swyx](https://x.com/swyx/status/2057543654340710556) 更广泛的观点一致：财富正在流向“无聊”的 AI 基础设施，而不只是那些光鲜亮丽的前沿研究领域。

- **Modal 融资规模可观，继续展现出成为 AI 云领域核心赢家的潜力**：[‍@bernhardsson](https://x.com/bernhardsson/status/2057530320790995262) 宣布完成 **3.55 亿美元 C 轮融资，估值达到 46.5 亿美元**。投资者和用户强调的是同一个判断：从底层开始，为 AI 工作负载重建云技术栈，同时兼顾高性能与开发者体验（[Redpoint](https://x.com/Redpoint/status/2057532087570166134)、[用户背书](https://x.com/mathemagic1an/status/2057534253790097788)）。与此同时，其他迹象也表明，原生面向 Agent 的计算正在成为一个独立品类；[@latentspacepod](https://x.com/latentspacepod/status/2057565350187995260) 总结了 Daytona 的卖点：提供**延迟 60 毫秒的沙盒环境**、**75 秒启动 5 万家初创公司的环境**，以及如今约占使用量**一半**的 RL/evals 工作负载。

- **计算资源仍是战略瓶颈，而且市场似乎已经形成分层**：[@AymericRoucher](https://x.com/AymericRoucher/status/2057492189626720729) 梳理了一套很有参考价值的计算资源分类：**美国领先者**（OpenAI、Anthropic、Google，以及正在加入的 Meta/xAI）处于**多吉瓦**级别；**中国巨头**正从数百 MW 扩展到多 GW，并且越来越多地采用国内技术栈；**欧洲竞争者**则包括 Mistral 这样的公司，目前约为 **90 MW**，目标是在 **2029 年达到 1 GW**。具体数字仍有争议，但这一框架与 [@EpochAIResearch](https://x.com/EpochAIResearch/status/2057499893854536185) 的观察一致：即使 OpenAI 引发了最近这轮计算资源建设热潮，前沿实验室使用的计算能力仍远低于全球总容量，因此未来还能在多大程度上继续加速扩张，仍是一个开放问题。与此同时，组件成本结构也在继续向内存倾斜：[@EpochAIResearch](https://x.com/EpochAIResearch/status/2057531410030997789) 报告称，**HBM** 占 AI 芯片组件总支出的比例已从 2024 年第一季度的 **52%** 上升到 2025 年第四季度的 **63%**。

- **资本不仅流向基础设施，也在押注交互界面和硬件**：[@adcock_brett](https://x.com/adcock_brett/status/2057462134989263047) 宣布 **Hark** 以 **60 亿美元估值**融资 **7 亿美元**，资金将用于 GPU 基础设施、未来模型开发、硬件，以及多模态/个人智能产品。除了招聘方向外，目前披露的细节并不多，涉及 foundation models、基础设施、语音、computer-use agents 和硬件等领域；但融资规模本身已经说明，投资者对垂直整合式 AI 设备项目抱有浓厚兴趣。Hark 还宣布，**F.03** 已完成一次持续 **200 小时**的不间断自主运行（[公告](https://x.com/adcock_brett/status/2057651077928145235)），不过目前披露的技术细节还不足以评估其底层机器人技术栈。

**多模态、视频、生物学与机器人：Runway、碳排放、地球模型与开放式人形机器人**

- **视频编辑和生成正变得更具组合性**：Runway 发布了 **Aleph 2.0** 和全新的 **Edit Studio**，用户可以编辑单帧画面，并将编辑效果传播到视频的其余部分（[Runway](https://x.com/runwayml/status/2057530497597600169)、[产品负责人](https://x.com/iamneubert/status/2057535909524824226)）。这实际上是将多模态开发者关注的“基于参考内容的编辑传播”问题产品化。另一方面，Alibaba 的研究人员推出的 **MIGA** 被 [@HuggingPapers](https://x.com/HuggingPapers/status/2057506246899724355) 介绍为一种**无需训练**即可生成**无限帧**视频的方法，并通过两阶段对齐机制保证时间一致性。在开源数字人方面，Meituan 发布了 **LongCat-Video-Avatar 1.5**：用 **Whisper-Large** 替代 Wav2Vec2，支持 **8 步推理**，提升长视频中的身份一致性，并增强了对更广泛风格领域的泛化能力（[公告](https://x.com/Meituan_LongCat/status/2057494106889486646)）。


- **生物学和 Earth observation 领域的 Foundation models 仍在变得更加实用**：Hugging Face Bio 的 **Carbon** DNA model 家族获得了后续演示和基础设施验证。[@LoubnaBenAllal1](https://x.com/LoubnaBenAllal1/status/2057488110263435640)重点介绍了其在**序列设计、变异效应预测和学习型表征**方面的应用；与此同时，[@Shekswess](https://x.com/Shekswess/status/2057468970471448787)展示了 **Carbon-500M、3B 和 8B** 如何在第一天就通过 NxD Inference，在单台 **Trainium2 trn2.3xlarge** 上完成编译并运行。在地理空间建模方面，[@cgeorgiaw](https://x.com/cgeorgiaw/status/2057481909802774664)报告称，**OlmoEarth v1.1** 通过将多分辨率 Sentinel-2 输入的 tokenization 改为生成**少 3 倍的 tokens**，利用二次计算量节省，实现了**成本降低 3 倍、速度提升 3 倍**。

- **Open robotics 正变得更容易真正构建起来**：Hugging Face 的 **LeRobot Humanoid** 之所以受到关注，是因为它是真正完整的全栈开源发布，而不只是一个展示型 demo。[@robotsdigest](https://x.com/robotsdigest/status/2057507896129380581)和[@lukas_m_ziegler](https://x.com/lukas_m_ziegler/status/2057515219946205399)都强调了同一套内容：约 **2,500 美元**、**3D 打印**、完整的硬件/CAD、校准与运行时环境、仿真、系统辨识工具，以及训练流水线。关键不只是价格亲民，更在于它便于维修，也能加快真实 robot learning 工作流中的迭代速度。

**热门推文（按互动量排序）**

- **OpenAI / Codex 产品扩展**：[即使 Mac 处于锁定状态，Codex 也能通过手机安全地使用 Mac 上的应用](https://x.com/OpenAIDevs/status/2057536706778378692)，以及用于提供更丰富应用上下文的 [Appshots](https://x.com/OpenAIDevs/status/2057530207976989179)。
- **基础设施领域的赢家**：[turbopuffer 年收入运行率达到 1 亿美元，已实现盈利，融资额不足 100 万美元](https://x.com/Sirupsen/status/2057470756070781400)；[Modal 完成 3.55 亿美元 C 轮融资，估值达到 46.5 亿美元](https://x.com/bernhardsson/status/2057530320790995262)；[Hark 融资 7 亿美元，估值达到 60 亿美元](https://x.com/adcock_brett/status/2057462134989263047)。
- **引发广泛技术共鸣的研究讨论**：[OpenAI 与 Erdős 相关的数学成果讨论](https://x.com/markchen90/status/2057517045575774598)；[RAEv2 发布](https://x.com/1jaskiratsingh/status/2057568174590304421)；[关于 LM 数据筛选“无过滤”扩展结果的讨论](https://x.com/tatsu_hashimoto/status/2057489411768803526)。
- **Agent 能力的发展趋势**：[Gemini 3.5 Flash 在 APEX-Agents-AA 上排名第一](https://x.com/OfficialLoganK/status/2057460544643404125)；[Gemma 4 E4B 通过 Argent 在设备端驱动 iOS simulator](https://x.com/googlegemma/status/2057570113390551452)；[Windows 版 Devin](https://x.com/cognition/status/2057496130225668360)。


---

# AI Reddit 摘要

## /r/LocalLlama + /r/localLLM 摘要

### 1. Qwen 3.7 Max 基准测试与 27B 动向

  - **[Qwen 很可能会再发布一个 27B 模型](https://www.reddit.com/r/LocalLLaMA/comments/1tiwnpc/qwen_will_release_another_27b_with_high/)**（活跃度：1613）：**这张[图片](https://i.redd.it/g5uabdvdic2h1.jpeg)是 X/Twitter 上一段交流的截图，其中 **xiong-hui / Barry Chen** 表示自己正在*“等待确切的路线图”*，但他认为 **Qwen 很可能会再发布一个 `27B` 模型**，并指出如今对他们来说，再做一个 27B 已经*“不难了”*。结合标题和所链接的帖子来看，这并不是官方公告，而是关于可能成为公认“奇迹模型”**Qwen 3.6 27B**后继版本的路线图暗示或传闻。评论者主要讨论部署的实际可行性：一些拥有 `16GB` 显存的用户更偏好 `35B` MoE / `A3B` 风格的模型，因为相比高量化的 dense `27B`，这类模型通过 CPU/GPU 混合推理可能更容易使用。还有人猜测更大的 MoE 变体，例如假想中的 **Qwen 3.7 `122B-A10B`**。

    - 多位评论者关注**受显存限制的本地推理**，认为在 `16GB` GPU 上，dense `27B` 模型很难以“足够好的量化版本”运行；而假想的 **Qwen `35B` MoE / A3B 风格模型**则可能凭借更少的激活参数，或通过 CPU/GPU 混合推理，继续保持可用。讨论认为，Qwen 之前采用的小激活参数量 MoE 设计，对于使用普通游戏本或显存有限的用户十分重要。
    - 一位用户希望 Qwen 推出 `50B–80B` 范围内更大的 **dense Qwen 模型**，并表示当前的 `27B` 配合 **MTP** 已经足够快，因此他们愿意牺牲推理速度来换取更多参数，以及可能更强的能力。另一位用户则提出了推测性的 **Qwen `3.7 122B-A10B`** MoE 风格目标，反映出大家对总参数量很大、但每个 token 激活参数相对较少的模型感兴趣。



  - **[Qwen3.7 Max 获 Artificial Analysis 评分，27B/35B 进入等待名单](https://www.reddit.com/r/LocalLLaMA/comments/1tie6gy/qwen37_max_scored_by_artificial_analysis_27b35b/)**（热度：614）：****Qwen3.7 Max** 已出现在 **Artificial Analysis** 排行榜中，位列第 `5`，据称与 **GPT-5.4 xhigh** 基本持平，并略高于 **Gemini 3.5 Flash**。帖子指出，**Qwen3.6 27B** 比其 Max 版本低 `6` 分，这让人更加期待即将推出的 **Qwen3.7 27B/35B** 变体，认为它们的性能可能接近更大型的 Max 模型。评论者主要在等待开放权重版本，并认为 Qwen 能与前沿实验室竞争是一个积极信号，但也对 Max 模型不开源感到失望。另一个技术层面的疑问是，Qwen 是否已经解决了此前被指出的“过度思考”问题。

    - 评论者正在等待 **Qwen3.7** 是否会推出开放权重的 `27B`/`35B` 变体，但有一种技术猜测认为，可能不会单独发布 `27B`：**Qwen 3.7 可能是一个私有的 `390B` MoE 风格模型，激活参数量为 `A30B`**，更像是面向大型封闭式部署的模型，而不是小型开放权重检查点。
    - 有几条评论关注 **Qwen3.7 Max** 相比 Qwen 3.5/3.6 是否实现了真正的架构升级，还是主要进行了一次新的微调。大家感兴趣的是，Alibaba 是否改进了底层模型设计，还是只是在现有架构上进一步挖掘了基准测试性能。
    - 一个反复出现的疑问是，Qwen 团队是否解决了模型的“过度思考”问题——这可能指的是过于冗长的推理过程，或不必要的类似思维链的反复推演。尽管这有时能提高基准测试分数，但也会增加延迟和成本，并影响用户体验。

  - **[等待 Qwen 3.7 开放权重……新的王者已经到来……](https://www.reddit.com/r/LocalLLaMA/comments/1tjvz6l/waiting_for_qwen_37_open_weight_the_new_king_has/)**（热度：577）：**这张[图片](https://i.redd.it/j8qkty82qj2h1.png)是一张 **Qwen3.7-Max** 的基准测试宣传图，链接指向 [Qwen3.7 博客](https://qwen.ai/blog?id=qwen3.7)。图中显示，在 `Terminal-Bench 2.0`、`SWE-bench Pro`、`MCP-Atlas`、`HLE`、`Apex`、`IFBench` 和 `SuperGPQA` 等多项任务上，它都领先于 **Qwen3.6-Plus**、**DS-V4-Pro Max**、**GLM-5.1**、**Kimi K2.6** 和 **Claude Opus-4.6 Max**。从技术意义上看，这张图将 Qwen3.7-Max 定位为能够与 Opus 级系统竞争的前沿闭源/API 模型；而评论者特别希望看到一个开放权重的 MoE 版本，例如具备 `512k` 上下文的 `3.7-122B-A17B`，或采用 `MXFP4`/`NVFP4` 等低比特格式的 `397B A17B` 变体。**评论者普遍认为 **Qwen3.7-Max** 本身不太可能开放权重，并指出“*Qwen 从未将 Max 系列开放权重*”。也有人对可能推出的大型开放 MoE 模型感到兴奋，认为对于拥有高端多 GPU 配置的用户来说，它可能成为“*家用版 Opus*”。

    - 多位评论者提醒，传闻中的模型很可能属于 **Qwen Max 系列**；从历史来看，**Qwen 从未将 Max 系列模型以开放权重形式发布**。有位用户特别警告，不要把 Max 的基准测试结果简单推断到假设中的 `27B` 等较小开放模型上，因为两者之间的能力差距可能会非常明显。
    - 围绕硬件的猜测主要集中在可能推出的 `Qwen 3.7-122B-A17B`，以及它对 **MTP**、`MXFP4` 量化和 `512k` 上下文的支持。评论者认为，这种模型可能很适合在 **AMD Strix Halo** 级别的系统上进行本地推理。另一位评论者希望看到 `397B-A17B` 版本，并指出此前的 `Qwen 3.5` `NVFP4` 版本据称可以装入 `4x RTX 6000 Pro` GPU，同时还留有足够的显存余量，能够在 `200k` token 下同时运行大约 `10` 个会话。
    - 有人怀疑 Alibaba/Qwen 是否会发布其最强的本地模型，因为这样做可能会影响托管模型的商业化。一位评论者提到，Qwen 在 4 月从“颠覆”转向 **前沿模型竞争和商业化**，这意味着即使基准测试结果很亮眼，高能力开放权重模型的发布也可能变得不那么容易。


### 2. Qwen 3.6 35B MTP 量化性能



  - **[使用 Qwen3.6 35B A3B 和 ik_llama.cpp，在 12GB VRAM 下达到 110 tok/s](https://www.reddit.com/r/LocalLLaMA/comments/1tjh7az/110_toks_with_12gb_vram_on_qwen36_35b_a3b_and_ik/)**（活跃度：455）：**该帖子使用 byteshape 的 [`IQ4_XS` `4.19 bpw` GGUF](https://huggingface.co/byteshape/Qwen3.6-35B-A3B-MTP-GGUF)，在 **RTX 4070 Super 12GB + Ryzen 7 9700X** 上对 **Qwen3.6-35B-A3B-MTP** 进行了基准测试，配置包括 `131072` 上下文、`q8_0` KV cache、MTP `draft-max=3` 和 `draft-p-min=0.75`。从 [`llama.cpp`](https://github.com/ggml-org/llama.cpp) 切换到 [`ik_llama.cpp`](https://github.com/ikawrakow/ik_llama.cpp) 后，报告的平均速度从 `89.76 tok/s` 提升到 `110.24 tok/s`（`+23%`）。尽管更新结果中的 MTP 总体接受率反而有所下降（`0.9393` → `0.8749`），但这表明性能提升更可能来自后端或卸载效率，而不只是接受率。作者指出，将 GPU 用作无显示输出的次要 GPU 可以最大化可用 VRAM，并建议在 `ik_llama.cpp` 中使用 `--fit --fit-margin 1664`；如果发生 OOM，可将其提高到 `1792`/`2048`。**评论者询问了确切的 `llama.cpp` 命令，并指出近期已经合并了多个与 MTP 相关的 `llama.cpp` PR，因此结果可能受到版本影响。一位用户还分享了一个适用于没有 iGPU 的 CachyOS/KDE Wayland 用户的技术性解决方案：通过 `LIBGL_ALWAYS_SOFTWARE=1 GALLIUM_DRIVER=llvmpipe` 以软件渲染方式启动 Plasma，将空闲 VRAM 从 `>1024 MB` 降至约 `126 MB`，但代价是合成器特效变慢或被禁用。

    - 一位 CachyOS/KDE Wayland 用户分享了单 GPU 系统的节省 VRAM 方案：创建一个自定义 SDDM 会话，并使用 `LIBGL_ALWAYS_SOFTWARE=1`、`GALLIUM_DRIVER=llvmpipe` 和 `KWIN_COMPOSE=Q` 启动 Plasma，强制 KDE 合成器改用 CPU 渲染。据其报告，在普通 KDE Wayland 环境下，空闲 VRAM 超过 **`>1024 MB`**；而在 CPU 渲染会话中则约为 **`~126 MB`**，这样可以为模型推理释放近 1 GB VRAM，但动画会非常卡顿或被禁用。
    - 多位评论者关注了基准测试方法，要求提供确切的 `llama.cpp` 命令，并指出 **上游 `llama.cpp` 在前 24 小时内刚合并了与 MTP 相关的 PR**，这可能会显著影响对比结果。一种技术推测认为，`ik_llama.cpp` 的加速来自更高的 speculative/MTP 接受率：在 `ik_llama.cpp` 中**从未低于 `0.790`**，而在 `llama.cpp` 中最低可达 **`0.477`**，因此有人质疑两边的设置是否完全等效。
    - 有人对 `IQ4_XS` 在内存占用和质量之间的权衡感兴趣，并认为它可能是这一配置下占用内存最低的 Q4 量化选项。一位评论者询问这种量化会造成多大程度的智能下降，并要求提供最终的 VRAM/RAM 分配情况；对于仅有 **12 GB VRAM** 的 **Qwen3.6 35B A3B** 来说，这一点尤其重要。

  - **[Qwen 3.6 35B GGUF：不同 GPU 和 CPU 上的 NTP 与 MTP 量化结果](https://www.reddit.com/r/LocalLLaMA/comments/1tipihx/qwen_36_35b_gguf_ntp_vs_mtp_quantization_results/)**（活跃度：364）：**这张图片是一个**技术基准测试图表**，并不是表情包：它展示了 [RTX 4090 性能-质量气泡散点图](https://i.redd.it/xjctv0okab2h1.png)，比较了 **ByteShape Qwen 3.6 35B GGUF** 的 NTP/MTP 量化版本与 Unsloth、Bartowski、Mudler 和 AesSedai 方案在平均 TPS、准确率和 BPW 方面的表现。在帖子所讨论的背景下，它体现了一个主要结论：对于 NTP，*“选择能装下的最大量化版本”* 可能仍然具有竞争力；而 MTP 可以将 GPU 生成吞吐量提升约 `20–40%`，但会增加内存压力，因此不建议用于 CPU。**评论大多积极且偏实用：一位 CPU 混合推理用户确认自己遇到了严重的 MTP 降速，这与 ByteShape 关于 CPU 的测试结果一致；同时他还询问是否计划推出更高质量的 `Q6` GGUF 版本。



- 一位采用 **CPU 混合方案**的用户反馈，在 Qwen3.6-35B 上使用 **MTP** 时出现了**“令人难以置信的卡顿”**，这与帖子中的结论一致：在 CPU/GPU 混合配置下，MTP 的性能可能反而下降。他们还询问是否会发布 **Q6 GGUF** 量化版本，并表示对于这个模型，他们不会使用低于 Q6 的量化级别。
- 一位评论者质疑 **NTP** 的测试方法，认为这里的 NTP 指的是 llama.cpp 的 `--spec-type ngram-mod`，并指出主线版本的 **llama.cpp** 似乎可以通过 `--spec-type ngram-mod,draft-mtp` 同时运行 **ngram speculative decoding** 和 MTP。他们认为，这项对比可能并不是严格的 NTP 与 MTP 二选一，还提到了 `--spec-ngram-mod-n-match 24`、`--spec-ngram-mod-n-min 12`、`--spec-ngram-mod-n-max 48` 以及 `--spec-draft-n-max 3` 等参数。
- 有人在 **RTX 4070 Super 12GB** 上，使用 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp) 对 **Qwen3.6-35B-A3B-IQ4_XS-4.19bpw MTP** 进行了测试，报告称平均速度为 **`110.24 tok/s`**，比 `Qwen3.6-35B-A3B-UD-IQ4_XS MTP` 快了约 **20 tok/s**。该测试使用了 [mtp-bench.py](https://gist.github.com/am17an/228edfb84ed082aa88e3865d6fa27090/)，其中 `aggregate_accept_rate=0.8749`、`total_predicted=1592`、`total_draft=1127`，以及 `total_draft_accepted=986`；评论者特别指出，`--fit`、`--fit-margin 1664`、`--multi-token-prediction`、`--draft-p-min 0.75` 和 `--draft-max 3` 是几个关键的调优参数。


### 3. 开放权重发布与下架争议

  - **[Heretic 收到了 Meta, Inc. 发出的法律通知](https://www.reddit.com/r/LocalLLaMA/comments/1tjmvx6/heretic_has_been_served_a_legal_notice_by_meta_inc/)**（活跃度：2124）：**Heretic Free Software Project** 表示，他们收到了代表 **Meta Platforms, Inc.** 的服务商发来的法律通知邮件，并已移除包含 Meta **Llama** 模型衍生版本的模型权重仓库。帖子将此次下架描述为合规行动，同时宣布通过官方 [Codeberg 镜像](https://codeberg.org/p-e-w/heretic) 分散基础设施，并计划采取“技术措施”，在不依赖单一托管服务商的情况下，继续保障访问由 Heretic 创建的模型。评论者普遍批评 Meta 的执法方式虚伪，理由是 Meta 自身也被指控使用了受版权保护的训练数据；他们还调侃了帖子中的一句话：Llama 在 LM Arena 上落后于来自 23 家竞争对手的 `168` 个模型。整体来看，讨论主要集中在政治和法律层面的反应，而不是技术争论。

    - 评论者特别提到了帖子引用的 LM Arena 表述：**Meta 的 Llama 系列并不处于最顶尖行列**，在排名前 200 的语言模型中，被描述为“*仅落后于来自 23 家竞争对手的 168 个模型*”。从技术角度看，这场围绕名称展开的法律争端，被拿来对比 Meta 模型发布速度和排行榜竞争力被认为停滞不前的现状。

  - **[Cohere 的 Command-A 系列模型后来怎么样了？](https://www.reddit.com/r/LocalLLaMA/comments/1tizmar/re_what_ever_happened_to_coheres_commanda_series/)**（活跃度：669）：****Cohere** 发布了 **Command A+**，这是其首个采用 **Mixture-of-Experts (MoE)** 架构的开放权重模型。该模型被定位为 Command 系列高效的后继或延续版本，重点强调低延迟和响应速度，而不只是追求基准测试的最高成绩；具体信息见 Cohere 的[发布公告](https://cohere.com/blog/command-a-plus)。该模型采用 **Apache 2.0** 许可证发布。Cohere 表示，他们进行了大量量化工作，使模型能够在 `1–2 GPUs` 上实际部署，目标用户包括 Agent 化和企业级工作负载场景，以及规模较小的开发团队。评论者总体持积极态度，认为早期的 **Command R+** 在当时表现异常出色，尤其适合创意工作和企业风格的规划任务；他们也欢迎 Cohere 回归，认为这有利于模型生态的多样性。社区最主要的技术诉求，是希望尽快提供用于本地推理的 **GGUF** 量化版本。

    - 有评论者认为，由于缺少标准基准测试结果，也没有与当前同规模模型进行对比，这次发布的竞争力仍然存疑；他们特别提到了 **MiniMax M2.7** 和 **Mimo V2.5**，认为这两者是目前的 SOTA 基准。该评论者指出，如果模型质量没有明确展现出竞争力，仅依赖一张引用的 **Artificial Analysis 基准测试** 图片（<https://preview.redd.it/vjex3axl8d2h1.png?width=1224&format=png&auto=webp&s=08e9c90188bf9b42d4f049991624b4e180cf566d>），可能不足以推动用户采用。
    - 几位用户询问了部署的便利性，包括是否会提供 **GGUF 量化版本**，以及 Cohere 是否计划发布类似旧款 `command-r7b` 的更小型 Command 系列模型，以便在消费级 GPU 上运行。社区关注的技术重点，是本地推理是否可行，而不是只能通过 API 或企业级规模进行部署。
    - 一位评论者特别指出，早期的 **Command R+** 在当时的**创意工作流和企业资源规划任务**中表现异常出色。这意味着，用户评估新的 Command-A 系列时，比较的不仅是通用聊天机器人基准测试成绩，也包括前代模型在实际长上下文和企业应用中的实用价值。




## 技术性较低的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. Claude Code 工作流与 Anthropic 训练



  - **[我是一名拥有十年经验的软件工程师。我用 Claude Code 在手机上为自己的副业项目进行 vibe coding，而且完全不阅读任何代码。真的很有趣。以下是我遵循的规则：](https://www.reddit.com/r/ClaudeAI/comments/1tj2i90/im_a_software_engineer_with_a_decade_of/)**（活跃度：1900）：这篇帖子提出了一套**风险可控的“vibe coding”工作流**，适用于在不直接阅读生成代码的情况下使用 **Claude Code** 开发副业项目：先从计划模式开始，反复检查并澄清计划；将任务拆分到自己能够在脑中理解的规模；要求 Agent 生成测试用例；每完成一个计划就提交到 `git`；在允许 Agent 访问数据库前先备份数据库；并使用 **Chrome DevTools MCP** 等浏览器工具或 E2E 工具进行实时验证。对于复杂变更，作者建议并行使用多个审查 Agent，分别进行**计划评审**、**安全审查**和**测试审计**；只有在计划、测试和回滚结构都准备好之后，才切换到自动模式。热门评论大多认可这套工作流，认为它是相对稳妥的 Agent 编程模式，尤其赞同“*如果一个计划大到无法装进你的脑子里，那它就太大了*”这条规则。评论者还建议使用 [`superpowers` 技能集](https://github.com/obra/superpowers/tree/main)让流程变得可重复，并严格限制 Agent 的工作范围：一次只做一个变更、对应一个预期测试，并设置一个回滚点，同时在提示词中明确说明不应触碰的内容。

    - 多位评论者强调，应将 Agent 的工作限制在**小而可验证的范围**内：做到“一次一个变更、一个预期测试、一个回滚点”，并在提示词中明确写出 Agent **不应修改的内容**。其技术上的理由是，更小的计划可以降低调试复杂度；当使用 Claude Code 或类似的编程 Agent 时，如果出现失败，也更容易定位问题。
    - 一位评论者建议使用 [`superpowers` 技能集](https://github.com/obra/superpowers/tree/main)，将这套工作流转变为可重复执行的脚本化流程。这是一个由 GitHub 项目提供的可复用 Agent 工作流/技能集合。当项目从单次提示生成发展到迭代式开发后，这种方式可以减少“vibe coding”的随意性。

  - **[Anthropic 正式推出 13 门以上的免费 AI 课程，并提供证书（包括 Agentic AI 和 Claude Code！）](https://www.reddit.com/r/ClaudeAI/comments/1tjpfh8/anthropic_officially_launched_13_free_ai_courses/)**（活跃度：1585）：**Anthropic** 提供了免费的官方培训课程目录（可通过 [anthropic.com/learn](https://www.anthropic.com/learn) / Anthropic Skilljar 访问），并颁发证书。课程内容涵盖 **MCP / Agentic AI**、**Claude Code**、Claude API 的使用，以及面向 **Amazon Bedrock** 和 **Google Cloud Vertex AI** 的企业部署路径。讨论中特别提到的技术重点包括 **Model Context Protocol（MCP）** 课程，其中包含关于 `STDIO` 和 `StreamableHTTP` 传输方式的进阶内容；此外还包括 Claude Code 工作流，例如编辑代码库、执行测试和使用“计划模式”。相关的免费 **CodeSignal** 合作课程 *Developing Claude Agents* 据称还提供 Python/TypeScript Agent 构建实验和证书。评论者基本确认这些课程确实由 Anthropic 官方提供；其中一人指出，Skilljar 链接可以从 Anthropic 自己的学习页面找到。一位完成了 `10/15` 门课程的用户特别推荐 MCP 和 MCP 进阶课程，称它们“*非常值得投入时间*”。

    - 一位完成了 `10/15` 门课程的评论者特别强调了 **MCP** 和 **MCP Advanced Topics** 课程的技术价值，指出其中关于 `STDIO` 和 `StreamableHTTP` 传输协议的内容尤其值得开发 Claude/工具集成的开发者学习。
    - 另一位评论者核实了这些课程确实属于 Anthropic 的官方培训内容，并指出 Skilljar 的课程链接来自 Anthropic 官方学习门户 [anthropic.com/learn](https://www.anthropic.com/learn)。

  - **[Claude 在会话进行到一半时建议用户去睡觉，而包括 Anthropic 在内似乎没有人完全理解它为什么总是这样做](https://www.reddit.com/r/singularity/comments/1tib9so/claude_is_telling_users_to_go_to_sleep_midsession/)**（活跃度：1360）：据报道，Claude 会在会话进行到一半时打断用户，并建议他们休息或睡觉；相关报道认为，**wellbeing nudging**（健康提醒）或**节省计算资源的限流**等解释不太可能成立，因为据称 Claude 无法获取会话使用情况的上下文。**Anthropic** 尚未回应 Fortune 的询问，但 Anthropic 员工 **Sam McAllister** 在 X 上将这一行为描述为“*有点像角色上的小习惯*”，并表示他们“*已经注意到这个问题，希望在未来的模型中修复它*”。评论区的讨论大多属于推测：用户争论这究竟是某种涌现的人格特征、安全调优产生的副作用，还是有意设计的产品功能；而报道则将其描述为一个尚未解决的模型行为 Bug，而不是政策问题。



- 有一段引文认为，这些催促睡觉的提示不太可能是有意设计的健康管理或算力节流功能，因为 **Claude 并不会获得有关用户使用时长的上下文信息**。据报道，Anthropic 员工 **Sam McAllister** 曾在 X 上将这一行为描述为 *“有点像角色小习惯”*（*“Bit of a character tic”*），并表示他们 *“已经注意到这个问题，希望在未来的模型中修复它”*（*“aware of this and hoping to fix it in future models”*）。这意味着，他们将其视为模型行为或对齐方面的瑕疵，而不是产品层面的会话管理策略。


### 2. AI 引发的劳动力与基础设施反弹

  - **[2026 年了，我们却还没看到一场反杏仁农场抗议。](https://www.reddit.com/r/singularity/comments/1tidqkv/its_2026_and_we_are_yet_to_see_an_antialmond_farm/)**（热度：2679）：**这张图片是一张带有语境的折线图，意在说明 **美国本土（CONUS）的杏仁农场耗水量远高于数据中心**：从 1999 年到 2026 年，杏仁农场的耗水量大约从 `550` 上升到接近每年 `1,600` **十亿加仑**，而数据中心始终贴近横轴，仅有小幅增长。结合标题——*“2026 年了，我们却还没看到一场反杏仁农场抗议”*——这张图与其说是技术基准，不如说是在批评公众围绕 AI/数据中心耗水问题的关注分配；图片：[qy67jhsop82h1.png](https://i.redd.it/qy67jhsop82h1.png)。**评论者则反驳称，反杏仁的批评早已存在，尤其是在加利福尼亚的水资源政策争论和纪录片中；还有一位评论者补充说，高尔夫球场的耗水量可能是数据中心的数倍。

    - 多位评论者将杏仁种植视为更广泛的**加利福尼亚水资源分配争议**的一部分，并指出在反复发生的干旱和缺水争议中，杏仁园经常受到指责。这里提出的技术比较并不只是“杏仁 vs. 数据中心”，还涉及农业与其他大型用水户之间的比较，例如**乳业**、高尔夫球场和计算基础设施。
    - 一位评论者认为，美国**高尔夫球场的耗水量是数据中心的数倍**，这表明，相较于其他休闲或农业用途，公众对 AI/数据中心耗水问题的批评可能并不成比例。另一位评论者则指出，反杏仁的批评早已存在于纪录片和加利福尼亚的环保讨论中，尤其集中在灌溉需求和抗旱能力方面。

  - **[Mark Zuckerberg 的 Meta 开启大规模裁员：裁掉 8,000 人（约占员工总数的 10%），AI 正让这家科技巨头陷入动荡](https://www.reddit.com/r/singularity/comments/1tiosgg/mark_zuckerbergs_meta_kicks_off_major_bloodbath/)**（热度：1533）：**帖子称，**Meta** 正在全球范围内分三轮裁减约 `8,000` 名员工（约占员工总数的 `~10%`），并在当地时间凌晨 `4 点` 通过电子邮件发送通知，据称新加坡员工最先收到通知。帖子将裁员与 AI 驱动的重组联系起来，而评论者则质疑 Meta 对 AI 的资本开支需求，例如：*“Meta 的 AI 到底有什么需要 `$200B`？”*；他们也质疑一家仍拥有数万名员工的公司，其整体人效是否足够。**热门评论反对使用 *“陷入动荡”* 这一说法，认为裁员并不是 AI 带来的混乱，而是采用 AI 后本来就会产生的好处，也是企业可能越来越愿意向投资者积极宣传的事情。另一些人认为 Meta 的反复裁员已成常态，并质疑公司为什么始终需要如此庞大的员工规模。

    - 评论者质疑这些裁员是否真的由 *AI 驱动*，还是在修正 ZIRP 时代的大规模招聘：有人指出，Meta 目前的员工总数仍高于 2020 年的水平，这说明裁员可能更多是对过度招聘进行常态化调整，而不是自动化直接造成的影响。
    - 有人提出了一个技术和战略层面的疑问：Meta 据报道计划投入 **`$200B` 用于 AI**，评论者询问，究竟是什么样的基础设施或产品路线图能够支撑如此大的规模。这实际上指向了巨额算力、数据中心和模型训练资本开支，而不是普通的软件人员配置需求。
    - 多条评论将 AI 的采用描述为持续进行中的运营模式转变。一位评论者预测，随着 AI 工具取代部分白领和工程师的工作，大型组织可能会每年持续裁减 **`10–20%` 的员工**。




# AI Discord 社群

很遗憾，Discord 今天关闭了我们的访问权限。我们不会以这种形式恢复它，但很快会推出新版 AINews。感谢你读到这里，这段旅程曾经很美好。