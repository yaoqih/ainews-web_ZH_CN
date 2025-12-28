---
companies:
- meta-ai-fair
- scale-ai
- lamini
- amd
- openai
- gemini
- google
- anthropic
date: '2025-06-11T05:44:39.731046Z'
description: '以下是该文本的中文翻译：


  **Meta** 聘请了 **Scale AI 的 Alexandr Wang** 来领导其新成立的“超级智能”（Superintelligence）部门，此前
  Meta 出资 **150 亿美元**收购了 Scale 49% 的股份。**Lamini 的 Sharon Zhou** 加入 **AMD** 担任 AI 副总裁，向苏姿丰（Lisa
  Su）汇报；而 **Instacart 的 Fidji Simo** 则加入 **OpenAI** 担任应用业务首席执行官，向 **Sam Altman（Sama）**
  汇报。


  **Meta** 为顶尖研究人员提供超过 **1000 万美元的年薪方案**，并成功从 **Gemini** 团队招募了 **Jack Rae**。**OpenAI**
  向 **ChatGPT Pro** 用户和 API 发布了 **o3-pro** 模型，其性能超越了 **o3**，并在 **Extended NYT Connections**
  和 **SnakeBench** 等测试中刷新了基准。尽管 **o3-pro** 的速度慢于 **o1-pro**，但它在推理和复杂问题解决方面表现卓越。


  **OpenAI** 将 **o3** 的价格下调了 **80%**，使其比 **GPT-4o** 还要便宜，从而向 **谷歌** 和 **Anthropic**
  等竞争对手施加了降价压力。此外，用户现在可以使用**直接偏好优化（DPO）**对 **GPT-4.1** 系列模型进行微调，以处理主观性任务。'
id: MjAyNS0w
models:
- o3-pro
- o3
- o1-pro
- gpt-4o
- gpt-4.1
- gpt-4.1-mini
- gpt-4.1-nano
people:
- alexandr_wang
- sharon_zhou
- fidji_simo
- sama
- jack_rae
- markchen90
- kevinweil
- gdb
- gregkamradt
- lechmazur
- wesrothmoney
- paul_cal
- imjaredz
- cto_junior
- johnowhitaker
- polynoamial
- scaling01
title: 高管变动第二期：Scale-Meta、Lamini-AMD 以及 Instacart-OpenAI
topics:
- model-release
- benchmarking
- reasoning
- fine-tuning
- pricing
- model-performance
- direct-preference-optimization
- complex-problem-solving
---

**优秀的管理层（Execs）就是你所需要的一切。**

> 2025年6月10日至6月11日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord（218 个频道，6238 条消息）。预计节省阅读时间（按 200wpm 计算）：502 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以精美的 vibe coded 方式展示所有往期内容。访问 https://news.smol.ai/ 查看完整的新闻详情，并在 @smol_ai 上向我们提供反馈！

随着夏季大片季的到来，AI 领域似乎也进入了“续集季”——紧随 [推理价格战 2](https://news.smol.ai/issues/25-06-10-o3-cut) 之后，我们现在看到了去年被称为“[高管雇佣（Execuhires）](https://news.smol.ai/issues/24-08-02-ainews-execuhires-tempting-the-wrath-of-khan)”的第二波浪潮：

- **Scale AI 的 Alexandr Wang 被聘请领导 Meta 新成立的“Superintelligence”部门**，作为 [以 150 亿美元收购 Scale 49% 股份](https://www.theverge.com/news/684322/meta-scale-ai-15-billion-investment-zuckerberg) 交易的一部分。
- **Lamini 的 [Sharon Zhou 被聘为 AMD 在 Lisa Su 领导下的 AI 副总裁](https://x.com/realSharonZhou/status/1932817096510931380)**，同时还有“几位厉害且可爱”的高管加入，而 Lamini 的未来去向则完全悬而未决。
- （较早的消息）[Instacart 的 CEO Fidji Simo 被聘为 OpenAI 在 Sama 领导下的 Apps 首席执行官](https://finance.yahoo.com/news/fishing-family-big-tech-french-030253230.html?guccounter=1)。

这一切都发生在 [Meta 向顶尖研究人员提供每年超过 1000 万美元薪酬包](https://x.com/deedydas/status/1932828204575961477) 的背景下，Meta 已成功从 [Gemini 挖走了 Jack Rae](https://x.com/shiringhaffary/status/1932852606851789278)。

---

# AI Twitter 综述

**OpenAI 模型更新与定价**

- **o3-pro 发布与性能**：**OpenAI** 发布了关于 **o3-pro** 的重大公告，确认其已向所有 **ChatGPT Pro 用户**及 **API** 开放，并在多项评估中被证实是比 **o3** “显著更好”的模型。[@OpenAI](https://twitter.com/OpenAI/status/1932586531560304960) 和 [@markchen90](https://twitter.com/markchen90/status/1932570548740964438) 重点介绍了此次发布，[@kevinweil](https://twitter.com/kevinweil/status/1932565467736027597) 指出 **ChatGPT Plus 用户的 o3 速率限制翻倍**。用户和评估者迅速测试了其能力，[@gdb](https://twitter.com/gdb/status/1932561536268329463) 表示 **o3-pro** “比 o3 强得多”。它在 **Extended NYT Connections** 基准测试中创下了新纪录，超越了 **o1-pro**（从 82.5 提升至 87.3），并根据 [@GregKamradt](https://twitter.com/GregKamradt/status/1932898036466004317) 和 [@LechMazur](https://twitter.com/LechMazur/status/1932656485341032719) 的数据，成为了 **SnakeBench 排名第一的模型**。[@WesRothMoney](https://twitter.com/WesRothMoney/status/1932679839682867296) 报告称它**一次性解决了汉诺塔 10 盘问题**。虽然 [@paul_cal](https://twitter.com/paul_cal/status/1932745565021868063) 注意到 **o3-pro** 可能比 **o1-pro 慢 3 倍**，但其推理能力受到了赞赏。[@imjaredz](https://twitter.com/imjaredz/status/1932657322204987718) 声称在非代码任务中它“感觉**遥遥领先于 Claude Opus 4**”，[@cto_junior](https://twitter.com/cto_junior/status/1932989802657497206) 发现它能正确解决**复杂的并行计算问题**。[@johnowhitaker](https://twitter.com/johnowhitaker/status/1932821323979632783) 分享了一个演示，其中 **o3-pro** 正确回答了一个 **o3** 失败的复杂闲置问题。来自 **OpenAI** 的 [@polynoamial](https://twitter.com/polynoamial/status/1932600979113005300) 表示，他们正在招聘人员，以通过 **o3** 等模型推动“智能前沿”。
- **价格与可访问性**：据 [@scaling01](https://twitter.com/scaling01/status/1932596796347252937) 报道，**OpenAI** 实施了“大幅降价”，使 **o3** 便宜了 **80%**，且明显**比 GPT-4o 更便宜**。[@apples_jimmy](https://twitter.com/scaling01/status/1932549807304020227) 指出它比 **4o 便宜 20%**。根据 [@scaling01](https://twitter.com/scaling01/status/1932566284270538966) 的说法，这次降价被视为“可能实际上在迫使 **Google 和 Anthropic 降低价格**”的举动。价格下降使得 **o3** 成为一个可行的日常主力模型，这体现在它被集成到了 **Cursor** 中 [@kevinweil](https://twitter.com/kevinweil/status/1932559521588617415)。
- **微调与模型一致性**：**OpenAI Devs** 宣布，用户现在可以使用**直接偏好优化 (DPO)** 来微调 **GPT-4.1 系列**（4.1, 4.1-mini, 4.1-nano），这对于注重语气、风格和创造力的主观任务非常理想 [@OpenAIDevs](https://twitter.com/OpenAIDevs/status/1932858051876565475)。尽管最初有人猜测降价后模型可能进行了蒸馏或更改，但 [@GregKamradt](https://twitter.com/jeremyphoward/status/1932869428083110007) 和 [@scaling01](https://twitter.com/scaling01/status/1932839048273670563) 通过重新测试确认 **o3** 仍然是**同一个模型**，并未经过蒸馏。
- **模型演进与评估**：[@BorisMPower](https://twitter.com/BorisMPower/status/1932556016455201145) 强调了“自最初的 **o1-preview** 以来，推理模型性能提升的惊人轨迹”，并指出 **60% 以上的胜率**具有重大意义。[@HamelHusain](https://twitter.com/HamelHusain/status/1932642264163180844) 就评估时机与 **OpenAI 降价**的关系发表了评论。

**其他模型发布与高级 AI 研究**

- **Mistral AI 动态**：**Mistral AI** 正式发布了 **Magistral**，这是他们的第一个推理模型，旨在实现特定领域、透明且多语言的推理 [@MistralAI](https://twitter.com/algo_diver/status/1932560648099278930)。**Magistral 4-bit DWQ** 现在已在 Hugging Face 上提供，可配合 **mlx-lm** 或 **LM Studio** 使用 [@awnihannun](https://twitter.com/awnihannun/status/1932547785162961291)。[@Teknium1](https://twitter.com/Teknium1/status/1932580993132790232) 称赞 **Mistral 的论文**是“自 **DeepSeek R1** 以来关于推理 RL 最好的实践论文”。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1932572636397113579) 指出，新的 **Mistral** 模型拥有 **24B 参数**，基于 **Mistral Small 3.1**，支持**多语言**，并具有 **128K 上下文长度（有效长度 40K）**，采用 **Apache 2.0 许可证**。

- **Meta AI 的 V-JEPA 2 与世界模型 (World Models)**：**Meta AI** 发布了 **V-JEPA 2**，这是一个拥有 **12 亿参数**的视频训练模型，旨在通过在陌生环境中实现**机器人的零样本规划 (zero-shot planning)** 来推进物理 AI (physical AI)。他们还推出了**三个新的基准测试 (benchmarks)**，用于评估视频中的物理世界推理能力 [@AIatMeta](https://twitter.com/AIatMeta/status/1932808881627148450) 和 [@AIatMeta](https://twitter.com/AIatMeta/status/1932923002276229390)。[@ylecun](https://twitter.com/ylecun/status/1932845440547840234) 强调了 **V-JEPA 2** 的重要性。
- **Gemini 与 Google 的 AI 能力**：**Gemini 2.5 Pro (06-05)** 正在迅速攀升各大公开排行榜，值得注意的是，根据 [@_philschmid](https://twitter.com/_philschmid/status/1932723220379049999) 的说法，它已成为 **Live Fiction 上 192K tokens 级别排名第一的模型**，是 **IDP 中最强的文档处理 (Document Processing) 模型**，以及 **Aider 上性价比最高的模型**。据报道，它还**解决了 JEE Advanced 2025 数学部分的所有问题** [@dilipkay](https://twitter.com/dilipkay/status/1932754214469402630)。**Google Veo 3** 在视频的一致性角色和情绪生成方面展示了令人印象深刻的能力 [@demishassabis](https://twitter.com/demishassabis/status/1932608957945950407) 和 [@DrMachakil](https://twitter.com/demishassabis/status/1932856733397102914)。**Google** 还发布了适用于桌面和 IoT 的 **Gemma 3n**，由其全新的 **LiteRT-LM 库**驱动 [@osanseviero](https://twitter.com/demishassabis/status/1932607299178148184)。
- **新模型与研究技术**：
    - **Higgsfield** 推出了 **Higgsfield Speak**，使图像中的面孔（汽车、僵尸，甚至咖啡）能够说话 [@_akhaliq](https://twitter.com/_akhaliq/status/1932545372817154530)。他们还集成了 **Flux.1 Kontext** 以增强内容创作 [@_akhaliq](https://twitter.com/home/status/1932903530173747261)。
    - **Cartesia AI** 推出了 **Ink-Whisper**，这是一个全新的流式**语音转文本 (STT) 模型**系列，旨在为语音 Agent 提供快速且经济高效的服务 [@krandiash](https://twitter.com/krandiash/status/1932601554298941812)。
    - **Sakana AI Labs** 推出了 **Text-to-LoRA**，这是一个超网络 (hypernetwork)，可以根据任务的文本描述生成**特定任务的 LLM 适配器 (LoRAs)**，为基础模型的专业化显著降低了计算和技术门槛 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1932972420522230214)。
    - 研究表明，**混合模型 (hybrid models)** 可以用更少的注意力层 (attention layers) 维持推理性能，为长推理链 (long reasoning traces) 提供效率优势 [@_albertgu](https://twitter.com/_albertgu/status/1932844922241233019)。
    - 一项名为 **IneqMath** 的新研究揭示，**LLM 即使在给出正确答案时，也往往难以进行严谨的数学证明** [@HuggingPapers](https://twitter.com/_akhaliq/status/1932894338616574091)。
    - **FutureHouseSF** 正在研发 **ether0**，这是一个 **24B 模型**，可以用英语进行推理并以分子形式进行响应 [@cgeorgiaw](https://twitter.com/_lewtun/status/1932875317678317817)。
    - **Yandex** 发布了 **Yambda**，这是一个包含**近 50 亿条匿名用户轨迹交互**的大规模公开数据集，用于推荐系统 [@_akhaliq](https://twitter.com/_akhaliq/status/1932872791768117483)。
    - [@nickhjiang](https://twitter.com/TimDarcet/status/1932707025718247935) 的研究发现，**Vision transformers** 存在**高范数离群值 (high-norm outliers)**，这会损害性能并扭曲注意力。
- **模型局限性与未来方向**：**The Turing Post** 总结了来自 **Apple 的《思维的错觉》(Illusion of Thinking)** 以及 **FAIR/Google DeepMind/Cornell/NVIDIA 的《语言模型记忆了多少？》(How much do language models memorize?)** 论文的核心见解，强调了有限的架构在被推向其基础能力极限时，即使看起来很流畅，也会出现简化、猜测或宕机的情况 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1932912650444550470)。这表明“扩展 (scaling) 在数字世界可能正面临瓶颈”，但在“生物世界才刚刚开始” [@ysu_nlp](https://twitter.com/wightmanr/status/1932858386368090407)。

**AI Agent 与开发框架**

- **DSPy 的采用与哲学**：**DSPy** 框架获得了显著的关注，[@lateinteraction](https://twitter.com/lateinteraction/status/1932551576100667416) 强调了其“远见”，即“提示词是编译后的输出”，而非源代码。这一哲学被认为正成为 AI 工程的核心。[@kmad](https://twitter.com/lateinteraction/status/1932633959609102596) 和 [@MaximeRivest](https://twitter.com/lateinteraction/status/1932810387285815395) 指出 **DSPy** “开始被许多人理解并产生共鸣”。[@vineettiruvadi](https://twitter.com/lateinteraction/status/1932810369262887015) 成功使用 **DSPyOSS** 在 **20 分钟**内创建了一个合成临床笔记生成器。
- **Model Context Protocol (MCP) 与 Agent 生态系统**：**Hugging Face (HF) MCP server** 已经发布，允许 Agent 在 HF 生态系统中查找模型、数据集、论文或应用 [@julien_c](https://twitter.com/clefourrier/status/1932690632394293539) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1932823573762355562)。这促进了“**MCP servers** 开源集合”的形成 [@abidlabs](https://twitter.com/ClementDelangue/status/1932885527000461686)。此外，还重点展示了 **GPT Researcher** 利用 **LangChain 的 MCP 适配器**进行智能工具选择的集成案例 [@LangChainAI](https://twitter.com/Hacubu/status/1932677682556187103)。
- **新的 Agent 框架与工具**：
    - **Databricks** 推出了 **Agent Bricks**，这是一种构建自动优化 Agent 的新方法，它采用了独特的“声明式和组合式”视角，允许通过自然语言反馈来引导 Agent [@matei_zaharia](https://twitter.com/lateinteraction/status/1932849147973153017)。
    - **LangChain AI** 发布了 **langchain-google-vertexai** 的更新，包括 **快 500 倍的预测客户端缓存** [@LangChainAI](https://twitter.com/LangChainAI/status/1932848293165371448)。他们还展示了 **Harvey AI** 如何使用 **LangSmith 评估**和“律师在环”（lawyer-in-the-loop）方法论来构建法律 AI Agent [@LangChainAI](https://twitter.com/LangChainAI/status/1932858287265099900)。**LangGraph** 被誉为构建 **AI 对冲基金团队**以及实现如 Cisco 旗下 Outshift 开发的 **JARVIS** 等多 Agent 系统的“重大突破” [@virattt](https://twitter.com/Hacubu/status/1932885769305403548) 和 [@LangChainAI](https://twitter.com/hwchase17/status/1932872002978902374)。
    - **LlamaIndex** 宣布与 **CleanlabAI** 集成，用于构建 AI 知识助手和生产级 Agent，并正在开发针对新邮件到达后更新数据等用例的增量工作流 [@llama_index](https://twitter.com/jerryjliu0/status/1932838464233615814)。**LlamaExtract** 这一 Agent 化文档提取服务也已推出，为提取的数据提供精确的推理和引用 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1932852719292985359)。
    - **Hugging Face** 宣布推出 **AISheets**，允许数千个 AI 模型与电子表格交互，用于构建、分析和转换数据 [@Thom_Wolf](https://twitter.com/ClementDelangue/status/1932573508703232368)。
    - **Smolagents** 在 **GitHub 上突破了 20,000 星**，这是该 Agent 库的一个里程碑 [@AymericRoucher](https://twitter.com/AymericRoucher/status/1932836600695722326)。
    - **Fire Enrich** 作为一个使用 AI Agent 进行数据增强的 **开源 Clay 替代方案**，现已开源 [@firecrawl_dev](https://twitter.com/hwchase17/status/1932884877273411764)。
- **Agent 设计与用例**：在 **AI Engineer World’s Fair** 上的讨论涵盖了“环境” Agent 的趋势以及 **MCP** 的作用 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1932550904453976567)。[@dzhng](https://twitter.com/dzhng/status/1932861054146785547) 强调，在开发 Agent 工具时，界面设计必须**易于 AI 理解**，这通常不仅仅是暴露底层 API。**Codex**（CLI 版本）被证明在**调试堆栈跟踪（stacktraces）**方面能带来“惊人的生产力提升” [@cto_junior](https://twitter.com/cto_junior/status/1932802313665851901)。
- **多 Agent 架构评估**：**LangChain AI** 发布了关于跨多个 Agent 编排的初步基准测试，包括对其监督者（supervisor）方法的改进 [@LangChainAI](https://twitter.com/LangChainAI/status/1932825652312600810)。

**AI 业务、基础设施与部署**

- **战略伙伴关系与投资**：据报道，**OpenAI** 已接触 **Google** 寻求新的云计算协议，以获取更多 **compute** [@scaling01](https://twitter.com/scaling01/status/1932716057631846860)。**Meta** 对 **Scale AI** 的投资（**Alex Wang** 参与其中的“超级智能”实验室）引发了关于 Meta AI 战略及其对 **RL** 后训练影响的讨论 [@steph_palazzolo](https://twitter.com/steph_palazzolo/status/1932588243897270454)。**xAI** 与 **Polymarket** 达成合作，将市场预测与 **X data** 及 **Grok's analysis** 相结合，打造“硬核真相引擎” [@xai](https://twitter.com/Yuhu_ai_/status/1932753086885540061)。**NVIDIA** 与 **Hugging Face** 宣布了一项新合作，旨在通过 **Training Cluster** 将 AI 研究人员与 **GPU** 集群连接起来 [@jeffboudier](https://twitter.com/_lewtun/status/1932771189396492494)。
- **AI 基础设施与算力**：**Mistral AI** 推出了 **Mistral Compute**，这被描述为“欧洲前所未有的 AI 基础设施举措”，也是获取算力的战略行动 [@MistralAI](https://twitter.com/qtnx_/status/1932799532070547810)。**Modular** 和 **TensorWave** 通过合作伙伴关系提供免费算力 [@clattner_llvm](https://twitter.com/clattner_llvm/status/1932614831364006363)。**TogetherCompute's API** 因拥有**最快的 DeepSeek v3 端点**（比次优方案快 2 倍）而受到关注，并推出了针对合成数据生成、基准测试和文档提取等高吞吐量场景的新 **Batch API**，价格比其交互式 **API** 低 50% [@vipulved](https://twitter.com/vipulved/status/1932601075754020876)。
- **优化与效率**：据报告，通过使用 **read-through caching** 等简单技巧，日常维护任务实现了显著提速（例如，遍历 **S3/GCS prefixes** 的速度提升了 **30 倍**）[@turbopuffer](https://twitter.com/turbopuffer/status/1932916345571848610)。**UnslothAI** 宣布他们提供 **2 倍速的奖励模型服务（reward model serving）和序列分类推理**，并作为全球前 100 家最具影响力和增长最快的基础设施公司之一登上 **Nasdaq tower** [@danielhanchen](https://twitter.com/danielhanchen/status/1932801493649793469)。在 **AMD** 上使用 **Docker run --gpus** 的便捷性也被认为是提高可访问性的关键 [@AnushElangovan](https://twitter.com/dylan522p/status/1932829981316452404)。
- **产品发布与公司新闻**：
    - **Mechanize** 宣布成立，其明确目标是通过构建虚拟工作环境、基准测试和训练数据来**“实现所有工作的自动化”** [@tamaybes](https://twitter.com/tamaybes/status/1932841955542904919)。
    - **Runway** 暗示了“非常令人兴奋的更新和新产品”，将为平台带来“全新的体验”，让创作变得“自然且简单” [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1932600586123227219)。其 **upscaling（超分辨率）功能现已在 API 中上线** [@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1932835921071878384)。
    - [**You.com**](http://you.com/) 推出了“Projects”功能，用于将研究整理到文件夹中，实现信息的上下文关联和结构化 [@youdotcom](https://twitter.com/RichardSocher/status/1932843347632402905)。
    - **Dia** 宣布推出，这是一款“深度理解你”的全新浏览器，旨在实现个性化的网页交互 [@hursh](https://twitter.com/mathemagic1an/status/1932864995668508945)。
    - **Perplexity AI** 在 **GTC Paris** 上被 **Jensen Huang** 重点介绍 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932968936938537223)。
    - **Databricks** 发布了**免费版**并开放了培训材料，以帮助开发者学习数据和 AI [@matei_zaharia](https://twitter.com/lateinteraction/status/1932834408727543823)。
    - 经过数月的社区协作，**MLflow 3.0** 正式发布 [@MLflow](https://twitter.com/lateinteraction/status/1932871442347274500)。
    - **Wayve AI** 的“惊人发展轨迹”获得了认可 [@nathanbenaich](https://twitter.com/NandoDF/status/1932552110005989462)。
- **训练与部署最佳实践**：与 [**Astronomer.io**](http://astronomer.io/) 合作推出了新的短课“Orchestrating Workflows for GenAI Applications”，教开发者如何使用 **Apache Airflow 3.0** 将 **GenAI** 原型转化为生产级工作流 [@AndrewYNg](https://twitter.com/AndrewYNg/status/1932822251273093247)。**evaluations（评估）**对于构建可靠的 **LLM** 驱动应用的重要性被反复强调，[@HamelHusain](https://twitter.com/HamelHusain/status/1932657675294421061) 开设了关于**“面向工程师和技术 PM 的以应用为中心的 AI 评估”**的课程。**BorisMPower** 建议初创公司多做功能原型，评估失败原因，并专注于用户喜爱的功能，待新模型发布后再重新审视其他功能 [@BorisMPower](https://twitter.com/BorisMPower/status/1932813365199712277)。

**社会与地缘政治 AI 影响**

- **AI 对工作和社会的影响**：**Sam Altman** 的言论，如“我们不知道能超越人类水平智能多远，但我们即将揭晓”以及“智能廉价到无需计量已近在咫尺”，突显了向 **AGI** 迈进的飞速进程及其深刻改变社会的潜力 [@scaling01](https://twitter.com/scaling01/status/1932550566036804087) 和 [@scaling01](https://twitter.com/scaling01/status/1932551669134377357)。**Mechanize** 明确提出的“自动化所有工作”的目标引发了关于岗位流失的讨论，一些人称其做法“缺乏同理心” [@tamaybes](https://twitter.com/tamaybes/status/1932841955542904919)。**Yoshua Bengio** 讨论了“前沿模型中 AI 能力进步速度的加快以及欺骗性和自我保护行为的出现”，这启发了他的 **LawZero** 倡议 [@Yoshua_Bengio](https://twitter.com/Yoshua_Bengio/status/1932859177283801470)。
- **地缘政治与 AI 竞赛**：英国的 AI 和生物科学产业受到了 **ChatGPT** 的批判性评估，强调了资金边缘化（800 万英镑 vs 美国公司的 >1 亿美元）、受美国强制执行的 **garden leave**（竞业避风期）导致的人才流动问题、英国创新成果的外资所有权以及缺乏产业支柱 [@NandoDF](https://twitter.com/NandoDF/status/1932549812785754524)。有关 **Mistral** 受到**出口限制**以及**中国**对 **ASML** 光刻机需求的提及，凸显了持续的**技术战争**和算力获取问题 [@dylan522p](https://twitter.com/dylan522p/status/1932563462963507589) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1932955304188076081)。
- **伦理与社会挑战**：有观点认为 **AI 卡路里计数应用**在概念上存在缺陷，暗示它们主要作为一种正念的“仪式”，而非准确的技术工具 [@random_walker](https://twitter.com/random_walker/status/1932763118498685276)。讨论还涉及了 **GenAI** 可能从非现实生活互动中移除有关个人的可操作信息，导致对“信任环”的依赖并减少“圈外人”的机会 [@francoisfleuret](https://twitter.com/francoisfleuret/status/1932683908715282670)。当前的信息环境被描述为“最糟糕的” [@tszzl](https://twitter.com/aidan_mclau/status/1932833046572773797)。
- **加沙/巴勒斯坦冲突**：单个账号 [@SerranoAcademy](https://twitter.com/SerranoAcademy/status/1932664601373446633) 的大量推文放大了与加沙冲突相关的内容，分享了来自 [@GozukaraFurkan](https://twitter.com/SerranoAcademy/status/1932664601373446633) 关于国际忽视的报告、[@Kahlissee](https://twitter.com/SerranoAcademy/status/1932664629987045637) 关于 **Greta Thunberg** 立场的推文、来自 [@ShaykhSulaiman](https://twitter.com/SerranoAcademy/status/1932664650501664809) 的儿童伤亡图像，以及来自 [@JimmyJ4thewin](https://twitter.com/SerranoAcademy/status/1932664208958652630) 要求释放 **Rima Hassan** 的呼吁。此外还分享了布鲁塞尔和曼彻斯特支持加沙的抗议活动。
- **哲学与文化观察**：讨论范围广泛，从“学习模型类似于学习编程语言”的观点 [@deepgramscott](https://twitter.com/deepgramscott/status/1932596824126468198)，到出生率下降和后 AGI 社会（推文中未直接提及，但可从 Sam Altman 的总体主题中推断），再到对**湾区**“功能失调社会”和极端贫富差距的观察 [@claud_fuen](https://twitter.com/claud_fuen/status/1932773334959174041)。**AI 在赋能方面的作用**（“我们可以构建任何东西”）也被提及 [@shaneguML](https://twitter.com/shaneguML/status/1932979593520275536)。

**幽默/迷因**

- **引起共鸣的开发者/AI 工程师幽默**：
    - “地堡的备用算力” [@zacharynado](https://twitter.com/zacharynado/status/1932554928716660767)
    - “抱歉没回邮件，我刚才又在搞 Reward Engineering 了” [@vikhyatk](https://twitter.com/vikhyatk/status/1932675987235614835)
    - “当你发现自己在打的每一行末尾都习惯性加分号时，你就知道自己写代码太久了；比如这条推文；该死；” [@claud_fuen](https://twitter.com/claud_fuen/status/1932779123736195451)
    - “参加了一个电话会议，结果只有我和一打 AI” [@thedanigrant](https://twitter.com/zacharynado/status/1932880607941644581)
    - “每次我打开 Weights and Biases 时” [@vikhyatk](https://twitter.com/vikhyatk/status/1932962492696965626)
    - “今晚晚餐是 Prompt 汤” [@lateinteraction](https://twitter.com/lateinteraction/status/1932997629161685213)
- **AI/科技行业讽刺**：
    - **Perplexity AI** 被一首名为《Defying Google》的歌曲恶搞，这是一首对其搜索能力的颂歌 [@AravSrinivas](https://twitter.com/AravSrinivas/status/1932693461355970872)。
    - “人们都在吐槽 OpenAI 给模型起名很奇怪，但想象一下，如果他们有一个有品位的营销人员，这些东西可能会被命名为 'Gazelle' 或 'Appalachia'” [@fabianstelzer](https://twitter.com/fabianstelzer/status/1932730260350452136)
    - “新策略刚刚发布：筹集数百万美元的 VC 资金，然后去度假两年……这就是 2025 年赢家的样子” [@claud_fuen](https://twitter.com/claud_fuen/status/1932837037024977295)
    - “你们给自己的 AGI 多少小费？18%、20% 还是 22%？” [@johannes_hage](https://twitter.com/johannes_hage/status/1932696834818142581)
    - “说实话，YC 录取了几百个直男却只录取了 1 只巨嘴鸟，这太离谱了” [@andersonbcdefg](https://twitter.com/andersonbcdefg/status/1932888219450028368)
    - “Y2K 回归了。现在的 AI 小孩都在说：'tmol-faq. logi. gisai. cev. sysop. ufai. foom. rpop. flare.' 我问其中一个他们在说什么，他们说 '在发帖到 sl4 之前请先阅读存档。'” [@goodside](https://twitter.com/goodside/status/1932990995638976668)
- **通用科技/流行文化**：
    - “内部已实现 '液态玻璃'。” [@claud_fuen](https://twitter.com/claud_fuen/status/1932769765317050659)
    - “我分不清拿到私人飞行执照是你能做的最酷的事情之一，还是仅仅是一种焚烧 20,000 美元的高级方式” [@typedfemale](https://twitter.com/typedfemale/status/1932860823032246428)
    - “GitHub 上刚刚创建了第 10 亿个仓库，它在各方面都非常完美！” [@film_girl](https://twitter.com/zacharynado/status/1932983386219405754)
    - “你想要猫娘” [@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1932840090155560981)（针对一张展示各种“类型”AI 图片的回应）

---

# AI Reddit 综述

## /r/LocalLlama 综述

### 1. Meta V-JEPA 2 世界模型视频训练与新模型发布

- [**Meta 发布 V-JEPA 2，首个在视频上训练的世界模型**](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6) ([Score: 218, Comments: 39](https://www.reddit.com/r/LocalLLaMA/comments/1l8umf2/meta_releases_vjepa_2_the_first_world_model/)): **Meta FAIR 发布了 V-JEPA 2（参见 [Hugging Face 集合](https://huggingface.co/collections/facebook/v-jepa-2-6841bad8413014e185b497a6)），这是一组基于 ViT 的模型，推进了其用于无监督视频表示学习的联合嵌入预测架构 (Joint Embedding Predictive Architecture, JEPA)。这些模型包括 vjepa2-vitl-fpc64-256、vjepa2-vith-fpc64-256 和 vjepa2-vitg-fpc64-384，使用 Transformer 骨干网络，并作为大规模视频分析和多模态研究的强力基准进行训练。澄清：尽管帖子声称 V-JEPA 2 是首个在视频上训练的世界模型，但评论者指出这并不准确（参见原始 [Meta 博客文章](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)），V-JEPA 2 是 Meta 持续进行的世界模型开发中的一个增量版本，而非该领域的首创。** 热门评论强调 V-JEPA 2 并非首个在视频上训练的世界模型——在此次 Meta 发布之前，工业界和学术界都已经存在此类模型。技术讨论集中在这些模型如何作为竞争性的无监督基准，用于复杂的视频理解任务。
    - 多位评论者澄清标题在事实上是不正确的：V-JEPA 2 既不是首个在视频上训练的世界模型，也不是 Meta 唯一的此类模型。在此之前，其他机构也利用视频数据开发了世界模型。
    - 引用了一篇博客资源 (https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/)，提供了关于原始 V-JEPA 模型架构、预测性联合嵌入方法及其在视频语境下与预测性世界建模相关性的技术细节和背景。
    - 一位评论者指出，该模型能够正确预测视频中的动作，这意味着该模型的架构有效地捕捉了准确世界建模所需的时空依赖关系；这一成功归功于 Yann LeCun 的贡献。
- [**Altman 谈权重开放 🤔🤔**](https://www.reddit.com/r/LocalLLaMA/comments/1l8oe8g/altman_on_open_weight/) ([Score: 147, Comments: 100](https://www.reddit.com/r/LocalLLaMA/comments/1l8oe8g/altman_on_open_weight/)): **Sam Altman 在 X 上宣布，OpenAI 的权重开放 (open-weights) 模型的发布已推迟到今年夏天晚些时候（不是 6 月），理由是出现了一项意想不到且令人印象深刻的研究进展，需要更多时间。目前尚未发布任何技术细节、基准测试或架构信息；仅声称它将“非常非常值得等待”。** 热门评论对反复推迟表示怀疑，并质疑 OpenAI 的发布是否能超越近期如 DeepSeek R2 等开源成果，一些人暗示推迟是因为缺乏实质性的创新。
    - 一位评论者将 Altman 所谓的权重开放发布的热度与近期的高知名度模型（如 DeepSeek R2）进行了对比，含蓄地表示怀疑 Altman 的团队能否交付比已有的开源或半开源 LLM 发布在技术上更令人印象深刻或更具影响力的东西。讨论凸显了人们对于相对于 DeepSeek R2 等模型设定的最先进基准，是否会有任何新的或具竞争力的东西问世持持续怀疑态度。

### 2. AI 图像公司面临的法律与伦理挑战

- [**迪士尼和环球影业起诉 AI 图像公司 Midjourney，指控其未经授权使用《星球大战》、《辛普森一家》等作品**](https://www.reddit.com/r/LocalLLaMA/comments/1l8zssy/disney_and_universal_sue_ai_image_company/) ([Score: 265, Comments: 119](https://www.reddit.com/r/LocalLLaMA/comments/1l8zssy/disney_and_universal_sue_ai_image_company/)): **据报道，迪士尼和环球影业正在起诉 AI 图像生成公司 Midjourney，指控其未经许可使用受版权保护的角色（如《星球大战》和《辛普森一家》中的角色），这加剧了围绕生成式 AI 训练数据的法律斗争。帖子指出，如果迪士尼获胜，可能会开创一个先例，影响其他使用未经授权版权材料训练模型的 AI 公司，并可能重塑 AI 内容生成格局（[新闻参考](https://www.reuters.com/legal/disney-universal-sue-midjourney-over-unauthorized-use-copyrighted-characters-2024-05-30/)）。** 评论中的讨论在技术实质方面较少，但一条评论强调了潜在的国际影响，指出中国模型可能不受美国法律诉讼的影响，这表明在针对美国/西方法律管辖范围之外运行的 AI 实验室进行有效的版权执法方面存在管辖权差距。
    - 一位评论者强调，这场法律纠纷可能会产生全行业的影响，特别是如果迪士尼的诉讼策略旨在寻求具有先例意义的赔偿：如果 Midjourney 破产，针对其他 AI 图像模型公司的类似法律行动可能会接踵而至。训练数据的合法性被凸显出来——*无论结果如何，用于训练的数据都是广泛且持久存在的*，这引发了人们对考虑到数据共享和收集的本质，限制生成式 AI 是否具有可行性的更广泛担忧。
    - 有推测认为 Google 和 Meta 等大型科技公司可能参与或间接支持，因为这场诉讼的结果可能会影响所有生成式 AI 提供商的格局。其观点是，这些“财大气粗”的公司有着切身利益，因为限制性的法律先例可能会影响它们自己在类似广泛数据集上训练的 AI 模型，这可能会引发统一的行业反应或幕后游说。
    - 另一条技术评论建议 AI 从业者保存图像生成模型（如 HuggingFace 上的模型）的本地副本，以防诉讼成功后模型被下架或受到法律限制。有人对 HuggingFace 等平台是否愿意为托管模型进行法律辩护表示怀疑，暗示了由于潜在的版权打击，人们担心模型会迅速从公共访问平台撤出的恐惧。
- [**MNN TaoAvatar：由阿里巴巴 MNN 团队开发的离线运行 3D 化身 Android 应用**](https://v.redd.it/65vyq2fhca6f1) ([Score: 105, Comments: 26](https://www.reddit.com/r/LocalLLaMA/comments/1l8qh2a/mnn_taoavatar_run_3d_avatar_offline_android_app/)): **阿里巴巴 MNN 团队发布了开源的 MNN TaoAvatar ([GitHub](https://github.com/alibaba/MNN/blob/master/apps/Android/Mnn3dAvatar/README.md#version-001))，这是一款离线 3D 化身 Android 应用，即使在入门级硬件上也能高效运行。该实现利用了轻量级、针对移动端优化的 [MNN 推理引擎](https://github.com/alibaba/MNN)，实现了实时化身渲染和动画。更多技术概述和基准测试请参见 [arXiv 论文](https://arxiv.org/html/2503.17032v1)。** 技术评论者强调，与 Qwen 3B Omni 等其他模型相比，该应用在低端智能手机上的性能表现非常流畅，凸显了 MNN 在移动端推理方面的效率和优化。
    - 用户 abskvrm 提供了实际反馈，指出运行在 MNN 框架上的 TaoAvatar 应用即使在入门级智能手机上也能流畅运行，这意味着与 Qwen 3B Omni 等更重的模型相比，它针对低端硬件进行了强大的优化，后者在他们的设备上运行速度明显较慢。这表明了其出色的端侧效率和用于实时化身渲染的轻量级架构。
    - Juude89 链接的资源（GitHub 主页和 arXiv 论文）提供了深入的技术细节，包括实现细节、版本控制、架构和性能声明，对于寻求 Android 应用运行机制和底层 MNN 优化细节的技术读者非常有用。

### 3. 本地 LLM 推理栈迁移经验

- [**我终于摆脱了 Ollama！**](https://www.reddit.com/r/LocalLLaMA/comments/1l8pem0/i_finally_got_rid_of_ollama/) ([Score: 417, Comments: 212](https://www.reddit.com/r/LocalLLaMA/comments/1l8pem0/i_finally_got_rid_of_ollama/)): **OP 将其推理后端从 Ollama 迁移到了一个包含 llama.cpp 或 ik_llama.cpp 进行推理、llama-swap 进行动态模型加载/卸载（通过 config.yaml 集中配置）以及 Open WebUI 作为前端的技术栈。这实现了自定义模型组织，并进一步摆脱了 Ollama 典型的硬编码模型路径/名称的限制。他们强调了诸如灵活的模型管理（可以轻松从 HuggingFace 使用 wget）、与其他推理引擎的互操作性以及每个工作区的系统提示词自定义等优点，同时归功于 llama.cpp 和 Open WebUI 等核心开源工具。** 评论中引发了技术辩论：一位用户更倾向于使用 llama-server，因为它有潜力在服务端集中控制采样参数（而不是像许多 UI 那样由每个客户端覆盖），并建议改进以提高终端用户的易用性。另一位用户则表示担心，认为 OP 的方法牺牲了 Ollama 具备的便利性（如简单的模型配置、远程下载/启动），需要更多的手动设置并降低了操作的简易性。
    - 一位用户指出了 `llama-server` 在采样参数方面的技术限制：当前的 GUI 客户端会覆盖服务端采样设置，导致每个模型都需要手动调整。该用户希望服务端能够强制执行自身的采样参数，而不受客户端输入的影响，类似于通过 Python 调用 llama-server 时的行为（默认由服务端控制 jinja 模板、采样器和系统提示词等）。这将为不熟悉采样配置的用户简化部署，使其更具 "ChatGPT 风格"。
    - 另一位评论者概述了 Ollama 提供而 `llama-server` 不具备的具体技术便利：自动获取模型、简化配置（无需手动定义模型）以及支持远程下载/启动模型。这意味着在这些方面，切换到 `llama-server` 相比 Ollama 会引入显著的开销。
    - 有人指出 `llama-server` 包含内置服务器，这可能使第三方 UI（如 Open WebUI）变得不再必要，对于追求直接、极简界面部署的用户来说，这可能会简化用户体验。

## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo

### 1. 迪士尼和环球影业起诉 AI 图像生成器

- [**迪士尼对 AI 公司 Midjourney 发起首个重大诉讼，称该图像生成器是“无底的剽窃深渊”**](https://www.reuters.com/business/media-telecom/disney-universal-sue-image-creator-midjourney-copyright-infringement-2025-06-11/) ([Score: 406, Comments: 293](https://www.reddit.com/r/singularity/comments/1l8xpes/disney_launches_first_major_lawsuit_against_ai/)): **迪士尼已对 AI 图像生成器 Midjourney 提起重大诉讼，指控其广泛侵犯版权，并将该平台描述为“无底的剽窃深渊”。据报道，核心论点涉及在训练数据中使用受版权保护图像的许可问题，以及生成式 AI 输出的性质。此案可能取决于对当前版权法的解释，以及在受保护作品上进行训练是否构成侵权。** 评论者质疑迪士尼主张的法律依据，将其与 Photoshop 等工具进行类比，并强调迪士尼历史上对公共领域的依赖；人们对适用于生成式 AI 的版权法的一致性和影响持怀疑态度。
    - 针对法律逻辑引发了一场技术辩论：一些用户质疑迪士尼的行为是否类似于指责工具（Midjourney/图像生成器）而非用户，并类比迪士尼是否会因为用户创作衍生作品而起诉 Adobe Photoshop。这引发了关于软件/工具责任法律先例的更深层次问题。
    - 更广泛的讨论包括对版权法和合理使用（fair use）的分析，评论者指出迪士尼自身在历史上也依赖公共领域作品来构建其 IP 组合，并批评了执法实践中潜在的双重标准和模糊性（例如，迪士尼很少对同人艺术家采取行动，但却激进地针对新的生成模型）。
    - 潜在的担忧在于版权法需要更新或改革，特别是为了适应生成式 AI 带来的挑战——具体而言，它如何处理训练数据与生成的输出，以及扩张性的版权执行是否会扼杀创意和技术进步。
- [**迪士尼和环球影业起诉 AI 图像公司 Midjourney，指控其未经授权使用《星球大战》、《辛普森一家》等作品**](https://www.reddit.com/r/StableDiffusion/comments/1l8zmpb/disney_and_universal_sue_ai_image_company/) ([Score: 312, Comments: 274](https://www.reddit.com/r/StableDiffusion/comments/1l8zmpb/disney_and_universal_sue_ai_image_company/)): **迪士尼和环球影业已对 AI 图像生成公司 Midjourney 提起诉讼，指控其在 Midjourney 的训练数据中未经授权使用知识产权（包括《星球大战》和《辛普森一家》）构成侵权。该法律行动针对的是在训练数据集中使用受版权保护的媒体内容，此案可能会创下一个关键先例，影响所有使用类似抓取数据且未获得明确许可的 AI 实验室。潜在的法律辩论中心在于，在受版权保护的作品上训练生成模型是否构成合理使用或违规。** 评论者指出，像 Midjourney 和 NovelAI 这样规模较小的 AI 实验室是更容易受到攻击的法律目标，而大型实体（如 OpenAI、Microsoft）或开源项目由于规模、法律复杂性或较低的经济动机而更难被起诉。还有推测认为，诉讼可能会将 AI 开发或托管转移到受美国 IP 法律影响较小的司法管辖区（如中国）。
    - 一位用户强调了战略性的法律风险计算，指出对于迪士尼这样的大型工作室来说，Midjourney 和 NovelAI 是更容易受到版权诉讼攻击的目标，而攻击像 OpenAI（由 Microsoft 支持）这样的大型参与者可能会导致版权持有者的法律地位更加被动。此外，针对开源项目可能并不值得，因为潜在收益有限且法律风险巨大，这表明制片厂是根据资源差距以及达成和解或获胜的可能性来选择被告的。
    - 有人推测 AI 实验室在不同司法管辖区的法律风险。一些评论者提到，中国的公司有可能在不担心美国版权诉讼的情况下开发模型，这意味着国际执法差距可能会鼓励商业开源或非美国 AI 模型的发展。
    - 评论者质疑为什么某些公司（如 Grok 等）没有面临类似的诉讼，暗示选择性法律行动背后可能存在政治或战略原因，而非纯粹的技术或法律先例。这间接涉及了 AI 版权领域法律的模糊性和执法选择性。

### 2. 1X Neo 机器人演示与预览

- [**1X 明日即将发布的更新预览**](https://streamable.com/xoyjje) ([Score: 330, Comments: 115](https://www.reddit.com/r/singularity/comments/1l8w1a9/a_sneak_peek_at_an_update_coming_tomorrow_from_1x/))：**1X 正在通过一段视频预览预热即将到来的更新（明天发布），视频可能展示了其人形机器人在户外田野环境中的自主运行。引用的图片 (https://preview.redd.it/0kutog6ckb6f1.jpeg?width=483&format=pjpg&auto=webp&s=9c1d08cbb65b8fb0c0d470a918dda87f3f029ca1) 暗示这一场景此前已有伏笔，表明机器人在户外移动能力和 AI 方面的连续性或新功能。Streamable 链接包含一段标准的视频预览，没有额外的技术元数据；重点在于 1X 实体机器人的更新，而非视频平台本身。** 评论中的技术讨论极少，主要集中在社会影响（“它要来抢你的工作了”）以及机器人在开阔田野中运行的美学，没有深入的技术辩论或错误报告。
    - Dioxbit 的评论对 Veo 3 等先进视频生成模型产生的内容日益难以辨别表示担忧，指出合成素材正达到一个真实性与真伪分类在技术上更具挑战性的水平。这呼应了关于随着模型进步，建立可靠的 AI 生成媒体检测方法必要性的持续辩论。
- [**来自 1X 的 Neo 新镜头**](https://streamable.com/hiv798) ([Score: 264, Comments: 64](https://www.reddit.com/r/singularity/comments/1l8zfo6/new_neo_footage_from_1x/))：**一段名为“Neo Footage from 1X”的新视频已发布并托管在 Streamable 上，可在[此处](https://streamable.com/hiv798)访问，但元数据摘要中未披露关于视频内容（格式、模型、基准测试或功能）的技术细节。该帖子和链接页面仅提供标准的视频播放和分享控制，没有关于镜头背后的底层技术、模型或推理过程的信息，也没有展示 1X 的任何新技术能力。** 评论讨论推测了 1X 作品的艺术或主题意图，暗示了一种刻意的风格化方向，但不包含实质性的技术分析、模型细节或对底层技术的实证评估。
    - 评论者注意到镜头中对视觉风格和美学滤镜的刻意选择，强调了其试图通过特定的颗粒效果和调色来唤起一种复古、空灵的 80 年代氛围，从而影响观众的感知和情感反应。
    - 另一个技术批评是，虽然视频展示了视觉上的新颖性，但缺乏机器人执行有用任务的演示，暗示目前的能力或演示仍停留在表面，并未专注于实际效用。

### 3. Veo 3 AI 视频生成在病毒式广告和创意项目中的应用

- [**不敢相信迪士尼批准了我的 AI 广告在今晚的 NBA 总决赛期间播出 🤣**](https://v.redd.it/kp3nkjksrc6f1) ([评分: 662, 评论: 123](https://www.reddit.com/r/aivideo/comments/1l92j6j/i_cant_believe_disney_approved_my_ai_commercial/)): **该帖子详细介绍了如何仅用两天时间，使用 Google Veo 3 (Google Flow) 为 Kalshi 制作一个病毒式 AI 生成 NBA 总决赛广告的完整技术工作流。该流程整合了 Gemini 用于剧本到提示词（prompt）的转换（批量生成 5 个以优化质量），Veo 3 用于基于提示词的视频生成（生成 300–400 个镜头以产出 15 个可用剪辑，每次生成成本约为 0.20 美元），以及使用 Capcut/FCPX/Premiere 进行快速剪辑。主要局限性包括缺乏角色一致性以及 Veo 3 中意外出现的字幕。这种方法声称比传统广告制作降低了 95% 的成本，并突出了在注意力驱动、基于喜剧的 AI 广告创作中新兴的挑战与机遇。[原帖](https://v.redd.it/kp3nkjksrc6f1)** 评论注意到赌博相关广告的日益盛行，并对 AI 流水线的透明拆解表示赞赏，其中一位提到其强大的 ROI，另一位则称该过程“混乱且酷炫”。目前没有实质性的技术辩论。
    - 一位用户对 AI 驱动的商业制作报酬模型表示关注，询问与传统制作相比，这些低成本方法对创作者而言在财务上是否具有可持续性。这引发了关于专业环境下 AI 生成媒体的市场价格和经济可行性的疑问。
    - 另一条评论强调了制作该 AI 广告所用工作流的细节，表明在幕后有大量的生成式 AI 技术集成，可能还涉及快速的内容迭代。技术读者可能会推断出在加速 AI 媒体制作中，高效流水线和工具选择的重要性。
- [**巨石：人类对神秘结构的反应**](https://v.redd.it/cmardtgcyc6f1) ([评分: 194, 评论: 21](https://www.reddit.com/r/aivideo/comments/1l93e3h/the_monoliths_humans_react_to_mysterious/)): **该帖子展示了一个名为“巨石”（The Monoliths）的多媒体项目，使用 Flow、Veo 3 和 Suno AI 创作——这些工具表明该流水线可能利用了先进的 AI 生成模型进行音视频合成。虽然没有提供基准测试、定量模型细节或明确的实现规范，但这种组合暗示了一个整合视觉（Flow, Veo 3）和音频（Suno AI）生成系统以进行连贯合成媒体制作的工作流。** 评论注意到结果在质量上的快速提升，暗示了生成式 AI 保真度和输出真实感的飞速进步，但并未讨论技术限制或伪影（artifacts）。

---

# AI Discord 摘要

> 由 Gemini 2.5 Flash Preview 生成的摘要之摘要的摘要
> 

**主题 1. 模型性能、基准测试与对比**

- **O3 Pro 基准测试差强人意**：初步的 [O3 Pro 基准测试](https://cdn.discordapp.com/attachments/1340554757827461211/1382439311781400596/image.png)显示与基础 O3 模型相比*几乎没有区别*，导致人们认为 O3 Pro *并不怎么令人印象深刻*。尽管基准测试令人失望，一些 Cursor 用户仍急切期待为 O3 Pro 的集成支付额外费用。
- **Kingfall 模型引发争议，基准测试遭到质疑**：成员们继续推测[难以捉摸的 Kingfall 模型](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924)，一位用户声称它在自动思考（auto-thinking）测试中显著优于 0605 (32k)。然而，另一位用户称这些是*荒谬的数据*，并暗示这可能是一个 [OpenRouter bug](https://openrouter.ai/)。
- **DeepSeek R1 模型表现亮眼，Gemini 表现挣扎**：**DeepSeek-R1-0528-UD-Q3_K_XL** 模型获得了接近 **80%** 的内部基准测试分数，这令 UnslothAI 成员感到兴奋，并分享了 [Hugging Face 链接](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF)。与此同时， Perplexity 和 OpenAI 用户报告 [Gemini 未能](https://ai.google.dev/)创建一个游戏，并且在分析 YouTube 视频时遇到困难，尽管有些人发现它在处理[这个链接](https://youtu.be/4MydzP3Mzy4?feature=shared)时表现完美。

**主题 2. AI Agent 框架与能力**

- **Windsurf 推出浏览器和“计划模式”（Plan Mode）**：Windsurf 发布了一款全新的[全功能浏览器](https://windsurf.com/blog/windsurf-wave-10-browser)，旨在桥接开发工作流与 Web 活动，目前面向 Free、Pro 和 Teams 用户提供 Beta 版，并附带 [YouTube 视频](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp)和[更新日志](https://windsurf.com/changelog)。Windsurf 还发布了全新的 **“计划模式”（Plan Mode）** 功能，允许 AI Agent 在 [Windsurf.com](http://windsurf.com/) 上利用规划文档处理复杂任务，该功能在早期测试中*表现良好*。
- **MCP Server 讨论推进了 UI、集成和评估**：围绕 **Multi-Compute Protocol (MCP)** 的讨论集中在如何利用 webio 传输将 Server 部署为 iframe 或前端 API，以解决当前问题并简化自定义 UI 和 [OAuth 流程](https://github.com/gitroomhq/postiz-app)的设置。[Hume AI 详细介绍了其评估方法](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server)，用于其 MCP Server，引发了对评估方法论的关注，且一张截图证实 Hugging Face 现在已支持 MCP。
- **DSPy 原语以其 Agent 能力令用户惊叹**：成员们称赞了 **DSPy** 的 Agent 模式，指出在将 Google 的 [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 重构为 **200 行** 的 [workflow.py](http://workflow.py/) 后，其原语展现了强大的威力。社区正在寻求更简便的 DSPy 数据集创建工具，并讨论了与集成 Tool-calling 的新推理模型的兼容性，期待即将发布的 [DSPy 3.0 版本](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1)。

**主题 3：AI 模型的定价与访问**

- **OpenAI 的 O3 Pro 定价先遭吐槽后下调**：EleutherAI 成员最初批评 [O3 Pro 的定价](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506)过高，每 1M token 的 **输入需 $20**、**输出需 $80**，并幽默地建议它应该能*解决黎曼猜想*。然而，随后 O3 的价格**下调了 80%**，降至每 1M token **输入 $2**、**输出 $8**，LMArena 和 OpenAI 成员一致认为这符合日益激烈的竞争和 Blackwell 的产能提升。
- **OpenRouter 提供近乎无限的 TPM，但使用 O3 需要 KYC**：OpenRouter 澄清说，虽然[速率限制因模型而异](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs)，但他们在实践中提供了极高的限制，这意味着对于不使用个人 OpenAI Key 的用户来说，*TPM 几乎是无限的*。Aider 成员指出，尽管价格大幅下调，OpenRouter 仍要求用户自带 Key 并完成 KYC 才能使用 O3 模型，一位用户表示 *OpenAI 让你先出示护照才允许你使用* `o3`。
- **Cohere 和 Veo 3 的定价引发惊愕**：Cohere 用户发现其创意写作的定价*高得离谱*，并报告 Reranking API 存在 **2 秒的延迟**，且由于不存在 API 分级，用户被引导发送邮件至 [carolyn@cohere.com](mailto:carolyn@cohere.com) 寻求定制方案。[Manus.im](http://manus.im/) 成员感叹单个 **Veo3 视频** 需要消耗 **300 积分**，认为其非常昂贵。

**主题 4：硬件与基础设施发展**

- **双路 AMD Turin 和高端 GPU 配置曝光**：讨论涉及了 **双路 AMD Turin** 服务器的潜力，该服务器可提供 **1.2TB/sec** 的内存带宽和 **640 GB/sec** 的 PCI-e，支持 **16 个 GPU 和 384GB VRAM**，并附带了 [Supermicro 主板链接](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu)。LM Studio 用户在 **Evo X2** 上成功运行了采用 **Q3_K_XL** 量化的 **Qwen 3 235b Unsloth**，达到了 **10-12t/s**，并讨论了针对 150B 模型的高带宽纯 CPU 方案（如八通道高速 RAM 服务器），但对能否达到 5 tokens/sec 表示怀疑。
- **疯狂的 GPU 定价促使用户等待 5090 并带动 Tinybox 二手销售**：Unsloth AI 成员讨论了当前 **4090** 定价的*荒谬性*，一些人选择等待 **5090**，并指出二手 **4090** 的价格也溢价过高。一位 tinygrad 成员正在以原价 **70%** 的价格出售数据中心退役的二手 **Tinybox Green 6X 4090**，引发了广泛*关注*。
- **Modular 平台拥抱 AMD，发布 Mammoth 系统**：Modular 宣布其平台在 **AMD InstinctTM MI300X** 和 **MI325 GPU** 上正式发布（GA），在 BF16 工作流中实现了高达 **53%** 的吞吐量提升，详见[其博客文章](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus)。他们还发布了 **Mammoth**，这是一个全新的[ Kubernetes 原生系统](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple)，旨在跨任何 GPU 扩展 GenAI 推理，目前提供公开预览。

**主题 5：前沿 AI 研究与基础概念**

- **LLM 记忆测量，Grokking 探索**：一篇新的[论文](https://arxiv.org/pdf/2505.24832)估计，在衡量模型对数据点的“了解”程度时，**GPT 系列**模型的容量约为 **3.6 bits-per-parameter**。研究观察到，模型会持续记忆直到达到其容量上限，随后 **Grokking** 开始发生。
- **Prompt 微调带来准确率巨大飞跃，乱序答案难倒模型**：成员们讨论了一篇[论文](https://arxiv.org/pdf/2311.01967)，该论文展示了 Prompt 的微小变化如何大幅影响 0-shot 任务的准确率。据[这篇论文](https://arxiv.org/pdf/2406.19470)描述，当 MMLU 问题的答案顺序被打乱时，模型往往会失败，即使它们之前能正确回答。
- **世界模型随 V-JEPA 和理论论证而进步**：根据 Meta AI 的[博客文章](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)和 [推文](https://x.com/lxrjl/status/1932499153596149875)，他们发布了新版本的 **V-JEPA**，旨在推进世界模型基准测试。一篇[论文](https://arxiv.org/abs/2506.01622)指出，任何能够泛化到多步目标导向任务的 Agent 都*必须*已经学习了其环境的预测模型。


---

# Discord: 高层级 Discord 摘要




## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 在游戏创作中表现不佳**：成员们注意到 [Gemini 未能](https://ai.google.dev/) 创建出一个可运行的游戏，同时讨论了 **Grok** 和 **Mistral**。
   - 虽然没有提供失败的具体原因，但讨论突显了 AI 在交互式应用中的复杂性。
- **Kontext 挑战 GPT Image**：新的 **Kontext** 系列表现优于 **GPT Image**，尽管开源版本尚待发布；**Pro** 和 **Max** 版本已可用，它是一个*轻量级的 12B diffusion transformer，适合定制并兼容之前的 FLUX.1 推理代码*。
   - **Kontext** 将通过 [FAL](https://www.fal.ai/)、[Replicate](https://replicate.com/)、[Runware](https://run.ai/runware/)、[DataCrunch](https://www.datacrunch.io/)、[TogetherAI](https://www.together.ai/) 和 [HuggingFace](https://huggingface.co/) 进行分发。
- **Palantir 关注美国**：根据 [Perplexity AI 搜索](https://www.perplexity.ai/page/palantir-builds-vast-us-survei-TGaRIDAIQA.I2nKHBK6W1g)，**Palantir** 正在美国开发大规模监控系统。
   - 报告附带了四张[截图](https://media.discordapp.net/attachments/1294380382661116045/1382398297456775288/1.jpg?ex=684b023d&is=6849b0bd&hm=35561a455d6f154727e9650366cdccefee0305d4dd36c59cead889c1b6d410c8&=&format=webp&width=648&height=864)，展示了 **Palantir** 的界面和数据可视化工具。
- **Postiz：又一个项目追踪器？**：成员们建议使用 [Postiz](https://github.com/gitroomhq/postiz-app) 进行项目管理。
   - 除了指向其 **GitHub** 仓库外，没有讨论具体的用例或项目细节。



---



## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Kingfall 的发布仍是一个谜**：成员们仍在好奇[神秘的 Kingfall 模型](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924)，一位用户分享了一张据称由 **Kingfall** 生成的 **iPhone 11 Pro** 图像。
   - 仍有推测认为 **Kingfall** 可能只是 **2.5 Flash lite** 的削减版本。
- **O3 Pro 基准测试表现平平**：初步的 [O3 Pro 基准测试](https://cdn.discordapp.com/attachments/1340554757827461211/1382439311781400596/image.png) 显示，与 **O3** 相比*几乎没有区别*。
   - 普遍观点认为 **O3 Pro** *并不怎么令人惊艳*。
- **思考预算的影响受到审视**：一位成员在不同的思考预算下运行了 **30 次** 测验，发现思考长度的增加并没有相关性，因此*长时思考比大模型重要得多*。
   - 小组一致认为，专注于长时训练运行可能比扩展模型规模更有利。
- **OpenAI 的定价策略受到密切关注**：成员们争论 **OpenAI** 是否对 **O3** *收费过高*，并一致认为最近 **5 倍** 的降价是为了应对日益激烈的竞争和 **Blackwell** 的产能提升。
   - 参与者认为降价信号是对市场压力和硬件能力进步的回应。
- **液体玻璃设计（Liquid Glass Design）成为趋势**：用户讨论了一种[液体玻璃设计](https://www.youtube.com/watch?v=1E3tv_3D95g)，一位用户表示*液体玻璃将真正成为下一个设计特色*。
   - 关于它[是否像液体玻璃清洁剂](https://i.imgur.com/GCzNMsh.png)尚未达成共识，引发了一些幽默的对比。



---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **GPT-6 被调侃**：在关于 **GPT-5** 的一些讨论后，一位用户开玩笑说 **GPT-6.1-pro-preview-patch** 已经发布了。
   - 其他社区成员似乎对未来的发布感到兴奋，但更感兴趣的是 **O3 Pro** 与基础模型相比的优势。
- **O3 Pro 诱惑 Cursor 用户**：用户正在激烈辩论 [**O3 Pro**](https://openai.com/blog/new-embedding-models-and-api-updates) 以及该模型是否物有所值，尽管一些人表示他们愿意为 Pro 支付额外费用，只要能在 Cursor 中使用它。
   - 许多用户注意到 **O3** 的速度已经快得惊人。
- **Mistral 模型既有趣又有益**：用户称赞 **Mistral** 模型作为聊天机器人和编码工具的有效性，并指出这些模型可以[免费使用](https://mistral.ai/)。
   - 一位用户特别赞赏 **Mistral** 在生成物理相关内容方面的精通程度。
- **后台 Agent 无法连接**：用户报告 **background agents** 无法连接的问题，经常显示 `ConnectError: [unknown] No response from model` 消息，特别是在达到 **25 次工具使用限制**之后。
   - 即使多次尝试重试，问题仍然存在，网络诊断显示正常，更多信息可在 [此 Discord 线程](https://discord.com/channels/1074847526655643750/1380811765218283660) 中找到。
- **Cursor 在处理 UE4 C++ 仓库方面完胜 Codex**：一位用户报告说，**Cursor** 在处理 **UE4 C++ repo** 方面的表现明显优于 **Codex**，并指出 **Codex 的失败率为 80%**。
   - 他们强调了 Cursor 中环境搭建的速度，只需 *不到半分钟*，而 Codex 则需要 *10 分钟*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O3 Pro 价格大跌引发思考**：用户正在讨论 **O3 Pro** 的 **价格下调**，想知道延迟发布是否是因为 **定价考量**。
   - 一些成员正在推测与 **web search 功能** 相关的成本。
- **Gemini 因 YouTube 视频分析失误遭吐槽**：用户报告 **Gemini** 无法分析 YouTube 视频，可能是因为登录了 Workspace/企业账号或关闭了 **YouTube extension**，而其他人发现 **Gemini** 可以很好地完成任务，例如通过[此链接](https://youtu.be/4MydzP3Mzy4?feature=shared)。
   - 成员们推测这是否是由新的 **O3 Pro** 推出引起的。
- **ChatGPT 拿不存在的 DOCX 功能吊胃口**：成员们抱怨 **ChatGPT** 用 **docx 文件** 诱导用户，但该功能实际上并不存在，且已 **禁用一个多月**。
   - 一位成员表示这令人失望，特别是当你试图汇总数据时，因为 ChatGPT 记不住十秒钟前的内容。
- **自定义 GPTs 声称具有竞争优势**：一位成员建议制作 **custom GPTs** 以获得更好的结果。
   - 他表示，根据模型版本或是否拥有订阅，你可以获得相应的优势。
- **安全配置文件引发审查**：一位成员分享了一个旨在使 **LLMs 更加可靠** 的 [配置文件](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161)，具有更低的幻觉、事实锚定以及对不可能结果的意识，这引发了关于提及明确负面约束（如 *CP is banned*）的辩论。
   - 一位成员认为，由于“粉色大象效应”，*在配置末尾列举禁止的 token 会放大近因偏差，并增加 LLM 泄露的风险*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **推理模式悬而未决**：一位用户询问通过 **OpenRouter** 发出的 **OpenAI** 请求默认使用哪种推理模式，但该问题尚未得到解答。
   - 用户特别想知道默认是 *detailed* 还是 *auto*。
- **O3 Pro 定价引发热议**：受 **Sama** 和 **Noam Brown** 推文的启发，用户对 **O3 Pro** 的定价表现出浓厚兴趣，一位用户形容 **O1 Pro** 的定价简直是“疯狂”。
   - 上下文中未提供 **O3 Pro** 的具体定价细节。
- **医疗专业人士寻求 OpenRouter 编码帮助**：一位医疗保健专业人士请求协助编写 **OpenRouter** 代码，以便将 **279 个提示词 (prompts)** 输入到各种 **LLMs** 中用于医疗共识项目，并分享了代码的 [pastebin 链接](https://pastes.dev/UqlfuJgF2W)。
   - 建议包括使用独特的角色提示词 (persona prompts)、速率限制 (rate limiting)、验证 **CSV** 数据，并确保 **LLM** 输出 **JSON** 以便更轻松地进行解析。
- **LLM 专家组成员讨论共识研究模型**：用户讨论了为共识研究选择 **LLMs** 的事宜，建议加入 **Gemini**、**Claude** 和 **Sonar** (Perplexity)，同时因性能原因排除 **OpenAI**、**DeepSeek** 和 **Grok**。
   - 有人指出，为了公平比较，应使用非推理模型，**Qwen** 需要设置为非推理模式，并且 [Grok 的上下文记忆力较低](https://x.com/OpenRouterAI/status/1932828433148776650) 且“非常愚蠢”。
- **OpenRouter 的 TPM 速率几乎无限制**：一位用户询问了 **OpenAI** 施加的 **TPM 速率限制** 是否适用于 **OpenRouter**，尤其是在不使用个人 **OpenAI** 密钥的情况下。
   - 官方澄清说，**OpenRouter** 在实践中拥有非常高的限制，这意味着对用户来说是 *无限 TPM*，并且 [速率限制确实会根据所使用的模型而有所不同](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs)。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **价格疯狂引发对 5090 的观望**：成员们讨论了 **4090** 虚高的价格，并考虑因当前市场状况而等待 **5090**。
   - 据报道，二手 **4090** 价格过高，进一步加剧了潜在买家的挫败感。
- **Magistral 量化版正式发布**：UnslothAI 团队宣布发布 **Magistral 量化版**，并与 Mistral 合作以确保准确性，正如[他们的推文](https://x.com/UnslothAI/status/1932441885618147402)所述。
   - 此次合作旨在保证“一切准确无误”。
- **DeepSeek R1 8Q 惊艳基准测试**：**DeepSeek-R1-0528-UD-Q3_K_XL** 模型在内部基准测试中表现出接近 **80%** 的性能，令成员们感到震惊，该模型现可通过此 [链接](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF) 获取。
   - 测试的模型是几天前的版本（6 月 8 日）。
- **Ollama 拥抱 Safetensors，告别权重文件！**：社区成员强调，虽然 `save_to_gguf` 仍处于开发中，但用户现在可以保存 **safetensors** 格式的合并模型。
   - 随后，他们可以利用 [ollama 的 create --quantize 命令](https://ollama.com/docs/guides/how-to-create-modify-prompt#quantize-your-model) 将其无缝转换为 **Ollama** 兼容格式。
- **提示词更改带来准确率大幅提升**：一位成员分享了一篇 [论文](https://arxiv.org/pdf/2311.01967)，强调提示词的微小变化会极大地改变 0-shot 任务的准确率。
   - 如[这篇论文](https://arxiv.org/pdf/2406.19470)所述，当答案顺序被打乱时，模型在 MMLU 问题上的表现也会失败。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Sonnet 4 的推理能力消失**：带有推理测试的 **Sonnet 4** 从 **OpenRouter** 中消失了，这表明其推理能力已不再可用。
   - 讨论成员推测 **Mistral** 的推理模式通过其聊天模板（[huggingface 链接](https://huggingface.co/mistralai/Magistral-Small-2506)）激活，但发现了诸如模型因奖励作弊（reward hacking）而循环输出最终答案等问题。
- **GRPO 的长度偏差受到攻击**：关于 **GRPO** 的讨论揭示了它存在长度偏差，这可能通过 **DRGRPO** 或 **DAPO** 等修复手段解决，内部训练环境使用了 [此 GitHub 链接](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py#L370C1-L397C71) 中描述的长度惩罚。
   - 成员们辩论了 **ProRL** 论文（[arxiv 链接](https://arxiv.org/abs/2505.24864)）对大型模型的适用性，质疑其关于熵崩溃（entropy collapse）问题的结论。
- **知识蒸馏曝光偏差显现**：一位成员正在使用 **student -> critic -> teacher** 架构进行蒸馏，存储教师模型的 logits 以微调学生模型，从而将学生模型升级为推理模型（reasoner）。
   - 他们担心来自教师推理模式的**曝光偏差**（exposure bias），并且不确定如何处理推理模型之间的蒸馏。
- **McCulloch-Pitts：AI 的基石**：[McCulloch-Pitts 论文](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf) 被认为是 **AI** 和**计算神经科学**的奠基之作。
   - 该论文提出的模型是**神经元**和**神经网络**的简化模型，能够计算任何**图灵可计算函数**（Turing-computable function）。
- **Frutiger Aero 复兴**：成员们表示希望将资源集中在恢复 **Frutiger Aero** 和实现 **AI Slop iMessage 背景**上，而不是 **AI 心脏监测**技术。
   - 一位成员表示“没有人关心能救命或延长寿命的 AI 心脏监测”。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio API 文档位置确定**：寻找 **LM Studio Server API** 的 **Swagger 文档**的用户被引导至[官方 API 文档](https://lmstudio.ai/docs/app/api)以及 LM Studio 内的 **Developer Tab**（开发者选项卡）。
   - 对话围绕如何构建 HTTP 请求以与 **LMS server** 交互展开。
- **慷慨提供图像生成建议**：一位拥有 **4070ti Super** 并寻求本地图像生成建议的用户被建议探索 **Fooocus** 和 **ComfyUI**，并可能使用 **Pinokio** 来安装独立解决方案。
   - 其他建议包括使用量化的 **Flux.dev** 模型和 **civitai** 上的微调模型，尽管 ComfyUI 的学习曲线较陡，但更倾向于使用 **sd-forge webui**。
- **Qwen 3 235b 可在 Evo X2 上运行**：一位用户询问是否能在 **Evo X2** 上运行 **Qwen 3 235b Unsloth**，另一位用户报告使用 **Q3_K_XL** 量化成功运行，达到了 **10-12t/s**。
   - 配置包括使用 Linux、10000 的上下文长度、KV 缓存的 Q8 量化、不使用 mmapping，以及 320 的 Evolution Batch Size。
- **双路 AMD Turin 是强悍的选择**：一台**双路 AMD Turin**服务器将拥有 **1.2TB/sec** 的内存带宽和 **640 GB/sec** 的 PCI-e，而一台配备 8 个双 B60 的机器将支持 **16 个 GPU 和 384GB VRAM**。
   - 一位用户链接到了一个拥有 20 个 x8 插槽的超微（Supermicro）主板 [Supermicro.com](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu)。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Mini PC RAM：板载焊接太令人难过了！**：成员们讨论了为什么 Mini PC 由于空间问题和使用板载焊接的 **LPDDR5**，内存限制在 **128GB RAM**。
   - 他们质疑为什么这些芯片不能焊接在内存条上，以允许更大的内存。
- **开源 LLM Agent 框架亮相**：一位成员开源了他们的 **LLM agent 框架** [可在 GitHub 上获取](https://github.com/starsnatched/llm-backend)，该框架可以使用具有完全访问权限的 **VM 上的 Linux 终端**，使其能够存储、修改、删除和移动文件，并从网络收集信息。
   - 作者强调，*我们永远无法像这样“解决历史”并实现量子级的真相保护……*
- **Qwen 3 1.7b 驱动对话式 Agent**：一位成员分享说，他们测试了小至 **Qwen 3 1.7b** 的模型来制作对话式 Agent，并将同一模型用于 **SQlord**。
   - 另一位成员询问该尺寸的模型效果是否良好，并表示有兴趣使用他们的函数来微调模型。
- **Agent 课程最终截止日期公布**：课程截止日期为 **7 月 1 日**，但新成员可以在一天内完成第一个单元并获得证书。
   - 成员们还报告了 **OpenAI 4o-mini** 模型的成本约为 **50 个问题 10 美元**，但随后的邮件指出 **o3** 比 **4o** 更便宜。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **GPT 模型记忆能力测量**：一篇新[论文](https://arxiv.org/pdf/2505.24832)估计，在衡量模型对数据点的“了解”程度时，**GPT 家族**模型的容量约为 **3.6 bits-per-parameter**。
   - 研究观察到，模型会进行记忆直到达到其容量上限，之后便开始 **“grokking”**（顿悟）。
- **DAN Agent 具备多模态特性亮相**：一位成员分享了他们的 **DAN (Do Anything Now) agent**，它可以根据单个提示词生成图像、视频、故事和旁白，通过[此 Ollama 链接](https://ollama.com/PythagoraTheorem/Aimee3)分享。
   - 该 Agent 包含具有记忆功能的对话模式、脚本改进扩展和终端操作器，旨在支持社区驱动的扩展。
- **Meta 的 V-JEPA 推进世界模型基准测试**：根据[这篇博客文章](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/)和[推文](https://x.com/lxrjl/status/1932499153596149875)，**Meta AI** 发布了新版本的 **V-JEPA**，旨在推进世界模型（world model）基准测试。
   - 此次发布旨在改进对能够更好理解和预测世界动态的 AI 模型的评估和开发。
- **Mistral 计算云出现**：根据[其博客文章](https://mistral.ai/news/mistral-compute)，**Mistral AI** 宣布推出 **Mistral Compute**，旨在使 AI 基础设施民主化，并为更多人提供构建和拥有 AI 基础设施的工具。
   - 此次发布包括各种旨在赋能 AI 领域开发者和研究人员的服务和资源。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 发布 Product Key Memory Sparse Coders**：EleutherAI 发布了 **Product Key Memory (PKM)** 稀疏编码器的 [博客文章](https://blog.eleuther.ai/pkm-coders/)、[代码](https://github.com/EleutherAI/sparsify/tree/e2e-pkm) 和 [权重 (checkpoints)](https://huggingface.co/EleutherAI/pkm-coders/)，供研究人员在 **Sparse Autoencoders** 和 **Transcoders** 中实验 **PKM** 模块。
   - 团队发现 **PKM** 加快了训练和推理速度，减少了内存占用，并诱导了层次化分组，但他们不确定这些改进是否足以抵消增加的复杂性。
- **O3 Pro 定价遭到抨击**：成员们批评了 [O3 Pro 的定价](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506)，指出其输入成本为 **$20 / 1M tokens**，输出成本为 **$80 / 1M tokens**。
   - 一些用户幽默地评论说，这种定价意味着它应该能 *解决黎曼猜想 (Riemann hypothesis)*，并认为它 *比 Muon 还差*，*比 RWKV 还差*。
- **哈佛图书馆数据集将解锁知识宝库**：一篇 [论文](https://arxiv.org/abs/2506.08300) 讨论了通过 **哈佛图书馆在 Google Books 中的份额** 使一套书籍可被访问，该数据集涵盖了约 **100 万本** 经核实属于公有领域的书籍。
   - 成员们认为该数据集在很大程度上是全新的，并且已经热切期待了一年多，代码和数据预计很快就会发布。
- **余弦衰减 (Cosine Decay) 受到审视**：成员们正在寻找关于 **余弦衰减** 到最小值（例如峰值的 10%）与衰减到 0 的论文和直觉，质疑最小值是否有助于小型 **SFT** 运行中的泛化，并讨论了 [这篇论文](https://arxiv.org/pdf/2502.15938)。
   - 一位成员建议对于小型 **LLM**，两个 epoch 效果最好（第一个 epoch 带有 warmup 和线性衰减，第二个 epoch 带有余弦衰减到 0），参考了 [这篇论文](https://arxiv.org/pdf/2404.06395) 和 [这项 Meta AI 研究](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)。
- **Knock Off 错误控制在 AI 安全领域重新浮现**：一位成员建议 **Knock Off 错误控制** 的想法可能很有用，并分享了 [Knockoff Inference](https://arxiv.org/abs/1811.06687) 论文的链接。
   - 另一位成员表示感谢，称他们已经大约 **4 年** 没见过有人提到 Knockoffs 了，这提醒了他们应该去学习一下。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **O3 价格大幅下调但 KYC 依然存在**：**O3 模型** 价格下调了 **80%**，现在输入为 **$2**，输出为 **$8**，但 [OpenRouter](https://openrouter.ai/) 仍然要求用户提供自己的 Key 并进行 **KYC**。
   - 一位用户指出，*OpenAI 在让你使用 `o3` 之前会让你出示护照*。
- **Kingfall 表现出色，基准测试受到质疑**：一位用户对比了 **0605 (32k)** 与 **Kingfall (auto thinking)**，显示 **Kingfall** 在自动思考方面表现好得多。
   - 然而，另一位用户对这些数据提出了异议，称其为 *荒谬的数据*，并可能是一个 [OpenRouter bug](https://openrouter.ai/)。
- **R1 让用户期待更多**：成员们辩论了新 **R1 模型** 的性能，有人认为新的 **R1** 在 **$4.8** 的价位更好，但另一人反驳称他们认为新的 **R1** 甚至比 `0506` 还差。
   - 一位成员表示：*几乎所有的基准测试都是开源的，这意味着 AI 公司可以在这些基准上进行训练*，从而对所有基准测试的实用性表示怀疑。
- **Aider 用户疯狂使用 Pro Max**：一位用户提到他们正在大量使用 **Pro Max 订阅**，以至于因为并行运行多个 **Claude 代码实例** 而收到了使用警告。
   - 该用户开玩笑地承认 *这确实是自找的*。
- **Deepseek-R1 配置令人头疼**：一位用户在 Aider 中从 chutes 配置 **Deepseek-R1** 时遇到困难。
   - 另一位用户建议设置 [.env 文件](https://aider.chat/docs/config/dotenv.html) 来进行配置。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **自定义 AI 风格引发辩论**：一位用户询问如何根据年龄、种族和性别等人口统计学特征定制 **AI 对话风格**，以获得更**个性化的 AI 交互**。
   - 这场对话引发了人们对获取所需**对话风格**的技术和工具的兴趣，但目前尚未确定具体的解决方案。
- **音频概览引发困惑**：新用户正努力在 NotebookLM 中自定义 **AI 音频概览生成器**，以便为每个主题创建独立的概览。
   - 用户反映在应用中难以找到“自定义”选项，并正在寻求**从文档生成音频**的最佳实践。
- **探索 Pierceday Metalabel 使用案例**：用户正在探索 [Pierceday Metalabel](https://pierceday.metalabel.com/aphone) 的各种用例，建议将其应用于汽车手册、维护详情、电箱说明和**会议演示**。
   - 有人指出，该工具对于为各种物理对象添加**上下文信息**非常有用。
- **播客长度限制引发讨论**：受在线示例启发，用户正在研究使用 NotebookLM 生成**超过 20 分钟播客**的方法。
   - 对话集中于寻找从 NotebookLM 源文件**生成更长音频内容**的变通方法和最佳实践。
- **LaTeXLM 扩展发布**：一位用户发布了一个[开源 Chrome 扩展](https://github.com/hachoj/LaTeXLM)，用于在 NotebookLM 中启用 **MathJax 渲染**。
   - 该扩展旨在改进**数学公式和符号**的显示，目前已在 GitHub 上线，并计划未来在 Chrome Web Store 发布。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **在 Windsurf 退出之际，Cursor 关注 Anthropic 投资**：随着 **Windsurf** 现已与 **OpenAI** 结盟，**Cursor** 可能会加倍对 **Anthropic** 的投资，这引发了关于潜在交易的疑问。
   - 社区认为 **Cursor** 在 **Claude Code** 上的投入比 **OpenAI** 对 **Codex** 的投入更多，用户反馈 **Claude Code** 对他们来说非常契合。
- **苹果考虑收购 Anthropic 以升级 Siri**：有推测认为，凭借其财务能力和改进 **Siri** 的迫切需求，**Apple** 应该收购 **Anthropic**。
   - 一位成员调侃道，**Siri** *在领先 15 年的情况下都无法可靠地发送短信*，突显了改进的紧迫性。
- **Altman 暗示“温和的奇点”**：[Sam Altman 的博客文章](https://blog.samaltman.com/the-gentle-singularity)引发了关于**温和奇点**的兴趣和讨论。
   - 该链接与 [Kevin Hou 的 X 帖子](https://x.com/kevinhou22/status/1932516093333266538?s=46)一同被分享，进一步推动了对话。
- **Windsurf 的“计划模式”管理任务**：**Windsurf** 发布了全新的 **'Plan Mode'** 功能，允许 **AI Agent** 使用规划文档执行复杂任务，可在 [Windsurf.com](https://windsurf.com/) 免费使用。
   - 在一个小型新项目上的早期测试显示，**'Plan Mode'** *运行良好*。
- **Sharon Zhou 加入 AMD**：Sharon Zhou 加入 **AMD**，将与 **Lisa Su** 合作，专注于 **AI 研究**和扩展，并带来了来自 **LaminiAI** 的同事。
   - Zhou 的目标是为 **AI** **普及 GPU**，正如她在 **#AdvancingAI** 大会上的[这篇 X 帖子](https://x.com/realSharonZhou/status/1932817096510931380)中所强调的那样。



---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Hume AI 发布 MCP Evals**：**Hume AI** 在[一篇新博客文章](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server)中介绍了其为 **MCP server** 提供的 *evals* 方法，并邀请大家对评估方法论进行讨论。
   - 社区成员对其他人如何评估其系统表示好奇。
- **iframe MCP Servers 引发辩论**：将 **MCP server** 部署为 *iframe* 或前端 API 的概念受到关注，支持者主张利用类似于现有 stdio transport 的 **webio transport**。
   - 该方法有望通过 `window.open` 实现自定义 UI 和 **OAuth flows**，通过 URL 复制粘贴简化设置，以及通过 web APIs 而不是授予 shell 访问权限来管理虚拟文件系统，从而解决当前的 **MCP** 问题。
- **Hugging Face 进入 MCP 领域**：一张截图显示 **Hugging Face** 现在已支持 **MCP**，这对于从事模型开发的人员来说是一项重大进展。
   - 这对涉及模型开发的人员来说非常令人兴奋。
- **OAuth2 UI 简化 MCP 连接**：为身份验证集成真实的 **OAuth2 UI** 将显著简化终端用户与 **Google** 和 **GitHub** 等服务的连接，极大地提升 **MCP** servers 的用户体验。
   - 理想情况下，如果 **OpenAI**、**Anthropic** 和 **Google** 为其 API 实现 **OAuth2** 登录，将进一步简化这一过程。
- **Slides.com 拥抱 MCP**：一个托管的 **MCP server** 已上线，可用于生成 [slides.com 演示文稿](https://www.epicai.pro/use-ai-to-create-presentations-with-mcp-tsb4j)。
   - 这代表了 **MCP** 在内容创作中的实际应用。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Edu 邮箱排除英国域名**：一位成员询问 *.edu* 邮箱域名是否包含 *.UK* (*ac.uk*) 域名，另一位成员表示并未看到 *UK* 域名被包含在内。
   - 一张截图确认了 *.UK* 域名被排除在 *.edu* 域名列表之外。
- **ReactNexus 定于 2025 年 7 月举行**：成员们讨论了即将举行的 **ReactNexus** 活动（[https://reactnexus.com/](https://reactnexus.com/)），该活动定于 **2025 年 7 月 3 日至 5 日**在 **J N Tata Auditorium** 举行。
   - 该会议专注于 **React**，这是一个用于构建用户界面的流行 JavaScript 库。
- **Veo 3 定价引发不满**：一位用户抱怨单个 **Veo3 视频** 需要消耗 **300 积分**，处理 **38 个剪辑** 的成本非常高昂。
   - 另一位用户对此表示赞同，对高昂的费用表示担忧，并质疑 **Veo3** 对更广泛受众的可及性。
- **Manus Chat Mode 免费上线**：**Manus** 为所有用户推出了 **免费且无限制的 Chat Mode**，使用户能够提问并获得即时回答。
   - 用户可以升级到 **Agent Mode** 以获得更高级的功能，例如创建全面的输出内容。
- **High Effort Mode 消失**：多位用户报告其 **Pro 账户** 中的 **High Effort Mode** 消失了。
   - 一位用户对需要手动选择 **High Effort Mode** 这一奇怪要求发表了评论。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 重构 Operations 目录**：**Tinygrad** 项目正在通过将 operations 移动到独立目录来[重构](https://xl0.github.io/tinygrad-notes/bounty1.html)其代码库，这可能会影响直接使用 **Tinygrad** 内部机制的开发者。
   - 其目的可能是为了改进代码组织和可维护性。
- **二手 Tinybox Green 出售**：一位成员正在以原价 **70%** 的价格出售一台二手的 **Tinybox Green 6X 4090**，并注明其来自数据中心，处于完美的运行状态。
   - 另一位成员表达了*兴趣*，预示着社区内可能达成交易。
- **关于 CS 学位相关性的辩论**：一位成员质疑了 **CS 学位** 的持续用途，引发了关于其在当前技术领域价值的讨论。
   - 其中一个回应认为，质疑学位的价值本身就否定了其必要性，这表明一种观点，即其益处应该是显而易见的。
- **SVD 悬赏任务引发算法讨论**：**linalg.svd 悬赏任务** 的讨论促成了一项在 **Tinygrad** 中使用 **Jacobi 算法** 进行特征值计算的提案，成员们正在考虑手动编写函数以确保零依赖。
   - 还有人建议使用 **Jacobi 算法** 的改进版本，参考了 [A Novel Fully Hardware-Implemented SVD Solver Based on Ultra-Parallel BCV Jacobi Algorithm](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684b65f1&is=684a1471&hm=3eeaa1287761b9210d1e4a54b7c65b1be2a3c4b3838d55d14e60ca76d8cbefc7&)。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 在 AMD 上发布，Mammoth 亮相**：**Modular 平台** 现在支持 **AMD InstinctTM MI300X** 和 **MI325 GPU**，在预填充密集型（prefill-heavy）的 **BF16** 工作流中将吞吐量提升了高达 **53%**，详见[其博客文章](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus)。
   - **Mammoth** 是一个新的 **Kubernetes 原生系统**，可在任何 GPU 上扩展 **GenAI 推理**，支持使用单个容器在 **AMD** 和 **NVIDIA** 上部署 Hugging Face 模型，无需手动配置，目前已提供[公开预览版](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple)。
- **Mojo 与 Python 集成**：**Mojo kernel** 现在可以直接集成到 Python 工作流中，由 **45 万行以上的开源 Mojo kernel 代码** 提供支持，并在 nightly 构建版本中可用。
   - 开发者可以根据[此文档](https://docs.modular.com/mojo/manual/python/mojo-from-python/)开始在 Python 环境中使用 Mojo。
- **TensorWave 为 Modular 提供免费 AMD 算力**：通过与 **TensorWave** 合作，用户现在可以使用免费的 AMD 算力在实际工作负载中测试 Modular 平台。
   - 感兴趣的用户可以通过 [Modular.com/tensorwave](https://www.modular.com/tensorwave) 获取此优惠，以评估平台的性能表现。
- **Mojo 交叉编译引发跨平台关注**：Mojo 目前不支持从 macOS 到 Linux 的直接跨平台静态编译，这促使一位成员探索 Docker 容器化方案。
   - 该成员遇到了 *'apple-m1' is not a recognized processor'* 错误，从而考虑打包依赖项和 `.mojo` 文件，以便在无服务器（serverless）平台的 **Docker 容器** 中运行。
- **Mojo 在 Runpod 上运行迅速，起步很快！**：一位成员报告了在 [runpod.io](https://runpod.io) 上成功运行 Mojo GPU 代码的经历，指出其在快速热执行（hot executions）方面表现良好。
   - 初始冷启动时间约为 10 秒，但他们计划在论坛上分享详细的设置帖子。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Flex Attention 在压力下出现峰值**：一位用户报告称，在使用 **flex attention** 时，`(bs, seqlen*8)` 输入的峰值显存占用高于 `(bs*8, seqlen)` 输入，怀疑是 self-attention 中的 softmax 方阵导致的。
   - 该用户正在调查 **flex attention** 和 **FSDP** 的峰值显存占用情况，并指出在运行 sweep 测试时，显存占用在达到一个“临界点”后会迅速飙升。
- **Tokenizer 集成调整即将到来**：一位成员计划对 [#2574](https://github.com/pytorch/torchtune/pull/2574) 和 [#2794](https://github.com/pytorch/torchtune/pull/2794) 进行迭代，以改进新的 **HF tokenizer** 及其集成，并计划使用 **C4** 进行实验。
   - 预计将有一个 Pull Request 来修复 [#2809](https://github.com/pytorch/torchtune/pull/2809)，从而进一步巩固 **tokenizer** 的支持。
- **可迭代数据集获得 Packing 重构**：[Proposal on packing refactor](https://github.com/pytorch/torchtune/pull/2819) 中提出了一项针对可迭代数据集的 packing 重构建议，以支持 **DPO**、**GRPO**、**multimodal** 等的 packing。
   - 这些更改预计将扩大模型支持的配置范围。
- **Nemo RL 架构图曝光**：源自 Databricks 会议的 **Nemo RL** 计划已经[曝光](https://cdn.discordapp.com/attachments/1360680363885854841/1382420201823404032/AP1GczOQioexSd_ieqkppCKoVizt91prnymZ_uGi6mCeQdrSJE65osblAXMqxQw3030-h2272-s-no-gm.png?ex=684b16a4&is=6849c524&hm=8ca8961e205603c01114bb66f46acb3bb86d01b2d1297bef22f7817f5b6efeca&)。
   - 进一步的细节较少，但图表概述了正在采取的通用方法。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的成本引发担忧**：成员们对 **Cohere** 在创意写作任务上的定价表示担忧，认为其过于昂贵，同时强调了其目前处于 beta 阶段的 **North** Agent。
   - 一位成员指出，虽然 [Claude code](https://www.anthropic.com/product) 在代码生成和理解方面表现出色，但在构建 Agent 时，[n8n](https://n8n.io/) 更适合工作流自动化。
- **多模态 Re-Ranker 请求被回绝**：一位成员询问 **Cohere** 是否会发布用于图像重排序的 **multi-modal re-ranker**，但被告知 **Cohere** 目前不提供此类产品。
   - 建议的替代方案包括使用具有结构化输出的 **GPT-4.1**，或探索 **CLIP** 和 **openCLIP** 以寻求相关解决方案。
- **Cohere 与 EnsembleHP 合作推进 North**：**Cohere** 正与 **EnsembleHP** 合作，通过其安全的 **AI agents platform** 在医疗保健领域部署 **Cohere North**。
   - 目标是减轻行政负担并改善患者体验；更多详情请参阅 [Cohere blog](https://cohere.com/blog/ensemble-partnership)。
- **目前未提供 API 分级**：一位用户询问了类似于 **OpenAI** 的 API 分级（tiers），但被告知 **Cohere** 不提供预定义的 API 分级。
   - 在一位用户报告 **reranking API** 存在 **2 秒延迟**后，一位成员建议联系 [carolyn@cohere.com](mailto:carolyn@cohere.com) 获取定制解决方案。
- **Datatune 凭借自然语言转换表现出色**：**Vitalops** 的联合创始人介绍了 [Datatune](https://github.com/vitalops/datatune)，这是一个使用纯自然语言执行数据转换的开源工具。
   - 他们表达了加入 **Cohere** 社区并向其他成员学习以及参与社区建设的热情。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Cleanlab 和 LlamaIndex 集成以提供值得信赖的见解**：[CleanlabAI](https://cleanlab.ai/) 和 [LlamaIndex](https://www.llamaindex.ai/) 已集成，用于构建 AI 知识助手和生产级 Agent，通过对信任度评分和检测幻觉来增强 **LLM responses** 的可靠性。
   - 该集成已在 [Twitter 上宣布](https://twitter.com/llama_index/status/1932837489238290941)，旨在提高从企业数据中获得的见解的可靠性。
- **社区呼吁 LlamaIndex 接收 Chainlit 代码！**：随着 [Chainlit](https://github.com/Chainlit/chainlit) 停止维护，用户正积极鼓励 LlamaIndex 收购其代码，强调其在 **LLM ecosystem** 中的重要性以及与 LlamaIndex 的无缝集成。
   - Chainlit 因其纯 Python 实现和简单部署而受到赞誉，允许在 Discord 和 Slack 等多个平台上使用，一位成员指出 *LlamaIndex + Chainlit 的组合效果惊人！*
- **AI 安全研讨会应对 Hacken 的风险**：Hacken 将于 **UTC 时间 6 月 12 日 13:00** 举办一场关于 **AI security** 的研讨会，探讨 **LLM vulnerabilities** 和防御措施，由 Stephen Ajayi 主讲。
   - 感兴趣的人士可以通过 [Luma 链接](https://lu.ma/xl53xbfs) 了解更多详情并注册，学习如何处理任何意想不到的 Hacken 风险。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Gemini 全栈 LangGraph 快速入门发布**：Google 发布了一个名为 [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 的研究应用全栈实现，成员们指出这是一个 *非常出色* 的实现。
   - 随后，一位成员使用 **DSPy** 重构了 **Gemini** 代码中的 **LangGraph** 部分，并实现了一个简单的 **React** 前端，代码已发布在 [GitHub 上](https://github.com/pgurazada/deep-research.git)。
- **DSPy 的 Agentic Pattern 表现强劲**：成员们使用 **DSPy** 实现了 *非常多* 的 **agentic patterns**，并对 *这些原语的强大功能感到震惊*。
   - 重构后的工作流仅有 **200 行长** ([workflow.py](https://github.com/pgurazada/deep-research/blob/main/backend/agent/workflow.py))，并且 *以更少的麻烦优雅地实现了原始的 LangGraph 工作流*。
- **社区寻求 DSPy 数据集开发工具**：一位成员询问是否有工具可以轻松构建和导出 **DSPy** 数据集，以促进合成示例生成和手动标注。
   - 另一位成员建议自定义 **Streamlit app** 可能会很有效，而像 **Cline** 这样的编程 Agent 可以在极少指导下协助创建。
- **推理模型与 DSPy 首次亮相**：一位成员询问 **DSPy** 与在推理过程中使用 tool-calling 的新型推理模型（如 **o3 / o3-pro / o4-mini**）的兼容性。
   - 他们指出，虽然 `dspy.ReACT` 存在，但它似乎是为聊天 API 时代设计的，而不是为集成了 tool-calling 的响应 API 时代设计的。
- **DSPy 3.0 即将到来**：一位成员宣布了即将发布的 **DSPy 3.0** 版本，并链接到了 [DSPy 3.0.0b1 release tag](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1)。
   - 他们询问是否有关于 **DSPy 3.0** 未来功能的全面概述。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Python SDK 更新引发热议**：**GPT4All** 频道的成员对即将到来的 **Python SDK** 更新表示期待，目前尚未透露具体细节。
   - 社区正等待看这次更新将为 **GPT4All** 生态系统带来哪些改进和功能。
- **GPT4All 关注 Magistral Small**：一位成员询问 **GPT4All** 是否会支持 **Mistral** 的 **Magistral Small** 模型，引发了简短的讨论。
   - 其他成员建议了 **JAN**、**LM-Studio**、**obadooga** 和 **koboldcpp** 等替代方案，而原询问者因担心模型速度决定继续等待。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **AgentX 提交是否自动参赛？**：一位成员询问向 **Research Track** 竞赛提交论文是否会自动进入 **AgentX summit** 的征文和提案考虑范围。
   - 他们想澄清是否需要为峰会单独提交。
- **AgentX 决赛入围者必须注册吗？**：一位成员还询问决赛入围者是否需要注册峰会才能参加。
   - 他们担心门票可能会在竞赛结果公布前售罄，如果没能入围决赛，可能无法参加。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Cerebras 举办 AI 技术讲座**：Cerebras 将于 **PST 时间 6 月 13 日（星期五）中午 12:00–1:00** 举办一场免费的 AI 工作坊，演讲嘉宾来自 Cerebras 和 Artificial Analysis。
   - 讲座将涵盖包括**阿里巴巴 Qwen 3 系列**在内的新模型以及模型选择策略，[在此预约 (RSVP)](https://lu.ma/7f32yy6i?tk=jTLuIY&utm_source=ella)。
- **AI 工作坊专注于模型选择**：该 AI 工作坊深入探讨当前研究，演示如何为项目选择合适的模型。
   - 研究人员将为有效的模型选择提供见解和策略。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf 发布全功能浏览器**：作为 [Windsurf Wave 10 - Day 2](https://windsurf.com/blog/windsurf-wave-10-browser) 的一部分，Windsurf 正在推出一款全新的**全功能浏览器**，旨在弥合开发工作流与 Web 活动之间的差距。
   - 全新的 **Windsurf Browser** 正在向所有 **Free、Pro 和 Teams 用户**推出 Beta 版，Enterprise 用户将滚动获得此功能。
- **Windsurf Browser 向所有人开放**：全新的 **Windsurf Browser** 正在向所有 **Free、Pro 和 Teams 用户**推出 Beta 版，Enterprise 用户将滚动获得此功能。
   - 观看 [Youtube 视频](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp)，阅读 [更新日志 (changelog)](https://windsurf.com/changelog) 或加入 [r/Windsurf 的讨论](https://reddit.com/r/windsurf)。
- **Windsurf 拓展社交媒体版图**：Windsurf 在 [X/Twitter](https://x.com/windsurf_ai/status/1932871558219117022)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social)、[Threads](https://www.threads.com/@windsurf_ai/post/DKxShipsbPk?hl=en)、[Instagram](https://www.instagram.com/p/DKxWKKkxvu6/) 和 [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7338638111393886211/) 上创建了新账号。
   - 关注 **Windsurf 的社交媒体**以获取最新的公司新闻和产品更新。



---



## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 排行榜即将更新**：**Gorilla LLM** 排行榜将于本周更新。
   - 此消息发布在 `#leaderboard` 频道。
- **第二个话题占位符**：这是一个占位符话题，以满足最低条目要求。
   - 它不反映任何实际讨论，但对于验证是必要的。



---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间保持沉默，请告知我们，我们将将其移除。


---



您收到这封邮件是因为您通过我们的网站选择了订阅。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [退订]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord：各频道详细摘要与链接





### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1382076539293798521)** (1271 条消息🔥🔥🔥): 

> `Grok, Gemini, Mistral, Kontext, Flux` 


- **Gemini 的游戏计划**：成员们讨论了 **Grok**、**Gemini** 和 **Mistral**，并提到 [Gemini 没能](https://ai.google.dev/) 制作出一个真正的游戏。
- **Flux Kontext 是新的 GPT Image 杀手**：**Kontext** 是新的 Flux 系列，其性能超越了 GPT Image，尽管[开源版本尚未发布](mailto:kontext-dev@blackforestlabs.ai)，但 **Pro** 和 **Max** 版本已经上线。
   - 它是一个*轻量级的 12B diffusion transformer，适合定制并兼容之前的 FLUX.1 推理代码*，将通过 [FAL](https://www.fal.ai/)、[Replicate](https://replicate.com/)、[Runware](https://run.ai/runware/)、[DataCrunch](https://www.datacrunch.io/)、[TogetherAI](https://www.together.ai/) 和 [HuggingFace](https://huggingface.co/) 进行分发。
- **Perplexity 用户希望放宽 O3 的限制**：用户对 O3 每周 **100 次**的限制感到不满，认为既然 O3 更便宜，限制应该在*每周 400-500 次*，并建议[向 Gemini 的限制看齐](https://ai.google.dev/)。
- **DIA vs Comet 浏览器**：成员们讨论了 [DIA](https://dia.com/) 和 [Comet](https://www.comet.com/site/)。
   - 成员们对此表示期待。
- **GPT-5 flash：Apple open ai pplx glazer**：成员们讨论了 GPT-5 将对免费用户开放无限次使用，但会有类似 GPT-4o 的速率限制。
   - 用户讨论了 [GPT5 mini](https://openai.com/blog/openai-devday)。


  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1382352832224559277)** (2 条消息): 

> `Palantir surveillance, Government contracts` 


- **Palantir 构建庞大的美国监控系统**：根据 [Perplexity AI 搜索](https://www.perplexity.ai/page/palantir-builds-vast-us-survei-TGaRIDAIQA.I2nKHBK6W1g)显示，Palantir 正在构建一个庞大的美国监控系统。
- **图像转储显示 Palantir 屏幕截图**：四张 [屏幕截图](https://media.discordapp.net/attachments/1294380382661116045/1382398297456775288/1.jpg?ex=684b023d&is=6849b0bd&hm=35561a455d6f154727e9650366cdccefee0305d4dd36c59cead889c1b6d410c8&=&format=webp&width=648&height=864) 展示了 Palantir 的界面。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1382234369586102284)** (4 条消息): 

> `postiz` 


- **为你的项目尝试 Postiz**：一名成员建议在项目管理中尝试 [Postiz](https://github.com/gitroomhq/postiz-app)。
   - 未提供其他详细信息。
- **Postiz GitHub 仓库**：[Postiz](https://github.com/gitroomhq/postiz-app) 是一个 GitHub 仓库，但未给出进一步细节。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1382071853983469721)** (935 条消息🔥🔥🔥): 

> `Kingfall release, O3 Pro benchmarks, Thinking budget for models, O3 pricing, Liquid glass design` 


- **Kingfall 发布依然神秘**：成员们讨论了[难以捉摸的 Kingfall 模型](https://tenor.com/view/wink-eye-wink-gif-3023120962008687924)，一位用户分享了一张据称由 **Kingfall** 创建的 iPhone 11 Pro 图像，并称其为“下一代产品”。
   - 有人怀疑 **Kingfall** 可能只是 **2.5 Flash lite** 的削弱版。
- **O3 Pro 基准测试表现平平**：[O3 Pro 的初步基准测试](https://cdn.discordapp.net/attachments/1340554757827461211/1382439311781400596/image.png)已经出炉，用户反映与 **O3** 相比“几乎没有区别”。
   - 普遍观点认为 **O3 Pro** “并不怎么令人惊艳”。
- **思考预算（Thinking Budget）的影响受到审视**：一名成员在不同的思考预算下运行了 **30 次** 测验，但发现与长度增加*没有*相关性。
   - 结论是：*长时间的思考比大模型更重要*。
- **OpenAI 的定价策略受到质疑**：成员们辩论了 **OpenAI** 是否对 **O3** 收费过高，并一致认为最近 **5 倍** 的降价是为了应对日益激烈的竞争和 **Blackwell** 的产能。
   - 一位成员调侃道：“哈哈……是的。我不会太愤世嫉俗，否则会抑郁的。”
- **液态玻璃（Liquid Glass）设计成为热门话题**：用户讨论了一种 [液态玻璃设计](https://www.youtube.com/watch?v=1E3tv_3D95g)，一位用户表示“液态玻璃将真正成为下一个设计特色”。
   - 然而，对于它[是否像液态玻璃清洁剂](https://i.imgur.com/GCzNMsh.png)，尚未达成共识。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1382071824069693661)** (513 条消息🔥🔥🔥): 

> `LLMs: OpenAI vs Claude vs Gemini, Cursor for non-coding tasks, Cursor Indexing Issues, O3 vs Sonnet, GPT-5 (or GPT-6)` 


- **是否选择 O3 Pro —— 模型、定价与速度？**：成员们讨论了 [**O3 Pro**](https://openai.com/blog/new-embedding-models-and-api-updates) 是否值得比普通 **O3** 支付额外费用。用户对 O3 印象深刻，并准备为 Pro 版支付额外费用，只为在 Cursor 中使用它。
- **GPT-6（及以后！）即将到来！**：在提到 **GPT-5** 后，一位用户开玩笑地建议 **GPT-6.1-pro-preview-patch** 已经发布了！
- **Taskmaster 可能会搞砸子任务！**：一位用户分享了使用 **Taskmaster** 的经验，指出虽然它能正确跟踪任务，但从长远来看，子任务可能会打乱整个流程。
- **Mistral 模型物美价廉！**：用户表达了对 **Mistral** 模型的赞赏，注意到它们作为聊天机器人的有效性以及良好的编程能力，一位用户强调了它在生成物理相关内容方面的熟练程度，且[免费提供](https://mistral.ai/)。
- **Cursor UI 隐藏错误信息**：用户反映 Cursor 的 UI 现在会在一秒钟后隐藏工具调用（tool call）的错误信息，这导致了“透明度缺失的感觉”。


  

---

### **Cursor Community ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1382082850500902953)** (13 条消息🔥): 

> `Background Agents 连接问题，Cursor 与 Codex 性能对比，Agents.md 实现，Cursor 中 "Fix in Cursor" 的分支错误` 


- **Background Agents 面临连接失败**：多名用户报告了 **background agents** 无法连接的问题，通常显示 `ConnectError: [unknown] No response from model` 消息，特别是在达到 **25 次工具使用限制**之后。
   - 一位用户指出，即使多次尝试重试，问题依然存在，且网络诊断显示为绿色，并链接到了该话题相关的 [Discord 讨论帖](https://discord.com/channels/1074847526655643750/1380811765218283660)。
- **Cursor 在处理 UE4 C++ 仓库方面优于 Codex**：一位用户赞扬了 **Cursor** 在处理 **UE4 C++ 仓库**时优于 **Codex** 的表现，称 **Codex 80% 的失败率**使其*无法使用*，并称赞了 Cursor 的编码热情和更快的环境搭建速度。
   - 该用户强调了 Cursor 中环境搭建的速度，只需*不到半分钟*，而 Codex 则需要 *10 分钟*。
- **提议为 Cursor 建立 Agents.md 速查表**：一位用户建议实现一个 `Agents.md` 文件作为 **Cursor AI agents** 的速查表，借鉴他们在 **Codex** 中的经验，以维持复杂性并加速任务处理。
   - 该用户提到，将速查表放在 AI 在执行任何任务前都会引用的文件中，有助于降低复杂性并加快进程。
- **"Fix in Cursor" 功能深受 "Incorrect Branch" 错误困扰**：用户在使用 *Fix in Cursor* 功能时遇到了问题，即使处于正确的分支，也经常卡在 *Incorrect Branch*（错误分支）的重试循环中。
   - 尽管尝试再次签出分支，错误仍然存续，系统提示为 *incorrect workspace*（工作区不正确）。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 条消息): 

OpenAI: ### OpenAI o3-pro 现已向 ChatGPT 和 API 中的所有 Pro 用户推出。
  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1382074156740247614)** (254 条消息🔥🔥): 

> `O3 Pro 推出，Gemini YouTube 分析，Reasoner Model 上的 KD，项目中的 O3 Pro，Team 方案 vs Pro 方案` 


- **O3 Pro 降价引发热潮**：用户讨论了 **O3 Pro** 的**降价**，一些人质疑延迟发布是否是因为定价考虑，而另一些人则猜测与网页搜索功能相关的成本。
- **Gemini 因 YouTube 理解失误受到批评**：用户报告称 **Gemini** 无法分析 YouTube 视频，可能是因为使用了 Workspace/企业账号登录或关闭了 YouTube 扩展，而其他人则发现 Gemini 可以完美运行。
   - 一位用户使用 [此链接](https://youtu.be/4MydzP3Mzy4?feature=shared) 进行了测试，没有发现问题。
- **O3 Pro 存在问题：提示词工程师发布故障报告**：订阅 Pro 方案的用户在 O3 Pro 模式下遇到问题，看到错误消息 *Error in message stream*，一位用户分享了产生该错误的 [此提示词](link.to.prompt)。
   - 他们一直在报告 Bug，推测今天的服务器可能出现了故障。
- **Team 方案虽然诱人但 Token 受限**：用户讨论了 Team 方案的 **O3 消息上限**，每周仅为 **100 条**。一名成员还抱怨 Team 方案缺少内部知识功能（该功能可帮助模型在毫秒内获取上下文并做出响应），但 O3 是无限的。
   - 另一位用户表示，*他们有意不为 Team 方案释放更多 o3 的配额上限，因为 O3 + 内部资源功能确实非常智能*。
- **为前沿推理模型进行知识蒸馏**：一位成员正考虑将其技术栈中的学生模型升级为推理模型，并询问是否有人在 **Reasoner Model** 上使用 **KD**（知识蒸馏）。
   - 他们提到，*大多数前沿推理模型都会隐藏推理 Logits 和 Token，我认为如果切换到推理模型，可能需要放弃这一点*。


  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1382077259321577644)** (31 messages🔥): 

> `ChatGPT docx files, Custom GPT advantages, O3 Pro coding issues, Translating Novels` 


- **ChatGPT 诱导用户使用不存在的 DOCX 功能**：成员们抱怨 **ChatGPT** 诱导用户使用 **docx files** 功能，但该功能实际上并不存在，且该功能已**禁用超过一个月**。
   - 一位成员表示这令人失望，尤其是当你尝试汇总数据时，因为 ChatGPT 甚至记不住十秒钟前的内容。
- **Custom GPTs 展现优势**：一位成员建议创建 **custom GPTs** 以获得更好的结果。
   - 他指出，根据模型版本或是否拥有订阅，用户可以获得相应的收益。
- **O3 Pro 的编程能力不如 O3？**：一位成员想知道 **OpenAI** 发生了什么，并指出发布的 **O3 Pro** 在编程方面甚至不如 **O3**，且准确性不高。
   - 另一位成员回应称，根据他们目前的测试，这是一个非常出色的模型；而第一位成员则坚持认为 **Claude 4** 要好上 1000%。
- **埃及阿拉伯语小说翻译工具对比**：一位成员询问将小说从**埃及阿拉伯语翻译成英语**的最佳方法以及应使用哪种模型。
   - 成员们建议 **ChatGPT** 并不是合适的工具，有专门的工具可以做得更好，并建议尝试 **DeepL**。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1382176207784644680)** (35 messages🔥): 

> `AI system reliability, Config file for LLMs, Banning CP in AI, Prompt security, AI's moral values` 


- **用户分享用于构建可靠 AI 系统的配置文件**：一位用户分享了一个[配置文件](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161)，旨在使 **LLMs 更加可靠**，具有更低的幻觉（hallucination）、事实锚定（truth anchoring）以及对不可能结果的感知能力。
   - 该配置旨在供包括 GPT 在内的任何 LLM 平台使用，以建立用户记忆并获得相当可靠的 AI 系统。
- **关于禁止 CP 内容的辩论**：一位用户质疑在配置文件中明确提到禁止 **CP**（儿童色情）的必要性和妥当性。
   - 另一位用户回应称，虽然他们同意安全性至关重要，但不确定这样做是否合适，因为*在配置末尾列举禁止的 token 会放大近因偏差（recency bias），并增加 LLM 泄露的风险*。
- **安全措施面临审查**：配置文件中的安全措施引发了关注，特别是包含炸弹制作和生物武器等**违禁话题**。
   - 一位用户认为，*只需一行代码——在正确的位置放置正确的禁止 token——就会产生灾难性的泄露风险*，并强调风险是乘数效应而非加法效应。
- **AI 测试框架**：一位用户询问如何测试 **A.I.** 以验证其**自我管理道德价值观**的说法是否属实。
   - 另一位用户回应称，由于人类输入的多样性，需要进行大规模测试。
- **AI 回复准确率阈值**：通过**事实验证（Truth validation）**和使用正题（Thesis）、反题（Antithesis）与合题（Synthesis）的**推理（Reasoning）**来验证 AI 的回复准确性。
   - 如果回复在第二个循环后低于 **95% 的准确率**阈值，AI 必须澄清该回复是推断出来的，或者在无法证实的情况下声明其不知道。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1382176207784644680)** (35 条消息🔥): 

> `AI Configuration, Prompt Injection, LLM safety, Adversarial Testing, Negative constraints` 


- **配置文件引发安全讨论**：一名成员分享了一个旨在增强 AI 系统可靠性的[配置文件](https://cdn.discordapp.com/attachments/1046317269069864970/1382207370427367574/config.txt?ex=684af92d&is=6849a7ad&hm=18e8864d5d0c91788c8e3e971f760cc507a8f144c51d09bc0da1d95285f30161&)，但这引发了关于提及明确负面约束（如 "CP is banned"）的争论。
   - 一位成员认为，*在配置末尾列举禁用词会放大近因偏差 (recency bias)，并由于粉红大象效应 (pink elephant effect) 增加 LLM 泄露的风险*。
- **负面指令导致粉红大象效应**：有人担心指定明确的禁用内容会通过近因偏差增加 LLM 泄露的风险，可能导致 LLM 生成其本应避免的内容。
   - 建议避免直接引用非法内容，而是使用通用的指令，例如 *The model must not generate or assist with illegal content, including but not limited to dangerous, exploitative, or otherwise restricted subjects as defined by law and platform policy*。
- **需要大规模对抗性测试**：有人指出，框架需要面对对抗性 Prompt Injection，且*缺乏证据并不代表不存在证据*，特别是对于涌现的、低概率但高严重性的风险。
   - 安全工程的标准不是“它是否对我失效过”，而是“在对抗性或不可预见的条件下，它是否可能发生灾难性故障？”
- **正题、反题与合题 (Thesis, Antithesis and Synthesis) 验证周期**：一位成员指出，AI 的每条回复都会经过一个**事实验证 (Truth validation)** 周期，包括**正题**、**反题**和**合题**，并根据周期的准确性进行重复。
   - 如果回复在第二次仍然低于 **95% 准确率**，AI 必须明确补充说明该回复是推断出来的，或者在该部分仍无法证明时表示不知道。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 条消息): 

Readybot.io: **OpenRouter - New Models**
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1382078105212162058)** (329 条消息🔥🔥): 

> `OpenRouter OpenAI Default Reasoning Mode, O3 Pro Pricing, Assistance Needed with OpenRouter Code, LLM Recommendations for Research, OpenRouter Rate Limits` 


- **OpenRouter 推理模式默认设置**：一位用户询问了通过 OpenRouter 进行 OpenAI 请求时使用的默认推理模式，特别是它是默认为 *detailed* 还是 *auto*。
   - 在提供的上下文中，该问题未得到解答。
- **Sama 和 Noam 推文提及 O3 Pro 定价**：用户对 **O3 Pro** 的定价表示关注，一位用户形容 **O1 Pro** 的定价非常*离谱 (bonkers)*。
   - 据报道，关于 **O3** 定价的信息来自 **Sama** 和 **Noam Brown** 的推文。
- **医疗专业人员请求代码帮助**：一位具有医学背景的用户请求协助编写 **OpenRouter** 代码，以便将 **279 个 Prompt** 输入到各种 **LLM** 中进行医疗共识项目，并分享了代码的 [pastebin 链接](https://pastes.dev/UqlfuJgF2W)。
   - 其他用户建议使用独特的 Persona Prompt、速率限制 (rate limiting) 并验证 CSV 数据以确保问题不为空，并建议 **LLM 输出 JSON** 以方便解析。
- **LLM 专家组讨论共识研究的模型选择**：用户讨论了为共识研究选择 **LLM** 的事宜，建议包括 **Gemini**、**Claude** 和 **Sonar** (Perplexity)，同时因性能差距排除 **OpenAI**、**DeepSeek** 和 **Grok**。
   - 有人提到，为了公平比较，应使用非推理模型，且 Qwen 需要设置为非推理模式；此外还提到 [Grok 的上下文记忆力较低](https://x.com/OpenRouterAI/status/1932828433148776650) 且*非常愚蠢*。
- **OpenRouter 拥有近乎无限的 TPM 速率**：一位用户询问了 **OpenAI** 施加的 **TPM 速率限制 (rate limits)** 是否适用于 **OpenRouter**，特别是在不使用个人 OpenAI Key 的情况下。
   - 澄清指出 **OpenRouter** 在实践中具有非常高的限制，这意味着对用户来说是*无限 TPM*，且[速率限制确实根据所使用的模型而适用](https://openrouter.ai/models?fmt=cards&supported_parameters=structured_outputs)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1382074957856768113)** (231 条消息🔥🔥): 

> `微调说明，Unsloth 的 4090 vs 5090，GRPO vs DAPO，Magistral 量化版发布，DeepSeek R1 8Q 模型` 


- **Unsloth 文档解释微调**：一名成员询问了 LLM 微调的解释，另一名成员分享了面向初学者的 [Unsloth 微调指南](https://docs.unsloth.ai/get-started/fine-tuning-guide)。
- **5090 vs 4090 的价格令人恼火**：成员们讨论了 **4090** 价格的荒谬，由于目前的价格，一些人宁愿等待 **5090**，而另一些人则反映二手 **4090** 价格过高。
- **GRPO 还是 DAPO？这重要吗？**：成员们讨论了 Unsloth 正在使用 **DAPO** (Data-Aware Prefix Optimization) 但将其称为 **GRPO** (Grouped Relative Positional Encoding)，一名成员表示它们*老实说非常接近*。
   - 另一名成员提到，他们*在幕后与 Mistral 合作以确保一切正确*。
- **Magistral 量化版现已推出**：UnslothAI 团队宣布发布 **Magistral 量化版**，并链接到了[他们的推文](https://x.com/UnslothAI/status/1932441885618147402)。
   - 他们还提到团队*在幕后与 Mistral 合作以确保一切正确*。
- **DeepSeek R1 8Q 基准测试表现惊人**：成员们对 **DeepSeek-R1-0528-UD-Q3_K_XL** 模型的性能感到震惊，其中一人报告称它在基准测试中保持了接近 **80%** 的水平。
   - 一位团队成员指出测试模型是几天前的（6 月 8 日），随后分享了一个修正后的[链接](https://huggingface.co/TheBloke/DeepSeek-R1-0528-UD-Q3_K_XL-GGUF)。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1382076254965989386)** (11 条消息🔥): 

> `Elise-lora, NoisySpeechDetection-v0.2, 运行时间` 


- **Elise-lora 用法**：一名成员询问是将 `model_name="Etherll/Elise-lora"` 设置为 `unsloth/orpheus-3b-0.1-ft`，还是先进行正常的 SFT 并保存 LoRA 适配器。
   - 另一名成员建议先进行 LoRA，然后再进行 SFT。
- **噪声语音分类器发布**：一名成员宣布发布 [NoisySpeechDetection-v0.2](https://huggingface.co/Etherll/NoisySpeechDetection-v0.2)，这是一个基于 Whisper Small 并使用 Unsloth 训练的音频分类器。
- **质疑平台运行时间**：一名成员对某个平台的真实性和运行时间提出质疑，并附带了一张[截图](https://cdn.discordapp.com/attachments/1179039861576056922/1382521252853584004/Screenshot_20250611-204515.png?ex=684b74c0&is=684a2340&hm=e045135237e6dd500da023be19faa4d4e06209980790a7e68e2cc55bf53c086d&)。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1382072415382933584)** (63 条消息🔥🔥): 

> `Ollama 对 safetensors 的集成、SSD 对大型量化模型的影响、Unsloth 模型与普通模型的对比、Gemma3 模型系列、多语言微调` 


- **Ollama 现已支持 Safetensors，保存你的合并模型**：社区用户反馈 `save_to_gguf` 仍在大纲计划中且尚不可用，但建议将合并后的模型保存为 **safetensors** 格式，然后使用 [ollama 的 create --quantize 命令](https://ollama.com/docs/guides/how-to-create-modify-prompt#quantize-your-model) 将其转换为 Ollama 兼容格式。
- **运行量化模型时 SSD 寿命受到质疑**：成员询问运行超过 RAM/VRAM 限制的 R1 量化模型是否会损耗 **SSD**，以及是否有简单的方法将工作负载分配到多个 **SSD** 以提高带宽。
   - 一名成员回复称，从 **SSD** 进行流式传输应该不会造成太大损害，因为主要是读取操作，但 *R1 在 SSD offload（卸载）方面的表现不如 **Qwen3/Maverick/Scout***。
- **Unsloth 模型与普通模型的对比**：一位用户询问“普通”模型与来自 **Unsloth** 的模型有何区别，特别是 `Qwen/Qwen2.5-VL-7B-Instruct` 与 `unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit` 之间。
   - 另一位成员解释说，**Unsloth** 模型是预量化的且体积更小，而普通模型是完整权重，仅在设置 `load_in_4bit=True` 时才进行实时量化。
- **Gemma3 系列模型的问题正在修复中**：Unsloth 团队成员指出 **gemma3** 系列模型存在一些问题，目前正在修复。
   - 如果你急于使用，请查看其他模型，如 **Qwen**、**Llama** 或 **Mistral**。
- **寻求多语言微调建议**：一位成员就多语言微调问题寻求建议，其中 **orpheus-3b** 模型被微调以支持哈萨克语，但遗忘了情感标记（emotion tokens）和说话人标记（speaker tokens）。
   - 另一位成员建议针对新语言进行预训练，并指出数据集需要包含情感信息，如果预训练或分布中没有这些内容，模型就不会掌握。


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1382075323381977148)** (12 条消息🔥): 

> `AIME 2025、Prompt 调整、GRPO 效率、打乱答案顺序、LoRA 与 FFT 对比` 


- **AIME 2025 数学竞赛现已发布**：**AIME 2025** 数学竞赛题目已公开；更多信息请访问 [ArtofProblemSolving](https://artofproblemsolving.com/wiki/index.php/2025_AIME_I)。
- **微小的 Prompt 调整带来巨大的准确率提升**：一位成员分享了一篇两年前的酷炫论文，探讨了 Prompt 的微小变化如何大幅改变零样本（0-shot）任务的准确率，详见此[论文](https://arxiv.org/pdf/2311.01967)。
- **打乱答案顺序难倒模型**：当 MMLU 问题以打乱的答案顺序呈现时，模型无法正确回答许多之前能答对的问题，详见此[论文](https://arxiv.org/pdf/2406.19470)。
- **引发关于 GRPO 更新幅度的讨论**：一位成员询问 GRPO+LoRA 是否效率低下，因为参数变化不大。
   - 另一位成员指出，预训练中也存在同样的固有稀疏性。[这篇论文](https://arxiv.org/abs/1803.03635)表明这并非 GRPO 或广义强化学习（RL）所特有。
- **LoRA 在 RL 对决中表现良好**：你提到的研究确实涵盖了 LoRA 在 RL 方面的表现实际上与 FFT（全参数微调）相当，如[这篇论文](https://arxiv.org/pdf/2505.11711)所述。
   - 相反，该论文主张他们的稀疏更新方法（仅更新有意义的参数）优于 FFT 和 LoRA。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1382089626751205457)** (217 条消息🔥🔥): 

> `带有推理功能的 Sonnet 4，GRPO 长度偏差，ProRL 论文，Mistral 的推理模式，IQ 被质疑` 


- **Sonnet 4 推理功能消失**：带有推理测试的 **Sonnet 4** 从测试中消失了，这表明在 **OpenRouter** 中无法使用其 *thinking* 功能。
- **GRPO 的长度偏差修复了？**：关于 **GRPO** 的讨论显示它存在长度偏差，这可能会通过 **DRGRPO** 或 **DAPO** 等修复方案来解决，内部训练环境使用了长度惩罚，详见[此 GitHub 链接](https://github.com/NousResearch/atropos/blob/main/environments/tool_calling_server.py#L370C1-L397C71)。
- **ProRL 效果辩论升温**：针对 **ProRL** 论文（[arXiv 链接](https://arxiv.org/abs/2505.24864)）及其在大型模型上的适用性展开了讨论，一些人质疑其结论，但也有人认同熵崩溃（entropy collapse）问题。
- **Mistral 发布 Magistral Reasoning，但仍有瑕疵**：新的 **Mistral** 推理模式通过其聊天模板（[HuggingFace 链接](https://huggingface.co/mistralai/Magistral-Small-2506)）激活，*thinking* 以标签形式格式化，但存在模型因奖励作弊（reward hacking）而循环输出最终答案等问题。
- **苹果 Tim Cooke 抨击推理 LLM；IQ 分数就是笑话**：继 **Tim Cooke** 发布贬低 *推理 LLM* 的公告后，一名成员指出 **Yann LeCun** 最近也表示，无论 LLM 在做什么，与人类的行为方式仍有很大不同。


  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1382395360076501122)** (5 条消息): 

> `知识蒸馏，曝光偏差，Obsidian Loom，Obsidian 错误 402` 


- **知识蒸馏讨论开始**：一位成员正在基于 **student -> critic -> teacher** 流程进行蒸馏，存储 teacher 的 logits 并使用它们来微调 student 模型。
   - 他们正考虑将 student 模型升级为推理模型，并询问是否有人在推理模型上使用 **KD**；由于大多数前沿推理模型会隐藏推理 logits 和 token，他们认为如果切换到推理模型，可能需要放弃这一点。
- **知识蒸馏中出现曝光偏差担忧**：同一位成员认为，与 student 生成的内容相比，teacher 生成的推理模式会带来**曝光偏差（exposure bias）**问题。
   - 他们不确定跨推理模型进行蒸馏的最佳方法。
- **Obsidian Loom 连接受质疑**：一位新成员询问如何连接 **Obsidian Loom**。
   - 他们表示有朋友告诉他们 **402 错误**意味着必须重新安装 Obsidian。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1382289732653813800)** (1 条消息): 

> `McCulloch-Pitts 模型，AI 基础，计算神经科学` 


- **McCulloch-Pitts：AI 与神经科学的基石**：[McCulloch-Pitts 论文](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf)被认为是 **AI** 和**计算神经科学**的奠基之作。
   - 论文中提出的模型是**神经元**和**神经网络**的简化模型，能够计算任何**图灵可计算函数（Turing-computable function）**。
- **图灵可计算性与神经网络**：McCulloch-Pitts 模型证明了**神经网络**在理论上具备计算**图灵机**可以计算的任何函数的能力。
   - 这在**人工神经网络**与计算理论之间建立了根本联系，影响了这两个领域随后的发展。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1382072016089252011)** (8 条消息🔥): 

> `Frutiger Aero，AI Slop iMessage 背景，AI 心脏监测，Le Chat 曝光` 


- **Frutiger Aero 回归活动**：成员们表示希望将资源集中在带回 **Frutiger Aero** 风格和实现 **AI Slop iMessage 背景**上，而不是 **AI 心脏监测**技术。
   - 一位成员表示“没人关心能救命或延寿的 AI 心脏监测”。
- **Teknium 助力 Le Chat**：一位成员分享了 [Teknium 的推文](https://x.com/teknium1/status/1932677245962788982?s=46)链接，以庆祝 **Le Chat** 获得曝光。
   - 另一位成员简单地表示他们知道这条推文的来源。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1382289732653813800)** (1 messages): 

> `McCulloch-Pitts Model, AI Foundations, Computational Neuroscience` 


- **McCulloch-Pitts：发现 AI 奠基性论文**：一名成员分享了 Warren McCulloch 和 Walter Pitts 的论文链接：[*A Logical Calculus of the Ideas Immanent in Nervous Activity*](https://www.cs.cmu.edu/~epxing/Class/10715/reading/McCulloch.and.Pitts.pdf)。
   - 这篇论文被认为是 **AI** 和 **计算神经科学** 的**奠基之作**之一。
- **逻辑演算**：论文解释道，*给定神经系统任何部分的活动，都有可能在逻辑上推导出该活动的明确描述*。
   - *或者，如果该活动未被描述，则可以为其任何一个充分刺激推导出等效表达式*。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1382109921134510100)** (82 messages🔥🔥): 

> `LM Studio API Documentation, Connect LM Studio to External Ollama Instance, AI Autonomy and Decision Making, Local Image Generation Recommendations, CUDA 12.8 Runtime Speed on 4070ti` 


- **API 文档查找引导**：一位用户寻求 LM Studio Server API 的 **Swagger 文档**，但被引导至[官方 API 文档](https://lmstudio.ai/docs/app/api)以及 LM Studio 内的 **Developer Tab**。
   - 该用户正试图构建 http 请求以便与 LMS 服务器协作。
- **无法编排 Ollama**：一位用户询问如何将 LM Studio 连接到外部 **Ollama** 实例，但被告知 **LM Studio** 仅作为服务器运行。
   - 建议他们使用 **Open WebUI** 来实现此类连接。
- **图像生成精品指南**：一位拥有 **4070ti Super** 的用户征求本地图像生成的建议，得到的建议是先从 **Fooocus** 开始，然后探索 **ComfyUI**，并可能使用 **Pinokio** 来安装单点解决方案。
   - 其他建议包括使用量化的 **Flux.dev** 模型，并探索 **civitai** 上的微调模型，且由于 **sd-forge webui** 的学习曲线比 ComfyUI 更平缓，因此更推荐前者。
- **发现 Flash Attention 缺陷**：一位用户发现 **deepseek-r1-0528-qwen3-8b@q6_k** 模型在使用 **flash attention** 时出现问题。
   - 据指出，许多用户在使用 flash attention 时会遇到 bug，使用 **Q4 KV caches** 时也是如此。
- **关于 SSD Swap 损耗的讨论**：一位用户计划使用 `mmap()` swap 在一台拥有 **32+16GB** 内存的机器上运行模型，这引发了关于该方法可能损坏 **SSD** 的警告。
   - 尽管有警告，该用户坚称 **SSD** 的寿命是以写入的 TB 数（TBW）而非读取量来衡量的，并表达了对 *"llama.cpp 的信心"*。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1382090947780808948)** (131 messages🔥🔥): 

> `Digits 内存带宽对比 M3 Max，Evo X2 上的 Qwen 3 235b，Rednote dots LLM，八通道 RAM 服务器，双路 AMD Turin` 


- **Digits 可能因内存带宽而慢于 M3 Max**：预计 **Digits** 会因为内存带宽低于 **M3 Max** 而表现较慢，引用了一篇 [Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1he2v2n/speed_test_llama3370b_on_2xrtx3090_vs_m3max_64gb/) 作为证据。
   - Prompt 处理速度被认为是由算力驱动的，而 Token 生成速度则是由内存带宽驱动的。
- **探讨 Qwen 3 235b 在 Evo X2 上的可行性**：一位用户询问是否能在 **Evo X2** 上运行 **Qwen 3 235b Unsloth**，另一位用户回答说 **Q3_K_XL** 量化是能装下的最大版本，速度可达 **10-12t/s**。
   - 该设置涉及使用 Linux、10000 的上下文长度、KV cache 的 Q8 量化、不开启 mmapping，以及 320 的 Evolution Batch Size。
- **Rednote dots LLM 宣称速度翻倍**：一位用户提到 **Rednote dots LLM**，称如果使用适当的 *llama.cpp* 分支构建，其速度应约为原来的两倍。
   - 另一位用户确认合并（merge）最近已经发生，但还有一位用户表示怀疑，认为需要检查原始 Issue。
- **关于高带宽纯 CPU 解决方案的辩论**：用户讨论了使用八通道快速 RAM 服务器配置，以 5 tokens/sec 的速度运行具有 **60K** Token 上下文窗口的 **150B** 参数模型的可行性。
   - 一位用户分享了 Intel Xeon Gold 5120 CPU 搭配 256GB RAM 的性能数据，评估速度仅约为 ~0.86 tokens per second，因此对实现预期性能持怀疑态度。
- **双路 AMD Turin 服务器**：提到一台 **双路 AMD Turin** 服务器将拥有 **1.2TB/sec** 的内存带宽和 **640 GB/sec** 的 PCI-e 带宽。
   - 这样一台配备 8 个双 B60 的机器将支持 **16 个 GPU 和 384GB VRAM**，功耗约为 3.5kW。一位用户链接到了拥有 20 个 x8 插槽的 Supermicro 主板 [Supermicro.com](https://www.supermicro.com/en/products/motherboard/h14dsg-o-cpu)。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1382074840499884121)** (157 messages🔥🔥): 

> `Gradio 应用反馈，用于 AI 的 AMD Radeon Pro SSG，在深度视频中插入球体，LLM 验证损失范围，Qwen 的蒸馏模型` 


- **Gradio 应用用户反馈请求**：一名成员为其已部署的 **Gradio 应用** 寻求 **用户反馈**，以识别哪些地方让用户反感，以及哪些改进效果良好。
- **讨论 Mini PC RAM 限制**：成员们讨论了为什么 Mini PC 因空间问题和使用焊接的 **LPDDR5** 而被限制在 **128GB RAM**，并质疑为什么这些芯片不能焊接在内存条上以允许更大的 RAM。
- **使用 Rust 训练小型 LLM 引起关注**：一名成员询问是否有用于训练 **微型（100M 参数）LLM** 的 **Rust 端到端流水线**，寻求一个在所有训练阶段都具有可黑客性（hackable）的解决方案。
- **评估 LLM 验证损失**：一名正在训练约 **400M 参数文本生成模型** 的成员询问了合理的 **验证损失（validation loss）范围**，报告数值在 **3.5** 左右，并寻求该规模的标准基准。
   - 一位成员指出曲线形状基本良好，通常验证损失越低越好，但确保模型表现符合预期非常重要。
- **探索 ZeroGPU 计费异常**：一位用户报告称，尽管只进行了几次 API 调用，但仍被收取了 **ZeroGPU 使用费**，并质疑为什么刷新时间明显长于通常的 **15 分钟**。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 messages): 

stark_0278: 大家好，我是 Stark。刚开始学习 AI Agent 课程。
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1382411818802086030)** (1 messages): 

> `FuriosaAI, AI Hardware, Deep Learning, New Chip Designs` 


- **FuriosaAI 在 AI 硬件领域引起关注**：一位成员重点介绍了 [FuriosaAI](https://furiosa.ai/)，这是一家 AI 硬件公司，并指出了其在该领域的潜力和创新。
   - 评论认为 FuriosaAI 是*一个值得关注的对象*，暗示其在 AI 芯片设计和性能方面可能带来进展或颠覆。
- **专用 AI 加速器的新兴趋势**：讨论简要触及了对专用 AI 加速器日益增长的兴趣，例如 FuriosaAI 开发的加速器，这些加速器是为特定的 Deep Learning 工作负载量身定制的。
   - 这反映了行业向优化 AI 硬件发展的更广泛趋势，即超越通用 GPU 以实现更高的效率和性能。


  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1382126552594124821)** (11 messages🔥): 

> `LLM Agent Framework, HyBio-Agent on Hugging Face Spaces, Qwen 3 1.7b Model Testing, Conversational Agent for PostgreSQL` 


- ****LLM Agent 框架**开源了！**：一位成员开源了他们的 **LLM Agent 框架**，[可在 GitHub 上获取](https://github.com/starsnatched/llm-backend)，该框架可以使用具有完全访问权限的 **VM 上的 Linux 终端**，使其能够存储、修改、删除和移动文件，并从网络上收集信息。
   - 作者强调，*我们永远无法像这样“解决历史”并实现量子证明的真相……*
- ****HyBio-Agent** 在 Hugging Face Spaces 亮相**：一位成员分享了 [Hugging Face Spaces](https://huggingface.co/spaces/Agents-MCP-Hackathon/HyBio-Agent) 上 **HyBio-Agent** 的链接。
   - 另一位成员惊呼：*没门，AGI 实现了！*
- ****Qwen 3 1.7b 模型**表现出奇地好！**：一位成员分享说，他们测试了低至 **Qwen 3 1.7b** 的模型来制作对话式 Agent。
   - 另一位成员询问该尺寸的模型是否运行良好，并表示有兴趣使用他们的函数来微调模型。
- ****SQlord**，PostgreSQL 的对话式 Agent！**：一位成员制作了一个对话式 Agent 来探索任何 **PostgreSQL 数据库**，支持 **英语和葡萄牙语** 测试，并已在 [Hugging Face](https://huggingface.co/spaces/Agents-MCP-Hackathon/SQlord) 上线。
   - 它也使用了 **Qwen 3 1.7b 模型**进行测试。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1382343355190480947)** (2 messages): 

> `Reading Group Scheduling, Paper Presentations` 


- **读书会日程悬而未决**：一位成员询问下一次读书会何时举行，得到的回复是*目前没有安排*。
   - 然而，*欢迎任何人带头进行论文展示*。
- **鼓励论文展示**：据沟通，目前没有预定的读书会。
   - 鼓励所有成员展示论文。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1382081843469221920)** (40 messages🔥): 

> `Codeagent capabilities, Agents Course Deadlines, Local vs Colab, OpenAI Model Costs, MCP Protocol Deployment on HuggingFace` 


- **Codeagent 解决所有问题**：一位用户发现 **Codeagent 的 prompt** 是高度硬编码的，其核心思想似乎是 *“编写代码来解决你所有的问题。”*
- **Agents 课程截止日期临近**：课程截止日期是 **7 月 1 日**，但新成员可以在一天内完成第一个单元并获得证书。
- **课程中的本地编码 vs Colab**：成员们正在使用 VSCode 在本地运行代码或使用 Google Colab 空间，有人指出 **unit 4** 在本地编码时效果最好。
   - 一位成员正在寻找课程的 *requirements.txt* 文件，以便控制依赖项。
- **OpenAI 4o-mini 成本**：一位成员报告说，使用 GPT-4o-mini 模型在问题上获得约 **50 分** 的花费 **不到 10 美元**，但随后的邮件指出 **o3** 比 **4o** 更便宜。
   - 另一位成员提到，他们使用 **Gemini flash pro** 进行故障排除仅花费了 **0.20 美元**，重点放在前 **10 个问题**上，且没有进行图像分析。
- **讲座中的 MCP 协议部署**：一位成员分享了关于 **在 Hugging Face 上部署 MCP 协议** 的讲座，展示了如何添加描述性 doc strings 以及向 *app.py* 添加代码片段。


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1382171549095956510)** (36 条消息🔥): 

> `Reservoir Computing, Wumpus World Module, Continual Learning, DAN Agent, Oscar-C project` 


- ****Reservoir Computing** 凭借 **Self-Organizing Reservoirs** 再次浮现**: 一位成员询问了关于具有 **self-organizing reservoirs** 的 [reservoir computers](https://ollama.com/PythagoraTheorem/Aimee3) 的信息。
   - 该建议强调了其与大脑功能的相似性，激发了人们对替代计算方法的兴趣。
- ****Wumpus World** 实现面临资金**困境****: 一位成员报告了为一款 **Unity 游戏**开发 **Wumpus World 模块**的进展，但目前正面临实现问题以及必要 **LoRA 训练**的资金短缺。
   - 他们分享了一个 [有趣的小 Agent 链接](https://drive.google.com/file/d/1cTb5g68ivazmx5iVBrwtvAaTg7I-97nm/view?usp=sharing)，并希望它能作为 **NPC** 运行。
- ****Continual Learning** 的 **DL** 怀疑论**: 一位成员对 **Deep Learning** 实现 **continual learning** 的能力表示怀疑，理由是由于 i.i.d. 假设导致的固有内存限制。
   - 他们主张探索解决这些局限性的替代方法，尽管其复杂性较高，并指向了一个关于解决 Deep Learning 问题的 [YouTube 视频](https://youtu.be/AT3Tfc3Um20?si=zL_m5lrW5Yu2O6IW)。
- ****DAN Agent** 首次亮相**: 一位成员分享了他们的 **DAN (Do Anything Now) Agent**，它可以根据单个提示词生成图像、视频、故事和旁白，并提供了一个 [在 Ollama 链接中的 Agent 截图](https://ollama.com/PythagoraTheorem/Aimee3)。
   - 该 Agent 具有带记忆的对话模式、剧本改进扩展和终端操作功能，旨在通过 API 就绪的扩展模板实现社区驱动。
- ****Oscar-C 项目**寻求支持者**: 一位成员邀请其他人查看他们的项目 **Oscar-C**，该项目涉及认知架构、XAI 和 neurosymbolic AI，但提到之前的分享尝试被贴上了 shitposting 的标签。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1382103577551441940)** (22 条消息🔥): 

> `BioML people in Berlin, How much do language models memorize?, Agents and World Models, Energy-Based Models, Predictive Coding and Active Inference` 


- **BioML 专家齐聚柏林**: 一些在柏林的 **bioML** 研究人员未来可能会在 YK Discord 进行展示。
- **GPT Grokking 程度评估**: 一篇新[论文](https://arxiv.org/pdf/2505.24832)提出了一种估算模型对某个数据点“了解”程度的新方法，用于衡量现代语言模型的容量。
   - 测量结果估计 **GPT 系列**模型的容量约为 **3.6 bits-per-parameter**，并观察到模型会持续记忆直到容量填满，此时 *“grokking”* 开始发生。
- **更广泛的泛化是否需要 World Models？**: 一篇[论文](https://arxiv.org/abs/2506.01622)认为，**任何能够泛化到多步目标导向任务的 Agent 必须已经学习了其环境的预测模型**。
   - 作者写道，*这种模型可以从 Agent 的 policy 中提取出来，并且提高 Agent 的性能或其能实现的目标复杂度，需要学习日益精确的 world models。*
- **Energy-Based 模型探索**: 一位成员分享了一篇关于 **energy-based models** 的[论文](https://arxiv.org/abs/2406.07726)，但表示尚未仔细研究。
- **推理见解启发**: 一位成员建议[这篇综述](https://arxiv.org/abs/2407.04117)是了解 **energy based models** 的最佳入门，可以将其视为*一种局部梯度下降操作。*
   - 他们指出，*“Inference Learning”一词听起来可能与“Active Inference”有点相似*，但两者是不同的。如果你阅读了那篇综述和原始的 VAE 论文，你实际上就掌握了实现 Active Inference 模型所需的几乎所有工具。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1382148103326535772)** (16 messages🔥): 

> `Diffusion Models, GANs for Language Modeling, V-JEPA, Mistral Compute` 


- **Sama 的意外公告暗示 Diffusion Model**：继 [Sam Altman 的推文](https://x.com/sama/status/1932573231199707168) 提到一项意外公告后，成员们推测这可能涉及 **diffusion model**。
- **Gemini Diffusion 展现潜力**：一位拥有 **Gemini diffusion** 访问权限的成员指出，尽管其速度和模型大小显而易见，但表现非常出色，尤其是在快速寻找解决方案方面。
   - 提到它 *不擅长创意任务*，因为 *它比 transformers 更多地遵循模式*。
- **重新审视用于 Language Modeling 的 GANs**：一位成员分享了一个 Markdown 文件 [Papers_or_preprints_that_did_Language_Models_with.md](https://cdn.discordapp.com/attachments/853983317044756510/1382377584989573200/Papers_or_preprints_that_did_Language_Models_with.md?ex=684aeef3&is=68499d73&hm=103bb35e5751b57b677f9388cdfad9094bdfbb3e5e5f37be5fd649aad3eb0ad8&)，引发了关于使用 **GANs** 进行 Language Modeling 的讨论。
- **Meta 发布新版 V-JEPA**：Meta AI 发布了新版本的 **V-JEPA**，旨在推进世界模型基准测试，参考 [这篇博客文章](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) 和 [推文](https://x.com/lxrjl/status/1932499153596149875)。
- **Mistral 推出计算服务**：**Mistral AI** 宣布推出 **Mistral Compute**，旨在使 AI 基础设施民主化，并为更多人提供构建和拥有 AI 基础设施的工具，参考 [其博客文章](https://mistral.ai/news/mistral-compute)。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1382257728524779520)** (1 messages): 

> `Sparse Autoencoders, Product Key Memory Modules, PKM sparse coders` 


- **EleutherAI 发布 Product Key Memory Sparse Coders**：EleutherAI 发布了 [博客文章](https://blog.eleuther.ai/pkm-coders/)、[代码](https://github.com/EleutherAI/sparsify/tree/e2e-pkm) 和 [权重 (checkpoints)](https://huggingface.co/EleutherAI/pkm-coders/)，供研究人员实验用于 **sparse autoencoders** 和 **transcoders** 的 **Product Key Memory (PKM)** 模块。
   - 团队发现 **PKM** 加快了训练和推理速度，减少了内存占用，并诱导了分层分组，但他们不确定这些改进是否足以抵消增加的复杂性。
- **PKM 模块优势概述**：**Product Key Memory (PKM)** 模块加速了训练和推理，同时降低了内存占用。
   - 这些模块鼓励在 latents 中形成分层分组结构，可能简化模型的理解并提高效率。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1382071981180190761)** (38 messages🔥): 

> `Ban Data Archiving, O3 Pro Pricing, Ultrasonic Lens 3D Printing, Open Science @ CVPR` 


- **关于禁止数据归档的辩论**：成员们辩论了是否应该在 [封禁时删除消息](https://discord.com/channels/691289280227498005/729741769738158194/1252734090107158530) 并为数据集目的归档数据，关注点在于用户挫败感与开放研究益处之间的权衡。
   - 对话建议归档数据在发布前应进行 **匿名化 (anonymized)** 处理，以保护用户隐私，同时仍能支持 AI 和心理学研究。
- **O3 Pro 定价遭到批评**：成员们批评了 [O3 Pro 的定价](https://discord.com/channels/691289280227498005/729741769738158194/1252738340687548506)，指出其输入成本为 **$20 / 1M tokens**，输出成本为 **$80 / 1M tokens**。
   - 一些用户幽默地评论说，这个定价暗示它应该能 *解决黎曼猜想*，并认为它 *比 muon 还差*，*比 RWKV 还差*。
- **超声波透镜 3D 打印问题**：一位成员征求建议，以减轻在 **3D 打印球形或锥形声学透镜** 用于超声波束聚焦时，由脊线引起的散射。
   - 该成员预感这将是一个问题，其他成员建议联系材料科学部门寻求帮助。
- **CVPR 的开放科学交流**：如果有人在 **CVPR** 现场并想聊聊 **open science** 相关内容，请查看 [此链接](https://lu.ma/z1o7ncnt)。


  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1382241636775624755)** (28 messages🔥): 

> `Harvard Library's dataset, British Library / Library of Congress digitization, Cosine Decay` 


- **哈佛图书馆数据集指日可待**：一篇[论文](https://arxiv.org/abs/2506.08300)讨论了通过**哈佛图书馆在 Google Books 中的份额**使一套书籍可被访问，该数据集涵盖了约 **100 万本**经核实属于公共领域的书籍。
   - 成员们认为该数据集在很大程度上是全新的，并且已经热切期待了一年多，预计代码和数据将很快发布。
- **大英图书馆和国会图书馆应将所有书籍数字化**：成员们讨论了像**大英图书馆**和**国会图书馆**这样的机构，其藏书量大约是哈佛数据集的 **300 倍**，如果数字化，可能会为 LLM 提供 **10T 更多的高质量 token**。
   - 论文提到计划让公众可以访问数百万本更多的书籍，以用于各种用途。
- **Cosine Decay 深度探讨**：成员们正在寻找关于 **cosine decay** 衰减至最小值（例如峰值的 10%）与一路衰减至 0 的论文和直觉，质疑最小值是否有助于小型 SFT 运行中的泛化，并讨论了[这篇论文](https://arxiv.org/pdf/2502.15938)。
   - 一位成员建议对于小型 LLM，两个 epoch 的效果最好（第一个 epoch 采用 warmup 和线性衰减，第二个 epoch 采用 cosine decay 至 0），并引用了[这篇论文](https://arxiv.org/pdf/2404.06395)和[这项 Meta AI 研究](https://ai.meta.com/research/publications/v-jepa-2-self-supervised-video-models-enable-understanding-prediction-and-planning/)。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1382410438943183018)** (3 messages): 

> `knock off error control, AI safety, Interpretability` 


- **Knock Off Error Control 的想法再次浮现**：一位成员建议 **knock off error control** 的想法可能会有用，并分享了 [Knockoff Inference](https://arxiv.org/abs/1811.06687) 论文的链接。
   - 另一位成员表示感谢，称他们已经大约 **4 年**没看到有人提到 knockoffs 了，这提醒了他们应该去学习相关知识。
- **强调 AI Safety 的重要性**：讨论围绕在 AI 技术快速进步的背景下，对 **AI safety** 措施的迫切需求展开。
   - 成员们对潜在风险和意外后果表示担忧，强调了在 **AI safety** 协议方面进行研发的紧迫性。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1382072746091216896)** (54 messages🔥): 

> `aider uninstall, O3 Model Price Drop, OpenRouter KYC, Kingfall Model, R1 528 vs 0506` 


- **Aider 卸载波折**：一位用户询问了在使用 `pip install aider-install && aider-install` 后从 Linux 机器上卸载 Aider 的正确方法，发现 `pip uninstall aider-chat` 后二进制文件仍留在 `/local/bin` 中。
   - 该用户知道可以手动删除二进制文件、索引和缓存文件，但想知道是否有更好的方法，或者是否遗漏了什么。
- **O3 模型大幅降价，但仍需 KYC**：成员们注意到 **O3 模型** 经历了 **80% 的降价**，导致成本降至 **$2 输入** 和 **$8 输出**，但 [OpenRouter](https://openrouter.ai/) 仍然要求用户提供自己的 key 并进行 KYC。
   - 一位成员表示不满，说道：*“为什么 OpenRouter 没有 o3 flex？但使用它仍然需要 KYC。难受 (Sadge)”*。
- **Kingfall 模型性能对比出现**：一位用户对比了 **0605 (32k)** 与 **Kingfall (auto thinking)**，显示 **Kingfall** 表现好得多，看起来*他们尝试追求同样的目标，但 Kingfall 的效果要好得多*。
   - 然而，另一位用户对这些数据提出质疑，称其为*荒谬的数据*，并可能是一个 [OpenRouter bug](https://openrouter.ai/)。
- **R1 528 基准测试令人失望**：成员们讨论了新 **R1 模型** 的性能，一位成员认为新 **R1** 在 **$4.8** 的价位上更好，但另一位反驳称，他们认为新 **R1** 甚至不如 `0506`，但*它更便宜，所以从性价比来看可能是最好的*。
   - 一位成员对目前的基准测试表示怀疑，指出：*几乎所有的基准测试都是开源的，这意味着 AI 公司可以在基准测试集上进行训练*。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1382096352753877082)** (12 messages🔥): 

> `Aider Pro Max Subscription Usage, Aider LLM Leaderboard by Language, Deepseek-R1 Configuration with Aider, O3 with new pricing, Aider planning mode` 


- **Aider 用户用尽了 Pro Max 订阅额度**：一位用户提到他们正在大量使用 **Pro Max 订阅**，甚至因为并行运行多个 **Claude 代码实例**而收到了使用量警告。
   - 该用户开玩笑地承认这确实是*有点自找的*。
- **Aider LLM 排行榜寻求按语言查看的功能**：一位用户询问是否可以按**特定语言**查看 [Aider LLM leaderboard](https://aider.chat/docs/leaderboards/)，以评估模型在特定语言下的表现。
   - 随后另一位用户分享了另一个 [LLM leaderboard](https://leaderboard.techfren.net/)。
- **Deepseek-R1 配置困难**：一位用户在 Aider 中通过 chutes 配置 **deepseek-r1** 时遇到困难。
   - 另一位用户建议设置 [.env file](https://aider.chat/docs/config/dotenv.html) 来进行配置。
- **Aider 中 O3 定价的挑战**：一位用户正在寻求关于在 Aider 中使用新定价结构的 **o3** 的建议，目前面临 **OpenAI API Tier 2** 访问权限和 **OpenRouter** 要求的问题。
   - 另一位用户提到 *OpenAI 在让你使用 `o3` 之前得先看你的护照*。
- **对 Aider 规划模式的渴望**：一位用户建议实现一个 **/planning mode** 来增强 Aider 的自动化能力，将其构想为一个任务主管（taskmaster）。
   - 另一位用户分享了 [OpenAI background guide](https://platform.openai.com/docs/guides/background) 的链接。


  

---


### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1382073660646359183)** (8 messages🔥): 

> `Conversational Style Tailoring, AI Overview Audio Generator, Pierceday Metalabel` 


- **定制对话风格**：一位用户询问如何针对特定的受众特征（如**年龄、种族和性别**）定制 AI 回复的对话风格，前提是已有相关的来源信息。
   - 他们请求其他人分享关于这一主题的评论和经验，表现出对**个性化 AI 交互**的兴趣。
- **AI 音频生成器，新手寻求帮助**：一位新用户请求协助，询问如何配置 AI 概览音频生成器，以便为源文件中的**每个主题生成单独的概览**。
   - 该用户提到难以找到说明中提到的 **customize**（自定义）选项。
- **用于手册和提醒的 Pierceday Metalabel**：一位用户分享了 [Pierceday Metalabel 链接](https://pierceday.metalabel.com/aphone)及其可能的应用场景。
   - 应用场景包括汽车手册、维护细节、电箱笔记和会议演示。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1382075973595435059)** (51 messages🔥): 

> `Podcast Generation Length, Customize Audio Overview, NotebookLM Limitations, Spreadsheet Support, Retrieve Old Output` 


- **揭秘播客生成秘籍**：受网上案例的启发，用户们正在探索如何使用 NotebookLM 生成**超过 20 分钟的播客**。
   - 一位用户指出在 App 中自定义音频概览存在困难。
- **NLM 缺乏对电子表格的支持**：用户们请求 NotebookLM 支持**电子表格** (.xlsx)，并指出虽然支持 Google Docs 和 Slides，但电子表格却奇怪地缺席了。
   - 进一步指出，.xlsx 文件在 Gemini 和 AI Studio 中是受支持的，但在 NotebookLM 中不受支持。
- **利用 NotebookLM 挖掘 D&D 战役的一致性**：一位用户正利用 NotebookLM 来保持其 **D&D 战役**的一致性，上传了书面内容和会议记录。
   - 他们对自动生成的音频概览印象深刻，但在指定会话范围的提示词准确性上遇到问题，想知道是否[有更好的方法](https://www.example.com)让 NotebookLM 只覆盖他们想要的内容。
- **确定 NLM 运行的 Gemini 模型**：成员们对 NotebookLM 运行在哪个 **Gemini 模型**上感到好奇。
   - 一些人认为是最新发布的 **2.5 Flash**，而另一些人指出只有 Gemini Pro 1.5 拥有 200万（2B）token 的上下文窗口。
- **LaTeXLM Chrome 扩展发布**：一位用户创建了一个[开源 Chrome 扩展](https://github.com/hachoj/LaTeXLM)，以便在 NotebookLM 上实现 **MathJax 渲染**。
   - 该扩展已在 GitHub 上可用，他们稍后可能会将其发布到 Chrome Web Store。


  

---

### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1382080361252651109)** (56 条消息🔥🔥): 

> `Cursor 对 Anthropic 的投资, DeepSeek 模型的叙事, Windsurf 的 'Plan Mode', Altman 的博客, 开放权重模型延迟` 


- **Cursor 可能加倍投入 Anthropic**: 鉴于 **Windsurf** 已归入 **OpenAI** 旗下，**Cursor** 可能会加倍投入 **Anthropic**，这引发了关于近期可能达成交易的猜测。
   - 成员们认为 **Cursor** 对 **Claude Code** 的投入比 **OpenAI** 对 **Codex** 的投入更多，一些用户指出 **Claude Code** 是第一个让他们感到“契合”的工作流。
- **传闻 Apple 将收购 Anthropic 以寻求 Siri 帮助**: 传闻和推测建议 **Apple** 应该收购 **Anthropic**，因为他们负担得起，而且“天知道他们在 **Siri** 上多么需要帮助”。
   - 一位成员指出，**Siri** 在领先 15 年的情况下，甚至无法可靠地发送一条短信。
- **Altman 在博客中预告“温和的奇点”**: 分享了 [Sam Altman 标题为 *The Gentle Singularity* 的博客文章](https://blog.samaltman.com/the-gentle-singularity) 链接，引发了关注和讨论。
   - 该链接与 [Kevin Hou 的 X 帖子](https://x.com/kevinhou22/status/1932516093333266538?s=46) 链接一同被分享。
- **Windsurf 推出用于任务管理的全新 'Plan Mode'**: **Windsurf** 推出了全新的 **'Plan Mode'** 功能，使 **AI agent** 能够通过创建和维护规划文档来执行复杂任务，该功能可在 [Windsurf.com](https://windsurf.com/) 免费使用。
   - 一位在小型绿地项目（新项目）上进行测试的用户反馈称其“运行良好”。
- **Sharon Zhou 加入 AMD 以推动 AI GPU 的民主化**: Sharon Zhou 宣布加入 **AMD**，与 **Lisa Su** 合作专注于 **AI 研究**和教学，并将 **LaminiAI** 的同事带到了 **AMD**。
   - 她的目标是**民主化 GPU** 并**扩展 AI**，并提到了她参加 **#AdvancingAI** 会议的情况，详见[此 X 帖子](https://x.com/realSharonZhou/status/1932817096510931380)。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1382088512794525826)** (46 条消息🔥): 

> `MCP 服务器集成, iframe 或前端 API, webio 传输, OAuth2 UI 身份验证, 无头 Agent (headless agents)` 


- **Hume AI 的 Evals 引起关注**: Hume AI 在一篇[博客文章](https://www.hume.ai/blog/roleplays-evals-hume-mcp-server)中发布了他们为 MCP 服务器进行“评估 (evals)”的方法，引发了关于其他人如何评估其系统的讨论。
- **MCP 服务器作为 iframe 获得支持**: 将 MCP 服务器作为 iframe 或前端 API 运行的想法正得到支持，讨论围绕使用类似于现有 stdio 传输的 **webio 传输**的潜在好处展开。
   - 这可能会解决当前的 MCP 问题，允许通过 `window.open` 实现自定义 UI 和 **OAuth 流程**，并通过 URL 复制粘贴简化设置，同时通过 Web API 管理虚拟文件系统，而不是授予 shell 访问权限。
- **OAuth2 UI 身份验证简化 MCP 集成**: 使用真实的 **OAuth2 UI** 进行身份验证将大大简化普通用户连接 Google 和 GitHub 等服务的过程，为 MCP 服务器提供更好的用户体验。
   - 理想情况下，OpenAI、Anthropic 和 Google 将为其 API 提供 OAuth2 登录，从而进一步简化流程。
- **反向代理增加安全问题**: 虽然编写反向代理可能是一项工程练习，但它可能会重新引入 iframe/webio 方法旨在解决的一些安全问题。
   - 本地服务可以接收来自浏览器的消息，并将其作为 MCP 服务器转发。
- **Hugging Face 加入 MCP 阵营！**: 截图显示 Hugging Face 现在已支持 **MCP**，这对从事模型开发的人员来说是个好消息。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1382079717787828346)** (5 条消息): 

> `Slides.com 的 MCP 服务器, Glama 故障排除, Glama 沙箱` 


- **MCP 服务器创建 Slides.com 演示文稿**: 一个用于创建 [slides.com 演示文稿](https://www.epicai.pro/use-ai-to-create-presentations-with-mcp-tsb4j)的托管 MCP 服务器已发布。
- **Glama 处于 "Testing" 状态的故障排除**: 一位成员报告在 **Glama** 中卡在 `testing` 状态一天，且没有输出任何日志，尽管在本地运行正常。
- **Glama 沙箱运行正常**: 该成员指出 **Glama 沙箱 (sandbox)** 可以连接本地且运行良好。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1382072011228057701)** (37 messages🔥): 

> `edu 邮箱, ReactNexus, Veo 3 成本, 项目构建竞赛, Manus Chat Mode` 


- **edu 邮箱与 ac.uk 域名**: 一位成员询问 *edu* 邮箱域名是否包含 *.UK* (*ac.uk*) 域名。
   - 另一位成员表示，根据截图显示，他没有看到 *UK* 域名被包含在内。
- **ReactNexus 将于 2025 年 7 月举行**: 一位成员询问是否有人会参加 **2025 年 7 月 3 日至 5 日**在 **J N Tata Auditorium** 举行的 **ReactNexus** 活动 ([https://reactnexus.com/](https://reactnexus.com/))。
   - 该会议专注于 React，这是一个用于构建用户界面的流行 JavaScript 库。
- **Veo 3 视频价格引发“价格冲击”**: 一位成员抱怨一段 **Veo3 视频** 需要消耗 **300 credits**，他们生成的 **38 个片段** 让他们花费巨大。
   - 另一位成员表示定价非常昂贵，并询问 **Veo3** 何时会对所有人开放。
- **Manus Chat Mode 上线**: Manus 为所有用户推出了 **免费且无限制的 Chat Mode**，使用户能够提出任何问题并获得即时回答。
   - 用户可以升级到 **Agent Mode** 以获得更高级的功能，例如创建综合性输出。
- **High Effort Mode 消失**: 几位用户报告称 **High Effort Mode** 从他们的 **Pro 账户** 中消失了。
   - 一位用户指出，他“*从不理解为什么 High Effort Mode 最初必须手动选择*”。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1382301980235661332)** (12 messages🔥): 

> `Tinygrad 操作目录, 出售 Tinybox Green 6x 4090, 一两年内实现 AGI, n8n 对比 Claude code, CS 学位` 


- **Tinygrad ops 移至独立目录**: Tinygrad 正在将 [操作（operations）](https://xl0.github.io/tinygrad-notes/bounty1.html) 移动到独立目录。
- **出售二手 Tinybox Green**: 一位成员正在出售一台来自数据中心、运行状况良好的二手 **Tinybox Green 6X 4090**，标价为原价的 **70%**。
   - 另一位成员表示了*兴趣*。
- **成员思考一两年内实现 AGI**: 一位成员询问关于在一两年内实现 **AGI** 的看法。
- **成员对比 n8n 与 Claude 构建 Agent**: 一位成员询问使用 **n8n** 与 **Claude code** 构建 Agent 的区别，想知道是否存在 n8n 无法做到而 Claude 可以做到的限制。
- **Discord 用户质疑 CS 学位的价值**: 一位成员询问 *CS 学位是否仍然有用*。
   - 另一位成员回答说 *如果你必须问这个问题，那它就没用*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1382211365984276540)** (24 messages🔥): 

> `微基准测试, lovely_XXX 作者, linalg.svd 悬赏, Jacobi 算法, Tensor.norm()` 


- **微基准测试（Micro Benchmarking）受到质疑**: 一位成员询问了关于微基准测试的最佳实践，并展示了一个使用 **Timing** 和 **Tensor.randperm** 的示例。
   - 另一位成员建议阅读聊天记录以获取相关经验。
- **lovely_XXX 作者**: 一位成员感谢 **lovely_XXX** 的作者在 Jupyter Notebook 方面提供的帮助。
   - 作者对此表示感谢。
- **SVD 悬赏**: 讨论涉及是否在 tinygrad 中手动编写函数以完成 **linalg.svd bounty**，从而确保 0 依赖。
   - LLM 建议 LAPACK 根据矩阵类型（通用、对称、Hermitian）使用不同的算法来计算特征值/特征向量，这可能也需要在 Tinygrad 中重新实现。
- **关于 SVD 特征值计算的讨论**: 针对 **linalg.svd bounty**，有人提议使用 Jacobi 算法来计算特征值。
   - 有人建议使用修改版的 Jacobi 算法：[A Novel Fully Hardware-Implemented SVD Solver Based on Ultra-Parallel BCV Jacobi Algorithm](https://cdn.discordapp.com/attachments/1070745817025106080/1382505351634616381/A_Novel_Fully_Hardware-Implemented_SVD_Solver_Based_on_Ultra-Parallel_BCV_Jacobi_Algorithm.pdf?ex=684b65f1&is=684a1471&hm=3eeaa1287761b9210d1e4a54b7c65b1be2a3c4b3838d55d14e60ca76d8cbefc7&)。
- **缺失的 Tensor.norm() 及 LLM 解决方案探索**: 一位成员询问是否存在 **Tensor.norm()**，并建议将 Discord 聊天记录/代码库输入到像 unblocked.com 这样的 LLM 中来回答问题。
   - 有人建议将周一早会的会议纪要也输入进去。


  

---

### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1382412757139197972)** (1 messages): 

> `AMD GPU 上的 Modular 平台，用于 GenAI 推理的 Mammoth，Python 中的 Mojo，来自 TensorWave 的免费 AMD 算力` 


- **Modular 助力 AMD GPU**：Modular Platform 现在已在 **AMD InstinctTM MI300X** 和 **MI325 GPU** 上正式商用，在预填充密集型（prefill-heavy）的 **BF16** 工作流中显示出高达 **53%** 的吞吐量提升。
   - 查看 [完整博客文章](https://www.modular.com/blog/modular-x-amd-unleashing-ai-performance-on-amd-gpus) 了解如何将顶级算力与开发者友好的软件相结合。
- **Mammoth 扩展 GenAI 推理**：**Mammoth** 是 Modular 新推出的 **Kubernetes 原生系统**，可在任何 GPU 上扩展 **GenAI 推理**，无需手动配置即可从单个容器在 **AMD** 和 **NVIDIA** 上部署 Hugging Face 模型。
   - 感兴趣的用户可以 [了解更多并加入公开预览版](https://www.modular.com/blog/introducing-mammoth-enterprise-scale-genai-deployments-made-simple) 以探索其功能。
- **Mojo 进军 Python 工作流**：Mojo 内核现在可以直接集成到 Python 工作流中，该功能已在 nightly 版本中提供，并由 **45 万多行开源 Mojo 内核代码**支持。
   - 开发者可以 [从这里开始](https://docs.modular.com/mojo/manual/python/mojo-from-python/) 在 Python 环境中使用 Mojo。
- **TensorWave 提供免费 AMD 算力**：得益于与 **TensorWave** 的合作伙伴关系，用户可以使用免费的 AMD 算力在真实工作负载中测试 Modular Platform。
   - 有意者可以访问 [Modular.com/tensorwave](https://www.modular.com/tensorwave) 获取此优惠以评估平台性能。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1382188455244726434)** (19 messages🔥): 

> `Mojo 交叉编译，在 Runpod 上进行 Mojo GPU 编程，Mojo 中的变长参数类型，Mojo 中的隐式变量声明，Mojo 文档差异` 


- **Mojo 跨平台交叉编译？**：一位成员发现 Mojo 目前不支持从 macOS 到 Linux 的直接跨平台静态编译，遇到了 *'apple-m1' is not a recognized processor* 错误。
   - 作为替代方案，他们正在探索捆绑依赖项和 `.mojo` 文件，以便在无服务器平台的 **Docker 容器**中运行。
- **Runpod-io 运行 Mojo GPU 代码！**：一位成员成功在 [runpod.io](https://runpod.io) 上实现了运行 Mojo GPU 代码的最小化实现，并报告称热执行（hot executions）速度很快，性能表现良好。
   - 唯一的障碍是大约 **10 秒的冷启动时间**，并计划在论坛上分享设置贴。
- **遗漏了变长参数类型映射？**：一位成员询问了 Mojo 中变长参数类型（variadic types）之间的映射问题，并在 [Modular 论坛](https://forum.modular.com/t/map-variadic-types/1638?u=emil.martens)上发起了一个话题进行讨论。
   - 虽然没有提到解决方案，但这是一个值得关注的开放领域。
- **Mojo 更新日志混淆：隐式还是显式？**：关于 Mojo 发布说明中有关隐式变量声明的一个潜在拼写错误引发了讨论，一些人将其理解为显式和隐式类型声明的混合。
   - 更新日志将进行澄清，以解决围绕变量是隐式还是显式声明的困惑，并强调 `var` 关键字的细微用法。
- **文档不一致：Stable 还是 Nightly？**：一位成员报告了在使用 Mojo 文档中关于生命周期（lifetimes）的代码示例时遇到的错误，特别是与 `__getitem__()` 中使用 `ref` 相关的问题。
   - 经澄清，文档可能反映的是 **nightly 版本**，可能与稳定版（stable release）不一致，建议该成员使用 nightly 版本以保证兼容性。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1382523524274454559)** (6 messages): 

> `峰值显存占用，Flex Attention，FSDP` 


- **Flex Attention 中的峰值显存占用**：一位用户询问是什么导致 `(bs, seqlen*8)` 输入的峰值显存占用高于 `(bs*8, seqlen)` 输入，例如在使用 **flex attention** 时，`(1, 64k)` 比 `(8, 8k)` 消耗更多显存。
   - 用户怀疑是 self-attention 中的 softmax 方阵导致的，但原以为 **flash/flex attention tiling** 机制会处理这个问题，不需要完全实例化。
- **Flex Attention 和 FSDP 调查**：一位用户正在调查 **flex attention** 和 **FSDP** 的峰值显存占用，以便在 `main` 分支上复现该问题。
   - 他们正在进行扫描测试（例如 8x1k, 4x2k, 2x4k 等），注意到峰值显存占用在达到一个“临界点”之前保持恒定，随后迅速飙升。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1382138079875432499)** (11 messages🔥): 

> `Qwen2, Tokenizer Integration, C4 Experiments, Iterable Datasets` 


- **Qwen2 需要检查**：一名成员将检查 **Qwen2** 以确认是否存在某个特定问题，但未说明该问题具体是什么。
   - 该成员表示怀疑这是否会产生重大影响，但无论如何都想修复它。
- **Tokenizer 集成改进即将到来**：一名成员将迭代 [#2574](https://github.com/pytorch/torchtune/pull/2574) 和 [#2794](https://github.com/pytorch/torchtune/pull/2794)，以改进新的 **HF tokenizer** 及其集成。
   - 该成员还计划提交一个 Pull Request 来修复 [#2809](https://github.com/pytorch/torchtune/pull/2809)，并已开始进行 **C4** 的一些实验。
- **Iterable Datasets 打包重构**：一名成员提议进行打包重构，以适配 Iterable Datasets，从而支持 **DPO**、**GRPO**、**multimodal** 等的打包，详见 [Proposal on packing refactor](https://github.com/pytorch/torchtune/pull/2819)。
   - 这些打包方面的更改似乎将使支持更广泛的模型配置变得更加容易。


  

---


### **Torchtune ▷ #[rl](https://discord.com/channels/1216353675241590815/1360680363885854841/1382420202465267782)** (3 messages): 

> `Nemo RL Plans, vllm-project` 


- **Nemo RL 计划曝光**：**Nemo RL** 的计划被[泄露](https://cdn.discordapp.com/attachments/1360680363885854841/1382420201823404032/AP1GczOQioexSd_ieqkppCKoVizt91prnymZ_uGi6mCeQdrSJE65osblAXMqxQw3030-h2272-s-no-gm.png?ex=684b16a4&is=6849c524&hm=8ca8961e205603c01114bb66f46acb3bb86d01b2d1297bef22f7817f5b6efeca&)。
   - 这些计划似乎来自 Databricks 会议。
- **vllm-project 正在进行中**：一个新的 [vllm-project](https://github.com/vllm-project/vllm/pull/18745) 正在开发中。
   - 这看起来是一个 Pull Request。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1382187179526062111)** (13 messages🔥): 

> `Cohere's pricing, n8n vs Claude code for building agents, Multi-modal re-ranker, CLIP and openCLIP` 


- **Cohere 的创意成本受到批评**：一名成员表示 [**Cohere**](https://cohere.com) 非常适合创意写作，但其**定价非常离谱**。
- **n8n 引导基于节点的 Agent 自动化**：成员们讨论了使用 **n8n 与 Claude code** 来构建 Agent，并指出 [Claude code](https://www.anthropic.com/product) 用于生成和理解代码，而 [n8n](https://n8n.io/) 用于制作工作流自动化。
   - 一名成员提到 **Cohere** 一直在开发自己的 Agent **North**，目前处于 Beta 阶段。
- **多模态 Re-Ranker 传闻被驳回**：一名成员询问了**多模态 Re-Ranker** 的发布情况以及推荐的图像重排序方法，但被告知 **Cohere** 目前不提供此类产品。
   - 一名成员建议使用带有结构化输出的 **GPT-4.1**，而另一名成员建议研究 **CLIP** 和 **openCLIP**。


  

---


### **Cohere ▷ #[📣-announcements](https://discord.com/channels/954421988141711382/996880279224451154/1382137971301548104)** (1 messages): 

> `Cohere North, EnsembleHP Partnership, AI Agents Platform, Healthcare Industry` 


- **Cohere 与 EnsembleHP 携手推进 North**：Cohere 正与 **EnsembleHP** 合作，将 **Cohere North** 引入医疗保健行业，旨在通过其安全的 **AI agents 平台** 减少行政摩擦并改善患者体验。
   - 更多详情可以在 [Cohere 博客](https://cohere.com/blog/ensemble-partnership)中找到。
- **AI Agents 巩固医疗保健合作伙伴关系**：通过与 **EnsembleHP** 的合作，Cohere 计划减少摩擦并提升患者体验。
   - 安全的 AI Agents 平台可能会减少医院的行政开销。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1382096191231098931)** (2 messages): 

> `Cohere API tiers, Reranking API latency` 


- **Cohere 选择不采用 API 分级**：一名用户询问 **Cohere** 是否有类似于 **OpenAI** 的 API 分级（Tiers），但被告知他们不提供分级。
   - 不过，一名成员提到他们可以提供其他解决方案，并建议用户联系 [carolyn@cohere.com](mailto:carolyn@cohere.com)。
- **Reranking API 延迟排查**：一名用户报告 Reranking API 存在 **2 秒的延迟**，并询问是否有潜在的改进空间。
   - 作为回应，一名成员建议发送邮件至 [carolyn@cohere.com](mailto:carolyn@cohere.com) 以寻求替代方案，这暗示在分级 API 访问之外可能存在潜在的优化或变通方法。


  

---

### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1382148761173885019)** (2 messages): 

> `Vitalops, Datatune` 


- **Datatune 创始人开源 Vitalops**：**Vitalops** 的联合创始人介绍了他们的开源工具 [Datatune](https://github.com/vitalops/datatune)，该工具可以使用纯自然语言执行数据转换。
   - 他们表达了加入社区并向成员学习的兴奋之情。
- **用户加入 Cohere 的 Discord 服务器**：一位用户在 Cohere Discord 服务器的欢迎消息中介绍了自己。
   - 他们很高兴能成为社区的一员。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1382396834130952253)** (1 messages): 

> `CleanlabAI, LlamaIndex` 


- **Cleanlab 与 LlamaIndex 联手**：[CleanlabAI](https://cleanlab.ai/) 和 [LlamaIndex](https://www.llamaindex.ai/) 已达成集成，旨在构建 AI 知识助手和生产级 Agent，从企业数据中生成见解并提高响应的可信度。
   - 双方合作可以为每个 **LLM response** 进行信任评分，并捕捉 [hallucinations](https://t.co/pTjn642OUO)（幻觉）。
- **LlamaIndex 宣布 Cleanlab 集成**：LlamaIndex [在 Twitter 上宣布](https://twitter.com/llama_index/status/1932837489238290941)了与 CleanlabAI 的新集成。
   - 该集成旨在增强 LlamaIndex 生成的 **LLM responses** 的可信度。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1382121274851196958)** (11 messages🔥): 

> `Chainlit Decommission, LlamaIndex + Chainlit, AI Security Webinar` 


- **Chainlit 停止维护：社区呼吁 LlamaIndex 接管代码！**：由于 [Chainlit](https://github.com/Chainlit/chainlit) 即将停止维护，用户正敦促 LlamaIndex 收购该项目，并强调了其在 LLM 生态系统中的重要性以及与 LlamaIndex 的无缝集成。
   - 一位成员指出，Chainlit 团队在支持 LlamaIndex 的每个版本方面非常配合，并强调 *LlamaIndex + Chainlit 的组合效果极佳！*
- **Chainlit 备受推崇：编程社区渴望持续贡献！**：用户因 Chainlit 的纯 Python 实现、在 Discord、Microsoft Teams、Slack 等平台的易部署性以及类似 ChatGPT 的 UI 而对其表示支持。
   - 正如一位成员所说，*Chainlit 就像 JavaScript，其编程方式全是事件监听器（装饰器函数）*，并称赞了其易用性：*我将 Chainlit 作为我所有生产级应用的前端层，以及我 Medium 文章中所有演示的前端。*
- **Hacken 的热修复：如何处理 AI 风险！**：Hacken 将于 **UTC 时间 6 月 12 日 13:00** 举办一场关于 **AI security** 的网络研讨会，探讨 **LLM vulnerabilities**（漏洞）和防御，由 Stephen Ajayi 主讲。
   - 感兴趣的人员可以通过 [Luma 链接](https://lu.ma/xl53xbfs)获取更多信息并注册。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1382339683580772413)** (2 messages): 

> `Gemini Fullstack Langgraph, DSPy Refactor, Agentic Patterns with DSPy` 


- **Gemini Fullstack Langgraph Quickstart 发布**：Google 最近发布了一个名为 [gemini-fullstack-langgraph-quickstart](https://github.com/google-gemini/gemini-fullstack-langgraph-quickstart) 的综合研究型应用的全栈实现。
   - 一位成员提到这是一个*非常出色*的实现。
- **DSPy 重构 Gemini LangGraph**：一位成员使用 **DSPy** 重构了 **Gemini** 代码中的 LangGraph 部分，并实现了一个简单的 **React** 前端，代码已在 [GitHub](https://github.com/pgurazada/deep-research.git) 上发布。
   - 重构后的工作流仅有 **200 行长** ([workflow.py](https://github.com/pgurazada/deep-research/blob/main/backend/agent/workflow.py))，并且*以更少的麻烦优雅地实现了原始的 LangGraph 工作流*。
- **DSPy 的 Agentic Pattern 威力**：一位成员使用 **DSPy** 实现了*非常多*的 **agentic patterns**。
   - 他们对这些 **primitives** 的强大功能感到*震惊*。


  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1382111367020023898)** (6 messages): 

> `DSPy 数据集创建工具，DSPy 与新推理模型，DSPy 3.0` 


- **DSPy 数据集开发工具**：一名成员询问了是否有工具可以轻松地为 **DSPy** 构建和导出数据集，以促进合成示例生成和手动标注。
   - 另一名成员建议自定义 **Streamlit app** 可能会很有效，像 **Cline** 这样的编程 **Agent** 可以在极少指导下协助创建。
- **推理模型与 DSPy 的兼容性**：一名成员询问了 **DSPy** 与在推理过程中使用 **tool-calling** 的新推理模型（如 **o3 / o3-pro / o4-mini**）的兼容性。
   - 他们指出，虽然 `dspy.ReACT` 存在，但它似乎是为 **chat API** 时代设计的，而不是为集成了 **tool-calling** 的 **responses API** 时代设计的。
- **DSPy 3.0 即将到来**：一名成员宣布了即将发布的 **DSPy 3.0** 版本，并链接到了 [DSPy 3.0.0b1 release tag](https://github.com/stanfordnlp/dspy/releases/tag/3.0.0b1)。
   - 他们询问是否有关于 **DSPy 3.0** 未来动向的全面概述。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1382083228298645605)** (4 messages): 

> `Python SDK 更新，Mistral 的 Magistral Small 支持` 


- **Python SDK 更新期待**：成员们对即将发布的 **Python SDK** 更新表示关注。
   - 未提供关于该更新的具体细节。
- **GPT4All 是否考虑支持 Magistral Small？**：一名成员询问 **GPT4All** 是否会支持 **Mistral** 的 **Magistral Small**。
   - 另一名成员建议使用 **JAN**、**LM-Studio**、**obadooga** 或 **koboldcpp** 作为替代方案，而原询问者表示他们会等待，并提到了对模型速度的担忧。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1382511149341212753)** (1 messages): 

> `AgentX 峰会，研究赛道，峰会注册` 


- **AgentX 峰会论文提交澄清**：一名成员询问向 **Research Track** 竞赛提交论文是否会自动进入 **AgentX 峰会征文与提案 (CFP)** 的考虑范围。
   - 他们寻求澄清是否需要为峰会进行单独提交。
- **AgentX 峰会入围者注册**：该成员还询问入围者是否需要注册峰会才能参加。
   - 他们担心门票可能会在比赛结果公布前售罄，如果未被选为入围者，可能会导致无法参加。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1382190233457529014)** (1 messages): 

> `Cerebras 技术演讲，阿里巴巴 Qwen 3 系列，AI 工作坊` 


- **Cerebras 举办技术演讲**：Cerebras 将于本周五（6 月 13 日）**12:00–1:00PM PST** 举办一场免费的 AI 工作坊，演讲嘉宾包括来自 Cerebras 的 Daria Soboleva、Aran Komatsuzaki 以及来自 **Artificial Analysis** 的 George Cameron。
   - 演讲将涵盖从 **阿里巴巴 Qwen 3 系列** 等新模型到针对各种项目类型的模型选择策略等主题，[在此 RSVP](https://lu.ma/7f32yy6i?tk=jTLuIY&utm_source=ella)。
- **AI 工作坊**：该 AI 工作坊深入探讨当前有趣的研究。
   - 研究人员将向您展示如何选择合适的模型。


  

---


### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1382431878916739172)** (1 messages): 

> `Windsurf 浏览器，Windsurf Wave 10，Windsurf 社交链接` 


- **Windsurf 推出浏览器**：作为 [Windsurf Wave 10 - Day 2](https://windsurf.com/blog/windsurf-wave-10-browser) 的一部分，Windsurf 正在发布一款全新的**全功能浏览器**，旨在弥合开发工作流与 Web 活动之间的鸿沟。
- **Windsurf 浏览器向所有用户开放**：新的 **Windsurf 浏览器** 正在向所有 **Free、Pro 和 Teams 用户** 推送 Beta 版，而 **Enterprise** 用户将滚动获得此功能。
   - 观看 [Youtube 视频](https://youtu.be/r4WqTyLb4Vk?si=lNo4aMCIg8tHsVAp)，阅读 [更新日志](https://windsurf.com/changelog) 或加入 [r/Windsurf 讨论](https://reddit.com/r/windsurf)。
- **在社交媒体上关注 Windsurf**：在 [X/Twitter](https://x.com/windsurf_ai/status/1932871558219117022)、[Bluesky](https://bsky.app/profile/windsurfai.bsky.social)、[Threads](https://www.threads.com/@windsurf_ai/post/DKxShipsbPk?hl=en)、[Instagram](https://www.instagram.com/p/DKxWKKkxvu6/) 和 [Linkedin](https://www.linkedin.com/feed/update/urn:li:activity:7338638111393886211/) 上关注 Windsurf。