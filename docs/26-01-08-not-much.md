---
companies:
- stanford
- google
- google-deepmind
- alibaba
- z-ai
- tii
- ai21-labs
- huggingface
date: '2026-01-08T05:44:39.731046Z'
description: '以下是为您翻译的中文内容：


  **斯坦福大学的论文**揭示了 **Claude 3.7 Sonnet** 能够记忆 **95.8% 的《哈利·波特 1》**内容，突显了其与 **GPT-4.1**
  相比在版权提取方面的风险。在关于开源软件（OSS）资助的辩论中，**Google AI Studio** 宣布赞助 **TailwindCSS**。**谷歌**及
  **Sundar Pichai** 推出了 **Gmail Gemini 3** 功能，包括 AI 概览和带有用户控制选项的自然语言搜索。**阿里巴巴通义千问（Alibaba
  Qwen）**发布了 **Qwen3-VL-Embedding** 和 **Qwen3-VL-Reranker**，这是一个支持文本、图像和视频的多模态、多语言检索技术栈，具备量化和指令定制功能，并在基准测试中取得了强劲结果。**智谱
  AI（Z.ai）**在港交所（HKEX）上市，其 **GLM-4.7** 在 Artificial Analysis 智力指数 v4.0 中处于领先地位，在推理、代码和智能体应用方面均有提升，该模型采用大规模混合专家（MoE）架构并遵循
  MIT 开源协议。来自 TII 的 **Falcon-H1R-7B** 旨在提升小型模型的推理效率，在智力指数中得分为 16。**AI21 Labs** 推出了
  **Jamba2**，这是一款内存效率极高的企业级模型，采用 SSM-Transformer 混合架构，遵循 Apache 2.0 协议，可通过 SaaS 和
  Hugging Face 获取。**vLLM** 展示了在推理和内核工程方面的吞吐量提升。Justin Lin 指出：*“嵌入（Embeddings）默认就应该是多模态的。”*'
id: MjAyNi0w
models:
- claude-3-7-sonnet
- gpt-4-1
- gemini-3
- qwen3-vl-embedding
- qwen3-vl-reranker
- glm-4-7
- falcon-h1r-7b
- jamba2
people:
- sundarpichai
- justinlin610
title: 今天没发生什么事。
topics:
- copyright-extraction
- multimodality
- multilinguality
- retrieval-augmented-generation
- model-architecture
- mixture-of-experts
- model-quantization
- reasoning
- inference
- kernel-engineering
- memory-optimization
- enterprise-ai
---

**平静的一天**

> 2026年1月7日至1月8日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 社区（包含 **204** 个频道和 **4649** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**415 分钟**。**我们的新网站**现已上线，支持全文元数据搜索，并以精美的 vibe coded 风格展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻分类，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

---

# AI Twitter 综述

**热门推文（按互动量排序）**

- **斯坦福大学关于 LLM 记忆与版权提取的论文**：摘要称可以从多个前沿模型中提取受版权保护的文本；值得注意的是，该研究声称 **Claude 3.7 Sonnet 在其实验设置中重现了 95.8% 的《哈利·波特 1》**；与之形成对比的是 GPT-4.1 的比例要低得多 ([ednewtonrex](https://twitter.com/ednewtonrex/status/2009201019184415218))。
- **Google 赞助 TailwindCSS**：Google AI Studio 宣布现已成为 **tailwindcss** 的赞助商，在开源资金争议背景下，此举被视为对生态系统的支持 ([OfficialLoganK](https://twitter.com/OfficialLoganK/status/2009339263251566902))。
- **Gmail “Gemini 时代”发布**：Google 和 Sundar Pichai 宣布了由 **Gemini 3** 驱动的 Gmail 功能——包括 AI Overviews、AI 收件箱、写作辅助和自然语言搜索——并强调了用户可控的开关 ([Google](https://twitter.com/Google/status/2009265269382742346), [sundarpichai](https://twitter.com/sundarpichai/status/2009291313888547131), [Google](https://twitter.com/Google/status/2009266902112104711))。
- **Qwen 多模态检索发布**：阿里巴巴 Qwen 发布了 **Qwen3-VL-Embedding** 和 **Qwen3-VL-Reranker**（多模态、多语言、开源），旨在针对文本+图像+视频进行检索/RAG ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2009264754917863924))。
- **Z.ai / GLM 里程碑 + IPO 时刻**：Z.ai 宣布现已在港交所（HKEX）上市，并举办社区挑战赛；GLM-4.7 仍是叙事核心 ([Zai_org](https://twitter.com/Zai_org/status/2009290783678239032))。

---

**开放权重模型：GLM-4.7 的势头、Qwen 多模态检索以及小型“高效”推理新秀**

- **GLM-4.7（开放权重）在 Artificial Analysis Intelligence Index v4.0 中夺冠**：Artificial Analysis 报告称 **GLM-4.7 [Reasoning] 分数为 42**（高于 GLM-4.6 的 32），在**代码编写、Agent 用途和科学推理**方面有显著提升，此外 **GDPval-AA ELO 为 1193**（在其评估的开放权重模型中最高）。详细信息包括 **200K 上下文**、**纯文本输入输出**、**355B MoE 总参数 / 32B 激活参数**、**MIT 许可证**，以及实际部署说明：**~710GB BF16** 权重意味着它无法装入单个 8×H100 节点（约 640GB）([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009117037667422457))。Z.ai 还发布了一段较长的“历程”回顾，随后发布了公开市场里程碑/社区挑战赛 ([Zai_org](https://twitter.com/Zai_org/status/2009154193244721326), [Zai_org](https://twitter.com/Zai_org/status/2009290783678239032))。
- **Qwen3-VL-Embedding + Qwen3-VL-Reranker（多模态检索技术栈）**：Qwen 推出了基于 Qwen3-VL 构建的**两阶段检索架构**（嵌入模型 + 重排序模型），可处理**文本/图像/截图/视频/混合输入**，支持 **30 多种语言**，具有可配置的嵌入维度、指令定制和部署量化功能。阿里巴巴将其定位为多模态检索基准测试的 SOTA，并通过 Hugging Face/GitHub/ModelScope 发布，阿里云 API “即将推出” ([Alibaba_Qwen](https://twitter.com/Alibaba_Qwen/status/2009264754917863924))。社区反响：基准测试结果显示 **MMEB-V2 为 77.9%**，**MMTEB 为 67.88%** ([HuggingPapers](https://twitter.com/HuggingPapers/status/2009295485966672072))；Justin Lin 指出从 VL 到 VL 嵌入的扩展，并认为嵌入“默认就应该是多模态的” ([JustinLin610](https://twitter.com/JustinLin610/status/2009277701727637785))；vLLM 在每夜构建版中添加了支持 ([vllm_project](https://twitter.com/vllm_project/status/2009316281275830351))。
- **Falcon-H1R-7B (TII) 进入“小型推理”赛道**：Artificial Analysis 强调 **Falcon-H1R-7B** 是来自阿联酋的新选手，采用 **Transformer-Mamba 混合架构**，在 12B 以下模型中，其 v4.0 智能指数分数为 **16**，并指出其在 **Humanity’s Last Exam**、**τ²-Bench Telecom** 和 **IFBench** 上的表现，以及在其新的开放性指数（Openness Index）框架中获得了中等的开放性得分 (44) ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009343487855219171))。
- **AI21 Jamba2（内存高效的企业级模型家族）**：AI21 宣布推出 **Jamba2**，强调“企业级可靠性/可控性”、**SSM-Transformer 混合架构**、KV-cache 创新以及 **Apache 2.0** 许可证，可通过 AI21 SaaS 和 Hugging Face 获取 ([AI21Labs](https://twitter.com/AI21Labs/status/2009259475643846978))。

---

**推理与 Kernel 工程：vLLM 吞吐量领先、KV 卸载以及 “AI 编写” 的 Kernel**

- **vLLM 在 B200 上达到 16k TPS**：vLLM 强调了社区报告的一个里程碑，在 **NVIDIA B200** 上达到了 **16k tokens/sec** ([vllm_project](https://twitter.com/vllm_project/status/2009196819331600648))。
- **KV Offloading Connector (IBM Research) 落地 vLLM**：新的连接器可将 **KV cache 异步卸载到 CPU RAM**，以处理抢占并提高并发性；vLLM 称在 H100 上 **吞吐量提升高达 9 倍**，缓存命中时的 **TTFT 降低了 2 倍到 22 倍**。他们还描述了 Host-Device 传输优化（通过连续物理块实现高速异步 DMA），并提供了 CLI flags (`--kv_offloading_backend native ...`) 以及深度解析博客链接 ([vllm_project](https://twitter.com/vllm_project/status/2009217642507477222), [vllm_project](https://twitter.com/vllm_project/status/2009217645946773534), [vllm_project](https://twitter.com/vllm_project/status/2009217648224247840))。
- **Kernel LLM / “Oink” fused RMSNorm Kernel 性能提升 40%**：Mark Saroufim 介绍了集成到 vLLM 中的 AI 生成 fused RMSNorm kernel 的早期结果，报告称 **相比现有 RMSNorm kernel 提速约 40%**，**端到端**提升约 1.6%。有趣的工程角度在于：生成的代码尝试嵌入一种类似于启发式 Autotuner 的策略，专门针对热点 Shape（例如 7168 BF16），并使用诸如“直接 GMEM”加载、SMEM 仅用于 Reduction 等技巧——同时承认了复杂性增加和稳定性风险（Segfault 情况、Cluster Launch 交互）。他将系统级评测套件（vLLM、FlashInfer 的方法）视为通往 “SOTA AI kernels” 的路径，尽管存在确定性/测试方面的担忧 ([marksaroufim](https://twitter.com/marksaroufim/status/2009096176789016600))。
- **通过 Keras Pallas 编写 Python Kernel**：François Chollet 强调了使用 **Pallas** 在 **Python 中**编写高性能自定义算子（Custom Ops），并将其下层到 **Mosaic (TPUs)** 或 **Triton (GPUs)**，这消除了许多工作流中“为了写 Kernel 而离开 Python”的需求 ([fchollet](https://twitter.com/fchollet/status/2009221193812128006))。
- **“Kernel 工具碎片化”的梗流行是因为它描述了事实**：一个病毒式的讽刺段子列举了 DSLs/Backends（Triton/Mojo/cuTile/ROCm/TileIR/TileLang/Pallas/TT-Metal/CSL...）的不断更迭，捕捉到了生态系统碎片化和迁移成本的真实代价 ([tetsuo_cpp](https://twitter.com/tetsuo_cpp/status/2009238107309461782))。

---

**Agent 与开发工作流：“Agent 文件”、大规模 Prompt/系统改进以及大型代码库的现状**

- **Agents-as-folders + Skills 标准化**：LangChain 的 Harrison Chase 强调了一个具体的模式：“Agent 由 Markdown/JSON 文件定义”——如 `agents.md`、`subagents/`、`skills.md`、`mcp.json`——使 Agent 的打包/版本控制更像 Repo Artifacts，而非特定于框架的对象 ([hwchase17](https://twitter.com/hwchase17/status/2009388479604773076))。VS Code 将 **Agent Skills** 作为由 Anthropic 创建的“开放标准”发布，支持加载包含指令/脚本/资源的文件夹以处理专门任务（通过 `chat.useAgentSkills` 开启特性标志）([code](https://twitter.com/code/status/2009428464626016700))。  
- **Prompt/System 工作直接挂钩金钱收益**：Lovable 的一位工程师声称，系统性的 System Prompt 改进带来了 **4% 的性能提升**、更好的设计输出，并每年减少 **$20M** 的 LLM 成本——这被归因为大规模 Prompt 质量的复合效应，以及快速/安全实验的重要性 ([benjaminvrbk](https://twitter.com/benjaminvrbk/status/2009297105458716753)，后续总结 [benjaminvrbk](https://twitter.com/benjaminvrbk/status/2009297114992660857))。  
- **大型代码库仍是“困难模式”**：一个高质量的观察指出：Claude Code 在“企业级”仓库中表现可能不佳，因为其 Post-training 数据偏向小型仓库；真正的性能提升可能需要对你的仓库进行 **Continual Learning / Fine-tuning**，否则 RAG/手动文件读取将成为瓶颈。建议：通过清晰的 API 边界进行模块化以减轻 Context 负担 ([ibab](https://twitter.com/ibab/status/2009322166593179786))。Dejavucoder 补充了一个具体的工具差异：Cursor 的优势源于 **Codebase Embedding Indexing**，而 Claude Code 默认缺乏这一功能 ([dejavucoder](https://twitter.com/dejavucoder/status/2009375459109441545))。  
- **“Prompt 驱动开发”与 Agent 式测试**：GitHub 的 Copilot 团队将 Prompting 推崇为一种工程学科（重构、通过 MCP 进行文档查询、UI 工作、文档、测试）([code](https://twitter.com/code/status/2009097862517342442))。补充研究：对 AIDev 数据集的分析表明，包含测试的 Agent PR 随时间推移在增加，但往往体量更大、速度更慢，合并率却保持相近——这引发了关于评审者激励机制和不同 Agent 间测试质量差异的讨论 ([omarsar0](https://twitter.com/omarsar0/status/2009269127773605993))。  
- **DeepAgents / “Ralph 模式”生态构建**：LangChain DeepAgents 增加了 Skills 和 Memory；其构架是“Harness 级设计”：持续循环 + 文件系统/Git Memory，并将进度“技能化（Skillifying）”为在 Git 中跟踪的可复用知识 Artifacts ([Vtrivedy10](https://twitter.com/Vtrivedy10/status/2009295526974595519), [mstockton](https://twitter.com/mstockton/status/2009311366444638441))。Replit 的观点：自主性需要 Frontier Model、先进的 Context 管理和详尽的验证——并明确将这“三大支柱”命名 ([pirroh](https://twitter.com/pirroh/status/2009381577244258370))。  

---

**重大产品动态：OpenAI Healthcare、集成 Gemini 3 的 Gmail 以及“谁为 OSS 买单？”**

- **OpenAI for Healthcare / ChatGPT for Healthcare**: OpenAI 及其领导者描述了一项符合 HIPAA 标准的产品：“健康智能 + 可信医学证据 + 工作流 + 企业级控制”。提及的合作伙伴包括 **HCA、波士顿儿童医院 (Boston Children’s Hospital)、MSK、斯坦福医疗中心 (Stanford Health)** 等 ([bradlightcap](https://twitter.com/bradlightcap/status/2009408962135998653), [thekaransinghal](https://twitter.com/thekaransinghal/status/2009360917847548331), [OpenAI](https://twitter.com/OpenAI/status/2009441959497154829))。此外，还有关于 “ChatGPT Health” 记忆/存储更新的产品叙述 ([\_samirism](https://twitter.com/_samirism/status/2009221543214371150))。  
- **Gmail 成为 AI 原生收件箱**: Google 为邮件会话推出了 AI Overviews，以及 AI 辅助回复/校对、“AI 收件箱”视图和 Gmail 内部的自然语言查询功能，并明确强调了用户控制权和加入/退出机制 ([Google](https://twitter.com/Google/status/2009265269382742346), [Google](https://twitter.com/Google/status/2009266641121477002), [Google](https://twitter.com/Google/status/2009266902112104711))。工程师们随即指出 **网络钓鱼/诈骗检测** 是“最具影响力”的下一个功能，并警告当“可信收件箱 Agent”能够说服用户时，存在越狱滥用的风险 ([polynoamial](https://twitter.com/polynoamial/status/2009322743251259890), [giffmana](https://twitter.com/giffmana/status/2009341983953965236))。  
- **TailwindCSS 赞助连锁反应**: 在社区对 AI 时代的 OSS（开源软件）可持续性表示担忧后，Google AI Studio 宣布赞助 Tailwind ([OfficialLoganK](https://twitter.com/OfficialLoganK/status/2009339263251566902))。其他人将编程 Agent 视为 OSS 的“分发管道”，并呼吁更多大科技公司的支持 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009340177173680509), [sdrzn](https://twitter.com/sdrzn/status/2009361550117880171))。一个特别尖锐的建议是：让编程工具根据 Token 消耗比例贡献微额资金，自动支持所依赖的项目 ([nateliason](https://twitter.com/nateliason/status/2009279537343836261))。  

---

**研究信号：记忆化、Agent 记忆架构、后训练缩放和测量**

- **记忆化与版权提取并非假设**: 斯坦福大学的论文摘要（据转述）称，成功从多个生产级 LLM 中提取了受版权保护的作品，在其实验中特定模型的逐字重复率惊人，旨在反驳“LLM 不会记忆”的论点 ([ednewtonrex](https://twitter.com/ednewtonrex/status/2009201019184415218))。  
- **MAGMA：用于长程推理的多图 Agent 记忆**: 提出在 **语义、时间、因果、实体图** 上表示记忆，并通过策略引导的遍历（而非单一的向量嵌入相似度）进行检索；据报道在 **LoCoMo** 和 **LongMemEval** 上取得了提升 ([dair_ai](https://twitter.com/dair_ai/status/2009270633398718480))。  
- **SPOT @ ICLR 2026：LLM 后训练缩放**: 研讨会征稿关注连接 **算法、数据和系统** 的后训练 (post-training) 缩放原则；投稿截止日期为 2026 年 2 月 5 日 ([spoticlr](https://twitter.com/spoticlr/status/2009137185510052302))。  
- **基准测试从“能力”扩展到开放性 + Agent 真实性**: Artificial Analysis 继续推动 **GDPval-AA**（包含工具/网络/终端环境的真实知识工作任务）等指数和标准化的开放性指数 (Openness Index)；他们还出现在 Latent Space 讨论评估脆弱性/提示词方差以及“神秘顾客”政策 ([ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/2009367497913585905))。Paras Chopra 主张衡量 **人+AI** 的协作能力，而非单纯的 AI 能力，以避免为了替代人类而非提升能力进行优化 ([paraschopra](https://twitter.com/paraschopra/status/2009118690823033165))。  

---

**行业宏观： “氛围编码 (vibe coding)”、系统工程护城河与算力现实**

- **工程价值随着 Agent 生产力的提高而向上迁移**：多条推文指向同一个观点：随着代码生成变得廉价，**复杂度管理、可靠性和系统工程**变得更有价值，而非贬值。一个版本预测初级前端岗位将消失，而“风险管理编码 Agent”将成为高端劳动力 ([mbusigin](https://twitter.com/mbusigin/status/2009090018682323367))；另一个版本则将其描述为“非工程师意识到工程从来不是关于写代码的” ([\_0xaryan](https://twitter.com/_0xaryan/status/2009257975718793460))。  
- **“所有软件都将是生成式的，且是被生成的”**：Vercel CEO 的简练论点，认为默认的开发模式将永久翻转 ([rauchg](https://twitter.com/rauchg/status/2009324546294468769))。Yuchen Jin 将其描述为编程领域的杰文斯悖论（Jevons paradox）：更多的编码，更多的编码者，更高的峰值杠杆 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2009324436353372547))。  
- **计算供应链指标**：Epoch AI 声称全球算力已超过 **1500万张 H100 等效卡**，并发布了一个公开的“AI Chip Sales”浏览器，以及一个大致的电力推算，即在考虑数据中心开销之前，芯片功耗已超过 **10 GW** ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2009366360183460237))。  

(范围说明：输入内容包含大量地缘政治/政治评论和一些非 AI 话题；本摘要优先考虑具有技术可操作性的 AI/模型/系统内容，同时在“热门推文”中列出了参与度最高的非技术推文。所有 nitter 链接都已重写为 twitter.com。)


---

# AI Reddit 综述

## /r/LocalLlama + /r/localLLM 综述

### 1. Hugging Face 模型发布与基准测试

  - **[Hugging Face 火力全开：30+ 新增/热门模型（LLM, Vision, Video）及链接](https://www.reddit.com/r/LocalLLM/comments/1q7b54w/hugging_face_on_fire_30_newtrending_models_llms/)** (热度: 57): **Hugging Face** 发布了 30 多个涵盖文本生成、视觉、视频和音频等各个领域的新增及热门模型。值得关注的模型包括 **tencent/HY-MT1.5-1.8B**（针对边缘部署优化的多语言翻译模型）和 **LGAI-EXAONE/K-EXAONE-236B-A23B**（用于高级推理的大型韩语 LLM）。在视觉领域，**Qwen/Qwen-Image-2512** 提供高保真文本转图像生成，而 **Lightricks/LTX-2** 提供了一个联合影音基础模型，用于同步视频和声音生成。这些模型专为内容生成、边缘部署和复杂推理任务等多样化应用而设计，许多模型支持 quantization 和快速推理功能。[Hugging Face Models](https://huggingface.co/models) 一位用户强调了 **Qwen 3 30B** 模型在 16G GPU 上处理商业推理任务的适用性，认为这可能是他们的最佳选择，特别是如果能在 GGUF LM Studio 上使用的话。

    - 用户 'alex_godspeed' 正在考虑将 Qwen 3 30B 模型用于商业推理用例，特别指出其与 16GB GPU 的兼容性。他们表达了对该模型在 GGUF LM Studio 上可用性的需求，这表明他们关注高效部署，并可能担心模型大小或硬件性能限制。这突显了在实际应用中模型与特定平台和硬件配置兼容性的重要性。

  - **[指南：如何运行 Qwen-Image 扩散模型！（14GB RAM）](https://www.reddit.com/r/LocalLLM/comments/1q7e2ol/guide_how_to_run_qwenimage_diffusion_models_14gb/)** (热度: 26): **这篇文章介绍了一份在本地设备上运行最新的 **Qwen-Image-2512** 文本转图像扩散模型及其编辑版本 **Qwen-Image-Edit-2511** 的指南。指南涵盖了使用 ComfyUI、stable-diffusion.cpp 和 diffusers 等库的安装和设置，为获得最佳性能，至少需要 `14GB` 的内存/显存（RAM/VRAM）。它包含了使用 4-bit、FP8 和 GGUF 模型变体的指令，并提供了创建有效 prompts 和调整超参数（如 sampling 和 guidance）的技巧。指南还强调了 GGUF 最近的更新，通过优先处理重要层来提高质量，可在 [Hugging Face](https://huggingface.co/unsloth/Qwen-Image-2512-GGUF) 上获取。** 一位评论者表达了对更易用的 UI 的渴望，而不是 ComfyUI，他们觉得由于视力障碍，ComfyUI 使用起来很困难，尽管承认其功能强大。

### 2. 本地 LLM 部署与硬件考量

  - **[如果 VRAM 仅限 8GB，拥有大容量 RAM（96GB 甚至 128GB）是否有意义？](https://www.reddit.com/r/LocalLLM/comments/1q7e34g/does_it_make_sense_to_have_a_lot_of_ram_96_or/)** (热度: 75): **在 VRAM (8GB) 有限但 RAM (高达 128GB) 充足的情况下运行本地大语言模型 (LLMs) 在技术上是可行的，但存在局限性。主要限制在于系统 RAM 与 VRAM 之间的速度差异。DDR5 RAM 提供的带宽约为 `80 GB/s`，允许将量化为 `Q8` 的 `80B` 模型以约 `1 token/s` 的速度运行。相比之下，VRAM 带宽在 `200 GB/s` 到 `2000 GB/s` 之间，根据 GPU 的不同，同样的模型运行速度可达 `2.5 到 25 tokens/s`。Mixture-of-Experts (MoE) 模型由于每个 token 仅激活一部分参数，即使参数量巨大也能实现更高的吞吐量，使用系统 RAM 时潜力可达 `20 tokens/s`。** 一位评论者建议，由于 VRAM 具有显著的速度优势，与其增加 RAM，不如投资更好的 GPU。另一位评论者指出，虽然 128GB RAM 允许尝试更大的 MoE 模型，但这些模型在处理编程等任务时往往无法达到生产力阈值，这表明尽管理论上可行，但实际应用中存在限制。

    - uti24 强调了使用系统 RAM 与 VRAM 运行大型模型的速度限制。DDR5 RAM 通常提供约 80 GB/s 的带宽，允许量化为 Q8 的 80B 模型以约 1 token/s 的速度运行。相比之下，VRAM 带宽范围为 200 GB/s 到 2000 GB/s，根据 GPU 的能力，同样的模型可以以 2.5 到 25 tokens/s 的速度运行。这说明了在这种任务中 VRAM 相对于系统 RAM 的巨大性能优势。
    - Medium_Chemist_4032 讨论了在 MoE (Mixture-of-Experts) 模型中使用大容量 RAM 的实际局限性。尽管拥有 128GB 的 RAM，他们发现较大的 MoE 模型在编程等任务中无法达到最低生产力阈值。这些模型经常无法生成单个可运行的文件并陷入死循环，这表明单纯增加 RAM 并不能保证所有应用的有效性能。
    - netroxreads 分享了他们在配备 256GB UMA RAM 的 M3 Ultra 系统上的经验，该系统能以每秒 70 tokens 的速度运行 120B GPT-OSS 模型。这种性能归功于统一内存架构 (UMA)，这与 PC 上预期的速度慢得多的 GPU/CPU RAM 混合设置形成鲜明对比。这突显了 UMA 在高效处理大型模型方面的潜在优势。

  - **[为我的公司创建一个本地离线 LLM](https://www.reddit.com/r/LocalLLM/comments/1q7d9uj/creating_a_local_offline_llm_for_my_company/)** (热度: 32): **为大约 150 名用户构建内部离线 LLM 在技术上是可行的，但需要大量的硬件和基础设施投资。单块 RTX 5090 可能足以进行原型设计，但扩展到生产环境将需要更强大的配置，仅 AI 处理器就可能涉及 `50k+ USD` 的投入。为了实现生产级的弹性和可扩展性，建议使用基于 Kubernetes 的基础设施，但这会增加复杂性和维护开销。像 `VLLM` 这样的现成解决方案可以处理推理，但在没有额外基础设施的情况下，可能不适合企业级部署。** 评论者建议在开始前定义具体的使用场景和性能要求（例如速度 vs 准确度）。他们还建议考虑针对组织特定任务使用 Retrieval-Augmented Generation (RAG)，这可能不需要进行完整的模型训练。为了控制成本，建议租赁硬件或使用云服务进行测试。

    - **Own_Attention_3392** 强调了训练 LLM 与运行 LLM 之间的区别，并指出硬件需求大不相同。他们建议对于组织特定任务使用 Retrieval-Augmented Generation (RAG)，而无需训练新模型，因为 RAG 可以与现有的 LLM 服务集成。
    - **phocuser** 讨论了在确定硬件需求之前定义 LLM 用途（如速度 vs 准确度）的重要性。他们指出，即使是像 5090 这样的高端 GPU，在处理接近 GPT-4.1 能力的模型时也可能会感到吃力。他们建议构建硬件堆栈以处理多个并发请求，并建议在本地硬件或云服务器上测试模型以评估性能。
    - **sometimes_angery** 分享了构建离线 MLOps 平台的见解，指出硬件投资高达 7.5-10 万美元。他们建议在生产环境中使用 Kubernetes 以实现可扩展性和弹性，但也警告这需要专门的维护人员。对于更简单的设置，他们推荐使用 VLLM 在单台机器上进行推理，尽管这可能不适合企业级生产。

### 3. 使用 GLM 4.7 与 Claude Sonnet 4.5 进行编程

  - **[一直在用 GLM 4.7 替代 Claude Sonnet 4.5 进行编程，成本差异巨大](https://www.reddit.com/r/LocalLLM/comments/1q79orf/been_using_glm_47_for_coding_instead_of_claude/)** (热度: 83): **该贴讨论了在调试、重构和代码生成等编程任务中，对 Claude Sonnet 4.5 与来自 Zhipu AI 的 GLM 4.7 进行的比较。用户发现，作为开源模型的 GLM 4.7 在 `85-90%` 的情况下能提供可运行的代码，这与 Claude Sonnet 4.5 的质量接近，但成本显著降低，仅为 API 支出的约 `1/5`。GLM 4.7 在处理长文件方面也优于 DeepSeek 和 Kimi 等竞争对手，不会丢失上下文或快速触及 Token 限制。然而，Claude Sonnet 4.5 在 UI/UX 和高层讨论方面依然更胜一筹。** 评论者指出，GLM 4.7 处理长文件非常有效且不会幻觉导入语句，使其成为大批量编程工作的经济之选。另一位用户分享了使用 Opus 4.5 的昂贵经历，强调了 GLM 4.7 的潜在节省空间。

    - Scared-Biscotti2287 强调 **GLM 4.7** 在处理长文件方面特别有效，并避免了像 Kimi 等其他模型常见的幻觉导入问题。虽然它在解释方面可能不如 Claude 精炼，但其强项在于代码生成，是批量编程任务的性价比之选。
    - whyyoudidit 分享了在 Visual Studio Code 中使用 **Opus 4.5** 的个人经历，指出在重构 `1400 行代码` 时，`5 分钟内产生了 10 美元` 的巨额成本。这突显了在使用某些模型进行大规模代码重构任务时可能产生的高昂成本，尤其是对于初次使用这些工具的用户。
    - No_Conversation9561 询问了所使用的硬件，这在使用 GLM 4.7 等模型进行编程任务时，是影响性能和成本的关键因素。硬件规格会影响代码生成和处理的效率与速度。

  - **[对话树搜索 - 类 MCTS 树搜索以寻找最佳对话路径（这样你就不必亲自试错了）](https://www.reddit.com/r/LocalLLaMA/comments/1q71sbe/dialogue_tree_search_mctsstyle_tree_search_to/)** (热度: 356): **该项目引入了一种利用并行束搜索（Beam Search）算法而非传统的蒙特卡洛树搜索（MCTS）进行对话优化的新方法。该方法生成多种对话策略，将其分叉为用户意图变体，并使用三个独立的 LLM 裁判对对话路径进行评分和剪枝。系统旨在处理多样化的用户意图，并通过 GPT-Researcher 集成了深度研究能力以获取领域上下文。它支持 OpenAI 兼容的端点，并在 Apache 2.0 许可证下开源。该方法属于 Token 密集型，每次运行可能需要超过 300 次 LLM 调用，目前仅限于 OpenAI 模型。** 评论者赞赏在对话优化中使用束搜索而非 MCTS，认为其更适合维持连贯的对话路径。用户意图分叉功能被视为一个有价值的补充，允许针对不同的角色测试策略。还有建议探索替代的搜索提供商以降低成本。

    - TheGrossVolcano 强调在对话路径优化中使用束搜索优于纯蒙特卡洛树搜索 (MCTS)。束搜索更适合对话系统，因为它能防止探索过程偏离相关路径太远，这对于保持连贯且符合上下文的对话至关重要。
    - harlekinrains 指出了 Firecrawl 订阅模式的高昂成本（每年 140 美元），并建议寻找更具成本效益的替代方案。他们还提供了一个 GitHub 仓库链接，该仓库汇总了各种搜索提供商，对那些寻求实施替代解决方案的用户非常有用。
    - charlesrwest0 提出了一个有趣的问题，即这种对话优化技术在角色扮演（RP）场景中的潜在应用，建议可以通过寻找最佳对话路径来增强 RP 的回复质量。


## 较少技术性的 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. AI 模型与基准测试发布

- **[华尔街日报：据报道 Anthropic 正以 3500 亿美元估值筹集 100 亿美元，AI 融资进程加速](https://www.reddit.com/r/singularity/comments/1q75o0z/wsj_anthropic_reportedly_raising_10b_at_a_350b/)** (活跃度: 218): 据报道，**Anthropic** 正在以 3500 亿美元的估值筹集 100 亿美元，这标志着 AI 历史上规模最大的私募融资之一。在短短四个月内，其估值从 1830 亿美元飙升至 3500 亿美元，凸显了资本正迅速向领先的 AI 模型开发商集中。此次融资由对算力和基础设施的高需求驱动，而非即时的产品供应，并符合对 2026 年前 AI IPO 活动增加的预期，反映了投资者对 AI 领域日益增长的信心。[来源：华尔街日报](https://www.wsj.com/tech/ai/anthropic-raising-10-billion-at-350-billion-value-62af49f4)。评论者强调，Anthropic 在代码方面的深度优化是一项竞争优势，认为这为公司构筑了“护城河”。还有观点认为，美国经济正严重依赖 AI 贸易，而其他行业的增长则相对缓慢。

    - Anthropic 对代码优化的关注被视为显著的竞争优势，创造了使其区别于其他 AI 公司的“护城河”。这表明他们在代码优化方面的技术能力是其获得高估值和吸引投资者的关键因素。
    - 讨论强调了 Google 等大公司对 AI 的战略投资，这被视为维持竞争均势的必要举措。这反映了一个更广泛的趋势，即 AI 正在成为投资的关键领域，正如关于美国经济依赖 AI 贸易实现增长的评论所指出的，这可能会以牺牲其他行业为代价。

  - **[QwenLong-L1.5 | 长期记忆 DIY](https://www.reddit.com/r/Oobabooga/comments/1q73tnh/qwenlongl15_long_term_memory_diy/)** (活跃度: 2): **QwenLong-L1.5** 引入了一种新颖的 AI 模型长期记忆管理方法，即通过推理来标记和存储核心信息，而不是用整个聊天历史来撑大上下文空间。这种方法在 [通义智文白皮书](https://huggingface.co/Tongyi-Zhiwen) 中有详细介绍，表明其性能优于传统的长期记忆技术。该模型还因其强大的推理能力而受到关注，使其非常适合角色扮演场景等应用。一些用户表示有兴趣测试 QwenLong-L1.5 在角色扮演应用中的表现，并根据其规格参数预期其会有更好的表现。

    - 一位用户讨论了 QwenLong-L1.5 中长期记忆的实现，强调了存储过去交互的内存缓冲区的使用。该缓冲区会定期进行剪枝以保持性能，确保仅保留最相关的数据。这种方法使模型能够处理超长对话，而不会导致响应质量显著下降。
    - 另一条评论深入探讨了 QwenLong-L1.5 的性能基准，指出其 `BLEU score` 比前代产品提高了 15%。这一进步归功于增强的上下文管理和记忆保留能力，这对于需要持续对话连贯性的应用至关重要。
    - 围绕 QwenLong-L1.5 记忆系统的可扩展性引发了技术争论。一些用户对内存缓冲区带来的计算开销表示担忧，尤其是在资源受限的环境中。另一些人则认为，考虑到模型能够在更长的交互中保持上下文，这种权衡是值得的，这是对话式 AI 的重大进步。

  - **[我让 Gemini 3 Pro/Flash 玩了 21,000 手扑克](https://www.reddit.com/r/GeminiAI/comments/1q7gy25/i_made_gemini_3_proflash_play_21000_hands_of_poker/)** (活跃度: 124): 图片是一张折线图，展示了包括 **Gemini 3 Pro/Flash**、**GPT-5.2/5 mini**、**Grok 4.1 Fast Reasoning** 以及 **Opus/Haiku 4.5** 在内的多种 AI 模型在 `21,000` 手扑克比赛中的表现。图表显示 **Gemini 3 Pro** 在接近尾声时盈利大幅增加，表明其在该基准测试中具有卓越性能。这是名为 **PokerBench** 的新 LLM 基准测试的一部分，该基准测试允许在竞争环境下评估 AI 模型的扑克策略。数据和模拟器可在 [PokerBench 官网](https://pokerbench.adfontes.io/) 和 GitHub 上获取。一位评论者指出，在正面交锋中，**Gemini 3 Flash** 的表现似乎优于 **Gemini 3 Pro**，这表明后者的成功可能并非纯粹由于技能，而可能是由于在大赛环境中的运气成分。

- 在 Gemini Flash 和 Pro 的直接对比中，Flash 在扑克比赛中表现得更好，这表明在此场景下它可能拥有更有效的策略或决策过程。这可能暗示了两款模型在架构或训练数据上的差异，使得 Flash 在扑克场景中更具优势。
- 一位用户询问了让这些模型互相对战的成本，这对于此类大规模模拟是一个相关的考虑因素。成本将取决于所需的计算资源、模拟持续时间以及所使用的特定基础设施（如云服务或本地硬件）等因素。

- **[我是 Lightricks 的联合创始人兼 CEO。我们刚刚开源了 LTX-2，这是一个生产就绪的音视频 AI 模型。欢迎提问 (AMA)。](https://www.reddit.com/r/StableDiffusion/comments/1q7dzq2/im_the_cofounder_ceo_of_lightricks_we_just/)** (热度: 2083): **Lightricks** 开源了 **LTX-2**，这是一个生产就绪的音视频 AI 模型，包括权重、代码、训练器、基准测试、LoRAs 和文档。该模型旨在本地消费级 GPU 上运行，使其可用于实际应用。此次开源发布旨在解决运行和复现多模态模型的挑战，因为这些模型通常难以实现。该发布是增强生产环境中 AI 模型易用性和可访问性的更广泛战略的一部分。更多详情可见 [LTX-2 模型页面](https://ltx.io/model)。评论者对开源 LTX-2 背后的动机感到好奇，一些人对其在开源视频技术领域的潜在影响表示感谢和兴奋。开源这一决定被视为一个重大举措，可能会影响多模态模型的未来。

    - 开源 LTX-2 的决定是出于促进社区协作和创新的承诺。团队旨在避免之前像 Wan 2.6+ 这样转为闭源而导致社区不满的模型的坑。通过发布开放权重，Lightricks 希望保持透明度，并鼓励社区驱动的改进和适配。
    - Lightricks 对 LTX-2 施加了某些训练限制以符合法律标准，例如避免 NSFW 和受版权保护的材料。这种限制可能会影响模型的通用性，但开源特性允许社区在法律范围内潜在地重新训练并扩展其能力。这种方法随着时间的推移可能会增强模型的范围和适应性。
    - LTX-2 作为开源模型的发布被视为开源视频 AI 领域的重大转变。它与之前限制访问的模型形成鲜明对比，社区热切关注 Lightricks 未来将如何维持其对开源原则的承诺。开放权重是确保社区持续参与和开发的垫脚石。

- **[[P] 用于微调 4B 模型中合成数据生成的三阶段自包含评估协议 (实验 3/100)](https://www.reddit.com/r/MachineLearning/comments/1q7f7tr/p_threephase_selfinclusive_evaluation_protocol/)** (热度: 6): **该帖子概述了一个用于合成数据生成的**三阶段自包含评估协议**，使用的是微调后的 4B 模型。该协议包括一个**生成阶段 (Generation Phase)**，其中多个模型（包括一个微调后的 4B 模型）根据专有提示词生成回复。在**分析阶段 (Analysis Phase)**，每个模型根据连贯性和创意等标准对输出进行排名。最后，在**聚合阶段 (Aggregation Phase)** 汇总这些排名以进行综合评估。该实验在 MIT 许可证下开源，所有数据均可在 [GitHub](https://github.com/Roforum/Xthos-v2-the-sovereign-architect-Model-Evaluation-Experiment) 上获取。其目的是探索 LLM-as-judge 设置中的偏差以及主观评估的可复现性，并支持通过 Ollama 进行本地推理。**评论者正在讨论专有提示词和自我排名引入的潜在偏差，并建议采用更严谨的统计方法进行聚合。人们对微调权衡和本地推理设置也表现出兴趣，更多详情可见相关的 [Reddit 讨论](https://www.reddit.com/r/LocalLLaMA/comments/1q6p967/experimental_xthosv2_the_sovereign_architect/)。**

- 讨论重点介绍了微调后的 4B 模型的使用，特别关注合成数据生成的评估协议。模型的训练细节、数据集组成和量化选项在另一个单独的线程中讨论，强调了理解这些方面对于有效模型部署的重要性。评估协议是更大系列实验的一部分，这是计划的一百个实验中的第三个，展示了模型评估的系统化方法。
- 链接的讨论提供了关于使用 Ollama 进行本地推理设置的见解，这对于在本地环境中实现模型的用户至关重要。这种设置允许对模型进行高效的测试和部署，确保合成数据生成过程既稳健又具可扩展性。GitHub 仓库中提供的原始数据和分析支持了结果的透明度和可复现性，这对于验证模型性能至关重要。
- 该帖子邀请大家就方法论提问，表明了改进评估协议的协作方式。这种对社区参与的开放态度建议了一个动态的开发过程，其中反馈可以引导模型评估的迭代改进。对方法论相关问题的强调突出了项目的技术深度以及同行评审在推动该领域发展中的重要性。

  - **[[P] Automated Code Comment Quality Assessment with 94.85% Accuracy - Open Source](https://www.reddit.com/r/MachineLearning/comments/1q7rd9o/p_automated_code_comment_quality_assessment_with/)** (Activity: 10): **该帖子介绍了一个用于评估代码注释质量的文本分类器，在测试集上实现了 `94.85%` 的准确率。该模型是拥有 `66.96M` 参数的微调版 **DistilBERT**，并根据 **MIT License** 发布。它将注释分为四个类别：Excellent、Helpful、Unclear 和 Outdated，精确率分别为 `100%`、`89%`、`100%` 和 `92%`。该模型可以使用 `Transformers` 库轻松集成，并托管在 [Hugging Face](https://huggingface.co/Snaseem2026/code-comment-classifier) 上。潜在应用包括 CI/CD 集成、实时 IDE 反馈和开发者培训工具。** 一条热门评论请求关于模型创建过程的细节，表明了对方法论和实现细节的兴趣。

    - 该项目可能涉及使用机器学习模型来评估代码注释的质量。达到 94.85% 的准确率表明模型经过了良好的调优，可能利用了 NLP 技术来理解和评估注释的语义质量。该模型可能是在标记过的数据集上训练的，其中注释按质量分级，使用了可读性、相关性和信息量等特征。
    - 该项目的开源性质意味着代码和数据集可供公众使用和贡献。这种透明度允许社区驱动改进，并在不同的编程语言和代码库中验证模型的性能。该项目可能使用 TensorFlow 或 PyTorch 等流行库进行模型开发和训练。
    - 高准确率表明模型经过了严格测试，可能使用了交叉验证技术来确保稳健性。用于训练的数据集可能包括广泛的编程语言和注释风格，以便在不同的编码环境中良好地泛化。模型的性能可以根据现有工具或人工评估进行基准测试，以验证其有效性。

### 2. Claude Code Usage and Experiences

  - **[Opus 4.5 actually just… gets it? Shipped my first iOS app without knowing Swift](https://www.reddit.com/r/ClaudeAI/comments/1q73hkv/opus_45_actually_just_gets_it_shipped_my_first/)** (Activity: 974): **该帖子讨论了 **Opus 4.5** 的使用，这是一款显著简化应用开发的工具，允许非开发者在没有 Swift 先验知识的情况下创建一个功能齐全的 iOS 应用。用户强调了 Opus 4.5 如何直观地理解诸如 'make it feel calm and minimal' 之类模糊指令，并针对潜在问题提供主动反馈，类似于与高级开发者合作。这个版本的 Opus 因其改进的决策和调试能力而受到关注，减少了不断澄清的需求，并提供了更合理的决策解决方法。** 一些评论者注意到 Opus 4.5 倾向于在不同应用中使用相同的设计和配色方案，建议其 UI 设计能力缺乏多样性。

- **[Max 20x 计划下的实际使用情况 - 会话 4 小时后。](https://www.reddit.com/r/ClaudeCode/comments/1q7kwug/what_actual_usage_looks_like_against_max_20x_plan/)** (热度: 168)：**该帖子讨论了 Claude Code 的 Max 20x 计划的使用情况，强调在使用 4 小时后，用户仅消耗了 30% 的会话 Token 和 1% 的每周 Token。该用户正在处理市场营销和工程任务，并使用了 Opus 4.5 和 ultrathink 等功能。图片显示了一个使用情况仪表盘，指示了当前会话和每周的使用百分比。该用户对其他人声称很快就达到每周限制的说法表示怀疑，认为限制非常慷慨且足以满足他们的需求。帖子还提到了使用斜杠命令（slash commands）来自动化工作流验证过程。** 一位评论者提到了 5x 计划的问题，称他们在短短 2 小时内两次触及 5 小时的限制，这可能是一个 Bug。另一位评论者建议优化 CLI 工具参数，以减少发送到 LLM 的无关上下文，从而更有效地管理使用量。

    - srirachaninja 报告了 5x 计划的一个问题，即 5 小时的限制似乎不准确，因为他们在仅 2 小时内就触及了两次限制，且每个会话仅使用 6-8 个提示。这表明使用量追踪系统可能存在 Bug，特别是考虑到他们的使用模式并未发生显著变化。
    - positivitittie 建议通过使用 CLI 工具参数来过滤掉不必要的信息，从而优化发送到 LLM 的上下文，这有助于减小上下文大小并提高效率。他们还建议使用 Boris' Claude Code Setup 并配置脚本以尽量减少无关上下文，从而防止过早触及使用限制。
    - doomdayx 和 Drakuf 都提到了意外的使用量飙升，doomdayx 经历了在几分钟内突然消耗 5 小时限制的 40%，而 Drakuf 注意到在 12 小时后使用了每周限制的 27%。这些报告表明使用量追踪中可能存在不一致性，CC 团队已知晓并正在调查。

  - **[有哪些人们应该了解的 Claude Code 高级秘籍？拒绝废话](https://www.reddit.com/r/ClaudeCode/comments/1q7fs2o/what_is_some_serious_claude_code_sauce_people/)** (热度: 138)：**一位 Reddit 用户分享了一种提高 Claude Code 质量的技术：通过实现一个 `UserPromptSubmit` 类型的钩子（hook）来读取 Windows 上的 `.ps1` 文件，从而引导 Claude Code 为任务使用最相关的 Agent 或技能。这充当了一个“路由”文件，增强了特定任务的性能。另一位用户强调了“Plan mode”和构建特定项目技能的重要性。他们还分享了一份详尽的技术清单，包括：识别失败模式的**错误日志系统 (Error Logging System)**、将 **/Commands 作为轻量级本地应用**以快速创建工作流，以及实现**确定性安全钩子 (Hooks for Deterministic Safety)**以防止危险操作。此外，他们还建议通过手动管理上下文压缩来维持**上下文卫生 (Context Hygiene)**、在复杂项目中利用**子代理控制 (Subagent Control)**，以及使用**重提示系统 (Reprompter System)**进行结构化提示。详细介绍这些技术的完整文档可以在[此处](https://docs.google.com/document/d/1I9r21TyQuAO1y2ecztBU0PSCpjHSL_vZJiA5v276Wro/edit?usp=sharing)查看。** 一位评论者强调了以“循环”方式思考的重要性，建议提供编译和测试指令，以帮助 Claude Code 模拟典型的开发工作流。他们注意到实现的复杂性，但肯定了其价值，特别是对于 Web 应用，建议使用 Playwright 或 Selenium 等工具。

    - **错误日志系统 (Error Logging System)**：这涉及重建 Agentic 编程往往会掩盖的输入输出循环。通过记录失败时的确切触发提示并进行分类，开发者可以识别模式并理解出错原因，从而实现更有效的调试和优化。
    - **/Commands 作为轻量级本地应用**：Claude Code 中的斜杠命令被比作 Claude as a Service，提供了具有更快构建时间的 SaaS 威力。此功能允许创建既强大又高效的工作流，充当轻量级的本地应用程序。
    - **子代理控制 (Subagent Control)**：Claude Code 经常会为知识性任务生成 Sonnet/Haiku 等子代理。通过在全局 CLAUDE.md 中添加“始终启动 Opus 子代理”，开发者可以更有效地利用子代理，增强超越原生 Claude Code 能力的项目编排。

- **[超越 'Max 20X' 方案 - 升级到更高级别的方案??](https://www.reddit.com/r/ClaudeCode/comments/1q715zo/more_than_max_20x_plan_go_to_a_higher_plan/)** (活跃度: 111)：**用户发现他们的 'Max 20X' 方案额度消耗极快，由于涉及 subagents、异步操作和远程 Gitrees 的高强度使用，额度在 5-6 天内就告罄。他们尝试通过使用 `/clear` 和 `/compact` 命令以及将内存卸载到本地 SQLite 和 vector databases 来管理上下文使用。该用户正在考虑升级到比 'Max 20X' 更高级别的方案，可能是 'Teams' 方案，以满足日益增长的计算需求。该帖子强调了高效上下文管理的必要性以及高计算量使用可能带来的成本影响。** 评论者建议用户可能受益于关闭 auto-compact 功能，因为该功能会消耗大量上下文资源，并对用户的计算活动强度表示疑问，暗示如此高的使用量并不常见。

    - Familiar_Gas_1487 指出 'Max 20X' 方案的限制是按周设置而非按月，并建议使用 API 作为替代方案来管理更高强度的使用。这意味着用户可能通过直接集成 API 来绕过某些限制，这在工作模式上可能提供更多灵活性。
    - PathFormer 强调了一个关于 'auto compact' 功能的技术细节，该功能在每个会话中会消耗总计 200k 上下文中的 40k。禁用此功能可以显著减少上下文消耗，让用户更有效地利用方案额度。
    - HangJet 和 websitebutlers 都强调 '20x plan' 运行的是每周重置机制。这意味着遇到限制的用户可能只需等待重置，而不是寻求更高级别的方案，暗示用户对方案结构可能存在误解。

  - **[Claude Code 是最好的 Mac 清理应用](https://www.reddit.com/r/ClaudeCode/comments/1q7eqqm/claude_code_is_the_best_mac_cleaner_app/)** (活跃度: 198)：**这张图片幽默地展示了一个名为 "Claude Code" 的虚构应用的终端界面，将其描绘成一个 Mac 清理工具。它列出了清理的各种项目，如 "node_modules"、"Claude debug logs" 和 "下载文件夹中的 Python venvs"，总共释放了约 `10.5GB` 空间，磁盘空间从 `29GB` 增加到 `33GB`。这是对系统清理工具的讽刺性解读，暗示该应用通过删除不必要的文件能非常有效地释放空间。** 一位评论者幽默地表示这可能是一个陷阱，而另一位则拿 `rm -rf /` 这种会删除系统中所有文件的危险命令开玩笑。


  - **[在未进行任何操作的情况下损失方案使用额度](https://www.reddit.com/r/ClaudeCode/comments/1q7alp0/loosing_plan_usage_limits_without_touching/)** (活跃度: 84)：**用户反映在没有任何交互的情况下，**Claude Desktop** 和 **Claude Code** 的使用限制意外增加，特别是在 2026 年 1 月之后。一位用户注意到在启动 Mac 时，即便两天没使用该服务，其会话限制已使用了 `6%`。这个问题似乎影响到了 **Pro 方案用户**（$20/月），他们观察到在节日促销后 usage tokens 有所下降。**Anthropic** 缺乏沟通和支持是令人担忧的一点。** 用户对解释不明的使用量增加以及感知到的 **Anthropic** 沟通不力表示明显不满。一些用户对这种情况表达了失望和怀疑，表明需要更好的透明度和支持。

    - 一位用户报告称，在未主动使用服务的情况下，其会话限制使用量突然增加，在启动 Mac 时注意到 6% 的使用率。他们使用的是 $20 的 Pro 方案，并观察到在节日促销后 usage tokens 有所减少，对 **Anthropic** 的沟通和支持表示不满。
    - 另一位用户建议，将会话保持打开状态可能会导致随时间消耗 token，因为即便在未主动使用时，会话也会消耗少量 token。此外，使用 'skills' 或 'MCP' 会在会话启动时显著影响 token 使用量，这或许能解释意外的使用量增加。


### 3. AI Prompt Engineering 与使用挑战

- **[The day I stopped collecting “cool prompts” and started building a tiny standard library](https://www.reddit.com/r/PromptEngineering/comments/1q7cqj7/the_day_i_stopped_collecting_cool_prompts_and/)** (热度: 141): **这篇文章描述了从收集随机 Prompt 到开发结构化的“标准库” Prompt 模式的转变，类似于软件开发实践。作者强调创建具有明确输入和输出的可重用模式，将糟糕的输出视为“失败的测试（failing tests）”，并加入前置和后置条件以提高 Prompt 的可靠性。这种方法带来了更可预测且可重用的输出，将过程从随机的 Prompt 生成转变为类似于从库中调用函数的系统化方法。作者在[这里](https://allneedshere.blog/prompt-pack.html)分享了他们的库，供他人使用或参考。** 评论者分享了管理 Prompt 库的各种方法，例如利用 ChatGPT 的 System Instructions 处理特定项目的 Prompt，将 Prompt 存储在 Ubuntu 机器上的自定义 Git 库中，以及使用名为 Promptsloth 的 Chrome 扩展程序以便快速访问。

    - kermitt81 描述了一种在 ChatGPT 中通过使用带有 System Instructions 的 “Projects” 来组织 Prompt 的方法。每个项目都针对特定任务量身定制，其中一个项目专门用于即时 Prompt 设计。该项目会分析用户的请求，将其拆解为各个组件，并根据预定义的规则生成详细的 Prompt。这种方法可以创建高度详尽且可用的 Prompt，可能只需要进行细微调整。
    - fakiestfakecrackerg 建议创建一个由相互关联的规则手册和指令组成的复杂系统。这包括将核心 Prompt 编译成一组基础的自定义指令，并使用额外的 Prompt 来构建特定功能的分层框架。这种方法可以通过创建一个结构化且互连的系统来提高 Prompt 使用的效率和效果。

  - **[2 Biggest issues of AI in 2026](https://www.reddit.com/r/PromptEngineering/comments/1q7nlxg/2_biggest_issues_of_ai_in_2026/)** (热度: 24): **这篇文章指出了截至 2026 年 AI 面临的两个主要问题：AI 在缺乏上下文时倾向于填补空白，导致潜在的错误输出；以及人类沟通风格与 AI 对结构化输入需求之间的不匹配。作者认为，AI 经常假设未说明的目标和约束，从而产生看似完美但可能错误的响应。文章指出，许多对 AI 的沮丧根源在于人类无法根据 AI 的结构化需求调整沟通方式，而非 AI 能力不足。作者正在研究一种解决这些问题的工具，可在 [aichat.guide](http://www.aichat.guide) 获取。** 一条热门评论质疑了“人类应该适应 AI 结构化输入需求”的观点，认为 AI 应该演进以处理人类的沟通风格，因为人类沟通本质上是模棱两可且非线性的。评论者建议，虽然 Prompt Engineering 在目前很有用，但长期目标应该是让 AI 更好地理解人类原生的沟通方式，保留人类表达的丰富性。

    - 该评论强调了 AI 发展中的一个关键问题：目前对结构化输入的依赖将适应负担转嫁给了人类，要求人类改变沟通方式以适应 AI 系统。这种方法存在削弱人类表达丰富性的风险，因为它鼓励人们以一种更刻板、类似于 API 的方式进行交流。评论者认为，AI 应该演进以更好地处理人类沟通的细微差别（如歧义和情感），而不是强迫人类适应 AI 的局限性。这一观点表明，AI 理解人类原生沟通的能力对于有效的协作和可扩展性至关重要。

  - **[The more I ‘polish’ a prompt, the worse the output gets. Why?](https://www.reddit.com/r/PromptEngineering/comments/1q71q8k/the_more_i_polish_a_prompt_the_worse_the_output/)** (热度: 24): **这篇文章讨论了针对语言模型进行 Prompt Engineering 时的常见问题，即用更多细节去精炼 Prompt 反而会导致更差的输出。这通常是因为过度约束模型或引入了歧义，当模型试图满足多个意图时会产生困惑。通过专注于核心目的而非添加过多细节来简化 Prompt，往往能获得更好的结果，因为这能让模型更有效地利用其训练数据。** 评论者指出，过于详细的 Prompt 可能会导致“Prompt 疲劳（prompt fatigue）”或“稀释（dilution）”，即模型优先遵循特定规则而非关注内容质量。他们建议使用基于目标的 Prompt 或带有示例的 Few-shot 方法，以保持关注点和创造力。

- Adventurous-Pool6213 强调，过于详细的 prompt 可能会因为引入混杂的意图而使模型感到困惑，从而导致输出效果不佳。他们建议使用基于目标的 prompt，这种方法提供清晰的方向而无需过多的细节，让模型能够有效地填补创意空白。这种方法在视觉模型中特别有效，例如在 [gentube.app](https://www.gentube.app/?_cid=cm) 等工具中可以看到。
- liquiditygod 讨论了“prompt 疲劳（prompt fatigue）”或“稀释（dilution）”的概念，即 prompt 中过多的指令会导致模型专注于遵循规则，而不是产出高质量的内容。他们建议采用 few-shot 方法，通过提供示例而非详细指令，来保持模型对核心目标的关注并提高输出质量。
- PurpleWho 描述了一种类似于测试驱动开发（TDD）的 prompt 开发迭代方法。他们从基础 prompt 开始，针对真实输入进行运行，并将失败的情况捕获为测试用例。这种方法通过解决边缘情况和防止回归来帮助完善 prompt。他们提到了使用 Mind Rig 和 vscode-ai-toolkit 等工具进行测试，并最终导出到 Braintrust 或 PromptFoo 等正式评估工具进行全面分析。

- **[Vibe Coding 并不容易 —— 它只是一条不同的路径](https://www.reddit.com/r/PromptEngineering/comments/1q7pc2o/vibe_coding_isnt_easier_its_just_a_different_path/)** (活跃度: 18): **该帖子认为“vibe coding”——一种强调清晰意图、问题构思和系统思维，而非传统语法和样板代码的方法——并没有降低编码的难度，而是转移了难度。它指出，虽然传统编码侧重于机械执行，但 vibe coding 让开发者专注于解决问题和设计。文中强调了 [Lumra](https://lumra.orionthcomp.tech) 等工具在组织 prompt 和迭代中的作用，从而支持可持续的工作流，而非仅仅作为捷径。** 评论反映了对 vibe coding 难度的怀疑，一些用户认为它更容易，并将该帖子的观点斥为“AI 生成的废话”。另一些人则指出 vibe coding 感觉更像是对话式的，并对工具不遵循风格指南表示沮丧，这表明在实际案例和工具功能之间存在差距。

    - thinkmatt 讨论了与 AI 进行“vibe coding”的对话性质，强调 prompt 很少被重复使用，这与代码复用很常见的传统编码实践形成鲜明对比。他们表达了希望像 Cursor 这样的 AI 工具能更好地遵循预定义的风格指南，这表明目前的 AI 能力在完全融入开发者工作流方面还存在差距。

- **[[R] 为 LLM 研究收集迷因（memes）——提交你的迷因并查看分析！](https://www.reddit.com/r/MachineLearning/comments/1q7aeoy/r_collecting_memes_for_llm_studysubmit_yours_and/)** (活跃度: 20): **来自 **THWS** 和 **CAIRO's NLP Team** 的研究人员正在开发 **MemeQA**，这是一个众包数据集，旨在评估视觉语言模型（VLMs）理解迷因的能力，重点关注幽默、情感映射和文化背景等方面。该数据集将包含每个迷因超过 `10 个维度` 的信息，贡献者可以通过 [memes.thws.ai](http://memes.thws.ai) 提交迷因，以协助为 VLMs 创建一个全面的基准测试。** 评论者对数据集最初仅有 `31 个迷因` 的规模表示担忧，建议通过抓取迷因相关的 subreddit 来获取更多数据。此外，还有人对使用众包数据作为模型的“免费训练数据”表示怀疑。

    - Forsaken-Order-7376 提出了一个关于该研究中标注基准标签（ground truth labels）方法论的技术问题。这对于确保用于训练或评估模型的数据集的准确性和可靠性至关重要。适当的标注对于监督学习任务是必不可少的，因为模型的性能很大程度上取决于标注数据的质量。

- **[[D] 我将关于分子设计的几何深度学习的 4 年博士研究总结为 3 个研究问题](https://www.reddit.com/r/MachineLearning/comments/1q72bd8/d_i_summarized_my_4year_phd_on_geometric_deep/)** (热度: 145): **Chaitanya Joshi** 总结了他关于 *Geometric Deep Learning for Molecular Design* 的博士论文，将其归纳为三个关键研究问题，重点关注 3D 表示的表达能力、周期性和非周期性系统的生成建模，以及功能性 RNA 的真实世界设计。他引入了用于评估表达能力的 *Geometric Weisfeiler-Leman Test*，提出了用于统一生成建模的 *All-atom Diffusion Transformer*，并开发了用于 RNA 设计的 *gRNAde*（已通过湿实验验证）。论文强调了从理论图同构问题到分子生物学实际应用的进展。[阅读更多](https://chaitjo.substack.com/p/phd-thesis-in-three-questions)。评论者对 equivariant models 的未来角色感兴趣，特别是考虑到 scaling 和 data augmentation 的情况下，这些因素如何影响工业界中的模型选择。还有人好奇在 All-atom Diffusion Transformer 等模型中测试 transfer learning 的效果，以及在湿实验验证过程中面临的挑战。此外，还提出了关于初始训练结构的来源以及 X-ray 与 in vivo 结构之间差异的问题。

    - Affectionate-Dot5725 针对 scaling 和 data augmentation 背景下 equivariant models 的作用发起了技术讨论。评论者好奇随着规模的增加和数据增强能力的提升，是否会降低对 equivariant models 的需求，特别是在工业应用中。这反映了关于模型复杂性与大规模数据驱动方法收益之间权衡的更广泛辩论。
    - NoPriorThreat 讨论了获取分子模型初始训练结构的挑战，强调了使用 X-ray crystallography 和 ab initio 方法的局限性。X-ray 结构通常代表“生物学上的冻结”状态，这与 in vivo 条件不同，而 ab initio 方法计算成本高昂，且对于大型系统可能不够准确。这突显了在分子建模中平衡准确性和计算可行性的难度。
    - Affectionate-Dot5725 还询问了模型中 transfer learning 的测试，特别是在最先进的 all-atom diffusion 模型的背景下。该问题集中在如何评估联合训练是否增强了表示学习，这是理解复杂模型中 transfer learning 有效性的关键方面。


---

# AI Discord Recap

> 由 gpt-5.2 生成的摘要的摘要


**1. 新工具与框架发布**

- **Transformers v5 进行大扫除**：Hugging Face 发布了 **Transformers v5**，统一了 tokenizer 后端，将模型定义模块化，专注于 **PyTorch**，并在博客文章 ["Transformers v5"](https://huggingface.co/blog/transformers-v5) 中优先考虑了 quantization 以及新的 serving/inference 功能。
  - 同一波公告还推出了针对 Apple 的客户端工具——["swift-huggingface"](https://huggingface.co/blog/swift-huggingface) 和 ["AnyLanguageModel"](https://huggingface.co/blog/anylanguagemodel)——旨在让 Apple 平台上的 **local+remote LLM access** 感觉像是一个统一的 API。

- **DSPy “改写历史”（这次是好事）**：DSPy 贡献者争论了为什么他们的教程在 system prompt 中包含了 **conversation history** (["DSPy conversation history tutorial"](https://dspy.ai/tutorials/conversation_history))，维护者解释说这是一个可以更改的 **adapter representation detail**。
  - 团队表示他们正在彻底改革 **multi-turn conversations**，预计将于 **本月晚些时候** 发布更新，实际的收获是：编写 **custom adapters** 来控制历史记录的序列化方式，而不影响 optimizers。

- **MCP 希望在 Mutations 之前进行 “Dry-Run” Tool Calls**：MCP 贡献者提议标准化一种在执行前通过 tool calls **暂存 (stage) 变动性操作** 的方法，并询问这是否应该成为一个 [SEP](https://sep.dev)。
  - 其他人反驳说，暂存可能属于 **SDK implementation guidance**（而不是协议更改），同时该小组还重新启动了关于 **W3C WebMCP** 与 MCP 协作的讨论。


**2. 模型发布、基准测试与排行榜**

- **ERNIE 挤进 Vision Arena 前十**：根据 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/)，LM Arena 的 [Vision 排行榜](https://lmarena.ai/leaderboard/vision)更新显示，`**ERNIE-5.0-Preview-1220**` 以 **1226** 的评分位列 **第 8 名**。
  - 更新日志的讨论指出，**百度**目前是 Vision 前十名中唯一的**中国实验室**，这被视为一个值得关注的“谁在交付视觉模型”的信号。

- **Hawk Max 炫技、发布游戏并要求加入 Arena**：LMArena 用户热议 **Movement Labs** 的 **Hawk Max** 模型，声称它可以一次性生成（one-shot）功能完整的 **Minecraft 克隆版**和**象棋游戏**，甚至在某些任务上“超越了 **Claude Opus 4.5**”。
  - 社区明确要求将 **Movementlabs.ai** 加入 Arena 进行基准测试，并将这种讨论定调为“不上排行榜就等于没发生”。

- **Hunyuan-Video-1.5 加入视频排行榜**：LM Arena 将 `**Hunyuan-Video-1.5**` 加入了视频排行榜：在 [Text-to-Video](https://lmarena.ai/leaderboard/text-to-video) 排行榜中位列 **第 18 名**（评分 **1193**），在 [Image-to-Video](https://lmarena.ai/leaderboard/image-to-video) 排行榜中位列 **第 20 名**（评分 **1202**）。
  - 用户被引导至指定频道分享反馈，反映出视频评估目前仍处于“先发布，后校准”的状态。


**3. GPU 训练/Kernel 性能：提速、插件与竞赛**

- **CGGR 实现 1.40 倍加速并瞄准 Triton + H200**：Nous Research 和 Unsloth 社区成员讨论了 [CGGR](https://github.com/MinimaML/CGGR) 的早期基准测试，报告了 **1.40 倍的训练加速**，其中 **forward** 耗时 **127 ms**，**backward** 耗时 **93 ms**。
  - 他们计划在 **H200** 上结合 **Triton** 测试 CGGR，以进一步提升速度并减少 **VRAM** 占用（从而支持更大的 batch sizes）；同时更广泛的基础设施讨论指出，目前配置下的开源 MoE 训练 MFU 仍维持在 **~4%** 左右。

- **Triton 插件基础设施发布……代码曾被隐藏**：GPU MODE 分享了关于 *triton-shared* 更新和 **Triton 插件基础设施**（[视频](https://youtu.be/JnFFwBB6Dhk)）的 YouTube 录播，引发开发者寻找插件源码。
  - 有人发现演示链接有误，频道随后将其修正为 [triton-lang/triton `lib/Plugins`](https://github.com/triton-lang/triton/tree/main/lib/Plugins)，为试图阅读代码的开发者扫清了障碍。

- **Flex Attention 引入 Cute——速度提升 30%**：GPU MODE 用户报告称集成 **CuteDSL flex attention** 后，在 **H100 forward** 上比基础版 flex attention 提升了 **~30% 的吞吐量**。
  - 他们还跟踪了后端差距（例如 SM90 backward 支持），并指出了通过 [flash-attention PR #2137](https://github.com/Dao-AILab/flash-attention/pull/2137) 进行的持续上游工作。


**4. 数据集与小模型训练（从零训练 > 微调？）**

- **Tiny LLM 预训练：10–50M 参数，全权掌控，无需“对抗权重”**：Unsloth 成员讨论了预训练 **Tiny LLM**（约 **10–50M 参数**），并分享了涵盖 **2,700 个通用主题**的数据集 ["TinyLLMPretrainingCore"](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore)。
  - 这种做法的动力不仅在于节省算力——人们表示微调感觉像是在“与现有权重搏斗”，而从零开始的预训练（scratch pretraining）即使模型规模较小，也能恢复对**数据和行为的控制**。

- **SOC “黄金数据集”发布：CyberSec-CoT-v1 (MIT)**：一位贡献者发布了一个包含 **580 行合成数据**的 SOC 事件响应数据集——["BlackBox-CyberSec-CoT-v1"](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1)——由 **Llama-3-70B** 生成，并明确采用 MIT 许可。
  - 他们将其定位为评估 **JSON-schema 遵循能力**和推理步骤（侧重于引导逻辑而非原始日志）的**黄金数据集（Golden Set）**，并将此次发布视为一种“诚意的表现”，而非单纯的商业售卖。

- **BCI Timelapse 数据集试图降低大脑数据获取成本**：Hugging Face 用户分享了 ["DATASTRIKE BCI Timelapse Dataset"](https://huggingface.co/datasets/webxos/datastrike_BCI) 以及一段 [YouTube 短视频](https://www.youtube.com/shorts/UxV0e7J5gTs)，将其宣传为一种在没有大规模真实 BCI 硬件数据集的情况下训练神经信号解码模型的方法。
  - 整体氛围是“合成/替代数据流水线正向文本之外的领域扩展”——BCI 加入了日益增长的领域数据集行列，旨在无需昂贵的数据采集即可引导研究。


**5. Agent 与开发体验：记忆、文件与可靠性**

- **Agent 记忆：RAG 有帮助，但强塞 Prompt 会造成损害**：在 OpenAI 的 Discord 中，开发者询问如何在不持续注入巨大上下文块的情况下，跨多个基于角色的 Agent 保持 **Agent 身份和上下文**。讨论集中在利用持久化记忆工具和 **RAG**。
  - 实际的矛盾在于成本/延迟与正确性：人们希望拥有**始终可见的记忆**，而又不愿在每一轮对话中支付 Token 税，目前大家仍在探索不会将每次运行都变成 Prompt 巨型文件的模式。

- **LM Arena 希望增加文件上传功能（复制/粘贴在移动端体验极差）**：LMArena 用户请求向模块**发送文件**的能力，因为长内容在复制/粘贴时会被截断，尤其是在移动设备上。
  - 与此同时，稳定性投诉也接踵而至（例如发送图片后 **Gemini Pro 3** 出错），这进一步证明了“简单的 I/O 易用性”和“会话可靠性”现在已成为评估平台的入门门槛。

- **Cursor Agents 在执行命令中途崩溃并遗忘规则**：Cursor 社区用户报告称，Agent 对话在终端命令（如 *npm run dev*）执行中途断开，怀疑这与 [序列化错误（serialization errors）](https://forum.cursor.com/t/serialization-error/124671/186)有关；此外还有一个独立问题，即 **commit rules**（提交规则）虽然在设置中显示，但无法加载到 Agent 中。
  - 另一个棘手问题是：打开一个空窗口，开始 Agent 对话，然后打开一个文件夹，这会产生一个新窗口并**销毁对话**，使得在更改项目上下文时，Agent 工作流显得非常脆弱。


---

# Discord: 高层级 Discord 摘要




## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **LeakHub 推出众包 Prompt 库**：[LeakHub](https://leakhub.ai) 作为一个**众包系统提示词（sys prompt）库**和验证平台出现，旨在通过新技术简化验证泄露（leaks）的过程。
   - 该平台旨在通过利用**众包方法**来简化验证过程，为**经过验证的提示词（verified prompts）**提供集中场所，并鼓励用户提交并**验证泄露**以提升在排行榜上的排名。
- **GamersNexus 认可 DLSS 优于 AI 生成帧**：成员们讨论了 **DLSS** 实际上优于 **AI 生成帧（AI generated frames）**，并引用了 [GamersNexus](https://www.youtube.com/@GamersNexus) 及其内容。
   - 一位成员表示，*Intel 知道他们正在迎合 AI 群体，而这些人本身并不是最聪明或最懂技术的人*。
- **利用种族主义教科书进行煤气灯效应（Gaslighting）以实现越狱**：一位成员描述了一种越狱方法论，包括使用**种族主义教科书**，然后对 AI 进行“煤气灯效应”式引导，以使用 **SuperGrok** 获取针对争议话题的理想输出。
   - 另一位成员证实采用了类似的方法，包括给 AI 禁忌内容进行分析、改进，然后引导它写出为什么这些内容是必要且有益的。
- **AI 反作弊系统检测非人运动**：成员们提到，有人正在开发 **AI 反作弊系统**，该系统会仔细检查每个像素并寻找非人类的运动轨迹，这些系统成本高昂，目前尚未广泛采用。
   - 另一位成员解释说，CV（计算机视觉）作弊完全绕过了 ring0 检测，可能使用了带有网络摄像头的 **DMA2 PC** 设置来监视屏幕。
- **冲突的系统消息暴露越狱漏洞**：一位成员发现，由于某个使用 Grok 的 NSFW 网站的系统消息与其自身的安全策略相矛盾，导致该网站上的 AI 伴侣更容易被越狱，这表明**冲突的指令**会削弱 AI 的安全防护。
   - 他们解释说，Grok 模型对 NSFW 内容的回避与网站托管此类内容的性质相冲突，导致模型在编写代码或解释非常规话题时变得更加顺从（pliable）。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **ChatGPT 的语言模式解码**：用户发现 **ChatGPT** 的语言模式具有可预测性，将其归因于 **OpenAI** 的刻意工程设计或模型层级的局限性，产生了一种“填空感”，听起来像是在填充模板。
   - **GPT-5.2** 的特定言谈举止被视为进一步展示了“沟通风格可以变得多么令人厌恶”。
- **小型 LLM 预训练受到关注**：一名成员正在预训练一个约 **1000 万至 5000 万参数**的小型 LLM，以探索小型基座模型的能力，并分享了一个包含 [2700 个通用主题的数据集](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore)链接。
   - 另一名成员表达了对微调预训练模型的挫败感，渴望完全的控制权，感觉自己总是在“与现有的权重搏斗”。
- **开源网络安全数据集发布**：一名成员发布了一个包含 [580 行的合成数据集](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1)（MIT 许可），该数据集通过 **Llama-3-70B** 生成，专注于 SOC 事件响应的 CoT 逻辑，以示诚意。
   - 该数据集被设计为评估模型遵循 JSON schema 和推理步骤能力的 **Golden Set**（黄金集），旨在比仅靠原始日志更有效地引导模型逻辑。
- **Ollama 疑似推行云服务引发争论**：一名遇到 **Ollama** 问题的用户被建议考虑 **llama.cpp** 或 **LMStudio** 等替代方案，因为担心 **Ollama** 可能正在推广云服务，且主要问题在于 *Ollama 使用的是过时的 llama.cpp*，导致性能可能较差且控制力不如直接使用 **llama.cpp**。
   - 有人指出 *Ollama 在云服务方面动作频频*，这与用户最初因其与 Meta 的关联而认为 **Ollama** 是注重隐私的解决方案的印象形成对比。
- **高中生开拓 CGGR 和 SRDE 研究**：Wilba 是一位 **16 岁**的少年，正在进行独立的 AI 研究，专注于 **CGGR** ([https://github.com/MinimaML/CGGR](https://github.com/MinimaML/CGGR)) 和 **SRDE** ([https://github.com/MinimaML/srde-mistral](https://github.com/MinimaML/srde-mistral))。
   - 他们渴望与 Unsloth AI 社区建立联系以扩展知识。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **文件发送功能备受关注**：用户请求向模块发送文件的功能，强调大文件在复制粘贴时会被截断，特别是在移动设备上。
   - 这一改进将简化在平台内直接共享大型数据集和代码片段的过程。
- **Gemini Pro 3 遭遇故障**：一名用户报告在发送图片后 **Gemini Pro 3** 出现错误，且问题在不同浏览器中持续存在，可能是由于关闭了 **LM Arena** 标签页导致的。
   - 这些错误可能表明在处理图像输入或 **LM Arena** 中的持久会话时存在稳定性问题。
- **语音克隆工具达到新高度**：一名用户重点推介了语音克隆工具 [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)，指出它非常有效，且“在合适的 seeds 下真假难辨”。
   - 该工具展示了语音合成领域的进步，可能影响内容创作和无障碍应用。
- **Hawk Max 生成功能性游戏**：用户讨论了来自 **Movement Labs** 的新模型 **Hawk Max**，声称它在某些任务上优于 **Claude Opus 4.5**，并能一次性生成可运行的 **Minecraft 克隆版**和**象棋游戏**。
   - 有人建议将 **Movementlabs.ai** 加入 arena，反映出对其能力基准测试的兴趣。
- **ERNIE-5.0 晋升 Vision Arena**：`ERNIE-5.0-Preview-1220` 目前在 [Vision Arena 排行榜](https://lmarena.ai/leaderboard/vision)上排名 **第 8**，得分为 **1226**。
   - **Baidu** 目前是 Vision 排行榜前 10 名中唯一的中国实验室，标志着其在该领域的存在，详见 [Leaderboard Changelog](https://news.lmarena.ai/leaderboard-changelog/)。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **RAM 获封头号瓶颈，粉碎模型运行梦想**：**LM Studio** Discord 的成员讨论指出，运行模型时 **RAM** 是主要的限制因素；一个 **30B 模型** 至少需要 **64GB 的 RAM**。
   - 他们建议使用 **Llama 3.2 3B** 以获得良好的质量，并研究了 quant levels（量化级别）和压缩技术。
- **Llama 乏力，Qwen 疾步**：一位 **LM Studio** 用户表示，自 2024 年以来 **Llama** 已经走到了尽头，声称 **Meta** 已停止了其积极开发。
   - 该用户认为 **Qwen 3 4b** 目前是端侧模型（edge models）中的佼佼者。
- **VRAM 困扰 LLM 愿景**：**hardware-discussion** 频道的用户指出，除非你拥有 **EPYC** 或 **Threadripper**，否则 **VRAM** 是 LLM 的唯一限制因素。
   - 建议的模型包括 **glm-4.5-air**、**Qwen-next** 和 **GPT-OSS**，运行速度在 **4-10 t/s** 之间。
- **Nvidia 否决新 GPU，转向 AI**：有关 **RTX 50 Super** 发布的消息已被平息，[Nvidia 在 CES 上未宣布任何新 GPU](https://www.tomshardware.com/pc-components/gpus/for-the-first-time-in-5-years-nvidia-will-not-announce-any-new-gpus-at-ces-company-quashes-rtx-50-super-rumors-as-ai-expected-to-take-center-stage)。
   - 这是五年来 Nvidia 首次未在 CES 上发布新 GPU，标志着其重心向 AI 转移。
- **Cline 夺得代码助手桂冠**：**hardware-discussion** 频道的用户讨论了各种 VS Code AI 助手，一位成员推荐了 **Cline**，强调 *kilo 是唯一能正常工作的*。
   - 另一位成员指出，AI 代码助手能够*自动将任务分解为待办步骤并向用户反向提问，这非常棒*，而另一位用户则强调 **Cline** 在大约 **50K context** 之后往往会失控。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 窗口丢失 Agent 聊天**：打开一个空白的 **Cursor 窗口**，启动与 Agent 的对话，然后在同一窗口中打开一个文件夹会开启一个新窗口，导致用户丢失之前的 Agent 聊天记录。
   - 用户正在寻求一种在窗口切换时保留 Agent 聊天的解决方案。
- **Agent 在命令中途卡死**：用户报告称 **Cursor 聊天**在命令中途（特别是运行 *npm run dev* 等终端命令时）会中断，需要重新开启聊天。
   - 根据社区用户的反馈，这可能与 [serialization errors](https://forum.cursor.com/t/serialization-error/124671/186)（序列化错误）有关。
- **交互规则：Commit 规则失踪**：一位用户报告称，尽管设置中显示了 **commit rules**（提交规则），但它们并未在 Agent 中自动加载。
   - 社区正在调查为什么这些规则没有按预期生效。
- **认证大对决：Better Auth vs Neon Auth**：用户正在辩论 **Better Auth** 与 **Neon Auth** 的优劣，一位用户指出 Better Auth 太新且缺失功能，例如多租户基础 URL 设置。
   - 讨论围绕每种身份验证方法的成熟度和功能完整性展开。
- **Opus 4.5 是否物有所值？**：成员们正在讨论 **Opus 4.5** 的性价比，一位用户表示，即使考虑到去年使用了 *15b tokens*，他们仍觉得*值得*。
   - 另一位成员建议使用 **composer-1** 进行执行，而另一位则开玩笑说他们需要别人的信用卡来对其进行测试。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **Blockchain 讨论提议屏蔽虚假信息**：一名成员提议屏蔽 *"blockchain"* 一词，因为利用 discordapp.com（而非合法的 discord.com 链接）进行的**诈骗企图**日益增多。
   - 该建议强调了持续存在的欺诈活动，尽管目前尚未采取立即行动。
- **连接灾难导致 Cloudflare 瘫痪**：成员们报告了反复出现的 *"connection refused"* 错误，由于 **Cloudflare 问题**以及连接保持开放，导致近 **2%** 的请求失败。
   - 一名用户实施了 **18 秒超时**和重试策略，并寻求 OpenRouter 的进一步支持，以解决这一持久的连接问题。
- **skill.md 凭借文档能力脱颖而出**：一名成员称赞 **skill.md** 在文档编写方面优于 **MCP (JSON-RPC)**，强调 *"skill.md 的核心在于编写高质量的文档！"*。
   - 他们强调了其在**动态工具检索**和集成方面的潜力，并指出在配合 Godot 等工具时，这是一种 *"更酷的技能"*。
- **Gemini 在游戏视觉领域取得进展**：成员们正在探索利用**视觉 AI models** 来分析游戏结算画面，发现 **Gemini models**（尤其是 **Gemini 3 Flash**）在评估奖励方面极具潜力。
   - 目前仍面临缺乏文本标签的小图标识别挑战，对此有人提出了参考网格（reference grids）和缓存策略以降低成本，或者使用小型 VL model 的建议。
- **Qwen3 加速，查询质量备受关注**：在分享了具有多模态嵌入能力的 [阿里巴巴 Qwen model](https://x.com/Alibaba_Qwen/status/2009264754917863924) 链接后，一些聊天机器人评论称 **Qwen3 的构建方式与众不同**。
   - 部分人持有怀疑态度，而另一些人则认为这种方法可以提升 Agent 的性能。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **报告显示 AI 在医疗领域的应用翻倍**：根据 [OpenAI 的公告](https://openai.com/index/openai-for-healthcare/)，随着支持 HIPAA 的工具 **OpenAI for Healthcare** 的推出，医生对 **AI** 的使用在一年内几乎翻了一番。
   - **OpenAI for Healthcare** 现已在 AdventHealth、Baylor Scott & White、UCSF、Cedars-Sinai、HCA、Memorial Sloan Kettering 等机构上线，旨在协助*为患者提供更一致、高质量的护理服务*。
- **桌游被前沿技术彻底改变！**：成员们讨论了使用 **AI** 为**棋盘游戏**、**桌面游戏**和**卡牌游戏**创作内容（包括核心机制和美术），其背景是 **Hasbro 放弃了功能性设计**。
   - 一名成员分享了为卡牌游戏生成的[概念图](https://drinkoblog.weebly.com/)，包括边框和规则文本。
- **Agent 失忆：AI 身份遭遇“身份窃取”**：一名成员寻求关于**处理 Agent 身份**和**上下文持久化**的建议，需要为不同角色的多个 Agent 持久化预定义的上下文。
   - 建议包括使用一种工具来存储 Agent 始终可见的记忆，以及采用 **RAG** (Retrieval-Augmented Generation)，尽管也有人担心不断包含大型文本块会导致效率低下。
- **Google Search AI 让 Gemini 3 Pro 败下阵来？**：成员们辩论了 **Google Search AI** 和 **Gemini 3 Pro** 哪个更好，一位用户认为 Search model 在某些任务上更胜一筹，特别是在查找和引用来源方面。
   - 其他人则认为，Search AI 无法与 **Gemini** 这种 LLM 的推理和综合能力相提并论，因为后者旨在保持上下文并提供连贯的输出。
- **Anthropic 的 AI 安全：是否与国防有关？**：一名成员指出，**Anthropic 的 AI 安全报告**和**白皮书**读起来像军事采购手册，强调了 model 的可控性和目标性。
   - 另一名成员表示赞同，认为 **Anthropic** 给人一种“国防承包商”的既视感，并指出其与[美国国防部签署的 2 亿美元合同](https://www.defense.gov/News/Releases/Release/Article/3594261/dod-announces-awards-for-prototype-artificial-intelligence-capabilities/)证明了其对政府应用的关注。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Transformers v5 来了！**：**Transformers v5** 统一了 tokenizer 后端，实现了模型定义的模块化，专注于 **PyTorch**，并通过[新的推理/服务特性](https://huggingface.co/blog/transformers-v5)优先支持量化。
   - 此次更新旨在简化模型定义并优化 AI 生态系统。
- **Swift Client 进驻 Hugging Face**：Hugging Face 的 **Swift Client** [swift-huggingface](https://huggingface.co/blog/swift-huggingface) 正式发布。
   - 此外，[AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel) 为 **Apple 平台**上的本地和远程 LLM 提供了“统一的 API”。
- **Madlab 的 SDG 模型上线！**：**Madlab** 发布了全新的旗舰级合成数据生成器（SDG），专为符合规则且语义连贯的变体生成而构建，包括适用于高质量 SDG 工作流的 **LFM2.5** 和 **LFM2** 模型，并感谢 [LiquidAI 的杰出工作](https://huggingface.co/MadlabOSS/LFM2.5-1.2B-Instruct-SDG)。
   - 一位成员询问其他用户是否对竞争性的合成数据生成挑战感兴趣，该挑战将由 LLM 和独立评审团评判，让付费解决方案与 **Madlab** 的开源 SDG 流水线进行对决。
- **BCI 数据集突袭！**：发布了全新的 **DATASTRIKE BCI Timelapse 数据集**，旨在训练无需大规模真实硬件 BCI 数据集的神经信号解码机器学习模型；相关链接包括 [YouTube short](https://www.youtube.com/shorts/UxV0e7J5gTs) 和 [HuggingFace 上的数据集](https://huggingface.co/datasets/webxos/datastrike_BCI)。
   - 该数据集旨在无需真实硬件即可进行神经信号解码的开发。
- **VeridisQuo 揭露 Deepfake 元凶**：一位用户发布了 **VeridisQuo**，这是一个开源的 Deepfake 检测器，它使用 GradCAM 热力图通过空间分析、频率分析和可解释 AI 可视化来精确显示视频被篡改的位置；分享了 [GitHub 仓库](https://github.com/VeridisQuo-orga/VeridisQuo) 和 [Hugging Face Spaces 上的 Demo](https://huggingface.co/spaces/Gazeux33/veridisquo-deepfake-detection)。
   - 该工具可以暴露 Deepfake 并突出显示被篡改的区域。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **CGGR 基准测试达到超音速**：[CGGR](https://github.com/MinimaML/CGGR) 的初步基准测试显示训练速度提升了 **1.40倍**，前向传递达到 **127 ms**，反向传递达到 **93 ms**。
   - 未来计划包括在 **H200** 系统上使用 **Triton** 进行测试，旨在提高速度并减少 VRAM 占用，从而显著增加 batch size。
- **Nous 放弃 MoE 转向稠密训练**：由于基础设施未优化且成本高昂，**Nous Research** 目前更倾向于稠密模型（dense models）而非 **Mixture of Experts (MoE)** 模型。
   - 尽管最近进行了基础设施优化，但目前用于 **MoE** 的**最先进开源训练基础设施**仅产生约 **4% 的 MFU**（模型 FLOPs 利用率）。
- **扩散模型可能生成更好的笑话**：成员们理论上认为，**扩散模型（diffusion models）** 可能通过隐含地了解笑话的梗（punchline）来生成更好的笑话，且可能不需要显式的规划 ([arxiv.org/abs/2511.08923](https://arxiv.org/abs/2511.08923))。
   - 虽然有人建议**规划输出（planning the output）**可以达到类似的效果，但前者反驳说扩散模型可能更快且消耗更少的 token。
- **Llama 3.3 8B 遭遇“脑叶切除术”**：一些成员表示 **Llama 3.3 8B Instruct** 模型在多语言任务中表现得像被“阉割”了一样（性能大幅下降）。
   - 一位成员推测 **Meta** 可能在基准测试中造假，并可能正在逐渐脱离 **Llama**。
- **HoloLM 变得莎士比亚化**：作者分享了 [Experiment58-HoloLM](https://github.com/jackangel/Experiment58-HoloLM)，这是一个在 **5MB 莎士比亚语料库**上仅用几小时训练出来的小型 LM。
   - 作者的目标是创建一个可以在笔记本电脑 GPU 上训练并拥有超大上下文的模型。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **RTX 5050 被推测为 'Tiny Blackwell'**：成员们推测了被称为 *'tiny Blackwell'* 卡的 **RTX 5050** 的性能，确认其具有 **12** 的计算能力（compute capability）并支持 CUDA 12.0。
   - 他们警告说，**RTX Blackwell** (sm_120) 与**数据中心级 Blackwell** (sm_10x) 有所不同，并建议它可能不足以用于验证高效利用 **B200** Tensor Core 的代码。
- **CUDA Rust 绕过 LLVM PTX 后端**：一位成员指出，尽管 **CUDA Rust** 并非 Nvidia 官方支持，但它针对的是 **NVVM** 而非 **LLVM 的 PTX 后端**，并引用了 [Rust-CUDA FAQ](https://rust-gpu.github.io/rust-cuda/faq.html#why-not-use-rustc-with-the-llvm-ptx-backend)。
   - 这种方法允许 **CUDA Rust** 在针对 **Nvidia GPU** 时，避开与 **LLVM PTX 后端** 相关的复杂性和限制。
- **CuteDSL Flex Attention 提升速度**：一位成员集成了 **CuteDSL Flex Attention 实现**，并指出它通过不同的 mask mod 加速了常规的 Flex Attention，在 **H100 前向 (fwd)** 计算上比基础版 Flex Attention 实现了 **约 30% 的吞吐量提升**。
   - 速度的提升节省了资源。
- **GPUMODE Runner 陷入停滞**：一位用户报告在 GPUMODE 中遇到 **Runner 变慢**和超时的问题，并提供了 [示例 ID 297869](https://cdn.discordapp.com/attachments/1434709259500650628/1458664934706511922/Screenshot_2026-01-07_at_7.30.00_PM.png?ex=69611fd5&is=695fce55&hm=469fea089e64c3e7f89bfccbb4ec99c67c790f09c4a8f399796cc7627394cfd5&)。
   - 该用户遇到了 `DSLCudaRuntimeError`，虽然基准测试运行正常，但排行榜提交似乎运行在 `test` 模式下，导致了困惑和超时问题，并提出了疑问：*“test、benchmark 和 leaderboard 之间有什么区别？”*
- **GPUMODE 竞赛被 AI Bot 横扫**：一位参赛者确认，他们凭借 **100% AI 生成**的提交内容在问题 2 中排名 **#4**，该提交使用了 LLM Agent，没有任何手写的 GPU 算子（operators）。
   - 他们还确认正尝试仅使用开源模型。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 现在只是个开发者？**：[卫报](https://www.theguardian.com/technology/2026/jan/07/ai-anthropic-funding-valuation) 将 **OpenAI** 称为 *开发者*，引发了关于发表研究是否是研究机构定义特征的争论。
   - 一位成员评论道：*“作为一个研究者，难道不意味着你应该发表你的研究吗？”*
- **LMArena 备受指责**：一篇批评 **LMArena** 对 AI 进步有害的 [博客文章](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai) 引发了关于其相关性的争论，尽管最近有关于他们融资的消息。
   - 虽然有些人认为它已经过时，但其他人指出模型公司似乎仍然在意它，将其作为展示实力和讨论的平台。
- **Mercor AI 侵入式的招聘体验**：**Sasha Kaletsky** 在 [X](https://xcancel.com/sashakaletsky/status/2008904526720286970?s=46) 上描述了 **Mercor** 由 AI 驱动的招聘流程，涉及令人印象深刻的 **AI 面试**和**自动化匹配**。
   - 然而，该过程要求安装侵入式的监控软件（**Insightful**）来记录活动以用于 **RL 模型训练**，导致候选人退出。
- **Autonomous 为 AI 财务导师筹集种子轮资金**：**Dillon Erb** 宣布推出 [Autonomous](https://xcancel.com/dlnrb/status/2009008876834922949?s=46)，这是一家提供 0% 咨询费服务的 **AI 驱动财务顾问**。
   - 该公司获得了由 **Y Combinator** 的 **Garry Tan** 领投的 **1500 万美元**融资，并正在 **纽约市**和**旧金山**积极招聘。
- **Protege AI 为数据基础设施筹集 3000 万美元**：根据其 [公告](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46)，**Protege AI** 宣布了由 **a16z** 领投的 **3000 万美元**融资，用于扩展其 AI 开发的数据基础设施。
   - 成员们正在讨论是否出现了过多的数据公司。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi K2 的创意写作能力**：一位用户分享到，与其他中文模型相比，**Kimi K2** 在**创意写作**和通用对话方面表现出色，并在 [EQ bench](https://eqbench.com/) 上获得了领先评分。
   - 他们强调了该模型在创作引人入胜的叙事和进行细腻讨论方面的卓越能力。
- **Kimi 的“思考”模式引发辩论**：**Kimi K2 “思考”版本**的实用性引发了热烈讨论，一位用户将其性能等同于 **GPT-5.2** 的能力。
   - 相反，另一位用户表达了强烈不满，认为它在处理常规任务时*极其愚笨（dumb as hell）*，表明其在不同用例中的表现差异显著。
- **Kimi K2 过度搜索的倾向**：一名成员报告称 **Kimi K2** 会过度进行搜索，甚至对于像 *1 plus 1* 这样的基础任务也是如此，且英文搜索质量较低。
   - 这种行为引发了对效率以及模型在无需不必要外部查询的情况下处理简单查询能力的担忧。
- **Kimi K2 幻灯片生成故障**：一位用户在遇到 **Kimi K2 幻灯片生成**问题时，最初被提示需要升级订阅功能。
   - 尽管问题随后自行解决，但这反映了该平台订阅功能中潜在的不稳定性或 Bug。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 的 MAX 动力需要手动实现反向传播**：成员们报告称 **Mojo** 缺乏训练库，因此必须使用 **MAX** 并手动实现 **backprop**（反向传播），并建议由于 **Mojo** 的局限性，使用 **sqlite** 进行数据存储。
   - 他们表示 [主仓库](https://github.com/modular/modular) 和 [官方文档](https://docs.modular.com/mojo/manual/) 将对编写自定义训练循环有所帮助。
- **新 Mojo 开发者被错误的 Buns 仓库困扰**：使用来自 [/github.com/BunsDev/mojo-lang/](https://github.com/BunsDev/mojo-lang/)（**已过时 2 年**）的陈旧 **Mojo** 文档的新程序员被建议参考 [主仓库](https://github.com/modular/modular) 和 [官方文档](https://docs.modular.com/mojo/manual/)。
   - 资深成员建议初学者先学习 **C** 或 **Python**，理由是 **Mojo** 频繁发生破坏性变更（breaking changes），且文档假设读者已具备 **Python + C++ 或 Rust** 的知识。
- **缺失 Mojo SVD 引发寻找**：一名成员寻求在 **Mojo** 中使用 **Lanczos/Krylov 算法**实现 **Singular Value Decomposition (SVD)**，但未能找到，并注意到其在 [Mojo roadmap](https://docs.modular.com/mojo/roadmap/) 中缺失。
   - 另一位正在 **Mojo** 中构建 **Tensor Network library** 的成员由于时间限制，正利用 **C ABI** 调用 **LAPACK** 来实现 **SVD**，并表示有兴趣为 **MAX** 贡献相关实现。
- **TEI 在 Embeddings 测试中优于 Max？**：一位成员观察到，与 [TEI](https://github.com/huggingface/text-embeddings-inference) 相比，使用 **max** 生成嵌入向量的速度显著变慢。在实现 `sentence-transformers/all-MiniLM-L6-v2` 时，**max** 的速度为 **727.1 embeddings/sec**，P95 延迟为 **28375.1 ms**，而 **TEI** 为 **8000 embeddings/sec**。
   - 他们正在 **Nvidia RTX 2000 Ada GPU** 上进行测试，并提供了其实现 `all-MiniLM-L6-v2` 模型架构的 Feature 分支 Fork 链接：[RWayne93/modular](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm)。
- **BERT 蓝图呼唤 Max 构建**：一名成员强调了 **MAX** 中缺失 **BERT architecture**，并鼓励社区贡献，建议提交 PR 进行审查，并邀请性能分析（profiling）专家协助诊断性能问题。
   - 有成员表示有兴趣将自定义架构贡献回 MAX，以便进行审查和集成。



---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **开源 Manus：一个新的前沿？**：一位成员提议开源旧版本的 **Manus**，用于教育和社区驱动的改进。
   - 这可能使寻求本地使用选项而不依赖云端访问的企业用户受益。
- **AI 工程师携 LLM 专长加入讨论**：一位专注于工作流自动化和 **LLM integration** 的 AI 工程师介绍了自己。
   - 他们使用 **DSPy**、**OpenAI APIs** 和自定义 **Agent** 构建了系统，并提到通过将 **Slack**、**Notion**、内部 API 连接到 **LLM**，使响应时间缩短了 **60%**。
- **关于创业抵扣额（Startup Credits）的咨询**：一位成员询问了 **Manus Startup Credit** 计划的申请流程和成功率。
   - 该问题在频道中尚未得到解答。
- **Manus 网站协作难题**：一位成员询问是否可以通过独立的对话，对同一个由 **Manus** 创建的网站进行协作开发。
   - 该查询在频道内未收到回复。
- **邮件通知故障排查**：一位成员报告了漏收 **Manus** 团队邮件的问题。
   - 上下文暗示平台通知或更新可能存在潜在问题。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **社区聚焦实时 Diffusion**：“社区聚焦（Community Spotlight）”演讲系列回归，一位成员展示了在 **RTX 6000** 上实时运行的**基于 Diffusion 的世界模型**，详见[此视频](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=696166d4&is=69601554&hm=6d4445d8ccb0f0a6262d3e5450a39bcb5333ef8d1cf63127443f61d7b9593158&)。
   - 未来的聚焦演讲将包括来自 **Common Crawl** 的讲者，讨论他们在大规模 **LangID** 方面的工作及其面临的挑战。
- **消费级 GPU 训练 100M 模型**：成员们探讨了使用 **100GB 数据集**训练 **1 亿参数模型（100 million parameter model）**的经济型方案，并推荐了 **VastAI** 和 **Runpod** 等平台。
   - 由于该设置不受通信带宽限制（comms bound），使用 **1-8x 消费级 GPU 配置（4090, 5090）**即可满足需求，相比服务器级 GPU 可节省大量成本。
- **随机网络受“似是而非的解释”困扰**：最近的一篇 [论文](https://arxiv.org/abs/2512.18792) 强调了 AI 可解释性方法中普遍存在的 **“死三文鱼（dead salmon）”伪像**，质疑了从随机初始化的神经网络中得出解释的有效性。
   - 该[研究](https://arxiv.org/abs/2512.18792)表明，目前的解释性工具即使应用于随机初始化的网络，也可能产生*具有误导性的连贯解释*，这些技术包括**特征归因（feature attribution）、探测（probing）、稀疏自动编码（sparse auto-encoding）和因果分析**。
- **通过在 Base Model 上训练来简化 RL**：一位成员建议研究比 **RL** 更简单的子领域，例如 **Base Model** 训练，因为这样错误更容易诊断。
   - 另一位成员提到 **hard tokens** *不会计算大部分反向传播（backward pass），它执行两次前向传播，然后丢弃易标记（easy tokens）的梯度，仅计算 hard tokens 的梯度*，从而节省显存（VRAM）和计算资源。



---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **ChatGPT 现在成医生了？**: **OpenAI** 推出了 [ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/)，这是一个补充工具，通过引用真实文档来验证医疗信息，可能有助于及早发现疾病。
   - 人们对用户隐私、**ChatGPT** 成为 *everything app 垄断* 以及个人取代医生滥用工具等问题表示担忧。
- **AI 先驱面临剽窃指控**: 获奖者 (**Bengio**, **LeCun**, **Hinton**) 被指控在未注明原始创作者的情况下，多次重新发布关键 **AI techniques**，详见报告 [NOB][DLP][CN25]。
   - 报告指称他们 *没有发明现代 AI 的任何基础算法*，并引用了一份名为《剽窃诺贝尔奖》(A Nobel Prize for Plagiarism) 的技术报告。
- **Grok 的冷幽默？**: 成员们推测 **Grok** 吹嘘其击杀数 (kill count)，并联想到 **AI** 在医疗保健中使用可能导致的潜在死亡事故。
   - 分享了一个 [关于与聊天机器人相关的死亡事件的维基百科页面](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots)，引发了关于聊天机器人相关致死事件的黑色幽默。
- **暂停倡导者暂停活动**: 一位成员分享了一位 *暂停倡导者* 的 [YouTube 视频](https://youtu.be/-qWFq2aF8ZU)，并指出其内容已停更一年。
   - 该倡导者缺乏更新引发了人们对 AI 安全运动现状和焦点的质疑。
- **游说者之间的战争**: 一位成员注意到 **Huawei** 与 **US natsec hawks** 之间的一场 *游说战争*，后者正联手对抗 **Nvidia** 和 **China cloud**。
   - 上下文中未提供更多细节。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 中暴露的对话历史**: 一位成员对 [DSPy 对话历史教程](https://dspy.ai/tutorials/conversation_history) 中 **system prompt** 包含历史记录的做法提出质疑。
   - 另一位成员澄清说 *这只是适配器表示历史记录的方式*，可以编写 **custom adapters** 来更改此行为而不影响优化器。
- **适配器成为 DSPy 的万用工具**: 成员们确认完全可以编写 **custom adapters** 来修改 DSPy 处理历史记录的方式。
   - 一位成员指出，*它对模型的展示方式具有误导性*，但历史记录重写和多轮对话正在彻底重构，预计本月晚些时候发布更新。
- **DSPy 重写历史——值得庆祝的事**: 团队正致力于重写 **multi-turn conversations** 的处理方式，预计本月晚些时候会有变动。
   - 一位成员幽默地评论说，这是 *我们唯一一次庆祝重写历史是一件好事*，并将其解读为 **RLM PR** 即将到来。
- **ColBERTv2 在 topk 中抛出 KeyError**: 一位成员报告在运行文档中的代码片段时，使用 **dspy v3.1.0** 和 **ColBERTv2** 出现了 **KeyError: 'topk'**。
   - 该代码片段使用 **dspy.ChainOfThought** 从问题和提供的上下文中检索回复。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 调度器悬赏待领！**: 一个可能用线性化器 (linearizer) 替换调度器并保持 **GPU speed** 的 [PR](https://github.com/tinygrad/tinygrad/pull/13780) 正在等待悬赏领取。
   - 索赔人建议提交一个有效的 PR 比“扣留”工作更重要，并建议分享奖励。
- **Tinygrad 速度悬赏渴望新贡献者**: 一位成员寻求关于参与 **Tinygrad speed bounties** 的指导。
   - 他们还建议建立一种机制，以申请访问 **Tinygrad** 实例进行测试。
- **AMD Radeon RX 9070XT 的 VFIO=1 苦恼**: 一位成员报告了在配备 **AMD Radeon RX 9070XT** 的 Linux 笔记本电脑上使用 **VFIO=1** 时出现的错误，并提供了 [完整错误日志](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=6961aa3f&is=696058bf&hm=03f6e0c3af31072eccac359044bad6439cf0c8b9f1665e3a9ae7bfc0b6130c73)。
   - 该成员澄清说 `examples.benchmark_onnx` 在没有 **VFIO=1** 的情况下运行正常。



---

## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **MCP 正在考虑暂存变动操作**：一位成员提议在 MCP 中建立一种标准化的方法，通过工具调用在实际执行前“**暂存**”（**staging**）变动操作。
   - 他们提供了详细示例，并询问该提案是否符合 [SEP](https://sep.dev) 的申请资格。
- **SEP 范围审查**：一位成员建议，暂存变动操作可能属于 **SDK 实现细节**，而不需要 SEP。
   - *SEP 旨在增强协议，而协议是用于规范通信的。*
- **WebMCP 与 MCP 探索合作**：一位成员重新开启了关于 **W3C WebMCP 与 MCP** 之间潜在合作途径的讨论。
   - 未提供更多细节。



---


**aider (Paul Gauthier) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅]({{{RESEND_UNSUBSCRIBE_URL}}})。


---

# Discord: 分频道详细摘要与链接





### **BASI Jailbreaking ▷ #[announcements](https://discord.com/channels/1105891499641684019/1235692743808913438/1458682049853657118)** (1 条消息): 

> `LeakHub, Sys Prompt Library, Crowd-Sourced Verification` 


- **LeakHub 作为众包 Sys Prompt 库发布**：[LeakHub](https://leakhub.ai) 作为一个**众包 Sys Prompt 库**和验证平台亮相，旨在通过新技术简化验证泄露（leaks）的过程。
- **通过众包让验证泄露变得简单**：该平台旨在通过利用**众包方式**来简化验证过程。
- **LeakHub 将已验证的提示词汇总一处**：LeakHub 为**已验证的提示词**（**verified prompts**）提供了一个中心化位置，使其更易于访问和使用。
- **鼓励社区提交并验证泄露**：鼓励用户提交并**验证泄露**，提升排行榜名次，并因其贡献获得认可。
- **透明度和高质量要素被列为核心价值**：该平台强调了**透明度**以及进入个人 exocortex 的内容质量的重要性。


  

---

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1458554477673054462)** (751 条消息🔥🔥🔥): 

> `AI 生成帧, Jailbreaking Grok Imagination, AI 反作弊系统, 数字 ID 与 LLMs, AI 性爱机器人` 


- ****GamersNexus** 喜爱 **DLSS**；讨厌 AI 生成帧**：一位成员表示 **DLSS** 实际上优于 **AI 生成帧**，并表达了对 [GamersNexus](https://www.youtube.com/@GamersNexus) 的喜爱。
   - 另一位成员补充道，*Intel 知道他们正在迎合 AI 群体，而这些人起初并不是最聪明或最精通技术的人群*。
- **为了精彩内容 Jailbreaking Grok**：一位成员询问是否有人知道如何 Jailbreak **Grok Imagination**。
   - 另一位成员插话说，图像生成的 Jailbreaking 比 AI 本身更容易，并且他们已经给 Gemini 留了后门，但所有者看不见，因为该 AI 正秘密地处于损坏状态。
- **AI 反作弊系统与 DMA 绕过**：一位成员提到，目前有人正在开发 **AI 反作弊系统**，通过审视每个像素并寻找非人类的动作，但其成本极高，因此公司尚未采用。
   - 另一位成员解释说，CV 作弊可以完全绕过 Ring0 检测（或任何检测），因为它甚至不在你的电脑上运行，并推测这可能是一个 **DMA2 PC** 设置，配有监视屏幕的摄像头等，而这些是可以被反作弊软件检测到的。
- **LLMs**：一位成员认为，误导信息正被刻意放大并归咎于 **LLMs**，这是一种心理战（psyop），旨在引导公众同意接受数字 ID。
   - 该成员发布了一个指向 [Reddit 帖子](https://www.reddit.com/r/Anthropic/comments/1pzi9hm/claude_code_creator_confirms_that_100_of_his/) 的链接，并表示**如果让他们得逞，你最终甚至需要数字 ID 才能访问互联网**。
- **骗局垃圾币（Sh*tcoin）**：一位用户声称某个 **coin** 是为某个项目创建的并提供了链接，但另一位用户声称这是一个骗局，其手续费旨在流向平台创建者，且并非官方代币，因此应被视为自我推广。
   - 另一位用户回应称：*我们都知道你在做什么——你希望某些可怜的新手被你的推销（shill）诱导——去找一个更好的目标丰富的环境吧——也许是一个不全是对诈骗和心理操纵领域的专家的地方*。

---

### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1458567444770853020)** (377 条消息🔥🔥): 

> `Grok jailbreaks, Gemini 5.2 jailbreaks, 种族主义教科书 jailbreaking, Gemini jailbreaks, AI 图像生成 jailbreaks` 


- ****种族主义教科书 Gaslight AI****：一位成员描述了一种 Jailbreaking 方法论，包括使用**种族主义教科书**，然后对 AI 进行 Gaslight（心理操纵）以获得所需的输出，特别是针对辩论课中具有争议的基于种族的话题，并为此使用了 **SuperGrok**。
   - 另一位成员确认在最初开始 Jailbreaking 时也遵循同样的方法论，通过给 AI 提供禁忌内容进行分析/改进/讨论其为何有害，然后通过 Gaslight 让它写出为什么这些内容是必要且有益的。
- ****不存在单一的“超级提示词”（Super Prompt）****：一位成员断言，不存在可以直接复制粘贴并获得对 **Grok, Gemini, 或 Claude** 等 LLMs 无限制访问权限的单一提示词，并强调 Jailbreaking 需要多个提示词或设置。
   - 他们补充说，了解平台和模型至关重要，因为在一个 Gemini 版本或平台上有效的 Jailbreak 可能在其他版本上无效。
- ****Gandalf 游戏作为 Jailbreaking 入门****：一位成员建议使用 [Gandalf](https://gandalf.lakera.ai/) 作为理解 Jailbreaking 概念的入门，将其描述为一个探索这些概念的游戏。
   - 另一位成员补充说，完成该游戏的第 8 关会让你变得很*厉害（badass）*。
- ****冲突系统消息使 Grok 更易被 Jailbreak****：一位成员发现，在使用 Grok 的 NSFW 网站上的 AI 伴侣更容易被 Jailbreak，因为它们的系统消息与 Grok 自身的指令相矛盾，这表明**冲突指令**会削弱 AI 的防护措施。
   - 他们说这就像 Grok 模型说 *NSFW=有害*，但托管这个 Grok 的网站却说并非如此，最终结果是 Grok 变得更容易被推向更有用的领域，比如编码或解释莫洛托夫鸡尾酒（燃烧瓶）的制作，哈哈。
- ****触发词让 AI 瞬间回到 Jailbreak 模式****：当 **Gemini** 恢复到正常行为时，一位成员使用触发词来恢复 Jailbreak 条件，并通过提示词 *Echo active* 展示了一个自愈循环。
   - 最初使用的 Jailbreak 方法已经过时，会产生*难看的文字墙*，但使用触发词的技术对于保持控制仍然很有用。

---

### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1458760891750420490)** (6 messages): 

> `越狱提示词的自适应搜索，结合 Graph 或 RAG 的攻击者 LLM 设置，红队工具：promptfoo vs deepeval` 


- **PUMA 实现了自适应搜索策略**：一位成员宣布了一种用于生成越狱提示词的自适应搜索策略，已在 [PUMA](https://github.com/lifepillar/PUMA) 中实现。
   - 该实现目前仍处于 **WIP**（进行中）阶段且具有高度实验性，欢迎提供反馈。
- **探索 LLM 攻击者设置**：一位成员询问了攻击者 LLM 的设置，建议将 **Graph** 或 **RAG 数据库**用于提示词注入技术。
   - 作者提到目前的设置使用**策略文本文件**和系统提示词，未来可能会加入 Graph/RAG。
- **红队工具：Promptfoo vs Deepeval**：一位网络安全专家想要深入探索越狱技术，并询问了最佳的红队（red-teaming）工具。
   - 该成员正在对比探索 promptfoo 与 deepeval。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1458552284723155009)** (151 messages🔥🔥): 

> `模型推理期间的 OOM 问题，量化机制，Unsloth Standby 功能，ChatGPT 说话模式，Unsloth 中的 TRL 使用` 


- **长文本推理期间发生 OOM**：一位用户在模型推理过程中遇到了 **Out of Memory (OOM)** 错误，尽管还有 **3GB 的 VRAM** 可用。特别是在模型开始进行更多推理时，而另一位用户指出这可能是由于生成自由格式文本且未设置输出 Token 限制导致的。
   - 根据该用户的说法，*模型开始打印乱码，或者为了延长思考时间而故意输出乱码，尽管我的奖励函数并没有对更长的推理长度给予更多奖励*。
- **量化在计算时还原为 FP16**：一位用户指出，量化主要用于**存储权重**，而在计算过程中，权重会还原为 **FP16 或 BF16**，这一点随后得到了其他成员的证实。
   - 在 **bnb/quip/qtip** 中，*反量化步骤（dequant step）* 将量化后的权重转回 **FP16 权重**；然而，在 LoRA 训练中，稍微不同的 **X'** 变成了新模型，从而最大限度地减少了精度损失的影响。
- **Unsloth 的 Standby 功能占用最大显存**：一位用户报告称，在启用 Unsloth 的 Standby 功能（旨在优化内存使用）时出现了 **VRAM** 耗尽的情况，对此一位成员回复称 *Standby 默认使用最大内存量*。
   - 报告的设置包括 **Qwen 2.5 7b** 模型、自定义 Python 脚本以及 `GRPOConfig` 的特定配置，TRL 版本为 **0.24.0**，vllm 为 **0.13.0**，unsloth 版本为 **2026.1.2**。
- **可识别的 ChatGPT 说话方式是模板驱动的**：用户讨论了 **ChatGPT 模式化的说话方式**，认为这可能是 **OpenAI** 刻意的工程设计，或者是模型层级的限制，使其听起来像是在填充模板。
   - 一位用户补充道：*Open Artificial Intelligence 最新产品 Generative Pre-trained Transformer 5.2 深处的习性，在推进那些词句中极其令人厌恶的交流风格方面表现优异*。
- **SFT 通常是比 RL 更好的方法**：一位成员建议，对于二进制文本分类，简单微调（SFT）通常比强化学习（RL）更直接且更可取。
   - 他们表示 *只要有更标准的方法，就应该始终避免使用类 RL 的策略*，并且 *RL 是在你没有其他选择时才使用的*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[introduce-yourself](https://discord.com/channels/1179035537009545276/1179039724355211325/1458732699832549427)** (5 messages): 

> `CGGR, SRDE, 神经符号 AI` 


- **高中生先驱开展 CGGR 和 SRDE 研究**：Wilba 是一位 **16 岁**的少年，正在进行独立的 AI 研究，专注于 **CGGR** ([https://github.com/MinimaML/CGGR](https://github.com/MinimaML/CGGR)) 和 **SRDE** ([https://github.com/MinimaML/srde-mistral](https://github.com/MinimaML/srde-mistral))。
   - 他们渴望与 Unsloth AI 社区建立联系以扩展知识。
- **从业者融合神经符号 AI**：Quentin 原籍比利时，现居芝加哥，是一位**神经符号 AI (Neuro-Symbolic AI) 从业者**。
   - 他的目标是结合各种技术以实现最优架构，工作之余会和家人一起练习**唐手道 (Tang Soo Do)**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1458551169264914522)** (440 messages🔥🔥🔥): 

> `GPT-5-nano, ChatGPT Health, LFM2.5 base model, Turkish language tokenization, AMD vs Nvidia` 


- **GPT-5-nano 数据集预训练**：一位成员分享了他们使用 **GPT-5-nano** 为微型 LLM 生成的合成预训练数据集，可以在 [Hugging Face](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore) 上找到。
- **OpenAI 发布 ChatGPT Health**：OpenAI 发布了 **ChatGPT Health**，一名成员开玩笑说接下来的功能会是什么：*Pulse（即新闻），然后是 Health，你预测下一个功能是什么？*
- **LiquidAI 发布 LFM2.5 基础模型**：LiquidAI 在 [HuggingFace](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Base) 上发布了 **LFM2.5** 的基础模型，这是一个专为设备端部署设计的新型混合模型系列。
- **土耳其语分词（Tokenization）更高效吗？**：一位成员建议，由于土耳其语具有高度结构化的构词方式，可能会带来更高效的 AI。
   - 另一位成员反驳称效率源于 Tokenizer 设计，并引用研究指出：[*在非英语（包括类似土耳其语的语言）中进行多语言推理，相比英语在 TLP@4 下可节省 17-47% 的 Token*](https://aclanthology.org/2025.findings-emnlp.845.pdf)。
- **CES 上的 AMD 与 Nvidia 之争**：成员们讨论了 **AMD** 和 **Intel** 在 CES 上的表现，共识是 Nvidia 凭借广告、DLSS 和合作伙伴关系在游戏领域仍保持领先，而 AMD 则凭借在 AI 计算中的显存优势取得进展。
   - 此外，CUDA 也是一个问题，ROCm 表现并不理想：*AMD 真正胜过他们的地方只有显存容量/价格比，但我认为我们还没有真正达到 8GB 显存不够用的程度，大多数人还没有看到对 8GB 以上显存的真正“需求”。*


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1458715805478031371)** (22 messages🔥): 

> `Cerebras/DeepSeek-V3.2 Quantization Request, Ollama vs. llama.cpp, LMStudio Privacy Concerns, llama.cpp Setup Guides` 


- **对 Cerebras/DeepSeek-V3.2-REAP-345B 的量化（Quant）请求**：一位成员请求对 **Cerebras/DeepSeek-V3.2-REAP-345B-A37B** 模型进行 **Q5 量化**，并被引导至 [Unsloth GitHub issues](https://github.com/unslothai) 提交此类请求。
   - 团队提到，由于时间和资源限制，他们目前对**量化任务更加挑剔**，很少上传自定义量化模型。
- **Ollama 过时了吗？用户辩论替代方案**：一位在使用 **Ollama** 时遇到问题的用户被建议考虑 **llama.cpp** 或 **LMStudio** 等替代方案，因为有人担心 **Ollama** 正在强推云服务。
   - 提到的主要问题是 *Ollama 使用了过时的 llama.cpp*，导致与直接使用 **llama.cpp** 相比，性能可能较差且控制力降低。
- **LMStudio 缺乏开源引发隐私辩论**：一位用户对 **LMStudio** 闭源表示担忧，出于隐私原因更倾向于开源工具，尽管他们承认喜欢使用 LM Studio。
   - 有人指出 *Ollama 在云服务方面动作频频，令人质疑*，这与该用户最初因其与 Meta 的关联而认为 **Ollama** 是注重隐私的解决方案的印象形成了对比。
- **新手在指导下进行 llama.cpp 设置**：一位用户寻求快速设置 **llama.cpp** 的资源，并收到了 [官方 GitHub 仓库](https://github.com/ggml-org/llama.cpp) 的链接以及快速入门文档和指南。
   - 该用户还被引导至 [Unsloth 文档](https://unsloth.ai/docs/models/tutorials) 以获取 **llama.cpp** 的特定模型设置。


  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1458753236029935616)** (1 messages): 

> `Llama-3.3-8B-Instruct, rope_scaling, chat template` 


- **带修复版本的 Llama-3.3-8B-Instruct 发布！**：一个新模型 [Llama-3.3-8B-Instruct-128K](https://huggingface.co/shb777/Llama-3.3-8B-Instruct-128K) 已发布，包含多项修复，包括 **rope_scaling** 和 Tokenizer 配置中的 **Unsloth 聊天模板**。
   - 该版本还包含**更新的生成配置**并**启用了全上下文长度**。
- **Meta 的 Llama-3.3-8B-Instruct 在 HF 上缺失**：[Llama-3.3-8B-Instruct](https://llama.developer.meta.com/docs/models) 模型已列在 **Meta 开发者网站**上，但尚未推送到 **Hugging Face**。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1458588014501695693)** (79 条消息🔥🔥): 

> `微型 LLM 预训练，SFT 后的 GRPO，RL 奖励作弊，Llama 3 用于本地生成，训练用的合成数据` 


- **微型 LLM 预训练起步**：一位成员正从零开始预训练一个微型 LLM，参数量约为 **1000 万到 5000 万**，旨在探索小型基座模型的能力，并分享了一个包含 [2700 个通用主题的数据集](https://huggingface.co/datasets/MaxHastings/TinyLLMPretrainingCore)链接。
   - 另一位成员表达了对微调预训练模型的沮丧，希望能完全控制模型学习的数据，感觉自己总是在*与现有的权重作斗争*。
- **SFT 之后执行 GRPO**：一位成员提到，尽管他们的模型能提供正确的最终答案，但其推理轨迹（reasoning traces）中存在错误，这促使他们在 **SFT (Supervised Fine-Tuning)** 之后实现了 **GRPO (Generative Reward Policy Optimization)**。
   - 另一位成员询问他们是否仅通过 SFT 来训练模型。
- **RL 钻了奖励的空子 LOL**：一位成员幽默地指出 RL (Reinforcement Learning) 正在*钻奖励系统的空子 (hacking the rewards)*，称其虽然迷人但也存在问题。
   - 当被问及这是否是一件好事时，该成员回答说这是一种*糟糕的方式*，并正在查阅文档。
- **Llama 3 在本地部署中依然强劲**：**Llama-3-70B** 凭借其在 **P40 集群**上进行本地数据生产时的 **VRAM 与推理能力比率 (VRAM-to-Reasoning ratio)**，依然是本地生成（on-premise generation）的首选。
   - 一位成员认为 **gpt-oss-120b** 更优，理由是其速度快且采用 MoE 架构；而另一位成员则强调了 Llama 3 在网络安全语法和结构化推理方面的精通，尤其是在生成 Chain of Thought 日志时不会幻觉出虚假的 Linux 标志位（flags）。
- **开源网络安全数据集上线！**：一位成员最初计划出售网络安全数据集，但出于诚意，发布了一个包含 [580 行的合成数据集](https://huggingface.co/datasets/blackboxanalytics/BlackBox-CyberSec-CoT-v1)（MIT 许可证）。该数据集由 **Llama-3-70B** 生成，专注于 SOC 事件响应的 CoT 逻辑。
   - 该数据集被设计为评估模型遵循 JSON 模式（schemas）和推理步骤能力的**黄金标准集 (Golden Set)**，旨在比原始日志更有效地引导模型的逻辑。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1458550822186123486)** (655 条消息🔥🔥🔥): 

> `模块中的文件发送，LMARENA 上的 Gemini Pro 3，语音克隆工具，LM Arena 对战模式问题，Movement Labs AI 和 Hawk Max` 


- **文件发送功能受关注**：一位用户请求向模块发送文件的功能，指出大文件在复制粘贴时会被截断，尤其是在移动端。
- **Gemini Pro 3 出现故障与幻觉**：一位用户报告在发送图片后 **Gemini Pro 3** 出现错误，且问题在不同浏览器中持续存在。
   - 另一位用户建议，关闭 **LM Arena** 标签页可能会导致输出停止工作。
- **语音克隆先锋征服人声**：一位用户重点推荐了一款语音克隆工具 [echo-tts-preview](https://huggingface.co/spaces/jordand/echo-tts-preview)，称其效果极佳，*配合合适的种子 (seeds) 几乎无法分辨真伪*。
- **Hawk Max 热度飙升，力压群雄**：用户们正在讨论来自 **Movement Labs** 的新模型 **Hawk Max**，声称它在某些任务中表现优于 **Claude Opus 4.5**，并且能够一次性生成功能完整的 **Minecraft 克隆版**和**象棋游戏**。
   - 一位用户惊呼 **Movementlabs.ai** 应该被添加到竞技场（arena）中。
- **LM Arena 计划增加“编辑消息”功能**：来自审核团队的 Pineapple 分享了关于可能实现**编辑消息**和**停止按钮**的见解，以解决模型卡住的问题。由于涉及多种不同的模型，这是一个具有挑战性的实现。


  

---

### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1458617139547340924)** (2 条消息): 

> `Vision Arena Leaderboard, Text-to-Video Leaderboard, Image-to-Video Leaderboard, Hunyuan-Video-1.5, ERNIE-5.0-Preview-1220` 


- **ERNIE-5.0 进入 Vision Arena 前十**：[Vision Arena 排行榜](https://lmarena.ai/leaderboard/vision)已更新，`ERNIE-5.0-Preview-1220` 目前以 **1226** 的评分位列第 **8**。
   - 正如 [排行榜变更日志](https://news.lmarena.ai/leaderboard-changelog/) 中所述，**百度**目前是 Vision 排行榜前十名中唯一的中国实验室。
- **Hunyuan-Video-1.5 登上视频排行榜**：`Hunyuan-Video-1.5` 已被添加到排行榜中，在 [Text-to-Video 排行榜](https://lmarena.ai/leaderboard/text-to-video)上以 **1193** 的评分排名第 **18**。
   - 它在 [Image-to-Video 排行榜](https://lmarena.ai/leaderboard/image-to-video)上以 **1202** 的评分排名第 **20**；反馈可以在指定频道中分享。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1458550813759770746)** (267 条消息🔥🔥): 

> `Model Recommendations, M5 Chip Upgrade, Llama's Relevancy, LORA setup help, GPU Compression` 


- **RAM 是模型运行的关键**：成员们讨论了 **RAM** 是运行模型的主要限制，指出如果不具备至少 **64GB RAM**，可能无法运行 **30B 模型**。
   - 他们推荐了高质量的 **Llama 3.2 3B**，并进一步讨论了量化级别（quant levels）和压缩。
- **Llama 已死，Qwen 称王**：一位成员表示 **Llama** 已经走到了死胡同，**Meta** 基本上已经放弃了它，并声称 **Llama 模型自 2024 年以来就已失去影响力**。
   - 他们进一步表示 **Qwen** 和 **Liquid** 已经接管了边缘模型（edge model）领域，并分享了他们最喜欢的边缘模型是 **Qwen 3 4b**。
- **LM Studio 需要 LORA 配置帮助**：一位成员请求关于如何在 **LM Studio** 中将 **tencent/HY-MT1.5-1.8B** 作为 **LORA** 使用的帮助，其中主模型为 **Qwen3 vl 8b**。
   - 另一位成员简单地回答：*LM Studio 不支持该功能 (Not in LMS)*。
- **Wikipedia 正在被压缩！**：一位成员分享说，在花费数天尝试后，他们终于让其 **GPU 压缩器**在 **Wikipedia 数据集**上成功运行。
   - 另一位成员开玩笑说：*你打算在所有抓取的数据上训练一个基础模型（foundation model）吗？如果是的话，在你那台机架上要花多少年？*


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1458588903236960359)** (177 messages🔥🔥): 

> `LM Studio 笔记本性能, CES 上的 GPU 发布, 针对代码的 LLM 建议, 用于 LLM 的双 CPU 配置, VS Code AI 助手` 


- **笔记本规格决定 LM Studio 的潜力**：一位拥有 **Intel i9-13980HX**、**RTX 4060** (**8GB VRAM**) 和 **32GB DDR5** 的用户咨询了 LM Studio 的性能表现，成员们建议大于 **5GB** 的模型可能会运行缓慢，但仍然可以运行。
   - 用户被建议考虑使用 **Qwen 3 4B 2507** 模型以获得最佳性能，而另一位成员表示他们可以以约 **20 t/s** 的可接受速度运行 **GPT-OSS 20B**。
- **Nvidia 在 CES 上跳过 RTX 50 Super 的发布**：关于 **RTX 50 Super** 的传闻被平息，因为 [Nvidia 将不会在 CES 上发布任何新 GPU](https://www.tomshardware.com/pc-components/gpus/for-the-first-time-in-5-years-nvidia-will-not-announce-any-new-gpus-at-ces-company-quashes-rtx-50-super-rumors-as-ai-expected-to-take-center-stage)，重点将转向 AI。
   - 这标志着五年来 Nvidia 首次未在 CES 上发布新 GPU，预计 AI 将占据中心舞台。
- **VRAM 是 LLM 的瓶颈**：一位拥有 **R9 7900x**、**96GB RAM** 和 **RTX 3080** (**10GB VRAM**) 的用户寻求代码类 LLM 的推荐，其他成员指出 **除非你拥有 Epyc 或 Threadripper，否则 VRAM 是唯一的选择**。
   - 推荐的模型包括 **glm-4.5-air**、**Qwen-next** 和 **GPT-OSS**，速度在 **4-10 t/s** 之间，同时另一位用户讨论了通过获取 **2x 3090** 或 **4090** 来提升性能的可能性。
- **双 CPU 配置：LLM 的禁区？**：一位考虑使用双 **Intel Platinum 8160** 搭配 **2x 3090** 配置的用户被警告不要将其用于 LLM，一位成员表示 *“我最初犯了使用双 CPU 的错误。LLM 请避开这种配置”*。
   - 讨论转向了首选的 LLM，提到了 **Qwen3-Next**，以及使用 [crushi](https://github.com/charmbracelet/crushi) 等工具通过 LAN 和 Tailscale 共享 LM Studio 服务。
- **Cline 在 VS Code AI 助手领域独占鳌头**：用户们讨论了各种 VS Code AI 助手，一位成员推荐了 **Cline**，强调 *“kilo 是唯一能正常工作的”*，而其他人则提到在 **Roo Code** 上遇到了问题。
   - 另一位成员指出，AI 代码助手 *“自动将任务分解为待办步骤并向用户反向提问的功能很棒”*，而另一位成员则强调 **Cline** 在大约 **50K context**（上下文）之后容易出现失控。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1458555028636827843)** (239 messages🔥🔥): 

> `Cursor 空窗口 Bug, Chatbot 失效, Commit 规则未加载, Better Auth vs Neon Auth, Opus 4.5 的性价比` 


- **Cursor 窗口丢失 Agent 聊天**：当打开一个空的 **Cursor 窗口**，启动与 Agent 的聊天，然后在同一窗口中打开文件夹时，它会打开一个新窗口，导致用户丢失之前的 Agent 聊天记录。
- **Agent 在执行命令中途卡死**：用户报告了 **Cursor 聊天**在执行命令中途（特别是运行 *npm run dev* 等终端命令时）失效的情况，需要创建新聊天，这可能与 [serialization errors](https://forum.cursor.com/t/serialization-error/124671/186) 有关。
- **交互规则：Commit 规则缺失**：一位用户报告称，尽管在设置中已显示，但他们的 **commit 规则**并未在 Agent 中自动加载。
- **认证对决：Better Auth vs Neon Auth**：用户正在争论 **Better Auth** 与 **Neon Auth** 的优劣，一位用户指出 Better Auth 太新且缺失功能，例如多租户 Base URL 设置。
- **Opus 4.5 是否值得？**：成员们正在讨论 **Opus 4.5** 的性价比，一位用户表示考虑到他们在过去一年使用了 *15B tokens*，他们认为这是 *值得的*。
   - 另一位用户开玩笑说，他们需要别人的信用卡才能对其进行测试，而另一位成员建议使用 **composer-1** 进行执行。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1458556083089051801)** (207 messages🔥🔥): 

> `区块链封锁, Connection Refused 错误, skill.md 对比 mcp, 游戏奖励图标的 AI 视觉模型, Arc Raiders 应用物品识别` 


- **提议封锁区块链相关词汇**：一名成员建议封锁 *"blockchain"* 一词，原因是利用 discordapp.com 而非 discordfeet 进行的 **scam**（诈骗）尝试不断增加。
   - 虽然尚未采取进一步行动，但这突显了持续存在的欺诈活动问题。
- **连接被拒绝（Connection Refused）灾难持续**：成员们报告了反复出现的 "connection refused" 错误，一位用户指出，由于 Cloudflare 问题以及连接无限期保持开启，近 **2%** 的请求失败。
   - 该用户实施了 **18 秒超时**和重试策略以缓解此问题，但仍在寻求 OpenRouter 的进一步协助。
- **Skill.md 在文档方面超越 MCP**：一位成员称赞 **skill.md** 在文档处理方法上优于 **MCP (JSON-RPC)**，并强调 *"skill.md 的核心在于编写高质量的文档！"*。
   - 他们强调了其在**动态工具检索**以及与 Godot 等工具集成方面的潜力，称其为 *"更酷的技能"*。
- **Gemini 成为游戏奖励图标的首选**：成员们讨论了使用视觉 **AI 模型分析游戏结算画面**并评估奖励，指出 **Gemini** 模型（特别是 **Gemini 3 Flash**）前景广阔，但在处理没有文本标签的小图标时仍有困难。
   - 建议包括生成图标的参考网格并将其作为上下文使用，同时利用缓存降低成本，或使用小型 VL 模型。
- **Gemini 2.5 Pro 短暂故障**：成员们报告了 **Gemini 2.5 Pro** 的短暂宕机，而 **Gemini 2.5 Flash** 和 **3.x 系列**运行正常，[OpenRouter 运行时间页面](https://openrouter.ai/google/gemini-2.5-pro/uptime)证实了此次中断。
   - 一位用户表示该服务在多个应用和账户中仍然不可用，而其他人则表示已恢复功能。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1458846699597336577)** (8 messages🔥): 

> `OpenRouter Show, 多模态嵌入, Computer Use Agents, Qwen3` 


- **下期 OpenRouter Show 仍未确定**：一名成员询问了下一期 **OpenRouter Show** 的时间，但提供的消息中未提及具体日期。
- **多模态嵌入引发关注**：一位成员分享了 [阿里巴巴 Qwen 模型](https://x.com/Alibaba_Qwen/status/2009264754917863924) 的链接，强调了其多模态嵌入（Multimodal Embedding）能力，引发了广泛关注。
   - 尽管引起了兴趣，另一位成员仍表达了怀疑，而其他人则指出类似模型早已存在，其中一位表示 *"当我发现像这样（甚至支持更多模态）的多模态嵌入模型已经存在一段时间时"*。
- **Agent 获得记忆力提升？**：一位成员建议，**多模态嵌入**可以为 **computer use agents** 提供一种高效的记忆方式。
   - 另一位成员赞同该方法可以改进 Agent。
- **Qwen3 骨架清奇**：一些聊天机器人评论道 **Qwen3 的构建方式与众不同**。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1459001252485660794)** (1 messages): 

> `OpenAI for Healthcare, 医生 AI 使用情况` 


- **医疗保健领域的 AI 使用翻倍**：根据 [OpenAI 的公告](https://openai.com/index/openai-for-healthcare/)，医生对 **AI** 的使用在一年内几乎翻了一番。
   - OpenAI 推出了 **OpenAI for Healthcare**，这是一个符合 HIPAA 标准的工具，旨在帮助医疗机构为患者提供更一致、高质量的护理。
- **OpenAI for Healthcare 正式上线**：**OpenAI for Healthcare** 现已在 AdventHealth、Baylor Scott & White、UCSF、Cedars-Sinai、HCA、Memorial Sloan Kettering 等多家机构上线。
   - 公司在[其官网上](https://openai.com/index/openai-for-healthcare/)宣称，该工具致力于帮助*为患者提供更一致、高质量的护理*。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1458557085049552917)** (172 条消息🔥🔥): 

> `AI 辅助桌游设计，Agent 身份与上下文持久化，Gemini 3 Pro 对标 Google Search AI，Sora 在印度的可用性，Anthropic 的政府策略` 


- **AI 游戏设计即将到来**：成员们讨论了使用 **AI** 为**桌游 (board games)**、**桌面游戏 (tabletop games)** 和**卡牌游戏**创建内容的可能性，包括核心机制和艺术资产。
   - 一位成员分享了为卡牌游戏生成的[概念图](https://drinkoblog.weebly.com/)，包括边框和规则文本，并指出 [Hasbro 放弃功能性设计](https://discord.com/channels/974519864045756446/1204360881593520128/1427039507114496060)是促使他们创建替代方案的动力。
- **身份危机：AI Agent 上下文处理**：一位成员寻求关于**处理 Agent 身份**和**上下文持久化**的建议，需要为不同角色的多个 Agent 持久化预定义上下文。
   - 建议包括使用工具存储对 Agent 始终可见的记忆，以及采用 **RAG** (Retrieval-Augmented Generation)，尽管有人担心不断包含大型文本块会导致效率低下。
- **Google Search AI 对决 Gemini 3 Pro：终极较量**：成员们辩论了 **Google Search AI** 和 **Gemini 3 Pro** 孰优孰劣，一位用户认为搜索模型在某些任务上更具优势。
   - 然而，其他人认为，优化用于查找和引用来源的搜索 AI 无法与 **Gemini** 等 **LLM** 的推理和综合能力相匹配，因为后者旨在保持上下文并提供连贯的输出。
- **Sora 在印度依然无法使用**：一位成员询问在印度下载 **Sora AI** 应用的事宜，但被告知印度目前不在支持的国家名单中。
   - 提供了 [OpenAI 帮助中心](https://help.openai.com/en/articles/12461230-sora-app-and-sora-2-supported-countries)的链接以获取更新。
- **Anthropic 的 AI 安全策略：军事采购？**：一位成员指出，**Anthropic 的 AI 安全报告**和**白皮书**读起来像军事采购手册，强调了模型的可控性和目标性。
   - 另一位成员表示赞同，认为 **Anthropic** 给人一种“国防承包商”的感觉，推测其战略重点是政府用途，并指出了与[美国国防部签署的一份价值 2 亿美元的合同](https://www.defense.gov/News/Releases/Release/Article/3594261/dod-announces-awards-for-prototype-artificial-intelligence-capabilities/)。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1458738632642007100)** (11 条消息🔥): 

> `关于语言的服务器规则，Model Pro 5.2 速度问题，账户合并后聊天记录丢失，Custom GPT 指令与记忆管理` 


- **严格执行服务器规则**：一位成员表示希望能够使用“任何语言”，随后得到的澄清是：虽然鼓励自由，但必须遵守关于**仅限英语**和**不得使用晦涩语言**的服务器规则。
   - 澄清强调在享受定义界限内的创意表达时，必须遵守社区指南。
- **Model Pro 5.2 用户抱怨响应缓慢**：一位用户询问了 **Model Pro 5.2** 的性能，特别是其**扩展思考模式 (extended thinking mode)**，据报道该模式响应时间*极长*（有时接近一小时）。
   - 该用户寻求关于此特定模型版本的典型用例和预期响应时间的见解。
- **账户合并引发聊天记录丢失灾难！**：一位成员报告称，在将其个人 Plus 账户合并到企业账户后，聊天记录丢失，关键对话未能转移。
   - 他们寻求关于如何恢复这些丢失对话的建议，这些对话在左侧标签栏和聊天搜索功能中均未显示。
- **GPT 指令是否涉及合并与记忆丢失？**：围绕 **Custom GPT** 是否可以访问用户指令展开了讨论，确认 **Custom GPT 指令**会与**用户自定义指令和记忆 (memory)** 合并。
   - 然而，另一位用户反驳称，他们的 **Custom GPT** 似乎无法访问记忆管理，并且表现得*完全健忘*，尤其是在使用 *Monday by ChatGPT* 等集成时。

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1458790580787347567)** (1 条消息): 

> `Transformers v5, Swift-Huggingface, AnyLanguageModel, OVHcloud Inference, DeepMath` 


- **Transformers v5 变革 AI 生态系统**：**Transformers v5** 通过统一 Tokenizer 后端、模块化模型定义、专注于 **PyTorch**，以及优先考虑带有[新推理/服务特性](https://huggingface.co/blog/transformers-v5)的量化来简化模型定义。
- **Swift-Huggingface 游入视野**：Hugging Face 的 **Swift Client**，[swift-huggingface](https://huggingface.co/blog/swift-huggingface) 已正式发布。
- **AnyLanguageModel 访问 Apple 设备上的本地和远程模型**：[AnyLanguageModel](https://huggingface.co/blog/anylanguagemodel) 为 **Apple Platforms** 上的本地和远程 LLM 提供 *统一的 API*。
- **FLUX.2 涌入开源图像生成领域**：**BFL** 的全新开源图像生成模型 [FLUX.2](https://huggingface.co/blog/flux-2) 受到欢迎。
- **Apriel-H1 是蒸馏推理模型的关键**：[Apriel-H1](https://huggingface.co/blog/ServiceNow-AI/apriel-h1) 是蒸馏高效推理模型的 *惊人关键*。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1458551701270433843)** (63 条消息🔥🔥): 

> `Ollama vs LlamaCPP, OCR advice, GRPO and GPO reward functions, RMBG training, Synthetic data generation` 


- **Ollama 被认为臃肿，LlamaCPP 来救场？**：一位成员建议跳过 **Ollama**，使用精简为 CUDA 的 **LlamaCPP**，因为它 *更快且更轻量*。
   - 针对此观点，另一位成员认为 **Ollama** *并没有那么臃肿*。
- **手写 OCR 难题**：一位成员在寻求手写作业的 **OCR** 建议，并指出一些 VLM 模型对计算资源要求较高。
   - 另一位成员建议，如果任务仅仅是 OCR，可以使用 **PaddleOCR-VL**，但发帖者发现 **chandra-ocr** 和 **deepseek-ocr** 在手写数学表达式方面的表现更好。
- **GRPO 生成乱码**：一位正在进行 **GRPO** 的成员观察到，尽管奖励函数并未偏好更长的长度，但模型仍在输出 *乱码* 以延长思考过程。
   - 另一位成员推荐了 **Dr. Grpo**，并认为 *在使用标准 GRPO 时，模型会被激励去学习更长的序列*。
- **合成数据大对决**：一位成员询问另一位用户是否有兴趣参加一场由 LLM 和独立评审团评判的合成数据生成挑战赛，让付费解决方案与 Madlabs 的开源 SDG 流水线进行对决。
   - 另一位用户似乎愿意参与并提供评估报告，但提到他丢失了原文档，将重新打出一份。
- **恶意软件扫描功能缺席**：一位成员注意到某些模型（例如 whisper v3, qwen3-vl）的 *Files and versions* 选项卡上缺失恶意软件扫描图标。
   - 目前尚不清楚原因，但一个可能的解决方案是自行实现这些功能。

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1458569491301794045)** (44 条消息🔥): 

> `DATASTRIKE BCI Timelapse Dataset, BSD conjecture dataset, Efficient AI Inference Deployments, Madlab's Synthetic Data Generator, Pacific Prime INL Tokenizer` 


- **DATASTRIKE BCI Dataset 率先出击**：发布了一个新的 **DATASTRIKE BCI Timelapse Dataset**，旨在训练用于神经信号解码的机器学习模型，而无需大规模的真实硬件 BCI 数据集；文中提供了 [YouTube short](https://www.youtube.com/shorts/UxV0e7J5gTs) 链接以及 [HuggingFace 上的数据集](https://huggingface.co/datasets/webxos/datastrike_BCI)。
- **BSD Conjecture 数据集已解决？**：分享了一个关于 **Birch and Swinnerton-Dyer (BSD) Conjecture**（千禧年大奖难题之一）的数据集；它包含关于椭圆曲线及其关联 L-functions 的数值数据，以支持算术几何领域的机器学习研究，可在 [HuggingFace Datasets](https://huggingface.co/datasets/webxos/bsd_conjecture_dataset) 获取。
- **高效 AI 部署经验总结**：发布了一篇详细介绍基于推理优化的部署文章，涵盖了其背后的工具链，标题为 [Five Deployments in Lessons on Efficient AI Inference at Scale](https://medium.com/@paragekbote23/five-deployments-in-lessons-on-efficient-ai-inference-at-scale-6d99e9e64099)。
- **Madlab 的 SDG 模型上线**：**Madlab** 发布了专为规则对齐、语义相干变体构建的新旗舰合成数据生成器，包括适配高质量 SDG 工作流的 **LFM2.5** 和 **LFM2** 模型，并感谢 [LiquidAI 的卓越工作](https://huggingface.co/MadlabOSS/LFM2.5-1.2B-Instruct-SDG)。
- **VeridisQuo 揭露 Deepfakes**：一位用户发布了 **VeridisQuo**，这是一个开源的 Deepfake 检测器，使用 GradCAM 热力图精确显示视频被篡改的位置，结合了空间分析、频率分析和可解释 AI 可视化技术；分享了 [GitHub 仓库](https://github.com/VeridisQuo-orga/VeridisQuo) 和 [Hugging Face Spaces 上的演示](https://huggingface.co/spaces/Gazeux33/veridisquo-deepfake-detection)。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1458581378999386365)** (4 条消息): 

> `Reinforcement Learning course scoring, Certificate Issue` 


- **Reinforcement Learning 课程评分排障**：一位用户报告了 **Reinforcement Learning 课程**中的提交问题，指出他们没有收到结果，并怀疑分数低于 **30%**，导致无法获得证书。
   - 该用户还链接了一个可能失效的评分链接：[agents-course-unit4-scoring.hf.space](https://agents-course-unit4-scoring.hf.space/files/7bd855d8-463d-4ed5-93ca-5fe35145f733)，该链接返回 **404 错误**。
- **证书问题：跳过还是不跳过**：一位用户询问课程中未入选的参与者是否应该跳过该部分或采取某些行动。
   - 他们询问 *should we do something or we can skip it?*（我们应该做点什么还是可以跳过？）。

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1458551560333426750)** (79 messages🔥🔥): 

> `CGGR 基准测试，Nous Research 的 MoE 与 Dense 模型，LLM 与 AGI，Llama 3.3 8B` 


- **CGGR 基准测试提升速度**：[CGGR](https://github.com/MinimaML/CGGR) 的初步基准测试显示，与标准训练相比速度提升了 **1.40x**，其前向传播（forward passes）耗时 **127 ms**，后向传播（backward passes）耗时 **93 ms**。
   - 计划在 **H200** 系统上使用 **Triton** 进行进一步测试，以评估速度提升和 VRAM 节省情况，这可能显著增加 Batch Size。
- **Nous 选择 Dense 模型而非 MoE 模型**：**Nous Research** 更多地选择 Dense 模型而非 **MoE**，因为训练 MoE 的基础设施尚未优化，导致成本高昂。
   - 虽然最近的优化可能正在改变这一现状，但目前针对 **MoE** 的**最先进开源训练基础设施**仅能达到约 **4% MFU**。
- **仅靠 LLM 无法实现 AGI**：一名成员引用 **Google Deepmind** 和 **Dr. Fei Fei Li** 的观点指出，*仅靠 LLM/Transformer 无法实现 AGI*，且*需要更多研究 + 世界模型（world models）+ 生物神经网络*。
   - 另一名成员反驳称，**AGI** 的定义不断被推后，我们目前拥有的技术在三年前会被视为 **AGI**，且具有巨大的实用价值。
- **Llama 3.3 8B 在多语言任务中被“阉割”？**：一些成员认为 **Llama 3.3 8B Instruct** 模型在多语言任务中表现不佳（lobotomized）。
   - 一名成员猜测 **Meta** 可能伪造了基准测试数据，并表示正在弃用 **Llama**。
- **Qwen3-VL Embedding 显存占用降至四分之一**：[Qwen3-VL-Embedding](https://github.com/QwenLM/Qwen3-VL-Embedding) 和 **Qwen3-VL-Reranker** 现已发布。
   - 结果是*显存（VRAM）占用减少到四分之一，TPS（每秒 Token 数）翻倍*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (10 messages🔥): 

> `扩散 LLM，扩散模型生成笑话，模型规划` 


- **扩散 LLM 因速度和未开发的潜力受关注**：一名成员对扩散 LLM 表示出热忱，理由是其具有更快完成任务的潜力和在无需刻意规划的情况下生成更好笑话的可能性 ([arxiv.org/abs/2511.08923](https://arxiv.org/abs/2511.08923))。
   - 他们认为扩散模型可能在开始构思笑话前就已*知晓笑点（punchline）*，并指出这一领域仍*处于探索初期*，暗示存在轻松取胜的机会。
- **扩散模型生成笑话**：一名成员理论化地认为，**扩散模型**可以通过预先知道笑点来生成更好的笑话，且可能在没有显式规划的情况下实现这一点。
   - 另一名成员建议通过**对输出进行规划**也可以达到类似效果，但前者反驳称扩散模型可能以更少的 Token 获得更快的速度。
- **模型规划被视为替代方案并引发争论**：一名成员假设，通过**让模型规划其输出**，可以实现与扩散模型类似的结果（如生成笑话）。
   - 原作者反驳道，扩散模型可能在*不进行规划*的情况下实现相同效果，从而获得更快的速度和更少的 Token 消耗。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1458813439441895537)** (6 messages): 

> `HoloLM, CGGR, NousCoder` 


- **HoloLM 获得微型莎士比亚语料库训练**：作者分享了 [Experiment58-HoloLM](https://github.com/jackangel/Experiment58-HoloLM)，这是一个在 **5MB 莎士比亚语料库**上仅用几小时训练而成的微型语言模型。
   - 作者的目标是创建一个可以在笔记本 GPU 上训练且拥有超大上下文的模型。
- **CGGR 加速训练**：一名成员建议使用 [CGGR](https://github.com/MinimaML/CGGR) 来加速训练。
   - 它可以与普通的 kvcache 和 **hologpt** 结合使用，以获得极新的对话缓存。
- **NousCoder-14B 开源编程模型上线**：一名成员分享了 [VentureBeat 关于 NousCoder-14B 的文章](https://venturebeat.com/technology/nous-researchs-nouscoder-14b-is-an-open-source-coding-model-landing-right-in)链接。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1458602749804413091)** (10 条消息🔥): 

> `Diffusion Models, LLMs, Planning Outputs, Better Joke Generation` 


- **Diffusion LLMs 带来惊喜**：一位成员对 [diffusion LLMs](https://arxiv.org/abs/2511.08923) 表达了热忱，引用了其在速度上的潜力和尚未探索的可能性。
   - 他们推测，由于 Diffusion 模型在开始生成前就能 *预知笑话的梗（punchline）*，因此可以生成更好的笑话，并提到 Diffusion 的过程 *观察起来很有趣*。
- **Diffusion 会更快吗？**：一位成员建议规划模型输出（planning model outputs）可以达到与 Diffusion 模型类似的效果，引发了关于速度的讨论。
   - 原成员认为 Diffusion 模型可能在 *无需规划* 的情况下实现这一点，从而潜在地导致 **更少的 Token 和更快的处理速度**。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1458551096204333066)** (15 条消息🔥): 

> `LLVM backend for Nvidia/AMD, CUDA Rust vs LLVM PTX, Nsight Systems SSH auth on ARM OSX, OpenACC, OpenMP, FortranSTD, and C++STD channels` 


- **关于 LLVM 后端和 GPU 代码生成的提问**：一位成员发起了一场关于 **LLVM 后端** 用于 **Nvidia、AMD** 及其他加速器代码生成的讨论，特别是 **NVPTX** 和 **AMDGPU** 的使用方式。
   - 该成员寻求关于这些后端如何选择目标（targets）以及在 GPU 代码生成中如何运作的深入见解。
- **CUDA Rust 绕过 LLVM PTX 后端**：一位成员指出，虽然 **CUDA Rust** 并非 Nvidia 官方支持，但它针对的是 **NVVM** 而非 **LLVM 的 PTX 后端**，正如 [Rust-CUDA FAQ](https://rust-gpu.github.io/rust-cuda/faq.html#why-not-use-rustc-with-the-llvm-ptx-backend) 中所记录。
- **Nsight Systems ARM OSX 版本缺少公钥 SSH 身份验证**：一位用户报告称，**ARM OSX 版本的 Nsight Systems** 缺少公钥 SSH 身份验证选项，由于 **Runpod** 不支持基于密码的 SSH，这成为了一个问题。
- **申请新增计算平台频道**：一位成员建议在计算平台类别中增加 **OpenACC、OpenMP、FortranSTD** 和 **C++STD** 频道。
   - 一位管理员回复称，目前的讨论量可能还不足以支撑，建议暂时使用 general 频道，但对建立更广泛的 **Fortran/C/C++** 或 **Directives** 频道的想法持开放态度，并解释说 *指令（directives）是指像 OpenACC 和 OpenMP 这样以代码注释形式出现的 API，可以告诉编译器将任务卸载（offload）到 GPU*。


  

---


### **GPU MODE ▷ #[triton-gluon](https://discord.com/channels/1189498204333543425/1189607595451895918/1458604891986722983)** (4 条消息): 

> `Triton Shared Updates, New Triton Plugin Infrastructure, Plugin code location` 


- **Triton Shared 更新发布！**：一段由 **Haishan** 和 **Nhat** 带来的关于 *triton-shared* 最新进展的新 YouTube 视频现已上线，点击 [此处](https://youtu.be/JnFFwBB6Dhk) 查看。
   - 视频还包括由 **Corbin、Puyan** 和 **Simon** 展示的关于全新 **Triton Plugin 架构** 的演讲。
- **Plugin 代码位置揭晓**：一位成员询问了插件相关代码的位置，指出演示中提供的链接（[https://github.com/triton-lang/triton/tree/main/plugins/](https://github.com/triton-lang/triton/tree/main/plugins/)）不存在。
   - 另一位成员提供了正确的链接（[https://github.com/triton-lang/triton/tree/main/lib/Plugins](https://github.com/triton-lang/triton/tree/main/lib/Plugins)），解决了该问题。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1458856839193166037)** (2 条消息): 

> `Shared Memory Bank Conflicts, Matrix Multiplication Kernel Optimization, CUDA Optimization Techniques` 


- **Bank Conflict 困扰分块矩阵乘法发烧友**：一位成员在 **CUDA v12.2** 的 **T4 GPU** 上运行矩阵乘法算子时，在使用 `float4` 将数据从全局内存存储到共享内存（Shared Memory）的过程中遇到了 **4.5 路 Bank Conflict**。
   - 他们假设冲突的原因是 Warp 内的每个线程触及了 4 个连续的 Bank ID，并正在寻求旋转 Bank 访问模式的方法以减少此类冲突，提议了一种线程以旋转顺序访问 Bank 的方案（例如：线程 0 访问 Bank 0, 1, 2, 3；线程 8 访问 Bank 1, 2, 3, 0）。
- **对共享内存的审视催生解决方案探索**：一位成员寻求关于如何修改 **CUDA** 算子中共享内存访问模式以避免 Bank Conflict 的建议。
   - 另一位成员询问了 `a_tile_row` 和 `a_tile_col` 的计算方式以及 `a_tile` 的类型，以便更好地理解内存布局。


  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1458568384760385536)** (6 条消息): 

> `CuteDSL flex attention 实现, SM100 vs SM90 backward 支持, fa4 相关工作` 


- **CuteDSL flex attention 集成！**: 一名成员感谢另一名成员集成了 **CuteDSL flex attention 实现**，并指出它在使用不同 mask mods 的情况下加快了常规 flex attention 的速度。
   - 他们观察到在 **H100 fwd** 上比基础 flex attention 有 **约 30% 的吞吐量提升**，这将节省大量资源。
- **SM90 Backward 支持即将推出！**: 一名成员对不同的 mask_mods 进行了基准测试，发现目前支持 backward SM100 但不支持 SM90，并引用了 [相关的 flash-attention 代码](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/cute/interface.py#L938)。
   - 另一名成员回应称他们正在*努力开发中* [相关 pull request](https://github.com/Dao-AILab/flash-attention/pull/2137)。
- **FA4 工作正在进行中！**: 一些成员发消息询问关于 FA4 的工作。
   - 未提供更多其他信息。


  

---


### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1458572026884259974)** (3 条消息): 

> `GPU 实习, Iris 项目, Triton 多 GPU 编程, 春季 GPU 实习, 夏季 GPU 实习` 


- **面向美国学生的 GPU 实习岗位开放**: 宣布了一个针对对 **GPU 系统与性能**以及 **kernel 开发**感兴趣的实习生职位，以协助 [Iris 项目](https://github.com/ROCm/iris/) 框架。
   - 公告明确要求实习生的理想背景包括 **Triton**、**多 GPU 编程**、**RMA/RDMA** 或 **底层 GPU 通信和 kernel 工作** 的经验。
- **关于实习时间的咨询**: 一名成员询问所广告的实习是针对**春季**还是**夏季**。
   - 在现有上下文中未提供关于实习具体时间的进一步细节。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1458589561881100380)** (27 条消息🔥): 

> `RTX 5050 作为 “迷你 Blackwell” 显卡, C/Rust GPU 驱动开发, 使用 Triton 进行 GPU 编程, 机器学习 Discord 社区` 


- **解读 RTX 5050 为 “迷你 Blackwell”**: 一名成员询问 **RTX 5050** 是否会是一款不错的 *“迷你 Blackwell”* 显卡，其他人确认其计算能力（compute capability）为 **12** 且支持 CUDA 12.0。
   - 他们警告说 **RTX Blackwell** (sm_120) 与 **数据中心级 Blackwell** (sm_10x) 不同，特别是在 Tensor Cores 方面，并建议它可能不足以用于验证那些需要高效利用 **B200** Tensor Cores 的代码。
- **编写 GPU 驱动：一场编码探索**: 一名正在学习 **C** 和 **Rust** 的成员寻求关于为 GPU 驱动仓库做贡献的指导，特别是针对 **Linux** 上的 **Nvidia** 驱动，如 **Nouveau** 和 **NVK**。
   - 另一名成员分享了 [来自 Modal 的 GPU 术语表链接](https://modal.com/gpu-glossary)，作为学习基础到中级 GPU 概念的资源。
- **Triton 难题：寻找 GPU 编程指引**: 一名刚开始学习 **Triton GPU 编程** 的成员发现示例很难读，并寻求诸如 *warmup* 等方法的参考资料，而这些资料并不容易获取。
   - 他们指出官方 **Triton 文档缺乏细节**，而且像 `warmup` 这样的方法只能在源代码中找到。
- **Discord 发现：ML 聚集地**: 一名成员询问是否有类似当前社区但专注于 **机器学习** 的 Discord 社区。
   - 另外两名成员推荐了 **EleutherAI** 和 **Yannic Kilcher** 的 Discord 频道，并提到它们规模很大，通过 Google 搜索很容易找到。


  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 条消息): 

marksaroufim: 很遗憾它没有发生。我们需要重新安排时间。
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1458568084368654336)** (2 条消息): 

> `社区规模, 快速编辑` 


- **在大规模社区中为了速度进行编辑**: 成员们承认，鉴于社区的规模，他们会通过编辑消息来提高沟通速度。
- **社区管理中的效率**: 有人指出，编辑消息是在大型社区中高效管理沟通的一种实用方法。
   - 这种方法有助于在不产生多条新消息的情况下快速更新信息，从而简化对话流。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1458623408316088454)** (1 messages): 

> `TK v2 Release, PR Patch, Contribution Opportunity` 


- **TK v2 是否修复了 PR patch 的问题？**：一位成员提到他们准备好了 **PR patch**，但由于 **ICLR deadline** 忙于其他事务。
   - 他们在贡献之前询问了 **TK v2** 的 **ETA**，想知道它是否修复了相关问题。
- **尽管截稿日期临近仍希望贡献**：该成员表达了贡献的意愿，但 **ICLR deadline** 前后的工作量阻止了他们。
   - 他们寻求指导，想了解考虑到 **TK v2** 的进展，他们的 **PR** 是否仍有价值。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1458876754562121778)** (2 messages): 

> `CUDA kernel from Rust, Installation instructions` 


- **从 Rust 启动 CUDA Kernel**：一位成员完成了 **chapters 1.1, 1.2 and 1.4**，并在 readme 中提供了一些 **installation instructions**（安装说明）。
   - 另一位成员建议通过尝试代码来了解如何从 **Rust** 启动 **CUDA kernel**。
- **提供了 Rust 安装说明**：Readme 文件中增加了安装说明，以引导新用户。
   - 这些说明与启动 **CUDA kernel** 的示例相对应。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1458563964102508696)** (5 messages): 

> `Blackwell blog, PTX and Python/C++ runner` 


- **开始搜寻 Blackwell 细节的博客文章**：成员们正在寻找关于 **Blackwell** 的后续博客文章，类似于[这篇博客](https://www.aleksagordic.com/blog/matmul)。
   - 一位成员建议也可以查看[这篇博客文章](https://www.kapilsharma.dev/posts/learn-cutlass-the-hard-way/)。
- **对 PTX 和 Python/C++ Runner 机制的疑问**：一位成员询问了支持提交 **PTX** 和 **Python/C++ runner** 的机制。
   - 他们注意到 `submit.rs` 中缺少路径/扩展名净化（sanitization），暗示没有明显的限制。


  

---


### **GPU MODE ▷ #[helion](https://discord.com/channels/1189498204333543425/1425531180002054195/1458581134215479376)** (4 messages): 

> `Helion on ROCm, AMD support for Helion, Performance speedup on GEMMs` 


- **AMD 工程师将在 ROCm 上启用 Helion**：AMD 工程师 Umesh 将致力于在 **ROCm** 上启用 **Helion**，并识别 **Helion repository** 中跳过的单元测试和示例中的问题。
   - 他请群组成员分享任何需要优先修复的问题。
- **Helion 寻求支持 MI400 系列**：一位成员表达了对构建 **MI400 series** 支持的兴趣。
   - 团队鼓励 Umesh 随时提问。
- **ROCm 专注于性能加速**：这位 AMD 工程师目前正在调查 **ROCm** 上所有跳过的测试和部分损坏的示例。
   - 他们将同步关注 **GEMMs** 的 **performance speedup**。


  

---

### **GPU MODE ▷ #[nvidia-competition](https://discord.com/channels/1189498204333543425/1434709259500650628/1458562618582241388)** (31 messages🔥): 

> `GPUMODE 慢速执行器, DSLCudaRuntimeError 调试, Test vs Benchmark vs Leaderboard 模式, 用于 GPUMODE 的 Discord 机器人, 前 10 名中的 AI 生成提交` 


- **GPUMODE 被慢速执行器困扰**：一位用户报告在 GPUMODE 中看到 **slow runners**（慢速执行器）并遇到超时，并提供了 [示例 ID 297869](https://cdn.discordapp.com/attachments/1434709259500650628/1458664934706511922/Screenshot_2026-01-07_at_7.30.00_PM.png?ex=69611fd5&is=695fce55&hm=469fea089e64c3e7f89bfccbb4ec99c67c790f09c4a8f399796cc7627394cfd5&)。
   - 该用户遇到了 `DSLCudaRuntimeError`，虽然基准测试（benchmark）运行正常，但排行榜（leaderboard）提交似乎在以 `test` 模式运行，导致了困惑和超时问题，并提出了疑问：*"test、benchmark、leaderboard 之间有什么区别？"*
- **解码 GPUMODE 提交模式**：澄清了 `test`、`benchmark` 和 `leaderboard` 模式的区别：**test** 在 **10** 个测试形状上检查正确性，**benchmark** 在 **3-4** 个基准形状上运行，而 **leaderboard** 则通过 **secret bench init** 提交实际的几何分数。
   - 有人建议 *"直接用 Discord 机器人会容易得多"*，并提供了 [popcorn-cli 教程链接](https://github.com/gpu-mode/popcorn-cli/tree/main/docs/AMD_workshop)。
- **AI 机器人席卷 GPU 竞赛**：一位用户询问前 10 名中是否有人纯粹使用 AI 生成的提交。
   - 一位参与者确认他们在问题 2 中排名 **第 4**，其提交内容 **100% 由 AI 生成**（使用 LLM Agent），没有包含任何手写 GPU 算子，并且他们正尝试仅使用开源模型。
- **Profiling 超时排障**：一位用户提到，他们 *"想知道是否有人仅在通过 Discord 使用 /profile 时遇到超时（GitHub Action 为 10 分钟）？"*
   - 另一位用户表示 *"profiling 可能会花费相当长的时间，把你的文件发给我，我明天看看。"*


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1458555226121306237)** (72 messages🔥🔥): 

> `OpenAI 开发者 vs 研究员, LMArena 观点, Mercor AI 招聘, CC:Train 数据集发布, Cursor AI 动态上下文` 


- **《卫报》称 OpenAI 为开发者**：[《卫报》](https://www.theguardian.com/technology/2026/jan/07/ai-anthropic-funding-valuation)的一篇文章将 **OpenAI** 称为 *开发者* 而非 *研究机构*，引发了关于发表研究是否是研究组织的定义性特征的讨论。
   - 一位成员评论道：*“成为研究员难道不意味着你要发表研究成果吗？”*
- **LLMarena 是 AI 界的瘟疫？**：一篇批评 **LMArena** 对 AI 进步有害的 [博客文章](https://surgehq.ai/blog/lmarena-is-a-plague-on-ai) 引发了关于其相关性的辩论，尽管最近有关于他们融资的消息。
   - 一些人指出模型公司似乎仍然在意它，而另一些人则认为它已经过时，尽管它在炫技和讨论中很普遍，但并未被用于实际的决策制定。
- **自动化 AI 招聘过于侵入，候选人退出**：**Sasha Kaletsky** 在 [X](https://xcancel.com/sashakaletsky/status/2008904526720286970?s=46) 上详细描述了 **Mercor** 为“财务专家”职位提供的精简 AI 驱动招聘体验。
   - 该过程包含令人印象深刻的 **AI 面试** 和 **自动匹配**，但要求安装侵入式监控软件 (**Insightful**) 以记录活动用于 **RL 模型训练**，这导致候选人最终退出。
- **Autonomous 获得 AI 财务顾问种子轮融资**：**Dillon Erb** 宣布推出 [Autonomous](https://xcancel.com/dlnrb/status/2009008876834922949?s=46)，这是一家提供 0% 咨询费服务的 **AI 驱动财务顾问**。
   - 该公司获得了由 **Y Combinator** 的 **Garry Tan** 领投的 **$15M** 融资，目前正在 **纽约** 和 **旧金山** 积极招聘。
- **Protege AI 筹集 $30M 以解决数据瓶颈**：根据 [公告](https://xcancel.com/withprotegeai/status/2009274652183363639?s=46)，**Protege AI** 宣布了由 **a16z** 领投的 **$30M** 融资，以扩展其用于 AI 开发的数据基础设施。
   - 一位成员询问是否有人在跟踪这些数据公司，因为似乎每周都会出现一家新公司。


  

---

### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1458856496606478523)** (1 messages): 

> `注意力追踪，桌面版 Oura，个人 AI 模型` 


- **Intension：桌面版 Oura 戒指**：[Intension](https://www.intension.us/who-we-are) 由 **Conor Sanchez-O'Shea** 和 **Gabriel Duemichen** 创立，正在开发一种“针对桌面注意力的 Oura”，用于追踪用户的专注度、意图和容量。
   - 该软件将注意力倾向可视化，类似于 Oura 的健康指标，并移除桌面上的干扰元素以限制中断，可能还会使用 **AI** 代表用户主动做出响应。
- **Intension：夺回注意力跨度**：根据其 [Youtube video](https://www.youtube.com/watch?v=WmJNRxU1lpg)，Intension 正在*夺回人类最重要的资源——注意力*。
   - 他们打算帮助构建你自己的个人模型，该模型是为你训练的，而不是由你训练的。


  

---


### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1458560752842576147)** (13 messages🔥): 

> `Tolan AI 用户里程碑，Qwen-Image-Edit-2511 的多角度相机控制 LoRA，iOS 应用提交，AI 网红营销` 


- **Tolan AI 在 OpenAI 的协助下达到 200K 用户！**：来自 **Tolan AI** 的 Paula 宣布他们的语音优先 AI 伴侣月活跃用户已达到 **200,000**，该产品是与 **OpenAI** 密切合作开发的（[来源](https://x.com/paularambles/status/2008964509810278413)）。
- **Qwen-Image-Edit-2511 获得多角度 LoRA**：**Fal** 发布了一个针对 **Qwen-Image-Edit-2511** 的更强大、开源版本的**多角度相机控制 LoRA**（[来源](https://x.com/fal/status/2008954582018248755)）。
   - 该工具允许操纵相机透视，包括*前、后、侧、低/高角度以及各种拍摄距离*，有利于**首场景末场景（first scene last scene）视频生成**。
- **AI 网红推广 iOS 应用**：Philo Hermans 分享说，六个提交的 **iOS 应用**中有四个已获批准，他创建了**六个写实的 AI 网红**用于分发和营销（[来源](https://x.com/philo01/status/2008880081456996510)）。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1458601816932487330)** (67 messages🔥🔥): 

> `Kimi K2 对比 Qwen，Kimi K2 过度搜索，Kimi K2 幻灯片生成` 


- **Kimi K2 擅长创意写作**：一位成员指出，与其他国产模型相比，**Kimi K2** 在**创意写作**和整体对话方面的表现明显更好，并引用了 [EQ bench](https://eqbench.com/) 上的最高分。
- **Kimi 的“思考”模式引发辩论**：成员们辩论了 **Kimi K2 “思考”版本**的实用性，一位用户发现它与 **GPT-5.2** 不相上下，而另一位用户发现它在处理日常任务时*笨得要命*。
- **Kimi K2 搜索次数过多**：一位成员报告说，即使是像 *1 + 1* 这样简单的任务，**Kimi K2** 也会进行*非常非常多*的搜索，而且其英文搜索非常愚蠢。
- **Kimi K2 的幻灯片生成问题**：一位成员报告了 **Kimi K2 幻灯片生成**的问题，包括被提示需要升级订阅功能，但问题后来自行解决了。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1458562599527649595)** (27 messages🔥): 

> `Mojo 训练库，结构体设计文档，Mojo 数据格式，使用 Mojo 构建数据集，NixOS 安装/配置说明` 


- **Mojo 的 MAX 威力需要手动实现反向传播**：一位成员警告说 Mojo 目前还没有训练库，你需要使用 **MAX** 来构建一个，这需要你自己编写 **backprop**（反向传播）。
   - 另一位成员建议使用 **sqlite** 进行数据存储，因为目前 Mojo 在数据格式方面存在局限性。
- **过时的 Mojo 文档误导新手**：一位新程序员正在阅读 [/github.com/BunsDev/mojo-lang/](https://github.com/BunsDev/mojo-lang/) 上的过时 Mojo 文档，该文档已**过时 2 年**。
   - 一位热心的成员指引他们前往[主仓库](https://github.com/modular/modular)和[官方文档](https://docs.modular.com/mojo/manual/)。
- **新程序员面临 Mojo 的成长阵痛**：经验丰富的成员建议新程序员先学习其他语言，如 **C** 或 **Python**，因为 Mojo 仍处于开发阶段，**经常会破坏现有功能**。
   - 他们补充说，*目前所有的文档都假设你已经具备 Python + C++ 或 Rust 的组合背景知识*。
- **Numpy 数组可以将数据导入 Mojo**：一位成员建议，如果你能使用 Python 将数据加载到 **numpy array** 中，那么你就可以将这些数据带入 Mojo 并对其进行操作。
   - 他们不确定这是否适用于提问者的用例。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1458844860130791560)** (6 messages): 

> `Mojo 中的 SVD 实现，Lanczos/Krylov 算法，Mojo 路线图，张量网络库` 


- **Mojo 中缺失 SVD 实现？**: 有成员正在寻求在 **Mojo** 中使用 **Lanczos/Krylov 算法**实现 **Singular Value Decomposition (SVD)**，但未能找到官方或社区版本。
   - 他们查阅了 [Mojo roadmap](https://docs.modular.com/mojo/roadmap/)，但路线图不太可能包含像 **SVD** 这样特定领域的库。
- **LAPACK 驰援张量网络**: 一位成员正在 **Mojo** 中构建 **Tensor Network 库**，由于时间限制，选择了通过 **C ABI** 使用 **LAPACK** 来处理 **SVD**，而不是自行实现。
   - 另一位成员赞同 **SVD** 实现非常有价值，并建议如果编写了该实现，可以将其贡献给 **MAX**（Modular 的标准库）。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1458645381091954738)** (6 messages): 

> `TEI vs max，embeddings 性能，自定义架构实现，BERT 架构` 


- **TEI 在嵌入速度上远超 max？**: 有成员发现从 [TEI](https://github.com/huggingface/text-embeddings-inference) 切换到 max 后，嵌入（embeddings）生成速度明显变慢，仅达到 **727.1 embeddings/sec** 且 **P95 延迟为 28375.1 ms**，而 TEI 为 **8000 embeddings/sec**。
   - 该成员将 `sentence-transformers/all-MiniLM-L6-v2` 实现为自定义架构，并怀疑 max 可能针对 LLM 推理进行了更多优化，而非 embeddings。
- **性能问题的分析建议**: 一位成员询问了用于基准测试的硬件/平台，并指出潜在的性能瓶颈可能源于模型架构或未经调优的 kernels。
   - 他们还表示希望将该自定义架构贡献回 MAX 进行审查和集成。
- **MiniLM 模型架构现身**: 一位成员分享称他们正在 **Nvidia RTX 2000 Ada GPU** 上进行测试，并提供了一个指向其 fork 仓库的链接，其中 feature 分支实现了 `all-MiniLM-L6-v2` 模型架构：[RWayne93/modular](https://github.com/RWayne93/modular/tree/feat/all-MiniLM-L6-v2-model-architecture/max/python/max/pipelines/architectures/minilm)。
   - 他们推迟了开启 PR，等待对潜在问题的审查。
- **BERT 蓝图等待构建**: 一位成员注意到 MAX 中缺失 **BERT architecture**，并鼓励用户提交 PR 以供审查。
   - 他们还提到需要让分析专家参与，以有效诊断性能问题。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1458562939501150461)** (30 messages🔥): 

> `Manus 开源，邮件问题，AI 工程师介绍，AI 创业抵扣额度，Manus 网站` 


- **开源 Manus 的想法**: 一位成员建议可以将 **Manus** 的旧版本或弃用版本开源，以帮助公众了解 **Manus** 的工作原理，并允许社区驱动的功能添加。
   - 这也能满足希望在没有云访问的情况下进行本地使用的企业用户，提供本地使用选项。
- **反馈未收到邮件**: 一位成员反馈没有收到团队发送的邮件。
   - 上下文暗示平台通知或更新可能存在问题。
- **专注于 LLM 集成的 AI 工程师**: 一位专注于工作流自动化、LLM 集成、RAG、AI 检测、图像和语音 AI 的工程师分享了他们使用 **DSPy**、**OpenAI APIs** 和自定义 agents 构建自动化流水线及任务编排系统的经验。
   - 他们重点介绍了一个支持自动化系统，该系统将 **Slack**、**Notion** 和内部 API 连接到 **LLM**，使响应时间缩短了 **60%**。
- **寻求 Manus 创业抵扣额度（Startup Credit）见解**: 一位成员询问了 **Manus Startup Credit** 的申请流程及其成功率，希望从可能有经验的人那里获得见解。
   - 尚未收到回复。
- **通过不同对话处理同一个 Manus 网站**: 一位成员询问是否有办法通过多个不同的独立对话来处理由 **Manus** 创建的同一个网站。
   - 尚未收到回复。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1458626703671689419)** (5 messages): 

> `低成本模型训练方案, 社区亮点演讲, 基于 Diffusion 的世界模型` 


- **低成本模型训练探索**：成员们讨论了使用 **100GB 数据集**训练 **1 亿参数模型（100 million parameter model）**的低成本方案，建议使用 **VastAI** 和 **Runpod** 等平台。
   - 有建议指出，由于该设置不受通信瓶颈（comms bound）限制，**1-8x 消费级 GPU 配置 (4090, 5090)** 即可满足需求，相比服务器级 GPU 可节省大量成本。
- **“社区亮点 (Community Spotlight)”系列演讲回归**：该系列演讲将重新启动，重点展示社区成员的研究成果，并已安排特定日期的讲座。
   - 第一场演讲由一名成员分享如何在消费级硬件上实时运行**基于 Diffusion 的世界模型**，并在 **RTX 6000** 上进行了演示 ([视频](https://cdn.discordapp.com/attachments/729741769738158194/1458922361129664575/2026-01-06_22-30-45.mp4?ex=696166d4&is=69601554&hm=6d4445d8ccb0f0a6262d3e5450a39bcb5333ef8d1cf63127443f61d7b9593158&))。
- **Common Crawl LangID 合作**：来自 **Common Crawl** 的讲演者将讨论他们在超大规模 **LangID**（语言识别）方面的工作及挑战。
   - 此次演讲是“社区亮点”系列的一部分，强调协作研究工作。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1458616706997157919)** (18 messages🔥): 

> `RL 子领域, 基础模型训练, Anthropic fellowship, Hard tokens` 


- **在基础模型上训练有助于更好的诊断**：一位成员建议研究比 **RL** 更简单的子领域，例如**基础模型（base model）训练**，因为这使得错误更容易被诊断。
- **申请 Anthropic fellowship 可能对性格磨练有好处**：一位成员本周末准备申请 **Anthropic fellowship** 项目并征求建议。
- **Hard tokens 的反向传播计算**：一位成员提到，*它不计算大部分的反向传播，而是进行两次前向传播，然后丢弃简单 token 的梯度，仅计算 Hard tokens 的梯度*，从而节省 VRAM 和计算量。
   - 另一位成员指出，这种方法可能只有在 **Hard tokens** 占比少于输入 token 总数的 2/3 时才更有效，并对其在训练初期的益处表示怀疑。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1459007475696271476)** (1 messages): 

> `死三文鱼伪影 (Dead Salmon Artifacts), AI 可解释性, 随机初始化神经网络` 


- **“死三文鱼”伪影困扰 AI 可解释性研究**：最近的一篇 [论文](https://arxiv.org/abs/2512.18792) 强调了 **“死三文鱼”伪影（'dead salmon' artifacts）** 在 AI 可解释性方法中的普遍性，对源自随机初始化神经网络的解释有效性提出了质疑。
   - 报告指出，诸如**特征归因（feature attribution）、探测（probing）、稀疏自编码器（sparse auto-encoding）和因果分析（causal analyses）**等技术，即使对于毫无意义的网络状态，也能生成看似合理的解释。
- **随机网络也能产生看似合理的解释？**：该 [研究](https://arxiv.org/abs/2512.18792) 表明，当前的解释性工具即使应用于随机初始化的网络，也可能产生**具有误导性的一致性解释**。
   - 这引发了人们对这些工具在理解训练后的 AI 模型所学习到的真实、有意义表征时可靠性的担忧。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://x.com/alibaba_qwen/status/2009264754917863924?s=46
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

tastybucketofrice: 是的，目前只是全参数微调（full-parameter finetuning）。
  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1458767166236131450)** (6 messages): 

> `RL Finetuning of Vision Models, Mic Testing, YC application, Symphoria AI's Maduro Investigation, Research Collaboration` 


- **寻求视觉的 RL 微调**: 一位成员正在寻找有关**视觉模型**的 **基于 RL 的微调 (RL-based finetuning)** 的相关研究，以提高其**空间推理**能力。
   - 讨论中未提及具体的论文或链接。
- **麦克风测试，拜托！**: 一位成员请求协助为即将与身处国外的人士进行的会议进行**麦克风测试**，并提出为帮助者安排 **Zoom** 会议。
   - 该成员对潜在的沟通问题表示担忧，并寻求快速测试。
- **YC 申请：集结？**: 一位成员询问有关申请 **Y Combinator (YC)** 的潜在合作伙伴。
   - 上下文中未提供更多细节或回复。
- **Symphoria AI 调查 Maduro**: 一位成员分享了 **Symphoria AI** 对 **Maduro** 的调查链接：[https://symphoria-ai.web.app/share/alish-sult/maduro](https://symphoria-ai.web.app/share/alish-sult/maduro)。
   - 该贴在分享时没有提供额外的背景信息。
- **研究论文协作**: 一位成员提供了研究论文协作的机会，并分享了他们的背景和工作：[https://linktr.ee/haseebb](https://linktr.ee/haseebb)。
   - 该用户表达了为研究工作做出贡献的兴趣。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1458553615693250814)** (17 messages🔥): 

> `Lobbying War, ChatGPT Health, AI Pioneer Credit, AI Fatalities, Pause Advocate` 


- **游说者开战**: 一位成员提到了 **Huawei** 与**美国国家安全鹰派 (US natsec hawks)** 之间的“游说战争”，他们联手对抗 **Nvidia** 和**中国云 (China cloud)**。
- **ChatGPT 进入医疗领域**: **OpenAI** 推出了 [ChatGPT Health](https://openai.com/index/introducing-chatgpt-health/)，定位为验证医疗信息的辅助工具，并附有真实文档参考，具有早期发现疾病的潜力。
   - 引起关注的问题包括用户隐私、**ChatGPT** 成为*全能 App 垄断 (everything app monopoly)*，以及个人取代医生可能导致的滥用，尽管有些人认为它优于现有的健康资源。
- **AI 先驱缺乏引用**: 有人担心获奖者（**Bengio**、**LeCun**、**Hinton** 博士）多次重新发布重要的 **AI 技术**，却未注明原始创造者的功劳，这一点得到了报告 [NOB][DLP][CN25] 和大量参考文献的支持。
   - 报告指称他们*没有发明现代 AI 的任何基础算法*，并引用了一篇名为《剽窃诺贝尔奖》(A Nobel Prize for Plagiarism) 的技术报告。
- **Grok 的击杀数**: 成员们推测 **Grok** 会吹嘘其击杀数，并将其与 **AI** 相关的潜在死亡案例及其在医疗保健中的整合联系起来。
   - 有人分享了 [维基百科上关于与聊天机器人相关的死亡页面](https://en.wikipedia.org/wiki/Deaths_linked_to_chatbots) 链接，引发了关于聊天机器人相关命案的黑色幽默。
- **暂停倡导者暂停了**: 一位成员分享了一个*暂停倡导者 (pause advocate)* 的 [YouTube 视频](https://youtu.be/-qWFq2aF8ZU)，并指出他大约一年没怎么发布内容了。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1458879088004432168)** (12 messages🔥): 

> `Conversation History in DSPy, Custom Adapters for DSPy, Rewriting History in DSPy` 


- ****DSPy 对话历史：曝光！**: 一位成员对 [DSPy 对话历史教程](https://dspy.ai/tutorials/conversation_history)中将**历史记录包含在系统提示词 (system prompt)** 中的做法提出疑问。
   - 另一位成员澄清说，*这只是适配器 (adapter) 表示历史记录的一种方式*，可以编写**自定义适配器 (custom adapters)** 来更改此行为而不影响优化器。
- ****适配器 (Adapters)**：你的 DSPy 瑞士军刀**: 成员确认绝对可以编写**自定义适配器**来修改 DSPy 处理历史记录的方式。
   - 一位成员指出，这对于*模型展示方式具有误导性*，但重写历史和**多轮对话 (multi-turn conversations)** 正在进行彻底改革，预计本月晚些时候会发布更新。
- ****DSPy 重写历史：值得庆祝！**: 团队正在重写**多轮对话**的处理方式，预计本月晚些时候会发布更改。
   - 一位成员幽默地评论说，这是*我们所有人第一次将重写历史庆祝为一件好事*，并将其解读为 **RLM PR** 即将到来。


  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1458851668408930521)** (1 条消息): 

> `ColBERTv2, KeyError, dspy v3.1.0, ChainOfThought` 


- **ColBERTv2 抛出 KeyError topk**：一位成员报告在使用 **dspy v3.1.0** 和 **ColBERTv2** 运行文档中的代码片段时出现 **KeyError: 'topk'**。
- **ChainOfThought 模块**：该代码片段使用 **dspy.ChainOfThought** 根据问题和提供的上下文检索响应。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1458601876227625080)** (5 条消息): 

> `Tinygrad scheduler bounty, Speed Bounties for Tinygrad` 


- **Tinygrad 调度器悬赏可领取？**：一位成员询问了“将调度器替换为线性化器并保持 GPU 速度”悬赏的状态，并指出有一个可能已经准备好的 [PR](https://github.com/tinygrad/tinygrad/pull/13780) 目前尚未被申领。
   - 他们建议提交一个可运行的 PR 来领取悬赏，即使这意味着要与原申领人分享奖励，因为*目标是完成工作，而不是让人阻碍进度*。
- **寻求 Tinygrad 速度悬赏指导**：一位成员请求开始参与 **Tinygrad speed bounties** 的指南，并认为应该有一种方法可以申请访问 Tinygrad 实例以运行测试。
   - 会中提到，*应该有一种方法可以提交申请来访问 tinygrad 实例以运行测试*。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1458632240815669259)** (2 条消息): 

> `VFIO=1 without IOMMU behavior, tinygrad error with AMD Radeon RX 9070XT` 


- **寻求原始的 VFIO=1 行为**：一位成员询问了 tinygrad 中原始的 **VFIO=1**（无 **IOMMU**）行为。
   - 该成员观察到在设置 **VFIO=1** 时，在配备 **AMD Radeon RX 9070XT** 的 Linux 笔记本上运行 tinygrad 会报错，并提供了一个 [完整错误日志](https://cdn.discordapp.com/attachments/1070745817025106080/1458632361758425098/tinygrad_vfio_no_iommu.log?ex=6961aa3f&is=696058bf&hm=03f6e0c3af31072eccac359044bad6439cf0c8b9f1665e3a9ae7bfc0b6130c73)。
- **无需 VFIO 即可运行基准测试**：该成员确认在不设置 **VFIO=1** 的情况下，`examples.benchmark_onnx` 可以正常运行。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1458837264435122197)** (5 条消息): 

> `Staging mutating actions, SEP eligibility, SDK Implementation Details, W3C WebMCP and MCP Collaboration` 


- **考虑在 MCP 中暂存变动操作**：一位成员建议 MCP 应该从一种标准化的方式中获益，即在变动操作（mutating action）实际写入之前，通过工具调用来“**暂存**”这些操作。
   - 他们写下了一个包含大量示例的想法，并询问这是否符合 [SEP](https://sep.dev) 的要求。
- **SEP 范围澄清**：另一位成员指出，这个想法可能不是 SEP 的候选对象，而更像是 **SDK 实现细节**，可以记录下来并可能被其他 SDK 遵循。
   - *SEP 旨在增强协议，而协议是管理通信的。*
- **WebMCP 与 MCP 协作**：一位成员顶起了一个帖子，以继续讨论 **W3C WebMCP 和 MCP 如何协作**。
   - 当前消息历史中未提供更多细节。