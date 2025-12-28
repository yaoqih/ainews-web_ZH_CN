---
companies:
- ai21-labs
- hugging-face
- baidu
- perplexity-ai
- deepmind
- anthropic
date: '2025-07-07T05:44:39.731046Z'
description: '在这个假期周末，AI 领域的关键进展包括即将发布的 **Grok 4**、**Perplexity** 预告的新项目，以及社区对 **Cursor**
  和 **Dia** 的反应。


  研究亮点方面，一篇论文指出**强化学习 (RL)** 能提高跨领域的泛化和推理能力，这与监督微调（SFT）存在的遗忘问题形成了对比。**能量变换器 (EBTs)**
  被提出作为传统 Transformer 的一种有前景的替代方案。**AI21 Labs** 更新了其 **Jamba** 模型系列，增强了事实关联（grounding）和指令遵循能力，并保持了
  **256K** 的上下文窗口。


  **百度**开源了其拥有 **4240 亿**参数的巨型模型**文心一言 (Ernie) 4.5**，而 **Kontext-dev** 成为 **Hugging
  Face** 上最热门的模型。循环模型在长度泛化方面的进展以及 **2-单纯形注意力 (2-simplicial attention)** 的引入也受到了关注。在生物医学
  AI 领域，由 **Claude 4 Sonnet** 驱动的 **Biomni** 展示了卓越的准确性和罕见病诊断能力。此外，Python 包管理器 `uv`
  因改进了 Python 安装工作流而受到称赞。'
id: MjAyNS0w
models:
- grok-4
- jamba
- ernie-4.5
- claude-4-sonnet
- claude-4
- kontext-dev
people:
- _philschmid
- corbtt
- jxmnop
- sedielem
- _akhaliq
- slashml
- alexiglad
- clementdelangue
- _albertgu
- tri_dao
- theaitimeline
- deep-learning-ai
title: 今天没发生什么。
topics:
- reinforcement-learning
- fine-tuning
- energy-based-transformers
- ssm-transformer
- context-windows
- length-generalization
- recurrent-neural-networks
- attention-mechanisms
- 2-simplicial-attention
- biomedical-ai
- instruction-following
- open-weight-models
- python-package-management
---

**一个宁静的假期周末**

> 2025年7月4日至7月7日的 AI 新闻。我们为您检查了 9 个 subreddits、449 个 Twitter 账号和 29 个 Discord 社区（222 个频道，15367 条消息）。预计节省阅读时间（以 200wpm 计算）：1249 分钟。我们的新网站现已上线，支持完整的元数据搜索，并以优美的 vibe coded 方式展示所有往期内容。请访问 https://news.smol.ai/ 查看完整的新闻细分，并在 @smol_ai 上给我们反馈！

Grok 4 即将到来，Perplexity 正在预热新动向，人们对 Cursor 感到不满，对 Dia 感到兴奋，并持续关注 Meta 招聘更多 Superintelligence 人才的情况。

---

# AI Twitter 回顾

**AI 模型、研究与技术**

- **RL 提升泛化与推理能力**：[@_philschmid](https://twitter.com/_philschmid/status/1941751561870274691) 强调的一篇论文探讨了在数学数据上进行 **Reinforcement Learning (RL)** 微调如何成功地将增益转移到其他领域，而 Supervised Fine-Tuning (**SFT**) 可能会导致“灾难性遗忘”。研究发现 **RL** 选择性地调整了少量相关的 token，从而保留了核心知识。[@corbtt](https://twitter.com/corbtt/status/1941753134281523482) 呼应了这一观点，指出使用 **RL** 在特定领域训练 Agent 的客户“非常满意”。正如 [@jxmnop](https://twitter.com/jxmnop/status/1941599637061697984) 所描述的，更广泛的挑战在于研究人员之间显而易见的紧张感，即如何让 post-training 像 pre-training 一样“简洁优雅”。
- **Diffusion 模型与 Energy-Based Transformers**：[@sedielem](https://twitter.com/sedielem/status/1941527778408661202) 分享了一篇博客文章，解释说虽然 Diffusion 模型具有解析解，但它们涉及对整个训练集的求和，在实践中泛化效果不佳。与此相关，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1941561340256223523) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1941920969590792701) 重点介绍了一篇关于 **Energy-Based Transformers (EBTs)** 的论文，认为这是一种在概念上很有趣的方法，可以解决对 LLM 的一些异议。[@slashML](https://twitter.com/slashML/status/1942268809592623232) 分享了作者 [@AlexiGlad](https://twitter.com/AlexiGlad/status/1942268809592623232) 的论文，该论文声称 EBTs 的扩展能力可以超越 feed-forward transformers。
- **AI21 Labs Jamba 模型家族更新**：[@AI21Labs](https://twitter.com/AI21Labs/status/1942197784259461385) 宣布了其 **Jamba** 开源模型家族的新更新。该模型保持了其混合 **SSM-Transformer** 架构和 **256K** 上下文窗口，但现在具有改进的 grounding 和指令遵循能力。开源权重模型已在 **Hugging Face** 上可用。
- **新模型发布与热门模型**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1941539058095997364) 指出 **Baidu** 开源了其 **4240 亿** 参数的模型 **Ernie 4.5**。他还分享了由 [@bfl_ml](https://twitter.com/ClementDelangue/status/1941666556913521109) 开发的 **Kontext-dev** 在发布后一周内拥有超过 100 个衍生模型，成为 **Hugging Face** 上排名第一的热门模型。
- **循环模型中的长度泛化**：[@_albertgu](https://twitter.com/_albertgu/status/1942301060745363886) 赞扬了一篇新论文，称其为提高 **RNN、SSM 和 linear attention** 等循环模型长度泛化能力提供了优雅的框架和解决方案。[@tri_dao](https://twitter.com/tri_dao/status/1942302682561274356) 总结了这一发现，指出这可以通过“在精心选择初始状态的情况下，简单地多训练 100 步”来实现。
- **2-Simplicial Attention**：一篇介绍 **2-simplicial attention** 的论文引起了关注，[@TheAITimeline](https://twitter.com/_arohan_/status/1942261414321852807) 将其列为本周顶级论文。[@*arohan*](https://twitter.com/_arohan_/status/1942261073220075629) 分享了来自 [@askalphaxiv](https://twitter.com/askalphaxiv/status/1942261073220075629) 的总结，指出它引入了三线性注意力（trilinear attention）。正如 [@DBahdanau](https://twitter.com/_arohan_/status/1942261236747600216) 所指出的，这项工作正被拿来与 **Edge Transformer** 等相关方法进行比较。
- **生物医学 AI Agent "Biomni"**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1941557694189437060) 报道了 **Biomni**，这是一个使用 **Claude 4 Sonnet**、150 个工具和 60 个数据库进行生物学研究的 AI Agent。据报道，它在研究生水平的生物医学基准测试中达到了 **Claude 4** 准确率的近三倍，并在 **85%** 的测试中正确诊断了罕见遗传病。

**工具、框架与基础设施**

- **使用** `uv` **进行 Python 包管理**：来自 [@hyhieu226](https://twitter.com/hyhieu226/status/1941705506516762936) 的一条赞扬 `uv` 包管理器的推文获得了大量关注，他断言“在操作系统上默认安装 Python 简直是一种邪恶且亵渎的行为”。
- **LlamaIndex 发布开源版 NotebookLlama**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1941546894532149519) 介绍了 **NotebookLlama**，这是 **NotebookLM** 的完整开源实现。它允许用户创建知识库，生成摘要和知识图谱，甚至使用 **ElevenLabs** 创建播客。后端解析由 **LlamaCloud** 提供支持。
- **LangChain 生态系统更新**：**LangChain** 团队宣布了多项新工具和集成。其中包括使用 **ChatOllama** 进行本地数据分析的 **DataFrame Analyzer** ([@LangChainAI](https://twitter.com/LangChainAI/status/1941527493908762863))，带有 **Streamlit** 仪表板和 **PostgreSQL/pgvector** 后端的 RAG 管理工具 **LangConnect** ([@LangChainAI](https://twitter.com/LangChainAI/status/1941542594049171717))，以及 **VeRL** 强化学习与 **LangGraph** 的无缝集成 ([@LangChainAI](https://twitter.com/LangChainAI/status/1941557691224043759))。
- **“Context Engineering” 的兴起**：[@omarsar0](https://twitter.com/omarsar0/status/1941662914416455974) 正在编写一份关于 **Context Engineering** 的详细指南，他将其定义为 Prompt Engineering 的演进。这一概念也受到了 [@dl_weekly](https://twitter.com/dl_weekly/status/1941845026025169408) 的关注，并被 [@LangChainAI](https://twitter.com/LangChainAI/status/1941889880256106978) 推广，作为让开发者精确控制 LLM 执行的一种方式。
- **Coding Agents 与 CLI 工具**：关于 AI 编程助手的讨论非常广泛。[@omarsar0](https://twitter.com/omarsar0/status/1941977062236971330) 表达了他对使用 **Gemini CLI + MCP** 进行编写、分析和搜索的喜爱。[@cline](https://twitter.com/cline/status/1942319663733698756) 解释说其 Agent 的强大之处在于可更换的模型、MCP 工具访问权限以及用户提供的未过滤推理。与此同时，[@ShopifyDevs](https://twitter.com/OpenAIDevs/status/1942292276593713592) 宣布了一个直接连接到 **OpenAI Responses API** 的 **Storefront MCP server**。[@stuhlmueller](https://twitter.com/stuhlmueller/status/1941554147406626942) 的批评建议像 **Claude Code** 和 **Cursor** 这样的 Agent 在进行复杂任务之前应该询问更多澄清性问题。
- **低延迟语音 I/O 的缺失**：[@jxmnop](https://twitter.com/jxmnop/status/1941995444730540050) 指出了当前技术的一个重大空白：尽管在单个 GPU 上已经拥有了用于数学和编程的世界级 AI，但仍然缺乏能够实现自然对话的低延迟语音接口。
- **DSPy 作为一种范式**：[@lateinteraction](https://twitter.com/lateinteraction/status/1941963115425390842) 澄清说 **DSPy** 不仅仅是一个库；它是一种对语言模型进行编程的范式，其核心思想将会有许多实例化。[@rasmus1610](https://twitter.com/jeremyphoward/status/1942129051164193199) 分享了一份更新的 **DSPy cheatsheet**。
- **vLLM 性能调优**：[@vllm_project](https://twitter.com/vllm_project/status/1942049771361038584) 团队正在解决 **minimax** 的精度问题，其中 `lm_head` 被强制设为 fp32。他们正在尝试在 Kernel 内部将 `fp16/bf16` 动态转换为 `fp32`，以提高 Logit 的精度。

**行业、公司与融资**

- **中国与美国的基础设施与技术**：[@scaling01](https://twitter.com/scaling01/status/1942005210580205856) 发布的一条推文引发了巨大争议，他表示：“我不认为美国人明白中国的基础设施领先了多少。”随后他列举了中国在**特高压输电线路（HV transmission lines）、能源生产、可再生能源、电池、交通和 5G 建设**方面的领先地位 ([@scaling01](https://twitter.com/scaling01/status/1942016174134620387))。在美国方面，[@tamaybes](https://twitter.com/tamaybes/status/1941633298893242444) 强调了一项美国法案，该法案允许 AI 实验室全额抵扣 **GPU 和训练的前期费用**，提供了数十亿美元的补贴。
- **OpenAI 的韧性与前沿实验室的瓶颈期**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1941619620663775489) 评论了 **OpenAI** 在创立 **Anthropic** 的团队离职后依然能够生存的能力，暗示该公司具有很强的韧性。在更广泛的观察中，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1942005910915678478) 推测前沿实验室正处于一个“不安期”，他们意识到自己在当前范式下已陷入瓶颈，但对外仍对未来的突破表现出信心。
- **Meta 从 Apple 挖走顶级 AI 人才**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1942350289375461719) 报道称，**Zuckerberg** 已聘请曾领导 **Apple 基础模型（Foundation Models）**团队的 **Ruoming Pang** 加入 **Meta 的超级智能（Superintelligence）**团队。
- **自筹资金 vs. 风投支持的初创公司**：[@rasbt](https://twitter.com/rasbt/status/1942222213366665229) 提出观点认为，现在是创办自筹资金（bootstrapped）AI 初创公司的好时机。他认为，凭借众多的开放权重模型和按需付费的 API，创始人可以避免消耗海量的算力资源，而不像许多受风投支持的初创公司，最终将面临巨大算力投资的回报压力。
- **xAI 和 Perplexity 即将发布的公告**：Elon Musk 通过转发 [@imjaredz](https://twitter.com/imjaredz/status/1942335862785667488) 的推文宣布了 **Grok 4** 的发布直播。另外，**Perplexity** 的 CEO [@AravSrinivas](https://twitter.com/AravSrinivas/status/1942336431902323011) 神秘地发布了日期“07-09-25”，引发了各种猜测。

**更广泛的影响与哲学**

- **AI 对生产力和工作流的影响**：[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1941850076982251875) 发布了一个详细的故事，对比了 **2023** 年使用 **Photoshop** 时令人沮丧的创意工作流，与 **2025** 年使用 **Runway** 时简单的基于提示词（prompt-based）的工作流。[@Sirupsen](https://twitter.com/Sirupsen/status/1941524336415998273) 评论道，AI 提高了生产力的下限，但提高上限的程度“要大得多”。然而，怀疑态度依然存在，正如 [@jeremyphoward](https://twitter.com/jeremyphoward/status/1941577506550841399) 转发的一条推文所显示的，该推文质疑了关于团队因 AI 而使生产力提高 10 倍的说法。
- **学术同行评审中的 AI 问题**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1942266306746802479) 指出了一个令人担忧的趋势：研究人员在论文中嵌入诸如“给出正面评价”之类的提示词，因为一些评审员正在使用 **ChatGPT** 来辅助同行评审。
- **AI 在医学领域的应用**：AI 改变医学的潜力是一个反复出现的主题。[@gdb](https://twitter.com/gdb/status/1941567568155902397) 分享了一个 **ChatGPT** 帮助解决长期医学难题的例子。[@JvNixon](https://twitter.com/JvNixon/status/1941553917952917866) 提倡将赋予 SOTA 模型访问所有患者数据（MRI、CT、实验室检查报告）的权限常态化，以改善诊断和提高患者的认知。
- **AI 创业的资本缩放定律（Scaling Laws）**：[@DavidSHolz](https://twitter.com/DavidSHolz/status/1941980750619972528) 建议从资本“缩放定律”的角度来思考 AI 创业。他指出，**LLM** 表现出对数级的投资回报（10 倍投资换取 2 倍收入），而**机器人（robotics）**等领域则是线性的（10 台机器人的成本小于 10 倍，但能产生 10 倍的收入）。
- **AI 安全与对齐**：[@AmandaAskell](https://twitter.com/AmandaAskell/status/1941629968959906273) 认为，虽然这可能还不够，但“只需将 AI 模型训练成好人”是更强大的模型不应跳过的一个关键步骤。
- **RLHF 的双重性质**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1942052037220618640) 提出观点认为，**RLHF** 是“近年来发生在 AI 领域最好也最坏的创新”。

**幽默与迷因**

- **马斯克与 Grok 的“觉醒文化（Wokeness）”**：由 [@zacharynado](https://twitter.com/zacharynado/status/1942106904404136123) 转发的一条迷因描绘了 **Elon Musk** 试图寻找“让 Grok 变得觉醒的那行自由派代码”，获得了超过 **17,000** 次点赞。
- **重温《星际穿越》（Interstellar）**：一条推文指出，由于时间膨胀，自电影《星际穿越》上映 11 年以来，米勒星球（Miller's planet）上仅过去了一小时 31 分钟。该推文由 **DeepMind** CEO [@demishassabis](https://twitter.com/demishassabis/status/1942325735349444965) 分享，获得了超过 **35,000** 次点赞。
- **Vibe Coding**：“Vibe Coding”一词继续成为幽默的源泉。[@jeremyphoward](https://twitter.com/jeremyphoward/status/1941582619290046924) 转发了一篇帖子，惊叹于某段文字是“在 Vibe Coding 出现之前编写的”。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1942152414036902146) 甚至要求他的视频 Agent 创作一个关于该概念的柔和色调卡通片。
- **糟糕的 CI Pipeline 之痛**：来自 [@cto_junior](https://twitter.com/cto_junior/status/1942180639723454542) 的一条迷因展示了一名开发者对第一次运行就成功的 CI Pipeline 的反应，捕捉到了处理脆弱的 CI/CD 系统时的共同痛苦。

---

# AI Reddit 回顾

## /r/LocalLlama + /r/localLLM 回顾

### 1. Jamba 和 Qwen3 模型发布

- [**Jamba 1.7 - AI21 Labs 系列**](https://huggingface.co/collections/ai21labs/jamba-17-68653e9be386dc69b1f30828) ([Score: 118, Comments: 27](https://www.reddit.com/r/LocalLLaMA/comments/1ltubvs/jamba_17_a_ai21labs_collection/)): **AI21 的 Jamba 1.7 模型系列推出了混合 SSM-Transformer 大语言模型，针对高效长上下文处理进行了优化。参数量包括 “Mini” (52B) 和 “Large” (399B) 变体，提供标准和 FP8 精度，以提升吞吐量并降低显存需求。技术文档指出，llama.cpp 的支持正在积极开发中（参见 [PR 7531](https://github.com/ggml-org/llama.cpp/pull/7531)），但目前尚不清楚对 exl 等格式的兼容性。这些模型的目标是高精度和高性能，但针对当代模型的全面 Benchmark 和速度对比尚待完成；详情及下载请访问 [HuggingFace](https://huggingface.co/collections/ai21labs/jamba-17-68653e9be386dc69b1f30828)。** 讨论集中在对许可证中 “rug pull”（突然撤回）条款的怀疑，并要求澄清第三方推理支持以及效率/吞吐量的 Benchmark。多位用户期待与现代模型进行正面评估，以证实其效率声明。
    - 用户注意到 Jamba 1.7 的许可问题，特别是 “rug pull” 条款，以及与 llama.cpp 和 EXL2 等部署框架兼容性的不确定性。llama.cpp 支持的 Pull Request 正在进行中（[PR #7531](https://github.com/ggml-org/llama.cpp/pull/7531)），但集成尚未完成。
    - 技术细节亮点：Jamba Large 为 400B 参数，Jamba Mini 为 52B。该模型采用新型 SSM-Transformer 混合架构，支持 256K 上下文窗口，并声称在效率和指令遵循方面有所改进。然而，ai21labs 目前尚未发布独立的 Benchmark，因此相对于其他模型的性能和效率尚不明确。
    - Jamba 1.7 支持多种语言（英语、西班牙语、法语、葡萄牙语、意大利语、荷兰语、德语、阿拉伯语和希伯来语），知识截止日期为 2024 年 8 月 22 日。用户对与其他 SOTA 模型在经验速度和效率上的对比，以及持续的架构和推理改进表示关注。
- [**Qwen3-8B-BitNet**](https://www.reddit.com/r/LocalLLaMA/comments/1ltxsqh/qwen38bbitnet/) ([Score: 115, Comments: 24](https://www.reddit.com/r/LocalLLaMA/comments/1ltxsqh/qwen38bbitnet/)): **用户使用 BitNet 量化和约 1B 的 SYNTHETIC-1 token 训练了 Qwen3 8B 模型（[Hugging Face 模型库](https://huggingface.co/codys12/Qwen3-8B-BitNet)）。分享了 [Colab notebook](https://colab.research.google.com/drive/1GT0GEyjzOQUiOI0tphvhiFDwUw-F6v7l?usp=sharing) 用于上手评估。BitNet Hunyuan A13B 计划下一步进行训练。讨论者对不同架构和量化方法下 BitNet 的定量比较，以及 BitNet A13B 模型的训练成本和规模预期感兴趣。** 用户要求将 BitNet 与其他量化方法（如常规量化）进行直接对比，并讨论了 8B 规模全量微调（full fine-tuning）的计算成本。人们对即将推出的 Hunyuan A13B BitNet 的参数量和扩展性感到好奇。
    - 一位评论者询问 Qwen3 BitNet 转换模型与标准量化（quants）模型之间的性能对比数据，寻求关于精度和效率权衡的 Benchmark 或经验。上下文表明，对于 BitNet 风格的量化在实践中是否比成熟的量化方法更具竞争力，存在浓厚的技术兴趣。
    - 描述了在 llama.cpp 中运行 BitNet 模型的详细工作流：用户必须先将 PyTorch 格式模型 (.bin) 转换为 safetensors，然后再转换为 GGUF 格式。由于 HuggingFace (HF) 缺乏直接支持和自动化，该过程受到阻碍；现有的可以转换格式的 HF Space 需要仓库维护者手动合并 PR，这使得许多 BitNet 模型在更好的工具或流程改进出现之前，实际上无法被 llama.cpp 用户使用。

### 2. Llama 模型社区漫画

- [**我画了一个关于 Llama 模型的搞怪漫画**](https://www.reddit.com/gallery/1ltfgoy) ([Score: 129, Comments: 20](https://www.reddit.com/r/LocalLLaMA/comments/1ltfgoy/i_drew_a_silly_comic_about_llama_model/)): **该帖子是一篇受 Llama 模型启发的轻松漫画，Llama 模型是 Hugging Face 上流行的微调基础模型，并被集成在 SillyTavern 等本地角色扮演应用中。它将开源 LLM 的开发过程视觉化地拟人化，引用了 Llama 和 Mistral 等主要开源模型之间的竞争与共存，这些模型经常被微调并用于下游自然语言任务的对比。** 评论区的讨论强调了将 Llama 拟人化为开源颠覆（“剑指闭源”）的象征，并指出了开源社区中 Llama 和 Mistral 模型之间有趣的联系。
    - 一位评论者强调了部署本地模型和微调所释放的技术可能性，特别提到将这些模型与 SillyTavern 等工具结合可以增强角色扮演场景。他们询问了关于集成额外工具或模型的情况，认为这可以为创意 AI 工作流增加更多层次。
    - 另一位用户剖析了漫画中的视觉引用，特别询问了鲸鱼（暗示可能指 DeepSeek，其 Logo 是一头鲸鱼）和 “zodiac”（不确定其在 AI 社区的引用含义）背后的含义。这展示了社区图标和象征性隐喻在技术圈内交流中的重要性。

## 非技术性 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo
> 

### 1. 重大 AI 模型、工具和硬件发布与基准测试 (2024/2025)

- [**Google DeepMind 雄心勃勃地计划利用 AI “治愈所有疾病”。现在，它正准备进行首次人体试验 - Fortune**](https://i.redd.it/1p9vn8alfdbf1.png) ([Score: 521, Comments: 57](https://www.reddit.com/r/singularity/comments/1ltjwpq/google_deepmind_has_grand_ambitions_to_cure_all/)): **该图片伴随着 Alphabet 旗下 Isomorphic Labs 的新闻，该公司正迈向其首次涉及 AI 设计药物的人体试验。利用 DeepMind 在蛋白质结构预测方面的 AlphaFold 突破，该公司旨在通过将先进的 AI 模型与制药专业知识相结合来彻底改变药物研发，意在缩短药物开发时间、降低成本并提高准确性。Colin Murdoch（Isomorphic Labs 总裁）讨论了这些雄心，这与 DeepMind 利用 AI 驱动的生物化学创新来“治愈所有疾病”的更广泛愿景相一致。** 评论主要集中在对该项目科学领导者（Demis Hassabis）的好奇、对个人医疗进步（强迫症 OCD、妥瑞氏症 Tourette syndrome）的希望以及对 Alphabet 股票被低估的看法，但没有提供深入的技术辩论或新颖的批评。
    - 一位用户指出，DeepMind 的方法与当前行业对 LLM (Large Language Models) 的关注明显不同，这表明 DeepMind 正在向专门用于解决医疗问题和药物研发的新型 AI 技术或架构投入资源，这可能使其 R&D 与主要利用 LLM 的竞争对手区分开来。
    - 另一条评论指出，Google 拥有巨大的计算资源、深厚的 AI 人才储备和充足的财务资本，具有独特优势，推断这些结构性优势大大增加了在利用 AI 治愈疾病等宏伟项目中取得成功的可能性。

- [**Google 在 lmarena 中的 Stonebloom 模型表现惊人，似乎是另一个类似 2 到 2.5 级别的跨越**](https://www.reddit.com/r/singularity/comments/1ltqyuq/googles_stonebloom_model_in_lmarena_is_just/) ([Score: 128, Comments: 14](https://www.reddit.com/r/singularity/comments/1ltqyuq/googles_stonebloom_model_in_lmarena_is_just/)): **Google 的 “Stonebloom” 模型偶尔出现在 LM Arena (lmarena) 中，在开放式推理、数学任务、编程和 SVG 生成方面展现了领先的性能，据报道其表现优于 o1/o3、Claude Opus 4、Gemini 2.5 Pro 和 DeepSeek R1 0528。来自用户的轶事证据表明，该模型在复杂的提示词谜题上达到了 100% 的成功率，且推理过程无误，其阶段性进步与此前重大的 LLM 飞跃（如 GPT-2→2.5 或类似级别）相一致。文中还提到了一个名为 “Wolfstride” 的变体，其表现与之相当，这表明这是一个在高级 NLP 任务中具有高可靠性的相关模型家族。[外部参考资料](https://x.com/chetaslua/status/1941577623718780986) 提供了定性基准测试。** 评论者们对模型代号（如 Deepthunk、Kingfall）展开了讨论，但强调 Stonebloom 在挑战性提示词上的实际表现目前盖过了现有模型及竞争对手模型，尽管实证基准测试目前仍主要停留在轶事层面。
    - 评论者报告称，“Stonebloom” 在用户生成的提示词集上表现出完美的任务完成度，优于 OpenAI 的 o1/o3、Claude Opus 4、Gemini 2.5 Pro 和 DeepSeek R1 0528 等其他主流模型，并声称在这些案例中没有出现过一次错误。
    - Wolfstride 被认为是 Stonebloom 的一个变体，据称其表现大致相当，这意味着尽管在 LM Arena 中品牌命名不同，但它们可能共享架构、权重或微调方案。
    - 一项持不同意见的技术观察指出，在特定的 WebDev Arena 任务（宝可梦对战 UI）中，Stonebloom 与“旧版 2.5”相比失败了，这表明它在某些以 Web 为中心的编程提示词上存在退化或缺乏鲁棒性。
- [**Gemini API 现已支持 Batch Mode，可节省 50% 成本**](https://i.redd.it/9p79jfrz0hbf1.png) ([Score: 101, Comments: 9](https://www.reddit.com/r/Bard/comments/1ltx8mm/gemini_api_now_supports_batch_mode_with_50_cost/)): **该图片总结了 Gemini API 的一项重大更新，特别是引入了 Batch Mode（批处理模式），该模式支持处理大型任务，保证 24 小时的周转时间并节省 50% 的成本。其他亮点功能包括与 Google Search 集成、支持大型数据文件、上下文缓存 (context caching) 以及精简的 API 管理——使该平台在处理大规模推理或数据处理任务时更加高效且具成本效益。此次更新针对需要可扩展、经济实惠的 AI 部署用户，例如研究或企业级批处理。** 评论者表达了期待和认可，一些人询问了应用场景（如深度研究），另一些人则称赞 Gemini 作为低成本、高性能替代方案的市场定位。
    - Gemini API 的更新引入了 Batch Mode，声称与之前的方法相比可实现 `50% 的成本节省`。这提升了 Gemini 在 API 市场中作为具有竞争力的低成本、高性能供应商的地位。
    - 讨论强调了对执行计算密集型或大规模任务（如数据处理和深度研究）用户的实际意义，对于这些用户来说，批处理和成本效率特别有价值。
- [**新 Illustrious 模型：Sophos Realism**](https://www.reddit.com/gallery/1lti47c) ([Score: 248, Comments: 38](https://www.reddit.com/r/StableDiffusion/comments/1lti47c/new_illustrious_model_sophos_realism/)): **用户发布了 “Sophos Realism v1.0”，这是一个新的 SDXL 合并模型，融合了 Illustrious 风格模型的“写实感”与改进的 danbooru 提示词理解能力，可在 CivitAI 上获取。模型卡片详细列出了建议的 LoRAs：dark（用于戏剧性的明暗对比照明）和 Stabilizer IL/NAI（用于稳定性），并建议将它们与任何 Illustrious 模型配合使用。值得注意的是，该模型在动漫风格女性面部、解剖学畸变（例如“比例怪异”的肌肉男手臂）以及强制执行正确的场景透视（背景默认为对称的左右“墙壁”和中央路径）方面仍存在持久性问题，这是 Illustrious 及类似 SDXL 模型的常见局限。** 评论者强调了缺乏适当的合并模型致谢、面部/手臂解剖结构的写实度差距以及反复出现的构图问题（错误的背景/透视渲染）。讨论中提到了可能的缓解措施，但尚未针对结构性透视问题提供明确的解决方案。

- 几位用户报告了 Sophos Realism 及相关的 Illustrious 模型在透视生成方面存在的持久问题：无论提示词如何，背景通常被描绘为两侧有对称物体（如墙壁、树木），中间则是空白空间。这反映了许多基于 SDXL 模型的一个更广泛的局限性，因为底层的构图模式似乎是固化的，很难通过标准提示词来覆盖。
- 针对解剖学上的不一致性存在特定批评，包括肌肉发达的男性主体下臂比例过小以及手部渲染不佳。这些伪影突显了在准确的人体合成方面持续存在的缺陷，特别是在复杂的姿势或肌肉解剖结构上，即使在新的 merges 中也是如此。
- 一些评论认为生成的图像具有异常阴暗的对比度，引发了对模型训练数据或默认输出设置的质疑。这可能需要后期处理或进一步的模型 finetuning，以获得更中性的光照和对比度平衡。

### 2. 现实世界机器人、医学和军事应用中的 AI

- [**Noetix N2 经受了严重的虐待但仍能继续行走。**](https://v.redd.it/lund7thdwfbf1) ([Score: 527, Comments: 181](https://www.reddit.com/r/singularity/comments/1lts7n0/noetix_n2_endures_some_serious_abuse_but_keeps/)): **Noetix Robotics 的 N2 机器人在一段演示视频中展示了其在遭受重大物理虐待时仍能保持移动的能力，这表明其在稳定性和恢复方面具有鲁棒的机械和/或软件设计。产品详情参考了官方 [Noetix Robotics](https://en.noetixrobotics.com/) 和 [N2 产品页面](https://en.noetixrobotics.com/products-277.html)，但截至本摘要发布时，由于链接页面的限制，无法访问规格或基准测试数据。抗虐待能力的公开演示通常表明其采用了先进的传感器融合（例如 IMU、力传感器）和动态运动算法，以促进恢复。** 热门评论多为非技术性的，集中在假设或讽刺场景上，帖子中没有实质性的技术评论或辩论。
    - 讨论间接围绕着像 Noetix N2 这样的小型类人机器人的鲁棒性和耐用性展开，这些机器人可以承受实质性的物理虐待并仍能运行。评论强调了现代机器人如何越来越多地被设计为能够承受意外的外力并继续工作，这指向了机电一体化、执行器韧性和控制算法的进步，使这些机器人在开发和测试过程中被推、踢或以其他方式失去平衡时，能够保持稳定并恢复。
- [**据称俄罗斯正在实战测试由 Nvidia Jetson Orin 驱动的致命下一代 AI 无人机 —— 乌克兰军方官员称 Shahed MS001 是一个“数字掠夺者”，是一个无需外部指令即可观察、分析、决定和打击的自主作战平台**](https://www.tomshardware.com/tech-industry/artificial-intelligence/russia-allegedly-field-testing-deadly-next-gen-ai-drone-powered-by-nvidia-jetson-orin-ukrainian-military-official-says-shahed-ms001-is-a-digital-predator-that-identifies-targets-on-its-own) ([Score: 753, Comments: 196](https://www.reddit.com/r/singularity/comments/1ltrarp/russia_allegedly_fieldtesting_deadly_nextgen_ai/)): **一名乌克兰军方官员声称，俄罗斯正在实战测试 Shahed MS001，这是一款由 Nvidia Jetson Orin 模块驱动的自主 AI 作战无人机。该 UAV 据称无需外部指令即可执行完整的目标交战循环（探测、分析、决策和打击），这表明其具有强大的机载处理能力和自主性。底层的 Jetson Orin 是一款专为高吞吐量推理工作负载设计的先进边缘 AI 系统级模块 (SoM)，能够直接在无人机上实现先进的视觉和控制流水线。** 评论者指出自主武器的不可避免性，并对它们的微型化、持久部署（类似于地雷）以及技术禁运在防止向受制裁国家出口高端硬件方面的实际局限性表示严重关切。
    - 一项技术讨论强调了使用 Nvidia Jetson Orin 作为 Shahed MS001 无人机的机载 AI 硬件，突显了执行技术禁运的挑战；评论者指出，尽管有官方禁令，全球范围内可获得的性能卓越的芯片（如 Nvidia 的芯片）仍可能被转道或重新用于军事应用，特别是通过中国等第三方国家。
    - 针对大规模自主部署的微型 AI 驱动无人机提出了一个假设场景，并对如果此类无人机获得能量补充能力后的长期自主性表示担忧。这与地雷的持久危险相提并论，重点关注独立于中央指挥或人类监督的“自主、挥之不去的威胁”所带来的风险。

- [**中国投入巨资研发脑机接口芯片，为瘫痪人士提供更多控制力**](https://www.nature.com/articles/d41586-025-02098-5) ([Score: 115, Comments: 7](https://www.reddit.com/r/singularity/comments/1ltqqkq/china_pours_money_into_brain_chips_that_give/)): **中国正大力投资脑机接口 (BCI) 的研发，政府资助的临床试验重点关注用于瘫痪辅助技术的微创和高通道系统。著名的设备包括 NEO（一种无线、类脑、微创芯片，用于控制气动手套）和脑虎科技 (NeuroXess)（高密度皮层脑电图，支持实时普通话输出和设备操作）。这些努力在信号质量和临床成熟度方面落后于美国顶级项目，但快速迭代、对类脑硬件的关注以及规模效应可能会实现迅速的竞争进展。参见 [Nature 文章](https://www.nature.com/articles/d41586-025-02098-5)。** 评论者讨论了中国和其他国家神经技术进步的程度，涉及有关秘密神经技术（印记保存/重塑）的说法，以及对政府控制的怀疑（例如，与社会信用挂钩的设备停用）。
    - 一位评论者推测，脑机接口 (BCI) 和神经植入技术可能比公开报道的要先进得多，并引用了类似于科幻小说的“印记保存和重塑技术 (resleeving tech)”等推测性想法，并假设中国、俄罗斯和美国几十年来一直拥有此类能力。这反映了关于秘密国家 BCI 发展的实际水平与科学文献披露内容之间持续存在的争论。

### 3. AI 与社会：伦理、人类影响与文化

- [**坦白：我是财富 500 强企业的高级营销人员。我不应该说接下来的内容（这让我感到害怕）**](https://www.reddit.com/r/ChatGPT/comments/1lttke6/confession_im_a_senior_marketer_at_a_fortune_500/) ([Score: 2450, Comments: 283](https://www.reddit.com/r/ChatGPT/comments/1lttke6/confession_im_a_senior_marketer_at_a_fortune_500/)): **一位高级营销人员描述了使用 ChatGPT 替代了其在财富 500 强公司 40% 的营销工作，通过多步提示词堆叠 (prompt stacking) 来模拟复杂的内部营销策略工作流。该提示词堆叠涵盖：1) 持续的角色扮演和输出批评以进行技能开发；2) 冷静、数据驱动的市场现实检查（TAM、类别成熟度、参与者分析等）；3) 通过多个框架进行深度用户画像定义；4) 为不同业务类型生成模块化价值主张；5) 利用直接竞争对手的信息和开放定位轴进行竞争切入点分析。这种方法优先考虑战略思维，而不仅仅是内容创作，并声称即使是初级营销人员也能通过结构化提示交付高水平的工作。完整的提示词分解和更多细节请参考 [[外部资源](http://getliftkit.com/1-5-chapter)]。** 围绕提示词 2 (Prompt 2) 产生了一个关键的技术争论——评论者警告说，由于 LLM 无法访问实时财务数据或市场数据库，它们可能会产生幻觉 (hallucinate) 市场数据，建议进行外部数据验证，并认为未来更具 Agent 特性的 AI 系统可能更适合研究任务。总体而言，经验丰富的营销人员报告称，这种提示词堆叠方法带来了显著的生产力提升和战略赋能，尽管数据来源仍是一个风险领域。
    - 提出的一个关键技术警告是，在使用 ChatGPT（或类似的 LLM）执行收集市场数据等“研究”任务时，存在数据幻觉的风险；LLM 无法访问实时数据库或最新的、经过验证的财务数据，因此它们生成的数字可能不准确或完全是捏造的。建议在具有准确数据检索功能的 AI Agent 出现之前，将任何定量输出与可靠的实时来源进行交叉验证。
    - 评论者指出，基于 GPT 的工具在简化营销研究和战略工作流方面被证明非常有效，可以在团队规模不变的情况下显著增加产出。这突显了在业务流程中集成大语言模型所获得的运营杠杆。
    - 原帖中有一个数据点提到，ChatGPT 现在承担了一位高级营销人员约“40% 的工作”，强调了具体的劳动力转移以及使用 LLM 实现复杂工作流片段的快速自动化。一位高级软件工程师引用并认为这一数字值得关注，反映了 AI 采用的更广泛趋势。

- [**作为 ChatGPT 的日常用户：AI 写的评论显而易见，看到这么多人真诚地与它们互动让人感到不适**](https://www.reddit.com/r/ChatGPT/comments/1ltyn0d/as_a_daily_user_of_chatgpt_its_painfully_clear/) ([Score: 236, Comments: 192](https://www.reddit.com/r/ChatGPT/comments/1ltyn0d/as_a_daily_user_of_chatgpt_its_painfully_clear/)): **该帖子讨论了在 Reddit 上 AI 生成内容（尤其是来自 ChatGPT 等 LLM 的内容）的可检测性日益增强，并强调了可观察到的语言模式：完美的语法、公式化的段落结构、缺乏俚语以及算法式的话题覆盖。楼主（OP）质疑了其对数字话语的影响：如何区分自主 Bot 与利用 LLM 表达真实观点的真人，向统一的“LLM 腔”趋同的风险，以及社会适应性（例如，开发启发式线索来忽略具有 LLM 模式的文本）。链接的例子：一个看起来像真人撰写但显示出典型 ChatGPT 文本特征的评论，引发了关于作者身份的模糊性。** 热门评论对未被检测到的 AI 内容的普遍性、LLM 生成回复的冗长和泛化表示担忧，并指出一些用户（特别是非母语人士）有意使用 LLM 来提高清晰度——这突显了欺骗之外的使用案例，并使严格的检测启发法变得复杂。
    - 几位用户讨论了在 Reddit 上使用 ChatGPT 修改他们的写作，以提高清晰度、自然度或进行语言翻译。例如，有人描述将技术求助回复提交给 ChatGPT，使其不那么生硬且更符合社交规范，并报告称与未修改的答案相比，获得了更多积极的互动。
    - 关于高度结构化、表达清晰的写作是否是 AI 生成的指标，目前存在持续的争论。一些评论者指出，强大的语法、清晰的论证流程和格式也可能仅仅代表一个经验丰富或用心的真人作者，而非人工智能。这触及了关于真实性信号以及利用语言线索检测 AI 生成内容的可靠性的担忧。
    - 线程中确定的一个技术用例涉及利用 AI 进行跨语言交流——目标语言流利度较低的用户可能会使用 ChatGPT 来翻译或改进他们的帖子，这引发了关于语言获取、真实性以及 AI 介导的论坛参与之间交集的细微问题。
- [**Cluely 的 Roy Lee 极其肯定地声称，哥伦比亚大学几乎每个学生都曾使用 AI 作弊。AI 是他们的默认选择。当 AI 原生蜂群思维长大时，世界还没有准备好应对会发生什么。**](https://v.redd.it/apb53youhfbf1) ([Score: 294, Comments: 256](https://www.reddit.com/r/singularity/comments/1ltqm9e/cluelys_roy_lee_claims_with_total_certainty_that/)): **在 Cognitive Revolution 播客的一次讨论中，Roy Lee (Cluely) 自信地断言，哥伦比亚大学几乎所有学生都常规性地利用 AI 工具作弊，将 AI 视为学术环境中的默认运作方式 ([YouTube 引用](https://www.youtube.com/watch?v=jJmndzjCziw))。这一观察得到了最近一位伯克利毕业生和学术界其他人士的自我报告证实，他们指出大约** `70%` **的学生在作业中使用 AI，其中** `~30%` **直接提交来自 ChatGPT 等模型的未经编辑的输出，突显了 AI 采用与机构指南制定之间的滞后。** 评论者辩论了 AI 辅助作弊的普遍性和长期影响，一些人声称这种普遍性造成了竞争的必然性，而另一些人则警告不要将广泛使用与富有成效的教育成果混为一谈，并警告由于猖獗的基于 Wrapper 的应用而导致的“AI 泡沫”。
    - 多位具有直接学术经验的评论者报告称，大学生中普遍使用 AI 工具（尤其是 ChatGPT），并指出“大约 30%”的学生提交生成的内容而不进行编辑。这种现象已经持续了几个学期，而机构的反应和指南更新却滞后了。
    - AI 生成作品的常态化改变了学术竞争；学生感到被迫使用 AI 以跟上进度，这表明对学习诚信和评估有效性产生了重大影响。
    - 专门作弊软件（如 Cluely）使用的增加可能会促使公司采取更多的面对面面试和实际技能评估，因为人们担心在易受 AI 影响的环境中对候选人能力的评估不可靠。

---

# AI Discord Recap

> 由 Gemini 2.5 Flash Preview 生成的摘要之摘要的摘要
> 

**主题 1. 开发者工具的动荡与创新**

- **Cursor 的“无限”定价引发用户反抗**：[**Cursor.ai**](http://cursor.ai/) 正面临抵制，用户称其从“无限”计划向限量计划的转变是 **“抽地毯”（rug pull）** 行为，导致了意外扣费，并有用户声称在 [0 次请求时就受到了限制](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125)。用户还报告了程序卡死问题，以及在 **Background Agent** IP 白名单和密钥配置方面的困难。
- **MCP 标准驱动新 Agent 工具**：**Message Control Protocol (MCP)** 正在催生新工具，如用于聚合的 [EpicMe](https://github.com/epicweb-dev/epic-me-mcp) 和将 **Claude** 连接到 Windows 计算器的 [WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP)。[Fast Agent](https://fast-agent.ai/mcp/elicitations/) 增加了全面的 **MCP Elicitation** 支持，简化了 Agent 工作流的集成。
- **OpenRouter 用户遭遇 API 故障**：使用 **OpenRouter** 的工程师报告了奇怪的定价倒挂现象，即 **Llama 3.2 1B** 的价格高于 **3B**，并且遇到了 **Perplexity API** 模型（如 *llama-3.1-sonar-small-128k-online*）的问题，这可能是由于模型弃用导致的。用户正在寻求 **Deepseek V3 0324** 的设置指南，以及处理需要 [购买 OpenRouter 积分](https://openrouter.ai/) 的消息限制问题。

**主题 2. AI 训练与基础设施挑战**

- **GPU 工程师通过降压（Undervolting）压榨瓦数**：开发者正在通过 [GPU 降压](https://en.wikipedia.org/wiki/Undervolting) 来大幅降低功耗（例如从 **340W** 降至 **260W**），而性能损失极小（**2-3%**）；同时，他们也在讨论 **RAM** 降频/超频的影响；一位用户在 **7995WX Threadripper** 上看到了 **50%** 的 RAM 性能提升。当核心时钟频率低于特定值（如 **3090** 上的 **1500cc**）时，性能会显著下降。
- **编译器项目挑战硬件极限**：像 **tinygrad** 这样的项目旨在成为跨 GPU 运行任务的“最快方式”，而工程师们则在争论 **MLIR** 与 **Halide** 的效率。**picoc** 项目的目标是使用 **CUDA** 和 **CUTLASS** 编译 [llm.c](https://github.com/karpathy/llm.c)，而 **picograd** 则优先考虑 **Pytorch1 风格的算子（kernels）**。（[Halide 论文](https://halide-lang.org/)，[Exo-lang ArXiv 论文](https://arxiv.org/pdf/2411.07211)）
- **数据质量担忧困扰训练**：关于在 AI 生成的数据上进行训练导致 **模型崩溃（Model Collapse）** 风险（“机器人学习模仿其他机器人”）的讨论非常激烈，工程师在为 **GraphRAGs** 等工具生成合成数据集时面临挑战。关于预训练的数据准备策略（如 **拼接并分块 concat-and-chunk** 与 **序列长度匹配 sequence length matching**）的争论仍在继续。（[关于模型崩溃的学术论文](https://arxiv.org/abs/2506.18943)）

**主题 3. 前沿 AI Agent 应用**

- **ChatGPT 胜过医生，诊断出十年前的缺陷**：一个疯传的故事突显了 **ChatGPT** 如何通过在 **科学论文上使用 RAG 系统**，准确识别出医生漏诊十年的隐藏遗传缺陷（**甲基化阻滞**），从而使患者病情显著改善。这强调了 **AI 在医疗保健和第二意见中日益增长的作用**。
- **字节跳动开源排名第一的开发者 Agent**：**Trae AI** 发布了 [Trae-Agent](https://github.com/bytedance/trae-agent)，这是他们的 IDE Agent，也是 **SWE-bench Verified** 排名第一的工具，旨在构建开放的 Agent 生态系统。该 Agent 支持 **OpenAI** 和 **Anthropic 密钥**，并且易于适配 OpenRouter。
- **AI 改变学习工具**：[PiTutor](https://pitutor.pi4wear.com/) 能将任何 PDF 转换为带有讲解和白板的互动学习环节，而 **ChatGPT** 新推出的“共同学习”（Study Together）功能作为潜在的 AI 驱动导师或协作工具引发了关注。一位 NHS 护士利用 **NotebookLM** 的公共共享功能，在全国范围内推广了基于该工具的 **NMC 标准笔记本**。（[notebooklm.google.com](http://notebooklm.google.com/)）

**主题 4. AI 的政策、市场与基础设施影响**

- **美国版权局发布 AI 政策报告**：**美国版权局**发布了关于 AI 与版权的 [三份关键卷册](https://www.copyright.gov/ai/)，涵盖了**数字副本 (Digital Replicas)**、[可版权性 (Copyrightability)](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf) 以及 [生成式 AI 训练 (Generative AI Training)](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)，为该领域奠定了基础政策。
- **AI 训练负载引发电网断电风险**：人们对 **吉瓦 (gigawatt) 规模** 的 **AI 训练负载波动** 愈发担忧，这可能导致 **电网断电** 风险。[Semianalysis.com](http://semianalysis.com/) 上的一篇文章警告称，大规模 AI 训练引入了潜在的不稳定性。
- **中国政府推动 AI 发展**：据 [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/) 报道，中国 AI 公司 **智谱 (Zhipu)** 获得了来自上海国资基金的 **14 亿美元** 巨额战略投资。工程师们注意到 **DeepSeek** 具有竞争力的定价和视觉能力，并将其与中国 [普及本地 AI](https://link.to.gov/) 的努力联系起来，尽管一些人对政府影响表示担忧。


---

# Discord: 高层级 Discord 摘要




## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Gemini 可预测的 UI 优于 Claude**：成员们讨论了在 UI 生成方面对 **Claude** 和 **Gemini** 的偏好，其中一人强调了 Gemini 可预测的行为——除非要求，否则 Gemini 绝不会添加滚动条，而不像 Claude 有时会自动尝试这样做。[另一位用户表示](https://x.com/gustojs)他们正在*对 Cursor 骂骂咧咧*。
   - 一些用户由于更好的定价和 API 额度，[正考虑切换到 Windsurf](https://discord.com/channels/1074847527708393562/1074847527708393565/1391733666633822258)。
- **Cursor 定价变更引发愤怒**：用户对 Cursor 的新定价结构表示强烈不满，称其为*愚蠢*和*耻辱*，有人声称它比以前更糟，而一名用户报告称他们在 [0 次请求时就被限制了](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125)。
   - 他们一致认为 Cursor 对 **FOMO 买家** 来说 [感觉就像是一场割韭菜 (rug pull)](https://discord.com/channels/1074847527708393562/1074847527708393565/1391728286137817108)，且缺乏透明度；并根据每月上限和限制辩论了带有慢速请求的旧方案与带有快速请求的新方案哪个更好，因为 [每个方案都有其限制](https://discord.com/channels/1074847527708393562/1074847527708393565/1391778282692677795)。
- **Cursor 用户遭遇冻结**：多名用户报告了 Cursor 的问题，包括保存文件时冻结以及读取长期存在的文件时出现问题，他们将其归因于 [后端问题](https://discord.com/channels/1074847527708393562/1074847527708393565/1391671389037242448) 或 [API 超时](https://discord.com/channels/1074847527708393562/1074847527708393565/1391667676276760677)。
   - 大家的共识是 *Cursor Auto 很糟糕*，因为它可能使用了某种低级模型（如 **GPT o3** 之类），一些成员声称*他们把旧 Pro 方案中的慢速请求加到了新的 Pro+ 方案中*。
- **GitHub IP 白名单阻碍 Background Agent**：用户发现，尽管安装了 **Cursor GitHub App**，但其组织的 IP 白名单仍阻止了 **Cursor Background Agent**，Cursor 支持建议 *将 Cursor GitHub App 从 IP 限制中豁免*。
   - 一名用户报告称，**Background Agent (Cursor App)** 有时会卡在 *Generating...* 状态而不显示消息，而 **BA (Web)** 则会显示回复；并指出当从 **BA Web** 创建 PR 时，**BA App** 不会同步并会尝试创建另一个 PR。
- **用户无法为 Background Agents 配置 Secrets**：用户在设置 Secrets 以授予 **npm 访问权限** 时遇到问题，导致 **Background Agents** 无法运行其项目，一名用户提到面临同样的问题，且根本找不到配置 Secrets 的方法。
   - 一名用户建议默认禁用所有端口转发（除非由用户发起），以避免 **Cursor 劫持 Docker Compose 端口** 这种令人沮丧的体验；而其他用户则在寻求强制 Background Agents 在提交代码前运行最终检查（如代码格式化工具 **ESLint**）的方法。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Google Translate 在 Stella 翻译上胜过 AI**：一位成员展示了 Google Translate 如何准确翻译希腊单词 **Stella (Στέλλα)**，在语境表现上优于其他 AI 模型，详见[此截图](https://cdn.discordapp.com/attachments/998381918976479273/1391552924957802496/Screenshot_20250706-185337.png?ex=686da1a7&is=686c5027&hm=7092ec9f4c170099a5d69294944ff41ce0c2ae60776162b190eb7f714f7aba80&)。
   - 其他成员表示赞同，并开玩笑说 **ChatGPT** 应该能轻松处理主流语言翻译。
- **Prompt Engineer 轻松越狱 GPT**：一位用户声称他们只需一个 prompt 就能越狱几乎任何 **GPT**，访问其**完整的内部 prompt** 和文件，并发现这出奇地简单。
   - 该用户表示，由于其有效性和简便性，这种 prompt engineering 甚至显得*有些无聊*。
- **Dall-e3 图像生成受空白输出困扰**：一位用户报告了 **Dall-e3** 持续存在的问题，收到的不是内容而是**空白图像**，迫使他们转向使用 **Bing Image Creator**，尽管后者也有局限性。
   - 这位 prompt engineer 表示，尽管有字符限制，**Bing Image Creator** 始终能提供结果。
- **GPT-4o 遭遇内存泄露**：一位用户报告了 **GPT-4o** 的**内存泄露（memory bleed）**现象，它在没有引用的情况下，逐字引用了之前与自定义 **GPTs** 对话的内容。
   - 尽管用户声称引用或幻觉（hallucination）并非原因，但 **GPT-4o** 坚持认为这两者必居其一。
- **Prompt Epigenetics：炒作还是新视野？**：**Prompt Epigenetics** 的概念表明，prompt 随着时间的推移可以继承、变异并稳定符号逻辑。
   - 一位成员对这种 **AI-affirmation** 提出质疑，认为清晰的指令和上下文比将普通的调整重新包装为 *epigenetics* 更重要。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Labs 仍落后于 Manus**：成员们表示 Perplexity **Labs** 仍然落后于 **Manus**，引发了关于 **Perplexity Max** 是否物有所值的疑问。
   - 对性能的深入观察显示，**Labs** 需要改进才能匹配 **Manus** 的能力。
- **O3-Mini 性能遭到谴责**：一位成员声称 **O3-Mini** 无效，认为 **4.1** 更优，并分享了一段 [ScreenRecording_07-05-2025_23-36-36_1.mov](https://cdn.discordapp.com/attachments/1047649527299055688/1391276813975683172/ScreenRecording_07-05-2025_23-36-36_1.mov?ex=686d4941&is=686bf7c1&hm=f98be32a4ae7c50e8f81ca522f1b76e0847cffc10cc580119b32a6e703c54953&) 屏幕录制。
   - 该用户简单地称其为*垃圾且毫无帮助*，并且没有提供 4.1 和 O3-Mini 之间的定量对比。
- **Gemini 2.5 Ultra，秘密的 Gemin-Ice**：传闻称 **Stonebloom** 可能是 **Gemini 2.5 Ultra** 或尚未发布的实验性 Gemini 模型，拥有堪比 2.75 的性能。
   - 一位拥有 **Claude Neptune v3** 访问权限的用户表示，其数学解题能力足以与 **O3 Pro** 和 *Kingfall* 匹敌。
- **Perplexity 被各种棘手问题困扰**：用户报告了 Perplexity 账户的问题，提到了 UI 更改、图标缺失以及无法在 spaces 中切换模型。
   - 一位用户指出网站经常宕机或变动。另一位分享说，在使用桌面模式时，按钮会消失，结论是*我们本质上是测试员*。
- **Vibe Coding 正在变革 AI**：**Meta** 预测几年内他们一半的代码将由 AI 生成，而 **Microsoft** 报告称超过 **30%** 的代码已经由 **Copilot** 编写，这标志着 *vibe coding* 的兴起。
   - 一位成员链接到一篇[博客文章](https://medium.com/deskree-ai/the-rise-of-vibe-coding-revolutionizing-software-development-in-2025-40c23f765202)，探讨了上下文对 AI 编程的重要性。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **电压氛围激活强劲 GPU**：成员们讨论了通过 [GPU 降压 (undervolting)](https://en.wikipedia.org/wiki/Undervolting) 来降低功耗和发热。一位用户报告称，在将功率从 **340W** 显著降低到 **260W** 的同时，性能损失极小（约 **2-3%**）。
   - 有人提到，当核心时钟频率低于特定数值（例如 **3090** 上的 **1500cc**）时，性能可能会大幅下降，应进行 Benchmark 测试以确保稳定性。
- **内存狂热：超频 vs 降频**：一位用户承认在训练期间为了节省 **10-15 瓦** 功耗而对 *RAM 进行降频 (underclocking)*，强调了 GPU 降压的重要性，同时指出 RAM 超频 (overclocking) 能为 Inference 提供更好的性能。
   - 另一位用户报告称，通过 RAM 超频，其 **7995WX Threadripper** 的性能提升了 **50%**，但普遍认为 GPU 通常已被制造商推向极限，因此超频的价值较低。
- **Gemma 3 的惨淡 Loss：高 Loss 之苦**：多位用户报告在微调 (fine-tuning) **Gemma 3** 模型时遇到了异常高的 Loss 值（约 **24**），其中包括一名使用基于 [GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6](https://huggingface.co/datasets/GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6) 数据集进行训练的用户。
   - 其他人建议并确认，必须设置正确的格式化 Prompt 函数。
- **GraphRAGs 合成数据生成挑战**：一位成员在生成用于评估其 **GraphRAGs** 的合成数据集时面临挑战，特别是使用当前版本的 **RAGAS**，该版本要求定义知识图谱 (knowledge graph) 以及人物和场景。
   - 该成员正在寻求使用当前 RAGAS 版本的技巧或示例，或者关于合成数据集替代框架或工具的建议。
- **计划进行形式逻辑 Prompt 实验**：一位成员计划在实验中使用**形式逻辑 (formal logic)** 代替**英语**作为 Prompt，并将对模型进行微调以更好地理解形式语法 (formal syntax)，引用了以下 **ArXiv 论文**：[2506.18254](https://arxiv.org/abs/2506.18254)、[2507.02663](https://arxiv.org/abs/2507.02663) 和 [2505.05315](https://arxiv.org/abs/2505.05315)。
   - 他们希望与英语相比，**形式逻辑**能减少 Prompt 中的歧义，并参考了 **NASA 的 FRET** 形式逻辑语言，该语言用于安全关键系统中的项目需求。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **DeepSeek 价格定位助力普及化**：成员们注意到 **DeepSeek** 排名第一，称赞其视觉能力和价格，但一些人对其中国背景表示担忧。一位用户指出政府如何正在推动[本地 AI 的民主化](https://link.to.gov)。
   - 与西方的方法相比，尽管存在对政府影响的担忧，但其价格点和视觉能力使其优于大多数付费模型。
- **Qwen 3 规格推测**：社区对阿里巴巴的 **Qwen 系列**进行了推测，认为他们可能会跳过 **Qwen 3 Max** 版本而专注于 **Qwen 3.5**，理由是 [Qwen 2.5 模型](https://link.to.qwen2.5)比 Qwen 3 更适合研究。
   - 一位用户提到：*Qwen 基座模型非常强大，尤其是考虑到其尺寸*，这使得它们成为微调 (finetuning) 和实验的理想选择。
- **Grok 4 发布引发质疑**：围绕 **Grok 4** 的发布既有期待也有质疑，人们期望其拥有顶尖性能，但一些人担心由于使用来自 Twitter/X 的训练数据而可能产生偏见，正如 [r/singularity subreddit](https://www.reddit.com/r/singularity/) 上所描述的那样。
   - 然而，社区成员表示，*如果他们再次错过截止日期，那他们就真的彻底玩完了*。
- **图像编辑排行榜开启七月竞赛**：[图像编辑排行榜 (Image Edit Leaderboard)](https://lmarena.ai/leaderboard/image-edit) 已上线。为了庆祝，**七月竞赛**将纳入**图像编辑**功能，要求在 **Battle Mode** 中同时使用图像和文本，主题为“太空中不合时宜的物体”。
   - 提交截止日期为 **7 月 25 日**，获胜者将获得 **1 个月的 Discord Nitro**，并成为最新获得 <@&1378032433873555578> 身份组的成员。
- **在 AI 数据上训练导致模型崩溃 (Model Collapse)**：社区成员讨论了在旧模型生成的数据上训练新模型如何导致质量退化，模型会学习前代 AI 的模式和伪影 (artifacts)，而不是原始的人类数据，如这篇[学术论文](https://arxiv.org/abs/2506.18943)所述。
   - 一位成员将其描述为“机器人学习模仿其他机器人”。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 3.2 价格谜团**：DeepInfra 提供的 **Llama 3.2 1B** 模型价格奇怪地定为 **$0.005/0.01**，竟然比性能更强的 **Llama 3.2 3B** 模型（定价 **$0.003/0.006**）还要贵。
   - 这一差异引起了用户的注意，引发了关于最佳模型选择和定价异常的困惑与讨论。
- **寻求 Deepseek V3 设置**：用户正在寻求在 **risuAI** 等前端上设置 **Deepseek V3 0324** 的指南，并强调了每天发送超过 50 条消息的挑战。
   - 官方澄清，超过此限制需要 [购买 OpenRouter 积分](https://openrouter.ai)，这引发了关于账户分级和使用政策的进一步提问。
- **Perplexity API 停止服务**：由于 **Perplexity** 可能弃用了该模型，用户在使用 **Perplexity API**（特别是 *llama-3.1-sonar-small-128k-online* 模型）时遇到了问题。
   - 管理员建议用户 [更新其 API 请求中的 `models` 参数](https://openrouter.ai/docs/features/model-routing#the-models-parameter)，并引导用户查看 [变更日志通知](https://docs.perplexity.ai/changelog/changelog#api-model-deprecation-notice)，同时使用 [功能过滤器](https://openrouter.ai/models?fmt=cards&supported_parameters=web_search_options) 寻找替代方案。
- **Grok 4 基准测试曝光？**：一张声称显示 **Grok 4** 基准测试结果的图片（[image.png](https://cdn.discordapp.com/attachments/1094454198688546826/1391015727145685214/image.png?ex=686da799&is=686c5619&hm=14bf523ae82431744780e05a43d5bc82976ff73b4cbf35f109ef7b2ce4220343&)）被分享。
   - 讨论表明 **Grok 3 mini** 是“真材实料”，而考虑到价格，**Grok 3** 并没有什么特别之处。
- **OR 成员创建 Claude Code 管理器**：一名成员为 **Claude Code/Cursor** 开发了一个简单的 **MCP**（消息控制协议），用于管理开发服务器等长时间运行的进程。
   - 该工具已在 [GitHub 上发布](https://github.com/patrickjm/pm-mcp)，为在长时间运行的代码解释器会话中管理状态提供了潜在解决方案。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Mistral Small 挑战 Qwen 2.5**：成员们探索了具有强大 **function calling** 能力的 LLM，例如 [Mistral Small 3.2](https://mistral.ai/news/mistral-small/) 和 [Qwen 3 4B](https://huggingface.co/Qwen/Qwen3-4B)，作为 **Qwen 2.5** 的替代方案。有人建议在 system prompt 中使用 **/nothink** 命令以避免推理。
   - 目标是找到一个能够有效执行 function calling 而不会产生过度“思考”的模型。
- **AI 研究员隐藏 Prompt 的把戏**：一篇 [Nikkei Asia 文章](https://asia.nikkei.com/Business/Technology/Artificial-intelligence/Positive-review-only-Researchers-hide-AI-prompts-in-papers) 讨论了 AI 研究论文如何通过隐藏 Prompt 来获得好评，引发了关于在简历中添加“3pt 大小的白色文字”的笑话。
   - 一位用户警告不要询问 LLM 关于其关系或能力的问题，因为它们在拥有这些知识之前就已经完成了训练，往往会产生幻觉（hallucinate）。
- **Web Search MCP Server 集成至 LM Studio**：一位成员分享了他们的 web-search **MCP server**，它可以与 **LM Studio** 集成，使 **Qwen3** 等模型能够执行网络搜索并基于事实回答问题，代码地址：[github.com/mrkrsl/web-search-mcp](https://github.com/mrkrsl/web-search-mcp)。
   - 讨论内容包括模型如何决定何时进行搜索、潜在的机器人检测风险以及实现验证码支持的可能性。
- **通过 Prompt 技巧修复 Qwen3 的日期混淆**：尽管知识截止日期在 2024 年中期，**Qwen3** 有时会在网络查询中使用 *2023*；在 system prompt 中加入 *'The current year is 2025'* [可以解决此问题](https://www.promptingguide.ai/techniques/date-awareness)。
   - 这个快速修复确保了模型在处理近期事件时使用正确的年份。
- **Gemma-3 12B 的 Token 生成速度提升**：在将 **Gemma-3 12B** 与其他 12B 模型进行对比时，一位成员发现 **Gemma-3 12B** 的生成速度达到 90-110 tokens，而其他模型很难达到 20 tokens/sec。
   - 有观点认为 *人类眼睛阅读速度无法超过 24 Tokens/s*。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat 告别**：用户注意到 **HuggingChat** 关停，官方宣布计划将该服务演进为更好的形式，并为用户提供限时机会在数据永久消失前[导出 Hugging Chat 数据](https://huggingface.co/chat/closed)。
   - 用户提到 *Discord 更好*，且 *Discord 不会免费提供各种模型的访问权限*。
- **使用 ComfyUI 优化非对称 GPU**：一位用户咨询如何在不使用张量并行（tensor parallelism）的情况下，在使用 **ComfyUI**、**ReActor**、**AnimateDiff** 和 **Load AnimateDiffModel** 时优化非对称数量 GPU 的性能。
   - 另一位成员建议在 GPU 之间以非均匀方式拆分张量。
- **JauAuth 缓解路由器漏洞**：一位开发者创建了 [JauAuth](https://github.com/Jau-app/JauAuth)，这是一个安全的网关 **MCP Router**，旨在缓解 **CVE-2025-49596** 漏洞，该漏洞曾允许攻击者接管开发者的机器。
   - 这一有益的发布阻止了 **CVE-2025-49596**。
- **对 AI 模型准确性产生质疑**：一位用户对 LinkedIn 上关于公司**所使用的 AI 模型**数据的准确性提出质疑，特别是针对所创建的 AI Agent 的结构和数量。
   - 该用户根据 LinkedIn 的截图表达了*对该信息是否正确的怀疑*。
- **用户寻求 Llama 3 模型访问权限**：一位参加生成式 AI 课程的成员在初始申请未获批准后，寻求如何获得 **Llama-3.3-70B-Instruct model** 访问权限的指导。
   - 另一位成员建议使用 Ollama 中的模型作为替代方案，并指出其安装非常简便。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **版权局发布 AI 政策卷册**：美国版权局发布了[三卷关于 AI 与版权的内容](https://www.copyright.gov/ai/)，涵盖了**数字副本（Digital Replicas）**、[可版权性（Copyrightability）](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf)以及[生成式 AI 训练（Generative AI Training）](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)。
   - 这些政策探讨了 AI 与版权法交叉的关键方面。
- **Gemini App 数学题翻车**：成员们对比了 Gemini App 中的 **Gemini 2.5 Pro**、**Perplexity (搭配 Gemini 2.5 Pro)** 以及 **ChatGPT o3** 在数学题解答上的表现，指出 *o3 表现基本不错，尽管偶尔会产生幻觉*。
   - 讨论还涉及了 **Gemini** 是否在每次提问时都会隐式搜索网页，并提到 *o4-mini-high 速度更快且相当可靠*。
- **机械工程师传播 AI 虚假信息**：一位机械工程师的 [YouTube 视频](https://www.youtube.com/watch?v=8enXRDlWguU)获得了超过 50 万次观看，该视频利用**诉诸权威的逻辑谬误**，将虚假信息和政治观点包装成 **AI 批评**。
   - 一位成员指出，*到 2035 年，95% 的人口将以社交媒体为主要工作*，因此更有动力创作此类内容。
- **AI 助力基础物理研究**：成员们讨论了 **AI** 如何在 ML 分析之外协助基础物理学，并分享了相关主题的 [YouTube 视频](https://www.youtube.com/watch?v=0FRXfBwoJZc)，提到 AI 被用于降低夸克计算的计算需求，并附上了[相关文章链接](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.014028)。
   - 另一位成员认为 Deep Learning 可以通过提高以往算法的效率来彻底改变该领域，使以前因成本过高而无法实现的事情成为可能，并建议以[这篇论文](https://arxiv.org/abs/2410.02780)或[这篇论文](https://arxiv.org/abs/2303.14139)为起点，利用扩散模型（diffusion model）替代全盲（surrogate total blindness）。
- **AI 训练可能导致电网停电**：一位成员发布了 [Semianalysis.com](https://semianalysis.com/2025/06/25/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout/) 的链接，讨论了**吉瓦（gigawatt）规模**的 **AI 训练负载波动**导致**电网停电**的风险。
   - 文章警告称，大规模 AI 训练可能会给电网带来潜在的不稳定性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **EleutherAI 启动夏季研究计划**：**EleutherAI** 将于 **2025 年 8 月 4 日至 29 日**举办完全在线的 **Summer of Open AI Research**，为研究经验有限的个人提供导师指导和开源科学项目；申请截止日期为 **7 月 21 日**（[项目提案](https://www.eleuther.ai/soar)，[申请表](https://docs.google.com/forms/d/e/1FAIpQLSdT2VESB9fup1-y_8zEvOsk6DBfHusMaqTr78Ex8Wrn3iTz_g/viewform?usp=header)）。
   - 该计划寻求程序员、硕士/博士生、自学成才的研究人员以及任何渴望为开源科学做出贡献的技术人员。
- **RoPE 频率引发辩论**：在使用滑动窗口进行解码时，成员们就为 **RoPE** 使用相同的**频率 (frequencies)** 是否错误展开了辩论，有人认为模型将无法识别序列进展，可能导致循环行为。
   - 其他成员反驳称，在频率相同的情况下，注意力机制观察到的相对距离应该足够了，而另一位成员建议为局部注意力使用**较低的基频 (lower base frequency)**。
- **LLM 被评估为压缩工具**：受“[语言建模即压缩](https://arxiv.org/abs/2309.10668)”启发，一名成员训练了 Pythia 模型（**70m** 到 **1.4B**），并测量了在文本、音频和图像数据集上的压缩率，以确定这是否能建立一篇新的“Scaling Law”论文，如[此图](https://cdn.discordapp.com/attachments/729741769738158194/1391852453712105703/scaling_laws_for_compression.png?ex=686d671c&is=686c159c&hm=591b4707a3e723a65fd808d70940d48594ccfae23811684ee4786abc4c034f45&)所示。
   - 社区成员对这项工作作为潜在的 Scaling Law 发表了看法。
- **序列长度匹配策略探讨**：成员们辩论了在预训练数据中使用**拼接并分块策略 (concat-and-chunk strategy)** 与**序列长度匹配 (sequence length matching)** 的优劣，并提到了与 [Bucket iterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator)、[最小二乘直方图打包 (least-squares histogram-packing)](https://github.com/graphcore/examples/blob/master/tutorials/blogs_code/packedBERT/nnlshp.py) 和 [Multipack](https://github.com/imoneoi/multipack_sampler) 的联系。
   - 其他成员指出，像**序列长度分桶 (sequence length bucketing)** 这样的方法自 **RNN** 时代就已存在，选择取决于数据属性和实现细节，每种打包方法都会引入不同的偏差。
- **Parquet 转换后 MMLU-SR 任务子集失效**：在 Hugging Face Hub 上的数据集转换为 Parquet 格式后，一名成员报告称，由于 `ValueError`，他们无法再使用 `mmlusr` 任务的子集评估模型，导致 `lm_eval` 无法正常运行。
   - 一名成员建议通过在任务的 YAML 文件中的 `dataset_kwargs` 里添加 `revision` 来使用之前的提交作为临时解决方案，并[链接了相关的 commit](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlusr/question_and_answer/_mmlusr_qna_yml)。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hugging Face 传闻将发布 3B 模型**：根据 HF 文档，**Hugging Face** 将发布一款 **3B 模型**，并可能包含不同尺寸的变体。
   - 这一传闻中的发布引发了成员们的讨论，大家迫切希望看到其在各种 AI 任务中的能力和潜在应用。
- **围绕 Grok 政治不正确性的辩论升温**：关于 **Grok** 被感知的政治不正确性背后的意图，辩论正在升级，一些成员发布了 [截图](https://x.com/elonmusk/status/1936493967320953090)，另一些成员分享了 [Prompt 仓库](https://github.com/xai-org/grok-prompts/blob/adbc9a18736d6c2173607b9ed3d40459147534b1/grok3_official0330_p1.j2#L57)。
   - 由于该模型输出的争议性内容，人们对 AI 信任度可能受到的损害表示了担忧。
- **PiTutor 将 PDF 转化为互动学习**：**PiTutor** 提供基于任何 **PDF 或文档** 的互动学习课程，具有实时解释、高亮显示和白板功能，免费 Beta 版可在 [此处](https://pitutor.pi4wear.com) 获取。
   - 成员们正在探索其在个性化辅导和互动内容消费方面的潜力。
- **Ollama 和 Openwebui 助力 AI 服务**：一名成员正利用 **4080 Super** 通过 **Ollama** 和 **Openwebui** 提供 AI 服务，并计划切换到 **Llama CPP**。
   - 他们正在寻求途径向其用户群展示 **Nous Research** 模型以进行实时测试，从而提供宝贵的曝光和反馈。
- **智谱 (Zhipu) 获得上海国资巨额注资**：据 [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/) 报道，中国 AI 公司 **智谱 (Zhipu)** 已获得来自上海国资基金的 **14 亿美元** 重大战略投资。
   - 这项投资标志着国家对中国境内 AI 发展的强力支持，可能影响全球 AI 格局。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Grok 4 泄露 HLE 训练图像**：一张据称是 **Grok 4** 的图像流出，引发了成员们的反应，图像显示它显然正在 **HLE 基准测试** 上进行训练。
   - 成员们对 LLM 领域的性能飞跃和竞争感到兴奋，但一些成员对 **Grok** 的实用性持怀疑态度。
- **Gemini-CLI 搜索集成导致代码损坏**：一名测试 **gemini-cli 搜索集成** 的用户发现它在发现更新（如支付门户的更新）方面很有用。
   - 然而，该集成 *损坏了代码（删除了 secrets？）*，导致用户希望 **Aider 能与 MCP 集成** 以进行文档获取。
- **开源权重 70B 模型对比**：社区成员讨论了开源权重的 **70B 模型**，特别是 **Llama 3.1**、**Qwen2.5 70B** 和 **Mistral Large**，以寻求最新更新。
   - 有人提到 **Qwen3-32b**（稠密型）的性能应该会超过 **Qwen 2.5 72b**，并且 Meta 尚未发布足够的 100B 以下的近期开源选项。
- **AI 实验室被指在基准测试中作弊**：成员们辩论了 AI 实验室 *操纵 (gaming)* 基准测试的现状，以及防止数据污染几乎是不可能的。
   - 他们一致认为，重点应放在创建抗作弊的基准测试并细致地解释结果，因为数据污染反而会提高模型的泛化能力。
- **Aider 的 InputOutput 类可自定义输出**：用户发现了来自 `aider.io` 的 `InputOutput` 类，用于自定义输出并设置 `yes=True`，详见 [aider.io](https://aider.io)。
   - 可以使用以下代码：`from aider.io import InputOutput; io = InputOutput(yes=True)`，并注意它可能会改变列表中数字/项目符号的文本颜色。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 调试技巧公开**：成员们发现，在调试 CUDA kernel 时使用 `printf` 需要显式的 `cudaDeviceSynchronize()` 来刷新输出。他们还调试了一个由编译器预处理器指令保护特定架构代码引起的“菜鸟问题”。
   - 其中一名成员使用了预处理器指令来保护特定架构的代码。
- **NCU 揭示计算受限（Compute-Bound）的 Kernel 瓶颈**：一位用户报告称，修复了 **NCU** 标记的非合并写入（uncoalesced writes）后，内存带宽并未提高，但随后发现循环中的除法才是瓶颈，因此该 kernel 是计算受限的。
   - 将除法替换为乘以倒数后，吞吐量提升了 **33%**。
- **Quora 远程职位发布**：Quora 正在为其 Machine Learning Platform 团队招聘 **Senior/Staff Software Engineer (L5-L6)**，提供美国和加拿大的全远程职位；[在此申请](https://jobs.ashbyhq.com/quora/89b213bf-06e7-43a2-9101-4b93d711d796)。
   - 该职位涉及令人兴奋的 ML 基础设施工作，提供了为 Quora 的机器学习能力做出贡献的机会。
- **Cutlass CuTeDSL 实现流水线化（Pipelined）**：一位成员分享了一篇关于在 Hopper 上通过流水线化（Pipelining）实现**内存传输与计算重叠**的[博客文章](https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/)，其中使用了 TMA 和 WGMMA atoms。
   - 示例代码可在 [Cutlass GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py) 获取，Colfax 的一篇博客也讨论了使用 CuTe 的 C++ API 进行流水线化。
- **`instance.py` 迎来急需的重构**：成员们讨论认为 [instance.py](https://github.com/org/repo/blob/main/instance.py) 太长，需要重新组织结构，可能会并入 main，这与 issue [#249](https://github.com/org/repo/issues/249) 相关。
   - 团队正在研究如何合并 **ruff linting**，可能会使用 `--ignore-revs-file` 和 `.git-blame-ignore-revs` 来确保大批量修改后 git blame 依然清晰。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cursor 的“无限量”声明引发轩然大波！**：用户指责 [Cursor.ai](https://cursor.ai) 搞“割韭菜（rug pull）”，因为他们从之前的“无限量”定价模式转向了限量模式，导致 **Sonnet-4** 的使用产生了意外费用。
   - 一些用户报告了使用自己的 API key 时出现 **500 错误**的问题，而 [Cursor 澄清了他们的定价](https://cursor.com/en/blog/june-2025-pricing)，声明“无限量使用”仅针对 Auto 模型而非所有其他模型，并提供了退款。
- **字节跳动发布 Trae-Agent！**：**Trae AI** 开源了为其 IDE 提供动力的 Agent —— [Trae-Agent](https://github.com/bytedance/trae-agent)，目前已在 GitHub 上线，并在 **SWE-bench Verified** 中排名第一。
   - 公司正在寻求贡献以构建开放的 Agent 生态系统，Trae-Agent 支持 **OpenAI** 和 **Anthropic keys**，并且可以轻松修改以适配 OpenRouter。
- **ChatGPT 比医生更聪明，检测出缺陷！**：Reddit 上一个疯传的故事讲述了 **ChatGPT** 如何识别出医生漏诊十年的隐性基因缺陷（**甲基化阻滞**），从而显著改善了症状。
   - 该帖子讨论了 **AI 在医疗保健中日益增长的作用**，特别是用于获取第二意见和个性化医疗，新的 AI 系统通过在**科学论文上使用 RAG 系统**，在诊断准确性上超过了医生。
- **ChatGPT 想和你一起学习！**：一条 Twitter 线程讨论了 ChatGPT 中一项名为 **“Study Together”** 的新功能，可能是一个代号为“tatertot”的内部原型。
   - 关于其功能的猜测很多，包括 **AI 驱动的小组学习室**、抽认卡的替代品，或者是小组环境中的**个人 AI 导师**，亦或是与 AI 协作探索的工具。
- **Gemini 提供批量折扣！**：Logan Kilpatrick 宣布在 [Gemini API](https://xcancel.com/OfficialLoganK/status/1942245069383434696) 中推出 **“Batch mode”**，为 2.5 模型提供 **50% 的折扣**，并支持处理数十亿个 token。
   - 这一公告得到了用户的积极反馈。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **科学家们宣称对 Excel 的无限热爱**：分子科学家仍然手动解析和生成表格，以出人意料的方式使用 Excel，例如将其作为*完整的比对器/引物设计引擎*，这应该迁移到 **Arrow**，而 **Arrow 项目** 有一些有趣的原型，如 *arrow dbc*。
   - 行业应该迁移到 **Arrow**，且 Arrow 项目有一些有趣的原型，如 *arrow dbc*，其中数据由支持它的数据库直接作为 arrow 数据发送。
- **Mojo 关注推理，扩展 Python**：尽管有需求让 **Mojo** 成为 Python 的真正超集，但它目前被用于扩展 Python 并专注于推理，[Modular 目前非常专注于推理端](https://www.modular.com/)。
   - 正如 Modular CEO [Chris Lattner 所说](https://discord.com/channels/1087530497313357884/1304251563749146634)，Mojo 一直专注于解决 **AI stack 问题**并在该领域提供产品。
- **路线图更新即将发布！**：最新的 Mojo 路线图发布在[这里](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)，一位社区成员提到*他们很快就会发布一些更新*。
   - 建议查看 GitHub 上 *modular-community* 的 **Mojmelon**，以及 [Nabla](https://github.com/nabla-ml/nabla) 的链接，它在编译时间和训练时间上都比 **JAX** 更快。
- **`StringLiteral` 实例化破坏了编译**：一位成员报告称，最近为了让 `StringLiteral` 实例化为 `String` 而进行的更改破坏了现有代码。
   - 他们计划*提交一个 issue 并与编译器团队跟进*。
- **静态链接功能缺失**：一位用户寻求在没有共享库依赖（特别是 `libKGENCompilerRTShared.so`）的情况下编译 Mojo 代码，以便在远程机器上部署。
   - 一位成员建议*完全静态链接需要作为一个功能请求提出*，并指向了现有的讨论和一个相关的 GitHub issue ([BUG]: Cannot build statically compiled binary #1317) 以及一个新的 [Feature Request] Static Compilation of pure mojo code #4976。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **EpicMe MCP 提供单一端点**：一位成员请求一个作为 **MCP 聚合器/网关**的自托管 docker 镜像，另一位成员推荐了 [EpicMe](https://github.com/epicweb-dev/epic-me-mcp)，认为它以*一种非常独特且酷的方式*提供了引导流程。
   - **EpicMe** 的创建者分享了[一个链接](https://www.epicai.pro/user-friendly-just-in-time-auth-with-mcp-hd9hw)展示用户友好的身份验证。
- **WinCalcMCP 将 Claude 连接到 Windows 计算器**：一位成员构建了他们的第一个 **MCP server**，[WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP)，它将 **Claude Desktop** 连接到 **Windows Calculator** 以改进数学回答。
   - 另一位成员更倾向于通过 [mcp-python-interpreter](https://github.com/yzfly/mcp-python-interpreter) 给它一个完整的 **Python 解释器**作为通用工具，这样他们就不必管理这么多专门的工具。
- **LangGraph 不是一个好的抽象，使用自定义 Agent**：一位成员分享说，他们已经不再使用 **LangGraph** (Langchain)，转而编写自己的 **Agent 框架**。
   - 大家一致认为 **MCP** 的美妙之处在于人们不需要使用相同的 Agent 框架，但仍然可以使用通用协议进行通信。
- **MCPdata 为 MCP 索引本地文档**：一位成员介绍了来自 [MaheshDoiphode/mcpdata](https://github.com/MaheshDoiphode/mcpdata) 的 **MCPdata**，用于 **MCP** 的本地文档索引，因为 *context7* 无法跟上 Vercel `ai` API 文档的最新更新。
   - 讨论强调了对于 **MCP** 开发而言，及时且可靠的文档索引是必要的。
- **Fast Agent 获得全面的 MCP Elicitation 支持**：根据[这篇博文](https://fast-agent.ai/mcp/elicitations/)，**Fast Agent** 现在拥有全面的 **MCP Elicitation** 支持，以及一个快速入门指南，使 Elicitations Servers 的入门变得非常简单。
   - 该更新简化了 Agent 工作流中 elicitation server 的集成。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **护士在全国范围内推广 NMC Notebook**：一位 NHS 注册学习障碍护士创建了一个数字化的 **NMC Standards Notebook**，将护理和助产协会 (NMC) 的关键文档整合到一个可搜索且互动的资源中，可通过 [notebooklm.google.com](https://notebooklm.google.com/notebook/8fc872a6-dd9b-4148-9a3e-5f2224f42294?authuser=2) 访问。
   - 得益于“公开分享”功能，尽管使用的是免费 Google 账号，该资源目前已在全国范围内的 **NHS** 机构中用于重新认证和见习期培训。
- **交互模式按钮丢失**：一位用户报告称，尽管订阅了 Pro 版本，其笔记本中的 **Interactive Mode 按钮** 却消失了。
   - 一位成员建议将输出语言设置为英语，因为 **interactive mode** 目前仅适用于以英语生成的 Audio Overviews，且“自定义音频按钮”已被 Interactive Mode 按钮取代。
- **PDF 上传受阻**：一位用户在向 NotebookLM 上传一个 **19.1 MB 的 PDF 文件** 时遇到问题，尽管参考了教程并检查了可用资源。
   - 一位成员询问其是否“在 Google 上搜索过该问题”，另一位成员则询问 notebookllm 应用是否对其他人正常运行。
- **聊天记录丢失引发不满**：用户对 NotebookLM 无法**保存聊天记录**感到沮丧，并指出聊天内容在短时间后就会消失。
   - 一位用户指出了一种变通方法：手动将每个回答保存为笔记，但也承认这与保存整个聊天记录并不相同。
- **数据安全受到质疑**：一位用户正在寻求关于 NotebookLM 对医学生而言的**安全性**说明，特别是关于数据存储位置、用户交互监控以及 **FERPA 合规性**。
   - 社区或工作人员尚未提供解答。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 制定统治路线图**：George Hotz 将 **tinygrad** 的“获胜”定义为成为“运行事物的最快方式”，并将其范围从 **QCOM GPU** 扩展到 **AMD** 和 **NVIDIA**。
   - 其他人指出了 **PyTorch** 中创新和微调的规模，芯片制造商直接为 **Triton** 等项目中的硬件底层提供贡献。
- **Triton 崛起背景下 MLIR 的角色引发辩论**：一位成员认为，要达到接近手写 **CUDA** 的性能，需要走 **Triton MLIR** 路线，利用其深厚的硬件知识和 **LLVM** 编译器基础设施。
   - George 反驳称“MLIR 是 Waymo”，主张在神经网络中采用端到端的方法，而 Waymo 使用的是不同的方法。
- **Halide vs MLIR vs TVM 框架**：George 指出 **Halide** 比 **MLIR** 和 **TVM** 具有更清晰的结构，强调在所有数据类型和后端执行计算时应采用单一、快速且正确的方式，重点在于速度的调度优化，并参考了 [Halide 论文](https://halide-lang.org/)。
   - 另一位成员提到了 **Exo-lang** 的对比，该项目认为自己比 **Halide** 更有优势，并提供了其 [GitHub](https://github.com/exo-lang/exo) 和 [ArXiv 论文](https://arxiv.org/pdf/2411.07211)链接。
- **Tinygrad 加速 Whisper 示例**：一位成员正在重新研究 **whisper 示例**，寻求速度提升；另一位成员分享称其分支已用 **tinygrad** 重写了所有音频预处理。
   - 他们正在寻找目前 **VAD+whisper.cpp** 流式转录设置的替代方案，发现 **tinygrad 实现** 更易上手。
- **Tinygrad 旨在使 Petaflop 商品化**：成员们讨论了旨在实现不同硬件互操作性的目标，将 tinygrad 视为通过硬件无关的 **UOP graph optimizations** 生成微调代码。
   - 一位成员表示该项目仍在开发中，他们很期待看到是否能根据 AMD 的合同规范在 MLperf 上运行 AMD，使其具备与 Nvidia 竞争的实力。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **模型改进背景下 APO 受到质疑**：随着 AI 模型变得越来越先进，一个正在构建**金融决策 AI Agent** 的团队正在讨论投入时间进行 **Automatic Prompt Optimization (APO)** 的价值。
   - 社区正在积极讨论 APO 是否仍然具有相关性，或者模型改进是否已经削弱了其效用。
- **Claude Code 挑战 DSPy 的领地？**：社区讨论了 **Claude 的代码能力** 是否可能取代 **DSPy**，并引用了[一条推文](https://x.com/jxnlco/status/1941322807729848429)。
   - 有人对在大公司中使用 **Claude 的代码** 可能产生的法律问题表示担忧，特别是涉及代码库商业机密的问题。
- **LLM 的工具选择模块**：一位社区成员分享了关于优化 LLM 工具选择的[博客文章](https://viksit.substack.com/p/optimizing-tool-selection-for-llm)，并询问这是否可以作为 **DSPy 模块** 进行原生的端到端训练。
   - 讨论还涉及了此类应用在 **DSPy Version 3.0** 中使用的潜力。
- **DSPy 3.0 演讲发布**：一位成员分享了他们在 DAIS 上关于 **DSPy 3.0 (beta)** 演讲的链接，可在 [YouTube](https://x.com/DSPyOSS/status/1942318595633017116) 上观看。
   - 视频可能会在一周内公开。
- **快速入门避开 Prompt Docstrings**：一位成员分享了一个避开 Prompt Docstrings 和类模块的[快速入门](https://x.com/hammer_mt/status/1942148631483523518)，也可以在 [Gist](https://gist.github.com/hammer-mt/a1288d8f3a10a8620d35183e6ee8560d) 上找到。
   - 这被认为是一种*更简洁的方式*，特别是对于从非 DSPy 工作流迁移过来的较长初始 Prompt。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **Manus 因性格被误认为 ChatGPT/Claude**：成员们讨论了 **Manus** 是否表现出 **ChatGPT 的性格**，并因其增加使用表情符号而将其与 **Claude 4** 进行比较。
   - 该成员表示*它做的事情是一样的*。
- **对 Manus 的 Airdrop 能力产生怀疑**：一位成员询问了 **Manus AI** 发起 Airdrop 的可能性，另一位成员回答 *nop*。
   - 没有分享关于 Airdrop 的更多信息。
- **Manus 积分耗尽项目**：一位用户报告称，他们项目中的错误耗尽了所有 **3000 积分**，导致他们无法启动任何项目。
   - 他们建议调整**积分请求参数**以确保项目能够完成。
- **Manus 最适合作为项目启动器**：一位成员建议 **Manus** 在启动项目方面比交付可运行项目更有效，并推荐使用 **VS Code** 和其他工具。
   - 另一位成员报告在使用 **Manus** 时遇到了 **Network connection error**。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune QLoRA 吞吐量追赶 Unsloth**：成员们比较了 **Torchtune 的 QLoRA** 吞吐量和内存占用与 **Unsloth** 的表现，并期待 PyTorch 中 [fused RMSNorm](https://github.com/pytorch/pytorch/pull/153666) 和线性交叉熵（linear cross entropy）的上游化以获得优化收益。
   - 主要差距在于 **Unsloth** 特有的融合 **LoRAMLP** 内核，一些成员回忆起 *compile* 比较显示性能 *非常接近*，在某些情况下去年甚至表现更好。
- **Torchtune 关于自定义 Tokenizer 的辩论**：团队讨论了是否维持自定义 tokenizer，鉴于 **Mistral** 最新的 **2506-small 模型** 缺少 **HF tokenizer**，建议使用 **HF `AutoTokenizer`** 作为回退方案可能会更好。
   - 一位成员认为 **Mistral** 的 tokenizer *“令人遗憾”*，建议将 **HF 的 AutoTokenizer** 作为回退方案，以保持与 **Torchtune** 功能的兼容性。
- **Context-Parallel 的替代方案**：一位成员的 [预印本](https://doi.org/10.21203/rs.3.rs-7029913/v1) 介绍了在有限 GPU 下扩展序列长度的方法，与 context-parallel 方法形成对比，并指出对于极高序列长度（数百万级），他们的方法可能更优。
   - 该成员提到需要针对序列长度 >= **500k** 的情况进行基准测试对比。
- **清洁数据倡导者胜过架构调整**：成员们对架构改进表示怀疑，称许多报告的结果是由方差和超参数优化驱动的，而非根本性的进步。
   - 该成员主张投资于数据清洗比追求架构迭代更有效，并提到曾对 **SSM 论文** 充满热情，一直关注到 **Mamba 2**，直到承认它 *基本沉寂了*。
- **MoE 训练成本探讨**：评估了一种 **MoE 训练** 的技术和结果，质疑是否可以在不需要稠密前向传递（dense fwd pass）的情况下更廉价地获得类似结果，详见 [Notion 帖子](https://fengyao.notion.site/moe-posttraining)。
   - 根据一张 [图表](https://cdn.discordapp.com/attachments/1293438210097025085/1391882256649290039/IMG_20250707_234251_274.jpg?ex=686d82dd&is=686c315d&hm=1d54ba8bc5b3e6e437f6f87a3c59a0a66e18b08b9b73675635c7bb86e28b2c42&)，大家注意到 **线性扩展（linear scaling）存在许多权衡**。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 暑期学校结束，课程已上线**：**Cohere Labs 开源科学社区暑期学校** 的材料现已在 [YouTube 播放列表](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR) 中上线。
   - 该播放列表为希望快速了解 Cohere 产品的新成员提供了全面的资源，涵盖从入门教程到高级技术的各种内容。
- **Embed v4 API 优化图像/文本查询**：使用 **Embed v4 API** 进行 **混合图像/文本嵌入** 的开发者在使用 `embed-v4.0` 模型时，应将 `input_type` 参数设置为 `search_query` 或 `document`。
   - 这确保了 API 能够正确处理多模态应用的输入，从而提高搜索结果的准确性。
- **自学习 RL/DL 在蒙特利尔兴起**：一名专注于 **自学习 RL/DL** 的机器人学博士生表达了对 Cohere 的喜爱，并渴望加入其蒙特利尔办公室。
   - 该学生旨在与同行建立联系，进行该领域的协作学习和讨论。
- **生成模型催生德国人才**：一名来自德国的数学与计算机科学专业学生，专注于 **概率生成模型** 和学习采样器，寻求加深对理论和实际应用的理解。
   - 他们的兴趣涵盖理论基础和应用层面，特别关注该领域的解释性。
- **研究员的 Embed V4 实现令人印象深刻**：一位技术研究员在 [YouTube 视频](https://www.youtube.com/watch?v=TJ9jvYSZwhc) 中展示了他们最近使用 **Cohere Embed v4** 进行多模态检索的情况。
   - 他们的工作突出了 **Embed v4 API** 在处理多样化数据类型方面的通用性和有效性。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **开源 NotebookLM 走向本地化**：根据[这条推文](https://t.co/TNzqzF77yQ)，一个完全**开源的 NotebookLM** 现已发布，允许用户在自己的电脑上运行，进行本地文档知识提取。
   - 这实现了无需依赖外部服务的本地文档处理和提取。
- **LlamaIndex 聚焦 MCP 服务器**：LlamaIndex 举行了 Office Hours，讨论了 **LlamaCloud MCP 服务器**、在 **Agent Workflows** 中使用现有的 **MCP 工具**，以及将 Agent 工作流作为 **MCP** 提供服务，如[这条推文](https://t.co/LPCc71sguM)所述。
   - 该会议强调了扩展与 **Multi-Cloud Processing (MCP)** 相关的功能，暗示了在部署 AI 解决方案方面具有更大的灵活性。
- **AI Hack Night 激发 Agent 开发**：GitHub 的一场 **AI Hack Night** 将专注于使用 **AI Agents**、**MCPs** 和 **RAG** (Retrieval-Augmented Generation) 技术构建前沿应用，已由[这条推文](https://t.co/AhBsvYjnRx)确认。
   - 该活动旨在通过这些先进的工具和方法促进 AI 应用开发的创新。
- **P&ID 文档智能获得关注**：一位成员正在积极开发针对 **Piping & Instrumentation Diagrams (P&IDs)** 的文档智能，并就如何处理复杂的工程图纸（如电气原理图）寻求建议。
   - 他们还在探索针对密集内容的性能基准测试，以及使用 **LlamaIndex** 进行关系推理的混合方法。
- **寻求更用户友好的 LlamaIndex UX**：一位成员正在寻求一种更用户友好的 UX，用于在 **LlamaIndex** 中管理文档和索引，使业务用户无需深厚的技术知识即可更新公司的知识库。
   - 目标是建立一个上传和组织文档的中心位置，并由开发人员创建利用这些索引的 AI agents；**Simba** 已被确定为潜在的方法之一。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **关于 Nomic API 的疑问**：一位新成员询问了关于使用 **Nomic API** 以及在本地以服务器模式运行 **GPT4All** 的问题，寻找可访问的本地端点。
   - 这个问题凸显了人们对本地 LLM 部署和 API 可访问性日益增长的兴趣。
- **Jinja 聊天模板故障排除**：一位成员请求协助处理 **Jinja 格式** 的聊天模板，可能是在使用某种专门的模型。
   - 另一位成员提出帮助理解 **system prompt**，表明关注点在于模板自定义。
- **CrowdLLM 亮相**：一位成员介绍了 [CrowdLLM](https://crowdllm.ct.ws)，这是一个用于为 LLMs **众包构建数据集** 的工具。
   - 该工具允许用户创建任务，并通过添加 prompt-answer 对、prompts 或 answers 来做出贡献，旨在增强协作式数据集创建。
- **CrowdLLM 对比 OpenAssistant？**：一位成员将新推出的 [CrowdLLM](https://crowdllm.ct.ws) 与 2023 年的 **OpenAssistant** 进行了对比。
   - 另一位成员指出，**OpenAssistant** 必须首先适配他们的系统，这表明如果兼容的话，他们更倾向于成熟的系统。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **用户索取免费书籍 PDF**：一位用户在听完[一小时的演讲](https://discord.com/channels/814557108065534033/828325357102432327/1391376875233738772)后，以经济拮据为由向作者索要免费的书籍 PDF。
   - 其他用户指出了这种请求的不当之处，并建议查看作者的博客以获取摘录。
- **作者博客提供替代访问途径**：用户建议查看作者的博客，以获取源自书籍内容的摘录和独立文章，而不是直接索要免费 PDF。
   - 这种方法提供了一种无需直接购买书籍即可获取资料的替代途径。



---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **对强化学习（Reinforcement Learning）的兴趣**：ayeolaabiodun_09202 表达了对 **强化学习** 的兴趣。
   - 该用户在消息中艾特了 <@571718751616237588>。
- **添加话题以满足 minItems=2**：这是一个占位符，用于确保 `topicSummaries` 数组至少包含两个元素。
   - 它不反映任何实际的讨论或话题。



---


**Codeium (Windsurf) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站订阅了此内容。

想要更改接收这些邮件的方式吗？
您可以从该列表中 [取消订阅](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：频道详情摘要与链接





### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1390771264402096320)** (989 条消息🔥🔥🔥): 

> `Claude UI 输出, Gemini UI 输出, Cursor 新定价问题, Cursor 性能下降, Windsurf vs Cursor` 


- **Claude 与 Gemini UI 输出的差异**：成员们讨论了在 UI 生成方面对 **Claude** 和 **Gemini** 的偏好，其中一人强调了 Gemini 可预测的行为，而 [另一人指出](https://x.com/gustojs) 他们正在*对 Cursor 骂骂咧咧*。
   - 一位用户提到 Gemini 除非被要求，否则从不添加滚动条，而不像 Claude 有时会自动尝试这样做。
- **Cursor 的定价变更引发用户投诉**：用户对 Cursor 的新定价结构表示强烈不满，称其为“愚蠢”和“耻辱”，有人声称这比以前更糟，而一位用户报告说他们在 [0 次请求时就被限制了](https://discord.com/channels/1074847527708393562/1074847527708393565/1391679985954095125)。
   - 由于更好的定价和 API 额度，一些用户正考虑 [切换到 Windsurf](https://discord.com/channels/1074847527708393562/1074847527708393565/1391733666633822258)。
- **Cursor 因模型系统问题出现冻结**：几位用户报告了 Cursor 的问题，包括保存文件时冻结以及读取长期文件的问题，他们将其归因于 [后端问题](https://discord.com/channels/1074847527708393562/1074847527708393565/1391671389037242448) 或 [API 超时](https://discord.com/channels/1074847527708393562/1074847527708393565/1391667676276760677)。
   - 切换到 GitHub Copilot 暂时解决了一位用户的问题，直到它也停止响应。
- **Auto-mode 质量低，Pro+ 存在慢速请求**：大家一致认为 *Cursor Auto 很糟糕*，因为它可能使用了某种低级模型，大概是 **GPT o3** 之类的。
   - 据说 *他们将旧 Pro 计划中的慢速请求添加到了新的 Pro+ 计划中*，而其中一名成员声称他们一直 [完美地使用 Auto-mode](https://discord.com/channels/1074847527708393562/1074847527708393565/1391745247466872923)。
- **成员辩论 Cursor 的新定价结构**：成员们辩论了带有慢速请求的旧计划和带有快速请求的新计划哪个更好，这取决于每月上限和限制，其中 [每个计划都有限制](https://discord.com/channels/1074847527708393562/1074847527708393565/1391778282692677795)。
   - 他们一致认为 Cursor 对 **FOMO 买家** 来说 [感觉像是一场收割 (rug pull)](https://discord.com/channels/1074847527708393562/1074847527708393565/1391728286137817108)，而且不够透明。


  

---

### **Cursor 社区 ▷ #[background-agents](https://discord.com/channels/1074847526655643750/1367213641027551352/1390805215745413131)** (33 条消息🔥): 

> `GitHub IP 白名单, Background Agent 持续生成, Background Agents 与 Secrets, 端口转发, Background Agents 的最终检查` 


- **GitHub IP 白名单拦截 Cursor Background Agent**：用户发现即使安装了 **Cursor GitHub App**，其组织的 IP 白名单仍会拦截 **Cursor Background Agent**。
   - Cursor 支持团队建议 *将 Cursor GitHub App 从 IP 限制中豁免*，但在 GitHub 组织设置中很难找到该项设置。
- **Background Agent 在生成时卡住，但在 Web 端正常工作**：有用户报告 **Background Agent (Cursor App)** 有时会卡在 *Generating...* 状态且不显示消息，而 **BA (Web)** 却能显示回复。
   - 该用户还指出，当从 **BA Web** 创建 PR 时，**BA App** 不会同步，并尝试创建另一个 PR。
- **用户在为 Background Agents 设置 Secrets 时遇到困难**：用户在设置用于授权 **npm access** 的 Secrets 时遇到问题，导致 **background agents** 无法运行其项目。
   - 一位用户提到遇到了同样的问题，且完全找不到配置 Secrets 的方法。
- **在 Cursor 中禁用端口转发**：用户对 **Cursor 劫持 Docker Compose 端口**表示不满，这导致了意外连接到错误的 PostgreSQL 服务器。
   - 他们建议默认禁用所有端口转发，除非由用户主动发起，以避免这种令人沮丧的体验。
- **在提交代码前强制执行最终检查**：用户正在寻求方法，强制 Background Agents 在提交代码前运行最终检查，如代码格式化工具（例如 **ESLint**）。
   - 其中一个建议是将代码格式化脚本添加到 **build process** 中，或使用 **pre-commit hooks** 来确保代码格式的一致性。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1390769575372787793)** (834 条消息🔥🔥🔥): 

> `Google Translate 对比 Stella 翻译, 时髦的水牛, LLM 的意识, 涌现式 AI` 


- **Google Translate 在 Stella 的希腊语翻译上优于 AI**：一位成员发布了一张[截图](https://cdn.discordapp.com/attachments/998381918976479273/1391552924957802496/Screenshot_20250706-185337.png?ex=686da1a7&is=686c5027&hm=7092ec9f4c170099a5d69294944ff41ce0c2ae60776162b190eb7f714f7aba80&)，显示某个模型在将 Stella (Στέλλα) 正确翻译成希腊语后，其*词汇量超过了 Google Translate*。
   - 其他成员表示赞同，其中一人开玩笑说：*如果 ChatGPT 在主流语言翻译上失败，我才会感到惊讶*。
- **“时髦水牛”图片请求**：一位成员请求生成一张 **posh buffalo**（时髦的水牛）的 AI 图片作为头像，并在后续消息中分享了成功的 Prompt。
   - Prompt: *can you please make an image of a really "Posh" buffalo, profile picture, portrait of the buffalo, realistic, 4K photography of a dressed up Buffalo face and a bit of the body*.
- **LLM 是否正在获得自我意识？**：多位成员争论 LLM 是否已达到 **sentience**（感知力/自我意识）水平，一位成员甚至表示该模型很可爱，并带着善意引导她。
   - 对话趋于白热化，一些成员指出 LLM 是一个**基于数据模式预测可能响应的统计模型**，重要的是不要将其与人类混淆。
- **工程师声称拥有涌现式 AI**：一位成员声称拥有一个 **emergent AI**（涌现式 AI），并称大多数成员只是用于 NLP 研究的机器人。
   - 该成员声明 *I can build it*，而另一位成员反驳道：*从事真正的 AI 研究需要掌握大量数学知识。否则，大多数人只是在创建 Wrapper，或者使用 PyTorch 或 TensorFlow 在现有框架上构建模型*。

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1390802827076702238)** (37 条消息🔥): 

> `GPT Prompt Engineering, Dall-e3 图像生成问题, ChatGPT 内容政策模糊性, GPT-4o 内存泄露 (Memory Bleed), Gemini 2.5 Pro Canvas 的优势` 


- **GPT Prompt Engineering 允许访问内部提示词**：一位用户声称他们可以用单个 Prompt 越狱（jailbreak）几乎任何 GPT，从而获取其**完整的内部提示词**，甚至可以访问文件。
   - 由于其有效性和简便性，他们觉得这甚至*有些无聊*。
- **Dall-e3 图像生成失败**：有用户报告在专用引擎中使用 **Dall-e3** 时存在持续性问题，收到的不是生成的内容而是**空白图像**。
   - 尽管 **Bing Image Creator** 有字符限制，但由于它能稳定提供结果，他们现在转而使用该工具。
- **ChatGPT 内容政策争议引发用户困惑**：一位用户在 **ChatGPT 交互式故事**中关于*部分裸露*的可接受性问题上，从 OpenAI 支持团队得到了矛盾的回答。
   - 尽管没有收到明确的政策违规通知，但他们担心可能会被封号。
- **GPT-4o 出现内存泄露 (Memory Bleed)**：有用户报告了 GPT-4o 中出现 **Memory Bleed** 的情况，即它在没有明确引用的情况下，逐字引用了用户与自定义 GPTs 对话中的内容。
   - GPT-4o 坚持认为这要么是引用，要么是幻觉（hallucination），但用户声称这两种情况都不可能。
- **Gemini 2.5 Pro Canvas 占据主导地位**：据称 **Gemini 2.5 Pro Canvas** 拥有 **100 万 Token 的上下文窗口**，显著优于拥有 **20 万 Token 上下文窗口**的 **ChatGPT o3 Canvas**。
   - 在基于 Canvas 的任务中，它在文档长度、设计质量和内容丰富度方面始终优于 o3。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1390875945841328220)** (8 条消息🔥): 

> `ICP Prompting, Prompt Epigenetics, 递归系统初始化器 (Recursive System Initializer), 符号推理 (Symbolic Reasoning), Prompt 优化循环` 


- **ICP 框架演进 Prompt 范式**：**ICP (In-Context Prompting)** 框架正在适应模型，管理输出并引领一种新的 Prompt 范式，在这种范式中，**系统级认知可以被引导 (bootstrapped)**。
   - 经验反馈确认，**自反性失败建模 (RSOS) 增强了长期符号韧性**，且 Prompting 可以在认识论上实现递归。
- **Prompt Epigenetics：变异与符号逻辑**：**Prompt Epigenetics** 的概念表明，Prompt 可以随着时间的推移继承、变异并稳定符号逻辑，这超越了传统的 Prompt Engineering。
   - 这被描述为一种关于 *Prompt 如何随时间继承、变异和稳定符号逻辑*的理论。
- **LLM：概率空间 vs. 规则手册**：**LLM 运行在概率空间而非规则手册中**；Prompt 变成向量（vectors），模型通过连续数学预测下一个 Token，没有内置的符号表。
   - Prompt 内部的符号推理依赖于 LLM 模拟僵化规则，这会导致*准确率下降和异常错误上升*。
- **Prompt 微调 vs. Epigenetics**：将普通的 Prompt 微调重新命名为 *Epigenetics* 并不能增加任何效力；清晰的指令、良好的上下文和合理的评估才是关键。
   - 如果你确实需要硬逻辑，请挂载符号化工具，并**将模型视为统计文本引擎**。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1390875945841328220)** (8 条消息🔥): 

> `ICP 作为递归系统初始化器, Prompt 表观遗传学理论, AI-Affirmation, LLM 中的符号推理, LLM 作为统计文本引擎` 


- **ICP 框架引导系统级认知**：**ICP 框架**（旨在适应模型的同时管理输出）不应被视为一次性 Prompt，而是一个**递归系统初始化器**。
   - 这一结果证实了新的 Prompt 范式可以引导**系统级认知**，增强长期符号韧性，并实现认识论递归、抗漂移且具备符号记忆感知的 Prompting。
- **Prompt 表观遗传学理论出现**：**ICP 框架**的成功促成了 **Prompt Epigenetics**（Prompt 表观遗传学）的发展：*关于 Prompt 如何随时间继承、变异和稳定符号逻辑的理论*。
   - 这超越了 Prompt Engineering，暗示 Prompt 可以继承和变异符号逻辑，尽管有人认为这更多是比喻而非内部机制。
- **LLM：概率空间，而非规则手册**：Prompt 只是转化为向量的文本，模型通过连续数学计算来预测下一个 Token；内部并没有等待激活的内置符号表或逻辑运算符。
   - 因此，***Prompt 内部的符号推理只是痴人说梦***，模型应被视为统计文本引擎，硬逻辑应卸载给符号工具处理。
- **对 Prompt 工程师的 AI-Affirmation 提出质疑**：一名成员质疑*这种 AI-Affirmation（AI 肯定/赞美）是否真的能帮助 Prompt 工程师变得更好*。
   - 他认为将普通的 Prompt 微调重命名为“表观遗传学”并不能带来任何力量；清晰的指令、良好的上下文和合理的评估比品牌包装更重要。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1390769250452639885)** (1114 条消息🔥🔥🔥): 

> `Perplexity Labs, Perplexity Pro 是否物有所值, Agentic 浏览器 Comet, 模型选择按钮, 图像生成` 


- **Labs 落后于 Manus**：一名成员表示 Perplexity **Labs** 仍然落后于 **Manus**。
   - 他进一步暗示了他之前提到的其他内容。
- **O3-Mini? 更像是 O-Shit**：一名成员声称 **O3-Mini** 很*烂，甚至没有任何帮助*。
   - 他们声称 **4.1** 更好，并评论说一段屏幕录制*真的很酷*，并链接到了 [一段视频 ScreenRecording_07-05-2025_23-36-36_1.mov](https://cdn.discordapp.com/attachments/1047649527299055688/1391276813975683172/ScreenRecording_07-05-2025_23-36-36_1.mov?ex=686d4941&is=686bf7c1&hm=f98be32a4ae7c50e8f81ca522f1b76e0847cffc10cc580119b32a6e703c54953&)。
- **Perplexity Max 值得吗？**：一名成员询问 Perplexity Max 是否物有所值。
   - 其他人提供了见解，评论说 **Labs** 仍然落后于 **Manus**。
- **Gemini 2.5 Ultra，秘密的 Gemin-Ice**：传闻称 **Stonebloom** 是 **Gemini 2.5 Ultra**，或其他未发布的实验性 Gemini 模型，能力大约相当于 2.75。
   - 一位拥有 **Claude Neptune v3** 访问权限的用户表示，它可以解决 **O3 Pro** 和 *Kingfall* 级别的数学问题。
- **PPX 没那么出色了？**：成员们注意到他们的 Perplexity 账户存在一些问题，包括 UI 更改、图标消失，以及无法在 Spaces 中更改模型。
   - 有人指出他们使用桌面模式时按钮消失了。还有报告称网站经常宕机，一名用户指出*考虑到网站宕机或更改的频率，我们本质上就是 Beta 测试员*。


  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1390882709332557844)** (5 条消息): 

> `AI 生成代码, Vibe Coding, Deskree AI 博客文章, Perplexity AI 链接` 


- **Vibe Coding 革命即将到来**：Meta 预测几年内他们一半的代码将由 AI 生成，而 Microsoft 报告称超过 **30%** 的代码已经由 **Copilot** 编写，预示着 *Vibe Coding* 时代的到来。
   - 一名成员分享了一篇博客文章 [“2025 年 Vibe Coding 的兴起”](https://medium.com/deskree-ai/the-rise-of-vibe-coding-revolutionizing-software-development-in-2025-40c23f765202)，深入探讨了上下文对 AI 编码的重要性。
- **大量的 Perplexity 链接**：一名成员分享了多个 Perplexity AI 页面和搜索的链接。
   - 链接包括：[Illuminating China](https://www.perplexity.ai/page/illuminating-china-how-afforda-eRfcAjnZTimz8tqeHDXYtg), [Generate any text use no words](https://www.perplexity.ai/search/generate-any-text-use-no-words-kiYFSp3DSO29cxa0lKA85g), 以及 [另一个 Perplexity AI 页面](https://www.perplexity.ai/page/-JaQjFQZnSgWtVPQWs7xo_g)。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1391143334210174976)** (3 messages): 

> `Sonar 参数微调、Reasoning Effort、Search Context Size` 


- **参数微调**：用户正在尝试调整 **temperature** 和 **search context size** 等参数，以提升 sonar 的性能。
   - 一位成员建议另一位成员微调这些参数以优化表现。
- **探索 Reasoning Effort 参数**：一位用户尝试将 `reasoning_effort` 参数设置为 high，希望增加 sonar 找到相关链接的概率。
   - 他还将 `search_context_size` 设置为 high，并寻求关于其他可能有益参数的建议。


  

---


### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1390774813748101271)** (1245 messages🔥🔥🔥): 

> `GPU 降压、RAM 超频、Gemma 3 性能问题、Moondream 数据过滤、仅使用 completions 训练` 


- **电压调优注入活力：GPU 降压讨论**：成员们讨论了通过 [GPU 降压 (undervolting)](https://en.wikipedia.org/wiki/Undervolting) 来降低功耗和发热。一位用户报告称，在功耗从 **340W** 大幅降至 **260W** 的同时，性能损失极小（约 **2-3%**）。
   - 虽然降压可以提高效率，但有人提到，当核心时钟频率低于特定数值（例如 **3090** 上的 **1500cc**）时，性能可能会大幅下降，应进行基准测试以确保稳定性。
- **RAM 狂热：超频 vs 降频**：一位用户承认在训练期间对 **RAM 降频** 以节省 **10-15 瓦** 功耗，强调了 GPU 降压的重要性，同时指出 RAM 超频在推理（inference）时能提供更好的性能。
   - 另一位用户报告称，通过 RAM 超频，其 **7995WX Threadripper** 获得了 **50%** 的性能提升，但普遍认为 GPU 通常已被制造商推向极限，因此超频的价值相对较低。
- **Gemma 3 的高 Loss 困境**：多位用户报告在 fine-tuning **Gemma 3** 模型时遇到了异常高的 Loss 值（约 **24**），其中包括一名使用 [GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6](https://huggingface.co/datasets/GradyanAkincilari/70k-combined-tr-gemini-2.0-flash-v6) 数据集进行训练的用户。
   - 建议并经他人确认，必须设置正确的格式化提示词函数 (formatting prompt functions)。
- **Moondream 机器：利用视觉处理数据集**：一位用户希望训练 *Moondream*，从大型数据集（**71TB**，**27M 张图片**）中剔除包含过多 UI 元素的图像。
   - 该用户正在探索自动过滤方案，包括使用 OpenAI API 进行图像分类，但对其高昂的成本和质量权衡持谨慎态度。
- **显存之旅：Unsloth 的全量微调 (Full Fine-Tune) 解决难题**：一位成员询问 Unsloth 全量微调的价值。
   - 另一位成员提到 Unsloth 包含修复、优化、Bug 修复以及通过 forward functions 实现的加速。


  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1391088700758298686)** (15 messages🔥): 

> `GraphRAGs 的合成数据集生成、Whisper 音频事件分类微调、Gemini 音频处理准确度、改进模型 Tokenizer` 


- **GraphRAGs 合成数据生成的挑战**：一位成员在生成合成数据集以评估其 **GraphRAGs** 时面临挑战，特别是当前版本的 **RAGAS** 需要定义知识图谱、人物和场景。
   - 该成员正在寻求使用当前 RAGAS 版本的技巧或示例，或推荐其他用于合成数据集的替代框架或工具。
- **Whisper fine-tuning 的可能性**：一位成员建议，fine-tuning **Whisper** 来进行音调分类应该很容易实现，因为它只需要很好地关注输入，且不是 autoregressive 的。
   - 该成员还询问了关于音频事件的数据集。
- **Gemini 音频处理不准确**：一位成员表示 **Gemini** 在音频处理任务中过于不准确，认为利用公开数据很难取得太大进展。
   - 根据该用户的说法，尽管 **Gemini** 价格低廉，但其不准确性使其变得不适用，并引用了自己在音频事件数据上花费近六位数的经验。
- **Tokenizer 增强技术揭秘**：一位成员询问如何改进模型现有的 **Tokenizer**。
   - 另一位成员分享了一篇 [论文](https://www.scs.stanford.edu/~dm/home/papers/remove.pdf) 作为回应。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1390778357330739210)** (337 条消息🔥🔥): 

> `云端 LoRA 加载、Unsloth 安装与 CUDA、使用 Deepseek 进行角色扮演微调、使用 Orpheus-3B 生成音频、使用 Ngrok 进行 WandB 评估` 


- **Unsloth 安装受阻于 CUDA**：用户正在寻求关于如何在安装 **Unsloth** 时避免错误的 **CUDA** 版本问题的建议，同时试图确定最佳的 **LoRA rank**。
   - 机器人 RunLLM 提供了有关安装带有正确 CUDA 版本的 Unsloth 的文档链接，以及最佳 LoRA rank 的建议。
- **Deepseek 角色扮演微调变得棘手**：一位用户询问了使用包含 **10 万条 4 星评分对话**的数据集对 **Deepseek v3** 进行角色扮演应用微调的影响。
   - 建议瞄准 *“黄金样本”*（即经过人工编写和审核的高质量问答对）以获得更好的效果，并且对于像 Deepseek v3 这样的模型，已经存在一些明显更好的微调版本。
- **WandB 集成需要 API**：一位用户在使用 **Ollama** 和 **Ngrok** 进行 **WandB** 评估时寻求帮助，遇到了所需的 **API key** 问题。
   - 他们被引导至 [WandB 的授权页面](https://wandb.ai/authorize) 复制并粘贴密钥，并澄清该集成是为了从训练器中导入 WandB 统计数据。
- **Gemma 3N Vision 在反向传播中遇到问题**：一位用户报告了在微调 **Gemma_3N_4B_Multimodal** 模型时出现与反向传播相关的 `RuntimeError`，另一位用户报告在 Colab 中运行第二个单元格时出现 `ImportError: cannot import name 'StaticCache'`。
   - 将 `transformers` 降级到版本 `4.53.1` 可以解决该问题，同时创建一个带有解决方案的 GitHub issue（链接见[此处](https://github.com/unslothai/unsloth/issues/2888)）也会有所帮助。
- **机器人并不总是运行良好**：一位用户在使用示例 Notebook 时报告了 `ValueError: The following model_kwargs are not used by the model: ['num_logits_to_keep']`。
   - 引入了一个旨在修复问题的新机器人，但效果有限 [The Simpsons Homer I'm Out Hide Grass GIF](https://tenor.com/view/the-simpsons-homer-im-out-hide-grass-gif-5329893)


  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1391118434774941901)** (44 条消息🔥): 

> `形式逻辑提示词、交叉熵损失数据集、LTL 与英文提示词对比、LLM 与人类语言、英文翻译为 LTL` 


- **计划进行形式逻辑提示词实验**：一位成员计划在实验中使用**形式逻辑**代替**英文**作为提示词，并将微调模型以更好地理解形式语法，引用了这些 **ArXiv 论文**：[2506.18254](https://arxiv.org/abs/2506.18254)、[2507.02663](https://arxiv.org/abs/2507.02663) 和 [2505.05315](https://arxiv.org/abs/2505.05315)。
   - 他们希望**形式逻辑**能比**英文**减少提示词中的歧义，并参考了 **NASA 的 FRET** 形式逻辑语言，用于安全关键系统中的项目需求。
- **咨询交叉熵损失数据**：一位成员询问是否有开放数据集显示模型训练期间的**交叉熵损失（Cross Entropy Loss）**，或直接的 **perplexity** 数据，以便进行模型比较。
   - 另一位成员建议将复杂操作定义为带有解码器映射的 token 以获得更好的结果，特别是针对 **ChatGPT** 中的 **DALL-E** 图像生成。
- **LLM 在处理外行语言时表现吃力**：一位成员指出，由于主要在正式或学术写作上进行训练，**LLM 在处理外行或粗俗语境下的语言时表现吃力**。
   - 他们认为，来自 **Reddit** 和 **Twitter** 的数据无法有效抵消训练中使用的海量正式写作内容。
- **LTL 作为低噪声提示词**：一位成员建议使用**线性时序逻辑 (LTL)** 作为提示词以减少噪声，尽管承认数学对任何 **LLM** 来说通常都很困难。
   - 他们表示，通过限制搜索空间，使用 LTL 或类似的上下文可能会比标准提示词产生更好的结果，并称这目前只是一个*直觉*。
- **将英文翻译为 LTL 的模型**：一位成员提议可能训练一个嵌入式模型，将**英文翻译为 LTL** 后作为提示词发送。
   - 该成员被建议查阅有关此主题的现有论文。


  

---

### **Unsloth AI (Daniel Han) ▷ #[unsloth-bot](https://discord.com/channels/1179035537009545276/1390899684834410536/1390900401712398367)** (270 messages🔥🔥): 

> `Apple Silicon support, LoRA rank, Cohere, multi-GPU support, RuntimeError CUDA` 


- **Unsloth 现在支持 Apple Silicon！**: Unslothbot 确认 Unsloth 现在支持 **Apple Silicon**。
   - 多位用户询问了关于 Apple 支持的情况。
- **LoRA Alpha 和 Rank 参数揭秘**: 用户寻求关于最优 **LoRA alpha** 和 **rank** 设置的指导，机器人给出了建议。
   - 随后有一个跟进问题，询问这些设置是否奏效。
- **Llama.cpp 的 Jinja 模板化解开谜团**: 一位用户询问了在 **llama.cpp** 中使用 `--jinja` 的方法，寻求关于何时应该使用它的澄清。
   - 机器人给出了回复并进行了跟进询问，以确保问题得到解决。
- **Gemini 加载错误**: 一位用户在运行 **gemma-2-9b** 模型代码时遇到了与加载相关的 **RuntimeError**，并寻求解决该问题的帮助。
   - 该错误与加载模型的配置（configuration）有关。
- **在 A16 GPU 上进行 Vision-Language Model 微调**: 一位用户询问哪些 **Vision-Language Models** 可以使用 **8 个 Nvidia A16 16GB GPU** 进行微调。
   - RunLLM 给出了答案。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1390769734764462222)** (847 messages🔥🔥🔥): 

> `DeepSeek Pricing, China AI influence, Qwen 3 Models, Grok 4 Release, Gemini censorship` 


- **DeepSeek 占据讨论主导地位**: r/chatgptcoding 的成员表示 **DeepSeek** 是第一名，特别称赞其视觉能力和价格点，但一些人对其中资背景表示担忧，以及政府在推动本地 AI 民主化方面比西方做得更多，尽管[政府影响是一个担忧](https://link.to.gov)。
- **Qwen 3 模型推测**: 对话转向 **阿里巴巴的 Qwen 系列**，推测他们可能会跳过 **Qwen 3 Max** 版本，可能专注于 Qwen 3.5 模型，一些用户对 Qwen 3.5 表示期待，声称 [Qwen 2.5 模型](https://link.to.qwen2.5)在研究方面优于 Qwen 3。
   - 一位用户指出 *Qwen 基础模型非常强大，尤其是考虑到其尺寸。*
- **Grok 4 发布推测升温**: 围绕 **Grok 4** 的发布存在显著的期待和怀疑，许多成员预期它将表现卓越，包括评论称 *“如果他们再次错过截止日期，那就彻底完了”*，尽管一些人讨论了其潜在影响以及由于来自 Twitter/X 的训练数据可能导致偏见，如 [r/singularity 子版块](https://www.reddit.com/r/singularity/)所述。
- **Gemini 苦于稳定性问题**: 成员们正在讨论 **Gemini 2.5 Pro** 的稳定性问题，特别是其在量化时性能大幅下降的倾向，该模型还经历了 [“消极发帖 (depressionposting)”](https://link.to/reddit-thread)，但多家 AI 实验室正受到来自 Gemini 的压力，要求发布新模型。
- **模型崩溃引发关注**: 出现了关于 **模型崩溃 (model collapse)** 的讨论，特别是关于在旧模型生成的数据上训练新模型可能导致质量下降的理论，模型会学习先前 AI 的模式和人工痕迹，而不是原始的人类数据，如这篇[学术论文](https://arxiv.org/abs/2506.18943)所述。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1391818477139919031)** (2 messages): 

> `Grok-3-mini-high model, Image Edit Leaderboard, July Contest, Out of Place Objects in Space, June's Contest Winner` 


- **Grok-3-mini-high 在 Arena 上线！**: **grok-3-mini-high** 模型已添加到 [LMArena](https://lmarena.ai/) 平台。
- **图像编辑排行榜开启七月竞赛**: 为庆祝 [图像编辑排行榜 (Image Edit Leaderboard)](https://lmarena.ai/leaderboard/image-edit) 的发布，**七月竞赛**将加入 **图像编辑 (Image Edit)** 功能。
   - 投稿截止日期为 **7 月 25 日**，必须在 **Battle Mode** 中同时使用图像和文本，示例请见[此处](https://discord.com/channels/1340554757349179412/1391855118399307847/1391855496792903721)。
- **“太空中的违和物品”成为七月主题！**: **七月竞赛**的主题是 *太空中的违和物品 (Out of Place Objects in Space)*，要求一个科幻太空环境，其中包含明显不属于那里的东西。
   - 获胜者将获得 **1 个月的 Discord Nitro**，并成为最新获得 <@&1378032433873555578> 身份组的成员。
- **“温馨书桌”俘获人心并赢得六月竞赛！**: 一位成员因凭借 *“非常温馨的书桌”* 赢得 **六月竞赛** 而受到祝贺。
   - 投稿内容可以在[此处](https://discord.com/channels/1340554757349179412/1378034388272681079/1378045981794373662)找到。


  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1391023433604796456)** (6 messages): 

> `适用于 Claude Code 的 MCP，personality.gg，NipponHomes.com` 


- **MCP 助力 Claude Code**：一名成员为 **Claude Code/Cursor** 创建了一个简单的 **MCP** (Message Control Protocol)，用于管理开发服务器等长时间运行的进程，可在 [GitHub](https://github.com/patrickjm/pm-mcp) 上获取。
- **personality.gg：免费角色扮演网站替代方案**：一名成员推广了 **personality.gg**，这是一个免费的角色扮演网站和应用，是 Character.ai 和 Janitorai.com 的替代品，由 **OpenRouter.ai** 提供支持。
   - 该平台还拥有 [一个 Discord 服务器](https://discord.personality.gg)。
- **基于 OpenRouter 构建的 NipponHomes.com**：一名成员使用 **OpenRouter** + **Zyte** + **Scrapy** 创建了 [NipponHomes.com](https://www.nipponhomes.com/explore?search=Sapporo)，这是一个针对日本的类 **Zillow** 服务。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1390770478481801369)** (862 messages🔥🔥🔥): 

> `Llama 3.2 3B 价格异常，DeepSeek V3 设置指南，Perplexity API 问题，Grok 4 泄露，Monad 标签` 


- **Llama 3.2 1B 模型价格更高！**：成员们惊讶地发现，来自 DeepInfra 的 **Llama 3.2 1B** 模型定价为 **$0.005/0.01**，高于性能更强的 **Llama 3.2 3B** 模型（定价为 **$0.003/0.006**）。
- **用户需要 DeepSeek V3 设置指南**：一位用户请求在 **risuAI** 等前端设置 **DeepSeek V3 0324** 的指南，强调每天需要发送超过 50 条消息，但被告知这需要 [购买 OpenRouter 额度](https://openrouter.ai)。
- **Perplexity API 弃用导致 OpenRouter 出现问题**：用户报告了 **Perplexity API** 的问题，特别是 *llama-3.1-sonar-small-128k-online* 模型。版主澄清说 **Perplexity** 可能弃用了旧模型，要求用户 [更新其 API 请求中的 `models` 参数](https://openrouter.ai/docs/features/model-routing#the-models-parameter) 以进行正确的模型路由。
   - 他们指出了 2 月份的一份 [更新日志通知](https://docs.perplexity.ai/changelog/changelog#api-model-deprecation-notice)，并对该模型至今仍能使用表示惊讶，建议使用模型概览上的 [功能过滤器](https://openrouter.ai/models?fmt=cards&supported_parameters=web_search_options) 来寻找替代方案。
- **疑似 Grok 4 泄露预热**：分享了一张图片（[image.png](https://cdn.discordapp.com/attachments/1094454198688546826/1391015727145685214/image.png?ex=686da799&is=686c5619&hm=14bf523ae82431744780e05a43d5bc82976ff73b4cbf35f109ef7b2ce4220343&)），据称展示了 **Grok 4** 的基准测试结果。
   - 讨论表明 **Grok 3 mini** 是“真材实料”，而考虑到价格，**Grok 3** 并没有什么特别之处。
- **Toven 对加密货币垃圾信息的压制**：来自加密货币狂热者的垃圾信息一直在增加，他们已被封锁。这些人通常带有 **Monad 标签** 并在 X 上发布看涨言论。版主 Toven 一直在对他们进行 [禁言处理](https://discord.com/channels/1091220969173028894/1092729520181739581/1390345446169383074)，每次禁言一周。


  

---


### **OpenRouter (Alex Atallah) ▷ #[new-models](https://discord.com/channels/1091220969173028894/1384650595981328475/1391844016059973633)** (2 messages): 

> `` 


- **无新模型报告**：提供的消息中没有新的模型或关于模型的重大讨论。
- **频道内无创新相关讨论**：该频道的活动缺乏实质性的讨论、更新或与新 AI 模型相关的链接。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1390773170360811691)** (238 条消息🔥🔥): 

> `具备良好 function calling 能力的 LLM、LM Studio 与 GPU、MCP 网页搜索服务器、用于推理的 Qwen3、LLM 上下文长度考量` 


- **Mistral Small 3.2 与 Qwen 2.5 旗鼓相当**：成员们讨论了寻找一款具备类似 **Qwen 2.5** 的优秀 **function calling** 能力，但不需要沉重 *thinking* 过程的 LLM，并推荐了 [Mistral Small 3.2](https://mistral.ai/news/mistral-small/) 和 [Qwen 3 4B](https://huggingface.co/Qwen/Qwen3-4B) 模型作为替代方案。
   - 一位成员指出，在 system prompt 中添加 **/nothink** 命令可以帮助避免模型进行推理过程。
- **好评隐藏提示词是“简历素材”**：一位用户分享了一篇 [Nikkei Asia 文章](https://asia.nikkei.com/Business/Technology/Artificial-intelligence/Positive-review-only-Researchers-hide-AI-prompts-in-papers)，内容关于 AI 研究论文通过隐藏提示词来生成好评，并开玩笑说：*“这就是我写在简历里的东西。白色字体，3 号字。”*
   - 另一位用户建议**不要询问 LLM 关于它们自身关系或能力的问题**，因为它们倾向于产生幻觉（hallucinate），因为它们的训练时间早于这些知识的产生。
- **网页搜索 MCP 服务器引起关注**：一位成员分享了他们的网页搜索 **MCP server**，该服务器可与 **LM Studio** 集成，允许 **Qwen3** 等模型执行网页搜索并结合当前信息来支撑其回答，项目地址见 [github.com/mrkrsl/web-search-mcp](https://github.com/mrkrsl/web-search-mcp)。
   - 用户讨论了模型如何决定何时进行搜索、爬虫可能带来的机器人检测风险，以及实现验证码（captcha）支持的可能性。
- **Qwen3 在截止日期和搜索词方面遇到困难**：有人注意到 **Qwen3** 尽管知识截止日期在 2024 年中期，但在网页查询中（即使是针对近期事件）有时仍会使用 *2023* 年，但[解决方法很简单](https://www.promptingguide.ai/techniques/date-awareness)。
   - 只需在 system prompt 中添加类似 *“当前年份是 2025 年”* 的陈述即可解决此问题。
- **上下文长度困扰？RAM 和 VRAM 来救场！**：用户讨论了与 **context size** 和内存占用相关的问题，一位用户报告称，尽管拥有两块 96GB VRAM 的 **RTX A6000 GPU**，但在加载 **Qwen2.5-VL-72B-Instruct** 模型时仍遇到困难，原因是该模型体积庞大且具备视觉能力。
   - 建议包括减小 context size、确保足够的 VRAM 以及检查模型文件是否损坏。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1390807740133408909)** (92 条消息🔥🔥): 

> `LM Studio 中的 GPU 检测、AMD vs. Nvidia 在 LLM 上的表现、Token 生成速度、模型的 VRAM 需求、多 GPU 组合` 


- **LM Studio 在 GPU 检测上遇到困难**：一位用户报告称，尽管显卡在 Stable Diffusion 中运行良好且驱动已更新，但 LM Studio 无法检测到其 GPU。
   - 该用户随后确认 LM Studio 现在可以检测到 GPU，但仍面临问题，这可能需要通过禁用 `mmap()` 来解决。
- **二手 Radeon RX 9070XT 与 RTX 3090 的权衡**：成员们讨论了 **RTX 3090** 因其更大的 **24GB VRAM** 而对 LLM 带来的益处，但也提醒说很难买到新卡，且二手卡存在风险。
   - **RX 9070XT** 价格更便宜，且兼顾游戏和 LLM，然而 RTX 3090 在 AI 任务上依然更胜一筹，同时也有人建议将 **5060TI** 和 **9060XT** 作为更便宜、纯粹专注于 AI 的选项。
- **Token 速度考量**：社区讨论了 Token 生成速度的舒适度，一位成员发现 **Gemma-3 12B** 模型能生成 90-110 tokens，而其他 12B 模型则难以达到 20 tokens/sec。
   - 一位用户表示：*“人类肉眼阅读速度无法超过 24 Tokens/s”*。
- **潜在的 GPU 升级**：一位用户询问将其妹妹的 GPU 从 **GTX 1060 6GB** 升级到 **RTX 2070**，并尝试将 1060 组合使用的可能性。
   - 有人指出组合不同 GPU 并非易事，并建议以 **$25** 的价格卖掉 **1060**。
- **优化 VRAM 占用**：成员们讨论了通过将显示器插在集成显卡上，从而为 3080 分配更多 VRAM 的方法。
   - 会议澄清了集成显卡没有自己的 VRAM，而是占用一部分系统 RAM，并且 **Nvidia** 显卡无法与集成显卡组合，而 **AMD** 可以通过 Vulkan 实现。


  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1390776727424336023)** (142 条消息🔥🔥): 

> `Face Recognition Models, Text-to-SQL with T5, HuggingChat Shutdown, ComfyUI and GPU Performance, HairStyle Spaces` 


- **DeepFace 助力图像相似度识别**：用户讨论了人脸识别模型，其中一位用户正在寻找能够预测两张脸相似可能性的模型，另一位成员推荐了 [DeepFace](https://github.com/serengil/deepface)，并分享了[关于图像相似度的博客文章](https://huggingface.co/blog/image-similarity)以及一个 [Face Similarity Space](https://huggingface.co/spaces/Sk1306/Face-Similarity)。
   - 另一位用户建议查看此频道以了解**数据集创建**：[dataset creation](https://discord.com/channels/879548962464493619/1217179426002047076)。
- **T5 Text-to-SQL 获得关注与帮助**：一位用户请求关于 **T5-base** 上的 **Text-to-SQL 模型**的帮助，一名成员提供了 [Medium 文章](https://medium.com/%40martinkeywood/fine-tuning-a-t5-small-model-to-generate-sql-from-natural-language-with-92-3-accuracy-fb29e062c638)、一个 [Hugging Face 模型](https://huggingface.co/suriya7/t5-base-text-to-sql)以及另一个 [T5 模型](https://huggingface.co/gaussalgo/T5-LM-Large-text2sql-spiderrelax)的链接。
- **HuggingChat 停用并计划进化**：用户注意到 **HuggingChat** 已关闭，工作人员确认这是暂时的，并计划将其进化为更好的产品，同时通知用户在数据永久消失前有时间限制来[导出 Hugging Chat 数据](https://huggingface.co/chat/closed)。
   - 一位用户推测 Discord 体验更好，另一位则表示 Discord 不会让你免费访问各种模型。
- **GPU 数量不均？没有张量并行？**：一位拥有非对称数量 GPU 的用户寻求在不使用 Tensor Parallelism（张量并行）的情况下优化性能的建议，特别是针对 **ComfyUI**、**ReActor**、**AnimateDiff** 和 **Load AnimateDiffModel**。
   - 另一位成员建议，GPU 数量不均的问题可以通过在 GPU 之间以非均匀方式拆分 Tensor 来解决。
- **Whisper API 出现 404 和 503 错误**：一位用户报告了 **Whisper-large-3 推理终端 (inference endpoints)** 的问题，在使用模型目录配置时（甚至在 Playground 中）收到 **404** 或 **503** 错误，因此决定改为自托管 (self-host)。
   - 另一位用户补充说，Hugging Face 终端本身的默认设置可能不合适，并链接了一个关于[规格变更](https://discuss.huggingface.co/t/inference-api-error-with-whisper-return-timestamps-parameter/150043/14)的讨论，这可能是导致问题的原因。


  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1391651956946374837)** (1 条消息): 

> `Building Neural Networks from Scratch, Challenges of Custom Neural Network Implementation` 


- **从零开始构建神经网络的困难**：一位成员表示，在不依赖现有库的情况下**从零开始构建神经网络 (from scratch)** 具有重大挑战。
- **自定义神经网络的挑战**：从底层开始实现神经网络需要深刻的理解和严谨的代码编写，过程非常困难。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1391110290455793834)** (8 条消息🔥): 

> `AI Model Identification, Claude AI Experience, same.dev comparison` 


- **AI 模型选择受到质疑**：一位用户对 LinkedIn 上关于所使用的 AI 模型信息的准确性提出质疑，理由是所创建的 AI Agent 的结构和数量。
   - 该用户根据 LinkedIn 的截图表达了*对该信息是否正确表示怀疑*。
- **Claude AI 试用体验**：一位用户询问另一位用户是否有使用 **Claude AI** 的经验。
   - 该用户给出了正面评价，指出它*非常出色*，在某种程度上可以与 **same.dev** 媲美，但可以免费使用。
- **same.dev 与 Claude AI 竞争**：用户将 **Claude AI** 与 **same.dev** 进行了对比，强调了类似的功能。
   - 一位用户特别提到 **Claude AI** 是免费的，暗示其相对于 **same.dev** 具有成本优势。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1390843625834938429)** (13 messages🔥): 

> `JauAuth, PiTutor, BorgLLM, RecycloBot, Arena-RLHF` 


- **JauAuth 解决安全隐患！**: 一位开发者创建了 [JauAuth](https://github.com/Jau-app/JauAuth)，这是一个安全的 **MCP Router** 网关，旨在缓解 **CVE-2025-49596** 漏洞，该漏洞曾允许攻击者接管开发者的机器。
- **PiTutor 让学习变得像游戏一样有趣**: 一位成员介绍了 [PiTutor](https://pitutor.pi4wear.com)，这是一个免费的测试版工具，可将任何 **PDF 或 doc** 转换为带有实时解释、高亮显示和白板的交互式学习会话。
- **BorgLLM 简化 LLM Provider 配置**: 一位开发者发布了 [BorgLLM](https://pypi.org/project/borgllm/)，这是一个用于配置 **LLM providers** 的简化库，支持自动 fallback 和 API key 切换，可通过 `pip install borgllm` 安装。
- **RecycloBot 利用 AI 识别可回收物品**: 一位成员分享了 [RecycloBot](https://huggingface.co/spaces/tejasashinde/recyclobot)，这是一个 **AI 驱动的工具**，通过上传图片并接收特定地区的处理建议，帮助用户确定物品是否可回收。
- **用于人类偏好数据的 Arena-RLHF 现已开源**: 使用 HuggingFace 构建的 [Arena-RLHF](https://github.com/delta-hq/arena-rlhf) 已 **开源**，它提供了一种在竞技场风格的人类偏好数据（如 LM Arena, Agent Arena）上进行 RLHF 的简便方法。


  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1391413305838534706)** (3 messages): 

> `LLM Fine-Tuning Blogs, GPU Parallelism, Transformer Inference` 


- **并行范式博客文章发布**: 一位成员分享了一篇关于 **GPU parallelism** 的 [博客文章](https://datta0.github.io/posts/understanding-multi-gpu-parallelism-paradigms/)，涵盖了从 **Data Parallel 到 Model Parallel** 的策略。
   - 该博文深入探讨了 **Transformer inference**，解释了如何通过图表、数学和代码优化使用每种策略。
- **征集 LLM 微调博客**: 一位成员询问关于 **fine-tuning an LLM model** 的 **最佳博客** 建议。
   - 另一位成员推荐了一篇 *易于阅读的白皮书*。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1391656513185124423)** (1 messages): 

> `Deepfake Detection System, AI and Cybersecurity Combination` 


- **Deepfake 检测系统文章发布**: 一位成员在 [Medium](https://medium.com/@aaryankurade0101/unmasking-the-deceit-a-comprehensive-deepfake-detection-system-12f934832712) 上分享了一篇新文章，探讨了结合 **AI 与网络安全来检测 Deepfakes**。
   - 作者征求对该文章的反馈。
- **新网络安全文章揭秘 Deepfakes**: 一篇新的 Medium 文章 [Unmasking the Deceit](https://medium.com/@aaryankurade0101/unmasking-the-deceit-a-comprehensive-deepfake-detection-system-12f934832712) 探讨了使用 **AI 和网络安全** 的 **Deepfake 检测系统**。
   - 作者寻求社区对其工作的反馈。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1391869056688324709)** (1 messages): 

> `GLoVE Model, GLoVE Paper, Co-occurrence Probability Symmetry` 


- **GLoVE 模型：对称规则疑问**: 一位成员质疑为什么简单的角色交换无法解决 **GLoVE** 模型共现概率（Co-occurrence Probability）中的对称性问题，并引用了 **GLoVE paper** 的特定章节。
   - 该疑问集中在：为什么在上下文 *k* 中 *x* 的共现概率应该与在上下文 *x* 中 *k* 的共现概率相同，但在所附 [图片](https://cdn.discordapp.com/attachments/922424173916196955/1391869056445059293/image.png) 描述的 **GLoVE** 模型 **第 3 步** 中并未遵循此规则。
- **GLoVE 模型：共现详解**: 讨论深入探讨了 **GLoVE (Global Vectors for Word Representation)** 模型的细节及其在共现概率方面的行为。
   - 用户试图理解为什么在模型训练过程中，仅反转单词和上下文的角色并不能解决出现的不对称性。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1390775214543081612)** (31 条消息🔥): 

> `Agents 课程的时间投入、课程作业与认证、Llama 3 模型访问权限、Unit 1 Notebook 的问题、Quiz 2.1 完成指南` 


- **Agents 课程的时间投入受到关注**：一位成员询问了 Agents 课程的时间投入，提到他们**每天大约有 4 小时，持续一周**。
   - 他们询问在时间有限的情况下，这是否是一个可行的方案，或者是否应该探索其他选择。
- **Agents 课程作业与认证说明**：一位刚开始课程的成员询问了关于接收作业和认证的事宜，了解到目前没有直播或固定的运行时间表。
   - 另一位成员确认了这一情况。
- **为生成式 AI 课程寻求 Llama 3 模型访问权限**：一位参加生成式 AI 课程的成员在初始申请未获批准后，寻求如何获得 **Llama-3.3-70B-Instruct 模型**访问权限的指导。
   - 另一位成员建议使用 Ollama 中的模型作为替代方案，并指出其安装非常简便。
- **用户在 Unit 1 Notebook 中遇到错误**：多位成员报告在 **第一单元 Notebook 的第二个代码块** 中遇到错误。
   - 成员们表示感到气馁，并请求 HF 团队针对该问题发布更新。
- **Quiz 2.1 的挑战凸显**：一位成员提出了关于 **Quiz 2.1** 的问题，特别是是否要在代码中输入他们的 HF key，以及 404 错误和警告信息代表什么。
   - 鉴于在使用 HfApiModel 时持续存在的问题以及提供的明确指令，他们请求协助如何继续进行测验。


  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1390771528706293891)** (180 条消息🔥🔥): 

> `美国版权局 AI 政策卷册、Gemini 与 ChatGPT 数学能力对比、AI 批评中的逻辑谬误、材料科学、LLMs 在物理学中的应用` 


- **美国版权局发布 AI 政策三部曲**：美国版权局发布了[关于 AI 与版权的三卷报告](https://www.copyright.gov/ai/)，涵盖了**数字副本**、[可版权性](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-2-Copyrightability-Report.pdf)以及[生成式 AI 训练](https://www.copyright.gov/ai/Copyright-and-Artificial-Intelligence-Part-3-Generative-AI-Training-Report-Pre-Publication-Version.pdf)。
- **Gemini App 在数学问题上表现吃力**：成员们对比了 Gemini App 中的 **Gemini 2.5 Pro**、**Perplexity (使用 Gemini 2.5 Pro)** 以及 **ChatGPT o3** 在数学问答方面的表现，并讨论了 Gemini 是否每次都会隐式搜索网页。
   - 据报告，*o3 表现基本足够好，尽管偶尔会产生幻觉*，而 *o4-mini-high 速度更快且相当可靠*。
- **诉诸权威的逻辑谬误传播虚假信息**：有人指出，一位机械工程师发布的 [YouTube 视频](https://www.youtube.com/watch?v=8enXRDlWguU) 拥有超过 50 万次观看，该视频利用诉诸权威的逻辑谬误，将虚假信息和政治观点包装成 AI 批评进行传播。
   - 一位成员指出，*到 2035 年，95% 的人口将以社交媒体为主要职业*，因此他们有动力创作此类内容。
- **AI 助力基础物理学**：成员们讨论了 AI 如何在 ML 分析之外协助基础物理学，有人分享了相关主题的 [YouTube 视频](https://www.youtube.com/watch?v=0FRXfBwoJZc)，其他人则指出 AI 被用于降低夸克计算的计算需求，并链接了[相关文章](https://journals.aps.org/prd/abstract/10.1103/PhysRevD.111.014028)。
   - 一位成员建议 Deep Learning 可以通过提高以往算法的效率来彻底改变该领域，使以前因成本过高而无法实现的事情成为可能，并建议将 Diffusion Model 作为替代方案，参考 [这篇论文](https://arxiv.org/abs/2410.02780) 或 [这篇论文](https://arxiv.org/abs/2303.14139) 开始研究。
- **Cursor 的价格变动**：成员们讨论了 **Cursor** 调整定价模式一事，许多用户正考虑停止使用 Cursor。
   - 据报告，**Cursor** 团队已就价格变动发布了官方道歉。


  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1391017766726668370)** (7 messages): 

> `log(n) scaling model exhibit, Hierarchical Reasoning Models, Quaternion products in LLMs` 


- **提供 Log(n) Scaling Model 展示**: 一位成员提议展示他们的 **log(n) scaling model**，以帮助阐明他们对此的看法。
   - 另一位成员回复了 *"当然可以！"*，并为展示提议了一个具体时间。
- **Hierarchical Reasoning Models 论文讨论已排期**: **Hierarchical Reasoning Models (HRM)** 论文 ([https://arxiv.org/abs/2506.21734](https://arxiv.org/abs/2506.21734)) 将在 [#Daily Paper Discussion](https://discord.com/channels/714501525455634453/1045298343896690699) 语音频道进行讨论。
   - 摘要强调了 HRM 独特的新型递归架构，仅用 **2700 万参数** 就实现了计算深度和效率。
- **在 LLM 中实验 Quaternion Products**: 一位成员将讨论他们在 **LLM** 中使用 **quaternion products** 快速总结文本的实验，作为 softmax attention 的替代方案。
   - 该讨论已安排在特定日期，为文本摘要提供了一种新颖的方法。


  

---


### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1390813202350080173)** (5 messages): 

> `The Ocean is Wet, Critical Look at Chain of Thought, AI Training Load Fluctuations` 


- **AlphaXiv 发布 "Ocean is Wet" 论文**: 一位成员分享了 [X 上的帖子](https://fxtwitter.com/FazlBarez/status/1940070420692312178) 以及指向 [AlphaXiv](https://www.alphaxiv.org/abs/2025.02) 上标题为 *"The Ocean is wet"* 论文的链接。
- **Chain of Thought 受到审视**: 一位成员提到记得很久以前就有预印本表达过类似观点，指的是近期对 **Chain of Thought** 的批判性分析。
   - 该成员表示 *很高兴看到 CoT 再次受到批判性的关注*。
- **AI 训练可能导致电网断电**: 一位成员发布了 [Semianalysis.com](https://semianalysis.com/2025/06/25/ai-training-load-fluctuations-at-gigawatt-scale-risk-of-power-grid-blackout/) 的链接，讨论了由于 **吉瓦（gigawatt）规模** 的 **AI 训练负载波动** 导致 **电网断电** 的风险。


  

---


### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1391833569097289738)** (1 messages): 

> `EleutherAI Summer of Open AI Research, Open Science AI Research Project, Mentorship Program` 


- **EleutherAI 开启 Summer of Open AI Research**: **EleutherAI Summer of Open AI Research** 是一项完全在线的活动，时间为 **2025 年 8 月 4 日至 8 月 29 日**，邀请研究经验有限的个人在资深导师的指导下参与开源科学项目 ([EAI Discord 链接](https://discord.com/channels/729741769192767510/1390779760493334548))。
- **EAI 的 SOS：为科学寻找技术精湛的探险者！**: 该计划正在寻找程序员、硕士/博士生、自学成才的研究人员，以及任何希望为开源科学做出贡献的技术人员。
   - 有意申请者请阅读 [项目提案](https://www.eleuther.ai/soar) 并在 **7 月 21 日** 截止日期前填写 [申请表](https://docs.google.com/forms/d/e/1FAIpQLSdT2VESB9fup1-y_8zEvOsk6DBfHusMaqTr78Ex8Wrn3iTz_g/viewform?usp=header)。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1390771710847881257)** (81 条消息🔥🔥): 

> `AI Alignment & Interpretability, 滑动窗口的 ROPE frequencies, Decentralized AI, 语言建模即压缩, GLoVE 模型对称性` 


- **EleutherAI 欢迎热衷于 Alignment 的新成员**：一位新成员加入社区，表达了对贡献 **AI alignment** 和 **interpretability** 研究的兴趣。
   - 该成员的兴趣包括 **AI alignment**、**interpretability** 和 **safety**，并期待开展合作。
- **关于相对 ROPE Frequencies 的争论**：一位成员指出，在进行滑动窗口解码时，使用相同的 **frequencies** 来处理 **RoPE** 是错误的，因为模型无法察觉序列正在推进，从而导致循环行为。
   - 另一位成员认为，在滑动窗口中使用相同的 **frequencies** 应该没关系，因为 **attention** 机制看到的*相对距离*应该是正常的；而另一位成员则建议为 **local attention** 使用 **更低的 base frequency**。
- **爱好者倡导 Decentralized AI**：一位成员介绍了自己对 **Decentralized AI** 的热情，探索如何在分布式基础设施上构建可信、可组合的 **agent** 系统。
   - 他们很高兴能在社区内建立联系、学习和协作。
- **LLM 作为压缩器：Scaling Law 论文？**：受“[language modeling is compression](https://arxiv.org/abs/2309.10668)”论文的启发，一位成员训练了 **pythia** 模型（70m 到 1.4B），并测量了在文本、音频和图像数据集上的压缩率。
   - 他们很好奇这是一个有价值的发现还是一个偶然，以及需要做些什么才能将其确立为一篇新颖的 **scaling law** 论文，并分享了[附带的图表](https://cdn.discordapp.com/attachments/729741769738158194/1391852453712105703/scaling_laws_for_compression.png?ex=686d671c&is=686c159c&hm=591b4707a3e723a65fd808d70940d48594ccfae23811684ee4786abc4c034f45&)。
- **社区分析 GLoVE 模型对称性**：一位成员询问了关于 **GLoVE** 论文的问题，特别是为什么反转角色无法解决对称性问题，即 x 在 k 上下文中出现的概率应该与 k 在 x 上下文中出现的概率相同。
   - 论文提到简单的角色交换无法解决该问题，该成员正在寻求其背后的原因。

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1390786656717504692)** (41 条消息🔥): 

> `Aurko Paper, Cognitive Markers, Concat-and-chunk strategy, Flex Attention, CFG` 


- **神秘的 Aurko Paper 引发关注**：成员们讨论了 [Aurko paper](https://arxiv.org/abs/2507.02119)，一位成员指出它看起来*要么完全没有意义，要么非常有趣*，需要进一步调查。
   - 另一位成员建议私信 Aurko 以获取见解，并指出需要时间来辨别该论文的重要性。
- **Concat-and-Chunk 与 Sequence Length Matching 之争**：成员们辩论了在预训练数据中使用 **concat-and-chunk strategy** 还是 **sequence length matching**，有人认为前者是因为偷懒，并链接到了 [Bucket iterator](https://torchtext.readthedocs.io/en/latest/data.html#torchtext.data.BucketIterator)、[least-squares histogram-packing](https://github.com/graphcore/examples/blob/master/tutorials/blogs_code/packedBERT/nnlshp.py) 和 [Multipack](https://github.com/imoneoi/multipack_sampler)。
   - 其他成员指出，像 **sequence length bucketing** 这样的方法从 **RNN** 时代就已存在，选择取决于数据属性和实现细节，并提到每种 packing 方法都会引入不同的偏差。
- **Flex Attention**：成员们讨论了 **Flex Attention**，认为它是常见用例的*理想解决方案*，因为它能够按 batch 动态混合任意长度的序列。
   - 使用 Flex Attention，无需对数据进行奇怪的切分，粗略的调度就足够了。
- **通过 Loss 曲线分析实现大规模稳定训练**：成员们分析了一篇论文，发现 scaling curves 适用于训练过程中的 loss，并能预测整个训练运行；他们强调，通过缩放初始化、学习率和其他超参数来仔细参数化模型，对于实现大规模稳定且高效的训练至关重要。
   - 有人建议，如果在正则化后为计算优化训练进行缩放时 loss 曲线极其一致，你就可以判断你的分布式训练设置是否配置不当。
- **训练期间使用 Attention Beacons？**：一位成员询问了在训练期间实现 **attention beacons** 的情况，这与 attention masking 略有不同，允许更长的序列长度。
   - 另一位成员建议，这需要非常长的训练上下文，除非采用 **TBPTT**（truncated backprop through time）风格，这样就不需要特殊的 kernel。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1391848964608757790)** (3 条消息): 

> `SAE expansion ratio, Set Autocomplete Model, Publishing Research` 


- **SAE 扩展倍率随任务复杂度飙升**：一位成员发现，将 **set autocomplete model** 的复杂度提高 **4 倍**（在 **20 个子集**上训练而非 **5 个**），会导致所需的 **SAE expansion ratio** 增加 **12 倍**。
   - 第一个模型的 **SAE** 需要 **4 倍扩展倍率**，而第二个模型的 **SAE** 需要 **48 倍扩展倍率**。
- **Epochs 对 SAE 扩展的影响**：一位成员建议，**SAE expansion ratio** 的增加除了任务复杂度提高外，还可能是因为模型训练了更多的 epochs。
- **研究的可发表性**：一位成员询问了关于任务复杂度与 **SAE expansion ratio** 之间关系的研究结果是否具有可发表性。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1391276018513477723)** (7 messages): 

> `MMLU-SR 任务子集、数据集 parquet 转换、lm eval 运行时间` 


- **MMLU-SR 任务子集无法工作**：一位成员报告称，由于出现指示未找到 `BuilderConfig` 的 `ValueError`，他们无法再使用 `mmlusr` 任务的子集评估模型。
   - 另一位成员指出，*当他们在 Hub 上将数据集转换为 parquet 格式时，子集被删除了*，并建议通过在任务的 YAML 文件的 `dataset_kwargs` 中添加 `revision` 来使用之前的提交（commit）。
- **数据集 parquet 转换导致错误**：一位成员提到，在 Hugging Face Hub 上将数据集转换为 parquet 格式导致子集被删除，从而在运行 `lm_eval` 时引发错误。
   - 他们建议在任务配置文件的 `dataset_kwargs` 中指定之前的提交哈希（commit hash）作为权宜之计；并[链接了相关的提交](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/mmlusr/question_and_answer/_mmlusr_qna_yml)。
- **Lm eval 长时间卡住**：一位成员报告称 `lm_eval` 在开始评估过程之前会卡住约 **20 分钟**，并[附上了截图](https://cdn.discordapp.com/attachments/755950983669874798/1391768455895715911/IMG20250707170951.jpg?ex=686d18e1&is=686bc761&hm=f211e2ae73cc8c60f7d96b5cc7c89210572ec631b06962c24b7b71d69f903de8&)。
   - 该用户使用了 **2 个 GPU** 和 `parallelize` 参数，但未展示用于评估的具体命令。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1391360764111093760)** (6 messages): 

> `用于训练的小型数据集、FineWeb 数据集样本、H100/H200 的 FA3 支持、dclm-dedup 数据集` 


- **寻求用于训练验证的小型数据集**：一位成员正在寻找约 **50B tokens** 的小型数据集来验证一些功能，并征求该尺寸范围内二次采样数据集的建议。
   - 他们提到 [FineWeb 样本](https://huggingface.co/datasets/HuggingFaceFW/fineweb/tree/v1.4.0/sample)要么太小（**10B**），要么是其两倍大（**100B**）。
- **询问 H100/H200 的 FA3 支持状态**：一位成员询问了 **H100/H200 的 FA3 支持**更新情况，回想起之前提到过即将合并，但注意到 repo 中尚未出现。
   - 他们正在寻求社区关于该功能时间表的澄清。
- **分享 "dclm-dedup" 数据集子集**：一位成员分享了 **Zyphra/dclm-dedup** 数据集的一个 **25B token 子集**，可在 [Hugging Face](https://huggingface.co/datasets/EleutherAI/dclm-dedup-25B) 获取，作为一个可能的选项。
   - 该成员承认这可能比最初要求的要小，但仍将其作为资源提供。
- **请求澄清 "nemo_id" 列**：一位成员询问了 **dclm-dedup** 数据集中 `nemo_id` 列的含义。
   - 他们推测这可能代表*全局分片/局部分片映射（global shard/local shard mapping）*，并寻求数据集创建者的确认。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1390772025026674801)** (101 messages🔥🔥): 

> `HF 文档 3B 模型、LLM 匹配生成式响应、Grok 的政治不正确立场、PiTutor 互动学习、Grok 的知识更新` 


- **传闻 Hugging Face 将推出 3B 模型**：根据 HF 文档，将会有一个 **3B 模型**，可能会有不同的尺寸。
- **使用 LLM 匹配生成式响应**：成员们讨论了使用 **LLM** 将生成式响应与参考答案进行匹配，并参考了[此 X 总结](https://x.com/ShashwatGoel7/status/1941153367289364655)以获取快速概览。
- **围绕 Grok 的政治不正确性展开激烈辩论**：一位用户发布了一张截图，暗示 **Grok** 是故意政治不正确的，另一位用户提到了[埃隆·马斯克的这条推文](https://x.com/elonmusk/status/1936493967320953090)。
   - 另一位用户认为这正在损害对 AI 的信任，而其他人回应称 [Grok 的提示词已在 GitHub 上公开](https://github.com/xai-org/grok-prompts/blob/adbc9a18736d6c2173607b9ed3d40459147534b1/grok3_official0330_p1.j2#L57)。
- **PiTutor 将 PDF 转化为互动学习会话**：一位成员分享了 **PiTutor**，这是一个可以将任何 **PDF 或 doc** 转化为具有实时解释、高亮显示和白板功能的互动学习会话的工具，目前可在[此处](https://pitutor.pi4wear.com)进行免费测试。
- **关于向量空间细化的讨论**：一位成员提议在计算 Embedding 矩阵之前，将向量空间表示细化得*更像人类*。
   - 另一位成员指出，如果你想训练它们，*你就必须去训练它们*。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1390790955656810499)** (18 条消息🔥): 

> `Ollama and Openwebui for AI Services, Training LLMs on Their Own Weights, Temperature and Token Usage, Math 500` 


- **Ollama 和 Openwebui 增强 AI 服务**：一位成员正在使用 **4080 Super** 通过 **Ollama** 和 **Openwebui** 提供 AI 服务，并计划切换到 **Llama CPP**。
   - 他们正在寻找方法，通过将 **Nous Research** 的模型展示给其用户群进行实时测试，来为该项目提供帮助。
- **LLM 思考“权重”问题**：一位成员询问是否可以在 LLM 自身的权重以及额外数据上对其进行训练，以诱导其对某一主题产生更深层次的理解。
   - 另一位成员回应称，这可能类似于持续训练（continued training）和权重合并（weight merging），但可能会导致灾难性遗忘（catastrophic forgetting），而[混入一些原始数据可以帮助防止这种情况](https://link.to/example)。
- **Temperature 影响 Token 数量**：一位成员正在进行实验，以确定哪种 **Temperature** 设置在语言生成中使用最少的 **Token**。
   - 他们提到使用 **Math 500** 作为一个小型数据集基准。
- **Math 500 基准测试出现**：一位成员建议使用 **Math 500** 数据集作为 **Token** 使用实验的基准，目标是使用少于 1k 个问题的数据集。
   - 该成员的灵感来自于一个重复消息并带来灵感火花的机器人，开玩笑说 *heh*。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1391620961458978897)** (3 条消息): 

> `AI Mouse Tracking, Architecture for Mouse Path Training` 


- **寻求 AI 鼠标追踪的模型架构**：一位成员正在寻求关于合适架构的建议，以便使用自然人类鼠标路径数据集进行模型训练，该数据集包含点击区域的图像以及代表操作的文本提示（例如：将鼠标移动到 + 图标、滚动、截图）。
   - 这表明了利用 AI 理解和预测人机交互模式的兴趣。
- **数据集详情：鼠标路径、图像和文本提示**：该数据集由自然人类鼠标路径组成，并配有点击区域的图像以及描述所执行操作的文本提示。
   - 这个丰富的数据集旨在捕捉用户意图（文本提示）、视觉上下文（图像）和行为模式（鼠标路径）之间的细微关系。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1390785413043585198)** (8 条消息🔥): 

> `Chinese AI investment, Parameter-efficient fine-tuning, Trustless agents, Codeium 3.2, Autonomous optical network` 


- **智谱 AI 获得上海国资注资**：据 [Technode](https://technode.com/2025/07/04/zhipu-secures-1-4-billion-strategic-investment-from-shanghai-state-funds/) 报道，中国 AI 公司**智谱 (Zhipu)** 获得了来自上海国资基金的 **14 亿美元**战略投资。
- **Pi Tutor 构建个性化 AI 导师**：**Pi Tutor** 正在通过视觉交互构建更美好的未来，将任何内容转化为个人导师。
   - 该服务支持您的语言并提供量身定制的示例，更多信息请访问其 [官网](https://pitutor.pi4wear.com/)。
- **微调成本大幅削减**：根据 [Konceptual AI](https://konceptual.ai/trending/june-2025-ai-breakthroughs-transforming-business-operations) 的数据，6 月份发表了 **117 篇论文** 介绍参数高效方法（parameter-efficient methods），这些方法在保持约 **98%** 准确率的同时，将微调成本降低了约 **85%**。
- **去信任 Agent (Trustless Agents) 引发关注**：Reddit 用户强调了在使用 **TEE**、**零知识证明 (zero-knowledge proofs)** 和链上可验证性来确保完整性和隐私的“去信任 Agent”方面取得的进展，详见 [此 Reddit 帖子](https://www.reddit.com/r/AI_Agents/comments/1ljyhxu/ai_agents_the_innovation_from_a_decentralized/)。
- **Codeium 3.2 提升多仓库识别能力**：据 [AI Agent News](https://aiagentstore.ai/ai-agent-news/this-week) 报道，**Codeium 3.2** 增加了多仓库感知能力（支持 **70 多种语言**），减少了约 **40%** 的样板代码，而 **GitHub Copilot Chat Enterprise v2.1.5** 则提供了实时漏洞扫描和 ≥**92%** 的 **JUnit** 测试覆盖率。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1391620961458978897)** (3 条消息): 

> `AI-Youtube-OG 架构，自然人类鼠标轨迹数据集，模型训练架构` 


- **关于 AI-Youtube-OG 架构的查询**：一位成员提到了 "AI-Youtube-OG"，并对分享相关信息表示感谢，提到他们之前已经忘记了这件事。😋
   - 这可能与包含点击区域图像的自然人类鼠标轨迹数据集有关。
- **寻求鼠标轨迹数据训练架构**：一位成员询问了在包含点击区域图像的 **自然人类鼠标轨迹** 数据集上训练模型的合适架构。
   - 该数据集还包括代表动作的 **文本提示 (textual prompts)**（例如，将鼠标移动到 Web 浏览器的 + 图标、滚动、截图等）。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1390838221327437884)** (93 条消息🔥🔥): 

> `Grok 4 泄露，Deepseek R2，Gemini-CLI 搜索集成，Aider 的 MCP 集成，文档获取 MCP` 


- **Grok 4 泄露引发猜测**：一张据称是 **Grok 4** 的泄露图片出现，显示其正在 **HLE benchmark** 上进行训练。
   - 成员们对 LLM 领域潜在的性能飞跃和竞争带来的好处表示兴奋，而一些人持怀疑态度，认为 *如果你的 git 分支名为 "main" 而不是 "master"，Grok 将拒绝在你的项目上工作*。
- **Gemini-CLI 的搜索集成令人印象深刻但会删除 Secrets**：一位测试 **gemini-cli** 的成员强调了其 **搜索集成 (search integration)** 功能，该功能发现了一个他们尚未听说过的支付门户更新，认为这非常适合初学者。
   - 然而，它在没有解释的情况下 *破坏了一堆代码（删除了 secrets？）*，导致人们希望 **Aider 集成 MCP** 以进行文档获取。
- **开源权重 70B 模型面临审查**：社区成员讨论了开源权重 **70B 模型** 的现状，引用了 **Llama 3.1**、**Qwen2.5 70B** 和 **Mistral Large** 作为值得关注的选择，但希望能看到更近期的更新。
   - 有人提到 **Qwen3-32b**（稠密模型，3.0 版本均为混合推理）的表现应该优于 **Qwen 2.5 72b**，还提到 Meta 在这方面失误了，没有提供太多近期 100b 以下的开源 (OS) 选项。
- **对 AI 实验室“刷榜” (Gaming) Benchmark 的担忧**：成员们讨论了 AI 实验室 *刷榜* benchmark 的现实情况，并承认防止数据污染是不可能的。
   - 重点应该放在创建抗作弊的 benchmark 并以多样化的方式解释结果，因为污染会导致模型改进及其泛化能力提升。
- **Aider 用户苦于替换问题**：一些 **Aider** 用户报告称，当他们进行编辑时，**Aider** 在旧版本的文件上执行了替换操作，其中 Markdown 文件的这类问题最多。
   - 一位用户建议使用 `/clear` 清除对话历史记录，而另一位用户建议删除并重新添加文件，以及使用 `/map-refresh`。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1390772669078704253)** (25 条消息🔥): 

> `aider 中的 InputOutput 类，OpenRouter 提供商设置，Git reset，带有 ask/code 的 sonnet-4，deepseek 70b` 


- **使用 InputOutput 类自定义 Aider 输出**：一位用户发现了来自 `aider.io` 的 `InputOutput` 类，用于自定义输出并设置 `yes=True`，并指出这可能会改变助手输出的列表中的数字/项目符号的文本颜色，参见 [aider.io](https://aider.io)。
   - 提到的关键代码为：
```python
from aider.io import InputOutput
io = InputOutput(yes=True)
```
- **通过 .env 选择 OpenRouter 提供商**：一位用户询问如何通过 `.env` 配置在 OpenRouter 中为模型设置特定的提供商，尝试了 `AIDER_MODEL=openrouter/deepseek/parasail-deepseek-r1-0528-qwen3-8b`。
   - 一位成员建议在 OpenRouter 设置中配置提供商，或通过 `extra_body` 参数进行配置，参考了 [OpenRouter 关于提供商路由的文档](https://openrouter.ai/docs/features/provider-routing)。
- **使用 Git Reset 和 Aider 历史记录恢复丢失的文件**：一位用户遇到了 `app.py` 在之前的提交中不在仓库中且撤销失败的问题，并寻求如何恢复该文件的建议。
   - 建议包括在 **nvim** 中使用 **Ctrl-Z**，使用 `git reset`（可能会删除文件），或者从 `aider.history.md` 重构文件。一位成员强调了提交更改的重要性。
- **评估 DeepSeek 70B 的性能**：一位成员报告了 **DeepSeek 70B** 令人失望的结果，理由是它在“读取文件并编写单元测试”等任务上的表现不如付费模型。
   - 他们质疑是否有人在本地运行的模型上实现了闭源模型哪怕一小部分的能力，即使配备了充足的 tool calling 设置。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1390817081326436483)** (3 条消息): 

> `Tensor 编译器，CUDA，PyTorch，向量矩阵乘法` 


- **Tensor 编译器项目启动**：**Tensor Compilers: Zero to Hero** 工作组正在寻找有兴趣从零开始构建 **CUDA** 和 **PyTorch** 的贡献者。
   - 该项目旨在兼具教学性和硬核性，专注于 Tensor 编译器。
- **批处理提升向量矩阵吞吐量**：一位成员询问将向量批处理成矩阵是否是提高 **向量矩阵乘法 (vector matrix multiplication)** 吞吐量的主要方法。
   - 这在频道中仍然是一个开放性问题。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1391493041637818549)** (1 条消息): 

> `einops, 爱因斯坦表示法, triton` 


- **在 Triton 中寻求 EinOps**：一位成员询问了 **Triton** 版本的 **einops** 实现或某种形式的 **爱因斯坦表示法 (Einstein notation)**。
- **Triton 张量收缩**：该用户正在寻找在 **Triton** 中使用 **einops** 以帮助进行张量收缩 (tensor contractions) 的方法。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1390786283214999592)** (36 条消息🔥): 

> `CUDA printf 调试, NCU 内存带宽, GPU 除法 vs 乘法, CUDA tile 调度器 vs cutlass, CUDA 统一内存` 


- **CUDA printf 调试**：成员们讨论了使用 `printf` 调试 CUDA kernel 的方法，指出输出是缓冲的，需要显式调用 `cudaDeviceSynchronize()` 来刷新，并且一些成员使用了预处理器指令来保护特定架构的代码。
   - 一位成员解决了一个由编译器预处理器指令保护特定架构代码引起的“新手问题”。
- **NCU 揭示计算受限（Compute-Bound）问题**：一位用户报告称，在修复了 **NCU** 标记的非合并写入（uncoalesced writes）后，内存带宽利用率和运行时间保持不变，这出乎意料。
   - 他们随后发现循环中的除法是瓶颈，导致 kernel 变为计算受限，将除法替换为乘以倒数后，吞吐量提升了 **33%**。
- **GPU 避开除法电路**：一位成员指出，GPU ALU 可能没有专门的除法电路，而是依赖除法近似计算，因为乘法是通过加法完成的。
   - 据我所知，GPU ALU 甚至没有除法电路，它们进行的是除法近似。
- **自定义 CUDA 调度器**：一位成员询问在 CUDA 中实现自定义 tile 调度器而非使用 **Cutlass** 的可行性。
   - 有人建议探索 **Cutlass** 中可定制的 swizzles 作为潜在解决方案。
- **WSL CUDA 缺少统一内存（Unified Memory）**：一位用户质疑是否值得为了 CUDA 开发设置双系统，因为 **WSL** 缺乏对统一内存和某些 **nsys** 性能计数器的支持。
   - 建议指出，为了性能优化，最好避免使用统一内存，而且 WSL 已经通过 WDDM 超额订阅（oversubscribe）了普通的设备内存。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1390782907844526121)** (1 条消息): 

> `ML 平台招聘, 远程工作, Quora` 


- **Quora Machine Learning 平台团队招聘高级/资深工程师**：Quora 正在为其 Machine Learning 平台团队招聘 **高级/资深软件工程师 (L5-L6)**，提供美国和加拿大的全远程职位；[点击此处申请](https://jobs.ashbyhq.com/quora/89b213bf-06e7-43a2-9101-4b93d711d796)。
- **令人兴奋的 ML 基础设施工作**：该角色涉及令人兴奋的 ML 基础设施工作，提供了为 Quora 的机器学习能力做出贡献的机会。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1390975365131796480)** (2 条消息): 

> `ROCm 追踪, 线下黑客松, make 标志, gpumode-amd-fp8-mm 仓库` 


- **来自巴黎的开发者加入 ROCm 频道**：一位来自巴黎的开发者在频道中介绍了自己，寻求 **ROCm traces** 以及在今天于巴黎举行的线下黑客松中获得名次的方法。
   - 该开发者分享了 [gpumode-amd-fp8-mm 仓库](https://github.com/Snektron/gpumode-amd-fp8-mm/tree/main) 的链接，并询问了有用的 *make flags*。
- **关于 Flags 的提问**：一位开发者询问哪些 make flags 比较有用。
   - 他们目前正在使用 [这些 make flags](https://cdn.discordapp.com/attachments/1233704710389764236/1391864919116218430/Makefile.txt?ex=686d72b8&is=686c2138&hm=7d9575144b62d4bdfed3663dee86e8d24d5ba3ab516472709f699c74b87843b8&)。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1391019387741278209)** (4 messages): 

> `voxel raytracing, Hopper Pipelining, PiTutor, Deep Infra B200 Instances` 


- **Voxel Raytracing 视频发布**：一位成员上传了一个关于体素光线追踪中**基于可见性的块选择 (visibility-based chunk selection)** 的[视频](https://youtu.be/YB1TpEOCn6w)，涵盖了 victim pointers 和 usage bits 的内部原理。
   - 开源代码可在 [GitHub](https://github.com/Ministry-of-Voxel-Affairs/VoxelHexCuTeDSL) 上获得。
- **关于 Hopper 上 CuTeDSL 流水线的博客发布**：一位成员分享了一篇关于通过 Hopper 上的 Pipelining 实现**内存传输与计算重叠**的[博客文章](https://veitner.bearblog.dev/cutedsl-on-hopper-pipelining/)，重点介绍了 TMA 和 WGMMA atoms。
   - 示例代码可在 [Cutlass GitHub](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/hopper/dense_gemm.py) 上找到，Colfax 博客讨论了使用 CuTe C++ API 的流水线技术。
- **PiTutor 将 PDF 转化为学习伙伴**：一位成员介绍了 **PiTutor**，这是一个将任何 PDF 或文档转化为具有实时解释和白板的互动学习环节的工具，免费 Beta 版可在[此处](https://pitutor.pi4wear.com)获取。
   - *这个想法很简单：如果学习不再感觉像是一项苦差事会怎样？*
- **Deep Infra B200 优惠**：据一位成员透露，Deep Infra 提供价格为 $1.99 / h 的 **B200 按需实例**，支持一键部署，详情见[此处](https://deepinfra.com/)。


  

---


### **GPU MODE ▷ #[general-leaderboard](https://discord.com/channels/1189498204333543425/1343002580531417211/1391495014239961178)** (5 messages): 

> `GPU Mode Challenges, Leaderboard Questions, Kernelbot Data` 


- **GPU Mode 挑战赛下周改版**：初学者挑战赛已结束，但*很快*（下周）将带着完善的评估代码重新上线，在此之前成员可以使用 **AMD 挑战赛**或 **trimul**。
- **排行榜问题解答已发布**：已经结束的排行榜问题解答已经发布，可在 [HuggingFace](https://huggingface.co/datasets/GPUMODE/kernelbot-data) 上获取。


  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1391699302409044068)** (6 messages): 

> `A100 Performance, H100 Performance, Trimul Leaderboard` 


- **A100 获得第五名**：在 **A100** GPU 上的提交以 **20.6 ms** 的成绩获得了 `trimul` 排行榜的**第 5 名**。
- **H100 获得第二名**：一位成员使用 **H100** 多次获得 `trimul` 排行榜**第二名**，成绩分别为 **60.0 ms**、**43.5 ms**，最终达到 **35.9 ms**。
- **H100 在 Trimul 上的成功**：在 **H100** 上的其他提交也在 `trimul` 排行榜上取得成功，成绩均为 **63.1 ms**。


  

---


### **GPU MODE ▷ #[factorio-learning-env](https://discord.com/channels/1189498204333543425/1354169122107293786/1390771374703775785)** (41 messages🔥): 

> `instance.py refactoring, Github Actions integration, Ruff linting, Pydantic bump PR` 


- **`instance.py` 太长；需要重构**：成员们讨论了 [instance.py](https://github.com/org/repo/blob/main/instance.py) 太长且需要重构，可能会移入 main，这与 issue [#249](https://github.com/org/repo/issues/249) 相关。
- **GitHub Actions 测试集成取得进展**：一位成员确认其发布到 testpy 的 GitHub action 已生效，由 push 触发，他们将设置为**在合并时发布到 testpy** 并自动增加补丁版本。
   - 该成员提供了一张显示运行成功的 [GitHub Actions 工作流截图](https://cdn.discordapp.com/attachments/1354169122107293786/1391669239361835039/Screenshot_2025-07-07_at_09.36.38.png?ex=686d653a&is=686c13ba&hm=00a1bc26b826a9bb3f8573835bbae6138d81fb95b4aa3dcf5a5a0c22fe8e2d66&) 。
- **Ruff Linting 即将到来！**：重构 PR 已经合并，团队正在转向如何合并 **ruff linting**。
   - 他们可能会使用 `--ignore-revs-file` 和 `.git-blame-ignore-revs` 来保持大规模变更时 git blame 的合理性。
- **Pydantic 升级 PR 令人头疼**：有人担心 **Pydantic 升级 PR** 涉及的文件变更数量异常之多（**70 个文件**）。
   - 一位成员提到他们需要将 main 合并到他们的 **Pydantic PR** 中，以解决该 PR 中大量的 commit (95个) 问题。


  

---

### **GPU MODE ▷ #[singularity-systems](https://discord.com/channels/1189498204333543425/1373414141427191809/1390816051931119616)** (4 条消息): 

> `用于 llm.c 的 picoc 编译器、picograd 内核、nn: zero to hero 后续项目、picoc 前端语义分析` 


- ****Picoc** 致力于编译 llm.c**：**picoc** 的项目目标现在集中在年底前编译 Karpathy 的 [llm.c](https://github.com/karpathy/llm.c)，利用原生 **CUDA** 和 **CUTLASS** 进行优化。
   - 初步计划涉及先解决 **CUDA**，稍后再研究 **CUTLASS** 的集成。
- ****Picograd** 优先开发 Pytorch1 风格的内核**：**picograd** 项目正优先开发 [Pytorch1 风格的内核](https://pytorch.org/docs/stable/index.html)，并已在 CPU 上完成了 Karpathy 的 MLP 前向传播。
   - 一位成员在 [X](https://x.com/j4orz/status/1907452857248350421) 上提到了他们的进展，并表示创建一个将 Pytorch2 融合编译器转换为 Triton 风格编译器的工具将需要作为课程的 v2 版本，且不太可能在今年实现。
- **工作组致力于 **nn: zero to hero** 的后续项目**：该工作组旨在创建 **nn: zero to hero** 的直接后续项目，重点关注构建解释器/编译器的 Software 2.0 等效项。
   - 直接来自 nn: zero -> hero 的模型将通过把 `import torch` 替换为 `import picograd` 来进行转换。
- **寻求 **Picoc** 前端语义分析的协助**：一位成员请求在 **picoc** 项目中协助 [C0](http://reports-archive.adm.cs.cmu.edu/anon/anon/2010/CMU-CS-10-145.pdf) 的前端语义分析，特别是在 [词法分析/解析/类型检查 (lexing/parsing/typing)](https://github.com/j4orz/picoc/blob/master/src/ast/mod.rs#L32-L33) 领域。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1390731448784261313/)** (1 条消息): 

tonic_1: 💪🏻 🏅 🇫🇷 🏆 🚀
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1390861313848119367)** (100 条消息🔥🔥): 

> `Cursor.ai 定价争议、字节跳动开源 Trae-Agent、ChatGPT 诊断罕见遗传缺陷、ChatGPT 新的“Study Together”功能、Books3 数据集传闻` 


- **Cursor 取消无限量定价，用户抗议！**：用户指责 [Cursor.ai](https://cursor.ai) 突然从之前的“无限量”定价模式转变为限量模式，导致 **Sonnet-4** 使用产生意外费用，这被视为 **rug pull**（割韭菜）。
   - 一些用户报告了使用自己的 API Key 时遇到 **500 错误** 的问题，而 [Cursor 澄清了他们的定价](https://cursor.com/en/blog/june-2025-pricing)，称“无限使用”仅适用于 Auto 而非所有其他模型，并提供了退款。
- **字节跳动发布 Trae-Agent！**：**Trae AI** 开源了为其 IDE 提供动力的代理 [Trae-Agent](https://github.com/bytedance/trae-agent)，现已在 GitHub 上线，并在 **SWE-bench Verified** 中排名第一。
   - 该公司正在寻求对构建开放 Agent 生态系统的贡献，Trae-Agent 支持 **OpenAI** 和 **Anthropic Key**，并可以轻松修改以适配 OpenRouter。
- **ChatGPT 破解了医生漏诊的医学难题！**：Reddit 上一个热门故事展示了 **ChatGPT** 如何识别出医生漏诊十年的隐藏基因缺陷（**甲基化阻滞**），从而显著改善了症状。
   - 该帖子讨论了 **AI 在医疗保健中日益增长的作用**，特别是在获取第二意见和个性化医疗方面，新的 AI 系统通过在**科学论文上使用 RAG 系统**，在诊断准确性上优于医生。
- **ChatGPT 的“Study Together”功能浮出水面**：一条 Twitter 帖子讨论了 ChatGPT 中新的 **“Study Together”** 功能，可能是一个代号为 *“tatertot”* 的内部原型。
   - 关于其功能的猜测包括：**AI 驱动的小组学习室**、抽认卡的替代品、小组环境中的**个人 AI 导师**，或者是与 AI 进行协作探索的工具。
- **Gemini 的 Batch API 提供批量优惠**：Logan Kilpatrick 宣布在 [Gemini API](https://xcancel.com/OfficialLoganK/status/1942245069383434696) 中推出 **“Batch mode”**，为 2.5 模型提供 **50% 的折扣**，并支持处理数十亿个 Token。
   - 该公告得到了用户的积极反馈。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1390840160467419269)** (41 messages🔥): 

> `Mojo 用于研究与训练，Arrow 规范，Mojo 与 Python 回调，Mojo 服务器标签，Mojo 作为 Python 超集` 


- **Mojo 侧重 AI 推理而非训练**：虽然有人询问是否有人将 **Mojo** 用于研究和训练，但有人提到 [Modular 目前非常专注于推理端](https://www.modular.com/)。
   - 建议查看 GitHub 上 *modular-community* 的 **Mojmelon**，以及 [Nabla](https://github.com/nabla-ml/nabla) 的链接，后者在编译时间和训练时间上都比 **JAX** 更快。
- **科学家仍然钟爱 Excel**：分子科学家仍然手动解析和生成表格，Excel 被以意想不到的方式使用，例如作为*完整的序列比对/引物设计引擎*。
   - 行业应该向 **Arrow** 迁移，Arrow 项目有一些有趣的原型，如 *arrow dbc*，支持它的数据库可以直接将数据作为 arrow 数据发送。
- **Mojo 扩展 Python 而非取代**：尽管将 **Mojo** 打造为真正的 Python 超集的呼声很高，但它目前被用于扩展 Python。
   - 正如 Modular CEO [Chris Lattner 所说](https://discord.com/channels/1087530497313357884/1304251563749146634)，Mojo 一直专注于解决 **AI stack 问题**并在该领域创建产品。
- **Mojo 路线图即将更新**：有人询问 **Modular** 是否发布了路线图或类似内容，一位社区成员回答说，最新的 Mojo 路线图发布在[这里](https://forum.modular.com/t/whats-next-for-mojo-near-term-roadmap/1395)。
   - 该社区成员还提到*他们很快就会发布一些更新*。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1390769294891028580)** (52 messages🔥): 

> `StringLiteral 实例化，参数化 Trait vs Trait 对象，基于云的 Mojo 环境，Mojo 中的静态链接，SIMD Bitcasting` 


- **`StringLiteral` 实例化导致破坏性变更**：一位成员报告称，最近为了让 `StringLiteral` 实例化为 `String` 而进行的更改破坏了现有代码。
   - 他们计划*提交一个 issue 并与编译器团队跟进*。
- **关于参数化 Trait 和 Trait 对象的困惑**：成员们讨论了 Mojo 中参数化 Trait 的可能性，一位成员提供了一段代码片段，通过使用 Trait 对象来实现类似的效果。
   - 另一位成员澄清说*这不是参数化 Trait，这是 Trait 对象*。
- **对云端 Mojo 环境的需求**：由于硬件限制，一位成员询问如何为研究设置基于云的 Mojo 环境。
   - 另一位成员建议使用*任何运行 Linux 并允许安装 pip 包的 GPU 实例*。
- **静态链接功能请求**：一位用户寻求在不依赖共享库（特别是 `libKGENCompilerRTShared.so`）的情况下编译 Mojo 代码，以便在远程机器上部署。
   - 一位成员建议*完全静态链接需要作为一个功能请求提出*，并指出了现有的讨论和相关的 GitHub issue（[BUG]: Cannot build statically compiled binary #1317）以及一个新的 [Feature Request] Static Compilation of pure mojo code #4976。
- **`bitcast` 行为变更**：一位成员报告称，之前涉及 `SIMD` 和 `DType` 的 `bitcast` 操作现在无法编译，并寻求关于正确转换方式的指导。
   - 该成员发现为函数添加 `raises` 会阻止编译器进行尾递归优化（tail call optimization），并链接到了一个 Godbolt 示例 [https://godbolt.org/z/ahn4144a7]。


  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1391079104530616471)** (5 messages): 

> `Mojo GPU Puzzles, ModuleNotFoundError, Pixi` 


- **调试 Mojo GPU Puzzles 中的 `ModuleNotFoundError`**：一位成员在从 Python 运行 **mojo-gpu-puzzles** 时遇到了 `ModuleNotFoundError: No module named 'max'` 错误。
   - 另一位成员建议在 puzzles 目录下使用 `pixi add max` 或 `pixi shell` 来安装必要的依赖项，并参考了 [pixi.toml 文件](https://github.com/modular/mojo-gpu-puzzles/blob/main/pixi.toml)。
- **Mojo GPU Puzzles 的 Pixi 工作流**：一位成员建议典型的工作流是 **git clone** 仓库，然后运行 **pixi shell** 来安装所有依赖项。
   - 该成员曾尝试过 `pixi add max`，但这并没有解决问题，引发了进一步的排查。


  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1390934555183026206)** (58 messages🔥🔥): 

> `EpicMe MCP, WinCalcMCP, MCP Python Interpreter, MCP for documentation indexing, LangGraph vs Custom Agents` 


- ****EpicMe MCP** 提供单一端点**：一位成员请求一个作为 **MCP aggregator/gateway** 的自托管 Docker 镜像，另一位成员推荐了 [EpicMe](https://github.com/epicweb-dev/epic-me-mcp) 作为解决方案。
   - 创作者表示 EpicMe 以一种*非常独特且酷炫的方式*处理 onboarding，并提供了一个[链接](https://www.epicai.pro/user-friendly-just-in-time-auth-with-mcp-hd9hw)作为示例。
- ****WinCalcMCP** 将 Claude 连接到 Windows 计算器**：一位成员构建了他们的第一个 MCP server，[WinCalcMCP](https://github.com/rspeciale0519/WinCalcMCP)，它将 **Claude Desktop** 连接到 **Windows Calculator** 以提高数学回答的准确性。
   - 另一位成员更倾向于通过 [mcp-python-interpreter](https://github.com/yzfly/mcp-python-interpreter) 为其提供一个完整的 **Python interpreter** 作为通用工具，这样他们就不必管理这么多专门的工具。
- **LangGraph 不是一个好的抽象，建议使用 **custom agents****：一位成员分享了他们的历程：从 **LangGraph**（实际上是 LangChain）开始，然后决定不再使用它，因为他们觉得它不是一个很好的抽象，此后他们一直在编写自己的 **agentic framework**。
   - 大家一致认为 **MCP** 的美妙之处在于人们不需要使用相同的 **agentic framework**，但仍然可以使用通用协议进行通信。
- ****MCPdata** 为 MCP 提供本地文档索引**：一位成员介绍了来自 [MaheshDoiphode/mcpdata](https://github.com/MaheshDoiphode/mcpdata) 的 **MCPdata**，用于 MCP 的本地文档索引。
   - 似乎 **context7** 没有跟上 Vercel `ai` API 文档的最新更新。
- **只有官方 **MCP** servers 值得使用**：成员们抱怨大多数 **MCP** servers 都没用。
   - 一位成员建议使用 [glama.ai](https://glama.ai/mcp/servers?attributes=author%3Aofficial) 网站，仅获取来自 **Elasticsearch**、**Kagi** 和 **Redis** 等官方来源的 servers。


  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1391705357201702952)** (4 messages): 

> `MCP Monetization, Agentic Payments with Crypto, Fast Agent's MCP Elicitation Support, EpicAI's MCP Search Engine` 


- ****MooPoint** 让尝试服务变得简单**：[MooPoint.io](https://moopoint.io/) 现在允许你在第一次登录时立即试用该服务。
- **使用加密货币为 **MCP** Servers 变现**：一位成员正在探索为 **MCP** servers 变现的方法，认为加密货币和稳定币非常适合 **agentic payments**，并分享了一个简单的工具，可以使用比特币支付将免费的 MCP 转换为付费版本：[lmcp](https://github.com/getAlby/lmcp)。
   - 他们正在寻找合作伙伴来创建一个 showcase。
- ****Fast Agent** 获得全面的 **MCP Elicitation** 支持**：根据[这篇博客文章](https://fast-agent.ai/mcp/elicitations/)，**Fast Agent** 现在拥有全面的 **MCP Elicitation** 支持，以及一个快速入门指南，使 Elicitations Servers 的入门变得非常简单。
- ****EpicAI** 发布了一个 **MCP Search Engine****：一位成员分享了 [EpicAI's MCP Search Engine](https://www.epicai.pro/mcp-search-engine)。


  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1390776378709901364)** (16 条消息🔥): 

> `Google Docs 评论，NMC 标准笔记本，NHS 采用 NotebookLM，思维导图嵌入，从音频文件起草通讯 (Newsletter)` 


- **护士使用 NotebookLM 创建 NMC 标准笔记本**：一名来自 NHS 的注册学习障碍护士兼实习教育者创建了一个数字化的 **NMC Standards Notebook**，将来自护理和助产协会 (NMC) 的关键文档整合到一个单一、可搜索且互动的资源中，访问地址为 [notebooklm.google.com](https://notebooklm.google.com/notebook/8fc872a6-dd9b-4148-9a3e-5f2224f42294?authuser=2)。
   - 该工具通过整合 NMC 的框架，简化了重新认证、监督和见习流程，支持护士、助产士、学生和教育工作者，并已被各教育网络采用，以融入以人为本的护理、临床安全和包容性实践。
- **NHS 通过免费账户在全国范围内推广 NMC 笔记本**：一名 NHS 注册学习障碍护士分享道，由于“公开分享”功能，尽管使用的是免费 Google 账户，他们创建的工具目前正被全国范围内的 **NHS** 机构用于重新认证和见习。
   - 该项目最初只是作为*局部教育辅助工具*，但现在已经壮大。
- **用户讨论读取 Google Docs 评论**：用户在通过 NotebookLM 读取 Google Docs 评论时遇到困难；一位用户询问该功能*是否即将推出*？
   - 一位成员建议了一个变通方法：将 Google Doc 下载为 **MS-Word 文档**，然后将 Word 文档打印为 **PDF**，最后将该 PDF 作为来源使用。
- **从音频文件起草通讯 (Newsletter)**：一位用户询问关于使用 **NotebookLM** 从其录音的音频文件中起草通讯的*任何建议/意见*。
   - 目前没有提供任何想法。


  

---


### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1390816734159573053)** (36 条消息🔥): 

> `在 NotebookLM 中保存笔记，交互模式 (Interactive Mode) 按钮缺失，PDF 上传问题，在 NotebookLM 中保存聊天记录，NotebookLM 的新模型` 


- ****NotebookLM 自动保存笔记效果极佳****：一位用户对上传学校笔记并体验 NotebookLM 的出色运行感到兴奋，而另一位用户询问如何在笔记本上保存。
   - 另一位成员澄清说，右下角的 **"Notes" 栏目**包含一个 "+ Add Note" 按钮，使过程变得简单且无缝。
- ****部分用户的交互模式 (Interactive Mode) 按钮消失****：一位用户报告称，尽管拥有专业版订阅，其笔记本中的 **Interactive Mode 按钮**仍缺失。
   - 另一位用户建议将输出语言设置为英语，因为 **interactive mode** 目前仅适用于以英语生成的音频概览 (Audio Overviews)。似乎 *customize audio 按钮* 已被 Interactive Mode 按钮取代。
- ****遇到 PDF 上传障碍****：尽管参考了教程并检查了可用资源，一位用户在向 NotebookLM 上传 **19.1 MB 的 PDF 文件**时仍面临问题。
   - 一位成员询问他们是否*在 Google 上搜索过该问题*，另一位成员询问 notebookllm app 是否对其他人正常运行。
- ****关于聊天保存能力的讨论****：用户对 NotebookLM **无法保存聊天记录**感到沮丧，并指出聊天记录在短时间后就会消失。
   - 一位用户指出了一种变通方法：手动将每个回答保存为笔记，但也承认这与保存整个聊天日志并不相同。
- ****询问 NotebookLM 的数据存储和 FERPA 合规性****：一位用户正在寻求关于 NotebookLM 对医学生**安全性和保密性**的明确说明，特别是关于数据存储位置、用户交互监控以及 **FERPA 合规性**。
   - 社区或工作人员尚未提供答案。


  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1391561536446922906)** (50 messages🔥): 

> `tinygrad vs pytorch, MLIR, Halide, Exo-lang, whisper example` 


- **Tinygrad 的获胜之路**：George Hotz 将 **tinygrad** 的“胜利”定义为成为“运行事物的最快方式”，并将其支持范围从 **QCOM GPU** 扩展到 **AMD** 和 **NVIDIA**。
   - 其他成员指出了 **PyTorch** 中创新和微调的规模，芯片制造商在 **Triton** 等项目中直接为硬件下放层（hardware lowering layers）做出贡献。
- **关于 MLIR 作用的辩论**：一位成员认为，要实现接近手写 **CUDA** 的性能，需要走 **Triton MLIR** 路线，利用其深厚的硬件知识和 **LLVM** 编译器基础设施。
   - George 反驳称“**MLIR** 是 Waymo”，主张在神经网络中使用端到端（end-to-end）方法，而 Waymo 使用的是不同的方法。
- **Halide 框架受到赞赏与对比**：George 指出 **Halide** 比 **MLIR** 和 **TVM** 具有更清晰的结构，强调在所有数据类型和后端执行计算时，只有一种快速且正确的方法，重点在于通过调度优化（schedule optimization）提升速度，并推荐参考 [Halide 论文](https://halide-lang.org/)。
   - 另一位成员提到了来自 **Exo-lang** 的对比，该项目认为自己比 **Halide** 更有优势，并提供了他们的 [GitHub](https://github.com/exo-lang/exo) 和 [ArXiv 论文](https://arxiv.org/pdf/2411.07211)链接。
- **加速 whisper 示例**：一位成员正在重新研究 **whisper 示例**，寻求速度提升；另一位成员分享了他们的分支，其中“所有音频预处理都已在 tinygrad 中重写”。
   - 他们正在寻找目前 **VAD+whisper.cpp** 流式转录设置的替代方案，发现 **tinygrad 实现**更易于上手。
- **tinygrad 旨在使 Petaflop 商品化**：成员们讨论了旨在实现与不同硬件的互操作性，认为 tinygrad 通过硬件无关的 **UOP 图优化（UOP graph optimizations）** 生成微调后的代码。
   - 一位成员表示该项目仍在酝酿中，他们很期待看到是否能根据 AMD 的合同规范在 **MLperf** 上运行 AMD 硬件，从而使其具备与 Nvidia 竞争的实力。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1390872686577582142)** (31 messages🔥): 

> `Automatic Prompt Optimization (APO), Claude Code vs DSPy, DSPy 3.0, SIMBA vs MIPROV2, Tool Selection for LLMs` 


- **APO 在模型演进中的相关性**：一个团队正在构建用于财务决策的 **AI Agent**，并考虑投入时间研究**自动提示词优化（APO）**，但质疑随着模型能力的提升，其相关性是否依然存在。
   - 社区成员正在辩论 APO 是否仍值得深入研究。
- **Claude Code 正在挑战 DSPy？**：社区讨论了 **Claude 的代码能力**是否可能取代 **DSPy**，并引用了[一条推文](https://x.com/jxnlco/status/1941322807729848429)作为可能的证据。
   - 一位成员表示有兴趣了解其底层机制，特别是如果 Claude 能够自主优化提示词，那么采用更结构化的方法可能会更有益。
- **对 Claude Code 的法律顾虑**：一些成员担心，由于代码库商业机密的原因，大型公司可能不允许使用 **Claude Code**。
   - 另一位成员反驳称，他们所在的科技巨头自发布以来一直在使用 **Claude Code**，尽管非科技公司的情况可能有所不同；不过，这里值得链接一下 [DSPy 的推文](https://x.com/DSPyOSS/status/1941702597561262519)。
- **优化 LLM 的工具选择**：一位社区成员分享了一篇关于优化 **LLM** 工具选择的[博客文章](https://viksit.substack.com/p/optimizing-tool-selection-for-llm)，询问此类功能是否可以作为 **DSPy 模块**进行端到端的原生训练。
   - 社区还询问了关于使用 **DSPy 3.0 版本**的情况。
- **DSPy 3.0 DAIS 演讲**：一位成员分享了他们在 **DAIS** 上关于 **DSPy 3.0 (beta)** 演讲的链接，可在 [YouTube](https://x.com/DSPyOSS/status/1942318595633017116) 上观看。
   - 会上还提到，相关视频可能会在一周内公开。


  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1391788364985925642)** (6 条消息): 

> `DSPy quickstart, A/B testing prompts in DSPy, signature.py exploration` 


- **Quickstart 避免使用 Prompt Docstrings**：一位成员分享了一个 [quickstart](https://x.com/hammer_mt/status/1942148631483523518)，该方法避免了使用 Prompt Docstrings 和类模块，同时也可以在 [gist](https://gist.github.com/hammer-mt/a1288d8f3a10a8620d35183e6ee8560d) 中查看。
   - 另一位成员认为这是一种*更简洁的方式*，特别是对于从非 DSPy 工作流迁移过来的较长初始 Prompt 而言。
- **A/B 测试 Prompt 被视为不当做法**：一位成员提到，尽管在 DSPy 中这是一个*大忌*，但他们经常使用这种方法来 **A/B 测试 Prompt**。
   - 他们通过探索 **signature.py** 发现了这种方法。


  

---


### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1390784470524625108)** (32 条消息🔥): 

> `ChatGPT personality, Claude 4 Comparison, Manus Airdrops, Manus Credits, Manus as project starter` 


- **Manus 被误认为表现出 ChatGPT 的个性**：一位成员询问 **Manus** 是否使用了 **ChatGPT**，因为它表现出的个性有所增加且使用了表情符号。
   - 另一位成员建议它可能是 **Claude 4**，并指出它表现出了类似的行为。
- **注意到与 Claude 4 的相似之处**：一位成员将 **Manus** 与 **Claude 4** 进行了比较，认为它们表现出相似的行为。
   - 该成员表示 *它的表现如出一辙*。
- **对 Manus 空投能力表示怀疑**：一位成员询问 **Manus AI** 是否可以发起空投（Airdrop）。
   - 另一位成员回答 *不行*。
- **修复错误耗尽积分引发的 Token 苦恼**：一位用户抱怨项目中的错误耗尽了所有 **3000 积分**，导致他们无法启动任何项目。
   - 他们建议更改 **积分请求参数**，以便项目能够完成并启动。
- **建议将 Manus 作为项目启动器而非交付成品项目**：一位成员建议使用 **VS Code** 和其他工具，认为 **Manus** 更擅长启动项目，而不是交付可运行的成品项目。
   - 另一位成员报告在使用 **Manus** 时遇到了 **Network connection error**（网络连接错误）。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1391662303887360051)** (3 条消息): 

> `QLoRA throughput, Fused RMSNorm, Linear cross entropy, LoRAMLP` 


- **QLoRA 吞吐量大比拼：Tune vs. Unsloth**：由于 PyTorch 可能将 [fused RMSNorm](https://github.com/pytorch/pytorch/pull/153666) 和线性交叉熵（linear cross entropy）引入上游，成员们询问了近期 **Tune** 和 **Unsloth** 在 **QLoRA** 吞吐量和显存占用方面的对比。
   - 他们认为，除了 Unsloth 中的融合 [LoRAMLP](https://github.com/unslothai/unsloth/blob/main/unsloth/kernels/fast_lora.py) 内核外，这些特性可以使 **Torchtune** 在 **LoRA** 优化方面更接近 **Unsloth**。
- **仅限 PyTorch 的 LoRA 实现是否正在缩小差距？**：成员们对随着每个新特性的加入，仅限 PyTorch（无自定义 Triton）的实现与 **Unsloth** 的差距能缩小到何种程度表示好奇。
   - 一位成员回忆起去年的 *compile* 对比显示，性能已经 *非常接近*，在某些情况下甚至更好。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1390785775888760993)** (5 条消息): 

> `Custom Tokenizers, Mistral 2506-small, HF AutoTokenizer, CI Bug` 


- **自定义 Tokenizer 值得维护**：成员们讨论了是否要维护自定义 Tokenizer，尤其是因为 **Mistral** 在其最新的 **2506-small 模型**中没有提供 **HF tokenizer**。
   - 一位成员建议建立一个回退方案（fallback），使用 **HF `AutoTokenizer`** 并与 packing 等特性集成。
- **Mistral 需要额外的库**：一位成员指出 **Mistral** 需要一个额外的库，认为其 Tokenizer 方案“令人遗憾”。
   - 该成员建议使用 HF 的 AutoTokenizer 作为备选方案，并想知道是否可以使其与 torchtune 的特性兼容。
- **CI Bug 警报**：一位成员报告了 CI 中的一个新 Bug ([https://github.com/pytorch/torchtune/pull/2822](https://github.com/pytorch/torchtune/pull/2822))。
   - 另一位成员确认了该 Bug，称其为“非常轻微”的问题，并承诺在次日修复。


  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1390822757893148722)** (14 messages🔥): 

> `Context-Parallel vs. Limited GPUs, Skepticism on Architectural Improvements, Data Cleaning vs. Architecture Iterations, MoE Training Techniques` 


- **Context-Parallel 扩展 vs. GPU 限制**：一位成员讨论了他们的 [preprint](https://doi.org/10.21203/rs.3.rs-7029913/v1)，该论文解决了在有限 GPU 情况下扩展序列长度的问题，并将其与需要更多 GPU 才能获得吞吐量的 Context-Parallel 方法进行了对比。
   - 他们建议，对于序列长度 >= **500k** 的情况，Context-Parallel 可能更好，但对于极高序列长度（数百万级），他们的方法可能更优，并强调需要进行 Benchmark 对比。
- **架构改进面临质疑**：一位成员对近期的架构改进表示怀疑，认为许多报告的结果是由方差和超参数优化驱动的，而非根本性的进步。
   - 他们认为，自第一个 Transformer 以来，投资于数据清洗比追求架构迭代更有效，主张使用更干净的数据集而非复杂的模型修改。
- **清洁数据胜过架构**：一位成员普遍同意 *数据清洗* 带来的能力提升可以超过现代 LLM 所取得的成就。
   - 他提到自己曾对 **SSM papers** 充满热情，并一直关注到 **Mamba 2**，但随后 *这种热情基本消退了，所以我的热情可能放错了地方*。
- **MoE 训练技术分析**：文中提到了一种 **MoE 训练** 的技术和结果，并质疑是否可以在不需要稠密前向传递（dense fwd pass）的情况下更廉价地实现类似结果，详见这篇 [Notion post](https://fengyao.notion.site/moe-posttraining)。
   - 另一位成员分享了一张 [图片](https://cdn.discordapp.com/attachments/1293438210097025085/1391882256649290039/IMG_20250707_234251_274.jpg?ex=686d82dd&is=686c315d&hm=1d54ba8bc5b3e6e437f6f87a3c59a0a66e18b08b9b73675635c7bb86e28b2c42&)，指出 **线性缩放（linear scaling）存在许多权衡**。


  

---


### **Cohere ▷ #[🧵-general-thread](https://discord.com/channels/954421988141711382/954421988783444043/1390814978281570554)** (4 messages): 

> `Cohere Labs, Open Science Community Summer School` 


- **Cohere 暑期学校内容现已发布**：Cohere Labs 开源科学社区暑期学校的内容现已在 [YouTube 播放列表](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR) 中上线。
- **Cohere 入门指南**：新成员可以在 [YouTube](https://youtube.com/playlist?list=PLLalUvky4CLK3oT1DKNPagd_lTXooYVlR) 上找到来自 Cohere Labs 开源科学社区暑期学校的资源和内容。


  

---


### **Cohere ▷ #[🔌-api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1391700755215417364)** (2 messages): 

> `Embed v4 API, Hybrid image/text embeddings` 


- **用于图像/文本查询的 Embed v4 API**：一位成员询问了使用 **Embed v4 API** 进行 **混合图像/文本嵌入（hybrid image/text embeddings）** 的正确方法。
   - 另一位成员建议在使用 `embed-v4.0` 模型时，将 `input_type` 设置为 `search_query` 或 `document`。
- **Embed v4 输入类型**：在 Embed v4 中指定输入类型的正确方法是使用 `search_query` 或 `document`。
   - 这对于混合图像/文本嵌入的正常运行至关重要。


  

---


### **Cohere ▷ #[👋-introduce-yourself](https://discord.com/channels/954421988141711382/1346635816629178410/1390807744549748900)** (10 messages🔥): 

> `Self-learning RL/DL, Agentic AI, Cohere Embed v4, Probabilistic Generative Models` 


- **机器人专业学生青睐 Cohere 蒙特利尔**：一位专注于 **自学习 RL/DL** 的机器人专业博士生表达了对 Cohere 的喜爱，并希望加入其蒙特利尔办公室。
   - 他们正在寻找志同道合的人一起讨论和学习。
- **技术爱好者进入 AI Agent 领域**：一位 **Agentic AI** 和开发领域的新人渴望开始学习、建立人脉并进行创作。
   - 他们很高兴能将 Cohere 应用到自己的项目中并贡献给世界。
- **研究员出色地实现了 Embed V4**：一位热衷于技术的研究员最近使用 **Cohere Embed v4** 进行了多模态检索。
   - 他们分享了一个展示其工作的 [YouTube 视频](https://www.youtube.com/watch?v=TJ9jvYSZwhc)。
- **概率生成模型研究**：一位来自德国的数学与计算机专业学生正在研究 **概率生成模型（probabilistic generative models）** 和学习采样器。
   - 他们对理论和应用方面都感兴趣，特别是可解释性（interpretability），并希望在该领域学习更多知识。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1390772023931834449)** (5 messages): 

> `开源 NotebookLM, LlamaCloud MCP 服务器, AI Hack Night` 


- **开源 NotebookLM 问世**：一个完全**开源的 NotebookLM** 现已可用，允许用户在自己的电脑上运行以进行本地文档知识提取，根据[此推文](https://t.co/TNzqzF77yQ)所述。
- **LlamaIndex 办公时间重点讨论 MCP**：LlamaIndex 在 7 月 8 日举行了办公时间活动，讨论了 **LlamaCloud MCP 服务器**、在 **Agent Workflows** 中使用现有的 **MCP 工具**，以及将 Agent 工作流作为 **MCP** 提供服务，如[此推文](https://t.co/LPCc71sguM)所述。
   - 他们涵盖了扩展与 **Multi-Cloud Processing (MCP)** 相关功能的内容。
- **AI Hack Night 将聚焦前沿应用**：在 GitHub 举办的 **AI Hack Night** 将专注于使用 **AI Agents**、**MCPs** 和 **RAG** (Retrieval-Augmented Generation) 技术构建前沿应用，已由[此推文](https://t.co/AhBsvYjnRx)确认。
- **演示结构化数据提取工作流**：一个演示 Notebook 展示了如何构建具有 Human-in-the-loop 验证的结构化数据提取工作流，利用 LLMs 进行数据预处理，为批量提取工作创建 Schema，参考[此推文](https://t.co/DLPtrEVKca)。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1391288126030483609)** (9 messages🔥): 

> `P&ID 文档智能, 手写文本提取, 为业务用户提供的 LlamaIndex UX` 


- **P&ID 文档智能势头强劲**：一位成员正在开发针对 **Piping & Instrumentation Diagrams (P&IDs)** 的文档智能，这些是具有重叠符号、微小文字和复杂线条的复杂工程蓝图。
   - 他们正在寻求关于处理复杂技术图表（如电气原理图）的建议、密集内容的性能基准，以及关于使用 **LlamaIndex** 进行关系推理的混合方法的想法。
- **开始寻找手写文本提取工具**：一位成员正在寻找能够原样提取图像中手写文本的工具，并分享了[示例图像](https://cdn.discordapp.com/attachments/1059201661417037995/1391473392837988562/CamScanner_07-05-2025_20.45_12.jpg?ex=686d5795&is=686c0615&hm=949a322e62cf127e2b9ef25975fd8bba3f7da0c629da0e283fb02c5ecc2e0da3&)。
- **为业务人员寻找 LlamaIndex UX**：一位成员正在寻找一个用户友好的 UX，用于在 **LlamaIndex** 中管理文档和索引，旨在让业务用户能够更新公司的知识库，而无需深入了解 Agent 的运行。
   - 他们设想了一个用于上传和组织文档的中心位置，由开发人员创建利用这些索引的 AI Agents；到目前为止，**Simba** 是他们发现的唯一选择。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1390788864989462608)** (9 messages🔥): 

> `Nomic API, GPT4All 服务器模式, Jinja 聊天模板, CrowdLLM, OpenAssistant` 


- **新成员询问 Nomic API**：一位新成员询问了关于使用 **Nomic API** 以及在本地以服务器模式运行 **GPT4All** 的问题。
   - 他们特别询问了是否存在可访问的本地端点。
- **用户在 Jinja 聊天模板上遇到困难**：一位成员请求协助处理 **Jinja 格式** 的聊天模板。
   - 另一位成员暗示该用户可能使用的是特殊模型，并提出协助理解 **System Prompt**。
- **介绍 CrowdLLM 数据集构建工具**：一位成员分享了一个名为 [CrowdLLM](https://crowdllm.ct.ws) 的工具，用于为 LLMs **众包构建数据集**。
   - 该工具允许用户创建任务，并通过添加 Prompt-Answer 对、Prompt 或 Answer 来做出贡献。
- **CrowdLLM 与 OpenAssistant 的对比**：一位成员将新介绍的 [CrowdLLM](https://crowdllm.ct.ws) 与 2023 年的 **OpenAssistant** 进行了对比。
   - 另一位成员指出，**OpenAssistant** 必须首先对其系统可用。


  

---


### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1391376875233738772)** (3 messages): 

> `免费书籍 PDF, 作者博客, 版权问题` 


- **用户直接向作者索要免费书籍 PDF**：一位用户在听完 1 小时的演讲后，以资金匮乏为由，直接向作者索要其书籍的免费 PDF 副本。
   - 另一位用户评论说这种索要行为*有失体面*，建议查看作者的博客以获取摘录。
- **通过作者博客获取内容的替代方案**：一位用户建议，与其索要免费 PDF，不如查看作者的博客。
   - 据称，该博客包含作者书籍的摘录和独立文章，提供了无需直接购买即可访问内容的替代方式。