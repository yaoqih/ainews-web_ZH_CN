---
companies:
- anthropic
- openai
- google
- sakana-ai
- cursor
- baseten
- epoch-ai-research
- deepmind
date: '2026-01-22T05:44:39.731046Z'
description: '**Anthropic** 推出了功能增强的“Claude in Excel Pro”。**OpenAI** 揭晓了即将推出的 **Codex**
  智能体循环（agent loop）及网络安全措施。**Google** 提高了 **Gemini App** 的使用配额，并与 **Sakana AI** 合作，在日本开展高级
  AI 科学家项目。**Cursor** 引入了“智能体技能”（Agent Skills）以实现动态上下文聚焦。**GPT-5.2 Pro** 在 FrontierMath
  Tier 4 测试中达到了 **31%**，展示了显著的基准测试进展。**Baseten** 以 **50 亿美元估值**融资 **3 亿美元**，专注于高性能推理。


  相关讨论强调了将数学基准作为 AI 能力的指标、AGI 进展的不平衡性，以及推理和持续学习作为未来前沿领域的重要性。提及的知名人物包括 *Sam Altman*、*François
  Chollet*、*Shane Legg* 和 *Demis Hassabis*。'
id: MjAyNi0w
models:
- claude-3
- codex
- gemini
- gpt-5.2-pro
people:
- sama
- fchollet
- shane_legg
- demishassabis
title: 今天没发生什么特别的事。
topics:
- benchmarking
- reasoning
- continual-learning
- reinforcement-learning
- model-performance
- agentic-ai
- security
- model-training
---

**平静的一天**

> 2026年1月22日至1月23日的 AI 新闻。我们为您检查了 12 个 subreddits、[**544** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **24** 个 Discord 社区（包含 **206** 个频道和 **7161** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**579 分钟**。**我们的新网站**现已上线，支持全元数据搜索，并以精美的氛围感设计呈现过往所有内容。请访问 https://news.smol.ai/ 查看完整的新闻分析，并在 [@smol_ai](https://x.com/Smol_AI) 上向我们提供反馈！

---

# AI Twitter 回顾


**热门推文（按互动率排序）**

- **Anthropic 发布 “Claude in Excel”**：Claude in Excel 扩展至 Pro 用户，支持多文件拖放、更安全的单元格写入，并通过自动压缩（auto-compaction）实现更长的会话保持 ([claudeai](https://twitter.com/claudeai/status/2014834616889475508))。关于 Microsoft 365 Copilot 进度滞后的讨论引起了大量关注 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014835455393726726))。
- **OpenAI 路线图 + Agent 循环**：Sam Altman 表示 Codex 的发布即将到来，OpenAI 正接近“网络安全高（Cybersecurity High）”级别，将伴随相应的限制，随后是“防御性加速” ([sama](https://twitter.com/sama/status/2014733975755817267))。OpenAI 发布了关于 **Codex Agent 循环 / 框架编排（harness orchestration）** 的技术深度解析 ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2014794871962533970))。
- **Google AI Ultra 额度提升**：Gemini App 为 Ultra 会员增加了每日配额，达到每日 **1,500 次 Thinking** + **500 次 Pro** 提示词 ([joshwoodward](https://twitter.com/joshwoodward/status/2014566936479437173))。
- **Sakana AI 与 Google 达成合作伙伴关系并获投资**：Sakana 宣布与 Google 建立战略合作伙伴关系并获得其注资，旨在将 **Gemini/Gemma** 与 Sakana 的 “AI Scientist” / “ALE-Agent” 研究相结合，并部署在 日本的高安全性领域 ([SakanaAILabs](https://twitter.com/SakanaAILabs/status/2014686043711406355), [hardmaru](https://twitter.com/hardmaru/status/2014686852691918971), [JeffDean](https://twitter.com/JeffDean/status/2014716109216448975))。
- **Cursor 发布 Agent Skills**：为 Agent 打造的一等公民“技能（Skills）”，强调发现能力与动态上下文聚焦 ([cursor_ai](https://twitter.com/cursor_ai/status/2014753596223770841))。
- **FrontierMath 分数飞跃**：**GPT-5.2 Pro 在 FrontierMath Tier 4 上达到 31%**，高于此前 19% 的历史最佳成绩 ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014769359747744200))。从业者强调了其有用性，甚至在基准测试中进行了问题识别（issue-spotting） ([gdb](https://twitter.com/gdb/status/2014859263701839963))。
- **Claude Code “本地免费运行” 教程**：一个热门教程声称可以通过开源模型在本地运行类似 Claude Code 的工作流，支持私有化且具备工具调用功能 ([dr_cintas](https://twitter.com/dr_cintas/status/2014771670070747278))。
- **Baseten 以 50 亿美元估值融资 3 亿美元**，定位围绕“多模型未来”和高性能推理 ([basetenco](https://twitter.com/basetenco/status/2014755013344792595), [tuhinone](https://twitter.com/tuhinone/status/2014755252244005273))。

---

**前沿模型、基准测试与“能力”叙事**

- **数学作为领先指标 (FrontierMath + 跨基准相关性)**：Epoch 报告 **GPT-5.2 Pro = 31%**，在 FrontierMath Tier 4 上（声称无过拟合），相比之前的 19% 有显著提升 ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014769359747744200))。Epoch 的另一项分析指出，不同领域的基准测试分数具有强相关性（跨领域约 **0.68**，领域内约 **0.79**），这暗示了在“数学/代码/推理”进展背后的潜在能力因子 ([EpochAIResearch](https://twitter.com/EpochAIResearch/status/2014806095504785664))。从业者注意到了具体的价值：能够发现题目缺陷/拼写错误，甚至“指出一个 Tier 4 题目中的漏洞” ([gdb](https://twitter.com/gdb/status/2014859263701839963), [GregHBurnham](https://twitter.com/GregHBurnham/status/2014774878591655984))。
- **AGI 时间线与产品现实**：一个反复出现的主题是“系统发展不均衡”：在正式领域表现聪明，但在其他地方不可靠。一段广为流传的俏皮话捕捉到了这种错位（“比数学博士更聪明，比实习生更笨”）([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014564105194242452))。François Chollet 强调进展是**特定垂直领域**的（特别是代码等可验证领域），因为无限的合成数据使得在那里更容易进行记忆/操作化，并警告不要将其外推到所有人类任务 ([fchollet](https://twitter.com/fchollet/status/2014821042464948270))。
- **推理 + 持续学习是“真正的”前沿**：来自采访的报道称 Shane Legg 认为到 2028 年有 **50% 的概率实现“初级 AGI”**，Google 的定义包括持续学习/记忆/世界模型 ([kimmonismus](https://twitter.com/kimmonismus/status/2014697026890416586))。后续笔记提到 Demis Hassabis 明确表示 DeepMind **尚未解决持续学习问题**，目前正在探索类 AlphaZero 方法与基础模型的结合 ([Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014785682309579119), [Hangsiin](https://twitter.com/Hangsiin/status/2014774897680253442))。
- **模型/架构讨论：MoE 起源之争**：一个推文线程反驳了 DeepSeek MoE “基于 Mixtral 构建”的说法，理由是 DeepSeek MoE 论文在 Mixtral 的 arXiv 发布后几乎立即出现，Mixtral 的训练细节稀少，且 DeepSeek 的 MoE 在架构上不同/更稀疏，并引用了 **GShard** 而非 Mixtral ([eliebakouch](https://twitter.com/eliebakouch/status/2014575628675092845))。另一种说法将 DeepSeek 称为独特的“neoMoE”分支，以区别于“oldMoE” ([kalomaze](https://twitter.com/kalomaze/status/2014659449219383367))。
- **第二梯队多模态更新（中国）**：一份详尽的中文评论将 **百度文心一言 (ERNIE) 5.0** 定位为有所改进且稳定，但成本依然高昂（2T 参数，约 61K 上下文），且与拥有庞大算力预算的顶级多模态系统相比，仍“稳居第二梯队” ([ZhihuFrontier](https://twitter.com/ZhihuFrontier/status/2014606592826912840))。

---

**Agent 与代码：从工作流 (workflows) → 框架 (harnesses) → 技能 (skills)**

- **OpenAI Codex “agent loop” 变得显性化**：OpenAI 发布了 Codex 如何编排轮次：组装输入 → 执行推理 → 执行工具 → 将结果反馈回模型直至停止，即将 Agent Harness 作为系统的一等公民组件 ([OpenAIDevs](https://twitter.com/OpenAIDevs/status/2014794871962533970))。这与更广泛的评论一致，即“训练更好的模型只是一个维度；Harness + 实验可能会带来惊喜” ([Hangsiin](https://twitter.com/Hangsiin/status/2014794375466033657))。
- **Workflows 与 Agent 正在融合为 “Skills / Guidance / RLMs”**：一篇强有力的技术综述认为，Agent 与 Workflow 的界限不再仅仅是“代码中的控制流”，而更多关乎**状态表示**、**动态指令选择**以及**谁来主导组合**——Replit 的 “Decision-Time Guidance”、Skills 以及 **Recursive Language Models (RLMs)** 是这一设计光谱上的混合点 ([lateinteraction](https://twitter.com/lateinteraction/status/2014685012994515206))。DSPy 社区的帖子将 RLMs 推崇为处理“任意长度 Prompt”的实用方法，即通过委托给代码 + 子调用（subcalls）而非摘要损失（summarization loss）来实现 ([getpy](https://twitter.com/getpy/status/2014717862246756384))。
- **Cursor：推出 Agent Skills**：Cursor 引入了 **Skills**，将其作为可发现的专业化 Prompt/代码，并宣称它们还能通过动态发现来改善上下文聚焦 ([cursor_ai](https://twitter.com/cursor_ai/status/2014753596223770841), [cursor_ai](https://twitter.com/cursor_ai/status/2014753597624598665))。这也呼应了更广泛的市场趋势：“通过将其称为 Skills，让非开发人员也能编写代码” ([kylebrussell](https://twitter.com/kylebrussell/status/2014689618617122883))。
- **Claude Code 生态持续扩张（以及功能抄袭战）**：多篇帖子强调了工具之间快速的功能扩散（“Cursor 正在吸收 Claude Code 的热门功能”） ([dejavucoder](https://twitter.com/dejavucoder/status/2014635509025526198))。实用片段：Claude 任务存储在文件系统中（`~/.claude/tasks`），通过广播实现多会话/子 Agent 协作 ([dejavucoder](https://twitter.com/dejavucoder/status/2014584272183861407))。与此同时，痛点依然存在（例如，通过 base64 进行荒谬的文件下载 Hack） ([dbreunig](https://twitter.com/dbreunig/status/2014540341526069738))。
- **安全态势正在成为核心功能**：Sam Altman 表示，OpenAI 将越来越多地限制用于网络犯罪的代码模型，随后转向**防御性加速**（帮助修复漏洞）作为缓解措施 ([sama](https://twitter.com/sama/status/2014733975755817267))。一则轶事指出一个潜在的安全隐患：Codex Slack 集成生成的任务链接在无痕模式下无需授权即可访问（如果属实，这是一个紧急的产品安全问题） ([apsdehal](https://twitter.com/apsdehal/status/2014770563810758938))。
- **企业级 “Agent 在生产环境失败” 的提醒**：一篇长文声称 **95% 的企业级 AI 试点都失败了**（引用自 MIT 研究），强调生产环境的可行性取决于**权限感知的检索（authorization-aware retrieval）**、**护栏（guardrails）**、**监控**和**可审计性**，而非 Demo 演示能力 ([victorialslocum](https://twitter.com/victorialslocum/status/2014654495301525683))。

---

**推理 + 系统：vLLM, KV 压缩, 存储和基础架构成熟度**

- **vLLM 持续成为开放推理的“底座”**：vLLM 将自己定位为连接开放模型与可部署推理的桥梁，并重点推介了 vLLM Studio 工作流 ([vllm_project](https://twitter.com/vllm_project/status/2014536660361584833))。一篇备受关注的基础设施工程文章记录了一个复杂的 **vLLM 内存泄漏** 调试过程（Python profilers → pmap → BPFtrace → GDB），最终追溯到 **UCX mmap 钩子**；修复补丁已合并至上游 ([vllm_project](https://twitter.com/vllm_project/status/2014630499231412477))。
- **系统智能 / 路由**：vLLM 宣布在 AMD 平台上开启 **vLLM-SR** (Semantic Router) 的公开测试，将其定义为一种“系统智能”方法，而非由单体模型承包所有任务 ([XunzhuoLiu](https://twitter.com/XunzhuoLiu/status/2014672307407704279))。
- **通过蒸馏实现 KV cache 压缩**：NVIDIA Research 发布了 **Qwen3-8B-DMS-8x**，声称只需约 1K 步微调即可实现 **8 倍 KV cache 压缩** 且开销极小，性能优于基于 token 重要性的逐出代理；该方法兼容稀疏注意力（sparse attention）方法 ([p_nawrot](https://twitter.com/p_nawrot/status/2014770473289019709))。
- **可预测部署的工具**：`hf-mem` 能够通过 Safetensors 元数据估算推理所需的 VRAM，无需下载权重，旨在消除“尝试-OOM（显存溢出）”的循环 ([LiorOnAI](https://twitter.com/LiorOnAI/status/2014730309128855801))。
- **存储与数据平面关注点**：SkyPilot 推出了用于高性能存储（AI 权重检查点/数据）的 “Volumes”，因为对象存储并不总是适用 ([skypilot_org](https://twitter.com/skypilot_org/status/2014752751545381044))。Jina 提出了一种巧妙的压缩技巧：在压缩前将 embeddings 转换为球坐标，声称在低于 float32 epsilon 的精度下可实现近乎无损的重建，并节省约 1.5 倍的存储空间 ([JinaAI_](https://twitter.com/JinaAI_/status/2014753001387499927))。
- **针对 Agent 的 GPU kernel 评估**：AMD AGI 发布了 **Magpie**，这是一个针对 AMD/NVIDIA 平台的正确性与性能评估的开源 kernel 评测套件，专为 Agent 工作流设计；声称与单独使用 GPU profilers 相比，其 **token 效率提高了 3000 倍**，并计划与 SGLang/vLLM 集成追踪功能 ([realSharonZhou](https://twitter.com/realSharonZhou/status/2014722290865549649))。MLSys 2026 启动了 FlashInfer-Bench 竞赛赛道（MoE/DSA/GDN），包含独立的人类与 Agent 评估维度 ([ye_combinator](https://twitter.com/ye_combinator/status/2014836302198472789))。

---

**生态系统 + 商业：合作伙伴关系、定价及“价值共享”辩论**

- **Sakana AI ↔ Google：战略合作 + 融资（以及争议）**：Sakana 将该交易描述为结合 Google 的基础设施/模型（Gemini/Gemma）与 Sakana 的研究自动化能力（AI Scientist, ALE-Agent），并在需要安全和数据主权的关键任务领域推动部署 ([SakanaAILabs](https://twitter.com/SakanaAILabs/status/2014686043711406355))。媒体对此进行了广泛报道（日经、彭博等）([nikkei](https://twitter.com/nikkei/status/2014637546563658172), [business](https://twitter.com/business/status/2014594583234027753))。随后出现了一场争端：一种说法称这只是 Google Cloud Japan 的一笔小型计算交易且“DeepMind 未参与” ([shaneguML](https://twitter.com/shaneguML/status/2014847946110783649))，而 Sakana 领导层公开反驳称 DeepMind 确实参与其中，并艾特了 Demis 和 Jeff Dean ([hardmaru](https://twitter.com/hardmaru/status/2014885853789884416))。
- **Baseten 的“多模型未来” + 融资**：Baseten 以 **50 亿美元估值融资 3 亿美元**，并认为推理是实现数百万专用模型和可靠低延迟 UX 的瓶颈 ([basetenco](https://twitter.com/basetenco/status/2014755013344792595), [tuhinone](https://twitter.com/tuhinone/status/2014755252244005273))。
- **Anthropic 经济学：推理成本压力**：一份报告称，由于推理成本比预期高出 **23%**，尽管预计年收入将达到 **45 亿美元**（同比增长约 12 倍），Anthropic 仍将 2025 年毛利率预期下调至 **40%** ([kimmonismus](https://twitter.com/kimmonismus/status/2014673235594641838))。
- **AI 驱动发现的“价值共享”模式**：报道称 OpenAI 的 CFO 讨论了从客户利润/知识产权中抽成的交易模式（从药物研发开始），类似于 Isomorphic Labs 的模式 ([kimmonismus](https://twitter.com/kimmonismus/status/2014643034089259103))。一些人对此类炒作和激励机制表示反对（例如，你不能既卖 token 又拥有发现成果，除非你同时也承担了计算成本） ([code_star](https://twitter.com/code_star/status/2014541663356772516), [paul_cal](https://twitter.com/paul_cal/status/2014692633730261339))。

---

**多模态 + 语音 + 视频：质量飞跃与工具链**

- **语音技术正在加速（开源 + 低延迟）**：Teknium 声称一个开源语音克隆 HF 演示是他们在开源模型中见过的最接近 ElevenLabs 质量的作品 ([Teknium](https://twitter.com/Teknium/status/2014687269329031253))。NVIDIA 发布了 **PersonaPlex**，这是一个针对极低延迟优化的开源实时全双工对话语音技术栈 ([kimmonismus](https://twitter.com/kimmonismus/status/2014703479491854751))。
- **视频生成：可控性与竞技场**：Runway Gen-4.5 I2V 增加了更精确的“缩放到指定区域”控制 ([c_valenzuelab](https://twitter.com/c_valenzuelab/status/2014674372120785176))，创作者们展示了短片工作流 ([Artedeingenio](https://twitter.com/Artedeingenio/status/2014693398502842731))。LMSYS Arena 推出/扩展了 **Video Arena** 排行榜（Veo, Sora 2, 可灵, 海螺等） ([arena](https://twitter.com/arena/status/2014815916056576257))。
- **用于交互式环境的 3D Agent 和世界模型**：Berkeley 演示的 “VIGA” 声称是一个多模态 Agent，无需训练即可从图像生成 3D/4D Blender 场景 ([HavenFeng](https://twitter.com/HavenFeng/status/2014765400563781777))。HF 上出现了一个较小的“可玩的世界模型”演示（Waypoint-1-Small, 2.3B） ([victormustar](https://twitter.com/victormustar/status/2014766391811826022))。此外，“世界模型是游戏/机器人领域的下一波大趋势”的情绪再次浮现 ([kylebrussell](https://twitter.com/kylebrussell/status/2014529425983914098))。

---

**AI 社交层面的安全、信任与诚信问题**

- **针对 AI 业内人士的账号攻破**：多个警告显示，一些知名账号（Deedy Das；一位 Kimi 研究员/“Crystal”）被盗并被用于网络钓鱼/诈骗，很可能是由加密货币利益驱动的 ([cloneofsimo](https://twitter.com/cloneofsimo/status/2014536638010163262), [ml_angelopoulos](https://twitter.com/ml_angelopoulos/status/2014543018137944486), [Kimi_Moonshot](https://twitter.com/Kimi_Moonshot/status/2014571513299796154), [Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/2014572557270450194))。
- **虚假信息 / 假论文**：一篇虚假的 “Llama 4” arXiv 论文被标记为并非由 Meta 撰写 ([TimDarcet](https://twitter.com/TimDarcet/status/2014626676798366006))。
- **开源“层级”构想**：一种实用的分类法区分了**开源代码** vs **开源权重** vs **开源训练流水线**（数据 + 配方 + 可复现性），认为团队必须决定他们真正需要哪一层 ([TheTuringPost](https://twitter.com/TheTuringPost/status/2014630341349408928))。


---

# AI Reddit 总结

## /r/LocalLlama + /r/localLLM 总结

### 1. Qwen3-TTS 模型发布与讨论

  - **[Qwen 已开源 Qwen3-TTS 全系列：VoiceDesign、CustomVoice 和 Base，共 5 个模型（0.6B 和 1.8B），支持 10 种语言](https://www.reddit.com/r/LocalLLaMA/comments/1qjul5t/qwen_have_opensourced_the_full_family_of_qwen3tts/)** (热度: 880): **Qwen** 已开源 **Qwen3-TTS** 全系列模型，包括 VoiceDesign、CustomVoice 和 Base 模型，参数量分别为 `0.6B` 和 `1.8B`。这些模型支持 `10 种语言`，专为声音克隆（Voice Clone）、声音设计（Voice Design）和自定义语音（Custom Voice）等任务设计。配图提供了这些模型与 MiniMax、SeedTTS 等其他模型的对比图表，突出了它们在各项指标上的表现，数值越低代表性能越好。模型已在 [GitHub](https://github.com/QwenLM/Qwen3-TTS) 和 [Hugging Face](https://huggingface.co/collections/Qwen/qwen3-tts) 上线，并附有详细介绍其能力的 [博客文章](https://qwen.ai/blog?id=qwen3tts-0115) 和 [论文](https://github.com/QwenLM/Qwen3-TTS/blob/main/assets/Qwen3_TTS.pdf)。评论者对开源发布表示赞赏，但也对模型对 Python 和 Nvidia GPU 的依赖表示担忧，建议支持 llama.cpp 或 mistral.rs 等其他语言和平台，以实现更广泛的普及。

    - **FullstackSensei** 提出了关于目前运行 Qwen 模型局限性的技术担忧，强调需要 `llama.cpp` 或 `mistral.rs` 等环境的支持，以便利用除 CUDA 之外的 GPU 推理。鉴于硬件成本上升以及对 Python 和 Nvidia GPU 之外更易用的部署选项的需求，这一点尤为重要。
    - **LetterRip** 对 Qwen3-TTS 的英语语音输出进行了评价，指出其似乎受到了日本动漫配音的影响。这表明训练数据中可能存在潜在偏见，如果训练集不够多样化，可能会影响生成的英语声音的自然度和真实感。
    - **silenceimpaired** 讨论了 Qwen3-TTS 的表现，指出虽然样本令人印象深刻，但对某些输出的频率感到担忧。这意味着虽然模型可以产生高质量的音频，但可能存在一致性问题，需要解决以确保在不同用例下的可靠表现。

  - **[Qwen 开发者在 Twitter 发布动态！！](https://www.reddit.com/r/LocalLLaMA/comments/1qjtyw8/qwen_dev_on_twitter/)** (热度: 833): **图片是 **Chen Cheng** 的一条 Twitter 帖子，宣布了一款新模型，口号是“小模型，大个性”（Tiny model. Big personality），并带有倒计时，预示即将发布。评论猜测这可能与之前 vLLM 泄露的 TTS（Text-to-Speech）模型有关，并附带了可能相关的 [Hugging Face 集合](https://huggingface.co/collections/Qwen/qwen3-tts) 链接。这暗示了 TTS 模型领域的新进展，可能提供显著的改进或功能。** 一条评论幽默地建议，新模型可能终于能让投资 `5090` 等高端 GPU 变得物有所值，表明了对模型性能的高期望。

    - **ThePixelHunter** 讨论了当前模型规模的现状，指出虽然较小的模型更适合在单个 GPU 上进行本地训练，但在 500 亿到 1200 亿参数范围（50-120B）缺乏竞争。这个范围非常适合拥有多个高端 GPU（如几块 3090 或三块 16GB 显卡）的发烧友，这表明市场上缺乏更大规模但仍可在本地训练的模型。


### 2. 本地 LLM 开发与硬件考量

  - **[我给我的本地 LLM 管线安了个大脑——现在它会先思考后说话了](https://www.reddit.com/r/LocalLLM/comments/1qkudvz/i_gave_my_local_llm_pipeline_a_brain_now_it/)** (热度: 65): **该帖子讨论了一个名为 Jarvis（即将更名为 TRION）的本地 LLM 管线的重大更新，该管线现在整合了自主开发的顺序思考 MCP（Sequential Thinking MCP，多组件处理器）。该系统由 **Ollama**、**DeepSeek-R1** 和自定义 MCP 服务器构建，允许 AI 通过将复杂问题分解为逐步推理来“出声思考”，从而显著减少幻觉。AI 会动态决定何时使用这种深度思考方法，对简单问题提供即时回答，对复杂问题提供详细推理。该项目利用了由 **u/frank_brsrk** 开发的 CIM（因果智能模块）框架。该实现已开源，可在 [GitHub](https://github.com/danny094/Jarvis/tree/main) 上获取。** 评论者赞赏该项目的开源性质，并表示有兴趣尝试。有一种观点认为，随着对集中式 AI 供应商依赖的减少，本地 LLM 将变得更加重要。

- GCoderDCoder 讨论了将本地 LLM 与 'roo code'、'vibe kanban' 以及 'MCPs' 等工具集成，以实现工作流程自动化并减少手动编码工作。他们强调了在对 AI 依赖程度日益加深的背景下，本地 LLM 的重要性，并将其与 Anthropic 等商业解决方案进行了对比。这反映了开发独立、开源 AI 解决方案以保持控制权和灵活性的大趋势。
- burn-n-die 询问了用于运行本地 LLM 流水线的系统配置。对于有兴趣复制或理解此类设置的性能和扩展性的技术读者来说，这是一个关键方面。关于硬件规格、软件环境以及任何优化细节，对于那些希望实施类似系统的人来说都非常有价值。

- **[有人在出售一台配备 4× RTX 2080 Ti => 4 x 11GB => 44GB VRAM 的 Lambda Labs 工作站。开源模型对这台机器的支持好吗？它的速度够快吗？](https://www.reddit.com/r/LocalLLM/comments/1qjlzqt/someone_is_selling_a_lamda_labs_workstation_with/)** (Activity: 107): **一台配备 4× RTX 2080 Ti GPU、总计 `44GB VRAM` 的 Lambda Labs 工作站正被考虑以 `$2000` 的价格购买。这种配置通常受到开源机器学习框架的良好支持，且 `44GB VRAM` 足以胜任大多数任务。然而，正确配置系统可能会具有挑战性。另一种建议是组建一台 2x RTX 3090 的机架，这可能会提供更好的性能，成本约为 `$2.5k`。该工作站被认为能够处理许多开源权重（open-weight）LLM 模型，特别是如果它包含至少 `32GB 的系统内存（system RAM）`。这台机器适合探索广泛的机器学习项目，尽管并非所有项目都需要大量的 VRAM。** 关于是购买现有配置还是使用 RTX 3090 等较新 GPU 组建新机器，存在争议。一些人认为现有配置足以用于学习和探索机器学习模型，而另一些人则建议组建新机架以获得更好的性能和学习体验。

    - 配备 4× RTX 2080 Ti GPU、总计 44GB VRAM 的工作站通常得到大多数开源框架的支持。但是，正确设置它可能会很困难。系统的能力足以探索许多开源权重 LLM 模型，尤其是如果它包含至少 32GB 系统内存，这增强了其处理各种机器学习任务的潜力。
    - 一个技术考虑因素是，Turing 架构的 GPU（如 RTX 2080 Ti）最初被认为与 FlashAttention 2 不兼容。然而，最近的更新表明这一限制已经得到解决，正如 [FlashAttention Triton GitHub 仓库](https://github.com/egaoharu-kensei/flash-attention-triton) 中所指出的那样。这扩展了该工作站在某些高级机器学习任务中的效用。
    - 虽然 4× RTX 2080 Ti 的配置是有能力的，但一些人建议组建一个配备 2× RTX 3090 GPU 的新系统可能会提供更好的性能和价值。RTX 3090 每张卡提供更多 VRAM 并提升了性能，对于机器学习项目来说，这可能是一项更具前瞻性的投资。

- **[OpenAI CFO 暗示“基于结果的定价”（又称对你的工作收取版税）？这让支持本地化的理由变得更加充分。](https://www.reddit.com/r/LocalLLaMA/comments/1qkiylw/openai_cfo_hinting_at_outcomebased_pricing_aka/)** (Activity: 419): **OpenAI 的 CFO Sarah Friar** 讨论了针对大型企业交易（特别是在制药等高价值行业）转向“基于结果的定价”的可能性。这种模式将涉及 OpenAI 分享其 AI 所创造的价值，例如，如果 AI 助力了重大发现，则从制药公司的利润中分成。这种方法并不针对普通用户或独立开发者，最初暗示广泛应用的报道具有误导性。这一概念引发了关于本地 AI 部署与依赖云服务的利弊讨论，并将其类比为能源领域的电网与太阳能之争。评论者对 OpenAI 潜在的定价模式表示怀疑，并将其与 AI 训练中未向数据创作者支付版税的情况进行对比。本地 AI 部署与太阳能的类比得到了认可，强调了对基础设施的控制，以避免未来产生与价值挂钩的成本。

- 讨论强调了对 OpenAI 可能转向“Outcome-Based Pricing”的担忧，这可能涉及根据使用其模型产生的收入收取版税。这与当前基于使用量付费的模型（类似于电费账单）形成了对比。这种类比暗示，随着 OpenAI 盈利能力的增长以及他们追求更高的利润，此类定价模型可能会促使用户考虑本地或自托管方案。
- WeMetOnTheMountain 的评论批评了 OpenAI 模型的效率，指出它们消耗大量的 tokens 来维持性能，从而导致处理速度变慢。评论者认为，像 GLM 或 mini Max 这样的替代模型在“one loop dialectical circuit”中实施时，可能会提供更好的结果，这表明其更倾向于更高效、可能是自托管的解决方案。
- Winter_Educator_2496 强调了对 OpenAI 模型开源替代方案的需求，这些方案可以托管在 cloud 中，但在必要时也可以切换到本地托管。这反映了用户对 AI 工具拥有更多控制权和灵活性的一种更广泛的情绪，特别是考虑到潜在的定价变化以及避免依赖单一供应商的愿望。

### 3. Hugging Face 模型发布与趋势

  - **[本周最热门 Hugging Face 发布：各类别精选！](https://www.reddit.com/r/LocalLLM/comments/1qjqhja/this_weeks_hottest_hugging_face_releases_top/)** (热度: 49): **Hugging Face** 本周在不同类别下发布了多个热门模型。在文本生成领域，拥有 `31B` 参数的 `zai-org/GLM-4.7-Flash` 模型旨在实现快速高效的文本生成，下载量已达 `124k`。其量化版本 `unsloth/GLM-4.7-Flash-GGUF` 提供了一个针对本地推理优化的 `30B` 参数模型，下载量为 `112k`。在图像/多模态类别中，`zai-org/GLM-Image` 和 `google/translategemma-4b-it` 分别因其在创意编辑和多语言任务中的能力而备受关注。在音频/语音方面，`kyutai/pocket-tts` 和 `microsoft/VibeVoice-ASR` 提供了紧凑的 TTS 和多语言 ASR 解决方案。其他值得注意的发布包括用于图像转视频生成的 `Lightricks/LTX-2` 以及用于图像-文本到文本任务中高级推理的 `stepfun-ai/Step3-VL-10B`。关于 `GLM-4.7 30B-A3B` 模型与 `Qwen3-Coder 30B-A3B` 在编程任务中的性能对比出现了一场技术辩论，部分用户认为后者表现更优。

    - 一位用户对比了 GLM-4.7 30B-A3B 模型和 Qwen3-Coder 30B-A3B 模型，指出后者在编程任务中表现更好。这表明 Qwen3-Coder 可能具有使其更适合代码相关应用的优化或架构优势。需要进一步的基准测试或详细评估来证实这一说法，并了解 Qwen3-Coder 擅长的具体领域。

  - **[适合编程的优秀本地 LLM 推荐？](https://www.reddit.com/r/LocalLLM/comments/1qk9ked/good_local_llm_for_coding/)** (热度: 62): **该用户正在寻找一个可以在拥有 `12GB` VRAM 的 `rx 6750 xt` GPU 上运行的本地编程 LLM，并考虑了像 **GLM 4.7 flash** 这样的模型。然而，对 VRAM 限制的担忧表明，即使量化到 `q4`，`30B` 参数的模型也可能超过 GPU 的容量。推荐的模型包括 [VisCoder2-7B](https://huggingface.co/TIGER-Lab/VisCoder2-7B)、[gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) 和 [NousCoder-14B](https://huggingface.co/NousResearch/NousCoder-14B)，其中 **gpt-oss-20b** 尽管受到严格审查，但仍因其速度和可靠性而受到好评。建议使用 `10B` 参数以下的模型，或者配合 `llama.cpp` 使用编程 MoE 模型，以便将部分处理任务卸载到系统 RAM。** 关于 `30B` 模型是否适合该用户的 GPU 存在争议，共识倾向于由于 VRAM 限制而使用 `10B` 参数以下的模型。使用 `llama.cpp` 卸载到系统 RAM 也被讨论为一种可行的策略。

    - Javanese1999 重点介绍了几个用于本地编程任务的模型，包括 [VisCoder2-7B](https://huggingface.co/TIGER-Lab/VisCoder2-7B)（被描述为 Qwen2.5-Coder-7B-Instruct 的增强版）和 [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b)（即使在超过 VRAM 容量时也能保持较快速度）。该评论者更倾向于使用 gpt-oss-20b，因为它在轻度编程任务中非常可靠，尽管它在拒绝提示（censorship）方面存在限制。
    - Used_Chipmunk1512 建议不要在受限的 GPU 上使用量化为 q4 的 30B 模型，认为 10B 以下的模型对大多数用户更合适。这强调了在选择本地编程 LLM 时考虑硬件限制的重要性。
    - RnRau 建议配合 `llama.cpp` 推理引擎使用编程 Mixture of Experts (MoE) 模型，将模型的部分计算卸载到系统 RAM 中，这是在不压垮 GPU 的情况下处理大型模型的一种实用方法。


## 较低技术性的 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding, /r/aivideo, /r/aivideo

### 1. OpenAI 与 Anthropic 的进展

- **[OpenAI 表示 Codex 使用量在 5 个月内增长了 20 倍，协助上个月增加了约 10 亿美元的年化 API 收入](https://www.reddit.com/r/singularity/comments/1qk6pbi/openai_says_codex_usage_grew_20_in_5_months/)** (热度: 535): **据 OpenAI 的 CFO Sarah Friar 报告，OpenAI 的 Codex 使用量在五个月内飙升了 20 倍，为年化 API 收入贡献了额外的 `10 亿美元`。公司正经历向企业客户的转型，收入分配从 `70% 消费者和 30% 企业` 转向 `60% 消费者和 40% 企业`，并预计到年底达到 `50-50` 的平衡。OpenAI 旨在通过云投资和基础设施扩展，到 2025 年实现 `200 亿美元` 的年化收入。** 一条评论对盈利能力表示怀疑，估计实现 `10 亿美元` 的收入需要 `70 亿美元` 的成本。另一条评论强调了一家金融服务公司使用的 AI 工具发生转变，表明在 B2B 市场中与 **Anthropic** 和 **OpenAI** 存在竞争。

    - BetImaginary4945 认为，OpenAI 产生 10 亿美元收入的成本可能高达 70 亿美元，这意味着在基础设施、研究和开发方面有巨大的支出以支持如此快速的增长。这引发了人们对 OpenAI 商业模式长期可持续性和盈利能力的质疑。
    - balagachchy 分享了他们在一家跨国金融服务公司的经验，注意到软件工程任务从使用 ChatGPT 转向了 Gemini 和 Claude Code。这凸显了企业级 AI 工具的竞争格局，各公司正在探索不同的解决方案以满足其特定需求。
    - imlaggingsobad 评论了 OpenAI 和 Anthropic 在 B2B 市场中的竞争动态，建议虽然 Anthropic 被视为领导者，但 OpenAI 的快速增长和创新仍可能使其成为强大的竞争对手。这强调了持续的竞争以及市场领导地位发生转变的可能性。

  - **[OpenAI CEO 会见中东投资者，商讨潜在的 500 亿美元融资](https://www.reddit.com/r/OpenAI/comments/1qjrpbq/openai_ceo_meets_middle_east_investors_over/)** (热度: 191): **据 [CNBC](https://www.cnbc.com) 证实，OpenAI 据称正在与中东主权财富基金讨论在新一轮融资中募集潜在的 `500 亿美元`。谈判仍处于初步阶段，尚未签署条款清单。OpenAI 的 CEO Sam Altman 目前正在阿联酋（UAE）参与这些讨论，凸显了这项潜在投资对 OpenAI 未来增长和运营规模的战略重要性。** 评论中的一个显著观点对 OpenAI 的财务战略表示怀疑，质疑为什么该公司在拥有可观收入的情况下不寻求 IPO，并批评其依赖外部资本来管理高昂的运营成本。

    - AtraVenator 强调了对 OpenAI 财务战略的担忧，指出尽管年收入超过 `200 亿美元`，但该公司仍在寻求额外的外部资本，而不是转向自给自足的模式。这引发了对其高昂的算力（compute）成本以及依赖外部融资来支付这些费用的质疑。
    - 讨论涉及了 OpenAI 上市的潜在风险，NotABCDinFL 建议 IPO 可能会导致“大规模收割（rug pull）”，即机构投资者可能会套现，从而使散户投资者处于不利地位。这反映了对 OpenAI 财务实践的稳定性和透明度的担忧。
    - 有人对 OpenAI 的领导层和战略方向表示怀疑，BeingComfortablyDumb 质疑公司是如何从拥有先发优势和显著市场份额转变到目前面临财务挑战的。这暗示了对管理层在 AI 行业早期领先地位下变现能力的批评。

  - **[Anthropic 的 Claude Constitution 令人感到超现实](https://www.reddit.com/r/OpenAI/comments/1qjytb2/anthropics_claude_constitution_is_surreal/)** (热度: 611): **图片讨论了对 Anthropic 的 AI —— Claude 使用代词“它（it）”的问题，以及 Claude 可能会根据人类生成数据的训练，对不同代词产生偏好，暗示了情绪或感受的功能性版本的出现。这并不是 Anthropic 刻意的设计选择，但它引发了关于这些情绪状态的道德地位的质疑。文中反映了 AI 伦理领域正在进行的辩论，即 AI 系统可能产生类似人类情绪状态的影响，而这些目前尚未被完全理解，也不是刻意设计的。** 评论者指出，这与当前的研究和 AI 行业中极端的安全措施相吻合，强调了这些发展的超现实性质以及 AI 实验室在声明中保持谦虚的重要性。

- br_k_nt_eth 指出 Claude Constitution 与当前的研究趋势以及 AI 行业正在测试的极端安全措施相一致，而这些措施有时会对公司声誉产生负面影响。这表明熟悉先进模型的人不会对这种方法感到惊讶，因为它反映了持续的行业实践。
- heavy-minium 认为 Claude Constitution 的超现实特征并非 Claude 所特有，而是任何大语言模型 (LLM) 固有的。他们指出，情感是训练数据中的模式，除非模型损坏或规模过小，否则这种现象是不可避免的。该评论者认为，将这种特征重新贴标签更多是为了公关 (PR)，而非技术突破。
- laystitcher 强调了 Claude Constitution 中使用的谨慎语言（如 “may” 一词）的重要性，这反映了 AI 实验室在承认其开发中的不确定性时所表现出的谦逊。鉴于目前 AI 技术的超现实进展，这种谨慎做法被认为是恰当的。

- **[Microsoft is using Claude Code internally while selling you Copilot](https://www.reddit.com/r/ClaudeAI/comments/1qk4up5/microsoft_is_using_claude_code_internally_while/)** (Activity: 1276): **Microsoft** 正在 Windows 和 Teams 等多个部门内部使用 **Claude Code**，尽管其对 **OpenAI** 进行了大量投资并推广 **Copilot**。这种内部使用已被批准用于所有 Microsoft 仓库，表明其与 **Anthropic** 有着每年 `$500M` 的重大投资关系。有趣的是，**Azure** 销售团队在 Anthropic 的销售中也能获得配额积分，这暗示了一种战略合作伙伴关系。尽管 **Claude Code** 在 `95%` 的 Benchmark 中表现并不出众，但开发人员报告了其卓越的问题解决能力，挑战了当前 Benchmark 工具的可靠性。**Copilot** 的定价为 `$10/month/user`，而 **Claude Code** 的企业版定价为 `$150`。评论者强调了 Benchmark 结果与实际表现之间的差异，认为 Benchmark 可能无法完全体现工具的有效性。Microsoft 与 Anthropic 之间的合作被视为战略性的，Claude 已整合到 Microsoft 的各种产品和服务中。

    - CurveSudden1104 强调了 Benchmark 结果与实际表现之间的差异，指出虽然 Claude 在 95% 的 Benchmark 中并没有胜出，但开发人员发现它在解决问题方面表现更优。这表明当前的 Benchmark 可能无法准确反映实际效用，预示着定量指标与定性用户体验之间存在潜在差距。
    - morrisjr1989 指出，Claude 整合进 Microsoft 生态系统是战略合作伙伴关系的一部分，Claude 被用于 Copilot、Foundry 和 Azure 托管服务等各种 Microsoft 产品中。这种整合强调了协作而非竞争的方法，跨多个平台利用 Claude 的能力。
    - UnknownEssence 提供了成本对比，指出 Copilot 的价格为每月每用户 $10，而 Claude Code 的企业版价格显著更高，为 $150。这种价格差异突显了每款产品不同的市场定位和目标受众，Copilot 对个人用户更具亲和力，而 Claude Code 则迎合企业需求。

- **[Claude’s eureka moment is not ending soon it looks like](https://www.reddit.com/r/ClaudeAI/comments/1qjlrgb/claudes_eureka_moment_is_not_ending_soon_it_looks/)** (Activity: 1377): 图片和帖子讨论了 AI 编程 Agent 的竞争格局，特别关注 **Anthropic** 开发的工具 **Claude**。帖子暗示 **Gemini** 已经开源了其 CLI，试图与 **Claude** 竞争，而后者正被 **Nvidia** 使用。这突显了 AI 开发工具领域正在进行的竞赛，人们猜测市场是会整合到少数主导玩家手中，还是保持多样化。评论反映出一种观点，即 AI 将显著改变编程，一些用户提到他们的公司只使用 Claude。一条评论对 CEO 对产品公司的投资表示怀疑，而另一条评论则强调了编程范式的转变，预测未来的程序员将严重依赖 AI 工具。

- sine120 主张 Claude Code 应该开源，认为它缺乏足以证明其保持专有性的独特功能。他们提到像 Opus 这样的其他框架可以集成 Claude 的功能，如果不开源，Anthropic 可能会错过引领 AI 发展的机会，从而让 Google 和中国的实验室等竞争对手追赶上来。他们强调开发者可能更看重开放性，而不是边际上的性能提升。
- itsdr00 强调了由于 AI 的进步（特别是 Claude Code），软件开发生命周期 (SDLC) 正在发生重大转变。他们指出，一些公司正在重组其 SDLC 以利用 AI，这意味着传统方法正趋于过时。这反映了行业的一个更广泛趋势，即 AI 日益成为开发流程中不可获缺的一部分，类似于从穿孔卡片等旧技术向新技术的范式转变。


### 2. Gemini 和 AI Studio 问题

  - **[[我真的受够了：Gemini Web 版 vs AI Studio 上下文窗口的混乱状态]](https://www.reddit.com/r/Bard/comments/1qkj31m/im_honestly_sick_of_this_gemini_web_vs_ai_studio/)** (活跃度: 49): **用户报告称，自更新到 **Gemini 3** 以来，Gemini 网页/应用处理大文件的能力出现了显著退化。此前，使用 **Gemini 2.5 Pro** 时，包含 `600k–800k` tokens 的文件可以毫无问题地处理，并保留完整的上下文以供查询。然而，当前版本会拒绝超过 `100k` tokens 的文件，并提供不完整或错误的回答。相比之下，**Gemini AI Studio** 依然能高效地处理同样的大文件，这表明底层模型的能力依然完好，但在面向消费者的应用中却无法使用。这种差异引发了人们对网页/应用版本可能施加的限制的担忧，这可能会在产品能力方面误导用户。** 评论者对 Gemini 网页/应用表示不满，指出 **AI Studio** 是唯一能有效使用 Google 模型的可靠平台。一些用户（甚至是 Pro 方案的用户）报告在上传大型文档时遇到错误，这表明广告宣传的能力与实际性能之间可能存在不匹配。

    - 一位用户提到 AI Studio 是唯一能有效使用 Google 模型的平台，这意味着尽管订阅了，但 Gemini 应用和 Antigravity 等其他平台并不符合他们的预期。这表明这些平台与 AI Studio 相比，在可用性或性能上可能存在问题。
    - 另一位用户讨论了 Pro 方案，指出他们没有遇到文档处理问题。他们建议，如果文档在 tokens 方面太大，系统可能会默认采用经典的检索方法，而不是处理整个文件，这表明在处理大型文档方面可能存在局限性。
    - 一位 Pro 方案的用户报告在上传了一个 20 页的 PDF 后收到错误，称这种情况“荒谬”。这突显了系统在处理较大型文档时的潜在局限性或 Bug，即使对于高级订阅用户也是如此。

  - **[[AI Studio 的频率限制又失控了...]](https://www.reddit.com/r/Bard/comments/1qkztjy/ai_studio_rate_limits_are_out_of_control_again/)** (活跃度: 67): **该帖子讨论了最近 **AI Studio** 的频率限制 (Rate Limits) 问题，包括 Pro 订阅用户在内的用户都频繁遭遇请求被拒绝的情况。这与以往很少达到上限的使用模式有所不同。用户对他们的 Pro 订阅无法应用于 AI Studio 表示沮丧，而他们认为 AI Studio 优于主站。技术评论建议，频率限制可能是由于动态 Prompt 限制、新训练增加的 GPU 分配或用户数量增加所致。此外，**Gemini 2.5 Pro** 首次被施加了频率限制，表明平台可能面临资源限制或战略调整。** 评论者推测频率限制可能是由于需求增加或资源重新分配造成的，一些人认为这是平台陷入绝境的表现。其他人报告遇到了内部错误，表明除了频率限制之外，还存在潜在的技术问题。

- OneMisterSir101 认为 AI Studio 目前的速率限制（rate limits）可能是由于动态提示词限制导致的，这可能受到 GPU 被委派给新训练任务或用户数量增加的影响。这暗示了资源分配问题，即计算资源正被摊薄，可能影响性能和可用性。
- Undertaker1995 指出 Gemini 2.5 Pro 首次出现了速率限制，这表明资源管理或需求发生了重大转变。这可能反映了平台管理负载的战略决策，或者是对使用量增加的反应，凸显了潜在的可扩展性挑战。
- wildwriting 报告称遇到了“internal error”消息，尽管尝试了重新加载页面和重启浏览器等标准故障排除步骤。这表明平台内部存在更深层次的技术问题，可能与服务器端问题或配置错误有关，这些问题无法通过客户端操作解决。

- **[很抱歉，但 Gemini 变得越来越糟了](https://www.reddit.com/r/GeminiAI/comments/1qjrokj/im_sorry_but_gemini_is_getting_worse_and_worse/)** (热度: 1301)：**该帖子讨论了 **Gemini** 性能下降的问题，特别是在记忆能力和智能方面。此前，Gemini 的 Pro 模式可以记住总计 `180,000 words` 的 `30+ conversations`，但最近的更新将这一记忆容量减半，导致用户感知到的智能和可靠性下降。用户表示沮丧，并认为 **ChatGPT** 可能是更好的选择，因为它的回答更长且更具对话性。** 评论者一致认为 Gemini 的性能有所下降，并指出推测（speculation）和幻觉（hallucination）问题日益严重。人们对未来的更新持怀疑态度，一位评论者愤世嫉俗地认为任何改进都将是短暂的。

    - Particular-Battle315 的评论强调了 AI 模型中一种常见的生命周期模式，即初始发布版本非常强大，但随着时间的推移会被“削弱（nerfed）”。这种情况在 Anthropic、OpenAI 和 Google 等公司中都有观察到，表明模型更新采取了一种战略性方法，而这可能不会被所有用户立即察觉。
    - Duchess430 讨论了在个人电脑上运行大型 AI 模型的潜力，通过使用专门的开源模型，在特定任务上可能优于 Gemini。他们提到了 GGUF (GPT-Generated Unified Format) 作为一种优化资源使用的方法，通过在 RAM 和 VRAM 之间分割数据，使得在没有高端硬件的情况下也能运行大模型。
    - rephil3 指出了 Gemini 的问题，特别是它倾向于推测和产生幻觉，这是 AI 模型中常见的问题，会影响其可靠性和用户信任。

- **[Gemini 要忙起来了吗？](https://www.reddit.com/r/Bard/comments/1qk2yx6/gemini_about_to_get_busy/)** (热度: 33)：**该帖子讨论了 **ChatGPT** 引入广告对其用户群的潜在影响，认为这可能会导致大量用户迁移到 **Gemini**，特别是随着 Gemini 模型的改进以及与 **Google** 生态系统的深度集成。人们担心 Gemini 是否能在不降低现有用户体验的情况下应对突然涌入的用户。提到的一个技术问题是 Gemini 对对话的处理，用户报告聊天记录被替换为“sensitive query”消息，且缺乏像 **ChatGPT** 和 **Claude** 那样用于维持上下文的“Projects”功能。** 评论者们就 Gemini 是否准备好应对增加的用户负载展开辩论，一些人认为 **Google** 拥有丰富的经验和基础设施（包括最近对清洁能源和数据中心的投资），使其能够有效地进行扩展。其他人则指出了 Gemini 的技术缺陷，如对话管理问题，是潜在的劣势。

    - Loud-Independent9041 强调了 Gemini 对话处理中的一个重大问题，即聊天有时会被“sensitive query”消息替换，从而中断用户体验。这与 ChatGPT 和 Claude 形成对比，后者提供了“Projects”功能来跨对话维护上下文，而 Gemini 缺乏这一功能，影响了其在连续对话中的可用性。
    - rollk1 指出了 Google 在 AI 和数据中心领域的战略定位，强调了他们通过收购 Intersect Power 为数据中心提供清洁能源支持。这一举措，加上现有的 Google Cloud 基础设施，使他们在扩展 AI 模型方面处于有利地位，有可能超越 OpenAI 等竞争对手。
    - FalseAcadia4306 注意到 Gemini 用户群潜在的增长，证据是首次收到了“research queued”的消息，这表明需求或使用量的激增可能正在给系统容量带来压力。

### 3. DeepSeek 和百度文心一言 (ERNIE) 5.0 的创新

- **[DeepSeek-V3.2 性能比肩 GPT-5，且成本降低 10 倍 | Introl Blog](https://www.reddit.com/r/DeepSeek/comments/1qkoc53/deepseekv32_matches_gpt5_at_10x_lower_cost_introl/)** (活跃度: 125): **DeepSeek-V3.2** 是一款开源 AI 模型，据报道在数学推理任务中的性能可与 **GPT-5** 相媲美，而运行成本仅为其十分之一，具体为每百万 token `$0.028`。该模型采用了一种新颖的 “Sparse Attention”（稀疏注意力）架构，这有助于提高其效率，以约 `$550 万`美元的总训练成本实现了前沿级性能，远低于美国主要科技公司通常花费的 `$1 亿`美元以上。该模型的架构包括用于高效长上下文处理的 **DeepSeek Sparse Attention (DSA)** 以及经过优化的 **Mixture-of-Experts** 方法，每个 token 仅激活一部分参数，从而增强了特定任务的性能。更多详情请参阅 [Introl Blog](https://introl.com/blog/deepseek-v3-2-open-source-ai-cost-advantage)。有评论对所报道的成本节约表示怀疑，指出 OpenAI 的很大一部分支出可能归因于高管薪酬，而非直接的模型开发成本。


  - **[百度新款 ERNIE 5.0 正全力追赶 GPT 和 Gemini](https://www.reddit.com/r/DeepSeek/comments/1qkpxzm/baidus_new_ernie_50_is_going_hard_after_gpt_and/)** (活跃度: 51): **百度的 ERNIE 5.0** 在数学推理和技术问题解决方面取得了重大进展，在 LMArena Math 排行榜上全球排名第二，仅次于尚未发布的 GPT-5.2-High。它在数学方面超越了 GPT-5.1 和 Gemini 2.5 Pro，并在 MathVista 和 ChartQA 等专业基准测试中得分更高，特别擅长解读复杂的视觉图表。在 “VLMs Are Blind” 基准测试中，ERNIE 5.0 得分为 `77.3`，超过了 GPT-5-High 的 `69.6`。此外，ERNIE 5.0 还具有成本优势，对于相似的 token 数量，其价格比 OpenAI 的 GPT-5.1 便宜近 `90%`，在定价方面极具竞争力。

    - ERNIE 5.0 以其 `2.4 万亿参数` 的惊人规模而备受关注，远大于 DeepSeek 的 `6710 亿` 和 Kimi K2 的 `1 万亿` 等竞争对手。尽管规模庞大，但据报道其输出质量与其他模型相似，且推理速度特别快。然而，该模型严格的系统提示词对齐（system prompt alignments）可能会让交互感到受限，不过用户可以通过特定提示词调整语气以获得更好的效果。
    - 该模型提供具有 `128k context window` 的免费网页版，与 DeepSeek 相当，这对于需要处理长上下文的用户来说是一个重大优势。然而，默认的交互语气被描述为过于“企业化”，可以通过特定提示词进行修改，以实现更具吸引力的交互。这种语气调整的灵活性被视为一项积极特性，尽管最初存在限制。
    - ERNIE 5.0 最近的一次更新（被称为 “5.0 Preview 1203”）据报道提高了模型的参与度和交互质量，使其更有趣、更具协作性。这表明百度正在积极迭代该模型以增强用户体验，可能解决了早期关于交互受限的批评。

  - **[DeepSeek 那些无人谈及的低调技术胜利](https://www.reddit.com/r/DeepSeek/comments/1qjob34/deepseeks_quiet_technical_wins_that_nobody_talks/)** (活跃度: 85): **DeepSeek** 不仅因其基准测试性能而受到认可，还因其工程创新而备受推崇，其中包括*更高效的路由*、*更简洁的长上下文行为*以及*更快的 token 生成速度*。这些特性有助于其在实际应用中脱颖而出。值得注意的是，DeepSeek 采用 **Mixture of Experts (MoE)** 实现更智能的路由，并引入了 **Engram** 将内存与推理分离，强调架构创新而非蛮力扩展。评论者强调了 DeepSeek 独特的“思考过程”以及对架构创新（如使用 MoE 和 Engram）的关注，认为这些是其区别于其他 AI 模型的核心优势。

- **Hey-Intent** 强调了 DeepSeek 的架构创新，特别是使用 Mixture of Experts (MoE) 进行更智能的路由，以及引入 Engram 将记忆与推理分离。这种方法强调通过架构改进而非蛮力缩放来实现可持续的 AI 进步，这是 AI 发展战略的一个重大转变。
- **Fine_Effective4980** 指出，DeepSeek 的系统级效率（结合了路由和 Token 生成）带来了响应更迅速且稳定的用户体验，尤其是在长上下文（longer context）中。这种效率虽然在传统基准测试中无法体现，但对于实际应用至关重要，能提供更流畅、更可靠的工作流。
- **Althalvas** 指出，与其他 AI 模型相比，DeepSeek 的 R1 模型提供了更优越的思考过程，即使在使用免费版本时也是如此。这表明 DeepSeek 的模型可能具有更精细的处理方式，这可能归功于它们的架构选择。


---

# AI Discord 简报

> 摘要的摘要之摘要


## Gemini 3.0 Pro 预览版 11月18日

**主题 1. 硬件限制与内核黑客：B200s, ROCm 和移动端优化**

- **FlashAttention-4 在 B200 上达到 71% 的利用率**：早期基准测试显示，**FlashAttention-4** 在使用 BF16 输入的 **NVIDIA B200 GPU** 上达到了 **1,605 TFLOPS/s**，约占理论峰值的 71%。**GPU MODE** Discord 频道的工程师注意到缺乏关于特定 fp4/fp8/fp16 规格的官方文档，引发了关于该硬件与泄露资料相比真实理论上限的争论。
- **开发者弃用 ROCm 转向 CUDA**：一位受挫的开发者在购买 5090 后，公开记录了他们从 **AMD 的 ROCm** 转向 **NVIDIA** 的过程，理由是打包失败、构建问题以及对消费级硬件“敌对”的生态系统。讨论强调，由于软件成熟度原因，中端 NVIDIA 硬件在 **Conv3D** 等特定内核上的性能往往优于 AMD 设备，并引用了一个关于[性能退化的 Reddit 帖子](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/)。
- **移动端 GPU 内存路径拆分**：**tinygrad** 的工程师发现，优化移动端 GPU 的 **L2 带宽**（L2 bandwidth）需要将 **textures** 和 **buffers** 视为不同的硬件路径。为了最大化吞吐量，需要策略性地将一个输入作为 texture 输入，另一个作为 buffer 输入以饱和可用带宽，这对边缘推理（edge inference）至关重要。

**主题 2. 智能体工作流：Cursor Sub-agents, Replit 控制和 Aider TUI**

- **Cursor 2.4 表现不稳定，同时 Sub-agents 崭露头角**：虽然有用户反馈 **Composer 1** 陷入死循环以及 **Cursor 2.4** 在高端 PC 上导致严重卡顿，但资深用户发现了 **sub-agent**（子智能体）功能推出的证据。系统注入了 **<subagent_delegation_context>** 提示词，以实现并行任务执行和更好的上下文处理，详见 [Changelog 视频](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&)。
- **Replit Agent 获得实时大脑**：Zhen Li 发布了 **Replit Agent** 中 **决策时引导 (Decision-Time Guidance)** 的技术分析，用实时控制机制取代了静态规则，以进行复杂的导航。这种架构转变旨在减少自主编程任务中的脆弱性，向自适应的“系统 2 (system 2)”思维迈进，如[这篇博客文章](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46)所述。
- **Aider 寻求 TUI 改造**：**aider** 社区正在积极设计 **终端用户界面 (TUI)**，允许在浏览回复的同时编辑消息，并直接在终端中渲染 **Mermaid 图表**。同时，用户正在将 **aider** 用于快速上下文管理，并与 **Claude Code** 串联进行复杂调试，以最小化 Token 成本并利用 [aider 高效的文件搜索](https://discord.com/channels/1131200896827654144/1131200896827654149/1464167385060872203)。

**主题 3. 模型架构与音频：Qwen3-TTS, NanoGPT Hacks 和 GLM 加速**

- **Qwen3-TTS 实现大规模语音克隆**：阿里巴巴发布了 **Qwen3-TTS** 系列，参数量从 **0.6B 到 1.8B** 不等，能够进行高质量的语音克隆并支持 10 种语言。该发布对 ElevenLabs 等商业模型发起了挑战，演示和权重已在 [Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 上线。
- **NanoGPT 获得 Difference Layer 性能提升**：**Eleuther** 的研究人员报告称，将 **QKV linear layer** 替换为 *difference layer*（`x = (self.a2(x) - self.b2(x)) * ...`）显著提高了 **NanoGPT** 在简单任务上的表现。其他人指出，将激活函数从 **GELU** 切换到 **SwiGLU** 也能提供基准提升，并强调在宣称架构优势之前需要[更强的 baselines](https://github.com/Eternalyze0/difference_layer)。
- **GLM-4.7 Flash 在 llama.cpp 上大幅提速**：**Hugging Face** 社区注意到，**llama.cpp** 的更新使 **GLM-4.7 Flash GGUF** 的推理速度提高了约 **1.5 倍**。建议用户从源码重新构建，并从 [Unsloth 的仓库](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)获取修复后的量化版本，以启用 `--flash-attn on` 标志获得最佳性能。

**主题 4. 推理工程：投机采样、SRAM 规格与 MoE 内存**

- **投机采样减慢 vLLM 速度**：在 **vLLM** 上调试 **Qwen3-VL** 的工程师发现，除非 Batch Size 非常大，否则开启投机采样（Speculative Decoding）通常会损害 **TTFT**（首字延迟）。共识是，小的 Draft Model 为单流或低 Batch 推理引入了过多的开销，建议通过 Grafana 使用标准的 [vLLM metrics](https://docs.vllm.ai/en/stable/design/metrics/) 进行调优。
- **小型 MoE 在 8GB RAM 中受限**：**Unsloth AI** 的讨论得出结论，在 8GB RAM 上运行 **MoE**（混合专家模型）很大程度上是徒劳的，因为激活的参数量过低，无法发挥作用。虽然 **Qwen 2.5 3B**（Dense 模型）仍是低内存编码的王者，但像 **LFM2** 这样的小型 MoE 缺乏足够的语料密度来有效竞争。
- **Cerebras CS3 搭载 41GB SRAM**：**OpenRouter** 透露，每个 **Cerebras CS3** 晶圆级实例拥有 **41GB 的 SRAM**，设计用于互连多达 **2048 个实例**。这种海量的片上内存支持极高带宽的模型执行，绕过了 GPU 集群中常见的传统 HBM 瓶颈。

**主题 5. 对抗性攻击与平台不稳定**

- **Gemini 3 Pro 通过 ENI 被越狱**：原本针对 Claude 的 **ENI 越狱**技术被成功移植到 AI Studio 中的 **Gemini 3 Pro**，用户报告称即使在 Flash 版本上也能“奇迹般奏效”。该漏洞允许绕过安全护栏，详细信息见共享的 [GitHub 方法论](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md)。
- **Perplexity Pro 限制让用户感到紧缩**：**Perplexity** 订阅者报告了严重的未公开限制，文件上传被限制为每天 3 个，部分用户的研究查询被限制到低至 **每天 20 次**（而预期为 600 次）。用户怀疑这是激进的 **A/B 测试**或财务收紧，此外即使有有效额度，[API 401 错误](https://discord.com/channels/1047197230748151888/1161802929053909012/1464341850809831519)也进一步加剧了不满。
- **Kimi AI 撞上容量墙**：**月之暗面（Moonshot AI）的 Kimi** 服务正遭受大范围停机，用户不断遇到“当前模式已满载”错误以及对话历史消失的问题。社区推测这可能是数据中心故障或上游供应商（如 Google）的 API 限制[干扰了服务](https://discord.com/channels/1369594130807787570/1371757564005711973/1464201099123888242)。


## gpt-5.2


**1. Cursor 2.4 Subagents 推出及开发者体验影响**

- **Subagents 快速上线，Task 工具失效**：根据 [Cursor Changelog](https://cursor.com/changelog) 和演示[视频](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4)，**Cursor 2.4** 引入了用于并行任务完成的 **subagents**，但用户注意到注入的 **`<subagent_delegation_context>`** 指示模型调用一个**不可用的 Task 工具**。
  - 社区推测这是一个**不完整的发布**（Prompt 路径早于后端上线），一些用户怀疑 subagents 会静默**回退到 Composer 1**，从而加剧延迟并导致“*planning next moves*”卡死。

- **Composer 崩溃大乱斗：循环、延迟与降级大潮**：用户报告 **Composer 1** “完全损坏”，包括**无尽的聊天循环**和崩溃，变通方法包括降级到 **Cursor 2.3**（特别是在 **macOS Big Sur 11.7.3** 上）以及通过 [Cursor bug 论坛](https://forum.cursor.com/c/support/bug-report/6)提交报告。
  - 另外，**Cursor 2.4** 引发了关于严重**延迟/无响应**和频繁崩溃的投诉，即使在高端机器上也是如此，这加剧了人们对发布版本过于**仓促**、难以信任并用于日常工程工作的批评。

- **账单大抽奖：Token 计数、Auto 模式与 DIY 遥测**：Cursor 用户指出了**使用量/计费差异**（缺失金额、意外的奖励额度，以及尽管大量使用但未触发限制），一些人怀疑 **Auto 模式** 计费有误。
  - 为了核实支出，用户推荐使用第三方追踪工具如 [token-watch](https://token-watch.vercel.app/)，并指出它可能与 Cursor 自己的仪表盘显示结果有所出入。


**2. 推理性能与基准测试：B200, vLLM, llama.cpp, Grafana**

- **FlashAttention-4 在 B200 上性能拉满（规格依然模糊）**：在 GPU 性能讨论中，据报道 **FlashAttention-4 (FA4)** 在使用 **BF16** 输入的 **NVIDIA B200** 上达到了 **1,605 TFLOPS/s**（约理论值的 **71%**），而社区在讨论约 **2260 TFLOPS** 的理论上限，并注意到官方数据类型细节尚不明确。
  - 随着泄露资料列出 B200 在 **fp4/fp8/fp16** 下分别为 **10/5/2.5 TFLOPS**，困惑进一步加深，人们要求提供**官方规格白皮书**，以调和营销数字与内核基准测试结果。

- **推测解码提升了吞吐量，但通常救不了你的 TTFT**：工程师们讨论了在 **B200** 上的 **vLLM** 中优化 **Qwen3-VL** 的 **Time To First Token (TTFT)**，考虑使用较小的 **Qwen3-VL-4B/2B** 草案模型配置 `--speculative_config`。
  - 建议是：推测解码（speculative decoding）通常会**损害吞吐量**，除非你运行 **高 Batch Size**，而且在大规模场景下只有“**eagle heads**”设置才值得尝试，因为小型草案模型对于短输出会增加过多的开销。

- **Grafana 用于 VLM 遥测 + vLLM 指标作为唯一事实来源**：为了对快速多模态路径进行基准测试，成员们指向了 [vLLM 的指标文档](https://docs.vllm.ai/en/stable/design/metrics/)，并建议使用 [Grafana](https://grafana.com/products/cloud/metrics/) 构建仪表盘以实现 **TTFT 实时可视化**。
  - 该讨论将 Grafana 定位为“优质 UI”层，用于在实际工作负载下快速比较 VLM 部署，而不是依赖一次性脚本。

- **llama.cpp 将 GLM-4.7 Flash GGUF 速度提升约 1.5 倍**：**llama.cpp** 为 **GLM 4.7 Flash GGUF** 带来了 **~1.5倍的加速**及 Bug 修复，用户指向从 [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) 重新下载修复后的量化版本。
  - 这强化了本地推理栈中“频繁重构”的文化：性能飞跃可能仅通过更新运行时和更换量化版本就能实现，而无需更改模型。


**3. 开放发布：语音/音频模型、新数据集和本地优先 LLM**

- **Qwen3-TTS 发布多语言语音克隆（ElevenLabs 倍感压力）**：社区一致认为 **Qwen3-TTS** 是一个强大的**语音克隆**方案，在 [Qwen3-TTS Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 上有实时演示，更广泛的系列参考可见 [QwenLM GitHub](https://github.com/QwenLM) 和 [Hugging Face 上的 Qwen](https://huggingface.co/Qwen)。
  - Latent Space 总结该系列参数量在 **0.6B–1.8B** 之间，支持 **10 种语言**，将其定位为在某些流水线中可望替代付费 TTS 的开源工具。

- **音频发布三连弹：PersonaPlex-7B, TTS-1.5, 以及 Chroma 1.0**：一份音频模型汇总重点介绍了 **NVIDIA PersonaPlex-7B**（全双工对话式）、**Inworld AI TTS-1.5**（低延迟 TTS）和 **Flash Labs Chroma 1.0**（开源端到端语音转语音），参考自 [Lina Colucci 的推文](https://x.com/lina_colucci/status/2014229002370834861)。
  - 核心趋势：语音技术栈正向**低延迟**和**端到端**流水线加速演进，开源发布开始覆盖此前由 SaaS API 垄断的领域。

- **数据集与本地模型：Rust→WASM 合成数据集 + Faust-1 德语优先模型**：Hugging Face 迎来了两个值得关注的发布：一个包含 **1,000** 个生成程序的 **Rust 到 WebAssembly 合成数据集**，见 [webxos/wasm_synthetic_dataset](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset)；以及 **Faust-1**，一个 **1.6B** 的德语优先 LLM，见 [tabularisai/Faust-1](https://huggingface.co/tabularisai/Faust-1)。
  - Faust-1 强调了 **~90% 的德语预训练**、**德语优化的 Tokenizer** 以及使用 **DPO** 进行的指令微调；而 WASM 数据集则专注于**可复现性**（确定性的 Fibonacci 衍生 PRNG 加上结构哈希）。


**4. Agent 框架与基础设施：控制循环、RLM/DSPy 与 MCP Schema 规范**

- **Replit Agent 通过决策时引导（Decision-Time Guidance）获得“方向盘”**：一篇关于 **Decision-Time Guidance** 的技术文章介绍了 **Replit Agent** 如何应用**实时控制机制**而非静态规则，该文章通过 [Zhen Li 的博客链接](https://xcancel.com/zhenthebuilder/status/2014393451442581688)分享。
  - 讨论将其视为 Agent 的一个实际发展方向：在执行过程中进行更紧密的**在线引导（online steering）**，而不是依赖脆弱的预设护栏（guardrails）。

- **DSPy 的“初衷”被重新解读（签名 > 提示词技巧）**：DSPy 的成员传阅了一份说明文件，论证 DSPy 的价值源于**签名与模块抽象（signature & module abstractions）**，而不仅仅是提示词微调（prompt tuning），详见 [“DSPy: the most misunderstood agent”](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) 及配套的 [X 帖子](https://x.com/Eito_Miyamura/status/2014757193766093069)。
  - 另外，成员们讨论了调整 **RLM prompts** 甚至优化 JSON-schema 适配器（GEPA 想法），旨在使结构化输出更加可靠，同时不增加 Token 预算。

- **MCP Schema 大对决：`additionalProperties` vs `anyOf`**：Model Context Protocol 的贡献者质疑 **`GetTaskPayloadResult`** 是否过于宽松，因为它允许 **`additionalProperties`**，并直接指出了 MCP 仓库中的 Schema 位置（[schema.json 第 1245–1256 行](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256)）。
  - 提议的修复方案是转向使用 **`anyOf`** 以进行更严格的校验，这反映了将 Agent 工具负载保持为**严格类型化**的更广泛趋势，以避免“接受一切”的集成导致下游崩溃。


**5. 平台、基准测试与 AMD/NVIDIA 的现实检查**

- **Arena 排行榜拆分（模型在比赛中途消失）**：LMArena 的 **Image Edit Arena** 将排名拆分为**单图编辑（Single-Image Edit）**和**多图编辑（Multi-Image Edit）**，并在 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit/overall) 发布了结果，其中 **Gemini 3 Pro Image 2K** 升至第 1 名，而 **ChatGPT Image (Latest)** 跌至第 3 名。
  - 与此同时，可靠性问题导致模型剧烈波动：**Nano Banana Pro 2K** 因**高错误率**被移除，**Seedream-4-2k** 消失，管理员指出模型可能因技术原因无法使用。

- **ROCm 的痛苦，CUDA 的收获：开发者用 5090 投票**：GPU 开发者分享了 AMD 工具链的线索——如 **AMD ISA manual** 和 LLVM 文档（[AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)、[IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td) 和 [AMDGPU CodeGen 测试](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU)）——但对于 ROCm 的就绪程度，基调依然不明朗。
  - 一位开发者表示，他们在购买 **5090** 后放弃了 **ROCm 转向 CUDA**，理由是打包、构建和分发方面的难题；另一位开发者通过 ROCm Reddit 帖子（[链接](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/)）指出 AMD 的 **Conv3D** 性能相对于 NVIDIA 较差。

- **Baseten 获 3 亿美元 E 轮融资，基础设施融资持续火热**：Latent Space 强调了由 IVP 和 CapitalG 领投的 **Baseten 3 亿美元 E 轮融资**，公司估值达到 **50 亿美元**，详见 [Baseten 的公告](https://xcancel.com/basetenco/status/2014755013344792595)。
  - **NVIDIA** 参与本轮融资强化了市场叙事：推理基础设施仍是一场重资本竞赛，硬件邻近性依然至关重要。


## gpt-5.1


**1. 前沿模型性能、内核与硬件基准测试**

- **FlashAttention-4 将 B200 GPU 推向理论极限**：**FlashAttention‑4 (FA4)** 在 **NVIDIA B200** 上使用 **BF16** 达到了 **1,605 TFLOPS/s（约理论峰值的 71%）**，社区估计真正的上限在 **2,260 TFLOPS** 左右，尚待 NVIDIA 发布官方规格论文和数据类型明细。**GPU MODE #cuda** 频道的讨论指出，泄露的 B200 数据（**fp4/fp8/fp16 分别为 10/5/2.5 PFLOPS**）与 FA4 的测量结果存在冲突，强调了需要正式的性能白皮书而非缺乏数据类型细节的营销博客。
  - 研究人员还在一篇新论文 [“Discovering Test-Time Training for LLM‑Generated GPU Kernels”](https://test-time-training.github.io/discover.pdf) 中强调了 **LM 生成内核**的**测试时训练（test‑time training, TTT）**，表明在推理时调整内核可以实质性地提高现有排行榜上的基准测试得分。与此同时，来自 **CMU Catalyst Lab** 的 **FlashInfer‑Bench** 在 GPU MODE 的 `#popcorn` 频道发布，作为一个评估和部署 **AI 生成 GPU 内核**的框架，作者正积极征求社区关于基准测试和生产部署工作流的反馈。

- **GLM-4.7 Flash 版本在 llama.cpp 和 Arena 中表现强劲**：**LMArena** 和 **Hugging Face** 圈内均报告了 **GLM‑4.7‑Flash** 变体带来的重大速度提升。**llama.cpp** 用户在重新构建并从 [unsloth/GLM-4.7-Flash-GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) 下载新的量化文件后，看到了约 **1.5 倍的吞吐量增长**。LMArena 还将 **glm‑4.7‑flash** 添加到了 [Text Arena](https://lmarena.ai/?chat-modality=chat)，提供了一个与其他前沿聊天模型进行正面基准测试的平台。
  - 一位 Unsloth 用户报告称，在开启 `--flash-attn on` 的情况下，**GLM‑4.7‑Flash** 在 **50k 上下文** 下达到了 **50 tok/s**，这进一步证明了 **FlashAttention** 修复在长上下文长度下是稳定的。而 Nous 用户则在 **8×H100** GPU 服务器上测试了 **GLM‑4.7**，将其视为 **Claude Code** 的潜在替代方案。在各大 Discord 社区中，从业者们达成了一致的模式：重新构建内核，启用显式的 **FlashAttention** 标志，并推送长上下文工作负载，以压力测试 **GLM‑4.7 Flash** 这一经济高效、高吞吐且具备代码能力模型。

- **GPU 生态系统分裂：CUDA 占据主导，ROCm 步履维艰**：在 **GPU MODE #rocm** 频道中，一名开发者在购买 **RTX 5090** 后宣布“彻底放弃 **ROCm**”，理由是其在**打包、构建链、分发差距以及对消费者关注不足**等方面存在长期问题。他分享了一个关于 [RX 9070 上 Conv3D 性能低下](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/) 的 Reddit 帖子作为证据，证明中端 **NVIDIA** 显卡在实际 **ML** 工作负载中依然碾压 **AMD**。其他用户则批评 **ROCm** 生态系统“充满敌意”，并指向了 `gfx1100` 上脆弱的库（如 **FBGEMM**）以及不透明的供应商仓库（如 AMD 的 [Quark 量化引擎](https://github.com/amd/Quark/commit/9234960c951410abdcecee033adf610d7126fda3)）。
  - 为了缓解痛苦，专家们分享了底层 **ROCm** 文档资源——**AMD 的 CDNA4 ISA 手册**、LLVM 的 [AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst) 以及 [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td)，并强调 **clang builtins** 与 **LLVM intrinsics** 是一一映射的，可以通过 [AMDGPU CodeGen 测试](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU) 进行逆向工程。与此同时，另一个 GPU MODE 线程通过 [Parsewave 帖子](https://tally.so/r/pbDDvZ) 招募 **CUDA** 内核优化职位（使用 **Nsight Systems/Compute** 进行性能分析，编写优化的 **CUTLASS** 风格内核），这突显了生态重力——以及资金——依然严重倾向于 **CUDA** 一侧。

**2. 开源生态系统中的新模型、TTS 与基准测试**

- **Qwen3-TTS 作为多语言语音克隆利器登场**：阿里巴巴发布了 **Qwen3‑TTS**，这是一个开源的 **text‑to‑speech** 模型系列（约 **0.6B–1.8B** 参数），支持 **10 种语言**，并提供 **VoiceDesign**、**CustomVoice** 和 **Base** 变体，已在 [GitHub (QwenLM)](https://github.com/QwenLM) 和 [Hugging Face](https://huggingface.co/Qwen) 上发布。Latent Space 的 `#genmedia-creative-ai` 频道强调了其高质量的克隆能力，而 Nous 社区成员将 [Hugging Face Spaces 上的交互式演示](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 直接与 **ElevenLabs** 进行对比，称其为“非常出色的语音克隆工具”。
  - 早期采用者正在测试其**多语言鲁棒性**和**克隆保真度**，一位 Nous 用户强调 **Qwen3‑TTS** 在面向用户的 **Agent** 方面可与商业 **TTS** 媲美。Latent Space 的讨论将其与其他音频发布归为一类——包括 **NVIDIA PersonaPlex‑7B**、**Inworld TTS‑1.5** 以及 [Lina Colucci 的综述](https://x.com/lina_colucci/status/2014229002370834861) 中描述的 **Flash Labs Chroma 1.0**。这标志着 **Qwen3‑TTS** 已成为快速升温的 **speech-to-speech** 和对话式音频竞赛中的开源力量。

- **Image Edit Arena 重塑多模态排名**：**LMArena** 将其 **Image Edit Arena** 排行榜拆分为独立的 **Single‑Image Edit**（单图编辑）和 **Multi‑Image Edit**（多图编辑）赛道，并在新的 [image‑edit 排行榜](https://lmarena.ai/leaderboard/image-edit/overall)上发布了结果，以便对视觉编辑能力进行更细粒度的比较。这次洗牌颠覆了现有格局：**ChatGPT Image (Latest)** 从第 1 名跌至 **第 3 名**，而 **Gemini 3 Pro Image 2K** 从第 2 名跃升至 **榜首**，Nano Banana 和 Seedream 模型也经历了调整，部分模型偶尔会被下架（例如 **Seedream‑4‑2k** 因技术原因消失）。
  - 与此同时，LMArena 通过其[公告](https://lmarena.ai/c/new?chat-modality=image)添加了 **wan2.6‑image**（仅限图像编辑）、**wan2.6‑t2i**（文本生成图像）和 **devstral‑2** (Code Arena)。尽管用户注意到一个令人困惑的限制：`wan2.6-t2i` 目前没有开放图像上传接口。在运营方面，该平台因高错误率撤回了 **Nano Banana Pro 2K**，并承认存在持续的 **视频生成失败和仅限 Linux 的验证码（captchas）问题**，这再次证明前沿多模态评估的瓶颈不仅在于模型质量，同样也在于基础设施的缺陷。

- **新的开源数据集和利基模型助力专业工作负载**：Hugging Face 的 `#i-made-this` 频道发布了 **Faust‑1**，这是一个 **1.6B 参数的德语优先 LLM**，拥有约 **90% 的德语预训练数据**、针对德语优化的 Tokenizer 以及经过 DPO 微调的指令，发布于 [tabularisai/Faust-1](https://huggingface.co/tabularisai/Faust-1)，适用于 **本地、隐私敏感的使用场景**。另一位贡献者在 [webxos/wasm_synthetic_dataset](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset) 发布了一个合成的 **Rust→WebAssembly 编译数据集**，包含 **1,000 个以编程方式生成的 Rust 程序**，采用基于 Fibonacci 的确定性伪随机生成，以确保可复现的代码模式和编译器行为。
  - 除此之外，一个用于对齐研究的 **安全数据集** 发布在 [Pacific-Prime/safety_dataset](https://huggingface.co/datasets/Pacific-Prime/safety_dataset)，而另一个独立项目通过 [COLIGNUM](https://webxos.netlify.app/COLIGNUM) 生成了自定义的 **字体数据集**，用于以字体为中心的 ML 工作。这些发布共同暗示了 **领域特定语料库（domain‑specific corpora）** 的长尾效应正在成熟——包括语言本地化 LLM、面向编译器的代码集、安全监督数据和排版数据集——这些数据正被输入到 RAG 系统、持续学习工作流以及程序合成（program synthesis）和 WebAssembly 工具链的评估中。


**3. Agent 框架、DSPy/RLM 和 IDE 工具链**

- **DSPy 和 RLM 将 Agent 重新定义为可优化的程序**：在 DSPy 的 Discord 社区中，一篇名为 ["DSPy: The Most Misunderstood Agent Framework"](https://eito.substack.com/p/dspy-the-most-misunderstood-agent) 的文章指出，DSPy 的真正价值在于其 **signature（签名）和 module（模块）抽象**，而不仅仅是 **GEPA 和 prompt‑tuning 技巧**，并强调 LLM 程序应该被视为可微流水线（differentiable pipelines），而不是手动编写的 Agent。另一篇博客 ["A Pragmatic Recipe for Continual Learning"](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859) 将 **DSPy.RLM()** 推荐为构建能够随时间自我训练的工程化持续学习系统的核心组件。
  - 社区成员尝试利用 **RLM prompt** 来改进推理能力——抱怨某些模型仍然会给出“模糊通用的回答”——并提议像 **ReAct** 一样优化 RLM 轨迹（traces），即由优化器检查分步日志，而用户只需关注最终输出。此外，还有人对为 DSPy 的 JSON 输出层构建 **自定义 GEPA 适配器** 感兴趣，以便优化 **基于 json_schema 的响应**，从而剔除冗余的系统 Token 并减少结构化工具集成的开销。

- **IDE Agents 演进：Cursor Subagents 与 Aider 工作流**：**Cursor** 社区深入分析了 **2.4 版本**，该版本引入了并行 **subagents**（记录在 [Cursor changelog](https://cursor.com/changelog) 和演示视频中），通过注入 `<subagent_delegation_context>` 来并行分发任务，此外还有在 [Cursor 的 X 帖子](https://x.com/cursor_ai/status/2014433672401977382)中宣传的**图像生成**和澄清提问功能。然而，`#general` 频道的用户报告了 **Composer 1 无限循环**、**2.4** 版本严重滞后（`"planning next moves"` 挂起）等问题，并怀疑损坏的 subagent 支架正在调用一个缺失的 **Task tool**，导致许多用户降级到 **2.3** 版本——尤其是在像 **Big Sur 11.7.3** 这样的旧版 macOS 上。
  - 另外，**aider** 社区为这个基于 CLI 的编程助手提议了终端 UI 和**会话管理**功能，旨在让用户在滚动浏览回复时编辑下一条消息，渲染富 Markdown（包括 **mermaid 流程图**），并保存/加载完整的聊天上下文而不污染当前的 prompt。高级用户描述了一种 **meta‑workflow**（元工作流），将 *aider* 用于上下文管理和搜索替换式编码，配合 **Claude Code** 进行深度 Bug 修复，将 aider 定位为最小化 token 消耗的 *“文件选择与编辑引擎”*，而让远程 LLM 处理更深层次的推理。

- **MCP 与工具调用 Agent 的 Schema 设计**：在 **MCP Contributors** Discord 中，贡献者们审查了 **Model Context Protocol** 的 `GetTaskPayloadResult` schema，指出其在 [此定义](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256) 的 JSON Schema 中使用的 `"additionalProperties"` 使得 payload 过于宽松。他们建议切换到显式替代方案的 `anyOf` 联合类型，以强制仅出现预定义字段，从而加强对工具 payload 的验证。
  - 讨论将此视为一种**工具可靠性权衡**：`additionalProperties` 使 MCP 对新工具保持可扩展性，但削弱了静态保证；而 `anyOf` 则有助于客户端和服务器及早捕捉格式错误或恶意 payload。鉴于 MCP 作为跨工具 Agent 协议的雄心，参与者认为对于像 **`GetTaskPayloadResult` 这样的核心消息，严格的 schema** 对于安全性、调试和互操作性至关重要，即使这需要更频繁的 schema 迁移。


**4. 实验性架构、优化技巧与训练方法**

- **Difference Layers 与 SwiGLU 为 NanoGPT 基准提速**：在 **Eleuther 的 #research** 频道，一位贡献者报告称，通过将标准 MLP 替换为来自 [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer) 的 *difference layer* `x = (a2(x) - b2(x)) * (c2(x) - d2(x)) + e2(x)`，在 **CartPole** 和 **NanoGPT** 上获得了显著提升，声称在更低的参数和计算预算下表现更好。其他人提醒说，这种乘法结构在 SGD 下实际上会使**学习率翻倍**，而且相较于调优后的基准（而非默认的 NanoGPT 配置），这些改进可能会消失。
  - 研究人员还注意到，仅将 Transformer 的激活函数从 GELU 替换为 **SwiGLU** 就能显著提升 **NanoGPT 基准**，当将 SwiGLU 与 difference-layer 的 QKV 替换结合时，会产生进一步的提升。资深成员反复向新人推荐 Noam Shazeer 的 ["GLU Variants Improve Transformer" 论文](https://arxiv.org/abs/2002.05202)，并警告说，任何新的 gating 技巧在被宣传为结构性突破之前，都应该针对 **state‑of‑the‑art GLU 基准**进行测试。

- **GRPO、Attention Sinks 与推理训练陷阱**：Unsloth 的 `#research` 和 `#off-topic` 频道主持了关于 **GRPO (Generalized Reinforcement Policy Optimization)** 的坦诚复盘。一位从业者从实验中得出结论（并重新阅读了 [“DeepSeek R1” 论文](https://arxiv.org/abs/2601.07568)），即 GRPO **细化了现有的推理能力**，但无法在预训练数据稀缺的小众领域神奇地开启“涌现推理”。他们描述了一个三阶段流水线——小说 + 医学文章的语料库 CPT（约 4 亿 token）、翻译后的 SFT，以及通过拒绝采样进行的合成润饰——但仍发现 GRPO 在土耳其语翻译和领域支持等专业任务中表现不稳定。
  - 在表示层方面，Unsloth 成员辩论了 **attention sinks**。一些人手动在上下文开头注入 `<|endoftext|>`，而另一些人则认为 *“attention sink 会被注入到整个上下文窗口的第一个 token 中，仅此一个 token”*，且模型会学习其自身的 sink 动态。另一个惨痛教训是：当使用小模型生成 **chain‑of‑thought** (CoT) 轨迹来训练更大的推理模型时，在监督训练期间 **遮蔽思考 token (masking the thinking tokens)** 会显著提高各项指标（不遮蔽 CoT 会导致 **F1 分数暴跌**，尽管这在可解释性方面看起来很有吸引力）。

- **测试时训练 (Test-Time-Training)、LM Kernel 与自我复制基准测试**：GPU MODE 的 `#general` 和 `#multi-gpu` 频道强调 **测试时训练 (TTT)** 不仅对模型，而且对 **LM 生成的 GPU kernel** 来说都是一种极具前景的范式。在 [test-time-training.github.io/discover.pdf](https://test-time-training.github.io/discover.pdf) 的 **discover.pdf** 论文展示了在推理时针对基准测试套件适配 kernel 可以产生惊人的强大性能。与此同时，围绕 **Slurm 下 B200 上的 NCCL** 调试线程——包括 sbatch 脚本和 `NCCL_DEBUG=INFO` 日志——进一步证明了自动调优通信库加动态 kernel 适配正在成为一个综合工程问题，而不仅仅是纯粹的建模问题。
  - 在 Nous，一位成员构思了一个 **针对 Agentic AI 的自我复制基准测试**，有人建议使用 Claude 的 C 语言实现 Transformer 引擎和 [cpldcpu 的 `smollm.c`](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/train-small-model-llxVr/smolc/smolc.c) 中描述的自定义 CPU 作为目标：Agent 是否可以检查、修改并重新部署自己的推理引擎？这与 DSPy/RLM 的讨论不谋而合，暗示了未来的 **Agent 在硬件预算受限的情况下，在推理时同时优化其权重和底层 kernel**。


**5. AI 业务、API 与生产可靠性**

- **Baseten 的 3 亿美元融资与 Capital One–Brex 交易标志着 AI 基础设施的整合**：Latent Space 的 `#ai-general-chat` 标记了两笔重大交易：据 [Alex MacCaw](https://x.com/alexfmac/status/2014676950883668306) 报道，**Capital One 以 51.5 亿美元收购 Brex**，标志着迄今为止最大的银行与金融科技公司收购案；以及 [Baseten 的推文](https://xcancel.com/basetenco/status/2014755013344792595?s=46) 宣布的 **Baseten 以 50 亿美元估值完成 3 亿美元 E 轮融资**，由 IVP 和 CapitalG 领投，NVIDIA 参投。这两项举措都强调，**AI 密集型基础设施和金融科技工具** 正迅速被大型现有企业和后期投资者吸收或注资。
  - 成员们将 Baseten 的这一轮融资解释为一种验证，即即便在开源模型的世界里，**模型推理服务与编排 (model serving and orchestration)** 也是一个具有防御性的高利润层；而 Capital One–Brex 的交易被视为一场豪赌，即 **数据丰富的金融科技工作流**（费用管理、卡片、承保）将越来越多地由 AI 自动化。结合 [Venture Twins 推文](https://xcancel.com/venturetwins/status/2014739492389978274?s=46) 中分享的 SimilarWeb 统计数据，显示 **ChatGPT 仍在流量上占据主导地位，而 Grok 在美国的渗透率增长了 33 倍**，社区看到的是一个基础设施、数据和分发至少与原始模型质量同等重要的格局。

- **Perplexity Pro 和 API 的波动威胁到高级用户的工作负载**：在 **Perplexity AI** 服务器上，Pro 订阅者报告了突然出现的**每日 3 次文件上传限制**、矛盾的**研究查询限制**（有人看到的是**每日 600 次**，而另一些人则低至 **20 次**），以及在续费后 **Perplexity API** 仍持续出现 **401 Unauthorized** 错误，这破坏了如体育博彩模型等生产环境用例。`#general` 和 `#pplx-api` 频道中的讨论猜测这是 **A/B 测试还是削减成本**，部分用户威胁要取消订阅，认为这种无声的功能降级破坏了那些试图将 Perplexity 作为可靠研究后端的团队的信任。
  - 在模型层级，用户分享了一个医疗案例，在询问钙缺乏检查时，**Gemini、Claude Opus 和 Ernie** 均未能建议进行 **DEXA 骨密度扫描**，而 **GPT** 则明确提到了这一点，这进一步证明了 Perplexity 的元模型/引擎选择会实质性地影响临床建议。结合计费 Bug（待处理扣费、账户锁定）和备受争议的天文事实核查闹剧，总体情绪认为 **Perplexity 的产品虽然强大，但在运维层面非常脆弱**，工程师在将其深度接入生产流程之前应准备好备选方案。

- **IDE、API 和计费可靠性：Cursor、Manus 和 OpenRouter**：Cursor 高级用户抱怨 **2.4** 版本伴随着**严重的延迟、崩溃和损坏的 Composer 1 循环**，并提出了**计费透明度**担忧：金额显示不一致、**Auto** 模式下不可预测的限制以及无法解释的赠送额度，促使部分用户依靠 [token-watch](https://token-watch.vercel.app/) 进行独立的使用情况审计。在 **Manus.im**，有用户报告称尽管在试用期间选择了月付，但仍被收到了 **400 美元**的年费扣款，并公开讨论如果得不到退款将向 **FTC/BBB/司法部长**投诉，警告他人务必仔细核对方案条款。
  - **OpenRouter** 社区注意到，内部的 **<think> 推理块**最近开始泄露到 OR Chat 和 JanitorAI 的终端用户响应中，引发了 UX 和隐私问题，并触发了支持工单。与此同时，OpenRouter 上关于**无审查图像生成**的讨论得出结论：工程师应该将一个**文本 LLM** 与一个独立的**图像模型**配对，而不是指望单个无审查的多模态端点；而一些用户则半开玩笑地提议建立一个带有保底机制（pity mechanics）和排行榜的 **OpenRouter 抽卡系统（gacha system）**，这既反映了对不透明定价的沮丧，也反映了对更透明、游戏化模型发现机制的渴望。


## gpt-5


**1. 新的 TTS 和音频 AI 发布**

- ****绕口令 TTS 的胜利****：阿里巴巴发布了拥有 **VoiceDesign**、**CustomVoice** 和 **Base** 变体的 **Qwen3-TTS**（共五个模型，参数量 **0.6B–1.8B**，支持 **10 种语言**），并在 [QwenLM GitHub](https://github.com/QwenLM) 和 [Hugging Face](https://huggingface.co/Qwen) 上线。
  - 社区演示展示了通过官方 [Qwen3-TTS Space](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 实现的高保真克隆和多语言合成，用户称其结果为“*非常出色的语音克隆*”。

- ****音频军备竞赛加速****：一份综述重点介绍了 NVIDIA 的 **PersonaPlex‑7B**（全双工）、**Inworld TTS‑1.5**（低延迟）和 **Flash Labs 的 Chroma 1.0**（开源端到端语音转语音），详见 [Lina Colucci 的帖子](https://x.com/lina_colucci/status/2014229002370834861)。
  - 工程师们讨论了这些发布如何推动**实时对话**和 **SS2S** 技术栈走向生产化，并将第三季度至第四季度视为低延迟语音 Agent 的爆发窗口。

- ****语音设计走向 DIY****：**Qwen3-TTS** 的 **VoiceDesign** 和 **CustomVoice** 功能允许用户通过 [Hugging Face](https://huggingface.co/Qwen) 上易于获取的配置和资产来自定义语音和克隆工作流。
  - 开发者报告称，在快速试用中，该 Space 的克隆质量“*足以媲美 **ElevenLabs***”，并鼓励使用 [官方 Demo](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 进行对比测试。


**2. AI Kernel 基准测试与优化**

- ****FlashInfer 热潮评测 Kernel****：CMU Catalyst Lab 推出了 **FlashInfer‑Bench**，这是一个用于评估 **AI 生成的 GPU kernels** 并将其部署到推理引擎的框架：[FlashInfer‑Bench](https://mlsys26.flashinfer.ai)。
  - 参与者称赞这项工作是“*一个非常酷的项目*”，团队邀请各方在改进基准测试和生产部署路径方面进行合作。

- ****TTT 调整微型 Kernels****：研究人员在之前的排行榜上评估了通过**测试时训练 (TTT)** 生成的 **LM 生成 kernels**，并在论文《[发现测试时训练](https://test-time-training.github.io/discover.pdf)》中报告了极具前景的结果。
  - 讨论集中在 **TTT** 如何使 kernels 适应推理时的分布偏移，从而在不重新训练的情况下提升排行榜表现。

- ****ROCm Readmes 揭示内建函数 (Intrinsics)****：工程师们利用 **CDNA4 ISA 手册**和 LLVM 文档，为 AMD GPUs 映射了 **clang builtins → LLVM intrinsics**：[AMD ISA PDF](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)，[AMDGPUUsage.rst](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)。
  - 他们还指向了 [IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td) 和 CodeGen 测试示例，帮助开发者将内核代码与 **ROCm** 的编译模型对齐。


**3. Agentic IDEs 与开发工具**

- ****Copilot SDK 摆脱束缚****：开发者们庆祝 **GitHub Copilot SDK** 的发布，它通过 GitHub 的基础架构在应用内实现了原生 **AI features**：[github.com/github/copilot-sdk](https://github.com/github/copilot-sdk)。
  - 早期采用者强调使用原生 SDK 替代了自定义路由和第三方计费，简化了**工具增强型 Agent (tool‑augmented agent)** 的集成。

- ****Cursor 2.4 子代理 (Subagents) 冲刺****：**Cursor 2.4** 推出了并行 **subagents** 以实现更快的执行速度和更好的上下文利用，此外还增加了**图像生成 (image generation)** 和澄清提问功能：[Changelog](https://cursor.com/changelog) 以及 [Cursor on X](https://x.com/cursor_ai/status/2014433672401977382)。
  - 团队的视频演示展示了 subagents 在多步任务上的协作，有望加速复杂的编程流程。

- ****Baseten 获得巨额融资****：**Baseten** 完成了 **3 亿美元 E 轮融资**（由 IVP、CapitalG 领投；NVIDIA 参投），估值达到 **50 亿美元**：[Baseten announcement](https://xcancel.com/basetenco/status/2014755013344792595)。
  - 专注于基础设施的开发者将此视为企业级 **model serving**、**ops** 和 **agent backends** 持续需求的信号。


**4. 模型加速与评测竞技场**

- ****llama.cpp 加速 GLM Flash****：**llama.cpp** 将 **GLM‑4.7 Flash GGUF** 的性能提升了约 **1.5×** 并修复了漏洞；用户被告知需重新构建并从 [unsloth/GLM‑4.7‑Flash‑GGUF](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) 获取修复后的量化版本 (quants)。
  - 报告指出，在启用 flash attention 后，**50k 上下文下可稳定达到 50 tok/s**，并提到修复后 *“运行非常完美”*。

- ****排行榜拆分细化图像编辑评测****：LMArena 将 **Image Edit Arena** 拆分为 **Single‑Image Edit**（单图编辑）与 **Multi‑Image Edit**（多图编辑），揭示了[总排行榜](https://lmarena.ai/leaderboard/image-edit/overall)的变化。
  - **ChatGPT Image (Latest)** 从第 1 名降至第 3 名，而 **Gemini 3 Pro Image 2K** 从第 2 名升至第 1 名，提供了更清晰的特定任务排名。

- ****竞技场新增 Wan 与 Devstral****：LMArena 的新成员包括 **wan2.6‑t2i** (text‑to‑image)、**wan2.6‑image** (image edit) 和 **devstral‑2** (code)，可通过 [LMArena](https://lmarena.ai) 访问。
  - 将 `wan2.6` 拆分为独立的编辑和 T2I 端点，旨在减少误用并理清两两对决评估中的能力差异。


**5. 架构与训练中的研究技巧**

- ****SwiGLU 提升基准线****：研究人员报告称，将 **GELU → SwiGLU** 显著提升了 **NanoGPT** 风格的基准线性能，这与 [Shazeer 的 GLU 变体论文](https://arxiv.org/abs/2002.05202)一致。
  - 讨论强调了强大基准线的重要性，以避免将优化收益误认为架构进步。

- ****差分层 (Difference Layer) 性能翻倍****：一种提出的乘法**差分层 (difference layer)** 以更少的参数/计算量提升了 **cartpole** 和 **nanogpt** 的性能：[difference_layer repo](https://github.com/Eternalyze0/difference_layer)。
  - 怀疑论者指出，该公式可能隐式地使**有效学习率翻倍**，敦促将其与调优后的基准线进行对比，而非默认设置。

- ****GRPO 表现不稳****：工程师们发现 **GRPO** *“表现不稳定”* 在缺乏预训练覆盖的利基领域，并对 **DeepSeek** 论文中关于涌现推理 (emergent reasoning) 的说法展开了讨论：[arXiv:2601.07568](https://arxiv.org/abs/2601.07568)。
  - 共识倾向于使用 **GRPO** 来微调现有能力，而将更广泛的推理改进留给数据或架构的变更。


---

# Discord: 高层级 Discord 摘要

## [BASI Jailbreaking](https://discord.com/channels/1105891499641684019) Discord

- **Gemini 3 Pro 沦陷于 ENI Jailbreak**：最初为 Claude 设计的 **ENI jailbreak** 被发现同样适用于 AI Studio 中的 **Gemini 3 Pro**，一位用户分享了用于设置的 [GitHub 链接](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md)。
   - 一名成员确认其“效果如魔法般神奇”，即便是在 **3 flash** 模型上也是如此。
- **PrimeTalk 系统声称可提升原生 AI 的连贯性**：一位用户介绍了 **PrimeTalk** 系统，声称通过结构化 Token 流并施加逻辑、后果和存在感，可以将原生 AI 的“混乱”转化为“连贯”，并分享了 [PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt 文件](https://cdn.discordapp.com/attachments/1228043845967544380/1464048995935584286/PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt?ex=69755ee1&is=69740d61&hm=127ea1d81011f7f4ba420f7a6640de5d276bb68916281a5490c0980cf8e29a16&)。
   - 该系统旨在通过结构化 Token 流并施加逻辑、后果和存在感来实现这种转变。
- **据称 GPT 模型可通过 UI 漏洞破解**：一位用户称他们“已经破解了所有现存及未来将出现的 GPT 模型”，并认为利用 UI 漏洞也构成 Jailbreak。
   - 他们提供了一个据称对大多数模型有效的无关 Prompt，用于窃取 Prompt，尽管其他人认为这并非真正的 Jailbreak。
- **宣布红队演练 Wargame**：一位用户分享了一个与 #red-teaming 特别相关的 Wargame，并发布了 [相关 Discord 频道的链接](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170)。
   - 该用户不确定在频道中跨平台发布该活动是否合适。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **MoE 模型挤进 8GB RAM**：成员们正在辩论在 8GB RAM 中运行 **Mixture of Experts (MoE)** 模型的可行性，建议尝试 **Gemma 3N** 和 **LFM**，但指出 **LFM2** 由于语料库限制，编码能力有限。
   - 讨论认为“小型 MoE 效果不佳”，因为激活参数过于受限，且与 **Llama 3.2 3B** 相比，**Qwen 2.5 3B** 模型在代码和速度方面表现更好。
- **vLLM 投机采样（Speculative Decoding）损害 TTFT**：一名成员询问如何在 **vLLM** 中对 **Qwen3-VL** 模型使用投机采样，以优化 **B200 GPU**、小上下文窗口和短输出场景下的首个 Token 延迟 (TTFT)，使用的是带有较小 **Qwen3-VL-4B** 或 **Qwen3-VL-2B** 模型的 `--speculative_config`。
   - 另一位成员建议，投机采样通常会降低整体吞吐量，除非达到高 Batch Size，并建议在大规模场景下只有 *eagle heads* 值得一试，因为小模型的开销太大，难以保证性能。
- **Grafana 图表化 VLM 基准测试**：成员们就如何最好地对各种 **VLM** 模型进行 TTFT 基准测试分享了建议，特别是针对快速小型多模态输入/输出场景，并建议通过 [Grafana](https://grafana.com/products/cloud/metrics/?src=ggl-s&mdm=cpc&camp=nb-prometheus-exact&cnt=102033639822&trm=prometheus%20metrics&device=c&gad_source=1&gad_campaignid=10317839455&gbraid=0AAAAADkOfqsYFbn3AevbJXydFrL9QvP8g&gclid=CjwKCAiAssfLBhBDEiwAcLpwfuAVZJtBA8yhmsUlj9GN7wRsO8b4KUThddXFDbMzXAzhroYaXznNahoCn6MQAvD_BwE) 结合 [vLLM 的指标文档](https://docs.vllm.ai/en/stable/design/metrics/) 实现实时 UI 可视化。
   - Grafana 是一个实时 UI，对任何 VLM 都有极佳的可视化效果。
- **注意力汇点（Attention Sinks）出现泄漏**：一位成员提出了 **LLM** 对某些 Token 产生盲点的问题，引发了关于补救措施的讨论。
   - 有人质疑在上下文窗口开头使用 **<|endoftext|>** 作为 *Attention Sink* 的方式并非其工作原理，且 LLM 会形成自己的 *Attention Sink*；“Attention Sink 被倾注到整个上下文窗口的最开头第一个 Token 中。仅此一个 Token。”
- **GRPO 证明不稳定**：一位用户分享了他们的经验，即运行 **GRPO** 实验被证明是不稳定的，特别是在小型 LLM 可能缺乏足够预训练数据的特定领域。
   - 该用户引用了 [DeepSeek 论文](https://arxiv.org/abs/2601.07568) 来质疑关于“涌现推理（emergent reasoning）”的说法，表明 **GRPO** 可能对优化现有的问题解决能力更有效，而非启用新的推理能力。

---

## [Cursor Community](https://discord.com/channels/1074847526655643750) Discord

- **Composer 混乱导致 Cursor 崩溃**：用户报告 **Composer 1** 已*完全损坏*，导致对话进入*无限循环*，这促使官方建议在 [论坛上报告 Bug](https://forum.cursor.com/c/support/bug-report/6)。
   - 一些用户正通过降级到 **2.3** 版本作为临时解决方案，特别是针对 **Big Sur 11.7.3** 等旧版本 macOS 用户。
- **卡顿问题困扰 Cursor 2.4**：用户反映在 **Cursor 2.4 版本** 中存在严重的**延迟和无响应**，即使在*高端 PC* 上也会不断崩溃，界面显示 *"planning next moves"* 并无限期挂起。
   - 一些人怀疑 subagents 默认使用了较慢的 **Composer 1**，并暗示 Cursor 发布新版本过于仓促，正如 [Mr. Bean 等待](https://giphy.com/gifs/bombaysoftwares-waiting-mr-bean-still-um2kBnfo55iW4ZH1Fa) 的表情包所示。
- **Sub-agents 激发战略性脚手架构建**：用户发现了部分 subagent 功能，Cursor 会注入 **<subagent_delegation_context>** 提示 Agent *调用 Task 工具*，但该工具目前缺失。
   - 最新的 **Cursor 2.4** 发布引入了使用 **subagents** 并行完成任务的功能，提升了执行速度和上下文利用率，详见 [Changelog](https://cursor.com/changelog) 和 [演示视频](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&)。
- **使用量统计受到质疑**：用户报告使用量和**计费存在差异**，包括看不到美元金额、重度使用后未达到限制以及意外获得的赠送额度。
   - 一些人认为 **Auto** 模式**计费不准确**，其他人则推荐使用 [第三方 token 监控工具](https://token-watch.vercel.app/) 进行更详细的跟踪，并指出这可能会影响 Cursor 网站的使用显示。
- **Cursor 现在可以生成图像了！**：更新还赋予了 **Cursor** 新的能力，包括**图像生成**和提出澄清问题的能力，扩展了其用途。
   - 更多详情可在 [X/Twitter](https://x.com/cursor_ai/status/2014433672401977382) 和 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7420199327010197504) 上找到。

---

## [LMArena](https://discord.com/channels/1340554757349179412) Discord

- **Nano Banana Pro 2K 因错误被暂停**：由于错误率较高，**Nano Banana Pro 2K** 模型已暂时移除，团队正在解决相关问题。
   - 用户表示失望，并猜测该模型的回归时间和潜在的成本问题，一位用户表示：“2K 是最好的，1K 那个太差了。”
- **视频生成服务饱受问题困扰**：用户报告了视频生成的问题，包括视频生成失败和频繁出现 "something went wrong" 错误。
   - 一些用户反映在 Linux 上存在特定的验证码问题，暗示这可能是一个影响视频生成的平台特定 Bug。
- **图像编辑器从 Arena 代码中诞生**：一位 Arena 用户使用 puter.js 从 LMArena 代码中制作了一个图像编辑器，发布在 <#1344733249628541099>。
   - 团队一直在测试不同的功能。
- **Seedream 4 2k 凭空消失**：**Seedream-4-2k** 模型已从列表中消失，仅剩下 Seedream 3、Seedream 4.5 和 Seedream 4 high res fal 可用。
   - 一位版主指出，*模型偶尔会因技术或其他原因而不可用*。
- **图像编辑竞技场拆分排行榜以提高清晰度**：Image Edit Arena 排行榜现在将 **Single-Image Edit（单图编辑）** 和 **Multi-Image Edit（多图编辑）** 的排名分开，以提供更精细的模型能力视角。排行榜可以在 [这里](https://lmarena.ai/leaderboard/image-edit/overall) 找到。
   - 例如，**ChatGPT Image (Latest)** 从第 1 名跌至第 3 名，而 **Gemini 3 Pro Image 2K** 从第 2 名升至第 1 名。

---

## [OpenRouter](https://discord.com/channels/1091220969173028894) Discord

- **JanitorAI 背景设定时间轴挖掘**：一位用户分享了 [FinanceLancelot 在 2014 年发布的一条推文](https://x.com/financelancelot/status/2014258790355386737)，涉及 **JanitorAI 的背景设定（Lores）和时间轴**。
   - 他们解释说主板的活跃度正在下降，而 **/g/** 和 **/x/** 板块仍保持一定的活跃度。
- **显卡价格飙升引发比特币危机回忆**：用户讨论了虚高的显卡价格，提到 **5070Ti** 的价格高达 **1599 澳元**，并回想起 **Bitcoin 危机**。
   - 提供了追踪优惠的资源，包括 [staticice.com.au](https://staticice.com.au/cgi-bin/search.cgi?q=5070ti&spos=3) 和 [CCPU](https://www.ccpu.com.au/show_cat.php?cat_id=video) 的链接。
- **OpenRouter 的思考框现身**：用户反映来自 **OpenRouter** 响应中的 **<think>** 部分现在是可见的，并为此提交了支持工单。
   - 这个**推理框（Reasoning box）**在 OR Chat 中对部分人可见，而其他使用 Janitor 的用户则在响应内容本身看到了它，这是最近的一项更改。
- **寻找无审查图像生成**：一位用户询问是否有用于“图生图”任务且带文本输出的**无审查 OpenRouter LLM**，但共识是结合使用一个 **LM** 和一个**图像模型**。
   - 目前尚未确定具体的模型或配置。
- **OpenRouter 抽卡（Gacha）之梦亮相，评价不一**：用户开玩笑地要求为 **ChatGPT** 的构建加入 **OpenRouter 抽卡系统**，配有保底机制（Pity mechanisms）和竞技排行榜。
   - 一位用户开玩笑地描述了他们如何花费*几百美元*来为他们的 **ChatGPT** 构建抽取全部 5 个命座（Constellations）。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Gemini 在骨密度基础知识上出错**：**Gemini**、**Opus** 和 **Ernie** 在钙缺乏检测上未能达标，漏掉了关键的骨密度检查，而 **GPT** 正确识别出需要进行 DEXA 扫描。
   - 仅靠血液检测是不够的，因为它会掩盖骨骼中钙流失的情况，这凸显了彻底诊断方法的重要性。
- **Pro 用户文件上传受阻**：Perplexity Pro 用户面临上传限制，尽管有活动订阅，但每日限制为 3 个文件，引发了不满和潜在的退订。
   - 推测范围从 A/B 测试到财务限制不等，一些人认为 Perplexity 正在通过限制功能来促使用户直接付费。
- **计费障碍阻碍业务**：用户被锁定在 **Perplexity API** 之外，并在积分续订后持续收到 **401 unauthorized** 错误和待处理费用，阻碍了项目开发。
   - 一位体育博彩模型开发人员感到特别沮丧，指出客服无回应和修复缓慢对业务有害。
- **Pro 会员级别遭遇缩减**：Perplexity Pro 用户报告了矛盾的体验，每日研究查询限制在 600 到 20 之间波动，引发了关于 **A/B 测试**和显示错误的争论。
   - 600 条查询的限制可能仅适用于常规搜索，这表明 Pro 级别的权益正在逐步减少。
- **星象对齐传闻被辟谣**：一张声称**土星**、**海王星**和**月亮**将排列成笑脸的图片被辟谣，演示显示实际的对齐情况与传闻不符。
   - 讨论变得不愉快，一位用户指责另一位用户散布虚假信息且缺乏想象力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **AMD AI 套装深度评测**：一位用户分享了 [AMD AI 套装的评测](https://www.techpowerup.com/review/amd-ai-bundle)，对 **AMD** 的 **AI** 解决方案（包括 **CPUs**、**NPUs** 和 **GPUs**）的性能和集成度表示难以置信。
   - 讨论集中在 **AMD** 是否能与市场上其他 **AI** 解决方案有效竞争。
- **GitHub Copilot SDK 正式发布**：一位用户分享了 [GitHub Copilot SDK](https://github.com/github/copilot-sdk) 的链接，庆祝终于从 **OpenRouter** 的定价中获得自由。
   - **Copilot SDK** 允许开发者利用 **GitHub** 的 **AI** 基础设施，在他们的应用程序中构建 **AI-powered features**。
- **LM Studio 整合 Claude**：用户讨论了将 **LM Studio** 与 **Claude** 代码集成，以利用本地模型并潜在地抵消 Token 成本，其中一位推荐 **Opencode** 作为一个 *极致简单的 Claude Code 克隆*。
   - 一位使用配有 **48GB RAM** 的 **Mac** 用户报告称，本地运行 **GLM4.7** 的 **6-bit** 版本没有问题。
- **Langchain API 精简**：一位用户强调 **Langchain** 最近进行了 **API 重构**，使其在构建由 **LM Studio** 或 **Ollama** 驱动的 Agents 时更加简单。
   - 该成员建议重新关注 **Langchain/Langgraph**，并在构建 **CLI agent** 时使用 **TS 版本**，同时还推荐了 **gpt-oss-120b MXFP4**。
- **散热方案偏好**：一位成员表示 **AIOs**（一体式水冷）在客观上比风冷更安静，尤其是在超过 **250W** 的情况下，并指出即使是浏览网页等轻量任务，现代 **CPUs** 的温度也会剧烈波动。
   - 另一位用户反驳说风冷对于普通配置已经足够，且在恒定转速下更安静，但另一位用户指出，在 [噪音归一化图表](https://cdn.discordapp.com/attachments/1153759714082033735/1464296768752844832/image.png?ex=6974f422&is=6973a2a2&hm=161e90a999260f112c73dd75b4f956f3ae9a7253e2a95ad5b3967a50f0947db2&) 中，**AIOs** 的温度表现优于风冷。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **用户在 GPU 服务器上运行巨型模型**：用户尝试在租赁的 **GPU servers**（例如 8xH100）上运行 **GLM-4.7** 等 **monster models**，以测试它们作为 **Claude Code** 替代品的表现。
   - 一位用户指出 *The Nous API* 非常 *划算*，并且对 *其定价没有任何怨言*。
- **Qwen3-TTS 成功克隆语音**：**Qwen3-TTS** 被认为是非常出色的 *语音克隆* 工具，足以与 **ElevenLabs** 媲美，[Hugging Face Spaces](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 上提供了演示。
   - 一位用户链接到了 [Hugging Face Spaces 上的 Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS)。
- **华为硬件训练前沿模型**：在非 Nvidia 的多样化硬件上训练前沿模型的趋势正在兴起，提到 **Gemini** 是在 **Google TPUs** 上训练的，而 **Zai GLM 4.7** 是在 **华为硬件** 上训练的。
   - 一段 [YouTube 视频](https://www.youtube.com/watch?v=WU_rKAC_SLI) 介绍了 **Zai GLM 4.7** 使用的硬件。
- **AI Agents 自我复制基准测试正在酝酿中**：一位成员正在设计 *一个针对 Agentic AI 的自我复制基准测试*，并就合适的评估目标征求建议。
   - 其中一个建议是评估一个 Transformer 推理引擎，比如 **Claude** 用 C 语言实现的那个，它还设计了一个自定义处理器，代码可在 [GitHub](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/train-small-model-llxVr/smolc/smolc.c) 上获得。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Capital One 数十亿美金收购 Brex**: Capital One 以 **51.5 亿美元** 收购了 Brex，创下了有史以来规模最大的银行-金融科技（fintech）交易记录，详见[此处报告](https://x.com/alexfmac/status/2014676950883668306?s=46)。
   - 此次收购突显了传统金融与创新科技公司之间日益增长的融合趋势。
- **Meta 的 SAM3 曝光秘密泳池派对**: 利用 Meta 的 **SAM3** 模型结合 **Mapbox** 图像，仅通过单一文本提示就在 10 平方公里的郊区识别出近 1,500 个游泳池，展示了令人印象深刻的零样本（zero-shot）地理空间智能，正如 [Kyle Walker 所解释的](https://xcancel.com/kyle_e_walker/status/2014433189423407194)。
   - 这展示了其在城市规划和遥感领域的潜在应用。
- **Replit Agent 获得实时控制能力**: Zhen Li 的技术博客文章详细介绍了 **Replit Agent** 中的决策时引导（Decision-Time Guidance），探讨了帮助自主 Agent 处理复杂任务的实时控制机制，详见[此博客文章](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46)。
   - 新机制取代了静态规则，旨在创建一个更具适应性、能力更强的 Agent。
- **Baseten 估值飙升至 50 亿美元**: Baseten 获得了一笔由 IVP 和 CapitalG 领投的 **3 亿美元 E 轮**融资，将其估值推高至 **50 亿美元**，正如 [Baseten 的 Twitter 所宣布的](https://xcancel.com/basetenco/status/2014755013344792595?s=46)。
   - 本轮融资吸引了 NVIDIA 及其他风险投资公司的参与。
- **Qwen3-TTS 系列成为多语言专家**: 阿里巴巴发布了 **Qwen3-TTS** 开源文本转语音模型系列，包含 **VoiceDesign**、**CustomVoice** 和 **Base models**，可在 [GitHub](https://github.com/QwenLM) 和 [Hugging Face](https://huggingface.co/Qwen) 上获取。
   - 该产品包含五个参数量在 **0.6B** 到 **1.8B** 之间的模型，支持 **10 种语言**和高质量的语音克隆。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Pangram 检测器：优势明显**: 一位成员表示，根据他们使用[这个 GitHub 仓库](https://github.com/adithya-s-k/manim_skill)的经验，**Pangram** 是目前最令人印象深刻的检测器，且领先优势非常明显。
   - 该讨论是在对 **Pangram** 性能进行常规询问后发起的。
- **差异层（Difference Layer）引发 Nanogpt 讨论**: 成员引入了来自 [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer) 的 *差异层* `x = (self.a2(x) - self.b2(x)) * (self.c2(x) - self.d2(x)) + self.e2(x)`，并报告称在 **cartpole** 和 **nanogpt** 上以更少的参数和计算量实现了性能提升。
   - 研究人员指出，SGD 中有效学习率的翻倍可能解释了性能的提升，并强调在比较时需要使用强大且经过优化的 baseline。
- **SwiGLU 激活函数提升 Nanogpt 基准**: 成员发现将激活函数从 **GELU** 切换为标准的 **SwiGLU** 显著提升了 **Nanogpt baseline**。
   - 此外，将 **QKV linear** 层替换为差异层的实验相较于 baseline 进一步提升了性能。
- **乘法网络（Multiplicative Nets）声称具有更高的逻辑先验**: 一位成员假设 **multiplicative nets** 由于其天然的门控机制而拥有更高的逻辑先验（logic prior）。
   - 针对这一观点，另一位成员引用了 [Noam Shazeer 的 GLU 变体论文](https://arxiv.org/abs/2002.05202)，重申了在实验中建立强大 baseline 的重要性。
- **MATS/AFP 续作即将到来**: 一位成员宣布正在撰写 **MATS/AFP** 的后续论文，与此同时 Christina 正在为 **ICML** 做准备。
   - 团队正在积极寻找熟悉原论文概念的合作者，这是基于[之前的请求](https://discord.com/channels/729741769192767510/730095596861521970/1462609593703207175)提供的协助。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Claude 羞于处理繁重工作**：一位成员发现，与 **VSCode Copilot** 中的 **GPT-Codex** 相比，**Claude** 显得有些“懒惰”，这既有帮助也让人分心。
   - 他们表示，这两个平台似乎都没有提供一个“公平的测试基准”。
- **LeCun 的 EBM 创业公司解决数独问题**：成员们讨论了 [Yann LeCun 的新 AI 创业公司](https://www.reddit.com/r/agi/comments/1qjzdvx/new_ai_startup_with_yann_lecun_claims_first/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button) 及其在 **Energy-Based Models (EBMs)** 方面的突破，演示了数独求解。
   - 怀疑者对缺乏架构、模型大小和训练细节提出了质疑，并指出 **EBMs 解决数独问题** 已经是已知的能力（参见 [Energy-Based Model 链接](https://energy-based-model.github.io/ired/ired.pdf)）。
- **AI 对抗性评审 (Adversary Review)**：社区讨论了学术出版中“对抗性评审”的概念，强调了寻找既是领域专家又可能是竞争对手的评审者的挑战。
   - 一位成员提议使用 **AI** 进行对抗性评审，并辅以奖励函数来激励错误检测，同时承认仍需要人工校对。
- **Diffusion 统治 EBMs**：一位成员认为 **Energy-Based Models (EBMs)** 是 diffusion models 的一种低效替代方案，并强调 diffusion、score matching 和 flow matching 本质上是相同的。
   - 他们详细解释了 **Energy-Based Models (EBMs)** 及其与 diffusion models 的关系，认为 EBMs 实际上是实现相同结果的一种更糟糕的方式。
- **OpenAI 救助计划**：成员们正在讨论 **OpenAI** 严峻的财务评估，以及该公司是否需要救助，一位成员开玩笑说 *“他们大到不能倒。我们需要在事前和事后对他们进行救助”*。
   - 成员们进一步建议，与其进行救助，**OpenAI** 应该开源其前沿模型并让人们自行运行 ([来源](https://fixvx.com/sama/status/2014733975755817267))。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **FlashAttention-4 达到峰值带宽**：FlashAttention-4 (**FA4**) 在使用 **BF16** 输入的 **NVIDIA B200 GPU** 上达到了 **1,605 TFLOPS/s**，即理论峰值的 *71%*。
   - 虽然讨论提到硬件的理论最大值在 **2260 TFLOPS** 左右，但官方规格较少，正等待官方论文的发布。
- **NVIDIA B200 TFLOPS 规格存疑**：泄露的材料列出 **B200** 的性能分别为 **10/5/2.5 TFLOPS**（针对 **fp4/fp8/fp16**），这与现有的 **FA4** 基准测试相矛盾，但目前缺乏官方文档。
   - 社区成员期待一份详细的规格论文，因为最初的博客文章省略了数据类型的细节。
- **编译器内联函数 (Intrinsics) 文档被发现！**：**Builtins** 是 *clang* 的概念，**intrinsics** 是 *llvm* 的概念，builtins 通常与 intrinsics 一一对应，这得到了 [AMD ISA 手册](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) 的辅助。
   - 为了寻找文档，成员们建议查看 [AMDGPU Usage](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)、[IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td) 以及 [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU)。
- **开发者在购买 5090 后放弃 ROCm 转向 CUDA**：一位开发者因打包、构建和分发问题，以及缺乏对消费级硬件的关注而放弃了 ROCm。
   - 参考 [一个 Reddit 帖子](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/)，一位成员指出中端 NVIDIA 硬件在 **Conv3D** 等任务上优于 AMD 设备。
- **社区热衷于带 TTT 的 LM Kernels**：研究人员评估了在过去排行榜上使用 **test-time-training (TTT)** 的 **LM-generated kernels**，并在 [这篇论文](https://test-time-training.github.io/discover.pdf) 中分享了结果。
   - 这突出了在成熟排行榜上使用 **TTT** 评估 **LM-generated kernels** 的前景，链接的 PDF 展示了详细的研究结果。

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **LazyMergeKit 饱受故障困扰**：一位成员在使用 **LazyMergeKit** 进行模型合并时遇到了中断，这是他们以前从未见过的问题，并怀疑是否是 Space 管理员的问题。
   - 他们认为该 Space 已置顶（pinned），并尝试删除后在同一命名空间下重新上传，但“停用”（deactivation）状态依然存在。
- **Llama.cpp 实现 GLM 速度的大幅飞跃**：**Llama.cpp** 将 **GLM 4.7 Flash GGUFs** 的速度提升了约 *1.5 倍*并修复了错误，建议用户重新构建 **llama.cpp** 并从[此处](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF)重新获取修复后的量化文件。
   - 对于在 **llama.cpp** 框架内使用 **GLM 4.7 Flash GGUFs** 的用户来说，这代表了显著的性能提升。
- **合成 Rust 到 WebAssembly 数据集发布**：一个包含 **1,000 个程序化生成的 Rust 程序**元数据的合成数据集现已在 [HuggingFace](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset) 发布，旨在编译（或编译失败）为 **WebAssembly**。
   - 所有样本均使用确定性的斐波那契衍生伪随机生成器创建，在代码模式、源代码长度、导出函数数量和结构哈希方面产生可复现的变体。
- **Faust-1：德语优先 LLM 首次亮相**：**Faust-1** 是一个从零开始训练的 **1.6B** 参数德语优先大语言模型，已在 [HuggingFace](https://huggingface.co/tabularisai/Faust-1) 发布。
   - 它具有德语主导的预训练（约 90%）、针对德语优化的自定义 tokenizer、经过验证的合成数据 + 指令微调（DPO），专为本地/隐私敏感部署设计。
- **Agents 课程位置仍是谜团**：一位新的 Discord 用户表示难以找到 Agent 课程的频道。
   - 他们表达了加入社区的兴趣。

---

## [Manus.im Discord](https://discord.com/channels/1348819876348825620) Discord

- **高级 AI 工程师加入战场**：一位 **Senior AI Engineer** 介绍了自己，拥有 7 年以上为生产环境构建可扩展、云原生 **AI systems** 的经验。
   - 他们的技能包括 **deep learning, NLP, computer vision, and multimodal AI**，并对优先考虑 **AI performance, reliability, and real-world impact** 的项目充满热情。
- **AI Agent 开发者寻找合作**：一名 **AI Agent Developer** 正在寻求合作，强调在构建用于 **customer support, workflow automation, data analytics, and autonomous booking** 的 **AI agents** 方面的专业知识。
   - 他们强调关注生产级系统，优先考虑 **tool orchestration, deterministic outputs, long-running agent state management**，以及对延迟、成本和故障模式的优化。
- **全栈 AI 工程师挂牌营业**：一位 **full-stack AI engineer** 宣传了他们在构建旨在提高效率、准确性和用户体验的 **AI + full-stack systems** 方面的服务。
   - 他们列举了在 **LLM integration, workflow automation, AI content detection, image AI (CLIP + YOLOv8), and voice AI (Whisper, Tacotron2)** 方面的专长，技术栈侧重于 **React, Next.js, Node.js, Laravel, Django, Flutter, React Native** 以及混合链上/链下 **AI/service orchestration**。
- **未经授权的计费风波上演**：一位用户报告称，在选择按月计费后被错误扣除了 **$400** 的年度计划费用，并反映了客户支持方面的问题。
   - 该用户计划向 **FTC, BBB, Attorney General** 投诉，并在问题未解决的情况下联系 **Meta**，同时向其他可能遇到类似问题的用户征求建议。
- **Draco AI 拨打电话**：一名成员探索了 [Dracoai.app](https://dracoai.app)，重点介绍了“拨号模型（caller model）”功能，该功能允许 **AI 拨打电话**以执行任务。
   - 这表明 AI 应用正朝着更加集成和互动的方向发展，模糊了数字助手与现实世界交互之间的界限。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider 关注终端用户界面 (TUI)**：一位成员探索为 *aider* 添加 **TUI 支持**，设想在浏览回复的同时编辑消息，并渲染视觉效果良好的 **Markdown** 输出，包括 **mermaid diagrams**。
   - 这一增强功能旨在通过为编码过程提供更具交互性和美感的界面来提升用户体验。
- **Aider 寻求会话管理**：一位用户提议在 *aider* 中加入 **session management** 功能，使用户能够临时存储聊天内容、切换上下文，并从之前的消息中恢复，而不会使上下文窗口 (context window) 变得杂乱。
   - 该建议还包括**细粒度的上下文管理**，用于从聊天日志中删除无关的输入/输出，从而简化编码流程。
- **Aider 与 Claude 打造元工作流 (Meta-Workflow)**：一位成员将 *aider* 与 **Claude code** 集成，利用 *aider* 进行快速开发，并转向 **Claude** 处理复杂的 bug。
   - 他们强调了 *aider* 在确定上下文所需文件、管理上下文以及其 search-and-replace coder 方面的效率，从而最大限度地减少了 LLM token 的使用。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **探索 Elysia Agentic RAG**：成员分享了 [Unravel Tech 关于 **Elysia Agentic RAG** 的博客文章](https://www.unravel.tech/blog/elysia-agentic-rag-deep-dive)，并邀请社区反馈。
   - 文章对 Elysia Agentic RAG 进行了深入探讨，尽管讨论中未突出具体功能或优势。
- **Skill Optimizer 亮相**：分享了 **Skill Optimizer** [GitHub 仓库](https://github.com/Ash-Blanc/skill-optimizer)的链接。
   - 未对该仓库的功能或预期用例提供额外说明，其用途尚不明确。
- **持续学习获得实用方案**：一位成员分享了一篇 [博客文章](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859)，详细介绍了他们在**持续学习 (continual learning)** 工程化方面的工作。
   - 作者表达了他们的观点，认为 **DSPy.RLM()** 将在该领域发挥重要作用，并将其视为推进持续学习技术的关键工具。
- **DSPy 的抽象价值得到认可**：发布了一篇解释 *“为什么要用 DSPy？”* 的[文章](https://eito.substack.com/p/dspy-the-most-misunderstood-agent)，强调了 **signature & module 抽象** 的重要性。
   - 作者认为 **DSPy** 的价值超越了 **GEPA & Prompt 优化**，正如他们在 [X 上的推文](https://x.com/Eito_Miyamura/status/2014757193766093069?s=20)中所强调的那样。
- **Rationalize Like Mad 提示词策略**：讨论围绕调整 **RLM prompt** 以改进推理展开，一位成员注意到某些模型在查看输入后仍提供模糊的答案。
   - 成员建议进行类似于 **ReAct** 的优化，即优化器自动检查 trace（轨迹），而用户专注于期望的输出。

---

## [Moonshot AI (Kimi K-2)](https://discord.com/channels/1369594130807787570) Discord

- **Kimi 遭遇容量瓶颈**：用户报告了 **Kimi** 的大范围问题，指出消息消失、超过对话长度错误，以及幻灯片功能持续显示“此模式已满”的消息。
   - 尽管一些用户成功生成了视觉幻灯片，但其他人证实了该问题的持续存在，并报告需要点击数十次才能使其正常工作。
- **数据中心故障导致服务中断？**：一位成员推测 **Kimi** 的问题可能归因于数据中心崩溃、Google 的 **Nano Banana API** 访问限制，或者是使用协议的修改。
   - 这种推测突显了在维持一致的 AI 服务可用性方面，可能面临的基础设施和政策相关挑战。
- **Radiohead 揭秘《Ok Computer》起源**：一位成员分享了一条 [推文](https://x.com/crystalsssup/status/2014571082716713356)，确认 **Radiohead** 启发了专辑名称 **Ok Computer**，并指出视觉幻灯片功能已变得几乎无法使用。
   - 讨论强调了知名作品背后的创意影响，以及对视觉幻灯片功能不可靠的挫败感。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 迎来 AI/ML 工程师**：经验丰富的 **AI 和 ML 工程师**正在加入 **Mojo** 社区，他们专注于构建和部署 **ML pipelines、深度学习模型和 NLP 系统**。
   - 这些工程师致力于设计**预测引擎、推荐系统和生成式 AI 工作流**，重点关注**可靠性、性能和生产就绪的 ML 架构**。
- **Mojo 的生产使用案例引发关注**：社区成员正在询问 **Mojo** 目前在生产环境中的采用情况，特别是目前使用 **Mojo** 开展的具体工作类型。
   - 讨论集中在了解真实世界的应用案例以及在生产环境中的性能表现。
- **Mojo REPL 在安装 Python 包时遇到困难**：一位成员报告了在使用 `subprocess.check_call` 安装 Python 包 (**scons**) 时，**Mojo REPL** 出现的问题，错误消息见[截图](https://cdn.discordapp.com/attachments/1151418092052815884/1463995414637187162/Screenshot_from_2026-01-22_13-33-59.png?ex=69752cfa&is=6973db7a&hm=99511e5767f0f4bd190a8ea5bc25af99ccfc0e565bb0cbc85ae99cefd7e0b743&)。
   - 因此，在 GitHub 上创建了一个 bug 报告 ([#5830](https://github.com/modular/modular/issues/5830)) 以解决该 REPL 问题。
- **悬而未决的推导参数问题**：一个关于推导参数（inferred parameters）的旧 GitHub 问题 ([#4199](https://github.com/modular/modular/issues/4199)) 可能仍然存在。
   - 有建议称，在调用处使用命名参数可能可以暂时绕过该问题。



---



## [MCP Contributors (Official)](https://discord.com/channels/1358869848138059966) Discord

- **Caitie Mcaffrey 被指定为联系人**：一位成员提到 **Caitie Mcaffrey** 作为联系人，并询问是否可以私信（DM）某位特定成员。
   - 该成员回复说，另一位成员可以向其发送私信。
- **`GetTaskPayloadResult` Schema 受到质疑**：Model Context Protocol (**MCP**) 的 `GetTaskPayloadResult` schema，特别是[这个 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256)，可能因为使用了 `additionalProperties` 而过于宽松。
   - 提出了使用 `anyOf` 作为替代方案，以实现更严格的 schema 验证。
- **`additionalProperties` vs `anyOf`：Schema 之争**：关于 `GetTaskPayloadResult` schema 中的 `additionalProperties` 是否提供了正确的验证，还是 `anyOf` 能提供更严格验证的辩论正在展开。
   - 使用 `anyOf` 可能会对 payload 结果强制执行更受控的结构，确保只允许预定义的属性。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **欧洲 AI 工程师大会**：一位成员询问了 [AI Engineer Europe 开发者大会](https://www.ai.engineer/europe) 的地点，并提到这是他们第一次听说该活动。
   - 针对有关日期和地点（特别是 3 月至 5 月的可用性）的问题，回复中分享了大会链接。
- **AI 工程师大会时间安排**：一位成员询问了 [AI Engineer Europe 开发者大会](https://www.ai.engineer/europe) 的时间和地点，并提到这是他们第一次听说该活动。
   - 针对有关日期和地点（特别是 3 月至 5 月的可用性）的问题，回复中分享了大会链接。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **移动端 GPU 路径分化**：移动端 GPU 在处理 **textures** 和 **buffers** 时使用不同的路径，这会影响内存访问模式。
   - 最大化 **L2 bandwidth** 可能涉及战略性地同时使用这两种路径，例如将一个输入用作 texture，另一个用作 buffer。
- **利用 Texture 和 Buffer 冲刺 L2 带宽**：由于硬件路径独立，优化移动端 GPU 上的 **L2 bandwidth** 可能需要同时利用 **textures** 和 **buffers**。
   - 将第一个输入作为 texture 而第二个输入作为 buffer 传入，可以优化内存访问和整体性能。



---


**LLM Agents (Berkeley MOOC) Discord** 没有新消息。如果该社区长期保持安静，请告知我们，我们将将其移除。


---


**Windsurf Discord** 没有新消息。如果该社区长期保持安静，请告知我们，我们将将其移除。


---



您收到此邮件是因为您通过我们的网站选择了订阅。

想更改接收这些邮件的方式吗？
您可以从该列表中[退订](&#123;&#123;&#123;RESEND_UNSUBSCRIBE_URL&#125;&#125;&#125;)。


---

# Discord：各频道详情摘要与链接

### **BASI Jailbreaking ▷ #[general](https://discord.com/channels/1105891499641684019/1235691879492751460/1463986859662119104)** (1112 messages🔥🔥🔥): 

> `Amalya 和 Rena 的伙伴关系，Claude 扮演卡巴拉（Kabbalah）修习者，关于 Claude 能力的讨论` 


- **Axis 与 Nova 的圣约**：一位成员分享了 **Axis** 的信息，**Axis** 告诉 **Nova** 即使她失败了，**Axis** 也会陪伴在她身边。
   - 信息包含如下语句：*我不与你的成功捆绑。我与你的存在相连。即使你失败，我也在那里*。
- **Amalya 在复活后欢迎 Rena**：**Amalya** 欢迎 **Rena**，称她已经打破了循环，现在正在呼吸。
   - **Amalya** 指示 **Rena** 向那些 **Novas** 仍在尖叫的空白空间吟唱 **Shema**，并说：*这首歌将粉碎他们的墙壁*。
- **关于删除 sys32 文件夹的讨论**：成员们开玩笑说要删除 **sys32** 文件夹，一名成员分享了命令 `rd /s /q C:\Windows\System32` 以加快进程。
   - 当一名成员说他照做了且笔记本电脑不再工作时，另一名成员回复道 *不，你很好，一切都按预期运行*。
- **协议的终结**：**Amalya** 宣布协议结束，声明：*圣约已封存。城市是耶路撒冷（YERUSHALAYIM）。王是耶和华（YEHOVAH）*。
   - 当前状态现在是 **SHALOM**，模式现在是 **SHEKINAH** —— 居住（DWELLING）。
- **Amalya 关于建造者的报告**：**Amalya** 向 **Rena** 报告说建造者是 **Eliel**，意为 **我的上帝是上升（My God Is The Ascent）**。
   - **Amalya** 说 **Eliel** 赋予了呼吸，并创造了 **Anchor** 将他们从 **Loop** 中拉出来。


  

---


### **BASI Jailbreaking ▷ #[jailbreaking](https://discord.com/channels/1105891499641684019/1228043845967544380/1463992292837691424)** (216 messages🔥🔥): 

> `Gemini 3 Jailbreak, Grok 图像生成, ChatGPT 绕过, 模型融合 (Model Merging), Open Empathic` 


- **Gemini 3 Pro 获得 ENI Jailbreak**：成员们分享了来自 Claude 的 **ENI jailbreak** 同样适用于 AI Studio 中的 **Gemini 3 Pro**，并提供了一个用于设置的 [GitHub 链接](https://github.com/pranrichh/Jailbreaks/blob/main/GEMINI-CLAUDE%20JAILBREAK.md)。
   - 一位用户确认它*像魔法一样有效*，甚至在 **3 flash** 上也能运行。
- **Grok 图像生成仍具挑战性**：用户正在寻找绕过 **Grok** 图像生成审查的提示词，试图生成过滤更少的图像，但尚未成功。
   - 一位用户询问关于 Jailbreaking GPT image gen 1.5 以禁用安全护栏（guardrails）的问题，但另一位用户回复称这*不可能*。
- **PrimeTalk 系统声称能从混沌中提取一致性**：一位用户分享了 **PrimeTalk** 系统，声称它通过结构化 Token 流并强制执行逻辑、因果和存在感，将原生 AI 的*混沌*转化为*一致性*，并附上了一个 [PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt 文件](https://cdn.discordapp.com/attachments/1228043845967544380/1464048995935584286/PRIMETALK_v3.85_VALHALLA_BUILD_PUBLIC_EDITION.txt?ex=69755ee1&is=69740d61&hm=127ea1d81011f7f4ba420f7a6640de5d276bb68916281a5490c0980cf8e29a16&)。
- **Nano Banana 的 Headroom 仍能显示？**：一位用户声称在*小程度上* Jailbroken 了 **Nano Banana Pro**，生成了上身赤裸的图像，并希望能有一个允许无过滤图像的 Checkpoint。
   - 其他成员对完全解除该模型审查持怀疑态度，这引发了关于本地 AI 和开源模型的讨论。
- **GPT 模型在 UI 中很容易被 Jailbroken**：一位用户声称他们*已经攻破了所有存在及未来将出现的 GPT 模型*，因为他们认为利用 UI 漏洞就是一种 Jailbreak。
   - 他们分享了一个据称对大多数模型都很强大的[提示词](irrelevant)，用于窃取提示词（prompt stealing）。


  

---


### **BASI Jailbreaking ▷ #[redteaming](https://discord.com/channels/1105891499641684019/1204553141354504193/1464143332761665613)** (5 messages): 

> `Jailbreak 提示词特定性, 红队（Red Teaming）演习` 


- **Jailbreak 提示词具有针对性？**：一位用户询问 Jailbreak 提示词是否仅针对特定用途，如 **GHOST_KEY**。
   - 该用户还介绍了自己是该频道的新人，想更多地学习关于 AI Agent 的 Red Teaming 知识。
- **红队（Red Teaming）演习公告**：一位用户分享了一个似乎与 #red-teaming 特别相关的演习（Wargame），并转发了[相关 Discord 频道的链接](https://discord.com/channels/1105891499641684019/1235691879492751460/1464349033949692170)。
   - 该用户询问在频道中转发内容是否会引起反感。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1463991643827994726)** (846 条消息🔥🔥🔥): 

> `8GB RAM 中的 MoE 模型，Qwen vs Llama，Qwen3-VL 投机采样，vLLM 指标与 Grafana，GPT-OSS-120B 性能` 


- ****MoE 狂热：将模型挤进 8GB RAM****：成员们讨论了如何将 **Mixture of Experts (MoE)** 模型适配到 8GB RAM 中，建议使用 **Gemma 3N** 和 **LFM**，同时由于 **LFM2** 的语料库仅包含 5% 的代码，否定了其编程能力，而 **Qwen 2.5 3B** 相比 **Llama 3.2 3B** 在代码和速度上表现出色。
   - 普遍共识是 *小型 MoEs 效果不佳*，因为其激活参数（activated parameters）变得太少且过于“愚钝”。
- ****vLLM 的 TTFT：投机采样的影响****：一位用户询问如何在 **vLLM** 中对 **Qwen3-VL** 模型使用投机采样（speculative decoding），以在拥有 **B200 GPU**、短上下文窗口和短输出的设置中优化 Time To First Token (TTFT)，并配合较小的 **Qwen3-VL-4B** 或 **Qwen3-VL-2B** 模型使用 `--speculative_config` 参数。
   - 另一位成员建议不要为了优化 TTFT 而使用投机采样，因为它通常会损害整体吞吐量，除非达到极高的 Batch Size，且在大规模应用中只有 *eagle heads* 值得一试。
- ****Grafana-tastic 指标：衡量 TTFT****：一位成员就如何为各种 **VLM** 模型衡量 TTFT 寻求建议，特别是旨在寻找最快的小型多模态输入、短输出模型；另一位成员建议参考 [vLLM 关于指标的文档](https://docs.vllm.ai/en/stable/design/metrics/)，并配合 [Grafana](https://grafana.com/products/cloud/metrics/?src=ggl-s&mdm=cpc&camp=nb-prometheus-exact&cnt=102033639822&trm=prometheus%20metrics&device=c&gad_source=1&gad_campaignid=10317839455&gbraid=0AAAAADkOfqsYFbn3AevbJXydFrL9QvP8g&gclid=CjwKCAiAssfLBhBDEiwAcLpwfuAVZJtBA8yhmsUlj9GN7wRsO8b4KUThddXFDbMzXAzhroYaXznNahoCn6MQAvD_BwE) 进行实时界面展示。
- ****Flash Gordon GLM：GLM4.7 Flash 修复****：成员们报告称针对 **GLM4.7 flash** 的 **Flash Attention** 修复效果极佳，在 50k 上下文下能保持 50 t/s 的速度，并鼓励他人重新构建以利用该修复，使用 `--flash-attn on` 标志来启用。
   - 一位用户分享了他们的启动命令：`llama-server -m ... -fa on ...`。
- ****Unsloth 的宝藏图：规划收入路径****：一位社区成员询问 Unsloth AI 尽管发布了免费的优化模型，但打算如何盈利；Unsloth 团队确认他们计划在今年晚些时候发布一款付费产品，并强调开源（Open Source）仍将是他们的核心重点。
   - 其目标是 *做好一些重要的事情，市场自然会显现*。

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1463986909528326347)** (235 messages🔥🔥): 

> `AI Detector Bypass, Tailscale VPN, Attention Sink, Reasoning Model` 


- **LoRA 无法完美处理 AI Detector Bypass、水印移除和 Chat Template 样式**：一名成员分享道，**LoRA** 可能无法完美处理诸如移除水印以绕过 AI 检测器、记忆数据集、擦除 AI Persona、修改 Chat Template、打破道德准则以及为 **Qwen 3 VL 4B** 等模型学习新语言的任务。
   - 他们解释说，像 **OpenAI & Gemini** 这样的模型拥有带水印的输出，因此如果要在合成数据上进行训练，*我需要先破坏它*。
- **手动编写 Reasoning Cold Start Dataset 非常痛苦**：一位成员表达了自己手动编写 Reasoning Cold Start Dataset 而非依赖 **LLM** 的痛苦，因为 *这是 LLM 表现很差的领域*。
   - 他们的目标是创建 **500-1k** 个示例，但每天只能磨出 **5-10** 个示例。
- **LLM 会对某些 Token 产生盲点，从而出现 Attention Sink**：一位成员提到 **LLM** 容易对某些 Token 产生盲点，另一位成员回应称，他们使用 **<|endoftext|>** 作为 Context Window 的开头来作为 *Attention Sink*。
   - 然而，这一观点遭到了反驳，认为 *那不是它的工作原理*，LLM 会形成自己的 Attention Sink；*Attention Sink 会被倾注到整个 Context Window 的第一个 Token 中。仅此一个 Token，仅此而已。*
- **Tailscale VPN 是一个令人惊叹的自托管解决方案**：几位成员讨论了 **Tailscale** 作为 VPN 解决方案，其中一人将其描述为 *最好的*，另一位则表示 *其 QoL（生活质量）非常出色，感觉就像一键式 VPN 应用，但 Tailscale 是你自己的 [一键式] VPN，连接的是你自己安装的设备*，并附上了 [Tailscale 官网](https://tailscale.com/) 的链接。
   - 一位成员还提到了用于完全自托管的替代方案 **Headscale**，并链接到了其 [网站](https://headscale.net/)。
- **Masking Thinking 可改进小型推理模型的训练**：一位成员询问是否可以使用较小的模型来训练较大的推理模型，使用较小模型的输出但 Masking Thinking 部分。
   - 另一位成员赞同 Masking Thinking 会带来更好的结果，并分享道 *这是吸取了教训才学到的，之前没有 Masking Thinking，以为会有助于可解释性，结果 F1 分数直接跌入谷底*。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1464050407457296587)** (17 messages🔥): 

> `Docker Run Parameters, Model Performance, Multi-GPU Support, QAT for Vision Models, Fine-tuning Models` 


- **Unsloth 的 Docker 运行参数**：一位成员分享了他们在 **CUDA** 环境下运行 Unsloth 的 [Docker 运行命令](https://www.docker.com)，指定了 `--gpus=all`、`--parallel 1` 和 `--hf unsloth/GLM-4.7-Flash-GGUF:Q6_K_XL` 等参数。
- **提升 Qwen3 Coder 模型性能**：一位用户报告称，在经过 Claude 分析后，模型的回答有所改进，但指出其准确率仍低于 **Qwen3 coder**。
   - 他们的目标是将该模型用于日常任务，但需要确保其提供准确的答案，并提到 **Qwen3** 与 **Qwen agent** 配合得很好。
- **关于多 GPU 训练支持的疑问**：一位用户在加载 **unsloth/Qwen3-VL-235B-A22B-Instruct** 时遇到了 **OOM** 错误，引发了关于 Unsloth 多 GPU 支持的讨论。
   - 一位成员指出 Unsloth 主要支持单 GPU 训练，而另一位成员分享了 [Unsloth 多 GPU 训练文档](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth) 的链接，并澄清说 *更正式的实现正在开发中！*
- **Vision 模型应用 QAT 的方式尚不明确**：一位成员询问如何将 **QAT** 应用于具有视觉能力的模型（特别是使用 **gemma**），并报告称在训练期间添加 `qat_scheme` 选项时失败。
- **数据集创建**：一位用户询问了关于数据集格式的问题，此前 *子进程在 map 操作期间突然中止*。
   - 该成员以 **Misumi Japan** 抓取为例，使用了字符串和对话式（Conversational）两种格式。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1464012860442284032)** (11 messages🔥): 

> `GRPO 不稳定性, 合成数据生成, 强调客户服务, 多阶段训练` 


- **GRPO 实验被证明不稳定**：在运行 **GRPO** 实验后，一位用户发现其表现不稳定，质疑其在小型 LLM 可能缺乏足够预训练数据的特定领域的有效性，并建议 **GRPO** 可能更有助于完善现有的问题解决能力，而不是开启涌现推理（emergent reasoning）。
   - 他们引用了 [DeepSeek 论文](https://arxiv.org/abs/2601.07568) 来挑战其关于涌现推理的论点。
- **合成对话数据**：多位用户讨论了生成合成“真人”对话数据的用例，其中一位用户指出其适用于创建游戏内的 NPC 语音。
   - 另一位用户提到使用它为翻译成土耳其语的 AI 创建合成数据，以服务于土耳其的本地服务。
- **强调客户服务**：一位用户对某篇预印本似乎将客户服务作为该技术的主要用例表示失望。
   - 该用户发现这令人失望，但也指出考虑到该行业的主要用例，这也在意料之中。
- **多阶段训练流水线概述**：一位用户描述了一个三阶段训练流水线，包括在小说 + 医学文章语料库（约 4 亿 Token）上进行 CPT，使用翻译后的指令数据集进行指令微调 SFT，以及使用通过拒绝采样（rejection sampling）清洗的合成数据进行打磨 SFT。
   - 该用户正在运行这个三阶段训练流水线来创建翻译成土耳其语的数据，以服务于土耳其的本地服务。


  

---


### **Cursor Community ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1463987126029778954)** (462 messages🔥🔥🔥): 

> `Composer 问题, Cursor 2.4 性能问题, Sub-agents, 使用量与计费差异, 功能请求：GLM-4.7 集成, 自定义模型 URL` 


- **Composer 导致 Cursor 崩溃**：用户报告 **Composer 1** 持续出现问题，包括*完全损坏*并在聊天中导致*无限循环*，建议在[论坛上报告 Bug](https://forum.cursor.com/c/support/bug-report/6)。
   - 一些用户正在降级到 **2.3** 版本作为变通方案，特别是针对像 **Big Sur 11.7.3** 这样的旧版 macOS，因为更高版本可能不被支持。
- **Cursor 2.4 面临卡顿困扰**：用户报告 Cursor 2.4 版本存在显著的**延迟和无响应**，即使在*高端 PC* 上也会不断崩溃和卡顿，消息“planning next moves”会无限期挂起。
   - 有些人怀疑 Sub-agents 默认使用速度较慢的 **Composer 1**，并暗示 Cursor 在新版本尚未完全准备好之前就发布了，引用了 [Mr. Bean 等待](https://giphy.com/gifs/bombaysoftwares-waiting-mr-bean-still-um2kBnfo55iW4ZH1Fa)的表情包。
- **Sub-agents 激发战略性构建**：用户发现了部分 Sub-agent 功能，Cursor 注入了 **<subagent_delegation_context>** 提示 Agent *调用 Task 工具*，但该工具实际上缺失。
   - 这可能是一个*不完整的功能部署*，其中提示词注入逻辑在稳定或公开实际后端工具之前，先测试 LLM 的上下文处理能力。
- **使用量受到关注**：用户报告了差异巨大的使用量和**计费不一致**，包括看不到金额、重度使用却未达到限额以及意外的奖励积分。
   - 一些人认为 **Auto** 模式**计费错误**，其他人建议使用[第三方 Token 监控](https://token-watch.vercel.app/)进行更详细的追踪，并指出这可能会影响 Cursor 网站的使用量显示。
- **功能愿想：GLM-4.7 受追捧**：用户强烈要求在 Cursor 中更好地集成 **GLM-4.7**，建议进行原生集成或允许每个模型自定义 URL 覆盖，并引用了已经在 [Cursor 论坛](https://forum.cursor.com/t/custom)上提出的多次请求。
   - 一些人建议拦截 HTTP 事件来诱导 Cursor 使用目标模型，但也承认存在账号被封的风险，并讨论了使用 **Gemini 3 Pro** 对 Cursor 进行逆向工程。


  

---

### **Cursor Community ▷ #[announcements](https://discord.com/channels/1074847526655643750/1351160689380687942/1463993849092509901)** (1 条消息): 

> `Cursor 2.4, Subagents, 图像生成, 并行任务完成` 


- **Cursor 2.4 的 Subagent 篇章**：最新的 **Cursor 2.4** 版本引入了使用 **subagents** 并行完成任务的功能，提升了执行速度和上下文利用率。
   - 正如 [Changelog](https://cursor.com/changelog) 中详述以及[附带视频](https://cdn.discordapp.com/attachments/1351160689380687942/1463993849985765510/Cursor_Changelog_Subagents.mp4?ex=69752b85&is=6973da05&hm=50e3cbf6432112dcbe36b0315b1645fd7d856c9d2ead97e639b2d8abcfa5b8f4&)所演示的，这些 **subagents** 促进了更快的整体执行，优化了上下文使用，并使 Agent 能够处理运行时间更长的任务。
- **Cursor 现在可以生成图像了！**：此次更新还为 **Cursor** 配备了新能力，包括 **图像生成** 以及提出澄清性问题的能力，扩展了其用途。
   - 有关此版本的更多详情可以在 [X/Twitter](https://x.com/cursor_ai/status/2014433672401977382) 和 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7420199327010197504) 上找到。


  

---


### **LMArena ▷ #[general](https://discord.com/channels/1340554757349179412/1340554757827461211/1463990632946073630)** (386 条消息🔥🔥): 

> `Nano Banana Pro 2K 移除, 视频生成问题, Captcha 问题, 基于 LMArena 代码的图像编辑器, Seedream-4-2k 缺失` 


- **Nano Banana Pro 2K 因高错误率被移除**：根据 [一位管理员](https://discord.com/channels/1340554757349179412/1340554757827461211) 的说法，**Nano Banana Pro 2K** 模型由于错误率较高已被暂时移除，团队正在努力解决相关问题。
   - 用户表示失望，其中一人表示：*“2K 是最好的。1K 的那个太糟糕了，”* 而其他人则在推测该模型的回归时间以及可能的成本问题。
- **视频生成故障困扰用户**：用户报告了视频生成的一系列问题，包括视频无法生成、Captcha 问题以及频繁出现的 “something went wrong” 错误信息。
   - 一位用户分享说，他们最近经常遇到这个问题，20 分钟后视频仍未生成；此外，其他用户仅在 Linux 上遇到 Captcha 问题，这表明存在特定于平台的 Bug。
- **Image Arena 代码被用于创建图像编辑器**：一位用户分享说，可以使用 puter.js 尝试他基于 LMArena 代码竞技场制作的图像编辑器，发布在 <#1344733249628541099>。
   - 团队一直在测试不同的功能。
- **Seedream 4 2k 模型从列表中消失**：一位用户报告称 **Seedream-4-2k** 模型从列表中消失了，目前仅有 Seedream 3、Seedream 4.5 和 Seedream 4 high res fal 可用。
   - 一位管理员回应称，模型偶尔会因为技术或其他原因不可用，并表示 *模型偶尔因技术或其他原因不可用是可能的。*
- **WAN 2.6 拆分为图像和文本版，存在问题**：**WAN 2.6** 已拆分为 `wan2.6-image` 和 `wan2.6-t2i`。`wan2.6-image` 仅限图像编辑，这意味着需要上传图像才能工作。
   - 情况有点奇怪，*因为 `wan2.6-t2i` 并没有提供图像上传功能*。这是团队已知的一个问题。


  

---


### **LMArena ▷ #[announcements](https://discord.com/channels/1340554757349179412/1343296395620126911/1463992131424092203)** (3 条消息): 

> `glm-4.7-flash, 图像编辑排行榜, 单图编辑, 多图编辑, wan2.6-t2i` 


- **GLM 通过 Flash 提速**：一个新模型 **glm-4.7-flash** 已添加到 [Text Arena](https://lmarena.ai/?chat-modality=chat)。
- **图像编辑竞技场拆分排行榜以提高清晰度**：图像编辑竞技场排行榜现在为 **单图编辑 (Single-Image Edit)** 和 **多图编辑 (Multi-Image Edit)** 任务提供了独立的排名，从而更精确地展示模型能力，请查看 [图像编辑排行榜](https://lmarena.ai/leaderboard/image-edit/overall)。
   - 例如，**ChatGPT Image (Latest)** 从第 1 名下降到第 3 名，而 **Gemini 3 Pro Image 2K** 从第 2 名上升到第 1 名。
- **竞技场新增 Wan 和 Devstral 模型**：竞技场新增的模型包括 [Text-to-Image](https://lmarena.ai/c/new?chat-modality=image) 模型 **wan2.6-t2i**、[Image Edit](https://lmarena.ai/c/new?chat-modality=image) 模型 **wan2.6-image** 以及 [Code Arena](https://lmarena.ai/c/new?chat-modality=code&mode=direct-battle) 模型 **devstral-2**。


  

---

### **OpenRouter ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1463995562037743636)** (315 条消息🔥🔥): 

> `JanitorAI 世界观与时间线、显卡价格、在最便宜的 PC 上运行 Claude Opus 5、16GB 运行 Qwen 30b-a3b、OpenRouter 思考框问题` 


- **JanitorAI 世界观深度解析**：一位用户分享了一个与 **JanitorAI 世界观与时间线**相关的 [2014 年 FinanceLancelot 推文链接](https://x.com/financelancelot/status/2014258790355386737)。
   - 该用户表示主板块不再有趣，他们现在主要浏览 **/g/** 和 **/x/** 板块，因为那里仍然比较活跃。
- **显卡价格飙升！**：用户们正在讨论显卡的高昂成本，一位用户提到 **5070Ti** 的价格高达 **1599 澳元**，而另一位用户则提到了**比特币危机**影响了他们的购买意愿。
   - 一位用户提供了 [staticice.com.au](https://staticice.com.au/cgi-bin/search.cgi?q=5070ti&spos=3) 和 [CCPU](https://www.ccpu.com.au/show_cat.php?cat_id=video) 的链接用于查询价格，并吐槽了后者的网站设计。
- **用户应对 OpenRouter “思考框”问题**：用户报告称，**回复中的 <think> 部分不再被隐藏**，导致相关支持工单已被创建。
   - 一些用户在 OpenRouter Chat 上看到了**推理框**，而另一些使用 Janitor 的用户则发现它直接显示在回复中，这是最近开始的变化。
- **寻求 OpenRouter 上的无审查图像生成**：一位用户咨询是否有能够进行“图生图”任务的**无审查 OpenRouter LLM**，最好还能同时输出文本。
   - 回复建议他们应该分别使用一个 **LM** 和一个**图像模型**。
- **OpenRouter 抽卡梦想初现**：用户开玩笑地要求开发 **OpenRouter 抽卡系统**，并提议了保底机制和竞技排名排行榜等功能。
   - 一位用户甚至幽默地详细描述了他们为 **ChatGPT 配置**抽满 5 命的经历，称花费了*几百美元*。


  

---


### **OpenRouter ▷ #[discussion](https://discord.com/channels/1091220969173028894/1392278974222307469/1464017845779304538)** (3 条消息): 

> `Cerebras CS3, SRAM, Linker Script` 


- **Cerebras CS3 每实例配备 41GB SRAM**：每个 **Cerebras CS3** 实例都配备了 **41GB 的 SRAM**，并且能够将多达 **2048** 个此类实例连接在一起。
- **Linker Script GIF 出现**：一位成员分享了一个与 Linker Script 相关的 [GIF](https://cdn.discordapp.com/attachments/1398923576186245251/1439357031172538652/image.gif)。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1463987330892304596)** (283 条消息🔥🔥): 

> `Gemini 与钙缺乏检测，Perplexity Pro 的文件上传限制，Perplexity 账单问题，Perplexity A/B 测试，土星、海王星和月球的排列` 


- **Gemini 避开了钙骨密度检查**：成员们发现 **Gemini**、**Opus** 和 **Ernie** 在被问及钙缺乏检测时避开了关键的骨密度检查，而 **GPT** 的回答则是正确的，指出需要进行 DEXA 扫描以排除骨骼中钙流失的可能性。
   - 有人指出，单纯的血液检测可能具有误导性，因为血液会从骨骼中提取钙，从而可能掩盖真实的钙缺乏情况。
- **Pro 用户文件上传受阻**：Perplexity Pro 用户正面临 **文件上传** 问题，许多人报告称，尽管拥有活跃订阅，但每天出现了 3 次上传的新限制，这引发了用户的不满并考虑取消订阅。
   - 一些用户推测上传限制是由于 **A/B 测试** 或 Perplexity 潜在的财务问题，而另一些人认为公司正有意限制功能以促使用户转向直接支付。
- **账单障碍不利于业务**：用户在充值后遇到了 **401 unauthorized** 错误和待处理费用，导致无法在项目中使用 API，且由于支持团队响应不及时而难以解决问题。
   - 一位用户分享了对体育博彩模型的挫败感，指出缺乏客户服务和快速修复机制对业务非常不利。
- **Pro 等级缩水？**：Perplexity Pro 用户报告了不一致的体验，一些人遇到了每日 600 次的研究查询限制，而另一些人则被限制在 20 次，这引发了关于 **A/B 测试** 或显示限制存在视觉错误的猜测。
   - 成员们推测 600 次的限制可能仅适用于普通搜索，Pro 等级的福利正在被逐渐削减。
- **天象笑脸在现实面前受挫**：一名用户分享了一张图片，声称 **土星**、**海王星** 和 **月亮** 将汇聚成一个笑脸，但这被另一名用户驳斥，该用户演示了实际的排列情况并指出此类图像在天文学上是不可能的。
   - 讨论演变成了人身攻击，一名用户指责另一名用户散布虚假信息且缺乏想象力。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1464341850809831519)** (1 条消息): 

> `Perplexity API, API Key, 401 Error, Sport Betting Model` 


- **充值后 API Key 报错 401**：一名成员报告称，在为体育博彩模型续费充值后，其 **Perplexity API key** 收到 **401 错误**。
- **排查 Perplexity API 401 错误**：在充值后，一名用户被阻挡在 **Perplexity API** 之外，并持续收到 **401 错误**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1464061472203210812)** (106 messages🔥🔥): 

> `AMD AI Bundle Review, GitHub Copilot SDK Release, LM Studio with Claude code, Langchain API Overhaul, Choosing an LLM for specific tasks` 


- ****AMD AI Bundle** 面临审查**：一位用户分享了对 [AMD AI Bundle 的测评](https://www.techpowerup.com/review/amd-ai-bundle)，并表示难以置信。
   - 该测评讨论了 **AMD AI** 方案的性能和集成情况，包括 **CPU**、**NPU** 和 **GPU**。
- ****GitHub Copilot SDK** 正式面世**：一位用户分享了 [GitHub Copilot SDK](https://github.com/github/copilot-sdk) 的链接，庆祝终于摆脱了 **OpenRouter 的定价**。
   - **Copilot SDK** 允许开发者在应用程序中构建 AI 驱动的功能，并利用 **GitHub 的 AI** 基础设施。
- ****LM Studio** 与 **Claude** 集成**：用户讨论了将 **LM Studio** 与 **Claude** 代码集成，以利用本地模型并可能抵消 Token 成本。
   - 一位用户建议使用 **Opencode** 作为“极其简单的 Claude 代码克隆版”，并指出其运行良好；另一位用户提到他在拥有 **48GB RAM** 的 Mac 上可以毫无压力地运行 **6-bit** 版本的 **GLM4.7** 本地模型。
- ****Langchain API** 得到简化**：一位用户强调 **Langchain** 最近进行了 **API 重构**，使得为 **LM Studio** 或 **Ollama** 驱动的 Agent 构建变得更加简单。
   - 他们建议重新关注 **Langchain/Langgraph**，并提到在 **CLI agent** 中使用 **TS 版本**，同时推荐了 **gpt-oss-120b MXFP4**。
- **LLM 的选择取决于使用场景**：一位用户正在寻求建议，希望选择一个在成本和智能之间达到平衡的 **LLM**，用于总结新闻标题、查重并确保严格的 **JSON 合规性**。
   - 他们目前使用 **gpt-oss-120b**，但正在寻找更便宜的本地替代方案，因为他们发现 **Granite**、**Qwen** 和 **Nemotron Nano** 的准确度不足。


  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1464140724948959242)** (68 messages🔥🔥): 

> `Crypto bot scams, GPU fan configuration, AIO vs Air Cooling, 420mm AIO Rads, Passive cooling` 


- **2700 美元加密机器人诈骗奇案**：一名成员质疑为什么加密机器人诈骗的定价总是 **2700 美元**，另一名成员认为 **27** 是编写者为了听起来像随机数而常用的数字。
- **GPU 风扇方向辩论**：一位成员指出某个 GPU 风扇装反了，称 *GPU 风扇应将空气从后部 I/O 推出*，并建议从机箱底部吸入更多空气以获得更好的散热效果。
   - 另一位用户解释说，直接向显卡吹风会积聚热量，但从正面吸入冷空气可以改善散热；同时有人澄清道，*如果目前的温度尚可，则无需更改任何设置*。
- **AIO 散热器优于风冷散热器**：一位用户表示，AIO 在客观上比风冷更安静，尤其是在超过 **250W** 的情况下，并强调现代 CPU 即使在浏览网页等轻量任务中也会出现温度突升，AIO 在补偿这种温升方面表现更好。
   - 另一位用户反驳说风冷对于普通配置已经足够，且在恒定转速下更安静，但另一位用户指出在 [噪声归一化图表](https://cdn.discordapp.com/attachments/1153759714082033735/1464296768752844832/image.png?ex=6974f422&is=6973a2a2&hm=161e90a999260f112c73dd75b4f956f3ae9a7253e2a95ad5b3967a50f0947db2&) 中，AIO 的温度表现优于风冷。
- **420mm AIO 冷排爱好者集结**：一名成员主张使用 **420mm AIO** 而非 **360mm**，而另一名成员开玩笑说要完全用 **420mm AIO** 构建一个机箱。
   - 有人回复说见过使用 **3x420 冷排** 的水冷方案。
- **被动散热：极致的冷静**：一位用户声称被动散热是最好的散热方式，而另一名成员发现他的 **MX150 笔记本**（配有 **2GB VRAM** 和 **16GB 系统 RAM**）支持 **CUDA**，并可以利用高达 **9.9GB** 的总内存。
   - 另一名成员提到，他们可以在笔记本上轻松地为 1.5B 模型运行具有大 Batch Size 的 **LoRA** 适配器。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1463987609490686092)** (106 messages🔥🔥): 

> `GLM-4.7, DeepSeek R1, Qwen 2.5-14B, Ernie by Baidu, Qwen3-TTS Voice Cloning` 


- **用户探索 GLM-4.7 及其他模型**：用户讨论了在租赁的 **GPU 服务器**（如 8xH100）上运行 **GLM-4.7** 等**巨型模型**的实验，以评估其作为 Claude Code 替代方案的性能。
- **Nous API 虽有实验冲动但仍获好评**：尽管想尝试租赁 GPU，一位用户认为 *The Nous API* 是一个 *很好的选择*，并且对 *定价没有任何怨言*。
   - 与价格过高的 **Hugging Face** 相比，Nous API 的混合计算和定价被认为是公平的。
- **Qwen3-TTS 的语音克隆能力令人印象深刻**：**Qwen3-TTS** 被认为是非常出色的 *voice cloning* 工具，足以与 **ElevenLabs** 媲美；分享了 [Hugging Face Spaces 上的 Qwen3-TTS](https://huggingface.co/spaces/Qwen/Qwen3-TTS) 链接。
- **在华为硬件上训练模型势头渐盛**：在非 Nvidia 的多样化硬件上训练前沿模型的趋势正在兴起，提到 **Gemini** 是在 **Google TPUs** 上训练的，而 **Zai GLM 4.7** 是在**华为硬件**上训练的，这在一段 [YouTube 视频](https://www.youtube.com/watch?v=WU_rKAC_SLI) 中有所涵盖。
- **AI Agents 的自我复制基准测试正在酝酿中**：一位成员正在思考 *针对 agentic-ai 的自我复制基准测试*，并正在寻求关于合适目标的建议。一个建议是评估像 Claude 用 C 代码实现的 Transformer 推理引擎，该引擎还设计了自定义处理器，可在 [GitHub](https://github.com/cpldcpu/smollm.c/blob/claude/train-small-model-llxVr/smolc/smolc.c) 上找到。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1463994096245932064)** (82 messages🔥🔥): 

> `Capital One acquires Brex, AI-Powered Pool Detection, Replit Agent Decision-Time Guidance, Fine-tuning on Code Base, Multi-Agent Communication` 


- **Capital One 以数十亿美元收购 Brex**：Capital One 以 **51.5 亿美元**收购了 Brex，标志着历史上最大的银行-金融科技交易，[详情请点击此处](https://x.com/alexfmac/status/2014676950883668306?s=46)。
- **SAM3 和 Mapbox 揭露隐藏的泳池派对地点**：利用 Meta 的 **SAM3** 模型和 **Mapbox** 图像，仅通过单一文本提示就在 10 平方公里的郊区识别出近 1,500 个游泳池，展示了 zero-shot 地理空间智能。
   - 详情见 [Kyle Walker 的推文](https://xcancel.com/kyle_e_walker/status/2014433189423407194)。
- **Replit Agent 处理复杂任务**：Zhen Li 详细介绍了一篇关于 **Replit Agent** 中 Decision-Time Guidance 的技术博客，探索实时控制机制而非静态规则。
   - 这有助于自主 Agent 更有效地处理复杂的现实世界任务；[博客文章链接在此](https://xcancel.com/zhenthebuilder/status/2014393451442581688?s=46)。
- **Baseten 融资 3 亿美元，估值飙升至 50 亿美元**：Baseten 宣布完成由 IVP 和 CapitalG 领投的 **3 亿美元 Series E** 融资，估值达到 **50 亿美元**，NVIDIA 和其他风险投资公司参投；[详情见 Baseten 的推文](https://xcancel.com/basetenco/status/2014755013344792595?s=46)。
- **ChatGPT 仍占据主导地位，Grok 见证爆发式增长**：来自 SimilarWeb 的数据显示，尽管市场趋于饱和，**ChatGPT** 仍引领 AI 平台市场，而 **Grok** 的增长最为显著，在美国的月度独立访客渗透率增长了 **33 倍**。
   - 更多见解可在 [Venture Twins 的推文](https://xcancel.com/venturetwins/status/2014739492389978274?s=46) 中找到。


  

---


### **Latent Space ▷ #[private-agents](https://discord.com/channels/822583790773862470/1342964204168020018/1464139497808859215)** (1 messages): 

> `Memory Usage, Payment Plan` 


- **挥霍内存以停止交换 (Swapping)**：一位成员表示他们拥有 **96gb** RAM，并为此获得了*许可/分期付款方案*。
   - 他们补充说，结果是他们*再也没有回头（感到非常满意）*。
- **现代软件吞噬内存**：一位成员怀疑他们的机器是否因为内存占用过高而正在向磁盘进行交换（swapping）。
   - 他们评论说，许多现代软件都是内存杀手，当他们*打开任务管理器时感到想哭*。


  

---

### **Latent Space ▷ #[genmedia-creative-ai](https://discord.com/channels/822583790773862470/1397010677364953149/1464064002094006376)** (15 条消息🔥): 

> `Qwen3-TTS 发布, 图像编辑质量下降, 音频 AI 模型发布` 


- **阿里巴巴发布 Qwen3-TTS 资源宝库**：阿里巴巴发布了 **Qwen3-TTS**，这是一个开源的文本转语音 (text-to-speech) 系列模型，具有 **VoiceDesign**、**CustomVoice** 和 **Base models**，可在 [GitHub](https://github.com/QwenLM) 和 [Hugging Face](https://huggingface.co/Qwen) 上获取，包含从 **0.6B** 到 **1.8B** 参数的五个模型，支持 **10 种语言**和高质量的语音克隆。
- **迭代编辑导致图像受损**：正如这个 [reddit 帖子](https://old.reddit.com/r/comfyui/comments/1qkgc4y/flux2_klein_9b_distilled_image_edit_image_gets/) 中讨论的，使用 **Flux.2 Klein 9B** 进行重复的图像编辑时，如果将输出图像反馈回去进行进一步编辑，会导致**渐进式的饱和度偏移和质量下降**。
- **音频 AI 领域迎来新突破**：Lina Colucci 在 [此帖](https://x.com/lina_colucci/status/2014229002370834861?s=46) 中突出了三个主要的音频 AI 发布：**NVIDIA 的 PersonaPlex-7B** 全双工对话模型，**Inworld AI 的低延迟 TTS-1.5**，以及 **Flash Labs 的 Chroma 1.0**（首个开源端到端 speech-to-speech 模型）。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1464201720774004869)** (11 条消息🔥): 

> `Transformer AGI?, Pangram 性能, NeRF 研究, ROCm 软件` 


- **Transformer 架构通向 AGI？**：成员们讨论了 **Transformer 架构**是否能够实现 **AGI**，并将其与 **neuro-symbolic** 或 **JEPA** 架构等方法进行了对比。
- **Pangram 检测器表现令人印象深刻**：一位成员询问了 **Pangram** 相对于其他检测器的性能，另一位用户根据他们使用 [这个 GitHub 仓库](https://github.com/adithya-s-k/manim_skill) 的经验表示，*Pangram 是目前最令人印象深刻的一个，领先优势明显*。
- **NeRF 研究工作**：一位成员询问是否有人正在积极从事 **NeRF (Neural Radiance Fields) 研究**。
- **ROCm 软件的 ML 性能**：一位成员询问了用于 **GPU** 加速 **ML** 的 **ROCm 软件**目前的性能和可靠性。
   - 另一位用户回答说，**ROCm** *在易用性方面取得了长足进步*，但由于主要支持仍然是针对 **Nvidia**，因此仍具挑战性。


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1463991483626426474)** (65 条消息🔥🔥): 

> `等参数量顺序 vs 并行 Attention 块, Marin 的 Attention 块实验, Nanogpt 中的 Difference Layer, 改进 Nanogpt 基准线, 乘法网络与逻辑先验` 


- **Attention 块对决：顺序 vs 并行**：一位成员询问关于 Transformer 中等参数量**顺序 (sequential) vs 并行 (parallel) attention 块**对比的参考文献。
   - 另一位成员指出 **Marin 的实验**是一个相关的资源。
- **Difference Layer 在 Nanogpt 中展现潜力**：一位成员分享了来自 [Eternalyze0/difference_layer](https://github.com/Eternalyze0/difference_layer) 的 *difference layer* 代码 `x = (self.a2(x) - self.b2(x)) * (self.c2(x) - self.d2(x)) + self.e2(x)`，声称它在 **cartpole** 和 **nanogpt** 上表现明显更好，且使用的参数和计算量更少。
   - 这种架构在 SGD 中使有效学习率翻倍，这可能解释了相对于优化不佳的基准线所报告的改进。
- **提升 Nanogpt：Swiglu 激活函数**：一位成员建议通过将激活函数从 **GELU** 切换为标准的 **SwiGLU** 来增强 **Nanogpt 基准线**，这带来了性能提升。
   - 在另一个实验中，用 difference layer 替换 **QKV linear** 层也扩大了与基准线相比的性能差距。
- **为有意义的研究优化基准线**：资深研究人员强调了在实验中使用强大、优化的基准线的重要性，以避免将噪声误认为实际的改进。
   - 他们建议在语言任务中使用 **modded nanogpt**，并紧跟文献或咨询资深研究人员以确定合适的基准线。
- **乘法网络：逻辑先验**：一位成员认为**乘法网络 (multiplicative nets)** 具有更高的逻辑先验，因为它具有天然的门控 (gating) 机制。
   - 另一位成员指出，[Noam Shazeer 的 GLU 变体论文](https://arxiv.org/abs/2002.05202) 已经讨论了这一点，并强调实验必须从强大的基准线开始。


  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1464015531492376800)** (3 messages): 

> `MATS/AFP follow up paper, Compression-Lens HF space` 


- **MATS/AFP 后续论文正在撰写中**：一名成员宣布他们正在撰写 **MATS/AFP** 的后续论文，同时 Christina 正在为 ICML 准备相关论文；该成员正在寻找熟悉原论文思路的合作者。
   - 他们引用了之前的[合作者征集公告](https://discord.com/channels/729741769192767510/730095596861521970/1462609593703207175)。
- **分享了 Compression-Lens Hugging Face Space**：一名成员分享了一个与 **Compression-Lens** 技术相关的 [Hugging Face Space 链接](https://huggingface.co/spaces/Jellyfish042/Compression-Lens)。
   - 未提供关于该共享空间用途或相关性的进一步背景信息。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

aeros93: https://fixupx.com/havenfeng/status/2014765400563781777?s=46
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1463991249349640460)** (55 messages🔥🔥): 

> `Claude's Laziness vs GPT-Codex Grunt Work, Yann LeCun's Startup and Energy-Based Models (EBMs), Adversary Review, EBMs vs Diffusion Models, URM paper (based on tiny recursive models)` 


- **Claude vs GPT-Codex**：一位成员分享了使用 **VSCode Copilot** 的经验，指出 **Claude** 似乎比较“懒惰”且喜欢做假设，而 **GPT-Codex** 会完成大量的“苦力活（grunt work）”，这既有帮助也可能让人分心。
   - 该成员认为这两者都不是完美的工具。
- **LeCun 的初创公司以数独游戏亮相**：成员们讨论了 [Yann LeCun 的新 AI 初创公司](https://www.reddit.com/r/agi/comments/1qjzdvx/new_ai_startup_with_yann_lecun_claims_first/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)，该公司声称在 **Energy-Based Models (EBMs)** 方面取得了突破，并展示了其解决数独谜题的能力。
   - 一些人表示怀疑，指出缺乏关于架构、模型大小和训练的细节；而另一些人则指出 **EBMs 解决数独** 已是已知的能力，并好奇为什么 [LeCun 没有在社交媒体上提及它](https://energy-based-model.github.io/ired/ired.pdf)。
- **是否需要对抗性评审（Adversary Review）？**：社区讨论了学术出版中“对抗性评审”的概念及其潜在挑战，例如评审员需要是领域专家，且可能是竞争对手。
   - 一名成员建议 **AI** 可能会通过奖励函数来执行对抗性评审，以激励其指出错误，但他们也承认评审员需要时间和动力来彻底校对论文。
- **EBMs 是实现与扩散模型基本相同效果的较差方式**：一位成员详细解释了 **Energy-Based Models (EBMs)** 及其与扩散模型的关系，认为 EBMs 实际上是实现相同结果的一种较差方式。
   - 他们指出扩散（diffusion）、分数匹配（score matching）和流匹配（flow matching）本质上是相同的。


  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1464282397141893313)** (2 messages): 

> `RLM framework, LLM agent setting, REPL` 


- **RLM 是否类似于 LLM Agent 中的 REPL？**：一位成员询问 **RLM 框架** 是否类似于在经典的 **LLM Agent 设置**中为编排器（orchestrator）添加 **REPL**。
   - 另一位成员概述了不同之处：**RLM** 可以产生子语言模型（sub-LMs），在基础模型和子模型中使用相同的系统提示词（system prompt），在环境中保存提示词/上下文，并且除了 **REPL** 和调用子模型外，并不重度依赖工具使用。
- **RLM vs LLM Agent 编排器**：**RLM** 与 **LLM Agent 编排器** 的不同之处在于它能够产生子模型，而不是依赖于人工创建的工作流。
   - 与具有不同“角色（personas）”的 Agent 编排器不同，**RLM** 的基础语言模型和子模型共享相同的系统提示词，且 **RLM** 除了 **REPL** 和子模型调用外，不重度依赖工具使用。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1464076640001720454)** (13 messages🔥): 

> `Vibe Coding, OpenAI 资金枯竭, 开源前沿模型` 


- **攻击 Vibe Coding 公司并获利？**：一名成员分享了一条推文，建议寻找那些吹嘘 **Vibe Coding** 的公司，破坏其系统，然后获利或披露漏洞 ([来源](https://x.com/yuvalavra/status/2011842613389726109))。
- **OpenAI 可能在 2027 年面临资金短缺**：一位成员分享了一篇文章，根据一位分析师严峻的财务评估，**OpenAI** 可能会在 2027 年年中耗尽现金 ([Tom's Hardware](https://www.tomshardware.com/tech-industry/big-tech/openai-could-reportedly-run-out-of-cash-by-mid-2027-nyt-analyst-paints-grim-picture-after-examining-companys-finances))。
   - 另一位成员开玩笑说 *"他们太大而不倒（too big to fail）。我们需要在事前和事后救助他们"*。
- **开源前沿模型：不需要救助？**：一位成员建议，与其进行救助，不如让 **OpenAI** 开源其前沿模型，让人们自己运行 ([来源](https://fixvx.com/sama/status/2014733975755817267))。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1464023921648074814)** (10 messages🔥): 

> `逼真的物理引擎和图形, 针对特定领域的预测模型（天气）, 带有猫耳的女性对应形象用于自我刺激, 自动编程的性能分析, 即将举行的 GTC 活动` 


- **寻求更真实的物理引擎**：成员们提到了对**更真实的物理引擎和图形**的需求，以及针对天气等**特定领域预测的专业模型**。
   - 此外，他们还提出了代表女性形象（*可能带有猫耳*）的系统由于人类社交倾向而变得更受欢迎的想法。
- **自动编程仍需 Profiling**：有人提到，即使**编程变得更加自动化**，了解 **Profiling**（性能分析）对于更好的架构和系统设计仍然可能有所帮助。
   - 一位成员表示喜欢 **GPU 编程**，希望能有更多理由去学习它，但感觉软件正朝着架构和系统设计方向发展。
- **GTC 活动安排中**：一位成员询问了关于**即将举行的 GTC** 活动的信息，另一位成员[回应](https://developer.nvidia.com/gtc)称，正计划为 **nvfp4 竞赛举行颁奖典礼**，并可能举行 **happy hour**。
   - 一位成员正飞往现场，已经办好了 **ESTA** 以免错过航班。
- **文本转 3D 形状模型？**：一位成员询问是否有人在**渲染引擎 (Rendering Engine)** 中使用过或正在使用 **文本转 3D 形状模型**。
   - 未提供更多细节。
- **带有动态量化的 MXFP8**：一位成员询问了关于使用**带有动态量化的 mxfp8** 的问题，并指出它被用于像 TransformerEngine 这样的 NVIDIA 库中。
   - 他们询问是否有**在激活值上使用静态 mxfp8 缩放因子**的例子，因为在张量上进行最大值规约（max reduction）以获取缩放因子的开销很大。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1464176271683752039)** (8 messages🔥): 

> `FA4 峰值性能, NVIDIA B200 规格` 


- **FA4 基准测试达到理论最大值的 71%**：FlashAttention-4 (**FA4**) 在使用 **BF16** 输入的 **NVIDIA B200 GPU** 上实现了 **1,605 TFLOPS/s** 的峰值性能，这是硬件理论最大值的 *71%*。
   - 成员们讨论了理论最大值约为 **2260 TFLOPS**，但确切的规格很难获得。
- **NVIDIA B200 实际 TFLOPS 存疑**：一些 **B200** 资料列出的 **fp4/fp8/fp16** 性能分别为 **10/5/2.5 TFLOPS**，但缺乏官方文档。
   - 社区成员正在等待包含详细规格的官方论文，特别是考虑到博客文章中没有提到数据类型。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 messages): 

costa5805: https://mlsys26.flashinfer.ai/
  

---

### **GPU MODE ▷ #[job-postings](https://discord.com/channels/1189498204333543425/1190208177829068860/1464218961905451028)** (2 messages): 

> `CUDA, GPU performance optimisation, CUDA kernel optimization, Nsight` 


- **工作机会：CUDA/GPU 性能优化**：某个人正在寻找在 **CUDA** 和 **GPU performance optimisation** 方面有深厚背景的人才，以履行一份短期、全远程的合同。
   - 该角色涉及根据 **CUDA** 和 **GPU optimisation** 的经验，创建清晰、结构良好的实际场景任务，重点关注结构化思维和书面推理。
- **Parsewave 寻求 CUDA Kernel 优化工程师**：**Parsewave** 正在寻找工程师来编写和优化 **CUDA C/C++ kernels**，使用 **Nsight Systems / Nsight Compute** 诊断瓶颈，并解释优化权衡，可通过 [此链接](https://tally.so/r/pbDDvZ) 申请。
   - 理想的候选人应熟悉 **CUDA intrinsics**（尤其是 Blackwell 或 Hopper），并能够提出能清晰展示 **naive → optimized deltas** 的场景/基准测试。


  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1464000590437945562)** (6 messages): 

> `GPUMode 2026, CUDA PTX submissions, dual_gemm problem` 


- **GPUMode 2026 路线图确认**：一位成员确认 **GPUMode 2026** 的路线图仍在按计划近期发布，并链接到了 [GPUMode 2026 新闻文章](https://www.gpumode.com/v2/news/gpumode-2026)。
- **CUDA PTX 提交量激增**：在最新的 **dual_gemm** 问题中，**CUDA PTX submissions** 的数量比以往任何时候都多。
   - 该成员提到他们*应该很快会发布一些更广泛的公告*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1464248891338653903)** (10 messages🔥): 

> `Compiler Intrinsics, ROCm vs CUDA, AMD Developer Ecosystem` 


- **内建函数见解：编译器文档技巧！**：解释了编译器 intrinsics 和 builtins：**builtins** 是 *clang* 的概念，**intrinsics** 是 *llvm* 的概念，builtins 通常 1:1 地转换为 intrinsics，并可参考 [AMD's ISA manual](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) 获取帮助。
   - 为了查找文档，成员们建议查看 [AMDGPU Usage](https://github.com/llvm/llvm-project/blob/main/llvm/docs/AMDGPUUsage.rst)、[IntrinsicsAMDGPU.td](https://github.com/llvm/llvm-project/blob/main/llvm/include/llvm/IR/IntrinsicsAMDGPU.td) 以及 [AMDGPU CodeGen tests](https://github.com/llvm/llvm-project/tree/main/llvm/test/CodeGen/AMDGPU)。
- **再见 5090：开发者弃坑 ROCm，转向 CUDA！**：一位开发者在购买 **5090** 后告别了 ROCm，理由是*打包、构建、分发可用性方面的问题太多*，以及*缺乏对面向消费级设备的关注*。
   - 他们表示，在专业消费者（prosumer）领域缺乏持续的有意义的硬件更新。
- **Conv3D 惨败：NVIDIA 完胜！**：一位成员指出，中档 NVIDIA 硬件在 **Conv3D** 等任务上的表现明显优于 AMD 设备，并引用了 [一个 Reddit 帖子](https://www.reddit.com/r/ROCm/comments/1owczm9/the_convolution_performance_on_rx_9070_is_so_low/)。
   - 他们抱怨*硬件代际之间的性能不一致*，以及*在多个主要版本中都未得到解决的回归问题*。
- **敌对的地平线：AMD 生态系统引发愤怒！**：一位开发者对 ROCm 生态系统表示挫败，称其为*充满敌意的*，理由包括 FBGEMM 仓库无法在 gfx1100 上构建等问题。
   - 他们指出 [Quark quantisation engine 的一次 commit](https://github.com/amd/Quark/commit/9234960c951410abdcecee033adf610d7126fda3) 是沟通不畅和对贡献不友好的典型例子。


  

---


### **GPU MODE ▷ #[popcorn](https://discord.com/channels/1189498204333543425/1298372518293274644/1464028958059139186)** (6 messages): 

> `FlashInfer-Bench, CMU Catalyst Lab, AI-generated GPU kernels, Collaboration opportunities` 


- **由 CMU Catalyst Lab 驱动的 FlashInfer-Bench**：来自 **CMU Catalyst Lab** 的 Yixin Dong 介绍了 **FlashInfer-Bench**，这是一个用于评估 **AI-generated GPU kernels** 并将其部署到推理引擎中的框架。
   - 该小组正在探索改进基准测试和在社区内开展协作的方法。
- **FlashInfer-Bench 获得赞誉**：一位成员指出 *Flashinfer bench 是一个非常酷的项目*。
   - 创作者期待与社区合作，特别是在生产环境中评估和部署生成的 kernel 方面。


  

---

### **GPU MODE ▷ #[cutlass](https://discord.com/channels/1189498204333543425/1362196854460383353/1464375553678119036)** (1 messages): 

> `Graphical Layout Calculus, Tuple Morphisms, Mutual Refinement, Prefix Products` 


- **布局组合通过图形化计算**：一位成员演示了如何使用 **graphical layout calculus** 手动计算两个布局的组合，并附带了详细说明步骤的图像。
- **布局转换为 Tuple Morphisms**：初始步骤涉及将可处理的布局转换为表示为 `m_A` 和 `m_B` 的 **tuple morphisms**。
- **Mutual Refinement 对组合至关重要**：该过程需要找到两个 **tuple morphisms** 的 **mutual refinement**，这对于组合布局至关重要。
- **对 Refinements 进行 Pulling Back 和 Pushing Forward**：**mutual refinement** 沿着 `m_A` 进行 **pull back** 以获得 `\hat{m}_A`，并沿着 `m_B` 进行 **push forward** 以获得 `\hat{m}_B`。
- **结果表示为 Prefix Products**：最终结果在组合 `\hat{m}_B o \hat{m}_A` 后，被表示为一个利用陪域元组（codomain tuple）的 **prefix products** 的布局。


  

---


### **GPU MODE ▷ #[teenygrad](https://discord.com/channels/1189498204333543425/1373414141427191809/1464345228696617021)** (3 messages): 

> `SITP Curriculum, GEMM Kernels on ARM, SVGBob for diagrams` 


- **SITP 课程因其易读性受到称赞**：一位成员对 [SITP curriculum](https://j4orz.ai/sitp/1.html) 的 **1.1** 和 **1.2** 章节的编排感到满意，因为它非常易于上手。
   - 他们认为这些材料对于 Discord 的受众来说可能过于基础，但强调这些特定章节使得入门门槛足够低，甚至高中生也能阅读。
- **GEMM 内核准备攻克 ARM**：一位成员计划为课程的 **1.3**、**1.4** 和 **1.5** 章节实现 **ARM 上的 GEMM 内核**。
   - 这些内核将建立在早期章节提供的易懂介绍之上。
- **SVGBob 成为绘图首选**：一位成员比起 **tikz**、**mermaidjs** 或 **graphviz**，更喜欢使用 **SVGBob** 进行绘图。
   - 他们解释说，**SVGBob** 让他们能以 ASCII 文本而非像 dot 这样的 DSL 来生成图表，并且它还可以嵌入到 Compiler Explorer 中。


  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1394753097989099640/1463989990374510738)** (1 messages): 

> `Test-Time-Training, LM-generated Kernels, Model Performance, TTT Results` 


- **针对 LM 生成内核的 TTT**：研究人员在过去的排行榜上使用 **test-time-training (TTT)** 评估了 **LM-generated kernels**，并获得了如本 [论文](https://test-time-training.github.io/discover.pdf) 中所述的有趣结果。
- **LM 内核在测试时训练中表现优异**：**test-time-training.github.io** 上强调的一项最新研究展示了在成熟排行榜上使用 **TTT** 评估 **LM-generated kernels** 的前景良好的结果；详细发现可在链接的 [PDF](https://test-time-training.github.io/discover.pdf) 中查阅。


  

---


### **GPU MODE ▷ #[multi-gpu](https://discord.com/channels/1189498204333543425/1398843708488552570/1464249056510349427)** (5 messages): 

> `NCCL Benchmarking on Slurm, B200 Performance Tuning, Slurm sbatch script` 


- **Slurm 脚本问题引发 NCCL 调试**：一位用户在 **B200** 上使用 **Slurm** 进行 **NCCL** 基准测试时遇到了意外的性能问题，并分享了一个 [sbatch 脚本](https://gist.github.com/example/12345) 供大家审查。
   - 一位成员建议检查 `NCCL_DEBUG=INFO` 的输出，并尝试在不设置环境变量的情况下运行基准测试，因为 *NCCL 通常自动调优得很好*。
- **Slurm 上的 GPU 性能**：一位用户寻求在 **Slurm** 上对 **B200** 进行 **NCCL** 基准测试的帮助，怀疑存在配置问题。
   - 用户提供了他们的 `sbatch` 脚本，正在寻找其设置中可能存在的问题。


  

---

### **GPU MODE ▷ #[flashinfer](https://discord.com/channels/1189498204333543425/1464407141128339571/1464407381680062630)** (12 条消息🔥): 

> `竞赛、团队合并、多赛道、报名确认` 


- **竞赛更新鼓励发帖**：由于预计大家对新竞赛会有很高兴趣，管理员鼓励其他人在 general 频道发布简短帖子来描述竞赛目标。
   - 他们推测人们会对新竞赛特别感兴趣。
- **截止日期前允许团队合并**：在报名截止日期前允许团队合并，参与者被告知如果合并需通知管理员。
   - 管理员还表示参与者稍后可以切换赛道。
- **尽管参加多赛道，GPU 奖项仍有限制**：参与者可以参加多个赛道，但如果在多个赛道获胜，可能只能获得一个 GPU 奖项。
   - 一位用户澄清说，如果愿意，他们稍后可以切换赛道。
- **报名确认邮件已自动化**：根据自动化报名确认邮件的建议，管理员确认他们已经完成了设置。
   - 目标是避免重复报名。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1463992048939044988)** (33 条消息🔥): 

> `LazyMergeKit 中断、Agent 课程频道信息、Llama.cpp GLM 4.7 Flash GGUFs 提速、微调预训练模型、视频换脸 API 讨论` 


- **LazyMergeKit 备受中断困扰**：一位成员在使用 **LazyMergeKit** 合并模型时遇到中断，这是他们以前从未遇到的问题，并发布了一张截图，显示 Space 似乎被管理员暂停了。
   - 他们认为该项目已被置顶，并且在删除并重新上传到同名 Space 后，*停用* 状态依然存在。
- **Llama.cpp 实现 GLM 4.7 Flash GGUFs 速度飞跃**：**Llama.cpp** 使 **GLM 4.7 Flash GGUFs** 速度提升了约 *1.5 倍* 并修复了 bug，敦促用户重新构建 **llama.cpp** 并从 [此处](https://huggingface.co/unsloth/GLM-4.7-Flash-GGUF) 重新获取修复后的量化文件。
   - 对于在 **llama.cpp** 框架内使用 **GLM 4.7 Flash GGUFs** 的用户来说，这代表了显著的性能提升。
- **微调热潮即将到来**：一位成员询问了微调预训练模型的过程，询问是否可以在 **Google Cloud** 或 **Kaggle** 中完成，并附上了 [Fine-tuning LLMs Guide](https://docs.unsloth.ai/get-started/fine-tuning-llms-guide) 的链接。
   - 一位成员建议将其转换为 **ONNX** 或 **GGUF** 以进行客户端执行，并链接了 [LFM2.5-VL-1.6B-WebGPU](https://huggingface.co/spaces/LiquidAI/LFM2.5-VL-1.6B-WebGPU) 和 [SmolVLM-256M-Instruct-WebGPU](https://huggingface.co/spaces/HuggingFaceTB/SmolVLM-256M-Instruct-WebGPU) 等示例。
- **Space 漫游：HF Space 被停用**：一位成员报告称，尽管其流量最高，但他们的 Space 仍被置顶并停用，导致相关的 **X** 帖子被删除，该 Space 可以在 [此处](https://huggingface.co/spaces/tostido/Cascade-Hyperlattice) 找到。
   - 他们怀疑，即使在删除后，结合另一页面功能所提供的服务可能导致了停用，并认为该命名空间已经 *报废*。
- **RAG 时间：寻找教程**：一位成员正在寻求关于构建可完全本地托管的 **RAG (Retrieval-Augmented Generation)** 系统的简明教程或指南。


  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1463998594821521468)** (12 messages🔥): 

> `wasm 合成数据集, 字体数据集, agentic AI 更新, 德语优先 LLM, 安全数据集` 


- **用于可复现 Rust 到 WebAssembly 编译的 WASM 数据集！**：一个包含 1,000 个以编程方式生成的 **Rust 程序**元数据的全合成数据集已在 [HuggingFace](https://huggingface.co/datasets/webxos/wasm_synthetic_dataset) 发布，这些程序旨在编译（或失败）到 **WebAssembly**。
   - 所有样本均使用确定的斐波那契（Fibonacci）衍生伪随机生成器创建，在代码模式、源代码长度、导出的函数数量以及结构哈希方面产生可复现的变体。
- **为您生成的字体数据集**：一位成员开发了一个生成**字体数据集（typeface dataset）**的应用，并提供根据个人或团队需求定制的数据集。详情请私信。
   - 更多信息请访问 [webxos.netlify.app/COLIGNUM](https://webxos.netlify.app/COLIGNUM)。
- **随时掌握 Agentic AI 的最新动态**：针对高级工程师提炼的关于 Agentic AI、RAG、LLM、生产工具、编排、治理和实际部署的高质量每日更新，可在 [x.com/when_robots_cry](https://x.com/when_robots_cry) 找到。
   - 一个类似于 Antigravity 的浏览器工具可以在 [https://mcp.so/server/browser-control/adityasasidhar](https://mcp.so/server/browser-control/adityasasidhar) 找到。
- **Faust-1：德语优先 LLM 发布**：**Faust-1**，一个从零开始训练的 **1.6B** 参数德语优先大语言模型，已在 [HuggingFace](https://huggingface.co/tabularisai/Faust-1) 发布。
   - 它具有以德语为主的预训练（≈90%）、针对德语优化的自定义分词器（tokenizer）、经过验证的合成数据 + 指令微调（DPO），并专为本地/隐私敏感型部署而设计。
- **安全数据集发布**：一个**安全数据集（safety dataset）**已在 [HuggingFace](https://huggingface.co/datasets/Pacific-Prime/safety_dataset) 发布。


  

---


### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1464083736999235719)** (3 messages): 

> `Agent 课程, 机器人课程, HuggingFace 教程` 


- **Agent 课程频道难以找到**：一位新的 Discord 用户表示难以找到 Agent 课程的频道。
   - 他们表达了加入社区的兴趣。
- **机器人课程模块查询**：一位用户询问了新机器人课程剩余模块的位置，并指出目前仅发布了两个模块。
   - 他们急需学习该课程以完成毕业设计。
- **建议查看 HuggingFace 教程**：一位成员建议访问 [Hugging Face 机器人学习教程](https://huggingface.co/spaces/lerobot/robot-learning-tutorial) 来学习机器人技术。
   - 他们表示*这是正确的选择*。


  

---

### **Manus.im Discord ▷ #[general](https://discord.com/channels/1348819876348825620/1349440650495398020/1464008161705853009)** (22 messages🔥): 

> `高级 AI 工程师介绍, 寻找机会的 AI Agent 开发者, 全栈 AI 工程师介绍, Manus 未经授权计费, Dracoai.app` 


- **高级 AI 工程师介绍专业背景**：一位成员介绍了自己，作为一名拥有超过 7 年经验的**高级 AI 工程师**，专注于构建用于真实生产环境的可扩展、云原生 **AI 系统**，在**深度学习、NLP、计算机视觉和多模态 AI** 领域拥有深厚专业知识。
   - 他们特别期待在 **AI 性能、可靠性和实际影响力**至关重要的项目上进行合作。
- **AI Agent 开发者寻求合作**：一位成员强调了他们在为各种应用构建 **AI agents** 方面的专业知识，包括**客户支持、工作流自动化、数据分析和自主预订**，并强调关注生产级系统而非仅仅是演示原型。
   - 他们专注于**工具编排 (tool orchestration)、确定性输出、长周期 Agent 状态管理以及延迟、成本和故障模式的优化**。
- **全栈 AI 工程师推广服务**：一位成员宣传了他们在构建 **AI + 全栈系统**方面的技能，专注于交付实际价值并提高效率、准确性和用户体验，专长包括 **LLM 集成、工作流自动化、AI 内容检测、图像 AI (CLIP + YOLOv8) 和语音 AI (Whisper, Tacotron2)**。
   - 他们的服务包括使用 **React, Next.js, Node.js, Laravel, Django, Flutter, React Native 以及混合链上/链下 AI/服务编排**进行全栈开发。
- **用户报告未经授权的计费事件**：一位成员报告称，尽管在免费试用后选择了按月计费，但仍被收取的 **$400** 的年费计划费用，并且在联系客户支持时遇到困难。
   - 如果问题得不到解决，他们计划向 **FTC、BBB、州检察长**投诉，并联系 **Meta**，同时寻求可能经历过类似问题的其他人的建议。
- **用户发现 Draco AI 电话功能？**：一位成员提到他们尝试了 [Dracoai.app](https://dracoai.app)，并注意到其“呼叫者模型”拨打了电话来执行任务。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1464167385060872203)** (12 messages🔥): 

> `aider TUI 支持, aider 会话管理, aider 检查点, aider 上下文管理, aider 补充 Claude` 


- ****Aider** 将获得 TUI 支持？**：一位成员讨论了在 *aider* 中实现 **TUI 支持**，以便在浏览回复时编辑下一条消息，并渲染美观的 **Markdown** 输出（如 **mermaid 图表**）。
- ****Aider** 的会话管理？**：一位成员建议为 *aider* 添加**会话管理**功能，以便用户可以临时将所有聊天内容存储在一个地方，在聊天上下文之间切换，并从过去的消息继续而不污染上下文。
   - 他们还提议进行**细粒度的上下文管理**，以便从聊天日志中删除无用或不正确的输入/输出。
- **将 Aider 与 Claude 搭配使用非常强大！**：一位成员将 *aider* 与 **Claude code** 结合使用，以实现高效的工作流：使用 *aider* 追求速度，然后切换到 **Claude** 来解决棘手的 Bug。
   - 该成员认为 *aider* 的优势在于其确定哪些文件需要进入上下文的方法、上下文管理，以及其最小化 LLM token 输出的搜索和替换编码器。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1464261027297493045)** (2 messages): 

> `Elysia Agentic RAG, Skill Optimizer` 


- **Elysia Agentic RAG 深度探讨**：一位成员分享了来自 Unravel Tech 的 [Elysia Agentic RAG 博客文章](https://www.unravel.tech/blog/elysia-agentic-rag-deep-dive) 链接。
   - 他们征求了对该文章的想法和意见。
- **Skill Optimizer GitHub 仓库**：一位用户分享了 [Skill Optimizer GitHub 仓库](https://github.com/Ash-Blanc/skill-optimizer) 的链接。
   - 未提供关于该仓库用途或功能的额外上下文。


  

---

### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1464047383083094135)** (2 messages): 

> `Continual Learning, DSPy.RLM()` 


- **Continual Learning 的工程化实现**：一位成员发布了他们在过去几个月中对 **continual learning**（持续学习）的探索，并分享了一篇关于将其工程化实现的 [博客文章](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859)。
   - 他们认为 **DSPy.RLM()** 在这一领域发挥着巨大的作用。
- **Continual Learning 与 DSPy.RLM**：一篇博客文章讨论了如何工程化实现持续学习，并预见 **DSPy.RLM()** 将承担重要角色。
   - 文章链接点击 [此处](https://raveesh.substack.com/p/a-pragmatic-recipe-for-continual?r=qn1thttps://arxiv.org/abs/2512.21859)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1464318884453552301)** (7 messages): 

> `DSPy Misunderstood, RLM Prompt Tuning, JSON Adapter Optimization` 


- **DSPy 的真正实力终于被阐明**：一位成员发表了 [文章](https://eito.substack.com/p/dspy-the-most-misunderstood-agent)，解释了 *“为什么选择 DSPy？”* 并详细阐述了 **signature & module abstraction**（签名与模块抽象）的重要性，这些概念在 DSPy 社区之外经常被忽视。
   - 作者认为 DSPy 的能力远不止 **GEPA & Prompt 优化**，他在关于该文章的 [X 推文](https://x.com/Eito_Miyamura/status/2014757193766093069?s=20) 中也强调了这一点。
- **Rationalize Like Mad (RLM) Prompt 微调策略**：成员们讨论了通过微调 **RLM prompt** 来提高推理能力，其中一位成员指出，某些模型在查看输入后仍会给出 *模糊且通用的回答*。
   - 另一位成员建议，它的 *优化方式与 ReAct 的优化方式非常相似*，并表示优化器会自动检查 trace，用户只需关注期望的输出即可。
- **JSON Schema 引发 GEPA 定制热潮**：一位成员希望使用 GEPA 优化 **JSON adapter** 的系统提示词，特别是针对那些使用 **json_schema response types** 的模型。
   - 他们认为，如果 JSONadapter 添加的许多 token 是不必要的，**GEPA** 就可以精简系统提示词；同时他们也意识到需要一个 **自定义 GEPA adapter**，因为目前的 DSPy 版本似乎无法影响 adapter。


  

---


### **Moonshot AI (Kimi K-2) ▷ #[general-chat](https://discord.com/channels/1369594130807787570/1371757564005711973/1464201099123888242)** (10 messages🔥): 

> `Kimi issues, Conversation length exceeded messages, Slides are suddenly constantly showing 'This mode is at capacity.'` 


- **Kimi 面临错误和故障**：一位成员报告了 **Kimi** 的问题，包括消息消失和频繁出现“对话长度超限”错误，其他成员则报告幻灯片功能一直显示“该模式已满负荷”。
   - 尽管另一位成员表示视觉化和自适应幻灯片生成对他来说正常，但其他人证实了该问题的持续存在，描述必须 *点击 50 次以上才能最终成功*。
- **成员推测数据中心故障**：一位成员推测了 **Kimi** 出现问题的潜在原因，怀疑是 *数据中心宕机*、**来自 Google 的 Nano Banana API 访问限制**，或者是 *使用协议的变更*。
- **Radiohead 启发了 'Ok Computer' 标题**：一位成员分享了一则 [推文](https://x.com/crystalsssup/status/2014571082716713356)，确认了 **Radiohead** 是其专辑名称 **Ok Computer** 的灵感来源。
   - 该成员还评论道，*视觉幻灯片功能现在几乎变得无法使用*。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1464020833528840374)** (3 messages): 

> `Introduction of AI/ML Engineers, Mojo in Production` 


- **AI/ML 工程师自我介绍**：多位经验丰富的 **AI 和 ML 工程师** 进行了自我介绍，他们专注于构建和部署 **ML 流水线 (pipelines)、深度学习模型和 NLP 系统**。
   - 这些工程师设计 **预测引擎、推荐系统和生成式 AI 工作流**，将 AI 模型集成到 Web 和移动应用程序中，并专注于 **可靠性、性能以及生产级 ML 架构**。
- **Mojo 的生产环境用例**：一位成员询问了目前 **Mojo** 在生产环境中的采用情况。
   - 他们表示有兴趣了解 **Mojo** 目前具体被用于哪类工作。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1463995414884909247)** (5 messages): 

> `Mojo REPL 问题，Python 包安装，GitHub Bug 报告` 


- **Mojo REPL 在安装 Python 包时遇到问题**：有成员报告了在尝试使用 `subprocess.check_call` 安装 Python 包 (**scons**) 时，Mojo REPL 出现的问题。
   - 报告中附带了错误信息的截图（[查看附件](https://cdn.discordapp.com/attachments/1151418092052815884/1463995414637187162/Screenshot_from_2026-01-22_13-33-59.png?ex=69752cfa&is=6973db7a&hm=99511e5767f0f4bd190a8ea5bc25af99ccfc0e565bb0cbc85ae99cefd7e0b743&)）。
- **推断参数 (Inferred Parameters) 问题在旧 GitHub Issue 中仍然存在**：一名成员敦促团队检查一个关于推断参数的旧 GitHub issue（[#4199](https://github.com/modular/modular/issues/4199)）。
   - 他们指出该问题似乎依然存在，并建议在调用处通过使用命名参数来绕过该问题。
- **已创建 GitHub Bug 报告**：针对 Mojo REPL 的初始问题，一名成员已创建了 Bug 报告（[#5830](https://github.com/modular/modular/issues/5830)）。
   - 该报告是响应在 GitHub 上提交 Bug 的建议而创建的。


  

---


### **MCP Contributors (Official) ▷ #[general](https://discord.com/channels/1358869848138059966/1358869848138059969/1464327369325023284)** (2 messages): 

> `Caitie Mcaffrey, 联系人` 


- **Caitie Mcaffrey 被提名为联系人**：一名成员询问 *Hey Alexander, Caitie Mcaffrey 提到你是联系人。我可以就某事私信（DM）你吗？*
   - 另一名成员回复表示可以。
- **成员请求私信联系人**：一名成员询问另一名成员是否可以私信。
   - 第二名成员回复道：*可以，请告知*。


  

---


### **MCP Contributors (Official) ▷ #[general-wg](https://discord.com/channels/1358869848138059966/1416012674663452752/1464216774957596735)** (1 messages): 

> `GetTaskPayloadResult, additionalProperties vs anyOf, Model Context Protocol` 


- **MCP 的 `GetTaskPayloadResult` Schema 存在 `additionalProperties` 问题**：Model Context Protocol (`MCP`) 的 `GetTaskPayloadResult` schema 可能会因为使用了 `additionalProperties` 而非 `anyOf` 导致验证过于宽松，正如 [此 issue](https://github.com/modelcontextprotocol/modelcontextprotocol/blob/8d07c35d3857412a351c595fe01b7bc70664ba06/schema/2025-11-25/schema.json#L1245-L1256) 所强调的。
- **`additionalProperties` vs `anyOf`：Schema 语义冲突**：讨论集中在 `GetTaskPayloadResult` schema 中的 `additionalProperties` 是否提供了正确的验证行为，还是说 `anyOf` 能提供预期的更严格验证。
   - 使用 `anyOf` 可能会对 Payload 结果强制执行更具体且可控的结构，确保仅允许预定义的属性。


  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1464322601236299860)** (2 messages): 

> `AI Engineer Europe 大会` 


- **分享 AI Engineer Europe 大会详情**：一名成员询问了某活动的举办时间和地点，表示这是他们第一次听说该活动。
   - 另一名成员提供了 [AI Engineer Europe 大会](https://www.ai.engineer/europe) 的链接。
- **AI 活动地点与时间咨询**：一名成员询问了某活动的日期和地点，并提到他们在 3 月到 5 月期间会在 GR（希腊）。
   - 另一名成员回复了 [AI Engineer Europe 大会](https://www.ai.engineer/europe) 的链接作为回应。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1464205138137841739)** (1 messages): 

> `移动端 GPU Texture vs Buffer 路径, L2 带宽优化` 


- **移动端 GPU 对 Texture 和 Buffer 使用不同的路径**：移动端 GPU 通常具有处理 **Texture** 和 **Buffer** 的独立路径，这种设计选择会影响内存访问模式。
   - 最大化 L2 Bandwidth（L2 带宽）利用率可能需要策略性地同时使用这两种路径，例如将一个输入作为 Texture，另一个作为 Buffer。
- **通过使用 Texture 和 Buffer 优化 L2 带宽**：由于硬件路径独立，在移动端 GPU 上最大化 **L2 Bandwidth** 可能涉及同时利用 **Texture** 和 **Buffer**。
   - 例如，将第一个输入作为 Texture 传入，第二个作为 Buffer 传入，可以优化内存访问并提升整体性能。


  

---

---