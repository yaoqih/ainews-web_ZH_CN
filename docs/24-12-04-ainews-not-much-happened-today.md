---
companies:
- openai
- google-deepmind
- anthropic
- nvidia
- huggingface
date: '2024-12-05T02:41:39.435306Z'
description: '**OpenAI** 宣布了“OpenAI 的 12 天”活动，届时将进行每日直播，并可能发布包括 **O1 完整版模型**、**Sora
  视频模型**以及 **GPT-4.5** 在内的新产品。**Google DeepMind** 发布了 **GenCast 天气模型**，该模型利用 TPU 芯片可在
  8 分钟内完成 15 天的气象预测；此外还推出了 **Genie 2**，这是一个能从单张图像生成可交互 3D 世界的模型。顶尖视觉研究员 **Lucas Beyer**、**Alexander
  Kolesnikov** 和 **Xiaohua Zhai** 已从 DeepMind 跳槽至 OpenAI，后者目前正在苏黎世开设办事处。与此同时，针对 OpenAI
  的策略及其模型质量（相较于 **Anthropic** 的 **Claude 3.5 Sonnet**）的批评声也随之出现。在 Reddit 上，一个修改版的
  **llama.cpp** 现已支持英伟达的 **Llama-3_1-Nemotron-51B**，该模型通过 NAS（神经架构搜索）优化，性能可媲美更大规模的
  70B 模型。'
id: 276161f6-797f-4b69-898f-ddb7654dba8e
models:
- o1-full
- sora
- gpt-4.5
- gpt-4
- claude-3.5-sonnet
- llama-3-1-nemotron-51b
- llama-3-1
- llama-3
- nemotron-51b
original_slug: ainews-not-much-happened-today-1872
people:
- lucas-beyer
- alexander-kolesnikov
- xiaohua-zhai
- aidan_mclau
- giffmana
- joannejang
- sama
title: 今天没发生什么。
topics:
- vision
- model-performance
- neural-architecture-search
- model-optimization
- multimodality
- model-release
- model-training
- reinforcement-learning
- image-generation
---

<!-- buttondown-editor-mode: plaintext -->**另一个平静的日子正是我们所需要的。**

> 2024年12月3日至12月4日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务器（**198** 个频道，**2915** 条消息）。预计节省阅读时间（以 200wpm 计算）：**317 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

*Smol.ai 更新：[Smol Talk 现在具备视觉能力了！](https://www.loom.com/share/34b37822c6784989bafd6fcc5fee6420?sid=75bf3b4c-61b5-46fd-a2b1-7c7fe911df89) 以前如果它遇到图像会产生幻觉（hallucinate），现在我们进行了必要的提示词工程（prompting）。请参阅今天的 Reddit 回顾中的示例，现在您的个性化回顾也将包含这些内容。*

**如果您对下周的 NeurIPS 感兴趣，我们的[年终回顾活动](https://lu.ma/LSLive)还剩 50 张门票（提供直播，不需要 NeurIPS 门票）。[大多数演讲嘉宾已经公布](https://x.com/swyx/status/1864423257266639166)。**

[Genie 2](https://news.ycombinator.com/item?id=42317903) 全天占据 HN 榜首，我们[之前报道过 SIMA](https://buttondown.com/ainews/archive/ainews-deepmind-sima-one-ai-9-games-600-tasks/)，但鉴于这仍然是（令人印象深刻的）精选演示（cherrypickware），我们没有将其作为头条新闻。

o1-full 预计将在其[新的降临节日历活动（advent calendar）](https://x.com/OpenAI/status/1864328928267259941)期间发布，与此同时，他们[从 DeepMind 挖走了多名研究员](https://x.com/iScienceLuvr/status/1864217903232385348)。也许 [OpenAI 真的回归了（openai is so back）](https://x.com/tszzl/status/1863882905422106851)。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是来自 Twitter 数据的关键主题和讨论，按主题分类：

**OpenAI 的“圣诞 12 天”发布公告**

- **重大产品发布**：[@sama](https://twitter.com/sama/status/1864335461268754712) 和 [@OpenAI](https://twitter.com/OpenAI/status/1864328928267259941) 宣布从明天开始举办“OpenAI 的 12 天”活动，每天都会有包含发布和演示的直播。社区正在猜测可能发布的产品，如 **o1 全尺寸模型**、**Sora 视频模型**和 **GPT-4.5**。
- **发布物流**：[@joannejang](https://twitter.com/joannejang/status/1864344210327130357) 指出了连续发布 12 个公告的挑战，并建议了备选方案，比如在需要时让高管表演杂耍。

**DeepMind 的重大研究发布**

- **GenCast 天气模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1864340994965098513) 在 Nature 上发布了一个 AI 天气预报系统，可以使用 TPU 芯片在 **8 分钟内做出 15 天的预测**，具有最先进的准确率。
- **Genie 2 世界模型**：[@GoogleDeepMind](https://twitter.com/GoogleDeepMind/status/1864367798132039836) 发布了一个可以从单张图像创建可交互 3D 世界的模型，旨在虚拟环境中训练未来的 AI Agent。

**高端人才变动**

- **视觉研究团队加入 OpenAI**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1864217903232385348) 报道称，顶尖计算机视觉研究员 **Lucas Beyer**、**Alexander Kolesnikov** 和 **Xiaohua Zhai** 已从 Google DeepMind 转投 OpenAI。[@giffmana](https://twitter.com/giffmana/status/1864419226649546883) 确认他们将在苏黎世开设办公室。

**对 AI 模型质量的批评**

- **OpenAI 战略担忧**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1864367068314730778) 批评了 OpenAI 与客户竞争但模型质量却落后的战略，建议他们应该像 Anthropic 那样专注于构建优秀的模型。
- **模型性能**：多位用户指出，尽管 **Claude/Sonnet** 价格更低，但其表现优于其他模型，引发了关于不同 API 定价策略相对优劣的辩论。

**梗与幽默**

- [@scaling01](https://twitter.com/scaling01/status/1864330169898684622) 开玩笑说想要“computer use agents sora o1 GPT-5 全多模态 4o 更便宜的 o1 模型”

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Nemotron-51B 发布：Nvidia 的 NAS 优化模型性能媲美 70B**

- **修改 llama.cpp 以支持 Llama-3_1-Nemotron-51B** ([分数：79，评论：31](https://reddit.com/r/LocalLLaMA/comments/1h6724m/modified_llamacpp_to_support_llama3_1nemotron51b/))：一位开发者成功修改了 **llama.cpp** 以支持 **Nvidia 的 Llama-3_1-Nemotron-51B** 模型。该模型通过 **Neural Architecture Search (NAS)** 优化，性能与体量更大的 **70B** 版本相当。修改后的模型已发布在 [HuggingFace](https://huggingface.co/ymcki/Llama-3_1-Nemotron-51B-Instruct-GGUF) 上，提供 **Q3_K_S**、**Q4_0**、**Q4_0_4_8** 和 **Q4_K_M** 等量化选项，并有望集成到 **llama.cpp** 的主分支中。
  - **51B 模型** 的 **Q3_K_S** 量化版本表现优于 **70B 模型** 的 **IQ2_XS**，用户在实际测试中确认了效果的提升。**51B Q3_K_S** 版本需要 **22.7GB** 的 VRAM。
  - 技术讨论显示，**51B 模型** 的 **IQ4_XS** 量化大约需要 **27.84GB** VRAM，超过了 **3090** GPU 的容量，而 **70B** 模型的同等量化则需要 **37.9GB**。
  - 在没有 **imatrix** 的情况下，较低的量化级别会导致性能下降，这在 **Q2_K_S** 实现中得到了证实。官方性能数据可参考 [NVIDIA 的博客文章](https://developer.nvidia.com/blog/advancing-the-accuracy-efficiency-frontier-with-llama-3-1-nemotron-51b/)。


**主题 2. 动态 4-bit 量化：通过选择性层精度获得更好性能**

- **全量化至 4-bit 会破坏模型 - 动态量化 10% FP16 与 90% 4-bit** ([分数：119，评论：50](https://reddit.com/r/LocalLLaMA/comments/1h6ojwr/quantizing_to_4bits_can_break_models_dynamic/))：**Unsloth** 研究人员发现，将所有层都量化为 **4-bit** 精度会降低模型性能。他们以 **Qwen2-VL-2B Instruct** 为例进行了演示：全 4-bit 量化会导致错误的图像描述，而使用 **10% FP16** 和 **90% 4-bit** 精度则能在保持准确性的同时，将模型大小从 **4.11GB** 压缩至 **1.81GB**。对 **Llama 3.2 11B Vision Instruct** 的分析揭示了 **MLP 层** 存在显著的激活错误，以及 **Cross Attention 层** 存在权重量化错误。为此，他们在 **HuggingFace** 上发布了新的动态量化模型，实现了 **2倍更快** 的推理速度并减少了 **50% 的 VRAM** 占用。
  - **Unsloth** 开发者确认 **QwQ 动态量化** 同时适用于视觉和文本模型，其首个文本模型 [QwQ-32B-Preview](https://huggingface.co/unsloth/QwQ-32B-Preview-unsloth-bnb-4bit) 现已在 HuggingFace 上线。他们指出，**视觉编码器 (vision encoders)** 通常不应使用 **4-bit** 量化，特别是在基于 **Llava** 的模型中。
  - 用户对实现这些混合量化技术表现出浓厚兴趣，讨论集中在 **GGUF 量化** 的相似性，以及对本地 VLM 部署所需的 **兼容 OpenAI 的 API 服务器** 的需求。开发者表示计划将此功能集成到更广泛的 **Unsloth** 框架中。
  - 研究团队分享了额外的分析图表，展示了 4-bit 量化中的 **激活峰值 (activation spikes)**，模型配置文件标出了有问题的层。社区反应非常积极，尤其是对其详细的模型调试方法表示赞赏。


**主题 3. FishSpeech v1.5：多语言零样本语音克隆突破**

- **FishSpeech v1.5 - 多语言、零样本即时语音克隆、低延迟，仅 500M 参数 - TTS-Arena 排名第 2** ([分数：91，评论：10](https://reddit.com/r/LocalLLaMA/comments/1h6p335/fishspeech_v15_multilingual_zeroshot_instant/))：**FishSpeech v1.5** 是一款多语言语音克隆模型，在包含 **13 种语言** 的 **100 万小时** 数据上进行了训练。它在 **TTS-Arena** 上排名 **第 2**，同时仅凭 **5 亿参数** 保持了 **<150ms 的延迟**。该模型目前已开源，可通过 [fish.audio](http://fish.audio/)、[GitHub](http://github.com/fishaudio/fish-speech) 和 [Hugging Face](http://huggingface.co/spaces/fishaudio/fish-speech-1) 等多个平台获取，提供自托管和云端部署选项。
  - 用户询问了关于 **语音克隆能力** 以及添加类似 **Bark** 的 **情感范围** 的问题，突出了 TTS 技术未来发展的关键领域。
  - 该模型带有其 [Hugging Face](https://huggingface.co/fishaudio/fish-speech-1.5) 仓库中指定的 **非商业许可限制**。


**主题 4. 字节跳动实习生风波：800 万人民币诉讼获胜者摘得 NeurIPS 最佳论文奖**

- **前实习生破坏 ByteDance 的 AI 训练，面临 800 万人民币诉讼，却斩获 NeurIPS 2024 最佳论文** ([Score: 79, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1h6i1m9/former_intern_sabotages_bytedances_ai_training/)): **Keyu Tian**，一名 **ByteDance** 前实习生，因涉嫌在 **2024 年 8 月** 破坏公司涉及 **8,000 多个 GPU** 的 AI 模型训练，面临 **800 万人民币** 的诉讼，据称造成了数千万美元的损失。尽管存在法律争议，**Keyu Tian** 仍凭借其在 **ByteDance** 实习期间进行的研究赢得了 **NeurIPS 2024 最佳论文奖**，其论文 "[VAR](https://arxiv.org/abs/2404.02905)" 是与该公司的商业化技术部合作开发的。
  - 根据 **ByteDance 的官方声明**，该实习生仅恶意干扰了**商业化技术团队**的模型训练，并未影响其他业务运营。公司澄清称，“**8,000 个 GPU**”和“**数千万**”损失的说法**被严重夸大**。
  - **Keyu Tian** 已于 **8 月** 被辞退，此事已通报其就读大学及行业联盟。该事件具体影响了其团队内部的研究项目，未涉及 **ByteDance** 的 **AI Lab** 或大模型。
  - 技术专家指出，现代 AI 训练包含广泛的 **logging**、**real-time analytics** 和 **checkpoint testing**，因此整个模型训练成果全部丢失的可能性较低。损失可能主要源于 **GPU** 集群停机带来的机会成本。


## 其他 AI 子版块回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. OpenAI “12 Days of Shipmas” 活动将包括 Sora 和 O1 模型发布**

- **[OpenAI 的 “12 Days of Shipmas” 包括 Sora 和新的推理模型](https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas)** ([Score: 203, Comments: 60](https://reddit.com/r/OpenAI/comments/1h6ib0o/openais_12_days_of_shipmas_include_sora_and_new/)): **OpenAI** 宣布了为期 **12 天的产品发布计划**，其中包括他们新的 **Sora** 视频生成模型和 **O1** 推理模型。目前尚未提供关于这些模型的具体发布日期或技术能力的更多细节。
  - **Sam Altman 的推文**确认了每日都会有产品发布和演示的直播，但社区成员对实际发布表示怀疑，并指出 **OpenAI** 有过宣布功能为“*将在几周内推出*”但未立即部署的历史。
  - 关于**计算资源**的讨论表明，**O1** 从预览版过渡到稳定版不会显著增加系统负载，而社区则在推测 **OpenAI** 是否有足够的 **GPU** 容量来同时处理像 **Sora** 这样多个重大发布。
  - 为**高级语音模式 (Advanced Voice Mode)** 宣布的 **Santa Voice** 功能引发了对潜在亲子互动的兴奋，尽管一些用户开玩笑地引用了标准的 AI 模型免责声明：“*对不起，作为一个语言模型，我不能给你带玩具*”。


- **[接下来会发生什么？你的猜测是什么？](https://i.redd.it/tplh9liduu4e1.jpeg)** ([Score: 392, Comments: 126](https://reddit.com/r/OpenAI/comments/1h6jjrt/whats_coming_next_whats_your_guess/)): **OpenAI** 宣布了 “**12 Days of OpenAI**”，从明天开始将进行一系列共 **12 场直播**，届时将发布各种公告。社区对这些公告的内容进行了推测，**OpenAI** 将其描述为从“大到小”的各种进展。
  - 社区期待集中在 **O1**、**Sora** 和 **Operator** 的发布上，许多用户提到 **Anthropic 的 MCP 发布** 给 **OpenAI** 带来了交付压力。获赞最多的评论对能否及时获得所宣布的功能表示怀疑。
  - 用户预测这将是立即发布和未来承诺的混合体，并对 **GPT-4 Mini 更新**、**更便宜的实时 API 定价**以及**高级语音模式**功能表现出浓厚兴趣。一些评论认为这些公告的时机可能是为了与 **Google/Gemini** 竞争。
  - 技术推测侧重于潜在的 **Agent 模型**、**无限记忆**功能和**全浏览器控制能力**。大多数开发者表示，相比于华丽的公告，他们更渴望看到像更好的 **API** 定价这样实用的改进。


**主题 2. 新开源 AI 视频模型：Tencent Hunyuan vs LTX 对比**

- **[腾讯全新的开源 AI 文本生成视频模型 Hunyuan 可以实现弹跳物理效果。一切都结束了。](https://v.redd.it/mmjvx1xbjs4e1)** ([Score: 771, Comments: 120](https://reddit.com/r/ChatGPT/comments/1h6b9h8/tencents_new_open_source_ai_texttovideo_model/)): **Tencent** 在 **HuggingFace** 上发布了他们的 **Hunyuan** 文本生成视频模型，访问地址为 [Tencent-Hunyuan-Large](https://huggingface.co/tencent/Tencent-Hunyuan-Large)。由于无法访问引用的视频内容，无法验证关于物理能力或模型性能的具体说法。
  - 用户注意到该模型令人印象深刻的**物理模拟能力**，特别是**头发运动**和其他动态元素，并将其与 **GTA VI** 和 **Stellar Blade** 等游戏进行了比较。
  - 社区讨论了中国公司发布模型的**开源动机**，**Tencent** 的官方声明称其目标是*“通过创新想法启发更多研究人员，共同推动 AI 技术的进步”*。正确的模型链接已分享至 [HunyuanVideo](https://huggingface.co/tencent/HunyuanVideo)。
  - 多条评论对 **AI-generated content** 可能颠覆各行各业表示担忧，并预测未来几年内，某些在线内容的很大一部分将由 AI 生成。


- **[LTX Video vs. HunyuanVideo 在 20 个提示词下的对比](https://v.redd.it/y6comqv9lw4e1)** ([Score: 60, Comments: 57](https://reddit.com/r/StableDiffusion/comments/1h6sdsp/ltx_video_vs_hunyuanvideo_on_20x_prompts/)): 无法提供有意义的摘要，因为帖子正文为空且无法分析视频内容。准确的摘要需要标题中提到的关于 **LTX** 和 **HunyuanVideo** 模型的实际内容、讨论点或对比分析。
  - **Hunyuan** 需要大量的计算资源，在 **720x1280** 分辨率下至少需要 **60GB GPU memory**，生成一段 6 秒的视频需要 **2 小时**。用户指出，当模型适配 **VRAM** 时，在 **544x960** 分辨率下性能为 **15 分钟**，而当溢出到 **RAM** 时则需要 **2 小时**。
  - 对比方法受到质疑，因为 **LTX** 受益于 **100+ step counts**，而测试中显然只使用了 **10 steps**。批评者指出，**LTX** 需要详细的提示词，且仍处于 **version 0.9** 训练阶段。
  - 完整的对比可以在 [checkbin.dev](https://app.checkbin.dev/snapshots/70ddac47-4a0d-42f2-ac1a-2a4fe572c346) 查看，用户指出虽然 **Hunyuan** 为开源视频模型带来了希望，但未来的**量化版本（quantized versions）**可能会改善目前对 **A100** GPU 的硬件要求。


**Theme 3. OpenAI 周活跃用户达到 3 亿，签署国防合同**

- **[ChatGPT 现在拥有超过 3 亿周活跃用户](https://www.theverge.com/2024/12/4/24313097/chatgpt-300-million-weekly-users)** ([Score: 200, Comments: 19](https://reddit.com/r/OpenAI/comments/1h6m4so/chatgpt_now_has_over_300_million_weekly_users/)): **ChatGPT** 已实现 **3 亿周活跃用户**，标志着这款 **OpenAI** 聊天机器人的用户基数达到了一个重要的里程碑。
  - **300M weekly users** 展示了显著的主流采用率，用户将 **ChatGPT** 与 **Google** 的搜索统治地位进行比较，并指出其颠覆传统搜索商业模式的潜力。
  - 用户强调 **ChatGPT** 代表了一场真正的技术革命，许多人将其比作“世界上最聪明的人”，可以帮助完成无穷无尽的任务，尽管仍有人将其误认为是像 **NFTs** 或 **cryptocurrency** 那样的噱头。
  - 讨论集中在变现策略上，用户在订阅模式和基于数据的收入之间进行辩论，同时表达了希望 **OpenAI** 不要像传统搜索引擎那样诉诸广告变现。

- **[OpenAI 的新国防合同完成了其军事转型](https://www.technologyreview.com/2024/12/04/1107897/openais-new-defense-contract-completes-its-military-pivot/?utm_medium=tr_social&utm_source=reddit&utm_campaign=site_visitor.unpaid.engagement)** ([Score: 31, Comments: 22](https://reddit.com/r/OpenAI/comments/1h6odpi/openais_new_defense_contract_completes_its/)): **OpenAI** 尚未正式宣布任何国防合同或军事应用，这似乎是缺乏可靠来源或正文分析的误导性信息。在没有可验证内容参考的情况下，无法提供事实摘要。
  - **OpenAI** 宣布与国防科技公司 **Anduril** 建立合作伙伴关系，部署 AI 模型以防御无人机攻击，重点是为**美国及其盟军**提供数据综合和态势感知。
  - 该伙伴关系专门针对**无人机威胁**，旨在保护**美国人员和设施**。发言人 **Liz Bourgeois** 强调这符合公司政策，且不会开发有害系统。
  - 社区反应对 **AI safety** 的主张表示怀疑，并注意到 **Sam Altman** 与 **Palmer Luckey** 之间的合作，对公司宣称的安全优先事项持讽刺态度。


**Theme 4. Claude 3.5 vs ChatGPT: 用户迁移与对比趋势**

- **Claude 3.5 如何帮助我击退了 10,000 美元的租车损坏索赔并获胜** ([Score: 99, Comments: 21](https://reddit.com/r/ClaudeAI/comments/1h6pxdn/how_claude_35_helped_me_fight_off_a_10000_rental/)): **Enterprise** 租车公司试图向一名用户收取 **10,000 美元** 的损坏费，声称其**损毁免责险 (LDW)** 仅适用于商务旅行，尽管该免责险是在通过母校租车计划预订时自动包含且无法移除的。通过使用 **Claude 3.5** 分析租赁文件和往来信件，用户发现承保条款中不存在商务用途限制，并在学校**风险管理办公室**的支持下成功对索赔提出异议，最终导致 **Enterprise** 完全放弃了 **10,000 美元** 的费用。
  - 一名用户目前正利用 **Claude** 在一审程序中对一项 **30,000 美元** 的保险索赔提出抗辩，展示了 AI 在法律文件分析中的效用。该案例显示了在不升级法律行动的情况下解决问题的潜力。
  - 用户强调了**人机协作**在法律纠纷中的有效性，在提供完整背景和文件的情况下，**Claude** 在文件分析和证据发现方面表现出卓越的准确性。
  - 多位用户报告 **Enterprise** 的服务质量下降，其中一人详细描述了收到的租赁选项是严重损坏的 **Ram 1500** 和高里程的 **Chrysler 300c**，而另一人确认在 **10,000 美元** 损坏索赔事件后失去了他们的业务。


- **[你也注意到这个模式了吗？](https://i.redd.it/y1tbmo0l2w4e1.png)** ([Score: 50, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1h6pt4s/have_you_noticed_this_pattern_too/)): **@Aella_Girl** 的一条推文观察到，越来越多的人在个人建议和决策方面从 **ChatGPT** 转向 **Claude**。该推文在 **2024 年 12 月 4 日** 获得了显著关注，拥有 **284,600 次查看**、**2,100 个赞**、**171 次转发**和 **98 条评论**。
  - 用户强调 **Claude** 能够提供**细致入微的回答**并对糟糕的想法**提出异议**，尽管对于新用户来说，它可能比 **ChatGPT** 更难上手。默认的 **Claude** 个性更具对话性，而 **ChatGPT** 的回答则较为平淡。
  - 一位用户分享了他们在 **Claude** 上使用 **"Style > Intellectual Inquisitor"** 提示词的成功经验，该提示词创建了一种专注于解构论点和识别逻辑谬误的分析思维。他们针对不同目的仅维持 **3 种不同的风格**。
  - 尽管存在个人偏好，**ChatGPT** 仍保持着**市场领先地位**，不过 **Claude** 在 **X** (Twitter) 上的受欢迎程度被视为一个重要信号。用户强调应根据有效性而非品牌忠诚度来选择工具。


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要的摘要

**主题 1：Amazon 发布 Nova AI 模型，撼动 AI 领域格局**

- [**Amazon 发布六款 Nova 新模型，对标 GPT-4**](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)：Amazon 在 re:Invent 大会上发布了 **Nova** 系列的六款新基础模型，旨在与 GPT-4 竞争，支持高达 **300K tokens** 的上下文和 **200 多种语言**。

- [**用户热议 Nova 的速度与定价**](https://x.com/_philschmid/status/1864016010464080260)：早期用户对 Nova 令人印象深刻的**速度**和极具竞争力的**定价**感到兴奋，并热切期待其集成到 **Perplexity Pro** 等平台中。

- [**AWS Bedrock 随 Nova 发布获得强力升级**](https://aws.amazon.com/bedrock/pricing/)：Amazon 的 Nova 模型仅通过 **Amazon Bedrock** 提供，这增强了 AWS 的 AI 产品线并影响了开发者的选择。


**主题 2：OpenAI 的“12 天发布活动”引发高度期待**

- [**OpenAI 预告“12 Days of OpenAI”；社区反响热烈**](https://x.com/OpenAI/status/1864328928267259941)：OpenAI 宣布从明天开始进行为期 **12 天**的新品发布和演示直播，引发了 AI 社区的兴奋和猜测。

- [**关于 OpenAI 即将发布的惊喜传闻四起**](https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas)：用户猜测可能发布的内容包括界面更新、**ChatGPT** 的新功能，甚至是**文本生成视频 AI 工具**。

- [**开发者为 OpenAI 的重大发布做好准备**](https://x.com/sama/status/1864335461268754712)：社区正为重大公告做准备，期待能出现改变其项目和工作流的工具与改进。


**主题 3：Cursor IDE 宕机促使用户转向替代方案**

- [**Cursor 崩溃；开发者转投 Windsurf**](https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221)：**Cursor IDE** 面临宕机和性能问题，导致沮丧的用户退回到使用 **ChatGPT** 或转向 **Windsurf** 寻求代码辅助。

- [**移除长上下文模式引发用户抵制**](https://forum.cursor.com/t/long-context-mode-gone-in-newest-update/29449)：Cursor 取消了**长上下文模式 (long context mode)** 等关键功能以及界面更改，引发了广泛的不满和抵制。

- [**Cursor 陷入困境，Windsurf 顺势崛起**](https://discord.com/channels/1074847526655643750)：随着 Cursor 出现问题，**Windsurf** 作为一个可靠的替代方案脱颖而出，因其能更好地处理编码任务且不会产生不必要的代码改动而获得赞誉。


**主题 4：NVIDIA 的 SANA 模型因严苛的许可协议遭抨击**

- [**快而严苛：NVIDIA SANA 的许可协议引发愤怒**](https://x.com/cloneofsimo/status/1864309440356470894)：**SANA** 模型以其速度令人印象深刻，但其限制性的**非商业许可**和仅限 NVIDIA GPU 使用的要求激怒了用户。

- [**开发者对 SANA 的 GPU 锁定感到愤怒**](https://x.com/cloneofsimo/status/1864312857674043599)：社区批评 NVIDIA 限制 SANA 在 **AMD 机器**上使用，并保留对生成内容的权利。

- [**SANA 的许可失误促使用户寻找其他选择**](https://nvlabs.github.io/Sana/)：受挫于 SANA 的限制性条款，开发者正转向其他替代模型和开放获取的选项来开展其 AI 项目。


**主题 5：Pydantic AI 通过新集成助力开发加速**

- [**Pydantic AI 联手 DSLModel 和 DSPy；开发者欢呼**](https://ai.pydantic.dev/)：**Pydantic AI** 与 **DSLModel** 及 **DSPy** 的集成提供了一个增强的 **Agent** 框架，简化了 AI 开发。

- [**直播演示承诺掌握 AI 开发的魔力**](https://youtube.com/live/mBQFKo8bPBI "Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive")：即将举行的名为“*Master AI Development*”的直播演示将深入探讨如何结合 **PydanticAI**、**DSPy** 和 **DSLModel**。

- [**编码未来：Pydantic AI 让 LLM 使用变得轻而易举**](https://ai.pydantic.dev/)：开发者赞扬 Pydantic AI 使大语言模型集成变得无缝，尤其是与 **FastAPI** 等熟悉工具配合使用时。


---

# 第一部分：Discord 高层级摘要

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 遭遇服务中断**：许多用户报告 **Cursor 正在经历停机**，导致严重的延迟和无法生成响应。
   - 用户对状态更新和响应质量的缺乏表示沮丧，一些人转回使用 **ChatGPT** 或切换到 **Windsurf**。
- **Cursor 功能变更引发担忧**：**Cursor** 中 **long context mode** 的移除以及最近的界面更改引起了用户的广泛不满。
   - 许多用户注意到模型响应效果下降，暗示可能存在模型质量降级或性能问题。
- **Windsurf 成为可靠的替代方案**：一些用户报告 **Windsurf** 是一个值得信赖的替代品，声称它在不显著改动代码的情况下能更好地处理编程任务。
   - 这引发了关于 **Cursor** 最近的更新是否是对 **Windsurf** 功能和日益增长的普及率的直接回应的讨论。
- **OpenAI 宣布为期 12 天的更新**：**OpenAI** 将从明天开始，在接下来的 **12 天** 内每天发布新更新，这引起了用户的兴奋。
   - 用户希望这些公告能为现有工具带来改进，从而可能解决 **Cursor** 最近面临的挑战。
- **Cursor 性能问题持续存在**：开发者指出，**Cursor** 最近的更新不仅减慢了响应速度，还增加了代码编辑中的错误。
   - 用户正在质疑这些更改的有效性，并寻求潜在的解决方案或变通方法。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX 在 TPU 性能上优于 PyTorch**：关于 **JAX** 在大型 AI 实验室中是否优于 **PyTorch** 的争论激增，特别是关于 TPU 利用率与 PyTorch 在 GPU 方面的优势。
   - 观点各异，一些成员强调了 [Hacker News 的讨论](https://news.ycombinator.com/item?id=39876444)，重点讨论了 JAX 在 TPU 上的效率，而其他人则指出了 PyTorch 在 GPU 任务中的广泛采用。
- **Apple 利用 AWS 定制 AI 芯片**：在一次 [AWS 活动](https://www.macrumors.com/2024/12/03/apple-amazon-ai-chips-search/)中，**Apple** 宣布其在搜索服务中使用了 AWS 定制的 **Inferentia** 和 **Graviton** AI 芯片。
   - 尽管有此合作，讨论指出 **Apple** 在其广泛的机器学习工作负载中仍继续偏好 GPU 解决方案。
- **对二阶优化器持怀疑态度**：成员们质疑 **二阶优化器 (second-order optimizers)** 在非凸优化中的有效性，理由是与 **AdamW** 相比，其实证结果褒贬不一。
   - 虽然有些人认为二阶优化器在微小特征值下可能表现出色，但共识倾向于认为没有显著的性能提升，正如最近的社区研究所强调的那样。
- **Mira Virtual AI 在 2GB VRAM 上赋能多模态任务**：**Mira Virtual AI** 作为一个 [GitHub 项目](https://github.com/Mirror-Prismals/Mira-Virtual-Ai)被推出，提供仅需 **2GB VRAM** 即可在消费级硬件上运行的多模态转换工具。
   - 这些独立脚本专为编程经验有限的用户设计，旨在让 AI 实验变得触手可及，并为多模态工作流注入 **乐趣和自动化**。
- **通过外部可加载评估增强 lm-eval-harness**：有人提议通过 [Hugging Face](https://huggingface.co) 在 **lm-eval-harness** 中启用外部可加载评估，从而实现数据集和评估配置的无缝集成。
   - 针对可复现性和数据集版本控制提出了担忧，尽管 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197) 目前支持一些外部评估功能，但挑战依然存在。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI 翻译工具大对决**：成员们讨论了各种 **AI translation tools**，相比 **Google Translate** 和 **Microsoft** 的替代方案，他们更倾向于 **DeepL**，因为其准确度更高。建议包括利用 **Cohere's API** 和使用 **open-webui filters** 来增强聊天机器人的多语言能力。
   - 社区强调了 AI 应用中精确翻译的重要性，并讨论了通过潜在的集成来优化针对不同用户群体的语言支持。
- **GPT 停止图像处理**：有成员报告 **GPT** 不再能够处理图像，这引发了对该功能变化所产生影响的担忧。这一调整标志着 **GPT's functionalities** 的重大转变。
   - 这一限制激发了成员们对其背后原因的好奇，以及它将如何影响未来的 **AI workflows**。
- **投票系统中的量子计算**：讨论探索了通过高级算法将 **quantum computing** 应用于增强投票系统。成员们辩论了量子算法在现实投票场景中的实用性。
   - 一种观点指出 *选民并非处于叠加态 (superposition)*，质疑量子技术在选举过程中的直接益处。
- **Cohere AI 在匈牙利语翻译方面表现出色**：**Cohere AI** 平台因支持包括 **Hungarian** 在内的 **100 多种语言**且翻译准确率极高而受到认可。成员们分享了他们对 **Cohere AI's multilingual capabilities** 的正面体验。
   - [Mark Johns 的 YouTube 视频](https://www.youtube.com/watch?v=nUa_r9GKjtI)和 [OpenEmpathic 项目](https://dct.openempathic.ai/)等资源被引用为在多语言项目中使用 **Cohere AI** 的宝贵工具。
- **创新的 Prompt Engineering 技术**：成员们交流了增强 Prompt Engineering 的策略，包括使用 **YAML structures** 和 **markdown formatting** 来提高提示词的清晰度和上下文。重点强调了在构建有效提示词时 **contextual attention** 的重要性。
   - 讨论还涉及了评估提示词有效性的挑战，以及 **API automation** 作为各种提示词策略测试场的潜力。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Amazon Bedrock Nova 模型发布**：Amazon 发布了全新的 **Nova** 系列基础模型，仅通过 [Amazon Bedrock](https://aws.amazon.com/blogs/aws/reduce-costs-and-latency-with-amazon-bedrock-intelligent-prompt-routing-and-prompt-caching-preview/) 提供，其特征是上下文长度高达 **300K tokens**。
   - 性能可与 **Llama 3** 媲美，并针对不同的模型能力提供了极具竞争力的定价。
- **Aider 的新 watch-files 功能**：[Aider](https://aider.chat/docs/usage/browser.html) 中新引入的 `--watch-files` 功能允许通过 AI 注释与代码进行无缝交互，并根据指定的标记触发操作。
   - 早期反馈称赞该功能是一项重大进步，尽管文档仍在完善中。
- **QwQ 模型表现不佳**：**QwQ 32B Preview** 模型在整体编辑格式上得分为 **54%**，在 diffs 上得分为 **50%**，低于预期。
   - 鼓励用户考虑使用 **Qwen** 或 **Sonnet** 模型以获得更好的结果，这反映了对 QwQ 实际效用的担忧。
- **Aider Docker 设置与超时挑战**：成员们讨论了使用共享卷设置 [Aider in Docker](https://aider.chat/docs/install/docker.html)，在对齐 CentOS 容器中的用户设置时遇到了 'Permission denied' 错误。
   - 此外，在使用 `--timeout 5000` 的本地服务器运行 Aider 时，超时问题仍然存在，可能是由于 litellm 的 bug 导致的。
- **MCP 的采用与 OpenAI 的开发策略**：**MCP** 被成员们视为未来的基石，社区对其采用表现出浓厚兴趣。
   - 有人担心 **OpenAI** 可能会选择*另起炉灶 (reinvent the wheel)*，而不是将 MCP 集成到他们的开发策略中。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 网络功能等待更新**：讨论强调了 **Mojo 网络能力** 的持续开发，目标是通过 [io_uring](https://github.com/marti) 的改进，实现单核 **25-40 Gbps 的 TCP 吞吐量**。
   - 成员们强调，更新后需要高效的 **API 设计** 以满足现代需求。
- **在 Mojo 中探索 SIMD 操作**：成员们探索了在 **Mojo** 中使用 [SIMD](https://github.com/simdjson/simdjson) 操作，并指出其实现比 C/C++ intrinsics 更易用。
   - **Darkmatter** 建议将大多数 SIMD intrinsics 嵌入到标准库中，以减少对直接 intrinsic 调用的依赖。
- **开发高性能文件服务器**：一位成员分享了为游戏开发 **高性能文件服务器** 的计划，目标是比 Nginx 的 200 字节 HTTP header 解析提高 **30% 的 packets/s**。
   - 讨论的策略包括实现效率以及对强大网络 API 支持的必要性。
- **提出内联引用（Inline References）概念**：提议引入 `InlineReference` 类型，在不存储地址的情况下促进内存高效的访问模式，可能通过启用 **连续内存读取** 来增强性能。
   - 讨论涉及平衡 **引用可用性** 和 **编译器可见性**，并对集成该功能表示关注。
- **Mojo 中的内存优化策略**：专注于 **小字符串和向量优化**，成员们强调这些优化可以通过在大数组扫描期间启用 **零拷贝场景（zero-copy scenarios）** 来提升 **性能**。
   - 成员们对这些优化的实际用例和有效实现方法表达了兴趣。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **动态 4-bit 量化**：Unsloth 推出了 [动态 4-bit 量化](https://x.com/UnslothAI/status/1864380913922265300)，与传统的 4-bit 方法相比，在减少 VRAM 使用的同时增强了 **模型准确性**。
   - 该方法动态地选择不对某些参数进行量化以防止精度损失，要求用户将其模型重命名为 'unsloth-bnb-4bit' 以激活该模式。
- **Llama 3 微调挑战**：用户在微调 **Llama 3** 时遇到错误，由于 `llama.cpp` 中缺少文件，在将模型保存为 GGUF 格式时遇到运行时问题。
   - 通过切换 notebook 版本的解决尝试均告失败，目前唯一的权宜之计是使用 **Unsloth 框架** 进行 GGUF 转换。
- **GGUF 转换技术**：在 **GGUF 转换挑战** 中，社区成员正在探索替代方法和 **Colab 设置** 以正确转换模型，主要利用 **Unsloth 框架**。
   - 参与者分享了 [Colab 资源](https://colab.research.google.com/drive/12hkbsOMJfYfmqJLA93cV5tGoPIeZ5gDK#scrollTo=oAC_WYSUX7k_) 和潜在解决方案，以应对当前转换过程中的限制。
- **持续预训练（Continued Pretraining）的作用**：社区强调了 **持续预训练 (CPT)** 对 **Llama 3** 等模型的重要性，使其能够适应新领域并有效地获取新 token。
   - 虽然基础模型在大数据集上经过了广泛的预训练，但 **CPT** 对于法律和医学等领域的专业应用仍然至关重要，以保持相关性和准确性。
- **Claude vs CodeLlama：模型性能**：关于 **Claude** 和 **CodeLlama** 的对比引发了辩论，成员们认为 **CodeLlama** 已经过时，并主张 **Qwen2.5-coder** 是更优的替代方案。
   - **Qwen2.5-coder** 被指出能提供类似于 **Claude** 的性能，巩固了其在当前模型讨论和应用中的地位。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Amazon Nova 模型发布**：[Amazon Nova 的发布](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)以其**速度**和**准确性**给用户留下了深刻印象，引发了将其集成到 **Perplexity Pro** 的热切期待。
   - 早期实验显示了积极的反馈，强调了 Nova 在工程社区中执行高性能 AI 驱动任务的潜力。
- **Perplexity Pro 订阅问题**：用户对 **Perplexity Pro** 的订阅成本表示不满，特别是从 **首月 4.99 美元** 定价过渡到更高费用时缺乏明确的沟通。
   - 这引发了关于支持学生免费访问的财务模型以及对 **API 访问** 和 Pro 功能影响的广泛讨论。
- **Perplexity API 质量担忧**：成员们提出了关于 **Perplexity API 质量** 的重大问题，指出它在某些用例中已变得**无法使用**。
   - 随着多位用户表达不满，人们开始猜测潜在的供应商变更以及 API 性能持续面临的挑战。
- **Mac 上的用户界面问题**：**Perplexity AI 的 Mac 应用程序**因**性能缓慢**以及与 Web 版本相比界面尴尬而受到批评。
   - 用户还报告了**电池耗尽**问题，引发了关于即将到来的修复和改进的讨论。
- **Heisenberg Heat 咨询**：发起了一场围绕 **Heisenberg Heat** 概念的讨论，邀请大家探索其原理及其对 AI 工程的影响。
   - 鼓励成员深入研究分享链接中提供的相关**理论探究**和**实际应用**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Claude 3.5 Haiku 降价**：OpenRouter 宣布 **Claude 3.5 Haiku 降价 20%**，旨在提高该模型的可访问性。
- **Hermes 405B 服务终止**：**Hermes 405B** 的免费服务已停止，可能是由于供应商的决定，导致用户感到失望。
   - 尽管服务终止，**基础 405B 模型**仍可免费使用，促使一些用户探索替代方案。
- **Gemini Ultra 访问限制**：**Gemini 1.0 Ultra** 目前受白名单限制，在可能停用的担忧中流传着可用性的传闻。
   - 用户对 Google 模型的推出和版本控制感到困惑，推测 Ultra 可能会在不久后停用。
- **Amazon Nova 用于创意写作**：人们对 **Amazon Nova** 模型在创意写作任务中的有效性感到好奇，用户正在寻求个人经验分享。
   - 随着评估的继续，Nova 与 Runway 等替代方案相比的能力规格仍不确定。
- **Custom Provider Keys Beta 测试访问**：**Custom Provider Keys** 功能处于 Beta 测试阶段，用户请求早期访问并预见未来可能产生的费用。
   - 一位成员恳求道：*“我也想要 Custom Provider Keys 的 Beta 访问权限！”*，而另一位成员则对团队的努力表示感谢，无论时间表如何。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **分布式训练运行接近完成**：一个**分布式训练运行（distributed training run）**目前正在进行中，预计将在一天多后完成，从一开始就有预先安排的算力合作伙伴参与。
   - 关于训练运行完成的更多细节预计很快公布，社区内也讨论了潜在的**公众参与**可能性。
- **Forge Reasoning API Beta 正式发布**：Nous Research 推出了 **Forge Reasoning API Beta**，旨在缩短各种模型的推理时间，并可能提升 **Hermes 70B** 的能力。
   - 这一进展回应了社区对**大规模基础模型（large-scale foundation models）**及其实际应用的兴趣，正如[官方公告](https://x.com/NousResearch/status/1856417883934601246)中所述。
- **关于在 LLM 中实现实时记忆（Live Memory）的辩论**：成员们讨论了在 **LLM** 架构中实现**实时记忆**的策略，权衡了使用函数调用与 **RAG** 方法在提高一致性和性能方面的优劣。
   - 社区达成共识，倾向于使用**经典方法**来更可靠地锚定神经网络，同时保持风格的一致性。
- **提议将 Linux from Scratch 作为 AI Benchmark**：有人提出了一项咨询，探讨利用《Linux from Scratch》一书作为评估 **AI Agent** 的 **Benchmark** 的可行性。
   - 这表明人们正倾向于建立**具体指标**，以评估 **Agent** 在现实场景中的表现。
- **将 Momentum 整合进 Residual Stream 架构**：一位成员提议将 **Momentum** 的概念引入 **Residual Stream** 架构，并对其数学基础提出了疑问。
   - 这引发了关于**加法和跳跃连接（skip connections）**是否足以实现类似性能增强的讨论。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 与 Spotify 合作推出 AI 播客**：在 [2024 年 12 月 4 日](https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/)，**NotebookLM** 与 **Spotify** 合作推出了 **Spotify Wrapped AI Podcast**，为用户提供年度音乐偏好的个性化音频回顾。
   - 该播客利用 **NotebookLM** 分析用户最喜爱的曲目和艺人，并由 **AI 主持人** 剖析他们音乐年度中的定义性时刻。
- **NotebookLM 中的 AI 音频生成增强**：成员们展示了 **AI 生成的多语言音频**片段，突显了 **NotebookLM** 制作多语言内容的能力，尽管偶尔会出现焦点丢失的情况。
   - 讨论内容包括对**波兰语支持**的查询，表明语言处理设置正在持续改进。
- **利用 NotebookLM 变革体育新闻**：**NotebookLM** 正被用于为职业运动队创建每晚的赛前和赛后专题报道，从而实现规模化的内容生成。
   - 用户强调了生成品牌化化身以及通过自动化叙事增强粉丝参与度的便利性。
- **通过 NotebookLM 简化法律内容**：用户称赞 **NotebookLM** 能够有效解析复杂的法律术语，使各州数据法的信息更加易于获取。
   - 它被视为简化法律文件的日常工具，增强了非专业人士的理解。
- **NotebookLM 中的语言设置挑战**：用户报告了在 **NotebookLM** 中更改语言设置的困难，特别是播客内容，尽管已将 Google 账户调整为印度尼西亚语等语言。
   - 有用户表示，在上传脚本后尝试生成葡萄牙语等语言的音频失败，感到困惑和失望。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Amazon 发布 6 款全新基础模型**：在 [re:Invent](https://link.to.amazonbedrock) 期间，**Amazon** 宣布了 **6 款全新基础模型**，包括 **Nova Micro** 和 **Reel**，支持 **高达 300K tokens** 和 **200 多种语言**。
   - 这些模型仅通过 [Amazon Bedrock](https://link.to.amazonbedrock) 提供，旨在提供文本生成视频（text-to-video）能力，Micro 模型的起售价为 **$0.035**。
- **NVIDIA 的 SANA 许可证面临抵制**：**NVIDIA** 推出了 **SANA 模型**，因其速度受到赞誉，但因许可证限制其仅能用于非商业应用且 **仅限 NVIDIA GPU** 而受到批评。
   - 用户对无法在 AMD 机器上使用以及 **NVIDIA** 保留生成内容权利等限制表示担忧，详见[这条推文](https://x.com/cloneofsimo/status/1864309440356470894)。
- **IFEval 基准测试饱和度受到质疑**：成员们讨论了 **IFEval 基准测试** 的相关性，指出 **90% 的基准测试得分** 现在已很常见，许多模型都取得了高分。
   - 这引发了关于是否需要新的元基准测试（meta benchmarks）来更好地评估 AI 模型性能的讨论。
- **Anduril 与 OpenAI 合作以保持美国 AI 领先地位**：**Anduril Industries** 和 **OpenAI** 建立合作伙伴关系，以推进 **美国人工智能** 的领导地位，将 **Lattice** 系统集成到跨领域的安全保障中。
   - 此次合作重点在于利用创新的 AI 技术支持武装部队任务，详见 [Anduril 的公告](https://www.anduril.com/article/anduril-partners-with-openai-to-advance-u-s-artificial-intelligence-leadership-and-protect-u-s/)。
- **Mistral Large 2 在 Bash 脚本编写方面超越 GPT-4**：**Mistral Large 2** 因在处理 Bash 脚本和查询方面优于 **GPT-4** 和 **3.5 Sonnet** 而受到称赞，如 [Xeophon 的推文](https://x.com/TheXeophon/status/1833921199170355480)所示。
   - 用户幽默地指出，有了 AI 和在线 Bash 解释器，再也不需要死记硬背 **ffmpeg flags** 了。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Gram 矩阵计算效率提升**：一位用户讨论了高效计算 Gram 矩阵上三角（**A@A^T**）的方法，而无需执行标准矩阵乘法后再调用 triplet upper 函数，建议使用 [Triton](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html) 仅计算相关的 tiles，或使用 **cuBLAS 的 syrk** 和 **cutlass** 等替代方案。
   - 共享了如 [Triton 的 matmul 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)等资源以帮助掌握 matmul kernel 优化，尽管有人指出这些材料对初学者可能不太友好。
- **Triton 的 MLIR 文档深度探讨**：讨论集中在 Triton 的 MLIR Dialects 文档可用性上，引用了 [Triton Ops 文档](https://triton-lang.org/main/dialects/TritonOps.html)并指出其[编程指南](https://github.com/triton-lang/triton/tree/main/docs/programming-guide)内容较少。
   - 讨论了在 Triton 中使用 TMA 编写 Grouped GEMM 的挑战，并提到了一个旨在增强该功能的 [pull request](https://github.com/triton-lang/triton/pull/4498)，但完整支持仍不确定。
- **KernelBench 的关键基准测试**：🌽 [KernelBench](https://twitter.com/anneouyang/status/1864014135824162995) (预览版) 作为一种新的编码基准测试推出，旨在评估 LLM 生成用于神经网络优化的**高效** GPU kernel 的能力。
   - 有人担心排行榜上的一些**最快 kernel** 似乎并不完整，用户分享了具体的解决方案如 [incomplete kernel](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py) 以供审查。
- **Tenstorrent AI 融资大幅增长**：一位成员宣布 **Tenstorrent** 本周获得了 **7 亿美元** 融资，这是近期 AI 领域融资热潮中的显著一笔。
   - 该公告包含了一个 [YouTube 视频](https://www.youtube.com/watch?v=_aqMdhAgGG8)链接，其中 Jim Keller 讨论了 AI 对计算领域即将产生的冲击。
- **Thunderkittens 处理竞态条件**：一位用户报告在使用 **TK 的 WGMMA+tma** 实现自定义 kernel 时遇到了**竞态条件（race condition）**，这是由 K 维度的对齐问题引起的。
   - 他们开发了一个创新的**掩码函数（masking function）**，通过将零加载到共享内存（shared memory）来处理越界行，但 **memcheck/synccheck/initcheck** 未报告任何错误，增加了调试难度。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Discord 欺诈机器人攻击社区**：多个 **bots** 正在渗透 Discord 社区，执行诸如**庞氏骗局**或冒充 **Discord support** 的诈骗行为。建议用户[举报这些 bots](https://discord.com/report) 并避免与其互动。
   - 社区成员强调要对这些 **bots** 保持警惕，以维护 Discord 环境的完整性。
- **Stable Diffusion 初学者寻求工具指导**：一位新手对 **Stable Diffusion** 中的工具和模型表示困惑，担心遭遇诈骗。用户推荐使用 **[Vast.ai](https://vast.ai/)** 进行云端 GPU 租赁，并建议观看 YouTube 上 Scott 的 **ComfyUI** 教程以简化工作流。
   - 社区强调了利用 **Vast.ai** 等可靠资源的重要性，以降低在入门过程中遇到诈骗的风险。
- **ComfyUI 助力高级 AI 艺术工作流**：**ComfyUI** 被强调为创建 AI 艺术的最佳平台，对初学者特别有利。用户强调了**观看入门视频**以发挥其最大潜力的重要性。
   - 此外，还强调了本地 AI 运行需要强大 GPU 的必要性，关于云端选项的讨论将其呈现为具有成本效益的替代方案。
- **Stable Diffusion 中的 LoRA 模型故障**：用户报告了 **LoRA models** 的问题，指出在 prompt 中需要特定的触发词才能正常工作。导致图像结果出现混乱的问题归因于各种 **Stable Diffusion** 设置。
   - 社区讨论了优化设置以解决图像生成不一致的问题并增强整体性能。
- **使用性能分析工具增强 SD**：一位用户表示打算为 **Stable Diffusion** 开发性能分析工具，理由是目前此类资源匮乏。这一倡议得到了其他人的赞同，他们认为 **SD ecosystem** 需要增强以改善用户体验。
   - 社区认识到性能工具在提升 **Stable Diffusion** 能力和可用性方面的潜在影响。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Amazon Nova 模型发布**：在 [AWS re:Invent](https://youtu.be/LY7m5LQliAo?t=6657) 上，Amazon 介绍了其 **Nova** 系列基础模型，包括在 **Amazon Bedrock** 上提供的文本和视频生成模型，将其定位为 **GPT-4** 等领先竞品的对手。
   - 社区反馈正在涌现，重点关注 **Nova's performance** 与 **OpenAI** 产品的对比，初步基准测试显示出具有竞争力的结果。
- **AWS 推出全新 Usage API**：AWS 发布了 **Usage API**，允许开发者通过编程方式跟踪使用情况和成本。这包括按时间监控 token 使用情况以及通过各种标识符进行过滤。
   - 新功能旨在提高使用 **AWS services** 的开发者的透明度和管理效率，促进更好的资源分配。
- **PydanticAI 框架发布**：**Pydantic** 推出了 **PydanticAI**，这是一个旨在简化由大语言模型驱动的应用开发的框架，强调**类型安全**和**模块化**。它目前处于 **beta** 阶段，并根据 **MIT License** 开源。
   - 该框架针对寻求将 **LLMs** 整合到项目中的便捷选项的开发者，促进了集成的简便性和可扩展性。
- **OpenAI 的“12 天公告”活动**：**OpenAI** 于 12 月 5 日开始了其 **12 Days of Announcements** 活动，特色是每日发布、演示和更新。早期统计数据包括 **3 亿 ChatGPT 周活跃用户**和平台上每日发送的 **10 亿条消息**。
   - 预计的关键亮点包括可能推出的 **text-to-video AI tool**，这在 AI 工程社区中引起了兴奋。
- **Google 推出 Genie 2**：**Google** 发布了 **Genie 2**，这是一种**自回归潜扩散模型 (autoregressive latent diffusion model)**，专为**视频生成**和**交互式环境**设计。该模型利用 **Transformer dynamics** 框架来增强生成内容中的动作可控性。
   - 社区讨论集中在该模型的**输出长度**及其生成**视频**的实用性上，表明了对其应用的浓厚兴趣。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio Windows 下载故障**：用户报告了下载 Windows x86 版本 [LM Studio](https://lmstudio.ai) 时遇到的问题，提示文件不可用。
   - 其他人建议可能是 CDN 问题，并建议使用 VPN 重新尝试下载。
- **LM Studio 在 Windows 与 Mac 上的性能差异**：一位成员在 Windows 上运行 **LM Studio** 时遇到了明显的性能问题（相比 Mac），包括模型输出异常字符。
   - 排查建议包括切换 `Flash Attention` 开关并核实系统规格。
- **利用 LLM 作为 RPG 游戏主持人**：一位用户分享了使用 **LLM** 进行预设 RPG 冒险的经验，强调了用泰语编写大纲以防止预知的创新做法。
   - 该实验取得了引人入胜的结果，引发了对 AI 驱动的 RPG 玩法的方法论和社区资源的讨论兴趣。
- **利用局域网 GPU 优化 LM Studio**：一位用户询问如何将笔记本电脑上的 **LM Studio** 连接到拥有多块 GPU 的本地服务器以提升性能。
   - 另一位成员确认了可行性，并指出需要一个前端来确保功能正常。
- **对 Intel Arc Battlemage GPU 的质疑**：用户对新款 **Arc Battlemage** 显卡表示担忧，质疑 **Intel GPU** 在 AI 任务中的可靠性，原因是驱动支持不足。
   - *一条评论强调，使用较少但显存较大的 GPU（如 3090）更为理想。*

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **在 Vercel 上构建 AI 应用变得更加简单**：[LlamaIndex 的最新更新](https://twitter.com/llama_index/status/1864002184138170677)简化了在 Vercel 上的 AI 应用开发，增强了与 LlamaCloud 的集成能力。
   - 这一进展可能会提高开发者的生产力并简化 AI 应用的部署流程。
- **Amazon 发布具有竞争力的 Nova 模型**：Amazon 的新基础模型系列 **Nova** 拥有极具竞争力的基准测试结果和更具吸引力的定价；可以通过 `pip install llama-index-llms-bedrock-converse` 安装以支持 [链接在此](https://twitter.com/llama_index/status/1864080917029085459)。
   - 这些基础模型旨在为用户在 AI 模型领域提供高性价比且性能驱动的选择。
- **使用 LlamaIndex Workflows 快速实现 RAG**：学习使用 LlamaIndex Workflows 构建高性能的检索增强生成（RAG）系统，其特点是事件驱动架构 [详情在此](https://twitter.com/llama_index/status/1864377849295327365)。
   - 该指南将此方法与 LangGraph 等其他框架进行了比较，强调了在复杂 AI 场景中的效率。
- **Summary Index 性能问题**：一位用户提出了使用 **sentencesplitter** 的 **summaryindex** 响应缓慢的问题，称生成摘要大约需要 **2 分钟**，而 ChatGPT 仅需 **8 秒**。
   - 他们探讨了潜在的改进方案，但也承认使用路由和索引方法会引入延迟。
- **为 LLM 优化 Prompt**：一位在使用 OpenAI LLM 时遇到幻觉问题的用户被建议尝试 **prompt 优化**，以提高响应准确性。
   - 建议通过编写更好的指令来提升语言模型的性能。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank 3.5 的多语言提升**：Cohere 推出了 **Rerank 3.5**，支持超过 **100 多种语言**的**多语言**和**英语**排名，增强了搜索能力，详情见我们的 [博客文章](https://cohere.com/blog/rerank-3pt5)。
   - 一位用户报告使用 'rerank-multilingual-v3.0' 时出现 **30% 的性能下降**，并对新 **Rerank 3.5** 模型的有效性表示担忧，促使 Cohere 的**支持团队**协助进行故障排除。
- **Cohere Toolkit 错误修复**：用户在运行 **cohere-toolkit** 时遇到警告，特别是与 **alembic** 相关的错误以及与 **PyTorch 2.5.1** 的兼容性问题。
   - 社区成员正在寻求解决方案，并建议咨询 Cohere 的**支持团队**以解决这些问题。
- **Harmony 的 LLM 匹配竞赛**：**Harmony 项目**正在 [DOXA AI](https://harmonydata.ac.uk/doxa/) 上举办一场竞赛，旨在增强其 **LLM 匹配算法**，为参赛者提供高达 **£500** 的代金券奖励。
   - 参与者可以通过 Harmony Discord 服务器的 🏅「matching-challenge」频道加入，无需具备 LLM 经验。
- **模型弃用指南**：Cohere 更新了其**模型弃用**政策，概述了模型的生命周期阶段，包括 **Active**（活跃）、**Legacy**（遗留）和 **Deprecated**（已弃用），详见 [Deprecations — Cohere](https://docs.cohere.com/docs/deprecations) 文档。
   - 鼓励开发者查阅文档，以为任何已弃用的 Endpoint 和模型寻找推荐的替代方案。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic AI 提升 DSLModel 能力**：将 [Pydantic AI](https://ai.pydantic.dev/) 与 **DSLModel** 集成，引入了一个 Agent 框架，通过 Pydantic 的强大功能增强了 LLM 的可用性。
   - 一位成员强调了 **Pydantic** 在与 **FastAPI** 等框架结合时如何简化 AI 项目的开发。
- **精通 AI 开发直播演示预告**：一场题为 [Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive](https://youtube.com/live/mBQFKo8bPBI) 的直播演示即将举行，旨在探索先进的 AI 开发技术。
   - 该活动旨在展示在 AI 项目中利用 **PydanticAI** 及相关工具的创新方法。
- **DSPy 优化触及 AWS Lambda 时间限制**：成员们讨论了在 **AWS Lambda** 上执行 **DSPy 优化**的挑战，特别是针对长时间任务强制执行的 **15 分钟执行限制**。
   - 提议的解决方案包括使用 **/tmp 文件夹**进行缓存，以解决 Lambda 的只读文件系统问题并提高处理速度。
- **ProgramOfThought 将在 v2.6 中进行翻新**：**ProgramOfThought** 计划在 **v2.6** 中进行翻新，以解决 **v2.5** 之后关于其支持状态的担忧。
   - 建议用户谨慎使用当前版本，因为预计年内将进行升级。
- **在类别不平衡情况下开发精确率指标**：一位成员询问如何在具有显著类别不平衡的**多类别分类**问题中，为特定类别开发**精确率指标 (precision metric)**。
   - 推荐使用 **dspy.Example(batch=[...])** 来处理评估，尽管由于**类别不平衡**，挑战依然存在。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Sierra AI 信息说明会**：举办了一场独家的 [Sierra AI 信息说明会](https://www.youtube.com/watch?v=-iWdjbkVgGQ)，展示了他们的对话式 AI 平台，并邀请优秀的开发者参与。
   - Sierra AI 渴望在黑客松之前与开发者建立联系，并强调了即将到来的 **12 月 17 日** 提交截止日期。
- **黑客松提交流程变更**：**LLM Agents MOOC 黑客松** 的提交流程已从 **Devpost 转移到 Google Forms**，[提交表单](https://forms.gle/jNr8nSH9Cy9qpYcu5) 现已上线。
   - 鼓励参与者参考 [提交要求指南](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?usp=sharing) 为 **12 月 17 日** 的截止日期准备项目。
- **证书申报与完成等级**：**证书申报表单** 现已在 [此处](https://forms.gle/nYGJLPTdb7af2Dq59) 发布，概述了五个课程完成等级：Trailblazer、Mastery、Ninja、Legendary 和 Honorary。
   - 参与者必须在 **2024 年 12 月 12 日** 之前完成所有课程作业，包括 **12 个测验** 和一篇书面文章，才有资格获得所选等级。
- **GPT-4 数据泄露担忧**：人们对 **GPT-4** 潜在的数据泄露表示担忧，特别是它是否影响消费者或企业版本，以及用户数据共享默认设置的影响。
   - 可能的 **GPT-4 jailbreak** 可能会暴露训练集中的真实 PII（个人身份信息），引发了与历史性的 **AOL 案例** 的对比。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **解决 Anthropic 分支的 TypeError**：用户在 Open Interpreter 最新的 **Anthropic 开发分支** 中遇到了与意外的 'proxies' 参数相关的 **TypeError**。[讨论线程](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464) 建议将检查自定义 API base 作为首要排查步骤。
   - 另一位成员建议验证客户端初始化设置，指出 'proxies' 参数可能是导致该问题的唯一变更。
- **Open Interpreter 为了性能重写安装方式**：**Open Interpreter** 已完全重写以增强性能。鼓励用户使用 `pip install --force-reinstall git+https://github.com/OpenInterpreter/open-interpreter.git@development` 重新安装最新的开发版本。
   - 开发者强调了用户反馈对于识别缺失功能并确保新实现优于旧版本的重要性。
- **增强的 Linux 兼容性确认**：经用户确认，**Open Interpreter** 在 **Garuda-Linux**（Arch-Linux 的一个分支）上运行顺畅。[完整的兼容性详情](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464) 还强调了在 **Manjaro** 和 **OpenSuse** 发行版上的成功测试。
   - 在多个 Linux 版本上的广泛测试突显了该软件在不同环境中的适应性和可靠性。
- **LiveKit 驱动远程设备连接**：**O1** 利用 **LiveKit** 将 **iPhone** 等设备与笔记本电脑或 **Raspberry Pi** 连接以处理请求。这种设置通过本地 **OpenInterpreter** 实例促进了高效的远程访问。
   - 该集成允许用户远程控制他们的机器，利用 **LiveKit** 的功能增强设备间的互操作性。
- **OpenInterpreter 的 CLI 保持强大的功能**：尽管是以 **CLI 形式** 存在，**OpenInterpreter** 仍提供了有效的计算机操作能力。用户可以使用 `interpreter -y` 命令绕过审批要求，实现无缝的代码执行。
   - 此功能通过在执行代码前要求审批来确保用户安全，同时仍为高级操作提供灵活性。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Genie 2 占据中心位置**：有请求建议在未来一天内将 **Genie 2**（一个大规模基础世界模型）的信息添加到 torchtune 中。更多详情见 [官方博客](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/)。
   - 致谢部分强调了 **Jack Parker-Holder** 和 **Stephen Spencer** 等关键人物的贡献，突出了项目开发中的协作努力。
- **Federated Learning 显示出前景**：正如一份分享的 [论文](https://arxiv.org/pdf/2411.19870) 中所讨论的，底层的 **Federated Learning** 方法可能比完全同步的方法产生更好的结果。
   - *训练仅剩 22 小时* 表示即将完成。
- **Generalist Agents 团队取得进展**：由 Vlad Mnih 领导的 **Generalist Agents** 团队在 **Harris Chan** 和 **Maxime Gazeau** 等成员的贡献下取得了重大进展，展示了 Agent 开发的综合方法。
   - 来自 **SIMA** 团队（包括 **Frederic Besse** 和 **Tim Harley**）的进一步支持，强调了该计划中多样化的专业知识。
- **社区主导的 GPU 贡献潜力**：类似于 **Folding@home** 的社区主导工作具有有趣的潜力，个人可以贡献 GPU 时间。
   - 随着模型规模超过单个数据中心，这可能变得至关重要。
- **MMLU Pro 设定验证标准**：为了验证所讨论框架中的一个区块，模型需要在 **MMLU Pro** 上达到 **90%**。
   - 这突显了成功部署所需的严格性能标准。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Mechanistic Interpretability 增强细胞分析**：研究人员引入了 **Mechanistic Interpretability**，这是一种探索细胞如何对其环境建模的工具，将重点从基因转向 **gene regulatory modules** 和 **sub-cellular locations**。
   - 这种方法可能允许构建一种“细胞行为的通俗心理学”，提供对 **inner life of cells** 的见解。
- **Diffusion Model 的非商业许可限制了采用**：一位成员强调，**Diffusion Model** 的**非商业许可**应该会阻止广泛实施它的尝试。
   - 这种限制可能会影响开发者对该模型的采用和实验。
- **EDM2 框架应用于文本条件 Diffusion Models**：一位成员询问关于利用 **EDM2** 框架训练具有文本条件的 **Diffusion Models**。
   - 他们引用了一篇展示了**令人印象深刻的结果**的 [论文](https://arxiv.org/pdf/2312.02696)，强调了在具体实现方面的空白。
- **Class Conditioning 限制了 Diffusion Model 的灵活性**：论文讨论了 **Class Conditioning**，它将模型限制在为少数预定义类别生成输出。
   - 这种受限的方法与文本条件所需的灵活性形成对比，后者允许在生成中发挥更广泛的创造力。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Meta 的 SAM 凭借用户友好的 Demo 令人惊艳**：一名成员展示了 **Meta 的 SAM** 在其 [Demo 网站](https://segment-anything.com/demo)上的表现，重点介绍了其在云端运行的 **600M 图像嵌入 Transformer** 以及直接在浏览器中运行的小型模型。
   - 该 Demo 强调了 SAM 模型开箱即用的**有效性**，并为未来的 **tinygrad** 模型和社区吸引力设定了**质量基准**。
- **Web 模型随 ONNX 集成而激增**：讨论强调了如**云端 ONNX** 等 **Web 模型**的发展，增强了机器学习工具的**可访问性**。
   - 这些模型提供了既能在云端运行又能直接在浏览器中运行的功能，展示了增加**用户参与度**的潜力。
- **在 tinygrad 中调整 Threadgroup/Grid 大小**：一位用户询问在 `uopgraph.py` 的图重写优化期间如何更改 **threadgroup/grid 大小**，George Hotz 回复称可以在 `kernel.py` 的 **OptOps** 中进行修改。
   - 这种灵活性允许在 **tinygrad** 架构中采用定制的优化策略。
- **分享 BEAM Search 见解**：一位用户发布了关于 [BEAM Search](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md) 的内容，解释了 **tinygrad** 中的 **beam search** 和**内核优化选项**。
   - 该资源是理解这些概念及其在 **tinygrad** 开发中应用的重要指南。
- **JIT 函数覆盖输出**：关于 **JIT 函数** 的一条注释指出，在第一次调用后，jitted 函数会**重用相同的输出缓冲区**，这可能会覆盖之前的结果。
   - 为了保留结果，有必要在每次调用后使用 `.clone().realize()`。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **ADOPT 优化器集成至 Axolotl**：**ADOPT 优化器**已集成到 Axolotl 代码库中以增强**训练稳定性**，详见 [Pull Request #2104](https://github.com/axolotl-ai-cloud/axolotl/pull/2104)。
   - 此次更新确保了与当前 **torch 版本**的兼容性，并合并了原作者在[此处](https://github.com/iShohei220/adopt)的最新修改。
- **ADOPT 优化器实现最优收敛**：成员们讨论了 **ADOPT 优化器**在任何 beta 值下实现**最优收敛**的能力。
   - 这种灵活性被认为是一项核心优势，适用于多种训练场景。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Unternet 招聘开源工程师**：[Unternet 正在招聘一名**开源工程师**](https://discord.com/channels/1089876418936180786/1313839138562248737)，负责贡献开源项目、编写技术文档并与社区互动。
   - 该职位强调了与社区协作以及开发技术文档的重要性，面向对开源贡献充满热情的人士。
- **社区参与机会**：该职位强调了在开发技术文档的同时与社区协作的重要性。
   - 此角色旨在吸引对开源贡献有热情的个人。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 模型启动失败**：一位用户在尝试启动其 **Gorilla 模型**时遇到错误，提示与 **tokenizer** 相关的依赖问题。
   - 错误信息显示缺少 **protobuf 库**，尽管该库已安装在他们的环境中。
- **Protobuf 库未被识别**：用户确认已安装版本为 **5.29.0** 的 **protobuf** 包，但系统仍报告缺失。
   - 这引发了关于导致环境无法识别已安装包的原因的疑问。

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **成员跟进工单消息**：一名成员促请 **Nick** 查看他们发送的关于 **ticket**（工单）的消息，请求他在有空时查看。
   - 他们强调了及时响应的重要性，暗示需要快速解决。
- **工单对话缺乏额外上下文**：关于 **ticket** 的对话除了跟进之外，没有提供任何进一步的上下文。
   - 没有讨论额外的评论或链接。



---


**MLOps @Chipro Discord** 没有新消息。如果这个服务器沉寂太久，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果这个服务器沉寂太久，请告知我们，我们将将其移除。


---

# PART 2: 频道详细摘要与链接


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1313490637697323070)** (476 条消息🔥🔥🔥): 

> `Cursor 停机, Cursor 功能变更, Windsurf vs. Cursor 性能, OpenAI 12 天发布会, Cursor 性能问题` 


- **Cursor 遭遇停机**：许多用户报告了 Cursor 宕机的问题，经历了严重的延迟，且无法生成响应。
   - 用户对状态更新和响应质量的缺乏表示不满，一些用户回退到 ChatGPT 或切换到 Windsurf。
- **Cursor 功能变更引发担忧**：移除 long context mode 以及 Cursor 中新的界面更改导致了用户的不满。
   - 许多用户注意到模型响应的有效性下降，暗示可能存在模型质量降级或性能问题。
- **Windsurf 成为可靠的替代方案**：一些用户报告了使用 Windsurf 的积极体验，声称它能更好地处理编码任务而不会过度修改代码。
   - 这引发了关于 Cursor 最近的更新是否是对 Windsurf 功能和热度的直接回应的讨论。
- **OpenAI 宣布为期 12 天的更新**：OpenAI 将从明天开始连续 12 天每天发布新公告，这引起了用户的兴奋。
   - 用户希望这些公告能带来现有工具的改进，可能解决 Cursor 最近面临的挑战。
- **Cursor 性能问题持续存在**：许多开发者注意到 Cursor 最近的更新不仅减慢了响应速度，还导致代码编辑中的错误增加。
   - 用户正在质疑这些更改的有效性，并寻求潜在的解决方案或临时对策。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com">Medium: Read and write stories.</a>: 在 Medium 上，任何人都可以向世界分享深刻的见解、有用的知识和人生智慧。</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">How to do `Fix in Composer` and `Fix in Chat` actions from keyboard</a>: 这两个操作：我在设置中找不到。</li><li><a href="https://forum.cursor.com/t/long-context-mode-gone-in-newest-update/29449/34">Long context mode gone in newest update</a>: 感谢分享你的想法，并为之前的沉默感到抱歉！我想解释一下我们弃用 0.43 功能背后的原因。我们喜欢发布早期实验版本以获取反馈...</li><li><a href="https://forum.cursor.com/t/feature-request-long-context-mode-upvote/32187">Feature request: Long context mode (upvote!)</a>: 能够利用完整的 LLM 上下文将非常有帮助。请恢复这个功能 🙂</li><li><a href="https://medium.com/@NFAblog/connect-github-codespaces-to-cursor-ai-ai-friendly-vs-code-clone-243fa5f79414">Connect Github CodeSpaces to Cursor Ai (Ai friendly vs code clone)</a>: 将 GitHub Codespaces 连接到 CURSOR.DEV：开发者指南</li><li><a href="https://status.cursor.com/">Cursor Status</a>: 未找到描述</li><li><a href="https://github.com/TheGalaxyStars/KEPLER-COMMUNITY">GitHub - TheGalaxyStars/KEPLER-COMMUNITY</a>: 通过在 GitHub 上创建账户来为 TheGalaxyStars/KEPLER-COMMUNITY 的开发做出贡献。</li><li><a href="https://cursor.directory/">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor 规则</li><li><a href="https://github.com/getcursor/cursor/issues/2027">WSL extension is supported only in Microsoft versions of VS Code · Issue #2027 · getcursor/cursor</a>: 如果可以，请附上问题的截图。请包含你的操作系统名称。如果可以，重现步骤会非常有帮助。我正在使用 Windows 11 + WSL: Ubu...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1313603284367577128)** (198 messages🔥🔥): 

> `JAX vs PyTorch 性能，Apple 对 AWS AI 芯片的使用，训练方法与框架，Schedule-free 优化器，带坐标的图像 Embedding 技术` 


- **JAX 在大型实验室中的采用情况**：关于领先的 AI 实验室是否主要使用 JAX 而非 PyTorch 存在争议，对其性能优势和行业使用情况意见不一。
   - 一些成员认为，虽然 JAX 在 TPU 上备受青睐，但许多组织在 GPU 任务上仍然严重依赖 PyTorch。
- **Apple 与 AWS 的关系**：Apple 在一次 AWS 活动中确认，他们使用了 AWS 定制 AI 芯片，并表示双方在 AI 研究方面拥有稳固的合作伙伴关系。
   - 讨论指出，尽管 Apple 使用了 AWS 硬件，但对于大规模机器学习任务，他们仍然更倾向于选择 GPU。
- **训练框架的演进**：讨论了机器学习训练中不同优化器的使用，特别是像 muon 这样的 schedule-free 优化器是否比 AdamW 更受欢迎。
   - 虽然 schedule-free 优化器被认为是小众的，但 AdamW 在实践中似乎继续被广泛采用。
- **优化图像 Embedding**：一位用户正在探索将 2D 坐标合并到图像 Embedding 中的方法，讨论是采用通道拼接还是应用其他技术。
   - 讨论涉及了 rotary embeddings 以及 StyleGAN 等示例，强调了提高模型效率的各种方法。
- **机器学习研究的新进展**：提到 KellerJordan 的一个 GitHub 仓库揭示了 muon 优化器的使用，引发了对其与现有方法相比能力的关注。
   - 引用了一篇关于 nanogpt 的早期学术论文，表明在新型优化器及其评估方面存在竞争态势。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.macrumors.com/2024/12/03/apple-amazon-ai-chips-search/">Apple 在搜索服务中使用 Amazon 定制 AI 芯片</a>: Apple 在搜索服务、Apple 机器学习和 AI 中使用了来自 Amazon Web Services 的定制 Inferentia 和 Graviton 人工智能芯片...</li><li><a href="https://news.ycombinator.com/item?id=39876444">几乎每个大型生成式 AI 玩家（Anthropic, Cohere, DeepMind, Mid...）都在使用 JAX | Hacker News</a>: 未找到描述</li><li><a href="https://github.com/stanford-cs149/asst4-trainium">GitHub - stanford-cs149/asst4-trainium</a>: 通过创建账户为 stanford-cs149/asst4-trainium 的开发做出贡献。</li><li><a href="https://github.com/apple/axlearn">GitHub - apple/axlearn: 一个可扩展的深度学习库</a>: 一个可扩展的深度学习库。通过创建账户为 apple/axlearn 的开发做出贡献。</li><li><a href="https://github.com/KellerJordan/modded-nanogpt">GitHub - KellerJordan/modded-nanogpt: 5 分钟实现 NanoGPT (124M)</a>: 5 分钟实现 NanoGPT (124M)。通过创建账户为 KellerJordan/modded-nanogpt 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1313556962859941970)** (114 messages🔥🔥): 

> `Gradient Synchronization in Large Models, Performance of Second Order Optimizers, Random Number Generators, Flow Matching vs Diffusion Training, Machine Unlearning Literature` 


- **梯度同步在大模型中并非主要关注点**：有人指出，一旦模型超过 4000 亿（400 billion）参数大关，梯度同步（Gradient Sync）的重要性就会降低，因为大部分同步负载并不单纯与梯度同步相关。
   - 将优化器状态（optimizer state）减少 **4 字节 (4 bytes)** 被强调为一项有意义的改进，特别是在分布式训练工作中。
- **关于二阶优化器（Second Order Optimizer）有效性的争论**：一些成员对二阶优化器在非凸优化（non-convex optimization）中的益处表示怀疑，理由是尽管有一些关于收敛性改善的报告，但实证研究结果褒贬不一。
   - 其他人建议，二阶优化器在处理极小特征值（tiny eigenvalues）时会更有效，但从经验上看，预计性能不会有显著差异。
- **生成随机数生成器 (RNGs)**：讨论了生成 RNG 算法的可行性，并建议避免重新发明已有的算法，因为确保随机性质量涉及复杂的因素。
   - 有人指出，与尝试从头开始创建新的 RNG 相比，现有的 RNG（如 **Threefry** 和 **Philox**）对并行计算友好且非常有效。
- **Flow Matching 与 Diffusion 训练的对比**：Flow Matching 因其更简单的公式和更直的采样轨迹（straighter sampling trajectories）而受到关注，这引发了关于其相对于 Diffusion 模型的优势的讨论。
   - 尽管公式有所不同，但研究表明，当应用于高斯分布（Gaussian distributions）时，Flow Matching 与 Diffusion 模型是等价的，这使得整合这两种方法的技巧成为可能。
- **机器去学习 (Machine Unlearning) 的挑战**：目前对于衡量微调（fine-tuning）能在多大程度上将模型性能恢复到预训练状态缺乏信心，因为大多数研究使用代理指标来评估性能一致性。
   - 成员们建议探索 Machine Unlearning 文献以获取见解，同时也承认目前的方法可能无法可靠地量化去学习后的模型行为。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/wortsman21a.html">Learning Neural Network Subspaces</a>：最近的观察推进了我们对神经网络优化景观（optimization landscape）的理解，揭示了以下内容的存在：(1) 包含多样化解的高精度路径，以及 (2) 更宽的 mi...</li><li><a href="https://diffusionflow.github.io/">Diffusion Meets Flow Matching</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2006.08381">DreamCoder: Growing generalizable, interpretable knowledge with wake-sleep Bayesian program learning</a>：专家级的问题解决是由用于思考问题及其解决方案的强大语言驱动的。获得专业知识意味着学习这些语言——即概念系统，以及相关的技能...</li><li><a href="https://arxiv.org/abs/2401.14953">Learning Universal Predictors</a>：Meta-learning 已成为一种强大的方法，用于训练神经网络从有限的数据中快速学习新任务。广泛接触不同的任务可以产生通用的表示，从而实现 ge...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1313918808397840394)** (1 messages): 

> `Scaling Law Codebases, Examples of Scaling Experiments` 


- **咨询 Scaling Law 资源**：一位成员开始尝试 **scaling law**，并请求推荐用于 scaling 实验的优秀 **代码库 (codebase)** 或示例代码。
   - *非常感谢*为寻找资源提供的任何帮助！
- **请求 Scaling 实验案例**：另一位用户表示有兴趣寻找各种 **Scaling 实验案例**，以更好地理解这一概念。
   - 他们寻求社区的指导，以指出有帮助的文档或仓库。


  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/)** (1 messages): 

deku7041: https://transformer-circuits.pub/
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1313499025277976607)** (7 条消息): 

> `External Loadable Evals, lm-eval-harness, Dataset Visibility and Versioning, Reproducibility Concerns` 


- **提议外部可加载评估 (External Loadable Evals)**：有想法提出让评估任务像数据集一样通过 [Hugging Face](https://huggingface.co) 进行外部加载，允许用户在不修改 *lm-eval-harness* 的情况下加载数据集和评估配置。
   - *Jonabur* 强调了定义评估“格式”以实现更好集成的潜力。
- **现有的外部加载能力**：目前在某种程度上可以通过 [include_path](https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197) 加载外部评估，该参数允许传递包含配置的目录。
   - *Baber_* 分享了对这一现有功能的优势见解。
- **复现性与外部评估的权衡**：*Baber_* 对使用外部仓库进行评估时的可见性和版本控制表示担忧，强调了对复现性（reproducibility）的挑战。
   - *Jonabur* 同意复现性在评估过程中的重要性。
- **数据集版本控制挑战**：讨论转向了版本控制和复现性是否也会对现有评估中使用的原始数据集产生问题。
   - *Baber_* 承认了这一担忧，但指出目前这还没有成为一个显著的问题。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197),">lm-evaluation-harness/lm_eval/__main__.py at f49b0377bf559f5558e8cd9ebd1190218c7df2a4 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1313790501127979071)** (1 条消息): 

> `Mira Virtual AI tools, Multimodal conversions, Consumer-level GPU frameworks` 


- **介绍 Mira Virtual AI 工具**：一位成员展示了他们的 [GitHub 项目](https://github.com/Mirror-Prismals/Mira-Virtual-Ai) **Mira Virtual Ai**，该项目提供了一系列用于多模态转换和其他基础任务的实用工具，旨在消费级硬件上运行。
   - 这些脚本仅需 **2GB VRAM** 即可运行，是自包含的，旨在为编程经验有限的用户提供可访问的 AI 解决方案。
- **专注于易用性和可访问性**：该成员强调，他们的工具是为可能不具备编程技能的用户量身定制的，使在本地进行 AI 实验变得更加容易。
   - 他们表示希望为更广泛的受众带来多模态任务中的**趣味性和自动化**。



**提及的链接**：<a href="https://github.com/Mirror-Prismals/Mira-Virtual-Ai">GitHub - Mirror-Prismals/Mira-Virtual-Ai: Ai Frameworks for Consumer Level GPU&#39;s</a>：适用于消费级 GPU 的 AI 框架。通过在 GitHub 上创建账号为 Mirror-Prismals/Mira-Virtual-Ai 做出贡献。

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1313495582525882428)** (2 条消息): 

> `Logging Configuration, Optimizer Performance Metrics` 


- **理解日志输出**：一位成员寻求关于详述**优化器操作和耗时指标**的特定日志消息来源的澄清。
   - 他们注意到消息中包含了各种优化器步骤（包括 **fwd** 和 **bwd** 操作）的详细耗时信息。
- **配置选项解析**：该成员发现 **'wall_clock_breakdown'** 配置选项启用了他们所询问的详细日志记录。
   - 此配置选项提供了训练期间不同操作耗时细分的见解。


  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/)** (1 条消息): 

OpenAI: -# @everyone 12 Days of OpenAI
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1313502276760899614)** (242 条消息🔥🔥): 

> `AI 翻译工具, 投票中的量子计算, Cohere AI 功能, OpenAI 文件处理问题, 匈牙利语翻译准确性` 


- **关于 AI 翻译工具的讨论**：成员们交流了对各种 AI 翻译工具的看法，**DeepL** 因其准确性比 **Google Translate** 和 **Microsoft** 更受青睐。
   - 建议使用 **Cohere's API** 或 open-webui 过滤器来实现聊天机器人的多语言能力。
- **OpenAI 文件处理的挑战**：一位用户报告了 **ChatGPT 4o** 无法处理文件和图像的问题，引发了关于这是否是一个普遍 bug 的讨论。
   - 建议确保使用正确的模型，并考虑就该问题提交 bug 报告。
- **量子计算见解**：讨论涉及了 **quantum computing** 在各个领域的应用，包括通过高级算法在投票系统中带来的潜在益处。
   - 关于量子算法在实际投票场景中的相关性存在分歧，强调 *选民并不处于叠加态 (superposition)*。
- **Cohere AI 的匈牙利语翻译功能**：**Cohere AI** 平台因拥有支持超过 100 种语言（包括 **Hungarian**）的模型而受到关注，用户分享了他们的使用体验。
   - 值得注意的是，尽管是一个大模型，但对匈牙利语翻译的高准确性使其成为需要多语言支持的用户的强力选择。
- **OpenAI 未来发展与想法**：对话还包括对 OpenAI 发展方向的反思，一些成员建议改进推理模型以增强推理能力。
   - 探讨了 AI 驱动工具的潜力，包括集成本地模型和多语言支持，以促进学习和可访问性。



**提到的链接**：<a href="https://www.youtube.com/watch?v=nUa_r9GKjtI">Mark Johns (@Doomlaser) 谈人工智能、符号逻辑、企业公平等 ∰$❤️🏤。</a>：在 Twitter 上关注 https://x.com/DoomlaserCORP。

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1313649255856209930)** (4 条消息): 

> `GPT 图像读取限制, LLM 与翻译问题, Custom GPTs 的高级语音模式` 


- **GPT 不再读取图像**：一位成员注意到 **GPT** 无法再读取图像，引发了关于这一变化影响的疑问。
   - 这一限制凸显了成员们渴望了解的能力转变。
- **LLM 难以处理不可翻译字符串**：一位成员幽默地指出，**LLM 经常无法**识别标记为 `i18n` 的代码中不可翻译的字符串，展示了它们的逻辑局限性。
   - *这为 LLM 在代码解释方面面临的挑战提供了有趣的见解*。
- **关于 Custom GPTs 中高级语音模式的咨询**：一位成员询问是否有计划在 Custom GPTs 中实现 **Advanced Voice Mode**。
   - 这一咨询反映了用户对增强 Custom GPT 功能以更好满足需求的持续关注。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1313608463850536981)** (11 messages🔥): 

> `改进 Prompt Engineering，诱导模型进行更深层次的思考，在 Prompt 中使用 Markdown，关于 GPT 响应时间的研究，模型对比` 


- **频道作为 Prompt Engineering 的资源**：一位成员建议，该频道本身就是改进自定义 GPTs 的 Prompt Engineering 的绝佳起点，并推荐通过提问和私信获取更多选项。
   - 他们强调了 Prompt 中上下文（Context）和注意力（Attention）的重要性。
- **寻求深化模型思考的策略**：一位成员询问了如何“诱导（Bait）”像 o1 这样的模型在回答前进行更彻底的思考，并引用了 OpenAI 的研究，该研究表明更长的思考时间会带来更好的答案。
   - 另一位成员警告不要使用“诱导”一词，因为这可能会改变模型对 Prompt 的理解。
- **Prompt 的 Markdown 结构**：成员们讨论了使用 Markdown 以层级结构呈现 Prompt，这可能有助于引入复杂的考量因素并增强 Prompt 的清晰度。
   - 一位参与者提到，提出难题可以作为促使模型进行更深层次思考的一种方式。
- **OpenAI 模型测试的局限性**：有成员对使用 OpenAI 模型测试各种 Prompting 策略的局限性表示担忧，指出理解“正常”响应是主观的。
   - 缺乏无限次 Prompting 的权限限制了在优化 Prompt 以获得更好响应方面的实验。
- **质疑 Prompt 的有效性**：分享的一种可能策略包括提示模型进行深度反思，并建议在回答前考虑多种途径。
   - 然而，一位成员指出，确定此类 Prompt 的实际用途本身就是另一个挑战。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1313608463850536981)** (11 messages🔥): 

> `Prompt Engineering, 诱导深度响应, YAML Prompt 结构化, 模型思考时间, API 自动化测试用例` 


- **探索 Prompt Engineering 策略**：成员们讨论了如何改进 Prompt Engineering 以利用 OpenAI ChatGPT 构建更好的自定义 GPTs，并强调该频道是咨询问题的绝佳起点。
   - 有人建议使用像 YAML 这样的 Markdown 语言来分层构建 Prompt，强调了上下文和注意力的重要性。
- **诱导 GPT 进行更深层次的思考**：一位用户询问了诱导 OpenAI 模型 (o1) 在回答前思考更长时间的方法，并参考了 Deepseek 等其他 AI 模型的类似功能。
   - 另一位成员提醒说，“诱导”会改变模型对 Prompt 的理解，并建议改为提出难题。
- **使用安慰剂 Prompt 进行反思**：一位成员提议使用诸如“深入反思并考虑多种可能的途径”之类的 Prompt 来鼓励 AI 进行更深层次的思考。
   - 然而，他们承认这类方法的有效性难以进行彻底评估。
- **模型行为的不确定性**：大家对 OpenAI 模型如何处理 Prompt 存在不确定性达成共识，特别是与未定义的“正常”行为相关时。
   - 成员们表示，如果有无限的模型访问权限，希望能进行更广泛的测试。
- **API 自动化作为测试场**：讨论涉及了使用 API 自动化来高效测试各种 Prompt 策略的想法。
   - 这被认为是评估 Prompt 技术细微差别及其在模型中产生结果的一个很好的案例。


  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1313501394023616572)** (175 messages🔥🔥): 

> `Amazon Bedrock 模型, Aider 新功能, QwQ 模型性能, Aider 用户体验, 基准测试结果`

- **Amazon Bedrock Nova 模型发布**：Amazon 发布了多个新的基础模型，包括 **Nova** 系列，仅通过 Amazon Bedrock 提供，支持高达 **300K tokens** 的上下文长度。
   - 在基准测试中的表现与 **Llama 3** 相当，其定价旨在针对不同的模型能力保持竞争力。
- **Aider 的新 watch-files 功能**：Aider 中新引入的 `--watch-files` 功能允许用户通过 AI 注释与代码无缝交互，根据指定的标记触发操作。
   - 文档仍在完善中，但早期反馈称赞该功能是一项重大进步。
- **QwQ 模型性能不及预期**：据报告，**QwQ 32B Preview** 模型在全量编辑格式（whole edit formats）中获得了 **54%** 的分数，在 diffs 中获得了 **50%** 的分数，表明其性能弱于预期。
   - 建议用户考虑使用 **Qwen** 或 **Sonnet** 模型以获得更好的结果，这反映了对 QwQ 实际效用的担忧。
- **用户体验与反馈**：有一些关于个人使用 Aider 体验的讨论，包括对用户交互和平台熟悉度的挫败感。
   - 值得注意的是，一位用户表示希望使用 GUI 而非 CLI，这种偏好反映了社区中的情绪。
- **开发与改进讨论**：关于如何在 Aider 中实现各种功能和改进的对话正在进行中，包括对新 Nova 模型的更好支持。
   - 合作者分享了关于基准测试结果的见解，以及与添加新模型支持相关的潜在架构变更。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://aws.amazon.com/blogs/aws/reduce-costs-and-latency-with-amazon-bedrock-intelligent-prompt-routing-and-prompt-caching-preview/">通过 Amazon Bedrock 智能提示词路由和提示词缓存（预览版）降低成本和延迟 | Amazon Web Services</a>：路由请求并缓存提示词中常用的上下文，以降低延迟并在性能与成本效率之间取得平衡。</li><li><a href="https://aider.chat/docs/usage/browser.html">浏览器中的 Aider</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://x.com/_philschmid/status/1864016010464080260">Philipp Schmid (@_philschmid) 的推文</a>：出乎意料。@amazon 带着 Foundation Models 回归了。作为 re:Invent 的一部分，他们发布了 6 个全新的基础模型，涵盖从纯文本到文本生成视频（text-to-video）！👀 Nova 模型将通过 Am... 独家提供。</li><li><a href="https://aider.chat/2024/12/03/qwq.html">QwQ 是代码架构师，而非编辑器</a>：QwQ 是类似于 o1 的 reasoning model，需要作为架构师（architect）与其他作为编辑器（editor）的模型配合使用。</li><li><a href="https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/providers?model=amazon.titan-image-generator-v1"">未找到标题</a>：未找到描述</li><li><a href="https://aider.chat/docs/usage/tutorials.html">教程视频</a>：由 aider 用户制作的入门和教程视频。</li><li><a href="https://github.com/BerriAI/litellm/releases/tag/v1.53.5">Release v1.53.5 · BerriAI/litellm</a>：更新内容：LiteLLM 小幅修复与改进 (12/03/2024)，由 @krrishdholakia 在 #7008 中提交；为 Azure OpenAI gpt-4o-2024-08-06 添加 prompt caching 标志，由 @fengjiajie 在 #7020 中提交；修复：添加凭据 t...</li><li><a href="https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-options.html">AWS CLI 中的命令行选项 - AWS Command Line Interface</a>：未找到描述</li><li><a href="https://youtube.com/@codingthefuture-jg1he?si=mjqG_DrpgMJcYG8C">与 AI 一起编码未来</a>：欢迎来到 Coding the Future With AI！我们的频道致力于帮助开发者和技术爱好者学习如何利用 AI 来增强技能和生产力。通过教程、专家访谈...</li><li><a href="https://aider.chat/docs/config/options.html#--gitignore">选项参考</a>：关于 aider 所有设置的详细信息。</li><li><a href="https://aider.chat/docs/usage/watch.html">IDE 中的 Aider</a>：Aider 可以在浏览器中运行，而不仅仅是在命令行中。</li><li><a href="https://youtu.be/t-i2x3APvGQ?si=pAp8W8-as258a-Sg">通过工作流驱动、调优的提示词链解锁 AI 编码 🔑</a>：在本教程中，我们将深入探讨一种系统化的 AI 软件构建方法，为您介绍一个由高度调优的提示词链驱动的工作流系统...</li><li><a href="https://github.com/Aider-AI/aider/issues/2525#issue-2715377909">请添加对 Anthropic 的 Model Context Protocol 的支持 · Issue #2525 · Aider-AI/aider</a>：Issue：请添加对 Anthropic 的 Model Context Protocol 的支持。版本和模型信息：最新</li><li><a href="https://github.com/lee88688/aider-composer">GitHub - lee88688/aider-composer: Aider 的 VSCode 扩展，无缝集成到 VSCode 中</a>：Aider 的 VSCode 扩展，无缝集成到 VSCode 中 - GitHub - lee88688/aider-composer</li><li><a href="https://github.com/BerriAI/litellm/pull/7019#issuecomment-2518028160">由 iwamot 添加 Amazon Nova 模型 · Pull Request #7019 · BerriAI/litellm</a>：标题：添加 Amazon Nova 模型。https://docs.aws.amazon.com/nova/latest/userguide/what-is-nova.html https://aws.amazon.com/bedrock/pricing/ 相关 Issue 类型：🆕 新功能。变更：[REQUIRED] T...</li><li><a href="https://github.com/aj47/100x-orchestrator">GitHub - aj47/100x-orchestrator：一个用于管理 AI 编码 Agent 的 Web 编排系统。该系统使用 Aider（一款 AI 编码助手）来处理编码任务，并通过用户友好的界面提供 Agent 输出的实时监控。</a>：一个用于管理 AI 编码 Agent 的 Web 编排系统。该系统使用 Aider（一款 AI 编码助手）来处理编码任务，并提供实时的...</li><li><a href="https://github.com/BerriAI/litellm/pull/7008">LiteLLM 小幅修复与改进 (12/03/2024)，由 krrishdholakia 提交 · Pull Request #7008 · BerriAI/litellm</a>：fix(key_management_endpoints.py)：在更新时覆盖 metadata 字段值；允许用户覆盖标签；feat(init.py)：暴露新的仅限 Prometheus 的 disable_end_user_cost_tracking 指标；允许禁用...</li><li><a href="https://github.com/chrishayuk/mcp-cli">GitHub - chrishayuk/mcp-cli</a>：通过在 GitHub 上创建一个账号来为 chrishayuk/mcp-cli 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=9mciRwpcLNY)">配合 Ollama 使用 Anthropic MCP，没有 Claude？看这个！</a>：Anthropic 发布了 Model Context Protocol，允许你将 LLM 连接到你自己的数据...</li>

nd 工具。在这段视频中，Chris 展示了如何将 MCP 与 c... 解耦</li><li><a href="https://docs.litellm.ai/docs/providers/bedrock">AWS Bedrock | liteLLM</a>：支持所有 Bedrock 模型（Anthropic、Meta、Mistral、Amazon 等）</li><li><a href="https://aider.chat/docs/llms/bedrock.html">Amazon Bedrock</a>：aider 是你终端中的 AI 结对编程工具</li><li><a href="https://github.com/Aider-AI/aider/issues/713">[FEATURE] 支持 Amazon Bedrock Claude Sonnet 3.5 · Issue #713 · Aider-AI/aider</a>：该 Issue 希望它不仅能通过 Anthropic 提供，也能通过 Amazon Bedrock 提供。https://aws.amazon.com/blogs/aws/anthropics-claude-3-5-sonnet-model-now-available-in-amazon-bedrock-the...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1313503299693187084)** (67 条消息🔥🔥): 

> `Aider Docker 设置, Aider 超时问题, 在 Aider 中使用本地模型, 在 Aider 中使用 MCP, 使用 Aider 进行函数重构` 


- **为本地模型设置 Aider Docker**: 成员讨论了在 Docker 中使用 Aider，特别是在 dev containers 和 Aider 容器之间共享卷时遇到的文件权限问题。
   - 该设置涉及在 CentOS 容器中运行 Aider，并尝试对齐用户设置，但导致了 "Permission denied" 错误。
- **处理 Aider 的超时问题**: 成员报告了在使用 `--timeout 5000` 运行 Aider 与本地服务器时出现超时错误，暗示这可能源于与超时设置相关的 litellm bug。
   - 尽管为 Aider 和本地模型都配置了超时，该过程仍遇到连接错误，其他用户也确认了与相同超时设置相关的持续问题。
- **在 Aider 中设置模型配置**: 成员讨论了 `.aider.model.settings.yml` 文件的正确设置，特别是提到 Aider 无法识别本地模型设置的问题。
   - 寻求关于这些配置文件位置以及如何确保 Aider 成功加载它们的澄清。
- **探索使用 Aider 进行函数重构**: 用户询问如何使用 Aider 在重构期间查找代码库中函数的所有实例，并指出了 Aider 在自动执行此任务方面的局限性。
   - 建议包括使用 IDE 工具或 RAG 工具来完成此类任务，并建议使用 shell 命令手动查找函数出现的位置。
- **在 Aider 中使用 Architect 模式**: 用户讨论了 Aider 中 Architect 模式的功能和设置，包括对该模式设置自定义模型的不确定性。
   - 确认通过 `--model` 参数指定的模型将决定 Architect 模式，从而允许在选择工作模型时具有灵活性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 aider 的常见问题。</li><li><a href="https://aider.chat/docs/config/aider_conf.html">YAML 配置文件</a>: 如何使用 yaml 配置文件配置 aider。</li><li><a href="https://aider.chat/docs/install/docker.html">Aider with docker</a>: aider 是你终端里的 AI 配对编程工具</li><li><a href="https://tenor.com/view/yup-dale-doback-step-brothers-yes-i-agree-gif-1579811350903017250">Yup Dale Doback GIF - Yup Dale doback Step brothers - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/Aider-AI/aider/issues/2209#issuecomme">功能请求：支持 llama.cpp · Issue #2209 · Aider-AI/aider</a>: llama.cpp 以服务器模式运行，如何使用？有关于用法的文档吗？</li><li><a href="https://github.com/BerriAI/litellm/issues/7001">[Bug]: 在 config.yaml 的模型层级设置 "timeout" 和 "stream_timeout" 不起作用 · Issue #7001 · BerriAI/litellm</a>: 发生了什么？我在 config.yaml 中设置了 "timeout" 和 "stream_timeout"，如下所示。- model_name: "gpt-4o" litellm_params: model: "azure/gpt-4o" api_k...</li><li><a href="https://m.youtube.com/watch?v=tElgVPUargw">使用 Aider Architect, Cursor 和 AI Agents 进行 AI 编程（基于 o1 的工程计划）</a>: 🔥 AI 代码编辑器大战已经开启！你的编码工作流为 o1 发布做好准备了吗？不要被淘汰！🚀🔥🔗 资源- 💻 Computer Use Bash &amp; ...</li><li><a href="https://youtu.be/9mciRwpcLNY?si=IqPQDJ-lgBlYGUre)">带有 Ollama 的 Anthropic MCP，没有 Claude？看这个！</a>: Anthropic 发布了 Model Context Protocol (MCP)，允许你将 LLM 连接到你自己的数据和工具。在这个视频中，Chris 展示了如何将 MCP 与 C 解耦...</li><li><a href="https://github.com/Aider-AI/aider/issues/2209#issuecomment-2453597627">功能请求：支持 llama.cpp · Issue #2209 · Aider-AI/aider</a>: llama.cpp 以服务器模式运行，如何使用？有关于用法的文档吗？
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1313948551742492783)** (3 messages): 

> `MCP 采用，OpenAI 的开发策略` 


- **MCP 引起关注**：一名成员表示 **MCP** 就是未来，并称对其重要性深信不疑。
   - 这种情绪反映了社区内对其潜在影响日益增长的热情。
- **对 OpenAI 发展方向的担忧**：人们希望 **OpenAI** 能采用 MCP，而不是在其开发选择上*重新造轮子*。
   - 这突显了在尊重该领域现有进展的同时进行创新的愿望。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1313504024993202197)** (119 messages🔥🔥): 

> `Mojo 网络特性，Mojo 中的 SIMD，高性能文件服务器，可扩展 Socket 开发，异步编程考量` 


- **等待语言更新的 Mojo 网络特性**：讨论强调了 Mojo 网络能力的持续开发，重点关注可插拔的网络后端，旨在利用 io_uring 的进步实现每核心 **25-40 Gbps 的 TCP 吞吐量**。
   - 正如多位成员指出的，基础网络功能预计将在更新后发布，以建立满足现代需求的高效 API。
- **Mojo 中 SIMD 的探索**：成员们讨论了在 Mojo 中使用 [SIMD](https://github.com/simdjson/simdjson) 操作的潜力，强调其实现比 C/C++ intrinsics 更易于使用。
   - Darkmatter 指出，大多数 SIMD intrinsics 理想情况下应该嵌入到标准库中，以减少对直接 intrinsic 调用的依赖。
- **构建高性能文件服务器**：一位成员提到正在为一款游戏开发**高性能文件服务器**，初步目标是比 Nginx 的 200 字节 HTTP 头部解析性能提升 **30% 的每秒数据包处理量**。
   - 对话涉及了实现效率的策略以及对健壮网络 API 支持的需求。
- **可扩展 Socket 框架的开发**：关于为 [可扩展 Socket](https://github.com/martinvuyk/forge-tools/tree/main/src/forge_tools/socket) 构建脚手架的讨论浮出水面，揭示了在 io_uring 和 POSIX sockets 等不同系统之间 **API 一致性** 的重要性。
   - Darkmatter 敦促相关开发者进行协作以达成一致，从而在 Mojo 中推广**坚实的网络基础**。
- **异步编程及其挑战**：辩论了**异步编程**的复杂性，特别是在处理协程和静态分派方面，重点关注潜在的性能陷阱。
   - 参与者强调了理解硬件差异的重要性以及不断改进方法的必要性，并举例说明了使用 trait 对象以获得最佳性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/s">s - 概览</a>：s 有 49 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://godbolt.org/z/E3381jM43">Compiler Explorer - C (x86-64 clang (trunk))</a>：/* 在此处输入代码，或加载示例。 */void square(__m128i a, __m128i b, __mmask8* k1, __mmask8* k2) {    _mm_2intersect_epi32(a, b, k1, k2);}</li><li><a href="https://github.com/marti">marti - 概览</a>：GitHub 是 marti 构建软件的地方。</li><li><a href="https://github.com/martinvu">MartinVu - 概览</a>：MartinVu 有 5 个可用仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d">在 Mojo 中使用 SIMD 统计字符</a>：Mojo 是一种非常年轻（实际上还在开发中）的编程语言，由一家名为 Modular 的新公司设计和开发。这里有一个……</li><li><a href="https://github.com/intel/hyperscan">GitHub - intel/hyperscan: 高性能正则表达式匹配库</a>：高性能正则表达式匹配库 - intel/hyperscan
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1313507812080619585)** (112 条消息🔥🔥): 

> `Inline Reference Concept, Memory Optimization Techniques, Compiler Support for Reference Traits, Bounds Checking for Mojo Lists, Auto-tuning in Compilation Phases` 


- **内联引用引发讨论**：提出了 `InlineReference` 类型的概念，允许在不存储地址的情况下实现更高效的内存访问模式，通过启用连续内存读取来潜在地提高性能。
   - 讨论强调了引用可用性与编译器可见性之间所需的平衡，以及对集成该功能可能产生的影响的担忧。
- **探索内存优化策略**：讨论重点关注小字符串（small string）和向量优化，强调这些优化如何通过在大型数组扫描期间启用零拷贝（zero-copy）场景来增强性能。
   - 社区成员表示有兴趣了解这些优化的实际用例，以及如何有效地实施它们。
- **编译器 Trait 受到关注**：关于 `Mutable` 和 `Addressable` trait 的提案引发了对其对编译器功能影响的辩论，建议这些 trait 可以在保持其实现不透明的同时获得原生支持。
   - 该模型有望在处理引用的方式上赋予更大的自由度，同时可能消除函数执行期间的别名（aliasing）问题。
- **边界检查机制正在审查中**：关于 Mojo 列表缺乏边界检查及其对安全性的影响正在进行讨论，目前已具备用于越界访问通知的调试检查（debug checking）。
   - 随着语言的成熟，未来的发展可能会包括改进的边界检查，这取决于用户反馈和已实现的功能。
- **考虑编译阶段的自动调优**：有人担心修订后的编译结构是否允许自动调优（auto-tuning）功能，暗示之前的某些功能可能会被重新整合。
   - 强调了在编译阶段需要专门支持，以提高未来版本的性能和适应性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/faq#distribution">MAX FAQ | Modular Docs</a>: 关于 MAX Engine 预期问题的解答。</li><li><a href="https://github.com/ParkMyCar/compact_str">GitHub - ParkMyCar/compact_str: A memory efficient string type that can store up to 24* bytes on the stack</a>: 一种内存高效的字符串类型，可以在栈上存储多达 24* 个字节 - ParkMyCar/compact_str
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1313511899119358013)** (124 messages🔥🔥): 

> `Dynamic 4-bit Quantization, Training Qwen Models, Using Colab for Fine-tuning, Model Performance Issues, SGLang Opinions` 


- **Unsloth 发布 Dynamic 4-bit Quantization**：Unsloth 宣布了他们全新的 **Dynamic 4-bit Quantization**（动态 4-bit 量化），旨在提高模型准确性的同时，比传统的 4-bit 方法占用更少的 VRAM。
   - *Naive quantization*（朴素量化）可能会损害模型准确性，但他们的方法会动态地选择不对某些参数进行量化。
- **Qwen 模型 Fine-tuning 问题**：用户报告称 **Qwen 2 VL 7B finetunes** 经常忽略训练数据，除非调整 *repetition penalty*（重复惩罚）和 *temperature*（温度）等特定参数。
   - 性能问题在 **Qwen** 和 **Pixtral models** 中似乎更为明显，导致训练过程中出现糟糕的结果。
- **使用 Colab 进行大数据集 Fine-tuning**：一位用户讨论了在 Colab A100 上使用 **304k 对话数据集** 对 *unsloth/Llama-3.2-1B-Instruct* 进行 Fine-tuning。
   - 用户对优化训练参数表示关注，因为在 Colab 上进行训练可能非常昂贵，特别是对于 **LATAM**（拉美）地区的用户。
- **关于 SGLang 的反馈**：用户 @bharatdeep04myfi_35111 询问了关于 **SGLang** 的使用体验，收到的反馈是它可以使用，但与 **VLLM** 相比速度较慢。
   - 普遍共识认为，虽然 SGLang 功能正常，但用户可能更倾向于选择性能更好的 VLLM。
- **模型的 Dynamic 4-bit 模式激活**：官方澄清，要利用 **Dynamic 4-bit mode**，用户必须将模型名称更改为以 'unsloth-bnb-4bit' 结尾。
   - 这一调整对于启用改进的性能至关重要，无需手动开启该功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">未找到标题</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1864380913922265300">来自 Unsloth AI (@UnslothAI) 的推文</a>：我们很高兴推出 Unsloth Dynamic 4-bit Quantization！朴素量化通常会损害准确性，使模型无法使用，但我们动态地选择不对某些参数进行量化。我们的方法...</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=QmUBVEnvCDJv">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h6ojwr/quantizing_to_4bits_can_break_models_dynamic/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: 可扩展且优化的 Transformers 构建模块，支持组合式构建。</a>：可扩展且优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://www.youtube.com/watch?v=pwGzyh3IiLU">Sports! | Tim and Eric Awesome Show, Great Job! Adult Swim DE</a>：Pünktlich zum Super Bowl führen euch Tim und Eric in die Welt des Sports ein...Ständig neue Videos gefällig? Abonniere den YouTube-Kanal von [adult swim] Deu...</li><li><a href="https://www.reddit.com/r/unsloth/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 llama 和 bert 进行文本分类的脚本 - timothelaborie/text_classification_scripts</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建一个账号来为 huggingface/smol-course 做出贡献。</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Fine-tuning Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80%</a>：Fine-tuning Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Fine-tuning Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80%</a>：Fine-tuning Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1313512234202431652)** (23 messages🔥): 

> `引用格式, 持续预训练, 模型对比, Reddit 社区` 


- **澄清引用格式**：一位成员询问如何正确引用 Daniel Han 和 Michael Han，另一位成员提供了一个示例引用格式以及 [GitHub 仓库链接](https://github.com/unslothai/unsloth)。
   - 对话中包括建议在仓库中添加 LaTeX/BibTeX 引用代码，以便于引用。
- **持续预训练的重要性**：讨论强调了 **Continued Pretraining (CPT)** 对于 **Llama-3** 等模型适应新领域和有效学习新 Token 至关重要。
   - 成员们指出，许多基座模型虽然在大规模数据集上进行了预训练，但在法律和医学等特定领域仍需要进行 CPT。
- **模型对比引发辩论**：**Claude** 和 **CodeLlama** 之间的对比凸显了 CodeLlama 已被视为过时，成员们建议使用 **Qwen2.5-coder** 等替代方案。
   - 成员分享了 **Qwen2.5-coder** 的结果与 Claude 相似的见解，表明了其在当前讨论中的相关性。
- **Reddit 社区衰落**：一位成员分享了他们对 Reddit 社区衰落的沮丧，他们之前活跃于 50 个 subreddits，现在减少到几个感觉像“坟场”的社区。
   - 他们指出，像 **localLlama** 这样特定的 subreddits 在一次崩溃事件后变得越来越消极，导致参与度下降。
- **理解模型局限性**：有人提出了关于直接使用训练数据集样本的影响问题，这与 **temperature** 和数据集多样性的概念相关。
   - 一位成员评论了他们在 **wizardlmMath** 上的成功经验，同时也对模型在没有创造力的情况下证实有限样本的情况表示担忧。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1313513970057084949)** (35 messages🔥): 

> `微调 Llama 3, GGUF 转换问题, Google Colab 中的 ReadTimeout 错误, 在 Unsloth 中使用多 GPU, 训练中的 Adapter 配置错误` 


- **微调 Llama 3 面临问题**：用户在微调 Llama 3 时遇到了问题，包括由于 llama.cpp 中缺少文件，在将模型保存为 GGUF 时出现运行时错误。
   - 此外，一些用户注意到切换到不同的 Notebook 版本并不能解决这些问题，他们正在等待更新。
- **GGUF 转换方法**：在 GGUF 转换挑战中，用户讨论了潜在的解决方案和替代方法，一些人建议使用不同的 Colab 设置。
   - 参与者分享了正确转换方法的链接和资源，并指出目前唯一的选择是使用 Unsloth 框架。
- **解决 ReadTimeout 错误**：几位用户在尝试在 Google Colab 中加载模型时面临 ReadTimeout 错误，这表明可能存在连接问题。
   - 其他人建议重新构建 Docker 或检查容器中的互联网访问可能会解决这些超时问题。
- **多 GPU 的使用**：关于 Unsloth 隐藏多 GPU 的讨论引发了关于在微调模型时同时利用它们执行其他任务的问题。
   - 提到了一项可能的 Pull Request 来解决这一限制，尽管它仍在审查中。
- **微调期间的 Adapter 配置错误**：一些用户在尝试微调模型时遇到了与 Adapter 配置相关的错误，特别是 'Requested bias: none' 未被实现。
   - 调整 bias 设置等替代方案也产生了类似的错误，导致用户寻求解决这些配置问题的指导。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/12hkbsOMJfYfmqJLA93cV5tGoPIeZ5gDK#scrollTo=oAC_WYSUX7k_">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit">unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/phi3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-4285.">使用 Unsloth 微调 Phi-3</a>: 通过 Unsloth 轻松微调微软的新模型 Phi 3 medium, small &amp; mini，支持 6 倍长的上下文长度！</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct/blob/main/config.json">config.json · unsloth/Phi-3.5-mini-instruct at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1313909574209634334)** (1 条消息): 

> `Fimbulvntr 的文章` 


- **Fimbulvntr 发布首篇文章**：Fimbulvntr 刚刚在 [X](https://x.com/fimbulvntr/status/1864350663204852054) 上发表了他们的第一篇文章，展示了对相关主题的精彩见解。
   - 有兴趣深入了解的人可以通过此 [链接](http://x.com/i/article/1864344035466637312) 直接访问该文章。
- **Fimbulvntr 对新趋势的见解**：在文章中，Fimbulvntr 讨论了技术领域的新兴趋势，强调了适应性的重要性。
   - 鼓励读者参与内容互动，并在评论中提供想法。



**提到的链接**：<a href="https://x.com/fimbulvntr/status/1864350663204852054">来自 Fimbul (@fimbulvntr) 的推文</a>：http://x.com/i/article/1864344035466637312

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/)** (1 条消息): 

edd0302: https://x.com/ruliad_ai/status/1864394941029322890
  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1313503386049450047)** (156 条消息🔥🔥): 

> `Amazon Nova 模型、用户界面问题、Perplexity AI 性能、Pro 订阅问题、模型可用性与扩展` 


- **Amazon Nova 发布令用户印象深刻**：用户讨论了新的 **Amazon Nova** 基础模型，注意到其**速度**和**准确性**，并热切期待在 **Perplexity Pro** 中使用。
   - 早期实验获得了积极反馈，用户强调了这些模型在 AI 驱动任务中实现高性能的潜力。
- **Mac 应用界面投诉**：许多用户对 **Mac 应用** 表示不满，理由是与 Web 版本相比，其**性能缓慢**且**界面笨拙**。
   - 用户还提出了电池消耗过快的问题，引发了关于未来修复的讨论。
- **Pro 订阅困惑**：几位用户对订阅费用和不一致性表示沮丧，特别是关于**首月 4.99 美元**的价格变成了更高的费用。
   - 用户对支持学生免费访问的财务模式感到好奇，从而引发了关于 API 访问和 Pro 功能的更广泛讨论。
- **模型访问与变更问题**：用户对某些模型（如 **O1-mini**）的受限访问表示担忧，质疑这些限制是否与订阅级别或整体服务变更有关。
   - 用户还讨论了围绕 **Complexity 扩展**的困惑，包括其合法性以及无法在界面中添加新模型的问题。
- **语言与回答质量**：一些用户遇到了 AI 意外的语言输出，特别是回答出现 **中文** 或其他与语言偏好相关的错误。
   - 讨论包括调整回答语言设置的技巧，以及在不同模型之间切换的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sse-next-one.vercel.app/">Server Sent Events</a>：未找到描述</li><li><a href="https://tenor.com/view/men-i-trust-emma-poulx-show-me-how-you-deserve-this-gif-25757415">Men I Trust Emma Poulx GIF - Men I Trust Emma Poulx Show Me How - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">介绍 Amazon Nova，我们的新一代基础模型</a>：来自 Amazon 的新型最先进基础模型，提供前沿智能和行业领先的性价比。</li><li><a href="https://tenor.com/view/bella-ketchup-swan-twilight-edward-gif-18684497">Bella Ketchup GIF - Bella Ketchup Swan - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/spelling-gif-9068510">Spelling GIF - Spelling - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=APO7WHP8Ozw">实时 AI 搜索对决：ChatGPT Search vs. Perplexity vs. Google vs. Copilot vs. Grok</a>：AI 正在接管搜索。🤖 无论你喜欢还是讨厌，LLM 驱动的搜索正在进入你的设备。↳ ChatGPT Search 及其 Chrome 扩展。↳ Go...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1313530032865411104)** (3 条消息): 

> `Heisenberg Heat, 软件优化工具, Perplexity API 功能` 


- **Heisenberg Heat 询问引发关注**：分享了一个讨论 **Heisenberg Heat** 概念的链接，邀请大家探索其原理和影响。
   - 鼓励成员深入研究相关的**理论探讨**和**实际应用**。
- **Careit 被评为顶级软件优化工具**：讨论强调了 **Careit** 在软件优化工具中荣获 **#1 顶级排名**，这归功于团队的努力。
   - 这一成就引发了社区内的兴奋，大家对他们的努力和成果表示赞赏。
- **了解 Perplexity API 机制**：分享了一个解释 **Perplexity API** 运作方式的资源，详细介绍了其特性和能力。
   - 该说明旨在为社区开发者澄清其**功能**和潜在的使用场景。


  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1313531881316159613)** (8 条消息🔥): 

> `API 支付问题, 企业版候补名单, API 质量投诉, 支持沟通, GitHub 讨论论坛` 


- **尽管有余额但仍存在 API 支付问题**：一位成员质疑，尽管 API 账户中有余额，为什么仍从信用卡扣款，并提到发送支持邮件两天后仍未收到回复。
   - 由于支持响应不及时，用户的挫败感日益增加，部分用户因这些问题考虑更换服务商。
- **企业版候补名单详情**：另一位成员分享道，根据团队成员的邮件确认，企业版访问的候补时间大约为几周。
   - 这反映了持续的需求以及在处理企业申请方面的一些积压。
- **对 API 质量的担忧**：一位用户对 **API 质量**提出了重大担忧，声称它在他们的使用场景中已变得无法使用，这可能导致他们更换服务商。
   - 这一投诉暗示了一个更广泛的问题，最近几周有多位用户表达了不满。
- **关于支持邮件有效性的疑问**：在关于支持响应速度的讨论中，一位成员建议联系支持邮箱寻求帮助，并指出由于**企业咨询**较多可能会有延迟。
   - 一位成员鉴于目前对响应时间的不满，对联系支持的有效性表示怀疑。
- **访问 GitHub 讨论论坛**：一位成员指向了 [GitHub 讨论论坛](https://github.com/ppl-ai/api-discussion/discussions)，作为发表对 API 投诉的场所，鼓励其他人也在那里发布问题。
   - 另一位成员提到，他们也提交了一个关于系统提示词（system prompt）失效的讨论话题，显示出在寻求解决方案方面的积极参与。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/discussions/discussions">Forum - Perplexity</a>：未找到描述</li><li><a href="https://github.com/ppl-ai/api-discussion/discussions/80">Web search not being performed · ppl-ai/api-discussion · Discussion #80</a>：我正在提交一个带有示例的 "system" 提示词，然后是一个类似于下面的 "user" 提示词，要求它搜索互联网。直到一两周前，这还是一致的...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/)** (1 条消息): 

alexatallah: Claude 3.5 Haiku 降价 20%！
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1313491688936968252)** (148 条消息🔥🔥): 

> `Hermes 405B 免费服务状态、Gemini Ultra 访问权限、Amazon Nova 模型讨论、模型记忆功能、自定义提供商密钥 (Custom Provider Keys) Beta 测试` 


- **Hermes 405B 免费服务停止**：**Hermes 405B** 的免费服务已移除，这很可能是由于提供商的决定而非 OpenRouter 的操作，导致用户感到失望。
   - 一些用户正在探索其他选择，但尽管该服务取消，**基础 405B 模型**仍然可以免费使用。
- **Gemini Ultra 的可用性**：关于 **Gemini 1.0 Ultra** 的讨论正在进行，传闻其已可用，但目前访问受限于白名单 (allowlists)。
   - 用户认为 Google 模型的发布和版本命名导致了混乱，并推测 Ultra 可能很快会被停用。
- **关于 Amazon Nova 用于创意写作的讨论**：用户对 **Amazon Nova** 模型在创意写作任务中的效果感到好奇，并正在寻求个人使用经验。
   - 有推测认为，虽然 Nova 正在接受评估，但其与 Runway 等其他模型相比的能力仍不确定。
- **模型记忆与上下文扩展**：一位用户询问模型是否具有记忆以保留之前的交互，建议倾向于使用自托管方案来进行上下文扩展。
   - 推荐了诸如总结过去消息以延长上下文长度等方法作为替代方案。
- **申请自定义提供商密钥的早期访问**：用户希望获得自定义提供商密钥 (Custom Provider Keys) 功能的访问权限，该功能目前处于 Beta 阶段，未来可能会产生费用。
   - 要申请早期访问，用户被引导至特定的 Discord 频道以获取更多信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://developers.cloudflare.com/ai-gateway/">AI Gateway · Cloudflare AI Gateway 文档</a>: Cloudflare 的 AI Gateway 让您可以获得对 AI 应用的可视化和控制。通过将应用连接到 AI Gateway，您可以利用分析功能深入了解人们如何使用您的应用程序...</li><li><a href="https://developers.cloudflare.com/ai-gateway/providers/open-router/">OpenRouter · Cloudflare AI Gateway 文档</a>: OpenRouter ↗ 是一个为访问和使用大语言模型 (LLMs) 提供统一接口的平台。</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://lambdalabs.com/blog/unveiling-hermes-3-the-first-fine-tuned-llama-3.1-405b-model-is-on-lambdas-cloud">揭晓 Hermes 3：首个全参数微调的 Llama 3.1 405B 模型已上线 Lambda 云</a>: 与 Nous Research 合作推出 Hermes 3，这是 Meta Llama 3.1 405B 模型的首个微调版本。使用 Lambda 训练、微调或部署 Hermes 3。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/start/express-mode/overview">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/meta-llama/llama-3.1-405b-instruct:free">Llama 3.1 405B Instruct (免费) - API、提供商、统计数据</a>: 备受期待的 Llama3 400B 级别模型来了！它拥有 128k 上下文和令人印象深刻的评估分数，Meta AI 团队继续推动开源 LLM 的前沿。Meta 最新的...</li><li><a href="https://openrouter.ai/anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet - API、提供商、统计数据</a>: 新的 Claude 3.5 Sonnet 以相同的 Sonnet 价格提供了优于 Opus 的能力和快于 Sonnet 的速度。通过 API 运行 Claude 3.5 Sonnet。</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/models#gemini-1.0-ultra">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=HQ8AUBn-4DY">如何判断你的社交媒体成瘾是否已经过头</a>: 如何判断你对 FarmVille 的痴迷是否是一个大问题：http://www.yourtango.com/201064181/social-media-addiction-are-you-risk 呈现 A YourTango...</li><li><a href="https://aws.amazon.com/cn/bedrock/pricing/">使用基础模型构建生成式 AI 应用 - Amazon Bedrock 定价 - AWS</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1313640513349357760)** (6 messages): 

> `Custom Key Beta Access` 


- **社区渴望获得 Custom Key Beta 访问权限**：多位成员表达了希望获得 **custom key beta** 访问权限的愿望，并纷纷举手申请。
   - 一位成员恳求道：*'我也想要 custom key beta 访问权限！'*，而另一位成员则对团队的努力表示感谢，无论时间表如何。
- **询问密钥访问的时间表**：一位成员询问了获取 **custom keys** 的预计时间表，想知道是否有人能给出一个猜测。
   - 他们对不确定性表示理解，称：*'我们完全理解，并感谢你们所有人的辛勤工作。'*


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1313497012351668256)** (115 messages🔥🔥): 

> `Distributed Training Run, Forge Reasoning API Beta, Live Memory in LLMs, Genesis of AI World Models, Nous Research Art and Design` 


- **分布式训练运行接近完成**：一个分布式训练运行目前正在进行中，将在一天多后完成，预先安排的计算合作伙伴从一开始就参与其中。
   - 预计在不久的将来会发布有关该训练运行的更多细节，并且公众参与的可能性也得到了认可。
- **发布 Forge Reasoning API Beta**：Nous Research 发布了 Forge Reasoning API Beta，旨在缩短各种模型的推理时间，并可能增强 Hermes 70B 的能力。
   - 这一新进展紧随社区对大规模基础模型及其实际应用的兴趣之后。
- **关于在 LLM 中实现实时内存的讨论**：成员们探讨了在 LLM 架构中实现实时内存的想法，在是使用 function calls 还是 RAG 方法以获得更好的连贯性和性能之间进行了辩论。
   - 大家达成共识，认为经典方法可以更好地以可靠的方式锚定神经网络，同时实现风格的一致性。
- **生成式世界模型的创新**：对话转向了生成式“世界模型”的创建（类似于视频游戏），以及它们如何结合经典软件进行可靠的数据操作。
   - 参与者建议使用混合系统，通过结合神经和传统方法来提高输出质量。
- **对 Nous Research 的艺术贡献**：社区成员对 Nous Research 的艺术方向表示感兴趣，透露 John Galt 担任他们的首席设计师。
   - 艺术与 AI 系统设计之间的相互作用被幽默地提及，反映了团队内部独特的文化。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://distro.nousresearch.com/">Nous DisTrO</a>: 互联网上的分布式训练</li><li><a href="https://modal.com/pricing">Plan Pricing</a>: 简单透明的定价，根据你使用的算力规模进行扩展。</li><li><a href="https://x.com/jparkerholder/status/1864314826891079787">来自 Jack Parker-Holder (@jparkerholder) 的推文</a>: 介绍 🧞Genie 2 🧞 —— 我们最强大的大规模基础世界模型，能够生成各种一致的世界，可玩时长达一分钟。我们相信 Genie 2 可以开启...</li><li><a href="https://x.com/NousResearch/status/1856417883934601246">来自 Nous Research (@NousResearch) 的推文</a>: 今天我们将发布 Forge Reasoning API Beta 版，这是推理时间扩展（inference time scaling）方面的一项进展，可以应用于任何模型或模型集，面向我们社区中的特定群体。https:/...</li><li><a href="https://www.jetson-ai-lab.com/tutorial_llamaspeak.html#function-calling">
   llamaspeak - NVIDIA Jetson AI Lab
  </a>: 未找到描述</li><li><a href="https://x.com/SHL0MS/status/1864371949322829978?t=yDG98l6fCD23fuGjamiC2Q&s=19">来自 𒐪 (@SHL0MS) 的推文</a>: 你好 @s8n 😈上帝与撒旦现在作为 @NousResearch 模型统一了。我们将在未来几天对两者进行迭代，以完善它们的动态和发布风格。引用 𒐪 (@SHL0MS)，正如你们许多人已经...</li><li><a href="https://www.youtube.com/watch?v=gzuYdUAPXxw">Elliott Smith - 13 - Independence Day</a>: 纽约市政厅...曲目列表：Son of Sam, Happiness, Between the Bars, LA, Rose Parade, Pretty Mary K, Angeles, Needle in the Hay, Say Yes, Waltz #2, St. Ide's Heaven, Easy ...</li><li><a href="https://github.com/archit-spec/modal-scripts/blob/main/jupyter_training.py#L75">modal-scripts/jupyter_training.py at main · archit-spec/modal-scripts</a>: 用于训练 SLM/TTS 模型的 Modal 脚本示例。通过在 GitHub 上创建账户，为 archit-spec/modal-scripts 的开发做出贡献。</li><li><a href="https://www.are.na/john-galt/nous-research-john-galt">NOUS RESEARCH / JOHN GALT | Are.na</a>: 我在 Nous Research 工作的一个样本。</li><li><a href="https://github.com/archit-spec/modal-scripts/tree/main">GitHub - archit-spec/modal-scripts: example modal scripts for training slm/tts models</a>: 用于训练 SLM/TTS 模型的 Modal 脚本示例。通过在 GitHub 上创建账户，为 archit-spec/modal-scripts 的开发做出贡献。</li><li><a href="https://t.co/5be7RgCUTL">DeMo: Decoupled Momentum Optimization</a>: 训练大型神经网络通常需要通过专门的高速互连在加速器之间共享梯度。借鉴频率分解的信号处理原理...</li><li><a href="https://github.com/bloc97/DeMo">GitHub - bloc97/DeMo: DeMo: Decoupled Momentum Optimization</a>: DeMo: Decoupled Momentum Optimization。通过在 GitHub 上创建账户，为 bloc97/DeMo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1313556527017230449)** (5 messages): 

> `Nous Research 兴趣，Linux from Scratch 作为基准测试，语音 Agent 的精度，Residual Stream 中的动量概念` 


- **技术社会主义者对 Nous 的兴趣**：一位成员作为**技术社会主义者**对 **Nous** 表达了浓厚的兴趣，强调了他们的价值观与项目目标的一致性。
   - 这反映了社区内对 **AI 进步**的社会影响日益增长的好奇心。
- **使用 Linux from Scratch 作为基准测试**：有人提出了关于使用 **Linux from Scratch** 一书作为评估 AI Agent 的基准测试的实用性问题。
   - 这表明人们有兴趣建立**具体指标**，以评估 Agent 在现实应用中的性能。
- **实现语音 Agent 的精度**：一位成员询问了实现**语音 Agent 精度**的方法，特别是针对销售等特定用例。
   - 讨论指出，在定制数据集上进行 **fine-tuning** 是提高准确性的一种潜在方法。
- **在数学概念中引入动量**：一位成员提出了将**动量（momentum）**概念整合到 **residual stream** 架构中的想法，并对其数学基础提出了疑问。
   - 这引发了一场有趣的对话，讨论**加法和 skip connections** 是否足以实现类似的效果。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/)** (1 messages): 

jellyberg: https://theaidigest.org/agent
  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1313539156831637626)** (5 messages): 

> `DisTro issues, Logical Consistency, DeLorean Reference` 


- **DisTro 与 Flux Capacitor 的难题**：一位成员幽默地质疑 **DisTro** 是否是与 **flux capacitor**（通量电容器）一同发明的，暗示对其功能的困惑。
   - “*它拒绝承认这里存在问题*”是一个表达出的显著观点。
- **逻辑一致性讨论**：一位成员评论道，“*它是逻辑自洽且一致的——如果没别的话……*”，暗示对话的某些方面经受住了推敲。
   - 这一评论紧随之前的幽默批评之后，反映了一种轻松但带有批判性的基调。
- **对 DeLorean 的渴望**：一位成员以俏皮的语气表达了想要拥有属于自己的 **DeLorean** 的愿望，引用了其标志性的地位。
   - 这一评论捕捉到了一种怀旧的奇思妙想，反映了讨论中共享的热情。



**Link mentioned**: <a href="https://hermes.nousresearch.com)">no title found</a>: no description found

  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1313860690754404462)** (1 messages): 

> `NotebookLM+Spotify, Spotify Wrapped AI Podcast` 


- **NotebookLM 与 Spotify 合作**：**NotebookLM** 与 **Spotify** 联手打造个性化的 **AI podcast**，总结你的年度音频，该消息于 [2024年12月4日](https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/) 发布。
   - **Spotify Wrapped AI podcast** 提供动态的音频回顾，利用 **NotebookLM** 来解析用户最喜爱的曲目和艺术家。
- **Spotify Wrapped AI podcast 的精彩功能**：今年的 **Spotify Wrapped** 通过 **AI** 功能增强了用户体验，呈现关于听歌习惯的定制内容。
   - 当听众参与播客时，会有 **AI hosts** 带领他们探索定义其音乐年度的内容。



**Link mentioned**: <a href="https://blog.google/technology/google-labs/notebooklm-spotify-wrapped/">Listen to your first-ever 2024 Spotify Wrapped AI podcast, built with Google&#x27;s NotebookLM</a>: NotebookLM 正在与 Spotify 合作，打造个性化的 Wrapped AI podcast。

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1313563337820209193)** (23 条消息🔥): 

> `AI 音频生成，NotebookLM 用于体育新闻，法律内容简化，多语言 AI 讨论，使用 AI 的创意项目` 


- **AI 生成搞笑的多语言音频**：一段展示 AI 多语言能力的音频片段获得了积极反馈，一位成员指出它有时会失去焦点，但偶尔能回到正确的语言。
   - 另一位成员询问是否包含波兰语，这表明语言设置的效果参差不齐。
- **NotebookLM 彻底改变体育专题报道**：一位成员强调了使用 NotebookLM 为职业运动队创建每日赛前和赛后专题报道的潜力，并建议这可以轻松扩展到不同队伍。
   - 他们强调了在品牌化 Avatar 的同时生成内容的简便性，这可以增强粉丝的参与度。
- **NotebookLM 让法律内容变得简单**：另一位成员称赞了 NotebookLM 在解析复杂法律术语方面的有效性，使普通用户更容易理解法律内容，特别是关于各州数据法律的内容。
   - 这被认为是一个简化法律信息的日常工具。
- **使用 AI 的独特创意项目**：一位用户分享了一个由 AI 创建的德语恶搞小组讨论，探讨了诸如生命意义之类的哲学主题，展示了 AI 生成内容的幽默潜力。
   - 成员们对 AI 在制作引人入胜且具有娱乐性的对话方面的潜力非常感兴趣。
- **从导出的聊天记录生成音频**：一位用户对从讨论泰国美食优惠的聊天记录导出并生成音频感到兴奋，并认可了 NotebookLM 中集成的有效音频功能。
   - 文中提到了公开分享此类内容所需的权限，突显了社区协作的方面。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson: Exploring the Intersection of AI and Rendering</a>: Zap Andersson 测试 AI Video 工具，并分享他奇特的 YouTube 系列《UNREAL MYSTERIES》中的技巧和心得。</li><li><a href="https://notebooklm.google.com/notebook/50b3f4f0-7701-4242-a705-1bf9fd7a0c35?_gl=1*1loyke6*_ga*MTgzODEzOTkwNS4xNzMxNzQ5NjYx*_ga_W0LDH41ZCB*MTczMzMyMDIyMC4yLjEuMTczMzMyMDIyMC42MC4wLjA.&original_referer=https:%2F%2Fnotebooklm.google%23&pli=1">未找到标题</a>: 未找到描述</li><li><a href="https://notebooklm.google.com/notebook/50b3f4f0-7701-4242-a705-1bf9fd7a0c35/audio">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/D7qZ2VphetU">NBA CUP POC</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1313490556407648317)** (92 条消息🔥🔥): 

> `Notebook LM Language Settings, Notebook LM PDF Capabilities, Notebook LM Features Requests, Google Job Listings, Notebook LM Podcast Integration` 


- **Notebook LM 语言设置面临的挑战**：用户在更改 Notebook LM 的语言设置时遇到困难，特别是针对 Podcast。一位用户提到，尽管他们将 Google 账户设置为印尼语，但并没有改变 Podcast 内容的语言。
   - 另一位用户指出，在上传脚本后尝试生成葡萄牙语的音频内容时感到困惑和失望。
- **对 PDF 读取能力的担忧**：关于 Notebook LM 读取长篇 PDF 并准确提取相关信息的能力（特别是与其他 AI 模型相比）出现了一些疑问。用户对文档访问不完整和摘要更新不及时表示沮丧。
   - 一位用户特别指出，在上传两个 PDF 后，摘要仅反映了第一个文档的内容，强调了对更好的刷新选项的需求。
- **功能请求和用户体验增强**：用户请求了诸如对 Notebook 进行分类以及为 Podcast 生成转录文本等功能，这可能符合企业政策。对当前框架的反馈表明，用户希望拥有允许手动编辑以保持合规性的功能。
   - 另一个请求集中在保存常用问题模板，以便在不同的 Notebook 中轻松使用，强调了用户对高效学习工具的需求。
- **分享 Google 工作机会**：一位 Google 员工分享了 Google 开放职位的链接，提供了软件工程角色所需资格的见解。讨论的职位有丰富的经验要求，表明对技术专长的强烈关注。
   - 对话还幽默地提到了聘请一位“NotebookLM 宣传员”的想法，展示了尽管不是技术开发人员，但对该产品的热情。
- **对 Notebook LM 发展的热情表达**：用户对 Notebook LM 与 Spotify 的集成及其对个人体验的影响表示兴奋。许多人指出他们对主流采用和该技术潜力的期待，表明有一个充满活力的社区在支持这项创新。
   - 评论展示了幽默与对产品的钦佩，并提到了引起观众共鸣的个人经历。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson: Exploring the Intersection of AI and Rendering</a>：Zap Andersson 测试 AI Video 工具，并分享他奇特的 YouTube 系列《UNREAL MYSTERIES》中的技巧和窍门。</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/137740784886522566-senio">Senior Software Engineer, Full Stack, Labs — Google Careers</a>：未找到描述</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/137740784886522566-senior-software-engineer-full-stack-labs">Senior Software Engineer, Full Stack, Labs — Google Careers</a>：未找到描述</li><li><a href="https://www.google.com/about/careers/applications/jobs/results/101552613576581830-software-engineer-iii-full-stack-labs">Software Engineer III, Full Stack, Labs — Google Careers</a>：未找到描述</li><li><a href="https://youtu.be/wEAeP1Po3EI?feature=shared">🎉🤖 AI Clown Bot Unmasked! 🤖🎉</a>：💥 世界接管……用小丑表演？💥🚨 观看由 AI 小丑机器人制作的首个 Podcast 🚨 这不是典型的小丑表演。想象一下：🎭 AI 生成的视觉效果……
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1313894573990481991)** (3 条消息): 

> `NeurIPS Meetup, Interconnects Open Hangouts` 


- **同事们计划在 NeurIPS 见面**：一位成员表达了参加 NeurIPS 的兴奋之情，并希望加入 **Interconnects 聚会**。
   - 他们提到将与一位同事一同参加。
- **Nat 的 Open Hangout 计划**：Nat 表示他们将在本周晚些时候为 NeurIPS 期间的 **Open Hangouts** 提议几个时间点。
   - Nat 表示他们将在下周三通过电子邮件提供详细信息。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1313591150568931328)** (26 messages🔥): 

> `Amazon Foundation Models, Genie 2, 12 Days of OpenAI, ChatGPT interface updates, Anduril and OpenAI partnership` 


- **Amazon 发布新款 Foundation Models**：Amazon 在 re:Invent 大会上推出了 **6 款全新的 Foundation Models**，包括 **Micro**、**Lite**、**Pro** 和 **Premier**，功能涵盖从文本生成到文本生成视频，仅通过 [Amazon Bedrock](https://link.to.amazonbedrock) 提供。
   - *Performance benchmarks* 与 Llama 3 持平，支持高达 **300K tokens**，Amazon 旨在为开发者提供多样化的解决方案。
- **为 Embodied Agents 推出 Genie 2**：[Genie 2](https://fxtwitter.com/jparkerholder/status/1864314826891079787) 承诺可以生成长达 **一分钟** 的多样且一致的世界，增强了 Embodied Agents 的能力。
   - 成员们对其在未来 AI 创新方面的潜力感到兴奋。
- **OpenAI 的 12 天直播活动开启**：'**12 Days of OpenAI**' 开始，包含 **12 场直播** 展示各种公告，第 1 天发布了关于新员工的新闻稿。
   - 成员们推测可能存在与此次活动直接相关的 *interface* 变更、新计划和更新。
- **ChatGPT Interface 可能的更新**：成员们讨论了潜在的功能，如 'pro plan'、更新的语音以及可能与 ChatGPT 即将到来的更新相关的图像生成新功能。
   - 随着对功能增强和 Sora API 发布推测的继续，现场气氛非常热烈。
- **Anduril 与 OpenAI 建立合作伙伴关系**：**Anduril** 与 OpenAI 的合作旨在通过由 Lattice 驱动的系统提升 **美国人工智能** 的领导地位，实现跨领域的集成安全。
   - 该伙伴关系强调致力于通过创新技术支持武装部队的任务。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fxtwitter.com/jparkerholder/status/1864314826891079787">来自 Jack Parker-Holder (@jparkerholder) 的推文</a>: 介绍 🧞Genie 2 🧞 - 我们最强大的大规模 Foundation World Model，它可以生成各种一致的世界，可播放长达一分钟。我们相信 Genie 2 可以开启...</li><li><a href="https://x.com/_philschmid/status/1864016010464080260">来自 Philipp Schmid (@_philschmid) 的推文</a>: 意料之外。@amazon 带着 Foundation Models 回来了。作为 re:Invent 的一部分，他们宣布了 6 个新的 Foundation Models，从纯文本到文本生成视频！👀 Nova 模型将仅通过 Am... 提供。</li><li><a href="https://www.anduril.com/article/anduril-partners-with-openai-to-advance-u-s-artificial-intelligence-leadership-and-protect-u-s/">Anduril 与 OpenAI 合作，提升美国人工智能领导地位并保护美国及其盟军</a>: 国防技术公司 Anduril Industries 与 ChatGPT 及 GPT-4o、OpenAI o1 等前沿 AI 模型的制造者 OpenAI 自豪地宣布建立战略合作伙伴关系，以开发和研究...</li><li><a href="https://x.com/OpenAI/status/1864328928267259941">来自 OpenAI (@OpenAI) 的推文</a>: 12 天。12 场直播。一堆大大小小的新事物。12 Days of OpenAI 明天开始。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1313612647287226509)** (18 messages🔥): 

> `Mistral Large 性能, OpenAI 苏黎世办公室, Giffmana 伦理辩论` 


- **Mistral Large 在 CLI 测试中表现出色**：一位成员称赞 **Mistral Large 2**，表示它在处理 bash 脚本和查询方面优于 **3.5 Sonnet** 和 **GPT-4**，并引用了一篇 [tweet](https://x.com/TheXeophon/status/1833921199170355480) 称其对 shell 了如指掌。
   - 另一位用户幽默地指出，有了 AI 和在线 bash 解释器，再也不需要记住 **ffmpeg flags** 了。
- **关于 OpenAI 苏黎世新办公室的猜测**：讨论暗示 OpenAI 将为某位知名人物及其同事在 **Zürich** 设立办公室，这引发了对其近期关于 ML 伦理社交媒体帖子的质疑。
   - 有人对 **GDM** 人员离开他们舒适的 **TPU** 环境去寻求新机会表示担忧。
- **AI 伦理受到审查**：用户猜测某位人物近期发帖的伦理影响，将其与他加入 OpenAI 的转变联系起来，并暗示其在线讨论背后有更深层的动机。
   - 一位成员指出，转向使用 GPU 进行“家庭实验”解释了他之前的活动和动机。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1833921199170355480">来自 Xeophon (@TheXeophon) 的推文</a>: 顺便说一句：bash 的最佳模型是——我不骗你——Mistral Large 2。在我测试脚本或一般问题时，它的表现都优于 3.5 Sonnet 和 GPT-4。后者经常尝试奇怪的东西...</li><li><a href="https://x.com/_xjdr/status/1833921835320443002">来自 xjdr (@_xjdr) 的推文</a>: @TheXeophon 哈哈，这正是驱动这次交互的模型（以 deepseek coder 作为备选）
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1313578112864354304)** (13 messages🔥): 

> `Amazon 新的基础模型, 关于 NVIDIA SANA 许可的担忧, IFEval 基准测试饱和` 


- **Amazon 新的基础模型发布**：Amazon 在 re:Invent 期间宣布了 **6 款新的基础模型 (Foundation Models)**，包括 **Nova Micro**（纯文本）和 **Reel**（视频生成）模型，这些模型仅通过 Amazon Bedrock 提供。
   - 这些模型将支持 **高达 300K tokens** 和 **200 多种语言**，Micro 模型的定价细节为 **$0.035**。
- **NVIDIA 的 SANA 许可证引发愤怒**：来自 NVIDIA 的 **SANA 模型** 速度很快，但其 **许可证限制使用** 于非商业应用且仅限在 NVIDIA GPU 上运行，许多人认为这不合理。
   - 讨论中提出了对其强制执行的担忧，例如限制在 AMD 机器上运行，以及公司保留对生成输出的权利。
- **关于 IFEval 基准测试相关性的讨论**：成员们质疑 **IFEval 基准测试** 是否仍然具有相关性，或者它是否已经变得饱和，因为许多模型都能轻松获得高分。
   - 评论指出，**90% 的基准测试分数** 正在变得司空见惯，引发了关于什么是新的 meta 基准测试的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1864016010464080260">来自 Philipp Schmid (@_philschmid) 的推文</a>: 意料之外。@amazon 带着基础模型回来了。作为 re:Invent 的一部分，他们宣布了 6 款新的基础模型，从纯文本到文本转视频！👀 Nova 模型将通过 Am 独家提供...</li><li><a href="https://fxtwitter.com/cloneofsimo/status/1864309440356470894">来自 Simo Ryu (@cloneofsimo) 的推文</a>: 来自 nvidia 的 SANA 速度很快且相当不错，但它的许可证简直是场悲剧。它仅供非商业使用（这没问题），但由于某些糟糕的原因，你只能在 NVIDIA-gpu 上运行它。这到底是怎么回事...</li><li><a href="https://x.com/cloneofsimo/status/1864312857674043599">来自 Simo Ryu (@cloneofsimo) 的推文</a>: * 忘记放 http://SANATransformerModel.to(&#34;cuda&#34;)* 模型在 intel CPU 上运行。nvidia：引用 青龍聖者 (@bdsqlsz) @cloneofsimo device=cpu × deivce=cuda √
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1313559319685632040)** (32 messages🔥): 

> `Reward Function Design, Challenges with Stabilization, Experimentation Procedures` 


- **稳定化挑战的挣扎**：对话强调了空中启动与起飞的复杂性，成员们分享道 *在空中启动要困难得多*，而 *起飞可能有助于避免某些诡异行为*。
   - 一位成员建议通过可视化 RL rollouts 来确定 RL 方法是否有效地利用了 Simulator。
- **简化的 Reward Functions 胜出**：一位成员提出，基础的 Reward Function（例如 *最小化 Yaw, Pitch 和 Roll*）可以简化学习过程，并建议如果 Simulator 崩溃则给予巨大的负奖励。
   - 讨论中还幽默地提到要确保 RL 系统不会直接 *飞向远方（失控）*。
- **宝贵的实验日志记录技巧**：强调了实验和记录结果的重要性，并引用了一段关于实际记录实验而非依赖记忆的重要性的话。
   - 成员们分享了对以往实验的反思，指出在为未来参考提供妥善文档方面存在差距。
- **Reward Functions 的直觉性质**：讨论了设计 Reward Function 背后的直觉，共识是 *这主要靠直觉*，且更简单的方法通常会产生更好的结果。
   - 一位成员询问了关于 Reward Function 设计的资源，强调随着时间的推移需要对该主题有清晰的理解。
- **在 Wheelies 现象中观察到的独特行为**：一位成员在他们的 Simulation 中观察到一种有趣的现象，模型会执行 *wheelies*（翘头），即抬起一侧却无法抬起后半部分，导致翻车。
   - 这种 *诡异行为* 强调了正确的 Reward Functions 的重要性，以及为了获得稳定结果进行调整的必要性。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1313853849735528459)** (17 messages🔥): 

> `OLMo1 Naming Controversy, Discussion on Naming Trends, Nerdsniped Reactions` 


- **OLMo1 命名争议**：一位用户表达了他们对名称中包含 **O1** 的反感，称其很 **cringe**（尴尬）。
   - *“我们喜欢 O1Mo 哈哈”* 引发了一些关于命名规范的笑声和辩论。
- **关于替代命名路线的辩论**：展开了关于在考虑新名称时是否可能走 **Qwen 路线** 的讨论。
   - 一位成员建议命名为 **OwOlmo**，延续了这场充满趣味的命名辩论。
- **提议名称 OLMoFans**：另一位成员提到他们讨论过使用 **OlmoFans** 作为潜在名称。
   - 随着他们意识到 **O1 博客文章依然自带流量**，笑声在轻松的氛围中持续。
- **社区对命名想法的反应**：成员们评论了人们对命名话题产生的 **Nerdsniped** 反应。
   - 该讨论反映了粉丝们围绕这一主题表现出的有趣特质。


  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1313683518982983700)** (17 条消息🔥): 

> `Efficient Gram Matrix Computation, Triton for Upper Triangle, cuBLAS and cutlass for Gram matrices, HPC Interview Expectations` 


- **探索高效的 Gram Matrix 计算**：一位用户提出了如何高效计算 Gram Matrix (**A@A^T**) 的上三角部分，而不是使用标准的 matmul 后接 triu。
   - 回复建议利用 Triton 仅计算相关的 tiles，并提到 **cuBLAS 的 syrk** 和 **cutlass** 作为潜在的替代方案。
- **用于自定义 Kernel 的 Triton**：讨论了对于没有经验的人来说，在 Triton 中编写自定义 kernel 是否困难，成员们指出学习 matmul kernel 至关重要。
   - 社区成员建议，在理解 matmul 之后，支持上三角计算的修改应该是比较直接的。
- **Triton 中 Matmul Kernel 的资源**：成员们分享了帮助快速学习 Triton 中 matmul kernel 的资源，包括一个 [官方教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)。
   - 然而，有人指出这些资源可能对初学者不太友好，这可能会给新学习者带来挑战。
- **理解 Gram Matrices**：对 Gram matrix 的不同定义进行了区分，该成员确认他们特别感兴趣的是 **A@A^T**。
   - 另一位成员指出文献中存在不同的形式，表明术语上存在一些混淆。
- **GPU 编程岗位的面试预期**：一位用户询问了 HPC/Storage 团队中 GPU 编程岗位的面试预期，特别是考虑到他们背景是 leetcode 风格的面试。
   - 针对普通面试与特定 GPU 编程岗位面试之间的差异提出了担忧，特别是对于申请中级职位的应届毕业生。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLAS/Level-3/syrkx/cublas_syrkx_example.cu">CUDALibrarySamples/cuBLAS/Level-3/syrkx/cublas_syrkx_example.cu at master · NVIDIA/CUDALibrarySamples</a>: CUDA Library Samples。通过在 GitHub 上创建账号为 NVIDIA/CUDALibrarySamples 的开发做出贡献。</li><li><a href="https://github.com/Infatoshi/cuda-course/blob/master/05_Writing_your_First_Kernels/02%20Kernels/02%20matmul.cu#L42).">cuda-course/05_Writing_your_First_Kernels/02 Kernels/02 matmul.cu at master · Infatoshi/cuda-course</a>: 通过在 GitHub 上创建账号为 Infatoshi/cuda-course 的开发做出贡献。</li><li><a href="https://github.com/NVIDIA/cutlass/blob/main/examples/31_basic_syrk/basic_syrk.cu">cutlass/examples/31_basic_syrk/basic_syrk.cu at main · NVIDIA/cutlass</a>: 用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1313499429835116595)** (8 条消息🔥): 

> `Triton MLIR Dialects 文档, 结合 TMA 的 Grouped GEMM, Triton 支持渠道, 与 Stages 相关的 Kernel 崩溃, Triton Gist 问题` 


- **Triton MLIR Dialects 文档的可用性**：用户讨论了 Triton MLIR Dialects 文档的可用性，并指向 [Triton Ops 文档](https://triton-lang.org/main/dialects/TritonOps.html) 作为参考资源。
   - 还提到了一个极简的 [编程指南](https://github.com/triton-lang/triton/tree/main/docs/programming-guide)，尽管它看起来尚未完成。
- **结合 TMA 的 Grouped GEMM 面临的挑战**：一位用户询问如何在 Triton 中编写结合 TMA 的 Grouped GEMM，特别是关于传递 descriptors 张量而非地址的问题。
   - 有人提到了一个 [Pull Request](https://github.com/triton-lang/triton/pull/4498)，旨在解决这一限制，但可能无法完全支持所需的功能。
- **寻求 Triton 技能问题的支持**：成员们讨论了在哪里寻求可能与技能相关而非 Bug 的 Triton 问题支持，建议将此频道作为资源。
   - 对于严重的 Bug，建议使用 Triton Slack。
- **Kernel 崩溃与共享内存问题**：一位用户分享了一个 Gist，展示了 Kernel 的两个版本，并指出循环版本因错误而崩溃。
   - 另一位成员强调，这个崩溃问题与 stages 的数量有关，观察到只有在 `num_stages=1` 时才能成功运行。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/dialects/dialects.html">Triton MLIR Dialects and Ops &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://triton-lang.org/main/dialects/TritonOps.html">TritonOps &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/tree/main/docs/programming-guide">triton/docs/programming-guide at main · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton</li><li><a href="https://github.com/triton-lang/triton/pull/4498">[nvidia] Support passing TMA descriptors by-value by embg · Pull Request #4498 · triton-lang/triton</a>：动力：目前 Triton 通过全局内存以引用方式传递 TMA descriptors。这存在许多问题：主机到设备的 memcpy 导致显著的启动开销（5-10us），用户必须插入...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1313692632584814695)** (5 条消息): 

> `代码工作招聘, GEMM Kernel 性能, GPU 计算中的缓存行为` 


- **代码工作招聘讨论**：一位用户表达了雇人进行代码工作的兴趣，并被建议在特定频道发布以获得更好的回应。
   - 这表明这是一个以社区为中心的环境，用户可以有效地寻求帮助或招聘协助。
- **GEMM Kernel 优化中的性能瓶颈**：一位用户报告称，在 A100 上的 GEMM Kernel 中，发出 `cp.async.cg` 指令与 `cp.async.ca` 相比，性能大幅下降。
   - 他们指出，在使用 **L1 cache** 时没有遇到这些问题，这指向了 GPU 操作中缓存行为的复杂性。
- **Layout 更改改进内存访问**：同一位用户通过在内存访问期间切换特定的 layout 解决了最初的性能问题，这有助于消除 bank conflicts。
   - 他们表示，这种 layout 调整利用了 **swizzling**，确保了优化的访问模式，尽管他们最初的直觉是错误的。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1313524615611682836)** (30 条消息🔥): 

> `高效 ML 课程, Stanford 的 CS 229S 课程, CUDA vs Triton, MIT Han Lab 课程, Washington 的 CSE 599K 课程` 


- **高效 ML 课程讨论**：成员们重点推荐了几门**高效机器学习课程**，包括 MIT 的 [Efficient AI Computing](https://hanlab.mit.edu/courses/2024-fall-65940) 课程和 Stanford 的 [CS 229S - Systems for Machine Learning](https://cs229s.stanford.edu/fall2023/)。MIT 的课程涵盖了**模型压缩**、**剪枝**以及针对**资源受限设备**的优化。
   - 参与者还指出需要实际的实现资源，一些人认为某些课程偏向理论而非应用。
- **课程资源与作业**：几位成员讨论了各门课程的**作业可用性**，提到 Stanford 的 CS 229S 通过 Google Colab 提供了**实验（Labs）**，方便使用。此外，华盛顿大学的课程 [CSE 599K](https://courses.cs.washington.edu/courses/cse599k/24au/) 通过各种作业提供了对 ML 系统的**深入理解**。
   - 成员们鼓励检查预备知识和资源，以便从这些学习机会中充分受益。
- **在使用 Triton 之前熟悉 CUDA**：一位成员询问在深入学习 **Triton** 之前是否建议先熟悉 **CUDA**，并表示在编写 Kernel 时更倾向于 **CUDA** 的**直观性**。另一位成员分享了观点，认为深入专注于一种语言或框架比选择哪种框架更有益。
   - 交流强调了在理解底层 Kernel 开发与优化跨不同平台技能之间的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cs229s.stanford.edu/fall2023/">主页</a>: Systems for Machine Learning</li><li><a href="https://courses.cs.washington.edu/courses/cse599k/24au/">CSE 599K</a>: 未找到描述</li><li><a href="https://hanlab.mit.edu/courses/2024-fall-65940">MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing</a>: 未找到描述</li><li><a href="https://hanlab.mit.edu">MIT HAN Lab</a>: 欢迎来到 MIT HAN Lab，在这里效率与性能相遇，创新与卓越在人工智能 (AI) 和计算机体系结构领域交汇。我们的实验室处于最前沿...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1313786282421391420)** (7 条消息): 

> `CUDA 预备知识, Warp 调度困惑, GPU 中的 Core 定义, 混合执行单元` 


- **CUDA 主题初学者指南**：一位新成员寻求学习 CUDA 和 PMPP 书籍的预备知识建议，想知道是否需要了解 **Operating Systems**（操作系统）和 **Computer Architecture**（计算机体系结构）等学科。
   - “直接读那本书就行”是一个幽默的回应，建议直接开始学习材料。
- **关于 A100 中 Warp 调度的困惑**：一位新用户对 PMPP 书中描述的 **Warp 调度**表示困惑，特别是关于 A100 GPU 中 **2048 个线程**与 **64 个 Core** 之间的差异。
   - 他们引用了 [NVIDIA 文档](https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/)，其中指出每个分发单元（dispatch unit）可以**每个时钟周期为 32 个线程**分配指令，从而对书中的说法提出质疑。
- **澄清 A100 中的 Core 数量**：讨论围绕 A100 中 **Core** 的定义展开，一位成员澄清说 Core 指的是 GPU 上不同的执行单元，通常被称为 *pipes*。
   - 他们进一步解释了使用 64 个 Core 执行操作的复杂性，以及轻量级整数运算和重量级浮点运算的良好结合如何能实现多达 **128 个 Core** 的有效执行。



**提到的链接**: <a href="https://developer.nvidia.com/blog/nvidia-ampere-architecture-in-depth/">NVIDIA Ampere Architecture In-Depth | NVIDIA Technical Blog</a>: 今天，在 2020 年 NVIDIA GTC 主旨演讲中，NVIDIA 创始人兼 CEO 黄仁勋介绍了基于全新 NVIDIA Ampere GPU 架构的新型 NVIDIA A100 GPU。本文将带您深入了解...

  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1313502237321859084)** (33 messages🔥): 

> `Mastodon 概览，核能与 GPU，GPU 的环境影响，高效训练框架，AI 融资新闻` 


- **对 Mastodon 的好奇**：一名成员对 [Mastodon](https://letmegooglethat.com/?q=mastodon) 表示好奇，希望进一步了解该平台。
   - *Pessimistic_neko* 建议通过简单的搜索即可轻松解决此疑问。
- **核能在能源效率方面的潜力**：讨论围绕**大型科技公司**利用核能为其 GPU 运行提供动力的前景展开，并探讨了其对效率和可持续性的影响。
   - *Marksaroufim* 指出，核能的可靠性可以支持白天的能源需求，同时在夜间进行模型训练。
- **GPU 的环境影响**：一名成员强调，公众对使用 GPU 集群带来的环境影响缺乏认识，并指出这一话题并不常被讨论。
   - *Rizzware* 提到了向科技领域以外的更广泛受众传达能源影响所面临的挑战。
- **创建更智能的训练框架**：分享了关于开发训练框架的想法，该框架可以通过在电价较低的时段调度模型训练来动态优化电力成本。
   - *S1r_o* 幽默地建议系统可以根据预测的电价调整训练时间。
- **AI 领域令人兴奋的融资新闻**：一名成员宣布 **Tenstorrent** 本周筹集了 **7 亿美元**，这为近期 AI 领域的融资热潮做出了贡献。
   - 该公告包含了一个 [YouTube 视频](https://www.youtube.com/watch?v=_aqMdhAgGG8) 链接，其中 Jim Keller 讨论了 AI 对计算领域即将产生的冲击。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://letmegooglethat.com/?q=mastodon">Mastodon</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=_aqMdhAgGG8">Tenstorrent 的 Keller：我们正处于 AI 炒作周期中</a>：Tenstorrent 的 Jim Keller 预计 AI 将在未来十年主导计算领域。他加入“Bloomberg Technology”与 Caroline Hyde 进行讨论。-----...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[liger-kernel](https://discord.com/channels/1189498204333543425/1275130785933951039/)** (1 messages): 

0x000ff4: 好的，我已经更新了关于 kto loss 的 PR
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1313573550548058133)** (3 messages): 

> `KernelBench，GPU kernel 评估，排行榜问题` 


- **KernelBench 发布，包含令人兴奋的功能**：介绍 🌽 [KernelBench](https://twitter.com/anneouyang/status/1864014135824162995) (预览版)，这是一个新的编程基准测试，旨在评估 LLM 生成**高效** GPU kernel 以优化神经网络性能的能力。
   - 该工具旨在增强 GPU 计算领域的基准测试实践。
- **排行榜上的 Kernel 性能疑虑**：一位用户注意到排行榜上一些**最快的 kernel** 似乎并不完整，凸显了性能评估中潜在的问题。
   - 他们提供了一个被认为存在争议的特定 kernel 解决方案链接：[incomplete kernel](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py)。
- **KernelBench GitHub 仓库已上线**：可以在 [GitHub 页面](https://github.com/ScalingIntelligence/KernelBench)上进一步探索 KernelBench 的开发，欢迎贡献和协作。
   - 该平台允许用户参与基准测试工具的持续开发和测试。



**提到的链接**：<a href="https://github.com/ScalingIntelligence/KernelBench">GitHub - ScalingIntelligence/KernelBench</a>：通过在 GitHub 上创建账户来为 ScalingIntelligence/KernelBench 的开发做出贡献。

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1313598091684745227)** (1 条消息): 

> `TK 的 WGMMA+tma 中的竞态条件 (Race Condition)，自定义 Kernel 实现，矩阵掩码技术，Shared Memory 利用，CUDA 版本兼容性` 


- **自定义 Kernel 中掩码导致的竞态条件**：一位用户报告在使用 **TK 的 WGMMA+tma** 实现自定义 Kernel 时遇到了 **竞态条件 (race condition)**，原因是 K 维度的对齐问题。
   - 用户发现他们的掩码技术除非调用 **10 次** 否则无法保持一致，这引发了对线程同步 (thread synchronization) 的担忧。
- **矩阵操作的创新掩码技术**：他们开发了一种基于 `load` 的新 **掩码函数**，通过将零加载到 Shared Memory 中来处理越界行。
   - 然而，尽管有此创新，**memcheck/synccheck/initcheck** 均未报告错误，这使得调试过程变得复杂。
- **探索 Shared Memory 问题**：该实现对 Shared Memory 和 barriers 的依赖促使用户考虑代码库近期 **重构 (refactoring)** 可能产生的影响。
   - 考虑到所使用的 fork 版本的历史，他们思考更新到 **最新版本** 是否能解决一些集成问题。
- **CUDA 兼容性疑虑**：讨论中提到用户在 **CUDA 12.5** 环境下使用 **bf16s**，这让他们质疑当前设置的兼容性。
   - 他们表达了对 **ThunderKittens** 的钦佩，承认尽管存在技术障碍，但与其他替代方案相比，它非常易于使用。


  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1313507390288695296)** (88 条消息🔥🔥): 

> `Discord 中的诈骗和机器人，Stable Diffusion 入门，使用 ComfyUI 进行 AI 艺术生成，Stable Diffusion 和 LoRA 故障排除，SD 性能分析工具` 


- **诈骗和机器人困扰 Discord 社区**：据观察，社区中出现了多个 **机器人**，试图进行 **庞氏骗局 (Ponzi schemes)** 或冒充 **Discord support**。
   - *用户的建议是将这些机器人报告给 Discord 并避免与其互动。*
- **Stable Diffusion 入门及工具**：一位新手寻求进入 **Stable Diffusion** 领域的建议，对工具和模型表示困惑，并对诈骗保持警惕。
   - 用户推荐使用 **Vast.ai** 租用云端 GPU，并建议从 YouTube 上 Scott 的 **ComfyUI** 教程开始，以获得更好的工作流。
- **ComfyUI：高级 AI 艺术工作流的理想选择**：一位用户推荐 **ComfyUI** 作为创建 AI 艺术的平台，特别是对于初学者，并强调了观看入门视频的重要性。
   - 强调了拥有高性能 GPU 进行本地 AI 工作的重要性，一些用户讨论了云端选项作为一种具有成本效益的解决方案。
- **Stable Diffusion 问题排查**：几位用户报告了 **LoRA 模型** 的问题，意识到在 Prompt 中使用触发词 (trigger words) 对其正常运行是必要的。
   - 其他人遇到了图像结果变得混乱的问题，将其归因于 **Stable Diffusion** 中的各种设置。
- **需要更好的性能分析工具**：一位用户表示有兴趣为 **Stable Diffusion** 社区贡献性能分析工具，并指出目前缺乏此类资源。
   - 其他人表示赞同，认为 **SD 生态系统** 需要增强性能分析以改善用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://nvlabs.github.io/Sana/">Sana</a>: 未找到描述</li><li><a href="https://vast.ai/">租用 GPU | Vast.ai</a>: 通过最佳的云端 GPU 租用服务，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面允许对所有供应商的 GPU 租用进行公平比较。</li><li><a href="https://dontasktoask.com/">别问怎么问，直接问</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo&list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x">ComfyUI - 入门：第 1 集 - 在 Stable Diffusion AI 艺术生成方面优于 AUTO1111</a>: 今天我们介绍如何使用 ComfyUI 通过 Stable Diffusion 模型创建 AI 艺术的基础知识。这个基于节点的编辑器是一个理想的工作流工具...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1313548920646930506)** (72 条消息🔥🔥): 

> `Amazon Nova 模型，AWS 公告，PydanticAI，OpenAI 的 12 天活动，Google 的 Genie 2`

- **Amazon Nova 模型发布**：Amazon 在 AWS re:Invent 上推出了其全新的基础模型系列 Nova，旨在与 GPT-4 等顶尖模型竞争。此次发布包括在 Amazon Bedrock 上提供的多个文本和视频生成模型。
   - 社区反馈仍在不断涌现，初步观点将其性能与 OpenAI 的产品进行了对比。
- **AWS 发布新 API**：AWS 发布了多项 API 更新，包括一个 Usage API，允许开发者以编程方式跟踪使用情况和成本。监控功能包括按时间统计的 Token 使用量以及按各种标识符进行的过滤。
   - 该功能旨在提高开发者使用 AWS 服务时的透明度和管理效率。
- **PydanticAI 框架发布**：Pydantic 推出了 PydanticAI，旨在简化由 LLM 驱动的应用开发，强调类型安全和模块化。该工具目前处于 Beta 阶段，并在 MIT License 下开源。
   - 该框架被定位为开发者在项目中利用 LLM 的一个易于上手的选择。
- **OpenAI 的 12 天发布活动**：OpenAI 开启了为期 12 天的活动，从 12 月 5 日开始，每天进行发布、演示和更新。分享的初步数据包括 ChatGPT 每周活跃用户达 3 亿，平台上每天发送 10 亿条消息。
   - 围绕重大发布的期待正在升温，其中包括一个潜在的文本转视频 AI 工具。
- **Google 推出 Genie 2**：Google 发布了 Genie 2，这是一种专为视频生成和交互式环境设计的自回归潜扩散模型（autoregressive latent diffusion model）。它利用了 Transformer 动力学模型，旨在增强生成内容中的动作可控性。
   - 社区讨论重点关注该模型的输出长度和实用性，特别是在生成的视频方面。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.theverge.com/2024/12/4/24312352/openai-sora-o1-reasoning-12-days-shipmas">OpenAI 的 12 天 “shipmas” 包括 Sora 和新的推理模型</a>：OpenAI 计划了 12 天的圣诞活动。</li><li><a href="https://x.com/sama/status/1864335461268754712?s=46">Sam Altman (@sama) 的推文</a>：🎄🎅 从明天太平洋时间上午 10 点开始，我们将进行为期 12 天的 OpenAI 活动。每个工作日，我们都会进行一场包含发布或演示的直播，其中有一些重大发布，也有一些圣诞袜小礼物。我们准备了一些很棒的...</li><li><a href="https://x.com/hello__caitlin/status/1864367028758565216?s=46">c a i t l i n (@hello__caitlin) 的推文</a>：我从今年的 Spotify Wrapped 中学到了两件事：1. 我可能应该取消我的 Spotify premium 2. 他们的年终指标 100% 是编造的。</li><li><a href="https://nvlabs.github.io/Sana/">Sana</a>：未找到描述</li><li><a href="https://x.com/reach_vb/status/1863956316634403260?s=46">Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：smol 课程 - 学习指令微调（instruction tuning）、模型评估、合成数据集、推理（inference）等！！🔥 100% 免费且完全开源，与社区一起学习，在年底大放异彩！💥</li><li><a href="https://x.com/mrdbourke/status/1863870479167279486?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Daniel Bourke (@mrdbourke) 的推文</a>：新视频：使用 Google Gemini 通过视频追踪我家的每件物品 🎥 -> 🛋️ 我称之为 "KeepTrack" 😎 输入：10 分钟的随手拍视频。输出：包含 70 多件物品的结构化数据库...</li><li><a href="https://www.interconnects.ai/p/openais-o1-using-search-was-a-psyop">OpenAI 的 o1 使用 “搜索” 是一场心理战（PSYOP）</a>：如何理解 OpenAI 的 o1 模型，它实际上只是一个古怪、奇妙且漫长的思维链（chain of thought）</li><li><a href="https://x.com/iScienceLuvr/status/1864217903232385348">Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：顶尖计算机视觉研究员 Lucas Beyer (@giffmana)、Alexander Kolesnikov (@__kolesnikov__)、Xiaohua Zhai (@XiaohuaZhai) 已离开 Google DeepMind 加入 OpenAI！他们是近期 SOTA v... 的幕后推手。</li><li><a href="https://ndurner.github.io/amazon-nova">Amazon Nova 基础模型发布</a>：鉴于社区对如何设置 AWS 以使用新的 Amazon Nova 模型很感兴趣，这里有一份分步指南供大家参考：</li><li><a href="https://x.com/exaailabs/status/1864013080944062567?s=46">Exa (@ExaAILabs) 的推文</a>：宣布推出 Exa Websets - 迈向完美网络搜索的突破。在下方注册候补名单👇</li><li><a href="https://x.com/ExaAILabs/status/1806444570210934949">Exa (@ExaAILabs) 的推文</a>：Exa 如何提供十亿级规模的向量搜索？我们将二进制量化（binary quantization）、俄罗斯套娃嵌入（Matryoshka embeddings）、SIMD 和 IVF 结合到一个可以击败 HNSW 等替代方案的新系统中。@shreyas4_ 今天做了一个演讲...</li><li><a href="https://x.com/lmarena_ai/status/1864062852589605156?s=46">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：祝贺 @amazon 发布了最新的前沿模型 Nova！⭐ Nova 在标准基准测试中与 GPT-4o 等顶级模型具有竞争力。现在，真正的挑战开始了——Nova 已进入 Arena 进行人类评估...</li><li><a href="https://www.interconnects.ai/?r=1h4isl&utm_campaign=referrals-subscribe-page-share-screen&utm_medium=web">Interconnects | Nathan Lambert | Substack</a>：连接 AI 的重要思想。高层思考与技术思考的边界。每周三早上供顶尖工程师、研究员和投资者阅读。点击阅读 Nathan 的 Interconnects...</li><li><a href="https://x.com/openainewsroom/status/1864373399218475440?s=46">OpenAI Newsroom (@OpenAINewsroom) 的推文</a>：@sama 今天早些时候分享的最新数据：3 亿 ChatGPT 周活跃用户；ChatGPT 每天发送 10 亿条用户消息；美国有 130 万开发者基于 OpenAI 进行构建。</li><li><a href="https://x.com/openainewsr">FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">介绍 Amazon Nova，我们的新一代基础模型</a>：来自 Amazon 的全新 SOTA 基础模型，提供前沿智能和行业领先的性价比。</li><li><a href="https://x.com/chipro/status/1864384749911065035">Chip Huyen (@chipro) 的推文</a>：完成了！150,000 字，200 多张插图，250 个脚注，以及超过 1200 个参考链接。我的编辑刚刚告诉我，手稿已经送往印刷厂。- 电子书将于今年晚些时候推出...</li><li><a href="https://bsky.app/profile/jparkerholder.bsky.social/post/3lcijlzafhs2b">Jack Parker-Holder (@jparkerholder.bsky.social)</a>：介绍 🧞Genie 2 🧞 - 我们最强大的大规模基础世界模型（world model），它可以生成各种连贯的世界，可玩时长达一分钟。我们相信 Genie 2 可以开启...</li><li><a href="https://x.com/openaidevs/status/1864369714925064606?s=46">openaidevs 的推文</a>：

<li><a href="https://x.com/OpenAIDevs/status/1864011400219148560">OpenAI Developers (@OpenAIDevs)</a>: 🆕 Usage API——以编程方式跟踪 API 使用情况和成本。👀 按分钟/小时/天监控 Token 使用情况 🔎 按 API key、项目 ID、用户 ID、模型等过滤使用情况 💹 通过 Costs 检查每日支出...</li><li><a href="https://x.com/skirano/status/1864014133756129752">来自 Pietro Schirano (@skirano) 的推文</a>: 我添加了一个新的 MCP server，让 Claude 在回答之前可以逐步思考。Claude 能够预先决定需要多少思考步骤，追溯其想法，甚至在看到...时分叉。</li><li><a href="https://bsky.app/profile/m--ric.bsky.social/post/3lcifklp5wc2b">@m--ric.bsky.social</a>: 𝗦𝗵𝗼𝘄𝗨𝗜：一个可以导航任何 𝗨𝗜 📲 且击败了更大规模 VLMs 的小型端到端 Agent！来自 NUS & Microsoft 的新论文，可在任何 UI（桌面...）上运行的 Agent。</li><li><a href="https://x.com/dylan522p/status/1864089972644749722">来自 Dylan Patel (@dylan522p) 的推文</a>: 亚马逊为 Anthropic 提供的 400k Trainium 2 集群。谁说 Scaling Law 已死？查看下方的完整架构、服务器和软件详情。引用 Dylan Patel (@dylan522p) 亚马逊的 AI 自给自足...</li><li><a href="https://x.com/giffmana/status/1864214549076844556">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: 哟，刚醒！发生什么了？为什么他们喜欢这条随机帖子？总之，正和孩子吃早餐，稍后和大家聊！引用 Lucas Beyer (bl16) (@giffmana) 好了，大家...</li><li><a href="https://x.com/openai/status/1864328928267259941?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 OpenAI (@OpenAI) 的推文</a>: 12 天。12 场直播。一堆大大小小的新东西。OpenAI 的 12 天活动明天开始。</li><li><a href="https://x.com/_philschmid/status/1864016010464080260?s=46">来自 Philipp Schmid (@_philschmid) 的推文</a>: 意料之外。@amazon 带着 Foundation Models 回归了。作为 re:Invent 的一部分，他们发布了 6 个新的 Foundation Models，涵盖从纯文本到 text-to-video！👀 Nova 模型将通过 Am... 独家提供。</li><li><a href="https://news.ycombinator.com/item?id=39509937">Genie: 生成式交互环境 | Hacker News</a>: 未找到描述</li><li><a href="https://x.com/lukeharries_/status/1864017453358932448?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Luke Harries (@LukeHarries_) 的推文</a>: 构建可以说话的 AI agents 过去需要数月时间。在 @elevenlabsio，我们亲眼见证了这一点。我们与数百家初创公司合作，他们都花了 3-6 个月构建相同的 Conversational AI 技术栈：- Speech...</li><li><a href="https://x.com/swyx/status/1864137540518990281">来自 swyx 🔜 @NeurIPSConf x Latent.Space (@swyx) 的推文</a>: 对 Amazon Nova 印象深刻：6 个新的内部模型，可与 Gemini/Llama/ Dalle3/SD3.5/ @runwayml 视频竞争，并正在开发语音/全能模态（omnimodal）。剪辑了整个 @AWSCloud Re:invent 主旨演讲...</li><li><a href="https://venturebeat.com/programming-development/python-data-validator-pydantic-launches-model-agnostic-ai-agent-development-platform/">Python 数据验证器 Pydantic 发布模型无关的 AI agent 开发平台</a>: 一个新的 Agent 框架，旨在简化由 LLM 驱动的生产级应用程序的开发</li><li><a href="https://venturebeat.com/programming-development/python-data-validator-pydantic-launch">Python 数据验证器 Pydantic 发布模型无关的 AI agent 开发平台</a>: 一个新的 Agent 框架，旨在简化由 LLM 驱动的生产级应用程序的开发</li><li><a href="https://x.com/sama/status/1864335461268754712">来自 Sam Altman (@sama) 的推文</a>: 🎄🎅 从明天太平洋时间上午 10 点开始，我们将进行 OpenAI 的 12 天活动。每个工作日，我们都会进行一场包含发布或演示的直播，有一些大动作，也有一些小礼物。我们准备了一些很棒的东西...</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking">modelcontextprotocol/servers 仓库中的 servers/src/sequentialthinking</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card">Amazon Nova 系列模型：技术报告和模型卡片</a>: 我们介绍了 Amazon Nova，新一代最先进的 Foundation Models，提供前沿智能和行业领先的性价比。Amazon Nova Pro 是一款功能强大的 multimodal...</li><li><a href="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2：大规模基础世界模型 (foundation world model)</a>: 为未来的通用 Agent 生成无限多样的训练环境</li><li><a href="https://wattenberger.com/thoughts/fish-eye">LLM 是思考的工具</a>: 未找到描述</li><li><a href="https://x.com/Wattenberger/status/1863977304126603309">来自 Amelia Wattenberger 🪷 (@Wattenberger) 的推文</a>: 🐟 关于...的一些沉思 </li>

我们可能会使用 LLMs🐠 以受鱼眼镜头启发的多级抽象方式与文本进行交互。</li><li><a href="https://youtu.be/v-EYzZCLF48?si=6zA8LCMxk3VQDXWw">介绍 ElevenLabs Conversational AI</a>：Conversational AI 已至。在几分钟内构建能够说话的 AI agents，具有低延迟、完全可配置性和无缝扩展性。让我们来处理语音...</li><li><a href="https://techcrunch.com/2024/12/03/amazon-announces-nova-a-new-family-of-multimodal-ai-models/">亚马逊发布 Nova，一个新的多模态 AI 模型家族 | TechCrunch</a>：在其 re:Invent 2024 大会上，亚马逊云计算部门 Amazon Web Services (AWS) 宣布了一个名为 Nova 的新 AI 模型家族。</li><li><a href="https://youtu.be/LY7m5LQliAo?si=gHqvXgAz6Bv9fZIB&">AWS re:Invent 2024 - Matt Garman 的 CEO 主旨演讲</a>：AWS CEO Matt Garman 讲述了 AWS 如何在世界领先的云服务的各个方面进行创新。探索 AWS 如何重塑基础构建...</li><li><a href="https://youtu.be/LY7m5LQliAo?si=gHqvXgAz6Bv9fZIB&t=6657">AWS re:Invent 2024 - Matt Garman 的 CEO 主旨演讲</a>：AWS CEO Matt Garman 讲述了 AWS 如何在世界领先的云服务的各个方面进行创新。探索 AWS 如何重塑基础构建...</li><li><a href="https://buttondown.com/ainews/archive/ainews-olympus-has-dropped-aka-amazon-nova/">[AINews] Olympus 已发布（又名 Amazon Nova Micro|Lite|Pro|Premier|Canvas|Reel）</a>：Amazon Bedrock 就是你所需要的一切吗？2024/12/2-2024/12/3 的 AI 新闻。我们检查了 7 个 subreddits、433 个 Twitter 和 29 个 Discord（198 个频道和 2914 条消息）以获取...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/)** (1 条消息): 

swyxio：宣布了下周的重量级论文俱乐部 https://x.com/swyx/status/1864423257266639166
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1313574753709326428)** (51 条消息🔥): 

> `LM Studio 下载问题、Windows 性能问题、LLM 的 RPG 实验、Chat API 功能、本地网络 GPU 使用` 


- **Windows x86 版本的下载问题**：一位用户报告无法从 [lmstudio.ai](https://lmstudio.ai) 下载 Windows x86 版本，收到文件不可用的消息。
   - 其他用户建议可能是 CDN 问题，并建议使用 VPN 再次尝试下载。
- **LM Studio 在 Windows 上的性能下降**：一位成员在 Windows 上运行 LM Studio 时遇到了明显的性能问题（与 Mac 相比），且模型输出了意外字符。
   - 建议包括切换 `Flash Attention` 开关并检查系统规格以排查问题。
- **尝试将 LLM 用作 RPG 游戏主持人**：一位用户分享了使用 LLM 进行预先计划的 RPG 冒险的经验，强调了用泰语编写大纲以避免预知的新颖性。
   - 实验产生了引人入胜的结果，引发了对 AI RPG 玩家的方法论和社区资源的讨论兴趣。
- **Chat API 功能和特性**：一位用户询问了如何在 API 调用中使用 RAG 功能，并表达了希望在 API 模式中看到输入可见性的愿望。
   - 讨论显示在 API 使用中需要为文件附件和系统提示进行自定义编码，并建议将性能与现有解决方案进行比较。
- **在本地网络 GPU 上使用 LM Studio**：一位用户询问是否可以从笔记本电脑连接到具有多个 GPU 的本地服务器来运行 LM Studio。
   - 另一位成员确认这是可能的，但需要一个前端来实现正常功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tlkh/asitop">GitHub - tlkh/asitop: 适用于 Apple Silicon 的性能监控 CLI 工具</a>：适用于 Apple Silicon 的性能监控 CLI 工具。通过在 GitHub 上创建账号为 tlkh/asitop 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/cli/log-stream">lms log stream - CLI | LM Studio 文档</a>：从 LM Studio 流式传输日志。对于调试发送到模型的提示词非常有用。</li><li><a href="https://lmstudio.ai/docs/api/rest-api">LM Studio REST API (beta) - API | LM Studio 文档</a>：REST API 包括增强的统计数据，如每秒 Token 数和首个 Token 时间 (TTFT)，以及关于模型的丰富信息，如已加载与未加载、最大上下文、量化等。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1313516118757081199)** (13 条消息🔥): 

> `Arc Battlemage 显卡, 在 iGPU 上运行 LMS, 为写作助手选择模型, 双 3090 的 PCIe 配置` 


- **Intel 的 Arc Battlemage 显卡面临质疑**：一些用户对新款 **Arc Battlemage 显卡** 表示担忧，认为由于驱动支持不佳，**Intel GPU** 在 AI 任务中可能不够可靠。
   - *有评论指出，优先选择数量更少但显存更大的 GPU（如 3090）会更好*。
- **强制 LMS 在 iGPU 上运行**：一位用户询问如何强制 **LMS** 在 **iGPU** 而非 **dGPU** 上运行，并提到在选择 Vulkan 运行时后缺少相关选项。
   - *回复指出，调整 CUDA 可见设备（CUDA visible devices）是目前选择 LMS 使用哪个 GPU 的方法*。
- **为笔记摘要选择合适的模型**：一位成员就如何选择适合其电脑配置（包含 **4070Ti Super**）的 **笔记摘要** 模型寻求建议。
   - *其他人建议确保以 GB 为单位的模型大小符合可用 VRAM，并留出足够的余量以保证性能。*
- **双 3090 的 PCIe 配置影响**：一位用户询问由于空间限制，通过转接线在 PCIe 4.0 x8 上使用第二块 **3090** 是否会带来潜在的性能损失。
   - *已确认虽然副卡可以工作，但在 Windows 上将模型拆分到两块显卡可能会导致性能问题。*


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1313561971517558826)** (9 条消息🔥): 

> `在 Vercel 上构建 AI 应用, 智能法律文档导航, Amazon Nova 基础模型, 连接 Google Cloud 的 AI Agent, 使用 LlamaIndex 实现超快速 RAG` 


- **在 Vercel 上构建 AI 应用变得更加简单**：[LlamaIndex 的最新更新](https://twitter.com/llama_index/status/1864002184138170677) 简化了在 Vercel 上的 AI 应用开发，增强了与 LlamaCloud 的集成能力。
   - 这一进展可以提高开发者的生产力，并简化 AI 应用的部署流程。
- **像专业人士一样导航法律文档**：一篇文章展示了如何使用先进的多图（multi-graph）和多 Agent 技术构建 *智能法律文档导航系统*，[链接在此](https://twitter.com/llama_index/status/1864037791019188331)。
   - 它详细介绍了如何创建文档层级，并为法律文档实现智能遍历工作流。
- **Amazon 发布具有竞争力的 Nova 模型**：Amazon 的新基础模型系列 **Nova** 与竞争对手相比，拥有极具竞争力的基准测试结果和更具吸引力的价格；通过 `pip install llama-index-llms-bedrock-converse` 安装以确保支持，[链接在此](https://twitter.com/llama_index/status/1864080917029085459)。
   - 这些基础模型旨在为 AI 模型领域的用户提供一种高性价比且性能驱动的替代方案。
- **使用 LlamaIndex 将 AI Agent 连接到 Google Cloud**：LlamaIndex 为 Google Cloud 的 AlloyDB 和 Cloud SQL for PostgreSQL 推出了新的开源集成，使 AI Agent 的开发变得无缝，[来源](https://twitter.com/llama_index/status/1864364299063578964)。
   - 这一举措允许开发者有效地利用云数据库来增强 AI 功能。
- **使用 LlamaIndex Workflows 快速实现 RAG**：学习使用 LlamaIndex Workflows 构建高性能的检索增强生成（RAG）系统，其特点是采用事件驱动架构，[详情在此](https://twitter.com/llama_index/status/1864377849295327365)。
   - 该指南将此方法与 LangGraph 等其他框架进行了比较，强调了在复杂 AI 场景下的效率。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1313509318208913488)** (53 条消息🔥): 

> `Summary Index 性能、在 Workflows 中使用聊天历史、AI 社区协作、LLM 提示词优化、BM25Retriever 中的错误处理` 


- **Summary Index 性能问题**：一位用户提出了使用 **sentencesplitter** 的 **summaryindex** 响应速度慢的问题，称生成摘要需要约 **2 分钟**，而 ChatGPT 仅需 **8 秒**。
   - 他们探索了潜在的改进方案，但也承认使用路由和索引方法会引入延迟。
- **Workflows 简化聊天会话**：一名成员询问了在 **workflows** 中管理聊天历史的问题，特别是关于在步骤之间轻松传递消息的选项。
   - 建议包括使用 **Context** 功能和 **ChatMemoryBuffer** 进行消息管理。
- **不断增长的社区合作伙伴关系**：一名成员表示有兴趣与 **AIVisuals** 社区合作，并索要 LlamaIndex 社区的描述和加入链接。
   - 这表明了扩大合作伙伴关系以增强社区资源的潜力。
- **为 LLM 优化提示词**：一位在使用 OpenAI LLMs 时遇到幻觉问题的用户被建议尝试 **prompt optimization**（提示词优化）以提高响应准确性。
   - 建议指出，精心编写更好的指令可以提升语言模型的性能。
- **排查 BM25Retriever 错误**：一位用户报告了在使用 **BM25Retriever** 时出现 **ValueError**，提示必须准确传递 index、nodes 或 docstore 中的一个。
   - 这突显了在 LlamaIndex 库中配置检索器时面临的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/i8bow7sr">Voice &amp; Video AI Agents Hackathon · Luma</a>：Gen AI AgentsCreatorsCorner，与 AWS、Temporal、Modal、Tandem、Marly、Retell、Senso、Unified、Speedlegal、Corval、Simli、PolyAPI 等合作……</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#working-with-global-contextstate">Workflows - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1313518093439336488)** (15 条消息🔥): 

> `Rerank 3.5 多语言支持、Google Gemini 功能、Cohere Toolkit 错误、R+ 词汇使用观察、通用 AI 偏好` 


- **Rerank 3.5 支持多语言功能**：成员们确认可以切换到 **Rerank 3.5**，尽管最初的文档暗示其功能有限，但它实际上支持 **multilingual**（多语言）和 **English**（英语）排序。
   - 正如一位成员指出的，“文档指出 Rerank 3.5 允许对英语进行重排序。”
- **需要对 Google Gemini 进行澄清**：一位成员询问了关于 **Google Gemini** 的解释，指出存在无法一致访问 Google Drive 文档等问题。
   - 回复建议去 Reddit 等其他论坛寻求帮助，因为在当前频道内解决此问题的手段有限。
- **Cohere Toolkit 遇到错误**：一位用户报告了运行 **cohere-toolkit** 时的警告，特别是与 *alembic* 相关以及缺乏对 PyTorch 和 TensorFlow 等库的支持。
   - 他们提到其 **PyTorch 版本**为 **2.5.1**，并询问是否有人知道如何修复这些问题。
- **注意到 R+ 异常的词汇重复**：一位成员强调，即使在较高的 temperature 设置下，“section”一词也频繁出现在 **R+** 的响应中。
   - 观察发现这种奇特行为每六或七个响应就会出现一次，引发了对生成模式的质疑。
- **关于 AI 的通用偏好讨论**：一位用户发起了一个轻松的话题，询问人们最喜欢哪种 **AI**，以促进社区互动。
   - 回复中充满了趣味性，但没有详细阐述具体的偏好。


  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1313552078953648159)** (1 messages): 

> `Rerank 3.5, 模型弃用, 多语言性能, 增强的推理能力` 


- **Rerank 3.5 发布，具备 SOTA 性能**: Cohere 宣布发布 **Rerank 3.5**，在处理复杂用户查询方面提供 SOTA 性能，并增强了推理能力。
   - *查看我们的 [博客文章](https://cohere.com/blog/rerank-3pt5) 了解完整详情*，其中强调了对多样化数据类型和语言的改进兼容性。
- **改进了 100 多种语言的多语言能力**: **Rerank 3.5** 在包括阿拉伯语、法语、日语和韩语在内的 **100 多种语言** 中表现出更强的性能，从而在多语言环境中实现更好的搜索。
   - 这一增强功能使用户能够更高效地从电子邮件和报告等长文档中提取相关信息，满足全球应用需求。
- **现已提供模型弃用文档**: Cohere 提供了关于 **模型弃用 (model deprecations)** 的更新，详细说明了模型的生命周期阶段，包括 **Active**、**Legacy** 和 **Deprecated**。
   - 开发者可以参考 [此文档](https://docs.cohere.com/docs/deprecations) 获取任何已弃用端点和模型的推荐替代方案。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/rerank-3pt5">Introducing Rerank 3.5: Precise AI Search</a>: Rerank 3.5 提供改进的推理和多语言能力，以更高的准确度搜索复杂的企业数据。</li><li><a href="https://docs.cohere.com/docs/deprecations">Deprecations — Cohere</a>: 了解 Cohere 的弃用政策和推荐的替代方案。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1313752063402053704)** (6 messages): 

> `API Key 类型, ReRanker 性能问题, Cohere 团队访问, 模型共享` 


- **API Key 类型说明**: Cohere 提供两种类型的 API Key：**trial**（限制使用）和 **production**（限制较少）。用户可以在 [API keys 页面](https://dashboard.cohere.com/api-keys) **创建**这些 Key，并查看各个端点的 [rate limits](https://docs.cohere.com/v2/docs/rate-limits)。
   - 有关定价的更多信息，用户可以参考 [定价文档](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work)。
- **注意到 ReRanker 性能下降**: 一位用户报告称，自昨天发生更改以来，'rerank-multilingual-v3.0' 模型的 **性能下降了 30%**。新的 **rerank 3.5** 模型表现甚至更差，这引起了关注。
   - Cohere 的 **支持团队** 已确认该问题，并将协助排除故障。
- **访问 Cohere 团队**: 一位用户在受邀后仍难以在 Cohere 平台内切换团队。建议他们联系 **团队管理员** 以确保已被正确添加。
   - 如果添加后问题仍然存在，鼓励用户通过 **support@cohere.com** 联系支持部门。
- **模型共享协作**: 明确了如果用户使用相同的微调模型，可以与同事共享模型 Key。Cohere API Key 授予对这些模型的访问权限，从而允许协作。
   - 有关可用模型的更多详细信息，用户可以查看 [Cohere 模型文档](https://docs.cohere.com/v2/docs/models)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/v2/docs/rate-limits">API Keys and Rate Limits — Cohere</a>: 此页面描述了 Cohere API 对 production 和 evaluation key 的速率限制。</li><li><a href="https://docs.cohere.com/v2/docs/models">Models Overview — Cohere</a>: Cohere 拥有多种涵盖不同用例的模型。如果您需要更多定制化，可以训练模型以针对您的特定用例进行微调。
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1313738460632449025)** (2 messages): 

> `V3.5 发布, Fine-Tuning API, 基础模型更新` 


- **V3.5 发布不包括 Fine-Tuning API 更新**: 一位用户询问 Fine-Tuning API 是否随着 **v3.5** 的发布而同步更新。
   - 另一位成员回答说，目前 **rerank 3.5** 尚未通过 Fine-Tuning API 提供，基础模型仍保持在 **2.0**。
- **基础模型状态未定义**: 针对查询，明确了 Fine-Tuning API 尚未采用 **v3.5**，并继续使用基础模型 **2.0**。
   - 这意味着用户在使用 Fine-Tuning API 时将无法访问 **v3.5** 的更新或功能。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1313518121478520904)** (1 messages): 

> `Harmony project, LLM matching competition, Data harmonisation, Natural Language Processing, Discord Community` 


- **Harmony 项目助力问卷协调**：**Harmony 项目** 专注于回顾性地协调问卷项目和元数据，可在 [Harmony Data](https://harmonydata.ac.uk/) 访问。它利用 **Natural Language Processing** 协助研究人员跨不同研究和语言比较问卷。
   - 对于感兴趣的人，这里有[比较问卷项目](https://harmonydata.ac.uk/compare-harmonise-instruments/gad-7-vs-beck-anxiety-inventory/)的方法，并确保同一问卷不同版本之间的兼容性。
- **LLM 匹配算法竞赛**：Harmony 项目正在 [DOXA AI](https://harmonydata.ac.uk/doxa/) 上举办一场竞赛，以增强其 **LLM** 匹配算法，奖金高达 **£500** 的代金券。参与者可以注册并针对心理健康数据微调他们自己的语言模型。
   - 参加竞赛不需要具备 **LLM** 的先验经验，并且有机会通过 Harmony 的 Discord 服务器进行交流，特别是在 🏅「matching-challenge」频道。
- **评估 Harmony 的算法性能**：一篇博客文章评估了 Harmony 的匹配算法，指出其偶尔会在心理学家感知的句子相似度上出现误判，这可能导致问卷比较中的差异。性能指标和见解可以在[评估博客文章](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/)中找到。
   - 此次评估提出了关于提高算法准确性和增强用户对工具功能信任的重要观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://harmonydata.ac.uk/">Harmony | 全球上下文数据协调平台</a>: 一个全球性的上下文数据协调平台</li><li><a href="https://harmonydata.ac.uk/doxa/">在 DOXA AI 上为 Harmony 训练大语言模型的竞赛 | Harmony</a>: 一个全球性的上下文数据协调平台
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1313568775341080648)** (3 messages): 

> `Pydantic AI, DSLModel development, AI Development Live Demo` 


- **Pydantic AI 与 DSLModel 无缝集成**：在 DSLModel 中加入 [Pydantic AI](https://ai.pydantic.dev/) 提供了一个 **Agent** 框架，通过 Pydantic 的强大功能增强了 **LLM** 的可用性。
   - 一位成员对 **Pydantic** 在与 FastAPI 等框架配合使用时，如何促进 AI 项目中更符合人体工程学的开发表示兴奋。
- **开始使用 DSLModel**：开发者可以通过 pip 使用命令 `pip install dslmodel` 轻松安装 DSLModel，开始利用其功能。
   - 该项目在名为 [Welcome to DSLModel](https://www.loom.com/share/67dd1db910ae424eb89e249e676bbaf0) 的介绍视频中进行了进一步讨论。
- **AI 开发直播演示活动**：一场名为 **Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive** 的直播演示活动将探索 AI 开发中的尖端技术。
   - 该活动可以在 [YouTube](https://youtube.com/live/mBQFKo8bPBI) 上观看，旨在揭示在项目中使用 PydanticAI 及相关工具的创新方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtube.com/live/mBQFKo8bPBI">精通 AI 开发：PydanticAI + DSPy + DSLModel 深度解析 (直播演示)</a>: https://ai.pydantic.dev/https://dspy.ai/https://pypi.org/project/dslmodel/🚀 加入我们的直播，探索 AI 开发的前沿！发现如何...</li><li><a href="https://ai.pydantic.dev/">简介</a>: 用于将 Pydantic 与 LLM 结合使用的 Agent 框架 / 适配层</li><li><a href="https://pypi.org/project/dslmodel/">dslmodel</a>: 基于 prompt 和 Jinja 的 Pydantic + DSPy 实例。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1313549828353032222)** (19 条消息🔥): 

> `AWS Lambda 上的 DSPy 优化、ProgramOfThought 弃用、多分类任务中的精度评估` 


- **DSPy 优化面临 AWS Lambda 的 15 分钟限制**：成员们讨论了在 **AWS Lambda** 上运行 **DSPy 优化** 的挑战，特别是长运行任务的 **15 分钟执行限制**。
   - 一位用户建议利用 **/tmp 文件夹** 进行缓存，以缓解 Lambda 只读文件系统带来的速度问题。
- **ProgramOfThought 将在 v2.6 版本中重构**：针对 **v2.5** 之后 **ProgramOfThought** 的支持状态引发了关注，成员们指出它将在预计今年发布的 **v2.6** 中进行重构。
   - 建议用户在升级临近时谨慎使用当前版本。
- **类别不平衡下的精度评估方法**：一位成员询问如何在存在显著类别不平衡的 **多分类问题** 中，构建用于评估特定类别 **precision** 的指标。
   - 其他人建议使用 **dspy.Example(batch=[...])** 来处理评估，但也承认由于 **类别不平衡** 带来的困难。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1313553912363749387)** (2 条消息): 

> `Sierra AI 信息说明会、黑客松提交表单、提交要求指南、使用 Google Forms 提交、评审团及时间线` 


- **加入 Sierra AI 信息说明会！**：领先的对话式 AI 平台 **Sierra AI** 的独家信息说明会正在进行中。你可以在[这里](https://www.youtube.com/watch?v=-iWdjbkVgGQ)观看直播。
   - Sierra 渴望与优秀的开发者建立联系，所以不要错过这个机会！
- **黑客松提交表单已上线！**：LLM Agents MOOC 黑客松的 **提交表单和要求指南** 现已发布，项目提交截止日期为 **12 月 17 日**。提交流程已从 **Devpost 切换至 Google Forms**。
   - 你可以通过提供的链接找到提交项目的所有详情，并获得由杰出评审团进行评估的资格。
- **重要的提交链接**：参与者可以访问[此处](https://forms.gle/jNr8nSH9Cy9qpYcu5)的 **黑客松提交表单** 以及[此处](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?usp=sharing)的 **提交要求指南**。
   - 请务必认真准备并提交你的创新方案，以角逐奖项！
- **黑客松获奖者公布时间线**：LLM Agents MOOC 黑客松的获胜者将于 **2025 年 1 月** 上旬公布。鼓励参与者在截止日期前提交以供评估。
   - 组织者期待看到所有参与者的创意方案，欢迎在频道中提问。



**提到的链接**：<a href="https://www.youtube.com/watch?v=-iWdjbkVgGQ">LLM Agents MOOC Hackathon - Sierra Information Session</a>：未找到描述

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1313721707990749305)** (1 条消息): 

> `Certificate Declaration Form, Course Completion Tiers, Submission Checklist, Important Due Dates` 


- **证书申报表 (Certificate Declaration Form) 已发布**：针对尝试获得课程结业证书的人员，**Certificate Declaration Form** 已经发布，可以通过[此处](https://forms.gle/nYGJLPTdb7af2Dq59)的链接填写。参与者在所有提交过程中必须使用相同的电子邮件地址，以确保能够收到证书。
   - *请务必在提交所有作品时使用相同的电子邮件地址*；区分大小写，但额外的标点符号不影响。
- **五种课程完成等级说明**：参与者可以获得五个完成等级之一的证书：**Trailblazer、Mastery、Ninja、Legendary 或 Honorary Tier**。每人只能获得一张证书，进度将通过电子邮件进行跟踪。
   - 提醒：每个等级都有其必须满足的特定要求才能获得资格。
- **证书资格清单**：要获得证书，请完成所有必要的课程作业，包括 **12 个测验 (quizzes)、一篇撰写文章**，以及任何特定等级的要求（如 Lab 提交和项目表格）。课程网站上提供了清单供参考。
   - 确保同时也提交了 **Certificate Declaration Form** 以获得证书。
- **提交的重要截止日期**：所有文章提交、测验和 Lab 的截止日期为 **2024 年 12 月 12 日**晚上 11:59 (PST)。Hackathon 项目提交和 Certificate Declaration Form 的截止日期为 **2024 年 12 月 17 日**晚上 11:59 (PST)。
   - 请留意这些截止日期，以确保获得您的课程结业证书。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1313601425355440199)** (14 条消息🔥): 

> `Project Submission Requirements, Quizzes and Certificate Deadlines, Certification Declaration Categories, Feedback on MOOC Experience` 


- **项目提交细节已澄清**：成员们询问了项目提交中应包含的具体文件，并确认可以接受多个评估赛道。有关详细要求，请参阅[项目提交文档](https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?tab=t.0#heading=h.s229pxj2lhn2)。
   - 成员还被提醒检查是否可以同时申请 Masters 和 Trailblazer 类别。
- **测验和证书开放至 12 月**：已确认所有测验将保持开放至 **12 月 12 日**，证书申报截止日期为 **12 月 17 日**。这为参与者完成必要的评估留出了充足的时间。
   - 成员们表示有兴趣了解必须完成哪些内容才能获得证书。
- **Masters 与 Trailblazer 认证类别**：一位成员询问参与者是否可以同时申请 Masters 和 Trailblazer 类别，并得到保证，如果有必要，他们可以自动降级到 Trailblazer 等级。在这种情况下不需要重新提交表格。
   - 这为未达到 Masters 门槛的参与者提供了灵活性。
- **对 MOOC 的正面反馈**：参与者对 MOOC 的组织表示感谢，强调了整个课程期间提供的支持性环境。他们强调，该课程有助于理清当前 LLM 生态系统的复杂性。
   - 此外，演讲者阵容因其为学习体验增添的价值而受到赞扬，重点在于了解每位演讲者的背景。



**提到的链接**：<a href="https://docs.google.com/document/d/1WgWLZocBFM08cVVxo9P-ZMCnHBLGmQ7v8PbH4-AwnSk/edit?tab=t.0#heading=h.s229pxj2lhn2">Hackathon Track Submission Requirements</a>：所有赛道的一般提交要求视频演示：提供一个 YouTube 视频链接（最长 3 分钟；请上传至 YT），介绍您的项目概况并演示 k...

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1313653357201592351)** (4 messages): 

> `GPT-4 泄露，自动闭路字幕` 


- **对 GPT-4 数据泄露的担忧**：*一位用户针对 GPT-4 的泄露源提出了疑问*，特别是该泄露是涉及消费者版本还是企业版本。有迹象表明，默认设置可能已被重置，以便在至少 **30 天** 内共享用户数据用于建模目的。
   - 另一条评论提到 **GPT-4** 可能存在越狱风险，从而泄露训练集中的真实 PII（个人身份信息），并引用了具有里程碑意义的 **AOL 案例**。
- **对自动闭路字幕的请求**：*一位成员强调了上一节课缺少自动闭路字幕的问题*，指出这对于听障人士非常重要。他们建议启用此功能以提高无障碍性。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1313500744263143464)** (15 messages🔥): 

> `Anthropic 开发分支，Open Interpreter 安装问题，Linux 兼容性，OpenAI SDK 反馈` 


- **用户在使用 Anthropic 分支时遇到困难**：一位用户报告在尝试将最新的开发分支与 Anthropic 配合使用时遇到了 `TypeError`，具体表现为参数 'proxies' 是非预期的。
   - 另一位成员建议检查自定义 API base，指出这可能是影响客户端初始化的唯一变动。
- **安装命令与建议**：有人建议使用命令 `pip install --force-reinstall git+https://github.com/OpenInterpreter/open-interpreter.git@development` 重新安装 Open Interpreter 的最新开发版本。
   - 开发者确认该项目为了更好的性能已经完全重写，并鼓励用户报告任何缺失的功能。
- **征求对新实现的反馈**：随后讨论了对新的 OpenAI 兼容实现进行用户反馈的必要性，以确保其超越之前的版本。
   - 一位开发者表示，在收到用户输入后，希望为所有 OpenAI SDK 提供全面的支持。
- **不同 Linux 发行版的用户体验**：一位用户确认 Open Interpreter 在 Garuda-Linux（Arch-Linux 的一个分支）上运行正常，并对其兼容性表示赞赏。
   - 该用户还分享了在 Manjaro 和 OpenSuse 等多个其他 Linux 发行版上的经验，强调了他们的广泛测试。
- **Open Interpreter 的审批要求**：另一位用户指出，Open Interpreter 在执行代码前需要审批，但可以通过 `interpreter -y` 命令绕过。
   - 这揭示了软件中内置的部分功能，旨在允许代码执行前确保用户安全。



**提到的链接**：<a href="https://tenor.com/view/so-close-this-the-office-gif-1505267913606309297">So Close GIF - So Close This - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1313557411969241088)** (1 messages): 

> `LiveKit 使用，通过 O1 进行远程控制，计算机作为工具，OI 的 CLI 能力` 


- **利用 LiveKit 进行设备连接**：O1 通常利用 **LiveKit** 连接两个设备，例如用于通信的 **iPhone** 和用于接收请求的笔记本电脑或 **Raspberry Pi**。
   - 这种设置允许通过本地 OpenInterpreter 实例高效地远程访问并控制你的机器。
- **O1 增强的能力**：在 **computer use**（计算机使用）方面，O1 的能力超过了其他设置，使得将设备作为工具使用时更具灵活性。
   - 即使是以 **CLI** 形式运行，OpenInterpreter 依然能够有效地操作计算机。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

pjbontrager: 我不知道你在说什么 😗😅
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1313934393399574548)** (2 messages): 

> `Genie 2 Foundation Model, Generalist Agents Team` 


- **Genie 2 成为焦点**：有人请求在未来一天内将大规模 Foundation World Model **Genie 2** 的信息添加到 torchtune 中。更多细节可以在 [官方博客](https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/) 中找到。
   - 致谢部分强调了 **Jack Parker-Holder** 和 **Stephen Spencer** 等关键人物的贡献，突出了项目开发中的协作努力。
- **Generalist Agents 团队亮点**：由 Vlad Mnih 领导的 **Generalist Agents 团队**在 **Harris Chan** 和 **Maxime Gazeau** 等成员的贡献下取得了重大进展。这些努力彰显了该项目在 Agent 开发方面的综合方法。
   - 来自 **SIMA 团队**（包括 Frederic Besse 和 Tim Harley）的进一步支持，展示了为该倡议汇聚的多样化专业知识。
- **社区对更新的反应**：社区对这些更新表现出极大的热情，被简单地描述为“疯狂（insane）”。这反映了社区对 AI 项目进展的渴望。



**提到的链接**：<a href="https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/">Genie 2: A large-scale foundation world model</a>：为未来的通用 Agent 生成无限多样的训练环境

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1313501402873593948)** (12 messages🔥): 

> `Federated learning approaches, Community-led GPU contributions, MMLU performance validation, Training timelines, Meta's technology comparison` 


- **Federated Learning 展现前景**：正如一份分享的 [论文](https://arxiv.org/pdf/2411.19870) 中所讨论的，底层的 **Federated Learning** 方法可能比全同步方法产生更好的结果。
   - *训练仅剩 22 小时* 表示即将完成。
- **社区贡献有望复兴 Folding@home 模式**：类似于 **Folding@home** 的社区主导努力具有有趣的潜力，个人可以贡献 GPU 时间。
   - 随着模型规模超过单个数据中心，这一点可能变得至关重要。
- **MMLU Pro 设定验证标准**：为了在讨论的框架中验证一个区块，模型需要在 **MMLU Pro** 上达到 **90%** 的准确率。
   - 这突显了成功部署所需的严格性能标准。
- **Meta 技术对比**：针对关于 Fat Clusters 的讨论，有人担心 **Meta** 是否拥有类似的技术。
   - 一位贡献者表示，无论拥有多少 GPU，更大的模型都可能需要一些有趣的方法。
- **对资源密集型进展的兴奋**：用户对 AI 及相关领域的进步表示热烈欢迎，并注意到更快的训练时间线可能带来的影响。
   - 有人提到，“该死，这太疯狂了（Damn that's crazy）”，反映了对该领域持续进展的兴奋。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sakana.ai/cycleqd/">未找到标题</a>：未找到描述</li><li><a href="https://distro.nousresearch.com/">Nous DisTrO</a>：互联网上的分布式训练
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1313711136960548925)** (5 messages): 

> `Mechanistic Interpretability, Cellular Behavior, Epistemic Advantage` 


- **利用 Mechanistic Interpretability 深入研究细胞思维**：研究人员强调了一种名为 **Mechanistic Interpretability** 的新工具，用于探索细胞如何对其环境建模，将重点从基因转向 **Gene Regulatory Modules**（基因调节模块）和 **Sub-cellular Locations**（亚细胞位置）等概念。
   - 这种方法可能允许我们构建一种“细胞行为的通俗心理学”，并以更易理解的方式理解**细胞的内在生命**。
- **细胞疗法的荒谬性**：一位成员指出了将细胞视为需要治疗的对象的荒谬性，承认细胞并不像人类那样思考。
   - 尽管如此，以这种方式重新思考我们的理解，与传统的细胞功能观点相比，可能会提供 **Epistemic Advantages**（认知优势）。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1313703981482250350)** (3 条消息): 

> `非商业许可担忧，EDM2 框架扩散模型，扩散模型中的类别条件化` 


- **非商业许可可能限制实现**：一位成员指出，扩散模型的**非商业许可（non-commercial license）**可能会阻碍广泛实现的尝试。
   - 这种限制可能会影响开发者对该模型的采用和实验。
- **关于扩散模型 EDM2 框架的咨询**：另一位成员询问是否有人利用 **EDM2 框架**进行带有文本条件（text conditioning）的扩散模型训练。
   - 他们提到了一篇展示了**令人印象深刻的结果**的[论文](https://arxiv.org/pdf/2312.02696)，并强调了在具体实现方面的空白。
- **扩散模型中类别条件化的局限性**：论文提到了类别条件化（class conditioning），指出该模型只能生成特定于少数**预定义类别**的输出。
   - 这种受限的方法与所期望的文本条件化的灵活性形成对比，后者可以允许更广泛的生成创意。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1313783589472833578)** (1 条消息): 

> `Web 模型，来自 Meta 的 SAM，Tinygrad 展示` 


- **Web 模型正受到关注**：讨论的一个重点是开发 **Web 模型**（如云端的 ONNX），以增强机器学习工具的可访问性。
   - 这些模型通过提供既能在云端运行又能直接在浏览器中运行的功能，展示了吸引用户的潜力。
- **关于 Meta SAM 的有趣演示**：一位成员介绍了来自 **Meta 的 SAM**，强调了其被认为用户友好的演示网站，展示了开箱即用的有效模型。
   - **600M 图像嵌入 Transformer** 在云端运行，而较小的模型则直接在浏览器中运行，展示了一个实际应用的例子。
- **未来模型的质量基准**：SAM 演示为未来旨在展示 **tinygrad** 的模型和网页设定了一个可能的质量基准，旨在增加社区的吸引力。
   - 虽然并非完美无缺，但演示中注意到的改进恰当地反映了未来 AI 工具预期的进步。



**提到的链接**：<a href="https://segment-anything.com/demo.">Segment Anything</a>：Meta AI 计算机视觉研究

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1313628553568321596)** (6 条消息): 

> `Threadgroup/Grid 大小，BEAM Search 解释，JIT 中的共享输出缓冲区，循环的手动向上转型` 


- **可以在 OptOps 中更改 Threadgroup/Grid 大小**：一位用户询问在 `uopgraph.py` 的图重写优化期间是否可以更改 threadgroup/grid 大小。George Hotz 回复说它们可以被修改，具体是在 `kernel.py` 的 **OptOps** 中。
- **解释了 BEAM Search 和内核优化选项**：一位用户分享了一篇关于 [BEAM Search](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md) 的帖子，以及对内核优化选项的解释。该帖子可作为理解 **tinygrad** 中这些概念的资源。
- **JIT 函数重用输出缓冲区**：关于 JIT 函数的一个说明显示，在第一次调用后，jitted 函数会重用相同的输出缓冲区，这可能会覆盖之前的结果。为了保留结果，有必要在每次调用后使用 `.clone().realize()`。
- **大循环的手动向上转型（Upcasting）**：一位用户询问是否可以为大循环手动强制进行向上转型。对话仍在继续，尚未提供直接回答。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/m">m - 概览</a>：打字员、工程师、代码诗人、优美数据结构的爱好者。- m</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241203_beam.md">tinygrad-notes/20241203_beam.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账号为 mesozoic-egg/tinygrad-notes 做出贡献。
</li>
</ul>

</div>
  

---

### **Axolotl AI ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1313703914096427079)** (1 条消息): 

> `Office Hours, Axolotl Survey, Axolotl Swag` 


- **即将举行的 Office Hours 提醒**：温馨提醒，Office Hours 定于本周四，**12/5**，**东部时间下午 1 点** / **太平洋时间上午 10 点**。
   - 团队非常期待在本次会议中与大家交流！
- **征集意见：填写 Axolotl Survey**：为了有效地定制讨论内容，诚邀参与者填写 **Axolotl Survey**。
   - 您的反馈将指导改进工作，参与者将获得专属的 **Axolotl swag**！
- **完成调查问卷可获得限量版 Axolotl Swag**：作为对完成调查的感谢，受访者将获得即将发布的 **Axolotl swag**（送完即止）。
   - 这一激励措施彰显了团队对参与者时间和意见的重视。



**提及的链接**：<a href="https://gravel-salmon-db9.notion.site/1421d2ab4f4081168f6fe3770fae446c">Notion – 笔记、任务、维基和数据库的一体化工作区。</a>：一款将日常工作应用融合为一的新工具。为您和您的团队提供的一体化工作区。

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1313557490226565193)** (3 条消息): 

> `ADOPT optimizer, Axolotl updates` 


- **ADOPT Optimizer 更新已集成至 Axolotl**：**ADOPT optimizer** 的最新更新已集成到 Axolotl 代码库中，旨在提高**训练稳定性**。查看 [pull request #2104](https://github.com/axolotl-ai-cloud/axolotl/pull/2104) 中的更改。
   - 该 pull request 确保了与 **torch 版本**的兼容性，并合并了原作者在 [此处](https://github.com/iShohei220/adopt) 进行的最新修改。
- **讨论 ADOPT Optimizer 的核心优势**：一位成员在实现后询问了 **ADOPT optimizer** 的优势，表现出对其益处的好奇。
   - 作为回应，有人指出该优化器可以在任何 beta 值下实现**最优收敛**。



**提及的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/pull/2104">Check torch version for ADOPT optimizer + integrating new ADOPT updates by bursteratom · Pull Request #2104 · axolotl-ai-cloud/axolotl</a>：描述：确保在使用 ADOPT optimizer 时 torch 版本兼容。合并了原作者对 ADOPT optimizer 的最新更改。https://github.com/iShohei220/adoptMotiv...

  

---


### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1313947537845129267)** (1 条消息): 

> `Open Source Engineer Roles, Unternet Hiring` 


- **Unternet 招聘开源工程师**：[Unternet 正在招聘一名 Open Source Engineer](https://discord.com/channels/1089876418936180786/1313839138562248737)，负责贡献开源项目、编写技术文档并与社区互动。
   - 欢迎感兴趣的候选人在上方链接的讨论帖中进一步咨询。
- **社区参与机会**：该职位强调了与社区合作以及编写技术文档的重要性。
   - 此角色面向对开源贡献充满热情的个人。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1313856112520593461)** (1 条消息): 

> `Gorilla Model Issue, Protobuf Dependency Error` 


- **Gorilla 模型启动失败**：一位用户在尝试使用命令启动模型时遇到错误，提示与 tokenizer 相关的依赖问题。
   - *错误信息强调缺少 protobuf 库*，尽管其环境中已安装该库。
- **未找到 Protobuf 库**：用户确认已安装版本为 **5.29.0** 的 protobuf 包，但系统仍报告缺失。
   - 这引发了关于为何环境无法识别已安装包的疑问。


  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1313660575003443271)** (1 条消息): 

> `Ticket Messaging` 


- **工单消息跟进**：一名成员提醒 Nick 查看他们发送的关于工单的消息，请求他在有空时查看。
   - 他们强调了及时响应的重要性，暗示需要快速解决。
- **未提供额外上下文**：除了关于工单的跟进之外，对话没有提供任何进一步的上下文。
   - 没有讨论额外的评论或链接。


  

---


---


{% else %}


> 完整的频道逐个分析已为邮件格式进行截断。 
> 
> 如果您想查看完整的分析，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}