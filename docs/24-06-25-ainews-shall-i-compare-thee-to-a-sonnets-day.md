---
companies:
- anthropic
- lmsys
- glif
- comfyui
date: '2024-06-26T00:39:44.720143Z'
description: Anthropic 的 **Claude 3.5 Sonnet** 在编程和硬核提示词（hard prompt）竞技场中位居榜首，超越了
  **GPT-4o**，并以更低的成本与 **Gemini 1.5 Pro** 展开竞争。**Glif** 展示了一个全自动的 **Wojak 迷因生成器**，该工具利用
  Claude 3.5 进行 JSON 生成，并结合 ComfyUI 生成图像，体现了其全新的 JSON 提取能力。**Artifacts** 功能支持快速开发小众应用，例如在不到
  5 分钟内制作出的双显示器可视化工具。**François Chollet** 强调，与现有的核裂变电站相比，核聚变能源并非短期解决方案。**Mustafa Suleyman**
  指出，目前 75% 的办公室职员都在使用 AI，这标志着工作模式正向 AI 辅助生产力转变。
id: a7147fb1-bfa9-47d9-b734-cbb42c629bd8
models:
- claude-3.5-sonnet
- claude-3.5
- gpt-4o
- gemini-1.5-pro
original_slug: ainews-sonnet
people:
- fchollet
- mustafasuleyman
title: 我可否将你比作十四行诗的一天？
topics:
- hard-prompts
- json
- json-extraction
- meme-generation
- instruction-following
- app-development
- fusion-energy
- nuclear-fission
- productivity
---

<!-- buttondown-editor-mode: plaintext -->**Claude 3.5 Sonnet 就够了。**

> 2024年6月24日至6月25日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（包含 **415** 个频道和 **2614** 条消息）。
预计为您节省阅读时间（以 200wpm 计算）：**260 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

 
![image.png](https://assets.buttondown.email/images/5bb77dd0-5fd1-4526-a833-52919ac96f78.png?w=960&fit=max)
 

在代码的领域中，**Claude Sonnet** [冉冉升起](https://twitter.com/lmsysorg/status/1805329822748655837)，

一位身着**硅装**的数字吟游诗人。

穿梭于 **Hard Prompts** 的迷宫，其威力非凡，

然而怀疑者对其自信的锋芒仍存疑虑。

**LMSYS** [授予](https://twitter.com/lmsysorg/status/1805329826951348504)其银牌，离金牌仅一步之遥，

它**强健**的大脑优雅地处理各项任务。

但质疑的低语如阴影般蔓延：

**Anthropic** 的宠儿真能保持这一步调吗？

在 **Glif** 的[领地](https://twitter.com/fabianstelzer/status/1805326248261910552)，它孕育了 **Wojak** 之梦，

一位以闪电般速度工作的 **Meme** 匠人。

五分钟便能打造出看似不可能之物，

**JSON** 提取，一项强大的功绩。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 简报

> 所有摘要均由 Claude 3 Opus 完成（4 次运行中的最佳结果）。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

**来自 Anthropic 的 Claude 3.5 Sonnet**

- **令人印象深刻的性能**：Claude 3.5 Sonnet 在 Coding Arena、Hard Prompts Arena 夺得第一，总榜排名第二，以更低的成本超越了 Opus，并与 GPT-4o/Gemini 1.5 Pro 旗鼓相当。[@lmsysorg](https://twitter.com/lmsysorg/status/1805329822748655837)
- **超越 GPT-4o**：Sonnet 在 Arena 总榜中位列第二，超越了 GPT-4o。[@lmsysorg](https://twitter.com/lmsysorg/status/1805329826951348504)
- **在 "Hard Prompts" 中表现稳健**：Sonnet 在具有特定筛选标准的 "Hard Prompts" Arena 中同样表现强劲。[@lmsysorg](https://twitter.com/lmsysorg/status/1805329824770400402)
- **态度与指令遵循（instruction-following）批评**：一些人认为 Sonnet 的态度暗示了它可能并不具备的能力，并指出 Anthropic 的指令微调（instruction-tuning）不如 OpenAI 强大。[@teortaxesTex](https://twitter.com/teortaxesTex/status/1805316991848267832)

**Glif 与 Wojak Meme 生成器**

- **全自动 Meme 生成器**：在 Glif 中仅用 5 分钟就构建了一个 Wojak Meme 生成器，使用 Claude 3.5 生成 JSON，ComfyUI 生成 Wojak 图像，并通过 JSON 提取器 + Canvas Block 进行集成。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326248261910552)
- **JSON 提取器模块展示**：这展示了 Glif 新的 JSON 提取器模块的实用性，该模块可让 LLM 生成 JSON 并将其拆分为变量。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326958986936679)
- **来自 Claude 的犀利输出**：Claude 3.5 的 Meme 生成器产生的一些输出出人意料地犀利（edgy）。[@fabianstelzer](https://twitter.com/fabianstelzer/status/1805326958986936679)

**Artifacts 与小众应用创建**

- **让原本不会被编写的软件成为可能**：Artifacts 使得快速创建小众应用、内部工具或趣味项目成为可能，而这些项目在以前可能永远不会被开发。[@alexalbert__](https://twitter.com/alexalbert__/status/1805261958134055409)
- **双显示器可视化工具示例**：Claude 在不到 5 分钟的时间内制作了一个实用的应用，用于可视化双显示器如何摆放在桌面上——虽然不是开创性的，但考虑到创建速度，它非常有价值。[@alexalbert__](https://twitter.com/alexalbert__/status/1805261958134055409)

**聚变能与核裂变**

- **聚变并非短期内的游戏规则改变者**：与技术乐观主义相反，现今可行的聚变技术在未来 100 年内几乎不会影响能源经济。[@fchollet](https://twitter.com/fchollet/status/1805343413669446022) 
- **裂变作为现有的清洁能源解决方案**：核裂变已经提供了近乎无限的清洁能源，1970 年代的电厂建设和运营成本比假设的聚变电厂还要便宜。[@fchollet](https://twitter.com/fchollet/status/1805343413669446022)
- **燃料成本是次要因素**：裂变发电成本的约 100% 来自电厂（80%）和输电（20%），而非燃料。维持 1.5 亿度等离子体的聚变反应堆在建造和运营上也不会是免费的。[@fchollet](https://twitter.com/fchollet/status/1805343413669446022)

**AI 采用与生产力**

- **75% 的员工正在使用 AI**：对于办公桌工作，不将 AI 融入工作的人正变得罕见。向 AI 辅助生产力的转型正在进行中。[@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1805245591993160087)
- **增量生产力的提升至关重要**：即使是 AI 带来的微小生产力提升，对于忙碌的人和初创公司来说也极具价值。[@scottastevenson](https://twitter.com/scottastevenson/status/1805293727964938705)

**Together Mixture-of-Agents (MoA)**

- **MoA 以 50 行代码实现**：Together 仅用 50 行代码就实现了他们的 Mixture-of-Agents (MoA) 方法。[@togethercompute](https://twitter.com/togethercompute/status/1805289022241259803)

**Retrieval Augmented Generation (RAG) Fine-Tuning**

- **RAG 微调优于大型模型**：在流行的开源代码库上，使用 RAG 微调的 Mistral 7B 模型可以媲美或击败 GPT-4o 和 Claude 3 Opus 等更大型的模型，且在 Together 上的成本降低了 150 倍，速度提高了 3.7 倍。[@togethercompute](https://twitter.com/togethercompute/status/1805340734918476076)
- **代码库性能提升**：RAG 微调在 5 个测试代码库中的 4 个上提升了性能。[@togethercompute](https://twitter.com/togethercompute/status/1805340738529771609)
- **使用合成数据集**：这些模型是在由 Morph Code API 生成的合成数据集上进行微调的。[@togethercompute](https://twitter.com/togethercompute/status/1805341892978393344)

**Extending LLM Context Windows**

- **KVQuant 用于 10M token 上下文**：KVQuant 将缓存的 KV 激活量化为超低精度，从而在 8 个 GPU 上将 LLM 上下文扩展到 10M token。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382644659896515) 
- **Activation Beacon 用于 400K 上下文**：Activation Beacon 压缩 LLM 激活以在有限窗口内感知 400K token 上下文，在 8xA800 GPU 上训练时间小于 9 小时。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382628742234179)
- **Infini-attention 用于 1M 序列长度**：Google 的 Infini-attention 使用压缩内存和局部/长期注意力，将 1B LLM 扩展到 1M 序列长度。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382639152775392)
- **LongEmbed 用于 32K 上下文**：Microsoft 的 LongEmbed 使用并行窗口、重组位置 ID 和插值，在无需重新训练的情况下将嵌入模型上下文扩展到 32K token。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382655946768548)
- **PoSE 用于 128K 上下文**：PoSE 在固定窗口中操纵位置索引以模拟更长的序列，使 4K LLaMA-7B 能够处理 128K token。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382661223203256)
- **LongRoPE 用于 2M 上下文**：Microsoft 的 LongRoPE 在保留短上下文性能的同时，将预训练 LLM 上下文扩展到 2M token，无需长文本微调。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382661223203256)
- **Self-Extend 用于长上下文**：Self-Extend 通过 FLOOR 将未见过的相对位置映射到已见过的相对位置，从而在无需微调的情况下激发 LLM 固有的长上下文能力。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382676859470113)
- **Dual Chunk Attention 用于 100K 上下文**：DCA 将注意力分解为块内/块间注意力，使 LLaMA-70B 在无需持续训练的情况下支持 100K token 上下文。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382682555089364)

**Many-Shot In-Context Learning**

- **显著的性能提升**：Google 发现 many-shot 相比 few-shot 上下文学习有重大提升，即使使用 AI 生成的示例也是如此。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **机器翻译和摘要改进**：Many-shot ICL 有助于低资源语言翻译，并接近微调后的摘要性能。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **强化 ICL 与模型推理**：使用经过正确性过滤的模型生成推理（rationales）的强化 ICL，在数学/问答任务上媲美或击败了人类推理。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943) 
- **无监督 ICL 的前景**：无监督 ICL（仅使用问题进行提示）显示出前景，尤其是在 many-shot 的情况下。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)
- **适应新的标签关系**：通过足够的示例，many-shot ICL 可以适应与预训练偏见相矛盾的新标签关系。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805382666612551943)

**杂项**

- **120 FPS 下的时间抖动 (Temporal dithering)**：对于大多数人来说，用于色深/超采样的时间抖动在 120 FPS 时是不可见的。如果 120 FPS 配合抖动，2D VR 窗口可以超过显示分辨率。[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1805295838211571878)
- **先发效应 (First-mover effect)**：存在性证明驱动了快速追赶。Sonnet-3.5 现在略高于曾经领先的 GPT。在 4 个月内出现了 4-5 个达到 Sora 70-80% 质量的克隆版本。[@DrJimFan](https://twitter.com/DrJimFan/status/1805265388256837842)
- **240T token 数据集**：一个 240T token 的数据集现已可用于 LLM 训练，比之前的 SOTA 大 8 倍。FineWeb 的 15T 数据量为 48 TB。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1805389473892778185)
- **iOS 18 运动提示 (motion cues)**：iOS 18 增加了随车移动的屏幕圆点，以减轻玩手机时的晕车感。[@kylebrussell](https://twitter.com/kylebrussell/status/1805211971731533848)
- **开源与企业利益**：当开源被战略性地用于企业利益时，开源很难做到真正的开放。[@fchollet](https://twitter.com/fchollet/status/1805315872988479788)

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 发展与进步**

- **应对 AI 不平等**：在接受 Business Insider 采访时，Anthropic CEO Dario Amodei 建议，[**需要全民基本收入（UBI）以外的解决方案来解决 AI 驱动的不平等问题**](https://www.businessinsider.com/anthropic-ceo-dario-amodei-universal-basic-income-ubi-ai-inequality-2024-6)。Microsoft AI CEO Mustafa Suleyman 预测，[预计在 2 年内推出的 GPT-6 将能够遵循指令并采取一致的行动](https://x.com/tsarnick/status/1805104305139232975)，一些人将 GPT 模型的炒作与 iPhone 相提并论。

- **为计算的未来提供动力**：Bill Gates 揭晓了一种[**革命性的核反应堆设计，旨在为怀俄明州未来的计算提供动力**](https://www.youtube.com/watch?v=LbWNXX4aLM8)。与此同时，一项演示展示了在[资源受限的设备（如 1GB RAM 的复古掌机）上运行大型 AI 模型（如 3.3B BITNET）的潜力](https://streamable.com/gwt5fm)。

**AI 模型、框架与基准测试**

- **Anthropic 的 Claude 取得长足进步**：Anthropic 的 Claude 3.5 Sonnet 模型在 [LMSYS Arena 基准测试中已超越 OpenAI 的 GPT-4o](https://x.com/lmsysorg/status/1805329822748655837)。一位用户还展示了一个[由 Claude 创建的分形浏览器，能够显示并缩放四种不同的分形](https://v.redd.it/dhvzpd5h7h8d1)。

- **新模型发布**：[Dolphin-2.9.3-Yi-1.5-34B-32K 模型已发布](https://i.redd.it/yq78y02vqi8d1.png)，最新的 Chrome Canary 版本现在[能够本地运行 Gemini 模型](https://twitter.com/mortenjust/status/1805190952358650251)。一位用户还提供了[针对摘要和指令遵循任务的各种模型的评估报告](https://www.reddit.com/r/LocalLLaMA/comments/1dnavrt/update_model_review_for_summarizationinstruct_1gb/)。

**AI 伦理、监管与社会影响**

- **对 AI 公司的挫败感**：一位用户表达了[对 OpenAI 延迟发布 GPT-4o 语音功能的挫败感](https://www.reddit.com/r/OpenAI/comments/1dn7dwq/im_sick_of_waiting_for_chatgpt_4o_voice_and_i/)，导致他们对该公司失去尊重并转向 Anthropic 的 Claude AI。讨论涉及了公司履行承诺的重要性以及对用户信任的影响。

- **AI 的“西部大荒野”**：有人认为我们目前正处于 [AI 发展的“西部大荒野”阶段](https://www.reddit.com/r/singularity/comments/1dnxd9o/we_are_in_the_wild_west_of_the_preai_era/)，一切都在迅速演变，好比电子游戏的序幕。还讨论了 [AI 对齐对创意作品的潜在影响](https://www.reddit.com/r/singularity/comments/1dnqe2i/what_are_some_of_the_best_books_movies_and_games/)。

- **怀疑论与迷因**：一个[迷因嘲讽了 LLM 怀疑论者](https://i.redd.it/wbhy32608m8d1.png)，即使在模型不断进步的情况下，他们仍继续质疑 AI 的能力；评论指出，一旦 AI 达到人类水平的性能，怀疑论者可能会将争论转向 AI 是否拥有“灵魂”。

**AI 应用与使用案例**

- **自动化与创意产业**：Apple 计划将其 [iPhone 最终组装线的 50% 实现自动化](https://9to5mac.com/2024/06/24/iphone-supply-chain-automation-workers/)，用机器取代人工。唱片公司已使用 AI 工具 [Udio 和 Suno 来重新创作著名歌曲的版本](https://www.reddit.com/r/singularity/comments/1dnlext/record_labels_were_able_to_basically_recreate/)，引发了关于版权和音乐产业的疑问。

- **写实 AI 图像与编程**：一位用户展示了[令人印象深刻的写实 AI 生成图像](https://www.reddit.com/gallery/1dntyba)，评论建议使用 stock photo checkpoints 和 realism LoRAs 等技术来实现自然的效果。[DeepseekCoder-v2 模型因其编程性能受到赞誉](https://www.reddit.com/gallery/1dncebg)。

- **Web UI 自动化**：据报道，Claude 3.5 是[第一个能可靠用于 Web UI 自动化和交互的 LLM](https://www.reddit.com/r/singularity/comments/1dnmatp/claude_35_is_the_first_llm_i_was_able_to_reliably/)，在可访问性和前端测试等任务上表现优于 GPT-4o。

**AI 研究与开发**

- **教育资源与即将发布的内容**：Andrej Karpathy 的 GitHub 仓库 [“Let's build a Storyteller”](https://github.com/karpathy/LLM101n) 旨在教育社区如何构建 AI 模型。人们对 [Ray Kurzweil 即将出版的新书《The Singularity is Nearer》](https://www.reddit.com/r/singularity/comments/1dnbgkx/ray_kurzweils_book_is_being_released_in_a_few/) 的发布表示期待。

- **新模型与工具**：Salesforce 发布了 [Moirai-1.1，一个更新的时间序列基础模型 (time series foundation model)](https://www.reddit.com/r/LocalLLaMA/comments/1dnajuy/salesforce_releases_moirai11_time_series/)，用于多种预测任务。开源项目 [WilmerAI 正式推出](https://www.reddit.com/r/LocalLLaMA/comments/1dnsfh9/sorry_for_the_wait_folks_meet_wilmerai_my_open/)，旨在通过 prompt routing 和多模型工作流管理最大化本地 LLM 的潜力。[Rensa，一个高性能的 MinHash 实现](https://www.reddit.com/r/LocalLLaMA/comments/1dn7erd/rensa_a_high_performance_minhash_implementation/)，也被宣布用于快速相似度估计和去重。

**杂项**

- **脑细胞与 AI**：讨论了使用[人脑细胞为研发计算机提供动力](https://www.reddit.com/r/singularity/comments/1dnwyuu/human_brain_cells_now_running_computers_for_rd/)的话题，并将其与《黑客帝国》(Matrix) 进行类比，探讨了对 AGI/ASI 发展的影响。文中澄清这些是细胞而非整个大脑，且缺乏产生意识的复杂性。

- **多模态 AI 与太阳能**：人们对 [Meta 的多模态 AI 模型 Chameleon](https://www.reddit.com/r/LocalLLaMA/comments/1dnm1v1/have_any_of_you_tried_metas_multimodal_chameleon/) 表示关注，并注意到社区讨论较少。一篇《经济学人》(The Economist) 的文章预测，随着[太阳能价格日益亲民，它将成为主要的能源来源](https://www.economist.com/interactive/essay/2024/06/20/solar-power-is-going-to-be-huge)。

- **独特的 AI 项目**：介绍了字体 [llama.ttf，它同时也是一个语言模型](https://fuglede.github.io/llama.ttf/)。对 [Anthropic 的 Claude 3.5 Sonnet 所使用的 system prompt 进行的取证分析](https://tyingshoelaces.com/blog/forensic-analysis-sonnet-prompt) 引入了 AI prompt 中 “Artifacts” 的概念。

---

# AI Discord 摘要

> 摘要的摘要的摘要

## Claude 3 Sonnet

**1. LLM 进展与基准测试**

- 来自 Meta 的 **[Llama 3](https://lmsys.org/blog/2024-05-08-llama3/)** 在 **ChatbotArena** 等排行榜上迅速攀升至榜首，在超过 50,000 场对决中表现优于 **GPT-4-Turbo** 和 **Claude 3 Opus** 等模型。

- 来自 IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 等新模型增强了代码任务的指令遵循能力，而 **[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2)** 则拥有 **236B 参数**。

- 某些基准测试受到质疑，人们呼吁像 Meta 这样可靠的来源建立现实的 LLM 评估标准。

**2. 优化 LLM 推理与训练**

- **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 承诺在 GPU 上进行大规模模型训练时，将通信开销降低 4 倍。

- **[vAttention](https://arxiv.org/abs/2405.04437)** 系统动态管理 KV-cache 内存，在不使用 PagedAttention 的情况下实现高效的 LLM 推理。

- **[QServe](https://arxiv.org/abs/2405.04532)** 引入了 **W4A8KV4 量化**，以提升 GPU 上基于云的 LLM 服务性能。

- **[Consistency LLMs](https://hao-ai-lab.github.io/blogs/cllm/)** 等技术探索了并行 Token 解码，以降低推理延迟。

**3. 开源 AI 框架与社区努力**

- **[Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/)** 支持多种数据集格式，用于 LLM 的指令微调和预训练。

- **[LlamaIndex](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex)** 为吴恩达（Andrew Ng）关于构建 Agentic RAG 系统的新课程提供支持。

- **[RefuelLLM-2](https://huggingface.co/refuelai/Llama-3-Refueled)** 已开源，声称是处理“枯燥数据任务”的最佳 LLM。

- **[Modular](https://www.modular.com/blog/developer-voices-deep-dive-with-chris-lattner-on-mojo)** 展示了 Mojo 在 Python 集成和 AI 扩展（如 _bfloat16_）方面的潜力。

**4. 多模态 AI 与生成模型创新**

- **[Idefics2 8B Chatty](https://twitter.com/sanhestpasmoi/status/1787503160757485609)** 专注于提升聊天交互体验，而 **[CodeGemma 1.1 7B](https://twitter.com/reach_vb/status/1786469104678760677)** 则精进了编程能力。

- **[Phi 3](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/)** 模型通过 WebGPU 将强大的 AI 聊天机器人引入浏览器。

- 结合 **Pixart Sigma + SDXL + PAG** 旨在实现 **DALLE-3** 级别的输出，并具有通过微调进一步优化的潜力。

- 开源项目 **[IC-Light](https://github.com/lllyasviel/IC-Light)** 专注于改进图像重光照（image relighting）技术。

## Claude 3.5 Sonnet

1. **新型 LLM 撼动排行榜**：

   - [Replete-Coder-Llama3-8B 模型](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) 因其精通 100 多种编程语言和先进的代码编写能力，在多个 Discord 社区引起了广泛关注。

   - 拥有 236B 参数的 [DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2) 和 [Hathor_Fractionate-L3-8B-v.05](https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05) 因其在各项任务中的表现而受到讨论。

   - 对 Benchmarks 的怀疑是一个共同的主题，用户强调相比排行榜名次，更需要进行实际场景测试。

2. **开源工具赋能 AI 开发者**：

   - [Axolotl](https://openaccess-ai-collective.github.io/axolotl/docs/dataset-formats/) 因支持 LLM 训练中的多种数据集格式而受到青睐。

   - [LlamaIndex](https://www.latent.space/p/llm-training-2024) 因其与 DSPy 的集成而受到关注，增强了 RAG 能力。

   - [llamafile v0.8.7](https://www.phoronix.com/news/Llamafile-0.8.7-Released) 的发布带来了更快的量化操作和 Bug 修复，并暗示了潜在的 Android 兼容性。

3. **优化技术突破 LLM 边界**：

   - [Adam-mini 优化器](https://arxiv.org/abs/2406.16793) 在各 Discord 社区引发讨论，因为它与 AdamW 相比能减少 45-50% 的显存占用。

   - [Sohu 的 AI 芯片](https://x.com/Etched/status/1805625693113663834) 宣称在运行 Llama 70B 时每秒可处理 500,000 个 Token，尽管社区对这些性能指标表示怀疑。

4. **AI 伦理与安全成为焦点**：

   - Ollama 项目中的一个[远程代码执行漏洞 (CVE-2024-37032)](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) 引发了多个 Discord 社区对 AI 安全性的担忧。

   - 关于 [AI 实验室安全](https://x.com/jordanschnyc/status/1805340489391997177) 的讨论强调了加强措施的必要性，以防止“超人类黑客攻击”和未经授权的访问等风险。

   - 据 [Music Business Worldwide](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/) 报道，针对 Suno 和 Udio 的 AI 音乐生成诉讼，引发了各社区关于版权和伦理 AI 训练的辩论。

## Claude 3 Opus

**1. 新 LLM 发布与 Benchmarking**

- **Replete-Coder-Llama3-8B** 模型在 100 多种语言的编程熟练度和无审查训练数据方面表现出色 ([Hugging Face](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B))。
- 关于 Benchmark 可靠性的讨论，有人认为它们不能反映真实世界的性能 (Unsloth AI Discord)。
- **DeepSeek-V2** 在 **AlignBench** 和 **MT-Bench** 测试中，在某些任务上的表现优于 GPT-4 ([Twitter 公告](https://x.com/deepseek_ai/status/1787478986731429933))。

**2. 优化 LLM 性能与效率**

- **Adam-mini optimizer** 与 AdamW 相比，在保持相似或更好性能的同时，减少了 45-50% 的内存占用 ([arXiv 论文](https://arxiv.org/abs/2406.16793))。
- **AQLM** 和 **QuaRot** 等量化技术使得在单张 GPU 上运行大型模型成为可能，例如在 RTX3090 上运行 **Llama-3-70b** ([AQLM 项目](https://github.com/Vahe1994/AQLM))。
- **Dynamic Memory Compression (DMC)** 提升了 Transformer 的效率，在 **H100 GPUs** 上可能将吞吐量提高多达 370% ([DMC 论文](https://arxiv.org/abs/2403.09636))。

**3. 开源 AI 框架与协作**

- **Axolotl** 支持多种用于 LLM 指令微调（Instruction Tuning）和预训练的数据集格式 ([Axolotl prompters.py](https://github.com/OpenAccess-AI-Collective/axolotl/blob/3367fca73253c85e386ef69af3068d42cea09e4f/src/axolotl/prompters.py#L47))。
- **LlamaIndex** 与吴恩达（Andrew Ng）关于构建 Agentic RAG 系统的新课程进行了整合 ([DeepLearning.AI 课程](https://www.deeplearning.ai/short-courses/building-agentic-rag-with-llamaindex))。
- **Mojo** 语言暗示了未来与 Python 的集成以及针对 AI 的特定扩展，如 _bfloat16_ (Modular Discord)。
- **StoryDiffusion** 作为 Sora 的开源替代方案，已在 MIT 许可证下发布 ([GitHub 仓库](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file))。

**4. 多模态 AI 与生成模型**

- **Idefics2 8B Chatty** 和 **CodeGemma 1.1 7B** 模型分别专注于聊天交互和编程能力 ([Twitter](https://twitter.com/sanhestpasmoi/status/1787503160757485609) [帖子](https://twitter.com/reach_vb/status/1786469104678760677))。
- **Phi 3** 利用 WebGPU 将强大的 AI 聊天机器人带入浏览器 ([Reddit 帖子](https://www.reddit.com/r/LocalLLaMA/comments/1cn2zwn/phi3_webgpu_a_private_and_powerful_ai_chatbot/))。
- 结合 **Pixart Sigma**、**SDXL** 和 **PAG** 旨在实现 **DALLE-3** 级别的输出 (Latent Space Discord)。
- **IC-Light** 是一个专注于图像重光照（Relighting）技术的开源项目 ([GitHub 仓库](https://github.com/lllyasviel/IC-Light))。

## GPT4O (gpt-4o-2024-05-13)


1. **性能改进与技术修复**：
   - **PyTorch Tensor 对齐问题受到关注**：用户讨论了为了提高内存使用效率而进行的 PyTorch Tensor 对齐，并引用了代码和文档来解决诸如 `torch.ops.aten._weight_int4pack_mm` 等问题 [源代码](https://github.com/pytorch/pytorch/pull/110914)。
   - **LangChain 增强功能**：成员们赞扬了 [LangChain Zep](https://www.getzep.com/) 的集成，它提供了持久的 AI 记忆，能够总结对话以实现有效的长期使用。
   - **Tinygrad 中发现 LazyBuffer Bug**：记录了 Tinygrad 中 “LazyBuffer” 缺少属性 “srcs” 的问题，并建议了诸如 `.contiguous()` 以及使用 Docker 进行 CI 调试等修复方案 [Dockerfile 地址](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1)。

2. **AI 领域的伦理与法律挑战**：
   - **AI 音乐生成器因版权侵权被起诉**：主要唱片公司正在起诉 Suno 和 Udio，指控其未经授权在受版权保护的音乐上进行训练，这引发了对伦理 AI 训练实践的质疑 [Music Business Worldwide 报告](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/)。
   - **Carlini 为其攻击研究辩护**：Nicholas Carlini 为其关于 AI 模型攻击的研究辩护，指出这些研究揭示了关键的 AI 模型漏洞 [博客文章](https://nicholas.carlini.com/writing/2024/why-i-attack.html)。
   - **Probllama 的安全漏洞**：Rabbithole 的安全披露揭示了由于硬编码 API 密钥导致的严重漏洞，可能导致 ElevenLabs 和 Google Maps 等服务被广泛滥用 [完整披露](https://rabbitu.de/articles/security-disclosure-1)。

3. **新发布与 AI 模型创新**：
   - **EvolutionaryScale 凭借 ESM3 取得突破**：ESM3 模型模拟了 5 亿年的进化，获得了 1.42 亿美元的融资，旨在达到编程生物学的新高度 [融资公告](https://x.com/pdhsu/status/1805563282746446116)。
   - **Gradio 的新功能集**：最新发布的 Gradio v4.37 引入了全新的聊天机器人 UI、动态图表和 GIF 支持，同时进行了性能改进以提升用户体验 [变更日志](https://www.gradio.app/changelog)。
   - **OpenRouter 上兴起的 AI 模型**：新的 AI 模型如 [AI21 的 Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct) 和 NVIDIA 的 [Nemotron-4 340B](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) 已添加到平台，为各种应用集成了多样化的能力。

4. **数据集管理与优化**：
   - **解决数据集加载中的 RAM 问题**：讨论了使用 `save_to_disk`、`load_from_disk` 和启用 `streaming=True` 等技术，以减轻在 AI 模型中处理大型数据集时的内存问题。
   - **Minhash 优化带来性能提升**：一位成员展示了使用 Python 进行 Minhash 计算时 12 倍的性能提升，引发了进一步优化的兴趣和协作 [GitHub 链接](https://github.com/beowolx/rensa)。

5. **会议、活动与社区参与**：
   - **AI Engineer World's Fair 亮点**：随着工程师们对 AI Engineer World's Fair 的期待，气氛日益热烈，届时将有主题演讲和引人入胜的演讲，包括来自 LlamaIndex 团队的见解 [活动详情](https://twitter.com/llama_index/status/1805366745877856363)。
   - **检测 LLM 中的机器人与欺诈**：6 月 27 日的一场活动将邀请来自 hCaptcha 的 Unmesh Kurup 讨论对抗基于 LLM 的机器人策略以及现代 AI 安全中的欺诈检测 [活动注册](https://lu.ma/y6hz8cod?utm_source=diskord)。
   - **OpenAI 的 macOS 版 ChatGPT 桌面应用**：新应用允许 macOS 用户访问具有增强功能的 ChatGPT，标志着 AI 可用性和集成迈出了重要一步 [macOS 版 ChatGPT](https://openai.com/chatgpt/mac/)。

---

# 第 1 部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Python 中的“脑力胜过体力”**：在关于数值精度的热烈讨论中，Python 用户分享了处理常导致 `OverflowError` 的大浮点数计算的代码片段。解决方案围绕在不损失精度的情况下计算高幂浮点数的替代方法展开。

- **AI 数据集的内存乱象**：一位用户在拥有 130GB RAM 的情况下加载数据集仍面临内存限制，并获得了关于磁盘存储技术的建议。建议使用 `save_to_disk`、`load_from_disk` 和 `streaming` 标志等选项来缓解该问题。

- **模型小型化之谜**：讨论转向了量化（Quantization），将其作为在适度硬件上运行大型 AI 模型、平衡性能与精度的方法。

- **Git 中的 Graphviz 故障**：尝试在 Hugging Face Spaces 中使用 `graphviz` 的用户遇到了 `PATH` 错误，并分享了修复该问题的系统配置经验。

- **技能而非领域孕育机会**：在技能讨论中，一位用户强调在考虑技术领域的职业机会时，参与项目比特定技术领域更有价值。

- **对 LLM JSON 结构化的兴奋**：一位 Langchain Pydantic Basemodel 用户寻求将文档结构化为 JSON 以避免表格结构混乱的建议，引发了同行的热烈讨论。

- **网络安全策略待命**：随着 6 月 27 日关于机器人和欺诈检测活动的宣布，社区正准备向 hCaptcha 的 ML 总监学习先进策略。

- **分词（Tokenization）讨论引发争议**：[Apehex](https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight) 发表了反对分词的观点，主张直接使用 Unicode 编码。这引发了关于各种编码方法权衡的激烈讨论。

- **可定制地图和媒体友好型起始页**：创意开发者展示了他们的作品，如用于制作风格化城市地图的 [Cityscape Prettifier](https://github.com/C1N-S4/cityscape-prettifier)，以及为媒体爱好者设计的浏览器起始页扩展 [Starty Party](https://startyparty.dev/)。

- **论文领域的进展**：阅读小组的成员寻找并推荐了关于代码基准测试（coding benchmarks）污染等主题的研究，而其他人则暗示即将发布与更新论文相关的代码。

- **视觉工具故障排除**：用户发现 `hf-vision/detection_metrics` 由于依赖问题容易出错，并讨论了 GitHub 上记录的持续性问题，例如[此 issue](https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972) 中提到的问题。

- **寻找表格数据的 LLM 专家**：有人询问是否有开源项目能够对话讨论表格数据中的趋势，而不涉及建模或预测。同时，一位社区成员表示打算为 [关于基于 RoBERTa 的缩放点积注意力的 PR](https://github.com/huggingface/transformers/pull/30510) 做出贡献，尽管面临仓库访问障碍。

- **Gradio 增强聊天机器人和图表**：Gradio v4.37 的发布带来了重新设计的聊天机器人 UI 和动态图表，以及在聊天中嵌套画廊（galleries）和音频等组件的能力。GIF 支持也得到了认可，详见 Gradio 的 [changelog](https://www.gradio.app/changelog)。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **对齐 PyTorch Tensors**：一位用户寻求关于如何在内存中**对齐 PyTorch tensors**的建议，这对于使用 `float2` 高效加载 Tensor 对至关重要，因为这涉及到对齐问题。

- **理解 PyTorch 中的反量化**：在一场热烈的讨论中，工程师们剖析了 `torch.ops.aten._weight_int4pack_mm` 函数，并参考了 [GitHub 源代码](https://github.com/pytorch/pytorch/pull/110914) 以更好地理解反量化（Dequantization）和矩阵乘法，并抱怨缺乏信息丰富的自动生成文档。

- **Quantum Quake 挑战**：一个名为 Q1K3 的 **13kb JavaScript** 重制版 Quake，通过一段 [YouTube 制作视频](https://www.youtube.com/watch?v=F9MLpuvPDio) 展示，同时提供了[游戏试玩地址](https://phoboslab.org/q1k3/)，并在[博客文章](https://phoboslab.org/log/202)中进行了深入讨论。

- **HF 中的生成问题**：讨论突出了 transformers 库中 `HFGenerator` 在缓存后逻辑更新后的问题，促使需要重写以修复在使用 `torch.compile` 时因 Prompt 长度变化导致重新编译的问题。

- **软件与硬件的碰撞**：工程师们分享了一项突破，**Windows 构建的 cuDND 修复已合并**，讨论了在 **H100 上使用 cuDNN** 进行训练时的稳定性挑战，思索了 **AMD GPU 支持**，强调了一个用于**设备端 Reduction** 以限制数据传输的 PR，并讨论了路线图，包括 **Llama 3 支持和 v1.0**，目标是实现滚动检查点（rolling checkpoints）和 **StableAdamW 优化器**等优化。

- **评估 AMD 的未来**：链接了一篇评估 AMD 即将推出的 **MI300x** 性能的文章，表明了对 AMD GPU 发展方向的关注。

- **PyTorch 设备分配调查**：针对 **PyTorch Tensor 设备调用问题**提出了一项技术修复，参考了 `native_functions.yaml` 中的一行，这可能有助于解决 Tensor 中的设备调用不匹配问题。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Llama3-8B 在 100 多种语言中表现强劲**：工程师们正在讨论 [Replete-Coder Llama3-8B 模型](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B)，赞扬其在多种语言中的高级编程能力，以及其避开重复数据的独特数据集。
- **显微镜下的基准测试**：基准测试（Benchmarks）的可靠性引发了辩论，人们认识到基准测试往往无法准确反映实际性能；这意味着需要更全面的评估方法。
- **减轻负载的优化器**：*[Adam-mini 优化器](https://arxiv.org/abs/2406.16793)* 因其在显著降低内存占用和提高吞吐量的同时，能够提供类似 AdamW 的性能而受到关注。
- **Ollama 的致命弱点已修复**：围绕 Ollama 项目中 [CVE-2024-37032 漏洞](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032) 的讨论强调了快速响应以及用户更新到修复版本的紧迫性。
- **GPU 协同工作**：对于那些在使用 Unsloth 时遇到多 GPU 故障的用户，共识是采用实际的变通方法，例如限制 CUDA 设备，相关见解可在 [GitHub issue 660](https://github.com/unslothai/unsloth/issues/660) 中找到，同时模型微调（Fine-tuning）中的挑战正通过模型合并（Model Merging）等新技术得到解决。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **对 Perplexity Pro 功能的困惑**：用户对 **Perplexity AI** 的功能表示担忧，主要问题是 UI 语言会随机从英语切换到其他语言，以及 Pro Search 与标准搜索功能之间的混淆。还有报告称 PRO 订阅用户在生成下载链接时遇到问题，并有人询问 Pro 计划是否包含用于国际内容本地化的 "Pages" 功能。

- **Starliner 的麻烦与地方新闻亮点登上 YouTube**：讨论了一个 [YouTube 视频](https://www.youtube.com/embed/xUsxGDrwzls)，重点介绍了 **Starliner 航天器** 的问题以及 Panthers 队的最新胜利。此外，**Samantha Mostyn** 被任命为澳大利亚新任总督也引起了用户的关注。

- **Perplexity API 未能提供完整输出**：使用 **Perplexity API** 的用户报告称，该 API 在摘要中未能包含引用和图像，建议使用代码块作为权宜之计。

- **寻求 Pro 故障排除**：一位成员对处理紧迫工作却需要 Pro 功能感到失望，并被引导寻求 "f1shy" 的帮助以尝试解决问题。

- **技术内容精选**：提到了用于 **Agentic RAG 的 Jina Reranker v2**，指其具有超快速、多语言 function-calling 和代码搜索能力，这被认为是技术受众的宝贵信息。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**RTX 3090 无法应对高负载**：用户对 **RTX 3090 eGPU 设置** 无法加载较大的模型（如 **Command R (34b) Q4_K_S**）感到沮丧，这引发了关于利用 **exl2 格式** 以提高 VRAM 利用率的建议，尽管目前针对 exl2 的工具和 GUI 选项较少。

**澄清不同 Llama 版本的混淆**：对 **Llama 3 模型变体** 进行了澄清：未标记的 **Llama 3 8B** 是基础模型，与针对特定任务进行微调的 *Llama 3 8B text* 和 *Llama 8B Instruct* 有所区别。

**模型的惊喜与遗憾**：**Hathor_Fractionate-L3-8B-v.05** 的创造力和 **Replete-Coder-Llama3-8B** 的编程能力受到了称赞，而 **DeepSeek Coder V2** 因高 VRAM 需求被标记，**New Dawn 70b** 因其在高达 32k 上下文下的角色扮演能力而受到好评。

**技术支持难题**：LM Studio 中出现了 **Ubuntu 22.04 网络错误** 问题，可能的补救措施包括禁用 **IPv6**，并指出 **LM Studio 目前不支持 Lora 适配器或图像生成**。

**硬件调侃与瓶颈**：一段幽默的交流突显了高性能 GPU 的价格亲民度与其在高级 AI 工作中的必要性之间的巨大鸿沟，老旧设备被嘲讽为属于“19 世纪”。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 音乐生成器面临法律困境**：在 **RIAA** 的协调下，包括 **Sony Music Entertainment** 和 **Universal Music Group** 在内的主要唱片公司已针对 AI 音乐生成器 Suno 和 Udio 发起版权侵权诉讼。社区讨论集中在 AI 训练的伦理问题上，并探讨了创建一个规避版权问题的开源音乐模型的可能性。[Music Business Worldwide 报告](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/)。

- **Carlini 澄清其撰写攻击性论文的初衷**：Nicholas Carlini 发布了一篇 [博客文章](https://nicholas.carlini.com/writing/2024/why-i-attack.html) 回应包括 Ben Zhao 教授在内的批评，为其撰写攻击性研究论文的理由进行辩护，这些论文引发了关于 AI 模型漏洞和社区标准的重要对话。

- **抹除争议内容**：Glaze 频道被删除，引发了关于成本、法律担忧或试图抹除过去争议性言论的猜测，突显了 AI 研究社区中内容审核与自由讨论之间持续存在的紧张关系。

- **Nightshade 的法律迷雾**：名为 Nightshade 的 AI 保护方案在正式发布前被指出存在潜在的法律和伦理风险，反映了社区对部署模型保护措施复杂性的担忧。这些担忧的细节可以在文章《[Nightshade：伪装成艺术家保护措施的法律毒药](https://undeleted.ronsor.com/nightshade-legal-poison/)》中找到。

- **关于模型投毒的争议**：围绕 Zhao 教授支持将模型投毒（model poisoning）作为一种合法策略的辩论引发了争议，强调了篡改 AI 模型这一分歧性问题以及工程社区内部可能产生的反弹。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

**ChatGPT 应用登陆 macOS**：**ChatGPT 桌面应用**现已在 **macOS** 上可用，通过 Option + Space 快捷键提供便捷访问，并增强了针对电子邮件、屏幕截图和屏幕内容进行对话的功能。详情请访问 [ChatGPT for macOS](https://openai.com/chatgpt/mac/)。

**关于 Token 大小的热烈讨论**：工程师们辩论了包括 **ChatGPT4** 在内的各模型的 Token 上下文窗口大小，其中 **ChatGPT4 为 Plus 用户提供 32,000 个 Token**，**为免费用户提供 8,000 个 Token**，而 **Gemini** 或 **Claude** 等其他模型则提供更大的容量，Claude 达到了 200k Token。

**澄清对 Custom GPT 的误解**：成员们澄清了 **CustomGPT 的文档附件功能**与实际模型训练之间的区别。CustomGPT 不提供跨对话的持久化记忆，而是通过外部文档增强模型的知识。

**GPT 性能问题的报告**：Discord 用户报告了 GPT 在处理大型文档以及从上传文件中提供错误信息的问题，同时还存在性能波动和 JSON 输出困难，突显了对复杂查询和输出进行更好处理的需求。

**AI 芯片与进化突破**：社区对 *[EvolutionaryScale 的 ESM3](https://www.evolutionaryscale.ai/blog/esm3-release)*（模拟再现了 5 亿年的生物进化）以及 *[Sohu 的 AI 芯片](https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw)*（在运行 Transformer 模型方面能够超越当前的 GPU）表现出共同的兴奋。



---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **艺术天赋助力 AI 艺术销售**：具有艺术背景的专业人士在销售 AI 生成艺术方面取得了成功，这说明先进的 Prompting 技巧与现有的艺术基础相结合可能是商业成功的关键。
- **CUDA 与 PyTorch 故障排除**：工程师们在访问 GitHub 仓库时遇到问题，并遇到了与 PyTorch 和 GPU 兼容性相关的 *RuntimeError*，共识建议检查 CUDA 与 PyTorch 版本之间的兼容性。
- **对 Open Model Initiative 的质疑**：Open Model Initiative 在工程师中引发了分歧，尽管得到了 Reddit 等社区的支持，但一些人出于伦理考量对其诚信表示怀疑。
- **对 Google Colab 使用限制的担忧**：由于 Stable Diffusion 的大量使用，用户担心 Google Colab 可能会实施限制，并建议使用 RunPod 等替代方案，其类似用途的成本约为每小时 30 美分。
- **Stability.AI 的未来受到质疑**：如果 Stability.AI 不解决现有问题并在 SD3 等产品中撤销审查，人们对其在竞争激烈的市场中的长久性表示怀疑，这挑战了其当前和未来的市场地位。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **生成式超网络（Hypernetworks）实现 LoRA 化**：关于生成 **Low-Rank Adaptations (LoRAs)** 的超网络讨论浮出水面，这表明了超参数的灵活性，并标志着向更具可定制性的 AI 模型迈进，特别是那些针对秩（rank）为 1 的特定模型。

- **“Nous” 的细微差别**：语言学的碰撞引发了澄清：Nous Research 中的 “Nous” 取自希腊语，意指智慧（intelligence），而非法语中的 “我们（our）”，这凸显了社区中集体激情与智慧的融合。

- **安全警报：Probllama 漏洞曝光**：Twitter 上的热议指出 **Probllama** 存在远程代码执行（RCE）漏洞，详情见[此推文](https://x.com/sagitz_/status/1805261557481312623)，该漏洞目前已被分配编号 **CVE-2024-37032**。

- **通过 Coder Llama3-8B 进入 Llama 宇宙**：[Replete-AI/Replete-Coder-Llama3-8B](https.detoken://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) 强势进入 AI 领域，展示了其在 100 多种编程语言中的实力，并有望凭借其 390 万行精心策划的训练数据重塑编程格局。

- **LLM 研究揭示决策边界**：一篇 [arXiv 论文](https://arxiv.org/abs/2406.11233) 揭示了 LLM 在上下文学习（in-context learning）中具有非平滑且复杂的决策边界，这与决策树（Decision Trees）等传统模型的预期行为形成对比。这项研究为模型的可解释性和优化提供了新的思考。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **新 AI 模型上线 OpenRouter**：OpenRouter 展示了其 2023-2024 年的模型阵容，引入了 [AI21 的 Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct)、[NVIDIA 的 Nemotron-4 340B Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) 以及 [01-ai 的 Yi Large](https://openrouter.ai/models/01-ai/yi-large)。不过，他们也报告了“推荐参数（Recommended Parameters）”标签页数据错误的问题，并向用户保证正在修复中。

- **从游戏到 AI 控制**：开发者 **rudestream** 展示了一个针对 *Elite: Dangerous* 的 AI 集成项目，该项目使用 **OpenRouter 的免费模型** 来实现游戏内飞船电脑的自动化。虽然该项目正受到关注，但开发者正在寻求进一步增强 **Speech-to-Text** 和 **Text-to-Speech** 能力，正如在 [GitHub](https://github.com/RatherRude/Elite-Dangerous-AI-Integration) 和 [演示视频](https://www.youtube.com/watch?v=nvuCwwixvww) 中所展示的那样。

- **测试延迟与 AI 发展反思**：OpenRouter 推迟了一篇公告发布，以便对新的 Jamba 模型进行进一步测试；同时，一位用户引发了关于 AI 创新现状的讨论，建议爱好者们听听 François Chollet 对 [AI 未来的见解](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/)。

- **Jamba Instruct 模型故障与最佳实践**：用户在使用 **AI21 的 Jamba Instruct** 模型时遇到了技术问题；即使在修正了隐私设置后，不一致的情况依然存在。另外，社区交流了 Prompt Engineering 策略，并推荐参考 [Anthropic Claude 的指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)。

- **AI 个性之争真实存在**：关于 Large Language Models (LLMs) 中立性的辩论愈演愈烈，共识倾向于更喜欢限制较少、能进行更多原创和动态对话的 AI，而不是只会复读中立、“文本墙”式回复的 AI。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **字体排印学与 AI 的碰撞：llama.ttf**：工程师们探索了 [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news)，这是一个创新的字体文件，它将大型语言模型与基于文本的 LLM 推理引擎相结合，利用了 HarfBuzz 的 Wasm shaper。这种巧妙的融合引发了关于 AI 在软件开发中非常规用途的讨论。

- **Karpathy 开启 AI 盛会**：[Andrej Karpathy](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 宣布了在旧金山举办的 AI World's Fair（AI 世界博览会），引发了巨大轰动。他强调在活动门票已售罄的情况下仍需要志愿者，这标志着 AI 社区聚会的关注度正在不断攀升。

- **MARS5 TTS 模型突破**：技术社区介绍了 [MARS5 TTS](https://x.com/reach_vb/status/1805336863101620699)，这是一款前卫的开源 Text-to-Speech 模型，承诺提供无与伦比的韵律控制（prosodic control）以及仅需极少音频输入即可实现声音克隆的能力，引发了对其底层架构的兴趣。

- **EvolutionaryScale 的 1.42 亿美元种子轮震撼业界**：[EvolutionaryScale](https://x.com/pdhsu/status/1805563282746446116) 宣布完成高达 1.42 亿美元的巨额融资，用于支持其 ESM3 模型的开发。该模型旨在模拟 5 亿年的蛋白质进化，突显了将 AI 与生物学结合的巨大前景。

- **Sohu 速度震惊 Nvidia**：讨论围绕着 [Sohu](https://x.com/Etched/status/1805625693113663834) 展开，这是目前最新的 AI 芯片，声称在运行 Llama 70B 时每秒可处理 500,000 个 token，超越了 Nvidia 的 Blackwell。这催生了关于基准测试方法论以及这些主张能否在现实场景中立足的辩论。

- **播客畅谈 AI 未来**：**Latent Space** 播客的预告片带来了惊喜，包括 [AIEWF 会议预览](https://x.com/latentspacepod/status/1805468252828844052) 以及关于 **DBRX** 和 **Imbue 70B** 的讨论，这些内容塑造了围绕当前 LLM 格局和创新 AI 媒体内容的辩论 [[点击此处收听](https://www.latent.space/p/llm-training-2024)]。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 巡演动态**：LlamaIndex 团队将参加 AI Engineer World's Fair。@jerryjliu0 将于周三（26日）发表关于“知识助手的未来”的[主旨演讲](https://twitter.com/llama_index/status/1805366745877856363)。千万不要错过！

- **RAG 获得 DSPy 助力**：LlamaIndex 通过与 DSPy 合作增强了 RAG 能力，通过卓越的数据处理优化了 Retriever-Agent 的交互。有关此次增强的完整细节可以在其[公告](https://twitter.com/llama_index/status/1805622004030284036)中找到。

- **解决 PGVectorStore 中的维度难题**：一位用户发现了一个由 bge-small 模型的 Embedding 维度不匹配触发的匹配错误。在正确设置 `embed_dim` 以保持一致性后，该问题已得到解决。

- **RAG 架构揭秘**：分享了关于 RAG 内部机制的资源，引导用户查看有关[概念](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)和 [Agent 工作流](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/)的图表和详细文档，以及一篇关于该主题的基础性[论文](https://arxiv.org/abs/2312.10997)。

- **vllm 的 Prompt 模板潜力**：关于 vllm 中 Prompt 模板的对话澄清了如何使用 `messages_to_prompt` 和 `completion_to_prompt` 函数钩子将 Few-shot Prompting 集成到 LlamaIndex 模块中。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**利用 Git 日志高效查看变更日志**：工程师们发现使用 "git log -S" 可以搜索特定代码更改的历史记录，这在查阅 [Mojo 变更日志](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md)时非常有价值，尤其是因为文档重建会消除三个月以上的可搜索历史。

**Mojo 与 MAX 互连的潜力**：讨论表明，虽然 Mojo 目前可能不支持与 Torch 轻松地同时使用，但未来的集成旨在同时发挥 Python 和 C++ 的能力。此外，对于 AI 模型服务，MAX graph API 的 Serde（序列化/反序列化）正在开发中，承诺未来将支持 Triton 等框架的自定义 AI 模型。

**MAX 24.4 拥抱 MacOS 与本地 AI**：随着 [MAX 24.4 的发布](https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support)，MacOS 用户现在可以利用该工具链构建和部署生成式 AI 流水线，并引入了对 Llama3 等本地模型的支持以及原生量化。

**Mojo 的 SIMD 与向量化热点话题**：工程师们正在研究 Mojo 中的 SIMD 和向量化，其中手写 SIMD、LLVM 的循环向量化器状态以及 SVE 支持等特性成为关键考量。这些讨论促成了提交功能需求或 PR 的建议，以便更好地对齐 SIMD 标准。

**Nightly 编译器更新驱动 Mojo 优化**：Mojo Nightly 版本 `2024.6.2505` 和 `2024.6.2516` 带来了大量问题修复和增强，重点强调了通过列表自动解引用（list autodereferencing）和字典中更好的引用处理带来的性能提升。故障排除亮点包括处理编译时布尔表达式，并引用了[特定提交](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5)。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LingOly 基准测试受到审查**：工程师们讨论了 [LingOly 基准测试](https://arxiv.org/abs/2406.06196v2)的潜在缺陷，质疑其范围和评分方式，特别是当测试集公开时存在的记忆化风险。
- **庆祝伦理 AI 制造者的崛起**：社区认可了 Mozilla 的 [Rise25 奖项](https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards)，表彰获奖者在伦理和包容性 AI 方面做出的贡献。
- **MoE 在参数扩展中的优势**：专家混合模型（MoE）中的稀疏参数成为首选的扩展路径，这对深化架构提出了挑战。
- **联邦学习与 AI 中的后门威胁**：讨论集中在联邦学习中潜在的对抗性后门攻击及其对 Open Weights 模型的影响，参考了[这篇论文](https://arxiv.org/abs/2206.10341)的研究。
- **强调 AI 中初始化的重要性**：一位成员在讨论神经网络中被低估的初始化结构作用时引用了 *"Neural Redshift: Random Networks are not Random Functions"*，并推荐阅读 [AI 公案（AI koans）](http://www.catb.org/esr/jargon/html/koans.html)以增加趣味性。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 欢迎 Multi App**：[Multi 宣布](https://multi.app/blog/multi-is-joining-openai)将加入 OpenAI，旨在探索人类与 AI 之间的协作工作，服务将提供至 2024 年 7 月 24 日，并详细说明了终止后的数据删除计划。
  
- **苹果选择 ChatGPT 而非 Llama**：[苹果拒绝了 Meta 的 AI 合作伙伴提议](https://archive.is/uUv1L)，转而倾向于与 OpenAI 的 ChatGPT 和 Alphabet 的 Gemini 结盟，主要原因是担心 Meta 的隐私实践。

- **Rabbithole 的硬编码密钥风险**：rabbitude 的[一次代码库安全泄露](https://rabbitu.de/articles/security-disclosure-1)暴露了硬编码的 API 密钥，存在未经授权访问包括 ElevenLabs 和 Google Maps 在内的多种服务的风险，并引发了关于潜在滥用的讨论。

- **Nvidia 的现状被打破**：[市场转变反映出一种认知](https://www.ft.com/content/7332b1f8-cf7c-4bfa-82f4-88d0deb23f98)，即 Nvidia 并非 GPU 领域的唯一巨头；Imbue AI 发布的一个针对 70B 参数模型的工具包受到了质疑与关注。

- **AI 实验室安全亟需关注**：对 Alexandr Wang 的采访见解强调了 AI 实验室加强安全性的紧迫需求，暗示 AI 可能通过“超人类黑客攻击（superhuman hacking）”等途径带来比核武器更显著的风险。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

**Llama3-8B Coder AI 震撼社区**：[Replete-Coder-Llama3-8B 模型](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B)凭借其精通 100 多种语言和先进的代码编写能力给工程师留下了深刻印象，尽管它并非为 Vision 任务量身定制。

**技术突破与怪异问题并存**：工程师们在解决了一些标志位故障后，成功使用 `claude-3-5-sonnet-20240620` 进行代码执行，但兼容性和函数支持问题表明需要更精细的模型配置。

**Vision 功能的挫败感依然存在**：尽管付出了协同努力，像 daniel_farinax 这样的用户在本地使用 Vision 功能时仍面临处理时间缓慢和 CUDA 显存错误的问题，凸显了模拟 OpenAI 的 Vision 函数的成本和复杂性。

**有限的本地 Vision 功能引发讨论**：用户尝试激活 `--local --vision` 等 Vision 特性，但收效甚微，揭示了 Llama3 能力的差距以及对更易用、更高效的本地 Vision 任务执行的需求。

**单条 AI 内容侧记**：关于 AI 生成视频令人不安的一条评论暗示了用户 m.0861 的潜在担忧，尽管这并未在工程社区内扩展成更广泛的讨论。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **ChatOllama 处理流程简化**：实验 **Ollama** 的工程师可以使用一个实验性的封装器，使其 API 与 OpenAI Functions 保持一致，如[此笔记本](https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/)所示。为了高效地向聊天机器人添加知识，工程师建议使用向量数据库的 `add_documents` 方法配合 **FAISS** 进行索引，而无需进行完整的重新处理。

- **异步 API 难题**：成员们讨论了如何处理对 OpenAI 的 **ChatCompletion** 端点的并发请求，需要一种异步解决方案来同时通知多个用户，这与 **GPT-4** 的批量请求有所不同。

- **提升流式传输性能**：为了优化 **Ollama** 的响应时间，建议用户导入 `ChatOllama` 并利用其 `.stream("query")` 方法，这是推荐用于加快基于 Token 输出的技巧。

- **长期记忆方案**：[Zep](https://www.getzep.com/) 被讨论为 AI 长期记忆的潜在解决方案，它与 LangChain 集成以维护持久的对话摘要并有效地保留关键事实。

- **展示 AI 健身与商业洞察**：*Valkyrie* 项目在 AI 私人教练中融合了 **NVIDIA, LangChain, LangGraph, 和 LangSmith 工具**，详情见 [GitHub](https://github.com/pannaf/valkyrie)。另一项创新亮点是一个 Python 脚本，用于抓取 Instagram 上的肯塔基州商业线索，并附带 [Google Sheet](https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022) 数据和用于 Visual Agents 中 Lambda 集成的 [YouTube 教程](https://youtu.be/3xZlvR3aPQI)。

- **框架适配还是徒劳**：在一个 [YouTube 视频](https://youtu.be/uG0cs8AlnHw)中总结了将 AI 框架集成到应用中的决策过程，剖析了 **GPT-4o, Gemini, Claude, 和 Mistral** 的关键特性，以及 **LangChain** 等设置在开发工作流中的作用。



---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Claude-3.5-Sonnet 热度消退**：关于 **Claude-3.5-Sonnet** 的猜测有所减少，内部人士确认缺乏关于其开发的特权信息，仅指向公开可用的细节。
  
- **Cohere 对 Rerank 模型统计数据保密**：Cohere 对其 Rerank 模型的参数规模保持沉默，尽管社区成员多次询问，仍未透露相关信息。

- **全球 AI 人才集结**：**[Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home)** 正式发布，这是由 Cohere 举办的为期六周的活动，旨在促进全球合作构建多语言 AI 模型，并为参与者提供 API 额度和奖品。

- **Preambles 受到关注**：通过讨论和资源共享，Cohere 的 **[Command R 默认 Preamble](https://docs.cohere.com/docs/preambles)** 变得更加清晰，揭示了它如何塑造模型交互和预期。

- **关注 Cohere 开发者谈话**：*Cohere Developer Office Hours* 鼓励热心的开发者深入研究 Command R+ 的功能，并邀请通过以下 [Discord 邀请链接](https://discord.gg/CsqyQFhEXT?event=1248301309233336350)加入对话。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **发现 Tinygrad "LazyBuffer" Bug**：用户在 [tinygrad Tensor 库](https://github.com/tinygrad/tinygrad/issues/5134)中发现了一个 `'LazyBuffer' object has no attribute 'srcs'` 错误；George Hotz 承认了 `lazy.py` 中的该 Bug，并表示需要进行彻底测试和修复。

- **提出 Clip() 变通方案**：针对 "LazyBuffer" Bug，有人提出在 tinygrad 中使用 `.clip()` 时，用 `.contiguous()` 替代 `realize`，这一调整避开了该问题。

- **使用 Docker 进行 CI 调试**：为了解决 Mac 上的 CI 差异，一名成员建议通过 [Docker 使用 Linux 环境](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1)，这种方法在解决类似问题方面已有先例。

- **征集 Qualcomm 驱动程序的悬赏任务**：目前有一个 700 美元的悬赏，用于开发 Qualcomm GPU 驱动程序，讨论详情参考了某条 [推文](https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19)，建议参考 `ops_amd.py` 进行指导，并使用安装了 Termux 和 tinygrad 的 Android 手机进行环境搭建。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **对多模态模型的期待**：成员们担心 *LLM3 多模态* 可能会在 720 亿参数模型于 7 月中旬完成训练之前发布。该模型训练大约需要 20 天，每个 Epoch 持续 5 天。
  
- **使用 Adam-mini 提升优化效率**：arXiv 上的 [Adam-mini 优化器](https://arxiv.org/abs/2406.16793) 论文引起了成员们的关注，该优化器通过减少独立学习率的数量，与 AdamW 相比可减少 45% 到 50% 的内存占用。

- **Hugging Face 上的自定义 LR 调度器**：一位用户寻求关于使用 Hugging Face 创建余弦学习率 (LR) 调度器的建议，希望实现一个大于零的最小 LR 以微调模型训练。

- **使用 Python 加速 Minhash 计算**：一名成员声称使用 Python 将 Minhash 计算性能提升了 12 倍，引发了广泛关注，并邀请大家提供协作反馈以进一步改进此优化。

以上是 **OpenAccess AI Collective** 内部最受关注的讨论和技术热点。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 上的 Tokenizer 争议**：**Phi-3-mini** 和 **Phi-3-medium** 之间的 Tokenizer 配置差异可能会影响 Torchtune 的性能，前者包含起始符 Token (`"add_bos_token": true`)，而后者则不包含 (`"add_bos_token": false`)。
- **TransformerDecoder 故障排除**：工程师在 `TransformerDecoder` 参数（如 `attn.q_proj.weight`）中遇到了运行时尺寸不匹配错误，这表明 **Phi-3-Medium-4K-Instruct** 的配置或实现可能存在问题。
- **Phi-3-Medium-4K-Instruct 兼容性困境**：持续出现的错误表明 **Torchtune** 对 **Phi-3-Medium-4K-Instruct** 的支持尚不完整，需要额外的调整才能实现完全兼容。
- **构建自定义 Tokenizer 解决方案**：为了解决 Tokenizer 的差异，成员们提议通过调整 `phi3_mini_tokenizer` 配置并设置 `add_bos = False` 来创建一个专门的 `phi3_medium_tokenizer`。

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Beowulf 的重大速度突破**：一名成员宣布了 **beowulfbr** 效率工具的显著 **速度提升**，使其比 datasketch 快了 **12 倍**。

- **Simon 表示：“精简你的命令！”**：**Simon Willison** 分享了他关于将 **Large Language Models 与命令行界面 (command-line interfaces)** 集成的演讲，其中包括一段 [YouTube 视频](https://www.youtube.com/watch?v=QUXQNi6jQ30) 和他 [演讲的注释版本](https://simonwillison.net/tags/annotatedtalks/)。

- **创新的数据集生成方法揭晓**：一种用于 LLM 指令微调 (instruction finetuning) 的高质量数据集生成新方法受到关注。该方法被描述为全自动、无需种子问题 (seed questions) 且可在本地运行，详细信息见链接中的 [帖子](https://x.com/rasbt/status/1805217026161401984)。

- **使用 Linus Lee 的 Prism 进行合成孔径编码 (Synthetic Aperture Encoding)**：该公会讨论了 **Linus Lee** 在 **Prism** 微调方面的工作，对他创建更具人类可解释性模型的方法表示关注，详见其博客 [文章](https://thesephist.com/posts/prism/)。

- **私有模型，Gradio 故障**：一名成员在尝试通过 **AutoTune** 使用私有微调模型创建 **Gradio space** 时遇到错误，由于模型的私有状态，需要提供 `hf_token`。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Llamafile v0.8.7 上线**：[llamafile v0.8.7](https://www.phoronix.com/news/Llamafile-0.8.7-Released) 的发布引入了 *更快的量化操作 (quant operations)* 和错误修复，并暗示了即将到来的 Android 兼容性。
  
- **准备迎接七月的 AI 演讲和工具**：本月将有两场关键活动：[Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134) 和 [Sentry.io 的 AutoFix](https://discord.com/events/1089876418936180786/1245836053458190438)，以及 [AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249)，旨在吸引社区参与。
  
- **Mozilla AI 亮相会议巡展**：成员们将在 [World's Fair of AI](https://www.ai.engineer/worldsfair) 进行演讲，并在 [AI Quality Conference](https://www.aiqualityconference.com/) 担任主持人；同时 Firefox Nightly 正在开辟新路径，其 [Nightly 博客](https://discord.com/channels/1089876418936180786/1254858795998384239) 详细介绍了可选的 AI 服务。

- **阅读最新的 ML 论文精选**：精选的近期机器学习研究现已发布，提供了来自社区的见解和讨论。

- **提升 Llamafile 的新用户体验**：有建议提出为初学者提供分步的 **llamafile 和配置指南**，并且关于 Firefox 可能集成内置本地推理 (local inference) 功能以简化设备端推理的讨论正在进行中。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Racy AI 进入 Beta 阶段**：**AI 生成成人内容平台** [Honeybot.ai](https://honeybot.ai/) 宣布开始其 Beta 阶段，并表示该服务对 18 岁以上人士免费。

- **项目活跃度受到质疑**：一位用户对某个项目的活跃状态提出担忧，指出垃圾信息 (spam) 的泛滥可能表明该项目已不再活跃。



---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **机器人战场：检测数字欺骗者**：一场名为 *“LLM 时代的机器人与欺诈检测”* 的即将举行的活动将揭示识别和减轻基于 LLM 的机器人在自动化和安全领域影响的策略。该讨论定于 2023 年 6 月 27 日举行，将探讨机器人的演变以及专家目前使用的检测方法。

- **结识 AI 哨兵 – Unmesh Kurup**：随着复杂的 LLM 日益普及，领导 Intuition Machines/hCaptcha 的 ML 团队的 **Unmesh Kurup** 将担任此次数字活动的主讲嘉宾，剖析用于区分机器人和人类交互的高级安全系统。该领域的工程师和专家可以[免费注册](https://lu.ma/y6hz8cod?utm_source=diskord)，以从 Kurup 在 AI/ML 领域的丰富经验中获取见解。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将予以移除。

---

# 第二部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1255261278994305024)** (1 条消息): 

```html
- **Argilla 2.0 提升数据集标注**：[Argilla 2.0](https://x.com/argilla_io/status/1805250218184560772) 发布，配备了用于数据集集成的全新 Python SDK 和用于数据标注的灵活 UI。此次更新承诺“更高效地创建高质量数据集”。
- **微软 Florence 模型横扫基准测试**：微软发布了 [Florence](https://x.com/osanseviero/status/1803324863492350208)，这是一个用于字幕生成（captioning）和 OCR 等任务的视觉模型，模型大小为 200M 和 800M，采用 MIT-licensed。通过 DocVQA 数据集上的新 [notebook 和教程](https://x.com/mervenoyann/status/1805265940134654424)，“*在任何任务上微调 Florence-2*”。
- **秒级生成 GGUF 量化**：新增[支持](https://x.com/reach_vb/status/1804615756568748537)“在不到 120 秒内生成 GGUF 量化”，包括自动上传到 Hub 以及对私有和组织仓库的支持。已创建超过 3500 个模型检查点。
- **AWS 嵌入模型指南**：一份关于如何使用 Sentence Transformers 在 AWS SageMaker 上[训练和部署嵌入模型](https://www.philschmid.de/sagemaker-train-deploy-embedding-models)并针对金融数据微调 BGE 模型的综合指南。在 ml.g5.xlarge 实例上训练耗时约 10 分钟，成本约为 $0.2。
- **关于数据质量的伦理与社会简报**：最新的[伦理与社会简报](https://huggingface.co/blog/ethics-soc-6)强调了数据质量的重要性。与伦理专家的合作促成了对这一关键主题的详细讨论。
```
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/argilla_io/status/1805250218184560772)">来自 Argilla (@argilla_io) 的推文</a>：📢 另一个重大发布：Argilla 2.0 rc！这对 AI 构建者意味着什么？ 🤺 统一的反馈收集框架 🐍 用于处理数据集的新 Python SDK，包括一个新的 @huggingface da...</li><li><a href="https://x.com/osanseviero/status/1803324863492350208)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Microsoft 刚刚悄悄发布了 Florence 👀 视觉模型，可以处理许多视觉任务（字幕、检测、区域建议、OCR） 🤏 小型模型（200M 和 800M），其质量媲美比其大 100 倍的模型...</li><li><a href="https://x.com/mervenoyann/status/1805265940134654424)">来自 merve (@mervenoyann) 的推文</a>：在任何任务上微调 Florence-2 🔥 今天我们发布了一个 Notebook 和一篇关于在 DocVQA 数据集上微调 Florence-2 的实战博客 @andi_marafioti @skalskip92 继续阅读 ⇓</li><li><a href="https://x.com/reach_vb/status/1804615756568748537)">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：在不到 120 秒内生成 GGUF 量化！⚡ &gt; 增加了对 imatrix 量化的支持 &gt; 对较大量化支持 GGUF-split &gt; 自动上传到 Hub &gt; 支持私有和组织仓库 U...</li><li><a href="https://x.com/osanseviero/status/1804136001465442530)">来自 Omar Sanseviero (@osanseviero) 的推文</a>：Microsoft 刚刚（再次！）悄悄发布了 Instruction Pre-Training！ 👀 通过生成指令来增强预训练数据集 🦙 一个性能可与 70B 媲美的 Llama 3 8B！ 🔥 通用+领域模型 (m...</li><li><a href="https://x.com/vanstriendaniel/status/1804078257488495099)">来自 Daniel van Strien (@vanstriendaniel) 的推文</a>：Instruction Pre-Training 是一种新方法，通过使用来自指令合成器的指令-响应对而不是原始数据来增强 LLM 预训练。 在这个 @gradio S... 中探索这种方法。</li><li><a href="https://x.com/danieldekok/status/1804224598721830954)">来自 Daniël de Kok (@danieldekok) 的推文</a>：🐬 更多 Marlin 功能将进入下一个 @huggingface TGI 版本：支持将现有的 GPTQ 量化模型与快速的 Marlin 矩阵乘法 Kernel 结合使用。 ⚡ 这一功能得以实现...</li><li><a href="https://x.com/eustachelb/status/1805262952913858919)">来自 Eustache Le Bihan (@eustachelb) 的推文</a>：Distil-Whisper 迈向多语言！！ 🤗 Whisper 的法语蒸馏版来了！ 🇫🇷 与 large-v3 一样准确，比 tiny 更快。两全其美！ 🚀 查看下方详情 ⬇️</li><li><a href="https://x.com/_philschmid/status/1805593591223398832)">来自 Philipp Schmid (@_philschmid) 的推文</a>：Embedding 模型对于成功的 RAG 应用至关重要，但它们通常是在通用知识上训练的！很高兴分享一份关于如何训练和部署开源 Embedding 模型... 的端到端指南。</li><li><a href="https://x.com/FrG_FM/status/1803703761119871122)">来自 F-G Fernandez (@FrG_FM) 的推文</a>：Xavier 和 @osanseviero 在 @linuxfoundation 的 #AIDev 上展示了 @huggingface 的机器人计划 🤗（包括由 @RemiCadene 领导的 LeRobot） 期待那一天...</li><li><a href="https://x.com/RisingSayak/status/1805521415543697582)">来自 Sayak Paul (@RisingSayak) 的推文</a>：你是否知道我们有一份关于不同提示机制以提高图像生成质量的专门指南？ 🧨 带你了解简单的 Prompt Engineering、提示词权重、提示词增强...</li><li><a href="https://x.com/evijitghosh/status/1805312283628761446)">来自 Avijit Ghosh (@evijitghosh) 的推文</a>：季度性的 @huggingface 伦理与社会简报发布了！很高兴能与 @frimelle 合作，并得到伦理常驻人员的支持。本季度简报的主题是...
</li>
</ul>

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1254876632255172749)** (436 条消息🔥🔥🔥): 

- **浮点数的乐趣**：用户讨论了 Python 中 float 与 integer 类型的实用性，并进行了多次代码迭代来处理大型浮点数计算（例如 `pi**pi**pi`）。一位用户指出在使用 `math.pow` 时的一个常见问题：*"OverflowError: (34, 'Result too large')*"。

- **AI 模型的 RAM 故障排除**：尽管拥有 128GB 的 RAM，一位用户在加载数据集时仍面临内存溢出的困扰。提出的解决方案包括使用 `save_to_disk`、`load_from_disk` 以及启用 `streaming=True`。

- **关于 Quantization 的讨论**：成员们解释了 Quantization 是一种在低端硬件上运行大型 AI 模型的方法。它降低了模型参数的精度，虽然可能会影响性能，但允许模型在内存限制内运行。

- **Git 使用问题**：用户讨论了在 Hugging Face spaces 上使用 `graphviz` 相关的效率低下和错误，排查了关于缺少可执行文件的错误并提出了潜在的修复方案。一个有效的解决方案是确认 graphviz 是否已正确添加到系统的 `PATH` 中。

- **职业与学习路径建议**：用户讨论了哪些技术技能最容易就业，辩论了网络安全与数据科学等领域。给出的建议是：*"比起任何特定领域，参与实际项目会非常有帮助。"*

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>: 未找到描述</li><li><a href="https://llava-vl.github.io/">LLaVA</a>: 未找到描述</li><li><a href="https://huggingface.co/Azazelle/L3-RP_io/tree/main">Azazelle/L3-RP_io at main</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=eId6K8d0v6o">如何在 Windows 11 上安装 WSL2 (Windows Subsystem for Linux)</a>: 在本视频教程中，我们将向您展示如何在 Windows 11 上安装 WSL2，让您直接在 Windows 环境中运行 Linux 命令和工具...</li><li><a href="https://huggingface.co/Azaz">azaz (Z)</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/nroggendorff/357091156426242">Hugging Face 上的 @nroggendorff: "@osanseviero 该你出招了"</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/pluh-veilbound-gif-4315180366045476816">Pluh Veilbound GIF - Pluh Veilbound - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/wat-gif-5001440307614895833">Wat GIF - Wat - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=-4Oso9-9KTQ">ChatGPT 完全解析。</a>: 使用代码 KYLEHILL16 在 HelloFresh 享受 16 顿免费餐食加免运费。ChatGPT 现已成为人类历史上增长最快的消费级应用...</li><li><a href="https://tenor.com/view/white-cat-eating-salad-meme-blonde-woman-crying-gif-26971383">白猫吃沙拉 GIF - 白猫吃沙拉表情包 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://untitled.stream/library/track/JcLqx6Kwgwn39scJDdMuX">SHITLORD (POOPMASTER DISS)</a>: mikusss 在 [untitled] 上的轨道</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch#the-big-table-of-tasks">transformers/examples/pytorch at main · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的先进机器学习库。 - huggingface/transformers</li><li><a href="https://tenor.com/view/old-man-frustrated-dog-gif-15305478">老人沮丧 GIF - 老人沮丧的狗 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://untitled.stream/library/track/w5TYpULbkmTPsnWPLQ5WH">Altman</a>: mikusss 在 [untitled] 上的轨道</li><li><a href="https://github.com/beowolx/rensa">GitHub - beowolx/rensa: Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大型数据集的高效相似度估计和去重</a>: Rust 编写的高性能 MinHash 实现，带有 Python 绑定，用于大型数据集的高效相似度估计和去重 - beowolx/rensa</li><li><a href="https://www.pythonanywhere.com/">在云端托管、运行和编写 Python 代码: PythonAnywhere</a>: 未找到描述</li><li><a href="https://wiki.civitai.com/wiki/Civitai_API#GET_/api/v1/models-versions/by-hash/:hash">Civitai API - Civitai Wiki</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1255059213109956760)** (2 条消息): 

```html
- **Challenges with Langchain Pydantic and LLM**: 一位成员正尝试使用 **Langchain Pydantic Basemodel** 将文档数据结构化为包含额外见解的 JSON。由于表格结构导致 LLM 对数据产生误解，他们正面临问题，并寻求评估策略或更好的方法。

- **Expression of Interest in the Topic**: 另一位成员通过表示“我感兴趣...”来表达了他们对该话题的兴趣。
```
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1254913112554078309)** (3 条消息): 

- **参加机器人与欺诈检测活动**：一场关于“LLM 时代的机器人与欺诈检测”的活动定于 2023 年 6 月 27 日上午 10 点（PDT）举行。主讲人 Unmesh Kurup（Intuition Machines/hCaptcha 的 ML 总监）将分享关于高级检测策略的见解（[在此注册](https://lu.ma/y6hz8cod?utm_source=diskord)）。

- **在 HuggingFace 上查看 T2V-Turbo**：一位成员分享了 HuggingFace Spaces 上的 [T2V-Turbo](https://huggingface.co/spaces/TIGER-Lab/T2V-Turbo) 链接。他们指出它提供了清新且令人印象深刻的体验。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/TIGER-Lab/T2V-Turbo">T2V Turbo - a Hugging Face Space by TIGER-Lab</a>：未找到描述</li><li><a href="https://lu.ma/y6hz8cod?utm_source=diskord">A Million Turing Tests per Second: Detecting bots and fraud in the time of LLMs · Luma</a>：Data Phoenix 团队邀请您参加我们即将举行的网络研讨会，时间为 6 月 27 日上午 10 点（PDT）。主题：每秒百万次图灵测试：…
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1255017509107535893)** (159 条消息 🔥🔥): 

- **[Apehex](https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight) 认为 Tokenization 不切实际**：Apehex 在文章中辩称 Tokenization 方法是低效的，并建议使用神经网络直接对 Unicode 字符序列进行编码。随后引发了详细讨论，涵盖了 embedding、模型大小以及浮点精度可能存在的问题等技术方面。
- **使用 [Cityscape Prettifier](https://github.com/C1N-S4/cityscape-prettifier) 个性化城市地图**：Deuz_ai_80619 分享了一个 GitHub 项目，允许用户使用 Flask、Prettymaps 和 Python 创建精美的个性化城市地图，将 OpenStreetMap 数据转化为时尚的可视化效果。
- **面向媒体爱好者的起始页扩展 [Starty Party](https://startyparty.dev/)**：Desmosthenes 介绍了一个专注于媒体和内容的新浏览器起始页扩展，可在 [marketing.startyparty.dev](https://marketing.startyparty.dev) 安装。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/apehex/tokenization-is-a-dead-weight">Is Tokenization Necessary?</a>：未找到描述</li><li><a href="https://github.com/C1N-S4/cityscape-prettifier">GitHub - C1N-S4/cityscape-prettifier: Make beautiful, personalized city maps with ease. With Flask, Prettymaps, and Python, turn OpenStreetMap data into elegant, stylish visualizations. Ideal for designers, developers, and urban aficionados.</a>：轻松制作精美、个性化的城市地图。通过 Flask、Prettymaps 和 Python，将 OpenStreetMap 数据转化为优雅、时尚的可视化效果。非常适合设计师、开发者和城市爱好者。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1255094329580585073)** (4 条消息): 

- **寻找关于数据污染（contamination）的论文**：一名成员请求推荐关于数据污染的论文，特别是针对编程基准测试（coding benchmarks）的。他们分享了阅读清单中的三篇相关论文：[按月份标注测试集](https://arxiv.org/abs/2403.07974)、[ML 基准测试中的普遍饱和现象](https://www.semanticscholar.org/paper/Mapping-global-dynamics-of-benchmark-creation-and-Ott-Barbosa-Silva/43ae69101c302628b9f7186ec5f35f99bb89d5d6)以及[编程基准测试的鲁棒性](https://arxiv.org/abs/2212.10264)。

- **探索使用 Hilbert curve 进行 2D 到 1D 的转换**：一名成员询问关于使用 Hilbert curve 将 2D 图像扫描为 1D 的建议。他们指出其优势在于没有跳跃，并且在不同尺寸的正方形图像上表现良好。

- **对无序路径信息丢失的担忧**：另一名成员提醒，所建议的使用 Hilbert curve 的方法可能会导致信息丢失。他们认为无序路径并不合理，并提到在另一个平台上有一个后续问题。

- **论文更新和代码发布**：一名成员宣布他们正准备更新论文，并将在未来几天内发布代码。这标志着向公开分享研究成果迈进了一步。
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1255204935075758101)** (3 条消息): 

- **`hf-vision/detection_metrics` 受到依赖项问题的困扰**：一名成员在尝试于 `evaluate` 中使用 `hf-vision/detection_metrics` 时遇到错误，由于缺少依赖项导致 `ImportError`。他们指出该包似乎不存在，或者他们遗漏了某些内容。

- **检测指标功能被标记为存在问题**：同一名成员指出，`hf-vision/detection_metrics` 的问题已在 Hugging Face 的 GitHub issues 中记录，具体见[此 issue 评论](https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972)。

- **`evaluate` 无法定位 `detection_util`**：发现 `evaluate` 无法找到 `detection_util`，因为它位于 Space 内的一个文件夹中，这导致该工具无法正常运行。

**提到的链接**：<a href="https://github.com/huggingface/evaluate/issues/111#issuecomment-1855872972.">Add COCO evaluation metrics · Issue #111 · huggingface/evaluate</a>：我目前正致力于将 Facebook AI 的 DETR 模型（使用 Transformers 的端到端目标检测）添加到 HuggingFace Transformers 中。该模型运行良好，但在评估方面，我...

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1255054037364244551)** (3 条消息): 

- **关于用于表格数据交互的 LLM 的咨询**：一名成员询问是否有专门针对表格数据（特别是存储为 CSV 或 pandas DataFrames）进行推理的开源 **LLM 项目**或产品。他们有兴趣通过聊天机器人询问关于数据趋势的问题，而无需进行建模或预测。

- **有兴趣为 GitHub PR 做出贡献**：一名成员表示有兴趣帮助处理一个[与 RoBERTa 系列模型相关的 PR](https://github.com/huggingface/transformers/pull/30510)，重点是添加对 Scaled Dot Product Attention (SDPA) 的支持。由于缺乏对原始仓库的访问权限，他们遇到了问题并寻求如何贡献的建议。


**提到的链接**：<a href="https://github.com/huggingface/transformers/pull/30510">[RoBERTa-based] Add support for sdpa by hackyon · Pull Request #30510 · huggingface/transformers</a>：这个 PR 做了什么？为 RoBERTa 系列模型添加 SDPA (scaled dot product attention) 支持。更多背景见 #28005 和 #28802。在提交之前，此 PR 修复了一个拼写错误或改进了文档...

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1255207180878221352)** (1 条消息): 

- **Gradio v4.37 发布**：最新版本 Gradio v4.37 具有**重新设计的 chatbot UI**、动态图表和 GIF 支持，以及显著的性能提升。它还增强了可定制性并修复了大量 Bug，以提供更流畅的用户体验。
- **宣布令人兴奋的新特性**：**新版 chatbot UI** 支持直接在聊天中嵌入 `gr.Gallery` 和 `gr.Audio` 等组件，同时 `gr.Image` 现在支持 GIF。查看 [Gradio 变更日志](https://www.gradio.app/changelog)了解完整详情。
  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1255191730324766882)** (3 条消息): 

- **需要在内存中对齐 PyTorch tensor**：一位成员询问是否“有任何方法可以强制 PyTorch tensor 按特定字节数进行内存对齐”，以便使用 `float2` 成对加载 tensor。他们正面临对齐方面的问题。
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1254916229521608704)** (11 条消息🔥): 

- **通过源码链接澄清反量化函数**：用户讨论了 `torch.ops.aten._weight_int4pack_mm` 函数，并提供了一个指向 [GitHub 源码](https://github.com/pytorch/pytorch/pull/110914) 的有用链接。该函数执行反量化以及与单位矩阵的矩阵乘法。
- **自动生成函数的文档没有帮助**：用户指出该函数的自动生成文档基本上是空白的，完全没有提供任何信息（*“documentation is blank 🤣”*）。
- **8-bit Adam 协作线程**：启动了一个关于 8-bit Adam 优化的协作线程。关键问题包括动态量化方案的使用，以及反量化和 adam-step 操作是否融合在单个 kernel 中。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pytorch.org/cppdocs/api/function_namespaceat_1adeda9630914278ac02d7fd758da19e3d.html">函数 at::_weight_int4pack_mm &mdash; PyTorch 主要文档</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/110914">Quant: 由 yanboliang 添加 weight int4pack mm kernel · Pull Request #110914 · pytorch/pytorch</a>: 添加 weight int4pack mm CUDA kernel。该 kernel 来自 Jeff Johnson 开发的 tinnygemm 项目。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1255179530121904321)** (3 条消息): 

- **可爱的 Quake 重制版令人印象深刻**：成员们分享了一个名为 [“Q1K3 – Making Of” 的 YouTube 视频](https://www.youtube.com/watch?v=F9MLpuvPDio)，展示了一个为 2021 年 js13kGames 比赛创作的、仅用 **13kb JavaScript** 编写的向 Quake 致敬的作品。他们还提供了[游戏试玩链接](https://phoboslab.org/q1k3/)和一篇[博客文章](https://phoboslab.org/log/202)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=F9MLpuvPDio]">Q1K3 – Making Of</a>: 一个用 13kb JavaScript 编写的向 Quake 致敬的作品，为 2021 年 js13kGames 比赛制作。在此试玩：https://phoboslab.org/q1k3/ 博客文章：https://phoboslab.org/log/202...</li><li><a href="https://www.youtube.com/watch?v=F9MLpuvPDio">Q1K3 – Making Of</a>: 一个用 13kb JavaScript 编写的向 Quake 致敬的作品，为 2021 年 js13kGames 比赛制作。在此试玩：https://phoboslab.org/q1k3/ 博客文章：https://phoboslab.org/log/202...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1255013900718702592)** (4 条消息): 

- **cache 逻辑更改后 HFGenerator 损坏**：一位成员报告了 `HFGenerator` 的问题，指出虽然原生的 `model.generate(input_ids)` 函数运行良好，但自从 transformers 更新了 cache 逻辑后，前者一直存在问题。
- **mobicham 确认需要重写**：一位成员承认了这个问题，表示：*“我需要重写，这周会完成，”* 并提到了在使用 `torch.compile` 时 `model.generate` 可能出现的问题，特别是由于 prompt 长度变化导致的重编译（recompile）行为。
- **在不编译的情况下验证输出**：讨论包括在不使用 `torch.compile` 的情况下验证输出，这表明重点主要在于功能正确性而非性能优化。
- **特定模型的问题和替代方案**：对话转向了特定模型的问题，提到了 Llama2-7B 的潜在问题，并将其与 Llama3 进行了比较，其中提供了 `axis=1` 设置和各种配置供参考：*“Llama3-8b-instruct GPTQ (gs=64): 66.85, AWQ (gs=64): 67.29, HQQ (axis=1, gs=64): 67.4。”*

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1254873802899460218)** (402 条消息🔥🔥): 

- **Windows 构建 cuDNN 修复已合并**：经过一些故障排除，修复 Windows cuDNN 构建中断的补丁已合并 ([#639](https://github.com/karpathy/llm.c/pull/639))。问题与宏重定义有关，需要进行调整，例如添加 `WIN32_LEAN_AND_MEAN`。
  
- **探索 cuDNN 在 H100 上的训练不稳定性**：在 H100 上使用 cuDNN 进行训练时表现出不稳定性，特别是在 bf16 训练中，而关闭 cuDNN 时则不会出现此问题。调查指向 cuDNN flash attention 的 tile 大小可能存在差异。
  
- **关注 AMD GPU 支持**：成员们讨论了加入对 AMD GPU 的支持。目前维护着一个该仓库的 AMD 分支 [anthonix/llm.c](https://github.com/anthonix/llm.c)，但对 AMD GPU 的广泛兴趣仍处于发展阶段。
  
- **设备端归约（on-device reductions）的 PR 正在评审中**：有一个旨在通过将更多计算移至设备端来减少 GPU ↔ CPU 传输的拉取请求 ([#635](https://github.com/karpathy/llm.c/pull/635))。它包括一些微优化，例如避免在验证步骤中进行重复计算。
  
- **关于 Llama 3 支持和 v1.0 路线图的讨论**：计划发布 v1.0 版本，重点是分别处理 GPT-2/3 支持，并在后续版本中引入 Llama 3。**滚动检查点（rolling checkpoints）** 和 **StableAdamW 优化器** 的 PR 是关键组成部分 ([#636](https://github.com/karpathy/llm.c/pull/636))。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpath">karpath - 概览</a>：GitHub 是 karpath 构建软件的地方。</li><li><a href="https://github.com/karpathy/llm.c/pull/640">由 chinthysl 提交的 PR #640：添加缺失的 MULTI_GPU 编译器标志</a>：用于运行文件共享的 NO_USE_MPI 构建。</li><li><a href="https://github.com/anthonix/llm.c">GitHub - anthonix/llm.c：针对 AMD GPU 的简单、原生 C/HIP LLM 训练</a>：针对 AMD GPU 的简单、原生 C/HIP LLM 训练。通过在 GitHub 上创建账号为 anthonix/llm.c 的开发做出贡献。</li><li><a href="https://x.com/Etched/status/1805625693113663834">Etched (@Etched) 的推文</a>：认识 Sohu，有史以来最快的 AI 芯片。Sohu 运行 Llama 70B 每秒超过 500,000 个 token，让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 台 H100。Soh...</li><li><a href="https://github.com/karpathy/llm.c/pull/636">karpathy 提交的滚动检查点 PR #636</a>：检查点分为 MINOR 或 MAJOR，minor 检查点会在滚动窗口中被删除。这是一种优化，允许我们更频繁地保存状态，同时节省总体磁盘空间。...</li><li><a href="https://github.com/karpathy/llm.c/pull/633">chinthysl 提交的 Socket 服务端/客户端接口 PR #633</a>：用于利用 PR #632 中分布式接口的虚拟 PR。</li><li><a href="https://github.com/karpathy/llm.c/pull/637">karpathy 提交的 PR #637：添加离群值检测器、相关测试，并开始跟踪 loss 的 z-score</a>：待办事项：除了 loss 之外还要跟踪梯度范数（是否必须拆分 gpt2_update 函数）；添加 argparse 选项以监控 loss 和梯度范数的离群值（例如 z > 3）；添加不稳定性处理...</li><li><a href="https://github.com/karpathy/llm.c/pull/635">ngc92 提交的设备端归约 PR #635</a>：将 loss 计算移至 backward，并确保我们可以进行更多设备端归约，减少 host <-> device 传输。还启用了一个微优化，即 validate 不再计算 dlogit...</li><li><a href="https://github.com/warner-benjamin/optimi/blob/4542d04a3974bb3ac9baa97f4e417bda0432ad58/optimi/stableadamw.py#L28">warner-benjamin/optimi 中的 stableadamw.py</a>：快速、现代、内存高效且低精度的 PyTorch 优化器 - warner-benjamin/optimi</li><li><a href="https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/">使用 Megatron 将语言模型训练扩展到万亿参数 | NVIDIA 技术博客</a>：随着大规模计算的普及和数据集的增大，自然语言处理 (NLP) 近年来取得了飞速进展。同时，最近的研究表明...</li><li><a href="https://community.flexera.com/t5/FlexNet-Publisher-Forum/FlexNet-11-19-5-build-on-Visual-Studio-2015/m-p/306967">在 Visual Studio 2015 上构建 FlexNet 11.19.5</a>：大家好，我正尝试用 FlexNet 11.19.5 构建我的应用。我遇到了一些编译器问题 (Visual Studio 2015): c:\program files (x86)\windows kits\8.1\include\shared\ws2def.h(100): warning C4005: '...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 条消息): 

iron_bound: https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/
  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1254909421511770227)** (1 条消息): 

- **调试 PyTorch tensor device 调用**: 一位成员通过引用 [native_functions.yaml 中的特定行](https://github.com/pytorch/pytorch/blob/18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c/aten/src/ATen/native/native_functions.yaml#L7680) 讨论了 PyTorch 中 device 调用问题的潜在修复方案。他们建议尝试使用 `BitnetTensor(intermediate).to(device=tensor.device)` 来替换原始代码。

**提到的链接**: <a href="https://github.com/pytorch/pytorch/blob/18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c/aten/src/ATen/native/native_functions.yaml#L7680,">pytorch/aten/src/ATen/native/native_functions.yaml at 18fdc0ae5b9e9e63eafe0b10ab3fc95c1560ae5c · pytorch/pytorch</a>: Python 中的 Tensor 和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1254877097969582150)** (131 messages🔥🔥): 

- **Replete-Coder Llama3-8B 模型亮相**：大家对新的 [Replete-Coder Llama3-8B 模型](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) 进行了热烈讨论，重点强调了其在 100 多种语言中先进的编程能力，以及其经过完全去重（deduplicated）且无审查的训练数据。该模型由 TensorDock 提供云算力租赁支持。
- **对 Benchmark 的质疑**：用户讨论了 Benchmark 的可靠性，强调它们可能会出现过拟合（overfitted），且并不总是能代表真实世界的性能。一位成员表示：“……Benchmark 能告诉你的信息非常有限……它需要针对更大规模的问题范围进行实地考察。”
- **Adam-mini 优化器**：[新的 Adam-mini 优化器](https://arxiv.org/abs/2406.16793) 声称能达到与 AdamW 相当或更好的性能，同时减少 45% 到 50% 的显存占用，并提高 49.6% 的吞吐量（throughput）。这引发了关于潜在实现以及与现有优化器相比的优势讨论。
- **Ollama 项目中的 Probllama 漏洞**：讨论了一个严重的远程代码执行（RCE）漏洞 [CVE-2024-37032](https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032)。该问题已迅速修复，用户强调了升级到最新版本以确保安全的重要性。
- **关于模型训练和 Fine-tuning 的常规讨论**：成员们分享了在模型训练期间优化显存和吞吐量的见解，讨论了不同上下文长度（context lengths）的使用，并表达了对微调 Yi-1.5-34B 等模型的兴趣，同时考虑了 GPU 容量等限制因素。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，它能以减少 45% 到 50% 的显存占用实现与 AdamW 相当或更好的性能。Adam-mini 通过减少学习率的数量来降低显存……</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/faridofanani96/status/1804079517193113850">Mochamad Farido Fanani (@faridofanani96) 的推文</a>：Pangu Model 5.0 发布，首先分为四个系列：Pangu E 系列：10-90 亿参数级，应用于手机和 PC；Pangu P 系列：100 亿-900 亿参数，适用于……</li><li><a href="https://huggingface.co/google-bert/bert-base-uncased">google-bert/bert-base-uncased · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/_clashluke/status/1805522937744597316">Lucas Nestler (@_clashluke) 的推文</a>：@arankomatsuzaki Adam-Mini（左图）在常数因子范围内与 NVIDIA 的 NovoGrad（右图）完全相同。主要区别在于……</li><li><a href="https://x.com/danielhanchen/status/1805608733671833974">Daniel Han (@danielhanchen) 的推文</a>：我今天太平洋标准时间上午 9 点会在旧金山万豪侯爵酒店参加我们的 @aiDotEngineer World's Fair 工作坊！请前往 Golden Gate Ballroom B。如果你在旧金山，很高兴能当面交流！</li><li><a href="https://www.wiz.io/blog/probllama-ollama-vulnerability-cve-2024-37032">Probllama: Ollama Remote Code Execution Vulnerability (CVE-2024-37032) – Overview and Mitigations | Wiz Blog</a>：Wiz Research 发现了 CVE-2024-37032，这是开源 AI 基础设施项目 Ollama 中一个易于利用的远程代码执行漏洞。</li><li><a href="https://ai.meta.com/research/cicero/diplomacy/">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-exl2">bartowski/Replete-Coder-Llama3-8B-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-GGUF">bartowski/Replete-Coder-Llama3-8B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：以 2-5 倍的速度和减少 80% 的显存微调 Llama 3, Mistral, Phi &amp; Gemma LLM - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1254887346701336636)** (251 messages🔥🔥): 

```html
- **Checkpoints and Finetuning**: 建议“使用 save checkpoints 并从 checkpoints 继续进行 finetuning”，并提供了指向 [Unsloth wiki](https://github.com/unslothai/unsloth/wiki) 的链接以获取更详细的说明。
- **Multi GPU Issues**: 用户报告了在多 GPU 设置上运行 Unsloth 时的运行时错误，并讨论了潜在的解决办法，包括限制 CUDA 设备和降级到之前的 Unsloth 版本。分享了 [GitHub issue 660](https://github.com/unslothai/unsloth/issues/660) 的相关链接。
- **Vision Models and OCR**: 讨论了 GPT4o 在 OCR 方面的表现，一些用户对 LLAVA 模型能否达到类似效果持怀疑态度。另一个建议是使用 [openedai-vision](https://github.com/matatonic/openedai-vision)。
- **Experimentation with LLaMA Models**: 用户分享了在微调 "unsloth/Phi-3-mini-4k-instruct" 时的困难和潜在解决方案，以及关于数据集和训练设置的其他问题。建议使用模型合并（model merging）以在 Hugging Face 上获得更好的结果。
- **Training Statistics and Callbacks**: 讨论了如何在训练期间追踪 loss 和其他指标，建议在 Hugging Face 中使用 wandb、TensorBoard 和自定义 callbacks。提供了 [TensorBoardCallback 文档](https://huggingface.co/docs/transformers/main_classes/callback#transformers.integrations.TensorBoardCallback)的链接。
```
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ibm/MoLFormer-XL-both-10pct">ibm/MoLFormer-XL-both-10pct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki">Home</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLM 快 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on">How do I select which GPU to run a job on?</a>: 在多 GPU 计算机中，如何指定 CUDA 任务应该在哪个 GPU 上运行？例如，在安装 CUDA 时，我选择了安装 NVIDIA_CUDA-&amp;lt;#.#&amp;gt;_Samples 然后运行...</li><li><a href="https://huggingface.co/docs/transformers/v4.41.3/en/main_classes/callback#transformers.integrations.TensorBoardCallback">Callbacks</a>: 未找到描述</li><li><a href="https://github.com/matatonic/openedai-vision">GitHub - matatonic/openedai-vision: An OpenAI API compatible API for chat with image input and questions about the images. aka Multimodal.</a>: 一个兼容 OpenAI API 的 API，用于图像输入聊天和图像问题解答。即多模态（Multimodal）。 - matatonic/openedai-vision</li><li><a href="https://github.com/unslothai/unsloth/issues/660">CUDA_VISIBILE_DEVICES not functioning · Issue #660 · unslothai/unsloth</a>: 我在尝试使用 4xA100 GPU 进行监督微调（SFT）时看到了错误信息。所以免费版本不能在多 GPU 上使用吗？RuntimeError: Error: More than 1 GPUs have a lot of VRAM usa...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1254874126611513435)** (182 messages🔥🔥): 

```html
<ul>
    <li><strong>语言切换 Bug 困扰用户</strong>：多位用户报告了一个 Bug，即 Perplexity 的 UI 语言会随机更改为英语以外的其他语言，尽管设置显示为英语。一位用户指出：“它显示是英语，但实际上是西班牙语。”</li>
    <li><strong>Pro Search 功能令用户困惑</strong>：用户询问了 Pro Search 与标准搜索之间的区别，对据称标准用户也可以使用的功能表示困惑。另一位用户希望能够更加明确，并指出新的多步流程感觉更慢。</li>
    <li><strong>文件下载问题困扰 PRO 用户</strong>：一位用户报告了尽管拥有 PRO 订阅，但在为上传的文件生成可访问的下载链接时遇到问题。回复指出 Perplexity 缺乏 “code interpreter”。</li>
    <li><strong>Perplexity Pro 功能受到质疑</strong>：来自巴西的用户遇到了 Perplexity 无法从本地化来源获取搜索结果的问题，而是主要返回英语结果。一位来自阿根廷的用户询问订阅 Pro 计划是否能解锁 “Pages” 功能。</li>
    <li><strong>API 摘要功能表现不佳</strong>：一位使用 Perplexity API 的用户注意到它未能返回引用和图像。另一位用户建议要求 Perplexity 创建代码块，作为生成文档的变通方法。</li>
</ul>
```

**提到的链接**: <a href="https://jina.ai/news/jina-reranker-v2-for-agentic-rag-ultra-fast-multilingual-function-calling-and-code-search/">Jina Reranker v2 for Agentic RAG: Ultra-Fast, Multilingual, Function-Calling &amp; Code Search</a>：Jina Reranker v2 是专为 Agentic RAG 构建的一流重排序器。它具有 Function-Calling 支持、超过 100 种语言的多语言检索、代码搜索功能，并提供 6 倍的速...

  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1254897118095736956)** (8 messages🔥): 

<ul>
<li><strong>Starliner 面临危机</strong>：Perplexity AI 重点介绍了一个 <a href="https://www.youtube.com/embed/xUsxGDrwzls">YouTube 视频</a>，涵盖了一系列话题，包括 Starliner 航天器的问题、Apple AI 在欧洲的延迟以及 OpenAI 的一项重大收购。</li>
<li><strong>Panthers 获胜</strong>：一位用户分享了 Perplexity AI 的链接，展示了 Panthers 最近的胜利，并承诺提供有关该事件的更多细节。<a href="https://www.perplexity.ai/search/Panthers-win-XXbt_oX7S_SW5q43L9XrNA">阅读更多</a>。</li>
<li><strong>澳大利亚新任总督</strong>：公告称，澳大利亚将任命新的气候和性别倡导者 Samantha Mostyn 为总督。在此获取完整故事 <a href="https://www.perplexity.ai/page/Ms-Samantha-Mostyns-lVVHDtSRQeWmfvxzFMRk.Q">here</a>。</li> 
<li><strong>如果我是...</strong>：一位用户在 Perplexity AI 上提出了一个有趣的搜索问题，标题为 <a href="https://www.perplexity.ai/search/If-Im-a-UwBIlYUgT6ms6xsVqQ.OGA">“如果我是...”</a>，鼓励其他人探索对自我身份的追求。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/peoJ7ftVVqY">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/embed/xUsxGDrwzls">YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1255173219921694802)** (2 messages): 

- **Pro 功能是必需的**：一位成员表示这“很遗憾”，因为他们的工作需要 “Pro features”。表情符号 😦 凸显了他们的失望。
- **寻求 f1shy 的帮助**：另一位成员建议联系 “f1shy” 来解决问题。他们的语气表明 f1shy 可以提供所需的帮助。

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1254879432301350955)** (75 messages🔥🔥): 

- **RTX 3090 在运行模型时遇到困难**：一位用户反馈其 **RTX 3090** eGPU 设置表现不佳，无法在长 token 上下文下加载像 **Command R (34b) Q4_K_S** 这样较大的模型。建议探索 **exl2 格式** 的模型以获得更好的 VRAM 利用率。
  
- **exl2 格式仍然受限**：一位用户指出 GitHub 上 exl2 格式的**规模较小且 GUI 选项有限**。为了获得更好的性能，建议使用 **tabbyAPI** 和 **open-webui** 等工具。

- **对 Llama 3 模型标签的困惑**：用户讨论了 **Llama 3 8B text**、**Llama 8B Instruct** 以及未标记的 **Llama 3 8B** 之间的区别。明确了未标记的版本是基础模型（base model），并未针对特定任务进行微调（finetuned）。

- **LM Studio 对 AMD 和 Intel GPU 的支持**：一位用户询问了 LM Studio 对 **Intel 和 AMD GPU 的支持**情况；目前通过 **OpenCL** 支持，但缺乏对 **RoCM** 和 **Vulkan** 的支持。分享的 [配置说明](https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md) 链接帮助解决了一些问题。

- **关于视觉模型的咨询**：另一位用户询问了 ML Studio 的**图像生成能力**。目前 **LM Studio** 不支持图像生成，建议使用 [Fooocus](https://github.com/lllyasviel/Fooocus) 等外部工具来实现这些功能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1">Vision Models (GGUF) - a lmstudio-ai Collection</a>：未找到描述</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/Extension-Pack-Instructions.md">configs/Extension-Pack-Instructions.md at main · lmstudio-ai/configs</a>：LM Studio JSON 配置文件格式及示例配置集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus: Focus on prompting and generating</a>：专注于提示词和生成。通过在 GitHub 上创建账号为 lllyasviel/Fooocus 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1254877779569148016)** (26 messages🔥): 

- **Hathor_Fractionate-L3-8B-v.05 的性能受到称赞**：一位用户分享了在 Hugging Face 上使用 **Hathor_Fractionate-L3-8B-v.05** 的积极体验，强调了它在创意写作和教育支持方面的能力。他们强调将输出张量（tensors）和嵌入（embeddings）保留在 F32 格式中有助于提升写作质量。

- **Replete-Coder-Llama3-8B 在编程任务中表现出色**：**Replete-Coder-Llama3-8B** 模型因精通 100 多种编程语言并结合了安全性、漏洞预防和高级数学能力而受到关注。它使用了大量未经审查的编程指令数据进行训练，适用于通用和专业编程应用。

- **New Dawn 70b 在角色扮演中令人印象深刻**：成员们讨论了对 **New Dawn 70b** 在复杂角色扮演场景中的满意度。它展示了在性能下降前处理高达 32k 上下文的创意能力。

- **DeepSeek Coder V2 资源需求**：围绕 **DeepSeek Coder V2** 的讨论强调了其对大量 VRAM 的需求，特别是建议轻量版（lite version）使用 24GB，并提到对于更大的模型，最佳配置是结合系统 RAM 和 GPU 的 VRAM。

- **奇幻故事写作模型推荐**：对于奇幻故事创作和角色扮演，用户被引导尝试特定模型，包括 **bartowski/aya-23-8B-GGUF** 以及 Discord 上的其他推荐。强调了通过实验不同模型来寻找最适合特定需求模型的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/friends-ross-geller-david-schwimmer-drink-a-lot-gif-15947149">Friends Ross Geller GIF - Friends Ross Geller David Schwimmer - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://huggingface.co/Nitral-AI/Hathor_Fractionate-L3-8B-v.05">Nitral-AI/Hathor_Fractionate-L3-8B-v.05 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Hathor_Stable-L3-8B-v0.5-GGUF">bartowski/Hathor_Stable-L3-8B-v0.5-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1254882262554509423)** (3 条消息): 

- **Ubuntu 22.04 上的 LM Studio 网络错误**：一名用户报告在 Ubuntu 22.04 上使用 **LM Studio** 搜索 HuggingFace 模型时出现 **"network error"**，但在 Mac M1 上运行正常。他们提到唯一的改动是注释掉了用于端口 3001 的 ser2net 配置文件，该端口曾被 AnythingLLM Web 服务器使用。

- **建议添加到功能请求**：另一名用户询问该网络错误问题是否已添加到 **feature requests** 中，认为这可能是一个相关的补充。

- **IPv6 潜在解决方案**：一位 IT 专家建议在受影响的 Ubuntu 机器上禁用 **IPv6**，以尝试解决网络错误问题。他们幽默地指出，许多问题的报告开头都是“我什么都没改”。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1254876809938604084)** (7 条消息): 

- **旧硬件在 AI 工作负载下表现吃力**：一位成员分享了在 *"16GB DDR4 和 GTX 1060"* 配置下运行语言 AI 的困难，提到尽管进行了各种设置，仍有明显的延迟。另一位成员幽默地表示，与最先进的云端 AI 相比，旧配置 *"简直像是在 19 世纪"*。
- **高性能 GPU 的高昂成本**：成员们调侃了 *"NVIDIA 那些单价 40,000-8,000 美元的 100,000 系列 GPU"* 的负担能力。对话强调了顶级 AI 硬件带来的高昂成本，一位成员补充道：*"即使你不需要问价格，你可能还是买不起。"*
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/)** (1 条消息): 

uniartisan_86246: 我想请问一下，当我作为服务器运行时，是否可以设置 CPU 线程数？
  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1254888695530782730)** (5 条消息): 

- **上下文长度问题在 3000 tokens 处解决**：一名用户报告在 3000 tokens 的上下文窗口下结果稳定，并疑惑为什么在还有 4GB RAM 可用的情况下无法进一步提高。
- **LM Studio 不支持 Lora 适配器**：一位成员询问是否可以在 LM Studio 托管的 GGUF 模型上使用训练好的 Lora 适配器。另一位成员回答说：“LM Studio 不支持 Lora。”
  

---


### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1255216237928255589)** (1 条消息): 

- **更详细的 System Prompt 会带来更好的结果**：聊天机器人的行为处理是通过 System Prompt 控制的。如文中所述：*"System Prompt 越详细，结果就越好。"*
  

---


### **LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1255175274597847161)** (11 条消息🔥): 

- **不支持 Gemini Nano 模型导致本地运行错误**：一名用户尝试在本地 LM Studio 上运行 **Gemini Nano** 但遇到错误。另一位成员澄清说 **Gemini Nano** 是不受支持的，它不适用于 **llama.cpp** 或 LM Studio，且不是官方发布版本。
- **LM Studio 仅支持 GGUF 模型**：当被问及 **Gemini Nano** 是否可以量化为 **GGUF** 时，得到的回答是否定的。LM Studio 仅限于 **GGUF 模型**，因此无法使用 **Gemini Nano**。

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1254883110403969075)** (81 messages🔥🔥): 

- **唱片公司起诉 AI 音乐生成器**：包括 **Sony Music Entertainment** 和 **Universal Music Group** 在内的主要唱片公司正在起诉 Suno 和 Udio 等 AI 音乐生成器，指控其“大规模侵犯版权”。**RIAA** 正在协调这些诉讼，声称这两种 AI 模型都在未经授权的情况下使用了受版权保护的音乐进行训练。[来源](https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/)。

- **对音乐 AI 训练伦理的批评**：成员们讨论了 Suno 和 Udio 如何通过处理 Caption 以删除艺术家姓名并避免模型过参数化（overparameterization）来规避版权问题。一种观点认为，如果训练受版权保护的材料不导致记忆化（memorization），则可被视为合理使用（fair use），但目前的 AI 实践与这一理想状态相去甚远。[示例](https://www.404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued/)。

- **开源音乐 AI 的潜力**：针对 Suno 和 Udio 的诉讼引发了关于创建一个开源、符合伦理构建的音乐模型的讨论。该模型理想情况下将使用公共领域或无版权歌曲，并采用创新架构，以最大限度地减少对受版权保护材料的依赖。

- **模型训练最佳实践**：对话中提到的观点认为，使用较少参数训练的模型可以避免过拟合（overfitting）和记忆化（memorization），而这两者是版权侵权指控的主要原因。有人建议遵循正确的训练方法论以避免此类陷阱。

- **AI 图像描述模型**：简要讨论了寻找能够排除图像某些部分（例如人）的图像描述（image captioning）模型。虽然提出了注意力掩码（attention masking），但实际实现和结果可能有所不同，像 **LLaVA Llama3 8B** 这样的一些模型可能具有忽略特定元素的内置功能。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.404media.co/listen-to-the-ai-generated-ripoff-songs-that-got-udio-and-suno-sued/">Listen to the AI-Generated Ripoff Songs That Got Udio and Suno Sued</a>：唱片业提交了一份包含数千首歌曲的名单，认为这些歌曲在未经许可的情况下被抓取，并使用 Udio 和 Suno 重新制作了著名歌曲的版本。</li><li><a href="https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-generators-suno-udio-for-mass-infringement-of-copyright/">Major record companies sue Suno, Udio for ‘mass infringement’ of copyright</a>：Universal、Sony、Warner 联合在新的诉讼中起诉 AI 音乐生成器……</li><li><a href="https://www.musicbusinessworldwide.com/major-record-companies-sue-ai-music-gene">Major record companies sue Suno, Udio for ‘mass infringement’ of copyright</a>：Universal、Sony、Warner 联合在新的诉讼中起诉 AI 音乐生成器……</li><li><a href="https://en.wikipedia.org/wiki/Sarah_Kubitschek">Sarah Kubitschek - Wikipedia</a>：未找到描述</li><li><a href="https://maxread.substack.com/p/my-kindle-thinks-im-stupid-now">My Kindle thinks I'm stupid now</a>：与 Leah Beckmann 一起进入 Kindle AI 垃圾地狱之旅</li><li><a href="https://tenor.com/0uH6.gif">Hitchhikers Guide To The Galaxy Leigh574 GIF - Hitchhikers Guide To The Galaxy Leigh574 Question - Discover &amp; Share GIFs</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1255113208088363101)** (27 条消息🔥): 

- **对开源多模态模型的兴趣被激发**：一位用户询问是否有协作开发开源多模态模型的频道，类似于[这篇文章](https://laion.ai/notes/open-gpt-4-o/)中讨论的模型，重点关注来自 OpenAI 的 GPT-4-OMNI 等技术。
- **Carlini 为攻击性论文辩护**：一名成员分享了 Nicholas Carlini 的一篇博客文章（[链接在此](https://nicholas.carlini.com/writing/2024/why-i-attack.html)），他在文中解释了撰写攻击性论文（attack papers）的动机，并回应了 Ben Zhao 教授的批评。
- **Glaze 频道因争议被删除**：用户讨论了 Glaze 频道被删除一事，有人推测是由于成本和法律问题，而另一些人则认为是为了删除过去的争议性言论。
- **关于 Nightshade 的法律担忧**：一位用户提供了一篇博客文章，解释了名为 Nightshade 的保护方案可能存在的法律和伦理风险，指出尽管 “[Nightshade](https://undeleted.ronsor.com/nightshade-legal-poison/)” 尚未正式发布，但已引发了重大担忧。
- **关于模型投毒的争议**：有人指出 Zhao 教授之所以面临问题，主要是因为他鼓励对模型进行投毒（poisoning models），这在社区内引起了强烈的抵制。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://laion.ai/notes/open-gpt-4-o/">呼吁为个人助手构建开源多模态模型 | LAION</a>：&lt;p&gt;OpenAI 最近推出的 GPT-4-OMNI 等技术再次展示了强大的多模态模型在积极转型方面的潜力...</li><li><a href="https://undeleted.ronsor.com/nightshade-legal-poison/">Nightshade：伪装成艺术家保护方案的法律毒药</a>：正如我在之前的文章中所述，生成式 AI 对许多艺术家来说仍然是一个备受争议的话题，为了抵制模型训练，出现了各种方案。上一篇文章...</li><li><a href="https://nicholas.carlini.com/writing/2024/why-i-attack.html">
      我为何发起攻击
    </a>：未找到描述
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1255250846426595428)** (1 条消息): 

- **ChatGPT 桌面应用登陆 macOS**：适用于 **macOS** 的 **ChatGPT 桌面应用** 现已面向所有用户开放。使用 Option + Space 快捷键即可[更快速地访问](https://openai.com/chatgpt/mac/) ChatGPT，就电子邮件、屏幕截图以及屏幕上的任何内容进行对话。
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1254883207510495263)** (79 条消息🔥🔥): 

- **好奇 LLM 是否支持“实时”聊天**：一位用户询问 LLM 是否可以在没有提示的情况下主动发起提问，对此澄清这与指令相关，且是 Gemini 提供的一项功能。
  
- **关于 Token 上下文窗口限制的讨论**：多位用户讨论了不同模型（如 **ChatGPT4**、Claude 和 Gemini）的 Token 上下文窗口，指出 **ChatGPT 对免费用户的限制为 8k**，而付费或其他模型则提供更大的容量（例如 Claude 提供 200k Token）。

- **区分训练与微调**：讨论了 **CustomGPT** 与通过 API 进行实际训练之间的区别，详细说明了 CustomGPT 涉及附加文档以获取额外知识，而非对模型进行深度训练。*“它无法记住单个聊天中的信息并用于新的聊天。”* 

- **EvolutionaryScale 和硬件进展**：分享了 *[EvolutionaryScale 关于 ESM3 的发布公告](https://www.evolutionaryscale.ai/blog/esm3-release)*，这是一个模拟了 5 亿年进化的模型；并讨论了 *[Sohu，一种新型专用 AI 芯片](https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw)*，旨在更高效地运行 Transformer 模型。

- **细胞智能及其对 AI 的启示**：讨论了 **Michael Levin 和 Denis Noble 关于细胞智能的研究**，以及在 AI 模型中模拟这些生物现象以实现高级问题解决能力的可能性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>：矩阵乘法（MatMul）通常占据了大型语言模型（LLM）的主要计算成本。随着 LLM 扩展到更大的嵌入维度和上下文长度，这种成本只会增加...</li><li><a href="https://x.com/alexrives/status/1805559211394277697">来自 Alex Rives (@alexrives) 的推文</a>：我们训练了 ESM3，很高兴介绍 EvolutionaryScale。ESM3 是一个用于编程生物学的生成式语言模型。在实验中，我们发现 ESM3 可以模拟 5 亿年的进化...</li><li><a href="https://ai.meta.com/blog/brain-ai-image-decoding-meg-magnetoencephalography/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLr">来自 Etched (@Etched) 的推文</a>：认识 Sohu，有史以来最快的 AI 芯片。Sohu 运行 Llama 70B 的速度超过每秒 500,000 个 Token，让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 块 H100。Soh...</li><li><a href="https://x.com/etched/status/1805625693113663834?s=46&t=tqMTDs9oHX6jLrqq8NrMdw">来自 Etched (@Etched) 的推文</a>：认识 Sohu，有史以来最快的 AI 芯片。Sohu 运行 Llama 70B 的速度超过每秒 500,000 个 Token，让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 块 H100。Soh...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1254888302905921606)** (17 条消息🔥): 

- **上下文窗口：ChatGPT 处理大文档时遇到困难**：用户讨论了 ChatGPT 在处理长文档时的困难。上下文窗口对 **Plus 用户限制为 32k Token**，对 **GPT-3.5 限制为 8k Token**，建议对于更大的 Token 需求使用 **Gemini 或 Claude** 等替代模型。
- **JSON 输出问题**：一位成员寻求帮助，希望从助手那里获得有效 JSON 格式的输出。尽管提供了指令，他们仍无法获得有效的 JSON 响应。
- **数学公式字体限制**：用户注意到 ChatGPT 倾向于以特定的“数学字体”输出数学公式。建议指定 **LaTeX 格式** 以获得更好的结果。
- **性能下降**：有用户抱怨 **GPT-4 最近的性能**问题，包括响应时间增加以及在分析和历史研究查询中的可靠性问题。
- **桌面应用可用性**：成员们对**桌面应用仅适用于搭载 Apple Silicon 的 macOS** 表示沮丧，并指出 Windows 版本预计将于今年晚些时候推出。
  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1254890043420708954)** (4 messages): 

- **GPT 在处理基于文件的信息时遇到困难**：一位用户表达了对 GPT 在根据上传文件查询时提供错误答案的沮丧。他们征求 Prompt 建议，以提高 GPT 提供准确信息的可靠性。
- **流式代码迭代**：一位用户询问如何保存 GPT 编写的用于从文档中提取 URL 的代码迭代。另一位成员建议使用代码块展开功能和图标来复制代码，以便在未来的交互或自己的环境中使用。
  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1254890043420708954)** (4 messages): 

- **苦于编写获取准确信息的 Prompt**：一位成员征求关于创建 Prompt 的建议，以便在查询上传文件时让 GPT 提供可靠信息。他们分享说 **GPT 有时会根据上传文档提供错误答案**。
- **保存迭代代码以备后用**：另一位成员询问如何保存用于从文档中提取 URL 的代码迭代，以便通过 GPT 的自然语言界面重新使用。建议他们通过展开代码块或点击特定图标来复制并在未来的对话或个人环境中使用。
  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1254885656837099674)** (90 messages🔥🔥): 

- **销售 AI 艺术品需要熟练的 Prompt 技巧**：讨论强调，拥有高级 Prompt 技巧、艺术背景和良好场景构图能力的人已成功通过销售 AI 生成的艺术品获利。*"如果你已经具备一些艺术技能……并且也精通这些东西，否则你卖出任何东西的机会都很渺茫。"*
- **GitHub 和 CUDA 测试问题**：成员们在访问 GitHub 仓库时遇到问题，后来证明是临时故障。另一位成员描述了 *"RuntimeError: Torch is not able to use GPU"*，并收到了检查 CUDA 和 PyTorch 兼容性的建议。
- **Open Model Initiative 辩论**：对于宣布的 Open Model Initiative 评价褒贬不一，一些成员对其真实性表示怀疑，而另一些人则支持这一想法。*"Reddit 社区的人想讨厌它，因为提到了伦理。"*
- **Google Colab 过度使用担忧**：用户担心因在 Google Colab 上过度使用 Stable Diffusion 而被封禁或标记。*"他们最终会限制你的使用……在 RunPod 上运行实际上每小时只需约 30 美分。"*
- **Stability.AI 的未来受到质疑**：成员对 Stability.AI 的生存能力表示担忧，如果他们不修复并取消对 SD3 的审查。*"在当前市场中，他们有什么可以竞争并赚钱的东西？"*

**提到的链接**：<a href="https://civitai.com/articles/5884">Civitai 加入 Open Model Initiative | Civitai</a>：今天，我们很高兴地宣布启动 Open Model Initiative，这是一项新的社区驱动努力，旨在促进……的开发和采用。

  

---



### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1254910403771895952)** (4 messages): 

- **Replete-Coder-Llama3-8b 旨在挑战 GPT 的主导地位**：一位成员分享了新的 Replete-Coder-Llama3-8b 模型，称其“不仅仅是一个编程模型”，能够使用 100 多种语言编写代码。该模型由 [TensorDock](https://tensordock.com) 提供云算力支持，因其 390 万行未经审查且去重的训练数据而备受关注。
- **在 Ollama 上支持 Claude 3.5 的请求**：一个 GitHub Issue 请求支持在 [Ollama](https://github.com/ollama/ollama/issues/5235) 上加载开源的 Claude 3.5 模型。一位成员强调该 Issue 是正在进行的开发讨论的一部分。
- **排除赛博格（Cyborg）引发关注**：一位成员评论了社区中感知到的“后人类主义偏见”，幽默地指出了对赛博格（Cyborg）的排除。
- **性能提升亮点**：最后一条评论提到了显著的性能提升，称其“现在快了 12 倍”。这突显了优化现有系统的持续努力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ollama/ollama/issues/5235">Claude 3.5 模型 · Issue #5235 · ollama/ollama</a>：是否可以支持加载开源的 Claude 3.5 模型？谢谢。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1254903587541221397)** (5 条消息): 

- **LLM 中的决策边界是不规则的**：一则帖子分享了一项研究，探讨了 LLM 中 In-context Learning 的决策边界与决策树（Decision Trees）和 KNN 等传统模型的对比。研究揭示了 *“LLM 的 In-context 决策边界中存在意想不到的不规则性和非平滑性。”* [在 arXiv 上阅读更多](https://arxiv.org/abs/2406.11233)。

- **用于 LLM 压缩的 Sparse Attention**：分享了论文 "MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression" 的 [官方实现](https://github.com/thu-nics/moa) GitHub 链接。该项目专注于使用混合稀疏注意力（Mixture of Sparse Attention）来实现大语言模型的自动压缩。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/siyan_zhao/status/1805277462890492321?s=46&t=zdoDWYj2oTzRaTJHApTcOw">来自 Siyan Zhao (@siyan_zhao) 的推文</a>：你是否想过 LLM 中 In-context Learning 的决策边界与决策树和 KNN 等传统模型相比如何？🤔 我们的研究发现了意想不到的不规则性和非平滑...</li><li><a href="https://github.com/thu-nics/moa">GitHub - thu-nics/MoA: 论文 &lt;MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression&gt; 的官方实现</a>：论文 &lt;MoA: Mixture of Sparse Attention for Automatic Large Language Model Compression&gt; 的官方实现 - thu-nics/MoA
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1254880860596076557)** (64 条消息🔥🔥): 

- **Hypernetwork 生成 LoRA**：讨论中提到了一个能够生成 Rank 为 1 的 LoRA 的 Hypernetwork。这种能力可能意味着 AI 模型的高级定制选项。
  
- **关于 Nous Research 中 "Nous" 的澄清**：一位成员解释说，Nous Research 中的 **"Nous"** 在法语中意为“我们的”，代表团队合作和集体热情。另一位成员澄清说，它实际上源自希腊语，意为 *智慧*（intelligence）。

- **Ollama 中的远程代码执行漏洞**：一条 [推文](https://x.com/sagitz_/status/1805261557481312623) 强调了 GitHub 上一个流行的 AI 推理项目中的远程代码执行（RCE）漏洞，被称为 **Probllama (CVE-2024-37032)**。

- **Replete-Coder-Llama3-8B 发布**：Replete-AI 发布了他们的新模型 [Replete-Coder-Llama3-8B](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B)，该模型具备通用能力，并在 100 多种编程语言中表现出卓越的代码熟练度。

- **关于 Llama 70B 性能声明的讨论**：[此处](https://twitter.com/Etched/status/1805625693113663834?s=19) 链接的 Twitter 讨论涉及 Llama 70B 的性能声明——每秒 500,000 个 token——这引发了对其可能配置和现实预期的质疑与分析。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://glif.app/@fab1an/glifs/clxtc53mi0000ghv10g6irjqj">glif - fab1an 制作的 WOJAK MEME 生成器</a>：未找到描述</li><li><a href="https://tenor.com/view/cats-toby-the-cat-nod-yes-yes-yes-hooman-gif-17105827">Cats Toby The Cat GIF - Cats Toby The Cat Nod - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/bing-chilling-ice-cream-john-cena-chinese-bing-gif-23622676">Bing Chilling Ice Cream GIF - Bing Chilling Ice Cream John Cena - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cool-beans-thumbsup-gif-13344631">Cool Beans GIF - Cool Beans Thumbsup - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/sagitz_/status/1805261557481312623">来自 sagitz (@sagitz_) 的推文</a>：我们在 @Ollama 中发现了一个远程代码执行（RCE）漏洞 - 这是 GitHub 上最受欢迎的 AI 推理项目之一。这里有关于 #Probllama (CVE-2024-37032) 你需要知道的一切 🧵👇</li><li><a href="https://github.com/teknium1/Prompt-Engineering-Toolkit">GitHub - teknium1/Prompt-Engineering-Toolkit</a>：通过创建账号为 teknium1/Prompt-Engineering-Toolkit 的开发做出贡献。</li><li><a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-exl2">bartowski/Replete-Coder-Llama3-8B-exl2 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Replete-Coder-Llama3-8B-GGUF">bartowski/Replete-Coder-Llama3-8B-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1254961530223460384)** (5 条消息): 

- **角色分配已确认**：**Teknium** 宣布新角色已创建，并标记了多位用户以确认设置。**Interstellarninja** 随后宣布正式完成，并鼓励团队继续推进。
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1255079543778054215)** (3 条消息): 

- **发布 AI21 的 Jamba Instruct**：查看新的 [AI21: Jamba Instruct](https://openrouter.ai/models/ai21/jamba-instruct) 模型。该模型已添加到 OpenRouter 的 2023-2024 阵容中。

- **探索 NVIDIA 的 Nemotron-4 340B**：OpenRouter 推出 [NVIDIA Nemotron-4 340B Instruct](https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct) 模型。现已可用，属于 2023-2024 系列。

- **探索 01-ai/yi-large**：新模型 [01-ai/yi-large](https://openrouter.ai/models/01-ai/yi-large) 现已在 OpenRouter 上线。此发布属于 2023-2024 集合。

- **注意：推荐参数（Recommended Parameters）选项卡数据错误**：模型页面的“推荐参数”选项卡目前显示的数据不正确。修复工作正在进行中，稍后将分享更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/01-ai/yi-large)">Yi Large by 01-ai</a>：Yi Large 模型由 01.AI 设计，考虑了以下用例：知识搜索、数据分类、类人聊天机器人和客户服务。它在多语言专业能力方面表现出色...</li><li><a href="https://openrouter.ai/models/ai21/jamba-instruct)">AI21: Jamba Instruct by ai21</a>：Jamba-Instruct 模型由 AI21 Labs 推出，是其混合 SSM-Transformer Jamba 模型的一个指令微调变体，专门针对企业应用进行了优化。- 256K Context Window...</li><li><a href="https://openrouter.ai/models/nvidia/nemotron-4-340b-instruct)">NVIDIA Nemotron-4 340B Instruct by nvidia</a>：Nemotron-4-340B-Instruct 是一款针对合成数据生成进行优化的英语聊天模型。该大语言模型 (LLM) 是 Nemotron-4-340B-Base 的微调版本，专为单轮对话设计...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1255221743207972915)** (1 条消息): 

- **AI 进军 Elite: Dangerous！**：**rudestream** 开发了一个 **Elite: Dangerous 的 AI 集成**，充当飞船计算机，对游戏内事件和玩家请求做出反应和响应。在 GitHub [此处](https://github.com/RatherRude/Elite-Dangerous-AI-Integration) 查看他们的项目，并在此处观看演示视频 [此处](https://www.youtube.com/watch?v=nvuCwwixvxw)。
- **呼吁 STT 和 TTS 支持**：开发者提到该项目主要使用 **OpenRouter 的免费模型** 创建，但表达了对 **语音转文本 (STT) 和 文本转语音 (TTS) 模型** 支持的需求。

**提到的链接**：<a href="https://www.youtube.com/watch?v=nvuCwwixvxw">A Day in the Life of a Bounty Hunter | Elite: Dangerous AI Integration</a>：🌟 GitHub 项目地址：https://github.com/RatherRude/Elite-Dangerous-AI-Integration ( github.com/RatherRude/Elite-Dangerous-AI-Integration ) 💬 加入我们的 ...

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1255008821697318933)** (63 messages🔥🔥): 

- **OR 公告发布延迟的原因解释**：用户询问了为何删除某篇公告帖子，解释称这与最新的 Jamba 模型有关，该模型需要进一步测试。“*帖子已经上线，但我们需要再多测试一下。*”

- **LLM 创新时代结束了吗？**：一位用户对大语言模型进步进入平台期表示担忧，指出距离 GPT-4 发布已经过去了一年。他们推荐了 [Francois Chollet 的一个播客](https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/)，讨论了 AI 的现状和未来。

- **AI21 的 Jamba Instruct 模型问题**：多位用户报告在使用 ai21/jamba-instruct 模型时遇到错误，即使调整了隐私设置后依然感到沮丧。一位用户在解决本地缓存问题后成功运行，并指出聊天界面与 API 使用之间存在不一致。

- **处理 LLM 指令**：用户讨论了在 LLM prompt 中处理指令的最佳实践，考虑了针对特定模型使用 XML 标签等替代方案。Fry69_61685 分享了一个有用的资源：[Anthropic Claude 提示工程指南](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)，供进一步阅读。

- **关于 AI 模型中立性与原创性的辩论**：一场关于大型企业 LLM 局限性的讨论展开，这些模型往往避免在哲学或争议性话题上采取明确立场。一位用户强调，相比于能够进行更动态、更有趣对话的模型，人们不太喜欢“对着一堵文字墙说话”。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>: 未找到描述</li><li><a href="https://openrouter.ai/settings#privacy">Settings | OpenRouter</a>: 管理您的账户和偏好</li><li><a href="https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/">280 | François Chollet on Deep Learning and the Meaning of Intelligence &#8211; Sean Carroll</a>: 未找到描述</li><li><a href="https://www.preposterousuniverse.com/podcast/2024/06/24/280-francois-chollet-on-deep-learning-and-the-meaning-of-intelligence/).">280 | François Chollet on Deep Learning and the Meaning of Intelligence &#8211; Sean Carroll</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1254888596528431234)** (57 messages🔥🔥): 

- **"llama.ttf" 将字体文件与 LLM 融合**：一位成员分享了 [llama.ttf](https://fuglede.github.io/llama.ttf/?utm_source=changelog-news)，这是一个包含大语言模型和推理引擎的字体文件。这种创新的融合利用了 HarfBuzz 字体整形引擎的 Wasm shaper 来执行基于文本的 LLM 推理。

- **Karpathy 提倡举办 AI 世界博览会**：[Karpathy 宣布](https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)将在旧金山举办 AI 世界博览会，强调了其在 AI 领域的独特地位。由于展位和门票已售罄，物流需求巨大，目前需要志愿者协助管理。

- **MARS5 TTS 模型发布**：[MARS5 TTS](https://x.com/reach_vb/status/1805336863101620699) 推出了一款具有先进韵律控制的开源文本转语音模型，支持仅需不到 5 秒音频的语音克隆。它采用包含 AR 和 NAR 模型的两阶段架构，以实现精确的音频输出。

- **EvolutionaryScale 融资 1.42 亿美元**：[EvolutionaryScale](https://x.com/pdhsu/status/1805563282746446116) 获得了 1.42 亿美元资金用于开发其 ESM3 模型，该模型能够模拟 5 亿年的进化以生成新蛋白质。Nat Friedman 和 Daniel Gross 等关键人物共同领投了这一规模巨大的种子轮融资。

- **Sohu 声称获得最快 AI 芯片称号**：[Sohu](https://x.com/Etched/status/1805625693113663834) 作为最快的 AI 芯片崭露头角，据称在 Llama 70B 上达到了每秒 500,000 个 token，超越了 Nvidia 的 Blackwell。关于基准测试中使用的真实性能指标和比较方法的辩论也随之浮现。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://fuglede.github.io/llama.ttf/?utm_source=changelog-new">llama.ttf</a>：未找到描述</li><li><a href="https://fuglede.github.io/llama.ttf/?utm_source=changelog-news">llama.ttf</a>：未找到描述</li><li><a href="https://www.youtube.com/live/EjgTv6aSeqk">PyTorch Documentary Virtual Premiere: Live Stream</a>：加入我们，见证 PyTorch 纪录片的正式发布！听取项目关键人物分享从早期到现在的历程。</li><li><a href="https://www.youtube.com/watch?v=5zE2sMka620">AI Engineer World’s Fair 2024 — Keynotes &amp; CodeGen Track</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Ys4rFt8XbMM">Transformers Explained From The Atom Up (Many Inaccuracies! I Am Fixing Them Now!)</a>：你是否曾想过从原子层面了解 Transformers 的工作原理？这里就是学习的地方！:) 请关注我以获取更多纳米技术和 AI 视频。...</li><li><a href="https://x.com/pdhsu/status/1805563282746446116?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Patrick Hsu (@pdhsu)</a>：很高兴能与 @natfriedman @danielgross @Lux_Capital 共同领投 EvolutionaryScale 的 1.42 亿美元种子轮融资。这是一家前沿 AI 实验室，目前已训练出 ESM3，这是一种原生多模态且生成式的语言...</li><li><a href="https://x.com/karpathy/status/1805328398920958214?s">Tweet from Andrej Karpathy (@karpathy)</a>：本周在旧金山举行的 @aiDotEngineer 世界博览会 🔥 https://www.ai.engineer/worldsfair 让我想起了我最近一次演讲的第 1 张幻灯片：“以防万一你想知道……不，这不是一个正常的...”</li><li><a href="https://x.com/elevenlabsio/status/1805627120028184825?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from ElevenLabs (@elevenlabsio)</a>：隆重推出 ElevenLabs Reader App。随时随地使用最高质量的 AI 语音聆听任何文章、PDF、ePub 或任何文本。立即下载，让你的生活被讲述：https://elevenlabs.io/text...</li><li><a href="https://x.com/sharongoldman/status/1805562061583253765?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Sharon Goldman (@sharongoldman)</a>：最新消息：@Meta 在 2023 年裁撤了其 AI 生物学研究团队。现在 @alexrives 的新初创公司 Evolutionary Scale 已筹集 1.42 亿美元，继续致力于构建生成配方的 LLM...</li><li><a href="https://x.com/karpathy/status/1805328398920958214?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Andrej Karpathy (@karpathy)</a>：本周在旧金山举行的 @aiDotEngineer 世界博览会 🔥 https://www.ai.engineer/worldsfair 让我想起了我最近一次演讲的第 1 张幻灯片：“以防万一你想知道……不，这不是一个正常的...”</li><li><a href="https://x.com/pdhsu/status/1805563282746446116?">Tweet from Patrick Hsu (@pdhsu)</a>：很高兴能与 @natfriedman @danielgross @Lux_Capital 共同领投 EvolutionaryScale 的 1.42 亿美元种子轮融资。这是一家前沿 AI 实验室，目前已训练出 ESM3，这是一种原生多模态且生成式的语言...</li><li><a href="https://x.com/ylecun/status/1805581310548697360?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">Tweet from Yann LeCun (@ylecun)</a>：http://EvolutionaryScale.ai ：一家刚刚结束隐身状态的蛋白质组学 AI 初创公司。他们宣布推出 ESM3，这是一个拥有 98B 参数的生成式 LLM，用于“编程生物学”。使用 ESM3 和一个...</li><li><a href="https://x.com/Etched/status/1805625693113663834">Tweet from Etched (@Etched)</a>：认识 Sohu，史上最快的 AI 芯片。运行 Llama 70B 时每秒超过 500,000 个 tokens，Sohu 让你能够构建在 GPU 上无法实现的产品。一台 8xSohu 服务器可替代 160 块 H100。Soh...</li><li><a href="https://x.com/cHHillee/status/1805696613480022238">Tweet from Horace He (@cHHillee)</a>：我完全支持新芯片，很高兴看到新的竞争对手！话虽如此，我认为有几点具有误导性，而且我看到人们对此感到困惑：1. 每秒 500k tokens 是...</li><li><a href="https://x.com/ludwigabap/status/1805571904654254560?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from ludwig (@ludwigABAP)</a>：我的朋友 @jacobrintamaki 制作了一个 20 分钟的视频，内容涵盖了从原子一直到 PyTorch 和 Transformer，目前播放量不足 100 次。他还有一个很棒的“从沙子到 GPU”的速通视频，可以作为一个很好的...</li><li><a href="https://x.com/reach_vb/status/1805336863101620699">Tweet from Vaibhav (VB) Srivastav (@reach_vb)</a>：MARS5 TTS：具有惊人韵律控制能力的开源文本转语音！🔥 &gt; 仅需不到 5 秒的音频即可进行语音克隆 &gt; 两阶段自回归 (750M) + 非自回归 (450M) 模型架构...</li><li><a href="https://tenstorrent.com/cards/">Cards</a>：我们第一台配备 PCIe Gen4 的高性能 AI 计算机。</li><li><a href="https://tenstorrent.com/">Tenstorrent</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=40790911">Anthropic Introduces Claude Projects | Hacker News</a>：未找到描述
</li>
</ul>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1255028418949414992)** (4 messages): 

- **AIEWF 大会的惊喜播客预告**：**Latent Space** 播客的新一期节目预告了即将举行的 AIEWF 大会，并庆祝 "Rise of the AI Engineer" 播客成立一周年。他们还采访了 @RazRazcle，以帮助他启动新的播客 High Agency。[在此收听](https://x.com/latentspacepod/status/1805468252828844052)。

- **Latent Space 的返场嘉宾**：Latent Space 发布了另一期惊喜播客，邀请了 **Imbue 和 Databricks**。本期节目讨论了 Databricks 发布的 **DBRX** 以及 **Imbue 70B**（一个新的内部 LLM，据称在各种基准测试中表现优于 GPT-4，且使用的数据量显著减少）。[在此收听](https://www.latent.space/p/llm-training-2024)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/latentspacepod/status/1805468252828844052">来自 Latent Space Podcast (@latentspacepod) 的推文</a>：🆕 播客警报：特别版 @aidotengineer 预告！https://www.latent.space/p/high-agency 我们非常激动能与 @RazRazcle 互换角色，并帮助他启动他的新播客 High Agency....</li><li><a href="https://www.latent.space/p/llm-training-2024">尖端技术：在 10,000 个 H100 集群上训练 >70B LLM</a>：Imbue 的 CTO Josh Albrecht 和 Databricks 的首席 AI 科学家 Jon Frankle 畅谈了在最大的集群上训练最大模型所需的条件……包括对抗 Infiniband Porch Pirates。
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1254925728521388116)** (3 messages): 

- **本周与 LlamaIndex 线下见面**：LlamaIndex 分享了多个与团队线下见面的机会。*"6 月 26 日星期三 - @jerryjliu0 将在 AI Engineer World's Fair 上发表关于知识助手未来 (Future of Knowledge Assistants) 的闭幕主题演讲！"* 更多详情请见其 [Twitter 帖子](https://twitter.com/llama_index/status/1805366745877856363)。
- **使用 LlamaIndex + DSPy 优化 RAG**：LlamaIndex 宣布了一系列与 DSPy 的新集成，将 DSPy 的类 PyTorch 语法和优化能力与 LlamaIndex 的数据及 RAG/Agent 编排工具相结合。阅读其 [Twitter 帖子](https://twitter.com/llama_index/status/1805622004030284036) 上的完整公告。

**提到的链接**：<a href="https://t.co/0rhUE1kpGM">AI Engineer World's Fair</a>：加入 2,000 名由 AI 赋能并利用 AI 进行构建的软件工程师。2024 年 6 月 25 日至 27 日，旧金山。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1254881145770872985)** (52 条消息🔥): 

- **PGVectorStore 中的维度不匹配问题**：一位用户在使用 bge-small 嵌入模型与 PGVectorStore 时遇到了由于维度不匹配导致的 DataError。该问题在正确配置 `embed_dim` 并确保各组件维度一致后得到解决。

- **HuggingFaceLLM 的内存问题**：一位用户在使用 HuggingFaceLLM 加载 "meta-llama/Meta-Llama-3-8B" 时遇到了与内存限制相关的 ValueError。建议的解决方案是由于性能问题不要卸载（offload）到磁盘，并考虑在本地开发时使用 ollama 等替代模型。

- **定位 RAG 架构图**：一位成员在 LlamaIndex 文档中寻找 RAG 相关的图表，并引用了一篇特定的 [arXiv 论文](https://arxiv.org/abs/2312.10997)。建议的资源包括 LlamaIndex 文档中关于 [概念](https://docs.llamaindex.ai/en/stable/getting_started/concepts/) 和 [Agent 流程](https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/) 的链接。

- **Retriever top_k 设置问题**：用户的 Retriever 没有遵循 10 个节点的 `top_k` 设置，仅返回了 2 个节点。通过在 Retriever 配置中将 `similarity_top_k` 设置为 10，该问题得到了解决。

- **vllm 的 Prompt 模板**：用户寻求关于在 LlamaIndex 中将 Prompt 模板与 vllm 配合使用的说明。官方解释说 `messages_to_prompt` 和 `completion_to_prompt` 提供了函数钩子，而 few-shot prompting 应该通过更新特定模块的 Prompt 来实现。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2312.10997">Retrieval-Augmented Generation for Large Language Models: A Survey</a>：大语言模型（LLM）展示了令人印象深刻的能力，但也面临着幻觉、知识过时以及推理过程不透明、不可追溯等挑战。检索增强...</li><li><a href="https://www.youtube.com/watch?v=qwSAKg1YafM">Aqua Voice (W24) Demo</a>：请访问 withaqua.com 查看</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/llms/vllm/">Vllm - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/concepts/">High-Level Concepts - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/understanding/agent/basic_agent/">Building a basic agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/duckdb_sql_query/?h=sql">SQL Query Engine with LlamaIndex + DuckDB - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1254874684395229255)** (9 条消息🔥): 

- **Mojo 更新日志和 git log 搜索技巧**：一位成员分享了一个有用的命令 *"TIL: `git log -S'<code text here> -p`"*，并链接了 [Mojo 更新日志](https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25)。他们提到了 autotune 的移除日期，以及文档是如何在没有三个月前可搜索历史的情况下重建的。

- **Torch 与 Mojo 的未来集成**：在回答关于同时使用 Torch 和 Mojo 的问题时，一位成员解释说，虽然现在还不直接，但**最终的集成**将使其变得简单。他们强调 Mojo 旨在结合 Torch 的 Python 和 C++ 能力，并建议为了性能进行完全重写。

**提到的链接**：<a href="https://github.com/modularml/mojo/blob/1b79ef249f52163b0bafbd10c1925bfc81ea1cb3/docs/changelog.md#v070-2024-01-25">mojo/docs/changelog.md at 1b79ef249f52163b0bafbd10c1925bfc81ea1cb3 · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建一个账户来为 modularml/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot：来自 *Modular*:
<https://twitter.com/Modular/status/1805642326129492195>

### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1255200900339863632)** (1 条消息): 

- **MAX 24.4 登陆 MacOS，支持 Llama3 等更多模型**：最新的 [MAX 24.4 版本](https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support) 支持 MacOS、本地生成式 AI 模型（如 [Llama3](https://llama.meta.com/llama3/)），并引入了原生量化和 GGUF 支持。这一新功能集允许开发者使用单一工具链构建和部署具有顶级性能的生成式 AI 流水线。
- **运行 MAX 流水线**：要探索 MAX 24.4 的 GenAI 应用功能，你需要先完成[安装](https://docs.modular.com/max/install)。安装成功后，运行 `_max -v_` 应确认你的版本号为 `24.4.0 (59977802)`。

**提到的链接**：<a href="https://www.modular.com/blog/whats-new-in-max-24-4-max-on-macos-fast-local-llama3-native-quantization-and-gguf-support">Modular: MAX 24.4 有哪些新变化？MAX 登陆 MacOS、快速本地 Llama3、原生量化和 GGUF 支持</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新博文：MAX 24.4 有哪些新变化？MAX 登陆 MacOS、快速本地 Llama3、原生量化和 GGUF 支持。

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1254884479076663296)** (3 条消息): 

- **手动数据标注自动化趋势上升**：一位成员讨论了他们使用微调模型自动化 PDF 手动数据标注的工作，提到 [Haystack](https://haystack.deepset.ai/) 是一个很有前景的工具，但指出**准确性是关键**。他们认为将其与用于 ERP 系统的 Quickbooks 集成具有潜力，可以减轻许多用户目前正在进行的手动数据录入工作。

- **AI 用于 ERP 集成的兴趣增加**：另一位成员表示有兴趣为其旨在标注大量数据的独立工具探索 ERP 集成。他们认为之前关于自动化数据录入流程的对话特别有启发性。

- **ARC 测试与 AI 智能辩论变得微妙**：一位用户对 ARC 测试发表了评论，指出它衡量的是文化通用的模式，如封闭区域、对称性和物体特征。他们幽默地建议建立一个以狗为中心的测试版本，采用与狗相关的标准（如粪便气味和吠叫音调），并认为 IQ 测试并不能衡量真正的智能，因此很容易被 AI 解决。

**提到的链接**：<a href="https://haystack.deepset.ai/">Haystack | Haystack</a>：Haystack，可组合的开源 AI 框架。

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1254900143677771796)** (2 条消息): 

- **不含系统调用的 Mojo 代码应能在 GPU 上运行**：一位成员指出：*“不进行系统调用（除了请求内存）的代码将在 GPU 上运行。”* 这一提示强调了 Mojo 未来的能力。

- **MAX Graph API 将促进通过 Mojo 进行 GPU 编程**：Brad Larson 提到：*“通过 Mojo 对 GPU 进行编程的一个重要方式将是通过 MAX Graph API。”* 用户目前可以构建针对 CPU 的计算图，当 MAX 发布支持时，这些计算图将扩展到 GPU。
  

---


### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1254895496900776097)** (24 条消息🔥): 

- **Mojo 缺少 `parallelize_tile` 功能**：一位成员询问关于实现 `parallelize_tile` 的问题，得到的澄清是 **Mojo 目前没有这个功能**。建议在此期间对结构体进行填充（padding）以防止伪共享（false sharing）。

- **手写 SIMD 与向量化**：成员们讨论了**手写 SIMD** 以及手动内联循环与编译器向量化之间的区别。有人指出 Mojo 编译器**禁用了 LLVM 的循环向量化器**。

- **Mojo 中 SIMD 和 SVE 的挑战**：由于 SVE (Scalable Vector Extension) 对循环收尾（loop drains）和列表对齐的独特处理方式，成员们对**支持 SVE** 表示了担忧。一位成员指出，Mojo 当前的实现可能因为没有完全支持这些特性而人为地限制了 SIMD 的收益。

- **鼓励提交功能请求**：成员们鼓励提交 **功能请求（feature request）或 PR**，以使列表对齐符合 SIMD 要求，例如 NEON 的 128 位或 AVX-512 的 512 位。这是在可能添加“如果可用则由 hugepage 支持”的列表以获得更好性能的背景下提出的。
  

---

### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1255073564093190215)** (2 messages): 

- **自定义 AI 模型在 Triton Serving 设置中面临挑战**：一位用户询问了如何使用 **Triton Inference Server** 部署使用 **MAX graph API** 编写的自定义 AI 模型，以及如何构建模型文件以实现兼容性。他们参考了[文档](https://docs.modular.com/max/serve/get-started)并询问了如何将模型转换为标准格式。
- **MAX graph serde 功能正在开发中**：作为回应，另一位成员提到，计划发布针对此类推理用例的、特定于目标的编译型 **MAX graph serde**，并建议“保持关注”以获取更新。
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1255037321020047483)** (9 messages🔥): 

- **Mojo 发布新的 nightly 编译器版本**：发布了 Mojo 编译器的 nightly 更新公告，版本号为 `2024.6.2505` 和 `2024.6.2516`。用户可以使用 `modular update nightly/mojo` 进行更新，详细的变更日志可以在[此处](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)查看，同时还提供了[原始差异 (raw diffs)](https://github.com/modularml/mojo/compare/ddf9b1183c22fac2d1deb2ad95d7fc90ae051542...6961ce560d0457689f8667986b94a7ea02940cea)。
- **List 自动解引用 (autodereferencing) 提升性能**：一项新更改使 `List[T]` 下标返回自动解引用而非副本，从而显著提升了性能。`Dict` 也需要类似的行为，以实现另外 15%-20% 的性能提升。
- **编译器在处理布尔表达式时遇到困难**：发现了一个在编译时处理布尔表达式的问题，特别是在使用 `@parameter` 装饰器时。移除 `not` 等特定部分或切换到 `var` 可以缓解该问题，这可能与[此 commit](https://github.com/modularml/mojo/commit/57ab0baf2352abee9e83e30967634c2be357afd5) 有关。
- **建议更好的引用处理**：一位用户指出，对字典使用 `__get_ref(k)` 可以获得更好的性能。他们建议将 `__getitem__` 更改为返回自动解引用，以优化当前的实践。
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1254873923376513186)** (21 条消息🔥): 

- **关于 LingOly Benchmark 有效性的辩论**：一位用户分享了 [LingOly benchmark 论文](https://arxiv.org/abs/2406.06196v2)，引发了关于其范围和记忆化（memorization）问题的讨论。一位参与者强调了评分方面的问题，而另一位则怀疑如果测试集是公开的，该 Benchmark 的可信度。

- **Mozilla 表彰 AI 创新者**：Mozilla 关于第二届年度 Rise25 Awards 表彰 AI 创新者的 [公告](https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards) 引发了社区内的祝贺。获奖者因其在 AI 领域具有伦理性和包容性的工作而受到赞赏。

- **欢迎新成员**：新成员 Eitan（一位生成模型研究员）和另一位对安全漏洞（security exploits）有浓厚兴趣的个人介绍了自己。Eitan 分享了他的背景以及目前在 Lightricks 的工作，并表达了加入 EleutherAI Discord 的兴奋之情。

- **安全漏洞与模型脆弱性**：一位用户在本地 Llama3 模型中发现了一个漏洞，允许其提供禁止活动的指令。最初被认为是个偶然，但后来确认该漏洞是可复现的，引发了对模型安全性的担忧。

- **社区庆祝与问候**：多位用户就成就和公告互相祝贺，营造了活跃且相互支持的氛围。讨论中充满了幽默和情谊。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.06196v2">LINGOLY: A Benchmark of Olympiad-Level Linguistic Reasoning Puzzles in Low-Resource and Extinct Languages</a>: 在这篇论文中，我们提出了 LingOly benchmark，这是一个针对 Large Language Models 高级推理能力的新型 Benchmark。通过使用具有挑战性的语言奥林匹克谜题，我们评估了 (i) 能力...</li><li><a href="https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-the-2nd-annual-rise25-awards/">Mozilla announces finalists for the 2nd annual Rise25 Awards | The Mozilla Blog</a>: 25 位获奖者和 AI 创新者将在爱尔兰都柏林的活动中受到表彰。Mozilla 自豪地宣布第二届年度 Rise25 Awards 的 25 位获奖者。</li><li><a href="https://blog.mozilla.org/en/mozilla/mozilla-announces-finalists-for-t">Mozilla announces finalists for the 2nd annual Rise25 Awards | The Mozilla Blog</a>: 25 位获奖者和 AI 创新者将在爱尔兰都柏林的活动中受到表彰。Mozilla 自豪地宣布第二届年度 Rise25 Awards 的 25 位获奖者。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1254896464472309871)** (17 messages🔥): 

- **MoE 在特定参数增加方案中受青睐**：一位成员指出，在“某些特定方案”下，所讨论的现象在增加参数时更倾向于 Mixture of Experts (MoE) 架构，因为它避免了增加深度维度。他们认为“MoE 是增加参数的最佳方式”。
- **Federated Learning (FL) 和 AI 模型中的后门漏洞**：讨论揭示了对 Federated Learning 在训练期间易受对抗性后门攻击的担忧，如[这篇论文](https://arxiv.org/abs/2206.10341)所示。担忧还延伸到开放权重模型，思想实验在思考像 Google 或 Meta 这样的大型实体是否会故意分发带有后门的模型。
- **开放权重的安全风险**：成员们争论开放权重分发是否比私有托管更不安全，因为“开放权重模型开发者计划”可能会在不通知的情况下分发并激活后门。对话涉及了检测方法以及在 LLaMA 3 等广泛使用的模型中理论上的后门植入。
- **同态加密的低效性**：虽然同态加密等加密方法被提及作为保护 Federated Learning 的潜在解决方案，但它们因“效率极低而无法使用”受到批评，在实践中主要仍处于理论阶段。这种低效导致一些人建议完全避免使用 Federated Learning。
- **神经网络中的归纳偏置 (Inductive Biases)**：最近的一篇[论文](https://arxiv.org/abs/2403.02241)因探索独立于梯度下降的神经网络架构归纳偏置而受到赞赏。该论文强调了替代架构如何偏向于复杂性，并重新审视了先前关于这些偏置的理解。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.16858">EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees</a>：现代大语言模型 (LLMs) 的推理成本高且耗时，投机采样已被证明是一种有效的解决方案。大多数投机采样方法（如 EAGLE）使用……</li><li><a href="https://arxiv.org/abs/2406.16838">From Decoding to Meta-Generation: Inference-time Algorithms for Large Language Models</a>：现代大语言模型 (LLMs) 研究中最显著的发现之一是，在训练期间扩大计算规模会带来更好的结果。然而，对于推理时的……关注较少。</li><li><a href="https://arxiv.org/abs/2403.02241">Neural Redshift: Random Networks are not Random Functions</a>：我们对神经网络 (NNs) 泛化能力的理解仍不完整。流行的解释基于梯度下降 (GD) 的隐式偏置，但它们无法解释……</li><li><a href="https://arxiv.org/abs/2206.10341">Neurotoxin: Durable Backdoors in Federated Learning</a>：由于其去中心化的特性，Federated Learning (FL) 系统在训练期间天生容易受到对抗性后门攻击。在这种类型的攻击中，攻击者的目标是……</li><li><a href="https://arxiv.org/abs/2401.17948">HyperZ$\cdot$Z$\cdot$W Operator Connects Slow-Fast Networks for Full Context Interaction</a>：Self-attention 机制利用大型隐式权重矩阵，通过基于点积的激活（仅含极少可训练参数）进行编程，以实现长序列建模。在本文中……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1254882986529394738)** (3 messages): 

- **分享 Neural Redshift 论文**：一位成员分享了一篇与 Scaling Laws 相关的有趣论文，题目为 *"Neural Redshift: Random Networks are not Random Functions"* [CVPR 2024 论文](https://openaccess.thecvf.com/content/CVPR2024/papers/Teney_Neural_Redshift_Random_Networks_are_not_Random_Functions_CVPR_2024_paper.pdf)。该论文深入探讨了神经网络初始化如何比之前假设的更具结构性。
- **初始化是关键**：另一位成员强调了初始化在 AI 中的重要性，称其比研究人员通常认为的要重要得多。他们幽默地将这种感悟与达到“启蒙”联系起来，并分享了一个 [AI 公案 (AI Koans) 的链接](http://www.catb.org/esr/jargon/html/koans.html)，这是来自 MIT AI 实验室的幽默禅宗式故事。

**提及的链接**：<a href="http://www.catb.org/esr/jargon/html/koans.html">一些 AI 公案 (AI Koans)</a>：未找到描述。

  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1255015122762465353)** (4 messages): 

- **SAEs 从叠加态（superposition）中恢复线性特征**：Loganriggs 强调了 Lee Sharkey 等人的工作，展示了稀疏自编码器（SAEs）可以从过完备基（overcomplete basis）中恢复线性特征。源文章 [Interim Research Report: Taking Features Out of Superposition](https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition) 转发自 [AI Alignment Forum](https://alignmentforum.org/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition)。
- **对 SAE 测试的玩具模型（toy models）的兴趣**：受 Apollo Research 另一篇名为 [SAE Feature Geometry Is Outside the Superposition Hypothesis](https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis) 的帖子启发，Loganriggs 对其他玩具模型和测试 SAEs 表达了兴趣。该帖子指出，基于叠加态的神经网络激活空间解释具有局限性，并强调了特征几何（feature geometry）的重要性。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.lesswrong.com/posts/MFBTjb2qf3ziWmzz6/sae-feature-geometry-is-outside-the-superposition-hypothesis">SAE feature geometry is outside the superposition hypothesis — LessWrong</a>：由 Apollo Research 撰写 • 摘要：基于叠加态的神经网络激活空间解释是不完整的。特定位置……</li><li><a href="https://www.lesswrong.com/posts/z6QQJbtpkEAX3Aojj/interim-research-report-taking-features-out-of-superposition.">[Interim research report] Taking features out of superposition with sparse autoencoders — LessWrong</a>：我们感谢 Trenton Bricken、Eric Winsor、Noa Nabeshima 和 Sid Black 提供的有益建议。……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1255107515172261899)** (5 messages): 

- **机器翻译的 ARC Challenge PR 引发辩论**：一名成员就 [机器翻译 ARC Challenge 的 PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/1900) 寻求反馈，提到他们需要在评估结果出来之前合并更改，以避免维护分支（fork）。一位审查者批准立即合并，而另一位审查者指出关于该方法发布状态存在沟通误解，最终将讨论定性为沟通失误并解决。

**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1900">add arc_challenge_mt by jonabur · Pull Request #1900 · EleutherAI/lm-evaluation-harness</a>：此 PR 为 11 种语言的机器翻译版 ARC Challenge 添加了任务。我们未来还将添加更多语言。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1254900083950882878)** (7 messages): 

```html
<ul>
  <li><strong>Multi 应用加入 OpenAI 大家庭</strong>：<a href="https://multi.app/blog/multi-is-joining-openai">Multi 的博客文章</a>宣布该应用将加入 OpenAI，探索如何与 AI 一起使用计算机。活跃团队可以使用该应用直至 2024 年 7 月 24 日，之后所有用户数据将被删除。</li>
  <li><strong>苹果拒绝与 Meta 的 AI 合作伙伴关系</strong>：<a href="https://archive.is/uUv1L">苹果拒绝了 Meta 的提议</a>，即在 iPhone 中集成 Llama AI 聊天机器人，转而选择与 OpenAI 的 ChatGPT 和 Alphabet 的 Gemini 达成协议。对 Meta 隐私惯例的担忧是苹果做出这一决定的原因之一。</li>
</ul>
```
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://multi.app/blog/multi-is-joining-openai">Multi Blog &#x2013; Multi is joining OpenAI </a>：最近，我们越来越多地问自己应该如何使用计算机。不是在计算机上或利用计算机，而是真正地与计算机协作。与 AI 协作。我们认为这是最重要的……</li><li><a href="https://archive.is/uUv1L">Apple Spurned Idea of iPhone AI Partnership With Meta Months Ago - Bl&#x2026;</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1255268298577154152)** (5 messages): 

- **Rabbithole 漏洞暴露关键安全缺陷**：来自 [rabbitude](https://rabbitu.de/articles/security-disclosure-1) 的一篇关于安全披露的文章揭示，在 rabbit 代码库中发现了多个**硬编码 API keys**。这些泄露的密钥允许未经授权的访问，可以读取每一条响应、使设备砖化、篡改响应，并使用 **ElevenLabs**、**Azure**、**Yelp** 和 **Google Maps** 等服务替换语音。
- **讨论 ElevenLabs 额度被滥用的可能性**：一些成员对这次安全披露反应幽默，考虑使用被泄露的 **ElevenLabs** 额度。有人指出，“*这玩意儿可不便宜*”，强调了安全漏洞可能带来的财务影响。

**提到的链接**：<a href="https://rabbitu.de/articles/security-disclosure-1">rabbit 数据泄露：所有给出的 r1 响应均可被下载 - rabbitude</a>：rabbit inc 已经知道我们掌握他们的 ElevenLabs (TTS) API key 一个月了，但他们没有采取任何行动来轮换 API keys。

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1254960113890885813)** (27 messages🔥): 

- **市场意识到 nVidia 并非垄断**：随着 Apple 讨论在服务器端使用自家芯片运行 LLM，市场正在调整，意识到“nVidia 实际上并不是垄断者”。这反映在 TAM (Total Addressable Market) 的变化中。

- **SemiAnalysis “GPU Rich” 讨论冷落了 TSMC**：值得注意的是，讨论几乎刻意排除了 TSMC 最大的客户，在考虑晶圆厂产能（fab capacity）时，这凸显了一个重大的遗漏。

- **解释 nVidia 25% 的跌幅**：成员们讨论了 nVidia 突然出现的“单跳 25% 跌幅”，一些人指出这是由于盘后交易和缺乏流动性导致的 Google 股票数据异常。[Quora](https://www.quora.com/Why-does-Googles-stock-GOOG-frequently-tank-for-a-moment-in-after-hours-trading) 和 [Money StackExchange](https://money.stackexchange.com/questions/142738/nvda-stock-after-hours-spike-on-google-finance) 的链接被分享以提供更多背景信息。

- **Imbue AI 发布新工具包**：尽管存在质疑，Imbue AI 还是发布了一个用于训练针对推理和编程优化的 70B 模型的工具包，包括各种基准测试和基础设施脚本。[阅读更多](https://imbue.com/research/70b-intro)关于他们发布的资源。

- **在 Imbue AI 的招聘体验**：成员们回顾了过去在 Imbue AI 面试的经历，现在看来他们似乎走上了“更好的轨道”，尽管对他们的创始人及其 AGI 雄心仍有复杂的情绪。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/imbue_ai/status/1805629542914211951">来自 Imbue (@imbue_ai) 的推文</a>：今年早些时候，我们训练了一个针对推理和编程优化的 70B 模型。尽管训练数据减少了 7 倍，该模型仍能大致匹配 LLAMA 3 70B。今天，我们发布了一个工具包来帮助其他人...</li><li><a href="https://www.ft.com/content/7332b1f8-cf7c-4bfa-82f4-88d0deb23f98">Nvidia 股价下跌抹去了超过 5500 亿美元的市值</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1254936550370705450)** (10 messages🔥): 

- **强调 AI 实验室安全威胁**：在一条[推文](https://x.com/jordanschnyc/status/1805340489391997177)中，对 **Alexandr Wang** 的采访讨论了加强 AI 实验室安全以防止间谍风险的紧迫性。Wang 强调，强大的 AI 系统可能会超越核威慑，提供诸如“超人类黑客攻击”和“自主无人机群”等能力。

- **Alex 很特别**：Nathan Lambert 对 Alexandr Wang 表示赞赏，说“Alex 很特别”。这种情绪在对话中得到了共鸣，一位成员坦言：“看完之后我确实挺喜欢 Alex 的。”

- **帽子市场笑话**：发生了一段幽默的对话，一名成员开玩笑说想买一顶像 Alexandr Wang 那样的帽子，说：“*如果他买了一个用来戴（rock），一个用来存（stock），请转告他我也在市场上求购*”。

**提到的链接**：<a href="https://x.com/jordanschnyc/status/1805340489391997177">来自 Jordan Schneider (@jordanschnyc) 的推文</a>：美国政府应该对目前 AI 实验室的安全状况感到恐惧。在我们明天发布的对 @alexandr_wang 的采访中，当我问他美国政府应该...

  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1254911314560811098)** (36 messages🔥): 

- **Replete-Coder-Llama3-8B 模型引起关注**：一个名为 [Replete-Coder-Llama3-8B](https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B) 的新模型由 Rombodawg 进行微调，号称在 100 多种语言中具备先进的编程能力。它承诺成为一个包含广泛非编程数据、无审查且完全清洗的通用模型。

- **关于 OI 和 Llama3 的视觉模型混淆**：用户 itsahill 和 bebo.gpt 指出 Llama3 并不是视觉模型，需要 Moondream 或 GPT4o。尽管尝试了 `--local --vision` 等 flags，用户在本地运行视觉功能时仍面临挑战。

- **OpenInterpreter 配置中的成功与挑战**：techfren 协助 kenharris 使用正确的 flags 通过 `claude-3-5-sonnet-20240620` 执行代码并获得成功。然而，执行中的一些怪癖，特别是围绕函数支持的问题，引发了关于模型兼容性和设置的疑问。

- **视觉能力方面的困扰**：包括 daniel_farinax 在内的几位参与者报告了在使用 `--os --local --vision` 等配置在本地尝试视觉任务时出现的问题和处理速度缓慢。有人抱怨 OpenAI 视觉功能的高昂成本，以及使用本地 GPU 时的 CUDA 内存错误。

**提到的链接**：<a href="https://huggingface.co/Replete-AI/Replete-Coder-Llama3-8B">Replete-AI/Replete-Coder-Llama3-8B · Hugging Face</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

m.0861：伙计们，AI 视频真的让我感到毛骨悚然。
  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1254907798022524989)** (32 messages🔥): 

- **ChatOllama 更新查询及用法展示**：一位用户询问了 ChatOllama 的更新，并分享了一个 [notebook](https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/) 链接，展示了如何使用一个具有与 OpenAI Functions 相同 API 的 Ollama 实验性封装。它重点介绍了初始化 OllamaFunctions 并使用 JSON Schema 参数绑定函数的方法。
- **高效地向聊天机器人知识库追加文件**：一位用户寻求关于在不重新处理每个文件的情况下向聊天机器人知识库追加文件的建议。另一位用户建议使用向量数据库的 "add_documents" 方法来添加新文档，而无需重新创建整个索引，并提到在 FAISS 中使用 "save_local" 和 "load_local" 方法。
- **OpenAI API 的并发请求**：一位用户寻求帮助，希望使用 GPT-4 同时向多个用户发送通知而无需单独请求，寻求一个具体的异步解决方案。他们之前在 completion 端点使用批量请求，但现在在 ChatCompletion 端点上遇到困难。
- **Ollama 中的流式响应**：一位用户寻求通过流式传输优化 Ollama 的响应速度。得到的建议是从 `langchain_community` 导入 `ChatOllama`，使用 `.stream("query")` 方法，并迭代打印 token 以实现更快的输出。
- **探索用于长期记忆的 Zep**：一位用户询问了在 AI 应用中使用 Zep 作为长期记忆的看法。他们分享了 [Zep](https://www.getzep.com/) 的链接，该工具与 LangChain 集成，用于持久化对话摘要和相关事实保留。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/integrations/chat/ollama_functions/">OllamaFunctions | 🦜️🔗 LangChain</a>：此 notebook 展示了如何使用 Ollama 的实验性封装，使其具有与 OpenAI Functions 相同的 API。</li><li><a href="https://github.com/ollama/ollama-python/?tab=readme-ov-file#streaming-responses">GitHub - ollama/ollama-python: Ollama Python library</a>：Ollama Python 库。通过创建账号为 ollama/ollama-python 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langgraph/tree/main/examples">langgraph/examples at main · langchain-ai/langgraph</a>：将弹性的语言 Agent 构建为图。通过创建账号为 langchain-ai/langgraph 的开发做出贡献。</li><li><a href="https://www.getzep.com/">Zep - Long-Term Memory for AI Assistants</a>：召回、理解并解析聊天对话，以助力个性化体验。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1255007310343045160)** (3 messages): 

- **AI 个人健身教练工具包发布**：一位成员分享了他们的项目 *Valkyrie*，这是一个使用 **NVIDIA 和 LangChain 工具**构建的 AI 个人健身教练。他们强调使用了 **LangGraph** 进行执行流管理，**LangSmith** 进行执行追踪，以及 **NVIDIA AI Foundation Endpoints** 用于基于 Llama 3 70b LLM 的语音等工具。[GitHub - pannaf/valkyrie](https://github.com/pannaf/valkyrie)
- **Instagram 潜在客户抓取工具演示**：另一位用户展示了一个用于从 Instagram 抓取商业潜在客户（特别是美国肯塔基州）的 Python 脚本。他们提供了一个 [Google Sheet](https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022)，其中包含收集到的数据，如姓名、简介、电子邮件、网站和粉丝数量。
- **Visual Agents 中的 Lambda 集成**：一位用户为 Visual Agents（由 LangChain 驱动）添加了 Lambda 支持，并分享了一个 [YouTube 教程](https://youtu.be/3xZlvR3aPQI)，解释了如何在流程中使用 Javascript 对象负载调用 Lambda 函数。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/3xZlvR3aPQI">如何从你的流程中调用 Lambda</a>: 在这段视频中，我展示了如何连接一个 Lambda 模块，它被输入一个 Javascript 对象负载和一个用于选择哪个 AWS Lambda 函数的输入参数...</li><li><a href="https://docs.google.com/spreadsheets/d/1IYiaqHm_PmX5FdhZTIolxQhE3pWJ5T9X/edit?gid=87752022#gid=87752022).">肯塔基州商业数据.xlsx</a>: Sheet1 url, 用户名, 描述, 网站, 电子邮件, 电话, 个人资料名称, 个人资料类型, 帖子, 粉丝, 关注中, 加入日期, 地点, 已验证, facebook, instagram, twitter, tiktok, youtube, linkedin https://www.in...</li><li><a href="https://github.com/pannaf/valkyrie">GitHub - pannaf/valkyrie: V 是一款 AI 个人健身教练，使用 NVIDIA 和 LangChain 工具构建。</a>: V 是一款 AI 个人健身教练，使用 NVIDIA 和 LangChain 工具构建。 - pannaf/valkyrie
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1254886695481114694)** (1 messages): 

- **选择 AI 框架：关键问题视频**：一位成员分享了一段 [YouTube 视频](https://youtu.be/uG0cs8AlnHw)，强调了开发者在将 AI 集成到应用程序之前应该评估的关键考虑因素。视频探讨了 **GPT-4o, Gemini, Claude, 和 Mistral** 等模型，以及 **LangChain** 等框架。

**提及的链接**: <a href="https://youtu.be/uG0cs8AlnHw">你的应用真的需要 AI 框架或 GPT-4o 吗？</a>: 那么，你想把 AI 集成到你的产品中，对吧？别急，没那么快！有了 GPT-4o, Gemini, Claude, Mistral 等模型以及各种框架...

  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1254874214016487425)** (30 messages🔥): 

- **Claude-3.5-Sonnet 传闻被澄清**：一位成员澄清了关于寻求 Claude-3.5-Sonnet 内部消息的传闻，指出他们不认识 Anthropic 的任何人，除了公开推测外没有具体细节。

- **Rerank 模型参数量仍处于保密状态**：当被问及 Cohere rerank 模型的规模时，另一位成员确认该信息尚未公开。

- **Expedition Aya 邀请全球 AI 协作**：[Expedition Aya](https://sites.google.com/cohere.com/expedition-aya/home) 是 Cohere 发起的一项为期 6 周的倡议，旨在邀请全球 AI 研究人员共同构建多语言 AI 模型，并提供获得独家资源、API 额度和奖金的机会。

- **关于 Cohere preambles 的澄清**：讨论和分享的链接澄清了 Cohere 模型中使用的 preambles，包括指导模型行为的 [Command R 默认 preamble](https://docs.cohere.com/docs/preambles) 的具体细节。

- **Cohere 开发者办公时间（Developer Office Hours）已举行**：Cohere 宣布并举行了一场开发者办公时间会议，讨论 Command R+ 的工具使用（tool use）和功能，并鼓励参会者通过提供的 [Discord 链接](https://discord.gg/CsqyQFhEXT?event=1248301309233336350)加入。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/CsqyQFhEXT?event=1248301309233336350">加入 Cohere 社区 Discord 服务器！</a>：Cohere 社区服务器。来聊聊 Cohere API、LLM、生成式 AI 以及相关的一切。 | 17249 名成员</li><li><a href="https://sites.google.com/cohere.com/expedition-aya/home">Expedition Aya</a>：   </li><li><a href="https://docs.cohere.com/docs/introduction-to-text-generation?_gl=1*1n88nt9*_gcl_au*MTU2NjA5OTA4LjE3MTcxMzg4NjA.*_ga*MTMwMjc2MTU2NC4xNzA5MjQwMjYw*_ga_CRGS116RZS*MTcxOTMzNTU2MS4xODIuMS4xNzE5MzM4OTc2LjM5LjAuMA..">文本生成简介</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/preambles?_gl=1*dwa5mc*_gcl_au*MTU2NjA5OTA4LjE3MTcxMzg4NjA.*_ga*MTMwMjc2MTU2NC4xNzA5MjQwMjYw*_ga_CRGS116RZS*MTcxOTMzNTU2MS4xODIuMS4xNzE5MzM4OTM3LjExLjAuMA..">Preambles</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/introduction-to-text-generation?_gl=1*1n88nt9*_gcl_au*MTU2NjA5OTA4LjE3M">文本生成简介</a>：未找到描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1254931447073079376)** (19 messages🔥): 

- **'LazyBuffer' 对象没有 'srcs' 属性的错误已调试**：用户在使用 `.clip()` 时遇到了 *`'LazyBuffer' object has no attribute 'srcs'`* 错误。成员们建议使用 `.contiguous()` 代替 `realize`，并提到这是 `lazy.py` 中的一个 bug，George Hotz 指出需要修复并增加测试。

- **在 Mac 上调试 CI 问题**：一位用户询问了在本地 Mac 上复现 CI 错误时遇到的困难。Qazalin 建议使用带有[特定 Dockerfile](https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1) 的 Docker 来模拟 Linux 环境，这已被证明在解决此类问题上非常有用。

- **Qualcomm GPU 驱动悬赏**：一位用户引用了一篇关于 700 美元 Qualcomm GPU 驱动悬赏的 [Twitter 帖子](https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19)。该帖子提供了有关如何使用 Termux 和 tinygrad 设置 Android 手机以协助开发的说明。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1805317200200581282?t=0bk72a1BFj_jqFAJgwqLjA&s=19">来自 tiny corp (@__tinygrad__) 的推文</a>：我们为 Qualcomm GPU 驱动提供 700 美元的悬赏。Radeon 驱动请参考 ops_amd.py，Adreno 是 Radeon 的变位词是有原因的。如果你有一台搭载 Qualcomm 的 Android 手机，你就拥有了一个开发环境...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/5134">'LazyBuffer' 对象没有 'srcs' 属性 · Issue #5134 · tinygrad/tinygrad</a>：最小复现：from tinygrad import Tensor timesteps = 16 s = 0.008 def cosine_beta_schedule(timesteps, s=0.008): x = Tensor.arange(0, timesteps+2) alphas_cumprod = (((x / (timesteps+1)) + s) / (1 ...</li><li><a href="https://github.com/Qazalin/containers/blob/main/tinygrad/Dockerfile#L1C1-L14C1">containers/tinygrad/Dockerfile at main · Qazalin/containers</a>：通过在 GitHub 上创建账号来为 Qazalin/containers 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1255000825898733612)** (9 messages🔥): 

- **LLM3 多模态发布担忧**：一位成员表示担心，等到 720 亿参数模型完成训练时，**LLM3 多模态模型**可能已经发布了。他们预计训练将在 7 月中旬左右完成，大约需要 4 个 epoch，每个 epoch 耗时 5 天。
  
- **漫长的训练过程**：另一位成员询问了数据集大小和训练时长，结果显示该数据集每个 **epoch 需要 5 天**。计划训练 4 个 epoch，总计 20 天后完成。

- **Adam-mini 在 arXiv 上发布**：一位成员分享了 arXiv 上的 [Adam-mini 优化器论文](https://arxiv.org/abs/2406.16793) 链接，强调了其显著的显存减少能力。**Adam-mini** 通过减少独立学习率的数量，在显存占用减少 45% 到 50% 的情况下，实现了与 AdamW 相当的性能。

**提到的链接**：<a href="https://arxiv.org/abs/2406.16793">Adam-mini: Use Fewer Learning Rates To Gain More</a>：我们提出了 Adam-mini，这是一种优化器，它以减少 45% 到 50% 的显存占用实现了与 AdamW 相当或更好的性能。Adam-mini 通过减少学习率的数量来降低显存...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1255240413720215695)** (1 messages): 

- **在 Hugging Face 上创建带有最小 LR 的 Cosine LR 调度器**：一位用户询问如何在 Hugging Face 上轻松创建一个 Cosine 学习率 (LR) 调度器，并将最小 LR 设置为大于 0 的值。该问题表明了在 Hugging Face 框架中自定义学习率调度策略的兴趣，这可能是为了获得更好的模型训练性能。
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1255095810266366083)** (3 messages): 

- **Minhash 优化令成员兴奋**：一位用户回忆起在使用简单 Python 时发现 **Minhash** 速度很慢。另一位用户分享说他们将性能提升了 12 倍，并邀请其他人尝试并提供反馈。
  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1255061763817340980)** (8 messages🔥): 

- **Phi-3 中的 Tokenizer 配置不匹配**：成员们讨论了 Phi-3-mini 和 Phi-3-medium 的 Tokenizer 配置差异。mini 配置中 `"add_bos_token": true`，而 medium 配置中 `"add_bos_token": false`；这种差异引发了关于其对 Torchtune 影响的疑问。
  
- **TransformerDecoder 中的运行时错误**：在设置过程中出现了 traceback 错误，指示模型参数中存在尺寸不匹配，特别是在 `attn.q_proj.weight`、`attn.k_proj.weight` 和 `attn.v_proj.weight` 中。这些不匹配突显了在 Torchtune 中实现 Phi-3-Medium-4K-Instruct 支持时的潜在问题。

- **Phi-3-Medium-4K-Instruct 需要更多支持**：错误和配置错误表明 Torchtune 尚未完全支持 Phi-3-Medium-4K-Instruct。一位成员幽默地指出了目前存在的问题，说道：“*看来 Torchtune 要正式支持 Phi3-Medium-4K-instruct 还有不少工作要做 😂*”。

- **Tokenizer 调整建议**：贡献者建议创建一个 `phi3_medium_tokenizer` 来解决 Tokenizer 配置差异。建议“*直接复制并粘贴 phi3_mini_tokenizer 并设置 `add_bos = False`*”以与 medium 的设置保持一致。
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1255121210790903868)** (3 messages): 

- **beowulfbr 工具的速度提升**：“现在比 datasketch 快了 12 倍。” 显著的效率提升声明受到了关注。
  
- **寻找会议演讲 Zoom 链接**：一位用户请求“会议演讲：Simon Willison 的命令行语言模型”的 Zoom 链接。有人担心视频仅嵌入在 Maven UI 中。

- **Simon Willison 分享演讲视频和笔记**：Simon 关于从命令行访问 Large Language Models 的演讲现已在 [YouTube](https://www.youtube.com/watch?v=QUXQNi6jQ30) 上线。内容包括一份带有详细笔记和截图的[带注释的演示文稿](https://simonwillison.net/tags/annotatedtalks/)，重点介绍了 [LLM Python 命令行工具](https://llm.datasette.io/)。

**提到的链接**：<a href="https://simonwillison.net/2024/Jun/17/cli-language-models/">Language models on the command-line</a>：上周我作为“精通 LLM：开发者与数据科学家会议”的一部分，做了一个关于从命令行访问 Large Language Models 的演讲，该会议为期六周...

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1254892734003548211)** (2 messages): 

- **讨论了自动化数据集生成技巧**：一名成员发现了一篇关于为 LLM 指令微调（instruction finetuning）生成高质量数据集的方法的精彩帖子。正如[帖子](https://x.com/rasbt/status/1805217026161401984)中所详述的，该技巧是*“全自动的，在本地运行且不需要种子问题”*。

- **Linus Lee 对 Synthetic Aperture Encoding 的探索**：成员们讨论了 Linus Lee 在构建自己的 **Prism** 进行微调方面的工作。他们引用了 [Linus 的个人网站](https://linus.zone/prism)及其关于 [Prism](https://thesephist.com/posts/prism/) 的详细博客文章，强调当前的基础模型（foundation models）对人类来说过于不透明，需要更好的可理解性以实现更丰富的界面。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/rasbt/status/1805217026161401984">来自 Sebastian Raschka (@rasbt) 的推文</a>：这个周末我读到了一个关于为 LLM 指令微调生成高质量数据集的迷人技巧。这是一种全自动的方法，不需要任何种子问题，甚至可以运行在...</li><li><a href="https://thesephist.com/posts/prism/.">Prism：在语言潜空间中映射可解释的概念和特征 | thesephist.com</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/)** (1 messages): 

raminparker: 非常酷。谢谢分享这篇文章！
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1254920051878920222)** (1 messages): 

- **Gradio 中的私有模型加载问题**：一名用户尝试使用通过 AutoTune 微调的私有模型创建 Gradio space。他们收到了一个错误消息，指出由于模型位于私有仓库中，需要提供 `hf_token`。
  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1254906057256468573)** (2 messages): 

- **llamafile v0.8.7 获得提升**：[llamafile v0.8.7](https://discord.com/channels/1089876418936180786/1182689832057716778/1254823644320763987) 已发布，具有*更快的量化（quant）操作*和 *Bug 修复*。此外还有一个关于可能兼容 Android 的神秘暗示。

- **旧金山 AI 活动亮点**：本周的关键线下活动包括成员在 [World's Fair of AI](https://www.ai.engineer/worldsfair) 上的演讲，以及在 [AI Quality Conference](https://www.aiqualityconference.com/) 担任主持工作。

- **调查 Firefox Nightly 的 AI 尝试**：Firefox Nightly 正在测试新的可选 AI 服务，详情可以在 [Nightly 博客](https://discord.com/channels/1089876418936180786/1254858795998384239)上探索。

- **了解最新的 ML 研究**：由社区成员策划的最新 [ML Paper Picks](https://discord.com/channels/1089876418936180786/1253145681338830888) 现已发布。

- **参与 Mozilla AI 七月活动**：即将举行的活动包括 [Jan AI](https://discord.com/events/1089876418936180786/1251002752239407134) 和 [Sentry.io 的 AutoFix](https://discord.com/events/1089876418936180786/1245836053458190438) 等演讲和环节，以及 [AI Foundry Podcast Roadshow](https://discord.com/events/1089876418936180786/1253834248574468249)。
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1254883641230758060)** (4 messages): 

- **新用户需要 llamafile 指引**：一名成员建议提供推荐的 **llamafile 和配置**以及分步指南，以帮助新用户入门。他们强调了探索本地 LLM 的挑战，以及确保入门过程顺畅以避免劝退新手的初衷。

- **平衡用户易用性与高级功能**：另一名成员同意**更多的引导或限制功能**可能会让新用户受益。他们讨论了 Firefox 加入内置本地推理（inference）的可能性，虽然速度较慢，但可以让用户在无需复杂设置的情况下轻松体验私有的设备端推理。

- **Llamafile 版本更新**：一名成员分享了 [Llamafile 0.8.7 发布](https://www.phoronix.com/news/Llamafile-0.8.7-Released)的链接。
  

---

### **AI Stack Devs (Yoko Li) ▷ #[app-showcase](https://discord.com/channels/1122748573000409160/1122748840819306598/1255037953831473203)** (1 messages): 

- **Honeybot.ai Beta 版发布**：[Honeybot.ai](https://honeybot.ai/) 的 Beta 版本刚刚发布。该网站仅限成年人使用，包含 **AI 生成的成人内容**，根据网站上列出的使用条款和隐私政策免费提供。

**提到的链接**：<a href="https://honeybot.ai/"> Honeybot </a>：未找到描述

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-companion](https://discord.com/channels/1122748573000409160/1122788693950857238/1255035853672288326)** (1 messages): 

- **Honeybot.ai Beta 版发布**：一位成员宣布了 [Honeybot.ai](https://Honeybot.ai) 的 Beta 版发布，这是一个专门为 18 岁及以上用户提供的 AI 生成成人内容平台。他们强调 **该平台完全免费** 并征求反馈。

**提到的链接**：<a href="https://Honeybot.ai"> Honeybot </a>：未找到描述

  

---


### **AI Stack Devs (Yoko Li) ▷ #[ai-town-discuss](https://discord.com/channels/1122748573000409160/1132926337598902293/1254915683230158891)** (1 messages): 

- **对项目活跃度的批评**：一位用户对项目的现状表示担忧，强调 *“所有频道中的垃圾信息并没有给我留下这仍然是一个活跃项目的印象。”*
  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1254913124658839754)** (1 messages): 

- **在 LLM 时代检测机器人和欺诈**：一场名为 *“Detecting Bots and Fraud in the Time of LLMs”* 的活动将于 2023 年 6 月 27 日上午 10 点（PDT）举行。会议将涵盖机器人和欺诈检测的机制、机器人带来的挑战、LLM 使用的演变，以及识别和对抗基于 LLM 的机器人的当前方法论。
- **Unmesh Kurup 谈论高级安全系统**：特邀演讲者是 **Unmesh Kurup**，他是 Intuition Machines/hCaptcha 的 ML 总监，在 AI/ML 领域拥有深厚的背景。活动注册免费，可以在[此处](https://lu.ma/y6hz8cod?utm_source=diskord)完成。

**提到的链接**：<a href="https://lu.ma/y6hz8cod?utm_source=diskord">每秒百万次图灵测试：在 LLM 时代检测机器人和欺诈 · Luma</a>：Data Phoenix 团队邀请您参加我们即将举行的网络研讨会，该研讨会将于 6 月 27 日上午 10 点（PDT）举行。主题：每秒百万次图灵测试：…

  

---



---



---



---



---



{% else %}


> 完整的各频道详细内容已在邮件中截断。
> 
> 如果您想查看完整的详细内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}