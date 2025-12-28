---
companies:
- deepseek
- smol-ai
date: '2024-12-31T23:55:07.995126Z'
description: '**强化微调 (RFT)** 作为一种**高效利用数据**的方法被引入，它通过**首次正确解 (FCS)** 和**贪婪多样化解 (GDS)**
  等策略，利用极少的**训练数据**来提升**大语言模型 (LLM) 的推理能力**。**DeepSeek-V3** 是一款拥有 **6710 亿参数的 MoE（混合专家）语言模型**，在
  **14.8 万亿个 token** 上通过 **FP8 混合精度训练**而成，突显了大模型和开源 LLM 领域的进步。


  对 **2025 年 AI** 的预测包括**小模型**的增长、**多模态化**以及**开源 AI** 面临的挑战。AI 对软件开发岗位的影响表明，随着 AI
  自动化低技能任务，开发者需要具备**更高的智能水平**和**专业化能力**。**CodeLLM** 的增强功能通过**原地编辑**和**流式响应**等特性提升了编程辅助体验。


  **自然语言强化学习 (NLRL)** 为 AI 规划和评判提供了更好的可解释性和更丰富的反馈。AI 招聘正迅速增长，初创公司正在寻找 **机器学习 (ML)**
  和**系统架构**方面的优秀工程师。**Rivet**、**Buzee** 和 **Konfig** 等新型 AI 驱动工具利用 **Rust** 和 **V8
  隔离机制 (V8 isolates)** 等技术，改进了实时应用、搜索和 SDK 生成。'
id: 32000e08-2c5b-4185-98e0-9d301a5b3035
models:
- deepseek-v3
- code-llm
- o1
- sonnet-3.5
original_slug: ainews-not-much-happened-to-end-the-year
people:
- corbtt
- tom_doerr
- cognitivecompai
- alexalbert__
- theturingpost
- svpino
- bindureddy
title: '这句话可以翻译为：


  *   **年底没发生什么特别的事。** (最直接的翻译)

  *   **这一年平平淡淡地结束了。** (更具文学感，侧重于氛围)

  *   **岁末并没有什么大事发生。** (稍微正式一点)'
topics:
- reinforcement-learning
- reasoning
- training-data
- mixed-precision-training
- open-source
- multimodality
- software-development
- natural-language-processing
- interpretability
- developer-tools
- real-time-applications
- search
- sdk-generation
---

<!-- buttondown-editor-mode: plaintext -->**一个安静的跨年夜正是我们所需要的。**

> 2024/12/30-2024/12/31 的 AI News。我们为你检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discords（**215** 个频道和 **1948** 条消息）。预计节省阅读时间（以 200wpm 计算）：**238 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

如果你正缺少“Year In Review”类型的内容，你可能会喜欢 [Latent.Space 2024 Year in Review](https://www.latent.space/p/2024-review) 和 [2025 AI Engineer Reading List](https://www.latent.space/p/2025-papers)。

--- 

**AInews 2025 年广告位已开放**！发送邮件至 swyx@smol.ai 并抄送 will@diamondquarters.com，让你的内容每天展示在 3 万名 AI Engineers 面前。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与研究**

- **强化微调 (Reinforcement Fine-Tuning, RFT)**：[@corbtt](https://twitter.com/corbtt/status/1873864746023477482) 介绍了 **Reinforcement Fine-Tuning (RFT)**，这是一种增强 **LLM 推理能力** 的**数据高效**方法。RFT 通过利用 **First-Correct Solutions (FCS)** 和 **Greedily Diverse Solutions (GDS)** 等策略，使模型能够从极少的**训练数据**中学习，从而提高**结果**和**过程效率**。

- **DeepSeek-V3 与开源 LLM**：[@tom_doerr](https://twitter.com/tom_doerr/status/1874031396013879744) 展示了 **DeepSeek-V3**，这是一个拥有 **671B 参数的 MoE 语言模型**，在 **14.8 万亿 token** 上通过 **FP8 混合精度训练**而成。此外，[@cognitivecompai](https://twitter.com/cognitivecompai/status/1873868452638974144) 强调了像 **DeepSeek** 这样的**开源 LLM** 的重要性，突出了它们在**扩展推理 (scale inference)** 和**增强可访问性**方面的潜力。

**AI 预测与趋势**

- **2025 年的 AI**：[@alexalbert__](https://twitter.com/alexalbert__/status/1874181739381432380) 和 [@TheTuringPost](https://twitter.com/TheTuringPost/status/1874120830738911341) 分享了对 **2025 年 AI 的全面预测**，涵盖了 **benchmark 分数**、**模型进展**、**行业动态**以及 **Agent** 的兴起。这些预测包括**小型模型**的激增、**多模态 (multimodality)** 的增加以及**开源 AI** 面临的持续挑战。

- **AI 对软件开发岗位的影响**：[@svpino](https://twitter.com/svpino/status/1874085600296657325) 预测 **AI** 将显著**提高软件开发者的门槛**，为了保持竞争力，开发者需要具备**更高的智力水平**和**专业化能力**。随着 **AI 处理更多低技能任务**，预计这一趋势将导致**开发者数量随时间减少**，迫使专业人士不断**提升技能 (upskill)** 并进行**适应**。

**AI 工具与开发**

- **CodeLLM 增强功能**：[@bindureddy](https://twitter.com/bindureddy/status/1874158369029689652) 宣布了 **CodeLLM 的更新**，包括**原地编辑代码 (edit code in-place)**、**流式响应 (streaming responses)**，以及对所有 **SOTA 模型**（如 **CodeLLM**、**o1** 和 **Sonnet 3.5**）提供**无限配额**。这些增强功能旨在使**编程助手**更加**高效**且**易于使用**。

- **自然语言强化学习 (Natural Language Reinforcement Learning, NLRL)**：[@TheTuringPost](https://twitter.com/TheTuringPost/status/1873867266376527986) 详细介绍了 **NLRL 的优势**，例如**更好的可解释性**、**更丰富的文本反馈**，以及对 **LLM 规划和批判能力**的增强。NLRL 利用**自然语言**进行**决策**并提供**解释**，从而提高 AI 系统的**稳定性**和**有效性**。

**AI 行业与就业**

- **AI 招聘机会**：[@corbtt](https://twitter.com/corbtt/status/1874159180032205310) 正在扩大团队，寻找 **ML** 和**系统**领域的**资深工程师**。该公司拥有 **40% 的月环比增长**，且**技术团队仅 5 人**，提供了一个从**快速增长的 AI 初创公司**中学习并产生**重大行业影响**的机会。鼓励感兴趣的候选人带着令人印象深刻的项目进行 **DM**。

- **AI 工具发布与集成**：[@tom_doerr](https://twitter.com/tom_doerr/status/1874034786244673883) 等人介绍了各种 **AI 驱动的工具**，如用于**实时应用**的 **Rivet**、用于**全文搜索**的 **Buzee**，以及用于**生成 SDK 和 API 文档**的 **Konfig**。这些工具利用 **Rust**、**V8 isolates** 和 **PostgreSQL** 等技术来增强**开发者工作流**和**应用功能**。

**AI 政策、伦理与社会**

- **监管挑战与合作伙伴关系**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1873881752545923334) 讨论了**科技巨头**如何与 **AI 初创公司**建立**创意合作伙伴关系**，作为应对**日益严格的监管审查**而采取的**收购**替代方案。这一策略旨在**应对监管挑战**，同时继续在 **AI 行业**内进行**创新**。

- **AI 法案 (AI Act) 与竞争担忧**：[@BrivaelLp](https://twitter.com/BrivaelLp/status/1874028894892024220) 主张**废除 AI 法案 (AI Act)**，认为**监管约束**正在阻碍 **AI 领域**的**竞争力**。这一立场反映了关于**先进 AI 技术**开发过程中**监管与创新之间平衡**的持续辩论。


---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek V3：硬件要求与性能**

- **[运行在 llama.cpp 上的 DeepSeek V3 祝你新年快乐！](https://youtu.be/FzCEoTiqP7I)** ([Score: 175, Comments: 51](https://reddit.com/r/LocalLLaMA/comments/1hqidbs/deepseek_v3_running_on_llamacpp_wishes_you_a/))：该帖子重点介绍了在 **llama.cpp** 上运行的 **DeepSeek V3**，可能展示了其性能潜力，但缺乏关于实现或结果的具体细节或背景。
  - **性能指标与硬件详情**：**DeepSeek V3** 在配备 **12x32GB RAM**（总计 **384GB**）的 **Epyc 9374F** 配置上实现了约 **7-9 tokens per second** (t/s)。该模型被量化为 **Q4_K_M**，占用 **377GB** 磁盘空间，性能指标因内存位置和 Prompt 具体情况而异。
  - **实现与开发**：该模型尚未完全投入运行，因为开发人员仍在努力在 **llama.cpp** 中实现新的 **pre-tokenizer regex**。该 Regex 详情为：`"Regex": "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_\{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]|\s[\r\n]+|\s+(?!\S)|\s+"`。
  - **社区参与及未来前景**：用户对该项目的进展和潜力表示热烈欢迎，一些人预测到 **2025** 年会出现更经济的模型。讨论还强调了在模型开发中使用 Regex 的挑战和好处，一些用户赞赏语言模型生成 Regex 模式的能力。


- **为什么还没有大量的 DeepSeek V3 第三方供应商？** ([Score: 63, Comments: 59](https://reddit.com/r/LocalLLaMA/comments/1hqbqqq/why_there_is_not_already_like_plenty_3rd_party/))：**DeepSeek V3** 的最先进模型已开放下载和商业使用，但目前仍缺乏提供相关服务的第三方供应商。作者表示愿意为受信任公司提供的 Prompt 立即删除服务支付溢价，并质疑为什么其他国家没有利用非制裁渠道获取顶尖 AI 芯片。
  - **DeepSeek V3 的规模与托管挑战**：**DeepSeek V3** 是一个拥有超过 **6000 亿参数** 的巨型模型，这使得第三方供应商托管它的挑战大且成本高。许多供应商（如 Together）曾尝试托管，但由于模型规模以及 DeepSeek 自身提供的促销定价，面临着吞吐量低和盈利难等问题。
  - **市场时机与基础设施就绪情况**：讨论指出假期季节可能会影响托管服务的可用性，预计随着新一年的推进，将会出现更多供应商。目前用于托管像 DeepSeek V3 这样的大型模型的基础设施尚未优化，影响了托管的速度和成本效益。
  - **数据隐私担忧与定价**：数据隐私问题备受关注，一些用户愿意支付溢价以防止其数据被 DeepSeek 用于训练。此外，DeepSeek 的官方 API 因其价格和速度受到称赞，但目前的促销定价使得第三方供应商在不亏损的情况下难以竞争。


**主题 2. 阿里巴巴 LLM 降价：一次颠覆性的举动**

- **[随着中国 AI 竞争加剧，阿里巴巴将大语言模型价格下调高达 85%](https://www.cnbc.com/2024/12/31/alibaba-baba-cloud-unit-slashes-prices-on-ai-models-by-up-to-85percent.html)** ([Score: 250, Comments: 95](https://reddit.com/r/LocalLLaMA/comments/1hqkxy0/alibaba_slashes_prices_on_large_language_models/))：**Alibaba** 大幅下调了其 **large language models (LLMs)** 的价格，降幅高达 **85%**，反映了 **中国 AI 市场** 日益激烈的竞争。此举是科技公司为应对 AI 开发领域日益增长的竞争而采取的更广泛成本削减趋势的一部分。
  - **中国的绿色能源与 AI 进展**：评论者强调了中国在绿色能源领域的领导地位，指出其生产了全球 **30% 以上的绿色能源**，并有望提前 **六年** 实现气候承诺。中国对 AI 和电动汽车 (EVs) 的关注得到了政府巨额补贴和产业协同效应的支持，使其在价格和创新方面具有竞争力。
  - **排放对比与工业产能**：讨论强调，尽管中国的工业产出巨大，但其 **人均二氧化碳排放量** 低于美国。美国仍是主要的化石燃料生产国，而中国正在扩大其绿色能源产能，包括大规模的太阳能装置。
  - **AI 与技术发展**：提到了中国在 AI 领域的进步，例如 **Qwen** 和其他 LLM 的开发，一些评论者对在西方获取这些技术表现出兴趣。竞争格局正在压低成本，**Qwen-VL-Plus** 的定价为 **每千 tokens 0.0015 元**。

- **[有趣的 DeepSeek 行为](https://www.reddit.com/gallery/1hqntx4)** ([Score: 118, Comments: 86](https://reddit.com/r/LocalLLaMA/comments/1hqntx4/interesting_deepseek_behavior/)): 这篇题为 **"Interesting DeepSeek behavior"** 的帖子没有正文内容，未提供关于 **Alibaba** 及其对全球 **AI 市场**影响的具体细节或背景。
  - 讨论重点关注 AI 模型中的**审查 (censorship)** 制度，并对**中国**和**美国公司**进行了比较。评论者指出审查是行业标准做法，**DeepSeek** 因其位于中国而面临更严格的监管，而像 **ChatGPT** 这样的美国模型也遵循当地法律和准则。
  - 针对**模型行为**和审查实施方式展开了辩论，一些用户认为模型拥有辅助审查机制，而不是修改基础训练数据。这在 **Gemini** 等模型中有所体现，它们拒绝参与某些话题，表明使用了 **guard model** 来管理敏感内容。
  - 对话涉及过滤训练数据以避免不良内容的**经济和技术可行性**。一位用户认为从训练集中排除特定内容可能更有效，而另一位用户指出大规模执行此操作的计算成本很高，且模型受益于接触正面和负面样本，以提高对齐 (alignment) 和可控性 (steerability)。


**主题 3. Qwen：适用于各种应用的首选 LLM**

- **2024 年底你主要使用的本地 LLM 是什么？** ([Score: 285, Comments: 185](https://reddit.com/r/LocalLLaMA/comments/1hqak1f/whats_your_primary_local_llm_at_the_end_of_2024/)): **Qwen2.5 32B** 被强调为作者首选的本地 **LLM**，因为它在 **24GB GPUs** 上表现出色，即使在发布三个月后依然如此。作者征求社区关于年底最受欢迎的本地 **LLM** 选择的意见。
  - **Qwen 模型**：许多用户在各种任务中青睐 **Qwen2.5** 模型，特别提到了用于通用场景的 **Qwen2.5-32B** 和用于编程的 **Qwen2.5-Coder 32B**。一些用户还更喜欢体量更大的 **Qwen2.5-72B** 进行编程，尽管在某些硬件配置上运行较慢。
  - **替代方案与比较**：**Mistral Large 2411** 和 **Gemma 2** 系列经常被用于通用目的和创意任务，一些用户认为 **Mistral Large** 优于更新的模型。**Llama** 系列，特别是 **Llama 3.1** 和 **Llama 3.3**，也因其在创意写作和通用任务中的多功能性而广受欢迎。
  - **技术偏好**：用户讨论了模型大小、量化级别（如 **Q4**、**Q5**、**Q6**）与性能之间的权衡，一些人选择像 **Gemma-2-9b** 这样的小型模型以获得高性价比的性能。此外，对特定用例（如编程）也存在浓厚兴趣，**Deepseek v3** 等模型因其在回答特定编程问题时的卓越表现而受到关注。


**主题 4. 2024 年的 DeepSeek：影响力和市场渗透**

- **2025 年你希望在 Unsloth 中看到什么？** ([Score: 55, Comments: 108](https://reddit.com/r/LocalLLaMA/comments/1hqkeyn/what_would_you_like_to_see_in_unsloth_for_2025/)): **Unsloth** 开发者对社区支持表示感谢，并征求用户对 2025 年未来功能的意见。他们邀请用户提出宏大或微小的更新建议，例如 **Diffusion/Whisper 支持**、**Unsloth RAG** 或 **Apple 兼容性**，并征求关于当前功能、缺失特性、易用性和文档需求的反馈。
  - 用户表达了对 **UI 改进**的强烈愿望，以简化模型微调和管理，建议开发基于 **Gradio 的 UI** 以增强初学者的易用性并简化数据集处理。**Apple/Mac 支持**也是一个热门请求，以便在 MacBook Pro 上进行本地训练。
  - 技术需求包括对 10B 以下模型的**全量微调 (full-finetuning) 支持**、跨多块 GPU 的**分布式训练**，以及 **AWQ 转换和微调**能力。用户发现当前的转换过程非常耗时，一位用户提到 **Llama 3.3 70B** 模型的转换时间长达 8 小时。
  - 重点在于为更智能的推理模型创建**高性价比的数据集**和训练参数，特别是针对那些 GPU 资源有限的用户。社区对现有的 **AMD 和 Intel GPU** 支持表示赞赏，并期待即将推出的**多 GPU 支持**，该功能预计将于明年年初开源。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Deepseek 对决 OpenAI 01：争议性主张与社区反应**

- **Deepseek 声称在多个推理基准测试中击败了 OpenAI 的 01 模型** ([Score: 109, Comments: 89](https://reddit.com/r/OpenAI/comments/1hqjimz/deepseek_claims_they_beat_openais_01_model_on/))：据 **Hacker News** 报道，中国 AI 初创公司 **Deepseek** 声称其最新的 **R1 模型** 在多个推理基准测试中表现优于 **OpenAI 的 01**。该帖子引发了人们对这是一项真正的成就还是公关噱头的怀疑。更多详情可以在[此处](https://www.chinatalk.media/p/deepseek-ceo-interview-with-chinas)链接的文章中找到。
  - **怀疑与批评**：社区对 **Deepseek 的 R1 模型** 存在显著的怀疑，许多评论者对其优于 **OpenAI 的 01** 表示怀疑。**flysnowbigbig** 和 **FranklinLundy** 等用户批评了该模型的性能和可信度，认为这可能只是为了博取关注，或者是对西方模型的复制，缺乏真正的创新。
  - **开源 vs. 私有模型**：一些评论者（如 **SonOfThomasWayne** 和 **informationWarfare42**）讨论了像 **Deepseek** 这样的开源 AI 模型的优势，强调开放权重（open weights）可以使 AI 开发民主化，而不像 **OpenAI** 那样的封闭模型。
  - **地缘政治担忧**：讨论还涉及对中国 AI 发展战略的担忧，**HateMakinSNs** 和 **iperson4213** 对中国可能通过复制和压低成本在 AI 领域占据主导地位表示担忧，这可能会产生全球性影响，包括对核心技术和资源的控制。

**主题 2. 用于邮件知识保留的 RAG：隐私担忧与实现**

- **对 40GB 的 Outlook 收件箱进行 RAG - 长期员工离职，保留知识（理论）** ([Score: 113, Comments: 79](https://reddit.com/r/OpenAI/comments/1hqco4f/rag_a_40gb_outlook_inbox_long_term_staff_member/))：该帖子讨论了使用 **Retrieval-Augmented Generation (RAG)** 技术从一名长期员工的 40GB Outlook 收件箱中保存公司知识的概念。作者设想使用本地 **LLM** 和开源 Web UI 从收件箱创建一个数据库，然后将其交给 **Hugging Face** 来管理查询，并根据历史通信数据建议回复。
  - **隐私与法律问题**：包括 **GamleRosander** 和 **-Akos-** 在内的几位评论者强调了潜在的隐私问题和法律限制，特别是在欧盟的 **GDPR** 法规下，未经所有相关方同意，可能禁止对个人电子邮件数据进行索引。**-Akos-** 还指出了将数据暴露给 **Hugging Face** 等外部方的风险。
  - **技术实现与替代方案**：**edemmeister** 描述了一个成功实现的 **RAG app**，该应用使用在本地部署的 **embeddings model** 和 **LLM**，可以处理各种数据源并自动回复服务台请求。**SpecialistCobbler206** 建议创建电子邮件的简缩版本，以便在构建有用的知识图谱的同时维护隐私。
  - **数据准确性与相关性**：**Fast-Satisfaction482** 对信息的演变性质表示担忧，即过去正确的答案随着时间的推移可能会变得错误，并建议 **temporal graph RAG**（时序图 RAG）可能比静态数据库更有效。

---

# AI Discord 回顾

> 由 o1-mini-2024-09-12 生成的摘要之摘要之摘要

**主题 1. AI 模型性能之战愈演愈烈**

- [**DeepSeek vs. Claude：谁是代码之王？**](https://discord.com/channels/1076964370942267462/1090427154141020190/1323495969421262868)：**DeepSeek Coder V2 Lite** 的表现持续优于 **Sonnet** 等旧模型，在 [WebDev Arena 排行榜](https://web.lmarena.ai/leaderboard)上获得了 **960.01** 分，而 **Claude 3.5 Sonnet** 以 **1218.58** 分领先，引发了 **Gemini**、**GPT-4o** 和 **Qwen** 模型之间的激烈竞争。
- [**Steiner 推理模型在 LM Studio 中大放异彩**](https://huggingface.co/peakji/steiner-32b-preview-gguf)：用户发现了 **Steiner 推理模型**，它在特定推理任务中的表现优于 **Llama 3.3 Q4 70B** 等更大规模的对手，凸显了其在 **LM Studio** 框架内先进的逻辑能力。
- [**ModernBERT Embeddings 增强 LocalDocs 性能**](https://huggingface.co/peakji/steiner-32b-preview-gguf)：**modernbert-embed-base** 的引入为 **LocalDocs** 提供了改进的分词器（tokenizer）和更快的推理速度，为文本分析和检索任务提供了强大的后端支持。

**主题 2. AI 工具与平台增强**

- [**Codeium 的 Windsurf 在额度与等待时间方面遭遇困境**](https://codeium.com/support)：用户面临购买后未收到 **User Prompt credits** 以及 **Windsurf** 响应时间过长（超过 **20 分钟**）的问题，导致为寻求解决而提交的支持工单数量增加。
- [**LM Studio 的 Steiner 模型表现超出预期**](https://huggingface.co/peakji/steiner-32b-preview-gguf)：集成到 **LM Studio** 中的 **Steiner 推理模型**在推理任务中展现了卓越的性能，超越了更大体量的模型，并因其高效性和先进逻辑而受到关注。
- [**OpenAI 的 API 讨论与 Prompt Engineering 的挫败感**](https://discord.com/channels/974519864045756446/1046317269069864970/1323402298746994729)：社区成员辩论了直接提示（direct prompts）的有效性，对有限的 Markdown 支持表示不满，并探索了如 **LexiDeck** 等用于 **multi-agent** 交互的工具，旨在简化功能研究并改进 **Prompt Engineering** 实践。

**主题 3. 数据隐私与 AI 伦理担忧**

- [**Codeium 用户辩论数据隐私与 AI 伦理**](https://codeium.com/blog/pricing-windsurf)：成员们对在敏感代码上使用专有 AI 工具表示怀疑，权衡了先进 AI 建议的益处与潜在的数据窥探风险，并更倾向于使用 **open-source** 解决方案以确保数据安全。
- [**Nous Research AI 强调治疗技术的纠葛**](https://www.bbc.co.uk/news/articles/c78llg7n5d5o)：讨论集中在 **AI 在治疗中的应用**，强调了数据泄露的风险以及维护**患者隐私**的挑战，特别是在 2022 年 NHS IT 公司遭黑客攻击之后。
- [**Stability.ai Discord 呼吁采取防诈骗措施**](https://discord.com/channels/1002292111942635562/1002292112739549196/1323380801941016660)：成员们呼吁加强安全措施，如**电话验证**和验证码，以打击反复出现的诈骗企图，强调了保护社区免受身份盗用和数据抓取的重要性。

**主题 4. 硬件与 GPU 优化策略**

- [**Groq 的 LPU 推理引擎刷新 AI 速度纪录**](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/)：**Groq LPU 推理引擎**实现了 **241 tokens per second**，挑战了传统 GPU，并引发了关于系统 RAM 与 **Cerebras WSE-3** 等专用硬件的讨论。
- [**Raspberry Pi 5 测试凸显 GPU 局限性**](https://github.com/pixelfung/fluidsCUDA)：在 Raspberry Pi 5 上使用 **llama.cpp** 的测试揭示了在 **VideoCore VII** 上编译 **Vulkan 后端**的挑战，其中 **Bielik-1.5B** 模型仅达到约 **7 tok/sec**，强调了更广泛的 **LLM** 工作负载需要更高功率的加速器。
- [**CUDA 重叠与 Triton 性能调整**](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)：社区成员深入研究优化 **CUDA** 数据传输，将 GPU 运行时间从 **15 秒**缩短至接近 **1 秒**，同时还通过禁用 `TRITON_INTERPRET` 环境变量解决了 **Triton** 的性能不佳问题。

**主题 5. 技术问题与社区支持挑战**

- [**Codeium Windsurf 编辑器的订阅困扰**](https://codeium.com/support)：用户报告了从 **Pro Ultimate** 意外降级为免费计划的情况，以及购买的 **flex credits** 到账延迟，引发了紧急支持工单的提交以及社区对可靠性的沮丧。
- [**Aider 的命令执行与 Token 限制困惑**](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)：成员们面临 **Aider** 命令执行尽管有设置但仍需手动确认的挑战，并遇到了持续的 **Token 限制错误**，导致用户请求更清晰的指导和 Prompt 管理策略。
- [**OpenRouter 的模型集成障碍**](https://www.notdiamond.ai)：用户在向 **OpenRouter** 添加自定义模型时遇到困难，怀疑其对资金充足的提供商有限制，而其他人则探索将个人托管作为变通方案，凸显了对小型开发者提供更好支持和文档的需求。

---

# 第 1 部分：Discord 高层级摘要

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **额度混乱与订阅困扰**：在 **Codeium (Windsurf)** 的讨论中，用户在 **User Prompt credits** 上遇到了麻烦，一位用户表示 *“我四天前支付了 10 美元购买灵活额度（flex credits），但一直没收到。”*
   - 其他人报告了从 **Pro Ultimate** 突然降级为免费版的情况，这促使人们建议提交 [支持工单](https://codeium.com/support) 以快速解决问题。
- **Windsurf 等待时间令人疲惫**：一些人发现 **Windsurf** 反应迟缓，即使是在付费计划中，提示词之间也要等待超过 **20 分钟**。
   - 用户要求更快的响应和更智能的 **guardrails**，希望能减少失误并保持编码过程无压力。
- **WSL 担忧与 Linux 偏好**：开发者抱怨 **Windows Subsystem for Linux (WSL)** 的可靠性，理由是代码执行障碍和烦人的设置步骤。
   - 许多人拥护直接安装 Linux 以规避这些陷阱，更倾向于在调试时减少故障。
- **网页爬取愿望与仓库变通方案**：用户强烈要求 Windsurf 支持 **web crawling**（网页爬取）和直接的仓库摄取，希望能尽快推出。
   - 在此之前，一位成员建议使用 [Gitingest](https://gitingest.com/) 将 Git 仓库转换为文本，以便更好地进行 LLM 集成。
- **数据隐私与伦理辩论**：参与者质疑在敏感代码上使用专有 AI 工具的安全性，表示不愿信任封闭系统。
   - 他们权衡了先进 AI 建议带来的好处与潜在的窥探风险，一些人为了安心更倾向于开源方案。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **治疗技术纠葛与隐私风险**：团队成员剖析了 **AI 在治疗中** 的使用，强调了 [监管机构将对医疗记录黑客攻击后的 NHS IT 公司处以罚款](https://www.bbc.co.uk/news/articles/c78llg7n5d5o) 中提到的 2022 年数据泄露事件，该事件揭示了 **患者机密** 方面的漏洞。
   - 他们得出结论，如果 **unique patterns**（独特模式）被复杂的模型处理，匿名数据仍可能暴露身份，这引发了对 **医疗数据处理** 的更深层担忧。
- **Claude 的代码热潮与复杂性难题**：爱好者们分享了尝试使用 **Claude 3.5 Sonnet** 和 **Haiku** 生成 **简洁代码** 的经历，展示了不同程度的 Token 节省，但在处理更复杂的任务时仅取得适度成功。
   - 他们辩论了 **紧凑输出** 是否会妨碍长期可读性，指出 **代码简洁性** 与可维护性之间存在持久的张力。
- **Hermes 3 奇特用法与 Amnesia 模拟**：一位用户尝试使用 **Hermes 3**（非 405b 版本）复制 **Amnesia** 效果以实现刻意遗忘，认为移除 **prompt** 可能会模拟这种效果。
   - 其他人开玩笑说 *“白板”* 方法是最简单的路径，尽管他们承认可能需要更深层的代码调整来确保一致的 **内存重置**。
- **无反向传播突破与 MCU 魔法**：参与者引用了两篇论文：[**Gradients without Backpropagation**](https://arxiv.org/abs/2202.08587) 和 [**Poor Man's Training on MCUs: A Memory-Efficient Quantized Back-Propagation-Free Approach**](https://arxiv.org/abs/2411.05873)，这些论文探讨了 **非反向传播** 方法和前沿优化。
   - 这些引用引发了关于在微控制器上进行 **轻量化训练** 的讨论，说明了在没有标准梯度方法的情况下实现先进 AI 的可行性。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Gemini 的进步与 Discord 的冷清**：用户批评 OpenAI 在 Discord 上的参与度极低，而 **Gemini 2 Flash** 展示了实时搜索并引发了关于竞争的讨论。
   - 一位参与者提到每月在 AI API 上花费 **130 美元**，这表明用户正在寻求更高效的使用方式和成本控制。
- **审核策略与 GPT-4o 的怪癖**：社区成员遇到了内容审核障碍，特别是在涉及未成年人的敏感话题上，促使一些人完全禁用过滤器。
   - 其他人对 **GPT-4o** 的角色一致性和图像生成功能的缺失表示担忧，引发了失望情绪。
- **剧本提升与编码者成长**：一位用户在社区的帮助下改进了一个电影剧本，将更流畅的动作和结构归功于一次 [Discord 交流](https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158)。
   - 新手编码者通过小组调试提升了技能，称赞反馈增强了他们的信心。
- **提示词、Markdown 与 LexiDeck**：贡献者们拥护使用简洁的提示词来引导 ChatGPT，同时抱怨 Discord 在分享示例时对 Markdown 的支持有限。
   - 一个名为 **LexiDeck** 的工具作为 ChatGPT 的多 Agent 框架出现，尽管它目前缺乏 Canvas 功能。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 缺失的 Canvas 功能**：一位用户询问在 **LM Studio** 中生成图像的问题，但目前尚不支持该功能。
   - 另一位用户报告在更新到 **v0.3.5 (build 2)** 时出现 macOS 权限提示，这归因于 'Squirrel' 更新程序。
- **Steiner 推理模型惊艳大模型**：一位用户在 [Hugging Face](https://huggingface.co/peakji/steiner-32b-preview-gguf) 上发现了 **Steiner 推理模型**，声称它在 LM Studio 的推理任务中超越了更大的模型。
   - 他们指出在特定场景下其表现优于 **Llama 3.3 Q4 70B**，引起了对高级逻辑用例的关注。
- **Coral 难题：16W 功率下的 Llama 3.2**：成员们讨论了模型大小 <2GB 的 **Llama 3.2 1b** 潜在运行在限制为 16 瓦的 **Coral.ai TPUs** 上的可能性。
   - 他们得出结论，TPU 可能难以处理更广泛的 LLM 工作负载，从而促使考虑具有更高功率容量的加速器。
- **Groq 以 241 TPS 提速**：**Groq LPU 推理引擎** 因达到 **241 tokens per second** 而受到赞誉，引发了对其性能和价格的兴趣。
   - 一份 [基准测试报告](https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/) 显示了令人印象深刻的吞吐量，引发了关于系统 RAM 与 **Cerebras WSE-3** 等硬件对比的问题。
- **MacBook Pro：RAM vs. 仅 CPU**：有人认为从 **16GB** 升级到 **32GB** 的 MacBook Pro 对 LLM 速度的提升微乎其微，尤其是对于写作任务。
   - 其他人建议如果预算允许可以配置高达 **128GB**，尽管许多人同意仅 CPU 的设置在性能上仍落后于专用硬件。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 占据主导地位及模型限制**：社区成员称赞 **DeepSeek** 的表现优于 **Sonnet** 等旧模型，理由是速度提升和解决了竞争对手的问题。
   - 该模型在 [WebDev Arena 排行榜](https://web.lmarena.ai/leaderboard) 上排名 **960.01**，激发了人们对未来增强功能的期待。
- **O1 API 访问权限困惑**：参与者讨论了不同组织之间 **O1** 和 **o1-preview** 可用性不一致的问题，引发了对当前访问标准的疑问。
   - 他们请求官方澄清，强调了使用 **O1** 处理高级任务的兴趣日益增长。
- **Aider 工作流与命令执行怪癖**：一些用户报告了 **Aider** 命令执行方面的挑战，指出即使设置了 `AIDER_YES_ALWAYS`，直接的 shell 命令仍需要手动批准。
   - 关于 **token 限制错误** 存在困惑，导致有人建议咨询 `/tokens` 以深入了解上下文使用情况。
- **模型切换与基于文件的 Prompt**：工程师们探索了在用于编辑的 **deepseek** 和用于繁重任务的 **o1** 之间轻松切换的方法，考虑使用脚本或智能命令。
   - 其他人询问将 prompt 保存在专用文件中以便快速重用，认为这与 **clinerules** 等解决方案具有潜在的协同效应。
- **WebDev Arena 引发激烈的 AI 竞争**：新推出的 [WebDev Arena](https://web.lmarena.ai) 挑战参与者制作顶尖网站，**Claude 3.5 Sonnet** 以 **1218.58** 的高分领先。
   - **Gemini-2.0-Flash-Thinking-1219** 和 **GPT-4o-2024-11-20** 等高分竞争者突显了竞争的激烈，而实时 [排行榜](https://web.lmarena.ai/leaderboard) 则鼓励社区持续参与。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的统一 Hymba 行动**：工程师们分享了在 Unsloth 流水线中结合**两个 LLM** 的策略，并讨论了 [Hymba-1.5B-Instruct 模型](https://huggingface.co/nvidia/Hymba-1.5B-Instruct)在处理**高级任务**时的表现，尽管目前存在一些支持方面的小问题。
   - 一些人强调了**微调（fine-tuning）最佳实践**，而另一些人则指出了高效使用 Unsloth 时可能存在的兼容性问题。
- **纯干货微调 LLaMA 3**：一位用户分享了关于在 **Ollama** 中优化 **LLaMA 3** 的[教程](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)，指导大家构建本地个人助手。
   - 社区对 **Unsloth** 创作者提供的这份结构良好的教程表示赞赏，称赞其改进的设计和参考资料。
- **TTT 攻克 ARC 任务**：关于 **Test Time Training (TTT)** 的讨论显示，该方法在 **ARC** 数据集上取得了显著进展，在某些情况下**准确率提升了 6 倍**。
   - 引用了一篇[论文](https://arxiv.org/abs/2411.07279)，引发了关于代码可用性的提问，以便对 TTT 方法进行更深入的审查。
- **反馈热潮与友好的 Discord 氛围**：成员们称赞了 **Discord 框架**，对服务器积极的氛围和凝聚力表示感谢。
   - 他们还提出了 2025 年 **Unsloth** 的新功能需求，强调了协作和每个人的开放投入。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **关于 Bolt 成本的 Token 之争**：一位用户报告称，在使用 ChatGPT 和 **Bolt prompt enhancer** 时，两天内消耗了 **3000 万**个 Token，并提醒注意严重的成本影响。
   - 他们告诫社区要更谨慎地管理每月额度，避免为微小的代码调整产生不必要的支出。
- **项目内的重载困扰**：多位贡献者讨论了重载 Bolt 项目应该依赖浏览器刷新还是专门的按钮，一些人倾向于使用基于 AI 的页面特定修复方案。
   - 他们强调，像 **Claude** 这样的代码提取解决方案通过专注于狭窄的代码段，简化了迭代部署。
- **Bolt Pro 订阅困惑**：成员们确认 **Bolt Pro** 按月提供 Token，澄清了关于每日限额与每月限额的不确定性。
   - 他们还讨论了平台的平台使用限制，对缺乏官方 Bolt 支持表示遗憾，并严重依赖社区见解。
- **Facebook API 带来的挫败感**：爱好者们尝试将 **Facebook Marketing API** 整合到 Bolt 中，产生了巨额 Token 费用但成功有限。
   - 一位用户成功同步了一些数据，但在处理高级权限请求时遇到困难，且缺乏来自 Bolt 方面的直接协助。
- **表格数据与 AI 工具试验**：成员们研究了使用 **.csv** 格式在 Bolt 提示词中实现平滑数据导入，旨在简化表格处理。
   - 他们还讲述了使用 AI 工具进行编码的成败参半的结果，指出更复杂的构建需要大量的辅助手动干预。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek v3 崭露头角**：社区成员在 Cursor 中测试了 **DeepSeek v3**，称赞其在处理大型数据库和**复杂查询**时的速度。
   - 他们将其与其他模型进行了比较，强调了其*令人惊讶的可用性*，而一些人则在寻求许可细节方面的澄清。
- **托管方案：快速选择**：爱好者们讨论了 **Hetzner** 和 **Digital Ocean** 的性价比和简单设置。
   - 其他人则称赞了 **Vercel** 加 **AWS** 的协同效应，认为 Docker 技能是实现稳健部署的优势。
- **Next.js 聊天机器人热潮**：社区成员分享了使用 **Next.js** 和 **shadcn** 构建聊天机器人的参考资料，推荐使用 [vercel/ai-chatbot](https://github.com/vercel/ai-chatbot) 作为可定制的方案。
   - 他们建议添加 **API key** 并遵循设置说明，还引用了 [modals-next-test](https://github.com/RezixDev/modals-next-test) 用于基于 TypeScript 的模态框实现。
- **GitHub Models 助力 AI 工程**：**GitHub** 的一项新更新在 GitHub Models 下引入了先进的 AI 工具，[这篇官方博客文章](https://github.blog/news-insights/product-news/introducing-github-models/)对此进行了重点介绍。
   - 用户对 AI 开发者的潜在益处以及通过 GitHub 市场提供*免费模型*的趋势感到兴奋。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 新模型的准入门槛**：一位用户询问如何将他们的模型添加到 **OpenRouter**，怀疑它可能只适用于资金雄厚的提供商，而其他人则鼓励尝试个人托管方案。
   - 贡献者指出 [Not Diamond](https://www.notdiamond.ai) 是另一个多模型 Router，建议小规模开发者仍可以尝试。
- **DeepSeek v3 表现出色**：许多人称赞 **DeepSeek v3** 在额度消耗和稳定性方面表现一致，特别是与 **Claude** 等更昂贵的替代方案相比。
   - 一些人坚持认为它在窄任务中仍然有效，并指出了成本与性能之间的权衡。
- **Gemini 2.0 在 NSFW 方面遇到障碍**：用户报告称 **Gemini 2.0 Flash** 在处理 NSFW 图像描述生成时表现不佳，称其在 OpenRouter 上无法使用。
   - 他们还提到了性能问题和严格的上下文限制，这些都阻碍了高级图像分析。
- **Sonnet vs. DeepSeek：竞争之声**：参与者对比了 **Sonnet** 和 **DeepSeek**，其中 Sonnet 在指令遵循（instruction-following）和复杂查询方面更受青睐。
   - 批评者认为 **DeepSeek** 在高级编程任务上表现不足，尽管它更便宜。
- **自我审查模型引发辩论**：一位参与者询问自我审查（self-moderation）的工作原理，引发了关于违反服务条款时如何触发拒绝消息（refusal messages）的澄清。
   - 一些人引用了 [Building effective agents](https://www.anthropic.com/research/building-effective-agents) 来阐述合规策略，强调了特定提供商指南的作用。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **模型自我评估：是魔法还是神话？**：成员们质疑为什么类 o1/o3 模型在自我评估方面看起来很有效，讨论了它们可能并没有真正意识到自己的局限性，并怀疑采样方法（sampling methods）才是这些说法背后的推手。
   - 其他人指出强化学习（reinforcement learning）具有路径依赖性，认为自我修正并不是结果质量的核心因素。
- **Nvidia 以 7 亿美元收购 Run:ai**：Nvidia 以约 **7 亿美元**收购了 [Run:ai](https://www.run.ai/)，旨在提升 AI 工作负载中的 GPU 调度。
   - 他们计划开源 Run:ai 的工具，引发了关于此举将如何重塑企业级 GPU 编排（orchestration）的讨论。
- **Gary Marcus 挑起争议**：批评者指责 Gary Marcus 很少调整自己的立场，但同时也承认了他的一些观点。
   - 他和其他人辩论了 **GPT-4** 的真实进展和幻觉（hallucinations）问题，反映了对短期内大规模改进的怀疑态度。
- **2024 Interconnects 年度回顾洞察**：Nathan Lambert 总结了两年的 AI 发展，重点介绍了 **RLHF** 和开源，以及对 OpenAI **o1 模型**的期待。
   - 他还评论说，**Meta** 可能无法仅从 AI 中获得明显优势，并警告说不断扩大的模型规模可能会超过目前的硬件水平。
- **简短推文与“蜗牛”的回归**：社交媒体上的讨论显示，像“we are so back”这样简短、随意的帖子往往能吸引意想不到的互动。
   - Lambert 调侃说，这些随手写的文字可能会引发过度反应，最终演变成了有趣的“蜗牛回归（Return of the Snail）”梗。

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Google 严密保护的 Gemini 变得更加严苛**：一位用户观察到，与开源 LLM 相比，**Google AI** 在*敏感话题*上的限制更为严格，并以 **Gemini** 为主要案例，引用了 [Google Vertex AI 文档](https://cloud.google.com/vertex-ai)。
   - 其他人指出，这种谨慎可能会阻碍医疗或法律领域的高级应用，将其描述为*既是安全措施也是一种困扰*。
- **播客永久加载困境**：一位用户发现 **Notebook LM** 播客生成器卡在“正在生成对话（Generating conversation）”状态，引发了对潜在性能瓶颈的担忧，并参考了 [NotebookLM 文档](https://support.google.com/notebooklm/answer/15678219)。
   - 参与者建议核实公交路线等数据输入，但尚未确认官方补丁或变通方法。
- **NotebookLM Plus 的进阶特权**：**NotebookLM Plus** 扩展了资源使用量并集成了 **Gemini Business**，参考了[此升级指南](https://support.google.com/notebooklm/answer/15678219)，并允许用户嵌入 PDF 和 YouTube 链接。
   - 然而，用户报告目前尚不支持**批量 YouTube 视频**上传，只能逐个插入链接。
- **语音变化烦恼**：成员们批评了**语音模型**在多语言表现上的不一致，参考 [Cloud TTS 文档](https://cloud.google.com/text-to-speech/docs/voice-types)寻找潜在解决方案。
   - 他们希望 2025 年的改进能解决音调稳定性和跨语言转换的问题。
- **紧急情况下的 UI 升级**：一些人认为新的 **NotebookLM** 界面过于拥挤，称其具有“幽闭恐惧感”，并希望获得更多屏幕空间。
   - 社区反馈呼吁提供高级布局选项，尽管目前尚未引用官方设计路线图。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 重叠与 HPC 收益**：成员们探讨了 [CUDA 中的重叠数据传输](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)和流体模拟调整，参考了 [fluidsCUDA](https://github.com/pixelfung/fluidsCUDA)。
   - 他们的目标是通过优化内存使用，将 15 秒的 GPU 运行时间缩短到接近 1 秒的 OPENMP 速度。
- **Genesis 模拟器神话破灭**：一篇新博客透露 **Genesis** 比旧的 GPU 模拟器慢达 **10 倍**，打破了此前 **430,000 倍提速**的说法。
   - [这一解释](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)与 [Stone Tao 的推文](https://x.com/stone_tao/status/1870243004730225009)一致，指出之前的指标大多是静态环境测试。
- **Triton 性能波折与 Kernel 技巧**：在向量加法（vector-add）测试中，Triton 的表现不如 Torch，直到用户发现是 **TRITON_INTERPRET=1** 导致了减速。
   - 他们还辩论了整数算术限制，以及手动 Kernel 调优是否能超越 Triton 的自动调度逻辑。
- **树莓派 5 LLM 测试与速度限制**：在配备 **VideoCore VII** 的树莓派 5 上使用 **llama.cpp** 进行试验时，Vulkan 后端遇到了编译障碍。
   - 同时，**Bielik-1.5B** 模型的运行速度维持在 **7 tok/sec** 左右，而 **OpenBLAS** 减慢了输入解码速度，而非提高输出速度。
- **新的 GPU 职位空缺与 HPC 忙碌**：[Cracked 研究工程师职位](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243)已发布，面向对高级 GPU 项目感兴趣的人士。
   - 成员们还在寻找位于旧金山的 CUDA 工程师职位、远程 LLM 基础设施工作以及 Triton Kernel 开发机会。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Pro Reasoning 与 Deepseek 的惊喜**：成员们注意到 [Perplexity 的 Pro Reasoning 模式](https://discord.com/channels/1047197230748151888/1054944216876331118/1323055993311203498) 在处理复杂查询时会自动启动，增强了 AI 的内部分析能力，而 **Deepseek** 则在不同的监管下运行。
   - 参与者想知道**中国的规则**如何赋予 Deepseek 更多灵活性，并引发了关于法律如何影响输出的讨论。
- **OpenAI 考虑 PBC 路径**：贡献者们讨论了 **OpenAI** 转向公共利益公司（Public Benefit Corporation）模式，旨在平衡利润与社会目标。
   - 他们将这一转变视为对问责制辩论的直接回应，并引用了关于商业 AI 应承担更广泛责任的论点。
- **Sonar 模型与 Perplexity API 备受关注**：成员们澄清了 **Sonar 模型** 擅长提供带有引用的实时网页答案，并建议不要将其分发到其他地方。
   - 其他人探讨了 **Perplexity AI API** 如何集成到未来的应用中，强调了增强 AI 驱动项目的潜力。
- **Discord 机器人进入高级功能领域**：一位用户希望利用 **Perplexity AI** 的高级会员权益创建一个 **Discord 机器人**，旨在为聊天体验提供高级功能。
   - 他们计划将这些权益整合到更具动态性的交互中，期待与 API 产生直接的协同效应。
- **随机视频与优化热议**：与会者评估了 [YouTube 的随机视频按钮](https://www.perplexity.ai/page/youtube-s-random-video-button-00KFpoLLThS8boDmTNk3wg)，以查看它是否能提高观众参与度。
   - 他们还指出了[内容优化技巧](https://www.perplexity.ai/search/how-can-i-optimize-content-for-K.VTSaD0R0yS2gxv7SXBuA)，重点强调了强大的关键词和受众洞察。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **指针转向：切换到 OwnedPointer**：Mojo 开发者将 **BoxPointer[Self]** 替换为 **OwnedPointer[Self]**，这让一些人措手不及，因为旧名称在 nightly 构建版本中消失了。他们强调了更安全的指针用法，以符合 **Mojo** 围绕引用和所有权更严格的不变量（invariants）。
   - 反馈显示，一些参与者最初难以找到新的指针类型，因此要求在文档中提供更清晰的参考。这次更名被誉为对 **Mojo** 指针体系的改进，尽管高级指针模式仍然让人感觉棘手。
- **自引用传奇：ArcPointer 登场**：Mojo 爱好者测试了 **ArcPointer** 在链式数据结构中用于共享引用的效果，发现可选引用（optional references）通常需要结构性重组。他们辩论是依赖 `ArcPointer` 还是重新组织代码以避免自引用陷阱。
   - 一些用户指出，如果使用不当，**UnsafePointer** 可能会引入风险。其他人建议采用替代设计，以获得更可预测的所有权模式和更清晰的生命周期规则。
- **破坏性变更：Mojo 的 6 个月重写周期**：Mojo 维护者确认在 **1.0** 版本之前，兼容性大约每六个月就会发生变化，这引发了对重复重写的担忧。用户对代码稳定性表示担忧，一些人甚至考虑将 **Rust** 作为备选方案。
   - 一些参与者对这些变化表示欢迎，认为这有利于在 **Mojo** 稳定之前进行快速迭代和完善。其他人则建议等待接近 **v1.0** 里程碑时再使用，以避免过多的迁移烦恼。
- **提升 'max'：API 现代化**：参与者观察到 **Mojo 的 'max'** 函数依赖于较旧的语义，且缺乏稳健的安全引用。他们建议进行彻底的 **API 审查**，以采用更精细的指针用法和高级类型特性。
   - 当前设置中的隐患可以通过更好地利用**值语义（value semantics）**和**移动语义（move semantics）**来修复。对更精简方法的呼吁凸显了 **Mojo** 强化其核心库的雄心。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Discord 困境：诈骗审查！**：成员们指出了 **Discord** 中反复出现的诈骗企图，敦促使用**手机验证**或验证码（captchas）来威慑恶意行为者，并提到攻击者是如何不断重新出现的。他们指出，虽然手机验证并不完美，但会增加每次诈骗尝试的成本。
   - 一些人将其描述为“机器人打地鼠”，认为对信任与安全的担忧掩盖了身份窃取和数据收割等**真实危害**。该小组建议采取紧急方法来保护空间免受渗透。
- **SD3 安全大辩论**：一些参与者辩论了 **SD3** 的**信任与安全**方面，部分人希望将这些措施扩展到社区的聊天环境中。他们认为，安全辞令往往会转移人们对紧迫的渗透企图的注意力。
   - 一位用户表示，这些策略分散了对诈骗的关注，揭示了产品营销姿态与真实安全之间的不匹配。另一位用户则认为，讨论被给社区带来负担的“持续渗透”所掩盖。
- **Stability.ai 中的 Faceswap 困局**：一位用户询问了 **Stability.ai API** 中的 **faceswap** 功能，寻找官方文档中缺失的细节。他们了解到，虽然存在图像处理功能，但缺乏针对 **faceswap** 强大的时序一致性（temporal consistency）。
   - 响应者强调了该库的局限性，表明它还不是高级面部重建的一站式解决方案。他们建议评估具有更可靠面部对齐功能的第三方工具。
- **LoRA 与 Checkpoint 的抉择**：**LoRA** 更新侧重于局部参数，而完全微调的 **checkpoints** 通常涉及更大的变化，但以磁盘占用为代价。成员们得出结论，两种方法都能产生类似的收益，但 **LoRA** 通常对资源更友好。
   - 一些人认为完全更新 **checkpoints** 最适合重大变革，但其他人发现 **LoRA** 是进行适度改进的理想选择。这种在尺寸和能力之间的平衡使得 **LoRA** 对那些 GPU 开销有限的人具有吸引力。
- **新手挑战模型修补！**：新用户介绍了自己，寻求关于 Prompt 设计和模型构建的技巧。一些人对 **checkpoint** 的创建感到迷茫，渴望得到有经验者的建议。
   - 资深用户表示欢迎，建议将 **LoRA** 或部分微调作为在没有巨大开销的情况下改进模型的有效方法。他们还分享了经过验证的迭代改进技巧。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **基于 Tanh 的 RMSNorm 引发讨论**：一种使用 **tanh** 来维持输入 2-范数的新型 **Lipschitz-1 RMSNorm** 变体因其在 **GANs** 和残差模型中的潜力而引起关注。
   - 怀疑者担心它可能会阻碍普通模型，但一致认为严格的 **Lipschitz** 边界对于稳定的**残差流（residual flows）**至关重要。
- **Pile 数据集的 260B Token 揭秘**：一次讨论指出 [这篇论文](https://pile.eleuther.ai/paper.pdf) 确认了在约 **825.18 GiB** 的 **Pile** 数据集中包含约 **260B GPT-2 tokens**，有时会上采样至约 **400B** tokens。
   - 参与者分析了实际 Token 数量与估计数量之间的差距，以微调训练设置。
- **神经 SDFs 与 NeRFs 获得 Lipschitz 关注**：成员们强调了 **Lipschitz** 边界如何加速**神经 SDFs** 和 **NeRFs** 中的网络追踪。
   - 他们将这些收益与 **RMSNorm** 方法联系起来，并看到了显著的性能提升。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 凭借 LlamaParse 自动模式势头大增**：Hanane Dupouy 展示了**优化的 RAG 流水线**如何使用 [LlamaParse 自动模式](https://t.co/WptFCIoyp6) 来平衡**财务报告**的成本和性能。
   - 成员们强调了**成本效益**和实时切换是主要优势，引发了关于改进数据处理的讨论。
- **Milvus + FAISS 混搭中的异常检测**：一位用户分享了一种用于异常检测的**混合方法**，结合了 Milvus 和 FAISS 来处理嵌入（embeddings）和聚类。
   - 其他人建议直接使用 **Milvus client** 以避开内存限制，并指出某些向量数据库会跳过存储嵌入。
- **聊天机器人并发难题**：长时间运行的后台任务导致了基于**多进程（multiprocess-based）**的延迟挑战，引发了关于管理聊天机器人并发性的辩论。
   - 社区成员建议使用 **asyncio.create_task** 进行异步操作，理由是其流程控制更精简且响应更快。
- **微调 Llama？有些好奇，但无具体步骤**：关于**微调 Llama 模型**的暗示出现了，但具体细节仅限于简短的提及。
   - 开发者对可能的扩展充满热情，尽管没有提供进一步的说明或代码。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **ModernBERT 微调模型涌现**：一个新的名为 `modernbert-embed-base` 的 **ModernBERT** 嵌入模型已经发布，它改进了 Tokenizer 并实现了更快的推理，详见 [Zach Nussbaum 的帖子](https://x.com/zach_nussbaum/status/1873813021786767699)。该模型在公开的 Nomic Embed 数据集上进行训练，为 Embedding 生成提供了一种替代方案。
   - 一些成员对 Twitter 上分享的**视觉表示**表示赞赏，认为 **ModernBERT** 是精细化大规模嵌入 (LSE) 迈出的坚实一步。
- **Arc AGI 图表再次确认 AI 势头**：[Douwe Kiela](https://x.com/douwekiela/status/1873755176940765300) 分享的一张进展图表证实，**AI 发展**没有放缓的迹象，该图表引用了原始的 **Dynabench** 论文。这张图表突出了模型性能在多个基准测试中的持续飞跃。
   - 成员们指出，这张图表提醒人们突破性进展不断出现的惊人速度，并敦促大家持续关注 **AGI 趋势**。
- **OpenAI 向营利性转型引发辩论**：[Jan Leike](https://x.com/janleike/status/1872909496777134127) 对 **OpenAI 转型**为营利性实体提出质疑，认为这削弱了其非营利愿景。批评者感叹，最初造福人类的使命现在被企业目标所掩盖。
   - 一些参与者认为这一举动是*不可避免的*，而另一些人则希望**非营利**端仍能捍卫伦理 AI 的理想。
- **Hugging Face 的 Agent 系统登场**：[Aymeric](https://x.com/aymericroucher/status/1874116324898598934) 宣布了一个名为 `smolagents` 的新 **Agent 系统**库，被誉为构建强大 Agent 的“最简库”。它专注于最小的代码开销和*自然代码编写*能力，使其有别于传统的工具包。
   - 社区对这种方法表示欢迎，认为它在现代 AI 工作流中具有简化 Agent 组装和快速原型设计的潜力。
- **ts_zip 提供实验性 LLM 压缩**：一种名为 **ts_zip** 的新型 LLM 驱动压缩工具出现，声称对文本文件具有更高的压缩率，详见[项目页面](https://bellard.org/ts_zip/)。它依赖 GPU 加速，且速度明显慢于标准压缩器。
   - 爱好者们渴望测试其早期阶段的优势，同时也承认其*实验性*状态和潜在的缺陷。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **HMM 的 Tokenization 沿用成熟方案**：一位成员确认 **隐马尔可夫模型 (HMM)** 的 **Tokenization** 保持不变，并参考了 2022 年早期框架的一致性。
   - 他们指出这些方法下的性能稳定，*HMM* 脚本无需修改，表明既有的最佳实践依然有效。
- **新年祝福，技术动态较少**：多位成员互致**新年**问候，标志着深度话题讨论的短暂休息。
   - 他们暂停了高级讨论以庆祝节日，没有提到进一步的进展更新或新发布。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Tinygrad 中的可逆谜题**：一位用户询问在**可逆变换**系统中，是否需要中间汇编步骤或直接的 uop 到二进制路径来生成**机器码**，并质疑这如何与最终的重写状态对齐。
   - 他们还探讨了每个变换是否转化为一个 **uop** 序列或最终的一对一映射，引发了关于 **tinygrad** 如何实现完全可逆性的好奇。
- **pcode 在 Tinygrad 中取得进展**：社区成员赞扬了 **sleigh 文档**，强调了 **pcode** 翻译与 **tinygrad** 中 **uop** 方法之间的共同理念。
   - 他们指出 **pcode** 定义以类似于汇编的风格处理 dtype 和元数据，引发了关于如何将这些概念融入 **tinygrad** 的推测。
- **新手指南与内部机制介绍**：一位用户在“good first issue”之外寻求适合新手的任务，随后有人推荐了 [tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes)，以获取关于 **tinygrad** 基础知识的逐步帮助。
   - 贡献者们还分享了[一份关于 tinygrad 内部机制的新介绍](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241231_intro.md)，呼吁提供更多学习材料和社区贡献。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **GH200 访问触发调试行动**：一名成员请求 **GH200 访问权限**以运行 Python 复现脚本并验证 **D2H 内存传输**配置。
   - 他们希望确保问题并非由本地设置的特殊性引起，并确认不同系统间行为的一致性。
- **D2H 内存传输引发关注**：聊天中指出特定配置可能导致 **D2H 内存传输**故障。
   - 他们强调要交叉检查设置，以排除非预期的设备或驱动程序不匹配导致的问题。



---



## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **DeepSeek 稳定，GigaChat 尚未尝试**：一位成员报告 **DeepSeek Coder V2 Lite** 表现可靠，在代码任务中显示出一致的结果。他们没有尝试 **GigaChat**，因此该模型的能力尚待探索。
   - 虽然没有提供 Benchmark 数据，但人们对 **GigaChat** 在未来测试中的功能感到好奇。
- **提到 Modernbert 与 Localdocs 嵌入**：一位参与者在 Hugging Face 上看到了 **Modernbert**，并提出了关于增强 **localdocs** 嵌入后端的问题。他们建议这些更新可以提升文本分析或检索任务。
   - 这反映了社区对不断演进的 Embedding 方法的关注，期待与 **Modernbert** 的顺利集成。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **无重大更新 #1**：提供的内容中未出现先进的技术或产品进展。
   - 仅提到 MOOC 报名日期，缺乏针对 AI 工程受众的新模型、数据集或关键突破。
- **无重大更新 #2**：未分享关于新 Benchmark 或工具的额外讨论或相关参考。
   - 社区关于课程物流的咨询未达到深度报道或分析的标准。



---


**DSPy Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Torchtune Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**LAION Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**OpenInterpreter Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。


---

# 第 2 部分：各频道详细摘要与链接


{% if medium == 'web' %}

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1323443464372752398)** (60 条消息🔥🔥): 

> `User Prompt 额度问题、Flex Credits 延迟、Windows 兼容性担忧、Code Completion 上下文、订阅困惑` 


- **User Prompt 额度查询与支持响应**：一名用户询问如何购买额外的 **User Prompt credits**，认为流程不清晰，而另一名用户确认他们很快得到了支持协助。
   - *Southbayjay* 建议检查账单页面，这引发了关于支持响应速度的褒贬不一的评价。
- **Flex Credits 购买问题**：一名用户报告购买了 **$10 flex credits**，已被扣款但四天后仍未到账，引发了对客服可靠性的担忧。
   - 另一名用户的体验较为顺畅，但对在现场演示期间使用 flex credits 的效用表示焦虑。
- **Windows Subsystem for Linux (WSL) 评论**：一名用户对 **Windows Subsystem for Linux (WSL)** 上的代码执行不一致表示沮丧，强调了有效运行代码的困难。
   - 这引发了关于软件开发中更倾向于使用 Linux 而非 Windows 的讨论。
- **Code Completion 上下文的挑战**：一名用户询问代码补全是否可以访问依赖项的源代码，回复建议上下文可能仅限于项目的结构。
   - 几位成员提供了潜在的变通方法，包括固定相关文件以及利用项目文档来辅助代码补全。
- **订阅状态困惑**：一名用户发现其 **Pro Ultimate subscription** 在没有解释的情况下降级为免费计划，表达了解决问题的紧迫性。
   - 其他人也有类似的被锁定感，并被引导创建支持工单以解决账户问题。


---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1323383782027428063)** (320 条消息🔥🔥): 

> `Windsurf 反馈与性能, Windsurf 功能与局限, Codeium 工具对比, 用户体验与问题, 数据隐私与 AI 伦理` 


- **Windsurf 的性能问题**：多位用户报告称 Windsurf 的响应时间往往非常长，提示词之间的等待时间超过 20 分钟，尤其是在使用 Pro Ultimate 方案时。
   - 一些用户建议需要改进，理由是对 AI 逻辑的担忧，以及需要适当的基于项目的规则来减少错误。
- **对网络爬虫功能的请求**：用户一直对 Windsurf 抓取网页和特定仓库的能力感兴趣，并渴望了解该功能何时推出的更新。
   - 与此同时，一位用户建议使用 [Gitingest](https://gitingest.com/) 将 Git 仓库转换为文本格式，以便 LLM 摄取。
- **与其他工具的对比**：用户讨论了 Windsurf、Cursor 和其他 AI 代码助手之间的对比，指出虽然 Windsurf 功能全面，但一些人发现像 Continue 这样的工具具有某些优势。
   - 社区表示希望继续探索替代方案，特别是关于 Cascade Base 的能力和性价比。
- **关于 AI 伦理和数据隐私的讨论**：用户对 AI 工具中的数据隐私表示担忧，对在专有软件中使用敏感代码持怀疑态度。
   - 虽然有些人信任 Codeium，但其他人仍保持谨慎，更倾向于开源选项以降低风险，并强调了伦理 AI 实践的重要性。
- **用户体验与调试挑战**：一些用户分享了他们在 Windsurf 中遇到错误的挫败感，报告了诸如屏幕冻结和 AI 建议后代码损坏等问题。
   - 大家的共识是，虽然 AI 可以提供帮助，但监督是必要的，以避免重大干扰，并讨论了如何有效地引导 AI 以获得更好的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/getstarted/overview">未找到标题</a>：未找到描述</li><li><a href="https://discordapp.com/channels/1027685395649015980/1027698450562814042/1323465463870656595">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友放松或建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://docs.codeium.com/best-practices/use-cases#search-repo-for-apis-with-natural-language-and-generate-code-for-integrations">常见用例 - Codeium 文档</a>：未找到描述</li><li><a href="https://tenor.com/view/burn-elmo-pyro-burn-it-down-ashes-gif-5632946">Elmo 焚烧 GIF - Burn Elmo Pyro - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/home-alone-when-shes-alone-tesla-jump-im-coming-gif-16164614">小鬼当家当她一个人在家时 GIF - Home Alone When Shes Alone Tesla - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gitingest.com/">Git ingest</a>：在任何 GitHub URL 中将 'hub' 替换为 'ingest'，即可获得适合提示词的文本</li><li><a href="https://codeium.com/windsurf">Codeium 开发的 Windsurf 编辑器</a>：未来的编辑器，就在今天。Windsurf 编辑器是首个由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器和 Codeium 扩展</a>：需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.com/blog/pricing-windsurf">方案与价格更新</a>：我们对 Cascade 定价模型进行了一些更改。</li><li><a href="https://github.com/orgs/Exafunction/repositories?type=all">Exafunction</a>：Exafunction 有 38 个可用仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1323384409499635767)** (269 条消息🔥🔥): 

> `AI 在治疗中的应用与保密性，LLM 在代码生成中的比较，医疗保健中的数据隐私与安全，AI 回复的功能性与简洁性，AI 伦理与患者信息` 


- **AI 在治疗中引发隐私担忧**：参与者讨论了治疗师在确保患者保密性的同时如何使用 AI 工具，强调了对敏感信息进行去标识化（depersonalizing）的重要性。
   - 一位成员指出，即使数据经过匿名化处理，AI 仍有可能根据数据中的独特模式识别出个人身份。
- **探索用于代码生成的 LLM**：用户分享了使用 Claude 3.5 Sonnet 和 Haiku 等不同 LLM 生成简洁代码的经验，其中一位成员测试了一个高效的 system prompt。
   - 尽管最初节省了 token，但进一步探索发现，其有效程度各异，尤其是在处理更复杂的代码时。
- **医疗保健中的数据隐私法规**：对话强调了英国在管理患者数据方面的严格法规，重点是保护个人健康信息。
   - 成员们讨论了研究所需的访问权限与维护患者护理保密性之间的平衡。
- **AI 输出的简洁性与可读性**：关于 AI 回复中的简洁编码是否牺牲了可读性和功能性展开了辩论，一些人倾向于高度简洁的方法。
   - 参与者承认，虽然像 Claude 这样的模型可以生成简洁的代码，但理解和调试可能仍需要更具可读性的格式。
- **AI 在医学中的伦理影响**：讨论了在医疗保健中使用 AI 工具的伦理影响，特别是关于数据管理和患者隐私。
   - 成员们对数据滥用的可能性以及在使用 AI 技术时需要严格保护措施表示担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2202.08587">Gradients without Backpropagation</a>: 使用反向传播来计算目标函数的梯度以进行优化一直是机器学习的支柱。反向传播，或称反向模式微分，是一个特例...</li><li><a href="https://arxiv.org/abs/2411.05873">Poor Man&#39;s Training on MCUs: A Memory-Efficient Quantized Back-Propagation-Free Approach</a>: 反向传播 (BP) 是神经网络训练中梯度计算的默认方案。然而，在 FPGA、微控制器 (MCU) 等各种边缘设备上实现基于 BP 的训练...</li><li><a href="https://tenor.com/view/mario-on-ice-gif-12914463099975653658">Mario On Ice GIF - Mario on ice - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://xkcd.com/2318/">Dynamic Entropy</a>: 未找到描述</li><li><a href="https://www.bbc.co.uk/news/articles/c78llg7n5d5o">Watchdog set to fine NHS IT firm after medical records hack</a>: 2022 年的泄露事件涉及 890 人的医疗记录和进入其住所的信息。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1323398498779140148)** (6 条消息): 

> `LlamaCpp Discord, Hermes 3 Amnesia 复现` 


- **在 Discord 上寻找 LlamaCpp 开发者**：一位成员表示需要找到 **LlamaCpp** 开发者可能活跃的 **Discord**。
   - 另一位成员建议在 **GitHub** 上提交 issue 或发起讨论，这是联系他们的最佳方式。
- **理解代码复杂性**：在继续分析代码时，一位成员指出他们面临的问题**并非显而易见能解决的**。
   - 他们提到目前的方法是加深对代码库的理解。
- **在 Hermes 3 中复现 Amnesia**：一位成员请求协助使用 **Hermes 3** 复现 **Amnesia**，特别是非 405b 版本。
   - 作为回应，另一位成员幽默地建议，诀窍只需简单地移除 **prompt** 即可达到预期效果。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1323395181810286593)** (166 条消息🔥🔥): 

> `OpenAI 的 Discord 互动、与 Gemini 2 Flash 的竞争、API 使用与模型测试、内容审核挑战、用户对 AI 模型的见解` 


- **OpenAI 用户对互动表示沮丧**：许多用户对 OpenAI 在 Discord 中缺乏响应式互动表示担忧，有些人甚至指责他人是受雇的演员。
   - 提到了 *"I do not spend any time on Anthropics discord, and I am not even sure it exists?"*（我从不在 Anthropic 的 Discord 上花时间，我甚至不确定它是否存在？），以此来强调平台之间的差异。
- **像 Gemini 2 Flash 这样的竞争对手正在取得进展**：*Gemini 2 Flash* 推出了实时搜索等功能，一些用户认为这给 OpenAI 带来了压力，迫使其在产品中追赶类似的功能。
   - 用户期待 OpenAI 在 *O1 Pro* 等模型中实现搜索功能，并强调竞争可以推动改进。
- **API 使用的多样性与担忧**：几位用户分享了使用包括 *Anthropic* 和 *OpenAI* 在内的不同 AI API 的见解，并讨论了使用成本和模型优势。
   - 一位用户提到查看了他们的 API 支出，声称 *"I used $130 last month"*（上个月我花了 130 美元），主要是通过大批量运行产生的。
- **敏感话题的内容审核障碍**：用户讨论了 AI 在为敏感文档（如涉及未成年人的文档）生成关键词时面临的内容审核挑战，这表明 AI 在理解上下文方面存在困难。
   - 一位用户的解决方案是禁用审核功能，并表示 *"Gotcha, so I just need to turn it off, lol."*（明白了，所以我只需要把它关掉，哈哈。）
- **GPT-4o 的角色一致性问题**：一位用户询问如何复制最初 GPT-4o 发布时展示的角色一致性水平，这表明目前模型的行为存在局限性。
   - 其他人则反思了 4o 模型缺乏原生图像生成功能，对其能力表示失望。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1323428455190368327)** (6 条消息): 

> `脚本更新、编程协助、社区支持` 


- **增强型脚本实现了电影化目标**：一位用户在更新了脚本以获得“更连贯的电影体验和自然动作”后表示满意，展示了社区内的协作努力。
   - 分享的 [Discord 消息](https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158) 反映了积极的参与和脚本质量的提高。
- **非编程人员通过社区获得信心**：一位用户提到，通过让其他人解释代码结构并实施建议的更改，他们对学习编程感到兴奋。
   - 这展示了社区的互助性质，因为他们鼓励成员走出舒适区。
- **对社区学习之旅的感激**：一位贡献者表达了对社区的感谢，指出他们在短短一年内从“零基础”成长为技能显著提高。
   - *社区支持* 在个人学习中起着至关重要的作用，营造了一个知识共享和发展的环境。



**提到的链接**: <a href="https://discordapp.com/channels/974519864045756446/1315696747279810711/1323428129083097158">Discord - 充满乐趣与游戏的群聊</a>: Discord 非常适合玩游戏和与朋友闲逛，甚至可以建立一个全球性的社区。定制你自己的空间来聊天、玩耍和聚会。

  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1323402298746994729)** (18 条消息🔥): 

> `Prompt 清晰度, Discord 中的 Markdown 使用, LexiDeck 框架, 简化专题研究, Discord Prompt 库` 


- **优先考虑直接的 Prompt**：一位成员认为最好的 Prompt 是最直接的，强调了尽管 ChatGPT 具有对话式设计，但清晰度的重要性。
   - *即使实施了自定义指令，在 Prompt 中保持非常清晰和简洁也是有帮助的*。
- **Discord 频道中的 Markdown 争议**：成员们对 Prompt Engineering 频道缺乏 Markdown 支持表示沮丧，称这使得分享示例变得困难。
   - 一位成员解释说，*如果没有 Markdown，这个频道就不再是关于 Prompt Engineering，而更多是关于与 AI 的对话*。
- **LexiDeck 框架介绍**：一位成员介绍了 LexiDeck 框架，这是一个用于 ChatGPT 的 Multi-Agent Prompt 工具，旨在简化交互。
   - 他们指出 LexiDeck 目前正处于更新过渡期，且缺乏 Canvas 支持。
- **关于活动研究 Prompt 的建议**：一位成员分享了一个 Prompt，用于协助研究大预算专题的地点、供应商和活动。
   - 另一位成员强调了 Prompt 中具体细节的重要性，以获得更好的协助并简化研究流程。
- **在 Discord 中寻找 Prompt 库**：一位成员询问 OpenAI Discord 内部是否存在 Prompt 库。
   - 另一位成员提供了一个指向潜在相关资源的链接，建议以此作为访问过往讨论的方式。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1323402298746994729)** (18 条消息🔥): 

> `直接 Prompt 的有效性, Discord 中的 Markdown 使用, LexiDeck 框架, 专题制作研究, Prompting 技巧` 


- **直接的 Prompt 效果最佳**：一位成员强调，ChatGPT 最好的 Prompt 是最直接的，并指出虽然直接性有帮助，但对话性质可能仍需要一些往复沟通。
   - *由于模型反应不同，确保每次对话的一致性具有挑战性*。
- **频道中的 Markdown 限制**：一位成员对 Discord 频道缺乏 Markdown 表示遗憾，分享说这导致丢失了一条包含有用建议且精心编写的消息。
   - 另一位成员建议，允许 Markdown 将有助于分享正确的示例和 Prompt Engineering 讨论。
- **LexiDeck 框架介绍**：一位成员介绍了他们名为 LexiDeck 的框架，该框架对 ChatGPT 交互采用了 Multi-Agent 方法，尽管目前缺乏 Canvas 支持。
   - LexiDeck 源于希腊语和赛博朋克根源，象征着其对文字和重度计算的关注。
- **简化专题制作研究**：一位成员寻求帮助以简化大预算专题的研究，讨论了在电子表格中组织地点、供应商和活动的需求。
   - 另一位成员建议使用示例 Prompt 来寻求模型协助，并建议沟通中的具体性会带来更好的回复。
- **与 ChatGPT 有效互动**：参与者讨论了与模型有效沟通的策略，强调解释请求时的清晰度是获得满意协助的关键。
   - 鼓励用户像与博学的人交流一样与模型沟通，以改善交互结果。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1323439583614079026)** (43 条消息🔥): 

> `LM Studio 图像生成, LM Studio 更新问题, ML/LLM 领域的就业机会, Steiner 推理模型, 云端 VM 管理` 


- **LM Studio 缺乏图像生成功能**：一位用户询问了在 **LM Studio** 中生成图像的可能性，另一位用户确认目前不支持该功能。
   - 这表明用户正在探索该软件的创意能力。
- **LM Studio 更新问题**：一位用户报告在将 LM Studio 更新到 **0.3.5 (build 2)** 时，遇到了 macOS 系统提示请求权限的消息。
   - 有人建议这可能与 'Squirrel' 更新系统有关，并且在未来的更新中将不再是问题。
- **ML/LLM 领域的就业前景**：一位用户表示担心，进入 **ML/LLM** 工作岗位似乎仅限于拥有多个学位的计算机科学毕业生。
   - 这突显了在不断发展的机器学习领域中，人们感知到的准入门槛。
- **探索 Steiner 推理模型**：一位用户分享了在 Hugging Face 上发现的 **Steiner 推理模型**，该模型在 LM Studio 中表现良好。
   - 该模型处理推理任务的独特能力在某些场景下似乎超过了像 Llama 3.3 Q4 70B 这样的大型模型。
- **管理云端 VM**：几位用户讨论了他们使用云端 VM 的经验，特别是容易忘记正在运行的设置。
   - 一位用户强调了仔细预算和管理使用情况以避免意外费用的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/new-year-happy-new-year-penguin-pudgy-new-years-gif-9445254434650071400">新年快乐 GIF - 新年快乐企鹅 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/peakji/steiner-32b-preview-gguf">peakji/steiner-32b-preview-gguf · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1323643423781031947)** (149 messages🔥🔥): 

> `Llama 3.2, Coral AI TPUs, GPU Alternatives, Groq LPU Inference Engine, MacBook Pro Performance` 


- **Llama 3.2 的模型大小与性能**：一位成员指出 **Llama 3.2 1b** 的模型大小在 **2GB** 以下，并质疑其与功耗最高似乎为 **16 watts** 的 **Coral.ai TPUs** 的兼容性。
   - 针对 TPU 在 LLM 上的有限应用场景提出了担忧，并建议考虑其他加速器。
- **LLM 的 GPU 替代方案**：针对高功耗 GPU 的替代建议包括 **Jetson Nano** 和 **Mac Mini**，讨论强调了性能优势与功耗之间的平衡。
   - 强调了对**低功耗选项**的需求，特别是对于由游戏后端的 Java 应用处理的任务。
- **Groq LPU Inference Engine 的速度**：**Groq 的 LPU Inference Engine** 因其效率受到称赞，拥有 **241 tokens per second** 的吞吐量，其性能指标和价格引起了关注。
   - 出现了关于各种单元 RAM 规格的问题，特别是 **Groq LPU** 与 **Cerebras WSE-3** 等其他模型之间的差异。
- **用于 AI 工作负载的 MacBook Pro**：一位成员建议，从 **16GB MacBook Pro** 升级到 **32GB 型号** 可能不会对 LLM（尤其是写作任务）带来显著的性能提升。
   - 共识倾向于尽可能扩大 RAM 容量，一些人主张如果预算允许应选择 **128GB**，以便更高效地处理**更大的模型**。
- **CPU 性能与推理**：讨论指出 CPU 推理速度受限于 RAM 速度，较小的模型（≤3b）在消费级 CPU 上表现尚可。
   - 一些成员对 CPU 在 LLM 上的可行性表示怀疑，更倾向于使用专用资源，同时承认内存带宽（memory bandwidth）的重要性。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://imgur.com/a/lNUBRMY">imgur.com</a>: Discover the magic of the internet at Imgur, a community powered entertainment destination. Lift your spirits with funny jokes, trending memes, entertaining gifs, inspiring stories, viral videos, and ...</li><li><a href="https://cryptoslate.com/groq-20000-lpu-card-breaks-ai-performance-records-to-rival-gpu-led-industry/">Groq&#039;s $20,000 LPU chip breaks AI performance records to rival GPU-led industry</a>: Groq&#8217;s LPU Inference Engine, a dedicated Language Processing Unit, has set a new record in processing efficiency for large language models. In a recent benchmark conducted by ArtificialAnalysis....</li><li><a href="https://inference.cerebras.ai/">Cerebras Inference</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=GBR6pHZ68Ho"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1323390293340324001)** (130 messages🔥🔥): 

> `Deepseek performance, Video transcription solutions, O1 API access criteria, Architect mode, Model limitations and improvements` 


- **Deepseek 的采用与令人印象深刻的特性**：许多用户表达了对 **Deepseek** 的满意，指出其性能优于 **Sonnet** 等旧模型，并声称它解决了竞争对手遇到的问题。
   - 用户强调了其极快的输出速度，尽管有些人仍在寻找降低速度以提高可读性的方法。
- **视频转录的困扰**：一位用户询问如何从视频（特别是引用了一个 **YouTube** 链接）中获取转录文本，对快速呈现的大量信息感到沮丧。
   - **Whisper** 等工具为视频转录提供了解决方案，一些人分享了高效获取 Transcript 的脚本。
- **对 O1 API 访问权限的关注**：几位成员讨论了他们组织对 **O1** 和 **o1-preview** 的访问权限，尽管处于同一层级，但可用性却各不相同。
   - 出现了关于访问 **O1 API** 当前标准的查询，用户寻求澄清各自组织的限制。
- **Architect 模式的使用体验**：一位用户分享了在 **Architect 模式** 下的挣扎，特别是在尝试从头开始构建新应用的脚手架（scaffold）时，最终选择了使用现有文件。
   - 对话表明 Architect 模式内的脚手架构建过程普遍需要改进。
- **探索模型局限性**：围绕各种模型局限性的对话（特别是 **Deepseek v3**）引发了对推理速度和输出准确性的讨论。
   - 用户提出了未来的改进建议，并对技术进步使这些模型随着时间的推移变得更快、更高效表示乐观。


  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1323387609933811722)** (39 条消息🔥): 

> `Aider 命令执行、Token 限制错误、使用基于文件的 Prompt、模型切换、Web UI 开发` 


- **Aider 的命令执行限制**：用户对 Aider 在没有人工批准的情况下无法直接运行 shell 命令表示沮丧，尽管设置了 `/run` 命令和 `AIDER_YES_ALWAYS` 变量。
   - 有人指出，这种限制是为了安全起见，因为来自 LLM 的命令需要人工验证，以防止产生意外后果。
- **对 Token 限制的困惑**：一些用户在使用了 `/clear` 命令并减少上下文后，仍然遇到 **token limit errors**，这表明该命令可能无法完全重置 token 计数。
   - 一位用户建议检查 `/tokens` 命令以获取详细分析，从而了解 token 使用方面的持续问题。
- **在 Aider 中利用基于文件的 Prompt**：一名成员询问关于使用 Markdown 文件跟踪 Aider 进度的问题，以及它是否可以实现类似于 **clinerules** 的功能。
   - 另一个建议是将 prompt 保存在专用文件中，以便在不同会话中轻松重复使用，避免重复输入。
- **通过脚本优化模型使用**：讨论集中在将 **DeepSeek** 作为主要编辑模型，同时仅将 **o1** 用于架构师（architect）任务，并考虑通过脚本实现更简单的模型切换。
   - 想法包括使用类似 `/ask-arch` 的命令或使用智能注释，以便在需要时高效地调用更强大的模型，而不浪费 token。
- **Aider Web UI 的开发**：有人询问了 Aider 的 **Web UI** 版本的进展情况，表示对进一步开发和功能的兴趣。
   - 用户渴望了解新的 Web 界面可能会如何改变他们与工具的交互，以及它可能带来的任何增强功能。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1323715027416645693)** (2 条消息): 

> `WebDev Arena、AI 竞技场排名、Claude 模型评分、Gemini 性能、GPT-4o 更新` 


- **WebDev Arena 推出竞争性网站生成器**：介绍了新的 [WebDev Arena](https://web.lmarena.ai)，用户在这里竞争构建最好的网站，并提供实时排名更新。
   - 在[此处](https://web.lmarena.ai/leaderboard)查看当前排行榜。
- **Claude 模型在网站构建评分中占据主导地位**：**Claude 3.5 Sonnet** 以 **1218.58** 的竞技场得分领跑比赛，展示了 Anthropic 模型的强劲性能。
   - **Haiku** 变体紧随其后，得分为 **1137.96**，两款模型都获得了大量投票。
- **Gemini 模型表现出强劲的竞争力**：包括 **Gemini-2.0-Flash-Thinking-1219** 在内的多个版本的 **Gemini** 排名靠前，得分在 **1022.92** 左右。
   - 值得注意的是，**Gemini-Exp-1206** 和 **Gemini-2.0-Flash-Exp** 也表现出了极具竞争力的性能，保持了 Google 在排行榜上的地位。
- **OpenAI 的 o1-mini 和 GPT-4o 参与排名**：**o1-mini** 获得了 **1065.10** 的分数，而 **GPT-4o-2024-11-20** 得分为 **964.35**，位列顶尖表现者之列。
   - OpenAI 的模型继续进化，并在投票过程中保持高参与度。
- **DeepSeek 和 Qwen 模型展示独特优势**：**DeepSeek v3** 以 **960.01** 的得分脱颖而出，而 **Qwen2.5-Coder-32B-Instruct** 以 **909.17** 紧随其后，展示了多样化的能力。
   - 这些表现突显了竞争激烈的 AI 领域中各种不同的方法。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1323385550840791052)** (118 messages🔥🔥): 

> `Unsloth 集成, Hymba 模型讨论, Fine-tuning 技术, Continued pretraining, Unsloth 社区反馈` 


- **关于 Unsloth 集成的讨论**：一名成员表示有兴趣将不同的组件集成到他们的训练流水线中，特别是通过堆叠两个 LLM 来实现自定义方法。
   - 这引发了关于现有模型是否能有效支持此类集成的对话。
- **Hymba 模型能力**：成员们讨论了 [Hymba-1.5B-Instruct](https://huggingface.co/nvidia/Hymba-1.5B-Instruct) 模型，注意到它处理复杂任务的能力，以及尽管目前存在支持问题，但在 Unsloth 中的使用情况。
   - 成员们对兼容性表示担忧，现有的模型在 Unsloth 框架中并未按预期运行。
- **有效的 Fine-tuning 策略**：讨论中出现了关于 Fine-tuning 实践的内容，建议通过监控 loss 来确定最佳的 epochs 数量。
   - 普遍共识建议从 3 个 epochs 左右开始，同时考虑特定数据集的动态变化。
- **Continued Pretraining 的挑战**：一位成员详细描述了在保加利亚语数据集上进行 Continued pretraining 过程中的挫折，指出了其中的复杂性。
   - 他们发现，与预训练大型模型所需的大量资源相比，embedding 模型训练效率更高。
- **Unsloth 开发的社区建议**：向社区征集关于 Unsloth 在 2025 年未来功能的反馈，欢迎各种建议。
   - 鼓励成员就缺失的功能、文档改进和整体可用性发表看法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 进行 Finetuning | Unsloth 文档</a>：Checkpoint 允许你保存 Fine-tuning 进度，以便暂停后继续。</li><li><a href="https://x.com/UnslothAI/status/1874146501019963821">Unsloth AI (@UnslothAI) 的推文</a>：学习如何在 6 分钟内免费 Fine-tune Llama！在这段视频中，@jasonzhou1993 使用 Unsloth 和自定义数据集对 Llama 3.2 (3B) 进行 Fine-tune，显著增强了 MidJourney 提示词。Jason 涵盖了...</li><li><a href="https://tenor.com/view/always-has-been-among-us-astronaut-space-betrayal-gif-23836476">Always Has Been Among Us GIF - Always Has Been Among Us Astronaut - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/nvidia/Hymba-1.5B-Instruct">nvidia/Hymba-1.5B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hqkeyn/what_would_you_like_to_see_in_unsloth_for_2025/">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1323391073648640060)** (2 messages): 

> `Discord 服务器致谢, 框架反馈` 


- **对 Discord 框架的支持**：一位成员对 Jed.T 表示**全力支持**，称赞 **Discord 服务器和框架**非常棒。
   - 另一位成员也回应了“非常感谢”，表达了对持续付出的感激。
- **社区鼓励**：另一位成员强化了这种情绪，对那些为**服务器氛围**做出贡献的人表示感谢。
   - 这展示了**积极的社区精神**，促进了成员之间的协作和赞赏。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1323494672768696425)** (5 messages): 

> `Unsloth 文档, Fine-tuning LLaMA 3, 个人助手创建` 


- **Fine-tuning LLaMA 3 的逐步指南**：一位用户分享了一份详尽的[文档](https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama)，其中包含在 **Ollama** 中使用 **LLaMA 3** 进行 Fine-tuning 的逐步教程。
   - 该教程旨在指导用户创建一个类似于 **ChatGPT** 的定制化个人助手，并在本地运行。
- **对 Unsloth 创作者的感谢**：一位用户表达了对 **Unsloth** 创作者的感激之情，称赞其周到的设计和详尽的文档。
   - *Theyruinedelise* 接受了这份感谢，进一步肯定了对该平台及其开发者的正面反馈。



**提到的链接**：<a href="https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama">教程：如何 Fine-tune Llama-3 并在 Ollama 中使用 | Unsloth 文档</a>：在 Ollama 上本地运行定制化个人助手（如 ChatGPT）的初学者指南

  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1323397055066144778)** (8 messages🔥): 

> `Test Time Training (TTT), ARC 性能提升, RL 比较, 模型参数更新` 


- **理解 Test Time Training (TTT)**：Test Time Training (TTT) 涉及在推理（inference）过程中根据输入数据产生的损失（loss）临时更新模型参数，旨在增强推理能力。
   - 一位成员建议*应该存在*类似于 Reinforcement Learning (RL) 的相关机制，表明方法论上可能存在重叠。
- **TTT 在 ARC 上的显著成果**：TTT 在 **Abstraction and Reasoning Corpus (ARC)** 上的表现显示出显著改进，准确率比基础微调模型（fine-tuned models）提升了高达 **6 倍**。
   - 来自[这篇论文](https://arxiv.org/abs/2411.07279)的初步发现强调了 TTT 如何增强模型在推理任务中的效能。
- **关于论文和代码可用性的讨论**：一位成员指出需要调查现有的关于 TTT 的论文，并建议可能已经发布了与该概念相关的代码。
   - *你得去挖掘一下相关的论文*，这引发了对进一步探索可用资源的关注。



**提到的链接**：<a href="https://arxiv.org/abs/2411.07279">The Surprising Effectiveness of Test-Time Training for Abstract Reasoning</a>：语言模型在训练分布内的任务中表现出色，但在处理需要复杂推理的新问题时往往力不从心。我们研究了 TTT 的有效性...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1323464630978478140)** (6 messages): 

> `Token 消耗关注, 项目重载方法, UI 设计的数据收集, Bolt 中的表格数据格式, 编码问题与语言兼容性` 


- **Token 消耗问题引发关注**：一位成员分享说，他们在将 ChatGPT 用于架构任务和 **Bolt prompt 增强器**时，仅两天内就**消耗了超过 30M tokens**。
   - 他们强调了有效管理额度的重要性，以避免在细微修改上产生过度支出。
- **关于项目重载技术的争论**：关于如何在 Bolt 中**重载项目**提出了疑问，特别是应该刷新浏览器还是使用专门的重载按钮。
   - 另一位成员讨论了他们严谨的审查流程，利用 Claude 等工具进行特定页面的代码修复。
- **使用 Bolt 简化 UI 设计**：一位成员分享了他们的策略，即在添加 API 和动画等元素之前，先收集**必要的数据**来简化项目设置。
   - 完成后，他们强调了使用 Bolt 命令在不同板块间**复制设计**是多么容易。
- **Bolt 中的表格数据格式**：关于在 Bolt prompt 中**提供表格数据**的首选格式提出了疑问，特别是 .csv 是否为首选格式。
   - 这表明需要明确格式标准以确保无缝集成。
- **应对编码挑战**：一位成员指出，编码问题可能源于早期的技术栈问题，并强调了**具备编程素养**的重要性。
   - 他们断言，理解编程语言及其特性对于成功构建项目至关重要。



**提到的链接**：<a href="https://usmanaicareer.netlify.app/">Vite + React + TS</a>：未找到描述

  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1323394871255633981)** (106 条消息🔥🔥): 

> `Bolt Pro 订阅, Bolt 的 Git 集成, Facebook API 集成挑战, AI 工具的用户体验, 新年问候` 


- **了解 Bolt Pro 订阅**：一位用户询问订阅 Bolt Pro 是按月还是按天提供 Token，另一位用户确认是按月提供。
   - 讨论揭示了用户对于 Bolt 环境内的项目管理和操作限制普遍存在困惑。
- **Git 集成仍待完善**：一位用户表达了将项目导出到 Git 的愿望，并指出目前尚不支持该集成，且未提供未来更新的时间表。
   - 这引发了一个提议，即此类查询更有可能由社区成员而非官方支持提供反馈。
- **Facebook API 集成挑战**：多位用户讨论了将 Facebook Marketing API 与 Bolt 集成的困难，并指出在未能成功连接的情况下消耗了大量的 Token 成本。
   - 一位用户强调了他们在同步 Facebook 个人资料数据方面的进展，同时正在寻求高级权限以实现进一步的功能。
- **AI 工具的混合体验**：用户分享了使用 AI 工具构建应用程序的各种经历，一些人对其实际能力与预期之间的差距表示失望。
   - 用户对创建更复杂应用程序时感知到的效率低下和挑战表示担忧，这通常需要相当程度的编程知识。
- **社区参与和新年问候**：频道内出现了成员们爆发式的新年问候，营造了友好的用户氛围。
   - 用户们互相提供支持和鼓励，增强了社区在应对技术挑战时的协作精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://localhost:port">未找到标题</a>: 未找到描述</li><li><a href="https://feego.xyz/">Feego - Connect & Trade</a>: 未找到描述</li><li><a href="https://tenor.com/view/shake-my-head-mike-mclusky-mayor-of-kingstown-smh-disappointed-gif-293488442475603142">Shake My Head Mike Mclusky GIF - Shake my head Mike mclusky Mayor of kingstown - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="http://bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: 使用任何你想要的 LLM 来提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1323381494428729424)** (105 条消息🔥🔥): 

> `DeepSeek, Web Hosting 选项, Chatbot 开发, 调试与错误, GitHub 新功能` 


- **DeepSeek 性能与对比**：用户讨论了在 Cursor 中使用 **DeepSeek v3** 的体验，指出它在处理大型数据库和复杂查询方面非常有效。
   - 一位用户强调了 DeepSeek 与其他模型的对比，对其能力表示兴奋，而其他用户则试图澄清其可用性。
- **为应用选择合适的托管方案**：多位用户分享了关于部署应用托管选项的看法，推荐 **Hetzner** 和 **Digital Ocean**，因为它们价格实惠且易于设置。
   - 其他用户提到利用 **Vercel** 处理前端，**AWS** 处理后端，并指出具备 Docker 经验会大有裨益。
- **Chatbot 开发见解**：一位用户询问了关于构建 Chatbot 的问题，并收到了查看 GitHub 上基于 **Next.js** 和 **shadcn** 框架仓库的建议。
   - 社区提供了示例项目的链接，强调了为了有效实现 Chatbot，需要 API key 和安装说明。
- **使用情况追踪与错误问题**：关于 **Cursor 使用情况追踪** 更新的担忧被提出，因为许多人发现尽管在持续使用，但他们的请求计数却滞后了。
   - 由于报告了类似的经历，用户推测这是一个孤立问题还是更广泛的后端问题。
- **对 GitHub 新功能的反馈**：围绕 **GitHub** 最近的更新展开了讨论，包括引入模型以赋能开发者构建 AI 工具。
   - 社区成员对 AI 工程的影响以及转向免费可用模型的潜力表示了兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://twomgg.onrender.com/">TwomGG</a>: 未找到描述</li><li><a href="https://x.com/cursor_ai/status/1874082036253942161?s=46">来自 Cursor (@cursor_ai) 的推文</a>: 实现这种速度和准确性需要在训练和推理方面进行巧妙的工作。我们希望很快发布更多改进。</li><li><a href="https://www.youtube.com/watch?v=NCaRixtXNIo"> - YouTube</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=xvBDzc6QafQ"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/vercel/ai-chatbot">GitHub - vercel/ai-chatbot: A full-featured, hackable Next.js AI chatbot built by Vercel</a>: 由 Vercel 构建的功能齐全、可黑客定制的 Next.js AI chatbot - vercel/ai-chatbot</li><li><a href="https://github.com/RezixDev/modals-next-test">GitHub - RezixDev/modals-next-test: Github Modals Test with Next.js and TypeScript</a>: 使用 Next.js 和 TypeScript 的 Github Modals 测试。通过在 GitHub 上创建账号为 RezixDev/modals-next-test 开发做出贡献。</li><li><a href="https://github.blog/news-insights/product-news/introducing-github-models/">Introducing GitHub Models: A new generation of AI engineers building on GitHub</a>: 我们正通过 GitHub Models 推动 AI 工程师的崛起——将行业领先的大型和小型语言模型的力量直接带给 GitHub 上超过 1 亿的用户。</li><li><a href="https://github.com/market">Market</a>: GitHub 是 Market 构建软件的地方。</li><li><a href="https://github.com/marketplace/models/catalog">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1323385633460453448)** (71 条消息🔥🔥): 

> `OpenRouter 模型新增, DeepSeek v3 性能, Gemini 2.0 限制, Sonnet 对比, 自我审查聊天模型` 


- **向 OpenRouter 添加模型**：一位用户询问了将其模型添加到 OpenRouter 的可行性，推测这可能仅限于拥有大量资金的公司。
   - 其他人建议启动个人 provider 来托管模型，强调无论初始障碍如何，都值得一试。
- **DeepSeek v3 表现优于其他模型**：多位用户称赞了 **DeepSeek v3** 的表现，特别提到了与 **Claude** 等模型相比，其在额度使用（credit usage）方面的长期稳定性。
   - 讨论强调了它相对于更昂贵模型的吸引力，一些人声称尽管存在局限性，但它在某些任务中非常有效。
- **Gemini 2.0 的局限性**：一位用户指出使用 **Gemini 2.0 Flash** 面临的挑战，特别是在 NSFW 图像字幕生成方面，使其在 OpenRouter 上显得无法使用。
   - 对其性能和 context 限制表示担忧，尤其是在处理复杂图像时。
- **Sonnet 与 DeepSeek 的对比**：用户讨论了 **Sonnet** 和 **DeepSeek** 在指令遵循（instruction-following）和复杂查询方面的差异，一些参与者更倾向于 Sonnet 的能力。
   - DeepSeek 的批评者指出，尽管价格更具优势，但在处理复杂的编程任务时，它还无法与对手抗衡。
- **理解自我审查模型**：一位用户对模型中自我审查（self-moderation）的概念提出疑问，引发了关于违反服务条款时拒绝消息如何运作的讨论。
   - 澄清强调，聊天模型的审查版和非审查版都受其各自 provider 条款的约束。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.anthropic.com/research/building-effective-agents">Building effective agents</a>: 一篇为开发者提供的关于构建高效 AI Agent 的建议和工作流的文章</li><li><a href="https://www.notdiamond.ai">Not Diamond</a>: Not Diamond 是全球最强大的 AI 模型路由。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1323380477377249330)** (22 条消息🔥): 

> `模型评估策略, 办公室配置升级, 眼神交流摄像头技术, Reinforcement Learning 动态, AI 自我纠错机制` 


- **模型评估困惑：为什么是 o1/o3？**：*为什么类 o1/o3 模型有效仍然是一个谜，* 成员们讨论了语言模型中**内在自我评估（intrinsic self-evaluation）**的挑战，强调这些模型可能并不真正知道自己不知道什么。
   - 一位成员表示打算进一步探索 **QwQ** 中的失败案例，怀疑**采样技术（sampling techniques）**可能解释了生成过程中自我评估的表面有效性。
- **令人兴奋的办公室升级！**：关于工作场所改进的讨论非常热烈，特别是 AI2 办公室中**新的录音设置**，承诺提供令人印象深刻的背景视野。
   - 成员们指出了**升级办公空间**的好处，分享了对创意氛围和促进日常高效运作的热情。
- **利用技术掌握眼神交流**：视频通话中保持眼神交流的创新解决方案浮出水面，一位成员提到使用 **Nvidia 串流软件**来增强他们的镜头表现。
   - 另一位成员提到了一种确保一致眼神交流的设置，但这在 Zoom 会议中让同事感到有些不安。
- **Reinforcement Learning 与自我纠错**：关于自我纠错在 Reinforcement Learning (RL) 模型中重要性的辩论，一些见解表明它主要是一个特性，而非性能的关键。
   - 成员们讨论了由于价值函数（value functions）等学习方法，**RL 结果具有路径依赖性（path-dependent）**，表明学习策略中存在复杂的相互作用。
- **语言模型中的自我纠错**：讨论预示了对语言模型中自我纠错重要性的怀疑，指出即使具备该功能，也可能不会显著影响结果。
   - 这一观点有助于澄清某些特征（如循环 token）可能只是固有模型设计的一部分，而不是学习效能的指标。



**提及的链接**：<a href="https://x.com/aidan_mclau/status/1873122732680134960">Aidan McLau (@aidan_mclau) 的推文</a>：你应该基本上假定让模型思考更长时间等同于构建一个更大的模型。遵循数学逻辑非常有趣，并揭示了行业进展中一些巧妙的事物。

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1323386634380771348)** (13 messages🔥): 

> `Gary Marcus 的预测、Nvidia 收购 Run:ai、GPT-4 模型进展、AI 幻觉问题、企业 AI 支出趋势` 


- **Gary Marcus 的预测引发辩论**：成员们讨论了 *Gary Marcus 的预测*，其中一人表示，*无论证据如何，他都不更新自己的先验 (priors)*，突显了他的预测所具有的争议性。
   - 另一种观点指出，尽管他有 *不错的想法*，但他的表达风格是 *反科学的*，且不利于公共话语。
- **Nvidia 收购 Run:ai 用于 AI GPU 编排**：Nvidia 已完成对 [Run:ai](https://www.run.ai/) 的收购，此举旨在增强其在 AI GPU 云编排 (orchestration) 方面的能力，据报道收购金额为 **7 亿美元**。
   - *Run:ai 的软件*（用于为 AI 调度 Nvidia GPU 资源）将变为 **开源 (open-source)**，尽管这一决定背后的原因尚未明确。
- **GPT-4 级别模型的持续进展**：对于 **GPT-4 级别模型** 的现状，人们情绪复杂，并对缺乏像 **GPT-5** 这样重大进展表示担忧。
   - 一位成员评论了 *企业支出 (enterprise spending)*，指出尽管该行业普遍认为利润微薄，但支出依然保持高位。
- **对 AI 幻觉的担忧**：讨论强调了 AI 模型中持久存在的 **幻觉 (hallucinations)** 问题，成员们一致认为目前缺乏解决这些问题的稳健方案。
   - 尽管取得了持续进展，但在幻觉和可靠性方面的 *实质性进展* 仍然是社区对话中强调的重点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://vxtwitter.com/GaryMarcus/status/1766871625075409381">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://x.com/BLUECOW009/status/1766966415800570243">来自 @bluecow 🐮(schizo) (@BLUECOW009) 的推文</a>: @GaryMarcus Claude 已经击败了 gpt4>dyor</li><li><a href="https://x.com/GaryMarcus/status/1873856666334015499">来自 Gary Marcus (@GaryMarcus) 的推文</a>: @jessi_cata 7/7 除非你算上 o3（已宣布未发布），并且他们在除了增强过的半封闭领域之外的幻觉和可靠性方面展示了真正的进展。</li><li><a href="https://venturebeat.com/ai/nvidia-acquires-software-maker-runai-to-orchestrate-gpu-clouds-for-ai/">Nvidia 将开源 Run:ai，这是其以 7 亿美元收购的用于帮助公司管理 AI GPU 的软件</a>: Nvidia 已完成对 Run:ai 的收购，这是一家让客户更容易编排 AI GPU 云的软件公司。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1323711917487030345)** (21 条消息🔥): 

> `2024 Interconnects 年度回顾，Meta 的 AI 策略，社交媒体上的算法与参与度，Open Source 模型，稳扎稳打的发展` 


- **2024 Interconnects 年度回顾亮点**：Nathan Lambert 回顾了他两年来每周撰写的 AI 文章，并在其 [2024 Interconnects Year in Review](https://www.interconnects.ai/p/2024-interconnects-year-in-review) 中总结了 **RLHF** 和 **open-source** 等核心观点。他指出 **AI** 继续主导着科技领域的对话，尤其是在预期 2024 年将发生重大事件的情况下。
   - 他指出即将推出的 OpenAI **o1 model** 可能是 AI 训练范式的潜在关键转折点。
- **Meta 对 AI 的日益关注**：Lambert 评论说 **AI** 正变得与 **Meta** 的业务战略更加密不可分，但表示它可能无法直接作为公司的竞争护城河。他建议 **Meta** 正在探索在不完全依赖 AI 获得优势的情况下整合 AI 的方法。
   - 这一观点反映了对大型科技公司如何根据不断发展的 AI 技术调整其基础战略的审慎看法。
- **社交媒体参与策略**：围绕低思考量推文的有效性展开了热烈讨论，正如 Xeophon 所指出的，他提到他们的算法技巧生成了诸如“**we are so back**”之类的短语。共识是，**欠缺考虑的帖子**似乎在社交平台上能获得更大的传播范围。
   - Lambert 幽默地承认了随口评论是如何被误解为完整的世界观的，特别是在涉及 **open source** 领域中 **US vs China** 等复杂话题时。
- **对 AI 模型大小的思考**：讨论包括对 AI 模型规模增长的担忧，有引用暗示模型可能会变得庞大到超出当前的能力范围。Xeophon 强调了对模型未来将继续剧烈增长的预期，这将使它们的部署变得复杂。
   - 参与者们以轻松的方式调侃了模型扩张的荒谬性，并对不断增大的模型所带来的挑战开起了玩笑。
- **蜗牛的回归**：Nathan Lambert 分享了一个关于“蜗牛回归”（Return of the Snail）的轻松评论，暗示了群组内一个持续且有趣的梗。这反映了在讨论严肃话题时，群组内部潜在的社区感和幽默感。
   - 这种在严肃 AI 讨论中穿插幽默的风格，展示了该群体充满活力且多元化的交流方式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/TheXeophon/status/1874155124278616150,">Xeophon (@TheXeophon) 的推文</a>: It might be so over. 引用 Nathan Lambert (@natolambert) 的 2024 Interconnects 年度回顾：两年来每周撰写关于 AI 的文章。如果你想了解我的写作进度，这是一个很好的起点...</li><li><a href="https://www.interconnects.ai/p/2024-interconnects-year-in-review">2024 Interconnects 年度回顾</a>: 两年来每周撰写关于 AI 的文章。</li><li><a href="https://x.com/teortaxesTex/status/1874158126846665195">Teortaxes▶️ (@teortaxesTex) 的推文</a>: 如果不是 Meta，那么我知道谁会分享最大的模型。模型会变得非常大，你甚至会厌倦它们的膨胀。你会说，“求你了，求你了。它太...”
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1323387133771382847)** (9 条消息🔥): 

> `Google AI 敏感性, Podcast 生成问题, Google Maps 的使用, 西班牙语对话中的 LLMs, Notebook LMS Plus 账户` 


- **Google AI 对敏感话题的严格限制**：一位用户指出，与其他公司相比，Google 的 AI 在处理**敏感话题**方面相当严格。
   - 这与其他成员分享的个人测试经验一致，特别是在使用 **Gemini** 模型时。
- **Podcast 生成器卡在“正在生成对话”**：一位成员报告了 **Notebook LM** 中 Podcast 生成器的问题，称其卡在“正在生成对话”阶段。
   - 这引发了关于故障排除以及对平台内 Podcast 创建功能预期的讨论。
- **在没有 Google Maps 的情况下提供巴士路线**：一位成员分享了他们在没有访问 **Google Maps** 的情况下，向 Podcast 提供**巴士路线**的经验。
   - 另一位用户建议从 Google 获取路线以改进 Podcast 功能。
- **关于西班牙出租车 LLM 的幽默讨论**：一位用户分享了一个关于**西班牙出租车司机**及其使用 **LLMs** 经历的幽默遭遇，包括照片和音频剪辑。
   - 这引发了轻松的评论，以及对将现实生活场景整合到语言模型中的思考。
- **咨询 Notebook LMS Plus 状态**：一位 Google Workspace Business 账户用户咨询了他们的 **Notebook LMS Plus** 状态，并指出他们已经集成了 **Gemini Business**。
   - 他们不清楚是否需要采取额外步骤来确认自己处于 **Plus** 层级。


  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1323385064502984825)** (34 条消息🔥): 

> `Podcast 音频概览, NotebookLM Plus 功能, YouTube 视频上传, 语音模型性能, 用户界面反馈` 


- **Podcast 音频概览令人印象深刻**：一位用户分享了使用 NotebookLM 创建引人入胜的 Podcast 内容的兴奋之情，成功地将现有音频源与新材料整合在一起。
   - 他们指出，该工具允许在片段之间进行平滑过渡，创造了无缝的收听体验。
- **NotebookLM Plus 提升功能**：NotebookLM Plus 为企业和教育工作者提供增强功能，允许上传 PDF 和 YouTube URL 等各种格式。
   - 用户可以创建摘要、时间线和音频概览，与免费版相比，每个笔记本提供的资源多出 5 倍。
- **目前尚不支持批量上传 YouTube 视频**：一位用户询问了批量上传 YouTube 视频的事宜，但被告知目前尚无此功能。
   - 互动显示，用户仍需逐个输入视频链接。
- **语音模型需要改进**：反馈指出语音模型的性能存在问题，特别是在多语言设置中，语调切换不一致。
   - 用户希望在 2025 年能有更好的语音模型性能和更多的语言支持。
- **注意到用户界面的局促感**：一些用户评论说 NotebookLM 的新 UI 让他们感到局促，表明需要更宽敞的设计。
   - 社区正在积极讨论用户体验，以及技术改进和功能请求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">升级到 NotebookLM Plus - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://cloud.google.com/text-to-speech/docs/voice-types">无标题</a>: 未找到描述</li><li><a href="https://youtu.be/3OFeH9YFxjM?si=98ZYlw4Eevm8e32q">UNREAL MYSTERIES 6: 圣诞特辑 - 后末日音乐剧</a>: 每部好剧都有圣诞特辑，而每部优秀的圣诞特辑都是音乐剧……David 和 Hannah 对抗僵尸驯鹿、澳大利亚外星人以及……
</li>
</ul>

</div>

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1323391290913718383)** (4 条消息): 

> `CUDA programming projects, Overlap data transfer in CUDA, Fluids simulation optimization` 


- **寻找用于求职展示的 CUDA 项目**：一位成员完成了 **CUDA programming** 课程，正在寻找有关在求职期间展示技能的项目建议。
   - 他们正在寻找能够让潜在雇主眼前一亮的**有趣**且**具有挑战性**的项目。
- **需要关于数据传输重叠的帮助**：另一位用户正在询问 CUDA 中的 **overlap data transfer** 技术，希望获得高效方法的指导。
   - 他们参考了一篇讨论数据传输与计算重叠的 [CUDA 博客文章](https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/)。
- **请求代码优化建议**：一位成员分享了他们的 **fluidsCUDA** GitHub 仓库，并征求改进其 **fluid_solver.cu** 文件执行时间的建议。
   - 他们指出，**CUDA** 执行需要 **15 秒**，而其 **OPENMP** 实现仅需 **1 秒**，目前正在寻找共享内存使用方面的潜在优化方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pixelfung/fluidsCUDA">GitHub - pixelfung/fluidsCUDA</a>：通过在 GitHub 上创建账户来为 pixelfung/fluidsCUDA 的开发做出贡献。</li><li><a href="https://developer.nvidia.com/blog/how-overlap-data-transfers-cuda-cc/">如何在 CUDA C/C++ 中重叠数据传输 | NVIDIA 技术博客</a>：在上一篇 CUDA C/C++ 文章中，我们讨论了如何在主机和设备之间高效传输数据。在本篇中，我们将讨论如何将数据传输与主机上的计算重叠……
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1323442155800236085)** (1 条消息): 

> `Image Analysis Feedback` 


- **关于图像分析的令人惊讶的反馈**：一位成员表示，对所附图像的解读**读起来出奇地顺畅**，并引用了一个[附件](https://cdn.discordapp.com/attachments/1189607595451895918/1323442155913740370/image.png?ex=67752fce&is=6773de4e&hm=789537413fe228ead79b74fca093e47988e43f8e08c76d860703fd1a76444521&)。
   - 他们指出**发现这一点很奇怪**，表明对分析结果有出乎意料的反应。
- **图像附件背景**：消息包含一个[附件图像](https://cdn.discordapp.com/attachments/1189607595451895918/1323442155913740370/image.png?ex=67752fce&is=6773de4e&hm=789537413fe228ead79b74fca093e47988e43f8e08c76d860703fd1a76444521&)，引发了关于其解读的讨论。
   - 该附件成为了该成员对其可读性以及发现的奇特之处发表评论的焦点。


  

---


### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1323674668334579723)** (1 条消息): 

> `Genesis Simulator, Benchmark Corrections` 


- **澄清 Genesis 模拟器的速度误解**：在 **Genesis simulator** 发布后，一篇博客文章透露其速度比现有的 GPU 模拟器慢多达 **10 倍**，这与之前声称的快多达 **430,000 倍**相反。
   - 次日发布的博客文章纠正了关于基准测试的误解，特别指出之前的测量主要基于静态环境，正如 [@Stone_Tao 的推文](https://x.com/stone_tao/status/1870243004730225009?s=46&t=LBFTca4dqDdDCjhzaM56tA) 中所强调的那样。
- **Genesis 基准测试的详细检查**：此处链接的[博客文章](https://stoneztao.substack.com/p/the-new-hyped-genesis-simulator-is)对与 **Genesis simulator** 相关的开源基准测试提供了全面的修正。
   - 其目的是消除自该模拟器备受期待的发布以来，关于其性能能力的误解。



**提到的链接**：<a href="https://x.com/stone_tao/status/1870243004730225009?s=46&t=LBFTca4dqDdDCjhzaM56tA)">Stone Tao (@Stone_Tao) 的推文</a>：昨天备受关注的 Genesis 模拟器发布了。但它比现有的 GPU 模拟器慢多达 10 倍，而不是快 10-80 倍或比实时快 430,000 倍，因为他们测试的主要是静态环境……

  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1323417048692559872)** (1 条消息): 

> `Cracked Research Engineer Job, CUDA Engineer Roles, Remote LLM Infra Positions, Triton Kernel Development` 


- **发现 Cracked Research Engineer 职位**：一名成员偶然发现了一个 [cracked research engineer job](https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243)，对于技术圈的人来说看起来很有前景。
   - 对于社区内对创新项目感兴趣的人来说，这可能是一个绝佳的机会。
- **CUDA Engineer 职位咨询**：成员们在探索新机会时，正在寻找位于**旧金山**的 **CUDA engineer 职位**。
   - *示例问题*表明成员们渴望在尖端领域寻找专业职位。
- **寻求远程 LLM Infra Engineer 职位**：社区成员对**远程 LLM infra engineer 职位**的兴趣日益浓厚。
   - 这反映了 AI 工程领域向灵活工作选项的转变，突显了对此类职位的需求。
- **讨论 Triton Kernel 开发职位**：围绕涉及 **Triton kernel 开发**职位的讨论表明，编程中对性能和优化的关注。
   - 鼓励成员利用该领域的知识来获得更好的就业前景。



**提到的链接**：<a href="https://crackedengineers.com/job/p-1-ai-7f41fa30-6cfa-4e9a-8943-2324dc21d243">Cracked Engineers</a>：为您的初创公司招聘顶尖的 AI 和软件工程师。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1323621294159298631)** (4 条消息): 

> `SSH into Vast AI GPU, Using CUDA on Ubuntu, PyTorch image for GPU, SSH key generation` 


- **通过 SSH 访问 Vast AI GPU**：要通过 SSH 进入从 **Vast AI** 租用的 **Ubuntu** 机器，请生成 **SSH key** 并在实例设置期间输入，系统将提供用于连接的 **IP address** 和 **port**。
   - *适用标准 SSH 流程*，并且有许多文章详细解释了这一过程。
- **结合 CUDA 使用 PyTorch**：建议为 **Vast AI 实例**使用 **PyTorch (cuDNN devel)** 模板/镜像，以确保预装了 **nvcc** 和 **gcc** 等编译器。
   - 这种设置对于在租用的 GPU 上有效运行 **CUDA 程序**至关重要。
- **使用 Windows 搭配 WSL2 进行 GPU 访问**：**Windows 搭配 WSL2** 被提及为访问和利用 GPU 的有效环境。
   - 这种设置因其与各种应用程序和 CUDA 编程的兼容性而受到赞赏。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 条消息): 

iron_bound: https://www.youtube.com/watch?v=VpAZPPCLCUI
  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1323600782943064084)** (19 条消息🔥): 

> `Triton Performance vs Torch, Benchmarking Add Function, Triton Environment Variable, GPU Configuration, Code Comparison` 


- **Triton 性能数据受到关注**：一名成员报告了他们的 Triton vs Torch 性能结果，在 `vector-add` 基准测试中显示出显著差异，特别是在较小尺寸下。
   - 他们质疑这些差异是由于他们的实现方式导致的，还是 Triton 的性能确实优于 Torch。
- **Triton 加法函数的代码细节**：成员们讨论了 Triton 的 `add` 函数实现，强调其依赖 kernel 在 GPU tensor 上执行加法。
   - 基准测试结果显示，在许多情况下，Triton 的表现比 Torch 慢。
- **环境变量导致的问题**：一名成员发现将环境变量 `TRITON_INTERPRET` 设置为 `1` 会导致基准测试中的性能差异。
   - 注释掉该变量后解决了性能问题，得到了与同事一致的预期结果。
- **本地与 Colab 实验对比**：成员们表示有兴趣在不同平台上测试他们的代码，并指出性能差异可能源于特定的 GPU 配置。
   - 一名成员计划在 Colab 上运行相同的代码，而其他人则记录了他们本地机器的性能。
- **感谢同伴支持**：几位成员感谢他人在调试 Triton 代码和分享性能问题见解方面提供的帮助。
   - 尽管正值跨年夜，成员们仍愿意共同解决问题，现场氛围积极。


  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1323412783295107124)** (4 messages): 

> `对 Triton 的贡献、整数量化挑战、Triton 的优化主张` 


- **呼吁为 Triton 做出贡献**：目前有一项公开邀请，欢迎有兴趣的人士为 Triton 做出贡献，并承诺欢迎任何新增内容。
   - *如果你或其他人也有兴趣贡献，我们非常乐意将其纳入*。
- **Triton 与整数算术的挑战**：一名博士生对 Triton 对 **integer arithmetic**（整数算术）的支持表示担忧，指出目前该领域支持较差。
   - 该学生询问了是否有任何正在进行的会议或指南，表明希望改进 Triton 的功能。
- **Triton 的决策性能**：一位成员强调，对于 Triton 在其决策过程中的 **optimal**（最优性）程度缺乏直观认识。
   - 他们认为 *尝试手动超越它* 可能会有好处，这表明了对潜在性能提升的好奇。
- **内核执行管理方面的担忧**：讨论涉及内核可能需要的 **fine-grained asynchronous execution management**（细粒度异步执行管理），这可能会阻碍 Triton 的性能。
   - 有人提到，如果这些控制权没有在 Triton 中充分暴露，将很难实现 **peak performance**（峰值性能）。


  

---


### **GPU MODE ▷ #[edge](https://discord.com/channels/1189498204333543425/1303441437592911912/1323610659551973416)** (7 messages): 

> `树莓派 5 测试、Bielik 模型性能、OpenBLAS 影响、PP 和 TG 测试名称、树莓派 5 中的 GPU` 


- **使用 llama.cpp 测试树莓派 5**：**树莓派 5** 专门使用 **llama.cpp** 进行了测试，突出了该平台的能力。
   - 然而，用户在为 **VideoCore VII** 上的 Vulkan 后端编译 llama.cpp 时遇到了挑战。
- **Bielik 模型性能测试进行中**：波兰迷你 LLM 模型 **(Bielik)** 的性能测试目前正在进行中，重点关注 **Bielik-1.5B** 变体。
   - 在理想条件下，树莓派 5 上 **Bielik** 的预测最大 tokens 每秒约为 **7 tok/sec**。
- **OpenBLAS 减慢了输入解码速度**：据观察，**OpenBLAS** 减慢了输入解码速度，且没有提高输出速度。
   - 性能测试显示，在现有硬件上，Q8 量化可达到约 **6 tok/sec** 的速率。
- **理解 PP 和 TG 测试名称**：'PP' 代表 **prompt processing**（提示词处理），而 'TG' 指的是 **text-generation**（文本生成），这明确了测试术语。
   - 这一澄清有助于理解讨论中关于模型输出的各项指标。
- **树莓派 5 的 GPU 能力受到质疑**：有人提出了关于哪些指标利用了树莓派 5 中的 GPU 的疑问，该设备主要搭载 **Broadcom BCM2712 CPU**。
   - 现有的 **VideoCore VII** GPU 未能得到充分利用，因为用户无法为 Vulkan 后端编译 llama.cpp。


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1323621501630677003)** (1 messages): 

> `SSH 进入 Ubuntu、GPU 租赁流程、创建实例` 


- **SSH 进入 Ubuntu 机器的步骤**：一位用户在租赁 **GPU** 并创建实例后，请求关于如何 SSH 进入其 **Ubuntu** 机器的帮助。
   - 这突显了对安全访问云实例需要更清晰指导的需求。
- **GPU 租赁说明**：该用户目前正在寻求专门针对租赁 **GPU** 后流程的帮助。
   - 这表明用户在连接到其租赁资源时存在常见疑问。
- **在 Ubuntu 上创建实例**：提到了在租赁 GPU 后，需要明确在 **Ubuntu** 上创建 **instance**（实例）所涉及的步骤。
   - 这指向了文档中可能存在的空白，或用户对设置过程知识的匮乏。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1323385459153571862)** (33 条消息🔥): 

> `Pro reasoning mode, Deepseek regulations, Joke-telling abilities of AI, New Year's celebrations, AI predictions for 2025` 


- **Pro reasoning mode 针对复杂问题激活**：成员们讨论了 Perplexity 的 [reasoning mode](https://discord.com/channels/1047197230748151888/1054944216876331118/1323055993311203498)，该模式会自动处理复杂查询以增强思考过程。
   - 一位用户质疑该功能是否标志着与以往方法的转变，并认为它类似于 Perplexity 之前采用的思考过程。
- **Deepseek 在不同的监管下运行**：一位成员强调 **Deepseek** 模型在**中国监管**下运行，与受**美国法律**影响的其他模型（如 ChatGPT）相比，可能拥有更多自由度。
   - 这引发了关于监管环境如何影响各种 LLM 能力的问题。
- **Perplexity 有限的幽默能力**：一位用户注意到 **Perplexity** 并不擅长讲笑话，他们对此表示欣赏，因为这让 AI 感觉更像人类。
   - 他们讲述了一次幽默的交流，其编程助手 AI 承认自己讲的笑话很重复，并决定转而专注于编程任务。
- **新年问候与传统**：成员们交换了 **New Year** 问候，一位用户分享了节日图片，另一位用户鼓励大家加入庆祝活动。
   - 氛围十分愉快，在新年到来之际庆祝社区互动。
- **2025 年 AI 预测**：一位成员在分析 [AI predictions for 2025](https://www.linkedin.com/pulse/ai-predictions-2025-scott-weiner-3xm3e) 时讨论了人类预测的挑战，强调了认知偏差和情感因素。
   - 这反映了商业领袖和技术专家在考虑未来趋势时应采取的务实方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.perplexity.ai/docs/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/apostraphi/status/1871409446292987909?s=61">Phi Hoang (@apostraphi) 的推文</a>：还有一个装饰品要挂起来
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1323432616430534708)** (4 条消息): 

> `YouTube Random Video Button, Content Optimization Techniques, OpenAI Public Benefit Corporation, Tibet Mega Dam Approval, Encyclopedia Britannica Updates` 


- **探索 YouTube 的新随机视频按钮**：讨论重点介绍了 [YouTube Random Video Button](https://www.perplexity.ai/page/youtube-s-random-video-button-00KFpoLLThS8boDmTNk3wg)，旨在通过为观众随机选择视频来增加用户参与度。
   - 该功能反映了 YouTube 保持内容消费多样化和趣味性的策略。
- **优化内容以获得更好的传播**：成员们分享了关于[优化内容](https://www.perplexity.ai/search/how-can-i-optimize-content-for-K.VTSaD0R0yS2gxv7SXBuA)的见解，以提高在各平台上的可见度和参与度。
   - 有效策略包括利用关键词和了解受众偏好。
- **OpenAI 转向 Public Benefit Corporation**：社区讨论了 OpenAI 转型为 Public Benefit Corporation 的提议，旨在平衡盈利与社会责任。
   - 此举被视为对当前关于 AI 社会影响辩论的回应。
- **中国批准西藏巨型大坝**：成员们深入探讨了中国批准在西藏建设新巨型大坝的影响，预计该项目将产生重大的环境后果。
   - 该项目引发了关于可持续发展和区域影响的讨论。
- **大英百科全书（Encyclopedia Britannica）最新更新**：分享了来自大英百科全书的更新，详细介绍了旨在提高事实准确性的新条目和重大修订。
   - 这些变化反映了为学习者和研究人员提供可靠信息的持续承诺。



**提到的链接**：<a href="https://www.youtube.com/embed/jCmQSLgYP4g">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1323386490025410641)** (3 messages): 

> `Sonar models usage, Perplexity AI API features, Discord bot with premium features` 


- **Sonar 模型被误用**：一名成员澄清说，**Sonar models** 并不适用于所展示的用例，因为它们的设计初衷是利用当前的 Web 资源提供答案并引用它们。
   - 这一评论强调了按照设计意图使用模型的重要性。
- **关于 Perplexity AI API 的咨询**：一位用户询问了 **Perplexity AI API** 的可能应用场景，对其用途提出了疑问。
   - 这个问题反映了用户对该 API 在不同项目中能力的广泛好奇。
- **创建具有高级特性的 Discord 机器人**：一名成员询问在购买 **Perplexity AI** 后，是否可以使用其**高级特性 (premium features)** 创建一个 **Discord bot**。
   - 这表明用户有兴趣利用该平台的优势来构建交互式应用程序。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1323383589907202130)** (30 messages🔥): 

> `BoxPointer renamed to OwnedPointer, Pointer issues in Mojo, Self-referential structures, Mojo update compatibility concerns` 


- **BoxPointer 重命名为 OwnedPointer**：`BoxPointer[Self]` 已重命名为 `OwnedPointer[Self]`，这引起了用户的一些困惑。
   - 一位用户对这一澄清表示感谢，但在 nightly **stdlib** 中找不到 `BoxPointer`。
- **Mojo 中的指针限制和 'unsafe' 用法**：讨论围绕 Mojo 中的 `UnsafePointer` 展开，它比 C/C++ 具有更多的不变性 (invariants)，这可能为用户带来潜在陷阱。
   - 有人指出，使用指针可能会导致不稳健 (unsound) 的情况，尤其是在尝试初始化递归数据结构中的父指针时。
- **创建自引用结构**：用户探索了在 Mojo 中构建自引用节点结构的各种方法，并建议使用 `ArcPointer`。
   - 然而，在尝试可选引用 (optional references) 时遇到了问题，导致一位用户考虑重构其实现。
- **Mojo 即将到来的破坏性变更**：据确认，Mojo 在达到 1.0 版本之前，将继续大约每 6 个月打破一次兼容性。
   - 一位用户对因这些更新而需要重写代码表示担忧，表示倾向于探索 Rust 等其他语言。
- **Mojo 中报告的 Bug**：一位用户报告了一个关于使用调试器运行脚本时出现 **segfaults** 的 Bug，聊天中的其他人也确认了这一点。
   - 由于假期期间，预计对该报告的回复会有所延迟。



**提到的链接**：<a href="https://github.com/modularml/mojo/issues/3917">[BUG] --debug-level full crashes when importing · Issue #3917 · modularml/mojo</a>：Bug 描述：使用调试器运行 Mojo 脚本会发生 **segfaults**，而运行常规 Mojo 脚本则可以运行完成（尽管我注意到常规脚本中也存在奇怪的行为...）

  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1323497914139541504)** (2 messages): 

> `Mojo APIs for max, API modernization, Type system enhancements` 


- **Mojo API 需要现代化改造**：讨论强调，用于 max 的 **Mojo APIs** 是在早期构建的，利用了 **value semantics**（值语义）和 **move semantics**（移动语义）等特性，但缺乏稳健的安全引用。
   - 参与者认为，有必要进行一次全面的 **API review** 来对其进行现代化改造，并充分利用先进的 **Mojo features**。
- **API footguns 带来挑战**：一位成员指出，当前的 API 虽然可用，但对外部用户来说包含许多 **footguns**（易误用特性），需要解决。
   - 他们建议，这些问题可能通过**对类型系统的充分应用**来得到解决。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1323380801941016660)** (31 messages🔥): 

> `Discord 中的诈骗者、SD3 的问题、Stability.ai API 中的 Faceswap 功能、Checkpoint 和 Lora 模型、对更好验证流程的需求` 


- **呼吁在 Discord 中控制诈骗**：成员们对渗透进 Discord 的诈骗者表示沮丧，建议实施验证码或手机验证来管理这一问题。
   - 一位成员评论道：*这里就像在玩机器人打地鼠*，而另一位成员指出，手机验证可能会增加攻击者的成本。
- **关于 SD3 安全措施的辩论**：针对应用于 **SD3** 的 **trust and safety**（信任与安全）措施的看法引发了讨论，一些人呼吁在 Discord 中强制执行这些控制。
   - 一位成员认为，当前的安全言论实际上分散了人们对诈骗和数据收割等**真实威胁**的注意力。
- **关于 Stability.ai API 功能的疑问**：一位用户询问 **Stability.ai API** 是否支持 faceswap 功能，因为他们在文档中找不到相关信息。
   - 回复指出，虽然无法保证时间一致性（temporal consistency），但实现某些图像操作是可能的，但受到限制。
- **理解 Checkpoint 和 Lora 模型**：成员们讨论了 **Lora** 模型与微调后的 **checkpoints** 之间的区别，澄清了 Lora 更新特定权重，而微调后的 checkpoints 体积更大但能达到类似效果。
   - 共识是，这两种方法都比保留整个模型更具成本效益，这使得它们对于希望改进模型的用户非常有吸引力。
- **新成员加入及资源分享**：新用户表示欢迎并寻求社区中关于 prompt 和模型创建的最佳实践建议。
   - 一位新成员对 checkpoint 创建的细微差别表示困惑，并请求经验丰富的成员提供指导。



**提到的链接**：<a href="https://frrouting.org/">FRRouting</a>：未找到描述

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1323398077976936550)** (14 messages🔥): 

> `Lipschitz-1 RMSNorm 替代方案、Pile 数据集的预估 Token 数、Residual Flows 实现、使用 Lipschitz 常数进行训练、Neural SDFs 和 NeRFs 的应用` 


- **探索 Lipschitz-1 RMSNorm 替代方案**：分享了一个新的 **RMSNorm replacement** 实现，旨在通过 **tanh** 函数保持输入的 2-范数（2-norm）同时防止其增加。
   - 该实现在其可能用于 **GANs** 和具有严格 Lipschitz 常数的残差模型背景下进行了讨论。
- **关于 Pile 数据集大小的澄清**：探讨了 Pile 数据集中 GPT-2 tokens 的预估数量，并提供了指向[研究论文](https://pile.eleuther.ai/paper.pdf)的链接，以澄清用于测量的指标。
   - 经计算，大约 **260B GPT-2 tokens** 分布在约 **825.18 GiB** 的数据中，并讨论了一些数据被上采样以达到约 **400B tokens**。
- **Lipschitz 常数及其对训练的影响**：一位成员对新 RMSNorm 在普通模型上可能表现更差表示怀疑，但指出对于实现需要 Lipschitz 常数小于 1 的 residual flows 来说，它是必要的。
   - 社区讨论认为，维持 Lipschitz 边界可以促进模型中更快的 tracing，从而增强各种应用的性能。
- **在 SDFs 和 NeRFs 中的潜在用途**：一位成员建议实现 Lipschitz 边界可能对 **neural SDFs** 和 **NeRFs** 有益，因为它允许对这些网络进行更高效的 tracing。
   - Lipschitz 常数在这些领域的应用被认为有助于更有效的训练过程和结果。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1323701446734118975)** (1 messages): 

> `优化的 RAG 流水线、LlamaParse` 


- **为财务报告构建优化的 RAG 流水线**：Hanane Dupouy 分享了一篇关于使用 [LlamaParse auto-mode](https://t.co/WptFCIoyp6) 构建优化 **RAG pipeline** 的见解文章，该模式可根据成本效益智能地在基础模式和高级模式之间做出决策。
   - 这种方法旨在增强专门为财务报告定制的处理能力，确保效率和有效性。
- **LlamaIndex 增强功能讨论**：社区讨论了使用更新的 **LlamaParse** 功能的影响，特别是关注其针对高级处理模式的自动决策能力。
   - 成员们表达了将这些增强功能集成到工作流中的渴望，以便在数据处理方面获得更好的结果。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1323455306201694319)** (10 messages🔥): 

> `Anomaly Detection, Vector Store Embeddings, Chatbot Background Process, Finetuning Llama Model` 


- **异常检测实现挑战 (Anomaly Detection Implementation Challenges)**：一位成员分享了他们尝试使用 **Milvus** 和 **FAISS** 的混合方法来处理 Embeddings 和聚类，从而实现异常检测。
   - 其他成员建议直接使用底层 Client 以获得更好的性能，并提醒大多数 Vector Store 出于节省内存的考虑可能不会返回 Embeddings。
- **聊天机器人后台进程处理 (Chatbot Background Process Handling)**：一位成员讨论了在聊天机器人中运行长时间后台进程的困难，并尝试使用 **multiprocessing** 来处理延迟。
   - 另一位成员建议对于任何异步函数，应切换到 **asyncio.create_task** 而不是使用 **multiprocessing**。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1323459622685642812)** (10 messages🔥): 

> `ModernBERT finetunes, Return of Sesame Street models, AI progress and saturation charts, OpenAI's transition to for-profit, New agentic systems from Hugging Face` 


- **ModernBERT 微调模型涌现 (ModernBERT Finetunes Flood the Scene)**：围绕 modernbert-embed-base 的发布展开了热烈讨论，这是一个基于 **ModernBERT** 构建的新 Embedding 模型，在 Tokenizer 和推理速度上有显著提升。
   - 成员们非常喜欢 Twitter 上分享的关于 ModernBERT 更新的**可视化表示**。
- **AI 进展图表持续更新 (AI Progress Plots Keep Coming)**：一份广受好评的更新显示了 **arc AGI 图表**中持续的 AI 进展，再次证明 AI 发展并未放缓。
   - 成员们思考了这份基于 **@Dynabench** 论文的最新进展图表的意义。
- **OpenAI 的转型引发失望 (OpenAI's Shift Sparks Disappointment)**：对于 **OpenAI 向营利性结构的转型**出现了不满情绪，批评其使命发生了偏移。
   - 成员们对这一转变破坏了确保“AGI 造福全人类”的原始目标表示失望。
- **新智能体框架发布 (New Agentic Framework Unveiled)**：一位成员分享了对 Hugging Face 发布 **agentic systems** 的兴奋之情，该系统被宣布为构建强大 Agent 的“最简单库”。
   - 新系统拥有简洁的代码库和**自然代码编写能力**，展示了优于以往标准的性能。
- **革命性压缩工具发布 (Revolutionary Compression Tool Released)**：介绍了一个名为 `ts_zip` 的新工具，承诺使用 **Large Language Model** 对文本文件实现高压缩率。
   - 虽然该工具需要 GPU 才能保证效率，但讨论也涉及了其实验性质和局限性，包括它比传统压缩器慢。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/zach_nussbaum/status/1873813021786767699?s=46&t=tMWvmS3OL3Ssg0b9lKvp4Q">Zach Nussbaum (@zach_nussbaum) 的推文</a>: 🧵 很高兴宣布 modernbert-embed-base，这是一个基于新发布的 ModernBERT 构建的新 Embedding 模型！在公共 Nomic Embed 数据集上训练，modernbert-embed-base 具有 ~nomic-embed~ 级别的质量...</li><li><a href="https://x.com/janleike/status/1872909496777134127">Jan Leike (@janleike) 的推文</a>: 鉴于所有竞争对手都是营利性的，OpenAI 向营利性的转型似乎不可避免，但“确保 AGI 造福全人类”让位于一个更... 确实令人失望。</li><li><a href="https://x.com/aymericroucher/status/1874116324898598934?s=46">Aymeric (m-ric) (@AymericRoucher) 的推文</a>: 几个月来，我们一直致力于构建 @huggingface 的新登月项目：agentic systems。所以今天我们非常自豪地宣布发布 𝚜𝚖𝚘𝚕𝚊𝚐𝚎𝚗𝚝𝚜！这是我们最简单的库...</li><li><a href="https://x.com/janleike/status/1872909498966524305">Jan Leike (@janleike) 的推文</a>: 这不是我加入 OpenAI 时所追求的。非营利组织需要维护 OpenAI 的使命！</li><li><a href="https://x.com/douwekiela/status/1873755176940765300">Douwe Kiela (@douwekiela) 的推文</a>: AI 没有放缓！这是一份最新的进展图表，标志着这一年的结束，基于 @Dynabench 论文中的原始图表。这意味着什么？在 🧵👇 中有一些思考和分析...</li><li><a href="https://bellard.org/ts_zip/">ts_zip: 使用大语言模型进行文本压缩</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1323638467896283176)** (4 messages): 

> `HMM 中的 Tokenization，新年庆祝` 


- **HMM 的 Tokenization 保持不变**：一位成员指出，**Hidden Markov Models (HMM)** 的 **tokenization** 不会有所不同，这表明了方法的一致性。
   - 讨论暗示了社区在该技术层面上已达成共识。
- **愉快的新年祝福**：多位成员分享了他们对**新年**的兴奋和问候，传播了节日气氛。
   - 社区共同庆祝，反映了迎接新年时的积极氛围。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1323498484585988096)** (6 messages): 

> `Cohere 支付问题，转向 OpenAI，RBI 指南变更影响交易` 


- **Cohere 支付方式问题**：一位用户报告在将他们的 **SBI Visa Global Debit Card** 添加为 Cohere 账户的支付方式时遇到困难，收到错误提示：*'We were unable to save your payment method. Please try again.'*
   - 他们对遇到这种可能推迟计划发布的意外障碍表示沮丧。
- **用户转向 OpenAI**：由于支付问题，该用户决定转向 **OpenAI**，并表示他们无法承受发布进度的任何延迟。
   - 他们对在测试一年后不得不放弃 Cohere 感到遗憾，但认为在当前情况下这是必要的。
- **遵循 RBI 指南产生的费用**：另一位成员更新称，印度持卡人的问题源于新的 **RBI 指南变更**，除了原帖用户外，还影响了多位用户。
   - 他们向大家保证，团队正在努力解决这些问题，并会让所有印度客户了解最新进展。
- **建议临时变通方法**：同一位成员建议在解决印度卡问题期间，尝试使用不同银行的账户作为临时变通方法。
   - 他们鼓励用户尝试此方法并反馈效果，同时团队将继续修复底层问题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1323417083849080893)** (5 messages): 

> `机器码中的可逆变换，pcode 概念在 tinygrad 中的应用，开始为 tinygrad 做贡献，tinygrad 教程资源，tinygrad 用户入门` 


- **澄清可逆变换**：一位成员询问在**机器码**与 **uops** 之间的可逆变换中，是否存在中间汇编步骤的可能性，质疑这是否必须是一个从 uop 到二进制的直接过程。
   - *他们还想知道可逆是否意味着它可以等同于一个 uop 序列或最终的重写状态。*
- **Pcode 概念与 tinygrad 产生共鸣**：一位成员对 **sleigh 文档** 进行了积极反思，注意到来自 **pcode** 转换的概念可能与 tinygrad 中的 **uop** 相似。
   - *他们观察到 pcode 定义包含 dtype 和 meta info，并强调了它与汇编语言而非 uops 的相似性。*
- **tinygrad 初学者的第一步**：一位新人表示在开始学习 tinygrad 时遇到困难，发现标记为 'good first issue' 的问题与他们的基础知识脱节。
   - *他们寻求相关资源建议，以便在贡献代码之前建立理解。*
- **为新人提供教程资源链接**：另一位成员提供了一个富有见地的 GitHub 仓库链接 [tinygrad notes](https://github.com/mesozoic-egg/tinygrad-notes)，该仓库提供了 tinygrad 的入门教程。
   - *该仓库旨在帮助新贡献者浏览并理解 tinygrad。*



**提及的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/tree/main">GitHub - mesozoic-egg/tinygrad-notes: Tutorials on tinygrad</a>：tinygrad 教程。通过在 GitHub 上创建账户为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1323702308810391692)** (1 messages): 

> `tinygrad internals, tinygrad notes` 


- **tinygrad 内部原理的增强版介绍**：一位成员分享了 tinygrad 内部原理的[新介绍](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241231_intro.md)，旨在改进文档。
   - 该介绍是为理解 tinygrad 创建全面资源而进行的持续努力的一部分。
- **tinygrad 笔记的 GitHub 仓库**：分享的介绍托管在 GitHub 的 **tinygrad-notes** 仓库中，该仓库包含有关 tinygrad 的各种教程。
   - 鼓励成员为 [tinygrad-notes 仓库](https://github.com/mesozoic-egg/tinygrad-notes)贡献力量，以增强学习材料。



**提及的链接**：<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241231_intro.md">tinygrad-notes/20241231_intro.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建账户为 mesozoic-egg/tinygrad-notes 的开发做出贡献。

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1323620107133653056)** (2 messages): 

> `GH200 Access, D2H Memory Transfer Issue` 


- **寻求 GH200 访问权限以进行调查**：一位成员请求拥有 **GH200 访问权限**的人运行一个简单的 Python 复现脚本，作为他们对 **D2H 内存传输问题**调查的一部分。
   - 目的是确保该问题不是由他们那边的**特定配置**引起的。
- **需要进行配置检查**：重点在于确认 **D2H 内存传输问题**与他们设置中的特定配置无关。
   - 这表明人们持续关注潜在的配置不匹配可能导致该问题。


  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1323495969421262868)** (2 messages): 

> `DeepSeek Coder V2 Lite, GigaChat, Modernbert, Embedding backend for localdocs` 


- **DeepSeek Coder V2 Lite 运行良好**：一位成员报告使用 **DeepSeek Coder V2 Lite** 没有任何问题，表示对其性能感到满意。
   - 然而，他们尚未测试 **GigaChat**，因此对该模型的看法尚不明确。
- **关于 Modernbert 更新的咨询**：一位成员提到在 **Hugging Face** 上看到了 **Modernbert**，并询问了有关 **localdocs** 的 Embedding 后端潜在更新的情况。
   - 讨论表明社区对与 **Modernbert** 相关的 Embedding 系统改进感兴趣。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/)** (1 messages): 

sevastopaul2041: 嘿，报名 Advanced LLM Agents MOOC 的截止日期是什么时候？
  

---


---


---


---


---


---


---


---


---


{% else %}


> 完整的频道细分详情已针对邮件进行截断。
> 
> 如果你想查看完整详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})!
>
> 如果你喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}