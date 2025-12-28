---
companies:
- amazon
- anthropic
- google-deepmind
- sakana-ai-labs
date: '2024-12-04T03:06:39.205571Z'
description: '**亚马逊**在 AWS Re:Invent 大会上发布了 **Amazon Nova** 系列多模态基础模型。该系列目前已立即开放使用，无需排队，提供
  Micro、Lite、Pro、Canvas 和 Reel 等多种配置，而 Premier 版本和语音对语音（speech-to-speech）功能将于明年推出。


  这些模型的 **Token 生成速度快了 2 到 4 倍**，且价格比 **Anthropic Claude** 等竞争对手**便宜 25% 到 400%**，这使得
  Nova 成为 AI 工程领域的有力竞争者。其定价低于 **Google DeepMind Gemini Flash 8B** 等模型，部分 Nova 模型的上下文长度可扩展至
  **30 万个 Token**。


  然而，目前存在基准测试争议，因为一些评估显示 Nova 在 **LiveBench AI** 指标上的得分低于 **Llama-3 70B**。


  另外，**Sakana AI 实验室**推出了 **CycleQD**，该技术利用演化计算进行基于种群的模型合并，旨在开发特定领域的 LLM 智能体。'
id: cfb67b82-e4f4-4777-b24c-0e7fb156170c
models:
- amazon-nova
- claude-3
- llama-3-70b
- gemini-1.5-flash
- gpt-4o
original_slug: ainews-olympus-has-dropped-aka-amazon-nova
people:
- philschmid
- bindureddy
title: Olympus 正式发布（即 Amazon Nova Micro|Lite|Pro|Premier|Canvas|Reel）
topics:
- multimodality
- benchmarking
- model-merging
- model-performance
- model-architecture
- model-optimization
- population-based-learning
---

<!-- buttondown-editor-mode: plaintext -->**Amazon Bedrock 就够了吗？**

> 2024年12月2日至12月3日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**198** 个频道，**2914** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**340 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

*我们对昨天的重复邮件表示歉意。这是我们无法控制的平台 Bug，但我们会密切关注，因为我们绝对不想骚扰您或损害我们的送达率。幸运的是，AINews 的理念之一就是邮件的长度和数量几乎（但不完全）是免费的。*

正如过去一年中[广泛传闻（代号为 Olympus）](https://lifearchitect.ai/olympus/)的那样，AWS Re:invent（[此处为完整直播](https://www.youtube.com/watch?v=LY7m5LQliAo)）拉开帷幕，前 AWS 负责人、现任 Amazon CEO Andy Jassy 投下了一个重磅炸弹：他们推出了一套真正的、具有竞争力的、毫不含糊的多模态基础模型——**Amazon Nova**（[报告](https://www.amazon.science/publications/the-amazon-nova-family-of-models-technical-report-and-model-card)，[博客](https://aws.amazon.com/blogs/aws/introducing-amazon-nova-frontier-intelligence-and-industry-leading-price-performance/)）：


![image.png](https://assets.buttondown.email/images/48dffc0c-5997-46e4-be85-a99782fa4dc7.png?w=960&fit=max)


作为一个令人难以置信的加码（对于大型科技公司的 Keynote 而言），**没有等待名单**——Micro/Lite/Pro/Canvas/Reel 立即正式发布（Generally Available），而 Premier、Speech-to-Speech 和 "Any-to-Any" 将于明年推出。

[LMArena 的 Elo 评分正在进行中](https://x.com/lmarena_ai/status/1864062852589605156?s=46)，但与之前的 [Titan 世代](https://docs.aws.amazon.com/bedrock/latest/userguide/titan-models.html) 相比，这已经是 AI Engineer 们更值得关注的竞争者了。虽然在 [Keynote](https://x.com/swyx/status/1864137540518990281) 中没有被强调，但极高的速度（比 Anthropic/OpenAI 快 2-4 倍的 tok/s）：


![image.png](https://assets.buttondown.email/images/dd0ee43b-c170-477e-b57d-be5ea744a5d3.png?w=960&fit=max)


以及极低的成本（比同级别的 Claude 便宜 25% - 400%）都至关重要：


![image.png](https://assets.buttondown.email/images/77971a26-bd1e-4fec-8c5b-50a1e4e37f4f.png?w=960&fit=max)


将它们的 Arena 分数与最接近的同类模型进行推算，这提供了接近前沿水平的性价比表现：


![image.png](https://assets.buttondown.email/images/1871d04f-e19f-4586-98f6-b4e5d8b2f82a.png?w=960&fit=max)


当然，每个人都在评论这与 [Amazon 再次向 Anthropic 投资 40 亿美元](https://news.ycombinator.com/item?id=42215126) 的关系，对此，这位“万能商店”（Everything Store）的 CEO 给出了一个答案：


![image.png](https://assets.buttondown.email/images/e7ea02ba-d0a5-4307-a848-9b49153e886f.png?w=960&fit=max)


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

**主题 1. Amazon Nova 基础模型：发布、定价与评估**

- **Amazon Nova 发布概览**：[@_philschmid](https://twitter.com/_philschmid/status/1864016010464080260) 全面介绍了**新款 Amazon Nova 模型**，强调了其极具竞争力的定价和基准测试表现。Nova 模型可通过 **Amazon Bedrock** 获取，包含 **Micro**、**Lite**、**Pro** 和 **Premier** 多种配置，部分模型的上下文长度扩展到了 **300k tokens**。
  - **定价策略**：正如 [@_philschmid](https://twitter.com/_philschmid/status/1864018565407650159) 所指出的，Nova 模型的定价低于 **Google DeepMind Gemini Flash 8B** 等竞争对手，在输入/输出 token 定价上极具竞争力。
  - **性能与使用**：根据 [@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1864023058429944147) 的说法，**Nova 系列**模型（尤其是 **Pro** 版本）在特定基准测试中超越了 **GPT-4o** 等模型。

- **评估与基准测试争议**：[@bindureddy](https://twitter.com/bindureddy/status/1864111030521221168) 提出了批评性观点，尽管参数看起来很有前景，但 Nova 在 **LiveBench AI** 指标中的得分低于 **Llama-70B**。这再次体现了模型基准测试的动态性和竞争性。

**主题 2. CycleQD：语言模型中的进化方法**

- **CycleQD 方法论与发布**：最重要的讨论来自 [@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1863773109318475994)，他们介绍了 **CycleQD**，这是一种通过 **Quality Diversity** 进行基于种群的模型合并方法。该方法利用进化计算来开发具有特定领域能力的 LLM Agent，旨在实现终身学习。[HARDMARU](https://twitter.com/hardmaru/status/1863791575492690136) 的另一条推文赞扬了生态位（ecological niche）类比，认为这是 AI 系统获取技能的一种极具吸引力的策略。

**主题 3. AI 幽默与迷因 (Memes)**

- **趣闻轶事与幽默**：[@_arohan_](https://twitter.com/_arohan_/status/1863818654502260973) 幽默地分享了一个瞬间，关于忘记告诉伴侣自己六个月前升职的事情。同时，[@tom_doerr](https://twitter.com/tom_doerr/status/1863958717506375684) 分享了一个关于“不可能”问题的迷因，强调了 AI 交互中轻松的一面。
  - **社交媒体幽默**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1863776928760779142) 提到了一项关于与“不可能”问题相关的 NFT 的戏谑策略。
 
**主题 4. Hugging Face 的担忧与社区回应**

- **存储配额与开源模型争议**：[@far__el](https://twitter.com/far__el/status/1863800159944974438) 等人对 **Hugging Face** 的存储限制表达了不满，认为这可能是 AI 开源社区的一个潜在障碍。[@mervenoyann](https://twitter.com/mervenoyann/status/1863876752201621527) 澄清说 **Hugging Face** 在存储方面依然慷慨，但强调了他们对社区驱动型仓库的适应性调整。
  - **新兴竞争对手**：作为对近期政策变化的回应，[@far__el](https://twitter.com/far__el/status/1864049293214220329) 宣布了 **OpenFace**，这是一项旨在独立于 Hugging Face 自托管 AI 模型的倡议。

**主题 5. 值得关注的新模型创新**

- **混元视频 (HunyuanVideo) 与情感协调模型**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1863836220423655926) 重点介绍了腾讯的 **HunyuanVideo**，指出了其权重开放以及对视频生成模型领域的贡献。同时，[@reach_vb](https://twitter.com/reach_vb/status/1864057723555389841) 发布了 **Indic-Parler TTS**，这是一款具备情感协调能力的文本转语音模型。
  - **模型性能更新**：[@cognitivecompai](https://twitter.com/cognitivecompai/status/1863958091676426427) 记录了关于 **GPT-4o** 性能更新的讨论，例如智力水平有明显提升。

**主题 6. AI 寒冬与行业展望**

- **对 AI 未来的担忧**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1864091994298991042) 对即将到来的 **AI 寒冬** 表示担忧，暗示 AI 的进步或投资热情可能会放缓或倒退，这展示了 AI 行业中波动不定的乐观情绪。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. HuggingFace 实施 500GB 限制，优先考虑社区贡献者**

- **[Huggingface 不再是无限的模型存储空间：新限制为每个免费账户 500 Gb](https://www.reddit.com/gallery/1h53x33)** ([Score: 249, Comments: 76](https://reddit.com/r/LocalLLaMA/comments/1h53x33/huggingface_is_not_an_unlimited_model_storage/)): **HuggingFace** 为免费层级账户引入了 **500GB 存储限制**，标志着其从之前的无限存储政策发生了转变。这一变化影响了该平台上免费用户的模型存储能力。
  - **Huggingface 员工** (**VB**) 澄清说，这是对现有限制的 **UI 更新**，而非新政策。该平台继续为具有价值的社区贡献（如 **model quantization**、**datasets** 和 **fine-tuning**）提供 **storage and GPU grants**，同时打击滥用和垃圾信息。
  - 社区成员报告了巨大的存储使用量，其中一名用户达到了 **8.61 TB/500 GB**，并对未来大型模型（如需要约 130GB 的 **LLaMA 65B**）的可用性表示担忧。讨论集中在包括本地存储和 torrents 在内的潜在解决方案上。
  - 用户辩论了商业影响，指出贡献者已经投入了大量时间和精力为社区创建 quantized 模型。这一变化引发了与 **YouTube** 模式的比较，即用户为消费内容付费，而不是为上传内容付费。


- **[Hugging Face 在所有 25 万+ 公共数据集上添加了 Text to SQL 功能 - 由 Qwen 2.5 Coder 32B 驱动 🔥](https://v.redd.it/e3t9ae0h3g4e1)** ([Score: 119, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1h4w5a3/hugging_face_added_text_to_sql_on_all_250k_public/)): **Hugging Face** 利用 **Qwen 2.5 Coder 32B** 模型，在其 **250,000+ 公共数据集** 中集成了 **Text-to-SQL** 功能。这种集成使得能够从自然语言输入中直接生成跨其整个公共数据集集合的 **SQL** 查询。
  - **VB**（**Hugging Face** 的 GPU Poor 成员）确认，该实现使用 **DuckDB WASM** 进行浏览器内 **SQL** 查询执行。该功能结合了用于查询生成的 **Qwen 2.5 32B Coder** 和基于浏览器的执行能力。
  - 用户对减少手动编写 **SQL** 的需求表示热衷，特别强调了这如何帮助那些在查询编写方面经验较少的人。
  - 该公告获得了积极的反响，评论者赞赏其庆祝的基调，包括演示中使用的纸屑动画。


**Theme 2. DeepSeek and Qwen Surpass Expectations, Challenge OpenAI's Position**

- **OpenAI CEO Sam Altman 称权重开放 AI 模型是坏事。因为 DeepSeek 和 Qwen 2.5 做了 OpenAI 应该做的事！** ([Score: 541, Comments: 216](https://reddit.com/r/LocalLLaMA/comments/1h4n1i9/openweights_ai_models_are_bad_says_openai_ceo_sam/)): 来自 **中国** 的 **DeepSeek** 和 **Qwen 2.5** 开源 AI 模型展示了足以与闭源替代方案相媲美的能力，促使 **Sam Altman** 在接受 **Fox News** 采访（主持人 **Shannon Bream**）时对 **open-weights** 模型表示担忧。这位 **OpenAI CEO** 强调了在 **AI development** 中保持美国对中国的领导地位的战略重要性，同时由于中国开源模型达到了具有竞争力的性能水平，他也面临着批评。
  - 社区情绪强烈批评 **Sam Altman** 和 **OpenAI** 被认为的虚伪，用户指出，鉴于开源模型日益增长的竞争，其 **1570 亿美元** 的估值似乎并不合理。许多人注意到，之前关于 **open-weights** 模型的安全担忧似乎是毫无根据的。
  - 用户强调 **OpenAI** 的技术优势或“护城河”正在迅速缩小，像 **DeepSeek** 和 **Qwen** 这样的 **Chinese models** 已经实现了具有竞争力的性能。一些评论认为 **OpenAI** 的主要优势在于 marketing 而非技术卓越。
  - 多位用户提到了 **OpenAI** 背离了其最初的开源使命，引用了早期与 **Elon Musk** 的沟通以及公司目前反对 **open-weights** 模型的立场。讨论表明 **OpenAI** 的商业策略严重依赖于维持闭源优势。

- **开源才是王道** ([Score: 60, Comments: 14](https://reddit.com/r/LocalLLaMA/comments/1h4n2gb/opensource_is_the_way/)): 在**推理能力**的对比中，**开源模型** (**Deepseek R1** 和 **QwQ**) 在复杂推理问题上超越了**闭源 API** (**Claude Haiku** 和 **OpenAI**)，其中 R1 使用 **Chain of Thought (CoT)** 在 **25 秒**内实现了最快的正确解。一位非编程用户发现 **R1** 和 **QwQ** 对编程任务特别有帮助，同时指出 **Claude Sonnet** 的实用性受限于其免费版本的访问限制和上下文长度约束。
  - **QwQ** 与 **GPT-4o** 的使用限制对比显示，**4o** 对普通用户有 **40 条消息**的严格限制，Plus 用户每 **3 小时**限 **80 条消息**，详见 [OpenAI's FAQ](https://help.openai.com/en/articles/7102672-how-can-i-access-gpt-4-gpt-4-turbo-gpt-4o-and-gpt-4o-mini)。
  - 用户预计 **2025** 年将是**开源模型**的突破之年，并指出 **QwQ** 目前的成本比 **GPT-4o** 低 **5 倍**，同时提供更优越的推理性能。目前 **QwQ** 和 **R1** 均处于 preview/lite 版本阶段。
  - **GPT-4o** 的免费版本很可能是 **4o-mini** 变体，用户根据 Benchmark 结果指出其性能相比 **QwQ** 存在局限。


**主题 3. 利用国家安全担忧推动 AI 监管**

- **[开源 AI = 国家安全：监管呼声愈发强烈](https://v.redd.it/7j5lxfjoyf4e1)** ([Score: 114, Comments: 87](https://reddit.com/r/LocalLLaMA/comments/1h4vk8t/opensource_ai_national_security_the_cry_for/)): **媒体机构**和**政策制定者**继续将**开源 AI** 的发展与**国家安全威胁**挂钩，推动加强监管和审查。这种叙事将不受限制的 AI 开发等同于潜在的安全风险，尽管具体的政策提案仍不明确。
  - 据报道，像 **Yi** 和 **Qwen** 这样的**中国 AI 模型**领先于西方的开源努力，用户指出它们并非基于 **Llama**。多位评论者指出，对美国开源模型的监管将主要使中国的 AI 发展受益。
  - 讨论将当前的 **AI 监管恐惧**与 21 世纪初对**开源软件**的历史性抵制进行了类比，特别是提到了 **Microsoft/SCO** 事件。用户认为，就像 Linux 一样，开源 AI 可能会加速行业创新。
  - 用户批评媒体的叙事是旨在通过监管建立 **AI 垄断**的恐吓行为。许多人提到了 **Fox News** 在技术问题上的可信度，并暗示这是由企业利益而非合法的安全担忧驱动的。


**主题 4. 新工具：功能增强的 Open-WebUI**

- **🧙‍♂️ 增强版 Open-WebUI：我的 ArXiv、ImageGen 和 AI 规划魔法工具箱！🔮** ([Score: 97, Comments: 11](https://reddit.com/r/LocalLLaMA/comments/1h4mq5f/supercharged_openwebui_my_magical_toolkit_for/)): 作者为 **Open-WebUI** 开发了多种工具，包括 **arXiv Search** 工具、**Hugging Face Image Generator**，以及各种功能管道，如使用 **Monte Carlo Tree Search** 的 **Planner Agent**，以及支持多达 **5 种不同 AI 模型**的 **Multi Model Conversations**。该 AI 技术栈运行在 **R7 5800X**、**16GB DDR4** 和 **RX6900XT** 的配置上，包括 **Ollama**、**Open-webUI**、**OpenedAI-tts**、**ComfyUI**、**n8n**、**quadrant** 和 **AnythingLLM**，主要使用 **8B Q6** 或 **14B Q4** 模型及 **16k context**，代码可在 [open-webui-tools](https://github.com/Haervwe/open-webui-tools) 和 [open-webui](https://github.com/open-webui/open-webui) 获取。
  - 用户建议使用 **Python 3.12** 来提升性能，但开发者表示目前由于时间限制暂未实施。
  - 用户对用于研究摘要的 **Monte Carlo Tree Search (MCTS)** 实现表现出兴趣，尽管讨论中未提供具体细节或论文。

- **[我构建了这个用于比较 LLM 的工具](https://v.redd.it/br8pidlihd4e1)** ([评分: 297, 评论: 55](https://reddit.com/r/LocalLLaMA/comments/1h4nz7b/i_built_this_tool_to_compare_llms/)): 提到的**模型比较工具**缺乏具体细节或功能说明，因此无法对其基准测试能力或实现细节提供有意义的技术总结。未提供关于该工具实际功能或特性的额外背景。
  - 用户建议在比较工具中增加**更小的语言模型**，特别提到了 **Gemma 2 2B**、**Llama 3.2 1B/3B**、**Qwen 2.5 1.5B/3B** 等模型，用于 **PocketPal** 等端侧应用。
  - 关于 **Token 计数归一化**引发了大量讨论，详细分析显示，对于相同的输入，**Claude-3.5-Sonnet** 使用的 Token 数量大约是 **GPT-4o** 的**两倍**，这影响了成本计算和上下文长度的比较。
  - 对**“开源 (Open Source)”**和**“开放权重 (Open Weight)”**模型进行了重要区分，指出列表中可自托管的模型在技术上属于“开放权重”，因为它们的训练数据并未公开。


## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. ChatGPT 助力赢得针对房东的 1180 美元小额法庭诉讼案**

- **更新：ChatGPT 让我无需法律代表即可起诉房东——而且我赢了！** ([评分: 1028, 评论: 38](https://reddit.com/r/ChatGPT/comments/1h5sij9/update_chatgpt_allowed_me_to_sue_my_landlord/)): 一名租客赢得了针对 **Jericho High School** 联合校长 **Dr. Joe Prisinzano** 的诉讼。该房东非法收取了 **2,175 美元**的押金（超过了一个月 **1,450 美元**的租金），且未能在冬季修复破碎的窗户。**ChatGPT** 帮助识别了 **2019** 年的一项法律违规行为，并针对报复性的 **5,000 美元**反诉准备了法律辩护。法院判给租客 **1,180 美元**并驳回了反诉。尽管 **Prisinzano** 威胁要上诉并追究诽谤责任，但该案件通过 **Sabrina Ramonov** 的一段播放量超过 **100 万**次的 **TikTok** 视频引起了病毒式关注 [病毒式 TikTok](https://www.tiktok.com/@sabrina_ramonov/video/7425790516278725931)。
  - 租客认为这涉及**公共利益**，因为 **Joe Prisinzano** 是美国顶尖公立高中之一的教育领导者，年薪 **25 万美元**，其不道德的房东行为与其领导地位相悖，理应引起公众关注。
  - 多位用户分享了使用 **ChatGPT** 寻求法律援助的类似经历，但提醒不要完全依赖 AI，并建议进行 **1 小时法律咨询**。一位用户指出，在**纽约**，租客可以针对非法押金索取**双倍赔偿**，针对无证明的扣款索取**三倍赔偿**。
  - 尽管房东书面承认明知故犯，但原审法官对租客提出的 **4,000 美元**惩罚性赔偿请求裁定较为**保守**，仅准予退还非法押金，并针对窗户破损期间给予 **7%** 的租金减免。


**主题 2. 混元视频 (HunyuanVideo) 声称达到 SOTA 视频生成水平，击败 Gen3 和 Luma**

- **[SANA，NVIDIA 图像生成模型终于发布](https://github.com/NVlabs/Sana)** ([评分: 136, 评论: 78](https://reddit.com/r/StableDiffusion/comments/1h5xujr/sana_nvidia_image_generation_model_is_finally_out/)): **NVIDIA** 的图像生成模型 **SANA** 已公开发布。帖子正文中未提供关于模型能力、架构或源代码位置的额外细节。
  - **许可证限制**非常严格——模型仅限**非商业用途**，必须在 **NVIDIA 处理器**上运行，需要 **NSFW 过滤**，并授予 **NVIDIA** 对衍生作品的商业权利。该模型可在 [HuggingFace](https://huggingface.co/Efficient-Large-Model/Sana_1600M_1024px) 上获取。
  - 技术要求包括训练 **0.6B** 和 **1.6B** 模型需要 **32GB VRAM**，推理则分别需要 **9GB** 和 **12GB** VRAM。未来的**量化版本**有望将推理需求降低至 **8GB** 以下。
  - 该模型使用 **Decoder-only LLM**（可能是 **Gemma 2B**）作为文本编码器而非 **T5**。虽然速度极快，但用户反映其图像质量存在问题，且文本生成能力逊于 **Flux**。演示地址：[nv-sana.mit.edu](https://nv-sana.mit.edu/)。

- **Tencent Hunyuan-Video：在文本生成视频领域击败 Gen3 和 Luma。** ([Score: 37, Comments: 15](https://reddit.com/r/StableDiffusion/comments/1h5kkuv/tencent_hunyuanvideo_beats_gen3_luma_for/)): **Tencent** 发布了 **Hunyuan-video**，这是一款开源的文本生成视频模型，声称在测试中表现优于闭源竞争对手 **Gen3** 和 **Luma1.6**。该模型包含音频生成功能，可以在其[演示视频](https://youtu.be/YbN8Am_0bpk?si=y1OciGLYmfGD713j)中预览。
  - 该模型已在 [GitHub](https://github.com/Tencent/HunyuanVideo) 和 [Hugging Face](https://huggingface.co/tencent) 上线，官方项目页面位于 [Tencent Hunyuan](https://aivideo.hunyuan.tencent.com/)。
  - 系统要求包括：**720x1280** 分辨率（**129 帧**）需要 **60GB GPU 显存**，或 **544x960** 分辨率（**129 帧**）需要 **45GB**，这引发了关于在消费级 GPU 上运行该模型的幽默评论。
  - **ComfyUI** 集成已列入项目路线图的未来开发项，表明计划扩大其易用性。


**主题 3. ChatGPT 母公司 OpenAI 考虑引入广告**

- **[我们确实应该担心 2025 年会以 W T F 开头](https://i.redd.it/mo6zf59jol4e1.jpeg)** ([Score: 190, Comments: 157](https://reddit.com/r/ChatGPT/comments/1h5jg8e/we_definitely_should_be_concerned_that_2025/)): **OpenAI** 计划到 **2025** 年在 **ChatGPT** 中实施**广告**，这引发了人们对 **AI 变现**未来走向的担忧。社区对这一进展表示怀疑，质疑其对用户体验的影响以及对 **AI 商业模式**的更广泛影响。
  - **数据变现**担忧主导了讨论，用户预测其演变过程将从针对免费用户的“推广建议”最终发展到涵盖所有层级的**赞助内容**。社区预期会整合基于对话数据的**定向广告**，类似于 **Prime Video** 的广告模式。
  - “**平台劣化**”（**enshittification**）的概念成为一个关键主题，用户预见到服务将从以用户为中心转向**利润最大化**。多位评论者指出 **Claude** 和**本地 LLM**（如 **Llama**、**QWQ** 和 **Qwen**）是潜在的替代方案。
  - 用户对 **AI 生成广告**可能带来的微妙操纵表示担忧，并指出传统的**广告拦截器**可能对集成在 AI 中的促销内容无效。讨论强调了 **ChatGPT** 的对话性质可能使赞助内容特别难以识别或监管。


- **[ChatGPT 可能会引入广告——尽管 Sam Altman 并不热衷于此](https://techcrunch.com/2024/12/02/ads-might-be-coming-to-chatgpt-despite-sam-altman-not-being-a-fan/)** ([Score: 70, Comments: 108](https://reddit.com/r/OpenAI/comments/1h5itfo/ads_might_be_coming_to_chatgpt_despite_sam_altman/)): 尽管 **CEO Sam Altman** 此前曾表示反感基于广告的营收模式，但 **OpenAI** 仍可能在 **ChatGPT** 中引入**广告**。仅标题就暗示了 **OpenAI** 变现策略的潜在转变，尽管尚未提供具体的时间表或实施细节。
  - **用户反应**绝大多数是负面的，许多人表示如果实施广告，他们将立即**取消订阅**。多位用户将其与 **Prime Video** 和 **Disney+** 等流媒体服务相提并论，这些服务甚至在付费层级中也引入了广告。
  - 原始文章似乎是**标题党**，正如用户指出的那样，**OpenAI** “目前没有”添加广告的“活跃计划”，一些人认为这仅仅是在测试公众反应。澄清这一点的热门评论获得了 **177 个点赞**。
  - 用户担心基于广告的激励机制会损害 **ChatGPT** 的完整性，将其比作信任的朋友推荐与拿佣金的销售人员之间的区别。几条评论强调了广告通常如何随着时间的推移从免费层级扩展到付费服务，并以**有线电视**和**流媒体平台**为例。


**主题 4. 沃达丰 (Vodafone) 的 AI 广告展示了 AI 视频制作的新标杆**

- **[沃达丰（Vodafone）绝对令人惊叹的 AI 广告。比可口可乐（Coca-Cola）的尝试好得多。](https://v.redd.it/b7dkisowdn4e1)** ([评分: 163, 评论: 63](https://reddit.com/r/ChatGPT/comments/1h5p92c/absolutely_incredible_ai_ad_by_vodafone_much_much/)): **Vodafone** 制作了一条 **AI 生成的商业广告**，获得了观众的积极反响，评论者特别将其与 **Coca-Cola** 之前的 AI 广告尝试进行了对比，并给予了更高评价。帖子中未提供关于该广告内容或创作过程的更多细节。
  - **观众反应** 大多是负面的，批评该广告缺乏连贯性且过度使用刻板镜头。多位用户指出，该广告在 **静音状态下观感极差**，且包含大量脱节的场景。
  - 该广告的 **成本效率** 受到关注，据估计仅为“普通商业广告的 **十分之一**”。用户讨论了其 **技术成就** 是否超过了艺术价值，其中视频剪辑师获得的赞誉多于 AI 本身。
  - 讨论集中在 **行业影响** 上，特别是关于演员和传统制作团队可能被取代的问题。引用了一篇关于该广告创作的 **Campaign Live 文章**，但[仍处于付费墙后](https://www.campaignlive.co.uk/article/behind-scenes-sebastian-strasser-directing-ai-only-ad-vodafone/1898326)。


---

# AI Discord 摘要回顾

> 由 O1-preview 提供的摘要之摘要的总结

**主题 1：新型优化器与训练技术革新 AI**

- [**DeMo 实现去中心化模型训练**](https://arxiv.org/abs/2411.19870)：**Nous Research** 发布了 **DeMo 优化器**，通过 [Nous DisTrO](https://distro.nousresearch.com) 实现 15B 模型的去中心化预训练，其性能可媲美中心化训练方法。实时运行展示了其效率，可以在[此处](https://distro.nousresearch.com)观看。
- [**Axolotl 集成 ADOPT 优化器**](https://github.com/axolotl-ai-cloud/axolotl/pull/2104)：**Axolotl AI** 引入了最新的 **ADOPT 优化器**，可在任何 beta 值下提供最佳收敛，并增强模型训练效率。邀请工程师在更新后的代码库中尝试这些增强功能。
- [**Pydantic AI 桥接 LLM 集成**](https://ai.pydantic.dev/)：**Pydantic AI** 的发布提供了与 **LLMs** 的无缝集成，增强了 AI 应用。它还与 DSPy 的 [DSLModel](https://pypi.org/project/dslmodel/) 集成，简化了 AI 工程师的开发工作流。

**主题 2：新型 AI 模型引发热议与辩论**

- [**亚马逊 Nova 瞄准 GPT-4o**](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)：**Amazon** 通过 Bedrock 发布了 **Nova 基础模型**，拥有极具竞争力的能力和高性价比的定价。Nova 支持零日集成，并以实惠的选择扩展了 AI 模型版图。
- [**混元视频（Hunyuan Video）树立文本生成视频新标杆**](https://x.com/angrypenguinpng/status/1863811509219950835?s=46)：**腾讯混元视频（Hunyuan Video）** 作为领先的开源文本生成视频模型发布，尽管资源需求较高，但仍令用户印象深刻。初步反馈积极，期待未来的效率优化。
- [**Sana 模型的效率受到质疑**](https://nvlabs.github.io/Sana/)：**Stability.ai** 社区对新型 **Sana 模型** 展开辩论，质疑其相对于 **Flux** 等现有模型的实际优势。一些人建议使用先前的模型可能会产生相似或更好的结果。

**主题 3：AI 工具面临性能与更新挑战**

- **Cursor IDE 的延迟促使用户转向 Windsurf**：由于对 **Cursor IDE** 在 **Next.js** 项目中的延迟感到沮丧，用户因 Cursor 持续的性能问题而回归 **Windsurf**。Cursor 的语法高亮和聊天功能也因导致“视觉不适”和阻碍可用性而受到批评。
- **OpenInterpreter 重构以提升速度与智能**：**OpenInterpreter** 开发分支的完全重构带来了一个“更轻、更快、更聪明”的工具。新的 `--serve` 选项引入了一个兼容 OpenAI 的 REST 服务器，增强了可访问性和可用性。
- **Unsloth AI 微调遇到障碍**：用户在 **Llama 3.2** 上进行 **LoRA 微调** 时遇到困难，并面临 **xformers** 兼容性问题，导致社区共享修复方案。挑战包括 OOM 错误以及训练期间序列长度配置的不一致。

**主题 4：社区探索 AI 方法与框架**

- **Function Calling vs MCP：AI 状态管理器的交锋**：**Nous Research** 讨论了 **function calling** 与 **Model Context Protocol (MCP)** 在管理 AI 模型状态和动作方面的优劣，强调了两者在应用场景上的困惑以及对更清晰指南的需求。
- **ReAct 范式的有效性取决于实现细节**：**LLM Agents** 课程参与者强调，**ReAct** 的成功取决于具体的实现细节，如 prompt 设计和状态管理。由于 AI 领域存在“模糊定义”，基准测试应反映这些细节。
- **DSPy 和 Pydantic AI 增强开发者工作流**：**DSPy** 集成了 **Pydantic AI**，允许通过 **DSLModel** 进行高效开发。现场演示展示了高级 AI 开发技术，激发了在项目中实现 Pydantic 功能的热情。

**主题 5：AI 社区参与机遇与活动**

- [**前 Google 员工创立新公司，邀请合作伙伴**](https://werebuilding.ai/)：**Raiza** 在 Google 工作 5.5 年后离职，与前 **NotebookLM** 团队成员共同创立了一家新公司，并邀请他人通过 [*hello@raiza.ai*](mailto:hello@raiza.ai) 加入。他们庆祝了重大成就，并计划与社区一起构建创新产品。
- [**Sierra AI 在宣讲会上挖掘人才**](https://youtube.com/live/-iWdjbkVgGQ?feature=share)：**Sierra AI** 举办了一场独家信息宣讲会，揭晓了他们的 **Agent OS** 和 **Agent SDK**，同时寻求优秀开发者加入团队。参与者可以[在此预约](https://lu.ma/agents-hackathon-sierra)以锁定机会。
- [**多 Agent 聚会凸显协作创新**](https://t.co/VqmlVGnWT4)：即将在 **GitHub 总部**举行的**多 Agent 聚会**将邀请专家讨论使用 **CrewAI** 自动化任务以及使用 **Arize AI** 评估 Agent，促进 Agentic Retrieval 应用中的协作。

---

# 第一部分：Discord 高层级摘要

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **使用 DisTrO 进行去中心化预训练**：Nous 已启动一个 15B 参数语言模型的**去中心化预训练**，使用了 [Nous DisTrO](https://distro.nousresearch.com) 以及来自 **Oracle** 和 **Lambda Labs** 等合作伙伴的硬件，展示了与传统使用 **AdamW** 的中心化训练相匹配甚至更优的损失曲线。
   - 实时运行情况可[在此](https://distro.nousresearch.com)观看，配套的 [DeMo 论文](https://arxiv.org/abs/2411.19870)和代码将很快公布。
- **DeMo 优化器发布**：**DeMo** 优化器通过在每个优化步骤中仅同步极小的模型状态，实现了神经网络的并行训练，在减少加速器间通信的同时增强了收敛性。
   - 有关此方法的详细信息可在 [DeMo 论文](https://arxiv.org/abs/2411.19870)中找到，源代码可在 [GitHub](https://github.com/bloc97/DeMo) 上获取。
- **DisTrO 训练更新**：正在进行的 **DisTrO** 训练运行已接近完成，预计本周末将公布有关硬件和用户贡献的具体细节。
   - 此次运行主要作为测试，可能不会立即为用户提供公开的注册表或教程。
- **AI 模型中的 Function Calling vs MCP**：**Function calling** 用于管理 AI 模型内部的状态和动作，而 **MCP** 在实现复杂功能方面提供了替代优势。
   - 在区分 MCP 与 function calling 方面存在一些困惑，强调了对各自应用场景提供更清晰指南的需求。
- **在特定任务中使用较小模型**：在某些创意任务中，较小的 AI 模型表现可能优于较大的模型，并具有处理速度更快、资源消耗更少等优点。
   - 建议采用平衡的方法，由较小的模型处理状态管理，而将较大的模型保留用于讲故事等更密集的任务。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX Adoption Accelerates in Major AI Labs**：**JAX** 在主流 AI 实验室的采用加速。成员透露，**Anthropic**、**DeepMind** 和其他领先的 AI 实验室正越来越多地在模型中使用 **JAX**，尽管其主要使用程度因机构而异。
   - 关于 **JAX** 是否会超越 **PyTorch** 的主导地位存在持续争论，呼吁在行业采用率和实践方面提高透明度。
- **Vendor Lock-in Raises Concerns in Academic Curricula**：学术课程中的**供应商锁定（Vendor Lock-in）**引发关注。讨论强调了学术界中的供应商锁定问题，科技公司通过为 **PyTorch** 和 **JAX** 等特定框架提供资源来影响大学课程。
   - 观点不一；一些人看到了建立合作伙伴关系的好处，而另一些人则担心这会限制学生接触更广泛的工具和框架。
- **DeMo Optimizer Enhances Large-Scale Model Training**：**DeMo** 优化器增强大规模模型训练。**DeMo** 优化器引入了一种通过解耦动量更新（momentum updates）来最小化加速器间通信的技术，这使得在无需高速网络完全同步的情况下实现更好的收敛。
   - 其极简设计将优化器状态大小减少了**每个参数 4 字节**，使其在训练超大规模模型时具有优势。
- **Externalizing Evals via Hugging Face Proposed**：提议通过 **Hugging Face** 外部化 **evals**。有人提议允许通过 **Hugging Face** 外部加载 **evals**，类似于数据集和模型的集成方式。
   - 这种方法可以简化数据集和相关 eval YAML 文件的加载过程，但需要解决可见性和版本控制问题以确保**可复现性（reproducibility）**。
- **wall_clock_breakdown Configures Detailed Logging**：**wall_clock_breakdown** 配置详细日志记录。成员们发现 **wall_clock_breakdown** 配置选项可以启用详细的日志消息，包括 **optimizer_allgather** 和 **fwd_microstep** 等优化器计时指标。
   - 澄清确认，启用此选项对于生成深入的性能日志至关重要，有助于性能诊断和优化。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Socket Communication Delays**：**Mojo** 套接字通信（**socket communication**）延迟。由于待处理的语言特性，**Mojo** 中的套接字通信实现被推迟，计划开发一个支持可交换网络后端（如 **POSIX sockets**）的标准库。
   - 计划进行重大重写，以确保在这些语言特性可用后进行正确集成。
- **Mojo's SIMD Support Simplifies Programming**：**Mojo** 的 **SIMD** 支持简化了编程。讨论强调 **Mojo** 的 **SIMD** 支持相比 **C/C++ intrinsics**（通常很混乱）简化了 **SIMD** 编程。
   - 目标是在未来的更新中将更多 **intrinsics** 映射到标准库，以减少直接使用。
- **High-Performance File Server Project in Mojo**：**Mojo** 中的高性能文件服务器项目。一个旨在为游戏开发高性能文件服务器的项目，其目标是实现比 **Nginx** 高出 **30%** 的每秒数据包处理率。
   - 目前，该项目在延迟的套接字通信功能可用之前，利用外部调用进行网络连接。
- **Reference Trait Proposal for Mojo**：**Mojo** 的 **Reference** Trait 提案。**Mojo** 中关于 `Reference` trait 的提案旨在增强对 **Mojo** 代码中可变和可读引用的管理。
   - 这种方法预计将改进借用检查（**borrow-checking**），并减少函数参数中关于可变性的困惑。
- **Magic Package Distribution Launch**：**Magic Package Distribution** 发布。**Magic Package Distribution** 正在开发中，早期访问预览即将推出，允许社区成员通过 **Magic** 分发软件包。
   - 团队正在寻找测试人员来完善该功能，邀请成员通过对软件包评审回复 **🔍** 或对安装回复 **🧪** 来参与。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenRouter 性能落后于直接 API**：基准测试分析显示，通过 OpenRouter 访问的模型性能低于直接通过 API 访问的模型，引发了关于优化策略的讨论。
   - 用户正在协作探索提高 OpenRouter 效率的解决方案，表明社区正在努力解决这些差异。
- **Aider 为开发者推出增强功能**：Aider 最新的 `--watch-files` 功能简化了 AI 指令到编码工作流的集成，同时还包括 `/save`、`/add` 和上下文修改等功能，详见其 [options reference](https://aider.chat/docs/config/options.html#--gitignore)。
   - 这些更新受到了好评，用户注意到透明度有所提高，编程体验更加明晰。
- **Amazon 在 re:Invent 上发布六款新基础模型**：在 re:Invent 期间，Amazon 宣布了六款新的基础模型，包括 **Micro**、**Lite** 和 **Canvas**，强调了它们的多模态能力和极具竞争力的定价。
   - 这些模型将仅通过 Amazon Bedrock 提供，并被定位为其他美国前沿模型的高性价比替代方案。
- **使用 Model Context Protocol 增强 Aider 的上下文**：用户一直在集成 **Model Context Protocol (MCP)** 以提高 Aider 的上下文管理能力，特别是在代码相关场景中，如[此视频](https://youtu.be/9mciRwpcLNY?si=IqPQDJ-lgBlYGUre)中所讨论的。
   - 正在利用 **IndyDevDan's agent** 和 **Crawl4AI** 等工具创建优化的文档，以便实现无缝的 LLM 集成。
- **解决 Python 3.12 下 Aider 更新挑战**：将 Aider 更新到版本 **0.66.0** 时遇到问题，包括包安装期间的命令失败，这些问题通过显式调用 Python **3.12** 解释器得到解决，如 [pipx installation guide](https://aider.chat/docs/install/pipx.html) 中所述。
   - 这种方法使客户能够成功升级并利用最新功能，而不会出现重复问题。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 在 Next.js 项目中出现延迟**：用户报告称，在对中大型项目使用 **Next.js** 进行开发时，**Cursor** 出现明显延迟，需要频繁执行 **'Reload Window'** 命令。
   - 性能因系统 RAM 而异，**16GB** 内存的用户比 **32GB** 配置的用户遇到更多延迟，这引发了对 **Cursor 性能一致性**的担忧。
- **Windsurf 可靠性优于 Cursor**：由于最新 **Cursor** 更新中修复重复失效，一些用户回退到了 **Windsurf**。
   - 他们强调 **Windsurf's agent** 能成功编辑多个文件且不丢失注释，而这是 **Cursor** 目前缺乏的功能。
- **Cursor Agent 的功能请求**：成员们请求在 **Cursor Agent** 中添加 **@web** 功能，以增强实时信息获取能力。
   - 提到了 Agent 无法识别文件更改的问题，导致了对其**可靠性**的沮丧。
- **Cursor 语法高亮的缺点**：初次使用者报告称 **Cursor 的语法高亮**导致视觉不适并阻碍了易用性。
   - 投诉包括各种 **VS Code addons** 在 **Cursor** 中运行异常，降低了整体用户体验。
- **Cursor 更新后的聊天问题**：在最近的更新后，用户遇到了 **Cursor 聊天功能**的问题，包括模型幻觉和性能不一致。
   - 反馈表明 **模型质量**有所下降，使得编码任务变得更加**具有挑战性**和**令人沮丧**。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 面临性能下降**：多名用户在使用 [Perplexity AI 的功能](https://discord.com/channels/1047197230748151888/1047649527299055688/1313130820814377100)时遇到持续的**性能下降**和无限加载问题，这表明可能存在**扩展性问题 (scaling issues)**。
   - 这些性能问题在其他平台上也有所体现，导致用户考虑转向 **API 服务**以获得更稳定的体验。
- **用户探索图像生成能力**：关于**图像生成工具**的讨论涉及分享能产生意想不到且极具创意结果的 **prompts**。
   - 用户尝试使用*量子主题提示词 (quantum-themed prompts)* 来生成独特的视觉输出，展示了图像生成模型的多样化应用。
- **Amazon Nova 与 ChatGPT 和 Claude 的对比**：社区对 AI 模型进行了深入对比，特别是 **Amazon Nova** 与 **ChatGPT** 和 **Claude** 等平台之间的对比。
   - 用户根据特定任务及其与 Perplexity 等工具的集成情况，评估了各种基础模型 (foundational models) 的有效性。
- **Google Gemini 与 Drive 集成的问题**：一位用户强调了通过 **Google Gemini** 访问 Google Drive 文档时存在**访问不一致**的问题，对其可靠性提出质疑。
   - 用户担心高级功能是否仅限于付费版本，并促请提供实际演示。
- **API 错误响应及用户规避方法**：用户报告了**间歇性的 API 错误**，如 `unable to complete request`，导致了困惑。
   - 一种临时的规避方法是在用户 **prompts** 中添加前缀，在等待解决方案期间减少错误发生。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **使用 LoRA 微调 Llama 3.2**：用户报告了使用 LoRA **微调 Llama 3.2** 时的挑战，特别是从 **tokenization** 到 **processor management** 的过渡，并建议修改 [Colab notebooks](https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh) 以成功执行。
   - 故障排除步骤包括解决 **xformers 安装问题**，并确保与当前的 **PyTorch** 和 **CUDA** 版本兼容。
- **模型兼容性和 xformers 问题**：几位用户遇到了 **xformers** 与其现有 **PyTorch** 和 **CUDA** 环境的**兼容性问题**，导致运行时错误。
   - 建议包括重新安装匹配版本的 xformers，并验证依赖项以解决这些问题。
- **使用 LLaVA-CoT 微调 QWen2 VL 7B**：一位成员使用 [LLaVA-CoT 数据集](https://huggingface.co/forcemultiplier/Qwen2-VL-7B-Instruct-LLaVA-CoT-2000steps-r16a16-merged)微调了 **QWen2 VL 7B**，并发布了训练脚本和数据集供社区使用。
   - 生成的模型具有 **8.29B 参数**，并使用 **BF16** 张量类型，训练脚本可在此处获取 [here](https://huggingface.co/datasets/forcemultiplier/LLaVA-CoT-30k-jsonl-trainkit)。
- **Unsloth 模型中的 GGUF 转换挑战**：用户在将模型保存为 **GGUF** 时遇到问题，在转换过程中遇到关于缺少 'llama.cpp/llama-quantize' 等文件的运行时错误。
   - 尝试通过重启 **Colab** 来解决这些问题的努力未获成功，这表明底层库可能发生了近期更改。
- **训练模型中的部分可训练 Embeddings**：一位用户讨论了创建**部分可训练的 Embeddings**，但在训练期间面临 **forward** 函数未被调用的挑战。
   - 社区反馈表明模型可能直接访问权重而不是修改后的 **head**，因此需要更深层次的集成。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Raiza 离开 Google 创办新公司**：**Raiza** 宣布在 **Google** 工作 **5.5 年**后离职，强调了在 **NotebookLM 团队**取得的显著成就，并开发了一款被数百万人使用的产品。
   - Raiza 正与两名原 **NotebookLM** 成员共同创办一家新公司，邀请合作者通过 [werebuilding.ai](https://werebuilding.ai/) 加入，并可通过 *hello@raiza.ai* 联系。
- **NotebookLM 在剧本和播客中的创意用途**：用户详细介绍了如何利用 **NotebookLM** 进行**剧本创作**、开发详细的摄像和灯光设置，以及将剧本集成到视频项目中。
   - 另一位用户通过逐章概述内容，成功生成了长篇播客剧集，并在纪录片风格的项目中使用 **Eleven Labs** 处理音频和视觉效果。
- **使用 PDF OCR 工具增强文档管理**：讨论强调了使用 **PDF24** 对扫描文档进行 OCR 处理，将其转换为具有强大安全协议的可搜索 PDF。
   - 推荐使用 **PDF24** 将图像和照片转换为可搜索格式，无需安装或注册即可优化文档的可用性。
- **NotebookLM 的功能请求与集成挑战**：用户表达了对 **NotebookLM** 中**无限制音频生成**的需求，建议采用订阅模式以突破目前每日 **20** 次的限制。
   - 用户注意到了处理长 PDF 的挑战，推测 **Gemini 1.5 Pro** 可能会提供更强大的能力，同时也对 **Google Drive** 集成的不稳定性表示沮丧。
- **多语言 AI 支持的进展与问题**：有关于更改 **NotebookLM** 语言设置以支持非英语输出的咨询，目前的指南建议通过修改 Google 账户设置来实现。
   - 用户报告了 AI 生成语言输出的成功率各不相同，特别是在苏格兰或波兰等口音方面，表明多语言能力仍有改进空间。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **意大利 AI 监管法案强制执行数据删除**：意大利宣布计划**禁用 AI 平台**（如 [OpenAI](https://openai.com/)），除非用户可以请求删除其数据，这引发了关于监管有效性的辩论。
   - 针对**地理位置封锁**的无效性提出了担忧，并讨论了用户绕过这些限制的潜在方法。
- **ChatGPT Plus 计划遭遇功能故障**：用户报告称，在支付 **$20** 订阅 **ChatGPT Plus** 计划后，**图像生成**和**文件读取**等功能无法正常运行。
   - 此外，多名成员指出他们收到的回复似乎过时，且该问题已持续一周多。
- **GPT 在账单汇编中面临功能问题**：一位用户强调了一个旨在汇编计费工时的 **GPT** 存在的问题，提到它会遗漏条目，且难以生成 **XLS 兼容列表**。
   - 出现了幽默的推测，质疑该 GPT 是否*对工作感到厌倦*，反映了用户对该工具可靠性的挫败感。
- **利用 Custom Instructions 定制 ChatGPT**：成员们正在利用 [custom instructions](https://help.openai.com/en/articles/8096356-custom-instructions-for-chatgpt) 来调整 **ChatGPT** 的写作风格，并将此方法与创建新的 **GPTs** 区分开来。
   - 建议提供**示例文本**以帮助 ChatGPT 调整其输出，增强与用户特定叙事偏好的契合度。
- **提升 AI 工程师的 Prompt Engineering 技能**：AI 工程师表示有兴趣获取**免费或低成本资源**，以改进他们的 **Prompt Engineering**，从而利用 **OpenAI ChatGPT** 开发自定义 GPT。
   - 讨论强调了优化交互技术对于最大化 ChatGPT 效能和能力的重要性。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **模型移除与降价**：两个模型 `nousresearch/hermes-3-llama-3.1-405b` 和 `liquid/lfm-40b` 已停止提供，提示用户**增加额度 (credits)** 以维持其 API 请求。
   - 显著的降价措施：**nousresearch/hermes-3-llama-3.1-405b** 从每百万 token 4.5 降至 0.9，**liquid/lfm-40b** 从 1 降至 0.15，在移除后提供了更实惠的替代方案。
- **Hermes 405B 模型移除**：**Hermes 405B** 不再可用，标志着该模型的逐步淘汰，用户正在讨论替代方案的成本，并倾向于现有的免费模型。
   - 移除引发了对模型可用性的担忧，一些用户考虑购买价格日益上涨的模型，而另一些用户则坚持使用免费选项。
- **OpenRouter API Key 管理**：OpenRouter 现在支持 **API keys** 的创建和管理，允许用户为每个 key 设置和调整**额度限制 (credit limits)**，且不会自动重置。
   - 用户通过手动管理 key 的使用情况来控制其应用程序访问，确保安全且受控的 **API 访问**。
- **Gemini Flash 错误**：用户在访问 **Gemini Flash** 时遇到了短暂的 **525 Cloudflare 错误**，该问题很快自行解决。
   - 注意到该模型的不稳定性，建议通过 OpenRouter 的聊天界面验证其功能。
- **BYOK 访问更新**：团队宣布 **BYOK (Bring Your Own Key)** 访问将很快对所有用户开放，尽管目前的私测阶段已暂停。
   - 正在进行调整以解决现有问题，然后再广泛推出该功能。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **掌握 LORA 创建**：用户分享了**创建有效 LORA 的策略**，例如使用由图像制成的背景 LORA，并使用 Photoshop 或 Krita 等软件精修输出。
   - 一位成员建议在训练前**精修生成的图像**，以确保更高质量的结果。
- **Stable Diffusion 设置技巧**：多位用户寻求 **Stable Diffusion 设置指南**，建议包括使用 [ComfyUI - 入门教程](https://www.youtube.com/watch?v=AbB33AxrcZo&list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x) 以及各种云端选项。
   - 成员强调了决定是在本地运行还是利用云端 GPU 的重要性，推荐使用 [Vast.ai](https://vast.ai/) 进行 GPU 租赁。
- **诈骗警报策略**：对服务器中**诈骗者**的担忧促使用户分享**警告**，并建议向 Discord 举报可疑账号。
   - 用户讨论了如何识别钓鱼尝试，以及某些账号如何冒充支持人员来欺骗成员。
- **比较 GPU 性能**：对话强调了 **GPU 性能**的差异，用户比较了不同型号的体验，并强调了**显存 (memory) 和速度**的重要性。
   - 一位用户指出，由于电费成本，较便宜的云端 GPU 选项可能比本地设置提供更好的整体性能。
- **评估 Sana 模型**：成员讨论了一个名为 [Sana](https://nvlabs.github.io/Sana/) 的新模型，注意到其与早期版本相比的**效率**和**质量**，同时对其商业用途持保留意见。
   - 有建议称，对于日常用途，使用 **Flux** 或之前的模型可能会产生相似或更好的结果。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Pydantic AI 与 LLM 集成**：来自 **Pydantic** 的新 [Agent Framework](https://ai.pydantic.dev/) 现已上线，旨在与 **LLM** 无缝集成，以实现创新的 **AI 应用**。
   - 然而，一些用户对其与 **LangChain** 等现有框架的差异化表示怀疑，认为它与目前的解决方案非常相似。
- **Bolt 在 2 个月内达到 800 万美元 ARR**：作为一款 **Claude Wrapper**，**Bolt** 在短短 **2 个月** 内突破了 **800 万美元 ARR**，嘉宾包括 [@ericsimons40](https://x.com/ericsimons40) 和 [@itamar_mar](https://x.com/itamar_mar)。
   - 播客节目深入探讨了 Bolt 的 **增长策略**，并讨论了 **code agent engineering**，重点介绍了与 [@QodoAI](https://x.com/qodoi) 的合作以及 [StackBlitz](https://stackblitz.com) 的首次亮相。
- **腾讯发布混元视频（Hunyuan Video），领跑开源领域**：**腾讯** 发布了 [Hunyuan Video](https://x.com/angrypenguinpng/status/1863811509219950835?s=46)，将其确立为以高质量著称的顶级 **开源 text-to-video** 技术。
   - 初步的用户反馈指出其渲染的 **资源需求较高**，但对其即将推出的 **效率提升** 持乐观态度。
- **亚马逊发布 Nova 基础模型**：**亚马逊** 宣布了其新的基础模型 [**Nova**](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)，定位是与 **GPT-4o** 等先进模型竞争。
   - 早期评估显示出潜力，但用户体验仍然 **褒贬不一**，一些人认为它不像亚马逊之前发布的 **模型** 那样令人印象深刻。
- **ChatGPT 面临姓名过滤故障**：**ChatGPT** 遇到了一个问题，即由于系统故障，特定姓名（如 *David Mayer*）会触发响应中断。
   - 此问题不影响 **OpenAI API**，并引发了关于 **姓名关联** 如何影响 **AI 行为** 的讨论。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank 3.5 增强多语言搜索**：Cohere 推出了 **Rerank 3.5**，通过 [`rerank-v3.5` API](https://cohere.com/blog/rerank-3pt5) 提供增强的推理能力，并支持包括阿拉伯语、法语、日语和韩语在内的 **100 多种语言**。
   - 用户对性能的提升及其与包括 *多媒体* 内容在内的各种数据格式的兼容性表示热烈欢迎。
- **Cohere 宣布 API 弃用**：**Cohere** 宣布 **弃用** 旧模型，并提供了关于 [弃用端点](https://docs.cohere.com/docs/deprecations) 的详细信息以及推荐的替代方案，作为其 **模型生命周期** 管理的一部分。
   - 此举影响了依赖旧模型的应用程序，促使开发者相应地更新其集成。
- **Harmony 项目发布 NLP 协调工具**：**Harmony** 项目推出了用于协调问卷项目和元数据的 **NLP** 工具，使研究人员能够 [跨研究比较问卷项目](https://harmonydata.ac.uk/compare-harmonise-instruments/gad-7-vs-beck-anxiety-inventory/)。
   - 该项目总部位于 **UCL**，正与多家大学和专业人士合作，以完善其 [文档检索能力](https://harmonydata.ac.uk/)。
- **API 密钥延迟触发 TooManyRequestsError**：有用户报告称，尽管升级到了生产环境 **API** 密钥，仍遇到 **TooManyRequestsError**，将问题归因于潜在的 **API 密钥设置延迟**。
   - 建议联系 [support@cohere.com](mailto:support@cohere.com) 寻求帮助，有迹象表明设置延迟通常很小。
- **Stripe 集成导致支付问题**：一些用户在 Cohere 平台上遇到了 **信用卡支付问题**，尽管之前的交易很成功。
   - 成员们建议问题可能出在用户的银行，并建议联系 [support@cohere.com](mailto:support@cohere.com)，因为支付是通过 **Stripe** 处理的。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Xmma 和 Nvjet 在特定基准测试中表现优于 Cutlass**：成员们评估了 **Xmma kernels**，注意到 **nvjet** 在小尺寸上正在赶上，对于 **N=8192**，自定义 kernel 的运行速度比 **cutlass** 快 **1.5%**。
   - **nvjet** 通常与 **cutlass** 竞争激烈，但在某些特定情况下 **cutlass** 可能略胜一筹。
- **Triton MLIR Dialects 文档批评**：一位成员询问了关于 **Triton MLIR Dialects** 的文档，指出大部分 [TritonOps documentation](https://triton-lang.org/main/dialects/TritonOps.html) 非常简略且缺乏详尽的示例。
   - 另一位成员指出 GitHub 上的 [programming guide](https://github.com/triton-lang/triton/tree/main/docs/programming-guide) 也很简略且未完成，旨在帮助使用 **Triton language** 的开发者。
- **CUDARC Crate 支持手动 CUDA 绑定**：[CUDARC](https://crates.io/crates/cudarc) crate 提供了 CUDA API 的绑定，由于是手动实现，目前仅支持 **matrix multiplication**。
   - 测试显示，优化 **matmul** 函数消耗了大部分开发时间。
- **GPU Warp Scheduler 和 FP32 核心分布见解**：一位成员解释说，一个 warp 包含 **32 threads**，并行使用 **32 FP32 cores**，导致每个 **SM** 有 **128 FP32 cores**。
   - 注意到 **A100** 的 4 个 warp schedulers 对应 **64 FP32 cores**，而 **RTX 30xx 和 40xx** 系列则拥有 **128 FP32 cores**，两者之间存在差异。
- **KernelBench 发布及排行榜完整性问题**：[@anneouyang](https://twitter.com/anneouyang/status/1864014135824162995) 推出了 **KernelBench (Preview)**，用于评估 LLM 生成的 **GPU kernels** 以进行神经网络优化。
   - 用户对排行榜上不完整的 **fastest kernels** 表示担忧，并引用了一个 [不完整的 kernel 解决方案](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **NVIDIA 财务洞察揭晓**：**@pyquantnews** 展示了如何利用 **NVIDIA** 的财务报表进行简单的收入查询和复杂的业务风险分析，并使用了 [设置 LlamaIndex 的实际代码示例](https://t.co/989lnoNeV2)。
   - 该方法通过利用结构化财务数据增强了商业智能。
- **使用 Google Drive 简化 LlamaCloud**：**@ravithejads** 概述了配置以 **Google Drive** 作为数据源的 **LlamaCloud** 流水线的逐步过程，包括 chunking 和 embedding 参数。完整的设置说明可在 [此处](https://t.co/KnQ9nUDWYE) 查看。
   - 该指南协助开发者将文档索引与 LlamaIndex 无缝集成。
- **亚马逊发布 Nova 基础模型**：**Amazon** 推出了 **Nova**，这是一系列基础模型（foundation models），与竞争对手相比价格更实惠，并提供 day 0 支持。通过 `pip install llama-index-llms-bedrock-converse` 安装 Nova，并在此处查看 [示例](https://t.co/KYZRdIIihI)。
   - Nova 的发布以具有成本效益和高性能的选项扩展了 AI 模型版图。
- **有效的 RAG 实现**：一位成员分享了一个包含 **10 多个 RAG 实现** 的仓库，包括 **Naive RAG** 和 **Hyde RAG** 等方法，帮助他人为自己的数据集定制 RAG。在此处查看 [仓库](https://github.com/athina-ai/rag-cookbooks)。
   - 这些实现促进了针对特定 AI 开发需求定制的 RAG 应用实验。
- **嵌入模型 Token 尺寸限制**：讨论强调了 **HuggingFaceEmbedding** 类会截断超过 **512 tokens** 的输入文本，这给嵌入较长文本带来了挑战。
   - 成员们建议选择合适的 `embed_model` 类来有效绕过这些限制。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Qwen LV 7B Vision 功能**：有用户询问 **Qwen LV 7B 模型**是否支持 **vision**（视觉）功能，引发了关于将视觉能力集成到各种 AI 模型中的讨论。
   - 社区成员正在探索将视觉与 Qwen LV 7B 结合的潜力，讨论了可能的应用场景和技术要求。
- **FP8 量化提升模型效率**：根据 [VLLM 文档](https://docs.vllm.ai/en/v0.6.2/quantization/fp8.html)，FP8 量化可以在几乎不影响准确性的情况下，实现 **2 倍**的模型内存减少和 **1.6 倍**的吞吐量提升。
   - 这种优化对于在资源有限的机器上优化性能尤为重要。
- **HF Spaces 现在支持 Docker 容器**：一位成员确认任何 **HF space** 都可以作为 [docker container](https://link.to/docker) 运行，为本地部署和测试提供了灵活性。
   - 这一增强功能为在 HF spaces 上工作的 AI 工程师提供了更简便的集成和扩展性。
- **Intel Arc Battlemage 显卡在 AI 任务中面临质疑**：一位成员对新款 **Arc Battlemage 显卡**表示怀疑，认为它们不适合 AI 任务。
   - 另一位成员反驳称，尽管对于构建**本地推理服务器**来说具有性价比，但此类应用对 Intel 的依赖性仍存疑问。
- **LM Studio 在 Windows 上的性能问题**：用户报告称，与 Mac 相比，在 Windows 上运行 **LM Studio** 时速度较慢且输出异常，特别是在使用 **3.2 模型**时。
   - 建议的解决方案包括切换 `Flash Attention` 开关以及检查系统规格的兼容性。



---



## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Sierra AI 宣讲会与人才招募**：Sierra AI 将于 **太平洋时间 12 月 3 日上午 9 点**为开发者举办**专属宣讲会**，可通过 [YouTube 直播](https://youtube.com/live/-iWdjbkVgGQ?feature=share) 观看，届时参与者将探索 **Sierra 的 Agent OS** 和 **Agent SDK** 能力。
   - 在会议期间，**Sierra AI** 将讨论他们对**优秀开发者**的招募需求，并鼓励感兴趣的人士 [RSVP](https://lu.ma/agents-hackathon-sierra) 报名以锁定职业机会。
- **Dawn Song 的 AI 安全结课讲座**：**Dawn Song** 教授将于今日 **PST 下午 3:00** 发表题为《迈向构建安全可靠的 AI Agents 以及基于科学和证据的 AI 政策之路》的最后一场讲座，并在 [YouTube](https://www.youtube.com/live/QAgR4uQ15rc) 同步直播。
   - 她将探讨与 **LLM agents** 相关的重大风险，并提出基于科学的 AI 政策以有效缓解这些威胁。
- **LLM Agents 课程作业与 Mastery 等级**：参与者仍可通过填写 [报名表](https://docs.google.com/forms/d/e/1FAIpQLSeBoUgiNGyB8pdRZdLp076wpn4LkUxzEt9uviKZTbCSSv-aBA/viewform) 注册 **LLM Agents 学习课程**，并在 [课程网站](https://llmagents-learning.org/f24) 获取所有材料。
   - 虽然实验作业并非所有证书的强制要求，但要获得 **Mastery 等级** 必须完成全部三项作业，允许晚加入的学习者按需补课。
- **对 GPT-4 PII 泄露的担忧**：一位成员担心 **GPT-4** 可能会泄露个人身份信息（**PII**），并将其与 2006 年的 [AOL 搜索日志泄露](https://en.wikipedia.org/wiki/AOL_search_log_release) 事件相类比。
   - 他们强调，尽管 **AOL** 声称进行了匿名化处理，但泄露内容包含了来自超过 **650,000 名用户** 的 **2000 万条搜索查询**，且这些数据至今仍可在网上获取。
- **ReAct 范式实现方式的影响**：**ReAct 范式** 的有效性高度依赖于具体的实现细节，如 **prompt design**（提示词设计）和 **state management**（状态管理），成员指出 Benchmark 应该反映这些细节。
   - 讨论中将其与传统机器学习中的基础模型进行了对比，引发了关于不同实现方式如何因 AI 领域内**模糊的定义**而导致 Benchmark 性能巨大差异的讨论。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **开发分支重写以增强性能**：最新的开发分支已完全重写，使其**更轻量、更快速、更智能**，给用户留下了深刻印象。
   - 成员们对测试这个活跃分支感到兴奋，并被鼓励针对旧版本中缺失的功能提供反馈。
- **新的 `--serve` 选项支持 OpenAI 兼容服务器**：新的 `--serve` 选项引入了一个 **OpenAI 兼容的 REST 服务器**，1.0 版本排除了旧的 **LMC/web socket 协议**。
   - 此设置允许用户通过任何 OpenAI 兼容的客户端进行连接，从而直接在服务器设备上执行操作。
- **Anthropic 集成遇到 TypeError**：用户报告在将开发分支与 **Anthropic** 集成时遇到了 **TypeError**，具体表现为意外的关键字参数 ‘proxies’。
   - 建议用户分享完整的 traceback 以便调试，并向其提供了正确的安装命令示例。
- **请求社区测试以增强开发分支**：成员们请求社区参与测试，以改进频繁更新的**开发分支功能**。
   - 一位成员表示依赖 **LMC 进行通信**，并发现过渡到新设置既*令人恐惧又令人兴奋*。
- **LiveKit 实现远程设备连接**：O1 利用 **LiveKit** 连接设备，如 iPhone 和笔记本电脑，或运行服务器的 **Raspberry Pi**。
   - 此设置便于通过在其上运行的本地 **OpenInterpreter (OI)** 实例**远程访问**并控制机器。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Pydantic AI 与 DSLModel 集成**：[Pydantic AI](https://ai.pydantic.dev/) 的引入增强了与 **DSLModel** 的集成，为开发者创建了一个无缝框架。
   - 此集成利用了 **Pydantic**，该工具在 Python 的各种 Agent 框架和 LLM 库中被广泛使用。
- **DSPy 在 AWS Lambda 上的优化挑战**：一位成员正考虑为 **LangWatch** 客户在 **AWS Lambda** 上运行 **DSPy 优化**，但 **15 分钟的限制**带来了挑战。
   - *他们表示需要绕过这一时间限制的策略。*
- **推荐使用 ECS/Fargate 而非 Lambda**：另一位成员分享了他们的经验，建议由于**存储限制**，在 **Lambda** 上运行 **DSPy** 可能不可行。
   - 他们建议探索 **ECS/Fargate** 作为一种可能更可靠的解决方案。
- **Program Of Thought 弃用担忧**：一位成员询问 **Program Of Thought** 是否在 **v2.5** 之后走向**弃用/无积极支持**的道路。
   - 这表明社区对该程序未来的持续关注。
- **DSPy 中的 Agentic 和 RAG 示例**：一位成员询问了 DSPy 中的 Agentic 示例，即一个 signature 的输出被用作另一个 signature 的输入，特别是针对电子邮件撰写程序。
   - 另一位成员建议查看 [RAG 示例](https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb)，但随后澄清相关示例的位置可能在 [dspy.ai 网站](https://dspy.ai)上。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 中的图像生成功能**：一位用户对在 Torchtune 中添加**图像生成**功能表示兴奋，并引用了 [Pull Request #2098](https://github.com/pytorch/torchtune/pull/2098)。
   - 该 Pull Request 旨在整合增强平台能力的新功能。
- **Torchtune 中的 T5 集成**：根据 [Pull Request #2069](https://github.com/pytorch/torchtune/pull/2069) 的见解，讨论表明 **T5** 可能会被集成到 Torchtune 中。
   - 预计此次集成将使 **T5** 功能与即将推出的**图像生成**增强功能保持一致。
- **在 Torchtune 中微调 ImageGen 模型**：一位成员强调了在 Torchtune 中微调**图像生成模型**的潜力，并将其描述为一个有趣的项目。
   - 这一评论引发了轻松的回应，表明成员之间的熟悉程度各不相同。
- **CycleQD Recipe 分享**：一位成员分享了 [CycleQD recipe](https://sakana.ai/cycleqd/) 的链接，并将其描述为一个有趣的项目。

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **成员对即将举行的活动感到兴奋**：成员们表达了参加即将举行的活动的**兴奋**和兴趣，其中一位成员提到在那段时间计划访问**印度**。
   - 分享了“*噢太棒了。是的。我会去的。希望能见面！*”，突显了参与者的热情。
- **活动注册流程说明**：一位用户询问了参加活动的**注册流程**，引发了关于如何有效操作注册系统的讨论。
   - 参与者分享了简化参会者引导体验的策略，以确保注册流程顺畅。



---



## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **ADOPT 优化器加速 Axolotl**：团队已将最新的 **ADOPT 优化器**更新集成到 [Axolotl 代码库](https://github.com/axolotl-ai-cloud/axolotl/pull/2104)中，鼓励工程师们尝试这些增强功能。
   - 一位成员询问了在 Axolotl 中使用 ADOPT 优化器的**优势**，引发了关于性能提升的讨论。
- **ADOPT 优化器的 Beta 提升**：**ADOPT 优化器**现在支持在任何 beta 值下实现**最优收敛**，增强了在不同场景下的性能。
   - 成员们在讨论中探索了这一能力，强调了其在各种部署场景中优化性能的潜力。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **PR#7987 凭借稳定的基准测试取得成功**：jewnex 指出 **PR#7987** 在运行基准测试后值得发推文，显示这次**使用 beam 时没有 GPU 挂起** 🚀。
- **调整 uopgraph.py 中的线程组**：**learn-tinygrad** 频道的一位成员询问在 `uopgraph.py` 的图重写优化期间是否可以更改**线程组/网格大小**。
   - 讨论集中在这些大小是基于 **pm_lowerer** 中早期的搜索而**固定**的，还是可以在优化后进行调整。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **2024 年的 Bio-ML 革命**：**2024** 年标志着生物机器学习（**bio-ML**）的激增，取得了显著成就，如因结构生物学预测授予的**诺贝尔奖**，以及对蛋白质序列模型的大量投资。
   - 该领域充满了兴奋，尽管对需要解决的计算最优蛋白质序列建模曲线仍存在担忧。[Through a Glass Darkly | Markov Bio](https://www.markov.bio/research/mech-interp-path-to-e2e-biology) 讨论了通往端到端生物学的路径以及人类理解的作用。
- **为单细胞生物学引入 Gene Diffusion**：描述了一种名为 **Gene Diffusion** 的新模型，该模型利用在单细胞基因计数数据上训练的连续扩散 Transformer 来探索细胞功能状态。
   - 它采用自监督学习方法，从基因 Token 向量中预测干净、无噪声的嵌入，类似于文本生成图像模型中使用的技术。
- **寻求 Gene Diffusion 训练机制的澄清**：人们对 **Gene Diffusion** 模型的训练机制感到好奇，特别是其输入/输出关系以及它的预测目标。
   - 成员们表达了对模型复杂细节进行澄清的愿望，强调了在理解这些复杂概念时需要社区的帮助。



---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **12月日程活动**：[12月日程](https://discord.com/channels/1089876418936180786/1089876419926032396/1311366440997355654)中新增了三项成员活动，旨在提高社区参与度。
   - 这些活动旨在展示成员的项目并提升社区活跃度。
- **下一代 Llamafile 黑客松演示**：学生们明天将展示他们使用 [Llamafile 构建个性化 AI](https://discord.com/events/1089876418936180786/1313249962582212708) 的项目，重点关注社会公益。
   - 鼓励社区成员支持学生们的创新举措。
- **Web Applets 介绍**：<@823757327756427295> 将进行 [Web Applets 介绍](https://discord.com/events/1089876418936180786/1311466292753989672)，解释用于高级客户端应用程序的开放标准和 SDK。
   - 参与者可以在社区内自定义角色以接收更新。
- **Theia IDE 实操演示**：<@1131955800601002095> 将演示 [Theia IDE](https://discord.com/events/1089876418936180786/1311841242262540298)，这是一个开放的 AI 驱动开发环境。
   - 演示将展示 Theia 如何增强开发实践。
- **Llamafile 发布与安全赏金**：宣布了 [Llamafile 的新版本](https://discord.com/channels/1089876418936180786/1262961704602570832/1312634808785965066)，包含多项软件改进。
   - <@&1245781246550999141> 在第一个月发放了 **42 份赏金**，用于识别生成式 AI 中的漏洞。

---

**HuggingFace Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1313189317438341120)** (2 条消息): 

> `DeMo 优化器发布, Nous DisTrO, 去中心化预训练, 分布式训练研究` 

- **Nous 为 15B 模型启动去中心化预训练**：Nous 已启动一个 15B 参数语言模型的**去中心化预训练**，使用了 **Nous DisTrO** 以及来自 Oracle 和 Lambda Labs 等合作伙伴的硬件。此次运行展示的损失曲线与使用 **AdamW** 的传统中心化训练相当甚至更优。
   - 您可以在[此处](https://distro.nousresearch.com)观看实时运行情况，并查看即将发布的配套 [DeMo 论文和代码](https://arxiv.org/abs/2411.19870)。
- **DeMo 研究论文公开发布**：**DeMo** 优化器允许并行训练神经网络，同时在每个优化步骤中仅同步极少的模型状态。该方法在减少加速器间通信的同时提高了收敛性，支持在多种硬件配置下进行训练。
   - 详述这一创新方法的论文可以在[此处](https://arxiv.org/abs/2411.19870)找到，源代码已在 [GitHub](https://github.com/bloc97/DeMo) 上发布。
- **DisTrO 优化器的未来发布**：**DisTrO 优化器**基于 DeMo 的原理，但在发布前仍需进一步开发。DisTrO 的后续论文和代码将在准备就绪后发布在 [GitHub](https://github.com/NousResearch/DisTrO) 上。
   - 该优化器旨在改善去中心化训练体验，并为更多 AI 从业者提供这些工具。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://distro.nousresearch.com">Nous DisTrO</a>：互联网分布式训练</li><li><a href="https://arxiv.org/abs/2411.19870">DeMo: Decoupled Momentum Optimization</a>：训练大型神经网络通常需要通过专门的高速互连在加速器之间共享梯度。借鉴信号处理的频率分解原理...</li><li><a href="https://github.com/bloc97/DeMo">GitHub - bloc97/DeMo: DeMo: Decoupled Momentum Optimization</a>：DeMo：解耦动量优化。通过在 GitHub 上创建账户为 bloc97/DeMo 的开发做出贡献。</li><li><a href="https://github.com/NousResearch/DisTrO">GitHub - NousResearch/DisTrO: Distributed Training Over-The-Internet</a>：互联网分布式训练。通过在 GitHub 上创建账户为 NousResearch/DisTrO 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1313149067626414201)** (426 条消息🔥🔥🔥): 

> `DisTrO 训练更新，针对不同任务使用较小模型，AI 模型中的 Function Calling 与 MCP 对比，社区对 AI 训练的贡献，AI 开发中的就业机会` 


- **DisTrO 训练更新**：当前的 DisTrO 训练运行预计很快结束，关于硬件和用户贡献的具体细节将在本周末公布。
   - 此次运行主要是测试性质，可能不会立即向公众提供 Registry 或教程。
- **针对不同任务使用较小模型**：在某些创意任务中，较小模型的表现可能优于较大模型，并具有处理速度更快、资源占用更低等优势。
   - 建议使用较小模型进行状态管理，同时利用较大模型处理讲故事等繁重任务，从而实现平衡。
- **AI 模型中的 Function Calling 与 MCP 对比**：Function Calling 被视为管理 AI 状态和动作的一种方法，而 MCP 在实现复杂功能时可能提供不同的优势。
   - 关于 MCP 和 Function Calling 之间的区别存在一些困惑，强调了需要澄清它们各自的用途。
- **社区对 AI 训练的贡献**：讨论了利用社区资源进行去中心化训练的潜力，强调了大规模模型训练高效方法的重要性。
   - 参与者认识到同步和通信开销带来的挑战，特别是对于需要大量资源的较大模型。
- **AI 开发中的就业机会**：一位用户正在为一家 AI 公司提供就业机会，寻求招聘具有 AI 和 ML 经验的人员。
   - 此外，提到有用户询问 OpenRouter 上 Hermes 405B 模型的可用性，表明了对 AI 模型可访问性的兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forge.nousresearch.com/">Forge Reasoning API by Nous Research</a>：Nous Research 的 Forge Reasoning API</li><li><a href="https://distro.nousresearch.com/">Nous DisTrO</a>：互联网分布式训练</li><li><a href="https://arcprize.org/">ARC Prize</a>：ARC Prize 是一项奖金超过 1,000,000 美元的非营利性公开竞赛，旨在击败并开源针对 ARC-AGI 基准测试的解决方案。</li><li><a href="https://x.com/chriscyph/status/1863792734647320954">来自 chris (@chriscyph) 的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/1704.04289">Stochastic Gradient Descent as Approximate Bayesian Inference</a>：具有恒定学习率的随机梯度下降（恒定 SGD）模拟了具有平稳分布的马尔可夫链。基于这一视角，我们推导出了几个新结果。(1) 我们证明了……</li><li><a href="https://paperswithcode.com/dataset/arc">Papers with Code - ARC (AI2 Reasoning Challenge) Dataset</a>：AI2 的推理挑战 (ARC) 数据集是一个多项选择问答数据集，包含 3 到 9 年级的科学考试问题。该数据集分为两个分区：Ea...</li><li><a href="https://www.jetson-ai-lab.com/tutorial_llamaspeak.html#function-calling">
   llamaspeak - NVIDIA Jetson AI Lab
  </a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Exascale_computing">Exascale computing - Wikipedia</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=gzuYdUAPXxw">Elliott Smith - 13 - Independence Day</a>：Town Hall, New York, New YorkSetlistSon of SamHappinessBetween the BarsLARose ParadePretty Mary KAngelesNeedle in the HaySay YesWaltz #2St. Ide&#39;s HeavenEasy ...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1313172233354743840)** (3 条消息): 

> `技术社会主义，Nous Research，XCLR8` 


- **技术社会主义对 Nous 的兴趣**：一位成员以**技术社会主义者**的身份表达了对 **Nous** 的热情，表明了对该平台兴趣的一致性。
   - 这突显了关于技术如何与社会意识形态交汇的潜在对话。
- **提到 XCLR8**：一位成员简要提到了 **XCLR8**，引发了对其与当前讨论相关性的好奇。
   - 进一步的探索可能会揭示关于该话题影响或应用的见解。


  

---

### **Nous Research AI ▷ #[reasoning-tasks](https://discord.com/channels/1053877538025386074/1264666760972472481/1313539156831637626)** (4 messages): 

> `DisTro Issues, Flux Capacitor Reference, DeLorean Nostalgia` 


- **对 DisTro 问题的否认**：一位成员幽默地指出，有人拒绝承认 **DisTro** 持续存在的问题，并将其比作发明 **Flux Capacitor**。
   - 这一俏皮的评论凸显了对持久问题的挫败感，并引发了轻松的交流。
- **一致性优于逻辑**：另一位成员评论说，尽管存在问题，情况仍然保持着*逻辑和一致性*，虽然并不理想。
   - 这一表态反映了在分歧中一种无奈的接受，增进了讨论中的同袍情谊。
- **对 DeLorean 的向往**：一位成员表达了对 **DeLorean** 的幽默渴望，引用了其在流行文化中的标志性地位。
   - 这一愿望为正在进行的对话增添了怀旧色彩，强化了聊天中轻松愉快的基调。



**Link mentioned**: <a href="https://hermes.nousresearch.com)">no title found</a>: no description found

  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1313156542500962344)** (181 messages🔥🔥): 

> `Use of JAX vs. PyTorch, Vendor Lock-in Concerns, Performance Optimizations with Torch Compile, AI Lab Hiring Practices, Collaboration Between Universities and Tech Companies` 


- **JAX 在大型实验室中受到关注**：几位成员讨论了包括 Anthropic 和 DeepMind 在内的许多 AI 实验室据报道都在其模型中使用 **JAX**，尽管主导程度各不相同。
   - 然而，关于 JAX 相对于 **PyTorch** 占据主导地位的说法仍存争议，强调了行业实践中需要更好的透明度。
- **关于 Vendor Lock-in 的担忧**：对话强调了学术界对 **Vendor Lock-in** 的担忧，特别是当科技公司通过为特定框架提供资源来影响大学课程时。
   - 虽然有些人认为利用供应商合作伙伴关系是有益的，但其他人对学生更广泛技能掌握的影响持怀疑态度。
- **利用 Torch Compile 进行性能优化**：关于 **torch.compile** 的讨论显示，虽然它相对较新且面临挑战，但正确使用时可以实现显著的性能优化。
   - AI 实验室经常寻求 **PyTorch compiler 专家**，以提升已经采用 torch.compile 的模型性能。
- **AI 实验室的招聘**：一位成员指出，知名的 AI 实验室对招聘来自 **TensorFlow**、**PyTorch** 和 **JAX** 等框架的贡献者表现出兴趣，可能是为了其团队寻找熟练的开发人员。
   - 社区紧密的性质促进了人际网络，一些与会者回忆起过去与这些实验室关键人员的互动。
- **大学与科技公司之间的合作**：成员们反思了像 **Amazon** 和 **Google** 这样的科技巨头如何与大学合作塑造课程并提供资源（如 TPU 访问权限）。
   - 虽然这有助于学生学习尖端技术，但人们担心教育可能会偏向于特定公司。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://eric-xia.com/saeviz_survey.html">Random Redirect</a>: no description found</li><li><a href="https://www.macrumors.com/2024/12/03/apple-amazon-ai-chips-search/">Apple Uses Amazon's Custom AI Chips for Search Services</a>: Apple uses custom Inferentia and Graviton artificial intelligence chips from Amazon Web Services for search services, Apple machine learning and AI...</li><li><a href="https://news.ycombinator.com/item?id=39876444">JAX is used by almost every large genAI player (Anthropic, Cohere, DeepMind, Mid... | Hacker News</a>: no description found</li><li><a href="https://arxiv.org/abs/2407.21783">The Llama 3 Herd of Models</a>: Modern artificial intelligence (AI) systems are powered by foundation models. This paper presents a new set of foundation models, called Llama 3. It is a herd of language models that natively support ...</li><li><a href="https://github.com/apple/axlearn">GitHub - apple/axlearn: An Extensible Deep Learning Library</a>: An Extensible Deep Learning Library. Contribute to apple/axlearn development by creating an account on GitHub.</li><li><a href="https://github.com/stanford-cs149/asst4-trainium">GitHub - stanford-cs149/asst4-trainium</a>: Contribute to stanford-cs149/asst4-trainium development by creating an account on GitHub.
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1313148666315542538)** (151 messages🔥🔥): 

> `DeMo Optimizer, Differential Attention, Second Order Optimization, NAS in ML, Moving Sofa Problem`

- **DeMo 优化器减少同步开销**：**DeMo** 优化器引入了一种通过解耦动量更新来减少加速器间通信需求的方法，在无需高速网络完全同步的情况下实现了改进的收敛。
   - 其极简方法将优化器状态大小减少了**每个参数 4 字节**，被强调为大规模模型的一个显著优势。
- **关于 Differential Attention 的讨论**：参与者讨论了 **differential attention** 在 **Hymba** 和 **Striped Mamba** 等各种架构中的有效性，质疑它们在缓存大小方面的表现以及与传统模型相比的整体效能。
   - 有人担心仅关注最大梯度是否能在不同 epoch 之间产生有效结果。
- **二阶优化器的探索**：辩论了**二阶优化 (second order optimization)** 方法对大模型的可行性，结果显示虽然经验证据表明其具有更快收敛的潜力，但研究工作中的结果复现并不一致。
   - 同时也强调了计算复杂度和保持动量中某些自适应特性等挑战。
- **对 NAS 时代的担忧**：社区对 **神经网络架构搜索 (NAS)** 的趋势表示怀疑，参考了过去在计算机视觉领域几乎没有产生持久影响的经验，并质疑这是否预示着模型设计中的创意枯竭。
   - 尽管如此，人们对混合模型以及尽管对优化方法存在担忧但仍具创新潜力持乐观态度。
- **Moving Sofa Problem 及其影响**：**Moving Sofa Problem** 因一项声称的解决方案而引发了兴奋，将数学证明的讨论与优化挑战交织在一起，展示了理论与应用的交汇。
   - 参与者质疑了其对实际优化方法的启示，同时对这一数学挑战的进展表示关注。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2411.17800">STAR: Synthesis of Tailored Architectures</a>: 模型架构的迭代改进是深度学习的基础：Transformer 首先实现了扩展，而最近在模型混合化方面的进展推动了质量-效率的前沿……</li><li><a href="https://arxiv.org/abs/2411.19870">DeMo: Decoupled Momentum Optimization</a>: 训练大型神经网络通常需要通过专门的高速互连在加速器之间共享梯度。借鉴频率分解 (frequency decomp) 的信号处理原理……</li><li><a href="https://arxiv.org/abs/2411.19722">JetFormer: An Autoregressive Generative Model of Raw Images and Text</a>: 消除建模约束并统一跨领域架构一直是训练大型多模态模型近期进展的关键驱动力。然而，大多数这些模型仍然依赖于……</li><li><a href="https://en.wikipedia.org/wiki/Moving_sofa_problem">Moving sofa problem - Wikipedia</a>: 未找到描述</li><li><a href="https://x.com/LiquidAI_/status/1863701726659772617">Tweet from Liquid AI (@LiquidAI_)</a>: Liquid 的新研究：STAR —— 定制架构的进化合成。在 Liquid，我们设计基础模型有两个宏观目标：最大化质量和效率。平衡两者是具有挑战性的……</li><li><a href="https://proceedings.mlr.press/v139/wortsman21a.html">Learning Neural Network Subspaces</a>: 最近的观察推进了我们对神经网络优化景观的理解，揭示了 (1) 包含多样化解决方案的高精度路径的存在，以及 (2) 更宽的微小……</li><li><a href="https://arxiv.org/abs/2411.19826">Optimality of Gerver&#39;s Sofa</a>: 我们通过证明具有 18 个曲线段的 Gerver 构造达到了最大面积 $2.2195\cdots$，从而解决了 Moving Sofa Problem。</li><li><a href="https://arxiv.org/abs/2411.18674">Active Data Curation Effectively Distills Large-Scale Multimodal Models</a>: 知识蒸馏 (KD) 是将大规模模型压缩为较小模型的行业标准。之前的工作探索了涉及不同目标函数的日益复杂的 KD 策略……</li><li><a href="https://arxiv.org/abs/2410.12361">Proactive Agent: Shifting LLM Agents from Reactive Responses to Active Assistance</a>: 由 LLM 驱动的 Agent 在解决复杂任务方面表现出卓越的能力。然而，大多数 Agent 系统仍然是被动的，限制了它们在需要预见性的场景中的有效性……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1313197141010808832)** (1 messages): 

> `常用方法参考，综述与教科书的区别` 


- **发布了关于方法的广泛入门读物 (Primer)**：一名成员分享了一篇 [primer](https://arxiv.org/abs/2405.00208)，提供了广泛的参考集，并使用标准符号将常用方法背景化。
   - 该资源处于 **survey**（综述）和 **textbook**（教科书）之间的灰色地带，欢迎提供反馈。
- **征求对入门读物的反馈**：该成员专门征求对入门读物的反馈，旨在提高其对用户的实用性和清晰度。
   - 该资源模糊了 **survey** 和 **textbook** 之间的界限，强调了社区投入的必要性。


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1313158364770865215)** (13 messages🔥): 

> `VLLM 种子配置，QwQ 预览版排行榜状态，外部可加载评估，版本控制与可复现性担忧` 


- **VLLM 设置自己的种子 (seed)**：一名成员指出 **VLLM** 管理自己的种子，可以通过 `model_args` 传入。
   - 另一名成员讨论了 **Hugging Face** 可能依赖 **torch seed** 进行操作。
- **QwQ 预览版在排行榜上表现不佳**：有人强调 **QwQ preview** 已列入 Open LLM Leaderboard，但由于“思考 (thinking)”部分的解析问题导致得分较低。
   - 一名成员强调它需要生成更长的输出来获得更好的评估结果。
- **关于外部可加载评估 (evals) 的想法**：有人建议让 **evals** 能够像 Hugging Face 上的数据集和模型一样从外部加载。
   - 该成员指出，这可以促进数据集和相关评估 YAML 文件的更轻松加载。
- **对可见性和版本控制的担忧**：一名成员对使用外部仓库进行 **evals** 时的可见性和版本控制表示担忧。
   - 有人指出，尽管可能出现挑战，确保评估的 **reproducibility**（可复现性）至关重要。
- **数据集的版本控制问题**：**versioning**（版本控制）和 **reproducibility** 问题被认为是现有评估中使用的原始数据集的潜在担忧。
   - 一名成员评论说，虽然这是一个合理的担忧，但到目前为止还没有引起重大问题。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/f49b0377bf559f5558e8cd9ebd1190218c7df2a4/lm_eval/__main__.py#L197),">lm-evaluation-harness/lm_eval/__main__.py at f49b0377bf559f5558e8cd9ebd1190218c7df2a4 · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness

  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1313495582525882428)** (2 messages): 

> `日志配置，性能细分` 


- **特定日志消息的来源**：一名成员询问了提供优化器计时细分的详细日志消息的来源，包括 **optimizer_allgather** 和 **fwd_microstep** 等指标。
   - 该成员随后确认这些日志是由 **wall_clock_breakdown** 配置选项启用的。
- **配置设置的澄清**：讨论集中在导致系统中出现详细日志消息的特定配置选项上。
   - 确认 **wall_clock_breakdown** 选项对于生成这些详细日志起到了关键作用。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1313406361106911252)** (120 messages🔥🔥): 

> `Mojo 中的 Socket 通信、Mojo 的 SIMD 支持、网络 API 设计、高性能文件服务器实现、自定义分配器` 


- **Socket 通信功能推迟**：由于待处理的语言特性，Mojo 中增加 **socket 通信** 的计划有所推迟，目标是建立一个支持包括 **POSIX sockets** 在内的可交换网络后端的标准库。
   - 目前的计划涉及一次大规模重构，以确保在这些特性就绪时能够正确实现。
- **Mojo 中令人兴奋的 SIMD 能力**：成员们讨论了 Mojo **SIMD 支持** 的优势，其中一位指出，与混乱的 C/C++ intrinsics 相比，它简化了 SIMD 编程。
   - 目标是在未来将更多的 intrinsics 映射到标准库中，以减少直接使用的需要。
- **高性能文件服务器项目**：一位成员提到正在为一个开发中的游戏编写 **高性能文件服务器** 项目，目标是每秒处理的数据包比 **Nginx** 高出 30%。
   - 在上述特性可用之前，他们目前利用外部调用来进行网络通信。
- **ASP 与汇编语言讨论**：对话涉及了 **inline assembly** 的使用，以及如何根据架构移植各种操作，强调了编译器支持的作用。
   - 有人建议撰写一篇“Mojo 中的 SIMD”博客文章，以帮助理解和利用这些高级特性。
- **利用自定义分配器提升性能**：参与者讨论了使用 **arena allocators** 的优势以及它们如何提高效率，特别是对于使用 continuations 的例程。
   - 讨论揭示了在性能优化与保持不同实现之间的可移植性之间进行平衡的必要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://mzaks.medium.com/counting-chars-with-simd-in-mojo-140ee730bd4d">Counting chars with SIMD in Mojo</a>: Mojo 是一门非常年轻（实际上仍在开发中）的编程语言，由一家名为 Modular 的新公司设计和开发。这里有一个……</li><li><a href="https://github.com/marti">marti - Overview</a>: GitHub 是 marti 构建软件的地方。</li><li><a href="https://godbolt.org/z/E3381jM43">Compiler Explorer - C (x86-64 clang (trunk))</a>: /* Type your code here, or load an example. */void square(__m128i a, __m128i b, __mmask8* k1, __mmask8* k2) {    _mm_2intersect_epi32(a, b, k1, k2);}</li><li><a href="https://github.com/intel/hyperscan">GitHub - intel/hyperscan: High-performance regular expression matching library</a>: 高性能正则表达式匹配库 - intel/hyperscan</li><li><a href="https://github.com/martinvu">MartinVu - Overview</a>: MartinVu 有 5 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/s">s - Overview</a>: s 有 49 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1313213733602066453)** (1 messages): 

> `Magic 包分发、早期访问预览、社区测试、功能迭代` 


- **社区获得 Magic 包分发功能**：一个允许社区成员通过 **Magic** 分发包的激动人心的功能正在开发中，早期访问预览即将推出。
   - 他们正在寻找测试人员来帮助完善该功能，邀请有兴趣的成员通过回复 **🔍**（进行包审查）或 **🧪**（进行安装）来参与。
- **呼吁社区参与**：团队正在寻找能够投入时间在正式发布前对新功能进行测试和迭代的奉献者。
   - 鼓励感兴趣的参与者通过相应的 emoji 表达参与意愿，展现社区参与的热情。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1313244044536582235)** (133 条消息🔥🔥): 

> `Mojo 开发见解、Inline References 概念、Reference Trait 提案、Mojo 当前的 Python 支持、编译结构更新` 


- **探索 Mojo 开发见解**：成员们讨论了在 Mojo 中将 `Int` 转换为 SIMD DTypes 的细微差别，指出需要明确语法和函数用法。
   - 大家公认 Mojo 对引用（references）和解引用（dereferencing）的处理是一个主要的困惑点，特别是关于 inline references。
- **Inline References 的概念**：Inline references 被提议作为 Mojo 的一个潜在特性，用于改进内存访问模式并安全地处理可变数据。
   - 讨论内容包括 inline references 对可变性的影响，以及有效管理指针地址空间的必要性。
- **Mojo 中 Reference Trait 的提案**：提出了一个 `Reference` trait 的提案，旨在增强 Mojo 代码中可变和可读引用的管理。
   - 这种方法将允许更好的 borrow-checking，并有可能减少函数参数中关于可变性的混淆。
- **Mojo 当前的 Python 支持**：成员们获悉目前尚不支持 Python 3.13，建议使用 3.8-3.12 版本以确保 Mojo 兼容性。
   - 进一步澄清了在使用 MAX engine 以及受支持的 PyTorch 版本时的系统要求。
- **Mojo 编译结构的更新**：讨论涉及了 Mojo 过去的 autotuning 系统，以及由于编译结构重构而将其移除的情况。
   - 有人提出了关于利用当前的编译阶段实现调优过程自动化的可能性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/max/faq#distribution">MAX FAQ | Modular Docs</a>: 关于 MAX Engine 预期问题的解答。</li><li><a href="https://docs.modular.com/mojo/manual/parameters/#automatic-parameterization-of-functions">参数化：编译时元编程 | Modular Docs</a>: 参数和编译时元编程的介绍。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1313128927048241173)** (154 条消息🔥🔥): 

> `OpenRouter 性能问题，Aider 新功能，Amazon Foundation Models 发布，Aider 用户体验，Repo-map 故障排除` 


- **OpenRouter 基准测试结果分析**：据透露，通过 OpenRouter 运行的模型基准测试结果比通过直接 API 运行的结果更差，引发了用户对性能差异的讨论。
   - 用户正在研究提高 OpenRouter 性能的解决方案，表明正在共同努力解决已确定的问题。
- **Aider 新功能提升用户体验**：多位用户分享了对 Aider 新功能 `--watch-files` 的积极体验，该功能简化了将 AI 指令集成到编码工作流的过程。
   - 用户赞赏 Aider 的透明度，提到了 `/save`、`/add` 等功能以及在使用过程中修改上下文的能力，这有助于获得更明晰的编程体验。
- **Amazon 发布全新 Foundation Models**：Amazon 在 re:Invent 活动中宣布发布六款全新的 Foundation Models，强调了它们的多模态能力和极具竞争力的价格。
   - Micro、Lite 和 Canvas 等模型将通过 Amazon Bedrock 独家提供，其价格结构与美国其他前沿模型相比，对用户非常有吸引力。
- **用户对 Aider 效率的见解**：用户发现 Aider 与其他编码辅助工具相比更具“思考性”，使他们能够有策略地提示 AI 以获得更定制化的响应。
   - 这种情绪反映了 Aider 在提高编码效率和用户满意度方面的有效性，用户正从其他平台迁移过来。
- **Repo-map 问题排查**：一位用户报告了 repo-map 功能的问题，指出可能意外删除了文件，从而询问如何恢复配置。
   - 社区成员建议重新建立 repo-map 配置，并分享了更新过程中常见错误的经验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/_philschmid/status/1864016010464080260">Philipp Schmid (@_philschmid) 的推文</a>: 出乎意料。@amazon 带着 Foundation Models 回来了。作为 re:Invent 的一部分，他们宣布了 6 款全新的 Foundation Models，涵盖从纯文本到文本转视频！👀 Nova 模型将通过 Am... 独家提供</li><li><a href="https://x.com/cohere/status/1863586939288367386">cohere (@cohere) 的推文</a>: 介绍我们最新的 AI 搜索模型：Rerank 3.5！Rerank 3.5 提供最先进的性能，具有改进的推理和多语言能力，可精确搜索复杂的企业数据...</li><li><a href="https://github.com/yoheinakajima/babyagi-2o">GitHub - yoheinakajima/babyagi-2o: 最简单的自构建通用自主 Agent</a>: 最简单的自构建通用自主 Agent - yoheinakajima/babyagi-2o</li><li><a href="https://github.com/codingthefuturewithai/software-dev-prompt-library/blob/main/docs/guides/getting-started.md#using-the-workflows.">software-dev-prompt-library/docs/guides/getting-started.md at main · codingthefuturewithai/software-dev-prompt-library</a>: 包含经过测试的可重用生成式 AI 提示词库，用于常见的软件工程任务 - codingthefuturewithai/software-dev-prompt-library</li><li><a href="https://youtube.com/@codingthefuture-jg1he?si=mjqG_DrpgMJcYG8C">Coding the Future With AI</a>: 欢迎来到 Coding the Future With AI！我们的频道致力于帮助开发者和技术爱好者学习如何利用 AI 提高技能和生产力。通过教程、专家访谈...</li><li><a href="https://aider.chat/docs/config/options.html#--gitignore">选项参考</a>: 关于 Aider 所有设置的详细信息。</li><li><a href="https://youtu.be/t-i2x3APvGQ?si=pAp8W8-as258a-Sg">通过工作流驱动、调优的提示链解锁 AI 编码 🔑</a>: 在本教程中，我们将深入探讨一种使用 AI 构建软件的系统化方法，向您介绍一个由高度调优的提示链驱动的工作流系统...
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1313150465487147109)** (82 条消息🔥🔥): 

> `在 Docker 中使用 Aider、更新 Aider、函数重构挑战、Aider 中的上下文管理、为 Aider 抓取文档` 


- **在 Docker 中运行 Aider 需要权限**：一位用户提到在运行 Aider Docker 容器时遇到权限问题，尝试与另一个容器共享卷，两者都设置为 UID:GID 1000:1000。
   - 当 Aider 尝试写入特定文件时，他们遇到了“Permission denied”错误，需要进一步调查。
- **更新 Aider 版本的问题**：一位用户在将 Aider 更新到 0.66.0 版本时遇到困难，并报告了在使用包安装工具时的各种命令失败。
   - 经过排查，他们发现通过在安装过程中专门调用 Python 3.12 解释器成功解决了问题。
- **跨文件重构函数**：一位用户寻求帮助，希望使用 Aider 在重构期间定位函数在跨文件中的调用位置，但得知 Aider 无法直接执行此任务。
   - 建议他们在涉及 Aider 之前，利用 IDE 功能或外部工具进行代码分析。
- **为 Aider 的上下文收集文档**：用户讨论了为 Aider 抓取外部文档的方法，表达了对自动生成参考文件解决方案的期望。
   - 一些人建议创建一个本地目录进行手动文档收集，或使用 Docker 工具进行抓取。
- **使用 MCP 进行上下文管理**：几位用户探索了使用 Model Context Protocol (MCP) 来增强 Aider 的上下文能力，特别是在代码相关的场景中。
   - 他们讨论了使用像 IndyDevDan 的 agent 和 Crawl4AI 这样的工具来为 LLM 集成创建优化的文档。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.14405">Marco-o1: Towards Open Reasoning Models for Open-Ended Solutions</a>：目前 OpenAI o1 引发了对大语言推理模型 (LRM) 研究的热潮。借此势头，Marco-o1 不仅关注具有标准答案的学科，如数学...</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/install/docker.html">Aider with docker</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/usage/modes.html">Chat modes</a>：使用 code、architect、ask 和 help 聊天模式。</li><li><a href="https://aider.chat/docs/install/pipx.html">Install with pipx</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://pastebin.com/zhYR4FcF">Aider v0.66.0Model: ollama_chat/qwen2.5-coder:14b with whole edit formatGit - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。</li><li><a href="https://m.youtube.com/watch?v=tElgVPUargw">AI Coding with Aider Architect, Cursor and AI Agents. (Plans for o1 BASED engineering)</a>：🔥 AI 代码编辑器战争已经打响！你的编程工作流为 o1 发布做好准备了吗？不要被淘汰！🚀🔥🔗 资源 - 💻 Computer Use Bash &amp; ...</li><li><a href="https://github.com/Aider-AI/aider/blob/main/.github/workflows/release.yml">aider/.github/workflows/release.yml at main · Aider-AI/aider</a>：aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 做出贡献。</li><li><a href="https://youtu.be/9mciRwpcLNY?si=IqPQDJ-lgBlYGUre)">Anthropic MCP with Ollama, No Claude? Watch This!</a>：Anthropic 发布了 Model Context Protocol，允许你将 LLM 连接到你自己的数据和工具。在这个视频中，Chris 展示了如何将 MCP 与 C... 分离。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/)** (1 条消息): 

pierrunoyt: https://supabase.com/blog/supabase-ai-assistant-v2 不错的东西
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1313128044814143538)** (213 条消息🔥🔥): 

> `Cursor 延迟问题，Windsurf 与 Cursor 性能对比，Agent 功能与限制，语法高亮问题，更新后 Chat 功能问题` 


- **Dev Server 中的 Cursor 延迟问题**：多位用户报告称 **Cursor** 存在延迟，特别是在中大型项目中使用 **Next.js** 进行开发时，需要频繁执行 'Reload Window' 命令。
   - 配备 **16GB** RAM 的用户注意到与 **32GB** 用户相比有明显的延迟，这引发了对性能一致性的质疑。
- **Windsurf 与 Cursor 的可靠性对比**：由于最新的 **Cursor** 更新存在持续问题（经常无效地重复相同的修复），一些用户已转回使用 **Windsurf**。
   - 有用户指出 **Windsurf** 的 **Agent** 在编辑多个文件时不会丢失注释，而这是目前 **Cursor** 难以处理的功能。
- **Agent 功能与限制**：对 **Cursor Agent** 中 @web 功能的需求凸显了其目前的局限性，用户希望改进对实时信息的获取。
   - 一位用户注意到，编辑文件时的更改未能被 **Agent** 正确识别，导致对其可靠性感到沮丧。
- **语法高亮问题**：初次使用者对 **Cursor** 的**语法高亮**表示不满，称其会导致视觉不适。
   - 有投诉称许多 **VS Code addons** 在 **Cursor** 中无法正常工作，阻碍了用户体验。
- **更新后 Chat 功能问题**：更新后，用户面临 **Chat 功能** 问题，指出幻觉（hallucinations）和模型性能不一致是主要问题。
   - 用户评论认为模型质量有所下降，使得任务变得更具挑战性且令人沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://medium.com">Medium: Read and write stories.</a>: 在 Medium 上，任何人都可以向世界分享深刻的见解、有用的知识和人生智慧。</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">How to do `Fix in Composer` and `Fix in Chat` actions from keyboard</a>: 这两个：我在设置中找不到。</li><li><a href="https://medium.com/@NFAblog/connect-github-codespaces-to-cursor-ai-ai-friendly-vs-code-clone-243fa5f79414">Connect Github CodeSpaces to Cursor Ai (Ai friendly vs code clone)</a>: 将 GitHub Codespaces 连接到 CURSOR.DEV：开发者指南</li><li><a href="https://forum.cursor.com/t/infinite-loading-issue-when-saving-files-in-cursor-app/26328">Infinite Loading Issue When Saving Files in Cursor App</a>: 大家好，我在使用 Cursor App 时遇到了一个持续存在的问题。每当我保存文件时，应用就会卡在无限加载状态。以下是我收到的错误信息：Error Messag...</li><li><a href="https://github.com/getcursor/cursor/issues/2027">WSL extension is supported only in Microsoft versions of VS Code · Issue #2027 · getcursor/cursor</a>: 如果可以，请附上问题的截图。请包含您的操作系统名称。如果可以，复现步骤非常有帮助。我正在使用 Windows 11 + WSL: Ubu... 开发
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1313130820814377100)** (188 条消息🔥🔥): 

> `Perplexity Pro 订阅问题、性能与速度问题、图像生成能力、AI 模型对比、Amazon Nova 基础模型` 


- **Perplexity Pro 订阅问题**：用户报告了 Perplexity Pro 订阅的各种问题，包括功能访问障碍和意外的价格变动。
   - 部分用户遇到了账号封禁，需要支持来解决黑客入侵事件，而其他用户则在咨询可用的折扣。
- **Perplexity AI 的性能下降**：多名用户在使用 Perplexity AI 功能时遇到了持续的变慢和无限加载问题，这被认为是一个潜在的扩展性（scaling）问题。
   - 这一问题在其他平台上也有所体现，促使用户考虑转向 API 服务以获得更稳定的体验。
- **图像生成工具的探索**：围绕图像生成工具展开了讨论，用户分享了产生意想不到且通常极具创意的结果的 Prompt。
   - 用户尝试使用量子主题的 Prompt 来生成独特的视觉输出，突显了图像生成模型的多样化应用。
- **AI 模型与功能对比**：关于 AI 模型的讨论包括对 Amazon Nova 与 ChatGPT 和 Claude 等现有平台相比的有效性见解。
   - 用户对各种基础模型在特定任务中的表现及其与 Perplexity 等工具的集成表示了兴趣。
- **Google Gemini 与 Drive 集成的挑战**：一位用户对 Google Gemini 访问 Google Drive 文档时的不稳定性表示沮丧，质疑其可靠性。
   - 人们对高级功能是否仅限于付费版本表示担忧，用户正在寻求实际的演示。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sse-next-one.vercel.app/">Server Sent Events</a>: 未找到描述</li><li><a href="https://docs.perplexity.ai/faq/faq">no title found</a>: 未找到描述</li><li><a href="https://x.com/apostraphi/status/1863641627627049066?">来自 Phi Hoang (@apostraphi) 的推文</a>: 如果你一直在犹豫是否升级到 Perplexity Pro，今天只需 5 美元即可获得第一个月。</li><li><a href="https://x.com/apostraphi/status/1863641627627049066?s=46">来自 Phi Hoang (@apostraphi) 的推文</a>: 如果你一直在犹豫是否升级到 Perplexity Pro，今天只需 5 美元即可获得第一个月。</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">介绍 Amazon Nova，我们的新一代基础模型</a>: 来自 Amazon 的全新 SOTA 基础模型提供了前沿的智能和行业领先的性价比。</li><li><a href="https://www.youtube.com/watch?v=APO7WHP8Ozw">实时 AI 搜索之战：ChatGPT Search vs. Perplexity vs. Google vs. Copilot vs. Grok</a>: AI 正在接管搜索。 🤖无论你喜欢还是讨厌，由 LLM 驱动的搜索正在进入你的设备。 ↳ ChatGPT Search 及其 Chrome 扩展程序。 ↳ Go...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1313131606567030876)** (12 条消息🔥): 

> `Pending Searches, Partition in Linux, Parameter Counts, Purchasing Items, Vesuvius Challenge Progress` 


- **探索 Pending Searches**：几位成员正在 [Perplexity AI](https://www.perplexity.ai/search/new?q=pending&newFrontendContextUUID=3b044394-2c45-4370-bf39-2285aa21d6be&focus=internet) 上研究 **pending search** 查询。针对同一主题分享了一个重复链接，其中包含各种查询的追踪链接。
   - 对 pending searches 的紧迫性表明用户对该功能的持续参与和好奇。
- **理解 Linux 中的 Partition**：一位成员分享了一个深入探讨 Linux 中 **partition** 主题的链接，可在此处[访问](https://www.perplexity.ai/search/what-is-partition-in-linux-bgucbm.YSwKwbq3nUcNDuw?login-source=floatingSignup)。这旨在为用户澄清基础的 Linux 概念。
   - 根据用户请求，可能详细讨论了磁盘管理和文件系统布局的问题。
- **深入研究参数数量 (Parameter Counts)**：出现了另一个关于特定模型使用的 **parameters** 数量的查询，见[此处](https://www.perplexity.ai/search/how-many-parameters-was-traine-Uz.QsFyLTY.zRixuKo8ROw)。这反映了用户对模型容量和设置的兴趣。
   - 此类问题突显了对 AI 模型配置及其性能指标的关注。
- **在哪里购买物品**：一位寻求帮助的用户询问在哪里可以购买特定物品，并链接到了一个购买查询，见[此处](https://www.perplexity.ai/search/where-can-i-buy-a-167000820489-0e2NiMecSrqjFTjSR5oq_w)。这表明社区对技术或相关产品的实际采购需求。
   - 分享的搜索链接展示了一种高效查找购买信息的直接方法。
- **Vesuvius Challenge 进展见解**：讨论转向了 **Vesuvius Challenge**，一位成员分享了记录当前进展的链接，见[此处](https://www.perplexity.ai/search/vesuvius-challenge-progress-9.0MWu2STT2i0elbsXvYqQ)。这表明了对该挑战赛结果和进展的持续热情。
   - 这种兴趣反映了用户对竞赛和集体项目的参与，可能促进了社区协作。



**提到的链接**：<a href="https://www.youtube.com/embed/RK3fdaJbtyU">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1313186518902374401)** (7 条消息): 

> `API Error Responses, Content Citation Issues` 


- **间歇性 API 错误困扰用户**：用户报告收到诸如 `An error occurred while making the request: unable to complete request` 的响应，尤其是在过去几小时内。
   - 一位用户提到，同样的请求有时能成功执行，这让问题的性质变得不确定。
- **针对无响应 API 的变通方法**：一位用户分享了一个临时修复方案，通过在用户 prompt 中添加前缀，目前似乎减轻了错误发生的频率。
   - 这表明在等待底层 API 问题解决期间，用户可能需要调整他们的请求。
- **尽管有用户输入，引用依然存在**：一位成员对 API 仍然在括号中生成内容引用表示沮丧，尽管使用了“不要在括号中生成内容引用”的指令。
   - 另一位用户指出，引用是自动添加的，目前没有停用此功能的选项。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1313132891806634034)** (115 条消息🔥🔥): 

> `LoRA 微调流程, 模型兼容性问题, 训练 Llama 3.2, xformers 安装问题, 微调模型的推理与分词` 


- **LoRA 微调流程困惑**：用户在微调 Llama 3.2 时遇到困难，特别是关于从 Tokenization 到 Processor 管理的过渡。
   - 建议对原始 Colab notebook 进行修改，以成功执行微调过程。
- **模型兼容性问题**：几位用户遇到了 xformers 与其当前 PyTorch 和 CUDA 版本的兼容性问题，导致出现错误信息。
   - 提供了重新安装正确版本 xformers 的建议以解决此类问题。
- **在低性能硬件上训练 Llama 3.2**：用户指出在 3090 等硬件上尝试训练较长样本时遇到了显存溢出 (OOM) 错误。
   - 切换到低资源设置或类似 marco O1 的配置是用户考虑的替代方案。
- **微调模型的推理与分词**：一位用户询问他们微调后的 LoRA 模型是否可以像其他 HF 模型一样正常加载和使用。
   - 澄清指出，如果模型合并正确，用户不需要依赖 `FastVisionModel.from_pretrained(...)` 来使用其微调版本。
- **微调作业帮助**：有关如何使用微调模型产生预期结果以完成作业的咨询。
   - 社区回复建议检查模型合并过程或使用现有教程来实现所需的输出。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://]">未找到标题</a>: 未找到描述</li><li><a href="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh">未找到标题</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">视觉微调 | Unsloth 文档</a>: 使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2Q">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNx">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing.">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://huggingface.co/models?search=accelerator">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1whHb54GNZMrNxIsi2wm2EY_-Pvo2QyKh?usp=sharing#scrollTo=MKX_XKs_BNZR">Google Colab</a>: 未找到描述</li><li><a href="https://embeddedllm.com/blog/vllm-now-supports-running-gguf-on-amd-radeon-gpu">vLLM 现已支持在 AMD Radeon GPU 上运行 GGUF</a>: 本指南展示了 Liger-Kernels 训练内核在 AMD MI300X 上的影响。该构建已针对 ROCm 6.2 进行了验证。</li><li><a href="https://docs.vllm.ai/en/v0.5.5/models/spec_decode.html">vLLM 中的投机采样 (Speculative decoding) &#8212; vLLM</a>: 未找到描述</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>: 关于对齐 smol 模型的课程。通过在 GitHub 上创建一个账户来为 huggingface/smol-course 做出贡献。</li><li><a href="https://github.com/facebookresearch/xformers#installing-xformers)">GitHub - facebookresearch/xformers: 可黑客化且优化的 Transformers 构建模块，支持组合式构建。</a>: 可黑客化且优化的 Transformers 构建模块，支持组合式构建。 - facebookresearch/xformers</li><li><a href="https://github.com/foundation-model-stack/fms-fsdp">GitHub - foundation-model-stack/fms-fsdp: 🚀 利用原生 PyTorch 特性高效地（预）训练基础模型，包括用于训练的 FSDP 和 Flash attention v2 的 SDPA 实现。</a>: 🚀 利用原生 PyTorch 特性高效地（预）训练基础模型，包括用于训练的 FSDP 和 Flash attention v2 的 SDPA 实现。 - foundation-model-stack/fms-fsdp</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 微调 Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80%</a>: 微调 Llama 3.2, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1313468652951048192)** (4 messages): 

> `Claude for Coding, Continued Pretraining, Citation of Founders, Understanding Numerical Data, Accounting Domain Tokens` 


- **Claude 在编程方面表现出色，但下一步是什么？**: 有人好奇 **Claude** 是否能处理 **数值表格数据理解**，并强调了它目前主要专注于编程任务。
   - 分享了一张图片来阐述关于 Claude 能力的这一观点。
- **持续预训练 (Continued Pretraining) 的重要性**: 一位成员强调，**持续预训练 (CPT)** 对于 **Llama-3** 和 **Mistral** 等语言模型适应新领域至关重要，特别是当初始数据集缺乏语言多样性或专业领域数据时。
   - 他们指出，**CPT** 能够帮助这些模型学习特定领域（如 **会计**）的新 Token。
- **如何引用 Unsloth 的创始人**: 一位成员询问如何引用 Unsloth 的联合创始人 **Daniel Han** 和 **Michael Han**，寻求一种可接受的格式。
   - 另一位成员提供了一个引用示例，用于参考他们在 GitHub 上发布的关于语言模型微调的工作：[Unsloth AI](https://github.com/unslothai/unsloth)。
- **多语言模型的挑战**: 讨论涉及了基础模型在有效理解各种语言或法律、医学等专业文本领域时面临的挑战。
   - 观点认为 **CPT** 是解决这些知识差距的基础。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1313134995077988486)** (48 messages🔥): 

> `Unsloth Model Issues, Fine-tuning Challenges, Llama-3 Model Conversion to GGUF, Partially Trainable Embeddings, Model Sequence Length Concerns` 


- **Unsloth 模型在 GGUF 转换中的问题**: 用户在尝试将模型保存为 GGUF 格式时遇到问题，经常出现指示缺少 'llama.cpp/llama-quantize' 等文件的运行时错误。
   - 另一位用户确认重启 Colab 无法解决该问题，这引发了对底层库近期更改的担忧。
- **微调过程中的挑战**: 一位用户提到他们是微调新手并遇到了困难，例如模型在训练期间丢失了 Prompt 之间的上下文。
   - 用户寻求关于这种“遗忘”是否符合预期的建议，强调了对跨微调会话状态管理的关注。
- **关于 Llama 模型序列长度的讨论**: 有人询问如何在 Llama 模型的训练代码中正确设置最大序列长度 (max sequence length)，用户确认这是一个模型配置问题。
   - 一位参与者指出 Tokenizer 不应影响序列长度，并强调训练配置才是关键。
- **探索部分可训练的 Embedding**: 在关于创建部分可训练 Embedding 的讨论中，一位用户详细说明了他们在自定义实现中遇到的挑战，并注意到训练期间没有调用 forward 函数。
   - 反馈建议模型可能直接访问了权重，而不是调用修改后的 Head，这表明需要更深层次的集成。
- **关于 Mistral 和 Llama 架构的不确定性**: 展开了一场关于 Mistral 和 Llama 模型差异的对话，有人声称 Mistral 使用了超越 Llama 架构的增强特性。
   - 针对配置文件中的模型类型分类提出了疑问，需要社区进一步澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing#scrollTo=IqM-T1RTzY6C">Google Colab</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/phi3?utm_source=ainews&utm_medium=email&utm_campaign=ainews-to-be-named-4285.">使用 Unsloth 微调 Phi-3</a>: 通过 Unsloth 轻松微调微软的新模型 Phi 3 medium, small &amp; mini，并获得 6 倍长的上下文长度！</li><li><a href="https://huggingface.co/unsloth/Phi-3.5-mini-instruct/blob/main/config.json">config.json · unsloth/Phi-3.5-mini-instruct at main</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1313411544838832158)** (2 messages): 

> `QWen2 VL 7B finetuning, LLaVA-CoT dataset, Hugging Face model card` 


- **使用 LLaVA-CoT 微调 QWen2 VL 7B**：一位成员使用 [LLaVA-CoT 数据集](https://huggingface.co/forcemultiplier/Qwen2-VL-7B-Instruct-LLaVA-CoT-2000steps-r16a16-merged) 微调了 **QWen2 VL 7B**，并发布了训练脚本和数据集。
   - 该模型包含 **8.29B 参数**，使用 **BF16** 张量类型，脚本可在[此处](https://huggingface.co/datasets/forcemultiplier/LLaVA-CoT-30k-jsonl-trainkit)获取。
- **Hugging Face 模型卡片**：尚未为该微调模型创建模型卡片，但发起了[贡献模型卡片](https://discuss.huggingface.co/c/model-cards/14)的号召。
   - 这被强调为提高 Hugging Face 社区文档完善度和可用性的必要步骤。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/forcemultiplier/Qwen2-VL-7B-Instruct-LLaVA-CoT-2000steps-r16a16-merged">forcemultiplier/Qwen2-VL-7B-Instruct-LLaVA-CoT-2000steps-r16a16-merged · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/forcemultiplier/LLaVA-CoT-30k-jsonl-trainkit">forcemultiplier/LLaVA-CoT-30k-jsonl-trainkit · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1313135340676190250)** (9 messages🔥): 

> `PhD students' working conditions, 996 work culture, Work output and relaxation, Tokenizers in research` 


- **博士生在工作间隙午睡**：讨论中提到中国的博士生工作非常努力，甚至在[办公室准备了枕头](https://example.com)以便在长时间工作期间小睡。
   - 一位成员指出，*“有人给他们送午餐”*，突显了研究工作的极端环境。
- **加班文化：996 的困境**：一位成员介绍了 **996** 工作文化的概念，将其描述为员工从 **上午 9 点工作到晚上 9 点**，每周工作六天。
   - 另一位成员插话道，像 996 这样的做法 *“从长远来看并不高效”*。
- **放松以提高生产力**：一位参与者断言，为员工提供 **豆袋沙发** 和床位可以让员工在放松的状态下工作，从而带来更高的生产力。
   - 这一观点得到了其他人的支持，大家一致认为照顾好员工非常重要。
- **研究中对 Tokenizers 的关注不足**：有人提问是否有人在研究中关注不同的 **tokenizers**，这表明讨论中可能存在疏漏。
   - 这反映了社区内日益增长的担忧，即某些工具在研究讨论中没有得到足够的重视。


  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1313204577419526145)** (1 messages): 

> `Raiza's departure from Google, NotebookLM team achievements, New venture announcement` 


- **Raiza 告别 Google**：今天是 Raiza 在 **Google** 工作 **5.5 年**后的最后一天，期间他领导了 **NotebookLM 团队**并培养了一个充满活力的社区。
   - *“这是我职业生涯的高光时刻之一，”* Raiza 分享道，并对这段经历和团队的支持表示感谢。
- **NotebookLM 的影响力之旅**：Raiza 回顾了从 **想法到原型** 再到深受数百万人喜爱的产品的历程，强调了团队的奉献和努力。
   - Raiza 对未来充满信心，表示：*“NotebookLM 的未来大有可为。”*
- **创办新公司**：Raiza 宣布计划与 **NotebookLM 团队**的其他两名成员共同创办一家新公司，并邀请感兴趣的人在 [werebuilding.ai](https://werebuilding.ai/) 注册以获取更新。
   - 他们鼓励合作，并表示：*“如果你有兴趣和我们一起构建，请通过 hello@raiza.ai 联系我。”*



**提及的链接**: <a href="https://werebuilding.ai/">We're Building</a>: 未找到描述

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1313152049806049291)** (28 条消息🔥): 

> `Notebook LM 用于脚本编写、PDF 处理的 OCR 工具、Podcast 创作、使用 AI 进行小说写作、AI 的多语言能力` 


- **使用 Notebook LM 进行脚本编写**：一位用户分享了他们使用 Notebook LM 为一部短片开发脚本的经验，并在音频剪辑中详细说明了摄像机和灯光设置。
   - 他们对将此脚本集成到视频项目中感到非常兴奋。
- **用于文档管理的 OCR 工具**：成员们讨论了 PDF24 在对扫描文档应用 OCR 方面的效用，通过创建可搜索的 PDF 来增强其可用性。
   - 该工具被推荐用于将图像和照片转换为可搜索格式，并强调了其安全协议。
- **从大纲生成 Podcast**：一位用户通过提供详细大纲并逐章生成内容，成功生成了一集长篇 Podcast。
   - 他们计划将生成的文本与 Eleven Labs 结合使用，为一个纪录片风格的项目创建音频和视觉效果。
- **小说写作中的 AI**：用户询问了如何将 Notebook LM 用于小说写作，寻求引导 AI 进行创意故事开发的方法。
   - 一位用户建议通过自定义设置来提高 AI 的脚本编写能力。
- **多语言 AI 的挑战**：讨论了 AI 用多种语言交流和处理不同口音的能力。
   - 用户注意到语言输出的成功率各不相同，特别是在尝试苏格兰或波兰等口音时。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tools.pdf24.org/en/ocr-pdf">PDF OCR - 识别文本 - 简单、在线、免费</a>：用于通过 OCR 识别文档中文本的免费在线工具。创建可搜索的 PDF 文件。选项丰富。无需安装。无需注册。</li><li><a href="https://hailuoai.video/">海螺 AI (Hailuo AI)：用 AI 将创意转化为视觉效果</a>：海螺 AI 工具 - 创新的 AI 视频生成器和提示词工具，将您的想法转化为惊艳的 AI 视频。利用尖端的 AI 技术在短时间内创建迷人的视觉效果...</li><li><a href="https://open.spotify.com/show/0MvNgBDb2NsZJN4cREl7yF?si=631a4782e6ee4fde">Top Shelf</a>：Podcast · Four By One Technologies · “Top Shelf” 是您获取当今畅销书快速、深刻见解的首选 Podcast。只需 15 分钟，即可获得精华、金句和新鲜视角...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1313133942408478730)** (140 条消息🔥🔥): 

> `NotebookLM 更新、Audio Overview 功能、语言支持、用户体验反馈、Google Drive 集成` 


- **用户对无限 Audio Overviews 的需求**：用户表达了对每天 **无限次音频生成** 的强烈兴趣，强调目前 **20 次** 的限制对于学习来说过于局促。
   - 关于通过订阅模式增加访问权限的建议引起了不同的反应，凸显了对更多灵活性的需求。
- **PDF 读取的挑战**：用户报告了 NotebookLM 在正确处理长 PDF 方面的问题，经常收到无法访问完整文档的消息。
   - 有推测认为，其他模型（如 **Gemini 1.5 Pro**）在这方面可能具有更优越的能力。
- **关于语言支持的问题**：许多用户询问如何更改 NotebookLM 中的语言设置，特别是为了生成非英语的输出。
   - 目前的指导建议是修改 Google 账户设置以反映首选语言，尽管用户对其支持程度仍存疑问。
- **用户组织功能的反馈**：参与者建议在 NotebookLM 中增加 **文件夹组织** 功能，以便更好地管理大量笔记本并避免混乱。
   - 在此期间，有人提出了一种使用短类别代码的变通方法，以方便手动组织。
- **关于 Google Drive 兼容性的担忧**：用户对 Google Gemini 访问 Google Drive 文档能力的不稳定性表示沮丧，并指出这可能与免费版本的潜在限制有关。
   - 用户提出了该功能是仅对付费用户开放，还是取决于特定功能的问题。



**提到的链接**：<a href="https://x.com/BryanKerrEdTech/status/1855790049151082683">来自 Bryan Kerr (@BryanKerrEdTech) 的推文</a>：我找到了在我的 Podcast 应用中收听 NotebookLM Audio Overviews 的方法。现在我可以更有目的地享受散步和通勤了。你也可以做到。你只需要一个 Dropbox 或 OneDrive 账户以及 Pu...

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1313127690768879687)** (122 条消息🔥🔥): 

> `意大利 AI 监管、ChatGPT 功能问题、投票与量子计算、内容审核挑战、AI 翻译对比` 


- **意大利 AI 监管法案**：意大利计划禁止包括 OpenAI 在内的 AI 平台，除非个人可以请求删除其数据，这引发了对其影响的讨论。
   - 有人担心地理位置封锁无效，因为用户可能会找到绕过这些限制的方法。
- **ChatGPT 的功能限制**：用户报告称在购买 ChatGPT Plus 计划后，遇到了图像生成等功能无法正常工作的问题。
   - 对生成不当内容的担忧引发了关于系统应如何处理此类 Prompt 的讨论。
- **量子计算与投票的相关性**：一场关于量子计算是否能辅助投票过程的辩论爆发，人们对其在此背景下的实际应用表示怀疑。
   - 有人指出，选民不能处于叠加态（superposition），且量子方法并不能增强经典的共识算法。
- **关于内容审核的辩论**：用户讨论了 AI 中内容审核的复杂性，指出当前的方法不足且面临许多边缘案例（edge cases）。
   - 强调虽然 AI 可以生成关于违反政策的警告，但在审核中实现完全自动化是不可行的。
- **AI 翻译选项**：有人提出了关于将内容翻译成匈牙利语的最佳 AI 工具的问题，对比了 DeepL、Claude 和 ChatGPT 等选项。
   - 这引发了对用户在不同 AI 翻译服务有效性方面的进一步经验探索。



**提到的链接**：<a href="https://ai-image-generator-3be8e.web.app/">React App</a>：未找到描述

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1313128045497810954)** (6 条消息): 

> `GPT 功能问题、ChatGPT Plus 计划、转录模型` 


- **用户面临 GPT 功能问题**：一位用户报告称，一个旨在编译计费工时的 GPT 存在困难，称其会遗忘条目，且难以生成兼容 XLS 的列表。
   - *难道是 GPT 对这项工作感到厌烦了？*
- **ChatGPT Plus 计划未提供承诺的功能**：另一位用户在花费 **$20** 购买 ChatGPT Plus 计划后表示沮丧，报告称图像生成和文件读取等功能出现故障。
   - 他们指出回复似乎过时了，另一位成员确认已经遇到同样的问题一周了。
- **关于转录模型的澄清**：一位成员对用于转录的不同模型进行了澄清，建议不应直接对比它们。
   - 他们引用了 [OpenAI 的语音模式常见问题解答](https://help.openai.com/en/articles/8400625-voice-mode-faq#h_b8d80d20be) 以获取更多细节。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1313165779675644016)** (9 条消息🔥): 

> `ChatGPT 中的自定义指令、改进提示工程、学习用户风格、ChatGPT 写作风格` 


- **自定义指令增强 ChatGPT 的输出**：一位成员建议使用 [自定义指令 (Custom Instructions)](https://help.openai.com/en/articles/8096356-custom-instructions-for-chatgpt) 来告知 ChatGPT 特定的风格要求，这可以简化编辑工作。
   - 另一位指出，这种方法与在 Explore GPTs 功能中创建新的 GPT 不同。
- **ChatGPT 学习用户风格的可能性**：一位成员询问 ChatGPT 是否可以学习他们的叙事风格以减少编辑需求。
   - 回复指出，展示所需风格的示例可以帮助 ChatGPT 相应地调整其写作。
- **增强提示工程 (Prompt Engineering) 的资源**：一位用户表示有兴趣寻找免费或低成本的资源，以改进他们针对自定义 GPT 的 Prompt Engineering。
   - 这反映了人们对开发更好的 ChatGPT 交互并最大化其能力的广泛兴趣。
- **使用 ChatGPT 以不同风格写作**：一位参与者强调尝试使用 ChatGPT 作为工具来尝试不同的写作风格，以精炼叙述表达。
   - 有人指出，调整模型的输出可能需要特定的指导，包括示例文本。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1313165779675644016)** (9 条消息🔥): 

> `Custom Instructions, Prompt Engineering, Storytelling Style Adaptation` 


- **探索用于风格调节的 Custom Instructions**：成员们讨论了使用 [custom instructions](https://help.openai.com/en/articles/8096356-custom-instructions-for-chatgpt) 来调整 ChatGPT 的写作风格，认为这有助于契合个人的叙事偏好。
   - 有人建议向 ChatGPT 提供示例，以展示所需的风格，从而改进输出效果。
- **Custom Instructions 与 Custom GPTs 的区别**：澄清了 custom instructions 与在 Explore GPTs 板块中创建 custom GPT 之间的区别。
   - Custom instructions 旨在调整模型的回复风格，而不是完全开发一个新的 GPT。
- **提升 Prompt Engineering**：一位用户表示打算提升其 prompt engineering 技能，以便利用 OpenAI ChatGPT 更好地开发 custom GPT。
   - 他们询问了有哪些免费或低成本的资源可以提升他们的理解和应用水平。


  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1313282231606968382)** (2 条消息): 

> `Model removals, Price reductions, Claude 3.5 Haiku discount` 


- **两款模型下架**：模型 `nousresearch/hermes-3-llama-3.1-405b` 和 `liquid/lfm-40b` 已停止提供。
   - 提醒用户通过充值额度来保持 API 请求的正常运行。
- **宣布大幅降价**：`nousresearch/hermes-3-llama-3.1-405b` 的价格从每百万 token **4.5** 降至 **0.9**，而 **liquid/lfm-40b** 从 **1** 降至 **0.15**。
   - 这些显著的降价是模型下架后的一丝慰藉。
- **Claude 3.5 Haiku 促销提醒**：宣布 Claude 3.5 Haiku 降价 **20%**，为用户提供更实惠的选择。
   - 此次折扣是持续致力于提高模型易用性工作的一部分。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1313181525189201995)** (117 条消息🔥🔥): 

> `Hermes 405B 模型状态、OpenRouter API 密钥管理、Gemini Flash 错误、新 Amazon Nova 模型、LLM Tokenization 见解` 


- **Hermes 405B 模型不再可用**：用户确认 **Hermes 405B** 模型已永久下线，并表达了这是一个时代终结的感慨。
   - 讨论了替代方案的成本，一些用户考虑购买模型，而另一些用户则表示更倾向于使用现有的免费模型。
- **OpenRouter 密钥管理功能**：OpenRouter 允许创建和管理 API 密钥，用户可以为每个密钥设置和调整信用额度限制，且限制更改时不会自动重置。
   - 用户被告知需要自行管理密钥访问权限，以保持对谁可以使用其应用程序的控制。
- **瞬时 Gemini Flash 错误**：用户报告在访问 **Gemini Flash** 时遇到 **525 Cloudflare 错误**，经发现这是一个很快就得到解决的瞬时问题。
   - 注意到了该模型的不稳定性，建议通过 OpenRouter 的聊天界面验证功能。
- **Amazon Nova 模型计划**：目前正在讨论集成新的 **Amazon Nova 模型**，这些模型目前由 AWS Bedrock 独占。
   - 用户对新模型表现出兴趣，认为它们似乎是值得尝试的不错选择。
- **LLM Tokenization 见解**：讨论包括 LLM 如何将未见过的字符串分解为可识别的 token，并强调了 token embedding 的重要性而非 token 本身。
   - 分享了一个用于实验各种 tokenizer 的外部资源，以便进一步探索该主题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>：LLM 聊天室是一个多模型聊天界面。添加模型并开始聊天！聊天室将数据本地存储在您的浏览器中。</li><li><a href="https://tiktokenizer.vercel.app/">Tiktokenizer</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/Xenova/the-tokenizer-playground">The Tokenizer Playground - Xenova 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/openai/tiktoken">GitHub - openai/tiktoken: tiktoken 是一款用于 OpenAI 模型的快速 BPE 分词器。</a>：tiktoken 是一款用于 OpenAI 模型的快速 BPE 分词器。 - openai/tiktoken</li><li><a href="https://openrouter.ai/docs/transforms">Transforms | OpenRouter</a>：为模型消耗转换数据</li><li><a href="https://aws.amazon.com/cn/bedrock/pricing/">使用基础模型构建生成式 AI 应用程序 - Amazon Bedrock 定价 - AWS</a>：未找到描述</li><li><a href="https://www.reddit.com/r/SillyTavernAI/s/esRCZFpBus">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1313152934560927835)** (5 条消息): 

> `自定义提供商密钥、BYOK 访问、Gemini 实验性模型` 


- **请求自定义提供商密钥访问权限**：多位用户表示有兴趣获得 **自定义提供商密钥** 的访问权限，表明频道内对此需求很高。
   - 一位成员特别将他们的访问查询与无法使用 **Gemini experimental 1121 模型** 联系起来。
- **BYOK 访问更新**：一份快速更新宣布，团队正致力于很快向所有人开放 **BYOK (Bring Your Own Key)** 访问，尽管目前私测已暂停。
   - 团队正在积极解决一些 **小问题 (kinks)**，然后再继续推进。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1313163616828260353)** (100 messages🔥🔥): 

> `LORA 训练, Stable Diffusion 指南, 诈骗警报, GPU 利用率, 新型图像合成模型` 


- **创建 LORA 的技巧**：用户讨论了创建有效 LORA 的各种策略，例如使用由图像制成的背景 LORA，并在 Photoshop 或 Krita 等软件中清理输出结果。
   - 一位成员建议在将生成的图像用于训练之前对其进行精炼，以确保更高质量的结果。
- **Stable Diffusion 设置指南**：多位用户寻求关于开始使用 Stable Diffusion 的建议，建议包括使用 ComfyUI 和各种云端选项。
   - 成员们强调了了解是本地运行还是利用云端 GPU 的重要性，并推荐了 Vast.ai 等资源作为租赁选项。
- **防范诈骗者意识**：对服务器中潜在诈骗者的担忧促使用户分享警告和建议措施，例如向 Discord 举报可疑账号。
   - 成员们讨论了如何识别网络钓鱼企图，以及某些账号如何冒充支持人员来欺骗用户。
- **了解 GPU 性能**：对话强调了 GPU 性能的差异，用户比较了不同型号的使用体验，并强调显存和速度是关键因素。
   - 一位用户指出，由于电费成本，与本地设置相比，更便宜的云端 GPU 选项可能会带来更好的整体性能。
- **关于新型图像合成模型的讨论**：成员们讨论了一个名为 Sana 的新模型，注意到其与之前版本相比的效率和质量，一些人对其商业用途表示怀疑。
   - 有人建议，对于日常用途，使用 Flux 或之前的模型可能会在闲暇时产生相似或更好的结果。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nvlabs.github.io/Sana/">Sana</a>：未找到描述</li><li><a href="https://vast.ai/">租赁 GPU | Vast.ai</a>：通过最优质的云端 GPU 租赁，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面允许对所有供应商的 GPU 租赁进行公平比较。</li><li><a href="https://dontasktoask.com/">不要问能不能问，直接问</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo&list=PLIF38owJLhR1EGDY4kOnsEnMyolZgza1x">ComfyUI - 入门：第 1 集 - 在 Stable Diffusion AI 艺术生成方面优于 AUTO1111</a>：今天我们介绍如何使用 ComfyUI 通过 Stable Diffusion 模型创建 AI 艺术的基础知识。这个基于节点的编辑器是一个理想的工作流工具...</li><li><a href="https://youtu.be/ng8WBNilBKA?si=aZs2uHkxAw053qmv">利用 AI 艺术革新您的 WordPress 网站：AI Artist 插件 + 免费预配置服务器！</a>：🎨 在您的 WordPress 网站上创建令人惊叹的 AI 艺术（我保证这很简单！）在这段视频中，我将介绍 WordPress 的 AI Artist 插件，这是一种超级简单的方法...
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1313130346371354647)** (84 messages🔥🔥): 

> `Pydantic AI, NotebookLM 团队变动, Hunyuan Video 发布, Amazon Nova 基础模型, ChatGPT 对名称的处理`

- **Pydantic AI 框架正式上线**：来自 Pydantic 的全新 [Agent 框架](https://ai.pydantic.dev/) 旨在与 LLM 集成，展示了在 AI 创新应用方面的潜力。
   - 然而，关于它与 LangChain 等现有框架的差异化存在质疑，一些用户认为它只是在模仿现有的解决方案。
- **NotebookLM 团队开启新征程**：来自 Google NotebookLM 团队的核心成员 @raizamrtn 和 @jayspiel_ 在做出多年重大贡献后，正离职共同创立一家新公司。
   - 他们的这段新旅程有望激发 AI 领域的进一步创新，并邀请关注者在 [werebuilding.ai](https://werebuilding.ai/) 关注他们的最新动态。
- **混元视频（Hunyuan Video）引发轰动**：腾讯发布的 [混元视频](https://x.com/angrypenguinpng/status/1863811509219950835?s=46) 使其成为开源文本生成视频技术的新领导者，展示了令人印象深刻的质量。
   - 最初的用户反馈强调了渲染对资源的巨大需求，但他们对未来的效率提升持乐观态度。
- **亚马逊发布 Nova 基础模型**：亚马逊宣布了其全新的基础模型 [Nova](https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws)，其定位是与 GPT-4o 等先进模型竞争。
   - 初步印象显示它很有前景，尽管用户的反馈褒贬不一，并没有像对待之前的模型那样感到完全惊艳。
- **ChatGPT 的姓名危机**：某些姓名（尤其是 David Mayer）被 ChatGPT 标记，导致在提及这些姓名时由于系统故障而中断响应。
   - 这一问题并不影响 OpenAI API，但引发了人们对姓名关联对 AI 行为影响的好奇和评论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://simonwillison.net/2024/Dec/3/names-make-chatgpt-grind-to-a-halt/#atom-everything">某些名字会让 ChatGPT 陷入停滞，我们已经知道原因了</a>：Benj Edwards 探讨了一个非常奇怪的行为，即 ChatGPT 在生成 David Mayer、Brian Hood、Jonathan Turley、Jonathan Zittrain、David Faber 等名字时会停止输出并报错，而不是正常显示。</li><li><a href="https://x.com/skirano/status/1864014133756129752">Pietro Schirano (@skirano) 的推文</a>：我添加了一个新的 MCP 服务器，让 Claude 在回答之前可以进行分步思考。Claude 能够预先决定需要多少个思考步骤，追溯其思路，甚至在发现...时进行分支。</li><li><a href="https://x.com/bdsqlsz/status/1863653398840840348">青龍聖者 (@bdsqlsz) 的推文</a>：Huggingface 突然增加了空间限制，现在免费额度为 500G，Pro 会员提供 1TB。对我们来说，这是一个噩梦。</li><li><a href="https://www.forbes.com/sites/rashishrivastava/2024/12/02/cognition-scott-wu-devin-ai/">程序员担心这家估值 20 亿美元初创公司的 AI 可能会取代他们的工作</a>：在 2 亿美元资金的支持下，28 岁的 Scott Wu 及其在 Cognition 的竞赛程序员团队正在构建一个可以完全自主编程的 AI 工具，就像“一支初级工程师大军”....</li><li><a href="https://x.com/kiwicopple/status/1863616764942176668?s=46">Paul Copplestone — e/postgres (@kiwicopple) 的推文</a>：今天我们发布了 @supabase AI 助手 v2，它就像是数据库领域的 Cursor。甚至连澳大利亚人都能用 ↓↓</li><li><a href="https://www.theguardian.com/technology/2024/dec/03/chatgpts-refusal-to-acknowledge-david-mayer-down-to-glitch-says-openai">OpenAI 表示，ChatGPT 拒绝承认“David Mayer”是由于故障所致</a>：聊天机器人的开发商表示，该名字被错误地标记，从而无法在回复中出现。</li><li><a href="https://ai.pydantic.dev/">简介</a>：用于在 LLM 中使用 Pydantic 的 Agent 框架 / 适配层</li><li><a href="https://x.com/theworldlabs/status/1863617989549109328">World Labs (@theworldlabs) 的推文</a>：我们一直忙于构建一个能从单张图像生成 3D 世界的 AI 系统。在我们的网站上查看一些早期成果，你可以直接在浏览器中与我们的场景进行交互！https://worldl...</li><li><a href="https://www.cognition.ai/blog/evaluating-coding-agents">Cognition | 对 OpenAI o1 的回顾以及我们如何评估编程 Agent</a>：我们是一家构建端到端软件 Agent 的应用 AI 实验室。</li><li><a href="https://x.com/kalomaze/status/1862981345531617732">kalomaze (@kalomaze) 的推文</a>：SillyTavern 是一个前端（面向小众群体），基本上可以连接任何 API；LM Studio 是一个前端+后端（面向普通人/Mac 开发者），是 llama.cpp 的专有分支；KoboldCPP 是一个前端+后端（...</li><li><a href="https://www.aboutamazon.com/news/aws/amazon-nova-artificial-intelligence-bedrock-aws">介绍 Amazon Nova，我们的新一代基础模型</a>：来自 Amazon 的全新 SOTA 基础模型，提供前沿智能和行业领先的性价比。</li><li><a href="https://neuralmagic.com/blog/24-sparse-llama-smaller-models-for-efficient-gpu-inference/">2:4 稀疏 Llama：用于高效 GPU 推理的小型模型</a>：探索 Sparse Llama：一个经过 50% 剪枝、针对 GPU 优化的 Llama 3.1 模型，具有 2:4 稀疏性，可在不牺牲准确性的情况下实现更快、更具成本效益的推理。</li><li><a href="https://githubnext.com/projects/github-spark">GitHub Next | GitHub Spark</a>：GitHub Next 项目：我们能否让任何人都能使用 AI 和全托管运行时，为自己创建或适配软件？</li><li><a href="https://x.com/angrypenguinpng/status/1863811509219950835?s=46">AP (@angrypenguinPNG) 的推文</a>：开源视频生成领域的新王者来了！腾讯刚刚发布了他们的开源视频模型：Hunyuan Video。</li><li><a href="https://x.com/jonathan_adly_/status/1857838506518917169?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Jonathan Adly (@Jonathan_Adly_) 的推文</a>：如果你对 ColPali 和 RAG 感兴趣。我们发布了一个生产就绪的 RAG API，它实现了论文内容并配有完整的评估流水线。我们的评估结果非常接近 @ManuelFays 最新的模型...</li><li><a href="https://simonwillison.net/2024/Dec/3">2024 年 12 月 3 日星期二存档</a>：未找到描述</li><li><a href="https://x.com/browsercompany/status/1863593525725556754?s=46">The Browser Company (@browsercompany) 的推文</a>：来帮我们打造第二个产品，一款名为 Dia 的智能浏览器。更多信息请访问 🔗 diabrowser [dot] com</li><li><a href="https://x.com/multimodalart/status/1864003035870978457">apolinario 🌐 (@multimodalart) 的推文</a>：我的第一个 HunyuanVideo 生成作品 🎥 “一只水豚在草地上行走，写实风格” 🌱₍ᐢ-(ｪ)-ᐢ₎🌱 绝对是 SOTA 级别的开源质量！🔥 不过耗费了 60GB VRAM 和 40 分钟 —— 以后能降低多少呢...</li><li><a href="https://x.com/exaailabs/status/1864013080944062567?s=46">Exa Labs (@exaailabs) 的推文</a>

<li><a href="https://x.com/ExaAILabs/status/1863649174358831312?s=46">来自 Exa (@ExaAILabs) 的推文</a>：发布 Exa Websets —— 迈向完美网页搜索的突破。请在下方注册候补名单👇</li><li><a href="https://x.com/scottwu46/status/1863673065684734240?s=46">来自 Scott Wu (@ScottWu46) 的推文</a>：Devin 已为公司节省了数百万美元，并在工程时间上展现了高达 8 倍的生产力提升。很高兴能与《福布斯》探讨 Devin 与 N... 等客户合作开展的工作。</li><li><a href="https://x.com/lmarena_ai/status/1864062852589605156?s=46">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：祝贺 @amazon 发布最新的前沿模型 Nova！⭐ Nova 在标准基准测试中可与 GPT-4o 等顶级模型相媲美。现在，真正的挑战开始了——Nova 已进入 Arena 进行人类评估...</li><li><a href="https://x.com/ExaAILabs/status/1806444570210934949">来自 Exa (@ExaAILabs) 的推文</a>：Exa 如何提供十亿级规模的向量搜索？我们将 binary quantization、Matryoshka embeddings、SIMD 和 IVF 结合到一个全新的系统中，其性能优于 HNSW 等替代方案。@shreyas4_ 今天发表了演讲...</li><li><a href="https://wattenberger.com/thoughts/fish-eye">LLMs 是思维的工具</a>：未找到描述</li><li><a href="https://x.com/Wattenberger/status/1863977304126603309">来自 Amelia Wattenberger 🪷 (@Wattenberger) 的推文</a>：🐟 关于我们如何利用 LLMs 🐠 在多个抽象层级上与文本进行交互的一些思考 🐡 灵感来自鱼眼镜头</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/sequentialthinking">modelcontextprotocol/servers 仓库 main 分支下的 servers/src/sequentialthinking</a>：Model Context Protocol 服务端。通过在 GitHub 上创建账号来为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://x.com/raizamrtn/status/1863645718159954272">来自 Raiza Martin (@raizamrtn) 的推文</a>：在 Google 工作 5.5 年后，今天是我在这里的最后一天。领导 @NotebookLM 从一个想法成长为服务数百万用户的产品，是我一生中最难忘的经历。但最棒的部分？是在并肩作战中找到了我未来的联合创始人...</li><li><a href="https://x.com/jayspiel_/status/1863653067079684582">来自 Jason Spielman (@jayspiel_) 的推文</a>：在 Google 度过了不可思议的 7.5 年后——最近在构建 NotebookLM 并为 Google 一些最具创新性的 AI 产品做出贡献——我将离职去创办一家公司。亮点：1️⃣ 2011 年我参与了...</li><li><a href="https://x.com/GoogleDeepMind/status/1861487975508431347">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：将内容转化为引人入胜的音频对话只是 @NotebookLM 的功能之一 🗣️ 来自 @LabsDotGoogle 团队的 @Stevenbjohnson 和 @Raizamrtn 认为它可能会对学习产生变革性影响...</li><li><a href="https://x.com/futurejurvetson/status/1863649174358831312?s=46">来自 Steve Jurvetson (@FutureJurvetson) 的推文</a>：摩尔定律更新。注意：这是一个半对数图，因此直线代表指数增长；y 轴的每个刻度代表 100 倍。这张图表涵盖了计算能力 1,000,000,000,000,000,000,000 倍的提升/...</li><li><a href="https://x.com/iscienceluvr/status/1863504851704910135?s=46">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：有人在一个现已删除的 GitHub 仓库中找到了 qianxyz 编写的一些用于解决 AoC 的代码。看起来像是一个使用 gpt-4o-mini 和非常基础的提示词构建的自动化流水线。引用 Tanishq M...</li>

### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1313253836915474463)** (3 条消息): 

> `Bolt Launch, AI Agents Discussion, Open Source Strategies, Revenue Growth in AI, AI Interface Dynamics` 


- **Bolt 令人兴奋的发布与营收**：最近一期由 **Bolt** 担纲的播客强调了其惊人的增长，作为一个 **Claude Wrapper**，在短短 **2 个月内实现了超过 800 万美元的 ARR**。讨论嘉宾包括 [@ericsimons40](https://x.com/ericsimons40) 和 [@itamar_mar](https://x.com/itamar_mar)。
   - *我们很高兴能与 @QodoAI 交流*，并在播客中让 *@stackblitz* 首次亮相，展示 **Code Agent** 工程化的可能性。
- **深入探讨 AI Agents**：播客涵盖了 **Generic vs. Specific AI Agents**（通用型 vs. 特定型 AI Agents）以及 **Human vs Agent Computer Interfaces**（人机 vs. Agent 计算机界面）的动态。关键时间点强调了关于使用 AI 进行维护与创造的讨论，以及为什么 **Docker** 不适合 **Bolt**。
   - 听众可以学习分解复杂任务的策略，以及对 **Bolt** 成功经验的反思。
- **Open Source 与未来增长**：本期讨论了 **Building in Open Source**（在开源中构建）的重要性以及创始人**选择产品**的策略。分享了关于 **Bolt 独特功能**的见解及其在 **AI Engineering** 领域的未来增长前景。
   - 还探讨了 **AI Capabilities and Pricing Tiers**（AI 能力与定价层级）以及**竞争格局**等话题，为听众提供了该领域的全面视角。
- **个人见解与对创始人的建议**：在节目最后，主持人分享了个人故事，包括*迎来新生命*和*完成铁人三项 (Iron Man)*，展示了初创公司的职场生活平衡。此外，他们还为**创始人**提供了关于在创业中拥抱 AI 的**宝贵建议**。
   - 这些个人反思与专业见解的结合，为有抱负的企业家提供了一场全方位的讨论。



**提及的链接**：<a href="https://x.com/latentspacepod/status/1863694873775440143">来自 Latent.Space (@latentspacepod) 的推文</a>：🆕 Bolt、面向 Code Agents 的 Flow Engineering，以及作为 Claude Wrapper 在 2 个月内实现 >$8m ARR，对话 @ericsimons40 和 @itamar_mar！我们很高兴在播客中与 @QodoAI 交流并让 @stackblitz 首次亮相...

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1313132282219073607)** (53 条消息🔥): 

> `Manufacturing Discussions, New Rerank 3.5 Features, Colpali and Tewi References, Multilingual Support in Rerank, Community Engagement` 


- **制造业岗位引发关注**：一位成员询问是否有人有**制造业工作**的经验，特别是机械师角色，并表示希望分享使用案例。
   - 一位参与者注意到 Discord 讨论相对安静，将其归因于目前每个人都专注于构建和开发。
- **Rerank 3.5 带来功能改进**：社区对新的 **Rerank 3.5** 表示赞赏，它提供了增强的推理和多语言能力，能够更好地搜索复杂的企业数据。
   - 成员们对性能提升感到兴奋，并讨论了它在多媒体等各种格式中的兼容性。
- **Colpali 与 Tewi：社区幽默**：社区围绕 **Colpali** 和 **Tewi** 进行了轻松的调侃，开玩笑地将各种问题归咎于这些实体。
   - 这种交流凸显了成员之间的情谊，他们在面对挑战时找到了幽默感。
- **多语言支持确认**：一位成员询问了 **Rerank 3.5** 的多语言支持情况，询问是否可以从之前的多语言版本切换过来。
   - 官方确认 Rerank 3.5 同时支持 **Multilingual 和 English** 功能，为用户提供了更多灵活性。
- **对 Google Gemini 的好奇**：一位成员表达了对 **Google Gemini** 的挫败感，询问其与 **Google Drive** 的配合功能以及访问文档的一致性。
   - 成员们担心某些功能是否仅对付费用户开放，从而引发了关于这些服务可靠性的对话。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://cohere.com/blog/rerank-3pt5">Introducing Rerank 3.5: Precise AI Search</a>：Rerank 3.5 提供了改进的推理和多语言能力，能以更高的准确度搜索复杂的企业数据。</li><li><a href="https://huggingface.co/blog/manu/colpali">ColPali: Efficient Document Retrieval with Vision Language Models 👀</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1313552078953648159)** (1 条消息): 

> `Rerank 3.5, API deprecations, Multilingual capabilities, Enhanced reasoning, Legacy model lifecycle` 


- **Rerank 3.5 正式发布！**：全新的 **Rerank 3.5** 模型提供了 SOTA 性能，具备增强的推理能力，并能更好地兼容长文档、电子邮件和半结构化数据的搜索。
   - 它还支持超过 **100 种语言**（如阿拉伯语、法语、日语和韩语）的**多语言性能**，可通过 `rerank-v3.5` API 别名访问；查看[博客文章了解更多详情](https://cohere.com/blog/rerank-3pt5)。
- **关于 API 弃用的重要更新**：Cohere 宣布了旧版模型的**弃用 (deprecations)**，并在[此文档页面](https://docs.cohere.com/docs/deprecations)上提供了针对已弃用端点和模型的推荐替代方案信息。
   - 该文档概述了**模型生命周期**，从“活跃 (Active)”到“已弃用 (Deprecated)”，这将影响依赖 Cohere 旧版模型的应用程序。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/rerank-3pt5">Introducing Rerank 3.5: Precise AI Search</a>: Rerank 3.5 提供改进的推理和多语言能力，以更高的准确度搜索复杂的企业数据。</li><li><a href="https://docs.cohere.com/docs/deprecations">Deprecations — Cohere</a>: 了解 Cohere 的弃用政策和推荐的替代方案
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1313168993451905055)** (9 条消息🔥): 

> `TooManyRequestsError, Payment Issues with Card, API Key Setup Delay` 


- **尽管使用生产密钥仍出现 TooManyRequestsError**：一位用户报告称，即使在升级到允许每分钟超过 **10 次调用**的生产 API Key 后，仍然遇到 **TooManyRequestsError**。
   - 另一位成员建议在 [support@cohere.com](mailto:support@cohere.com) 创建支持工单以获取进一步帮助，但该用户随后承认，问题可能是由于 **API Key 设置延迟**造成的。
- **信用卡支付被拒**：一位用户对信用卡支付被拒表示沮丧，尽管上个月还成功使用过。
   - 成员们建议这可能是银行的问题，并推荐联系 [support@cohere.com](mailto:support@cohere.com) 解决，因为支付是由 **Stripe** 处理的。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1313153691087536159)** (6 条消息): 

> `TooManyRequestsError, Production API Key Setup Delay` 


- **用户在扩容后仍面临 TooManyRequestsError**：一位用户提到，在为 Rerank 升级到允许每分钟超过 10 次调用的生产密钥后，仍收到 **TooManyRequestsError**。
   - 他们确认已更改 API Key 并检查了使用情况，表明可能存在设置或传播 (propagation) 方面的问题。
- **针对 API 问题建议联系支持人员**：另一位成员建议用户发送电子邮件至 [support@cohere.com](mailto:support@cohere.com) 并附上其客户 ID 以寻求帮助。
   - *传播不应该耗时太久*，另一位成员指出，并暗示设置延迟通常是极短的。
- **用户的 API Key 问题似乎已解决**：该用户随后报告称问题似乎已解决，推测 API Key 激活可能存在延迟。
   - 根据回复，设置延迟最多可能只有几分钟。


  

---

### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1313518121478520904)** (1 messages): 

> `Harmony Project, Large Language Model Competition, Natural Language Processing in Questionnaire Harmonisation` 


- **介绍 Harmony 项目**：**Harmony** 项目旨在利用 [Natural Language Processing](https://fastdatascience.com/guide-natural-language-pr) (NLP) 技术来协调问卷项和元数据。研究人员可以使用它来[比较不同研究中的问卷项](https://harmonydata.ac.uk/compare-harmonise-instruments/gad-7-vs-beck-anxiety-inventory/)。
   - 该项目总部位于 UCL，涉及多家大学和专业人士的合作。
- **LLM 匹配算法竞赛**：官方宣布了一项竞赛，旨在改进 Harmony 的 LLM 匹配算法，通过 [DOXA AI](https://harmonydata.ac.uk/doxa/) 参赛的选手有机会获得高达 **£500** 的奖金。该竞赛对初学者友好，不需要先前的 LLM 经验。
   - 感兴趣的参与者可以加入 **Harmony Discord 服务器**，并查看 🏅「matching-challenge」频道获取更多信息。
- **评估 Harmony 的性能**：正如其[博客文章](https://harmonydata.ac.uk/nlp-semantic-text-matching/measuring-the-performance-of-nlp-algorithms/)中所讨论的，Harmony 项目承认在准确匹配相似句子方面存在挑战。系统有时会错误标记一些专业人士可能有不同看法的相似性。
   - 他们非常希望获得社区的反馈，以完善 Harmony 的匹配能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://harmonydata.ac.uk/">Harmony | A global platform for contextual data harmonisation</a>: 全球上下文数据协调平台</li><li><a href="https://harmonydata.ac.uk/doxa/">Competition to train a Large Language Model for Harmony on DOXA AI | Harmony</a>: 在 DOXA AI 上为 Harmony 训练 LLM 的竞赛
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/)** (1 messages): 

mrdragonfox: - 嘿 "new" - 我是 mrdragonfox ^^
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1313189735107137646)** (7 messages): 

> `Xmma Kernels Performance, Nvjet vs Cutlass Comparisons, GEMM Toolkit Updates, Runtime Error with Meta Tensors` 


- **Xmma Kernels 表现有趣但缓慢**：成员们讨论了 **Xmma kernels**，其中一位指出在目前的实验中它们“差得令人尴尬”。
   - 在实验 **nvjet** 时，另一位成员发现它在较小尺寸上追赶了上来，尽管他们的自定义 kernel 在 **N=8192** 时仍然快 **1.5%**。
- **Nvjet 展示了具有竞争力的性能**：讨论强调 **nvjet** 与 **cutlass** 相比具有竞争力，个人经验表明它在某些情况下可以超越 **cutlass**。
   - 一位成员指出，**cutlass** 在特定情况下可能略优于 **nvjet**，但总体认为 **nvjet** 更具竞争力。
- **GEMM Toolkit 更新引发好奇**：一位成员透露 **12.6.2 toolkit** 已于 **10 月** 发布，引发了关于在此之前使用何种方法进行 **GEMM** 的疑问。
   - 共识是，在 toolkit 发布之前，许多人依赖 **cublas**，或者为了更深入的优化而使用 **cutlass/triton**。
- **Meta Tensors 的运行时错误**：一位用户就特定函数的运行时错误寻求帮助，该错误导致了围绕 **Tensor.item()** 方法的困惑。
   - 错误信息指向 **meta tensors** 的问题，引发了调试协助的请求。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1313236436178178189)** (3 条消息): 

> `Triton MLIR Dialects, Floating Point Representations in Triton, Documentation and Tutorials` 


- **Triton Kernel 中浮点参数的疑虑**：一名成员提出了关于在 Triton kernel 中将 **1.5** 之类的 `float` 用作 `tl.constexpr` 参数的安全性疑虑，询问是否存在潜在的**浮点表示（floating point representation）**问题。
   - 目前还没有针对该用法的潜在风险或预防措施的直接回复。
- **寻求 Triton MLIR Dialects 文档**：一名成员询问是否有关于 **Triton MLIR Dialects** 的可用文档或教程，并表示链接中的大部分内容似乎并不完整。
   - 另一名成员提供了一个指向 [Triton Ops 文档](https://triton-lang.org/main/dialects/TritonOps.html)的链接，并强调虽然内容很少，但包含相关的示例。
- **Triton 的极简编程指南**：有人注意到 GitHub 上有一个非常简略的 [编程指南](https://github.com/triton-lang/triton/tree/main/docs/programming-guide)，但似乎尚未完成。
   - 该指南旨在帮助使用 **Triton 语言**工作的开发者，但目前缺乏详尽的内容。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://triton-lang.org/main/dialects/dialects.html">Triton MLIR Dialects and Ops &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://triton-lang.org/main/dialects/TritonOps.html">TritonOps &mdash; Triton 文档</a>：未找到描述</li><li><a href="https://github.com/triton-lang/triton/tree/main/docs/programming-guide">triton/docs/programming-guide at main · triton-lang/triton</a>：Triton 语言和编译器的开发仓库 - triton-lang/triton
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1313184876324720743)** (5 条消息): 

> `Warp Schedulers in GPU Architecture, Comparison of FP32 Cores in Different Models` 


- **理解 Warp 调度器和 FP32 核心分布**：一名成员解释说，一个 warp 由 **32 个线程**组成，并行使用 **32 个 FP32 核心**，从而导致每个 **SM** 拥有 **128 个 FP32 核心**。
   - *这一概念阐明了现代 GPU 的并行处理能力。*
- **A100 与 RTX 核心数量的差异**：另一名成员指出，**A100** 的 **4 个 warp 调度器**仅对应 **64 个 FP32 核心**，这与拥有 **128 个 FP32 核心**的 **RTX 30xx 和 40xx** 系列不同。
   - *这突显了不同代 GPU 之间的架构差异。*
- **关于 Warp 执行的特定架构细节**：在继续讨论中，有人注意到 **Volta 架构**具有 **4 个 warp 调度器**，每个调度器有 **16 个 FP32 核心**，这意味着一个 warp 需要两次通过（two passes）才能执行每条指令。
   - *这引起了人们对不同架构执行效率的关注。*


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1313168030582308894)** (1 条消息): 

> `bf16 training, debugging tips` 


- **bf16-true 训练取得成功**：一名成员报告说，切换到 **bf16-true** 对他们的训练过程帮助极大，并表示现在训练情况良好。
   - 他们对之前分享的调试技巧表示感谢，这表明他们最初的问题得到了积极解决。
- **对调试帮助的感谢**：另一名成员认可了所提供的**调试技巧**的实用性，表示这些技巧简化了排错过程。
   - 这显示了社区对于分享有效训练的最佳实践和策略的强烈氛围。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1313245346725367940)** (26 messages🔥): 

> `MIT Efficient ML 课程, Stanford CS 229S 课程, ML 课程作业, 机器学习优化技术` 


- **针对高级技术的 MIT Efficient ML 课程**：来自 MIT 的 Han 教授提供了 [Efficient ML 课程](https://hanlab.mit.edu/courses/2024-fall-65940)，涵盖了量化 (quantization)、剪枝 (pruning) 和分布式训练 (distributed training) 等优化机器学习系统的关键主题。
   - 该课程还强调实际应用，允许学生练习模型压缩技术，并在资源受限的设备上部署像 Llama2-7B 这样的大型模型。
- **Stanford CS 229S 课程强调系统层面**：提到的另一个资源是 [Stanford CS 229S 课程](https://cs229s.stanford.edu/fall2023/)，该课程专注于机器学习系统，并包含各种编程练习。
   - 鼓励参与者探索作业和材料，以了解部署深度学习模型时的基础设施和生命周期挑战。
- **公开访问的实验和作业**：这些课程提供了托管在 Google Colab 上的公开实验和作业，在没有计算资源限制的情况下促进了更轻松的学习体验。
   - 参与者注意到，虽然有些作业具有挑战性，但他们非常欣赏课程材料中的组织和支持。
- **成功的知识前提**：讨论强调了 Han 教授提出的先验知识的重要性，并推荐了通过 OCW 和 GitHub 资源获取的先修课程。
   - 成员们一致认为，打好机器学习概念的坚实基础对于应对课程挑战至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cs229s.stanford.edu/fall2023/">Home</a>: Systems for Machine Learning</li><li><a href="https://hanlab.mit.edu/courses/2024-fall-65940">MIT 6.5940 Fall 2024 TinyML and Efficient Deep Learning Computing</a>: 未找到描述</li><li><a href="https://hanlab.mit.edu">MIT HAN Lab</a>: 欢迎来到 MIT HAN Lab，在这里，效率与性能相遇，创新与卓越在人工智能 (AI) 和计算机体系结构领域交汇。我们的实验室处于最前沿...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/)** (1 messages): 

mobicham: 这里的 3-bit 版本是对称的 (symmetric) 还是非对称的 (asymmetric)？
  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1313467416893849620)** (6 messages): 

> `适用于 AI/ML 的 Mastodon, Mastodon 上的 HPC 社区, Mastodon 概览` 


- **Mastodon 上的 GPU 社区存在感**：成员们讨论了 Mastodon 上是否有许多 **AI/ML** 领域的人，其中一人注意到那里有很多 **图形/计算/HPC** 爱好者，但不确定 AI 领域的情况。
   - *对那里网络情况的好奇是合理的*，这表明社区内的兴趣点可能各不相同。
- **关于 Mastodon 的询问**：有人提出了 *“什么是 Mastodon？”*，一名成员提供了一个 [Google 搜索链接](https://letmegooglethat.com/?q=mastodon) 而不是直接解释。
   - 这一回应被另一名成员视为不礼貌，强调在对话中寻求澄清是很自然的。
- **关于回应的社区礼仪**：一名成员指出，不应无视简单的问题，而应促进 **闲聊 (chit-chat)**。
   - 他们强调发送搜索链接可能会显得粗鲁，建议倾向于对话式的互动。



**提到的链接**：<a href="https://letmegooglethat.com/?q=mastodon">Mastodon</a>: 未找到描述

  

---

### **GPU MODE ▷ #[arm](https://discord.com/channels/1189498204333543425/1247232251125567609/1313232508967456888)** (3 messages): 

> `Low Bit ARM kernels, Low-bit operations, LUT techniques, Bitnet.cpp` 


- **关于 Low Bit ARM Kernels 的 YouTube 讲座**：对于错过最初演示的人，可以查看由演讲者 Scott Roy 带来的 [YouTube 视频：'Lecture 38: Low Bit ARM kernels'](https://youtu.be/2iNGuZxe1ms?si=iHtLdGN-vZl2_MMG)。
   - 幻灯片的具体细节尚待确定，但该讲座承诺会提供宝贵的见解。
- **对 Low Bit 技术的兴奋**：一位成员表达了热情，认为这是一个非常酷且值得进一步探索的**超级话题**。
   - *Low-bit operations* 在讨论中越来越受关注，表明兴趣日益增长。
- **提议类似于 Bitnet 的 LUT 方法**：有人建议通过 LUT 实现具有**真正 low-bit operations** 的想法，并参考 *Bitnet.cpp* 寻找灵感。
   - 这表明在 low-bit 处理讨论中，可能会向更高效的方法转变。



**提到的链接**：<a href="https://youtu.be/2iNGuZxe1ms?si=iHtLdGN-vZl2_MMG">Lecture 38: Low Bit ARM kernels</a>：演讲者：Scott Roy，幻灯片：待定

  

---


### **GPU MODE ▷ #[webgpu](https://discord.com/channels/1189498204333543425/1262121239044948009/1313290282904846427)** (1 messages): 

> `Performance Optimizations, TFLOP/s Metrics` 


- **探索性能优化**：一位成员表达了聊聊**潜在性能优化**的渴望，并暗示可能存在一些损害当前性能的做法。
   - *“我肯定我正在做一些愚蠢的事情，从而损害了性能。”*
- **以 1TFLOP/s 作为基准**：**1TFLOP/s** 被提及为一个似乎值得发布的里程碑，表明它是一个不错的整数目标，而非最终成就。
   - *“在那个点上，它似乎就值得发布了”* 😄。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1313167258872315907)** (6 messages): 

> `CUDARC Project, Luminal Framework, Talk Invitation` 


- **用于手动 CUDA 绑定的 CUDARC Crate**：[CUDARC](https://crates.io/crates/cudarc) Crate 提供了 CUDA API 的绑定，但由于是手动实现，目前仅支持**矩阵乘法**。
   - 测试表明，用户的大部分时间都花在优化这个 **matmul** 函数上。
- **Luminal 框架引发关注**：另一位用户强调了 [Luminal](https://github.com/jafioti/luminal)，这是一个类似的基于 Rust 的项目，并赞扬了不同语言中 ML 框架的增长。
   - 对话认可了在各种编程环境中扩展 ML 工具的积极趋势。
- **确认即将到来的演讲机会**：一位用户接受了明年年初进行演讲的邀请，并表达了热情。


  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1313573550548058133)** (3 messages): 

> `KernelBench introduction, Kernel performance evaluation, Leaderboard concerns` 


- **为 LLM 引入 KernelBench**：新的编码基准测试 **KernelBench (Preview)** 旨在评估 LLM 生成**高效 GPU kernels** 以优化神经网络性能的能力。它由 [@anneouyang](https://twitter.com/anneouyang/status/1864014135824162995) 在 Twitter 上正式介绍。
   - 初步反应显示了对其潜力的兴奋，用户们渴望探索其功能。
- **对 Kernel 排行榜完整性的担忧**：有人抱怨排行榜上某些**最快的 kernels** 似乎不完整，从而引发了对其准确性的质疑。一位用户引用了一个[不完整的 kernel 解决方案](https://raw.githubusercontent.com/ScalingIntelligence/KernelBenchLeaderboard/refs/heads/main/docs/assets/solutions/fc7b0633e1f8dca6653f552f2eeef450.py)，暗示了潜在的问题。
   - 这一讨论强调了对基准测试工具进行彻底评估的重要性。



**提到的链接**：<a href="https://github.com/ScalingIntelligence/KernelBench">GitHub - ScalingIntelligence/KernelBench</a>：通过在 GitHub 上创建账号来为 ScalingIntelligence/KernelBench 的开发做出贡献。

  

---

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1313598091684745227)** (1 条消息): 

> `WGMMA+TMA custom kernel, Race Condition in Kernel, Mask Implementation, Shared Memory Issues, Latest Fork Updates` 


- **在 WGMMA+TMA 中调查 Race Condition**：一名成员报告在通过自定义 Kernel 实现 WGMMA+TMA 时遇到了 **Race Condition**，并指出其矩阵的起始和结束位置非 16 字节对齐。
   - *Mask 并不总是被正确应用*，导致除非多次重复调用 Mask，否则会出现问题。
- **使用自定义逻辑屏蔽垃圾行 (Junk Rows)**：他们基于现有的加载函数创建了自定义 Mask 函数，以便在调用 WGMMA 之前**屏蔽垃圾行**，并调整了 row_start 和 row_end 参数。
   - 然而，该解决方案导致了非预期行为，Mask 并不总是按预期生效。
- **Shared Memory 和 Barrier 问题**：代码正确包含了 **__syncthreads()**，但在与其他函数（如 *kittens::warpgroup::mma_fence*）结合使用时，其有效性存疑。
   - 成员们正在探索同步方法的改变是否影响了 Kernel 的效率。
- **旧版 Fork 2024 的更新**：该实现基于 **2024 年 7 月 1 日**左右的一个较旧 Fork，可能缺少近期有关 Shared Memory 和 Barrier 的修复。
   - 建议用户考虑是否**基于最新版本重写**，以获得更好的稳定性和性能。
- **对 ThunderKittens 工具的赞赏**：发帖者表达了对 **ThunderKittens** 的喜爱，并称赞其相比其他替代方案更易于使用。
   - 他们强调团队的工作“非常酷”，对使用这些技术的开发者非常有帮助。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1313195887849111583)** (6 条消息): 

> `NVIDIA Financial Analysis, LlamaCloud Pipeline with Google Drive, Multi-Agent Meetup at GitHub, AI Apps on Vercel, Amazon's Nova Foundation Models` 


- **揭秘 NVIDIA 财务洞察**：在一段视频中，**@pyquantnews** 展示了如何利用 **NVIDIA** 的财务报表进行简单的营收数据查询，以及对业务风险和细分市场表现的复杂分析。查看更多关于 [设置 LlamaIndex 的实际代码示例](https://t.co/989lnoNeV2)。
   - 这种方法展示了如何利用财务报表进行更深层次的商业智能分析。
- **使用 Google Drive 优化 LlamaCloud 流水线**：在另一段视频中，**@ravithejads** 概述了设置以 **Google Drive** 作为数据源的 **LlamaCloud** 流水线的逐步过程，包括分块 (chunking) 和嵌入 (embedding) 的配置参数。在此发现完整设置 [here](https://t.co/KnQ9nUDWYE)。
   - 该指南对于希望轻松集成 LlamaIndex 文档索引的开发者来说是宝贵的资源。
- **Multi-Agent 见面会激发协作**：即将举行的 **GitHub 总部 Multi-Agent 见面会**将邀请专家讨论各种话题，包括使用 **CrewAI** 自动化任务以及使用 **Arize AI** 评估 Agent。了解更多活动信息 [here](https://t.co/VqmlVGnWT4)。
   - 此次见面会承诺探索 Agentic Retrieval 以及使用 **LlamaIndex** 构建应用的创新实践。
- **在 Vercel 上更轻松地进行 AI 开发**：随着 **LlamaIndex** 和 **LlamaCloud** 集成的增强，在 **Vercel** 上构建 AI 应用现在变得更加简单。更多详情请见 [此帖](https://t.co/nXen8N7cLf)。
   - 这一改进旨在简化使用 Vercel 的开发者的工作流程。
- **Amazon 发布 Nova 基础模型**：**Amazon** 推出了 **Nova**，这是一个极具竞争力的基础模型系列，与同类模型相比价格显著降低，并提供 Day 0 支持。使用 `pip install llama-index-llms-bedrock-converse` 进行安装，并查看示例 [here](https://t.co/KYZRdIIihI)。
   - 此次发布标志着 AI 模型领域的又一重要补充，承诺提供更高的性价比和性能。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1313177428453097563)** (43 条消息🔥): 

> `Embedding 模型限制, 季度报告生成, RAG 实现, 多模态模型的结构化输出, 工作流中的聊天历史管理` 


- **Embedding 模型受 Token 限制**: 一位成员询问如何处理超过 **512 tokens** 的 Embedding 模型输入文本，并指出 **HuggingFaceEmbedding** 类只是简单地截断了较长的输入。
   - 另一位成员强调了选择正确的 `embed_model` 类以在实践中避免这些限制的重要性。
- **创建详细的季度报告**: 一位用户表示需要从多份详细的月度财务报告中编译出一份 **4-6 页的季度报告**，要求高准确度且无幻觉。
   - 建议包括创建一个结合模板和 RAG 方法的结构化流程，以有效地综合见解。
- **重点介绍有效的 RAG 技术**: 一位成员分享了一个包含 **10 多种 RAG 实现**的综合仓库，包括 **Naive RAG** 和 **Hyde RAG** 等各种技术。
   - 这可以帮助其他人在针对其特定数据集和需求进行 RAG 应用实验时提供参考。
- **多模态模型结构化输出的挑战**: 讨论围绕 **MultiModalLLMCompletionProgram** 展开，该程序不使用 function calls，导致输出解析变得脆弱。
   - 成员们建议建立自定义的 function calling 接口，以实现更可靠的结构化数据输出。
- **在 Workflow 中管理聊天历史**: 在 **QueryPipeline**（该功能在此用途上更方便）被弃用后，一位用户寻求关于在 Workflow 步骤之间传递聊天历史的建议。
   - 建议包括利用 **Context** 进行状态管理，或使用 **ChatMemoryBuffer** 来处理跨会话的聊天历史。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/workflow/#working-with-global-contextstate">Workflows - LlamaIndex</a>: 未找到描述</li><li><a href="https://lu.ma/i8bow7sr">Voice &amp; Video AI Agents Hackathon · Luma</a>: Gen AI AgentsCreatorsCorner，与 AWS, Temporal, Modal, Tandem, Marly, Retell, Senso, Unified, Speedlegal, Corval, Simli, PolyAPI 等合作……</li><li><a href="https://github.com/athina-ai/rag-cookbooks">GitHub - athina-ai/rag-cookbooks: Cookbooks for LLM developers</a>: 面向 LLM 开发者的 Cookbooks。欢迎在 GitHub 上为 athina-ai/rag-cookbooks 贡献代码。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1313236877448318988)** (20 条消息🔥): 

> `LM Studio Windows 下载问题, LM Studio 在 Windows 上的性能, 社区支持与态度, Qwen LV 7B 模型功能` 


- **用户在下载 LM Studio 时遇到困难**: 一位用户报告在从 lmstudio.ai 下载 **Windows x86 版本**时遇到麻烦，在多个浏览器上都提示文件不可用。
   - 另一位用户建议这可能是 **CDN 问题**，并建议使用 VPN 更改位置进行下载。
- **LM Studio 在 Windows 上的性能问题**: 一位用户反映，与 Mac 相比，在 Windows 上运行 **LM Studio** 时性能较慢且输出异常，特别是在使用 **3.2 model** 时。
   - 其他人提出了解决方案，建议切换 `Flash Attention` 开关并检查系统配置的兼容性。
- **社区提供积极反馈**: 一位用户表达了感谢，注意到社区中的版主和开发者没有傲慢的行为。
   - 他们强调了建设性对话对改进产品的重要性，希望这种态度能保持下去。
- **具备视觉能力的 Qwen LV 7B**: 有人询问 **Qwen LV 7B 模型是否支持视觉**功能。
   - 这个问题引发了关于将视觉能力与各种 AI 模型集成的讨论。


  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1313224609331675247)** (15 messages🔥): 

> `Docker Containers for HF Spaces, Optimal GPU Configurations, FP8 Quantization in Models, Changing LLaMA.cpp Version, Intel Arc Battlemage Cards` 


- **HF Spaces 可作为 Docker 容器运行**：一位成员确认你可以将任何 **HF space** 作为 [docker container](https://link.to/docker) 运行。
   - 这为本地部署和测试提供了灵活性。
- **组装 GPU 设备的建议**：一位成员建议购买二手 **RTX 3090** 并将 RAM 升级到 64GB，并强调现在 CPU 的重要性已有所降低。
   - 他们强调了合适的电源供应以及为未来潜在升级提供兼容性的重要性。
- **FP8 量化提升模型效率**：根据 [VLLM docs](https://docs.vllm.ai/en/v0.6.2/quantization/fp8.html)，FP8 量化可使模型显存占用**减少 2 倍**，**吞吐量**提升 1.6 倍，且对准确率影响极小。
   - 对于那些希望在资源有限的机器上优化性能的用户来说，这可能特别具有参考价值。
- **更改 LLaMA.cpp 版本的限制**：关于更改 LM Studio 使用的 **llama.cpp** 版本的讨论，成员确认目前无法实现。
   - 有趣的是，一位成员注意到有一个新的 CUDA 版本可用，与之前的版本形成对比。
- **对 Intel Arc Battlemage 显卡的怀疑**：一位成员对新款 **Arc Battlemage 显卡**表示怀疑，认为它们不适合 AI 任务。
   - 另一位成员认为，尽管它们在构建**本地推理服务器**方面具有成本效益，但他们仍然不会在这些应用中信任 Intel。



**提及的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1h5h3lp/can_i_change_the_llamacpp_version_used_by_lm/">Reddit - Dive into anything</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1313220070180130909)** (2 messages): 

> `Sierra AI Info Session, Recruitment Opportunities at Sierra` 


- **Sierra AI 为开发者举办信息说明会**：领先的对话式 AI 平台 Sierra 将于 **PT 时间 12/3 上午 9 点**举行**独家信息说明会**，可通过 [YouTube 直播](https://youtube.com/live/-iWdjbkVgGQ?feature=share) 观看。参与者将深入了解 Sierra 的功能和职业机会。
   - 会议将涵盖 **Sierra 的 Agent OS** 和 **Agent SDK** 等主题，重点介绍大规模部署 AI Agent 的经验教训。
- **Sierra AI 寻找人才**：Sierra 正在寻找**优秀开发者**加入其团队，并将在信息说明会期间讨论令人兴奋的职业机会。感兴趣的人员请尽早 [RSVP](https://lu.ma/agents-hackathon-sierra) 以预留直播席位。
   - *“不要错过这次与 Sierra 建立联系并探索 AI Agent 未来的机会！”*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lu.ma/agents-hackathon-sierra">与 Sierra AI 见面：LLM Agents MOOC Hackathon 信息说明会 · Luma</a>：与 Sierra AI 见面：LLM Agents MOOC Hackathon 信息说明会。关于说明会：🔗 直播链接：https://youtube.com/live/-iWdjbkVgGQ?feature=share 加入……</li><li><a href="https://www.youtube.com/watch?v=-iWdjbkVgGQ">LLM Agents MOOC Hackathon - Sierra 信息说明会</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-announcements](https://discord.com/channels/1280234300012494859/1280369709623283732/1313218519743205437)** (2 条消息): 

> `结课讲座与展示、课程结业证书、测验提醒、课程网站资源` 


- **关于 AI Safety 的结课讲座**：今天 **PST 时间下午 3:00**，**Dawn Song** 教授将在结课讲座中发表题为 *《迈向构建安全可信的 AI Agents 以及基于科学和证据的 AI 政策之路》* 的演讲，可通过 [直播](https://www.youtube.com/live/QAgR4uQ15rc) 观看。
   - 她将探讨与 **LLM agents 相关的重大风险**，并提出一种**基于科学的 AI 政策**，以有效缓解这些威胁。
- **课程结业证书流程**：学生必须填写证书申报表才能获得课程结业证书，该表格将在今天的讲座后不久发布。
   - 请确保所有作业均使用相同的电子邮件地址提交，因为这是跟踪进度的依据。
- **即将到来的测验和截止日期**：测验 **11 和 12** 将于本周初发布，所有作业的截止日期为 **PST 时间 12 月 12 日晚上 11:59**。
   - 对于参与者，**Hackathon 项目**的截止日期定为 **PST 时间 12 月 17 日晚上 11:59**。
- **在线课程资源**：所有课程材料（包括直播链接和作业）均可在 [课程网站](http://llmagents-learning.org/f24) 上访问。
   - 鼓励学生通过指定频道与课程工作人员沟通，提出任何问题或反馈。



**提到的链接**: <a href="https://www.youtube.com/live/QAgR4uQ15rc."> - YouTube</a>: 未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1313149688291135548)** (18 条消息🔥): 

> `LLM Agents 学习课程、复盘作业、实验作业要求、文章撰写作业、社交媒体分享指南` 


- **LLM Agents 学习课程报名**：成员确认参与者仍可通过填写 [报名表](https://docs.google.com/forms/d/e/1FAIpQLSeBoUgiNGyB8pdRZdLp076wpn4LkUxzEt9uviKZTbCSSv-aBA/viewform) 注册 **LLM Agents 学习课程**。有关课程的更多详情，可以访问 [此课程网站](https://llmagents-learning.org/f24)。
   - 欢迎参与者加入，并建议通过参加测验和可能加入 Hackathon 来增强学习体验。
- **复盘作业（Post-mortem Assignment）说明**：对于复盘作业，已明确该作业主要基于参与度，允许参与者在 **500 字**内自由讨论主题。评分预计会比较宽松，以减轻围绕该作业的压力。
   - 这种鼓励旨在培养一种轻松的评估方式，同时确保学生参与到材料中。
- **实验作业与认证**：已确认虽然实验作业对所有认证并非**强制性**，但完成全部三个实验作业是获得 **Mastery 等级**所必需的。对于较晚加入的参与者，已向其保证仍可以赶上进度。
   - 这种结构有助于学习者根据自己的需求选择参与课程内容的程度。
- **文章撰写作业指南**：对于文章撰写作业，建议参与者将最终草稿直接粘贴到指定的 Google Form 字段中。同时提醒他们在 LinkedIn 帖子中链接回课程网站。
   - 官方表示撰写关于整个课程的内容也是可以接受的，这提供了一个反思整体学习体验的机会。
- **社交媒体分享选项**：如果不使用 Twitter 或 Threads，成员可以将社交媒体分享发布到 **Mastodon** 等平台，这被视为是可以接受的。在粘贴文章文本后，应在相应字段中添加帖子链接。
   - 这种分享媒介的灵活性体现了社区对沟通包容性的承诺。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>: MOOC, 2024 秋季</li><li><a href="https://docs.google.com/forms/d/e/1FAIpQLSeBoUgiNGyB8pdRZdLp076wpn4LkUxzEt9uviKZTbCSSv-aBA/viewform">LLM Agents MOOC 报名表</a>: 感谢您对本课程的关注！填写完表格后，您将通过电子邮件收到一份回复副本。如果您之后想更改回复，请填写...
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1313284650776662107)** (1 messages): 

> `GPT-4 PII leaks, AOL search log release` 


- **GPT-4 潜在的 PII 泄露风险**：一位成员对 **GPT-4 泄露个人身份信息 (PII)** 表示担忧，并将其与历史事件进行了类比。
   - 他们引用了 2006 年的 [AOL search log release](https://en.wikipedia.org/wiki/AOL_search_log_release) 事件，当时用户身份通过清理不彻底的数据被识别了出来。
- **AOL 臭名昭著的数据事故**：**AOL** 在 2006 年发布了一个包含来自超过 65 万名用户的 **2000 万条搜索查询**的数据集，尽管最初声称已匿名化，但其中仍包含了 PII。
   - 虽然 AOL 随后很快撤回了数据，但该数据已被广泛复制，至今仍可在网上获取。



**提到的链接**：<a href="https://en.wikipedia.org/wiki/AOL_search_log_release">AOL search log release - Wikipedia</a>：未找到描述

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1313260009609760790)** (5 messages): 

> `ReAct Paradigm, Implementation Quality, Benchmark Evaluations` 


- **ReAct 的有效性取决于实现方式**：讨论指出 **ReAct 范式** 的有效性因 **Prompt 设计** 和 **状态管理** 等实现细节而异。一位成员指出，Benchmark 应该反映具体的实现，而不是被视为对框架本身的通用评估。
- **实现方式与基础模型的对比**：这被比作传统 ML 中**数据集大小/质量**与 LLM 框架中**基础模型 (Foundation Models) 大小/质量**之间的关系。这一类比引发了关于不同实现如何导致 Benchmark 性能产生实质性差异的讨论。
- **AI 领域的定义仍然模糊**：成员们表示，AI 领域的定义，特别是关于 **ReAct** 及其 Benchmark 的定义，目前相当**模糊**。这种缺乏清晰度的情况为评估该范式的有效性增加了另一层复杂性。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1313194886027280394)** (21 messages🔥): 

> `Development Branch Updates, OpenAI Compatibility, Usage Issues with Anthropic, Testing Requests, Linux OS Compatibility` 


- **开发分支获得好评**：最新的开发分支经过了彻底重写，使其更**轻量、快速且智能**，给用户留下了深刻印象。
   - 成员们对测试这个活跃分支表现出极大的热情，并鼓励大家针对旧版本中缺失的功能提供反馈。
- **Anthropic 集成面临挑战**：有报告称在 Anthropic 中使用开发分支时出现问题，具体是与意外的关键字参数 ‘proxies’ 相关的 `TypeError`。
   - 建议用户分享完整的 Traceback 以便更好地调试，并提供了正确的安装命令示例。
- **OpenAI 服务器启动**：新的 `--serve` 选项允许建立一个兼容 OpenAI 的 REST 服务器，尽管 1.0 版本排除了旧的 LMC/web socket 协议。
   - 这种新设置允许用户通过任何兼容 OpenAI 的客户端进行连接，从而在服务器设备上执行操作。
- **征集测试以提升性能**：成员们请求社区参与测试，以增强频繁更新的开发分支的功能。
   - 一位成员表示他们依赖 LMC 进行通信，发现过渡到新设置既**令人畏惧又令人兴奋**。
- **Garuda-Linux 成功案例**：在关于各种 Linux 发行版的讨论中，一位成员确认在 **Garuda-Linux**（一个 Arch Linux 衍生版）上成功使用了开发分支。
   - 他们列举了尝试过的其他几种发行版，包括 Manjaro、Mint 和 Kali，展示了跨不同系统的广泛兼容性。



**提到的链接**：<a href="https://tenor.com/view/so-close-this-the-office-gif-1505267913606309297">So Close GIF - So Close This - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---

### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1313557411969241088)** (1 条消息): 

> `LiveKit 连接、设备交互、本地 OpenInterpreter 操作` 


- **LiveKit 桥接设备**: O1 通常使用 **LiveKit** 来连接两个设备，例如你的 iPhone 和运行服务器的笔记本电脑或 **Raspberry Pi**。
   - 这种设置实现了通过在其上运行的本地 OI 实例进行**远程访问**以控制机器。
- **使用 O1 增强计算**: O1 的能力允许更强大的 **computer use**，与 CLI 形式相比，它是一个更复杂的工具。
   - 尽管增加了额外功能，**CLI** 形式的 OI 仍然完全能够有效地操作计算机。


  

---


### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1313568775341080648)** (3 条消息): 

> `Pydantic AI, DSLModel, AI 开发, Pydantic Logfi, 现场演示` 


- **Pydantic AI 与 DSLModel 集成**: [Pydantic AI](https://ai.pydantic.dev/) 的引入增强了与 DSLModel 的集成，为开发者创建了一个无缝框架。
   - 这种集成利用了 **Pydantic**，它在 Python 的各种 Agent 框架和 LLM 库中被广泛使用。
- **DSLModel 引起用户共鸣**: [DSLModel](https://pypi.org/project/dslmodel/) 受到了热烈欢迎，允许通过 Pydantic 集成进行高效开发。
   - 鼓励用户通过 **pip install dslmodel** 安装以开始使用该框架。
- **AI 开发现场演示**: 分享了一个名为“Master AI Development: PydanticAI + DSPy + DSLModel”的 **YouTube 现场演示**，提供了对这些工具的实时探索。
   - 该演示旨在展示尖端的 AI 开发技术（[在此观看](https://youtube.com/live/mBQFKo8bPBI)）。
- **社区对 Pydantic 的参与**: 社区对在项目中实现源自 **Pydantic** 的高级功能表示兴奋。
   - 详细讨论强调了展示其实际应用的各种用例。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ai.pydantic.dev/">Introduction</a>: 用于将 Pydantic 与 LLM 结合使用的 Agent 框架 / 填充层</li><li><a href="https://pypi.org/project/dslmodel/">dslmodel</a>: 从 prompt 和 Jinja 生成 Pydantic + DSPy 实例。</li><li><a href="https://youtube.com/live/mBQFKo8bPBI">Master AI Development: PydanticAI + DSPy + DSLModel Deep Dive (Live Demo)</a>: https://ai.pydantic.dev/https://dspy.ai/https://pypi.org/project/dslmodel/🚀 加入我们的直播，探索 AI 开发的前沿！发现如何...
</li>
</ul>

</div>
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1313549828353032222)** (7 条消息): 

> `优化时长, AWS Lambda 上的 DSPy, Program Of Thought 弃用` 


- **优化过程可能耗时更长**: 一位成员指出，运行**优化**绝对可能超过 **15 分钟**，特别是如果程序包含多个步骤。
   - 这表明时长会根据复杂程度而有显著差异。
- **考虑在 AWS Lambda 上进行 DSPy 优化**: 一位成员正在考虑为 **LangWatch** 客户在 **AWS Lambda** 上运行 **DSPy 优化**，但 **15 分钟的限制**带来了挑战。
   - *他们表示需要绕过这一时间限制的策略。*
- **推荐使用 ECS/Fargate 而非 Lambda**: 另一位成员分享了他们的经验，建议由于**存储限制**，在 **Lambda** 上运行 **DSPy** 可能不可行。
   - 他们建议探索 **ECS/Fargate** 作为一种可能更可靠的解决方案。
- **关于 Program Of Thought 支持的查询**: 一位成员询问 **Program Of Thought** 是否在 **v2.5** 之后走向**弃用/无积极支持**的道路。
   - 这表明社区对该程序未来的持续关注。

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1313241677015679029)** (9 条消息🔥): 

> `DSPy 中的 Agentic 示例, DSPy 中的 RAG 示例, Codetree 快速版本, DSPy Module Class` 


- **寻找 Agentic 示例**：一位成员询问了 DSPy 中关于 Agentic 的示例，即一个 Signature 的输出被用作另一个 Signature 的输入，特别是针对电子邮件撰写程序。
   - 另一位成员建议查看 [RAG 示例](https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb)，但随后澄清相关示例的位置可能在 [dspy.ai 网站](https://dspy.ai)上。
- **Codetree 的极速版本**：一位成员分享了一个带有 Few-shots (k=1) 标记的 'codetree' 快速版本，并指出该版本尚未经过优化。
   - 该示例可以在[这里](https://gist.github.com/fullstackwebdev/a5025628613752449599f77ea3330fd1)找到，并参考了包含附录 A 的相关[论文](https://arxiv.org/pdf/2411.04329)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/stanfordnlp/dspy/blob/main/examples/llamaindex/dspy_llamaindex_rag.ipynb">dspy/examples/llamaindex/dspy_llamaindex_rag.ipynb at main · stanfordnlp/dspy</a>：DSPy：用于编程（而非提示）语言模型的框架 - stanfordnlp/dspy</li><li><a href="https://gist.github.com/fullstackwebdev/a5025628613752449599f77ea3330fd1">codetree.py</a>：GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1313315100651950161)** (4 条消息): 

> `Torchtune 中的图像生成, T5 集成, 微调模型` 


- **Torchtune 即将推出图像生成功能？**：一位用户对 Torchtune 可能集成 **图像生成 (image generation)** 功能的可能性感到兴奋，并引用了 [Pull Request #2098](https://github.com/pytorch/torchtune/pull/2098)。
   - 该 PR 的详细信息表明，其旨在为平台添加**新功能**。
- **T5 在即将推出的功能中的作用**：讨论暗示，根据 [Pull Request #2069](https://github.com/pytorch/torchtune/pull/2069) 的见解，**T5** 也可能被包含在 Torchtune 中。
   - 用户暗示 T5 的功能与 **图像生成** 集成计划是一致的。
- **微调 ImageGen 模型的乐趣**：一位用户对在 Torchtune 中微调 **图像生成模型** 的潜力表示热切期待，认为这将是一项有趣的尝试。
   - 这一评论引发了轻松的回应，表明成员们对该话题的熟悉程度参差不齐。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/torchtune/pull/2098">Flux Autoencoder by calvinpelletier · Pull Request #2098 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue。Changelog...</li><li><a href="https://github.com/pytorch/torchtune/pull/2069">T5 Encoder by calvinpelletier · Pull Request #2069 · pytorch/torchtune</a>：上下文：此 PR 的目的是什么？是添加新功能、修复 Bug、更新测试和/或文档，还是其他（请在此处添加）。请链接此 PR 解决的任何 Issue。Changelog...
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/)** (1 条消息): 

pjbontrager: 这将是一个有趣的 Recipe：https://sakana.ai/cycleqd/
  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1313327793664430142)** (4 条消息): 

> `活动出席, 注册流程, 访问印度` 


- **对即将举行的活动感到兴奋**：成员们表达了对参加即将举行的活动的兴奋和兴趣，其中一人提到他们计划在那段时间访问印度。
   - *“噢太棒了。是的。我会在那里。希望能见面！”*
- **咨询活动注册**：一位用户询问了注册成为活动参与者的流程。
   - 这引发了关于如何有效进行注册流程的讨论。

### **Axolotl AI ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1313611788482514945)** (1 messages): 

> `Office Hours 公告、Axolotl 调查、Swag 赠送` 


- **即将举行的 Office Hours：加入我们！**：请在日历上标记 **12/5 周四** **东部时间下午 1 点 / 太平洋时间上午 10 点**，我们将讨论 Axolotl。
   - *我们很高兴能与大家交流！*
- **通过 Axolotl 调查分享您的意见**：为了帮助改进 Axolotl，我们邀请您填写公告中链接的 **Axolotl Survey**。
   - 您的反馈对我们至关重要，通过参与调查，您可以帮助我们根据您的需求定制支持方案。
- **完成调查获取专属 Swag**：作为完成调查的感谢，参与者将获得**即将发布的 Axolotl swag**（送完即止！）。
   - *我们非常珍惜您的时间，* 请不要错过这个机会！



**Link mentioned**: <a href="https://gravel-salmon-db9.notion.site/1421d2ab4f4081168f6fe3770fae446c">Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.</a>: 一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一站式工作空间。

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1313557490226565193)** (3 messages): 

> `ADOPT 优化器更新、Axolotl 代码库` 


- **ADOPT 优化器最新更新已发布**：团队宣布已将 **ADOPT 优化器** 的最新更新集成到 [Axolotl 代码库](https://github.com/axolotl-ai-cloud/axolotl/pull/2104) 中，并邀请成员进行尝试。
   - 一位成员询问了该优化器的优势。
- **任意 beta 值下的最优收敛**：**ADOPT 优化器** 可以在任何 beta 值下实现**最优收敛 (optimal convergence)**，从而优化各种场景下的性能。
   - 在成员们讨论最近集成的过程中，这一能力脱颖而出。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

jewnex: PR#7987 值得发条推特，运行一些 benchmarks，这次 beam 没有 GPU 挂死 (gpu hang) 🚀
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1313628553568321596)** (1 messages): 

> `图重写中的线程组/网格大小、uopgraph.py 中的优化` 


- **关于更改线程组/网格大小的问题**：一位成员询问在 `uopgraph.py` 的图重写优化过程中是否可以更改 **thread group/grid sizes**。
   - 他们很好奇，当通过 **pm_lowerer** 降低大小时，这些大小是否已经根据流程中早期的某些搜索**固定**了。
- **关于图优化过程的说明**：讨论围绕理解 **uopgraph.py** 在优化阶段如何处理线程组大小展开。
   - 成员们对是否可以在优化后进行调整，以及初始搜索是否决定了最终大小表示关注。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1313300365663277076)** (1 messages): 

> `bio-ML 进展、Gene Diffusion 模型、机械可解释性 (mechanistic interpretability)、蛋白质序列建模、自监督学习` 


- **2024 年的 Bio-ML 革命**：**2024** 年标志着生物机器学习 (**bio-ML**) 的激增，取得了诸如结构生物学预测获得**诺贝尔奖**以及蛋白质序列模型的大规模投资等显著成就。
   - 尽管该领域兴奋不已，但对于需要解决的计算最优 (compute-optimal) 蛋白质序列建模曲线仍存在担忧。
- **为单细胞生物学引入 Gene Diffusion**：描述了一种名为 **Gene Diffusion** 的新模型，该模型利用在单细胞基因计数数据上训练的连续扩散 Transformer 来探索细胞功能状态。
   - 它采用自监督学习方法，从基因 token 向量中预测干净、无噪声的嵌入 (embeddings)，类似于文本生成图像 (text-to-image) 模型中使用的技术。
- **寻求 Gene Diffusion 训练机制的澄清**：成员们对 **Gene Diffusion** 模型的训练机制感到好奇，特别是其输入/输出关系以及它的预测目标。
   - 成员们表达了对模型复杂细节进行澄清的愿望，强调需要社区协助来理解这些复杂的概念。



**Link mentioned**: <a href="https://www.markov.bio/research/mech-interp-path-to-e2e-biology">Through a Glass Darkly | Markov Bio</a>: 通往端到端生物学的道路是怎样的，人类的理解在其中扮演什么角色？

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1313195704977723453)** (1 条消息): 

> `十二月活动日程、下一代 Llamafile Hackathon、Web Applets 介绍、Theia IDE 演示、Llamafile 更新` 


- **十二月活动日程公布**：[十二月日程表](https://discord.com/channels/1089876418936180786/1089876419926032396/1311366440997355654)中新增了三项成员活动，以保持大家的参与度。
   - 这些活动旨在增强社区参与感，并展示成员的项目。
- **下一代 Llamafile Hackathon 明日演示**：学生们明天将展示他们使用 [Llamafile 实现个性化 AI](https://discord.com/events/1089876418936180786/1313249962582212708) 的 Hackathon 项目，重点关注社会公益。
   - 鼓励社区成员加入并支持学生们的创新努力。
- **Web Applets 介绍**：<@823757327756427295> 将[介绍 Web Applets](https://discord.com/events/1089876418936180786/1311466292753989672)，详细讲解用于创建高级客户端应用程序的开放标准和 SDK。
   - 参与者可以通过在社区内自定义角色来选择接收更新。
- **Theia IDE 实操演示**：<@1131955800601002095> 将展示 [Theia IDE](https://discord.com/events/1089876418936180786/1311841242262540298) —— 一个开放且灵活的 AI 驱动开发环境。
   - 此演示旨在说明 Theia 如何促进更好的开发实践。
- **Llamafile 新版本发布与安全赏金**：宣布了 [Llamafile 的新版本](https://discord.com/channels/1089876418936180786/1262961704602570832/1312634808785965066)，提供了软件改进的更新。
   - <@&1245781246550999141> 在第一个月发放了 **42 项赏金**，重点在于揭露生成式 AI 的漏洞。


  

---


---


---


{% else %}


> 完整的频道细分内容已针对邮件进行截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}