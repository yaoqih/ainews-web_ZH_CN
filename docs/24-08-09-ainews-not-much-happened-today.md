---
companies:
- anthropic
- google
- mistral-ai
- llamaindex
date: '2024-08-10T05:51:12.824379Z'
description: '以下是为您翻译的中文内容：


  **Qwen2-Math-72B** 凭借合成数据和先进的优化技术，在数学基准测试中超越了 **GPT-4o**、**Claude-3.5-Sonnet**、**Gemini-1.5-Pro**
  以及 **Llama-3.1-405B**。**Google AI** 将 **Gemini 1.5 Flash** 的价格大幅下调，降幅最高达 78%。**Anthropic**
  扩大了其漏洞赏金计划，重点针对下一代安全系统中的通用越狱（jailbreaks）问题。针对 **IDEFICS3-Llama 8B** 进行 **QLoRA**
  微调以实现视觉问答（VQA）的教程现已发布。一款中国开源权重模型打破了此前 MATH 基准测试的记录。关于 **Mamba** 模型以及用于软件工程的 **LLM
  智能体** 的综述报告突出了相关领域的进展与应用。**R2R RAG 引擎** 和 **LlamaIndex Workflows** 等开源工具简化了复杂 AI
  应用的构建过程。**Mistral AI** 推出了可定制的 AI 智能体。此外，加州 **SB 1047** 法案对“生存风险”的关注引发了担忧，关于是否禁止开源
  AI 的辩论也在持续。AI 社区中依然充满了各种梗和幽默。'
id: 6d6bf91e-0eae-41b6-9792-5162e266a599
models:
- qwen2-math-72b
- gpt-4o
- claude-3.5-sonnet
- gemini-1.5-pro
- llama-3.1-405b
- idefics3-llama-8b
original_slug: ainews-to-be-named-3898
people:
- rohanpaul_ai
- anthropicai
- mervenoyann
- jeremyphoward
- omarsar0
- ylecun
- bindureddy
title: 今天没什么事。
topics:
- math
- fine-tuning
- synthetic-data
- reinforcement-learning
- bug-bounty
- visual-question-answering
- open-source
- retrieval-augmented-generation
- agentic-ai
- ai-safety
- policy
---

<!-- buttondown-editor-mode: plaintext -->**宁静的一周正是你所需要的。**

> 2024年8月8日至8月9日的 AI 新闻。我们为你检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **28** 个 Discord 社区（**249** 个频道，以及 **2549** 条消息）。预计节省阅读时间（按 200wpm 计算）：**278 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来进行 AINews 讨论！

与大多数新闻机构不同，当没有太多事情发生时，我们不会寻求或必须用内容来填充页面。本周最大的新闻是降价和 structured outputs。祝贺 Cursor AI 宣布其 [6000 万美元的 A 轮融资](https://techcrunch.com/2024/08/09/anysphere-a-github-copilot-rival-has-raised-60m-series-a-at-400m-valuation-from-a16z-thrive-sources-say/)。我们一直是 [Composer](https://x.com/shaoruu/status/1812412514350858634) 的忠实粉丝。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型更新与进展**

- **Qwen2-Math 模型**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821615977332682929) 报道称，Qwen2-Math-72B 在多项数学基准测试中超越了 GPT-4o、Claude-3.5-Sonnet、Gemini-1.5-Pro 和 Llama-3.1-405B。该系列模型基于 Qwen2，在数学网页文本、书籍、考试和代码上进行训练，利用了合成数据以及拒绝采样（rejection sampling）和群体相对策略优化（group relative policy optimization）等先进技术。

- **Google AI 定价**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821653314066399688) 分享了 Google AI 大幅下调 Gemini 1.5 Flash 价格的消息，对于 128K tokens 以下的 prompt，输入价格下调 78% 至 $0.075/100万 tokens，输出价格下调 71% 至 $0.3/100万 tokens。

- **Anthropic 漏洞赏金计划**：[@AnthropicAI](https://twitter.com/AnthropicAI/status/1821533729765913011) 宣布扩大其漏洞赏金计划（bug bounty program），重点在于寻找其下一代安全系统中的通用越狱（jailbreaks）方法。他们为包括网络安全在内的各个领域的各种新型漏洞提供奖励。

- **IDEFICS3-Llama 微调**：[@mervenoyann](https://twitter.com/mervenoyann/status/1821605881815147004) 分享了一个关于在 VQAv2 上对 IDEFICS3-Llama 8B 进行 QLoRA 微调的新教程，展示了视觉问答（VQA）的高效微调技术。

**AI 研究与基准测试**

- **中国开源权重模型**：[@jeremyphoward](https://twitter.com/jeremyphoward/status/1821557546102174194) 提到了一款中国开源权重模型，在 MATH 基准测试中超越了以往所有的闭源和开源模型。

- **Mamba 综述**：[@omarsar0](https://twitter.com/omarsar0/status/1821556218168549561) 分享了一份关于 Mamba 的综述，对现有基于 Mamba 的模型在各领域和任务中的表现进行了系统性回顾，重点关注了进展、适配技术以及 Mamba 表现优异的应用场景。

- **用于软件工程的基于 LLM 的 Agent**：[@omarsar0](https://twitter.com/omarsar0/status/1821549401866686604) 重点介绍了一篇关于软件工程中基于 LLM 的 Agent 的当前实践和解决方案的综述论文，涵盖了需求工程、代码生成和测试生成等主题。

**AI 工具与平台**

- **R2R RAG 引擎**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1821524844137091459) 讨论了 R2R，这是一个开源的 RAG 引擎，简化了 RAG 应用的开发，提供多模态支持、混合搜索和自动知识图谱生成等功能。

- **LlamaIndex Workflows**：[@llama_index](https://twitter.com/llama_index/status/1821575082516660440) 推出了 Workflows，这是一种用于构建复杂 Agentic 生成式 AI 应用的新抽象，并演示了如何使用该功能重建 LlamaIndex 内置的子问题查询引擎（Sub-Question Query Engine）。

- **Mistral AI Agent**：[@sophiamyang](https://twitter.com/sophiamyang/status/1821476909345128740) 宣布推出 Mistral AI Agent，允许用户基于 Mistral 模型或微调模型构建 Agent，以便在 Le Chat 上使用。

**AI 安全与监管**

- **加州法案 SB 1047**：[@ylecun](https://twitter.com/ylecun/status/1821650987926339955) 分享了众议院民主党议员 Zoe Lofgren 对加州 SB 1047 法案的担忧，指出该法案“严重偏向于应对生存风险（existential risk）”。

- **开源 AI 辩论**：[@bindureddy](https://twitter.com/bindureddy/status/1821656181833924752) 发起了一场关于禁止开源 AI 的讨论，强调了此类提议引发的争议。

**梗与幽默**

- **Heavenbanning Day**：[@nearcyan](https://twitter.com/nearcyan/status/1821663160396607670) 开玩笑说两年后的“Heavenbanning Day”，随后又发推澄清“heavenbanning 并不存在，因为什么都没发生”。

- **故事点批评**：[@svpino](https://twitter.com/svpino/status/1821528301061493215) 分享了对敏捷开发中故事点（story points）的幽默批评，将其比作“皇帝的新衣”，并称这种做法是一场“闹剧”。

- **AI 赞美**：[@AmandaAskell](https://twitter.com/AmandaAskell/status/1821575345671446531) 开玩笑地建议给未来的 AI 发推赞美，以博取它们的好感。

这份摘要涵盖了 AI 模型开发、研究、工具、安全和监管方面的关键讨论，以及一些关于 AI 和软件开发实践的幽默见解。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. 用于数学和技术任务的专业 AI 模型**

- **Qwen2-Math | 基于 Qwen2 的数学专用模型系列** ([Score: 73, Comments: 19](https://reddit.com//r/LocalLLaMA/comments/1en7tt8/qwen2math_mathspecific_model_series_based_on_qwen2/)): **Qwen** 发布了一系列基于其 **Qwen2** 架构的 **数学专用模型**，可在 **Hugging Face** 上获取。该系列包括各种规模的模型（**72B**、**7B** 和 **1.5B** 参数），提供 base 和 instruct-tuned 版本，旨在增强数学推理能力。

- **从零开始实现 LLaMA 3.1 8B 的 function calling，一些挑战与反馈！** ([Score: 60, Comments: 17](https://reddit.com//r/LocalLLaMA/comments/1enr1ug/implemented_llama_31_8bs_function_calling_from/)): 作者使用 **LlamaCPP Python 绑定**的 **generate() 函数**为 **LLaMA 3.1 8B** 实现了 **function calling**，并指出在从对话中分离自定义函数调用时面临的挑战。他们观察到像 LLaMA 3.1 8B 这样的小模型在没有特定指令的情况下很难使用工具，并表示由于 **token** 效率的原因，在 function calling 中更倾向于使用 **YAML** 而非 **JSON**。文章最后，作者考虑开发一个 **REST server** 来流式传输 **raw tokens**，或者为该功能提交功能请求。
  - 由于 **token** 效率和可读性，在 function calling 中 **YAML** 优于 **JSON**。用户讨论了让模型以 YAML 格式响应的 **prompting** 技巧，但提醒 **LLaMA 3.1 8B** 可能难以处理复杂的指令。
  - 人们对生成 **raw tokens** 和 **前 200 个 token 分布概率**的端点表现出浓厚兴趣，这可以实现一些巧妙的应用，但目前很难从现有的推理引擎中获取。
  - 用户将 **Gemma2** 与 **LLaMA 3.1** 进行了比较，一些人认为 Gemma2 更胜一筹。然而，有人指出 **Gemma2** 目前在 **Ollama** 等框架中不支持 **function calling**，限制了其在某些应用中的使用。


**Theme 2. Hugging Face 的战略扩张与开源 TTS 的进展**

- **[AI 独角兽 Hugging Face 收购一家初创公司，最终将托管数亿个模型 | 福布斯](https://www.forbes.com/sites/richardnieva/2024/08/08/hugging-face-xethub-acquisition/)** ([Score: 200, Comments: 43](https://reddit.com//r/LocalLLaMA/comments/1en5p3h/ai_unicorn_hugging_face_acquires_a_startup_to/)): **Hugging Face** 是一家估值 **45 亿美元**的 **AI 独角兽**，它收购了专注于 **AI 基础设施和云计算**的初创公司 **Paperspace**。此次收购旨在增强 Hugging Face 的能力，使其有可能托管 **数亿个 AI 模型**，并与 **Amazon**、**Google** 和 **Microsoft** 等主要云提供商竞争。此举是 Hugging Face 战略的一部分，旨在成为一个全面的 AI 开发和部署平台，提供从模型训练到推理的服务。

- **改进的文本转语音模型：Hugging Face 的 Parler TTS v1** ([Score: 111, Comments: 35](https://reddit.com//r/LocalLLaMA/comments/1encx98/improved_text_to_speech_model_parler_tts_v1_by/)): **Hugging Face** 发布了 **Parler TTS v1**，这是一款改进的开源 **Text-to-Speech 模型**，提供 **885M (Mini)** 和 **2.2B (Large)** 版本。该模型基于 **45,000 小时**的公开语音数据训练，生成速度提升高达 **4 倍**，支持 **SDPA** 和 **Flash Attention 2** 以提高速度，包含内置流式传输，并允许在自定义数据集上进行微调，在十几个说话者之间具有更好的说话者一致性。

**Theme 3. 新兴 AI 模型与性能基准测试**

- **为 Deepseek v2 点赞** ([Score: 56, Comments: 34](https://reddit.com//r/LocalLLaMA/comments/1enb0p2/shout_out_to_deepseek_v2/)): **Deepseek v2** 是一款拥有 **2000 亿参数的开源模型**，因其在编码任务中的表现而受到赞誉，可与顶级模型媲美，并在 **BigCodeBench** 上与 3.5 Sonnet 并列 **第 3 名**。该模型的 API 价格极具竞争力，**缓存命中率为每百万 token 0.017 美元**，用户仅需 **3.13 美元**即可处理 **6600 万个输入 token**。此外，该模型的效率表明它可以在 **四卡 3090 GPU 设置**上本地运行，使其成为开发者和研究人员的一个极具吸引力的选择。

- **LMSYS 上的新 sus-column-r 模型。简直太离谱了** ([Score: 62, Comments: 49](https://reddit.com//r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/)): 据报道，LMSYS 上的 **sus-column-r 模型**在**翻译**、**编程**、**数学**和回答**冷门问题**等各项任务中表现优于 **GPT-4** 和 **Claude 3.5 Sonnet**。帖子作者对该模型的能力表示难以置信，指出如果不是因为模型的自我识别回答，他会以为这是 **GPT-5**，并提到目前缺乏关于其开发者 **ColumnAI** 的信息。
  - 用户使用“高难度”提示词测试了 **sus-column-r 模型**，发现其表现与 **GPT-4o** 相似。一些人表示怀疑，要求提供实际案例，并提醒他人注意“*We Have No Moat*”（我们没有护城河）的概念。
  - 关于该模型的来源引发了讨论，有人猜测它来自 **Cohere 的 Column 系列**。其他人则警告不要将其视为事实，并指出 **Cohere 的现有模型**与较新的模型相比表现不佳。
  - 该模型展示了广泛的知识储备，正确识别了“*Die monster, you don't belong in this world*”的出处，据称还知道某位用户**八年级冬季学校旅行**的细节。一些用户觉得它平平无奇，而另一些人则称其“**体量巨大**”且“**可疑 (sus)**”。


**主题 4. 探索 LLM 的能力与局限性**

- **AI / LLM 还有哪些做不到的事？** ([Score: 79, Comments: 177](https://reddit.com//r/LocalLLaMA/comments/1en2res/what_cant_ai_llms_do_for_you/)): 该帖子讨论了 **AI 和 LLM** 的现状及未来预期，指出虽然有渐进式的改进，但**自 GPT-4 以来还没有出现颠覆性的进步**。作者观察到顶级模型之间出现了**能力趋同**，质疑我们是否只是在尝试 **GPT-4 已经能完成**的任务，并询问用户希望 AI 完成哪些目前还无法实现的实际任务。帖子认为，**限制可能在于聊天机器人界面**而非底层的 LLM 技术，并提议通过**不同的微调方法**以及创建 **Agent** 而非对话模型，可能会从现有的基础模型中激发更多有用的行为。
  - 大型应用程序的**代码生成**仍然具有挑战性，LLM 很难在没有大量人工修正的情况下生成超过 **200 行**的连贯代码。用户渴望在复杂、多功能的开发任务中获得更强的能力。
  - 诸如目标定位、漫画理解和结构化图像分析等**视觉理解**任务对 AI 来说仍然很困难。用户报告称，需要大量的预处理和专门工具才能在这些领域取得部分成功。
  - 用户希望 AI 能生成超出当前 Token 限制的**更长、更连贯的输出**。虽然像 **Sonnet 3.5** 和 **Gemini 1.5 Pro** 这样的模型在这一领域展现了潜力，但在超长上下文生成方面仍需进一步改进。

## All AI Reddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**AI 模型能力与进展**

- **GPT-4o 展示了非预期的语音克隆能力**：在 /r/singularity 中，一段来自 [OpenAI 的视频](https://www.reddit.com/r/singularity/comments/1enne2l/gpt4o_yells_no_and_starts_copying_the_voice_of/) 显示 GPT-4o 在测试过程中大喊“NO!”并简短地模仿了用户的声音。这突显了在控制先进 AI 模型方面的潜在风险和挑战。

- **Google DeepMind 的 AI 在乒乓球方面达到人类水平**：在 /r/singularity 中，[Google DeepMind 宣布](https://www.reddit.com/r/singularity/comments/1en8vrg/google_deepminds_aipowered_robot_plays_table/) 他们的 AI 驱动机器人成为第一个在乒乓球比赛中达到人类水平的“Agent”。

- **Gemini 1.5 Flash 价格下调**：在 /r/singularity 中，Google [宣布将 Gemini 1.5 Flash 的价格降低 70%](https://www.reddit.com/r/singularity/comments/1endo0r/gemini_15_flash_price_is_now_70_lower/)，使先进的 AI 能力变得更加普及。

- **OpenAI 开放免费 DALL-E 3 图像生成**：在 /r/singularity 中，OpenAI [宣布](https://www.reddit.com/r/singularity/comments/1engl4w/openai_were_rolling_out_the_ability_for_chatgpt/) ChatGPT 免费用户现在每天可以使用 DALL-E 3 创建最多两张图像。

**AI 在科学研究与数学领域**

- **AI 自动化数学证明**：在 /r/singularity 中，数学家 Terence Tao [讨论了如何利用 AI](https://www.reddit.com/r/singularity/comments/1emzf0i/mathematician_terence_tao_says_ai_is_already/) 来自动化数学证明，这可能会彻底改变该领域。

- **Google DeepMind 用于 AGI 开发的 CSCG**：在 /r/singularity 中，一篇关于 [克隆结构因果图 (CSCG)](https://www.reddit.com/r/singularity/comments/1en4t83/google_deepmind_cscg_clonestructured_causal/) 的论文被认为是迈向 AGI 的突破，重点关注模式学习（schema-learning）和重绑定（rebinding）机制。

**机器人技术进展**

- **Boston Dynamics 的 Atlas 执行复杂动作**：在 /r/singularity 中，一段 [视频展示了](https://www.reddit.com/r/singularity/comments/1eneffr/impressive_boston_dynamics_atlas_does_pushups_and/) Atlas 机器人进行俯卧撑和波比跳，展示了机器人在灵活性和控制方面的进步。

**迷因与幽默**

- **“未来已至”迷因**：在 /r/singularity 中，一个 [热门迷因帖子](https://www.reddit.com/r/singularity/comments/1en5p99/the_future_is_now/) 以幽默的方式评论了 AI 技术的飞速发展。


---

# AI Discord 摘要

> 摘要之摘要的总结

**1. LLM 进展与基准测试**

- **Gemini 1.5 Flash 大幅降价**：Google 宣布对 **Gemini 1.5 Flash** 进行大幅降价，对于 128,000 tokens 以下的 prompts，成本降低了高达 **70%**，至每百万 tokens 7.5美分，使其在快速且廉价的模型市场中极具竞争力。
   - 更新后的模型现在可以原生理解 PDF，并提高了文本和多模态查询的性能。此举被视为 AI 行业为提高效率而持续降价趋势的一部分。
- **DeepSeek-V2 声称超越 GPT-4**：据报道，新发布的 **DeepSeek-V2** 模型在 **AlignBench** 和 **MT-Bench** 等一些基准测试中超越了 **GPT-4**，展示了模型性能的进步。
   - 这一说法引发了关于 AI 社区需要标准化基准测试和透明评估方法以验证此类卓越性断言的讨论。
- **MiniCPM-V 2.6 挑战顶尖模型**：据开发者称，开源视觉多图模型 **MiniCPM-V 2.6** 的表现优于 **Gemini 1.5 Pro** 和 **GPT-4V** 等模型。
   - 社区分享了 [Hugging Face 模型](https://huggingface.co/openbmb/MiniCPM-V-2_6) 和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V) 的链接，邀请大家探索并验证这些性能声明。

**2. 模型优化与推理技术**

- **Tree Attention 算法优化长上下文处理**：一篇新论文介绍了 **Tree Attention 算法**，该算法通过在 GPU 集群上进行并行计算来优化 self-attention 计算，有望提高处理长上下文 attention 任务的效率。
   - 该实现已在 [GitHub](https://github.com/Zyphra/tree_attention) 上可用，旨在增强需要大量上下文处理场景下的性能，可能彻底改变模型处理大规模信息的方式。
- **Apple 开源 Matryoshka Diffusion Models**：Apple 开源了一个 Python 软件包，用于使用较小的数据集高效训练 **text-to-image diffusion models**，该项目与其 [ICLR 2024 论文](https://github.com/apple/ml-mdm) 相关联。
   - 该软件包旨在实现高质量结果，同时专注于减少数据和计算需求，可能使先进的 AI 图像生成技术更加普及。


**3. AI 初创公司融资**

- **Sequoia Capital 关注 AI 推理初创公司**：Sequoia Capital 已讨论资助一家由 **Robinhood CEO** 共同创立的 AI 推理初创公司，旨在增强 AI 在推理和决策方面的能力。
   - 据 [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo) 报道，这一潜在投资信号表明，人们对能够提高逻辑处理和决策能力的 AI 技术兴趣日益浓厚。
- **Anysphere 为 AI 编程助手获 6000 万美元融资**：AI 编程助手 Cursor 的开发商 **Anysphere** 已获得超过 **6000 万美元** 的 A 轮融资，估值达到 **4 亿美元**。
   - 本轮融资由 Andreessen Horowitz 领投，显示了投资者对 AI 驱动的编程解决方案及其改变软件开发实践潜力的强大信心。

**4. 开源 AI 框架与社区努力**

- **Replete-LLM-Qwen2-7b 发布**：新模型 **Replete-LLM-Qwen2-7b** 已发布，具有令人印象深刻的能力和基准测试结果，邀请用户通过 [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b) 进行测试。
  - 讨论建议，亲自测试对于理解性能差异至关重要。
- **Open Interpreter 黑客松引发关注**：Open Interpreter 正准备于 **9 月 20 日至 23 日** 在达拉斯举行“**Breaking Barriers**”黑客松，奖金总额为 **17,500 美元**。
  - 该活动鼓励现场参与，但也欢迎远程申请者，关于团队组建的社区讨论正在进行中。

**5. 新 AI 模型发布与创新**

- **Replete-LLM-Qwen2-7b 推出**：**Replete-LLM-Qwen2-7b** 已经推出，展示了强大的能力，并邀请用户通过 [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b) 进行测试。
  - 开发者强调了亲自测试的重要性，而不是仅仅依赖营销宣传的优越性主张。
- **用于 Function Calling 的 ActionGemma-9B 模型**：新的 **ActionGemma-9B** 模型专为 function calling 设计，利用了来自 Gemma 的多语言能力和 **xLAM** 数据集，增强了用户交互。
  - 有关其功能的详细信息可以在[此处](https://huggingface.co/KishoreK/ActionGemma-9B)访问。


**6. 社区支持与资源**

- **寻求 AI 研究社区**：成员们表达了对更活跃的**音频研究社区**的渴望，并指出之前的平台（如 harmonai）已变得不活跃。
  - 这突显了音频研究讨论支持方面的空白，以及对充满活力的社区的需求。
- **黑客松公告**：Open Interpreter 宣布参加“**Breaking Barriers**”黑客松，提供 **17,500 美元** 的奖金，鼓励社区参与。
  - 该活动强调 AI 领域的协作与创新，提供现场和远程参与选项。


---

# PART 1: High level Discord summaries

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **寻找活跃的音频研究社区**：一位成员正在寻求类似于 Nous 的**音频研究社区**推荐，理由是之前的 Discord 频道缺乏活跃的讨论。
   - *旧的 harmonai Discord 几乎已经沉寂，* 这凸显了音频研究支持方面的重大空白。
- **推出用于多模态 Agent 的 CRAB**：社区欢迎 🦀 **CRAB: Cross-environment Agent Benchmark** 的加入，该基准测试用于评估跨平台（包括 📱 Android 和 💻 Ubuntu）的多模态 Agent。
   - 其特性包括图形评估器和任务生成，旨在提升 **human-like performance**（类人性能）。
- **实习生 Eric 揭秘 ReFT 高级技巧**：明天**太平洋时间上午 10 点**，**Intern Eric** 将在演示中展示**“我如何使用 ReFT 在 14 分钟内微调 Llama3”**。
   - 该会议重点讨论 **Representational Fine Tuning** 的应用，有望为模型调优提供宝贵的见解。
- **澄清 ReFT 与 RLHF 的混淆**：成员们讨论了 **ReFT** 和 **RLHF** 之间的区别，一位用户强调了关于它们之间关系的误解。
   - 这种混淆表明在社区讨论这些技术时，需要更清晰的定义。
- **模型性能对比讨论**：讨论强调了整合 **A/B tests** 和稳健的 Benchmark 来验证新模型优越性声明的重要性，特别是针对 **Llama-3.1-8B** 和 **Gemma-2-9B**。
   - 用户对在没有适当 Benchmarking 的情况下随口称模型为 **state-of-the-art** 表示担忧。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 2 受到关注**：成员们注意到 **Gemma 2** 正变得越来越受欢迎，与 **Llama** 和 **Mistral** 等前辈相比，它正在吸引自己的受众。讨论强调了该模型相对于竞争对手的独特特征和性能细微差别。
   - *兴趣的转移表明社区对多样化架构的接受度正在提高。*
- **推出 Replete-LLM-Qwen2-7b**：新模型 **Replete-LLM-Qwen2-7b** 已发布，具有令人印象深刻的能力和 Benchmark 表现，邀请用户通过 [Hugging Face](https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b) 进行测试。开发者敦促用户亲自评估模型，而不是仅仅依赖市场宣传的优越性声明。
   - 讨论表明，亲自测试对于理解性能差异至关重要。
- **模型 Benchmarking 争议**：关于当前模型 Benchmark 缺点的对话不断出现，用户指出性能差异与不同的训练数据有关。一位成员指出，尽管在编程任务中表现更高，但由于训练目标的不同，Benchmark 分数可能无法反映质量。
   - *这次对话强调了在评估模型效能时上下文（Context）的重要性。*
- **模型中的连续批处理详述**：用户探索了模型进行连续微调（continuous finetuning）的适应性，讨论了诸如 **ReFT** 之类的增强功能。关于 **Unsloth** 如何在持续训练策略下支持额外功能的疑问也随之产生。
   - *这凸显了人们对动态模型调整技术日益增长的兴趣。*
- **Flash Attention 3 兼容性担忧**：根据 [MrDragonFox](https://link.to.paper) 的说法，Flash Attention 3 (FA3) 仅与 **H100** 硬件和 **Hopper** 架构兼容。这引发了关于在使用 Flash Attention 时自动安装 **FA2** 的澄清。
   - 讨论引发了对 Flash Attention 版本实际用法的询问，成员们好奇 **FA2** 是否仍然占据主导地位。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 面临性能下降**：用户报告 **LM Studio** 的加载时间变长且响应迟缓，尽管之前使用正常，但现在将问题归因于 context length 设置。
   - *报告表明*，性能滞后影响了模型加载和响应速度，理想情况下，在设置未更改时不应受到影响。
- **新用户寻求模型指导**：一位新手询问 **LM Studio** 中支持处理图像和 PDF 的模型，以及视觉生成模型。
   - 讨论强调了需要改进入门工具，以帮助用户熟悉模型功能。
- **Gemma 2 的性能给用户留下深刻印象**：用户建议尝试 **Gemma 2 27B**，并指出其表现非常出色，特别是与 **Yi 1.5 34B** 相比。
   - *反馈强调*了即使是较小的 **Gemma 2 9B** 模型在各项任务中也表现高效，引发了对其更大版本模型的期待。
- **关于 LLM 推理笔记本电脑选择的激烈辩论**：用户在配备 **RTX 4050** 或 **RTX 4060** 的机器之间权衡 LLM 推理的选择，讨论集中在额外 **2GB VRAM** 的重要性上。
   - 专家强调，虽然增加 RAM 有助于提升性能，但为了充分利用大型模型，最大化 **VRAM** 具有优先权。
- **Linux 上的 NVIDIA GPU 功耗限制**：用户讨论了在 Linux 上通过 **nvidia-smi** 等工具持久限制 **NVIDIA GPU** 功耗的方法，特别是针对 **RTX 3090**。
   - 建议使用脚本在重启后保持功耗限制，尽管企业级系统通常提供更好的功耗控制选项。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **SOTA 背景移除模型超越 RMBG1.4**：一位认证成员重点介绍了 **Bilateral Reference for High-Resolution Dichotomous Image Segmentation** 模型，得益于多所大学的贡献，其在背景移除方面的表现优于 **RMBG1.4**。更多详情请见 [模型页面](https://huggingface.co/ZhengPeng7/BiRefNet) 和 [arXiv 论文](https://arxiv.org/pdf/2401.03407)。
   - 该模型的进步展示了对低数据需求、高质量结果的日益关注，标志着背景移除技术的重大转变。
- **使用 ActionGemma-9B 进行 Function Calling**：新的 **ActionGemma-9B** 模型针对 **Function Calling** 进行了微调，利用了来自 Gemma 的多语言能力和 **xLAM** 数据集。详情请访问 [此处](https://huggingface.co/KishoreK/ActionGemma-9B)。
   - 这一进展通过启用特定的 **Function Calling** 增强了用户与模型的交互，推动了多语言模型在实际应用中的能力。
- **Unity ML-Agents 视频系列发布**：一段名为 **Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers** 的 YouTube 视频演示了如何使用 Unity 和 **Sentence Transformers** 创建聊天机器人。观看简介请点击 [此处](https://youtube.com/live/J-de9K_3xDw?feature=share)。
   - 这一举措代表了游戏开发与对话式 AI 的精彩融合，迎合了对在游戏环境中集成高级语言模型感兴趣的开发者。
- **Matryoshka Diffusion 模型发布**：Apple 开源了一个用于训练 **text-to-image diffusion models** 的 Python 包，该包使用较小的数据集，并与其 [ICLR 2024 论文](https://github.com/apple/ml-mdm) 相关联。这允许以更少的数据和计算需求获得**高质量结果**。
   - 这种方法可能会重新定义训练扩散模型的效率指标，潜在地影响 AI 生成媒体的未来研究。
- **关于 LoRA 训练技术的讨论**：成员建议专注于训练 **LoRA** 而不是全量模型，并指出训练更大架构的收益微乎其微。此外还讨论了运行 Flux 进行推理的内存要求。
   - 这些讨论强调了对高效模型训练实践的需求，反映了该领域向更轻量、更具适应性模型发展的趋势。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DALL·E 3 向免费用户开放权限**：OpenAI 宣布 **ChatGPT Free** 用户现在每天可以使用 DALL·E 3 创建最多 **两张图片**，支持个人和专业需求。
   - 反馈褒贬不一，一些用户对与其他模型相比的限制感到失望。
- **Gemini 1.5 降价 70%**：Gemini 1.5 Flash 实施了高达 **70%** 的降价，使其在 GPT4o 大幅降价的背景下更具竞争力。
   - 分析师认为，这种激进的定价策略提高了效率，反映了 AI 技术领域持续的竞争。
- **Deep-Live-Cam 实现实时 Deepfakes**：**Deep-Live-Cam** 允许用户通过单张图片实时生成高质量的 Deepfakes，令人印象深刻的实验证明了这一点。
   - 该项目因其在虚拟会议中的潜在应用而引发关注，展示了其强大的功能。
- **Anysphere 获得 6000 万美元融资**：Anysphere 成功筹集了 **超过 6000 万美元** 的 A 轮融资，为其 AI 编程助手 Cursor 锁定了 **4 亿美元** 的估值。
   - 此轮融资由 Andreessen Horowitz 领投，突显了投资者对 AI 驱动的编程解决方案的信心。
- **Llama 3.1 模型迎来关键更新**：Meta 发布了 **Llama 3.1** 405B 模型的更新版本，将 KV heads 从 16 个修改为 8 个，以符合其白皮书规范。
   - 这一变化引发了关于其对模型性能和架构影响的猜测。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 额度下调**：用户报告称 **Pro** 搜索限制已从 **600** 次降至 **450** 次，预计未来将降至 **300** 次，引发了关于透明度的不安。
   - 随着许多用户对在没有预警的情况下做出这一改变表示沮丧，担忧不断增加，引发了对服务可靠性的质疑。
- **OpenAI 的 Strawberry 模型引发热议**：OpenAI 的新 **'Strawberry'** 模型旨在增强推理能力，在 Sam Altman 通过社交媒体暗示后，在 AI 社区引发了轰动。
   - 该项目被视为解决复杂研究任务的重大进步，引起了工程师和研究人员的广泛兴趣。
- **Anduril 估值达到 140 亿美元**：**Anduril Industries** 融资 **15 亿美元**，估值从 **85 亿美元** 飙升至 **140 亿美元**，这主要归功于政府合同。
   - 随着收入翻倍至 **5 亿美元**，该公司的增长轨迹表明，在日益紧张的地缘政治局势下，国防科技需求强劲。
- **Perplexity 中的图像生成障碍**：用户对 Perplexity 中图像生成过程的复杂性表示沮丧，希望有更简单的功能，如直接提交 Prompt。
   - 讨论显示，当前的图像生成工具被认为有限且不切实际，亟需改进。
- **API 路线图查询**：一名成员提出了对 API 增加 **互联网访问** 功能路线图的需求，强调了用户对增强功能的兴趣。
   - 针对包含 'online' 字样的模型进行了说明，这表示部分互联网访问，虽然不是实时的，但强调了现有功能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **应对 NeurIPS 投稿流程**：一位成员分享了他们在 **NeurIPS** 的经历，对于在主要 AI 会议上获得高质量反馈和发表论文感到压力巨大。*这个过程非常令人不知所措，我不认识任何在主要 AI 会议上发表过论文的人*。
   - 他们也表达了同样的担忧，即参加这些顶会的经历可能会引发焦虑。
- **针对审稿人评分的 Rebuttal 策略**：出现了关于 Rebuttal 策略的建议，特别是针对置信度较低的审稿人，建议尽量减少对这些问题的关注。一位成员指出：*如果他们陈述了置信度低的原因，那么你可以尝试解决，否则我不会理会*。
   - 这一见解旨在优化 Rebuttal 流程并减少不必要的压力。
- **顶会的挑战**：对话强调了大型会议是多么令人望而生畏，并建议考虑参加较小的垂直领域会议，以获得更丰富的体验。一位参与者表示：*感觉一个人至少要在顶会上发表一次论文才能被认真对待*。
   - 这引发了关于声望与反馈质量之间平衡的讨论。
- **关于 RLHF 清理的讨论**：成员们辩论了在进行公开宣布之前，是否需要对 **RLHF** 实践进行清理流程。有人建议发布教程或博客文章，但普遍共识警告说这可能需要额外的时间。
   - 这次讨论强调了在对外宣传之前准备好完善叙述的重要性。
- **Qwen2 模型表现出异常的内存行为**：测试显示 **Qwen2** 模型在训练期间表现出明显的预留内存，特别是在 Batch Size 为 4 时，这引发了对潜在内存泄漏问题的警觉。成员们现在正寻求更彻底地分析这一行为。
   - 这一发现可能会导致未来训练协议中的关键优化和调整。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **PyTorch Profiler 内存泄漏 Bug**：一位成员在使用 `profile_memory=True` 的 PyTorch Profiler 时遇到了 **内存泄漏**，不确定设置中的根本原因。
   - 另一位成员通过切换到 `torch.cuda.memory._record_memory_history()` 进行分析并取得了成功，这表明了一种替代方法。
- **关于 4090 Tensor Cores 的见解**：讨论集中在何处获取 4090 上 **Tensor Cores** 的详细规格，建议查阅 **Ada whitepaper**。
   - **Ampere whitepaper** 被提及作为 **3090** 规格的参考，强调了详尽文档的必要性。
- **torch.compile 倾向于使用 Triton Kernels**：据分享，**torch.compile** 主要输出 **Triton Kernels**，提供了比 PyTorch Eager Mode 的 CUDA Kernel 输出更简洁的实现。
   - 提到了 **Cutlass Backend** 的存在，但进展仍不明确，突显了 Kernel 开发中持续的增强。
- **INT8 量化训练修复**：通过在调用 `torch.chunk()` 时设置 `requires_grad=False` 解决了 **INT8 量化训练** 中的一个错误，简化了实现。
   - 这表明了 PyTorch 在处理张量操作中的梯度时可能存在的复杂性，强调了精确性的重要性。
- **RoPE Kernel 重构**：进行了一场关于 **RoPE Kernel** 的讨论，成员们建议进行重构以使用显式的三角函数来提高代码清晰度。
   - 分享了一个不含复数的早期版本，展示了一种可能更易于维护的 Kernel 设计方法。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **辩论 AI 模型中的 CBRN 风险**：广泛的讨论强调了过滤 **CBRN 相关信息**（化学、生物、放射性、核）是否能在不损害模型能力的情况下降低风险。
   - 参与者指出了移除知识与仍可能产生有害输出风险之间的权衡。
- **AI Safety 研究机会**：一名成员提到了来自 Open Philanthropy 的 **职业转型资助 (career transition grant)**，旨在支持 AI Safety，并为教育练习寻求 GPU 资源。
   - 讨论了多种 GPU 访问选项，包括 **Colab 和 CAIS 集群**，以支持 AI 研究。
- **Karpathy 的 nanoGPT 评估挑战**：成员们讨论了 **lm-evaluation-harness** 在 **Karpathy 的 nanoGPT** 模型上的问题，指出其与 HF 格式不兼容。
   - 由于这些挑战，一位用户请求帮助使评估框架正常运行。
- **用于高效计算的 Tree Attention**：对话指向了一篇关于 **Tree Attention 算法**的论文，该算法通过 GPU 上的并行计算优化了 self-attention 计算。
   - 该实现有望提高长上下文 (long-context) 注意力任务的效率，并分享了 **GitHub 仓库**。
- **Zamba 模型性能惊人**：**Zamba 模型**因在训练 token 较少的情况下表现优于 **LLaMA 2 7B** 而受到关注，尽管其曝光度有限。
   - 其公开可用的数据集因模型令人印象深刻的效率和结果而引发了兴趣。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **不降级的情况下优化 VRAM**：用户指出，在 **Low VRAM Mode** 下，如果生成成功完成，则可能不需要切换到较低配置的模型，从而节省处理时间。
   - *尝试不同的模型选项有助于优化性能，* 减少不必要的调整。
- **换脸工具：Rope 占据领先地位**：成员推荐使用 [Rope](https://github.com/Hillobar/Rope) 进行换脸，因为与 Roop 相比，它的安装更简单，特别是对于使用 Intel CPU 的用户。
   - 重点是为热衷于执行换脸的用户寻找有效且简单的工具。
- **Stable Diffusion 性能存在波动**：用户观察到 **Stable Diffusion** 的采样速度 (s/it) 存在波动，据报道，切换模型大小时的延迟会影响整体性能。
   - *分享了关于 ROCm 和 WSL2 等设置的见解，* 表明了硬件配置的重要性。
- **安全地委托定制 Lora 模型**：参与者讨论了利用 **Civitai 的悬赏系统 (bounty system)** 来委托定制 pony lora 模型，旨在实现安全交易。
   - *强调对创作者进行彻底审查是确保委托实践可靠性的关键步骤。*
- **实时预览设置引起关注**：一位用户询问了 **A1111** 中最佳的实时预览设置，特别是质疑各种格式的用途以及是否保存帧。
   - *这反映了社区驱动优化图像生成工作流以提高效率的趋势。*

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **ChatGPT 用户可免费访问 DALL·E 3**：ChatGPT 免费用户现在每天可以使用 **DALL·E 3** 生成最多 **两张图片**，允许为幻灯片和个性化卡片等项目创建图像。
   - 此次更新简化了图像请求，让用户可以直接要求 ChatGPT 根据其规格定制图像。
- **Mistral NeMo 未达预期**：成员们对 **Mistral NeMo** 在 16GB RAM 的 M1 机器上的性能表示关注，并指出运行较大模型的限制。
   - 针对该模型在消费级硬件上的兼容性和性能效能出现了担忧。
- **关于 GPT-4 与 GPT-4o 性能的辩论**：用户批评 **GPT-4o**，认为其表现不如 **GPT-4**，特别是在图像分析任务中。
   - *GPT-4o* 因提供僵化的回答而受到指责，让人联想到程序员脱离了核心原则。
- **对本地 AI 模型工作流的兴趣**：一位参与者讨论了转向使用 **Open WebUI** 和 **Ollama** 来运行本地 AI 模型，并考虑停止其 ChatGPT+ 订阅。
   - *LLama* 的可靠性得到了认可，但自托管设置仍存在一些需要解决的挑战。
- **LangChain 与 CSV 集成咨询**：一位用户寻求在 **LangChain** 中将 **CSV 文件** 集成为 *检索增强生成* (RAG) 文档的资源。
   - 这显示了人们对使用语言模型处理结构化数据的兴趣日益浓厚，并提升了关于实际 AI 应用的讨论。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemini 1.5 Flash 价格大降**：多位用户注意到 **Gemini 1.5 Flash** 的价格已降至每百万 token 仅 **7.5 美分**，使其在快速、高性价比的模型方案中极具竞争力。
   - 该模型现在原生支持 PDF，并提升了处理 **text** 和 **multi-modal queries** 的能力。
- **GPT-4o Mini 在编程方面超越 Gemini 1.5**：**GPT-4o Mini** 因其比 **Gemini 1.5** 更低的幻觉率而受到称赞，尤其是在编程相关任务中。
   - 用户表示强烈倾向于那些能在优化编程功能的同时有效减少幻觉的模型。
- **OpenRouter API 的配置困扰**：一位开发者提出了在 TypeScript 中使用 **OpenAI SDK** 时，配置 **OpenRouter API** 的 `providers` 自定义参数所遇到的问题。
   - 该 API 目前缺乏对这些自定义参数的支持，导致持续出现 linting 错误。
- **达克效应（Dunning-Kruger）见解引发幽默**：一场围绕 **Dunning-Kruger Effect** 的热烈讨论展开，用户们幽默地批评了专业知识讨论中的自我评估。
   - 对话幽默地将自信与实际能力进行了对比，特别是在涉及盈利项目方面。
- **寻找日语 LLM**：一位用户请求推荐在日语能力上超越 **GPT-4o Mini** 的 **LLMs**，寻找高性价比的替代方案。
   - 这一需求反映了对在大型模型能力之外、擅长特定语言处理的模型日益增长的需求。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **新的 Sus-Column-R 模型表现优于竞争对手**：[Reddit](https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/) 上的一篇帖子讨论了新的 **sus-column-r model** 的性能，声称它在 **translation**、**coding** 和 **mathematics** 等任务中优于 **GPT-4** 和 **Claude 3.5**。
   - “我不明白这怎么可能，”该用户强调道，反映了社区的好奇。
- **API 响应质量受到关注**：成员们报告在使用 curl 进行 API 请求时遇到困扰的 **403 Forbidden** 错误，暗示这可能源于 **invalid API key** 或地理位置限制。
   - 尽管进行了排障，成员们仍无法解决该问题，并注意到 VPS 和本地请求成功率之间存在差异。
- **Docker 安装让用户感到困惑**：一位用户在 **Docker** 安装后遇到界面无法运行的问题，询问是否遗漏了任何步骤。
   - 作为回应，**Nick Frosst** 指出问题可能与 **backend setup** 配置错误有关，但具体细节尚不明确。
- **Langchain 的多步功能报错**：一位用户在 **Langchain** 的 **multistep_tool_use** 中遇到错误，收到一条指示无法解析多跳补全（multihop completion）的消息。
   - 在寻求帮助时，他们请求提供关于如何正确集成 **Cohere** 和 **Langchain** 的文档参考。
- **Embedding 模型质量差异**：一位用户报告在从 `embed-english-light-v2.0` 切换到 `embed-english-light-v3.0` 后感到不满，观察到检索质量不升反降，违背了预期。
   - 在详细说明其数据集时，他们指出较新的模型并未达到预期的性能提升。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **事件驱动的 Agent 系统增强了灵活性**：以**事件驱动的方式**构建 Agent 允许构建灵活的**循环、多 Agent** 系统，并具有复杂的通信模式。查看这个展示其优势的[精彩教程视频](https://t.co/he0Az19WJS)。
   - _“这是一个非常棒的教程视频”_ 强调了事件驱动方法在 Agent 系统中的实用性。
- **Mixture-of-Agents 克服了大型模型的局限性**：[Junlin Wang](https://t.co/C8pZBBIcOk) 的一篇新论文揭示了一种将较小 LLM 集成为 **Mixture-of-Agents** 系统的方法，该系统使用完全异步、**事件驱动的工作流**，性能超越了最先进的大型模型。
   - 实现细节在 [Twitter](https://t.co/hNpoZuBC5l) 上进行了讨论。
- **了解用于 GraphRAG 的属性图 (Property Graphs)**：一个重要的[视频教程](https://t.co/CdWqPOxt4c)解释了 LlamaIndex 的**属性图**，它允许每个节点和关系存储结构化的属性字典，从而解锁了各种技术。
   - _“这种底层抽象解锁了许多酷炫的技术”_ 突出了属性图的功能性。
- **为实际应用构建多模态 RAG 流水线**：新的 notebook 解释了如何针对复杂的**法律、保险和产品文档**创建实用的**多模态 RAG** 流水线，从解析**保险理赔**开始。
   - 详细的分解和实际用例可以在[这里](https://t.co/1w4IfXQ7CP)找到。
- **选择用于高效文档检索的 embedding 模型**：一位成员讨论了在 Llama 中使用 [HuggingFaceEmbedding](https://github.com/openai/tiktoken) 模型，并在查询调用前展示了文档加载示例。
   - 围绕 embedding 后的文档检索提出了疑问，澄清了实现预期结果的关键顺序步骤。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 黑客松引发关注**：Open Interpreter 正在筹备 **9 月 20 日至 23 日**在达拉斯举行的“**Breaking Barriers**”黑客松，奖金总额达 **$17,500**。
   - 该活动鼓励现场参与，但也欢迎远程申请者，社区关于组队的讨论正在持续进行。
- **MiniCPM-V 2.6 在竞争中脱颖而出**：据报道，**MiniCPM-V 2.6** 模型性能超越了 **Gemini 1.5 Pro** 和 **GPT-4V** 等知名竞争对手，引起了用户的兴趣。
   - [Hugging Face 模型](https://huggingface.co/openbmb/MiniCPM-V-2_6)和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V)的链接提供了关于其能力的进一步见解。
- **社区征求关于 ESP32S3 的见解**：一位用户寻求在 **ESP32S3** 上部署 **O1** 的帮助，并向其他成员询问现有经验。
   - 共享经验的请求旨在增强社区内感兴趣用户的实现策略。
- **请求 Linux 支持讨论**：成员们讨论了建立专门的 **#linux-something_or_other** 频道的必要性，以便更有效地处理 Linux 特定话题。
   - 这一建议获得了积极反馈，并将其链接到了一个旨在解决故障排除问题的现有频道。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 在 LLM 特性一致性方面面临挑战**：成员们对 **LangChain** 在所有 LLM 之间提供统一 API 的能力表示困惑，指出它在 **OpenAI** 上运行良好，但在 **Anthropic** 上则不然。
   - 澄清指出，虽然函数调用（function calls）类似，但由于 LLM 固有的差异，Prompt 修改是必不可少的。
- **Claude 3.5 遭遇停机**：**Anthropic** 的 **Claude 3.5** 经历了严重的停机，报告显示内部服务器错误代码 **500** 导致其功能中断。
   - 用户分享了错误消息，强调了 API 问题对运营能力的影响。
- **加入 1000 美元的 CTF 挑战赛！**：这是一个令人兴奋的**夺旗赛 (CTF) 挑战**，参与者的目标是从一个 AI Agent 中提取密码，奖金为 **1000 美元**。
   - 该竞赛引发了对数据隐私的关注，因为它研究了通过用户反馈表泄露秘密的风险。
- **Mood2Music 仪表板发布**：**Mood2Music** 仪表板公开展示，它根据用户情绪提供 AI 驱动的歌曲推荐，并链接到 **Spotify** 和 **Apple Music**。
   - 该工具通过策划与用户情感状态一致的播放列表，旨在解决音乐选择中的**决策疲劳**。
- **介绍 CRAB：多模态 Agent 基准测试**：**CRAB** 基准测试框架有助于在包括 **Android** 和 **Ubuntu** 在内的各种环境中构建和评估多模态语言模型 Agent。
   - 它具有**细粒度**的评估指标和任务生成能力，旨在提高类人任务的执行力，资源可在 [GitHub](https://github.com/) 和项目[网站](https://crab.camel-ai.org/)上获得。



---



## [LAION](https://discord.com/channels/823813159592001537) Discord

- **CC vs LAION 数据集之争**：关于 [Fondant 25M 数据集](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) 是否拥有最大的知识共享/公有领域图像集合的争论升温，触及了 **LAION-5B** 因依赖通常不相关的 Alt Text 而产生的可靠性问题。
   - 参与者强调，在对图像描述（captioning）敏感的任务中，**LAION-5B** 可能会带来更大的准确性风险。
- **Gemma 模型转向（Steering）咨询**：出现了一个关于使用 **Gemma Scope** 引导 **Gemma 2 2B** 的咨询，重点是为输出生成创建有效的控制向量。
   - 除了基础的 Google 搜索结果外，显然还需要更全面的见解，以提升对模型特性的理解。
- **描述（Captions）的可靠性受到质疑**：讨论集中在**大规模抓取的描述的不可靠性**上，有声音表示担心所有描述可能都缺乏精确的准确性。
   - 有人提出疑问，采用 CLIP 相似度分数是否能增强对新描述是否比**原始描述更不可靠**的评估。
- **Halva Assistant 见解**：分享了一个关于 [Halva Assistant](https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/) 的链接，该助手旨在减轻语言和视觉任务中的幻觉（hallucinations）。
   - 这一创新对于未来的 AI 发展可能至关重要，特别是在提高多模态系统的可靠性方面。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **红杉资本关注 AI 推理初创公司**：红杉资本（Sequoia Capital）讨论了为一家由 **Robinhood CEO** 共同创立的 AI 推理初创公司提供资金，旨在增强 AI 在**推理**和**决策**方面的能力。更多详情请参阅 [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo)。
   - 该初创公司专注于推进 AI 在逻辑语境下的交互方式，这是未来 AI 发展的关键领域。
- **Anaconda 的新商业许可政策**：研究和学术机构现在被要求为 **Anaconda** 的软件付费，因为该公司正在寻求其服务条款的合规性。报告显示，由于未经授权的使用，一些机构正面临商业许可的法律要求。
   - 成员们还提出了关于在 **Docker 容器**中使用 Anaconda 是否需要额外许可的问题，暗示这很可能需要。
- **uv 作为快速的 pip 替代方案出现**：**uv** 正在被讨论作为安装包时比 **pip** 更快的替代方案，用户注意到其速度有显著提升。该替代方案不需要额外的工具，只需在安装时将 `pip` 替换为 `uv pip` 即可。
   - 使用 uv 可以简化许多人的开发流程，特别是在需要快速包管理的场景中。
- **通过幽默改善讨论氛围**：关于讨论中糟糕观点的幽默评论建议，如果只有那些持有**糟糕观点**的人参与对话，世界将会受益。*“如果每个持有糟糕观点的人都只发表糟糕观点，世界会变得好得多”*反映了一种普遍的情绪。
   - 这一声明强调了在社区对话中进行更具建设性参与的愿望，呼吁更高质量的讨论。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **通过 YouTube 教程掌握 DSPy**：一位成员分享了一个关于 DSPy 的 [YouTube 教程](https://youtu.be/_ROckQHGHsU)，详细介绍了从基础到高级的 **8 个示例** LLM 项目，旨在增强用户的理解。
   - 这种结构化的方法让观众能够有效地掌握 DSPy 的核心概念，并将其应用到自己的项目中。
- **实验 OpenAI 的结构化输出 API**：一位成员宣布他们正在实验 OpenAI 新推出的**结构化输出 API**，以增强项目中的数据交互。
   - 该 API 旨在改进结构化数据输出的利用方式，引发了对更广泛实现的兴趣。
- **使用自定义 GPT 提升 DSPy 提示词**：成员们讨论了如何改进交织指令和示例的复杂提示词，重点关注 **Signature 适配器**和 **MIPRO 优化**。
   - 建议的起点是一个[自定义 GPT 指南](https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7)，用于更好地实现提示词的模块化。
- **探索 DSPy 在 RAG 中的用例**：一位成员寻求关于 DSPy 是否适合 RAG 任务的见解，并将其与微调过程进行了类比。
   - 另一位成员澄清说，成功的应用取决于对任务、指标和示例的优化，以提升 LLM 的性能。
- **Signature 适配器展现出 DSPy 的潜力**：讨论围绕在自定义 DSPy 提示词中使用 **Signature 适配器**的潜在好处展开。
   - 分享了一个关于该主题的进一步阅读链接：[Signature GPT 资源](https://chatgpt.com/g/g-JQgwRHI0D-dspy-signature-gpt-v2024-2-21)。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Poe 举办生成式 UI 黑客松**：Poe 正在举办为期一天的[黑客松](https://x.com/poe_platform/status/1820843642782966103)，旨在利用 **GPT-4o** 和 **Gemini 1.5 Pro** 等先进 LLM 开发生成式 UI 体验，线下活动地点在 **加州希尔斯伯勒 94010**。
   - 只有注册参与者才会收到独家详情，强调了此次活动的竞争优势。
- **AI-Health 倡议实习开放**：**Alliance AI-Health 研究倡议**正在寻找学生参加为期 **4 个月的远程实习**，以推进癌症检测和基于 AI 的中暑检测等领域的研究。
   - 申请截止日期为 **8 月 11 日**，实习生有机会在学术期刊上发表研究成果，[点击此处申请](https://tinyurl.com/applyalliance)。
- **计算机视觉中的特征存储受到关注**：一位成员对**计算机视觉**中 **Feature Stores**（特征存储）的有效性和价值提出了疑问，引发了关于它们在项目管理中作用的讨论。
   - 讨论强调了对真实世界实现的需求，因为示例可以证实 Feature Stores 在各种框架中的影响。

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 的许可证引发疑问**：一位成员指出，**Modular** 针对 **max/mojo** 的许可证是宽松的，除非有意将 AI 基础设施平台商业化。
   - 成员们对 Modular 如果进军**机器人技术**或 **AI 标注平台**可能产生的影响表示担忧。
- **未来竞争力的不确定性**：社区讨论了根据 Modular 的协议，目前被归类为非竞争性的软件在未来可能会变成竞争性软件。
   - 疑问在于，这类竞争性软件的开发在转型后是否必须“冻结”。
- **Triton 语言用户外联**：官方发起了一项号召，邀请编写过自定义 Kernel 的 **Triton lang** 用户与产品团队进行一对一交流，并提供 **Mojo swag** 作为奖励。
   - 该计划旨在收集用户见解以改进产品功能。
- **对 Triton 语言的好奇**：一位成员表示这是他们**第一次**听说 **Triton**，这表明人们对新兴编程语言的兴趣日益浓厚。
   - 这暗示了更广泛的社区参与**高级编程技术**的潜力。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Google Gemini 降价幅度惊人**：[标题为“Google Gemini Insane Price Cuts!!!”的 YouTube 视频](https://www.youtube.com/watch?v=3ICC4ftZP8Y)强调了 **Google Gemini 1.5 Flash** 的大幅降价。
   - 有关这些变化的详细信息也在 [Google Blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...) 中进行了分享。
- **关于 Gemini 与 GPT-4o 比较的困惑**：讨论围绕着是应该将 **Gemini 1.5 Flash** 与 **GPT-4o** 进行比较，还是应该与 **Gemini 1.5 Pro** 进行区分。
   - 成员们辩论了将标准版和 Mini 版的比较分开处理的价值。
- **Gemini 1.5 的免费微调功能备受关注**：有讨论认为 **Gemini 1.5 的免费 Finetuning** 特性影响了其与 Pro 版本的比较。
   - 这一区别已成为关于 Gemini 模型能力讨论的焦点。
- **咨询 Llama CPP 提示词缓存（Prompt Caching）**：一位成员寻求帮助，询问在使用 **Llama CPP server** 时应使用哪些参数来缓存提示词，目标是仅缓存初始提示词。
   - 他们澄清说，希望缓存第一个用户提示词（约 **1.5k tokens**），同时让 **Llama CPP** 管理其他内容。
- **询问 Llama 3 训练细节**：一位成员询问有关 Meta 的 [Llama 3 模型](https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml) 训练过程的文档，特别是关于所使用的数据和 Mask。
   - 他们注意到了重命名现有 tokens 以作为 Llama 3 模型中特殊 tokens 的方法。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **AMD 后端可能使用更多内存**：一位成员担心 **AMD backend** 是否比 **GPU backend** 消耗更多内存，从而引发了关于资源分配和性能的讨论。
   - 这突显了社区在针对各种后端优化内存管理方面的持续考量。
- **剧烈计算中报告 GPU 故障**：一位成员分享了他们的 **GPU** 损坏的坏消息，简单地说道：*“Rip my GPU got blown.”*
   - 这一事件引发了对高负载任务期间 GPU 可靠性的担忧。
- **为了简化而对模型进行去分片（De-sharding）**：一位用户询问如何通过对模型进行去分片将 **multi lazy buffer** 转换为 **normal lazy buffer**，表明了简化流程的需求。
   - 这指向了社区内模型优化和架构适配中普遍存在的挑战。
- **澄清 copy_to_device 函数用法**：关于 **copy_to_device** 函数的讨论出现，暗示了它在模型操作期间数据处理中的重要性。
   - 这强化了用户在工作流中对有效内存管理实践进行明确指导的需求。

---

**Alignment Lab AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接


{% if medium == 'web' %}




### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1271194238947823657)** (3 条消息): 

> - `音频研究社区`
> - `CRAB: Cross-environment Agent Benchmark` 


- **寻求音频研究社区**：一位用户询问是否有类似于 Nous Research AI Discord 但专注于音频而非语言的社区，表达了希望有一个可以提出困难研究问题的空间。
   - *“旧的 harmonai discord 已经基本沉寂了”，* 这表明需要更多关于音频研究的活跃讨论。
- **为多模态 Agent 引入 CRAB**：一位成员介绍了 🦀 CRAB: Cross-environment Agent Benchmark，它提供了一个端到端的框架，用于构建多模态 Agent，并在 📱 Android 和 💻 Ubuntu 等平台上进行评估。
   - 关键组件包括用于详细指标的图评估器（graph evaluator）和用于组合子任务的任务生成，旨在增强类人的任务执行能力。



**提到的链接**：<a href="https://x.com/camelaiorg/status/1821970132606058943?s=46">来自 CAMEL-AI.org (@CamelAIOrg) 的推文</a>：介绍 🦀 CRAB: Cross-environment Agent Benchmark for Multimodal Language Model Agents 🦀 CRAB 提供了一个端到端且易于使用的框架来构建多模态 Agent、操作环境...

  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1271283013120032850)** (7 条消息): 

> - `实习生 Eric 的 ReFT 演示`
> - `关于 ReFT 和 RLHF 的讨论`
> - `晚餐特色菜`
> - `社区活动` 


- **实习生 Eric 将演示 ReFT**：明天，**实习生 Eric** 将演示 **“我如何使用 ReFT 在 14 分钟内微调 Llama3”**，展示该技术的实际应用。
   - 演示将于 **太平洋时间上午 10:00** 进行，旨在深入探讨 **Representational Fine Tuning**。
- **ReFT 与 RLHF 之间的混淆**：一位成员表示困惑，称他们认为 **ReFT** 是 **RLHF** 的一种形式。
   - 这突显了社区中关于这些技术的具体定义和应用的持续讨论。
- **分享独特的晚餐菜单**：一位成员分享了他们独特的晚餐菜单，包括 **用牛骨和猪骨熬制的罗宋汤、鸡肉丸** 和酸奶油。
   - 其他项目还包括一个 **西红柿**、**两根黄瓜** 以及 **发酵的柳兰奶**。
- **社区参与提醒**：提醒大家每个 **周五** 都会聚集在一起讨论研究论文及其应用。
   - 鼓励成员加入这个由思想者和构建者组成的社区，共同营造协作氛围。
- **分享 YouTube 链接**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=KXJGrOqVrVQ)，可能与正在进行的讨论相关。
   - 这表明成员之间积极分享资源，以增强对主题的理解。



**提到的链接**：<a href="https://oxen.ai/community?utm_source=x&utm_content=y">社区资源 | Oxen.ai</a>：使用 Oxen AI 管理您的机器学习数据集。

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1271526592614760571)** (3 条消息): 

> - `Decoding the Decoder LLM`
> - `GPT2 in Excel`
> - `Fine-tuning Llama 3.1` 


- **使用电子表格解码 Decoder LLM**：一段名为 ["Decoding the Decoder LLM without de code: Ishan Anand"](https://youtu.be/NamKkerrlnQ?si=xMq8cXI7KZ4N-dU0) 的 YouTube 视频展示了电子表格如何简化对 AI 模型的理解。
   - 视频强调，即使是经验丰富的工程师也可能难以理解这些模型，这使得该教学工具显得尤为重要。
- **在 Excel 中实现 GPT2**：一位成员分享说，有人成功地在 Excel 电子表格中实现了 **GPT2**，并将其作为一种*教学工具*。
   - 这种方法旨在让 AI 模型的工作原理变得更加通俗易懂。
- **轻松微调 Llama 3.1**：一段名为 ["Fine tune 🦙 Llama 3.1 8b with Google Collab | LLM Tutorial"](https://youtu.be/y0EK1Xh4sMU) 的 YouTube 视频展示了如何使用 Q Lora 在 Google Collab 上仅用 **5 分钟** 免费微调 **Llama 3.1 8b**。
   - 评论区提供了资源链接，以协助观众完成该过程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/y0EK1Xh4sMU">Fine tune 🦙 Llama 3.1 8b with Google Collab | LLM Tutorial</a>: 评论区附带链接。使用 Unsloth 通过 Q Lora 在 Google Collab 上 5 分钟内免费微调 Llama 3.1 8b Instruct DPO。</li><li><a href="https://youtu.be/NamKkerrlnQ?si=xMq8cXI7KZ4N-dU0">Decoding the Decoder LLM without de code: Ishan Anand</a>: 电子表格就是你所需的一切：无需代码即可解码 Decoder LLM。即使是经验丰富的工程师，在掌握 AI 模型内部工作原理时也可能感到吃力...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1271181325688246444)** (200 messages🔥🔥): 

> - `模型性能对比`
> - `SOTA 声明与基准测试`
> - `Hermes 2 Pro 对比 Mistral`
> - `Replete-LLM Qwen2 发布`
> - `手动测试对比基准测试` 


- **模型性能对比**：一位用户强调，**有效的模型测试**理想情况下应包括 A/B 测试和多个基准测试，以确保可靠性，特别是在声称具有优越性时。
   - 讨论中将 **Llama-3.1-8B** 和 **Gemma-2-9B** 公认为对比新模型性能的参考基准。
- **SOTA 声明与基准测试**：有人对在没有正式基准测试结果验证的情况下将新模型称为 *SOTA (state-of-the-art)* 表示担忧。
   - 用户建议，虽然详细的模型卡片 (Model Cards) 和个人测试很有价值，但应通过与主流模型的透明基准测试对比来补充这些声明。
- **Hermes 2 Pro 对比 Mistral**：一位用户称赞 **Hermes 2 Pro** 在并行工具调用方面表现出色，在此方面显著优于 **Mistral**。
   - 这引发了关于日益增长的开源贡献如何突破模型能力边界的讨论，且这些贡献通常是在资金较少的情况下完成的。
- **Replete-LLM Qwen2 发布**：**Replete-LLM Qwen2-7b** 的发布已宣布，突出了其具有竞争力的特性和开源性质。
   - 用户对该模型表现出热情，但同时也对 SOTA 声明持怀疑态度，强调需要基准测试验证。
- **手动测试对比基准测试**：关于手动测试与标准基准测试在评估模型性能方面的可靠性存在激烈辩论。
   - 虽然一些人认为个人测试能更好地洞察模型的能力，但其他人坚持认为基准测试是进行比较的必要尺度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/rpnickson/status/1821634114274873850?s=46">来自 Roberto Nickson (@rpnickson) 的推文</a>: 天哪。毫无疑问，这是我见过的最真实的 AI 图像。我们已经完成了 99.7% 的路径，AI 图像将与现实完全无法区分。（缩放时仍能看到一些瑕疵...）</li><li><a href="https://tenor.com/view/star-wars-more-gif-21856591">Star Wars GIF - Star Wars More - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://new.reddit.com/r/singularity/comments/1envnur/5_new_models_on_lmsys/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1eog6es/repletellm_stateoftheart_model_releases_today/?utm_name=web3xcss">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/cognitivecomputations/grokadamw">GitHub - cognitivecomputations/grokadamw</a>: 通过创建账户为 grokadamw 的开发做出贡献。</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#ai-consultant---lead-ai-engineer">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#project-lead---innovative-technologies">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/dnhkng/RYS-Llama-3.1-8B-Instruct#research-scientist">dnhkng/RYS-Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/dnhkng">dnhkng - 概览</a>: dnhkng 拥有 59 个仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1271206575792590979)** (109 messages🔥🔥): 

> - `Claude's Upside Down Text Generation` (Claude 的倒置文本生成)
> - `Multi-GPU Setup for LLMs` (LLM 的多 GPU 配置)
> - `Qwen2-Audio Capabilities` (Qwen2-Audio 的功能)


- **Claude 可以有效地生成倒置文本**：成员们注意到 Claude 可以生成倒置的文本，一些人将其归功于潜在的“反向分词 Token (reverse tokenizing token)”。与其他据称在类似任务中表现吃力的模型进行了对比。
   - 一位用户在被封禁后表达了沮丧，而其他人则讨论了“antthinking”的影响以及 Claude 表现背后的整体逻辑。
- **多 GPU 配置的挑战**：围绕设置多 GPU 配置进行了讨论，特别是将 4090 与 3090 或 3060 搭配使用，并考虑了电源供应需求。建议为 GPU 使用独立的电源，以更好地管理功耗。
   - 用户分享了关于功耗的个人经验，强调了计算实际使用量以确保其 PSU 能够安全处理负载的重要性。
- **Qwen2-Audio 简介**：Qwen2-Audio 模型已发布，它允许音频和文本输入，在保持对话上下文的同时生成文本输出。用户对其功能感到兴奋，将其比作具有增强对话上下文功能的 Whisper。
   - 分享了该模型的 Demo 和特性链接，强调了语音聊天和多语言支持等功能，并提出了未来扩展模型能力的计划。



**提到的链接**：<a href="https://x.com/Alibaba_Qwen/status/1821945506463129905">来自 Qwen (@Alibaba_Qwen) 的推文</a>：今天我们发布了 Qwen2-Audio，这是 Qwen-Audio 的下一个版本，它能够接受音频和文本输入并生成文本输出。我们在 Hu... 开放了 Qwen2-Audio-7B 和 Qwen2-7B-Instruct 的权重。

  

---


### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1271316572530806927)** (7 messages): 

> - `Wordware template` (Wordware 模板)
> - `Benchmarking tasks` (基准测试任务)
> - `PR merge readiness` (PR 合并就绪状态)
> - `Converter adjustment` (转换器调整)
> - `Output length` (输出长度)


- **Wordware 模板揭示基准测试任务**：一位成员分享了一个 [Wordware 模板](https://app.wordware.ai/share/fda1ba91-b363-4814-a4e3-a997afef7949/playground)，旨在生成特定基准测试任务和查询的列表。
   - 他们表示这是一个**非常粗略的尝试**，并邀请大家提供改进建议。
- **关于 PR 准备好合并的讨论**：<@687315767208706059> 询问关于转换器的 **PR** 是否已准备好合并，引起了对正在进行的开发的关注。
   - 这个问题展示了团队在确保代码贡献被妥善集成方面的协作努力。
- **需要额外功能**：另一位成员指出，他们只需要**添加这个**功能即可完成待处理的任务。
   - 这表明正在对项目所需的功能进行持续的完善和协作。
- **输出长度问题**：一位成员表示模板的输出很长，大约 **1800 个字符**。
   - 这突出了一个潜在的优化领域，以增强用户体验。
- **模板的可调节性**：原发布者重申 Wordware 模板可以轻松调整，表达了其设计的灵活性。
   - 这表明了正在进行的实验以及根据用户反馈完善工具的意愿。



**提到的链接**：<a href="https://app.wordware.ai/share/fda1ba91-b363-4814-a4e3-a997afef7949/playground">Benchmark_Query_Creator</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1271187995302559794)** (216 条消息🔥🔥): 

> - `Gemma 2 的普及度`
> - `Replete-LLM-Qwen2-7b 发布`
> - `模型基准测试 (benchmarking) 的挑战`
> - `模型中的连续批处理 (Continuous batching)`
> - `训练与 Loss 计算` 


- **Gemma 2 正在走红**：成员们注意到 **Gemma 2** 正变得流行，并赢得了属于自己的受众，不像其前代产品曾被 **Llama** 和 **Mistral** 掩盖了光芒。
   - 对话强调了该模型的独特特征，参与者讨论了它与其他模型相比的性能表现。
- **Replete-LLM-Qwen2-7b 发布**：新模型 **Replete-LLM-Qwen2-7b** 已发布，分享了关于其能力和基准测试的细节，并邀请用户在 Hugging Face Space 中进行测试。
   - 开发者强调了亲测模型的重要性，而不是仅仅依赖于其优越性的宣传。
- **模型基准测试的挑战**：讨论围绕基准测试的不可靠性展开，成员们指出模型性能会根据训练中包含的数据而有所不同。
   - 一位成员提到，尽管某个模型在编程任务中表现更好，但由于训练目标的不同，其基准测试分数却出人意料地低。
- **模型中的连续批处理 (Continuous batching)**：用户讨论了模型的持续微调 (finetuning) 能力，提到了修改模型以支持 **ReFT** 等额外功能的灵活性。
   - 对话中还出现了关于 **Unsloth** 如何支持各种功能的询问和澄清。
- **训练后计算 Loss**：一位成员询问了在训练模型后，计算数据集中单个数据点 Loss 的正确方法。
   - 讨论了为 Tensor 输入提供的 Loss 计算代码片段，重点在于为了进行有效的 Loss 评估而进行的正确标签分配。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/buzz-lightyear-gif-21327280">Buzz Lightyear GIF - Buzz Lightyear - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/rombodawg/Replete-LLM-Qwen2-7b">Replete LLM Qwen2 7b - a Hugging Face Space by rombodawg</a>：未找到描述</li><li><a href="https://docs.vllm.ai/en/latest/dev/engine/async_llm_engine.html">AsyncLLMEngine &#8212; vLLM</a>：未找到描述</li><li><a href="https://www.reddit.com/r/nousresearch/comments/1eog6es/repletellm_stateoftheart_model_releases_today/?utm_name=web3xcss">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/ugshanyu/url">ugshanyu/url · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://youtu.be/TKmfBnW0mQA?si=lz2sHuGY_IXBYbN_">Fixing bugs in Gemma, Llama, &amp; Phi 3: Daniel Han</a>：关于我们为 Gemma 修复 8 个 bug、为 Llama 3 进行多次分词修复、为 Phi-3 修复滑动窗口 bug 并进行 Mistral 化的故事，以及了解我们如何……</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF">Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2">Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1271275160086904935)** (10 条消息🔥): 

> - `闲聊频道规则`
> - `消息删除权限` 


- **关于闲聊频道权限的讨论**：成员们辩论了像 “能私信我吗？” 这样的消息是否允许在闲聊频道发布，其中一人表示：**“哦是的，不允许”**。
   - 建议设立专门的 **rules 频道**来澄清此类事项。
- **在闲聊频道删除消息**：一位成员确认他们拥有**删除消息的权限**，回应了关于消息管理的问题。
   - 随后，另一位成员强调了建立规则频道以实现更好治理的重要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1271242752579080306)** (83 条消息🔥🔥): 

> - `在 Colab 中加载模型`
> - `微调 Llama 模型`
> - `LORA Adapter 与模型合并`
> - `GPU 和 CPU 问题`
> - `Llama-2-Chat 的数据集格式` 


- **在 Colab 上加载模型的挑战**：用户报告在 Colab 上训练模型时（特别是在处理 GGUF 文件时）磁盘空间耗尽。
   - 一位用户询问是否有解决方案可以避免保存到磁盘而直接上传到 Hugging Face，但被告知上传同样需要磁盘空间。
- **有效地微调 Llama 模型**：一位用户分享了微调 Llama 3.1 8b 的经验，建议增加训练样本可以提高模型性能。
   - 回复指出，生成更大的数据集并使用适当的 Chat Template 对成功的微调至关重要。
- **理解 LORA 和模型合并**：讨论围绕如何保存 LORA Adapter 和合并后的模型展开，明确了合并是将两者结合成一个新模型，而保存 LORA Adapter 则是将其保持分离。
   - 提醒用户，通常最好不要在已经微调过的模型上再次微调，以避免潜在的质量损失。
- **处理 GPU 和 CPU 限制**：用户在尝试运行模型时遇到了与 GPU RAM 限制相关的问题，并得到了检查 NVIDIA 驱动程序是否正确设置的建议。
   - 一位用户指出，由于 GPU RAM 不足时模型数据会传输到 CPU，某些操作会变慢。
- **为 Llama-2-Chat 准备数据集**：用户寻求关于微调 Llama-2-Chat 模型时数据集正确格式的澄清。
   - 共享了模板和示例，强调需要清晰的对话格式以确保训练效果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1vbH6h760iesRfcVQlm4-KVv1zYM5sB9k?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1lBzz5KeZJKXjvivbYvmGarix9Ao6Wxe5?usp=sharing>">Google Colab</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/saving-models/saving-to-gguf#manual-saving">Saving to GGUF | Unsloth Documentation</a>: 将模型保存为 16bit 的 GGUF 格式，以便你可以将其用于 Ollama, Jan AI, Open WebUI 等！</li><li><a href="https://arxiv.org/abs/2305.17493">The Curse of Recursion: Training on Generated Data Makes Models Forget</a>: Stable Diffusion 彻底改变了从描述性文本创建图像的方式。GPT-2, GPT-3(.5) 和 GPT-4 在各种语言任务中展示了惊人的性能。ChatGPT 引入了这种语言...</li><li><a href="https://huggingface.co/blog/mlabonne/sft-llama3">Fine-tune Llama 3.1 Ultra-Efficiently with Unsloth</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-2-7b-chat-bnb-4bit">unsloth/llama-2-7b-chat-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1OjbjU5AOz4Ftn9xHQrX3oFQGhQ6RDUuXQipnQ9gn6tU/edit?usp=sharing">SHARED Continuous Finetuning By Rombodawg</a>: 使用 Lora 和 Mergekit 进行无损连续微调。在本文中，我们将讨论如何使用 Lora Adapter 和 Mergekit 对开源 AI 模型进行持续微调...</li><li><a href="https://docs.unsloth.ai/basics/chat-templates>">Unsloth Documentation</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/tutorials/how-to-finetune-llama-3-and-export-to-ollama)">Unsloth Documentation</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1271208651624943636)** (12 条消息🔥): 

> - `Flash Attention 版本`
> - `Hopper 架构`
> - `H100 硬件限制` 


- **关于 Flash Attention 3 的讨论**：正如 [MrDragonFox](https://link.to.paper) 所指出的，Flash Attention 3 (FA3) 仅与 H100 硬件和 Hopper 架构兼容。
   - 讨论强调，当使用 Flash Attention 时会自动安装 FA2，这进一步澄清了 FA3 的有限使用场景。
- **对 Flash Attention 使用情况的好奇**：成员们对 Flash Attention 2 和 3 的使用表示好奇，询问 FA2 是否仍是目前最常用的版本。
   - *Flail_* 对这些版本之间的差异表现出兴趣，强调了 Attention 实现方式的不断演进。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1271181270638133381)** (244 条消息🔥🔥): 

> - `LM Studio 性能问题`
> - `模型加载与使用`
> - `Houdini 与 VFX 工具讨论`
> - `Flux 与 ComfyUI`
> - `针对新用户的社区支持` 


- **LM Studio 性能问题**：一位用户报告称，尽管已经成功使用了几个月，但 LM Studio 的模型加载时间异常长，且 AI 响应迟缓。
   - 这种变慢被归因于 context length 设置，尽管用户指出这与之前的用法并无不同。
- **模型加载与使用**：一位新用户询问 LM Studio 中支持哪些可以识别图像和 PDF 的模型，以及用于生成视觉内容的模型。
   - 对话强调了对用户友好功能的需求，以帮助新用户了解模型的能力。
- **Houdini 与 VFX 工具讨论**：参与者讨论了与 Autodesk 软件相比，使用 Houdini 作为 VFX 工具的优势，并提到了性能和用户体验。
   - 有人提到了像 Blender 这样的开源替代方案，以及它们颠覆当前 CG 软件市场的潜力。
- **Flux 与 ComfyUI**：用户表达了对不同 UI 的偏好，一些人更喜欢界面更整洁的 Forge，而另一些人认为 ComfyUI 更适合实验。
   - 预计 Flux 支持很快将加入 Forge，用户们渴望集成并实现便捷访问。
- **针对新用户的社区支持**：一位新用户寻求关于发布 LM Studio beta 版建议的最佳方式，社区成员将其引导至相关频道。
   - 强调了对新人在探索平台和改善体验方面的集体支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tauri.app">Build smaller, faster, and more secure desktop applications with a web frontend | Tauri Apps</a>: Tauri 是一个用于为所有主流桌面平台构建极小、极快二进制文件的框架。开发者可以集成任何可编译为 HTML、JS 和 CSS 的前端框架来构建其应用...</li><li><a href="https://huggingface.co/spaces/vilarin/Llama-3.1-8B-Instruct">Meta-Llama3.1-8B - a Hugging Face Space by vilarin</a>: 暂无描述</li><li><a href="https://tenor.com/view/aw-cry-sad-grandpa-gif-14766695">Aw Cry GIF - Aw Cry Sad - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://civitai.com/models/617609?modelVersionId=690425#_">FLUX.1 [dev] - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: 如果您还没有阅读所有建议，请不要下载。因为它很重，并且比 SD 需要更多的资源。我们有了运行 Flux 的简易新方法...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge.git">GitHub - lllyasviel/stable-diffusion-webui-forge</a>: 通过创建账号为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 中的 LLM 推理。通过创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1271217222093766859)** (34 条消息🔥): 

> - `Gemma 2 性能`
> - `用于 LLM 推理的笔记本电脑选择`
> - `Linux 上的 NVIDIA GPU 功耗限制`
> - `RAM 与 VRAM 对模型性能的影响`
> - `8700G 性能更新` 


- **Gemma 2 的性能给用户留下深刻印象**：用户推荐尝试 **Gemma 2 27B**，因为它的表现非常出色，尤其是与 **Yi 1.5 34B** 相比。
   - 反馈强调了 **Gemma 2 9B** 在各种任务中的有效性，引发了人们对更大的 **27B 模型** 的期待。
- **为 LLM 选择合适的笔记本电脑**：一位用户正在考虑使用配备 **RTX 4050** 或 **RTX 4060** 的笔记本电脑进行 LLM 推理，并讨论额外 **2GB VRAM** 的重要性。
   - 专家建议，虽然 RAM 有益，但关注 **VRAM** 至关重要，而笔记本电脑由于升级限制带来了挑战。
- **在 Linux 上限制 NVIDIA GPU 功耗**：用户讨论了如何使用 **nvidia-smi** 等工具持久地限制 **NVIDIA GPU** 的功耗，特别是针对 **RTX 3090**。
   - 建议使用脚本以确保在重启后应用功耗限制，尽管企业级服务器提供了消费级硬件可能缺乏的内置功耗限制功能。
- **RAM 和 VRAM 在模型性能中的平衡**：参与者强调 **8GB VRAM** 对于高要求模型来说是不够的，并指出显存的增加能显著扩展模型的可用性。
   - 有人指出，仅依赖 RAM 会降低性能，因此最大化 VRAM 对于高效运行大型模型至关重要。
- **8700G 性能更新**：一位用户报告了通过调整 RAM 设置对 **8700G** 进行的增强，在使用 **ollama** 运行 LLAMA 3.1 8B 时达到了 **16 tok/s**。
   - 他们指出了 LM Studio 在使用 AMD GPU 时的局限性和性能问题，影响了超过 **20GB RAM** 后的可用性，强调了持续优化的必要性。



**提到的链接**：<a href="https://llm.extractum.io/static/llm-leaderboards/">LLM Leaderboards</a>：所有 LLM 排行榜都在一个页面上。一份全面的 LLM 排行榜列表：深入了解自然语言处理领域 AI 语言模型的排名、挑战和进展，促进...

  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1271232340126863410)** (1 条消息): 

> - `Background Removal Improvements` (背景移除改进)
> - `Function Calling with ActionGemma-9B` (使用 ActionGemma-9B 进行 Function Calling)
> - `Unity ML-Agents Development` (Unity ML-Agents 开发)
> - `Segment Anything Model Insights` (Segment Anything Model 见解)
> - `Arabic Web Dataset Creation` (阿拉伯语 Web 数据集创建)


- **SOTA Background Removal Beats RMBG1.4**: 一位认证成员重点介绍了 **Bilateral Reference for High-Resolution Dichotomous Image Segmentation** 模型，得益于多所大学和实验室的贡献，该模型在背景移除方面的性能优于 **RMBG1.4**。
   - 更多信息可以在 [模型页面](https://huggingface.co/ZhengPeng7/BiRefNet) 和 [arXiv 论文](https://arxiv.org/pdf/2401.03407) 中找到。
- **Function Calling in ActionGemma-9B**: 发布了一个针对 Function Calling 进行微调的新 **ActionGemma-9B** 模型，利用了来自 Gemma 的多语言能力和 **xLAM** 数据集。
   - 您可以查看 [此处](https://huggingface.co/KishoreK/ActionGemma-9B) 了解该模型的更多详情。
- **Unity ML-Agents Video Series Launch**: 一段名为 **Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers** 的 YouTube 视频展示了使用 Unity 和 Sentence Transformers 从零开始创建聊天机器人的历程。
   - 点击 [此处](https://youtube.com/live/J-de9K_3xDw?feature=share) 观看精彩介绍。
- **Insights on Segment Anything Model**: 一篇博客文章讨论了 **Segment Anything Model** 及其在计算机视觉领域的相关进展，重点关注语言模型与视觉任务之间的差异。
   - 为了深入了解，可以在 [此处](https://www.lightly.ai/post/segment-anything-model-and-friends) 找到该文章。
- **Arabic Web-Only Dataset Pre-training**: **ArabicWeb24** 计划致力于创建一个高质量的阿拉伯语仅限 Web 的预训练数据集，以改进 NLP 模型。
   - 在 [此处](https://huggingface.co/blog/MayFarhat/arabicweb24) 探索详细介绍该计划的博客文章。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/ZhengPeng7/BiRefNet">ZhengPeng7/BiRefNet · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/KishoreK/ActionGemma-9B">KishoreK/ActionGemma-9B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/DamarJati/FLUX.1-DEV-Canny">FLUX.1-DEV Canny - a Hugging Face Space by DamarJati</a>: 未找到描述</li><li><a href="https://youtube.com/live/J-de9K_3xDw?feature=share">Unity ML-Agents | Pretrain an LLM from Scratch with Sentence Transformers| Part 1</a>: 欢迎来到我们使用 Unity ML-Agents 和 Sentence Transformers 创建智能聊天机器人的精彩旅程！🚀 在本视频中，我们将带您了解...</li><li><a href="https://www.lightly.ai/post/segment-anything-model-and-friends">Segment Anything Model and Friends</a>: Segment Anything Model (SAM) 及其继任者在计算机视觉领域取得了重大飞跃，特别是在图像和视频分割方面。伴随着 SAM 创新的可提示（promptable）方法...</li><li><a href="https://huggingface.co/spaces/Delik/Anitalker">Anitalker - a Hugging Face Space by Delik</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/LEGENT/LEGENT">LEGENT - a Hugging Face Space by LEGENT</a>: 未找到描述</li><li><a href="https://www.lightly.ai/post/using-self-supervised-learning-for-dense-prediction-tasks">Using Self-Supervised Learning for Dense Prediction Tasks</a>: 用于密集预测任务（如目标检测、实例分割和语义分割）的 Self-Supervised Learning 方法概述</li><li><a href="https://dev.to/tonic/dockers-testcontainers-are-great-42cl">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/prithivMLmods/lora-adp-01">Unlocking Creativity with Text-to-Image Generation: Exploring LoRA Models and Styles</a>: 未找到描述</li><li><a href="https://youtu.be/fnKrReaqQgc">How does Uber predict Arrival Times (ETA) for trips? | Uber ML System Design | #systemdesign</a>: 你知道像 Uber、Ola 和 Lyft 这样的网约车公司是如何预测行程的预计到达时间（ETA）的吗？在本视频中，我们设计了一个端到端的机器...</li><li><a href="https://huggingface.co/blog/MayFarhat/arabicweb24">ArabicWeb24: Creating a High Quality Arabic Web-only Pre-training Dataset </a>: 未找到描述</li><li><a href="https://github.com/Rivridis/LLM-Assistant">GitHub - Rivridis/LLM-Assistant: Locally running LLM with internet access</a>: 具有互联网访问权限的本地运行 LLM。在 GitHub 上为 Rivridis/LLM-Assistant 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1271184771065581669)** (174 条消息🔥🔥): 

> - `Hugging Face 模型与 API`
> - `Amazon Bedrock 定价`
> - `模型训练与架构`
> - `消息分类方法`
> - `LLM 中的采样参数` 


- **Hugging Face 模型的可用性与问题**：用户讨论了从 Hugging Face 下载模型并使用 `AutoModel.from_pretrained` 加载模型的问题，其中文件缺失可能导致错误。
   - 一位用户识别出了加载模型所需的一个缺失文件并解决了问题，而另一位用户在 Gradio 界面上遇到了挑战。
- **Amazon Bedrock 的高昂成本**：Amazon Bedrock 价格上涨被归因于供需问题，特别是提到的芯片短缺。
   - 一位用户分享说，由于合同中符合 GDPR 合规要求的子处理器条款，他们选择了 Amazon 服务。
- **分析模型训练的重要性**：对话围绕模型训练中数据集质量的重要性展开，强调了跨 Benchmark 的扩展结果。
   - 参与者指出，虽然架构很重要，但数据集和训练数据的完整性在性能中起着至关重要的作用。
- **探索消息分类方法**：一位用户展示了他们的消息分类指标，寻求与“公告”和“技术支持”等类别相关的方法。
   - 这表明在应用程序中需要结构化的方法来进行基于类别的消息处理。
- **调整 LLM 输出的创新想法**：一位用户建议根据输入序列在 LLM 采样中实施自适应温度调节，以改善输出变化。
   - 这一概念与 Diffusion 中现有的方法类似，但在实用性和重新训练要求方面提出了疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/hidden-cascade">Hidden Cascade - nroggendorff 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/cant-unsee-how-to-unsee-blind-homer-simpson-simpsons-gif-17613101">Cant Unsee How To Unsee GIF - Cant Unsee How To Unsee Blind - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/Skylion007/openwebtext/tree/main">Skylion007/openwebtext at main</a>: 未找到描述</li><li><a href="https://huggingface.co/facebook/m2m100_418M">facebook/m2m100_418M · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/jeremyphoward/status/1821557546102174194?t=MUNK-YPm71">Jeremy Howard (@jeremyphoward) 的推文</a>: 中国开源权重模型在 MATH 评测上轻松超越了以往所有模型（包括闭源和开源）：引用 Qwen (@Alibaba_Qwen) 今天我们发布了一个针对数学特定语言模型的新模型系列...</li><li><a href="https://x.com/jeremyphoward/status/1821557546102174194?t=MUNK-YPm71NgXJTbOLJDLA&s=19">Jeremy Howard (@jeremyphoward) 的推文</a>: 中国开源权重模型在 MATH 评测上轻松超越了以往所有模型（包括闭源和开源）：引用 Qwen (@Alibaba_Qwen) 今天我们发布了一个针对数学特定语言模型的新模型系列...</li><li><a href="https://huggingface.co/docs/transformers/model_doc/seamless_m4t_v2">SeamlessM4T-v2</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1271193792635998219)** (3 条消息): 

> - `Embedding 序列化`
> - `人类反馈强化学习 (RLHF)` 


- **学习 Embedding 序列化与反序列化**：一位成员分享说，他们正在学习如何在 **Python** 和 **C#** 之间**序列化和反序列化** Embedding 数据。
   - 这种技术技能对于促进不同编程环境之间的数据交换至关重要。
- **RLHF 精选资源**：一位成员分享了一个 [GitHub 仓库](https://github.com/opendilab/awesome-RLHF)链接，其中包含 **人类反馈强化学习 (RLHF)** 的精选资源列表。
   - 该仓库持续更新，为对这一前沿领域感兴趣的人提供有价值的信息。



**提到的链接**：<a href="https://github.com/opendilab/awesome-RLHF?tab=readme-ov-file#2024">GitHub - opendilab/awesome-RLHF: 人类反馈强化学习资源精选列表（持续更新）</a>: A curated list of reinforcement learning with human feedback resources (continually updated) - opendilab/awesome-RLHF

  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1271190710552494243)** (13 messages🔥): 

> - `Matryoshka Diffusion Models`
> - `ReFT Fine-Tuning`
> - `Flux Dev Styles Gallery`
> - `VFusion3D Model Release`
> - `SentenceTransformers in Unity` 


- **Apple 发布 Matryoshka Diffusion Models**：Apple 的一名研究员宣布开源一个 Python 包，用于高效训练 **text-to-image diffusion models**，该项目与其 [ICLR 2024 论文](https://github.com/apple/ml-mdm) 相关。
   - 该软件包旨在通过减少数据和计算需求来实现**高质量结果**。
- **Eric 的快速 Llama3 微调演示**：实习生 Eric 将演示他如何使用名为 **ReFT** 的方法在 **14 分钟**内微调 **Llama3**，该方法将表示（representations）集成到隐藏状态中，而不是修改参数。
   - 他的演示将于太平洋时间 **8 月 9 日星期五**上午 10:00 进行，更多信息可在定期日历邀请中查看。
- **新的 Flux Dev 风格画廊上线**：一名成员创建了一个 [GitHub 画廊](https://enragedantelope.github.io/Styles-FluxDev/)，以帮助识别 **Flux Dev** 中的各种风格，这些风格是使用 **ComfyUI** 和 **Mile High Styler** 生成的。
   - 该画廊突出了各种风格但并不详尽，改进 Prompt 提示词可以增强风格在生成图像中的应用效果。
- **Facebook 发布 VFusion3D**：**VFusion3D** 已在 Hugging Face 上发布，点击[此处](https://huggingface.co/spaces/facebook/VFusion3D)可查看 Demo。它展示了作为一个 3D 生成模型的能力，该模型在有限的 3D 数据和广泛的合成多视图数据上训练而成。
   - 这标志着在探索可扩展的 **3D 生成和重建模型**方面迈出了重要一步，是迈向 3D 基础模型的一部分。
- **SentenceTransformers 成功集成到 Unity**：一名成员成功将 **SentenceTransformers** 和 **AutoTokenizer** 集成到 **Unity** 中，这需要构建一个 Shell 来正确显示输出。
   - 这种集成展示了在游戏环境和交互式应用中使用高级模型的潜力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://enragedantelope.github.io/Styles-FluxDev/">Flux Style Test Gallery</a>：暂无描述</li><li><a href="https://huggingface.co/spaces/facebook/VFusion3D">VFusion3D - facebook 创建的 Hugging Face Space</a>：暂无描述</li><li><a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: 以数据和计算高效的方式训练高质量的文本到图像扩散模型</a>：apple/ml-mdm</li><li><a href="https://oxen.ai/community?utm_source=x&utm_content=y">社区资源 | Oxen.ai</a>：使用 Oxen AI 管理您的机器学习数据集。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1271588188321742879)** (3 messages): 

> - `SEE-2-SOUND Presentation`
> - `Hacking with LLMs`
> - `Benchmark Discussion` 


- **SEE-2-SOUND 演示详情**：标题为 *Hugging Face Reading Group 26: SEE-2-SOUND* 的上期会议录像已在 [YouTube](https://www.youtube.com/watch?v=7wkgFR-HYjY) 上线，演讲者为 Rishit Dagli。
   - [GitHub 链接](https://github.com/isamu-isozaki/huggingface-reading-group) 提供了往期演示内容的访问权限。
- **即将举行的关于使用 LLM 进行黑客攻击的讲座**：成员们计划在下周六讨论 *使用 LLM 进行黑客攻击 (hacking with LLMs)*，相关文章已发布在 [Medium](https://medium.com/gopenai/understanding-penetration-testing-with-llms-2b0ec6add14a) 上。
   - 讲座还将涉及一个 **Benchmark**（基准测试），详情将在会议期间分享。



**提及的链接**：<a href="https://www.youtube.com/watch?v=7wkgFR-HYjY">Hugging Face Reading Group 26: SEE-2-SOUND: Zero-Shot Spatial Environment-to-Spatial Sound</a>：演讲者：Rishit Dagli。往期演示：https://github.com/isamu-isozaki/huggingface-reading-group

  

---

### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1271359321824952456)** (8 条消息🔥): 

> - `Dreambooth LoRA 训练脚本`
> - `CLIP 文本编码器支持`
> - `README 中的链接问题`
> - `bf16 与 fp16 训练`
> - `Lora 训练的模型分发` 


- **Dreambooth LoRA 脚本发布**：团队宣布为 FLUX.1 发布了 [Dreambooth LoRA 训练脚本](https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md)，其中包括对 CLIP **文本编码器训练**的支持。
   - 他们警告说 **内存需求** 相当高，并敦促用户查看 README。
- **README 中的断链**：一名成员指出 README 中来自 **@bghira** 的指南链接已失效，团队迅速确认了该问题。
   - 团队回应道：*'感谢发现！将通过 PR 进行修复。'*
- **推荐使用 bf16 训练 LoRA**：讨论了 **LoRA** 是否应该使用 bf16 训练以保持与基础模型的一致性。
   - 一名成员确认：*'是的，我会坚持使用 bf16'，* 但也指出参考 GitHub 的修复后，使用 fp16 可能也能正常工作。
- **请求模型分发支持**：一位用户对 diffusers 中的 **balanced mode** 表示赞赏，该模式允许在 2x 16GB 上运行原生 Flux，并询问了对 **Lora 训练** 的模型分发支持。
   - *'我知道我要求得有点多，'* 他们补充道，强调了对该功能的长期愿景。
- **遇到运行时错误**：一位用户报告在运行 Dreambooth LoRA 训练脚本时遇到了与 shape 尺寸相关的 **RuntimeError**。
   - 错误消息详细说明了 shape 问题：*'shape '[1, 16, 8, 2, 8, 2]' is invalid for input of size 262144.'*


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>: GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_flux.md">diffusers/examples/dreambooth/README_flux.md at main · huggingface/diffusers</a>: 🤗 Diffusers: PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。 - huggingface/diffusers
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1271184281254891652)** (9 条消息🔥): 

> - `用于手写转换的 StrokeSet`
> - `图像标注软件` 


- **用于手写转换的 StrokeSet 概念**：一名成员讨论了将手写图像转换为由大量点组成的笔划构成的 **StrokeSet** 格式，而不是使用 SVG 格式。
   - 他们参考了 [IAM On-Line Handwriting Database](https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format) 获取数据格式规范。
- **探索图像标注工具**：另一名成员正在开发用于边界框（bounding box）图像标注的软件，并正在研究该领域已有的成熟工具。
   - 他们分享了 **[Humans in the Loop](https://humansintheloop.org/)**，其中包含十个开源标注工具的链接，并强调了有效数据集创建的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://fki.tic.heia-fr.ch/databases/iam-on-line-handwriting-database/data-format">Research Group on Computer Vision and Artificial Intelligence &mdash; Computer Vision and Artificial Intelligence</a>: 未找到描述</li><li><a href="https://humansintheloop.org/10-of-the-best-open-source-annotation-tools-for-computer-vision/">Humans in the Loop</a>: Humans in the Loop 通过人类输入提供持续的 ML 模型改进：从数据集收集和标注到模型验证和边缘情况处理。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1271593133976850565)** (2 条消息): 

> - `语音录音样本`
> - `Whisper 实现问题` 


- **音频格式 OGG 中的讽刺**：发现了一个非常有趣的语音录音样本，讽刺地讨论了音频格式 **OGG**。
   - 内容的意外性质引发了成员之间轻松的对话。
- **Whisper 想要将 OGG 转换为 ARG**：一名成员注意到他们本地的 **Whisper** 试图将 **OGG** 文件转换为 **ARG** 格式。
   - 他们开玩笑地建议许多实现可能会在处理这种转换时遇到问题，为讨论增添了幽默感。

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1271181234562928660)** (27 messages🔥): 

> - `LoRA training`
> - `CUDA resource management`
> - `Splitting models across GPUs`
> - `ONNX model quantization`
> - `Device mapping in model loading` 


- **LoRA 训练优于全模型训练**：一位成员建议专注于训练 **LoRAs** 而不是全模型，并指出训练大型架构的收益极小。
   - 另一位成员对加载 **Flux** 进行推理以及训练 **LoRAs** 可能需要的空间表示担忧。
- **CUDA 设置可以有效地聚合 VRAM**：成员们讨论了正确配置 **CUDA** 以便在不同安装环境间轻松管理 **VRAM** 聚合的重要性。
   - 有人指出，尽管训练 **LoRAs** 需要 CUDA 和 Flash attention，但使用多 GPU 的适当技术可以有效地分配资源。
- **关于跨 GPU 拆分模型的讨论**：大家一致认为在多个 GPU 之间拆分单个模型并不简单，建议改为合并 GPU 显存。
   - 一位成员确认，虽然单个模型分片（sharding）是可行的，但由于数据移动的开销，通常会导致延迟增加。
- **探索使用 ONNX 进行模型优化**：一位成员强调了使用 **ONNX** 提取子模型的潜力，这可能比单纯依赖 **Python/PyTorch** 提高效率。
   - 在转向 ONNX 时，将模型量化为 **4 或 8 位**也被认为是一种可行的优化方案。
- **设备映射（Device mapping）与模型加载错误**：一位成员在尝试使用 `device_map='auto'` 时遇到了 `NotImplementedError`，建议改用 `device_map='balanced'`。
   - 讨论引用了 **Hugging Face** 的相关文档和实现细节，强调了多组件模型的建模实践。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/main/en/tutorials/inference_with_big_models#device-placement.">Working with big models</a>：未找到描述</li><li><a href="https://github.com/onnx/onnx/blob/main/docs/PythonAPIOverview.md#extracting-sub-model-with-inputs-outputs-tensor-names">onnx/docs/PythonAPIOverview.md at main · onnx/onnx</a>：机器学习互操作性的开放标准 - onnx/onnx</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/tests/models/test_modeling_common.py#L855">diffusers/tests/models/test_modeling_common.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>：🤗 Diffusers：PyTorch 和 FLAX 中最先进的用于图像和音频生成的扩散模型。 - huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/tests/models/test_modeling_common.py#L955">diffusers/tests/models/test_modeling_common.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>：🤗 Diffusers：PyTorch 和 FLAX 中最先进的用于图像和音频生成的扩散模型。 - huggingface/diffusers</li><li><a href="https://github.com/huggingface/diffusers/blob/65e30907b5df1338ae65a20b78866c87e061c952/src/diffusers/models/transformers/pixart_transformer_2d.py#L81">diffusers/src/diffusers/models/transformers/pixart_transformer_2d.py at 65e30907b5df1338ae65a20b78866c87e061c952 · huggingface/diffusers</a>：🤗 Diffusers：PyTorch 和 FLAX 中最先进的用于图像和音频生成的扩散模型。 - huggingface/diffusers
</li>
</ul>

</div>
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1271204423254675560)** (56 messages🔥🔥): 

> - `DALL·E 3 updates`
> - `Gemini 1.5 price cuts`
> - `Deep-Live-Cam deepfake`
> - `Anysphere fundraising`
> - `Llama 3.1 updates`

- **DALL·E 3 允许免费用户创建图像**：OpenAI 宣布 ChatGPT Free 用户现在每天可以使用 DALL·E 3 创建最多 **两张图像**，用于个人和专业需求。
   - 然而，社区反馈不一，一些人对与其他模型相比的限制表示失望。
- **Gemini 1.5 迎来大幅降价**：最近的更新显示，Gemini 1.5 Flash 的价格下调了高达 **70%**，使其在 AI 领域更具竞争力，同时 GPT4o 也大幅降价。
   - 分析人士指出，这种降价趋势全面提高了效率，预示着市场将持续竞争。
- **Deep-Live-Cam 实现实时 Deepfakes**：一个名为 **Deep-Live-Cam** 的热门 GitHub 项目允许用户通过单张图片实时流式传输创建高质量的 Deepfakes。
   - 实验展示了令人印象深刻的实时能力，引发了人们对其在虚拟会议中潜在应用的兴奋。
- **Anysphere 在 A 轮融资中筹集 6000 万美元**：AI 编程助手 Cursor 的开发商 Anysphere 获得了 **超过 6000 万美元** 的 Series A 融资，估值达到 **4 亿美元**。
   - 本轮融资由 Andreessen Horowitz 领投，显示出投资者对 AI 驱动的编程解决方案的强大信心。
- **Meta 更新 Llama 3.1 模型**：Meta 发布了新版本的 **Llama 3.1 405B 模型**，将 KV heads 从 16 个更改为 8 个，使其与白皮书规范保持一致。
   - 此次更新引发了对其在性能和模型架构方面影响的推测和兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://facebookresearch.github.io/nougat/">Nougat</a>: 未找到描述</li><li><a href="https://junlinhan.github.io/projects/vfusion3d.html">Creative and Descriptive Paper Title.</a>: 论文描述。</li><li><a href="https://homebrew.ltd/blog/can-llama-3-listen">Homebrew</a>: 构建运行在高效能硬件上的增强人类能力的 AI。</li><li><a href="https://x.com/swyx/status/1821771182443540752?s=46">swyx 🍓 (@swyx) 的推文</a>: 根据我的计算，GPT-4o 的降价和 Gemini 1.5 Pro 八月的结果是相对于 2024 年第二季度“价格-智能前沿”的“补涨”。但我们现在处于第三季度……而且一直没有停歇……</li><li><a href="https://x.com/matthewberman/status/1821949143918489794?s=61">MatthewBerman (@MatthewBerman) 的推文</a>: 目前 GitHub 排名第一的热门仓库看起来太疯狂了。单张图片即可实现直播 Deepfake。看看这质量！！它叫 Deep-Live-Cam（链接在回复中）</li><li><a href="https://x.com/jay_wooow/status/1821989710207840730?s=46&t=6FDPaNxZcbSsELal6Sv7">joao (@jay_wooow) 的推文</a>: @MatthewBerman 一些实验——它运行得几乎完美，而且完全是实时的。我花了 5 分钟就安装好了。</li><li><a href="https://x.com/OpenAI/status/1821644904843636871">OpenAI (@OpenAI) 的推文</a>: 我们正在向 ChatGPT 免费用户推出每天使用 DALL·E 3 创建最多两张图片的功能。只需让 ChatGPT 为幻灯片创建图片、为朋友定制卡片，或者展示……</li><li><a href="https://x.com/MatthewBerman/status/1821949143918489794">MatthewBerman (@MatthewBerman) 的推文</a>: 目前 GitHub 排名第一的热门仓库看起来太疯狂了。单张图片即可实现直播 Deepfake。看看这质量！！它叫 Deep-Live-Cam（链接在回复中）</li><li><a href="https://x.com/altryne/status/1821617540227068017?s=46">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 这也是新功能！PDF 理解的新多模态能力 👀 抄送 @simonw 引用 Logan Kilpatrick (@OfficialLoganK) 的话：@GoogleAI 开发者们的好消息：- Gemini 1.5 Flash 现在的价格是……</li><li><a href="https://x.com/jay_wooow/status/1821989710207840730?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">joao (@jay_wooow) 的推文</a>: @MatthewBerman 一些实验——它运行得几乎完美，而且完全是实时的。我花了 5 分钟就安装好了。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1eoin62/meta_just_pushed_a_new_llama_31_405b_to_hf/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://youtu.be/IvjLXGR7-vM?si=KWCzye6rKoBv--uL">信任与安全的开放方法：Llama Guard 3, Prompt Guard 及更多 | Llama 开发者专栏</a>: 下载 Llama 3.1 ➡️ https://go.fb.me/kbpn54 来自 Meta 的 Llama 信任与安全团队的 Zacharie Delpierre Coudert 和 Spencer Whitman 加入我们的讨论……</li><li><a href="https://youtu.be/02fBBoZa9l4?feature=shared">马克·扎克伯格谈 Llama, AI, &amp; Minus One</a>: 马克·扎克伯格是如何将 Facebook 变成 Meta 的？为什么他认为开源是 AI 的未来？以及他是如何在公司保持 Minus One 心态的……</li><li><a href="https://youtu.be/Lba_MBZsR5s?si=tdMGsm3m0eQUFpg5">使用 DSPy 设计可靠的 AI 系统 | 神经搜索访谈：Omar Khattab</a>: 在本期神经搜索访谈中，我们与 ColBERT 和 DSPy 等热门 IR 和 LLM 框架的作者 Omar Khattab 进行了交流。Omar 描述了……</li><li><a href="https://techcrunch.com/2024/08/09/anysphere-a-github-copilot-rival-has-raised-60m-series-a-at-400m-valuation-from-a16z-thrive-sources-say/">独家：据消息人士称，GitHub Copilot 的竞争对手 Anysphere 已从 a16z、Thrive 筹集了超过 6000 万美元的 A 轮融资，估值为 4 亿美元</a>: Anysphere 是一家成立两年的初创公司，开发了一款名为 Cursor 的 AI 驱动编程助手，已在 A 轮融资中筹集了超过 6000 万美元……</li><li><a href="https://www.promptfiddle.com/gpt4o-pdf-extraction-4RHoa">gpt4o pdf extraction — Prompt Fiddle</a>: 未找到描述</li><li><a href="https://findoc-gloo.vercel.app/">BAML 演示</a>: 未找到描述</li><li><a href="https://github.com/BoundaryML/baml-examples/blob/main/findoc/src/app/actions/extract-pdf.ts#L17">baml-examples/findoc/src/app/actions/extract-pdf.ts (main 分支) · BoundaryML/baml-examples</a>: 通过在 GitHub 上创建账号，为 BoundaryML/baml-examples 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1271558940135784529)** (147 messages🔥🔥): 

> - `用于 AI 开发的 Ruby`
> - `AI 咨询与客户获取`
> - `Prompt 编写工作坊`
> - `研究型 Agent`
> - `AI 工具与自动化` 


- **Ruby 社区在 AI 应用领域不断扩大**：有一个虽小但正在增长的社区专注于使用 **Ruby** 构建 AI 应用，Ruby 因其能够有效地创建领域特定语言 (DSLs) 而受到关注。
   - 一位成员正在开发一个名为 Boxcars 的项目，旨在提升 Ruby 在 AI 方面的能力，强调了其在 **LLM coding** 方面的潜力。
- **AI 咨询与项目协作建议**：成员们讨论了 **AI 咨询** 的前景，强调了识别繁琐任务并进行自动化的重要性，以此在咨询领域展示技能。
   - 建议包括使用 **Elicit** 等工具进行问题发现，询问客户在工作中感到困扰的地方。
- **对协作研究型 Agent 的兴趣**：几位成员表示有兴趣探索 **research agents**，其中一人建议使用研究型 Agent 对该主题本身进行研究，例如使用 Elicit 来分析其文档。
   - 提议进行协作，并可能在两周内为更深入的讨论做准备。
- **Prompt 编写工作坊受到关注**：人们对举办 **prompt crafting workshops** 持续关注，特别是旨在帮助没有编程或机器学习背景的人学习如何有效地引导模型。
   - 参与者一致认为，缩小模型范围并在随后将其连接以实现实际解决方案具有重要价值。
- **Context 在 AI 开发中的重要性**：**Context** 被强调为创建有效 AI 工具的关键，专为执行特定任务而设计的工具因其效率而受到关注。
   - 成员们讨论了使用通用工具与针对特定应用进行定制的优势之间的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/thedayisntgray">来自 undefined 的推文</a>: 未找到描述</li><li><a href="https://docs.sublayer.com/">Sublayer - Ruby AI Framework</a>: 未找到描述</li><li><a href="https://translation-cellium.ngrok.app/">Translations</a>: 未找到描述</li><li><a href="https://tenor.com/bNGaG.gif">Boo GIF - Boo - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit?gid=0#gid=0">AI In Action: Weekly Jam Sessions</a>: 未找到描述</li><li><a href="https://tenor.com/bc64D.gif">Gimme Code Gimme GIF - Gimme Code Gimme Code - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://artium.ai/apex">APEX - AI 产品规划的未来</a>: APEX 是放置创意并探索无限可能的地方。在 15 分钟内消除数月的研讨会和无休止的规划。</li><li><a href="https://github.com/facebookresearch/faiss/issues/2359">正确发音 · Issue #2359 · facebookresearch/faiss</a>: 抱歉问这么愚蠢的问题。我到处都找不到这个信息。faiss 的正确发音是什么？像 facebook 这个词？还是像 feisty 这个词？提前感谢</li><li><a href="https://tenor.com/Ynql.gif">Jack Nicholson Yes GIF - Jack Nicholson Yes Nodding - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1271186400787697758)** (178 条消息🔥🔥): 

> - `Perplexity Pro 限制`
> - `订阅问题`
> - `图像生成复杂度`
> - `模型使用清晰度`
> - `浏览器集成` 


- **Perplexity Pro 限制下调**：多位用户报告 Pro 搜索限制最近从 **600** 次下调至 **450** 次，并预计很快将进一步降至 **300** 次。
   - 这一变化是在没有事先通知的情况下做出的，引发了订阅者对未来限制和透明度的担忧。
- **订阅购买挑战**：用户在 Stripe 支付平台遇到问题，导致使用各种支付方式购买 Pro 订阅时出现困难。
   - 一些人怀疑自己可能被封禁，而另一些人则报告说通过聊天或电话支持无法获得帮助。
- **图像生成困难**：一位用户对 Perplexity 中图像生成的复杂性表示沮丧，质疑为什么不能像输入提示词并点击按钮那样简单。
   - 回复指出，目前的图像生成工具功能有限，可能不符合用户的实际需求。
- **模型使用误区**：用户注意到在使用 Perplexity 时，系统默认使用 Pro 模型，但在如何切换可用模型方面存在困惑。
   - 一些用户报告无法重写或删除 Thread，表明平台正经历技术问题。
- **将 Perplexity 设置为默认搜索**：一位用户分享了他们将 Perplexity 设置为浏览器默认搜索引擎的经历，指出虽然失去了一些便利性，但将其视为一个适应过程。
   - 其他人也尝试将 Perplexity 集成到他们的工作流程中，在易用性和便利性之间寻找平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/aravsrinivas/status/1821637031002566671?s=61">Aravind Srinivas (@AravSrinivas) 的推文</a>: 浅色模式 引用 Aravind Srinivas (@AravSrinivas) 不要作恶 (Don't be evil)</li><li><a href="https://www.perplexity.ai/page/google-loses-search-antitrust-euX29JMyQJWU8YAEGeROzw>)">Perplexity</a>: Perplexity 是一款免费的 AI 驱动型回答引擎，可针对任何问题提供准确、可靠且实时的答案。</li><li><a href="https://x.com/aravsrinivas/status/1821785478183694579?s=61">Aravind Srinivas (@AravSrinivas) 的推文</a>: 🧵 在此处回复包含幻觉 (hallucinations) 和上下文丢失问题的 Perplexity 永久链接。这将非常有趣且具有教育意义！</li><li><a href="https://www.perplexity.ai/search/are-there-any-plans-by-perplex-iskQH6kpQh6IpJFJjbiRqg">Perplexity 是否有计划将 Gemini 1.5 Pro 添加到列表...</a>: 根据现有信息，Perplexity AI 目前尚未公开宣布将 Gemini 1.5 Pro 添加到其可用语言模型列表中的计划...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1271201418715467922)** (12 条消息🔥): 

> - `OpenAI 的 Strawberry 模型`
> - `小数比较`
> - `国防科技公司 Anduril 估值`
> - `滞留宇航员返回时间表`
> - `AI 辅助医疗倡导`

- **OpenAI 的 Strawberry 模型引发关注**：OpenAI 的新模型 **'Strawberry'** 旨在增强 AI 推理能力并处理复杂的研究任务，在 AI 社区引发了巨大轰动。
   - Sam Altman 在社交媒体上关于草莓的暗示被解读为这一创新项目的线索，点燃了爱好者的热情。
- **比较 3.33 和 3.4 的大小**：比较结果显示 **3.4 大于 3.33**，强调了对齐小数点以进行准确评估的重要性。
   - 这种方法有助于在科学和金融等领域进行精确测量，在这些领域中，即使是微小的差异也具有重要意义。
- **Anduril 估值达到 140 亿美元**：国防科技初创公司 **Anduril Industries** 已融资 15 亿美元，目前估值高达 **140 亿美元**，较此前的 85 亿美元估值大幅跃升。
   - 在政府合同和大公司投资的推动下，该公司的营收翻了一番，达到约 **5 亿美元**。
- **滞留宇航员返回推迟**：NASA 官员宣布，自 2024 年 6 月以来滞留在国际空间站的两名宇航员可能要到 **2025 年 2 月**才能返回地球。
   - 延迟是由于 **Boeing Starliner** 舱体的机械故障，这引发了对宇航员回家旅程的安全担忧。
- **AI 工具改变医疗倡导**：创新公司正在开发 **AI 工具**，以协助分析医疗笔记并帮助个人管理健康。
   - 这些进步为应对乳房植入物疾病的女性提供了必要的支持，增强了她们的理解和医疗体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/search/decimal-comparisons-3-33-vs-3-TtUoN0wVRhqXcBAb_tX.Ww">小数比较：3.33 vs 3.4</a>: 好的,我會用中文回答你關於小數比較的問題。  3.4 大於 3.33  1. 首先,我們要將這兩個小數對齊到相同的小數位:    3.33    3.40  2. 從左到右比較每一位數字:    - 整數部分都是3,相等    - 第一位小數都是3,相等    - 第二位小數,3.4的4大於3.33的3  3....</li><li><a href="https://www.perplexity.ai/search/what-insights-does-the-200-day-NJPf1o6GQ9C5OXy.forfmA">200 日 SMA 为 BTC-USD 交易对提供了哪些见解？</a>: 200 日简单移动平均线 (SMA) 为 BTC-USD 交易对提供了几个关键见解： 1. 趋势识别： - 200 日 SMA 帮助交易者...</li><li><a href="https://www.perplexity.ai/search/core-framework-NQi9hl9ySrKJX9eE4bcSoA#0">核心框架</a>: “核心框架”一词根据上下文可以指代几种不同的概念。以下是一些主要的解释： NIST...</li><li><a href="https://www.perplexity.ai/search/why-are-my-silver-spoons-disco-GAuUmAZXTtygkMgLKt3diQ">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/page/defence-tech-anduril-hits-14b-ZeHfS6seRCSRPwp4qitrbg">国防科技公司 Anduril 估值达到 140 亿美元</a>: 据 TechCrunch、Bloomberg 等媒体报道，国防科技初创公司 Anduril Industries 在新一轮融资中筹集了 15 亿美元，估值达到...</li><li><a href="https://www.perplexity.ai/page/enbloc-website-ai-assisted-med-j39p4Uv8SAysUM.iyXVcdA">Enbloc 网站：AI 辅助医疗倡导</a>: 领先的健康管理 AI 工具开发商正在彻底改变患者护理，并赋予个人掌控自身健康的能力。这些...</li><li><a href="https://www.perplexity.ai/search/how-can-i-use-inline-padding-vSnbnuMzStiI1sZKGs4HhA?__cf_chl_rt_tk=Ssx1APDVq7zXMbPOHijPl8mFfEfk1JbgdUF6zBGNwT8-1723119550-0.0.1.1-8468">Perplexity</a>: Perplexity 是一款免费的 AI 驱动问答引擎，可针对任何问题提供准确、可靠且实时的回答。</li><li><a href="https://www.perplexity.ai/page/decimal-comparisons-9-9-vs-9-1-efHZQQZGT6SQZ_GFVODu_g">小数比较：9.9 vs 9.11</a>: 小数比较虽然看似简单，但可能出人意料地棘手，正如这个问题所证明的：9.9 比 9.11 大还是小？这...</li><li><a href="https://www.perplexity.ai/page/stuck-astronauts-will-return-i-00a_0jYZRKGrA38tE5uaGA">滞留宇航员将于 2025 年返回</a>: 据 NASA 官员称，由于 Boeing Starliner 的问题，自 2024 年 6 月以来一直滞留在国际空间站的两名宇航员...</li><li><a href="https://www.perplexity.ai/page/openai-s-strawberry-model-a-ne-w.mxwLOcRSGrgiKFhCD2Gw">OpenAI 的 "Strawberry" 模型：AI 推理的新前沿</a>: OpenAI 的 "Strawberry" 模型是一项尖端的 AI 计划，因其在增强推理能力和处理复杂任务方面的潜力而引发关注...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1271237725768515584)** (9 条消息🔥): 

> - `Google Maps URL 效率`
> - `联网功能的 API 路线图`
> - `在线模型的使用成本`
> - `中文搜索结果的质量`
> - `以 JSON 格式搜索维基百科页面` 


- **Google Maps URL 难题**：一位用户在获取日常行程的准确 Google Maps URL 时面临挑战。
   - *是否有高效获取 URL 的方法，还是说这根本无法实现？*
- **询问联网功能的 API 路线图**：一位成员询问了将 **internet access**（联网）和 **PRO search** 功能添加到 API 的路线图。
   - 另一位成员澄清说，名称中带有 'online' 的模型具有一定的联网能力，但并非实时。
- **在线模型费用说明**：有人询问使用在线模型的收费结构，想知道是 1000 次搜索后收费 5 美元，还是每次查询收费 0.005 美元。
   - 回复确认是 **每次查询 0.005 美元**，引发了关于额度消耗过快的讨论。
- **对中文搜索结果质量的担忧**：一位成员分享了搜索中文资源的经验，认为结果质量可能低于预期。
   - 不过，他们指出，尽管有这些担忧，搜索中文维基百科页面仍能获得 **可靠的结果**。
- **维基百科搜索结果的 JSON 格式**：一位用户展示了一个 Prompt，用于搜索关于 **中华人民共和国** 的维基百科页面，并以 JSON 格式输出相关的 URL 和内容。
   - 他们还强调需要评估内容是否包含有关 **中华人民共和国** 的相关信息。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1271551907202076765)** (22 条消息🔥): 

> - `NeurIPS 经验`
> - `Rebuttal 策略`
> - `会议论文发表的挑战`
> - `审稿人置信度的影响`
> - `小规模会议 vs. 大规模会议` 


- **应对 NeurIPS 发表流程**：一位成员分享了他们在 NeurIPS 的经历，表达了在顶级 AI 会议上获取高质量反馈和发表论文的压力。
   - *这个过程非常令人不知所措，我不认识任何在顶级 AI 会议上发表过论文的人。*
- **针对审稿评分的 Rebuttal 策略**：有人就 Rebuttal 提出了建议，强调如果审稿人的置信度（confidence）较低，可能不值得在 Rebuttal 过程中投入过多精力。
   - 一位成员指出：*如果他们说明了置信度低的原因，你可以尝试解决，否则我不会理会。*
- **大型会议的挑战**：讨论强调了大型会议令人望而生畏的特性，成员们建议参加更小众的会议以获得更好的体验。
   - 一位参与者提到在顶会发表论文的重要性，称：*感觉一个人至少得在大型会议上发表一次论文才能被认真对待。*
- **鼓励尝试其他渠道**：成员们鼓励建设性地对待反馈，并考虑将论文投往其他地方，强调被拒绝是这一过程中的常态。
   - 有人表示：*论文在不同会议中经历几轮评审/被拒是很正常的。*
- **研究价值高于会议名声**：评论认为，优秀的研究通常会出现在 arXiv 上，而不是通过传统的会议渠道，无论在何处发表，其价值依然存在。
   - 一位成员举例说原始的 DQN 论文最初只是 workshop 论文，这表明有影响力的工作可以从较小的平台脱颖而出。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1271203318881845404)** (128 messages🔥🔥): 

> - `RLHF 清理讨论`
> - `Qwen2 模型行为`
> - `Expandable Segments 实现`
> - `训练中的内存管理`
> - `小模型的宣传` 


- **关于 RLHF 清理的讨论**：成员们讨论了在发布进一步公开公告之前，有必要对 RLHF 流程进行潜在的清理。
   - 建议编写教程或博客文章，但共识是这可能需要额外的准备时间。
- **Qwen2 模型表现出异常的内存行为**：测试显示 **Qwen2** 模型在训练期间（尤其是 batch size 为 4 时）有显著的预留内存（reserved memory），这表明可能存在问题。
   - 成员们表示有兴趣进一步分析这一行为，质疑这是否预示着内存泄漏。
- **提议实现 Expandable Segments**：由于对性能影响极小，有人提议在模型配置中默认启用 `expandable_segments:True`。
   - 虽然有人担心这是否会导致任何破坏，但许多人认为如果需要，可以很容易地将其关闭。
- **训练期间的内存管理挑战**：用户报告称，尽管进行了调整，但在训练 Qwen 模型的 0.5B 和 1.5B 变体时仍遇到 OOM (Out of Memory) 错误。
   - 注意到使用 Attention (AC) 对吞吐量没有显著影响，这表明可能需要不同的策略来优化性能。
- **小模型的公关策略**：成员们讨论了起草关于 Qwen2 等小模型的公开公告，强调它们在有限硬件上运行的能力。
   - 建议快速推进在各模型中添加对 expandable segments 的支持，目标是在未来的版本中被广泛采用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/salman-mohammadi/torchtune/runs/vh7uhuz1/overview">salman-mohammadi</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune/runs/3guul08l?nw=nwuserrdoublea">salman-mohammadi</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/jcummings/small-model-large-reserved-memory/runs/mqo9mayl?nw=nwuserjcummings">jcummings</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune/runs/djmo8cwh?nw=nwuserrdoublea">salman-mohammadi</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://wandb.ai/salman-mohammadi/torchtune?nw=nwusersalmanmohammadi">salman-mohammadi</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/pytorch/torchtune/blob/f6ddfccf7b9f9e8facb1fc42d2a2c5635d34d8ed/torchtune/models/qwen2/_tokenizer.py#L96">torchtune/torchtune/models/qwen2/_tokenizer.py at f6ddfccf7b9f9e8facb1fc42d2a2c5635d34d8ed · pytorch/torchtune</a>: 一个用于 LLM 微调的原生 PyTorch 库。可以通过在 GitHub 上创建账号来为 pytorch/torchtune 做出贡献。</li><li><a href="https://www.ventoy.net/en/index.html">Ventoy</a>: Ventoy 是一个开源工具，用于为 ISO 文件创建可启动的 USB 驱动器。使用 ventoy，你不需要一遍又一遍地格式化磁盘，你只需要将 iso 文件复制到 USB 驱动器并启动...</li><li><a href="https://github.com/pytorch/torchtune/issues/1278">[RFC] Optimizer CPU offload from torchao for single GPU low memory config · Issue #1278 · pytorch/torchtune</a>: torchao 中最近新增的 optimizer CPU offload 对于单 GPU 低显存配置非常有用。https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim#optimizer-cpu-offload...</li><li><a href="https://huggingface.co/datasets/Anthropic/hh-rlhf">Anthropic/hh-rlhf · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1271392626746327071)** (10 messages🔥): 

> - `PyTorch Profiler 内存泄漏`
> - `4090 的 Tensor Core 规格` 


- **关于 PyTorch Profiler 内存泄漏的困惑**：一位成员在使用 PyTorch Profiler 且设置 `profile_memory=True` 时遇到了 **内存泄漏**，不确定是哪部分设置导致的。
   - 另一位成员确认他们也遇到了类似问题，并改用 `torch.cuda.memory._record_memory_history()` 进行内存分析。
- **关于 4090 Tensor Core 规格的讨论**：一位用户询问在哪里可以找到 4090 或任何其他显卡的 **Tensor Core** 详细规格。
   - 一位成员建议搜索 4090 的 **Ada 白皮书**，并指出 **Ampere 白皮书** 包含 3090 的详细信息。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1271279698822824039)** (6 条消息): 

> - `torch.compile kernels`
> - `CUDA kernels visibility`
> - `torchao import error`
> - `Cutlass backend progress` 


- **torch.compile 主要使用 Triton kernels**：成员们讨论了 **torch.compile** 大多输出 **Triton kernels**，因为它们更易于编写和生成，同时保持了强大的性能。
   - 虽然 **PyTorch eager** 模式确实输出 **CUDA kernels**，但 **torch.compile** 也有一个 Cutlass 后端，尽管其目前进度尚不明确。
- **询问 eager 模式下 CUDA kernels 的可见性**：一位成员询问是否有办法查看 eager 模式生成的 **CUDA kernels**，类似于 **torch.compile** 的操作方式。
   - 这反映了用户对 **PyTorch** 框架内不同模式下 kernel 输出的持续关注。
- **报告 torchao 导入问题**：报告了一个关于 `from torch._inductor.runtime.runtime_utils import do_bench` 的导入问题，表明它在 nightly build 中已损坏。
   - 该问题被确认会影响 **torchao**，凸显了在 nightly 版本中维护更新的重要性。
- **torchao 导入问题已有解决方案**：另一位成员指出，导入问题已在最新版本的 **ao** 中修复，建议用户从最新的 main 分支进行合并。
   - 该解决方案为遇到导入错误的位用户提供了恢复功能的途径。


  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1271334549229666314)** (11 条消息🔥): 

> - `Flash Attention Paper`
> - `Cooperative Thread Array`
> - `Memory Access Issues`
> - `KV Block Ordering`
> - `Synchronization in CUDA` 


- **澄清 Flash Attention Block 顺序**：关于 **Flash Attention** 论文的讨论澄清了 KV blocks 的排序并不关键，因为它们可以以不同的顺序进行调度，而不会改变算法的正确性。
   - 然而，K 和 V 行的配对必须保持其逻辑对应关系，算法才能正常运行。
- **理解 Cooperative Thread Array (CTA)**：成员们解释说，CTA（即 **cooperative thread array**）是指 CUDA 中的并行工作单元，允许同一 CTA 中的线程访问 shared memory。
   - 这一概念至关重要，因为它界定了 CUDA 执行模型中 warps 与 CTAs 之间的关系。
- **朴素最大值计算的潜在问题**：有人提出了在跨具有不同 `j` 值的多个线程计算 `m_i = max(m_i, s_ij)` 时，内存访问的原子性问题。
   - 讨论明确了虽然特定线程的内存访问是有序的，但跨不同线程的可见性并不能保证原子性，因此需要同步（synchronization）。
- **Flash Attention 中 'i' 和 'j' 的意义**：讨论澄清了在 **Flash Attention** 算法中，`i` 是空间坐标，而 `j` 代表与处理的 KV blocks 数量相关的逻辑时间维度。
   - 这一区别对于理解算法内部如何统计最大值条目非常重要。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1271492294377209947)** (16 条消息🔥): 

> - `INT8 Quantized Training Issues` (INT8 量化训练问题)
> - `Observer Implementation for Quantization` (量化 Observer 实现)
> - `Blockwise Quantization Observer` (分块量化 Observer)


- **INT8 量化训练错误解决**：一位成员在为 **INT8 量化训练** 相关的子类实现 FSDP 支持时遇到了错误，特别是在对一个需要 grad 的 tensor 调用 `torch.chunk()` 时。
   - 他们注意到，在 `aten.split.Tensor` 的实现中设置 `requires_grad=False` 解决了该问题，这表明 PyTorch 内部有一些管理这一方面的逻辑。
- **静态量化中的 Observer 使用**：一位成员指出，在静态量化校准流程教程中，observer 是从 **torch.ao** 而不是 **torchao** 仓库导入的，这暗示了 observer 类可用性的混淆。
   - 他们提到可以不使用 observer 手动计算平均值，并幽默地评论了使用 `model(inputs).abs().mean()` 的简便性。
- **关于分块量化 Observer 的讨论**：强调了在 **torchao** 中为分块量化重新实现 observer 的需求，一位成员表示尽管时间有限，仍打算创建一个通用的 observer。
   - 另一位成员分享了一个 [GitHub Pull Request](https://github.com/pytorch/ao/pull/650) 用于 `AffineQuantizedObserver`，并建议针对特定应用（特别是 Adaptive Weight Quantization (AWQ)）可能需要进行定制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/650">Add AffineQuantizedObserver by jerryzh168 · Pull Request #650 · pytorch/ao</a>：摘要：在我们的 static_quant 流程教程中，我们仍在使用计划弃用的 torch.ao 中的 observer，此 PR 为 AffineQuantizedTensor 添加了一个更通用的 observer，并表明...</li><li><a href="https://github.com/pytorch/ao/pull/644/files">Add experimental INT8 quantized training by gau-nernst · Pull Request #644 · pytorch/ao</a>：解决 #554（但未关闭）
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1271543496230043748)** (1 条消息): 

> - `Python User Survey` (Python 用户调查)
> - `Community Feedback on Free Threading` (关于 Free Threading 的社区反馈)


- **NVIDIA 征求 Python 用户见解**：NVIDIA 正在通过一份简短的 [调查问卷](https://www.surveymonkey.com/r/FHF2RTL) 收集 Python 用户的反馈，以更好地了解他们在 CUDA Python 产品方面的经验和挑战。
   - 该调查是匿名的，回复将有助于根据社区需求确定未来功能的优先级。
- **敦促社区讨论 free threading**：一条消息强调了在社区向 **free threading** 过渡期间，社区就可提供协助的领域提供意见的重要性。
   - 鼓励用户参与，以便有效地定制支持和资源。



**提到的链接**：<a href="https://www.surveymonkey.com/r/FHF2RTL">CUDA Python Survey</a>：参与由 surveymonkey.com 提供的调查。免费创建您自己的调查。

  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1271181581423349892)** (85 条消息🔥🔥): 

> - `RoPE 实现`
> - `KV Cache 优化`
> - `代码中的复数`
> - `内存管理技术`
> - `PyTorch 的驱动问题` 


- **RoPE 重构讨论**：成员们讨论了当前 **RoPE kernel** 的实现，承认其复杂性，并探讨了使用显式三角函数代替复数的潜在好处，重点在于提高代码的可读性。
   - 分享了一个更旧、更简单的版本，它完全避免了复数，并实现了直接的旋转。
- **KV Cache 集成**：一位成员提到成功为 Llama31 实现了 **KV Cache** 优化，在 80GB GPU 上实现了高内存利用率的全量 **bfloat16 finetune**。
   - 他们表示希望在训练期间进一步优化内存使用，以缓解在推进概念验证（proof-of-concept）过程中的限制。
- **对复数逻辑的担忧**：围绕在代码中使用复数的逻辑展开了讨论，强调了清晰注释的必要性，以确保团队成员之间的可理解性。
   - 成员们一致认为，简化代码以使用正弦（sine）和余弦（cosine）可能会使其更易于上手和理解。
- **GPU 训练中的内存管理**：一个相关的 GitHub pull request 讨论了在设备内存耗尽时使用 **cudaMallocManaged** 管理内存，从而允许在受限系统上继续训练。
   - 对话建议利用 C 库进行分配，并结合动态内存管理方案使用 `torch.frombuffer`。
- **驱动兼容性修复**：一位成员在经历 AMD 驱动直通模式（passthrough mode）的问题后，通过恢复标准驱动设置并更新到 ROCm 6.2，解决了 **PyTorch** 的性能问题。
   - 这一调整使得他们在 llm.c 框架的使用中获得了更一致且可靠的结果。



**提到的链接**：<a href="https://github.com/karpathy/llm.c/pull/709">Allocate managed memory if device memory runs out by ngc92 · Pull Request #709 · karpathy/llm.c</a>：如果设备内存耗尽，则使用 cudaMallocManaged 分配优化器状态，以便在无法容纳优化器状态的情况下仍能（缓慢地）进行训练。这是基于 #694 的，应该是 m...

  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1271449050474086482)** (12 条消息🔥): 

> - `云端 GPU 租赁`
> - `Runpod 的 Profiling 问题`
> - `MI250 的可用性`
> - `购买 GPU 的性价比` 


- **寻求云端 GPU 租赁建议**：一位成员表示有兴趣为 **CK** 租赁 GPU，特别是询问了合适供应商的建议。
   - 他们强调更倾向于选择不会使 **profiling** 任务复杂化的服务。
- **对 Runpod 上 profiling 的担忧**：另一位成员建议使用 Runpod 的 **MI250**，但有人对该平台上的 profiling 问题提出了担忧。
   - “我可能记错了，但如果我没记错的话，人们在 Runpod 上进行 profiling 时遇到了困难”被作为警告提及。
- **MI GPU 的可用性有限**：成员们注意到托管 **MI300** GPU 的选项较少，并提到由于 **AMD hypervisor** 的限制，需要租赁整台机器。
   - 一位用户评论道：“AMD 还没有意识到你需要修补你的 hypervisor 来启用 PCIe atomics。”
- **7900XTX 作为潜在的低成本替代方案**：由于成本考虑，一位用户建议购买 **7900XTX** 可能比租赁云端 GPU 更便宜。
   - 这一言论反映了人们在评估成本与云服务之间日益增长的情绪。
- **推荐从 AMD 方面获取更多 MI250 信息**：有人建议联系一位在 AMD 负责 **llmc** 的成员，表示他们可能对获取 **MI250** GPU 有见解。
   - 另一位用户提到，该联系人是从 **Hyperbolic** 获取 **MI250** 的，这可能是一个有价值的资源。


  

---

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1271356591408676895)** (3 messages): 

> - `Bitnet model`
> - `AO integration`
> - `Quantization Aware Training (QAT)` 


- **运行 Bitnet 模型的会议安排**：会议定于 <t:1723309200:R> 举行，主要目标是实现一个集成了 **AO** 的可用 **bitnet model**。
   - 一位成员表示愿意提供帮助，尽管会议时间对他来说是**凌晨 1 点**，并幽默地提到他那时可能正在睡觉。
- **关于 Bitnet 的 QAT 方法的见解**：一位成员提到他们还没有深入阅读 **bitnet paper**，但观察到其似乎正在利用**量化感知训练 (Quantization Aware Training, QAT)**。
   - 会中指出，**master weight** 仍以 **FP32/BF16** 格式维护，这表明了一种特定的模型精度处理方法。


  

---


### **CUDA MODE ▷ #[cudamode-irl](https://discord.com/channels/1189498204333543425/1267896441989234709/1271496058563727360)** (2 messages): 

> - `Pure UNet optimization`
> - `From scratch model implementations` 


- **优化纯 UNet 的速度**：一位成员对一个旨在让 [pure UNet](https://github.com/clu0/unet.cu) 运行速度超过 **torch.compile** 的项目表示了兴趣。
   - *他们愿意与其他成员合作*，以提升模型的性能。
- **对从零开始的模型实现感到兴奋**：另一位成员强调，**from scratch model implementations**（从零开始的模型实现）作为一个项目构思听起来非常酷。
   - *这种方法可能会带来创新的解决方案，并加深对各种模型架构的理解*。


  

---



### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1271190433388695678)** (98 messages🔥🔥): 

> - `CBRN Risks and AI Filtering`
> - `AI Safety Measures`
> - `Career Transition Grants in AI`
> - `AI Models and Ethical Guidelines`
> - `GPU Resources for AI Research` 


- **辩论 AI 模型中的 CBRN 风险**：关于从 AI 训练数据中过滤 CBRN 相关信息是否能有效降低风险而不削弱模型的科学能力，展开了广泛的讨论。
   - 参与者争论了知识移除的复杂性，以及如果模型足够智能，是否仍有可能生成有害输出。
- **AI 安全研究的机会**：一位成员强调了来自 Open Philanthropy 的一项专注于 AI 安全和机械可解释性 (mechanistic interpretability) 的职业转型资助，并正在寻求 GPU 资源以协助教学练习。
   - 讨论了多种获取 GPU 的替代方案，包括 Colab、vast.ai 以及用于 AI 研究的 CAIS 集群。
- **替代 GPU 资源**：社区为学习者分享了几个 GPU 资源选项，包括用于 T4 GPU 的 Kaggle，以及利用 Apple M1 板载能力的建议。
   - 建议包括免费层级和租赁选项的组合，以高效支持机器学习练习。
- **AI 研究输出中的伦理**：对话集中在确保 AI 模型不会生成有害指令，反思了必要专业知识与危险知识之间的平衡。
   - 一些参与者表示担心，删除敏感信息可能会限制对负面案例的理解，并产生潜在的信息危害。
- **AI 代码编辑器的技术能力**：一位用户询问了能够操作文件、图像并具有广泛代码生成能力以修改视频游戏的 AI 代码编辑器。
   - 讨论涉及了当前 LLM 的局限性，以及寻找可能克服代码生成中这些限制的工具。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.safe.ai/work/compute-cluster">Compute Cluster | CAIS</a>：AI 安全中心 (Center for AI Safety) 正在启动一项计划，为机器学习安全研究提供大规模计算资源。在此申请。</li><li><a href="https://arena3-chapter0-fundamentals.streamlit.app/#option-1-colab)">未找到标题</a>：未找到描述</li><li><a href="https://arena3-chapter0-fundamentals.streamlit.app/).">未找到标题</a>：未找到描述</li><li><a href="https://arena-uk.slack.com/archives/C058P58400K/p1686507915681109?thread_ts=1686121368.323589&cid=C058P58400K)">Slack</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1271211243369271307)** (18 条消息🔥): 

> - `Synchronization of Model Curricula` (模型训练课程同步)
> - `Benchmarking and Evaluation Practices` (基准测试与评估实践)
> - `Tree Attention Algorithm` (Tree Attention 算法)
> - `Zamba Model Performance` (Zamba 模型性能)
> - `UT-RNN Hybrid Implementations` (UT-RNN 混合实现)


- **模型训练课程同步**：一位成员询问了如何同步两个不同模型的训练课程，以确保使用完全相同的 minibatch。
   - 另一位成员建议记录训练数据的顺序和分组，同时也有人对设定种子（seeded）的 dataloader 的确定性行为表示担忧。
- **关于基准测试实践的辩论**：一位成员报告称，他们的 NeurIPS 论文收到的评审意见将基准测试和最佳实践斥为“非真正的科学”。
   - 另一位成员指出，理所当然地接受基准测试可能是不可取的，并承认在评估方法论方面可能存在误解。
- **用于高效计算的 Tree Attention**：讨论重点介绍了一篇论文，该论文推导出了一个用于计算 self-attention 的标量能量函数，从而产生了一种 Tree Attention 算法，通过并行计算优化性能。
   - 该实现有望提高在 GPU 集群上处理长上下文（long-context）attention 的效率，并分享了相关的 GitHub 仓库。
- **Zamba 模型性能惊人**：Zamba 模型团队因在训练 token 较少的情况下表现优于 LLaMA 2 7B 而受到关注，尽管其曝光度较低。
   - 该模型的数据集已公开，因其显著的效率和性能引起了广泛兴趣。
- **UT-RNN 混合模型的潜力**：一位用户对将 Zamba 模型微调为 UT-RNN 混合模型的可能性表示兴奋，因为其架构利用了共享的 attention 和 MLP 模块。
   - 这种设计为保留输入信息提供了额外的路径，暗示了未来模型开发的一个有前景的方向。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2408.04220">Diffusion Guided Language Modeling</a>：目前的语言模型在文本生成方面表现出卓越的能力。然而，对于许多应用来说，控制生成文本的属性（如情感或毒性）是很有必要的...</li><li><a href="https://arxiv.org/abs/2408.04093">Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters</a>：Self-attention 是现代 Transformer 架构的核心数学运算，由于其在序列长度上的平方复杂度，也是一个显著的计算瓶颈。在本文中...</li><li><a href="https://github.com/Zyphra/tree_attention">GitHub - Zyphra/tree_attention: Tree Attention: Topology-aware Decoding for Long-Context Attention on GPU clusters</a>：Tree Attention：针对 GPU 集群上长上下文 Attention 的拓扑感知解码 - Zyphra/tree_attention</li><li><a href="https://github.com/Zyphra/Zamba2">GitHub - Zyphra/Zamba2: PyTorch implementation of models from the Zamba2 series.</a>：Zamba2 系列模型的 PyTorch 实现。 - Zyphra/Zamba2
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1271443945372647546)** (7 条消息): 

> - `GemmaScope 论文`
> - `SAE 训练过程`
> - `模型学习 SO(3) 群操作`
> - `分解模型激活值`
> - `稀疏自编码器 (Sparse autoencoders)` 


- **关于 SAE 训练顺序的好奇**：一位成员询问，为什么对于 MLP，**SAE** 是在 **RMS norm** 之后训练的，而对于注意力层，则是在 **RMS** 之前训练。另一位成员回答说，这是为了将其置于注意力层读取头（readout head）的 **w_0** 之前。
   - 这突显了一个有意的设计选择，旨在优化模型处理注意力机制的方式。
- **注意力头预训练的重要性**：讨论表明，在 **w_0** 之前训练 **SAE** 使得 **GemmaScope** 论文中的某些高级机制成为可能，强调了这种方法的优势。
   - 这表明战略性的训练序列可以增强模型的性能和可解释性。
- **寻找关于 SO(3) 群操作的论文**：一位成员正在寻找一篇证明模型学习 **SO(3)** 群操作以表示旋转的论文，随后找到了相关链接，并对其发表时间之早感到惊讶。
   - 这场对话强调了基础研究在当代机器学习讨论中持续存在的相关性。
- **相关论文推荐**：另一位成员推荐了另一篇与对称性相关的论文，分享了一个他们非常喜欢的链接，这与寻找可解释模型的研究相呼应。
   - 这体现了社区的协作性质，成员们积极支持彼此的研究兴趣。



**提到的链接**：<a href="https://arxiv.org/abs/2406.17759">Interpreting Attention Layer Outputs with Sparse Autoencoders</a>：将模型激活值分解为可解释的组件是机械可解释性（mechanistic interpretability）中的一个关键开放问题。稀疏自编码器 (SAE) 是一种流行的分解内部激活的方法...

  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1271300413328719882)** (23 条消息🔥): 

> - `Karpathy 的 nanoGPT 评估`
> - `lm-evaluation-harness 的不一致性`
> - `评估中的浮点数差异`
> - `Neurips 基准轨道评审` 


- **Karpathy 的 nanoGPT 评估挑战**：几位成员讨论了使用 **lm-evaluation-harness** 评估 **Karpathy 的 nanoGPT 模型** checkpoint 时遇到的问题，并指出它没有以兼容 HF 的方式保存。
   - 一位用户表示在运行评估工具链时遇到困难，并向他人寻求帮助。
- **lm-evaluation-harness 结果的不一致性**：一位用户分享了评估结果中存在**不一致性**的发现，这似乎与 **batch size** 和使用的 GPU 数量有关。
   - 另一位成员建议使用不同的 few-shot 采样方法，以在评估中获得更好的**确定性**。
- **影响评估结果的浮点数问题**：讨论围绕着**浮点数**计算的微小变化如何可能导致评估分数的差异展开，尽管这些差异应该是极小的。
   - 一位成员指出，**对于大型数据集**，变动不应显著影响结果，预计只会在小数点后第三位产生微小变化。
- **对 Neurips 基准测试结果的兴奋**：一位用户在 Neurips 基准轨道评审中获得了 **6/6/5 的评分**和 **3/2/3 的置信度**。
   - 他们对 rebuttal 后的机会感到乐观，另一位成员也鼓励说这些分数非常有前景。



**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/new_task_guide.md#task-name--tags-registering-a-task">lm-evaluation-harness/docs/new_task_guide.md at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness

  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1271189123050246224)** (117 messages🔥🔥): 

> - `Low VRAM Mode`
> - `Face Swapping Tools`
> - `Stable Diffusion Performance`
> - `Custom Lora Commissions`
> - `Live Preview Settings in A1111` 


- **Low VRAM Mode 使用**: 当进入 Low VRAM Mode 时，如果生成任务能成功完成，可能不一定需要切换到更小的模型，但使用不同版本可能会节省时间。
   - *尝试不同的模型选项有助于优化性能。*
- **Face Swapping 工具对比**: 对于在搭载 Intel CPU 的 PC 上进行换脸，成员建议使用 [Rope](https://github.com/Hillobar/Rope)，因为它比 Roop 更容易安装。
   - 讨论强调了对换脸感兴趣的用户需要设置简单且有效的工具。
- **Stable Diffusion 性能因素**: 用户报告了采样速度 (s/it) 的差异，部分用户在更改模型大小时遇到了处理时间的跳变，引发了对性能一致性的担忧。
   - *还分享了关于不同设置和硬件配置（如 ROCm 和 WSL2）的见解，以评估预期性能。*
- **定制 Lora 模型委托**: 参与者讨论了寻找可靠的定制 pony lora 模型委托，建议使用 Civitai 的悬赏系统以确保交易安全。
   - *几位推荐的创作者拥有良好的委托信誉，强调了彻底审查的重要性。*
- **Live Preview 设置查询**: 一位用户询问了 A1111 中最佳的实时预览设置，质疑某些格式的用途以及预览帧是否保存在硬盘上。
   - *讨论表明社区对优化图像生成工作流以提高性能感兴趣。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://civitai.com/search/models?sortBy=models_v9&query=baldur">Civitai | Share your models</a>: 未找到描述</li><li><a href="https://www.shakker.ai/">Shakker AI - Premium Stable Diffusion Model Hub</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>: 使用命令 wsl --install 安装 Windows Subsystem for Linux。在 Windows 机器上使用由你首选的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。</li><li><a href="https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/wsl/install-radeon.html">Install Radeon software for WSL with ROCm &#8212; Use ROCm on Radeon GPUs</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=44waH3sDYOM">Tiled Diffusion with Tiled VAE / Multidiffusion Upscaler, the Ultimate Image Upscaling Guide [A1111]</a>: #aiart, #stablediffusiontutorial, #generativeart 本教程将介绍如何使用 Tiled ... 将低分辨率图像放大到 4k 及以上分辨率。</li><li><a href="https://github.com/Hillobar/Rope">GitHub - Hillobar/Rope: GUI-focused roop</a>: 专注于 GUI 的 roop。通过在 GitHub 上创建账户来为 Hillobar/Rope 的开发做出贡献。</li><li><a href="https://github.com/apple/ml-mdm">GitHub - apple/ml-mdm: Train high-quality text-to-image diffusion models in a data &amp; compute efficient manner</a>: 以数据和计算高效的方式训练高质量的文本到图像扩散模型 - apple/ml-mdm
</li>
</ul>

</div>
  

---



### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1271207192938287344)** (1 messages): 

> - `DALL·E 3 image generation`
> - `ChatGPT Free users` 


- **DALL·E 3 面向免费用户开放**: ChatGPT 免费用户现在每天可以使用 **DALL·E 3** 生成最多 **两张图片**。
   - 此更新使用户能够为各种应用创建图像，例如 *幻灯片* 或 *个性化卡片*。
- **图像创建简化**: 用户只需让 ChatGPT 根据需求创建图像，提升了用户便利性。
   - 此功能使可视化概念和增强个人交流变得更加容易。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1271195097198891009)** (86 messages🔥🔥): 

> - `Mistral NeMo Performance`
> - `GPT-4 vs GPT-4o`
> - `Open WebUI & Ollama Integration`
> - `Neuroadaptive Language Research`
> - `Local AI Model Run Recommendations` 


- **Mistral NeMo 性能探讨**：成员们对 **Mistral NeMo** 在不同硬件上的性能表现表示好奇，特别是配备 16GB RAM 的 M1 机器。
   - 一位参与者提到由于硬件限制，无法运行更大的模型。
- **对 GPT-4 能力的担忧**：一位用户分享了对 **GPT-4o** 的挫败感，认为它与 **GPT-4** 相比能力较弱，尤其是在图像分析等特定任务中。
   - *GPT-4o* 因回答生硬而受到批评，有人将其比作失去了基础理解能力的程序员。
- **本地 AI 模型工作流**：一位参与者讨论了向 **Open WebUI** 和 **Ollama** 的迁移，并考虑停止订阅 ChatGPT+ 以转向本地模型。
   - 他们提到运行这些模型（尤其是 *LLama*）非常可靠，但也承认使用自托管方案存在挑战。
- **对神经自适应语言研究的兴趣**：一位成员正在开发一个用于在自闭症视角与神经典型（neurotypical）视角之间进行翻译的模型，并寻求相关领域研究人员的合作。
   - 另一位参与者表达了兴趣，并表示基于其作为自闭症患者和研究人员的双重个人经历，双方存在联系。
- **本地 AI 性能建议**：讨论了运行 AI 模型的各种 GPU 性能，有人建议使用 **RTX 2060** 以获得更高效的处理能力。
   - 参与者对通过 Discord 路由运行模型时的响应延迟表示担忧，并讨论了模型量化（quantization）的重要性。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1271230573951582229)** (4 messages): 

> - `LangChain with CSV`
> - `ChatGPT issues on Safari` 


- **LangChain 与 CSV 集成资源**：一位用户在寻求资源，希望在 **LangChain** 中结合 **OpenAI** 将 **CSV 文件** 作为 *检索增强生成* (RAG) 文档使用。
   - 这表明用户对使用语言模型处理结构化数据的兴趣日益增加。
- **ChatGPT 在 Safari 上无法运行**：一位用户表达了对过去两天在 iOS 的 **Safari** 浏览器上使用 **ChatGPT** 时反复出现错误消息的挫败感，称其非常令人烦恼。
   - 另一位用户建议他们考虑使用 **ChatGPT app** 而不是网页版，以避免这些问题。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1271360918655205387)** (4 messages): 

> - `Chat Prompt Library`
> - `Becoming a Prompt Engineer`
> - `Learning Resources for Prompt Engineering` 


- **Chat Prompt Library 的位置**：一位成员询问 **chat prompt library** 的位置。
   - 另一位成员指出它已经**更名**，并指向了一个特定的频道。
- **对准 Prompt 工程师的建议**：一位希腊本科生表达了成为 **prompt engineer** 的兴趣，并寻求入门指导。
   - 一位资深成员建议探索 **Arxiv** 和 **Hugging Face**，并强调了深入研究 **meta-prompting** 的重要性。
- **用于学习的 Discord 社区**：一位资深成员提到，有很多专门为对 prompt engineering 感兴趣的人开设的 **Discord** 社区。
   - 他们鼓励在特定网站之外，也去这些社区寻找进一步的学习资源。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1271360918655205387)** (4 messages): 

> - `Chat prompt library`
> - `Becoming a prompt engineer`
> - `Learning resources for prompt engineering` 


- **Chat prompt library 更名信息**：一位成员询问 **chat prompt-library** 的位置。
   - 另一位成员回复称其已**更名**，并提供了新位置的链接。
- **成为 prompt engineer 的路径**：一位来自希腊的用户表达了成为 **prompt engineer** 的兴趣，尽管该职业在当地的存在感有限。
   - 作为回应，另一位成员建议从 **Arxiv** 和 **Hugging Face** 开始，并加入各种 Discord 社区以获取支持。
- **Meta-prompting 作为一种强大的策略**：一位成员强调了将 **meta-prompting** 作为 prompt engineering 关键策略的重要性。
   - 他们强调该领域有很多东西需要学习，并鼓励通过提供的资源进行进一步探索。


  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1271218232656920648)** (95 messages🔥🔥): 

> - `Gemini 1.5 Flash 性能`
> - `GPT-4o Mini 对比 Gemini 1.5`
> - `OpenRouter API 配置`
> - `讨论中的达克效应 (Dunning-Kruger Effect)`
> - `日语模型推荐` 


- **Gemini 1.5 Flash 性能讨论**：多位用户注意到 **Gemini 1.5 Flash** 经历了大幅价格下调，使其在快速且廉价的模型市场中更具竞争力。
   - 更新后的模型现在可以原生理解 PDF，并提升了文本和多模态查询的性能。
- **GPT-4o Mini 与 Gemini 1.5 的对比**：**GPT-4o Mini** 因其比 **Gemini 1.5** 更低的幻觉率而受到称赞，特别是在编程任务中。
   - 用户表示更倾向于那些能减少幻觉并优化编程能力的模型。
- **OpenRouter API 配置挑战**：一位开发者在 TypeScript 中使用 **OpenAI SDK** 时，在传递自定义参数（特别是 `providers` 配置）方面遇到了问题。
   - 有人提到 API 目前原生不支持这些参数，从而导致了 Lint 错误。
- **讨论中强调的达克效应 (Dunning-Kruger Effect)**：一场幽默的辩论出现了，参与者使用 **达克效应 (Dunning-Kruger Effect)** 来阐明关于专业知识讨论中自我评估的观点。
   - 随着用户对自信心与能力（特别是在赚钱的背景下）的反思，对话变得富有喜剧色彩。
- **寻求日语 LLM 推荐**：一位用户询问在日语方面表现优于 **GPT-4o Mini** 的 LLM 模型，寻求在类似价格范围内的替代方案。
   - 持续的搜索反映了对在特定语言上表现出色（超越大型模型通用能力）的模型的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://aider.chat`">未找到标题</a>: 未找到描述</li><li><a href="https://aider.chat`.">未找到标题</a>: 未找到描述</li><li><a href="https://simonwillison.net/2024/Aug/8/gemini-15-flash-price-drop/">Gemini 1.5 Flash 降价</a>: Google Gemini 1.5 Flash 已经是性价比最高的模型之一，价格为 35c/百万 input tokens。今天他们将其降至仅 7.5c/百万（以及 30c/百万），针对 128,000 tokens 以下的 prompts。该……</li><li><a href="https://aitestkitchen.withgoogle.com/tools/image-fx">ImageFX</a>: 未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/document-understanding">未找到标题</a>: 未找到描述</li><li><a href="https://openrouter.ai/models?modality=text%2Bimage-%3Etext">Models | OpenRouter</a>: 在 OpenRouter 上浏览模型</li><li><a href="https://en.wikipedia.org/wiki/Dunning–Kruger_effect">Dunning–Kruger effect - Wikipedia</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Dunning%E2%80%93Kruger_effect">Dunning–Kruger effect - Wikipedia</a>: 未找到描述</li><li><a href="https://tenor.com/view/kenan-thompson-kenan-thompson-snl-mmhmm-gif-704775788876592410">Kenan Thompson Snl GIF - Kenan Thompson Kenan Thompson - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 在家运行 LLMs，BitTorrent 风格。微调和推理速度比 offloading 快达 10 倍</a>: 🌸 在家运行 LLMs，BitTorrent 风格。微调和推理速度比 offloading 快达 10 倍 - bigscience-workshop/petals</li><li><a href="https://openrouter.ai/models/01-ai/yi-vision">Yi Vision - API, 提供商, 统计数据</a>: Yi Vision 是一款复杂的视觉任务模型，提供基于多图的高性能理解和分析能力。它非常适合需要分析和解释的场景……</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b">FireLLaVA 13B - API, 提供商, 统计数据</a>: 一款极速的视觉语言模型，FireLLaVA 能快速理解文本和图像。它在测试中表现出令人印象深刻的对话能力，旨在模拟多模态 GPT-4。使用 FireLLaVA 13B 运行……</li><li><a href="https://openrouter.ai/settings/preferences">设置 | OpenRouter</a>: 管理您的账户和偏好设置</li><li><a href="https://openrouter.ai/docs/provider-routing#ignore-providers-for-a-request">提供商路由 | OpenRouter</a>: 跨多个提供商路由请求</li><li><a href="https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-api/">Gemini 1.5 Flash 降价且微调功能完成部署，以及更多</a>: 未找到描述</li><li><a href="https://github.com/mlc-ai/web-llm?tab=readme-ov-file">GitHub - mlc-ai/web-llm: 高性能浏览器内 LLM 推理引擎</a>: 高性能浏览器内 LLM 推理引擎。通过在 GitHub 上创建账户为 mlc-ai/web-llm 的开发做出贡献。
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1271221815028613132)** (8 条消息🔥): 

> - `欢迎消息`
> - `新的 sus-column-r 模型`
> - `与 GPT-4 和 Claude 3.5 的对比` 


- **热烈欢迎新成员**: 多位成员欢迎新用户加入服务器，为新人营造了友好的氛围。
   - 欢迎语包括简单的“Hello!”和“Welcome to Cohere!”，以此邀请大家参与互动。
- **关于新 sus-column-r 模型的讨论**: 一位用户分享了一个 [Reddit 链接](https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/)，讨论 LMSYS 上出现的一个名为 'sus-column-r' 的新模型，声称它在各种任务中优于 GPT-4 和 Claude 3.5。
   - *我不明白这怎么可能，* 该用户表示，并引用了在**翻译**、**编程**和**数学**方面提升的性能。



**提到的链接**: <a href="https://www.reddit.com/r/LocalLLaMA/comments/1enmcr9/new_suscolumnr_model_on_lmsys_its_just_f_up/">Reddit - 深入探索</a>: 未找到描述

  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1271222019110993920)** (22 messages🔥): 

> - `使用 Preamble ID`
> - `RAG 中的响应质量`
> - `Cohere Embedding 模型`
> - `限制输出 Token`
> - `结构化 JSON 输出` 


- **使用 Preamble ID 处理上下文**：一位用户寻求帮助，希望利用 preamble ID 在不同输入中通用化 prompt，而无需每次重新生成新的 preamble，并询问维持上下文的方法。
   - 另一位成员建议使用 **conversation ID**，这解决了用户的问题。
- **RAG 响应质量的挑战**：一位社区成员报告了在检索增强生成 (RAG) 设置中响应生成的问题，注意到当信息不足时，幻觉（hallucinations）会增加。
   - 尽管遵循了指南，AI 仍未遵循事实信息，从而引发了修改 prompt 的建议。
- **对 Cohere Embedding 模型的担忧**：一位用户讨论了从 `embed-english-light-v2.0` 切换到 `embed-english-light-v3.0` 模型的经验，并观察到尽管预期会有所改进，但检索质量却有所下降。
   - 他们提供了有关其数据集和使用情况的详细信息，表明较新的模型表现不佳。
- **限制 Command-R 模型的输出 Token**：一位用户询问如何限制 **command-r** 模型的输出 token，提到 `max_tokens` 参数似乎没有生效。
   - 在分享了他们的 API 调用后，需要确认输出是否超过了指定的 token 限制。
- **结构化 JSON 输出讨论**：一位成员讨论了尝试实现结构化 JSON 输出，表示其 prompt 设置为需要二元输出的分类任务。
   - 他们寻求澄清其现有输出是否超出了预期的 `Yes/No` 格式。



**提及链接**：<a href="https://docs.cohere.com/reference/chat">Chat - Cohere API 参考</a>：生成对用户消息的文本响应。要了解如何将 Chat API 与 Streaming 和 RAG 结合使用，请参阅我们的文本生成指南。

  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1271349639609581568)** (23 messages🔥): 

> - `403 Forbidden 错误`
> - `VPS 连接问题`
> - `Langchain 多步工具调用错误` 


- **403 Forbidden 错误排查**：成员们讨论了在使用 curl 发送请求时遇到 **403 Forbidden** 错误的问题，建议指出这通常表示 **API key 无效**或地理位置限制。
   - 一位成员确认其 VPS 位置在美国，而其他人提到同样的请求在本地机器上可以运行，导致对原因感到困惑。
- **关于使用 VPS 进行 API 请求的讨论**：有提到 **403 错误**也可能源于 VPS 的 IP 地址位置可能受限，从而要求提供更多关于用户位置的信息。
   - 尽管进行了讨论，成员们仍无法解决该错误，并指出之前的 API 请求可以正常返回流式响应。
- **Langchain 多步工具调用的错误**：一位成员报告了在使用 **Langchain** 的 **multistep_tool_use** 功能时出现错误，具体收到的消息为：`ERROR:root:Failed to parse multihop completion for input`。
   - 他们寻求他人的帮助以获取潜在的修复方法，或参考有关 **Cohere** 和 **Langchain** 集成的更好文档。



**提及链接**：<a href="https://docs.cohere.com/docs/implementing-a-multi-step-agent-with-langchain">使用 Langchain 实现多步 Agent</a>：在本文档中，我们将详细介绍如何使用 Cohere 的多步工具调用功能和 Langchain 框架构建生成式 AI Agent。构建 Langchain ReAct Agent ...

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1271457209863372875)** (3 messages): 

> - `Docker 安装问题`
> - `后端设置问题` 


- **Docker 安装引发疑问**：一位用户对使用 **Docker** 安装后界面无法操作表示困惑，询问是否遗漏了什么。
   - 这促使 **Nick Frosst** 建议问题可能出在后端设置上，尽管他承认对具体细节不确定。
- **指出可能存在后端配置错误**：Nick Frosst 回应了用户的查询，暗示 **backend** 设置可能存在配置错误。
   - 他承认自己对确切问题并不确定，表明关于解决该问题的讨论仍在进行中。

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1271257088106168465)** (4 条消息): 

> - `Event-Driven Agent Systems` (事件驱动的 Agent 系统)
> - `Mixture-of-Agents`
> - `Property Graphs` (属性图)
> - `Multimodal RAG Pipelines` (多模态 RAG 流水线)


- **事件驱动 Agent 系统的灵活性**：以**事件驱动的方式**构建 Agent 为用户提供了更大的灵活性，以创建具有复杂通信模式的**循环、多 Agent 系统**。查看这个[精彩的教程视频](https://t.co/he0Az19WJS)，其中对比了基于图的 Agent 编程。
   - _“这是一个非常棒的教程视频”_ 展示了在 Agent 系统中使用事件驱动方法的优势。
- **Mixture-of-Agents 表现优于大型模型**：[Junlin Wang](https://t.co/C8pZBBIcOk) 的一篇新论文介绍了如何集成较小的 LLM 以构建 **Mixture-of-Agents** 系统，其表现可以超越最先进的大型模型。这已在一个全异步的**事件驱动工作流**中实现。
   - 对于那些对实际应用感兴趣的人，[Twitter](https://t.co/hNpoZuBC5l) 上详细讨论了该实现。
- **理解用于 GraphRAG 的 Property Graphs**：关于 LlamaIndex **Property Graphs** 的一个重要[视频教程](https://t.co/CdWqPOxt4c)解释了每个节点和关系如何存储结构化的属性字典。在深入研究 **GraphRAG** 之前，这些基础知识至关重要。
   - _“这种底层抽象解锁了许多酷炫的技术”_ 强调了 Property Graphs 的功能。
- **构建多模态 RAG 流水线**：现在有一些令人兴奋的 Notebook，解释了如何针对复杂的**法律、保险和产品文档**构建实用的、现实世界的 **Multimodal RAG** 流水线。该系列从解析**保险索赔**开始。
   - 点击此[链接](https://t.co/1w4IfXQ7CP)获取详细分解和实际用例。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1271188452834279484)** (45 条消息🔥): 

> - `Embedding Models and Document Retrieval` (嵌入模型与文档检索)
> - `Using Llama-Index with Multi Models` (在 Llama-Index 中使用多模态模型)
> - `Configuring Filters in Query Engines` (在查询引擎中配置过滤器)
> - `Ingesting Language Documents` (摄取多语言文档)
> - `RAG Pipeline Workflows` (RAG 流水线工作流)


- **在 Llama 中选择嵌入模型**：一位成员讨论了在 Llama 中使用 [HuggingFaceEmbedding](https://github.com/openai/tiktoken) 模型，并展示了一个在进行查询调用之前加载特定模型以对文档进行嵌入的示例。
   - 另一位用户询问了在完成嵌入后的文档加载和检索，明确了实现预期结果所需的顺序过程。
- **使用多模态模型进行图像查询**：一位用户表达了对在 Llama 中利用多模态方法查询图像的兴趣，但在为 OpenAIMultiModal LLM 配置代理时遇到了问题。
   - 在尝试使用 httpx 设置 http_client 时，他们遇到了同步与异步客户端需求方面的挑战。
- **在查询引擎中过滤文档**：一位用户分享了关于文档摄取过程中节点过滤的经验，旨在根据 `business_id` 元数据检索特定文档。
   - 另一位成员建议在检索期间实现 `MetadataFilters`，并强调过滤理想情况下应在检索之前进行才能生效。
- **摄取德语文档**：一位成员在将德语文档摄取到向量数据库时遇到困难，发现尽管在代码中指定了德语，但返回的摘要却是英语。
   - 他们收到建议，更新摘要过程的 Prompt，以确保输出保持文档的原始语言。
- **RAG 流水线工作流实现**：关于新工作流架构增强 COA (Chain of Abstraction) 方法的讨论出现了，重点是分步执行以进行迭代优化。
   - 成员们一致认为创造性地利用 Agent 和工作流具有潜力，但也承认需要适当的文档或示例来进行实现。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1821626025584828747?s=46&t=lqHyE7PE7Ct6jUJNcJPPxg">来自 anton (@abacaj) 的推文</a>: 42 页的 PDF（显示为 30k tokens），包含图像和文本，投入 gemini-flash 1.5，每个答案都正确……一切都结束了 (it's so over)</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/workflow/rag/">带有重排序的 RAG 工作流 - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/openai/tiktoken">GitHub - openai/tiktoken: tiktoken 是一个用于 OpenAI 模型的快速 BPE 分词器。</a>: tiktoken 是一个用于 OpenAI 模型的快速 BPE 分词器。 - openai/tiktoken</li><li><a href="https://github.com/IntelLabs/RAGFoundry">GitHub - IntelLabs/RAGFoundry: 使用微调为检索增强生成任务定制 LLM 的框架。</a>: 使用微调为检索增强生成任务定制 LLM 的框架。 - IntelLabs/RAGFoundry</li><li><a href="https://github.com/run-llama/llama_index/blob/8a48fdc6f1086847ea7c0c9ef0df955b347ec366/llama-index-core/llama_index/core/extractors/metadata_extractors.py#L348">llama_index/llama-index-core/llama_index/core/extractors/metadata_extractors.py</a>: LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1271209109869559860)** (21 messages🔥): 

> - `Hackathon 公告`
> - `Open Interpreter 功能`
> - `MiniCPM-V 模型`
> - `Terminal Agent 环境`
> - `Linux 支持请求` 


- **令人兴奋的 Hackathon 公告！**：Open Interpreter 正在参加一场**大型**黑客松——“Breaking Barriers: A generative AI Hackathon for Digital Inclusion”，该活动将于 9 月 20 日至 23 日在德克萨斯州达拉斯举行，奖金总额达 **$17,500**。
   - 官方更倾向于线下参与，但也鼓励申请远程参加；社区内正在讨论组队事宜。
- **MiniCPM-V 模型超越同类**：一名成员强调，开源视觉多图模型 **MiniCPM-V 2.6** 据称在性能上超越了 **Gemini 1.5 Pro** 和 **GPT-4V**。
   - 分享了 [Hugging Face 模型](https://huggingface.co/openbmb/MiniCPM-V-2_6) 和 [GitHub 仓库](https://github.com/OpenBMB/MiniCPM-V) 的链接以供进一步探索。
- **对 Ollama 性能的担忧**：成员们对 **Ollama** 在 Windows 上通过 `interpreter --codestral` 或 LM Studio 运行时的缓慢表现感到沮丧。
   - 正在寻求替代方案或变通方法以提升用户体验。
- **请求 Linux 支持频道**：一位社区成员请求创建一个专门的 **#linux-something_or_other** 频道，以便集中讨论 Linux 相关话题。
   - 该建议得到了积极回应，并指向了一个现有的故障排除频道。
- **Terminal Agent 环境功能**：一名成员展示了 Terminal Agent 环境的能力，演示了截图功能，并强调了在灰度增强设置下的光标可见性。
   - 分享了相关的 GitHub issues 链接和截图，以说明所述功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/monkey-monkey-eating-monkey-eating-strawberries-kardie-gif-gif-22488578">Monkey Monkey Eating GIF - Monkey Monkey Eating Monkey Eating Strawberries - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://ollama.com/aiden_lu/minicpm-v2.6">aiden_lu/minicpm-v2.6</a>：MiniCPM-V 2.6 是 MiniCPM-V 系列中最新且最强大的模型。相比 MiniCPM-Llama3-V 2.5，它在性能上有显著提升。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1390">Add real terminal support · Issue #1390 · OpenInterpreter/open-interpreter</a>：你的功能请求是否与问题相关？请描述。OpenInterpreter 目前无法以异步方式与常见的 REPL 和 Shell 环境交互。它总是阻塞的。...</li><li><a href="https://huggingface.co/openbmb/MiniCPM-V-2_6">openbmb/MiniCPM-V-2_6 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/OpenBMB/MiniCPM-V">GitHub - OpenBMB/MiniCPM-V: MiniCPM-V 2.6: A GPT-4V Level MLLM for Single Image, Multi Image and Video on Your Phone</a>：MiniCPM-V 2.6：手机端支持单图、多图和视频的 GPT-4V 级 MLLM - OpenBMB/MiniCPM-V
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1271413670571933696)** (1 messages): 

> - `ESP32S3`
> - `O1 集成` 


- **用户寻求在 ESP32S3 上运行 O1 的帮助**：一名成员表示有兴趣在 **ESP32S3** 上运行 **O1**，并询问是否有人尝试过这种配置。
   - 他们请求社区协助在自己的设备上实现这一目标。
- **社区征集 ESP32S3 经验**：呼吁有 **ESP32S3** 经验的成员分享关于运行 **O1** 的见解。
   - 用户希望利用集体知识来获得更好的实现策略。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 messages): 

8i8__papillon__8i8d1tyr: https://www.youtube.com/watch?v=V5kAmFRwuxc

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1271209872326791272)** (18 条消息🔥): 

> - `LangChain API 差异`
> - `Anthropic Claude 3.5 停机`
> - `Discord 与产品公告之间的脱节`
> - `LangChain 支持与文档问题`
> - `LangChain 的社区支持` 


- **LangChain 在 LLM 功能一致性方面面临挑战**：一位成员对 **LangChain** 为所有 LLM 提供统一 API 的能力表示困惑，指出它对 **OpenAI** 有效，但对 **Anthropic** 无效。
   - 另一位成员确认，虽然 function calls 可能相似，但由于不同 LLM 之间的差异，prompt 修改是必要的。
- **Anthropic 的 Claude 3.5 经历严重停机**：一位成员报告称 **Anthropic 的 Claude 3.5** 整天处于宕机状态，并引用了错误代码为 **500** 的内部服务器错误。
   - 他们分享了错误消息，指出 API 的问题阻碍了功能运行。
- **Discord 讨论与官方产品公告**：有成员对 **Discord 讨论** 与 LinkedIn 上关于 **LangGraph Cloud** 和 **Studio** 的官方产品公告之间的脱节表示担忧。
   - 一位成员质疑了跨平台共享信息的清晰度和一致性。
- **对 LangChain 工具和文档的挫败感**：在再次尝试 **LangChain** 后，一位成员报告了不一致的 tool/function calling，并指出示例和文档不足是主要障碍。
   - 他们询问是否有可用的商业支持，以便为该平台提供更好的协助。
- **LangChain 的社区支持正在减少**：一位成员对 **LangChain** 社区支持的下降表示哀叹，并引用了 Hackernews 上的看法，即尽管最初很有前景，但它已经失去了势头。
   - 他们表达了合作的愿望，提到了个人的困境，并寻求关于 LangChain 使用方面的付费协助。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1271211436701515898)** (4 条消息): 

> - `CTF 挑战`
> - `Mood2Music 仪表板`
> - `CRAB 基准测试` 


- **加入 1000 美元的 CTF 挑战！**：鼓励参与者参加 **夺旗赛 (CTF) 挑战**，目标是从一个 AI Agent 中提取密码，奖金为 **1000 美元**。
   - 该挑战探索了通过用户反馈表单意外泄露秘密的风险，引发了关于数据隐私和安全的问题。
- **Mood2Music 仪表板发布**：**Mood2Music** 仪表板的一个令人兴奋的预览展示了基于用户情绪连接到 **Spotify** 和 **Apple Music** 的 AI 驱动歌曲推荐。
   - 该工具通过创建针对用户情感定制的播放列表来增强听歌体验，旨在缓解音乐选择中的**决策疲劳**。
- **介绍 CRAB：多模态 Agent 基准测试**：新的 **CRAB** 框架允许在 **Android** 和 **Ubuntu** 等多个环境中构建和评估多模态语言模型 Agent。
   - 它具有**细粒度**的评估指标、任务生成能力，并旨在增强类人任务的执行，相关资源可在 GitHub 和项目[网站](https://crab.camel-ai.org/)上获得。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/camelaiorg/status/1821970132606058943?s=46">来自 CAMEL-AI.org (@CamelAIOrg) 的推文</a>：介绍 🦀 CRAB：用于多模态语言模型 Agent 的跨环境 Agent 基准测试 🦀 CRAB 提供了一个端到端且易于使用的框架来构建多模态 Agent、操作环境...</li><li><a href="https://invariantlabs.ai/ctf-challenge-24">欺骗 Agent 以提取秘密密码</a>：宣布首个针对 Agent 系统安全的 Invariant 夺旗赛 (CTF) 挑战。</li><li><a href="https://mood2music.me">mood2music</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1271209576104198144)** (18 条消息🔥): 

> - `图像数据集对比`
> - `使用 Gemma 进行模型引导`
> - `字幕与可靠性`
> - `LAION 数据库讨论` 


- **图像数据集：CC vs LAION**：一场关于 [Fondant 25M 数据集](https://huggingface.co/datasets/fondant-ai/fondant-cc-25m) 是否是最大的知识共享 (Creative Commons)/公有领域图像集合的讨论展开了。
   - 有人担心 **LAION-5B** 数据集的可靠性较低，因为它依赖于通常与图像无关的 alt 文本。
- **引导像 Gemma 这样的模型**：一位成员询问是否有人尝试过使用 **Gemma Scope** 来引导像 **Gemma 2 2B** 这样的模型，以及如何制作有效的控制向量 (control vectors) 来生成输出。
   - 他们表示需要超越简单的 Google 查询的见解，以增强对模型特征的理解。
- **字幕的信任问题**：参与者讨论了大规模抓取字幕的不可靠性，强调这种规模的所有字幕都可能缺乏准确性。
   - 对话还涉及了使用 CLIP 相似度分数是否能帮助确定新字幕是否比**原始字幕更不可靠**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2211.15006">Fine-tuning language models to find agreement among humans with diverse preferences</a>：大型语言模型 (LLMs) 的近期工作已使用微调来使输出与典型用户的偏好保持一致。这项工作假设人类偏好是静态且同质的...</li><li><a href="https://huggingface.co/datasets/fondant-ai/fondant-cc-25m">fondant-ai/fondant-cc-25m · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 条消息): 

nodja: https://research.google/blog/halva-hallucination-attenuated-language-and-vision-assistant/
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1271190643984568320)** (2 条消息): 

> - `红杉资本 (Sequoia Capital) 融资`
> - `AI 推理初创公司`
> - `AI 中的思维链 (Chain of Thought)` 


- **红杉资本关注 AI 推理初创公司的融资**：据 [The Information](https://www.theinformation.com/articles/sequoia-capital-has-discussed-funding-ai-reasoning-startup-cofounded-by-robinhood-ceo) 报道，红杉资本已讨论为一家由 **Robinhood CEO** 共同创立的 AI 推理初创公司提供资金。
   - 该初创公司旨在增强 AI 在**推理**和**决策**方面的能力。
- **Ross 对思维链 (Chain of Thought) 的见解**：在与 Ross 的访谈中强调，**思维链 (Chain of Thought)** 技术将上下文维持在 token 中，而不是潜空间 (latent space) 中，这为 AI 处理提供了一个新视角。
   - *“我从未那样思考过”* 强调了理解 AI 机制内部上下文的重要性。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1271461440175341601)** (12 条消息🔥): 

> - `Anaconda 软件许可`
> - `pip 的替代方案` 


- **Anaconda 对研究机构强制执行商业许可**：研究和学术机构发现，他们现在必须为 **Anaconda** 制作的软件付费（此前认为这些软件是免费的），因为该公司正在追究违反其服务条款的行为。
   - 一位消息人士报告称收到了商业许可的法律要求，并指出他们的机构被警告可能会因未经授权的使用而产生“补缴账单”。
- **关于容器许可要求的澄清**：用户正在询问在 **Docker 容器**上使用 Anaconda 是否需要额外的许可，并暗示很可能需要。
   - 成员们指出，有关许可问题的引用并非直接来自 Anaconda，而是来自受影响的机构。
- **切换到 uv 以实现更快的安装**：一位成员建议考虑使用 **uv** 作为 **pip** 的安装替代方案，并强调其速度明显更快。
   - 他们提到使用 uv 不需要额外的工具链，用户只需在安装时将 `pip` 替换为 `uv pip` 即可。



**提到的链接**：<a href="https://www.theregister.com/2024/08/08/anaconda_puts_the_squeeze_on">Anaconda 向数据科学家施压</a>：学术、非营利组织被告知开始付费——否则后果自负

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1271629008773316712)** (1 条消息): 

> - `Bad Takes` (糟糕的观点)
> - `Improvement in Discourse` (讨论质量的提升)


- **没有糟糕观点的世界会更好**：*如果每个发表**糟糕观点 (bad takes)** 的人仅仅只发表**糟糕观点**，世界会变得好得多，哈哈。* 这反映了对负面观点在讨论中所产生影响的幽默看法。
   - 该陈述暗示，如果只有那些持有拙劣见解的人参与讨论，对话的质量可能会显著提高。
- **讨论中的幽默**：这句话展示了利用**幽默**来批评当前的讨论趋势，强调了对更具建设性意见的偏好。
   - 它概括了许多人共同的心声，即希望在社区对话中看到更多积极的参与。


  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 条消息): 

chygao: https://youtu.be/6QWuJRvMtxg?si=SYXsRvYbfcdtYLC2
  

---



### **DSPy ▷ #[show-and-tell](https://discord.com/channels/1161519468141355160/1202371242519441499/1271450524369162280)** (3 条消息): 

> - `DSPy Tutorial`
> - `OpenAI Structured Output API` 


- **DSPy 教程演示**：一位成员分享了一个关于 DSPy 的 [YouTube 教程](https://youtu.be/_ROckQHGHsU)，通过 **8 个示例**涵盖了从基础到高级 LLM 项目的主要概念。
   - 该教程旨在帮助观众以结构化的形式掌握 DSPy 的复杂性。
- **实验新的 OpenAI API**：另一位成员在语音休息室宣布，他们正在实验 OpenAI 新发布的 **Structured Output API**。
   - 该 API 旨在增强用户在项目中交互和利用结构化数据输出的方式。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1271296628585660437)** (8 条消息🔥): 

> - `DSPy Prompt Improvement`
> - `Tutorial on DSPy Concepts`
> - `DSPy Use Cases`
> - `Signature Adapters`
> - `RAG Optimization` 


- **使用自定义 GPT 改进 DSPy Prompt**：一位成员正在寻求改进一个交织了指令和示例的复杂 Prompt 的建议，并提到了可能使用 Signature Adapters 和 MIPRO 优化。
   - 另一位成员建议先参考 [自定义 GPT 指南](https://chatgpt.com/g/g-cH94JC5NP-dspy-guide-v2024-2-7) 来实现 Prompt 的模块化。
- **YouTube 上的 DSPy 教程**：一位成员在频道上分享了一个教程，通过八个示例解释了 DSPy 的主要概念，从基础项目进阶到高级 LLM 项目，可在[此处](https://youtu.be/_ROckQHGHsU)观看。
   - 另一位成员通过订阅该频道表示支持。
- **了解 DSPy 在 RAG 中的用例**：一位成员询问了 DSPy 的用例及其在 RAG 任务中的适用性。
   - 作为回应，另一位成员澄清说，它类似于 Fine-tuning，通过优化任务、指标和示例来获得更好的 LLM 性能。
- **探索 Signature Adapters**：成员们讨论了在为 DSPy Prompt 定制指令时使用 Signature Adapters 的潜力。
   - 分享了一个相关的 [Signature GPT 资源](https://chatgpt.com/g/g-JQgwRHI0D-dspy-signature-gpt-v2024-2-21) 链接以供进一步探索。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1271235461368778772)** (8 条消息🔥): 

> - `Poe Hackathon`
> - `Alliance AI-Health Research Initiative` 


- **Poe 宣布创新黑客松**：Poe (@poe_platform) 正在举办一场为期一天的 [Hackathon](https://x.com/poe_platform/status/1820843642782966103)，重点是使用 **GPT-4o** 和 **Gemini 1.5 Pro** 等先进 LLM 开发生成式 UI 体验。
   - 线下部分将在 **Hillsborough, CA 94010** 举行，具体细节仅对注册参与者开放。
- **AI 与健康领域的实习机会**：**Alliance AI-Health Research Initiative** 正在招募学生参加为期 **4 个月的远程实习**，针对癌症检测和基于 AI 的中暑检测等项目开展先锋研究。
   - 有兴趣的申请者可以在 **8 月 11 日**前[在此申请](https://tinyurl.com/applyalliance)，并有机会在学术期刊上发表研究结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/poe_platform/status/1820843642782966103">Poe (@poe_platform) 的推文</a>: 我们很高兴宣布与 @agihouse_org 围绕我们新的 Previews 功能举办为期一天的黑客松！竞争使用最新的 LL... 创造最具创新性和实用性的聊天内生成式 UI 体验。</li><li><a href="https://tinyurl.com/applyalliance">未找到标题</a>: 未找到描述</li><li><a href="https://x.co">出售域名 | 购买域名 | 停放域名</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MLOps @Chipro ▷ #[general-ml](https://discord.com/channels/814557108065534033/828325357102432327/1271668086008840304)** (1 messages): 

> - `Feature Stores in Computer Vision` 


- **评估 Computer Vision 的 Feature Stores**：一位成员询问了在 **computer vision** 中使用 **feature stores** 的情况，寻求关于其价值和有效性的见解。
   - 讨论开启了集成 feature stores 以管理和优化 computer vision 项目的潜在收益和注意事项。
- **对实际实现的兴趣**：有人呼吁提供 **feature stores** 在 computer vision 框架中的**实际实现**案例，以评估其影响。
   - 这突显了对真实世界案例研究的需求，以验证 feature stores 在特定应用中的有效性。


  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1271226593930510437)** (8 messages🔥): 

> - `Modular Licensing`
> - `Future of Modular's AI Applications`
> - `Triton Language`
> - `Custom Kernels` 


- **Modular 许可协议的宽松程度受到审查**：一位成员指出，Modular 使用 **max/mojo** 的许可协议是宽松的，除非你试图将 AI 基础设施平台商业化。
   - 成员们正在质疑，如果 Modular 扩展到其他领域（如 **robotics** 或 **AI 标注平台**），会发生什么。
- **非竞争性软件可能变为竞争性软件**：讨论显示，如果某软件目前不具竞争性，但在**未来变得具竞争性**，根据 Modular 的许可协议，它仍被视为非竞争性的。
   - 然而，关于此类软件一旦转为竞争性后，其开发是否必须“冻结”的问题随之而来。
- **征集 Triton Lang 自定义 Kernel 用户**：请求编写过自定义 kernel 的 **Triton lang** 用户与产品团队进行一对一交流。
   - 奖励包括为他们的贡献提供一些 **Mojo swag**（周边）。
- **对 Triton 语言的初步认识**：一位成员表达了好奇，并提到这是他**第一次听说 Triton**。
   - 这表明社区内对于扩展新语言和技术知识具有潜在兴趣。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1271192271496155290)** (5 messages): 

> - `Google Gemini Price Cuts`
> - `Comparison of Gemini and GPT-4o`
> - `Gemini 1.5 Free Finetuning` 


- **Google Gemini 价格大幅下调**：[标题为“Google Gemini Insane Price Cuts!!!”的 YouTube 视频](https://www.youtube.com/watch?v=3ICC4ftZP8Y) 强调了 **Google Gemini 1.5 Flash** 的显著降价。
   - 有关这些变化的详细信息也在 [Google Blog](https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...) 中分享。
- **关于 Gemini 与 GPT-4o 对比的困惑**：存在关于 **Gemini 1.5 Flash** 与 **GPT-4o** 对比的讨论，争论是否应该改为 **Gemini 1.5 Pro** 对阵 **GPT-4o**。
   - 一位成员质疑正确的对比是否应该区分标准版和 mini 版。
- **Gemini 1.5 免费 Finetuning 的影响**：一位参与者认为这种不寻常的对比是因为 **Gemini 1.5 具有免费 Finetuning** 功能，而 Pro 版本则没有。
   - 这一区别似乎正在影响关于 Gemini 模型能力和产品的持续讨论。



**提到的链接**：<a href="https://www.youtube.com/watch?v=3ICC4ftZP8Y">Google Gemini Insane Price Cuts!!!</a>：Google Gemini 1.5 Flash 迎来了疯狂降价！🔗 链接 🔗 详情 - https://developers.googleblog.com/en/gemini-15-flash-updates-google-ai-studio-gemini-...

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/1271325299979714570)** (1 messages): 

> - `Llama CPP Server`
> - `Prompt Caching`
> - `RAG-Based Interaction`
> - `Gemma 2` 


- **询问 Llama CPP Prompt Caching**：一位成员对使用 **Llama CPP server** 缓存 prompt 时应使用哪些参数表示困惑，寻求关于适当设置的澄清。
   - 他们的目标是缓存初始 prompt，同时允许 **Llama CPP** 管理动态 prompt 内容。
- **对选择性 Prompt Caching 的需求**：该成员澄清说，他们不想缓存所有用户交互，而是专注于第一个用户 prompt，该 prompt 较大，约为 **1.5k tokens**。
   - 他们正在探索将第一个/系统 prompt 保存在文件中，同时利用 **Llama CPP** 进行后续动态更新的可能性。


  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1271181377437696195)** (2 messages): 

> - `Llama 3 Model Details`
> - `Citing Axolotl in Academic Work` 


- **关于 Llama 3 训练细节的查询**：一位成员询问是否有关于 Meta 的 [Llama 3 模型](https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml) 训练过程的相关文档，特别是关于所使用的数据和 masks。
   - 他们注意到了一种独特的方法，即重命名现有 tokens 以充当模型中的 special tokens。
- **Axolotl 的引用偏好**：另一位成员寻求关于在学术论文或技术报告中引用 Axolotl 的首选方法的指导。
   - 这表明人们对在学术工作中正式认可 Axolotl 项目的兴趣日益浓厚。



**提到的链接**：<a href="https://huggingface.co/axolotl-ai-co/llama-3-8b-chatml">axolotl-ai-co/llama-3-8b-chatml · Hugging Face</a>：未找到描述

  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/)** (1 messages): 

drose0933: Yoooo
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1271297363029393481)** (4 messages): 

> - `AMD backend memory usage`
> - `GPU failure`
> - `De-sharding models`
> - `copy_to_device function` 


- **AMD backend 可能使用更多内存**：一位成员质疑 **AMD backend** 是否比 **GPU backend** 消耗更多内存。
   - 这引发了关于不同 backends 在内存管理方面的资源分配和性能的讨论。
- **报告 GPU 故障**：一位成员哀叹他们的 **GPU** 损坏了，简单地说道：*'Rip my GPU got blown.'*
   - 这引发了社区内对 GPU 可靠性以及在密集计算过程中面临的挑战的关注。
- **为了简化而进行模型去分片（De-sharding）**：一位用户询问如何对模型进行 “de-shard”，特别是通过将 **multi lazy buffer** 转换为 **normal lazy buffer**。
   - 这反映了社区在模型优化和架构适配方面面临的持续挑战。
- **copy_to_device 函数的使用**：提到了 **copy_to_device**，可能暗示了它在模型操作期间数据处理中的相关性。
   - 这表明用户需要进一步明确其工作流中的内存管理技术。


  

---



---



---



---



---



{% else %}


> 完整的逐频道细分内容已针对邮件进行了截断。
> 
> 如果您想查看完整的细分内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}