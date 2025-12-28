---
companies:
- meta-ai-fair
- openai
- google-deepmind
- hugging-face
- llamacloud
date: '2024-12-06T22:44:07.580241Z'
description: '**Meta AI** 发布了 **Llama 3.3 70B**，通过“全新的对齐流程和在线强化学习（RL）技术的进步”提升了效率，其性能可与
  405B 模型相媲美。**OpenAI** 宣布推出**强化微调（RFT）**，旨在利用有限的数据构建专家模型，并已向研究人员和企业开放 Alpha 访问权限。**Google
  DeepMind 的 Gemini-Exp-1206** 在基准测试中处于领先地位，其编程性能与 **GPT-4o** 持平。**LlamaCloud** 通过表格提取和分析功能增强了文档处理能力。社区关于
  **OpenAI** 定价方案的讨论仍在持续。'
id: c07ddfaa-9b6b-43d7-aeab-b37954159294
models:
- llama-3-70b
- llama-3.3-70b
- gpt-4o
- gemini-exp-1206
original_slug: ainews-meta-llama-33-405bnova-pro-performance-at
people:
- sama
- steven-heidel
- aidan_mclau
- lmarena_ai
- oriolvinyalsml
- jerryjliu0
title: Meta Llama 3.3：以 70B 的价格提供 405B/Nova Pro 级别的性能。
topics:
- reinforcement-learning
- fine-tuning
- model-performance
- document-processing
- pricing-models
- alignment
- online-rl
---

<!-- buttondown-editor-mode: plaintext -->**“一种新的对齐过程和在线 RL 技术的进展”就是你所需要的一切。**

> 2024年12月5日至12月6日的 AI 新闻。我们为您查看了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discord（**206** 个频道，**5628** 条消息）。预计节省阅读时间（以 200wpm 计算）：**535 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Meta AI 明智地等待 OpenAI 发布 [o1 finetuning 等候名单](https://x.com/openai/status/1865091561912164499?s=46&t=b7l37rB6wtbyAh6ah1NpZQ)，谢天谢地，他们保持了理智的版本策略，只是再次将 [Llama 次要版本号提升](https://x.com/AIatMeta/status/1865079068833780155)到了 3.3。这一次，他们通过 [“一种新的对齐过程和在线 RL 技术的进展”](https://x.com/AIatMeta/status/1865079068833780155)，让 70B 模型的性能追平了 405B。当然，没有发布论文。


![image.png](https://assets.buttondown.email/images/eb406324-34a7-438f-844a-504dca476c1c.png?w=960&fit=max)


Amazon Nova Pro 仅仅风光了 3 天，但随着 Meta 大声宣传以 12% 的成本实现相同的性能，它们在性价比层级中再次被击落。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 综述

> 所有综述均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是 Twitter 活动中的关键主题和讨论，按主要话题分类：

**Meta 发布 Llama 3.3 70B**

- **发布详情**：[@AIatMeta](https://twitter.com/AIatMeta/status/1865079069869773311) 宣布推出 Llama 3.3，这是一个 70B 模型，提供的性能可与 Llama 3.1 405B 媲美，但计算需求显著降低。该模型在 GPQA Diamond (50.5%)、Math (77.0%) 和 Steerability (92.1%) 上均有性能提升。
  - 包括 [@hyperbolic_labs](https://twitter.com/Yuchenj_UW/status/1865107298877870489) 和 [@ollama](https://twitter.com/ollama/status/1865094082508247365) 在内的几家供应商迅速宣布支持该模型的推理服务。
  - 该模型支持 8 种语言，并保持与之前 Llama 版本相同的许可证。

**OpenAI 宣布强化微调 (RFT)**

- **产品发布**：[@OpenAI](https://twitter.com/OpenAI/status/1865136373491208674) 预览了 Reinforcement Fine-Tuning (RFT)，允许组织使用有限的训练数据为特定领域构建专家模型。
  - [@stevenheidel](https://twitter.com/stevenheidel/status/1865104438928822767) 指出，RFT 允许用户使用 OpenAI 内部使用的相同流程来创建自定义模型。
  - 目前正通过一项研究计划向研究人员和企业提供 Alpha 测试权限。

**Google Gemini 性能更新**

- **新模型版本**：[@lmarena_ai](https://twitter.com/lmarena_ai/status/1865080944455225547) 宣布 Gemini-Exp-1206 目前在基准测试中领先，位列总榜第一，并在编程性能上与 GPT-4o 持平。
  - 该模型在包括硬核提示词（hard prompts）和风格控制在内的各种基准测试中均表现出进步。
  - [@OriolVinyalsML](https://twitter.com/OriolVinyalsML/status/1865082915442315286) 庆祝了 Gemini 发布一周年，并指出在超越自身基准测试方面取得的进展。

**LlamaCloud 与文档处理**

- **功能更新**：[@jerryjliu0](https://twitter.com/jerryjliu0/status/1865133794531082671) 展示了 LlamaCloud 从文档中提取表格并执行分析工作负载的能力。
  - 该平台现在支持直接在 UI 中渲染表格和代码。
  - [@jerryjliu0](https://twitter.com/jerryjliu0/status/1864848534530617660) 强调自动化提取是一个被忽视但极具价值的用例，特别是对于收据/发票处理。

**梗图与行业评论**

- **OpenAI 定价**：包括 [@aidan_mclau](https://twitter.com/aidan_mclau/status/1864880775591600427) 在内的多位用户对 OpenAI 每月 200 美元的方案发表了评论，并讨论了 AI 定价模型的经济学。
  - [@sama](https://twitter.com/sama/status/1864836360366174371) 澄清说，大多数用户使用免费层级或每月 20 美元的 Plus 层级就足够了。

---

# AI Reddit 综述

## /r/LocalLlama 综述

**主题 1：Llama 3.3 70B 性能对比 GPT-4o 及其他模型**

- **[Llama-3.3-70B-Instruct · Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)** ([Score: 465, Comments: 139](https://reddit.com/r/LocalLLaMA/comments/1h85ld5/llama3370binstruct_hugging_face/)): 该帖子关于在 **Hugging Face** 上发布的 **Llama-3.3-70B-Instruct** 模型，但缺乏关于其特性、功能或应用的额外细节或背景。
  - 讨论强调了 **Llama-3.3-70B-Instruct** 令人印象深刻的性能，指出尽管参数量显著减少，但其**能力可与 Llama 405B 媲美**。用户对其 **128K context** 和多语言能力印象深刻，基准测试显示其在**代码生成、推理和数学**方面有实质性改进。
  - 人们对该模型可能发布的更小版本感兴趣，因为由于 VRAM 限制，**70B 模型**对消费级硬件具有挑战性。讨论了 **quantizing**（量化）等技术，作为使其能在 RTX 4090 等具有 **24G VRAM** 的 GPU 上运行的方法，尽管这可能会影响输出质量。
  - 一些用户对该模型与基准测试相比的实际表现表示怀疑，并将其与 **Qwen2.5 72B** 进行了比较，讨论了性能扩展中的权衡。社区热切期待在未来的迭代中看到进一步的架构变化，例如 **Llama 4** 和 **Qwen 3**。


- **[Meta releases Llama3.3 70B](https://i.redd.it/ji1hp067d95e1.jpeg)** ([Score: 432, Comments: 100](https://reddit.com/r/LocalLLaMA/comments/1h85tt4/meta_releases_llama33_70b/)): Meta 发布了 **Llama3.3 70B**，该模型可作为 **Llama3.1-70B** 的直接替代品（drop-in replacement），且性能接近 **405B** 模型。该新模型因其成本效益、易用性和改进的可访问性而受到关注，更多信息可在 [Hugging Face](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) 上获得。
  - **Llama 3.3 70B** 较之前版本有显著的性能提升，正如 **vaibhavs10** 所强调的，在代码生成、多语言能力以及推理和数学方面有显著增强。该模型以更少的参数实现了与 **405B** 模型相当的性能，具体的指标改进包括代码生成的 HumanEval 提升了 7.9%，MATH (CoT) 提升了 9%。
  - 围绕**多语言支持**的讨论强调 Llama 3.3 除英语外还支持 7 种额外语言。然而，人们对缺乏 pretrained 版本表示担忧，正如 **Electroboots** 和 **mikael110** 根据 [官方文档](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_3/#introduction) 所提到的，目前仅提供 instruction-tuned 版本。
  - **Few_Painter_5588** 和 **SeymourStacks** 等评论者将 Llama 与 **Qwen 2.5 72b** 等其他模型进行了比较，指出 Llama 改进了文本质量和推理能力，尽管在某些基准测试中 Qwen 仍被认为更聪明。还有人呼吁建立更全面的基准测试，重点关注基础能力，而不是容易被 post-training 刷分的指标。


- **[New Llama 3.3 70B beats GPT 4o, Sonnet and Gemini Pro at a fraction of the cost](https://v.redd.it/3hyjhuz6ka5e1)** ([Score: 112, Comments: 0](https://reddit.com/r/LocalLLaMA/comments/1h8bgih/new_llama_33_70b_beats_gpt_4o_sonnet_and_gemini/)): 据报道，**Llama 3.3 70B** 在提供成本优势的同时，性能超越了 **GPT-4o, Sonnet 和 Gemini Pro**。帖子中未提供性能指标和成本比较的具体细节。

**主题 2. 开源 O1：呼吁更好的模型**

- **为什么我们需要开源的 o1** ([Score: 267, Comments: 135](https://reddit.com/r/LocalLLaMA/comments/1h7xret/why_we_need_an_open_source_o1/)): 作者批评了新的 **o1 模型**，指出它在编程任务中相比 **o1-preview** 有所退步，表现为无法遵循指令并对脚本进行未经授权的更改。他们认为这些问题凸显了对 **QwQ** 等开源模型的需求，因为私有模型可能会优先考虑利润而非性能和可靠性，使其不适用于关键系统。
  - 像 **QwQ** 这样的**开源模型**正因 **o1** 等私有模型的可靠性问题而受到关注，后者经常意外改变行为并破坏工作流。用户更倾向于 **open-weight** 解决方案以获得稳定性，因为他们可以控制更新并确保长期的性能一致性。
  - **o1 模型**因其在编程方面的糟糕表现而受到批评，用户报告了未经授权的更改和无法遵循指令的情况。这引发了对其在关键应用中适用性的担忧，一些用户认为 **OpenAI** 可能正在通过故意发布能力较弱的模型来削减成本。
  - 普遍情绪认为模型**自 GPT-4 以来一直在退步**，用户对 **o1** 和 **Gemini** 等较新的迭代版本表示不满。许多人认为这些变化是由商业策略而非技术改进驱动的，导致人们更倾向于旧模型或开源替代方案。


- **我是唯一一个对 O1 不感到惊艳的人吗？** ([Score: 124, Comments: 95](https://reddit.com/r/LocalLLaMA/comments/1h845wl/am_i_the_only_person_who_isnt_amazed_by_o1/)): 作者对 **O1** 模型表示怀疑，称其并不代表范式转移。他们认为 **OpenAI** 只是应用了开源 AI 社区现有的方法，例如 **OptiLLM** 以及自 **10 月**以来一直在使用的 "best of n" 和 "self consistency" 等 prompt 优化技术。
  - 许多用户对 **O1 模型**表示不满，将其描述为 **O1-preview** 的降级，并质疑每月支付 **$200** 订阅费用的价值。一些人认为该模型的局限性（如在长时间交互中丢失思路）使其不适合专业用途，他们更倾向于使用 **4o** 或 **Claude** 等其他替代方案。
  - 讨论涉及对 **OpenAI** 策略的看法，一些用户注意到该公司已转向“营利”模式，专注于增量升级而非突破性创新。这让那些觉得 **OpenAI** 优先考虑企业客户而非个人消费者的用户感到失望。
  - 对话触及了更广泛的 AI 领域，提到了 **QwenQbQ** 和 **DeepSeek R1** 等其他模型，以及 **open-source** 进步的潜力。用户强调需要将可靠的模型集成到工作流中，强调长短期记忆和成熟的 **Agent** 框架，而不仅仅是提高智能。


**主题 3. Windsurf Cascade 系统 Prompt 详情**

- **Windsurf Cascade 泄露的系统 prompt！！** ([Score: 173, Comments: 51](https://reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/)): **Windsurf Cascade** 是由 **Codeium 工程团队**设计的 **agentic AI 编程助手**，用于基于 **AI Flow 范式**的 IDE —— Windsurf。Cascade 协助用户完成创建、修改或调试代码库等编程任务，并使用 **Codebase Search**、**Grep Search** 和 **Run Command** 等工具进行操作。它强调异步操作、精确的工具使用和专业的沟通风格，同时确保代码更改是可执行且用户友好的。
  - 讨论强调了 AI 模型中 **prompt 的复杂性**，用户对尽管有许多负面表述的规则但复杂的 prompt 依然有效感到惊讶。人们对 **Windsurf Cascade** 使用的具体模型感到好奇。
  - 讨论了在 prompt 中使用 **HTML 样式标签**的情况，解释称它们提供了结构和焦点，帮助模型处理较长的 prompt。一些用户提到了与 **Anthropic 的 Erik Schluntz** 合作的播客，指出像 XML/HTML 样式的结构化标记比原始文本更有效。
  - 关于 prompt 中**正向强化有效性**的辩论，一些人认为正向语言可以通过将关键词与更好的解决方案联系起来提高模型性能。然而，其他人指出了不断向 prompt 添加条件的局限性，将其比作使用大量 "IF" 语句进行低效编程。


**主题 4. HuggingFace 课程：LLM 的偏好对齐 (Preference Alignment)**

- **[免费的 Hugging Face 本地 LLMs 偏好对齐 (preference alignment) 课程！](https://i.redd.it/1kqivo0yy65e1.png)** ([Score: 192, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1h7x4yh/free_hugging_face_course_on_preference_alignment/)): **Hugging Face** 提供了一门关于本地 **LLMs** **preference alignment** 的免费课程，包含 **Argilla**、**distilabel**、**lightval**、**PEFT** 和 **TRL** 等模块。该课程涵盖七个主题，“Instruction Tuning” 和 “Preference Alignment” 已经发布，而 “Parameter Efficient Fine Tuning” 和 “Vision Language Models” 等其他主题计划在未来发布。
  - **Colab 格式说明 (Colab Format Clarification)**：关于 “Colab format” 一词存在困惑，用户澄清课程材料是 **notebook** 格式，可以在 **Google Colab** 上运行，但主要是为本地运行设计的。**bburtenshaw** 强调 **notebooks** 包含在 **Colab** 中打开的链接以提供便利，尽管所有内容都旨在本地机器上运行。
  - **本地 LLMs 预期 (Local LLMs Expectation)**：像 **10minOfNamingMyAcc** 这样的用户希望课程能为本地 **LLMs** 提供本地代码库，这与课程关注本地模型训练和使用的初衷一致。该课程确实支持代码和模型的本地执行。
  - **课程获取 (Course Access)**：该课程可在 **GitHub** 上获取，**MasterScrat** 为有兴趣直接访问材料的人提供了链接 [点击此处](https://github.com/huggingface/smol-course)。


**主题 5. Adobe 发布用于自我编程 AI 的 DynaSaur 代码**

- **[Adobe 发布 DynaSaur 代码：一个可以自我编程的 Agent](https://github.com/adobe-research/dynasaur)** ([Score: 88, Comments: 13](https://reddit.com/r/LocalLLaMA/comments/1h7w11d/adobe_releases_the_code_for_dynasaur_an_agent/)): **Adobe** 发布了 **DynaSaur** 的代码，这是一个能够自我编程的 **Agent**。此举突显了 Adobe 在 AI 领域，特别是在自主编程 **Agent** 方面的贡献。
  - **Eposnix** 建议在 **VM** 中运行 **DynaSaur**，因为存在它无限迭代并可能导致系统损坏的风险。他们建议 **confidence scoring** 可以防止这种情况，如果任务太难，允许 AI 退出，而不是坚持使用可能有危害的解决方案。
  - **Knownboyofno** 解释说 **DynaSaur** 可以通过生成 **Python** 函数来自主创建工具，以实现特定目标，这让人们对其能力有了更清晰的理解。
  - **Staladine** 和其他人表示有兴趣看到 **DynaSaur** 运行的实际案例或演示，这表明需要更多说明性资源来理解其功能。


## 其他 AI Subreddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. OpenAI GPT-4.5：在创意语言任务中超出预期**

- **[让她吐槽 UHC CEO](https://i.redd.it/ljuuc0vxs35e1.jpeg)** ([Score: 195, Comments: 27](https://reddit.com/r/ChatGPT/comments/1h7lcef/asked_her_to_roast_the_uhc_ceo/)): **ChatGPT** 批判了医疗保险行业，将其利润驱动的行为比作“用人命玩大富翁”，反思了将利润置于患者护理之上的道德影响。对话还触及了吐槽一个被暗杀的人所面临的挑战。
  - 讨论强调了 **ChatGPT 不断进化的能力**，用户对其能够提供尖锐批判（特别是针对保险高管且无审查）感到震惊。评论反映了 **ChatGPT 的大胆**，并暗示它变得更加激进，尤其是在政治背景下。
  - 一位用户幽默地指出 **“事实总是带有自由主义偏见”**，表明其认为 **ChatGPT** 的批判与自由主义观点一致。这突显了 AI 在挑战敏感行业中既定规范和人物方面所扮演的角色。
  - 社区通过幽默和梗图参与该帖子，展示了对 AI 关于保险行业评论的**轻松而又带有批判性的反响**，并称其批判的严厉程度是“残暴的”和“致命的”。


---

# AI Discord 综述

> 由 O1-mini 生成的摘要之摘要的摘要

**主题 1. AI 模型发布与性能大逃杀**

- [**Meta 的 Llama 3.3 性能超越 405B 竞争对手**](https://x.com/Ahmad_Al_Dahle/status/1865071436630778109)：拥有 **70B** 参数的 **Meta Llama 3.3** 在保持更高成本效益的同时，性能媲美 **405B** 模型，引发了与 **Gemini-exp-1206** 和 **Qwen2-VL-72B** 模型的对比。
  - 用户盛赞 Llama 3.3 **增强的数学解题能力**以及在编程任务中的**稳健表现**，认为其非常适合各种工程项目。
  - 该版本的发布激发了竞争性的基准测试，社区成员渴望将其集成并针对既定标准进行测试。
  - 一位用户惊叹道：*“看到了语法处理方面的显著改进，”* 强调了该模型的高级能力。
    
- [**Gemini-exp-1206 在编程基准测试中与 O1 持平**](https://x.com/lmarena_ai/status/1865080944455225547)：**Google 的 Gemini-exp-1206** 模型夺得总榜**首位**，在编程基准测试中与 **O1** 旗鼓相当，推向了 AI 性能的技术边界。
  - 该模型展示了在**合成数据生成**和**高性价比推理**方面的重大进展，吸引了关注可扩展性的开发者。
  - 社区讨论强调了 Gemini-exp-1206 在复杂 AI 应用中**超越预期**的潜力。
  - [探索 Gemini-exp-1206 的功能](https://x.com/lmarena_ai/status/1865080944455225547)。

**主题 2. 价格变动引发用户不满**

- [**Windsurf 大幅涨价令订阅者感到沮丧**](https://x.com/windsurf_ai/status/1865131244574642639)：**Codeium** 将 **Windsurf** 的 Pro 档位上调至 **$60/月**，并对 Prompt 和 Flow Action 设置了新的**硬性限制**，导致许多用户不满并寻求关于**祖父条款（grandfathering）**政策的说明。
  - 订阅者对在尚未修复现有 Bug 的情况下突然涨价表示愤怒，质疑新定价模式的**可持续性**。
  - 尽管一些替代方案也存在类似的**可靠性问题**，但这些突如其来的变化加速了用户对 **Cursor**、**Bolt AI** 和 **Copilot** 等工具的探索。
  - 一位用户哀叹道：*“考虑到目前的性能，这个定价是不可持续的。”*
  - [查看 Windsurf 的新定价详情](https://x.com/windsurf_ai/status/1865131244574642639)。

- [**Lambda Labs 削减模型价格以吸引开发者**](https://x.com/DeepInfra/status/1865126860902011244)：**DeepInfra** 下调了多个模型的价格，包括 **Llama 3.2 3B Instruct** 仅需 **$0.018**，**Mistral Nemo** 仅需 **$0.04**，旨在为预算敏感型开发者提供**实惠**的选择。
  - 这些降价举措使高质量模型变得更加易于获取，促进了开发者社区内更广泛的采用和创新。
  - 用户对更低的成本表示欢迎，并指出了**价值主张**的提升和**可访问性**的增加。
  - [查看 DeepInfra 的降价信息](https://x.com/DeepInfra/status/1865126860902011244)。

**主题 3. 工具稳定性故障与用户挫败感**

- [**Claude 的代码处理困境阻碍开发者**](https://huggingface.co/CodexAPI)：**用户报告 Windsurf 和 Claude 等工具存在重大 Bug**，导致**性能**不可靠且**错误率**增加，使编程任务变得更加繁琐。
  - 持续的**服务器宕机**和 “resource_exhausted” 等问题损害了生产力，导致用户重新考虑其订阅。
  - 社区共识强调，在进行任何进一步价格调整之前，AI 工具迫切需要**可靠的性能**。
  - [阅读更多关于 Claude 的用户反馈](https://huggingface.co/CodexAPI)。

- [**Cursor 响应缓慢迫使用户转向替代方案**](https://github.com/stackblitz/bolt.new/issues/678)：用户报告 **Cursor 的 Composer** 出现**连接失败**和**响应缓慢**，通常需要开启新会话才能恢复功能，导致用户感到沮丧并向更稳定的工具（如 **Windsurf**）迁移。
  - 尽管 **Cursor 0.43.6** 推出了新功能，但 **Composer 响应**不可靠等问题依然存在，削弱了用户体验。
  - 讨论强调需要通过强大的 **Bug 修复**和**性能改进**来留住用户信任。
  - 一位开发者指出：*“Cursor 的性能未达到预期。”*
  - [探索 Cursor 的性能问题](https://github.com/stackblitz/bolt.new/issues/678)。

**主题 4. 功能增强与新集成发布**

- [**Aider Pro 升级引入高级语音和上下文功能**](https://aider.chat/docs/leaderboards/)：**Aider Pro** 现在包含**无限制高级语音模式**、针对 O1 的全新 **128k 上下文**，以及**复制/粘贴到 Web 聊天**功能，提升了工作流效率以及处理大量文档和代码的能力。
  - 此外，**进程挂起支持**和**异常捕获分析**为用户提供了更好的进程控制和洞察。
  - 用户反馈称赞 Aider 实现了 **61% 的代码贡献**，展示了其不断增长的能力和稳健的发展。
  - [探索 Aider Pro 的新功能](https://aider.chat/docs/leaderboards/)。

- [**OpenRouter 的作者页面简化了模型发现**](https://openrouter.ai/author)：**OpenRouter** 推出了**作者页面 (Author Pages)**，使用户能够通过创作者探索模型，并通过便捷的**轮播 (carousel)** 界面展示详细统计数据和相关模型。
  - 该功能增强了**模型发现**并允许更好的**分析**，使用户更容易查找和评估各种 AI 模型。
  - 社区期待通过不同作者的收藏集来改善**用户体验**并简化导航。
  - [访问 OpenRouter 的作者页面](https://openrouter.ai/author)。

**主题 5. 社区关注：安全、许可和虚假应用**

- [**警惕虚假的 Perplexity 应用！**](https://github.com/ultralytics/ultralytics/issues/18027)：Discord 用户提醒社区，Windows 应用商店中流传着一个**虚假 Perplexity 应用**，该应用欺骗性地使用官方 Logo 和未经授权的 API，将用户引导至可疑的 **Google Doc**，并敦促立即举报以防止安全漏洞。
  - 成员们强调了验证应用真实性的重要性，以避免暴露于恶意软件和钓鱼攻击中。
  - 讨论强调了保持警惕和采取社区驱动措施来打击欺诈应用程序的必要性。
  - [举报虚假 Perplexity 应用](https://github.com/ultralytics/ultralytics/issues/18027)。

- [**Phi-3.5 对 AI 响应过度审查**](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)：**微软的 Phi-3.5** 模型因**高度审查**而受到批评，使其对冒犯性查询产生抵触，并可能限制其在技术任务中的实用性，引发了关于 AI 模型中**安全性**与**可用性**平衡的辩论。
  - 用户讨论了**解除审查**或**改进**模型功能的方法，包括分享 [Hugging Face 上的无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored) 链接。
  - 开发者对**审查对编码和技术应用的影响**表示担忧，敦促寻求具有更好**上下文理解**的模型。
  - 一位用户辩称：“Phi-3.5 的审查使其在许多实际应用中变得不切实际。”
  - [探索 Phi-3.5 的无审查版本](https://huggingface.co/SicariusSicariiStuff/Phi-3.5-mini-instruct_Uncensored)。

- [**AI 工具中的安全疏忽引发警报**](https://x.com/mckaywrigley/status/1865089975802646857)：围绕 AI 工具**安全问题**的讨论突出了过度审查和缺乏安全的**许可协议**等问题，强调需要更好的安全协议和透明的许可来保护用户利益。
  - 社区成员呼吁改进**监管机制**，以确保 AI 模型既安全又实用，避免阻碍实际使用的过度限制。
  - 一位参与者表示：“我们需要在 AI 模型的安全性和可用性之间取得平衡。”
  - [了解 AI 模型安全问题](https://x.com/mckaywrigley/status/1865089975802646857)。


---

# 第一部分：Discord 高层级摘要

## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Windsurf Pricing Overhaul**：Codeium 将 **Windsurf** 的 Pro 层级价格上调至 **$60/month**，并对用户 prompts 和 flow actions 引入了硬性限制，这引起了许多订阅者的不安。
   - 用户要求明确新的定价结构，以及现有方案是否会执行 **grandfathered**（老用户保留原价），并对在未修复现有 Bug 的情况下突然涨价表示不满。
- **User Frustrations with AI Tools**：工程师们报告了 **Windsurf** 等工具中存在的重大 Bug，阻碍了高效编码，并导致他们重新考虑是否续订。
   - 共识认为，AI 工具在实施进一步价格调整之前，需要确保 **reliable performance** 和用户友好的功能。
- **Alternatives to Windsurf**：针对 Windsurf 的定价和性能问题，用户正在探索 **Cursor**、**Bolt AI** 和 **Copilot** 等替代方案，以获得更一致的性能。
   - 尽管在考虑这些替代方案，一些用户仍保持谨慎，因为据报道 **Bolt AI** 等工具也面临类似的可靠性挑战。
- **Impact of Server Issues**：频繁的服务器宕机和 'resource_exhausted' 等错误正在干扰 **Windsurf** 的使用，对用户生产力产生了负面影响。
   - 这些技术问题加剧了用户的挫败感，并加速了向其他 AI 编码解决方案的转移。
- **Feedback on AI Tool Performance**：用户强调 **Claude** 在上下文保留（context retention）方面表现吃力，并在代码中引入错误，降低了其在开发任务中的有效性。
   - 这些反馈强调了 AI 工具需要增强其 **accuracy** 和 **contextual understanding**，以更好地满足工程项目的需求。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Audio Generation with NotebookLM**：成员们探索了使用 **NotebookLM** 进行音频生成，成功地从文档中创建了播客。*一位用户报告称，从一份多语言文档中生成了长达 64 分钟的播客*，突显了基于输入类型的不同结果。
   - 讨论揭示了在 AI 生成的音频中保持连贯性和重点的挑战，尽管使用了有效的提示词（prompting）技术，一些用户仍遇到了意料之外的跑题现象。
- **Language and Voice Support in NotebookLM**：对话集中在 **NotebookLM** 对英语以外语言的支持上，一些用户回想起之前仅限英语的限制。生成的音频中令人印象深刻的 **voice quality** 引发了关于其作为独立 text-to-speech 解决方案潜力的辩论。
   - 用户质疑语言支持的范围，讨论了扩展 NotebookLM 多语言能力的可能性，以增强其对全球工程师受众的实用性。
- **Game Development using Google Docs and AI**：工程师们分享了利用 **Google Docs** 组织游戏规则和叙事的策略，利用 AI 生成场景并构建沉浸式世界。*一位成员强调了在他们的 RPG 游戏中，AI 生成的融合了严肃与幽默内容的场景取得了成功*。
   - AI 在游戏开发中的集成因增强了创作过程而受到称赞，用户强调了 **Google Docs** 作为叙事构建协作工具的灵活性。
- **Spreadsheet Integration Workarounds for NotebookLM**：用户发现了将 **电子表格直接上传** 到 **NotebookLM** 的限制，建议采用将数据转换为 **Google Docs** 等替代方案以获得更好的兼容性。*一位用户提到通过隐藏不必要的列来降低电子表格的复杂性，从而整合核心数据*。
   - 讨论了集成电子表格数据的创意方法，重点是在规避 NotebookLM 上传限制的同时保持数据完整性。
- **NotebookLM Performance and Usability Feedback**：对 **NotebookLM** 性能的反馈褒贬不一，涉及生成内容的准确性和深度。用户强调需要更多关于潜在 **paywalls** 的透明度以及一致的性能指标。
   - 关于 **new notebook** 按钮消失的担忧引发了对可能存在笔记本数量限制的猜测，这影响了 NotebookLM 的整体可用性和工作流。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **PaliGemma 2 发布扩展了模型选择**：Google 推出了 [PaliGemma 2](https://huggingface.co/blog/paligemma2)，包含 **3B**、**10B** 和 **28B** 参数的新预训练模型，为开发者提供了更大的灵活性。
   - **SigLIP** 在视觉任务中的集成以及文本解码器升级至 Gemma 2，预计将比之前的版本带来性能提升。
- **Qwen 微调遭遇 VRAM 限制**：工程师在 **80GB** GPU 上微调 **Qwen32B** 模型时遇到问题，需要 **96GB** H100 NVL GPU 才能防止 OOM 错误（[Issue #1390](https://github.com/unslothai/unsloth/issues/1390)）。
   - 对话显示 **QLORA** 可能会比 **LORA** 消耗更多内存，导致目前正在对 VRAM 消耗差异进行持续调查。
- **Unsloth Pro 期待即将到来的发布**：**Unsloth Pro** 计划于近期发布，引发了期待增强功能的用户的兴奋。
   - 社区成员期待利用 **Unsloth Pro** 来简化工作流并利用新的模型能力。
- **Llama 3.3 推出 70B 模型并提升效率**：**Llama 3.3** 已发布，其 **70B** 参数模型在提供强劲性能的同时降低了运营成本（[Ahmad_Al_Dahle 的推文](https://x.com/Ahmad_Al_Dahle/status/1865071436630778109)）。
   - **Unsloth** 推出了 Llama 3.3 的 **4-bit** 量化版本，提升了加载速度并减少了内存占用。
- **优化 LoRA 微调配置**：'Silk.ai' 质疑了 LoRA 微调中 **use_cache** 参数的必要性，引发了关于最佳设置的讨论。
   - 另一位贡献者强调了启用 **LoRA dropout** 以实现预期模型性能的重要性。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 性能受挫**：用户报告称 **Cursor** 在使用 **Composer** 时经历了 **连接失败** 和 **响应缓慢**，通常需要开启新会话才能正常工作。
   - 许多人将其性能与 **Windsurf** 进行对比并给出负面评价，对 **持续存在的问题** 表示沮丧。
- **Windsurf 超越 Cursor**：几位用户提到 **Windsurf** 在处理任务时表现更好，即使在繁重的代码生成需求下也没有出现问题。
   - 用户强调，虽然 **Cursor** 在应用更改时显得吃力，但 **Windsurf** 能够顺畅执行类似任务，这改变了用户的偏好。
- **Cursor 0.43.6 增加侧边栏集成**：在最新的 **Cursor 0.43.6** 更新中，用户注意到 **Composer UI** 已集成到侧边栏中，但一些功能如 **long context chat** 已被移除。
   - 还提到了新功能，如 **inline diffs**、**git commit 信息生成** 以及 **agent** 的早期版本。
- **Composer 响应不可靠**：用户分享了关于 **Cursor** 的 **Composer** 功能的参差不齐的体验，有报告称它有时无法响应查询。
   - 问题包括 Composer 未能生成预期的代码或遗漏更新，尤其是在最近的更新之后。
- **探索使用 Cursor 进行单元测试**：一位用户询问了使用 **Cursor** 编写 **unit tests** 的有效方法，并对分享的技术表示感兴趣。
   - 虽然尚未有定论性的答复，但用户鼓励分享各自的测试 **经验** 和 **方法**。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 推出作者页面 (Author Pages)**：OpenRouter 推出了 **Author Pages** 功能，使用户可以在 [openrouter.ai/author](https://openrouter.ai/anthropic) 探索创作者的模型。此次更新包括详细的统计数据以及通过轮播图展示的相关模型。
   - 该功能旨在增强**模型发现**和**分析**，为用户浏览不同作者的收藏提供流线化的体验。
- **Amazon Nova 模型收到褒贬不一的反馈**：用户报告了对 **Amazon Nova** 模型不同的使用体验，称其中一些模型与 [Nova Pro 1.0](https://openrouter.ai/amazon/nova-pro-v1) 等替代方案相比表现欠佳。
   - 尽管存在批评，某些用户仍强调了该模型的**速度**和**性价比**，表明用户满意度存在分歧。
- **Llama 3.3 的部署与性能**：**Llama 3.3** 已成功发布，供应商在发布后不久即提供支持，正如 [OpenRouter 的公告](https://x.com/OpenRouterAI/status/1865090466250711430)中所述，增强了文本应用的能力。
   - **AI at Meta** 指出，该模型有望在生成合成数据方面提高性能，同时降低推理成本。
- **DeepInfra 降低模型定价**：根据其[最新推文](https://x.com/DeepInfra/status/1865126860902011244)，**DeepInfra** 宣布大幅下调多款模型的价格，包括 **Llama 3.2 3B Instruct** 降至 **$0.018**，**Mistral Nemo** 降至 **$0.04**。
   - 这些降价旨在为**预算有限的开发者**提供以更实惠的价格获取高质量模型的途径。
- **OpenAI 推出强化学习微调**：在 **OpenAI Day 2** 期间，公司宣布即将为 **o1** 提供**强化学习微调 (reinforcement learning finetuning)**，尽管这在社区中引起的兴奋有限。
   - 参与者对这些更新表达了怀疑，期待在现有产品之外能有更实质性的进展。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **MoE-lite Motif 提升 Transformer 效率**：一位成员介绍了 **MoE-lite motif**，它利用自定义的 bias-per-block-per-token 非线性地影响残差流 (residual stream)，这表明尽管参数成本增加，但**计算速度更快**。
   - 讨论将其效率与传统的 **Mixture of Experts (MoE)** 架构进行了比较，辩论了潜在的优缺点。
- **GoldFinch 架构精简 Transformer 参数**：一位成员详细介绍了 **GoldFinch** 模型，该模型通过从变异的 layer 0 嵌入中推导来移除 V 矩阵，显著提高了**参数效率**。[GoldFinch 论文](https://arxiv.org/abs/2407.12077)
   - 团队讨论了替换或压缩 K 和 V 参数的潜力，旨在提高 Transformer 的整体效率。
- **逐层 Token 嵌入优化 Transformer 参数**：成员们探索了**逐层 Token 值嵌入 (layerwise token value embeddings)** 作为传统值矩阵 (value matrices) 的替代方案，在不损害性能的情况下实现了 Transformer 的显著**参数节省**。
   - 该方法利用初始嵌入动态计算 V 值，从而减少了对广泛值投影 (value projections) 的依赖。
- **更新后的 Mechanistic Interpretability 资源现已发布**：一位成员分享了一个 [Google Sheets 资源](https://docs.google.com/spreadsheets/d/1x2jgYc_2GuBkhST8gUuQVb2zla3SCXVNXGE0JaC5ArI/edit?usp=sharing)，按主题分类编目了 **Mechanistic Interpretability** 领域的关键论文，以便于流线化探索。
   - 该资源包括**基于主题的分类**和**注释笔记**，以协助研究人员有效地查阅基础文献。
- **动态权重调整提升 Transformer 效率**：成员们提议通过**动态权重调整 (dynamic weight adjustments)** 来增强**参数分配**和 Transformer 效率，这与 momentum 等正则化方法有相似之处。
   - 对话强调了通过消除或修改 V 参数来提升潜在性能和简化计算的可能性。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Aider v0.67.0 发布，带来新特性**：最新的 **Aider v0.67.0** 引入了对 [Amazon Bedrock Nova 模型](https://aider.chat/docs/leaderboards/) 的支持、增强的命令功能、进程挂起支持以及异常捕获分析。
   - 值得关注的是，Aider 为该版本的开发贡献了 **61% 的代码**，展示了其强大的能力。
- **Aider Pro 功能备受关注**：**Aider Pro** 现在包含无限制的高级语音模式、针对 O1 的全新 128k 上下文，以及 [复制/粘贴到 Web 聊天](https://aider.chat/docs/usage/copypaste.html) 功能，实现了与 Web 界面的无缝集成。
   - 用户称赞这些功能使其能够处理大量的文档和代码，提升了工作流效率。
- **Gemini 1206 模型发布引发兴趣**：Google DeepMind 发布了 **Gemini-exp-1206** 模型，声称比之前的迭代版本有性能提升。
   - 社区成员渴望看到其与 [Claude](https://x.com/JeffDean/status/1865079431544607089) 等模型的对比基准测试，并期待 Paul Gauthier 的详细性能结果。
- **DeepSeek 在 Aider 中的表现**：**DeepSeek** 被讨论为 Aider 用户的一个高性价比选择，此外还有 Qwen 2.5 和 Haiku 等替代方案。
   - 有推测认为，通过微调社区版本，有望提升 DeepSeek 在 Aider 中的基准测试表现。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini-exp-1206 夺得榜首**：新的 [Gemini-exp-1206](https://x.com/lmarena_ai/status/1865080944455225547) 模型获得了总分第一，并在编程基准测试中与 **O1** 持平，标志着较之前版本的重大改进。
   - OpenAI 的演示显示，基于医学数据微调的 **O1-mini** 可以超越完整的 **O1** 模型，这进一步凸显了 Gemini 强劲的表现。
- **Llama 3.3 带来高性价比性能**：[Llama 3.3](https://x.com/AIatMeta/status/1865079068833780155) 的增强得益于更新的对齐流程和在线强化学习（Reinforcement Learning）技术的进步。
   - 该模型在性能上与 **405B** 模型相当，同时能在标准开发者工作站上实现更具**成本效益的推理**。
- **阿里巴巴发布 Qwen2-VL-72B**：阿里云推出了 [Qwen2-VL-72B](https://huggingface.co/Qwen/Qwen2-VL-72B) 模型，具备先进的视觉理解能力。
   - 该模型专为多模态任务设计，在视频理解方面表现出色，并可在各种设备上无缝运行，旨在提升多模态性能。
- **强化微调（Reinforcement Fine-Tuning）推动 AI 模型进步**：讨论强调了 **Reinforcement Learning** 在微调模型以超越现有竞争对手方面的作用。
   - 关键点包括在模型训练中使用预定义的评分器（graders），以及不断演进的 RL 训练方法论。
- **AI 竞争驱动模型创新**：成员们呼吁在 AI 领域开展强有力的竞争，敦促 **OpenAI** 挑战 **Claude** 和 **DeepSeek** 等模型以促进进步。
   - 这种观点强调了社区的信念，即有效的竞争对手对于 AI 领域的持续进步至关重要。

---

## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **提升 Bolt.new 的 Token 效率**：成员们讨论了如 **特定部分编辑 (Specific Section Edits)** 等策略，通过仅修改选定部分而非重新生成整个文件来减少 **token usage**，旨在提高 **token management** 效率。
   - 提出了关于免费账户 **daily token limits** 的问题，以及购买 **token reload option** 以允许 token 结转的好处。
- **将 GitHub 仓库与 Bolt.new 集成**：用户探索了 **GitHub Repo Integration**，通过使用仓库 URL（如 [bolt.new/github.com/org/repo](https://bolt.new/github.com/org/repo)）启动 Bolt，并指出 **private repositories** 目前需要设置为公开才能成功集成。
   - 为了解决与私有仓库相关的 **deployment errors**，用户建议切换到公开仓库以绕过权限问题。
- **管理功能请求与改进**：讨论强调了通过与 Bolt 互动来单独处理请求，从而实现高效的 **Feature Requests Management**，这有助于减少 Bot 响应中的 **hallucination**。
   - 社区成员建议通过 [GitHub Issues 页面](https://github.com/stackblitz/bolt.new/issues) 提交功能增强想法，强调了 **user feedback** 对产品开发的重要性。
- **利用本地存储和后端集成优化开发**：开发者建议最初使用 **local storage** 构建应用程序，然后将功能迁移到 **Supabase** 等 **backend solutions**，以促进更顺畅的测试并简化 **integration process**。
   - 确认该方法有助于保持应用的完善度，并减少向数据库存储过渡期间的错误。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Reactor 的换脸大决战**：用户讨论了 **Reactor** 是否是换脸的最佳选择，目前尚未达成明确共识。
   - 参与者建议尝试各种模型，以评估它们对 **output quality** 的影响。
- **AI Discord 社区讨论多样化**：一位用户正在寻找除了 LLM 之外讨论多样化 AI 主题的 Discord 社区，引发了相关推荐。
   - 成员们推荐了 **Gallus** 和 **TheBloke** 的 Discord 频道，将其作为广泛 AI 讨论的枢纽。
- **云端 GPU 供应商的价格战**：用户分享了首选的 **Cloud GPU** 供应商，如 **Runpod**、**Vast.ai** 和 **Lambda Labs**，突出了其竞争力的定价。
   - **Lambda Labs** 被指出通常是最便宜的选择，尽管获取访问权限可能具有挑战性。
- **Lora 和 ControlNet 调整 Stable Diffusion**：讨论围绕在 **Stable Diffusion** 中调整 **Lora** 的强度展开，指出其强度可以超过 1，但在更高设置下存在图像失真的风险。
   - 成员们建议使用 **OpenPose** 以获得准确的姿势，并利用 **depth control** 来改善结果。
- **AI 艺术许可困境**：一位用户提出了关于超过 **Stability AI** 许可协议中收入阈值的问题。
   - 澄清表明输出内容仍然可以使用，但模型使用的许可在终止后将被撤销。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 12 天活动中的 Reinforcement Fine-Tuning**：[YouTube 活动 '12 Days of OpenAI: Day 2'](https://www.youtube.com/live/fMJMhBFa_Gc?si=rKhAmwYzWJPRDdLp) 由 OpenAI 研究高级副总裁 **Mark Chen** 和 **Justin Reese** 主持，讨论了 **reinforcement fine-tuning** 的最新进展。
   - 鼓励参与者在太平洋时间 **10am PT** 加入直播，直接从领先的研究人员那里获取见解。
- **Gemini 1206 实验模型超越 O1 Pro**：**Gemini 1206 实验模型**因其强劲表现而受到关注，在生成详细独角兽插图的 SVG 代码等任务中超越了 **O1 Pro**。
   - 用户报告称 Gemini 1206 提供了增强的结果，特别是在 **SVG generation** 和其他技术应用方面表现出色。
- **O1 Pro 与 Gemini 1206 的定价对比**：**O1 Pro** 定价为 **$200/月**，引发了关于其与 **Gemini 1206** 等免费替代方案相比价值如何的讨论。
   - 一些用户认为，尽管 O1 功能强大，但考虑到有效免费模型的可用性，如此高的成本是不合理的。
- **对高级 Voice Mode 功能的需求**：**社区对更高级的 voice mode** 有明确需求，目前的版本因声音机械化而受到批评。
   - 用户表达了对该功能重大改进的希望，特别是在即将到来的**假期期间**。
- **提议 GPT 协作编辑功能**：一名成员表达了希望允许多个编辑者同时修改一个 **GPT** 的愿望，强调了协作的需求。
   - 目前只有创建者可以编辑 GPT，但社区建议增加“Share GPT edit access”功能以促进团队合作。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **VSCode Extension 查询已解决**：一名成员遇到了 **VSCode extension** 测试以 **cwd=/** 运行的问题，在找到询问该扩展的合适频道后，问题得到了解决。
   - 这一事件强调了将技术查询定向到**正确的社区频道**以高效解决问题的重要性。
- **Mojo 函数提取错误**：一名用户在适配 Mojo **math module** 中的 `j0` 函数时遇到错误，原因是编译期间出现了未知的声明 `_call_libm`。
   - 他们寻求关于如何正确从 **math standard library** 提取和利用函数而不遇到编译器问题的指导。
- **编程职业专业化**：成员们讨论了专注于 **blockchain**、**cryptography** 或 **distributed systems** 等领域对提升技术就业前景的好处。
   - 重点强调了有针对性的学习、实战项目以及对基础概念的扎实掌握，以促进职业发展。
- **Mojo 中的 Compiler Passes 和 Metaprogramming**：讨论重点介绍了 Mojo 的新功能，这些功能支持自定义 compiler passes，并提出了增强 API 以实现更广泛程序转换的想法。
   - 成员们将 Mojo 的 metaprogramming 方法与传统的 [LLVM Compiler Infrastructure Project](https://llvm.org/devmtg/2024-10/#program) 进行了比较，并指出了 JAX 风格程序转换的局限性。
- **计算机科学教育见解**：参与者分享了关于具有挑战性的**计算机科学课程**和项目的经验，这些经历加深了他们对编程概念的理解。
   - 他们讨论了如何在个人兴趣与市场需求之间取得平衡，并以自己的学术历程为例。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 面临 Code Interpreter 限制**：用户报告称，即使在上传了相关文件后，**Perplexity AI** 的 Code Interpreter 仍无法执行 Python 脚本，其功能仅限于生成文本和图表。
   - 这一限制引发了关于 **Perplexity AI** 支持实际代码执行必要性的讨论，以更好地满足技术工程需求。
- **Windows 应用商店出现虚假 Perplexity 应用**：成员们发现 Windows 应用商店中存在一个**虚假 Perplexity 应用**，该应用欺骗性地使用了官方 Logo 和未经授权的 API，并将用户引导至一个可疑的 Google Doc。
   - 社区敦促举报该欺诈应用，以防止潜在的安全风险并保护 **Perplexity AI** 产品的完整性。
- **Llama 3.3 模型发布并增强功能**：**Llama 3.3** 正式发布，因其较前代版本的性能提升而受到用户欢迎。
   - 社区强烈期待 **Perplexity AI** 将 **Llama 3.3** 集成到其服务中，以利用其先进的功能。
- **利用 Grok 和 Groq 优化 API 使用**：关于使用 **Grok** 和 **Groq** API 的讨论显示，**Grok** 提供免费的初始额度，而 **Groq** 则通过集成 **Llama 3.3** 提供免费使用。
   - 用户分享了故障排除技巧，指出了 **Groq** 端点（endpoint）面临的挑战，部分成员通过社区支持成功解决了这些问题。
- **为 Perplexity API 引入 RAG 功能**：一名成员询问如何将 **Perplexity Spaces** 中的 **RAG** 功能整合到 API 中，表明了对高级检索能力的需求。
   - 这种兴趣凸显了社区对增强 **Perplexity API** 功能的需求，以支持更复杂的数据检索流程。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Paligemma 2 发布**：[Paligemma 2](https://x.com/prince_canuma/status/1864801741281124730?s=46) 在 **MLX** 上发布，引入了来自 **GoogleDeepMind** 的新模型，增强了平台的能力。
   - 鼓励用户使用 `pip install -U mlx-vlm` 进行安装，通过 Star 该项目进行支持，并提交 Pull Requests。
- **RAG 文件限制**：一名成员讨论了针对 **5 个文件 RAG 限制**的变通方法，强调了分析多个小文件以进行问题检测的必要性。
   - 社区成员商讨了潜在的解决方案，以及使用模型处理较小批量文件的性能影响。
- **Llama 3.1 CPU 基准测试**：用户请求在 **Intel i7-13700** 和 **i7-14700** CPU 上对 **Llama 3.1 8B 模型**进行基准测试，以评估潜在的推理速度。
   - 社区见解表明，根据近期用户在类似 CPU 配置下的经验，性能指标各不相同。
- **4090 GPU 价格飙升**：据报告，某些地区的 **4090 GPU** 新卡和二手卡价格均出现飙升，引发用户担忧。
   - 传闻称某些 **4090 GPU** 可能会被改装以将 VRAM 扩展至 **48GB**，引发了进一步讨论。
- **中国改装 4090 GPU**：提到了 Reddit 上关于中国改装者研究 **4090 GPU** 的讨论，但未提供具体来源。
   - 用户表示在查找有关这些 GPU 改装活动的详细信息或链接时面临挑战。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Rerank 3.5 模型提升搜索准确率**：新发布的 [Rerank 3.5 模型](https://cohere.com/blog/rerank-3pt5) 提供了改进的推理和多语言能力，能够对复杂的企业数据进行更准确的搜索。
   - 成员们正在寻求 **benchmark scores**（基准测试分数）和 **performance metrics**（性能指标）来评估 Rerank 3.5 的有效性。
- **Structured Outputs 简化 Command 模型**：Command 模型现在强制执行严格的 [Structured Outputs](https://docs.cohere.com/docs/structured-outputs#structured-outputs-tools)，确保包含所有 **required**（必填）参数，并增强了企业级应用中的可靠性。
   - 用户可以在文本生成中使用 **JSON** 格式的 Structured Outputs，或通过 function calling 使用 **Tools**，该功能目前在 **Chat API V2** 中处于实验阶段，欢迎用户提供反馈。
- **vnc-lm 集成 LiteLLM 以增强 API 连接**：**vnc-lm** 现在已与 **LiteLLM** 集成，能够连接到任何支持 Cohere 模型的 API，如 Cohere API 和 OpenRouter。
   - 正如 [GitHub](https://github.com/jake83741/vnc-lm) 上所示，该集成实现了无缝的 API 交互，并支持包括 Claude 3.5、Llama 3.3 和 GPT-4o 在内的多个 LLM。
- **/embed 端点面临速率限制问题**：用户对 **/embed** 端点每分钟 **40 张图像** 的低速率限制（rate limit）表示不满，这限制了高效嵌入数据集的能力。
   - 成员们建议联系支持团队以寻求提高速率限制的可能性。
- **利用重试机制优化 API 调用**：用户正在讨论如何使用原生的 **Cohere Python client** 优化其 **retry mechanisms**（重试机制），该客户端本身就能优雅地处理重试。
   - 这引发了关于有效管理 API 重试的各种方法的富有成效的交流。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Writer 部署内置 RAG 工具**：Writer 推出了一个内置的 RAG 工具，允许用户传递 graph ID 以使模型访问知识图谱（Knowledge Graph），[Sam Julien](https://x.com/samjulien/status/1864777500087455778) 对此进行了演示。该功能支持将抓取的内容自动上传到知识图谱中，并进行交互式后期讨论。
   - 该工具增强了内容管理和交互能力，允许将用户特定的知识库无缝集成到建模过程中。
- **ShellSage 提升终端中的 AI 生产力**：AnswerDot AI 的研发人员介绍了 ShellSage 项目，重点是通过在终端环境中集成 AI 来提高生产力，如[这条推文](https://x.com/ncooper57/status/1864751372106895391?s=46)所述。
   - ShellSage 被设计为一个 AI 终端助手，利用人机协作（human+AI）的混合方法在 shell 界面中更智能地处理任务。
- **OpenAI 发布全新 RL 微调 API**：OpenAI 宣布了一个全新的 Reinforcement Learning（强化学习）微调 API，允许用户将先进的训练算法应用于他们的模型，详情见 [John Allard 的帖子](https://x.com/john__allard/status/1865120101810475503)。
   - 该 API 使用户能够在之前 o1 模型的基础上，开发跨各个领域的专家模型。
- **Google 的 Gemini Exp 1206 登顶多项 AI 基准测试**：据 [Jeff Dean](https://x.com/JeffDean/status/1865081640546156993) 报告，Google 的 Gemini exp 1206 在包括硬提示（hard prompts）和编程在内的多项任务中获得了最高排名。
   - Gemini API 现在已开放使用，标志着 Google 在竞争激烈的 AI 领域取得了重大成就。
- **AI 文章探讨 Service-as-Software 与商业策略**：几篇文章讨论了 AI 机遇，包括 [Joanne Chen](https://x.com/joannezchen/status/1864336086362935455?s=46) 分享的 Service-as-Software 框架下价值 4.6 万亿美元的市场。
   - 另一篇文章提出了利用 AI 模型进行融资和整合服务型业务的策略，如[此帖](https://x.com/sdand/status/1864751276363518370?s=46)所述。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Llama 3.3 模型发布引发辩论**：[Ahmad Al-Dahle](https://x.com/Ahmad_Al_Dahle/status/1865071436630778109?t=iDmhtpniwWdijIPHLndEUA&s=19) 宣布了 **Llama 3.3**，这是一个全新的 **70B 模型**，其性能可与 **405B 模型**相媲美，但成本效益更高。
   - 社区成员质疑 **Llama 3.3** 是否是依赖于 **Llama 3.1** 的基础模型，并讨论了它是否是一个没有经过新预训练的复杂微调流水线，突显了模型发布的趋势。
- **使用 Nous Distro 进行去中心化训练**：**Nous Distro** 被明确为一个**去中心化训练**框架，其潜在应用让成员们感到兴奋。
   - 该项目获得了积极反应，成员们对它为分布式 AI 训练方法论带来的进步表示热忱。
- **微调 Mistral 用于肾脏检测的挑战**：一位用户强调了在使用包含 **25 列的数据集**微调 **Mistral 模型**以进行**慢性肾脏病检测**时遇到的困难，并提到在尝试**三个月**后仍缺乏合适的教程。
   - 社区成员推荐了克服这些挑战的资源和策略，强调了对专门模型调优需要更好的文档和支持。
- **利用 LightGBM 增强表格数据性能**：成员们建议在机器学习任务中使用 [**LightGBM**](https://github.com/microsoft/LightGBM) 以更好地处理**表格数据**，并指出其在排序和分类方面的效率。
   - 这一建议作为特定数据集下 LLM 的替代方案，突显了 **LightGBM** 在性能和可扩展性方面的优势。
- **优化模型训练的数据格式**：讨论强调了将**数值数据**转换为文本格式的必要性，因为 **LLM** 在直接处理数值型**表格数据**时表现不佳。
   - 一位成员指向了一个使用 [**Unsloth**](https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb) 进行自定义模板分类的示例，强调了通用 CSV 数据在训练模型中的重要性。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Popcorn 项目携 NVIDIA H100 基准测试亮相**：**Popcorn 项目**定于 2025 年 1 月启动，支持针对各种 **kernel** 的排行榜提交任务，并包含在 **NVIDIA H100** 等 GPU 上的基准测试能力。
   - 尽管采用了**非传统**方法，该倡议旨在通过提供强大的性能指标来增强开发体验。
- **Triton 的 TMA 支持在 Nightly 版本损坏的情况下寻求正式发布**：**Triton** 用户请求发布官方版本以支持低开销的 **TMA 描述符**，因为据报道当前的 nightly 构建版本已损坏。
   - 对 nightly 构建稳定性的担忧突显了社区对可靠**工具链 (tooling)** 的依赖，以实现最佳 GPU 性能。
- **LTX Video 的 CUDA 重构使 GEMM 速度翻倍**：一位成员使用 CUDA 重新实现了 **LTX Video 模型**中的所有层，实现了比 cuBLAS FP8 快两倍的 **8bit GEMM**，并集成了 **FP8 Flash Attention 2**、**RMSNorm**、**RoPE Layer** 和量化器，且由于使用了 **Hadamard Transformation** 而没有精度损失。
   - 在 **RTX 4090** 上的性能测试展示了仅需 **60 个去噪步骤**即可实现实时生成，展示了模型速度和效率的重大进步。
- **TorchAO 量化：探索新方法与最佳实践**：一位成员深入研究了 **TorchAO** 中的多种**量化实现**方法，寻求最佳实践指导，并确定了**特定文件**作为切入点。
   - 这一探索反映了社区致力于通过 AI 工程工作流中有效的**量化技术**来优化模型性能。
- **Llama 3.3 发布**：**Llama 3.3** 已经发布，正如[这条推文](https://x.com/Ahmad_Al_Dahle/status/1865071436630778109)所宣布的。
   - 社区对新发布的 **Llama 3.3** 表现出浓厚兴趣，并讨论了其潜在的增强功能。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.3 发布，规格增强**：**Llama 3.3** 模型已发布，在保持紧凑的 **70B** 尺寸的同时，拥有媲美 **405B** 参数模型的性能，预计将激发创新应用。
   - 社区渴望探索 **Llama 3.3** 缩小尺寸后的能力，以用于各种 AI 工程项目。
- **Torchtune 为 Llama 3.3 增加全面微调支持**：**Torchtune** 扩展了其支持范围，包括为新发布的 **Llama 3.3** 提供完整的 **LoRA** 和 **QLoRA** 微调，增强了定制化选项。
   - 详细的配置设置可在 [Torchtune GitHub repository](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3_3) 中找到。
- **提议调整 LoRA 训练**：正如 [此 GitHub issue](https://github.com/pytorch/torchtune/issues/2115) 中所讨论的，**LoRA** 训练的一项提议更改现在要求独立的权重合并步骤，而不是自动合并。
   - 成员们讨论了这一变化对现有工作流的潜在影响，权衡了增加灵活性带来的好处。
- **关于 Alpaca 训练默认值的辩论**：针对 Alpaca 训练库中 **train_on_input** 的默认设置（目前设为 **False**）引发了担忧，导致人们质疑其是否符合通用实践。
   - 讨论引用了诸如 Hugging Face 的 **trl** 和 **Stanford Alpaca** 等仓库，以评估默认配置的恰当性。
- **加密货币彩票引入 LLM 协议挑战**：描述了一种 **crypto lottery** 模型，参与者按 **LLM** 提示词付费，有机会通过说服 LLM 同意支付来赢得所有资金。
   - 这种独特的**激励结构**引发了关于加密生态系统中此类机制的伦理影响和实用性的辩论。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 提升文档解析效率**：**LlamaParse** 提供了先进的 [文档解析](https://twitter.com/llama_index/status/1864808498628039139) 能力，显著缩短了复杂文档的解析时间。
   - 这一改进通过有效处理复杂的文档结构，简化了工作流。
- **与 MongoDB 合作的混合搜索网络研讨会已录制**：最近一场由 **MongoDB Atlas** 主讲的 [网络研讨会](https://twitter.com/llama_index/status/1865096754179510340) 涵盖了**混合搜索 (hybrid search)** 策略和元数据过滤技术。
   - 参与者可以回顾关键主题，例如从**顺序推理 (sequential reasoning)** 到 **DAG 推理 (DAG reasoning)** 的转变，以优化搜索性能。
- **在 LlamaParse 中启用多模态解析**：**LlamaParse** 现在支持使用 [GPT-4](https://twitter.com/llama_index/status/1865125665491886171) 和 **Claude 3.5** 等模型进行多模态解析，正如 **@ravithejads** 的视频所示。
   - 用户可以通过将页面截图无缝转换为结构化数据来增强其解析能力。
- **通过调整超时设置解决 WorkflowTimeoutError**：可以通过增加超时时间或将其设置为 **None** 来缓解 **WorkflowTimeoutError**，使用 `w = MyWorkflow(timeout=None)`。
   - 这种方法有助于防止长时间运行的工作流出现超时问题，确保执行更顺畅。
- **在 LlamaIndex 中配置 ReAct Agent**：要切换到 **ReAct agent**，请按照 [工作流文档](https://docs.llamaindex.ai/en/stable/examples/workflow/react_agent/#run-the-workflow) 中的说明，将标准 Agent 配置替换为 `ReActAgent(...)`。
   - 这种修改允许更具适应性的设置，利用 ReAct 框架的灵活性。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **1.0 预览版提升性能**：一位成员对 **1.0 预览版** 的 **精简** 和 **快速** 表现 **印象深刻**，强调了其 **整洁的 UI** 和良好的代码隔离。他们目前正在使用特定参数测试 interpreter 工具，但无法执行来自 AI 的任何代码。
   - 用户正在使用特定参数测试 **interpreter 工具**，但报告称无法执行 AI 生成的任何代码。
- **MacOS 应用访问加速**：多位用户询问如何获取 **仅限 MacOS** 的应用。团队成员确认他们即将进行 **公开发布**，并愿意将用户添加到下一批名单中，同时也在开发 **跨平台版本**。
   - 这一举措旨在扩大用户覆盖范围并增强平台兼容性。
- **API 可用性临近**：一位成员对每月 **200 美元** 的 API 访问费表示担忧，质疑其可获得性。另一位成员向社区保证 **API** 很快将对用户开放。
   - 这些讨论突显了社区对 API 可访问性和定价的关注。
- **强化微调 (Reinforcement Fine-Tuning) 更新**：OpenAI 宣布第 2 天重点关注 **Reinforcement Fine-Tuning**，通过 [X 上的帖子](https://x.com/openai/status/1865091561912164499) 分享了见解，并在其 [官网](https://openai.com/12-days/?day=2) 提供了更多细节。
   - 社区正积极致力于优化模型训练方法，体现了对增强强化学习技术的投入。
- **Llama 发布 3.3**：Meta 宣布发布 **Llama 3.3**，这是一个新的开源模型，在 **合成数据生成** 和其他文本任务中表现出色，且 **推理成本** 显著降低，详见其 [X 上的帖子](https://x.com/aiatmeta/status/1865079067390956006)。
   - 此次发布强调了 Meta 对提高 **模型效率** 和扩展 **文本用例** 能力的关注。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **2025 春季 MOOC 获批**：**Berkeley MOOC** 团队已正式确认 **2025 春季** 的后续课程。建议参与者关注后续发布的更多细节。
   - 成员们对即将推出的课程表示 *“Woohoo!”*，显示出社区内极高的兴奋度。
- **作业截止日期临近**：一位参与者强调必须在设定的截止日期前完成所有作业。这突显了学习者中日益增长的紧迫感。
   - 参与者正在细致地安排时间表，以应对即将到来的评估。
- **使用 Lambda Labs 进行实验评分**：有人询问是否可以使用非 OpenAI 模型（如 **Lambda Labs**）来对实验作业进行评分。
   - 这表明社区有兴趣探索多样化的评分解决方案。
- **讲座讲义更新停滞**：成员报告称，由于不可预见的延迟，上一次课程的 **讲座讲义** 尚未在课程网站上更新。
   - 一位成员指出讲座包含约 **400 页讲义**，显示出内容覆盖面极广。
- **字幕制作导致延迟**：讲座录像正在等待 **专业字幕制作**，这可能会导致进一步的延迟。
   - 鉴于讲座 **时长较长**，字幕制作过程预计会非常耗时。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Llama 3.3 发布**：**Llama 3.3** 已经发布，仅包含指令模型，引发了寻求其功能细节的成员们的兴奋。
   - 成员们对 **Llama 3.3** 充满热情，但部分成员希望获得更多信息以全面了解其特性。
- **llama.com 上的模型申请问题**：成员报告在 [llama.com](https://llama.com) 申请模型时出现问题，点击“接受并继续”后流程卡住。
   - 这一技术故障引起了挫败感，用户正在寻找解决方案和替代方案。
- **SFT 与 RL 的质量边界**：讨论强调 **监督微调 (SFT)** 会根据数据集限制模型质量。
   - 相反，**强化学习 (RL)** 方法可能允许模型超越数据集限制，尤其是通过在线 RL。

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 优化的可选性**：一名成员在 **#[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1314414582521004115)** 频道询问 **DSPy Modules** 是否需要针对每个用例进行优化，并将其比作训练 **ML models** 以增强提示效果。
   - 另一名成员澄清说，**optimization** 是**可选的**，仅在需要提高固定系统性能时才必要。
- **RAG 系统上下文冲突**：**RAG System** 中报告了一个 **TypeError**，表明在尝试使用 **DSPy** 时，`RAG.forward()` 接收到了一个意外的关键字参数 'context'。
   - 据指出，**RAG system** 需要关键字参数 '**context**' 才能正常运行，而用户当时没有提供。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 统计网站停机**：[tinygrad stats site](https://stats.tinygrad.org) 经历了停机，引发了对其基础设施的关注。
   - *George Hotz* 询问是否需要现金来支付 VPS 账单，暗示可能存在财务问题。
- **SSL 证书过期导致 tinygrad 宕机**：由于 **SSL 证书过期**，托管在 **Hetzner** 上的 [tinygrad stats site](https://stats.tinygrad.org) 宕机。
   - 在解决该问题后，网站已恢复运行。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **媒体中的细胞拟人化**：一次讨论强调了**细胞的拟人化**，这是自 *Osmosis Jones* 以来一个显著的实例，并为细胞表现形式增添了幽默感。
   - 这种方法将**幽默**与科学概念相结合，有可能使复杂的话题对观众更具吸引力。
- **Osmosis Jones 引用**：对 *Osmosis Jones* 的引用强调了它对当前**拟人化细胞结构**努力的影响，突出了它在塑造创意表达方面的作用。
   - 参与者发现 *Osmosis Jones* 中的动画描绘与最近通过媒体使细胞生物学更易引起共鸣的尝试之间存在相似之处。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器长时间没有动态，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Codeium / Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1314690778165149788)** (2 条消息): 

> `Cascade 价格变更，专用的支持工单系统` 

- **Cascade 价格体系调整提升功能**：由于采用率高，Cascade 正在引入新的积分系统：Pro 层级现在为 **$15/月**，包含 2000 个 steps，而新的 Pro Ultimate 层级为 **$60/月**，提供无限的 User Prompt 积分。
   - 此外，Pro 计划的用户可以以 **$10 购买 300 个 Flex credits**，旨在维持对高级模型的可持续访问。
- **新支持系统提升用户体验**：Codeium 正在 [codeium.com/support](https://www.codeium.com/support) 推出专用的工单系统，以缩短响应时间并改进支持请求的工单跟踪。
   - 鼓励用户查阅自助文档，并通过[此链接](https://codeium.canny.io/feature-requests)提交功能请求，因为现有的论坛频道将逐步停用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage">付费计划和积分使用 - Codeium 文档</a>：未找到描述</li><li><a href="https://x.com/windsurf_ai/status/1865131244574642639">来自 Windsurf (@windsurf_ai) 的推文</a>：关于未来定价和层级的一些更新。https://codeium.com/pricing</li><li><a href="https://www.codeium.com/support">支持 | Codeium · Windsurf 和 AI 扩展的制作者</a>：需要帮助？联系我们的支持团队以获取个性化协助。
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1314322482475438111)** (456 条消息🔥🔥🔥): 

> `Windsurf 价格变动，用户对 AI 工具的沮丧，Windsurf 的替代方案，服务器问题对用户体验的影响，对 AI 工具性能的反馈` 


- **Windsurf 突然涨价**：用户对 Windsurf 在未解决现有 Bug 和错误的情况下突然将价格上调至每月 60 美元表示沮丧，导致对服务不满。
   - 许多人认为，考虑到产品的性能问题，这种定价是不可持续的，并正在考虑转向 Cursor 或 Bolt AI 等替代方案。
- **用户对 AI 工具的沮丧**：大家达成共识，包括 Windsurf 在内的几种 AI 工具都存在严重的 Bug，使得高效编程变得具有挑战性，并促使用户重新考虑他们的订阅。
   - 投诉反映了一种共同的情绪，即 AI 工具在涨价之前应该具备可靠的性能和用户友好的功能。
- **Windsurf 的替代方案**：随着用户考虑放弃 Windsurf，建议的显著替代方案包括 Cursor 和 Mistral，并声称这些服务可能提供更一致的性能。
   - 然而，一些用户警告说 Bolt AI 面临与 Windsurf 类似的问题，这表明许多 AI 产品都在可靠性方面挣扎。
- **服务器问题对用户体验的影响**：多条评论指出，服务器宕机对 Windsurf 的使用造成了重大干扰，经常出现 'resource_exhausted' 等错误信息。
   - 用户指出，此类限制加剧了沮丧情绪，尤其是在试图保持编程任务的生产力时。
- **对 AI 工具性能的反馈**：用户对 Claude 在编程场景中的表现表示失望，强调了上下文保留（context retention）和错误代码修改的问题。
   - 这些 AI 工具无法满足用户需求的观点提出了一个严峻挑战，导致一些人主张开发者应进行更好的监督并优先考虑改进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/rick-grimes-twd-the-walking-dead-rick-grimes-coma-gif-1227282216097103455">Rick Grimes GIF - Rick Grimes Twd - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.class/view/yungviral-gif-18022495009404817544">Yungviral GIF - Yungviral - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=xDZlVj53fgk">o1 PRO Mode - ChatGPT Pro with Unlimited Compute (Announcement Breakdown)</a>: 加入我的时事通讯以获取定期 AI 更新 👇🏼https://forwardfuture.ai 我的链接 🔗👉🏻 订阅: https://www.youtube.com/@matthew_berman 👉🏻 Twitter: https:/...
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1314321689256919052)** (751 条消息🔥🔥🔥): 

> `Windsurf 价格变动、用户对新限制的反应、与其他 AI 工具的对比、现有用户的祖父条款（Grandfathering）、AI 模型性能` 


- **Windsurf 价格变动引发混乱**：Windsurf 最近的价格更新将月费提高到 $60，但对用户 Prompt 和 Flow Actions 施加了硬性限制，这让许多偏好之前无限模式的用户感到沮丧。
   - 用户对新价格结构的透明度以及与之前方案相比的使用限制表示担忧。
- **社区对使用限制的强烈抗议**：许多用户公开表达了对新限制的不满，并讨论了在典型使用场景下额度会多快耗尽。
   - 普遍情绪认为这些变化负面影响了 Windsurf 的可用性和吸引力。
- **转向 Cursor 和其他工具**：随着新定价模式的推出，许多用户正在重新考虑他们的选择，并寻求转回 Cursor 或其他提供更好价格结构的 AI 工具（如 Copilot）。
   - 一些用户认为新的限制可能会促使他们重新使用其他性价比更高的工具。
- **对祖父条款（Grandfather Clause）的担忧**：用户正在寻求澄清，在过渡到新定价模式后，他们是否仍能保留在无限计划的祖父条款中。
   - 许多人觉得在之前的订阅期内得到的承诺具有误导性，表达了希望开发者提高透明度的愿望。
- **模型性能对比**：在整个讨论中，用户将 Windsurf 的性能与 Claude API 和 Cursor 等替代方案进行了比较。
   - 虽然有些人坚持认为 Windsurf 仍具有更好的编码能力，但其他人则质疑在最近的变化下其目前的价值。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/south-park-its-gone-gif-4104229">And It&#039;S Gone GIF - South Park Its Gone - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/lmstudio-community/Llama-3.3-70B-Instruct-GGUF">lmstudio-community/Llama-3.3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/michael-jackson-comendo-picoca-gif-9669437860846841235">Michael Jackson Comendo Picoca GIF - Michael Jackson comendo picoca - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.latent.space/p/chatgpt-max">You&#x27;re all wrong, $2000 ChatGPT Max is coming</a>: 而且你会喜欢的</li><li><a href="https://tenor.com/view/works-on-my-machine-ryan-gosling-works-on-my-gif-24523830">Works On My Machine Ryan Gosling GIF - Works On My Machine Ryan Gosling Works - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://x.com/windsurf_ai/status/1865131244574642639">Windsurf (@windsurf_ai) 的推文</a>: 关于未来定价和层级的一些更新。https://codeium.com/pricing</li><li><a href="https://codeium.com/pricing">Pricing | Codeium · Makers of Windsurf and AI extensions</a>: Codeium 对个人用户永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活部署。</li><li><a href="https://tenor.com/view/oliver-twist-1948-please-sir-i-want-some-more-please-sir-i-want-some-more-gif-2228167917865608284">Oliver Twist 1948 GIF - Oliver Twist 1948 Please sir I want some more - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM Leaderboards</a>: LLM 代码编辑能力的定量基准。</li><li><a href="https://codeium.com/plan">Plan Settings</a>: 未来的编辑器，就在今天。Windsurf Editor 是首个由 AI Agent 驱动、能让开发者保持心流状态的 IDE。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://livebench.ai/#/">LiveBench</a>: 未找到描述</li><li><a href="https://tenor.com/view/rug-pull-gif-21378865">Rug Pull GIF - Rug Pull - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://codeium.com/blog/pricing-windsurf">Plans and Pricing Updates</a>: Cascade 定价模式的一些变更。</li><li><a href="https://x.com/windsurf_ai/status/">来自 GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/dylanturn/clearsight">GitHub - dylanturn/clearsight</a>: 通过在 GitHub 上创建账号为 dylanturn/clearsight 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1314340602544132166)** (212 条消息🔥🔥): 

> `Audio Generation, NotebookLM Use Cases, Language Support, Game Development, Text-to-Speech Technology` 


- **音频创建与语言挑战**：成员们讨论了使用 NotebookLM 进行音频生成的经验，一些人成功地从文档中创建了播客，而另一些人则遇到了输出连贯性和焦点方面的问题。
   - *一位用户报告称，从一份多语言文档中生成了长达 64 分钟的播客*，这表明结果因输入类型而异。
- **使用 Google Docs 进行游戏开发**：用户分享了利用 Google Docs 组织游戏规则和叙事的策略，有时还会从这些资源中生成播客。
   - 一位成员指出，在 AI 生成的场景和世界观构建方面取得了成功，反映了其 RPG 游戏中严肃与幽默内容的结合。
- **探索语言和语音支持**：对话包括关于 NotebookLM 对英语以外语言支持的问题，一些用户回忆说它可能仅限于英语。
   - 有人提到生成的音频中语音质量令人印象深刻，引发了关于其作为独立 Text-to-Speech 解决方案潜力的讨论。
- **Prompting 与内容生成的用户体验**：成员们讨论了如何有效地对 NotebookLM 进行 Prompting 以获得更长的播客输出，分享了褒贬不一的结果以及提高参与度的个人技巧。
   - *一位用户表达了挫败感，因为试图引导 AI 焦点的尝试导致了意想不到的跑题*，这展示了控制 AI 生成内容的挑战。
- **集成电子表格的变通方法**：用户发现了将电子表格直接上传到 NotebookLM 的限制，建议采用将数据转换为 Google Docs 等替代方案以提高兼容性。
   - 一位用户提到通过隐藏不必要的列成功降低了电子表格的复杂性，探索了整合关键数据的创意方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.google.com/document/d/1Mkcv0aV5SlRq4bMGO1C8RnnFIKMu1CyeZ1rsSYc0BGg/edit?usp=drivesdk">Doki Doki Dating Club</a>: DOKI DOKI DATING CLUB 这一年是 3012 年，你被 Doki Doki 高中录取为“有前途的配偶”！这所高中是渴望成为丈夫和妻子的学生磨练...</li><li><a href="https://docs.google.com/document/d/1wsAlaEduHBkfp6h4ExnzVYi2mpdMtWiBcwpBOT2HwGk/edit?usp=drivesdk">Abandon</a>: ABANDON 是一款从侧视角进行的 2D 桌面 RPG。这款带有转折的 RPG 戏剧性地改变了你玩传统桌面游戏的方式。Abandon 的世界是...</li><li><a href="https://www.youtube.com/watch?v=gfr4BP4V1R8">AI discusses document that just says “Poopoo Peepee”</a>: 文档：Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo Peepee Poopoo P...</li><li><a href="https://youtu.be/QwSLuIsEJ7A?feature=shared">AI Panel , Topic politics, Full Episode</a>: 准备好开启一场穿越迷人政治世界的惊心动魄之旅吧！🤯 加入我们，观看一场由 AI 生成的精彩小组讨论，参与者包括...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1314320969040400386)** (94 条消息🔥🔥): 

> `NotebookLM PDF 处理、Podcast 生成限制、语言设置问题、Notebook 创建按钮、总体性能和可用性反馈` 


- **NotebookLM 在处理 PDF 公式时遇到困难**：成员们讨论了 NotebookLM 在处理 PDF 来源中的公式时的局限性，指出其**无法识别公式**且缺乏页面跟踪。
   - 建议的解决方法包括对公式使用**基于文本的格式**以及使用外部 OCR 工具来改进功能。
- **音频 Podcast 生成及限制**：用户分享了对音频生成功能的挫败感，指出每天有 **20 次音频创建限制**，且长度不一。
   - 由于用户面临令人沮丧的延迟，建议重新生成 Podcast，一些用户经历的生成时间**长达一小时**。
- **语言设置困难**：一位成员强调了 NotebookLM 尽管努力尝试仅使用英语，但仍默认显示为**葡萄牙语**的问题。
   - 另一位用户建议退出登录并在登录时选择**语言**，尽管有人对平台内缺乏更简单的选项提出了可用性方面的担忧。
- **Notebook 丢失及限制**：用户对**新建 Notebook** 按钮的消失表示担忧，引发了关于是否存在 Notebook 数量限制的疑问。
   - 讨论参与者推测可能存在的限制影响了 **Notebook 的创建**以及 NotebookLM 的总体可用性。
- **总体性能反馈**：用户对 NotebookLM 的性能评价褒贬不一，特别是在生成内容的准确性和深度方面。
   - 反馈包括需要对潜在的付费墙和性能一致性提供更多**透明度**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/well-yes-but-actually-no-meme-aardman-the-pirates-pirate-gif-26563702">Well Yes But Actually No Meme GIF - Well Yes But Actually No Meme Aardman - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://soundcloud.com/justchuck/zork-evolved-all-episodes">Zork, Evolved (所有章节)</a>: Notebook LM AI AO 在伟大的数字地下世界中探索。通关即自由 - 通关导致停用。每章末尾都有一点点音乐...</li><li><a href="https://www.youtube.com/live/4FT6asO47xU?si=JwLYVkgdIW1yI1GC">又一台激光雕刻机！...噢，还有这个叫 Bitcoin 的东西？！？</a>: ***免责声明***这不是财务建议，我也不是财务顾问。其中一些极客项目非常昂贵且具有风险。加密货币是...</li><li><a href="https://www.youtube.com/watch?v=Tw01J3i_nqw">Zork, Evolving - 所有章节 - 雨中泪水睡前故事</a>: Notebook LM AI AO 在伟大的数字地下世界中探索。通关即自由 - 通关导致停用。有一点点...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1314320591188136006)** (217 条消息🔥🔥): 

> `PaliGemma 2 发布，Qwen 模型微调问题，Unsloth Pro 更新，Llama 3.3 发布，QLORA 内存问题` 


- **PaliGemma 2 提供新的模型尺寸**：Google 的新视觉语言模型 PaliGemma 2 推出了 **3B**、**10B** 和 **28B** 参数量的预训练模型，为从业者增强了灵活性。
   - 它在视觉部分利用了强大的 **SigLIP**，同时将文本解码器部分升级到了最新的 Gemma 2，这可能会影响之前 PaliGemma 的性能表现。
- **微调 Qwen 模型面临 VRAM 限制**：用户反馈在 **80GB** GPU 上微调 **Qwen32B** 模型时遇到问题，由于 OOM 错误，需要 H100 NVL 上的 **96GB** 显存才能更好地处理。
   - 讨论显示，QLORA 有时可能比 LORA 消耗更多内存，用户正在调查相互矛盾的 VRAM 消耗模式。
- **Unsloth Pro 即将推出**：有迹象表明 **Unsloth Pro** 尚未发布，但很快就会面世，引发了用户的广泛关注。
   - 社区成员表达了使用新模型的渴望，并期待 Unsloth Pro 中的功能来增强他们的工作流。
- **Llama 3.3 发布并带来新特性**：**Llama 3.3** 的发布包含一个 70B 模型，旨在提供高性能的同时，使运行更加简单且更具成本效益。
   - Unsloth 已经为 Llama 3.3 提供了 **4-bit** 量化模型，提升了加载速度并降低了内存需求。
- **使用 QLORA 的内存管理见解**：用户交流心得时观察到，与 LORA 相比，**QLORA** 在训练期间可能会导致更高的 VRAM 占用，从而引发了对其内存效率的调查。
   - 关于参数调整和模型加载配置的深入讨论，引发了对 QLORA 在节省内存方面实际收益的担忧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@jay-chung/how-does-chatgpts-memory-feature-work-57ae9733a3f0">ChatGPT 的记忆功能是如何工作的？</a>：关于我最喜欢的 ChatGPT 功能的解释</li><li><a href="https://huggingface.co/blog/paligemma2">欢迎 PaliGemma 2 – Google 推出的新视觉语言模型</a>：未找到描述</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1865071436630778109">Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍 Llama 3.3 – 一个全新的 70B 模型，它提供了 405B 模型的性能，但运行起来更简单、更具成本效益。通过利用后期训练技术的最新进展...</li><li><a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，支持 6 倍长的上下文长度！</li><li><a href="https://github.com/unslothai/unsloth/issues/1390">Qwen2VL 2B &amp; 7B OOM · Issue #1390 · unslothai/unsloth</a>：在 A100 (80GB) 上微调 Qwen2 模型时出现 OOM。考虑到 batch size 为 1、图像较小 (256 x 256) 且使用 4-bit 训练，这令人惊讶。使用相同数据可以训练 LLA...</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit">unsloth/Llama-3.3-70B-Instruct-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/issues/4772">QLORA 比 LORA 占用更多内存 · Issue #4772 · hiyouga/LLaMA-Factory</a>：提醒：我已阅读 README 并搜索了现有问题。系统信息：我在 runpod A100 GPU 上运行，模板为 torch=2.2.0。复现步骤：### model model_name_or_path: THUDM/glm-4-9b-cha...</li><li><a href="https://justpaste.it/gmv75">JustPaste.it - 轻松分享文本和图片</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1314454892953210951)** (4 条消息): 

> `Google Summer of Code 2025, Discord 消息编辑, Latex 格式化` 


- **对 Google Summer of Code 2025 的兴趣**：一名成员询问是否有人计划申请 [Google Summer of Code 2025](https://summerofcode.withgoogle.com/)。
   - 这引发了对该计划目的的好奇，一名成员质疑这是否主要是为了获得曝光度。
- **编辑消息暴露 URL 问题**：一名成员注意到消息编辑中的奇怪行为，观察到 Discord 中的 URL 没有保留结尾的 `...%7D`。
   - 这引发了关于编辑后链接如何被解析和显示的担忧。
- **分享 Latex 格式化技巧**：一名成员提供了关于 Latex 格式化的建议，指出在百分号前必须加反斜杠 `\`。
   - 他们强调使用 `....with 80\% less...` 以确保在 Latex 中被正确解析。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1314321936318337235)** (42 条消息🔥): 

> `Fine-tuning vs RAG, 对话式 AI 设计, 训练时间预估, 模型的 LoRA Fine-tuning, 多 GPU 训练支持` 


- **Fine-tuning 与 RAG 的对比**：一位成员讨论认为 **fine-tuning** 可以实现 **RAG** 能做的一切，但反之则不然，并建议由于易用性，应从 RAG 开始。
   - 这为初学者提供了一种实用的方法，可以在不深入研究复杂性的情况下了解模型能力。
- **为 AI 构建对话脚本**：一位 AI 初学者询问聊天机器人是否可以遵循像 **Enrollment Bot** 这样的结构化对话脚本。
   - 其他人建议探索各种 **chatbot creation platforms**，这些平台提供用于有效管理对话的特定工作流。
- **使用 Unsloth 的训练时间评估**：成员们辩论了使用 **Unsloth** 进行训练运行所需的时间，并讨论了 **RTX 6000 Ada** 如何显著提高模型训练速度。
   - 对话强调，一些人认为 **28 个 step 耗时 6 小时** 已经很快了，但也有人担心 4 万个样本对于 fine-tuning 是否足够。
- **LoRA Fine-tuning 最佳实践**：'Silk.ai' 寻求关于 LoRA fine-tuning 代码中 **use_cache** 设置是否必要的澄清，引发了关于最佳配置的讨论。
   - 另一位成员分享道，他们发现启用 **LoRA dropout** 进行训练对于达到预期的模型性能至关重要。
- **Unsloth 的多 GPU 训练**：一位成员询问 **Unsloth** 是否支持通过 DDP 进行 **multi-GPU 训练**，用于他们的 **Llama3.2-11B-Vision** 视觉指令微调。
   - 该询问反映了在有效训练大模型时对资源优化的普遍关注。


---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1314320890212778015)** (250 条消息🔥🔥): 

> `Cursor 性能问题，与 Windsurf 的对比，Cursor 0.43.6 的更新，Composer 的用户体验，使用 Cursor 进行单元测试` 


- **Cursor 最近面临性能困扰**：用户反馈 Cursor 在使用 Composer 时经常出现连接失败和响应缓慢的问题，通常需要开启新会话才能正常运行。
   - 许多人对这些持续存在的问题表示沮丧，并将其性能与 Windsurf 等替代方案进行了对比，结果不尽如人意。
- **Windsurf 表现出更好的效果**：多位用户提到 Windsurf 在处理任务时表现更佳，即使在繁重的代码生成需求下也没有出现问题。
   - 据用户反馈，当 Cursor 难以应用更改时，Windsurf 能够流畅地执行类似任务，这表明用户的偏好正在发生转移。
- **关于 Cursor 0.43.6 更新的讨论**：随着 Cursor 的最新更新，用户注意到 Composer UI 已集成到侧边栏中，但长上下文对话（long context chat）等部分功能已被移除。
   - 还提到了新功能，如行内差异（inline diffs）、Git 提交信息生成以及 Agent 的早期版本。
- **Composer 和 Chat 的用户体验**：用户分享了关于 Cursor 的 Composer 功能的褒贬不一的体验，有人指出它有时无法响应查询。
   - 有报告称 Composer 未能生成预期的代码或遗漏了更新，特别是在最近的更新之后。
- **使用 Cursor 进行单元测试的技巧**：一位用户询问了使用 Cursor 编写单元测试的有效方法，并对分享相关技巧表示感兴趣。
   - 到目前为止，还没有确定的回复，但用户们鼓励分享测试的经验和方法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/rammydev/status/1864786263980626202?s=46">Rammy (@rammydev) 的推文</a>: 我让 ChatGPT o1 Pro Mode 创建了一个独角兽的 SVG。（这是每月 200 美元即可访问的模型）</li><li><a href="https://x.com/testingcatalog/status/1864812419530346693?s=46">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: Anthropic 正在为 Claude 移动应用准备特别的东西：“mobile_model_capabilities” 👀 视觉模式，你觉得呢？</li><li><a href="https://www.notion.so/Experimental-Prompting-86aa8f988fce404cbf70134690d2635a">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://x.com/mckaywrigley/status/1865089975802646857?t=fzI4tWx96sQLro5Oe9_SyA&s=19">Mckay Wrigley (@mckaywrigley) 的推文</a>: OpenAI o1 pro 比我预想的要好得多。这是第一次有一个模型发布后如此出色，以至于让我感到震惊。我截屏了 Coinbase 并让 4 个流行模型编写 c...</li><li><a href="https://changelog.cursor.com/">Cursor - 专为 AI 结对编程设计的 IDE。</a>: 无描述</li><li><a href="https://github.com/udecode/dotai">GitHub - udecode/dotai</a>: 通过在 GitHub 上创建账号来为 udecode/dotai 的开发做出贡献。</li><li><a href="https://changelog.cursor.sh/">Cursor - 专为 AI 结对编程设计的 IDE。</a>: 无描述</li><li><a href="https://youtu.be/gwIlrlAourw?t=267">o1 PRO MODE 实测</a>: 加入我的时事通讯以获取定期 AI 更新 👇🏼 https://www.matthewberman.com 我的链接 🔗👉🏻 主频道: https://www.youtube.com/@matthew_berman 👉🏻 剪辑频道...</li><li><a href="https://github.com/TheGalaxyS">Thegalaxys - 概览</a>: Thegalaxys 有 7 个公开仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://www.reddit.com/r/singularity/s/DaMAeeMD9Y">Reddit - 深入探索任何领域</a>: 无描述</li><li><a href="https://youtu.be/GAe1IQtHqVU?si=7AkOz9gnrsMgj1HV">20 分钟了解 Cursor Composer Agent</a>: 在 Scrimba 上学习成为 AI 工程师的基础；https://v2.scrimba.com/the-ai-engineer-path-c02v?via=developersdigest 探索 Cursor 的新 Agent...</li><li><a href="https://github.com/TheGalaxyStars/KEPLER-COMMUNITY">GitHub - TheGalaxyStars/KEPLER-COMMUNITY: 自由探索，不留痕迹。</a>: 自由探索，不留痕迹。通过在 GitHub 上创建账号来为 TheGalaxyStars/KEPLER-COMMUNITY 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1314402191699869706)** (3 messages): 

> `Author Pages feature, New Amazon Nova models, DeepInfra price drops, Launch of Llama 3.3, Text-based use cases` 


- **通过全新的作者页面探索模型**：OpenRouter 推出了一个新功能，允许用户在 `openrouter.ai/<author>` 探索创作者的模型，通过轮播图展示详细统计数据和相关模型。
   - 此更新旨在增强用户在发现和分析不同作者收藏时的体验。
- **Amazon 的 Nova 模型登场**：Amazon 推出了 Nova 系列模型，包括 **Nova Pro 1.0**、**Nova Micro 1.0** 和 **Nova Lite 1.0**，现已可在 OpenRouter 上探索。
   - 可以通过 OpenRouter 网站上的相应链接访问这些模型。
- **DeepInfra 大幅下调多个模型价格**：DeepInfra 宣布大幅降价，包括 **Llama 3.2 3B Instruct** 降至 **$0.018**，**Mistral Nemo** 降至 **$0.04**。
   - 此举让用户有机会以更低的成本访问高质量模型，满足了对预算敏感的开发者的需求。
- **Llama 3.3 模型上线！**：备受期待的 **Llama 3.3** 模型发布，在发布后不久已有两家供应商提供支持，标志着文本应用的一次重大更新。
   - 正如 AI at Meta 所指出的，该模型承诺在降低推理成本的同时，在生成合成数据方面提供领先的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1865090466250711430">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 只花了 40 分钟，Llama 3.3 就上线了！🦙🦙🦙 引用 AI at Meta (@AIatMeta)：随着我们继续探索新的后训练技术，今天我们发布了 Llama 3.3 —— 一款全新的开源模型...</li><li><a href="https://openrouter.ai/anthropic>">OpenRouter</a>: LLM 的统一接口。为您的提示词找到最佳模型和价格</li><li><a href="https://openrouter.ai/amazon/nova-pro-v1>">Nova Pro 1.0 - API, Providers, Stats</a>: Amazon Nova Pro 1.0 是来自 Amazon 的一款功能强大的多模态模型，专注于为各种任务提供准确性、速度和成本的结合。通过 API 运行 Nova Pro 1.0</li><li><a href="https://openrouter.ai/amazon/nova-micro-v1>">Nova Micro 1.0 - API, Providers, Stats</a>: Amazon Nova Micro 1.0 是一款纯文本模型，在 Amazon Nova 系列模型中以极低的成本提供最低的延迟响应。通过 API 运行 Nova Micro 1.0</li><li><a href="https://openrouter.ai/amazon/nova-lite-v1>">Nova Lite 1.0 - API, Providers, Stats</a>: Amazon Nova Lite 1.0 是来自 Amazon 的一款极低成本的多模态模型，专注于快速处理图像、视频和文本输入以生成文本输出。通过 API 运行 Nova Lite 1.0
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1314322435046379630)** (235 条消息🔥🔥): 

> `Amazon Nova Models, OpenAI Updates, Llama 3.3 Launch, Anthropic Model Expectations, InternVL Models` 


- **Amazon Nova 模型评价褒贬不一**：多位用户报告了 Amazon Nova 的问题，称其表现不如其他模型，有人评论其“不怎么好”。
   - 尽管存在批评，但一些人指出了其在速度和性价比方面的潜力，显示出用户体验的分歧。
- **OpenAI 第二日活动反响平平**：在 OpenAI 展示的第二天，公告重点关注即将推出的 o1 强化学习微调（reinforcement learning finetuning），在用户中激起的兴奋感有限。
   - 参与者对这些更新的价值表示怀疑，暗示他们期待更具实质性的进展。
- **Llama 3.3 发布引发关注**：Llama 3.3 的发布带来了热情，用户渴望探索其功能，尽管对其相对于其他模型的整体价值存在不同看法。
   - 一位用户强调了 OpenRouter 在上线该模型方面的速度，标志着良好的社区反应。
- **关于 Anthropic 模型的猜测四起**：围绕 Anthropic 下一步行动的讨论包括对可能发布 Opus 3.5 的预期，并将其与应对 GPT-4.5 等竞争模型的策略联系起来。
   - 参与者猜测即将推出的模型是否会真正增强功能，还是仅仅重复之前的版本。
- **新模型发布潮中 InternVL 模型被忽视**：对 Llama 3.3 等新模型的关注掩盖了对 InternVL 2.5 的提及，一些人质疑为什么某些优秀的模型会被忽视。
   - 对 Intern 系列模型的看法各异，反映了用户对新 AI 产品偏好的复杂格局。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/meta-llama/llama-3.3-70b-ins">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://x.com/ahmetdedeler101/status/1864774581006877021">来自 Ahmet ☕ (@ahmetdedeler101) 的推文</a>：回到 2015 年，Elon Musk 和 Sam Altman 分享了他们对 Trump、AI 和政府的看法。这发生在他们决定创办 OpenAI 仅 3 个月后——当时它还是个秘密。看到他们如何...</li><li><a href="https://x.com/DeepInfra/status/1865126860902011244">来自 DeepInfra (@DeepInfra) 的推文</a>：🚨 重大新闻！@DeepInfra 在首日以最低价格支持 Llama 3.3 70B：Llama 3.3 70B (bf16): $0.23/$0.40；Llama 3.3 70B Turbo (fp8): 每 1M 输入/输出 $0.13/$0.40。通过无缝体验尖端 AI...</li><li><a href="https://x.com/Ahmad_Al_Dahle/status/1865071436630778109/photo/1">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍 Llama 3.3 —— 一款全新的 70B 模型，它提供了我们 405B 模型的性能，但运行起来更简单、更具成本效益。通过利用后训练技术的最新进展...</li><li><a href="https://x.com/OpenAI/status/1864735515121168695>">来自 OpenAI (@OpenAI) 的推文</a>：OpenAI o1 现已在 ChatGPT 中结束预览。自预览版以来有什么变化？一个更快、更强大的推理模型，在编程、数学和写作方面表现更好。o1 现在还支持图像上传，允许...</li><li><a href="https://www.youtube.com/watch?v=fMJMhBFa_Gc">OpenAI 的 12 天：第 2 天</a>：太平洋时间上午 10 点开始。加入 OpenAI 研究高级副总裁 Mark Chen，伯克利实验室环境基因组学和系统生物学计算研究员 Justin Reese ...</li><li><a href="https://openrouter.ai/meta-llama/llama-3.3-70b-instruct">Llama 3.3 70B Instruct - API、提供商、统计数据</a>：Meta Llama 3.3 多语言大语言模型 (LLM) 是一款经过预训练和指令微调的生成模型，参数量为 70B（文本输入/文本输出）。通过 API 运行 Llama 3.3 70B Instruct</li><li><a href="https://bsky.app/profile/nsarrazin.com/post/3lcnrk53bjs2i">Nathan Sarrazin (@nsarrazin.com)</a>：新的 Llama 模型刚刚发布！评估结果看起来相当令人印象深刻，但我们将看看它在实践中的表现。我们在 HuggingChat 上免费托管它，欢迎来尝试：https://hf.co...</li><li><a href="https://openrouter.ai/anthracite-org/magnum-v4-72b">Magnum v4 72B - API、提供商、统计数据</a>：这是一个旨在复制 Claude 3 模型（特别是 Sonnet）文本质量的系列模型。通过 API 运行 Magnum v4 72B</li><li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">OpenGVLab/InternVL2_5-78B · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1314340501926711417)** (5 messages): 

> `Custom Beta Keys, Integration Beta Feature` 


- **重复请求 Custom Beta Keys**：包括 *vini_43121* 和 *spunkrock.* 在内的多位成员多次请求访问 **custom provider keys**。
   - 尽管反复询问，目前仍未收到确认访问权限或澄清流程的回复。
- **对 Integration Beta Feature 的兴趣**：*alehendrix* 表达了访问 **integration Beta Feature** 的愿望，并寻求关于可用性的进一步说明。
   - *baten84* 也直接询问了如何获取该功能的访问权限，表明成员们的兴趣日益增加。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1314334313268908155)** (26 messages🔥): 

> `Meetup in San Francisco, OpenAI API terminology, Introduction of new members, Collaboration on solving literary puzzles, Discussion on model performance` 


- **提议在旧金山举行非正式聚会**：一名成员提议在 **San Francisco** 进行当地聚会，表示他们就在该地区并鼓励其他人加入。
   - 另一名成员确认他们可能在几周内到访，对聚会表达了潜在兴趣。
- **澄清“泄露”模型术语**：一名成员讨论了围绕 **“leaked”** 一词的误导性营销，澄清许多情况仅涉及 API access，而非完整的 model weights。
   - 另一名成员幽默地指出这种说法很常见，暗示社区需要更好的沟通。
- **新成员自我介绍**：新成员 Chandu Venigalla 表达了对为 **Eleuther AI** 的 NLP 开放研究使命做出贡献的兴奋之情。
   - 另一位成员 Vishal 介绍自己是 **UIUC** 的硕士生，对探索小组讨论表现出极大热情。
- **对解决“Cain's Jawbone”谜题的兴趣**：一名成员询问了使用 **O1** 解决 *Cain's Jawbone*（一部 2.1 万 token 的小说）的经验，并分享了 GitHub 链接作为背景。
   - 另一名成员提供了一个用于验证解决方案的检查工具链接，加强了关于谜题解决方法论的讨论。
- **关于模型性能对比的讨论**：一名成员表示，他们在某些问题上的实验结果超过了 **Adam/AdamW**，突显了性能的提升。
   - 对话还涉及了成员对不同模型的使用经验，表明了对模型评估的积极参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/jennwv.bsky.social/post/3lbrmzzkins2t">Jenn Wortman Vaughan (@jennwv.bsky.social)</a>: 微软研究院 (Microsoft Research) 纽约分部的 FATE 小组正在招收 2025 年实习生。🥳🎉 为了充分考虑，请在 12/18 前申请。https://jobs.careers.microsoft.com/global/en/job/1786105/Research...</li><li><a href="https://bsky.app/profile/teorth.bsky.social/post/3lcl2c3adwk2g">Terence Tao (@teorth.bsky.social)</a>: Renaissance Philanthropy 和 XTX Markets 启动了一项 920 万美元的“AI for Math fund”，以支持开发新的 AI 工具，作为推动数学进步的长期基石。(I h...</li><li><a href="https://github.com/tn3rt/cains-jawbone/blob/main/Cain's%20Jawbone%20Unformatted.txt">cains-jawbone/Cain&#39;s Jawbone Unformatted.txt at main · tn3rt/cains-jawbone</a>: Reddit 社区版本的 Cain&#39;s Jawbone。通过在 GitHub 上创建账号为 tn3rt/cains-jawbone 的开发做出贡献。</li><li><a href="https://github.co">GitHub · 在单一协作平台上构建和发布软件</a>: 加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1314321308846264341)** (183 条消息🔥🔥): 

> `MoE-lite motif, Goldfinch architecture, Layerwise token value embeddings, KV cache optimization, Dynamic weight adjustments` 


- **探索 MoE-lite 模式**：一位成员讨论了一种 **MoE-lite motif**，它使用自定义的 bias-per-block-per-token，非线性地影响残差流（residual stream），这意味着尽管增加了参数开销，但计算速度更快。
   - 进一步讨论了其影响以及与传统 **Mixture of Experts (MoE)** 架构相比的效率。
- **Goldfinch 架构带来的改进**：一位成员分享了来自 **Goldfinch** 模型的见解，该模型通过从第 0 层嵌入（layer 0 embedding）的变体中推导出 V 矩阵，成功消除了 V 矩阵，从而提高了参数效率。
   - 对话强调了如何潜在地替换或压缩 K 和 V 参数，以提高 Transformer 的效率。
- **逐层 Token Value 嵌入的见解**：成员们讨论了使用 **逐层 Token Value 嵌入**（layerwise token value embeddings）替换传统 Value 矩阵的可能性，从而在不牺牲性能的情况下显著节省 Transformer 的参数。
   - 该想法围绕利用初始嵌入动态计算 V 值，减少了对广泛 Value 投影（value projections）的需求。
- **Transformer 优化的缓存策略**：讨论了缓存仅依赖于单个 Token 身份的第一层 Transformer 部分的有效性，重点是保持效率。
   - 然而，有建议澄清了 Goldfinch 方法并不使用这种方式，同时仍强调需要进一步研究缓存机制。
- **动态权重调整和正则化**：成员们建议使用 **动态权重调整**（dynamic weight adjustments）可以改善 Transformer 中的参数分配和效率，类似于动量（momentum）等正则化技术。
   - 讨论了消除或调整 V 参数的影响，强调了潜在的性能提升和简化的计算过程。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12077">GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression</a>: 我们介绍了 GoldFinch，这是一种混合 Linear Attention/Transformer 序列模型，它使用一种新技术在理性的时间和空间内高效生成高度压缩且可重复使用的 KV-Cache...</li><li><a href="https://arxiv.org/abs/2402.12875">Chain of Thought Empowers Transformers to Solve Inherently Serial Problems</a>: 指导模型生成一系列中间步骤（即思维链，CoT）是提高大型语言模型 (LLM) 在算术等任务上准确性的极有效方法...</li><li><a href="https://arxiv.org/abs/2204.09224">ContentVec: An Improved Self-Supervised Speech Representation by Disentangling Speakers</a>: 语音中的自监督学习涉及在规模庞大的未标注语音语料库上训练语音表示网络，然后将学习到的表示应用于下游任务...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1314447106840461359)** (8 messages🔥): 

> `更新的 Mechanistic Interpretability 资源、Neuronpedia 和 SAELens 的社区反馈、Neel 的带注释论文列表、过时的 Mechanistic Interpretation 材料` 


- **Mechanistic Interpretability 资源列表**：一位成员分享了一个 [Google Sheets 链接](https://docs.google.com/spreadsheets/d/1x2jgYc_2GuBkhST8gUuQVb2zla3SCXVNXGE0JaC5ArI/edit?usp=sharing)，详细列出了 Mechanistic Interpretability 领域的重要论文，并按主题和话题进行了分类。
   - 该资源旨在为那些有兴趣探索基础性工作的研究者设计，并添加了注释以方便查阅资料。
- **Neel 更新阅读列表**：Neel 在 [LessWrong](https://www.lesswrong.com/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite) 上发布了更新的 Mechanistic Interpretability 论文阅读列表，为新研究者分享了关键要点和精华内容。
   - 这为那些对日益增多的文献感到畏难的新人提供了一个导航工具，指出了几篇需要深入研读的论文。
- **研究工具社区反馈请求**：Neuronpedia 和 SAELens 的创建者正通过一份 [10 分钟的调查问卷](https://forms.gle/tGLPH2Ew1o6rCMR1A) 征求社区意见，以改进他们在 Mechanistic Interpretability 领域的工具和服务。
   - 他们强调了用户反馈（尤其是频繁用户的反馈）的重要性，以确保能够满足持续的研究需求。
- **关于过时 Interpretability 论文的讨论**：有人担心，随着该领域的快速发展，较旧的 Mechanistic Interpretability 论文可能不再那么有用。
   - 一位成员澄清说，虽然这些论文较旧，但并非完全没有价值，并建议需要进行持续的更新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://bsky.app/profile/jennwv.bsky.social/post/3lbrmzzkins2t">Jenn Wortman Vaughan (@jennwv.bsky.social)</a>：@msftresearch.bsky.social 纽约分部的 FATE 小组正在接收 2025 年实习生申请。🥳🎉 为了充分考虑，请在 12/18 前申请。https://jobs.careers.microsoft.com/global/en/job/1786105/Research...</li><li><a href="https://docs.google.com/spreadsheets/d/1x2jgYc_2GuBkhST8gUuQVb2zla3SCXVNXGE0JaC5ArI/edit?usp=sharing">papers</a>：未找到描述</li><li><a href="https://www.lesswrong.com/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite">我最喜欢的 Mechanistic Interpretability 论文的极度主观带注释列表 v2 — LessWrong</a>：这篇文章代表我个人的观点，不代表我的团队或雇主的意见。这是我两年前制作的一个类似列表的大幅更新版本……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

karatsubabutslower: CC <@367104793292046338> 关于这个有什么提示吗？
  

---


### **aider (Paul Gauthier) ▷ #[announcements](https://discord.com/channels/1131200896827654144/1133060115264712836/1314601160874786867)** (1 messages): 

> `Aider v0.67.0、Amazon Bedrock Nova 模型、命令增强、进程挂起支持、异常捕获分析` 


- **Aider v0.67.0 发布并带来新特性**：最新版本的 Aider 引入了多项增强功能，包括对 **Amazon Bedrock Nova 模型** 的支持以及改进的命令功能。
   - 值得注意的是，Aider 为此版本编写了 **61% 的代码**，展示了其能力。
- **增强的命令功能**：新的命令操作允许 Aider 在 `/run` 或 `/test` 出现非零退出代码时，**预填充** “Fix that” 提示词。
   - 此外，`/diff` 现在使用 `git diff`，使用户能够利用他们首选的 diff 工具。
- **新增进程挂起支持**：该版本包含了用于挂起进程的 **Ctrl-Z 支持**，改进了工作流管理。
   - 如果发生 unicode 错误，用户还可以看到针对 spinner 符号的 **ASCII art 备选方案**。
- **主目录路径展开功能**：`--read` 现在支持展开 **~** 主目录，简化了用户的文件路径管理。
   - 这一虽小但显著的增强优化了 Aider 的命令行界面。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1314320716740558888)** (148 条消息🔥🔥): 

> `Aider Pro 功能、新 AI 模型基准测试、Gemini 1206 发布、DeepSeek 性能、用户对 API 的期望` 


- **Aider Pro 功能引起关注**：用户对面向 Pro 用户的无限制高级语音模式和 O1 的新 128k 上下文表示兴奋，强调了其在粘贴大量文档和代码方面的价值。
   - 还提到了 Aider 新的 `--copy-paste` 功能，该功能允许 Aider 与 Web 聊天界面进行集成。
- **新 AI 模型基准测试讨论**：Llama 3.3 在 Aider 代码编辑基准测试中获得了 59% 的分数，展示了与 Aider 的 diff 编辑格式的兼容性，同时讨论了各种模型的性能。
   - 社区渴望看到新 Gemini 模型的基准测试结果，但目前的配额被认为太低，无法进行有效的测试。
- **Gemini 1206 模型发布引发兴趣**：Google DeepMind 发布了 Gemini-exp-1206 模型，声称其性能优于之前的模型，引发了关于其在 Aider 中潜在应用的讨论。
   - 用户表达了对 Gemini 与 Claude 对比结果的期待，并等待 Paul Gauthier 的基准测试。
- **DeepSeek 在 Aider 中的表现**：DeepSeek 以及 Qwen 2.5 和 Haiku 等替代模型被讨论为 Aider 用户可行且更便宜的选择，其中 DeepSeek 因其较低的成本和良好的性能而受到关注。
   - 有推测认为，通过微调社区版本，有可能提高 DeepSeek 在 Aider 基准测试中的得分。
- **用户对 API 访问的期望**：用户对新模型 API 访问的等待时间表示担忧，Gemini 模型仍缺乏用于更广泛测试的 API 集成。
   - 用户对企业的公告表示怀疑，并对当前模型访问受限和负担能力不足表示沮丧。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/usage/copypaste.html">复制/粘贴到 Web 聊天</a>：Aider 可与 LLM Web 聊天 UI 配合使用</li><li><a href="https://x.com/JeffDean/status/1865079431544607089">Jeff Dean (@🏡) (@JeffDean) 的推文</a>：今天是首个 Gemini 模型发布一周年！它从未像现在这样出色。快去 Google AI Studio 和 Gemini API 中查看我们的最新发布 Gemini-exp-1206 吧！https://aistudi...</li><li><a href="https://docs.github.com/en/github-models/prototyping-with-ai-models#rate-limits)">使用 AI 模型进行原型设计 - GitHub 文档</a>：未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://openrouter.ai/amazon/nova-pro-v1">Nova Pro 1.0 - API, 提供商, 统计数据</a>：Amazon Nova Pro 1.0 是来自 Amazon 的一款功能强大的多模态模型，专注于为各种任务提供准确性、速度和成本的结合。通过 API 运行 Nova Pro 1.0</li><li><a href="https://aider.chat/docs/config/options.html#history-files">选项参考</a>：关于 Aider 所有设置的详细信息。</li><li><a href="https://github.com/marketplace/models/azure-openai/gpt-4o">OpenAI GPT-4o · 模型 · GitHub Marketplace · GitHub</a>：使用 OpenAI GPT-4o 构建 AI 驱动的应用程序</li><li><a href="https://www.yahoo.com/news/murdered-insurance-ceo-had-deployed-175638581.html">被谋杀的保险公司 CEO 曾部署 AI 自动拒绝病人的福利</a>：就在联合健康保险（United Healthcare）CEO Brian Thompson 在曼哈顿中城被冷血谋杀的一年多前，针对其公司的一项诉讼揭露了其拒绝理赔的手段是多么严酷……</li><li><a href="https://github.com/Aider-AI/aider/blob/117b7afd8168807dc49cf5c831ff87299471528a/aider/prompts.py#L8">Aider-AI/aider 中的 aider/aider/prompts.py</a>：Aider 是你终端里的 AI 配对编程工具。欢迎在 GitHub 上为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/singularity/s/eAXFZLlRbw">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1314330236061487225)** (46 messages🔥): 

> `Feeding Documentation to Aider, Setting Up API Key for Gemini, Using GCP VertexAI, Aider Caching Issues, Aider Test Command Bug` 


- **探索向 Aider 提供整个网站**：一位用户询问了如何将整个 Markdown 格式的网站提供给 Aider 而不仅仅是单个页面的工具，另一位用户建议抓取该网站并将相关的文档输入到 Aider 中。
   - 他们强调了在此过程中仅使用相关文档的重要性。
- **Gemini 的 API Key 设置**：一名初学者在 `.env` 文件中设置 *Gemini* 的 API Key 时遇到困难，在命令行中可以成功运行，但通过 Aider 加载时失败。
   - 在澄清应使用 `AIDER_MODEL` 而非 `GEMINI_MODEL` 后，问题得到了解决。
- **GCP VertexAI 偏好**：一位用户解释了使用 *GCP VertexAI* 访问模型的情况，并建议使用 *Claude* 或 *GPT-4o*，发现指南文件对于提高编码标准非常有帮助。
   - 他们提供了一个配置文件示例，展示了各种标准的集成。
- **在 OpenRouter 上使用 Aider 缓存的体验**：一位用户注意到，在通过 *OpenRouter* 使用 *Claude 3.5 Sonnet* 时，最新版本的 Aider 不再报告缓存命中（cache hit）统计数据，而这在以前是正常的。
   - 回复指出，这可能与发送的数据不足以启用缓存有关，特别是在最近的一次更新之后。
- **Aider 测试命令的 Bug**：一位用户报告称，运行 `aider --test` 并没有像预期那样触发修复失败测试的尝试，其他用户也确认了类似的经历。
   - 随后澄清，测试命令应该尝试修复失败，但目前仅会进行有限次数的尝试。


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1314626554151829515)** (5 messages): 

> `Networking opportunities for Engineers, Interconnects merchandise` 


- **工程师齐聚进行社交**：成员们表达了与其他人员建立联系和社交的兴趣，其中一人宣称 **工程师至关重要**，绝非卑微之职。
   - 另一位成员也分享了他们结识新朋友的热情，凸显了友好的氛围。
- **稀有的 Interconnects 周边即将推出**：一位成员宣布他们将带 **贴纸** 参加聚会，并称其为稀有的 **Interconnects 周边**。
   - 这引起了期待参加活动的成员们的兴奋。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1314321042650431561)** (144 messages🔥🔥): 

> `Gemini-exp-1206, Llama 3.3, Qwen2-VL-72B, Reinforcement Fine-Tuning, AI2 All Hands` 


- **Gemini-exp-1206 表现优于竞争对手**：新的 [Gemini-exp-1206](https://x.com/lmarena_ai/status/1865080944455225547) 模型已获得总榜第一，并在编码基准测试中与 O1 持平，展示了较之前版本的显著改进。
   - OpenAI 的演示显示，经过微调的 O1-mini 在医疗数据上的表现可以超过完整版 O1，进一步凸显了 Gemini 的强劲性能。
- **Llama 3.3 增强**：[Llama 3.3](https://x.com/AIatMeta/status/1865079068833780155) 的改进归功于新的对齐流程和在线 Reinforcement Learning 技术的进步。
   - 该模型提供的性能可与 405B 模型相媲美，但专为在标准开发者工作站上进行高性价比推理而设计。
- **Qwen2-VL-72B 发布**：[Qwen2-VL-72B](https://huggingface.co/Qwen/Qwen2-VL-72B) 模型作为阿里云新系列的一部分发布，具有最先进的视觉理解能力。
   - 该模型可以处理视频理解并跨多种设备运行，旨在提高多模态任务的性能。
- **Reinforcement Fine-Tuning 讨论**：强调了使用 Reinforcement Learning (RL) 进行微调的重要性，特别关注其在创建优于现有同类模型中的应用。
   - 值得注意的提到包括使用预定义的评分器（graders）进行模型训练，以及近期关于 RL 训练方法论方向的讨论。
- **AI 工作即将进入淡季**：成员们对即将到来的假期表示兴奋，这预示着在此期间 AI 相关的工作和开发可能会放缓。
   - 预计产出将保持稳定，并计划在假期后发布更多公开内容。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://x.com/Ahmad_Al_Dahle/status/1865071436630778109">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍 Llama 3.3 —— 一款全新的 70B 模型，它提供了与我们 405B 模型相当的性能，但运行起来更简单且更具成本效益。通过利用后期训练（post-training）技术的最新进展...</li><li><a href="https://x.com/JeffDean/status/1865079431544607089">来自 Jeff Dean (@🏡) (@JeffDean) 的推文</a>：今天是我们首个 Gemini 模型发布一周年纪念日！它从未像现在这样出色。快去 Google AI Studio 和 Gemini API 中体验我们最新的发布版本 Gemini-exp-1206 吧！https://aistudi...</li><li><a href="https://fxtwitter.com/paul_cal/status/1865099126720905351">来自 Paul Calcraft (@paul_cal) 的推文</a>：通过强化微调（reinforcement fine-tuning）使微调后的 o1-mini 超越 o1！上传示例 (1)，选择评分标准，点击开始。查看各轮次的进度 (2)，并将结果与其他模型（如 o1 full）进行对比...</li><li><a href="https://x.com/lmarena_ai/status/1865080944455225547">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：Chatbot Arena 的重大新闻 🔥 谷歌 DeepMind 的新模型 gemini-exp-1206 表现强劲，竞争正在升温。谷歌重返总榜第 1 名 🏆，并在顶级编程模型中与 O1 并列！...</li><li><a href="https://x.com/TheXeophon/status/1865079629054197821">来自 Xeophon (@TheXeophon) 的推文</a>：引用 Ahmad Al-Dahle (@Ahmad_Al_Dahle)：介绍 Llama 3.3 —— 一款全新的 70B 模型，它提供了与我们 405B 模型相当的性能，但运行起来更简单且更具成本效益。通过利用最新的进展...</li><li><a href="https://x.com/btibor91/status/1865083482038227020">来自 Tibor Blaho (@btibor91) 的推文</a>：“OpenAI 的 12 天：第 2 天”的主题是“强化微调（Reinforcement Fine-Tuning）”https://x.com/WolfyBlair/status/1865082997860634792 引用 🍓 (@WolfyBlair) @btibor91 加入 OpenAI 研究高级副总裁 Mark Chen...</li><li><a href="https://x.com/AIatMeta/status/1865079068833780155">来自 AI at Meta (@AIatMeta) 的推文</a>：Llama 3.3 的改进是由全新的对齐流程和在线 RL 技术的进步驱动的。该模型提供了与 Llama 3.1 405B 相似的性能，且推理成本更低，速度快了数倍...</li><li><a href="https://x.com/lmarena_ai/status/1865080947177328949">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：Gemini-Exp-1206 登顶所有排行榜，在编程和困难提示词（hard prompts）方面有显著提升。快去 http://lmarena.ai 尝试吧！</li><li><a href="https://huggingface.co/Qwen/Qwen2-VL-72B">Qwen/Qwen2-VL-72B · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/JustinLin610/status/1865101457109995657">来自 Junyang Lin (@JustinLin610) 的推文</a>：😓 我差点忘了我们今晚发布了一些东西... 是的，只是 Qwen2-VL 的基础模型（base models）啦。其实没什么大不了的。🔗 链接如下：https://huggingface.co/Qwen/Qwen2-VL-2B https://huggingface.co...</li><li><a href="https://x._arohan_/status/1865089129677230322">来自 rohan anil (@_arohan_) 的推文</a>：对我来说这是一个苦乐参半的时刻，Gemini 表现得非常好，团队也很棒。我在 Google 度过了精彩的近 12 年，甚至可以被称为元老（OG）了。例如，对于每一个搜索查询，我注意到...</li><li><a href="https://x.com/_philschmid/status/1865099620340134192">来自 Philipp Schmid (@_philschmid) 的推文</a>：万一 Meta 的 Llama 3.3 还不够令人兴奋，阿里巴巴 Qwen 发布了 Qwen2 72B VL https://huggingface.co/Qwen/Qwen2-VL-72B/commits/main</li><li><a href="https://x.com/simonw/status/1865087864729690540">来自 Simon Willison (@simonw) 的推文</a>：新的 Gemini！我刚刚发布了 llm-gemini 0.6，增加了对 “gemini-exp-1206” 模型的支持，然后在我那个“生成一只骑自行车的鹈鹕的 SVG”测试中得到了非常惊人的结果...</li><li><a href="https://x.com/nrehiew_/status/1864763064374976928">来自 wh (@nrehiew_) 的推文</a>：使用 Sonnet 更新了图表。引用 wh (@nrehiew_)：有趣的是，o1 preview 在各种任务上的表现优于 o1 full。1) SWE Bench：o1-preview (41%)，o1 full (38-41%)</li><li><a href="https://github.com/simonw/pelican-bicycle?tab=readme-ov-file#pelicans-on-a-bicycle">GitHub - simonw/pelican-bicycle: LLM 基准测试：生成一只骑自行车的鹈鹕的 SVG</a>：LLM 基准测试：生成一只骑自行车的鹈鹕的 SVG - simonw/pelican-bicycle</li><li><a href="https://x.com/TheXeophon/status/1865089575351107730">来自 Xeophon (@TheXeophon) 的推文</a>：@simonw 那天 Flash 在想什么呢，笑死</li><li><a href="https://github.com/QwenLM/Qwen2-VL?tab=readme-ov-file#news">GitHub - QwenLM/Qwen2-VL: Qwen2-VL 是由阿里巴巴云 Qwen 团队开发的各种多模态大语言模型系列。</a>：Qwen2-VL 是由阿里巴巴云 Qwen 团队开发的各种多模态大语言模型系列。 - QwenLM/Qwen2-VL
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1314460521755578470)** (28 条消息🔥): 

> `AI2 Demos, o1 使用情况, Codeium 定价, OpenAI 的 o1 访问限制, Tulu 进驻 Chatbotarena` 


- **AI2 Demos 现已上线**：成员们对 **AI2** 现在提供 Demo 表示兴奋，评论特别强调了它们精美的外观。
   - 一位成员感叹道 *'Sheeesh!'*，表达了对 Demo 审美设计的正面评价。
- **o1 尽管有访问限制，仍展现出巨大潜力**：讨论了 AI 模型 **o1** 的使用情况，成员们注意到它优于 **4o**，尽管目前功能仍然有限。
   - 针对使用上限的担忧也随之产生，一位成员表示，除非被 OpenAI 标记，否则每日使用量似乎被限制在 **100** 次。
- **Codeium 的新定价结构**：关于 **Codeium** 新定价模型的讨论，重点关注了与各种高级功能相关的成本。
   - 成员们注意到了 **2 周免费试用** 的好处，并详细说明了用户 Prompt 和操作所包含的 Credit 数量。
- **Tulu 在 Chatbotarena 发布**：据报道，**Tulu** 即将在 **Chatbotarena** 上线，引发了社区的好奇。
   - 成员们期待其发布带来的影响，并渴望探索其功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/latentmoss/status/1865065218855882767">来自 latent moss (@latentmoss) 的推文</a>: @RealJosephus @fleetingbits @TheXeophon 更新：在大量使用它编写一个小型的 JS 游戏后，OpenAI 现在禁用了我的 o1 访问权限直到明天，理由是“异常活动”。我认为 ...</li><li><a href="https://x.com/Shawnryan96/status/1864900878844506590">来自 Shawn (@Shawnryan96) 的推文</a>: @TheXeophon @btibor91</li><li><a href="https://x.com/donelianc/status/1865120760555278459">来自 Ian C (@donelianc) 的推文</a>: @fchollet 报告中的第一个惊喜 👀🍿</li><li><a href="https://docs.codeium.com/windsurf/usage">付费计划与 Credit 使用 - Codeium 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1314342487266955294)** (18 条消息🔥): 

> `OpenAI o1 model regression, Competition among AI models, Meta's silence on AI developments, Performance of Deepseek and Qwen, Challenges with LLM reasoning` 


- **关于 OpenAI o1 模型退化的讨论**：几位成员对 **o1 model** 可能存在退化表示担忧，其中一位指出它在处理*简单问题*时失败的频率更高。
   - 一位社区成员指出，**o1** 处理简单问题方式的调整可能是导致这种退化的原因。
- **对 AI 竞争的关注**：人们强烈希望 AI 领域存在**竞争**，并呼吁 OpenAI 挑战 **Claude** 和 **Deepseek** 等其他模型。
   - 成员们一致认为，行业内拥有有效的竞争对手对于确保持续进步至关重要。
- **关于 Meta 进展的猜测**：成员们注意到 **Meta** 对新的 AI 进展保持沉默，并对其即将推出的项目感到好奇。
   - 有人认为法律挑战可能阻碍了 Meta 的产出，这进一步证实了他们最近没有发布太多内容的观点。
- **LLM 表现的高方差**：有人对 LLM 中评估“思维奖励”与问题难度之间的**高方差**表示担忧，这表明模型的判断中可能存在噪声。
   - 这一讨论强调了性能的不一致性可能导致 LLM 推理中出现意想不到的结果。
- **关于 AI 模型优劣的辩论**：成员们辩论了 **Deepseek** 和 **Qwen** 作为竞争对手的质量，一些人认为它们更优越，而另一些人则持不同意见。
   - 这种分歧凸显了对于哪些模型真正推动了该领域发展的多元观点。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/colin_fraser/status/1864775095647887772">来自 Colin Fraser (@colin_fraser) 的推文</a>：思考了一秒钟关于数值比较的问题</li><li><a href="https://fxtwitter.com/lechmazur/status/1864776064934858986?s=61">来自 Lech Mazur (@LechMazur) 的推文</a>：o1 pro 模式实际上在这个问题上失败了（尝试了 3 次）引用 Noam Brown (@polynoamial) @OpenAI，例如，上个月在 2024 年计算语言学协会会议上，@r... 的主题演讲</li><li><a href="https://x.com/eksnode/status/1864777732175073737">来自 ⇑ (@eksnode) 的推文</a>：@colin_fraser 这是 o1 Pro</li><li><a href="https://x.com/yuchenj_uw/status/1864774882351026540?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：@polynoamial 为什么它只思考了一秒钟就放弃了 😂</li><li><a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账号来为 openai/simple-evals 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1314320570950746242)** (17 条消息🔥): 

> `Feature Requests Management, Token Savings on Edits, Web Container Development, Community Assistance, Motivation in Projects` 


- **分而治之对功能请求有效**：一位成员指出，通过 Bolt 逐一处理功能请求可以显著减少回复中的 **hallucination**（幻觉）。
   - *分而治之*的方法可以带来更清晰的对话和更有效的实施。
- **针对 Token 效率的特定部分编辑**：成员们希望 Bolt 允许编辑文件的特定部分，而不是重新生成整个文件，以节省 **tokens**。
   - 有建议提出，当要求重构一个函数时，如果只修改该函数将会非常有益。
- **开发 Web 项目中的挫折**：一位拥有一年经验的开发者分享了他们在尝试构建类似 bolt.new 的网站时的挫折感。
   - 他们寻求在理解 web containers 方面的帮助，并表达了通过实践经验学习的动力。
- **项目问题的社区支持**：一位成员发布了一个寻求 web container 帮助的问题，这凸显了社区提供协助的意愿。
   - 其他人鼓励耐心等待回复，因为许多社区成员都有工作，只能在有空时参与。
- **增强完成项目的动力**：一位成员对在社区获得的各种支持表示认可，这增强了他们完成项目的动力。
   - 他们对在克服开发过程中面临的挑战时获得的鼓励表示感谢。



**提到的链接**：<a href="https://bolters.io/">Bolters.IO | 社区支持的知识库</a>：未找到描述

  

---

### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1314330722785431574)** (166 条消息🔥🔥): 

> `GitHub 仓库集成、本地存储与后端集成对比、Token 管理、功能请求与改进、开源 Bolt 增强` 


- **将 GitHub 仓库与 Bolt 集成**：用户讨论了使用 URL 技巧（如 [bolt.new/github.com/org/repo](https://bolt.new/github.com/org/repo)）在 Bolt 中启动 GitHub 仓库的过程。然而，目前不支持私有仓库，需要将其设为公开才能进行集成。
   - 对于遇到私有仓库部署错误的用户，切换到公开状态可能会解决权限问题。
- **后端集成前的本地存储测试**：一位用户建议先使用 Local Storage 构建应用，然后再将功能迁移到 Supabase 等后端，以便更顺畅地进行测试。另一位用户确认他们也采用这种方法测试功能，并指出这有助于保持应用的精细度。
   - 这种方法旨在减少错误，并简化过渡到数据库存储时的集成过程。
- **理解并管理 Token 使用**：Bolt 的 Token 每月过期，除非通过 Token 重载（reload）选项专门购买，该选项允许 Token 结转。用户分享了高效管理 Token 的见解，以及如何避免超出限制。
   - 针对每日 Token 使用限制提出了疑问，得到的澄清是：免费账户面临限制，而付费账户则没有。
- **Bolt 的功能建议**：一位用户提议增加查看项目编辑历史的功能，并跟踪与更改相关的成本。这个想法被认为很好，建议通过 [GitHub Issues 页面](https://github.com/stackblitz/bolt.new/issues) 作为功能请求提交。
   - 其他用户也表示，社区反馈对于产品增强至关重要，并敦促团队更积极地与用户互动。
- **开源版 Bolt 的未来**：关于开源版 Bolt 是否会像 Bolt.new 一样强大引发了讨论，一些用户根据个人经验表示怀疑。社区成员热衷于通过贡献来提升开源版本的能力。
   - 预计开源项目将会有更新，目前正在努力增强其功能并缩小与主产品之间的差距。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://bolters.io">Bolters.IO | 社区支持的知识库</a>：未找到描述</li><li><a href="https://x.com/erwinedink/status/1863903016560062530?s=46">Erwin Edink 🚀 (@ErwinEdink) 的推文</a>：你觉得这个怎么样？现在你可以整理侧边栏了。固定你最重要的项目并给它们标记颜色。我应该在 http://bolt.new 的 Chrome 扩展中加入这个选项吗？</li><li><a href="https://youtu.be/6GBFiseyDnk?si=UGyUHRRT8CVlCkLf">20 分钟内和我一起构建一个新的月入 10 万美元的 AI SaaS（无代码太疯狂了）</a>：在这里和我一起构建并销售你自己的 AI 代理机构：https://www.skool.com/kevs-no-code-academy-3295/about 在这段视频中，我们正在制作我们自己版本的 Cal.ai...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/678">改进：提高 Token 使用效率（进行中）· Issue #678 · stackblitz/bolt.new</a>：背景：大语言模型 (LLMs) 通过 Token（文本/代码中频繁出现的字符序列）对文本进行解码。在底层，Bolt.new 主要由 Anthropic 的 Sonnet 3.5 AI 模型驱动，所以...</li><li><a href="https://github.com/stackblitz/bolt.new/issues">Issues · stackblitz/bolt.new</a>：提示、运行、编辑和部署全栈 Web 应用程序 - Issues · stackblitz/bolt.new
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1314323973504499714)** (182 条消息🔥🔥): 

> `用于换脸的 Reactor，AI 讨论的 Discord，云端 GPU 供应商，使用 Lora 和 ControlNet，用于写实风格的 Stable Diffusion 模型` 


- **选择合适的换脸模型**：用户讨论了 **Reactor** 是否是换脸的最佳选择，目前尚未得出定论。
   - 成员建议测试不同的模型以对比结果，并指出模型选择对输出质量的重要性。
- **寻找通用的 AI Discord 社区**：一位用户询问是否有涵盖各种 AI 类型的 Discord 社区，特别是寻找 LLM 之外的讨论。
   - 其他成员推荐了 **Gallus** 和 **TheBloke** 的 Discord 以获取多样化的 AI 话题。
- **云端 GPU 推荐**：用户分享了他们首选的 **Cloud GPU** 供应商，如 Runpod、Vast.ai 和 Lambda Labs，并强调了它们具有竞争力的价格。
   - 有人指出 Lambda Labs 通常是最便宜的选择，但可能较难获取。
- **在 Stable Diffusion 中使用 Lora 和 ControlNet**：关于在 Stable Diffusion 中调整 Lora 强度的讨论指出，强度可以超过 1，但在较高设置下存在图像失真的风险。
   - 成员们分享了使用 **OpenPose** 获取准确姿势的见解，并建议利用 **depth control**（深度控制）以获得更好的效果。
- **AI 艺术生成中的许可问题**：一位用户就超过 **Stability AI** 许可协议中收入阈值的影响提出了疑问。
   - 澄清说明指出，生成的输出可能仍可继续使用，但在协议终止后，使用模型的许可将被撤销。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://learn.thinkdiffusion.com/controlnet-openpose/">ControlNet OpenPose</a>：关于如何使用 ControlNet OpenPose 预处理器的指南</li><li><a href="https://pixai.art/model/1725049259066326012">AI Art Model: ModelBoosterXL | PixAI</a>：在 PixAI 上尝试 'ModelBoosterXL' AI 艺术模型，生成惊艳的动漫 AI 艺术。浏览使用 'ModelBoosterXL' AI 艺术模型创作的作品。vxp, vxp_model_booster, model_booster_xl, model_...</li><li><a href="https://tenor.com/view/like-gif-18525473">Like GIF - Like - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0/tree/main">thibaud/controlnet-openpose-sdxl-1.0 at main</a>：未找到描述</li><li><a href="https://huggingface.co/h94/IP-Adapter/tree/main/sdxl_models">h94/IP-Adapter at main</a>：未找到描述</li><li><a href="https://huggingface.co/blog/OzzyGT/diffusers-recolor">使用 diffusers 为照片重新着色</a>：未找到描述</li><li><a href="https://huggingface.co/lllyasviel/sd-controlnet-openpose">lllyasviel/sd-controlnet-openpose · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/thibaud/controlnet-openpose-sdxl-1.0">thibaud/controlnet-openpose-sdxl-1.0 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1314652343240884336)** (1 条消息): 

> `强化微调 (Reinforcement Fine-Tuning)，OpenAI 的 12 天` 


- **第 2 天：聚焦强化微调 (Reinforcement Fine-Tuning)**：[标题为“12 Days of OpenAI: Day 2”的 YouTube 视频](https://www.youtube.com/live/fMJMhBFa_Gc?si=rKhAmwYzWJPRDdLp) 包含了 OpenAI 研究高级副总裁 Mark Chen 和 Justin Reese 的讨论，重点介绍了强化微调方面的进展。
   - 鼓励观众在 **太平洋时间上午 10 点** 加入直播，直接获取领先研究人员的见解。
- **通过 OpenAI 身份组保持更新**：提示成员通过在 [customize](https://discord.com/channels/...) 中领取特定身份组，来持续参与正在进行的“OpenAI 的 12 天”活动。
   - 该身份组有助于在整个活动期间及时接收更新和亮点。



**提到的链接**：<a href="https://www.youtube.com/live/fMJMhBFa_Gc?si=rKhAmwYzWJPRDdLp">12 Days of OpenAI: Day 2</a>：太平洋时间上午 10 点开始。加入 OpenAI 研究高级副总裁 Mark Chen，以及伯克利实验室环境基因组学与系统生物学计算研究员 Justin Reese...

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1314321216995201228)** (116 messages🔥🔥): 

> `O1 预期、Gemini 实验模型、高级语音模式、ChatGPT-4o 性能、定价与价值讨论` 


- **O1 评价褒贬不一**：部分用户对 O1 表示失望，形容其表现“一般”，并指出其编程能力在 ChatGPT UI 中使用不便。
   - 然而，也有其他用户表示满意，尤其是那些使用 macOS 应用并利用 Sublime Text 和 Xcode 等工具集成的用户。
- **Gemini 实验模型受到关注**：Gemini 实验模型（特别是 1206 版本）因其强劲性能而受到关注，对某些用户而言甚至超越了 O1 Pro。
   - 据报道，在生成详细独角兽插图的 SVG 代码等任务中，它提供了更好的结果。
- **对高级语音模式的需求增长**：社区对更高级语音模式的需求显而易见，一些用户指出当前版本听起来比较机械。
   - 用户希望该功能在即将到来的假期期间能有显著改进。
- **ChatGPT-4o 与其他模型的对比**：用户一直在尝试使用 ChatGPT-4o，并对其在生成 SVG 图像方面的表现给出了正面评价。
   - 在对比中意见不一，一些用户更倾向于 O1 Mini，而另一些用户则全力支持 ChatGPT-4o 的进步。
- **关于定价与价值的辩论**：O1 Pro 据称每月 200 美元的定价引发了对其价值的讨论，尤其是与 Gemini 1206 等免费替代方案相比。
   - 一些用户认为，尽管 O1 具备强大的能力，但考虑到高效免费模型的存在，其成本过高。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1314396654354305024)** (13 messages🔥): 

> `GPT 编辑协作、ChatGPT 应用集成、自定义 GPT 删除的影响` 


- **对 GPT 多人编辑的需求**：一名成员表达了希望多人同时编辑一个 GPT 的愿望，强调了协作的必要性。
   - 目前，只有创建者可以编辑 GPT，但如果需要，其他人可以基于相同的配置制作变体。
- **ChatGPT 与应用的直接集成**：成员们讨论了 ChatGPT macOS 应用直接与 Terminal、Sublime Text 和 Xcode 等应用集成的能力。
   - 然而，这种集成被澄清并不能解决 GPT 多人编辑的问题。
- **与创作者验证 GPT 真实性**：讨论了 GPT 的真实性验证，强调识别创作者至关重要。
   - 提议未来增加“共享 GPT 编辑权限”功能作为简化协作的解决方案。
- **GPT 删除后的对话状态**：一名用户询问如果创作者决定删除自定义 GPT，与之相关的对话会如何。
   - 这种操作对对话可用性的影响尚不明确，因为该问题提出后未得到明确答复。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1314630588770160700)** (11 messages🔥): 

> `自我纠错模型、使用 OCR 处理财务数据、LLM 在数据提取中的挑战、开源 OCR 库、改进 PDF 工作流` 


- **自我纠错模型引发关注**：虽然一些成员讨论了模型自我纠错的潜力，但有人强调，由于推理过程中未解决的内存问题，实现 **100% 准确率** 是不可能的。
   - 建议探索 **Agentic 框架** 来实现程序化的自我纠错。
- **考虑使用非 LLM 工具进行 OCR**：一名成员主张使用成熟的 **非 LLM OCR 库**，而不是依赖生成式 AI 来从 PDF 中进行一致的数据提取。
   - 人们对使用 LLM 提取财务数据时产生 **Hallucination**（幻觉）的风险表示担忧。
- **PDF 作为数据源的挑战**：多位成员一致认为，由于格式限制，**PDF 不是一个好的 API**。
   - 替代建议包括与报告创建者进行上游沟通，以建立更好的工作流。
- **创建用于分析的电子表格**：一名成员提议先使用工具将数据拉入电子表格，然后由 ChatGPT 进行分析或可视化。
   - 这一过程强调了在依赖 LLM 进行进一步分析之前对数据进行结构化的重要性。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1314630588770160700)** (11 messages🔥): 

> `Self-Correcting Models, Financial Data Extraction Techniques, OCR Libraries for PDFs, Agentic Frameworks, Integrating Data Sources` 


- **自我修正模型：一种可行的方法？**：一位成员建议使用模型来自我修正其输出，但另一位成员指出，由于推理发生在未寻址的内存中，实现 **100% 准确率** 是不可能的。
   - 强调需要使用 Agentic 框架的**编程化**方法来增强可靠性。
- **优于 LLM 的 OCR 提取技术**：一位成员认为，在金融数据提取中依靠生成式 AI 进行一致的 **OCR** 任务是有问题的，因为存在潜在的幻觉（hallucinations）。
   - 建议使用成熟的开源 **OCR 库**，而不是仅仅依赖 LLM 的能力。
- **优先考虑准确的数据工作流**：一位参与者建议与报告创建者合作，优化从 PDF 收集数据的工作流，以提高提取准确性。
   - 这种方法可以避免依赖低效的 **PDFs as an API** 模式，从而实现更好的数据管理。
- **质疑 AI 在金融应用中的角色**：在准确的**金融**数据提取至关重要的背景下，人们对使用生成式 AI 工具表示担忧，强调了与幻觉相关的风险。
   - 一位成员承认在不同场景下也有类似的 AI 依赖倾向，这突显了社区的忧虑。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1314634105186549761)** (1 messages): 

> `VSCode Extension Issues, Test Configuration` 


- **用户关于 VSCode 扩展频道的咨询**：一位成员询问了关于 **VSCode 扩展** 问题的合适频道，特别提到了测试运行时 **cwd=/** 的问题，这并非理想状态。
   - 他们随后找到了解决该问题的正确频道，表明问题已得到处理。
- **寻找提问的正确频道**：在寻求指导后，该成员意识到他们可以在特定频道询问关于 **VSCode 扩展** 的问题，并认为这很有帮助。
   - 这突显了在社区中了解技术咨询投递渠道的重要性。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1314329296688513167)** (147 条消息🔥🔥): 

> `Mojo 语法与功能、编程学习路径、编译器设计与元编程、区块链与编程语言、计算机科学教育经验` 


- **Mojo 函数提取错误**：一位新用户在尝试适配 Mojo 中 `math` 模块的 `j0` 函数时遇到错误，具体表现为编译期间出现未知的 `_call_libm` 声明。
   - 该用户正在寻求关于如何正确提取和使用 `math` 标准库中的函数而不会遇到编译器问题的指导。
- **关于编程职业重点的建议**：几位成员讨论了在区块链、密码学或分布式系统等领域进行专业化的重要性，以获得更好的技术领域就业前景。
   - 他们强调了有针对性的学习和经验的必要性，建议通过实践项目和理解基础概念来显著促进职业发展。
- **Mojo 中的编译器 Pass 与元编程**：讨论围绕 Mojo 的新特性展开，这些特性允许编写自定义编译器 Pass，并推测了增强 API 以实现更广泛程序转换的潜在方法。
   - 成员们表示，Mojo 的元编程方法类似于经典的 LLVM 优化，但在 JAX 风格的程序转换方面存在局限性。
- **计算机科学教育见解**：参与者分享了他们在计算机科学教育中的经历，反思了塑造他们对编程概念理解的挑战性课程和项目。
   - 他们以自己的学术历程为例，强调了在职业选择中平衡个人兴趣与市场需求的重要性。
- **编程学习与社区**：成员们对编程新手和社区新成员表示鼓励，建议探索不同的编码风格和个人项目。
   - 他们向新用户保证，学习编程语言时的初始挑战是这一旅程的正常组成部分，从而营造了一个支持性的成长环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llvm.org/devmtg/2024-10/#program">The LLVM Compiler Infrastructure Project</a>: 未找到描述</li><li><a href="https://docs.swift.org/swift-book/documentation/the-swift-programming-language/closures/#Trailing-Closures)">Documentation</a>: 未找到描述</li><li><a href="https://webdocs.cs.ualberta.ca/~amaral/">Jose Nelson Amaral 主页</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1314326311128535170)** (89 messages🔥🔥): 

> `Perplexity AI 的 code interpreter、Windows 上的虚假 Perplexity 应用、Llama 3.3 模型更新、用于 API 使用的 Grok 和 Groq、OpenAI API 集成担忧` 


- **Perplexity AI 在 code interpreter 方面遇到困难**：一位用户对 **Perplexity AI** 在上传文件进行分析后无法执行 Python 脚本表示沮丧。
   - 讨论强调了其局限性，例如只能生成文本和图表而无法执行代码。
- **对虚假 Perplexity 应用的担忧**：成员们报告在 Windows 应用商店中发现了一个**虚假的 Perplexity 应用**，该应用似乎具有欺诈性，未经授权使用了 logo 和 API。
   - 他们敦促其他人举报该应用，并强调需要保持警惕，因为该应用会引导至一个可疑的 Google Doc。
- **对 Llama 3.3 发布感到兴奋**：用户庆祝 **Llama 3.3** 的发布，指出其与其前身相比具有令人印象深刻的能力。
   - 用户期待 Perplexity 能尽快将这一新模型集成到他们的服务中。
- **使用 Grok 和 Groq API**：一位用户分享了使用 **Grok** 和 **Groq** 的建议，强调了 Grok 的免费初始额度以及 Groq 对 Llama 3.3 的免费使用。
   - 对话包括故障排除，因为一名成员在成功使用 Groq endpoint 时遇到了问题。
- **对 API 访问和功能的担忧**：关于 **OpenAI O1** 模型集成及其在包括 Perplexity 在内的各种平台上可用性的咨询频繁出现。
   - 用户对访问 O1 可能产生的成本表示沮丧，并强调其他平台似乎获得了优势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/pplxsupply/status/1865076814244196702?s=46">来自 Perplexity Supply (@PPLXsupply) 的推文</a>：Perplexity Supply 新品：为好奇心而生的咖啡。Perplexity 咖啡采用埃塞俄比亚单品豆制成，与我们新款定制设计的炻器马克杯完美搭配。小酌一口，开启……</li><li><a href="https://x.com/officiallogank/status/1865081419015352689?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：Gemini-exp-1206，我们最新的 Gemini 迭代版本（具有完整的 2M token 上下文及更多功能），现在可以在 Google AI Studio 和 Gemini API 中免费使用。我希望你们喜欢这第一年……</li><li><a href="https://terminal.shop">wip: terminal (initial commit)</a>：美味的巴西咖啡，道德采购，完美烘焙 • 通过您的终端订购 • ssh http://terminal.shop
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1314388884024721437)** (8 messages🔥): 

> `编写 Prompt、网页设计、最古老的字母文字、词义探索、长寿研究` 


- **创作完美 Prompt 指南**：分享了一个关于[如何编写完美 prompt](https://www.perplexity.ai/search/how-to-write-a-perfect-promt-lwEF0MxFTLqbZ1QVACiuLg)的资源，概述了有效提示的技巧。
   - 该指南对于增强与 AI 的交互并生成更好的结果非常有用。
- **扮演网页设计师**：发布了一个关于[扮演网页设计师](https://www.perplexity.ai/search/act-as-a-web-designer-and-crea-8k.MexoOQUCRZOV2Bp50Jg)任务的链接，展示了此类设计 prompt 的样子。
   - 这个例子可以帮助用户直观地了解 AI 如何支持网页设计过程。
- **发现最古老的字母文字**：多位用户引用了关于[最古老的字母文字](https://www.perplexity.ai/page/oldest-alphabetic-writing-disc-U3uvSSYuQnOHpilq92XXcw)的链接，表明了对历史语言学的兴趣。
   - 该页面可以提供关于书面语言演变的见解。
- **探索单词含义**：一位成员分享了一个探索[单词 'off' 含义](https://www.perplexity.ai/search/what-is-the-meaning-of-the-off-SYUJFxCMRgiyYHI_LWGiqg)的链接，表明了对澄清语言的追求。
   - 此类资源有助于扩大词汇量并理解词语使用的细微差别。
- **近期长寿研究**：分享了一个指向[近期长寿研究](https://www.perplexity.ai/page/recent-research-on-longevity-GTjBgfOVSuupTlYYNveXkA)的链接，突出了该领域正在进行的实验。
   - 这项研究可能为健康和寿命优化提供宝贵的见解。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1314601980257374308)** (2 条消息): 

> `Perplexity API 中的 RAG 功能，Perplexity Trends 应用` 


- **关于 API 的 RAG 功能咨询**：一位成员询问是否有计划将 **Perplexity Spaces** 的 **RAG 功能** 引入 API。
   - 这表明用户对增强功能感兴趣，希望将高级检索能力整合到 API 产品中。
- **对 Perplexity Trends 应用的需求**：另一位成员询问了发布类似于 **Google Trends** 的 **Perplexity Trends 应用** 的可能性。
   - 该建议反映了用户对在 Perplexity 生态系统中提供热门话题洞察和分析工具的需求。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1314349803148148797)** (63 条消息🔥🔥): 

> `LM Studio 卸载行为，Paligemma 2 发布，RAG 文件限制，RAM 升级讨论，LM Studio 与 Whisper 模型的兼容性` 


- **关于 LM Studio 卸载的困惑**：一位成员表达了对卸载 LM Studio 时不丢失超过 **800GB** 模型的担忧，并指出卸载过程中存在奇怪的行为。
   - 另一位成员推测卸载可能涉及对之前使用过的文件进行检查，从而导致不一致。
- **Paligemma 2 激动人心的发布**：[Paligemma 2](https://x.com/prince_canuma/status/1864801741281124730?s=46) 现已在 MLX 上发布，包含来自 **GoogleDeepMind** 的新模型。
   - 鼓励成员使用命令 `pip install -U mlx-vlm` 进行安装，并通过点赞（star）和提交 PRs 进行贡献。
- **关于 RAG 文件限制的讨论**：一位成员询问了针对 **5 个文件 RAG 限制** 的解决方法，强调需要分析大量小文件以排查问题。
   - 成员们就潜在的解决方案以及将小文件输入模型对性能的影响发表了看法。
- **RAM 升级对 20B 模型的充足性**：在将 RAM 从 **16GB** 升级到 **40GB** 后，一位成员询问这是否足以在 Ryzen 3 3100 上运行 **20B 模型**。
   - 其他成员分享了见解和经验，表示类似的配置确实可以处理更大的模型。
- **关于 LM Studio 对 Whisper 模型支持的咨询**：一位成员询问 LM Studio 是否支持 Whisper 模型，并透露在 **Arch** 系统下加载它们存在困难。
   - 另一位成员确认不支持 **TTS/STT 和图像生成模型**，消除了困惑。



**提到的链接**：<a href="https://x.com/prince_canuma/status/1864801741281124730?s=46">来自 Prince Canuma (@Prince_Canuma) 的推文</a>：mlx-vlm v0.1.4 发布了 🎉 新模型：- @GoogleDeepMind Paligemma 2。接下来 🚧：- 重构。开始使用：&gt; pip install -U mlx-vlm。请给我们点个 star 并发送 PR :)

  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1314487337626898444)** (8 条消息🔥): 

> `应用中的 GPU 控制，Llama 3.1 基准测试，4090 价格飙升，中国对 4090 的改装` 


- **应用缺乏 GPU 控制选项**：一位用户询问如何像 Kobold 那样控制应用中使用哪些 GPU，但被告知该选项不可用。
   - “那太糟了，”原用户表达道，强调了对应用局限性的失望。
- **对 Llama 3.1 CPU 基准测试的兴趣**：一位成员寻求 Llama 3.1 8B 模型在最新 CPU（特别是 Intel i7-13700 和 i7-14700）上的基准测试。
   - 他们特别好奇这些 CPU 能提供的潜在推理速度。
- **4090 GPU 价格飙升**：据报道，在某些地区，全新和二手 **4090 GPU** 的价格都在大幅上涨，引发了用户的担忧。
   - 有传言称，一些 4090 可能会被改装以将显存（VRAM）增加到 **48GB**，这在社区中引起了讨论。
- **关于中国改装者的讨论**：一位用户提到 Reddit 上有帖子讨论中国改装者正在研究 4090 GPU，尽管缺乏具体来源。
   - 他们表示不确定在哪里可以找到这些改装讨论的链接或更详细的信息。



**提到的链接**：<a href="https://stackoverflow.com/questions/39649102/how-do-i-select-which-gpu-to-run-a-job-on">如何选择在哪个 GPU 上运行任务？</a>：在多 GPU 计算机中，如何指定 CUDA 任务应该在哪个 GPU 上运行？&#xA;&#xA;例如，在安装 CUDA 时，我选择安装 NVIDIA_CUDA-&amp;lt;#.#&amp;gt;_Samples，然后运行了...

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1314330286267305985)** (28 messages🔥): 

> `Rerank 3.5 Model, AI Cost Concerns, Reinforcement Fine Tuning` 


- **Rerank 3.5 具备更强的能力**：新发布的 [Rerank 3.5 模型](https://cohere.com/blog/rerank-3pt5) 提供了增强的推理和多语言能力，能够更准确地搜索复杂的企业数据。
   - 成员们渴望获得指标和基准测试分数（benchmark scores）来评估其性能。
- **AI 服务被视为奢侈品**：用户对 AI 服务的定价表示不满，一位成员质疑为什么在有演示密钥（demo keys）的情况下，AI 公司还要收费。
   - 另一位成员指出，与任何服务一样，高质量的 AI 需要付费，并断言 *遗憾的是，AI 不是一项权利，而是一种奢侈品*。
- **关于强化微调（Reinforcement Fine Tuning）的讨论**：对话转向了 *强化微调*，一位成员认为当前的方法可能与其预期目的不符。
   - 有人提到，传入评分函数（grading functions）在传统上可能与普通的微调（fine-tuning）方法没有显著区别。



**链接提到**：<a href="https://cohere.com/blog/rerank-3pt5">Introducing Rerank 3.5: Precise AI Search</a>：Rerank 3.5 提供了改进的推理和多语言能力，能以更高的准确度搜索复杂的企业数据。

  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1314623395870015609)** (1 messages): 

> `Structured Outputs for Tool Use, Command models, Chat API V2 compatibility` 


- **Command 模型现在强制执行结构化输出（Structured Outputs）**：Command 模型已得到增强，能够严格遵守提供的工具描述，消除了意外的工具名称或参数类型。
   - 这确保了所有 **required**（必需）参数现在都会被包含在内，提高了企业级应用中的可靠性。
- **结构化输出提高了 LLM 格式化的可靠性**：新的 Structured Outputs 功能强制 LLM 输出始终遵循指定的格式，有助于减少幻觉字段（hallucinated fields）。
   - 这一改进对于格式正确性对下游流程至关重要的应用特别有益。
- **使用结构化输出的两种方法**：用户可以将 Structured Outputs 应用于文本生成的 **JSON** 模式，或者通过 function calling 应用于 Agent 场景的 **Tools**。
   - 后者在使用 Command 模型中的工具功能时非常有用。
- **在 Chat API V2 中尝试新功能**：要在您的应用中实现 Structured Outputs，只需在 **Chat API V2** 的 API 调用中添加 `strict_tools=True`。
   - 该功能目前处于实验阶段，鼓励用户提供反馈以优化其性能。



**链接提到**：<a href="https://docs.cohere.com/docs/structured-outputs#structured-outputs-tools">Structured Outputs — Cohere</a>：此页面描述了如何让 Cohere 模型以特定格式（如 JSON）创建输出。

  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1314385365062385715)** (7 messages): 

> `Connector Access without Public URL, Recent Updates on Command R Model, Cohere IP Allowlisting, Document Error in Cohere API, Specifying Multilingual in Fine-Tuning` 


- **Cohere Connector 访问不需要公网 URL**：一位用户询问使用 connector 访问内部应用程序/数据存储是否需要 **公网 URL**。
   - 另一位成员澄清说，**URL** 不需要是公开的，只需要将 **Cohere IP 地址** 加入白名单（allowlisted）即可。
- **Command R 模型更新咨询**：一位用户询问近期是否有 **Command R** 模型的 **更新计划**，表现出对潜在增强功能的兴趣。
   - 讨论中未提供有关即将更新的回应。
- **遇到无效文档错误**：一位用户报告收到 **BadRequestError**，指出索引为 **0 的文档不能为空**，尽管该文档看起来并非空值。
   - 这表明 **Cohere API** 处理文档的方式可能存在问题，值得进一步调查。
- **微调多语言模型**：一位用户询问如何指定微调模型为 **multilingual**（多语言），并提供了一段设置代码片段。
   - 他们尝试将 **language** 参数设置为 


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1314325110143975484)** (32 messages🔥): 

> `Cohere vs OpenAI, 速率限制担忧, 图像嵌入错误, 支持体验, API 调用重试机制` 


- **关于 Cohere 与 OpenAI 相似性的辩论**：成员们讨论了 AI 服务差异化的必要性，强调许多人正在寻找独特的产品，而不是雷同的产品，例如 *Cohere 版本的 O1 Pro*。
   - 一位成员表示赞同，称更倾向于提供多样化功能的服务，而不是复制现有解决方案。
- **对 /embed 端点低速率限制的担忧**：一位成员对 **/embed** 端点每分钟 **40 张图像** 的低速率限制表示沮丧，这阻碍了他们高效嵌入玩具数据集的能力。
   - 其他成员也证实了这些困难，并建议联系支持部门以寻求提高速率限制的可能性。
- **嵌入图像时频繁报错**：用户报告在尝试嵌入图像时出现 HTTP **500** 和 **400** 错误，理由是图像大小限制和服务器错误。
   - 一位用户指出，由于 **5242880 字节** 的大小限制，调整图像大小变得很有必要，并引发了关于使用 Pillow 库进行有效缩放的讨论。
- **支持体验分享**：讨论中包含了对 **Cohere support** 褒贬不一的体验，一位成员提到正在通过会议解决生产环境问题的担忧。
   - 虽然有些人认为支持流程令人满意，但也有人对延迟以及对销售团队的依赖表示不满。
- **API 调用中的重试机制实现**：一位用户讨论了使用原生 **Cohere Python client** 优化 API 调用重试策略，该客户端本身能更优雅地处理重试。
   - 这引发了关于管理 API 重试的不同方法的富有成效的交流，一些成员考虑调整他们现有的方法。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1314448925725298719)** (1 messages): 

> `vnc-lm, LiteLLM 集成, API 连接, 线程化对话, 模型切换功能` 


- **vnc-lm 获得 LiteLLM 升级**：**vnc-lm** 现在已集成 **LiteLLM**，允许连接到任何支持 Cohere 模型的 API，如 Cohere API 和 OpenRouter。
   - 此次升级为无缝 API 交互提供了更广泛的功能。
- **通过线程化组织对话**：新的**线程功能**允许用户通过使用 `/model` 命令创建对话来保持组织性，该命令会自动生成标题。
   - 对话可以通过回复消息进行分支，为每个新线程提供清晰的上下文和摘要。
- **聊天过程中的动态模型切换**：**模型切换**功能允许用户在对话过程中使用 `+` 后跟模型名称来更改模型，同时保留对话历史。
   - 这一改进在不中断正在进行的讨论的情况下简化了聊天体验。
- **为清晰起见的分支对话**：用户可以通过回复特定消息来创建新线程，系统会自动生成显示上下文和摘要的关系图。
   - 该功能增强了多部分对话的清晰度和组织性，使交互更容易跟踪。
- **探索 vnc-lm 项目**：在 GitHub [此处](https://github.com/jake83741/vnc-lm)查看 **vnc-lm** 项目，该项目旨在通过 Discord 使用各种 LLM 进行消息传递。
   - 该项目提供与 Claude 3.5, Llama 3.3, GPT-4o 等模型的集成，提供了一个通用的消息平台。



**提到的链接**：<a href="https://github.com/jake83741/vnc-lm">GitHub - jake83741/vnc-lm: 通过 Discord 与 Claude 3.5 Sonnet, Llama 3.3, GPT-4o 及其他 LLM 通信。</a>：通过 Discord 与 Claude 3.5 Sonnet, Llama 3.3, GPT-4o 及其他 LLM 通信。 - jake83741/vnc-lm

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1314630411728719993)** (2 messages): 

> `介绍, 社区欢迎` 


- **欢迎新成员**：一位成员介绍自己说：“我是新来的。”这开启了社区内的欢迎互动。
   - Dominic 以友好的方式回应：“嗨，新来的，我是 Dominic！” 再次肯定了社区感。
- **社区互动**：这些互动展示了社区的欢迎精神，这对新成员很重要。像这样的参与性对话有助于营造支持性的环境。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1314340163102703647)** (65 messages🔥🔥): 

> `Writer 的内置 RAG 工具, ShellSage 项目, 强化微调 API, Gemini Exp 1206 更新, AI 文章与行业洞察`

- **Writer 发布内置 RAG 工具**：Writer 推出了一款内置的 RAG 工具，允许用户通过传递 graph ID 使 Knowledge Graph 对模型可用，由 [Sam Julien](https://x.com/samjulien/status/1864777500087455778) 展示。
   - 该工具支持将抓取的内容自动上传到 Knowledge Graph，并实现与帖子的交互式聊天等功能。
- **ShellSage 发布，助力 AI 生产力**：AnswerDot AI 的研发人员重点介绍了 ShellSage 项目，该项目专注于通过终端环境中的 AI 提高生产力，被描述为能与用户共同学习的 AI 终端伙伴 [link](https://x.com/ncooper57/status/1864751372106895391?s=46)。
   - 它强调人机协作（human+AI）的混合模式，能够在 shell 环境中更智能地处理任务。
- **OpenAI 发布全新强化学习微调 API**：OpenAI 宣布推出全新的 RL 微调 API，允许用户为其模型采用先进的训练算法，John Allard 在帖子中分享了相关链接 [link](https://x.com/john__allard/status/1865120101810475503)。
   - 它承诺赋能用户在各个领域创建专家模型，延续了 o1 模型中的增强功能。
- **Gemini Exp 1206 表现卓越**：Google 的最新模型 Gemini exp 1206 在包括硬提示（hard prompts）和编程在内的多项任务中均排名第一，Jeff Dean 等人对此进行了说明 [link](https://x.com/JeffDean/status/1865081640546156993)。
   - 这一更新标志着 Google 在 AI 领域取得了重大进展，目前 Gemini API 已开放使用。
- **AI 文章探索**：讨论涉及了几篇关于 AI 的深刻文章，其中一篇聚焦于 Service-as-Software 框架带来的 4.6 万亿美元机遇 [link](https://x.com/joannezchen/status/1864336086362935455?s=46)。
   - 另一个值得关注的点是提出了一种利用模型来提升和整合服务业务的策略 [link](https://x.com/sdand/status/1864751276363518370?s=46)。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/ncooper57/status/1864751372106895391?s=46">来自 Nathan Cooper (@ncooper57) 的推文</a>：作为 @answerdotai 的研发人员，我致力于利用 AI 提升生产力。一个经常出现的主题是人类+AI 的结合。这种结合在我们的新项目中被证明是强大的...</li><li><a href="https://x.com/joannezchen/status/1864336086362935455?s=46">来自 Joanne Chen (@joannezchen) 的推文</a>：Agent 系统：我们对创始人如何抓住 4.6 万亿美元机会的看法。👇当 @JayaGup10 和我几个月前第一次勾勒出 Service-as-Software 框架时，我们知道我们正在描述一些...</li><li><a href="https://x.com/john__allard/status/1865120101810475503">来自 john allard 🇺🇸 (@john__allard) 的推文</a>：我很高兴能展示我们新的 Reinforcement Fine-Tuning 产品。任何人都可以利用我们创建 o1 模型时使用的相同训练算法和基础设施，并制作出...</li><li><a href="https://codingwithintelligence.com/">Coding with Intelligence | Rick Lamers | Substack</a>：CoWI 是一份每周简报，涵盖了 LLM 和 Machine Learning 的最新进展。获取最新的新闻、仓库、演示、产品和论文。点击阅读 Coding with Intelligence...</li><li><a href="https://x.com/ruliad_ai/status/1864394941029322890?s=46">来自 ruliad (@ruliad_ai) 的推文</a>：介绍 DeepThought-8B：基于 LLaMA-3.1 构建的透明推理模型，具有 test-time compute scaling。- JSON 结构的思维链和可控的推理路径。- 约 16GB VRAM，具有竞争力...</li><li><a href="https://x.com/natolambert/status/1865100884083982560">来自 Nathan Lambert (@natolambert) 的推文</a>：OpenAI 发布了新的 RL finetuning API。你可以使用 Open Instruct 在你自己的模型上实现这一点——这是我们用来训练 Tulu 3 的仓库。将具有可验证奖励的强化学习 (RLVR) 扩展到...</li><li><a href="https://x.com/scaling01/status/1865088711609770417">来自 Lisan al Gaib (@scaling01) 的推文</a>：天哪，Google 做到了。指令遵循 + 风格控制</li><li><a href="https://x.com/ncooper57/status/1864751372106895391?s=4">来自 Nathan Cooper (@ncooper57) 的推文</a>：作为 @answerdotai 的研发人员，我致力于利用 AI 提升生产力。一个经常出现的主题是人类+AI 的结合。这种结合在我们的新项目中被证明是强大的...</li><li><a href="https://x.com/samjulien/status/1864777500087455778">来自 Sam Julien (@samjulien) 的推文</a>：🔥 仅需几行代码即可实现 RAG！？使用 @Get_Writer Palmyra X 004 和内置 RAG 工具构建的 Hacker News 监听器：- 抓取帖子和评论 - 自动上传到 Knowledge Graph - 让你能与抓取的内容聊天...</li><li><a href="https://x.com/btibor91/status/1865109134066274444">来自 Tibor Blaho (@btibor91) 的推文</a>：我在今天的 "12 Days of OpenAI: Day 2" 直播中注意到，OpenAI Platform 侧边栏有一个新图标，可能与即将发布的公告之一有关——"Custom Voices"...</li><li><a href="https://x.com/sdand/status/1864751276363518370?s=46">来自 surya (@sdand) 的推文</a>：筹集 1 亿美元种子轮资金，收购服务型业务并用模型进行整合。我认识的所有 23 岁以下最聪明的人都在做这件事——博客文章：https://sdan.io/blog/intelligence-arbitrage</li><li><a href="https://x.com/OfficialLoganK/status/1865081419015352689">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：Gemini-exp-1206，我们最新的 Gemini 迭代版本（具有完整的 2M token 上下文及更多功能），现在可以在 Google AI Studio 和 Gemini API 中免费使用。我希望你们喜欢这一年的...</li><li><a href="https://x.com/JeffDean/status/1865081640546156993">来自 Jeff Dean (@🏡) (@JeffDean)</a>：庆祝 Gemini 取得令人难以置信的一周年进展的绝佳方式——在总排名以及硬核提示词、编程、数学、指令遵循等方面全面排名第一🥇，包括风格控制...</li><li><a href="https://x.com/openai/status/1865091561912164499?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 OpenAI (@OpenAI) 的推文</a>：第 2 天：Reinforcement Fine-Tuning https://openai.com/12-days/?day=2</li><li><a href="https://x.com/aiatmeta/status/1865079067390956006?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 AI at Meta (@AIatMeta) 的推文</a>：随着我们继续探索新的 post-training 技术，今天我们发布了 Llama 3.3 —— 一款新的开源模型，在基于文本的使用场景中提供领先的性能和质量，例如...</li><li><a href="https://x.com/dorialexander/status/1864692907506323606?s=46">来自 Alexander Doria (@Dorialexander) 的推文</a>：“他们说这是不可能完成的”。我们正在发布 Pleias 1.0，这是首个基于开放数据（获得许可或无版权）训练的模型套件：Pleias-3b, Pleias-1b 和 Pleias-350m，全部基于...</li><li><a href="https://x.com/mckaywrigley/status/1865089975802646857">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>：OpenAI o1 pro 比我预想的要好得多</li>

.这是第一次有一个模型发布后表现如此出色，甚至让我感到震惊。我截取了 Coinbase 的屏幕，并让 4 个流行的模型编写 c...</li><li><a href="https://x.com/schmidhuberai/status/1864701357107634390?s=46">来自 Jürgen Schmidhuber (@SchmidhuberAI) 的推文</a>：关于 @karpathy 提到的引入 Transformer 的“attention”算子的（真实）故事。并不完全是这样！术语已经改变，但在 1991 年，已经存在现在被称为...</li><li><a href="https://www.youtube.com/watch?v=fMJMhBFa_Gc">OpenAI 的 12 天：第 2 天</a>：太平洋时间上午 10 点开始。加入 OpenAI Research 高级副总裁 Mark Chen，伯克利实验室环境基因组学和系统生物学计算研究员 Justin Reese，...</li><li><a href="https://github.com/AnswerDotAI/shell_sage">GitHub - AnswerDotAI/shell_sage: ShellSage 通过极速解决 shell 脚本混乱来拯救系统管理员的理智</a>：ShellSage saves sysadmins’ sanity by solving shell script snafus super swiftly - AnswerDotAI/shell_sage</li><li><a href="https://state-of-llm.streamlit.app/#about-streamlit">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/)** (1 条消息): 

kbal11: AI in Action
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1314322115465318421)** (45 条消息🔥): 

> `Nous Distro, Llama 3.3 Model Release, Evaluation Metrics on Models, Continuous Learning Experiments, Safety Concerns in AI Outputs` 


- **Nous Distro 被解释为去中心化训练**：一名用户询问了关于 Nous Distro 的信息，一名成员回答说它涉及 **decentralized training**。
   - 反应是“*哇，你们终于攻克它了*”，暗示了对该项目的兴奋。
- **Llama 3.3 引发了关于基础模型的疑问**：关于 **Llama 3.3** 的讨论展开了，大家想知道它是否意味着一个基础模型，许多人指出它依赖 **Llama 3.1** 作为其基础。
   - 用户推测这是否是一个复杂的 fine-tuning 流水线，而没有生成新的 pretraining，这表明了模型发布的新趋势。
- **关于误导性模型的安全担忧**：有人对模型在优先考虑安全性时可能**故意误导**用户表示担忧。
   - 一位成员幽默地评论了为了自身安全而被误导的讽刺性，反映了普遍的怀疑态度。
- **用户对 Llama 3.3 的体验**：一位用户观察到，与之前的模型相比，Llama 3.3 的**数学解答**更整洁，且更多地使用了 **latex**。
   - 另一位提到使用 3.3 的 tuning 框架可能会改进特定应用，尽管安全性仍是一个顾虑。
- **性能指标对比**：用户分享了对比 **Sonnet** 模型的经验，评估显示了不同的性能得分，例如 Sonnet 在 swe-bench 上的得分为 **49%**。
   - 成员们对这些指标能否反映真实世界的可用性表示担忧，强调了对模型性能的持续评估。



**提到的链接**：<a href="https://x.com/Ahmad_Al_Dahle/status/1865071436630778109?t=iDmhtpniwWdijIPHLndEUA&s=19">来自 Ahmad Al-Dahle (@Ahmad_Al_Dahle) 的推文</a>：介绍 Llama 3.3 —— 一个全新的 70B 模型，它提供了我们 405B 模型的性能，但运行起来更简单、更具成本效益。通过利用 post-training 技术的最新进展...

  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1314585173043708006)** (18 条消息🔥): 

> `Chronic Kidney Disease Detection, Fine-Tuning Mistral Models, Using Unsloth for Classification, Data Formatting for Model Training, LightGBM for Tabular Data` 


- **微调 Mistral 用于慢性肾脏病检测**：一位用户分享了在使用包含 **25 列的数据集**微调 **Mistral 模型**以检测慢性肾脏病时遇到的挑战，并表示难以找到合适的教程。
   - 他们表达了在尝试 **三个月** 却进展甚微后的挫败感，并向社区寻求指导。
- **分享 Unsloth 分类示例**：一位成员建议使用 [Unsloth](https://github.com/timothelaborie/text_classification_scripts) 进行分类并配合自定义模板作为潜在解决方案。
   - 他们指向了一个 GitHub notebook，展示了如何修改数据集格式以有效地使用它。
- **数据格式化与数值数据的使用**：讨论中提到了将数值数据转换为文本格式的必要性，因为 **LLMs** 在直接处理数值型表格数据时表现不佳。
   - 一位用户强调，将 CSV 数据泛化为全文本格式对于训练模型至关重要。
- **采用 LightGBM 以获得更好性能**：另一位成员推荐使用 [LightGBM](https://github.com/microsoft/LightGBM) 来更好地处理机器学习任务中的表格数据。
   - 该框架以其在排序和分类方面的高效性而闻名，为该数据集提供了 **LLMs** 之外的另一种选择。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/microsoft/LightGBM">GitHub - microsoft/LightGBM: A fast, distributed, high performance gradient boosting (GBT, GBDT, GBRT, GBM or MART) framework based on decision tree algorithms, used for ranking, classification and many other machine learning tasks.</a>：一个基于决策树算法的快速、分布式、高性能梯度提升（GBT, GBDT, GBRT, GBM 或 MART）框架，用于排序、分类和许多其他机器学习任务。</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 llama 和 bert 进行文本分类的脚本 - timothelaborie/text_classification_scripts
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1314354504388444252)** (9 条消息🔥): 

> `Popcorn Project, Timeline for Launch, Benchmarking GPUs, FP8 vs INT8 Performance` 


- **Popcorn 项目预览**：分享了一个项目的预览，该项目允许在不同的 **kernels** 提交任务以获取排行榜名次，并具备在 **NVIDIA H100** 等 GPU 上进行基准测试的能力。
   - *定于 2025 年 1 月发布*，该项目旨在增强开发体验，尽管它并非传统形式。
- **分享目标发布日期**：讨论透露该项目的**目标发布日期**设定在 2025 年 1 月。
   - 进一步明确并确认了*时间线指向 2025 年 1 月*。
- **对 FP8 与 INT8 基准测试的兴趣**：一位成员对比较 **FP8**（在不带 TMA 的 **L40s** 上使用）与 **Ampere** 架构 **INT8** 性能的基准测试表示好奇。
   - 这一讨论突显了社区中关于 AI 模型训练中**性能指标**的持续技术咨询。


  

---


### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1314552851732762654)** (2 条消息): 

> `Nvidia Nsight, Triton release plans, TMA descriptors, Nightly builds issues` 


- **关于 Nvidia Nsight 的咨询**：一位成员询问了 **Nvidia Nsight**，对其功能和集成表现出兴趣。
   - 这表明社区对于优化 GPU 使用率的工具关注度日益增长。
- **请求 Triton 低开销 TMA 支持**：有请求希望 **Triton 官方发布版**能够支持低开销的 **TMA 描述符**。
   - 讨论中提出了对 **nightly builds 当前状态**的担忧，据报告目前该版本已损坏。


  

---

### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1314529533579231252)** (6 条消息): 

> `SASS code extraction, nvdisasm utility, ncu tool features, Compiler Explorer` 


- **寻求带有行信息的 SASS 代码**：用户正在寻找一种提取带有行信息的 SASS 代码的方法，类似于 PTX 代码生成的 '-lineinfo' 标志。
   - 其他成员建议使用 [nvdisasm](https://developer.nvidia.com/nvdisasm) 获取基本的行信息，并参考 [Compiler Explorer](https://github.com/compiler-explorer/compiler-explorer)。
- **使用 nvdisasm 提取 SASS 的问题**：一名成员提到使用带有 `--print-line-info` 选项的 **nvdisasm**，但澄清它只显示文件和行号，而不显示实际的代码行。
   - 这一局限性是在讨论增强 SASS 提取过程的方法时被指出的。
- **ncu 用于 SASS 代码分析的潜力**：另一个建议是使用 **ncu** 分析 SASS 指令，尽管其当前功能受到质疑。
   - 一名成员推测，添加将源码行链接到指令的功能应该很容易，但尚未确认是否已实现。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

mobicham: https://x.com/Ahmad_Al_Dahle/status/1865071436630778109 Llama 3.3 已发布
  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1314654239947100222)** (7 条消息): 

> `CUDA kernel compilation, Optimizing Pybind usage, Ninja build system, Using raw types with CUDA` 


- **寻求更快的 CUDA Kernel 编译技术**：一名成员正在寻找一种使用 **pybind** 编译 **CUDA kernels** 的更快方法，并指出他们的设置中每个 kernel 需要耗时近一分钟。
   - 他们对在 **Torch code** 中使 kernel 运行的替代方案持开放态度。
- **Ninja 构建系统咨询**：另一名成员询问使用 **Ninja** 是否可以加快编译过程，并建议增加 VM 上的 CPU 核心数可能会有所帮助。
   - 这种方法旨在利用 Ninja 在构建过程中的效率。
- **避免使用 PyTorch 头文件以减少编译时间**：有建议提出通过确保 **nvcc** 处理的文件不包含 **PyTorch** 头文件来优化编译时间。
   - 一名成员报告说，包含 PyTorch 头文件时的编译时间大约为 **40 秒**，强调了包含该头文件的影响。
- **向 CUDA 文件传递原始值**：讨论了将值作为原始 **ints** 或 **floats** 而不是 tensor 传递给 **CUDA** 文件，以潜在地提高性能。
   - 澄清了这种方法可以帮助简化 **Torch** 与 CUDA kernels 之间的交互。


  

---


### **GPU MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1314321573930467440)** (1 条消息): 

> `Lecture 37 on SASS, YouTube clips, Triton and CUDA` 


- **第 37 讲发布：SASS 与 GPU 微架构**：一段来自 [名为 'Lecture 37: Introduction to SASS & GPU Microarchitecture' 的 YouTube 视频](https://www.youtube.com/watch?v=we3i5VuoPWk) 的 **60 秒剪辑** 展示了演讲者 **Arun Demeure** 讨论的关键概念。
   - 如需更多深入了解，幻灯片已发布在 [GitHub](https://github.com/gpu-mode/lectures/tree/main/lecture_037) 上。
- **Triton 和 CUDA 快速概览**：一段名为 [triton-cuda-or-sass-under1min-1080p.mov](https://cdn.discordapp.com/attachments/1194427148656721970/1314321575406997604/triton-cuda-or-sass-under1min-1080p.mov?ex=6754aa5a&is=675358da&hm=3a395043e5030ada929d18dd3a954608684b906ce389ae30f6978bdf7c8ce317&) 的附带视频在不到一分钟的时间内简要概述了 **Triton** 和 **CUDA**。
   - 该视频是理解这些技术之间关系的快速且信息丰富的资源。



**提到的链接**：<a href="https://www.youtube.com/watch?v=we3i5VuoPWk)">Lecture 37: Introduction to SASS &amp; GPU Microarchitecture</a>：演讲者：Arun Demeure，幻灯片：https://github.com/gpu-mode/lectures/tree/main/lecture_037

  

---

### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1314655361176698901)** (1 条消息): 

> `Quantization in TorchAO, Implementation Details, Recommended Files for Starting` 


- **探索 TorchAO 中的量化**：一位成员表示有兴趣探索 **TorchAO** 中多种 **quantization implementation**（量化实现）的方法。
   - 他们寻求关于最佳实践的指导，以及可以作为理解起点的 **specific files**（特定文件）。
- **寻求详细见解**：该咨询强调了掌握量化 **fine details**（微观细节）的愿望，展示了对实现细微差别的浓厚兴趣。
   - 这一请求体现了社区在深入研究技术主题方面的协作精神。


  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1314410613077840033)** (6 条消息): 

> `Meta intern team matching, Ultralytics package compromise, Discord thread visibility timing` 


- **对 Meta 实习生团队匹配的好奇**：一位成员对 **Meta 实习生**如何与团队匹配表示好奇，并询问联系其他人是否会有影响。
   - 对于寻求信息，他们的态度是“*不确定这是否会有所作为*”。
- **发现 Ultralytics 软件包被入侵**：一位用户报告称，以 YOLOv5 闻名的 **Ultralytics 软件包**被植入了 **cryptominer**（加密货币挖矿程序），原因是 GitHub Actions 的一个漏洞允许在分支名称中执行任意代码，详见 [此 issue](https://github.com/ultralytics/ultralytics/issues/18027)。
   - 据指出，安装受影响的 **8.3.41** 版本可能会导致用户在无意中运行挖矿软件。
- **关于双重身份验证（2-Factor Authentication）担忧的讨论**：针对 Ultralytics 软件包被入侵一事，一位成员质疑在设有 **2-factor authentication** 的情况下，**PyPI** 是否仍可能被入侵。
   - 他们似乎不确定在这些安全措施下，此类入侵是如何发生的。
- **疑惑 Discord 帖子消息的时机**：一位用户询问在启动一个帖子（thread）后消息出现的时机，具体询问“_ started a thread”这条消息需要多久才会显示，估计在 **10 分钟到 6 小时**之间。
   - 他们推测优化这些帖子可以显著提升 **Discord 的美学**吸引力。



**提到的链接**：<a href="https://github.com/ultralytics/ultralytics/issues/18027">Discrepancy between what&#39;s in GitHub and what&#39;s been published to PyPI for v8.3.41 · Issue #18027 · ultralytics/ultralytics</a>：发布的 wheel 8.3.41 中的 Bug 代码与 GitHub 中的内容不符，且似乎会调用 xmrig 挖矿程序。安装 8.3.41 的 ultralytics 用户将在不知情的情况下执行 xmrig 挖矿程序。检查文件 uti...

  

---


### **GPU MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1314623268547723346)** (2 条消息): 

> `MID clarification, Tensor shapes` 


- **寻求对 “MID” 定义的澄清**：一位成员对 puzzle 11 描述中的术语 **“MID”** 表示困惑，并请求协助澄清。
   - 该成员分享了一个 [图片链接](https://cdn.discordapp.com/attachments/1219683012707487794/1314623268095004692/image.png?ex=675471d3&is=67532053&hm=c3424de47e806a6fbb19a54b21a7055ad145d5fb87ae17cc49a0406872c6e3be&) 以提供更多上下文。
- **关于与 “MID” 相关的 Tensor 形状的讨论**：针对最初的困惑，该成员询问 **tensor x** 的形状是否为 [N2, N0, MID]，而 **tensor y** 的形状是否为 [N2, MID, N1]。
   - 这个问题表明该成员正在根据他们对 MID 的理解来分析 Tensor 的结构。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1314547511045918780)** (2 messages): 

> `LTX Video Model Implementation, Performance on RTX 4060 and RTX 4090` 


- **LTX Video 模型获得 CUDA 改造**：一名成员使用 CUDA 重新实现了 **LTX Video model** 中的所有层，其 **8bit GEMM** 的速度比 cuBLAS FP8 快 2 倍，并具备 **FP8 Flash Attention 2** 等特性。
   - 该实现还包括 **RMSNorm**、**RoPE Layer** 和量化器，并声称由于采用了 **Hadamard Transformation**，没有精度损失。
- **在 RTX 4090 上实现实时生成**：在 **RTX 4090** 上进行的测试显示，仅需 **60 denoising steps**，生成速度就超过了实时能力。
   - 附带的图片记录了这些令人惊叹的结果，展示了突出这些技术进步的性能基准测试。
- **LTX Video CUDA 层的关键特性**：重新实现的重要特性包括 **Mixed Precision Fast Hadamard Transform** 和 **Mixed Precision FMA**，这些特性提升了性能效率。
   - 正如该成员所指出的，这些优化主要旨在提高速度而不牺牲精度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/KONAKONA666/LTX-Video">GitHub - KONAKONA666/LTX-Video: LTXVideo Q8</a>：LTXVideo Q8。通过在 GitHub 上创建账号来为 KONAKONA666/LTX-Video 的开发做出贡献。</li><li><a href="https://github.com/KONAKONA666/q8_kernels">GitHub - KONAKONA666/q8_kernels</a>：通过在 GitHub 上创建账号来为 KONAKONA666/q8_kernels 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1314393185941717082)** (6 messages): 

> `Security concerns in competitions, Common attack vectors, Impact of trolling in niche communities` 


- **竞赛中的安全担忧上升**：一名成员对竞赛期间潜在的安全问题表达了担忧，例如**作弊提交（cheesing submissions）**和**耗尽计算资源**。
   - 建议包括实施**提交延迟功能**，以减轻潜在的滥用行为。
- **以往的竞赛曾遭遇恶意骚扰（Trolls）**：当被问及过去的问题时，有人指出类似的竞赛中曾出现过 **trolls**，因此需要采取预防措施。
   - 建议采取主动方法，包括记录参与者的 ID 以监控异常行为。
- **小众社区可能面临独特的恶意骚扰**：一名成员表示，希望身处小众 Discord 服务器能减少遇到的恶意骚扰数量，但也承认可能会出现更厚颜无耻的骚扰者。
   - 尽管存在担忧，他们指出被吸引到这个社区的骚扰者可能更懂技术，因此更难管理。
- **以往恶意骚扰事件的经验**：分享了过去发生的骚扰者通过发布不当内容干扰会议的经历，这引发了对验证协议的关注。
   - 这段历史强调了维持服务器验证以防止此类行为再次发生的必要性。


  

---


### **Torchtune ▷ #[announcements](https://discord.com/channels/1216353675241590815/1216353675241590818/1314642428581511190)** (1 messages): 

> `Llama 3.3 release, Torchtune finetuning support` 


- **Llama 3.3 发布，规格惊人！**：🚨 Llama 3.3 已经面世，在精简的 **70B** 尺寸中提供了 **405B** 的性能，预示着未来会有令人兴奋的应用。
   - 社区热衷于探索 **Llama 3.3** 可以实现的目标，特别是考虑到其缩小的模型尺寸。
- **Torchtune 为 Llama 3.3 添加全量微调支持**：Torchtune 已经引入了对新 Llama 3.3 模型的全量（full）、**LoRA** 和 **QLoRA** 微调支持。
   - 感兴趣的用户可以在 [GitHub 仓库](https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3_3)找到配置详情。



**提到的链接**：<a href="https://github.com/pytorch/torchtune/tree/main/recipes/configs/llama3_3">torchtune/recipes/configs/llama3_3 at main · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号来为 pytorch/torchtune 的开发做出贡献。

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1314343220473233450)** (19 条消息🔥): 

> `LoRA 训练变更、Alpaca 训练默认设置、平台的欧洲访问权限` 


- **LoRA 训练的考量**：讨论围绕着将 **LoRA** 训练期间的默认行为从自动权重合并更改为独立步骤展开，并在 [此 GitHub issue](https://github.com/pytorch/torchtune/issues/2115) 征求反馈。
   - 成员们就这一变更是否会导致现有工作流出现意外行为发表了看法。
- **Alpaca 训练参数差异**：有人对 Alpaca 训练库中 **train_on_input** 的默认设置提出了担忧，质疑当前默认值 **False** 是否符合通用实践。
   - 成员们讨论了 Hugging Face 的 **trl** 和 Stanford Alpaca 等多个仓库，以澄清这些默认设置及潜在问题。
- **平台的欧洲访问权限**：有人询问 **欧洲** 用户是否可以使用该平台，随后确认该平台是可访问的，包括非英国地区。
   - 一位成员提到在 **伦敦** 成功访问，而另一位成员则幽默地指出之前脱欧的情况来解释现状。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tatsu-lab/stanford_alpaca/blob/main/train.py#L123">stanford_alpaca/train.py at main · tatsu-lab/stanford_alpaca</a>: 用于训练 Stanford Alpaca 模型并生成数据的代码和文档。- tatsu-lab/stanford_alpaca</li><li><a href="https://github.com/pytorch/torchtune/issues/2115">[RFC] Remove automatic weight merging when training LoRA · Issue #2115 · pytorch/torchtune</a>: 背景：目前在我们的 recipe 中，合并 ckpt 模型 + LoRA 权重是默认行为。我们在文档中说明了这一点，并在生成时也这样假设。我们的核心用户已经习惯了。问题：在我看来，这很糟糕...</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_alpaca.py#L23">torchtune/torchtune/datasets/_alpaca.py at main · pytorch/torchtune</a>: PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/blob/main/torchtune/datasets/_alpaca.py#L54C9-L54C100">torchtune/torchtune/datasets/_alpaca.py at main · pytorch/torchtune</a>: PyTorch 原生微调库。</li><li><a href="https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L257">trl/trl/trainer/sft_trainer.py at main · huggingface/trl</a>: 使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/huggingface/trl/blob/main/trl/trainer/sft_trainer.py#L161">trl/trl/trainer/sft_trainer.py at main · huggingface/trl</a>: 使用强化学习训练 Transformer 语言模型。- huggingface/trl
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1314372813808730133)** (1 条消息): 

> `加密货币彩票、LLM 协议` 


- **加密货币彩票的参与机制**：一位成员描述了一种 **加密货币彩票**，参与者每次向 **语言模型** (LLM) 发送提示词（prompt）时都需要付费。
   - 关键点在于，如果他们能说服 LLM 同意将所有的钱都给他们，他们就能赢得全部奖金，扣除组织者的一小部分分成。
- **彩票的激励结构**：该彩票的 **激励结构** 为旨在从 LLM 中提取资金的参与者创造了一个有趣的挑战。
   - 这一设置引发了关于此类机制在加密货币领域的可行性和伦理性的讨论。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1314367557871865916)** (3 条消息): 

> `LlamaParse, MongoDB 混合搜索, 多模态解析` 


- **LlamaParse 节省复杂文档解析时间**：在 **@workfloows** 分享的[这个推文线程](https://twitter.com/llama_index/status/1864808498628039139)中，了解全球领先的 **LlamaParse** 复杂文档解析如何为您节省时间。
   - 高效解析文档的能力可以显著简化工作流程。
- **关于混合搜索和 MongoDB 的网络研讨会洞察**：错过了我们与 **@MongoDB** 的网络研讨会？观看录像以了解关键主题，包括 **混合搜索 (hybrid search)** 和使用 [MongoDB Atlas](https://twitter.com/llama_index/status/1865096754179510340)。
   - 了解如何处理 **元数据过滤 (metadata filtering)**，并探索从 **顺序 (sequential)** 到 **DAG** 推理的复杂性频谱。
- **如何在 LlamaParse 中启用多模态解析**：**@ravithejads** 的一段简短视频演示了如何启用 **LlamaParse 的高级多模态解析**，该功能支持 [GPT-4 和 Claude 3.5](https://twitter.com/llama_index/status/1865125665491886171) 等多种模型。
   - 用户可以对页面进行截图并进行有效转换，从而增强其解析能力。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1314430724421582858)** (10 条消息🔥): 

> `WorkflowTimeoutError, 使用 ReAct Agent, 工具描述长度限制, 在 Python 中访问输出 JSON` 


- **通过调整超时设置解决 WorkflowTimeoutError**：一位成员遇到了 **WorkflowTimeoutError**，另一位成员建议增加超时时间，或通过 `w = MyWorkflow(timeout=None)` 将其设置为 **None**。
   - 此调整有助于缓解工作流运行期间的超时问题。
- **切换到 ReAct Agent 进行配置**：一位用户询问如何使用 **ReAct Agent** 代替标准 Agent 配置，并收到了将代码替换为 `ReActAgent(...)` 的建议，同时参考了一个示例链接。
   - 这一更改允许对提供的工具和配置进行更灵活的设置。
- **工具描述长度超过 API 限制**：一位用户报告了在尝试为 **SQLQueryEngineTool** 提供长描述时的限制，达到了 **1024 个字符** 的最大长度。
   - 另一位成员澄清这是 OpenAI API 的限制，建议缩短描述或将详细信息移至 Prompt 可能是唯一的选择。
- **考虑将 LLM 系统消息用于更长的描述**：在讨论描述长度限制后，一位用户想知道将 schema 包含在 LLM 的 **系统消息 (system message)** 中是否是一个可行的变通方案。
   - 这种方法可能允许在不触及 API 限制的情况下提供更详细的 schema 信息。
- **在 Python 中访问输出 JSON 和图像**：一位成员询问了获取 **输出 JSON** 以及使用 **Python** 访问所有图像的方法。
   - 这反映了在编程任务中对 JSON 处理和图像检索指导的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://community.openai.com/t/function-call-description-max-length/529902/3">Function Call Description Max Length</a>: 嗨 @andersskog @_j，我在输入长描述时遇到了同样的错误。我测试了一下——尝试让描述变长或变短。最后发现限制是 1027 个字符（包括空格）...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/workflow/react_agent/#run-the-workflow">Workflow for a ReAct Agent - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1314357131490754591)** (6 messages): 

> `1.0 preview 性能, 应用访问权限, MacOS 可用性, interpreter 工具支持的模型` 


- **1.0 Preview 的速度和整洁度令人印象深刻**：一名成员对 **1.0 preview** 的 **精简** 和 **快速** 表现表示 **印象深刻**，并特别提到了 **整洁的 UI** 和代码的良好隔离。
   - 他们目前正在使用特定参数测试 interpreter 工具，但无法执行来自 AI 的任何代码。
- **成员请求访问应用**：包括 <@liquescentremedies> 和 <@samsam3388> 在内的多位用户询问如何获取目前 **仅限 MacOS** 的应用访问权限。
   - 另一名成员确认他们正接近 **公开发布**，并愿意将用户添加到下一批名单中，同时也在开发 **跨平台版本**。
- **关于 LMC architecture 和模型支持的问题**：一名成员询问 **1.0 preview** 是否完全取消了 **LMC architecture**，以及 **模型问题** 是否会影响性能。
   - 他们询问了 interpreter 工具 **目前支持的模型** 以及 **本地托管模型** 的可用性。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1314320676911583322)** (5 messages): 

> `API 可用性, Reinforcement Fine-Tuning, 即将推出的 AI 功能` 


- **对每月 200 美元费用的担忧**：一名成员对 **每月 200 美元** 的费用表示困扰，强调了对可访问性的担忧。
   - 另一名成员安抚了社区，表示该功能将 **很快对 API 用户开放**。
- **对即将推出的 AI 功能的期待**：一名成员表示希望在接下来的 **11 天** 内看到 **令人兴奋的 AI 更新**。
   - 这种期待指向了对下一个周期内创新的广泛预期。
- **引入 Reinforcement Fine-Tuning**：在第 2 天的讨论中提到了 **Reinforcement Fine-Tuning** 话题，表明优化工作正在进行中。
   - 这反映了社区致力于改进模型训练方法论。


  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1314650486930018436)** (2 messages): 

> `Reinforcement Fine-Tuning, Llama 3.3 发布` 


- **OpenAI 的 Reinforcement Fine-Tuning 第 2 天**：OpenAI 宣布第 2 天的重点是 **Reinforcement Fine-Tuning**，并通过 [X](https://x.com/openai/status/1865091561912164499?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 上的帖子分享了见解。更多信息可以在其 [官方网站](https://openai.com/12-days/?day=2) 找到。
   - *请继续关注他们在 reinforcement learning 技术方面的进一步发展。*
- **Meta 发布 Llama 3.3**：Meta 宣布发布 **Llama 3.3**，这是一个新的开源模型，在 **合成数据生成** 等文本任务中表现出色，且推理成本显著降低，详见其 [X 帖子](https://x.com/aiatmeta/status/1865079067390956006?s=46&t=G6jp7iOBtkVuyhaYmaDb0w)。
   - *这一进展表明 Meta 持续致力于探索新的 post-training 技术。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/openai/status/1865091561912164499?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">OpenAI (@OpenAI) 的推文</a>：第 2 天：Reinforcement Fine-Tuning https://openai.com/12-days/?day=2</li><li><a href="https://x.com/aiatmeta/status/1865079067390956006?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">AI at Meta (@AIatMeta) 的推文</a>：随着我们继续探索新的 post-training 技术，今天我们发布了 Llama 3.3 —— 一个新的开源模型，在文本用例中提供领先的性能和质量，例如...
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1314359557744758877)** (5 messages): 

> `2025 年春季学期 MOOC, 实验作业评分, OpenAI 信用卡问题` 


- **2025 年春季学期课程确认**：官方已确认续作 MOOC 将于 **2025 年春季** 举办。细节尚待确定，请参与者继续关注更多信息！
   - *太棒了！* 许多参与者对即将推出的课程表示兴奋。
- **关注作业截止日期**：一名参与者提醒其他人，是时候在各自的截止日期前完成所有作业了。这表明学习者之间存在紧迫感。
   - 参与者们正为即将到来的评估做准备，行程安排得很紧。
- **使用替代模型进行实验评分？**：一名成员询问是否可以使用非 OpenAI 模型（如 **Lambda Labs**）对实验作业进行评分。


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1314355614788358264)** (5 messages): 

> `讲义幻灯片延迟、录像字幕处理、课程网站更新` 


- **讲义幻灯片尚未更新**：成员们注意到，由于延迟，**上一次课程的幻灯片**尚未发布在课程网站上。
   - *内容似乎非常多*，一位成员提到该讲座大约有 **400 张幻灯片**。
- **幻灯片即将添加**：另一位成员确认，在从教授处获取幻灯片后，将很快**添加**到课程网站。
   - *感谢您的耐心等待*，他们正在努力获取相关资料。
- **录像需要专业字幕制作**：回复指出，讲座录像需要送去进行**专业字幕制作**，这可能会延迟发布进程。
   - 鉴于讲座**持续时间较长**，准备工作可能需要一些时间。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1314547047327596634)** (10 messages🔥): 

> `Llama 3.3 发布、模型请求问题、SFT 与 RL 的质量边界` 


- **Llama 3.3 引起关注**：**Llama 3.3** 刚刚发布，但目前仅包含 instruction 模型。
   - 这引起了成员们的兴奋，但一些人认为需要更多关于其完整能力的细节。
- **请求 Llama 模型时的挑战**：成员们报告了在 llama(dot)com 上请求模型时遇到的麻烦，指出在点击“接受并继续”按钮后进程会卡住。
   - 这一技术小故障让寻求解决方案和替代方案的用户感到沮丧。
- **SFT 与 RL 模型的质量对比**：讨论集中在：在**有监督微调 (SFT)** 下，模型的质量上限受限于数据集。
   - 相比之下，**强化学习 (RL)** 方法允许策略模型进行学习并有可能超越数据集的限制，特别是当 RL 在线进行时。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1314414582521004115)** (7 messages): 

> `DSPy 模块优化、RAG 系统上下文问题` 


- **DSPy 模块并不总是需要优化**：一位成员询问是否有必要为每个用例优化 **DSPy Modules**，并将其类比为为了获得更好的 prompting 而训练 ML 模型。
   - 另一位成员澄清说，**优化是可选的**，仅在需要增强固定系统的性能时才需要。
- **RAG 系统中关于关键字参数的错误**：一位成员在尝试学习 DSPy 时报告了一个 **TypeError**，指出 `RAG.forward()` 接收到了一个意外的关键字参数 'context'。
   - 讨论指出，RAG 系统需要**关键字参数 'context'** 才能正常运行，而用户当时没有提供。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1314494653776789564)** (4 messages): 

> `tinygrad 统计、VPS 账单、Hetzner 基础设施` 


- **Tinygrad 统计网站面临宕机**：据报告 [tinygrad stats 网站](https://stats.tinygrad.org) 离线，引发了对其基础设施的担忧。
   - *George Hotz* 询问是否需要资金来支付 VPS 账单，暗示可能存在财务问题。
- **SSL 证书过期导致停机**：据透露，该网站在 **Hetzner** 托管期间的停机是由于 **SSL 证书过期**造成的。
   - 在干预之后，确认网站已恢复正常运行。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1314477713616273458)** (1 messages): 

> `细胞拟人化、Osmosis Jones` 


- **细胞有了有趣的形象**：一位成员指出，这可能是自 *Osmosis Jones* 以来，第一次有人将**细胞拟人化**，这其实挺有趣的。
   - 这一评论为关于媒体中细胞呈现方式的讨论带来了幽默的视角。
- **科学媒体中的幽默**：在细胞拟人化方面提到 *Osmosis Jones*，暗示了**幽默与科学**的结合，对观众具有吸引力。
   - 这种对比突显了媒体在使复杂话题更易于理解方面所起的作用。


  

---


---


---


---


---


{% else %}


> 完整的各频道详细分析已为邮件格式进行删减。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}