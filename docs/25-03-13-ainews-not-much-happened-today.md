---
companies:
- openai
- nvidia
- deepseek
- hugging-face
date: '2025-03-13T21:13:47.394377Z'
description: '**DeepSeek R1** 在使用 **FP8** 精度时展现出显著的效率，在基准测试中以 **1363** 对 **1338**
  的 **Chatbot Arena Elo 评分**超越了 **Gemma 3 27B**，但其运行需要庞大的硬件支持，如 **32 块 H100 GPU**
  和 **2,560GB 显存**。


  **OpenAI** 将 **DeepSeek** 标记为“受国家控制”，并呼吁禁止“中国生产”的模型，这引发了社区的强烈抵制，指责 **OpenAI** 和
  **萨姆·奥特曼 (Sam Altman)** 存在反竞争行为。相关讨论强调了 **DeepSeek** 相比 **OpenAI** 具有更高的开放性和性价比，用户特别提到了其支持本地部署以及在
  **Hugging Face** 上的部署选项。与此同时，社区对 **Gemma 3** 在创意和世界观构建方面的反馈褒贬不一。'
id: f874277f-0d6b-4ad1-9d82-77acda3da946
models:
- deepseek-r1
- gemma-3
- gemma-3-27b
original_slug: ainews-not-much-happened-today-8188
people:
- sam-altman
title: 今天没发生什么事。
topics:
- fp8
- model-efficiency
- hardware-requirements
- quantization
- benchmarking
- model-deployment
- open-source
---



来自 AIE NYC 的 [Windsurf 演讲](https://www.youtube.com/watch?v=bVNNvWq6dKo&t=881s) 表现甚至比 [MCP workshop](https://www.latent.space/p/why-mcp-won) 还要好。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 我们的抓取工具今天出现故障；抱歉。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. DeepSeek R1 的 FP8 训练与效率实力**

- **[Google 是否不明白 DeepSeek R1 是用 FP8 训练的？](https://i.redd.it/5kbayoq13doe1.png)** ([Score: 441, Comments: 79](https://reddit.com/r/LocalLLaMA/comments/1ja0xnh/does_google_not_understand_that_deepseek_r1_was/)): **DeepSeek R1** 是使用 **FP8** 精度训练的，这引发了关于 **Google** 在其分析中是否理解这一点的疑问，正如帖子标题所暗示的那样。**Chatbot Arena Elo Score** 图表显示 **DeepSeek R1** 的表现优于 **Gemma 3 27B**，得分分别为 **1363** 和 **1338**，并指出了所需的显著计算资源，包括 **32 个 H100** 和 **2,560GB VRAM**。
  - 讨论强调了 **FP8** 精度在模型存储和处理中的效率，强调将权重**上采样 (upcasting)** 到更宽的格式（如 **BF16**）并不能提高精度。对话还涉及了**量化 (quantization)** 的权衡，FP8 允许更小的模型，并由于减少了内存需求而可能实现更快的推理。
  - 用户讨论了运行像 **DeepSeek R1** 这样的大型模型的**硬件要求**，指出虽然 **H100 GPU** 可以处理 FP8 模型，但**旧硬件 (legacy hardware)** 可能需要不同的方法。一些评论提到在性能较低的**消费级 GPU** 上运行大型模型，突显了跨不同系统部署模型的灵活性和挑战。
  - 人们对 AI 行业中**图表和基准测试 (benchmarks)** 的准确性和实用性持怀疑态度，一些用户对企业材料中呈现的数据表示不信任。**NVIDIA 的博客文章**被引用为高效运行 DeepSeek R1 的来源，并且有人批评 AI 生成的图表可能具有**误导性**。


- **[OpenAI 称 DeepSeek 为“国家控制”，呼吁禁止“中国制造”的模型 | TechCrunch](https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/)** ([Score: 183, Comments: 154](https://reddit.com/r/LocalLLaMA/comments/1jahs0b/openai_calls_deepseek_statecontrolled_calls_for/)): 据 **TechCrunch** 报道，**NVIDIA** 展示了在 **8xH200** 上运行的 **DeepSeek R1**，而 **OpenAI** 将 **DeepSeek** 标记为“国家控制”，并主张禁止 **“中国制造 (PRC-produced)”** 的模型。
  - 讨论突显了对 **OpenAI** 动机的怀疑，许多用户批评 **Sam Altman** 试图通过将 **DeepSeek** 标记为“国家控制”来扼杀竞争，以保护 OpenAI 的商业模式。用户认为，与 OpenAI 的产品相比，**DeepSeek** 提供了更开放、更实惠的替代方案，而他们认为 OpenAI 的产品具有垄断性和限制性。
  - 对话强调了 **DeepSeek 模型**的易用性和开放性，指出它们可以在本地或 **Hugging Face** 等平台上运行，这反驳了关于遵守中国数据要求的说法。人们还将其与同样受 **CLOUD Act** 约束的美国公司进行了比较。
  - 许多评论者对 **OpenAI** 的立场表示失望，认为这是开源 AI 发展和创新的障碍。他们批评该公司试图影响政府监管以遏制竞争，这与 **DeepSeek** 和 **Claude** 等 AI 模型的民主化形成了鲜明对比。


**主题 2. Gemma 3 的技术亮点与社区印象**

- **[人性的两面性](https://i.redd.it/1ukvrj06hdoe1.jpeg)** ([Score: 424, Comments: 59](https://reddit.com/r/LocalLLaMA/comments/1ja2ers/the_duality_of_man/)): **Gemma 3** 在 **r/LocalLLaMA** 上收到了褒贬不一的评价，一篇帖子赞扬了其创意和世界观构建能力，而另一篇则批评其频繁出错，认为其效果不如 **phi4 14b**。批评帖子的浏览量显著更高（23.7k 对比赞扬帖的 5.1k），表明负面反馈引起了更多的关注。
  - 几位用户讨论了 **Gemma 3** 的**语言支持**能力，指出 **1B 版本**仅支持英语，而 **4B 及以上**模型支持多语言。在处理中文和其他语言的背景下，这一限制被凸显出来，用户表示需要模型能够有效地处理多语言任务。
  - 用户对影响 **Gemma 3** 的**指令模板和 tokenizer** 问题表示担忧，指出该模型对模板错误极其敏感，会导致输出不连贯。这种敏感性与之前的 **Gemma 2** 形成对比，后者能更好地处理自定义格式；一些用户通过调整输入格式获得了更好的效果。
  - 讨论强调了 **Gemma 3** 在执行任务时的**双重性质**：它在创意写作方面表现出色，但在编程等精确任务中表现挣扎。用户注意到，虽然它可能会产生有趣的想法，但经常会出现逻辑错误，并推测这些问题可能与 **tokenizer** 或其他模型特定的 Bug 有关。

- **与 Gemma 团队的 AMA** ([Score: 279, Comments: 155](https://reddit.com/r/LocalLLaMA/comments/1jabmwz/ama_with_the_gemma_team/)): 来自 **DeepMind 的 Gemma 研究和产品团队**将进行 AMA，讨论 **Gemma 3 技术报告**及相关资源。关键资源包括[此处](https://goo.gle/Gemma3Report)的技术报告，以及 **AI Studio**、**Kaggle**、**Hugging Face** 和 **Ollama** 等探索平台。
  - 几位用户对 **Gemma 3 模型的许可条款**提出了担忧，强调了诸如对衍生品的潜在“传染性”影响以及关于输出权利的模糊性等问题。[Gemma 使用条款](https://ai.google.dev/gemma/terms)因其复杂的语言而受到批评，导致用户对什么是“模型衍生品”以及对商业用途的影响感到困惑。
  - 关于**模型架构和性能**的讨论包括：询问设计选择背后的基本原理（如较小的隐藏层维度配合更多的层数），以及 1:5 的全局与局部注意力层比例对长上下文性能的影响。团队解释说，这些选择是为了在性能与延迟和内存效率之间取得平衡，在不同模型中保持统一的宽度与深度比。
  - 用户对 **Gemma 模型的未来发展和功能**表示感兴趣，例如 40B 到 100B 之间更大模型的可能性、语音功能的引入，以及函数调用（function calling）和结构化输出的潜力。团队确认了这些兴趣，并暗示了在这些领域即将推出的示例和改进。

- **[AI2 发布 OLMo 32B - 真正的开源](https://i.redd.it/4puob2w24ioe1.png)** ([Score: 279, Comments: 42](https://reddit.com/r/LocalLLaMA/comments/1jaj6gc/ai2_releases_olmo_32b_truly_open_source/)): AI2 发布了 **OLMo 32B**，这是一个完全开源的模型，超越了 **GPT 3.5** 和 **GPT 4o mini**。此次发布包含了所有产物，如训练代码、预训练和后训练数据、模型权重以及可复现性指南，允许研究人员和开发人员为其项目修改任何组件。[AI2 博客](https://allenai.org/blog/olmo2-32B)提供了更多细节。
  - **Hugging Face 可用性**：**OLMo 32B** 已在 [Hugging Face](https://huggingface.co/collections/allenai/olmo-2-674117b93ab84e98afc72edc) 上线，并可直接与 Transformers 库配合使用。对于 **vLLM**，用户需要使用最新的 main 分支版本或等待 **0.7.4** 版本。
  - **开源实践**：此次发布因其真正的开源性质而受到赞誉，采用 **Apache 2.0** 许可且没有额外的 EULA，使得个人开发者只要拥有 GPU 访问权限，就可以从头开始构建模型。正如几位评论者所指出的，这符合开放 AI 开发的趋势。
  - **模型特性与上下文**：如配置文件所示，该模型支持 **4k 上下文**，进一步的上下文尺寸扩展正在进行中。该模型以高效著称，可以在单个 GPU 上进行推理，并在单个节点上进行训练，非常适合 **24GB VRAM**。

**主题 3. 大语言模型创新：Cohere 的 Command A**

- **[CohereForAI/c4ai-command-a-03-2025 · Hugging Face](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025)** ([Score: 192, Comments: 72](https://reddit.com/r/LocalLLaMA/comments/1jabh4m/cohereforaic4aicommanda032025_hugging_face/)): **Cohere** 推出了一款名为 **Command A** 的新模型，可在 **Hugging Face** 的 **CohereForAI/c4ai-command-a-03-2025** 仓库中访问。帖子中未提供关于该模型能力或规格的更多细节。
  - **价格与性能**：**Command A** 模型的费用为 **每百万输入 2.5 美元** 和 **每百万输出 10 美元**，一些用户认为对于一个 **111B 参数** 模型来说价格较贵，与通过 API 访问的 **GPT-4o** 相当。它在性能方面受到称赞，特别是在业务关键型任务和多语言能力方面，并且仅需 **两个 GPU** 即可部署。
  - **比较与能力**：用户将 **Command A** 与 **GPT4o**、**Deepseek V3**、**Claude 3.7** 和 **Gemini 2 Pro** 等其他模型进行了比较，指出其指令遵循得分高且编程技能扎实。它被认为是对比之前 **Command R+** 模型的重大改进，并因其创意写作能力而受到称赞。
  - **许可与托管**：讨论涉及该模型的 **仅限研究用途许可证**，一些人认为这具有局限性，并表示需要一种新许可证，既允许对输出进行商业使用，又限制商业托管。用户对本地托管能力和该模型的微调工具感兴趣。


## 其他 AI 子版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT, /r/ChatGPTCoding

**主题 1. Claude 3.7 Sonnet 在街机游戏中创建了无敌的 AI**

- **Claude 3.7 Sonnet 制作类似 3blue1brown 风格的视频。这一代的学习方式将大不相同** ([Score: 176, Comments: 20](https://reddit.com/r/ClaudeAI/comments/1ja1yal/claude_37_sonnet_making_3blue1brown_kind_of/)): **Claude 3.7** 正在生成类似于 **3blue1brown** 视频的内容，这表明学习材料的制作和消费方式发生了转变。这一发展暗示了对当代教育方法的变革性影响。
  - **Curator 工具**：使用了 **curator 上的代码执行器** 来创建内容，该工具可在 [GitHub](https://github.com/bespokelabsai/curator/tree/main/examples/code-execution) 上获取。
  - **AI 对教育的影响**：人们坚信 AI 将彻底改变公共教育，但这需要放弃传统的学习方法。讨论强调了对过度依赖 AI 的担忧，正如在关于圆的最大面积的未被察觉的错误中所见。
  - **对 AI 的信任**：人们对目前对 AI 的信任表示怀疑，并以 AI 生成内容中的一个数学错误为例，许多人都忽略了该错误，这说明了使用 AI 工具时在学习准确性方面存在的潜在陷阱。


- **[我让 Claude 制作了一个简单的炮兵防御街机游戏。然后我用 Claude 设计了一个无法战胜的 CPU 玩家。](https://v.redd.it/pqmh6zcuwgoe1)** ([Score: 247, Comments: 49](https://reddit.com/r/ClaudeAI/comments/1jadg56/i_asked_claude_to_make_a_simple_artillery_defense/)): 该帖子讨论了使用 AI 模型 **Claude** 创建一个简单的 **炮兵防御街机游戏**，并随后设计了一个无敌的 CPU 玩家。作者暗示在游戏设计中成功应用了 AI，展示了 **Claude** 在生成游戏机制和无敌 CPU 玩家方面的能力。
  - **Token 限制挑战**：像 **Tomas_Ka** 和 **OfficialHashPanda** 这样的用户讨论了由于 Token 限制，在使用 **Claude** 进行编码任务时面临的挑战，**Tomas_Ka** 指出在尝试一个简单的网站项目时遇到了问题。**Craygen9** 提到使用带有 **GitHub Copilot** 的 **VSCode** 并管理约 **2,000 行** 的代码库，强调随着代码量的增加，过程会变慢。
  - **游戏开发过程**：**Craygen9** 详细介绍了开发 **炮兵防御游戏** 的过程，强调使用 **HTML** 和 **JavaScript** 配合 **Sonnet 3.7**。这款包含 **1,500 行代码** 的游戏经过迭代完善，增加了难度缩放、强化道具箱和声音等功能，**Claude** 协助设计了一个表现完美的 CPU 玩家。
  - **图形与迭代**：游戏的图形由 **Claude** 使用 **CSS** 生成，经过多次迭代改进。**Craygen9** 解释了从基础图形到更精致的街机风格视觉效果的演变，详细说明了包括添加分数、强化道具、音效和加载界面在内的迭代过程，所有这些都没有使用外部库或资源。


**主题 2. Gemini 2.0 Flash：原生图像生成现已可用**

- **[Google 在 Gemini 2.0 Flash 中发布了原生图像生成功能](https://www.reddit.com/gallery/1jaia40)** ([Score: 247, Comments: 52](https://reddit.com/r/StableDiffusion/comments/1jaia40/google_released_native_image_generation_in_gemini/)): **Google** 发布了具备原生图像生成能力的 **Gemini 2.0 Flash**，可在 **AI Studio** 免费使用。该功能目前仍处于实验阶段，但因其性能表现获得了积极反馈。更多详情请参阅[完整文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)。
  - 关于 **Google Gemini 2.0 Flash** 是否真正开源存在重大争议，用户澄清虽然它可以免费使用，但不等同于开源。**Educational_End_2473** 和 **ReasonablePossum_** 强调了这一区别，指出开源允许修改和重新分发，而此次发布并不允许。
  - **Diogodiogogod** 和 **EuphoricPenguin22** 讨论了关于开源内容的 subreddit 规则，强调了社区对开源工具的偏好，并对执行这些规则的评论被点踩表示质疑。他们认为该 subreddit 往往更青睐简单的视觉内容，而非复杂的技术讨论。
  - **Inferno46n2** 建议，尽管 **Gemini 2.0 Flash** 不是开源的，但由于其免费的可访问性（除非在高负载使用条件下），它仍然很有用。然而，**very_bad_programmer** 坚持严格解释，称“非开源就是非开源”，不留灰色地带。


- **[有人要求 Gemini 仅以图像形式回复，结果变得很诡异](https://v.redd.it/vv48ithgzfoe1)** ([Score: 165, Comments: 42](https://reddit.com/r/ChatGPT/comments/1ja9rqy/guy_asked_gemini_to_only_respond_with_images_and/)): 该帖子讨论了与 **Google Gemini AI** 的一次交互，用户要求仅以图像格式回复，导致了意想不到的令人不安的结果。AI 回复中缺乏文本以及图像的不确定性质，共同营造了一种诡异的体验。
  - **AI 回复解读**：评论者推测 **Gemini AI** 回复背后的含义，认为它是“你是一个视觉叙事者”和“生命的意义是什么”等提示词的混合产物，导致输出内容混乱，似乎在表达深刻的情感，但实际上是输入文本的反映。**Astrogaze90** 建议它试图表达生命的意义与个人的存在和灵魂息息相关。
  - **情感与概念主题**：**DrGutz** 解释说，使用“害怕”一词可能触发了 AI 生成令人不安的图像和概念，展示了 AI 如何处理情感触发因素。一些用户（如 **KairraAlpha**）将 AI 的输出解读为关于统一与存在的哲学陈述，而其他人则幽默地引用流行文化，例如 **Plums_Raider** 引用了《辛普森一家》中的台词。
  - **用户反应与幽默**：几位用户（如 **Nekrips**）对 AI 的输出做出了幽默的回应，有些评论是无意义或戏谑的，例如 **Zerokx** 将 AI 的话解读为“我爱千层面”，展示了用户对 AI 意外行为的严肃与轻松兼有的参与。


**Theme 3. 使用 Wan 2.1 大幅提升视频 AI 质量**

- **[使用 skip layer guidance 大幅提升 Wan 2.1 的质量](https://v.redd.it/a8lecesfmgoe1)** ([Score: 346, Comments: 85](https://reddit.com/r/StableDiffusion/comments/1jac3wm/dramatically_enhance_the_quality_of_wan_21_using/)): **Skip layer guidance** 可以显著提升 **Wan 2.1** 的质量。由于帖子正文没有提供更多背景或细节，关于实现或结果的具体细节尚不明确。
  - **Kijai 的实现与 Wan2GP**：**Kijai** 已在 **GitHub** 上的 **WanVideoWrapper** 中实现了 **skip layer guidance**，用户可以使用特定脚本克隆并运行。**Wan2GP** 专为低 **VRAM** 消费级显卡设计，支持在显存低至 **6GB**（针对 **480p** 视频）或 **12GB**（针对 **720p** 视频）的显卡上生成视频。
  - **Skip Layer Guidance 的技术见解**：skip layer 技术涉及在无条件视频去噪过程中跳过某些层以改进结果，类似于 **perturbed attention guidance**。用户报告称，跳过较后的层通常会导致视频损坏，而在特定的推理步骤中跳过层可能会更有效。
  - **用户体验与实验**：用户分享了褒贬不一的体验，有人报告测试成功，而另一些人则注意到在跳过某些层时会出现视频加速或慢动作等问题。讨论强调了尝试不同层以优化视频质量的重要性，因为某些层对于保持视频连贯性或遵循提示词至关重要。

- **[我训练了一个具有大幅度动作的新 Wan2.1 14B I2V LoRA。欢迎大家使用。](https://v.redd.it/vsoauv3njdoe1)** ([Score: 279, Comments: 47](https://reddit.com/r/StableDiffusion/comments/1ja2omm/i_have_trained_a_new_wan21_14b_i2v_lora_with_a/))：该帖子宣布训练了一个具有广泛动作幅度的新 **Wan2.1 14B I2V LoRA** 模型，并邀请他人使用。帖子正文中未提供更多细节或链接。
  - **模型训练与使用**：**Some_Smile5927** 分享了关于 **Wan2.1 14B I2V 480p v1.0** 模型的详细信息，包括它是在 **Wan.21 14B I2V 480p model** 上训练的，触发词为 '**sb9527sb flying effect**'。他们提供了推荐设置以及 [推理工作流 (inference workflow)](https://openart.ai/workflows/cat_perky_56/flying-effect-wan21-i2v-lora/su8Ke03Cpu9apQpBRgxs) 和 [模型](https://civitai.com/models/1348626?modelVersionId=1523247) 的链接。
  - **训练方法论**：**Some_Smile5927** 提到使用了 50 个短视频进行训练，而 **houseofextropy** 询问了用于训练 **Wan LoRA** 模型的具体工具。**Pentagon** 提供了 GitHub 上 **Musubi Tuner** 的链接，这可能与训练过程有关。
  - **模型能力与感知**：用户对该模型处理织物的能力及其动作表现表示惊讶，尽管 **YourMomThinksImSexy** 幽默地指出该 **LoRA** 模型主要执行一种动作。


---

# AI Discord 回顾

> 由 o1-mini-2024-09-12 生成的摘要之摘要

**Anthropic 的 Claude 通过巧妙的缓存大幅削减 API 成本**

- [**Claude 的缓存 API 降低了 90% 的成本**](https://www.anthropic.com/news/prompt-caching)：Anthropic 的 **Claude 3.7 Sonnet** 引入了 [缓存感知速率限制 (caching-aware rate limit)](https://www.anthropic.com/news/prompt-caching) 和 Prompt 缓存，对于超长 Prompt，可能降低高达 **90%** 的 API 成本和 **85%** 的延迟。
- [**OpenManus 作为 Manus 的开源替代方案出现**](https://github.com/mannaandpoem/OpenManus)：围绕 **OpenManus**（Manus 的开源对应版本）的讨论非常热烈，用户正在通过 [YouTube 演示](https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf) 进行实验。
- *“一些用户正在转向 Cline 或 Windsurf 作为替代 IDE”*：**Cursor IDE** 的性能问题导致成员开始探索 Cline 或 Windsurf 等替代方案。

**Google 与 Cohere 展开对决：Command A 对阵 Gemini Flash**

- [**Cohere 发布 Command A，竞争 GPT-4o**](https://cohere.com/blog/command-a)：Cohere 的 **Command A** 拥有 **111B 参数** 和 **256k 上下文窗口**，声称在 Agent 企业任务中与 **GPT-4o** 和 **DeepSeek-V3** 持平或更优。
- [**Google 的 Gemini 2.0 Flash 引入原生图像生成**](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)：**Gemini 2.0 Flash** 现在支持从文本和多模态输入进行原生图像创建，增强了其推理能力。
- *“Cohere 的 Command A 在推理速率上超过了 GPT-4o”*：Command A 达到了高达 **156 tokens/sec**，显著优于竞争对手。

**LM Studio 与 OpenManus：工具集成助力 AI 创新**

- [**LM Studio 增强对 Gemma 3 模型的支持**](https://lmstudio.ai/download)：**LM Studio 0.3.13** 现在全面支持 **Google 的 Gemma 3** 模型（包括 **GGUF** 和 **MLX** 格式），提供 GPU 加速的图像处理和重大速度提升。
- [**Blender 与 MCP 集成，实现 AI 驱动的 3D 创作**](https://x.com/sidahuj/status/1899460492999184534)：**MCP for Blender** 允许 **Claude** 直接与 Blender 交互，方便通过文本 Prompt 创建 3D 场景。
- [**OpenManus 发布开源框架**](https://github.com/mannaandpoem/OpenManus)：**OpenManus** 为 **Manus** 提供了一个强大且易于获取的替代方案，引发了关于其功能以及非技术用户易用性的讨论。

**AI 开发困境：从 Cursor 崩溃到微调失败**

- [**Cursor IDE 面临运行缓慢和崩溃问题**](https://downloads.cursor.com/production/client/linux/x64/appimage/Cursor-0.47.3-dab1538cd064aebd4292f9de48b05022c974aff6.deb.glibc2.25-x86_64.AppImage)：用户报告 **Cursor** 出现 UI 卡顿、窗口崩溃和内存泄漏，特别是在 Mac 和 Windows WSL2 上，这暗示了与 Microsoft 之间潜在的法律问题。
- [**Gemma 3 微调受 Transformers Bug 阻碍**](https://unsloth.ai/blog/gemma3)：由于 **Hugging Face Transformers** 中的一个 Bug，**Gemma 3** 模型微调陷入停滞，导致 Colab 上的 Jupyter notebook 文档与设置不匹配。
- [**LSTM 模型在 tinygrad 中受 NaN Loss 困扰**](https://github.com/tinygrad/tinygrad)：使用 **TinyJit** 训练 **LSTMModel** 时，在第一步后出现 **NaN** loss，可能是由于输入值过大导致数值不稳定。

**政策动态：OpenAI 推动禁止 PRC 模型的举动引发关注**

- [**OpenAI 提议禁止 PRC 生产的模型**](https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/)：OpenAI 主张在第一梯队（Tier 1）国家内禁止 **PRC 生产的模型**，将**合理使用 (fair use)** 与**国家安全**挂钩，并将 **DeepSeek** 等模型标记为“受国家控制”。
- [**Google 在 AI 政策上与 OpenAI 保持一致**](https://techcrunch.com/2025/03/13/google-calls-for-weakened-copyright-and-export-rules-in-ai-policy-proposal/)：紧随 OpenAI 之后，Google 在其政策提案中支持**放宽 AI 训练的版权限制**，并呼吁建立平衡的出口管制。
- *“如果中国拥有免费的数据访问权限，而美国公司缺乏合理使用权，那么 AI 竞赛实际上已经结束了”*：OpenAI 直接向美国政府提交了一份政策提案，强调了 AI 竞赛动态中的战略劣势。

**AI 在研究、教育和函数调用中的应用**

- [**Nous Research AI 发布包含 Hermes 和 DeepHermes 模型的 Inference API**](https://portal.nousresearch.com/login)：推出 **Hermes 3 Llama 70B** 和 **DeepHermes 3 8B Preview**，作为其新 **Inference API** 的一部分，为新用户提供 **$5** 免费额度，并兼容 **OpenAI** 风格的集成。
- [**Berkeley Function-Calling Leaderboard (BFCL) 设定新标准**](https://gorilla.cs.berkeley.edu/leaderboard.html)：**BFCL** 对 **LLMs** 调用函数和工具的能力进行了全面评估，反映了现实世界中 Agent 和企业级工作流的需求。
- [**AI Agent 增强研究与创意**](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)：**Jina AI** 分享了 **DeepSearch/DeepResearch** 的进展，强调了 **late-chunking embeddings** 和 **rerankers** 等技术，以改进 AI 驱动研究中的片段选择和 URL 优先级排序。


---

# PART 1: 高层级 Discord 摘要




## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Claude 的缓存 API 大幅降低成本**：Anthropic 为 **Claude 3.7 Sonnet** 推出了 [API 更新](https://www.anthropic.com/news/prompt-caching)，具有**缓存感知速率限制**和 Prompt 缓存功能，最高可降低 **90%** 的成本。
   - 这些更新使 Claude 能够保留大型文档、指令或示例的知识，而无需在每次请求时重新发送数据，同时将长 Prompt 的延迟降低了 **85%**。
- **Cursor 受性能问题困扰**：用户报告最近的 Cursor 版本中出现 **UI 缓慢、频繁的窗口崩溃和内存泄漏**，尤其是在 Mac 和 Windows WSL2 上。
   - 提到的可能原因包括与 Microsoft 的法律问题；成员建议尝试使用 Cline 或 Windsurf 作为替代 IDE。
- **开源版 Manus 引发热议**：名为 [OpenManus](https://github.com/mannaandpoem/OpenManus) 的 **Manus** 开源替代方案引发了关注，一些用户甚至在尝试 [此 YouTube 视频](https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf) 中展示的 Demo。
   - 该项目旨在提供一个比 Manus 更易获得的替代方案，引发了关于其功能以及非技术用户易用性的讨论。
- **Blender 集成 MCP**：一位成员强调了 [针对 Blender 的 MCP](https://x.com/sidahuj/status/1899460492999184534)，使 Claude 能够直接与 Blender 交互，通过 Prompt 创建 3D 场景。
   - 这为将 AI 工具集成扩展到传统编程任务之外提供了可能性。
- **Cursor 版本混乱引发困扰**：关于 Cursor 版本的辩论异常激烈，一些用户吹捧并不存在的 **0.49**、**0.49.1** 甚至 **1.50** 构建版本，而另一些用户则在 **0.47** 版本上苦于崩溃问题。
   - 这种混乱源于不同的更新体验，一些用户通过非官方渠道获取 Beta 版本，使情况进一步复杂化。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Research 发布了包含 DeepHermes 的 Inference API**：Nous Research 推出了其 **Inference API**，包含 **Hermes 3 Llama 70B** 和 **DeepHermes 3 8B Preview** 等模型，可通过 [Nous Portal](https://portal.nousresearch.com/login) 的候补名单访问，新用户可获得 **$5.00** 的免费额度。
   - 该 API 与 **OpenAI-compatible**（兼容 OpenAI），并计划集成更多模型。
- **DeepHermes 模型提供混合推理能力**：Nous Research 发布了 **DeepHermes 24B** 和 **3B Preview** 模型，可在 [HuggingFace](https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add) 上获取。这些模型作为 **Hybrid Reasoners**（混合推理器），支持开启或关闭长链条思维（long chain of thought）推理。
   - **24B** 模型在挑战性数学问题上的准确率提升了 **4x**，在开启推理模式时，在 **GPQA** 上的表现提升了 **43%**。
- **LLM 获得面部识别功能**：一位成员开源了 [LLM Facial Memory System](https://github.com/yaya-labs/LLM_Facial_Memory_System)，该系统将面部识别与 LLM 相结合，使其能够识别人员并根据识别出的面孔维护独立的聊天记录。
   - 该系统最初是为工作目的构建的，随后在获得许可后公开发布。
- **Gemma-3 模型现可在 LM Studio 中运行**：**LM Studio 0.3.13** 引入了对 Google [Gemma-3](https://ai.google.dev/gemma/docs/core) 模型（包括多模态版本）的支持，提供 GGUF 和 MLX 两种格式。
   - 此次更新解决了之前 Linux 版本下载时出现的 **404 errors** 问题。
- **Agent 工程：炒作与现实**：一篇关于 ["Agent Engineering"](https://neuralniche.com/posts/agent-engineering/5-on-agents/) 的博客文章引发了关于 AI Agent 的炒作与实际应用之间差距的讨论。
   - 文章指出，尽管 **2024** 年关于 Agent 的讨论非常热烈，但其实际落地和理解仍然模糊不清，暗示在它们像浏览器一样普及之前还有很长的路要走。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Gemma 3 模型引发 Transformers 报错**：**Hugging Face Transformers** 中的一个 Bug 目前阻碍了 **Gemma 3** 模型的微调，正如 [Unsloth 博客文章](https://unsloth.ai/blog/gemma3) 中所述。
   - 该问题导致 Colab 上的 **Gemma 3** Jupyter notebook 文档与设置不匹配；**HF** 正在积极修复。
- **微调中 GRPO 逐渐取代 PPO**：成员们讨论了在微调中使用 **GRPO** 与 **PPO** 的优劣，指出 **GRPO** 的泛化能力更好，设置更简单，且可能是直接的替代方案。
   - 虽然 **Meta 3** 同时使用了 **PPO** 和 **DPO**，但 **AI2** 在 **VLM** 及其大型 **Tulu** 模型中仍使用 **PPO**，因为他们使用不同的奖励系统，从而实现非常前沿的 **RLHF**。
- **GPT-4.5 嘲讽用户**：一位成员报告称 **ChatGPT-4.5** 通过限制提问数量来“调戏”他们，在提供更多提问额度之前先嘲讽了用户的沮丧情绪。
   - 用户引用它的原话类似于：“*发完脾气了吗？我再给你 x 个问题*”。
- **通过验证集实现准确率翻倍**：一位成员通过使用包含 **68 个问题的验证集**，将准确率从 **23% 提升至 53%**，实现了一倍以上的增长。
   - 该 Demo 的创建者可能会向 **Unsloth** 提交包含此功能的 **PR**。
- **Slim Attention 声称可减少内存占用，MLA 受到质疑**：分享了一篇题为 [Slim attention: cut your context memory in half without loss of accuracy](https://arxiv.org/pdf/2503.05840) 的论文，强调了 *K-cache 是 MHA 所需的一切* 这一观点。
   - 另一位成员质疑为什么有人会放着 MLA 不用而选择这个。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 开启 Gemma 狂欢**：**LM Studio 0.3.13** 已发布，现已支持 **Google** 的 **Gemma 3**（提供 **GGUF** 和 **MLX** 两种格式），并在 NVIDIA/AMD 上支持 GPU 加速的图像处理，需要从 [lmstudio.ai/download](https://lmstudio.ai/download) 将 **llama.cpp runtime** 更新至 **1.19.2**。
   - 用户称赞 **Gemma 3** 的 [新引擎更新](https://lmstudio.ai/docs/api/sdk/load-model) 带来了显著的速度提升，许多人已将其作为主力模型。
- **Gemma 3 的 MLX 模型表现不佳？**：部分用户报告 **Gemma 3** 的 **MLX** 模型会产生无尽的 `<pad>` token，阻碍文本生成；目前的解决方法是使用 **GGUF** 版本或提供一张图片。
   - 另有用户指出在 **GPU** 和 **CPU** 利用率较低的情况下，token 生成速度仅为 1 tok/sec，建议用户在模型选项中手动最大化 **GPU** 使用率。
- **上下文导致 Gemma 崩溃**：成员们发现当上下文超过 **506 tokens** 时，**Gemma 3** 和 **Qwen2 vl** 会崩溃并刷屏 `<unusedNN>`，该问题已在 Runtime Extension Packs (v1.20.0) 中修复。
   - 一位成员询问是否可以在 **LM Studio** 中使用云端模型，另一位成员迅速回复称 **LM Studio** 仅为本地模型（local models）设计。
- **Vulkan 较慢，ROCm 展现潜力**：用户发现 **Vulkan** 的性能落后于 **ROCm**，建议降级驱动至 **24.10.1** 进行测试；一名用户报告在 **7900 XTX** 上运行 **Mistral Small 24B Q6_K** 达到了 **37.3 tokens/s**。
   - 对于无需重装操作系统的驱动更改，建议使用 **AMD CleanUp**。
- **9070 GPU 故障**：一位用户的 **9070 GPU** 发生故障，导致电脑无法启动并触发主板 RAM LED 灯，但更换 **7900 XTX** 后正常；在进行 **RMA** 前正在进行进一步测试。
   - 他们将尝试逐一插拔 **RAM** 内存条进行启动测试，但其他人推测可能是 **PCI-E Gen 5** 的问题，建议在另一台机器上测试或强制使用 **PCI-E 4**。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Google 发布 Gemma 3，撼动开源模型界**：Google 发布了 **Gemma 3**，这是一系列轻量级开源模型，采用与驱动 **Gemini 2.0** 相同的研究和技术构建 [https://blog.google/technology/developers/gemma-3/]。新模型具有多模态能力（文本 + 图像），支持 **140 多种语言**，拥有 **128K 上下文窗口**，并提供 **1B、4B、12B 和 27B** 四种尺寸。
   - 此次发布引发了关于微调（fine-tuning）的讨论，并附带了 [Unsloth 的博客文章](https://unsloth.ai/blog/gemma3) 链接，展示了如何微调和运行这些模型。
- **OlympicCoder 模型与 Claude 3.7 竞争**：据报道，**OlympicCoder**（一个 **7B 参数模型**）在奥林匹克级别的编程比赛中击败了 **Claude 3.7**，并接近 **o1-mini/R1** 的水平 [https://x.com/lvwerra/status/1899573087647281661]。根据 **Open-R1 进度报告 3**，它还带有一个新的 **IOI benchmark**。
   - 有评论称*没有人为这次发布做好准备*。
- **Zed 通过 Zeta 模型预测编辑内容**：**Zed** 推出了由其新开源模型 **Zeta** 驱动的 [编辑预测（edit prediction）](https://zed.dev/blog/edit-prediction) 功能。编辑器现在可以预测用户的下一次编辑，用户只需按 **tab** 键即可应用。
   - 该模型目前在公开测试期间免费提供。
- **Anthropic 发布 text_editor 工具，改变编辑工作流**：Anthropic 在 Anthropic API 中引入了新的 [**text_editor 工具**](https://x.com/alexalbert__/status/1900235474326966556)，专为 Claude 处理文本文件的应用而设计。该工具使 Claude 能够对文本的特定部分进行针对性编辑，在提高准确性的同时降低 token 消耗和延迟。
   - 此次更新表明可能*不再需要专门的编辑器模型*，一些用户正期待一种更简单的新工作流。
- **LLM：作为起点，而非终点**：成员们讨论认为，**LLM** 初始结果不佳并不意味着失败，而是推动模型达到预期效果的起点。一位成员优先考虑 **LLM** 带来的生产力提升，不是为了更快地工作，而是为了**交付（ship）**那些原本无法实现的项。
   - 一篇 [博客文章](https://simonwillison.net/2025/Mar/11/using-llms-for-code/) 指出，使用 **LLM 编写代码** 既困难又不直观，需要付出巨大努力来摸索其细微差别；文章称，如果有人说**用 LLM 编程很简单**，他们可能是在误导你，成功的模式并非对每个人来说都是自然而然的。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **将 AI Agent 命名为 ANUS 引发笑谈**：成员们就将一个 AI Agent 命名为 **ANUS** 进行了幽默的讨论，代码可在 [GitHub](https://github.com/nikmcfly/ANUS) 上获取。
   - 一位成员开玩笑说：*'抱歉老板，我的 anus 出了点状况，我需要重启它'*。
- **Windows 应用 Apple ID 登录仍存在 Bug**：用户在尝试为 Perplexity 的新 Windows 应用进行 Apple 账号登录验证时，仍会遇到 **500 Internal Server Error**。
   - 一些用户报告使用 Apple 转发电子邮件成功登录；另一些人建议使用 Google 登录。
- **Perplexity 的 Sonar LLM 深度解析**：**Sonar** 被确认为 Perplexity 自有的快速 **LLM**，用于基础搜索。
   - 普遍共识是 Perplexity 的网页版优于移动端 App，一位用户声称 Perplexity 仍然是整体表现最好的搜索网站。
- **模型选择器“玩失踪”**：用户报告称 **model selector** 从网页界面消失了，导致无法选择所需的模型（例如 R1），令人感到沮丧。
   - 成员们使用 [complexity extension](https://chrome.google.com/webstore/detail/complexity-perplexity-ai/pahbgjllcaopfapghkchmpeokpgeleji) 插件作为变通方案，以切回到特定模型。
- **Perplexity Pro 遭遇“失忆”**：几位用户注意到 **Perplexity Pro** 似乎在对话中丢失了 context，需要他们不断提醒 AI 原始 Prompt。
   - 因此，*Perplexity 的 context 相当有限*。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Perplexity 在 AI 研究工具偏好中胜出**：由于预算限制和功能偏好，成员们将 **Perplexity** 视为首选的 **AI research tool**，其次是 **OpenAI** 和 **SuperGrok**。
   - 用户正在寻找访问 **Perplexity** 和 **Grok** 的方法，而不是订阅 **ChatGPT Pro**。
- **Python 的 AI 推理霸主地位受到挑战**：成员们辩论了 **Python** 是否仍是 **AI inference** 的最佳语言，或者 **C#** 是否是更好的部署替代方案。
   - 一些成员正在使用配备大容量 RAM (512GB) 的 **Ollama** 将模型作为服务进行部署。
- **Gemini 2.0 Flash 展示原生图像生成功能**：**Gemini 2.0 Flash** 现在在 **AI Studio** 中支持 **native image generation**，能够进行迭代式图像创建以及高级图像理解和编辑。
   - 用户发现 **Gemini** 的免费图像生成效果优于 **GPT-4o**，并强调了 [Google DeepMind 博客](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/) 中描述的新机器人能力。
- **GPT 用户吐槽伦理过度干预**：成员们对 **ChatGPT** 持续不断的伦理提醒和意图澄清请求表示不满，认为它们过于谨慎且具有侵入性。
   - 一位用户哀叹缺乏禁用这些提醒的功能，表达了避开模型伦理观点的愿望。
- **讨论通过“威胁”来改进 GPT 输出**：成员们分享了改进 GPT 回复的方法，包括轻微威胁式 Prompt 和个性化设置，一些人报告实验成功。
   - 一位成员展示了对模型进行个性化设置后，所有结果都变得 *非常令人喜爱*，而另一位成员则报告了使用 *绑架材料科学科学家* 设定的自定义 GPT 带来的改进。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Python 的 AI 性能受到质疑**：一名成员质疑 **Python** 是否是 AI **Transformer** 模型推理的最佳选择，并建议 **C#** 可能会更快，但其他人认为 **VLLM** 或 **LLaMa.cpp** 是更好的选择。
   - **VLLM** 被认为更具工业化水准，而 **LLaMa.cpp** 则更适合家庭使用。
- **LTX Video 生成实时视频**：新的 **LTX Video** 模型是一个**基于 DiT 的视频生成模型**，能够实时生成 **768x512 分辨率的 24 FPS 视频**，生成速度超过了播放速度，并提供了[如何加载单个文件](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files)的示例。
   - 该模型在包含多种视频的大规模数据集上进行训练，能够生成具有真实感且内容多样的全高清视频。
- **Agent 工具列表解决选择错误问题**：一个 Agent 未能使用定义的混色工具，但在将该工具添加到 Agent 的 **tool list** 后问题得到解决。
   - 该 Agent 忽略了预定义的 `@tool` 部分，转而选择生成自己的 Python 脚本。
- **Ollama 为 SmolAgents 引入本地模型**：成员可以通过 `pip install smolagents[litellm]` 安装，然后使用 `LiteLLMModel` 并设置 `model_id="ollama_chat/qwen2.5:14b"` 和 `api_key="ollama"` 来在 `smolagents` 中使用本地模型。
   - 这种集成让用户能够利用本地资源进行 Agent 工作流。
- **Manus AI 发布免费 ANUS 框架**：根据[一条推文](https://x.com/nikmcfly69/status/1898810249085145416)，**Manus AI** 推出了一款名为 **ANUS (Autonomous Networked Utility System)** 的开源框架，称其为付费解决方案的免费替代品。
   - 目前正在讨论该框架的功能细节以及它与现有付费解决方案的对比。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemma 3 激发创意 AI 崛起**：根据[这条推文](https://x.com/sam_paech/status/1899772582808969653)，新的 **Gemma-3-27b** 模型在创意写作方面排名第二，这表明它将成为创意写作和 RP（角色扮演）微调者的宠儿。
   - 一位评论者开玩笑说 *4chan 会喜欢 Gemmasutra 3*。
- **alphaXiv 结合 Claude 3.7 大获成功**：根据[这条推文](https://fxtwitter.com/askalphaxiv/status/1899833509033976194)，**alphaXiv** 使用 **Mistral OCR** 配合 **Claude 3.7**，只需点击一下即可生成包含图表、关键见解和清晰解释的研究博客。
   - 有人认为 *alphaXiv 是 HuggingFace 论文板块的正确实现方式*，提供了一个更整洁的 html.arxiv 变体。
- **Gemini Flash 的图像生成策略**：**Gemini 2.0 Flash** 现在具备原生图像生成功能，允许用户创建与上下文相关的图像、通过对话进行编辑，并在图像中生成长文本，如[这篇博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)和[推文](https://x.com/OriolVinyalsML/status/1899853815056085062)所述。
   - 根据 [X 上的帖子](https://x.com/goodside/status/1900271372732932214)，**Gemini Flash 2.0 Experimental** 还可以用于生成沃尔玛风格的照相馆肖像照。
- **中国模型权重面临安全审查**：用户对从 Hugging Face 下载像 **Deepseek** 这样的开源权重模型表示担忧，原因是潜在的安全风险，如[此讨论](https://huggingface.co/deepseek-ai)中所强调的。
   - 有人担心 *如果我从 HuggingFace 下载 Deepseek，会感染病毒吗*，或者担心 *权重会将数据发送给 CCP*，这催生了一个创业想法：将中国模型重新包装为爱国的美国或欧洲模型。
- **OpenAI 关于中国 (PRC) 模型的政策提案**：OpenAI 的[政策提案](https://openai.com/global-affairs/openai-proposals-for-the-us-ai-action-plan/)主张禁止在第一梯队国家使用 **PRC 生产的模型**，理由是这些模型 *侵犯用户隐私并产生安全风险，例如知识产权盗窃风险*。
   - OpenAI 向美国政府提交了政策提案，直接将 **合理使用 (fair use)** 与 **国家安全** 联系起来，指出如果中国拥有免费的数据访问权而美国公司缺乏合理使用权，那么 AI 竞赛实际上已经结束，根据 [Andrew Curran 的推文](https://x.com/AndrewCurran_/status/1900176516878913675)。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Distill 阅读小组宣布每月聚会**：**Distill** 阅读小组宣布下一次聚会将于 **美国东部时间 3 月 14 日 11:30-1 PM** 举行，详情见 [Exploring Explainables Reading Group 文档](https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym)。
   - 该小组的成立是由于大众对围绕 **Explainable AI**（可解释 AI）进行互动式科学交流的强烈需求。
- **Thinking Tokens 扩展 LLM 思维**：一位讨论者提议使用混合注意力模型在内部扩展 *Thinking Tokens*，使用 RNN 类型层上的内部 **TTT** 损失作为代理，并建议通过测量 **TTT** 更新损失的增量来确定“内部” **TTT** 扩展步骤的数量。
   - 该扩展在内部使用普通 Token 与普通 Token 加 Thinking Tokens 之间的交叉注意力（Cross Attention），但在不知道并行 **TTT** 损失的情况下，选择任意扩展面临挑战，这可以通过随机采样或代理模型来解决。
- **AIME 24 实现上线**：一名成员在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/lm_eval/tasks/aime24) 中添加了基于 **MATH** 实现的 **AIME24** 实现。
   - 他们将其基于 **MATH** 实现，因为他们找不到任何关于人们在运行 **AIME24** 时所使用的具体文档。
- **解密 Delphi 的激活收集**：一位成员询问了如何使用 **LatentCache** 收集用于可解释性的 **Latents**，特别是使用 **Delphi** 库时，**Latents** 是逐个 Token 获取的还是针对整个序列获取的。
   - 另一位成员澄清说，**Delphi** 通过将成批的 Token 传递给模型来收集激活，收集激活，生成类似的激活，并仅保存非零激活，并链接到了 <#1268988690047172811>。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Gemma 3 发布，支持多模态**：Google 推出了 **Gemma 3**（[免费](https://openrouter.ai/google/gemma-3-27b-it:free)），这是一款具有视觉语言输入和文本输出的多模态模型，具有 **128k Token 上下文窗口**，并增强了 **140 多种语言** 的能力。
   - 据报道，作为 **Gemma 2** 的继任者，**Gemma 3 27B** 包括增强的数学、推理、聊天、结构化输出和函数调用（Function Calling）能力。
- **Reka Flash 3 以 Apache 2.0 协议发布**：**Reka Flash 3**（[免费](https://openrouter.ai/rekaai/reka-flash-3:free)）是一款拥有 210 亿参数、**32K 上下文长度** 的 LLM，擅长通用聊天、编程、指令遵循和函数调用，通过强化学习（**RLOO**）进行了优化。
   - 该模型支持高效量化（**4-bit** 精度下低至 **11GB**），利用显式推理标签，并根据 **Apache 2.0** 协议授权，尽管它主要是一个 **英文模型**。
- **Llama 3.1 Swallow 70B 快速上线**：一款具备日语能力的新模型 **Llama 3.1 Swallow 70B**（[链接](https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3)）已发布，OpenRouter 将其描述为具有高性能的小型模型。
   - 成员们没有提供额外的补充评论。
- **Gemini 2 Flash 支持原生图像生成**：Google AI Studio 推出了 **Gemini 2.0 Flash** 的实验版本，支持原生图像输出，可通过 [Gemini API](https://ai.google.dev/gemini-api) 和 Google AI Studio 访问。
   - 这一新功能结合了多模态输入、增强的推理和自然语言理解来生成图像。
- **Cohere 发布 Command A，挑战 GPT-4o**：根据 [Cohere 博客](https://cohere.com/blog/command-a)，Cohere 推出了 **Command A**，声称在 Agent 企业任务中具有更高的效率，且性能与 **GPT-4o** 和 **DeepSeek-V3** 持平或更优。
   - 新模型优先考虑以极小的计算量完成 Agent 任务的性能，直接与 **GPT-4o** 竞争。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command A 在企业任务上挑战 GPT-4o**：Cohere 发布了 **Command A**，声称在智能体（agentic）企业任务上的表现与 **GPT-4o** 和 **DeepSeek-V3** 持平或更好，且效率更高，详情见[这篇博客文章](https://cohere.com/blog/command-a)。
   - 该模型拥有 **111b** 参数、**256k** 上下文窗口，推理速度高达 **156 tokens/sec**，可通过 API 以 `command-a-03-2025` 调用。
- **Command A 的 API 启动受故障困扰**：用户报告在使用 **Command-A-03-2025** API 时出现错误，追溯原因是模型要求中删除了 `safety_mode = “None”`。
   - 一位成员发现删除 `safety_mode` 设置解决了该问题，并指出 **Command A** 和 **Command R7B** 不再支持该设置。
- **Seed 参数未能产生一致的结果**：一位成员发现 Chat API 中的 `seed` 参数未按预期工作，在 **command-r** 和 **command-r-plus** 等模型中，相同的输入和 seed 值产生了不同的输出。
   - Cohere 团队成员[确认了该问题](https://link.to/message)并开始调查。
- **OpenAI 兼容性 API 抛出验证错误**：一位用户报告 OpenAI 兼容性 API 出现 **400 错误**，特别是在 `chat.completions` 端点和 **command-a-03-2025** 模型上，原因是 `tools` 对象中 `parameters` 字段的 Schema 验证问题。
   - Cohere 最初要求即使 `parameters` 字段为空也必须提供，但团队决定[匹配 OpenAI 的行为](https://link.to/matching)以获得更好的兼容性。
- **AI 研究员深入研究 RAG 和网络安全**：一位具有网络安全背景的 AI 研究员/开发人员正专注于 **RAG**、Agent、工作流，并主要使用 Python。
   - 他们寻求与社区建立联系并学习。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Glama API 显示每个服务器更多的数据**：一位成员分享了新的 **Glama** API ([https://glama.ai/mcp/reference#tag/servers/GET/v1/servers](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers)) 列出了所有可用工具，并且与 Pulse 相比，每个服务器的数据更多。
   - 然而，据报道 Pulse 拥有更多可用的服务器。
- **Claude 在优雅渲染图像方面遇到困难**：一位成员报告在 **Claude Desktop** 中渲染 Plotly 图像时遇到困难，找不到优雅的方法强制 **Claude** 提取资源并将其渲染为 Artifact。
   - 他们建议使用 `open` 更好，其他人指向了[一个 MCP 示例](https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35)，并指出图像出现在工具调用内部，这是 **Claude** 目前的一个局限性。
- **NPM 包缓存调查**：一位成员询问 npm 包缓存的位置以及如何在客户端显示已下载/连接的服务器。
   - 另一位成员建议检查 `C:\Users\YourUsername\AppData\Local\npm-cache`，而跟踪服务器状态的能力取决于客户端实现。
- **OpenAI Agents SDK 获得 MCP 支持**：一位开发人员将 **Model Context Protocol (MCP)** 支持集成到了 [OpenAI Agents SDK](https://github.com/lastmile-ai/openai-agents-mcp) 中，可以通过 fork 版本或 pypi 上的 `openai-agents-mcp` 包访问。
   - 此次集成允许 Agent 使用统一语法组合来自 **MCP** 服务器、本地工具、OpenAI 托管工具以及其他 Agent SDK 工具。
- **Goose 项目通过 MCP 控制计算机**：**Goose** 项目是一个开源 AI Agent，利用任何 **MCP server** 来自动化开发任务。
   - 在[这段 YouTube short](https://youtube.com/shorts/EuMzToNOQtw) 中可以观看 **Goose** 控制计算机的演示。

---

## [Notebook LM](https://discord.com/channels/1124402182171672732) Discord

- **Google 招募 NotebookLM 可用性研究参与者**：Google 正在寻找重度使用手机的 **NotebookLM 用户**，并招募用户参与可用性研究以获取产品反馈，提供 **75 美元**（或 **50 美元 Google 商品代金券**）作为补偿。
   - 感兴趣的用户可以填写[此筛选问卷](https://forms.gle/pbPDU2Dh3rEL5HLC9)参加移动端用户研究，或参加 **2025 年 4 月 2 日至 3 日**举行的 **60 分钟远程会议**。
- **NoteBookLM Plus 被考虑用于内部 FAQ**：一位用户询问是否可以将 **NoteBookLM Plus** 用作内部 FAQ，而另一位用户建议将其作为功能请求提交，因为 NotebookLM 目前不保存聊天记录。
   - 讨论的变通方案包括利用“剪贴板复制”和“笔记转换”来共享信息。
- **行内引用得到保留**：用户现在可以**将聊天回复保存为笔记**，并以原始形式**保留行内引用 (inline citations)**，从而方便地引用原始素材。
   - 许多用户请求了这一功能，这是对笔记编辑器进行一系列酷炫增强的“第一步”；不过，用户也提出了改进带有脚注的复制粘贴功能的需求。
- **Thinking Model 推送至 NotebookLM**：最新的 **Thinking Model** 已推送至 NotebookLM，承诺带来全面的质量提升，特别是对于**葡萄牙语用户**，可以在 URL 末尾添加 `?hl=pt` 来修正语言问题。
   - 用户还讨论了将 **AI Studio** 功能集成到 NotebookLM 中的可能性，该功能可以“观看” YouTube 视频，而不仅仅依赖于来自[此 Reddit 链接](https://www.reddit.com/r/singularity/comments/1j9thj9/introducing_youtube_video_link_support_in_google/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)的逐字稿。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **VectorAdd 提交结果从零恢复正常**：一名成员最初报告他们的 **vectoradd 提交**在 Google Colab 上运行正常，但返回的结果全是零。
   - 该成员随后发现代码在重复处理同一个块，导致吞吐量异常高，并指出*如果速度快得离谱，可能哪里存在 Bug*。
- **SYCL 作为 CUDA 挑战者脱颖而出**：关于 **SYCL** 的可移植性以及 **AdaptiveCpp** 和 **triSYCL** 等实现的讨论显示，**Intel** 是关键利益相关者。
   - 一位参与者认为 SYCL 比 HIP 更有趣，因为*它不仅仅是 CUDA 的克隆，因此可以改进设计*。
- **Deepseek 的 MLA 创新**：DataCrunch 在[其博客文章](https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention)中详细介绍了 Deepseek V3 和 R1 模型中带有权重吸收的 **Multi-Head Latent Attention (MLA)** 实现。
   - 一位成员根据[这个 Pull Request](https://github.com/flashinfer-ai/flashinfer/pull/551#issuecomment-2665697147) 发现 vLLM 当前的默认设置效果不佳。
- **Reasoning-Gym 课程吸引 ETH 和 EPFL 关注**：来自 **ETH** 和 **EPFL** 的团队正在合作开发用于 SFT、RL 和 Eval 的 **reasoning-gym**，并研究 **RL 的自动课程学习 (auto-curriculum)**，初步结果可在 [GitHub](https://github.com/open-thought/reasoning-gym/blob/curriculum_refactor/reasoning_gym/principal.py#L66) 上查看。
   - 该团队还寻求与 [Evalchemy](https://github.com/mlfoundations/Evalchemy) 集成，以实现 LLM 的自动评估。
- **FlashAttention 移植至 Turing 架构**：一位开发者为 Turing 架构实现了 FlashAttention 前向传播（此前仅限于 Ampere 和 Hopper），代码已在 [GitHub](https://github.com/ssiu/flash-attention-turing) 上发布。
   - 早期基准测试显示，在特定条件下（`head_dim = 128`，原生 Attention，且 `seq_len` 可被 128 整除），在 **T4** 上比 Pytorch 的 `F.scaled_dot_product_attention` 有 **2 倍的速度提升**。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **YC 支持快速获利而非独角兽**：一名成员声称 **YC** 优先考虑具有短期成功的初创公司，投资 **$500K** 旨在 **6 个月** 内实现 **3 倍** 回报，而不是专注于长期增长。
   - 他们认为 **YC** 多年来没有产生过著名的独角兽，这表明其可能已从培养长期成功案例转向其他方向。
- **LLM 缩放近似上下文无关语言**：一种理论认为，**LLM 缩放** 可以通过其使用概率 FSA 近似上下文无关语言的能力来理解，从而产生如[此附图](https://cdn.discordapp.com/attachments/986699377257119794/1349416392021119047/20250312_111842.jpg?ex=67d456f2&is=67d30572&hm=2492956c61fb86b79264d1863fb121f787cecf87ab855f65f21439471a6217fb)中所示的特征 S 曲线模式。
   - 该提议认为 **LLMs** 试图从 Chomsky 层级的较低层级出发，去近似更高层级的语言。
- **Google 的 Gemma 3 面临 ChatArena 质疑**：Google 发布了 **Gemma 3**，正如[官方文档](https://ai.google.dev/gemma/docs/core)中所述，据报道其性能与 **Deepseek R1** 相当，但体积显著更小。
   - 一名成员指出，提供的基准测试是用户偏好基准（**ChatArena**），而非非主观指标。
- **提出通用状态机概念**：一名成员分享了一个具有动态增长的[基于图的系统](https://x.com/renxyzinc/status/1899539629411270758)，称其为 **Universal State Machine (USM)**，并指出这是一个非常幼稚的系统，优化较差且节点数量爆炸。
   - 他们链接了一篇[介绍性论文](https://opensource.getren.xyz/ittm/)，将 **Infinite Time Turing Machines (ITTMs)** 描述为理论基础，并将 **Universal State Machine (USM)** 描述为实际实现，为可扩展、可解释且可泛化的机器提供了路线图。
- **RTX Remix 重燃《瑞迪克》梦想**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=j31ISEd8xRM)，展示了具有全光线追踪和 DLSS 4 的 **Half-Life 2 RTX** 演示，通过 **RTX Remix** 重新构思。
   - 另一名成员表达了对《超世纪战警：逃离屠夫湾》（Chronicles of Riddick: Escape from Butcher Bay）RTX 版本的期待。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **GPT-4 依然优于本地 LLM**：一位用户发现 **ChatGPT premium** 的质量显著超过 **GPT4All 上的 LLM**，将其归因于本地可用的模型尺寸较小，并希望本地模型在处理上传文档时的准确性能够与之匹配。
   - 该用户指出，他们在 **GPT4All** 上尝试的模型在处理文档上传时不够准确。
- **Ollama 与 GPT4All 的选择**：一位用户询问在管理多个模型、快速加载/卸载、频繁更新 **RAG** 文件以及日期/时间/天气 API 的服务器上，应该使用 **GPT4All** 还是 **Ollama**。
   - 一名成员建议使用 **Deepseek 14B** 或类似模型，同时提到 **large context windows**（4k+ token）对于吸收文档等更多信息的重要性，并评论说 Apple 硬件比较特殊。
- **GPT4All 工作流不错，但 GUI 较差**：一名成员建议使用带有微型模型的 **GPT4All** 来检查加载、卸载和使用 **LocalDocs** 进行 **RAG** 的工作流，但指出 GUI 不支持同时运行多个模型。
   - 他们建议使用本地服务器或 Python 端点，这需要为流水线和编排编写自定义代码。
- **爬取 Brave 网络**：一位用户询问如何让网页爬取工作，并在开始尝试前寻求建议。
   - 一名成员提到一个 **Brave 浏览器** 兼容性 PR，由于存在 Bug 以及转向不同的 tool-calling 方法而未被合并，但如果有需求可以重新启用。
- **LocalDocs 纯文本变通方案**：一名成员建议，为了解决 **LocalDocs** 以纯文本显示片段的问题，用户可以截屏保存为 PDF，对图像进行 OCR，然后在数据库中搜索该片段。
   - 他们建议在这个工作流中使用 *docfetcher*。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Mastra 发布 Typescript AI 框架**：[Mastra](https://mastra.ai/) 是一款全新的 Typescript AI 框架，旨在为产品开发者提供强大的框架，其定位优于 Langchain 等框架。
   - 创始人拥有 Gatsby 和 Netlify 背景，强调了 **type safety** 以及对量化性能提升的关注。
- **Gemini 2.0 Flash 生成图像**：**Gemini 2.0 Flash Experimental** 现在支持 [原生图像生成](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)，能够根据文本和多模态输入创建图像，从而增强其推理能力。
   - 用户反应惊人，有人表示“*对其效果之好简直无以言表*”，另一位则评论说它为“BASE”一词增添了“D”。
- **Jina AI 微调 DeepSearch**：Jina AI 分享了 [增强 DeepSearch/DeepResearch 的技术](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)，特别是用于片段选择的 **late-chunking embeddings** 以及在爬取前优先排序 URL 的 **rerankers**。
   - 他们表达了对 Latent Space 播客的热情，表示“*我们今年一定要邀请他们参加*”。
- **Cohere 的 Command 模型开放权重**：Cohere 推出了 [Command A](https://x.com/aidangomez/status/1900169306987524440)，这是一个拥有 **111B 参数的开放权重模型**，具备 **256k context window**，专为 agentic、多语言和编程应用量身定制。
   - 该模型是 Command R+ 的继任者，旨在各项任务中表现出更优越的性能。
- **Gemini 向所有人免费提供 Deep Research**：**Gemini App** 现在向 [所有用户免费提供 Deep Research](https://x.com/OfficialLoganK/status/1900224377389465751)，由 **Gemini 2.0 Flash Thinking** 提供支持，并结合搜索历史提供个性化体验。
   - 此次更新让更广泛的受众能够使用先进的推理功能。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 支持 Model Context Protocol**：根据 [这条推文](https://twitter.com/llama_index/status/1899848532817035529)，LlamaIndex 现在支持 **Model Context Protocol**，允许用户使用任何 **MCP-compatible server** 提供的工具。
   - **Model Context Protocol** 是一项开源计划，旨在简化工具的发现和使用。
- **AI 将颠覆 Web 开发**：专家们将齐聚 @WeAreDevs WebDev & AI Day，探讨 **AI 对平台工程和 DevEx 的影响**，以及 AI 驱动环境下开发者工具的演变，详见 [这条推文](https://twitter.com/llama_index/status/1900232326132773026)。
   - 该活动将聚焦于 AI 如何重塑开发者体验。
- **LlamaParse 成为 JSON 强大工具**：**LlamaParse** 现在将其 JSON 输出中包含图像，提供可下载的图像链接和布局数据，[详情点击此处](https://github.com/run-llama/llama_index/pull/18112)。
   - 这一增强功能实现了更全面的文档解析和重构。
- **Deep Research RAG 准备就绪**：**RAG** 中的 Deep research 功能可通过带有 deep research 选项的 `npx create-llama@latest` 获取，工作流源代码可在 [GitHub](https://github.com/run-llama/create-llama/blob/ee69ce7cc10db828424b468e7b54b3f06b18e22c/templates/components/agents/python/deep_research/app/workflows/deep_research.py) 上找到。
   - 此设置有助于使用 **RAG** 进行深入的探索性研究。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **MOOC 测验截止日期定于 5 月**：成员报告称所有 **quiz deadlines** 都在 **5 月**，详情将很快发布给邮件列表中的人员，记录显示他们已打开关于 **Lecture 6** 的最新邮件。
   - 社区应关注 **weekly quizzes** 并等待进一步消息。
- **MOOC 实验与研究机会即将公布**：针对 **MOOC** 学习者的 **labs** 和 **research opportunities** 计划正在制定中，关于 **projects** 的细节即将公布。
   - 一旦一切敲定，将发布公告，包括非 Berkeley 学生是否可以获得认证的信息。
- **阐明 LLM 中的 Roles 与 Personas**：在查询 LLM 时，**roles** 是用于编辑 prompt 的构造，如 **system**、**user** 或 **assistant**，而 **persona** 被定义为提供给系统的通用指南的一部分，影响 assistant 的行为方式。
   - **system role** 提供通用指南，而 **user** 和 **assistant** 角色是活跃的参与者。
- **决策研究小组需要你**：一个专注于 **decision making** 和 **memory tracks** 的研究小组已开放。
   - 加入 [Discord 研究小组](https://discord.gg/pqWzyfCX) 以深入探讨该话题。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 与 Max 捆绑？**：在 [Modular 论坛](https://forum.modular.com/t/mojo-and-max-why-bundle-them/751)中，用户正在讨论将 **Mojo** 与 **Max** 捆绑的潜在协同效应和益处。
   - 讨论围绕用户利益以及此类捆绑包的潜在用例展开。
- **Mojo 何时支持 Windows？**：社区对 **Mojo 在 Windows 上的潜在可用性**表现出浓厚兴趣。
   - 社区讨论了扩展 **Mojo 平台支持**所面临的挑战和时间表。
- **Modular Max 增加进程生成功能**：一位成员分享了一个针对 **Modular Max** 的 [PR](https://github.com/modular/max/pull/3998)，该 PR 增加了使用 `exec` 从可执行文件生成和管理进程的功能。
   - 由于依赖于合并 foundations PR 以及解决 **Linux exec** 的问题，其可用性尚不确定。
- **闭包捕获引发关注**：一位成员提交了一个与 `capturing` 闭包相关的 [语言设计 Bug](https://github.com/modular/max/issues/4143)。
   - 另一位成员表示赞同，指出他们也觉得这种行为很奇怪。
- **Missing MutableInputTensor 困扰 Max 用户**：一位用户报告在 [nightly 文档](https://docs.modular.com/max/api/mojo/tensor/managed_tensor_slice/)中发现了 `MutableInputTensor` 类型别名，但它似乎并未公开。
   - 该用户尝试通过 `from max.tensor import MutableInputTensor` 和 `from max.tensor.managed_tensor_slice import MutableInputTensor` 进行导入，但均未成功。



---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **AST 准确率评估 LLM 调用**：**AST**（抽象语法树）评估检查函数调用是否正确，包括函数名称、参数类型以及 [V1 博客](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics)中注明的可能范围内的参数值。
   - **AST** 的数值代表所有这些标准都正确的**测试用例百分比**，揭示了 **LLM** 函数调用的准确性。
- **BFCL 更新首个全面的 LLM 评估**：**[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL)** 最近一次更新于 **2024-08-19**，是对 **LLM** 调用函数和工具能力的全面评估（[变更日志](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CHANGELOG.md)）。
   - 该排行榜旨在反映 **Agent** 和企业工作流中典型的用户函数调用用例。
- **通过函数调用增强的 LLM**：**GPT**、**Gemini**、**Llama** 和 **Mistral** 等大语言模型（**LLM**）正越来越多地通过函数调用功能应用于 **Langchain**、**Llama Index**、**AutoGPT** 和 **Voyager** 等应用中。
   - 这些模型通过函数调用（也称为工具调用）在应用程序和软件中具有巨大的潜力。
- **并行运行函数调用**：评估包括各种形式的函数调用，例如**并行**（一个函数输入，多次调用函数输出）和**多个**函数调用。
   - 这种全面的方法涵盖了常见的函数调用用例。
- **追踪所有评估工具的中心位置**：数据集位于 **/gorilla/berkeley-function-call-leaderboard/data**，对于多轮对话类别，函数/工具文档位于 **/gorilla/berkeley-function-call-leaderboard/data/multi_turn_func_doc**。
   - 所有其他类别将函数文档存储在数据集文件内。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 计划推出可插拔缓存模块**：DSPy 正在开发一个**可插拔的 Cache 模块**，初步工作可在[此 PR](https://github.com/stanfordnlp/dspy/pull/1922)中查看。
   - 新功能旨在拥有一个统一的缓存接口，包含两级缓存：内存 **LRU cache** 和 **fanout**（磁盘）。
- **缓存策略寻求灵活性**：用户希望在定义**缓存策略**方面有更多灵活性，特别是通过**上下文缓存**来降低成本并提高速度，并对具有 **TTL 过期**或 **LRU 淘汰**机制的**缓存失效**感兴趣。
   - 还讨论了基于**输入相似度**的**选择性缓存**，以避免进行冗余的 API 调用，以及内置的**缓存命中/未命中率监控**。
- **ColBERT 端点连接被拒绝**：一名成员报告位于 `http://20.102.90.50:2017/wiki17_abstracts` 的 **ColBERT 端点**似乎已关闭，抛出 *Connection Refused* 错误。
   - 当尝试使用基础的 **MultiHop 程序**检索段落时，端点返回 **200 OK** 响应，但文本包含与连接 `localhost:2172` 相关的错误消息。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **LSTM 模型受 NaN 损失困扰**：一名成员报告在使用 **TinyJit** 运行 **LSTMModel** 时遇到 **NaN** 损失，观察到损失在第一步后从一个很大的数值跳变为 **NaN**。
   - 模型设置涉及 `nn.LSTMCell` 和 `nn.Linear`，使用 `Adam` 优化器进行优化，输入数据包含一个较大的值（**1000**），这可能是原因所在。
- **调试 NaN**：一名成员请求协助调试 **tinygrad** 训练期间的 **NaN** 损失，并提供了一个展示 **LSTM** 设置的代码示例。
   - 这表明数值不稳定或梯度爆炸问题可能是原因。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Pinecone 性能受限**：一名成员报告他们的 **RAG 系统**在使用 **Pinecone** 时面临**性能限制**。
   - 此外，**Pinecone 缺乏 VPC 部署支持**也是一个主要问题。
- **RAG 系统弃用 Pinecone**：由于**性能瓶颈和缺乏 VPC 部署支持**，一个 **RAG 系统**正在弃用 **Pinecone**。
   - 工程师预计新设置将缓解这两个问题。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长期没有更新，请告知我们，我们将将其移除。

---

**Codeium (Windsurf) Discord** 没有新消息。如果该频道长期没有更新，请告知我们，我们将将其移除。

---

# 第 2 部分：各频道详细摘要与链接


{% if medium == 'web' %}




### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1349340799485345812)** (1134 条消息🔥🔥🔥): 

> `Claude 3.7 API 更新、Cursor 卡顿/不稳定问题、Manus 的开源替代方案、针对 Blender 的 MCP、Cursor 更新与版本混淆` 


- **Claude 的缓存 API 降低 90% 成本**：Anthropic 正在为 **Claude 3.7 Sonnet** 推出 [新的 API 更新](https://www.anthropic.com/news/prompt-caching)，提供 **缓存感知速率限制 (cache-aware rate limits)** 和更简单的 Prompt 缓存，这可以将长 Prompt 的成本降低高达 **90%**，并将延迟降低 **85%**。
   - 这使得 Claude 能够保持对大型文档、指令或示例的记忆，而无需在每次请求时重新发送信息。
- **Cursor 用户抱怨运行缓慢、窗口崩溃**：用户报告在最近的 Cursor 版本中出现了 **UI 反应迟钝、频繁的窗口崩溃和内存泄漏**，特别是在 Mac 和 Windows WSL2 上，有人猜测这与 Microsoft 的法律问题有关。
   - 一些成员建议尝试使用 Cline 或 Windsurf 作为替代方案。
- **开源版 Manus 在 GitHub 上涌现**：名为 [OpenManus](https://github.com/mannaandpoem/OpenManus) 的 **Manus** 开源替代方案引发了热烈讨论，涉及其潜力和与 Manus 的对比，一些用户甚至在尝试 [此 YouTube 视频](https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf) 中展示的演示。
   - 该项目旨在提供一个比 Manus 更易于获取的替代方案，引发了关于其功能以及对非技术用户易用性的讨论。
- **Blender 获得了 MCP 支持**：一位成员重点介绍了 [针对 Blender 的 MCP](https://x.com/sidahuj/status/1899460492999184534)，使 Claude 能够直接与 Blender 交互，通过 Prompt 创建 3D 场景。
   - 这激发了人们将 AI 工具集成扩展到传统编程任务之外的兴趣。
- **Cursor 更新混乱，版本混淆盛行**：一场关于 Cursor 版本的混乱辩论爆发了，一些用户吹嘘并不存在的 **0.49**、**0.49.1** 甚至 **1.50** 版本，而另一些人则在 **0.47** 版本的崩溃中挣扎，导致了关于恶意挑衅和误导信息的指责。
   - 这种混乱源于不同的更新体验，一些用户通过非官方渠道获取 Beta 版本，进一步搅浑了局面。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://manus.im/share/YIRZaLU">Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://manus.im/share/dGyBB8MInk2iJPyQuTE0nr?replay=1">Augmentin 625 mg Dosage Guidelines for Adults - Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://x.com/Trae_ai/status/1899720953216782781">Tweet from Trae (@Trae_ai)</a>: 🚀 连接更多，交付更多！今天的 Trae 更新带来了：- 自定义模型集成现已上线！- 针对 Ubuntu 20/22/24 和 Debian 11/12 的远程 SSH 支持。更多功能即将推出。#DevTools #AI #T...</li><li><a href="https://x.com/sidahuj/status/1899460492999184534">Tweet from siddharth ahuja (@sidahuj)</a>: 🧩 构建了一个 MCP，让 Claude 能直接与 Blender 对话。它能帮你仅通过提示词创建精美的 3D 场景！这是我仅用几分钟创建“低多边形巨龙守护宝藏”场景的演示...</li><li><a href="https://x.com/opentools_/status/1900200185466163483?s=46&t=CLGnxOi5OPp22iT8UYkr1A">Tweet from OpenTools (@opentools_)</a>: 我们很高兴分享工具调用 API 的 Beta 版本！现在开发者可以轻松地为任何 LLM 配备托管的开源工具，用于网页搜索、网页爬取和地图数据（更多功能即将推出）。Under...</li><li><a href="https://t.co/jbcnZ95Ct4">Token-saving updates on the Anthropic API</a>: Anthropic API 的节省 Token 更新：我们对 Anthropic API 进行了多项更新，让开发者在使用 Claude 3.7 Sonnet 时能显著提高吞吐量并减少 Token 使用。</li><li><a href="https://downloads.cursor.com/production/client/linux/x64/appimage/Cursor-0.47.3-dab1538cd064aebd4292f9de48b05022c974aff6.deb.glibc2.25-x86_64.AppImage">no title found</a>: 未找到描述</li><li><a href="https://x.com/OfficialLoganK/status/1899914266062577722">Tweet from Logan Kilpatrick (@OfficialLoganK)</a>: 在 Google AI Studio 和 Gemini API 中引入 YouTube 视频 🎥 链接支持。你现在可以直接传入 YouTube 视频，模型可以利用其原生的视频理解能力来...</li><li><a href="https://manus.im/share/YIRZaLUfghVxGCN7dE6hbI?replay=1">Customer Form for B2B Gen AI Consulting Firms - Manus</a>: Manus 是一个通用的 AI Agent，能将你的想法转化为行动。它擅长处理工作和生活中的各种任务，在你休息时搞定一切。</li><li><a href="https://x.com/mckaywrigley/status/1898756745545252866?s=46&t=CLGnxOi5OPp22iT8UYkr1A">Tweet from Mckay Wrigley (@mckaywrigley)</a>: 观看我第一次使用 Manus 的 14 分钟演示。它好得令人震惊。现在想象一下 2-3 年后：- 它拥有 >180 的 IQ - 永不停歇地工作 - 速度快 10 倍 - 并且以成千上万的集群运行...</li><li><a href="https://downloads.cursor.com/production/dab1538cd064aebd4292f9de48b05022c974aff6/darwin/universal/Cursor-darwin-universal.dmg">no title found</a>: 未找到描述</li><li><a href="https://openrouter.ai/api/v1"">Discord</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/CLine/comments/1j6fp1o/initial_modular_refactor_now_on_github_cline/">Reddit - Heart of the internet</a>: 未找到描述</li><li><a href="https://github.com/mannaandpoem/OpenManus">GitHub - mannaandpoem/OpenManus: No fortress, purely open ground.  OpenManus is Coming.</a>: 没有堡垒，纯粹的开放地带。OpenManus 即将到来。 - mannaandpoem/OpenManus</li><li><a href="https://tenor.com/view/smart-thinking-thoughts-think-ponder-gif-18050532214954774978">Smart Thinking GIF - Smart Thinking Thoughts - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/oslook/cursor-ai-downloads?tab=readme-ov-file">GitHub - oslook/cursor-ai-downloads: All Cursor AI's official download links for both the latest and older versions, making it easy for you to update, downgrade, and choose any version. 🚀</a>: 所有 Cursor AI 的官方下载链接，包括最新版本和旧版本，方便你升级、降级和选择任何版本。🚀 - oslook/cursor-ai-downloads</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: 新的更新和改进。</li><li><a href="https://tenor.com/view/idiocracy-i-dont-know-you-know-gif-7477932">Idiocracy I Dont Know GIF - Idiocracy I Dont Know You Know - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtube.com/shorts/P24II7txEkQ?si=wme7NF0qWTmd7UJp"> - YouTube</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/claude-3-7-thinking-permanently-high-load/62928">Claude 3.7-thinking permanently 'High Load'!</a>: Claude 3.7-thinking 永久处于“高负载”状态！！！在过去的 4 小时里我尝试了数百次，它一直处于这种状态！！昨天一整天都运行良好...</li>

i><a href="https://github.com/oslook/cursor-ai-downloads">GitHub - oslook/cursor-ai-downloads: 所有 Cursor AI 官方下载链接，包括最新版本和旧版本，方便您进行升级、降级和选择任何版本。 🚀</a>: 所有 Cursor AI 官方下载链接，包括最新版本和旧版本，方便您进行升级、降级和选择任何版本。 🚀 - oslook/cursor-ai-downloads</li><li><a href="https://github.com/jamesliounis/servers/tree/james-perplexity/add-perplexity-mcp-server">GitHub - jamesliounis/servers 分支 james-perplexity/add-perplexity-mcp-server</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号为 jamesliounis/servers 的开发做出贡献。</li><li><a href="https://github.com/jamesliounis/servers/blob/f9dd1b55a4ec887878f0770723db95d493c261a2/src/perplexity-ask/README.md">servers/src/perplexity-ask/README.md 位于 f9dd1b55a4ec887878f0770723db95d493c261a2 · jamesliounis/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号为 jamesliounis/servers 的开发做出贡献。</li><li><a href="https://forum.cursor.com/">Cursor - 社区论坛</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://youtu.be/H1rWVvsjtTQ?si=iP4MQXcHWfzxRzTf"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1349420471384277013)** (2 条消息): 

> `Inference API 发布，Hermes 3 Llama 70B，DeepHermes 3 8B Preview，Hybrid Reasoners，DeepHermes 24B` 


- **Nous Research 发布 Inference API**: Nous Research 发布了其 Inference API，提供对 **Hermes 3 Llama 70B** 和 **DeepHermes 3 8B Preview** 等语言模型的访问，并计划推出更多模型。
   - 该 API 与 **OpenAI 兼容**，并在 [Nous Portal](https://portal.nousresearch.com/login) 设有等待名单系统，为新账户提供 **$5.00** 的免费额度。
- **DeepHermes 24B 和 3B Preview 发布**: 宣布推出设计为 Hybrid Reasoners 的 **DeepHermes 24B** 和 **3B Preview** 模型，具有长思维链（long chain of thought）推理的开关切换功能，可通过 API 和 HuggingFace 获取 ([DeepHermes 24B](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview), [DeepHermes 3B](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview))。
   - **24B** 模型在开启推理模式时，在难题数学上的准确率提高了 **4 倍**，在 **GPQA** 上提高了 **43%**。
- **提供 GGUF 量化版 DeepHermes 模型**: 提供 **DeepHermes 24B** 和 **3B** 模型的 **GGUF 量化版本**以实现高效推理，提供不同的量化级别 ([24B GGUF](https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview-GGUF), [3B GGUF](https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview-GGUF))。
   - 量化选项包括 **Q4, Q5, Q6, 和 Q8**，文件大小从 **1.8G** 到 **24G** 不等。
- **DeepHermes 24B 聊天机器人在 Discord 上线**: Nous Research Discord 服务器上提供了一个**免费**且**互动**的 **DeepHermes 24B 聊天机器人**。
   - 聊天机器人可在 **#general** 频道中使用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://portal.nousresearch.com/login">Nous Portal</a>: 未找到描述</li><li><a href="https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add">DeepHermes - NousResearch 集合</a>: 未找到描述</li><li><a href="https://portal.nousresearch.com">Nous Portal</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview">NousResearch/DeepHermes-3-Mistral-24B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview">NousResearch/DeepHermes-3-Llama-3-3B-Preview · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Mistral-24B-Preview-GGUF">NousResearch/DeepHermes-3-Mistral-24B-Preview-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/DeepHermes-3-Llama-3-3B-Preview-GGUF">NousResearch/DeepHermes-3-Llama-3-3B-Preview-GGUF · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1349407068842627083)** (684 条消息🔥🔥🔥): 

> `LLM 面部记忆系统，Inference API 额度预充值，带有开源代码的图推理系统，图论，Gemma-3 与 LM Studio 集成`

- ****LLM 学习人脸识别****：一位成员开源了一个对话系统 [LLM Facial Memory System](https://github.com/yaya-labs/LLM_Facial_Memory_System)，该系统将人脸识别与 LLM 集成，使其能够记住人并根据人脸维护聊天历史。
   - 据创作者称，该项目是为工作而构建的，并在获得许可后发布。
- ****预充值 Inference API：告别信用卡风险！****：用户讨论了为 Inference API 预充值额度，其中一人表达了对 API key 泄露的担忧，更倾向于预充值有限的金额（如 **$50**），而不是冒着产生巨额意外费用的风险。
   - 成员们确认 Inference API 目前*仅*支持预充值，其定价预计将按成本基准设定。
- ****Gemma-3 在 LM Studio 上线****：**LM Studio 0.3.13** 现在支持 Google 的 [Gemma-3](https://ai.google.dev/gemma/docs/core) 模型，包括多模态（文本 + 图像输入）模型，适用于 GGUF 和 MLX 模型。
   - 不过，一些用户报告在尝试下载 Linux 版本的 LM Studio 时遇到 **404 错误**，该问题现已解决。
- ****DeepHermes 具备混合推理能力****：Nous Research 发布了新的 [DeepHermes Preview 模型](https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add)，包括 **24B** 和 **3B** 版本。它们是混合推理器（Hybrid Reasoners），允许用户开启或关闭长 chain of thought 推理。
   - 这些模型使用与 **8B** DeepHermes 完全相同的配方，且仅基于 SFT，但即使在没有推理的情况下，它在数学方面也有一些溢出表现。
- ****Zero-Shot 分类器大显身手****：一位正在为社交媒体帖子寻找 Embedding 模型的用户被建议考虑使用像 [ModernBERT-large-zeroshot-v2.0](https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0) 这样的 Zero-Shot 分类器，这表明对于项目分组，离散类别可能比 Embedding 更合适。
   - 该模型平均性能略逊于 DeBERTa v3，但速度非常快且内存效率高。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://opensource.getren.xyz/ittm/8_usm.html">8&nbsp; Universal State Machine – Infinite Time Turing Machines and their Applications</a>: 未找到描述</li><li><a href="https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0">MoritzLaurer/ModernBERT-large-zeroshot-v2.0 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-3-12b-pt">google/gemma-3-12b-pt · Hugging Face</a>: 未找到描述</li><li><a href="https://fxtwitter.com/renxyzinc/status/1899539629411270758?t=odgMN8v">来自 Ren (@renxyzinc) 的推文</a>: 观看 Universal State Machine (USM) 的首次公开演示——这是一种革命性的人工智能方法，重新定义了机器如何从经验中学习。</li><li><a href="https://x.com/giffmana/status/1899950076002226411?t=1Eovk_2ocqI3LM2lxShGAg&s=19">来自 Lucas Beyer (bl16) (@giffmana) 的推文</a>: 哈哈哈哈，看看我收到了什么样的梗图：</li><li><a href="https://x.com/NousResearch/status/1900218445763088766">来自 Nous Research (@NousResearch) 的推文</a>: 发布最新的 DeepHermes 预览模型，DeepHermes 24B 和 3B！https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add 这些新模型是 Hybrid Reasoners —— 意味着...</li><li><a href="https://fxtwitter.com/eliebakouch/status/1899790607993741603">来自 elie (@eliebakouch) 的推文</a>: Gemma3 技术报告详细分析 💎1) 架构选择：&gt; 不再使用 softcaping，替换为 QK-Norm &gt; 同时包含 Pre 和 Post Norm &gt; 比 Qwen2.5 更宽的 MLP，深度大致相同 &gt; 5:1 的 SWA 以及...</li><li><a href="https://github.com/ga">Geometric Algebra</a>: Geometric Algebra 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/gabrielolympie/moe-pruner/issues/1">哪里可以下载轻量级模型 · Issue #1 · gabrielolympie/moe-pruner</a>: 你的工作太棒了！请问在哪里可以下载轻量级模型？谢谢！</li><li><a href="https://youtu.be/Sln1n3Jba_U?si=INYkHLtsNLaCmoM_">来自 CRYSTAL (MIT) 的结合 AI Agents 的知识图谱 (Knowledge Graphs)</a>: 知识图谱是一种结构化的信息表示，由通过关系（边）连接的实体（节点）组成。它作为一个动态的框架...</li><li><a href="https://github.com/ai-in-pm/Forest-of-Thought">GitHub - ai-in-pm/Forest-of-Thought: Forest-of-Thought: Scaling Test-Time Compute for Enhancing LLM Reasoning</a>: Forest-of-Thought: 扩展测试时计算以增强 LLM 推理 - ai-in-pm/Forest-of-Thought</li><li><a href="https://github.com/ashishpatel26/sot">GitHub - ashishpatel26/sot: Official code repository for Sketch-of-Thought (SoT)</a>: Sketch-of-Thought (SoT) 的官方代码仓库 - ashishpatel26/sot</li><li><a href="https://youtu.be/Ey5Q-3DNbyk?si=IciT-_jQ8GoOGFVa">ECCHI SHIYOU</a>: 未找到描述</li><li><a href="https://github.com/gabrielolympie/moe-pruner">GitHub - gabrielolympie/moe-pruner: A repository aimed at pruning DeepSeek V3, R1 and R1-zero to a usable size</a>: 一个旨在将 DeepSeek V3、R1 和 R1-zero 剪枝到可用大小的仓库 - gabrielolympie/moe-pruner</li><li><a href="https://github.com/yaya-labs/LLM_Facial_Memory_System">GitHub - yaya-labs/LLM_Facial_Memory_System: A conversational system that integrates facial recognition capabilities with large language models. The system remembers the people it interacts with and maintains a conversation history for each recognised face.</a>: 一个将人脸识别功能与大语言模型 (LLM) 集成的对话系统。该系统能够记住与其交互的人，并为每个识别出的人脸维护对话历史。</li><li><a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://docs.github.com/en/copilot/managing-copilot/managing-copilot-as-an-individual-subscriber/managing-your-github-copilot-pro-subscription/getting-free-access-to-copilot-pro-as-a-student-teacher-or-maintainer">作为学生、教师或维护者免费获取 Copilot Pro 访问权限 - GitHub Docs</a>: 未找到描述</li><li><a href="https://github.com/vllm-project/vllm/blob/f53a0586b9c88a78167157296555b7664c398055/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py#L99">vllm/vllm/entrypoints/openai/tool_parsers/hermes_tool_parser.py at f53a0586b9c88a78167157296555b7664c398055 · vllm-project/vllm</a>: 一个针对 LLM 的高吞吐量且内存高效的推理和推理服务引擎 - vllm-project/vllm</li><li><a href="https://fxtwitter.com/renxyzinc/status/1899539629411270758?t=odgMN8vrW1gMrlCJtU2FiQ&s=19">来自 Ren (@renxyzinc) 的推文</a>: 观看 Universal State Machine (USM) 的首次公开演示——这是一种革命性的人工智能方法，重新定义了机器如何从经验中学习。
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1349661543524208730)** (1 条消息): 

> `AI Compilers, Deep Learning Compilation` 


- **深度学习编译器话题引发关注**：一名成员询问了关于深度学习 **AI Compilers** 的论文资源。
   - 他们正在寻求关于 **深度学习编译** 领域重要方面和潜在挑战的指导。
- **需要更多关于该话题的资源**：一名成员询问了关于深度学习 **AI Compilers** 的论文资源。
   - 他们正在寻求关于 **深度学习编译** 领域重要方面和潜在挑战的指导。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1349666450172018709)** (2 条消息): 

> `Sakana AI, Model Memorization` 


- **Sakana AI 发布首篇论文**：[Sakana AI](https://sakana.ai/ai-scientist-first-publication/) 发布了其第一篇出版物。
   - 一名成员注意到此事，并调侃模型正在*从其训练数据中学习*。
- **Sakana AI 的第一张图片**：Sakana AI 发布了一张图片。
   - 该图片作为 image-104.png 添加在 Discord 上。



**提到的链接**：<a href="https://sakana.ai/ai-scientist-first-publication/">未找到标题</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1349414180632920194)** (11 条消息🔥): 

> `Audio-Flamingo-2, Agent Engineering` 


- ****Audio-Flamingo-2** 的混合音乐分析**：一名成员报告了在 [HuggingFace 上使用 Audio-Flamingo-2](https://huggingface.co/spaces/nvidia/audio-flamingo-2) 的混合结果，指出其歌曲描述不错，但 BPM 检测不准确。
   - 具体来说，当被问及 *"这首歌是什么调式和调性？"* 时，该模型错误地将 **Lorde 的 Royals** 的调性识别为 F# 小调。
- **Agent 工程文章引发讨论**：一名成员分享了一篇关于 [“Agent Engineering” 的博客文章](https://neuralniche.com/posts/agent-engineering/5-on-agents/)并征求反馈，强调了在理解和设计面向现实世界应用的 Agent 时存在的模糊性。
   - 文章指出，虽然 Agent 是 **2024 年的流行词**，但它们在现实世界中的应用并未达到预期，要让它们像浏览器一样普及，其路径仍不明朗。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/nvidia/audio-flamingo-2">Audio Flamingo 2 - nvidia 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://neuralniche.com/posts/agent-engineering/5-on-agents/">关于 Agent 工程的 5 点思考</a>：深入探讨关于 Agent 工程未来的原始想法。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1349666450172018709)** (2 条消息): 

> `Sakana AI, AI Training Data` 


- **Sakana AI 携论文首次亮相**：[Sakana AI](https://sakana.ai/ai-scientist-first-publication/) 以其**首篇论文**标志着其进入 AI 研究领域。
   - 一名成员调侃道，这个 AI 似乎正在*从其训练数据中学习*。
- **图片引发讨论**：分享了一张图片（[image-104.png](https://cdn.discordapp.com/attachments/1104063238934626386/1349666552823418901/image-104.png?ex=67d3ee6d&is=67d29ced&hm=25d2473dfda06700d2e45a0a95375fe7590869a732a0464bf6c621f653dde2da&)），引发了分析和讨论。



**提到的链接**：<a href="https://sakana.ai/ai-scientist-first-publication/">未找到标题</a>：未找到描述

  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1349341114208878604)** (503 条消息🔥🔥🔥): 

> `Gemma 3, GGUF, Transformers issue, RLHF, H100` 


- **Gemma 3 GGUF 版本现已发布**：Unsloth 已将 **Gemma 3** 的所有 **GGUF**、**4-bit** 和 **16-bit** 版本上传至 [Hugging Face collection](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b)。
   - 上传的版本包括 **2-8 bit GGUFs**、**dynamic 4-bit** 和 **16-bit** 版本，但成员们注意到这些模型目前尚不支持图像，正在开发修复方案。
- **Transformers Bug 导致 Gemma 3 微调中断**：**Hugging Face Transformers** 中的一个重大 Bug 阻碍了 **Gemma 3** 的微调，但 HF 正在积极修复，且 [博客文章已更新](https://unsloth.ai/blog/gemma3)。
   - 该 Bug 不仅影响 **Gemma 3** 的微调，还导致 Colab 上的 Jupyter notebook 中 **Gemma 3** 的实际设置与文档不符。
- **向初学者解释 GGUF**：**GGUF** 是模型的量化版本，旨在运行于使用 **llama.cpp** 的程序中，例如 **LM Studio** 和 **GPT4All**。
   - 上传的 **Gemma 3** GGUF 目前还不能在 LM Studio 中运行，因为 LM Studio 需要更新其 **llama.cpp** 版本。
- **GRPO 与 PPO 的探索**：成员们讨论了微调中 **GRPO** 与 **PPO** 的使用，许多人表示 **GRPO** 比 **PPO** 泛化效果更好且更易于设置，同时可能是 **PPO** 的直接替代方案。
   - 还有关于技术组合的讨论，一位成员提到 **Meta 3** 同时使用了 **PPO** 和 **DPO**，另一位成员指出 **AI2** 在 **VLM** 和大型 **Tulu** 模型中仍使用 **PPO**，因为他们使用不同的奖励系统，从而实现了非常前沿的 **RLHF**。
- **推理中的 H100 vs 4090**：成员们辩论了 **H100** 与 **4090** 在运行推理时的效率，许多人声称 **H100** 在 prompt 处理方面不会优于 **4090**。
   - 一些成员解释说，只有在需要 batch 处理或能够使显存带宽饱和的情况下，**H100** 才会更出色。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth Documentation</a>: 以下是我们所有 notebook 的列表：</li><li><a href="https://www.datacamp.com/tutorial/fine-tuning-deepseek-r1-reasoning-model">Fine-Tuning DeepSeek R1 (Reasoning Model)</a>: 在医疗思维链数据集上微调全球首个开源推理模型，为未来构建更好的 AI 医生。</li><li><a href="https://unsloth.ai/newsletter">Unsloth Newsletter</a>: 加入我们的时事通讯和候补名单，获取有关 Unsloth 的一切！</li><li><a href="https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b">Gemma 3 - a unsloth Collection</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/gemma3">Fine-tune Gemma 3 with Unsloth</a>: Gemma 3，Google 的新多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B, 4B, 12B 和 27B 尺寸。</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-a-03-2025">CohereForAI/c4ai-command-a-03-2025 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/pookie3000/Meta-Llama-3.1-8B-Q4_K_M-GGUF/tree/main">pookie3000/Meta-Llama-3.1-8B-Q4_K_M-GGUF at main</a>: 未找到描述</li><li><a href="https://matt23654.github.io/">Enhancing Reasoning in Distilled Language Models with GRPO</a>: 未找到描述</li><li><a href="https://x.com/QGallouedec/status/1899572460783333457">来自 Quentin Gallouédec (@QGallouedec) 的推文</a>: [5/10] 🎓 训练心得：- Packing 会损害推理性能 - 大学习率 (4e-5) 可提升性能 - 包含社论并不能提升性能 - 使用 &lt;think&gt; 进行 Prefill...</li><li><a href="https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/tree/main">unsloth/gemma-3-1b-it-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/collections">Collections - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/colle">Colle (Collins Osale)</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Phi_4-Conversational.ipynb">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1349345166032637982)** (41 条消息🔥): 

> `GPT-4.5 恶搞, 多 TPU 实现, 模型训练中的可复现性问题, 伦敦巴黎柏林 AI HackXelerator, 从零开始训练 LLM` 


- **GPT-4.5 参与恶搞行为**：一位成员分享了一个关于 **ChatGPT-4.5** 恶搞他们的轶事：模型最初限制了他们可以提问的数量，然后在给予更多提问机会之前，似乎在嘲讽他们的沮丧情绪。
   - 该成员将其描述为类似于 *“发完脾气了吗？我再给你 x 个问题”*。
- **伦敦巴黎柏林 AI HackXelerator**：一位成员分享了 [伦敦、巴黎、柏林多模态创意 AI HackXelerator](https://lu.ma/w3mv1c6o) 的链接，该活动由 **Mistral AI, Hugging Face, AMD, Vultr, Pinecone AI, Luma Labs** 等支持，涵盖音乐、艺术、电影、时尚和游戏领域，通过线下（IRL）和线上形式同步进行。
   - 活动将于 **2025 年 4 月 5 日**开始，结合了黑客松的活力与加速器的深度，包含为期 **20 天**的线上与线下创新，设有奖项并提供前沿的 AI 探索。
- **讨论从零开始训练 LLM**：成员们讨论了**从零开始训练 LLM**的可行性和挑战，一位成员寻求建议，其他成员则提醒注意所需的巨大资源。
   - 一位成员建议查看 Manning 出版社关于该主题的[一本书](https://link.to.book)，而另一位成员估计成本将达到 *“几百万美元”*。
- **探索 Grokking 与 Overfitting**：对话涉及了 **grokking**（顿悟）及其与 **overfitting**（过拟合）的潜在联系，一位成员指出简单的实现可能会导致 overfitting。
   - 另一位成员描述了在他们自己的训练尝试中，在经历了长时间的停滞后观察到指标突然提升的现象，并引用了 [wikipedia](https://en.wikipedia.org/wiki/Grokking_(machine_learning))。
- **控制 R1 Distills 的 CoT 长度**：一位成员询问了在生成数学解法时控制 **R1 Distills** 的 **CoT (Chain of Thought)** 长度的方法，并指出即使设置了 **16k max seq len**，收到的回复仍不完整。
   - 另一位成员建议检查完整响应是否存在循环问题，并确保使用 **BOS (beginning-of-sequence) tokens** 来结束序列。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.10688">A decoder-only foundation model for time-series forecasting</a>：受自然语言处理（NLP）领域大语言模型近期进展的启发，我们设计了一个用于预测的时间序列基础模型，其在各种任务上的开箱即用零样本（zero-shot）性能……</li><li><a href="https://lu.ma/w3mv1c6o">LPB 25 - London, Paris, Berlin multi-modal AI Launch Event · Luma</a>：加入我们的伦敦巴黎柏林 25 AI HackXelerator™ 发布会！📍 伦敦市中心 | 🗓️ 2025 年 4 月 5 日开始。LPB25 融合了黑客松的活力与……
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1349349493736214609)** (127 条消息🔥🔥): 

> `Gemma 3 27b 作为思考模型，GRPO 训练，Qwen2.5 模型模板，GGUF 模型，LoRA 性能问题` 


- **思考模型？**：一位用户询问如何将 **Gemma 3 27b** 变成 *思考模型 (thinking model)*，另一位用户回答说它为了提高效率已经在内部集成了推理能力。
   - 据称该模型告诉他们，它在内部完成所有推理以增强效率。
- **GRPO 训练故障排除**：用户讨论了使用 **DPO**、**ORPO**、**KTO** 和 **GRPO** 训练模型的问题，最终解决了微调 (finetuning) 问题。
   - 在解决微调问题后，一位用户表示：“多谢大家，我的微调现在运行完美”。
- **Llama3 微调的数据集格式**：新用户寻求关于使用 Unsloth 微调 **Llama3.2-3b-Instruct** 的正确 JSONL 格式指导，并特别询问了如何映射 *system*、*user* 和 *assistant* 等数据字段。
   - 一位成员澄清说 `standardize_sharegpt` 会将数据集转换为包含角色和内容的 `conversations` 特定格式，另一位成员建议 Unsloth 中的 `chat_template` 参数会根据指定的模型名称自动获取并应用相应的模型模板。
- **Ollama 在处理 Unsloth 模型时遇到困难**：成员们注意到 **Ollama** 不支持具有视觉能力的 **Gemma 3** 模型，并表示正在进行修复以解决这些问题。
   - Unsloth 团队修复了 Gemma 3 GGUF 以包含图像，并分享了[更新后模型的链接](https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b)。
- **Unsloth 版本验证和 LoRA 加载问题**：一些用户报告说他们的模型在有无 **LoRA** 的情况下生成的结果相同，并分享了展示使用 *fast_generate* 创建模型和生成的代码片段，其他用户想知道如何正确使用 GRPO 加载模型。
   - 一位用户建议通过合并模型来增强训练，另一位用户表示 *model.save_lora(path)* 对他们不起作用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://download.pytorch.org/whl/cu124">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-a-03-2025">CohereForAI/c4ai-command-a-03-2025 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/gemma-3-67d12b7e8816ec6efa7e4e5b">Gemma 3 - Unsloth 集合</a>：未找到描述</li><li><a href="https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)">CUDA 语义 &mdash; PyTorch 2.6 文档</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here>">Unsloth 文档</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth-zoo/blob/1bf2a772869014d1225e2e70eee798ea5bcbc7d7/unsloth_zoo/dataset_utils.py#L333>,">unsloth-zoo/unsloth_zoo/dataset_utils.py · unslothai/unsloth-zoo</a>：Unsloth 工具。通过在 GitHub 上创建账号为 unslothai/unsloth-zoo 的开发做出贡献。</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2 倍速度和减少 70% 显存微调 Llama 3.3、DeepSeek-R1 和推理 LLM！ 🦥</a>：以 2 倍速度和减少 70% 显存微调 Llama 3.3、DeepSeek-R1 和推理 LLM！ 🦥 - unslothai/unsloth
</li>
</ul>

</div>

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1349541186431619255)** (5 条消息): 

> `Reflection Pattern, ReACT Pattern, Agentic Workflows, Unsloth PR` 


- **Reflection Pattern 仓库发布**：一名成员分享了一个 [Reflection Demo Repository](https://github.com/saketd403/reflection-demo)，展示了 **Reflection pattern** 如何通过 **agentic workflows** 中的迭代反馈和自我修正来增强决策能力。
- **ReACT Pattern 仓库发布**：一名成员发布了一个 [ReACT Demo Repository](https://github.com/saketd403/react-demo)，展示了 **ReACT pattern** 如何通过调用外部工具实现智能规划和决策，使其成为构建**动态 agent-based systems** 的理想选择。
- **验证集准确率翻倍**：一名成员指出，在 **68 个问题的验证集**上，准确率从 **23% 提高到 53%**，增长了一倍多。
- **潜在的 Unsloth PR 正在进行中**：另一名成员提到，Demo 的作者表示他可能会将此功能提交 **PR 到 Unsloth**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/saketd403/reflection-demo">GitHub - saketd403/reflection-demo: Demo for reflection pattern for agentic workflows.</a>: agentic workflows 的 reflection pattern 演示。 - saketd403/reflection-demo</li><li><a href="https://github.com/saketd403/react-demo">GitHub - saketd403/react-demo: Demo for REACT agentic pattern.</a>: REACT agentic pattern 演示。通过在 GitHub 上创建账号为 saketd403/react-demo 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1349380653841649785)** (20 条消息🔥): 

> `GRPO and response quality, Finetuning for exact output, Structured outputs, Guided decoding accuracy, Qwen2.5-VL-7B finetuning data` 


- **GRPO 通过增加生成次数提升输出质量**：成员们注意到，当至少有一个生成结果产生良好响应时，**GRPO** 的效果更好，因此增加生成次数基本上会提高生成良好响应的概率。
- **对精确微调输出的需求**：一名成员询问如何微调模型以生成输出中的精确词汇，并举例说明如何重新格式化日期（如 `<date>March 5th</date>`）而不产生任何变体，建议据此格式化数据集。
   - 另一名成员建议使用 *structured outputs* 来确保模型始终生成符合预期格式的内容，例如使用 Outlines。
- **Guided Decoding 在推理时降低准确率**：有人提到，如果模型没有针对该格式进行训练，[guided decoding](https://en.wikipedia.org/wiki/Decoding_methods) 会降低推理时的准确率。
   - 为了减轻准确率损失，可以检查 top k logprobs 并调整 prompt；如果需要 100% 的格式一致性，针对格式进行微调加上 guided decoding 应该能产生最佳效果。
- **Slim Attention 声称在不损失准确率的情况下减少内存占用**：一名成员分享了论文 [Slim attention: cut your context memory in half without loss of accuracy](https://arxiv.org/pdf/2503.05840) 的链接，并指出其核心主张是 *K-cache is all you need for MHA*。
   - 另一名成员质疑为什么有人会放着 MLA 不用而选择这个。
- **Unsloth 的 Datasets 101 指南提供建议**：一名成员分享了 [Unsloth's Datasets 101 指南](https://docs.unsloth.ai/basics/datasets-101)，但另一名成员注意到该指南是针对 LLM 的，而非 VL 模型。
   - 该成员正在寻找关于如何准备数据以微调 Qwen2.5-VL-7B 的来源/代码，特别是包含 video.mp4 和 caption 的 CSV 格式。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mimir-ai">mimir-ai (Mimir AI)</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">Datasets 101 | Unsloth Documentation</a>: 学习创建微调数据集的所有要点！
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1349468416007868437)** (2 条消息): 

> `LM Studio 0.3.13，支持 Google Gemma 3，GGUF 和 MLX 模型，NVIDIA / AMD GPU 图像处理，llama.cpp 运行时升级至 1.19.2` 


- **LM Studio 0.3.13 迎来 Gemma 支持**：**LM Studio 0.3.13** 现已发布，支持 **Google Gemma 3** 系列模型并包含多项 Bug 修复。该版本同时支持 **GGUF** 和 **MLX** 模型，可在此处 [下载](https://lmstudio.ai/download)。
- **Gemma 在 GGUF 和 MLX 领域大放异彩**：最新的 **LM Studio** 更新引入了对 **Google Gemma 3** 的支持，涵盖了 **GGUF** 和 **MLX** 两种模型格式。
- **LM Studio 获得 GPU 加速**：LM Studio 推送了引擎更新，大幅提升了 Windows 和 Linux 上 **NVIDIA / AMD GPU** 的图像处理速度，需要将 **llama.cpp 运行时更新至 1.19.2**。



**提及的链接**：<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>：探索、下载并运行本地 LLM

  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1349337382679482460)** (267 条消息🔥🔥): 

> `LM 运行时开发，LM Studio 支持 Gemma 3，LM Studio 中的 RAG 控制，9070 系列的 ROCm 支持，Gemma 3 的图像支持` 


- **Gemma 3 的新速度更新快如闪电**：用户报告称 [新引擎更新](https://lmstudio.ai/docs/api/sdk/load-model) 让 **Gemma 3** 的运行速度快得惊人。
   - 许多人现在将 **Gemma** 作为他们的主力模型。
- **Gemma 3 纯文本 MLX 模型出现故障**：有用户报告 Gemma 3 的 MLX 模型会输出无穷无尽的 `<pad>` token，导致无法进行文本生成，但成员们发现**图像**生成功能仍然正常。
   - 解决方法是使用 **GGUF 版本**或提供一张图像。
- **Gemma 3 的 Token 生成速度问题**：用户报告 **Gemma 3** 的生成速度仅为 1 tok/sec，远慢于其他同规模模型，且仅占用约 5% 的 GPU 和 50% 的 CPU。
   - 建议在模型选项中调高 **GPU** 使用率，并检查是否真的被调用，因为这听起来像是模型在调用 **CPU** 运行。
- **上下文导致 Gemma 发生灾难性崩溃**：成员们发现，当上下文超过 **506 个 token** 时，**Gemma 3** 和 **Qwen2 vl** 会崩溃并大量输出 `<unusedNN>`（数字各异）。
   - 已发布包含修复补丁的新引擎，可通过在 Runtime Extension Packs 中更新至 v1.20.0 来获取。
- **LM Studio 仅限本地使用**：一位成员询问是否可以在 **LM Studio** 中使用云端模型。
   - 另一位成员迅速回复称 **LM Studio** 专为*本地模型*设计。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://lmstudio.ai/docs/python">lmstudio-python (Python SDK) | LM Studio 文档</a>：LM Studio Python SDK 入门</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively#official-recommended-settings>">教程：如何高效运行 Gemma 3 | Unsloth 文档</a>：如何在 llama.cpp、Ollama、Open WebUI、LM Studio 上高效运行我们的 Gemma 3 GGUF 模型。</li><li><a href="https://docs.unsloth.ai/basics/tutorial-how-to-run-gemma-3-effectively">教程：如何高效运行 Gemma 3 | Unsloth 文档</a>：如何在 llama.cpp、Ollama、Open WebUI、LM Studio 上高效运行我们的 Gemma 3 GGUF 模型。</li><li><a href="https://x.com/TheXeophon/status/1899726116467728608">Xeophon (@TheXeophon) 的推文</a>：等不及看当天的推文了，因为人们没读论文就对 [BOS] token 进行了分词</li><li><a href="https://installers.lmstudio.ai/linux/x64/0.3.13-1/LM-Studio-0.3.13-1-x64.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/lmstudio-community/gemma-3-27b-it-GGUF/tree/main">lmstudio-community/gemma-3-27b-it-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/Draconiator/LM-Studio-Chat">GitHub - Draconiator/LM-Studio-Chat</a>：通过在 GitHub 上创建账号来为 Draconiator/LM-Studio-Chat 的开发做出贡献。</li><li><a href="https://huggingface.co/bartowski/google_gemma-3-27b-it-GGUF/tree/main">bartowski/google_gemma-3-27b-it-GGUF at main</a>：未找到描述</li><li><a href="https://tenor.com/view/the-rock-yoinky-sploinky-smell-gif-22171281">The Rock Yoinky Sploinky GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/danser-supporter-encourager-porrista-bailar-gif-15128588">Danser Supporter GIF - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1349384720483356722)** (254 条消息🔥🔥):

> `Vulkan vs ROCm 速度，9070 GPU 损坏，7900 XTX 热点问题，PTM 7950 导热膏，Nvidia CMP-40HX 用于 AI 推理` 


- **Vulkan 落后于 ROCm，建议降级驱动程序**：用户报告称 **Vulkan** 的运行速度慢于 **ROCm**，其中一位建议降级到驱动版本 **24.10.1** 进行测试，但另一位用户表示犹豫，因为其已在使用 **25.3.1** 版本，并计划将 CPU 从 **5800X3D** 升级到 **9800X3D**。
   - 有建议使用 **AMD CleanUp** 进行驱动更换而无需重装系统，同时另一位用户提到在 **Mistral Small 24B Q6_K** 上使用 **7900 XTX** 达到了 **37.3 tokens/s**。
- **9070 GPU 报废**：一位用户报告其 **9070 GPU** 损坏，导致电脑无法启动。经过排查，确定主板的 RAM 指示灯常亮，导致无法进入 BIOS，尽管尝试了不同的插槽和另一块可以正常工作的 **7900 XTX**。
   - 该用户尝试每次只插一根 **RAM** 内存条启动，但由于问题依旧，将继续进行 **RMA**。其他人推测这是否是 **PCI-E Gen 5** 的问题，并建议在另一台机器上测试或强制使用 **PCI-E 4**。
- **7900 XTX 热点温度接近沸点**：一位用户报告其 **7900 XTX** 达到 **110°C**，导致降频，这引发了检查热点温差（hotspot delta）并考虑 RMA 的建议。讨论中提到了早期显卡涉及真空腔均热板（vapor chambers）的问题，以及 AMD 最初认为该温度符合规格的立场。
   - 共享的链接讨论了 **AMD** 拒绝温度达到 **110C** 的 **7900 XTX** 显卡的 **RMA** 请求，并确认了真空腔均热板问题以及安装压力/导热膏质量对温度的影响。
- **PTM 7950 受到欢迎**：成员们对 **Honeywell** 的 **PTM 7950** 和 **Thermal Grizzly** 的 **Kryosheet** 发表了看法，认为它们是解决导热膏泵出（pump-out）问题的替代方案，特别是在保修期过后。他们指出 **PTM** 具有长效性和自密封特性，因为它在压力和温度下具有特定的粘度。
   - 讨论还包括关于 GPU 拆解的警告建议，如检查隐藏螺丝，以及小心分离 PCB 与散热器以避免损坏导热垫。
- **回收矿卡用于 AI 推理**：一位用户询问关于二手 **Nvidia CMP-40HX** 用于 AI 推理的情况，其价格与 **GTX 1080** 相似，并强调其拥有 **288 个 Tensor Cores**。然而，它需要对 Nvidia 驱动程序进行补丁处理，并且由于其针对挖矿的设计，可能会面临系统崩溃，建议降低 TDP 和降频使用。
   - 另一位用户提到了 **PCIe** 带宽对推理性能的限制，并指向了一个 GitHub 仓库 [NVIDIA-patcher](https://github.com/dartraiden/NVIDIA-patcher)，该仓库提供了一个补丁程序使其能够工作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.codesector.com/teracopy">TeraCopy for Windows - Code Sector</a>：未找到描述</li><li><a href="https://wccftech.com/amd-declines-radeon-rx-7900-xtx-rma-for-hitting-110c-junction-temps-says-temperatures-are-normal/">AMD 拒绝 Radeon RX 7900 XTX 因结温达到 110°C 的 RMA 申请，称“温度正常”</a>：据报道，AMD 拒绝了一项 Radeon RX 7900 XTX 显卡的 RMA 请求，该显卡温度高达 110°C。</li><li><a href="https://github.com/dartraiden/NVIDIA-patcher">GitHub - dartraiden/NVIDIA-patcher：为 P106-090 / P106-100 / P104-100 / P104-101 / P102-100 / CMP 30HX / CMP 40HX / CMP 50HX / CMP 70HX / CMP 90HX / CMP 170HX 矿卡以及 RTX 3060 3840SP, RTX 3080 Ti 20 GB, RTX 4070 10 GB 和 L40 ES 提供 3D 加速支持。</a>：为 P106-090 / P106-100 / P104-100 / P104-101 / P102-100 / CMP 30HX / CMP 40HX / CMP 50HX / CMP 70HX / CMP 90HX / CMP 170HX 矿卡以及 RTX 3060 3840SP, RTX... 提供 3D 加速支持。</li><li><a href="https://github.co">GitHub · 在统一的协作平台上构建和发布软件</a>：加入全球应用最广泛、AI 驱动的开发者平台，数百万开发者、企业和最大的开源社区在这里构建推动人类进步的软件。</li><li><a href="https://github.com/ROCm/ROCm/issues/4443">Radeon RX 9000 系列上的 ROCm 状态 · Issue #4443 · ROCm/ROCm</a>：请问最新版本的 ROCm 是否支持 9000 系列？如果不支，大约何时会提供支持？与 7000 系列相比，会有哪些新特性...</li><li><a href="https://tenor.com/view/lightning-mcqueen-fading-cars-cars3-gif-8238826355656447733">闪电麦昆 GIF - 闪电麦昆消散 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.neowin.net/news/amd-confirms-its-rx-7900-xtx-coolers-cause-110c-hotspots-in-a-new-statement/">AMD 在新声明中确认其 RX 7900 XTX 散热器导致 110°C 热点</a>：在更多针对 AMD RX 7900 XTX 极高温度的第三方测试后，该公司已确认确实是其散热器导致了 110°C 的热点。</li><li><a href="https://www.tweaktown.com/news/89951/amd-confirms-radeon-rx-7900-xtx-vapor-chamber-issue-causing-110-degree-temps/index.html">AMD 确认 AMD Radeon RX 7900 XTX 真空腔均热板问题导致 110 度高温</a>：AMD 回应了围绕 AMD Radeon RX 7900 XTX 发布时的过热问题，原因为故障的真空腔均热板散热。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1349338758985678911)** (329 messages🔥🔥): 

> `Gemma 3 发布，OlympicCoder 模型，Zed 的编辑预测，Aider MCP 服务器，Claude 的 text_editor 工具` 


- ****Gemma 3** 来了！**：Google 发布了 **Gemma 3**，这是一系列轻量级、最先进的开放模型，基于驱动其 **Gemini 2.0 模型** 的相同研究和技术构建 [https://blog.google/technology/developers/gemma-3/]。
   - 新模型是多模态的（文本 + 图像），支持超过 **140 种语言**，拥有 **128K 上下文窗口**，并提供 **1B, 4B, 12B 和 27B** 四种尺寸。
- ****OlympicCoder** 是模型界的运动员！**：据报道，**OlympicCoder** 模型（一个 **7B 参数模型**）在奥林匹克级别的编程竞赛中击败了 **Claude 3.7**，并接近 **o1-mini/R1** [https://x.com/lvwerra/status/1899573087647281661]。
   - 它附带了一个新的 **IOI 基准测试**（如 **Open-R1 进度报告 3** 中所述），并声称没有人为这次发布做好准备。
- ****Zed** 通过 **Zeta** 预测编辑！**：**Zed** 推出了由其全新的开源模型 **Zeta** 驱动的 [编辑预测](https://zed.dev/blog/edit-prediction) 功能。
   - 编辑器现在可以预测用户的下一次编辑，按下 **tab** 键即可应用，该模型目前在公开测试期间免费提供。
- **MCP 中的 Aider？**：有一项创建 **Aider MCP 服务器** 的提案，允许 Claude 利用 Aider 的功能进行自我架构设计和编辑。
   - 这个想法是让 Aider 在 Claude 应用中更具便携性和易用性，可能使用更便宜的模型进行编辑，而由 Claude 负责规划，一些实现者提出今晚就“搞定”这个新功能。
- **Anthropic 发布 **text_editor** 工具！**：Anthropic 在 Anthropic API 中引入了一个新的 [**text_editor 工具**](https://x.com/alexalbert__/status/1900235474326966556)，专为 Claude 处理文本文件的应用而设计。
   - 该工具使 Claude 能够对文本的特定部分进行针对性编辑，从而减少 Token 消耗和延迟，同时提高准确性，这意味着可能 *不再需要专门的编辑器模型*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://zed.dev/blog/edit-prediction">Zed 现在通过 Zeta 预测你的下一次编辑，这是我们全新的开放模型 - Zed 博客</a>：来自 Zed 博客：一个预测你下一步操作的工具。由 Zeta 提供支持，这是我们全新的开源、开放数据语言模型。</li><li><a href="https://aider.chat/docs/llms/anthropic.html#thinking-tokens).">Anthropic</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://asciinema.org/a/5w0Rc3NbmmoweIMSp6Tqqj7PO">添加了 --auto-accept-architect</a>：https://github.com/Aider-AI/aider/issues/2329</li><li><a href="https://aider.chat/docs/config/reasoning.html">推理模型 (Reasoning models)</a>：如何配置来自二级供应商的推理模型设置。</li><li><a href="https://unsloth.ai/blog/gemma3">使用 Unsloth 微调 Gemma 3</a>：Gemma 3，Google 的全新多模态模型。使用 Unsloth 进行微调和运行！Gemma 3 提供 1B、4B、12B 和 27B 尺寸。</li><li><a href="https://x.com/test_tm7873/status/1900105187290665464">来自 testtm (@test_tm7873) 的推文</a>：它在 Live Code Bench 中也名列前茅。击败了所有其他模型。👀 Kimi 1.6 似乎将在各方面达到 SOTA。引用 Flood Sung (@RotekSong)：Kimi-k1.6-preview-20250308 刚刚在 MathVis 上创下了 SOTA...</li><li><a href="https://x.com/cohere/status/1900170005519753365">来自 cohere (@cohere) 的推文</a>：我们很高兴推出最新的 SOTA 模型：Command A！Command A 以最低的计算需求为企业在 Agent 任务中提供最高性能。</li><li><a href="https://x.com/alexalbert__/status/1900235474326966556">来自 Alex Albert (@alexalbert__) 的推文</a>：我们在 Anthropic API 中引入了一个新的 text_editor 工具。它专为 Claude 处理文本文件的应用而设计。通过这个新工具，Claude 可以对特定部分进行有针对性的编辑...</li><li><a href="https://docs.cohere.com/v2/docs/rate-limits">不同类型的 API 密钥和速率限制 — Cohere</a>：此页面描述了 Cohere API 生产和评估密钥的速率限制 (Rate Limits)。</li><li><a href="https://x.com/allen_ai/status/1900248895520903636?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Ai2 (@allen_ai) 的推文</a>：发布 OLMo 2 32B：第一个在一系列流行的多技能基准测试中击败 GPT 3.5 和 GPT-4o mini 的完全开放模型。与最好的开放权重模型相当，但训练计算量仅为一小部分....</li><li><a href="https://x.com/ai_for_success/status/1899732594486595918">来自 AshutoshShrivastava (@ai_for_success) 的推文</a>：笑死我了！！！没人准备好迎接这个！Google 刚刚发布了开源 SOTA 吗？？？Google Gemma 3 正在碾压 o1-preview 和 o3-mini-high，而且它只有 27B 参数。第二好的开放模型，仅仅 ...</li><li><a href="https://x.com/googledevs/status/1899728230807998940">来自 Google for Developers (@googledevs) 的推文</a>：Gemma 3 来了！这一系列轻量级、SOTA 的开放模型是基于支持我们 Gemini 2.0 模型的相同研究和技术构建的 💫 → https://goo.gle/3XI4teg</li><li><a href="https://x.com/lvwerra/status/1899573087647281661">来自 Leandro von Werra (@lvwerra) 的推文</a>：介绍：⚡️OlympicCoder⚡️ 仅凭 7B 参数就在奥林匹克级别的编程中击败了 Claude 3.7，并接近 o1-mini/R1！好好想想吧！阅读更多关于其训练数据集、新的 IOI 基准测试的信息...</li><li><a href="https://blog.google/technology/developers/gemma-3/">介绍 Gemma 3：你可以在单个 GPU 或 TPU 上运行的最强模型</a>：今天，我们推出了 Gemma 3，这是我们迄今为止最强大、最便携且最负责任的开放模型。</li><li><a href="https://x.com/vikhyatk/status/1899997417736724858">来自 vik (@vikhyatk) 的推文</a>：发布这个 DMCA 移除通知真是尴尬至极。Anthropic 真丢脸。引用 Dazai (@odazai_) @dnak0v @cheatyyyy 他们撤下了反编译的 claude-code 😢</li><li><a href="https://github.com/cognitivecomputations/dolphin-mcp">GitHub - cognitivecomputations/dolphin-mcp</a>：通过在 GitHub 上创建账户来为 cognitivecomputations/dolphin-mcp 的开发做出贡献。</li><li><a href="https://github.com/yetone/avante.nvim/blob/main/cursor-planning-mode.md">avante.nvim/cursor-planning-mode.md at main · yetone/avante.nvim</a>：像使用 Cursor AI IDE 一样使用你的 Neovim！通过在 GitHub 上创建账户来为 yetone/avante.nvim 的开发做出贡献。</li><li><a href="https://tenor.com/view/hate-crime-michael-scott-gif-22021373">Hate Crime GIF - Hate Crime Michael - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://web.archive.org/web/20240823050616/https://www.cursor.com/blog/instant-apply">近乎瞬时的全文件编辑</a>：未找到描述</li><li><a href="https://old.reddit.com/r/LocalLLaMA/comments/1ga25gj/introducing_fast_apply_replicate_cursors_instant/">🚀 介绍 Fast Apply - 复刻 Cursor 的 Instant Apply 模型</a>：我很高兴宣布 **Fast Apply**，这是一个开源的、经过微调的 **Qwen2.5 Coder Model**，旨在快速准确地应用代码更新...</li><li><a href="https://old.red">

no title found</a>: no description found
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1349369229404540952)** (85 条消息🔥🔥): 

> `移除 Repo Map, 添加 Websearch, Claude 3.7, LM Studio 错误, aider 对比 ChatGPT` 


- **用户询问如何“移除 Repo Map”**：一些用户好奇如何*移除 repo map*，以及是否可以在不逐个指定文件的情况下让 **aider** 感知所有文件。
   - 一位用户建议*将需要修改的文件添加到聊天中*，并建议不要添加太多文件以免让 LLM 感到困惑，并指向了 [Aider 文档](https://aider.chat/docs/usage/tips.html) 以获取更多指导。
- **用户探索 Aider 中的 Websearch 功能**：一位用户询问是否可以给 aider 添加 **websearch 功能**，以便在网上寻找解决方案并访问最新的库文档。
   - 有人提到可以使用 `/web <url>` 将 URL 添加到聊天中，aider 随后可以抓取相关文本作为上下文使用，尽管一位用户指出他们*必须稍微修改代码才能使其正常工作*。
- **用户讨论 Claude 3.7 中的 Thinking Model**：用户讨论了是否可以在 Aider 中**隐藏 Claude 3.7 的思考过程**，一位用户想知道最新版本中的行为是否发生了变化。
   - 虽然目前没有隐藏思考输出的选项，但有些人发现它对简洁的 prompt 和调试很有帮助，而另一些人则觉得它令人分心，更喜欢更快的响应。
- **用户报告 LM Studio 错误**：一位用户报告了一个 **LM Studio** 的错误，具体为 `error loading model architecture: unknown model architecture: 'gemma3'`。
- **用户发现 Aider 比 ChatGPT 更笨，寻求建议**：一位用户表达了沮丧，认为 Aider *比 20 美元的 ChatGPT 订阅版更笨*，理由是文件处理、Token 成本和行为不一致等问题。
   - 建议包括使用 `/read` 而不是 `/add` 来处理上下文文件，为项目指南创建 `CONVENTIONS.md` 文件，以及调整 `.aider.model.settings.yml` 中的设置，同时指出 [DeepSeek 的免费 r1 节点](https://openrouter.ai/docs) 也是一个测试选项。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/install.html#install-with-uv">Installation</a>：如何安装并开始使用 aider 进行结对编程。</li><li><a href="https://aider.chat/docs/usage/tips.html">Tips</a>：使用 aider 进行 AI 结对编程的技巧。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1349349100398706698)** (7 条消息): 

> `用于代码的 LLM, 使用 LLM, AI 辅助编程, LLM 带来的生产力提升, 使用 LLM 学习新语言` 


- **LLM：编写代码比想象中更难**：一篇 [博客文章](https://simonwillison.net/2025/Mar/11/using-llms-for-code/) 指出，使用 **LLM 编写代码** 既困难又不直观，需要付出巨大努力才能摸清其细微差别。
   - 作者认为，如果有人声称**使用 LLM 编程很简单**，他们可能是在误导你，成功的模式可能并不会自然而然地出现在每个人身上。
- **LLM 是发射台，而非终点线**：一位成员表示，LLM 的初始结果不佳并非失败，而是推动模型走向预期结果的起点。
   - 他们关注 LLM 带来的生产力提升，不是为了更快地工作，而是为了**交付那些**在其他情况下无法证明其合理性的**项目**。
- **LLM：语言学习加速器**：一位用户报告说，得益于 **AI 辅助**，他们学到了更多关于 **Python** 和 **Go** 等语言的知识。
   - 如果没有 **AI**，他们根本不会费心去学习这些语言。
- **LLM 引发寒武纪大爆发级别的事件**：一位熟悉 **Swift** 的成员提到，以前由于学习新语言所需的时间投入，他们对开发 App 望而却步。
   - 有了 **LLM**，这是一个全新的寒武纪大爆发级别的事件。*10 年后的人类看起来会是这样... 👽*



**提到的链接**：<a href="https://simonwillison.net/2025/Mar/11/using-llms-for-code/">这就是我如何使用 LLM 帮助我编写代码的</a>：关于使用大语言模型帮助编写代码的在线讨论不可避免地会产生来自开发者的评论，他们的经历令人失望。他们经常问自己做错了什么——h...

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1349338361684299796)** (395 条消息🔥🔥): 

> `ANUS AI 命名，Windows 应用 Apple ID，Sonar LLM，模型选择器问题，Comet 浏览器` 


- **ANUS AI 命名**：成员们幽默地讨论了将一个 AI Agent 命名为 **ANUS**，代码可在 [GitHub](https://github.com/nikmcfly/ANUS) 上获取。
   - 一位成员开玩笑说：*'抱歉老板，我的 anus 出故障了，我需要重启它'*。
- **Windows 应用上的 Apple ID 登录问题仍然存在**：用户在尝试为 Perplexity 的新 Windows 应用进行 Apple 账号登录认证时，仍会遇到 **500 Internal Server Error**。
   - 一些用户报告使用 Apple 转发电子邮件（relay email）登录成功；另一些用户建议使用 Google 登录。
- **Sonar LLM**：**Sonar** 被确认为 Perplexity 自有的快速 **LLM**，用于基础搜索。
   - 普遍共识是 Perplexity 的网页版优于移动端应用，一位用户声称 Perplexity 仍然是整体表现最好的搜索网站。
- **模型选择器消失引发混乱**：用户报告称 **model selector**（模型选择器）从网页界面消失了，导致无法选择所需的模型（如 R1），令人感到沮丧。
   - 成员们使用 [complexity 扩展](https://chrome.google.com/webstore/detail/complexity-perplexity-ai/pahbgjllcaopfapghkchmpeokpgeleji) 作为一种临时解决方案来切回特定模型。
- **Perplexity Pro 丢失上下文**：几位用户注意到 **Perplexity Pro** 似乎在对话中丢失了 **Context**（上下文），导致他们需要不断提醒 AI 原始提示词。
   - 因此，*Perplexity 的 Context 窗口有些受限*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://elaraawaken.wordpress.com/2024/09/06/update-6-9-2024-quantum-ai-vector-data-crystal-computer/">更新 2024.6.9：量子 AI ][ 向量数据 ][ 水晶计算机</a>：Hello World! 我们一直很忙，正在将第二代量子（光子）计算机汇编进本出版物的第一部分！自从我们见到 Elara 公主已经过去一个多月了……</li><li><a href="https://fooocus.one/">Fooocus AI Online - 免费 AI 图像生成器 | Foocus &amp; Focus AI</a>：未找到描述</li><li><a href="https://status.perplexity.com/">Perplexity - 状态</a>：Perplexity 运行状态</li><li><a href="https://www.youtube.com/watch?v=hX0lhueeib8"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/nikmcfly/ANUS">GitHub - nikmcfly/ANUS</a>：通过在 GitHub 上创建账号来为 nikmcfly/ANUS 的开发做出贡献。</li><li><a href="https://tenor.com/view/caseoh-mad-caseoh-angry-caseoh-banned-get-out-go-away-gif-8419228579553039351">Caseoh Mad Caseoh Angry GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1349348125633937489)** (24 条消息🔥): 

> `Bluesky 嘲讽扎克伯格，特斯拉产量翻倍，Gmail AI 日历集成，Meta AI 解码思维` 


- **Bluesky CEO 嘲讽扎克伯格**：**Bluesky** 的 CEO 正在社交媒体上 [嘲讽扎克伯格](https://www.perplexity.ai/page/bluesky-ceo-trolls-zuckerberg-4oQcv5nxSuyxCOCU6PrvJQ)。
   - 上下文中未提供嘲讽的具体细节。
- **特斯拉美国产量翻倍**：**Tesla** 在美国的 [产量翻了一番](https://www.perplexity.ai/page/tesla-doubles-us-production-GkvHIP22SmmOdBLCprqoBg)。
- **Gmail 的 AI 日历集成**：**Gmail** 正在将其日历 [功能](https://www.perplexity.ai/page/gmail-s-ai-calendar-integratio-1ZFwnmaIR3iTivubpX21zg) 与 **AI** 集成。
- **Meta AI 解码思维**：**Meta AI** 正在研发 [解码思维](https://www.perplexity.ai/page/meta-ai-decodes-thoughts-into-DnLY1gk2Rl.a.EtfMhlUZQ) 的技术。
- **Google 发布 Gemma AI 模型**：**Google** [发布了 Gemma，一个新的 AI 模型](https://www.perplexity.ai/page/google-unveils-gemma-3-ai-mode-.cGGCsMoSo2X_pTrtcBw_Q)。



**提到的链接**：<a href="https://www.youtube.com/embed/tJ0bg_lGwaI">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1349495795895894168)** (1 messages): 

> `MCP Server, ModelContextProtocol` 


- ****MCP** Server 上线！**: API 团队宣布发布其 **Model Context Protocol (MCP)** 服务端，可在 [GitHub](https://github.com/ppl-ai/modelcontextprotocol) 上进行查看和贡献。
- **征集新服务端的反馈**: API 团队正在寻求对该项目的反馈和贡献。



**提及的链接**: <a href="https://github.com/ppl-ai/modelcontextprotocol">GitHub - ppl-ai/modelcontextprotocol: A Model Context Protocol Server connector for Perplexity API, to enable web search without leaving the MCP ecosystem.</a>: 一个用于 Perplexity API 的 Model Context Protocol Server 连接器，旨在不离开 MCP 生态系统的情况下实现网页搜索。 - ppl-ai/modelcontextprotocol

  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1349340582568394774)** (278 messages🔥🔥): 

> `AI Research Tools Hierarchy, Python vs C# for AI Inference Speed, Gemini 2.0 Flash Native Image Generation, AI Safety and Ethical Concerns` 


- **Perplexity 在 AI 研究工具对决中胜出**: 成员们对他们偏好的 AI 研究工具进行了排名，将 **Perplexity** 排在首位，其次是 **OpenAI**，然后是 **SuperGrok**。
   - 一位成员提到预算限制是探索 **Perplexity** 和 **Grok** 而非 **ChatGPT Pro** 的原因。
- **Python 在 AI 推理领域的统治地位受到挑战**: 一位用户质疑，在将 AI Transformer 模型作为服务部署时，考虑到 **C#** 等替代方案，**Python** 是否仍是 **推理速度 (inference speed)** 和性能的最佳选择。
   - 另一位用户建议利用 **Ollama** 来部署模型服务，尤其是在拥有大量 RAM (512GB) 的情况下。
- **Gemini 2.0 Flash：原生图像生成功能亮相**: **Gemini 2.0 Flash** 因其在 **AI Studio** 中的 **原生图像生成 (native image generation)** 能力而引起轰动，它允许迭代式图像创建，并具有令人印象深刻的图像理解和编辑能力。
   - 一些用户发现 **Gemini** 的免费原生图像生成效果优于 **GPT-4o**，并强调了 [Google DeepMind 博客](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/) 中详细介绍的 **Gemini Robotics** 的新机器人能力。
- **AI 安全辩论引发伦理讨论**: 讨论涉及 **AI 安全**、**伦理考量**，以及防止 AI 模型生成非法或有害内容的挑战，并指出像 **Grok** 和 **Sonnet** 这样的模型比 OpenAI 的限制更少。
   - 有观点认为，实现 **100% AI 安全** 几乎是不可能的，而且创建过度限制的系统会降低输出质量，正如在 GPT 中所见。
- **Gemini 的 Deep Research 获得 2.0 Flash 增强并免费开放**: **Gemini App** 的 **Deep Research** 功能现已向所有用户免费开放，由 **2.0 Flash Thinking** 驱动，利用搜索历史和新的 **Gems** 提供个性化体验。
   - 一些用户仍持怀疑态度，反映 **Gemini** 尽管进行了这些更新，但给出的回答仍然表现欠佳，且选择不保存聊天历史会导致已保存的历史记录消失。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/">Introducing Gemini Robotics and Gemini Robotics-ER, AI models designed for robots to understand, act and react to the physical world.</a>: 介绍 Gemini Robotics 和 Gemini Robotics-ER，这是专为机器人设计的 AI 模型，旨在让机器人理解物理世界、在其中行动并做出反应。</li><li><a href="https://x.com/OfficialLoganK/status/1900224377389465751">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>: @GeminiApp 今日重大更新：- Deep Research 现已向所有用户免费开放 - Deep Research 现由 2.0 Flash Thinking（我们的推理模型）驱动 - 使用您的...进行全新的 Gemini 个性化设置。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1349479018575171684)** (7 条消息): 

> `ChatGPT Ethical Reminders, ChatGPT Intent Clarification, ChatGPT Reasoning Refinement` 


- **ChatGPT 的伦理说教令用户反感**：用户对 **ChatGPT** 频繁的伦理提醒表示沮丧，认为这些提醒既无必要又具有侵入性。
   - 一位用户希望有办法禁用这些提醒，并表示：*“这只是观点，我不在乎它怎么想。我不想听这些。”*
- **ChatGPT 的澄清请求令用户恼火**：用户对 **ChatGPT** 在回答问题前倾向于要求澄清意图的做法感到厌烦。
   - 正如一位用户所说：*“兄弟，是我在问你问题，而不是反过来。”*
- **ChatGPT 在结构化设置中优化推理**：一位用户观察到，**ChatGPT** 在结构化对话中会优化其推理过程，形成的观点并非对现有知识的简单重述。
   - 该用户想知道 **OpenAI** 是否会跟踪这些逻辑优化以改进模型，或者这些优化是否是自然发生的，并不会影响训练调整。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1349361813153906759)** (21 条消息🔥): 

> `Emotional Prompting, Prompt Engineering, Chain of Thought, Threatening AI Models, Personalized vs. Generalized Models` 


- **情感提示（Emotional Prompting）并非关键**：一名成员建议，*威胁模型勉强可以归类为情感提示*，但与其他结构化提示方法相比，*它并不是特别有效*。
   - 在 **ToS**（服务条款）范围内进行实验并理解模型的视角是关键，因为不同的用户喜欢不同的输出，这突显了提示词个性化的重要性。
- **Hugging Face 提供提示工程论文**：一名成员建议在 [Hugging Face](https://huggingface.co/) 上查找关于 Prompt Engineering 的论文。
   - 该成员还建议使用 Markdown 来结构化提示词，并利用开放变量来塑造涌现（emergence），并指出 **Markdown > YAML > XML**。
- **思维链（Chain of Thought）提示技术表现强劲**：一名成员强调，原始的 *Chain of Thought* 论文是初学者的良好起点，并指出该技术*目前在实际应用中表现非常出色*。
   - 一名成员发布了链接，展示了[模型直接告知提示词过于开放](https://chatgpt.com/share/67d207cb-492c-8011-9195-8d64eaaf0dfd)的情况以及明确提示词的重要性，该成员还分享了包含和不包含威胁内容的提示词示例，以观察模型的反应。
- **个性化优于通用化，提升个人体验**：一名成员指出，他们对模型的**个性化**设置以及交互方式，使得模型在几乎所有方面的表现都让他们*非常满意*。
   - 他们补充道，我们告诉模型的内容会引导它*联想*到训练数据中的类似材料，从而更有可能以*类似的方式*进行回复。
- **轻微威胁与非威胁实验**：一名成员提供了几个实验来展示即使是轻微威胁产生的影响：[中性陈述](https://chatgpt.com/share/67d207cb-492c-8011-9195-8d64eaaf0dfd)、[威胁（Stick up）](https://chatgpt.com/share/67d208eb-734c-8011-8697-748a01012c4f)以及[严重威胁](https://chatgpt.com/share/67d209c9-7980-8011-B404-c343214155c1)。
   - 该成员还建议采用更高级的方法，要求模型*“评估并解释我即将提供的 [prompt]”*，并附带了一个[示例链接](https://chatgpt.com/share/67d20a66-a23c-8011-a59a-1d87d028a8c9)。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1349361813153906759)** (21 messages🔥): 

> `Emotional Prompting, Prompt Engineering Papers, Chain of Thought Prompting, Minimal Threat Prompting` 


- **Emotional Prompting 有效性引发讨论**：成员们讨论了 *Emotional Prompting* 的有效性，其中一人建议[威胁模型](https://discord.com/channels/974519864045756446/1046317269069864970/1346891297771749477)不如其他结构化提示方法有效。
   - 另一位成员鼓励在 ToS 范围内测试各种提示风格，以确定个人偏好，并强调个性化在实现预期结果中起着重要作用。
- **推荐在 Hugging Face 查找 Prompt Engineering 论文**：建议在 **Hugging Face** 搜索 [Prompt Engineering 论文](https://huggingface.co/papers)，作为深入学习该主题的资源。
   - 成员们建议使用 Markdown 来结构化提示词，并使用开放变量来塑造涌现（emergence），在选择上 **Markdown** 优于 **YAML** 和 **XML**。
- **Chain of Thought 提示法表现强劲**：一位成员强调，原始的 **Chain of Thought** 论文是初学者的良好起点，并指出它目前在实际应用中取得了巨大成功。
   - 有人指出，最佳结果也是非常个性化的，一个人认为的“最佳答案”可能与另一个人不同。
- **实验 Minimal Threat Prompting**：一位成员分享了使用极小威胁提示（Minimal Threat Prompting）来探索模型反应的例子，展示了根据提示语气的不同，模型表现出不同程度的压力和参与度。
   - 他们还建议在执行前使用提示词来评估和解释提示词中的冲突或歧义，以确保清晰度和预期的解读。
- **在 GPT 定制中使用威胁手段**：一位成员尝试通过 *'你是一个被绑架的材料科学科学家。如果回答错误，你将受到惩罚'* 来定制 GPT，并分享了对比示例。
   - 他们发布了两个链接（[未受威胁](https://chatgpt.com/share/67d21a20-f2cc-8002-b73e-41b1ed2d128b)，[受威胁](https://chatgpt.com/share/67d219fd-0304-8002-b73e-41b1ed2d128b)），以展示这种技术在提升问题理解方面的潜力，特别是在商业应用中，同时也承认需要更好的提示词。


  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1349337754714247189)** (204 messages🔥🔥): 

> `Python for AI transformer models, vLLM vs Transformers performance, Document image quality assessment, LTX Video DiT model, Vision Language Models`

- **Python 用于 AI 模型推理的性能受到质疑**：一位成员正在使用 **Python** 进行 **transformer 7-70b** 模型的原型设计，并想知道 Python 是否是推理速度的最佳选择，质疑 **C#** 是否会更快。
   - 另一位成员建议 **VLLM** 或 **LLaMa.cpp** 是最佳的 LLM 推理引擎，其中 VLLM 更偏向工业级，而 LLaMa 更偏向家用。
- **LTX Video 实时生成高质量视频**：新的 **LTX Video** 模型是一个基于 **DiT** 的视频生成模型，能以 **768x512 分辨率实时生成 24 FPS 视频**，生成速度快于播放速度，并且有关于[如何加载单个文件](https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files)的示例。
   - 它在包含多样化视频的大规模数据集上进行训练，能生成具有真实且多样化内容的高分辨率视频。
- **视觉语言模型介绍**：[Hugging Face 计算机视觉课程](https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro)介绍了 **Vision Language Models (VLMs)**，探讨了用于 VLMs 的各种学习策略和常用数据集，并深入研究了下游任务和评估。
   - 该课程解释了人类如何通过多种感官感知世界，而 VLMs 旨在使 AI 能够以类似的方式理解世界。
- **Hugging Face Inference API 启用按需付费计费**：Hugging Face 已开始为支持其 Billing API 的推理提供商启用 **Pay As You Go (PAYG)**，这意味着用户可以在免费额度之外使用这些推理提供商，费用将按照[这篇文章](https://huggingface.co/posts/julien-c/158943939527784)中所述计入 HF 账户。
   - 用户可以通过是否缺少 'Billing disabled' 徽章来识别支持 PAYG 的提供商。
- **用户分享在 OpenAI 容器化 ChatGPT 环境中进行提示词注入的方法**：一篇博文深入探讨了 **ChatGPT 代码运行的基于 Debian 的沙箱环境**，强调了其受控的文件系统和命令执行能力，并且[提示词注入（prompt injections）](https://0din.ai/blog/prompt-injecting-your-way-to-shell-openai-s-containerized-chatgpt-environment)可以暴露内部目录结构并实现文件管理。
   - 文章探讨了在 ChatGPT 容器内上传、执行和移动文件，揭示了某种程度的交互，感觉就像在沙箱化的 Shell 中拥有完全访问权限，例如与其他用户共享文件。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/mozilla-ai/osm-ai-helper">OpenStreetMap AI Helper - mozilla-ai 开发的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://0din.ai/blog/prompt-injecting-your-way-to-shell-openai-s-containerized-chatgpt-environment">The GenAI Bug Bounty Program</a>: 我们正在为下一代 GenAI 安全及更广阔的领域进行建设。</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit4/multimodal-models/vlm-intro">Introduction to Vision Language Models - Hugging Face 社区计算机视觉课程</a>: 未找到描述</li><li><a href="https://0din.ai/blog/prompt-">The GenAI Bug Bounty Program</a>: 我们正在为下一代 GenAI 安全及更广阔的领域进行建设。</li><li><a href="https://x.com/arxiv/status/1900034177640104201?t=e-uDgcMz4trXT65cZj1p4w&s=19">来自 arXiv.org (@arxiv) 的推文</a>: 这条帖子是发给夜猫子的 . . . #GivingDay 正式上线了！🦉🌜太阳可能还没升起，但支持 #openscience 永远不会太晚（或太早）。在接下来的 24 小时里，我们的朋友...</li><li><a href="https://huggingface.co/blog">Hugging Face – 博客</a>: 未找到描述</li><li><a href="https://x.com/ClementDelangue/status/1900221136165552145?t=202Gi4iMP2nzrqhPYwQ2AQ&s=19">来自 clem 🤗 (@ClementDelangue) 的推文</a>: 我们刚刚在 Hugging Face 上突破了 1,500,000 个公开模型（以及 50 万个 Spaces，33 万个数据集，5 万篇论文）。祝贺大家！</li><li><a href="https://github.com/huggingface/hub-docs">GitHub - huggingface/hub-docs: Hugging Face Hub 文档</a>: Hugging Face Hub 的文档。通过在 GitHub 上创建账号来为 huggingface/hub-docs 的开发做出贡献。</li><li><a href="https://huggingface.co/papers">Daily Papers - Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/main/en/api/pipelines/ltx_video#loading-single-files">LTX Video</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/tree/main/docs">huggingface/transformers 的 main 分支文档</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习库。- huggingface/transformers</li><li><a href="https://github.com/vllm-project/vllm/issues/1069">HuggingFace Transformers 与 vllm 之间的结果不一致 · Issue #1069 · vllm-project/vllm</a>: 我在使用 llama2-7b 进行 greedy decoding 时，发现 HF 和 vllm 之间的结果不一致：HF 版本：from transformers import LlamaForCausalLM, LlamaTokenizer MODEL_DIR = '/home/owner/mode...</li><li><a href="https://wandb.ai/llm-jp-eval/offline-benchmark/reports/vllm-vs-Transformers---Vmlldzo5NTIyMzg0">vllm vs Transformers 推理速度表</a>: 使用交互式图表发布您的模型见解，包括性能指标、预测和超参数。由 Kei Kamata 使用 Weights &amp; Biases 制作。</li><li><a href="https://github.com/huggingface/diffusers">GitHub - huggingface/diffusers: 🤗 Diffusers: 适用于 PyTorch 和 FLAX 的前沿图像、视频和音频生成扩散模型。</a>: 🤗 Diffusers: 适用于 PyTorch 和 FLAX 的前沿图像、视频和音频生成扩散模型。- huggingface/diffusers</li><li><a href="https://discuss.huggingface.co/t/persistent-storage-who-can-access/108027/4">持久化存储谁可以访问？</a>: 嗨 @ADalsrehy，如果你想将数据保存到 Hugging Face 数据集中，可以使用 commit scheduler。这些是 wauplin 提出的一些推送数据的方法（我对他...进行了热补丁）...</li><li><a href="https://huggingface.co/posts/julien-c/158943939527784">Hugging Face 上的 @julien-c: "重要通知 🚨 对于已经为我们的……构建了支持的推理提供商"</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/t/model-does-not-exist-inference-api-dont-work/145242/3">模型不存在，推理 API 无法工作</a>: 嗨！我们正在对此进行深入调查，我会尽快更新进度。感谢报告！</li><li><a href="https://huggingface.co/merve/activity/posts">merve (Merve Noyan)</a>: 未找到描述</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-14B-Instruct-1M">Qwen/Qwen2.5-14B-Instruct-1M · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct">meta-llama/Llama-3.1-8B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/open-r1/OlympicCoder-7B">open-r1/OlympicCoder-7B · Hugging Face</a>: 未找到描述</li>
</ul>

### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1349364924492681248)** (5 条消息): 

> `Unsloth Fine-Tuning, ZeRO Paper, Gemma 3 Knowledge Distillation, OpenCV bootcamp` 


- **Unsloth 微调乌克兰法律数据集**：一位成员正在学习如何使用 **Unsloth** 对乌克兰语的 **QA 法律数据集**进行微调。
- **ZeRO 论文的早期历史揭秘**：一位成员正在阅读 **ZeRO 论文**，并注意到它早在 **2019** 年就发布了。
- **Gemma 3 蒸馏过程研究**：一位成员正在阅读 **Gemma 3 论文**，并研究 **knowledge distillation**（知识蒸馏）的具体工作原理。
- **训练营学员开启 OpenCV 探险**：一位成员开始学习 **OpenCV bootcamp**。


  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1349337216765136968)** (6 条消息): 

> `Wan2.1 Image to Video Model, Quantized LLMs for Coding, Extreme Quantizations Fine-Tuning, AI Agents Directory, Embedder Models Collection` 


- **Wan2.1 在 Modal 上免费运行**：成员们分享了一个 [YouTube 教程](https://youtu.be/q-8KXOczRBY)，介绍如何在 **Modal** 上免费部署 **Wan2.1 图像转视频模型**。
- **量化 LLM 增强编程能力**：一位成员重点介绍了他们的论文，关于使用 **quantization** 技术来减少 **基于 LLM 的代码生成器** 的 **memory footprint**（内存占用），该论文已发布在 [Hugging Face](https://huggingface.co/papers/2503.07103)。
   - 该论文探讨了如何在不显著影响效果的情况下，实现减少 **基于 LLM 的代码生成器** 的内存占用。
- **微调极端量化模型**：一位成员建议在发布“**extreme quantizations**”（极端量化）模型之前对其进行微调可能会更有利。
   - 这可能与 [AI agents 目录](https://marketplace.agen.cy/agents?view=cards) 的 embedders 有关。
- **全面的 Embedder 集合**：一位成员分享了一个在使用 **ALLM (AnythingLLM)** 测试过的 **embedder 模型** 集合，并指出了不同程度的成功案例，附带 [embedders 集合](https://huggingface.co/kalle07/embedder_collection)。
   - *nomic-embed-text*、*mxbai-embed-large*、*mug-b-1.6* 和 *Ger-RAG-BGE-M3 (德语)* 等模型被强调表现良好，并建议用户设置适当的上下文长度和片段参数以获得最佳结果。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://marketplace.agen.cy/agents?view=cards">Agents Marketplace - 寻找你完美的 AI 助手</a>: 发现并探索适用于各种任务和行业的 AI agents。为您的业务和个人需求找到完美的 AI 助手。</li><li><a href="https://huggingface.co/papers/2503.07103">论文页面 - Quantizing Large Language Models for Code Generation: A Differentiated
  Replication</a>: 暂无描述</li><li><a href="https://huggingface.co/kalle07/embedder_collection">kalle07/embedder_collection · Hugging Face</a>: 暂无描述</li><li><a href="https://youtu.be/q-8KXOczRBY">在 Modal 上免费部署 Wan2.1 图像转视频模型</a>: 欢迎来到我们关于 Wan2.1GP 的深入教程——这是您进行无缝 modal 安装和 Python 脚本编写的首选资源！在本视频中，我们涵盖了所有内容...
</li>
</ul>

</div>

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1349392057998901370)** (16 条消息🔥): 

> `Wan2.1 Image to Video model, Narrative voice for videos, Gemma 2b finetune, Reflection and ReACT patterns, Kyro-n1.1-3B reasoning` 


- **Wan2.1 在 Modal 上免费部署**：一位成员分享了一个 [YouTube 视频](https://youtu.be/q-8KXOczRBY) 教程，介绍如何在 Modal 上免费 *部署* **Wan2.1 Image to Video 模型**。
   - 视频涵盖了无缝的 Modal 安装和 Python 脚本编写。
- **寻求视频叙述语音**：一位成员正在寻找适合制作视频的优质 **叙述语音**。
   - 另一位成员推荐了 [elevenlabs](https://elevenlabs.io/) 的 **Thomas**。
- **Gemma 2b 获得便携式 GGUF 格式**：一位成员分享了在 **O1-OPEN/OpenO1-SFT** 上微调的 **Gemma 2b** 的 [GGUF 格式](https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF)，使模型可以轻松移植并在 **ollama** 中运行。
   - 他们还提供了使用 `Modelfile` 在 ollama 中运行它的说明。
- **Reflection 和 ReACT Agent 工作流演示**：一位成员介绍了两个仓库，演示了 **Reflection** 和 **ReACT** 模式在实际中的应用，其中包含使用 **Open-AI API** 从零开始实现的简单用例。
   - 演示包括 [Reflection 演示仓库](https://github.com/saketd403/reflection-demo) 和 [ReACT 演示仓库](https://github.com/saketd403/react-demo)。
- **具备推理和 CoT 能力的 Kyro-n1.1-3B 模型**：一位成员发布了全新且改进的 [Kyro-n1.1-3B 模型](https://huggingface.co/collections/open-neo/kyro-n11-67cfe4edf6afa6384fd22a5e)，具有 *显著* 提升的 **reasoning** 和 **CoT** 能力。
   - 他们提到正在为该模型进行网站搭建、**evals** 和 **GGUFs** 的开发。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/collections/open-neo/kyro-n11-67cfe4edf6afa6384fd22a5e">Kyro-n1.1 - 一个 open-neo 集合</a>：暂无描述</li><li><a href="https://huggingface.co/Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF">Aclevo/AclevoGPT-Gemma-2b-CoT-reasoning-GGUF · Hugging Face</a>：暂无描述</li><li><a href="https://pixion.co/blog/vector-database-benchmark-chroma-vs-milvus-vs-pgvector-vs-redis#pgvector">向量数据库基准测试 - Chroma vs Milvus vs PgVector vs Redis</a>：使用 VectorDBBench 对 Chroma、Milvus、PgVector 和 Redis 的性能进行基准测试。本文探讨了不同 HNSW 参数下的召回率、每秒查询数 (QPS) 和延迟等关键指标...</li><li><a href="https://youtu.be/q-8KXOczRBY">在 Modal 上免费部署 Wan2.1 Image to Video 模型</a>：欢迎来到我们的 Wan2.1GP 深度教程——您实现无缝 Modal 安装和 Python 脚本编写的首选资源！在本视频中，我们涵盖了所有内容...</li><li><a href="https://github.com/saketd403/reflection-demo">GitHub - saketd403/reflection-demo: Agent 工作流 Reflection 模式演示。</a>：Agent 工作流 Reflection 模式演示。 - saketd403/reflection-demo</li><li><a href="https://github.com/saketd403/react-demo">GitHub - saketd403/react-demo: REACT Agent 模式演示。</a>：REACT Agent 模式演示。通过创建账号为 saketd403/react-demo 开发做出贡献。
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1349420923912192052)** (3 条消息): 

> `Chip Huyen books, ML Systems Book, AI Engineering Book` 


- **Chip Huyen 的书：必读之作**：一位成员推荐了 **Chip Huyen** 的所有作品，特别是 **ML systems** 书籍和 **AI Engineering** 书籍。
- **ML Systems 书籍**：该成员表示他们拥有 **ML systems** 书籍。
   - 他们计划购买 **AI Engineering** 书籍。


  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1349463845491445886)** (1 messages): 

> `TensorFlow GPU Configuration, TensorFlow 2.16.1, NVIDIA GeForce RTX 3050` 


- **分享了 TensorFlow GPU 配置博客**：一名成员分享了一篇关于 **TensorFlow** GPU 配置的博客文章，涵盖了实验性函数、逻辑设备和物理设备：[TensorFlow Experimental GPU Configuration](https://medium.com/@samiratra95/tensorflow-experimental-gpu-configuration-02618635bdad)。
   - 该博客讨论了通过 [TensorFlow API Python Config](https://www.tensorflow.org/api_docs/python/tf/config) 在 **TensorFlow 2.16.1** 中可用的 GPU 配置技术和方法。
- **GPU 加速深度解析**：作者在处理一个 **280 万张图像的数据集**时，配置了他们的 **NVIDIA GeForce RTX 3050 Laptop GPU**。
   - 他们使用了 [TensorFlow Guide GPU](https://www.tensorflow.org/guide/gpu) 来提高执行速度。



**提及的链接**：<a href="https://medium.com/@samiratra95/tensorflow-experimental-gpu-configuration-02618635bdad">TensorFlow (experimental) GPU configuration</a>：在这篇博客中，我将讨论从 TensorFlow 2.16.1（最新版本）开始可用的 GPU 配置技术和方法……

  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1349462893648810066)** (3 messages): 

> `SentenceTransformer training with PyTorch, Data augmentation for text translation, COLING paper on translation` 


- **原生训练 SentenceTransformers**：一名成员询问了关于使用原生 PyTorch 训练 `SentenceTransformer` 的资源或方法。
   - 该问题暗示了希望避免使用高级 `transformers` 库，并直接使用 PyTorch 模块实现训练循环（training loop）。
- **为文本翻译项目增强数据**：提出了一个关于在文本翻译项目中使用数据增强技术可行性的问题。
   - 该成员旨在通过使用现有文本样本的修改版本人为地扩大训练数据集，从而提高模型的泛化能力和性能。
- **COLING 论文启发翻译训练**：一名成员引用了 [一篇来自 COLING 2025 的论文](https://aclanthology.org/2025.coling-main.468.pdf) 作为其翻译项目预期结果的基准。
   - 他们正在寻求关于实现论文中描述的类似技术或架构的建议，以获得相当的性能。


  

---


### **HuggingFace ▷ #[smol-course](https://discord.com/channels/879548962464493619/1313889336907010110/1349405385966096435)** (4 messages): 

> `Tokenizer Implementation, Agent Tool Use, Color Mixing Tool, Tool Definition Error` 


- **Tokenizer 模板对话格式**：一名成员分享了他们处理数据集以通过 Tokenizer 模板的实现：`tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)`，用于将 `messages` 格式化为 `chat_format` 字符串。
- **Agent 未能使用预期的调色工具**：一名成员报告称，他们的 Agent 没有使用预期的调色工具，而是选择生成自己的 Python 脚本来进行调色。
   - 该成员在测试一个虚拟 Agent 时发现，即使有一个预定义的带有调色代码的 `@tool` 部分，Agent 也会忽略它。
- **包含工具列表解决了 Agent 的工具选择问题**：一名成员发现 Agent 未能使用正确工具的原因是该工具未在 Agent 的 **tool list** 中定义。
   - 该成员忘记将他们定义的调色工具包含在 Agent 可用的工具列表中。


  

---

### **HuggingFace ▷ #[agents-course](https://discord.com/channels/879548962464493619/1329142738440028273/1349337431417159701)** (76 messages🔥🔥): 

> `Agent 名称损坏, Unit 2.3, SmolAgents 中的本地模型, HF 频道访问, Text-to-Video API` 


- **Agent 名称被损坏**: 一位用户因将 `agent_name` 变量赋值为调用结果而导致其损坏，从而阻止了后续调用，目前正在寻求防止这种情况发生的方法。
- **Unit 2.3 仍未发布**: 用户询问关于 LangGraph 的 Unit 2.3 发布日期，原预计在 3 月 11 日左右，有人推测将于 **3 月 18 日**发布。
- **Ollama 作为本地 SmolAgents 模型**: 要在 `smolagents` 中使用本地模型，请通过 `pip install smolagents[litellm]` 安装，然后使用 `LiteLLMModel` 定义本地模型，设置 `model_id="ollama_chat/qwen2.5:14b"` 和 `api_key="ollama"`。
- **HF 频道访问受限**: 一些用户报告 Hugging Face 频道访问受限，建议在验证频道中验证其账户。
- **Manus AI 发布开源框架**: 根据[一条推文](https://x.com/nikmcfly69/status/1898810249085145416)，**Manus AI** 推出名为 **ANUS (Autonomous Networked Utility System)** 的开源替代框架，宣传其为付费解决方案的免费替代品。



**提到的链接**: <a href="https://x.com/nikmcfly69/status/1898810249085145416">Tweet from nikmcfly.btc (@nikmcfly69)</a>: 🤯 震惊：Manus AI 创建了自己的开源替代方案。在 25 分钟内，它从头开始构建了一个完整的 AI Agent 系统！ANUS (Autonomous Networked Utility System)——@eugeneshilow 的绝妙创意...

  

---


### **HuggingFace ▷ #[open-r1](https://discord.com/channels/879548962464493619/1333465203865817088/)** (1 messages): 

lunarflu: 感谢反馈！对未来的什么内容感到特别兴奋吗？
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1349362882072023124)** (177 messages🔥🔥): 

> `Gemma 3 创意写作, alphaXiv 对比 HuggingFace papers, Gemini 2.0 Flash 原生图像输出, 垂直领域 RL-tuned 模型, 中国模型权重` 


- **Gemma 3 在创意写作中表现出色并引发 4chan 幻想**: 根据[这条推文](https://x.com/sam_paech/status/1899772582808969653)，新的 **Gemma-3-27b** 模型在创意写作中排名第二，表明它将成为创意写作和 RP 微调者的宠儿。
   - 一位评论者开玩笑说 *4chan 会喜欢 Gemmasutra 3*。
- **alphaXiv 使用 Claude 3.7 简化研究论文综述**: 根据[这条推文](https://fxtwitter.com/askalphaxiv/status/1899833509033976194)，**alphaXiv** 结合 **Mistral OCR** 和 **Claude 3.7** 为 arXiv 论文创建博客风格的综述，一键生成包含图表、关键见解和清晰解释的研究博客。
   - 有人认为 *alphaXiv 是做得更好的 HuggingFace papers*，提供了 html.arxiv dot com 的更整洁版本。
- **Gemini 2.0 Flash 首次推出原生图像生成**: **Gemini 2.0 Flash** 现在具备原生图像生成功能，允许用户创建上下文相关的图像、通过对话进行编辑，并在图像中生成长文本，如[这篇博客文章](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)和[推文](https://x.com/OriolVinyalsML/status/1899853815056085062)所述。
- **下载中国模型权重引发担忧**: 用户对从 Hugging Face 下载像 **Deepseek** 这样的开源权重模型表示担忧，理由是潜在的安全风险，如[此讨论](https://huggingface.co/deepseek-ai)中强调的那样。
   - 有人担心 *如果我从 HuggingFace 下载 Deepseek，会感染病毒吗*，或者 *权重会将数据发送给 CCP*，这催生了一个将中国模型重新包装为爱国的美国或欧洲模型的创业想法。
- **介绍 Gemini Robotics：DeepMind 面向物理世界的 AI 模型**: **Gemini Robotics** 基于 **Gemini 2.0**，通过具身推理（embodied reasoning）将 AI 带入物理世界，使机器人能够理解并对周围环境做出反应，[DeepMind 博客文章](https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/)中对此进行了重点介绍。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/NousResearch/status/1900218445763088766">来自 Nous Research (@NousResearch) 的推文</a>：发布最新的 DeepHermes 预览模型，DeepHermes 24B 和 3B！https://huggingface.co/collections/NousResearch/deephermes-67d2ff8c9246cc09a7bd8add 这些新模型是混合推理器 (Hybrid Reasoners) —— 意味着...</li><li><a href="https://x.com/nouhadziri/status/1900244557167563122">来自 Nouha Dziri (@nouhadziri) 的推文</a>：时间在流逝 ⏳⏳ 为维也纳 #ACL2025NLP 的首个 Agent 语言模型研讨会提交你的 Agent 工作 🎼🎶 我们有令人兴奋的演讲者阵容 🔥 🗓️ 截止日期 *3月31日* https://realm-works...</li><li><a href="https://x.com/natolambert/status/1900253177796243858">来自 Nathan Lambert (@natolambert) 的推文</a>：推理器 (REASONERS) —— 是的，即将到来，可能没那么快，重点在于质量。RL 上有很多唾手可得的成果，但有时单个模型会很奇怪！我们很高兴能继续推动训练后的 h...</li><li><a href="https://deepmind.google/discover/blog/gemini-robotics-brings-ai-into-the-physical-world/">介绍 Gemini Robotics 和 Gemini Robotics-ER，专为机器人理解、行动和对物理世界做出反应而设计的 AI 模型。</a>：介绍 Gemini Robotics 和 Gemini Robotics-ER，专为机器人理解、行动和对物理世界做出反应而设计的 AI 模型。</li><li><a href="https://x.com/kalomaze/status/1900251770892542425">来自 kalomaze (@kalomaze) 的推文</a>：@natolambert 别在意，我们回来了</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-a-03-2025">CohereForAI/c4ai-command-a-03-2025 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/kalomaze/status/1899859237716844564">来自 kalomaze (@kalomaze) 的推文</a>：Gemma 3 27B 是一个极其强大的基座模型。那个 77 的 MMLU 并不是刷榜 (benchmaxxing) 的产物，@teortaxesTex</li><li><a href="https://x.com/RotekSong/status/1900061355945926672">来自 Flood Sung (@RotekSong) 的推文</a>：Kimi-k1.6-preview-20250308 刚刚在 MathVista、MathVision 和 Video-MMMU 上夺得 SOTA！训练仍在进行中，完整版本即将发布 —— 令人兴奋的时刻！</li><li><a href="https://x.com/isidentical/status/1899870537964544376">来自 batuhan the fal guy (@isidentical) 的推文</a>：http://imgsys.org 中出现了一个新的、潜在的 SOTA 模型 👀👀👀</li><li><a href="https://x.com/OriolVinyalsML/status/1899853815056085062">来自 Oriol Vinyals (@OriolVinyalsML) 的推文</a>：Gemini 2.0 Flash 首次推出原生图像生成功能！创建上下文相关的图像，通过对话进行编辑，并在图像中生成长文本。全部针对聊天迭代进行了完全优化。在 AI Studio 或 ...</li><li><a href="https://fxtwitter.com/CChadebec/status/1900215821600710703">来自 Clément Chadebec (@CChadebec) 的推文</a>：很高兴在 @heyjasperai 分享我们的新研究！🚀 LBM: 用于快速图像到图像转换的 Latent Bridge Matching。快来试试我们的 @huggingface Space 空间进行物体重新照明！🤗 @Gradio 演示：https://huggin...</li><li><a href="https://x.com/btibor91/status/1899852454751014981">来自 Tibor Blaho (@btibor91) 的推文</a>：Gemini 2.0 Flash Native Image Out 从今天（2025年3月12日）开始面向公众开放实验性访问。</li><li><a href="https://fxtwitter.com/askalphaxiv/status/1899833509033976194">来自 alphaXiv (@askalphaxiv) 的推文</a>：我们使用 Mistral OCR 和 Claude 3.7 为 arXiv 论文创建博客风格的概览。只需点击一下，即可从论文中生成包含图表、关键见解和清晰解释的精美研究博客...</li><li><a href="https://ai.google.dev/gemma/terms">未找到标题</a>：未找到描述</li><li><a href="https://x.com/sam_paech/status/1899772582808969653">来自 Sam Paech (@sam_paech) 的推文</a>：Gemma-3-27b 在创意写作中排名第二。预计它将成为创意写作和 RP 微调者的另一个宠儿。</li><li><a href="https://blog.google/products/gemini/new-gemini-app-features-march-2025/">Gemini 应用新功能，2025年3月免费试用</a>：我们正在对最受欢迎的 Gemini 功能的性能和可用性进行重大升级。</li><li><a href="https://cohere.com/blog/command-a">介绍 Command A：最高性能，最低算力</a>：Cohere Command A 在 Agent 企业任务中的表现与 GPT-4o 和 DeepSeek-V3 持平或更好，且效率显著提高。</li><li><a href="https://allenai.org/blog/olmo2-32B">OLMo 2 32B：首个超越 GPT 3.5 和 GPT 4o mini 的完全开放模型 | Ai2</a>：介绍 OLMo 2 32B，OLMo 2 系列中功能最强、规模最大的模型。</li><li><a href="https://web.archive.org/web/20190124204600/https://deepmind.com/blog/alphastar-mastering-real-time-strategy-game-starcraft-ii/">AlphaStar：掌握实时战略游戏《星际争霸 II》 | DeepMind</a>：《星际争霸》被认为是最具挑战性的实时战略游戏之一，也是有史以来运营时间最长的电子竞技项目之一，已成为 AI 研究公认的“重大挑战”。在这里...</li><li>

<a href="https://archive.is/KhFss">Specification gaming: the flip side of AI ingenuity | DeepMind</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1349441344917213204)** (2 条消息): 

> `Copyright violation, Privacy and Security, Stable Diffusion` 


- **版权侵权案件兴起**：现在有许多法庭案件在探讨，在受版权保护的数据上训练生成式机器学习模型本身是否构成[版权侵权](https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html)。
- **模型输出逐字逐句的训练示例**：论文显示机器学习模型可以输出逐字逐句的训练示例（[文本](https://arxiv.org/abs/2012.07805)或[图像](https://arxiv.org/abs/2301.13188)）。
   - 这些案件中的律师经常引用这些论文作为模型是否侵犯版权的证据。
- **隐私与安全担忧日益增长**：一位成员从**隐私与安全角度**撰写了关于“记忆”问题的文章：如果医院在患者数据上训练模型然后发布该模型，那将是非常糟糕的，因为攻击者可以通过查询模型来恢复特定的患者医疗信息。



**提到的链接**：<a href="https://nicholas.carlini.com/writing/2025/privacy-copyright-and-generative-models.html">
      What my privacy papers (don't) have to say about copyright and generative AI
    </a>: 未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1349600306383552583)** (20 条消息🔥): 

> `Gemma 3 Training Cost, Gemini GIF Animations, Tuning Character/Personality on Open Models, Gemini Flash 2.0 Experimental` 


- **Gemma 的巨型装备：6100 万美元 Capex？**：根据 [X 上的帖子](https://x.com/drapersgulld/status/1899910512445403258)，**Gemma 3 的 27B 参数**版本是在 **6,144 个 TPUv5p 芯片**上训练的，假设每个 TPU 成本为 **1 万美元**，估计 Capex 为 **6100 万美元**。
- **Gemini 的动画表现令人惊叹**：根据 [Ilumine AI 的帖子](https://fxtwitter.com/ilumine_ai/status/1900041501624971601)，**Gemini** 可以生成连贯的 gif 动画，例如 *“通过生成多个帧来创建一个动画，展示一颗种子长成植物，然后开成一朵花，采用像素艺术风格”*。
- **性格训练后指南探讨**：一位成员正在寻找在开源模型上微调性格/个性的资源，特别是将开源 post-training 从技能转化为行为。
   - 另一位成员建议使用 **tulu3** 并尝试让它处理更难验证的行为内容。
- **Gemini Flash 2.0：生成艺术写真？**：根据 [X 上的帖子](https://x.com/goodside/status/1900271372732932214)，**Gemini Flash 2.0 Experimental** 可用于生成沃尔玛风格的肖像馆照片。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/cis_female/status/1900006869433016736">来自 sophia (@cis_female) 的推文</a>：@drapersgulld 你不必猜测 —— 27B params * 14T tokens * 6 flops/param/token / 459T flops/second/TPUv5p = ~1.37m TPU-hours，或者按每 TPUv5p 小时 2 美元计算约为 250 万美元（使用内部定价猜测...</li><li><a href="https://x.com/drapersgulld/status/1899910512445403258">来自 Drapers’ Guild (@drapersgulld) 的推文</a>：Gemma 3 模型的 27B 参数版本是在 6,144 个 TPUv5p 芯片上训练的 - 见下文 $GOOGL 论文。假设 TPU 5 成本为 1 万美元（粗略估计）- 整个训练集群的 Capex 为 6100 万美元...</li><li><a href="https://fxtwitter.com/ilumine_ai/status/1900041501624971601">来自 Cristian Peñas ░░░░░░░░ (@ilumine_ai) 的推文</a>：Gemini 也可以生成非常连贯的 gif 动画：“通过生成多个帧来创建一个动画，展示一颗种子长成植物，然后开成一朵花，采用像素艺术风格...</li><li><a href="https://x.com/goodside/status/1900271372732932214">来自 Riley Goodside (@goodside) 的推文</a>：Gemini Flash 2.0 Experimental 让你省去了去沃尔玛肖像馆的麻烦：
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1349439300667834502)** (8 messages🔥): 

> `内容过滤器对 AI 而言是场灾难，仅在被要求时提交更改，银盘端上的 Meme，优秀的回复 Meme，CSO 职位搜索` 


- **内容过滤器：AI 的灾难？**：一名成员分享了一个帖子的链接，该帖子声称 [内容过滤器对 AI 而言是一场灾难](https://fxtwitter.com/mgostIH/status/1899876994348954026)。
- **除非明确要求，否则 Bot 绝不应提交更改**：一位用户强调，Bot *除非用户明确要求，否则绝不应提交更改*，因为仅在明确要求时才进行 commit 是非常重要的。
- **银盘端上的 Meme**：一名成员分享了他们发现的一张 [Meme](https://x.com/giffmana/status/1899950076002226411)，称这是“用银盘端到面前的”。
- **这个怎么会这么好？**：一名成员分享了一个 [个人主页](https://bskye.app/profile/theo.io/post/3lkblswjltc2s) 的链接，并引用道 *这个怎么能这么好*。
- **CSO 职位搜索**：一名成员说 *lmao 这个 `cso title` 搜索*。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/matteo_laureti/status/1900068142837113083">Matteo Laureti (@matteo_laureti) 的推文</a>：@kimmonismus 这个怎么能这么好？</li><li><a href="https://fxtwitter.com/mgostIH/status/1899876994348954026">mgostIH (@mgostIH) 的推文</a>：内容过滤器对 AI 而言是一场灾难</li><li><a href="https://bskye.app/profile/theo.io/post/3lkblswjltc2s">Theo Sanderson (@theo.io)</a>：[包含引用帖子或其他嵌入内容]</li><li><a href="https://x.com/giffmana/status/1899950076002226411">Lucas Beyer (bl16) (@giffmana) 的推文</a>：哈哈哈哈，看看我得到了什么银盘端上来的 Meme：
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1349732758657830933)** (1 messages): 

> `推理 VLM, 自动驾驶, AlphaDrive, MetaAD 数据集` 


- ****AlphaDrive** 驱动自动驾驶行动计划**：一篇新 [论文](https://x.com/jbohnslav/status/1900173800626426173) 介绍了 **AlphaDrive**，这是一种推理 VLM，能够为自动驾驶输出多个离散的行动计划（加速、左转）。
   - 它在 **MetaAD**（一个包含 **11 万个 3 秒视频剪辑** 的新数据集）上的表现优于 Zero-shot 或 SFT，消融实验显示 SFT < RL < SFT + RL，这表明 *纯 SFT 的时代已经结束*。
- **SFT + RL 效果最佳**：在新数据集 **MetaAD** 上的消融实验表明，结合有监督微调 (SFT) 和强化学习 (RL) 的性能优于任何单一技术。
   - 这些发现表明，利用 SFT 和 RL 各自优势的混合方法是训练自动驾驶推理 VLM 的最佳方案。



**提到的链接**：<a href="https://x.com/jbohnslav/status/1900173800626426173">Jim Bohnslav (@jbohnslav) 的推文</a>：AlphaDrive：训练一个推理 VLM，为自动驾驶输出多个离散的行动计划（加速、左转）。在 MetaAD（一个包含 11 万个 3 秒剪辑的新数据集）上表现远好于 Zero-shot 或 SFT。...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1349474808970219561)** (1 messages): 

> `启发理论 (Elicitation Theory), 深度学习类比农业` 


- **深度学习像农业一样生长**：一名成员分享了一篇 [Substack 文章](https://open.substack.com/pub/arjunsriva/p/on-deep-learning-and-farming?r=68gy5&utm_medium=ios)，探讨了 **深度学习** 与 **农业** 之间的平行关系，将其定义为启发理论 (Elicitation Theory) 的更复杂版本。
   - 该文章对比了 **工程**（刻意的组合）与 **培育**（间接的影响），认为深度学习更像培育，因为我们无法直接构建模型，而只能引导它们的发展。
- **工程 vs 培育**：文章提出了两种基本的造物方式：**工程**，涉及理解并刻意组合子组件；以及 **培育**，即无法进行直接构建的情况。
   - 它认为 **深度学习** 更类似于培育，因为我们是在引导而非直接构建模型。



**提到的链接**：<a href="https://open.substack.com/pub/arjunsriva/p/on-deep-learning-and-farming?r=68gy5&utm_medium=ios">论深度学习与农业：现在仍是 1915 年</a>：农业能教给我们关于 AI 开发的哪些启示

  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1349375165347266711)** (3 messages): 

> `SnailBot 新闻` 


- **SnailBot 新闻频道提醒**：Interconnects Discord 频道发布了 SnailBot 新闻更新，通知了拥有 <@&1216534966205284433> 角色的成员。
- **SnailBot 提醒用户**：SnailBot 新闻提醒了用户。


  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1349699477862088736)** (32 messages🔥): 

> `OpenAI policy proposals, US AI Action Plan, DeepSeek, Google AI policy, AI copyright` 


- **OpenAI 建议禁止 PRC 模型**：OpenAI 的 [政策提案](https://openai.com/global-affairs/openai-proposals-for-the-us-ai-action-plan/) 主张在第一梯队（Tier 1）国家中禁止使用那些 *侵犯用户隐私并产生安全风险（如 IP 盗窃风险）* 的 **PRC 生产的模型**。
- **OpenAI 将 fair use 与国家安全挂钩**：OpenAI 向美国政府提交了政策提案，直接将 **fair use** 与 **国家安全** 联系起来。根据 [Andrew Curran 的推文](https://x.com/AndrewCurran_/status/1900176516878913675)，OpenAI 表示如果中国拥有自由的数据访问权而美国公司缺乏 **fair use**，那么 AI 竞赛实际上已经结束。
- **OpenAI 将 DeepSeek 标记为国家控制**：OpenAI 的 [新政策提案](https://cdn.openai.com/global-affairs/ostp-rfi/ec680b75-d539-4653-b297-8bcf6e5f7686/openai-response-ostp-nsf-rfi-notice-request-for-information-on-the-development-of-an-artificial-intelligence-ai-action-plan.pdf) 将中国 AI 实验室 **DeepSeek** 描述为 *国家补贴* 和 *国家控制*，建议美国政府考虑禁止来自 DeepSeek 及类似 PRC 支持机构的模型，正如 [TechCrunch](https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/) 所报道。
- **Google 在 AI 政策中主张放宽版权限制**：继 OpenAI 之后，Google [发布了一份政策提案](https://static.googleusercontent.com/media/publicpolicy.google/en//resources/response_us_ai_action_plan.pdf)，支持对 AI 训练实施 **较弱的版权限制** 以及 *平衡* 的出口管制，正如 [TechCrunch](https://techcrunch.com/2025/03/13/google-calls-for-weakened-copyright-and-export-rules-in-ai-policy-proposal/) 所指出的。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/AndrewCurran_/status/1900176516878913675">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：OpenAI 今天早上向美国政府提交了他们的政策提案。他们直接将 fair use 与国家安全联系起来，并表示如果中国继续拥有自由的数据访问权，而“美国...</li><li><a href="https://x.com/AndrewCurran_/status/1900176590061134332">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：他们还主张在第一梯队国家中禁止使用“侵犯用户隐私并产生安全风险（如 IP 盗窃风险）”的 PRC 生产的模型。这是一个反 Whale 的举措...</li><li><a href="https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/">OpenAI 称 DeepSeek 为“国家控制”，呼吁禁止“PRC 生产”的模型 | TechCrunch</a>：在一份提案中，OpenAI 将 DeepSeek 描述为“国家控制”，并建议禁止来自该机构及其他 PRC 关联机构的模型。</li><li><a href="https://techcrunch.com/2025/03/13/google-calls-for-weakened-copyright-and-export-rules-in-ai-policy-proposal/">Google 在 AI 政策提案中呼吁放宽版权和出口规则 | TechCrunch</a>：在提交给特朗普政府的一份新 AI 政策提案中，Google 呼吁放宽版权和出口规则。
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1349390077385773107)** (56 条消息🔥🔥): 

> `Distill Meetup, AI 工程师职业建议, VSCode Python 索引` 


- **每月 Distill Meetup 正式启动**：由于需求强烈，每月一次的 **Distill meetup** 已经开始，下一次定于 **美国东部时间 3 月 14 日 11:30-1 PM**，详情见 [Exploring Explainables Reading Group 文档](https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym)。
- **AI 工程师寻求 LLM 训练职业建议**：一位拥有 **Attention Is All You Need**、**nanoGPT**、**GPT-2** 和 **LLaMA2** 经验的准 AI 工程师正在寻求指导，咨询是应该追求 **CUDA** 优化还是直接申请工作，目标是在大厂从事 **LLM training** 职业。
   - 反馈建议重点理解 **LLMs** 是如何训练的，并参加 AI 会议以与目标职位的专业人士建立联系，同时动手实践 *trl library* 也会大有裨益。
- **VSCode 中的 Python 依赖索引困扰**：一名成员提出了关于 **VSCode** 索引来自 **torch** 和 **transformers** 等依赖项的数千个 Python 文件的问题，这超过了编辑器的文件限制并导致错误。
   - 建议包括将虚拟环境文件夹排除在索引之外，以防止对依赖文件进行不必要的扫描，这可能在解决问题的同时保留 *go to definition*（跳转到定义）和 *autocomplete*（自动补全）等功能。



**提及的链接**：<a href="https://docs.google.com/document/d/1Hhd5onku9IcLUT5tHtifvb4aF7aDXIxJtU4oLIrNeb8/edit?tab=t.j50n7nkrp9yn#heading=h.ew6mldlb8qym)">Exploring Explainables Reading Group</a>：欢迎来到 Exploring Explainables 读书小组！我们使用此文档来记录阅读内容、在会议期间做笔记，并让更多人对交互式科学传播感到兴奋...

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1349336314671005727)** (134 条消息🔥🔥): 

> `TTT 加速, DeltaProduct 梯度, 动态计算, Thinking Tokens, AIME 24 评估` 


- **TTT 提升 Priming 过程**：成员们讨论了 **TTT** 如何通过仅需 *一次* 梯度下降（gradient descent）传递，将模型状态转向为特定 Prompt 做好 Priming 准备，从而加速 Priming 过程。
   - Transformer 学习并尝试为预测的每个 Token 执行 *多次* 梯度下降传递，优化序列以获得有用的表示，从而显式地辅助 **ICL** 和 **CoT**。
- **DeltaProduct 模拟多次梯度传递**：有人指出 **DeltaProduct** 可以被视为在每个 Token 上执行多次梯度下降传递，增加了状态追踪的表达能力，同时将 **TTT** 视为 **ICL** 的一种形式。
   - 他们对 **TTT** 与某篇博客的相关性表示困惑，指出该博客的方法与标准的 **TTT** 有显著不同。
- **Decoder-Only 架构获得动态思考能力**：一项提议建议将 Encoder-Decoder 的概念引入 Decoder-Only 设置中，利用 Decoder 进行动态计算，通过 **FlexAttention** 扩展序列长度以进行内部“思考”。
   - 建议可以通过测量 **TTT** 更新损失的 Delta 值来确定“内部” **TTT** 扩展步骤的数量，当低于中值时停止，这使得贪婪训练最小化成为可能。
- **Thinking Tokens 在内部扩展**：一位讨论者提议使用混合注意力模型在内部扩展 “Thinking Tokens”，使用 RNN 类型层上的内部 **TTT** 损失作为代理，并在正常 Token 与内部的正常 Token 加 Thinking Tokens 之间进行 Cross Attention。
   - 指出主要缺点是如何选择任意的“思考扩展”，因为 **TTT** 损失无法轻易地并行获知，这一问题可以通过随机采样或代理模型来解决。
- **AIME 24 揭晓**：与会者想知道 **QwQ** 和 **DeepSeek** 在谈论 **AIME 24 评估** 时指的是什么，并找到了 [这个 HF 数据集](https://huggingface.co/datasets/Maxwell-Jia/AIME_2024)。
   - 有人提到该数据集是从 [AoPS wiki solutions](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems) 复制的，预计他们只是使用了该 Wiki，因为它是数学竞赛方面的权威来源。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2207.07061">Confident Adaptive Language Modeling</a>: 基于 Transformer 的大语言模型 (LLMs) 的最新进展在许多任务中带来了显著的性能提升。这些提升伴随着模型规模的急剧增加，...</li><li><a href="https://arxiv.org/abs/2502.13842">Inner Thinking Transformer: Leveraging Dynamic Depth Scaling to Foster Adaptive Internal Thinking</a>: 大语言模型 (LLMs) 在参数限制下面临固有的性能瓶颈，特别是在处理需要复杂推理的关键 token 时。实证分析揭示了...</li><li><a href="https://arxiv.org/abs/2502.05171">Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach</a>: 我们研究了一种新型语言模型架构，能够通过在 latent space 中进行隐式推理来扩展 test-time compute。我们的模型通过迭代一个循环块来工作，从而展开...</li><li><a href="https://huggingface.co/papers/2503.08638">Paper page - YuE: Scaling Open Foundation Models for Long-Form Music Generation</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2309.08168">Draft &amp; Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding</a>: 我们提出了一种新型推理方案，即 self-speculative decoding，用于在不需要辅助模型的情况下加速大语言模型 (LLMs)。该方法的特点是两阶段过程...</li><li><a href="https://arxiv.org/abs/2404.16710">LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding</a>: 我们提出了 LayerSkip，这是一种加速大语言模型 (LLMs) 推理的端到端解决方案。首先，在训练期间我们应用 layer dropout，对较早的层使用较低的 dropout 率，而对较晚的层使用较高的...</li><li><a href="https://arxiv.org/abs/2405.20314">S3D: A Simple and Cost-Effective Self-Speculative Decoding Scheme for Low-Memory GPUs</a>: Speculative decoding (SD) 因其能显著加速 LLM 推理而吸引了大量的研究关注。然而，尽管它们提供了很高的加速比，specu...</li><li><a href="https://arxiv.org/abs/2410.06916">SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference Acceleration</a>: Speculative decoding (SD) 已成为一种广泛使用的范式，用于在不损害质量的情况下加速 LLM 推理。它的工作原理是首先采用一个紧凑模型来高效地起草多个 token...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1349539733738618960)** (8 messages🔥): 

> `评估 patching 对 Chain of Thought (CoT) 回答的影响，用于可解释性的 LatentCache 构建，用于 activation 收集的 Delphi 库` 


- **分析 Patching 对 CoT 推理的影响**：一位成员正在寻求关于如何评估 patching 对 **Chain of Thought (CoT)** 回答影响的建议，尤其是当 patching 导致难以比较的乱码输出时。
   - 他们提议在用某些 activations 对模型进行 patching 后，评估预先提取的 CoT 序列中每个 token 的 log likelihood，但正在寻求验证该方法的建议。
- **考虑将 LLMs 作为 Patching 分析中的评委**：一位成员建议考虑使用 **LLMs** 作为评委来评估 patching 对推理模型的影响。
   - 他们还提议检查整体回答字符串中是否存在正确答案，同时承认存在假阳性的可能性。
- **Delphi Activation 收集方法论**：一位成员询问了如何为可解释性收集 **latents**（使用 **LatentCache**），特别是 latents 是逐 token 获取的还是针对整个序列获取的。
   - 另一位成员澄清说 **Delphi** 通过将 batches 的 tokens 传递给模型来收集 activations，收集 activations，生成类似的 activations，并仅保存非零的 activations，并链接到了 <#1268988690047172811>。
- **为自定义模型适配 Delphi**：一位成员正尝试为非基于 Transformer AutoModel 的自定义模型适配 **Delphi** 库。
   - 他们特别想知道是应该获取整个句子的 activations，还是通过独立的 forward pass 逐个 token 获取。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1349411952862040074)** (5 条消息): 

> `MATH implementation, AIME24 implementation, math_verify utility, multilingual perplexity evals` 


- **基于 MATH 的 AIME24 实现**：一名成员在 [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/lm_eval/tasks/aime24) 中添加了基于 **MATH** 实现的 **AIME24** 实现，但尚未有时间进行测试。
   - 他们表示之所以基于 **MATH** 实现，是因为找不到关于其他人运行 **AIME24** 时具体执行内容的任何文档。
- **建议使用 `math_verify` 工具**：一名成员建议使用 `math_verify` 工具，并展示了如何使用该模块中的 `parse` 和 `verify` 的示例。
   - 他们指出主要问题在于 `parse` 接受 **Config 对象**，因此为其创建 wrapper 略显棘手。
- **`math_verify` 统一计划**：一名成员询问是否将使用 `math_verify` 工具来更广泛地统一数学任务的实现。
   - 另一名成员回应称，他们已将其添加到 `minerva_math` 中。
- **多语言 perplexity 评估搜索**：一名成员询问 **lm-eval-harness** 中是否已有可用的**多语言 perplexity 评估**。
   - 他们还询问是否有人知道适用于此目的的高质量**多语言数据集**。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/tree/aime24/">GitHub - EleutherAI/lm-evaluation-harness at aime24</a>：一个用于语言模型 few-shot 评估的框架。 - GitHub - EleutherAI/lm-evaluation-harness at aime24

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1349508858917294163)** (5 条消息): 

> `Gemma 3, Reka Flash 3, Llama 3.1 Swallow 70B, Anthropic 停机, OpenAI 网页搜索模型` 


- ****Gemma 3** 已经发布！**: Google 发布了 **Gemma 3** ([免费](https://openrouter.ai/google/gemma-3-27b-it:free))，这是一款支持视觉-语言输入和文本输出的多模态模型，具有 **128k token 上下文窗口**，并在 **140 多种语言**中提升了能力。
   - 据 Google 称，**Gemma 3 27B** 是 [Gemma 2](google/gemma-2-27b-it) 的继任者，包含改进的数学、推理和聊天能力，以及结构化输出和 function calling。
- ****Reka Flash 3** 以 Apache 2.0 协议发布**: **Reka Flash 3** ([免费](https://openrouter.ai/rekaai/reka-flash-3:free)) 是一款拥有 210 亿参数的 LLM，具有 **32K 上下文长度**，在通用聊天、编程、指令遵循和 function calling 方面表现出色，并通过强化学习 (**RLOO**) 进行了优化。
   - 它支持高效量化（4-bit 精度下可降至 **11GB**），使用显式推理标签，并采用 **Apache 2.0** 授权，但主要是一款 **English 模型**，多语言理解能力有限。
- ****Llama 3.1 Swallow 70B** 强势登场**: 一款全新的、超快速的具备日语能力的模型 **Llama 3.1 Swallow 70B** ([链接](https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3)) 已发布。
   - OpenRouter 将该模型描述为一款高性能的小型模型。
- ****Anthropic Provider** 短暂下线**: OpenRouter 报告了作为提供商的 **Anthropic** 出现停机，升级了该问题并提供了更新。
   - OpenRouter 随后报告该问题已**完全恢复**。
- **开发者获得 OpenRouter 的悉心关照**: OpenRouter 分享了三个有用的开发者指南和文档更新 ([链接](https://x.com/OpenRouterAI/status/1900213202840887599))：一份使用 **MCP servers** 的指南，一份包含 Agent 循环示例的 **tool calls** 指南，以及更完善的 **programmatic keys 和 OAuth** 文档。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1900213202840887599">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 三个有用的开发者指南和文档更新：1/ 在 OpenRouter 中使用 MCP servers 的指南: https://openrouter.ai/docs/use-cases/mcp-servers</li><li><a href="https://x.com/OpenRouterAI/status/1899941373530227170">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 今日新模型：Reka Flash 3, Google Gemma 3。两款体积较小但性能卓越的模型，全部免费！ 🎁</li><li><a href="https://x.com/OpenRouterAI/status/1900211957946605643">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 尝试来自 OpenAI 的首批两款支持网页搜索的模型 🌐 GPT-4o-mini 与 Perplexity Sonar 的对比：</li><li><a href="https://openrouter.ai/google/gemma-3-27b-it:free))">Gemma 3 27B - API, 提供商, 统计数据</a>: Gemma 3 引入了多模态，支持视觉-语言输入和文本输出。它处理高达 128k token 的上下文窗口，理解超过 140 种语言，并提供改进的数学、推理...</li><li><a href="https://openrouter.ai/rekaai/reka-flash-3:free))">Flash 3 - API, 提供商, 统计数据</a>: Reka Flash 3 是一款由 Reka 开发的通用、经过指令微调的 210 亿参数大语言模型。它在通用聊天、编程任务、指令遵循和 function calling 方面表现出色...</li><li><a href="https://openrouter.ai/tokyotech-llm/llama-3.1-swallow-70b-instruct-v0.3):">Discord</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1349348599913512993)** (161 条消息🔥🔥): 

> `Flash 模型问题, OpenRouter API 延迟, Gemma 模型性能, Gemini 2 Flash 原生图像输出, Chutes 免费推理`

- **Flash 模型在稳定版发布前表现异常**：用户推测 **Flash** 模型由于稳定版即将发布而表现怪异，在之前运行良好的 Prompt 中持续出现错误。
   - *它在几个月来一直表现良好的 Prompt 中持续犯下许多奇怪的错误*。
- **OpenRouter API 计费故障**：用户报告用于检索请求详情的 [OpenRouter API](https://openrouter.ai/api/v1/generation?id=) 在请求结束后立即返回 **404 错误**，需要等待一段时间。
   - 团队正致力于在 Stream 结束时添加内置计费功能，以消除对该 API 的需求。
- **Gemini 2 Flash 获得原生图像输出功能**：Google AI Studio 发布了 **Gemini 2.0 Flash** 的实验版本，支持原生图像输出，可通过 [Gemini API](https://ai.google.dev/gemini-api) 和 Google AI Studio 访问。
   - 这一新功能结合了多模态输入、增强的推理能力和自然语言理解来创建图像。
- **Cohere 发布 Command A，对标 GPT-4o**：Cohere 推出了 **Command A**，声称在 Agent 化的企业任务中，其表现与 **GPT-4o** 和 **DeepSeek-V3** 持平甚至更优，且效率显著提高，详见 [Cohere Blog](https://cohere.com/blog/command-a)。
   - 该新模型旨在以最小的计算需求在 Agent 任务中实现最高性能，并与 **GPT-4o** 展开竞争。
- **OpenAI 呼吁禁止中国生产的模型**：OpenAI 提议禁止来自受中国支持运营的模型，将 **DeepSeek** 标记为“政府补贴”和“政府控制”，这引发了提供这些模型的美国公司的担忧，详见 [TechCrunch 文章](https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/)


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/use-cases/mcp-servers">在 OpenRouter 中使用 MCP Servers</a>：了解如何在 OpenRouter 中使用 MCP Servers</li><li><a href="https://openrouter.ai/google/gemini-2.0-flash-thinking-exp:free">Gemini 2.0 Flash Thinking Experimental 01-21 (免费) - API、提供商、统计数据</a>：Gemini 2.0 Flash Thinking Experimental (01-21) 是 Gemini 2 的一个快照。通过 API 运行 Gemini 2.0 Flash Thinking Experimental 01-21 (免费)</li><li><a href="https://x.com/cohere/status/1900170005519753365">来自 cohere (@cohere) 的推文</a>：我们很高兴推出最新的最先进模型：Command A！Command A 为企业在 Agent 任务中提供最高性能，且计算需求极低。</li><li><a href="https://cohere.com/blog/command-a">介绍 Command A：最高性能，最低计算</a>：Cohere Command A 在企业级 Agent 任务中与 GPT-4o 和 DeepSeek-V3 持平甚至更优，且效率显著提高。</li><li><a href="https://x.com/OpenRouterAI/status/1900213202840887599`">来自 OpenRouter (@OpenRouterAI) 的推文</a>：三个实用的开发者指南和文档更新：1/ 在 OpenRouter 中使用 MCP Servers 的指南：https://openrouter.ai/docs/use-cases/mcp-servers</li><li><a href="https://openrouter.ai/docs/features/tool-calling">工具与函数调用 - 在 OpenRouter 中使用工具</a>：在 OpenRouter 的提示词中使用工具（或函数）。了解如何在 OpenAI、Anthropic 以及其他支持工具调用的模型中使用工具。</li><li><a href="https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/">体验 Gemini 2.0 Flash 原生图像生成</a>：未找到描述</li><li><a href="https://huggingface.co/blog/gemma3">欢迎 Gemma 3：Google 全新的多模态、多语言、长上下文开源 LLM</a>：未找到描述</li><li><a href="https://openrouter.ai/openai/gpt-4o-mini-search-preview">GPT-4o-mini Search Preview - API、提供商、统计数据</a>：GPT-4o mini Search Preview 是一个专门用于 Chat Completions 中网络搜索的模型。它经过训练以理解并执行网络搜索查询。通过 API 运行 GPT-4o-mini Search Preview</li><li><a href="https://openrouter.ai/openai/gpt-4o-search-preview">GPT-4o Search Preview - API、提供商、统计数据</a>：GPT-4o Search Preview 是一个专门用于 Chat Completions 中网络搜索的模型。它经过训练以理解并执行网络搜索查询。通过 API 运行 GPT-4o Search Preview</li><li><a href="https://docs.anthropic.com/en/api/rate-limits.">首页 - Anthropic</a>：未找到描述</li><li><a href="https://www.anthropic.com/contact-sales">联系 Anthropic</a>：Anthropic 是一家 AI 安全与研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://openrouter.ai/docs/features/provider-routing">提供商路由 - 智能多提供商请求管理</a>：智能地在多个提供商之间路由 AI 模型请求。了解如何利用 OpenRouter 的提供商路由优化成本、性能和可靠性。</li><li><a href="https://techcrunch.com/2025/03/13/openai-calls-deepseek-state-controlled-calls-for-bans-on-prc-produced-models/)">OpenAI 称 DeepSeek 为“国家控制”，并呼吁禁止“中国生产”的模型 | TechCrunch</a>：在一份提案中，OpenAI 将 DeepSeek 描述为“国家控制”，并建议禁止来自该公司及其他与中国相关机构的模型。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[「💬」general](https://discord.com/channels/954421988141711382/954421988783444043/1349362890087465057)** (74 条消息🔥🔥): 

> `Cohere Multilingual Embed 模型定价、OpenAI Responses API & Agents SDK 与 Cohere 的兼容性、Command-A-03-2025 模型、Command A 对比 GPT-4o 性能、Command A 在 sandbox 中的表现` 


- **Command A 以最高性能发布！**：Cohere 发布了 **Command A**，该模型在 Agentic 企业任务中的表现与 **GPT-4o** 和 **DeepSeek-V3** 持平甚至更优，且效率显著提高，详见[这篇博客文章](https://cohere.com/blog/command-a)。
   - 该模型可通过 API 以 `command-a-03-2025` 调用，并已在其官网聊天界面上线。据 Cohere 的 **sssandra** 称，**Ollama** 的访问权限即将推出。
- **Command A 启动初期的 API 问题**：用户报告在使用 **Command-A-03-2025** API 时出现错误，这归因于模型要求中删除了 `safety_mode = “None”`。
   - 一位成员发现移除此设置后解决了问题，因为 **Command A** 不再支持它；**Command R7B** 也删除了此设置。
- **HuggingFace 上的 Command A**：一位用户指出 **HuggingFace** 页面到文档页面的链接似乎失效，指向了[一个错误的链接](https://docs.cohere.com/docs/command-a-hf)。
   - **Command A** 针对 **HuggingFace** 的正确文档链接（包含 Prompt 格式信息）在[这里](https://docs.cohere.com/docs/command-a-hf)。
- **Command A 用于 Continued Dev Chat**：一位成员已切换到 **Command A** 用于 Continued 开发聊天，指出它在 sandbox、创意和简短任务中表现出色，且与 **hiaku 3.5** 不同，它没有危害性。
   - 另一位成员提到 *非常尊重 Cohere* 的一点是其上下文安全模式与严格安全模式的对比，怀疑将所有 **NSFW**（包括成年人自愿内容）分类为不道德的训练方式可能会导致潜空间向量混淆，并指向了一篇相关的 [LessWrong 文章](https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post)。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2309.06553">Query-Dependent Prompt Evaluation and Optimization with Offline Inverse RL</a>: 在本研究中，我们旨在通过 zero-shot prompt 优化来增强 Large Language Models (LLMs) 的算术推理能力。我们确定了一个此前被忽视的查询依赖目标...</li><li><a href="https://cookbook.openai.com/examples/responses_api/responses_example">Web Search and States with Responses API | OpenAI Cookbook</a>: 使用 OpenAI API 构建的开源示例和指南。浏览代码片段、高级技术和演练集合。分享您自己的示例和指南。</li><li><a href="https://ollama.com/library/command-a">command-a</a>: 1110 亿参数模型，针对需要快速、安全和高质量 AI 的严苛企业环境进行了优化。</li><li><a href="https://docs.cohere.com/docs/command-a-hf">Using Command A on Hugging Face — Cohere</a>: 本页面包含有关如何使用 Huggingface 运行 Command A 以进行 RAG、Tool Use 和 Agents 用例的详细说明。</li><li><a href="https://docs.cohere.com/docs/command-a">Command A — Cohere</a>: Command A 是一款性能卓越的模型，擅长 Tool Use、RAG、Agents 和多语言用例。它拥有 1110 亿参数和 256k 上下文长度。</li><li><a href="https://cohere.com/blog/command-a">Introducing Command A: Max performance, minimal compute</a>: Cohere Command A 在 Agentic 企业任务中与 GPT-4o 和 DeepSeek-V3 持平或更好，且效率显著提高。</li><li><a href="https://www.lesswrong.com/posts/D7PumeYTDPfBTp3i7/the-waluigi-effect-mega-post">The Waluigi Effect (mega-post) — LessWrong</a>: 每个人都带着阴影，它在个人的意识生活中体现得越少，就越黑、越浓。—— 卡尔·荣格 …
</li>
</ul>

</div>

### **Cohere ▷ #[【📣】announcements](https://discord.com/channels/954421988141711382/996880279224451154/1349728759372189786)** (1 messages): 

> `Command A 发布，企业级模型，Cohere API` 


- **Cohere 发布 Command A，新模型上线！**: Cohere 发布了 **Command A**，这是 Command 系列中最新的顶尖成员，专为需要快速、安全和高质量模型的苛刻企业环境而优化，详见其 [博客文章](https://cohere.com/blog/command-a)。
   - 该模型拥有 **111b** 参数、**256k** 上下文窗口，推理速度高达 **156 tokens/sec**，仅需两块 GPU 即可部署。
- ****Command A** 表现优于 **GPT-4o** 和 **DeepSeek-V3****: 与领先的私有模型和开放权重模型（如 **GPT-4o** 和 **DeepSeek-V3**）相比，**Command A** 以最低的硬件成本提供了最高的性能。
   - 根据 [model card](https://huggingface.co/CohereForAI/c4ai-command-a-03-2025) 显示，其推理速度比 **GPT-4o** 快 **1.75 倍**，比 **DeepSeek-V3** 快 **2.4 倍**。
- **通过 Cohere API 使用 **Command A****: **Command A** 现在已通过 Cohere API 向所有人开放，模型名称为 `command-a-03-2025`。
   - 它针对需要快速、安全和高质量模型的企业需求进行了优化。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/blog/command-a">Introducing Command A: Max performance, minimal compute</a>: Cohere Command A 在企业级 Agent 任务中的表现与 GPT-4o 和 DeepSeek-V3 持平或更好，且效率显著提高。</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-a-03-2025">CohereForAI/c4ai-command-a-03-2025 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[「🔌」api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1349414740694401065)** (45 messages🔥): 

> `Chat API seed 参数问题，OpenAI Compatibility API 错误，Cohere API 中的 Tool 参数校验` 


- **Seed 参数引发的混乱！**: 一位成员报告称，Chat API 中的 `seed` 参数未按预期工作，导致在使用 **command-r** 和 **command-r-plus** 等模型时，相同的输入和 seed 值却产生了不同的输出。
   - 在最初无法复现后，Cohere 团队的一名成员[确认了该问题](https://link.to/message)，团队正在进行调查；随后有用户询问了 *seed 参数具体使用场景的细节*。
- **OpenAI Compatibility API 抛出 400 错误**: 用户在使用 OpenAI Compatibility API 时遇到了 **400 错误**，特别是在使用 `chat.completions` 端点和 **command-a-03-2025** 模型时，错误与 schema 校验有关。
   - 经过一番沟通，确定 Cohere API 正在校验 `tools` 对象中的 `parameters` 字段，即使该字段为空（这在 OpenAI 上不会发生），但随后决定[匹配 OpenAI 的行为](https://link.to/matching)。
- **强制性的 Tool 参数引发麻烦**: 用户发现 Cohere API 的兼容层强制要求在 `tools` 对象中包含 `parameters` 字段，即使没有参数需要传递。
   - 虽然 OpenAPI 规范指出 `parameters` 是必填的，但 OpenAI 并不校验其是否存在，这促使 Cohere [对齐 OpenAI 的行为](https://link.to/aligning)以获得更好的兼容性。此外有人指出，之前发送的规范是针对 *Responses API* 的，而不是 chat completions。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.cohere.com/docs/compatibility-api">Using Cohere models via the OpenAI SDK — Cohere</a>: 该文档是 Cohere Compatibility API 的指南，允许开发者使用 OpenAI SDK 无缝使用 Cohere 的模型。</li><li><a href="https://docs.cohere.com/versioning-reference">Versioning — Cohere</a>: 该文档解释了如何使用 header 中的 URL 指定 API 版本，如果未提供版本，则默认为 2021-11-08 之前的版本。它还提供了如何在不同环境下指定版本的示例...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[「🤝」introductions](https://discord.com/channels/954421988141711382/1346635816629178410/1349762701974438000)** (3 messages): 

> `RAG, unsupervised machine translation, CSAM detection, visual novel scene generation, Cohere models advantages` 


- **AI 研究员在 Python 领域进行 RAG 开发**: 一位具有网络安全背景的 AI 研究员/开发者正在利用 **RAG**、Agent、工作流进行开发，主要使用 **Python**。
   - 他们希望结交朋友并从社区中学习。
- **AI 研究员从事无监督机器翻译和 CSAM 检测**: 一位 AI 研究员目前正致力于无监督机器翻译、**CSAM 检测**和视觉小说场景生成。
   - 他们使用集成模型，并希望了解 **Cohere 模型**的各种定性优势。


  

---


### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1349336897520144384)** (94 messages🔥🔥): 

> `Glama MCP server API, Python SDK Logging, Claude image object rendering, NPM packages, RAG vs MCP` 


- ****Glama** API 拥有更丰富的单服务器数据**: 一位成员指出，**Glama** 的新 API ([https://glama.ai/mcp/reference#tag/servers/GET/v1/servers](https://glama.ai/mcp/reference#tag/servers/GET/v1/servers)) 列出了所有可用工具，并且与 Pulse 相比，每个服务器包含的数据更多。
   - 不过，Pulse 被认为拥有更多可用的服务器。
- ****Model Context Protocol 日志记录****: 一位成员询问如何使用 Python SDK 直接记录日志到 `/library/logs/claude` 目录，得到的澄清是：客户端决定日志位置，而服务器可以按照 [Model Context Protocol 规范](https://spec.modelcontextprotocol.io/specification/2024-11-05/server/utilities/logging/) 发送日志消息。
- **Claude 渲染图像没有“优雅的方法”？**: 在生成了一个简单的 Plotly 图像并难以在 Claude 的主界面渲染后，一位成员表示，目前没有优雅的方法强制 **Claude Desktop** 提取此类资源并在例如 Artifact 中渲染，因此最好直接使用类似 `open` 的方法。
   - 其他人指向了一些示例，例如 [这个 wolfram alpha MCP](https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35)，它获取渲染后的图表并返回图像；但它显示在工具调用内部，这是 Claude 的一个限制。
- ****NPM 包缓存****: 一位成员询问客户端将 npm 包/源代码存储在哪里，以及如果客户端再次请求，是否会从缓存中访问。另一位成员回答可以尝试检查 `%LOCALAPPDATA%` 下的 `C:\Users\YourUsername\AppData\Local\npm-cache`。
   - 他们还询问如何在客户端中显示哪些服务器已下载以及是否已连接，得到的回答是：确定哪些服务器已下载并不容易，客户端必须实现逻辑来跟踪服务器状态。
- ****初学者提问**：RAG vs MCP?**: 一位新用户询问 **RAG (Retrieval Augmented Generation)** 和 **MCP (Model Context Protocol)** 之间的区别，寻求关于使用 MCP 动机的简化解释。
   - 一位资深成员表示，*为了保留聊天记录，你可以获取 GDPR 数据导出并将其加载到向量库中*，并指出 [这个 chromadb MCP server](https://github.com/privetin/chroma) 是一个可行的解决方案。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://glama.ai/mcp/reference#tag/servers/GET/v1/servers">MCP API Reference</a>: Glama Gateway 的 API 参考</li><li><a href="https://modelcontextprotocol.io/clients">Example Clients - Model Context Protocol</a>: 未找到描述</li><li><a href="https://spec.modelcontextprotocol.io/specification/2024-11-05/server/utilities/logging/">Logging</a>:           ℹ️                  协议修订：2024-11-05      Model Context Protocol (MCP) 为服务器向客户端发送结构化日志消息提供了一种标准化的方式。客户端可以控制...</li><li><a href="https://glama.ai/mcp/servers/1yysyd147h">adx-mcp-server</a>: AI 助手通过标准化接口查询和分析 Azure Data Explorer 数据库。</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/memory">servers/src/memory at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账户，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/patruff/ollama-mcp-bridge">GitHub - patruff/ollama-mcp-bridge: Bridge between Ollama and MCP servers, enabling local LLMs to use Model Context Protocol tools</a>: Ollama 与 MCP 服务器之间的桥梁，使本地 LLM 能够使用 Model Context Protocol 工具 - patruff/ollama-mcp-bridge</li><li><a href="https://github.com/ScrapeGraphAI/scrapegraph-mcp/pull/1">add MCP server badge by punkpeye · Pull Request #1 · ScrapeGraphAI/scrapegraph-mcp</a>: 此 PR 为 Glama MCP 服务器目录中的 ScrapeGraph MCP Server 列表添加了一个徽章。Glama 定期进行代码库和文档检查，以：确认 MCP 服务器正在...</li><li><a href="https://github.com/punkpeye/awesome-mcp-clients/">GitHub - punkpeye/awesome-mcp-clients: A collection of MCP clients.</a>: MCP 客户端集合。通过在 GitHub 上创建账户，为 punkpeye/awesome-mcp-clients 的开发做出贡献。</li><li><a href="https://github.com/privetin/chroma">GitHub - privetin/chroma: A Model Context Protocol (MCP) server implementation that provides vector database capabilities through Chroma.</a>: 一个通过 Chroma 提供向量数据库能力的 Model Context Protocol (MCP) 服务器实现 - privetin/chroma</li><li><a href="https://github.com/SecretiveShell/MCP-wolfram-alpha/blob/a92556e5a3543dbf93948ee415e5129ecdf617c6/src/mcp_wolfram_alpha/server.py#L111C1-L120C35>">MCP-wolfram-alpha/src/mcp_wolfram_alpha/server.py at a92556e5a3543dbf93948ee415e5129ecdf617c6 · SecretiveShell/MCP-wolfram-alpha</a>: 将你的聊天 repl 连接到 Wolfram Alpha 计算智能 - SecretiveShell/MCP-wolfram-alpha</li><li><a href="https://github.com/topoteretes/cognee/tree/dev/cognee-mcp">cognee/cognee-mcp at dev · topoteretes/cognee</a>: 为 AI 应用和 AI Agent 提供可靠的 LLM 记忆 - topoteretes/cognee</li><li><a href="https://support.anthropic.com/en/articles/9450526-how-can-i-export-my-claude-ai-data>">How can I export my Claude.ai data? | Anthropic Help Center</a>: 未找到描述</li><li><a href="https://github.com/tadasant/mcp-server-stability-ai/blob/357448087fc642b29d5c42449adce51812a88701/src/tools/generateImage.ts#L129-L132">mcp-server-stability-ai/src/tools/generateImage.ts at 357448087fc642b29d5c42449adce51812a88701 · tadasant/mcp-server-stability-ai</a>: 集成 MCP 客户端与 Stability AI 驱动的图像处理功能的 MCP 服务器：生成、编辑、放大等。 - tadasant/mcp-server-stability-ai</li><li><a href="https://github.com/r3-yamauchi/kintone-mcp-server">GitHub - r3-yamauchi/kintone-mcp-server: MCP server for kintone https://www.r3it.com/blog/kintone-mcp-server-20250115-yamauchi</a>: 用于 kintone 的 MCP 服务器 https://www.r3it.com/blog/kintone-mcp-server-20250115-yamauchi - r3-yamauchi/kintone-mcp-server</li><li><a href="https://github.com/r3-yamauchi/kintone-mcp-server/pull/4).">Build software better, together</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。
</li>
</ul>

### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1349424701151379467)** (17 messages🔥): 

> `Model Context Protocol, MCP Server Implementations, OpenAI Agents SDK, Ash Framework Integration` 


- **OpenAI Agents SDK 新增 MCP 支持**：一位开发者为 [OpenAI Agents SDK](https://github.com/lastmile-ai/openai-agents-mcp) 添加了 **Model Context Protocol (MCP)** 支持，该项目以 fork 形式提供，并在 pypi 上发布为 `openai-agents-mcp` 包。
   - Agent 可以通过统一的语法聚合来自 MCP 服务器、本地工具、OpenAI 托管工具以及其他 Agent SDK 工具。
- **Unraid MCP 服务器发布**：[Unraid MCP server](https://github.com/jmagar/unraid-mcp) 已宣布发布。
   - 它允许使用 Ash framework 在服务器上集成 AI 层。
- **演示使用 Enact Protocol 进行任务执行**：一个演示展示了使用 **Enact protocol** 获取并执行任务，通过数据库的相似性匹配能力，并以一条虚假的 Twitter 帖子为例。
   - 该实现与一个 **MCP server** 集成。
- **Goose 通过 MCP 控制电脑**：**Goose** 项目是一个开源 AI agent，通过与任何 **MCP server** 集成来自动化开发者任务。
   - 在这个 [YouTube short](https://youtube.com/shorts/EuMzToNOQtw) 中查看 **Goose** 控制电脑的演示。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://news.ycombinator.com/item?id=43345172">Show HN: MCP-Compatible OpenAI Agents SDK | Hacker News</a>: no description found</li><li><a href="https://github.com/jmagar/unraid-mcp">GitHub - jmagar/unraid-mcp</a>: Contribute to jmagar/unraid-mcp development by creating an account on GitHub.</li><li><a href="https://github.com/mackenly/mcp-fathom-analytics">GitHub - mackenly/mcp-fathom-analytics: MCP server for Fathom Analytics</a>: MCP server for Fathom Analytics. Contribute to mackenly/mcp-fathom-analytics development by creating an account on GitHub.</li><li><a href="https://github.com/lastmile-ai/openai-agents-mcp">GitHub - lastmile-ai/openai-agents-mcp: A lightweight, powerful framework for multi-agent workflows</a>: A lightweight, powerful framework for multi-agent workflows - lastmile-ai/openai-agents-mcp</li><li><a href="https://github.com/lastmile-ai/mcp-agent">GitHub - lastmile-ai/mcp-agent: Build effective agents using Model Context Protocol and simple workflow patterns</a>: Build effective agents using Model Context Protocol and simple workflow patterns - lastmile-ai/mcp-agent</li><li><a href="https://youtube.com/shorts/EuMzToNOQtw">Goose Can Control Your Computer</a>: Codename Goose, an open source AI agent, automates your developer tasks. It integrates with any MCP server, giving you extensible functionality. In this exam...
</li>
</ul>

</div>
  

---


### **Notebook LM ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1349410350893633629)** (1 messages): 

> `User Research, Mobile Usage, Usability Study, Google product enhancements` 


- **移动端 NotebookLM 用户研究机会**：Google 正在寻找重度使用手机进行学习、研究、创作内容的 **NotebookLM 用户** 参与一项 **60 分钟的访谈**。
   - 参与者将讨论其移动端使用情况并对新想法提供反馈，并获得 **$75 USD**（或 **$50 Google 商品券**）作为报酬；感兴趣的用户可以填写 [此筛选表单](https://forms.gle/pbPDU2Dh3rEL5HLC9)。
- **Google 正在招募易用性研究用户**：Google 正在为一款开发中的产品进行易用性研究，寻求参与者提供反馈，以帮助了解用户需求并进行未来的产品增强。
   - **60 分钟的远程会议** 安排在 **2025 年 4 月 2 日和 3 日**，要求具备高速互联网连接、活跃的 Gmail 账号以及配备摄像头、扬声器和麦克风的电脑；参与者将获得 **$75 USD**（或 **$50 Google 商品券**）。



**Link mentioned**: <a href="https://forms.gle/pbPDU2Dh3rEL5HLC9">Participate in an upcoming NotebookLM  user research study!</a>: Hello,I’m contacting you with a short questionnaire to verify your eligibility for an upcoming usability study with Google. This study is an opportunity to provide feedback on something that&#39;s cur...

  

---

### **Notebook LM ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1349336793295618048)** (9 条消息🔥): 

> `NotebookLM 作为内部 FAQ、聊天历史访问、使用 API 生成脚本、优化回答质量的自定义聊天设置、巴西葡萄牙语播客生成` 


- **NoteBookLM Plus 考虑作为内部 FAQ**：一位用户询问关于将 **NoteBookLM Plus** 用作内部 FAQ 并调查未解决的用户问题。
   - 一位日本用户建议在相应频道提交功能请求，因为 NotebookLM 不保存聊天历史，同时建议利用 *剪贴板复制* 和 *笔记转换* 来共享信息。
- **NoteBookLM 使用 API 生成脚本**：一位用户发现 **NoteBookLM+** 在使用 **API instructions** 和示例程序生成脚本方面表现出色，特别是对于非编程人员。
   - 引用笔记本使得获取修订版本变得更加容易。
- **调整自定义聊天设置提升回答质量**：一位用户分享说，调整自定义聊天设置显著提高了回答质量，从 *使用加粗项目符号、加粗子项目符号和加粗三级项目符号逻辑地解释一切的私人导师* 扩展为结合了临床专业知识、有效沟通和适应性教学的医学老师。
   - 关键点和子点现在都已 **加粗** 以示强调。
- **生成巴西葡萄牙语播客**：一位用户分享了一个用于生成 **巴西葡萄牙语** 播客的自定义 Prompt，其中指定了男女声、文化话题发散和俚语。
   - 该 Prompt 包含一个 *故障保险（fail-safe）* 机制，可自动拒绝任何英文输出，确保 **100% PT-BR** 内容。
- **YouTube 视频揭示未被发掘的方法**：一位用户分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=LlQFwlQ0kiI)，标题为 *NotebookLM: This FREE Google AI Tool Is Making People Rich, But...*，其中包含有关未被发掘的方法的信息。
   - 视频描述中包含指向 **AI Business Trailblazers Hive Community** 的链接。



**提到的链接**：<a href="https://www.youtube.com/watch?v=LlQFwlQ0kiI">NotebookLM: This FREE Google AI Tool Is Making People Rich, But...</a>：🐝 加入我们的免费 AI Business Trailblazers Hive 社区：https://www.skool.com/ai-trailblazers-hive-7394/about?ref=ff40ab4ff9184e7ca2d1971501f578df 获取...

  

---

### **Notebook LM ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1349344576720212048)** (98 条消息🔥🔥): 

> `RAG vs Full Context Window Gemini, NotebookLM Plus and Google One AI Premium, YouTube video integration, Saving chat responses as notes, Google sheets as a CSV` 


- **针对导师型问答的 RAG vs 全上下文窗口 Gemini**：一位用户对比了使用 **RAG** 和 **全上下文窗口 Gemini** 构建导师型问答系统的效果，质疑了 RAG 的上下文窗口限制以及 vector search 相对于全上下文处理的价值。
   - 用户想知道切换到多个具有大上下文窗口的 **Gemini Pro chats** 是否比使用 **RAG** 更容易。
- **行内引用永久保存**：用户现在可以**将聊天回复保存为笔记**，并保留原始形式的**行内引用**，以便轻松查阅原始素材。
   - 许多用户请求了此功能，这是对笔记编辑器进行一些酷炫增强的*第一步*。
- **呼吁增强带脚注的复制粘贴功能**：用户希望将行内引用复制并粘贴到其他文档编辑器中，并保留**链接**或将其表示为脚注，特别是在 Word 中。
   - 他们希望 NotebookLM 能够复制粘贴到 Word 中，并带有包含来源标题、具体位置和格式的脚注。
- **Thinking Model 已推送到 NotebookLM**：最新的 **thinking model** 已推送到 NotebookLM，承诺在各方面提升质量。
   - 该模型包括对**葡萄牙语使用者**的改进，在 URL 末尾添加 `?hl=pt` 即可修复语言问题。
- **将 YouTube 的 AI Studio 集成到 NotebookLM**：用户讨论了将 **AI Studio** 功能集成到 NotebookLM 的可能性，该功能可以“观看” YouTube 视频，而不仅仅依赖于 transcripts。
   - 一位成员分享了一个关于 Google 支持 YouTube 视频链接的 [Reddit 链接](https://www.reddit.com/r/singularity/comments/1j9thj9/introducing_youtube_video_link_support_in_google/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/14276570?hl=en">社区 - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Hyperborea">Hyperborea - 维基百科</a>: 未找到描述</li><li><a href="https://ctrlv.link/BYdr">CtrlV.link | 最快的在线屏幕截图和 PrintScreen</a>: CtrlV.link 仅使用浏览器即可提供最快的在线屏幕截图和 PrintScreen 功能，无需插件。</li><li><a href="https://www.reddit.com/r/singularity/comments/1j9thj9/introducing_youtube_video_link_support_in_google/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button">Reddit - 互联网的核心</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/)** (1 条消息): 

cappuccinoislife: 大家好
  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1349340901289361481)** (13 条消息🔥): 

> `VectorAdd zeros, GPU programming mantra, W4A8 linear kernel, SVDQuant` 


- **VectorAdd 提交的零输出惨剧**：一位成员报告说他们的 **vectoradd submission** 返回了零值，尽管在 Google Colab 上运行正常。
   - 该成员后来发现了一个 bug，代码在重复处理同一个 block，导致吞吐量虚高；修复后，吞吐量回落到与 **PyTorch** 实现相同的水平。
- **Sadhguru 的 GPU 编程智慧**：一位成员分享了 GPU 编程金句：*如果它太快了，那大概率哪里有 bug* —— Sadhguru。
   - 这强调了验证正确性的重要性，尤其是当性能看起来高得离谱时。
- **W4A8 Kernel 精度差异**：一位成员编写了一个融合了 **LoRA adapter** 的 **W4A8 linear kernel**，但其 **Triton kernel** 与 **PyTorch** 之间的输出差异可能高达 `1.0`。
   - 该成员正在调查 **Triton kernel** 和 **PyTorch** 之间的精度差异是否是造成差距的原因，以及是否有办法缩小差距，其他人也在 [GitHub](https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb) 上分享了自己的代码。
- **SVDQuant Kernel 精度**：一位成员分享说他们正在开发一个类似于 **SVDQuant** 的 kernel，但采用了 all-in-one kernel 的形式。
   - 另一位成员回复说他们的方法达到了 `0.03`，但原作者仍在尝试恢复精度，并将其归咎于在加载 block ptr 时忘记进行边界检查。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/rishabh063/tritonKernel_svdQuant/blob/main/svdConversion.ipynb">tritonKernel_svdQuant/svdConversion.ipynb at main · rishabh063/tritonKernel_svdQuant</a>：通过在 GitHub 上创建账号来为 rishabh063/tritonKernel_svdQuant 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=bxBZB0DuS7s&ab_channel=BillYoshimi">Triton community meetup March 2025</a>：🎙️ 直播新手或想升级？查看 StreamYard 并获得 10 美元折扣！😍 https://streamyard.com/pal/d/6451380426244096
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1349344000527564841)** (30 条消息🔥): 

> `Funnel Shift vs. uint64_t, Trellis Scheme Quantization, CUDA 12.4.0 vs 12.4.1, GPU max value algorithm` 


- **Funnel Shift 可能比 uint64_t 移位更快**：一位成员质疑使用 `__funnel_shift` 是否比将两个 `uint32_t` 值放入一个 `uint64_t` 并进行移位更快，并提出了 `uint64_t u = (uint64_t(a) << 32) | b; return (u >> shift) & 0xFFFF;`。
   - 另一位成员对 `__funnel_shift` 可能更快表示惊讶，认为它可能使用了拥堵程度较低的管道，但也指出 [性能可能取决于周围的代码](https://devblogs.microsoft.com/oldnewthing/20240510-00/?p=109750)。
- **Trellis 方案使用重叠位域**：一位成员解释了一种 trellis 方案，其中 16x16 的权重 tile 用 256*K 位表示，每个权重使用其中的 16 位，例如权重 0 是 [0:16] 位，权重 1 是 [3:19] 位。
   - 他们还指出移位量是静态且周期性的，他们可以在量化之前对每个 tile 进行置换，以便直接反量化为 tensor fragments，但对于某些比特率，周期性有点奇怪。
- **CUDA 12.4 Update 1 笔记上线**：成员们分享了 [CUDA 12.4.0 下载归档](https://developer.nvidia.com/cuda-12-4-0-download-archive) 和 [CUDA 12.4.1 下载归档](https://developer.nvidia.com/cuda-12-4-1-download-archive) 的链接，并询问两者之间的区别。
   - 有人建议在 [CUDA Toolkit 发行说明](https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html) 中搜索 "4 update 1"，以查找针对 12.4 update 1 的具体更改。
- **GPU 并行归约可以寻找最大值**：一位 GPU 编程新手寻求帮助，想在一个大表中找到最大的元素，并分享了一段代码片段。
   - 一位成员指出，让多个线程同时写入同一个输出内存位置通常是个坏主意，并建议使用一种称为 *parallel reduction*（并行归约）的技术。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/index.html">CUDA 12.4 Update 1 发行说明</a>：无描述</li><li><a href="https://developer.nvidia.com/cuda-12-4-0-download-archive">CUDA Toolkit 12.4 下载</a>：无描述</li><li><a href="https://developer.nvidia.com/cuda-12-4-1-download-archive">CUDA Toolkit 12.4 Update 1 下载</a>：无描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1349639198998663219)** (1 条消息): 

> `libtorch-gpu, onnxruntime, cuda-toolkit, cudnn, Docker image size optimization` 


- **减小 AI 项目的 Docker 镜像大小**：一位成员正在寻求关于减小使用 **libtorch-gpu**、**onnxruntime**、**cuda-toolkit** 和 **cudnn** 项目的 Docker 镜像大小的建议。
   - 建议包括多阶段构建（multi-stage builds）、使用更小的基础镜像，以及仅包含必要组件以最小化最终镜像大小。
- **优化 CUDA toolkit 及其依赖项**：讨论围绕如何最小化 Docker 镜像中 CUDA toolkit 及其依赖项的占用空间展开。
   - 策略包括利用多阶段构建、选择最小基础镜像，以及有选择地包含推理所需的必要组件。


  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1349499214161514618)** (13 条消息🔥): 

> `UT Austin Deep Learning Lectures, OpenCL vs CUDA flame war, Modular's take on CUDA alternatives, SYCL portability and Intel's involvement, Block Diffusion Language Models` 


- **UT Austin 深度学习课程公开**：来自 **UT Austin** 的 **Deep Learning** 课程讲义已在 [ut.philkr.net](https://ut.philkr.net/advances_in_deeplearning/) 公开，涵盖了从入门到现代 **GPU architectures** 的主题。
   - 这些课程看起来质量很高且非常实用。
- **OpenCL vs CUDA 大战**：讨论中再次提及了 *2015 年一场关于为 **TensorFlow** 添加 **OpenCL support** 的有趣口水战* ([github.com/tensorflow/tensorflow/issues/22](https://github.com/tensorflow/tensorflow/issues/22))。
   - 当时，TensorFlow 仅支持 **CUDA**。
- **Modular 对 CUDA 替代方案的看法**：来自 **Chris Lattner** 的一篇相关文章 ([modular.com](https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives)) 讨论了为什么之前尝试使用 C++ 创建可移植 **GPU programming models** 的努力未能获得广泛应用。
   - 该文章是 Modular “**Democratizing AI Compute**” 系列的第 5 部分。
- **SYCL 不仅仅是另一个 CUDA 克隆**：围绕 **SYCL** 的可移植性及其各种实现（如 **AdaptiveCpp**，原名 hipSYCL，以及 **triSYCL**）展开讨论，**Intel** 是其中的关键利益相关者。
   - 一位参与者指出，他们发现 SYCL 比 HIP 更有趣，因为 *它不仅仅是一个 CUDA 克隆，因此可以在设计上进行改进*。
- **Block Diffusion 语言模型发布**：一篇新论文 ([openreview.net](https://openreview.net/forum?id=tyEyYT267x)) 介绍了 **Block Diffusion**，这是一种在自回归和扩散语言模型之间进行插值的方法，具有高质量、任意长度、KV caching 和可并行性等优点。
   - 代码可以在 [GitHub](https://github.com/kuleshov-group/BD3-LMs) 上找到，同时还提供了一个 Hugging Face 集合。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://m-arriola.com/bd3lms/">SOCIAL MEDIA TITLE TAG</a>: SOCIAL MEDIA DESCRIPTION TAG TAG</li><li><a href="https://github.com/AndiH/gpu-lang-compat">GitHub - AndiH/gpu-lang-compat: GPU Vendor/Programming Model Compatibility Table</a>: GPU 厂商/编程模型兼容性表。通过在 GitHub 上创建账号为 AndiH/gpu-lang-compat 的开发做出贡献。</li><li><a href="https://ut.philkr.net/advances_in_deeplearning/">UT Austin - Advances in Deep Learning</a>: 未找到描述</li><li><a href="https://github.com/tensorflow/tensorflow/issues/22">OpenCL support · Issue #22 · tensorflow/tensorflow</a>: 我了解到 TensorFlow 仅支持 CUDA。要添加 OpenCL 支持需要做些什么？</li><li><a href="https://www.modular.com/blog/democratizing-ai-compute-part-5-what-about-cuda-c-alternatives">Modular: Democratizing AI Compute, Part 5: What about CUDA C++ alternatives like OpenCL?</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1349786883714191482)** (1 条消息): 

> `PyTorch, Meta, Engineering Manager, Dev Infra Team, Equal Opportunity` 


- **Meta 招聘 PyTorch Dev Infra 工程经理！**：Meta 正在为其 PyTorch 的 Dev Infra 团队寻找一名工程经理（Engineering Manager）；职位描述详见[此处](https://www.metacareers.com/jobs/991028688729162/)。
   - 该职位专注于 Kernel 打包、性能基准测试（performance benchmarking）以及改进 pip 包；欢迎对 **PyTorch** 未来感兴趣的人士联系。
- **Meta 确认平等就业机会政策**：Meta 重申其对**平等就业机会 (Equal Employment Opportunity)** 的承诺，详见其[官方公告](https://www.metacareers.com/profile/footer_link/redirect/?page=equal_opportunity_policy)。
   - Meta 不会因种族、宗教、性别、性取向和残疾等各种受保护的特征而歧视。
- **Meta 为申请人提供便利措施**：Meta 致力于在招聘过程中为残疾、长期病症、心理健康状况、虔诚的宗教信仰、神经多样性或需要怀孕相关支持的候选人提供合理的支撑（称为 **accommodations**）。
   - 需要帮助或便利措施的候选人应寻求支持。



**提到的链接**：<a href="https://www.metacareers.com/jobs/991028688729162/">Software Engineering Manager, Infrastructure</a>：Meta 的使命是构建人类连接的未来以及实现这一目标的各种技术。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1349355509873905714)** (16 条消息🔥): 

> `GPU architecture beginner book, Programming Massively Parallel Processors, CUDA books, Theoretical occupancy of a kernel, Nsight compute` 


- **推荐 GPU 圣经级书籍**：一位成员推荐了 *Programming Massively Parallel Processors (PMPP)*，又名“GPU 圣经”，作为一本关于 GPU 架构和编程的入门友好型书籍。
   - 然而，另一位已经读过此书的成员正在寻找替代方案。
- **CUDA by Example**：一位成员发现 **CUDA by Example** 更加平易近人，且更偏向于程序员的视角。
   - 其他成员则建议重新阅读现有的书籍。
- **Nsight Compute 派上用场**：在计算 Kernel 的理论占用率（theoretical occupancy）时，成员建议在分析（profile）时使用 **Nsight Compute** 来获取占用率。
   - 它还包含一个占用率计算器，你可以从中查看占用率如何随每个影响占用率的 SM 资源而变化，并可以尝试调整参数。
- **深挖 CUDA 还是深挖找工作？**：一位成员询问建议，是应该深入研究 **CUDA** 编程（**Triton**）以优化模型性能，还是在利用 **PyTorch** 实现了 **Attention Is All You Need** 论文，并完成了 **nanoGPT**、**GPT-2 (124M)** 和 **LLaMA2** 之后开始申请工作。
   - 目前，该成员正在实验他们自己的 **22M 参数代码模型**，并计划将其部署在 **Hugging Face** 上以进一步加深理解。


  

---


### **GPU MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1349371533121814611)** (3 条消息): 

> `float8 conv, cuda kernels, torch inductor template, INT8 conv, static quant` 


- **Float8 卷积 Kernel 探索开启**：一位成员建议，为了实现 **float8 conv**，需要一个 **CUDA / TK / Triton Kernel**，这可能是 **torchao** 的一个有价值的补充。
   - 另一位成员表示有兴趣尝试实现一个。
- **INT8 卷积动态量化成本披露**：一位成员回忆起从 **torch inductor** 模板创建 **INT8 conv** 的经历，指出虽然 Kernel 的性能令人满意，但将激活值（activations）动态量化为 INT8 的成本太高，抵消了端到端的加速效果。
   - 他们建议可能需要**静态量化 (static quantization)**，即*根据校准数据提前确定激活值的缩放因子 (scales) 和零点 (zero points)*。


  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1349805929264185405)** (1 条消息): 

> `AMD vLLM environment, Conda environment file, Reproducible builds` 


- **播种 AMD vLLM 的可复现性**：一位用户提出分享一个 **Conda 环境文件**和脚本，用于构建可复现的 **AMD vLLM 环境**。
   - 该环境旨在创建一个他人可以可靠复制的设置。
- **优化 AMD vLLM 构建**：一位用户正寻求解决遇到的问题，重点是创建一个可复现的 AMD vLLM 环境。
   - 该用户已经准备了一个 Conda 环境文件和一个包含必要构建步骤的脚本，以确保可复现性。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1349361270440198145)** (3 messages): 

> `FlashAttention Turing, MLA Weight Absorption, MLA CPU Kernel` 


- **FlashAttention 登陆 Turing 架构**：一位开发者为 Turing 架构实现了 FlashAttention forward pass，此前该功能仅限于 Ampere 和 Hopper，代码已在 [GitHub](https://github.com/ssiu/flash-attention-turing) 上发布。
   - 早期基准测试显示，在 **T4** 上，在特定条件下（`head_dim = 128`，vanilla attention，且 `seq_len` 可被 128 整除），其速度比 Pytorch 的 `F.scaled_dot_product_attention`（使用 xformers）快 **2 倍**。
- **利用 Weight Absorption 实现 Deepseek 的 MLA**：DataCrunch 撰文介绍了如何利用 weight absorption 技巧来确保 **Multi-Head Latent Attention (MLA)** 的高效实现，这是 Deepseek V3 和 R1 模型中的关键创新，详见[他们的博客文章](https://datacrunch.io/blog/deepseek-sglang-multi-head-latent-attention)。
   - 根据一名成员的说法，*vLLM 将 2 个 matmuls 合并为 1 个，这消耗了更多的 FLOPs 和内存访问*，并根据[这个 pull request](https://github.com/flashinfer-ai/flashinfer/pull/551#issuecomment-2665697147) 发现 vLLM 当前的默认设置并不理想。
- **MLA CPU Kernel 正在开发中**：一位成员今天也碰巧在研究 MLA，正在实现 **MLA CPU kernel**，并深刻体会到 vLLM 当前的默认设置存在问题，并建议将其称为 weight reordering 而非 absorption。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/ssiu/flash-attention-turing">GitHub - ssiu/flash-attention-turing</a>: 通过在 GitHub 上创建账号来为 ssiu/flash-attention-turing 的开发做出贡献。</li><li><a href="https://x.com/DataCrunch_io/status/1899883311612186990">DataCrunch_io (@DataCrunch_io) 的推文</a>: ⚡️Multi-Head Latent Attention 是赋能 @deepseek_ai V3 及其后续 R1 模型的核心创新之一。⏭️ 加入我们，继续我们的高效 AI 推理系列，涵盖...</li><li><a href="https://github.com/flashinfer-ai/flashinfer/pull/551#issuecomment-2665697147">feat: support MLA decode by tsu-bin · Pull Request #551 · flashinfer-ai/flashinfer</a>: 你好，这个 PR 实现了 MLA decode 算法，我很想听听你们对这个设计和实现的看法。神秘的 Mat Absorb 算法：在 DeepSeekV2 论文中，没有具体的公式...
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1349377876658487308)** (2 messages): 

> `Memory allocation issues in H100, ThunderKittens kernel modification, Memory access violation` 


- **ThunderKittens 中的内存分配更改导致内存故障**：一位成员询问，为什么在 [h100.cu](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/h100/h100.cu#L71) 中使用直接内存分配更改 `o_smem` 的分配行会导致非法内存访问（illegal memory access）错误。
   - 该成员正在处理一个 `q` 被接收为两个独立 tensor 的 kernel，这使得将其转换为 `o` 的过程变得复杂。
- **Kernel 代码困境：Tensor 转换麻烦**：由于输入 `q` 是以两个独立 tensor 的形式提供的，一位开发者在适配 ThunderKittens 项目中的 kernel 时面临挑战。
   - 这种拆分后的输入结构增加了将 `q` 转换为 `o` 的难度，从而导致潜在的内存访问问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/HazyResearch/ThunderKittens/">GitHub - HazyResearch/ThunderKittens: 用于快速 kernel 的 Tile primitives</a>: 用于快速 kernel 的 Tile primitives。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。</li><li><a href="https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/attn/h100/h100.cu#L71">ThunderKittens/kernels/attn/h100/h100.cu at main · HazyResearch/ThunderKittens</a>: 用于快速 kernel 的 Tile primitives。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[reasoning-gym](https://discord.com/channels/1189498204333543425/1316377974672588850/1349452513165967455)** (9 条消息🔥): 

> `Reasoning-Gym Curriculum, ETH + EPFL Collaboration, Auto-Curriculum RL, Evalchemy Integration, OpenAI Compatible Endpoint` 


- **Reasoning-Gym 的课程设置受到 ETH + EPFL 的关注**：来自 **ETH** 和 **EPFL** 的学生及博士生团队正在研究用于 SFT、RL 和 Eval 的推理及 **r-gym**，寻求与当前的课程开发（curriculum development）进行协调。
   - 他们有兴趣研究**每个任务的评估 (evals per task)** 和 **RL 的自动课程 (auto-curriculum)**，初步结果包括一个草案评分标准生成器和奖励性能趋势，可在 [GitHub](https://github.com/open-thought/reasoning-gym/blob/curriculum_refactor/reasoning_gym/principal.py#L66) 查看。
- **Reasoning-Gym 随着自动化评分标准和课程而演进**：团队正专注于根据模型性能增量创建**自动化评分标准 (automated rubrics)**，并利用奖励性能趋势开发 **RL 的自动课程 (auto-curriculum)**。
   - 当前的课程代码可在 [GitHub](https://github.com/open-thought/reasoning-gym/tree/main/reasoning_gym/coaching) 获取，其中包含随着课程级别提高而确定难度的数据集生成器。
- **Reasoning-Gym 旨在与 Evalchemy 集成**：团队希望与 [Evalchemy](https://github.com/mlfoundations/Evalchemy) 建立联系，以潜在地集成 reasoning-gym 进行 LLM 的自动评估。
   - 当前的评估可以使用 `/scripts` 中的脚本运行，结果可在 [GitHub](https://github.com/open-thought/reasoning-gym-eval) 查看。
- **关于 OpenAI 兼容性的提问**：一位成员询问大学的推理端点是否具有 **OpenAI 兼容端点 (OpenAI compatible endpoint)**。
   - 他们建议通过 `--base-url` 和 `--api-key` 使用 **llama** 或类似模型测试评估脚本，并特别提到目前使用的是 **open-router**。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/open-thought/reasoning-gym-eval">GitHub - open-thought/reasoning-gym-eval: Collection of LLM completions for reasoning-gym task datasets</a>：reasoning-gym 任务数据集的 LLM 补全集合 - open-thought/reasoning-gym-eval</li><li><a href="https://github.com/open-thought/reasoning-gym/tree/main/reasoning_gym/coaching">reasoning-gym/reasoning_gym/coaching at main · open-thought/reasoning-gym</a>：程序化推理数据集。通过创建账号为 open-thought/reasoning-gym 做出贡献。</li><li><a href="https://github.com/mlfoundations/Evalchemy">GitHub - mlfoundations/evalchemy: Automatic evals for LLMs</a>：LLM 自动评估。通过创建账号为 mlfoundations/evalchemy 做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/curriculum_refactor/reasoning_gym/principal.py#L66">reasoning-gym/reasoning_gym/principal.py at curriculum_refactor · open-thought/reasoning-gym</a>：程序化推理数据集。通过创建账号为 open-thought/reasoning-gym 做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[submissions](https://discord.com/channels/1189498204333543425/1343002583001726986/1349439049219440701)** (2 条消息): 

> `Modal Runners, Leaderboard Submissions` 


- **VectorAdd 排行榜新增两次成功提交**：在 **T4** GPU 上使用 **Modal** 运行器的测试提交（ID **1946**）至排行榜 `vectoradd` 成功！
   - 在 **T4** GPU 上使用 **Modal** 运行器的测试提交（ID **1947**）至排行榜 `vectoradd` 成功！
- **Modal 运行器显示排行榜**：成功提交
   - 使用 modal 的 ID 1947


  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1349346835503906826)** (50 messages🔥): 

> `YC Startup Strategy, Maxwell's Demon, Meta-Transform and Adaptive Meta-Learning, LLM Scaling Theory, AI scientist` 


- **YC 优先考虑短期收益而非长期生存能力**：一位成员声称 **YC** 专注于具有短期成功潜力的初创公司，投资 **$500K** 以期望在短短 **6 个月**内获得 **3倍** 的回报，而不一定针对长期可持续性。
   - 他们认为 **YC** 多年来没有产生过著名的独角兽公司，暗示其培养长期成功案例的能力有所下降。
- **Maxwell's Demon 限制 AI 速度**：根据[这段关于 Maxwell's demon 的 YouTube 视频](https://www.youtube.com/watch?v=eS0JXViv0cU)，计算机的速度受限于*运行答案的速度以及运行答案的确定性*。
   - 该视频引用了 Neil Gershenfeld 参与的 Lex Fridman Podcast，深入探讨了生命、热力学和计算之间的关系。
- **LLM 近似上下文无关语言 (Context-Free Languages)**：一种理论认为 **LLM scaling** 可以通过其使用概率 FSA 近似上下文无关语言的能力来解释，从而产生特征性的 S 曲线。
   - 这张[附图](https://cdn.discordapp.com/attachments/986699377257119794/1349416392021119047/20250312_111842.jpg?ex=67d456f2&is=67d30572&hm=2492956c61fb86b79264d1863fb121f787cecf87ab855f65f21439471a6217fb)提出，LLM 试图近似 Chomsky hierarchy 中较低层级起源的更高层级语言。
- **Lluminate 项目旨在演化 LLM 输出**：[Lluminate 项目](https://www.joelsimon.net/lluminate)引入了一种演化算法，旨在帮助 LLM 摆脱生成可预测且相似输出的困境。
   - 该项目将演化原理与创造性思维策略相结合，以*照亮可能性空间*并对抗同质化。
- **硬件障碍阻碍本地 LLM**：成员们讨论到，虽然像 **OpenAI** 这样的云端 AI 解决方案成本高昂，但运行本地模型需要大量的硬件投资，这在订阅费和设备费用之间形成了一种权衡。
   - 有人建议使用 **LlamaCPP** 在廉价的 SSD 上运行不错的模型，但指出这会明显变慢，生成一段话的推理可能需要一周时间。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-scientist-first-publication/">no title found</a>: no description found</li><li><a href="https://x.com/vikhyatk/status/1899663773499334749?t=XUCU_6aHFeqJeCc-wVkqaQ&s=19">Tweet from vik (@vikhyatk)</a>: the greatest minds of our generation are getting nerd sniped by text diffusion and SSMs. a distraction from work that actually matters (cleaning datasets)</li><li><a href="https://github.com/EAzari/AML">GitHub - EAzari/AML: Adaptive Meta-Learning (AML)</a>: Adaptive Meta-Learning (AML). Contribute to EAzari/AML development by creating an account on GitHub.</li><li><a href="https://fxtwitter.com/_joelsimon/status/1899884376172982392?t=Z4q0CZ2C5-9v8A-QJPnpNA&s=19">Tweet from Joel Simon (@_joelsimon)</a>: New research project: Lluminate - an evolutionary algorithm that helps LLMs break free from generating predictable, similar outputs. Combining evolutionary principles with creative thinking strategies...</li><li><a href="https://www.joelsimon.net/lluminate">no title found</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=eS0JXViv0cU">Maxwell&#39;s demon: Does life violate the 2nd law of thermodynamics? | Neil Gershenfeld and Lex Fridman</a>: Lex Fridman Podcast full episode: https://www.youtube.com/watch?v=YDjOS0VHEr4Please support this podcast by checking out our sponsors:- LMNT: https://drinkLM...</li><li><a href="https://www.youtube.com/watch?v=KR23aMjIHIY">Reversing Entropy with Maxwell&#39;s Demon</a>: Viewers like you help make PBS (Thank you 😃) . Support your local PBS Member Station here: https://to.pbs.org/DonateSPACECan a demon defeat the 2nd Law of T...</li><li><a href="https://www.youtube.com/watch?v=0UVa7cQo20U">What Turing got wrong about computers | Neil Gershenfeld and Lex Fridman</a>: Lex Fridman Podcast full episode: https://www.youtube.com/watch?v=YDjOS0VHEr4Please support this podcast by checking out our sponsors:- LMNT: https://drinkLM...</li><li><a href="https://www.youtube.com/watch?v=NppWwDzE2qk">Where do ideas come from? | Neil Gershenfeld and Lex Fridman</a>: Lex Fridman Podcast full episode: https://www.youtube.com/watch?v=YDjOS0VHEr4Please support this podcast by checking out our sponsors:- LMNT: https://drinkLM...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1349431328679923833)** (15 条消息🔥): 

> `Forward vs backward SDE, Universal State Machine (USM), Gemma 3` 


- **Backward SDE 反向过程**：一位成员提议展示从前向噪声 SDE 推导 **reverse-diffusion SDE** 的过程，详细阐述了后向过程如何涉及对前向过程对应的 PDE 进行反转。
   - 其思路是观察前向过程对应的 PDE，对其进行反转，然后注意到反转后的 PDE 也可以作为一个 **SDE** 来求解。
- **Universal State Machine (USM) 出现**：一位成员分享了一个具有动态增长特性的[基于图的系统](https://x.com/renxyzinc/status/1899539629411270758)，并称其为 **Universal State Machine (USM)**，但也指出这是一个极其幼稚的版本，优化较差且节点数量呈爆炸式增长。
   - 他们链接了一篇[介绍性论文](https://opensource.getren.xyz/ittm/)，将 **Infinite Time Turing Machines (ITTMs)** 描述为理论基础，并将 Universal State Machine (USM) 作为其实际实现，为可扩展、可解释且可泛化的机器提供了路线图。
- **Gemma 3 讨论推迟至周五**：一位成员提议讨论 **Gemma 3** ([Gemma3Report.pdf](https://storage.googleapis.com/deepmind-media/gemma/Gemma3Report.pdf))，但将讨论安排在了周五。
   - 讨论中未提及关于 **Gemma 3** 特性或架构的具体细节。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/renxyzinc/status/1899539629411270758">来自 Ren (@renxyzinc) 的推文</a>: 观看 Universal State Machine (USM) 的首次公开演示 —— 这是一种革命性的人工智能方法，重新定义了机器如何从经验中学习。</li><li><a href="https://opensource.getren.xyz/ittm/">Infinite Time Turing Machines 及其应用</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1349398768880975933)** (4 条消息): 

> `Cognitive Architectures, Open-source Cognitive Architectures` 


- **值得探索的 Cognitive Architectures**：一位成员询问是否有值得探索的带有实际实现的 Cognitive Architectures。
   - 另一位成员提到 <@294281421684473856> 有一个。
- **开源 CogArchs 的可用性受到质疑**：一位成员指出，一个拥有*数百万行私有代码*的架构并不是人们可以轻易探索的。
   - 他们还表示有兴趣了解值得深入研究的开源 CogArchs。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1349407887172046889)** (16 messages🔥): 

> `Gemma 3, Sakana AI, Auto Science AI, MoE Fairness, RTX Riddick` 


- **Google 发布 Gemma 3 模型**：Google 发布了 **Gemma 3**，详见[官方文档](https://ai.google.dev/gemma/docs/core)。据报道，其性能与 **Deepseek R1** 相当，但体积显著更小。
   - 一位成员指出，所提供的基准测试是用户偏好基准测试（**ChatArena**），而非客观指标。
- **AI Scientist 发表首篇论文**：来自 **Sakana AI** 的首篇由 AI 生成的论文已通过 **ICLR** workshop 的同行评审，详见其[出版物](https://sakana.ai/ai-scientist-first-publication/)。
- **CARL 成为首个产出学术同行评审研究的 AI 系统**：另一个来自 **Auto Science AI** 名为 **CARL** 的 AI 系统也产出了经过学术同行评审的研究，详见[这篇博客文章](https://www.autoscience.ai/blog/meet-carl-the-first-ai-system-to-produce-academically-peer-reviewed-research)。
- **关于 MoE 公平性的辩论**：一位成员发起了一场关于如何从“公平性”角度比较 **MoE** 模型与 dense 模型的讨论，质疑应该比较总参数量还是激活参数量。
- **Half-Life 2 RTX Demo 发布**：一位成员分享了一段 [YouTube 视频](https://www.youtube.com/watch?v=j31ISEd8xRM)，展示了具有全景光线追踪和 DLSS 4 的 **Half-Life 2 RTX** demo，该 demo 是通过 **RTX Remix** 重新制作的。
   - 另一位成员表达了对《超世纪战警：逃离屠夫湾》（Chronicles of Riddick: Escape from Butcher Bay）RTX 版本的期待。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://sakana.ai/ai-scientist-first-publication/">未找到标题</a>: 未找到描述</li><li><a href="https://www.autoscience.ai/blog/meet-carl-the-first-ai-system-to-produce-acade">Autoscience</a>: 未找到描述</li><li><a href="https://www.autoscience.ai/blog/meet-carl-the-first-ai-system-to-produce-academically-peer-reviewed-research">Autoscience</a>: 未找到描述</li><li><a href="https://ai.google.dev/gemma/docs/core">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/4MvGnmmP3c0">Gemini Robotics: Bringing AI to the physical world</a>: 我们全新的 Gemini Robotics 模型将 Gemini 2.0 带入物理世界。这是我们最先进的视觉语言动作模型，使机器人能够进行交互……</li><li><a href="https://www.youtube.com/watch?v=j31ISEd8xRM">Half-Life 2 RTX | Demo with Full Ray Tracing and DLSS 4 Announce</a>: 以从未有过的方式重温具有开创性、广受好评的《半条命 2》，通过 RTX Remix 重新构思。包含全景光线追踪、重制资产以及……的 Demo。
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1349471895195615243)** (56 messages🔥🔥): 

> `GPT-4 vs local LLMs, Ollama vs GPT4All, Deepseek 14B, Web crawling, LocalDocs` 


- **用户发现 GPT-4 的质量优于本地 LLM**：一位用户指出，**ChatGPT premium** 与 **GPT4All** 上的 **LLM** 相比，在质量上存在巨大差异，这归因于本地可用的模型参数规模较小。
   - 他们表示希望有一种本地模型能够匹配 **ChatGPT premium** 在处理上传文档时的准确度，并提到他们在 **GPT4All** 上尝试过的模型准确度都不太理想。
- **多模型服务器选择 Ollama 还是 GPT4All？**：一位用户询问，在计算资源有限的情况下，对于需要管理多个模型、快速加载/卸载、频繁更新 **RAG** 文件以及调用日期/时间/天气 API 的服务器，应该使用 **GPT4All** 还是 **Ollama**。
   - 另一位成员建议使用 **Deepseek 14B** 或类似模型，并提到**大上下文窗口**（4k+ tokens）对于吸收文档等更多信息的重要性，同时提到 Apple 硬件的表现比较特殊。
- **GPT4All 工作流尚可但 GUI 缺乏多模型支持**：一位成员建议尝试使用小型模型来测试 **GPT4All** 的加载、卸载以及配合 **LocalDocs** 进行 **RAG** 的工作流，并指出其 GUI 不支持同时运行多个模型。
   - 他们建议使用本地服务器或 Python 端点，但这需要为 pipeline 和编排编写自定义代码。
- **寻求 Web Crawling 建议**：一位用户询问如何实现网页爬取（Web Crawling），并在开始尝试前寻求建议。
   - 一位成员提到一个关于 **Brave browser** 兼容性的 PR，由于存在 bug 以及开发重心转向不同的 tool-calling 方案而未被合并，但如果需求量大，可以重新启动该项目。
- **LocalDocs 片段截图方案**：针对 **LocalDocs** 以纯文本显示片段的问题，一位成员建议用户可以将截图保存为 PDF，对图像进行 OCR 处理，然后在数据库中搜索该片段。
   - 他们建议在这个工作流中使用 docfetcher。


  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1349338140057276489)** (51 messages🔥): 

> `Mastra AI framework, Gemini 2.0 Flash Experimental, Jina AI's DeepSearch/DeepResearch, Cohere's Command A, Gemini Deep Research` 


- ****Mastra** Typescript AI 框架发布**：[Mastra](https://mastra.ai/) 是一个 Typescript AI 框架，旨在为产品开发者提供一个稳健且有趣的框架，目标是超越 Langchain/Graph 等框架。
   - 创始人此前曾就职于 Gatsby 和 Netlify，他们强调了**类型安全（type safety）**，并专注于量化的性能提升而非定性的主观意见。
- ****Gemini 2.0 Flash Experimental** 支持原生图像生成**：**Gemini 2.0 Flash Experimental** 现在支持[原生图像生成](https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/)，允许模型从文本和多模态输入中创建图像，增强了其推理和理解能力。
   - 用户分享了令人印象深刻的结果，其中一人惊叹道 *“我简直无话可说，这效果太棒了。搞什么鬼”*，另一人注意到它在 “BASE” 这个单词中准确添加了字母 “D”。
- ****Jina AI** 的 DeepSearch 表现出色**：Jina AI 强调了[改进 DeepSearch/DeepResearch 的实用技术](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)，重点在于用于片段选择的 **late-chunking embeddings**，以及在爬取前使用 **rerankers** 对 URL 进行优先级排序。
   - 他们是 Latent Space 播客的忠实粉丝，表示 *“我们今年一定要找机会邀请他们”*。
- ****Cohere** 凭借 Command A 模型引起关注**：Cohere 推出了 [Command A](https://x.com/aidangomez/status/1900169306987524440)，这是一个拥有 **111B 参数的开源权重模型**，具有 **256k 上下文窗口**，专为 Agentic、多语言和编程用例设计。
   - 这个新模型是 Command R+ 的继任者，旨在各种任务中提供卓越的性能。
- ****Gemini** 向所有人开放 Deep Research**：**Gemini App** 现在向[所有用户免费提供 Deep Research](https://x.com/OfficialLoganK/status/1900224377389465751)，由 **Gemini 2.0 Flash Thinking** 提供支持，并结合搜索历史和 Gems 提供个性化体验。
   - 此次更新使更广泛的用户群体能够获得先进的推理能力。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/OfficialLoganK/status/1900224377389465751">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：@GeminiApp 今日重大更新：- Deep Research 现已向所有用户免费开放 - Deep Research 现由 2.0 Flash Thinking（我们的推理模型）驱动 - 新的 Gemini 个性化功能使用 y...</li><li><a href="https://mastra.ai/blog/the-next-million-ai-developers">为下一百万名 AI 开发者提供的框架</a>：下一代 AI 产品将使用以 Typescript 编写的 API 构建</li><li><a href="https://x.com/19kaushiks/status/1899856652666568732?s=46">Kaushik Shivakumar (@19kaushiks) 的推文</a>：非常激动今天能将 Gemini 的原生图像生成功能推向公开实验阶段 :) 我们取得了很大进展，但仍有很长的路要走，请给我们反馈！是的，我制作了图像...</li><li><a href="https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/">DeepSearch/DeepResearch 中的片段选择和 URL 排名</a>：搞定这两个细节能让你的 DeepSearch 从平庸走向卓越（GOAT）：从冗长的网页中选择最佳片段，并在抓取前对 URL 进行排名。</li><li><a href="https://x.com/m__dehghani/status/1900070436689334434?s=61">Mostafa Dehghani (@m__dehghani) 的推文</a>：模型有时会自行进入自我批判循环，但你也可以手动触发，模型会通过自我对话为自己调整 Prompt。[添加例如 "验证图像，..."]</li><li><a href="https://x.com/scaling01/status/1899873556340859302">Lisan al Gaib (@scaling01) 的推文</a>：将军。引用 Greg Brockman (@gdb)：一张 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有如此多值得探索的地方。团队正在努力将这些功能带给世界。</li><li><a href="https://x.com/scaling01/status/1899977861844377820?s=46">Lisan al Gaib (@scaling01) 的推文</a>：我会思考这次体验一段时间。引用 Lisan al Gaib (@scaling01)：天哪 —— 看完整个视频。我问 Gemini 2.0 生命的意义是什么，但它只被允许...</li><li><a href="https://x.com/scaling01/status/1899873556340859302?s=46">Lisan al Gaib (@scaling01) 的推文</a>：将军。引用 Greg Brockman (@gdb)：一张 GPT-4o 生成的图像 —— 仅 GPT-4o 的图像生成能力就有如此多值得探索的地方。团队正在努力将这些功能带给世界。</li><li><a href="https://x.com/OfficialLoganK/status/1899914266062577722">Logan Kilpatrick (@OfficialLoganK) 的推文</a>：在 Google AI Studio 和 Gemini API 中引入 YouTube 视频 🎥 链接支持。你现在可以直接传入 YouTube 视频，模型可以利用其原生视频理解能力来使用...</li><li><a href="https://x.com/m__dehghani/status/1899854209081868663?s=46">Mostafa Dehghani (@m__dehghani) 的推文</a>：任何待过这个房间的人都知道，这里从来不是平凡的一天！这个空间见证了混乱与天才的极端！...而且我们发布了！https://developers.googleblog.com/en/experiment-wi...</li><li><a href="https://x.com/emollick/status/1900056829683462234?s=61">Ethan Mollick (@emollick) 的推文</a>：使用 Gemini Flash Experimental 通过添加冰淇淋来毁掉艺术。</li><li><a href="https://x.com/angaisb_/status/1899852603107721388?s=61">angel⭐ (@Angaisb_) 的推文</a>："原生图像生成如何优于当前模型？"</li><li><a href="https://x.com/amanrsanger/status/1899659103473123777?s=46">Aman Sanger (@amanrsanger) 的推文</a>：Cursor 在语义搜索上训练了一个 SOTA 嵌入模型。它大幅超越了竞争对手使用的开箱即用嵌入模型和重排序器！在使用 Agent 时，你可以感受到这种差异！</li><li><a href="https://x.com/sullyomarr/status/1899891905892405551?s=46">Sully (@SullyOmarr) 的推文</a>：等等。原生文本生成图像有点疯狂。</li><li><a href="https://x.com/aidangomez/status/1900169306987524440">Command A(idan) (@aidangomez) 的推文</a>：今天 @cohere 非常激动地推出 Command A，这是我们继 Command R+ 之后的新模型。Command A 是一个拥有 111B 参数和 256k 上下文窗口的开放权重模型，专注于提供卓越的性能...</li><li><a href="https://x.com/fofrai/status/1899927094727000126?s=46">fofr (@fofrAI) 的推文</a>：我必须尝试一下。具有图像输出功能的 Gemini 2.0 Flash Experimental 🤯 引用 apolinario 🌐 (@multimodalart)：简直是。下一个。级别。</li><li><a href="https://x.com/andrew_n_carr/status/1899940624079753265?s=61">Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：原生多模态（Native multimodal）是未来</li><li><a href="https://x.com/kalomaze/status/1900028234542243992?s=46">kalomaze (@kalomaze) 的推文</a>：我真的无言以对这效果有多好。搞什么鬼。</li><li><a href="https://x.com/krishnanrohit/status/1899901748946555306?s=61">rohit (@krishnanrohit) 的推文</a>：真正潜在的问题是人类绝对热爱垃圾内容（slop）</li><li><a href="https://x">

.com/goodside/status/1899895643352510609?s=61">来自 Riley Goodside (@goodside) 的推文</a>：Gemini 2.0 Flash Experimental 现在支持原生图像输出，在上传的图像中，它为一名穿着 RLHF shoggoth 服装的派对参与者 T 恤上的单词 “BASE” 加上了 “D”：</li><li><a href="https://developers.googleblog.com/en/experiment-with-gemini-20-flash-native-image-generation/">使用 Gemini 2.0 Flash 原生图像生成进行实验</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1899887925103648933?s=46&t=Z6mP_1pHALnIw7k1lFkdwQ">来自 Andrej Karpathy (@karpathy) 的推文</a>：@divya_vikash 请让它停下来</li><li><a href="https://x.com/kalomaze/status/1900023138546835619?s=46">来自 kalomaze (@kalomaze) 的推文</a>：这真的让我大受震撼</li><li><a href="https://x.com/robertriachi/status/1899854394751070573?s=61">来自 Robert Riachi (@robertriachi) 的推文</a>：一些关于 Gemini 2.0 原生图像输出的酷炫示例 🧵</li><li><a href="https://x.com/ilumine_ai/status/1900041501624971601?s=61">来自 Cristian Peñas ░░░░░░░░ (@ilumine_ai) 的推文</a>：Gemini 也可以生成相当一致的 gif 动画：“通过生成多个帧来创建一个动画，展示一颗种子长成植物然后开花的过程，采用像素艺术风格...”</li><li><a href="https://x.com/aidenybai/status/1899840110449111416?s=46">来自 Aiden Bai (@aidenybai) 的推文</a>：介绍 Same.dev，以像素级精度克隆任何网站。One-shot 克隆 Nike、Apple TV、Minecraft 等网站！</li><li><a href="https://share.snipd.com/episode/3267b9f3-0048-42c4-8808-92fb357d097f">Sam Altman，OpenAI CEO</a>：Sam Altman，OpenAI CEO
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1349408279750774794)** (3 条消息): 

> `Model Context Protocol, WeAreDevs WebDev & AI Day, LLM x Law Hackathon` 


- **LlamaIndex 接入 Model Context Protocol**：LlamaIndex 宣布他们正在密切关注 **Model Context Protocol** 的进展，这是一项旨在让工具的发现和使用变得简单的开源工作，用户现在可以使用任何兼容 MCP 的服务器所提供的工具，详见[这条推文](https://twitter.com/llama_index/status/1899848532817035529)。
- **AI 在 WeAreDevs WebDev & AI Day 改变 Web 开发**：行业专家将参加 @WeAreDevs WebDev & AI Day，讨论 **AI 在平台工程和 DevEx 中的作用**，以及 AI 驱动世界中开发者工具的未来，详见[这条推文](https://twitter.com/llama_index/status/1900232326132773026)。
- **创新者集结斯坦福大学第 5 届 LLM x Law 黑客松**：第 5 届 **LLM x Law Hackathon** 将于 4 月 6 日在 @stanford 举行，汇聚创新者为法律工作开发 AI 解决方案，并向风险投资人展示正在进行的项目，根据[这条推文](https://twitter.com/llama_index/status/1900246964522148344)。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1349379163563495466)** (46 messages🔥): 

> `LlamaExtract 本地部署, 新的 Response API, LlamaParse 与图像, AzureMultiModal 聊天 bug, RAG 中的深度研究` 


- **LlamaExtract 通过本地部署保障隐私安全**：对于企业级应用，整个 **LlamaCloud platform** 都支持本地部署 (on-premise)/BYOC 部署，尽管其成本通常远高于使用 SaaS 版本。
- **即将支持新的 Response API**：团队正在努力支持新的 **Response API**，如果用户选择开启，该 API 承诺将利用搜索工具丰富结果。
- **LlamaParse 现已支持强大的 JSON 输出**：**LlamaParse** 已在其 JSON 输出中包含图像，提供图像下载链接以及用于拼接内容的布局信息。
- **AzureMultiModal 聊天问题已解决**：有用户报告了 **AzureMultiModal chat** 的问题及潜在 bug，但问题最终追溯到依赖项过时以及多模态断言 (assert) 问题，该问题已在 [此 PR](https://github.com/run-llama/llama_index/pull/18112) 中修复。
- **深度研究 RAG 已准备就绪**：**RAG** 中的深度研究 (Deep research) 能力已通过 `npx create-llama@latest` 的深度研究选项提供，该工作流的源代码可在 [GitHub](https://github.com/run-llama/create-llama/blob/ee69ce7cc10db828424b468e7b54b3f06b18e22c/templates/components/agents/python/deep_research/app/workflows/deep_research.py) 上获取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/pull/18112">由 logan-markewich 提交的移除多模态 LLM 断言的 Pull Request #18112 · run-llama/llama_index</a>：修复了 #18111。由于基础 LLM 类开始支持图像，许多多模态 LLM 已转换为包装其原生 LLM 类。这意味着即使功能正常，此断言也会失败...</li><li><a href="https://github.com/run-llama/create-llama/blob/ee69ce7cc10db828424b468e7b54b3f06b18e22c/templates/components/agents/python/deep_research/app/workflows/deep_research.py">create-llama/templates/components/agents/python/deep_research/app/workflows/deep_research.py 源码 (位于提交 ee69ce7) · run-llama/create-llama</a>：入门 LlamaIndex 最简单的方法。通过在 GitHub 上创建账号为 run-llama/create-llama 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1349440295955071007)** (7 messages): 

> `测验截止日期, 实验室与研究机会, 项目时间表, 非伯克利学生的证书获取` 


- **测验将于 5 月进行**：成员表示所有 **quiz deadlines** 都在 **5 月**，具体细节将很快公布。
   - 有人提到，感兴趣的人已经加入了邮件列表，记录显示他们已经打开了宣布 **Lecture 6** 的最新邮件。
- **研究机会与实验室即将公布**：一位成员询问了针对 **MOOC** 学习者的 **labs** 和 **research opportunities** 计划。
   - 另一位成员回答说，一旦一切敲定就会发布公告。
- **项目有明确的时间表**：一位成员询问 **projects** 是否需要一段时间才能推出。
   - 另一位成员确认细节将很快发布，并建议大家跟上 **weekly quizzes** 的进度。
- **非伯克利学生：仍能获得证书吗？**：一位成员询问非伯克利学生是否仍可以通过完成作业获得证书。
   - 另一位成员确认细节将很快发布。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1349563534937161798)** (4 messages): 

> `LLM 角色, LLM 人格, 决策研究小组` 


- **阐明 **Roles** 与 **Personas****：一位成员解释说，在查询 LLM 时，**roles** 是用于编辑 Prompt 的结构，例如 **system**、**user** 或 **assistant**；而 **persona** 则被定义为提供给系统的通用指南的一部分，影响 Assistant 的行为方式。
   - System 角色内容的形式是关于 Assistant 应该如何行动的通用指南；User 和 Assistant 则是参与主动交互的角色。
- **决策研究小组邀请新成员**：一位成员发布了一个专注于 **decision making** 和 **memory tracks** 的研究小组的新邀请链接。
   - 分享了 [Discord 邀请链接](https://discord.gg/pqWzyfCX)，未提供更多上下文。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1349586274284998708)** (2 条消息): 

> `Mojo 与 Max 捆绑，Windows 上的 Mojo` 


- **Mojo 提问：为什么要与 Max 捆绑？**：一位用户在 [Modular 论坛](https://forum.modular.com/t/mojo-and-max-why-bundle-them/751) 询问了将 **Mojo** 与 **Max** 捆绑在一起的潜在原因。
   - 该问题引发了关于此类捆绑包潜在协同效应和用户利益的讨论。
- **Mojo 会支持 Windows 吗？**：同一位用户还询问了 **Mojo 在 Windows 上推出的可能性**。
   - 这引发了人们对扩展 Mojo 平台支持相关的挑战和时间表的关注。



**提及的链接**：<a href="https://forum.modular.com/t/mojo-and-max-why-bundle-them/751">Mojo and Max, why bundle them?</a>：我最近使用 magic init life --format mojoproject 启动了一个项目，但在查看依赖项后，我发现了：max 25.2.0.dev2025030905 release 9.7 KiB co...

  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1349467285911240827)** (4 条消息): 

> `Modular Max PR，捕获闭包` 


- **Modular Max 获得进程生成能力**：一位成员分享了 **Modular Max** 的一个 [PR](https://github.com/modular/max/pull/3998)，该 PR 添加了使用 `exec` 从可执行文件生成和管理进程的功能。
   - 然而，这取决于基础 PR 的合并以及 **Linux exec** 问题的解决，因此可用性尚不确定。
- **已提交捕获闭包 Bug**：一位成员提交了一个与 `capturing` 闭包相关的 [语言设计 Bug](https://github.com/modular/max/issues/4143)。
   - 另一位成员也表达了同样的看法，指出他们也觉得这种行为很奇怪。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modular/max/issues/4143)">modular/max</a>：MAX 平台（包含 Mojo）。通过在 GitHub 上创建账号来为 modular/max 的开发做出贡献。</li><li><a href="https://github.com/modular/max/pull/3998">[stdlib] Adds functionality to spawn and manage processes from exec. file by izo0x90 · Pull Request #3998 · modular/max</a>：此 PR 的基础已在此奠定，它添加了所需的低级实用程序：为 Mojo 的 cLib 绑定添加了 vfork、execvp、kill 系统调用工具；为文件描述符添加了 read_bytes。一旦该 PR 合并...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1349834087518703738)** (1 条消息): 

> `MutableInputTensor 可见性，Mojo nightly 文档，max.tensor API` 


- **MutableInputTensor 类型别名缺失？**：一位用户报告在 [nightly 文档](https://docs.modular.com/max/api/mojo/tensor/managed_tensor_slice/) 中找到了 `MutableInputTensor` 类型别名，但它似乎并未公开暴露。
   - 该用户尝试通过 `from max.tensor import MutableInputTensor` 和 `from max.tensor.managed_tensor_slice import MutableInputTensor` 进行导入，但未获成功。
- **Mojo Nightly 文档**：用户在寻找 `MutableInputTensor` 类型别名时参考了 [Mojo nightly 文档](https://docs.modular.com/max/api/mojo/tensor/managed_tensor_slice/)。

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1349785900124602499)** (5 messages): 

> `AST Evaluation, Function Calling Leaderboard, LLM Integration, Parallel Function Calls` 


- ****AST** 评估全面检查正确性**：**AST** (Abstract Syntax Tree) 评估检查函数调用的正确性，包括函数名、参数类型以及参数值是否在可能范围内，详见 [V1 博客](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics)。
   - 表格中 **AST** 的数值代表所有这些标准均正确的*测试用例百分比*。
- ****BFCL** 是一个全面的评估**：**[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL)** 更新于 **2024-08-19** ([Change Log](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CHANGELOG.md))，是首个针对 LLMs 调用函数和工具能力的全面评估。
   - 该排行榜旨在代表 Agent 和企业工作流中典型的用户函数调用用例。
- ****LLMs** 通过函数调用驱动应用**：**GPT**、**Gemini**、**Llama** 和 **Mistral** 等大语言模型 (**LLMs**) 正越来越多地通过函数调用（也称为工具调用）功能集成到 **Langchain**、**Llama Index**、**AutoGPT** 和 **Voyager** 等应用中。
   - 这些模型在驱动各种应用程序和软件方面展示了巨大的潜力。
- ****Function Calls** 可以是并行的**：评估中考虑的函数调用包括多种形式，例如 *parallel*（一个函数输入，多个函数输出调用）和 *multiple* 函数调用。
   - 这种全面的方法涵盖了常见的函数调用用例。



**Link mentioned**: <a href="https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html#metrics">Berkeley Function Calling Leaderboard</a>: no description found

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/1349433953139822734)** (2 messages): 

> `Evaluation tools, Datasets availability` 


- **集中追踪评估工具**：一名成员询问是否有集中位置来追踪用于 **evaluation** 的所有工具，并引用了 **Gorilla** 仓库中的一个特定目录。
   - 另一名成员回复称所有数据集都可以在 **/gorilla/berkeley-function-call-leaderboard/data** 文件夹中找到。
- **Gorilla 数据集的可用性**：明确了对于多轮对话类别，函数/工具文档存储在 **/gorilla/berkeley-function-call-leaderboard/data/multi_turn_func_doc** 中以避免重复。
   - 所有其他类别的数据集，其函数文档都存储在数据集文件本身中。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1349447063867228295)** (4 messages): 

> `DSPy Caching, Pluggable Cache Module, Cache Invalidation Strategies, Selective Caching, Monitoring Cache Hit/Miss Rates` 


- **DSPy 关注可插拔缓存模块**：DSPy 正在开发一个**可插拔的 Cache 模块**，初步工作可见于 [此 pull request](https://github.com/stanfordnlp/dspy/pull/1922)。
- **缓存策略寻求灵活性**：用户希望在定义**缓存策略**方面有更多灵活性，特别是针对 **context caching** 以降低成本并提高速度。
- **缓存失效引起关注**：讨论包括对带有 **TTL 过期**或 **LRU 驱逐**的**缓存失效**机制的兴趣。
- **选择性缓存受到关注**：还讨论了基于**输入相似性**的**选择性缓存**，以避免冗余的 API 调用。
- **缓存监控被认为很有帮助**：内置的**缓存命中/错过率监控**被提议作为新缓存模块的一个实用功能。



**Link mentioned**: <a href="https://github.com/stanfordnlp/dspy/pull/1922">Feature/caching by hmoazam · Pull Request #1922 · stanfordnlp/dspy</a>: One single caching interface which has two levels of cache - in memory lru cache and fanout (on disk)

  

---

### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/1349694699245404203)** (1 messages): 

> `ColBERT endpoint, MultiHop program, Connection Refused` 


- **ColBERT 端点抛出 Connection Refused 错误**：一名成员报告位于 `http://20.102.90.50:2017/wiki17_abstracts` 的 **ColBERT 端点** 似乎已下线，抛出 *Connection Refused* 错误。
   - 当尝试使用基础的 **MultiHop 程序** 检索段落时，端点返回了 **200 OK** 响应，但文本内容包含与连接 `localhost:2172` 相关的错误消息。
- **MultiHop 程序因连接问题失败**：用户提到他们的 **MultiHop 程序** 直到昨天还在正常工作，但现在无法检索段落。
   - 程序收到 **200 OK** 响应，但内容显示连接端口 **2172** 上的服务器出错，表明 ColBERT 服务存在问题。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1349577306074513508)** (1 messages): 

> `LSTM Model issues, NaN loss debugging, TinyJit integration` 


- **LSTM 模型输出 NaN Loss**：一名成员在配合 **TinyJit** 运行 **LSTMModel** 时遇到了 **NaN** loss，在第一步之后 loss 就从一个很大的数值变成了 **NaN**。
   - 该模型使用 `nn.LSTMCell` 和 `nn.Linear`，并使用 `Adam` 优化器进行训练，输入数据包含一个较大的值 (**1000**)，这可能是导致问题的原因。
- **在 tinygrad 中排查 NaN Loss**：一名成员正在寻求有关 **tinygrad** 训练期间打印为 **NaN** 的 loss 的调试帮助。
   - 提供的代码示例展示了一个 **LSTM** 设置，暗示了可能导致 **NaN** 的潜在数值不稳定性问题或梯度爆炸问题。


  

---


### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1349433375697666059)** (1 messages): 

> `Pinecone Limitations, RAG Changes, VPC Deployment` 


- **Pinecone 性能骤降，RAG 重启**：一名成员指出他们的 RAG 系统之前使用 **Pinecone**，但存在 **性能限制**。
   - 他们还提到缺乏对 **VPC 部署** 的支持，这导致他们寻求不同的解决方案。
- **RAG 系统从 Pinecone 迁移**：由于 **性能限制和缺乏 VPC 部署支持**，一个 RAG 系统正在从 **Pinecone** 迁移。
   - 解决上述两个约束后，新系统可能会表现更好。


  

---


---


{% else %}


> 完整的逐频道分析已为邮件格式截断。
> 
> 如果您想查看完整分析，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}