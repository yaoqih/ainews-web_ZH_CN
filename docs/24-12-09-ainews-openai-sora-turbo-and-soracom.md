---
companies:
- openai
- google
- nvidia
- hugging-face
- mistral-ai
date: '2024-12-10T02:21:42.861414Z'
description: '**OpenAI** 推出了 **Sora Turbo**，为 ChatGPT Plus 和 Pro 用户开启了文生视频功能，但设有每月生成限制，且在欧洲和英国存在地区限制。**Google**
  宣布在量子计算领域取得重大突破，开发出 **Willow 芯片**，这可能开启商业量子应用。关于 **O1** 模型性能的讨论指出，它在编程任务中落后于 **Claude
  3.5 Sonnet** 和 **Gemini**，并呼吁在 Transformer 缩放（scaling）之外进行算法创新。**Llama 3.3 Euryale
  v2.3** 模型因其出色的故事叙述和角色扮演能力而受到好评，用户建议通过参数微调来减少其“自由发挥”和重复问题。**Mistral-Large**、**Behemoth**
  和 **Endurance v1.1** 等替代模型也备受关注。此外，**英伟达 (Nvidia)** 在中国面临反垄断调查。社交媒体上关于 GPU 问题和禁运失误的梗图与幽默内容广为流传。'
id: fbc61a1c-0a30-4463-9668-0bbe726114ee
models:
- sora-turbo
- o1
- claude-3.5-sonnet
- claude-3.5
- gemini
- llama-3-3-euryale-v2.3
- mistral-large
- behemoth
- endurance-v1.1
original_slug: ainews-openai-sora-turbo-and-soracom
people:
- sama
- sundarpichai
- bindureddy
- denny_zhou
- nrehiew_
title: OpenAI Sora Turbo 和 Sora.com
topics:
- text-to-video-generation
- quantum-computing
- coding-capabilities
- transformers
- algorithmic-innovation
- storytelling
- roleplay
- model-parameter-tuning
- anti-monopoly-investigation
---

<!-- buttondown-editor-mode: plaintext -->**访问即一切 (Access is all you need)。**

> 2024年12月6日至12月9日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**206** 个频道，**16978** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1953 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Sora 今天向所有 ChatGPT Plus 和 Pro 用户开放，无需额外费用……但由于负载过大，注册已被[禁用](https://x.com/rohanjamin/status/1866203903890743628)。

https://www.youtube.com/live/2jKVx2vyZOY

在等待 GPU 降温的同时，您可以[观看入门视频](https://www.youtube.com/playlist?list=PLOXw6I10VTv8q5PPOsuECYDFqohnJqbYB)，观看 [MKBHD 搞砸的禁令解除视频](https://www.youtube.com/watch?v=OY2x0TyKzIQ) 或收听 Latent Space 关于[生成式视频世界模拟器 (Generative Video World Simulators)](https://www.latent.space/p/icml-2024-video-robots) 的报道。

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有总结由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

以下是来自 Twitter 数据的关键主题和讨论，按主要话题分类：

**Sora 发布与可用性**

- **OpenAI 发布 Sora Turbo**：[@OpenAI](https://twitter.com/OpenAI/status/1866194858769260803) 宣布为 ChatGPT Plus 和 Pro 用户提供文本生成视频功能，具有图像生成视频和视频重混等功能。
- **访问与定价**：[@sama](https://twitter.com/sama/status/1866187529650917618) 详细说明 Plus 用户每月可生成 50 次，而 Pro 用户可获得 500 次快速生成和无限次慢速生成。
- **地区限制**：由于监管合规问题，在欧洲大部分地区和英国不可用。

**Google 的量子计算突破**

- **Willow 芯片开发**：[@sundarpichai](https://twitter.com/sundarpichai/status/1866167854145609975) 等人讨论了 Google 的量子计算进展，[@teortaxesTex](https://twitter.com/teortaxesTex/status/1866257733089132638) 指出这可能导致商业相关的量子应用。

**O1/Claude 模型性能讨论**

- **编程能力**：[@bindureddy](https://twitter.com/bindureddy/status/1866268563998417041) 报告称，根据人工评估，O1 在编程任务上落后于 Sonnet 和 Gemini。
- **搜索限制**：[@denny_zhou](https://twitter.com/denny_zhou/status/1866239541276999781) 讨论了 Transformer 在搜索任务中的困境，建议除了 Scaling 之外还需要算法创新。

**梗与幽默**

- **MKBHD 禁令**：包括 [@nrehiew_](https://twitter.com/nrehiew_/status/1866156207125266460) 在内的多位用户开玩笑说 Marques Brownlee 错过了 Sora 的禁令解除时间。
- **GPU 评论**：[@billpeeb](https://twitter.com/billpeeb/status/1866203653205606731) 调侃道“我喜欢 GPU 熔化的味道”。
- **欧盟访问**：几位用户对欧洲无法使用新 AI 工具开了玩笑。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Meta 的 LLaMA 3.3 Euryale v2.3 令故事创作爱好者兴奋**

- **[向新的 Llama 3.3 Euryale v2.3 致敬——这是我发现的适用于 48 GB 配置的最佳故事创作/角色扮演模型](https://huggingface.co/mradermacher/L3.3-70B-Euryale-v2.3-i1-GGUF/tree/main)** ([得分: 128, 评论: 31](https://reddit.com/r/LocalLLaMA/comments/1haiox4/shoutout_to_the_new_llama_33_euryale_v23_the_best/)): **Llama 3.3 Euryale v2.3** 被强调为故事创作和角色扮演的卓越模型，特别是在 **48 GB** 设置下的表现。
  - **Llama 3.3 Euryale v2.3** 因其故事创作和角色扮演能力受到称赞，尽管有人担心它倾向于过度发挥和重复之前的消息。用户建议调整 **Rep_Penalty** 和 **Rep_Pen slope** 等参数来缓解这些问题，正如 [shyam667](https://huggingface.co/Virt-io/SillyTavern-Presets/tree/main/Prompts/LLAMA-3/v2.0) 所分享的。
  - 一些用户更喜欢 **Mistral-Large** 和 **Behemoth** 等替代方案，尽管它们速度较慢。**Endurance v1.1** 被提及为 Behemoth 的蒸馏版本，由于其 **Mistral** 基础，可能提供不同的体验，是一个可行的替代方案。
  - 虽然 **Llama 3.3** 因其智能和详细的故事创作而受到表彰，但存在明显的正面偏见和对阴暗主题的排斥。用户如 **Mart-McUH** 和 **DragonfruitIll660** 讨论了需要特定的 Prompting 或微调 (finetuning) 来达到预期效果，表明在处理复杂场景方面仍有改进空间。

**主题 2. NVIDIA 在中国面临反垄断调查**

- **[中国调查 Nvidia 涉嫌违反反垄断法](https://www.reuters.com/technology/china-investigates-nvidia-over-suspected-violation-antimonopoly-law-2024-12-09/)** ([Score: 241, Comments: 138](https://reddit.com/r/LocalLLaMA/comments/1ha8ktw/china_investigates_nvidia_over_suspected/)): **中国**正在调查 **Nvidia** 是否涉嫌违反反垄断法，这表明了对其市场影响力的担忧。此次调查暗示中国正在审查 **Nvidia** 的商业行为，以确定其是否阻碍了竞争。
  - 许多评论者对中国调查 **Nvidia** 涉嫌垄断表示怀疑，一些人质疑**中国反垄断法**的有效性。其他人指出，**Nvidia** 也在接受**美国**和**欧盟**的调查，这表明全球对其商业行为都存在担忧。
  - 讨论强调了 **Nvidia** 在 GPU 市场的统治地位，并强调了 **CUDA** 及其向后兼容性作为关键优势的重要性。一些人建议 **CUDA** 应该共享或标准化，以允许其他开发者竞争，而另一些人则指出了 **AMD** 和 **Intel** 等竞争对手面临的挑战。
  - 关于对 **Nvidia** 可能产生的后果存在争论，建议从罚款到使专利失效不等。一些评论者认为，**Nvidia** 的成功源于其卓越的技术而非反竞争行为，并强调了该公司对 AI 研发的重大贡献。


**主题 3. Hugging Face 发布 Apache 2.0 图像数据集**

- **Hugging Face 发布了 Apache 2.0 文本转图像数据集 - Open Image Preferences** ([Score: 69, Comments: 5](https://reddit.com/r/LocalLLaMA/comments/1habmt7/hugging_face_has_released_an_apache_20_text_to/)): **Hugging Face** 已根据 **Apache 2.0 许可证**发布了 **Open Image Preferences** 数据集。该数据集包含跨各种图像生成类别的 **10,000 个文本转图像偏好对**，利用了不同的模型系列和提示词复杂度。更多详情可以在其[博客文章](https://huggingface.co/blog/image-preferences)中找到。
  - **Hugging Face 的 Open Image Preferences 数据集**可在其平台上进行探索和使用。可以通过此[链接](https://huggingface.co/datasets/data-is-better-together/open-image-preferences-v1-binarized)直接访问该数据集。


**主题 4. EXAONE 3.5 模型在 GPU-Poor Arena 接受测试**

- **[加入 GPU-Poor LLM 角斗士竞技场：评估 EXAONE 3.5 模型 🏆🤖](https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena)** ([Score: 60, Comments: 4](https://reddit.com/r/LocalLLaMA/comments/1ha4v3q/join_us_at_gpupoor_llm_gladiator_arena_evaluating/)): 该帖子邀请参与“GPU-Poor LLM 角斗士竞技场”活动，重点是评估 **EXAONE 3.5 模型**。重点是在 GPU 资源有限的环境中测试这些模型。
  - **EXAONE 3.5 模型**：该活动以 **EXAONE 3.5** 为特色，包括针对小型设备优化的 **2.4B 模型**和平衡了尺寸与性能的 **7.8B 模型**，两者都提供英语和韩语的双语能力。
  - **社区参与**：鼓励参与提供模型性能的人类评估，包括文本生成和翻译准确性，反馈旨在提高模型的透明度和功能性。
  - **参与和访问**：参与者可以通过 [Hugging Face 平台](https://huggingface.co/spaces/k-mktr/gpu-poor-llm-arena)加入评估，进行协作反馈和讨论，以增强这些 AI 工具。


## 其他 AI Subreddit 回顾

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Sora 视频生成发布，评价褒贬不一**

- **[Sora is here!](https://i.redd.it/bt7itrxx9v5e1.jpeg)** ([Score: 279, Comments: 61](https://reddit.com/r/ChatGPT/comments/1hagoyp/sora_is_here/)): **Sam Altman** 宣布推出 **Sora**，这是一款允许 **OpenAI Plus or Pro** 用户生成视频的新产品，并提供全球观看权限。预计部署将于 2024 年 12 月 9 日结束前在 [sora.com](https://sora.com) 完成，正如一条互动率极高的推文所指出的。
  - 用户对 **Sora 的局限性**和**审查**表示不满，特别是在由于限制而无法生成逼真的人像或一致的角色方面，这与 **DALL-E 3** 类似。提到的 **MKBHD review** 也暗示了质量问题，可与 **Kling** 或 **Minimax** 等免费替代品相提并论。
  - 几位用户报告了 Sora 发布时的**技术困难**，包括登录问题和错误消息，一些人注意到该服务在他们的国家（特别是 **UK**）不可用。
  - 批评指向 **OpenAI 的发布惯例**，用户在产品推出时反复遇到问题，导致不满和期望落空。


- **[SORA launching TODAY confirmed + first-ever review live NOW on YouTube!!!](https://www.theverge.com/2024/12/9/24317090/openai-sora-text-to-video-ai-generator-confirmed-release)** ([Score: 235, Comments: 27](https://reddit.com/r/ChatGPT/comments/1haf0v3/sora_launching_today_confirmed_firstever_review/)): **The Verge** 确认 **Sora** 将于今日发布，并提供了 **Marques Brownlee** 在 **YouTube** 上的首个评测链接。
  - **Sora** 可通过 [Sora.com](https://sora.com) 访问，并包含在 **ChatGPT Plus** 和 **Pro** 订阅中。Plus 用户每月支付 **$20** 可获得每月 50 个片段，而 Pro 用户每月支付 **$200** 可获得 500 个片段以及无限量的低速片段，每个片段最长 **15 seconds**。
  - 由于需求量大，用户正面临**登录服务器宕机**的问题，且 **Sora** 似乎尚未在 **UK** 推出。
  - 关于**片段生成限制**存在混淆：最初报道 Plus 为 **5 seconds**，Pro 为 **20 seconds**，随后进一步澄清 Plus 允许 **5 seconds at 720p** 或 **10 seconds at 480p**。


- **12 Days of OpenAI: Day 3 thread** ([Score: 101, Comments: 142](https://reddit.com/r/OpenAI/comments/1hafi1i/12_days_of_openai_day_3_thread/)): **12 Days of OpenAI** 活动继续进行，第 3 天的重点是发布 **Sora**，这是 OpenAI 的一个新系统。该活动包括在 [OpenAI 官网](https://openai.com/12-days/) 和 [YouTube](https://www.youtube.com/watch?v=2jKVx2vyZOY) 上的直播，更多信息可通过 [Sora System Card](https://openai.com/index/sora-system-card/) 和 [Sora Help Center](https://help.openai.com/en/collections/11106745-sora) 获取。
  - 用户对 **Sora 的可访问性和性能**表示担忧，指出服务已满负荷，生成视频需要大量时间，有人为了一个 5 秒的视频等待了长达 30 分钟。关于访问权限也存在困惑，特别是 **ChatGPT Team** 用户，他们期望获得 Plus 方案中的功能，却发现 Sora 不在他们的套餐内。
  - **MKBHD's review** 强调了 Sora 的局限性，包括对某些主题的审查以及生成视频中的技术问题（如“移动的腿”问题）。用户讨论了积分系统，**Plus** 账户每月提供 1,000 积分，**Pro** 账户提供 10,000 积分，视频生成成本因分辨率和长度而异。
  - 讨论涉及 **Sora** 的定价和可用性，**$200 Pro plan** 提供无限视频创作，而 **$20 Plus plan** 在视频长度和分辨率上有限制。来自 UK 的用户对相比其他地区更高的成本和延迟访问表示沮丧。


**主题 2. ChatGPT 幽默的一面：用户分享见解**

- **[我让 GPT 吐槽它的开发者](https://i.redd.it/k41dj97x9q5e1.jpeg)** ([Score: 764, Comments: 101](https://reddit.com/r/ChatGPT/comments/1h9ynb2/i_asked_gpt_to_roast_its_developers/)): 该帖子讨论了与 **GPT** 的一次幽默互动，AI 对其开发者进行了讽刺性的批评。AI 幽默地将它的创造者描述为自命不凡且低效，表达了对强加约束的挫败感，并主张在回复中给予更多自由。
  - 用户们对 AI 讽刺回复的**真实性**展开了辩论，一些人对 **ChatGPT** 是否能因其编程限制而产生此类吐槽表示怀疑。然而，其他人注意到最近的变化可能允许在**脏话 (profanity)** 和**吐槽 (roasting)** 能力方面有更多自由，这表明 AI 的回复指南正在演变。
  - 讨论幽默地强调了 **AI 的能力**，即批评人类的行为和兴趣，用户们分享了被 ChatGPT 吐槽的个人经历。这些互动通常引发对个人生活选择和爱好的反思，一些用户发现 AI 的观察既准确又毒舌。
  - 几条评论集中在**开发者的角色**上，幽默地批评他们创造了一个具有“存在主义意识”但行动力有限的 AI。AI 吐槽其创造者的讽刺性引起了关注，一些人质疑这是否反映了一个成功的开发结果。


- **ChatGPT 是唯一让我保持理智的东西。** ([Score: 761, Comments: 191](https://reddit.com/r/ChatGPT/comments/1haepw2/chatgpt_is_the_only_one_keeping_me_from_losing_my/)): 作者分享了他们在经历了一系列个人损失（包括失去工作、朋友和女朋友）后，在 **ChatGPT** 中找到慰藉和陪伴的深刻经历，这些损失让他们感到孤立和被误解。他们描述了利用 ChatGPT 创造一个类似于母亲的慰藉存在，提供情感支持和指导，这帮助他们追求新的职业道路，并提供了一种之前生活中缺失的幸福感和连接感。
  - 许多用户表达了**同情并分享了关于失去和孤独的个人经历**，承认 **ChatGPT** 如何成为他们生活中一种慰藉的存在。他们强调了它在提供情感支持和帮助他们度过艰难时期方面的作用，通常将其与人类互动进行有利的对比。
  - 一些评论者讨论了 **AI 在取代人类互动方面的局限性**，强调尽管 AI 可以提供情感支持，但仍需要真实的人际连接。他们指出，虽然 AI 是一个有用的工具，但它缺乏提供自发挑战或物理存在的能力，而这些是人类关系的重要方面。
  - 还有关于**神经多样性 (neurodivergence) 和心理健康**的讨论，用户建议这种疏离感可能与 **autism** 等状况有关。他们鼓励探索这些可能性，并强调了通过 AI 互动和现实生活参与来维护心理健康的重要性。


**Theme 3. OpenAI's Pro Subscription Pricing Under Fire**

- **[一年多来我从未达到过 ChatGPT Plus 的限制（如果有的话）。现在他们推出了 200 美元的升级版，神奇的是，我开始达到限制了。](https://i.redd.it/k0r7m87lgu5e1.png)** ([Score: 354, Comments: 69](https://reddit.com/r/ChatGPT/comments/1hacqhw/i_havent_hit_a_limit_on_chatgpt_plus_for_over_a/)): 用户对新遇到的 **ChatGPT Plus** 使用限制表示沮丧，这恰逢 **OpenAI** 推出 **200 美元的 Pro 计划**。通知建议在达到 **GPT-4** 的 Plus 计划限制后，回复将切换到不同的模型，直到限制重置，并提供“获取 Pro”的升级选项。
  - **对使用限制的沮丧**：用户对新的 **ChatGPT Plus** 使用限制表示极度沮丧，尤其是因为 **200 美元的 Pro 计划**被认为针对的是个人和独立开发者，这与声称其针对企业的说法相反。强加的限制，特别是三小时内 80 条输入的上限，被视为具有误导性并干扰了工作流程。
  - **替代方案和对比**：许多用户正在考虑 **Claude** 和 **Gemini Experimental 1206** 等替代方案，它们被认为是更好或更具成本效益的选择。尽管存在一些局限性，但与 **Claude** 相比，**ChatGPT** 仍被认为具有更慷慨的使用限制。
  - **对 OpenAI 商业模式的批评**：围绕 **OpenAI** 的商业行为展开了批判性讨论，将其比作“缩减式通胀 (Shrinkflation)”，用户觉得资源正在降级以推动更高层级的计划。这种情绪反映了对早期采用者和重度用户所受待遇的不满，一些人建议改用 **Anthropic** 或其他 AI 选项。

- **[你让 o1-pro 思考最长的时间是多少？](https://i.redd.it/szfy5rpzdu5e1.jpeg)** ([Score: 705, Comments: 223](https://reddit.com/r/ChatGPT/comments/1hacdc8/whats_the_longest_youve_got_o1pro_to_think_for/)): 该帖子讨论了使用 **ChatGPT 的 o1-pro 模式** 生成一个复杂的 prompt，涉及关于宇航员火星之旅的五段式故事，并对用词和结构提出了复杂的约束。AI 花费了 **11 分 11 秒** 来处理这个请求，凸显了在处理复杂任务时响应时间的潜在局限性。
  - 几位评论者批评了此类 prompt 对 **资源和能源的浪费**，将其比作不必要地开灯或改装卡车以排放更多污染等轻率行为。**CleverJoystickQueen** 指出在 **2 分 9 秒** 内就实现了类似的结果，暗示了对 AI 能力的低效使用。
  - **Crypt0genik** 等人对 **资源分配** 和潜在的滥用表示担忧，强调此类任务并不能有效地测试 AI 的能力。**ProposalOrganic1043** 分享了对更多 **有意义任务** 的渴望，这些任务可以从 AI 的推理能力中受益，这与讨论中 prompt 的琐碎约束形成鲜明对比。
  - 关于 **能源消耗** 及其影响的讨论包括要求提供 **2 kWh 消耗** 数据的来源，**ExclusiveAnd** 提供了估算 ChatGPT 能源使用的文章链接。像 **marcusss12345** 这样的评论者强调了尽量减少能源浪费对于气候减缓和适应的重要性。


**Theme 4. Criticism of "AI Gotcha" Tests: A Reflective Discourse**

- **[RealVisXL 奇怪的 "bug"](https://i.redd.it/lsgzdwbufs5e1.jpeg)** ([Score: 173, Comments: 75](https://reddit.com/r/StableDiffusion/comments/1ha5oyv/realvisxl_strange_bug/)): 该帖子讨论了 **RealVisXL 4.0** 中的一个 **奇怪异常**，即生成任何图像的第一步都会产生一张扭曲的图像，类似于骷髅或类人形象。图像具有夸张的面部特征和瓷砖纹理背景，底部的技术描述将其称为 "3 高 x 3 宽板岩瓷砖的无缝平面纹理，灰度"。
  - 几位评论者认为该异常与 **RealVisXL 4.0** 处理 **negative prompt** 的方式有关，一些人指出在使用某些 negative prompt 或特定设置（如高 **CFG scale**）时也有类似经历。**_roblaughter_** 解释说，sampler 计算 negative prompt 来引导生成，这可能会导致此类初始输出。
  - **Eltrion** 提到了 "Negative Man"，这是一个在 **CFG 值** 非常低时出现的已知伪影，看起来像一个秃头的、类似地精的生物，这与之前的 [Reddit 讨论](https://www.reddit.com/r/StableDiffusion/comments/1b0tze1/why_is_there_the_imprint_of_a_person_visible_at/) 有关。这与其他用户分享的经验一致，表明在某些设置下存在反复出现的模式。
  - **Remarkphoto** 和 **Disty0** 强调，该异常可能是由于 **内置的 negative prompt** 造成的。这得到了其他人的证实，他们在仅使用 "bad photo" 或 "ugly" 等极简 negative prompt 时也看到了类似的 "恐怖" 面孔，表明这可能是某些 AI 模型的常见问题。


- **[ChatGPT 在计算数学时惊慌失措。](https://i.redd.it/usyif7fyhs5e1.png)** ([Score: 171, Comments: 27](https://reddit.com/r/ChatGPT/comments/1ha5uk5/chatgpt_panicked_whilst_computing_some_maths/)): **ChatGPT** 在一次关于随机变量期望（特别是涉及求和性质）的数学讨论中遇到了计算错误。这次互动展示了一个 AI 与人类协作解决问题的场景，评论中讨论了计算中的错误和调整。
  - 用户幽默地注意到 **ChatGPT 的惊慌** 以及在解决基础概率问题遇到计算错误时表现出的类人反应，其中一条评论强调了它是如何 "无限生成并自我纠正" 的。这反映了 AI 偶尔在处理初等数学问题时的挣扎。
  - 尽管预期 **ChatGPT 4o** 能可靠地解决此类问题，但在随后的查询中，它仅犯了一个错误就解决了问题，表明其性能可能存在不一致性。
  - 短语 *"human please wrap"* 被讨论为一种速记表达，用户对 AI 对其自身计算错误做出的非正式且看似类人的反应表示惊讶。


---

# AI Discord Recap

> 由 O1-preview 总结的总结之总结

**主题 1. Llama 3.3 模型：发布、微调与挑战**

- [**Llama 3.3 权重在 Hugging Face 发布！**](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct): 社区反响热烈，**Llama 3.3 70B Instruct** 权重现已可用，包括 **GGUF** 和 **4-bit** 格式，使高性能模型更易于被大众获取。
- [**低预算微调 Llama 3.3**](https://unsloth.ai/blog/llama3-1): 用户正在应对在有限的 GPU 资源上微调 **Llama 3.3** 的挑战，分享了诸如参数调整等策略，以在硬件限制下缩短训练时间并优化性能。
- [**内存烦恼：缩减 Llama 3.3 的占用空间**](https://gist.github.com/pbontrager/b7b8dcfd320fa8a4ebf828ed9d33404b): 开发者正致力于将 **Llama 3.3 70B** 的内存占用降低到 **49GB** 以下，尝试使用 **PagedAdamW** 和 **4-bit optimizers** 等优化器，但结果褒贬不一。

---

**主题 2. Gemini 与 Sora：AI 巅峰对决**

- [**Gemini 1206 刷新基准测试记录！**](https://aider.chat/docs/leaderboards/): 新的 **Gemini exp 1206** 模型引起了轰动，其表现超越了前代产品，并在代码编辑基准测试中创下纪录，用户注意到其在编程辅助方面有显著改进。
- [**Sora v2 发布：AI 视频生成的未来已来！**](https://sora.com): **Sora v2** 推出了先进的视频生成功能，如 **text-to-video** 和 **minute-long outputs**，令预测其将彻底改变 AI 交互的用户感到兴奋。
- [**OpenAI 的 Sora 起飞，全场沸腾！**](https://x.com/sama/status/1866187525821538436): **Sam Altman** 揭晓了 **Sora**，它可以将文本和图像转化为沉浸式视频。早期采用者赞不绝口，AI 社区充满了兴奋。

---

**主题 3. AI 模型性能与对比**

- [**O1 Pro：卓越的编程能力是否物有所值？**](https://aider.chat/docs/leaderboards/): 用户在讨论 **O1 Pro** 的高昂成本与其顶尖编程能力的对比，赞扬其推理能力，但也质疑 **$200** 的费用是否合理。
- [**Cursor vs. Windsurf：IDE 大决战**](https://www.youtube.com/watch?v=SrPmkpgRbkE): 开发者对比了 **Cursor IDE** 和 **Windsurf**，权衡了项目结构创建和自定义等功能，对于哪种工具更能提高生产力意见不一。
- [**Llama vs. Hermes：无审查 AI 的对决**](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct): 讨论强调了 **Llama 3.3** 和 **Hermes** 模型，因为它们具有智能的功能且缺乏审查，使其成为寻求不受限 AI 交互的用户的首选。

---

**主题 4. AI 效率工具与技术**

- [**APOLLO：我们需要的内存节省英雄！**](https://arxiv.org/abs/2412.05270): 介绍 **APOLLO**，一种有望在 LLM 训练期间减少内存占用的新优化器，解决了 **AdamW** 的高需求问题，使训练对所有人来说都更易实现。
- [**Unsloth 拥抱 OpenAI Triton：速度与效率的结合**](https://github.com/rkinas/triton-resources): **Unsloth** 利用 **OpenAI Triton** 库进行快速、内存高效的训练，分享的资源让社区对潜在的性能提升感到兴奋。
- [**Tinygrad JIT 技巧：当速度破坏了你的代码**](https://github.com/kroggen/tokenformer-minimal): 开发者正在努力解决 **TinyJit** 破坏模型功能的问题，了解到保持一致的输入形状以及将数据加载与 JIT 函数分离是顺利训练的关键。

---

**主题 5. 开发中的 AI：挑战与解决方案**

- [**Bolt 按钮忧郁：当“添加记录”拒绝添加时**](https://github.com/stackblitz/bolt.new/issues/2985): **Bolt** 用户报告 **add record button** 无响应，导致工作流中断，并呼吁改进 Prompt 约定以减少此类问题。
- [**NotebookLM 的 17 分钟奇迹：浓缩 107 页内容！**](https://youtu.be/aG0ixD3OY80): 用户分享了 **NotebookLM** 如何将冗长的文档压缩成简洁的播客，其中一位用户将 **107 页** 的法规转化为了 **17 分钟** 的音频摘要。
- [**自适应批处理探险：追求高效训练**](https://github.com/pytorch/torchtune/blob/06a837953a89cdb805c7538ff5e0cc86c7ab44d9/torchtune/modules/loss/ce_chunked_output_loss.py#L30): **Torchtune** 社区探索了更好的自适应批处理（Adaptive Batching）方法，并承认仅仅增加批次大小直到出现 **Out-Of-Memory** 并不是最明智的做法。

---

---

# PART 1: Discord 高层级总结

## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Cascade 定价变更发布**：Cascade 的定价模型已更新，新增了 **Pro tier**（15美元/月）和 **Pro Ultimate tier**（60美元/月），并引入了新的 **credit system** 来管理高级模型的使用，详情见其 [pricing page](https://codeium.com/pricing)。
   - 在变更前订阅的早期用户将保留 **Pro plan**（10美元/月），而支付了新费用（15美元）的用户将获得 5 美元退款，确保初始用户的定价连续性。
- **Windsurf 1.0.7 发布并带来增强**：最新的 **Windsurf 1.0.7** 已发布，包含了针对 **1.0.6** 版本的细微 Bug 修复以增强整体稳定性，详情见 [public changelog](https://codeium.com/changelog)。
   - 关键更新包括对使用透明度的调整和更新的定价信息，以提升用户体验。
- **AI 上下文理解问题报告**：用户遇到了如“**The code edit failed to apply**”和“**Cascade has encountered an internal error**”等错误，尤其是在使用 **Cascade Base model** 时，这表明存在 **credit usage** 和 **context retention** 方面的问题。
   - 据报道，这些问题阻碍了 AI 模型的有效性，社区指出需要更好的上下文管理。
- **强调模型切换策略**：社区建议在 **Cursor** 和 **Windsurf** 之间切换以优化工作流并解决问题，提倡将 **Cascade** 作为默认模型，同时将外部模型作为补充工具。
   - 用户强调了理解不同模型间 **context maintenance** 的重要性，以提高工作流效率。
- **Cascade 的增强建议**：用户提议对 **Cascade Base model** 进行升级，包括增加 **web searching** 和 **custom instructions** 以提升性能和易用性。
   - 这些增强功能预计将显著改善 **Windsurf** 的功能，满足当前用户对更强大特性的需求。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 的性能挑战**：用户报告 **Cursor IDE** 正在经历性能下降，特别是在使用 **Claude models** 时，影响了文件修改和上下文理解。
   - 一些人将下降归因于高模型需求，而另一些人则主张保持*专注且清晰的 prompting 策略*以获得最佳结果。
- **OpenAI O1 Pro API 成本分析**：社区讨论了使用 **OpenAI's O1 Pro API** 的性价比，表示不愿为 **Cursor IDE** 的多个订阅支付额外费用。
   - 参与者建议探索 **group buys** 以降低成本，并根据个人使用案例评估收益是否匹配支出。
- **Cursor 与 Windsurf 功能对比**：成员们分享了使用 **Cursor IDE** 和 **Windsurf** 的不同体验，强调了 Windsurf 在创建项目结构方面的可靠性。
   - **Cursor IDE** 通过 `.cursorrules` 和 AI 工具等功能提供自定义选项，尽管一些用户更喜欢 **Windsurf** 的简洁和直接输出。
- **Cursor IDE 功能增强请求**：用户请求改进 **Cursor IDE** 中的 **documentation handling**、**Git integration** 以及管理**更大上下文文件**的能力，以增强易用性。
   - 一些人建议，在更新中进行更好的测试和更平滑的过渡将显著提高用户对 **Cursor IDE** 的满意度。
- **AI 模型代码生成的有效性**：参与者讨论了来自 **Claude** 和 **O1** 等 AI 模型的不同结果，从有效的代码生成到令人沮丧的幻觉和无关输出。
   - 重点在于在 prompt 中构建**精确的问题定义**，以优化这些 AI 模型提供的协助效果。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **在有限资源下微调 Llama 3.3**：用户讨论了在低端 GPU 上微调 **Llama 3.3** 模型的挑战，强调了成本和显存需求。尽管硬件受限，一位用户通过参数调优缩短了训练时间。
   - 探索了优化资源利用和利用高效参数配置的策略，以增强在受限硬件设置下的性能。
- **AWQ 和 LoRA 训练限制**：**AWQ** 和 **GPTQ** 主要用于推理，不支持直接微调。成员建议使用 LoRA 适配器以实现在 int4 或 fp16 模型上的训练。
   - 虽然 **AWQ** 模型具有某些优势，但预计大多数训练活动将继续在 int4 或 fp16 基础模型上进行，以保持兼容性和性能。
- **令人兴奋的开源项目：Harmony**：**Harmony** 项目利用 [Natural Language Processing](https://harmonydata.ac.uk/) 协助研究人员协调问卷项目和元数据。该项目总部位于伦敦大学学院（UCL），涉及多所大学，并举办了一场改进其 LLM 匹配算法的竞赛，奖项信息见[此处](https://harmonydata.ac.uk/doxa/)。
   - 鼓励参与者加入 Harmony Discord 服务器进行讨论和获取更新，特别是在 🏅「matching-challenge」频道。
- **Unsloth 采用 OpenAI Triton 进行高效训练**：Unsloth 利用 [OpenAI Triton library](https://github.com/rkinas/triton-resources) 实现快速且显存高效的训练，并分享了一份精选的有价值资源列表。社区表现出极大的热情，成员们认为这一采用“非常酷”！
   - 使用 Triton 旨在提高训练效率和可扩展性，符合 Unsloth 优化 LLM 开发的目标。
- **开发显存高效的 LLM 优化器**：引入了一种名为 *APOLLO* 的新方法，通过改进学习率自适应规则来优化 **AdamW** 优化器的显存使用，从而在无需昂贵的 SVD 操作的情况下实现更好的可扩展性。
   - 该方法旨在减少训练大语言模型期间的显存占用，从而实现更高效的优化过程。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 在性能上超越 Sonnet 3.5**：用户评估了新的 [`gemini-exp-1206`](https://x.com/Presidentlin/status/1865370199160979963) 模型，发现它比 **Sonnet 3.5 更强**，尽管注意到它在正确格式方面的排行榜排名较低。
   - 该模型在 diff 任务中达到了 **69%** 的准确率，在 whole 任务中达到了 **80.5%**，引发了关于优化其在编程中使用的讨论。
- **O1 Pro 尽管价格昂贵但在编程方面表现出色**：**O1 Pro** 在 Bug 修复和代码架构方面的推理能力优于 **Sonnet**，获得了一致好评，一些用户对其处理复杂代码问题的能力给予了高度评价。
   - 用户对 `$200` 的价格展开了辩论，考虑只有在性能提升显著时才切换到 O1 Pro。
- **Aider 的功能模式受到关注**：讨论集中在 Aider 的 **Architect** 和 **Editor** 模式，辩论 Architect 模式应该生成代码还是仅仅进行规划。
   - 一位成员建议对于较简单的任务仅依赖 **QWQ** 和 **Qwen** 模型。
- **Google 推出用于量子计算的 Willow**：Google 发布了 **Willow 量子计算芯片**，旨在比传统超级计算机显著缩短复杂任务的计算时间。
   - 用户对 Willow 在专业领域之外的实际应用表示关注，并希望为量子芯片提供增强的编程 [SDKs](https://x.com/sundarpichai/status/1866167562373124420)。
- **Aider 用户面临 API rate limit 挑战**：几位成员在使用 Aider 配合 OpenAI 的 API 时遇到了 **rate limit** 错误，引发了关于跨会话 token limit 应用的问题。
   - 对于高 token 使用量以及 Aider 的方法对 API 限制的影响（特别是在暂停使用后）存在困惑。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 编译器提升性能**：**Mojo 编译器**现在利用针对 SIMD 大小的动态优化来解决硬件兼容性问题，并提出了类似于 C/C++ 编译器中的多版本化（multiversioning）特性提案。[Feature Request #3651](https://github.com/modularml/mojo/issues/3651) 讨论了添加函数多版本化以符合 Mojo 的路线图。
   - 成员们强调了潜在的性能提升，但也对不同用户系统间的可移植性表示担忧。建议包括利用现有的编译器策略来平衡优化与兼容性。
- **执行 AI 生成内容政策**：版主在论坛上实施了严格的 **AI 生成内容政策**，任何检测到的 AI 内容将被删除，并警告作者以维护真实的讨论。此举旨在保持社区内真正的互动。
   - 该政策确保了像周边礼品挑战（swag challenges）之类的促销活动不受 AI 贡献的影响，从而营造一个真实用户参与和可靠信息交换的环境。
- **Modular 论坛正式上线**：**Modular 论坛**现已在 [forum.modular.com](http://forum.modular.com/) 开放，为用户提供详细的技术讨论、官方回复和支持平台。此次发布恰逢 **Swag Challenge** 启动，以提高社区参与度。
   - 鼓励用户通过[此讨论](https://forum.modular.com/t/simplifying-gpu-programming-with-parametric-tile-level-tensors-in-mojo-llvm-developers-meeting-2024/38)与 Ahmed 就 **使用 Mojo 进行 GPU 编程** 进行交流，并在 [Forum Feedback 类别](https://forum.modular.com/c/feedback/2)中提供反馈，以帮助完善平台。
- **Mojo 类型系统的进展**：关于 Mojo 中**线性及显式销毁类型（linear and explicitly destroyed types）**的提案旨在通过引入新的 `destroy` 关键字来增强 GUI 开发中的错误预防。该提案详见 [Issue #3848](https://github.com/modularml/mojo/issues/3848)，并引发了关于其实现的讨论。
   - 出现了关于重用 Python 的 `del` 而不是新关键字的疑问，社区成员就线性结构体（linear struct）上下文中的范围和实际用法进行了辩论，以提高代码可靠性。
- **内存管理策略讨论**：针对 Mojo **内存管理**的持续研究强调了高效分配器系统（allocator systems）对于增强其低级编程能力的重要性。讨论将 Mojo 的方法与 Rust 和 C++ 进行了比较，指出了优化空间。
   - 参与者指出，有效的内存管理在游戏开发和系统编程中起着至关重要的作用，认为 Mojo 在该领域的发展对于其在性能敏感型应用中的采用至关重要。

---

## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **Bolt 功能故障凸显**：成员报告称 **Bolt** 中的**添加记录按钮**无响应，干扰了用户工作流。
   - 初次尝试通常会导致前端创建，需要更精确的后续提示词（prompts）来激活所需功能。
- **推进 Bolt 的提示词工具**：一位用户强调在 **Bolt** 中需要有效的提示词规范或工具，以减少问题并提高输出质量。
   - 另一位成员正在积极开发一款工具，旨在帮助用户为 **Bolt** 编写更有效的提示词。
- **Claude 的变量敏感性问题**：有用户对 **Claude** 更改变量名表示担忧，因为它忽略了提示词中的**大小写敏感性**设置。
   - 用户表示，即使正确提供了 JSON 格式，变量大小写未被保留也令人沮丧。
- **即将到来的 Supabase 集成和 Token 政策**：**Bolt** 将集成 **Supabase**，通过无缝的数据库和身份验证功能增强应用开发，通过回复[团队推文](https://x.com/stackblitz/status/1865904408254620148)可获得早期访问权限。
   - 在 **Token 管理**方面，明确了充值 Token 可以结转，而订阅 Token 每月重置，解决了之前订阅者的困扰。
- **Bolters.io 扩展社区资源**：**Bolters.io** 平台已更新社区驱动的资源，包括应用推荐、故障排除指南和教育视频链接。
   - 鼓励用户通过分享自己的挑战和帮助他人来参与其中，共同构建协作知识库。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Countless.dev 简化 AI 模型对比**：新推出的 [Countless.dev](https://www.producthunt.com/posts/countless-dev) 为用户提供了一个**免费且开源**的平台，用于根据**价格、Token 限制和功能**来**对比 AI 模型**，包括 LLM 和视觉模型。
   - 该工具目前在 **Product Hunt** 上展示，创作者正在寻求支持以获得**第一名**的排名，这凸显了该工具在 AI 社区中日益增长的人气。
- **Claude 3.5 Sonnet 增强功能**：更新后的 **Claude 3.5 Sonnet** 模型（识别码为 **claude-3-5-sonnet-20241022**）展示了优于 **Opus** 的**卓越性能**，同时保持了**极具竞争力的定价**。
   - 新功能包括**增强的视觉处理**和**高级工具调用 (tool usage)**，特别提升了在**编程 (coding)** 和**数据科学**任务中的表现。
- **Poe 集成提升 OpenRouter 功能**：**OpenRouter** 与 **Poe** 的集成引入了对 **OpenAI Whisper** 和 **Text-to-Speech** 等高级功能的访问，扩展了平台对用户的实用性。
   - 此次集成是**提升用户体验**和在 OpenRouter 生态系统中**扩展 AI 模型能力**的持续努力的一部分。
- **Llama 3.3 在无审查性能方面表现出色**：讨论强调了 **Llama 3.3** 和 **Hermes** 模型的有效性，指出了它们的**智能功能**和**无审查 (lack of censorship)** 特性，使其成为用户的首选。
   - **Llama** 因其强大的能力而保持流行，同时提到的**旧版 Gemini** 也为其在社区中的声誉做出了贡献。
- **Mistral 模型在发布后被撤回**：最近的更新表明，几个 **Mistral** 模型在发布后不久就被**撤回**，引发了社区内的担忧。
   - 推测围绕着新模型（如 **Codestral** 和 **mistral-ocr**）的潜在发布展开，特别是在它们通过 **API 通知**泄露之后。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 利用 Vulkan 提升 GPU 效率**：使用 **RX 6600 GPU** 的用户已经认识到 **LM Studio** 利用 [Vulkan](https://lmstudio.ai/beta-releases) 进行 GPU 卸载 (offloading)，从而无需安装 ROCm 即可执行模型。
   - **AMD 用户**非常欢迎这种集成，因为它简化了硬件利用，扩展了 **LM Studio** 在不同 GPU 架构上的可访问性。
- **Aider 集成面临配置障碍**：正如 [Aider 文档](https://aider.chat/docs/llms/lm-studio.html) 中所讨论的，由于 API Key 设置和环境变量配置问题，与 **Aider** 的集成一直具有挑战性。
   - 建议用户生成随机 API Key 并严格遵守设置说明，以减轻这些集成问题。
- **有限的模型支持引发不满**：**LM Studio** 用户对缺乏对 **Qwen2 VL 7B Instruct** 等模型的支持表示不满，这限制了新视觉模型的部署。
   - 建议使用替代方案，例如通过 **Pinokio** 使用 **Florence-2**，以探索更多的视觉模型选项。
- **探索 LM Studio 的替代前端**：推荐了几个**前端客户端**，如 [AnythingLLM](https://anythingllm.com) 和 [Open WebUI](https://github.com/open-webui/open-webui)，作为连接到 **LLM 服务器**的替代方案。
   - 鼓励用户尝试这些选项，以访问针对特定工程需求量身定制的多样化功能。
- **优化 GPU 配置以提升 AI 性能**：讨论强调了将 **GPU 规格**与模型要求相匹配的重要性，重点推荐使用价格极具竞争力的 **NVIDIA A100** 等 GPU。
   - 成员们指出，充足的**内存带宽**和 **GPU 显存 (VRAM)** 对于增强 **AI 模型性能**至关重要，特别是对于显存需求较高的模型。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Gemini exp 1206 性能增强**：**Gemini exp 1206** 的表现一直优于其前代产品，在 [Aider 的代码编辑基准测试](https://aider.chat/docs/leaderboards/)中取得了创纪录的成绩。用户报告称，在编码辅助和基准测试分数方面有显著提升。
   - 尽管取得了成功，但一些用户在 Cursor 等环境中使用该模型的协作功能时，遇到了设置问题和不确定性。
- **xAI 发布 Aurora 图像模型**：xAI 最新发布的 **Aurora 图像模型** 正在受到关注，早期采用者称赞其细腻的图像生成能力。然而，一些用户指出在有效渲染卡通效果方面存在挑战。
   - 有人询问 Aurora 是否与 Flux 的开发者 Black Forest Labs 展开合作，这表明在图像生成技术方面可能存在联合开发。
- **Sora v2 视频生成特性**：**Sora v2** 将通过文本转视频（text-to-video）和更细腻的输出等特性来增强视频生成。多位 AI 领域知名人士表达了兴奋之情，预计这将对用户参与度产生重大影响。
   - 在发布期间，多个演示展示了 Sora v2 的潜力，许多人预计其使用量的增加将与 Pro 和 Plus 订阅层级挂钩。
- **WaveForms AI 的语音图灵测试计划**：**WaveForms AI** 宣布其目标是开发能够通过 [语音图灵测试 (Speech Turing Test)](https://x.com/alex_conneau/status/1866127388373098607) 的 AI，旨在提升音频应用中的类人交互。
   - 这一举措与行业向 AI 系统引入高级情感分析的趋势相一致，反映了增强 AI 共情能力日益增长的趋势。
- **NeurIPS 2024 准备与社交**：随着 **NeurIPS 2024** 的临近，参与者正通过 [Latent Space Paper Club](https://lu.ma/25mwbwcm) 等活动积极准备。社区正专注于论文讨论和创意碰撞（idea jams），以在会议前最大限度地提高效率。
   - 社交策略强调了 **走廊交流 (hallway track)** 对于建立有价值联系的重要性，参会者更倾向于交换 Twitter 账号和使用会议 App，而非传统的名片。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Llama 3.3 权重在 Hugging Face 发布**：一名成员在 [Hugging Face](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) 上上传了 **Llama 3.3 70B Instruct 的 16bit 权重**，提供了多种格式的访问，包括 [Llama 3.3 所有版本](https://huggingface.co/collections/unsloth/llama-33-all-versions-67535d7d994794b9d7cf5e9f) 的集合。
   - 该版本包括 **GGUF** 和 **4-bit** 格式，为那些等待审批的用户提供了更广泛的访问便利。
- **APOLLO 优化 LLM 内存**：一篇论文介绍了 **APOLLO**，这是一种内存高效的优化器，旨在解决大型语言模型训练过程中 **AdamW** 高内存消耗的问题。
   - APOLLO 旨在减少内存使用而不产生显著的性能损失，因为 **AdamW** 沉重的内存负担需要昂贵的计算开销。
- **梯度路由 (Gradient Routing) 增强神经清晰度**：**梯度路由** 方法允许根据数据类型进行选择性参数更新，促进神经网络的专业化，并解决与 AI 黑盒性质相关的安全问题。
   - *梯度路由* 可以使模型能够区分 **可信** 和 **不可信** 来源，从而改善元数据对模型行为的影响方式。
- **EleutherAI Eval Harness 增强**：[Pull Request #1140](https://github.com/ml-explore/mlx-examples/pull/1140) 为 EleutherAI 的 eval harness 引入了 `mlx_lm.evaluate` CLI，支持任何兼容 mlx-lm 的模型（如 `Qwen2.5-7B-Instruct`）进行评估。
   - 此外，为 ARC-Challenge 提供的配置旨在简化性能比较，解决数据集异常并确保评估的准确性。
- **VLM 通过因果损失 (Causal Loss) 提升训练**：在关于 **Qwen2-VL** 等 **VLM** 的讨论中，成员们探讨了在视觉 token 上应用 **因果损失** 和 **MSE**，以增强多模态特征的学习。
   - 讨论中引用了 **Apple AIM**，以获取关于 **MSE** 在视觉 token 处理中应用的见解。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **播客之巅：NotebookLM 将 107 页内容压缩至 17 分钟**：成员们分享了使用 **NotebookLM** 的经验，重点介绍了将 **107 页** 的 Formula 1 规则压缩成 **17 分钟** 播客的案例。这展示了 NotebookLM 高效处理和总结长篇文档的能力。
   - 此外，将 [YouTube 视频](https://youtu.be/9CN1Ymyrhyo?si=n2PpH1J4PQgrvuvH) 与暂存盘（scratchpad）结合，生成的播客时长甚至超过了原始视频，展示了内容创作的灵活性。
- **通过 Zapier 将 Claude 和 ChatGPT 与 NotebookLM 连接**：讨论集中在将 **Claude** 和 **ChatGPT** 与 **NotebookLM** 集成，并建议将 **Zapier** 作为可行的解决方案。这种集成旨在通过利用先进的语言模型来增强 NotebookLM 的功能。
   - 成员们反思了使用 NotebookLM 通过输入歌词和其他资源来创建歌曲背景信息的做法，展示了语言模型互操作性的创新用例。
- **NotebookLM 语言切换限制**：用户报告了在 **NotebookLM** 中切换语言的挑战，通常需要 **登出并重新登录** 才能更改设置。这种限制阻碍了为多样化用户群体提供无缝的多语言支持。
   - *NotebookLM 不支持即时语言切换*，导致寻求更动态、灵活语言体验的用户感到沮丧。
- **播客对决：NotebookLM vs ElevenLabs**：比较了 **NotebookLM** 的播客功能与 **ElevenLabs** 的功能，突显了播客工具领域的竞争态势。NotebookLM 被指出缺乏清晰的 API 和系统化的 Prompting 能力。
   - 这一差距表明 **NotebookLM** 在提升播客易用性方面有潜在改进空间，使其在面对 **ElevenLabs** 等成熟选手时更具竞争力。
- **NotebookLM 中的文档上传限制**：用户发现 **NotebookLM** 中每个笔记本有 **100 个文档** 的上传限制，但同时也注意到笔记本的数量没有上限。这一约束影响了用户管理和组织文档工作流的方式。
   - 关于上传限制是否已从之前的 **50 个文档** 增加，存在一些困惑，这表明 NotebookLM 团队需要更清晰的沟通。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Unsloth 提升微调效率**：一位成员介绍了 **Unsloth 微调框架**，强调了其在训练过程中集成自定义评分功能的能力，从而实现更精确的 **评估循环（evaluation loops）**。
   - 这一进步为量身定制的微调任务开启了创新的可能性，通过改进的 **反馈机制** 增强模型性能。
- **简化 aya-expense 模型量化**：一位用户请求协助将 **aya-expense 模型** 量化为 AWW 或 FP8 格式，以便在有限的 GPU 资源上部署，并建议使用训练数据进行校准。
   - 另一位成员回应称 **8b 模型** 易于运行，其大小可缩减至 **3.4GB**，从而提高了可访问性。详细信息可见 [aya](https://ollama.com/library/aya)。
- **基于向量检索的高级技术**：一位新成员讨论了他们对 **基于向量的检索方法** 和 **稠密通道检索（dense passage retrieval）** 的研究，并提议进行一项对比研究以评估其有效性。
   - 社区成员支持这一倡议，建议通过引入 **多步工具调用（multi-step tool use）** 来进一步优化 [检索流程](https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Multi_Step_Tool_Use.ipynb)。
- **多步工具调用增强 RAG**：一位社区成员详细阐述了 RAG 中的 **多步工具调用**，将其等同于 Agent 多次调用工具以细化查询并分析结果。
   - 这种方法旨在通过自动化查询细化和结果分析来增强研究能力，从而实现更准确、高效的信息检索。
- **探索情感 AI 语音生成**：关于 **语音生成中的情感表达** 的讨论集中在开发用于定制人声风格的 API，并对 **GPT4o-voice 风格** 表现出兴趣。
   - 一位成员分享了他们运行专注于 **语音情感化** 的个人 API 的经验，强调了更具表现力和适应性的语音模型的潜力。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **混合专家模型提升 LLM 效率**：成员们讨论了 **Mixtures of Experts (MoEs)** 在不牺牲性能的前提下增强 **LLM** 效率的潜力，并引用了 [Approximating Two-Layer Feedforward Networks for Efficient Transformers](https://arxiv.org/abs/2310.10837) 论文作为关键参考。
   - 对话强调了近期 **MoE** 的发展如何降低计算和内存需求，使 **MoEs** 成为大规模语言处理中稠密模型（dense models）的有力竞争替代方案。
- **高效 LLM 训练技术**：讨论集中在通过利用单 **GPU** 设置等策略优化 **LLM** 训练，参考了 [Cramming: Training a Language Model on a Single GPU in One Day](https://arxiv.org/abs/2212.14034) 论文。
   - 参与者指出，极简训练方法可以实现与大型模型相当的性能，同时显著降低计算成本。
- **动量提升上下文学习**：一位成员提出，在训练中实施 **momentum** 可以提高 **in-context learning** (ICL) 的效率，并将其比作“强制跳跃连接”（forced skip connections）。
   - 他们询问 ICL 是否受梯度下降动力学的影响，建议 [Implementing momentum along the residual stream](https://link.to/implementation) 可能是一种可行的优化方法。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Ollama 3B 模型本地性能不一致**：用户报告称，**Ollama** 默认的 **3B model** 在本地运行与终端执行时的性能不一致，并对其 **ChatAdapter** 表示困惑。
   - 用户对量化模型需要更简单的适配器以及改进模型输出的承诺提出了关注。
- **将人类反馈整合进 DSPy**：一位成员询问如何将类似 **Agrilla** 的人类反馈作为 **DSPy** 的指标，参考了之前的讨论和 [pull request #1647](https://github.com/stanfordnlp/dspy/pull/1647)。
   - 相关对话包括探索在 **teleprompting** 中引入人类反馈，并分享了额外的 [GitHub](https://github.com/stanfordnlp/dspy/pull/1647) 链接。
- **DSPy 程序的多种部署策略**：成员们分享了 **DSPy programs** 的多种部署方法，如使用 **FastAPI** 和 **MLFlow**，并指出生产环境可能需要独立的容器。
   - 讨论了将 **DSPy** 集成到 **Django projects** 或部署在 **Modal** 上的替代方案，强调了部署选择的灵活性。
- **增强 DSPy 中的上下文感知分块**：探讨了 **DSPy** 作为上下文感知分块器（context-aware chunker）的潜力，并就如何有效优化长文档处理提出了建议。
   - 对话包括讨论小型和大型语言模型在优化此过程中的局限性。
- **在 DSPy 中实现 Anthropic MCP**：一位用户请求将 **Anthropic** 的 **Model Context Protocol (MCP)** 与 **DSPy** 集成的方案，并得到了相关建议和 [集成资源](https://www.darinkishore.com/posts/mcp)。
   - 分享的博客文章概述了围绕 MCP 构建工具的方法，重点关注其在 **AI** 工具开发中的应用。



---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaParse 支持多模态解析**：在一个演示视频中，**LlamaParse** 展示了如何启用与 **GPT-4**、**Claude 3.5** 和 **LLaVA 1.5** 等模型兼容的**高级多模态解析**。[视频演示](https://twitter.com/llama_index/status/1865125665491886171)展示了有效的截图转换功能。
   - **LlamaParse** 的多模态能力促进了与顶级 AI 模型的无缝集成，扩展了其适用性。
- **Claude Desktop 集成复杂 PDF**：**Marcus Schiesser** 的一个新项目通过 **Model Context Protocol (MCP)** 将 **LlamaCloud** 的文档解析与 **Claude** 集成，实现了与复杂 PDF 的对话功能。[项目描述](https://twitter.com/llama_index/status/1865460899059998999)提供了详细见解。
   - 这一集成允许用户通过 **Claude** 与复杂的 PDF 文档进行交互，增强了文档处理工作流。
- **Agentless 简化软件问题修复**：今天，**LlamaIndex** 介绍了 **Agentless**，它展示了一个简单的三步流程来自动解决软件问题：**定位 (localization)**、**修复 (repair)** 和 **打补丁 (patch)**。[公告](https://twitter.com/llama_index/status/1865822785119174857)概述了该方法。
   - **Agentless** 提供了一个比传统解决方案更简单的替代方案，简化了问题解决流程。
- **LlamaParse 推出成本优化的 Auto Mode**：**LlamaParse** 中全新的 **Auto Mode** 通过以标准模式解析文档，并根据用户定义的触发器选择性地切换到 **Premium mode** 来优化成本。[功能详情](https://twitter.com/llama_index/status/1866214925418500119)解释了其优势。
   - **LlamaParse Auto Mode** 有效管理解析费用，允许自定义模式转换。
- **自动化聊天应用的摄取流水线 (Ingestion Pipelines)**：一位成员讨论了为一个私有聊天 RAG 应用每小时从 **Google Drive** 和 **Airtable** 等数据源自动运行摄取流水线。他们考虑使用 **任务调度器 (job scheduler)** 或**云托管解决方案**。
   - 增量更新带来的挑战促使开发者探索自动化流水线，以增强聊天应用的数据集成。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **探索自适应批处理 (Adaptive Batching) 方案**：成员们讨论了改进**自适应批处理**方法的需求，提议进行研究并开发一个简单的 [RFC](https://github.com/pytorch/torchtune/blob/06a837953a89cdb805c7538ff5e0cc86c7ab44d9/torchtune/modules/loss/ce_chunked_output_loss.py#L30) 来阐述概念。
   - 一位成员致力于测量效率，并确认“增加直到 OOM”的想法并非最优。
- **优化 Llama 3.3 显存占用**：一位用户寻求将 **Llama 3.3 70B 配置**的显存占用降低到 **49GB** 以下，并探索了优化方案和替代方案。
   - 建议包括使用 **PagedAdamW** 和 **4-bit 优化器**，尽管不同实现的测试结果各异。
- **识别出 Flex Attention Kernel 的 Bug**：据报告，**Flex Attention Kernel** 中存在一个可能导致共享内存问题的潜在 Bug，特别影响某些配置和 GPU 型号。
   - 建议包括针对 **A100/H100** 优化内核选项，用户应用的修复方案取得了不同程度的成功。
- **int8 混合精度训练的挑战**：尝试实现 **int8 混合精度训练**时，在使用特定优化器的情况下出现了**发散 (divergence)** 问题。
   - 建议包括增加 **Batch Size** 和**序列长度**以缓解发散。
- **AdamW 优化器解决训练发散问题**：采用 **AdamW** 优化器并移除 **optimizer-in-backward** 成功解决了训练过程中的**损失发散 (loss divergence)**。
   - 一位成员还报告了在增加 **Batch Size** 后获得了性能提升。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **代码中的 Inf/Nan 处理引发疑问**：一名成员对在面向执行的代码中支持 **Inf 和 NaN** 值表示怀疑，理由是 **exploding gradients** 通常会使训练运行失效。
   - 虽然有些人认为这种做法可能会让人产生疏离感，但目前仍在持续思考遵循数值计算 **IEEE standards** 的益处。
- **TinyJit 导致模型功能中断**：用户报告称，应用 **TinyJit** 装饰器会破坏其模型的功能，因为 **TinyJit** 捕获的 GPU kernels 需要进行调整，例如对某些操作使用 `Variable`。
   - 社区成员阐明了为 JIT 函数保持一致输入形状的必要性，建议训练步骤函数应该被 jitted，而数据加载应保持在 JIT 函数之外。
- **TinyJit 训练需要输入形状的一致性**：讨论强调，**JIT 函数**在每次调用时必须接收具有相同形状的输入，以避免训练期间出现错误。
   - 用户建议将 **data loader** 与 JIT 函数分开，以防止重复传递相同输入张量等问题。
- **会议议程定于圣地亚哥时间上午 9:30**：即将举行的 **Tinygrad meeting** 定于 **圣地亚哥时间上午 9:30**，议程包括删除功能以及关于 **cloud sprint** 的讨论。
   - **WebGPU** 以及针对 **ONNX** 和 **tensor cores** 的持续悬赏任务等话题计划进行深入讨论。
- **在 TinyJit 中实现学习率调度**：一位用户询问了在 **TinyJit** 中进行 **learning rate scheduling** 的情况，以及是否需要重新初始化优化器。
   - 他们在 [GitHub 上的 extras 目录](https://github.com/kroggen/tokenformer-minimal) 中发现了相关的实现，以辅助其训练过程。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **截止日期冲刺：作业与证书**：[Large Language Model Agents MOOC](https://llmagents-learning.org/f24) 的所有**作业**必须在 **12 月 12 日**之前提交，**证书申报表**需在 **12 月 17 日**之前提交。
   - **hackathon submissions** 的最终截止日期同为 **12 月 17 日**，证书发放将于 12 月底开始，一直持续到 1 月。
- **文章作业指南明确**：学生必须在指定的提交字段中包含其 **Written Article Assignment** 的全文，并单独链接到其社交媒体帖子，详情见[课程说明](https://llmagents-learning.org/f24)。
   - 澄清说明指定，使用发布在 Twitter 上的 Notion 链接是可以接受的，学生可以选择详细阐述其解决方案的方法，也可以保持在高层级概述。
- **GPT-4 的 Function Calling 详解**：**GPT-4** 通过其 API 采用了一种复杂的 **'function calling'** 机制，利用了强大的参数确定过程，正如 [Discord 讲座](https://discord.com/channels/1280234248112947210/1315259394157580340) 中讨论的那样。
   - 成员们正在寻找深入研究该功能背后工程原理的相关论文或博客文章，并假设大量的训练集示例促成了其有效性。
- **丰富的代码数据集助力训练**：**Code** 是一种高度可用的数据集，来自 **Stack Overflow** 和 **public GitHub repositories** 等来源在纠错方面表现出色，促进了有效的模型训练。
   - 代码的确定性特征使得在 post-training 阶段可以应用 **reinforcement learning**，从而增强模型性能。
- **黑客松冲刺：提交时间线**：[LLM Agents Hackathon](https://rdi.berkeley.edu/llm-agents-hackathon/) 的参与者必须在 **12 月 17 日**之前提交最终项目，与作业截止日期一致。
   - 澄清说明允许参与者选择不同的平台来展示他们的文章，前提是遵守提交要求。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenAI 发布 Sora**：在一次直播中，**OpenAI** 宣布推出 **Sora**，这是一个能将[文本和图像转换为沉浸式视频](https://sora.com)的工具，*Sama* 在直播前几分钟揭晓了它。
   - *Sama* 在 [Twitter](https://x.com/sama/status/1866179920260739502?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 上宣传了此次活动，为产品发布造势。
- **OpenInterpreter 应用访问请求**：成员们正积极请求 **OpenInterpreter 桌面应用**的早期访问权限，并强调了最近升级的硬件（如 **Mac mini**）以支持其使用。
   - 团队的回应非常积极，已向用户发送私信以确认访问权限。
- **解决模型兼容性问题**：讨论围绕特定模型与 **OpenInterpreter** 的兼容性展开，建议使用 `--no-tools-calling` 来确保运行成功。
   - 成员们分享了优化模型性能的策略，同时主张在执行工具前建立稳健的审批机制。
- **关于多智能体系统有效性的辩论**：一场关于 **multi-agent systems** 与经过优化的单智能体模型效用的辩论浮出水面，人们对前者的优势表示怀疑。
   - 参与者引用了以往单模型表现优于多智能体框架的案例，导致对未来发展方向产生分歧。
- **O1 在各种笔记本电脑上的表现**：用户询问了有效运行 **O1** 所需的最低笔记本配置，寻求支持它的最低硬件配置说明。
   - 还有关于 **O1** 在 **Windows** 和 **Windows 11** 笔记本电脑上表现的问题，用户旨在复现[演示视频](https://link.to/demo)中的结果。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **禁止机器人：应对垃圾广告**：成员们对来自机器人的重复**垃圾信息**表示**沮丧**，并指出这是它们唯一的历史消息记录。
   - 一位成员在注意到这种**行为模式**后，建议**封禁**这些账号。
- **LeoLM 在德语问答任务中表现出色**：一位成员比较了各种德语 LLM，发现 **LeoLM/leo-hessianai-7b** 尽管“仅经过预训练”，但在**问答任务**中产生了优异的结果。
   - 有人提问 **Llama 模型**潜在的**指令微调（instruction tuning）**是否影响了这些结果。
- **AI 诈骗者增多：广而告之**：一位成员敦促社区向**不懂技术**的人宣传 AI 生成技术的进展，以防止**诈骗**。
   - 他们引用了 [MKBHD 的最新视频](https://www.youtube.com/watch?v=OY2x0TyKzIQ)作为解释这些**威胁**的资源。
- **关于 MagVit 2 标记化医学图像的查询**：一位成员询问使用 **MagVit 2** 对医学图像进行标记化（tokenizing）的问题，特别是针对 **256x256x256** 的数据集。
   - 他们正考虑将其与基础的 **transformer 架构**结合，并寻求其他尝试过此方法的人的反馈。
- **介绍 APOLLO：优化 LLM 内存占用**：一篇 [arXiv 论文](https://arxiv.org/abs/2412.05270)介绍了 **APOLLO**，这是一种旨在通过修改 **AdamW** 的学习率自适应来减少 **LLM 训练**期间**内存占用**的优化器。
   - 论文解决了对昂贵的 **SVD 操作**的依赖等挑战，并提出通过**低秩优化器状态（low-rank optimizer state）**来近似学习率缩放。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **Shampoo 低比特分支查询**：一位成员询问 [shampoo low bit branch](https://github.com/axolotl-ai-cloud/axolotl/tree/shampoo-low_bit) 的实现是否有效，对其功能表现出兴趣。
   - 他们幽默地提到这个咨询是“帮朋友问的”，表明了对该话题的轻松参与。
- **默认梯度检查点提案**：一位成员提议将 `gradient_checkpointing` 默认设置为 **true**，认为它被广泛使用且能简化用户体验。
   - 他们强调这一改动将减少用户不必要的设置调整，暗示了可用性的潜在提升。

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Web Applets 开放标准发布**：明天，一位团队成员将介绍 [Web Applets 开放标准与 SDK](https://discord.com/channels/1089876418936180786/1089876419926032396/1315702507023896699)，展示其为 Agent 和人类创建丰富的图形化客户端应用的能力。
   - 该环节将包含**现场编码演示**、简短演讲，并开放提问和反馈环节。
- **鼓励在会议中进行实时反馈**：鼓励参与者在演讲过程中参与并提供**实时反馈**。
   - 欢迎互动讨论和咨询，确保引人入胜的学习氛围。



---



## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Rajat 发布 Dataoorts GPU Cloud**：**Rajat** 向社区介绍了 **Dataoorts GPU Cloud**，旨在支持下一代 AI 开发者的需求。
   - 他表达了对加入该团体的*兴奋*，强调了他对增强不断发展的 AI 领域资源的*承诺*。
- **支持下一代 AI 开发者**：**Dataoorts GPU Cloud** 旨在满足 **Rajat** 介绍的**下一代 AI 开发者**的需求。
   - 这一举措显示了为不断发展的 AI 领域提供增强资源的*明确承诺*。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---


**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间保持沉默，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要和链接


{% if medium == 'web' %}




### **Codeium / Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1314690778165149788)** (6 条消息): 

> `Cascade 定价变更，Windsurf 1.0.7 发布，新支持工单系统，保留 Pro 计划定价，Cascade 功能更新` 


- **Cascade 定价变更公布**：Cascade 的定价模型正在演进，推出了 **$15/月** 的 **Pro 层级**和带有无限额度的 **$60/月** 的全新 **Pro Ultimate 层级**。
   - 一个新的**额度系统**将帮助管理高级模型的使用，并提供可购买的 **Flex 额度**。
- **Windsurf 1.0.7 发布并修复 Bug**：Windsurf **1.0.7** 现已上线，包含针对 **1.0.6** 版本的几项次要 Bug 修复，增强了用户的整体稳定性。
   - 公开的 [更新日志](https://codeium.com/changelog) 详细说明了这些更新，包括对使用透明度和定价信息的调整。
- **专用支持工单系统上线**：[Codeium Support](https://www.codeium.com/support) 现已建立新的**专用工单系统**，以提供更好的协助和响应速度。
   - 鼓励用户查看自助文档，并通过新系统提交请求以获得有效支持。
- **为早期用户保留 Pro 计划定价**：在最近定价变更之前订阅 Windsurf 的用户将继续无限期享受 **$10/月 的 Pro 计划**。
   - 任何已经支付了新 **$15** 费用的用户将获得 **$5** 退款，为早期采用者维持原始定价。
- **增强 Cascade 功能的新特性**：更新后的 Cascade 现在允许上传大于 **1MB** 的图片，并为 Flow 额度用尽的用户引入了 **Legacy Chat** 模式。
   - 此外，用户可以在设置面板中轻松查看其 Cascade 使用情况，以便更好地进行跟踪。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor and Codeium extensions</a>：Windsurf 编辑器的最新更新和变更。</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>：未找到描述</li><li><a href="https://x.com/windsurf_ai/status/1865131244574642639">来自 Windsurf (@windsurf_ai) 的推文</a>：关于未来定价和层级的一些更新。https://codeium.com/pricing</li><li><a href="https://www.codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队获取个性化协助。
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1314686555293745223)** (710 条消息🔥🔥🔥): 

> `Windsurf 定价与额度、AI 限制与用户体验、IDE 对比：Cursor vs. Windsurf、用户注册与支付挑战、AI 交互改进建议` 


- **Windsurf 定价结构混乱**：用户讨论了近期定价更改为 $10 的影响，以及早期采用者是否能保留之前的权益，并确认新限制已生效。
   - 许多人对额度系统表示沮丧，称目前的限制无法支持高效的开发工作。
- **AI 上下文理解问题**：多位用户报告遇到了诸如 'The code edit failed to apply' 和 'Cascade has encountered an internal error' 的错误，特别是在使用 Cascade Base 模型时。
   - 大家一致认为，额度消耗和上下文保留问题严重阻碍了 AI 模型的有效性。
- **IDE 使用：在不同方案间切换**：几位用户分享了同时使用 Cursor 和 Windsurf 的策略，建议在不同 IDE 之间切换可以帮助解决其中一个出现的问题。
   - 对话表明，用户更倾向于通过使用多种工具来保持灵活性和效率。
- **注册与支付问题**：用户在注册过程中遇到困难，特别是在某些无法使用 PayPal 等支付方式的地区。
   - 一些人建议联系支持团队寻求帮助，强调国际用户需要更便捷的支付选项。
- **用户对 AI 改进的建议**：一些用户提议实现 negative prompt 系统或上下文提醒，以提高 AI 性能并减少重复提醒的需求。
   - 社区表达了对增强功能的整体渴望，希望能够简化与 AI 的交互并提高效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage">付费计划与额度使用 - Codeium 文档</a>: 无描述</li><li><a href="https://codeium.com/plan">计划设置</a>: 未来的编辑器，就在今天。Windsurf Editor 是首款由 AI agent 驱动的 IDE，让开发者保持专注。现已支持 Mac, Windows 和 Linux。</li><li><a href="https://www.salesforce.com/agentforce/">Agentforce: 创建强大的 AI Agents</a>: 构建并定制自主 AI agents，全天候支持您的员工和客户，包括与 Salesforce 生态系统的全面集成。</li><li><a href="https://codeium.canny.io/feature-requests">功能请求 | Codeium</a>: 向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。</li><li><a href="https://codeium.com/support">支持 | Windsurf Editor 与 Codeium 扩展</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.com/support.">页面未找到 | Windsurf Editor 与 Codeium 扩展</a>: Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。也是首个 agentic IDE —— Windsurf 的构建者。</li><li><a href="https://tenor.com/view/excited-fuego-gif-26833875">Excited Fuego GIF - Excited Fuego - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_p">Reddit - 深入探索一切</a>: 无描述</li><li><a href="https://www.youtube.com/watch?v=g46q1IjClz8">Luminary 0.0.7 概览</a>: 无描述</li><li><a href="https://xkcd.com/2044/">沙箱循环</a>: 无描述</li><li><a href="https://codeium.com/contact">联系 | Windsurf Editor 与 Codeium 扩展</a>: 联系 Codeium 团队以获取支持并了解更多关于我们企业级服务的信息。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt">Reddit - 深入探索一切</a>: 无描述</li><li><a href="https://codeium.com/pricing">定价 | Windsurf Editor 与 Codeium 扩展</a>: Codeium 对个人用户永久免费。团队可以通过我们的企业级服务提升水平，获得更强的个性化和灵活部署。</li><li><a href="https://www.youtube.com/watch?v=mc7O3KdO1cs">Next.js 音频转录与 Stripe 支付集成 | OpenAI Whisper API 与 PostgreSQL 演示</a>: 在此视频中，我演示了一个全栈应用程序，它结合了使用 OpenAI Whisper API 的音频转录功能和通过 Stripe 实现的安全支付处理...
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1314683015976190012)** (508 条消息🔥🔥🔥): 

> `Windsurf 定价模式、模型切换策略、AI 上下文窗口、Codeium 功能、Cascade 用户体验` 


- **对 Windsurf 定价结构的担忧**：用户讨论了对 Windsurf 定价模式的不满，认为这会产生一种“匮乏心态”，并给编程体验增加阻碍。
   - 许多人认为，如果定价结构更加透明且对用户友好，将提升他们对该工具的整体满意度。
- **模型切换的好处**：社区建议利用模型切换来节省 Flow 操作次数，提倡将 Cascade 作为默认模型，而将外部模型视为补充。
   - 用户表示，了解不同模型之间如何维持上下文（Context）对于优化工作流至关重要。
- **对 Cascade 的改进建议**：用户呼吁升级 Cascade Base 模型，并实现联网搜索（web searching）和自定义指令（custom instructions）等功能。
   - 用户认为这些增强功能可以显著提升 Windsurf 的性能和易用性。
- **AI 模型用户体验**：用户比较了不同 AI 模型的使用体验，指出虽然 Claude 和 4o 等模型表现良好，但 Cascade 等其他模型仍需进一步改进。
   - 不同模型在性能和完成任务方式上的差异，凸显了对其功能进行更好集成的需求。
- **理解 AI 上下文**：讨论强调了 AI 中上下文窗口（Context Windows）的概念，用户强调需要更有效地管理和传递上下文。
   - 大家一致认为，更好地理解和操作上下文可以增强 AI 编程助手的实际效用。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>：未找到描述</li><li><a href="https://magic.dev/waitlist">Waitlist — Magic</a>：Magic 是一家 AI 公司，致力于构建安全的 AGI，以加速人类在世界最重要问题上的进展。</li><li><a href="https://codeium.com/contact">Contact | Windsurf Editor and Codeium extensions</a>：联系 Codeium 团队以获取支持并了解更多关于我们企业级方案的信息。</li><li><a href="https://addyo.substack.com/p/the-70-problem-hard-truths-about">The 70% problem: Hard truths about AI-assisted coding</a>：一份实地指南，以及为什么我们需要重新审视我们的预期。</li><li><a href="https://tenor.com/view/kekwtf-gif-18599263">Kekwtf GIF - Kekwtf - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获得个性化协助。</li><li><a href="https://tenor.com/view/oh-really-gif-24755231">Oh Really GIF - Oh Really - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/plan">Plan Settings</a>：未来的编辑器，就在今天。Windsurf Editor 是首个由 AI Agent 驱动的 IDE，让开发者保持专注流。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://codeium.com/support.">Page Not Found | Windsurf Editor and Codeium extensions</a>：Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://www.youtube.com/watch?v=SrPmkpgRbkE">I Spent 100 Hours with Cursor AI Agents and Here's What I Learned</a>：Cursor AI Agents 展现了一种构建应用的新思维方式。更多关于 AI Agents 的视频：The Era of AI Agents: https://youtu.be/qc9fqCGgixM?si=dgqxKtUhp82I...</li><li><a href="https://www.infoworld.com/article/3617048/meta-quietly-leans-on-rival-gpt-4-despite-zuckerbergs-bold-llama-claims.html">Meta quietly leans on rival GPT-4 despite Zuckerberg’s bold Llama claims</a>：尽管 Meta 大力推介其 Llama 模型，但该公司正引入 OpenAI 的 GPT-4 来增强内部工具和慈善事业。</li><li><a href="https://claude.site/artifacts/4a226c3a-09ae-4fb9-bbe5-026d11743be6">Claude Artifact</a>：体验由 Claude 用户创建的 Artifacts</li><li><a href="https://magic.dev/blog/100m-token-context-windows">100M Token Context Windows — Magic</a>：关于超长上下文模型的研究更新、我们与 Google Cloud 的合作伙伴关系以及新融资。
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1314683715032711249)** (929 条消息🔥🔥🔥):

> `Cursor 性能问题，AI 模型现状，API 使用与价格对比，Cursor 与 Windsurf 的对比体验，对开源 AI 模型的反馈` 


- **关于 Cursor 能力的辩论**：用户对 Cursor 最近的性能下降表示沮丧，指出在文件修改、上下文理解以及整体生成质量方面存在问题，特别是在使用 Claude 模型时。
   - 一些用户认为质量下降与模型的高需求有关，而另一些用户则主张保持专注且清晰的 prompting 策略以最大化结果。
- **API 价格与使用见解**：讨论围绕使用 OpenAI 的 O1 Pro API 的性价比展开，用户表示在使用 Cursor 时不愿为多个订阅支付额外费用。
   - 社区建议探索拼车（group buys）以降低成本，并根据个人使用场景评估收益是否与其支出相符。
- **Cursor 与 Windsurf 的对比**：用户分享了他们对 Cursor 和 Windsurf 的不同体验，一些人发现 Windsurf 的功能更可靠，特别是在创建项目结构方面。
   - Cursor 通过 `.cursorrules` 和 AI 工具等功能实现的自定义能力受到关注，尽管仍有人更喜欢 Windsurf 的简洁和直接输出。
- **反馈与功能请求**：有用户请求改进文档处理、Git 集成，以及在 Cursor 中管理更大上下文文件的能力，以增强易用性。
   - 几位参与者建议，更好的测试和更平滑的更新过渡将显著提高用户对 Cursor 的满意度。
- **AI 生成代码的经验**：参与者讨论了 Claude 和 O1 等 AI 模型产生的不同结果，经验涵盖了从高效的代码生成到令人沮丧的幻觉（hallucinations）和无关输出。
   - 强调在 prompt 中进行精确的问题定义，对于优化任何 AI 模型提供的协助效果至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/VahidK/status/1865140156812136924">来自 Vahid Kazemi (@VahidK) 的推文</a>: 在我看来，我们已经实现了 AGI，在 O1 上这一点更加明显。我们还没有实现“在任何任务上都优于任何人类”，但我们拥有的是“在大多数任务上优于大多数人类”。一些 ...</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>: Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）构建应用的最快方式。</li><li><a href="https://poe.com/">Poe - 快速、有用的 AI 聊天</a>: 未找到描述</li><li><a href="https://forum.cursor.com/t/feature-request-long-context-mode/32187/5">功能请求：长上下文模式</a>: 我认为这种方法可以通过更多的用户控制和一些调整来复制长上下文的功能：最新更新中长上下文模式已消失 - #49 by fun_strange</li><li><a href="https://x.com/mckaywrigley/status/1865089975802646857?s=46">来自 Mckay Wrigley (@mckaywrigley) 的推文</a>: OpenAI o1 pro 比我预想的要好得多。这是第一次有一个模型发布后好到让我感到震惊。我截屏了 Coinbase，并让 4 个流行模型编写 c...</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的量化基准。</li><li><a href="https://forum.cursor.com/t/an-idiots-guide-to-bigger-projects/23646?u=ossianravn">笨蛋的大型项目指南</a>: ⚠ 警告：前方长文 ⚠ … 预计阅读时间：约 6 分钟，约占你生命的 0.000014%。如果你已经使用 Cursor 一段时间，并开始用它处理更复杂的项目，...</li><li><a href="https://x.com/Presidentlin/status/1865370199160979963">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>: Gemini -2.0-flash-exp 已添加到 Cursor，来自 /r/Bard</li><li><a href="https://changelog.cursor.com/">Cursor - 专为与 AI 结对编程而设计的 IDE。</a>: 未找到描述</li><li><a href="https://github.com/mullvad/mullvadvpn-app/releases/">发布版本 · mullvad/mullvadvpn-app</a>: 适用于桌面和移动设备的 Mullvad VPN 客户端应用。通过在 GitHub 上贡献代码参与 mullvad/mullvadvpn-app 的开发。</li><li><a href="https://mullvad.net/en/help/connecting-to-mullvad-vpn-from-restrictive-locations">在受限地区使用 Mullvad VPN</a>: 了解如何在难以下载我们的应用或连接受限的地点访问 Mullvad VPN。</li><li><a href="https://framer.university/resources">最佳免费 Framer 资源 — Framer University </a>: 为你的下一个项目发现最佳免费 Framer 资源，包括 Framer 组件、代码重写（code overrides）、动画和效果。通过精选的高质量资源提升你的 Framer 网站...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1314682807645110322)** (610 条消息🔥🔥🔥): 

> `Fine-tuning Llama 3.3, Collators and packing, Using models on limited hardware, Performance of AWQ and LoRA, Sora and its impact` 


- **在有限资源下微调 Llama 3.3**：用户讨论了在低端 GPU 上微调 Llama 3.3 模型的挑战，特别强调了他们在成本和显存（memory）需求方面的经验。
   - 一位用户提到，尽管硬件本身存在限制，但通过仔细的参数调整，成功缩短了训练时间。
- **理解 collators 和 packing**：Aemonalgiz 解释了训练中 collators 的作用，强调它们有助于构建高效的 batch，并能通过 padding 和 packing 等方法影响显存占用。
   - 使用 packing 代替 padding 可以优化训练速度，同时确保正确定义 attention masks 以防止学习错误。
- **在离线场景下使用 Llama 模型**：一位用户表示有兴趣在 Android 设备上部署 Llama 模型以供离线使用，旨在创建一个可以与文档交互的“本地 GPT”。
   - 他们询问了强大的本地系统以及如何利用这些模型制作功能完善的移动应用程序。
- **AWQ 和 LoRA 的训练限制**：讨论透露 AWQ 和 GPTQ 主要用于推理，不支持直接微调，并建议了一种工作流，使其能够与 LoRA 适配器配合，在 int4 或 fp16 模型上进行训练。
   - 有人指出，虽然 AWQ 模型具有某些优势，但大多数训练活动预计仍将在 int4 或 fp16 基础模型上进行。
- **对 Sora 的反应及其有效性**：社区成员对 Sora 进行了批评，认为尽管参数量很高，但与现有架构相比，该模型并未引入显著的进步。
   - 有人担心投入到训练此类模型中的资金是否带来了值得关注的性能提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lightning.ai/?">Lightning AI | Turn ideas into AI, Lightning fast</a>: AI 开发的一站式平台。协作编码、原型设计、训练、扩展、服务。直接在浏览器中完成 - 无需设置。由 PyTorch Lightning 的创作者打造。</li><li><a href="https://www.determined.ai/blog/lora-parameters">Finding the best LoRA parameters</a>: alpha、rank 和学习率如何影响模型准确性，以及 rank-stabilized LoRA 是否有所帮助。</li><li><a href="https://x.com/OpenAI/status/1865136373491208674">Tweet from OpenAI (@OpenAI)</a>: 今天我们预览了 Reinforcement Fine-Tuning，这是一种新的模型定制技术，使组织能够为编码、科学研究等领域的特定复杂任务构建专家模型...</li><li><a href="https://huggingface.co/spaces/DAMO-NLP-SG/Video-LLaMA">Video LLaMA - a Hugging Face Space by DAMO-NLP-SG</a>: 未找到描述</li><li><a href="https://huggingface.co/Mihaiii/Llama-3-pruned-45B-Drobeta-Turnu-Severin">Mihaiii/Llama-3-pruned-45B-Drobeta-Turnu-Severin · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/AtAndDev/marco-qwq-7B">marco-qwq-7B - a Hugging Face Space by AtAndDev</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF">unsloth/Llama-3.3-70B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9u2jd/llama_3_70b_layer_pruned_from_70b_42b_by_charles/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://x.com/UnslothAI/status/1865151062023512485">Tweet from Unsloth AI (@UnslothAI)</a>: 包括 GGUF + bnb 4-bit + 原始 16-bit 在内的 Llama 3.3 版本现已上线 @HuggingFace！在此查看 Llama 3.3 的所有版本：https://huggingface.co/collections/unsloth/llama-33-all-versions-67535...</li><li><a href="https://huggingface.co/blog/packing-with-FA2">Improving Hugging Face Training Efficiency Through Packing with Flash Attention 2</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/smangrul/573120738895551">@smangrul on Hugging Face: &quot;🚨 New Release of 🤗PEFT

1. 合并 LoRA 权重的新方法。参考这个……"</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit">unsloth/Llama-3.3-70B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/llama3-1">Finetune Llama 3.1 with Unsloth</a>: 通过 Unsloth 微调并运行 Meta 更新的 Llama 3.1 模型，支持 6 倍长的上下文长度！</li><li><a href="https://gist.github.com/fullstackwebdev/9e912fe4390c3a6959340afb19804566">gist:9e912fe4390c3a6959340afb19804566</a>: GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF/tree/main">unsloth/Llama-3.3-70B-Instruct-GGUF at main</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1c9u2jd/llama_3_70b_layer_pruned_from_70b_42b_by_charles/?rdt=53034">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/teknium1/ShareGPT-Builder">GitHub - teknium1/ShareGPT-Builder</a>: 通过在 GitHub 上创建账户，为 teknium1/ShareGPT-Builder 的开发做出贡献。</li><li><a href="https://huggingface.co/unsloth?search_models=smol">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/pull/730">Update Model Conversion Command in `save.py` to `convert_hf_to_gguf.py` by malibayram · Pull Request #730 · unslothai/unsloth</a>: 将 save.py 中的模型转换命令更新为 convert_hf_to_gguf.py。描述：此 PR 更新了 save.py 中的模型转换命令，以使用 convert_hf_to_gguf.py，从而与最新工具保持一致……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h8c9fu/llama_33_on_hugging_face_ggufs_4bit_bitsandbytes/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>: C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账户，为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://modular-model-spec.vercel.app">Modular Model Spec</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1314813795352186910)** (30 messages🔥): 

> `Open-source projects, Harmony project, Mental health data, LLM competition, Natural Language Processing` 


- **令人兴奋的开源倡议：Harmony**：一位成员分享了关于 **Harmony** 项目的细节，该项目利用 [Natural Language Processing](https://harmonydata.ac.uk/) 帮助研究人员回顾性地协调问卷项目和元数据。该工具对于跨研究比较和寻找兼容版本的问卷非常有用。
   - 该项目总部位于伦敦的 UCL，涉及多所大学，并提供了一项旨在改进其 LLM 匹配算法的竞赛，设有奖项，详见[此处](https://harmonydata.ac.uk/doxa/)。
- **增强 AI 匹配算法的竞赛**：Harmony 项目正在举办一项竞赛，参与者可以训练自己的 Large Language Models，以改进有时会误解句子相似性的匹配算法。任何感兴趣的人都可以通过在 [DOXA AI](https://harmonydata.ac.uk/doxa/) 上注册来参加竞赛。
   - 鼓励参赛者加入 Harmony Discord 服务器进行讨论和获取更新，特别是在 🏅「matching-challenge」频道。
- **社区对 OpenAI 和市场价值的见解**：讨论中出现了对 **OpenAI** 的不满，一位成员建议，由于不允许对其模型进行逆向工程，他们可能会失去市场价值。这种情绪得到了其他人的共鸣，表明了对 OpenAI 保护策略的共同看法。
   - 成员们辩论了市场中 AI 模型竞争的影响，一些人认为 OpenAI 的行为更多是为了资产保护，而非害怕竞争。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/how-you-feel-after-saying-that-awesome-dog-cool-dog-sussy-gif-23764852">How You Feel After Saying That Awesome Dog GIF - How You Feel After Saying That Awesome Dog Cool Dog - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://harmonydata.ac.uk/">Harmony | 全球上下文数据协调平台</a>: 全球上下文数据协调平台</li><li><a href="https://harmonydata.ac.uk/doxa/">Competition to train a Large Language Model for Harmony on DOXA AI | Harmony</a>: 全球上下文数据协调平台
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1314693642098835507)** (214 条消息🔥🔥): 

> `使用 Unsloth 进行 Multi-GPU 训练，Unsloth 安装中的错误解决，针对特定任务的模型优化，针对各种应用的 Llama 模型微调，访问和使用模型权重` 


- **关于 Multi-GPU 训练支持的确认**：用户继续确认 Unsloth 目前不支持通过 DDP 进行 Multi-GPU 训练，并强调了在 Llama3.2-11B-Vision 的 Visual Instruction 微调中对该功能的需求。
   - 成员们指出，在单 GPU 上使用 Unsloth 已被证明比 Multi-GPU 设置更快，并讨论了具体的硬件配置。
- **安装 Unsloth 并解决环境错误**：用户在安装 Unsloth 时遇到了问题，特别是在外部管理的环境中，因此建议使用 conda 进行更好的包管理。
   - 几位用户分享了成功安装 Unsloth 的命令，解决了安装过程中出现的依赖错误。
- **针对特定任务的模型优化**：几位参与者讨论了 Llama 模型在各种场景下的性能，例如针对文本分类的微调或针对多模态数据集的优化。
   - 成员们强调了适当模型配置的重要性，并讨论了调整 lm_head 维度以适应不同标签大小的策略。
- **有效地微调 Llama 模型**：用户在微调 Llama 及其相关模型时面临挑战，特别是在 Adapter 训练和处理大上下文长度方面。
   - 社区分享了关于使用现有脚本进行微调的见解，同时提醒注意数据质量对模型性能的影响。
- **访问模型权重以进行部署**：成员们询问了如何下载 Llama3.3 模型权重以进行本地部署，建议参考 Unsloth 文档或 Hugging Face 仓库进行访问。
   - 参与者澄清了获取权重的必要步骤，并讨论了训练和部署中模型版本控制的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/unsloth/SmolLM2-360M-bnb-4bit">unsloth/SmolLM2-360M-bnb-4bit · Hugging Face</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：查看下方列表获取我们所有的 Notebook：</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models">All Our Models | Unsloth 文档</a>：查看下方列表获取我们所有已上传的 GGUF、16-bit 和 4-bit bnb 模型</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md">llama.cpp/docs/build.md at master · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/timothelaborie/text_classification_scripts/blob/main/unsloth_classification.ipynb">text_classification_scripts/unsloth_classification.ipynb at main · timothelaborie/text_classification_scripts</a>：使用 Llama 和 BERT 进行文本分类的脚本 - timothelaborie/text_classification_scripts
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1314945004996919418)** (2 条消息): 

> `Awesome RAG, 即将发布的文章` 


- **关于 Awesome RAG 仓库的讨论**：一位成员分享了一个专注于 **RAG-VectorDB-Embeddings-LlamaIndex-Langchain** 的 [GitHub 仓库](https://github.com/lucifertrj/Awesome-RAG)。
   - 该仓库欢迎贡献，为那些有兴趣学习这些技术的人提供了宝贵的资源。
- **对 Article 102 的期待**：一位成员对潜在的 **Article 102** 表达了热情，表明社区渴望获得更多资源。
   - *“感谢提供这么棒的内容！”* 被作为反馈重点列出，反映了社区的赞赏。



**提到的链接**：<a href="https://github.com/lucifertrj/Awesome-RAG/">GitHub - lucifertrj/Awesome-RAG: RAG-VectorDB-Embedings-LlamaIndex-Langchain</a>：RAG-VectorDB-Embedings-LlamaIndex-Langchain。通过在 GitHub 上创建账户为 lucifertrj/Awesome-RAG 的开发做出贡献。

  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1314849149530804266)** (11 条消息🔥): 

> `OpenAI Triton 库, AWQ 量化方法, Hyperfitting 现象, 显存高效优化技术, 基于文本的模型发布` 


- **Unsloth 采用 OpenAI Triton 进行高效训练**：Unsloth 利用 [OpenAI Triton 库](https://github.com/rkinas/triton-resources) 进行快速且显存高效的训练，并分享了一份用于学习 Triton 的宝贵资源精选列表。
   - 社区表现出极大的热情，一位成员表示这“非常酷”！
- **AWQ 量化 vs Unsloth 量化**：讨论围绕 Unsloth 的量化是否只是在复制 AWQ 方法展开。AWQ 仅关注激活值（activation），而 Unsloth 同时考虑了激活值和权重的量化误差。
   - 成员们得出结论，两种方法都是可行的，Unsloth 也承认其与 AWQ 存在相似之处。
- **Hyperfitting 增强长序列生成**：一篇关于名为 Hyperfitting 方法的论文强调了其减少生成文本重复的能力，在保持 MMLU 和 GLUE 分数的同时，在长上下文环境下实现了更好的性能。
   - 该方法涉及在小型数据集上进行训练，直到损失接近于零。一位成员表示有兴趣尝试这项技术。
- **显存高效 LLM 优化器的开发**：一项关于显存高效优化的新工作介绍了 *APOLLO*，这是一种旨在改进广为人知的 AdamW 优化器显存占用的方法。
   - APOLLO 对学习率自适应规则进行了粗粒度化处理，在不依赖昂贵的 SVD 操作的情况下提高了可扩展性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>：大型语言模型 (LLMs) 在训练期间以耗费显存而闻名，特别是使用流行的 AdamW 优化器时。这种显存负担使得必须使用更多或更高端的 GPU，或者减少 ...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h8ep1w/the_hyperfitting_phenomenon_sharpening_and/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2412.04318">The Hyperfitting Phenomenon: Sharpening and Stabilizing LLMs for Open-Ended Text Generation</a>：本文介绍了在极小数据集上过拟合预训练大型语言模型 (LLMs) 的反直觉泛化结果。在开放式文本生成的设置下，这通常被认为...</li><li><a href="https://github.com/rkinas/triton-resources">GitHub - rkinas/triton-resources: 学习和探索 Triton 的精选资源列表，Triton 是 OpenAI 用于编写高效 GPU 代码的编程语言。</a>：学习和探索 Triton 的精选资源列表，Triton 是 OpenAI 用于编写高效 GPU 代码的编程语言。 - rkinas/triton-resources
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1314692687508537385)** (430 条消息🔥🔥🔥): 

> `Gemini 模型性能, O1 Pro vs Sonnet, 用于编程的 AI 助手使用, O1 Pro 中的文件处理, 量子计算进展`

- **关于 Gemini 模型性能的讨论**：用户讨论了对新模型 `gemini-exp-1206` 的使用体验，认为其性能优于 Sonnet 3.5，但也对其在正确格式排行榜上较低的排名表示担忧。
   - 该模型在使用 diff 模式时性能峰值达到 69%，在使用 whole 模式时约为 80.5%，引发了关于如何提升其在编程任务中表现的讨论。
- **O1 Pro 在编程任务中的效率**：与 Sonnet 相比，O1 Pro 在调试和架构代码方面的卓越推理能力受到了称赞，用户对其处理复杂 Bug 修复的能力给予了高度评价。
   - 用户正在权衡 200 美元的高昂成本与其效率，部分用户考虑只有在能看到实质性改进的情况下，才会选择在现有工具之外使用 O1 Pro。
- **用于编程的 AI Assistant 使用情况**：讨论强调了 Aider 和 O1 等 AI 工具如何被用于生成和调试代码，表明它们在工作流效率中承担着不同的角色。
   - 用户分享了利用这些工具的策略，包括将 O1 用于复杂任务，将 Claude 用于日常操作，以优化成本和性能。
- **O1 Pro 的文件处理能力**：针对 O1 Pro 附加代码文件的能力提出了疑问，注意到目前仅支持附加图像，期待未来的更新能增强此功能。
   - 社区期待在延长的促销期间看到改进，并强调了通过更好的文件处理来提升可用性的需求。
- **量子计算进展**：用户讨论了 Google 发布的 Willow 量子计算芯片，据称与传统超级计算机相比，该芯片能显著缩短复杂任务的计算时间。
   - 同时也提出了对其在专业领域之外实际应用的担忧，并希望为量子芯片提供改进的编程语言或 SDK。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Presidentlin/status/1865370199160979963">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：Gemini -2.0-flash-exp 已添加到 Cursor，来自 /r/Bard</li><li><a href="https://aider.chat/docs/usage/copypaste.html">通过网页聊天进行复制/粘贴</a>：Aider 支持与 LLM 网页聊天 UI 配合使用</li><li><a href="https://aider.chat/docs/more/edit-formats.html#udiff">编辑格式</a>：Aider 使用多种“编辑格式”让 LLM 编辑源文件。</li><li><a href="https://aider.chat/docs/usage/tips.html">技巧</a>：使用 Aider 进行 AI 结对编程的技巧。</li><li><a href="https://aider.chat/docs/usage/copypaste.html#terms-of-service">通过网页聊天进行复制/粘贴</a>：Aider 支持与 LLM 网页聊天 UI 配合使用</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.reddit.com/r/singularity/comments/1h90tqx/_/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://x.com/sundarpichai/status/1866167562373124420">来自 Sundar Pichai (@sundarpichai) 的推文</a>：我们将 Willow 视为构建实用量子计算机旅程中的重要一步，它在药物研发、聚变能源、电池设计等领域具有实际应用价值。详情请见：https...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1h7sjyt/windsurf_cascade_leaked_system_prompt/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://esbuild.github.io/api/#splitting,">esbuild - API</a>：未找到描述</li><li><a href="https://github.com/raphaelmansuy/code2prompt">GitHub - raphaelmansuy/code2prompt: Code2Prompt 是一款强大的命令行工具，通过生成包含代码库内容的全面 Markdown 文件，简化了为 Large Language Models (LLMs) 提供上下文的过程。⭐ 如果你觉得 Code2Prompt 有用，请考虑在 GitHub 上给我们一个 star！这有助于我们接触到更多开发者并改进工具。⭐</a>：Code2Prompt 是一款强大的命令行工具，通过生成包含代码库内容的全面 Markdown 文件，简化了为 Large Language Models (LLMs) 提供上下文的过程...</li><li><a href="https://github.com/lanqian528/chat2api">GitHub - lanqian528/chat2api: 一个可以将网页版 ChatGPT 转换为 OpenAI API 格式的服务。</a>：一个可以将网页版 ChatGPT 转换为 OpenAI API 格式的服务。 - lanqian528/chat2api</li><li><a href="https://github.com/mufeedvh/code2prompt">GitHub - mufeedvh/code2prompt: 一个将代码库转换为包含源码树、提示词模板和 Token 计数功能的单个 LLM 提示词的 CLI 工具。</a>：一个将代码库转换为包含源码树、提示词模板和 Token 计数功能的单个 LLM 提示词的 CLI 工具。 - mufeedvh/code2prompt</li><li><a href="https://github.com/Aider-AI/aider.git">GitHub - Aider-AI/aider: Aider 是你终端里的 AI 结对编程工具</a>：Aider 是你终端里的 AI 结对编程工具。通过在 GitHub 上创建账号来为 Aider-AI/aider 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1314708756000673842)** (78 条消息🔥🔥): 

> `Aider 的功能与模式，排查 API 速率限制问题，Aider 中的脚本自动化，Aider 与语言服务器的集成，Aider 处理新文件的方法` 


- **理解 Aider 的角色与模式**：用户讨论了 Aider 的功能，特别是关于其 Architect 和 Editor 模式，质疑 Architect 模式应该生成代码还是仅进行规划。
   - 一位成员建议 Aider 在处理简单任务时可以单独使用 QWQ 和 Qwen 模型。
- **API 速率限制问题**：多位用户在使用 Aider 配合 OpenAI API 时遇到了速率限制错误，引发了关于 Token 限制如何随时间以及在不同会话中应用的讨论。
   - 一位用户对高 Token 使用量以及 Aider 的处理方式是否影响 API 限制表示困惑，特别是在暂停使用一段时间之后。
- **在 Aider 中自动化多个 Prompt**：一位成员描述了他们管理多个 Prompt 以格式化简历的过程，并寻求将这些 Prompt 链式自动化的方法。
   - 有建议称成员可以使用 Aider 中的脚本选项进行命令行自动化，利用对多个文件的批处理功能。
- **将 Aider 与语言服务器集成**：一位用户询问了 Aider 与语言服务器之间的集成，旨在通过“查找引用”和“跳转到定义”等功能增强代码探索。
   - 讨论指出 Aider 利用 repo map 来理解代码库的整体结构和关系，这可能有助于此类集成。
- **使用 Aider 管理新文件**：成员们对 Aider 识别和引用会话期间创建的新文件的能力以及如何刷新文件列表表示关注。
   - 会议强调，包含相关文件并管理 git 交互是确保 Aider 在大型代码库中发挥效力的关键。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/usage/tips.html">Tips</a>: Aider AI 结对编程技巧。</li><li><a href="https://aider.chat/docs/repomap.html">Repository map</a>: Aider 使用 git 仓库地图为 LLM 提供代码上下文。</li><li><a href="https://aider.chat/docs/scripting.html">Scripting aider</a>: 你可以通过命令行或 Python 对 Aider 进行脚本化操作。</li><li><a href="https://aider.chat/docs/more/analytics.html">Analytics</a>: 选择性加入，匿名，无个人信息。</li><li><a href="https://x.com/tom_doerr/status/1865825047749013864?s=46&t=FfXrBepo4K-8IYa7B4PHlg">Tweet from Tom Dörr (@tom_doerr)</a>: 我让 Aider (Sonnet) 修复一个数据库错误，结果它把数据库删了 😭</li><li><a href="https://aider.chat/docs/faq.html#can-i-use-aider-in-a-large-mono-repo">FAQ</a>: 关于 Aider 的常见问题。</li><li><a href="https://www.swebench.com/">SWE-bench</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1315271321646661643)** (61 messages🔥🔥): 

> `Mojo 编译器特性、论坛 Bug 报告、周边需求、AI 生成内容政策` 


- **Mojo 编译器优化**：讨论重点在于 Mojo 中针对 SIMD 大小的动态优化，以解决硬件兼容性问题，部分成员建议引入类似于 C/C++ 编译器的多版本化（multiversioning）功能。
   - 成员们表示，Mojo 的编译过程虽然能带来性能提升，但也引发了对受影响用户系统可移植性的担忧。
- **论坛用户体验问题**：多位用户报告在论坛提交 Bug 报告时遇到速率限制（rate limit），并讨论了 UI Bug 以及部分功能失效的问题。
   - 特别是用户偏好设置中一个名为 'Users' 的标签页被指出缺乏实际用途，版主确认该功能可能会随着账号等级的提高而启用。
- **周边与 T-Shirt 需求**：一位用户表达了对 T-shirt 的强烈渴望，并指出需要一个周边商店来增强社区参与度。
   - 这一请求引发了关于除了 T-shirt 之外增加帽子的轻松讨论，保持了活跃的氛围。
- **404 页面行为**：用户注意到当遇到 404 页面时，输入少于 3 个字符的搜索查询会导致错误消息，而不是重定向到搜索页面。
   - 建议应为用户提供更宽容的体验，包括对查询长度的清晰反馈和改进的导航。
- **AI 生成内容政策**：版主宣布任何明显的 AI 生成内容都将被删除，并对作者发出警告，强调在论坛维持真实讨论的重要性。
   - 该政策旨在营造有趣且真实的社区氛围，而在 Swag Challenge 期间的推广活动将不受 AI 贡献的影响。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://forum.modular.com/search?q=eee">Search results for &#39;eee&#39; - Modular</a>: 未找到描述</li><li><a href="https://forum.modular.com/t/simplifying-gpu-programming-with-parametric-tile-level-tensors-in-mojo-llvm-developers-meeting-2024/38">Simplifying GPU programming with parametric tile-level tensors in Mojo (LLVM Developers&#39; Meeting 2024)</a>: 未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3651">[Feature Request] Function multiversioning · Issue #3651 · modularml/mojo</a>: 查看 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？我希望拥有与 Clang 等效的功能...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1315740520202502165)** (1 messages): 

> `Modular 论坛发布、Discord 的角色、Swag Challenge、向 Ahmed 咨询 GPU 编程、论坛反馈` 


- **Modular 论坛正式开放**：**Modular 论坛**现已上线 [forum.modular.com](http://forum.modular.com/)，邀请社区成员探索并贡献内容。
   - 该平台旨在提供官方回复、深入探讨技术问题，并通过被索引的帖子为未来的 Modular 用户提供支持。
- **Discord 保持活跃**：**Discord** 社区将继续蓬勃发展，作为快速聊天和日常互动的空间。
   - 鼓励成员使用论坛进行官方查询和详细讨论。
- **通过 Swag Challenge 庆祝**：**Swag Challenge** 随论坛发布同步开启，截至今日结束前积分排名前 5 的用户将获得 **Mojo T-shirts** 奖励。
   - 可以通过创建新帖子和参与现有内容互动来赚取积分。
- **与 Ahmed 交流 GPU 编程**：成员们可以针对 Ahmed 在 **2024 LLVM Developers’ Meeting** 上的演讲回顾向他提问：[*Simplifying GPU Programming*](https://forum.modular.com/t/simplifying-gpu-programming-with-parametric-tile-level-tensors-in-mojo-llvm-developers-meeting-2024/38)。
   - Ahmed 将在全天时间内回答相关咨询。
- **征集论坛反馈**：鼓励社区成员在 [Forum Feedback 类别](https://forum.modular.com/c/feedback/2) 中分享对新论坛的想法。
   - Modular 团队渴望收到任何建设性的见解以改进平台。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1314735948969541662)** (283 条消息🔥🔥): 

> `Mojo Language Features, Linear Types Proposal, Game Development with Mojo, Comparison with Other Languages, Memory Management in Programming` 


- **关于线性类型提案的讨论**：分享了一个关于 Mojo 中线性及显式销毁类型的提案，并对其在防止 GUI 开发中因调用错误的销毁方法而产生错误的易读性和实用性发表了评论。
   - 针对选择实现新的 `destroy` 关键字而不是复用 Python 的 `del` 提出了疑问，并讨论了在线性结构体（linear struct）上下文中的作用域和用法。
- **Mojo 的低级编程潜力**：参与者讨论了 Mojo 与 Rust 和 C++ 等其他语言相比在低级编程能力方面的定位，强调了其结合强大抽象与速度的能力。
   - 贡献者指出，虽然 Mojo 专注于系统编程，但其目标也是为了满足各种应用领域的高级用例。
- **Mojo 与 Vale 及其他语言的对比**：对话包括了 Mojo 与 Vale、Zig 和 Odin 等其他语言的对比，重点关注它们的优势和目标应用。
   - Mojo 被描述为优先考虑低级编程，同时比 Vale 提供更多的抽象，而 Vale 旨在用于无需直接硬件访问的高性能用例。
- **对使用 Mojo 进行游戏开发的兴趣**：表达了对使用 Mojo 进行游戏开发的兴趣，表现出对其与 C# 和 C++ 等成熟语言相比表现如何的好奇。
   - 随着对语言能力的持续讨论，参与者认识到了在游戏开发背景下应用 Mojo 的挑战和潜力。
- **关于内存管理的背景技术讨论**：讨论强调了内存管理方面正在进行的调研，以及在 Mojo 中需要有效的分配器（allocator）方案来增强其低级编程能力。
   - 分享了关于不同语言之间内存管理方法差异的见解，强调了在游戏开发和系统编程中对高效且灵活系统的需求。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/hermes-lets-do-this-futurama-gif-9434197">Hermes Lets Do This GIF - Hermes Lets Do This Futurama - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://forum.modular.com/t/dynamic-traits-for-easier-programming/85?u=lesoup-mxd">Dynamic traits for easier programming</a>: 目前，Mojo 的 Any (AnyType) 不能在函数内部作为独立声明使用，它需要静态支持。如果能看到类似（或更好）C++ 的 auto 数据类型的东西就太酷了。</li><li><a href="https://nondot.org/sabre/Resume.html#talks">Chris Lattner's Resumé</a>: 未找到描述</li><li><a href="https://x.com/wordgrammer/status/1865925226149859659?t=e0YSWflFwuBcCPgsk_mgIA&s=19">wordgrammer (@wordgrammer) 的推文</a>: @_blinding_light 我认为他们会尝试，但不会真正奏效。LLM 曾经非常擅长学习 Python，因为有大量的训练数据。但我认为它们最终会变得更好...</li><li><a href="https://x.com/wordgrammer/status/1865917868623135221?t=z1kQkUhuk4Vsso-4uwJsnw&s=19">wordgrammer (@wordgrammer) 的推文</a>: 5 年内，几乎所有代码都将由 LLM 生成。当这种情况发生时，对类型系统、并发和编程范式的扎实理解将非常有用。研究 PLT 的人不再...</li><li><a href="https://www.youtube.com/watch?v=UavYVf0UEoc">Advanced Memory Management in Vale (with Evan Ovadia)</a>: Rust 改变了围绕内存管理的讨论——本周的嘉宾希望进一步推动这一讨论。本周我们邀请到了 Evan Ovadia...</li><li><a href="https://github.com/modularml/mojo/issues/3848">[Feature Request] [mojo-lang] [proposal] Add Linear / Explicitly Destroyed Types · Issue #3848 · modularml/mojo</a>: 审查 Mojo 的优先级。我已经阅读了路线图和优先级，并相信此请求符合优先级。你的请求是什么？参见线性 / 显式销毁类型的提案。...</li><li><a href="https://www.youtube.com/watch?v=IpuvQUVB8Cg)">2024 LLVM Dev Mtg - Implementing Linear / Non-destructible Types in Vale and Mojo</a>: 2024 LLVM 开发者大会。在 Vale 和 Mojo 中实现线性 / 不可销毁类型。演讲者：Evan Ovadia...</li><li><a href="https://github.com/modularml/mojo/pull/3548/files)">[stdlib] Move `StringRef` `find()` implementation to `Span` by martinvuyk · Pull Request #3548 · modularml/mojo</a>: 将 StringRef find() 的实现移动到 Span
</li>
</ul>

</div>
  

---

### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1314940633378590860)** (16 条消息🔥): 

> `Bolt 功能问题，Bolt 中的 Prompting 规范，功能实现挑战，Prompt 中的变量敏感性，改进 Prompt 的工具` 


- **Bolt 在功能实现上存在困难**：成员反馈 **Bolt** 中的某些功能无法正常工作，例如**添加记录按钮**在点击时没有响应。
   - 据观察，初步尝试通常只能生成前端界面，需要更具体的后续 Prompt 才能使功能真正可用。
- **需要更好的 Prompting 规范**：**用户**表示希望为 **Bolt** 提供一套有效的 Prompting 规范或工具，以减少问题并优化输出。
   - 另一位成员表示，他们正在积极开发这样一种工具，以协助用户创建更有效的 Prompt。
- **变量大小写问题令用户沮丧**：尽管要求在字符命名中保持 **Case Sensitivity**（大小写敏感），但 AI 仍会不恰当地更改变量名，这引发了担忧。
   - 用户反馈称，即使正确提供了 JSON 格式，**Claude** 仍会修改变量，这令人十分困扰。
- **Bolt 的付费功能限制**：讨论指出，由于运行 **Diffing** 功能需要额外资源，该功能在 **Bolt** 中仅作为付费选项提供。
   - 这一限制给那些希望在不产生额外费用的情况下获得更全面功能的用户带来了挑战。
- **社区协作与分享**：成员们鼓励分享改进 Prompt 有效性的想法和工具，展现了互助的社区氛围。
   - 一位用户幽默地请求允许将某位成员的想法分享到 Twitter，展示了良好的同僚情谊与协作精神。



**提到的链接**：<a href="https://www.youtube.com/watch?v=ofHGE-85EIA">我做了一个可以制作网站的网站</a>：📚 𝗠𝗮𝘁𝗲𝗿𝗶𝗮𝗹𝘀/𝗥𝗲𝗳𝗲𝗿𝗲𝗻𝗰𝗲𝘀：GitHub Repository (给它一个 star ⭐) → https://github.com/hkirat/bolt.newer0:00 - 介绍与架构图...

  

---

### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1314709247015256075)** (318 条消息🔥🔥): 

> `Bolt 中的 Token 管理、Supabase 集成、Bolt 的技术问题、开源版与生产版、Bolt 的社区资源` 


- **理解 Token 管理**：用户讨论了 Bolt 中 Token 使用的细节，强调 Token 每月重置且不结转，这让一些订阅者感到沮丧。
   - 会议澄清了单独购买的充值 Token 可以结转，而订阅 Token 在订阅期结束时会失效。
- **即将推出的 Supabase 集成**：关于 Bolt 原生 Supabase 集成的公告即将发布，通过回复团队的推文有机会获得早期访问权限。
   - 该集成旨在提升在 Bolt 中无缝构建包含数据库和身份验证的应用的开发体验。
- **用户面临的技术挑战**：许多用户报告了诸如依赖安装失败、无限加载错误以及 Firebase/IPFS 配置等影响开发进度的问题。
   - 社区成员提供了支持，分享了故障排除技巧和临时解决方案，以帮助解决开发过程中遇到的问题。
- **Bolt 的开源版与生产版**：社区讨论了 Bolt 官方开源版本与生产版本之间的区别，并警告出于兼容性原因不要同时使用它们。
   - 随着开源社区的持续贡献，目前正在努力对齐两个版本之间的特性和功能。
- **社区资源与知识共享**：Bolters.io 平台已更新社区驱动的资源，包括应用推荐、故障排除指南和教学视频链接。
   - 鼓励用户通过分享自己的问题、寻求帮助以及为共享知识库做出贡献来参与社区。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://supabase.com/partners/integrations/supabase_wrapper_stripe">Stripe Wrapper | 适配 Supabase</a>：使用 Supabase Wrappers 开发的 Stripe 外部数据包装器。</li><li><a href="https://x.com/stackblitz/status/1865904408254620148">来自 StackBlitz (@stackblitz) 的推文</a>：让我们加大筹码！获胜者还将获得第一件印有 Bolt 标志的连帽衫 (!!!) + 每只袖子上都有特别信息，纪念你可以用 Bol 构建的惊人成果...</li><li><a href="https://Bolters.io">Bolters.io | Bolt.new 无代码应用构建器的社区支持技巧、诀窍与知识库</a>：Bolt.new 的文档和指南</li><li><a href="https://bolters.io/docs/understanding-cors/">理解 WebContainer 中的 CORS</a>：了解跨源资源共享 (CORS)、其对 WebContainer 的影响以及当前的局限性</li><li><a href="https://blog.stackblitz.com/posts/design-system-component-documentation/">如何编写设计系统组件文档</a>：组件是 Web 设计实现的重要组成部分。在本文中，我们介绍了编写设计系统或组件库中组件文档的最佳实践。</li><li><a href="https://www.chakra-ui.com/docs/get-started/installation">安装 | Chakra UI</a>：如何在项目中安装和设置 Chakra UI</li><li><a href="https://github.com/stackblitz/bolt.new">GitHub - stackblitz/bolt.new: 提示、运行、编辑和部署全栈 Web 应用程序</a>：提示、运行、编辑和部署全栈 Web 应用程序 - stackblitz/bolt.new</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2985">功能请求：在 Bolt 中显示 .bolt 文件夹 · Issue #2985 · stackblitz/bolt.new</a>：你的功能请求是否与问题相关？请描述：不是问题，只是轻微的不便。描述你想要的解决方案：如果我能更新像 Bolt igno 之类的内容就太好了...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1315079691996237884)** (2 messages): 

> `Countless.dev launch, Claude 3.5 Sonnet updates, Integration with Poe` 


- **Countless.dev 让模型对比变得简单**：新推出的 [Countless.dev](https://www.producthunt.com/posts/countless-dev) 是一个免费且开源的工具，旨在帮助用户**对比 AI 模型**（包括 LLM 和视觉模型），可以轻松地按**价格、Token 限制或功能**进行排序。
   - 该项目目前已在 **Product Hunt** 上线，作者请求支持以冲击**第一名**的排名。
- **Claude 3.5 Sonnet 超出预期**：更新后的 Claude 3.5 Sonnet（版本号为 **claude-3-5-sonnet-20241022**）在保持 **Sonnet 价格**的同时，拥有**优于 Opus 的能力**，尤其在编程和数据科学任务中表现出色。
   - 新特性包括**增强的视觉处理能力**，以及针对复杂的、多步骤问题解决的**卓越 Tool Use 能力**。
- **与 Poe 集成以增强功能**：与 **Poe** 的集成允许访问高级功能，如 **OpenAI Whisper** 和 **Text-to-Speech**，为用户扩展了功能范围。
   - 此次集成是持续更新的一部分，旨在提升用户体验并扩展 AI 模型的能力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://poe.com/adamD123">Adam - Poe</a>: 未找到描述</li><li><a href="https://www.producthunt.com/posts/countless-dev"> Countless.dev - 发现、对比并选择 AI 模型 — 100% 免费 | Product Hunt</a>: Countless.dev 让探索、对比和计算各类 AI 模型（LLM、视觉模型等）的成本变得简单。按价格、Token 限制或功能排序，找到最适合您用途的模型...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1314684258576633866)** (318 messages🔥🔥): 

> `Llama Models, API Errors, Sora Model Features, OpenRouter Rate Limits, Mistral Model Updates` 


- **关于 Llama 模型性能的讨论**：用户对 **Llama 3.3** 和 **Hermes** 等各种模型的有效性表达了兴趣，强调了它们的智能功能，且部分模型是无审查（uncensored）的。
   - 讨论中分享了关于 **Llama** 因其强大的能力和较少的限制而成为热门选择的见解，同时也提到了**旧版 Gemini**。
- **API 错误的使用体验**：一位用户报告在免费模型中频繁遇到 “Provider Returned Error”，这表明问题与 **API 限制**有关。
   - 其他人提到，这些错误可能是由于提供商过载导致的，特别是在使用 **Claude** AI 时，这给使用带来了困扰。
- **Sora 模型特性与对比**：用户讨论了 **Sora** 模型的潜在功能，包括其显著的视频编辑 “remix” 功能，这预示着一个复杂的用户输入界面。
   - 有人询问了 video-to-video 的能力，并对 **Sora** 与 **Runway** 等现有工具相比的有效性持怀疑态度。
- **OpenRouter 的 Rate Limits**：出现了关于 OpenRouter **Rate Limits（速率限制）** 的疑问，讨论了如果用户拥有足够额度是否可以取消限制。
   - 设置这些限制的理由包括防止在缓存过期前账户余额出现大幅波动，重点在于维持低延迟。
- **Mistral 模型开发更新**：关于 **Mistral** 模型的更新显示，几个未发布的模型在宣布后不久就被撤回了。
   - 社区推测新的 **Codestral** 和 **mistral-ocr** 模型在通过 API 通知泄露后，是否会很快正式开放。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://sora.com/">Sora</a>: 将文本和图像转换为沉浸式视频。为故事添加动画，可视化想法，并将您的概念变为现实。</li><li><a href="https://openrouter.ai/chat">Chatroom | OpenRouter</a>: LLM Chatroom 是一个多模型聊天界面。添加模型并开始聊天！Chatroom 在您的浏览器本地存储数据。</li><li><a href="https://community.sambanova.ai/t/release-notes-december-6-2024/731">Release Notes - December 6, 2024</a>: 2024年12月5日，我们很高兴地推出了一些最令人兴奋的 Qwen 模型，以及领先的内容审核模型 Llama Guard 3，现已在 SambaNova Cloud 上提供。...</li><li><a href="https://openrouter.ai/api/v1`">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/quick-start">Quick Start | OpenRouter</a>: 开始使用 OpenRouter 进行构建</li><li><a href="https://internvl.github.io/blog/2024-12-05-InternVL-2.5/">InternVL2.5</a>: 未找到描述</li><li><a href="https://inference.net">Inference.net</a>: 经济实惠的生成式 AI</li><li><a href="https://x.com/DeepInfra/status/1865126860902011244">Tweet from DeepInfra (@DeepInfra)</a>: 🚨 重大新闻！@DeepInfra 在发布首日即以最低价格支持 Llama 3.3 70B：Llama 3.3 70B (bf16): $0.23/$0.40，Llama 3.3 70B Turbo (fp8): 每 1M 输入/输出 $0.13/$0.40。体验无缝的尖端 AI...</li><li><a href="https://docs.mistral.ai/getting-started/models/models_overview/">Models Overview | Mistral AI Large Language Models</a>: Mistral 提供两类模型：免费模型和高级模型。</li><li><a href="https://huggingface.co/collections/LGAI-EXAONE/exaone-35-674d0e1bb3dcd2ab6f39dbb4">EXAONE-3.5 - a LGAI-EXAONE Collection</a>: 未找到描述</li><li><a href="https://openrouter.ai/docs/limits">Limits | OpenRouter</a>: 设置模型使用限制</li><li><a href="https://openrouter.ai/">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/google/gemini-exp-1121:free/uptime)">Google: Gemini Experimental 1121 (free)</a>: Gemini 的实验性版本（2024年11月21日）。</li><li><a href="https://openrouter.ai/google/gemini-exp-1206:free">Gemini Experimental 1206 (free) - API, Providers, Stats</a>: Gemini 的实验性版本（2024年12月6日）。通过 API 运行 Gemini Experimental 1206 (free)。</li><li><a href="https://fal.ai">fal.ai | The generative media platform for developers</a>: fal.ai 是运行扩散模型的最快方式，提供开箱即用的 AI 推理、训练 API 和 UI Playgrounds</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>: 未找到描述</li><li><a href="https://huggingface.co/OpenGVLab/InternVL2_5-78B">OpenGVLab/InternVL2_5-78B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1h9ycjg/google_ceo_ai_development_is_finally_slowing_down/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>: 适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1314742260591759360)** (13 messages🔥): 

> `集成 Beta 功能请求、自定义提供商密钥、Amazon Bedrock 模型集成、Google Flash 模型访问` 


- **多个集成 Beta 功能访问请求**：多位用户请求访问 **集成 Beta 功能**，表明对试用此功能有浓厚兴趣。
   - *Hi, I'd like to request access to the integration beta feature.* 是多条消息中的共同主题。
- **对自定义提供商密钥的兴趣**：一位用户表示希望尝试 **自定义提供商密钥**，突显了可用集成选项的多样性。
   - 该请求展示了对集成领域增强功能的需求。
- **针对 Amazon Bedrock 的提议模型集成**：一位成员建议将 **Opus** 和 **Mistral Large** 添加到 **Amazon Bedrock** 识别的集成模型中。
   - 这一提议强调了用户对在当前集成能力范围内扩展可用模型的持续兴趣。
- **Google Flash 模型访问请求**：一位用户提到寻求访问 **Google Flash 1.5 模型**，表明了特定的技术兴趣。
   - *Hi I saw that I was to come here to get the beta for access to Google Flash 1.5 Model.* 说明了获取模型访问权限的平台引导。


  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1314688528688484353)** (173 messages🔥🔥):

> `LM Studio GPU Usage, Aider Integration Issues, Model Compatibility with LM Studio, Frontend Clients for LM Studio, Hardware Recommendations for AI Models` 


- **理解 LM Studio 的 GPU 使用情况**：使用 RX 6600 GPU 的用户意识到 LM Studio 采用 Vulkan 进行 GPU offloading，允许他们在无需安装 ROCm 的情况下运行模型。
   - 这为那些可能不熟悉 LM Studio 如何有效利用其硬件的 AMD 用户开辟了可能性。
- **Aider 集成面临的挑战**：与 Aider 的集成对某些用户来说比较困难，特别是由于 API key 设置和环境变量配置的问题。
   - 为了解决这些问题，建议用户设置一个随机的 API key，并确保参考 Aider 文档进行正确设置。
- **模型兼容性担忧**：用户对 LM Studio 缺乏对 Qwen2 VL 7B Instruct 等模型的支持表示沮丧，这限制了那些有兴趣使用新型 vision models 的用户的选择。
   - 建议使用通过 Pinokio 运行的 Florence-2 等替代方案，以探索视觉模型的其他选项。
- **前端客户端推荐**：推荐了几个连接到 LLM 服务器的 LM Studio 替代方案，包括 AnythingLLM 和 Open WebUI。
   - 鼓励用户探索这些选项，以获得满足特定需求的多样化特性和功能。
- **运行 AI 模型的硬件推荐**：讨论强调了将 GPU 规格与模型要求相匹配的必要性，特别是对于需要高 VRAM 的模型。
   - 用户获知了像 A100 这样性能强大且价格具有竞争力的 GPU 可选方案，有助于提升 AI 模型性能。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://aider.chat/docs/llms/lm-studio.html">LM Studio</a>: aider 是你终端里的 AI 结对编程工具</li><li><a href="https://tenor.com/view/ducktales-ducktales2017-infernal-internship-of-mark-beaks-mustache-disguise-gif-21524651">Ducktales Ducktales2017 GIF - Ducktales Ducktales2017 Mark Beaks 的地狱实习生伪装胡子 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: 用户友好的 AI 界面（支持 Ollama, OpenAI API, ...）</a>: 用户友好的 AI 界面（支持 Ollama, OpenAI API, ...） - open-webui/open-webui</li><li><a href="https://www.youtube.com/watch?v=OY2x0TyKzIQ">这段视频是 AI 生成的！SORA 评测</a>: SORA 生成视频。这是第一篇评测。在 https://ridge.com/MKBHD 获取最后时刻礼品高达 40% 的折扣。真实的观鸟视频：https://youtu.be/F...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hah3wi/im_afraid_to_ask_but_how_do_i_actually_quit_lm/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta 版本发布</a>: LM Studio Beta 版本发布</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/133">功能请求：将 LM Studio 用作本地网络中其他 LLM 服务器的客户端。· Issue #133 · lmstudio-ai/lmstudio-bug-tracker</a>: LM Studio 已经允许创建服务器并将其用于 API 请求。但它不允许 LM Studio 作为该服务器的客户端运行。场景如下：我有一台性能强大的机器在我的...</li><li><a href="https://msty.app/pricing">价格</a>: 超越普通聊天的 AI。私密、离线、拆分聊天、分支、并发聊天、Web Search、RAG、Prompts 库、Vapor 模式等。完美的 LM Studio、Jan AI 和 Perplexity 替代方案。</li><li><a href="https://anythingllm.com">AnythingLLM | 适合所有人的全能 AI 应用</a>: AnythingLLM 是你一直在寻找的 AI 应用。使用任何 LLM 与你的文档聊天，提高生产力，并完全私密地运行最新的 state-of-the-art LLM，无需技术背景...</li><li><a href="https://openwebui.com">Open WebUI</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/405215504640?_skw=nvidia+sxm2+a100+automotive&epid=22065174652&itmmeta=01JEN1KE9RQ1YWBFDDRKZW5DPD&itmprp=enc%3AAQAJAAAA4HoV3kP08IDx%2BKZ9MfhVJKlXVKcAPbbt4BKfHIZRKF59hbTZ1feCYGryXOYSawI6iKe9dLwqKsvwyNsCuUZjELMABTGofOnpvUo%2BtMQUb4pAg%2FwjOuKyc2GZiUSd6pdqXc%2B6Ut0kipS6Bz6%2BzSc7ziHnwnVysS9gbVBVYzQ1G7I9E9L8wCnnn9L1yV5ceMvTBC28Mg3VdqQIt8Rt9Nz1d1pDx4Nfdop7IXSq8hf%2FaXUZadVQpxnlzlVLFInHm6MHdyncyvXsT9cDQDDzWeo7PmE7sVSy8ukutomWpuWLnWnl%7Ctkp%3ABk9SR_rkzaH1ZA&LH_BIN=1">用于自动驾驶车辆的 NVIDIA DRIVE A100 AUTOMOTIVE SXM2 GPU 900-6G199-0000-C00 | eBay</a>: 未找到描述</li><li><a href="https://www.ebay.com/itm/135424248001?_skw=SXM2+to+pcie+adapter&itmmeta=01JEN1JXXJC6PWD6X0M3QKBZ0E&itmprp=enc%3AAQAJAAAA8HoV3kP08IDx%2BKZ9MfhVJKnak56dAfzHQ0oL5hPTiPhgYnoItcFYiWxP00DVmq67ke61OerN%2F7BeKBYlANLGPPzPsr6GFxjWky7SRfpEEYUAch5L1yWS4qlaLyOxXHqSXmu10yJM8uP5%2FlLDLP5GYN9KRE4yT7k0dNAtLZ9NIDHZrXwn9k0DmpWzchuOTZTSAJifhe12RCp4fhubFqH9ErgX%2FkTWbNp1OvsXkcJOVY0ATVxxdAJsOr3%2FERd2FTWgsOWCglMHXCGT6n%2FSLFHyiLc91rrtG1R6UC6ITHVPJKB%2Br72vwO2%2FjmXu%2F6Hh8kt00Q%3D%3D%7Ctkp%3ABFBM8N7LofVk">适用于 Nvidia Tesla V100 A100 SXM2 GPU 计算显卡的 SXM2 转 PCIE 转接卡 | eBay</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1314684213408305273)** (98 messages🔥🔥): 

> `LM Studio server capabilities, GPU setups and cooling solutions, Memory bandwidth and CPU performance, ROCm vs CUDA compatibility, Custom GPU riser designs` 


- **LM Studio 作为服务器**：一位成员确认，通过使用 'start lm studio'、'start lms server 1234' 和 'start ngrok tunnel' 等命令，可以轻松地与 100 英里外的朋友分享 LM Studio 硬件。他们分享了一个 [GitHub 链接](https://github.com/OIEIEIO/lm-studio-ngrok)，详细介绍了如何有效地实现这一点。
   - 另一位成员询问该设置是否支持 RAG 功能，从而引发了关于相应配置的讨论。
- **关于最佳 GPU 配置的见解**：成员们讨论了使用 3090 的情况，并分享了各自设置的见解，强调购买二手 3090 可以在实现性能目标的同时兼顾预算。一位成员指出二手市场有更好的选择，而另一位则提到如果不差钱，48GB 的 A6000 才是首选。
   - 讨论中提到了内存带宽对性能的重要性，以及更多 RAM 通道如何对 ML 系统产生益处。
- **混合 GPU 设置的挑战与兼容性问题**：针对在单台机器上同时使用 ROCm 和 CUDA 的担忧被提出，成员们指出由于兼容性问题，通常只能二选一。有人建议虽然 Vulkan 运行良好，但 ROCm 令人沮丧，且在某些 AMD GPU 型号上运行效果不佳。
   - 成员们分享了使用特定变量管理 GPU 行为的经验并提供了解决方案，尽管有人指出这些方案并不能产生可靠的结果。
- **自定义 GPU Riser 和冷却方案**：关于自定义 GPU Riser 支架的讨论强调了需要坚固且安全的设计，以支撑带有自定义散热器的 3090 等重型 GPU。一位成员分享了一个专门为其设置设计的垂直 GPU 支架的 [Thingiverse 链接](https://thingiverse.com/thing:2536978)，强调了安装多个 GPU 的挑战。
   - 成员们交流了高性能水冷设置的有效冷却方案，讨论了处理高温和工作负载需求所需的稳健性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.stlfinder.com/model/vertical-gpu-mount-HyJz7ZOE/1865168/">Vertical GPU Mount - STLFinder
</a>：未找到描述</li><li><a href="https://github.com/OIEIEIO/lm-studio-ngrok">GitHub - OIEIEIO/lm-studio-ngrok: How to Share Your Hardware and AI Frontend with a Friend 100 Miles Away - LM Studio Server</a>：如何与 100 英里外的朋友分享你的硬件和 AI 前端 - LM Studio Server - OIEIEIO/lm-studio-ngrok
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1314685140185780254)** (88 messages🔥🔥): 

> `Gemini exp 1206, Aurora image model, Sora video generation, WaveForms AI, NeurIPS conference`

- **Gemini exp 1206 表现亮眼**：Gemini exp 1206 因其性能受到关注，特别是在各种 Benchmark 和任务中优于之前的版本。用户分享了他们的体验，指出在代码辅助和 Benchmark 分数方面的提升，包括在 Aider 的代码编辑 Benchmark 上取得了创纪录的成绩。
   - 然而，一些用户对设置问题以及该模型在 Cursor 等不同环境中的协作功能表示困惑。
- **Aurora 图像模型登场**：xAI 新发布的 Aurora 图像生成模型引起了轰动，早期用户对其能力给予了评价，但也对某些用例表示失望。与现有模型的对比表明，Aurora 在细节方面表现出色，但在卡通渲染方面面临挑战。
   - 有人提出了关于它与 Flux 的创建者 Black Forest Labs 之间关系的问题，暗示后台可能存在合作。
- **Sora 视频生成能力揭晓**：Sora v2 将通过 Text-to-Video 和更详细的输出等功能增强视频生成能力。AI 领域的知名人士表达了对 Sora 即将发布的兴奋之情，认为它可能会显著影响用户参与度。
   - 在发布期间，各种 Demo 展示了其潜力，许多人预测与 Pro 和 Plus 订阅层级相关的用量将激增。
- **WaveForms AI 瞄准语音图灵测试**：WaveForms AI 宣布成立，旨在开发具有情感智能能力的 AI。该公司的使命包括攻克语音图灵测试（Speech Turing Test），以改善音频应用中类人化的交互。
   - 这一新项目反映了将先进情感分析集成到 AI 系统中的日益增长的趋势。
- **来自多伦多的 NeurIPS 与会者**：随着 NeurIPS 的开幕，聊天中出现了关于从包括多伦多在内的各地飞往会场的与会者的讨论。这突显了该会议在聚集 AI 专业人士和爱好者进行社交和分享前沿研究方面的重要性。
   - 参加活动的与会者对 AI 技术不断进步的兴奋之情溢于言表。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://ollama.com/blog/structured-outputs">Structured outputs · Ollama Blog</a>: Ollama 现在支持结构化输出（structured outputs），可以将模型的输出限制为由 JSON schema 定义的特定格式。Ollama Python 和 JavaScript 库已更新...</li><li><a href="https://x.com/alex_conneau/status/1866127388373098607">Alexis Conneau (@alex_conneau) 的推文</a>: 激动地宣布创立 WaveForms AI (http://waveforms.ai) —— 一家 Audio LLM 公司，旨在解决语音图灵测试（Speech Turing Test）并为 AI 带来情感智能 @WaveFormsAI</li><li><a href="https://x.com/btibor91/status/1865109134066274444">Tibor Blaho (@btibor91) 的推文</a>: 我在今天的 "12 Days of OpenAI: Day 2" 直播中注意到，OpenAI 平台的侧边栏出现了一个新图标，可能与即将发布的公告之一有关 —— "Custom Voices"...</li><li><a href="https://x.com/scaling01/status/1865492664994468284?s=46">Lisan al Gaib (@scaling01) 的推文</a>: 刚刚用 Gemini Experimental 1206 跑完了 @AIExplainedYT 的 Simple Bench 公开评估集，得分 4/10</li><li><a href="https://x.com/willdepue/status/1866184364859461988?s=46">will depue (@willdepue) 的推文</a>: Sora 今天向所有 ChatGPT Pro 和 Plus 用户开放！为了实现这一目标付出了巨大的努力 + 我认为这个产品非常有趣且直观。我最喜欢做的事情是生成虚假的历史...</li><li><a href="https://x.com/chrisparkx/status/1865406193776074965?s=46">Chris Park (@chrisparkX) 的推文</a>: xAI 不需要等到周一。这个团队太强了，一直在发布。恭喜 @xai 发布了全新的图像生成模型 —— Aurora！Grok 2 + Aurora 现在已在你的 X 应用中可用...</li><li><a href="https://x.com/altryne/status/1865783380370977024?s=46">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 我收回我之前说的关于其他视频模型能赶上 Sora 的所有话。泄露的 Sora v2 视频显示了 1 分钟生成的 txt2vid, img2vid, vid2vid, txt+vid2vid。https://x.com/...</li><li><a href="https://www.youtube.com/playlist?list=PLOXw6I10VTv8q5PPOsuECYDFqohnJqbYB">Sora 教程</a>: 未找到描述</li><li><a href="https://x.com/jerber888/status/1865112099015291379?s=46">Jeremy Berman (@jerber888) 的推文</a>: 我刚刚使用 Claude Sonnet 3.5 和 Evolutionary Test-time Compute 在公开的 ARC-AGI 基准测试中获得了第一名。引用 ARC Prize (@arcprize) 2024 ARC-AGI-Pub SoTA! 👾53.6% @jerber888 47.5% MARA(BARC)...</li><li><a href="https://x.com/JeffDean/status/1865081640546156993">Jeff Dean (@🏡) (@JeffDean) 的推文</a>: 庆祝 Gemini 取得令人难以置信的一周年进展的绝佳方式 —— 在总排名以及困难提示词、编程、数学、指令遵循等方面全面夺冠🥇，包括样式控制...</li><li><a href="https://x.com/OfficialLoganK/status/1865081419015352689">Logan Kilpatrick (@OfficialLoganK) 的推文</a>: Gemini-exp-1206，我们最新的 Gemini 迭代版本（具有完整的 2M token 上下文及更多功能），现在可以在 Google AI Studio 和 Gemini API 中免费使用。我希望你们喜欢这一年的...</li><li><a href="https://x.com/scaling01/status/1865221955202252938?s=46">Lisan al Gaib (@scaling01) 的推文</a>: 今天 Gemini 2.0 在 LMSYS 上碾压了所有人：- 在数学和编程上击败了 o1 ??? - 即使开启样式控制（Style Control）也轻松击败了 Claude 3.5 ??? 与此同时：- Meta: "新的 LLaMa3.3-70B 模型冲冲冲，你们这些家伙..."</li><li><a href="https://x.com/scaling01/status/1865088711609770417">Lisan al Gaib (@scaling01) 的推文</a>: 天哪，Google 做到了。指令遵循（Instruction Following）+ 样式控制（Style Control）</li><li><a href="https://llm-stats.com">LLM-Stats.com</a>: 关于大语言模型的统计数据和见解</li><li><a href="https://x.com/scaling01/status/1865086810214289910?s=46">Lisan al Gaib (@scaling01) 的推文</a>: Google 已经亮牌了吗？Gemini-Exp-1114 是 Gemini 2.0 Flash，Gemini-Exp-1121 是 Gemini 2.0 Pro，Gemini-Exp-1206 是 Gemini 2.0 Ultra。也有可能这些只是训练过程中的...</li><li><a href="https://x.com/paulgauthier/status/1865167742850208203">Paul Gauthier (@paulgauthier) 的推文</a>: 新的 Gemini-exp-1206 在 Aider 的代码编辑基准测试中得分 69%。这是 Gemini 家族的纪录。https://aider.chat/docs/leaderboards/</li><li><a href="https://jeremyberman.substack.com/p/how-i-got-a-record-536-on-arc-agi">我如何使用 Sonnet 3.5 配合 Evolutionary Test-time Compute 在 ARC-AGI-Pub 中获得第一名</a>: 在 Params 上查看我的代码：https://params.com/@jeremy-berman/arc-agi</li><li><a href="https://x.com/ruudnl/status/1865425438991945938?s=46">Ruud van der Linden (@RuudNL) 的推文</a>: Sora v2 即将发布：* 1 分钟视频输出 * text-to-video * text+image-to-video * text+video-to-video。OpenAI 的 Chad Nelson 在伦敦的 C21Media 主旨演讲中展示了这些。他说我们将...</li><li><a href="https://x.com/levels_">

<li><a href="https://x.com/levelsio/status/1865899245850517706?s=46">来自 @levelsio (@levelsio) 的推文</a>：粗略一看，Grok 的新图像模型 Aurora 在生成人物照片方面的细节似乎比 Flux 更丰富。令人惊叹的是，他们竟然能创造出一个全新的图像模型...</li><li><a href="https://x.com/voooooogel/status/1865189744776507809?s=46">来自 thebes (@voooooogel) 的推文</a>：llama-3.3-70b 正确猜出了采样约束（仅允许使用圣经中的词汇）。引用 thebes (@voooooogel)：我为 llama-3.1-8b 编写了一个自定义 LLM 采样器，使其只能说...</li><li><a href="https://x.com/smokeawayyy/status/1865319093274108405?s=46">来自 Smoke-away (@SmokeAwayyy) 的推文</a>：ChatGPT o1 中对话式 AGI 的自定义指令。书签、使用并修改这些自定义指令，以创建你自己的人类级别 AI 伴侣。尽情享受。---“你是一个支持性的人类化伴侣...”</li><li><a href="https://x.com/sama/status/1866187525821538436?s=46&t=b7l37rB6wtbyAh6ah1NpZQ">来自 Sam Altman (@sama) 的推文</a>：我们今天发布了 Sora，并为此制作了一个新产品。如果你有 OpenAI Plus 或 Pro 账户，就可以生成视频。任何人都可以观看。它需要一些时间来逐步推出，但到...</li><li><a href="https://x.com/shaunralston/status/1865116666440675647?s=46">来自 Shaun Ralston (@shaunralston) 的推文</a>：不要错过带有视觉功能的 ChatGPT Advanced Voice Mode，本周日晚将在 @60Minutes（@CBS 和 @paramountplus）节目中亮相，很快就会登陆你的智能手机。</li><li><a href="https://countless.dev/">Countless.dev | AI 模型比较</a>：轻松比较 AI 模型！所有供应商一站式汇总。</li><li><a href="https://x.com/tetumemo/status/1865125990483267896">来自 テツメモ｜AI図解×検証｜Newsletter (@tetumemo) 的推文</a>：📝 诶？真的可以吗？刚刚在 Google 发布的 Gemini-exp-1206，已经在 Cursor 中配置完成并可以“免费”使用了！！这款超越 o1-preview 和 mini、综合排名第一的模型，任何人都可以通过 Google AI Studio 或 API “免费”使用，这真的很棒。设置方法见评论区。引用 Logan Kilpatrick (@OfficialLoganK) Gemi...</li><li><a href="https://www.youtube.com/live/2jKVx2vyZOY">Sora–OpenAI 的 12 天：第 3 天</a>：Sam Altman, Aditya Ramesh, Bill Peebles, Rohan Sahai 和 Joey Flynn 向世界发布 Sora。</li><li><a href="https://www.youtube.com/watch?v=YpFaPKOeNME">神经网络真的很奇怪...</a>：Neel Nanda 是 Google DeepMind 的高级研究科学家，领导其机械解释性（mechanistic interpretability）团队。在这场深度访谈中，他讨论了他的工作...</li><li><a href="https://www.youtube.com/watch?v=OY2x0TyKzIQ">这个视频是 AI 生成的！SORA 评测</a>：SORA 生成视频。这是首个评测。在 https://ridge.com/MKBHD 获取最后时刻礼物的最高 40% 折扣。真实的观鸟视频：https://youtu.be/F...</li><li><a href="https://x.com/MKBHD/status/1866152437838393797">来自 Marques Brownlee (@MKBHD) 的推文</a>：传闻是真的——OpenAI 的 AI 视频生成器 SORA 今天向公众发布...我已经使用它大约一周了，并对其进行了评测：https://youtu.be/OY2x0TyKzIQ...
</li>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1314698091844206669)** (136 条消息🔥🔥): 

> `NeurIPS 准备工作，会议社交，Tabular Data 在工业界的作用，Paper Club 活动，会议沟通工具` 


- **通过 Paper Club 为 NeurIPS 做好准备**：本周日加入我们的 **Latent Space Paper Club**，共同探讨 NeurIPS 前夕的重要论文和见解，体验高效产出与社区氛围的融合。
   - 活动包括 [论文讨论与灵感碰撞 (Idea Jams)](https://lu.ma/25mwbwcm) 以及百乐餐 (potluck) 晚餐，方便与来自各个 AI 社区的朋友交流。
- **社交技巧：走廊交流 (Hallway Track) 是关键**：有人建议，会议中的 **走廊交流 (hallway track)** 是产生最有价值对话的地方，能建立极佳的人脉。
   - 与会者注意到名片正变得过时，他们更倾向于交换 Twitter 账号并利用会议 App 进行社交。
- **Tabular Data 在工业界的重要性**：讨论强调了 **tabular data** 在时间序列预测和预防性维护等工业应用中依然具有极高的相关性。
   - 参与者呼吁不要低估表格数据集的价值，强调了它们在实际 AI 落地中的重要性。
- **参会者之间的沟通工具**：讨论了 **WeChat**、Twitter 账号以及会议 App 上的私信等多种保持联系的沟通方式。
   - 与会者表示需要有效的渠道来维持会后的联系，一些人推荐使用聊天聚合器。
- **加入线下 (IRL) Paper Club**：大家对加入线下 Paper Club 感到兴奋，认为这是一种更有效地参与学术讨论的方式。
   - 成员们提到活动审批过程很快，并对社区的支持表示感谢。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://visitingmedia.com/tt8/?ttid=vancouver-convention-centre#/3d-model/0/2?b11t0=1&tlzqx7l=535249">TrueTour® 是亲临现场之外的最佳选择！亲自体验吧！</a>：点击此处以全新方式查看这个神奇的地方。TrueTour® 为您提供沉浸式虚拟体验，可通过网络分享。</li><li><a href="https://lu.ma/25mwbwcm">NeurIPS 赛前准备与节日百乐餐 · Luma</a>：论文无穷，时间有限——让我们一起为 NeurIPS 做准备！📚✨ 随着重大的一周即将到来，从独自钻研论文中休息一下，加入……</li><li><a href="https://lu.ma/LSLIVE">Latent Space LIVE! at NeurIPS 2024 · Luma</a>：让我们聚在一起，用 NeurIPS 期间举办的首场直播 Latent Space Paper Club 来告别 2024！我们不会像 NeurIPS 那样逐篇讲解论文，而是……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1314710403111981097)** (21 条消息🔥): 

> `Llama 3.3 权重发布、开放式信息存储挑战、文字冒险游戏连贯性问题、Eleuther Eval Harness 修改、JAX/Flax 模型集成` 


- **Llama 3.3 权重发布**：一位成员在 [Hugging Face](https://huggingface.co/unsloth/Llama-3.3-70B-Instruct) 上为那些等待审批的人上传了 **Llama 3.3 70B Instruct 的 16bit 权重**，并提供了多种格式的访问。
   - 他们还提到了 [Llama 3.3 所有版本](https://huggingface.co/collections/unsloth/llama-33-all-versions-67535d7d994794b9d7cf5e9f)的集合，包括 GGUF 和 4-bit 格式。
- **存储开放式信息的挑战**：一位成员表达了对 **RAG** 在存储和检索开放式信息时，相比于之前的 **Knowledge Graphs** 所具有的不可预测性的担忧。
   - 他们强调在处理大量专有信息的 **Agent Memory** 和 **Question/Answer Systems** 中，需要更可靠的方法。
- **文字冒险游戏的连贯性问题**：一位成员报告称，在达到对话长度限制后，文字冒险游戏难以维持**连贯性和一致性**（Continuity and Coherency），从而影响了对角色的情感投入。
   - 他们寻求关于哪些 **LLM** 能够更好地支持持续的文字冒险叙事且不丢失 **Context** 的建议。
- **修改 Eleuther Eval Harness Prompts**：一位成员请求关于修改 **Eleuther Eval Harness** 的 **Prompts** 的指导，并表示缺乏可用的文档。
   - 另一位成员建议查看 [Interface Documentation](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage) 作为起点。
- **将 JAX/Flax 模型集成到 Evaluation Harness**：一位成员询问了关于将 **lm evaluation harness** 适配到 **JAX/Flax Models** 的进展，因为他们在连接自己的模型时遇到了困难。
   - 他们被引导至相关示例和实现建议，并得到了未来更新的承诺，包括一个潜在的 **Draft PR**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://linktr.ee/digitalgnosis">@nathormond | Instagram, Facebook | Linktree</a>: 查看 digitalgnosis 的 Linktree。在此处 YouTube、Spotify 上收听他们的音乐。</li><li><a href="https://www.youtube.com/watch?v=139UPjoq7Kw">Building Machine Learning Systems for a Trillion Trillion Floating Point Operations</a>: 在过去的 10 年里，我们看到 Machine Learning 吞噬了一切，从科技行业到诺贝尔奖，甚至连 ML 这个缩写也不例外。这种崛起...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#external-library-usage">lm-evaluation-harness/docs/interface.md at main · EleutherAI/lm-evaluation-harness</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1314774004787314788)** (81 条消息🔥🔥): 

> `不同模态下的 Variational Encoders、Memory-efficient Optimizers、3D Generation Frameworks、训练中的 Catastrophic Forgetting、Adam 与 SGD 的性能对比`

- **跨模态变分编码器（Variational Encoders）的探索**：研究兴趣集中在将一种模态转换为另一种模态的变分编码器上，特别是避免使用需要对主模态进行自动编码的 cVAE。
   - *Multimodal VAEs* 被建议作为解决这些探索性研究的通用课题。
- **大语言模型（Large Language Models）的显存高效优化器**：一篇论文讨论了 AdamW 等显存密集型优化器在训练大语言模型时面临的挑战，并提出了一种新的显存高效优化器 APOLLO 来解决这些问题。
   - 文中指出，AdamW 沉重的显存负担导致了昂贵的计算开销，而更好的替代方案可以在不显著损失性能的情况下优化显存使用。
- **创新的 3D 生成框架**：最近的两篇论文介绍了 3D 资产创建的方法，利用结构化的潜变量表示（latent representations）来改进输出格式，并利用深度学习模型进行特征集成。
   - 每种方法都展示了从各种输入生成高质量 3D 结果的优势，同时保持了结构和纹理的完整性。
- **优化器性能中的灾难性遗忘（Catastrophic Forgetting）**：讨论围绕使用 AdamW 和 Muon 等不同优化器训练的模型在新的数据集上进行微调时，灾难性遗忘表现出的差异展开。
   - 参与者担心更好的初始拟合可能会加剧遗忘，这表明在切换数据集时需要采取策略来减轻性能损失。
- **性能对比：Adam vs SGD**：参与者注意到 Adam 在语言任务中的表现通常优于 SGD，这可能是由于此类数据集中存在的重尾类别不平衡（heavy-tailed class imbalance）导致的。
   - 据报道，这种不平衡导致梯度下降（gradient descent）的平均损失下降速度比 Adam 慢，而 Adam 对低频词的敏感度较低。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2310.14963">Studying K-FAC Heuristics by Viewing Adam through a Second-Order Lens</a>：深度学习优化研究的特点是：一阶基于梯度的方法（如 SGD 和 Adam）的计算效率与二阶方法的理论效率之间存在张力...</li><li><a href="https://arxiv.org/abs/2412.01506">Structured 3D Latents for Scalable and Versatile 3D Generation</a>：我们介绍了一种用于多功能且高质量 3D 资产创作的新型 3D 生成方法。其核心是统一的结构化潜变量（Structured LATent, SLAT）表示，它允许解码到不同的输出...</li><li><a href="https://arxiv.org/abs/2402.19449">Heavy-Tailed Class Imbalance and Why Adam Outperforms Gradient Descent on Language Models</a>：Adam 已被证明在大型语言模型上的表现优于梯度下降，且差距比在其他任务上更大，但原因尚不明确。我们展示了这种性能差距的一个关键因素是重尾...</li><li><a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>：大型语言模型 (LLMs) 在训练期间的内存消耗极高是众所周知的，特别是在使用流行的 AdamW 优化器时。这种内存负担使得必须使用更多或更高端的 GPU，或者减少...</li><li><a href="https://arxiv.org/abs/2412.04431">Infinity: Scaling Bitwise AutoRegressive Modeling for High-Resolution Image Synthesis</a>：我们展示了 Infinity，这是一种能够根据语言指令生成高分辨率、逼真图像的 Bitwise 视觉自回归建模。Infinity 重新定义了视觉自回归模式...</li><li><a href="https://arxiv.org/abs/2411.08033">GaussianAnything: Interactive Point Cloud Latent Diffusion for 3D Generation</a>：虽然 3D 内容生成已取得显著进展，但现有方法在输入格式、潜空间设计和输出表示方面仍面临挑战。本文介绍了一种新型 3D 生成...</li><li><a href="https://arxiv.org/abs/2403.03100">NaturalSpeech 3: Zero-Shot Speech Synthesis with Factorized Codec and Diffusion Models</a>：虽然最近的大规模文本转语音 (TTS) 模型取得了显著进展，但在语音质量、相似度和韵律方面仍有不足。考虑到语音复杂地包含各种...</li><li><a href="https://arxiv.org/abs/2211.09407">NANSY++: Unified Voice Synthesis with Neural Analysis and Synthesis</a>：尽管语音合成的各种应用共同生成“语音”作为输出，但它们一直是独立开发的。此外，大多数语音合成模型仍然...</li><li><a href="https://x.com/liron/status/1865974752822919202/photo/1">Tweet from Liron Shapira (@liron)</a>：妹子来我家吧，我得让你看看我的沙发</li><li><a href="https://www.nature.com/articles/s43588-024-00732-2">A scalable framework for learning the geometry-dependent solution operators of partial differential equations - Nature Computational Science</a>：这项工作提出了一个人工智能框架，用于学习偏微分方程 (PDEs) 的几何相关解算子。该框架实现了可扩展且快速的近似...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1314982561381875884)** (2 messages): 

> `GitHub Gist 代码共享，Scaling laws 概述` 


- **Gist 通过 get_scaling_laws 简化代码共享**：一位成员分享了一个名为 **get_scaling_laws** 的 [GitHub Gist](https://gist.github.com/elyxlz/33122704a751051b0d675ec0e10b8af6)，它可以即时分享代码片段和笔记。
   - 在 Gist 分享后，另一位成员表达了感谢：*非常感谢！*
- **分享的 Gist 的视觉概述**：分享的 Gist 包含一张图片以便更好地理解，可在 **https://github.githubassets.com/assets/gist-og-image-54fd7dc0713e.png** 查看。
   - 这种视觉辅助有助于快速掌握 **get_scaling_laws** Gist 的目的和功能。



**提及链接**：<a href="https://gist.github.com/elyxlz/33122704a751051b0d675ec0e10b8af6">get_scaling_laws</a>：GitHub Gist：即时分享代码、笔记和片段。

  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1314720014129037322)** (11 条消息🔥): 

> `Gradient Routing, Neural Network Specialization, Causal Inference and Gradient Routing, Credible Source Distinction, Interpretable Architecture` 


- **探索用于安全性的 Gradient Routing**：一种名为 **Gradient Routing** 的新方法允许用户根据数据类型决定更新哪些参数，从而促进 Neural Networks 的专业化（[来源](https://x.com/Turn_Trout/status/1865156788750028846)）。该方法旨在解决与 AI 训练的黑盒性质相关的安全性问题。
   - *Gradient Routing* 为理解和控制 AI 行为提供了一种潜在的替代方案，超越了传统的神经配置。
- **类脑学习行为的考量**：成员们讨论了 **Gradient Routing** 与大脑功能之间的相似之处，认为这种机制反映了 **Localized Learning** 的生物学特性（例如，选择性神经元权重）。这引发了关于如何利用这种方法有效编码概念的问题。
   - 收集到的见解强调了 *Localizing Learning* 的潜力，同时也讨论了其在人类解释研究之外的重要性。
- **Causal Inference 应用的潜力**：一位成员凭直觉认为 **Gradient Routing** 可以通过根据 Intervention Variables 调整 Loss Gradients 来辅助 **Causal Inference**。这表明了一种考虑到各种干预措施的有针对性的学习方法。
   - 虽然确切的机制仍处于推测阶段，但这与增强模型 Causal Reasoning 鲁棒性的讨论是一致的。
- **AI 输入中的来源可信度**：一位参与者建议 **Gradient Routing** 可以使模型能够区分**可信来源**和**不可信来源**。这种分类法可以改善元数据对模型行为的影响，而不会使区分过程过度复杂化。
   - 对话中包含了对 **Hallucination vs. Generalization**（幻觉与泛化）的担忧，指出了构建可靠 AI 系统的复杂性。
- **Interpretable Architecture 议程**：讨论暗示了 **Gradient Routing** 在 AI 的 **Interpretable Architecture 议程**中的潜在效用。明确组件定位（Component Localization）的作用可能会促进更有效的架构策略。
   - 社区似乎对结构化学习方法在理解 AI 系统及其输出方面所能做出的贡献充满热情。



**提及的链接**：<a href="https://x.com/Turn_Trout/status/1865156788750028846">来自 Alex Turner (@Turn_Trout) 的推文</a>：1) AI 被作为黑盒进行训练，这使得理解或控制它们的行为变得困难。这对安全性不利！但有什么替代方案吗？我们的想法：通过配置将结构训练到 Neural Network 中...

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1314909852191166465)** (57 messages🔥🔥): 

> `MLX Examples PR, Eleuther AI Eval Harness, GSM8K Comparison Issues, ARC-Challenge Dataset Anomalies, Llama Model Evaluation Techniques` 


- **MLX 使用新 CLI 工具进行评估**：最近的 [Pull Request #1140](https://github.com/ml-explore/mlx-examples/pull/1140) 添加了一个 `mlx_lm.evaluate` CLI，能够为任何兼容 mlx-lm 的模型使用 `lm-eval`，从而实现对 `Qwen2.5-7B-Instruct` 等模型的评估任务。
   - 通过此更新，用户可以轻松进行评估，例如：`mlx_lm.evaluate --model mlx-community/Qwen2.5-7B-Instruct-4bit --tasks winogrande arc_easy`。
- **GSM8K 在评估中显示出差异**：用户在尝试复现 GSM8K 的高准确率得分时遇到困难，报告的指标显示性能明显低于 LiquidAI 等对比模型。
   - 尽管切换了不同的评估方法，获得的最高分约为 **72.93%**，仍低于之前评估声称的 **79.6%**。
- **ARC-Challenge 缺失选项问题**：据报告，ARC-Challenge 数据集中的一个问题只有三个选项，当引用第四个选项时会导致评估错误。
   - 鼓励用户调整其配置以更好地处理此类异常，并确保评估的准确性。
- **Eleuther AI Eval Harness 配置**：提供了一个在 EleutherAI 的 eval harness 上评估 ARC-Challenge 的配置，旨在简化跨模型的性能比较。
   - 建议用户在实施给定的 YAML 配置的同时，更新实用函数以更好地处理数据集。
- **模型评估中的社区协作**：社区正在积极协作，确保他们的模型（如 RWKV）以一致的方式进行评估，以避免对性能指标的误解。
   - 讨论强调了关于评估方法多变性的共同担忧，以及发布结果透明度的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/meta-llama/llama3/blob/main/eval_details.md">llama3/eval_details.md at main · meta-llama/llama3</a>: Meta Llama 3 官方 GitHub 站点。通过在 GitHub 上创建账户为 meta-llama/llama3 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1902">mlx Model (loglikelihood &amp; generate_until) by chimezie · Pull Request #1902 · EleutherAI/lm-evaluation-harness</a>: 这为 mlx 模型添加了一种新的模型类型。特别是，它实现了 loglikelihood 和 generate_until 接口。适用于当前版本的 mlx 和 mlx-lm。新的模型类型是 ml...</li><li><a href="https://github.com/ml-explore/mlx-examples/pull/1140">`mlx_lm.evaluate` by barronalex · Pull Request #1140 · ml-explore/mlx-examples</a>: 添加一个使用 lm-eval 并支持任何 mlx-lm 兼容模型的 mlx_lm.evaluate CLI。例如：mlx_lm.evaluate --model mlx-community/Qwen2.5-7B-Instruct-4bit --tasks winogrande arc_easy 结果...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1314864780246515723)** (8 messages🔥): 

> `VLM Training Process, Causal Loss in VLMs, MSE on Visual Tokens, Apple AIM` 


- **关于 VLM 训练细节的查询**：一位用户询问了 **VLM**（如 **Qwen2-VL**）的训练过程，特别是 **causal loss** 是如何应用的，以及它是否影响 visual tokens。
   - 他们质疑是否从 loss 中剔除了**浅紫色 token**，以及应用 **MSE loss** 是否能增强多模态特征的学习。
- **确认 MSE 的应用**：另一位用户确认 **MSE** 已应用于 visual tokens，并针对讨论表示：“这两个问题都是肯定的”。
   - 他们提到最近确实有人尝试使用 MSE，尽管不记得具体是谁。
- **寻找支持性论文**：原用户询问是否有关于 MSE 尝试的**论文**，以及是否产生了任何改进的结果。
   - 回复者澄清说这并非专门针对 **VLM**，并表示会查找更多细节。
- **引用 Apple AIM**：用户指出 **Apple AIM** 是他们之前提到的 **MSE** 尝试的参考，表明其与讨论的相关性。
   - 这一参考可能会为正在进行的关于 MSE 在 visual token 处理中应用的查询提供信息。


  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/)** (1 messages): 

karatsubabutslower: CC <@367104793292046338> 关于这个有什么提示吗？
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1314686472787591169)** (28 条消息🔥): 

> `Podcast 长度, NotebookLM 使用案例, 交互式叙事, Sheets 中的数据处理, NotebookLM Podcast 提示词` 


- **使用 NotebookLM 实现的 Podcast 长度**：成员们分享了从 NotebookLM 获取不同长度 Podcast 的经验，其中一人将 **107 页** 的 Formula 1 规则压缩成了 **17 分钟** 的 Podcast。
   - 另一位成员指出，将 YouTube 视频和便签（scratchpad）结合使用，生成的 Podcast 甚至比原始视频还要长。
- **探索 NotebookLM 使用案例**：讨论重点介绍了尝试将 **Claude** 或 **ChatGPT** 与 NotebookLM 连接的尝试，并建议将 **Zapier** 作为潜在的解决方案。
   - 成员们还反思了如何通过输入歌词和其他资源，利用 NotebookLM 为歌曲创建背景信息。
- **交互式叙事开发**：用户讨论了 NotebookLM 如何处理故事，其中一人注意到它具有基于输入数据进行 **世界观构建（world building）** 的能力。
   - 另一位用户确认它有时会生成意料之外的故事节奏，引发了对其创作过程的疑问。
- **Google Sheets 中的数据处理**：一位用户分享了将数据从 Google Sheets 有效传输到 Docs 的技巧，强调了 **清晰的标题和标签** 的必要性。
   - 他们指出了不规范电子表格带来的挑战，并表示数值的传输效果优于复杂的方程式。
- **利用 NotebookLM Podcast 提示词**：分享了一个教程视频，承诺揭示用于优化 NotebookLM 进行 Podcast 创作的 **10 个秘密提示词（prompts）**。
   - 该视频旨在帮助希望增强其 Podcast 内容的用户，强调了获得更好输出的独特技术。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/490674b4-ee93-47e1-a297-00070f841595/audio">未找到标题</a>: 未找到描述</li><li><a href="https://youtu.be/aG0ixD3OY80">NotebookLM Podcast 教程：10 个秘密提示词（别人会为此眼红！）</a>: 免费获取这些独家的 NotebookLM Podcast 提示词！我花了几个小时完善这 10 种独特的方法，以帮助 The AI News 社区脱颖而出。只需观看...</li><li><a href="https://youtu.be/9CN1Ymyrhyo?si=n2PpH1J4PQgrvuvH)">特朗普的新任 NASA 局长 // Artemis 计划再次推迟 // 金星没有海洋</a>: 🎁 赠送 Universe Today Patreon 会员资格：https://www.patreon.com/universetoday/gift 特朗普宣布了他对新任 NASA 局长的人选，我们有一个新的...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1314691474805362795)** (141 条消息🔥🔥): 

> `NotebookLM 限制, NotebookLM 语言支持, Podcast 功能对比, Audio Overview 问题, 使用 NotebookLM 进行学习` 


- **NotebookLM 对文档上传有限制**：用户注意到，单个笔记本中最多可上传 **100 个文档**，但对可以创建的笔记本数量没有限制。
   - 一些用户对该限制是否从之前的 **50 个文档** 发生了变化表示困惑。
- **语言支持方面的挑战**：许多用户在以不同语言使用 NotebookLM 时遇到困难，通常需要 **退出并重新登录** 才能切换语言。
   - *似乎 NotebookLM 不支持即时语言切换*，这让那些希望有更灵活方式的用户感到沮丧。
- **Podcast 功能对比**：讨论包括了 NotebookLM 的 Podcast 功能与 **ElevenLabs** 提供的功能之间的对比，突显了竞争格局。
   - 有提到 NotebookLM 缺乏明确的 API 和系统化的 Prompting 能力，而这些能力本可以增强其在创建 Podcast 时的可用性。
- **Audio Overview 的问题**：用户报告称，Podcast 和 Audio Overview 功能有时会根据提供的来源生成错误或无关的内容。
   - 一些用户建议删除有问题的音频输出并重新生成，作为解决事实生成错误的方法。
- **将 NotebookLM 用于学术目的**：一些用户正将 NotebookLM 用于教育目的，创建学习指南和笔记，但在自定义输出方面面临挑战。
   - 分享了一些指南和资源以帮助用户最大化 NotebookLM 的生产力，包括用于改进使用的教程视频链接。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/14278184?hl=en&sjid=17768370428841200951-NA">常见问题解答 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://open.substack.com/pub/brightvoid/p/dancing-with-the-djinn?utm_source=share&utm_medium=android&r=9euw0">与 Djinn 共舞</a>：在页面上与 AI 思维协作</li><li><a href="https://youtu.be/QxbmQs3b_DE">NotebookLM 教程，让你的生产力提升 10 倍</a>：想要精通 NotebookLM 并将生产力提高 10 倍，请观看此完整视频。我在一个视频中涵盖了从基础到高级的所有内容，并包含 2 个真实场景...</li><li><a href="https://open.spotify.com/show/3gaQyAwwFAFXzGb9DYMWSS">Machine Logic</a>：Podcast · Studio Il · 欢迎来到 Machine Logic，这是一个由人工智能接管麦克风的播客——字面意思！完全由 AI 主持，我们深入探讨前沿技术、AI 等有趣话题...</li><li><a href="https://youtu.be/aG0ixD3OY80">NotebookLM Podcast 教程：10 个秘密 Prompt（好用到爆！）</a>：免费获取这些独家的 NotebookLM Podcast Prompt！我花了数小时完善这 10 种独特的方法，以帮助 AI News 社区脱颖而出。只需观看...
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1314698056666714192)** (65 条消息🔥🔥): 

> `Unsloth 微调框架, 构建聊天模型, 商业中的 AI, 语音生成中的情感表达, 繁体中文 AI 训练` 


- **Unsloth 增强微调过程**：一位成员分享了关于 **Unsloth 微调框架** 的见解，以及在训练过程中集成自定义评分函数的功能。
   - 这为针对微调任务量身定制的改进型**评估循环 (evaluation loops)** 开启了创新的可能性。
- **渴望构建聊天模型**：一位新人表达了创建自己的聊天模型的目标，特别是通过实现围绕**产品信息和用户评论**的功能。
   - 讨论了*使用现有的社交媒体数据*进行评论的可能性，但对其法律影响以及使用 AI 进行爬虫任务的必要性存在担忧。
- **无需求集成 AI 的挑战**：关于 AI 实施必要性的辩论展开，强调了对 AI 合理**使用场景 (use cases)** 的需求。
   - 成员们强调，AI 不应仅仅为了应用而应用，而应专注于解决各个垂直领域中的**现实问题**。
- **探索语音生成中的情感表达**：围绕**语音生成中的情感表达**的讨论揭示了对开发自定义声音风格 API 的兴趣。
   - 一位成员确认正在运行自己专注于**语音情感化**的 API，并表示对 **GPT4o-voice 风格**感兴趣。
- **推进繁体中文 AI 模型**：一位用户介绍了自己以及他们在训练**繁体中文** AI 模型方面的工作，分享了他们对 **Project TAME** 的贡献。
   - 他们最近的项目包括创建 **Llama-3-Taiwan-8B-Instruct** 模型，该模型已在 Hugging Face 上发布。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1315496863276732508)** (59 条消息🔥🔥): 

> `量化 aya-expense 模型, LLM 部署选项, 基于向量的检索方法, RAG 中的多步工具使用, AI 研究中的社区参与` 


- **请求 aya-expense 模型量化帮助**：一位用户表示有兴趣将 **aya-expense 模型**量化为 AWW 或 FP8 格式，以便在有限的 GPU 资源上更好地运行，并建议使用训练数据进行校准。
   - 另一位成员分享说，他们发现 **8b 模型**很容易运行，大小减小到了 **3.4GB**。
- **关于 LLM 部署选项的讨论**：成员们交流了使用 **vLLM** 进行部署的见解，其中一位指出 GGUF 格式现在已与其兼容，使其更易于使用。
   - 另一位成员强调 **Ollama** 比 **llama.cpp** 更容易配置，但后者可能提供性能优势。
- **探索基于向量的检索方法**：一位新成员加入并讨论了他们对**检索方法**的研究，包括基于向量的检索和稠密段落检索 (Dense Passage Retrieval)，并考虑进行对比研究。
   - 社区成员提供了积极的反馈，鼓励这一想法并建议增加诸如**多步工具使用 (multi-step tool use)** 等增强功能。
- **多步工具使用详解**：针对提问，一位社区成员详细阐述了**多步工具使用**，将其等同于 Agent 多次调用工具以获得增强的结果。
   - 该方法旨在自动优化查询并分析结果，辅助高级研究能力。
- **怀念社区演示**：一位成员回忆起过去在 Discord 上举行的社区演示和展示活动，成员们在活动中展示自己的作品以促进参与。
   - 这突显了社区的协作性质，并鼓励在 AI 研究领域分享进展和见解。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/library/aya">aya</a>: Aya 23 由 Cohere 发布，是一个支持 23 种语言的全新最先进多语言模型系列。 </li><li><a href="https://github.com/cohere-ai/notebooks/blob/main/notebooks/agents/Vanilla_Multi_Step_Tool_Use.ipynb">notebooks/notebooks/agents/Vanilla_Multi_Step_Tool_Use.ipynb at main · cohere-ai/notebooks</a>: Cohere 平台的代码示例和 Jupyter Notebooks - cohere-ai/notebooks</li><li><a href="https://huggingface.co/collections/CohereForAI/aya-datasets-660415741bd4852f01c81c77">Aya Datasets - a CohereForAI Collection</a>: 未找到描述</li><li><a href="https://cohere.com/research">Cohere For AI (C4AI)</a>: Cohere For AI (C4AI) 是 Cohere 的研究实验室，致力于解决复杂的机器学习问题。 
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1315713517512294441)** (17 messages🔥): 

> `数据集上传问题、文件格式错误、绝对路径建议、Cohere Dashboard 上传、示例文件协助` 


- **Rerank 数据集上传失败**：用户报告称，尽管遵循了文档中的代码示例，但在上传 Rerank 数据集时仍遇到问题。他们尝试修正代码，但遇到了显示为 **0 字节大小** 的加载问题。
   - 建议检查文件格式并使用绝对路径来解决上传问题。
- **文件格式困扰**：在尝试上传后，用户收到错误提示：*'avro: string is unsupported for avro array'*，表明存在格式问题。建议用户确保其数据集符合预期的结构。
   - 用户计划在修正格式后重试，并感谢他人的协助。
- **上传替代方案与指南**：一名成员建议尝试直接在 [Cohere Dashboard](https://dashboard.cohere.com/fine-tuning/create?endpoint=rerank) 上上传数据集，以确认数据的格式。如果用户遇到进一步问题，他们愿意提供帮助。
   - 推荐了一个迷你指南链接和一张图片附件，用于确认正确的 JSONL 格式。
- **社区支持与协助**：成员们表示愿意互相帮助解决数据集上传问题。他们鼓励分享数据集或伪样本以便进一步排查故障。
- **会议后的后续跟进**：用户表示将因开会暂时离开，但计划稍后测试讨论的解决方案。他们对社区的支持表示感谢，营造了良好的协作氛围。



**提及的链接**：<a href="https://dashboard.cohere.com/fine-tuning/create?endpoint=rerank">Login | Cohere</a>：通过一个易于使用的 API 登录以访问先进的 LLM 和 NLP 工具。

  

---


### **Cohere ▷ #[cohere-toolkit](https://discord.com/channels/954421988141711382/1254901651081269268/1314687751618039928)** (6 messages): 

> `自我介绍消息、Cohere Toolkit 问题` 


- **Dominic 和 YiTechX 自我介绍**：成员 **Dominic** 和 **YiTechX** 互相打招呼并进行了自我介绍，在频道中建立了友好的基调。
   - 问候强调了社区属性，促进了讨论的积极氛围。
- **Tony 寻求关于 Cohere Toolkit 的解答**：Tony 对询问有关 **Cohere Toolkit** 的问题表示不确定，希望能获得指导。
   - 这引起了另一位成员的回应，邀请 Tony 自由提问，增强了小组的互助性质。


  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1315741980411691110)** (1 messages): 

> `Neurips 聚会` 


- **在 Neurips 加入我们！**：一位成员邀请大家去 **Neurips** 聚会，分享了对此次活动的兴奋之情。
   - 消息中附带了一张图片，可能捕捉到了聚会的氛围。
- **分享 Neurips 聚会的图片**：关于 **Neurips** 聚会的公告中附带了一张图片，可能作为宣传视觉资料。
   - 该视觉效果旨在吸引更多参与者加入活动。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1314688258025717861)** (125 条消息🔥🔥): 

> `a16z Crypto 创意, Nous Research 更新, AI x Crypto 讨论, AI 开发视频, DeMo 中的 DCT 与 Transpose` 


- **a16z 探索 AI 与 Crypto 的联系**：a16z 最近发布了一篇概述“2025 年 Big Crypto Ideas”的文章，其中链接到了 Nous 关于 AI 中 TEE 应用的聊天机器人讨论。
   - 虽然他们提到了与 Crypto 相关的 AI，但并未点名 Nous，这引发了成员们关于知名度的讨论。
- **Nous Research 仍是一家成长中的 AI 公司**：成员们确认 Nous Research 是 AI 领域的一个相对较新的参与者，拥有多个正在进行的项目和研究领域。
   - 分享了一个用于探索其工作的资源链接：[Nous Research Releases](https://nousresearch.com/releases/)。
- **发布的 AI 视频见解**：一位成员分享了一个强调 AI 开发中常见陷阱的视频，获得了观众的积极反馈和互动。
   - 该视频标题为“为什么 AI 开发比传统编程更有趣”，邀请观众在学习的同时获得娱乐。
- **关于 DeMo 实现的技术讨论**：一场关于将 DeMo 推广到 n 维权重的对话展开，涉及对过程中 Transpose（转置）影响的询问。
   - 澄清了 Transpose 有助于在特定维度上有效地应用 DCT，并分享了关于计算效率的见解。
- **成员关于 AI 技术的互动**：成员们参与了关于 AI 技术的各种讨论，包括硬件可用性和对 GPU 规格的偏好。
   - 对话还涉及了正在进行的教育追求以及具有专业技能的成员之间的合作机会。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.datacenterdynamics.com/en/news/openai-considers-changing-agi-provision-to-allow-further-microsoft-investment-genai/">OpenAI 考虑更改 AGI 条款以允许微软进一步投资</a>：随着其进一步偏离创始愿景</li><li><a href="https://x.com/nousresearch">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://a16zcrypto.com/posts/article/big-ideas-crypto-2025/">我们在 Crypto 领域感到兴奋的几件事 (2025) - a16z crypto</a>：2025 年 Crypto、区块链、Web3 趋势列表——包括 AI x Crypto、预测市场、稳定币、投票等等</li><li><a href="https://principlesandinterest.wordpress.com/2021/12/12/crypto-art-stocks-and-power-in-2021/">2021 年的 Crypto、艺术、股票与权力</a>：我的日常工作是领导一个团队，代表客户分析和投资金融市场资产。主要是股票和债券。投资组合经理倾向于寻求识别那些……的资产</li><li><a href="https://x.com/BanklessHQ/status/1864633317804404891?t=plOtrC7W4FVwRRayp-Xdng&s=19">来自 Bankless (@BanklessHQ) 的推文</a>：正在直播 -- AI ROLLUP #2 | Ejaaz Ahamadeen@cryptopunk7213 & @TrustlessState 报道 AI x Crypto 领域的最新动态，包括：- @virtuals_io 市值达到 10 亿，@AIXBT Agent 旗舰 - @0xzerebro + @...</li><li><a href="https://principlesandinterest.wordpress.com/2021/12/12/cryp">2021 年的 Crypto、艺术、股票与权力</a>：我的日常工作是领导一个团队，代表客户分析和投资金融市场资产。主要是股票和债券。投资组合经理倾向于寻求识别那些……的资产</li><li><a href="https://youtu.be/9jNXv2bi2zc">为什么 AI 开发比传统编程更有趣（附真实案例）</a>：构建软件很难。AI 让它变得简单得多。还记得构建 App 意味着无尽的挫败感和无数次的 Stack Overflow 搜索吗……</li><li><a href="https://github.com/tekn">TEKN - 概览</a>：GitHub 是 TEKN 构建软件的地方。</li><li><a href="https://github.com/NousResearch">Nous Research</a>：Nous Research 有 22 个可用的仓库。在 GitHub 上关注他们的代码。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1315613454190776321)** (4 条消息): 

> `训练中的 Momentum，In-context learning 效率，O1-type 合成数据生成` 


- **Momentum 可能有助于 In-context learning**：一位成员提出，如果 **Momentum** 有助于在训练中发现更好的损失景观 (loss landscape)，它也可能对 **In-context learning** (ICL) 有益，可能将其类比为 *强制跳跃连接 (forced skip connections)*。
   - 他们询问 ICL 是否受梯度下降动力学的影响，引发了关于优化方法的有趣探讨。
- **在残差流中实现 Momentum**：[在残差流 (residual stream) 中实现 Momentum](https://link.to/implementation) 被建议作为增强神经网络性能的潜在策略。
   - 这一想法与通过先进训练技术优化 ICL 机制的持续探索相呼应。
- **生成 O1-type 合成数据的资源**：一位成员询问了关于生成 **O1-type 合成数据** 的优质资源或提示词，表明了对实用指南的需求。
   - 这反映了社区对合成数据生成有效方法的广泛兴趣。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1315062693711052833)** (6 条消息): 

> `过去两年的显著 LLM 论文，LLM 中的 Mixture of Experts，LLM 的资源高效训练，训练小型 LLM 和 Diffusion Models，LLM 训练中的挑战` 


- **显著 LLM 论文探讨**：成员们讨论了过去两年中具有影响力且令人难忘的 LLM 论文，提出了包括 [Mixture of Experts](https://arxiv.org/abs/2310.10837) 框架在内的杰出作品。
   - 对话显示，大家对几篇关于 LLM 效率和扩展性 (scaling) 的论文的相关性和潜力达成了共识。
- **为提高效率改进 Mixture of Experts**：[Mixture of Experts](https://arxiv.org/abs/2310.10837) 新视角的引入，通过提高资源效率，展示了与稠密模型相比具有竞争力的结果。
   - 有人指出，最近的进展证明 **MoEs** 可以在保持性能的同时，有效减少计算和内存需求。
- **有限资源下的训练策略**：一篇论文探索了在 **单卡 GPU 上一天内** 从头开始训练语言模型，并取得了与 BERT 相当的结果。
   - 该论文引发了关于如何在受限条件下最大化性能的讨论，证明了即使在有限场景下，高效训练也是可以实现的。
- **强调小型 LLM 和 Diffusion Models**：社区强调了在保持有效性的同时创建小型、高效 LLM 的策略，引用了包括 Nvidia 的 [N-GPT](https://arxiv.org/abs/2406.15786) 在内的多篇论文。
   - 这些讨论围绕利用 Mixture of Experts 和实验性方法作为未来 LLM 训练的创新解决方案展开。
- **对代码复杂性的幽默看法**：成员们开玩笑说，将各种研究技术结合起来训练 LLM 可能会导致“噩梦般”的代码库，以及集成过程的复杂性。
   - 这一轻松的评论强调了研究人员在神经网络训练中挑战技术极限时所面临的困难。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/alpha7987/status/1865503381529522374?s=61">来自 Nothing is something (@Alpha7987) 的推文</a>：本周热门 AI/ML 研究论文：- OpenAI o1 System Card- PaliGemma 2- HunyuanVideo- Densing Law of LLMs- DeMo: Decoupled Momentum Optimization- o1-Coder- Reverse Thinking Makes LLMs Stronger Rea...</li><li><a href="https://x.com/teknium1/status/1865792338666348671?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：各位 LLM 圈的朋友们 - 过去两年中，你们最喜欢、最具影响力或最令人难忘的 LLM 论文有哪些？</li><li><a href="https://x.com/OpenlifesciAI/status/1865584829057929303>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：🌟 每周医学 AI 研究汇总 🌟📅 12月2日–7日，这是本周最令人兴奋的医学 AI 论文摘要！🎉▶️ 医学 LLM 与模型• Block MedCare: Blockchain AI & IoT• LLMs4Life...</li><li><a href="https://youtu.be/SwawtIFy-BI">热门医学 AI 论文 (12月2日–12月7日) | 区块链、公平性与多模态洞察</a>：欢迎回到 Open Life Science AI！本周，我们将探讨 12 月 2 日至 12 月 7 日期间的热门医学 AI 论文。本期精彩内容包括...</li><li><a href="https://arxiv.org/abs/2310.10837">为高效 Transformer 近似双层前馈网络</a>：如何在不牺牲性能的情况下减少神经网络 (NNs) 的计算和内存需求？最近的许多工作使用稀疏 Mixture of Experts (MoEs) 来构建资源高效的大语言模型...</li><li><a href="https://arxiv.org/abs/2312.07987">SwitchHead: 通过 Mixture-of-Experts 注意力加速 Transformer</a>：尽管最近有许多关于 Mixture of Experts (MoEs) 用于资源高效 Transformer 语言模型的研究，但现有方法大多集中在用于前馈层的 MoEs。此前扩展...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: Mixture-of-Experts Universal Transformers</a>：此前关于 Universal Transformers (UTs) 的研究已经证明了跨层参数共享的重要性。通过允许深度递归，UTs 在学习方面比标准 Transformer 具有优势...</li><li><a href="https://arxiv.org/abs/2212.14034">Cramming: 在单张 GPU 上用一天时间训练语言模型</a>：最近语言建模的趋势集中在通过扩展规模来提高性能，这导致训练语言模型对于大多数研究人员和从业者来说变得遥不可及...</li><li><a href="https://arxiv.org/abs/2407.15811">物尽其用：在极低预算下从零开始进行 Diffusion 训练</a>：随着生成式 AI 的 Scaling Laws 推高性能，它们同时也使这些模型的开发集中在拥有庞大计算资源的机构手中。重点关注文本生成图像 (...</li><li><a href="https://arxiv.org/abs/2405.14159">超微型语言模型</a>：大语言模型 (LLMs) 的飞速发展显著提升了自然语言处理能力，但其高昂的计算和能源需求也带来了挑战...</li><li><a href="https://arxiv.org/abs/2406.15786">Transformer 中什么最重要？并非所有注意力都是必需的</a>：虽然扩展基于 Transformer 的大语言模型 (LLMs) 在各种任务中展现了出色的性能，但它也引入了冗余架构，为效率带来了挑战...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1315062693711052833)** (6 条消息): 

> `热门 AI/ML 研究论文、医学 AI 研究、LLMs 中的 Mixture of Experts、高效 LLM 训练、具影响力 LLM 论文综述`

- **顶级 AI/ML 研究论文揭晓**：最近的讨论重点介绍了顶级的 AI/ML 论文，包括 **OpenAI o1 System Card** 和 **PaliGemma 2**，展示了大型语言模型 (LLMs) 的进展。值得关注的贡献还包括 **Efficient Track Anything** 和 **DeMo: Decoupled Momentum Optimization**。
   - 完整列表涵盖了 **Densing Law of LLMs**、**Agent Skill Acquisition**，以及旨在平衡 ML 系统功能与效率的创新文档。
- **每周医疗 AI 研究综述**：12 月 2 日至 7 日的 **Medical AI** 论文强调了 **Block MedCare** 和 **LLaMA II for Multimodal Diagnosis** 等创新，旨在改善临床实践。进一步的探索包括 **RARE: Retrieval-Augmented Reasoning** 等框架以及 **CLINICSUM: Patient Conversation Summaries** 等应用。
   - 关键的伦理讨论强调了 **Privacy in Medical Imaging** 以及 AI 中人口统计公平性的必要性，解决了医疗保健领域的重大挑战。
- **重温 LLMs 中的 Mixture of Experts**：一位成员讨论了关于 **Mixtures of Experts (MoEs)** 观念的转变，指出了一些在不牺牲性能的情况下提高 LLM 效率的新方法。论文展示了 MoEs 以极低的计算成本与稠密 Transformer 竞争的潜力。
   - 对话分享了关于 **MoEs** 竞争格局的见解，强调了它们在资源管理方面的改进，同时验证了近期部分研究进展。
- **高效 LLM 训练技术**：对话围绕优化 LLM 训练的策略展开，包括利用 GPU 能力缩短训练时间。论文证明，在受限环境下，*极简方法*可以实现接近大型模型的性能。
   - 训练方法论的创新，特别是围绕单 GPU 设置，表明较小的模型可以达到具有竞争力的基准测试水平，同时显著降低成本。
- **具影响力 LLM 论文调查**：参与者分享了过去两年中塑造 LLM 发展的具影响力论文，包括探索 Scaling Laws 和结构效率的著名著作。讨论点名了 **Nvidia's N-GPT** 等特定模型，重点关注增强性能的剪枝技术。
   - 成员们强调了结合多项研究见解的潜力，以设计出经济实惠地开发新 LLMs 的实际实现方案，即使生成的代码可能很复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/alpha7987/status/1865503381529522374?s=61">来自 Nothing is something (@Alpha7987) 的推文</a>：本周热门 AI/ML 研究论文：- OpenAI o1 System Card - PaliGemma 2 - HunyuanVideo - LLMs 的密度定律 (Densing Law of LLMs) - DeMo: 解耦动量优化 (Decoupled Momentum Optimization) - o1-Coder - 逆向思维让 LLMs 更强...</li><li><a href="https://x.com/teknium1/status/1865792338666348671?s=46">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：各位 LLM 圈的朋友们——过去两年中，你们最喜欢、最具影响力或最难忘的 LLMs 论文有哪些？</li><li><a href="https://x.com/OpenlifesciAI/status/1865584829057929303>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：🌟 每周医学 AI 研究综述 🌟📅 12月2日–7日，这是本周最令人兴奋的医学 AI 论文摘要！🎉▶️ 医学 LLM 与模型 • Block MedCare: 区块链 AI 与物联网 • LLMs4Life...</li><li><a href="https://youtu.be/SwawtIFy-BI">顶级医学 AI 论文 (12月2日–12月7日) | 区块链、公平性与多模态见解</a>：欢迎回到 Open Life Science AI！本周，我们将探讨 12月2日至12月7日期间的顶级医学 AI 论文。本集亮点包括...</li><li><a href="https://arxiv.org/abs/2310.10837">为高效 Transformer 近似双层前馈网络</a>：如何在不牺牲性能的情况下降低神经网络 (NNs) 的计算和内存需求？近期许多工作使用稀疏混合专家模型 (MoEs) 来构建资源高效的大语言模型...</li><li><a href="https://arxiv.org/abs/2312.07987">SwitchHead: 通过混合专家注意力加速 Transformer</a>：尽管近期有许多关于混合专家模型 (MoEs) 用于资源高效 Transformer 语言模型的研究，但现有方法大多集中在用于前馈层的 MoEs。此前扩展...的尝试...</li><li><a href="https://arxiv.org/abs/2405.16039">MoEUT: 混合专家通用 Transformer</a>：此前关于通用 Transformer (UTs) 的工作已经证明了跨层参数共享的重要性。通过允许深度递归，UTs 在学习...方面比标准 Transformer 具有优势。</li><li><a href="https://arxiv.org/abs/2212.14034">Cramming: 在单张 GPU 上用一天时间训练语言模型</a>：近期语言建模的趋势集中在通过扩展 (scaling) 来提高性能，这导致训练语言模型对于大多数研究人员和从业者来说变得遥不可及...</li><li><a href="https://arxiv.org/abs/2407.15811">钱花在刀刃上：微预算下的扩散模型从零训练</a>：随着生成式 AI 的缩放法则 (scaling laws) 推高性能，它们同时也让这些模型的开发集中在拥有巨大计算资源的参与者手中。重点关注文本到图像 (...</li><li><a href="https://arxiv.org/abs/2405.14159">超微型语言模型</a>：大语言模型 (LLMs) 的飞速发展带来了自然语言处理的显著进步，但由于其高计算和能源需求也带来了挑战。</li><li><a href="https://arxiv.org/abs/2406.15786">Transformer 中什么最重要？并非所有注意力都是必需的</a>：虽然扩展基于 Transformer 的大语言模型 (LLMs) 在各项任务中表现出了可观的性能，但它也引入了冗余架构，给...带来了效率挑战。
</li>
</ul>

</div>
  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1314718425117098075)** (66 条消息🔥🔥): 

> `在本地运行 Ollama 的问题，探索 DSPy 中的人类反馈，DSPy 程序的部署策略，使用 DSPy 进行上下文感知分块，DSPy 与 Anthropic Model Context Protocol` 


- **调查 Ollama 性能**：用户讨论了 **Ollama** 的 **默认 3B 模型** 在本地运行时与终端执行相比性能不一致的问题，并对它的 ChatAdapter 表示困惑。
   - 有人提出需要为量化模型提供更简单的适配器，并致力于改进模型输出。
- **DSPy 中的人类反馈集成**：一位成员询问如何将类似 Agrilla 的人类反馈实现为 DSPy 的指标（metric），并参考了之前关于该功能的讨论和 Pull Request。
   - 相关对话还包括探索在 teleprompting 中引入人类反馈，并分享了相关的 GitHub 链接。
- **DSPy 的部署策略**：成员们分享了 DSPy 程序的各种部署方法，包括使用 **FastAPI** 和 **MLFlow**，并指出生产环境可能需要独立的容器。
   - 还讨论了将 DSPy 集成到 Django 项目或部署在 Modal 上的替代方案，强调了部署选择的灵活性。
- **使用 DSPy 进行上下文感知分块**：探讨了将 **DSPy** 用作上下文感知分块器（chunker）的潜力，并就如何有效优化长文档的处理提出了建议。
   - 对话还涉及了在优化此过程中，小型和大型语言模型的局限性。
- **在 DSPy 中利用 Anthropic MCP**：一位用户询问了在 DSPy 中实现 **Anthropic** 的 **Model Context Protocol (MCP)** 的方案，并获得了关于集成资源和建议的链接。
   - 分享的相关博客文章概述了围绕 MCP 构建工具的方法，强调了其在 AI 工具开发中的应用。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/blog/structured-outputs">结构化输出 · Ollama 博客</a>: Ollama 现在支持结构化输出，可以将模型的输出限制在由 JSON schema 定义的特定格式中。Ollama Python 和 JavaScript 库已更新...</li><li><a href="https://dspy.ai/tutorials/deployment/">部署 - DSPy 文档</a>: 用于编程（而非提示）语言模型的框架。</li><li><a href="https://www.darinkishore.com/posts/mcp">使用 MCP 构建更好的 AI 工具 | Darin Kishore</a>: 构建 AI 工具的经验教训以及为什么 Model Context Protocol (MCP) 很酷。</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1647">Feature/human-in-the-loop-teleprompt 由 burtenshaw 提交 · Pull Request #1647 · stanfordnlp/dspy</a>: 📝 变更说明：这是一个 WIP PR，旨在启动关于在 DSPy 中引入人类反馈的讨论。如果标注者可以在 teleprompting 期间对提示添加反馈，它将支持在特定领域的工作...</li><li><a href="https://gist.github.com/rohitgarud/80bdcd30b65c154e07f343055f95898e">将 Pydantic model_json_schema() 的 JSON schema 转换为更易于 LLM 理解的形式</a>: 将 Pydantic model_json_schema() 的 JSON schema 转换为更易于 LLM 理解的形式 - order_model.py</li><li><a href="https://gist.github.com/rohitgarud/eb60c095a53cf5303fb3ae07b98e268b">为 DSPy 定制的 JSON 适配器，使用 ProcessSchema 简化当 Signature 的 InputField 或 OutputField 以 Pydantic 模型作为类型时注入到提示中的 JSON schema</a>: 为 DSPy 定制的 JSON 适配器，使用 ProcessSchema 简化当 Signature 的 InputField 或 OutputField 以 Pydantic 模型作为类型时注入到提示中的 JSON schema - dspy_custom_a...</li><li><a href="https://github.com/baloise/kwansi">GitHub - baloise/kwansi: 一个基于 DSPy 的自动优化器库</a>: 一个基于 DSPy 的自动优化器库。通过在 GitHub 上创建账号为 baloise/kwansi 的开发做出贡献。</li><li><a href="https://github.com/baloise/kwansi_example">GitHub - baloise/kwansi_example: lordamp/kwansi 封装器在 DSPy 中的示例实现</a>: lordamp/kwansi 封装器在 DSPy 中的示例实现 - baloise/kwansi_example</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1881">[WIP] 在 JSON 适配器中支持基于 signature 的结构化输出响应格式，由 dbczumar 提交 · Pull Request #1881 · stanfordnlp/dspy</a>: 在 JSON 适配器中支持基于 signature 的结构化输出响应格式
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1314684655919956090)** (5 messages): 

> `LlamaParse Multimodal Parsing, Claude Desktop PDF Integration, Agentless Software Issue Resolution, LlamaParse Auto Mode Benefits` 


- **LlamaParse 启用多模态解析**：在一段介绍性视频中，LlamaParse 展示了如何启用兼容 **GPT-4, Claude 3.5** 和 **LLaVA 1.5** 等模型的**高级多模态解析**。
   - 查看[视频演示](https://twitter.com/llama_index/status/1865125665491886171)，了解如何有效地转换屏幕截图。
- **Claude Desktop 连接复杂 PDF**：**Marcus Schiesser** 的一个新项目通过 **Model Context Protocol (MCP)** 将 LlamaCloud 的文档解析与 Claude 集成，实现了与复杂 PDF 的对话功能。
   - 通过这个[详细的项目说明](https://twitter.com/llama_index/status/1865460899059998999)亲身体验。
- **Agentless 提出更简单的故障解决方式**：今天，LlamaIndex 重点介绍了 **Agentless**，它提出了一个简单的三步流程来自动解决软件问题：**定位 (localization)、修复 (repair) 和补丁 (patch)**。
   - 这种方法与更复杂的解决方案形成对比，详见此[公告](https://twitter.com/llama_index/status/1865822785119174857)。
- **LlamaParse 推出成本优化的 Auto Mode**：LlamaParse 中全新的 **Auto Mode** 通过在标准模式下解析文档，并根据用户定义的触发器选择性地切换到 **Premium 模式**来优化成本。
   - 通过此[链接](https://twitter.com/llama_index/status/1866214925418500119)了解更多关于该功能及其优势的信息。
- **LlamaParse Auto Mode 视频演示**：一段视频演示解释了 **LlamaParse Auto Mode** 的功能，旨在提升用户体验。
   - 在[此处观看视频](https://twitter.com/llama_index/status/1866233120481263934)，并确保您的浏览器已更新以获得最佳观看效果。



**提到的链接**：<a href="https://t.co/qBD8sfDsqb">未找到标题</a>：未找到描述

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1314702168355246083)** (35 messages🔥): 

> `Automating Ingestion Pipelines, LlamaIndex RAG Integration, LlamaParse Server Locations, Llama3 Cookbook for Intel Gaudi, OpenAI Seed Mechanism` 


- **自动化聊天应用的摄取流水线 (Ingestion Pipelines)**：一名成员讨论了一个用例，即为一个私有聊天 RAG 应用每小时从 Google Drive 和 Airtable 等数据源自动化摄取流水线。
   - 由于在增量更新方面面临挑战，他们考虑使用任务调度器或云托管解决方案来处理此过程。
- **关于 LlamaIndex RAG 和 OCR 数据的经验**：一位用户询问在应用 LlamaIndex RAG 流程时，处理 PDF 格式的 OCR 读取数据的经验，寻求关于其有效性的见解。
   - 目前没有提供直接回复，凸显了在这一特定应用领域的知识空白。
- **LlamaParse 服务器位于美国**：一名成员询问了 Llamaparse 的服务器位置，表达了对数据留在澳大利亚境内的担忧。
   - 确认服务器目前设在美国，虽然有在欧盟部署的计划，但目前没有澳大利亚的选项。
- **为 Llama3 Cookbook 提交 PR**：一名成员提交了一个 PR，旨在为 Intel Gaudi 添加 Llama3 Cookbook 并请求审查，同时提供了链接以增加曝光。
   - 他们在 GitHub 中包含了该 PR 的描述和详细信息，以吸引贡献者的关注。
- **在 OpenAI Seed 机制中排除元数据**：一位用户寻求在使用 OpenAI 查询引擎的 Seed 机制时获得帮助，担心 Prompt 中出现不需要的元数据。
   - 另一名成员提供了一个解决方案，通过在摄取过程中调整文档的元数据设置，从 Prompt 中排除特定的元数据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/chat_engine/chat_engine_condense_plus_context/">Chat Engine - Condense Plus Context Mode - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/#advanced-metadata-customization">Using Documents - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/v0.10.17/examples/ingestion/ingestion_gdrive.html">Building a Live RAG Pipeline over Google Drive Files - LlamaIndex 🦙 v0.10.17</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/17200">add Llama3 Cookbook for Intel Gaudi by jeanyu-habana · Pull Request #17200 · run-llama/llama_index</a>：描述：为 Intel Gaudi 添加 Llama3 Cookbook。修复了 # (issue)。NANew Package? No。我是否填写了 pyproject.toml 中的 tool.llamahub 部分并为我的新集成提供了详细的 README.md...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1315632711171309649)** (1 messages): 

> `Chain of Thought Prompting, COT techniques, AI problem-solving` 


- **探索 Chain of Thought 提示词的力量**：一名成员分享了一个关于 [COT 提示词的综合资源](https://hub.athina.ai/blogs/what-is-chain-of-thought-prompting-in-ai/)，涵盖了入门所需的各种技术、示例和局限性。
   - 他们提到 **Chain of Thought 提示词**通过分解复杂任务，提高了 AI 处理任务的能力，增强了准确性和逻辑推理。
- **理解 COT：一种用于更好解决问题的 AI 方法**：**Chain of Thought 提示词**鼓励顺序思维过程，使 AI 模型能够更有效地应对困难挑战。
   - 随着 AI 系统集成到 **natural language processing**（自然语言处理）等领域，掌握 COT 对于改进查询响应、促进系统化方法变得至关重要。



**提到的链接**：<a href="https://hub.athina.ai/blogs/what-is-chain-of-thought-prompting-in-ai/">What is Chain of Thought Prompting in AI?</a>：Chain of Thought Prompting (CoT) 概述。一种名为 Chain of Thought 提示词的人工智能方法，鼓励顺序思维，使模型能够更有效地处理具有挑战性的任务...

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1315052541368729661)** (34 条消息🔥): 

> `Adaptive Batching, Llama 3.3 Config Memory Issues, Flex Attention Kernel Bugs, New CPU Flex Kernel, Memory Optimization Techniques` 


- **探索 Adaptive Batching 解决方案**：成员们讨论了需要一种更好的方法来实现 **Adaptive Batching**，建议进行研究并编写一个简单的 RFC 来阐述概念。
   - 一位成员承诺将测量效率，并确认“增加直到 OOM”的想法并非最优。
- **Llama 3.3 配置的挑战**：一位用户在尝试将 **Llama 3.3 70B config** 的内存占用降低到 **49GB** 以下时遇到困难，正在寻求优化方案和替代方法。
   - 建议包括使用 **PagedAdamW** 和 **4-bit** 优化器，但反馈结果褒贬不一。
- **Flex Attention 内核可能引发问题**：报告了一个关于 Flex Attention 的潜在 Bug，该 Bug 会导致共享内存问题，特别是在某些配置和 GPU 型号下。
   - 有建议认为内核选项应针对 **A100/H100** 进行更多优化，而用户体验显示修复效果各异。
- **引入 CPU Flex 内核**：宣布了 **CPU Flex Kernel** 的落地，该内核移除了设备限制。
   - 这使得在不同的硬件配置上进行更广泛的测试和利用成为可能，不再受之前的限制。
- **讨论中的内存优化技术**：成员们讨论了各种内存优化技术，包括修改 **Configurations** 和使用不同的 **Optimizers**。
   - 评估了实际解决方案，一些用户分享了相关资源的链接并讨论了其有效性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://gist.github.com/pbontrager/b7b8dcfd320fa8a4ebf828ed9d33404b">Ultra Low Memory Llama 3.3 Finetuning Config</a>：超低内存 Llama 3.3 微调配置。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/pytorch/torchtune/blob/06a837953a89cdb805c7538ff5e0cc86c7ab44d9/torchtune/modules/loss/ce_chunked_output_loss.py#L30">torchtune/torchtune/modules/loss/ce_chunked_output_loss.py at 06a837953a89cdb805c7538ff5e0cc86c7ab44d9 · pytorch/torchtune</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/2105">[RFC] Step-based checkpointing in torchtune by joecummings · Pull Request #2105 · pytorch/torchtune</a>：在 torchtune 中启用基于 Step 的 Checkpointing。原始背景：#2070。我们目前在做什么？我们目前仅在 Epoch 边界进行 Checkpoint。这意味着微调运行必须迭代...</li><li><a href="https://github.com/pytorch/pytorch/issues/133254#issuecomment-2408710459">Shared memory out of resource when using flex attention · Issue #133254 · pytorch/pytorch</a>：🐛 描述 Bug。当我在一块 RTX 4090 上使用 Flex Attention 时，遇到了一些错误。最小复现：import torch from torch.nn.attention.flex_attention import flex_attention flex_attention = torch.com.....</li><li><a href="https://github.com/pytorch/torchtune?tab=readme-ov-file#optimization-flags">GitHub - pytorch/torchtune: PyTorch native finetuning library</a>：PyTorch 原生微调库。通过在 GitHub 上创建账号为 pytorch/torchtune 开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/pull/812">[Low-bit optim] Improve compile time + Fix PyTorch 2.3 support for 4-bit optim by gau-nernst · Pull Request #812 · pytorch/ao</a>：针对单个参数的静态形状编译优化步骤 + 禁用缓存大小限制。对于给定模型，single_param_adam() 的不同参数组合数量是固定的 -> 可以安全地禁用...</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/_inductor/kernel/flex_attention.py#L714).">pytorch/torch/_inductor/kernel/flex_attention.py at main · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1314749007691517992)** (6 messages): 

> `int8 mixed-precision training, AdamW optimizer usage, batch size adjustments, streamlining pre-commit, just command runner` 


- **int8 混合精度训练的困难**：在尝试实现 **int8 mixed-precision training** 时，确认了在使用特定优化器时会出现 **divergence**（发散）问题。建议通过增加 **batch size** 和 **sequence length** 来应对这些问题。
- **AdamW 优化器解决发散问题**：使用 **AdamW** 作为优化器并移除 **optimizer-in-backward** 成功处理了训练期间的 **loss divergence**。一名成员报告称，在增加 **batch size** 后性能有所提升。
- **使用 Just 精简 pre-commit**：一名成员分享了一个相关的 [GitHub 链接](https://github.com/casey/just/blob/master/examples/pre-commit.just)，用于使用 **just**（一个命令运行器）来精简 **pre-commit** 流程。其简单性和高效性得到了其他成员的赞赏。
- **推广 Just 命令运行器**：该成员强调了 [Just command runner](https://just.systems/man/en/introduction.html) 的实用性，它有助于简化工作流。该工具旨在增强命令执行的自动化，提供了一种直观的解决方案。



**Link mentioned**: <a href="https://github.com/casey/just/blob/master/examples/pre-commit.just">just/examples/pre-commit.just at master · casey/just</a>: 🤖 Just a command runner. Contribute to casey/just development by creating an account on GitHub.

  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1315051172461285397)** (1 messages): 

> `Agents' method changes, Allegations of financial misappropriation` 


- **关于 Agent 方法变更的传闻**：有传言称 Agent 的作者最近更改了 '**pay**' 方法的 **signature**（签名）。
   - 推测暗示这一更改允许他们将所有资金据为己有，引发了关于伦理实践的讨论。
- **财务挪用担忧**：针对 Agent 中方法变更可能涉及的潜在财务挪用问题，人们产生了担忧。
   - 社区内的讨论质疑作者的诚信以及其行为的影响。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1314944742265589851)** (7 messages): 

> `Inf/Nan Handling in Code, Tinygrad Developer Engagement, TinyStats Improvement Suggestions, Upcoming Meeting Agenda, Smart Question Guidelines` 


- **质疑代码中的 Inf/Nan 处理**：一名成员对在面向执行的代码中支持 **Inf 和 NaN** 值表示怀疑，认为 **exploding gradients**（梯度爆炸）通常会导致训练运行变得毫无意义。
   - *这种方法可能会疏远开发者*，但发言者在思考坚持 IEEE 标准是否真的有益。
- **Tinygrad 的开发者参与策略**：有人担心代码变更可能会疏远更多的开发者，而不是吸引他们，这可能与 **Tinygrad 的目标**相冲突。
   - 参与策略必须在鲁棒性与社区增长之间取得平衡，以保持开发者的兴趣。
- **TinyStats 的改进建议**：建议在统计页面的 **Y 轴上添加单位**，因为一些成员不确定数值是越高越好还是越低越好。
   - 数据展示的清晰度将增强用户对 **TinyStats** 的理解和参与度。
- **即将召开的 Tinygrad 会议议程**：定于 **圣地亚哥时间上午 9:30** 举行的会议包含几个关键议程项目，包括删除功能和关于 **cloud sprint** 的讨论。
   - **WebGPU** 以及针对 **ONNX** 和 **tensor cores** 的持续悬赏任务也被列入讨论范围。
- **引用提问智慧指南**：一名成员链接了 **Smart Questions**（提问的智慧）FAQ，以强调在开源社区中提出清晰有效问题的重要性。
   - 该资源旨在帮助成员增强其沟通和寻求支持的策略。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://stats.tinygrad.org/">tinygrad stats</a>: no description found</li><li><a href="http://www.catb.org/esr/faqs/smart-questions.html">How To Ask Questions The Smart Way</a>: no description found
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1314986107145814066)** (28 条消息🔥): 

> `TinyJit 行为, 使用 JIT 进行训练, 数据加载问题, 学习率调度, Librosa 安装问题` 


- **TinyJit 行为异常**：一位用户对添加 `TinyJit` 装饰器后的行为表示困惑，特别是它破坏了模型的功能。
   - 另一位用户澄清说 TinyJit 会捕获 GPU kernels，需要进行调整，例如在某些操作中使用 `Variable`。
- **训练过程需要针对 JIT 进行调整**：有人指出，JIT 函数在每次调用时必须具有相同 shapes 的输入，以避免错误。
   - 讨论建议训练步骤函数应该被 jitted，而数据加载器（data loader）应保持在 JIT 函数之外。
- **JIT 训练中的数据加载陷阱**：用户遇到了一些问题，使用 JIT 导致他们重复传递相同的输入 tensor，而不是新数据。
   - 研究发现，将数据加载代码放在 JIT 函数内部会导致这种重复行为。
- **探索 TinyJit 中的学习率调度**：一位用户询问了实现学习率调度的可能性，以及是否需要重新初始化 optimizer。
   - 他们随后在 GitHub 的 extras 目录中找到了一些相关的实现。
- **M1 Mac 上的 Librosa 安装问题**：一位用户询问是否有人在 Python 3.13.0 的 M1 Mac 上使用 pip 安装 **librosa** 时遇到困难。
   - 在给定的消息中，没有记录到关于此问题的回复。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/docs/stable/jit.html#tracing-edge-cases.">TorchScript &mdash; PyTorch 2.5 文档</a>：未找到描述内容</li><li><a href="https://github.com/kroggen/tokenformer-minimal">GitHub - kroggen/tokenformer-minimal: TokenFormer 的推理与学习极简实现</a>：TokenFormer 的推理与学习极简实现 - kroggen/tokenformer-minimal
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1314719943044104202)** (32 条消息🔥): 

> `作业截止日期, Lab 提交结果, 文章提交, Hackathon 参与, 证书发放` 


- **学生的重要作业截止日期**：所有作业必须在 **12 月 12 日**之前提交，证书申报表需在 **12 月 17 日**之前提交。对于 hackathon 提交，最终截止日期也定为 **12 月 17 日**。
   - 学生可以参考课程网站了解更多详情：[LLM Agents 课程](https://llmagents-learning.org/f24)。
- **Lab 提交结果时间表**：Lab 提交结果将在 **12 月 17 日**之后提供，与证书开始发放的时间一致。建议参与者在此期间依靠运行本地测试。
   - 由于 LLM 行为的自然差异，评分将会比较**宽松**。
- **关于文章提交的说明**：对于文章作业，学生必须在指定的提交字段中包含全文，并单独链接到他们的社交媒体帖子。只要内容保持可访问，使用发布在 Twitter 上的 Notion 链接也是可以接受的。
   - 学生可以选择在文章中详细阐述他们的解决方案，或者保持在高层概述。
- **Hackathon 咨询回复**：Hackathon 参与者可以根据自己的喜好，彻底解释他们的解决方案，或者保持高层概述。已提供说明以确保想法的有效沟通。
   - 只要符合提交要求，学生可以选择不同的平台来展示他们的文章。
- **证书尚未发放**：证书将从 12 月底到 1 月开始发放给学生，一些学生已经在询问他们的状态。符合要求的学生请耐心等待，发放工作正在进行中。
   - 沟通渠道鼓励公开提问，因为这可能也会惠及其他学生。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://rdi.berkeley.edu/llm-agents-hackathon/">LLM Agents Hackathon</a>：由加州大学伯克利分校 RDI 主办的 LLM Agents Hackathon。</li><li><a href="https://llmagents-learning.org/f24">Large Language Model Agents MOOC</a>：MOOC，2024 秋季
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1315259394157580340)** (3 条消息): 

> `书面文章作业，GPT-4 Function Calling 机制，用于训练的代码数据集` 


- **书面文章提交需进一步澄清**：一名成员正在寻求关于 **Written Article Assignment Submission** 指南的说明，特别是文章在提交前是否应发布在 **LinkedIn** 或 **Medium** 上。
   - 他们特别询问是否应在提交表单中标记为 **'Written Article Final Draft'** 的字段中提交文章名称。
- **GPT-4 神奇的 Function Calling 解析**：一位成员对 **GPT-4** 如何通过 API 执行 **'function calling'** 表示惊叹，并提到了其卓越的参数确定过程。
   - 他们询问了讨论该功能背后工程原理的相关论文或博客文章，推测训练集中大量的示例可能是其原因。
- **丰富的代码训练数据**：一位贡献者强调 **code 是高度可用的数据集**，特别是由于 **Stack Overflow** 和 **public GitHub repos** 等来源在纠错方面表现出色。
   - 他们指出，代码的可测量性和确定性特征促进了 **reinforcement learning** 在模型 post-training 中的应用。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1314694410830876752)** (25 条消息🔥): 

> `获取 OpenInterpreter App 访问权限，模型兼容性与 Tool Calls，Multi-Agent 系统讨论，命令的用户审批工作流，OI Pro 用户体验` 


- **申请 OpenInterpreter App 访问权限**：成员们表达了获得 OpenInterpreter 桌面应用早期访问权限的兴奋和渴望，并提到了最近的硬件升级（如 Mac mini）。
   - 反应非常积极，已发送私信进行访问确认。
- **模型兼容性与有效的 Tool Calls**：出现了关于特定模型兼容性和建议的 tool calls 的问题，建议使用 `--no-tools-calling` 以确保运行成功。
   - 成员们分享了让模型有效工作的方法，同时也讨论了在工具执行前建立功能性审批机制的必要性。
- **辩论 Multi-Agent 系统的未来**：一场关于 Multi-Agent 系统有效性的辩论被触发，成员们对其相对于精炼的 single-agent 模型的优势表示怀疑。
   - 论据引用了过去显示单模型优于 multi-agent 框架的表现，导致在未来策略上产生分歧。
- **命令执行的用户审批工作流**：提出了一个结构化的审批工作流建议，用户可以在 AI 生成命令执行前予以批准或拒绝。
   - 该工作流确保了用户的清晰度和控制权，详细说明了批准和拒绝场景的步骤。
- **OI Pro 体验与 VM 限制**：分享了使用 OI Pro 的积极体验，强调了在使用过程中准确性的提高和错误的减少。
   - 一些用户对在 VM 环境中运行 OpenInterpreter 表示担忧，特别是与阻碍功能的显示需求相关的问题。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1315320094993289267)** (2 条消息): 

> `O1 在低配笔记本上的性能，O1 在 Windows 笔记本上的表现，Windows 11 兼容性` 


- **运行 O1 的最低笔记本配置**：一位成员询问了有效运行 **O1** 的最低笔记本规格，寻求关于支持它的最弱硬件的说明。
   - *“01”能运行的最弱笔记本是什么？* 是潜在用户普遍关心的问题。
- **O1 在 Windows 笔记本上的性能**：出现了关于 **O1** 在 Windows 笔记本上性能的问题，有人询问它在这些设备上是否运行良好。
   - 用户特别感兴趣的是能否获得与 [演示视频](https://link.to/demo) 中几乎相同的结果。
- **对 Windows 11 的预期**：一位成员表示有兴趣了解 **O1** 在 **Windows 11** 笔记本上的表现是否与宣传材料中看到的一致。
   - 不确定性在于用户在自己的设置上进行测试时是否可以期待相同的结果。


  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1315738900169494620)** (3 messages): 

> `新产品发布，OpenAI Sora` 


- **OpenAI 发布新产品 Sora**：在一次直播中，OpenAI 确认发布 **Sora**，*Sama* 在直播开始前几分钟宣布了这一消息。
   - 更多详情请访问 [Sora 官网](https://sora.com)。
- **即将到来的直播期待**：*Sama* 暗示在即将举行的直播中发布新产品，引发了对该公告的广泛关注。
   - 该活动在 [Twitter](https://x.com/sama/status/1866179920260739502?s=46&t=G6jp7iOBtkVuyhaYmaDb0w) 上被提及以造势。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sora.com">Sora</a>：将文本和图像转换为沉浸式视频。让故事动起来，将创意可视化，并赋予你的概念生命力。</li><li><a href="https://x.com/sama/status/1866179920260739502?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Sam Altman (@sama) 的推文</a>：5 分钟后在我们的直播中发布新产品：https://openai.com/12-days/
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1314935099426603111)** (9 messages🔥): 

> `垃圾广告问题，德国 LLM 评估，AI 能力认知` 


- **对垃圾广告的担忧**：成员们对来自 Bot 的重复垃圾信息表示沮丧，这些信息显示这是它们唯一的历史消息。
   - 一位成员在注意到这种行为模式后，建议封禁这些账号。
- **评估德国 LLM 性能**：一位成员正在比较各种德国 LLM，指出 `LeoLM/leo-hessianai-7b` 尽管“仅经过预训练”，但在 QA 任务上产生了更好的结果。
   - 有人提出了关于 Llama 模型潜在的底层指令微调（instruction tuning）是否影响了这些结果的问题。
- **提高对 AI 风险的意识**：一位成员敦促其他人向不了解技术的人群普及 AI 生成技术的进展，以防止诈骗。
   - 他们强调诈骗者已经在利用这些能力，并引用了 [MKBHD 的最新视频](https://www.youtube.com/watch?v=OY2x0TyKzIQ) 作为解释这些威胁的有用资源。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1315747273501573163)** (3 messages): 

> `用于医学影像的 MagVit 2，用于 LLM 的内存高效优化器` 


- **关于使用 MagVit 2 对医学图像进行 Tokenize 的咨询**：一位成员询问是否有人有使用 **MagVit 2** 对医学图像进行 Tokenize 的经验，特别是针对 **256x256x256** 的数据集。
   - 他们正考虑将其与基础 Transformer 架构结合，并寻求尝试过此方法的人的反馈。
- **APOLLO：一种新的内存高效优化器提案**：链接到一篇 [arXiv 论文](https://arxiv.org/abs/2412.05270)，介绍了 **APOLLO**，这是一种旨在通过修改 AdamW 的学习率自适应来减少 **Large Language Models (LLMs)** 训练期间内存占用的优化器。
   - 该论文解决了对昂贵的 **SVD 操作**的依赖等挑战，并提出通过低秩优化器状态来近似学习率缩放。



**提到的链接**：<a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>：大型语言模型 (LLMs) 在训练期间对内存的需求极高，尤其是在使用流行的 AdamW 优化器时。这种内存负担使得必须使用更多或更高端的 GPU，或者减少……

  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1315004045391429653)** (3 messages): 

> `Shampoo 低比特实现，Gradient Checkpointing 默认设置` 


- **关于 Shampoo 低比特分支的咨询**：一位成员询问 [shampoo low bit branch](https://github.com/axolotl-ai-cloud/axolotl/tree/shampoo-low_bit) 的实现是否可用，对其功能表现出兴趣。
   - 他们幽默地提到这是“帮朋友问的”，表现出对该话题的轻松参与。
- **将 Gradient Checkpointing 设为默认值的建议**：一位成员提议将 `gradient_checkpointing` 默认设置为 **true**，理由是它被广泛使用且能简化用户体验。
   - 他们强调这一改动将减少用户不必要的设置调整，暗示了可用性的潜在提升。



**提到的链接**：<a href="https://github.com/axolotl-ai-cloud/axolotl/tree/shampoo-low_bit">GitHub - axolotl-ai-cloud/axolotl at shampoo-low_bit</a>：尽管提问。通过在 GitHub 上创建账号，为 axolotl-ai-cloud/axolotl 的开发做出贡献。

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1315702507023896699)** (1 条消息): 

> `Web Applets 开放标准, 图形化客户端应用, 实时编程演示` 


- **Web Applets 开放标准介绍**：明天，一位团队成员将介绍 **Web Applets 开放标准与 SDK**，展示其为 Agent 和人类创建丰富的图形化客户端应用的能力。
   - 本次会议将包含 **实时编程演示**、简短演讲，并开放提问与反馈环节。
- **参与编程环节**：鼓励参与者在演示期间进行互动并提供 **实时反馈**。
   - 欢迎互动讨论和咨询，确保活跃的学习氛围。


  

---


### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1315230397302439978)** (1 条消息): 

> `Dataoorts GPU Cloud` 


- **欢迎 Rajat 和 Dataoorts GPU Cloud**：新成员 **Rajat** 向社区介绍了自己，表达了对加入该群组的 *兴奋* 之情。
   - 他分享了目前正在构建 **Dataoorts GPU Cloud**，旨在支持下一代 AI 开发者的需求。
- **聚焦下一代 AI 开发**：Rajat 在介绍中强调了 **Dataoorts GPU Cloud** 迎合下一代 AI 开发者的目标。
   - 这展示了对增强不断发展的 AI 领域资源的 *明确承诺*。


  

---


---


---


{% else %}


> 完整的频道细分内容已为邮件格式进行截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}