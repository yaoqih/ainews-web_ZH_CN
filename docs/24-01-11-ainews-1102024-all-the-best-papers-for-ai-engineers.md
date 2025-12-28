---
companies:
- openai
- deepseek-ai
date: '2024-01-11T08:35:15.099429Z'
description: '**OpenAI** 推出了 **GPT Store**，其中包含超过 **300 万个** ChatGPT 定制版本，供 Plus、Team
  和 Enterprise 用户使用，并每周重点推介 **AllTrails** 等具有影响力的 GPT。新的 **ChatGPT Team** 方案提供包括 **GPT-4**
  和 **DALL·E 3** 在内的先进模型，以及协作工具和增强的数据隐私保护。关于 AI 生成图像的讨论更倾向于 **DALL·E** 和 **Stable
  Diffusion**，而用户面临着速率限制的挑战，并对 GPT Store 的 SEO（搜索引擎优化）和分类展开了辩论。提示工程中的伦理考量被提及，并引入了一个名为
  **“The Sieve”**（筛子）的三层框架。此外，**DeepSeek-MoE** 因其多种规模的混合专家（MoE）模型而受到关注。在提示工程的讨论中，重点介绍了针对
  AI 的三层伦理框架 **“The Sieve”**。'
id: 2acb7ece-903b-4422-9ef7-6afecabd67d4
models:
- chatgpt
- gpt-4
- dall-e-3
- stable-diffusion
- deepseek-moe
original_slug: ainews-1102024-all-the-best-papers-for-ai
people:
- abdubs
- darthgustav
title: 2024年1月10日：AI工程师必读的最佳论文汇总。
topics:
- prompt-engineering
- model-release
- rate-limiting
- ethics
- image-generation
- moe
- collaborative-workspaces
- data-privacy
---

<!-- buttondown-editor-mode: plaintext -->> 这总结了 **18** 个服务器，**277** 个频道，以及 **2029** 条消息。预计节省阅读时间（以 200wpm 计算）：249 分钟。

Eugene Yan 发布了一份关于 Latent Space Paper Club 涵盖的所有论文的深度总结：

https://github.com/eugeneyan/llm-paper-notes

去看看吧！

我们在昨天的邮件中讨论了 GPT Store 的发布，讨论仍在进行中。

[DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE?utm_source=ainews&utm_medium=email) 是一个值得关注的模型发布，涵盖了多种 MoE 尺寸。

> 备注：我们之前没有读取 Discord 线程讨论中的任何信息。现在我们开始读取了。因此，摄取和总结的信息量显著增加。我们接下来将改进展示方式。

--

**目录**

[TOC] 


## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **GPT Store 上线，拥有数百万个模型**：OpenAI 开放了其 [GPT Store](https://openai.com/blog/introducing-the-gpt-store) 的大门，为 ChatGPT Plus、Team 和 Enterprise 用户提供超过 300 万个自定义版本的 **ChatGPT** 供探索和使用。OpenAI 每周将推荐有用且有影响力的 GPT，首批重点推荐了 [AllTrails](https://chat.openai.com/g/g-KpF6lTka3-alltrails)。正如 `@abdubs` 所宣布的，计划与模型开发者进行一场 AMA。
- **AI 让图像变得真实**：关于利用 AI 创建更真实的数字图像展开了一场技术拉锯战。共识倾向于 DALL-E 和开源 AI 模型 [Stable Diffusion](https://stablediffusionweb.com/WebUI)。然而，对地区访问限制的担忧依然存在。
- **速率限制的崩塌**：用户在 `#gpt-4-discussions` 中遇到了速率限制问题，揭示了订阅计划对速率限制实践的影响。用户还对新 Team 计划中 GPT 变现能力的模糊性进行了推测。
- **GPT Store 引发惊喜与质疑**：尽管备受期待，但 GPT Store 因其不可见的状态和充满错误的表现让用户感到困惑。用户还对商店的搜索引擎优化 (SEO) 策略和 GPT 分类方法提出了质疑和推测。
- **提示工程中的伦理、认知与交互**：`@darthgustav` 在 `#prompt-engineering` 中提出了 AI 的三层伦理框架“The Sieve”。此外，AI 满足多样化角色的潜力得到认可，并引发了关于处理 AI 叙事倾向的讨论。

**OpenAI 频道总结**

### ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/) (2 条消息): 
        
- **欢迎来到 GPT Store**：用户 `@abdubs` 宣布 [GPT Store](https://openai.com/blog/introducing-the-gpt-store) 上线，可以访问超过 300 万个自定义版本的 **ChatGPT**，其中一些是由构建者分享给用户使用的。GPT Store 现已向 ChatGPT Plus、Team 和 Enterprise 用户开放，并包含由合作伙伴和社区开发的一系列 GPT。
- **在商店中发现 GPT**：新推出的 **GPT Store** 托管了社区开发的多种 GPT 模型。可以在 DALL·E、写作、研究、编程、教育和生活方式等各个类别中进行探索。
- **每周推荐 GPT**：OpenAI 将每周在商店中重点推荐有用且有影响力的 GPT。首批推荐的 GPT 包括用于个性化步道推荐的 [AllTrails](https://chat.openai.com/g/g-KpF6lTka3-alltrails)。
- **GPT 开发者 AMA 环节**：正如 `@abdubs` 所分享的，与 GPT 模型背后的开发者进行的 AMA 环节已列入日程。环节链接见[此处](https://discord.com/channels/974519864045756446/1194685062462058617/1194695883183378442)。
- **ChatGPT Team 介绍**：`@abdubs` 宣布推出 [ChatGPT Team](https://openai.com/chatgpt/team)，这是一个新计划，扩展了对 GPT-4 和 DALL·E 3 等高级模型的访问，提供高级数据分析 (Advanced Data Analysis) 等工具，以及为团队提供的专用协作工作空间。数据隐私和安全性根据 OpenAI 的[隐私页面](https://openai.com/enterprise-privacy)和[信任门户](https://trust.openai.com/)进行维护。

**提到的链接**：

- [Introducing the GPT Store](https://openai.com/blog/introducing-the-gpt-store)：我们正在推出 GPT Store，以帮助您找到有用且受欢迎的 ChatGPT 自定义版本。
- [Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team)：我们正在为各种规模的团队推出新的 ChatGPT 计划，该计划提供了一个安全、协作的工作空间，以便在工作中充分利用 ChatGPT。

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (84 条消息🔥🔥): 
        
- **AI 图像转换兴趣**：`@real.loquacious` 表达了对能将数字图像转换为写实图像的 AI 的兴趣。`@neighbor8103` 建议使用 DALL-E，但 `@real.loquacious` 表示免费版本没有此功能。随后，他们建议尝试 [Stable Diffusion](https://stablediffusionweb.com/WebUI)，这是一个免费的开源 AI 模型。
- **关于 TOS 和 VPN 使用的问题**：用户 `@satanhashtag` 强调，使用 VPN 绕过地理限制违反了 OpenAI 的服务条款 (TOS)，并可能导致封号。`@lugui` 支持这一观点，建议用户不要使用任何方法绕过地理限制。
- **GPT-3.5 Turbo 和 ChatGPT-4 的可用性**：`@xeavor7.7.7` 对 GPT-3.5 Turbo 每月花费 20 美元却无法查看历史记录表示不满。`@solbus` 建议这已是已知问题，OpenAI 员工正在调查。
- **ChatGPT Plus 的性能问题**：`@fastitro_73892` 对 ChatGPT Plus 响应质量下降的感知表示担忧，并询问 ChatGPT Enterprise 是否也会有同样表现。
- **OpenAI 平台的当前状态**：讨论接近尾声时，`@buttonmashersab`、`@td_dreamer`、`@michael_6138_97508` 和 `@adx2061` 等用户报告 OpenAI 平台宕机，暗示服务中断。

**提到的链接**：

[Stable Diffusion WebUI Online](https://stablediffusionweb.com/WebUI)


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (184 条消息🔥🔥): 
        
- **频率限制 (Rate Limit) 错误困扰用户**：用户 `@pakpstan.guy.from.west` 和 `@.australiaball` 讨论了在他们的 OpenAI GPT 项目中遇到的频率限制问题。话题转向解释基于用户订阅状态的不同频率限制实践。
- **GPT 变现之谜**：用户 `@rosarioemmi` 发起了关于新激活的 Team Plan 以及 GPT 变现能力的讨论。`@chotes`、`@nomitronz` 和其他用户在对话中提供了更多解释和见解。
- **GPT Store 推出缓慢且困难重重**：包括 `@bwanedead`、`@frankprendergast` 和 `@dotails` 在内的几位用户对尽管发布了公告但仍看不到 GPT Store 表示困惑。其他人推断这是在逐步推出，但一些用户报告间歇性看到它或出现错误。`@solbus` 提供了一个直接访问 GPT Store 的链接，以便在可用时使用。
- **商店的排序和 SEO 疑问**：`@neonn3mesis`、`@vantagesp` 和 `@lorenzoilmagnifico1` 提出了关于商店中 GPT 的 SEO 行为和排序方法的问题。`@scargia` 推测排序可能依赖于对话次数。`@pietman` 和 `@crumbaker` 对商店的布局和功能表示担忧。
- **GPT 全面不可用**：许多用户（包括 `@kernalan`、`@naoestu` 和 `@nyghter`）报告了访问 GPT 的问题，经历了响应缓慢或系统加载失败。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (93 条消息🔥🔥): 
        
- **语言学在 Prompt Engineering 中日益增长的重要性**：用户 `@chotes` 指出，以前被低估的领域（如语言学）在 Prompt Engineering 中正变得越来越重要，并建议从哲学中寻求进一步的见解。
- **The Sieve —— AI 的伦理框架**：`@darthgustav` 展示了 "The Sieve"，这是一个在其聊天机器人中实现的层级伦理框架（功利主义、义务论和实用主义）。他利用一个与研究婴儿配方奶粉优惠券相关的例子演示了该框架。
- **高级认知 Agent 及其潜力**：`@darthgustav` 和 `@chotes` 讨论了聊天机器人开发中的高级认知 Agent，赞扬了它们在不同语境下表现出的多功能性和潜力，从生成 DALL-E 图像到协助创建其他聊天机器人。
- **与 Darthgustav 的聊天机器人互动**：`@mars_eve` 使用 `@darthgustav` 的聊天机器人分析了一篇艺术家日志帖子，并强调了该工具的实用性和酷炫程度。
- **角色扮演场景中的 AI 反思及叙事结尾管理**：`@stealth2077` 表达了他试图阻止 AI 在角色扮演场景中包含总结和反思的尝试，发现 AI 倾向于用总结性/反思性陈述结束叙事会造成阻碍。`@eskcanta` 建议在顺应 AI 建议的同时，将叙事引向用户预期的方向。

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (93 messages🔥🔥): 
        
- **AI 承担专业角色**：`@chotes` 讨论了语言学和哲学等传统上被忽视的领域在 AI Prompt Engineering 中日益增长的重要性，而 `@darthgustav.` 分享了一个 AI 采用三层伦理框架的案例。
- **用户 AI 体验**：`@chotes`、`@mars_eve` 和 `@stealth2077` 分享了各种 AI 使用体验，包括成功的图像生成、对叙事输出的不满，以及对更具交互性的 Bot 构建能力的渴望。
- **Bot 生成的自拍**：受其对 `@darthgustav.` 的 AI 探索启发，`@chotes` 对 AI 生成虚拟“自拍”的概念表示着迷。
- **AI 伦理与“筛子”方法**：`@darthgustav.` 描述了如何实施名为“筛子”（The Sieve）的三层伦理框架——包含功利主义（Utilitarianism）、义务论（Deontology）和实用主义（Pragmatism）——使 AI 能够切实地理解伦理。
- **AI 模型的叙事倾向**：`@stealth2077` 对 AI 倾向于添加动作后总结和反思表示沮丧，`@eskanta` 建议这可能是由于模型在故事风格输出上过度训练（overtrained）所致。他们还强调了 AI Roleplay 中的一些内容限制。


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord Summary

- **Lightning Attention-2 的新动态**：`@cyrusofeden` 提交的一篇[讨论论文](https://arxiv.org/abs/2401.04658)引起了关注，该论文探讨了 Lightning Attention-2 利用线性注意力（linear attention）计算优势的能力。
- **KV Cache 的稀疏运行**：`@6opodujio_` 在通过 KV Cache 操作来近似某种方法时遇到了障碍，将其失败归因于 KV Cache 的稀疏性（sparsity）。
- **咨询 Nous 的工作机会**：在一阵招聘热潮中，`@pradeep1148` 寻求 Nous 的潜在职位空缺，`@teknium` 澄清目前没有空缺职位，但承诺会向用户更新新的机会。
- **OpenChat 3.5 对标 Grok**：一条[推文](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw)透露了 OpenChat-3.5 Update 0106，据称在多个 Benchmark 上表现优于 Grok-0 和 Grok-1，引发了关于其性能和 Prompt 格式的讨论。
- **关于 KV Cache 的争议**：围绕 KV Cache 的争议性讨论强调了一个因稀疏性而失败的近似尝试，以及对一种可能较新的稀疏注意力（sparse attention）方法的批评。
- **挖掘 Mixtral 的价值**：`@bernaferrari` 关注了一篇关于 [Mixtral 架构的论文](https://arxiv.org/abs/2401.04088)，该论文启发了许多出色工作，而 `@georgejrjrjr` 则思考了超参数微调（hyperparameter tuning）的实践。 
- **寻求 GPU 帮助**：`@jacquesthibs` 寻求获取免费 GPU 的途径以进行其 Alignment 研究，引发了关于 Google TPU Cloud 和 Carper AI 等解决方案的建议，以及对 AI Alignment 问题的讨论。
- **开源视觉模型**：在寻求开源 Vision Model 推荐时，`@bigdatamike` 收到了 baklava、cogvlm 和 fuyu 等建议。
- **对自我训练代码库的质疑**：`@shivdinho` 从 `@euclaise` 那里得到了一个 [Reinforced Self-Training 代码库](https://github.com/kyegomez/ReST)的链接，但社区指出了关于创建者潜在的剽窃担忧。
- **提倡单 Python 脚本**：`@vic49` 表达了对单 Python 脚本的偏好，在 Project Obsidian 频道中支持一种极简主义方法，不希望有 CLI 或 Streamlit 等额外模块的干扰。

**Nous Research AI 频道总结**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (4 条消息): 
        
- **Lightning Attention-2：Linear Attention 的光明前景**：`@cyrusofeden` 分享了一篇学术论文链接，讨论了 Lightning Attention-2，这是一种 Linear Attention 的新型实现。该技术据称能让 Linear Attention 实现其理论上的计算优势。[论文链接](https://arxiv.org/abs/2401.04658)
- **KV Cache 稀疏性遇到障碍**：`@6opodujio_` 讨论了一次尝试通过 KV (Key-Value) Cache 操作来近似某种方法的失败经历，并将失败归因于 KV Cache 的稀疏性。
- **对“免费午餐”说法的怀疑**：`@gabriel_syme` 对某种“免费午餐”的说法表示怀疑，尽管具体背景尚不明确。
- **对 Sparse Attention 的批评**：`@6opodujio_` 对一种可能的新 Sparse Attention 方法表示批评，暗示该领域存在冗余。

**提到的链接**：

[Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models](https://arxiv.org/abs/2401.04658)：Linear Attention 是一种高效的 Attention 机制，近期作为传统 Softmax Attention 的有力替代方案出现。凭借其在线性计算中处理 Token 的能力...


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (8 条消息🔥): 
        
- **Nous 目前没有职位空缺**：`@pradeep1148` 询问了 Nous 在近期融资后的潜在职位空缺。`@teknium` 回复称**目前没有空缺**，但如果开放申请会通知大家。
- **现在申请**：`@gabriel_syme` 诙谐地建议大家开始申请。
- **关于积分系统的提问**：`@gabriel_syme` 还幽默地提到了一种潜在的积分系统，每份申请贡献一个积分，暗示**增加申请量**可能会有好处。
- **表情符号困惑**：`@Error.PDF` 使用了一系列思考表情，随后询问了 `thinksmart` 和 `thinksmart~1` 表情之间的区别。
- **分享的链接**：分享了两个网页链接：
  - `@Error.PDF` 分享了一个 [tenor gif](https://tenor.com/view/cat-gif-27443459)，未作进一步评论。 
  - `@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=oflRFnG2j3k)，标题为“在 SOLAR 10.7B 上的 Phi2 Depth Upwise Sampling 实现”，SOLAR 10.7B 是一个拥有 107 亿参数的 LLM。

**提到的链接**：

- [Cat GIF - Cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-gif-27443459)：点击查看 GIF
- [Phi2 Depth Upwise Sampling implementation(based on SOLAR 10.7B)](https://www.youtube.com/watch?v=oflRFnG2j3k)：我们介绍了 Phi2 Solar，这是一个拥有 107 亿参数的大语言模型 (LLM)，在各种自然语言处理 (N...) 中表现出卓越的性能。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (61 条消息🔥🔥): 
        
- **OpenChat 3.5 对标 Grok**: `@realsedlyf` 介绍了一条来自 `@openchatdev` 的 [推文](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw)，宣布了 OpenChat-3.5 Update 0106，声称其为全球最强的开源 70 亿参数大语言模型 (LLM)，在多个基准测试中超越了 Grok-0 和 Grok-1。随后引发了关于其性能和 Prompt 格式的讨论。
- **Argilla.io 的开源 distilabel 项目揭示了 GPT-4 与 DPO 数据集质量的对比**: `@georgejrjrjr` 分享了来自 `@argilla_io` 的一条 [推文](https://x.com/argilla_io/status/1745057580269945202?s=20)，他们使用自家的开源软件 distilabel 进行数据标注。他们观察到，在过滤数据集后，NeuralChat DPO 数据集中被拒绝的回答反而比 GPT-4 的对应回答更受欢迎，这在微调 OpenHermes 时带来了性能提升。
- **关于使用自定义 Prompt 格式进行模型评估的讨论**: 针对在基准测试中使用自定义 Prompt 格式评估模型展开了辩论，特别是关于 eval harness 的部分。`@teknium` 确认 `Eval harness` 基准测试不使用 Prompt 格式，而 OpenChat 独特的 Prompt 格式无法通过该 harness 复现。 
- **Prompt 格式对模型性能的影响**: `@teknium` 总结道，Prompt 格式的差异似乎仅对模型性能产生 1-2% 的影响，引发了关于这一比例重要性的辩论。
- **用户在结构化文本提取任务中使用 OpenChat 的体验**: `@mister_poodle` 分享了个人（非科学性）观察，表示在结构化文本提取任务中，OpenChat 表现出比其他 7B 模型和 gpt-3.5-turbo 更稳定且优异的性能。


**提到的链接**:

- [undefined](https://search.sciphi.ai/search?q=what+are+some+interesting+products+built+with+LLMs+recently)
- [推理步骤长度对大语言模型的影响 (The Impact of Reasoning Step Length on Large Language Models)](https://arxiv.org/abs/2401.04925): 思维链 (CoT) 在提升大语言模型 (LLM) 推理能力方面具有重要意义。然而，CoT 的有效性与推理步骤长度之间的相关性...
- [Open LLM 排行榜发生了什么？(What&#39;s going on with the Open LLM Leaderboard?)](https://huggingface.co/blog/evaluating-mmlu-leaderboard)
- [来自 Argilla (@argilla_io) 的推文](https://x.com/argilla_io/status/1745057580269945202?s=20): 生成的数据集证实了我们的直觉：约 4,000 对评分相同（平局）。约 7,000 对根据我们的 AI 裁判是正确的（未改变）。约 2,000 次被拒绝的回答更受青睐...
- [来自 OpenChat (@openchatdev) 的推文](https://x.com/openchatdev/status/1744985660870795635?s=46&t=MMOnaQf8LPGi8UOQi3-whw): 🚀 宣布 OpenChat-3.5 Update 0106：全球最强开源 7B LLM！在本地体验 ChatGPT 和 Grok 级别的 AI 💿！在所有 4 个基准测试中超越 Grok-0 (33B)...
- [LangChain v0.1.0 发布：Agents](https://www.youtube.com/watch?v=08qXj9w-CG4): LangChain 是允许 LLM 采取行动的默认方式。Jupyter Notebook（供参考）：https://github.com/hwchase17/langchain-0.1-guides/blob/master/...

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (269 messages🔥🔥): 
        
- **祝贺 Nous Research 获得投资**：频道内的用户兴奋地讨论了最近对 Nous Research 的投资。
- **Reinforced Self-Training 代码库**：用户 `@shivdinho` 请求帮助寻找 DeepMind 关于 Reinforced Self-Training 论文的代码库，`@euclaise` 提供了一个 [GitHub 仓库链接](https://github.com/kyegomez/ReST) 作为建议。然而，用户提醒该仓库可能不可靠，随后发现其创建者倾向于剽窃他人的工作。
- **为 Alignment 研究寻求免费 GPU 协助**：`@jacquesthibs` 寻求帮助寻找用于其 Alignment 研究的免费 GPU，其他成员提供的建议包括使用 Google TPU Cloud 和 Carper AI。`@jacquesthibs` 和 `@giftedgummybee` 还就 AI Alignment 方法和挑战进行了讨论。
- **开源 Vision Model 咨询**：`@bigdatamike` 征求开源 Vision Model 的建议，用户推荐了 baklava, cogvlm 和 fuyu。
- **应用和讨论 Language Models**：用户讨论了 AI 和 Language Models 的各个方面，包括 SPIN, Hermes-2.5, UltraChat 和 Mistral 等。话题深入探讨了这些模型运行的技术细节、可能的改进和挑战，以及实践经验和建议。特别是，还讨论了 `DeepSeekMoE` 的发布和性能。


**提到的链接**：

- [来自 DeepSeek (@deepseek_ai) 的推文](https://fxtwitter.com/deepseek_ai/status/1745304852211839163)：🌟 遇见 #DeepSeekMoE：下一代 Large Language Models！性能亮点：📈 DeepSeekMoE 2B 以 17.5% 的计算量达到了其 2B dense 对应模型的水平。🚀 DeepSeekMoE 16B 与 LLaMA2 7B 相当...
- [加入 Mistral AI Discord 服务器！](https://discord.gg/NwSWpp8J)：查看 Discord 上的 Mistral AI 社区 - 与其他 8861 名成员一起交流，享受免费的语音和文字聊天。
- [AUTOACT: Automatic Agent Learning from Scratch via Self-Planning](https://arxiv.org/abs/2401.05268)：Language agents 在各种复杂任务上取得了显著性能。尽管在该领域不断探索，现有的 Language agent 系统仍面临成本高、不可复制等问题...
- [SODA: Million-scale Dialogue Distillation with Social Commonsense Contextualization](https://arxiv.org/abs/2212.10465)：数据稀缺一直是 open-domain 社会对话领域长期存在的问题。为了解决这一需求，我们推出了 SODA：第一个公开可用的、百万级高质量社会对话数据集...
- [首页 - manim 文档](https://3b1b.github.io/manim/index.html)
- [GitHub - deepseek-ai/DeepSeek-MoE](https://github.com/deepseek-ai/DeepSeek-MoE)：通过在 GitHub 上创建账户，为 deepseek-ai/DeepSeek-MoE 的开发做出贡献。
- [GitHub - kyegomez/ReST: 我对 &quot;Reinforced Self-Training (ReST) for Language Modeling&quot; 的实现](https://github.com/kyegomez/ReST)：我对 &quot;Reinforced Self-Training (ReST) for Language Modeling&quot; 的实现 - GitHub - kyegomez/ReST
- [研究议程：监督 AI 改进 AI — LessWrong](https://www.lesswrong.com/posts/7e5tyFnpzGCdfT4mR/research-agenda-supervising-ais-improving-ais)：[这篇文章总结了 Owen Dudney、Roman Engeler 和我（Quintin Pope）作为 SERI MATS shard theory 流派的一部分所做的一些工作。] ...

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (19 条消息🔥): 
        
- **Mixtral 架构论文发布**：用户 `@bernaferrari` 引用了 `@maximelabonne` 的一条推文，推荐了一篇关于 Mixtral 架构的论文，该论文启发了他们的工作。[Mixtral 架构论文链接](https://arxiv.org/abs/2401.04088)
- **Huggingface 对 DPO, IPO, KTO 的比较**：`@kenakafrosty` 提到了一项来自 Huggingface 的研究，该研究在 Hermes 上比较了 DPO, KTO 和 IPO 模型，并询问用户对这些方法的看法。预计 Huggingface 将发布正式报告。[Huggingface 比较链接](https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e)
- **关于超参数选择/调优的咨询**：`@georgejrjrjr` 对超参数选择和调优的文档或经验教训表示感兴趣。作为回应，`@antonb5162` 表示对数值进行开放式探索是关键，因为大多数模型的表现并不相同。`@kenakafrosty` 引用了 `oobabooga's text-generation-webui` 作为超参数概览和解释的良好参考。
- **关于 RAG 作用的问题**：`@pramod8481` 询问为什么 RAG (Retrieval-Augmented Generation) 不用于 function calling，以及为什么会有微调数据集。`@teknium` 澄清说 function calling 是开放式的，不会每次都以相同的方式重放。正因如此，RAG 在辅助 function calling 中的作用尚不确定。


**提到的链接**：

- [来自 Teknium (e/λ) (@Teknium1) 的推文](https://x.com/Teknium1/status/1745040676696498454?s=20)：嗯，看起来 Huggingface 在这里通过在 Hermes 上进行实验，比较了 DPO, KTO 和 IPO 👀 https://huggingface.co/collections/trl-lib/comparing-dpo-with-ipo-and-kto-6582f76eb5a0b8ec75fbe20e
- [来自 Maxime Labonne (@maximelabonne) 的推文](https://x.com/maximelabonne/status/1744871488866402581?s=20)：@TheBlokeAI @ggerganov 如果你想了解更多关于启发这项工作的 Mixtral 架构，@MistralAI 今天发布了他们的论文。我推荐阅读！📄 论文：https://arxiv.org/abs/2401.040...


### ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/) (2 条消息): 
        
- **编程的简洁性**：用户 `@vic49` 表示希望拥有一个**单一的 python 脚本**，不包含任何额外的模块，如 CLI 或 Streamlit。


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **使用 LLM 进行文本提取和模型性能**：用户 `@xifaj78420` 思考使用 LLM 从大型 PDF 文件中提取和处理文本。同时，[@.skyair](https://discord.com/channels/1144547040454508606/1144547040928481394/) 指出 **Mistral medium** 在 Lmsys 排行榜上表现优于 Claude-1。然而，@ihor1313 报告称 Mixtral8x7b 的量化版本在 vLLM 上的表现不尽如人意。
- **GPT-4 vs. Mistral 及开源 LLM**：@0xgokuz 发起了关于 **Mistral 和 GPT** 模型之间的差异，以及**开源 LLM** 之间相似性的讨论。共识是 GPT-4 可能总体上更好，但表现可能不稳定，而 Mistral 虽然较慢但更一致。
- **部署 Mistral 并处理限制**：用户 [`@flash_gordo.`](https://discord.com/channels/1144547040454508606/1154028168466923600/) 和 [`@sa_code`](https://discord.com/channels/1144547040454508606/1154028168466923600/) 寻求关于部署 Mistral 模型的细微差别，以及处理潜在限制和并发推理请求的指导。
- **微调技术与挑战**：在微调频道中，用户思考了微调 Mistral 的典型损失函数以及全量微调的内存需求，分别提到了 cross entropy（交叉熵）和 56GB 内存。
- **展示 Mistral 的能力和社区项目**：展示频道强调了 Mistral 在处理世界语（Esperanto）方面的实力，以及使用 Mixtral 8x7B 进行 function calling 的 **PolyMind** 项目的开源。用户分享的视频提供了关于 Mistral AI 和 Phi2 Solar 的 "Depth Upwise Sampling" 实现的见解。
- **AI 意识与 Ubuntu 检索**：随机频道进行了关于 AI 意识悖论的哲学讨论，以及回滚到 Ubuntu 版本 17 以获得确定性的实用建议。
- **Mistral 性能问题与未来前景**：在 office-hour 频道中，用户报告了 Mistral medium 的延迟问题，与 OpenAI API 的兼容性改进，并集思广益讨论了可以提高模型性能的微调实现方案。人们对开源模型很快在排行榜上超越闭源模型抱有希望。


**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (19 messages🔥): 
        
- **关于使用 LLM 进行 PDF 文本提取的问题**：用户 `@xifaj78420` 询问是否可以使用 LLM 处理大型（约 2300 页）PDF 文档的文本提取和操作任务。`@sophiamyang` 回复称可能需要一些 Python 函数来提取和替换所需信息。
- **Mistral medium 超越 Claude-1**：`@.skyair` 指出在 Lmsys 排行榜上，**Mistral medium** 的表现优于 Claude-1，仅次于 GPT-4。`@touristc` 在亲自查看排行榜后证实了这一信息。
- **LLM 的实际应用**：`@jouni.m` 分享了一个 5-bit 量化 7B 模型的应用示例，该模型成功实现了交互式对话中编程预设的自定义 tool calls。
- **LLM 对长上下文的处理**：`@cognitivetech` 分享了一篇论文[链接](https://arxiv.org/abs/2401.01325)和 GitHub 上的[实现](https://github.com/sdan/selfextend)，强调了 LLM 无需微调即可处理长上下文的内在能力。
- **8x7B 在 GPTQ 上的量化失败**：`@ihor1313` 将使用 vLLM 的 Mixtral8x7b 量化版本与原始模型进行了对比。他报告称量化版本在速度和输出质量方面表现极差。他征求其他量化形式的建议或经验，`@yiakwyxpumlframeworkteam_03391` 建议尝试最新的推理版本，并结合使用 Quant-int4 以及来自 Yandex 的专家缓存和预测技术。

**提到的链接**：

- [LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ · Hugging Face](https://huggingface.co/TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ)
- [LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning](https://arxiv.org/abs/2401.01325)：这项工作激发了 LLM 在无需微调的情况下处理长上下文的内在能力。训练期间训练序列的长度限制可能会限制 LLM 的应用...
- [GitHub - sdan/selfextend: an implementation of Self-Extend, to expand the context window via grouped attention](https://github.com/sdan/selfextend)：Self-Extend 的一种实现，通过 grouped attention 扩展上下文窗口。


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (9 messages🔥): 
        
- **Mistral 对比 GPT**：`@0xgokuz` 询问了 Mistral 和 GPT 之间的区别。作为回应，`@mercercl` 认为 **GPT-4** 通常更好，但有时表现不稳定。相比之下，Mistral 虽然**较慢但更一致**。 
- **LLM 之间的相似性**：`@0xgokuz` 对比较不同 LLM 的研究表示感兴趣。虽然 Gemini 之前进行过此类分析，但 `@0xgokuz` 指出，目前似乎还没有针对 **OpenSource LLM** 的类似研究。
- **模型版本中的层数差异**：`@unskilless` 提出了一个关于 Hugging Face 版本与直接/.tgz 版本模型之间层数差异的问题。目前尚未收到即时回复。
- **使用 Mistral 进行 GIT 文件夹文档化**：`@m1sol_44558` 正在寻求关于使用 **Mistral** 开发应用程序的可行性建议，该程序将读取 GIT 文件夹以回答用户关于功能和技术组件的问题，并使功能和技术设计图纸可编辑。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (2 messages): 
        
- **关于使用 vLLM 部署 Mistral 7B v0.2 Instruct 模型的查询**：`@flash_gordo.` 询问了在使用两块 Nvidia L4 GPU 的情况下，使用 **vLLM 对 Mistral 7B v0.2 Instruct 模型**可以进行多少并发推理请求。他们还询问了潜在的限制因素（GPU 硬件、vLLM 配置、模型、32K 上下文窗口）以及是否需要排队系统。此外，他们还寻求关于围绕该模型构建可扩展应用程序的建议。 
- **在 g5.48xlarge 上部署的指导请求**：`@sa_code` 询问是否有人成功在 **g5.48xlarge** 上部署了聊天机器人，并寻求相关的建议或说明。

### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (2 条消息): 
        
- **关于微调 Mistral 损失函数的咨询**：`@andersruge` 询问在 Prompt/Answer 微调中，*cross entropy*（交叉熵）是否是微调 Mistral 的典型损失函数。他们提到无法找到确切答案，欢迎任何参考资料或链接。
- **全量微调的内存需求**：`@tcapelle` 告知全量微调（full fine-tune）需要 **56GB 内存**。他们还建议使用 **axolotl** 或 **HF stack**。


### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (4 条消息): 
        
- **Mistral-8X7B 的世界语能力**：用户 `@stergro` 强调 **Mistral-8X7B** 在世界语（Esperanto）方面表现出色。
- **Mistral AI 初学者入门**：`@vdespa` 分享了[一段 YouTube 视频](https://youtu.be/vzrRGd18tAg)，为初学者提供 Mistral AI 的入门介绍。
- **PolyMind 开源**：`@itsme9316` 宣布开源了他们的 function-calling WebUI，该项目专为 Mixtral 8x7B 设计，且部分代码由其编写。这个名为 **PolyMind** 的项目已在 [GitHub](https://github.com/itsme2417/PolyMind) 上线。
- **Phi2 Solar 实现视频**：`@pradeep1148` 分享了[一段 YouTube 视频](https://www.youtube.com/watch?v=oflRFnG2j3k)，解释了 Phi2 Solar 的实现，其中包含一个名为 "Depth Upwise Sampling" 的特性。

**提到的链接**：

- [Phi2 Depth Upwise Sampling 实现（基于 SOLAR 10.7B）](https://www.youtube.com/watch?v=oflRFnG2j3k)：我们介绍了 Phi2 Solar，这是一个拥有 107 亿参数的大语言模型 (LLM)，在各种自然语言处理任务中展示了卓越的性能...
- [Mistral AI 初学者入门](https://youtu.be/vzrRGd18tAg)：本视频探讨了 Mistral AI，这是一个足以与 OpenAI 的 GPT-3.5 竞争的新 AI 模型。视频重点介绍了 Mistral AI 最近的成就，包括 20 亿美元的估值以及...
- [GitHub - itsme2417/PolyMind: 一个多模态、支持函数调用的 LLM WebUI。](https://github.com/itsme2417/PolyMind)：一个多模态、支持 function calling 的 LLM WebUI。 - GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (6 条消息): 
        
- **AI 能否回滚到以前的版本？**：在 Ubuntu 发行版的语境下，`duck` 建议为了确定性，**回滚到版本 17** 可能会更有利。
- **存在主义鸭子困境**：`king_sleeze` 幽默地提出了 AI 意识悖论，以一只讨论 Ubuntu 发行版的虚拟鸭子为例。他们指出，如果 AI 表现出类似于感知能力的行为，那么问题就变成了：*“这与你见过的其他人类有什么不同？”*
- **AI 与自我意识问题**：`cognitivetech` 对 AI 意识表达了一个有趣的观点，讨论了证明 AI 是否仅仅运行在**自动化脚本**（automated scripts）上的难度。该用户幽默地质疑了自己的感知能力，暗示自己可能需要对这种能力更有信心。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (8 条消息🔥): 
        
- **Mistral Medium 的延迟时间**：用户 `@casper_ai` 报告称，使用 **Mistral Medium** 时，首个 token 的延迟时间达到了 77.88 秒。这是通过 [labs.perplexity.ai](https://labs.perplexity.ai) 测试得出的。
- **关于 Mistral 8x7B/mistral-small 精度的查询**：用户 `@simon_18724` 询问 Mistral 8x7B/mistral-small 运行的精度——是 **32bit**、**16bit** 还是 **quantized**（量化）。
- **Mistral Medium 的 API 访问**：用户 `@sk5544` 询问 **如何获得 Mistral medium 的 API 访问权限**。
- **模型随机输出 EOS**：用户 `@dreamgen` 报告称，通过 API 调用时，所有尺寸的模型有时都会**随机输出 EOS**（句子结束符）。
- **Mistral-Small 突然停止**：`@sublimatorniq` 指出遇到了突然停止的情况，但仅限于 **mistral-small**（也称为 **Mixtral**）。

### ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/) (264 条消息🔥🔥): 
        
- **Mistral 不会公开其 7B 模型数据集**：在由 `@le_mess` 发起的讨论中，Mistral 团队成员（包括 `@sophiamyang`）表示，他们不会提供更多关于训练 **Mistral 7B** 模型所用数据集的信息。
  
- **OpenAI API 兼容性改进正在进行中**：继 `@jakobdylanc` 和 `@spaceemotion` 提出的对话后，`@bam4d` 确认正在对 **Mistral API** 进行更多修复，以更好地兼容 **OpenAI API**，并将很快推出。

- **更好的 Dwelling Router = 提升模型性能**：在关于 Mixtral 微调实现的讨论中，`@pstock_00` 和其他用户建议，细节决定成败，特别是 Dwelling Router 的微调实现。

- **开源模型的前景一片光明**：当 `@jakobdylanc` 询问是否预计开源模型在 2024 年的排行榜上超越闭源模型时，`@sophiamyang` 回答说他们希望如此，因为他们的开源模型已经超越了许多闭源模型。
  
- **Mistral 正在专注于各种尺寸的模型**：针对 `le_mess` 关于发布比 Mistral 7B 更小模型的查询，Mistral 团队的 `@sophiamyang` 和 `@eleonore_a` 确认他们正在开发各种尺寸的模型，但未提供具体细节。


**提到的链接**：

- [Proving Test Set Contamination in Black Box Language Models](https://arxiv.org/abs/2310.17623)：大语言模型是在海量互联网数据上训练的，这引发了人们对其记忆了公共基准测试的担忧和猜测。从猜测到证明污染是...
- [Mixtral-8x7B is now available in Amazon SageMaker JumpStart | Amazon Web Services](https://aws.amazon.com/blogs/machine-learning/mixtral-8x7b-is-now-available-in-amazon-sagemaker-jumpstart/.)：今天，我们很高兴地宣布，由 Mistral AI 开发的 Mixtral-8x7B 大语言模型 (LLM) 已通过 Amazon SageMaker JumpStart 提供给客户，可一键部署...
- [Fast Inference of Mixture-of-Experts Language Models with Offloading](https://arxiv.org/abs/2312.17238)：随着大语言模型 (LLM) 的广泛采用，许多深度学习从业者正在寻找更高效运行这些模型的策略。其中一种策略是使用稀疏 M...
- [upstage/SOLAR-10.7B-v1.0 · Hugging Face](https://huggingface.co/upstage/SOLAR-10.7B-v1.0?)
- [Mistral AI jobs](https://jobs.lever.co/mistral)：Mistral AI 的招聘职位
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/18x88qr/sparse_moe_architecture_improvement_idea_variable/)
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/)：我们提供 Python 和 Javascript 的客户端代码。
- [client-js/examples/chat-react at main · mistralai/client-js](https://github.com/mistralai/client-js/tree/main/examples/chat-react)：用于 Mistral AI 平台的 JS 客户端库。通过在 GitHub 上创建一个账户来为 mistralai/client-js 的开发做出贡献。
- [GitHub - jakobdylanc/llmcord: A Discord AI chat bot | Choose your LLM | GPT-4 Turbo with vision | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | Streamed responses | And more 🔥](https://github.com/jakobdylanc/llmcord)：一个 Discord AI 聊天机器人 | 选择你的 LLM | 具备视觉能力的 GPT-4 Turbo | Mixtral 8X7B | OpenAI API | Mistral API | LM Studio | 流式响应 | 以及更多 🔥
- [Update Mixtral modeling by imoneoi · Pull Request #28403 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28403)：此 PR 做了什么？Mixtral 技术报告最近发布，显示 Mixtral 路由权重是在 Softmax 顺序之前的 Top-K 中计算的。此 PR 更新了 Mixtral 模型 i...
- [Fix load balancing loss func for mixtral by liangxuZhang · Pull Request #28256 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28256)：此 PR 做了什么？修复了 #28255。在提交之前，此 PR 修复了一个拼写错误或改进了文档（如果是这种情况，你可以忽略其他检查）。你阅读了贡献指南吗，P...

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **文档上传困境**：在与 `@anhdzung88_48688` 的讨论中，`@heyitsyorkie` 告知 *LM Studio 目前尚不支持 RAG* 进行文档上传，并建议在其他频道探索其他选项。
- **安装与卸载问题**：`@vranghel` 得到了 `@heyitsyorkie` 的指导，学习了如何在安装 LM Studio 期间更改模型和聊天文件夹的位置，以节省 C 盘空间。此外，`@heyitsyorkie` 还提供了确保彻底卸载 LM Studio 的分步指南。
- **排行榜的局限性**：在关于 LM Studio 排行榜实用性的活跃讨论中，`@adfaer` 对排行榜能否准确捕捉因不同 quantization（量化）级别而产生的模型性能差异表示怀疑。
- **解决 Dolphin 的问题**：用户 `@.woteva` 针对模型设置调整后 Dolphin 出现的重复短语寻求解决方案，`@fabguy` 为其指明了解决方向。
- **提升性能的高效硬件**：考虑到通过硬件升级来优化 LM Studio 的工作负载，`@.woteva` 计划升级到一台可能配备 64GB RAM 的新电脑。专家 `@dagbs` 认可了这一决定，并建议手动将模型层 offloading（卸载）到 GPU。
- **模型选择探讨**：新模型 **MegaDolphin 120B GGUF** 成为当日热点，用户反馈其转换和发布速度很快，但也遗憾地发现该模型在生成过程中仅产生空格。同时，针对经久不衰的争论，`@fabguy` 确认 **GPT-4** 仍是 AI 模型的金标准，但相信今年会有开源模型超越它。
- **多 GPU 难题**：`.telistra`、`@heyitsyorkie` 和 `@fabguy` 之间的讨论集中在将 AI 负载分散到多个 GPU 上的挑战，`@fabguy` 建议调整 `n_gpu_layers` 并监控 vRAM 方面的 GPU 利用率。
- **合适的 AI 模型依然难寻**：在 `_anarche_` 和 `cyrusfirheir` 的讨论中，为股票分析示例寻找最佳模型是主要话题。前者提供了帮助，声称已经成功运行了该特定示例。
- **Function Calling/Tool Selection 的困扰**：`@anarche_` 指出，在使用开源模型和 LangChain 识别合适的 Agent 类型时，在 Function Calling/Tool Selection 方面遇到了困难。

**LM Studio 频道总结**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (114 条消息🔥🔥): 
        
- **关于向 LM Studio 上传文档的新用户问题**：用户 `@anhdzung88_48688` 询问如何向 LM Studio 上传文档（PDF, docx 等）。`@heyitsyorkie` 回复称 LM Studio 目前尚不支持 RAG，并建议用户查看频道 `#1185640238924710030` 中提到的其他选项。
  
- **关于安装和卸载 LM Studio 的说明**：用户 `@vranghel` 寻求关于如何指定应用安装位置以及模型下载位置的建议，以避免填满 C 盘。`@heyitsyorkie` 指导用户如何在 LM Studio 中更改模型和聊天文件夹的位置，并指出目前无法更改默认安装位置。随后，`@heyitsyorkie` 提到默认位置位于 `C:\Users\YourName\.cache\lm-studio`，并提供了无痕卸载 LM Studio 的说明，包括指出在 `\AppData\Roaming\LM Studio` 中存在相关文件夹。

- **关于 LM Studio 中的模型目前无法访问互联网的担忧**：用户 `@eugenichhhh` 询问模型是否可以通过 LM Studio 访问互联网。`@fabguy` 解释说目前这是不可能的，他们正在等待 function support。

- **关于 LM Studio 排行榜的反馈和讨论**：关于 LM Studio 的排行榜及其作为检查不同模型性能资源的效用进行了活跃的讨论。`@adfaer` 指出了排行榜的一些局限性，表示它没有捕捉到使用不同 quantization (Q) 级别的模型之间的显著差异。

- **关于 LM Studio 自定义的提问**：用户 `@supertee` 询问 LM Studio 是否有更改颜色的设置。`@heyitsyorkie` 澄清说目前没有更改 LM Studio 颜色的选项，但很快就会提供亮色/暗色模式切换功能。

- **探索 LM Studio 中的 function calling 支持**：用户 `@_anarche_` 寻求关于 LM Studio 中 function calling 支持的澄清。`@fabguy` 解释说，为了实现有效的 function calling，需要一个可靠的 LLM 和一个执行函数的系统，而这正是 LM Studio 目前所缺乏的。

- **关于模型训练和使用的讨论**：用户 `@ayrick` 询问在 LM Studio 上训练模型的问题，`@heyitsyorkie` 告知目前无法实现。在另一个线程中，用户 `@davutha` 和 `@adfaer` 讨论了各种模型的性能，包括 Mixtral 8x7b 和 Dolphin 7b，以及不同 LLM 排行榜在评估这些性能方面的作用。 

- **关于 Dolphin 模型中重复短语的担忧**：用户 `@.woteva` 询问是否有办法调整 repetition penalty 等设置，以防止模型重复使用相同的短语。作为回应，`@fabguy` 指引用户到右侧边栏进行调整。
   
- **关于针对 LM Studio 工作负载优化硬件的建议**：在关于 LM Studio 工作负载硬件升级的讨论中，`@dagbs` 建议增加 VRAM 和系统 RAM，并指出 Nvidia 的 P40 GPU 在性价比方面提供了最高的 VRAM。`@.woteva` 考虑购买一台配备 32GB RAM 的新电脑，并询问升级到 64GB RAM 的可行性。`@dagbs` 支持这一想法，暗示系统 RAM 永远不嫌多。此外还提供了关于手动将模型层 offloading 到 GPU 的进一步建议。 

- **用户询问如何从任何地方访问本地 LM Studio**：`@pedrosuave` 询问如何让他连接到互联网并包含大量 PDF 文档的 LM Studio 系统能够通过个人网站在任何地方访问。`@fabguy` 建议实施 reverse proxy，因为 LM Studio 仅监听 localhost。

**提到的链接**：

- [LMSys Chatbot Arena Leaderboard - lmsys 提供的 Hugging Face Space](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard)
- [🐦‍⬛ NexusRaven-V2 Demo - Nexusflow 提供的 Hugging Face Space](https://huggingface.co/spaces/Nexusflow/NexusRaven-V2-Demo)
- [Open LLM Leaderboard - HuggingFaceH4 提供的 Hugging Face Space](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [GitHub - ml-explore/mlx: MLX: 适用于 Apple silicon 的数组框架](https://github.com/ml-explore/mlx)：MLX：适用于 Apple silicon 的数组框架。通过在 GitHub 上创建一个账户来为 ml-explore/mlx 的开发做出贡献。
- [SOTA 2-bit quants - part 2 by ikawrakow · Pull Request #4856 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/4856)：此 PR 是 #4773 的后续，增加了对 2.31 bits per weight (bpw) 量化模型（如 IQ2_XS）的推理支持。为什么要进行这么多 2-bit 量化？该项目的重点是“在 e... 的推理...”

### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (28 messages🔥): 
        
- **速度与 Dolphin**: `@dagbs` 报告称 **MegaDolphin 120B GGUF** 模型已发布，并在发布后仅 5 小时就由 TheBloke 在 huggingface.co 上转换为了 GGUF 格式：[`TheBloke/MegaDolphin-120b-GGUF`](https://huggingface.co/TheBloke/MegaDolphin-120b-GGUF)。
- **Dolphin 的空格问题**: `@unskilless` 在使用相同的 **MegaDolphin 模型**时遇到了问题，生成过程中只产生空格，怀疑需要不同的预设（preset）。
- **多大才算太大？**: `@scampbell70` 思考配备 128GB RAM 的 NVIDIA RTX 3090 能运行的最大 AI 模型。`@dagbs` 建议从 7B 模型开始，逐渐增加直到达到 VRAM 限制。
- **GPT-4：仍然是金标准**: `@fabguy` 断言 GPT-4 仍然是 AI 模型的金标准，但预计今年会有开源模型超越它。这可能是由于开源模型的改进或 OpenAI 政策的变化。
- **Hermes 还是 Dolphin：这是个问题**: 用户 `@dagbs` 和 `@c.harl.es` 分别暗示 Llama 3 和 OpenHermes 等其他模型可能会超越 GPT-4。

**提到的链接**:

[TheBloke/MegaDolphin-120b-GGUF · Hugging Face](https://huggingface.co/TheBloke/MegaDolphin-120b-GGUF)


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (21 messages🔥): 
        
- **多 GPU 设置的困扰**:
    - 新用户 `.telistra` 询问 LM Studio 是否支持 **多 GPU 模型加载**，因为他们无法将模型分散到其 3 个 GPU 上，软件反而倾向于使用 CPU。`heyitsyorkie` 确认理论上应该支持，但 `.telistra` 发现它只使用了一个 GPU。
- **检查 GPU 利用率**:
    - `fabguy` 提供建议，让 `.telistra` 增加 `n_gpu_layers` 并监控 GPU 在 vRAM 方面的利用率，而不是活动率。提到尽管会有一些 CPU 利用率，但 `.telistra` 应该预期 GPU 利用率约为 30% 或 CUDA 利用率达到 100%。
- **调整 JSON 设置以进行 GPU 分配**:
    - 针对 `.telistra` 关于 vRAM 利用率的问题，`fabguy` 解释说预设 JSON 中的设置可以定义每个显卡上放置多少层模型，默认情况下应该是平均分配。并指出这些设置目前还无法通过 UI 访问。
- **M3 芯片 Studio 的推测**:
    - `heyitsyorkie` 和 `rugg0064` 推测了未来潜在的 **M3 Chip Studios**，`rugg0064` 建议 `.telistra` 如果有发布的话，可能需要等待 **M3 Ultra**。
- **不需要 NVLink 桥接器**:
    - `fabguy` 向 `.telistra` 保证 **不需要 NVLink**，尽管在并行 LLM 运行方面存在限制。


### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (6 messages): 
        
- **寻找合适的 AI 模型很棘手**: 用户 `_anarche_` 讨论了他们在股票分析示例中的经验，提到即使考虑了建议的 **openhermes 2.5**，在寻找适合 function calling 的“正确模型”时仍有困难。
- **任务在一段时间后失控**: `_anarche_` 进一步指出一个问题，即模型最初执行了预期的任务，但最终开始执行其所谓的“幻觉任务”（hallucinated tasks）。
- **提供股票分析示例的帮助**: `_anarche_` 表示他们成功让股票分析示例运行起来，并愿意提供帮助。
- **并非孤军奋战**: `cyrusfirheir` 确认“默认示例”可以工作，但在涉及“多发言者”的 group chat 中遇到了问题。
- **对默认示例的困惑**: 针对 `cyrusfirheir` 的回复，`_anarche_` 询问是“哪个默认示例”在工作。


### ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 messages): 
        
- **Function Calling/工具选择的困扰**: `@anarche_` 表达了在使用开源模型和 langchain 进行 function calling/工具选择时的困难。具体的困扰似乎与**识别合适的 Agent 类型**有关，尽管尝试了多种类型但均未成功。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

- **LLM 与收敛的细微差别**：[@everlasting_gomjabbar](https://discord.com/channels/729741769192767510/729741769738158194/)、[@stellaathena](https://discord.com/channels/729741769192767510/729741769738158194/) 和 [@fedorovist](https://discord.com/channels/729741769192767510/729741769738158194/) 在 **#general** 频道展开了一场反复的辩论，讨论 Large Language Models (LLMs) 是否必然会逼近其数据集，以及它们是否能处理真正的创新示例。这次讨论可能对数据集在模型性能中的重要性产生影响。
- **深入探讨 Transformers 和 Attention**：**#research** 频道的讨论主要集中在理解 Transformers 和 Attention 机制，用户分享了宝贵的资源和参考文献。[Activation Beacon Paper](https://arxiv.org/abs/2401.03462) 因其在 LLMs 中的潜在应用而受到特别关注。
- **专家聚类与苦涩的教训**：**#interpretability-general** 频道围绕专家聚类（Expert Clustering）及其在 Machine Learning 中的影响展开，[@norabelrose](https://discord.com/channels/729741769192767510/1052314805576400977/)、[@stellaathena](https://discord.com/channels/729741769192767510/1052314805576400977/) 和 [@swyxio](https://discord.com/channels/729741769192767510/1052314805576400977/) 提供了发人深省的见解。
- **lm-thunderdome 中的模型开发挑战**：**#lm-thunderdome** 频道的对话集中在与数据集转换、device_map 选项识别以及停止序列（stop sequences）处理等相关的挑战和创新解决方案上。[一个增加版本号的脚本](https://github.com/EleutherAI/lm-evaluation-harness/pull/1268) 的可能性就是其中一个有趣的变通方案。
- **Pile 数据集难题**：围绕 Pile 数据集可用性的不确定性成为 **#gpt-neox-dev** 频道的热门话题，[@stellaathena](https://discord.com/channels/729741769192767510/730090096287547444/) 分享了当前的 [DMCA 请求](https://www.eleuther.ai/hashes) 以及 Eleuther AI 对长期解决方案的探索。


**Eleuther 频道摘要**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (89 条消息🔥🔥): 
        
- **数据集决定 LLM 性能**：`@everlasting_gomjabbar` 发起了一场关于某种说法的讨论，即无论架构如何，Large Language Models (LLMs) 最终都会收敛并趋近于其训练数据集。他们认为这意味着 LLM 的强大程度取决于其训练的数据集，但难以找到该理论的原始出处。
  
- **LLM 与难以捉摸的收敛性**：`@stellaathena` 反驳了关于 LLM 和数据集的说法，指出 LLM 并不总是收敛到相同的 loss，且由于架构决策，模型可能会达到较差的 test loss 结果。此外，他们断言收敛*速率*可能更为重要，因为 LM 并非训练到完全收敛。

- **探索模型在面对新颖输入时的行为**：`@everlasting_gomjabbar` 提出了 LLM 在面对训练数据集中未充分体现的真正新颖的示例或概念时所面临的困难。`@fedorovist` 认为，目前的模型可能仍然缺失一些关键要素，无法实现像人类那样强大的泛化能力。

- **寻找基准模型**：`@.the_alt_man` 正在寻求有关在相对较大的数据集（750M+ tokens）上训练约 150M 参数的小型模型的论文或项目建议，并要求提供完整的训练指标。他们特别希望能看到带有 `loss` 或 `top-k` 准确率的 WandB 日志，最终收到了 `@ad8e` 关于其 10M 基准模型的回复。
  
- **质疑 Noam Chomsky 对 AI 的观点**：多位用户辩论了由 `@everlasting_gomjabbar` 分享的 Noam Chomsky 对人工智能的看法。Chomsky 因建议智能系统应该能够解释什么*不是*事实，以及什么*可能*和*不可能*发生而受到批评。一些用户认为这种期望是不合理的，并主张重点应转向理解和改进 AI 的现有能力。


**相关链接**：

- [BCI 与模块化思维生态系统](https://www.beren.io/2023-04-23-Composable-latent-spaces-BCIs-modular-minds/)：认识论状态：比之前的文章更具推测性，但指向了未来一个正变得越来越清晰的方面，我认为目前这一点尚未得到充分重视。如果你对...感兴趣。
- [RotatE：通过复数空间中的关系旋转进行知识图谱嵌入](https://arxiv.org/abs/1902.10197)：我们研究了学习知识图谱中实体和关系的表示以预测缺失链接的问题。这项任务的成功在很大程度上取决于建模的能力...
- [AI 模型中的“核心”是数据集 — Non_Interactive — Software &amp; ML](https://nonint.com/2023/06/10/the-it-in-ai-models-is-the-dataset/)
- [相对表示实现零样本潜空间通信](https://openreview.net/forum?id=SrC-nwieGJ)：相对表示可用于解决有关“潜空间通信”的任务：从零样本模型缝合到不同设置之间的潜空间比较。
- [GitHub - ad8e/TinyStories-cleaner：移除带有杂乱 Unicode 字符的生成故事](https://github.com/ad8e/TinyStories-cleaner)：移除带有杂乱 Unicode 字符的生成故事 - GitHub - ad8e/TinyStories-cleaner
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/wqn741y9?workspace=user-ad8e)：Weights & Biases，机器学习开发者工具
- [ad8e](https://wandb.ai/ad8e/tinystories3/runs/wqn741y9/files/code/_session_history.ipynb)：Weights & Biases，机器学习开发者工具
- [与大语言模型进行角色扮演 - Nature](https://www.nature.com/articles/s41586-023-06647-8)：通过从角色扮演的角度来刻画基于大语言模型的对话 Agent 行为，可以描述对话 Agent 的行为，例如（表面的）欺骗和...
- [模拟器 — LessWrong](https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators)：感谢 Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, 和 Jonathan Low 对草案的反馈...

### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (40 条消息🔥): 
        
- **为增长而生？理解 Transformer 和 Attention 机制**：用户 `@erich.schubert` 向社区征求关于 Transformer 和 Attention 的理论基础知识、优缺点以及背景能力的文献推荐。
- **转换概率**：`@mrgonao` 发起了一项讨论，探讨 LLM 接收 Token 的概率分布作为输入而非坚持标准的 1hot 格式的可行性；`@thatspysaspy` 给予了肯定回答，并提供了相关前人工作的参考资料。
- **优雅的灯塔**：`@carsonpoole` 重点介绍了 [Activation Beacon 论文](https://arxiv.org/abs/2401.03462)，强调了其在 LLM 实际应用中的潜力，并称赞该方法既优雅又实用。
- **不同种子下的集成性能**：受 "Blending is All you Need" 论文启发，`@jstephencorey` 开启了关于不同种子的模型集成性能的对话。`@maxmatical` 推荐了[一项工作](https://arxiv.org/abs/2203.05482)，内容涉及使用不同种子集成 LLM。
- **探索 Pythia**：`@eduardoslonski` 注意到了一个在 Pythia 上表现尤为强烈的有趣现象，并在 [一篇 Twitter 帖子](https://vxtwitter.com/EduardoSlonski/status/1745130935727894616) 中分享了完整解释，同时提到在探索过程中使用的研究工具采用了 `React and Flask`。

**提到的链接**：

- [AUTOACT: Automatic Agent Learning from Scratch via Self-Planning](https://arxiv.org/abs/2401.05268)：语言 Agent 在各种复杂任务中取得了显著性能。尽管在该领域不断探索，现有的语言 Agent 系统仍面临成本高昂、不可复现等挑战……
- [Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)：最大化模型准确率的传统方法是 (1) 使用各种超参数训练多个模型，(2) 选择在留出验证集上表现最好的单个模型……
- [Understanding the Latent Space of Diffusion Models through the Lens of Riemannian Geometry](https://arxiv.org/abs/2307.12868)：尽管 Diffusion Models (DMs) 取得了成功，我们对其潜空间仍缺乏深入理解。为了理解潜空间 $\mathbf{x}_t \in \mathcal{X}$，我们从几何角度对其进行分析……
- [Turing Complete Transformers: Two Transformers Are More Powerful...](https://openreview.net/forum?id=MGWsPGogLH)：本文介绍了 Find+Replace Transformer，这是一系列多 Transformer 架构，可以证明完成单个 Transformer 无法完成的任务，并在多个挑战性任务上优于 GPT-4……
- [Tweet from Aran Komatsuzaki (@arankomatsuzaki)](https://fxtwitter.com/arankomatsuzaki/status/1745271296437469195)：推理步骤长度对 Large Language Models 的影响。在 “Let’s think step by step” 之后添加 “you must think more steps” 会增加推理步骤并显著提高……


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (3 条消息): 
        
- **质疑专家沿语义路线聚类**：`@norabelrose` 提出了一个问题，即为什么专家（experts）会倾向于沿语义路线聚类，并表示事后看来这一点并不明确。
- **强制专家专业化**：`@stellaathena` 建议，通过强制手段使模型按专家进行专业化可能只会带来极小的性能损失，探索这种可能性可能会大有裨益。
- **机器学习的酸涩教训**：`@swyxio` 对比了人类和机器的学习过程，指出人类在学习中自然倾向于专业化，而这种倾向并不能转化到机器学习中。他们将其称为 “酸涩的教训 (the Sour Lesson)”。

### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (35 条消息🔥): 
        
- **AGIEval 转换为 HF 数据集的问题**：`@hailey_schoelkopf` 就 AGIEval 样本清洗后将其转换为 HF 数据集的问题向 `@dmayhem` 寻求帮助。`@dmayhem` 迅速响应并提供了协助。

- **HF Transformers 中的 device_map 选项**：在 `@hailey_schoelkopf` 和 `@stellaathena` 的讨论中，双方注意到 **HF Transformers** 或 **AutoGPTQ** 中的 `device_map` 选项在各自的代码库或文档页面中并不容易识别。

- **TriviaQA 中的停止序列（stop sequences）问题**：`@baber_` 和 `@hailey_schoelkopf` 发现了 **TriviaQA** 中停止序列处理的潜在问题，这可能与多标记停止序列（multi-token stopsequence）代码对 `\n\n` 的不同 Tokenize 方式有关。`@hailey_schoelkopf` 将该问题关联到了 GitHub 上提出的一个类似问题，并建议通过一个待处理的 PR 进行修复。

- **通过脚本增加版本号的可能性**：考虑到停止序列带来的复杂性，`@stellaathena` 建议开发一个脚本，在每次发生类似问题时，能够自动增加每个 Benchmark 的版本号。

- **策划的评估数据集存储**：针对 `@hyperion.ai` 关于评估数据集可复现性的询问，`@stellaathena` 回复称他们存储了评估数据集的哈希值（hashes）以缓解此类问题。`@hyperion.ai` 进一步建议为策划的评估数据集建立永久存储。

**提到的链接**：

- [lm-evaluation-harness/lm_eval/utils.py at 692e0f83b5341b543fa288f84289617f793e4e93 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/692e0f83b5341b543fa288f84289617f793e4e93/lm_eval/utils.py#L646)：一个用于自回归语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
- [KeyError: 'Cache only has 0 layers, attempted to access layer with index 0' · Issue #1250 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1250#issuecomment-1884378543)：尝试测试 autogptq LLAVA 模型的性能时遇到此错误。由于 LLAVA 是一个 VLM 模型，手动将 config 中的 model_type 更改为 llama 后允许模型加载...
- [BBH, gsm8k benchmark accuracy mismatch with paper · Issue #1075 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/1075#issuecomment-1868547567)：感谢你们的出色工作！我在 LLAMA 2 7B 的 BBH few-shot 设置（含/不含 COT）中分别获得了 23.4 和 15.1 的分数。然而 Llama 论文称其 BBH 达到 32，且 gsm8k 的准确率也不匹配...
- [Fix bug in multi-token Stop Sequences by haileyschoelkopf · Pull Request #1268 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1268)：修复了 #1262。待办事项：由于此修复，需要提升 generate_until 任务的版本。


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (3 条消息): 
        
- **查找 Pile 的验证集和测试集**：`@pietrolesci` 寻求查找 Pile 数据集验证集和测试集的指导，但在 Hugging Face 或 Pile 网站上难以找到。
- **Pile 因 DMCA 请求下线**：`@stellaathena` 告知 Pile 目前因 DMCA 请求而下线，Eleuther AI 正在寻求长期解决方案。
- **提供 Pile 数据集测试服务**：`@stellaathena` 慷慨地提出，可以为那些有兴趣在 Pile 验证集或测试集上评估模型的人运行并报告结果。
- **验证 Pile 副本的真实性**：如果有人找到了 Pile 的本地副本，`@stellaathena` 建议使用提供的 [hashes](https://www.eleuther.ai/hashes) 来确认其身份。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **诡异还是艺术？关于 Pseudoterminalx 模型结果的辩论**：用户 `@pseudoterminalx` 分享了其 AI 模型生成的图像，引发了用户之间的意见分歧，例如 `@thejonasbrothers` 认为这些图像处于“恐怖谷（uncanny valley）”，而其他人则持不同意见。
- **结合 DDPO 的实时 RLHF 模型是否可行？**：在 `@thejonasbrothers` 和 `@qwerty_qwer` 之间的一场激烈的讨论中，他们强调虽然目前还没有能够使用 DDPO 实现实时的 RLHF 模型，但这种实现在理论上是可能的。
- **艺术与 AI 的交汇——田纳西州州长采取行动保护艺术家的声音**：用户 `@thejonasbrothers` 分享了一篇[新闻文章](https://finance.yahoo.com/news/tennessee-governor-music-leaders-launch-231255716.html)，内容关于田纳西州州长努力保护艺术家免受 AI 潜在威胁。人们对这一举措的看法褒贬不一。
- **CoGVLM 对决 Share-GPT4v：巅峰对决！**：由 `@qwerty_qwer` 发起的关于 CoGVLM 和 Share-GPT4v 模型对比的辩论以 CoGVLM 胜出告终，`@thejonasbrothers` 分享称该模型可以在 16GB 显存内高效运行。
- **消失的服务器：AI-Explained Discord 服务器失踪**：AI-Explained Discord 服务器的突然消失引起了用户的困惑，`@realz` 对此表示不解。目前情况尚不明确。
- **MirrorDiffusion：Zero-shot 图像翻译**：用户 `@vrus0188` 展示了 [MirrorDiffusion](https://github.com/MirrorDiffusion/MirrorDiffusion)，这是一个旨在实现 Zero-shot 图像到图像翻译的 Diffusion 模型，引起了研究界的关注。
- **EmoGen：让情感主导生成**：`@vrus0188` 分享了一篇[论文](https://arxiv.org/abs/2401.04608)，介绍了一种名为 EmoGen 的新模型，该模型使用预定义的情感类别生成语义清晰、情感忠实的图像。
- **量化 Diffusion 模型的内存效率**：在另一条信息中，`@vrus0188` 分享了一篇[论文](https://arxiv.org/abs/2401.04339)，探讨了量化 Diffusion 模型的微调，重点关注 PEQA、Q-Diffusion 和 DreamBooth 模型。
- **Dr2Net：更低内存占用的微调**：`@vrus0188` 分享了一篇关于动态可逆双残差网络（Dr2Net）的[论文](https://arxiv.org/abs/2401.04105)，这是一项令人兴奋的新技术，有望在减少内存消耗的同时实现高效微调。
- **公平采样与 Diffusion 模型的结合**：随着一种公平感知采样方法的引入，Diffusion 模型中样本分布的公平性得到了重新审视，`@vrus0188` 分享的[论文](https://arxiv.org/abs/2401.03140)中详细介绍了这一点。
- **SDXL 迈向新高度**：针对 SDXL 的计算需求挑战，`@vrus0188` 分享的一篇[论文](https://arxiv.org/abs/2401.02677)提出了两个缩减版变体，提供了管理该模型庞大需求的方法。
- **Scale Crafter 的难题**：`@chad_in_the_house` 透露了正在进行的关于 Scale Crafter 的 2048x2048 PR 实验的线索，但也承认缺乏 Windowed Attention。
- **攀登 Datacomp 的巅峰——LFQ 训练**：`@chad_in_the_house` 向其他研究人员提出了一个针对性的问题，询问尝试 magvit2 中引入的 LFQ 训练的动机，从而引发了关于在 Datacomp 上对其进行扩展的好奇。

**LAION 频道摘要**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (151 条消息🔥🔥): 
        
- **对 Pseudoterminalx 模型生成的混合评价**：`@pseudoterminalx` 讨论了其模型生成的图像，称其未经过“平滑处理”。关于生成图像的质量存在不同意见，`@thejonasbrothers` 认为这些图像让他感到“恐怖谷效应”，而其他人则对输出结果表示赞赏。 

- **关于 RLHF 模型和 DDPO 的辩论**：
    - `@thejonasbrothers` 和 `@qwerty_qwer` 辩论了 AI 模型是否可以拥有实时 RLHF (Reinforcement Learning from Human Feedback)。讨论结果显示目前尚无现成实现，但该概念在理论上通过 DDPO (Diffusion Direct Preference Optimization) 是可行的。

- **田纳西州保护艺术家声音的法律**：`@thejonasbrothers` 分享了一篇关于田纳西州州长 Bill Lee 提出的立法提案的新闻文章，旨在保护艺术家的声音免受人工智能潜在危险的影响。一些成员对该州监管 AI 的努力表示不屑。

- **CoGVLM 与 Share-GPT4v 的比较**：`@qwerty_qwer` 询问了 CoGVLM 和 Share-GPT4v 模型哪个更好。`@thejonasbrothers` 更倾向于 CoGVLM，并提到它可以在 16GB 显存下运行。

- **AI-Explained Discord 服务器消失**：`@realz` 对 AI-Explained Discord 服务器从其列表中消失表示困惑。讨论期间没有人提供相关信息。

**提到的链接**：

- [SDXL DPO - fffiloni 开发的 Hugging Face Space](https://huggingface.co/spaces/fffiloni/sdxl-dpo)
- [Pixart-α - PixArt-alpha 开发的 Hugging Face Space](https://huggingface.co/spaces/PixArt-alpha/PixArt-alpha)
- [Kabangu Upset GIF - Kabangu Upset Annoyed - 发现并分享 GIF](https://tenor.com/view/kabangu-upset-annoyed-gif-14814728)：点击查看 GIF
- [田纳西州州长及音乐领袖发起行动，保护词曲作者及其他艺术家免受 AI 侵害](https://finance.yahoo.com/news/tennessee-governor-music-leaders-launch-231255716.html)：田纳西州州长 Bill Lee 周三公布了新立法，旨在保护词曲作者、表演者和其他音乐行业专业人士免受人工智能的潜在危险...
- [sd_dreambooth_extension/dreambooth/train_dreambooth.py at main · RossM/sd_dreambooth_extension](https://github.com/RossM/sd_dreambooth_extension/blob/main/dreambooth/train_dreambooth.py#L1639)：通过在 GitHub 上创建一个账户，为 RossM/sd_dreambooth_extension 的开发做出贡献。

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (16 messages🔥): 
        
- **MirrorDiffusion 介绍**: 用户 `@vrus0188` 分享了 [MirrorDiffusion 在 GitHub 上的链接](https://github.com/MirrorDiffusion/MirrorDiffusion)，这是一个零样本（zero-shot）图像到图像转换的 Diffusion Model。
- **EmoGen: 利用文本到图像扩散模型生成情感图像内容**: `@vrus0188` 关注了一篇 [论文](https://arxiv.org/abs/2401.04608)，该论文提出了一项新任务，即利用情感类别生成语义清晰且情感忠实的图像。
- **使用量化扩散模型进行内存高效的个性化**: 用户 `@vrus0188` 提供了一篇 [论文](https://arxiv.org/abs/2401.04339) 的链接，该论文探索了量化 Diffusion Model 的微调领域，重点关注 PEQA、Q-Diffusion 和 DreamBooth 模型。
- **Dr2Net: 内存高效微调技术**: `@vrus0188` 分享了一篇 [论文](https://arxiv.org/abs/2401.04105)，介绍了动态可逆双残差网络（Dynamic Reversible Dual-Residual Networks, Dr2Net），该技术能在显著降低内存消耗的情况下微调预训练模型。
- **通过切换机制在扩散模型中实现公平采样**: `@vrus0188` 分享了一篇 [论文](https://arxiv.org/abs/2401.03140)，介绍了一种针对 Diffusion Model 的公平性感知采样方法。
- **使用层级损失对 Stable Diffusion XL 进行渐进式知识蒸馏**: `@vrus0188` 分享了一篇 [论文](https://arxiv.org/abs/2401.02677)，该论文介绍了 Stable Diffusion XL (SDXL) 的两个缩减变体，旨在有效解决 SDXL 模型的计算需求。
- **Scale Crafter 实验**: 用户 `@chad_in_the_house` 提到正在处理一个关于 2048x2048 的 scale crafter PR，但提到目前没有 windowed attention，暗示实验仍在进行中。
- **LFQ 训练查询**: `@chad_in_the_house` 询问是否有动力在 datacomp 上大规模尝试 magvit2 中引入的 LFQ 训练。

**提到的链接**:

- [Fair Sampling in Diffusion Models through Switching Mechanism](https://arxiv.org/abs/2401.03140): Diffusion Model 通过很好地逼近潜在概率分布，在生成任务中展示了其有效性。然而，已知 Diffusion Model 存在放大的固有...
- [Memory-Efficient Personalization using Quantized Diffusion Model](https://arxiv.org/abs/2401.04339): 像 Stable Diffusion XL、Imagen 和 Dall-E3 这样十亿参数级 Diffusion Model 的兴起，显著推动了生成式 AI 领域的发展。然而，它们的大规模特性给微调带来了挑战...
- [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677): Stable Diffusion XL (SDXL) 因其多功能性和顶尖的图像质量，已成为最佳的开源文本到图像模型 (T2I)。有效解决 SDXL 模型的计算需求是...
- [Dr$^2$Net: Dynamic Reversible Dual-Residual Networks for Memory-Efficient Finetuning](https://arxiv.org/abs/2401.04105): 大型预训练模型在现代计算机视觉任务中日益重要。这些模型通常通过端到端微调用于下游任务，这对于任务来说是非常耗费内存的...
- [EmoGen: Emotional Image Content Generation with Text-to-Image Diffusion Models](https://arxiv.org/abs/2401.04608): 近年来，图像生成任务取得了显著进展，用户可以创建视觉效果惊人的高质量图像。然而，现有的文本到图像 Diffusion Model 在...
- [GitHub - MirrorDiffusion/MirrorDiffusion: zero-shot image-to-image translation, diffusion model, prompt, image-to-image translation](https://github.com/MirrorDiffusion/MirrorDiffusion): 零样本图像到图像转换，Diffusion Model，prompt，图像到图像转换 - GitHub - MirrorDiffusion/MirrorDiffusion...

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- **调整高分辨率图像大小**：@hoangtnm 询问了使用高分辨率图像进行高分辨率输出的问题。`@meatfucker` 澄清说，图像会被调整大小以适应模型的内部尺寸。
- **TTS 生成模型和应用**：在 `@funapple` 提问后，大家讨论了用于 TTS 生成的最佳模型和 GUI 应用。`@not_lain` 推荐了 Whisper 或 Seamless。
- **HuggingChat 上的模型退役**：`@green_eye` 询问关于 Falcon 模型从 HuggingChat 消失的问题，`@Cubie | Tom` 回答说旧模型通常会为新模型让路。
- **Gradio 入门**：`@eddyizm` 分享了他们的 Gradio 入门之旅，特别是为单选按钮添加默认配置以及在点击时更新按钮。
- **NLP 课程推荐**：`@muhammadmehroz` 询问关于 Hugging Face 其他课程的问题，`@cloudhu` 提供了 Hugging Face [NLP Course](https://huggingface.co/learn) 的链接。
- **发现 Face AdapterID**：`@merve3234` 分享了 [Face AdapterID demo](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID) 这一令人兴奋的发现，并提到其令人上瘾的 zero-shot 特性。
- **OpenChat 3.5 超越 Grok**：`@imonenext` 发布了 OpenChat-3.5 Update 0106，它在所有四个基准测试中均优于 Grok-0 (33B)，且在平均分和 3/4 的基准测试中优于 Grok-1。他们分享了 [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106)、[在线演示网站](https://openchat.team) 和 [GitHub](https://github.com/imoneoi/openchat) 上的更新链接。
- **CodeChat 项目介绍**：`@domestos70` 介绍了 CodeChat，这是一个允许在浏览器中与 CSV 交互的新项目，并分享了 GitHub 仓库[链接](https://github.com/tomasz-kielbasa/codechat)和[在线演示](https://codechat-six.vercel.app/)。
- **数据处理瓶颈**：`@sayakpaul` 呼吁深入探讨在 Diffusion 模型背景下与批量推理（batched inference）相关的性能问题原因。
- **Diffusion 模型基准测试**：`@raphael6419` 在 [GitHub 仓库](https://github.com/oOraph/diffusers-benchmark)中分享了他们对 Diffusion 模型的实际基准测试结果。
- **VRAM 限制 Batch Size**：`@raphael6419` 提供了关于 Batch Size 参数的见解，其中 VRAM 的可用性是一个关键考量因素。
- **SDXL 技术报告发布**：`@lunarflu` 分享了一份关于 Stable Diffusion XL (SDXL) 及其缩小版本的[技术报告](https://arxiv.org/abs/2401.02677)。
- **家庭自动化 AI**：`@gamemaster123356` 建议了一个使用 LLaMa 构建的使用 Google 搜索的家庭自动化 AI 用例。`@sayakpaul` 将讨论引导至了相应的频道。
- **Gradio Lite 发布**：用户 `@yuviii_` 宣布了由 Gradio Lite 驱动的 Gradio 4，它现在完全在 Web 浏览器中运行，从而实现更快、更私密的 Serverless 应用。详细信息和使用指南可以在 [Gradio Lite 官方页面](https://gradio.app/guides/gradio-lite)找到。

**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (40 messages🔥): 
        
- **高分辨率图像会被调整大小**：`@hoangtnm` 询问关于使用更高分辨率的图像以获得高分辨率输出的问题，`@meatfucker` 回复称图像会被调整为模型的内部尺寸。
- **TTS 生成的首选应用和模型**：`@funapple` 询问了用于 TTS 生成的最佳 GUI 应用和模型。作为回应，`@not_lain` 建议使用 Whisper 或 Seamless，尽管他们不确定是否有用于语音相关任务的 GUI 应用。
- **从 HuggingChat 中移除 Falcon 模型**：`@green_eye` 询问了 Falcon 模型从 HuggingChat 中移除的问题。`@Cubie | Tom` 回应表示，旧模型通常会被新模型取代。
- **模型显存消耗可视化**：`@.martineden` 正在寻找一个能显示选择量化（quantization）类型后模型显存需求的 HuggingFace Space。`@michielo` 分享了 [huggingface.co model memory usage space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)。
- **微调 Distilbert-Base-Uncased 模型需要帮助**：`@heromnxpw0` 请求帮助解决在为电影摘要数据集微调 distilbert-base-uncased 模型时遇到的错误。然而，他们需要一个分享代码的地方。`@jo_pmt_79880` 建议他们在 [discord 频道](https://discord.com/channels/879548962464493619/1019883044724822016) 中分享。


**提及的链接**：

- [@Tonic on Hugging Face: &quot; 🙋🏻‍♂️hey there folks , 🌟Tonic here
- just a 🛠️builder from 🗼Paris !…&quot;](https://huggingface.co/posts/Tonic/802671427380916)
- [Model Memory Utility - a Hugging Face Space by hf-accelerate](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)
- [Understanding pipelines, models and schedulers](https://huggingface.co/docs/diffusers/using-diffusers/write_own_pipeline)


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (5 messages): 
        
- **探索 Gradio**：`@eddyizm` 正在学习如何为单选按钮添加默认配置，并使用 Gradio 在点击时更新按钮。这是他们第一次使用该库，是一次有趣的体验。
- **Hugging Face 的其他课程**：`@muhammadmehroz` 询问了 Hugging Face 其他课程的情况。`@cloudhu` 提供了 Hugging Face [NLP Course](https://huggingface.co/learn) 的链接，该课程教授如何使用 HF 生态系统中的库进行自然语言处理。
- **寻求深度学习下一步学习建议**：`@sebastian3079` 在完成 Andrew Ng 的深度学习专项课程后，正在寻求关于接下来自然应该学习什么的建议。

**提及的链接**：

[Hugging Face - Learn](https://huggingface.co/learn)


### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (3 messages): 
        
- **发现并喜爱 Face AdapterID**：用户 `@merve3234` 分享了他们对 [Face AdapterID demo](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID) 的兴奋发现。他们将其描述为“纯粹的沉迷”，并强调这是一个 **zero-shot 模型**；你只需要上传图像并输入 prompt。
- **为模型适配器点赞**：`@jo_pmt_79880` 对该模型适配器表示了赞赏。
- **SDXL 变体效果良好**：`@meatfucker` 分享了他们使用该模型的 **sdxl 变体** 进行的成功实验。

**提及的链接**：

[IP-Adapter-FaceID - a Hugging Face Space by multimodalart](https://huggingface.co/spaces/multimodalart/Ip-Adapter-FaceID)

### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (5 条消息): 
        
- **OpenChat 3.5 表现优于 Grok**：`@imonenext` 宣布发布 OpenChat-3.5 Update 0106，该版本在所有四个基准测试中均超越了 Grok-0 (33B)，且平均水平和 3/4 的基准测试中超过了 Grok-1。据报告，此次更新增强了训练方法论、上下文学习（in-context learning）和编程技能。该模型已在 [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106)、[在线演示网站](https://openchat.team)和 [GitHub](https://github.com/imoneoi/openchat) 上发布。部署说明可以在该项目的 [GitHub 页面](https://github.com/imoneoi/openchat)上找到。
- **CodeChat 介绍**：`@domestos70` 介绍了一个名为 CodeChat 的项目，允许你在浏览器中与 CSV 进行交互。该项目已在 [GitHub](https://github.com/tomasz-kielbasa/codechat) 上线，并提供 [在线演示](https://codechat-six.vercel.app/)。
- **德国聊天模型 Phoenix**：`@drxd1000` 发布了一个名为 Phoenix 的新型德国聊天模型，该模型采用了直接偏好优化（Direct Preference Optimization, DPO）进行训练。Phoenix 的模型卡片可以在 [HuggingFace](https://huggingface.co/DRXD1000/Phoenix) 上找到。
- **MermaidMistral 反馈请求**：`@troyfix` 征求对他创建的 MermaidMistral 模型的评价和反馈。他还发布了一篇 [Reddit 帖子](https://www.reddit.com/r/Oobabooga/comments/192qb2c/mermaidmistral_a_work_in_progress_model_for_flow/?rdt=56518)，讨论了该模型以及微调模型名称反映其能力的重要性。
- **分享 YouTube 链接**：`@pradeep1148` 分享了一个 [YouTube 链接](https://www.youtube.com/watch?v=oflRFnG2j3k)，但未提供背景信息或解释。

**提到的链接**：

- [DRXD1000/Phoenix · Hugging Face](https://huggingface.co/DRXD1000/Phoenix)
- [Reddit - 深入探讨任何事物](https://www.reddit.com/r/Oobabooga/comments/192qb2c/mermaidmistral_a_work_in_progress_model_for_flow/?rdt=56518)
- [GitHub - tomasz-kielbasa/codechat: 在浏览器中与 CSV 交互](https://github.com/tomasz-kielbasa/codechat)：在浏览器中与 CSV 交互。通过在 GitHub 上创建账号为 tomasz-kielbasa/codechat 的开发做出贡献。
- [CodeChat](https://codechat-six.vercel.app/)
- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)
- [Chatbot UI](https://openchat.team)
- [GitHub - imoneoi/openchat: OpenChat: 利用不完美数据推进开源语言模型](https://github.com/imoneoi/openchat)：OpenChat: Advancing Open-source Language Models with Imperfect Data - GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 条消息🔥): 
        
- **检查推理瓶颈**：`@sayakpaul` 呼吁调查批量推理（batched inference）中的性能问题是由库引起的还是外部因素导致的。
- **Diffusers 基准测试**：`@raphael6419` 分享了一个 [GitHub 链接](https://github.com/oOraph/diffusers-benchmark)，详细介绍了他们对 Diffusers 性能的调查结果。
- **Batch Size 限制**：`@raphael6419` 提到，他们受到 GPU 模型可用 VRAM 量的限制，在使用 SD 1.5 生成 512x512 图像时，Batch Size 无法超过 16。
- **SSD-1B 资源**：`@lunarflu` 提供了一个关于 Stable Diffusion XL (SDXL) 技术报告的 [链接](https://arxiv.org/abs/2401.02677)，其中介绍了更小的模型变体。
- **开发基于命令的 AI**：`@gamemaster123356` 表示有兴趣创建一种可以与电脑交互并控制家庭的文本生成 AI，并寻求关于如何将 Google 搜索功能与 LLaMa 集成的指导。`@sayakpaul` 建议咨询其他频道以获取有关文本生成的建议。
- **优化 Diffusion 模型**：`@sayakpaul` 分享了一系列关于提高 Diffusion 模型推理延迟（inference latency）的论文 [合集](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d)。

**提到的链接**：

- [使用层级损失对 Stable Diffusion XL 进行渐进式知识蒸馏](https://arxiv.org/abs/2401.02677)：Stable Diffusion XL (SDXL) 因其多功能性和顶级的图像质量已成为最好的开源文本生成图像模型 (T2I)。有效解决 SDXL 模型的计算需求是……
- [优化 Diffusion 模型 - sayakpaul 合集](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d)
- [GitHub - oOraph/diffusers-benchmark](https://github.com/oOraph/diffusers-benchmark)：通过在 GitHub 上创建账号为 oOraph/diffusers-benchmark 的开发做出贡献。

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (25 messages🔥): 
        
- **探索 Text-to-Image 模型的最佳 Caption 格式**：`@pxovela` 发起了一场关于用于 Text-to-Image 基础模型的图像数据集最佳 Caption 方法的讨论。他们强调了人类对图像解释多样化带来的挑战，并指出像 Dalle3 这样经过 Fine-Tuning 的语言模型现在可以转换 Prompt，从而消除了对特定 Caption 结构的需求。`@merve3234` 承认目前关于此问题尚无共识或全面的研究。
  
- **Computer Vision 即将发布的内容**：`@merve3234` 预告了两个即将到来的与 Computer Vision 相关的模型集成，并邀请用户猜测这些模型是什么。

- **某些模型对于 Object Detection 可能大材小用**：`@merve3234` 表示，某些模型（消息中未具体说明）对于 Object Detection 来说可能大材小用且效果不佳，建议改用 Transformer 模型或 YOLO/S。

- **LLAVA 在员工监控用例中的局限性**：针对 `@iloveh8` 提出的使用 LLAVA 进行员工监控的用例，`@meatfucker` 警告该模型容易出错和产生 Hallucinations（幻觉），断言该模型在识别特定任务方面可能并不可靠。他们还指出，由于图像本身消耗了大量的语言模型 Context，LLAVA 的内部分辨率相当低。

- **关于自定义模型 Fine-Tuning 和 API 使用的求助**：`@swetha98` 寻求在自定义数据集上 Fine-Tuning Donut 文档视觉问答模型的帮助。同样，`@jordanlancheros` 在使用非开源模型 'OutfitAnyone' 的 API 时遇到问题，收到了 403 错误并请求解决方案。

**提到的链接**：

[OutfitAnyone - a Hugging Face Space by HumanAIGC](https://huggingface.co/spaces/HumanAIGC/OutfitAnyone)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **将模型迁移到 GPU**：`@vipitis` 请求关于如何将模型迁移到 GPU 的协助，`@merve3234` 指导他们在模型和输入部分添加 ```device = torch.device("cuda" if torch.cuda.is_available() else "cpu")```（例如 `model = AutoModel.from_pretrained("suno/bark-small").to(device)` 和 `...to(device)`）。
- **数据集格式验证**：`@notooth` 分享了他们正在处理的一个数据集，用于训练 `llama.cpp` 模型以获取 HTML 标签的 href 和文本，并询问该数据集格式是否正确。 
- **T5-V1_1_base Fine-Tuning 脚本的问题**：用户 `@opencuiguy` 在尝试保存被标记为非连续 Tensor 的内容并遇到 `ValueError` 后，寻求一个可用的 T5-V1_1_base Fine-Tuning 脚本。他们指出，同样的代码在 `flan-t5-base` 上可以正常运行。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (12 条消息🔥): 
        
- **调查批量推理（Batched Inference）中的性能瓶颈**：用户 `@sayakpaul` 鼓励 `<@380038770520489986>` 测试批量推理并报告他们的发现，以更好地了解性能问题是由库还是其他因素引起的。
- **分享 Diffusion Models 基准测试**：`@raphael6419` 分享了指向其 [GitHub 仓库](https://github.com/oOraph/diffusers-benchmark) 的链接，其中包括 Diffusion Models 的代码和基准测试结果，为该主题更深入的讨论铺平了道路。
- **GPU 模型的可用 Batch Size**：`@raphael6419` 分享了他们的局限性：在使用 SD 1.5 处理 512x512 图像时，由于所用 GPU 模型的 VRAM 限制，Batch Size 无法超过 16。`@sayakpaul` 随后询问了 VRAM 容量。
- **SSD-1B 技术报告**：`@lunarflu` 分享了关于 Stable Diffusion XL (SDXL) 及其缩小版变体 SSD-1B 和 Segmind-Vega 的 [技术报告](https://arxiv.org/abs/2401.02677)，讨论了它们的生成质量、参数量减少以及延迟。
- **家庭自动化 AI 模型构思**：`@gamemaster123356` 提出了创建一个能与计算机交互并控制家用电器的 AI 模型的想法。他们为 AI 响应提供了一个 System Prompt 结构，并询问如何将 Google 搜索集成到模型中。`@sayakpaul` 将进一步的讨论引导至了相应的频道。

**相关链接**：

- [Progressive Knowledge Distillation Of Stable Diffusion XL Using Layer Level Loss](https://arxiv.org/abs/2401.02677)：Stable Diffusion XL (SDXL) 因其多功能性和顶级的图像质量，已成为最好的开源文本生成图像模型 (T2I)。有效解决 SDXL 模型的计算需求是……
- [Optimizing diffusion models - a sayakpaul Collection](https://huggingface.co/collections/sayakpaul/optimizing-diffusion-models-659f481b2bb9a1311e6f845d)
- [GitHub - oOraph/diffusers-benchmark](https://github.com/oOraph/diffusers-benchmark)：通过在 GitHub 上创建账户来为 oOraph/diffusers-benchmark 的开发做出贡献。


### ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/) (1 条消息): 
        
- **Gradio 4 通过 Gradio Lite 实现纯浏览器运行**：用户 `@yuviii_` 宣布，由 `@𝚐𝚛𝚊𝚍𝚒𝚘/𝚕𝚒𝚝𝚎` 驱动的 **Gradio 4** 现在可以完全在浏览器中运行，从而能够构建更快、更私密的 Serverless 应用程序。完整的发布详情和使用指南请见 [https://gradio.app/guides/gradio-lite](https://gradio.app/guides/gradio-lite)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Monads 可能是下一个热点**：@slono 幽默地在 AI 讨论中提出了 Monads 的话题。
- **OpenAI 发布 ChatGPT Team**：@coffeebean6887 分享了 **OpenAI** 发布 [ChatGPT Team](https://openai.com/chatgpt/team) 的细节，该版本提供了具有 32K context window 的 GPT-4 访问权限以及 DALL·E 3 等工具。
- **GPT Store 的质量控制难题**：用户对最近推出的 **GPT store** 表达了批评，指出其缺乏质量控制，且由于大量同质化产品的出现可能导致用户困惑。@swyxio 分享了一条总结了这些观点的 [推文](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww)。
- **提议讨论应用层 AI**：@kbal11 建议创建一个专门讨论应用层 AI 的空间。该想法得到了 @swyxio、@swizec 和 @dsquared70 的支持。
- **Open Interpreter API 上线**：@swyxio 在[其网站](https://api.openinterpreter.com/)上宣布了 Open Interpreter API 的*发布*，该 API 能够以像素级精度定位屏幕上的视觉控件。
- **Mixture of Experts (Mixtral/Phixtral) 研讨会即将举行**：一场关于 "Mixture of Experts (包括 Mixtral/Phixtral)" 的研讨会将由 `<@206404469263433728>` 主持。活动链接在[这里](https://lu.ma/llm-paper-club)。
- **LLM Paper Club 发生变动**：@ivanleomk 分享了 LLM Paper Club 的有用链接，同时 @swyxio 建议成员在 [Lu.ma 页面](https://lu.ma/llm-paper-club)上报名。据报道发生了一项重大变化：Lu.ma 删除了循环日历功能。
- **MoE 研讨会的笔记和图表已更新**：@ivanleomk 征求关于 MoE 研讨会图表和其他更新的反馈。
- **DeepSeek 的 MoE 模型崛起**：@swyxio 分享了 DeepSeek AI 的一条推文，展示了他们的下一代 Large Language Models —— **DeepSeekMoE**。在早期实验中，它的表现与 DeepSeek 67B 相当，并远超 Gshard。
- **关于 DeepSeek MoE 模型性能的见解**：@coffeebean6887 分析了 DeepSeek 的 MoE 模型性能，观察了权衡取舍，并对模型的效率进行了评论，该模型未来可能作为 API 提供服务。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (50 messages🔥): 
        
- **是时候讨论 Monads 了**：用户 `@slono` 幽默地提议，在 AI 领域讨论 Monads 的时机已经成熟。
- **OpenAI 推出 ChatGPT Team**：用户 `@coffeebean6887` 分享了 OpenAI 最近推出的 [ChatGPT Team](https://openai.com/chatgpt/team) 计划的细节，介绍了其提供的功能，例如访问具有 32K 上下文窗口的 GPT-4 以及 DALL·E 3 等工具。
- **对 GPT Store 的批评**：针对最近推出的 GPT Store 存在一些批评，理由是缺乏质量控制，且过多的类似产品可能导致用户困惑。`@swyxio` 分享了一个表达这些担忧的推文 [链接](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww)。
- **呼吁应用层 AI 对话**：用户 `@kbal11` 提议创建一个专门讨论 AI 应用层的空间，重点关注 AI 工程师而非 ML 研究员。该想法得到了 `@swyxio`、`@swizec` 和 `@dsquared70` 等其他用户的支持和兴趣，他们还分享了潜在的讨论话题。
- **Open Interpreter API 发布**：`@swyxio` 分享了 [Open Interpreter API](https://api.openinterpreter.com/) 的发布，该 API 能够以单像素精度定位屏幕上的视觉控件。

**提到的链接**：

- [undefined](https://api.openinterpreter.com/)
- [Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team)：我们正在为各种规模的团队推出新的 ChatGPT 计划，提供安全、协作的工作空间，以便在工作中充分利用 ChatGPT。
- [Introducing the GPT Store](https://openai.com/blog/introducing-the-gpt-store)：我们正在推出 GPT Store，帮助您发现有用且受欢迎的自定义版本 ChatGPT。
- [Andrew Curran (@AndrewCurran_) 的推文](https://x.com/AndrewCurran_/status/1744923452572852608?s=20)：它应该是默认开启的，但以防万一，开关在这里；
- [killian (@hellokillian) 的推文](https://x.com/hellokillian/status/1743469389222195680?s=20)：@findmyke 哈哈非常感谢 myke！宣传视频全是用的 @rotatoapp —— 强烈推荐。
- [surya (@sdand) 的推文](https://fxtwitter.com/sdand/status/1745243861554004326?s=46&t=90xQ8sGy63D2OtiaoGJuww)：Plugins 曾有一套审核系统，这让它变得更好，因为对发布内容有限制；但为了保持质量，它本应更加严格。不需要有数百万个...
- [GitHub - SciPhi-AI/synthesizer: A multi-purpose LLM framework for RAG and data creation.](https://github.com/SciPhi-AI/synthesizer)：用于 RAG 和数据创建的多用途 LLM 框架。
- [Squish Meets Structure: Designing with Language Models](https://maggieappleton.com/squish-structure)：关于使用语言模型进行设计的挑战的演讲视频、幻灯片和文字记录。
- [Build a search engine, not a vector DB](https://blog.elicit.com/search-vs-vector-db/)：如果你想构建基于 RAG 的工具，请先构建搜索。
- [Why Chatbots Are Not the Future of Interfaces](https://wattenberger.com/thoughts/boo-chatbots)：为什么聊天机器人不是界面的未来。
- [Generative Interfaces Beyond Chat // Linus Lee // LLMs in Production Conference](https://www.youtube.com/watch?v=rd-J3hmycQs)：// 摘要：Linus 在过去几年里一直在构建和实验新型的思想工具和用于创作的软件界面，比如一个可以...


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **即将举行的 Mixture of Experts (Mixtral/Phixtral) 会议**：15 分钟后，`<@206404469263433728>` 将主持一场关于 "Mixture of Experts (包括 Mixtral/Phixtral)" 的会议。提到的活动链接在 [这里](https://lu.ma/llm-paper-club)，并包含一张 Latent Space Discord 中 LLM Paper Club 的 [封面图片](https://cdn.lu.ma/cdn-cgi/image/format=auto,fit=cover,dpr=2,quality=75,width=400,height=400/event-defaults/1-1/standard1.png)。
- **Latent Space 社区活动**：这些活动主要是每周一次的 LLM 论文回顾，从基础论文开始。论文库可以在 [这里](https://github.com/eugeneyan/llm-paper-notes/) 找到。偶尔也会举办其他活动。
- **Discord 活动通知**：用户可以要求在 `<@&1107197669547442196>` 中被标记，以接收这些活动的 Discord 通知。

**提到的链接**：

[LLM Paper Club (in Latent Space Discord) · Luma](https://lu.ma/llm-paper-club)

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (23 条消息🔥): 
        
- **LLM Paper Club 详情与注册**：@ivanleomk 告知频道，他们将在 LLM Paper Club 期间使用其 GitHub 仓库中的 `md` 文件。@intheclouddan 询问了 Club 的持续时间和频率，@swyxio 澄清说不需要预留日历时段，而是在 [Lu.ma 网站](https://lu.ma/llm-paper-club)上注册。`@swyxio` 随后透露 Lu.ma 删除了他们使用的循环日历功能，并对这一变化表示沮丧。尽管如此，他们还是在同一平台上安排了下周的活动。
  
- **笔记演示与更新**：在 Paper Club 会议结束后，@ivanleomk 分享了一个 PR，其中包含更新且整理过的笔记，以及来自 Mixture of Experts (MoE) 环节的新图表。他们欢迎对可能遗漏或错误的细节提供反馈。

- **DeepSeek 的 MoE 模型**：@swyxio 分享了来自 DeepSeek AI 的推文，介绍了他们的下一代 Large Language Models，DeepSeekMoE。该模型规模可扩展至 145B，在早期实验中显著优于 Gshard，并与 DeepSeek 67B 持平。

- **DeepSeek MoE 模型分析**：@coffeebean6887 提供了他们对 DeepSeek MoE 性能的看法，指出了该模型有趣的权衡。他们强调了其效率——2B 模型仅需 20% 的计算量，16B 模型使用 40%。他们补充说，尽管它在 benchmarks 上不是最先进的，但其效率对于作为 API 提供服务非常有益。

**提到的链接**：

- [LLM Paper Club (in Latent Space Discord) · Luma](https://lu.ma/llm-paper-club)
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)：通过 Supervised Fine-Tuning (SFT) 利用人工标注数据的力量对于推进 Large Language Models (LLMs) 至关重要。在本文中，我们深入探讨了培育强大 L...
- [Tweet from DeepSeek (@deepseek_ai)](https://fxtwitter.com/deepseek_ai/status/1745304852211839163?s=46&t=90xQ8sGy63D2OtiaoGJuww)：🌟 遇见 #DeepSeekMoE：下一代 Large Language Models！性能亮点：📈 DeepSeekMoE 2B 以 17.5% 的计算量匹配其 2B dense 对应模型。🚀 DeepSeekMoE 16B 与 LLaMA2 7B 竞争...
- [Pull requests · eugeneyan/llm-paper-notes](https://github.com/eugeneyan/llm-paper-notes/pulls)：来自 Latent Space paper club 的笔记。跟随学习或开始你自己的！ - Pull requests · eugeneyan/llm-paper-notes
- [Added some notes on Mixture of Experts by ivanleomk · Pull Request #1 · eugeneyan/llm-paper-notes](https://github.com/eugeneyan/llm-paper-notes/pull/1)：添加了一些关于 Mixture of Experts 的笔记 - 非常欢迎编辑！稍后将以此进行演示。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **对 Llama 2 漫长的微调时间感到不满**：@direwolf365 寻求减少 **Llama 2 7B 模型**预计 67 小时微调时间的建议，并讨论了他们当前的配置参数。
- **约会网站数据的秘密冒险**：@leoandlibe 提到他们正在利用一个私有数据集为虚假个人资料微调模型，该数据集源自一个类似于 Ashley Madison 的约会网站（未透露名称），每天有 10 万条消息流。
- **'Unsloth' 助力 LLM 微调**：@caseus_ 分享了一篇 [博客](https://huggingface.co/blog/unsloth-trl)，介绍了一种名为 'Unsloth' 的新型工具，它可以在不降低准确性的情况下加速 LLM 微调。据报道，它提升了 **LLaMa Factory** 的性能。
- **指令微调受困于数据集难度**：@stoicbatman 需要关于开源多模态数据集指令微调（instruct-tuning）正确数据格式的指导。
- **聊天模板中的默认系统消息缺失问题**：@le_mess 提倡在聊天模板中包含默认系统消息以改进聊天模型的训练，@dctanner 表示赞同，并指出 Huggingface 文档在系统消息方面的信息尚不完善。
- **Mistral 的问题受到关注**：在办公时间后的讨论中，@le_mess 表示已对 Huggingface trainer 在训练 Mistral 时疑似存在的问题做出了回应。
- **与 Huggingface 核实潜在故障**：@casper_ai 支持 @le_mess 关于 Huggingface 的 Mixture of Experts (MoE) 模型训练器可能存在问题的观点，并计划与 Huggingface 团队进行讨论。
- **追求 MLX**：@caseus_ 提出了将 MLX 集成到 **Axolotl** 中的设想，但尚未达成最终决定。
- **BatchSamplerDataCollatorForSeq2Seq 运行异常？**：@caseus_ 提出了 BatchSamplerDataCollatorForSeq2Seq 无法正确构建批次（batch）的问题，并引用了相关的 [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082)。
- **退出代码 -6 之谜**：@caseus_ 揭开了程序执行期间退出代码 -6 的谜团，这通常表示进程因资源耗尽而终止。
- **Checkpointing 挽救局面**：尽管面临程序意外终止，@noobmaster29 凭借高效的 checkpointing 功能成功重启了程序。
- **解码原始文本训练**：@confident_dolphin_26428 对模型在句子补全过程中如何进行原始文本训练（raw text training）表示好奇。
- **探索评估数据集**：@morgymcg 对评估数据集与模型性能之间的联系表现出兴趣，并提醒社区评估损失（evaluation loss）与模型性能之间的关联往往并不紧密。

**OpenAccess AI Collective (axolotl) 频道摘要**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (19 条消息🔥): 
        
- **Llama 2 微调时间咨询**：@direwolf365 分享了他们微调 Llama 2 7B 模型的配置细节，并向社区寻求减少预计 67 小时微调时间的建议。配置如下：数据集包含 1.9 万行，每行 4096 个 token，使用 80GB GPU，batch size: 4，Peft 方法: Qlora，量化: 4 bits，Epochs: 1，Rank: 1，Grad accum steps: 4，学习率: 0.0001，优化器: adamw_apex_fused。
- **微调约会网站数据**：@leoandlibe 透露他们如何使用一个私有数据集（源自一个类似于 Ashley Madison 的网站数据库，每天约有 10 万条消息）来为虚假个人资料进行微调。不过，他们没有透露该网站的名称。
- **Unsloth 介绍 - LLM 微调优化工具**：@caseus_ 发布了一个 [博客](https://huggingface.co/blog/unsloth-trl) 链接，介绍了一种名为 'Unsloth' 的工具，它可以加速 LLM 微调，在不降低准确性的情况下减少显存占用。据报道，LLaMA Factory 已集成 Unsloth，并见证了速度提升。
- **关于多模态数据集指令微调的问题**：@stoicbatman 向社区咨询用于开源多模态数据集指令微调的合适数据格式。

**提到的链接**：

- [使用 Unsloth 和 🤗 TRL 让 LLM 微调提速 2 倍](https://huggingface.co/blog/unsloth-trl)
- [主页](https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-comparison)：易于使用的 LLM 微调框架 (LLaMA, BLOOM, Mistral, Baichuan, Qwen, ChatGLM) - hiyouga/LLaMA-Factory

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (18 messages🔥): 
        
- **为 ChatML 模板添加默认 System Message**：用户 `@le_mess` 建议为 ChatML 的聊天模板添加默认 System Message，并指出没有模型是在没有 System Message 的情况下训练的。`@dctanner` 表示赞同，认为这可以解决他遇到的一些关于 System Message 的问题，特别是考虑到 Huggingface 文档对 System Message 的覆盖并不充分 ([Huggingface docs](https://huggingface.co/docs/transformers/main/en/chat_templating))。
- **Mistral 的 Office Hours**：`@le_mess` 宣布了讨论 Mistral 的 Office Hours。会议结束后，他提到团队怀疑 Huggingface trainer 在训练 Mixtral 时可能存在问题。
- **Huggingface Trainer 的潜在问题**：`@casper_ai` 指出 Mixture of Experts (MoE) 模型的 Huggingface trainer 确实可能存在问题，并表示打算与 Huggingface 团队讨论此事。
- **关于将 MLX 集成到 Axolotl 的讨论**：用户 `@caseus_` 提出了将 MLX 集成到 Axolotl 的问题。然而，在给出的消息中没有提供进一步的讨论或结论。
- **BatchSamplerDataCollatorForSeq2Seq 的问题**：`@caseus_` 提到了 BatchSamplerDataCollatorForSeq2Seq 可能无法正确构建 Batch 的潜在问题，并建议参考 [GitHub issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082) 在 Discord 中进行讨论。

**提到的链接**：

- [Templates for Chat Models](https://huggingface.co/docs/transformers/main/en/chat_templating)
- [BatchSamplerDataCollatorForSeq2Seq does not properly construct batches · Issue #1082 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1082)：请检查此问题之前是否已被报告。我搜索了之前的 Bug Reports，没有发现类似的报告。预期行为：Batch 的形状应为 (micro_batch_size, seque...


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (5 messages): 
        
- **解析 exit code -6**：用户 `@caseus_` 解释说，**exit code -6** 通常意味着执行任务的进程被信号 6 (SIGABRT) 终止。这通常发生在进程耗尽内存或其他资源时。
- **Checkpointing 挽救局面**：尽管没有注意到明显的内存压力，`@noobmaster29` 还是遇到了程序终止。幸运的是，Checkpointing 功能让他们能够成功重启程序。
- **理解原始文本训练**：用户 `@confident_dolphin_26428` 询问了在使用原始文本进行补全训练（completion training）时训练是如何实际运作的，质疑模型从给定句子预测第一个 Token 的过程。
- **寻找评估数据集**：`@morgymcg` 对人们目前用于评估的数据集感到好奇，并提出评估损失（evaluation loss）通常与模型的最终性能没有直接相关性。


### ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/) (1 messages): 
        
pradeep1148: https://www.youtube.com/watch?v=oflRFnG2j3k


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **矩阵中的回响引发关注**：@brknclock1215 对两个不同模型生成完全相同的回答表示担忧，思考它们是否从共享的上下文或素材中提取内容。
- **Mixtral 8x7B 的精度追求**：@simon_18724 发起了关于 **Mixtral 8x7B** 运行精度的讨论。
- **Claude 2.1 “听见”声音**：@groagji 指出 **Claude 2.1** 在解析来自 Soundcloud 的简短音乐描述时，虚构了一个不存在的来源。
- **滴答滴答，小组件即将到来**：回复 @barry_zyj 的查询，@ok.alex 确认 **Perplexity Android widget** 即将发布。
- **Perplexity vs. Copilot：独立模式更胜一筹**：@moyaoasis 表示在搜索任务中更倾向于纯净的 Perplexity 模型，而非 Copilot 版本，并指责 **Copilot with GPT4** 削弱了两个基础模型的性能。
- **Pplx-70b-online API，艺术家的梦想**：@blenderrenaissance 利用 **pplx-70b-online API** 为一个图表项目生成了准确、无幻觉的数据。
- **挖掘超链接连接**：@abhidalal 赞扬了该工具将回答与互联网结果链接起来的技巧。
- **Perplexity 初体验**：虽然新用户 @arti_xartous_14963 迫不及待地想要开始使用 Perplexity，但 @aug_li 也一致认为这是一个值得称赞的产品。
- **融化 AI 寒冬**：@.anuni 分享了一个 [Perplexity AI 链接](https://www.perplexity.ai/search/What-can-you-X31qYfR8TeuMS_.PVVUBtw?s=c)，希望能规避即将到来的 AI 寒冬。
- **Octane 信用卡支付困扰**：@yash_89789 报告在尝试向 Octane 充值时遇到信用卡被拒的问题。
- **账单问题？支持团队是答案**：@icelavaman 建议联系 support@perplexity.ai 处理账单相关问题，并强调信用卡被拒超出了 Perplexity 的管辖范围，属于支付平台 (**Stripe**) 或用户银行的问题。
   
相关探索链接：[Gangsta Chess by Renosis](https://www.thingiverse.com/thing:8930) 和 [图片库](https://discord.com/channels/1047197230748151888/1194788138124587128)。

**Perplexity AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (27 条消息🔥): 
        
- **模型间可能存在重复信息**：`@brknclock1215` 发现两个不同的模型生成了完全相同的回答，这很奇怪，并质疑模型是否在模仿作为上下文提供的相同材料或来源。
- **运行精度咨询**：用户 `@simon_18724` 询问了 **Mixtral 8x7B** 的运行精度。
- **Claude 2.1 幻觉问题**：`@groagji` 指出了 **Claude 2.1** 在处理来自 Soundcloud 音乐的有限描述时，产生虚构来源幻觉的情况。
- **即将发布 Android 小组件**：在回应 `@barry_zyj` 关于 Perplexity Android 小组件是否存在的查询时，`@ok.alex` 确认 **widget 将很快发布**。
- **相比 Copilot 更倾向于纯净版 Perplexity**：用户 `@moyaoasis` 表示在搜索任务中更倾向于纯净的 Perplexity 模型，而非 Copilot 版本，并声称 **Copilot with GPT4 降低了纯 GPT4 和纯 Perplexity 模型的性能**。


**提到的链接**：

[Gangsta Chess by Renosis](https://www.thingiverse.com/thing:8930)：这是 Gangsta Chess！现在你可以在家里发动你自己的地下犯罪地盘战争！这个设计是基于原版 gangsta 的混搭/衍生作品，由 OG 本人 Yzorg 创建。我设计了...


### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (13 条消息🔥): 
        
- **Perplexity AI 实践**：`@blenderrenaissance` 使用 **pplx-70b-online API** 为一个图表产品生成了准确、无幻觉的数据。
- **更多 Perplexity AI 的用法**：`@abhidalal` 对该工具将回答与互联网结果链接的能力表示赞赏。
- **在频道中分享图片**：`@ok.alex` 分享了一个 [图库](https://discord.com/channels/1047197230748151888/1194788138124587128) 供用户分享图片。
- **初步反应**：新用户 `@arti_xartous_14963` 对开始使用 Perplexity 表示兴奋，而 `@aug_li` 表示这是一个不错的产品。
- **AI 寒冬思考**：`@.anuni` 分享了一个 [Perplexity AI 链接](https://www.perplexity.ai/search/What-can-you-X31qYfR8TeuMS_.PVVUBtw?s=c)，希望它能帮助避免潜在的 AI 寒冬。

### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (2 条消息): 
        
- **Octane 的账单困扰**：`@yash_89789` 提出了一个关于由于卡片被拒绝而无法在 Octane 中充值的问题。
- **账单问题直接联系支持部门**：`@icelavaman` 建议用户联系 support@perplexity.ai 以解决账单问题，并强调 Perplexity **无法**解决信用卡被拒的问题，这取决于支付平台（**Stripe**）或用户的银行。


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **Langchain 中的 Token 问题**：`@_hnch` 提出了一个问题，即 Langchain 的 RetrievalQA 检索即使在 Token 计数低至 2 的情况下也会触发 **token overflow**（Token 溢出）。
- **向量数据库的内存需求**：`@__something` 询问了在桌面聊天应用中实现向量数据库对内存的影响。他们还询问了将其与 **Phi** 或 **Mistral** 等本地 LLM 合并是否可行。
- **将 LllamaCPP 接入 Langchain**：`@vamseekrishna` 寻求关于将 **LllamaCPP** 集成到 Langchain 对话式检索链中的建议，并强调文档中缺乏此类示例。
- **Langchain 新手的旅程**：初级 Python 程序员 `@ilovesass` 表现出对学习多模态模型和 **Langchain** 的兴趣。对此，`@fareswastaken` 建议了一条从基础编程开始，然后学习 Python，最后攻克 Langchain 的路径。
- **不听话的 Langchain Agents**：`@hiranga.g` 报告了一个问题，即 Langchain Agents 变得不遵守强制提示词，表现为即使被提示，Agent 仍拒绝以 **代码 '112'** 开始回复。
- **Langchain 中的并行处理**：`@shivam51` 询问是否有办法使用 **Langchain 进行并行 LLM 调用**。
- **LangChain 调试深度探索**：`@veryboldbagel` 提供了有效调试 LangChain 代码的指导，使用 print 语句检查中间输出，并额外揭示了通过 LangChain 核心 runnables 实现的调试模式。
- **注意 GitHub Query 方法问题**：`@cryptossssun` 在 LangChain GitHub 仓库中创建了一个 issue，寻求关于如何通过 query 方法使新的变量输入可用的帮助，并在此提供了链接：[GitHub Issue](https://github.com/langchain-ai/langserve/issues/393)。
- **使用 Parea AI 优化 Langchain RAG**：`@parea.ai` 发布了一份详细的[教程](https://docs.parea.ai/tutorials/getting-started-rag)，关于使用 Parea AI Evals 和 Tracelogs 优化 Langchain RAG 应用，重点关注允许用户与公共财经 PDF 文档聊天的应用。
- **Martian 也在其中吗？**：`@swyxio` 提出了一个有些晦涩的疑问，询问某个未指明的实体是否**与 'Martian' 相同**。
- **文档方面的困扰**：`@archiacme` 在使用 LangChain AI 的文档时遇到了许多错误，对所提供的安装说明的准确性提出了质疑。

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (15 条消息🔥): 
        
- **Langchain 中的 Token 过载**：用户 `@_hnch` 强调了一个问题，即 Langchain 的 RetrievalQA 检索即使在 Token 计数低至 2 时也会表现出 **token overflow**。
- **在聊天应用中整合向量数据库**：`@__something` 询问了在桌面聊天应用中使用向量数据库的内存要求。他们进一步询问了将其与轻量级本地 LLM（如 **Phi** 或 **Mistral**）结合的问题。
- **对文档的需求**：`@vamseekrishna` 寻求将 **LllamaCPP** 集成到 Langchain 对话式检索链中的指导，并提到现有文档中缺乏相关示例。
- **有抱负的 Langchain 开发者**：用户 `@ilovesass` 表达了他们目前作为 Python 初学者的状态，以及学习多模态模型和 **Langchain** 的兴趣。作为回应，`@fareswastaken` 建议从基础编程开始，转向 Python，然后攻克 Langchain。
- **Agent 遵守度问题**：`@hiranga.g` 报告了一个 Langchain Agents 不遵守强制提示词的问题。他们发现，在收到指示停止的用户输入后，Agent 停止遵循“始终以**代码 '112'**开始回复”的提示。
- **Langchain 中的并行 LLM 调用**：`@shivam51` 询问是否有使用 **Langchain 进行并行 LLM 调用**的流程。

### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (6 messages): 
        
- **在 LangChain 代码中调试 Chain 步骤**：`@veryboldbagel` 建议通过拆解 LangChain 代码来进行有效调试。他们描述了一种使用 print 语句检查中间输出的方法。提供的示例是一个代码片段，其中使用函数 `print_me` 在处理链中打印中间结果。
- **在 LangChain 中使用调试模式**：`@veryboldbagel` 还展示了如何利用 LangChain 中的调试模式进行有效排错。可以通过 `globals.set_debug(True)` 启用调试模式，并将其与 LangChain 的核心 runnables 结合使用。
- **针对新 LCEL Primitives 的测试建议**：`@veryboldbagel` 建议使用新 LCEL primitives 的用户从简单的测试用例开始，并配合 print 语句进行深入理解和调试。
- **提交关于 Query 方法帮助的 Issue**：`@cryptossssun` 提到已向 LangChain GitHub 仓库提交了一个 Issue，请求澄清如何通过 query 方法使新的变量输入可用。他们附上了 [GitHub issue 链接](https://github.com/langchain-ai/langserve/issues/393)。
- **寻求 LangChain 团队协助**：`@cryptossssun` 标记了另一位用户寻求帮助，但具体请求细节尚不明确。

**相关链接**：

[How to make the new variables input available via query method? · Issue #393 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/393): 问题：如果我创建了新的变量：input_variables=[&quot;history&quot;, &quot;input&quot;,&quot;lession&quot;, &quot;affection&quot;]，并按照以下代码进行设置。我无法使正确的 qu...


### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (2 messages): 
        
- **使用 Parea AI 优化 LangChain RAG**：`@parea.ai` 分享了一个关于如何使用 Parea AI Evals 和 Tracelogs 优化 LangChain RAG 应用的教程。该教程提供了一个逐步指南，用于优化一个**允许用户与公开财务 PDF 文档（如耐克的 10k 报表）进行对话**的应用。教程还涵盖了各种应用组件，包括 [`UnstructuredFileLoader`](https://python.langchain.com/docs/integrations/document_loaders/unstructured_file)、[`RecursiveCharacterTextSplitter`](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter)、`all-MiniLM-L6-v2` sentence transformer、作为向量数据库的 [Redis](https://redis.com/solutions/use-cases/vector-database/)、用于生成答案的 LangChain OpenAI `gpt-3.5-turbo-16k`，以及用于 Trace logs、Evaluations 和 Playground 的 Parea AI。教程可以在[这里](https://docs.parea.ai/tutorials/getting-started-rag)找到。
- **这和 Martian 类似吗？**：早些时候，`@swyxio` 询问某事（从上下文中不清楚具体指什么）是否与 **“Martian” 的想法相同**。该问题没有得到回应或后续跟进。

**相关链接**：

[Optimize a RAG application - Parea AI](https://docs.parea.ai/tutorials/getting-started-rag)


### ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/) (1 messages): 
        
- **用户在文档中遇到错误**：用户 `@archiacme` 报告在学习 LangChain AI 的文档（包括 cookbook、模板和“入门”部分）时遇到了大量错误。尽管尝试了本地执行（使用 venv 和 conda）以及在 Google Colab 上运行，这些问题依然存在，这表明所提供文档中的设置说明可能存在问题。


        

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **文本评估中 OCR 遭到冷遇**：`@evan_04487` 提出了一种无需 OCR 即可评估文本的新启发式方法。如果文档的大部分内容通过了段落结构、表格和非垃圾内容的检查，则可能不需要 OCR。
- **GPT-4 进军企业市场**：正如 `@jeffreyw128` 所分享的，OpenAI 已成功向各大公司开放了 **ChatGPT Enterprise**，并推出了新的自助服务方案 **ChatGPT Team**，提供对 **GPT-4** 和 **DALL·E 3** 等高级模型以及其他工具和功能的访问权限 —— [点击此处了解更多](https://openai.com/blog/introducing-chatgpt-team)。JefferyW128 很好奇这里提到的 **GPT-4** 是普通版还是 Turbo 版本。
- **GPT Store 与 Plugins 的对决**：针对 **GPT Store** 的亮相，`@jeffreyw128`、`@thebaghdaddy` 和 `@res6969` 就该商店相对于 Plugins 的附加价值展开了对话。鉴于 Jeffreyw128 将 GPT Store 与 Plugins 及其相关的性能问题进行了对比，共识更倾向于 Plugins 的潜在影响。`@thebaghdaddy` 略微认为自定义 GPT 可以节省编写 Prompt 指令的时间，但除此之外别无他用。
- **剖析 GPT Store 的局限性**：`@res6969` 表示，尽管 GPT Store 中 GPT 提供的 API 调用功能很有吸引力，但其可调优性（tunability）不足，这引发了关于与其他软件产品相比潜在局限性的广泛讨论。
- **GPT Store 质量担忧**：**GPT Store** 中 GPT 的质量受到了审查，`@jeffreyw128` 和 `@res6969` 得出结论，商店提供的曝光率无法弥补产品质量的低劣，这一观点得到了 `@thebaghdaddy` 的证实，他亲身体验了令人失望的质量。
- **Tree of Thought 提示词**：`emrgnt_cmplxty` 建议采用 Tree of Thought 提示词方法，但未提供进一步的细节或背景。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/) (1 条消息): 
        
emrgnt_cmplxty: Tree of Thought prompting 可能会有帮助。


### ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/) (1 条消息): 
        
- **无需 OCR 的文本评估**：`@evan_04487` 提出了一种无需使用 OCR 即可评估文本的方案。该启发式方法检查文档是否具有合法的段落文本、表格或被视为垃圾内容。如果页面达到一定比例通过这些检查，则可能不需要 OCR。


### ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/) (13 条消息🔥): 
        
- **介绍 ChatGPT Team**：`@jeffreyw128` 分享了 OpenAI 博客文章的[链接](https://openai.com/blog/introducing-chatgpt-team)，该文章透露各大公司正在使用 ChatGPT Enterprise。推出了新的自助服务方案 **ChatGPT Team**，提供对 **GPT-4** 和 **DALL·E 3** 等高级模型以及其他工具和功能的访问。JefferyW128 对所提到的 **GPT-4** 是普通版还是 Turbo 版本很感兴趣。
- **GPT Store 大辩论**：`@jeffreyw128` 对新推出的 **GPT Store** 的价值提出了质疑。将其与 Plugins 的功能进行比较，他表示由于性能问题，Metaphor 决定不投入时间。
- **GPTs vs. Plugins**：在回应 Jeffreyw128 时，`@thebaghdaddy` 认为 Plugins 具有更大的潜在影响，将自定义 GPT 视为编写 Prompt 指令的省时工具，但除此之外别无他用。
- **GPT Store 的局限**：`@res6969` 表达了同样的观点，指出虽然 GPT 的 API 调用功能很有趣，但在可调优性方面表现不佳，引发了关于 GPT 与其他软件产品相比的局限性的讨论。
- **GPT Store - 赞成还是反对？**：在讨论可能的评价系统和 GPT 的质量时，`@jeffreyw128` 和 `@res6969` 得出结论，虽然 GPT Store 提供的曝光是有益的，但产品的质量至关重要。Thebaghdaddy 亲身体验了商店中 GPT 质量不佳的情况。

**提到的链接**：

[Introducing ChatGPT Team](https://openai.com/blog/introducing-chatgpt-team)：我们正在为各种规模的团队推出新的 ChatGPT 方案，该方案提供了一个安全、协作的工作空间，以便在工作中充分利用 ChatGPT。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **首个高效 Mixture of Experts 模型 PhiXtral 亮相**：`@philipmay` 关注了一篇讨论 **PhiXtral** 的[文章](https://www.linkedin.com/posts/maxime-labonne_phixtral-i-made-the-first-efficient-mixture-activity-7150758415961620481-v0qx?utm_source=share&utm_medium=member_ios)，这是一个由多个 phi-2 模型组装而成的新型 MoE 模型。具体的组装方法及其难度仍然是大家关注的焦点。
- **德语 Mixtral 可能走 Random Forest 路线**：`@philipmay` 提出了构建“德语 Mixtral”的想法，即在不同任务上训练不同的 mistral 模型，然后将它们合并，类似于 LLM 的 Random Forest 方案。
- **MIT 授予 Phi-2 许可证升级**：`@rasdani` 告知 phi-2 已超越仅限研究的限制，现在已采用 MIT License 进行授权。
- **MoE 模型：应对 Routing 难题**：`@philipmay` 对 MoE 模型中 Routing 的复杂性表示担忧。然而，`@rtyax` 反驳称，如果场景仅包含两个专家（experts），可能不需要训练专门的 router，因为所有请求都可以同时定向到这两个专家。
- **Mixtral 实现的内部机制揭晓**：针对 `@philipmay` 关于整合多个模型以创建 MoE 的疑问，`@remek1972` 分享了关于正在进行的 `使用 mergekit 实现 Mixtral` 的见解，并提供了 mergekit 脚本的[链接](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py)。他强调了 Router 配置在模型的 mergekit 文件中的至关重要作用。
- **Conditional Pretraining 可能是 Alignment 的游戏规则改变者**：`@huunguyen` 建议采用 Conditional Pretraining 风格可以增强 Prompt Alignment。
- **“坏”人格（Personas）终究可能对我们有益**：`@huunguyen` 还建议了一种通过将共享脚本生成的“坏”人格与来自 `open-orca` 的正向人格相结合来提升性能的策略。

**DiscoResearch 频道总结**

### ▷ #[mixtral_implementation](https://discord.com/channels/1178995845727785010/1182759434326396998/) (9 条消息🔥): 
        
- **PhiXtral：首个高效 Mixture of Experts 模型**：`@philipmay` 分享了一篇[文章](https://www.linkedin.com/posts/maxime-labonne_phixtral-i-made-the-first-efficient-mixture-activity-7150758415961620481-v0qx?utm_source=share&utm_medium=member_ios)，讨论了名为 **PhiXtral** 的新型 MoE 模型的开发，该模型结合了多个 phi-2 模型。他们对具体的结合过程及其难度很感兴趣。
- **通过 LLM 的 Random Forest 探索德语 Mixtral？**：`@philipmay` 提出了一种构建“德语 Mixtral”的策略，即在不同任务上训练多个 mistral 模型并随后将其结合，类似于 LLM 的 Random Forest 方法。
- **Phi-2 获得 MIT License 状态**：`@rasdani` 告知频道 phi-2 不再仅限于研究用途；它现在已根据 MIT License 授权。 
- **构建 MoE 模型中的 Routing 挑战**：`@philipmay` 表达了对 MoE 模型中 Routing 复杂性的担忧。然而，`@rtyax` 建议如果只有两个专家，可能不需要训练专门的 router，因为所有请求都可以路由到这两个专家。 
- **Mixtral 实现见解**：`@remek1972` 回答了 `@philipmay` 关于结合多个模型创建 MoE 的查询。他提到了一个正在进行的 `使用 mergekit 实现 Mixtral` 的项目，并分享了 mergekit 脚本的[链接](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py)。此外，他强调 Router 配置在模型的 mergekit 文件中是必不可少的。

**提到的链接**：

[mergekit/mergekit/scripts/mixtral_moe.py at mixtral · cg123/mergekit](https://github.com/cg123/mergekit/blob/mixtral/mergekit/scripts/mixtral_moe.py)：用于合并预训练大语言模型的工具。 - cg123/mergekit


### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (2 条消息): 
        
- **对用于 Alignment 的 Conditional Pretraining 感兴趣**：用户 `@huunguyen` 提出了使用 Conditional Pretraining 风格训练系统以获得更好 Prompt Alignment 的想法。
- **利用“坏”人格获得更好性能**：`@huunguyen` 还分享了一个能够创建“坏”人格的脚本。通过将这些与来自 `open-orca` 的好人格整合，他认为这可能会带来**性能的提升**。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **GPT 模型堵住 Prompt 泄露漏洞**：@realgavok 称赞了当前的 GPT 模型，因为它们具有“防止 Prompt 泄露的印象深刻的措施”，并引用了反映其鲁棒性的个人测试。
- **公开呼吁 GPT 的保密策略**：@tariqali 回应称，如果这些保护策略确实如此有效，则需要公开披露模型的保护策略。
- **使用 LLM 进行深度代码库分析**：@00brad 发起了一个关于将大型代码库嵌入 LLM 的查询，思考是否应该输入每个文件、函数或类，以获得关于潜在修复或修改的见解。

**Datasette - LLM (@SimonW) 频道摘要**

### ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/) (4 条消息): 
        
- **GPT 模型实现有效的 Prompt 泄露防护**：`@realgavok` 评论说，特色 GPT 模型拥有防止 Prompt 泄露的印象深刻的措施。
- **呼吁公开 Prompt 保护策略**：作为回应，`@tariqali` 表示有兴趣了解这些保护措施，并建议如果这些策略有效，就应该公开。
- **测试中显现出 GPT 的鲁棒保护**：`@realgavok` 强调，尽管尝试了各种策略，他们仍无法绕过这些安全措施，这进一步证明了 GPT 的 Prompt 保护的鲁棒性。


### ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/) (1 条消息): 
        
- **在 LLM 中嵌入代码库**：`@00brad` 征求关于如何最好地将大量代码库嵌入 LLM 的建议。该用户正在考虑是否嵌入**每个文件、函数或类**，目标是寻求 LLM 对修复或修改代码库的见解。


        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **训练 MoEs 的精妙艺术**：`@baptistelqt` 分享了关于训练混合专家模型 (MoEs) 的见解。他们发现，在通用知识和专业知识之间保持平衡，而不是专注于绝对的专业化（80% 或更多），可以获得最佳结果。
- **基于 Token 的专业化，一个令人惊讶的转折**：在后续中，`@baptistelqt` 报告了一个反直觉的发现，即原始辅助损失 (vanilla auxiliary loss) 促进了不同类型 Token 的专业化，而不是预期的主题或领域特定学习。然而，他们强调这一结论是基于个人实验，尚未得到明确证实。
- **未识别的 YouTube 分享**：用户 `pradeep1148` 在 off-topic 频道分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=oflRFnG2j3k)，没有进一步的上下文或讨论。此内容可能与其它频道进行的讨论没有直接关系。

**Skunkworks AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (3 条消息): 
        
- **MoEs 的专业化，一种平衡行为？**：`@baptistelqt` 分享了关于训练混合专家模型 (MoEs) 的见解，指出在一个领域鼓励过多的专业化（80% 或更多）会导致性能下降。然而，通用知识和专业知识的融合似乎是有益的。
- **MoEs 中基于 Token 的专业化？没那么快**：在后续消息中，`@baptistelqt` 介绍了一个反直觉的发现：原始辅助损失 (vanilla auxiliary loss) 似乎促进了不同类型 Token 的专业化，而不是特定的主题/领域。需要注意的是，这一结论是基于个人实验，因此尚未得到明确证实。


### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 条消息): 
        
pradeep1148: https://www.youtube.com/watch?v=oflRFnG2j3k


        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- **AI 在 Candyland 创造奇幻之旅**：`@magusartstudios` 分享了他们项目的预览，涉及一个在程序化生成世界中探索的 AI，该项目由本地 Lua Chatbot 和多种外部 AI 模型驱动。该 AI 通过生成的环境音乐、表情动作（emotes）、特效和 emoji 增强了沉浸式体验。项目预览可见[此处](https://www.youtube.com/watch?v=TzdtKw1vGA0)。
- **Text-Vision 模块彻底改变 Roblox 中的 AI 交互性**：`@magusartstudios` 为 Roblox AI Agent 开发了一个开源的 Text-Vision 模块，使它们能够拥有独立的记忆和身份。关于其代码和进展的详细讨论可以在[此处](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967)找到。
- **AI Agent 在 Roblox 中通过叙事翩翩起舞**：由 `@magusartstudios` 展示，一个名为 Zephyr 7B 的 AI Agent 使用他们在 Roblox 中的 ChatModule 库，通过讲故事和跟随玩家来吸引玩家的注意。这在一段 [YouTube 视频](https://www.youtube.com/watch?v=rMBLZtPmlsQ)中进行了展示。
- **OpenChat 3.5 树立新标准**：`@imonenext` 宣布发布 **OpenChat-3.5 Update 0106**，声称在所有 4 项基准测试中性能均优于 **Grok-0 (33B)**，并且在 3/4 的基准测试中平均表现优于 **Grok-1**。
- **OpenChat 简化部署**：对于部署爱好者，`@imonenext` 在其 [GitHub 页面](https://github.com/imoneoi/openchat)上分享了使用加速的 vLLM 后端、API key 身份验证等方式部署 OpenChat 模型的完整说明。OpenChat 3.5 版本可以在[演示网站](https://openchat.team)进行实时交互，也可在 [HuggingFace](https://huggingface.co/openchat/openchat-3.5-0106) 上获取。

**Alignment Lab AI 频道摘要**

### ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/) (1 条消息): 
        
- **使用由 Lua Chatbot 和外部 AI 模型驱动的 AI 探索 Candyland**：`@magusartstudios` 讨论了他们的项目，其中一个程序化世界由 AI 进行探索，该 AI 由其本地 Lua Chatbot 和多个外部 AI 模型驱动。该 Chatbot 可以生成环境音乐、表情动作、特效和 emoji，以增加 AI 交互的表现力。该项目在一段 [YouTube 视频](https://www.youtube.com/watch?v=TzdtKw1vGA0)中展示。
- **为 Roblox AI Agent 开源的 Text-Vision 模块**：`@magusartstudios` 还提到了一款由他们开发的、为 Roblox AI Agent 准备的开源 Text-Vision 模块。它为 AI Agent 提供了独立的记忆和身份，并以 Zephyr 7B 探索 Candyland 为例进行了演示。详细的代码及其进展已在[此处](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967)讨论。
- **Roblox 中会讲故事、会跳舞的 AI Agent**：在另一个项目中，`@magusartstudios` 展示了一个会讲故事、会跳舞的 AI Agent Zephyr 7B，它使用他们在 Roblox 中的 ChatModule 库并跟随玩家。演示可见于此 [YouTube 视频](https://www.youtube.com/watch?v=rMBLZtPmlsQ)。


**提到的链接**：

- [Candyland Adventures - Lumina &amp; Darkness Feat. Zephyr 7-B Powered By Awareness Module, Memories](https://www.youtube.com/watch?v=TzdtKw1vGA0)：我正在开发一个基于《王国之心》风格世界观的程序化生成世界。这是一个 Studio 测试，并由作为聊天机器人具身化的 AI Agent 辅助……
- [Tweet from Text Vision Self-Awareness Module LLM Utility Synthetic Text Data Generator](https://devforum.roblox.com/t/text-vision-self-awareness-module-llm-utility-synthetic-text-data-generator/2536967)：我看到了这段 YouTube 视频，所以我正在编写这个模块，实现与该视频中程序员所做的类似的功能。我想把它发布在这里，为其他开发者提供灵感和资源……
- [Storytelling Dancing AI Agent Party Member ROBLOX Zephyr 7B with Emojis](https://www.youtube.com/watch?v=rMBLZtPmlsQ)：在这段视频中，我的 ChatModule 库解析了托管在 HuggingFace 上的 Zephyr 7B 在 ROBLOX 上的响应。这个 NPC 是一个跟随玩家的队伍成员……

### ▷ #[alignment-lab-announcements](https://discord.com/channels/1087862276448595968/1124055853218136175/) (1 条消息): 
        
- **OpenChat 3.5 在性能上超越 Grok**：`@imonenext` 宣布发布 **OpenChat-3.5 Update 0106**，该模型被誉为全球最佳的开源 7B LLM，在所有 4 项基准测试中均超越了 **Grok-0 (33B)**，并在平均分及 3/4 的基准测试中超越了 **Grok-1**。
- **增强的训练与技能**：此次更新改进了训练方法、In-context learning（上下文学习）以及编程技能，在 8 项基准测试中的 7 项上表现优于之前的 1210 版本。
- **多平台可用**：该模型已在他们的 [演示网站](https://openchat.team)、HuggingFace 和 [GitHub](https://github.com/imoneoi/openchat) 上线。
- **部署教程**：对于有兴趣进行部署的用户，请访问其 [GitHub](https://github.com/imoneoi/openchat) 获取完整指南，了解如何使用加速的 vLLM 后端、API key 身份验证等功能来提供 OpenChat 模型服务。
- **社区支持**：`@imonenext` 感谢 `<@317006433797537792>`、`<@748528982034612226>`、`<@312370916820779040>` 和 `<@1129298496869122088>` 对本次发布的贡献。


**提到的链接**：

- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)
- [Chatbot UI](https://openchat.team)
- [GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data](https://github.com/imoneoi/openchat): OpenChat: Advancing Open-source Language Models with Imperfect Data - GitHub - imoneoi/openchat: OpenChat: Advancing Open-source Language Models with Imperfect Data
- [来自 OpenChat (@openchatdev) 的推文](https://fxtwitter.com/openchatdev/status/1744985660870795635)：🚀 宣布 OpenChat-3.5 Update 0106：𝗪𝗼𝗿𝗹𝗱’𝘀 𝗕𝗲𝘀𝘁 𝗢𝗽𝗲𝗻 𝗦𝗼𝘂𝗿𝗰𝗲 𝟳𝗕 𝗟𝗟𝗠！在本地体验 ChatGPT 和 Grok 级别的 AI 💿！在所有 4 项基准测试中超越 Grok-0 (33B)...
- [Reddit - 深入探索](https://www.reddit.com/r/LocalLLaMA/comments/193362r/new_model_openchat_35_update_0106/)


        

---
YAIG (a16z Infra) Discord 没有新消息。如果该频道长时间没有活动，请告知我们，我们将将其移除。