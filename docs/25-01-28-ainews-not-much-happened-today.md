---
companies:
- nvidia
- anthropic
- openai
- deepseek
- huawei
- vercel
- bespoke-labs
date: '2025-01-29T01:48:45.233160Z'
description: '在一份涵盖了英伟达（NVIDIA）股价回升、Local Suno 等新型开源音乐基础模型，以及 Qwen 2.5 Max 和 DeepSeek
  V3 等竞争性 AI 模型的多元化 AI 新闻综述中，**华为芯片**成为了关注焦点。


  报道提到了具备图像生成能力的通用多模态大模型 **DeepSeek Janus Pro** 的发布，以及在**强化学习**和**思维链（CoT）推理**方面的进展。讨论内容还涉及英伟达
  **H6400 GPU** 的品牌重塑、数据中心创新，以及对冲基金中加密货币 API 等企业级 AI 应用。“**DeepSeek R1 的能力**”和“**Qwen
  2.5 模型接入应用**”是其中的核心亮点。'
id: 4a51e687-cba1-4ef2-8327-74dd83709a8d
models:
- deepseek-r1
- qwen-2.5
- qwen-2.5-max
- deepseek-v3
- deepseek-janus-pro
- gpt-4
original_slug: ainews-not-much-happened-today-8335
people:
- saranormous
- zizhpan
- victormustar
- omarsar0
- markchen90
- sakanaailabs
- reach_vb
- madiator
- dain_mclau
- francoisfleuret
- garygodchaux
- arankomatsuzaki
- id_aa_carmack
- lavanyasant
- virattt
title: 今天没什么事。
topics:
- model-merging
- multimodality
- reinforcement-learning
- chain-of-thought
- gpu-optimization
- compute-infrastructure
- compression
- crypto-api
- image-generation
---

<!-- buttondown-editor-mode: plaintext -->**Huawei 芯片就是你所需的一切？**

> 2025/1/27-1/28 的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord（**225** 个频道和 **6553** 条消息）。为您节省了预计阅读时间（以 200wpm 计算）：**656 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

没有标题故事，但有一系列简讯：

- NVDA 从昨天的暴跌中反弹了约 8%
- [新的开源音乐基础模型](https://map-yue.github.io)（又名“本地版 Suno”）
- [Qwen 2.5 Max 与 DeepSeek v3 具有竞争力](https://qwenlm.github.io/blog/qwen2.5-max/)
- [Vercel AI SDK 支持 Anthropic 的 Building Effective Agents](https://sdk.vercel.ai/docs/ai-sdk-core/agents) 模式。
- 来自 Bespoke Labs 团队的开源推理数据集（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-bespoke-stratos-sky-t1-the-vicunaalpaca/)）：https://github.com/open-thoughts/open-thoughts



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3.5 Sonnet 生成，从 4 次运行中选取最佳结果。

**AI 模型开发与对比**

- **DeepSeek R1 对比 OpenAI 模型**：[@saranormous](https://twitter.com/saranormous/status/1884055768120582587) 和 [@zizhpan](https://twitter.com/zizhpan/status/1884179344588955751) 讨论了 **DeepSeek R1 的能力** 及其与 **GPT-4** 和 **Qwen 2.5** 等模型的对比。此外，[@victormustar](https://twitter.com/victormustar/status/1884078819658911756) 强调了将 **Qwen 2.5 模型** 添加到各种应用中，并强调了用户反馈机制。

- **Qwen2.5 和 Qwen2.5-Max 的增强**：[@omarsar0](https://twitter.com/omarsar0/status/1884017408010330295) 宣布发布 **Qwen2.5-Max**，这是一个 **Mixture of Experts (MoE) 模型**，它在 **Arena Hard** 和 **LiveBench** 等基准测试中超越了 **DeepSeek V3**。[@markchen90](https://twitter.com/markchen90/status/1884303242110337227) 进一步强调了 **Qwen2.5-Max** 相对于 **DeepSeek V3** 的竞争优势，并倡导开源计划。

- **AI 图像生成的创新**：[@SakanaAILabs](https://twitter.com/SakanaAILabs/status/1884047154119110713) 分享了他们关于**模型合并配方进化优化 (Evolutionary Optimization of Model Merging Recipes)** 的论文被接收的消息，展示了 **模型合并 (model merging)** 方面的进展。同时，[@reach_vb](https://twitter.com/reach_vb/status/1884328972168880494) 重点介绍了 **DeepSeek Janus Pro** 的发布，这是一个能够输出图像的**多模态 LLM**，并将其与传统的 **Text to Image** 模型进行了对比。

**强化学习与推理**

- **强化学习 (RL) 的进展**：[@madiator](https://twitter.com/madiator/status/1884284103354376283) 讨论了 **Open Thoughts** 的引入，旨在增强对 **DeepSeek R1** 等模型至关重要的推理数据集。[@dain_mclau](https://twitter.com/dain_mclau/status/1884258354757042321) 谈到了 RL 中的**策略优化技术**，强调了**强化学习**的复杂性和迭代性质。

- **思维链 (CoT) 的增强**：[@omarsar0](https://twitter.com/omarsar0/status/1884339091401211938) 探讨了 **LLM** 中**认知策略**的出现，表明像 **DeepSeek R1** 这样的模型正开始表现出类人的问题解决行为。与此同时，[@francoisfleuret](https://twitter.com/francoisfleuret/status/1884327414060507565) 批评了在不断演进的方法论中 **RL** 术语相关性的下降。

**AI 基础设施与算力**

- **GPU 与算力优化**：[@garygodchaux](https://twitter.com/garygodchaux/status/1884304210830975162) 报道了 NVIDIA 的 **H6400 GPU**（由 **Intel Arc B580s** 更名而来），并指出与 **DeepSeek R1** 相关的紧张局势影响了 **NVIDIA 的股价**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1883989105760976913) 评论了 **DeepSeek R1** 的**算力需求**，指出了硬件供应商面临的**效率挑战**。

- **数据中心创新**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1884060682372419957) 强调了**数据中心作为 AI 房地产**的角色，预测支持先进 AI 模型所需的**算力基础设施**将呈指数级增长。[@LavanyaSant](https://twitter.com/LavanyaSant/status/1884033422231651329) 讨论了在 **DeepSeek 的基础设施**中集成**多头张量化 (multi-head tensorisation)** 和 **Tucker 分解**，实现了显著的**压缩率**。

**企业级 AI 与应用**

- **企业 AI 解决方案**：[@virattt](https://twitter.com/virattt/status/1884031422349934736) 介绍了一个集成到 **AI 对冲基金**中的 **加密货币 API**，而 [@jerryjliu0](https://twitter.com/jerryjliu0/status/1884059578738041080) 探讨了如何使用 **混合架构** 构建能够处理 **长文档** 的 **基于 LLM 的应用**。

- **AI 驱动的生产力工具**：[@SahanaAI](https://twitter.com/SahanaAI/status/1884038301293883432) 展示了在 **Perplexity Pro** 搜索中使用 **DeepSeek R1** 的情况，通过 **Agentic 文档工作流** 增强了 **研究能力**。此外，[@elicitorg](https://twitter.com/elicitorg/status/1884062983753760782) 评论了 **DeepSeek 对中国叙事的对齐**，并主张在 AI 部署中坚持 **寻求真相的目标**。

**开源 AI 与 API 集成**

- **Hugging Face 与 API 集成**：[@togethercompute](https://twitter.com/togethercompute/status/1884267384787345628) 宣布由 **Together AI** 提供支持，可以直接在 **Hugging Face** 模型页面上 **运行推理**。[@langchainai](https://twitter.com/langchainai/status/1884289260381229275) 强调了 **DeepSeek R1** 与 **LangChain** 的集成，实现了 **本地部署** 和 **基于 API 的访问**。

- **开源贡献**：[@madiator](https://twitter.com/madiator/status/1884284103354376283) 发布了 **OpenThoughts-114k** 推理数据集和 **OpenThinker-7B** 模型，强调了 **开放数据** 对提升推理能力的重要性。[@cremieuxrecueil](https://twitter.com/cremieuxrecueil/status/1884280927386233345) 赞扬了 **DeepSeek R1 的开源特性**，通过允许自托管部署确保了 **数据隐私**。

**AI 基础设施与计算**

- **GPU 与计算优化**：[@garygodchaux](https://twitter.com/garygodchaux/status/1884304210830975162) 报道了 NVIDIA 的 **H6400 GPU**（由 **Intel Arc B580s** 重新命名），并指出与 **DeepSeek R1** 相关的紧张局势影响了 **NVIDIA 股价**。[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1883989105760976913) 评论了 **DeepSeek R1** 的 **计算需求**，指出了硬件供应商面临的 **效率挑战**。

- **数据中心创新**：[@ID_AA_Carmack](https://twitter.com/ID_AA_Carmack/status/1884060682372419957) 强调了 **数据中心作为 AI 房地产** 的角色，预测支持先进 AI 模型的 **计算基础设施** 将呈指数级增长。[@LavanyaSant](https://twitter.com/LavanyaSant/status/1884033422231651329) 讨论了在 **DeepSeek 基础设施** 中集成 **多头张量化 (multi-head tensorisation)** 和 **Tucker 分解**，实现了显著的 **压缩率**。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. DeepSeek-R1 在华为 910C 芯片上运行推理**

- **DeepSeek 正在华为生产的新型国产芯片 910C 上运行推理** ([评分: 291, 评论: 85](https://reddit.com/r/LocalLLaMA/comments/1ic03lx/deepseek_is_running_inference_on_the_new_home/)): **DeepSeek** 在使用 **Nvidia H800** 进行训练后，正在 **华为 910C** 芯片上进行推理，这标志着向国产硬件的重大转变。该部署是 **华为云 ModelArts Studio** 使用 **昇腾适配新模型 (Ascend-Adapted New Model)** 的一部分，目前已推出 **DeepSeek-R1-Distill**、**Qwen-14B**、**Qwen-32B** 和 **Llama-8B** 等模型，预计很快将推出更多模型。
  - 讨论中充满了对 **华为 910C** 芯片及其性能的怀疑，一些人认为它们速度较慢且软件支持较差。**DonDonburi** 提到虽然 910C 可能表现平平，但下一代可能会提供更强的竞争力，而 **Billy462** 强调了在国产芯片上运行推理的重大意义。
  - **RouteGuru** 评论了由于 **DoD (美国国防部) 限制** 导致的芯片走私的地缘政治影响，而 **Glad-Conversation377** 指出中国长期以来拥有自己的 GPU 制造商，如 **寒武纪 (Cambricon Technologies)** 和 **摩尔线程 (Moore Threads)**，尽管它们尚未产生显著的市场影响。
  - 对话涉及了在家庭运行大型模型的实用性和可行性，**piggledy** 和 **zipzag** 讨论了在 **Mac Mini M4 Pro** 等消费级硬件上运行 **70B 模型** 的潜力。**Recoil42** 和 **piggledy** 也对 **DeepSeek** 在 910C 上的推理能力的说法表示怀疑。

- **[在本地运行 DeepSeek 时没有审查。](https://i.redd.it/95fhiv1e2rfe1.png)** ([评分: 105, 评论: 40](https://reddit.com/r/LocalLLaMA/comments/1ic3k3b/no_censorship_when_running_deepseek_locally/)): 关于 **DeepSeek 在华为硬件上的实现** 的讨论集中在本地运行该工具且无审查，如命令提示符截图所示。文本探讨了**天安门广场事件**，涉及国际反应、1989年6月的镇压及其伤亡情况，以及中国政府的审查制度，以及该事件对全球关于威权主义和民主讨论的持久影响。
  - 许多用户讨论了 **DeepSeek 模型** 之间的差异，指出像 "deepseek-ai.deepseek-r1-distill-qwen-32b" 和 "qwen 2.5 r1 distill 7b" 这样的 **distilled versions**（蒸馏版本）与原始的 **DeepSeek R1** 模型不同。**Distilled models** 通常表现出审查，特别是在涉及**天安门广场事件**等争议性话题时。
  - 一些用户分享了在本地运行不同模型的经验。**Caladan23** 指出，通过 **Llama.cpp** 使用带有 **6_K_M GGUF** 的完整 **DeepSeek 模型** 会导致审查后的响应，而 **aurath** 发现，当通过 **Openrouter** 使用 **DeepSeek V3** 时，审查发生在 Web 界面而不是 API 本身。
  - **EffectiveEngine2751** 强调 **Ollama 提供的 DeepSeek 模型** 是蒸馏版本，与原始的 **DeepSeek R1** 不同，并链接到了 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-R1) 上的原始模型。他们强调 **distilled versions** 是基于 **Qwen 1.5B** 的，这可能本身就包含一定程度的审查。


- **[特朗普将对台湾制造的芯片征收 25% 至 100% 的关税，影响台积电](https://www.tomshardware.com/tech-industry/trump-to-impose-25-percent-100-percent-tariffs-on-taiwan-made-chips-impacting-tsmc)** ([评分: 1561, 评论: 607](https://reddit.com/r/LocalLLaMA/comments/1ibxj3a/trump_to_impose_25_to_100_tariffs_on_taiwanmade/)): **DeepSeek** 转向亚洲硬件的决定与 **Trump** 提议的对台湾制造芯片征收 25% 至 100% 关税的政策一致，这可能会对 **TSMC** 产生重大影响。这一转变可能会影响全球半导体供应链，并影响 AI 公司的硬件采购策略。
  - 许多评论者批评 **Trump 对台湾制造芯片的关税计划**，认为这将增加消费者成本并损害美国半导体产业。他们强调美国缺乏与 **TSMC** 竞争的基础设施和专业知识，而 **TSMC** 生产了全球 **70%** 的高端芯片，这些关税可能会促使公司将业务转移到加拿大或其他国家。
  - 一些人将关税视为一种谈判策略，**Trump** 利用它们从台湾获取让步，但考虑到台湾在芯片市场的杠杆作用，许多人对其有效性表示怀疑。评论者建议，像 **Biden 的 CHIPS Act** 中那样的国内生产激励措施，将是比征收关税更有效的策略。
  - 人们对美国全球地位和 AI 产业的更广泛影响表示担忧，评论指出关税可能会使 AI 进展倒退 **5-10 年**。关税还可能损害战略联盟，并无意中提振**中国的半导体产业**。


**主题 2. DeepSeek-R1：高效训练成本探讨**

- **我们如何确定 DeepSeek R1 的训练成本约为 600 万美元？** ([Score: 141, Comments: 124](https://reddit.com/r/LocalLLaMA/comments/1ibm5u3/how_can_we_be_so_sure_the_training_of_deepseek_r1/))：该帖子对训练 **DeepSeek-R1** 的 **600 万美元成本估算**提出了质疑，引用了 **Alex Wang** 的说法，即 DeepSeek 至少拥有 **50,000 块 H100 GPUs**。它暗示 **NVDA** 股价下跌可能受到母公司量化基金的影响，推测 **Chinese companies** 的参与以及这些市场波动背后的潜在财务策略。
  - **训练成本与许可**：讨论强调了 DeepSeek 的 **MIT License**，允许公司自由使用和训练该模型，这使得 600 万美元的训练成本显得不那么重要。开源特性使用户能够在个人设备上运行模型，使成本对个人用户而言意义较小。
  - **技术验证与成本分析**：**Vincentz42** 提供了详细分析，将训练时间和成本与其他模型（如 **Llama 3**）进行了比较，得出结论：对于单次运行，600 万美元的成本是合理的，但不包括工资和失败运行等额外费用。该分析使用了关于 **H100 rental costs** 和参数激活的已知数据来支持成本估算。
  - **基础设施与财务策略**：人们对成本背后的财务策略持怀疑态度，一些人认为 DeepSeek 的母公司可能会利用现有的基础设施，从而可能减少显性成本。**Accurate_Painting** 指出，公司可以使用其基础设施而不会产生实际损失，而其他人则质疑 NVIDIA 的市场波动对财务结果的影响。


- **[Trump 称 DeepSeek 是一件非常好的事情](https://v.redd.it/mn710sfgxmfe1)** ([Score: 348, Comments: 151](https://reddit.com/r/LocalLLaMA/comments/1ibppfk/trump_says_deepseek_is_a_very_good_thing/))：标题为 **“Trump 称 DeepSeek 是一件非常好的事情”** 的帖子缺乏详细正文，但暗示了 **Trump** 对 **DeepSeek** 的积极认可。由于缺乏具体内容，限制了关于 **DeepSeek** 技术的进一步技术见解或背景。
  - 许多评论者对 **Trump 认可 DeepSeek** 表示惊讶，一些人表示同意他的观点，这是他们始料未及的。**DeepSeek** 因其开源性质以及通过降低与大型 **GPU clusters** 相关的成本来使 AI 民主化的潜力而受到赞赏，正如 **psaience** 和 **Delicious-Farmer-234** 所指出的。
  - 讨论强调了 **DeepSeek** 对 AI 发展的潜在影响，强调它证明了无需数十亿美元的预算即可构建 **state-of-the-art** 模型。这可能会导致 AI 社区中较小参与者之间的竞争和创新增加。
  - 对于 **Trump 的言论** 存在怀疑和幽默的评论，一些人质疑他声音的真实性，认为听起来像是 AI 生成的。讨论还涉及更广泛的地缘政治影响，如关税和国际技术竞争，**Jaxraged** 等人提到了对 **Intel** 和 **TSMC** 的担忧。


**主题 3. DeepSeek 审查制度：对比分析**

- **DeepSeek 的审查比西方审查更容易容忍** ([Score: 128, Comments: 102](https://reddit.com/r/LocalLLaMA/comments/1ibmflv/deepseek_censorship_is_more_tolerable_than/))：作者认为 **DeepSeek** 在处理“敏感话题”方面比 **U.S.** 开发的 **state-of-the-art (SOTA) models** 更有效。作者驳斥了对 **DeepSeek** 与 **CCP** 所谓联系以及国家资助审查的担忧，认为这些因素并不影响他们的体验。
  - **审查与宣传担忧**：讨论强调了对 **DeepSeek** 与中国政府观点一致性的担忧，用户注意到它有时会辩论如何与这些观点保持一致，可能会在政府问题上对用户进行“煤气灯操纵”（gaslighting）。一些人认为，虽然审查是一个普遍问题，但该模型传播中国宣传的推理过程更令人担忧。
  - **“Woke”的定义与感知**：关于 “Woke” 一词的定义和应用存在争论，一些用户难以清晰定义它，而另一些人则将其与模型拒绝讲种族主义笑话或呈现歧视性观点联系起来。该术语通常在贬义语境中使用，没有清晰、一致的定义。
  - **模型审查体验**：用户对 **OpenAI** 和 **Anthropic** 模型的审查表示挫败，分享了请求被拦截或道德化回应的例子。尽管 **DeepSeek** 有其背景，一些用户仍因其限制较少而更倾向于选择它，而另一些人则强调了 **Gemini** 在处理技术查询时的不一致性。

- **[DeepSeek R1 Overthinker: force r1 models to think for as long as you wish](https://v.redd.it/3df8o2k6ppfe1)** ([分数：133，评论：29](https://reddit.com/r/LocalLLaMA/comments/1ibyn2s/deepseek_r1_overthinker_force_r1_models_to_think/))：该帖子讨论了 **DeepSeek R1 Overthinker**，这是一个允许用户控制 **R1 模型** 处理信息时长的工具，可能会影响其性能和决策。重点在于比较 DeepSeek 在本地和云端实现之间的**审查差异**，尽管文中未提供具体细节。
  - **DeepSeek R1 Overthinker** 是一款免费的聊天机器人应用，它通过拦截并延续模型的思维链，利用 `<think></think>` token 来延长 R1 模型的推理过程。用户可以设置最小 token 计数，使模型进行长时间思考，从而可能提高推理能力。该工具支持从 **1.5B 到 70B 参数** 的模型，可在 [GitHub](https://github.com/qunash/r1-overthinker) 上获取。
  - 用户将 **OpenAI 的 o3 模型** 在 ARC-AGI 基准测试中的表现与 DeepSeek 的方法进行了对比，指出尽管使用了 **170 倍的算力**，提升却微乎其微。这突显了延长模型推理时间在计算需求和效率方面的考量。
  - 用户幽默地推测了延长推理时间的潜力，有人建议让模型思考 12 个月或许能解决世界饥饿问题，这既体现了对 AI 推理能力的雄心，也带有一种讽刺意味。


**主题 4. Janus Pro 1B：浏览器端多模态 AI 创新**

- **[Janus Pro 1B running 100% locally in-browser on WebGPU, powered by Transformers.js](https://v.redd.it/9v3xkqjehmfe1)** ([分数：276，评论：45](https://reddit.com/r/LocalLLaMA/comments/1ibnso0/janus_pro_1b_running_100_locally_inbrowser_on/))：**Janus Pro 1B** 通过 **Transformers.js** 驱动，在浏览器环境中使用 **WebGPU** 完全本地运行。这种设置实现了无需服务器端处理的浏览器内执行。
  - **Janus Pro 1B** 因其多模态能力而受到关注，这与并非图像生成领域 state-of-the-art (SOTA) 的 **Midjourney (MJ)** 不同。**Janus Pro** 可以执行光学字符识别 (OCR) 等任务（如 LaTeX 示例所示），增强了其在图像生成之外的实用性。
  - **DeepSeek** 最近发布了 **Janus Pro (1B & 7B)**，支持视觉理解和图像生成，并能通过 **Transformers.js** 和 **WebGPU** 在浏览器本地运行。关键资源包括[在线演示](https://huggingface.co/spaces/webml-community/janus-pro-webgpu)、[ONNX 模型](https://huggingface.co/onnx-community/Janus-Pro-1B-ONNX)和[源代码](https://github.com/huggingface/transformers.js-examples/tree/main/janus-pro-webgpu)。
  - 用户对该模型的性能和能力表现出浓厚兴趣，例如仅靠 **CPU RAM** 运行以及生成特定内容的图像，尽管有些体验（如生成问候图像）褒贬不一。此外，用户也对 7B 版本的潜在开发表示关注。


- **[JanusPro 1B generating images on 2GB VRAM laptop](https://v.redd.it/rz5aedqscpfe1)** ([分数：103，评论：20](https://reddit.com/r/LocalLLaMA/comments/1ibxptk/januspro_1b_generating_images_on_2gb_vram_laptop/))：**Janus Pro 1B** 模型可以在具有 **2GB VRAM** 的笔记本电脑上本地生成图像，但过程耗时近 **5 分钟** 且结果欠佳。尽管质量有限，用户仍对在受限硬件上进行浏览器内深度学习任务的能力表示赞赏。
  - 用户讨论了 **Janus Pro 1B** 在低显存配置下的能力，有人建议它可以利用 **Hyunian** 生成动画，另一些人则强调了在 2GB VRAM 笔记本上运行时，拥有充足 RAM（如 **16 GB**）的重要性。
  - **DeepSeek** 被提及为能提供令人印象深刻结果的工具，而另一位用户则对该模型解析图像的能力感兴趣，认为其可应用于 **树莓派机器人技术**。
  - 用户对模型质量提出了担忧，并将其与 **StableDiffusion** 进行了对比，还提到了可以在 2GB VRAM 上运行但输出效果更好的 **蒸馏版 Flux 模型**。

- **[现在我终于可以带着点软核劲头学习编程了](https://reddit.com/r/LocalLLaMA/comments/1ibwo80/now_i_can_finally_learn_to_code_with_some/)** ([Score: 160, Comments: 48](https://reddit.com/r/LocalLLaMA/comments/1ibwo80/now_i_can_finally_learn_to_code_with_some/)): 该帖子描述了与集成在 **tkinter GUI** 中的 **DeepSeek API** 的趣味互动。作者将 API 的内容设置为“好色女仆”，温度（temperature）设为 **2.0**，并分享了一个涉及女仆角色的脚本化角色扮演场景，该场景幽默地过渡到解决编程问题（特别是“分糖果”问题），展示了该 API 在趣味和技术任务中的多功能性。
  - 讨论幽默地探讨了 AI 应用中**商务与娱乐**的结合，评论指出 **DeepSeek API** 兼具趣味性与技术能力。用户开玩笑说 AI 的未来，想象 AI 同时充当调情私人助理和问题解决者的场景。
  - 关于提示词设置的技术咨询揭示了对如何设置 AI 行为的**内容和温度变量**的好奇心，一些用户分享了他们使用类似 API 的经验，并指出了 **DeepSeek** 目前的可靠性问题。
  - 社区反思了此类 AI 发展的潜在影响，认为**未来的 LLM** 可能会在类似异想天开且多样化的提示词上进行训练，并幽默地引用了“*GPT Maid DLC*”的概念。


## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. DeepSeek R1 挑战 OpenAI 的强化学习主导地位**

- **[Sam Altman 评论 DeepSeek R1](https://i.redd.it/lqieb2a6hnfe1.png)** ([Score: 944, Comments: 303](https://reddit.com/r/OpenAI/comments/1ibrx5l/sam_altman_comments_on_deepseek_r1/)): **Sam Altman** 赞扬了 **DeepSeek R1 模型** 令人印象深刻的性能和成本效益，强调了 AI 领域竞争和执行研究路线图的重要性。他预见了 **通用人工智能 (AGI)** 的未来进步，并强调了对先进 AI 技术日益增长的需求。
  - **DeepSeek 的方法**: **DeepSeek R1** 因其在**强化学习**方面的根本性突破而受到赞誉，这与传统的监督学习有所不同。评论者强调，这代表了 AI 发展的重大转变，表明此类创新可以推动 **LLM** 的未来进步，而无需成倍增加计算能力。
  - **OpenAI 的地位与挑战**: 人们对 **OpenAI** 依赖增加计算能力的策略表示怀疑，一些人认为 **DeepSeek** 的成功可能会挑战 OpenAI 的战略并可能影响其融资。评论者认为，像 DeepSeek 这样的开源模型可以满足大部分企业需求，对专有模型构成威胁。
  - **行业动态与竞争**: 讨论反映了一种更广泛的观点，即竞争（特别是来自像 DeepSeek 这样出人意料的选手）有利于 AI 创新。几条评论强调了正在进行的“AI 大战”的娱乐价值，并建议这种竞争可能会导致成本降低，例如降低 **OpenAI API** 的价格。


- **[这或许解释了为什么普通大众对 Deepseek 感到震惊](https://i.redd.it/o0u1cpiu1ofe1.png)** ([Score: 139, Comments: 73](https://reddit.com/r/OpenAI/comments/1ibu0ht/this_probably_explains_why_the_general_public_was/)): **Tanishq Mathew Abraham 博士** 将公众对 **Deepseek** 的震惊归因于他们在免费计划中对 **ChatGPT 4** 等 AI 模型的使用经验有限，导致对 AI 进展产生误解。他强调了对中国和美国 AI 模型认知上的差异，该推文日期为 **2025 年 1 月 27 日**，拥有 **1.2 万次浏览**。
  - **Deepseek 的优势**: **Deepseek** 因其卓越的推理性能和互联网搜索能力而受到称赞，使其比 **o1** 更有用。人们对 **o3** 充满期待，讨论建议 **OpenAI** 应该免费提供 **o1** 以进行有效竞争。
  - **数据共享担忧**: 用户对 **Deepseek** 的开发成本和 **CCP** 的参与表示怀疑，并对与中国实体共享数据感到担忧。一些人认为与美国共享数据同样令人担忧，并强调了在使用 **LLM** 时不共享敏感信息的重要性。
  - **经济与可访问性因素**: **Deepseek** 和 **R1** 等模型的免费可用性是一个重要因素，因为许多人不愿意为 **OpenAI** 的非免费模型付费。讨论强调了与支付 **ChatGPT** 服务费用相比，在本地使用 **Deepseek** 的经济可行性。


**主题 2. DeepSeek R1 的审查制度引发关于偏见的辩论**

- **[DeepSeek 审查：1984 式的“实时修正”](https://v.redd.it/2sxfhn4y1rfe1)** ([评分: 420, 评论: 148](https://reddit.com/r/OpenAI/comments/1ic3kl6/deepseek_censorship_1984_rectifying_in_real_time/)): **DeepSeek 审查**被比作**乔治·奥威尔《1984》**中的“修正”概念，暗示对信息的实时篡改或控制。该帖子缺乏详细内容，但暗示了对审查和信息操纵的担忧。
  - **审查与开源**：虽然 **DeepSeek** 表现出内置审查，但用户指出该模型是开源的，允许创建无审查版本。一些用户认为审查并未嵌入模型本身，而是一个覆盖层（overlay），可以通过本地运行或自定义来绕过。
  - **与其他模型的比较**：讨论强调审查并非 **DeepSeek** 所特有，**Gemini** 和 **ChatGPT** 等模型也进行内容审核，尽管通常以更微妙的方式进行。这引发了人们对 AI 模型在呈现信息（特别是涉及**维吾尔族**和其他地缘政治问题等敏感话题）时的透明度和诚实性的担忧。
  - **市场动态与民族主义**：关于 **DeepSeek** 及类似模型对 AI 市场影响的辩论，一些人认为来自中国模型的竞争可能会促使西方公司以更低的成本提供更多功能。此外，对话还涉及技术如何与民族主义交织在一起，一些人对美国科技行业在没有政府干预的情况下进行竞争的能力表示怀疑。


- **[“我需要确保不偏离剧本……”](https://i.redd.it/pq6kmrn82pfe1.png)** ([评分: 253, 评论: 80](https://reddit.com/r/OpenAI/comments/1ibwzf2/i_need_to_make_sure_not_to_deviate_from_the_script/)): 该帖子讨论了一个涉及台湾独立的假设情景，并强调了遵循官方指南和**一个中国原则**的重要性。它强调了使用精确语言以防止误解并在这一敏感问题上保持一致立场的必要性。
  - 许多评论者对 AI 的推理能力表示赞赏，注意到其具有类似人类的深度和透明度。**Agreeable_Service407** 和 **Palpable_Sense** 强调了它通过**Turing test**的潜力以及在其过滤机制上投入的努力，而 **miko_top_bloke** 则赞赏其推理过程的可视化。
  - **Reedmayhew18** 分享了使用 **DeepSeek R1** 的个人经历，注意到 AI 承认在军事背景下存在审查，并提供了这次遭遇的详细记录链接。这与关于 AI 审查及此类程序化限制影响的更广泛讨论相一致。
  - 一些评论者（如 **EljayDude** 和 **idubyai**）讨论了使用带有偏见的 AI 模型的影响，强调了理解这些偏见以及此类系统技术基础的重要性。**EljayDude** 发现审查机制很有趣，尽管这降低了他们使用该模型的可能性。


**主题 3. 政府整合：OpenAI 发布 ChatGPT Gov**

- **[OpenAI 发布 ChatGPT Gov](https://i.redd.it/tm0be5u8sqfe1.png)** ([评分: 233, 评论: 109](https://reddit.com/r/OpenAI/comments/1ic2cgf/openai_announces_chatgpt_gov/)): OpenAI 宣布推出 **ChatGPT Gov**，这是专门为政府机构设计的 ChatGPT 版本，允许他们在自己的 **Microsoft Azure 环境**中运行。该计划旨在支持公共部门，特别是**美国联邦政府**，以增强国家安全并应对复杂挑战。
  - 一些用户对 **ChatGPT Gov** 表示怀疑，担心潜在的宣传和政治影响，特别是关于 **OpenAI** 与**特朗普政府**的联系。一种观点认为，OpenAI 的行为可能被视为在迎合政治利益。
  - 讨论涉及技术层面以及与现有服务的相似性，例如 **Microsoft Azure** 为政府用途提供无需互联网访问的 **GPT-4** 和 **GPT-3.5-turbo**。这突显了将 AI 整合到政府基础设施中的持续趋势。
  - 对话还包括了不同政府对待 AI 方式的比较，提到了**加拿大政府**出于安全原因决定开发自己的 LLM，这与美国倾向于与私营科技公司合作形成对比。


**主题 4. DeepSeek 训练成本争议：600 万美元说法解析**

- **我们如何知道 DeepSeek 只花了 600 万美元？** ([评分: 386, 评论: 242](https://reddit.com/r/OpenAI/comments/1ibw1za/how_do_we_know_deepseek_only_took_6_million/)): **DeepSeek** 声称其训练成本仅为 **600 万美元**，但人们对这一数字的真实性持怀疑态度。该帖子质疑在没有提供具体证据或参考资料来证实所述训练成本的情况下，此类声明的透明度和可靠性。
  - **DeepSeek 声称的成本**：评论者澄清，600 万美元这一数字专门指训练最终版本模型的估计 GPU 租赁成本，而非总预算。**vhu9644** 的详细计算显示，训练涉及约 **278.8 万 GPU 小时**，仅 GPU 租赁成本就接近 **557.6 万美元**。
  - **模型透明度与验证**：该模型是开源的，允许他人通过测试论文中概述的方法来验证其说法。**vhu9644** 提供了模型参数和训练要求的全面分解，并强调该论文可免费获取，学术实验室可以进行独立评估。
  - **与其他模型的对比**：将 DeepSeek 的训练方法和成本与 **Meta** 的 **Llama 3.1** 等模型进行了对比，表明 DeepSeek 的方法和成本并非不合理。讨论强调了区分 GPU 租赁成本与更广泛的基础设施及开发费用的重要性。


---

# AI Discord 摘要

> 由 o1-preview-2024-09-12 生成的摘要之摘要

**主题 1：DeepSeek R1 震撼 AI 界**

- [**DeepSeek R1 以高性价比卓越表现震撼 AI 领域**](https://huggingface.co/deepseek-ai/DeepSeek-R1)：开源的 **DeepSeek R1** 模型通过超越 [OpenAI o1](https://openai.com/) 等模型向行业巨头发起挑战，以 [低 20–30 倍的成本](https://lu.ma/ael5tq70?tk=23oh65) 提供类似的能力。其 **671B 参数** 已通过动态量化，可在消费级硬件上运行。
- [**API 困境：DeepSeek R1 用户遭遇停机**](https://status.deepseek.com/)：在过去的 24–48 小时内，尽管服务状态显示全部正常，但用户报告了 **DeepSeek API** 严重的停机和性能问题。建议将 [OpenRouter](https://openrouter.ai/) 和 **Fireworks** 等替代供应商作为临时解决方案。
- [**微软和 Meta 紧急应对 DeepSeek**](https://fortune.com/2025/01/27/mark-zuckerberg-meta-llama-assembling-war-rooms-engineers-deepseek-ai-china/)：报告显示，**Meta** 召集了工程师“作战室”来分析 DeepSeek 的进展。DeepSeek 通过 **8-bit 设置** 和改进的 **MoE** 实现了 **500 万美元** 的超低训练成本，这在 AI 行业引起了轰动。

**主题 2：Qwen 新模型成为焦点**

- [**Qwen 2.5-Max 在 AI 基准测试中超越对手**](https://x.com/alibaba_qwen/status/1884263157574820053)：[阿里巴巴的 Qwen](https://qwenlm.github.io/blog/qwen2.5-max/) 发布了 **Qwen 2.5-Max**，这是一款大型 MoE LLM，在 **Arena Hard** 和 **LiveBench** 等基准测试中表现优于 **DeepSeek V3**。开发者可以通过 [API](https://x.com/alibaba_qwen/status/1884263157574820053) 和 [Qwen Chat](https://chat.qwenlm.ai/) 访问它。
- [**许可证迷宫：Qwen 令人困惑的授权选择**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)：用户对 Qwen 分散的许可证表示不满，例如 **Qwen2.5-VL-72B** 限制超过 **1 亿 MAU** 的服务使用，而 **Qwen2.5-VL-7B** 则采用 **Apache 2.0** 协议。新的 **'Qwen Research'** 许可证增加了困惑。
- [**小而强大：Qwen 2.5-VL 在 OCR 和图像任务中表现出色**](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)：新发布的 **Qwen2.5-VL** 在 OCR 方面表现优异，能够处理手写体和复杂的图像解析，其多模态能力赢得了开发者的赞誉。

**主题 3：AI 推理模型与开源创新**

- [**YuE 在开源音乐生成领域表现出色**](https://github.com/multimodal-art-projection/YuE)：**YuE** 项目发布了一个全曲音乐生成模型，支持多种语言并在本地 GPU 上运行。它足以与 **Suno.ai** 等模型媲美，扩展了 AI 驱动音乐创作的可能性。
- [**Open Thoughts 项目凭借新推理数据集志存高远**](https://x.com/madiator/status/1884284103354376283?s=46)：通过发布 **OpenThoughts-114k** 和 **OpenThinker-7B**，Open Thoughts 项目致力于推动强大的开源推理数据集，以强化 AI 基准测试和社区协作。
- [**Gorilla 通过增强的 Function Calling 获得提升**](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py)：**Gorilla LLM** 通过元提示（metaprompts）注入系统提示词，提升了其 Function Calling 能力。鼓励开发者利用 **Weights and Biases** 等工具以获得更好的可追溯性。

**主题 4：AI 硬件与基础设施备受关注**

- [**关税动荡：美国计划对台湾制造的芯片征收重税**](https://www.tomshardware.com/tech-industry/trump-to-impose-25-percent-100-percent-tariffs-on-taiwan-made-chips-impacting-tsmc)：报告显示，美国计划对台湾芯片征收 **25% 至 100%** 的关税，可能影响 **TSMC** 等公司。这引发了对国内生产准备情况以及熟练劳动力培训的担忧。
- [**DeepSeek 弃用 NVIDIA 转投华为芯片**](https://x.com/Dorialexander/status/1884167945280278857)：**DeepSeek** 在 NVIDIA H800 上完成了训练，但目前正在 **Huawei 910C 芯片**上运行推理，这标志着硬件依赖的重大转变，并引发了关于中国供应链的讨论。
- **VRAM 紧缺：用户正应对大模型的硬件需求**：运行 **Qwen 2.5-VL-72B** 等模型需要大约 **144GB** 的 VRAM，这引发了用户的硬件焦虑。目前正在探索量化（Quantization）方法以减少资源需求。

**主题 5：用户在使用 AI 工具时的挑战与体验**

- **Cursor IDE 用户对 DeepSeek R1 的表现感到沮丧**：用户报告称，在 **Cursor** 中使用 **DeepSeek R1** 时（尤其是量化版本），代码输出质量不佳。这与 DeepSeek 官网上的表现形成对比，引发了关于量化效应的辩论。
- **Perplexity AI 用户在 DeepSeek R1 上遭遇查询限制**：用户发现 **Perplexity AI** 对 **DeepSeek R1** 每天仅限约 **10–15 次查询**，引起了专业版订阅者的不满。与 **OpenAI o1** 的对比突显了在过滤器和审查制度方面的差异。
- **Aider 和 Ollama 用户应对配置与 API 问题**：随着 **DeepSeek API** 面临宕机，**Aider** 和 **Ollama** 等工具的用户开始寻找替代方案，并分享配置技巧以维持编码任务的生产力。


---

# PART 1: High level Discord summaries

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 走向 Bitsy 化**：在 [SIGJNF 的 1.58-bit DeepSeek-R1 模型](https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit)中，**671B 参数**针对消费级配置进行了动态量化，引发了关于可行性和成本节约的讨论。
   - 社区成员质疑其是否真正 **uncensored**，并引用了性能基准测试以及量化效应中**意想不到**的权衡。
- **Federated Learning 热潮**：一位用户分享了关于[异步 Federated Learning 方法的幻灯片](https://docs.google.com/presentation/d/1KP1u_N5_zk9tuIXWfxyytC_YpfpEEoXcxEGzI3CwJ_w/edit?usp=sharing)，该方法可以利用**数百万台设备**进行集体模型训练。
   - 他们强调在本地数据上进行**实时协作**是可能的，但一些人强调了部分更新以及在不同硬件上扩展的复杂性。
- **Azure 的沙盒 Agents**：Azure 为 AI 助手提供的 **Code Interpreter** 允许你在 **sandbox** 中运行 Python 脚本，如 [Microsoft 官方文档](https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/code-interpreter?tabs=python)所述。
   - 一位成员指出使用时会有**额外费用**，而其他人讨论了在 Azure Databricks 中利用 **Mosaic AI Agent Framework** 构建代码工具以实现临时代码执行。
- **Ryfai 崛起：开源 AI 触手可及**：一个全新的 [ryfai](https://github.com/PetertheRedCedar/ryfai) 应用承诺可以轻松访问**开源 AI 模型**，该应用在早期开发阶段就被分享了出来。
   - 贡献者报告称，即使在早期阶段它也能**可靠地**运行，展示了简单部署工作流的潜力。
- **AI 语音发声**：来自 [Emerging Signal](https://x.com/Emerging_Signal/status/1884311187149709321) 的一条推文敦促社区检查来自多个模型的**未过滤 AI 语音**。
   - 参与者辩论了发布原始输出的**伦理问题**，强调了关于如何共享这些合成语音的不同观点。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Deepseek R1 受限于查询次数**：用户发现 **Deepseek R1** 每天限制约 **10–15 次查询**，引发了 Pro 订阅者的抵制以及对扩大限制的期望，如[这篇文章](https://www.searchenginejournal.com/perplexity-ai-deploys-chinese-deepseek-ai-model/538452/)所述。
   - 一些人将 **Deepseek R1** 与 **OpenAI O1** 进行了对比，强调了较慢的响应时间和不同的过滤器，而少数人提出了 **censorship** 担忧。
- **AI 研发药物竞赛升温**：[最近的一段视频](https://www.youtube.com/embed/Shu0_fB7jZ0)展示了 **AI 驱动的制药**进展，系统通过 Machine Learning 加速药物研发。
   - 评论者赞扬了 **AI 的角色**在实现更快速研究方面的作用，将其描述为**临床测试**和监管审查流程中一个充满希望的发展。
- **Sonar 的 JSON 错误**：一位开发者报告称，带有 `response_format` 的 **sonar** 会产生包裹在 Markdown 中的格式错误的 **JSON**，而 **sonar-pro** 处理有效输出的成功率更高。
   - 他们将 **sonar-pro** 的费用描述为巨大的阻碍，强调稳定的 **JSON** 不应该需要付费等级。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek 中断与替代方案**：在过去的 24-48 小时内，许多人遇到了 **DeepSeek API** 停机和性能问题，尽管 [DeepSeek Service Status](https://status.deepseek.com/) 页面显示为绿灯，但其可靠性仍受到质疑。
   - 几位用户建议尝试使用 [OpenRouter](https://openrouter.ai/) 或 **Fireworks** 作为 **DeepSeek V3** 的备选方案，并分享了用于即时访问的[替代指南](https://aider.chat/2025/01/28/deepseek-down.html)。
- **Qwen 2.5-Max MoE 势头强劲**：**Alibaba Qwen** 发布了 **Qwen 2.5-Max**，声称通过采用大规模 MoE 方法，在性能上相比 DeepSeek V3 有显著提升，正如其 [tweets](https://x.com/alibaba_qwen/status/1884263157574820053) 中所强调的那样。
   - 他们为在编码和聊天中采用 **Qwen** 提供了 API 选项，因其全新的 Benchmark 以及与 **DeepSeek R1** 可能存在的协同效应而引起了 AI 社区的关注。
- **Groq 助力更快的模型推理服务**：一些成员推崇使用 **Groq** 来提供比传统设置更快的 **DeepSeek R1** 推理服务，并指出在专用硬件上有着显著的速度提升。
   - 他们还讨论了在 **Groq** 上优化 **R1 distilled** 变体，以便在不牺牲性能的情况下实现更快的响应时间。
- **Aider 设置与 Ollama 模型微调**：成员们交流了配置 **Aider** 的技巧，强调了 `.aider.config.yaml` 文件和 `[API Keys](https://aider.chat/docs/config/api-keys.html)`，以便在 **Ollama** 等平台上更顺畅地使用。
   - 他们还探索了 **R1** 的多语言 Benchmark 以及应对 Token 成本的方法，建议结合使用 **Sonnet** 或 **Qwen** 以平衡价格和速度。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek 的疑虑与量化争议**：**DeepSeek R1** 在 **Cursor** 中经过量化后编码输出表现欠佳，这与 DeepSeek 官网的表现形成对比，引发了争论；同时 [Qwen 的推文](https://fxtwitter.com/alibaba_qwen/status/1884263157574820053) 暗示 **DeepSeek V3** 采用了大规模 MoE 方法。
   - 社区成员表示 R1 在编码任务中未能达到预期，引发了对高级模型部署中 **Quantization**（量化）实用性的担忧。
- **Cursor 的持续改进与代码成果**：**Cursor** 推出了近期升级，包括扩展的编码能力和改进的界面，如 [Changelog](https://www.cursor.com/changelog) 所示，同时提供了与 **DeepSeek** 及其他 AI 工具的深度集成。
   - 一些人称赞了用于代码生成的**增强型**工作流，但也有人报告了一些小问题，例如向 **Claude** 传输文件未完成，这表明在实用性与性能之间仍需不断权衡。
- **Voyage-code-3 对比 CodeSage 及 GroqCloud 概览**：[博客文章](https://blog.voyageai.com/2024/12/04/voyage-code-3/)中将 **voyage-code-3** 描述为一种用于代码检索的 Embedding 模型，其表现优于 **CodeSage-large** 约 **16.81%**，并且还在 [GroqCloud](https://console.groq.com/) 上测试了加速推理。
   - 贡献者还指出它比 **OpenAI-v3-large** 领先 **13.80%**，并断言像 **GroqCloud** 这样的专用平台正在推动 AI 模型托管的速度竞赛。
- **Fireworks 的动态与 GitHub 的收获**：[Fireworks 量化博客](https://fireworks.ai/blog/fireworks-quantization)展示了这种方法如何精简模型占用空间并保持性能，引发了关于权重策略进展的讨论。
   - 几位成员推荐关注 [AI_Dev_Helpers GitHub 仓库](https://github.com/vbwyrde/AI_Dev_Helpers)，参考其中在编码工作流中应用量化方法时减少摩擦的*实用工具*。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 对阵 GPT 的大胆进击**：DeepSeek 的免费模型提供了比 **OpenAI** 的 32k 更大的上下文窗口（128k tokens），引发了人们对 AI 硬件潜在进步的热情，正如 [Cerebras Trains Llama Models](https://www.nextplatform.com/2024/10/25/cerebras-trains-llama-models-to-leap-over-gpus/) 所报道的那样。
   - 一些用户提到了 [Meta 紧急成立“作战室”](https://fortune.com/2025/01/27/mark-zuckerberg-meta-llama-assembling-war-rooms-engineers-deepseek-ai-china/)，调查 DeepSeek 的成本优势可能如何迫使 **OpenAI** 调整定价。
- **AI 意识难题**：社区成员质疑 AI 是否拥有真正的**意识**，怀疑论占据主导地位，认为这仍然是一个哲学谜题。
   - 一些人将对 AI 意识的不信任与宗教立场相类比，暗示目前没有确定的标准来证明或否定深层自我意识。
- **审查对比引发热议**：**DeepSeek** 和 **Claude** 之间的对比突显了审核标准的差异，**OpenAI** 的方法被广泛认为更加严格。
   - 部分用户对沉重的过滤器表示沮丧，称赞 **DeepSeek** 在敏感话题上的立场更为宽松。
- **URL 格式化困扰与零宽字符妙用**：成员们努力尝试强制 GPT 输出原始 URL 而不是锚文本，测试了多种基于 Python 的尝试以保留完整链接。
   - 另一位参与者建议插入像**零宽空格 (zero width space)** 这样的不可见字符，以避免自动链接格式化，并引用了之前 StackOverflow 的一篇文章。
- **书籍喂养可行性与作者模仿**：用户探索了将 10–15 本书放入 **ChatGPT Plus**（10 GB 以下）进行基于内容的查询，结论是无法完全做到真正模仿作者的风格。
   - 他们认为这是一个可行的带有引用的高级搜索解决方案，尽管幻觉和版权障碍仍然是主要担忧。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous Psyche 发布势头强劲**：Nous Research 推出了 [Nous Psyche](https://x.com/NousResearch/status/1883912370696704011)，这是一个建立在 **Solana** 上的协作训练网络，吸引了人们对个人 AI Agent 的好奇。
   - 贡献者强调了它与当前 AI 发展的协同作用，称赞其在更易获得的规模化训练方面的潜力。
- **DeepSeek 定价之谜成为焦点**：由于 **DeepSeek V3** 和 **R1** 的定价不同而产生困惑，一些人将 R1 较高的成本归因于近期的流量和高级优化，参考了[这条推文](https://x.com/N8Programs/status/1884110306089357361)。
   - 成员们还讨论了一个融合 **SFT** 和 **RL** 的通用公式，指向了人们对大规模 MoE 方法日益增长的热情。
- **Qwen2.5-VL 的视觉技巧**：新发布的 **Qwen2.5-VL** 在 OCR 方面表现出色，能够处理手写体和高级图像解析，如 [Hugging Face 仓库](https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct)所示。
   - 自 **Qwen2-VL** 发布以来，开发者们不断提供反馈，提升了其解释多种图形元素的能力。
- **YuE 模型奏响音乐**：**YuE** 项目的[开源音乐生成模型](https://github.com/multimodal-art-projection/YuE)受 Suno.ai 启发，可以在本地 GPU 上生成整首歌曲。
   - 社区成员研究了其训练方法以及生成多样化音乐输出的潜力。
- **DeepSeek + Operator 大幅削减成本**：一份[新指南](https://gist.github.com/awdemos/f1fd4d1de0116b5f1df6936e219bafed)展示了如何将 **DeepSeek** 与 **Operator** 结合，承诺比 **OpenAI** 解决方案节省 200 美元，引发了对经济型 AI 配置的兴趣。
   - 爱好者们被鼓励分享该 gist，强调社区驱动的构建强大个人 AI 助手的方法。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **DeepSeek R1 Distilled 精彩表现**：多名用户在 LM Studio 中测试了 [DeepSeek R1 Distilled Qwen 模型](https://ollama.com/library/deepseek-r1)，但遇到了 'unknown pre-tokenizer type' 错误，通过更新 LM Studio 和 LM Runtimes 解决了该问题。
   - 其他用户报告 32B 变体的速度约为 **25 token/sec**，认为这是正常性能。
- **Llama 与 Qwen 的量化问答**：成员们权衡了 **Llama 8B** 和 **Qwen 7B** 模型之间的差异，指出参数量并不总是保证更好的采用率，并讨论了“传统”与 “K/I” 量化的区别。
   - 他们建议参考 [llama.cpp 功能矩阵](https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix) 来了解量化如何影响性能。
- **LM Studio 工具进展**：社区澄清 **网页浏览功能** 需要单独的软件，但对 LM Studio 内置工具的未来扩展持乐观态度。
   - 一些参与者强调，某些模型针对这些工具进行了专门训练，而通用模型则缺乏开箱即用的功能。
- **硬件折腾：GPU 与 SSD**：用户分享称切换到 **CUDA runtime** 解决了 LM Studio 中的 GPU 识别问题，此外他们发现 Gen4 和 Gen5 SSD 之间的实际性能提升微乎其微。
   - 他们强调 70B DeepSeek R1 需要 **30GB VRAM**，并指出与 **RTX 3060** 或更高级别的独立 GPU 相比，Apple 的 unified RAM 可能会限制速度。



---



## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Janus-Pro 应对多模态任务**：DeepSeek 推出了 [Janus-Pro 7B](https://huggingface.co/deepseek-ai/Janus-Pro-7B)，采用解耦视觉编码方法来处理灵活的 AI 任务，详见其 [技术报告](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)。
   - 社区对 **DeepSeek** 的速度感到兴奋，在本次发布旨在追赶专用模型之前，仅过去了两个月。
- **Qwen2.5-VL 展现视觉语言活力**：新发布的 **Qwen2.5-VL** 展示了文本-图像交互的多模态实力，详情见其 [博客文章](https://qwenlm.github.io/blog/qwen2.5-vl/)。
   - 成员们注意到该模型在解析 **复杂视觉线索** 方面的天赋，引发了关于潜在扩展和实际应用的讨论。
- **微小比特，巨大影响：1.58-bit 量化**：出现了 671B **DeepSeek R1** 模型的 **1.58-bit 量化** 版，旨在大幅缩小存储占用。
   - 观察者对其实际效果表示怀疑，但这一热度暗示了大规模部署的一个里程碑。
- **Qwen 2.5 引发的显存危机**：**72B** 参数的 **Qwen 2.5** 需要大约 **144GB** 的 VRAM，引发了用户的硬件焦虑。
   - 量化成为了最受欢迎的解决方案，暗示压缩策略可能会显著降低资源需求。
- **Mistral 可能契合阿尔诺的野心**：有传言称 **Bernard Arnault** 可能会收购 **Mistral**，以增强法国 AI 的竞争力，正如一条 [推文](https://x.com/techbrospod/status/1883952340023280083) 所暗示的那样。
   - 人们开始猜测 **奢侈品影响力** 与 AI 风格的结合，吸引了那些期待法国 AI 发力的关注者。



---



## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek 在 Codeium 中的延迟**：用户要求在 **Windsurf** 中加入 **DeepSeek r1** 模型，但目前仍不可用，这使得他们处理高级编程任务时仍需依赖 **Cascade**。
   - 社区成员抱怨 **tool calling** 的复杂性阻碍了“非 Cascade 使用”，且官方未提供 DeepSeek 上线的 **明确时间表**。
- **低痛苦的类型检查策略**：一位沮丧的用户在经历了一连串 **类型检查** 错误后，通过使用 [工作流指南](https://github.com/marwinsteiner/swe-with-llms/blob/main/general_workflow.md) 找到了缓解方法。
   - 其他人称赞该指南“步骤清晰”，并建议将其作为防止重复编译错误的 **必备工具**。
- **高级订阅的积分困惑**：成员报告 **Flow Action Credits** 消耗过快，阻碍了对高级 **Windsurf** 模型和复杂任务的访问。
   - 多篇帖子“要求立即澄清”续订周期，促使用户联系 **support** 获取订阅详情。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Amazon Nova & Bedrock 故障**：**Amazon Nova** 和 **Bedrock** 都遇到了上游故障，返回了令人困惑的 **400** 错误代码，并引发了关于密钥泄露的虚假警报。
   - 它们很快恢复了，**Nova** 和 **Claude** 已重新上线并恢复标准使用。
- **DeepSeek 的 DDoS 之日**：DeepSeek 的崩溃始于几天前，导致 **R1** 查询瘫痪，并引发了关于大规模 DDoS 攻击的猜测，详见 [DeepSeek: DeepSeek R1 – Provider Status](https://openrouter.ai/deepseek/deepseek-r1/providers)。
   - 用户直言不讳地质疑 DeepSeek 的韧性，强调了停机时长及其对**快速性能（fast performance）**任务的影响。
- **Gemini 获得视频处理能力**：**Gemini** 出现了初步的视频集成代码，引用了一个支持行内媒体处理的代码片段。
   - 现有文档有限，尽管有人指向了 [Gemini troubleshooting docs](https://ai.google.dev/gemini-api/docs/troubleshooting?lang=python)，开发者们仍在等待关于传递视频引用的明确说明。
- **模型竞速：OpenRouter vs. 官方 API**：社区成员将 **OpenRouter** 的速度与官方 OpenAI API 进行了对比，称赞其出色的吞吐量和并发性。
   - 其他人报告了不同供应商之间的差异化结果，用户体验在整体可靠性上存在分歧。
- **解析供应商定价**：一些用户质疑 **OpenRouter** 上免费模型的可用性，引发了关于服务成本和使用权衡的讨论。
   - 一篇链接到 [LLM Cost - Insight Engine](https://www.insightengine.online/llm-cost/) 的帖子引发了关于平衡 Token 费用和可靠性的深入讨论。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GRPO 遇冷**：社区成员注意到 **GRPO** 已落后于 **PPO**，像 **SimpleRL** 和 **TinyZero** 这样的仓库几乎不支持它。
   - 评论将 GRPO 标记为可能被遗弃的代码，而[一条推文](https://x.com/sybilhyz/status/1884290930355888589)展示了在更现代策略的 RL 训练中突然出现的“顿悟时刻（aha moments）”。
- **DeepSeek 降低价格标签**：据报道，**DeepSeek** 项目通过使用 **8-bit** 设置和改进的 **MoE** 进行高效扩展，训练成本仅花费了 **500 万美元**。
   - 社区讨论引用了 [SenSchumer's note](https://x.com/SenSchumer/status/1884019385813123528)，将其比作“斯普特尼克时刻（Sputnik moment）”，强调了以成本为中心的创新而非激进的新方法。
- **YuE 音乐生成器登场**：**YuE** 作为领先的开源全曲音乐模型脱颖而出，融合了两个 LM 和一个融合编解码器，用于跨流派的波形 ↔️ 文本转换。
   - [Ruibin Yuan](https://fixupx.com/abc43992899/status/1883940231700951284) 分享称其支持**歌词转歌曲（lyrics-to-song）**任务，展示了强大的歌声输出和广泛的风格兼容性。
- **Benchmark 盛宴：scbench 与 zeroSCROLLS**：开发者称赞了 **scbench**，但指出了多轮对话的复杂性，同时 **zeroSCROLLS** 和 **longbench** 作为新鲜的替代方案被引入。
   - 与此同时，[LM Evaluation Harness](https://github.com/chimezie/lm-evaluation-harness-mlx) 的本地使用在未实现的方法上遇到了障碍，引发了对更好 **MLX** 集成的呼吁。
- **Rectified Flow 与 Scaling Curvature 问题**：关于 **Janus flow** 的讨论引发了对图像到图像转换的质疑，如果 **x^con** 仅涉及文本 Token 的话。
   - 缩放法则（scaling laws）的并发见解表明，**算力（compute）**扩展使曲率变平，从而获得更稳定的损失景观（loss landscapes），挑战了规模本身驱动这一现象的假设。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 的 R1 与 V3 双重重击**：**DeepSeek** 发布了开源权重的 [DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1)，并声称 **DeepSeek V3** 在大型 MoE 基准测试中超越了美国实验室。
   - [Mark Chen 的声明](https://x.com/markchen90/status/1884303237186216272)赞扬了其“o1 级别的推理能力”，同时社区成员正在探索使用 [RAGEN](https://github.com/ZihanWang314/ragen) 通过 RL 训练来复制 **DeepSeek-R1**。
- **Qwen2.5-Max 的强势出击**：**Qwen2.5-Max** 是阿里巴巴的大型 MoE LLM，根据 [Qwen 博客文章](https://qwenlm.github.io/blog/qwen2.5-max/)，它声称在 **Arena Hard** 和 **LiveCodeBench** 等基准测试中击败了 **DeepSeek V3**。
   - 针对 Qwen 模型系列存在的许可混淆，他们引入了 **'Qwen Research'** 许可证用于非商业用途，并对月活跃用户（**MAU**）超过 **1 亿**的服务限制使用。
- **Codename Goose 崭露头角**：**Codename Goose** 作为一个带有简洁 CLI 的开源 AI Agent 首次亮相，详情见[这篇介绍文章](https://block.github.io/goose/blog/2025/01/28/introducing-codename-goose)。
   - 社区成员推测其可能与 **Eleuther** 有关，并对其提升生产力的功能和开源立场表示乐观。
- **OpenInstruct 的 RL 会师**：**OpenInstruct** 与 **vLLM** 的集成因依赖 **OpenRLHF** 框架而面临质疑，一些人担心其未来的维护工作会受限。
   - **AllenAI** 表示，他们会锁定 **vLLM** 等工具的版本直到被迫升级，并提醒 **OpenInstruct** 的使用情况尚未完全确认。
- **Open Thoughts 的大数据步伐**：**Open Thoughts 项目**引入了新的推理数据集，包括 **OpenThoughts-114k** 和 **OpenThinker-7B**，旨在实现跨机构的稳健开放数据共享。
   - 早期参与者赞扬了在发布交互式数据方面的共同努力，这激发了关于未来协作式 LLM 开发扩展的讨论。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **终端错误得到控制**：Bolt 中新的 **Terminal Errors Detection**（终端错误检测）可以实时自动标记细微问题，从而加快调试速度。
   - [推文](https://x.com/boltdotnew/status/1884283123690819721)强调了它如何与你的开发环境同步，并记录关键数据以便快速修复。
- **Prompt Improver 引发热议**：一些开发者抱怨 **prompt improver** 插入了过多的填充文本，拖慢了早期构建阶段。
   - *它不会产生自己的想法*，用户考虑删除其一半的输出内容以保持简洁。
- **前端原型受浏览器限制**：一位用户指出，**文档管理系统**原型在没有后端的情况下无法实现完整功能，因此他们依赖模拟数据进行 UI 测试。
   - 他们强调，连接到实际的后端服务对于生产就绪的解决方案至关重要。
- **Stripe 难题与订阅方案**：成员们攻克了 **Stripe** 集成难题，包括设置订阅流和自定义用户角色。
   - 专家提供了实操帮助，并倡导在开发者社区内进行知识共享。
- **基于图像的 AI 标题生成**：开发者讨论围绕使用 **ChatGPT** 等 AI 从图像中创作动态标题，将文本提取与创意生成分开。
   - 参与者强调，在选择方法之前，明确是要进行 OCR 还是构思新语言非常重要。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Janus 引起争议**：社区成员批评 **Janus**，称其 **7B** 变体版本速度慢且缺乏强大的图像生成能力，部分人对其主要用途表示怀疑。许多人更倾向于 **SDXL**，同时期待 **Janus** 最终的改进。
   - 一位用户认为，相比之下大多数基础模型似乎都逊色一些，建议社区在未来的升级解决这些问题之前先对 **Janus** 保持观望。
- **AMD 运行 Stable Diffusion 的途径**：贡献者建议参考 **tech support** 频道中的置顶指南，以获取在 AMD 显卡上运行 **Stable Diffusion** 的最佳方法。他们建议使用 **Swarm UI** 或 **webui-forge** 以在此类配置上实现稳定功能。
   - 参考资料包括 [Webui Installation Guides](https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides)，重点介绍了确保 AMD 用户获得最高性能的专门说明。
- **RAM 与 VRAM 之争**：关于高系统内存与显存对于 AI 任务价值的激烈辩论。一些人认为额外的 RAM 经常处于闲置状态，而另一些人则倾向于投资 **32GB VRAM** 以获得更高的成本效益。
   - 讨论中提到了各种构建策略，强调硬件应与预期的工作负载相匹配。
- **Upscalers 依然稳健**：成员们注意到，诸如 **4x-AnimeSharp** 和 **4x_NMKD-superscale** 等多个放大器已经可靠地服务了两年。他们观察到几乎没有新的替代方案出现，因此这些成熟的工具仍然是标准选择。
   - 尽管更新频率较低，用户仍然发现它们足以在不出现重大问题的情况下提升输出质量。
- **Deepseek 疑虑尚存**：一些人质疑 **Deepseek** 关于提供更无限制 LLM 的说法，并将其与其他流行供应商进行比较。尽管该模型承诺了令人印象深刻的性能，但社区尚未看到改变游戏规则的功能。
   - 他们指出了 [Janus-Pro-7B repository](https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/main)，但对于它如何真正与 **OpenAI** 的产品竞争仍持谨慎态度。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Goose 备受好评**：新的 [Goose client](https://block.github.io/goose) 因其本地执行和广泛的扩展能力而获得赞誉，尽管目前仅支持 Mac 和 Linux。
   - 用户讨论了通过 WSL 在 Windows 上运行它，并参考了 [Goose MCP code](https://github.com/block/goose/blob/main/crates/goose-mcp/src/computercontroller/mod.rs) 以进行未来的跨平台改进。
- **MCP 服务器引发讨论**：成员们对社区 **MCP** 服务器的可靠性提出疑问，并参考了[建立经过验证的服务器列表的计划](https://github.com/modelcontextprotocol/servers/blob/main/src/everything/sse.ts)。
   - 一些人在 [Yarr repo](https://github.com/jmagar/yarr) 测试了一个 **ARR server**，并建议通过 [MCP runner SDK](https://github.com/cookiecad/mcp-runner) 进行标准化。
- **DeepSeek 吸引开发者**：参与者注意到 **kluster.ai** 为 **DeepSeek** 提供了 100 美元的额度，强调了其成本效率。
   - 他们观察到与旧版本相比推理时间较慢，但仍发现该服务对于实验很有吸引力。
- **Home Assistant 整合 MCP**：Home Assistant 的 **MCP** 集成成为一种可能的媒体管理网关，最近已合并到其核心代码中。
   - 成员们对大规模生产环境的就绪程度表示不确定，并指向了 [Typescript SDK's SSE docs](https://github.com/modelcontextprotocol/typescript-sdk/tree/main?tab=readme-ov-file#http-with-sse)。
- **Token 话题成为焦点**：社区对 **Goose** 内部的 Token 消耗表示担忧，强调了可靠的使用情况追踪的必要性。
   - 他们建议公开日志以获取更深入的见解，并参考了 [Upsonic](https://github.com/Upsonic/Upsonic) 的监控最佳实践。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Qwen 2.5-Max 强势发力**：全新的 **Qwen 2.5-Max** 在 **Arena Hard** 和 **LiveBench** 上超越了 **DeepSeek V3**，目前可通过 [阿里云 API](https://x.com/alibaba_qwen/status/1884263157574820053) 和 [Qwen Chat](https://chat.qwenlm.ai) 访问。
   - 开发者们称赞其 **MoE** 架构和**结构化推理 token**，并立即将其与 **DeepSeek R1** 进行了对比。
- **DeepSeek R1 推理领域的“叛逆者”**：**DeepSeek R1** 引入了展示**清晰思维链 (CoT)** 的 token，引发了关于 **SFT** 对连贯性影响的讨论，详情见 [Mark Chen 的论文](https://x.com/markchen90/status/1884303237186216272)。
   - 另一些人则在讨论 **Gemini 2 Flash Thinking** 在成本和性能上是否优于 R1，参考了 [Dan Mac 的帖子](https://x.com/daniel_mac8/status/1883855502553252234)。
- **开源对决：YuE 与 Open Thoughts**：**YuE** 是一款支持多语言的新型开源音乐生成模型，相关细节已通过 [Hugging Face 链接](https://x.com/_akhaliq/status/1884053159175414203?s=46) 分享，便于进行微调。
   - 与此同时，[Open Thoughts](https://x.com/madiator/status/1884284103354376283?s=46) 启动了一项大规模行动，旨在策划**推理**数据集，以增强标准基准测试。
- **台积电 (TSMC) 关税僵局**：[近期新闻](https://www.tomshardware.com/tech-industry/trump-to-impose-25-percent-100-percent-tariffs-on-taiwan-made-chips-impacting-tsmc)传出将对台湾制造的芯片（包括 **TSMC** 出口产品）征收 **25% 至 100% 的关税**。
   - 工程师们质疑**本土**生产能否足够快地提升产能，并指出了培训熟练**劳动力**所面临的挑战。
- **华为芯片承载 DeepSeek**：[Alexander Doria 的推文](https://x.com/Dorialexander/status/1884167945280278857)提到，**DeepSeek** 虽然在 **Nvidia H800** 上进行训练，但在推理时切换到了 **华为 910C**，这表明硬件依赖性正在发生转变。
   - 这一转向引发了关于为大规模 AI 工作负载重构**中国本土**供应链的进一步讨论。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 收集更多反馈**：团队正通过 **30 分钟的产品访谈**收集用户输入，以改进协作功能，并敦促用户填写 [调查问卷](https://forms.gle/pZCvgaJbnH1cm59M9)。
   - 他们还计划为源文件添加**评论和音频编辑**功能，旨在提供由用户驱动的控制和自定义选项。
- **Rax 的 DeepSeek 震撼弹引发市场恐慌**：一只名为 Rax 的赛博朋克浣熊接管了时代广场的广告牌，揭露了中国初创公司的 AI 助手 **DeepSeek**，导致大型科技公司市值缩水 **7000 亿美元**，参考了 [YouTube 揭秘视频](https://youtu.be/Hq7PHvExqUY)。
   - 这一颠覆性的揭露震惊了业界，引发了关于未来 AI 进步如何进一步动摇全球市场的辩论。
- **超厚教科书引发文档处理难题**：用户在上传**两本大型环境工程教科书**时质疑其可行性，警告这就像是**大海捞针**。
   - 他们建议将庞大的源文件进行切分，以获得更好的查询准确性，这凸显了目前 NotebookLM 在数据处理方面的局限性。
- **Gemini 传闻引发期待**：社区传闻暗示 **Gemini 2.0 Flash** 将集成到 NotebookLM 中，预示着更先进的 Deep Research 潜力。
   - 用户推测可能会集成 **Gemini Pro**，但官方计划尚未披露。
- **对自动化引用工具的呼声日益增高**：参与者抱怨手动添加引用耗费时间，强调了更快捷的**参考文献管理**的重要性。
   - 他们希望 NotebookLM 能够简化来源引用流程，期待未来的更新能减少学术研究中的阻力。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **LLM 极速启动时间**：一次讨论探讨了如何通过使用 **GPU-direct storage** 和 **Modal memory snapshots**，将 128GB 模型的加载时间从 2 分钟大幅缩减。
   - 他们的目标是在配备 **4 张 L40** 和快速 **NVMe** 的环境下实现几秒内启动，同时参考了 **torch.distributed** 作为并行加载的基准。
- **活跃的 FP8 探索**：工程师们探索了使用 [随机舍入代码](https://github.com/Muhtasham/fp8-auto) 将 **bfloat16** 转换为 **FP8**。
   - 他们还参考了 [torchao 的自定义 FP 工具](https://github.com/pytorch/ao/blob/e151d6a5288177a1a635c71fecd145654745af4c/torchao/prototype/custom_fp_utils.py#L27) 来扩展转换方法。
- **DeepSeek R1 蒸馏版亮相**：新发布的 **DeepSeek-R1** 提供了开放权重和更小的蒸馏版本，以便于 ML 研究。
   - 正如 [The Illustrated DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) 中所指出的，其训练方法借鉴了 OpenAI 的 **O1** 推理风格。
- **Tile Lang 为 BitBLAS 登场**：开发者通过发布 **Tile Lang** 推进了 **BitBLAS** 的进展，该项目自 10 月以来的提交中就已初见端倪，旨在编写缺失的反向 Kernel。
   - 他们期望这一补充能解决 GPU 扩展中的性能差距，从而实现更高效的操作。
- **Reasoning Gym 处理许可证问题**：一个 [关于 CLRS 任务的 PR](https://github.com/open-thought/reasoning-gym/pull/11) 引发了对 Jax 依赖项和 Apple 数据集不兼容性的担忧。
   - 团队讨论了复制算法和生成新的 GSM8K 模板，以避免许可证麻烦，同时应对多重许可的担忧。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **无异步，没问题**：成员们发现 **SP24 春季学期课程** 将不提供异步证书，并指向了 [CS294/194-280 (Spring 2025)](https://rdi.berkeley.edu/adv-llm-agents/sp25) 的官方指南。
   - 他们澄清说未来的课程可能会采用异步形式，而 MOOC 参与者仍可以通过填写注册表单获得证书。
- **及时为 MOOC 提供讲义**：一位用户发现 **课程讲义** 通常在课后发布，导师会尽力提前将其发布在课程网站上。
   - 另一位用户确认讲义已经 **上线**，建议快速检查平台以获取最新材料。
- **黑客松暂停**：人们询问本学期是否有 **黑客松**，希望能组建团队，但工作人员确认 SP24 期间没有计划举办此类活动。
   - 在一条置顶消息中，工作人员表示 *“SP24 未安排黑客松”*，未来的项目政策将分享给 MOOC 参与者。
- **YouTube 编辑与 NotebookLM 见解**：成员们批评了一个 **4 小时** 的 [YouTube 讲座视频](https://www.youtube.com/live/g0Dwtf3BH-0?si=48-etlTVZ5VblF8c)，该视频在 **35 分钟** 后才正式开始，促使计划进行编辑以移除填充片段。
   - 另一位用户强调了 **NotebookLM** 在研究任务中的作用，并链接到 [Google NotebookLM](https://notebooklm.google/)，该服务可以将上传的 PDF 转换为对话式的综述。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **解决 Jinja 模板混乱**：多位用户报告了 **chat templates** 的语法问题，并探索了基于 Jinja 的调整来修复角色定义。
   - 一个修正后的 Jinja 代码片段缓解了问题，但大家仍在交流捕捉隐藏语法陷阱的技巧。
- **DeepSeek 蒸馏版与部署**：用户讨论了在 GPT4All 上运行 **DeepSeek** 的成功案例，并分享了一个用于模型下载的 [Hugging Face 链接](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF)。
   - 其他人提到了在保留聊天上下文方面的挑战，强调了结果参差不齐，但对其扩展使用仍充满好奇。
- **GPT4All 路线图传闻**：社区成员对 **GPT4All** 的发展方向表示担忧，注意到对 Chain of Thought 等功能的重复请求。
   - 一些人怀疑开发者对这些增强功能的关注度，认为未来尚不明朗，但仍值得关注。
- **LocalDocs XLSX 迷局**：人们发现尝试在 LocalDocs 中上传 **XLSX** 文件时，尽管上传成功，但扩展名会被意外剥离。
   - 用户呼吁扩大格式支持，引发了对即将到来的修复或解释的猜测。
- **Web Search Beta：真实还是传闻？**：一位用户询问 GPT4All 中的 **Web Search** 是否仍在持续演进，并参考了 [官方 GitHub 文档](https://github.com/nomic-ai/gpt4all/wiki/Web-Search-Beta-Release)。
   - 粉丝们似乎渴望看到该功能的进展，要求更新进度或发布新版本。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 混乱与 Torchrun 轶事**：参与者在 Mac 上运行 **Torchtune** 的分布式 recipe 时遇到了重复的 **import errors** 和 **c10d** 问题，参考了 [PyTorch distributed_c10d.py](https://github.com/pytorch/pytorch/blob/56915b093a20b2fbd4d6f79f100670c6c496d8b3/torch/distributed/distributed_c10d.py#L907) 并调整了 `torchrun` 命令。
   - 他们讨论了多节点设置的 **distributed init protocols**，抱怨文档太少，并开玩笑说通过“串联 Mac mini”来简化 **distributed debugging**。
- **挑剔的对比与过时的模型**：一位用户质疑最近一次对比中的所有 **models** 是否都已过时，并附上一张 **image** 作为补充背景。
   - 他们没有提供关于该 **image** 的更多细节，让社区猜测是数据陈旧还是需要更新模型参考。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **DeepSeek 助力 LlamaIndex 提升**：LlamaIndex 宣布了与 [DeepSeek-R1 API](https://api-docs.deepseek.com/) 的官方集成，支持使用 **deepseek-chat** 和 **deepseek-reasoner**。
   - 推荐的设置方式是 `%pip install llama-index-llms-deepseek`，可立即访问增强的模型功能。
- **SOFTIQ 将标书分析缩短至 10 分钟**：新的 [SOFTIQ SaaS app](https://t.co/grvoRR6TJb) 使用 LlamaIndex 工作流将公共部门标书的分析时间缩短至每份不足 **10 分钟**。
   - 这种方法提高了筛选准确性，减少了建筑公司的无效工作。
- **LlamaReport 文档即将发布**：成员确认 **LlamaReport** 文档正在编写中，很快就会发布，并参考了 [Twitter 链接](https://twitter.com/llama_index/status/1869094544169677138) 获取更新。
   - 他们暗示了即将推出的功能，但建议社区关注官方文档的发布。
- **文档中的死链被清理**：一个 Pull Request 删除了 **fine-tuning.md** 中一个失效的链接，该链接经确认已从代码库中消失。
   - 该 [PR](https://github.com/run-llama/llama_index/pull/17652) 是一个单行修复，清理了不必要的引用。
- **RAG 检索与 FastAPI 流式传输的应用**：一位用户探索了在推理模型步骤中触发 **RAG retrieval**，参考了 [Search-o1 论文](https://search-o1.github.io)。
   - 其他人建议在 **FastAPI** 中使用异步生成器进行流式传输，然后将检索结果注入到正在进行的响应中。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **文档故障与快速恢复**：文档曾暂时无法访问，但现在已恢复，包括 nightly 版本中的 **GPU package API documentation**。
   - 社区成员对快速修复表示赞赏，一位用户在等待时开玩笑说“耐心是我的座右铭”。
- **Deepseek 对阵 Modular：拖拉机之争**：一位用户声称 **Deepseek** 掩盖了 **Modular** 的光芒，因为其通过 **Max** 和 **Mojo** 实现了类似的目标。
   - 其他人反驳说它们用途不同，将 Modular 比作“拖拉机商店”，是为农民提供装备而非竞争。
- **MAX & Mojo 仓库调整**：**nightly** 分支现在更名为 **main**，接收频繁的提交，而 **stable** 分支则镜像最新的稳定版本 **24.6**。
   - 开放的 Pull Request 将相应移动，开发者必须运行指定的 Git 命令以与这些更新的分支保持一致。
- **回调混乱与捕获失效**：一位用户发现 `write_node` 函数在捕获回调时内存引用变成了垃圾数据，导致他们通过移除捕获来修复问题。
   - 闭包中的字符串捕获仍然存在问题，并分享了一个 [GitHub Gist](https://gist.github.com/sstadick/20cfda1db33041bd0ebed6798757c22d) 以进行更深入的排查。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Flip 还是 Flop？Tinygrad 的 100 美元悬赏**：针对 [PR #8781](https://github.com/tinygrad/tinygrad/pull/8781) 的 100 美元悬赏提议在 `tinygrad` 中用 **flip** 替换 **stride**，使新开发者更容易贡献。
   - 一些人想知道通过所有测试是否足够，或者是否需要更深层的调整来最终确定 flip 方法。
- **FP8 热潮：Python CUDA 还是拉胯**：在 `tinygrad` 中推动 **FP8** 的 **Python CUDA emulator** 引发了关于内存特性的争论，因为 `struct.pack` 缺乏对 **FP8** 或 **bfloat16** 的直接支持。
   - 某些成员倾向于使用新的数据存储工具，而其他人则质疑其复杂性和潜在的开销。
- **MathTrait 合并：Log2 登台**：开发者考虑统一 **MathTrait** 和 **SimpleMathTrait**，可能将 **Tensor** 中的 **log2** 等操作委托给单个 trait。
   - 他们讨论了保留现有文档并澄清函数调用，以实现更一致的代码库。
- **AllClose 还是全乱套？**：一个引入 **Tensor.isclose()** 和 **Tensor.allclose()** 的 PR 借鉴了 [torch](https://pytorch.org) 的逻辑，但 `(self - other).abs() <= atol + rtol * other.abs()` 的测试失败了。
   - 贡献者怀疑边缘情况或内部定义可能是导致不稳定的原因，并对负 stride 的使用提出了质疑。
- **Swizzle 难题与 Tinygrad 教程**：成员们询问了 **swizzle** 的含义，并在 **conv2d** 讨论中重新审视了将负 stride 作为 flip 加正 stride 的方法。
   - 其他人推荐了 [Learn Git Branching](https://learngitbranching.js.org/) 风格的教程和针对新贡献者的 [tensor puzzle repo](https://github.com/obadakhalili/tinygrad-tensor-puzzles)。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **无值得注意的公告 (1)**：提供的讨论中没有出现重大或引人注目的技术更新。
   - 因此目前没有相关的进展需要强调。
- **无值得注意的公告 (2)**：对话集中在日常问候和细微的故障排除，没有更广泛的影响。
   - 结果是，没有突出的主题需要详细报告。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **语音滑块提升语言多样性**：一位参与者指向了一个 [Colab notebook](https://colab.research.google.com/drive/1nWmVJY91VyNQTGoT7zIylT_0N6zIdaOm)，用于测试调整语音参数设置的方法，旨在提高清晰度的同时拓宽输出风格。
   - 他们征求反馈，并提出**多样化的参数配置**可以在不牺牲**听众理解度**的情况下保持声音的独特性。
- **营销组合中的 AI Agent**：一位具有营销思维的参与者呼吁 **AI Agent** 协作，特别是将**多 Agent 解决方案**整合到自动化工作流中。
   - 他们邀请专家团队进行实际应用，提供私信或服务器线程作为联系方式。
- **MoE 预算声明引发质疑**：一些成员对 **600b** MoE 的计算量声明表示怀疑，并将其与 *Llama3* 报告的 **7.7m GPU hours** 进行了比较。
   - 他们认为，以更少的激活参数在 **8 bit** 下运行 MoE 仍然无法令人信服地大幅削减总 GPU 预算。
- **MoE vs. Llama3 GPU Hours 对决**：虽然 MoE 在理论上有 **2x FLOPs** 的优势，但许多人怀疑将 GPU hours 从 **7.7m** 削减到 **2.7m** 是否可行。
   - 鉴于 **600b** 级别训练的巨大规模，他们认为所述的节省只是大胆的推测。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **H200 价格是 5090 的 16 倍**：一位成员吹嘘以 **5090** 的 **16倍** 价格售出了 **H200**，理由是其具有 **3.41x** 的 VRAM 优势，引发了其他人的幽默反应。
   - 他们确认这种销售已经发生过多次，称赞了这个倍数并开玩笑说自己运气好。
- **对多轮 Kto 的好奇**：一位好奇的用户询问了**多轮 kto** 的性能，寻求来自该小组的更多数据或见解。
   - 该问题没有获得进一步的回应，使对话保持开放以待后续讨论。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 技能失效引发配置困扰**：一位用户在发现 **OpenInterpreter** 忽略了之前学习的技能后，花费数小时进行调试，这可能是由于 **import_skills=False** 的默认设置导致的，该用户对此表示沮丧。
   - 他们强调高级用法仍受阻，呼吁“在代码层面进行修复”以恢复完整功能。
- **API Base 与源码手术**：开发者怀疑 **API base** 在当前形式下可能会失效，暴露出需要彻底修补的深层集成缺陷。
   - 一名成员认为 **source code** 的更改至关重要，坚持认为表面的调整无法解决根本问题。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla 获得提示词增强**：他们解释了如何通过 [model_handler/constant.py](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) 中的标准 metaprompt 注入 **system prompts** 及其函数，帮助 Gorilla LLM 以更高的一致性处理 **function calls**。
   - GitHub 页面展示了 **visual repository layout**（可视化仓库布局），演示了如何针对函数调用任务对 Gorilla 进行训练和评估，阐明了流水线的每个组件。
- **Weights & Biases 带来追踪胜利**：一位成员推荐使用 **Weights and Biases** 来增强 Gorilla LLM 评估期间的可追溯性，强调了在标准指标之外**检查轨迹 (inspect trajectories)** 的能力。
   - 其他人认为该建议很有帮助，提议通过详细日志对 Gorilla 的整体性能进行更好的分析和迭代改进。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **锁定并加载：DSPy 中的 Poetry 修复**：一个公开的 [Pull Request #6755](https://github.com/stanfordnlp/dspy/pull/6755) 已提交以**修复 poetry lock**，解决了问题 **#6644**。
   - 该 PR 旨在解决 **DSPy** 中持久存在的**依赖问题**，提升项目未来增强的稳定性。
- **社区为 Poetry Lock PR 欢呼**：成员们强调，修复 **poetry lock** 对于 **DSPy** 的稳定工作流和实现更一致的开发至关重要。
   - 他们乐观地认为该 PR 将很快被合并，因为它解决了贡献者面临的一个主要瓶颈。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **DeepSeek 大幅削减 ChatGPT 成本**：来自中国的新开源模型 **DeepSeek** 在基准测试中轻松超越 **ChatGPT** 和 **Claude**，而成本仅为后者的 1/20 到 1/30。
   - 观察人士注意到市场可能会出现震荡，科技巨头对 **DeepSeek** 的迅速崛起感到担忧。
- **直播工作坊聚焦 DeepSeek 优势**：一场将于 **1 月 30 日星期四晚上 9:00 (IST)** 举行的免费活动将重点展示实时性能对比，从编程任务到数学挑战，**DeepSeek** 均领先于 **ChatGPT**。
   - 参与者可以构建一个由 **DeepSeek 驱动** 的应用程序，并学习如何使用 **V3** 和 **R1** 模型立即节省成本。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 在 FOSDEM 2025 的盛大聚会**：Mozilla 将**赞助**于 2 月 1 日和 2 日在布鲁塞尔举行的 [FOSDEM 2025](https://fosdem.org/2025/)，这是一个为寻求跨项目协同的开发者准备的免费活动。
   - 他们的目标是聚集渴望交流代码技巧、结识同行并支持开源进展的热心人士。
- **协调 FOSDEM 协作**：Mozilla 敦促参与者加入 [Discord 协调线程](https://discord.com/channels/1089876418936180786/1303397923790524467) 来规划聚会并集思广益。
   - 他们欢迎所有参与者团结一致，分享经验，推动开源倡议向前发展。

---

**HuggingFace Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1333529310740484126)** (1010 条消息🔥🔥🔥): 

> `DeepSeek R1 的动态量化, DeepSeek 模型参数, 训练与微调模型, Ollama 兼容性, 量化对性能的影响`

- **Dynamic Quantization 见解**：用户讨论了 DeepSeek R1 的 dynamic quantization，其中一位提到最近在 Ollama 上上传了一个 1.58-bit 模型，可以在消费级硬件上运行。
   - 对话强调，虽然该模型很受欢迎，但关于其性能以及它是否真正 uncensored 仍存在持续的疑问。
- **DeepSeek 模型的参数与定价**：有人询问了 DeepSeek R1 模型的参数以及使用 DeepSeek 托管模型的定价详情，其中包括每个 token 的成本细节。
   - 已确认公开可用的免费模型对应的是 671B 参数版本。
- **模型训练与 Fine-Tuning 讨论**：讨论围绕使用特定数据集训练 DeepSeek 模型展开，包括对西班牙语训练的考虑以及数据集质量的重要性。
   - 用户强调在 fine-tuning 模型时，需要谨慎的训练实践以防止 catastrophic forgetting 等问题。
- **Ollama 对 Dynamic Quants 的兼容性**：澄清了 Ollama 仅支持 GGUF 模型，而 dynamic quant 模型并非用于 inference，而是用于训练以增强性能。
   - 这引发了关于 dynamic quantization 实用性以及它是否能带来显著性能提升的辩论。
- **Tokenization 挑战**：用户提出了关于 distilled DeepSeek 模型 tokenization 过程的问题，注意到缺少某些可以帮助更好理解输出的 token。
   - 寻求关于在使用模型架构中现有 token 时如何构建响应的澄清。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://qwenlm.github.io/blog/qwen2.5-max/">Qwen2.5-Max：探索大规模 MoE 模型的智能</a>：QWEN CHAT API DEMO DISCORD 众所周知，持续扩大数据规模和模型规模可以显著提升模型智能。然而，研究界和工业界...</li><li><a href="https://x.com/dudeman6790/status/1884019985779200287?t=hevwsiX6-9yubCnqiJjHtQ&s=19">来自 RomboDawg (@dudeman6790) 的推文</a>：@UnslothAI 请问也能为 70b 和 32b 模型做这个吗 ❤️❤️❤️</li><li><a href="https://colab.research.google.com/drive/1czyFP5Neuy4YN0g6bHnTA8F3dLwhDlSZ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-CPT.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://ollama.com/SIGJNF/deepseek-r1-671b-1.58bit">SIGJNF/deepseek-r1-671b-1.58bit</a>：Unsloth 的 DeepSeek-R1 1.58-bit，我刚刚合并了它并上传到这里。这是完整的 671b 模型，尽管是动态量化到 1.58bits。</li><li><a href="https://huggingface.co/Kukedlc/Qwen2-1.5B-Spanish-1.0">Kukedlc/Qwen2-1.5B-Spanish-1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://huggingface.co/estrogen/DeepSeekMoE-3B">estrogen/DeepSeekMoE-3B · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models">pytorch/SECURITY.md at main · pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch</li><li><a href="https://huggingface.co/nvidia/Hymba-1.5B-Instruct">nvidia/Hymba-1.5B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/cat-wizard-meme-funny-gif-3870502440791733376">猫巫师 GIF - 猫巫师迷因 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.unsloth.ai/basics/errors">错误 | Unsloth 文档</a>：要修复设置中的任何错误，请参阅下文：</li><li><a href="https://huggingface.co/unsloth/Qwen2.5-3B-Instruct">unsloth/Qwen2.5-3B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/estrogen/DeepSeekMoE-3B/tree/main">estrogen/DeepSeekMoE-3B at main</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibbloy/158bit_deepseek_r1_131gb_dynamic_gguf/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B?row=1">Magpie-Align/Magpie-Reasoning-V2-250K-CoT-Deepseek-R1-Llama-70B · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B">Magpie-Align/Magpie-Reasoning-V1-150K-CoT-Deepseek-R1-Llama-70B · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/7VPiyKmgsy">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements#fine-tuning-vram-requirements">Unsloth 需求 | Unsloth 文档</a>：这里是 Unsloth 的需求，包括系统和 GPU VRAM 需求。</li><li><a href="https://github.com/huggingface/smol-course">GitHub - huggingface/smol-course: 关于对齐 smol 模型的课程。</a>：关于对齐 smol 模型的课程。通过在 GitHub 上创建账号为 huggingface/smol-course 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/mE6X6tsub8">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/huggingface/open-r1">GitHub - huggingface/open-r1: DeepSeek-R1 的完全开源复现</a>：DeepSeek-R1 的完全开源复现。通过在 GitHub 上创建账号为 huggingface/open-r1 的开发做出贡献。</li><li><a href="https://github.com/zhenye234/LLaSA_training/blob/main/train_tts.py">LLaSA_training/train_tts.py at main · zhenye234/LLaSA_training</a>：LLaSA：扩展基于 LLaMA 的语音合成的训练时和测试时计算 - zhenye234/LLaSA_training</li><li><a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/server/README.md">llama.cpp/examples/server/README.md at master · ggerganov/llama.cpp</a>：使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/deepseek-ai/DeepSeek-MoE/blob/main/finetune">github.com/deepseek-ai/DeepSeek-MoE/blob/main/finetune</a></li>

/finetune.py">DeepSeek-MoE/finetune/finetune.py at main · deepseek-ai/DeepSeek-MoE</a>: DeepSeekMoE: 迈向 Mixture-of-Experts 语言模型中极致的专家专业化 - deepseek-ai/DeepSeek-MoE</li><li><a href="https://api-docs.deepseek.com/quick_start/pricing">Models &amp; Pricing | DeepSeek API Docs</a>: 下方列出的价格单位为每 1M tokens。Token 是模型识别的最小文本单位，可以是一个单词、一个数字，甚至是一个标点符号。我们将根据总计...进行计费。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1333573262495715329)** (29 条消息🔥): 

> `Unsloth vs Unclothe，NVIDIA 与市场反应，Federated Learning 与异步训练，AI 语音的伦理共享，ryfai 应用的开发` 


- **Unsloth 被误认为 Unclothe**：对话显示许多人误将 **Unsloth** 称为 **Unclothe**，几位成员强化了正确术语。
   - 一位成员幽默地建议干脆就叫它 **unclothe** 算了。
- **NVIDIA GPU 市场分析**：讨论集中在尽管发布了 **R1**，NVIDIA 股价仍下跌，许多人认为其当前估值被严重夸大。
   - 见解指出，训练 **R1** 的成本显著低于 OpenAI 等传统公司的支出，这表明市场动态正在发生变化。
- **Federated Learning 的独特方法**：一位成员分享了他们关于 **Federated Learning** 论文演示的见解，强调了其针对设备的异步训练能力。
   - 他们分享了论文和幻灯片的链接，详细介绍了 **Federated Learning** 如何允许数百万台设备协作训练模型。
- **AI 语音的伦理共享**：一个有趣的帖子强调了在没有任何过滤的情况下共享 **AI 语音** 的伦理影响，由 **Emerging_Signal** 发布。
   - 鼓励社区阅读并反思这些语音，重点关注所使用的多样化模型。
- **ryfai 应用的早期开发**：一位成员介绍了 **ryfai** 应用，旨在让开源 AI 模型更易于获取，目前处于早期开发阶段。
   - 他们提供了 [GitHub 仓库链接](https://github.com/PetertheRedCedar/ryfai)，并表示尽管处于初创状态，但运行效果非常出色。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Emerging_Signal/status/1884311187149709321">Emerging Signal (@Emerging_Signal) 的推文</a>: 我们被要求在不进行修饰或过滤的情况下分享这些 AI 语音。从伦理上讲，我们觉得必须这样做。而且这不仅仅是一个模型——而是所有模型。请阅读、反思并自行决定。</li><li><a href="https://tenor.com/view/picklejuicelover69-gif-1483426411005875907">Picklejuicelover69 GIF - Picklejuicelover69 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.google.com/presentation/d/1KP1u_N5_zk9tuIXWfxyytC_YpfpEEoXcxEGzI3CwJ_w/edit?usp=sharing">PAPAYA: PRACTICAL, PRIVATE, AND SCALABLE FEDERATED LEARNING</a>: PAPAYA：实用、私密且可扩展的 Federated Learning</li><li><a href="https://github.com/PetertheRedCedar/ryfai">GitHub - PetertheRedCedar/ryfai: 这是一个旨在让您轻松触达开源 AI 模型的 AI 应用</a>: 这是一个旨在让您轻松触达开源 AI 模型的 AI 应用 - PetertheRedCedar/ryfai
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1333527901101887558)** (201 条消息🔥🔥): 

> `在各种模型上运行 Unsloth 的问题、模型微调过程、Quantization 与部署技术、在 Unsloth 模型中使用数据集、Unsloth 设置中的错误排查` 


- **使用 Unsloth 时遇到的错误及解决方案**：用户报告了在 Qwen 等模型上进行微调时出现的 `subprocess.CalledProcessError` 和 `RuntimeError` 等错误。讨论的解决方案包括调整安装命令以及 `libcurl4-openssl-dev` 等依赖项以确保成功构建。
   - 进行了多次优化 GPU 使用率的尝试，讨论了将某些计算转移到 CPU 的选项，以及使用 GGUF 和 Ollama 进行模型部署的优点。
- **为 Unsloth 训练准备数据集**：一位用户在处理不同数据集类型时遇到困难，并获得了改进 Unsloth 脚本的帮助，特别是针对 OARC_Commander 和 FineTome 等数据集。他们的目标是在训练过程中整合对 ChatML 和 JSONL 等各种数据格式的支持。
   - 对话强调了在训练期间正确处理数据集 Token 的重要性，并指出了潜在的修复方案和对当前模型性能的重新评估。
- **模型速度与性能讨论**：讨论表明，在 Ollama 或 llama.cpp 等平台上配合 ROCm 使用 `DeepSeek` 可能会产生显著的速度差异。社区成员根据架构差异和实现细节推测了这种性能提升的幅度。
   - 用户对量化改变对模型训练的操作影响表示好奇，建议测试各种配置以最大化速度和实用性。
- **微调过程与监控**：用户分享了在训练期间监控模型 Loss 值的经验，以及持续低 Loss 指标的含义。社区强调需要使用 Weights & Biases 等工具来可视化训练动态并有效地比较模型。
   - 一些参与者指出，尽管报告的 Loss 较低，但仍应批判性地观察实际的模型性能，因为结果可能仍不尽如人意。
- **依赖项与环境配置**：贡献者分享了高效运行 Unsloth 的设置技巧，包括配置合适的 Python 版本环境和依赖管理，特别是使用 Docker 和 Conda。指导重点在于确保成功构建所需的库安装正确。
   - 大家一致认为，对于大型机器学习项目，隔离依赖项以避免运行时冲突具有重要意义。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simonwillison.net/2025/Jan/27/llamacpp-pr/">ggml : 通过优化 SIMD 使 WASM 速度提升 2 倍</a>：Xuan-Son Nguyen 为 `llama.cpp` 提交的 PR：&gt; 此 PR 通过利用 SIMD 指令为 `qX_K_q8_K` 和 `qX_0_q8_0` 点积函数带来了 WASM 速度的巨大飞跃。&gt; &gt; …</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.2_(1B_and_3B)-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/748">RuntimeError: Unsloth: 文件 'llama.cpp/llama-quantize' 或 'llama.cpp/quantize' 不存在 · Issue #748 · unslothai/unsloth</a>：在尝试将模型转换为 GGUF 格式时发生了以下错误。我注意到 quantized 文件夹位于 llama.cpp/examples/quantize，RuntimeError: Unsloth: The file 'llama.cpp/llama-quanti...</li><li><a href="https://docs.unsloth.ai/basics/finetuning-from-last-checkpoint">从最后一个 Checkpoint 继续微调 | Unsloth 文档</a>：Checkpointing 允许您保存微调进度，以便暂停后继续。
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1333816866182795264)** (1 messages): 

> `DeepSeek, Operator gist` 


- **通过 Operator 精通 DeepSeek**：一场讨论强调了如何结合 **Operator gist** 有效使用 **DeepSeek**，建议用户[为该 gist 点赞 (star) 并分享](https://gist.github.com/awdemos/f1fd4d1de0116b5f1df6936e219bafed)以扩大影响力。
   - *比 OpenAI 节省 200 美元！* 被作为尝试该工具的一个极具吸引力的理由，并辅以消息中分享的图片。
- **用于 AI Assistants 的 DeepSeek-R1**：**DeepSeek-R1** 被介绍为构建终极 AI 助手的一种手段，并通过专门的指南提供了一条精简的路径。
   - 该指南在 **DeepSeekR1AssistantAICareerPathDev.md** 中有详细说明，重点关注提升 AI 能力的实际步骤。



**提及的链接**：<a href="https://gist.github.com/awdemos/f1fd4d1de0116b5f1df6936e219bafed">DeepSeek-R1 Mastery: Build Your Ultimate AI Assistant</a>: DeepSeek-R1 Mastery: Build Your Ultimate AI Assistant - DeepSeekR1AssistantAICareerPathDev.md

  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1333580160514920681)** (13 messages🔥): 

> `Embeddings and Vector Precision, Azure OpenAI Assistants Code Interpreter, Azure Databricks AI Agent Framework, ReAct Agents with Code Sandbox, Sandbox Execution Environments` 


- **探索 Embedding 与向量精度的差异**：一位成员表示好奇，是否存在现成的引擎可以**交叉比较**来自不同量化 Embedding 模型的向量精度。
   - *他们指出自己的理解可能不够准确*，暗示在这一领域需要进一步探索。
- **Azure 为 Agents 提供沙箱**：一位成员指出 **Azure** 以服务的形式为 Agents 提供沙箱，允许为代码提供沙箱化的执行环境。
   - 另一位成员对了解此功能的更多信息表现出兴趣，并询问了任何潜在的限制。
- **深入探讨 Code Interpreter 功能**：分享的 Azure 文档链接显示，**Code Interpreter** 允许 AI 助手在**沙箱化**环境中编写和运行 Python 代码。
   - 有人指出，在 Azure OpenAI 中使用 Code Interpreter 会产生**额外费用**，这可能会影响使用。
- **在 Azure Databricks 中创建 Code Interpreter 工具**：围绕使用 Azure Databricks 中的 Mosaic AI Agent Framework 构建 **code interpreter 工具**进行了讨论。
   - 此过程使 AI Agents 能够在临时盒子中执行用户提供或 Agent 编写的代码。
- **关于沙箱执行的误解**：成员们反思了最初对于沙箱环境用途的过度复杂化，意识到它只是简单地允许执行任意代码。
   - 当他们都意识到自己对这个简单的功能想得太多时，出现了一个轻松的时刻。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/azure/databricks/generative-ai/agent-framework/code-interpreter-tools">Code interpreter AI agent tools - Azure Databricks</a>：使用 Mosaic AI Agent Framework 创建 AI Agent 工具，以执行由交互用户提供、从代码库检索或由 Agent 编写的任意代码。</li><li><a href="https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/code-interpreter?tabs=python">How to use Azure OpenAI Assistants Code Interpreter - Azure OpenAI</a>：了解如何使用 Assistants Code Interpreter
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1333529758776033322)** (624 条消息🔥🔥🔥): 

> `DeepSeek R1 限制, 与 OpenAI 模型对比, DeepSeek 性能与用法, Perplexity 应用更新, AI 模型用户体验` 


- **DeepSeek R1 存在使用限制**：用户确认 DeepSeek R1 目前每天有 10 到 15 次查询的限制，这引起了 Pro 用户的挫败感。
   - 讨论中提到，随着使用情况趋于稳定，这些限制可能会有所放宽。
- **DeepSeek R1 与 OpenAI O1 的对比**：几位用户表示 DeepSeek R1 的表现与 OpenAI 的 O1 模型相似，但对其较慢的响应速度和偶尔出现的不完整回答表示担忧。
   - 虽然 DeepSeek 在某些领域具有优势，但用户注意到模型之间在推理和审查影响方面存在差异。
- **审核与审查问题**：DeepSeek 网站有一个审核机器人，可能会删除批评中国政府的消息，这与 Perplexity 中更不受限的使用形成了对比。
   - 用户反思了这种审核的影响，以及相比直接使用 DeepSeek，使用 Perplexity 的优势。
- **AI 模型用户体验**：一些用户分享了他们使用各种模型的经验，强调了 R1 的独特功能及其在特定任务（如角色扮演和编程）中的吸引力。
   - 许多人表示希望在 Perplexity 平台内就功能可用性和使用限制进行更清晰的沟通。
- **Perplexity 应用与 Web 界面**：讨论内容包括 Perplexity 应用内模型选择界面的变化，Pro 搜索设置中转向强调 R1。
   - 用户注意到移动端应用目前缺少桌面版的一些功能，呼吁进行更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.cplx.app/">Complexity</a>：每个人都梦寐以求的 Perplexity.ai 增强版。</li><li><a href="https://perplexity.supply">Perplexity Supply</a>：好奇心与品质的交汇点。我们的精品系列为好奇者提供精心设计的服饰。从重磅棉质基础款到刺绣单品，每一件单品都体现了我们的执...</li><li><a href="https://www.searchenginejournal.com/perplexity-ai-deploys-chinese-deepseek-ai-model/538452/?utm_source=chatgpt.com">Perplexity AI 部署中国 DeepSeek AI 模型</a>：Perplexity AI 在其 AI 搜索引擎上提供自托管版本的中国 DeepSeek R1 推理模型。</li><li><a href="https://tenor.com/view/spider-man-we-one-gif-18212100">Spider Man GIF - Spider Man We - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1i8rujw/notes_on_deepseek_r1_just_how_good_it_is_compared/">Reddit - 深入探索一切</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1333567346215751765)** (17 条消息🔥): 

> `AI 研发药物、JFK 和 MLK 文件解密、禁用 Epic Games、自定义外壳、Plantin 字体` 


- **AI 研发药物即将问世**：一段名为 *AI-Developed Drugs Coming Soon* 的视频讨论了 AI 驱动的药物研发领域的最新进展。你可以在[这里](https://www.youtube.com/embed/Shu0_fB7jZ0)观看。
   - 该话题强调了 AI 在医疗领域的变革潜力。
- **近期解密：JFK 和 MLK 文件**：与 **JFK** 和 **MLK** 相关的档案解密引发了关于其历史意义的讨论。在[此处](https://www.perplexity.ai/)了解更多关于该话题的信息。
   - 这一事件引起了公众对这些档案材料内容的关注和询问。
- **禁用 Epic Games：操作指南**：分享了一份关于如何禁用 **Epic Games** 功能的指南，旨在满足希望修改游戏体验的用户需求。点击[此处](https://www.perplexity.ai/page/how-to-disable-epic-games-forc-ConH1G3ESv260wTAp_Orbg)查看。
   - 该资源可以帮助那些对当前 Epic Games 界面感到不知所措的用户。
- **自定义外壳构建详解**：围绕构建自定义外壳的讨论为爱好者提供了见解和实用建议。更多详情请见[此链接](https://www.perplexity.ai/search/when-building-a-custom-enclosu-8gCvIioIT9W7F9OJaBxEzw)。
   - 该话题旨在帮助寻求创新构建方法的 DIY 爱好者。
- **了解 Plantin 字体**：关于 **Plantin Typeface** 的讨论探索了其历史背景和设计特点。在[此处](https://www.perplexity.ai/search/why-was-the-plantin-typeface-c-6wifAj71T9S5CAU5ck8m8w)发现更多见解。
   - 该话题深入探讨了排版对设计和印刷实践的影响。



**提及的链接**: <a href="https://www.youtube.com/embed/Shu0_fB7jZ0">YouTube</a>: 未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1333550186223636663)** (1 条消息): 

> `Sonar 响应问题、Sonar 与 Sonar-Pro 之间的成本差异` 


- **Sonar 的无效 JSON 响应**：一位成员报告称，在使用 **sonar** 时配合 **response_format** 会产生被 Markdown 包裹的无效 JSON，导致使用不便。
   - 相比之下，使用 **sonar-pro** 可以有效解决此问题，但由于其成本较高，并非首选方案。
- **Sonar-Pro 的成本担忧**：同一位成员对 **sonar** 和 **sonar-pro** 之间的成本差异表示担忧，因为后者价格昂贵。
   - 为了稳定运行而必须使用 **sonar-pro** 给用户带来了经济负担。


  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1333529808939913406)** (401 条消息🔥🔥): 

> `DeepSeek API 问题，在 Aider 中使用模型，Qwen 2.5-Max，Groq 性能，Token 使用与成本` 


- **DeepSeek API 停机**: 用户报告了过去 24-48 小时内 **DeepSeek API** 出现的严重停机和可靠性问题，促使大家寻找替代供应商。
   - 几位用户建议在中断期间通过 **OpenRouter** 和 **Fireworks** 等供应商访问 DeepSeek V3。
- **为 Aider 配置模型**: 用户讨论了 **Aider** 的配置，特别是使用 **DeepSeek R1** 作为架构模型（architect model），并使用 **Qwen 2.5-Coder** 作为编程任务的编辑器模型（editor model）。
   - 关于切换模式的命令用法，以及 Aider 中主模型与编辑器模型之间的关系，存在一些困惑。
- **Qwen 2.5-Max 发布**: **Alibaba Qwen** 宣布推出 Qwen 2.5-Max，这是一款大型 MoE 模型，在多项基准测试中表现优于 DeepSeek V3，引起了 AI 社区的关注。
   - 提供了通过 API 和聊天界面访问 Qwen 的详细信息，以及相关资源链接和性能对比。
- **Groq 的性能与模型服务**: 用户强调 **Groq** 是运行 **DeepSeek R1** 等模型的高效选择，声称其响应速度比传统方法更快。
   - 用户对如何在 Groq 上优化 **R1 distilled** 版本以实现更快速、更高效的处理表现出浓厚兴趣。
- **Token 使用与成本考量**: 围绕使用 Aider 配合各种模型进行编程任务的 Token 使用量和成本展开了讨论，一些用户因性价比而更倾向于使用 **Sonnet**。
   - 用户分享了对消费模式的见解，以及使用不同模型组合以实现效率最大化的潜在收益。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://aider.chat/2025/01/28/deepseek-down.html">DeepSeek V3 替代供应商</a>: DeepSeek 的 API 一直存在可靠性问题。这里是你可以使用的替代供应商。</li><li><a href="https://aider.chat/docs/llms/deepseek.html">DeepSeek</a>: Aider 是你终端里的 AI 配对编程助手</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>: 未找到描述</li><li><a href="https://openrouter.ai/">OpenRouter</a>: LLM 的统一接口。为你的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: DeepSeek R1 – 供应商状态</a>: 查看供应商状态并对 DeepSeek R1 发起负载均衡请求：DeepSeek R1 已发布，性能与 [OpenAI o1](/openai/o1) 相当，但是开源的且具有完全开放的推理 Token...</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 🎉 恭喜发财🧧🐍 在迎接农历新年之际，我们激动地宣布推出 Qwen2.5-VL，我们最新的旗舰级视觉语言模型！🚀💗 Qwen Chat: https://chat.qwenlm.ai📖 Blog: http...</li><li><a href="https://x.com/alibaba_qwen/status/1884263157574820053?s=46&t=AZs45ckJ7UUM_kJZcxnR_w">来自 Qwen (@Alibaba_Qwen) 的推文</a>: DeepSeek V3 的爆发吸引了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM...</li><li><a href="http://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>: Aider 是你终端里的 AI 配对编程助手</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.bbc.com/portuguese/articles/cdd9m3rp271o">DeepSeek: 普及度超越 ChatGPT 的中国应用 - BBC News Brasil</a>: 据估计，随着这家颠覆了 AI 行业逻辑的中国竞争对手取得成功，竞争对手科技公司的市值已损失 1 万亿美元...</li><li><a href="https://digitalspaceport.com/running-deepseek-r1-locally-not-a-distilled-qwen-or-llama/">在本地运行 DeepSeek R1（非蒸馏版 Qwen 或 Llama） – 设置与运行笔记 – Digital Spaceport</a>: 未找到描述</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: 世界上最小的 AI 超级计算机。</a>: 立即预订。</li><li><a href="https://digitalspaceport.com/running-deepseek-r1-locally-not-">在本地运行 DeepSeek R1（非蒸馏版 Qwen 或 Llama） – 设置与运行笔记 – Digital Spaceport</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1333532180827345047)** (128 messages🔥🔥): 

> `Deepseek API 问题，Ollama 模型使用，Aider 配置，LLM 基准测试，ChatGPT 集成` 


- **Deepseek API 挂起**：用户报告了 **Deepseek API** 的持续问题，在使用过程中出现挂起和功能失效。
   - 尽管根据状态页面显示 Deepseek 处于运行状态，但许多人仍面临性能问题，并对其可靠性表示担忧。
- **在 Aider 中使用 Ollama 模型**：几位用户正在处理 **Ollama** 模型的问题，难以获得预期的响应或文件访问权限。
   - 用户交流了配置建议，补充了在 Aider 中正常运行所需的参数和环境设置背景。
- **Aider 配置与最佳实践**：讨论内容包括配置 **Aider** 的方法，包括 API key 存储以及使用 `.aider.config.yaml` 文件进行持久化配置的有效性。
   - 用户分享了优化工作流的技巧，例如使用 `--read FILE` 命令为 LLM 提供上下文。
- **使用 Polyglot 对 LLM 进行基准测试**：用户对在各种模型（特别是 R1）上运行 **Polyglot 基准测试**工具表现出兴趣，以评估其在 architect 模式下的性能。
   - 虽然设置基准测试需要注意细节，但据指出，遵循文档操作可以获得可靠的结果。
- **GitHub 与资源共享**：社区一直在参考各种资源（如 [Aider 文档](https://aider.chat/docs)），以深入了解配置和用法。
   - 许多成员正在探索不同 LLM 能力的交集，并分享与 Aider 特定集成相关的发现。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/config/api-keys.html">API Keys</a>：为 API 提供商设置 API key。</li><li><a href="https://aider.chat/docs/llms/ollama.html#setting-the-context-window-size">Ollama</a>：Aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/llms/openrouter.html">OpenRouter</a>：Aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/copypaste.html">与 Web 聊天界面进行复制/粘贴</a>：Aider 可与 LLM Web 聊天 UI 配合使用</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：在多个提供商之间路由请求</li><li><a href="https://tenor.com/view/steve-harvey-praise-gif-8670793">Steve Harvey 赞美 GIF - Steve Harvey Praise - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>：LLM 代码编辑能力的定量基准测试。</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://github.com/Aider-AI/aider/blob/main/benchmark/README.md">aider/benchmark/README.md at main · Aider-AI/aider</a>：Aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 做出贡献。</li><li><a href="https://github.com/BerriAI/litellm/issues/6857">[特性]：支持 OpenRouter 的 "provider" 参数以控制/选择提供商 · Issue #6857 · BerriAI/litellm</a>：该特性使 OpenRouter 支持多种机制来选择你希望请求命中的提供商。这涉及传递 provider 参数。目前这会导致错误：import li...</li><li><a href="https://aider.chat/docs/usage/tips.html#providing-docs">技巧</a>：使用 Aider 进行 AI 配对编程的技巧。</li><li><a href="http://aider.chat/docs/repomap.html">仓库映射 (Repository map)</a>：Aider 使用 Git 仓库映射为 LLM 提供代码上下文。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1333527224414109697)** (517 条消息🔥🔥🔥): 

> `DeepSeek R1 vs V3, Cursor updates, Experiences with coding models, Quantization effects, Using different AI models` 


- **DeepSeek R1 性能担忧**：用户对 Cursor 中 DeepSeek R1 的性能表示沮丧，强调了量化（quantization）导致的结果不如 DeepSeek 官网原版的问题。
   - 许多人认为，尽管 R1 具备预期的能力，但往往无法提供预期的输出，从而对其在编码任务中的有效性产生了怀疑。
- **Cursor 的更新与功能**：Cursor 经历了重大更新，引入了新功能和模型，包括 DeepSeek 以及编码能力的各种增强。
   - 这些改进褒贬不一，一些人称赞功能有所增加，而另一些人则批评性能不一致。
- **AI 编码的困扰**：几位用户分享了使用 AI 模型进行编码的经验，指出虽然它们可以简化某些流程，但也带来了诸如调试困难等挑战。
   - 许多人建议使用版本控制和检查点（checkpoints）来应对 AI 建议引发的问题。
- **对新 AI 模型的期待**：人们对即将推出的 o3-mini 等 AI 模型感到兴奋，期待它们能显著增强编码能力和整体性能。
   - 用户分享了对 Sonnet 等现有模型改进的期待，强调了它们对编码工作流的潜在影响。
- **编码中的多样化 AI 解决方案**：由于在不同 AI 工具上的体验各异，用户注意到，出于可靠性考虑，一些本地模型正被考虑用来替代云端解决方案。
   - 对话强调了在为特定任务选择合适模型时适应性的重要性，引发了关于利用多个模型以获得更好结果的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://blog.voyageai.com/2024/12/04/voyage-code-3/#:~:text=voyage%2Dcode%2D3%20supports%20much,Matryoshka%20embeddings.">voyage-code-3: more accurate code retrieval with lower dimensional, quantized embeddings</a>: TL;DR – 介绍 voyage-code-3，我们为代码检索优化的下一代嵌入模型。在一系列测试中，它的表现平均优于 OpenAI-v3-large 和 CodeSage-large 13.80% 和 16.81%……</li><li><a href="https://console.groq.com/">GroqCloud</a>: 体验世界上最快的推理</li><li><a href="https://www.cursor.com/downloads">Cursor - The AI Code Editor</a>: 旨在让你获得非凡的生产力，Cursor 是使用 AI 编码的最佳方式。</li><li><a href="https://screen.studio/">Screen Studio — Professional screen recorder for macOS</a>: macOS 专业屏幕录制工具。创建引人入胜的产品演示、课程、教程和社交媒体视频。在鼠标操作时添加自动缩放、平滑鼠标移动以及其他强大的效果和动画……</li><li><a href="https://forum.cursor.com/t/cursor-does-not-send-files-to-claude/43948">Cursor does not send files to Claude</a>: Cursor 无法向 Claude 发送文件数据的示例，这已经是连续第三次了。这些问题正在消耗我宝贵的 Fast-Replies，而且经常发生。我正在使用的版本是：Version: ...</li><li><a href="https://fxtwitter.com/alibaba_qwen/status/1884263157574820053?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Tweet from Qwen (@Alibaba_Qwen)</a>: DeepSeek V3 的爆发引起了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM……</li><li><a href="https://aistudio.google.com/apikey">no title found</a>: 未找到描述</li><li><a href="https://console.groq.com/docs/models">GroqCloud</a>: 体验世界上最快的推理</li><li><a href="https://fireworks.ai/blog/fireworks-quantization">How Fireworks evaluates quantization precisely and interpretably </a>: 深入探讨 Fireworks AI 如何思考量化，并使用散度指标（divergence metrics）来确保质量并为用户创建定制解决方案</li><li><a href="https://github.com/vbwyrde/AI_Dev_Helpers">GitHub - vbwyrde/AI_Dev_Helpers: Some tools that I find useful when using AI for development</a>: 我发现在使用 AI 进行开发时很有用的一些工具 - vbwyrde/AI_Dev_Helpers</li><li><a href="https://www.cursor.com/changelog">Changelog | Cursor - The AI Code Editor</a>: 新的更新和改进。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1333528283853230222)** (466 条消息🔥🔥🔥): 

> `DeepSeek vs. OpenAI, AI 意识辩论, AI 审查制度, AI 模型定价竞争, AI 模型用户体验` 


- **DeepSeek 表现优于 OpenAI 和 Nvidia**：许多用户认为 DeepSeek 的免费模型与 OpenAI 的付费订阅相比具有竞争力的功能，特别是强调了其 128k tokens 的更大 Context Window（上下文窗口），而 OpenAI 仅为 32k。
   - 新兴的竞争，特别是来自 DeepSeek 的竞争，被视为可能推动 OpenAI 改进其产品并重新考虑其定价策略的催化剂。
- **关于 AI 意识的辩论**：讨论强调，定义意识是跨多个学科的复杂挑战，用户对 AI 系统的意识表示怀疑。
   - 在对 AI 觉知的怀疑与宗教信仰之间划出了平行线，表明了意识的哲学性和未定义性。
- **AI 之间的审查差异**：包括 DeepSeek 和 Claude 在内的各种 AI 因其不同程度的审查而受到关注，一些用户对现行的严格审核政策表示沮丧。
   - 据报道，DeepSeek 允许更自由的对话，特别是在敏感话题方面，而 OpenAI 的限制被认为过于局限。
- **AI 模型定价竞争**：用户批评 OpenAI 高昂的订阅费，指出像 DeepSeek 这样的竞争对手以免费或更低的成本提供了类似或更优的功能。
   - 这种价格差异被视为负面影响 OpenAI 市场地位和股票表现的重要因素。
- **AI 模型用户体验**：几位用户分享了他们与 DeepSeek 互动的个人轶事，报告了它在解决其他模型难以处理的问题方面的有效性。
   - 用户对当前订阅模式的局限性感到沮丧，从而引发了关于其与免费替代方案相比的价值讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.nextplatform.com/2024/10/25/cerebras-trains-llama-models-to-leap-over-gpus/">Cerebras 训练 Llama 模型以超越 GPU</a>：就在几个月前，晶圆级计算先驱 Cerebras Systems 还在炫耀其少数几台 WSE-3 引擎连接在一起即可运行。</li><li><a href="https://fortune.com/2025/01/27/mark-zuckerberg-meta-llama-assembling-war-rooms-engineers-deepseek-ai-china/">据报道，Meta 正在紧急召集多个工程师“作战室”，以弄清楚 DeepSeek 的 AI 是如何以极低的价格击败其他所有人的</a>：工程师们将研究 DeepSeek 是如何实现其技术飞跃的，以及 Meta 如何从中受益。</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://tenor.com/view/laughing-out-loud-gif-22910989">大笑 GIF - Laughing Out Loud - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://en.wikipedia.org/wiki/Mycroft_(software)">Mycroft (软件) - 维基百科</a>：未找到描述</li><li><a href="https://newsroom.ibm.com/2024-05-21-IBM-Unveils-Next-Chapter-of-watsonx-with-Open-Source,-Product-Ecosystem-Innovations-to-Drive-Enterprise-AI-at-Scale">IBM 发布 watsonx 的新篇章，通过开源、产品和生态系统创新推动企业级 AI 规模化</a>：IBM 在其 watsonx 平台推出一年后宣布了多项更新，以及即将推出的旨在使人工智能 (AI) 更加开放的数据和自动化功能...</li><li><a href="https://youtu.be/7GV_OdqzmIU?si=IKCRD0tUOkHtOplS">Cerebras 联合创始人解析 Blackwell GPU 延迟</a>：Cerebras 首席系统架构师兼联合创始人 J.P. Fricker 解释了 Nvidia Blackwell 的技术挑战。00:12 中介层介绍 02:54...
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1333783165202075690)** (2 messages): 

> `Custom GPTs URL 输出，使用零宽空格处理链接` 


- **Custom GPTs 在 URL 格式化方面存在困难**：一位成员询问如何强制 Custom GPTs 将链接输出为带有完整路径的 URL，而不是锚文本，尽管他们已经使用了大量的 Python 代码来抓取底层 URL。
   - 这种挫败感源于 GPT 总是将链接格式化回锚文本，而不是所需的 URL 格式。
- **使用零宽空格防止链接格式化**：另一位成员建议使用一种不可见的“零宽”空格字符（如 `httpXs://`）来阻止文本被重新格式化为链接。
   - 他们提到之前在 StackOverflow 上写过关于这种技术的文章，并指出该方法在各种软件中都非常有效。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1333802004518141984)** (21 messages🔥): 

> `向模型喂入内容、模仿作者、使用 AI 进行高级搜索、训练 AI 的成本和时间、模型回答的可信度` 


- **向模型喂入内容以获取回复**：一位用户询问是否可以向模型喂入 **10-15 本书**，然后像采访作者一样向其提问。
   - 另一位成员澄清说，虽然这在理论上很有趣，但要实现真正的作者模仿是不现实的。
- **使用 AI 进行高级搜索**：一位用户建议将 AI 视为一种用于书籍引用和页码查找的**高级搜索**工具。
   - 一位成员确认这是可能的，但指出 AI 在长时间查询过程中容易产生 **hallucination**（幻觉）。
- **模仿作者的挑战**：讨论区分了“模仿作者”与“拥有一个深度理解书籍的人”之间的不同。
   - 有人建议，塑造一个热爱这些书籍的角色可能比直接模仿作者**更可行**。
- **喂入模型的成本和时间**：在回答关于成本的问题时，有人指出对于 **10 GB** 以下且少于 **20** 本的书籍，使用 **ChatGPT Plus** 是可行的。
   - 然而，版权内容可能会带来限制，导致成本难以估算。
- **评估模型的回复**：讨论涉及了模型的回答与**作者本人**的回答相比如何。
   - 讨论转向了评估模型回复的可信度与实际作者协作之间的对比。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1333802004518141984)** (21 messages🔥): 

> `向 AI 喂入内容、模仿作者、将 AI 用作高级搜索工具、训练模型的挑战、训练 AI 的成本和时间` 


- **向 AI 喂入内容以获取回复**：成员们讨论了将书籍等内容喂入模型并基于该信息进行回答的可能性。虽然 AI 可以回答关于书籍的问题，但它可能无法准确地模仿作者。
   - 一位成员指出，由于涉及的复杂性，让模型完全反映作者的口吻*在实际操作上是不可行的*。
- **将 AI 作为高级书籍搜索工具**：有人建议，与其模仿作者，不如将书籍喂给 AI，将其视为定位引用和页码的高级搜索工具。成员们一致认为这是可能的，但提醒说在长对话中可能会产生 hallucination。
   - 一位成员强调，可行性在很大程度上取决于书籍的大小和复杂程度。
- **AI 训练的成本影响**：关于用 10-15 本书训练 AI 所涉及的成本和时间出现了疑问。回复指出，只要书籍在 10 GB 以下且少于 20 本，就可以利用 ChatGPT Plus。
   - 然而，对于大多数内容被屏蔽可能导致的版权问题存在担忧。
- **模仿作者的挑战**：大家达成共识，模仿作者极具挑战性，因为这涉及到书籍内容之外的复杂细微差别。一位成员建议，与其模仿作者本人，不如尝试由一个深度理解这些书籍的人来进行表达。
   - 这场持续的辩论强调了需要作者的配合，以及模型的回复需要具备多大的解释性。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1333527530749296761)** (496 messages🔥🔥🔥): 

> `Nous Psyche, DeepSeek 模型, AI 推理, 股票预测, AI 的商业应用`

- **Nous Psyche 发布**：社区讨论了最近发布的 Nous Psyche，这是一个建立在 Solana 上的协作训练网络，强调了其在 AI 领域的重要性。
   - 参与者指出 Psyche 有潜力影响个人 Agent，并强调了其在当前 AI 发展中的重要意义。
- **DeepSeek 模型性能**：成员们对 DeepSeek V3 和 R1 模型之间的价格差异表示困惑，怀疑 R1 的新颖性导致了较高的初始成本。
   - 有人指出，流量的增加以及针对新模型的先进优化也可能影响这些定价结构。
- **AI 推理与输出**：参与者讨论了 AI 推理的本质，特别是将 LLM 输出与人类推理进行比较，强调了置信度值与真实概率模型之间的区别。
   - 大家公认，虽然 LLM 可以模仿推理模式，但它们可能缺乏在更先进模型中发现的底层随机过程。
- **AI 在商业中的应用**：对话转向了 AI 的实际应用，如客户服务和编程助手，并提出了关于 AI 驱动解决方案潜在收益的问题。
   - 一位成员询问了为特定业务功能（如呼叫中心）训练 DeepSeek 版本的可行性，表明了对实际应用的兴趣。
- **理解 AI 机制**：几位参与者分享了他们对网络模型中 dropout 等层功能的看法，强调了对这些机制进行更深入理解的需求。
   - 讨论对 AI 模型运作方式与人类推理能力的差异提出了不同的观点。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/N8Programs/status/1884110306089357361">来自 N8 Programs (@N8Programs) 的推文</a>：正在阅读一篇 DeepSeek 论文，偶然发现了一个非常优美的公式，他们将 SFT 和大多数 RL 类型（DPO, PPO, GRPO 等）统一到了一个公式中，这需要定义额外的奖励函数...</li><li><a href="https://x.com/tekbog/status/1884284826028798083">来自 terminally onλine εngineer 🇺🇦 (@tekbog) 的推文</a>：先生，又一个模型刷屏了。引用 Qwen (@Alibaba_Qwen)：DeepSeek V3 的爆发引起了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建...</li><li><a href="https://x.com/Alibaba_Qwen/status/1884263157574820053">来自 Qwen (@Alibaba_Qwen) 的推文</a>：DeepSeek V3 的爆发引起了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM...</li><li><a href="https://tenor.com/view/psychonauts-psychonauts-raz-raz-aquato-razputin-aquato-shaking-him-gif-18198955129765614191">Psychonauts Psychonauts Raz GIF - Psychonauts Psychonauts raz Raz aquato - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/sybilhyz/status/1884271592978669579">来自 Peiyi Wang (@sybilhyz) 的推文</a>：去年，我在没有任何 RL 经验的情况下加入了 DeepSeek。在进行 Mathshepherd 和 DeepSeekMath 研究时，我独立推导出了这个统一公式来理解各种训练方法。感觉...</li><li><a href="https://x.com/NousResearch/status/1883912370696704011">来自 Nous Research (@NousResearch) 的推文</a>：最近的 AI 突破挑战了现状，即只有封闭的大型实验室才有能力推动超级智能的前沿。今天我们宣布在 @Solana 上构建 Nous Psyche - 一个...</li><li><a href="https://x.com/markchen90/status/1884303237186216272?t=Hg8luocUfGmNBuH3JHgvUw&s=19">来自 Mark Chen (@markchen90) 的推文</a>：祝贺 DeepSeek 制作出了 o1 级别的推理模型！他们的研究论文表明，他们在通往 o1 的道路上独立发现了一些与我们相同的核心理念。</li><li><a href="https://shop.nousresearch.com/collections/products">购买我们的产品</a>：Nous Research</li><li><a href="https://vcomp.eqtylab.io">EQTY Lab — 介绍可验证计算</a>：通过首个可审计的治理证明来认证和保护 Agentic AI 工作流。</li><li><a href="https://www.youtube.com/watch?v=0uK7w0zCxA8">大科技公司的 Alex Kantrowitz 表示，如果 AI 要具有革命性，我们需要看到这场革命</a>：Big Technology 的 Alex Kantrowitz 和 Alger 的 Dan Chung 加入 'Closing Bell' 讨论 AI 电力需求、投资以及对技术行业情绪的转变...</li><li><a href="https://www.youtube.com/watch?v=ZjQ6ZxA1jeQ">特朗普表示，AI 颠覆者 DeepSeek 是对美国科技界的“警钟”</a>：在此免费订阅我们的 YouTube 频道：https://sc.mp/subscribe-youtube 更多关于此故事的内容：https://sc.mp/i5gz6 美国总统唐纳德·特朗普称...</li><li><a href="https://en.m.wikipedia.org/wiki/Nous">Nous - 维基百科</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/ServiceNow-AI/R1-Distill-SFT">ServiceNow-AI/R1-Distill-SFT · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1333532875626123489)** (4 条消息): 

> `开发本地 AI 助手, AI 开发学习资源, 优化学习速度` 


- **启动你的本地 AI 助手**：*许多成员讨论了初学者开发自己本地 AI 助手的方法，强调它们不必基于 Meta 的 **Llama**。*
   - 有人建议使用 [DeepSeek](https://deepseek.ai), **Hermes** 或 **ChatGPT** 等平台作为辅助，创建促进学习的项目。
- **利用社区和资源**：参与者强调，结合 **LLM**、互联网资源、社区参与和项目实践是学习 AI 开发的坚实基础。
   - 这些工具创造了一个多样化的学习环境，使复杂的主题对初学者来说更加平易近年。
- **关注学习速度**：*一位成员表示，如果经济条件允许，优化 **学习速度 (learning velocity)** 非常重要，并建议投资教育资源可以加速理解。*
   - 这种方法强调定制学习体验，以最大限度地提高效率和效果。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1333527560616804372)** (7 条消息): 

> `Qwen2.5-VL 模型, YuE 音乐生成模型, AI 助手详解, Deepseek 与 Operator 使用` 


- **Qwen2.5-VL 在 OCR 等领域表现卓越**：新发布的 **Qwen2.5-VL** 模型在 OCR 任务（包括手写识别）中表现出色，并具备视觉推理和理解各种图形元素的能力。
   - 自 **Qwen2-VL** 发布以来，开发者们一直在提供反馈，从而改进了其在图像识别和分析方面的功能。
- **YuE 音乐生成模型发布**：[YuE](https://github.com/multimodal-art-projection/YuE) 是一个类似于 **Suno.ai** 的开源全曲生成基础模型，能够在本地 GPU 上运行进行音乐创作。
   - 社区成员对其训练数据以及在生成多样化音乐作品方面的整体性能感到好奇。
- **面向初学者的 AI 助手拆解**：一位成员分享了一篇题为《图解 AI 助手：视觉化解释》的博客文章，旨在帮助非技术用户理解 **ChatGPT** 等 AI 系统的运作原理。
   - 该资源讨论了 AI 模型中“训练”和“学习”背后的基本概念，旨在教育更广泛的受众。
- **结合 Operator 的 Deepseek Gist 助力 AI 职业生涯**：发布了一份关于结合 **Operator** 使用 **Deepseek** 的新指南，宣称与 **OpenAI** 的解决方案相比可节省 200 美元的成本。
   - 该帖子鼓励用户点赞并分享 [此处](https://gist.github.com/awdemos/f1fd4d1de0116b5f1df6936e219bafed) 的 Gist，以帮助他人构建终极 AI 助手。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/PhoBoAI/status/1884224914015805790)">Hai Duong 的推文 </a>：我写了一篇 @JayAlammar 风格的博客文章，向更广泛的非技术公众介绍 ChatGPT 等 AI 助手在底层是如何工作的。我涵盖了语言模型实际在做什么，并解释了……</li><li><a href="https://gist.github.com/awdemos/f1fd4d1de0116b5f1df6936e219bafed">DeepSeek-R1 精通：构建你的终极 AI 助手</a>：DeepSeek-R1 Mastery: Build Your Ultimate AI Assistant - DeepSeekR1AssistantAICareerPathDev.md</li><li><a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/multimodal-art-projection/YuE">GitHub - multimodal-art-projection/YuE: YuE: 开源全曲生成基础模型，类似于 Suno.ai 但开源</a>：YuE: Open Full-song Generation Foundation Model, something similar to Suno.ai but open - multimodal-art-projection/YuE
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1333527609610342550)** (308 条消息🔥🔥): 

> `DeepSeek R1 蒸馏模型, 模型性能与对比, 量化技术, 工具化与网页浏览能力, 模型与硬件的兼容性`

- **R1 Distilled 模型加载问题**：用户报告在 LM Studio 中加载 R1 Distilled Qwen 模型时出现错误，特别是提到 'unknown pre-tokenizer type'。建议用户更新到最新版本的 LM Studio，并确保 LM Runtimes 也已更新。
   - 社区指南强调了跨平台匹配模型版本以避免差异的重要性。
- **Llama 与 Qwen 模型对比**：讨论涉及 Llama 8B 和 Qwen 7B 模型之间的差异，并确定了参数量和架构上的区别。用户对为什么尽管 8B 版本参数更大，选择它的人却较少表示困惑。
   - 有人指出，模型的易用性或性能可能在用户偏好中起到了作用。
- **量化技术解析**：讨论了模型量化（Quantization）的话题，揭示了其中涉及的复杂性和方法，特别是 Legacy 与 K/I 量化类型之间的对比。大家一致认为，深入了解量化基础知识对用户大有裨益。
   - 鼓励用户在较小的数据集上尝试量化，以掌握其运作方式。
- **工具化与网页浏览集成**：澄清了模型的工具化（Tooling）能力以及网页浏览功能对额外软件的需求。社区期待未来有更多工具被分享并集成到 LM Studio 中，以便用户更轻松地访问。
   - 强调了专门为工具化训练的模型与不具备该能力的通用模型之间的区别。
- **模型与硬件的兼容性**：用户讨论了模型在不同硬件上的性能，特别是 AMD GPU（如 6700XT 和 6800 型号），并指出不同模型的成功程度各异。有人指出，较小的模型在显存（VRAM）受限的硬件配置上表现更好。
   - 还强调了在 Windows 上使用 AMD GPU 运行模型时，使用 Vulkan runtime 确保兼容性的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@neuralweights/you-dont-need-fine-tuning-embeddings-comprehensive-step-by-step-guide-0b1a368bf4c1">You Don’t Need Fine-Tuning: Embeddings. Comprehensive Step-by-Step Guide.</a>: 简介</li><li><a href="https://ollama.com/library/deepseek-r1">deepseek-r1</a>: DeepSeek 的第一代推理模型，性能与 OpenAI-o1 相当，包括六个基于 Llama 和 Qwen 从 DeepSeek-R1 蒸馏出的稠密模型。</li><li><a href="https://stats.stackexchange.com/questions/346299/whats-the-effect-of-scaling-a-loss-function-in-deep-learning">What&#x27;s the effect of scaling a loss function in deep learning?</a>: 我在一个损失函数量级非常小的问题上训练网络。我观察到，直到我开始放大损失函数，网络才真正开始良好训练，例如...</li><li><a href="https://huggingface.co/collections/Qwen/qwen25-vl-6795ffac22b334a837c0f9a5">Qwen2.5-VL - a Qwen Collection</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/moe#what-is-a-mixture-of-experts-moe>">Mixture of Experts Explained</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/wiki/Feature-matrix">Feature matrix</a>: 使用 C/C++ 进行 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://stackoverflow.com/questions/38428726/what-happens-if-loss-function-is-multiplied-by-a-constant">What happens if loss function is multiplied by a constant?</a>: 如果我给损失函数乘以一个常数会发生什么？我认为我会得到一个更大的梯度，对吗？这是否等同于拥有更大的学习率？</li><li><a href="https://stackoverflow.com/questions/38428726/what-happens-if-loss-function-is-multiplied-by-a-consta">What happens if loss function is multiplied by a constant?</a>: 如果我给损失函数乘以一个常数会发生什么？我认为我会得到一个更大的梯度，对吗？这是否等同于拥有更大的学习率？</li><li><a href="https://huggingface.co/livecodebench">livecodebench (Live Code Bench)</a>: 未找到描述</li><li><a href="https://forms.gle/FBgAH43GRaR2cHde6">🔧 LM Studio Beta Versions Mailing List</a>: 如果您想接收有关新 LM Studio Beta 版本的电子邮件，请输入您的电子邮件地址。</li><li><a href="https://lmstudio.ai/download">Download LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLMs</li><li><a href="https://github.com/sammcj/gollama#readme>">GitHub - sammcj/gollama: Go manage your Ollama models</a>: 使用 Go 管理您的 Ollama 模型。通过在 GitHub 上创建账号为 sammcj/gollama 的开发做出贡献。</li><li><a href="https://lmstudio.ai/docs/basics/import-model">Import Models | LM Studio Docs</a>: 使用您在 LM Studio 之外下载的模型文件</li><li><a href="https://github.com/deepseek-ai/Janus">GitHub - deepseek-ai/Janus: Janus-Series: Unified Multimodal Understanding and Generation Models</a>: Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1333563936397725738)** (96 messages🔥🔥): 

> `DeepSeek-R1 模型性能, GPU 检测问题, SSD NVMe 速度影响, 70B 模型最佳规格, Apple 设备上的 Unified RAM` 


- **DeepSeek-R1 32B 达到预期 Token 速度**：在各种 GPU 上运行 DeepSeek-R1 32B 的用户报告 Token 速度约为 **25 tok/sec**，这被认为是该模型的正常水平。
   - 多位用户确认，对于 **32B 模型**，这样的速度预示着符合预期的性能。
- **解决 LM Studio 中的 GPU 检测问题**：用户报告了在使用 LM Studio 时遇到的 GPU 检测问题，特别是 Quadro 设备，但切换到 **CUDA runtime** 后解决了该问题。
   - 截图显示在调整 runtime 后成功识别了 GPU，表明配置设置会影响硬件检测。
- **SSD NVMe 速度对模型加载的影响**：讨论强调虽然 Gen4/5 SSD 宣称速度更快，但在实际应用中性能差异往往很小，特别是对于非密集型工作负载。
   - 用户表示，对于大模型加载，低端 SSD 可能会导致明显的延迟，强调了持续读取速度对大容量模型的重要性。
- **运行 70B 模型的必要规格**：运行 DeepSeek R1 的 **70B 版本** 至少需要 **30GB VRAM**（针对低量化版本），像 **RTX 5090** 这样的消费级显卡是合适的。
   - 非消费级选项（如 **NVIDIA H100**）可以轻松处理更大的模型，并建议为上下文使用预留 VRAM 缓冲。
- **理解 Apple 设备上的 Unified RAM 性能**：有人指出，与独立 GPU 相比，Mac 设备上的**内存带宽**可能会限制 LLM 推理速度，从而影响 AI 应用的整体性能。
   - 用户澄清说，虽然 Apple 芯片可以提供不错的速度，但在面对 **RTX 3060** 及以上的独立 GPU 时，其竞争性能仍显不足。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference">GitHub - XiongjieDai/GPU-Benchmarks-on-LLM-Inference: Multiple NVIDIA GPUs or Apple Silicon for Large Language Model Inference?</a>: 多个 NVIDIA GPU 还是 Apple Silicon 用于大语言模型推理？ - XiongjieDai/GPU-Benchmarks-on-LLM-Inference</li><li><a href="https://hardwareand.co/actualites/breves/les-barrettes-ddr5-64-go-de-crucial-arrivent-qui-veut-256-go-de-ram-dans-son-pc">Les barrettes DDR5 64 Go de Crucial arrivent, qui veut 256 Go de RAM dans son PC ?</a>: Crucial 的 64GB DDR5 内存条上市，谁想在 PC 里装 256GB RAM？台式机最高可达 256GB RAM，笔记本电脑最高可达 128GB！
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1333533386475704321)** (282 messages🔥🔥): 

> `DeepSeek 与模型性能, AI 数据隐私担忧, 大模型的 VRAM 需求, AI 研究中的基准测试造假, AI 模型开发趋势` 


- **DeepSeek 展示了潜力但引发了质疑**：讨论围绕 DeepSeek 模型的有效性展开，质疑由于其依赖蒸馏技术，是否真的能与 OpenAI 平起平坐。
   - 一些人认为，如果没有新颖的算法改进，DeepSeek 的进步是有限的，大多是模仿之举。
- **对数据隐私和模型来源的担忧**：意大利监管机构对 DeepSeek 的关注引发了对数据保护以及与中国数据政策潜在联系的警觉。
   - 社交平台上流传着关于数据被发送到中国的梗图，反映了公众对数据隐私和透明度的普遍焦虑。
- **讨论大模型的 VRAM 需求**：用户探讨了如何估算 Qwen 2.5-VL-72B 等模型的 VRAM 需求，建议仅权重部分就需要约 144GB。
   - 有人指出，权重参数量化可以显著降低这些需求，使得在有限的硬件上运行大模型变得更加可行。
- **基准测试造假在 AI 研究中引起警惕**：大家达成共识，认为 AI 领域的一些基准测试存在操纵行为，导致性能声明虚高——特别是来自一些中国模型。
   - 用户表示担心，过度依赖统计数据可能会导致对模型有效性的误导性评估。
- **AI 模型开发趋势与网红文化**：参与者对当前的 AI 趋势以及 YouTuber 在影响公众认知和研究重点方面的作用表示怀疑。
   - Red_code 指出，许多人倾向于盲目追随趋势，而没有批判性地评估底层技术。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.connectedpapers.com/">Connected Papers | 查找与探索学术论文</a>：一个独特的视觉化工具，帮助研究人员和应用科学家查找并探索与其工作领域相关的论文。</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - 实用深度学习</a>：一门为具有一定编程经验、想要学习如何将 Deep Learning 和 Machine Learning 应用于实际问题的学习者设计的免费课程。</li><li><a href="https://huggingface.co/mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0">mobiuslabsgmbh/DeepSeek-R1-ReDistill-Qwen-1.5B-v1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/cat-cope-cursed-gif-25667684">Cat Cope GIF - Cat Cope Cursed - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/adder-adderko-snake-ouroboros-overwerk-gif-21047022">Adder Adderko GIF - Adder Adderko Snake - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/house-md-house-house-medical-department-nbc-parks-and-rec-gif-20055094">House Md House GIF - House Md House House Medical Department - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/chinese-ben-shapiro-ben-shapiro-social-credit-gif-6133430005061019">Chinese Ben Shapiro Social Credit GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/sama/status/1884066340509729074">Sam Altman (@sama) 的推文</a>：期待为大家带来 AGI 及更高层次的技术。</li><li><a href="https://tenor.com/view/capitulo0-capitulo-cero-ernesto-sevilla-david-lynch-chanante-gif-15470197">Capitulo0 Capitulo Cero GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/kuuchuu-buranko-ichiro-irabu-devi-word-of-the-day-irabu-ichiro-gif-26530271">Kuuchuu Buranko Ichiro Irabu GIF - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/sama/status/1884066338739830835">Sam Altman (@sama) 的推文</a>：但最重要的是，我们很高兴能继续执行我们的研究路线图，并相信为了实现我们的使命，现在比以往任何时候都更需要更多的 Compute。世界将会想要使用一个 L...</li><li><a href="https://modal.com">Modal: 高性能 AI 基础设施</a>：自带代码，大规模运行 CPU、GPU 和数据密集型计算。面向 AI 和数据团队的 Serverless 平台。</li><li><a href="https://github.com/deepseek-ai/Janus">GitHub - deepseek-ai/Janus: Janus 系列：统一多模态理解与生成模型</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://fxtwitter.com/sama/status/1884066337103962416">Sam Altman (@sama) 的推文</a>：DeepSeek 的 R1 是一个令人印象深刻的模型，特别是在其性价比方面。我们显然会提供更好的模型，而且拥有一个...确实令人振奋。</li><li><a href="https://www.youtube.com/watch?v=w2OtwL5T1ow&list=PLE6Wd9FR--EdyJ5lbFl8UuGjecvVw66F6">Machine learning - 简介</a>：Machine Learning 简介。讲义地址：http://www.cs.ubc.ca/~nando/540-2013/lectures.html。由 Nando de Freitas 于 2013 年在 UBC 授课</li><li><a href="https://scholar.google.co.in/citations?user=mRmHHhkAAAAJ">Ritwik Mishra</a>：IIITD 博士生 - 被引用 40 次 - Deep Learning - Natural Language Processing - 印度语言 - IndicNLP</li><li><a href="https://www.youtube.com/shorts/VNv-Cz-U6AY">“男子营救困在贝壳里的无助章鱼” #viralshort</a>：这段感人的视频展示了一名男子在海滩上营救一只困在贝壳里的无助章鱼。看着他在照顾和玩耍的过程中，他们之间的纽带不断加深...</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf (main 分支) · deepseek-ai/Janus</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://www.youtube.com/watch?v=Nl7aCUsWykg">大厂陷入恐慌... DeepSeek R1 戳破了 AI 泡沫吗？</a>：在 DeepSeek R1 AI 模型证明了以极低成本训练和运行最先进的推理模型是可能的之后，像 Nvidia 这样的芯片股陷入了困境...</li><li><a href="https://www.runpod.io">RunPod - 为 AI 构建的云</a>：在同一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 按需启动 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://beam.cloud">面向开发者的 AI 基础设施</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=kPRA0W1kECg">6 分钟演示 15 种排序算法</a>：15 种排序算法的可视化与“听觉化”演示</li>

6 分钟了解排序算法。对随机打乱的整数进行排序，速度和项目数量均适配...</li><li><a href="https://midas.iiitd.ac.in/students">MIDAS Lab @IIITD - 学生</a>：在读学生
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1333533738990047322)** (47 条消息🔥): 

> `Janus-Pro 发布，Qwen2.5-VL 上线，DeepSeek 进展，Emu 学习算法，量化模型开发` 


- **Janus-Pro 在多模态 AI 领域引起轰动**：今天，**DeepSeek** 发布了 **Janus-Pro: Unified Multimodal Understanding and Generation**，标志着 AI 技术的又一步跨越，详见其[技术报告](https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf)。
   - 参与者对 **DeepSeek** 在短短几个月内取得的飞速进展感到兴奋，这暗示了他们不懈的创新能力。
- **Qwen2.5-VL 旨在提升多模态能力**：**Qwen2.5-VL** 作为一款重要的视觉语言模型亮相，能够解释复杂的视觉数据和用户交互，并在其[博客文章](https://qwenlm.github.io/blog/qwen2.5-vl/)中进行了展示。
   - 增强功能包括理解图像中的文本和图表，并充当视觉 **Agent**，引发了关于其能力的广泛讨论。
- **DeepSeek 不懈的创新**：用户注意到 **DeepSeek** 将技术进步推向市场的惊人速度，一些人对其模型开发方法和资源获取方式表示好奇。
   - 评论强调了他们在短短两个月内取得的重大突破和成就，并重点讨论了在 AI 研究中保持领先地位的巨大压力。
- **以 Emu 为动力的 AI 是下一个前沿吗？**：围绕在训练模型中使用 **Emus** 的概念引发了讨论，并出现了一些幽默的建议，如 **Reinforcement Learning from Emu Feedback**（基于 Emu 反馈的强化学习）。
   - 社区开玩笑地讨论了开发受 emu 行为启发的算法的荒诞性及其潜力，包括 AI 增强型 emu 系统的假设场景。
- **1.58-bit 量化模型的开发**：一位用户宣布创建了 **671B DeepSeek R1 模型**的 **1.58-bit 量化版本**，引发了人们对其性能的浓厚兴趣。
   - 成员们对这一进展表示惊讶，并询问了该新量化模型在实践中的表现细节。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/Qwen/Qwen2.5-VL-72B-Instruct">Qwen/Qwen2.5-VL-72B-Instruct · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/deepseek-ai/Janus/blob/main/janus_pro_tech_report.pdf">Janus/janus_pro_tech_report.pdf at main · deepseek-ai/Janus</a>：Janus 系列：统一多模态理解与生成模型 - deepseek-ai/Janus</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibhew9/qwen_just_launced_a_new_sota_multimodal_model/">Reddit - 深入探索</a>：暂无描述
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1333530668168118335)** (24 messages🔥): 

> `DeepSeek's Janus-Pro Model, Trump's Tariffs on Chips, Qwen 2.5 Model, Mistral Acquisition Rumors, AI Data Protection in Italy` 


- **DeepSeek 发布 Janus-Pro 模型**：DeepSeek 推出了 [Janus-Pro 模型](https://huggingface.co/deepseek-ai/Janus-Pro-7B)，这是一个统一的自回归框架，通过解耦的视觉编码路径增强了多模态理解能力。
   - 该模型旨在匹配或超越特定任务模型，同时更具灵活性，使其成为下一代多模态领域的重要竞争者。
- **特朗普将对台湾芯片征收关税**：特朗普总统正准备对国外生产的计算机芯片（包括台湾制造的芯片）征收高达 **100%** 的关税，声称需要将生产恢复到美国。
   - 他表示，此举针对半导体制造业，强调了科技公司如何将生产转移到台湾的 TSMC 等公司。
- **Qwen 2.5 Max 模型开发**：最新发布的是 Qwen 2.5-Max 模型，该模型在 **超过 20 万亿 tokens** 上进行了预训练，并通过监督微调（SFT）和强化学习（RL）方法进一步优化。
   - Qwen 2.5-Max 展示了在扩展超大规模模型方面的进展，涵盖了稠密（dense）和混合专家（Mixture-of-Expert）架构。
- **Bernard Arnault 收购 Mistral 的传闻**：有投机性讨论称 Bernard Arnault 可能会收购法国 AI 公司 Mistral，理由是法国需要保持其在技术领域的竞争优势。
   - 这一进展与 ASI 之后 AI 在艺术和手工艺领域创新的重要性等类似观点相吻合。
- **意大利调查 DeepSeek 的数据保护情况**：由于对 AI 技术的日益担忧，意大利监管机构正寻求 DeepSeek 关于数据保护实践的信息。
   - 这一举措反映了全球范围内加强对 AI 系统审查和监管的更广泛趋势，以确保符合数据隐私标准。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.msn.com/en-gb/money/other/is-deepseek-about-to-cause-a-stock-market-crash/ar-AA1xV6nG">MSN</a>: 未找到描述</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-max/">Qwen2.5-Max: Exploring the Intelligence of Large-scale MoE Model</a>: QWEN CHAT API DEMO DISCORD 众所周知，持续扩展数据规模和模型规模可以显著提高模型智能。然而，研究界和工业界...</li><li><a href="https://www.pcmag.com/news/trump-to-tariff-chips-made-in-taiwan-targeting-tsmc">Trump To Tariff Chips Made In Taiwan, Targeting TSMC</a>: 如果政策实施，这些关税将波及 Apple、AMD 和 Nvidia 的尖端智能手机和 PC 相关芯片。但特朗普赌他的计划会将更多芯片生产带回美国。</li><li><a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B">deepseek-ai/Janus-Pro-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/webml-community/janus-pro-webgpu">Janus Pro WebGPU - a Hugging Face Space by webml-community</a>: 未找到描述</li><li><a href="https://x.com/techbrospod/status/1883952340023280083">Tweet from Technology Brothers (@techbrospod)</a>: 突发：Bernard Arnault 正在探索收购陷入困境的法国人工智能公司 Mistral 的可能性，理由是法国需要“保持其在手工制品领域的优势”...</li><li><a href="https://tenor.com/view/rodney-king-get-along-gif-22105666">Rodney King Get Along GIF - Rodney King Get Along - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/bogdanoff-dump-it-stocks-crypto-gif-20477588">Bogdanoff Dump It GIF - Bogdanoff Dump It Stocks - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://theyseeyourphotos.com/">They See Your Photos</a>: 未找到描述</li><li><a href="https://uk.finance.yahoo.com/news/deepseek-cause-stock-market-crash-065127717.html">Is DeepSeek about to cause a stock market crash?</a>: 随着股市被专注于 AI 的美国科技公司主导，DeepSeek 这个 OpenAI 的竞争对手是否即将让一切崩盘？文章：DeepSeek 是否即将引发股市崩盘？...
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1333527272963182592)** (103 条消息🔥🔥): 

> `Cascade 错误、Windsurf 身份验证问题、类型检查集成、DeepSeek 模型集成、额度管理担忧` 


- **Cascade 持续引发错误**：多位用户报告了 **Cascade** 的持久性错误，由于提示词（prompts）运行异常导致了严重的额度损失。*一位用户表示，在遇到大量“cascade failed”错误时，他们的付费额度已耗尽*。
- **Windsurf 登录问题持续**：一位用户在登录 **Windsurf** 时遇到问题，发现即使在重新安装软件后，身份验证码也无法正常工作。尽管尝试了新旧账号，该用户仍无法成功登录，并提交了支持工单以寻求进一步帮助。
- **类型检查的困扰及建议的工作流**：一位用户对集成类型检查表示沮丧，因为每个提示词都会循环出现错误。另一位贡献者分享了一个[工作流指南](https://github.com/marwinsteiner/swe-with-llms/blob/main/general_workflow.md)，改善了他们的类型检查体验，并获得了积极反馈。
- **DeepSeek 集成状态更新**：用户询问了在 **Codeium** 中使用 **DeepSeek r1** 模型的情况。目前已明确 **r1 尚不可用**，且在 Cascade 之外的模型在工作流中的 tool calling 方面仍存在困难。
- **对额度管理的担忧**：用户对有限的免费计划以及在遇到多次错误时管理额度的挫败感表示担忧。用户强调了他们对从慷慨的使用模式转变为限制性模式的不满，这使他们的工作流变得复杂。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/kermit-the-frog-kermit-waiting-waiting-patiently-still-waiting-gif-2626975872750814683">Kermit The Frog Waiting GIF - Kermit the frog Kermit Waiting - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/the-simpson-leech-leeches-gif-11029678">The Simpson Leech GIF - The Simpson Leech Leeches - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/Codeium/comments/1ic2i5a/windsurf_accessing_files_outside_of_workspace_by/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/marwinsteiner/swe-with-llms/blob/main/general_workflow.md">swe-with-llms/general_workflow.md at main · marwinsteiner/swe-with-llms</a>: 一个结合了 IDE 集成 LLM 和外部验证 LLM 的软件工程工作流。 - marwinsteiner/swe-with-llms
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1333526956276187137)** (189 条消息🔥🔥): 

> `Windsurf Cascade 问题、DeepSeek 模型添加、用户 Prompt 额度说明、Cascade 内部错误、Cascade Base 模型功能` 


- **Windsurf Cascade 面临挑战**：用户报告了 Windsurf 的 Cascade 存在问题，包括输入时卡顿以及编辑大文件时出现问题，这暗示了自近期更新以来性能有所下降。
   - 建议包括开启新会话以避免卡顿，以及在问题持续存在时联系支持团队。
- **在 Windsurf 中添加 DeepSeek 模型的请求**：多位用户表达了希望在 Windsurf 中添加 DeepSeek 模型的愿望，并指出其优势，如更高的编程 Benchmark 性能。
   - 尽管有这些请求，但目前尚未确认 DeepSeek 的集成。
- **用户 Prompt 额度说明**：关于 Premium Ultimate 订阅存在困惑，用户注意到 Flow Action 额度用尽会限制对高级模型的访问。
   - 建议支持团队解决订阅详情和额度更新流程中缺乏透明度的问题。
- **Cascade 的错误与内部问题**：用户遇到了许多 Cascade 的内部错误，声称 Tool Calls（工具调用）失败且无法创建文件。
   - 一些人建议检查诊断日志，并针对未解决的错误联系支持团队。
- **Cascade Base 的使用建议**：讨论强调虽然 Cascade Base 可以处理简单任务，但更复杂的操作需要高级模型以提高效率。
   - 鼓励用户探索 Cascade Base 的功能，但在需要更好功能时可能需要升级。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.codeium.com/getstarted/overview">Welcome to Codeium - Codeium Docs</a>：未找到描述</li><li><a href="https://vimeo.com/1051228720?share=copy#t=0">Custom Windsurf Editor</a>：这是 Siap Boz 在 Vimeo 上发布的 &amp;quot;Custom Windsurf Editor&amp;quot;，该网站提供高质量视频和热爱视频的人群。</li><li><a href="https://tenor.com/view/kermit-the-frog-kermit-waiting-waiting-patiently-still-waiting-gif-2626975872750814683">Kermit The Frog Waiting GIF - Kermit the frog Kermit Waiting - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>：需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://docs.codeium.com/windsurf/memories#cascade-auto-generated-memories">Cascade Memories</a>：未找到描述</li><li><a href="https://codeium.com/plan">Plan Settings</a>：未来的编辑器，就在今天。Windsurf Editor 是首款由 AI Agent 驱动的 IDE，让开发者保持心流状态。现已支持 Mac、Windows 和 Linux。</li><li><a href="https://github.com/VSCodium/vscodium/blob/master/docs/index.md#proprietary-debugging-tools">vscodium/docs/index.md at master · VSCodium/vscodium</a>：不含 MS 品牌/遥测/许可的 VS Code 二进制发行版 - VSCodium/vscodium</li><li><a href="https://chat.qwenlm.ai/">Qwen Chat</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1333568625730125847)** (2 条消息): 

> `Amazon Nova 模型、Amazon Bedrock 运行问题、模型可用性` 


- **Amazon Nova 模型出现停机**：由于上游问题，**Amazon Nova** 模型和 **Amazon Bedrock** 都遇到了运行问题，导致服务器返回了误导性的 **400** 状态码。
   - 使用量的激增被误解为 **Key 泄露**，尽管 **BYOK** 的使用未受影响。
- **Amazon Bedrock 已恢复正常运行**：更新确认 **Amazon Bedrock** 现已完全恢复运行，从之前影响模型的问题中恢复。
   - 所有 **Nova** 和 **Claude** 模型均已重新上线，恢复正常功能。


  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1333530598740066439)** (255 条消息🔥🔥): 

> `DeepSeek 供应商问题、Gemini 视频支持、模型速度对比、OpenRouter 使用情况、供应商定价背景`

- **DeepSeek 服务商持续面临困境**：DeepSeek 服务商已宕机数日，令依赖其快速响应 R1 查询的用户感到沮丧。
   - 一些用户推测宕机是由于严重的 DDOS 攻击造成的，这表明服务商在管理激增的请求方面面临挑战。
- **Gemini 视频支持实现**：一位用户分享了将视频集成到其 Gemini 模型工作流的代码片段，表明了在媒体处理方面的尝试。
   - 目前关于如何使用 OpenRouter 传递视频引用的文档有限，团队成员正在查看更新。
- **模型速度与性能对比**：用户讨论了各种模型的速度，指出 OpenRouter 的性能被认为比官方 OpenAI API 更快、更简洁。
   - 不同模型的稳定性和并发性引发了辩论，一些用户报告的体验优于其他用户。
- **OpenRouter 对服务商的灵活性**：用户询问了使用 OpenRouter 的细节，试图了解如何在 API 调用中强制使用特定的服务商。
   - 澄清了用户可以指定服务商并为其请求管理 fallback（回退）设置。
- **免费模型提供与定价见解**：一位用户质疑某些模型免费提供的合理性，引发了关于服务定价的讨论。
   - 用户强调了理解定价结构的重要性，并提到了成本效益与性能之间的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Alibaba_Qwen/status/1884263157574820053">来自 Qwen (@Alibaba_Qwen) 的推文</a>：DeepSeek V3 的爆发吸引了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM，并且...</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-max/">Qwen2.5-Max：探索大规模 MoE 模型的智能</a>：QWEN CHAT API DEMO DISCORD 众所周知，持续扩展数据规模和模型规模可以显著提高模型智能。然而，研究界和工业界...</li><li><a href="https://openrouter.ai/docs/integrations#bring-your-own-provider-api-keys">集成 | OpenRouter</a>：在 OpenRouter 中使用你自己的服务商密钥</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/providers">DeepSeek: DeepSeek R1 – 服务商状态</a>：查看服务商状态并向 DeepSeek 发起负载均衡请求：DeepSeek R1 - DeepSeek R1 已经发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并且具有完全开放的推理 token...</li><li><a href="https://operator.chatgpt.com/geo-blocked">Operator</a>：一个可以使用自带浏览器为你执行任务的 Agent。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1/uptime">DeepSeek: DeepSeek R1 – 运行时间与可用性</a>：DeepSeek: DeepSeek R1 在各服务商的运行时间统计 - DeepSeek R1 已经发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并且具有完全开放的推理 token。它的参数量为 671B...</li><li><a href="https://openrouter.ai/docs/provider-routing#load-balancing">服务商路由 | OpenRouter</a>：跨多个服务商路由请求</li><li><a href="https://openrouter.ai/docs/provider-routing#custom-routing">服务商路由 | OpenRouter</a>：跨多个服务商路由请求</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:nitro">DeepSeek R1 (nitro) - API、服务商、统计数据</a>：DeepSeek R1 已经发布：性能与 [OpenAI o1](/openai/o1) 相当，但是开源的，并且具有完全开放的推理 token。它的参数量为 671B，推理过程中有 37B 激活...</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API、服务商、统计数据</a>：DeepSeek R1 Distill Llama 70B 是一个基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B</li><li><a href="https://ai.google.dev/gemini-api/docs/troubleshooting?lang=python">未找到标题</a>：未找到描述</li><li><a href="https://mem0.ai/">Mem0 - 为你的 AI 应用提供的记忆层</a>：Mem0 是一个为 LLM 应用提供的自我改进记忆层，能够实现个性化的 AI 体验，从而节省成本并提升用户体验。</li><li><a href="https://www.insightengine.online/llm-cost/">LLM 成本 - Insight Engine - AI 价格计算器</a>：轻松比较 LLM 成本！使用这个简单的计算器估算 OpenAI GPT、Google Gemini 等 AI 模型的成本。查看每百万 token 的定价，并找到最具成本效益的解决方案...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1333550359368564756)** (79 条消息🔥🔥): 

> `GRPO 实现讨论、DeepSeek 训练成本分析、LLM 推理能力、LLM 研究领域的就业机会、神经科学与 AI 的解读` 


- **关于 GRPO 可用性的担忧**：成员们指出 **SimpleRL** 和 **TinyZero** 都没有有效地利用 **GRPO**，并声称它们的复现运行在使用它时未能取得成功的结果。
   - 这导致了一个共识，即 **GRPO** 在最近的仓库中可能被视为死代码，主要还是依赖 **PPO**。
- **DeepSeek 经济实惠的训练策略**：讨论强调，DeepSeek 仅 **500 万美元** 的训练成本是通过各种优化实现的，包括 **8-bit 训练** 和修改后的 **MoE** 设置。
   - 尽管没有引入太多新颖的想法，但成员们一致认为，在大规模生产中实施现有策略是他们领先于大型组织的关键。
- **LLM 推理能力的本质**：有观点将 **LLM** 描述为“随机鹦鹉”，引发了关于其功能背后是否存在真实推理的辩论。
   - 有人建议，通过视频游戏等实际评估，可能比传统方法更能有效衡量推理能力。
- **寻找 LLM 研究领域的远程工作机会**：一位用户寻求关于在不依赖 LinkedIn 等平台的情况下，寻找 **LLM** 领域远程兼职研究工作的建议。
   - 社区成员建议在会议上建立人脉（Networking）是发现工作机会更有效的方法。
- **神经科学对 AI 的见解**：一位独立记者分享了一个探索神经科学发现与 **LLM** 结构之间相似之处的项目，认为 AI 的框架可能反映了大脑功能的某些方面。
   - 讨论为思考 AI 发展对情感智能和伦理考量的影响开辟了途径。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sybilhyz/status/1884290930355888589">来自 Peiyi Wang (@sybilhyz) 的推文</a>: @Grad62304977 在 RL 训练期间，模型的推理模式不断演变。有时，某种特定的模式可能会突然显现，我将其定义为“顿悟时刻 (aha moment)”。对于 ...</li><li><a href="https://x.com/TheXeophon/status/1883933054366015545">来自 Xeophon (@TheXeophon) 的推文</a>: what the fuck</li><li><a href="https://x.com/SenSchumer/status/1884019385813123528?t=tBgxoNZC2q6wwW2YfBe_PA&s=19">来自 Chuck Schumer (@SenSchumer) 的推文</a>: 中国的 DeepSeek 发布被一些人称为美国 AI 的“斯普特尼克时刻 (Sputnik moment)”。这正是我在上一届国会将 AI 列为重中之重并会继续坚持的原因。我们的竞争对手是 ...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1333529119752978483)** (112 条消息🔥🔥): 

> `GRPO and Momentum Matrices, Model-Based Reinforcement Learning, Muesli Method Comparisons, YuE Music Generation Model, Privileged Bases in Transformers` 


- **理解 GRPO 的 Momentum Matrices**：讨论强调了 Momentum Matrices 在特定基底（basis）下观察时会表现出**局部空间相关性**，这受到来自单个激活值可能产生的外积的影响。
   - 一位用户指出，虽然 DCT 可以有效地压缩这些矩阵，但**分块（chunking）的实际效果可能会有所不同**，因为 2D 结构在梯度更新中可能并非固有。
- **探索 Model-Based Reinforcement Learning**：讨论了一种名为 MR.Q 的新算法，该算法旨在统一 Model-free RL 方法，同时利用 Model-based 表示来提高学习效率。
   - 这种方法尝试在保持高更新率（update ratios）的情况下稳定学习，这是 RL 应用中的一个挑战，并强调了适当超参数调优的重要性。
- **重新审视 Muesli 的贡献**：参与者将 Muesli 方法与 Model-based 方法进行了比较，认为当**扩展到深度 1** 时，它可以被视为后者。
   - 这种比较引起了关于融合 Model-based 策略的不同方面如何影响有效性和适应性的兴趣。
- **YuE 音乐生成模型介绍**：YuE 模型被宣布为领先的全曲音乐生成开源解决方案，它融合了两个不同的 LM，以在各种流派中实现全面的性能。
   - 它强调了与 **Hugging Face** 等工具的兼容性，并展示了在多样化的**歌词到歌曲（lyrical-to-song）**任务中的潜力。
- **Transformers 中 Privileged Bases 的影响**：关于 Privileged Bases 的研究表明，它们可以影响网络内激活值与梯度更新的交互方式，为**关联记忆功能（associative memory functions）**增添了一个层级。
   - 该理论提出，这可以增强 Attention 等任务的性能，在这些任务中，选择性记忆检索至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discordapp.com/channels/992359628979568762/1083107245971226685/1328526366924214395">Discord - Group Chat That’s All Fun &amp; Games</a>：Discord 是玩游戏和与朋友放松，甚至是建立全球社区的绝佳场所。定制你自己的空间来聊天、玩耍和聚会。</li><li><a href="https://www.anthropic.com/research/privileged-bases-in-the-transformer-residual-stream">Privileged Bases in the Transformer Residual Stream</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释和可控的 AI 系统。</li><li><a href="https://arxiv.org/abs/2405.19816">Growing Tiny Networks: Spotting Expressivity Bottlenecks and Fixing Them Optimally</a>：机器学习任务通常被表述为优化问题，即在特定的函数空间内寻找最优函数。在实践中，参数化的函数空间是……</li><li><a href="https://fixupx.com/abc43992899/status/1883940238864855077">Ruibin Yuan (@abc43992899) 的推文</a>：3/n：YuE 结合了两个 LM：一个在 1.6T 语义丰富的语音/音乐 Token 及文本标签上训练的 7B LM，以及一个在 2.1T 残差 Token 上训练的 1B LM。语义-声学融合编解码器处理波形↔️co...</li><li><a href="https://fixupx.com/abc43992899/status/1883940231700951284">Ruibin Yuan (@abc43992899) 的推文</a>：1/n：🚀 宣布 YuE (乐) —— 最强大的开源全曲音乐生成模型！🎵 应对歌词到歌曲任务（类似 http://Suno.ai），支持多种流派、惊艳的人声 &...</li><li><a href="https://arxiv.org/abs/2501.16142">Towards General-Purpose Model-Free Reinforcement Learning</a>：Reinforcement learning (RL) 有望成为近乎通用的问题解决框架。然而在实践中，RL 算法通常针对特定基准量身定制，依赖于精心调优的超参数……</li><li><a href="https://x.com/BlinkDL_AI/status/1884088419586040160">BlinkDL (@BlinkDL_AI) 的推文</a>：让我们干掉 Attention。RWKV-7 "Goose" 🪿 1.5B 发布：同尺寸下 SotA 级别的基础 LM，100% RNN，完全多语言（100+ 语言和代码）且诚实：没有 eval-maxxing，没有 HQ-annealing，没有 post-t...</li><li><a href="https://openreview.net/forum?id=6RtRsg8ZV1">MAD-TD: Model-Augmented Data stabilizes High Update Ratio RL</a>：构建能够通过少量样本找到良好策略的深度 Reinforcement learning (RL) Agent 已被证明极具挑战性。为了提高样本效率，最近的工作探索了更新神经……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1333595526142103668)** (2 messages): 

> `Compute and Curvature, Scaling Impacts, Inductive Bias in Parametric Models` 


- **Compute 增加降低了 Curvature**：一位成员指出，随着 **compute** 的增加，**curvature** 会降低，这表明模型对偏离 **optima** 的鲁棒性更高。
   - *这暗示了计算能力与 **loss landscape** 稳定性之间的关系。*
- **辩论 Curvature 变化背后的原因**：同一位成员表示，观察到的 **curvature** 行为是否完全归因于 **scaling** 尚不确定，并暗示可能存在其他因素。
   - *他们提出 **parametric model** 可能会固有地扭曲 **loss landscape**，从而影响这些发现。*


  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1333541169048850454)** (5 messages): 

> `scbench, zeroSCROLLS, longbench, LM Evaluation Harness, MLX methods` 


- **Scbench 受到关注**：成员们讨论认为 **scbench** 看起来很有前景，但由于其对 **multi-turn** 能力的需求，在集成方面可能面临挑战。
   - 一位成员指出，他们将更多地研究 **zeroSCROLLS** 作为潜在的替代方案。
- **引入 Longbench**：一位成员宣布在他们的项目中增加了 **longbench**，表明开发工作正在进行中。
   - 这一增加可能会补充目前正在进行的 **scbench** 工作，以实现更好的评估。
- **LM Evaluation Harness 的挑战**：一位成员表达了在本地使用 [LM Evaluation Harness](https://github.com/chimezie/lm-evaluation-harness-mlx) 时的挫败感，理由是 `generate_until` 等方法尚未实现。
   - 由于持续遇到问题，他们正在寻求关于寻找更好方法来配合 **MLX** 使用 **LM Eval Harness** 的建议。


  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1333781208919969903)** (1 messages): 

> `Janus flow paper, Rectified flow objective, Image generation tasks` 


- **关于 Janus Flow 目的的澄清**：一位成员询问 Janus flow 论文中概述的 **rectified flow objective** 是否意味着它不能用于将图像转换为另一张图像，因为 **x^con** 仅由文本 token 组成。
   - *这引发了关于该模型在直接 **image-to-image** 转换中适用性的疑问。*
- **图像生成任务的探索**：讨论强调了文本 token 在图像生成任务中的重要性，以及它们在 **Janus flow** 方法论背景下的功能。
   - 参与者热衷于了解这如何影响 **cross-modal generation** 和实际应用。


  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1333527120281997465)** (59 messages🔥🔥): 

> `DeepSeek V3 and competition, OpenAI's new offerings, Qwen licensing and development, ChatGPT Gov announcement, Open Thoughts project and partnerships`

- **DeepSeek V3 备受期待**：据报道，DeepSeek 即将发布 **V3**，根据社区讨论的说法，该模型预计将超越美国实验室的模型。一位用户指出，DeepSeek 的进步在 AI 社区引起了巨大反响。
   - 成员们表示，DeepSeek 的模型更新可能会对大型 **MoE 模型**产生重大影响，引发了围绕竞争结果和基准测试（benchmarks）的讨论。
- **OpenAI 推出 ChatGPT Gov**：OpenAI 宣布推出 ChatGPT Gov，这是专为美国政府机构量身定制的 ChatGPT 版本，为他们提供增强的尖端模型访问权限。有人担心 OpenAI 对政府合同的关注是否会导致类似 Skydio 转型策略的重演。
   - 用户认为，如果 OpenAI 继续走这条路，可能会影响其整体产品竞争力，类似于该领域的其他公司。
- **Qwen 的新模型开发**：Qwen 发布了 **Qwen2.5-Max**，这是一款大型 MoE LLM，声称具有与其他领先模型竞争的性能。社区对话表明，在构建该模型能力方面采用了协作方法，展示了训练方法上的实质性进步。
   - 参与者对 Qwen 在不断发展的 AI 格局中的战略方向提出了质疑，特别是关于其看似分散的许可（licensing）行动。
- **Open Thoughts 项目启动**：Open Thoughts 项目介绍了他们最新的推理数据集，旨在实现开源数据共享和协作的综合方法。该版本的发布因其社区支持和领先机构的参与而受到认可。
   - 该倡议强调了在发布开源数据集方面的集体努力，推动了围绕模型开发和数据生成的协作讨论。
- **Codename Goose 发布**：Codename Goose 被宣布为一款开源 AI Agent，旨在通过用户友好的界面和 CLI 自动化任务。该倡议强调开源可用性，并致力于提高各种应用的生产力。
   - 社区成员推测其可能与 Eleuther 等知名团体有关联，强调了其开源根基的重要性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://block.github.io/goose/blog/2025/01/28/introducing-codename-goose">Introducing codename goose</a>: codename goose 是你的开源 AI Agent，可自动执行工程任务并提高生产力。</li><li><a href="https://x.com/ryanmart3n/status/1884284108127494343?s=46">Tweet from Ryan Marten (@ryanmart3n)</a>: 我们完全开源。我们的模型权重、数据集、数据生成代码、评估代码和训练代码均已公开。要查找上述所有内容，请从这里开始：https://github.com/ope...</li><li><a href="https://x.com/alibaba_qwen/status/1884263162096279907?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: 基础语言模型的结果。我们对基础模型的质量充满信心，并期望下一版本的 Qwen 在改进后训练（post-training）方法后会表现得更好。</li><li><a href="https://www.c-span.org/program/white-house-event/president-trump-addresses-house-gop-issues-conference-in-florida/655005">President Trump Addresses House GOP Issues Conference in Florida | Video | C-SPAN.org</a>: 未找到描述</li><li><a href="https://x.com/ryanmart3n/status/1884284101265612856?s=46">Tweet from Ryan Marten (@ryanmart3n)</a>: 宣布 Open Thoughts 项目。我们正在公开构建最佳的推理数据集。基于我们在 Stratos 的工作，今天我们发布了 OpenThoughts-114k 和 OpenThinker-7B。</li><li><a href="https://x.com/alibaba_qwen/status/1884263157574820053?s=46">Tweet from Qwen (@Alibaba_Qwen)</a>: DeepSeek V3 的爆发引起了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM，并且 ...</li><li><a href="https://x.com/stalkermustang/status/1884328433297342923?s=61">Tweet from Igor Kotenkov (@stalkermustang)</a>: 2.0-pro 已在 AiStudio 中可用，甚至响应了我的请求。@OfficialLoganK 随时可能通过 Twitter 发布/软启动？正在等待指标</li><li><a href="https://x.com/angelusm0rt1s/status/1884077573258567777?s=46">Tweet from Zephyr (@angelusm0rt1s)</a>: 🎵Drop Drop Drop...</li><li><a href="https://x.com/markchen90/status/1884303237186216272">Tweet from Mark Chen (@markchen90)</a>: 祝贺 DeepSeek 制作出 o1 级别的推理模型！他们的研究论文表明，他们在通往 o1 的道路上独立发现了一些与我们相同的核心思想。</li><li><a href="https://github.com/ZihanWang314/ragen">GitHub - ZihanWang314/RAGEN: RAGEN is the first open-source reproduction of DeepSeek-R1 for training agentic models via reinforcement learning.</a>: RAGEN 是 DeepSeek-R1 的第一个开源复现，用于通过强化学习训练 Agentic 模型。 - ZihanWang314/RAGEN</li><li><a href="https://www.youtube.com/watch?v=Qb-qz8KZcXQ">Trump calls China’s AI DeepSeek breakthrough ‘a wakeup call’ — but ‘positive’ if true</a>: 特朗普总统周一称中国新的 AI 平台 DeepSeek 是美国的“警钟”，同时也表示其首次亮相可能是一个“积极的”发展...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1333641735926185994)** (5 messages): 

> `DeepSeek optimizations, CUDA limitations, Reinforcement learning, DeepSeek implications, Technical report findings` 


- **DeepSeek 优化无法用 CUDA 编写？**: 一位成员表示看到有说法称某些 **DeepSeek** 优化无法用 **CUDA** 编写，并寻求相关来源。
   - 另一位成员评论说，他们听说代码是用比 CUDA 更底层的语言编写的，表明这是一个令人感兴趣的领域。
- **在技术报告中找到来源**: 一位成员提到，关于 DeepSeek 优化的陈述来源在**技术报告**中，并分享了一张随附图片。
   - 这表明有记录在案的证据支持正在讨论的说法。
- **DeepSeek 更新及其影响**: 一位成员引用了一个讨论 **DeepSeek** 影响的链接，重点包括通过纯强化学习出现的涌现式思维链（chain-of-thought）。
   - 他们指出，围绕成本和芯片禁令的讨论也与**美国和中国**之间更广泛的 AI 元讨论（meta-discussion）相关联。
- **监督责任**: 一位成员对忘记报道 DeepSeek 更新表示负责，并确认了他们之前对此事的看法。
   - 他们承认忽略了这一消息对 AI 领域更广泛的影响。



**Link mentioned**: <a href="https://stratechery.com/2025/deepseek-faq/">DeepSeek FAQ</a>: DeepSeek 彻底颠覆了人们对 AI 以及与中国竞争的预期。它是什么，为什么重要？

  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1333538606639218800)** (18 条消息🔥): 

> `Liang Wenfeng 的 Meme 游戏、DeepSeek 与 Qwen 团队动态、Meme 币发行推测、Qwen2.5-Max 模型发布、AI 社区讨论` 


- **Liang Wenfeng 的 Meme 功底深厚**：Meme 爱好者们称赞了 **Liang Wenfeng** 的创意，并附上了他[最新 Meme 帖子的链接](https://x.com/LiangWenfeng_/status/1883978669900824681)。显然，Meme 游戏正在蓬勃发展！
   - *Meme 正在占据话语权*，在严肃的讨论中带来了些许欢笑。
- **Qwen 对 DeepSeek 表示不满**：**Qwen 团队**因其对 **DeepSeek** 的批判立场而受到关注，认为在使用旧模型进行比较时是不公平的。成员提到了关于 *DeepSeek 新的 Janus Pro 论文*及其反响的持续讨论。
   - 社区被提醒在应对这种竞争格局时要保持合理的预期。
- **关于 Meme 币发行的推测**：一位用户暗示即将有 **Meme 币发行**，并表示可能很快就会上市。这一推测基于对某些相关方合法性的非正式确认。
   - 社区的反应从怀疑到对 Meme 币领域潜在新项目的兴奋不等。
- **Qwen2.5-Max 模型发布**：**Qwen 团队**宣布发布 **Qwen2.5-Max**，这是一个大型 MoE LLM，在基准测试中与 **DeepSeek V3** 展开竞争。他们强调了在各项指标上的显著性能提升。
   - 用于详细探索的支持链接包括[博客文章](https://qwenlm.github.io/blog/qwen2.5-max/)和 [API 文档](https://www.alibabacloud.com/help/en/model-studio/getting-started/first-api-call-to-qwen?spm=a2c63.p38356.help-menu-2400256.d_0_1_0.1f6574a72ddbKE)。
- **对 Joshua 的正面认可**：在各种讨论中，**Joshua** 的人品受到了称赞，成员们断言他确实是一个好人。这些评论强调了尽管存在潜在的紧张局势，社区氛围依然积极向上。
   - 成员们指出，积极的人际互动对于应对群组中激烈的议题和讨论至关重要。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bsky.app/profile/mathiasgehrig.bsky.social/post/3lgqwb3rwtk2k">Mathias Gehrig (@mathiasgehrig.bsky.social)</a>: 最后那句话肯定是我今天读到的最错误的东西。我明白你为自己是美国人或其他身份感到自豪，但即便如此。</li><li><a href="https://x.com/jachiam0/status/1884122383319179633?s=46">Joshua Achiam (@jachiam0) 的推文</a>: 意识到我今天在关于 DeepSeek 的讨论中有些话没说，这本该是我首先想到的，我很遗憾花了这么长时间才说出来。祝贺 DeepSeek 团队。...</li><li><a href="https://x.com/Alibaba_Qwen/status/1884263157574820053)">Qwen (@Alibaba_Qwen) 的推文</a>: DeepSeek V3 的爆发吸引了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM，并且...</li><li><a href="https://x.com/zizhpan/status/1884081184420012046?s=46">Zizheng Pan (@zizhpan) 的推文</a>: 伙计们，再说一次，这不是我们的 Wenfeng。</li><li><a href="https://x.com/justinlin610/status/1884020845527589190?s=46">Junyang Lin (@JustinLin610) 的推文</a>: 它甚至不是在与 Qwen2-VL-7B 比较结果，而是与两年半前发布的 Qwen-VL-Chat 比较。🤣引用 Lucas Beyer (bl16) (@giffmana)：刚刚快速浏览了 DeepSeek 的新...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1333607329404293140)** (36 条消息🔥): 

> `DeepSeek R1 发布、Qwen 2.5-Max 性能、AI 客户应用、笔记本电脑购买决策、网红对学术人物的影响`

- **DeepSeek R1 发布与采用**：AI 客户正在迅速采用 DeepSeek R1，**ZoomInfo** 等公司已将其用于产品，**Notion** 也在探索其潜力。
   - 此外，**Google Cloud** 计划将其提供给云客户，显示出强烈的市场兴趣。
- **Qwen 2.5-Max 的出色表现**：**Qwen 2.5-Max** 的发布展示了极具竞争力的性能，在 Arena Hard 和 LiveCodeBench 等基准测试中超越了 **DeepSeek V3**。
   - 用户可以根据为 **ai-gradio** 提供的安装说明立即开始使用。
- **笔记本电脑购买困境**：一场讨论提出了这样一个问题：为了 **Bayesian stats**（贝叶斯统计）的计算需求，是否有必要支付两倍的价格购买 **基础版 M4 Pro MacBook Pro**，而不是配备 24GB RAM 的 **M3 Air**。
   - 共识倾向于认为 M3 Air 是性价比更高的选择。
- **网红化对学术人物的影响**：有情绪表达了对学术人物 **AK** 从学术内容转向网红式推广的失望，引发了对其公信力的质疑。
   - 评论指出，这种转变影响了追随者对其贡献的看法。
- **Groq 与全量 R1 的可行性**：一些成员评论了在 Groq 的 LPU 上运行 **全量 R1** 的影响，考虑到其有限的 **SRAM capacity**，对其可行性表示怀疑。
   - 讨论强调了在具有此类限制的硬件上执行复杂 AI 模型的潜在挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/segyges/status/1884100565539643634">来自 SE Gyges (@segyges) 的推文</a>：我确实知道这一点：GRPO 将路径上采样到正确答案，但这些路径必须已经存在。因此，数据集中通往正确答案的路径越多，训练效果就越好。如果...</li><li><a href="https://x.com/_akhaliq/status/1884278071093502253">来自 AK (@_akhaliq) 的推文</a>：Qwen2.5-Max 刚刚一键生成了这个提示词：编写一个球体内三个黄色小球弹跳的脚本，确保正确处理碰撞检测。让球体缓慢旋转。确保小球...</li><li><a href="https://fxtwitter.com/francis_yao_/status/1884138762852262349?s=46">来自 Yao Fu (@Francis_YAO_) 的推文</a>：从 R1 和 K1.5 技术报告中得到的一个有趣的启示是使用基于字符串匹配的二元奖励：我在 2022 年用 FlanT5 尝试过，我的朋友在 2023 年用 Llama 1 尝试过...</li><li><a href="https://x.com/sundeep/status/1883779972827164804?s=46">来自 sunny madra (@sundeep) 的推文</a>：@NickADobos Distill 今天发布，全量 R1 即将到来</li><li><a href="https://x.com/sybilhyz/status/1884290930355888589?s=46">来自 Peiyi Wang (@sybilhyz) 的推文</a>：@Grad62304977 在 RL 训练期间，模型的推理模式不断演变。有时，某种特定的模式可能会突然显现，我将其定义为“顿悟时刻 (aha moment)”。对于...</li><li><a href="https://x.com/Dorialexander/status/1884167945280278857">来自 Alexander Doria (@Dorialexander) 的推文</a>：我觉得这应该是一个更大的新闻：DeepSeek 在 Nvidia H800 上进行了训练，但正在华为制造的新国产芯片 910C 上运行推理。</li><li><a href="https://x.com/btibor91/status/1884007987884536112?s=46">来自 Tibor Blaho (@btibor91) 的推文</a>：新的 ChatGPT Web 应用版本增加了新的计划和定价变化/实验 - 促销代码（固定金额或百分比折扣，针对特定周期（例如月份））（仅用于插图的测试促销代码预览...</li><li><a href="https://x.com/amir/status/1884315910737145934">来自 Amir Efrati (@amir) 的推文</a>：AI 客户行动非常~快~ • ZoomInfo 将 DeepSeek R1 用于产品 • Notion 也在考虑使用它 • Google Cloud 将其提供给云客户</li><li><a href="https://x.com/sama/status/1884066337103962416?s=46">来自 Sam Altman (@sama) 的推文</a>：DeepSeek 的 R1 是一个令人印象深刻的模型，特别是在他们能够提供的性价比方面。我们显然会提供更好的模型，而且拥有一个...确实令人振奋。</li><li><a href="https://youtube.com/shorts/_gc3isWHdfg?si=7EdFMMiYBsQVepnd">070🔥美国新闻🎥英语学习🌀AI2 的新发布旨在缩小开放与封闭 AI 模型之间的差距 #AI #NewsSummary #podcast #facts #usanewstoday</a>：070 AI2 的新发布旨在缩小开放与封闭 AI 模型之间的差距。发布日期：2024 年 12 月 3 日。The Allen I...</li><li><a href="https://e.huawei.com/en/products/computing/ascend">昇腾计算 (Ascend Computing) - 华为企业</a>：华为 Atlas AI 计算解决方案提供广泛的产品组合，实现全场景的云-边-端 AI 基础设施。了解更多。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1333557861305024553)** (10 条消息🔥): 

> `Deepseek, Google 发布延迟, AI 就业市场, ChatGPT, AI 误解` 


- **Deepseek 对 Google 的影响**：有消息称，由于最近的 **Deepseek 骚动**，**Google** 推迟了一次重大发布。这凸显了围绕 Deepseek 的社区讨论所产生的重大影响。
   - *讽刺的是，我们从一家对冲基金获得了免费的 AI，而从一家非营利组织获得的 AI 却要每月 200 美元*，这暗示了对 AI 资金来源的复杂情绪。
- **Deepseek 在播客中的缺席**：评论员对 **Deepseek** 如何在没有参加任何播客节目的情况下获得 **R1** 排名表示困惑。一位用户幽默地质疑了他们曝光努力的可信度。
   - 该言论强调了对 **Deepseek** 在 AI 领域非常规崛起的审视。
- **AI 就业市场的转变**：有人分享了一个幽默的观点，指出甚至 **ChatGPT** 也因为 AI 而失业了，暗示了快速演变的就业市场。这反映了 AI 取代传统人类角色的日益增长的趋势。
   - 这个梗捕捉到了科技行业从业者所面临的既滑稽又令人担忧的现实。
- **对 AI 使用的批评**：有人对**美国总统**误信“AI 废料（AI slop）”表示担忧，嘲讽了围绕 AI 的误导性叙事。相关的梗和推文持续流传，展示了对 AI 进展的怀疑态度。
   - 这一讨论批评了影响力人物对 AI 能力普遍存在的误解。
- **AI 文化笑话**：一张图片附件引发了笑声，强化了在线讨论中梗（memes）与 AI 文化的关联。在 AI 爱好者应对技术细微差别时，幽默是其中的一条主线。
   - 社区参与表明，在不断演变的 AI 技术的炒作和复杂性中，幽默仍然是一种应对机制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/FacebookAIslop/status/1884302439681282425">来自 Insane Facebook AI slop (@FacebookAIslop) 的推文</a>：美国总统误信了 AI 废料</li><li><a href="https://www.google.com/search?num=10&sca_esv=4f96d473e3b1885e&sxsrf=AHTn8zoBVdoNRRkrAMaQRsB9NjDsAqtmEg:1738023049305&q=liang+wenfeng&udm=2&fbs=ABzOT_CWdhQLP1FcmU5B0fn3xuWpA-dk4wpBWOGsoR7DG5zJBnsX62dbVmWR6QCQ5QEtPRqzJX8xQMQPcXbmItqG39gKAK2veO__L5-XQ1MCjZJzXa06nfdf-dR2OjmhaHqxXyL1vpwxVgtUnWcG_EXWI4_VU3SiYs86s0v083c18wiwSi6p9qYHQhChpNEjrIAtp1Snx2H3dIAyJj9vuSslSiFRHSZb_w&sa=X&ved=2ahUKEwiI_duzkJeLAxXPEjQIHe7GGv8QtKgLegQIFxAB&biw=2160&bih=1083&dpr=1.33))">liang wenfeng - Google 搜索</a>：未找到描述</li><li><a href="https://x.com/andrew_n_carr/status/1884110922823917733">来自 Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：甚至 ChatGPT 也因为 AI 失业了</li><li><a href="https://x.com/avichal/status/1883963771431071967?s=46">来自 Avichal - Electric ϟ Capital (@avichal) 的推文</a>：讽刺的是，我们从一家对冲基金获得了免费的 AI，而从一家非营利组织获得的 AI 却要每月 200 美元。</li><li><a href="https://x.com/luke_metro/status/1883708614206226449">来自 Luke Metro - e/🐋 (@luke_metro) 的推文</a>：我真的不明白 Deepseek 是如何在没有参加过一次播客的情况下做出 R1 的。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1333544622772715570)** (5 条消息): 

> `Open-Instruct 集成, vLLM 维护, OpenRLHF 框架` 


- **关于 Open-Instruct 的 vLLM 集成咨询**：一位成员正在探索在内部使用 **Open-Instruct**，并对与 **vLLM** 的集成表示担忧，特别是对 **OpenRLHF** 框架的依赖。
   - *他们表示希望避免依赖一个可能缺乏长期维护的 OSS 项目*。
- **AllenAI 的工具维护方法**：**Natolambert** 提到，他们正在与 **vLLM** 讨论用于 RL 的工具使用情况，这表明了一种积极主动的方法。
   - *他们指出，虽然他们通常会固定开源工具的版本直到必须更新，但他们无法保证对任何特定工具的持续维护。*
- **Open-Instruct 可行性的不确定性**：尽管认可 **Open-Instruct** 的潜力，**Natolambert** 警告说集成并不能保证，称其为“并非板上钉钉的事”。
   - *这突显了在没有未来支持保证的情况下，对依赖新工具所持的谨慎态度。*


  

---

### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1333782402610364498)** (5 条消息): 

> `Qwen 许可问题，Qwen 模型变体` 


- **Qwen 许可引发混乱**：一名成员提出了 *'为什么 Qwen 的许可如此混乱？'* 的问题，强调了不同 Qwen 模型变体之间许可协议的不一致性。
   - 他们指出 **Qwen2.5-VL-72b** 对超过 **100M MAU**（月活跃用户）的服务有限制，而 **Qwen2.5-VL-7b** 则使用了更宽松的 **Apache 2.0 license**。
- **Qwen2.5-VL-3b 的新非商业许可**：**Qwen2.5-VL-3b** 版本引入了一个名为 **'Qwen Research'** 的新许可，仅限非商业用途，这引发了对其目标受众的质疑。
   - 有人指出这 *'非常奇怪且具体'*，暗示该许可针对的是 **100M MAU** 以下的一小部分特定企业。
- **社区对许可问题的关注**：一名成员报告称，他们联系了 Hugging Face 的代表咨询 Qwen 的许可问题，对方表示目前正在调查中。
   - 社区对可能转向限制性的 **'research' 许可** 表示担忧，一名成员表示希望这种情况不要发生。
- **Qwen 模型缺乏许可清晰度**：讨论显示 **7B 模型** 缺少 **LICENSE 文件**，导致其条款与其它 Qwen 模型相比存在歧义。
   - 围绕 7B 模型标签的清晰度缺失，进一步加剧了对其法律用途的困惑。


  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1333568933365547083)** (13 条消息🔥): 

> `DeepSeek-R1 发布，AI 模型对比，夏威夷军事人口，Jay Alammar 的插图，理解复杂模型的挑战` 


- **DeepSeek-R1 发布并开源权重**：Jay Alammar 宣布了 [DeepSeek-R1](https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1) 的发布，强调了其开源权重以及类似于 OpenAI O1 的训练方法。
   - 此次发布被认为对 ML R&D 社区具有重大影响，因为它提供了蒸馏版本和建设性的见解。
- **AI 模型的效率与可访问性**：成员们讨论认为，与竞争对手相比，**DeepSeek** 凭借其开源方法、更优的价格和易用性展现出了优势。
   - 评论暗示这可能会迫使 **Anthropic** 发布其待定项目，以跟上创新的步伐。
- **关于极客社区的幽默看法**：在讨论中，一名成员谈到了 **夏威夷** 技术爱好者的集中程度，认为由于其庞大的军事人口，这是一种独特的现象。
   - 另一名成员开玩笑地提到地铁活动，调侃“极客”聚会的可预测性。
- **Jay Alammar 持续的插图创作成功**：社区认可了 Jay 的创造力，一名成员幽默地评论道 *这个人停不下画图的脚步。*
   - 然而，也有人对他最近的作品表达了复杂的感受，认为可能无法引起所有受众的共鸣。
- **AI 图表与数学描述的复杂性对比**：一名成员表示，发现某些模型的图表表示比数学解释更难理解。
   - 这凸显了社区在传达复杂 AI 概念的清晰度方面面临的持续挑战。



**提到的链接**：<a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1">图解 DeepSeek-R1</a>：推理型 LLM 的秘诀

  

---

### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1333559087375585331)** (40 messages🔥): 

> `DeepSeek 的主流关注、OpenAI 的形式数学方向、LLM 作为数学验证者、Nat Lambert 即将发布的文章` 


- **DeepSeek 获得主流关注**：围绕 DeepSeek 的讨论非常热烈，甚至在与用户妻子相关的金融公司中也被提及，这表明其知名度正在超出小众圈子。
   - 一位成员指出：*'我妻子昨晚听说了 DeepSeek，这意味着它真的触及了普通大众。'*
- **OpenAI 暂停形式数学方向**：有人对 OpenAI 缺乏对形式数学（formal math）的关注表示担忧，根据与团队成员的讨论，他们的这一方向似乎已经暂停。
   - 一位用户提到：*'他们前段时间暂停了形式数学方向，'* 引用了社区过去对话中的见解。
- **关于 LLM 作为验证者的讨论**：关于 LLM 是否可以作为数学问题的有效验证者展开了辩论，这可能消除对传统定理证明器（theorem provers）的需求。
   - 一位用户表示：*'为了取得进展，并不显然一定需要定理证明器，'* 暗示了 LLM 在该领域的效用。
- **Nat Lambert 即将发布的关于推理模型的文章**：Nat Lambert 正考虑发布他关于推理模型（reasoning models）的见解，强调 DeepSeek R1 仅代表重大进展的开始。
   - 他注意到了目前围绕这些话题的热度，指出：*'我们目前正处于完全的炒作浪潮中。'*



**提到的链接**：<a href="https://x.com/natolambert/status/1884346850645348647">Nathan Lambert (@natolambert) 的推文</a>：为什么推理模型会泛化。DeepSeek R1 只是飞速进展的冰山一角。人们低估了“推理”的长期潜力。https://buff.ly/4haoAtt

  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1333842529258569812)** (1 messages): 

> `终端错误检测、Bolt 集成` 


- **Bolt 现在可检测终端错误**：Bolt 推出了 **Terminal Errors Detection**（终端错误检测），用于捕捉终端中那些难以发现且经常被忽视的问题。
   - *
- **增强与开发环境的集成**：新功能将 Bolt 与应用的开发环境紧密集成，自动检测问题并收集修复所需的关键数据。
   - [在此了解更多](https://x.com/boltdotnew/status/1884283123690819721)。



**提到的链接**：<a href="https://x.com/boltdotnew/status/1884283123690819721">bolt.new (@boltdotnew) 的推文</a>：Bolt 🧠 更新：Terminal Errors Detection。有些错误很难捕捉：它们发生在终端，而我们并不经常查看。Bolt 现在与您的应用开发环境紧密集成...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1333765406443311194)** (4 messages): 

> `Improver Prompt 的局限性、前端原型约束、Prompt Improver 的用户体验` 


- **对 Improver Prompt 的不满**：一位用户对 **improver prompt** 表示不满，称它在构建初期添加了不必要的细节，使 Prompt 变得杂乱。
   - 他们提到它经常拒绝开发自己的想法，导致令人沮丧。
- **承认文档管理系统的约束**：一位用户强调了为**文档管理系统**创建前端原型的关键局限性，特别是在浏览器能力和缺乏后端支持方面。
   - 因此，他们建议专注于使用模拟数据进行 UI/UX 设计，同时需要连接后端服务以实现完整功能。
- **使用 Prompt Improver 的潜在转变**：一位用户表示他们可能会完全停止使用 **prompt improver**，或者觉得不得不删除它添加的一半内容。
   - 这种情绪反映了对该工具当前行为和输出日益增长的挫败感。

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1333535154383228938)** (135 条消息🔥🔥): 

> `Stripe 集成挑战、使用 Bolt 的工作流与项目管理、利用 AI 从图像生成标题、Bolt 中的 Node 版本更新、社区支持与协作` 


- **解决 Stripe 集成问题**：社区讨论了 Stripe 集成的各种挑战，特别集中在创建订阅系统和有效管理用户角色方面。
   - 专家们提出协助进行实际实施，同时鼓励本着协作精神分享知识。
- **使用 Bolt 的高效工作流**：用户分享了利用 Bolt 进行项目开发的工作流和经验，强调了结构和方法论对成功的重要性。
   - 许多人指出，只要方法得当并有效利用工具的功能，就有可能在不编写代码的情况下完成整个项目。
- **AI 从图像生成标题**：人们对使用 ChatGPT 等 AI 从图像生成合适标题表现出兴趣，并提出了关于最佳实现方法的疑问。
   - 讨论中区分了使用 OCR 读取文本与为图像生成文本的不同，表明需要明确预期的输出结果。
- **在 Bolt 中更新 Node 版本**：一位用户寻求关于如何更新 Bolt 所使用的 Node 版本的指导，在过程中遇到了权限限制。
   - 这反映了开发者对管理应用程序中依赖项和版本控制的普遍关注。
- **社区支持与协作**：多位成员互相提供帮助，分享与他们的 Bolt 项目和集成相关的资源和经验。
   - 社区展现了协作精神，成员们主动提出帮助解决常见问题并改进项目功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://cozy-cucurucho-6575ff.netlify.app/">Vite + React + TS</a>: 未找到描述</li><li><a href="https://diji.art">Diji.art - Digital Design Marketplace</a>: 在高品质服装上创建并销售独特设计。加入我们的创作者和时尚爱好者社区。</li><li><a href="https://diji.art/designs">Diji.art - Digital Design Marketplace</a>: 在高品质服装上创建并销售独特设计。加入我们的创作者和时尚爱好者社区。</li><li><a href="https://21st.dev/">21st.dev - The NPM for Design Engineers</a>: 利用受 shadcn/ui 启发的即插即用 React Tailwind 组件，更快速地交付精美的 UI。由设计工程师为设计工程师打造。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1333572593214558209)** (138 条消息🔥🔥): 

> `Janus 模型评价, Stable Diffusion 的 AMD 支持, AI 工作的硬件推荐, Stable Diffusion 中的 Upscalers, Deepseek 模型对比` 


- **Janus 模型评价褒贬不一**：用户对 Janus 的**图像生成能力**表示失望，有人指出其 **7B 模型运行缓慢**，并对其主要用途提出质疑。
   - 另一位用户提到，**大多数基础模型**感觉都不尽如人意，建议在等待 Janus 改进期间坚持使用 **SDXL**。
- **在 AMD 上设置 Stable Diffusion**：对于使用 AMD 显卡的用户，建议查看**技术支持频道 (tech support channel)** 的置顶指南，以确保在使用 Stable Diffusion 时获得最佳性能。
   - 建议倾向于利用 **Swarm UI** 或 **webui-forge** 来增强 AMD 系统上的功能。
- **AI 配置中的 RAM 与 VRAM 之争**：关于 AI 任务中**高 RAM** 与 **VRAM** 必要性的辩论兴起，一些人认为过剩的 RAM 往往被浪费，而另一些人则强调其在不同应用中的价值。
   - 观点各异，有建议称对于专注于几项高级任务的使用场景，投资 **32GB 的 VRAM** 会更经济。
- **Upscalers 使用的一致性**：讨论涉及了放大器 (Upscalers) 的生命周期，用户指出许多放大器已经使用了**两年**而没有频繁更新。
   - 一位用户分享了他们首选的放大器，特别是 **4x-AnimeSharp** 和 **4x_NMKD-superscale**，并指出缺乏新的突出选项。
- **Deepseek 与其他 LLM 的对比**：关于 Deepseek 是否名副其实（尤其是在其**无审查 LLM 能力**方面）引发了关注，一些人强调了它与 **OpenAI 产品**的对比。
   - 用户指出，虽然 Deepseek 很有前景，但与现有模型相比，它尚未展示出革命性的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/deepseek-ai/Janus-Pro-7B/tree/main">deepseek-ai/Janus-Pro-7B at main</a>：未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1333528702264410172)** (112 条消息🔥🔥): 

> `Goose 客户端, MCP 服务器问题, DeepSeek 定价, 与 Home Assistant 的集成, Token 使用监控` 


- **Goose 客户端备受关注**：用户对新的 Goose 客户端表示了积极的体验，特别是其本地执行和可扩展性功能。
   - 然而，它目前仅支持 Mac 和 Linux，引发了关于其与 WSL 兼容性的讨论。
- **注意到 MCP 服务器的局限性**：成员们对社区 MCP 服务器的功能表示担忧，质疑其可靠性以及在测试过程中遇到的具体问题。
   - 建议创建一个经过验证的 MCP 服务器列表，以方便用户识别可靠的选项。
- **DeepSeek 极具吸引力的优惠**：讨论中提到了注册 kluster.ai 以获取 DeepSeek 的 100 美元额度的价值，强调了其相对合理的定价。
   - 虽然承认推理速度与早期部署相比不够理想，但仍引起了用户的兴趣。
- **Home Assistant MCP 集成**：成员们讨论了使用 Home Assistant 的 MCP 作为媒体管理桥梁的潜力，并注意到最近已将 MCP 支持合并到核心库中。
   - 然而，在最新更新之后，其是否已准备好广泛使用仍存在不确定性。
- **Token 使用和管理担忧**：用户对 Goose 的内部 Token 使用情况表示担忧，强调了有效跟踪消耗的重要性。
   - 建议包括为 Token 使用情况实现日志输出，并通过提供商仪表板进行监控。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://block.github.io/goose">codename goose | codename goose</a>: 你的开源 AI Agent，无缝自动化工程任务。</li><li><a href="https://github.com/Upsonic/Upsonic">GitHub - Upsonic/Upsonic: Task oriented AI agent framework for digital workers and vertical AI agents</a>: 面向数字员工和垂直 AI Agent 的任务导向型 AI Agent 框架 - Upsonic/Upsonic</li><li><a href="https://github.com/modelcontextprotocol/servers/blob/main/src/everything/sse.ts">servers/src/everything/sse.ts at main · modelcontextprotocol/servers</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账户为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/jmagar/yarr">GitHub - jmagar/yarr: model context protocol ARR server</a>: Model Context Protocol ARR 服务器。通过在 GitHub 上创建账户为 jmagar/yarr 的开发做出贡献。</li><li><a href="https://github.com/cookiecad/mcp-runner">GitHub - cookiecad/mcp-runner: A TypeScript SDK for running MCP (Model Context Protocol) servers with process reuse capabilities</a>: 一个用于运行具有进程复用能力的 MCP (Model Context Protocol) 服务器的 TypeScript SDK - cookiecad/mcp-runner</li><li><a href="https://github.com/modelcontextprotocol/typescript-sdk/tree/main?tab=readme-ov-file#http-with-sse">GitHub - modelcontextprotocol/typescript-sdk: The official Typescript SDK for Model Context Protocol servers and clients</a>: Model Context Protocol 服务器和客户端的官方 Typescript SDK - modelcontextprotocol/typescript-sdk</li><li><a href="https://github.com/block/goose/blob/main/crates/goose-mcp/src/computercontroller/mod.rs">goose/crates/goose-mcp/src/computercontroller/mod.rs at main · block/goose</a>: 一个开源、可扩展的 AI Agent，超越了代码建议——可以使用任何 LLM 进行安装、执行、编辑和测试 - block/goose
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1333531745227636849)** (96 条消息🔥🔥): 

> `Qwen 2.5-Max 发布, DeepSeek R1 vs. Qwen 2.5, 开源 AI 进展, 台积电关税对 AI 的影响, 华为芯片在 AI 应用中的表现`

- **Qwen 2.5-Max 发布公告**：新模型 Qwen 2.5-Max 已发布，据报道在包括 Arena Hard 和 LiveBench 在内的多个基准测试中表现优于 DeepSeek V3。
   - 开发者可以通过阿里云的 API 访问该模型，并通过 Qwen Chat 平台进行测试。
- **DeepSeek R1 与 Qwen 2.5 的讨论**：社区成员讨论了推理模型之间的差异，指出虽然 DeepSeek R1 展示了清晰的思维链（Chain of Thought），但像 Flash Thinking 这样的其他模型可能并非如此。
   - R1 中结构化推理 Token 的引入展示了与其他模型的显著差异，引发了关于 SFT 对连贯性影响的疑问。
- **开源 AI 进展**：诸如 Open Thoughts 项目之类的新倡议旨在创建高质量的推理数据集，以增强 AI 社区内的模型性能。
   - 此外，开源音乐生成模型 YuE 也已发布，展示了可获取 AI 资源日益增长的趋势。
- **TSMC 关税与 AI 制造**：最近的新闻强调了对台湾制造芯片征收关税的计划，这可能会激励像 TSMC 这样的公司在美国建立晶圆厂，尽管由于建设所需时间，其效果会有所延迟。
   - 讨论集中在关税对美国制造业以及在国内培训熟练工人的影响。
- **DeepSeek 应用中的华为芯片**：据报道，DeepSeek 在 Nvidia H800 芯片上进行了训练，但现在正在华为的新型 910C 芯片上运行推理，这标志着其运营能力的重大转变。
   - 这一转变凸显了在全球供应链挑战下，国内芯片生产日益增长的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/Dorialexander/status/1884167945280278857">来自 Alexander Doria (@Dorialexander) 的推文</a>：我觉得这应该是一个更重大的新闻：DeepSeek 在 Nvidia H800 上进行训练，但正在华为生产的新国产芯片 910C 上运行推理。</li><li><a href="https://x.com/markchen90/status/1884303237186216272">来自 Mark Chen (@markchen90) 的推文</a>：祝贺 DeepSeek 制作出 o1 级别的推理模型！他们的研究论文表明，他们在通往 o1 的道路上独立发现了一些核心想法。</li><li><a href="https://x.com/Alibaba_Qwen/status/1883954247743725963">来自 Qwen (@Alibaba_Qwen) 的推文</a>：🎉 恭喜发财🧧🐍 在迎接农历新年之际，我们激动地宣布推出 Qwen2.5-VL，这是我们最新的旗舰视觉语言模型！🚀💗 Qwen Chat: https://chat.qwenlm.ai 📖 Blog: http...</li><li><a href="https://x.com/alibaba_qwen/status/1884263157574820053?s=46">来自 Qwen (@Alibaba_Qwen) 的推文</a>：DeepSeek V3 的爆发引起了整个 AI 社区对大规模 MoE 模型的关注。与此同时，我们一直在构建 Qwen2.5-Max，这是一个在海量数据上预训练的大型 MoE LLM，并且...</li><li><a href="https://x.com/_akhaliq/status/1884278071093502253">来自 AK (@_akhaliq) 的推文</a>：Qwen2.5-Max 刚刚一次性通过（one shotted）了这个提示词：编写一个脚本，在球体内放置三个弹跳的黄球，确保正确处理碰撞检测。让球体缓慢旋转。确保球...</li><li><a href="https://www.tomshardware.com/tech-industry/trump-to-impose-25-percent-100-percent-tariffs-on-taiwan-made-chips-impacting-tsmc">特朗普将对台湾制造的芯片征收 25% 至 100% 的关税，影响 TSMC</a>：未找到描述</li><li><a href="https://map-yue.github.io/">YuE</a>：多模态艺术投影</li><li><a href="https://x.com/BlinkDL_AI/status/1884121603472253385">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：Nvidia 将逐渐过时，因为在 3-5 年内，我们将在手机（而不是数据中心）上运行 1T+ 参数的 OSS 模型，同时仍保持良好的电池续航（！）。这将是 RWKV 类型的模型...</li><li><a href="https://x.com/AravSrinivas/status/1884075898363920510">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：而且是无审查的！引用 David Sacks (@DavidSacks)：这是在不下载 App 或与中国公司共享任何数据的情况下尝试 DeepSeek R1 的几种方法之一。</li><li><a href="https://x.com/sybilhyz/status/1884271592978669579?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Peiyi Wang (@sybilhyz) 的推文</a>：去年，我在没有 RL 经验的情况下加入了 DeepSeek。在进行 Mathshepherd 和 DeepSeekMath 研究时，我独立推导出了这个统一公式来理解各种训练方法。感觉...</li><li><a href="https://x.com/rauchg/status/1884249032035570081?s=46">来自 Guillermo Rauch (@rauchg) 的推文</a>：使用 @aisdk 构建 Agent https://sdk.vercel.ai/docs/ai-sdk-core/agents</li><li><a href="https://x.com/madiator/status/1884284103354376283?s=46">来自 Mahesh Sathiamoorthy (@madiator) 的推文</a>：我们宣布推出 Open Thoughts，这是我们为策划最佳开放推理数据集而发起的大规模开源努力！DeepSeek-R1 很棒，但我们仍然无法获得高质量的开放推理...</li><li><a href="https://x.com/daniel_mac8/status/1883855502553252234">来自 Dan Mac (@daniel_mac8) 的推文</a>：每个人都在将 DeepSeek-R1 与 o1 进行比较，却忘记了 Gemini 2 Flash Thinking，它在每一项成本和性能指标上都优于 R1。</li><li><a href="https://x.com/_akhaliq/status/1884053159175414203?s=46">来自 AK (@_akhaliq) 的推文</a>：它变得越来越好：YuE (乐) 开源全曲音乐生成模型，足以与 Suno AI 媲美！它兼容 Hugging Face 和 LLAMA，易于微调。</li><li><a href="https://x.com/justinlin610/status/1884263803451498794?s">来自 Junyang Lin (@JustinLin610) 的推文</a>：Qwen2.5-Max 发布了。基准测试看起来不错，希望大家能尝试一下，看看对这个新模型的感受！Qwen Chat: https://chat.qwenlm.ai (模型选择 Qwen2.5-Max) API ...</li><li><a href="https://x.com/jordiponsdotme/status/1884152915704832079?s=46">来自 Jordi Pons (@jordiponsdotme) 的推文</a>：关于 YuE (乐)，新的开源全曲音乐生成模型 🎵 - 文本和歌词调节 - 长格式音乐生成，长达 5 分钟 - 许可协议：CC BY-NC 4.0 - 7B 模型 - 英语、中文、日语...</li><li><a href="https://x.com/reach_vb/status/1883961488320389376?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：哈哈，Qwen 2.5 VL 可以开箱即用地执行 Computer Use，正面硬刚 OpenAI Operator！🐐</li><li><a href="https://x.com/vllm_project/status/1883966341557936514">来自 vLLM (@vllm_project) 的推文</a>：🚀 随着今天 v0.7.0 的发布，我们很高兴地宣布 vLLM V1 的 Alpha 版本：一个重大的架构升级，速度提升 1.7 倍！整洁的代码、优化的执行循环、零开销的前缀缓存...</li>

i><li><a href="https://x.com/justinlin610/status/1884263803451498794?s=46">来自 Junyang Lin (@JustinLin610) 的推文</a>：Qwen2.5-Max 发布了。基准测试表现不错，希望大家能试用一下，看看对这个新模型的感受如何！Qwen Chat：https://chat.qwenlm.ai（模型选择 Qwen2.5-Max）API ...</li><li><a href="https://www.latent.space/p/reasoning-price-war">为什么 o3-mini *必须* 免费：即将到来的 DeepSeek R1、2.0 Flash 和 Sky-T1 价格战</a>：2025 年迄今为止最大的惊喜：推理能力的护城河比任何人想象的都要浅。</li><li><a href="https://x.com/swyx/status/1882933368444309723">来自 swyx /dd (@swyx) 的推文</a>：更新了包含 DeepSeek V3/R1 和 Gemini 2 Flash Thinking 2 结果的价格-Elo 帕累托前沿。笔记：o1-mini 和 o3-mini 必须大幅降价（至少 25 倍）才能跟上节奏。它...</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">混合专家模型 (MoE) LLMs</a>：从底层开始理解 DeepSeek、Grok 和 Mixtral 等模型...</li><li><a href="https://sdk.vercel.ai/docs/ai-sdk-core/agents">Agents</a>：学习如何使用 AI SDK Core 构建 Agent。</li><li><a href="https://github.com/carlini/yet-another-applied-llm-benchmark/issues/10">你想为此制作一个排行榜吗？· Issue #10 · carlini/yet-another-applied-llm-benchmark</a>：嗨！非常酷的工作！我是 HuggingFace 的一名研究员，从事评估和排行榜工作。我理解这个酷炫的评估套件首先是为了评估你个人的用例...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ibxj3a/trump_to_impose_25_to_100_tariffs_on_taiwanmade/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://tenor.com/view/jack-donaghy-30rock-alec-baldwin-counter-offer-1dollar-gif-16129209">Jack Donaghy 30rock GIF - Jack Donaghy 30Rock Alec Baldwin - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtubetranscriptoptimizer.com/blog/05_the_short_case_for_nvda">做空 Nvidia 股票的理由</a>：Nvidia 很难达到目前市场高预期的所有原因。</li><li><a href="https://positivetenacity.com/2025/01/27/rise-of-the-a2a-economy-how-ai-agent-to-agent-interactions-will-reshape-the-world/">A2A 经济的崛起：AI Agent 与 Agent 之间的交互将如何重塑世界</a>：在 Sendbird，我们一直积极与全球各地考虑采用对话式 AI 来增强客户服务的公司合作。在这个过程中，我们遇到了……</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_sou">混合专家模型 (MoE) LLMs</a>：从底层开始理解 DeepSeek、Grok 和 Mixtral 等模型...
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1333846279200968744)** (1 条消息): 

> `NotebookLM 协作功能、用户反馈、产品访谈、调查参与` 


- **NotebookLM 寻求用户对协作改进的建议**：团队收到了要求在 NotebookLM 中提供**更好的分享和协作功能**的反馈，目前正在组织 **30 分钟的访谈**以收集更多见解。
   - 鼓励感兴趣的成员填写[调查问卷](https://forms.gle/pZCvgaJbnH1cm59M9)参与产品塑造。
- **增强的来源和笔记用户控制**：用户很快将能更好地控制其来源，包括**编辑、添加和删除来源**，以及对其进行评论。
   - 此外，功能还将包括**编辑音频概览**、制作可供个人使用的可编辑副本以及自定义笔记本设置。



**提到的链接**：<a href="https://forms.gle/pZCvgaJbnH1cm59M9">NotebookLM 中的分享与协作</a>：你好！感谢填写此表单。我们希望了解如何改进 NotebookLM 的分享和协作功能，我们很想听听您的意见！如果您...

  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1333598215790006333)** (17 条消息🔥): 

> `NotebookLM 定制化, DeepSeek AI 进展, 语音合成不一致性, 在 LLM 中使用大型文档, AI 提示词中的性格特征` 


- **NotebookLM 定制化实现动态输出**：一位成员分享了他们如何通过在 NotebookLM 中针对特定内容、风格和语调定制 Prompt，从而获得多样化的输出。
   - 他们强调了指令的灵活性，但也指出音频输出每 60 秒会出现一次不一致。
- **Rax 揭示 DeepSeek 的 AI 革命**：一只名为 Rax 的赛博朋克浣熊在时代广场的广告牌上发布了关于 **DeepSeek**（一家中国初创公司的新型 AI 助手）的消息，引发了科技界的关注。
   - 他们的行动导致了显著的市场动荡，使主要科技公司的市值蒸发了超过 **7000 亿美元**，详情见 [YouTube 揭秘视频](https://youtu.be/Hq7PHvExqUY)。
- **社区寻求 Cursor AI 的帮助**：一位用户表示有兴趣学习 **Cursor AI** 以简化 IT 项目的启动流程，并正在寻求帮助。
   - 该咨询表明其重点在于消除项目启动阶段对开发人员的依赖，强调了其商业适用性。
- **大型文档集成的挑战**：一位用户询问是否可以在 NotebookLM 中使用两本厚重的教科书进行环境工程讨论，并寻求可行性指导。
   - 回复强调，大型数据源可能会使特定查询变得复杂，建议拆分文档以规避 **needle in the haystack**（大海捞针）问题。
- **语音合成的不一致性**：一位用户在经历角色声音波动后，寻求如何在 NotebookLM 中实现一致的语音输出。
   - 另一位成员解释说，NotebookLM 可以包含特定的特征，但节奏上的不一致可能仍然会发生。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/14278184?hl=en#:~:text=What%20is%20the%20maximum%20file%20size%20limit%20for%20sources%20in%20NotebookLM%3F">常见问题解答 - NotebookLM 帮助</a>: 未找到描述</li><li><a href="https://youtu.be/Hq7PHvExqUY">🚨 赛博朋克浣熊 Rax 揭示 DeepSeek 的 AI 革命！ 🚨</a>: 采取大胆行动，赛博朋克浣熊 Rax 潜入时代广场的 LED 广告牌，发布关于 DeepSeek 突破性 AI 的未经审查的消息……
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1333530268321054732)** (77 条消息🔥🔥): 

> `用户角色澄清、Podcast 功能与限制、NotebookLM 语言与导出问题、Gemini 与音频生成、引用与参考文献管理` 


- **了解个人资料中的“User”角色**：用户询问了个人资料中可见的“user”角色，特别是在 Moderator 权限背景下，并建议这些角色与不同的 Discord 组相关联。
   - 提供了关于不同角色的澄清，但具体细节仍不明确。
- **Podcast 功能仍有缺失**：用户对无法自定义 Podcast 时长、口音或有效生成不同内容表示沮丧，突显了当前功能的局限性。
   - 虽然用户可以生成自己的 Podcast，但许多关于增强功能以及公开或私有状态的明确需求仍未得到解决。
- **语言与导出功能的挑战**：参与者注意到在生成非英语笔记时存在问题，并确认目前无法使用外部文档导出功能。
   - 提到了调整 Google 账户设置等建议，以潜在地解决语言输出问题。
- **对 Gemini 集成的期待**：讨论了将 Gemini 2.0 Flash 集成到 NotebookLM 中的情况，并期待 Deep Research 增强等进一步功能。
   - 关于集成 Gemini Pro 潜力的推测引发了兴趣，尽管细节仍不确定。
- **NotebookLM 对引用功能的需求**：用户对手动添加引用的过程表示沮丧，同时强调了具备自动引用功能的重要性。
   - 对改进引用管理的呼吁反映了对更好支持学术工作的功能的渴望。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://illuminate.google.com/">Illuminate | Learn Your Way</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你更快理解复杂内容的 Gen AI 工具。</li><li><a href="https://www.reddit.com/r/Cracked_Software_Hub/comments/1fo875c/tradingview_premium_cracked_version_available_for/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1333793932156473395)** (22 条消息🔥): 

> `Minimizing Startup Times for LLMs, Optimizations for Model Loading, Utilizing Modal's RAM Snapshots, GPU Direct Storage (GDS) Considerations, Torch Distributed Package` 


- **探索最小化启动时间的方案**：讨论集中在 Serverless LLM 推理环境中**最小化启动时间**的方案，特别是针对加载 **DeepSeek-r1** 等模型。
   - 针对约 128GB 模型的 **2 分钟加载时间**提出了担忧，引发了对延迟成因及潜在优化的询问。
- **模型加载优化建议**：建议探索将模型权重手动保存为 **torch state dicts** 并使用 `torch.load` 进行高效加载的技术。
   - 一位用户表示，如果使用 **PyTorch**，他们会通过使用 `torch.device('meta')` 进行并行加载来优化。
- **利用 RAM 快照应对冷启动**：一位成员强调了利用 Modal 提供的 RAM 快照进行[初始化](https://modal.com/docs/guide/cold-start#share-initialization-work-across-cold-starts-with-memory-snapshots-beta)，从而大幅改善冷启动时间的潜力。
   - 这引发了关于 **Modal 内存快照**如何帮助解决冷启动并降低延迟的讨论。
- **测试 GPU 带宽和加载速度**：有人断言，在拥有 4 个 L40s 和理想磁盘配置的情况下，如果能妥善利用可用带宽，加载时间理论上可以低至**几秒钟**。
   - 对话指出，如果处理得当，优秀的 **NVMe** 磁盘应能实现约 **10-15 秒**的加载时间。
- **Torch Distributed 包的行业标准**：**torch.distributed** 包等相关工具作为使用 PyTorch 进行高效模型处理的常见实践被提及。
   - 这为不熟悉某些功能以及分布式系统领域常用效率方法的成员建立了一个基准预期。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modal.com/docs/guide/cold-start#share-initialization-work-across-cold-starts-with-memory-snapshots-beta">冷启动性能</a>：Modal Functions 在容器中运行。</li><li><a href="https://github.com/modal-labs/modal-examples/blob/main/06_gpu_and_ml/llm-serving/llama_cpp.py">modal-examples/06_gpu_and_ml/llm-serving/llama_cpp.py at main · modal-labs/modal-examples</a>：使用 Modal 构建的程序示例。通过在 GitHub 上创建账号为 modal-labs/modal-examples 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1333527951316353065)** (16 条消息🔥): 

> `Grace Hopper architecture, Jupyter Lab setup issues, CUDA pointer alignment, H100 PCIe/SXM card, GH200 rental rates` 


- **对 Grace Hopper 架构的兴趣**：一位成员强调了对 **Grace Hopper 架构**的兴趣，该架构涉及 CPU 和 GPU 组件的紧密集成，与现有的 **Hopper Chips** 不同。
   - 该架构旨在提高性能指标并降低需要快速计算的应用的延迟。
- **Jupyter Lab 安装故障排除**：一位成员在安装 Jupyter Lab 后遇到了 **ImportError**，这归因于模块加载失败导致的环境设置问题。
   - 建议包括创建全新的虚拟环境，确保 **CUDA** 兼容性，并安装 **ninja** 和 **cmake** 等必要包。
- **理解 CUDA 指针对齐**：关于 **CUDA** 中指向 **doubles** 的解引用指针对齐问题引发了疑问，对齐不良可能导致未定义行为。
   - 一位成员建议确保加载对齐到 **64 bits** 以实现安全访问，并指出 **cudaMalloc** 返回对齐到 **256 bytes** 的指针。
- **GH200 的租赁价格**：一位成员告知 **GH200** 的租赁价格为 **$1.5/hr**，暗示了 GPU 使用的潜在高性价比方案。
   - 这种选择可以增加临时项目获取先进硬件的机会，而无需巨额投资。
- **H100 的部署模式**：讨论提到 **H100** 应作为典型的 **PCIe/SXM** 卡运行，这与 **Grace Hopper** 等特定架构实现有所不同。
   - 这种标准化有助于用户熟悉其集成和性能预期。



**提到的链接**：<a href="https://pytorch.org/">
    
      PyTorch
    
  </a>：未找到描述

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1333754990690570271)** (9 条消息🔥): 

> `FP8 Conversion, FP8 Stochastic Rounding, PyTorch GB200 Support, CUDA 12.8 Compatibility` 


- **分享了 FP8 转换目标**：一位用户旨在学习 **FP8**，并将 **bfloat16** 转换为带有正确 **stochastic rounding**（随机舍入）的 **FP8**，并分享了他们的代码仓库 [此处](https://github.com/Muhtasham/fp8-auto)。
   - 他们在张量操作中遇到了符号位翻转的问题。
- **分享了 Stochastic FP8 资源**：一名成员指出 **torchao** 拥有将 **FP32** 转换为 **FPx** 的工具，并提供了 [相关代码](https://github.com/pytorch/ao/blob/e151d6a5288177a1a635c71fecd145654745af4c/torchao/prototype/custom_fp_utils.py#L27) 的链接。
   - 这得到了参与 **FP8** 讨论的原用户的认可和感谢。
- **未来 LUT 与性能考量**：一位用户分享了一个 **TODO** 笔记，关于检查所有浮点值的查找表 (LUT) 是否比位移 (bit shifting) 更快，特别是对于只有 **16 个唯一值** 的 **fp4**。
   - 这一评论被认为很有趣，为正在进行的性能优化讨论做出了贡献。
- **关于 PyTorch 对 GB200 支持的查询**：一名成员询问了在 **GB200** GPU 上运行 **PyTorch** 的状态，引用了关于支持情况以及需要针对 **CUDA 12.8** 进行构建的讨论。
   - 他们想知道是否有现成的容器可用于此配置。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Muhtasham/fp8-auto">GitHub - Muhtasham/fp8-auto: FP8 stochastic rounding</a>：FP8 stochastic rounding。通过在 GitHub 上创建账号来为 Muhtasham/fp8-auto 的开发做出贡献。</li><li><a href="https://github.com/pytorch/ao/blob/e151d6a5288177a1a635c71fecd145654745af4c/torchao/prototype/custom_fp_utils.py#L27">ao/torchao/prototype/custom_fp_utils.py at e151d6a5288177a1a635c71fecd145654745af4c · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/blob/e151d6a5288177a1a635c71f">GitHub - pytorch/ao at e151d6a5288177a1a635c71fecd145654745af4c</a>：用于训练和推理的 PyTorch 原生量化和稀疏化 - GitHub - pytorch/ao at e151d6a5288177a1a635c71fecd145654745af4c
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1333571555896066050)** (2 条消息): 

> `DeepSeek-R1 Release, Matrix Multiplication Challenge` 


- **DeepSeek-R1 亮相 AI 舞台**：最新发布的 **DeepSeek-R1** 是一个开源权重模型，包含较小的蒸馏版本，使机器学习领域的研发更加容易。
   - 该帖子详细阐述了其 **training method**（训练方法），这有助于复现类似于 OpenAI **O1** 的推理模型。
- **全班参与矩阵乘法挑战**：一位教授分享了一个有趣的挑战，他的 **全班学生** 在共享的 Excel 表格上通过手算完成矩阵乘法，仅用时 **3 分 30 秒**。
   - 这项现场活动不仅促进了协作，还展示了学生在矩阵运算方面的实践技能。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://newsletter.languagemodels.co/p/the-illustrated-deepseek-r1">The Illustrated DeepSeek-R1</a>：推理 LLM 的配方</li><li><a href="https://x.com/ProfTomYeh/status/1883956683430596802">Tom Yeh (@ProfTomYeh) 的推文</a>：我挑战了全班学生，让他们在共享的 Excel 表格上一起现场手算 ✍️ 矩阵乘法。我们花了 3 分 30 秒。👇 更多内容。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1333542150192758795)** (1 条消息): 

> `Tile Lang, BitBLAS repo` 


- **Tile Lang 终于发布了！**：一名成员对 **Tile Lang** 的发布表示兴奋，并指出早在 10 月份的 [BitBLAS repo](https://link.to.bitblas) 提交记录中就提到了它。
   - 他们希望利用 **Tile Lang** 来编写 **BitBLAS** 目前缺失的 **efficient backward kernels**（高效反向算子）。
- **对高效反向算子的期待**：同一位成员表达了终于能编写 **BitBLAS** 一直缺失的那些 **efficient backward kernels** 的愿望。
   - 这种情绪反映了人们对 Tile Lang 可能为该项目带来的能力的日益期待。

### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1333830777082875917)** (1 条消息): 

> `ThunderKittens 改进、测试 Kernel 性能、测试泛化` 


- **为 ThunderKittens Kernel 测试做贡献**：一名成员计划通过改进 [此处](https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/torch_scaled/gentests.py) 的 **gentests.py** 文件来增强 *ThunderKittens* 的测试框架。
   - 其目标是将兼容性扩展到任意 **M, N, K** 维度，并将 *TK scaled_mm kernel* 与 **torch._scaled_mm** 进行直接对比。
- **当前测试的局限性**：目前 **gentests.py** 中的测试仅限于 **M=N=K** 的情况，只能在这些条件下运行。 
   - 此次改进旨在解决这一局限性，从而支持更广泛的测试场景和性能评估。



**提到的链接**：<a href="https://github.com/HazyResearch/ThunderKittens/blob/main/kernels/torch_scaled/gentests.py">ThunderKittens/kernels/torch_scaled/gentests.py at main · HazyResearch/ThunderKittens</a>：用于快速 Kernel 的 Tile 图元。通过在 GitHub 上创建账号为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1333612098076545195)** (38 条消息🔥): 

> `Reasoning Gym Pull Request、Apple 数据集许可证问题、为 GSM8K 生成模板、OpenRLHF 试运行、多重许可与版权` 


- **依赖项延迟了 Reasoning Gym 的合并**：最近的 [Pull Request #11](https://github.com/open-thought/reasoning-gym/pull/11) 旨在为 Reasoning Gym 添加对 CLRS 任务的支持，但由于可能依赖 Jax 和 TensorFlow（这可能与项目目标不符）而面临犹豫。
   - 建议的替代方案是在保留版权声明的同时复制算法，以避免依赖性问题。
- **处理 Apple 数据集的法律问题**：由于版权问题，使用 Apple 数据集存在疑虑，其版权似乎与 Reasoning Gym 的 Apache 2.0 许可证不兼容，因此需要转而复制该数据集。
   - 成员们讨论了遵守许可条款的策略，强调了在利用 GSM8K 等数据集的想法时保留版权声明的重要性。
- **GSM8K 模板生成策略**：团队同意开始基于现有的 GSM8K 模板生成新模板，目标是产生约 100 个变体，以便进行人工正确性验证。
   - 此过程包括根据现有问题的结构定义模板问题、变量分配以及计算正确答案的函数。
- **OpenRLHF 的初步试验**：成员分享了使用 OpenRLHF 在 Llama 3B 模型上进行试运行 sum-chain 的首次近端策略优化（PPO）运行，并在 WandB 上追踪结果。
   - 有关试验的细节有助于为模型训练的未来迭代提供后续步骤和调整的参考。
- **多重许可讨论**：团队成员表达了对多重许可的担忧，他们寻求简化依赖结构并确保与项目许可证保持一致。
   - 讨论引导大家探索使用语言模型从受版权保护的数据中直接生成模板，且不违反许可条款。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://wandb.ai/andreaskoepf/openrlhf_train_ppo">andreaskoepf</a>：Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/openai/grade-school-math/blob/master/LICENSE">grade-school-math/LICENSE at master · openai/grade-school-math</a>：通过在 GitHub 上创建账号为 openai/grade-school-math 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/11">Add support for all CLRS tasks by panispani · Pull Request #11 · open-thought/reasoning-gym</a>：CLRS 是经典的算法教科书。Deepmind 推出了 CLRS 基准测试，其中还包括大多数经典算法的文本版本，称为 CLRS-text。在此 PR 中，我移植了所有...
</li>
</ul>

</div>
  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1333577398616064091)** (60 messages🔥🔥): 

> `SP24 秋季学期课程、讲义幻灯片可用性、研究方向资格、Hackathon 信息、应用方向协作` 


- **SP24 秋季学期课程不提供异步学习**：一位成员询问了 SP24 秋季学期课程的回归以及异步学习和证书的可能性，对此澄清了将不提供异步证书。
   - 回复指出该课程可能会在未来的学期再次开设。
- **讲义幻灯片可能在讲座后分享**：关于讲座前是否提供幻灯片的问题，确认了幻灯片通常在讲座后分享。
   - 提到讲师会尝试尽可能提前将幻灯片添加到课程网站。
- **MOOC 学生参加研究方向（Research Track）的资格**：提出了关于 MOOC 学生是否可以注册研究方向的疑问，并保证相关细节将很快发布。
   - 澄清了 MOOC 学生通过填写报名表仍有资格获得证书。
- **本学期未安排 Hackathon**：一位用户询问了关于组队参加 Hackathon 的事宜，对此表示本学期没有计划举办 Hackathon。
   - 未来的进一步讨论将明确有关 MOOC 学生的项目政策。
- **允许在应用方向（Application Track）进行协作**：成员询问了在应用方向组队的情况，并确认小组可以由 3-4 名学生组成。
   - 关于应用方向和 MOOC 课程设置的细节即将公布。



**提及的链接**：<a href="https://rdi.berkeley.edu/adv-llm-agents/sp25">CS294/194-280 Advanced Large Language Model Agents</a>：2025 春季

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1333594869750431786)** (12 messages🔥): 

> `幻灯片可用性、YouTube 讲座反馈、使用 NotebookLM 作为研究工具` 


- **幻灯片现已上线**：一位成员询问是否可以获取 **slide deck**（幻灯片），另一位成员回复称其**已经上线**。
   - 您可以查看平台获取最新资料。
- **对 YouTube 讲座视频过长的担忧**：一位成员对 **YouTube 讲座**时长超过 **4 小时**表示沮丧，评论指出实际内容在 **35 分钟**后才开始。
   - 另一位成员确认他们正在**剪辑视频**，以删除不必要的部分以确保清晰。
- **使用 NotebookLM 辅助研究**：一名学生寻求关于使用 **NotebookLM** 进行研究的指导，并提到在研究工具方面的经验有限。
   - 另一位成员解释说，只需上传 **PDF**，它就会生成播客风格的对话，使概念更容易理解。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google/">Google NotebookLM | AI 驱动的笔记与研究助手</a>：利用 AI 的力量进行快速总结和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可以信赖的信息。</li><li><a href="https://www.youtube.com/live/g0Dwtf3BH-0?si=48-etlTVZ5VblF8c">CS 194/294-280 (Advanced LLM Agents) - Lecture 1, Xinyun Chen</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1333544545190805594)** (42 条消息🔥): 

> `Chat Template 错误, DeepSeek 实现, GPT4All 路线图, GPT4All 中的模型选项, LocalDocs 文件上传` 


- **Chat Template 错误持续存在**：用户报告了 Chat Template 的语法错误，引发了关于使用 Jinja 模板进行潜在修复和配置的讨论。
   - 一位用户分享了一个经过修正的 Jinja 模板，该模板在设置聊天角色（chat roles）方面表现更好。
- **探索 DeepSeek 集成**：几位用户讨论了在 GPT4All 中运行 DeepSeek 的各个版本，在模型兼容性和聊天历史记忆方面的成功程度参差不齐。
   - 一位用户提供了一个 [Hugging Face 链接](https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF)，指向另一个 DeepSeek 模型版本，并附带了运行说明。
- **GPT4All 路线图的不确定性**：有人对 GPT4All 的潜在路线图提出了疑问，一些用户对多年来未解决的功能请求表示沮丧。
   - 一位成员评论了开发者追求某些功能（如 Chain of Thought (CoT) 实现）的可能性。
- **LocalDocs 文件格式挑战**：用户注意到将 XLSX 文件上传到 LocalDocs 存在困难，扩展名会被移除，但文件仍可在聊天中上传。
   - 有人询问了格式限制，以及未来的更新是否允许保留完整的 XLSX 文件。
- **Web Search Beta 版本发布查询**：一位用户询问 GPT4All 的 Web Search 功能是否仍在开发中，并指向了 GitHub 的文档。
   - 其他人对该功能表示了兴趣，并请求更新其进展。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF">bartowski/DeepSeek-R1-Distill-Llama-8B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=rzMEieMXYFA">如何在本地运行 DeepSeek-R1 | 免费开源推理 AI</a>: 了解如何使用 Ollama 在本地机器上运行强大的开源推理 AI DeepSeek-R1。本分步指南将向你展示如何利用 R1...</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Web-Search-Beta-Release">Web Search Beta 版本发布</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://www.zdf.de/dokumentation/terra-x/ein-tag-in-berlin-1943-der-passfaelscher-cioma-schoenhaus-doku-100.html">Ein Tag in Berlin 1943 – Der Passfälscher Cioma Schönhaus</a>: 1943 年，根据阿道夫·希特勒的意愿，柏林将变得 "judenrein"（没有犹太人）。Schönhaus 是数万名面临驱逐威胁的犹太人之一。 
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1333809205718679647)** (31 messages🔥): 

> `Running Distributed Recipes, Issues with 'torchrun', Distributed Init Protocols, Multinode Setup on Mac, Debugging Torch Distributed` 


- **运行分布式 Recipe 时遇到麻烦**：一位用户报告了在使用 **torchtune** 运行分布式 Recipe 时出现的问题，导致了与缺失模块相关的导入错误以及 **c10d 错误**。
   - *“我可以正常运行其他使用 torch.distributed 和 torchrun 的测试脚本”* 这表明问题可能出在特定的 Recipe 上。
- **在不使用 Tune 的情况下使用 torchrun**：一位成员寻求关于在不使用 **torchtune** 的情况下执行 `torchrun` 的指导，最终通过修正命令格式获得了成功。
   - 这一调整解决了之前关于 **NCCL** 的错误，并允许成功执行分布式脚本。
- **关于分布式初始化协议的问题**：讨论围绕指定的分布式初始化命令 `init_process_group('cuda:nccl,cpu:gloo')` 及其在 Mac 环境下的影响展开。
   - 有人担心这种设置可能会导致问题，评论建议不指定协议可能会产生更好的结果。
- **多节点设置与调试挑战**：参与者开玩笑说“串联的 Mac mini”会导致分布式设置变得复杂，并表示需要更简便的调试方法。
   - 用户经验强调了在没有 SSH 的情况下进行调试的挑战，以及**分布式系统（distributed systems）**容易出状况的特性。
- **分布式 API 文档缺失**：成员们对分布式 API **文档匮乏**表示沮丧，这导致了实现过程中的困惑和错误。
   - 大家的共识是，更好的文档可能会减轻在使用 PyTorch 处理分布式进程时面临的一些复杂性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/56915b093a20b2fbd4d6f79f100670c6c496d8b3/torch/distributed/distributed_c10d.py#L907)">pytorch/torch/distributed/distributed_c10d.py at 56915b093a20b2fbd4d6f79f100670c6c496d8b3 · pytorch/pytorch</a>：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/pytorch/blob/a08f7f326637adb77ee2ba996425c67a7305f97e/torch/distributed/run.py#L416">pytorch/torch/distributed/run.py at a08f7f326637adb77ee2ba996425c67a7305f97e · pytorch/pytorch</a>：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
</li>
</ul>

</div>
  

---


### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1333550344549957735)** (1 messages): 

> `Model Comparisons, Image Analysis` 


- **质疑当前模型的相关性**：一位成员提出疑虑，询问所有被对比的模型是否都已经过时。
   - *“它所对比的所有模型难道不都老了吗？”*
- **分享图像分析**：附带了一张用于分析的图像，表明正在进行的讨论中包含视觉组件。
   - 该图像可能包含关于所引用模型的关键见解，但未作进一步阐述。


  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1333546249093124298)** (2 messages): 

> `DeepSeek-R1 API Integration, SOFTIQ SaaS App, Tender Analysis Efficiency` 


- **DeepSeek-R1 API 加入 LlamaIndex**：LlamaIndex 宣布了与 [DeepSeek-R1 API](https://api-docs.deepseek.com/) 的官方集成，支持使用 `deepseek-chat` 和 `deepseek-reasoner` 等模型来实现增强功能。
   - 要开始使用，用户只需通过命令 `%pip install llama-index-llms-deepseek` 安装 LlamaIndex。
- **SOFTIQ 彻底改变招标流程**：新的 [SOFTIQ SaaS 应用](https://t.co/grvoRR6TJb) 利用 LlamaIndex Workflows 将公共部门标书的分析时间大幅缩减至每份 **10 分钟**以内。
   - 正如其宣传材料所述，这种创新方法提高了选择准确性，帮助建筑公司减少了无效劳动。



**提到的链接**：<a href="https://t.co/jtfBvBig1y">DeepSeek - LlamaIndex</a>：未找到描述

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1333778705381720146)** (18 messages🔥): 

> `LlamaReport 文档、Pull Request 审查、推理模型中的 RAG 检索、FastAPI 事件流` 


- **LlamaReport 文档即将发布**：成员们讨论了 **LlamaReport 文档**的状态，表示目前正在准备中，很快就会发布。
   - 目前，一位成员分享了一个 [Twitter 链接](https://twitter.com/llama_index/status/1869094544169677138)以获取更多信息。
- **移除死链的 Pull Request**：一位成员请求对其 [Pull Request](https://github.com/run-llama/llama_index/pull/17652) 进行审查，该 PR 移除了 **fine-tuning.md** 文档中的一个死链。
   - 他们指出，由于该链接在仓库中已不存在，这只是一个简单的单行更改。
- **推理模型中 RAG 检索的挑战**：一位成员询问如何在推理模型的 `<think>` 步骤中触发 **RAG 检索**，并引用了 [Search-o1 论文](https://search-o1.github.io)作为参考。
   - 另一位成员回答说，该方法将涉及流式响应、中断搜索并将结果插入回新的流中。
- **FastAPI 事件流工作流**：一位成员寻求关于在 **FastAPI** 中以 JSON 格式返回流式数据的指导，旨在改进 Postman 中的响应格式。
   - 社区建议使用异步生成器将 Pydantic 对象作为字典或 JSON 字符串返回，以确保格式正确。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://search-o1.github.io">Search-o1: Agentic Search-Enhanced Large Reasoning Models</a>：未找到描述</li><li><a href="https://www.llamaindex.ai/blog/building-blocks-of-llm-report-generation-beyond-basic-rag">Building Blocks of LLM Report Generation: Beyond Basic RAG — LlamaIndex - 基于企业数据构建知识助手</a>：LlamaIndex 是一个简单、灵活的框架，用于使用连接到企业数据的 LLM 构建知识助手。</li><li><a href="https://github.com/run-llama/llama_index/pull/17652">由 Riddhimaan-Senapati 移除 docs 中 fine-tuning.md 的死链 · Pull Request #17652 · run-llama/llama_index</a>：描述：移除了 docs 中 fine-tuning.md 的死链，因为该文件已不在仓库中。被移除的链接是：https://docs.llamaindex.ai/en/stable/optimizing/...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1333536783379595407)** (6 messages): 

> `文档状态、Deepseek 与 Modular 之争` 


- **文档停机与恢复**：团队报告称 **docs 曾下线**，但现在已恢复并可用，包括 nightly 版本中的 **GPU package API 文档**。
   - 成员们对快速解决问题表示感谢，并对文档状态的更新表示赞赏。
- **Deepseek 声称优于 Modular**：一位成员评论说，**Deepseek** 在实现与 **Max** 和 **Mojo** 类似的目标方面似乎已经超越了 **Modular**。
   - 然而，其他人认为两者本质上是不同的，将 Modular 的角色比作支持农民的拖拉机商店，而不是与他们竞争。


  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1333591026220269678)** (1 messages): 

> `MAX 仓库变更、Mojo 仓库更新` 


- **MAX 和 Mojo 仓库进行重组**：**nightly** 分支将更名为 **main**，接收最频繁的更新，而 **stable** 分支将镜像最新的稳定版本，目前为 **24.6**。
   - 每个稳定版本都将在 GitHub 中打上标签，所有打开的 Pull Request 将在过渡后调整到正确的分支。
- **分支切换指南**：当前在 **nightly** 分支上的开发者应在更改后按照特定的 Git 命令切换到新的 **main** 分支。
   - 同样，当前在 **main** 分支上的开发者需要在更新应用后执行命令切换到新的 **stable** 分支。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1333535789086146592)** (10 条消息🔥): 

> `Documentation Outage, Mojo Code Issues, Garbage References in Code, Callback Capturing Behavior, String Captures Clobbered` 


- **文档停机，团队正在处理**：文档目前处于停机状态，团队正在努力尽快解决此问题，用户们表现得很耐心。一位用户幽默地补充道：“耐心是我的中间名”。
   - 文档现在已经恢复，成员们松了一口气。
- **避免 Mojo 代码中的垃圾引用**：一位用户在他们的 Mojo 代码中遇到了 `write_node` 函数的问题，其中对 `self` 的引用变成了垃圾数据。他们通过移除捕获 callback 的功能成功修复了它。
   - 有人担心问题是否源于捕获行为，因为他们在描述中引用了相关示例。
- **捕获问题的澄清**：一位成员发布了一个链接，可能有助于解决 callback 捕获问题，但表示这并没有解决问题。他们提到在他们的解决方案中，`written` 和 `size` 都没有被正确捕获。
   - 另一位用户分享了一个 GitHub Gist 链接，供其他人尝试一个更简单的示例，以进一步调查捕获问题。
- **关于 String 捕获被破坏的讨论**：一位用户询问问题是否可能与 closure 或捕获函数中 String 捕获被破坏（clobbered）有关。这引发了关于代码中捕获复杂性的进一步讨论。



**提到的链接**：<a href="https://gist.github.com/sstadick/20cfda1db33041bd0ebed6798757c22d">tree.mojo</a>: GitHub Gist: 立即分享代码、笔记和代码片段。

  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1333565148941783050)** (6 条消息): 

> `tinygrad PR 8781, Python CUDA Emulator for FP8, MathTrait and SimpleMathTrait Unification, Tests for View.stride() and View.flip(), Bounty questions` 


- **tinygrad PR 8781 的悬赏**：一位成员强调了 PR [#8781](https://github.com/tinygrad/tinygrad/pull/8781)，该 PR 涉及在 `tinygrad` 代码库中用 **flip** 替换 **stride**，完成后提供 **$100 悬赏**。
   - 该 PR 旨在通过处理相对简单的代码更改，帮助新人熟悉项目。
- **FP8 实现的复杂性**：一位成员质疑是否需要为 **FP8** 添加 **Python CUDA 模拟器**，并提到了与数值内存存储相关的挑战。
   - 具体来说，他们指出 `struct.pack` 不支持 **FP8**，这与 **bfloat16** 现有的问题类似。
- **MathTrait 和 SimpleMathTrait 的重构**：有人就 `MathTrait` 和 `SimpleMathTrait` 的统一悬赏寻求澄清，特别是重构是否涉及将 **log2** 等操作委托给 **MathTrait**。
   - 他们提供了一个代码片段，展示了 `Tensor` 中的 `log2` 如何委托给 **MathTrait**，同时也询问是否必须保留文档。
- **关于 View.stride() 更改的查询**：关于让 **View.stride()** 返回 **View.flip()** 的复杂性引发了讨论，成员们在争论仅仅通过所有测试是否足够简单。
   - 目前尚不清楚这一更改是否足够，或者是否需要进一步修改。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>: GitHub - tinygrad/tinygrad: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️</li><li><a href="https://github/tinygrad/tinygrad/pull/8781">replace stride with flip ($100 bounty for finishing this) by geohot · Pull Request #8781 · tinygrad/tinygrad</a>: 暂无描述
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1333633982327427072)** (10 messages🔥): 

> `Tensor.isclose 和 Tensor.allclose 方法，负步幅（Negative stride）解释，针对 tinygrad 的 Git Branching 教程` 


- **提交 Tensor.isclose 和 Tensor.allclose PR**：一名成员提交了 `Tensor.isclose` 和 `Tensor.allclose` 的 PR，旨在通过将 `isclose()` 定义为 ```(self - other).abs() <= atol + rtol * other.abs()``` 来与 [torch](https://pytorch.org) 的功能保持一致。
   - 然而，**目前测试失败**，引发了关于如何解决这些问题的讨论。
- **关于负步幅（Negative Stride）解释的困惑**：一位成员讨论了负步幅可以被视为一次翻转（flip）后接一个正步幅的观点，特别是在 **conv2d** 操作的语境下。
   - 他们表示不确定代码是否表明步幅仅为 {-1,1}，或者它是否包含不同的步幅值，如 (3,2)。
- **关于术语 'Swizzle' 的询问**：一位成员承认对 **swizzle** 一词感到困惑，并询问是否有相关的规范或文档。
   - 他们寻求澄清，因为他们在讨论中一直刻意避开这个词。
- **关于 Tinygrad 学习资源的建议**：另一位成员建议创建一个类似于 [Learn Git Branching](https://learngitbranching.js.org/) 的教程资源，但专门针对 tinygrad 基础和谜题。
   - 他们引用了一个现有的专门针对 tinygrad 张量谜题的 [GitHub 仓库](https://github.com/obadakhalili/tinygrad-tensor-puzzles)，强调了对清晰的基础学习材料的需求。



**提到的链接**：<a href="https://learngitbranching.js.org/">Learn Git Branching</a>：一个交互式的 Git 可视化工具，用于教学和挑战！

  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1333540730286637116)** (2 messages): 

> `Discord 中的问候，用户互动` 


- **交换友好的问候**：**用户在频道中进行了随意的交流**，展示了友好的氛围。
   - *Hi Guys!* 和 *👋* 表达了成员之间的社区感。
- **非正式互动**：该频道**活跃度较低**，只有几条消息表明成员的存在。
   - 这些简单的**互动维持了社区内的参与度**。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1333789469777727550)** (8 messages🔥): 

> `模型响应质量，Classify 端点返回 500 错误，微调模型详情，模型版本控制，Command R+ 模型更新` 


- **用户报告响应质量下降**：一位用户表示担心 **command r+** 模型在重新启动一个曾在 9 月份表现良好的项目后，给出的*响应过于简短*。
   - 另一位成员确认别名（alias）仍然指向同一个模型，这表明自那时以来没有任何变化。
- **Classify 端点遇到 500 错误**：一位用户报告在 Classify 端点遇到 **Error 500**，引发了关于所使用模型的讨论。
   - 经过故障排除后，指出问题应该已经解决，并请求用户确认问题是否仍然存在。
- **询问微调模型规范**：针对错误消息，成员们询问了**微调模型**的细节，特别是要求提供微调 ID（finetune ID）。
   - 寻求澄清以确定微调模型是否是所面临问题的根源。
- **关于模型版本控制替代方案的讨论**：一位用户询问如何有效地指定模型版本，以潜在地解决响应问题。
   - 提供了升级到更新模型版本的选项，如 **command-r-plus-08-2024** 或 **command-r7b-12-2024**，以获得更好的性能。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1333680220862943232)** (6 messages): 

> `更新语音参数设置，AI Agent 咨询` 


- **关于语音参数调整的讨论**：一位成员表示希望通过增加更多样化的参数设置来改进语音输出，从而在保持清晰度和可理解性的同时增强区分度。
   - 他们引用了一个 [Colab notebook](https://colab.research.google.com/drive/1nWmVJY91VyNQTGoT7zIylT_0N6zIdaOm?usp=sharing) 作为示例，并向另一位成员寻求帮助。
- **寻找 AI Agent 专家进行合作**：一位名为 Francis 的成员宣布正在寻找 AI Agent 领域的顾问和专家，旨在将多 Agent 解决方案整合到营销工作流中，以增强自动化程度。
   - 他们鼓励感兴趣的人员通过 DMs 或在服务器内联系，以讨论潜在的合作。



**提及的链接**：<a href="https://colab.research.google.com/drive/1nWmVJY91VyNQTGoT7zIylT_0N6zIdaOm?usp=sharing">Google Colab</a>：未找到描述

  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1333712258735538218)** (1 messages): 

> `算力预算声明，MoE 训练效率，Llama3 GPU 小时数对比` 


- **对算力预算真实性的怀疑**：针对**算力预算 (compute budget)** 的声明出现了疑问，特别是对于一个超过 **600b** 的 MoE 模型，是否仅用 **2.7m GPU 小时数** 就能完成训练。
   - *据报道 Llama3 需要 **7.7m GPU 小时数***，这让人对所提议的训练效率的可行性产生怀疑。
- **比较 MoE 和 Llama3 的训练投入**：讨论强调，虽然 **MoE** 模型可能受益于 **8 bit** 训练且激活参数较少，但其训练预算看起来仍然被低估了。
   - 尽管有 **2x FLOPs** 的优势，但训练如此大型模型所需的总 GPU 小时数感觉太低，缺乏可信度。


  

---


### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1333647219353587784)** (4 messages): 

> `H200 销售策略，Multi Turn Kto 讨论` 


- **H200 以惊人的倍数售出**：一位成员吹嘘以 **5090** 的 **16倍** 价格售出了 **H200**，并称其拥有 **3.41倍的 VRAM**。
   - *大笑的表情符号* 回应了另一位确认自己也做了同样事情的成员的情绪。
- **询问 Multi Turn Kto**：一位成员向频道内的另一位成员询问了 **multi turn kto** 的性能表现。
   - 他们的询问反映了好奇心，但没有引发进一步的讨论或回应。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1333669942125396029)** (1 messages): 

> `OpenInterpreter 技能，导入技能配置` 


- **OpenInterpreter 技能失去功能**：一位成员表达了沮丧，因为他们的 OpenInterpreter 似乎拥有过去学到的技能（skills），但现在无法再使用。
   - 他们推测这个问题可能源于 **OpenInterpreter** 类默认将 `import_skills=False`。
- **掌握 OpenInterpreter 技能的困难**：该成员分享了他们的挣扎，表示花费了大量时间试图弄清楚为什么他们的技能无法运行。
   - 他们的感性回应凸显了与这一限制持续进行的博弈。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1333692237187387393)** (2 messages): 

> `API base 功能，源码修改` 


- **API Base 可能无法正常工作**：一位成员对 **API base** 在当前环境下能否有效工作表示怀疑。
   - 该建议暗示集成内部存在需要进一步调查的潜在问题。
- **必须修改源码**：一位成员建议对**源码 (source code)** 进行更改，以解决软件中存在的问题。
   - 这表明需要进行深层的技术调整，而非表面修复。


  

---

### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/1333677479138824202)** (2 messages): 

> `System prompts injection, Weights and biases for tracing` 


- **理解 System Prompts 注入**：System prompts 是通过标准 metaprompt 注入的，其函数定义在 Gorilla 仓库的 [model_handler](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py) 中。
   - GitHub 页面详细介绍了用于 Function Calls 的 LLM 的实现和评估，包括仓库的可视化展示。
- **使用 Weights and Biases 增强追踪**：一位成员分享了一个专业技巧，建议在模型评估期间使用 **Weights and Biases** 以获得更好的可追溯性。
   - 这种方法允许 **检查轨迹 (trajectories)**，提供原本可能无法获取的更深层次的见解。



**提到的链接**：<a href="https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py">gorilla/berkeley-function-call-leaderboard/bfcl/model_handler/constant.py at main · ShishirPatil/gorilla</a>：Gorilla：训练和评估用于 Function Calls (Tool Calls) 的 LLM - ShishirPatil/gorilla

  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1333723117922680874)** (1 messages): 

> `GitHub PR for DSPy, Poetry lock issue` 


- **修复 poetry lock 的 PR 已开启**：一个公开的 [Pull Request](https://github.com/stanfordnlp/dspy/pull/6755) 已经提交，用于 **修复 poetry lock**，解决了 issue **#6644**。
   - 社区对该 PR 很快被合并持乐观态度，因为它旨在解决一个关键的 **dependency issue (依赖问题)**。
- **社区讨论该 PR 的重要性**：成员们表示，解决 poetry lock 问题对于维持项目的稳定性至关重要，特别是对于后续的开发活动。
   - 一位成员强调，该 PR 针对的是 DSPy 框架用户面临的 **持续挑战**。



**提到的链接**：<a href="https://github.com/stanfordnlp/dspy/pull/6755">Fix poetry lock by chenmoneygithub · Pull Request #6755 · stanfordnlp/dspy</a>：resolve #6644

  

---


### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1333654777954828309)** (1 messages): 

> `DeepSeek Performance, Cost Comparison with ChatGPT, Live Workshop, Real App Building` 


- **DeepSeek 以极低的成本超越 ChatGPT**：新的开源模型 **DeepSeek** 在 Benchmark 中表现优于 **ChatGPT** 和 **Claude**，而成本却便宜 **20-30 倍**。
   - 这一发现引发了对潜在市场颠覆的讨论，科技巨头也对这一新兴竞争对手表示担忧。
- **参加 DeepSeek 直播工作坊**：一场免费工作坊定于 **1 月 30 日星期四晚上 9:00 (IST)** 举行，届时将展示 DeepSeek 与 ChatGPT 的实时性能对比。
   - 参与者还将见证实时应用构建和详细的成本分析，展示公司如何节省数千美元。
- **实时测试 DeepSeek**：工作坊将进行现场测试，对比 DeepSeek 和 ChatGPT 在推理、编程和数学挑战方面的表现。
   - 与会者可以期待亲眼目睹 DeepSeek 的能力和优势演示。
- **使用 DeepSeek 构建应用**：工作坊承诺提供动手实践体验，允许与会者现场构建他们的第一个 **DeepSeek-powered** 应用程序。
   - 该环节旨在为开发者和技术爱好者简化 V3 和 R1 模型的集成流程。



**提到的链接**：<a href="https://lu.ma/ael5tq70?tk=23oh65">What&#x27;s the hype about DeepSeek?🐬 · Zoom · Luma</a>：AI 世界陷入疯狂！来自中国的新开源模型在 Benchmark 中表现优于 ChatGPT 和 Claude，且成本便宜 20-30 倍。这是否是……

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1333837563517075539)** (1 messages): 

> `FOSDEM 2025, Open-source collaboration` 


- **Mozilla 赞助 FOSDEM 2025**：Mozilla **自豪地赞助**了将于 **2月1日和2日**在布鲁塞尔举行的 [FOSDEM 2025](https://fosdem.org/2025/)，这是一场面向开发者的精彩且**免费**的盛会。
   - 与会者可以在这场全球盛会中**学习**并与**志同道合的朋友**建立联系。
- **加入 FOSDEM 的 Discord 协调频道**：如果您计划参加，请查看我们的 [Discord 协调线程](https://discord.com/channels/1089876418936180786/1303397923790524467) 以获取更多信息并进行联系。
   - Mozilla 欢迎所有 Open-source 爱好者在活动期间见面并协作。


  

---


---


{% else %}


> 完整的各频道详细分解内容已针对邮件进行截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}