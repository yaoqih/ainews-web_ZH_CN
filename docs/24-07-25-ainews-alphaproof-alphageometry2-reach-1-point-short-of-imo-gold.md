---
companies:
- google-deepmind
- meta-ai-fair
- mistral-ai
date: '2024-07-26T01:15:56.829913Z'
description: '“**搜索+验证器**”（Search+Verifier）突显了 2024 年数学奥林匹克竞赛期间神经符号人工智能（neurosymbolic
  AI）的进展。**Google DeepMind** 结合 **AlphaProof** 和 **AlphaGeometry 2** 解决了六道国际数学奥林匹克竞赛（IMO）题目中的四道。其中，AlphaProof
  是一个采用 AlphaZero 方法微调的 **Gemini** 模型，而 AlphaGeometry 2 则是在显著增加的合成数据上训练而成，并引入了一种新颖的知识共享机制。尽管成果斐然，人类评委指出
  AI 耗费的时间远超人类参赛者。


  与此同时，**Meta AI** 发布了拥有 4050 亿参数的 **Llama 3.1** 及其较小变体；**Mistral AI** 则推出了拥有 1230
  亿参数和 128k 上下文窗口的 **Mistral Large 2**，其在编程任务和多语言基准测试中的表现优于 Llama 3.1。这标志着 AI 在数学推理、模型扩展及多语言能力方面取得了重大进展。'
id: f7083578-ee7e-4149-ba26-dd24dc859768
models:
- gemini
- alphageometry-2
- alphaproof
- llama-3-1-405b
- llama-3-70b
- llama-3-8b
- mistral-large-2
original_slug: ainews-alphaproof-alphageometry2-almost-reach-imo
people:
- tim-gowers
- guillaume-lample
- osanseviero
title: AlphaProof + AlphaGeometry2 距离 IMO 金牌仅差 1 分。
topics:
- neurosymbolic-ai
- mathematical-reasoning
- synthetic-data
- knowledge-sharing
- model-fine-tuning
- alpha-zero
- multilinguality
- context-windows
- model-scaling
- benchmarking
- performance-comparison
---

<!-- buttondown-editor-mode: plaintext -->**Search+Verifier is all you need.**

> 2024年7月24日至7月25日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 服务器（**474** 个频道和 **4280** 条消息）。预计节省阅读时间（以 200wpm 计算）：**467 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

对于神经符号 AI (neurosymbolic AI) 来说，这是一个丰收的月份。当人类齐聚 2024 年夏季奥运会时，AI 在数学奥林匹克竞赛中也取得了巨大进步。本月初，[Numina](https://x.com/_lewtun/status/1811426187132166588) 赢得了首届 AIMO 进步奖，解决了 29/50 道奥数级别的私有集题目。

虽然 [美国队的 6 名青少年](https://maa.org/news/usa-first-at-imo/) 赢得了第 65 届国际数学奥林匹克竞赛 (IMO)，夺回了中国的桂冠，但 [Google DeepMind 宣布](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level) 他们结合了 **AlphaProof** 和新版 V2 [AlphaGeometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/) 的新组合解决了 IMO 六道题目中的四道（包括在 19 秒内 [解决了第四题](https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/imo-2024-solutions/P4/index.html)）。人类评委（包括 IMO 题目选拔委员会主席）给出了 42 分满分中的 28 分，距离金牌分数线仅差 1 分。

 
![image.png](https://assets.buttondown.email/images/feb994bf-9173-45d6-8e11-7562d3243903.png?w=960&fit=max)
 

**AlphaProof** 是一个经过微调的 Gemini 模型，结合了 AlphaZero ([论文](https://arxiv.org/abs/1712.01815))，它可以在 Lean 中证明数学陈述，并使用 AlphaZero 风格的方法寻找解决方案：
 
![image.png](https://assets.buttondown.email/images/9b361cd6-722b-4bd4-abdd-da67f639535e.png?w=960&fit=max)
 

**AlphaGeometry 2** 是一个神经符号混合系统，其语言模型基于 Gemini，并且 **在比其前身多一个数量级的合成数据上从头开始训练**。[它] 采用的符号引擎比其前身快两个数量级。当面对新问题时，会使用一种 **新颖的知识共享机制**，从而实现不同搜索树的高级组合，以应对更复杂的问题。在今年的比赛之前，AlphaGeometry 2 可以解决过去 25 年中 83% 的历史 IMO 几何问题，而其前身的解决率为 53%。 

然而，并非一切都尽如人意：IMO 人类评委之一 [Tim Gowers](https://x.com/wtgowers/status/1816509808876597264) 指出：

> 主要的限制在于，该程序需要比人类选手长得多的时间——对于 **某些题目超过了 60 小时**——当然，处理速度也比可怜的人类大脑快得多。如果允许人类选手在每道题上花费那样的时间，他们无疑会获得更高的分数。 

这与 [2022 年 OpenAI 在 Lean 证明器上的工作](https://x.com/Ji_Ha_Kim/status/1816527854655754566) 类似。

为什么 AI 既能解决 AIMO 问题，却又无法解决 9.11 > 9.9？关于“参差不齐的智能 (Jagged Intelligence)”有 [一些](https://x.com/karpathy/status/1816531576228053133) [思考](https://x.com/BlancheMinerva/status/1813955036277526691)，这归结于无处不在的泛化 (generalization) 问题。

尽管如此，对于关于 IMO 中 AI 表现的 [预测市场](https://x.com/prerationalist/status/1816504073115353116) 和 [私人赌注](https://x.com/esyudkowsky/status/1816511787560546465?s=46) 来说，这依然是重大的一天。

---


{% if medium == 'web' %}


**Table of Contents**

[TOC] 

{% else %}

**Table of Contents** 和 **Channel Summaries** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有总结均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**Llama 3.1 与 Mistral Large 2 发布**

- **模型规格**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816135838448972240) 宣布了 Meta 的 Llama 3.1（包含 **405B 参数模型**）以及 Mistral AI 的 Mistral Large 2（包含 **123B 参数**），两者均具备 **128k 上下文窗口**。Llama 3.1 还包括较小的 8B 和 70B 版本。

- **性能对比**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816135842764841009) 分享指出，Mistral Large 2 在 HumanEval 和 MultiPL-E 等 **编程任务上优于 Llama 3.1 405B**，而 Llama 3.1 405B 在 **数学方面表现更出色**。

- **多语言能力**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816135846254239853) 强调了 Mistral Large 2 在 **Multilingual MMLU** 上的强劲表现，显著超越了 Llama 3.1 70B 基础版。

- **许可与可用性**：[@osanseviero](https://twitter.com/osanseviero/status/1816035142462271539) 注意到 Llama 3.1 拥有 **更宽松的许可证**，允许利用其输出进行训练。正如 [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816135851530973265) 所述，Mistral Large 2 则在 **研究许可证** 下提供，仅限非商业用途。

- **部署选项**：[@abacaj](https://twitter.com/abacaj/status/1816213813449912690) 分享称，可以通过 **Together API 和 Fireworks** 访问 Llama 3.1。根据 [@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816136802299961404) 的说法，Mistral Large 2 可以在 **Le Chat** 上免费测试。

**开源 AI 与行业影响**

- **生态系统增长**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1816036185799619066) 强调了 **开源 AI 的快速进步**，目前的模型在性能上已足以与闭源替代方案相媲美。

- **计算需求**：[@HamelHusain](https://twitter.com/HamelHusain/status/1816144916764058044) 提到，在本地运行 Llama 3.1 405B 需要 **大量的硬件资源**，例如 8xH100 GPU。

**AI 开发与研究**

- **训练创新**：[@GuillaumeLample](https://twitter.com/GuillaumeLample/status/1816135838448972240) 透露，Llama 3.1 在训练过程中使用了 **大量的合成数据 (synthetic data)**。

- **评估挑战**：[@maximelabonne](https://twitter.com/maximelabonne/status/1816067644040118512) 讨论了对 **标准化基准测试** 的需求，并强调了当前评估方法的局限性。

- **新兴研究领域**：[@LangChainAI](https://twitter.com/LangChainAI/status/1816126876618047741) 和 [@llama_index](https://twitter.com/llama_index/status/1816195731826565586) 分别分享了在 **few-shot prompting** 和 **结构化提取 (structured extraction)** 方面的持续工作。

**行业趋势与观察**

- **模型生命周期**：[@far__el](https://twitter.com/far__el/status/1816152435112464844) 创造了“**智能毁灭周期 (Intelligence Destruction Cycle)**”一词，用来描述 AI 模型快速过时的现象。

- **实施挑战**：[@nptacek](https://twitter.com/nptacek/status/1816179089348427839) 强调了在模型能力之外，在生产环境中 **部署 AI 系统所面临的复杂性**。

- **伦理考量**：[@ylecun](https://twitter.com/ylecun/status/1816132491637375449) 参与了关于 **AI 安全** 以及大型语言模型对社会影响的持续讨论。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1：开源 AI 模型挑战封闭平台**

- **Anthropic Claude 随时可能封禁你。** ([评分: 84, 评论: 44](https://reddit.com//r/LocalLLaMA/comments/1eaw160/anthropic_claude_could_block_you_whenever_they/))：据报道，Anthropic 的 **Claude AI** 在没有明显原因的情况下 **封禁了一名用户**，这凸显了在其服务条款下可能存在的任意账号限制。作为回应，该用户正将所有任务 **转向 Meta 的开源 Llama 3.1 70B 模型**，强调了对可访问、不受限的 AI 模型的需求。
  - 用户对 **开源模型** 赶上专有模型表示感激，许多人将 **可靠性问题** 和 **任意的账号限制** 作为从 Claude 和 ChatGPT 等闭源 AI 平台转向开源的原因。
  - 几位用户报告称在没有解释的情况下被 **Claude 封禁**，通常是因为使用了 **VPN** 或在创建账号后几分钟内被封。关于账号停用的透明度和沟通缺乏是普遍的抱怨。
  - 讨论强调了 **开源 AI** 的优势，包括 **数据隐私**、**定制化** 以及对公司控制的 **独立性**。一些用户提到在工作流中切换到了 **Mixtral 8x22B** 和 **Llama 3.1 70B** 等模型。

- **随着最新一轮的发布，业界显然正在转向开源模型** ([Score: 196, Comments: 96](https://reddit.com//r/LocalLLaMA/comments/1ebhx80/with_the_latest_round_of_releases_it_seems_clear/))：AI 行业正在向 **open models** 转型。**Meta** 发布了 **Llama 3** 和 **Llama 3.1**（包括 **405B** 版本），而 **Mistral** 也提供了其最新旗舰模型 **Mistral Large 2** 的下载。**Google** 凭借 **Gemma 2** 进入了开源模型领域，**Microsoft** 继续在 Free Software 许可证下发布高质量的小型模型，**Yi-34B** 已转为 **Apache license**。这标志着自 2023 年底以来（当时似乎有可能放弃开源发布）的重大转变。这一趋势表明，尽管 **Anthropic** 即将发布 **Claude 3.5 Opus**，但像 **OpenAI** 这样仅提供封闭模型的厂商可能会面临来自快速进步的开源模型日益激烈的竞争。
  - **Apple**、**Nvidia**、**AMD**、**Intel**、**X.ai**、**Amazon** 和其他科技巨头是 AI 发展中潜在的“睡狮”。**Amazon** 已向 **Anthropic** 投资 **40 亿美元**，而据报道 **X.ai** 正在开发 **Grok 3**，这是一个融合了图像、视频和音频的 **multimodal** 模型。
  - 向开源模型的转变是由广泛测试和 R&D 的需求驱动的。开源社区提供了宝贵的见解、用例和问题解决方案，在公司和开发者之间建立了共生关系。这种方法在推进 AI 技术方面可能比封闭方法更有效。
  - 尽管开源模型进步神速，但一些用户对 **Transformer** 架构优化的潜在边际收益递减表示担忧。然而，也有人认为进步仍然是指数级的，并引用了 **Llama 3.1 8B** 超越早期大得多的模型（如 **175 billion** 参数的 **GPT-3.5**）作为例子。


**Theme 2. 专业 AI 能力的突破**

- **[DeepSeek-Coder-V2-0724 今日发布，在 aider 排行榜位列第二](https://platform.deepseek.com/api-docs/updates/#version-2024-07-24)** ([Score: 87, Comments: 15](https://reddit.com//r/LocalLLaMA/comments/1ebj49h/deepseekcoderv20724_released_today_2nd_place_in/))：DeepSeek 发布了 **DeepSeek-Coder-V2-0724**，该模型在编程助手 [aider leaderboard](https://github.com/paul-gauthier/aider/blob/main/docs/benchmarks.md) 中获得了 **第 2 名**。新版本在编程任务中表现出更强的性能，使其成为 AI 驱动的编程工具领域中的有力竞争者。
  - 用户赞赏 **DeepSeek 的频繁更新**和性能提升，一些人表示希望其他模型也能有类似的快速迭代，例如 **“下个月发布 Llama-3.2，再下个月发布 3.3”**。
  - DeepSeek-Coder-V2-0724 的 **API** 被描述为 **“极度廉价”**，并提供 **tools+json** 功能。然而，一些用户反映，尽管提示词要求不要这样做，模型仍会生成完整的代码块。
  - 社区对该模型在 **Hugging Face** 上的可用性很感兴趣，开发者指出权重发布可能需要一些时间，类似于之前的版本 (**Deepseek-V2-0628**)。

- **介绍 InternLM-Step-Prover：在 MiniF2F、Proofnet 和 Putnam 基准测试中达到 SOTA 的数学证明器。** ([Score: 68, Comments: 8](https://reddit.com//r/LocalLLaMA/comments/1ebj88o/introducing_internlmstepprover_a_sota_math_prover/))：**InternLM-Step-Prover** 在包括 **MiniF2F**、**Proofnet** 和 **Putnam** 在内的数学证明基准测试中实现了 state-of-the-art 的性能，解决了 MiniF2F 中的 **3 道 IMO 题目**，其中包括一道从未被 ATP 解决过的题目 (**IMO1983P6**)。该模型及其训练数据集（包含 **Lean-Github** 数据）已开源，可在 [Hugging Face](https://huggingface.co/internlm/internlm2-step-prover) 和 [GitHub](https://github.com/InternLM/InternLM-Math) 上获取，完整研究论文可在 [arXiv](https://arxiv.org/abs/2407.17227) 查阅。
  - 讨论强调了定义 AI 智能的**标准在不断变化**，用户注意到**证明数学定理**曾被认为是衡量真正智能的标准，而现在 LLM 已经可以实现。这种转变反映了 **Turing test** 作为标准被放弃的过程。
  - 一位用户指出，根据 **2010 年之前的定义**，当前的 LLM 将被视为具有智能，而最近的定义使“智能”一词几乎变得毫无意义。**ARC** (Abstract Reasoning Corpus) 分数的快速进步被引用为一个例子。
  - 一些评论认为，不断重新定义 AI 智能可能是由于知识分子对被机器超越的**恐惧**，导致他们否认并试图推迟承认 AI 的能力。


**Theme 3. 无审查 AI 模型与伦理考量**

- **Mistral Nemo 是未经过滤的** ([Score: 131, Comments: 40](https://reddit.com//r/LocalLLaMA/comments/1eawphb/mistral_nemo_is_uncensored/)): **Mistral Nemo** 是一款高性能且未经过滤的模型，在 [UGI 排行榜](https://huggingface.co/spaces/DontPlanToEnd/UGI-Leaderboard)上优于其他 **~13b 模型**，其 instruct 版本比基础模型更少受到过滤。尽管基准测试有限，Mistral 的过往记录表明它将与更大的模型竞争，Cognitive Computations 已经发布了 [Dolphin 微调版](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b)，这可能会使其更加未经过滤。
  - **Mistral Nemo 12b** 被誉为该尺寸类别中最好的模型，用户报告即使面对“棘手”的提示词也没有拒绝回答。然而，由于其 **12b 尺寸**，它仍然表现出一些局限性，包括常见的 GPT-isms 以及处理复杂指令的困难。
  - 用户将 **Mistral Nemo 12b** 与更大的模型进行了有利的对比，将其描述为“**Gemma 2 27b lite**”版本。它在角色扮演场景中表现良好，即使在量化（Q8_0）的情况下也能保持连贯性和角色追踪。
  - 该模型被认为非常“开放”，在 **temperature 为 0.3** 时会产生狂野的结果。它现在已有 **GGUF 格式**，兼容 **llama.cpp**，使得硬件有限的用户也可以使用。


- **[多模态 Llama 3 将不会在欧盟提供，我们需要感谢这家伙。](https://i.redd.it/rg8fto0dyfed1.png)** ([Score: 164, Comments: 78](https://reddit.com//r/LocalLLaMA/comments/1eaxs62/multimodal_llama_3_will_not_be_available_in_the/)): 该帖子批评了**欧盟内部市场专员** **Thierry Breton**，因为他可能限制了**多模态 Llama 3** 在欧盟的发布。作者认为 Breton 的行为（包括一条关于 AI 监管的推文）可能导致 Meta 不在欧盟提供多模态版本的 Llama 3，类似于 **GPT-4V** 目前在该地区无法使用的情况。
  - 用户讨论了 **EU 限制**的实际影响，指出个人仍然可以通过 **VPN** 或**自托管**来访问模型。然而，**欧盟企业**在商业化使用这些模型时可能会面临法律挑战，从而可能导致“**AI 殖民地**”的局面。
  - 值得注意的是，**Mark Zuckerberg** 讽刺地成为了开放 AI 访问的“救星”，这与 **Sam Altman** 此前限制开源模型的努力形成了鲜明对比。**德国**的用户报告称，他们使用 **LM Studio** 成功下载了 **Llama 3.1 模型**。
  - 批评指向了 **Thierry Breton** 和**欧盟对 AI 监管的方法**，一些人称其为“功能失调”，并可能导致欧盟在 AI 发展中落后。用户质疑封锁访问基于欧洲数据训练的模型的有效性。

## 全球 AI Reddit 综述

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity


**AI 模型发布与基准测试**

- **Llama 405B 达到 SOTA 性能**：在 /r/singularity 中，一则[帖子讨论了 Llama 405B 的成功](https://www.reddit.com/r/singularity/comments/1ebg730/llama_405bs_success_proves_that_openai_has/)如何挑战了 OpenAI 拥有专有技术的观念，即在没有使用新颖方法的情况下实现了相当的性能。

- **“AI Explained”频道的 Simple Bench 结果**：/r/singularity 上分享了 [Llama 405B 与其他模型](https://www.reddit.com/r/singularity/comments/1eb9iix/ai_explained_channels_private_100_question/)在一个名为 “Simple Bench” 的私有 100 题基准测试中的对比结果。

- **开源模型超越 GPT-4**：/r/singularity 报道了[第二个超越 GPT-4 的开源模型](https://www.reddit.com/r/singularity/comments/1eb92gj/sir_a_second_open_source_model_better_than_gpt4o/)，突显了公开可用 AI 的快速进步。

- **Mistral Large 2 发布**：据 /r/singularity 报道，Mistral AI [推出了 Mistral Large 2](https://www.reddit.com/r/singularity/comments/1eb5fpt/mistral_announces_mistral_large_2/)，这是其系列模型中的新成员。

**AI 应用与改进**

- **Udio 1.5 音频质量增强**：Udio [发布了 1.5 版本](https://www.reddit.com/r/singularity/comments/1ebc6ah/udio_introduces_udio_15_with_significantly/)，显著提升了音频质量，该消息分享于 /r/singularity。

**AI 生成挑战**

- **Stable Diffusion 提示词难题**：/r/StableDiffusion 上的一篇幽默帖子展示了[在生成特定内容时](https://www.reddit.com/r/StableDiffusion/comments/1ebnt6s/the_struggle_is_real/)排除不想要元素的挑战，特别是在角色生成方面。

  - 评论建议在正面提示词中使用 `rating_safe`，在负面提示词中使用 `rating_questionable, rating_explicit` 以获得更好的控制。
  - 讨论涉及模型偏见、打标系统以及精细提示词工程（prompt engineering）的重要性。


---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

**1. AI 模型发布与基准测试**

- **Mistral Large 2 对标 Llama 3.1**：Mistral AI 发布了 **Mistral Large 2**，这是一个拥有 1230 亿参数的模型，具有 128k 上下文窗口，在多语言基准测试中平均领先 **Llama 3.1 70B** 等竞争对手 6.3%。
   - 该模型在 **代码生成 (code generation)**、**数学**方面表现出色，并支持多种语言，专为高效的单节点推理 (single-node inference) 而设计。此次发布凸显了开源 AI 模型在与闭源产品竞争中的快速进步。
- **DeepMind 的 AlphaProof 在 IMO 中获得银牌**：Google DeepMind 宣布其 **AlphaProof** 系统结合 **AlphaGeometry 2**，在[国际数学奥林匹克竞赛 (International Mathematical Olympiad)](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 中达到了银牌水平，解决了 6 道题目中的 4 道。
   - 这一突破展示了 AI 在形式推理 (formal reasoning) 和数学方面日益增强的能力，尽管它比人类选手花费了更多时间。这一成就引发了关于 AI 对数学研究和教育潜在影响的讨论。
  


**2. AI 搜索与信息检索**

- **OpenAI 发布 SearchGPT 原型**：OpenAI 宣布开始测试 [SearchGPT](https://openai.com/index/searchgpt-prototype/)，这是一款全新的 AI 搜索功能，旨在提供快速、相关的答案并带有清晰的来源引用，初期涉及 10,000 名用户。
   - 此举标志着 OpenAI 进入搜索市场，可能对传统搜索引擎发起挑战。社区对此表达了兴奋与怀疑并存的态度，并讨论了其对 Perplexity 等现有 AI 驱动搜索工具的影响。
- **Reddit 与 Google 的独家协议引发关注**：Reddit 实施了一项政策，禁止除 Google 以外的大多数搜索引擎索引其内容，这与两家公司之间每年 **6000 万美元** 的协议有关。
   - 这一决定引发了关于开放互联网实践和数据可访问性的争议，特别是关于其对 AI 训练数据集的影响，以及对信息检索和模型开发的更广泛影响。
  


**3. 开源 AI 与社区努力**

- **Llama 3.1 引发优化热潮**：Meta 发布 **Llama 3.1**，尤其是 405B 参数版本，促使开源社区讨论并努力优化其在各种硬件设置下的部署和微调 (fine-tuning)。
   - 开发者正在探索**量化 (quantization)**、**分布式推理 (distributed inference)** 和内存优化等技术，以高效运行这些大型模型。[Hugging Face](https://huggingface.co/meta-llama) 等平台正在促进这些模型的获取和实现。
- **AI 开发的协作工具**：支持协作式 AI 开发的新工具和库正在涌现，例如用于管理堆叠拉取请求 (stacked pull requests) 的 [stack-pr](https://github.com/modularml/stack-pr)，以及关于共享优化内核以提高 GPU 效率的讨论。
   - 这些举措突显了社区对改进 AI 项目开发工作流和资源利用率的关注。人们对点对点共享优化和缓存以利用模型训练和推理中的集体力量越来越感兴趣。
  


**4. AI 伦理与数据使用**

- **Runway AI 训练数据争议**：一次泄密显示，Runway 的 [AI 视频生成工具](https://www.404media.co/email/64056c13-be6e-46e7-8c90-b53dd30026f2/) 是在从 YouTube 抓取的内容和盗版电影上训练的，引发了关于 AI 训练中数据使用的伦理问题。
   - 这一发现引发了 AI 社区关于使用公开但可能受版权保护的内容训练 AI 模型的激烈辩论，凸显了在平衡创新与知识产权方面的持续挑战。
- **Condé Nast 对 Perplexity 采取法律行动**：Condé Nast 向 AI 搜索引擎 **Perplexity** 发出了停止侵权函，要求其停止在 AI 回答中使用来自 Condé Nast 出版物的内容。
   - 这一法律行动强调了传统媒体公司与 AI 驱动平台之间在内容使用权方面日益紧张的关系，可能为 AI 公司如何使用和引用已出版材料设定先例。
  


---

# 第一部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Discord AI 训练中的数据隐私担忧**：关于在 **GDPR** 法规下使用 **Discord 日志** 进行 AI 训练的担忧浮出水面，表明重复使用公开数据可能仍需获得许可。
   - 参与者一致认为，尽管公开消息看起来可以随意获取，但忽视隐私权可能会导致严重的违规行为。
- **Llama 3 的微调挑战**：用户报告了在微调 **Llama 3** 时遇到的 **Out-Of-Memory (OOM)** 错误和推理质量问题，强调了数据集清洗（sanitization）的必要性。
   - 建议包括切换到 instruct 模型以提高响应质量，并解决数据集中格式不一致的问题。
- **Batching 对推理速度的重要性**：参与者强调，有效地对数据进行 **batching** 可以显著提高推理速度，并指出不使用 **HF transformers** 可能会阻碍性能。
   - 讨论指出，由于 **batching** 管理不当，许多用户的平均速度仅为 **30-100 tokens/sec**，体感极慢。
- **推理过程缓慢原因解析**：一位参与者解释了 **autoregressive inference process**（自回归推理过程）如何导致响应生成变慢，因为它需要按顺序计算每个 token。
   - 这种顺序生成因其低效而受到批评，引发了对改进实时应用方法的呼吁。
- **AI 就业保障辩论升温**：关于 AI 可能导致的职位取代（特别是软件工程领域）引发了讨论，揭示了对于这些影响紧迫性的不同看法。
   - 参与者反映了对 AI 融入职场的焦虑与接受，并对立法如何应对快速演变的现状提出了质疑。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.2.28 支持 Llama 3.1**：最新版本的 LM Studio **0.2.28** 对于有效利用 **Llama 3.1** 至关重要，因为用户在旧版本中遇到了限制。
   - 升级对于获取新功能至关重要，尤其是因为 **auto-updater** 缺少此版本。
- **了解 LLaMA 的预训练数据集**：**LLaMA** 模型的预训练数据集包含 **50% 通用知识**、**25% 数学推理**、**17% 代码**和 **8% 多语言内容**，这对它的整体性能至关重要。
   - 这一数据混合的重要性通过一份 [数据混合摘要](https://scontent-mxp2-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=t6egZJ8QdI4Q7kNvgGZp7hb&_nc_ht=scontent-mxp2-1.xx&oh=00_AYBEMFYwo-Teuskev4hTM7HJNFx6-I-WOJ_lKcobVJ70AA&oe=66A804CD) 进行了分享。
- **Beta 1 面临性能问题**：用户报告 **Beta 1** 存在严重的 **CPU 飙升**，导致聊天交互过程中性能迟缓，甚至有用户遇到了崩溃。
   - 用户的普遍情绪是希望在预期的 **Beta 2** 发布之前解决这些性能瓶颈。
- **Mistral Large 模型发布**：**Mistral Large** 现已发布，其特点是采用 **imatrix** 设计进行尺寸管理，能力可扩展至 **70GB**。
   - 鼓励用户通过其 [Hugging Face 页面](https://huggingface.co/lmstudio-community/Mistral-Large-Instruct-2407-GGUF/) 尝试此模型，它承诺提供强大的性能。
- **针对 LLM 优化 GPU 配置**：讨论强调了各种 GPU 配置，特别是 **P40** 与 **RTX 3090** 等较新模型的对比，揭示了在速度和散热管理方面的显著差异。
   - 值得注意的是，用户记录了在 **P40** 上运行 **Llama 3.1** 的速度为 **3.75 tokens/s**，但散热问题需要冷却方案才能维持性能。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Llama 3.1 登场**：备受期待的 **Llama 3.1** 现已发布，增强了社区最喜爱的 AI 聊天模型。可以在 [官方博客](https://huggingface.co/blog/llama31) 探索其功能，并通过 [此链接](https://huggingface.co/meta-llama) 使用。
   - 感兴趣的用户可以参考 [GitHub 上的 Hugging Face recipes](https://github.com/huggingface/huggingface-llama-recipes) 获取实现细节。
- **Hugging Face 在中国访问受阻**：讨论强调了在中国访问 Hugging Face 的挑战，由于该网站被封锁，导致一些开发者不得不依赖 VPN 来获取模型。
   - 建议包括与中国监管机构协商以恢复访问，以及推广本地化内容。
- **Dolphin 2.9.3 模型革新 AI**：由 Eric Hartford 策划的新发布的 [Dolphin 2.9.3 Mistral Nemo 12b 模型](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) 具有 **128K 上下文** 和 **8192 序列长度**。
   - 这一增强源自 Mistral-Nemo-Base-2407 模型，有望带来性能提升。
- **开源悬赏计划蓬勃发展**：成员们分享了几个 [开源悬赏计划](https://www.finegrain.ai/bounties)，鼓励为实现各种模型做出贡献。
   - 此类计划不仅为完成的工作提供报酬，还促进了技能发展和协作。
- **使用量化 Diffusers 进行优化**：通过 Quanto 支持 **量化 Diffusers 模型** 的新功能可减少 **50% 的内存占用**，详见 [此 GitHub PR](https://github.com/huggingface/optimum-quanto/pull/255)。
   - 此外，**Orig PixArt Sigma 权重文件大小** 显著下降，从 **2.44GB 降至 587MB**，提升了模型获取和处理速度。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Hermes 2 Theta 70B 超越 Llama-3**：Nous Research 发布了 **Hermes 2 Theta 70B**，它超越了 **Llama-3 Instruct** 设定的基准，并与 **GPT-4** 的性能持平。该模型具备 **function calling** 和 **feature extraction** 等功能。
   - 此次发布反映了模型架构的重大进展，表明在多功能 AI 应用中具有竞争优势。
- **Mistral Large 2 革新 AI**：2024 年 7 月 24 日，Mistral AI 推出了 **Mistral Large 2**，拥有 **1230 亿参数** 和 **128,000 token 的上下文窗口**。该模型在 **code generation** 和 **mathematics** 方面表现出色，略胜 **Llama 3.1**。
   - 该模型的推出是 AI 应用规模化的进步，可能已接近 **GPT-4** 等领先基准的水平。
- **Reddit 的新索引政策**：Reddit 更新政策以屏蔽除 **Google** 之外的大多数搜索引擎，这引发了与 Google 达成的 **6000 万美元** 协议相关的争议。此举阻止了未经授权的索引，引发了对开放互联网实践的质疑。
   - 成员们辩论了限制数据访问的影响，表达了对快速演变的数字格局中内容可用性的担忧。
- **Condé Nast 对 Perplexity 采取法律行动**：Condé Nast 向 Perplexity 发出了 **停止侵权函 (cease-and-desist)**，要求停止从其出版物中使用内容。在 Perplexity 估值上升之际，这加剧了传统媒体与 AI 驱动的搜索引擎之间的紧张关系。
   - 这一举动反映了 AI 驱动的信息检索时代下，内容所有权和使用权的广泛问题。
- **LLaMA 3.1 受到质疑**：用户报告 **LLaMA 3.1 instruct 模型** 的结果令人失望，在知识基准测试中表现不如 **LLaMA 3.0**。讨论集中在 RoPE 对性能的影响上，认为它可能有害。
   - 成员们指出，关闭 RoPE 可能会带来更好的结果，特别是对于较小的模型，这指明了潜在的优化方向。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Modular 发布新 Git 工具 - stack-pr**：Modular 推出了一款名为 [_stack-pr_](https://github.com/modularml/stack-pr) 的新开源工具，旨在管理 GitHub 上的堆叠式 Pull Requests (PR)，目标是为开发者简化集成流程。
   - 该工具支持更小的贡献，通过在 PR 评估过程中实现更平滑的更新，从而使 **Code Reviews** 受益。
- **对 AI 应用中 Posits 的兴趣**：关于 *Posits* 在 AI 中作用的讨论显示了对 **Gosit** 和 [llvm-xposit](https://github.com/artecs-group/llvm-xposit) 等实现的兴趣，未来可能会集成到 MLIR 中。
   - 然而，成员们指出，从传统的浮点系统过渡到 Posits 可能会面临**重大挑战**。
- **开源 Mojo 矩阵乘法**：一位成员宣布开源其在 **Mojo** 中的**矩阵乘法**实现，并邀请其他人分享在各自配置下的性能基准测试。
   - 该倡议旨在促进围绕所使用的性能指标的技术讨论和协作。
- **关于 SIMD 比较的讨论**：社区参与了关于 SIMD 比较的讨论，争论是否应同时保留逐元素（element-wise）和总计比较结果，以适应各种功能。
   - 社区正努力确保 SIMD 性能保持强劲，同时不损害其与列表行为的集成，特别是针对数据库。
- **推出具有增强能力的 Llama 3.1**：Meta 发布了 Llama 3.1 模型，现在具有 **128K 上下文长度**并支持**八种语言**，推动了开放智能进步的边界。
   - 该模型提供了与领先的**闭源模型**相匹配的独特能力，扩展了潜在的 AI 应用。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity AI 计划停机警报**：Perplexity 宣布在 <t:1722060000:R> 进行 **10 分钟的计划停机**，以进行必要的**数据库维护**，从而提高系统可靠性。
   - 团队对用户在这一关键维护期间的**耐心**表示感谢。
- **Mistral Large 2 在 AI 领域取得进展**：2024 年 7 月 24 日，Mistral AI 推出了 **Mistral Large 2**，拥有 **1230 亿参数**和 **128,000 token 上下文窗口**，在多语言 MMLU 基准测试中显著优于 **Llama 3.1 70B**。
   - Mistral Large 2 表现出比竞争对手平均高出 **6.3%** 的提升，特别是在**代码生成**和**数学**方面。
- **Reddit 对搜索引擎施加限制**：Reddit 最近的举动阻止了大多数搜索引擎索引其内容，仅因一项价值 **6000 万美元**的年度协议而授予 Google 访问权限。
   - 这一决定引发了关于 AI 模型爬取和训练中**数据访问**影响的辩论。
- **Condé Nast 质疑 AI 搜索做法**：Condé Nast 已向 **Perplexity** 发出停止并终止函，指控其未经批准使用其出版物，这表明媒体与 AI 内容使用之间的紧张关系正在升级。
   - 随着 AI 工具的激增并寻求将信息变现，这一法律行动使**内容权利**的复杂性成为焦点。
- **报告 Microsoft Teams 连接器错误**：一位用户在尝试将 **Perplexity Connector** ZIP 文件上传到 **Microsoft Teams** 时遇到了**未指定的错误消息**。
   - 这引发了社区内关于成功集成经验和潜在变通方法的咨询。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Llama 405B 降价 10%**：根据 [OpenRouterAI](https://x.com/OpenRouterAI/status/1816234833896694270) 的公告，**Llama 405B** 的价格已降低 **10%**，这是市场持续竞争策略的一部分。
   - 这一趋势表明，在 AI 模型产品的激进定价策略中，用户选择存在一种过滤机制。
- **Middle-out transform 将默认关闭**：从 **8 月 1 日**开始，**middle-out transform** 将默认关闭，从其历史设置转为增强用户控制。
   - 依赖此功能的用户应参考 [文档](https://openrouter.ai/docs/transforms) 以相应地调整其请求。
- **流量激增导致数据库压力**：OpenRouter 经历了 **5 倍的流量激增**，导致在 **东部时间晚上 10:05** 进行计划内停机以进行数据库升级。
   - 据报道，升级后的服务已迅速恢复在线，但由于反复出现的数据库问题，仍存在未解决的性能担忧。
- **Llama 3.1 表现不稳定**：报告指出 **Llama 3.1** 的输出不一致，特别是在高上下文负载期间，部分回答偏离主题。
   - 用户注意到更换供应商有时会提高输出质量，这表明推理引擎的有效性可能存在问题。
- **Mistral Large 2 展示了多语言实力**：**Mistral Large 2** 在多种语言中表现出色，在包括英语、西班牙语和中文在内的语言中展现了强大的能力。
   - 这一表现使其成为多语言语言处理领域的重要竞争者。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 测试 SearchGPT 原型**：OpenAI 推出了 **SearchGPT**，这是一个旨在通过快速、相关的答案和清晰的来源引用来增强搜索能力的原型，最初向选定用户推出以获取反馈。更多信息请访问 [OpenAI 的 SearchGPT 页面](https://openai.com/index/searchgpt-prototype/)。
   - 测试期间的用户反馈对于在 **SearchGPT** 完全集成到 ChatGPT 之前进行改进至关重要。
- **Mistral 模型下载时间长**：用户报告 **Mistral Large** 模型的下载时间很长，其中一位用户指出下载耗时 **2.5 小时**，在 MacBook Pro 上达到了 **18 tk/s** 的性能。尽管下载缓慢，但 MacBook Pro M2 Max 配合 **96GB RAM** 的能力引发了对未来改进的期待。
   - 对网络升级的期待显而易见，一位用户计划在 12 月将速度提升至 **1 Gbps**，这对于优化下载时间至关重要。
- **用户对 GPT-4o 的表现感到沮丧**：升级到 **GPT-4o** 后，用户表示失望，指出频繁出现错误且缺乏来源引用，一位用户感叹道：*“我觉得那位睿智的朋友早已离去，只剩下一个愚蠢的双胞胎兄弟。”*
   - 对 **SearchGPT API** 的担忧表明，普通访问可能需要 **数月** 时间，用户优先考虑功能改进而非 API 细节。
- **聊天机器人记忆功能的挑战**：开发者讨论了在实现聊天机器人记忆创建、编辑和删除的 function calls 时的困难，目前的准确率约为 **60%**。为了改进记忆存储决策，清晰的指导被认为是必要的。
   - 建议包括在保存重要事件的同时保存用户偏好，同时强调记忆输入指令需要具体化。
- **OpenAI 文件上传问题**：一位用户在尝试向 OpenAI 上传 **txt 文件** 时遇到了 **400 错误**，理由是不支持的文件扩展名，并引用了 [OpenAI 文档](https://platform.openai.com/docs/assistants/tools/file-search/supported-files)。
   - 尽管遵循了使用 Python 和 **FastAPI** 上传文件的详细文档，该用户在处理与文件上传失败相关的 vector store 配置时仍面临挑战。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Video 4D 颠覆视频生成**：Stability AI 推出了 **Stable Video 4D**，这是一款开创性的 video-to-video 生成模型，可在约 **40 秒**内从单个输入视频创建**动态多视角视频**。
   - 该工具能够生成 **8 个视角下的 5 帧画面**，为追求高质量视频制作的用户增强了创作流程。
- **Stable Assistant 获得新功能**：**Stable Assistant** 现在新增了 **Inpaint** 和 **Erase** 工具，允许用户在 **3 天免费试用**期内轻松清理生成内容并进行迭代。
   - 这些更新支持对输出进行微调，满足了用户在创意工作流中对精确度的需求。
- **关于模型性能的激烈辩论**：围绕模型效率的讨论非常热烈，部分成员声称某款模型性能优于 **SDXL**，而另一些成员则注意到来自 **Kolors** 和 **Auraflow** 等模型的日益激烈的竞争。
   - 讨论强调了在模型性能格局快速变化的背景下，保持关注最新发布版本的重要性。
- **掌握 Lora 训练以获得更佳输出**：社区成员交流了关于 **Lora 训练**最佳实践的见解，重点讨论了针对不同特征应使用完整图像还是裁剪图像。
   - 这一讨论突出了构建详细训练数据集以有效提升结果的关键策略。
- **深入探讨 Inpainting 技术**：用户探索了各种 **inpainting** 方法，并建议利用 **img2img** 流程和相关的教程资源以获得最佳效果。
   - 社区强调，使用上下文丰富的 Prompt 对于将物体成功融入场景至关重要。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Flash Attention 优化 VRAM 但不减少耗时**：**Flash Attention** 有助于实现线性 VRAM 占用，特别是在推理期间，但它并不能降低时间复杂度，时间复杂度仍为平方级。一位成员观察到，在长缓存和单查询（single query）的情况下使用 Flash Attention 实际上可能会因为并行化降低而减慢性能。
   - 讨论了 **KV-Cache** 等策略的影响，其随序列长度线性增加，在不显著改变计算时间的情况下影响 VRAM。
- **关于模型提供商推理成本的辩论**：成员们认为，像 **Mistral** 这样的大规模模型推理应该免费提供，并强调了利用单层或 **MoE** 框架的效率。有人担心，由于复杂性增加，批量推理（batch inference）的低效可能会抵消 **MoE** 带来的好处。
   - 讨论涉及对 **Meta** 运营策略的极少了解，挑战了那些似乎为了优化代码行数而忽略的运营效率。
- **对 Meta 的 Scaling Laws 的审查**：用户质疑 **Meta 的 Scaling Laws** 是否受到数据叠加（data superposition）的影响，建议通过**指数函数**实现最佳数据量的非线性缩放。这引发了关于计算和理解与模型性能相关的最佳数据量的对话。
   - 提到了 **Chinchilla** 推广到**每个参数 20 个 token** 的情况，揭示了缩放感知虽然看似扭曲，但在更深层次上是合理的。
- **探索 Awesome Interpretability 仓库**：[Awesome Interpretability in Large Language Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models) GitHub 仓库是专注于 **LLM 可解释性**的研究人员的重要汇编。它是深入研究大型语言模型行为复杂性的关键资源。
   - 参与 **NDIF** 计划可以获取 **Llama3-405b** 以进行大胆的实验，参与者将获得大量的 GPU 资源和支持——这是一个记录在 [这里](https://ndif.us/405b.html) 的开展有意义研究合作的新机会。
- **在外部 API 上进行 MMLU 评估**：一位成员正在寻求关于使用反映 OpenAI 设置的外部 API 测试 **MMLU** 性能的指导，特别是关于模型评估过程中的 log_probs。提到了一个相关的 [GitHub PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2008)，该 PR 引入了一个旨在实现 API 模块化的超类。
   - 出现了关于计算模型评估所需 **VRAM** 的担忧，强调了理解 VRAM 能力对实验结果的影响。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **NCCL 重叠挑战**：一位用户在参考 [NCCL Issue #338](https://github.com/NVIDIA/nccl/issues/338) 进行训练设置时，对在反向传播（backward pass）期间实现 **计算重叠（computation overlap）** 表示担忧。他们指出，实现讲座中建议的内容比预期的要复杂得多。
   - 这突显了在训练中有效利用 NCCL 优化 GPU 工作负载时面临的持续挑战。
- **引入 Flute 矩阵乘法**：一位成员分享了 [Flute](https://github.com/HanGuo97/flute) 的仓库，该项目专注于针对查找表量化（lookup table-quantized）的 LLM 进行 **快速矩阵乘法**。这旨在提升 LLM 处理应用的性能。
   - 该工具可能有助于简化需要高效矩阵处理的模型操作，这对于大规模部署至关重要。
- **使用 CUDA 工具分析 Triton Kernels**：你可以像分析其他 **CUDA** kernel 一样，使用 [Nsight Compute](https://developer.nvidia.com/nsight-compute) 等工具对 **Triton kernels** 进行详细的性能分析（profiling）。Nsight Compute 提供了全面的分析能力来优化 GPU 吞吐量。
   - 对于旨在提高 GPU 应用性能和效率的开发者来说，这款性能分析工具必不可少。
- **FP16 执行的内存限制**：一位用户对以 **FP16** 精度运行模型时内存不足表示沮丧，这突显了开发者面临的一个常见问题。这引发了关于探索优化内存使用的替代方案的讨论。
   - 解决这一问题对于提高在内存受限环境中部署大型模型的可行性至关重要。
- **使用 BnB 探索量化技术**：另一位用户建议研究使用 **bitsandbytes (BnB)** 库的 **量化（quantization）** 技术，作为内存问题的潜在变通方案。这引发了一些困惑，部分人对量化概念本身提出了疑问。
   - 理解量化的影响对于利用模型效率至关重要，尤其是在大型语言模型中。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepMind AI 在 IMO 2024 中获得银牌**：最近的讨论集中在 Google DeepMind AI 在 [IMO 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 中获得银牌，根据 Google 的博客，它达到了“银牌标准”。
   - 怀疑者对标准的清晰度提出了质疑，认为 Google 可能影响了挑战题目以展示其 AI 的性能。
- **Runway AI 的训练数据源曝光**：一次泄露透露，Runway 的 [AI 视频生成工具](https://www.404media.co/email/64056c13-be6e-46e7-8c90-b53dd30026f2/) 是在抓取的 YouTube 内容和盗版电影上训练的，这引发了伦理担忧。
   - 这一争议引发了激烈的讨论，暗示了关于内容创作者受影响程度的激烈辩论。
- **OpenAI 凭借 SearchGPT 进入搜索市场**：OpenAI 宣布测试 [SearchGPT](https://openai.com/index/searchgpt-prototype/)，旨在提供快速回答，最初将涉及 10,000 名用户。
   - 来自此次测试的反馈预计将影响其在 ChatGPT 中的集成，引发了人们对 AI 搜索功能改进的期待。
- **现代架构书籍推荐**：在寻找关于 **Diffusion** 和 **Transformers** 的资源时，一位社区成员为机器学习课程寻求书籍推荐，强调了对更具针对性的阅读材料的需求。
   - 其中一个建议是 rasbt 的书《Building LLMs from scratch》，但成员们正在寻找更多关于现代架构的综合性著作。
- **理解 LLAMA 3.1 退火（Annealing）**：讨论集中在 **LLAMA 3.1 技术报告**上，特别是将学习率降低到 0 如何有助于在不越过最佳点的情况下进行训练。
   - 这种策略可以通过精细的预训练策略提升模型在排行榜上的表现。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **OpenAI 的 SearchGPT 原型亮相**：OpenAI 宣布推出 [SearchGPT](https://openai.com/index/searchgpt-prototype/) 原型，旨在增强现有选项之外的搜索能力，首先从选定的用户群体开始收集反馈。
   - 这一初始阶段旨在将该原型集成到 ChatGPT 进行实时运行之前收集见解。
- **AI 在国际数学奥林匹克竞赛中大放异彩**：由 Google DeepMind 开发的混合 AI 系统在国际数学奥林匹克竞赛 (IMO) 中获得了银牌水平的表现，利用 AlphaProof 和 AlphaGeometry 2 解决了 6 道题目中的 4 道。
   - 这一成就突显了 AI 在应对复杂数学挑战方面的重大进展，尽管其耗时比人类参赛者更长。
- **OpenAI 用于更安全 AI 的基于规则的奖励**：OpenAI 发布了 [基于规则的奖励 (RBRs)](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)，旨在通过对齐行为来提高 AI 安全性，而无需大量的人工数据收集。
   - 这种方法允许使用更少的手动标记示例对安全协议进行更快速的调整，从而促进更具适应性的安全模型。
- **LLM 凭借评分注释晋升为裁判**：Databricks 引入了 [评分注释 (Grading Notes)](https://www.databricks.com/blog/enhancing-llm-as-a-judge-with-grading-notes)，通过创建结构化的评估量表来提高 LLM 在裁判角色中的可靠性。
   - 这些注释的加入通过为专业评估中的 LLM 提供明确指南，增强了特定领域的应用。
- **AI 训练中的合成数据面临批评**：最近的一篇论文对 AI 训练中过度依赖合成数据表示担忧，警告称这可能导致多代之后的模型崩溃 (model collapse)。
   - 专家强调保持训练输入的多样性，以维护信息质量并减轻性能下降。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **结构化提取功能发布**：新版本在任何 **LLM 驱动的 ETL、RAG** 或 **Agent** 流水线中实现了结构化提取功能，全面支持异步和流式处理功能。
   - 用户现在可以定义一个 **Pydantic 对象**，并使用 `as_structured_llm(…)` 将其附加到他们的 LLM 上，以实现流线型实现。
- **推出用于高效数据提取的 LlamaExtract**：披露了 **LlamaExtract** 的早期预览版，这是一款用于从非结构化文档中提取结构化数据的托管服务。
   - 该服务从文档中推断出 **人工可编辑的 Schema**，允许用户自定义结构化提取的标准。
- **OpenAI 调用重复的困惑**：用户对在 `MultiStepQueryEngine` 中看到重复的 OpenAI 调用表示担忧，引发了关于 Arize 日志问题的讨论。
   - 澄清确认这些并非实际的重复调用，结构化文本提取的工作仍在继续推进。
- **RAG 聊天机器人更新计划分享**：一位用户分享了升级其早期使用 LlamaIndex 构建的 RAG 聊天机器人的计划，并为开发者提供了 [GitHub 仓库](https://github.com/wadie999/Chat-Bot) 链接。
   - 他们强调，既然 RAG 现在如此流行，他们非常渴望增强聊天机器人的功能。
- **监控 Llama Agent 的文章获得好评**：成员们讨论了一篇题为《监控 Llama Agent：利用 LlamaIndex 和 Portkey 解锁可见性》的文章，链接见 [此处](https://medium.com/ai-advances/monitoring-llama-agents-unlocking-visibility-with-llamaindex-and-portkey-c2b15cb05d40)。
   - 一位成员评论说这是一篇 **很棒的文章**，强调了它对社区的重要性。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 与 OpenAI 相比表现出色**：Cohere 提供专注于通过 API 进行自然语言处理的**语言模型解决方案 (language model solutions)**，允许开发者创建诸如对话式 Agent 和摘要生成器之类的工具。欲了解全面信息，请访问 [Cohere API 文档](https://docs.cohere.com/)。
   - 他们的定价基于使用量，无需订阅，这使其与市场上的其他竞争对手区别开来。
- **撰写研究论文的指导**：成员们讨论了撰写研究论文的技巧，强调了大学导师对学术界新人的作用。他们指出 [Cohere For AI 社区](https://cohere.com/research) 是一个可以获得协作支持的资源。
   - 该社区提供必要的指导，帮助加强新作者学术研究的早期阶段。
- **理解 Langchain 的 optional_variables**：关于 Langchain 的 ChatPromptTemplate 中 '**optional_variables**' 的澄清浮出水面，强调了其允许 Prompt 中存在非必需变量的功能。这种灵活性对于创建自适应用户查询至关重要。
   - 然而，关于它与 '**partial_variables**' 的区别也产生了困惑，后者同样提供了在 Prompt 设计中处理可选元数据的功能。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral Large 2 树立了新标杆**：据报道，[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 以 **1230 亿参数 (123 billion parameters)** 和 **128k 上下文窗口 (context window)** 超越了 4050 亿参数的模型，使其适用于长上下文应用。
   - 该模型支持多种语言和编程语言，专为高效的单节点推理而设计，引发了对其性能潜力的期待。
- **探索多 Token 预测 (Multi-token Predictions)**：成员们对**多 Token 预测**表示好奇，指出其在使字节级模型在训练期间更可行、更高效方面的潜力。
   - 成员们对数据集中可能用于指定 Token 预测的注释感到兴奋，这与相关论文中讨论的方法论一致。
- **训练数据修改策略**：讨论围绕通过**掩盖不增加价值的简单词汇**来提高训练效率展开，类似于 Microsoft Rho 论文中的概念。
   - 成员们考虑了增强训练数据的策略，例如分析困惑度点 (perplexity spots) 并使用标签增强上下文，以提升训练效果。
- **对 Mistral 发布版本的困惑**：关于 Mistral Large 与 Mistral Large 2 的发布细节存在困惑，成员们对开源状态和改进声明提出了质疑。
   - 一些人对与 Claude 3.5 等现有模型相比的相对性能指标，以及该模型最终是否会开源表示担忧。
- **使用 FSDP 和 Zero3 加载 405B 的挑战**：一位用户报告了在使用 **QLoRA** 配合 **FSDP** 或 **Zero3** 加载 **405B** 模型时遇到困难。
   - 他们对导致这些加载失败的具体问题表示不确定。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **内核共享增强 GPU 效率**：成员们讨论了在搜索最优内核后，通过**点对点 (p2p) 内核共享**来提高 GPU 效率的潜力。
   - *之前的讨论强调了 p2p 搜索和共享 tinygrad 缓存的有效性。*
- **需要支持多次反向传播 (Backpropagation)**：社区强调了在 tinygrad 中需要一种一致的方法来进行多次反向传播，以实现神经网络势能 (neural network potentials)。
   - 虽然有些人认为*为 backward 调用合并损失 (combining losses)* 就足够了，但许多人寻求一种能够为复杂的梯度计算保留计算图 (computation graph) 的解决方案。
- **随机张量生成给出重复结果**：一位用户报告了 **get_random_sum()** 由于 **TinyJit** 的输出覆盖行为而重复返回相同输出的问题。
   - 建议在重复调用之前使用 `.numpy()` 可以解决此问题，从而确保输出的唯一性。
- **NumPy 转换过程中的优化**：一位用户报告通过在张量转换方法中移除 `.to('CLANG')`，将 NumPy 转换时间从 **6 秒缩短至 3 秒**。
   - 虽然出现了关于正确性的疑问，但他们验证了生成的 NumPy 数组仍然是准确的。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Mistral-Large-Instruct-2407 提供速度优势**：Mistral-Large-Instruct-2407 (128B) 比 **405B** 模型大约小 **3 倍**，从而减少了 **inference time**（推理时间）。
   - 这种规模的缩减可能会吸引那些寻求 **efficient models**（高效模型）的用户。
- **Llama 3.1 最大输出 Token 查询**：一位成员询问了 **Llama 3.1** 的 **maximum output tokens**（最大输出 Token 数），表明社区需要更多相关信息。
   - 了解这些限制可以优化用户使用 **Llama 3.1** 的体验。
- **对过时的 Ubuntu 安装说明的担忧**：讨论中提到 **Ubuntu 的安装指南** 可能已经过时。
   - 有人指出目前的说明 **已经不再适用**。
- **为优化而微调 GPT-4o-mini**：有人提出了关于在 **Open Interpreter** 框架内微调 **GPT-4o-mini** 以获得更好性能的问题。
   - 这一讨论反映了用户对利用现有 **free fine-tuning quota**（免费微调配额）的兴趣。
- **Deepseek coder 展示了令人期待的更新**：Deepseek coder 最近的 **update** 引起了关注，并分享了极具前景的性能规格。
   - **Deepseek** 每百万 Token **14-28 美分** 的性价比被强调为一个显著优势。



---



## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Llama 3.1 接近完成测试**：成员们表示正在完成 **Llama 3.1** 补丁的最终测试，重点是在单节点上集成 **405B QLoRA**。一位参与者指出在保存如此大模型的 **checkpoints** 时遇到了困难。
   - 目前的工作反映了重大进展，但在处理更重型模型时的内存管理方面仍面临挑战。
- **探索 Llama 3/3.1 的多 GPU 生产环境挑战**：有人询问了 **Llama 3/3.1 70B** 的分布式生成问题，并指出目前的功能原生不支持此操作；成员们建议查看一个 [GitHub 仓库](https://github.com/huggingface/llm-swarm) 以寻找变通方法。此外，单 GPU 适配也存在问题，用户被引导将模型量化为 **int4**。
   - 持续的讨论表明，虽然多 GPU **inference** 支持目前不是优先级，但 **torchchat** 库正在进行相关开发。
- **Snowflake 增强了微调内存管理**：一位成员强调了一篇 [blog post](https://www.snowflake.com/engineering-blog/fine-tune-llama-single-node-snowflake/)，概述了微调 **Llama 3.1** 的内存优化，指出在 **A100** 上使用 **bfloat16** 的峰值占用为 **66GB**。他们分享说，由于缺乏 **FP8** 内核，被迫做出了这一选择。
   - 这些见解通过分享处理大型模型架构的技术，为更高效的 AI 部署奠定了基础。
- **RFC 提议升级 Transformer 模块以支持 Cross Attention**：一项 **RFC** 提案寻求修改 **TransformerDecoderLayer**，以便在多模态应用中支持 **cross attention**。由于 [pull request](https://github.com/pytorch/torchtune/pull/1224) 中详述的更改，这将对现有的自定义构建器产生重大影响。
   - 成员们被提醒需要进行更新，并强调了为了保持兼容性，这些更改具有全面性。
- **分布式生成脚本的实验**：一位用户建议，对于精通 **FSDP** 集成技术的人，可以将现有的 **generate.py** 改编为 **generate_distributed.py**。他们建议利用分布式微调方案（recipes）以实现更平滑的过渡。
   - 这种方法可以简化多 GPU 实现，并在旨在最大化分布式环境效率的过程中增强协作努力。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Mistral Large 2 树立了新的 AI 标杆**：[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 拥有 **128k 上下文窗口**，支持十几种语言，并具备 **1230 亿参数 (123 billion parameters)**，用于增强 AI 应用。
   - **单节点推理 (Single-node inference)** 能力允许在长上下文任务中实现极高的吞吐量。
- **DFT Vision Transformer 重塑图像处理**：新型 **DFT Vision Transformer** 在每个模块中采用 **傅里叶变换 (Fourier transform)、MLP 和逆傅里叶变换**，在不产生数据瓶颈的情况下提升图像质量。
   - 该架构还高效集成了 **全图像归一化层 (image-wide norm layers)**，在整个过程中保持详细信息。
- **复数成为核心**：DFT Vision Transformer 完全使用 **复数参数 (complex number parameters)** 运行，增强了网络内的 *计算动态 (computational dynamics)*。
   - 这使得它能与 **旋转位置编码 (rotary position encoding)** 有效融合，从而优化整体性能。
- **旋转位置编码提升训练速度**：切换到 **旋转位置编码 (rotary position encoding)** 显著改善了 **损失曲线的下降率**，对训练产生了积极影响。
   - 参与者发现这种增强效果非常 **令人满意**，证实了该方法的有效性。
- **精简设计提升性能**：DFT Vision Transformer 采用由等尺寸模块组成的 **直线流水线 (straight pipeline)** 结构，最后以全局平均池化 (global average pool) 和线性层 (linear layer) 结束。
   - 这确保了 **图像永远不会被下采样 (downsampled)**，在整个处理过程中保留了所有信息。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **SymbolicAgentLearner 将 RAG 与符号学习相结合**：一名成员使用 DSPy 开发了 **SymbolicAgentLearner**，它集成了 **检索增强生成 (RAG)** 和符号化技术，用于问答和引用生成。
   - **SymbolicLearningProcedure** 类支持多跳检索和自动添加引用，显著增强了信息的丰富度。
- **共享 GitHub 仓库的计划**：为了回应大家的兴趣，有成员提到计划创建一个 **新的公共 GitHub 仓库**，以便与更广泛的社区分享开发成果。
   - 目前现有的代码库仍为私有，但这一转变旨在增加可访问性和协作。
- **litellm 代理实现完美集成**：成员们报告在所有模型中使用了 **litellm 代理 (litellm proxy)**，指出通过重定向 OpenAI 的 `api_base` 与 DSPy 集成时效果 **非常出色**。
   - 该解决方案简化了模型交互，增强了 DSPy 的易用性。
- **跨模型的 Function calling 需要额外努力**：一名成员成功实现了跨多种模型的 **函数调用 (function calling)**，尽管这需要额外的变通步骤。
   - 讨论了所采用的具体方法但未详述，强调了实现跨模型功能所需的努力。
- **DSPy 处理新闻分类的新方法**：一个新实施的 **新闻分类系统** 使用 DSPy 和 OpenAI 的 **GPT-3.5-turbo**，通过思维链 (Chain of Thought) 机制将文章分类为 **“虚假”或“真实”**。
   - 该方法采用 **ColBERTv2** 进行检索，并使用 **MIPRO** 进行优化，展示了用于评估误导信息有效性的自定义 **F1 分数 (F1 score)**。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain Agents 在一致性方面面临挑战**：用户对使用开源模型的 **LangChain agents** 表示沮丧，理由是 **性能不一致** 以及工具选择不当。
   - 多位测试者在评估本地 LLMs 时报告了类似令人失望的结果。
- **社区探索 Multi Agents 功能**：一位用户寻求关于实现 **multi agents** 的指导，引发了社区对感兴趣的特定功能的讨论。
   - 这次交流引发了关于这些 agents 的潜在应用和配置的进一步问题。
- **咨询在 Database Agents 中使用 ConversationSummary**：一位用户想知道是否可以将 **ConversationSummary** 与他们自己的 **database agent** 集成，并寻求实现建议。
   - 他们对建议持开放态度，特别是如果直接使用存在挑战的话。
- **LangChain 和 Ollama 发布了一个有用的视频**：一位成员推荐了一个名为 'Fully local tool calling with Ollama' 的 **YouTube 视频**，讨论了本地 LLM 工具及其用法。
   - 该视频旨在澄清 **工具选择**，并坚持认为如果设置正确，agents 可以稳定运行；点击[此处](https://www.youtube.com/watch?v=Nfk99Fz8H9k)观看。
- **LangGraph 寻找持久化选项**：一位用户请求关于 **LangGraph persistence** 潜在增强功能的更新，希望超出现有的 **SqliteSaver** 选项。
   - 社区成员对可以改进数据处理的替代存储解决方案表现出兴趣。



---



## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **对 AI Raspberry Pi 感到兴奋**：在最近的一次交流中，一位用户对 **AI Raspberry Pi** 项目表示热烈欢迎，引发了对其细节的好奇。
   - 对更多细节的请求表明，人们对其在低成本 AI 部署中的能力和应用可能感兴趣。
- **咨询更多细节**：一位成员针对 AI Raspberry Pi 的讨论请求更多信息，表示 *这很酷，告诉我们更多*。
   - 这表明社区围绕使用 Raspberry Pi 的创新 AI 项目进行了积极参与，可能希望探索技术细节。



---


**Alignment Lab AI Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**Mozilla AI Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**DiscoResearch Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期没有活动，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道划分的详细摘要和链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1265747222717272075)** (657 条消息🔥🔥🔥): 

> - `数据隐私与 GDPR`
> - `使用 Discord 日志进行 AI 训练`
> - `BTEC 教育体系`
> - `软件工程与数据科学的价值对比`
> - `AI 对就业保障的影响` 


- **数据隐私与 GDPR**：讨论强调了对使用 Discord 日志进行 AI 训练的担忧，并强调即使是公开消息，由于隐私法（尤其是欧盟的 GDPR），重新使用也可能需要许可。
   - 大家达成共识，虽然公开数据似乎可以随意使用，但在未经同意的情况下利用这些数据可能会导致侵犯隐私权。
- **使用 Discord 日志进行 AI 训练**：围绕在 Discord 聊天日志上训练模型的道德和法律问题展开了辩论，特别是涉及可能导致隐私泄露的敏感或个人信息。
   - 与会者指出，不应将此问题琐碎化，并强调了公开数据与私有数据语境之间的区别。
- **BTEC 教育体系**：讨论了 BTEC 体系在传统教育路径中的地位，并简要概述了其在英国教育框架内的运作方式。
   - 与会者分享了关于 BTEC 体系的个人经验，透露该体系比考试更注重实践作业。
- **软件工程与数据科学的价值对比**：关于软件工程和数据科学之间职业选择的对话，对于哪个领域更具吸引力或更赚钱有不同的看法。
   - 一位参与者表达了对软件工程的偏好，同时也承认了通常与数据科学职位相关的经济利益。
- **AI 对就业保障的影响**：人们对 AI 可能取代工作（特别是在软件工程领域）表示担忧，对于这种变化的紧迫性和影响，意见不一。
   - 参与者的情绪表明，他们既接受 AI 在劳动力中的角色，又担心立法者适应这些变化的速度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/mrCgn_zIAus">GitHub Accelerator Showcase 2024 | 下一波 AI 项目</a>: 准备好深入了解下一波 AI 创新！🌐✨ 见证开源 AI 项目的实际运行。我们的 Showcase 重点展示了项目演示、令人振奋的历程...</li><li><a href="https://x.com/OpenAI/status/1816536290822881780">OpenAI (@OpenAI) 的推文</a>: 我们正在测试 SearchGPT，这是一个新 AI 搜索功能的临时原型，旨在为您提供快速、及时的答案，并附带清晰且相关的来源。我们正向一小部分用户开放以获取反馈...</li><li><a href="https://www.youtube.com/watch?v=2wsLZyvaqlE">[Blue Archive] [AI Tendou Alice] Chipi Chipi Chapa Chapa (Dubidubidu)</a>: AI 演唱工具 歌聲音色轉換模型: so-vits-svc 4.1: https://github.com/svc-develop-team/so-vits-svc 角色声音: ブルアカ 天童アリス（CV：‎田中美海）原曲: Christell - Du...</li><li><a href="https://tenor.com/view/nuh-uh-beocord-no-lol-gif-24435520">Nuh Uh Beocord GIF - Nuh Uh Beocord No - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://deepmind.google/technologies/imagen-3/">Imagen 3</a>: Imagen 3 是我们最高质量的文本生成图像模型，能够生成比我们之前的模型具有更佳细节、更丰富光影且更少干扰伪影的图像。</li><li><a href="https://github.com/pytorch/pytorch/releases/tag/v2.4.0">Release PyTorch 2.4: Python 3.12, AOTInductor freezing, libuv backend for TCPStore · pytorch/pytorch</a>: PyTorch 2.4 发行说明 亮点、追踪的回归、后向不兼容的更改、弃用、新功能、改进、错误修复、性能、文档、开发者、安全。亮点包括...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1265965637423857714)** (2 条消息): 

> - `Slack 对话模板构建`
> - `Slack 频道发布` 


- **在 Slack 对话模板上遇到困难**：一位成员表示在为 Slack 对话微调 **LLama model** 时，构建 **template** 存在困难。
   - 他们正在寻求关于哪种模板最适合此用途的指导。
- **需要有针对性的频道发布**：另一位成员指出，在特定的 **Slack channel** 中发布消息就足够了，而不需要向所有频道广播。
   - 这强调了保持讨论相关性和局限性的重要性。


  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1265748516265787533)** (104 条消息🔥🔥): 

> - `SFTTrainer 中的 Max Sequence Length`
> - `Llama 3 微调问题`
> - `微调模型的推理挑战`
> - `多轮对话数据集格式化`
> - `在网站上部署模型` 


- **理解 SFTTrainer 中的 Max Sequence Length**：一位用户询问了 `SFTTrainer` 中的 `max_seq_length`，确认它是微调期间处理的最大 token 数。
   - 另一位用户指出，他们正在使用超长 Prompt 微调 Llama 3，这可能会导致问题。
- **微调 Llama 3 模型的挑战**：用户在微调 Llama 3 时遇到了各种问题，包括 Out-Of-Memory (OOM) 错误和推理质量问题。
   - 参与者建议清理数据集格式，并使用 instruct 模型以获得更好的回答。
- **微调后 Llama 3 模型的推理问题**：一位用户在对微调后的模型进行推理时遇到了胡言乱语的情况，尽管训练过程很成功。
   - 有建议认为问题可能源于训练期间使用的数据集格式或 Prompt 模板。
- **格式化多轮对话**：一位用户就如何为多轮对话格式化数据集寻求建议，并分享了他们的数据集结构。
   - 建议包括确保数据集的整洁性，以及与预期输出的映射模板保持一致。
- **本地运行模型与 WSL 建议**：用户讨论了在本地运行模型的复杂性，特别是在 Windows 环境下，因此建议使用 WSL 以获得更好的性能。
   - 提到了安装 xformers 等包时的挑战，建议使用预构建的 wheels。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://belladoreai.github.io/llama3-tokenizer-js/example-demo/build/">llama-tokenizer-js playground</a>: 暂无描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#train-on-completions--responses-only-do-not-train-on-inputs)">Home</a>: 微调 Llama 3.1, Mistral, Phi &amp; Gemma LLM 速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 暂无描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKkfyJml3Tn?usp=sharing">Google Colab</a>: 暂无描述</li><li><a href="https://colab.research.google.com/drive/11t4njE3c4Lxl-07OD8lJSMKk">Google Colab</a>: 暂无描述</li><li><a href="https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing">Google Colab</a>: 暂无描述</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3/">Llama 3 | Model Cards and Prompt formats</a>: Llama 3 使用的 Special Tokens。一个 Prompt 应包含一条 system 消息，可以包含多条交替的 user 和 assistant 消息，并始终以最后一条 user 消息结尾，后跟...</li><li><a href="https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing">Google Colab</a>: 暂无描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1265746625138131057)** (10 条消息🔥): 

> - `推理速度对比`
> - `使用 LLM 进行任务管理`
> - `推理中的 Batching`
> - `自回归推理过程` 


- **引发推理速度辩论**：一位用户质疑为什么 **inference**（推理）明显慢于训练，指出推理速度仅为 **30-100 tokens/sec**。
   - 另一位成员反驳了这一说法，建议使用 **vllm aphrodite** 等工具可以提高推理速度。
- **Batching 对提升性能至关重要**：一位成员指出，实现更高推理速度的关键在于有效地对数据进行 **batching**。
   - 有人提到，如果不实现 batching，使用 **HF transformers** 可能会导致性能较慢。
- **自回归过程减慢推理速度**：一位成员解释说，**autoregressive**（自回归）推理过程导致性能缓慢，因为每个 token 都需要按顺序进行新的模型计算。
   - 该分析解释了模型如何逐个 token 生成响应，说明了直接推理方法的低效性。
- **关于使用 LLM 进行任务管理的咨询**：一位用户就如何利用 **LLM** 在管理算法中**分配任务**寻求建议。
   - 回复中包含了对过多消息的警告，暗示讨论中的耐心有限。


  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1265749522760597676)** (298 条消息🔥🔥): 

> - `LM Studio Updates`
> - `Model Performance`
> - `GPU vs RAM Usage`
> - `Coding Models`
> - `Local Model Limitations` 


- **LM Studio 版本更新**：多位用户讨论了 LM Studio 的最新版本，指出需要 **0.2.28** 版本才能正确支持 **Llama 3.1**，而该版本目前尚无法通过自动更新程序（auto-updater）获取。
   - 建议用户从 LM Studio 网站手动下载最新版本，以获取新功能和改进。
- **模型性能与资源占用**：对话显示，在具有不同硬件配置（例如具有足够 VRAM 的 GPU）的系统上有效运行 **Llama 3.1** 模型会极大地影响性能。
   - 用户报告了不同的性能指标，强调了确保模型加载到 GPU 显存而非 RAM 中的重要性。
- **针对有限配置的最佳本地模型**：用户讨论了适用于资源有限机器的本地语言模型推荐，例如 **Mistral** 和 **WizardLM**。
   - 提到了 **DeepSeek** 等模型，对于那些在考虑硬件限制的同时寻求编程能力的用户来说，这是可行的选择。
- **系统规格对推理速度的影响**：强调了系统规格与推理速度之间的关系，部分用户在特定硬件配置上实现了低至 **0.21 tok/s** 的速度。
   - 尽管性能数据较低，参与者仍对结果表示满意，展示了本地模型与其规格相关的能力。
- **社区参与与支持**：社区成员积极参与故障排除，并就彼此在 LM Studio 和硬件设置方面的经验提供支持。
   - 协作解决问题以及分享关于模型能力和潜在问题的见解，营造了一个支持学习和实验的环境。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/mcmahon-crying-he-was-special-wwe-vince-mcmahon-gif-13313547165599993551">Mcmahon Crying He Was Special GIF - Mcmahon Crying He was special WWE - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1ebcld5/405b_q3_k_m_running_on_am5_7950x_192gb_cl30_6000/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://github.com/lmstudio-ai/model-catalog/pull/87">update quant for llama3.1 by yagil · Pull Request #87 · lmstudio-ai/model-catalog</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1265749456393863260)** (85 messages🔥🔥): 

> - `LLaMA Model Data Mix` (LLaMA 模型数据混合)
> - `Naming Preferences in AI` (AI 中的命名偏好)
> - `Model Performance Comparisons` (模型性能对比)
> - `GPU Support in LM Studio` (LM Studio 中的 GPU 支持)
> - `Dolphin Model Issues` (Dolphin 模型问题)


- **LLaMA Model Data Mix Overview**：LLaMA 模型预训练数据集概览。据报道，LLaMA 模型的预训练数据集包含 **50% 通用知识**、**25% 数学推理**、**17% 代码**和 **8% 多语言 Token**。
   - 该信息的来源包括一份 [数据混合摘要](https://scontent-mxp2-1.xx.fbcdn.net/v/t39.2365-6/452387774_1036916434819166_4173978747091533306_n.pdf?_nc_cat=104&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=t6egZJ8QdI4Q7kNvgGZp7hb&_nc_ht=scontent-mxp2-1.xx&oh=00_AYBEMFYwo-Teuskev4hTM7HJNFx6-I-WOJ_lKcobVJ70AA&oe=66A804CD)。
- **Naming Trends in AI Responses**：关于 AI 生成的角色内容中频繁使用 **Zorvath**、**Elara** 和 **Seraphina** 等名字的讨论。
   - 有人提出一种假设，认为这种趋势可能源于某位高产作家，其作品主题深度影响了 AI 训练数据集。
- **Model Performance Comparisons**：用户对比了 **LLaMA 3.1 8B** 和 **Yi 1.5** 等模型的性能，指出 LLaMA 需要多样本（multishot）总结策略，而 Yi 1.5 处理长上下文的能力更强。
   - 此外，在小型模型中，LLaMA 在涉及 JSON 输出的任务中更受青睐。
- **GPU Support Limitations in LM Studio**：已确认 LLaMA 3.1 在 LM Studio v0.27 上不支持 GPU offloading，导致在 CPU 上的性能极慢。
   - 必须升级到 LM Studio v0.28 才能有效利用新模型并获得完整的 GPU 支持。
- **Issues with Dolphin Model**：用户报告了加载 **Dolphin 2.9.3** 模型时的问题，原因是 LM Studio 中存在不支持的功能，导致出现未知 pre-tokenizer 类型的错误。
   - 该模型在各种基于 llama.cpp 的软件中均无法正常运行，表明其在发布前可能未经过测试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://releases.lmstudio.ai/linux/x86/0.2.28/beta/1/LM_Studio-0.2.28.AppImage">未找到标题</a>：未找到描述</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf/discussions/1">cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b-gguf · &quot;llama.cpp error: &#39;error loading model vocabulary: unknown pre-tokenizer type: &#39;dolphin12b&#39;&#39;&quot;</a>：未找到描述</li><li><a href="https://huggingface.co/microsoft/Phi-3-vision-128k-instruct/discussions/24">microsoft/Phi-3-vision-128k-instruct · Phi-3V 是否应该在 llama.cpp 中提供支持？</a>：未找到描述</li><li><a href="https://embed.wattpad.com/story/372087683-the-cosmic-union-dreamcatchers">Embed - The Cosmic Union: Dreamcatchers - Wattpad</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/8650">功能请求：在 llama.cpp 中提供完善的 Llama 3.1 支持 · Issue #8650 · ggerganov/llama.cpp</a>：前提条件：我正在运行最新代码。如果可能请注明版本。我仔细阅读了 README.md。我使用了与问题相关的关键词进行搜索，以确保我创建的是...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/)** (1 messages): 

melkanea: 如果单独计算 CUDA 核心，我得到了 +5600。
  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1265796106651045999)** (144 messages🔥🔥): 

> - `各种硬件上的 ML 推理`
> - `P40 GPU 使用体验`
> - `RTX 3090 vs M3 Max 的推理对比`
> - `Apple Silicon 在 AI 方面的性能`
> - `双 GPU 配置` 


- **探索 LLM 推理的 GPU 选项**：用户讨论了运行 LLM 模型时不同 GPU 的优劣，指出与 **RTX 3090** 等较新显卡相比，**P40** 在性能和散热管理方面的局限性。
   - 使用 **4 张 P40** 时，一位用户报告在运行 Llama 3.1 70B 时达到了 **3.75 tokens/s**，而其他用户则强调了 **M3 Max** 在推理任务中的效率。
- **P40 配置的挑战与解决方案**：用户对 **P40** 的高温和散热需求表示担忧，建议使用自定义散热方案来缓解发热问题。
   - 一位用户成功实施了**自定义散热风道**，尽管最初存在过热问题，但仍使其 P40 在高负载下保持正常运行。
- **性能对比：RTX 3090 与 M3 Max**：讨论强调了 **M3 Max** 在 AI 任务中的潜力，尤其是在舒适的生态系统中，这与 **RTX 3090** 等游戏 GPU 的高功耗和高发热形成鲜明对比。
   - 用户分享了性能指标，建议如果追求更快的推理速度，双 **3090 配置** 可能是更便宜的选择，尽管其功耗可能更高。
- **用于 AI 任务的 Apple Silicon**：**M3 Max** 因其在运行 LLM 推理时的安静运行和高效能耗而受到称赞，使其成为传统 GPU 的有力替代方案。
   - 用户对 **DiffusionBee** 以及在日常任务和 AI 推理中使用 **Apple 生态系统** 的整体便捷性表示满意。
- **组合使用 GPU 的潜在问题**：讨论了同时运行 **RTX** 和 **P40** GPU 的兼容性问题，指出用户的稳定性及性能体验差异很大。
   - 一些用户确认在没有额外驱动程序问题的情况下成功运行了两种 GPU，而另一些用户则建议在集成旧硬件时保持谨慎。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1265780705066221730)** (27 messages🔥): 

> - `Beta 1 CPU 问题`
> - `渲染器崩溃报告`
> - `新 UI 反馈`
> - `模型对比`
> - `即将发布的 Beta 2` 


- **Beta 1 在 CPU 性能方面存在困难**：用户报告在 **Beta 1** 中输入一定量的聊天文本后，会出现 **CPU 占用飙升**和打字迟钝的情况。
   - 一位用户还遇到了**渲染器崩溃 (Renderer crash)**，并计划通过官方渠道进行报告。
- **对新 UI 响应速度的反馈**：一位用户评论说新 **UI** 感觉**非常流畅 (snappy)**，表明对最近的更新反应积极。
   - 多位成员对 UI 的性能表达了普遍的热情。
- **关于量化模型效率的辩论**：展开了一场关于在 24GB VRAM 的 **GPU** 上使用 **70B** 参数模型的讨论，权衡了量化模型与非量化模型的优劣。
   - 用户提出了量化可能导致**质量下降**的观点，对于像 **120B Goliath** 这样的大型量化模型的有效性存在不同看法。
- **Beta 0.2.29 的技术问题**：一位用户报告了启动 **0.2.29 版本** 时的问题，引发了关于故障排除和重新安装 **LM Studio** 的建议。
   - 另一位用户提到 **v26** 也有类似问题，但在更新到 **v27** 后得到解决，这表明可能存在与版本相关的 Bug。
- **Beta 2 发布日期公布**：随着 **Beta 2** 预计明天发布，大家的期待感在增加，该版本承诺带来**新功能和错误修复**。
   - 参与者表达了对下一个 Beta 迭代中增强功能的渴望，并讨论了可能错过上一个版本的情况。


  

---

### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1265745943052029973)** (17 messages🔥): 

> - `Linux AppImage updates`
> - `GPU offloading with ROCm`
> - `Compatibility with 7800XT`
> - `Command line for ROCm`
> - `OpenCL performance` 


- **Linux AppImage 升级至 0.2.28 运行顺畅**：一位用户从 **Linux 0.2.27** 迁移到 **0.2.28** AppImage，并确认 **Llama 3.1 模型** 在其 **7800XT** 上开箱即用。
   - 另一位删除了 0.2.27 的用户也确认了新版本的功能正常，尽管最初出现了 GPU 检测错误。
- **关于 0.2.28 的 ROCm 扩展尚存疑虑**：讨论了 **0.2.28** 是否需要 **ROCm 扩展**，一位用户提到他们在 0.2.27 中使用了脚本，但在 0.2.28 中未做任何操作。
   - 达成的共识是，0.2.27 的要求可能同样适用于最新版本，无需额外步骤。
- **在其他 GPU 上成功使用 ROCm**：据报告，**ROCm** 在 **RC6600XT** 上也能有效运行，表明其在不同型号间具有广泛的兼容性。
   - 另一位用户建议，对于有兼容性问题的用户，可以通过命令行启动 `HSA_OVERRIDE_GFX_VERSION=10.3.0 lm-studio`。
- **OpenCL 目前提供足够的性能**：一位用户指出，虽然他们无法让 ROCm 正常工作，但 **OpenCL** 的性能已足以满足其需求。
   - 他们表示在进一步尝试 ROCm 之前，将等待 **Vulkan** 的开发进展。


  

---


### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1266066404147462237)** (1 messages): 

> - `Mistral Large` 


- **Mistral Large 模型发布**：**Mistral Large** 已正式发布，采用 **imatrix** 构建以增强尺寸管理，规模可达 **70GB**。
   - 该模型承诺具有**卓越的性能**，邀请用户通过 [Hugging Face 页面](https://huggingface.co/lmstudio-community/Mistral-Large-Instruct-2407-GGUF/) 探索其功能。
- **Mistral 模型尺寸与能力**：**Mistral Large** 的 **Q4_K_M** 模型配置使其在保持**超大尺寸**的同时，仍能实现最佳性能。
   - 鼓励用户尝试这款强大的模型并享受其带来的优势。


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1265778938328645752)** (3 messages): 

> - `Using Llama 3.1`
> - `VS Code Extensions`
> - `Codestral Setup` 


- **关于使用 Llama 3.1 的指导**：成员们讨论了如何在 **Cursor** 或 **VS Code** 中使用 **Llama 3.1**，并建议可能已有用于本地 LLM 集成的扩展。
   - 一位用户发起了讨论，寻求社区的具体指导。
- **设置 VS Code 自动补全**：据分享，**Continue** 现在支持 [VS Code](https://marketplace.visualstudio.com/items?itemName=Continue.continue) 和 [JetBrains IDEs](https://plugins.jetbrains.com/plugin/22707-continue/edit) 中的标签页自动补全。
   - 鼓励成员通过 [Discord](https://discord.gg/vapESyrFmJ) 频道提供反馈和建议。
- **Codestral 设置推荐**：为了获得最佳的自动补全体验，建议使用 **Codestral**，可通过 [Mistral API](https://console.mistral.ai/) 访问。
   - 要进行设置，用户需要获取 API key 并将其集成到 `config.json` 中。



**提到的链接**：<a href="https://docs.continue.dev/features/tab-autocomplete#setting-up-with-lm-studio">Tab Autocomplete (beta) | Continue</a>：Continue 现在为 VS Code 和 JetBrains IDE 提供标签页自动补全支持。我们将在接下来的几个版本中大幅提升体验，听取反馈总是很有帮助的。如果...

  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1265758130596348077)** (1 messages): 

> - `Llama 3.1 发布` 


- **Llama 3.1 已经到来！**：备受期待的 **Llama 3.1** 现已发布，为社区最受欢迎的 AI 聊天模型带来了增强。欲了解更多详情，请查看[官方博客文章](https://huggingface.co/blog/llama31)。
   - 通过[此链接](https://huggingface.co/meta-llama)探索新功能和模型，并通过 [Hugging Quants](https://huggingface.co/hugging-quants) 深入了解社区参与情况。
- **如何利用 Llama 3.1**：感兴趣的用户可以按照 [GitHub 上的 Hugging Face recipes](https://github.com/huggingface/huggingface-llama-recipes) 中的说明，学习如何有效地实现 **Llama 3.1**。
   - 若要直接试用，请访问 [Meta-Llama 3.1 聊天模型](https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8)进行亲身体验。



**提到的链接**：<a href="https://huggingface.co/chat/models/meta-llama/Meta-Llama-3.1-405B-Instruct-FP8)">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每一个人。

  

---


### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1265758520167497769)** (421 messages🔥🔥🔥): 

> - `Hugging Face 社区讨论`
> - `模型性能对比`
> - `训练与微调 LLM`
> - `音频降噪研究`
> - `中国监管对 AI 模型的影响` 


- **Hugging Face 在中国的访问情况**：社区成员讨论了在中国访问 Hugging Face 的挑战，指出虽然网站被封锁，但一些开发者使用 VPN 来访问模型。
   - 建议包括 Hugging Face 可能需要与中国监管机构协商以恢复访问，以及关于内容本地化的讨论。
- **Llama 模型的性能问题**：用户对 Llama 3.1 与之前模型相比的性能表示担忧，一些人认为它在指令任务中的得分低于预期。
   - 一些用户表示，为了提高工作效率，他们更倾向于选择更小的模型或 API 替代方案。
- **音频处理与模型优化**：一位用户分享了一个结合了使用神经网络进行音频降噪的项目，强调了针对实时性能进行有效优化的必要性。
   - 讨论集中在尽管线性神经网络结构简单，但在音频任务中使用它们的有效性。
- **微调大语言模型 (LLM)**：几位用户讨论了微调 LLM 的各种方法，分享了代码片段以及在实现中对高效架构的需求。
   - 尤其对应用 MCTS (Monte Carlo Tree Search) 方法来提高小型 LLM 的推理能力表现出浓厚兴趣。
- **本地模型推理资源**：一位用户询问了关于设置本地实例以使用 Whisper 等模型的问题，寻求有关文档和配置的指导。
   - 建议包括查看 Hugging Face 的私有模型空间，并探索其他社区资源以设置推理 API。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebase">🧑‍🎓 如何使用 Continue | Continue</a>：在编写代码时使用 Continue 调用 LLM</li><li><a href="https://x.com/reach_vb/status/1815859311161270456">来自 Vaibhav (VB) Srivastav (@reach_vb) 的推文</a>：以下是运行 L3.1 405B 所需的 GPU VRAM 大小 - fp/bf16 需要 810 GB，fp8/int8 需要 405 GB，int4 需要 203 GB。硬件 - 4 x H/A100 到 8 x H/A100（取决于所使用的量化方式）。70B - fp/bf1...</li><li><a href="https://andrew-devportfolio.vercel.app/">未找到标题</a>：未找到描述</li><li><a href="https://open.spotify.com/track/4Gi0onoWeLXG9S75e9HfRb?si=15a385daed9e46c1">Du hast kein Herz</a>：Till Lindemann · 歌曲 · 2023</li><li><a href="https://tenor.com/view/till-lindemann-spinning-gif-1391360951872139393">Till Lindemann 旋转 GIF - Till lindemann 旋转 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/troll-face-cube-funny-gif-26291753">Troll Face Cube GIF - Troll Face Cube 搞笑 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cuh-guh-buh-gif-26372267">Cuh Guh GIF - Cuh Guh Buh - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/what-gif-21384529">What GIF - What - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/spongebob-squarepants-nickelodeon-sigh-phew-gif-5752975">松了一口气 GIF - Spongebob Squarepants Nickelodeon - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/dpowe-gif-24107728">Dpowe GIF - Dpowe - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/cat-eating-eatin-gamer-gunk-gamer-gunk-cat-monkey-cat-gamer-gif-20643451">猫吃东西 Eatin Gamer Gunk GIF - 猫吃东西 Eatin Gamer Gunk Gamer Gunk - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://youtu.be/qpENmF1eTgI">Rammstein Ausländer 音频</a>：未找到描述</li><li><a href="https://tenor.com/view/troll-lol-gta-gta-san-andreas-running-gif-25040072">Troll Lol GIF - Troll Lol Gta - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/cognitivecompai/status/1816314744007004214">来自 Cognitive Computations (@cognitivecompai) 的推文</a>：Llama 3.1 的评分比 Llama 3 低？</li><li><a href="https://dontasktoask.com/">不要问能不能问，直接问</a>：未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolLM-1.7B-Instruct">HuggingFaceTB/SmolLM-1.7B-Instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/lmsys/chatbot_arena_conversations">lmsys/chatbot_arena_conversations · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://www.tomshardware.com/pc-components/cpus/amd-delays-its-ryzen-9000-launch-due-to-unspecified-quality-issue-new-launch-in-august-chipmaker-pulls-back-all-units-shipped-globally-for-quality-checks">AMD 因未指明的质量问题推迟 Ryzen 9000 发布 —— 8 月重新发布；芯片制造商召回全球范围内发货的所有单元进行质量检查 [已更新]</a>：AMD 紧急叫停了搭载 Zen 5 架构的 Ryzen 9000 处理器。</li><li><a href="https://modelscope.cn/papers">ModelScope 魔搭社区</a>：ModelScope——汇聚各领域先进的机器学习模型，提供模型探索体验、推理、训练、部署和应用的一站式服务。在这里，共建模型开源社区，发现、学习、定制和分享心仪的模型。</li><li><a href="https://github.com/wootwootwootwoot/ComfyUI-RK-Sampler">GitHub - wootwootwootwoot/ComfyUI-RK-Sampler: 适用于 ComfyUI 的批量 Runge-Kutta 采样器</a>：适用于 ComfyUI 的批量 Runge-Kutta 采样器。通过在 GitHub 上创建账号来为 wootwootwootwoot/ComfyUI-RK-Sampler 的开发做出贡献。</li><li><a href="https://github.com/mlfoundations/open_clip">GitHub - mlfoundations/open_clip: CLIP 的开源实现</a>：CLIP 的开源实现。通过在 GitHub 上创建账号来为 mlfoundations/open_clip 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/chinese-language-blog">为中文用户推出 Hugging Face 博客：促进与中国 AI 社区的合作</a>：未找到描述</li><li><a href="https://github.com/gfx-rs/wgpu-native/wiki/Contributing#update-to-latest-wgpu-core">贡献</a>：基于 wgpu-core 的原生 WebGPU 实现。通过在 GitHub 上创建账号来为 gfx-rs/wgpu-native 的开发做出贡献。</li><li><a href="https://www.newgrounds.com/audio/listen/744021">Ai じゃ ない</a>：受 Synthfunk/Vaporwave 启发的曲目</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb">HuggingFaceFW/fineweb · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://youtu.be/kDNG4QMKgs8">Dahmer</a>：由 CDBaby 提供给 YouTube 的 Dahmer · 6snotNadietepregunto℗ 2022 Daniel Tota，发布日期：2022-04-01，由 YouTube 自动生成。</li><li><a href="https://youtu.be/M6MV1nxxGf8">100 gecs - 757 {官方音频}</a>：10,000 gecs a

现已发布：https://100gecs.lnk.to/10000gecsIDFOLLOW 100 GECS:https://twitter.com/100gecshttps://soundcloud.com/100gecshttps://www.instagram.c...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1265767925974044804)** (12 条消息🔥): 

> - `Dolphin 2.9.3 Model Release` (Dolphin 2.9.3 模型发布)
> - `AI Solves Mathematical Olympiad` (AI 解决数学奥林匹克竞赛题)
> - `K-Nearest Neighbors Algorithm` (K-Nearest Neighbors 算法)
> - `AI Job Security Discussion` (AI 职业安全讨论)


- **Dolphin 2.9.3 Mistral Nemo 发布**：新的 [Dolphin 2.9.3 Mistral Nemo 12b 模型](https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b) 由 Eric Hartford 和 Cognitive Computations 策划并训练，基于 Mistral-Nemo-Base-2407 模型进行了增强。
   - 该模型拥有 **128K context**，并在微调期间使用了 **8192 sequence length**。
- **AI 在数学奥林匹克竞赛中获得银牌**：Google DeepMind 宣布了一项突破性的 AI，能够以银牌选手的水平解决国际数学奥林匹克（IMO）问题，该系统结合了 **AlphaProof** 和改进后的 **AlphaGeometry 2**。
   - 更多细节可在其 [公告推文](https://dpmd.ai/imo-silver) 中查看，展示了 AI 在形式化推理（formal reasoning）方面的潜力。
- **K-Nearest Neighbors 概览**：[这篇文章](https://medium.com/@shivam11/k-nearest-neighbor-knn-algorithm-overview-e18fb0e42a0c) 提供了 K-Nearest Neighbors (KNN) 算法的概览，这是一种用于回归和分类的有监督机器学习技术。
   - KNN 是非参数化的，这意味着它不假设任何底层数据分布，使其在各个领域都成为一种通用的选择。
- **AI 时代的职业安全**：[Bernard Marr 讨论了与 AI 相关的职业安全](https://bernardmarr.com/what-job-is-most-safe-from-ai/)，探讨了随着技术的发展，哪些职业可能保持不受影响。
   - 他的见解反映在他广泛的著作以及在技术领域的影响力中。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://fullbound.ai">Fullbound - 招聘人员的梦想工具</a>：未找到描述</li><li><a href="https://x.com/GoogleDeepMind/status/1816498082860667086">Google DeepMind (@GoogleDeepMind) 的推文</a>：我们展示了第一个能以银牌选手水平解决国际数学奥林匹克竞赛问题的 AI。🥈 它结合了 AlphaProof（一种用于形式化推理的新突破模型）和 AlphaGeome...</li><li><a href="https://medium.com/@shivam11/k-nearest-neighbor-knn-algorithm-overview-e18fb0e42a0c">K-Nearest Neighbor (KNN) 算法概览</a>：使用有监督机器学习，K-Nearest Neighbors (KNN) 技术被用于解决回归和分类问题。这……</li><li><a href="https://bernardmarr.com/what-job-is-most-safe-from-ai/">哪种工作在 AI 时代最安全？</a>：随着人工智能不断重塑行业，了解哪些工作仍然安全至关重要。虽然 AI 对就业市场的影响不可否认，但并非所有角色的脆弱程度都相同……</li><li><a href="https://huggingface.co/cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b">cognitivecomputations/dolphin-2.9.3-mistral-nemo-12b · Hugging Face</a>：未找到描述</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI 模型在递归生成的数据上训练时会发生崩溃 - Nature</a>：分析表明，不加区分地在真实内容和生成内容上训练生成式人工智能（通常通过从互联网抓取数据完成）会导致崩溃……
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1265911019935629345)** (5 messages): 

> - `W2V2-BERT Model for Ukrainian`
> - `Next Word AutoComplete`
> - `Community Engagement` 


- **针对乌克兰语微调的 W2V2-BERT 模型**：使用 **YODAS2 数据集**（包含 **400k 样本**）在乌克兰语上微调的模型，现已在 [Hugging Face](https://huggingface.co/Yehor/w2v-bert-2.0-uk-v2) 上线。
   - 用户还可以加入 **Discord 服务器** 讨论 Data Science 和 AI 相关话题，并受邀加入 Telegram 的 **Speech Recognition Group**。
- **下一个词自动补全（Next Word AutoComplete）和短语推理模型**：一个用于 Tokenization 的新自动补全组件，基于 **240k 词短语数据模型** 构建，目前提供集成支持，详见 [此 Demo](https://wiki-phrases-tokenizer.vtempest.workers.dev)。
   - 开发者已在该模型上工作超过 **6 个月**，并鼓励通过 [GitHub](https://github.com/vtempest/wiki-phrase-tokenizer/ 'Fork me on GitHub') 进行社区反馈和协作。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wiki-phrases-tokenizer.vtempest.workers.dev">AutoComplete - Wiki Phrases Tokenizer </a>: 未找到描述</li><li><a href="https://huggingface.co/Yehor/w2v-bert-2.0-uk-v2">Yehor/w2v-bert-2.0-uk-v2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Yehor/w2v-bert-2.0-uk-v2-demo">Speech-to-Text for Ukrainian v2 - a Hugging Face Space by Yehor</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1266021037078872156)** (8 messages🔥): 

> - `Open Source Bounty Programs`
> - `Diffusion Models`
> - `Finegrain Bounty`
> - `Tinygrad Bounties` 


- **探索开源赏金计划**：一位成员提到，虽然从零开始实现项目可能并非必要，但有几个用于实现 Diffusion 模型的 [开源赏金计划](https://www.finegrain.ai/bounties) 可供选择。
   - 这些计划在促进开发者贡献的同时，也为完成的工作提供潜在报酬。
- **Finegrain 赏金宇宙欢迎贡献者**：一位参与者分享了关于 Finegrain 赏金平台的见解，该平台通过为成功合并的 Pull Request 提供报酬来鼓励贡献者。
   - 计划详情列出了赏金的各种状态，并提供了明确的参与和提交指南。
- **Tinygrad 赏金获得认可**：一位成员表示他们熟悉 Tinygrad 赏金，并指出它为社区中的其他人提供了灵感。
   - 围绕知名赏金计划的讨论确认了它们的相关性，并鼓励探索这些机会。
- **赏金计划的成功案例**：讨论显示，一些成员甚至通过参与赏金计划获得了工作机会，说明了其有效性。
   - 这突显了通过参与开源项目实现职业晋升的潜力。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.finegrain.ai/bounties">Finegrain Bounties</a>: Bounties  &lt;a href=&quot;https://github.com/finegrain-ai/refiners&quot;&gt;Refiners&lt;/a&gt; 是我们的开源 (MIT) 适配器库。潜入 Finegrain 赏金宇宙：编码、征服、变现...</li><li><a href="https://docs.google.com/spreadsheets/u/3/d/1WKHbT-7KOgjEawq5h5Ic1qUWzpfAzuD_J06N1JwOCGs/htmlview#">Bounties - Google Drive</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1266021420773806202)** (1 messages): 

> - `Quantized Diffusers`
> - `Memory Optimization`
> - `Orig PixArt Sigma Checkpoint Reduction` 


- **通过 Quanto 运行量化 Diffusers**：一项新功能允许通过 Quanto 直接运行 **量化 Diffusers 模型**，显著提升了性能。
   - 这一变化使 **内存占用减少了 50%**，详见 [此 GitHub PR](https://github.com/huggingface/optimum-quanto/pull/255)。
- **原始 PixArt Sigma Checkpoint 大小显著缩减**：**原始 PixArt Sigma Checkpoint 大小** 已从 **2.44GB 缩减至 587MB**，从而实现了更轻松的访问和更快的处理。
   - 这一优化是模型管理方面的显著增强，在上述 [GitHub PR](https://github.com/huggingface/optimum-quanto/pull/255) 中有详细说明。



**提及的链接**: <a href="https://github.com/huggingface/optimum-quanto/pull/255">feat: support diffusion models. by sayakpaul · Pull Request #255 · huggingface/optimum-quanto</a>: 这个 PR 做了什么？修复了 #252

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1265802954397454420)** (7 messages): 

> - `Labeling Platforms`
> - `Road Detection from Satellite Images`
> - `Understanding LLaVa` 


- **标注平台讨论 (Labeling Platforms Discussion)**：几位成员讨论了图像标注的替代方案，特别强调了 **Labelstudio** 和 **CVAT** 作为需要自托管 (self-hosted) 选项的潜在解决方案。
   - 推荐使用 *Labelstudio* 是因为其易用性，同时也分享了关于安装困难的警告，特别是使用 *Docker* 时。
- **卫星图像分析的挑战**：有人询问关于使用基于 Transformer 的模型从**卫星图像中检测道路**的问题，引发了社区对现有方法的投入。
   - 一位用户询问是否有特定的可用模型，表现出对实际应用的浓厚兴趣。
- **寻求关于 LLaVa 的澄清**：一名成员表示难以理解 **LLaVa** 中 **SeparatorStyle** 的概念，特别是其对各种语言骨干网络 (language backbones) 的影响。
   - 对该主题详细解释的请求突显了社区内持续的学习和好奇心。


  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1265926850123792405)** (21 messages🔥): 

> - `Embedding Model Fine-tuning`
> - `RAG System Performance`
> - `Embedding Numerical Data Challenges`
> - `Collaborative LLM Projects`
> - `Llama 3.1 with Inf2 Guides` 


- **微调 Embedding 模型以获得更好性能**：一位成员表示需要微调他们的 Embedding 模型，因为目前的性能在真实数据上表现不足，尽管在合成数据上表现尚可。
   - 他们认为微调后的模型可以显著改善结果，特别是他们计划测试更大的模型选项。
- **RAG 系统中数值数据 Embedding 的挑战**：另一位成员分享了他们使用 qdrant 向量数据库的经验，他们在 RAG 准确检索数值数据方面遇到了效率低下的问题。
   - 尽管尝试了混合搜索 (hybrid search) 技术，他们发现仅搜索文本关键词并不能在数字检索方面取得令人满意的结果。
- **征集 LLM 项目协作**：一位成员寻求志同道合的人共同协作 LLM 项目，觉得独自工作变得很无聊。
   - 这突显了社区对协作努力以分享知识和增强项目成果的渴望。
- **关于 Llama 3.1 和 Inf2 服务器的咨询**：一位用户询问是否有关于在 Inf2 服务器上使用 Llama 3.1 的可用指南，表明了对该领域资源的需求。
   - 这反映了在不同计算环境中使用先进 LLM 框架的持续兴趣。


  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1265954992137699408)** (2 messages): 

> - `Diffusion techniques in biological sequence generation`
> - `Updates on ComfyUI`
> - `MediaPipe integration`
> - `TensorRT performance`
> - `Workflow changes in ComfyUI` 


- **生物序列的 Diffusion 技术**：一位用户询问了在利用其数据点和特征生成生物序列的 Diffusion 技术中，加噪 (noise addition) 的典型过程。
   - 他们具体询问了应该在原始数据、计算特征后的数据，还是在应用 Embedding 层之后添加噪声。
- **ComfyUI 迎来重大更新**：一位用户分享了他们在 ComfyUI 中实现新功能的经验，其中包括在社区支持下功能完备的 video2video 模式。
   - 他们提到为了改进应用程序付出了巨大努力，并且由于这些变化，旧的工作流 (workflows) 现在已经失效。
- **MediaPipe 替换 Insightface**：关于 ComfyUI 的更新强调了从 Insightface 向 MediaPipe 的转变，后者因其 Apache-2.0 许可证而更受青睐。
   - 这种转变相比之前 InsightFace 模型的非商业许可证，为用户提供了更多的灵活性。
- **TensorRT 支持效果参差不齐**：用户分享了他们尝试利用 TensorRT 支持的经验，但报告称在他们的硬件上收益微乎其微，或者是因为缺乏经验。
   - 尽管如此，他们成功优化并简化了其他功能，在 ComfyUI 的框架内实现了“实时”速度。



**提到的链接**：<a href="https://www.reddit.com/r/comfyui/s/9UEq6AFPYv">Reddit - Dive into anything</a>：未找到描述

  

---



### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/)** (1 messages): 

jsarnecki: https://github.com/mlfoundations/MINT-1T
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1265753334305001503)** (9 messages🔥): 

> - `Hermes 2 Theta 70B`
> - `Mistral Large 2`
> - `Reddit's indexing policy change`
> - `Condé Nast legal action`
> - `Wiki Phrases Tokenizer` 


- **Hermes 2 Theta 70B 超越 Llama-3**：Nous Research 宣布发布 **Hermes 2 Theta 70B**，超越了 **Llama-3 Instruct** 设定的基准测试，并实现了与 **GPT-4** 相当的性能。
   - 该模型引入了 **function calling** 和 **feature extraction** 等功能，增强了 AI 应用的多功能性。
- **Mistral Large 2 变革 AI**：2024 年 7 月 24 日，Mistral AI 发布了 **Mistral Large 2**，拥有 **1230 亿参数**和 **128,000-token context window**。
   - 该模型在 **code generation** 和数学方面表现卓越，超越了 **Llama 3.1**，几乎与 **GPT-4** 持平。
- **Reddit 屏蔽未付费搜索引擎**：Reddit 更新政策以屏蔽除 **Google** 之外的大多数搜索引擎，此举引发了争议，据称与 Google 达成的 **6000 万美元**交易有关。
   - 政策变更防止了未经授权的索引，引发了对未来开放互联网访问的担忧。
- **Condé Nast 对 Perplexity 采取法律行动**：Condé Nast 已向 Perplexity 发出**停止侵权函 (cease-and-desist)**，要求其停止在 AI 回复中使用其出版物的内容。
   - 在 Perplexity 获得高额估值后，这一法律行动加剧了传统媒体与 AI 驱动的搜索引擎之间的紧张关系。
- **Next Word AutoComplete 与短语推理模型**：推出了一款用于 Tokenization 的新**自动补全组件**，使用 **240k 词短语数据模型**，并提供 **LIVE DEMO**。
   - 该项目已活跃开发超过 **6 个月**，欢迎通过 [GitHub](https://github.com/vtempest/wiki-phrase-tokenizer/ 'Fork me on GitHub') 进行集成和社区贡献。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wiki-phrases-tokenizer.vtempest.workers.dev">AutoComplete - Wiki Phrases Tokenizer </a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2407.16312v1">MOMAland: A Set of Benchmarks for Multi-Objective Multi-Agent Reinforcement Learning</a>: 许多具有挑战性的任务（如管理交通系统、电网或供应链）涉及复杂的决策过程，必须平衡多个冲突的目标并协调...</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/playground">_Think-Lab Revised</a>: 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精炼的搜索查询。Scratchpad 是一个强大的工具，可以帮助您保持连贯性和准确性，尤其是...</li><li><a href="https://www.perplexity.ai/page/hermes-2-theta-70b-surpasses-l-Auq0bpLvSq6tpc4kxOxOMQ">Hermes 2 Theta 70B Surpasses Llama-3 Instruct</a>: Nous Research 宣布发布 Hermes 2 Theta 70B，这是一款与 Arcee AI 和 Charles Goddard 合作开发的强大新 AI 模型....</li><li><a href="https://www.perplexity.ai/page/mistral-large-2-revolutionizin-kUXugCSjRAevYdq7_cnYkA">Mistral Large 2: Revolutionizing AI</a>: 2024 年 7 月 24 日，Mistral AI 推出了 Mistral Large 2，这是一款强大的新语言模型，拥有 1230 亿参数和 128,000-token context window，...</li><li><a href="https://www.perplexity.ai/page/ai-search-engines-that-dont-pa-Cfeytd50SKmX8tr60Axxtw">Reddit Blocks Unpaid Search Engines</a>: Reddit 最近决定阻止大多数搜索引擎索引其内容（Google 除外），这引发了争议并提出了关于...</li><li><a href="https://www.perplexity.ai/page/conde-nest-file-cease-and-desi-zOe7YVNuTl.kxdsf3URlqw">Condé Nast Takes Legal Action Against AI Search Engine Perplexity</a>: 根据 The Information 的报道，出版巨头 Condé Nast 已向 AI 搜索引擎 Perplexity 发出停止侵权函，要求其停止...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/1265746471966216353)** (1 messages): 

> - `Nous Research subreddit`
> - `Upcoming AMA` 


- **Nous Research 推出 subreddit**：为 **Nous Research** 社区创建了一个新的 subreddit，用于讨论最新的 AI 研究和项目。
   - 鼓励成员加入并开启话题，分享见解和想法。
- **即将与 Nous 领导层进行 AMA**：计划在未来几周内与两名核心成员在 **Reddit** 上进行 AMA，回答社区问题。
   - 详情将很快公布，邀请成员参与并提交问题。



**提及的链接**: <a href="https://reddit.com/r/NousResearch">Reddit - Dive into anything</a>: 未找到描述

  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1265746646071771247)** (246 条消息🔥🔥): 

> - `Nous Research 更新`
> - `LLaMA 模型性能`
> - `AI 中的量化与精度`
> - `合成数据生成`
> - `OpenAI 功能与发布` 


- **LLaMA 模型表现参差不齐**：用户报告称，LLaMA 3.1 instruct 模型在多项基准测试中的表现似乎不如其前身 LLaMA 3.0，影响了知识相关任务的性能。
   - 有人对 RoPE 对性能的影响表示担忧，有迹象表明禁用 RoPE 会带来更好的结果，特别是在较小的模型中。
- **关于 GPU 使用和效率的讨论**：一位用户询问了在大规模微调期间 H100 GPU 与 A10G 相比的使用估算，强调了应对 GPU 可用性挑战的问题。
   - 对话还包括了如何量化 token 处理速度以评估性能提升的考量。
- **模型训练中的精度技术**：对量化进行了深入讨论，特别是 fp16、bf16 和 fp8 之间的细微差别，以及它们对模型训练和推理的影响。
   - 用户指出，虽然模型训练通常倾向于使用较低精度以提高效率，但某些配置可能会导致性能下降。
- **合成数据得到采用**：一位用户指出，他们的合成数据生成流水线显著提高了模型性能，特别是在巴西葡萄牙语方面。
   - 这突显了人们对通过生成的训练集来增强模型能力的替代方法的探索兴趣。
- **OpenAI 的功能开发**：一位用户质疑 OpenAI 的 SearchGPT 开发与之前的 Sora 和 GPT-4o 等功能相比是否成熟，并指出缺乏公开更新。
   - 对话表明了对预期发布的谨慎态度，呼应了之前只有炒作而没有实质性后续行动的情绪。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://jalammar.github.io/illustrated-transformer/">The Illustrated Transformer</a>: 讨论：Hacker News (65 分, 4 条评论), Reddit r/MachineLearning (29 分, 3 条评论) 翻译：阿拉伯语, 中文 (简体) 1, 中文 (简体) 2, 法语 1, 法语 2, 意大利语, ...</li><li><a href="https://x.com/cognitivecompai/status/1816314744007004214?s=46">Cognitive Computations (@cognitivecompai) 的推文</a>: llama 3.1 分数低于 llama 3？</li><li><a href="https://huggingface.co/casperhansen/mistral-large-instruct-2407-awq">casperhansen/mistral-large-instruct-2407-awq · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/google/gemma-2-9b#running-the-model-on-a-gpu-using-different-precisions">google/gemma-2-9b · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/clefourrier/status/1815998109958218200?s=46">Clémentine Fourrier 🍊 (@clefourrier) 的推文</a>: @paulml_fr 嗨！对于 ift-model，问题在于 MATH 分数：他们的模型系统地使用 CoT 进行回答，而不是遵循 few shot 示例格式 - 它几乎从不遵循预期的 temp...</li><li><a href="https://github.com/normand1/HyperFeeder/blob/master/audioScripts/ttsLocalScript.sh">HyperFeeder/audioScripts/ttsLocalScript.sh at master · normand1/HyperFeeder</a>: 自主播客生成器。通过在 GitHub 上创建账户为 normand1/HyperFeeder 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1eb9njj/claude_35_vs_llama_405b_vs_others_tested_by_ai/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1265755642744799232)** (33 messages🔥): 

> - `Hermes release on Llama 3.1`
> - `H100 GPUs vs Gaming GPUs`
> - `Data Synthesis in AI`
> - `Image-to-Text Finetuning`
> - `Consumer Grade Models` 


- **Hermes Llama 3.1 发布推测**：一位成员询问了针对 **Llama 3.1 8B** 的 Hermes 版本发布情况，并表示尽管目前尚未发布，但相信相关工作正在进行中。
   - *Teknium* 暗示内部测试正在进行，表明进展可能已接近。
- **H100 GPU 不适合游戏**：讨论了 **H100 GPU** 是否可以替代游戏 GPU，成员们确认由于缺乏显示输出，它们不适合游戏。
   - 一位成员幽默地指出，拥有这种硬件是一项挑战，并提到他们甚至处于“负资产”状态。
- **AI 模型中的数据合成问题**：成员们对 **数据合成 (Data Synthesis)** 表达了担忧，指出许多模型在这方面表现不佳，影响了训练结果。
   - 建议查阅 ***Wizardlm***、***Orca*** 和 ***Alpaca*** 论文中的材料以进一步了解。
- **图像转文本集成的微调更新**：一位新人询问最近的 **4o-mini Finetuning** 更新是否支持 **图像转文本 (Image-to-Text)** 微调，反映了对多模态能力日益增长的兴趣。
   - 这表明了在 AI 训练过程中整合各种数据类型的更广泛趋势。
- **在消费级硬件上运行大型模型**：成员们探讨了如何在消费级硬件上运行大型 AI 模型，并对 GPU 市场即将到来的竞争提出了见解。
   - 见解包括随着 AMD 准备挑战 NVIDIA 的主导地位，推理成本可能会下降。



**提到的链接**：<a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>：分析表明，不加区分地在真实内容和生成内容上训练生成式人工智能（通常通过从互联网抓取数据完成），可能导致崩溃 (collap)...

  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1265748904901607534)** (2 messages): 

> - `Grounded Refusals`
> - `Meta Team Intelligence` 


- **对 Grounded Refusals 的感悟**：一位成员表示惊讶，因为他们之前在讨论中没有考虑到 **Grounded Refusals**（有根据的拒绝）。
   - 这反映了对该话题涉及的复杂性和细微差别的领悟时刻。
- **感觉被 Meta 团队掩盖了光芒**：另一位成员评论说，他们觉得 **Meta 团队** 在方法上比他们更聪明。
   - 这种认可表达了对该团队能力和见解的欣赏。


  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/)** (1 messages): 

kentrid: 我猜没有可用的代码？
  

---

### **Nous Research AI ▷ #[reasoning-tasks-master-list](https://discord.com/channels/1053877538025386074/1264666760972472481/1265755172135764030)** (78 条消息🔥🔥): 

> - `Moral Reasoning Tasks`
> - `Syllogism Reasoning`
> - `Task Structuring`
> - `Dataset Collaboration` 


- **探索道德推理任务 (Moral Reasoning Tasks)**：成员们讨论了为复杂的道德查询（如 **trolley problem**）创建一个子板块的想法，以评估 AI 模型的推理能力。
   - 一项建议提议详细说明自动驾驶汽车在不可避免的碰撞场景中应如何优先考虑安全性，从而引发了对推理过程的进一步探究。
- **标准化任务格式**：开始努力将推理任务重构为独立的 Markdown 文档，以便更好地进行组织和文档清晰化。
   - 讨论内容包括标题、引用的格式考量，以及在主索引文档中链接任务的可能性。
- **数据集协作机会**：一位成员分享了一个专注于推理任务的精选数据集资源，并表示愿意与 Nous Research 团队共同努力。
   - 该倡议通过收集现有的 Benchmarks 和论文供共享使用，突显了 AI 推理领域协作研究的潜力。
- **改进任务文档**：为每个任务文档提议的最终字段列表包括描述、模态和引用类型的清晰分类。
   - 成员们还讨论了使用表格进行组织的益处，以及为主任务索引创建 Markdown 和 HTML 页面的可能性。
- **AI 在数学推理中的表现**：讨论引用了 **AlphaGeometry 2** 最近取得的成就，该模型在国际数学奥林匹克竞赛（International Mathematical Olympiad）的解题中展现出了银牌级别的表现。
   - 该模型的混合方法结合了语言模型与 Reinforcement Learning 技术，展示了 AI 数学推理能力的进步。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/sebkrier/status/1816500872806908306?t=qDY7edUwtRzFLOhc80mxQQ&s=19">Séb Krier (@sebkrier) 的推文</a>: 一些模型无法分辨 9.11 是否大于 9.9，而我们的模型在国际数学奥林匹克竞赛中获得了银牌水平的表现！AlphaGeometry 2 是一个神经符号混合系统，它...</li><li><a href="https://gist.github.com/pipinstallyp/28a5a67eca031fad12634b7b319ed2f2">sample.md</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/neurallambda/awesome-reasoning/">GitHub - neurallambda/awesome-reasoning: a curated list of data for reasoning ai</a>: 一个为推理 AI 精选的数据列表。通过在 GitHub 上创建账号为 neurallambda/awesome-reasoning 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1265807590302945372)** (20 条消息🔥): 

> - `开源 Git 工具 - stack-pr`
> - `Posits 与 MLIR`
> - `游戏开发与 AI 的交集` 


- **Modular 发布新 Git 工具 - stack-pr**: Modular 宣布发布名为 [_stack-pr_](https://github.com/modularml/stack-pr) 的新开源工具，用于在 GitHub 上管理堆叠式 Pull Requests (PRs)，旨在简化开发者的集成流程。
   - **Stacked PRs** 允许更小、更易于管理的贡献，从而增强 **Code Reviews**，并在 PR 评审过程中保持更平滑的更新。
- **对 AI 中使用 Posits 的兴趣**: 讨论围绕 *posits* 在 AI 应用中的实用性展开，并参考了 **Gosit** 和 [llvm-xposit](https://github.com/artecs-group/llvm-xposit) 等多种实现。
   - 成员指出，虽然 MLIR 可以集成 posits，但从传统浮点系统过渡可能会面临**重大挑战**。
- **游戏开发与 AI 潜在的交集**: 成员幽默地建议游戏开发和 AI 之间可能存在令人惊讶的重叠，并戏称这两个领域之间可能会有“亲密接触”。
   - 一位成员分享了一个可以探索这种交集的成熟游戏创意，但感叹自己既不是游戏开发者又缺乏资金的挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/announcing-stack-pr-an-open-source-tool-for-managing-stacked-prs-on-github">Modular: Announcing stack-pr: an open source tool for managing stacked PRs on GitHub</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：宣布 stack-pr：一个用于在 GitHub 上管理堆叠式 PR 的开源工具</li><li><a href="https://github.com/artecs-group/llvm-xposit">GitHub - artecs-group/llvm-xposit: The LLVM Project is a collection of modular and reusable compiler and toolchain technologies. This repository contains the Xposit RISC-V custom extension for posit arithmetic.</a>: LLVM 项目是模块化和可重用编译器及工具链技术的集合。此仓库包含用于 posit 算术的 Xposit RISC-V 自定义扩展。
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1265800268616826895)** (2 条消息): 

> - `Modular 更新`
> - `Modular 社区参与` 


- **Modular 宣布令人兴奋的更新**: Modular 分享了一条推文，强调了**新功能**和改进，鼓励用户在 [Modular 官方 Twitter](https://twitter.com/Modular/status/1816241375576482057) 探索最新功能。
   - 该帖子收到了社区的积极反馈，表明了对**功能增强**的浓厚兴趣。
- **与 Modular 社区的互动**: Modular 的另一条推文强调了**社区参与**的重要性，邀请用户在 [Modular 最新推文](https://twitter.com/Modular/status/1816241399060390235) 中提供反馈和对未来更新的建议。
   - 这一行动号召激发了成员分享他们的想法，凸显了协作的氛围。


  

---

### **Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1265821294465777717)** (5 条消息): 

> - `stack-pr 工具`
> - `对 stack-pr 的反馈`
> - `堆叠 PR (stacked PRs) 的好处` 


- **介绍适用于 GitHub 的 stack-pr 工具**：发布了一个名为 [_stack-pr_](https://github.com/modularml/stack-pr) 的新工具，旨在简化 GitHub 上堆叠 PR (stacked pull requests) 的管理，使开发者能够将变更拆分为更小、更易于管理的 PR。
   - 该工具处于早期开发阶段，诚邀社区通过[这篇博客文章](https://www.modular.com/blog/announcing-stack-pr-an-open-source-tool-for-managing-stacked-prs-on-github)提供反馈和建议。
- **关于使用 stack-pr 与简单标签的讨论**：一位成员表示担心，使用 stack-pr 工具似乎比他们在等待合并时标记分支的常规方法更复杂。
   - 另一位成员反驳道，虽然适应 stack-pr 需要时间，但它通过允许在评审期间持续提交，有效地防止了阻塞。
- **将重大变更拆分为多个 PR 的好处**：stack-pr 工具允许将大型变更拆分为较小的 PR，通过实现单个 PR 的并行评审来改进代码评审流程。
   - 随着每个 PR 被评审和合并，剩余的 PR 会自动更新，从而在没有瓶颈的情况下简化集成。



**提及的链接**：<a href="https://www.modular.com/blog/announcing-stack-pr-an-open-source-tool-for-managing-stacked-prs-on-github">Modular: Announcing stack-pr: an open source tool for managing stacked PRs on GitHub</a>：我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：宣布 stack-pr：一个用于管理 GitHub 上堆叠 PR 的开源工具

  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1265773744895557764)** (1 条消息): 

> - `Meta 对开放 AI 的承诺`
> - `Llama 3.1 模型进展`
> - `开放智能的可访问性`
> - `合成数据生成` 


- **Meta 倡导开放 AI 访问**：Meta 致力于开放可访问的 AI，并分享了 [Mark Zuckerberg 的信函](https://about.fb.com/news/2024/07/open-source-ai-is-the-path-forward/)，概述了开源对开发者、Meta 以及世界的益处。
   - 信中强调，**开源**促进了 AI 社区的协作与创新。
- **推出具有 128K 上下文长度的 Llama 3.1**：Meta 最新的模型（包括 Llama 3.1 405B）将上下文长度扩展至 **128K**，并支持 **8 种语言**，展示了其对开放智能的承诺。
   - 这一新模型非常独特，提供的能力足以与顶尖的 **closed source models** 媲美。
- **Llama 3.1 赋能新工作流**：Llama 3.1 405B 模型允许社区解锁新的工作流，重点展示了在 **synthetic data generation** (合成数据生成) 和模型蒸馏 (model distillation) 方面的能力。
   - 这些进步旨在增强 AI 的潜在应用，赋予开发者更高的**灵活性和控制力**。
- **持续开发 Llama 生态系统**：Meta 致力于通过提供与模型无缝协作的附加组件来扩展 Llama 框架。
   - Their goal is to equip developers with the tools necessary to create transformative AI applications.
   - 他们的目标是为开发者提供创建变革性 AI 应用所需的工具。



**提及的链接**：<a href="https://ai.meta.com/blog/meta-llama-3-1/">未找到标题</a>：未找到描述

  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1265748664467324928)** (97 条消息🔥🔥): 

> - `Mojo regex 支持`
> - `Tenka 包管理器`
> - `SDL 窗口创建`
> - `Iterator traits`
> - `Infrared 2D primitives` 


- **Mojo 缺乏 regex 库**：一名成员确认 **Mojo** 目前没有 **regex** 库，并分享了相关讨论链接以获取更多背景信息。
   - 该库的缺失引起了开发者对功能和便利性的担忧。
- **Tenka 包管理器发布**：一名成员宣布发布 **Tenka v0.1**（**Mojo** 的包管理器），并邀请大家贡献和反馈。
   - 记录了关于跨环境包版本兼容性的挑战，引发了对潜在解决方案的讨论。
- **在 Mojo 中创建 SDL 窗口**：一名用户在排查链接路径问题后，成功在 **Mojo** 中通过 **SDL** 创建了一个窗口。
   - 关于在定义中正确使用变量的讨论表明社区正在取得进展。
- **Iterator traits 和 Associated types**：成员们讨论了阻碍在 **Mojo** 中实现通用 **iterator API** 的基础性问题，特别是对 **Associated types** 的需求。
   - 表达了对带有字段的 **traits** 的担忧，并建议使用 **traits** 来增强 **iterator** 功能。
- **推进 Infrared 的 2D primitives**：一名开发者提到为 **Infrared** 添加了初始功能，并意识到许多 **2D** 形状在几何上可能与点对（point pairs）相关。
   - 他们表示有兴趣揭示这些 **2D primitives** 背后更深层次的数学抽象及其影响。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://doc.rust-lang.org/rust-by-example/generics/assoc_items/types.html">Associated types - Rust By Example</a>：未找到描述</li><li><a href="https://github.com/modularml/mojo/issues/3252)">Issues · modularml/mojo</a>：Mojo 编程语言。在 GitHub 上为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/Ryul0rd/sdl-bindings">GitHub - Ryul0rd/sdl-bindings</a>：在 GitHub 上为 Ryul0rd/sdl-bindings 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/3018/files">[stdlib] Iterator proposal by martinvuyk · Pull Request #3018 · modularml/mojo</a>：关闭了 #2629</li><li><a href="https://vega.github.io/vega/examples/pacman/">Pacman Example</a>：Vega - 一种可视化语法。Vega 是一种可视化语法，一种用于创建、保存和共享交互式可视化设计的声明式格式。通过 Vega，你可以描述视觉外观...</li><li><a href="https://github.com/modularml/mojo/issues/2629">[Feature Request] Introduce Iterator Trait · Issue #2629 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并相信此请求符合优先级。你的请求是什么？我希望拥有像 trait Iterato... 这样的 iterator trait。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1265749767518945402)** (198 messages🔥🔥): 

> - `SIMD 比较`
> - `EqualityComparable Trait`
> - `列表的 SIMD 行为`
> - `性能与 API 设计`
> - `函数重载与返回类型` 


- **关于 SIMD 比较的讨论**：社区正在讨论 SIMD 比较的处理方式，重点在于同时保留逐元素（element-wise）和整体比较结果，以满足 `any()` 和 `all()` 等不同用例的需求。
   - 大家的共识是，SIMD 的行为不应为了与列表（lists）兼容而牺牲性能，特别是在哈希表和数据库索引相关的用例中。
- **EqualityComparable Trait 与重载**：小组正在探讨是否应为 SIMD 类型引入 `Eq` 实现，以支持多态行为，同时避免标准库中出现过多的 Trait。
   - 建议包括设立独立的函数分别返回布尔值和 SIMD 逻辑，以便在不增加实现复杂性的情况下更好地满足 Trait 要求。
- **性能优先于 API 复杂度**：强调必须确保 SIMD 保持高效，而不是为了符合列表行为而破坏其功能，并主张在必要时使用专门的向量类型。
   - 结论倾向于保持低开销和直接使用 SIMD，而不是通过重载或修改现有特性来迎合列表兼容性。
- **改进 SIMD 功能的提案**：有提案建议创建 `AnyCmpSIMD` 和 `AllCmpSIMD` 等额外类型，专门用于明确和控制 SIMD 类型的比较行为。
   - 这些类型旨在弥合预期的数学行为与 SIMD 实现中的实际编码需求之间的差距，同时避免 Trait 系统过于臃肿。
- **SIMD 和 Trait 的未来方向**：讨论表明，迭代改进以及对 `FnAny` 和 `FnAll` 等函数 Trait 行为的正式认可可能是未来的方向。
   - 参与者热衷于确保自定义类型能够与 SIMD 操作无缝集成，同时期待框架内迭代器扩展的进展。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/builtin/simd/SIMD#__eq__)">SIMD | Modular Docs</a>：表示由硬件向量元素支持的小型向量。</li><li><a href="https://github.com/modularml/mojo/discussions/3233">[Proposal] 通过 `stdlib-extensions` 减轻 stdlib 维护者的工作量 · modularml/mojo · Discussion #3233</a>：此讨论旨在探讨以下提案：拉取请求 Markdown 文档。我们特别感兴趣于频繁贡献者以及 st... 的意见。</li><li><a href="https://github.com/modularml/mojo/pull/2412">[stdlib] SIMD 对 EqualityComparable 的一致性实现（由 helehex 提交） · Pull Request #2412 · modularml/mojo</a>：这允许 SIMD 符合 EqualityComparable，且不丢失任何原始行为。它使用第四条重载解析规则赋予新方法较低的优先级，同时仍符合...</li><li><a href="https://github.com/modularml/mojo/pull/2502">[stdlib] 将 `simd.bool()` 限制为 `size == 1`（由 helehex 提交） · Pull Request #2502 · modularml/mojo</a>：改用显式的 reduce_or()/reduce_and()，详见 #2412
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1266050133049610251)** (6 messages): 

> - `Mojo 实现`
> - `垃圾信息` 


- **开源 Mojo 矩阵乘法**：一名成员宣布开源了他们的 **Mojo** **矩阵乘法（matrix multiplication）**实现，并邀请他人在自己的机器上分享基准测试结果。更多详情可见 [Discord 消息](https://discordapp.com/channels/1087530497313357884/1266049763262992395/1266049763262992395)。
   - 该发布旨在促进用户之间围绕性能指标的协作和讨论。
- **对垃圾信息活动的担忧**：对话中提到了**垃圾信息（spam messages）**在多个频道蔓延的问题，造成了干扰。一名成员承认了该问题，但指出其他管理人员目前处于离线状态，无法立即处理。
   - 在用户寻求解决方案的同时，需要社区参与来有效解决此问题。



**提及的链接**：<a href="https://discordapp.com/channels/1087530497313357884/1266049763262992395/1266049763262992395">Discord - 充满乐趣与游戏的群聊</a>：Discord 是玩游戏、与朋友放松甚至建立全球社区的绝佳场所。定制你自己的空间来交流、玩耍和闲逛。

  

---

### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1266088560063418368)** (1 条消息): 

> - `计划停机`
> - `数据库维护` 


- **计划停机预告**：为了进行**数据库维护**，在 <t:1722060000:R> 将有一次计划内的 **10 分钟停机**。
   - 团队感谢您的耐心与理解，并对您的支持表示感谢。
- **数据库维护感谢**：团队对**计划停机**带来的不便表示歉意，并感谢用户的**支持**。
   - 此次维护对于确保持续的性能和可靠性至关重要。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1265746007061434461)** (305 条消息🔥🔥): 

> - `Mistral 与 Llama 模型对比`
> - `Perplexity 的 API 使用`
> - `SearchGPT 备受期待的发布`
> - `教育系统担忧`
> - `订阅与折扣问题` 


- **关于模型能力的辩论**：用户讨论了 Mistral 和 Llama 模型的性能，对其推理和写作能力看法不一，特别强调了 3.5 Sonnet 相比 GPT-4o 在写作方面的优势。
   - 一些用户对基准测试和感知到的不一致性表示怀疑，而另一些用户则指出了 4o 在代码编写方面的优势。
- **对 Perplexity 模型声明的信任**：用户对 Perplexity 使用 OpenAI 的 GPT-4o 等模型表示担忧，质疑如何验证所使用的 API 版本是否为原版。
   - 观点指出透明度的重要性，而一些人认为 Perplexity 模型的响应与直接从 OpenAI 获得的响应非常匹配。
- **对 SearchGPT 的期待**：社区推测即将发布的 SearchGPT 是免费还是基于订阅模式，并强调竞争如何使用户受益。
   - 用户表示如果证明是免费的，他们有兴趣尝试，并将其与目前使用 Perplexity 的体验进行对比。
- **教育中的批判性思维**：围绕 ChatGPT 等 AI 对教育影响的讨论突出了对批判性思维下降和依赖死记硬背的担忧。
   - 一些用户认为 AI 暴露了教育系统的缺陷，建议应优先考虑开卷评估和实际应用。
- **Perplexity 折扣码问题**：一位用户询问为什么他们的 Perplexity 折扣码无法使用，而其朋友的账户却可以。
   - 该查询指向了可能需要澄清的特定账户问题或资格差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.perplexity.ai/page/mistral-unveils-large-2-00GRlebXQQiufg1mtooQxg">Mistral 发布 Large 2</a>: Mistral AI 发布了其最新的大语言模型 Mistral Large 2，在多语言能力、推理等方面取得了显著进步...</li><li><a href="https://www.reddit.com/r/ClaudeAI/comments/1eb9njj/claude_35_vs_llama_405b_vs_others_tested_by_ai/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/search/google-is-the-only-search-engi-awmBq7utRw6FpNqjhY3szA">得益于 AI 协议，Google 现在是唯一能在 Reddit 上运行的搜索引擎</a>: Google 现在是唯一能检索 Reddit 最新结果的搜索引擎，这得益于两家公司之间达成的 6000 万美元协议，该协议允许...</li><li><a href="https://github.com/nuprl/MultiPL-E">GitHub - nuprl/MultiPL-E: 一个针对 LLM 的多编程语言基准测试</a>: 一个针对 LLM 的多编程语言基准测试。通过在 GitHub 上创建账户为 nuprl/MultiPL-E 的开发做出贡献。</li><li><a href="https://app.wordware.ai/share/999cc252-5181-42b9-a6d3-060b4e9f858d/history/3c76952a-c352-4520-95a2-ccf1a7b2b056?share=true">_Think-Lab 修订版</a>: 利用 ScratchPad-Think 的力量进行日常网络搜索。以 JSON 格式导出精炼的搜索查询。Scratchpad 是一个强大的工具，可帮助您保持连贯性和准确性，尤其是...</li><li><a href="https://www.perplexity.ai/search/google-is-the-only-search-engi-awmBq7utRw6FpNqjhY">Perplexity</a>: Perplexity 是一款免费的 AI 驱动型回答引擎，可为任何问题提供准确、可信且实时的答案。
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1265750530102267936)** (11 条消息🔥): 

> - `Mistral Large 2 发布`
> - `Reddit 屏蔽未付费搜索引擎`
> - `康泰纳仕对 Perplexity 采取法律行动`
> - `氢弹 vs 原子弹`
> - `第一民族资助机会` 

- **Mistral Large 2 树立 AI 新标准**：2024年7月24日，Mistral AI 推出了 **Mistral Large 2**，这是一款拥有 **1230 亿参数**（123 billion parameters）和 **128,000-token 上下文窗口**的语言模型，增强了在**代码生成**（code generation）、**数学**以及多语言任务方面的能力。
   - 它表现出极具前景的性能，在多语言 MMLU 基准测试中平均优于 **Llama 3.1 70B** 约 **6.3%**。
- **Reddit 限制搜索引擎访问**：Reddit 最近的更新阻止了除 Google 以外的大多数搜索引擎对其内容进行索引，这与这家科技巨头达成的 **6000 万美元**年度协议有关。
   - 这一政策变化引发了对开放互联网访问以及对数据抓取（data scraping）和 AI 训练影响的担忧。
- **Condé Nast 对 AI 搜索采取立场**：Condé Nast 已向 AI 搜索引擎 **Perplexity** 发送了停止侵权函（cease-and-desist letter），指控其未经许可使用其出版物的内容。
   - 这一法律行动凸显了传统媒体与 AI 驱动平台在内容使用方面日益加剧的紧张关系。
- **氢弹 vs 原子弹：了解差异**：氢弹利用核**聚变**（fusion），通过氢同位素结合产生比原子弹更强大的爆炸，而原子弹则利用核**裂变**（fission）来分裂重原子。
   - 这种根本差异导致了它们在爆炸威力和破坏效果上的显著不同。
- **原住民企业的融资机会**：**Aboriginal Business Investment Fund** (ABIF) 为加拿大的原住民企业提供关键的财务支持，赠款金额从 **150,000 美元到 750,000 美元**不等。
   - 包括 **Indigenous Growth Fund** 在内的联邦和省级计划，旨在加强经济发展倡议并补充技术创新。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/vdgw4JGH4WA">YouTube</a>: 未找到描述</li><li><a href="https://www.perplexity.ai/page/mistral-large-2-revolutionizin-kUXugCSjRAevYdq7_cnYkA">Mistral Large 2: Revolutionizing AI</a>: 2024年7月24日，Mistral AI 发布了 Mistral Large 2，这是一款强大的新型语言模型，拥有 1230 亿参数和 128,000-token 的上下文窗口...</li><li><a href="https://www.perplexity.ai/page/meta-drops-405b-params-model-hwb2ffonQ1eQ8xFUafmFWw">Meta Drops 405B Parameter Model</a>: Meta 发布了迄今为止最先进的 AI 语言模型 Llama 3.1 405B，拥有 4050 亿参数，其能力足以媲美领先的专有模型...</li><li><a href="https://www.perplexity.ai/search/mistral-large-2-revolutionizin-sVfT0LnmTJ2ER3WS5YqILQ#1">Mistral Large 2: Revolutionizing Language Models with Unprecedented Capabilities</a>: 这是我对使用 scratchpad 格式输出的评论：&lt;scratchpad&gt; [从提示中提取的关键信息] 关于 Mistral 文章的评审任务...</li><li><a href="https://www.perplexity.ai/page/mistral-unveils-large-2-00GRlebXQQiufg1mtooQxg">Mistral Unveils Large 2</a>: Mistral AI 发布了其最新的大语言模型 Mistral Large 2，在多语言能力、推理和...方面取得了显著进步。</li><li><a href="https://www.perplexity.ai/page/reddit-blocks-unpaid-search-en-Cfeytd50SKmX8tr60Axxtw">Reddit Blocks Unpaid Search Engines</a>: Reddit 最近决定阻止除 Google 以外的大多数搜索引擎索引其内容，这引发了争议并带来了关于...的问题。</li><li><a href="https://www.perplexity.ai/page/legal-trials-of-inanimate-obje-AGyEpycyQ6qVdMxEBIqUsg">Legal Trials of Inanimate Objects</a>: 纵观历史，法律系统曾处理过将无生命物体因对人类造成伤害或死亡而受审的异常做法。从...</li><li><a href="https://www.perplexity.ai/page/comparing-hydrogen-and-atomic-7aRv.AG6R4OpRYOPAOaiLw">Comparing Hydrogen and Atomic Bombs</a>: 氢弹和原子弹都是核武器的类型，但它们在基础反应、爆炸威力和...方面有显著不同。</li><li><a href="https://www.perplexity.ai/page/conde-nast-takes-legal-action-zOe7YVNuTl.kxdsf3URlqw">Condé Nast Takes Legal Action Against AI Search Engine Perplexity</a>: 根据 The Information 的报道，出版巨头 Condé Nast 已向 AI 搜索引擎 Perplexity 发送了停止侵权函，要求其停止...</li><li><a href="https://www.perplexity.ai/page/first-nations-funding-opportun-Itwe.QCHRPaxgM0zlG14Xg">First Nations Funding Opportunities</a>: Aboriginal Business Investment Fund (ABIF) 及类似计划为原住民企业和经济发展提供关键的财务支持...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1265767230793322547)** (3 messages): 

> - `Microsoft Copilot Studio`
> - `Llama 3.1 models API` 


- **Microsoft Teams 上传 Perplexity Connector 报错**：一名成员报告在将导出的 Perplexity Connector ZIP 文件上传到 **Microsoft Teams** 时遇到了**未指定的错误消息**。
   - 他们询问是否有人成功实现了该连接器，以及是否有可用的解决方案。
- **对 API 中增加 Llama 3.1 模型的兴趣**：一位用户询问是否可能将其他 **Llama 3.1 模型**（8B 和 70B）添加到 API 中。
   - 这一询问得到了另一位成员的赞同，突显了扩展可用模型选项的兴趣。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1265794018655211610)** (5 messages): 

> - `Llama 405B 降价`
> - `Middle-out transform 变更`
> - `数据库流量激增`
> - `Llama 3.1 降价`
> - `数据库性能问题` 


- **Llama 405B 降价 10%**：正如 [OpenRouterAI](https://x.com/OpenRouterAI/status/1816234833896694270) 所宣布的，**Llama 405B** 的价格已降低 **10%**。
   - 此次价格调整是市场持续竞争策略的一部分。
- **Middle-out transform 将默认关闭**：从 **8 月 1 日**起，**middle-out transform** 将默认关闭，不再沿用历史默认设置，以便为用户提供更好的控制。
   - 建议重度依赖此功能的用户根据 [文档](https://openrouter.ai/docs/transforms) 相应地更新其请求。
- **流量激增导致数据库压力**：平台经历了 **5 倍的流量激增**，导致数据库压力过大，因此安排在 **东部时间晚上 10:05** 进行停机升级。
   - 据报告，升级后服务已迅速恢复在线。
- **Llama 3.1-8b-instruct 降价 14%**：**meta-llama/llama-3.1-8b-instruct** 模型宣布降价 **14%**，延续了近期激进的价格调整趋势。
   - 这次价格变动引发了人们对价格竞争最终将在何处趋于稳定的疑问，尤其是在最近的产品发布之后。
- **数据库性能问题再次出现**：一些**数据库问题**再次浮现，导致在故障排除阶段性能可能下降。
   - 团队正在积极解决这些问题，以确保运行顺畅。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1816234833896694270">来自 OpenRouter (@OpenRouterAI) 的推文</a>: 🍕 降价：Llama 405B 价格降低 10%</li><li><a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct>)">Meta: Llama 3.1 8B Instruct (由 meta-llama 提供)</a>: Meta 最新级别的模型 (Llama 3.1) 发布了多种尺寸和版本。这个 8B 指令微调版本快速且高效。它在与...的对比中表现出了强劲的性能。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1265767247603826760)** (215 条消息🔥🔥): 

> - `Llama 3.1 性能`
> - `推理引擎问题`
> - `供应商之间的价格竞争`
> - `模型量化`
> - `OpenRouter 供应商问责制` 


- **Llama 3.1 表现出不稳定的性能**：用户报告 Llama 3.1 模型输出不一致，响应有时完全偏离主题或毫无意义，尤其是在高上下文负载下。
   - 切换供应商改善了一些用户的输出质量，这表明推理引擎的性能至关重要。
- **对推理引擎质量的担忧**：讨论强调，许多开源推理引擎可能会降低模型质量，当参数或上下文达到极限时，会导致胡言乱语般的响应。
   - 社区推测特定供应商及其部署实践可能存在潜在问题，这可能导致输出质量低下。
- **供应商参与价格竞争**：关于供应商为了吸引更多用户而压低价格的讨论正在进行，有时这是以牺牲模型质量和性能为代价的。
   - 这种定价行为引发了对 OpenRouter 上所提供模型的问责制和一致性的担忧。
- **模型量化技术**：用户讨论了向 FP8 等低精度量化方法的过渡，分析了其对性能和质量的影响。
   - 社区达成共识，虽然高质量的 FP8 几乎可以等同于 FP16，但问题可能会因推理引擎的具体实现而产生。
- **OpenRouter 在确保供应商质量方面的作用**：有人指出 OpenRouter 缺乏明确的问责制，担心供应商可能会对其托管的模型做虚假陈述，特别是在使用的量化方法方面。
   - 社区讨论了需要更好的验证流程，以确保供应商交付的模型符合预期的性能标准。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/meta-llama/llama-3.1-8b-instruct:free>">Meta: Llama 3.1 8B Instruct by meta-llama</a>: Meta 最新级别的模型 (Llama 3.1) 推出了多种尺寸和版本。这个 8B 指令微调版本快速且高效。与以下模型相比，它表现出了强大的性能...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: 最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://ollama.com/library/mistral-large">mistral-large</a>: Mistral Large 2 是 Mistral 的新旗舰模型，在代码生成、数学和推理方面能力显著增强，具有 128k 上下文窗口并支持数十种语言。</li><li><a href="https://github.com/princeton-nlp/SWE-agent">GitHub - princeton-nlp/SWE-agent: SWE-agent 接收一个 GitHub issue 并尝试使用 GPT-4 或你选择的 LM 自动修复它。它解决了 SWE-bench 测试集中 12.47% 的 bug，且运行仅需 1 分钟。</a>: SWE-agent 接收一个 GitHub issue 并尝试使用 GPT-4 或你选择的 LM 自动修复它。它解决了 SWE-bench 测试集中 12.47% 的 bug，且运行仅需 1 分钟。 - princ...</li><li><a href="https://github.com/BuilderIO/micro-agent">GitHub - BuilderIO/micro-agent: 一个为你编写（实际有用）代码的 AI Agent</a>: 一个为你编写（实际有用）代码的 AI Agent - BuilderIO/micro-agent</li><li><a href="https://huggingface.co/CofeAI/Tele-FLM-1T">CofeAI/Tele-FLM-1T · Hugging Face</a>: 未找到描述</li><li><a href="https://openrouter.ai/models/gryphe/mythomax-l2-13b">MythoMax 13B by gryphe</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演功能。#merge</li><li><a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>: 根据应用使用情况进行排名和分析的语言模型
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[일반](https://discord.com/channels/1091220969173028894/1246338143226167349/1265808013541773383)** (1 条消息): 

> - `Mistral Large 2` 


- **Mistral Large 2 展示了多语言实力**：**Mistral Large 2** 在多种语言中表现出色，包括英语、法语、德语、西班牙语、意大利语、葡萄牙语、荷兰语、俄语、中文、日语、韩语、阿拉伯语和印地语。
- **Mistral Large 2 令人印象深刻的语言模型**：**Mistral Large 2** 的性能使其成为多语言处理领域中值得关注的参与者。


  

---

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1266095286770733066)** (1 条消息): 

> - `SearchGPT`
> - `AI search features` 


- **推出 SearchGPT 原型**：OpenAI 正在测试 **SearchGPT**，这是一个新的原型，旨在通过清晰且相关的来源提供快速及时的回答，以增强搜索能力。
   - 该原型最初将向一小部分用户推出以获取反馈，随后将集成到 ChatGPT 中，更多详情请访问 [OpenAI 的 SearchGPT 页面](https://openai.com/index/searchgpt-prototype/)。
- **SearchGPT 的反馈循环**：用户将有机会在测试阶段对 **SearchGPT** 提供反馈，这对于优化搜索体验至关重要。
   - 收集到的反馈将影响 SearchGPT 的开发以及如何将其集成到 ChatGPT 主平台中。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1265745501370974339)** (177 条消息🔥🔥): 

> - `Mistral Model Download`
> - `MacBook Pro Performance`
> - `Internet Speed Upgrades`
> - `Voice Features in AI`
> - `Llama 3.1 Accessibility` 


- **Mistral 模型下载时间**：用户讨论了 **Mistral Large** 模型漫长的下载时间，一位用户报告称以目前的网速花费了 **2.5 小时**。
   - 另一位用户强调在他们的 MacBook Pro 上运行同一模型达到了 **18 tk/s**，表明尽管下载速度慢，但性能表现令人满意。
- **对 MacBook Pro 性能的热情**：对话强调了 **MacBook Pro M2 Max** 的能力，特别是配备 **96GB RAM** 的版本，非常适合在本地运行各种模型。
   - 用户比较了各自的配置，指出了性能差异，并对未来的 **M4 Max** 等升级表示期待。
- **对高速网络的期待**：几位用户表达了对更快互联网连接的渴望，其中一位期待在 12 月升级到 **1 Gbps** 光纤。
   - 其他人分享了他们目前的网速，一些人最近从 **50** 升级到了 **750 Mbps**，缩短了模型下载时间。
- **AI 工具中的语音功能**：讨论围绕新的 AI 语音功能展开，一些用户期待获得访问权限，而另一些人则指出并非所有人都收到了升级。
   - 一位用户幽默地提到了功能推出的挫败感，表示许多用户的功能更新仍然延迟。
- **不同平台上的 Llama 3.1 访问**：用户探索了在受地理位置限制的情况下访问和使用 **Meta Llama 3.1** 模型的方法。
   - 建议包括使用 **Groq** 或 **OpenWebUI** 等平台进行 API 访问，并强调需要为刚接触 AI 的年轻用户提供负担得起的解决方案。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mistralai/Mistral-7B-v0.3">mistralai/Mistral-7B-v0.3 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/EleutherAI/gpt-neo-125m">EleutherAI/gpt-neo-125m · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1265755519558090935)** (8 条消息🔥): 

> - `Feedback on GPT-4o`
> - `SearchGPT API Availability` 


- **用户对 GPT-4o 表示沮丧**：许多用户感叹升级到 **GPT-4o** 后，模型提供的错误信息越来越多，且无法直接引用来源，导致困惑。
   - 一位用户提到模型经常只是重复用户的问题而不是提供准确的答案，并表示 *“我觉得那位睿智的朋友早已离去，只剩下一个笨拙的孪生兄弟。”*
- **SearchGPT API 仍存疑问**：关于 **SearchGPT** 是否会通过 API 提供存在猜测，但用户认为首先建立通用访问权限更为重要。
   - 一位用户建议可能需要 **数月** 时间才能广泛可用，并强调功能性优先于 API 讨论。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1265807159657238642)** (7 条消息): 

> - `Memory Function Calls`
> - `Guidance for Memory Storage`
> - `Specificity in Events`
> - `Types of Information to Store` 


- **Memory Function Calls 实现**：一位用户寻求为聊天机器人实现 Function Calls，以创建、编辑和删除用户记忆，旨在提高性能。
   - *目前，该机器人仅在大约 60% 的时间内存储记忆。*
- **需要明确的指南**：一位成员强调了为聊天机器人提供关于何时以及如何保存记忆的精确指令的重要性。
   - *建议提供更多具体的示例将有助于模型做出准确的记忆决策。*
- **存储最喜欢和最讨厌的事物**：建议机器人应显式保存有关用户最喜欢和最不喜欢的事物的信息，如食物和游戏。
   - *用户提到了记住未来重要事件（如生日和发布日期）的价值。*
- **指南中的具体性与抽象性**：一位用户指出，对机器人的指令需要具体化，并指出模糊的输入会导致不准确的假设。
   - *建议使用开放变量来改进机器人处理各种记忆事件的方式。*


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1265807159657238642)** (7 条消息): 

> - `Function calls for chatbot memories`
> - `Guidance for memory storage`
> - `Event types for memory saving`
> - `Specificity in user memory requirements` 


- **聊天机器人记忆的 Function calls**：一位开发者正在为其聊天机器人开发 Function calls，用于创建、编辑和删除用户记忆，但在记忆准确性方面遇到困难。
   - 当前记忆存储成功率约为 **60%**，因此需要改进指令。
- **记忆存储需要更多指南**：建议为模型提供具体的指令，以确定何时以及将什么内容保存为记忆。
   - 这种指南可以增强模型决定哪些是有价值信息并加以记忆的能力。
- **要存储的记忆类型示例**：一位成员建议直接指示模型保存用户最喜欢和最不喜欢的项目，如食物、书籍和游戏。
   - 他们强调了保存对未来交互有用的细节的重要性，例如事件、年龄和姓名。
- **记忆事件类型的澄清**：讨论中包含了关于什么构成“事件”的模糊性，提到了日历事件（如生日和节日）。
   - 成员们指出了事件大类的重要性，同时也强调了在不限制范围的情况下保持具体性的必要性。
- **输入具体性的重要性**：一位参与者建议在要保存到记忆中的事件类型上保持具体，同时允许一定的抽象。
   - 建议使用开放变量，作为更好地捕捉可能事件多样性的一种手段。


  

---


### **OpenAI ▷ #[api-projects](https://discord.com/channels/974519864045756446/1037561385070112779/1266032474194968657)** (2 条消息): 

> - `Error uploading files to OpenAI`
> - `Python code for file upload`
> - `Vector stores configuration` 


- **向 OpenAI 上传文件时出错**：一位用户报告在尝试向 OpenAI 上传 **txt 文件**时收到 **400 错误**，提示不支持扩展名为 [none] 的文件。
   - 用户分享了详细的错误信息，并参考了关于支持文件类型的 [OpenAI documentation](https://platform.openai.com/docs/assistants/tools/file-search/supported-files)。
- **用于文件上传的 Python 代码**：用户上传文件的 Python 代码包括使用 **FastAPI** 和 OpenAI 客户端，但在执行过程中导致了错误消息。
   - 他们提到尝试了所有可用的文档但未获成功，表明在排查上传问题方面非常执着。
- **Vector stores 配置**：用户尝试在提供的 Python 代码中使用上传文件的 ID 来配置 Vector stores，但在文件上传和 Vector store 创建方面都面临错误。
   - 他们的代码流程重点似乎在于确保正确的文件处理和配置设置。


  

---

### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1265875618797588683)** (1 条消息): 

> - `Stable Video 4D`
> - `Dynamic multi-angle video generation`
> - `Technical report release` 


- **推出用于多视角生成的 Stable Video 4D**：我们很高兴地宣布 **Stable Video 4D**，这是我们的首个视频到视频（video-to-video）生成模型，可将单个视频转换为具有八个不同角度的**动态新视角视频**。
   - 该模型允许用户通过指定**摄像机角度**来定制输出，从而增强视频制作的创意。
- **使用 Stable Video 4D 快速生成帧**：**Stable Video 4D** 在大约 40 秒内生成 **8 个视角下的 5 帧**，显著提高了视频处理效率。
   - 这种创新方法为旨在快速创建高质量视频的用户提供了前所未有的通用性。
- **各领域的未来应用**：目前处于**研究阶段**，Stable Video 4D 旨在增强在游戏开发、视频编辑和虚拟现实中的应用。
   - 预计将进行持续改进，重点是进一步增强模型的能力和应用。
- **发布全面技术报告**：在宣布 Stable Video 4D 的同时，一份详细介绍方法论、挑战和突破的**全面技术报告**已经发布。
   - 用户可以点击[此处](https://stability.ai/news/stable-video-4d)访问该报告，深入了解模型的开发过程。
- **在 Hugging Face 上可用**：Stable Video 4D 模型现在已在 **[Hugging Face](https://huggingface.co/stabilityai/sv4d)** 上提供，方便用户使用这一尖端技术。
   - 这种开放访问旨在促进社区内的实验和进一步开发。



**提到的链接**：<a href="https://stability.ai/news/stable-video-4d">Stable Video 4D &mdash; Stability AI</a>：我们很高兴地宣布 Stable Video 4D 的可用性，这是一款创新模型，允许用户上传单个视频并接收八个新角度/视角的动态新视角视频，提供...

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1265746044252192928)** (147 条消息🔥🔥): 

> - `Stability AI 项目更新`
> - `Stable Diffusion 的使用`
> - `关于模型与性能的讨论`
> - `Lora 训练技巧`
> - `Inpainting 技巧` 


- **Stability AI 扩展 Stable Assistant 功能**：Stability AI 宣布为 Stable Assistant 推出新功能，包括 **Inpaint** 和 **Erase**，允许用户细化生成内容并增强创作工作流。
   - 这些工具支持无限次迭代和移除不需要的元素，可在此处申请 [3 天免费试用](https://stability.ai/stable-assistant)。
- **在 Discord 中使用 Stable Diffusion**：一位用户询问如何在 Discord 中使用 Stable Diffusion，并对其与 **Midjourney** 的用法差异表示困惑。
   - 建议用户查看相关的 Discord 频道，以获取有关 Stable Diffusion 集成的更新和潜在功能。
- **关于模型性能的辩论**：讨论涉及多种模型，有人断言某个特定模型的表现优于 **SDXL**，并强调了新版本发布时机的重要性。
   - 提到了 **Kolors** 和 **Auraflow** 等具有潜力的模型，尽管用户指出市场竞争激烈，有许多替代方案。
- **了解 Lora 训练**：用户讨论了训练 Lora 的最佳实践，重点在于针对眼睛和嘴巴等特定特征应使用完整图像还是裁剪后的图像。
   - 对话阐明了 Lora prompts 的策略，强调了训练数据集中细节对于提升效果的重要性。
- **Stable Diffusion 的 Inpainting 技巧**：用户探索了 Inpainting 方法，建议利用 **img2img** 流程和教程资源来优化结果。
   - 分享了使用带有上下文的 prompts 原则，作为将物体有效 Inpaint 到场景中的手段。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://images.flrty.li/">Image Viewer</a>: 未找到描述</li><li><a href="https://x.com/StabilityAI/status/1816520296775737642">来自 Stability AI (@StabilityAI) 的推文</a>: 今天，我们通过引入两个新功能扩展了 Stable Assistant 的能力：🖌️ Inpaint：用新内容替换指定区域，生成无限迭代。🫥 Erase：移除不需要的...</li><li><a href="https://x.com/elmanmansimov/status/1346552798528335875">来自 Elman Mansimov (@elmanmansimov) 的推文</a>: 文本生成图像：它是如何开始的 (2015) 以及现状如何 (2021)</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1dza7fy/comment/leneiip/">Reddit - 深入探索</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1265751656734265374)** (83 messages🔥🔥): 

> - `Flash Attention vs Traditional Attention`
> - `推理中的 VRAM 使用`
> - `Attention 机制中的分块 (Chunking)`
> - `Attention 算法对比`
> - `多选题数据集与 API` 


- **Flash Attention 优化了 VRAM 但未优化时间**：成员们讨论了 **Flash Attention** 在推理过程中有助于实现线性 VRAM 使用，但不会降低时间复杂度，其复杂度仍为二次方（Quadratic）。
   - 有人指出，在长缓存和单个 Query 的情况下使用 FA 实际上可能会更慢，因为在序列维度上的并行化程度较低。
- **Google 论文 vs Flash Attention**：关于开发 Flash Attention 的功劳是否应归于 **Google 论文** 存在分歧，成员们认为该论文在序列长度方面并没有实现线性空间占用。
   - 讨论强调了算法中影响内存和计算的**细微差异**。
- **Key-Value Cache 对性能的影响**：提出的一个关键点是 **KV-Cache** 的大小随序列长度线性增加，这一因素影响 VRAM，但对计算时间没有显著影响。
   - 成员们澄清说，虽然 Flash Attention 提高了内存效率，但其计算开销（Computational Overhead）保持不变。
- **Attention 机制的分块策略**：几位成员讨论了 Flash Attention 如何通过分块（Chunking）来减少内存带宽并提高效率，转向更小的矩阵。
   - 这种方法与朴素实现（Naive Implementations）形成对比，因为它在硬件上实现了更好的性能，从而实现了有效的并行处理。
- **多选题数据集 API 集成**：一位新成员询问如何使用各种 AI 服务测试非英语的多选题数据集，并寻求用于解析输出的模板。
   - 他们表示拥有 API keys 但在编程方面遇到困难，表明在实施高效测试方法论方面需要社区支持。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.cerebras.net/chip/context-is-everything-why-maximum-sequence-length-matters/">Context is Everything: Why Maximum Sequence Length Matters - Cerebras</a>: Cerebras 系统上 GPU Impossible™ 的序列长度可能在自然语言理解（Natural Language Understanding）、药物发现和基因组学方面取得突破。</li><li><a href="https://www.cerebras.net/chip/context-is-everything-why-maximum-sequence-le">Context is Everything: Why Maximum Sequence Length Matters - Cerebras</a>: Cerebras 系统上 GPU Impossible™ 的序列长度可能在自然语言理解、药物发现和基因组学方面取得突破。
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1265745350677893271)** (51 条消息🔥): 

> - `模型的 Inference 成本`
> - `MoE 效率`
> - `Meta 研究策略`
> - `AlphaProof 突破`
> - `xAI 的市场地位` 


- **讨论模型提供商的 Inference 成本**：成员们提出，像 **Mistral** 这样的模型在大规模应用时其 Inference 应该是免费的，并论证了在集群中使用单层或 MoE 的效率。
   - 有人担心，如果不能有效地使用 batch inference，由于复杂性增加，可能会削弱 MoE 的优势。
- **Meta 的研究策略受到审视**：讨论显示 **Meta 的方法**涉及利用各种外部研究，投入大量资源优化代码行，而不是利用更广泛的模型结构。
   - 一位成员指出 Meta 的运营策略令人费解，质疑他们不采用更高效方法的理由。
- **AlphaProof 在定理证明方面的成功**：对话提到了 **AlphaProof**，这是一个基于 AlphaZero 和 LLM 的应用，成功解决了 4 道 IMO 题目，根据 **DeepMind** 的说法，达到了银牌选手水平。
   - 围绕这一突破的兴奋点强调了 LLM 集成对竞争性数学方法的潜在影响。
- **xAI 在竞争中的地位变化**：对话反映了对 **xAI** 叙事的怀疑，成员表示由于 DeepMind 进展带来的有效竞争，其初始优势可能会减弱。
   - 讨论强调了马斯克的财务影响力，但质疑 xAI 的长期效能，重点在于资源的聪明利用与鲁莽支出。
- **蛋白质语言模型演示**：一位成员宣布参加 **ICML 的 ML4LMS Workshop**，展示关于 **protein language models** 如何揭示病毒模拟（viral mimicry）方面的研究。
   - 该公告引起了对生物学与 AI 之间新兴交叉领域的关注，表明机器学习社区内的关注度日益增长。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.17387">PERSONA: A Reproducible Testbed for Pluralistic Alignment</a>: 语言模型 (LMs) 的快速发展需要与多样化的用户价值观进行稳健的对齐。然而，当前的偏好优化方法往往无法捕捉用户的多样性...</li><li><a href="https://arxiv.org/abs/2405.16852">EM Distillation for One-step Diffusion Models</a>: 虽然 Diffusion Models 可以学习复杂的分布，但采样需要计算昂贵的迭代过程。现有的蒸馏方法可以实现高效采样，但存在明显的限制...</li><li><a href="https://x.com/mononofu/status/1816496369512612341?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">Julian Schrittwieser (@Mononofu) 的推文</a>: 我们最新的工作 AlphaProof，基于 AlphaZero、LLMs 和 @leanprover 定理证明器，结合 AlphaGeometry 2 成功解决了 4 道 IMO 题目并达到了银牌水平！🚀 更多信息请见...</li><li><a href="https://www.nature.com/articles/s41586-024-07566-y">AI models collapse when trained on recursively generated data - Nature</a>: 分析显示，不加区分地在真实和生成的内容上训练生成式人工智能（通常通过从互联网抓取数据完成），会导致模型崩溃...</li><li><a href="https://thesephist.com/posts/prism/">Prism: mapping interpretable concepts and features in a latent space of language | thesephist.com</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1266083883384967180)** (9 条消息🔥): 

> - `Meta scaling laws`
> - `数据 scaling 函数` 


- **Meta 的 Scaling Laws 受到审视**：一位用户质疑来自 **Meta** 的 **scaling laws** 是否受到数据叠加（data superposition）的影响，认为最佳数据量**并非**线性扩展。
   - 这引发了关于使用**指数函数**计算最佳数据量的讨论。
- **Chinchilla 的 Token 计算泛化**：对话提到了将 **Chinchilla** 泛化到**每个参数 20 个 token**，并指出根据他们的函数，最佳值没有显著变化。
   - 这导致人们承认，虽然 scaling 看起来有些扭曲，但推理过程似乎是合乎逻辑的。
- **对逆向数据分析的需求**：一位参与者表示，虽然发现很有趣，但**逆向分析**会更有益，重点是研究每个参数更多的数据而非模型规模。
   - 这一见解呼吁进一步研究增加数据如何相对于模型大小更好地优化性能。

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1265856859961561121)** (2 messages): 

> - `Awesome Interpretability Repository`
> - `NDIF Llama3-405b Access Opportunity` 


- **探索 Awesome Interpretability 仓库**：[Awesome Interpretability in Large Language Models](https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models) GitHub 仓库提供了一个专注于 **LLM 可解释性** 资源的全面集合。
   - 该仓库为研究人员深入理解大语言模型的细微差别提供了一个宝贵的枢纽。
- **NDIF 提供 Llama3-405b 实验访问权限**：National Deep Inference Fabric (NDIF) 正在邀请 AI 研究人员申请通过其 [网站](https://ndif.us/405b.html) 上描述的新编程接口，访问 **Llama3-405b** 模型进行突破性实验。
   - 参与者将获得 **数 TB 的 GPU 资源** 和支持，同时贡献超越传统基准测试的创新研究。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://ndif.us/405b.html">National Deep Inference Fabric</a>: NDIF 是一个研究型计算项目，旨在让研究人员和学生能够揭开大规模 AI 系统内部的奥秘。</li><li><a href="https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models">GitHub - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models: This repository collects all relevant resources about interpretability in LLMs</a>: 该仓库收集了所有关于 LLM 可解释性的相关资源 - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1265795942494244894)** (2 messages): 

> - `Evaluating MMLU on External APIs`
> - `Calculating VRAM Requirements` 


- **在外部 API 上评估 MMLU**：一位成员正在寻求帮助，希望在类似于 OpenAI 模式（包含 log_probs）的外部 API 上评估 **MMLU**。
   - 他们参考了一个 [GitHub PR](https://github.com/EleutherAI/lm-evaluation-harness/pull/2008)，该 PR 为 API 模型引入了一个超类（superclass），旨在实现模块化并改进请求处理。
- **如何计算模型评估的 VRAM 需求**：有人提出了关于计算有效评估模型所需 **VRAM** 方法的查询。
   - 这是一个普遍关注的问题，因为 **VRAM** 需求会显著影响模型评估期间的性能。



**提及的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/2008">Refactor API models by baberabb · Pull Request #2008 · EleutherAI/lm-evaluation-harness</a>：此 PR 为 API 请求模型引入了一个新的超类，提供：下游类的模块化、用于请求转换的可重载方法、API 请求和响应解析、Tokeniza...

  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1265770942077075518)** (2 messages): 

> - `NCCL Performance`
> - `Flute Matrix Multiplications` 


- **NCCL 重叠（Overlap）挑战**：一位用户在使用 [NCCL Issue #338](https://github.com/NVIDIA/nccl/issues/338) 的训练设置中，对反向传播（backward pass）期间实现 NCCL **计算重叠** 表示担忧。他们指出，虽然关于 NCCL 的讲座建议这是可行的，但实际实现起来比预想的要复杂。
- **为 LLM 引入 Flute**：另一位用户分享了 [Flute](https://github.com/HanGuo97/flute) 的仓库，这是一个专注于**快速矩阵乘法**的项目，专门为查找表量化（lookup table-quantized）的 LLM 及其应用而设计。该工具旨在优化 LLM 处理的性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/NVIDIA/nccl/issues/338">computation overlapped with nccl get much slower · Issue #338 · NVIDIA/nccl</a>: 我使用了来自 NVIDIA DeepLearningExamples 的环境，在多 GPU 上训练 ResNet-50（使用 horovod 和 nccl），发现 d...</li><li><a href="https://github.com/HanGuo97/flute">GitHub - HanGuo97/flute: Fast Matrix Multiplications for Lookup Table-Quantized LLMs</a>: 针对查找表量化 LLM 的快速矩阵乘法 - HanGuo97/flute
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1265922897785131068)** (1 条消息): 

> - `CUDA profiling tools`
> - `Nsight Compute`
> - `Triton testing helpers` 


- **使用 CUDA 工具分析 Triton Kernels**：你可以像分析其他 **CUDA** kernels 一样，使用 [Nsight Compute](https://developer.nvidia.com/nsight-compute) 等工具对 **triton kernels** 进行详细的 profiling。
   - Nsight Compute 提供引导式分析来优化 CUDA kernels，包括 GPU 吞吐量和 warp 状态统计数据。
- **开始使用 Nsight Compute**：对于有兴趣使用 **CUDA** 或 **OptiX** 优化 GPU 性能的人来说，[NVIDIA Nsight Compute](https://developer.download.nvidia.com/images/nvidia-nsight-compute-icon-gbp-shaded-128.png) 是一款必不可少的工具，它同时支持交互式 UI 和命令行用法。
   - 还有一个概览视频，展示了 Nsight Compute 中的引导式分析如何辅助进行 CUDA kernel 优化。
- **可用的 Triton 测试辅助工具**：Triton 提供了几个内置的辅助工具用于性能基准测试，包括 [triton.testing](https://triton-lang.org/main/python-api/triton.testing.html)。
   - 该功能包括 `do_bench` 和 `perf_report` 等函数，以便通过**简洁的 API** 进行性能测量。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.nvidia.com/nsight-compute">NVIDIA Nsight Compute</a>：用于 CUDA 和 NVIDIA OptiX 的交互式 profiler。</li><li><a href="https://triton-lang.org/main/python-api/triton.testing.html">triton.testing &mdash; Triton 文档</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 条消息): 

andreaskoepf: PyTorch 2.4 已发布：https://pytorch.org/blog/pytorch2-4/
  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1266077167146762383)** (1 条消息): 

> - `AlphaProof`
> - `AlphaGeometry 2`
> - `Mathematical reasoning`
> - `AGI potential in math` 


- **AlphaProof 和 AlphaGeometry 2 推动数学推理**：突破性模型 **AlphaProof** 和 **AlphaGeometry 2** 旨在解决数学中的高级推理问题，在竞赛中达到了**银牌水平**。
   - 这些模型标志着向开发具有增强数学推理能力的 **AGI** 迈进了一步，有可能开启科学和技术领域的进步。
- **当前 AI 在数学领域面临的挑战**：尽管取得了进展，但由于推理能力和可用训练数据的限制，目前的 AI 系统在**通用数学问题解决**方面仍面临挑战。
   - 之前的模型为**新算法**提供了见解并解决了一些**开放性问题**，但更广泛的数学应用仍需要持续开发。



**提到的链接**：<a href="https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/?utm_source=x&utm_medium=social&utm_campaign=&utm_content=">AI 在解决国际数学奥林匹克问题上达到银牌标准</a>：突破性模型 AlphaProof 和 AlphaGeometry 2 解决了数学中的高级推理问题

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1265752468273234024)** (3 条消息): 

> - `ML/AI Career Roadmap`
> - `Programming and Math Background` 


- **寻求 ML/AI 职业路线图指导**：一位成员正在寻求帮助，以设计一份旨在获得 **ML/AI** 全职职位和实习机会的**路线图 (roadmap)**，并分享了一份包含详细信息的 [Google Document](https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing)。
   - *他们提到愿意接受任何建议，并可以投入大量时间来实现目标。*
- **探索编程和数学背景**：另一位成员询问了那些追求 ML/AI 职位的人的**编程和数学背景**。
   - *这旨在了解在该领域取得成功所需的基础技能。*



**提到的链接**：<a href="https://docs.google.com/document/d/1s3H1ukZqAUuov_9LpQRRL6U1dI6WMiDrEirqN8ftK_A/edit?usp=sharing">ML Roadmap</a>：3 个月 - (9月, 10月, 11月) 路线图 统计学：https://www.youtube.com/watch?v=MXaJ7sa7q-8&amp;list=PL0KQuRyPJoe6KjlUM6iNYgt8d0DwI-IGR&amp;t=11s (1 周) 线性代数 - https://www.youtube.com/wat...

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1265778258624909323)** (6 messages): 

> - `模型的量化技术`
> - `FP16 执行时的内存问题` 


- **FP16 执行的内存限制**：一位用户表达了在以 **FP16** 精度运行模型时内存不足的困扰，强调了开发者面临的一个常见问题。
   - 这引发了关于探索优化内存使用的替代方案的建议。
- **探索使用 BnB 进行量化**：另一位用户建议研究使用 **bitsandbytes (BnB)** 库的 **Quantization** 技术，作为解决内存问题的潜在方案。
   - 这一建议引发了困惑，有用户对量化的概念提出了疑问。
- **理解量化以提高模型效率**：针对困惑，有人解释说量化通过使用更少的比特位（bits）来表示数据，从而减少内存占用，这对大型语言模型（LLMs）非常有益。
   - 讨论涵盖了各种量化方法，如 **AWQ**、**GPTQ** 和 **AQLM**，并详细说明了它们在优化模型性能方面的作用。



**提到的链接**：<a href="https://huggingface.co/docs/peft/main/en/developer_guides/quantization">Quantization</a>：未找到描述

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/)** (1 messages): 

marksaroufim: <@1213148470664495114>
  

---


### **CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1265754646509654149)** (18 messages🔥): 

> - `Blockwise Attention 实现`
> - `KV Cache 切分`
> - `Llama 3 中的 Ring Attention`
> - `Pipeline Parallelism`
> - `Llama 3.1 特性` 


- **Blockwise Attention 实现的困惑**：一位用户询问在 **Llama 3** 架构中，应该在何处将输入序列切分为块以进行 Blockwise Attention，特别是在将输入投影为 Q、K 和 V 之后。
   - *另一位成员澄清说，切分通常在输入层级进行，并认为无论是在投影之前还是之后进行，通常都不是问题。*
- **Ring Attention 中的 KV Cache 传递**：一位用户询问在输入序列切分后，模型如何处理跨 Token 的注意力，并指出此时缺少 **KV Cache**。
   - *一位成员回答说，“Ring”方法涉及在 Worker 之间传递 KV 分片（shards），确保每个 Worker 都能访问到完整的必要注意力数据。*
- **使用 Ring Attention 进行分层处理**：关于将输入块处理通过 **Llama 3** 的所有 28 层，并将计算出的 KV 传递给多个 GPU 进行并行处理的问题被提出。
   - *强调了必须在每一层计算完整的注意力分数，因此 Ring Attention 需要在每个注意力层都发挥作用。*
- **结合 Pipeline Parallelism 与 Context Parallelism**：一位用户讨论了在 GPU 之间同时实现 **Pipeline Parallelism** 和 **Context Parallelism** (Ring Attention)，并澄清了层如何在它们之间分配。
   - *成员们确认，跨多层管理 KV 块是必不可少的，并且这些方法可以在同一个系统中有效地共存。*
- **使用 Llama 3 进行长上下文模型的推理**：一位用户表达了在为长上下文模型推理实现 **Ring Attention** 时，**KV Cache** 大小带来的困难，强调了单设备上的内存限制问题。
   - *对话中提到，虽然 **Llama 3.1** 可能原生支持更长的上下文，但该用户目前仍在使用 Llama 3。*


  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1265880663530406031)** (6 条消息): 

> - `Slider 游戏发布`
> - `与 Baba Is You 的游戏对比`
> - `新成员介绍`
> - `商业模式讨论` 


- **Slider 作为免费解谜游戏发布**：[Slider](https://store.steampowered.com/app/1916890/Slider/) 是一款刚发布的全新免费解谜游戏，值得一试。
   - 创作者提到，这款游戏比 *Baba Is You* 更简单，因为玩家可以感觉到自己在取得 **进展 (progress)**。
- **游戏难度对比**：一位成员评论了 *Baba Is You* 的难度，表示自己不够聪明无法通关，但会去尝试 Slider。
   - 游戏创作者向他们保证，Slider 更 **容易**，并且可以更清晰地追踪进度。
- **欢迎新成员！**：一位新成员在聊天中介绍了自己，并表达了加入的兴奋之情。
   - 这种友好的问候为社区营造了温馨的氛围。
- **游戏领域商业模式的讨论**：一位成员推测可能会采用 **Adam Newman 商业模式**，即通过有争议的做法吸引 VC 资金。
   - 他们澄清说，虽然认为这种情况有可能发生，但实际上并未怀疑任何特定公司有此类行为。


  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1265791866125946954)** (2 条消息): 

> - `ICML 会议`
> - `咖啡聚会` 


- **抵达 ICML 及咖啡邀约**：@muhtasham 刚刚抵达 **ICML**，并表示有兴趣明天一起喝咖啡。
   - *可以与任何其他参会者建立联系*，在会议期间促进社交机会。
- **Erik 的延迟回复**：Erik 进行了回复，对延迟回信表示歉意，并确认他仍在参加会议。
   - 这突显了 ICML 忙碌的环境，参会者们都忙于各项活动。


  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1265747392041451610)** (96 条消息🔥🔥): 

> - `FP8 Training Challenges` (FP8 训练挑战)
> - `Outlier Detection in Training` (训练中的离群值检测)
> - `muP and Unit Scaling` (muP 与 Unit Scaling)
> - `Model Performance Improvements` (模型性能提升)
> - `GitHub Pull Requests` (GitHub Pull Requests)


- **FP8 训练中的挑战**：一位成员报告称其 FP8 124M 运行的收敛 Loss 未能达到 BF16 基准水平，可能仅与 GPT2 的性能相当。
   - 这一困境反映了在使用 FP8 与 BF16 相比时，关于训练稳定性和结果的更广泛担忧。
- **聚焦离群值检测机制**：在讨论因离群值（outliers）而跳过的更新时，强调了将离群值包含在移动平均（moving average）中会对结果产生负面影响，并可能导致收敛问题。
   - 通过一个 PR (pull request #711) 引入了一种新的离群值检测方法，旨在将离群值排除在移动平均计算之外。
- **muP 与 Unit Scaling 的探索**：成员们讨论了在 muP 背景下使用 Unit Scaling 方法的潜在益处，认为这可能会缓解 FP8 训练中出现的一些陷阱。
   - 尽管对于 Unit Scaling 是否能解决所有问题持怀疑态度，但其主作者的邻近性可能会促进进一步的协作。
- **训练性能的提升**：目前正在努力实施性能改进，特别是针对能显著惠及大模型的 matmul 操作。
   - 一位成员分享了近期引入另一项性能改进的计划，强调其对大模型的影响更大。
- **GitHub Pull Requests 进展**：在合并 PR 以简化模型初始化和解决平台兼容性方面取得了进展，对即将到来的变化感到兴奋。
   - 审查和完善 PR 的协作努力仍在继续，成员们互相鼓励检查潜在的竞态条件（race conditions）和冲突。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.17465">u-$μ$P: The Unit-Scaled Maximal Update Parametrization</a>：Maximal Update Parametrization ($μ$P) 旨在使模型的最佳超参数 (HPs) 与其规模无关，从而允许使用廉价的代理模型而非全尺寸模型来进行参数搜索...</li><li><a href="https://www.h-schmidt.net/FloatConverter/IEEE754.html">IEEE-754 Floating Point Converter</a>：未找到描述</li><li><a href="https://github.com/karpathy/llm.c/pull/702">Restore from master weights (&amp; allow restoring from a checkpoint of different precision) by ademeure · Pull Request #702 · karpathy/llm.c</a>：这对于保存了新 rng_state_last_update 的新 checkpoint 是完全确定性的，因此从 master weights 进行的 stochastic rounding 将使用完全相同的 seeds（在恢复时...）</li><li><a href="https://github.com/karpathy/llm.c/pull/694).">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/711">Outlier detection: catch more outliers by not updating moving average with skipped updates by ademeure · Pull Request #711 · karpathy/llm.c</a>：这是对 znorm/zgrad 更新跳过机制（-sl 和 -sg）的改进，以避免跳过离群值的更新。请注意，如果 zgrad 是导致跳过的离群值，znorm 仍将被更新...</li><li><a href="https://github.com/karpathy/llm.c/pull/694">Model init cleanup by ngc92 · Pull Request #694 · karpathy/llm.c</a>：将模型参数分配整合到单一源位置；使梯度缓冲区累加变为 eager 模式；移动了 encoder 确定性辅助缓冲区，使其由 forward 提前分配 -> ...</li><li><a href="https://huggingface.co/jrahn/gpt2_350M_edu_hermes">jrahn/gpt2_350M_edu_hermes · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/)** (1 条消息): 

andreaskoepf: https://x.com/AMD/status/1816168883587538946
  

---

### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1265783995237535897)** (2 条消息): 

> - `Lecture 24 Slides`
> - `GitHub Repository Updates` 


- **关于 Lecture 24 Slides 可用性的询问**：一位成员询问 [Lecture 24: Scan at the Speed of Light](https://github.com/cuda-mode/lectures/blob/main/TODO%20Lecture%2024:%20Scan%20at%20the%20Speed%20of%20Light) 的幻灯片是否很快会发布。
   - *这一请求突显了社区对 CUDA Mode 课程相关教学材料的持续关注。*
- **呼吁更新 GitHub Slides**：另一位成员询问同行是否手头有幻灯片，并建议通过 Pull Request 更新 [GitHub 仓库](https://github.com/cuda-mode/lectures)。
   - *这反映了社区内部为保持教学资源最新而进行的持续协作和贡献。*



**提及的链接**：<a href="https://github.com/cuda-mode/lectures">GitHub - cuda-mode/lectures: Material for cuda-mode lectures</a>：cuda-mode 课程材料。通过在 GitHub 上创建账号为 cuda-mode/lectures 的开发做出贡献。

  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1265980008287961098)** (11 条消息🔥): 

> - `DeepMind AI achievements`
> - `Runway AI training data leaks`
> - `OpenAI's SearchGPT prototype` 


- **DeepMind AI 在 IMO 2024 中获得银牌**：围绕 Google DeepMind AI 是否真的在 [IMO 2024](https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/) 中获得银牌展开了讨论，引用了 Google 官方博客称其达到了“银牌标准”。
   - 人们对标准的清晰度提出了质疑，怀疑论者认为 Google 可能调整了挑战内容以突出其 AI 的表现。
- **Runway AI 的训练数据源被曝光**：一次泄露显示，Runway 备受赞誉的 [AI 视频生成工具](https://www.404media.co/email/64056c13-be6e-46e7-8c90-b53dd30026f2/) 是利用从 YouTube 抓取的内容和盗版电影进行训练的，引发了伦理问题。
   - 这一揭露在社区中引起了轰动，评论表明这场讨论可能会变得非常激烈。
- **OpenAI 凭借 SearchGPT 进入搜索市场**：OpenAI 宣布测试 [SearchGPT](https://openai.com/index/searchgpt-prototype/)，这是一个旨在提供快速回答和相关来源的原型，将由 10,000 名用户进行试用。
   - 他们计划收集反馈以整合进 ChatGPT，这激发了人们对 AI 搜索功能潜在增强的期待。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.404media.co/email/64056c13-be6e-46e7-8c90-b53dd30026f2/">Runway Ripped Off YouTube Creators</a>：一份泄露的内部文件显示，Runway 著名的 Gen-3 AI 视频生成器收集了数千个 YouTube 视频和盗版电影作为训练数据。</li><li><a href="https://manifold.markets/ahalekelly/did-a-google-deepmind-ai-really-get">Did a Google Deepmind AI really get Silver on IMO 2024?</a>：已解决：是。The Verge 发布了一篇声称此事的文章，但现在已被删除。可能是 The Verge 不小心违反了新闻禁令？如果有人有全文，将不胜感激。htt...</li><li><a href="https://x.com/AndrewCurran_/status/1816537157831655484">Andrew Curran (@AndrewCurran_) 的推文</a>：OpenAI 正在进入搜索市场。10,000 名测试用户将获得早期访问权限。引用 OpenAI (@OpenAI) —— 我们正在测试 SearchGPT，这是一个临时原型，具有新的 AI 搜索功能，可以为您提供快速的...</li><li><a href="https://deepmind.google/discover/blog/ai-solves-imo-problems-at-silver-medal-level/">AI achieves silver-medal standard solving International Mathematical Olympiad problems</a>：突破性模型 AlphaProof 和 AlphaGeometry 2 解决了数学中的高级推理问题</li><li><a href="https://docs.google.com/spreadsheets/d/1eO5cwguMHeu63F0vsKRXs_dLlcuy_P4F/edit?usp=sharing&ouid=105662557213053165487&rtpof=true&sd=true>">Video sourcing - Jupiter.xlsx</a>：来自 [已编辑] 的关键词（已清洗 3d,Yes,[列内容被 404 MEDIA 编辑] aerial,Yes alien,Yes animation,Yes anime,Yes apocalypse,Yes apollo,Yes astronaut,Yes beach,Yes bear,Yes beard,Yes bed,Ye...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1265922660890574888)** (9 messages🔥): 

> - `Books on Modern Architectures`（现代架构书籍）
> - `LLAMA 3.1 Annealing`（LLAMA 3.1 退火）
> - `Foundations of Computer Vision Book`（《计算机视觉基础》书籍）


- **现代架构书籍推荐**：一名成员正在为一门 ML 课程寻找关于 **Diffusion** 和 **Transformers** 等现代架构的书籍推荐。
   - *我刚刚抢购了几本 rasbt 的《Building LLMs from scratch》*，但还在寻找更多专注于上述架构的专业书籍。
- **理解 LLAMA 3.1 退火 (Annealing)**：讨论集中在 **LLAMA 3.1 技术报告**上，特别是关于 Annealing 的概念以及在训练过程中将 Learning Rate 降低到 0 的做法。
   - 一位成员解释说，这种低 Learning Rate 有助于防止错过最优值，并可能通过精细的 Pretraining 提升 Leaderboard 表现。
- **额外阅读材料建议**：一位成员推荐了新书 **Foundations of Computer Vision**，如果预算允许，该书涵盖了现代 Computer Vision 主题。
   - 还有人提到了 **Chris Bishop** 的深度学习新书和 **Kevin Murphy** 的概率 ML 书籍，其中可能包含相关的讨论。



**Link mentioned**: <a href="https://udlbook.github.io/udlbook/">Understanding Deep Learning</a>: 未找到描述

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1265758078079602762)** (19 messages🔥): 

> - `Student Open Letter Contest`（学生公开信竞赛）
> - `New York Times Opinions`（《纽约时报》观点栏目）
> - `B2B Pricing Competition`（B2B 定价竞争）
> - `GPT-4 Magnet Link`（GPT-4 磁力链接）
> - `Parker Conrad and Rippling`（Parker Conrad 与 Rippling）


- **学生公开信竞赛引发关注**：一位成员分享了一篇关于“学生公开信竞赛”的 [《纽约时报》文章](https://www.nytimes.com/2024/07/12/learning/a-letter-to-midjourney.html)，引发了对其报道内容的惊讶。
   - *为什么这会上《纽约时报》？* 另一位成员质疑道，对该报的观点文章表示怀疑。
- **对《纽约时报》观点栏目的批评**：几位成员批评了《纽约时报》的观点板块，其中一人评论说它很“烂”，并对其文章选择表示困惑。
   - 讨论凸显了对主流媒体叙事的普遍不适感。
- **B2B 定价动态**：一位成员评论说，某家公司竟然能与 Databricks 并驾齐驱，令人惊讶；另一位成员澄清说，这是由于他们的 B2B 定价策略和缺乏竞争。
   - 这引发了关于商业策略和市场地位的更广泛讨论。
- **对获取 GPT-4 的渴望**：一位用户幽默地表达了对 GPT-4 磁力链接（Magnet Link）的渴望，反映了 AI 社区对轻松获取资源的向往。
   - 另一位成员插话提到了未来的一个场景，他们会毫不犹豫地下载一个 xjdr 磁力链接。
- **Parker Conrad 的声誉**：关于 Rippling 创始人 Parker Conrad 的问题被提出，一位成员指出他们从未对他的公司产生过好感。
   - 对话暗示了对他创业历程的好奇与怀疑交织的情绪。



**Link mentioned**: <a href="https://x.com/anothercohen/status/1816338693575368755">Tweet from Alex Cohen 🤠 (@anothercohen)</a>: 更新：天哪。引用 Alex Cohen 🤠 (@anothercohen) 的话：你们想看具尸体吗？

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1265804163619754037)** (50 条消息🔥): 

> - `GPT-4o 训练数据洞察`
> - `Prompt 多样性的重要性`
> - `Galactica LLM 回顾`
> - `SearchGPT 测试`
> - `数据集多样性的挑战` 


- **结合 BPE Tokenizer 的 GPT-4o 训练数据洞察**：提到了一篇讨论 [BPE Tokenizer 揭示了哪些关于训练数据的信息](https://openreview.net/pdf?id=0SRg6Cwx3h) 的论文，针对 **GPT-3.5** 和 **GPT-4o** 等模型，重点关注跨语言和跨领域的 Token 分布。
   - 该论文基于 Token 分析，对这些模型中使用的数据混合（Data Mixture）提出了严肃的假设。
- **Prompt 多样性及其重要性**：成员们讨论了 **Prompt 多样性** 与偏好评分响应的数量和质量之间的关键关系。
   - 他们强调，虽然部分多样性源自**采样分布（sampling distributions）**，但获取真正新颖的 Prompt 仍然是一个重大挑战。
- **Galactica LLM 引领未来发展**：在即将进行的采访中，将寻求关于 **Galactica LLM** 及其负责人 Ross Taylor 的见解，特别是关于过去的挑战和潜在的未来工作。
   - 社区对该项目如何从 **L2** 面临的挑战演变为在 **L3** 达到 **SoTA** 表示了兴趣。
- **OpenAI 宣布 SearchGPT 测试**：OpenAI 宣布为 **SearchGPT** 设立一个小规模测试组，这是一款旨在提供快速、相关答案的新型 AI 搜索功能。
   - 用户推测了访问权限和相关功能等因素，并引发了一些关于为了获得访问权限而进行“贿赂”的幽默评论。
- **创建多样化数据集的挑战**：讨论了获取多样化数据集的难度，强调即使是付费 Prompt 也往往在格式和内容上缺乏真正的多样性。
   - 成员们分享了评估多样性的技术，例如根据人类知识分类法对 Prompt 进行分类，但也承认了数据集收集中的物流挑战和负面激励（perverse incentives）问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/apples_jimmy/status/1816388658062373032?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：空气中弥漫着某种气息。一种充满希望的狂热氛围。让我们开始数学运算吧。</li><li><a href="https://x.com/rosstaylor90/status/1724879570724393363">来自 Ross Taylor (@rosstaylor90) 的推文</a>：没想到随手写的、清晨的 Galactica 回顾会引起这么多反响——毕竟这在 AI 时代已经是 3000 年前的事了。为了结束这个话题，下面是 Sean Murra 的精彩演讲...</li><li><a href="https://x.com/dorialexander/status/1816237637998633190?s=46">来自 Alexander Doria (@Dorialexander) 的推文</a>：呃，那是给我的论文：BPE Tokenizer 揭示了哪些训练数据。基于 Token 分布对 GPT-3.5、GPT-4o、Claude 的各语言/领域数据混合提出了严肃的假设。https://open...</li><li><a href="https://x.com/testingcatalog/status/1816544468830687288?s=46">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：ChatGPT 将在主侧边栏顶部获得 SearchGPT 的独立入口 👀</li><li><a href="https://x.com/openai/status/1816536290822881780?s=46">来自 OpenAI (@OpenAI) 的推文</a>：我们正在测试 SearchGPT，这是一个新型 AI 搜索功能的临时原型，旨在为您提供快速及时的答案，并附带清晰相关的来源。我们正向一小部分用户开放以获取反馈...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1266119148430299298)** (39 messages🔥): 

> - `Perplexity 的过度炒作`
> - `Zuckerberg 与 OpenAI 的策略对比`
> - `LLM 的 Web Browsing 能力`
> - `研究查询与 Agent 效率` 


- **Perplexity 因过度炒作面临批评**：成员们对 **Perplexity** 表示怀疑，强调其过度依赖排名靠前的结果，且在处理复杂搜索时表现不足，导致其被指估值过高。
   - *一位用户指出，直接使用 Google 通常比依赖 Perplexity 产生更好、更快的搜索结果*。
- **Zuckerberg 的方法与 OpenAI 的对比**：讨论对比了 **Zuckerberg** 广泛传播的 Op-ed 策略与 **OpenAI** 更侧重于华盛顿特区内部人士的针对性策略，展示了在受众参与方面的差异。
   - *一位成员幽默地提到了科技领袖之间持续的“笼斗”，暗示在不同的发布策略中竞争日益加剧*。
- **Web Browsing 能力对 LLM 至关重要**：成员们讨论了 **LLM Web Browsing 能力** 的局限性，强调需要更深层次的搜索过程，以产生超出搜索结果第一页的有用信息。
   - *一位用户感叹道，虽然 Web Browsing 被期望能增强能力，但它往往导致更慢的处理速度和更高的 Inference costs*。
- **改进研究 Agent 的潜力**：用户建议，一个能够深入挖掘搜索结果的高级搜索 Agent 可能会提供巨大价值，尽管这本质上会显著提高成本。
   - *大家达成共识，认为目前像 Perplexity 这样的产品未能利用更深层的搜索方法，也无法针对复杂查询进行有效的迭代。*



**提到的链接**：<a href="https://fxtwitter.com/kifleswing/status/1816542216678179083?s=46">来自 kif (@kifleswing) 的推文</a>：在 ChatGPT 最近的搜索引擎公告中，他们询问“8 月份北卡罗来纳州布恩市的音乐节”。ChatGPT 博客文章的示例图片中有五个结果：...

  

---


### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1266071164334375105)** (1 messages): 

> - `Pluralistic Alignment`
> - `Synthetic Personas`
> - `Persona Hub` 


- **介绍关于 Pluralistic Alignment 的 PERSONA 论文**：Synth Labs 发布了一篇题为 [PERSONA: A Reproducible Testbed for Pluralistic Alignment](https://x.com/synth_labs/status/1816460910187237482) 的新论文，使用 **1,586 个 Synthetic Personas** 和 **317,200 个偏好对**评估语言模型如何与多样化的用户价值观对齐。
   - 这些 Persona 反映了**现实世界的多样性**，整合了基于美国人口普查的特征以及独特的个性化特征。
- **与 Persona Hub 的比较**：讨论中将这篇新论文与最近讨论的 **Persona Hub** 项目进行了比较，尽管目前尚不清楚两者的实际相似程度。
   - 一位用户提到，根据 *goose man* 的说法，这两个概念实际上是不同的。



**提到的链接**：<a href="https://x.com/synth_labs/status/1816460910187237482">来自 SynthLabs (@synth_labs) 的推文</a>：🚨新论文🚨 PERSONA: A Reproducible Testbed for Pluralistic Alignment。我们使用 1,586 个 Synthetic Personas 和 317,200 个偏好对来评估 LM 如何与多样化的用户价值观对齐。Personas 反映了...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1266072524777590804)** (2 messages): 

> - `AI 控制的未来`
> - `OpenAI Rule-Based Reward 论文` 


- **关于 AI 控制的紧迫问题**：Sam Altman 强调，AI 的未来取决于美国是会培育一种全球受益的技术，还是允许威权政权获得权力。他指出*没有第三种选择*，并敦促就此事做出战略决策。
   - 随着 AI 的持续进步，Altman 警告说，威权政府正准备投入巨资以追赶并可能超越美国，暗示了其中涉及的风险。
- **关于 OpenAI Rule-Based Reward 论文的讨论**：一位成员询问是否有人读过 [OpenAI Rule-Based Reward 论文](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)，并将其比作 OpenAI 的 CAI 方法。
   - 一些成员指出，另一位贡献者确实读过该论文并参与了讨论，表明大家对其影响有共同兴趣。



**提到的链接**：<a href="https://archive.is/Jn5xv">评论 | Sam Altman：AI 的未来必须是民主的 - 华盛顿邮报...</a>：未找到描述

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1265747780719087627)** (127 messages🔥🔥):

> - `SearchGPT 发布`
> - `AI 在 IMO 的表现`
> - `基于规则的奖励 (Rule-Based Rewards)`
> - `LLM 作为评审 (LLM as Judge)`
> - `对合成数据的担忧` 


- **OpenAI 发布 SearchGPT**：OpenAI 宣布推出名为 [SearchGPT](https://openai.com/index/searchgpt-prototype/) 的原型，旨在提升超越现有产品的搜索能力。
   - 该原型最初将在一小部分用户中进行测试以获取反馈，随后将集成到 ChatGPT 中进行实时运行。
- **AI 在 IMO 获得银牌**：Google DeepMind 展示了一个混合 AI 系统，该系统在国际数学奥林匹克竞赛 (IMO) 中完全解决了 6 道题目中的 4 道，达到了银牌水平。
   - 该程序结合了用于形式化推理的 AlphaProof 和 AlphaGeometry 2，展示了 AI 在数学解题能力方面的重大进步。
- **OpenAI 用于 AI 安全的基于规则的奖励**：OpenAI 引入了 [基于规则的奖励 (RBRs)](https://openai.com/index/improving-model-safety-behavior-with-rule-based-rewards/)，旨在无需大量人工数据收集的情况下对齐 AI 行为，从而增强系统安全性。
   - RBR 方法利用较少的人工标注示例，同时允许对不断变化的安全策略做出自适应响应。
- **LLM as Judge 的评分说明 (Grading Notes)**：Databricks 引入了 [评分说明 (Grading Notes)](https://www.databricks.com/blog/enhancing-llm-as-a-judge-with-grading-notes)，作为评估准则，以增强 LLM 在专业领域作为评审的可靠性。
   - 这些说明通过为 LLM 评估提供结构化指南，支持特定领域的 AI 应用。
- **AI 训练中对合成数据的担忧**：最近的一篇论文对过度依赖合成数据进行 AI 训练的风险提出了担忧，指出这可能导致连续迭代后的模型崩溃 (Model Collapse)。
   - 该领域的专家强调了多样化训练输入的重要性，以维持信息质量并防止模型性能下降。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/drjimfan/status/1816521330298356181?s=46">来自 Jim Fan (@DrJimFan) 的推文</a>：LLMs 是外星怪兽。令人深感不安的是，我们的前沿模型既能在数学奥林匹克竞赛中获得银牌，却又无法回答“9.11 和 9.9 哪个数字更大”？后者...</li><li><a href="https://youtu.be/U_cSLPv"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/matei_zaharia/status/1816175703290962080">来自 Matei Zaharia (@matei_zaharia) 的推文</a>：如何让 LLM-as-judge 在专业领域变得可靠？我们的应用 AI 团队开发了一种简单但有效的方法，称为 Grading Notes，我们一直在 Databricks Assistant 中使用它。我们...</li><li><a href="https://x.com/aidan_mclau/status/1816537715393077324">来自 Aidan McLau (@aidan_mclau) 的推文</a>：&gt;作为 Google &gt;构建酷炫的 AI！ &gt;AI 在数学方面表现出色。 &gt;耶！ &gt;作为 OpenAI &gt;等待 Google 发布可爱的数学模型 &gt;发布极具竞争力的搜索引擎，可能直接引爆 Go...</li><li><a href="https://x.com/alexandr_wang/status/1816491442069782925?s=46">来自 Alexandr Wang (@alexandr_wang) 的推文</a>：1/ 《Nature》上的一篇新论文显示，随着连续的模型世代在合成数据上进行递归训练，会出现模型崩溃（model collapse）。这是一个重要的结果。虽然今天许多研究人员将合成...</li><li><a href="https://x.com/polynoamial/status/1816500904516051327?s=46">来自 Noam Brown (@polynoamial) 的推文</a>：来自 @GoogleDeepMind 的结果非常令人印象深刻！他们将复杂的数学问题转换为形式化推理语言 Lean，然后使用 AlphaZero 风格的方法来寻找解决方案。这与...结合...</li><li><a href="https://x.com/aditya_advani/status/1816187840163987654">来自 Aditya P. Advani (@aditya_advani) 的推文</a>：@latentspacepod @lvdmaaten @swyx @vibhuuuus @picocreator @eugeneyan 秉承快速回顾的精神，我的开源 Arxiv2Paper 生成器 ELDO 为俱乐部的观看乐趣制作了这段 2 分钟的视频...</li><li><a href="https://x.com/theseamouse/status/1816324300351099057?s=46">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>：这篇论文有几点有趣之处：1. 他们只需要手动标注约 500 个样本（gold data） 2. 行为策略（behavior policy）只是一个 prompt。我们开始看到合成数据如何...</li><li><a href="https://x.com/lilianweng/status/1816164033617445240?s=46">来自 Lilian Weng (@lilianweng) 的推文</a>：基于规则的奖励（RBRs）利用模型根据一套安全准则提供 RL 信号，使其更容易适应不断变化的安全政策，而无需沉重地依赖人工数据。它还使我们能够...</li><li><a href="https://x.com/deedydas/status/1816515078562431241">来自 Deedy (@deedydas) 的推文</a>：Google 刚刚发布了一个精英 AI 数学家。这是一个神经符号系统，它通过微调后的 Gemini 将问题形式化为 Lean（一种形式化语言），并使用 AlphaZero 风格的搜索来解决...</li><li><a href="https://x.com/Ji_Ha_Kim/status/1816527854655754566">来自 Ji-Ha (@Ji_Ha_Kim) 的推文</a>：如果有人感兴趣，OpenAI 在 2 年前就做过与新的 AlphaProof 类似的工作，规模较小，并为此写了一篇论文。https://openai.com/index/formal-math/</li><li><a href="https://x.com/kempelab/status/1800822751273636109?s=46">来自 Julia Kempe (@ICML) (@KempeLab) 的推文</a>：如何在不发生灾难性退化的情况下利用 AI 合成数据？来自人类甚至更弱模型的 Rank-and-prune 反馈，经证明可以恢复甚至超越原始性能！参见 https:/...</li><li><a href="https://x.com/morqon/status/1816540138274726085">来自 morgan — (@morqon) 的推文</a>：OpenAI 的搜索实验有一个专门的侧边栏用于显示链接结果，不会埋没网站（且没有广告）</li><li><a href="https://x.com/chipro/status/1816521492760580127?s=46">来自 Chip Huyen (@chipro) 的推文</a>：构建生成式 AI 应用平台 https://huyenchip.com/2024/07/25/genai-platform.html 在研究了公司如何部署生成式 AI 应用后，我发现许多相似之处...</li><li><a href="https://x.com/kwiens/status/1816128302542905620">来自 Kyle Wiens (@kwiens) 的推文</a>：嘿 @AnthropicAI：我知道你们渴望数据。Claude 确实很聪明！但你们真的需要在 24 小时内访问我们的服务器一百万次吗？你们不仅在不付费的情况下拿走我们的内容...</li><li><a href="https://x.com/wtgowers/status/1816509803407040909">来自 Timothy Gowers @wtgowers (@wtgowers) 的推文</a>：Google DeepMind 开发了一个程序，在某种意义上，它在今年的国际数学奥林匹克竞赛中达到了银牌水平。🧵 https://deepmind.google/discover/blo...</li><li><a href="https://x.com/martin_casado/status/1816298318215143901">来自 martin_casado (@martin_casado) 的推文</a>：我很震惊。对于在没有外生输入的情况下不断对数据语料库进行平均会导致信息质量下降这一点，我感到非常震惊。末日论者的衔尾蛇...</li>

rguments were always silly. But they get d...</li><li><a href="https://x.com/wtgowers/status/1816509808876597264">来自 Timothy Gowers @wtgowers (@wtgowers) 的推文</a>：主要的限制条件是，该程序需要比人类选手长得多的时间——对于某些问题超过了 60 小时——当然，处理速度也比可怜的人类快得多……</li><li><a href="https://x.com/kifleswing/status/1816542216678179083">来自 kif (@kifleswing) 的推文</a>：在 ChatGPT 最近的搜索引擎发布公告中，他们搜索了“8 月份北卡罗来纳州布恩市的音乐节”。在 ChatGPT 博客文章的示例图片中有五个结果：...</li><li><a href="https://x.com/lmsysorg/status/1816515251745214853">来自 lmsys.org (@lmsysorg) 的推文</a>：我们很高兴地宣布 SGLang Runtime v0.2 的里程碑版本发布，经过数月的努力，该版本包含了显著的推理优化。与...相比，它实现了高达 2.1 倍的吞吐量提升。</li><li><a href="https://x.com/openai/status/1816147248608403688?s=46">来自 OpenAI (@OpenAI) 的推文</a>：我们开发了 Rule-Based Rewards (RBRs)，以便在不需要大量人类数据收集的情况下安全地对齐 AI 行为，使我们的系统在日常使用中更安全、更可靠。https://openai.com/i...</li><li><a href="https://x.com/sama/status/1816551657158877187?s=46">来自 Sam Altman (@sama) 的推文</a>：我们认为搜索领域还有很大的提升空间。我们正在推出一个名为 SearchGPT 的新原型：https://openai.com/index/searchgpt-prototype/ 我们将从原型中学习，...</li><li><a href="https://x.com/prerationalist/status/1816504073115353116">来自 prerat (@prerationalist) 的推文</a>：正在发生 dot gif</li><li><a href="https://x.com/mononofu/status/1816496369512612341?s=46">来自 Julian Schrittwieser (@Mononofu) 的推文</a>：我们的最新工作 AlphaProof，基于 AlphaZero、LLMs 和 @leanprover 定理证明器，结合 AlphaGeometry 2，成功解决了 4 道 IMO 题目，并达到了银牌选手水平！🚀 更多信息请见...</li><li><a href="https://x.com/esyudkowsky/status/1816511787560546465?s=46">来自 Eliezer Yudkowsky ⏹️ (@ESYudkowsky) 的推文</a>：Paul Christiano 和我之前努力确定了具体的分歧；我们的一个标题是 Paul 认为“2025 年之前构建的 AI 在 IMO 中达到金牌水平”的概率为 8%，而...</li><li><a href="https://x.com/openai/status/1816536290822881780?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 OpenAI (@OpenAI) 的推文</a>：我们正在测试 SearchGPT，这是一个新 AI 搜索功能的临时原型，旨在为您提供快速、及时的答案，并附带清晰且相关的来源。我们正面向一小部分用户发布以获取反馈...</li><li><a href="https://x.com/hyhieu226/status/1816509696364397018?s=46">来自 Hieu Pham (@hyhieu226) 的推文</a>：这非常令人印象深刻。虽然是银牌，但几乎是金牌级别的银牌。看，AI 得到了 28 分，完整解决了 4 道题，而金牌的分数线是 29 分。如果 @GoogleDeepMind 尝试...</li><li><a href="https://x.com/cremieuxrecueil/status/1816532459393024052?s=46">来自 Crémieux (@cremieuxrecueil) 的推文</a>：在《Nature》上发表的一篇新论文发现，事实上，你不能在 AI 生成的数据上训练 AI 并期望它们继续改进。实际发生的情况是模型崩溃（model collapse），最终产生无意义的内容...</li><li><a href="https://x.com/RylanSchaeffer/status/1816535790534701304">来自 Rylan Schaeffer (@RylanSchaeffer) 的推文</a>：对于任何对模型崩溃（model collapse）感兴趣的人，我强烈建议大家看看我们的 COLM 2024 论文 https://arxiv.org/abs/2404.01413 当研究人员以特定方式故意诱导时，模型崩溃就会出现...</li><li><a href="https://x.com/googledeepmind/status/1816498082860667086?s=46">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：我们展示了第一个能以银牌选手水平解决国际数学奥林匹克（IMO）题目的 AI。🥈 它结合了 AlphaProof（一种用于形式化推理的新突破性模型）和 AlphaGeome...</li><li><a href="https://x.com/apples_jimmy/status/1816388658062373032?s=46">来自 Jimmy Apples 🍎/acc (@apples_jimmy) 的推文</a>：空气中弥漫着某种气息。一种充满希望的狂热氛围。让我们开始数学研究吧。</li><li><a href="https://x.com/sama/status/1816496304257941959?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Sam Altman (@sama) 的推文</a>：从现在开始，AI 的进展将是巨大的，AI 将成为一个关键的国家安全问题。我为《华盛顿邮报》写了一篇评论文章，阐述了为什么美国需要保持在 AI 开发方面的领先地位，而不是...</li><li><a href="https://x.com/datenschatz/status/1816567346242445644?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Datenschatz (@datenschatz) 的推文</a>：@casper_hansen_ 如果你相信 OpenAI 的视频，SearchGPT 为用户提供了在半月湾观察裸鳃类动物的两个具体日期，而 Perplexity 只是模糊地建议“冬季”...</li><li><a href="https://x.com/michael_nielsen/status/1816530386681470976">来自 Michael Nielsen 的推文</a>

et from Michael Nielsen (@michael_nielsen)</a>: 非常了不起：引用 Timothy Gowers @wtgowers (@wtgowers) 的话，Google DeepMind 开发的一个程序在某种意义上在今年的国际数学奥林匹克竞赛（IMO）中获得了银牌水平的表现...</li><li><a href="https://x.com/JeffDean/status/1816498336171753948">Tweet from Jeff Dean (@🏡) (@JeffDean)</a>: AI 系统在 IMO 中获得银牌水平的分数。国际数学奥林匹克竞赛（IMO）是面向年轻数学家历史最悠久、规模最大且最负盛名的竞赛。每年，各个国家...</li><li><a href="https://x.com/polynoamial/status/1816347598623834365?s=46">Tweet from Noam Brown (@polynoamial)</a>: 5 年前我们发布了 Pluribus，这是第一个超越人类的多人扑克 AI。它的训练成本仅为 150 美元。为什么扑克比围棋（Go）花费的时间更长？它是如何变得如此便宜的？答案是一个谨慎的...</li><li><a href="https://x.com/karpathy/status/1816531576228053133">Tweet from Andrej Karpathy (@karpathy)</a>: 参差不齐的智能（Jagged Intelligence）。我用这个词来描述一个（奇怪且不直观的）事实，即最先进的 LLMs 既能执行极其令人印象深刻的任务（例如解决复杂的数学问题），同时...</li><li><a href="https://archive.is/Jn5xv">Opinion | Sam Altman: AI&#x2019;s future must be democratic - The Washington&#x2026;</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1265754511478226997)** (2 messages): 

> - `Structured Data Extraction`
> - `LlamaExtract`
> - `Pydantic Integration`
> - `LLM-powered ETL` 


- **结构化提取功能发布**：新版本在任何 **LLM 驱动的 ETL、RAG** 和/或 **agent** 流水中实现了结构化提取功能，包括对异步（async）和流式（streaming）功能的全面支持。
   - 用户可以定义一个 **Pydantic 对象**，并使用 `as_structured_llm(…)` 将其附加到他们的 LLM 上，以实现流线型部署。
- **推出用于数据提取的 LlamaExtract**：今天推出了 **LlamaExtract** 的早期预览版，这是一个用于从非结构化文档中提取结构化数据的托管服务。
   - 该服务允许用户从文档中推断出**可由人工编辑的 Schema**，从而根据用户定义的标准进行结构化提取。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1265776369141420052)** (98 条消息🔥🔥): 

> - `使用 MultiStepQueryEngine 的 OpenAI 调用`
> - `RAG 聊天机器人开发`
> - `更新知识图谱节点嵌入 (Node Embeddings)`
> - `Document Summary Index 错误`
> - `分块 (Chunking) 与三元组提取 (Triple Extraction) 的修改` 


- **使用 MultiStepQueryEngine 的 OpenAI 调用**: 用户报告在使用 `MultiStepQueryEngine` 时看到重复的 OpenAI 调用，引发了关于 Arize 等工具日志记录问题的讨论。
   - 尽管存在困惑，但已澄清并非实际的重复调用，一名成员强调结构化文本提取方面仍在取得进展。
- **RAG 聊天机器人开发**: 一位用户分享了升级之前构建的 RAG 聊天机器人的动机，并提供了 [GitHub repo](https://github.com/wadie999/Chat-Bot) 链接供参考。
   - 他们表示有兴趣增强其功能，因为他们在 RAG 变得非常流行之前就已经构建了该聊天机器人。
- **更新知识图谱节点嵌入**: 讨论了如何管理 `PropertyGraphIndex` 中过时的知识图谱节点嵌入，特别是当文档发生变化时。
   - 用户讨论了 `refresh_ref_docs` 方法的相关性，并寻求关于如何有效更新这些嵌入的澄清。
- **Document Summary Index 错误**: 有报告称在 `DocumentSummaryIndex` 运行期间出现错误，特别是在最近消息大小和复杂性发生变化之后。
   - 讨论了可编程错误，并建议在排查突然出现的 SystemExit 错误时，确保在执行期间传递了正确的参数。
- **分块 (Chunking) 与三元组提取 (Triple Extraction) 的修改**: 一位用户提出了一种在属性图代码中集成语义分块 (semantic chunking) 和三元组提取的方法，旨在增强实体提取的上下文。
   - 通过建议将文档块与元数据结合，他们旨在改进三元组提取，同时通过向量嵌入 (vector embeddings) 保持查询效率。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/wadie999/Chat-Bot">GitHub - wadie999/Chat-Bot</a>: 通过在 GitHub 上创建账户，为 wadie999/Chat-Bot 的开发做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/issues/5772">indices base refresh_ref_docs 未按预期工作 · Issue #5772 · run-llama/llama_index</a>: 版本 0.16.19 中相同的 refresh_ref_docs 无法被识别为相同。从调试来看，找不到原始文件的哈希值。existing_doc_hash 总是返回 none，所有文件都被插入...</li><li><a href="https://github.com/run-llama/llama_index/pull/14963">为 Ollama 重新添加 kwargs，由 logan-markewich 提交 · Pull Request #14963 · run-llama/llama_index</a>: 自从为 Ollama 添加了实际的构造函数后，我们需要 kwargs 来允许传递父类属性</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/#sub-classing-extractors">属性图索引 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.pinecone.io/guides/data/upsert-data)">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1266053059411181670)** (4 条消息): 

> - `监控 Llama Agents`
> - `使用 RAG 进行路径规划` 


- **监控 Llama Agents 文章受到好评**: 成员们讨论了一篇名为《监控 Llama Agents：利用 LlamaIndex 和 Portkey 解锁可见性》的文章，可以在[这里](https://medium.com/ai-advances/monitoring-llama-agents-unlocking-visibility-with-llamaindex-and-portkey-c2b15cb05d40)找到。
   - 一位成员指出这是一篇**很棒的文章**，强调了它的价值。
- **探索使用 RAG 进行路径规划**: 一位成员询问是否有人在**路径规划任务**上尝试过 RAG。
   - 他们发现使用 **graphRAG** 基于复杂数据库进行规划任务**非常有趣**。


  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1265754577849024522)** (70 条消息🔥🔥): 

> - `Cohere 概览`
> - `撰写研究论文`
> - `Langchain 的 ChatPromptTemplate` 


- **Cohere 提供语言模型解决方案**：Cohere 被视为与 OpenAI 类似的 LLM 提供商，专注于自然语言处理，开发者可以通过其 API [文档](https://docs.cohere.com/) 获取相关功能。
   - 他们的 API 允许创建诸如对话式 Agent 和摘要工具之类的应用，定价基于使用量而非订阅制。
- **分享撰写研究论文的技巧**：成员们讨论了大学导师的重要性，特别是对于那些初次撰写研究论文的人，并强调了 [Cohere For AI 社区](https://cohere.com/research) 等资源提供的支持。
   - Cohere For AI 为学术研究提供协作和指导机会，为新研究人员的起步提供助力。
- **关于 Langchain 的 optional_variables 的澄清**：Langchain 的 ChatPromptTemplate 中的 `optional_variables` 参数允许用户定义非必需变量，以实现更具适应性的 Prompt。
   - 虽然 `optional_variables` 提供了灵活性，但也有人对其与同样处理可选元数据的 `partial_variables` 之间的区别提出了疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/">Cohere Enterprise Group</a>：无描述</li><li><a href="https://share.hsforms.com/10OrjljwpQ52ILJA6ftENIwch5vw">Form</a>：无描述</li><li><a href="https://docs.cohere.com/docs/the-cohere-platform">The Cohere Platform - Cohere Docs</a>：无描述</li><li><a href="https://cohere.com/pricing">Pricing</a>：直接通过我们的 API 访问模型，以创建可扩展的生产工作负载。</li><li><a href="https://dashboard.cohere.com/playground/chat">Login | Cohere</a>：通过一个易于使用的 API 登录以访问先进的 LLM 和 NLP 工具。</li><li><a href="https://cohere.com/research">Cohere For AI (C4AI)</a>：Cohere For AI 是一个非营利性研究实验室，致力于解决复杂的机器学习问题。我们支持探索未知的基本研究，并专注于创造更多……
</li>
</ul>

</div>
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1265761352224346223)** (31 条消息🔥): 

> - `Mistral Large 2`
> - `Multi-token 预测`
> - `训练数据效率`
> - `Perplexity 问题`
> - `发布困惑` 


- **Mistral Large 2 树立新标杆**：据报道，[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 以 **123B 参数**和 **128k 上下文窗口**超越了 405B 参数的模型，使其非常适合长上下文应用。
   - 该模型支持多种语言和编程语言，专为高效的单节点推理而设计，其性能潜力引发了广泛关注。
- **探索 Multi-token 预测**：成员们对 **Multi-token 预测** 表示好奇，指出它在提高字节级模型的可行性和训练效率方面的潜力。
   - 大家对数据集中可能出现的指定 Token 预测的标注感到兴奋，这与相关论文中讨论的方法论相一致。
- **训练数据修改策略**：讨论围绕通过 **掩码简单词汇**（这些词汇不增加价值）来提高训练效率展开，类似于 Microsoft Rho 论文中的概念。
   - 成员们考虑了增强训练数据的策略，例如分析 Perplexity 点并使用标签增强上下文，以提升训练效果。
- **对 Mistral 发布内容的困惑**：关于 Mistral Large 与 Mistral Large 2 的发布细节存在困惑，成员们对其开源状态和改进声明提出了质疑。
   - 一些人对该模型与 Claude 3.5 等现有模型相比的相对性能指标，以及该模型最终是否会开源表示担忧。
- **关于各种模型性能的见解**：关于 **405B** 与 Nvidia 模型性能的讨论揭示了基础设施对推理速度影响的见解。
   - 成员们注意到硬件规格的差异可能会影响模型在实际应用中的效能。



**提到的链接**：<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>：今天，我们宣布推出 Mistral Large 2，这是我们旗舰模型的下一代。与前代相比，Mistral Large 2 在代码生成、数学和推理能力上有了显著提升……

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1265811939401924671)** (5 条消息): 

> - `AdamW 8-bit 优化`
> - `FSDP 和 Zero3 的挑战`
> - `405B 模型加载问题`
> - `QLoRA 效率` 


- **使用 DeepSpeed 进行 AdamW 8-bit 优化**：一位成员分享了他们在 Docker 上进行全量微调（full finetunes）时，更倾向于使用 **AdamW 8-bit** 和 **DeepSpeed Stage 2**。
   - 根据他们在社区中的经验，这种配置似乎非常有效。
- **使用 FSDP 和 Zero3 加载 405B 的挑战**：一位用户报告了在使用 **QLoRA** 配合 **FSDP** 或 **Zero3** 加载 **405B** 模型时遇到困难。
   - 他们对导致这些加载失败的具体问题表示不确定。
- **405B 在 8x80GB 上的理论负载能力**：有人指出，理论上 **405B** 模型应该能在 **8x80GB** 硬件上加载，特别是在使用 **QLoRA** 时。
   - 这提醒了在理想条件下该配置预期的处理能力。


  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1265944791359946833)** (2 条消息): 

> - `训练配置` 


- **关于在训练中指定 max_steps 的疑问**：一位成员询问了在 **max_step** 和 **num_epochs** 之间通过指定 **max_steps** 数量进行训练的逻辑。
   - 收到的回复是 *“你能换个方式描述你的问题吗？”*，表明对其原始提问存在困惑。
- **关于训练逻辑的澄清请求**：另一位成员要求对有关训练过程的问题进行澄清，寻求更明确的表述。
   - 这一讨论凸显了在技术咨询中清晰沟通以避免误解的重要性。


  

---



### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1265766074079514726)** (37 条消息🔥): 

> - `内核共享讨论`
> - `Tinygrad 缓存共享`
> - `Tinygrad 中的多重梯度`
> - `随机张量生成问题`
> - `NumPy 转换中的优化` 


- **内核共享增强 GPU 效率**：成员们讨论了在花费 GPU 时间搜索后共享最优内核（optimal kernels）的潜力，指出**点对点 (p2p) 内核共享**可以利用整个用户网络的努力。
   - *一些参与者承认，之前的讨论提到过 p2p 搜索以及共享 tinygrad 缓存的能力*。
- **需要支持多次反向传播**：有人强调，为了在 tinygrad 中实现神经网络势能（neural network potentials），需要一种一致的方法来多次进行反向传播（backpropagate）。
   - 一些成员表示，虽然 *为 backward 调用合并损失（combining losses）* 应该是可行的，但更好的解决方案是保留计算图以支持更复杂的梯度计算。
- **随机张量生成给出重复结果**：一位用户报告了在另一个函数内部重复调用 `get_random_sum()` 时出现的异常行为，由于 **TinyJit** 的输出覆盖，导致输出了相同的结果。
   - 建议在重复调用之前调用 `.numpy()` 可以解决此问题，从而确保每次函数调用都有唯一的输出。
- **NumPy 转换过程中的优化**：一位用户指出，通过在张量转换方法中移除 `.to('CLANG')`，他们成功将 NumPy 转换所需的时间从 6 秒减半至 3 秒。
   - 这种修改引发了关于底层正确性的疑问，但生成的 NumPy 数组经核实是准确的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/issues/2417>,">Issues · tinygrad/tinygrad</a>: 你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - Issues · tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad/pull/5701">由 scdunne 提交的用于力匹配问题的多重梯度 · Pull Request #5701 · tinygrad/tinygrad</a>: 为了在 tinygrad 中实现神经网络势能，我们需要一种一致的方法来多次进行反向传播。此测试展示了一个使用 PyTorch 实现此功能的最小示例...</li><li><a href="https://github.com/openai/spinningup/blob/20921137141b154454c0a2698709d9f9a0302101/spinup/algos/pytorch/ppo/ppo.py#L231">spinningup/spinup/algos/pytorch/ppo/ppo.py · openai/spinningup</a>: 一个帮助任何人学习深度强化学习的教育资源。 - openai/spinningup</li><li><a href="https://github.com/tinygrad/tinygrad/pull/2445">由 pgkt04 移除 toCPU 拷贝 · Pull Request #2445 · tinygrad/tinygrad</a>: 移除了 lib.py 中的拷贝，改为在 numpy 中拷贝。
</li>
</ul>

</div>

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1265776555448205352)** (14 条消息🔥): 

> - `Mistral-Large-Instruct-2407`
> - `Llama 3.1 output token max`
> - `Ubuntu installation instructions`
> - `GPT-4o-mini fine-tuning`
> - `Deepseek performance` 


- **Mistral-Large-Instruct-2407 提供更快的速度**：Mistral-Large-Instruct-2407 (128B) 比 **405B** 模型大约 **小 3 倍**，从而缩短了 **inference time**。
   - 这种缩减可能会吸引那些寻求 **高效模型** 的用户。
- **Llama 3.1 最大输出 token 数咨询**：一名成员询问了 **Llama 3.1** 的 **最大输出 token 数**，表明社区需要更多相关信息。
   - 了解这些限制可以优化用户使用 **Llama 3.1** 的体验。
- **对过时的 Ubuntu 安装说明的担忧**：讨论指出 **Ubuntu 的安装说明** 可能已经过时。
   - 有人指出当前的说明 **已不再适用**。
- **微调 GPT-4o-mini 以进行优化**：有人提出了关于在 **Open Interpreter** 框架内微调 **GPT-4o-mini** 以获得更好性能的问题。
   - 这一讨论反映了用户对利用现有 **免费微调额度** 的兴趣。
- **Deepseek coder 展示了令人期待的更新**：最近 **Deepseek** coder 的 **更新** 令人兴奋，并分享了极具潜力的性能指标。
   - **Deepseek** 每百万 token 14-28 美分的性价比被强调为一个显著优势。



**提及的链接**：<a href="https://github.com/OpenInterpreter/open-interpreter/issues">Issues · OpenInterpreter/open-interpreter</a>：计算机的自然语言界面。通过在 GitHub 上创建账户，为 OpenInterpreter/open-interpreter 的开发做出贡献。

  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1266044798607949825)** (6 条消息): 

> - `Shipping updates for 01`
> - `React Native/Expo app development`
> - `WatchOS custom case for 01`
> - `Interpreter on Rabbit device` 


- **重大发货公告即将发布**：7 月份将发布关于 **01** 发货以及所有制造进度和材料开源的重大公告。
   - 团队对社区表现出的耐心表示感谢，并承认更新等待时间较长。
- **快速可靠的 React Native 应用更新**：由 [Ben Xu](https://github.com/benxu3) 开发的新版本 **React Native/Expo app** 基于 **WebRTC**，承诺提高速度和可靠性。
   - 团队已获得 **Apple Developer account**，并正准备在 **Play Store** 和 **iOS Store** 上发布该应用。
- **01 的 WatchOS 定制外壳正在研发中**：**WatchOS 版 01** 正在开发中，并计划推出定制外壳与之配套。
   - 团队对这一新方向充满期待。
- **在 Rabbit 设备上使用 Interpreter 的困扰**：一位用户正试图弄清楚如何让 **Interpreter** 在他们几周前收到的 **Rabbit device** 上运行。
   - 尽管早在 1 月份就购买了该设备，但他们对缺乏实用功能表示沮丧。



**提及的链接**：<a href="https://github.co">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪 bug 和功能...

  

---

### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1265772889534365846)** (5 messages): 

> - `数据库复杂性`
> - `商业演示需求`
> - `OpenInterpreter 的解决方案`
> - `案例研究`
> - `实施概览` 


- **对数据库复杂性的担忧**：一位成员对针对**复杂数据库**（由于跨表 joins）的解决方案有效性表示怀疑，建议需要访问完整的 schema。
   - 同时也提到了“感谢分享，做得好”，对贡献表示赞赏。
- **寻求商业化演示文稿**：一位社区成员询问是否有针对该解释器的**商业化演示文稿**可用，例如 PPT 或 PDF。
   - 他们列出了涵盖从企业面临的挑战到 OpenInterpreter 提供的解决方案等主题的幻灯片。
- **OpenInterpreter 对商业挑战的解决方案**：幻灯片强调了 OpenInterpreter 如何通过简化编码和自动化任务，旨在解决**高昂的人工成本**和**可扩展性**等主要商业挑战。
   - 重点放在提高生产力和减少对资深程序员的依赖上。
- **实施的成功案例**：该成员建议在演示文稿中加入**案例研究和证言**，以展示 OpenInterpreter 的成功实施。
   - 他们强调了真实案例对于说明解决方案有效性的重要性。
- **展示实施步骤**：演示幻灯片包括一份**实施概览**，详细说明了集成步骤、培训选项以及采用 OpenInterpreter 的时间表。
   - 这旨在指导利益相关者如何在工作流中有效地采用和利用该解释器。


  

---



### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1265914266519601173)** (6 messages): 

> - `Llama 3/3.1 70B 生成配方`
> - `多 GPU 推理`
> - `量化技术`
> - `FSDP 集成` 


- **Llama 3/3.1 70B 生成脚本咨询**：一位用户询问是否存在支持跨多 GPU 分布式生成的 **Llama 3/3.1 70B** 生成配方 (recipe)。
   - 另一位成员指出，目前不支持开箱即用的分布式生成，并建议查看[此 GitHub 仓库](https://github.com/huggingface/llm-swarm)以获取更多信息。
- **单 GPU 适配问题**：用户表达了对在单张 GPU 上使用 bfloat16 运行 **Llama 3 70B 模型**的担忧，并询问解决方案。
   - 一位成员回应并强调了将模型量化为 **int4** 以进行单 GPU 推理等选项。
- **Torchtune 多 GPU 支持的现状**：另一位参与者指出，Torchtune 尚未优先考虑多 GPU/分布式推理，但他们正在研究中。
   - 他们还提到，多 GPU 推理支持的开发正在 [torchchat](https://github.com/pytorch/torchchat) 库中进行。
- **转向分布式生成脚本**：一位成员强调，对于熟悉 FSDP 的人来说，现有的 **generate.py** 脚本可以通过一些调整转换为 **generate_distributed.py** 配方。
   - 他们建议可以利用分布式微调配方中的代码来辅助这种适配。


  

---

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1265773946024890538)** (9 条消息🔥): 

> - `Llama 3.1 更新`
> - `微调中的内存管理`
> - `关于 Cross Attention 的 RFC`
> - `使用 Snowflake 进行内存优化`
> - `模型中的新变换 (Transformations)` 


- **Llama 3.1 进度接近完成**：成员们讨论了他们正在完成 **Llama 3.1** 补丁的测试，重点是在单节点上集成 **405B QLoRA**。
   - 一位成员指出，虽然 Recipe 能够运行，但为此类大型模型保存 Adapter 的 Checkpoint 极具挑战性。
- **Snowflake 模型微调指南**：一位成员分享了一篇 [博客文章](https://www.snowflake.com/engineering-blog/fine-tune-llama-single-node-snowflake/)，详细介绍了针对 **Llama 3.1** 等大模型微调的优化方案。
   - 他们提到在 **A100** 上内存使用峰值约为 **66GB**，由于缺乏 **FP8** 内核，他们从 **bfloat16** 版本开始。
- **关于 FP8 和内存使用的澄清**：一位成员寻求关于 **FP8** 是否严格应用于基础权重的澄清，并指出由于其 **QLoRA** Recipe 中使用了 **NF4** 量化，内存需求应该更低。
   - 这表明他们期望优化能直接对内存效率产生积极影响。
- **TransformerDecoderLayer 修改的 RFC**：分享了一个新的 **RFC**，旨在支持多模态架构的 **Cross Attention**，这需要对 **TransformerDecoderLayer** 进行更改。
   - 成员们收到警告，由于 [Pull Request](https://github.com/pytorch/torchtune/pull/1224) 中概述的重大库变更，现有的自定义模型构建器将需要更新。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.snowflake.com/engineering-blog/fine-tune-llama-single-node-snowflake/">使用 Snowflake 的 AI 技术栈在单节点上微调 Llama 3.1 405B</a>: 了解 Snowflake AI Research 如何利用创新的内存管理技术优化 Meta Llama 3.1 405B 等海量 LLM 的微调，从而实现高效的 AI 部署。</li><li><a href="https://github.com/pytorch/torchtune/pull/1224">[RFC] 由 pbontrager 发起的 TransformerDecoderLayer 重构 · Pull Request #1224 · pytorch/torchtune</a>: [RFC] TransformerDecoderLayer 重构。重构 TransformerDecoder 以便其可用于多模态架构。摘要：将 TransformerDecoderLayer 替换为 TransformerSelfAttention 和 Transforme...
</li>
</ul>

</div>
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/)** (1 条消息): 

adiptamartu: Whisper Speech 模型支持印尼语吗？@here 谢谢提供信息
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1265776788148326451)** (10 条消息🔥): 

> - `Mistral Large 2`
> - `DFT Vision Transformer Architecture`
> - `Rotary Position Encoding`
> - `Complex Number Parameters`
> - `Normalization Techniques` 


- **Mistral Large 2 突破边界**：[Mistral Large 2](https://mistral.ai/news/mistral-large-2407/) 拥有 **128k 上下文窗口**，支持十多种语言，增强了 AI 应用构建能力。
   - 它拥有 **1230 亿参数**，专为长上下文应用的**单节点推理**设计，提供极高的吞吐量。
- **DFT Vision Transformer 的创新**：开发了一种在每个模块中使用**傅里叶变换 (Fourier transform)、MLP 和逆傅里叶变换 (inverse Fourier transform)** 的新架构，专注于保持图像质量。
   - 该设计结合了**全图像归一化层 (image-wide norm layers)**，在不产生信息瓶颈的情况下进行归一化。
- **利用复数参数 (Complex Number Parameters)**：整个 DFT Vision Transformer 网络使用**复数参数**运行，增强了其计算动态。
   - 该架构允许**旋转位置编码 (Rotary Position Encoding)** 的简洁集成，提高了效率和性能。
- **旋转位置编码的效果**：切换到**旋转位置编码**后，观察到**损失曲线下降率**有显著改善。
   - 这种变化被描述为“令人满意”，表明其对整体训练过程产生了积极影响。
- **流线型架构结构**：DFT Vision Transformer 采用由等大模块组成的**直线流水线**，最后以全局平均池化 (global average pool) 和线性层 (linear layer) 结束。
   - 该设计确保**图像永远不会被下采样**，始终保留所有可用信息。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-large-2407/">Large Enough</a>: 今天，我们发布了 Mistral Large 2，这是我们旗舰模型的下一代。与前代相比，Mistral Large 2 在代码生成、数学和推理方面的能力显著提升...</li><li><a href="https://github.com/mlfoundations/MINT-1T">GitHub - mlfoundations/MINT-1T: MINT-1T: A one trillion token multimodal interleaved dataset.</a>: MINT-1T：一个包含一万亿 token 的多模态交错数据集。 - mlfoundations/MINT-1T
</li>
</ul>

</div>
  

---



### **DSPy ▷ #[papers](https://discord.com/channels/1161519468141355160/1203568372667645963/1265769535294800034)** (7 条消息): 

> - `SymbolicAgentLearner 研发`
> - `GitHub 共享计划` 


- **SymbolicAgentLearner 结合了 RAG 和符号学习**：一名成员使用 DSPy 开发了 **SymbolicAgentLearner**，它集成了**检索增强生成 (RAG)** 和符号技术，用于回答问题并创建带有引用的详细段落。
   - 核心功能包括一个 **SymbolicLearningProcedure** 类，该类执行多跳检索并生成自动添加引用的文本，增强了信息的深度。
- **公共 GitHub 仓库计划**：在询问有关共享项目的 GitHub 仓库后，提到当前代码库是私有的，但计划创建一个**新的公共仓库**。
   - 此举旨在让社区中的其他人能够访问所开发的精华内容和技术。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1266061154220769414)** (1 条消息): 

> - `litellm proxy`
> - `跨模型的函数调用` 


- **litellm 代理运行完美**：一位成员建议在所有模型中使用 **litellm proxy**，并将 OpenAI 的 `api_base` 指向它，效果**非常出色**。
   - 这种解决方法实现了与 DSPy 的无缝集成。
- **跨模型的函数调用 (Function Calling) 需要额外工作**：该成员提到他们成功实现了跨模型的**函数调用**，但需要**相当多的变通方法**。
   - 未详细说明所使用的具体方法。


  

---

### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1265796972003070032)** (1 条消息): 

> - `News categorization`
> - `GPT-3.5-turbo`
> - `MIPRO`
> - `ColBERTv2`
> - `F1 score` 


- **DSPy 驱动新闻分类程序**：一个实现**新闻分类系统**的程序使用 DSPy 将文章分类为**“虚假”或“真实”**，该系统采用了 OpenAI 的 **GPT-3.5-turbo** 模型和 **ColBERTv2** 进行检索，并结合了 **Chain of Thought** 方法。
   - 它利用 **MIPRO** (Minimum Prompt Optimization) 进行提示词优化，并集成了自定义的 **F1 score** 计算进行评估。
- **新闻分类的新进展**：该程序引入了一种利用先进模型评估新闻文章的**新方法**，从而提高了分类准确性。
   - 此类实现展示了在过滤虚假信息中集成 **AI models** 的潜力。


  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1265966597516955688)** (7 条消息): 

> - `LangChain Agents Consistency Issues`
> - `Working with Multi Agents`
> - `Using ConversationSummary with Database Agents`
> - `LangChain and Ollama Video Release`
> - `LangGraph Persistence Options` 


- **LangChain Agents 面临一致性问题**：一位用户对使用开源模型的 **LangChain agents** 表示沮丧，原因是存在**一致性问题**且经常选错工具。
   - 另一位成员表示赞同，称他们所有的测试在本地 **LLM** 性能方面都显示出类似的结果。
- **多 Agent 功能探索**：一位用户询问了关于多 **agents** 协作的问题，寻求实现方面的见解或指导。
   - 社区成员通过询问正在探索的具体功能来推动进一步讨论。
- **关于 ConversationSummary 集成的咨询**：一位用户询问是否可以在自己的**数据库 agent** 中使用 **ConversationSummary**，并寻求实现建议。
   - 他们表示，如果不支持直接使用，渴望获得反馈或替代方案。
- **LangChain 与 Ollama 极具前景的新视频**：一位成员分享了名为“使用 Ollama 实现完全本地化的工具调用”的 **YouTube 视频**，讨论了本地 **LLM** 工具调用的潜力。
   - 他们指出，该视频解决了关于 **Agent** 中**工具选择**和**一致性使用**的常见误解。
- **LangGraph 持久化选项更新**：一位用户询问了除 **SqliteSaver** 之外，关于 **LangGraph persistence** 机制的任何更新。
   - 他们正在寻找 **LangGraph** 中数据存储选项的替代方案或改进。



**提到的链接**：<a href="https://www.youtube.com/watch?v=Nfk99Fz8H9k">Fully local tool calling with Ollama</a>：工具是可被 **LLM** 调用的实用程序（例如 API 或自定义函数），赋予模型新的能力。然而，**LLM** 需要能够 1) 选择...

  

---



### **AI Stack Devs (Yoko Li) ▷ #[ai-raspberry-pi](https://discord.com/channels/1122748573000409160/1234912245415280742/)** (1 条消息): 

felixultimaforeverromanempire: 这很酷，再多跟我们分享一些。
  

---



---



---



---



---



---



---



{% else %}


> 完整的频道细分内容已针对电子邮件进行截断。
> 
> 如果您想查看完整内容，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}