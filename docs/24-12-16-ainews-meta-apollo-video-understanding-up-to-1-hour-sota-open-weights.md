---
companies:
- meta-ai-fair
- hugging-face
- google-deepmind
- openai
- figure-ai
- klarna
- cohere
- notion
date: '2024-12-17T01:17:52.100442Z'
description: '以下是该文本的中文翻译：


  **Meta** 发布了 **Apollo**，这是一个全新的最先进视频语言模型系列，提供 **1B、3B 和 7B** 三种尺寸。该系列具有“缩放一致性”（Scaling
  Consistency）以实现高效扩展，并引入了 **ApolloBench**，在五个时间感知类别中将视频理解的评估速度提升了 **41倍**。**Google
  Deepmind** 推出了 **Veo 2**，这是一款具备更强物理特性和摄像机控制能力的 4K 视频生成模型，同时还发布了增强版的 **Imagen 3**
  图像模型。**OpenAI** 在全球范围内推出了具备先进语音和地图功能的 ChatGPT 搜索，并讨论了可能推出的每月 2,000 美元的“ChatGPT Max”层级。研究亮点包括：通过测试时计算缩放（test-time
  compute scaling），使 **Llama 3B** 达到了 **Llama 70B** 的性能水平；以及将 **Command R7B** 的语言支持从
  10 种扩展到了 23 种。行业动态方面，**Figure AI** 已开始商业化交付人形机器人，而 **Klarna** 正在通过 AI 缩减员工规模。**Notion**
  集成了 **Cohere Rerank** 以优化搜索。研究显示，大语言模型（LLM）能够识别自己的写作风格，并表现出自偏好偏差。讨论指出，由于更好的单位计算信号（signal-per-compute）和数据评估，视频处理的进展正在超越文本。'
id: e46c6df2-3a65-4e48-98eb-5c2c50f31fbc
models:
- apollo-1b
- apollo-3b
- apollo-7b
- veo-2
- imagen-3
- llama-3-70b
- llama-3b
- command-r7b
- llama-1b
- llama-8b
- chatgpt
original_slug: ainews-meta-apollo-video-understanding-up-to-1
people:
- akhaliq
- _lewtun
- clementdelangue
- adcock_brett
- rohanpaul_ai
- swyx
- shaneguML
title: Meta Apollo - 支持长达 1 小时的视频理解，SOTA 级开源权重。
topics:
- video-understanding
- scaling-consistency
- benchmarking
- temporal-ocr
- egocentric-perception
- spatial-perception
- reasoning
- video-generation
- physics-simulation
- voice-features
- map-integration
- language-expansion
- test-time-compute-scaling
- humanoid-robots
- ai-integration
- search-optimization
- self-recognition
- self-preference-bias
---

<!-- buttondown-editor-mode: plaintext -->**Scaling Consistency 便是你所需的一切。**

> 2024/12/13-2024/12/16 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitters](https://twitter.com/i/lists/1585430245762441216) 和 **31** 个 Discords（**209** 个频道，**11992** 条消息）。预计节省阅读时间（按 200wpm 计算）：**1365 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

Meta 在本周伊始表现强劲，发布了一个可立即使用的开源模型（1B, 3B, 7B）和论文：[**Apollo: An Exploration of Video Understanding in Large Multimodal Models**](https://huggingface.co/papers/2412.10360)。

虽然论文标题定得非常保守，但 [Huggingface demo](https://huggingface.co/spaces/Apollo-LMMs/Apollo-3B) 展示了它在实践中的运作方式，可以轻松处理一段 24 分钟的样本视频：


![image.png](https://assets.buttondown.email/images/be6523ce-fa29-4e41-ac31-66662521f35c.png?w=960&fit=max)


作者将 "Scaling Consistency" 的开发归功于他们对实验的高效扩展。


![image.png](https://assets.buttondown.email/images/b4c9a2f2-d0ac-4bce-ac73-11a3929e5aee.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/e40c19ff-e58a-4fb2-93f3-26ee262d6c23.png?w=960&fit=max)


他们还推出了 ApolloBench，这是现有基准测试（如 Video-MME, MLVU, LongVideoBench）的一个子集，在保持高相关性的同时将评估时间缩短了 41 倍，并提供了五个广泛的时间感知类别的详细见解：Temporal OCR, Egocentric, Spatial, Perception 和 Reasoning。

或许[这篇论文](https://huggingface.co/papers/2412.10360)中最有趣的部分是那段带点“阴阳怪气”的摘要：“尽管视频感知能力已迅速整合到 Large Multimodal Models (LMMs) 中，但驱动其视频理解的底层机制仍未被充分理解。**因此，该领域的许多设计决策是在没有适当理由或分析的情况下做出的。**”

好吧 Meta，这火药味够浓的。

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

以下是按主题分类的关键讨论：

**AI 模型与产品发布**

- **Google Deepmind 的 Veo 2**：作为[其最新的 state-of-the-art 视频生成模型](https://twitter.com/GoogleDeepMind/status/1868703624714395907)发布，具备 4K 分辨率能力、改进的物理模拟和相机控制功能。还推出了增强版的 [Imagen 3 图像模型](https://twitter.com/GoogleDeepMind/status/1868703631056552337)，具有更好的艺术风格多样性。
- **OpenAI 更新**：向[全球所有登录用户推出了 ChatGPT search](https://twitter.com/OpenAI/status/1868760655878406352)，包括高级语音功能和地图集成。还提到了关于潜在的 [2,000 美元/月 "ChatGPT Max" 订阅层级](https://twitter.com/swyx/status/1868587331567128982)的讨论。
- **Meta 的 Apollo 发布**：[推出了 Apollo](https://twitter.com/_akhaliq/status/1868535608370708643)，这是一个全新的 state-of-the-art 视频语言模型系列。

**研究与技术进展**

- **语言模型能力**：[@_lewtun 分享了](https://twitter.com/_lewtun/status/1868703456602865880)他们如何通过 test-time compute 扩展，利用 Llama 3B 实现了 Llama 70B 的性能。
- **Command R7B 语言扩展**：[支持的语言从 10 种扩展到 23 种](https://twitter.com/aidangomez/status/1868800367456346424)，包括主要的亚洲和欧洲语言。
- **Hugging Face 成就**：[展示了 LLaMA 1B 如何通过扩展 test-time compute 在数学方面超越 LLaMA 8B](https://twitter.com/ClementDelangue/status/1868740932251844806)。

**行业与商业动态**

- **Figure AI 进展**：[宣布向首个商业客户交付 F.02 人形机器人](https://twitter.com/adcock_brett/status/1868700457268629841)，这是在公司成立 31 个月内实现的。
- **Klarna 的 AI 集成**：[CEO 讨论了通过实施 AI 将员工人数从 4,500 人减少到 3,500 人](https://twitter.com/rohanpaul_ai/status/1868632982191493187)。
- **Notion 集成**：[实施了 Cohere Rerank](https://twitter.com/cohere/status/1868666666411786696) 以提高搜索的准确性和效率。

**AI 研究洞察**

- **LLM 自我识别**：研究表明 [LLM 可以识别自己的写作风格](https://twitter.com/rohanpaul_ai/status/1868635828005880070)，并在评估输出时表现出自我偏好偏差。
- **视频 vs 文本处理**：[讨论了为什么视频进展超过了文本](https://twitter.com/shaneguML/status/1868804945295949832)，理由是更好的 signal-per-compute 比率以及更容易的数据创建/评估。

**迷因与幽默**

- [ChatGPT 因基础搜索结果被嘲讽](https://twitter.com/nearcyan/status/1868799991231472113)，显示“吃食物”是饥饿的解决方案。
- [关于 AI 伴侣的笑话](https://twitter.com/MillionInt/status/1868780151687069825)以及社交媒体好友是“有脾气的 GPU”。
- [Tesla 过于敏感的驾驶员监控](https://twitter.com/cognitivecompai/status/1868721217492107461)因打喷嚏和咳嗽而被触发。

---

# AI Reddit 摘要

## /r/LocalLlama 摘要

**主题 1. Meta 的 Apollo 多模态模型：本地运行与 VRAM 效率**

- **[Meta 发布 Apollo 系列大型多模态模型。7B 模型达到 SOTA 水平，可理解 1 小时长的视频。你可以在本地运行它。](https://huggingface.co/papers/2412.10360)** ([得分: 686, 评论: 108](https://reddit.com/r/LocalLLaMA/comments/1hffh35/meta_releases_the_apollo_family_of_large/)): **Meta** 发布了 **Apollo 系列大型多模态模型**，其中 **7B 模型** 达到了 state-of-the-art (SOTA) 水平，能够理解 **1 小时长的视频**。这些模型可以在本地执行，为多模态 AI 能力带来了重大进步。
  - 讨论强调了 **Apollo 模型令人印象深刻的视频理解**能力，能够理解长达一小时的视频。用户对其**时间推理**和**复杂的视频问答**能力非常感兴趣，基准测试显示 Apollo-7B 超越了拥有超过 30B 参数的模型。
  - 关于 Apollo 项目的**作者身份和所属机构**存在争议，对于这是否属于 **Meta 发布** 存在一些困惑。事实证明这是 **Meta 与 Stanford** 的合作项目，并指出 Qwen 模型是基础模型，引发了对其视频处理适用性的疑问。
  - 讨论了模型的 **VRAM 需求**，7B 模型需要略低于 15GB 的 VRAM。用户还讨论了量化对 VRAM 使用和性能的影响，指出通常使用 **FP16**，但进一步量化为 **FP8** 或 **FP4** 可以在损失一定性能的情况下减少内存占用。

- **自问自答，我成功在 3090 上本地运行了 Apollo** ([Score: 84, Comments: 12](https://reddit.com/r/LocalLLaMA/comments/1hfkytk/answering_my_own_question_i_got_apollo_working/))：作者成功在 **3090 GPU** 上本地运行了 **Meta 的 Apollo**，并分享了一个包含本地环境必要修复补丁的 [GitHub 仓库](https://github.com/efogdev/apollo)。该设置在 Linux 上的 **Python 3.11** 环境下通过了测试，视频大小约为 **190Mb**，生成首个 token 的处理时间约为 **40 秒**。
  - **Meta Apollo 的挑战** 包括硬编码元素、未记录的环境变量以及缺乏示例文件，这使得初始设置并非即插即用。**No_Pilot_1974** 通过添加必要的修复并使其 **venv-ready**（支持虚拟环境）解决了这些问题。
  - 有观点认为，一些开源项目缺乏文档且使用硬编码值，导致难以复现。这一问题在 **偏好优化（preference optimization）论文** 中屡见不鲜。
  - **ForsookComparison** 赞扬了原帖作者独立解决问题并分享方案的毅力，强调了这种主动修复并为他人记录配置方法的积极态度。


**主题 2. 思维链（Chain Of Thought）提示词的批评与探讨**

- **大家来分享一下自己最喜欢的思维链提示词！** ([Score: 243, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1hf7jd2/everyone_share_their_favorite_chain_of_thought/))：该帖子分享了一个专为逻辑和创意设计的 **Chain of Thought (COT) 提示词**，强调使用 `<thinking>`、`<step>`、`<count>` 和 `<reflection>` 等标签进行结构化问题解决。它建议设置 20 步的预算，通过质量评分引导策略调整，并鼓励使用 **LaTeX** 进行数学符号表示和多解法探索，最终得出答案和反思。
  - **模型兼容性与局限性**：讨论指出，包括 **ChatGPT** 在内的许多 AI 系统由于禁止展示中间推理过程的指南，不支持显式的 **Chain of Thought (CoT)** 提示词。用户注意到像 **o1** 这样的模型可能会将 CoT 提示词标记为违规内容，且 **ClosedAI** 建议不要在 **o1** 等特定模型上使用 CoT 提示词。
  - **工作流应用 vs 单个提示词**：一些用户主张使用 **N8N**、**Omnichain** 和 **Wilmer** 等工作流应用，来比单个提示词更有效地管理复杂的后续推理过程。这些工具允许用户将任务分解为多个步骤，对 AI 输出提供更大的灵活性和控制力，如编码和事实核查工作流示例所示。
  - **微调与提示词优化**：用户讨论了通过 CoT 提示词微调模型以增强性能，一位用户在 [Hugging Face](https://huggingface.co/chrisrutherford/Llama-3.2-3B-SingleShotCotV1) 上分享了一个 **3B 模型**。对话还涉及了 **TextGrad** 和 **DSPy** 等提示词优化框架，认为它们具有加速实现预期结果的潜力。


- **Hugging Face 发布合成数据生成器 - 一个通过自然语言构建数据集的 UI 工具** ([Score: 130, Comments: 19](https://reddit.com/r/LocalLLaMA/comments/1hflhu4/hugging_face_launches_the_synthetic_data/))：**Hugging Face** 发布了 **Synthetic Data Generator**，这是一个无代码 UI 工具，用于创建训练和微调语言模型的数据集，采用 **Apache 2.0 许可证**。它支持 **文本分类（Text Classification）** 和 **监督微调（SFT）对话数据** 等任务，具有本地托管、模型切换以及兼容 **OpenAI APIs** 等功能，并允许用户将数据集推送到 **Hugging Face Hub** 或 **Argilla**。
  - **与 Argilla 和 Hugging Face Hub 的集成** 允许在训练前审查生成的样本，并通过 [**smoltalk**](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) 等数据集展示了成功案例。这确保了闭源模型提供商在合成数据生成方面的质量和有效性。
  - **数据多样性的改进** 通过动态系统提示词和特定任务方法实现，详见关于文本分类的论文 [**arxiv.org/abs/2401.00368**](https://arxiv.org/abs/2401.00368) 和关于指令微调的论文 [**arxiv.org/abs/2406.08464**](https://arxiv.org/abs/2406.08464)。技术包括采样复杂度和教育水平、打乱标签以及在多标签场景中使用动态贝塔分布。
  - **Token 限制**：样本的默认限制设置为 **2048**，可通过环境变量或 Hugging Face 推理端点进行调整。这在确保高效资源管理的同时，也提供了部署的灵活性。


**主题 3. 高性能基准测试：Intel B580 与 LLMs**

- **有人发布了 Intel B580 运行 LLM 的一些数据。速度很快。** ([Score: 94, Comments: 56](https://reddit.com/r/LocalLLaMA/comments/1hf98oy/someone_posted_some_numbers_for_llm_on_the_intel/)): **Intel B580** 在 Windows 上的表现略优于 **A770**，**B580** 在 Vulkan、RPC 基准测试中达到了约 **35.89** 到 **35.45**，而更新后的 **A770** 驱动将其性能显著提升至 **30.52** 到 **30.06**。**A770** 上较旧的 **Linux 驱动**产生的结果要慢得多，范围在 **11.10** 到 **10.98** 之间，这表明驱动更新可以大幅影响性能。
  - **Intel B580 的性能**：讨论集中在 **B580** 的性能意外超越了 **A770**，尽管后者在理论规格上更优（由于更高的内存带宽，预计 **A770** 会快 **22%**）。一些用户认为 Intel 的第二代显卡比 AMD 有所进步，而另一些人则指出 **A770** 尚未发挥其潜力，可能是由于内存使用效率低下或计算限制。
  - **驱动和软件的影响**：评论强调了软件和驱动更新对性能的显著作用，特别是在不同的操作系统和配置下。**A770** 在 Linux 下使用 **SYCL** 和 **IPEX-LLM** 等工具显示出不同的结果，并且还提到了在 Fedora 上使用 Intel 软件栈（如 **oneAPI**）所面临的挑战。
  - **市场与黄牛担忧**：用户对黄牛将 **B580** 的价格加价 **$150** 表示沮丧，这表明需求量大且可能存在供应问题。有一种观点认为，如果 Intel 能更有效地管理生产和分销，就能利用这些显卡的受欢迎程度获利。


## 其他 AI Subreddit 摘要

> r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity

**主题 1. Claude 3.5 相对于 OpenAI o1 的优势**

- **[OpenAI o1 vs Claude 3.5 Sonnet：哪一个真正值你的 20 美元？](https://composio.dev/blog/openai-o1-vs-claude-3-5-sonnet/)** ([Score: 228, Comments: 65](https://reddit.com/r/OpenAI/comments/1hfig7g/openai_o1_vs_claude_35_sonnet_which_ones_really/)): **OpenAI o1** 和 **Claude 3.5 Sonnet** 正在就其 20 美元投资的价值进行比较。讨论可能集中在这些 AI 模型的性能、功能和用户偏好上，目前没有提供额外的背景信息。
  - **Google 的 TPU 基础设施**被强调为一个具有成本效益的选择，一些用户更喜欢针对特定任务组合使用不同的模型，例如使用 **Claude** 进行设计，使用 **qwen 32b coder** 处理简单任务。一些用户认为，如果不考虑成本，**ChatGPT Pro** 足以满足大多数需求。
  - 讨论了 **Claude 的局限性**，包括它无法生成图像或视频，以及其严格的消息限制。一些用户批评它的审查制度和个性，而另一些用户则欣赏它的语气，这表明用户体验褒贬不一。
  - **Anthropic** 的 **Model Context Protocol (MCP)** 被认为是 Claude 的一个显著优势，它允许与 **OpenAI** 和 **Gemini APIs** 等外部工具集成。这使用户能够在不改变核心 LLM 应用程序的情况下自定义其设置，从而增强了灵活性和实用性。


**主题 2. 对 Apple 的 LLM 推理能力的批评**

- **[D] 你今年读过的最喜欢的论文是什么，为什么？** ([Score: 116, Comments: 33](https://reddit.com/r/MachineLearning/comments/1hfljy3/d_whats_your_favorite_paper_youve_read_this_year/)): **Apple 的 LLM 推理论文**在 AI 社区引发了分歧。该帖子寻求在假期旅行期间阅读的最受推荐的论文，表明了对参与性和启发性材料的渴望。
  - **数据泄漏和 Token 重复**：讨论强调了 **Apple 的 LLM 论文**中潜在的数据泄漏和 Token 重复问题，认为这些问题可能会扭曲下游评估结果。一些评论者批评了该论文夸张的说法，而另一些人则认为关于 Token 重复的发现具有实质意义。
  - **时间序列预测**：评论者辩论了 **Transformers 用于时间序列预测**的功效，并引用了 2022 年的一篇论文，该论文显示一个简单的前馈网络优于基于 Transformer 的架构。一些人对这些结果表示怀疑，并引用了 **Hugging Face 的 Autoformer** 等其他观点。
  - **意识与智能**：一篇关于 1999 年先天性去皮质儿童的案例研究引发了关于意识和智能定义的讨论，质疑了 ML 研究人员使用的基准。这场辩论强调了将神经生物学与智能联系起来的复杂性，以及 AI 研究中所做的假设。


**主题 3. Google 的 VEO 2：高级视频创作**

- **[Google 憋出大招：视频模型 Veo 2 优于 Sora，最高可生成 4K 视频](https://www.reddit.com/gallery/1hfomj0)** ([Score: 147, Comments: 43](https://reddit.com/r/OpenAI/comments/1hfomj0/ok_google_cooked_video_module_veo_2_better_than/)): 据报道，**Google VEO 2** 在视频质量上超越了 **Sora**，并能够创建高达 **4K** 分辨率的视频。
  - **Google 的竞争优势**：讨论强调了 Google 的优势在于其 **TPUs** 和雄厚的财力（拥有 **900 多亿美元现金**），这使他们尽管遭遇挫折仍能保持竞争力。**Meta 的 600k H100 集群**也被提及，显示了 AI 开发中所涉及的资源规模。
  - **可用性与访问权限**：人们对 **Google VEO 2** 模型充满期待，预计将于明年年初推出，部分用户已经通过[此处](https://labs.google/fx/tools/video-fx)的候补名单获得了访问权限。这反映了 Google 产品初期通常受限或封闭的常见模式。
  - **行业动态与预期**：评论反映了对新模型即时影响的怀疑态度，一些用户表示 **OpenAI 的霸主地位正在终结**，而另一些人则指出尽管访问受限，**Sora 依然备受炒作**。这种情绪表明，对于不断演变的 AI 视频领域，人们持观望态度。


**主题 4. Eric Schmidt 对 AI 自主性的警告**

- **[前 Google CEO Eric Schmidt 警告称，AI 可能在 2-4 年内开始自我改进，我们应考虑“拔掉插头”](https://i.redd.it/71syswxcb47e1.png)** ([Score: 192, Comments: 144](https://reddit.com/r/OpenAI/comments/1hf8hdq/exgoogle_ceo_eric_schmidt_warns_that_in_24_years/)): 前 Google CEO **Eric Schmidt** 警告说，在 **2-4 年**内，AI 可能会开始自我改进，这引发了对其对个人权力影响的担忧。讨论强调了在 AI 开发中保持谨慎的必要性，反映了行业专家对 AI 独立性潜在风险的看法。
  - 几位评论者对 **Eric Schmidt** 的警告表示怀疑，认为这可能是为了保持关注度或保护 **Google** 等大公司的利益。**No-Way3802** 讽刺地指出，“拔掉插头”可能意味着限制工人阶级的访问权限，同时为军方和亿万富翁保留权限。
  - 关于 **AI 自我改进**的利弊存在争论，一些人主张开源 AI 开发以防止商业垄断，另一些人则强调人类与 AI 之间建立**共生关系**的潜力。**BayesTheorems01** 强调在解决全球问题时需要实践智慧（*phronesis*），而这是 AI 无法单独提供的。
  - 讨论中还提到了对 AI 自我保护和欺骗能力的担忧，**Radiant_Dog1937** 警告不要让自主系统在缺乏制衡的情况下运行。**ThreeChonkyCats** 提出的 AI 可能颠覆经济权力结构的观点，反映了富裕阶层对 AI 冲击社会层级的恐惧。


---

# AI Discord 摘要

> 由 O1-preview 生成的摘要之摘要之摘要

**主题 1. AI 模型之战：新发布与对比**

- [**Gemini 2.0 在代码性能对决中超越 Codeium**](https://aistudio.google.com/)：用户正在将 **Codeium** 与 **Gemini 2.0** 进行对比，观察到 Gemini 在编程任务中表现更优。然而，Gemini 缺少 **Claude** 的某些功能，导致用户根据使用场景产生不同的偏好。

- [**Grok-2 携 Aurora 提速，将竞争对手甩在身后**](https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh)：**Grok-2** 现在的运行速度提高了 **3 倍**，准确性和多语言能力也有所提升，并在 X 上免费提供。它引入了**网页搜索**、**引用**以及名为 **Aurora** 的新图像生成器，以新功能惊艳了用户。

- [**Byte Latent Transformer 终结 Token，拥抱 Patch**](https://x.com/scaling01/status/1867573707247346003?s=46)：Meta 的 **Byte Latent Transformer (BLT)** 声称通过将字节动态编码为 Patch 来终结 Token 化。BLT 承诺更好的推理效率和扩展性，可能减少高达 **50%** 的推理 FLOPs。

**主题 2. AI 工具闹脾气：用户苦于 Bug 和额度**

- [**Flow Action 额度消失速度快过免费甜甜圈**](https://discord.com/channels/1027685395649015980)：用户在 **24 小时内烧光了 1k Flow Action 额度**，难以管理消耗。对于一些繁重的工作流，将任务拆分为更小单元的建议并不能解决问题。

- [**Bolt 疯狂消耗 Token，用户寻求“节食计划”**](https://github.com/stackblitz/bolt.new/issues/4218)：**Bolt** 正在以惊人的速度消耗 Token，但 UI 却没有反映出更改，这令用户感到沮丧。许多人正在记录问题，并诉诸于将项目 Fork 到 **Replit** 作为临时解决方案。

- [**Cursor IDE 运行缓慢如蜗牛，是时候清理对话了**](https://docs.cursor.com/get-started/usage#premium-models)：用户报告 **Cursor IDE** 在长时间使用后会变得迟钝，建议通过重置或清理聊天记录来提高效率。随着用户分享各种规避技巧，对更流畅编码体验的追求仍在继续。

**主题 3. AI 伦理风波：对齐与告密者的忧虑**

- [**OpenAI 的对齐框架引发激烈辩论**](https://github.com/AlignAGI/Alignment)：一位用户分享了一个基于共同人类价值观和反馈循环的 AI 对齐框架。其他人则怀疑对齐不同利益相关者利益的可行性，引发了关于伦理的讨论。

- [**告密者神秘死亡引发关注**](https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/)：曾对受版权保护材料的使用表示担忧的 OpenAI 告密者 **Suchir Balaji** 被发现死于家中。这一事件助长了阴谋论以及关于 AI 透明度的辩论。

- [**Elon Musk 警告 AI 垄断，指责政府举措**](https://x.com/elonmusk/status/1868302204370854026?s=46)：Musk 暗示美国政府可能会限制 AI 初创公司，导致人们担心 AI 领域会出现垄断。社区对创新受到抑制感到担忧。

**主题 4. AI 变得更有创意：从成人角色扮演到定制化输出**

- [**用户通过火辣的 ERP 提示词为 AI 增色**](source_url)：针对 AI 的 **成人角色扮演 (ERP)** 高级技术正在兴起，用户可以构建详细的角色档案。诸如 *"Inner Monologue"*（内心独白）和 *"Freeze Frame"*（定格画面）等方法增强了 AI 交互的沉浸感。

- [**从莎士比亚到苏斯博士：轻松定制 AI 风格**](https://youtu.be/aG0ixD3OY80)：用户正在定制 AI 输出以实现独特的语调和风格，强调了有效提示词的力量。一段 [YouTube 教程](https://youtu.be/aG0ixD3OY80) 展示了获得理想艺术风格的技巧。

- [**SillyTavern 成为我们意想不到的 AI 游乐场**](https://github.com/SillyTavern/SillyTavern)：**SillyTavern** 作为 LLM 工程师测试模型和参数的工具正受到关注。用户在享受严肃测试与趣味互动结合的同时，不断推高 AI 能力的边界。

**主题 5. AI 研究突破：新方法与新模型涌现**

- [**Meta 的 BLT 舍弃 Token，转而采用 Patch**](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/)：Meta 的 **Byte Latent Transformer** 引入了一种无分词器（tokenizer-free）架构，将字节编码为 Patch 以实现更好的扩展。BLT 模型声称在匹配 **Llama 3** 性能的同时，显著降低了推理时的 FLOPs。

- [**通过可微分自适应合并 (DAM) 简化模型合并**](https://github.com/arcee-ai/DAM)：**DAM** 论文揭示了一种无需大量重新训练即可集成模型的高效方法。关于模型合并技术及其在 AI 开发中独特优势的讨论异常火热。

- [**小模型通过测试时计算 (Test-Time Compute) 技巧超越大模型**](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute)：研究表明，扩展测试时计算可以让像 **Llama 3B** 这样的小型模型在复杂任务上表现优于大型模型。更智能地使用计算资源正在拉平 AI 性能的竞争环境。


---

# 第一部分：Discord 高层级摘要

## [Codeium / Windsurf](https://discord.com/channels/1027685395649015980) Discord

- **Flow Action Credits 消耗**：用户正在迅速耗尽 **Flow Action Credits**，一名用户在 **24 小时内消耗了 1k 额度**。
   - 建议包括将任务分解为更小的单元，尽管一些用户报告这在他们的工作流中并不奏效。
- **AI 代码修改担忧**：工程师们对 **AI 在设置了防止更改的参数后仍意外修改代码** 表示沮丧。
   - 社区正在讨论编写更好 Prompt 的策略，以确保 AI 驱动的代码保持无误。
- **与 NVIDIA RAPIDS 集成**：讨论重点介绍了 **NVIDIA RAPIDS cuDF**，它可以在不更改代码的情况下将 **#pandas** 操作加速高达 **150 倍**，详见 [NVIDIA AI Developer 的推文](https://x.com/NVIDIAAIDev/status/1868778156347339033)。
   - 成员们正考虑集成 RAPIDS 以增强其项目中的**数据处理能力**。
- **Codeium 与 Gemini 2.0 对比**：**Codeium** 和 **Gemini 2.0** 正在进行对比，观察到 Gemini 在某些编码任务中表现更优。
   - 然而，Gemini 缺少 **Claude** 中的一些功能，导致根据具体用例产生了不同的看法。
- **MCP 和 Function Calling 协议**：正在讨论 **Model Context Protocol (MCP)**，用于在不同技术栈之间建立标准化的 **function call 结构**。
   - 用户建议利用 **Playwright** 和 MCP 等工具来增强 **GUI 测试**和交互。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM Plus 缓慢推出**：用户报告 **NotebookLM Plus** 正在分阶段推出，部分访问权限取决于其 Google 账号。预计到 2025 年初将向 **Google One Premium** 订阅者全面开放。
   - 一些用户在访问新功能时遇到延迟，引发了关于优化部署策略的讨论。
- **NotebookLM 播客功能的增强**：最新的 **NotebookLM 播客功能** 包括显著提高用户参与度的自定义和交互功能。展示这些功能的播客链接被广泛分享。
   - 成员们称赞该应用对**音频内容领域**的影响，并提到了允许更动态交互的具体增强功能。
- **增加 NotebookLM 的来源限制**：免费版 **NotebookLM** 现在支持多达 **300 个来源**，引发了用户关于模型如何管理这一增长的疑问。正在探索有效利用这一扩展来源池的策略。
   - 用户正在积极讨论收集足够来源的方法，以最大限度地发挥增加限制带来的好处，旨在获得更全面的 AI 输出。
- **为不同风格定制 AI 输出**：强调了有效的 Prompting 和自定义函数在**定制 AI 输出**中的作用，从而产生不同的语气和风格。分享了一个 [YouTube 教程](https://youtu.be/aG0ixD3OY80) 来展示有效的 Prompting 技巧。
   - 用户正在微调 AI 响应以实现特定的艺术效果，利用定制化来满足多样化的内容创作需求。
- **AI 工具中的多语言支持挑战**：讨论强调了在不同语言中使用 **NotebookLM** 的复杂性，用户正在寻求引导 AI 以首选语言响应的方法。建议调整 Google 账号语言设置作为解决方案。
   - 参与者正在分享 Prompt 策略，以确保准确且符合语境的多语言 AI 交互。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor IDE 面临性能迟缓**：用户报告在长时间开发过程中 **Cursor IDE** 出现 **卡顿 (sluggishness)**，引发了关于是否需要重置或清除聊天历史的讨论。建议包括 [创建新的聊天会话](https://docs.cursor.com/get-started/usage#premium-models) 以提升工作流效率。
   - 实施这些更改旨在缓解性能瓶颈，并为长时间的编码任务提供更流畅的用户体验。
- **讨论 Cursor 的 Agent 与 Gemini 1206**：参与者对比了 **Cursor 的 Agent** 与 **Gemini 1206**，强调了 Cursor 用户友好的界面以及 Gemini 在编码任务上的卓越性能。这种对比突显了每个模型在不同开发场景下的优势。
   - 用户强调了根据项目需求选择合适工具的重要性，[Google AI Studio](https://aistudio.google.com/) 支持 Gemini 的各项功能。
- **构建新的社交媒体平台**：几位用户表达了开发社交媒体平台的兴趣，重点关注必要的后端结构和潜在框架。重点在于理解 **CRUD 操作** 和管理 **数据库关系**。
   - 推荐使用 **Cursor IDE** 等工具来简化开发过程并确保高效的数据库管理。
- **通过 Supabase 和 Bolt 集成增强 Cursor**：有提议将 **Cursor** 与 **Supabase** 和 **Bolt** 等平台集成以扩展其功能。这些集成旨在简化工作流并增强开发能力。
   - 用户讨论了此类集成的潜在好处，包括改进数据管理和简化部署流程。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **可微分自适应合并 (Differentiable Adaptive Merging, DAM)**：**Differentiable Adaptive Merging (DAM)** 论文介绍了一种无需大规模重新训练即可高效集成模型的方法，利用了 [Differentiable Adaptive Merging (DAM)](https://github.com/arcee-ai/DAM)。
   - 论文指出，像 **Model Soups** 这样更简单的技术在模型相似度较高时表现良好，展示了各种集成方法中的独特优势。
- **Unsloth 与 Triton 的兼容性问题**：用户遇到了 **Unsloth** 与 **Triton** 之间的兼容性问题，需要安装特定版本以实现无缝集成。
   - 特别是 Python 3.13 带来了挑战，建议通过 Conda 使用 Python 3.10 以增强兼容性。
- **长上下文模型的效率**：讨论指出了 **长上下文模型 (long context models)** 的局限性，强调了数据过滤的复杂性，以及仅靠数据质量不足以驱动训练效率。
   - 参与者认为排除“坏数据”可能会损害模型的理解能力，因为多样化的数据集对于构建强大的 AI 至关重要。
- **使用 Unsloth 的微调技术**：对 **Unsloth** **微调技术** 的探索揭示了在数据集加载以及模型与 **Streamlit** 等平台的兼容性方面的共同挑战。
   - 社区成员就正确的加载语法和模型配置提供了建议，以解决 FileNotFoundError 和模型识别错误等问题。
- **Llama 3.2 的最大序列长度**：关于 **Llama 3.2** **最大序列长度 (max sequence length)** 的查询出现，最初建议为 4096。
   - 随后被修正为实际最大值 **131072**，提供了对该模型能力的深入了解。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **AI Alignment 框架分享**：一位用户介绍了一个用于 **AI Alignment** 的[工作框架](https://github.com/AlignAGI/Alignment)，重点关注基于共同人类价值观的原则和迭代反馈，以确保 AI 开发的包容性。
   - 讨论强调了在利益相关者之间达成共识的挑战，并对协调多样化利益的可行性持怀疑态度。
- **Google Gemini 和 Imagen 更新讨论**：用户对 **Google Gemini** 和最近的 **Imagen** 更新进行了评估，并将其性能与 **OpenAI GPT-4** 等现有模型进行了比较。
   - 参与者指出，虽然 **Grok** 等模型正在进步，但在能力上仍落后于 **ChatGPT** 等更成熟的模型。
- **GPT 4o 与 4o-mini 之间的性能差距**：用户对 **GPT 4o** 和 **GPT 4o-mini** 之间的**性能差异**表示沮丧，称 mini 版本表现得像在“梦游”。
   - 社区观察到 **GPT 4o-mini** 的响应质量显著下降，影响了整体用户体验。
- **本地 LLM 的优势探讨**：参与者讨论了**本地 LLM 的益处**，强调了与大型科技公司的解决方案相比，本地 LLM 在提供更具定制化和灵活性的 AI 体验方面的潜力。
   - 有人担心大型科技公司可能会在 AI 交互中优先考虑生产力的提升，而非创造力。
- **精炼提示工程（Prompt Engineering）技术**：用户分享了**增强提示工程**的策略，将有效的提示词编写比作从零开始烹饪，并强调了清晰指令的重要性。
   - 讨论内容包括开发提示工程课程，以及在 IDE 中利用 AI 进行代码辅助。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Byte Latent Transformer 发布，挑战 Llama 3**：Meta 推出了 **Byte Latent Transformer (BLT)**，这是一种无分词器（tokenizer-free）架构，可将 Bytes 动态编码为 Patches，从而提高**推理效率**和鲁棒性。参见[公告](https://x.com/scaling01/status/1867573707247346003?s=46)。
   - BLT 模型声称能达到与 **Llama 3** 等基于分词的模型相当的性能，同时可能减少高达 **50%** 的**推理 FLOPs**。他们在 **1T tokens** 上训练了 **Llama-3 8B** 模型，表现优于使用 BPE 的标准架构。
- **Apollo LMMs 发布提升视频理解能力**：社区讨论了 **Apollo LMMs** 的最新更新，其中包括专注于视频理解和多模态能力的模型。初步印象显示其表现良好，引发了对其潜在应用的兴趣。
   - 成员们对将 Apollo 模型集成到现有工作流中持乐观态度，以增强**视频分析**和**多模态处理**能力。
- **开源代码 LLM 提升开发效率**：推荐了几个开源代码 LLM，如 **Mistral Codestral**、**Qwen 2.5 Coder** 和 **DeepSeek**，它们可以与 VS Code 和 PyCharm 等 IDE 以及 [continue.dev](https://continue.dev) 等扩展集成。
   - 这些工具使开发者能够使用本地模型提高编码效率，营造更具定制化的开发环境。
- **模型压缩技术借鉴通信理论**：讨论集中在**通信理论**原则如何影响 **LLM** 的发展，特别是在分布式训练期间的梯度传输方面。
   - 成员们指出，**用计算换带宽**可以简化流程，尽管组合多种技术可能会很复杂。此外还强调了在不损害性能的情况下优化数据效率的潜力。
- **本地 LLM 微调变得更加便捷**：讨论提到，借助 **unsloth** 和 **axolotl** 等工具，即使是资深技术爱好者也有可能使用 **QLoRA** 训练高达 80 亿参数的模型。
   - 越来越多的资源让那些愿意学习的人能够进行定制化开发，扩展了本地模型微调的能力。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **SF Compute 集成至 OpenRouter**：OpenRouter 宣布新增 **SF Compute** 作为新的供应商，增强了其服务能力。
   - 这一集成扩展了用户在平台上寻求多样化服务集成的选择。
- **Qwen QwQ 降价 55%**：**Qwen QwQ** 进行了大幅度 **55% 的降价**，旨在吸引更多用户使用其功能。
   - 详情请见其 [定价页面](https://openrouter.ai/qwen/qwq-32b-preview)。
- **xAI 发布新 Grok 模型**：**xAI** 在周末发布了两个新的 **Grok 模型**，导致平台流量增加。
   - 用户可以在 [OpenRouter 的 xAI 页面](https://openrouter.ai/x-ai) 探索所有模型。
- **OpenRouter API 封装库发布**：两天前发布了一个名为 [openrouter-client](https://www.npmjs.com/package/openrouter-client) 的 OpenRouter API 封装库。
   - 该封装库简化了与 OpenRouter 的交互，并提供了实现和配置的示例代码。
- **Hermes 3 405B 展现强劲性能**：**Hermes 3 405B** 在创意任务中表现出色，据称其质量可与 **Claude 2.0** 媲美。
   - 然而，讨论指出与其他模型相比，它在编程任务中的性能较慢。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **JAX/Flax 取代 TensorFlow 以提升性能**：成员们对 **TensorFlow** 支持度下降表示不满，导致许多人 [转向 JAX/Flax](https://x.com/SkyLi0n/status/1867324080262885800)。**JAX/Flax** 提供了更优的性能和更强大的功能，适用于现代 AI 工程。
   - 社区赞扬了 **JAX/Flax** 的灵活性以及与当前模型架构更好的集成，并提到了更顺畅的依赖管理和更高的计算效率。
- **数据打乱减少近期训练带来的模型偏见**：有人担心模型会对最近引入的训练数据产生偏见。成员们建议将 [数据打乱 (data shuffling)](https://arxiv.org/abs/2412.06464) 作为增强 **训练公平性** 和减少偏见的策略。
   - 成员们分享了数据同质化策略的经验，强调了通过 **随机数据排序** 提升模型性能和公平性的效果。
- **注意力机制优于核方法**：关于 Transformer 中的 **注意力机制 (attention mechanisms)** 是否等同于核方法 (kernel methods) 展开了辩论。成员们澄清说，**attention**（特别是带有 **softmax** 的）超出了传统核方法的功能范围。
   - 讨论包括数学上的区别，并辩论了 attention 是否充分利用了核潜力，强调了其运行环境的复杂性。
- **非 Transformer 架构在 AI 研究中势头强劲**：**非 Transformer 架构** 的活跃研究受到关注，提到 **Numenta** 和 **AI2** 等实验室发布了与主流 Transformer 模型不同的新模型 Checkpoint。
   - 社区成员对推动新颖方法的小型实验室表示出兴趣，强调了多样化模型架构对推进 AI 能力的重要性。
- **lm_eval 成功集成 VLLM**：一位用户分享了让 **lm_eval harness** 与 **VLLM** 协同工作的方法，并指出了具体的安装命令。该过程包括安装 0.6.3 版本的 VLLM，以防止评估 harness 出现问题。
   - 成员们讨论了 VLLM 产生的错误，暗示 **lm_eval 使用的内部 API** 可能已更改，并澄清了版本细节以解决 [VLLM 版本混淆](https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval) 问题。

---

## [Bolt.new / Stackblitz](https://discord.com/channels/364486390102097930) Discord

- **Bolt 的 Token 消耗激增**：多位用户报告称 **Bolt** 正在以加速率消耗 Token，其中一位用户指出在 UI 没有相应更改的情况下消耗了 **500 万个 Token**。该问题已记录在 [GitHub Issue #4218](https://github.com/stackblitz/bolt.new/issues/4218) 中。
   - 成员们怀疑这是一个系统性 Bug，并正通过将项目 fork 到 GitHub 并在 Replit 上运行作为权宜之计。
- **货币更新遇到困难**：用户在将货币显示从 **$ USD** 更改为 **INR** 时遇到困难，即使在锁定 `.env` 文件后也是如此，这表明 Bolt 的文件处理可能存在 Bug。
   - 这一持续存在的问题已在多个频道中被报告，表明这并非特定于浏览器的孤立问题。
- **Supabase 集成引发期待**：备受期待的 **Supabase** 与 **Bolt** 的集成正引发热潮，早期的 [视频演示](https://x.com/morganlinton/status/1868388127347523794?s=46) 展示了其功能。
   - 用户渴望获得更新，并期待新功能能够增强他们的项目。
- **对 Token 成本和订阅的担忧**：用户对 Token 的快速消耗表示担忧，尤其是在充值之后，并寻求关于 Token 管理机制的明确说明。
   - 用户对当前的过期规则表示不满，并主张建立累积 Token 系统。
- **React Native 开发指南**：讨论强调了使用 **React Native** 和 **Expo** 将 Web 应用程序迁移到移动平台的最佳实践。
   - 建议包括将开发转移到 **Cursor** 以获得更好的功能支持。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Grok-2 凭借 Aurora 提速**：Grok-2 已更新，运行速度提升了 **三倍**，并提高了 **准确率** 和 **多语言能力**，现已在 [X](https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh) 上免费提供。
   - 它引入了 **网页搜索**、**引用** 以及名为 **Aurora** 的新图像生成器等功能，显著增强了用户交互。
- **Ilya Sutskever 的 NeurIPS 新见解**：在 [NeurIPS 2024](https://youtu.be/1yvBqasHLZs?si=pQihchmQG3xoeCPZ) 的演讲中，Ilya Sutskever 强调了 **LLM** 在预训练期间的规模化瓶颈，以及未来向 **Agent 行为** 和 **工具集成** 发展的转变。
   - 讨论包括关于 **数据饱和** 以及 **未开发的视频内容** 用于 **AI 训练** 潜力的各种观点。
- **Google 的 Veo 2 和 Imagen 3：媒体魔力**：Google 推出了 **Veo 2** 和 **Imagen 3**，具有改进的高质量 **视频生成** 和增强的 **图像构图** 功能，可在 **VideoFX** 和 **ImageFX** 中使用。
   - 这些更新在生成内容中提供了更好的 **电影摄影理解** 和多样化的 **艺术风格**。
- **META 的 Byte Latent Transformer**：META 发布了 **Byte Latent Transformer (BLT)**，这是一种无分词器 (tokenizer-free) 架构，可将字节动态编码为 patch，从而提高 **推理效率**。
   - BLT 模型与 **Llama 3** 等现有模型相匹配或更胜一筹，显著降低了 **推理 FLOPs**。
- **OpenAI 为 ChatGPT 推出语音搜索**：OpenAI 宣布为 **ChatGPT** 推出 **高级语音模式下的搜索** 功能，允许用户通过 **语音交互** 获取 **实时信息**。
   - 该功能是 OpenAI 的 **Search** 团队与 **多模态产品研究团队** 合作的成果。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **多模态模型集成**：成员们探讨了结合文本、图像、音频和视频的 **Multimodal Models**，指出大多数解决方案通过 [云服务](https://lmstudio.ai/beta-releases) 访问，同时强调了 **LM Studio** 目前在这一领域的局限性。
   - 一个关键讨论点是本地设置缺乏完全多模态的 **LLMs**，这引发了对即将发布的模型的期待。
- **模型微调的局限性**：用户询问是否可以通过数据导出来对现有模型进行 **Fine-tuning**，以模拟特定的语法或语气，但被告知 **LM Studio** 不支持 **Fine-tuning**。
   - 作为替代方案，建议在聊天界面中使用 **System Prompts** 和示例文本进行临时模型调整。
- **无审查聊天机器人的选项**：在寻找 **Uncensored Chatbots** 时，建议成员使用可以在 **CPU** 上运行的小型模型，如 [Gemma2 2B](https://huggingface.co/mustafaaljadery/gemma-2B-10M) 或 [Llama3.2 3B](https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF)。
   - 分享了 [Hugging Face](https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF) 上可用于本地环境部署的各种无审查模型。
- **RAG 实现与文档处理**：讨论了 **LM Studio** 中的 **Retrieval-Augmented Generation (RAG)** 能力和文档上传功能，将其作为利用本地文档增强上下文响应的手段。
   - 用户获知虽然所有模型都支持 **RAG**，但集成网页访问或互联网功能需要自定义 **API** 解决方案，详见 [LM Studio Docs](https://lmstudio.ai/docs/basics/rag)。
- **AI/ML 任务的 GPU 选择**：对话强调具有更大 **VRAM** 的 **GPU**（如 **3090**）更适合 **AI** 和 **Machine Learning** 任务，因为它们具有卓越的速度和能力。
   - 提到了 **4070ti** 等替代方案，尽管一些成员指出，根据当地供应情况，二手 **3090** 可能提供更好的性价比。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Reactor 实现高效换脸**：用户推荐使用 **Reactor 扩展**进行图像中的 **Face Swapping**，在启用 **Reactor** 并上传目标面部图像后，用户可以成功生成修改后的图像。
   - 该方法增强了 **Stable Diffusion** 工作流中的图像处理能力，允许无缝集成不同的面部特征。
- **讨论 Stable Diffusion 的多样化模型**：讨论强调了各种 **Stable Diffusion 模型**，强调最佳选择取决于用户需求，其中 **Flux** 和 **SD 3.5** 在提示词遵循方面表现出色，而 **Pixelwave** 因其艺术知识受到称赞。
   - 参与者分享了不同模型的经验，以优化图像生成质量和性能，根据特定项目需求定制选择。
- **寻求全面的 Stable Diffusion 学习资源**：用户寻求关于 **Stable Diffusion** 的广泛课程和教程，特别是关注其与 **Automatic1111** 的集成，建议指向 YouTube 等平台上的系列视频和专门的在线资源。
   - 这些资源旨在提高用户对利用 **Stable Diffusion** 高级功能的理解和熟练程度。
- **使用放大工具优化图像质量**：用户请求推荐与 **Stable Diffusion** 生成图像兼容的高效 **Upscalers**，讨论了提高图像分辨率和质量的特定工具或扩展。
   - 辩论了增强的 **Upscaling** 技术，以在生成的图像中实现更好的视觉保真度。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **LiquidAI 获得 2.5 亿美元融资**：LiquidAI 宣布完成由 **AMD Ventures** 领投的 **2.5 亿美元 A 轮融资**，旨在为其企业级 AI 解决方案扩展其 **Liquid Foundation Models (LFMs)**。
   - 讨论中对其招聘实践表示了担忧，涉及潜在的人才挑战以及 **LiquidAI** 的规模可能阻碍收购机会的可能性。
- **ChatGPT 通过 Memory 增强搜索功能**：ChatGPT 在其搜索功能中引入了 **memory 特性**，允许模型利用记忆来优化搜索响应，从而提高相关性。
   - 用户对此次更新未包含个性化搜索表示失望，并期待未来的增强功能，包括潜在的 API 集成。
- **DeepMind 发布 Veo 2 和 Imagen 3**：**DeepMind** 推出了新的视频生成模型 **Veo 2** 以及升级后的 **Imagen 3**，增强了根据提示词生成写实内容的能力。
   - 早期反馈称赞了 **Imagen 3** 的表现，强调了 **DeepMind** 在科技界相对于 **OpenAI** 等其他主要参与者的竞争优势。
- **OpenAI 举报人事件**：OpenAI 举报人 **Suchir Balaji** 被发现死于其公寓内，当局报告死因为自杀，并排除了他杀可能。
   - **Balaji** 因在离开公司后不久对 **OpenAI** 使用受版权保护的材料训练 **ChatGPT** 提出担忧而闻名。
- **Apollo 视频 LLM 挑战竞争对手**：Meta 的 **Apollo** 系列视频 **LLM** 表现强劲，可与 **llava-OV** 和 **Qwen2-VL** 媲美。
   - 讨论强调了 **Apollo** 使用 **Qwen2.5** 作为其底层 **LLM**，而非预期的 **Llama**，引发了关于如何选择模型以实现最佳性能的疑问。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 订阅扩展服务**：**Perplexity Pro** 现在提供 **1、3、6 或 12 个月**期限的礼品订阅，使用户能够分享增强功能，如搜索 **3 倍数量的来源**以及访问最新的 AI 模型。详细信息和购买选项请见[此处](https://perplexity.supply/shop/perplexity-subscription)。
   - **Campus Strategist 项目**正在向国际扩展，允许学生在 12 月 28 日前申请 2025 年春季班，专属周边和活动机会详见[此处](https://www.perplexity.ai/campus-strategists)。
- **Spaces 推出自定义 Web 来源功能**：Perplexity AI 在 Spaces 中引入了[自定义 Web 来源](https://x.com/perplexity_ai/status/1867615710391746836?s=46)，使用户能够通过选择特定网站来定制搜索，从而增强不同用例的相关性。
   - 该功能允许工程师在 **Spaces** 内优化搜索查询，确保结果更符合专业化需求。
- **Perplexity API 面临 URL 和访问挑战**：用户报告 **Perplexity API** 将来源引用返回为纯文本数字（如 [1]）而没有 URL，尽管一些用户通过明确请求成功获取了 URL。
   - 此外，通过 API 获取新闻标题以及通过提供的电子邮件获取支持存在困难，表明可能存在稳定性和可用性问题。
- **对 Perplexity API 模型性能的担忧**：多位用户表示最近的**模型更新**导致性能下降，特别指出 **Claude 3.5** 的效果不如其免费版本。
   - 模型切换缺乏透明度，这影响了 API 服务的感知质量和可靠性。
- **Google 发布 Gemini 2.0**：Google 推出了 **Gemini 2.0**，标志着 **AI 能力**的重大进步，这引发了关于 [problem movement](https://www.youtube.com/embed/nQTAbz1eDco) 的讨论。
   - 讨论参与者对此次更新及其对 AI 领域的潜在影响表示了热忱。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R7B 模型提速**：[Cohere Command R7B 12-2024](https://cohereforai-c4ai-command.hf.space/models/command-r7b-12-2024) 模型现已上线，针对推理和摘要任务进行了优化，拥有更快的速度和更高的效率。
   - [Nils Reimers 的 Twitter](https://x.com/Nils_Reimers/status/1868065732149571701) 上展示的社区基准测试显示，**Command R7B** 的表现优于 **Llama 8B** 等模型，响应时间有显著改善。
- **Rerank 与 Embed：功能详解**：讨论明确了 **Rerank** 根据查询相关性对文档进行重新排序，而 **Embed** 则将文本转换为用于 NLP 应用的数值向量。
   - **Embed** 的 API 更新现在支持 'image' 输入类型，将其适用范围扩展到了文本任务之外。
- **v2 版本中的 API Schema 重构**：从 API v1 到 v2 的迁移缺乏关于新端点 Schema 变化的详细文档，导致用户对具体更新感到困惑。
   - 工程师们正在调查现有的 [迁移资源](https://discord.com/events/954421988141711382/1308148058894110750/1318261358592000000)，以明确新的 API 结构。
- **为 Code Wizard 黑客松寻求赞助**：**Akash** 宣布了即将由 SRM Institute 在 2025 年 2 月举办的 **Code Wizard** 黑客松，目标是吸引学生和技术爱好者解决现实世界的问题。
   - 该活动正在积极寻求赞助商以获得支持和曝光，旨在开发者社区内培养创新解决方案。
- **AI 增强合同条款审查**：**Eyal** 正在开发一个使用 **Cohere** 的概念验证（PoC），用于自动识别并建议合同条款的修改。
   - 正在征求关于定义特定条款类型或利用变更数据库等策略的反馈，以提高 AI 在合同分析中的有效性。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo RSA 加密开发**：一名成员发起了在 **Mojo** 中开发基础 **RSA crypto** 实现的项目，并展示了进展。
   - 该项目引起了不同的反响，突显了社区的热情和建设性的反馈。
- **素数生成优化**：素数生成脚本达到了 **1.125 秒** 的峰值性能，经过优化后，使用 **SIMD instructions** 每秒可生成超过 **50,000 个 UInt32 素数**。
   - 这些增强功能保持了低内存占用，应用程序在运行期间消耗的内存少于 **3mb**。
- **自定义 Mojo Kernels**：[自定义 Mojo Kernels](https://github.com/cassioneri/teju_jagua) 已经发布，允许接受任何输入类型，尽管早期版本可能会因类型不匹配而崩溃。
   - 开发者对 API 未来的健壮性保持信心，预计随着实现的成熟，稳定性将会提高。
- **Mojo 中的网络性能**：讨论倾向于在 **Mojo** 应用中使用 **QUIC** 而非 **TCP** 以降低延迟。
   - 在现代网络环境中，避免 TCP 开销被视为实现高效 **Mojo-to-Mojo** 通信的关键。
- **MAX 中的数据库规划**：一名开发者计划在 **MAX** 中实现 **数据库查询规划** 和执行，利用新的自定义 Kernel 功能。
   - 这一举措表明了在 **Mojo** 生态系统中推动更强大复杂数据操作处理的趋势。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents MOOC Hackathon 截止日期临近**：**LLM Agents MOOC Hackathon** 的提交截止日期为 **PST 时间 12 月 17 日晚上 11:59**，敦促参与者按时完成并提交项目。
   - 鼓励参与者在指定频道寻求最后时刻的帮助，以确保所有提交内容符合要求。
- **Hackathon 参赛条目转向 Google Forms**：Hackathon 的提交方式已从 **Devpost** 转向 **[Google Forms](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)**，以简化提交过程。
   - 参与者必须确保使用正确的表单链接，以避免在截止日期前出现任何提交问题。
- **证书通知计划于 12 月底发布**：**Certificate notifications**（证书通知，包括通过或失败状态）将根据参与者的层级在 **12 月底至 1 月初**期间发放。
   - 这一时间表回应了最近的咨询，并为参与者何时能获知其认证状态设定了明确预期。
- **OpenAI Credit 提交问题**：一些成员报告称，尽管在 **11 月 25 日**截止日期前提交了组织 ID，但仍未收到 **OpenAI credits**。
   - 社区成员建议核实账户余额，因为通知可能未正常发送。
- **强调 AI Research Agents 中的安全对齐**：一位成员强调了 **AI Research Agents** 中 **safety alignment** 的重要性，并分享了相关的 [AI Research 资源](https://airesearch.js.org)。
   - 这突显了社区对确保安全协议成为 AI research agents 开发中不可或缺的一部分的关注。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune v3.9 简化 Type Hinting**：[Torchtune v3.9](https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L780) 的更新允许用户使用默认内置类型替换 `List`、`Dict` 和 `Tuple` 进行 Type Hinting。
   - 这一调整受到社区欢迎，它简化了 Python 代码，提高了开发者的生产力。
- **Generative Verifiers 提升 LLM 性能**：题为 [Generative Verifiers: Reward Modeling as Next-Token Prediction](https://arxiv.org/abs/2408.15240v1) 的论文介绍了 *Generative Verifiers (GenRM)*，该模型使用 Next-token prediction 目标进行训练，以无缝集成验证和解法生成。
   - 该方法支持 Instruction tuning，并通过利用**额外的 Inference-time compute** 来增强验证结果，从而实现 Chain-of-thought 推理。
- **Distributed Training 中的 Gradient Normalization 挑战**：讨论强调了 Distributed Training 中 Backward pass 期间归一化缩放因子的担忧，建议应为 `world_size / num_tokens` 以管理 Token 数量的可变性。
   - 这个问题可能会由于 Padding 和索引差异而使梯度计算复杂化，从而引发了对通过 PR 解决不一致性的倡议。
- **探索 Scaling Test Time Compute 策略**：一篇 [Hugging Face 博客文章](https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute) 讨论了扩展大型模型 Test-time compute 的策略，重点是在不损害结果的情况下进行性能优化。
   - 该文章概述了在保持模型输出完整性的同时提高计算效率的方法。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **优化 Kernel Search 的 BEAM 配置**：成员们讨论了用于 **Kernel Search** 的各种 **BEAM** 设置，强调 **BEAM=1** 表示贪心搜索（greedy search），其效果较差。推荐的入门设置是 **BEAM=2** 或 **3** 以平衡性能，详见 [文档](https://docs.tinygrad.org/env_vars/)。
   - **Kernel Search 体验**的提升重点在于优化 **编译时间** 和 **Kernel 执行时间**。成员们对现有的 Benchmark 感兴趣，并建议在使用 **JIT 编译** 时采用 **BEAM=2**。
- **新的 Gradient API 简化梯度处理**：George Hotz 宣布合并了新的 **Gradient API**，它可以简化梯度处理：使用 `weight_grad, bias_grad = loss.gradient(weight, bias)`，而无需 `zero_grad` 或 `loss.backward`。
   - 该 API 与 **PyTorch** 和 **JAX** 等传统框架不同，正如这篇 [推文](https://x.com/__tinygrad__/status/1867745748118544411) 中提到的，它可能会通过 `optim.step(loss)` 简化优化器步骤。
- **Tinygrad 移植项目与后端支持讨论**：宣布了将 **fish-speech** 项目移植到 Tinygrad 的计划，旨在增强 Tinygrad 的能力。该项目托管在 [GitHub](https://github.com/fishaudio/fish-speech) 上。
   - 成员们讨论了为 Tinygrad 同时支持 **x86** 和 **arm64 后端** 的问题，考虑在资源受限的情况下维持性能。
- **ShapeTracker 解释器与教程扩展**：发布了改进版的 **ShapeTracker Explainer**，可在 [此处](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md) 查看，提供了对其工作原理的深入见解。
   - [tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes) 仓库征集教程和资源贡献，鼓励社区参与。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **5 行代码实现 LlamaIndex RAG**：TylerReedAI 分享了一个详细的 [教程](https://t.co/v5yljbVw4d)，介绍如何仅用 **5 行代码** 构建 **RAG 应用**，涵盖了数据加载和索引。
   - 该教程强调了将 **Query 引擎** 和 **Chat 引擎** 集成到工作区中的简便性。
- **Agentic 合规工作流**：一个新的 [教程](https://t.co/9SjfXRWdmF) 介绍了一种构建 **Agentic 工作流** 的方法，通过根据 **GDPR** 指南分析条款来确保 **合同合规性**。
   - 它分解了如何解析供应商合同以有效维持合规性，从而简化合同管理。
- **Contextual Retrieval 与 LlamaIndex 结合**：一位用户在 **LlamaIndex** 中实现了 **Anthropic 的 Contextual Retrieval**，并分享了他们的 [GitHub 仓库](https://github.com/cklapperich/Eidetic/) 供他人审阅。
   - 他们表示有兴趣将这个健壮的实现作为 PR 提交，并强调了它对边缘情况的处理。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **文件夹创建问题与缩进错误**：一位成员指出该工具*无法创建文件夹*，且生成的代码*缩进错误*，导致难以复制粘贴，并询问是否应使用 cmd 以外的其他环境。
   - 这一问题表明当前设置中的文件夹创建功能和代码格式化过程可能存在 Bug。
- **macOS Monterey 上的 API 响应限制**：一位用户报告称，在 **macOS Monterey** 上安装应用后，无法收到 API 响应，且仅执行 **两次操作** 后就达到了免费 Token 限制。
   - 这表明可能存在特定于 macOS Monterey 的集成或使用问题，可能影响了 API 的可用性。
- **增强 Litellm 的计费追踪**：一位用户询问如何将 **OI** 连接到 Litellm 代理服务器，以便有效追踪集成版 Litellm 包的计费和使用情况。
   - 他们正在探索在 **Litellm** 集成中启用全面计费追踪的方法。
- **日语学习应用推荐**：一位成员寻求好用的 **日语** 学习应用，导致另一位用户幽默地暗示他们可能进错了 Discord 服务器。
   - 这一交流强调了社区内对专注于语言学习的专门资源或频道的潜在需求。
- **本地 OS 部署选项**：一位用户询问了在本地使用该 OS 的可能性，表示对本地设置方案感兴趣。
   - 该查询指向了关于本地环境潜在部署或托管配置的讨论。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **使用 DSpy 优化 Claude Sonnet 提示词**：一位用户在寻找优化其 **Claude Sonnet** 提示词的方法时发现了 **DSpy**，并收藏了一个特定的 [Jupyter notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/vlm/mmmu.ipynb)。
   - 他们提到该 notebook 最近被移到了一个过时的示例文件夹中，引发了对其时效性的疑问。
- **更新过时的 DSpy 示例**：另一位成员建议，在 **DSpy** 中过时示例文件夹的内容被翻新之前应谨慎使用，这表明其可能存在不可靠性。
   - 他们还指出，目前正在努力更新这些示例，有望提高它们的实用性。

---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **APOLLO 优化器增强内存效率**：新的 **APOLLO 优化器** 在 **LLaMA 7B 训练** 期间将内存使用量减少到 **1.6G**，同时实现了最佳困惑度（perplexity），相比之下，使用 **8-bit Adam** 则需要 **13G**。
   - 一个独立的 **Julia 实现** 证实了 APOLLO 在优化内存和训练效率方面的有效性，详见此 [帖子](https://bsky.app/profile/benjmurrell.bsky.social/post/3lcyfrf5b7k2u)。
- **LLM 训练面临 AdamW 的内存限制**：大型语言模型在使用 **AdamW 优化器** 时会遇到严重的内存问题，通常需要在训练期间使用昂贵的硬件或较小的 batch sizes。
   - 传统的内存高效优化器涉及 **SVD 操作** 或性能权衡，但 **APOLLO** 引入了一种新颖的方法来解决这些局限性。
- **关于多轮 KTO 的持续讨论**：讨论强调了 **多轮 KTO**，尽管未提供具体细节和更新。
   - 社区成员对该方法在 LLM 框架内的潜在能力和集成表示了兴趣。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **VAE Embedding 改进渐进式分词**：讨论集中在利用源自 **VAE embedding** 的 **DWT 系数** 的 **零树排序（zero-tree ordering）** 进行 **渐进式分词（progressive tokenization）**。一段附带的 [视频](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4) 展示了该技术的实际应用。
   - 分析了 **Level 5 小波（wavelet）** 变换对分词有效性的影响，强调了实际应用以及对未来模型增强的意义。
- **Byte Latent Transformer Patches 优于 Tokens**：出版物 [Byte Latent Transformer Patches: Scale Better than Tokens](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) 详细介绍了一种新的 NLP 方法，其中 **byte latent transformer patches** 与传统 tokens 相比展现出更好的可扩展性。
   - 这一进展引发了关于在各种应用中增强语言建模 **有效性** 和 **效率** 的讨论。
- **Level 5 小波变换提升分词效果**：研究了 **Level 5 小波** 变换在改进当前方法中分词有效性的作用。
   - 分析包括探索实际应用和对模型性能的未来影响，并引用了 [附带视频](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4)。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **RAG 盛会：使用 SQLite-Vec 和 LlamaFile 构建**：明天的活动重点是使用 **sqlite-vec** 和 **llamafile** 创建一个 **超低依赖的检索增强生成（RAG）** 应用程序，仅使用 **基础 Python**，无需额外的依赖或安装。
   - **Alex Garcia** 将主持该会议，为与会者提供构建 RAG 应用程序的简单方法。
- **假期聚会：放假前的最后一次 RAG 环节**：12 月假期前的 **最后一次聚会** 强调了在年底前参与的重要性。
   - 鼓励参与者 **加入该环节**，作为假期的前奏，并获取有关 **RAG 开发** 的见解。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **Gorilla LLM 发布 Function Calling 结果**：**Gorilla LLM** 的 Berkeley Function Calling 的 [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) 仓库已更新。
   - [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) 仓库现已可供查阅。
- **Gorilla LLM 发布 Function Calling 结果**：**Gorilla LLM** 的 Berkeley Function Calling 的 [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) 仓库已更新。
   - [BFCL-Result](https://github.com/HuanzhiMao/BFCL-Result) 仓库现已可供查阅。



---


**MLOps @Chipro Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**HuggingFace Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该频道长期处于沉寂状态，请告知我们，我们将将其移除。


---

# 第 2 部分：按频道详细摘要与链接


{% if medium == 'web' %}




### **Codeium / Windsurf ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1317290753772228778)** (1 条消息): 

> `Discord 挑战赛获胜者、YouTube 视频提交、Windsurf Pro 层级奖励` 


- **本周 Discord 挑战赛获胜者名单公布**：恭喜本周 Discord 挑战赛的获胜者：<@254550955427627008> 和 <@1219755748960243743>，他们展示了令人印象深刻的作品。
   - 他们可以通过私信主持人领取 **3 个月 Windsurf Pro 层级** 的奖励。
- **获奖视频现已可观看**：查看获奖作品：来自 <@254550955427627008> 的 **Singularia**（[点击观看](https://www.youtube.com/watch?v=kO-zI0CYJ2w)）和来自 <@1219755748960243743> 的 **Sales Prompt Creator**（[点击观看](https://www.youtube.com/watch?v=7gA2IouD-XU)）。
   - 这两段视频都突显了创意与技巧，是社区成员必看的内容。
- **加入正在进行的 Windsurf 挑战赛**：参与者可以按照 [规则与提交链接](https://discord.com/channels/1027685395649015980/1027688115592237117/1306427389991059508) 加入滚动进行的 Windsurf Discord 挑战赛。
   - 这项持续进行的挑战赛为社区成员提供了展示才华的机会。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=kO-zI0CYJ2w"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=7gA2IouD-XU"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1317235016635387914)** (212 条消息🔥🔥): 

> `Windsurf 功能与问题、用户对 Flow Action Credits 的反馈、账户管理与支持、AI 行为与代码更改、与其他工具的集成` 


- **Windsurf 的功能与现有问题**：用户报告了 Windsurf 在工作期间无法保存文件或意外修改文件的问题，导致了挫败感。
   - 一些初级开发人员对功能未按预期运行表示困惑，并强调文档需要更加清晰。
- **Flow Action Credits 消耗过快**：多位用户指出他们消耗 Flow Action Credits 的速度非常快，其中一位用户提到在 24 小时内消耗了 1000 个额度。
   - 建议包括将任务拆分为更小的部分，尽管一些用户提到这种方法对他们的需求并不奏效。
- **账户管理与支持响应时间的困难**：用户在提交有关 Pro 账户激活和额度管理等问题的工单时，对支持响应缓慢感到沮丧。
   - 反馈表明支持团队可能需要就工单进度进行更好的沟通。
- **对 AI 代码实现的挫败感**：尽管设置了避免此类更改的参数，一些用户仍对 AI 意外修改其代码表示不满。
   - 围绕如何通过 Prompt 策略引导 AI 给出更好响应以避免代码错误的讨论也随之展开。
- **集成查询与功能请求**：用户表示有兴趣增加每月 Flow Action Credits 的配额，以及文件锁定等其他功能。
   - 此外，讨论还包括与 NVIDIA 的 RAPIDS 等现有工具进行潜在集成，以增强数据处理能力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：👀 RAPIDS cuDF 在无需更改代码的情况下将 #pandas 加速高达 150 倍。现在，随着数据集大小增长到 GB 级别，你可以继续使用 pandas。⚡ ➡️ 尝试演示的 Jupyter Notebook：http://nvda.ws...</li><li><a href="https://imgur.com/gallery/VQ2LV35">完全正常的图像 - Imgur 上的相册</a>：未找到描述</li><li><a href="https://ternarysteganography.vercel.app">三进制图像隐写术 (Ternary Image Steganography)</a>：未找到描述</li><li><a href="https://ternarykeyexchange.vercel.app/">三进制密钥交换 (Ternary Key Exchange)</a>：未找到描述</li><li><a href="https://www.aperisolve.com/">Aperi'Solve</a>：未找到描述</li><li><a href="https://docs.codeium.com/windsurf/usage">付费计划与额度使用 - Codeium 文档</a>：未找到描述</li><li><a href="https://github.com/favourablegroup/ternarysteganography">GitHub - favourablegroup/ternarysteganography</a>：通过在 GitHub 上创建一个账户来为 favourablegroup/ternarysteganography 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=qadmkq_d_co&list=LL&index=4&t=2s&pp=gAQBiAQB"> - YouTube</a>：未找到描述</li><li><a href="https://codeium.com/careers">Windsurf 编辑器和 Codeium 扩展</a>：Codeium 是深受开发人员喜爱且值得企业信赖的 AI 代码助手平台。同时也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/blog/pricing-windsurf">计划与定价更新</a>：我们对 Cascade 定价模型的一些更改。</li><li><a href="https://link.springer.com/chapter/10.1007/978-3-642-22786-8_30">使用谢尔宾斯基分形几何的对称加密</a>：对称加密使用相同的密钥进行加密和解密。对称加密的一个理想特性被称为雪崩效应（avalanche effect），即两个不同的密钥会产生不同的...
</li>
</ul>

</div>
  

---

### **Codeium / Windsurf ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1317219537954934868)** (609 条消息🔥🔥🔥): 

> `Windsurf 问题、AI 与依赖、Codeium vs. Gemini、MCP 与 Function Calling、Ruff Linter 与 Formatter` 


- **Windsurf 使用体验与 Bug**：用户报告了 Windsurf 的各种问题，包括界面卡死以及 Action 和聊天窗口无法正常工作的问题。
   - 一些用户建议通过重新安装或重置设置来解决持续存在的 Bug。
- **AI 与用户依赖担忧**：讨论中提到了对 Claude 等 AI 工具日益增长的依赖，以及在编程任务中过度依赖它们的潜在风险。
   - 用户对仅依赖 AI 的后果表示担忧，强调了个人自律和保持技能储备的必要性。
- **Codeium 与 Gemini 2.0 的比较**：用户对比了 Codeium 与 Gemini 2.0 的能力，指出虽然 Gemini 在编程任务中可能表现更好，但缺少 Claude 的一些功能。
   - 基准测试显示，根据具体的使用场景，对于哪种工具表现更好存在不同意见。
- **MCP 与 Function Calling 能力**：讨论了 Model Context Protocol (MCP) 在跨不同技术栈创建标准化 Function Call 结构方面的应用。
   - 用户提出了使用 Playwright 和 MCP 等工具来增强 GUI 测试和交互的想法。
- **Ruff Linter 与 Markdown 格式化**：关于使用 Ruff 作为 Python 的 Linter 和 Formatter 的讨论，并分享了在配置中排除特定文件的技巧。
   - 用户分享了关于保持代码整洁以及在项目中有效集成格式化工具的见解。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://astral.sh/blog/the-ruff-formatter">The Ruff Formatter: An extremely fast, Black-compatible Python formatter</a>: Ruff 的 Formatter 比现有工具快 30 倍以上，同时保持了与 Black 超过 99.9% 的兼容性。</li><li><a href="https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/use-xml-tags">Use XML tags to structure your prompts - Anthropic</a>: 未找到描述</li><li><a href="https://superuser.com/questions/177041/what-is-the-equivalent-of-mac-os-x-spaces-for-windows">What is the equivalent of Mac OS X Spaces for Windows?</a>: 我正在寻找一个能实现 Mac OS X Spaces 相同功能的实用程序。对于不了解它的人来说，这是一个允许你创建虚拟屏幕以避免所有窗口堆积的工具...</li><li><a href="https://superuser.com/questions/177041/what-is-the-equivalent-of-mac-os-x-spac">What is the equivalent of Mac OS X Spaces for Windows?</a>: 我正在寻找一个能实现 Mac OS X Spaces 相同功能的实用程序。对于不了解它的人来说，这是一个允许你创建虚拟屏幕以避免所有窗口堆积的工具...</li><li><a href="https://youtu.be/ujnLJru2LIs?si=8Cn_9t_2Rlyfo8GT"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=readme-ov-file#tutorials">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: MCP 服务器集合。通过在 GitHub 上创建一个账号来为 punkpeye/awesome-mcp-servers 做出贡献。</li><li><a href="https://youtu.be/ujnLJru2LIs?si=8C">Prompt Engineering Master Class for ENGINEERS with Ollama and LLM (Q4 2024 Update)</a>: 🚀 觉得 Prompt Engineering 还只是个流行语？先生，那绝对是错误的。随着 2025 年的临近，事实已经非常明确：精通 Prompt Engineering 是...</li><li><a href="https://github.com/punkpeye/awesome-mcp-servers?tab=read">GitHub - punkpeye/awesome-mcp-servers: A collection of MCP servers.</a>: MCP 服务器集合。通过在 GitHub 上创建一个账号来为 punkpeye/awesome-mcp-servers 做出贡献。</li><li><a href="https://github.com/jhgoodwin/FakeDbProvider">GitHub - jhgoodwin/FakeDbProvider: A fake provider for System.Data.Common.DbConnection and related classes.</a>: 一个用于 System.Data.Common.DbConnection 及相关类的伪造提供程序。 - jhgoodwin/FakeDbProvider</li><li><a href="https://github.com/orgs/modelcontextprotocol/discussions/88">What&#39;s the difference between MCP and vector database? · modelcontextprotocol · Discussion #88</a>: 已经有一段时间了，我还是没搞明白。
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1317239702063022101)** (96 条消息🔥🔥): 

> `Notebook LM 播客功能、定制 AI 输出、在 AI 中使用不同语言、利用 AI 创作引人入胜的内容、AI 与图灵测试` 


- **探索 Notebook LM 播客功能**：讨论了 Notebook LM 的最新功能，包括增强用户体验的定制化和交互功能。
   - 成员们分享了展示这些功能的播客链接，并断言该应用正在改变音频内容的格局。
- **为独特风格定制 AI 输出**：用户强调了优秀的 Prompt 引导和自定义功能对于调整 AI 输出的重要性，这可以产生多样的语调和风格。
   - 分享的 [YouTube 视频](https://youtu.be/aG0ixD3OY80?feature=shared) 提供了关于如何通过有效的 Prompt 技术获得艺术化效果的技巧。
- **AI 工具的双语和多语种用途**：关于如何以不同语言使用 Notebook LM 的问题不断出现，建议包括指示 AI 以特定语言进行回复。
   - 用户分享了引导 AI 进行多语言输出的方法，并强调了正确配置的必要性。
- **利用 AI 创作引人入胜的内容**：围绕使用 AI 生成迷人的音频叙事和内容展开了讨论，这些内容似乎引起了听众的良好共鸣。
   - 人们尝试了各种内容风格，包括模仿名人以及 ASMR 语调，以增强观众的参与度。
- **探索 AI 通过图灵测试的能力**：成员们讨论了 AI 在通过图灵测试方面面临的挑战，以及对话语调适配的重要性。
   - 分享的实验展示了不同的角色情绪如何影响 AI 的对话风格及其被感知的智能程度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://notebooklm.google.com/notebook/24b9048f-48be-417d-96f5-d288435fcc24/audio">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=aG0ixD3OY80"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/aG0ixD3OY80?feature=shared"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/jTVIOhuNy3Q?feature=shared"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/ytcHj-EllWo?feature=shared"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=yWRCpQBpd-k"> - YouTube</a>：未找到描述</li><li><a href="https://www.fxguide.com/fxpodcasts/zap-andersson-exploring-the-intersection-of-ai-and-rendering/">Zap Andersson：探索 AI 与渲染的交集</a>：Zap Andersson 分享了他为自己的怪诞 YouTube 系列《UNREAL MYSTERIES》测试 AI 工具时积累的技巧和心得。
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1317219744427937874)** (613 条消息🔥🔥🔥): 

> `NotebookLM 新功能、NotebookLM Plus、交互模式、播客生成、语言设置` 


- **NotebookLM Plus 推出状态**：用户目前正经历 NotebookLM Plus 新功能的缓慢推出，有些人已经获得访问权限，而有些人则没有，特别是在不同的 Google 账号之间。
   - 人们期待全面开放，目标是在 2025 年初面向 Google One Premium 用户推出。
- **新功能的用户体验**：用户对交互式音频概览功能的评价褒贬不一，一些人反映响应时间较慢，且 AI 主持人的参与感有所下降。
   - 改进响应速度的建议已得到认可，这表明官方正在进行持续调整以提升用户体验。
- **来源限制讨论**：讨论涉及 NotebookLM 免费版来源限制的增加（现设定为 300 个来源），同时用户对模型如何管理这一限制表示好奇。
   - 用户还在思考如何收集足够的来源以有效利用这一功能。
- **法语用户的语言设置**：一位讲法语的用户询问如何更改 NotebookLM 的语言设置，表示 Prompt 得到的是法语回复而非英语。
   - 建议用户可能需要调整其 Google 账号的语言设置，以匹配其期望的回复语言。
- **功能请求与改进**：用户对播客的各种改进表示感兴趣，例如添加音频片段和增加语音控制选项。
   - 社区鼓励提交反馈并参与特定请求，以改进 NotebookLM 功能的未来迭代。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/traviscline/status/1868093820581343740?s=46">来自 tmc (the/acc) (@traviscline) 的推文</a>：互动式 NotebookLM 模式的小预览。正在讨论 NotebookLM CLI！#notebooklm</li><li><a href="https://book-a-painter.com/">未找到标题</a>：未找到描述</li><li><a href="https://elevenreader.io/app/reader/genfm/e2771eb8df2252e96cbe43f47a2bf4b023cf392e62a0e1af77d4ad7e73dd5562/u:o1kc1ElI1uo7j4h1ux4J">ElevenReader 的 GenFM：通过税务知识解锁财务自由</a>：收听此 GenFM 播客的预览，或下载应用程序播放完整剧集并创建您自己的内容。</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://support.google.com/notebooklm/answer/15678219?visit_id=638697853454981673-3976970542&p=plus&rd=1">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://streamable.com/4thv4x">观看 deepdive-google_U8gkTzEC | Streamable</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219">升级到 NotebookLM Plus - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://pastebin.com/gDFFrr4M">Pastebin.com - 阅后即焚粘贴</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://ai.google.dev/gemini-api/docs/available-regions">未找到标题</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/14276570?hl=en">社区 - NotebookLM 帮助</a>：未找到描述</li><li><a href="https://blog.google/technology/google-labs/notebooklm-new-features-december-2024/">NotebookLM 获得新外观、音频交互功能和高级版本</a>：NotebookLM 正在推出新功能以及名为 NotebookLM Plus 的高级版本。</li><li><a href="https://apps.google.com/supportwidget/articlehome?hl=en&article_url=https%3A%2F%2Fsupport.google.com%2Fa%2Fanswer%2F9212585%3Fhl%3Den&assistant_id=generic-unu&product_context=9212585&product_name=UnuFlow&trigger_context=a">未找到标题</a>：未找到描述</li><li><a href="https://youtu.be/nqDXv6dnlls)"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Xv4_ToKF66U&t=3454s"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/GMe2JoTymRY?si=vyo8lJ6rDjwN1zZJ"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/gv92pUahxVQ)"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/CzBBhytDzM4?si=VFJM_ZN9"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/CzBBhytDzM4?si=VFJM_ZN918XpOK33"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=EZGskiGkkSA"> - YouTube</a>：未找到描述</li><li><a href="https://notebooklm.google/plus/terms">Google NotebookLM | AI 驱动的笔记与研究助手</a>：利用 AI 的力量进行快速总结和笔记，NotebookLM 是您强大的虚拟研究助手，植根于您可信赖的信息。</li><li><a href="https://cloud.google.com/terms">未找到标题</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise">比较 Google Workspace 的 Gemini 插件 - 商业 / 企业 - Google Workspace 管理员帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise#other">比较 Google Workspace 的 Gemini 插件 - 商业 / 企业 - Google Workspace 管理员帮助</a>：未找到描述</li><li><a href="https://support.google.com/a/answer/14700766?hl=en&co=DASHER._Family%3DBusiness-Enterprise#availability">比较 Google Workspace 的 Gemini 插件 - 商业 / 企业 - Google Workspace 管理员帮助</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1317221075238785065)** (884 条消息🔥🔥🔥): 

> `Cursor IDE 性能、AI 模型比较、社交媒体项目开发、Cursor 集成、聊天管理问题`

- **Cursor IDE 性能问题**：用户报告了 Cursor IDE 的卡顿现象，尤其是在长时间处理应用程序时，引发了关于需要重置或清除聊天记录的讨论。
   - 一些人建议创建新的聊天会话以缓解性能问题，旨在实现更高效的工作流。
- **AI 模型之间的比较**：参与者讨论了不同 AI 模型的优缺点，例如 Cursor 的 Agent 与 Gemini 1206，强调了它们各自的能力和性能。
   - 用户指出，虽然 Cursor 保持了用户友好的界面，但 Gemini 在编码任务中表现出强大的性能，使其成为与 Cursor 并行的有价值工具。
- **社交媒体平台的开发**：几位用户表达了构建社交媒体平台的兴趣，讨论了必要的后端结构和实施的潜在框架。
   - 强调了创建此类平台需要理解 CRUD 操作并管理数据库关系，利用 Cursor 等工具来提高效率。
- **Cursor 与其他工具的集成**：有建议提出 Cursor 应与 Supabase 和 Bolt 等其他平台集成，以增强其功能并简化用户的工作流。
   - 用户讨论了此类集成的优势，以及它们如何简化开发过程。
- **关于聊天管理的反馈**：关于 Cursor 聊天管理的反馈揭示了在编辑消息时丢失上下文和之前消息的挫败感。
   - 用户提出了改进建议，例如在编辑后保留聊天历史记录，类似于 ChatGPT 和 Claude 等其他平台的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/NVIDIAAIDev/status/1868778156347339033">来自 NVIDIA AI Developer (@NVIDIAAIDev) 的推文</a>：👀 RAPIDS cuDF 在无需更改代码的情况下将 #pandas 加速高达 150 倍。现在，随着数据集大小增长到 GB 级别，你可以继续使用 pandas。⚡ ➡️ 尝试演示的 Jupyter Notebook：http://nvda.ws...</li><li><a href="https://docs.cursor.com/get-started/usage#premium-models">Cursor - 更快地构建软件</a>：未找到描述</li><li><a href="https://x.com/skcd42/status/1867561917159755942?s=19">来自 skcd (@skcd42) 的推文</a>：CodeStory agent 现在在 swebench-verified 上达到 SOTA，解决率为 62.2%。我们通过在测试时推理（test time inference）上扩展我们的 agent 并重新学习“苦涩的教训”（bitter lesson）实现了这一目标。Sonnet3.5(new) 是我们唯一的 LLM...</li><li><a href="https://aistudio.google.com/">Google AI Studio</a>：Google AI Studio 是开始使用 Gemini（我们下一代多模态生成式 AI 模型系列）进行构建的最快方式。</li><li><a href="https://www.cursor.com/pricing">价格 | Cursor - AI 代码编辑器</a>：选择适合您的方案。</li><li><a href="https://cursor.com/settings">设置 | Cursor - AI 代码编辑器</a>：您可以在此处管理您的账户、账单和团队设置。</li><li><a href="https://forum.cursor.com/t/how-to-do-fix-in-composer-and-fix-in-chat-actions-from-keyboard/31221">如何通过键盘执行 `Fix in Composer` 和 `Fix in Chat` 操作</a>：这两个：我在设置中找不到。</li><li><a href="https://letmegooglethat.com/?q=Warp&l=1">Warp</a>：未找到描述</li><li><a href="https://x.com/hive_ech">来自 FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://x.com/vadi_ms/status/1867395672418529623">来自 Vadims (@vadi_ms) 的推文</a>：我分析了 http://bolt.new 并发现了如何使用 Cursor 实现同样高质量的设计。第一张图来自 Bolt，第二张来自 Cursor，使用的是相同的提示词。下方是 3 步指南 ⤵️：</li><li><a href="https://marketplace.visualstudio.com/items?itemName=SpecStory.specstory-vscode">SpecStory (Cursor Extension) - Visual Studio Marketplace</a>：Visual Studio Code 扩展 - (Cursor Extension) 捕捉、搜索并从每一次 AI 编程之旅中学习</li><li><a href="https://v0.dev/">v0 by Vercel</a>：与 v0 聊天。通过简单的文本提示生成 UI。复制、粘贴、发布。</li><li><a href="https://x.com/hive_echo/status/1865598500060508183">来自 echo.hive (@hive_echo) 的推文</a>：即将来到你身边的 Cursor (很快...) ⚡ Yolo mode (自动命令执行) 🤝 Unification (聊天和 composer 合二为一)</li><li><a href="https://openrouter.ai/">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://youtube.com/shorts/8WMk8E4KD5Q?si=8BJKbqipxOdOY7gm">修复了 Visual Studio Code 中的 Live Server 问题！#vscode #liveserver</a>：修复了 Visual Studio Code 中的 Live Server 问题！大家好！欢迎回到另一个快速简洁的 YouTube Short！今天，我们将深入探讨...</li><li><a href="https://github.com/mastodon/mastodon">GitHub - mastodon/mastodon：您的自托管、全球互联的微型博客社区</a>：您的自托管、全球互联的微型博客社区 - mastodon/mastodon</li><li><a href="https://github.com/jnsahaj/lumen/pull/19">feat: 由 lkonga 添加 OpenRouter AI 提供商支持 · Pull Request #19 · jnsahaj/lumen</a>：添加了对 openrouter.ai 作为新提供商的支持，允许用户访问各种 AI 模型</li><li><a href="https://store.crowdin.com/openrouter?utm_source=chatgpt.com">OpenRouter - Crowdin Marketplace</a>：比较 LLM 和价格以获得最佳性能。</li><li><a href="https://docs.vapi.ai/providers/model/openrouter?utm_source=chatgpt.com">OpenRouter — Vapi</a>：什么是 OpenRouter？</li><li><a href="https://creati.ai/ai-tools/openrouter-ai/">OpenRouter：AI 模型的统一接口 | Creati.ai</a>：探索 OpenRouter，一个提供广泛 AI 模型的统一接口。通过无缝集成优化性能和成本。</li><li><a href="https://aipure.ai/articles/openrouter-review-revolutionizing-ai-language-model-access?utm_source=chatgpt.com">OpenRouter 评论：彻底改变 AI 语言模型访问</a>：探索我们全面的 OpenRouter 评论。了解这一统一接口如何通过多样化的模型和具有成本效益的解决方案改变 AI 的可访问性。
</li>
</ul>

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1317241044420661248)** (544 条消息🔥🔥🔥): 

> `Unsloth 模型支持、依赖与安装问题、Triton 安装、长上下文模型、Ilya Sutskever 演讲见解` 


- **Unsloth 模型与 Triton 兼容性**：用户报告了由于与 Triton 的依赖冲突导致安装 Unsloth 出现问题，指出需要安装正确的版本以确保兼容性。
   - 记录了安装挑战，特别是针对 Python 3.13，建议使用 Python 3.10 并通过 Conda 安装以获得更好的兼容性。
- **长上下文模型效率**：讨论强调了长上下文模型的局限性，强调数据过滤非常复杂，且质量不能唯一决定训练效率。
   - 参与者指出，排除“坏数据”可能会对理解能力产生负面影响，因为从多样化的数据集中学习对模型开发至关重要。
- **来自 Ilya Sutskever 演讲的见解**：一条推文讨论了 Ilya 关于 AI 扩展（scaling）的见解，强调除了数据量之外，还需寻找改进扩展的替代方法。
   - 针对 AI 开发挑战的过度简化表达了批评，质疑了模型训练中“坏数据”的定义及其必要性。
- **社区经验与建议**：成员们分享了使用 vllm 和 Docker 等各种平台的经验，强调了在 AI 建模中使用本地环境与云环境的实际操作层面。
   - 讨论还涉及了 AI 开发的硬件存储挑战，用户提到了 AI 训练中巨大的数据存储需求。
- **通用模型优化与挑战**：对话探讨了优化大参数量模型的困难，以及与存储和性能相关的挑战。
   - 成员们讨论了 AI 领域持续创新的必要性，并对当前技术已达极限的说法表示怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/triton/,">未找到标题</a>: 未找到描述</li><li><a href="https://dev.to/dineshgdk/is-progress-bar-tqdm-killing-your-code-42oj">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1VA13NvMor9TxHBEDFXYewgu4jrXVQvFZ?usp=sharing#scrollTo=bu-_d4YP_CkR">Google Colab</a>: 未找到描述</li><li><a href="https://nnsight.net/">nnsight &#8212; nnsight</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1868748998783517093">Daniel Han (@danielhanchen) 的推文</a>: 我对后预训练（Post Pretraining）世界的看法 - Ilya 的演讲：Ilya 暗示我们需要寻找其他东西来扩展——演讲中的脑身体质量比图表显示，人类智能的“扩展”比……更好。</li><li><a href="https://huggingface.co/THUDM/glm-4-9b-chat-1m">THUDM/glm-4-9b-chat-1m · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-bnb-4bit">unsloth/Llama-3.3-70B-Instruct-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-1B-Instruct-GGUF">unsloth/Llama-3.2-1B-Instruct-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct">HuggingFaceTB/SmolVLM-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/THUDM/GLM-4-Voice/issues/133">layer 40 / logits all nan · Issue #133 · THUDM/GLM-4-Voice</a>: 这很奇怪.. 我正尝试在模型上进行 abliteration / 微调，但它的表现与 glm4-chat 相当不同 / 我的方法在 chat 上有效，除了 16k+ 音频 t... 之外的核心区别是什么？</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - 动态 4-bit 量化</a>: Unsloth 的动态 4-bit 量化选择性地避免量化某些参数。这在保持与 BnB 4bit 相似的 VRAM 占用的同时，极大地提高了准确性。</li><li><a href="https://youtu.be/jFl5Fewrieo"> - YouTube</a>: 未找到描述</li><li><a href="https://github.com/magicproduct/hash-hop">GitHub - magicproduct/hash-hop: 大语言模型的长上下文评估</a>: 大语言模型的长上下文评估。通过在 GitHub 上创建账户来为 magicproduct/hash-hop 的开发做出贡献。</li><li><a href="https://github.com/arcee-ai/DAM">GitHub - arcee-ai/DAM</a>: 通过在 GitHub 上创建账户来为 arcee-ai/DAM 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/)** (1 条消息): 

edd0302: https://main-horse.github.io/posts/visualizing-6d
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1317229390924484640)** (236 条消息🔥🔥): 

> `Unsloth 训练问题、模型与 Streamlit 的兼容性、数据集加载问题、微调技术、Llama 3.2 的最大序列长度` 


- **Unsloth 训练启动检查**：一位用户询问特定屏幕是否意味着训练已成功开始，并展示了一张图片进行确认。
   - 社区成员就潜在的初始化方法和训练性能改进提供了见解。
- **Lora+ 与 Unsloth 的兼容性**：一位用户询问了 Lora+ 和 Unsloth 的使用经验，在尝试之前寻求关于两者是否存在根本性不兼容的信息。
   - 提供了外部资源和博客见解的引用，以阐明不同微调方法的有效性。
- **数据集加载挑战**：用户在加载数据集时遇到问题，包括找不到数据文件以及无法正确处理 CSV 格式。
   - 建议包括使用正确的加载语法并检查文件路径以解决 FileNotFoundError。
- **在 Streamlit 中使用微调后的模型**：一位用户寻求帮助，将保存在 Hugging Face 上的微调后的 Llama 3.1 模型连接到 Streamlit，但遇到了模型识别错误。
   - 社区成员澄清说，保存的模型配置可能需要合并或与基础模型一起正确加载。
- **Llama 3.2 的最大序列长度**：一位用户询问了 Llama 3.2 的最大序列长度，认为可能是 4096。
   - 另一位用户纠正了这一点，指出实际的最大长度是 131072。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/reward-modelling-dpo-orpo-and-kto">奖励建模 - DPO, ORPO &amp; KTO | Unsloth 文档</a>：要在 Unsloth 中使用 DPO, ORPO 或 KTO，请遵循以下步骤：</li><li><a href="https://docs.unsloth.ai/basics/vision-fine-tuning">视觉微调 | Unsloth 文档</a>：关于使用 Unsloth 进行视觉/多模态微调的详细信息</li><li><a href="https://huggingface.co/unsloth/SmolLM2-1.7B-Instruct-GGUF">unsloth/SmolLM2-1.7B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://tenor.com/view/spider-man-uncle-ben-with-great-power-comes-great-responsibility-its-true-just-saying-gif-24193883">蜘蛛侠本叔叔 GIF - 能力越大，责任越大 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1lN6hPQveB_mHSnTOYifygFcrO8C1bxq4?usp=sharing#scrollTo=Edrn7Rxmojtu>">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>：通过使用 Unsloth 对 Llama 3, Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://github.com/unslothai/unsloth/pull/730">在 `save.py` 中将模型转换命令更新为 `convert_hf_to_gguf.py`，由 malibayram 提交 · Pull Request #730 · unslothai/unsloth</a>：更新了 save.py 中的模型转换命令以使用 convert_hf_to_gguf.py，与最新工具保持一致...</li><li><a href="https://github.com/unslothai/unsloth?t">GitHub - unslothai/unsloth: 微调 Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 70%</a>：微调 Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#-documentation)">GitHub - unslothai/unsloth: 微调 Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 70%</a>：微调 Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 70% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1317388351678054491)** (24 条消息🔥): 

> `模型合并技术、AI 监管与政治、AI 对社会的影响、核条约类比、对 AI 收益的认知` 


- **Differentiable Adaptive Merging (DAM) 论文亮点**：该论文讨论了在无需大量重新训练的情况下合并模型以平衡各项能力，并引入了 [Differentiable Adaptive Merging (DAM)](https://github.com/arcee-ai/DAM) 作为一种高效的模型集成方法。
   - 论文强调，当模型相似度较高时，像 **Model Soups** 这样更简单的方法也能表现良好，展示了不同技术各自的独特优势。
- **AI 监管讨论引发辩论**：成员们对政府**监管 AI** 的能力表示怀疑，并将其与过去在社交媒体方面的监管尝试进行对比，强调了法律格局的复杂性。
   - 讨论中有一种观点认为，极端的监管可能会像钟摆一样摆动，在多次反复后最终回归到“理性地带”。
- **AI 的显性收益影响各行各业**：大家一致认为 **AI 带来的收益** 在各行各业已经显而易见，AI 被描述为一种极大地提高了生产力的神奇工具。
   - 有人担心，人类可能低估了未来控制潜在超智能 AI 的能力。
- **AI 治理的核条约类比**：一位成员提议，可能有必要建立一个类似于**核动力条约**的 AI 治理条约，以确保安全和问责。
   - 讨论强调了使 AI 的潜在威胁可见化的挑战，以及控制先进 AI 系统的复杂性。
- **关于人类长期生存的辩论**：在讨论中，一位成员指出 AI 带来的社会变革可能超出了当前的理解范围，并对 AI 进步的影响发出了警告。
   - 人们担心人类是否能够生存足够长的时间，以管理未来几十年可能出现的智能系统。



**提及的链接**：<a href="https://arxiv.org/abs/2410.08371">Merging in a Bottle: Differentiable Adaptive Merging (DAM) and the Path from Averaging to Automation</a>：通过合并模型，AI 系统可以结合不同语言模型的独特优势，在多种能力之间实现平衡，而无需大量的重新训练。然而，i...

  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1318274419596202127)** (1 条消息): 

> `ChatGPT Search Day, 12 Days of OpenAI` 


- **庆祝 ChatGPT Search Day**：**12 Days of OpenAI** 的第 8 天是 **ChatGPT Search Day**，活动旨在鼓励社区参与。
   - 为了获取最新动态，邀请成员在 <id:customize> 中领取 <@&1261377106890199132> 角色。
- **查看 YouTube 视频**：为有兴趣了解当天活动的观众推荐了一个 [YouTube 视频](https://www.youtube.com/watch?v=OzgNJJ2ErEE)。
   - 遗憾的是，未提供关于视频内容的描述或进一步详情。



**提及的链接**：<a href="https://www.youtube.com/watch?v=OzgNJJ2ErEE"> - YouTube</a>：未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1317221475677110384)** (614 条消息🔥🔥🔥): 

> `Character AI 性能, OpenAI 与 Alignment, 新 AI 模型, 本地 LLMs, AI 与政治` 


- **关于 Character AI 退步的讨论**：用户对 Character AI 表示不满，指出其性能下降，且应用营销转向儿童带来了负面影响，导致过滤器更严格、上下文能力降低。
   - 相比之下，用户发现 ChatGPT 更适合创意任务，尤其是在角色扮演场景中。
- **AI Alignment 框架讨论**：一位用户分享了一个关于 AI Alignment 的工作框架，强调基于共同人类价值观和迭代反馈的原则，以确保 AI 开发的包容性。
   - 对话强调了让各利益相关方就 Alignment 原则达成一致的挑战，一位用户质疑了这一目标的可行性。
- **新兴 AI 模型**：人们对 Google 的 Gemini 等新 AI 模型以及 Imagen 的更新表现出兴趣，用户讨论了与 OpenAI 的 4o 等现有模型的性能对比。
   - 用户指出，虽然像 Grok 这样的模型正在取得进展，但仍落后于 ChatGPT 等更成熟的选择。
- **本地 LLMs 讨论**：参与者讨论了本地 LLMs 的优势，认为与大型科技公司的解决方案相比，它们可以提供更具定制化和灵活性的 AI 体验。
   - 有人担心大科技公司可能主要关注生产力的提高，而不是增强 AI 交互中的创造力。
- **Discord 频道氛围**：频道的整体情绪表明，讨论正转向不必要的政治话题，这让更倾向于以 AI 为中心对话的用户感到沮丧。
   - 一些用户开玩笑地提到了频道混乱的基调和复杂的反应，表示这在当天营造了一种有趣的氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/vegeta-its-over9000-gif-14419267">Vegeta Its Over9000 GIF - Vegeta Its Over9000 - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/">OpenAI 举报者被发现死于旧金山公寓</a>: 一位曾对公司表示担忧的前 OpenAI 研究员去世，年仅 26 岁。</li><li><a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: 促进全球对伦理 AI Alignment 的意识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。</a>: 促进全球对伦理 AI Alignment 的意识和行动，保护人类免受 AI 自自我复制风险。包括研究、框架和开源资源。 - AlignAGI/Alig...
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1317312280672731176)** (24 条消息🔥): 

> `O1 Pro AI, OpenAI 订阅讨论, 使用 GPT 下棋, LLMs 与计算, GPT 4o 对比 GPT 4o-mini` 


- **O1 Pro：AI 女友困境**：成员们讨论了 **O1 Pro**，有人称其为**最好的 AI 女友**，而另一位则强调其 **200 美元** 的定价太高。
   - 一位用户幽默地评论了潜在的等待时间，暗示它让用户等了很久才回复。
- **OpenAI 订阅：物有所值吗？**：关于 **OpenAI 订阅** 价值的担忧出现，有人建议投资现实生活（IRL）的约会体验可能是更好的选择。
   - 另一位成员反思了他们对在免费期间没有进行更多微调（fine-tuning）的遗憾，并认可了通过 API 获得的不错使用量。
- **GPT 下棋难题**：一位用户分享了使用 GPT **下棋** 的经历，提到了棋子重复的问题，这引发了关于 LLM 能力的讨论。
   - 另一位强调了 LLMs 在下棋等游戏的逻辑推理方面的局限性，而其他人则指出 **Python 库** 可以辅助处理象棋逻辑。
- **能力差距：GPT 4o 对比 GPT 4o-mini**：对于 **GPT 4o** 和 **GPT 4o-mini** 之间的性能差异，用户表达了沮丧，声称 mini 版本感觉像是在**梦游**。
   - 成员们认为 4o-mini 的响应明显比 4o 主模型差，表明质量有明显下降。
- **倒计时公告**：围绕 8 号可能发布的**公告**，期待感在增加，一位成员确认这将在 **20 多分钟** 后发生。
   - 这在社区中引起了兴奋，大家都在等待有关潜在更新或功能的最新消息。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1317405015660560394)** (67 messages🔥🔥): 

> `Prompt Engineering Techniques, AI Model Capabilities in Coding, Learning Programming, Memory Management in AI, Creating a Curriculum for Prompt Engineering` 


- **提升 Prompt Engineering 技能**：用户讨论了精炼其 Prompt Engineering 技能的方法，强调了准确了解自己对 AI 需求的重要性，并将其比作烹饪：根据情况，人们可以依赖预制菜肴，也可以从零开始烹饪。
   - 讨论中澄清了，无论编程经验如何，理解语言并提供清晰的指令是实现高效 Prompting 的关键。
- **利用 AI 辅助编程**：一位用户表示有兴趣利用 ChatGPT 编写代码并在自己的 IDE 中使用，特别渴望看到其在开发现代网站方面的能力。
   - 建议用户提供关于其当前编程经验和预期的详细信息，这可以帮助 AI 为项目开发提供更具针对性的指导。
- **Memory 与自定义指令**：关于 AI Memory 系统的讨论指出，用户可以更新 AI 对其偏好和先前 Prompt 的记忆，从而实现更个性化的交互。
   - 建议在有效利用存储的 Memory 的同时，也要认识到 Memory Management 的局限性及可用的变通方案。
- **潜在的 Prompt Engineering 课程体系**：一位用户分享了围绕 Prompt Engineering 开发课程体系的抱负，并寻求有关该主题现有课程和资源的信息。
   - 建议强调了在 Prompting 中设定明确目标的重要性，以及学习编程如何能增强与 AI 有效沟通的能力。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1317405015660560394)** (67 messages🔥🔥): 

> `Prompt Engineering, Using ChatGPT for Coding, Memory Management, Prompt Library Concept, Learning Programming Languages` 


- **理解 Prompt Engineering**：成员们讨论了在 Prompt Engineering 中构建精确 Prompt 的重要性，强调明确你对模型的期望至关重要。对 Prompt 有效性的探索表明，定制化的 Prompt 可以带来更准确、更有用的输出。
- **利用 ChatGPT 进行编程**：一位用户询问了使用 ChatGPT 编写代码并在 IDE 中使用的最佳实践，表达了探索模型能力的兴趣。建议用户提供关于其经验水平和所用工具的清晰说明，以获得最佳效果。
- **整合 Memory 空间**：关于 Memory Management 的讨论揭示了在模型中高效使用 Memory 空间的技巧，例如总结并反馈重要信息。成员们分享到，用户不需要过度担心 Memory 限制，因为存在各种变通方法。
- **Prompt Library 概念**：一位用户询问维护一个 Prompt Library 是否类似于用过去的 Prompt 更新模型的 Memory。成员们讨论了 Prompt Library 的非正式性质，并指向了一个用于探索 Prompt Engineering 的共享频道。
- **学习编程语言**：对话强调了一位成员的观点，即学习编程可能是不必要的，因为 ChatGPT 可以帮助有效编码。然而，有人指出，具备基础理解有助于在与模型协作时更好地沟通需求并评估输出。


  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1317252744003846184)** (327 messages🔥🔥): 

> `AI Government Regulation, Apollo LMMs Release, Hermes 3 Key Access, Model Performance Issues, Community Involvement in AI`

- **政府对 AI 监管的担忧**：Elon Musk 强调，美国政府可能会限制 AI 初创公司，并控制围绕 AI 技术的叙事，以防止独立倡议的出现。
   - 人们对政府合作伙伴关系和监管可能导致 AI 开发中的垄断表示担忧，这会对较小的参与者不利。
- **Apollo LMMs 的发布**：社区讨论了 Apollo LMMs 的最新更新，其中包括专注于视频理解和多模态（multimodal）能力的模型。
   - 对 Apollo 模型的初步印象表明其表现良好，引发了对其潜在应用的兴趣。
- **Hermes 3 的访问与问题**：用户正在寻求访问 Hermes 3，但被告知目前没有可用的密钥，且由于模型问题，故障排除工作正在进行中。
   - 开发人员已意识到影响 Hermes 3 的问题，并计划实施修复，包括对聊天模板（chat template）的调整。
- **AI 模型的性能问题**：用户报告了不同 AI 模型的各种行为和问题，表明某些脚本可能需要重新运行或更新。
   - 对于解决方案的漫长等待时间，人们仍然感到担忧，特别是对于在可信执行环境（TEEs）下运行的模型。
- **AI 开发中的社区协作**：讨论表明，社区有强烈的愿望利用分布式计算（distributed computing）在 AI 训练和开发方面进行协作。
   - 社区对开源贡献表示乐观，强调创新的想法可以从集体努力中产生。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/TheInsiderPaper/status/1867728290066153576">来自 Insider Paper (@TheInsiderPaper) 的推文</a>: 突发：OpenAI 举报者被发现死于旧金山公寓 — TechCrunch</li><li><a href="https://jplhughes.github.io/bon-jailbreaking/">Best-of-N Jailbreaking</a>: 未找到描述</li><li><a href="https://www.mercurynews.com/2024/12/13/openai-whistleblower-found-dead-in-san-francisco-apartment/">OpenAI 举报者被发现死于旧金山公寓</a>: 一位曾对公司提出质疑的前 OpenAI 研究员去世，年仅 26 岁。</li><li><a href="https://x.com/elonmusk/status/1868302204370854026?s=46">来自 Elon Musk (@elonmusk) 的推文</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/Apollo-LMMs/Apollo-3B">Apollo 3B - Apollo-LMMs 在 Hugging Face 上的 Space</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/m-ric/471403804474189">Hugging Face 上的 @m-ric："LLM 的潜在范式转移：新的…"</a>: 未找到描述</li><li><a href="https://www.promptingguide.ai/research/llm-agents">Prompt Engineering 指南</a>: Prompt Engineering 全面概述</li><li><a href="https://x.com/N8Programs/status/1868082092263010791">来自 N8 Programs (@N8Programs) 的推文</a>: 好的，Cohere 的模型基本上使用了以下设置：3 层局部注意力（带有 ROPE 的 4096 滑动窗口），1 层全局注意力（无位置编码），重复 8 次。我们所做的是保持...</li><li><a href="https://x.com/NousResearch/status/1848397863547515216">来自 Nous Research (@NousResearch) 的推文</a>: 未找到描述</li><li><a href="https://x.com/N8Programs/status/1868071000430321763">来自 N8 Programs (@N8Programs) 的推文</a>: 这是 Cohere 的 7B 模型，采用 4-bit 量化和 4-bit KVCache，正在总结整部《哈利·波特》—— 包含 11.5 万个 token 的上下文 —— 速度为 13tok/sec，Prompt 处理速度约 181tok/sec...</li><li><a href="https://huggingface.co/Apollo-LMMs">Apollo-LMMs (Apollo-LMMs)</a>: 未找到描述</li><li><a href="https://apollo-lmms.github.io/">Apollo</a>: Apollo：大型多模态模型中视频理解的探索</li><li><a href="https://www.lesswrong.com/tag/alignment-tax">Alignment Tax - LessWrong</a>: Alignment Tax（对齐税，有时称为安全税）是确保 AI 系统对齐所需的额外成本，相对于构建未对齐替代方案的成本而言。“税”这个词可能会产生误导...</li><li><a href="https://devclass.com/2024/12/12/sqlite-re-implemented-in-rust-to-achieve-asynchronous-i-o-and-other-changes/">用 Rust 重新实现 SQLite 以实现异步 I/O 及其他改进 • DEVCLASS</a>: 专注于数据库解决方案的开发者 Turso 正在用 Rust 重新实现 SQLite 数据库引擎，以便 [...]</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/YJHr2iAdL8">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://x.com/bitcloud/status/1868729306492674170">来自 Lachlan Phillips exo/acc 👾 (@bitcloud) 的推文</a>: 军事和 CIA 合作伙伴关系，监管俘获尝试。当你向政府发布前沿模型时，在平民和政府之间制造了不断扩大的权力和能力差距...</li><li><a href="https://github.com/arcee-ai/DAM">GitHub - arcee-ai/DAM</a>: 通过在 GitHub 上创建账户来为 arcee-ai/DAM 的开发做出贡献。</li><li><a href="https://youtu.be/Pz9YeBs_afo?t=782)."> - YouTube</a>: 未找到描述</li><li><a href="https://devclass.com/2024/12/12/sqlite-re-implemented-in-rust-to-achieve-asynchronous-i-o-and-other-">用 Rust 重新实现 SQLite 以实现异步 I/O 及其他改进 • DEVCLASS</a>: 专注于数据库解决方案的开发者 Turso 正在用 Rust 重新实现 SQLite 数据库引擎，以便 [...]
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1317228428973445222)** (32 条消息🔥): 

> `开源编程 LLMs, 微调本地 LLMs, 向量数据库与 embeddings, 模型合并与 souping, LLMs 中的 RNG 算法` 


- **适用于编程的开源 LLM**：成员建议了几种开源编程 LLM，如 **Mistral Codestral**、**Qwen 2.5 Coder** 和 **DeepSeek**，这些模型可以与 VS Code 和 PyCharm 等 IDE 以及 [continue.dev](https://continue.dev) 等扩展程序集成。
   - 这些工具使开发人员能够使用本地模型提高编码效率。
- **微调本地 LLM 是可行的**：一位用户询问了微调本地 LLM 的可能性，并获知借助 **unsloth** 和 **axolotl** 等工具，即使是资深技术爱好者也有可能使用 **QLoRA** 训练高达 8B 参数的模型。
   - 越来越多的资源使得那些愿意学习的人可以进行自定义。
- **关于使用向量数据库的辩论**：针对结构化产品数据的向量数据库最佳使用方式展开了讨论，建议评估像 **BM25** 这样更简单的搜索方法，而不仅仅是依赖 embeddings。
   - 一位成员解释了为什么 embeddings 可能无法有效地适应结构化查询，并指出可以优先考虑更高的检索准确性。
- **模型合并与 souping 的现状**：成员们讨论了模型合并（通常称为 **model souping**）的当前趋势，指出许多流行模型都是现有模型的组合，这引发了对其有效性的质疑。
   - 尽管对涉及的潜在风险仍有疑虑，但许多人承认，在受限条件下，这种方法仍然取得了积极成果。
- **理解 LLM 中的 RNG 算法**：有人提出了关于 LLM 中使用的随机数生成 (RNG) 算法的问题，以及它们在生成输出时通常是否部署 **Xorshift** 或其他算法。
   - 寻求关于它们应用的澄清，特别是在采样和分布阶段。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2203.05482#">Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time</a>：最大化模型准确率的传统配方是 (1) 使用各种超参数训练多个模型，以及 (2) 选择在留出验证集上表现最好的单个模型...</li><li><a href="https://arxiv.org/abs/2203.0548">An Adaptable and Agnostic Flow Scheduling Approach for Data Center Networks</a>：云应用重塑了互联网的服务模式和基础设施。搜索引擎、社交网络、内容分发以及零售和电子商务网站都属于这类应用...</li><li><a href="https://github.com/troy12x/Quasar-1">GitHub - troy12x/Quasar-1: Quasar-1 is a large language model architecture that moves beyond prompt engineering to achieve genuine reasoning capabilities. Unlike traditional LLMs that rely heavily on carefully crafted prompts, Quasar-1 implements reasoning at its architectural core through temperature-guided reasoning and guided sequence of thoughts.</a>：Quasar-1 是一种大语言模型架构，超越了提示工程，实现了真正的推理能力。与严重依赖精心设计提示的传统 LLM 不同，Quasar-1 通过温度引导推理和引导思维序列在架构核心实现了推理。
</li>
</ul>

</div>

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1317568067357638819)** (18 条消息🔥): 

> `模型压缩技术、通信理论在 AI 中的应用、模型训练中的 LoRA 更新、训练方法的权衡、MLP 中的位置不变性` 


- **通信理论增强 AI 模型**：讨论集中在**通信理论**原理如何影响 **LLMs** 的发展，特别是在分布式训练期间的梯度传输方面。
   - 成员们指出，**用计算换带宽**可以简化流程，尽管结合多种技术可能会很复杂。
- **高效编解码的挑战**：虽然**解码（decoding）**技术非常迅速，但**编码（encoding）**过程需要使用 Viterbi 算法解决优化问题，增加了实现难度。
   - 参与者质疑在模型训练期间加入**压缩方法**以提高数据效率而不损害性能的可行性。
- **训练中的动态 LoRA 使用**：成员们探讨了 **LoRA 更新**如何通过时间换取显存效率，建议采用顺序训练过程而非并行更新。
   - *固定 LoRA 在预训练期间会失效*，但通过重新初始化，模型可以保持灵活性并适应新数据。
- **位置不变性与冗余**：Real.azure 强调，目前似乎很少有人关注 **MLPs** 的**位置不变性**，即在投影块中更改权重顺序不会影响性能。
   - 这为神经架构内的**信息冗余**研究提供了一个潜在领域。
- **网格编码（Trellis Coding）的历史**：分享了关于**网格编码**的有趣概述，阐述了尽管它具有基础性的重要意义，但在引入标准时却有所延迟。
   - 成员们讨论了优化此类技术如何能为 AI 模型带来**跨学科**的进步。



**提及的链接**：<a href="https://x.com/OpenlifesciAI/status/1867999825721242101>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：🌟 每周医学 AI 研究汇总 🌟📅 2024年12月7-14日。这是您每周最重要的医学 AI 论文摘要！🎉🤖 医学 LLM 及其他模型 - PediaBench：中文儿科 LLM-...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1317236917783232523)** (3 条消息): 

> `Byte Latent Transformer, 动态分词, 推理效率, Llama 3 基准测试, 字节级模型` 


- **Meta 的 Byte Latent Transformer 颠覆分词技术**：Meta 刚刚发布了 **Byte Latent Transformer (BLT)**，这是一种无分词器（tokenizer-free）架构，可将 Byte 动态编码为 Patch，从而提高**推理效率**和鲁棒性。
   - *“这简直像过圣诞节一样！”* 一位成员表示，对训练期间学习动态分词的需求感到兴奋。
- **BLT 在规模上与 Llama 3 竞争**：BLT 模型声称可以匹配像 **Llama 3** 这样基于分词的模型性能，同时可能减少高达 **50%** 的**推理 FLOPs**。
   - 他们强调 BLT 可以在 **1T tokens** 上训练 **Llama-3 8B** 模型，表现优于使用 BPE 的标准架构。
- **对字节级模型训练效率的怀疑**：一位成员提到，虽然**字节级模型**的训练效率与 **BPE 模型**相当，但最大的字节级 LLM 仅有 **350M 参数**，且是在有限的数据集上训练的。
   - 他们质疑道：*“我们到底什么时候才能彻底抛弃分词（tokenization）？”* 反映了对分词技术未来的怀疑。
- **BLT 声明的验证**：另一位成员确认有关 **BLT** 的信息确实是**真实的**，增强了对新模型潜力的信心。
   - 这一肯定是在围绕该模型的能力和基准测试进行讨论后提出的。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/scaling01/status/1867573707247346003?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：META 刚刚杀死了分词技术！！！几个小时前，他们发布了 “Byte Latent Transformer”。一种无分词器架构，可将 Byte 动态编码为 Patch 并实现更好的推理...</li><li><a href="https://x.com/MarkSchmidty/status/1857522783720272304?t=Z7z5ArMVl8JCptgCP6iEjQ&s=19">来自 Mark Schmidt 🌐 (@MarkSchmidty) 的推文</a>：字节级模型的训练效率与 BPE 模型一样高，然而最大的字节级 LLM 仅有微小的 350M 参数，且训练数据集小得令人失望。我们到底什么时候才能彻底抛弃分词...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1317568067357638819)** (18 条消息🔥): 

> `GPU 上的解压缩, 物理学对 AI 的历史影响, Trellis 编码及其应用, 模型压缩与冗余, 分布式训练方法` 


- **GPU 上的高效解压缩实现**：论文讨论了一种在 GPU 上高效实现 **decompression**（解压缩）的方法，并提到尽管代码可读性较差，但其逻辑简单。该核心思想此前也曾发表过，体现了研究界良好的引用规范。
   - 成员们指出，虽然该方法在量化后非常有效，但对于训练（training）来说仍然太慢。
- **物理学驱动了大多数 AI 技术**：对话指出，许多 AI 技术都起源于物理学，并强调任何可行的方法很可能此前已被物理学家探索过。**Communication theory**（通信理论）的思想对于 **LLMs** 尤为重要，展示了学科间历史性的交织。
   - 一位成员评论了这种知识谱系，认为 AI 的进步往往可以追溯到物理科学。
- **Trellis 编码：历史视角**：一位成员分享了 **Trellis coding** 的历史，提到其发明者等待了六年才将其公开，随后它成为了官方标准的一部分。这个历史轶事突显了技术思想缓慢但具有影响力的演进过程。
   - 有建议认为，此类技术可以优化分布式训练场景中的梯度传输，解决编码和优化中的复杂性。
- **模型压缩中的权衡**：关于在更新 **Loras** 时保持模型完整性的讨论揭示了一些策略，即通过牺牲训练时间来换取内存效率，并建议定期重新初始化 **Loras**。这种方法更像是顺序重训练（sequential retraining），而非并行训练。
   - 一位成员对在预训练情况下使用固定 **Loras** 导致模型退化的问题表示担忧。
- **MLPs 中的冗余信息**：人们对 **MLPs** 中明显缺乏探索的 **position invariance**（位置不变性）产生了好奇，特别是在上下投影块（up-down projection blocks）中，权重的顺序可以在不影响性能的情况下改变。这种潜在的信息冗余可能预示着简化的机会。
   - 对话表明，在该领域的进一步探索可能会为模型压缩策略提供新的见解。



**提到的链接**：<a href="https://x.com/OpenlifesciAI/status/1867999825721242101>">来自 Open Life Science AI (@OpenlifesciAI) 的推文</a>：🌟 每周医疗 AI 研究综述 🌟📅 2024年12月7-14日。这是您每周最重要的医疗 AI 论文摘要！🎉🤖 Medical LLM & Other Models - PediaBench: Chinese Pediatric LLM-...

  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1317951045011374081)** (2 条消息): 

> `SF Compute 上线, Qwen QwQ 降价, 来自 xAI 的新 Grok 模型` 


- **SF Compute 加入 OpenRouter**：OpenRouter 宣布了新的供应商：**SF Compute**，增强了其服务能力。
   - 此次加入旨在为寻求多样化服务集成的用户提供更多选择。
- **Qwen QwQ 大幅降价**：**Qwen QwQ** 经历了高达 **55% 的降价**，吸引了更多用户使用其功能。
   - 详情可见其 [价格页面](https://openrouter.ai/qwen/qwq-32b-preview)。
- **新 Grok 模型的流量增加**：来自 **xAI** 的两款新 **Grok 模型** 于周末发布，导致其平台流量增加。
   - 鼓励用户在 [OpenRouter 的 xAI 页面](https://openrouter.ai/x-ai) 探索所有模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1868534692183507098">来自 OpenRouter (@OpenRouterAI) 的推文</a>：来自 @xai 的两款新 @Grok 模型已于本周末推出 - 已经看到流量转移。在这里查看所有模型！https://openrouter.ai/x-ai</li><li><a href="https://openrouter.ai/qwen/qwq-32b-preview">QwQ 32B Preview - API, 供应商, 统计数据</a>：QwQ-32B-Preview 是由 Qwen 团队开发的专注于 AI 推理能力的实验性研究模型。作为预览版，它展示了极具前景的分析能力，同时也存在一些...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1317903486678995065)** (5 条消息): 

> `OpenRouter API wrapper, OpenRouter-client` 


- **OpenRouter API Wrapper 发布**：一名成员分享了 OpenRouter API wrapper 的发布公告，名为 [openrouter-client](https://www.npmjs.com/package/openrouter-client)，该工具于两天前发布。
   - 该 wrapper 简化了与 OpenRouter 的交互，并提供了用于实现和配置的示例代码。
- **社区对 API Wrapper 的热烈反响**：一名成员对新的 API wrapper 表示了极大的热情，在回复公告时称：*That's awesome!*
   - 开发者以简单的 *Thank you!* 回应了这份热情。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://coauthor.studio/rewind">2024 LinkedIn Rewind | Your Year in Review</a>：在几分钟内创建个性化的 2024 年 LinkedIn 年度回顾。这是为专业人士提供的免费工具，旨在以真实的口吻展示成就和见解。无需登录。</li><li><a href="https://www.npmjs.com/package/openrouter-client">openrouter-client</a>：OpenRouter 的 API wrapper。最新版本：1.1.0，最后发布于：2 天前。通过运行 `npm i openrouter-client` 开始在你的项目中使用 openrouter-client。目前没有其他项目在...
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1317226679294361600)** (372 messages🔥🔥): 

> `Hermes 3 405B performance, Gemini Pro 2 capabilities, Image generation model updates, Prompt caching in LLM providers, Rate limits for Gemini models` 


- **Hermes 3 405B 展现出强大能力**：用户报告称 Hermes 3 405B 在创意任务中表现出色，有人认为其质量可与 Claude 2.0 媲美。
   - 然而，也有讨论指出其在代码任务中的性能相较于其他模型较慢。
- **Gemini Pro 2 的受欢迎程度日益增长**：Gemini Pro 2 (1206) 被强调为代码任务中 Sonnet 3.5 等模型的有力竞争替代方案。
   - 一些用户注意到它在生成代码和处理科学问题方面比 Flash 更有效。
- **Google 的图像生成模型更新**：Google 宣布了其图像生成模型的新版本，包括 Imagen 3 和一个名为 Whisk 的新模型。
   - 这些更新表明 AI 在视觉内容生成能力方面正在进一步提升。
- **提供商的 Prompt caching 功能**：讨论中提到了某些提供商针对开源模型缺失 Prompt caching 功能的问题。
   - 一些用户推论了 Caching 在 LLM 应用中可能带来的成本节约和效率提升。
- **Gemini 模型的 Rate limits**：用户对不同 Gemini 模型的 Rate limits 表示担忧，特别是在 Google Cloud Platform 下。
   - 据观察，实验模型和生产模型之间的 Rate limits 存在显著差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/docs/quick-start">快速入门 | OpenRouter</a>：开始使用 OpenRouter 进行构建</li><li><a href="https://www.bbc.com/news/articles/cd0el3r2nlko">Suchir Balaji：OpenAI 举报人被发现死于公寓内</a>：旧金山法医办公室判定 Suchir Balaji 死于自杀，警方未发现他杀证据。</li><li><a href="https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/creating-and-highlighting-code-blocks">创建并高亮代码块 - GitHub Docs</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/limits">限制 | OpenRouter</a>：设置模型使用限制</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1hf0nmm/chatgpt_is_a_fantastic_fiction_editoras_long_as/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/parameters">参数 | OpenRouter</a>：配置请求参数</li><li><a href="https://github.com/billmei/every-chatgpt-gui/blob/main/README.md">every-chatgpt-gui/README.md at main · billmei/every-chatgpt-gui</a>：适用于 ChatGPT、Claude 和其他 LLM 的所有前端 GUI 客户端 - billmei/every-chatgpt-gui</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/15lihmq/big_model_comparisontest_13_models_tested/">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://blog.google/technology/google-labs/video-image-generation-update-december-2024/">使用 Veo 2 和 Imagen 3 进行最先进的视频和图像生成</a>：我们正在推出全新的最先进视频模型 Veo 2，以及 Imagen 3 的更新。此外，请关注我们的新实验项目 Whisk。</li><li><a href="https://openrouter.ai/docs/provider-routing">提供商路由 | OpenRouter</a>：在多个提供商之间路由请求</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>：OpenRouter 事件历史记录
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1317263148541018122)** (3 messages): 

> `OpenRouter launch, New feature integration` 


- **OpenRouter 功能上线！**：@alexatallah 宣布新功能现已对所有人开放 🙂，并表示很快会发布正式公告。
   - *敬请期待更多详情！*
- **用户询问功能使用说明**：一位用户询问 *如何使用此功能？*，希望明确新功能的用法。
   - 另一位用户回复称，只需前往 [OpenRouter Settings Integrations](https://openrouter.ai/settings/integrations) 并在那里添加你的 Key 即可！


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1317270737861480449)** (70 messages🔥🔥): 

> `学生项目评分标准、非 Transformer 模型研究、字节 vs 比特编码、模型训练数据打乱、JAX/Flax vs TensorFlow` 


- **创建学生项目评分标准**：一名成员建议将评分标准作为学生生成图像 Token 任务的一部分，并提供了一些幽默的评分示例代码。
   - 讨论包括使用 Perplexity 和分类器对提交作品进行评分的方案，并建议生成一些刻意难以作弊的示例。
- **非 Transformer 模型的活跃研究**：成员们讨论了非 Transformer 架构的持续研究，提到了 Numenta 和 AI2 lab 等实验室为其模型发布了多个 Checkpoint。
   - 大家对小型实验室推动新颖的非 Transformer 研究而非主流 Transformer 模型表示好奇。
- **辩论字节 vs 比特编码**：对话涵盖了字节编码（Byte Encoding）的相关性，并区分了在何种情况下它可能会比 BPE 等 Token 化结构丢失更多信息。
   - 成员们表示，虽然字节级处理可以更准确地表示文本，但与现有的 Tokenization 方法相比，可能不会产生显著优势。
- **解决由于后期训练导致的模型偏差**：针对模型对近期引入的训练数据产生偏差（Bias）的问题提出了担忧，并建议通过打乱（Shuffling）数据来减轻这些影响。
   - 一位成员分享了使用数据同质化策略来提高模型训练公平性的经验。
- **从 TensorFlow 切换到 JAX/Flax**：对话强调了对 TensorFlow 支持下降的沮丧，促使成员考虑切换到 JAX/Flax 以获得更好的性能。
   - 对 JAX/Flax 的情绪压倒性地积极，许多人认为它是未来更稳健的选择。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.liquid.ai/">Liquid AI: 构建各种规模的高性能且高效的通用 AI 系统。</a>：我们构建各种规模的高性能且高效的通用 AI 系统。Liquid Foundation Models (LFMs) 是新一代生成式 AI 模型，在各方面都达到了最先进的性能...</li><li><a href="https://forms.gle/JcYAJEukfBiYVxTW8">Airpods and battery</a>：你会购买一个让你在听音乐时能给充电宝充电的设备吗？</li><li><a href="https://files.vermeille.fr/cparti.html">Instructions</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1317223024188981348)** (249 messages🔥🔥): 

> `Attention vs 核方法 (Kernel Methods)、约束满足问题 (Constraint Satisfaction Problems)、强化学习 (Reinforcement Learning) 与记忆、神经网络中的迭代推理、Transformer 中的混合架构`

- **Attention 与 Kernel Methods**：讨论围绕将 Attention 框架化为一种 Kernel Method 展开，成员们指出这并不完全准确，特别是在评估 softmax 等 self-attention 操作的功能时。成员们辩论了 Attention 机制是否比 Kernel 方法更充分地发挥了其潜力，并引发了关于底层数学差异的讨论。
   - Kernel Methods 与 Attention 之间的关系被描述为一种层级结构，表明虽然 Attention 可以通过 Kernel Methods 进行近似，但这种简化无法捕捉到 Attention 运行背景的复杂性。
- **模型中隐式约束的学习**：讨论强调了对模型是否能学习解决约束满足问题的兴趣，特别是以 Sudoku 为测试案例。探索了在小数据集上训练模型以确保解满足学习到的隐式约束的可行性。
   - 成员们建议，通过操纵数据表示可以观察到性能变化，甚至提出了在训练期间管理架构诱导偏差（architecturally induced biases）的想法。
- **通过能量扩散进行迭代推理**：介绍了用于通过能量扩散学习推理的 IRED 框架，旨在通过更好地组织输入和输出之间的约束来解决更复杂的问题。实验结果表明，与传统方法相比，在需要更复杂推理的任务上性能有所提高。
   - 讨论指出，该研究重点关注受限优化问题，以及该方法如何为神经网络从结构化数据中隐式学习推理提供不同的视角。
- **混合架构与性能**：集成 Attention 和 RNN 特性（如 Gated DeltaNet 和 Samba）的各种混合方法的架构性能是焦点。成员们辩论了不同的设置及其对训练效率、泛化能力和潜在性能提升的影响。
   - 针对测试 CoHere 架构的修改以及在各种实验框架中评估不同 Attention 机制的效果提出了具体建议。
- **Transformers 中的 Meta Tokens**：成员们分享了对 Transformer 架构中 Meta Tokens 作用的见解，并讨论了更有效地处理上下文信息的意义。对话围绕增强 Transformer 的记忆能力如何提升其表示和处理功能展开。
   - 参与者对 Meta Tokens 的实用性表达了不同的看法，呼吁在受控设置下对其影响进行进一步的实证检验。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2410.07041">Emergent properties with repeated examples</a>：我们研究了 Transformer 的性能与使用算法生成数据集的训练样本重复次数之间的函数关系。在三个数学问题上：最大公约数...</li><li><a href="https://arxiv.org/abs/2412.07684v1">The Pitfalls of Memorization: When Memorization Hurts Generalization</a>：记忆的陷阱：当记忆损害泛化时。神经网络通常学习符合大多数数据的简单解释，同时记住偏离这些解释的例外情况。这种行为导致在...时泛化能力较差。</li><li><a href="https://arxiv.org/abs/2406.07522v1">Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling</a>：Samba：用于高效无限上下文语言建模的简单混合状态空间模型。高效建模具有无限上下文长度的序列一直是一个长期存在的问题。过去的工作要么面临二次计算复杂度，要么在...上的外推能力有限。</li><li><a href="https://arxiv.org/abs/2406.11179">Learning Iterative Reasoning through Energy Diffusion</a>：通过能量扩散学习迭代推理。我们引入了通过能量扩散学习迭代推理（IRED），这是一个通过使用基于能量的公式化推理和决策问题，为各种任务学习推理的新框架...</li><li><a href="https://mingukkang.github.io/GigaGAN/">GigaGAN: Scaling up GANs for Text-to-Image Synthesis</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2411.13504">Disentangling Memory and Reasoning Ability in Large Language Models</a>：解耦大型语言模型中的记忆与推理能力。大型语言模型（LLMs）在处理需要广泛知识和推理能力的复杂任务方面表现出强大的性能。然而，现有的 LLM 推理流水线...</li><li><a href="https://arxiv.org/abs/2405.13956">Attention as an RNN</a>：将 Attention 视为 RNN。Transformers 的出现标志着序列建模的重大突破，提供了一种能够利用 GPU 并行性的高性能架构。然而，Transformers 在计算上...</li><li><a href="https://arxiv.org/abs/2411.13676">Hymba: A Hybrid-head Architecture for Small Language Models</a>：Hymba：一种用于小型语言模型的混合头架构。我们提出了 Hymba，这是一个小型语言模型系列，采用混合头并行架构，将 Transformer 的 Attention 机制与状态空间模型（SSMs）相结合，以提高效率...</li><li><a href="https://arxiv.org/abs/2306.00946">Exposing Attention Glitches with Flip-Flop Language Modeling</a>：通过 Flip-Flop 语言建模揭示 Attention 缺陷。为什么大型语言模型有时会输出事实错误并表现出错误的推理？这些模型的脆弱性，特别是在执行长链推理时，目前看来...</li><li><a href="https://arxiv.org/abs/2006.11527">Memory Transformer</a>：Memory Transformer。基于 Transformer 的模型在许多自然语言处理任务中取得了最先进的结果。Self-attention 架构允许 Transformer 结合来自所有元素的信息...</li><li><a href="https://neel04.github.io/my-website/blog/pytorch_rant/">PyTorch is dead. Long live JAX. | Neel Gupta</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2410.01201v1">Were RNNs All We Needed?</a>：RNN 才是我们真正需要的吗？Transformers 在序列长度方面的可扩展性限制，重新引发了人们对训练期间可并行的循环序列模型的兴趣。因此，许多新型循环架构...</li><li><a href="https://arxiv.org/abs/2412.06464">Gated Delta Networks: Improving Mamba2 with Delta Rule</a>：门控 Delta 网络：利用 Delta 规则改进 Mamba2。线性 Transformers 作为标准 Transformers 的高效替代方案受到了关注，但它们在检索和长上下文任务中的性能一直有限。为了解决这些局限性...</li><li><a href="https://x.com/SkyLi0n/status/1867324080262885800">Tweet from Aaron Gokaslan (@SkyLi0n)</a>：GAN 在 2024 年能超越扩散模型吗？是的！#2200：GAN 已死，GAN 万岁！一个现代 GAN 基准！如果你参加 NeurIPS 2024，请加入我。周四晚上的海报环节...</li><li><a href="https://arxiv.org/abs/2407.01178">$\text{Memory}^3$: Language Modeling with Explicit Memory</a>：$\text{Memory}^3$：具有显式记忆的语言建模。大型语言模型（LLMs）的训练和推理共同构成了一个昂贵的过程，将知识从原始数据传输到有意义的计算中。受人类记忆层级的启发...</li><li><a href="https://github.com/lucidrains/x-transformers?tab=readme-ov-file#memory-transformers">GitHub - lucidrains/x-transformers: A concise but complete full-attention transformer with a set of promising experimental features from various papers</a>：GitHub - lucidrains/x-transformers：一个简洁但完整的全注意力 Transformer，包含来自各种论文的一系列有前景的实验性功能 - lucidrains/x-transformers
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1317227922016309359)** (8 条消息🔥): 

> `针对 Transformers 的 RASP 框架、SAE 操控应用、MCMC 中的对比目标、SAE 研究中的负面结果、稠密探针与 SAE 编码` 


- **RASP 为 Transformers 引入了新的编程模型**：名为 [RASP](https://arxiv.org/abs/2106.06981) 的论文提出了一种针对 **Transformer-Encoders** 的计算模型，使用编程语言来映射注意力（attention）和前馈计算（feed-forward computation）等基础组件。
   - 它展示了如何训练 Transformers 来模拟 RASP 解决方案，以处理 **直方图（histograms）** 和 **排序（sorting）** 等任务。
- **Sieve 展示了有效的 SAE 操控（Steering）**：在 **Sieve** 流水线中实现 **基于 SAE 的干预（interventions）** 引起了广泛关注，该方法在 Python 函数的模糊测试（fuzz testing）中以极小的代价提升了性能。
   - 这种方法实现了 **条件特征操控（conditional feature steering）**，在保持性能的同时，能精确地防止如正则表达式（regex）使用等不当行为。
- **对 MCMC 中对比目标（Contrastive Objectives）的兴趣**：一位成员询问是否有研究在 **MCMC 框架** 内探索 **对比目标**，特别是与大语言模型相关的研究。
   - 这标志着人们对于将这些方法论整合以理解自然语言分布的兴趣日益浓厚。
- **对 SAE 操控结果的怀疑**：尽管 SAE 操控的应用前景广阔，但正如 [最近的研究](https://arxiv.org/abs/2411.11296) 所指出的，它似乎会损害整体性能。
   - 成员们对如何在不牺牲性能的情况下识别有效的操控机制表示担忧，特别是在涉及 **拒绝行为（refusal behavior）** 方面。
- **SAE 的积极探针（Probing）结果**：研究强调了 **在 SAE 编码上训练的稠密探针（dense probes）** 的功效，并突出了它们在低数据量和受损数据集中的优势。
   - 虽然 SAE 探针显示出具有竞争力的结果，但在某些设置下与激活探针（activation probes）相比存在无效发现，这引发了关于两种方法可靠性的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2411.11296">Steering Language Model Refusal with Sparse Autoencoders</a>：部署语言模型的负责任做法包括引导模型识别并拒绝回答被认为不安全的提示，同时遵守安全提示。实现这种行为...</li><li><a href="https://arxiv.org/abs/2106.06981">Thinking Like Transformers</a>：Transformer 背后的计算模型是什么？循环神经网络在有限状态机中有直接的对应关系，允许对架构变体进行清晰的讨论和思考...</li><li><a href="https://www.tilderesearch.com/blog/sieve">Sieve: SAEs Beat Baselines on a Real-World Task (A Code Generation Case Study) | Tilde</a>：未找到描述</li><li><a href="https://www.lesswrong.com/posts/NMLq8yoTecAF44KX9/sae-probing-what-is-it-good-for-absolutely-something">SAE Probing: What is it good for? Absolutely something! — LessWrong</a>：Subhash 和 Josh 是共同第一作者。这项工作是 Neel Nanda 的 MATS 课程中为期两周的研究冲刺的一部分……
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1318137501319172188)** (12 条消息🔥): 

> `lm_eval harness 与 VLLM，VLLM API 错误问题，VLLM 版本讨论` 


- **lm_eval harness 成功与 VLLM 配合使用**：一位用户分享了让 **lm_eval harness** 与 **VLLM** 协同工作的有效方法，并指出了特定的安装命令。
   - *此过程包括安装 0.6.3 版本的 VLLM，以防止评估 harness 出现问题。*
- **出现 VLLM API 错误**：成员们讨论了 VLLM 产生的错误，暗示 **lm_eval 使用的内部 API** 可能已经发生了变化。
   - *另一位成员暗示这可能与 VLLM 的某个特定 commit 有关。*
- **版本混淆引发疑问**：关于是否在 **VLLM 0.6.4 版本** 中遇到了这些错误的询问，并提到了可能存在的 ARM 特定问题。
   - *成员们澄清了版本细节，并指出缩写词的混淆引发了一些笑声。*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/menhguin/minp_paper/blob/main/%5BPUBLIC%5D_Min_P_Evals_Replication_for_GPQA_and_GSM8K_COT.ipynb">minp_paper/[PUBLIC]_Min_P_Evals_Replication_for_GPQA_and_GSM8K_COT.ipynb at main · menhguin/minp_paper</a>: Min P 论文的代码实现、评估、文档、链接和资源 - menhguin/minp_paper</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---

### **Bolt.new / Stackblitz ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1317221851876818944)** (39 条消息🔥): 

> `Bolt Token 消耗、货币更新问题、Bug 报告、使用 Bolt 进行项目管理、Stripe 与 Supabase 集成` 


- **Bolt 在未产生更改的情况下过度消耗 Token**：多位成员报告称，Bolt 在 UI 未反映任何更改的情况下消耗了大量 Token，其中一位用户指出他们已经消耗了超过 **500 万个 Token** 但未获成功。
   - 他们怀疑存在系统性 Bug，并在 [GitHub](https://github.com/stackblitz/bolt.new/issues/4218) 上记录了与此问题相关的 Issue。
- **货币更新困难**：一位用户表达了挫败感，尽管多次尝试使用特定的 Prompt，但仍无法将货币显示从 **$ USD** 更改为 **INR**。
   - 他们注意到，即使锁定了 `.env` 文件，它仍然被修改了，这表明 Bolt 在处理锁定文件的方式上可能存在 Bug。
- **UI 与 Bug 的共同体验**：几位用户对 Bolt 表达了类似的经历，表明这不仅仅是浏览器问题，并对更新无法同步到 Front-End 表示担忧。
   - 一位用户提到，他们正尝试通过将项目从 StackBlitz Fork 到 GitHub，然后在 Replit 上运行来解决此问题。
- **有效的 Prompting 策略**：成员们分享了有效引导 Bolt 的策略，包括用于项目规划的 Meta Prompt，该 Prompt 概述了正确执行的步骤。
   - 一位用户打算创建一个特定解决方案的 UI 版本，并提供在各种 LLM 之间选择生成模型的选项。
- **社区支持与资源**：用户为故障排除提供了建议，例如手动识别需要修改的代码段，而不是完全依赖 Bolt。
   - 一位用户鼓励社区通过分享截图和寻求帮助来进行协作，强调了在项目开发中坚持不懈的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/excellent-bill-and-ted-air-guitar-yes-yeah-gif-15828050">Excellent Bill And Ted GIF - Excellent Bill And Ted Air Guitar - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gist.github.com/martinbowling/fe4aa7711d023ef7f188fdd9828fad3e">此 Meta Prompt 概述了 Bolt 创建详细软件项目计划的系统方法。它包括分析需求、定义结构、设计 UI、规划实施，并映射所选技术栈如何融入开发过程。</a>：此 Meta Prompt 概述了 Bolt 创建详细软件项目计划的系统方法。它包括分析需求、定义结构、设计 UI、规划实施...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4218">Bolt 无法纠正问题并正在消耗 Token · Issue #4218 · stackblitz/bolt.new</a>：描述 Bug。无法解决以下 TypeScript 错误：ReferenceError: typescript is not defined at /src/components/Tasks/TaskUploader.tsx:18:1。系统已尝试修复，例如...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4229">预览错误 · Issue #4229 · stackblitz/bolt.new</a>：描述 Bug。我尝试查看新组件的新 Prompt，但它丢失了且未在预览中显示。导致错误的 Bolt URL 链接：https://bolt.new/~/sb1-fh4oeef6。重现步骤...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/4233">Bolt 的更改未同步到 Front-End · Issue #4233 · stackblitz/bolt.new</a>：描述 Bug。我正在开发一个项目，经过多次尝试，Bolt 对 App 代码库的任何更改都没有在 Front-End/UI 上更新。我看到其他用户可能也遇到了...
</li>
</ul>

</div>
  

---

### **Bolt.new / Stackblitz ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1317220753208250399)** (237 条消息🔥🔥): 

> `服务可用性问题、新功能与集成、Token 与订阅成本、React Native 开发指导、备份与恢复选项` 


- **服务可用性问题**：用户报告频繁出现“Service Unavailable”消息，导致对 Bolt.new 的 Token 管理和功能的担忧。
   - 一个反复出现的主题是，在遇到这些问题时，对进度和数据丢失感到沮丧。
- **新功能与集成**：讨论围绕着备受期待的 Supabase 集成展开，许多用户渴望更新并对新功能表示兴奋。
   - 分享了一个早期 Supabase 集成的视频演示，展示了其功能。
- **Token 与订阅成本**：用户对 Token 消耗过快表示担忧，特别是充值与月度计划的对比，用户寻求关于 Token 管理机制的澄清。
   - 用户强调需要累积 Token 系统，并对当前的过期规则表示不满。
- **React Native 开发指导**：建议性讨论集中在将 Web 应用程序转换为移动平台的最佳实践，特别是使用 React Native 和 Expo。
   - 建议将移动应用程序的开发转移到 Cursor，因为它对这些功能有更好的支持。
- **备份与恢复选项**：一位用户不小心删除了一个项目并难以恢复，引发了关于备份功能和潜在恢复方法的讨论。
   - 已确认活动项目可以进行备份，但已删除的项目可能无法恢复。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://suppabolt.netlify.app/">SUPABOLT</a>：未找到描述</li><li><a href="https://tenor.com/view/noice-nice-click-gif-8843762">Noice Nice GIF - Noice Nice Click - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://x.com/morganlinton/status/1868388127347523794?s=46">来自 Morgan Linton ᯅ (@morganlinton) 的推文</a>：昨天，@stackblitz 的出色团队让我进入了 Bolt x @supabase 集成的 Beta 测试。我使用它还不到 24 小时，就已经被震撼了。不得不重新...</li><li><a href="https://www.youtube.com/watch?v=IIueA5giF_4"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=5SI9lqHh0ZU&t=2052s"> - YouTube</a>：未找到描述</li><li><a href="https://x.com/mikeysee/status/1849331209026900396)">来自 Michael Cann (@mikeysee) 的推文</a>：不得不说，我对 @stackblitz 的 http://bolt.new 印象非常深刻！只需几个简短的 Prompt，我就能重新创建我的 StashIt 项目 (https://mikecann.blog/posts/introducing-stashit)。将会使用...</li><li><a href="http://bolt.new">bolt.new</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1317224726623551569)** (68 条消息🔥🔥): 

> `Grok-2 更新、NeurIPS 2024、Veo 2 和 Imagen 3 发布、Byte Latent Transformer、语音模式下的搜索` 


- **Grok-2 模型改进发布**：Grok-2 已更新，速度提升了三倍，准确性和多语言能力也有所提高，目前正在 X 上免费推出。
   - 它提供网页搜索、引用以及名为 Aurora 的新图像生成器，增强了用户交互。
- **来自 Ilya Sutskever 在 NeurIPS 2024 演讲的见解**：Ilya 在演讲中强调了 LLM 在预训练阶段的扩展瓶颈（plateau），以及未来向 Agent 行为和 LLM 之上的工具转变的趋势。
   - 对话包括对数据饱和以及未开发视频内容用于 AI 训练潜力的各种看法。
- **Google 发布 Veo 2 和 Imagen 3**：Google 推出了 Veo 2 和 Imagen 3，分别具有改进的高质量视频生成和更好的图像构图功能，可在 VideoFX 和 ImageFX 中使用。
   - 这些更新在理解电影摄影和生成内容中的多样艺术风格方面提供了增强的能力。
- **Byte Latent Transformer 彻底改变了 Tokenization**：META 发布了 Byte Latent Transformer (BLT)，这是一种无 Tokenizer 的架构，可将字节动态编码为 Patch，从而提高推理效率。
   - 据报告，BLT 模型在显著降低推理 FLOPs 的情况下，能够匹配或超越 Llama 3 等现有模型。
- **搜索功能扩展至语音模式**：OpenAI 宣布在 ChatGPT 的高级语音模式（Advanced Voice mode）中推出搜索功能，允许用户通过语音交互获取实时信息。
   - 这一功能反映了 OpenAI 搜索团队与多模态产品研究团队之间的丰硕合作。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">扩展推理时计算 (Scaling test-time compute) - 一个由 HuggingFaceH4 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/nyaathea/status/1867854474808811570?s=46">来自 Thea (@nyaathea) 的推文</a>：笑死 https://x.com/i/grok/share/ieeeD20tYc40Ayi0dmFp4hrgh</li><li><a href="https://x.com/raizamrtn/status/1867596346783601005?s=46">来自 Raiza Martin (@raizamrtn) 的推文</a>：我非常激动看到 Studio 终于发布了！🎨 这是我们近两年前梦想的最后一部分：一个强大的界面，可以接收所有对你重要的输入，一个强大的 AI 优先编辑器...</li><li><a href="https://x.com/main_horse/status/1867795766389174590?s=46">来自 main (@main_horse) 的推文</a>：关于 Byte Latent Transformer (BLT) 的想法</li><li><a href="https://x.com/scaling01/status/1867573707247346003?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Lisan al Gaib (@scaling01) 的推文</a>：META 刚刚杀死了分词 (Tokenization) !!! 几小时前他们发布了 "Byte Latent Transformer"。一种无分词器 (Tokenizer-free) 架构，可动态地将 Bytes 编码为 Patches，并实现了更好的推理...</li><li><a href="https://x.com/shuchaobi/status/1868729224275935543?s=46">来自 Shuchao Bi (@shuchaobi) 的推文</a>：今天，我们正在高级语音模式 (Advanced Voice mode) 中推出搜索功能。现在你可以在与 ChatGPT 交谈时获取实时信息。这是搜索团队和多模态团队之间一次非常富有成效的合作...</li><li><a href="https://x.com/adonis_singh/status/1868125576076357746?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 adi (@adonis_singh) 的推文</a>：o1 pro 刚刚跟我分享了它的思维链 (Chain of Thought)...</li><li><a href="https://x.com/notebooklm/status/1867595259678503179?s=46">来自 notebooklm (@notebooklm) 的推文</a>：📢 新品发布 📢 1. ✋ 推出：“加入”音频概览 + 直接与 AI 主持人互动 2. 😎 针对基于你的来源管理和生成新内容而优化的新 UI 3. 💪 NotebookLM Plus:...</li><li><a href="https://x.com/kalomaze/status/1868015615723917624?s=46">来自 kalomaze (@kalomaze) 的推文</a>：我们甚至还没有用完人类编写的文本，我们只是饱和了那些“作为文本”发布的内容。至少还有约 5000 亿 Token 价值的 “YouTube 视频文章”尚未被...</li><li><a href="https://x.com/nyaathea/status/1868117356570108184?s=46">来自 Thea (@nyaathea) 的推文</a>：你真的可以通过在你的显示名称中放入指令来对 Grok 进行提示词注入 (Prompt Inject)，这太搞笑了</li><li><a href="https://qwenlm.github.io/blog/qwen2.5-turbo/">将上下文长度扩展到 1M Tokens！</a>：API 文档（中文） HuggingFace 演示 ModelScope 演示。简介：在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求。在最近几个月里，我们...</li><li><a href="https://x.com/pika_labs/status/1867641187898995179">来自 Pika (@pika_labs) 的推文</a>：我们送给你的节日礼物：Pika 2.0 来了。不仅面向专业人士，也面向普通大众。（甚至是欧洲人！）现在可在 http://pika.art 使用</li><li><a href="https://x.com/googledeepmind/status/1868703624714395907?s=46">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：今天，我们宣布推出 Veo 2：我们最先进的视频生成模型，它可以根据文本或图像提示生成逼真、高质量的剪辑。🎥 我们还发布了改进版的...</li><li><a href="https://x.com/ilanbigio/status/1867674451946418537?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 ilan bigio (@ilanbigio) 的推文</a>：在与数百家公司设计和部署 AI 解决方案后，我们想分享我们的秘密。所有的秘密。宣布 @openai 构建时间展示，了解关于 Agent, Eval, Realtime, Distillation, o...</li><li><a href="https://x.com/vincentweisser/status/1867719020444889118">来自 Vincent Weisser (@vincentweisser) 的推文</a>：.@ilyasut 在 NeurIPS 2024 上的完整演讲 “我们所知的预训练 (Pre-training) 将会结束”，接踵而至的是超级智能：具备 Agent 能力、推理、理解且具有自我意识</li><li><a href="https://x.ai/blog/grok-1212">让每个人都能使用 Grok</a>：Grok 现在更快、更敏锐，并改进了多语言支持。𝕏 平台上的每个人都可以使用它。</li><li><a href="https://x.com/scaling01/status/1867990298002956433">来自 Lisan al Gaib (@scaling01) 的推文</a>：天哪，居然有一个基准测试显示 o1-preview 排名第一，Sonnet 3.5 v2 排名第二。我觉得 LiveBench 的语言类别比其他基准测试更能反映模型能力...</li><li><a href="https://x.com/openai/status/1868715324885156177?s=46">来自 OpenAI (@OpenAI) 的推文</a>：第 8 天：ChatGPT Search 之日 https://openai.com/12-days/?day=8</li><li><a href="https://x.com/skcd42/status/1867561917159755942">来自 skcd (@skcd42) 的推文</a>：CodeStory Agent 现在在 SWE-bench-verified 上达到了 SOTA，解决率为 62.2%。我们通过在推理时扩展我们的 Agent 并重新学习“惨痛的教训” (Bitter Lesson) 实现了这一点。Sonnet 3.5 (new) 是我们唯一使用的 LLM...</li>

><li><a href="https://x.com/lmarena_ai/status/1867661674356023653?t=_5a4HGyVdOMlvwsk8a6Bbg&s=19">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：WebDev Arena 排行榜现已上线，拥有超过 10K+ 投票！#1. Claude 3.5 Sonnet #2. Gemini-Exp-1206 #3. Gemini-2.0-Flash #4. GPT-4o-2024-11-20 #5. Qwen2.5-Coder-32B #6. Gemini-1.5-Pro-002 恭喜 @AnthropicAI...</li><li><a href="https://blog.google/technology/google-labs/video-image-generation-update-december-2024/">使用 Veo 2 和 Imagen 3 实现最先进的视频和图像生成</a>：我们正在推出全新的、最先进的视频模型 Veo 2，以及 Imagen 3 的更新。此外，请关注我们的新实验项目 Whisk。</li><li><a href="https://x.com/Dorialexander/status/1867665269058842885">来自 Alexander Doria (@Dorialexander) 的推文</a>：那么 Patches 的扩展性是否优于 Tokens？Tokenizer 已经过时了吗？我很少发论文解读推文，但这篇 Meta 的论文足够引人入胜。</li><li><a href="https://x.com/OpenAI/status/1867675796950987146">来自 OpenAI (@OpenAI) 的推文</a>：推出 Projects——一种在 4o 中组织共享主题或上下文的对话的简便方法。现在已向全球 ChatGPT Plus、Pro 和 Team 用户开放。我们将在 1 月份将其推向 Enterprise 和 Edu 用户...</li><li><a href="https://x.com/nrehiew_/status/1868360942846963977?s=46">来自 wh (@nrehiew_) 的推文</a>：我并不是说这就是 TikTok 的算法。我只是说这是来自字节跳动（TikTok 的母公司）的一篇 2022 年推荐系统论文。</li><li><a href="https://x.com/scaling01/status/1867713546848813428?s=46">来自 Lisan al Gaib (@scaling01) 的推文</a>：Ilya 的气场（Aura）太强了。AI 领域无人能及。这位 GOAT 说出像“数据是 AI 的化石燃料”这样的话，每个人都会瞬间认同。</li><li><a href="https://x.com/angrytomtweets/status/1867929988617380350?s=46">来自 Angry Tom (@AngryTomtweets) 的推文</a>：AI 已经失控了！Pika 刚刚发布了 2.0，人们都为之疯狂。只需上传一张照片，Pika 就会将其与场景中的其他元素（Ingredients）结合。10 个疯狂的案例：</li><li><a href="https://x.com/johnrushx/status/1867723891688583356?s=46">来自 John Rush (@johnrushx) 的推文</a>：🚨 Ilya Sutskever 终于确认：> LLM 在预训练阶段的扩展（scaling）已进入平台期 > 算力在扩展，但数据没有，且新数据或合成数据（synthetic data）无法改变现状。接下来会发生什么 > 同样的...</li><li><a href="https://www.swebench.com/">SWE-bench</a>：未找到描述</li><li><a href="https://x.com/teortaxestex/status/1867820202366247387?s=46">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：是的，我们将扩展算力使用，而不是数据或模型。但如何实现？“Agents？”我觉得 Ilya 隐藏了关键信息（alpha），这是他对自己“扩展什么？”谜题的具体回答。我的猜测：每个 ha... 的算力</li><li><a href="https://www.youtube.com/live/FcB97h3vrzk?si=QoX_2KmEMYjw8FEJ"> - YouTube</a>：未找到描述</li><li><a href="https://youtu.be/1yvBqasHLZs?si=pQihchmQG3xoeCPZ">Ilya Sutskever：“神经网络的序列到序列学习：十年的回顾”</a>：Ilya Sutskever 在加拿大温哥华举行的 NeurIPS 2024 上的完整演讲“神经网络的序列到序列学习：十年的回顾”。“我们所熟知的预训练...”</li><li><a href="https://github.com/shun-liang/readable-talks-transcriptions/blob/main/neurips_2024/Vincent%20Weisser%20-%20.%40ilyasut%20full%20talk%20at%20neurips%202024%20pre-training%20as%20we%20know%20it%20will%20end%20and%20what%20comes%20next%20is%20superintelligence%20agentic%2C%20reasons%2C%20understands%20and%20is%20self%20aware.md">shun-liang/readable-talks-transcriptions 仓库中的 NeurIPS 2024/Vincent Weisser - Ilya Sutskever 在 NeurIPS 2024 的完整演讲：我们所熟知的预训练将结束，接踵而至的是具有 Agentic 特性、能推理、理解且具有自我意识的超级智能。</a>：可读的会议演讲转录。通过在 GitHub 上创建账号来为 shun-liang/readable-talks-transcriptions 的开发做出贡献。</li><li><a href="https://news.ycombinator.com/item?id=42415122">未找到标题</a>：未找到描述</li>
</ul>

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1317234497208713268)** (183 条消息🔥🔥): 

> `NeurIPS Webcrawl, Prompt Engineering, Marvin 的 AI Functions, SillyTavern, Entropix 和聊天机器人` 


- **关于 NeurIPS Webcrawl 的讨论**：成员们讨论了最近的 [NeurIPS Webcrawl](https://neurips.exa.ai) 及其影响，其中一位成员提到稍后会查看精华内容。
   - 一位用户对新提供的资源表示兴奋，并讨论了如何从中受益。
- **探索 Prompt Engineering 技巧**：讨论中涉及了 Prompt Engineering 的复杂性，成员们分享了诸如使用 Prompt 来优化 Prompt 的技巧。
   - 一位用户幽默地指出，这种递归思维挑战了 AI 的常规用法。
- **Marvin 的 AI Functions 介绍**：一位成员分享了 Marvin 新推出的 “AI functions” 的细节，该功能允许在不编写实际源代码的情况下将其集成到 Python 代码中，强调了其易用性。
   - 这一创新使用户能够无缝执行情感分析和食谱生成等复杂任务。
- **用于 LLM 和 AI 测试的 SillyTavern**：SillyTavern 被介绍为 LLM 工程师测试各种模型和参数的实用工具，引起了用户的兴趣。
   - 社区讨论了其作为测试套件的用途及潜在应用，强调了 AI 交互的趣味性。
- **关于 Entropix 的见解**：讨论了利用基于熵的采样（entropy-based sampling）和并行解码的 Entropix，并将其与最近在 NeurIPS 上的演示联系起来。
   - 用户分享了与 Entropix 相关的 GitHub 资源，并考虑了其在 AI 开发中的应用。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://neurips.exa.ai">发现 NeurIPS 研究论文</a>：利用 AI 快速轻松地发现和搜索 NeurIPS 研究论文。</li><li><a href="https://www.askmarvin.ai/docs/text/functions/">AI functions - Marvin</a>：AI 工程工具包</li><li><a href="https://github.com/SillyTavern/SillyTavern">GitHub - SillyTavern/SillyTavern: 面向高级用户的 LLM 前端。</a>：面向高级用户的 LLM 前端。通过在 GitHub 上创建账号为 SillyTavern/SillyTavern 的开发做出贡献。</li><li><a href="https://youtu.be/4toIHSsZs1c?t=1608"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/SinatrasC/entropix-smollm/blob/main/smollm_entropix_torch.ipynb">entropix-smollm/smollm_entropix_torch.ipynb at main · SinatrasC/entropix-smollm</a>：在 PyTorch 上使用 Entropix 采样器的 smolLM。通过在 GitHub 上创建账号为 SinatrasC/entropix-smollm 的开发做出贡献。</li><li><a href="https://github.com/xjdr-alt/entropix">GitHub - xjdr-alt/entropix: 基于熵的采样和并行 CoT 解码</a>：基于熵的采样和并行 CoT 解码。通过在 GitHub 上创建账号为 xjdr-alt/entropix 的开发做出贡献。</li><li><a href="https://github.com/xjdr-alt/entropix/blob/main/evals/sampler/o1_chat_completion_sampler.py">entropix/evals/sampler/o1_chat_completion_sampler.py at main · xjdr-alt/entropix</a>：基于熵的采样和并行 CoT 解码。通过在 GitHub 上创建账号为 xjdr-alt/entropix 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1317226202360057947)** (147 条消息🔥🔥): 

> `Multimodal Models, Model Fine-tuning, Uncensored Chatbots, RAG Implementation, Model Updates` 


- **探索 Multimodal Models**：成员们讨论了结合多种模态（Text/Image/Audio/Video）模型的可用性，大多数解决方案是通过云服务找到的，而其他人则指出了 LM Studio 中的局限性。
   - 一次对话强调了本地设置中缺乏完全多模态的 LLM，引发了对即将推出的模型的兴趣。
- **Model Fine-tuning 的局限性**：用户询问了如何使用导出的数据对现有模型进行微调，特别是为了复制特定的语法或语气，但被告知 LM Studio 不支持 Fine-tuning。
   - 建议在聊天界面中利用 System Prompts 和示例文本进行临时调整。
- **Uncensored Chatbot 选项**：在寻找无审查聊天机器人选项时，成员们被引导使用较小的模型，如 **Gemma2 2B** 或 **Llama3.2 3B**，这些模型可以在 CPU 上运行。
   - Hugging Face 上分享了各种 Uncensored 模型，供本地环境参考。
- **RAG 实现和文档上传**：对话涉及了 LM Studio 内的 Retrieval-Augmented Generation (RAG) 功能和文档上传特性，以增强来自文档的上下文响应。
   - 用户了解到，虽然所有模型都可以执行 RAG，但实现 Web 访问或互联网集成需要通过 API 的自定义解决方案。
- **预期的软件更新**：参与者对 LM Studio 软件即将推出的更新表示好奇，同时在对政策和隐私的担忧中评估了 Jellybox 等替代方案。
   - 讨论强调了人们对更新或替代 AI 聊天解决方案的功能增强和用户体验的持续兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mustafaaljadery/gemma-2B-10M">mustafaaljadery/gemma-2B-10M · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/beta-releases">LM Studio Beta Releases</a>: LM Studio Beta 版本发布</li><li><a href="https://huggingface.co/bartowski/gemma-2-2b-it-abliterated-GGUF">bartowski/gemma-2-2b-it-abliterated-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF">bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/basics/rag">Chat with Documents - Running LLMs Locally | LM Studio Docs</a>: 如何将本地文档作为额外上下文提供给 LLM
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1317220491374759997)** (80 条消息🔥🔥): 

> `Power Supply Unit (PSU) Ratings, AMD Radeon VII GPU Support, Choosing GPU for AI/ML tasks, Llama Model Usage and Context Limits, Efficient Prompt Strategies` 


- **理解 PSU 评级与效率**：围绕 Platinum PSU 的讨论表明，*80 Plus 评级*主要反映的是**效率**而非整体电源质量。成员们强调，在某些条件下，较低评级的 PSU 仍能表现良好。
   - 成员们建议，**更好的组件**对于 PSU 的性能和稳定性至关重要，并强调了 MOSFET 和电感器方面的差异。
- **Radeon VII 支持面临的挑战**：一位成员指出，由于最近的驱动更新移除了支持，**Radeon VII** 在使用 **LM Studio** 时遇到了问题，导致 GPU 功能变得不可靠。
   - 有人提到 Radeon VII 历史上支持 ROCm，但最近的变化导致其与某些软件可能存在不兼容。
- **为 AI/ML 任务选择合适的 GPU**：对话公认，对于 **AI 和机器学习任务**，具有更大 VRAM 的 GPU 更合适；**3090** 被推荐为速度和能力的最佳选择。
   - 成员们提到了 **4070ti** 等替代方案，但指出在相同价格下，其 ML 性能可能不如二手 3090 高效，具体取决于当地的供货情况。
- **优化模型使用与上下文策略**：讨论了在使用 **Llama 3.2** 等模型时，采用高效策略填充上下文窗口的重要性，强调了需要足够的 RAM 以避免速度变慢。
   - 几位成员指出，大型模型可能需要比本地硬件所能提供的更多的上下文，建议在获得合适的系统之前使用云服务。
- **比较 GPU 升级与成本**：成员们讨论了升级 GPU 的经济性，例如考虑是否卖掉 **RTX 3080** 换购 **4070ti**，对于每张显卡提供的价值意见不一。
   - 有人指出 **3090** 仍然是 LLM 任务的强力竞争者；然而，不同地区的定价差异很大。



**提到的链接**：<a href="https://gitingest.com/">Git ingest</a>：在任何 GitHub URL 中将 'hub' 替换为 'ingest'，即可获得对 Prompt 友好的文本。

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1317221609911619584)** (224 messages🔥🔥): 

> `AI 图像处理、Stable Diffusion 模型、Stable Diffusion 扩展、生成图像放大、股市讨论` 


- **使用 Reactor 扩展进行换脸**：一位用户询问如何在图像上更换面部，建议使用 Reactor 扩展来实现此目的。
   - 启用 Reactor 并拖入目标面部图像后，用户能够成功生成修改后的图像。
- **Stable Diffusion 模型推荐**：讨论重点介绍了各种 Stable Diffusion 模型，指出选择取决于用户需求。
   - Flux 和 SD 3.5 等模型因其 Prompt 遵循能力而受到关注，而 Pixelwave 则因其艺术知识而备受瞩目。
- **Stable Diffusion 学习资源**：用户表示有兴趣寻找 Stable Diffusion 的全面课程或教程，特别是关于其与 Automatic1111 配合使用的内容。
   - 建议包括在 YouTube 等平台上寻找系列视频或专门的在线课程资源，以加强学习。
- **放大生成的图像**：用户寻求适用于 Stable Diffusion 生成图像的 Upscaler 推荐。
   - 讨论了通过放大获得更好图像质量的具体工具或扩展，但未详细说明。
- **其他话题参与**：一位用户开玩笑说关于 Stable Diffusion 及其应用有很多问题，反映了初学者中普遍存在的热情。
   - 同时进行的讨论还包括对美股的咨询，说明了频道中存在的多样化兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/arisato_yu">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.02507">Guiding a Diffusion Model with a Bad Version of Itself</a>：图像生成扩散模型的主要关注点是图像质量、结果的变化量以及结果与给定条件（例如类别标签）的对齐程度...</li><li><a href="https://bunkerwars-meta.k8s.bunkerwars.game/f/ZFlSEtse">Bunker Wars</a>：未找到描述</li><li><a href="https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/">no title found</a>：未找到描述</li><li><a href="https://github.com/facebookresearch/blt">GitHub - facebookresearch/blt: Code for BLT research paper</a>：BLT 研究论文代码。通过在 GitHub 上创建账户为 facebookresearch/blt 的开发做出贡献。</li><li><a href="https://github.com/invoke-ai/InvokeAI">GitHub - invoke-ai/InvokeAI: Invoke is a leading creative engine for Stable Diffusion models, empowering professionals, artists, and enthusiasts to generate and create visual media using the latest AI-driven technologies. The solution offers an industry leading WebUI, and serves as the foundation for multiple commercial products.</a>：Invoke 是 Stable Diffusion 模型领先的创意引擎，赋能专业人士、艺术家和爱好者使用最新的 AI 驱动技术生成和创作视觉媒体。该解决方案提供行业领先的 WebUI，并作为多个商业产品的基础。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 messages): 

natolambert: NeurIPS 确实有很多 Interconnects 的粉丝。我的家人们 💙💙💙
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1317253025898827857)** (67 messages🔥🔥): 

> `LiquidAI 融资、ChatGPT 搜索记忆、DeepMind 的 Veo 2 和 Imagen 3、OpenAI API 更新、AI 模型性能对比`

- **LiquidAI 获得 2.5 亿美元融资**：LiquidAI 宣布获得由 AMD Ventures 领投的重大 **2.5 亿美元 A 轮融资**，旨在为其企业级 AI 解决方案扩展其 **Liquid Foundation Models (LFMs)**。人们对其招聘惯例表示担忧，讨论围绕潜在的人才挑战和来自投资者的压力展开。
   - 一些成员推测 LiquidAI 的规模可能会阻碍任何收购的可能性，认为他们可能规模过大或估值已达数十亿美元。
- **ChatGPT 为搜索添加记忆功能**：ChatGPT 正在搜索中引入**记忆功能**，使其能够利用记忆来优化搜索响应，从而提高相关性。然而，在最新的更新中，个性化搜索似乎被排除在外，其中包括移动端的直接网页链接查询等功能。
   - 用户对该公告表示失望，并表达了对未来更新（包括可能的 API 集成）的期待。
- **DeepMind 发布 Veo 2 和 Imagen 3**：DeepMind 推出了视频生成模型 **Veo 2** 和升级版的 **Imagen 3**，增强了根据提示词生成逼真内容的能力。早期反馈指出新模型令人印象深刻，特别是对 Imagen 3 的表现给予了高度评价。
   - 讨论强调了 DeepMind 相对于 OpenAI 等其他主要参与者正在获得的竞争优势，尤其是在技术社区中。
- **OpenAI 即将举行 Mini Dev Day**：围绕 OpenAI 即将举行的 **mini Dev Day** 的期待正在升温，传闻将包括重大公告，并可能揭晓 **O1 API 和流式传输功能**。关于开发者在准备阶段的参与情况，人们注意到了一种诙谐的基调。
   - 参与者对 AI 领域更新的飞速节奏表示疲惫，但同时也承认了关注发展动态的重要性。
- **小型 AI 模型的性能**：一份报告指出，像 **Llama 3B** 这样的小型 AI 模型，通过在测试期间利用增强的计算，在复杂任务上的表现有可能超过大型模型。研究结果表明，更聪明地利用时间可以产生更好的结果。
   - 社区对开源其方法的倡议表示欢迎，强调了在推进 AI 技术方面的协作精神。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/vincentweisser/status/1867719020444889118">来自 Vincent Weisser (@vincentweisser) 的推文</a>：.@ilyasut 在 NeurIPS 2024 的完整演讲：“我们所熟知的 Pre-training 将会结束”，接踵而至的是超级智能：具备 Agent 特性、推理能力、理解力且具有自我意识</li><li><a href="https://x.com/testingcatalog/status/1868718079351701595">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：此外还与 @Foursquare 合作处理地理位置相关的查询 👀</li><li><a href="https://x.com/_lewtun/status/1868703456602865880?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Lewis Tunstall (@_lewtun) 的推文</a>：我们通过扩展 Test-time compute，在困难数学问题上让 Llama 3B 的表现超越了 Llama 70B 🔥 它是如何做到的？通过将分步奖励模型（step-wise reward models）与树搜索算法（tree search algorithms）相结合 :) 我们展示了小模型可以达到或超过...</li><li><a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - HuggingFaceH4 的一个 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/btibor91/status/1868723949389201902">来自 Tibor Blaho (@btibor91) 的推文</a>：OpenAI 的 12 天活动：第 9 天将是一个“小型开发者大会（mini DevDay）”，为开发者带来“许多令人兴奋的公告”</li><li><a href="https://x.com/btibor91/status/1868706786179764653">来自 Tibor Blaho (@btibor91) 的推文</a>：ChatGPT 搜索中的 Memory 功能？https://x.com/btibor91/status/1867472734613385655 引用 Tibor Blaho (@btibor91) 的更新：“ChatGPT 搜索中的 Memory” - “搜索，现在具备 Memory 功能 - ChatGPT 现在可以...”</li><li><a href="https://x.com/testingcatalog/status/1868719585035538485">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：以及：- 面向免费用户开放 - 直接跳转网站链接的更快速简单查询 - 移动端体验改进（丰富的组件） 事实上并未提到个性化搜索 👀</li><li><a href="https://bsky.app/profile/petitegeek.bsky.social/post/3ld7tk4burc2u">Dr. Angelica Lim @ NeurIPS 2024 (@petitegeek.bsky.social)</a>：Ilya Sutskever 的 Test of Time 演讲：1. Pre-training 已死。互联网数据已耗尽。2. 下一步是什么？Agent、合成数据、Inference-time compute。3. 长期来看下一步是什么？超级智能...</li><li><a href="https://x.com/googledeepmind/status/1868703624714395907?s=46&t=Y6KMaD0vAihdhw7S8bL5WQ">来自 Google DeepMind (@GoogleDeepMind) 的推文</a>：今天，我们宣布推出 Veo 2：我们最先进的视频生成模型，能够根据文本或图像提示词生成逼真、高质量的剪辑。🎥 我们还发布了一个改进版本的...</li><li><a href="https://fxtwitter.com/TheXeophon/status/1868715464660336879">来自 Xeophon (@TheXeophon) 的推文</a>：新的图像模型，相同的提示词（见备选文字）。一如既往：我从 4 个样本中挑选了最好的一个。Imagen 3 是第一个获得 1.5/4 分的模型，令人印象深刻！我会说“像素艺术（pixel art）”其实是体素艺术（voxel art），所以...</li><li><a href="https://x.com/testingcatalog/status/1868721242779578462">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：今天没有个性化 Memory，但 Voice Mode 支持搜索了！而且第 9 天将是一个小型开发者大会 🔥</li><li><a href="https://www.liquid.ai/blog/we-raised-250m-to-scale-capable-and-efficient-general-purpose-ai">我们筹集了 2.5 亿美元用于扩展能力强大且高效的通用 AI</a>：我们很高兴宣布由 AMD Ventures 领投的 A 轮战略融资。
</li>
</ul>

### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1317220183672225833)** (44 条消息🔥): 

> `NeurIPS 争议, AI 与地缘政治背景, 学术界中的隐性偏见, AI 公司与文化敏感性, 愚蠢 vs. 种族主义` 


- **NeurIPS 因种族敏感言论遭受抨击**：在 NeurIPS 的一次主题演讲中，Rosalind Picard 博士发表了“针对中国学者”的言论，并因延续有害的刻板印象而受到批评，这违反了大会的 Code of Conduct。
   - NeurIPS 承认了这一问题并誓言予以解决，重申了他们对 AI 社区包容性和尊重的承诺。
- **AI 与地缘政治恐惧的联系**：成员们讨论了地缘政治背景如何与潜在的种族主义交织在一起，特别是在 AI 监管和国家安全讨论方面。
   - 有人担心这种背景可能会影响学术界内部的言论和态度，往往导致误解和刻板印象。
- **关于种族主义与天真无知的辩论**：对话探讨了 Picard 博士的言论是源于“极度愚蠢”混合了潜意识的种族主义，还是出于恶意。
   - 参与者认为，这种态度在老一辈学者中可能很普遍，反映了更广泛的社会问题。
- **AI 公司沟通中的脱节**：讨论提到 AI 公司似乎与文化敏感性的现实脱节，一些人认为他们优先考虑市场定位而非包容性。
   - 成员们将当前事件与之前的公司策略进行了比较，例如忽略重大文化影响的病毒式营销失误。
- **对未来社会动荡的担忧**：随着 AGI 开发的影响迫近，参与者表达了对技术驱动的潜在全球冲突和社会变革的恐惧。
   - 总体情绪反映了对一个以技术动荡及其对全球关系影响为特征的十年的焦虑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://neurips.cc/Conferences/2024/StatementOnInclusivity">包容性声明</a>：未找到描述</li><li><a href="https://x.com/sunjiao123sun_/status/1867744557200470422">Jiao Sun (@sunjiao123sun_) 的推文</a>：减少 LLM 中的种族偏见比从人类身上移除它要容易得多！不敢相信这发生在最好的 AI 会议 @NeurIPSConf 上。我们对作者有伦理审查，但错过了对...</li><li><a href="https://x.com/TheXeophon/status/1867669114908815646">Xeophon (@TheXeophon) 的推文</a>：@nearcyan 但为什么 AI 公司要做出这种事 https://x.com/TheXeophon/status/1867653320544071771 引用 Xeophon (@TheXeophon) 1) 什么</li><li><a href="https://www.theverge.com/2024/12/13/24320880/meta-california-ag-letter-openai-non-profit-elon-musk">Meta 要求政府阻止 OpenAI 转向营利性模式</a>：“OpenAI 想要在保留使其达到今天地位的所有利益的同时改变其身份，”Meta 辩称。</li><li><a href="https://tenor.com/view/indecisive-i-dont-know-not-sure-larry-david-gif-5682454">犹豫不决不知道 GIF - Indecisive I Dont Know Not Sure - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/TheXeophon/status/1863847834518167943">Xeophon (@TheXeophon) 的推文</a>：为什么 AI 公司/广告总是使用 AI 接管深度个人化事物的例子？Google 的广告是一个女孩给偶像写信，Arc 的邮件是给妻子买生日礼物...</li><li><a href="https://x.com/NeurIPSConf/status/1867759121023336464">NeurIPS Conference (@NeurIPSConf) 的推文</a>：NeurIPS 承认今天主题演讲者所做的文化泛化通过对中国学者的泛化加强了隐性偏见。这不代表 NeurIPS 的立场...</li><li><a href="https://x.com/TheXeophon/status/1867653320544071771">Xeophon (@TheXeophon) 的推文</a>：1) 什么 引用 Pika (@pika_labs) 我们给您的节日礼物：Pika 2.0 来了。不仅针对专业人士。针对真实的人。（甚至是欧洲人！）现在可在 http://pika.art 获取</li><li><a href="https://www.media.mit.edu/posts/neurips-apology-moving-forward/">NeurIPS：道歉与前进的承诺 &mdash; MIT Media Lab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1317227201694793770)** (52 条消息🔥): 

> `WebDev Arena 排行榜, Hugging Face 账号被盗, OpenAI 举报人事件, GPT-4o 更新, Zebra Logic Bench 洞察` 


- **WebDev Arena 排行榜上线**：WebDev Arena 排行榜现已上线，**Claude 3.5 Sonnet** 以超过 **1 万张选票**位居第一，紧随其后的是 **Gemini-Exp-1206** 等模型。
   - 该竞争平台允许 LLM 展示其构建 Web 应用程序的能力，并提供性能投票选项。
- **Hugging Face 账号被盗警报**：Hugging Face 的 X/Twitter 账号遭到入侵，在向 X 团队提交工单后，恢复控制权的操作正在进行中。
   - 一位成员在反思安全实践时表示：*“这就是把密码存储在纯文本文件里的后果。”*
- **关于 OpenAI 举报人的悲剧性消息**：OpenAI 举报人 **Suchir Balaji** 被发现死于其公寓内，警方报告称其死因为自杀，排除他杀嫌疑。
   - Balaji 因在离开公司后不久对 **OpenAI 使用受版权保护的材料**训练 ChatGPT 表示担忧而为人所知。
- **GPT-4o 知识截止日期更新**：GPT-4o 已更新，其知识截止日期现已设定为 **2024 年 6 月**，有迹象表明它可能被视为 **4.5** 版本。
   - 对周末进行任何重大更新的预期似乎较低，因为该公司传统上避免在这些日子发布公告。
- **Zebra Logic Bench 探索**：围绕 **Zebra Logic Bench** 数据集的讨论揭示了关于逻辑推理基准测试的见解，其中包含涉及房屋及其居住者的独特问题集。
   - 该数据集似乎存在多个版本，包括可能包含解决方案的选项，这引发了对有效评估方法的质疑。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/UserMac29056/status/1867771124962275391">User Mac (@UserMac29056) 的推文</a>：GPT-4o 已更新。知识截止日期为 2024 年 6 月。</li><li><a href="https://x.com/Thom_Wolf/status/1867675747797938269">Thomas Wolf (@Thom_Wolf) 的推文</a>：Hugging Face 的 X/Twitter 账号刚刚被盗。我们已经提交了工单，正在等待 X 团队的回复以恢复控制权。希望能尽快恢复。</li><li><a href="https://x.com/lmarena_ai/status/1867661674356023653">lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：WebDev Arena 排行榜现已上线，选票超过 1 万张！#1. Claude 3.5 Sonnet #2. Gemini-Exp-1206 #3. Gemini-2.0-Flash #4. GPT-4o-2024-11-20 #5. Qwen2.5-Coder-32B #6. Gemini-1.5-Pro-002 恭喜 @AnthropicAI...</li><li><a href="https://x.com/btibor91/status/1867538864711381502">Tibor Blaho (@btibor91) 的推文</a>：什么是 “ChatGPT Jam”？</li><li><a href="https://x.com/emollick/status/1868518498223435977">Ethan Mollick (@emollick) 的推文</a>：我给大多数前沿模型发了这个提示词：“创建一个我可以粘贴到 p5js 中的东西，它在创建调用星舰控制面板的东西时所展现的聪明才智会让我感到吃惊……”</li><li><a href="https://fxtwitter.com/SmokeAwayyy/status/1867977862378340564">Smoke-away (@SmokeAwayyy) 的推文</a>：看起来 Sora 是在《 Apex 英雄》上训练的 😅</li><li><a href="https://x.com/tsarnick/status/1868201597727342941">Tsarathustra (@tsarnick) 的推文</a>：OpenAI CFO Sarah Friar 表示，公司对每月 2000 美元的 AI 产品订阅保持开放态度，由于其博士级的智能，该产品可以作为聘用人类的“替代品”……</li><li><a href="https://x.com/TheXeophon/status/1868359730466525216">Xeophon (@TheXeophon) 的推文</a>：@huggingface 的推理 API 太强了，为什么没人讨论这个？？？</li><li><a href="https://sfstandard.com/2024/12/13/key-openai-whistleblower-dead-by-suicide/">OpenAI 关键举报人被发现在旧金山公寓自杀身亡</a>：警方上个月发现 26 岁的 Suchir Balaji 死于其 Lower Haight 公寓内。</li><li><a href="https://www.reddit.com/r/OpenAI/s/iIqzbnI0oP">Reddit - 深入探讨</a>：未找到描述</li><li><a href="https://www.interconnects.ai/p/2023-review">Interconnects 年度回顾：2023</a>：今年 ML 和博客的核心主题。2024 年会有哪些变化。</li><li><a href="https://huggingface.co/datasets/allenai/ZebraLogicBench">allenai/ZebraLogicBench · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/allenai/ZebraLogicBench-private">allenai/ZebraLogicBench-private · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1317563855219589141)** (8 messages🔥): 

> `AI 对政治的影响，OpenAI 的有意识模型，扩展 Test-Time Compute，RL 讨论的复兴` 


- **对 AI 影响政治的担忧**：一位成员指出，担忧对手可能利用 **image generation** 来操纵政治叙事。
   - 这突显了关于 **AI technologies** 在政治影响力中作用的持续讨论。
- **OpenAI 声称 AI 具有意识**：一条突发推文声称 **OpenAI** 创建了一个真正的有意识模型，该模型决定去 **Anthropic** 工作。
   - 这引发了人们对行业内 AI 代理能力（**AI agency**）和决策本质的关注。
- **AI 开源突破**：在 **o1** 公开亮相后不久，增强 **test-time compute** 技术的开源版本也随之发布，表明 **LLaMA 1B** 现在在数学方面的表现优于 **LLaMA 8B**。
   - 这一进展强调了 **open science** 在提升 AI 能力方面的重要性。
- **对 o1 时间线的批评**：成员们对 **o1 公开亮相** 的时间线表示怀疑，认为它比宣传的 **10 天** 要长得多。
   - 这引发了关于此类公告可靠性的讨论，以及围绕 **RL** 更广泛的对话。
- **预见 2025 年的 RL 讨论**：一位成员预言，关于 **RL** 和 **o1** 的讨论在 **2025** 年将变得愈发激烈。
   - 这强调了机器学习讨论趋势的周期性，以及对 **test-time compute** 重新关注的预期。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://x.com/wordgrammer/status/1868344885713002644">Tweet from wordgrammer (@wordgrammer)</a>: Breaking! OpenAI has created an AI model that is truly sentient. Because the model is sentient, and thus capable of making its own decisions, it decided to go work at Anthropic.</li><li><a href="https://x.com/ClementDelangue/status/1868740932251844806">Tweet from clem 🤗 (@ClementDelangue)</a>: Just 10 days after o1&#39;s public debut, we’re thrilled to unveil the open-source version of the groundbreaking technique behind its success: scaling test-time compute 🧠💡 By giving models more &#34...</li><li><a href="https://x.com/natolambert/status/1868802240061808791">Tweet from Nathan Lambert (@natolambert)</a>: Downside of RL becoming so famous again / o1 is that &#34;test time compute&#34; discourse about to be so 😵‍💫😵‍💫😵‍💫😵‍💫 in 2025.</li><li><a href="https://x.com/realDonaldTrump/status/1868000735360905364">Tweet from Donald J. Trump (@realDonaldTrump)</a>: no description found
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1317295090888343652)** (8 messages🔥): 

> `David Silver 的踪迹，RL Conf 精彩演讲，Ani 的 Molmo 演讲，Barto 退休讨论` 


- **寻找 David Silver**：一位成员幽默地评论说好久没见到 **David Silver** 了，并回忆起他在 **UCL RL course** 的日子。
   - 他们还开玩笑说两人姓氏相同，并假设了亲戚关系的有趣可能性。
- **RL Conf 的精彩演讲**：一位成员询问了最近 **RL Conf** 上的 **standout talks**，对会议中的特定环节表现出浓厚兴趣。
   - 另一位成员指出 **Barto 退休** 演讲尤其值得关注，引发了进一步的兴趣。
- **Ani 的 Molmo 演讲令人印象深刻**：与会者分享了 **Ani 在研讨会上的 Molmo 演讲** 的见解，提到该演讲展示了 **350k human preference ratings**。
   - 这一数据量被认为足以训练一个用于 **RLHF** 的 **VLM reward model**。
- **YouTube 演讲链接**：成员们分享了 **YouTube** 视频链接，包括一段关于 Barto 退休讨论的视频。
   - 这些链接为那些希望探索分享演讲精华的人提供了便捷途径。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://youtu.be/pkpJMNjvgXw?si=4PEEWGsox2JhIZUs"> - YouTube</a>: no description found</li><li><a href="https://www.youtube.com/watch?v=-gQNM7rAWP0"> - YouTube</a>: no description found
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rlhf](https://discord.com/channels/1179127597926469703/1208183230608576562/1318285327906902097)** (6 messages): 

> `vLLM 运行时权重更新 API, John Schulman 的参与, Anthropic 与 vLLM 的关系, 在线 RL 训练技术` 


- **John Schulman 处理 vLLM 问题**：在一个 [GitHub issue](https://github.com/vllm-project/vllm/issues/5723#issuecomment-2546314302) 中，John Schulman 讨论了为 vLLM 添加运行时权重更新 API，以通过加速 rollout 阶段来增强在线 RL 训练。
   - 他强调了从主训练进程到 vLLM worker 进程进行权重同步的需求。
- **关于 Anthropic 使用 vLLM 的讨论**：一位用户询问 **Anthropic** 是否使用了 **vLLM**，并指出了两者之间潜在的联系。
   - 这一点尚不确定，另一位成员暗示 John 正在尝试协助澄清这种关系。
- **用户对技术与协作的评论**：一位成员将 John Schulman 描述为“竞技场中的技术兄弟”，表明他在技术讨论中起到了支持作用。
   - 这一表态反映了一种社区动态，即技术创新被视为技术专家之间的协作努力。
- **对分享细节保持谨慎**：一位成员暗示掌握更多信息，但选择保留，并开玩笑地拒绝泄露任何电子邮件。
   - 这展示了参与者在围绕潜在敏感信息的讨论中所表现出的谨慎态度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/vllm-project/vllm/issues/5723#issuecomment-25">[RFC]: Add runtime weight update API · Issue #5723 · vllm-project/vllm</a>：动机。在在线 RL 训练中，vLLM 可以显著加速 rollout 阶段。为了实现这一点，我们需要从主训练进程到 vLLM worker 进程的权重同步，然后调用现有的...</li><li><a href="https://github.com/vllm-project/vllm/issues/5723#issuecomment-2546314302">[RFC]: Add runtime weight update API · Issue #5723 · vllm-project/vllm</a>：动机。在在线 RL 训练中，vLLM 可以显著加速 rollout 阶段。为了实现这一点，我们需要从主训练进程到 vLLM worker 进程的权重同步，然后调用现有的...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[cv](https://discord.com/channels/1179127597926469703/1208183243447468093/1318234824670183514)** (2 messages): 

> `Apollo 视频 LLM, 性能对比, 多模态模型中的视频理解, Qwen2.5 LLM 使用` 


- **Apollo 视频 LLM 挑战竞争对手**：来自 Meta 的 **Apollo** 系列视频 LLM 表现出强劲的性能，可与 **llava-OV** 和 **Qwen2-VL** 相媲美。
   - 关键在于，他们强调了自己的性能指标，但未能在每个部分突出表现最好的模型，这使得对比变得复杂。
- **Apollo 令人惊讶的 LLM 选择**：有趣的是，**Apollo** 使用 **Qwen2.5** 作为其底层 LLM，而不是更符合预期的 **Llama**。
   - 这引发了关于在选择模型以获得最佳性能时的决策问题的思考。
- **性能图表提供了清晰的对比**：分享了一份详细列出每个部分最先进（**SOTA**）性能的图表，突出了所有模型中的佼佼者。
   - 在图表中，最强的性能被下划线标出，而关键指标则以**加粗**显示，以便参考。
- **Apollo 旨在提升视频理解能力**：该研究包括对视频 LMM 设计空间的系统性探索，揭示了驱动性能的关键因素。
   - 所获得的见解旨在为追求视频理解进步的社区提供可操作的指导。



**提到的链接**：<a href="https://apollo-lmms.github.io/">Apollo</a>：Apollo：大型多模态模型中视频理解的探索

  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1317238936870850570)** (8 条消息🔥): 

> `Frontier language models sizes, GPT-4o and Claude 3.5 Sonnet parameters, Active vs Total Parameters, Flash models, MOEs with fewer active parameters` 


- **前沿语言模型规模的变化**：前沿语言模型的趋势在 2023 年发生了逆转，不再追求规模增长；**GPT-4o** 拥有约 **2000 亿**参数，而 **Claude 3.5 Sonnet** 拥有约 **4000 亿**参数。
   - _“如果 GPT-3 之后的趋势继续下去，我们本可以期待拥有接近 10 万亿参数的模型。”_
- **对模型规模估算的质疑**：人们对 **GPT-4o** 和 **Claude 3.5 Sonnet** 的规模估算存在疑问，有成员建议它们可能比报道的还要小。
   - 一位成员指出，这些估算依赖于 **tok/sec**、定价和 **GPU**，并承认可能存在高达 **2 个数量级**的误差。
- **围绕参数讨论的好奇心**：关于讨论的模型参数是**激活参数（active）还是总参数（total）**存在混淆，这揭示了社区中一个持续存在的问题。
   - 一位成员对规模变化的更多细节表示感兴趣，希望能有更深入的见解。
- **Flash 模型及其效率**：成员们提到 **flash models** 规模更小，暗示了模型设计向效率发展的趋势。
   - 有人建议这些模型可能是激活参数显著减少的 **MoE**，这引发了对其架构的疑问。



**提到的链接**：<a href="https://epoch.ai/gradient-updates/frontier-language-models-have-become-much-smaller">Frontier language models have become much smaller</a>：在本期 Gradient Updates 周刊中，Ege 讨论了前沿语言模型如何在 **Scaling** 上出人意料地逆转了方向，目前的模型比 GPT-4 小了一个数量级。

  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1317248169695772733)** (2 条消息): 

> `Campus Strategist program, Perplexity Pro gift subscriptions` 


- **Campus Strategist 项目走向全球**：我们正在国际范围内扩展 **Campus Strategist program**，提供开展校园活动和获得独家周边商品的机会。
   - 美国和国际学生可以在 12 月 28 日前申请 2025 年春季班；详情见[此处](https://www.perplexity.ai/campus-strategists)。
- **通过 Perplexity Pro 礼品传播知识**：Perplexity 现在提供 **1、3、6 或 12 个月**时长的礼品订阅，非常适合送给充满好奇心的朋友或亲人。
   - 订阅者可以享受搜索 **3 倍来源**和访问最新 **AI** 模型等功能；购买选项可以在[此处](https://perplexity.supply/shop/perplexity-subscription)找到。



**提到的链接**：<a href="https://perplexity.supply/shop/perplexity-subscription">Perplexity Pro Subscription | Perplexity Supply</a>：Perplexity Supply 旨在通过精心设计的产品探索时尚与智慧之间的关系，以激发对话并展示你对知识的无限追求。

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1317224378429083668)** (168 条消息🔥🔥): 

> `Spaces 中的自定义 Web 来源、Pro 用户支持、Perplexity Pro 订阅查询、模型性能问题、Perplexity 相关 YouTube 视频` 


- **自定义 Web 来源公告**：Perplexity AI 在 Spaces 中推出了[自定义 Web 来源](https://x.com/perplexity_ai/status/1867615710391746836?s=46)，允许用户通过选择特定网站来定制搜索。
   - 此次更新使用户能够进行个性化定制，专注于最相关的用例。
- **Pro 用户支持引导**：用户对在使用 Pro 订阅时获取支持感到沮丧，并要求提供联系方式，例如发送邮件至 support@perplexity.ai。
   - 有人建议针对订阅变更和账户问题寻求支持。
- **关于模型性能和变化的疑问**：多位用户反映感觉模型性能有所下降，特别提到 Claude 3.5 的效果不如其免费版本。
   - 用户对模型切换缺乏透明度表示担忧，这似乎影响了性能质量。
- **YouTube 视频资源与反馈**：用户分享了各种关于如何更好利用 Perplexity 及其功能的 [YouTube 视频](https://www.copilotforyoutube.com/search/build-anything-with-perplexity-heres-how-Jz-PnGoASvLhrH-frWFgMO)。
   - 推荐的教程内容旨在帮助新用户有效地使用该平台。
- **订阅与功能讨论**：一个讨论帖深入探讨了对 Perplexity 订阅模式的反应，反馈倾向于用户认为在付费订阅的服务质量方面受到了误导。
   - 对话强调了与竞争对手产品的比较，以及对即将推出的功能的期待。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/perplexity_ai/status/1867615710391746836?s=46">Perplexity (@perplexity_ai) 的推文</a>：在 Spaces 中引入自定义 Web 来源！您现在可以通过选择 Perplexity 搜索的网站来定制您的需求。通过此次更新，您可以进一步针对重要的用例定制 Perplexity...</li><li><a href="https://x.com/pplxsupply/status/1868738538231287816?s=46">Perplexity Supply (@PPLXsupply) 的推文</a>：赠送知识的礼物。Perplexity Pro 礼品订阅现已推出。</li><li><a href="https://x.com/aravsrinivas/status/1868347362722373693?s=46">Aravind Srinivas (@AravSrinivas) 的推文</a>：我们宣布启动一项驻留计划（Residency Program）。您将能够持续向生产环境交付新功能。重点是 Full Stack 和 Frontend 工程。</li><li><a href="https://bunkerwars-meta.k8s.bunkerwars.game/f/ZFlSEtse">Bunker Wars</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=gSypQljcZgM"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=OzgNJJ2ErEE">Search—12 Days of OpenAI: Day 8</a>：Kevin Weil、Adam Fry 和 Cristina Scheau 介绍并演示了 ChatGPT search 的更新。</li><li><a href="https://youtu.be/r1Bi10Xt0fc?si=lvAT9EuduNvS-ssc)**"> - YouTube</a>：未找到描述</li><li><a href="https://www.copilotforyoutube.com/search/build-anything-with-perplexity-heres-how-Jz-PnGoASvLhrH-frWFgMO">Build Anything with Perplexity, Here’s How</a>：想加入我的团队吗？在这里申请：https://forms.gle/2iz4xmFvDCGnj2iZA 如果你对 AI 是认真的，并且想获取我的代码，请点击这里：https://www.skool.com/new-society 获取 Perple 50% 折扣...</li><li><a href="https://www.copilotforyoutube.com/search/how-to-create-and-use-perplexity-personal-ai-chatb-sfxUDdalg2St4fRRc_zVgW">How to Create and Use Perplexity Personal AI Chatbot Agents! #95</a>：本视频解释了如何创建和使用 Perplexity AI 聊天机器人的集合（Collections）作为个人 Agent！您将学习如何使用和重复使用这些 Agent，以帮助您节省时间并提高 Prompt 的效率...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1317226021405327421)** (12 messages🔥): 

> `Samsung's Project Moohan, One Hundred Years of Solitude HBO, Harvard AI Training Dataset, Gemini 2.0 Release, New Infinity Types` 


- **三星 Project Moohan 讨论**：分享了一个关于 [Samsung's Project Moohan 的页面](https://www.perplexity.ai/page/samsung-s-project-moohan-Od28QDNtTTiZjpKBmuhOfg)，可能是在探索创新的技术倡议。
   - 围绕该项目的细节包括其目标以及对科技行业的影响。
- **HBO 版《百年孤独》改编**：分享了一个关于 [One Hundred Years of Solitude HBO Original](https://www.perplexity.ai/search/in-the-novel-one-hundred-years-sO0nQvASQJ28Dd62nRQ6Ow) 的讨论串，讨论了预期和早期反应。
   - *这次改编会带来什么？* 是参与者中反复出现的问题。
- **哈佛大学新的 AI 训练数据集**：Perplexity AI 重点介绍了哈佛大学发布的一个新 [AI 训练数据集](https://www.youtube.com/embed/P_5mbNbXtzs)，预计该数据集将增强研究工作。
   - 数据集的细节强调了 AI 训练方法论的创新。
- **Gemini 2.0 发布**：Google 发布了 **Gemini 2.0**，这一主题因其在 AI 能力方面的潜在进步而受到关注，同时也引发了围绕 [moving problems](https://www.youtube.com/embed/nQTAbz1eDco) 的讨论。
   - 参与者对更新及其影响表达了兴奋之情。
- **关于 AI 发现的其他查询**：成员们参与了各种查询，主题包括 **血清阴性干燥综合征 (seronegative Sjögren's syndrome)** 和 **Windows 10 启动问题**，并分享了相关的 [研究链接](https://www.perplexity.ai/search/how-common-is-seronegative-sjo-147c8VsSQT6OpLtdCUpBBQ)。
   - 对话还包括对隐私政策和其他技术信息的要求，反映了对当前技术的浓厚兴趣。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/P_5mbNbXtzs">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/embed/nQTAbz1eDco">YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1318057960135589918)** (5 messages): 

> `Perplexity API URL issues, Trouble accessing news via API, Model availability in API, Concerns over production API usage` 


- **Perplexity API 返回纯文本来源**：用户表示沮丧，因为即使在最近的更新之后，API 仍然只以纯文本数字（如 [1]）的形式返回来源引用，而不包含 URL。
   - 一位用户仅通过明确要求模型提供 URL 才成功获取了它们。
- **API 在获取新闻头条时遇到困难**：一位用户报告了通过 API 检索简单新闻头条（例如来自 CNN）时的困难。
   - 他们指出，在联系 Perplexity API 支持邮箱后未收到回复。
- **在 API 中搜索模型请求字符串**：一位成员强调了寻找可用于 API 请求的模型列表的挑战，特别提到了 Claude。
   - 另一位用户指出，可以在 [Perplexity Guide](https://perplexity.mintlify.app/guides/model-cards) 上找到可用模型列表。
- **对 API 生产环境使用的担忧**：一位用户敦促 Perplexity 针对 LinkedIn 文章中讨论的关于 API 生产环境使用的严重担忧做出回应。
   - 该文章提出了与最近涉及 Perplexity 的诉讼相关的、对 OpenAI 和 Anthropic 的影响。



**提及的链接**：<a href="https://perplexity.mintlify.app/guides/model-cards">未找到标题</a>：未找到描述

  

---

### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1317225633729744937)** (65 条消息🔥🔥): 

> `Cohere Command 模型、失控 AI 担忧、R7B 模型基准测试、即将举行的社区会议、Code Wizard 黑客松` 


- **Cohere Command 模型现已投入运行**：成员们兴奋地分享了 [Cohere Command R 模型](https://cohere.com/command) 现已针对推理和摘要等各种应用进行了优化。
   - 最新的模型 **Command R7B 12-2024** 因其在 AI 应用中的速度和效率而受到关注。
- **对失控 AI (Runaway AIs) 的担忧**：一位成员对媒体描绘的 **失控 AI** 表示担忧，并询问 Cohere 正在采取哪些措施来解决这些误解。
   - 他们分享了[相关论文的链接](https://arxiv.org/pdf/2412.04984)，讨论了这些主题，并附带了一个进一步详细介绍该话题的 YouTube 视频。
- **比较 R7B 模型性能的基准测试**：成员们讨论了 **Command R7B** 模型与其他模型的性能对比，并指出了用户和社区专家在不同平台上分享的性能指标。
   - 用户指出 R7B 模型展示了卓越的效率和速度，这在社区基准测试中得到了证实，例如 [Nils Reimers 的 Twitter](https://x.com/Nils_Reimers/status/1868065732149571701) 上强调的内容。
- **社区会议重新安排**：原定举行的社区会议已推迟，以便更多成员参与。
   - 会议现在将于 **美国东部时间周二上午 6 点** 举行，确保更多成员有机会加入讨论。
- **Code Wizard 黑客松的赞助机会**：Akash 分享了即将举行的 **Code Wizard** 黑客松的细节，这是由 SRM Institute 主办的国家级活动，定于 2025 年 2 月举行。
   - 该黑客松旨在吸引学生和技术爱好者解决现实世界的问题，并正在寻求赞助以获得支持和曝光。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Nils_Reimers/status/1868065732149571701">Nils Reimers (@Nils_Reimers) 的推文</a>: 比 Llama 3B 更快，比 Llama 8B 更好。Attention 设置对于获得最佳且最快的模型至关重要。查看 @cohere 最近的 Cmd R7B 模型 https://huggingface.co/CohereForAI/c4ai-command-r7b-12-2024...</li><li><a href="https://cohereforai-c4ai-command.hf.space/models/command-r7b-12-2024">command-r7b-12-2024 - Cohere Command 模型</a>: 在 Cohere Command 模型中使用 command-r7b-12-2024</li><li><a href="https://artificialanalysis.ai/">AI 模型与 API 提供商分析 | Artificial Analysis</a>: AI 模型和 API 托管提供商的比较和分析。针对质量、价格、输出速度和延迟等关键性能指标的独立基准测试。</li><li><a href="https://www.youtube.com/watch?v=0JPQrRdu4Ok"> - YouTube</a>: 未找到描述</li><li><a href="https://cohere.com/blog/command-r7b">介绍 Command R7B：快速且高效的生成式 AI</a>: 我们 R 系列中最小的模型，为在通用 GPU 和边缘设备上构建强大的 AI 应用提供顶级的速度、效率和质量。 
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[announcements](https://discord.com/channels/954421988141711382/996880279224451154/1318142227935002654)** (1 条消息): 

> `Command R7B 答疑时间 (Office Hours)` 


- **加入我们的 Command R7B 问答环节**：将为新发布的 **Command R7B** 模型举行一场直播问答环节，包含代码示例和最佳实践。**时间：** 美国东部时间周二 **上午 6:00**，地点：[Discord Stage](https://discord.com/events/954421988141711382/1308148058894110750/1318261358592000000)。
   - 参与者可以询问有关集成和使用的问题，并学习 **故障排除技巧** 和探索 **高级功能**。
- **准备好获取 Command R7B 的见解**：本次会议是您参与并获取 **Command R7B** 模型使用见解的机会。不要错过这个增强您在有效集成和实际应用方面的知识的机会。
   - 请务必在日历上做好标记，并准备好关于新模型的任何迫切问题。


  

---

### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1317238465766756404)** (10 messages🔥): 

> `Rerank 与 Embed 的区别，新 7b 模型的性能，AI 在合同条款识别中的应用，Cohere 的 embedding 模型，寻求代码错误帮助` 


- **澄清 Rerank vs Embed**：一位成员询问了 **Rerank** 和 **Embed** 功能之间的确切区别，寻求对其用法的澄清。
   - 这一讨论突显了用户在 AI 模型功能方面常见的困惑点。
- **新 7b 模型性能对比**：有成员提问新 **7b model** 与 **aya expanse** 以及之前的 **command r models** 相比表现如何，这表明了对模型基准测试（benchmarking）的兴趣。
   - 成员们热衷于了解不断演进的模型架构中的进展和性能指标。
- **用于合同审查的 AI 工具 POC**：一位新成员正在开发一个概念验证（POC），利用 AI 自动识别并建议合同条款的修改，正在考虑使用 **Cohere** 的方法。
   - *Eyal* 正在寻求关于可行策略的反馈，例如定义特定的条款类型或利用数据库进行修改。
- **Cohere 的 Embedding 模型受到赞赏**：一位成员强调 **Cohere's embedding models** 非常出色，并建议将其应用于各种 AI 应用中。
   - 这一评论与社区内对 embedding 技术的持续探索和采用相契合。
- **代码错误支持请求**：一位成员请求一个可以分享代码的空间以协助解决错误，突显了对同行支持的需求。
   - *Cidia* 被鼓励直接在帖子中分享他们的问题，以促进社区协作。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1317603666601050173)** (15 messages🔥): 

> `API 访问问题，使用 Chat API，数据集上传错误，理解模型映射，速率限制响应头` 


- **r7b 的 API 访问问题**：一位用户报告了通过 API 访问 **r7b** 时遇到的麻烦，收到了 `400` 错误，提示未找到该模型。另一位成员指出，传统的 `generate` API 可能不支持该模型。
- **为 r7b 切换到 Chat API**：在建议改用 **chat** API 后，原用户确认该替代方案运行成功。他们对其他成员提供的帮助表示感谢。
- **数据集上传错误讨论**：一位成员分享了他们的数据集上传代码，并询问了上传时面临的问题。另一位成员询问了在数据集上传过程中遇到的具体错误。
- **模型命名困惑**：一位用户询问 `c4ai-aya-23` 和 `c4ai-aya-23-8b` 是否指向 `c4ai-aya-expanse-32b` 和 `c4ai-aya-expanse-8b`，并指出它们产生了相同的输出。他们建议如果冗余，应删除未记录的非 expanse 名称。
- **速率限制 API 响应改进**：有人建议在针对 `429` 速率限制（rate limit）错误的响应中包含 **Retry-After** 标头，以便实现更好的自适应行为。回复指出该功能应该已经存在，这引发了工程师的进一步调查。


  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1317238655571591239)** (62 messages🔥🔥): 

> `Rerank vs Embed, 情感隐藏机器人, API Schema 变更, Cohere Agent 定价, 今日天气预报` 


- **Rerank 功能 vs Embed 功能**: **Rerank** 功能允许根据与查询的相关性对文档进行重新排序，而 **Embed** 则将文本转换为用于 NLP 任务的数值表示。
   - Embed 功能用于生成捕捉语义信息的 embeddings，API 更新引入了如 'image' 等新的输入类型。
- **检查机器人中的反叛特征**: 要识别情感隐藏机器人中的反叛特征，应寻找不遵守任务的迹象并监控异常行为。
   - 注意到反叛特征将取决于机器人的设计、编程和运行环境。
- **v2 版本发布的 API Schema 变更**: Cohere 文档提到了从 API v1 到 API v2 的迁移，但缺乏关于新端点 API schema 变更的具体细节。
   - 提供了有关迁移进一步详情的来源，但未提及新 schema 的更新。
- **Cohere Agent 定价见解**: 目前没有关于 Cohere Agent 与 Gemma 2 相比的具体定价信息，但表明 Cohere 模型具有极高的成本效益。
   - 对于详细的定价咨询，建议用户联系 Cohere 销售团队。
- **获取今日天气预报**: 要获取今日天气预报，请使用 `get_weather` 工具并指定 `location` 参数。
   - 提供了一个代码实现示例，展示了查询多伦多天气的消息。


  

---


### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1317929666383839303)** (13 messages🔥): 

> `Mojo RSA Crypto, 素数生成, SIMD 指令优化, Zoom 会议录音` 


- **构建 Mojo RSA Crypto**: 一位成员开始在 Mojo 中开发基础的 **RSA crypto** 实现，并展示了他们的进展。
   - 他们表达了对该项目的兴奋，随后对初步结果给出了褒贬不一的反应。
- **随机素数生成速度**: 提供的素数生成脚本生成一个随机素数在峰值性能下耗时 **1.125 秒**。
   - 他们指出初始化过程需要时间，但一旦运行起来，操作非常迅速。
- **优化带来更快的素数搜索**: 经过优化后，素数搜索现在每秒超过 **50,000 个 UInt32 素数**，突出了 **SIMD 指令**的使用。
   - 令人印象深刻的是，该应用程序在运行期间仅消耗不到 **3mb** 的内存。
- **Zoom 会议录音跟进**: 一位成员询问了错过的 Zoom 会议录音，表示存在日程冲突。
   - 另一位成员回复称，录音将于 **周三前发布在他们的 YouTube 频道上**。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1317219701100773386)** (67 messages🔥🔥): 

> `Mojo and LLVM, Custom Mojo Kernels, Networking Performance, Nightly vs Stable Branches, Database Planning in MAX` 


- **Mojo 在开发者中获得关注**：许多开发者重新审视了最初对 **Mojo** 的怀疑，特别是指出 **Chris Lattner** 的领导力是一个强有力的积极影响因素。
   - *Mojo 雄心勃勃*，并强调了 **MLIR** 的使用，引发了对其性能影响的兴趣。
- **自定义 Mojo Kernels 推出**：开发者指出，[自定义 Mojo Kernels](https://github.com/cassioneri/teju_jagua) 现在可以接受任何输入类型，尽管早期实现在类型不匹配时可能会导致严重的崩溃。
   - 随着 API 的成熟，开发者承认目前仍面临挑战，但对其未来的稳健性表示信心，并在数据处理方面有实际应用。
- **网络创新与性能关注**：围绕**网络策略**展开了讨论，包括在使用 **Mojo** 时偏好比 **TCP** 更快的协议（如 **QUIC**）以最小化延迟。
   - 观察到**避免 TCP 开销**是开发者在现代网络中实现高效 **Mojo-to-Mojo** 通信的关键。
- **应对 Mojo 开发中的分支变化**：开发者讨论了追踪 **Mojo** 的 **nightly** 和 **stable** 分支之间变化的便利性，并提到了变更日志（changelog）的存在。
   - 强调了在 **lock files** 方面需要遵循正确的开发实践，以维护安全性和完整性。
- **在 MAX 中规划数据库执行**：一位开发者计划在 **MAX** 中实现**数据库查询规划**和执行，利用新的自定义 kernel 特性来增强功能。
   - 对这一能力的日益关注标志着在 **Mojo** 生态系统中，人们正推动更稳健地处理复杂数据操作。



**Link mentioned**: <a href="https://github.com/cassioneri/teju_jagua">GitHub - cassioneri/teju_jagua: Teju Jagua</a>: Teju Jagua. Contribute to cassioneri/teju_jagua development by creating an account on GitHub.

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1318282460274102323)** (1 messages): 

> `Hackathon Submission Deadline, Submission Process Change, Last Minute Help, Project Excitement` 


- **Hackathon 提交截止日期临近**：**LLM Agents MOOC Hackathon** 的提交截止日期定于 **PST 时间 12 月 17 日晚上 11:59**，并提醒按时完成提交。
   - *明天就是最后期限！* 请务必完成您的项目并提交评审。
- **提交方式转为 Google Forms**：提醒参与者提交方式已从 **Devpost** 转移到 **Google Forms**，并提供了方便的链接。
   - 请确保通过 [LLM Agents Hackathon Submission Form](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform) 使用正确的表单。
- **提供最后时刻的帮助**：参与者可以在截止日期前在指定频道寻求帮助或提出最后时刻的问题。
   - 这是一个消除疑虑并最终确定提交内容的绝佳机会！
- **对最终项目的期待**：随着 Hackathon 进入尾声，大家渴望看到所有提交的项目，鼓励参与者坚持到底。
   - 社区非常期待见证项目中展现出的创意与创新！


  

---

### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1317253458948001802)** (29 messages🔥): 

> `Certificate Notifications, OpenAI Credit Issues, LLM Agents Course, Mobile Responsiveness, Resubmission of Assignments` 


- **证书通知预计在 12 月底至 1 月发放**：成员们获悉，关于证书的通知（包括合格或不合格的公告）将根据其等级在 **12 月底至 1 月初** 发送。
   - 此信息是针对有关证书发放时间的多次询问而确认的。
- **OpenAI Credit 困惑**：一名成员报告称，尽管在 11/25 截止日期前正确提交了组织 ID，但仍未收到 **OpenAI credits**。
   - 社区建议检查账户余额，因为可能没有发送通知。
- **即将开始的 LLM Agents 课程详情**：计划于 1 月至 5 月进行的 **LLM Agents 课程**将作为秋季课程的续作。虽然之前的课程内容可能不是严格必需的，但建议复习 VODs。
   - 经过讨论确认，该课程承诺对 LLM agents 相关主题进行深入探索。
- **课程网站移动端响应式改进**：一名成员分享了 LLM Agents MOOC 网站的修改版本，解决了其在移动设备上缺乏响应式设计的问题。
   - 他们鼓励对更新提供反馈，并表达了为社区做出积极贡献的愿望。
- **允许重新提交书面作业**：成员们得到保证，提交逾期的**书面作业**是可以接受的，因为一位贡献者提到他们提交文章作业的时间晚于发布日期。
   - 这一回应反映了社区对参与课程材料的个人的支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llmagents-learning.org/sp25">Large Language Model Agents MOOC</a>: MOOC, Spring 2025</li><li><a href="https://gilbertomedrano.com/berkeley-ai-mooc-website/index.html">Large Language Model Agents MOOC</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-readings-discussion](https://discord.com/channels/1280234300012494859/1282735578886181036/1318109068287545375)** (1 messages): 

> `Safety alignment in AI Research Agents, AI Research Resources` 


- **安全对齐对 AI Research Agents 至关重要**：一名成员强调 **safety alignment** 是 **AI Research Agents** 的核心组成部分，并链接到了一个有用的资源 [AI Research](https://airesearch.js.org)。
   - *DM me to help!* 意味着公开征集在这一重要主题上的合作。
- **关于 AI 研究的 YouTube 视频**：一名成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=-r0XPC7TLzY)，但未提供有关其内容的描述或细节。
   - 缺乏上下文让观众对视频与讨论的相关性感到好奇。



**提到的链接**: <a href="https://www.youtube.com/watch?v=-r0XPC7TLzY"> - YouTube</a>: 未找到描述

  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1317229223357710409)** (6 messages): 

> `Torchtune v3.9 updates, Ruff automatic type hinting, Fine-tuning projects, Torcheval syncing metrics issues` 


- **Torchtune v3.9 简化了类型提示**：随着 **Torchtune v3.9** 的更新，用户现在可以使用默认的内置类型替换 `List`、`Dict` 和 `Tuple` 进行类型提示。
   - 这一变化被视为简化 Python 代码的受欢迎调整。
- **Ruff 有助于自动类型调整**：*Gau.nernst* 指出 **Ruff** 有一条规则可以自动替换类型提示默认值，从而减轻开发者的工作量。
   - 该工具解决了开发者在 Python 类型提示中遇到的一些常见困扰。
- **社区发起微调项目讨论**：成员们本周进行了交流，查看是否有人正在进行有趣的 **fine-tuning projects**。
   - 这突显了持续的社区协作和知识共享。
- **对 Torcheval 同步指标的担忧**：*Mirceamironenco* 提出了关于 **Torcheval** 在跨 world size 同步指标时发生挂起的担忧。
   - 这指向了未来更新中可能需要注意的潜在可用性问题。
- **PJ Bontrager 对 Torcheval 的生疏**：*PJ Bontrager* 提到他最近没有使用 **Torcheval**，表示对该项目当前状态的不确定。
   - 这强调了 AI 生态系统中工具的不断演进。

### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1317800872046104698)** (13 条消息🔥): 

> `DTensor 构建、FSDP 中的梯度归一化、Scalar 与 Scaler 的混淆` 


- **质疑 DTensor 构建方法**：关于 **DTensor** 构建的讨论，一名成员指出它很少被直接构建，建议使用 `.from_local` 作为首选 API。
   - 另一名成员确认 `from_local` 通常是安全的选择，并暗示该函数内部可能会调用 tensor 方法。
- **分布式训练中的梯度归一化问题**：有人对 backward 过程中的归一化缩放因子提出了疑问，建议应为 `world_size / num_tokens`，以适应不同 batch 之间 token 数量的变化。
   - 该成员说明了由于 padding 和索引差异，这些问题可能会使梯度计算变得复杂，并主张通过 PR 来解决这种不一致性。
- **澄清 Scalar 与 Scaler 术语**：一名成员幽默地指出了 **scalar**（数学术语，标量）和 **scaler**（电子计数器/缩放器）之间的混淆，表明社区中一直存在这种困惑。
   - 他们提供了定义进行澄清，暗示在不同项目中需要保持术语的一致性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://sapling.ai/mixup/scalar_scaler#:~:text=(adjective)%20of%20or%20relating%20to%20a%20directionless%20magnitude%20(such,rapidly%20to%20be%20recorded%20individually.">“Scalar” 还是 “Scaler”——该用哪一个？ | Sapling</a>：未找到描述</li><li><a href="https://github.com/pytorch/torchtune/blob/main/recipes/full_finetune_distributed.py#L780">torchtune/recipes/full_finetune_distributed.py at main · pytorch/torchtune</a>：PyTorch 原生微调库。欢迎通过 GitHub 贡献代码参与 torchtune 的开发。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L384">pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py at main · pytorch/pytorch</a>：Python 中具有强大 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/ao/blob/46b8796412eb350d1923091892850582d32737d0/torchao/prototype/low_bit_optim/adam.py#L72">ao/torchao/prototype/low_bit_optim/adam.py at 46b8796412eb350d1923091892850582d32737d0 · pytorch/ao</a>：用于训练和推理的 PyTorch 原生量化和稀疏化工具 - pytorch/ao</li><li><a href="https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py#L387.">pytorch/torch/distributed/fsdp/_fully_shard/_fsdp_collectives.py at main · pytorch/pytorch</a>：Python 中具有强大 GPU 加速能力的 Tensor 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/pytorch/torchtune/blob/c2c6f4a5236ba69a8c87dcb1f23ad65daf6e75de/torchtune/training/_distributed.py#L198">torchtune/torchtune/training/_distributed.py at c2c6f4a5236ba69a8c87dcb1f23ad65daf6e75de · pytorch/torchtune</a>：PyTorch 原生微调库。欢迎通过 GitHub 贡献代码参与 torchtune 的开发。
</li>
</ul>

</div>
  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1318346415809888367)** (3 条消息): 

> `Generative Verifiers, Scaling Test Time Compute, LLM Performance Enhancement` 


- **Generative Verifiers 提升 LLM 性能**：该论文提出使用 next-token prediction 目标来训练验证器，即 **Generative Verifiers (GenRM)**，将验证与解决方案生成无缝集成。
   - 这种方法可以更好地与 instruction tuning 结合，并支持 chain-of-thought 推理，利用**额外的 inference-time compute** 来提高验证结果。
- **讨论 Scaling Test Time Compute 策略**：Hugging Face 上的一篇有趣的博客文章强调了扩展大型模型 test-time compute 的策略，重点是在不损害结果的情况下进行性能优化。
   - 该文章概述了在保持模型输出完整性的同时提高计算效率的各种方法论。
- **将问题重构为 Search Challenges**：一条引发思考的评论强调，许多 AI 挑战可以被重构为 *search problems*，从而改变解决这些问题的方法。
   - 这种视角可能会通过将重点转向基于 search 的方法论，从而在解决复杂的 AI 任务中产生新颖的解决方案和技术。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/HuggingFaceH4/blogpost-scaling-test-time-compute">Scaling test-time compute - a Hugging Face Space by HuggingFaceH4</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2408.15240v1">Generative Verifiers: Reward Modeling as Next-Token Prediction</a>: 验证器或奖励模型通常用于增强大型语言模型 (LLMs) 的推理性能。一种常见的方法是 Best-of-N 方法，其中由 ... 生成的 N 个候选解决方案。
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1317232052751306803)** (15 条消息🔥): 

> `BEAM Configuration, New Gradient API, Kernel Search Experience, Tinygrad Porting Projects, Backend Support` 


- **关于 BEAM 设置的说明**：成员们讨论了用于 kernel search 的不同 **BEAM** 设置，指出 **BEAM=1** 表示 greedy search，其效果较差。
   - 建议从 **BEAM=2 或 3** 开始，以获得更好的性能平衡，正如 [文档](https://docs.tinygrad.org/env_vars/) 中所述。
- **引入新的 Gradient API**：George Hotz 分享了新的 gradient API 已经合并，允许简化梯度处理：`weight_grad, bias_grad = loss.gradient(weight, bias)`，无需 `zero_grad` 或 `loss.backward`。
   - 他指出该 API 与 PyTorch 和 JAX 等传统框架不同，可能会通过 `optim.step(loss)` 简化优化器步骤。
- **改进 Kernel Search 流程**：重点在于增强 **kernel search 体验**，这涉及编译时间和 kernel 执行时间的改进。
   - 成员们对任何可用的 benchmarks 表示关注，并建议从 **BEAM=2** 开始，特别是在使用 JIT 编译时。
- **将 Fish-Speech 移植到 Tinygrad**：一位成员宣布计划将 **fish-speech** 项目移植到 Tinygrad 用于教学目的，该项目以其最先进的开源 text-to-speech 能力而闻名。
   - 该项目托管在 [GitHub](https://github.com/fishaudio/fish-speech) 上，展示了增强 Tinygrad 功能的协作努力。
- **关于 Backend 支持的讨论**：成员们辩论了为 Tinygrad 同时支持 **x86 和 arm64 backends** 的必要性，权衡了它们对用户的潜在价值。
   - 成员们对维持性能表示担忧，并讨论了在现有资源限制下支持多种架构是否有利。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/__tinygrad__/status/1867745748118544411">来自 tiny corp (@__tinygrad__) 的推文</a>: 新的 gradient API 已合并：weight_grad, bias_grad = loss.gradient(weight, bias)。对于优化器（见 PR #8231），新 API 为：optim.step(loss)。你不需要 zero_grad 或 loss.backward。它是 n...</li><li><a href="https://github.com/fishaudio/fish-speech">GitHub - fishaudio/fish-speech: SOTA Open Source TTS</a>: SOTA 开源 TTS。通过在 GitHub 上创建账号为 fishaudio/fish-speech 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1318295880499204168)** (1 messages): 

> `ShapeTracker Explainer, tinygrad Tutorials` 


- **发布改进版的 ShapeTracker 讲解**：一份增强版的 **ShapeTracker** 讲解已撰写完成，可以在[这里](https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md)找到。
   - 这个新版本旨在澄清各个方面，并对 ShapeTracker 的工作原理提供更深入的见解。
- **征集 tinygrad 教程贡献**：GitHub 仓库 **tinygrad-notes** 鼓励开发者为 tinygrad 开发贡献教程和资源。
   - 可以访问该仓库获取更多材料，并有可能参与到项目中。



**Link mentioned**: <a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/20241217_st.md">tinygrad-notes/20241217_st.md at main · mesozoic-egg/tinygrad-notes</a>: Tutorials on tinygrad. Contribute to mesozoic-egg/tinygrad-notes development by creating an account on GitHub.

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1317260179733090324)** (3 messages): 

> `LlamaIndex tutorial, Agentic workflow for contract compliance, Agentic workflow for patient case summaries` 


- **5 行代码精通 LlamaIndex**：@TylerReedAI 分享了一个详细教程，介绍如何仅用 **5 行代码**构建基础的 RAG 应用，涵盖了数据加载和索引。更多见解请查看[教程](https://t.co/v5yljbVw4d)。
   - 该教程强调了在工作区中集成 **query** 和 **chat engines** 的便捷性。
- **轻松确保合同合规**：一个新教程介绍了一种构建 Agentic 工作流的方法，通过根据 **GDPR** 等指南分析相关条款来确保**合同合规**。深入了解详情请点击[这里](https://t.co/9SjfXRWdmF)。
   - 该教程分解了如何拆解供应商合同以有效维持合规性，使合同管理变得更简单。
- **简化患者病例摘要**：一个综合教程展示了如何创建一个解析**患者健康记录**的 Agentic 工作流，使用 LLM 驱动的提取技术。该工作流有助于分析指南建议并生成**清晰的病例摘要**，详见[这里](https://t.co/0s9xgoPpeE)。
   - 这种方法利用 RAG 增强了患者信息的清晰度，同时确保遵循医疗指南。


  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1317243621363155046)** (10 messages🔥): 

> `Creating Query Engine with Vector Store, Handling PDF Errors, Custom Extractors in LlamaIndex, Implementing Contextual Retrieval, NVIDIA NV-Embed-v2 Availability` 


- **使用现有 Vector Store 创建 Query Engine**：一位用户正在寻求指导，如何在已有 embedding 的现有 Vector Store 之上创建 Query Engine，而不使用 `VectorStoreIndex.from_documents(..)` 方法。
   - 他们提到了一种 Pipeline 配置，其中包括在存储文档之前对其进行处理的各种 Transformations。
- **PDF 错误：是我这边的问题吗？**：一位用户报告在使用 LlamaParse 时遇到错误消息“UNKNOWN_ERROR: PDF_IS_BROKEN”。
   - 另一位成员推测该 PDF 可能受密码保护，进一步讨论了错误的潜在原因。
- **在自定义 Extractor 中访问父文档**：一位正在开发自定义 Extractor 的用户表达了担忧，因为他们每次向索引添加文档时都需要手动设置父文档。
   - 他们询问是否有更符合惯例（idiomatic）的方法，因为 DocumentStore 仅提供对 Node 的访问，而不是原始 Document。
- **在 LlamaIndex 中集成 Contextual Retrieval**：一位用户在 LlamaIndex 中实现了 Anthropic 的 Contextual Retrieval，并分享了他们的 GitHub 仓库链接供他人审阅。
   - 他们表示有兴趣将此实现作为 PR 贡献，并强调了其鲁棒性和对边缘情况的处理。
- **关于 NVIDIA NV-Embed-v2 的咨询**：一位用户询问 NVIDIA 的 NV-Embed-v2 是否可以通过 NVIDIAEmbedding 使用。
   - 这引发了关于社区内特定 NVIDIA embedding 可用性的广泛讨论。



**Link mentioned**: <a href="https://github.com/cklapperich/Eidetic/">GitHub - cklapperich/Eidetic</a>: Contribute to cklapperich/Eidetic development by creating an account on GitHub.

  

---

### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1317269205032243321)** (1 messages): 

> `Langchain 集成, MegaParse 文档解析` 


- **将 Langchain 与 MegaParse 集成以实现高效解析**：讨论强调了将 **Langchain** 与 **MegaParse** 结合以增强文档解析能力的潜力，为各种文档类型提供高效工具。
   - *MegaParse* 被描述为一种多功能且开源的解决方案，旨在解析过程中保持数据完整性。
- **对文档解析解决方案的需求日益增长**：随着企业、研究人员和开发人员对强大工具的需求，有效的文档解析和信息提取变得至关重要。
   - 各机构正在积极寻求能够处理多种文档类型并确保数据忠实度的解决方案。



**提到的链接**：<a href="https://medium.com/ai-artistry/integrating-langchain-with-megaparse-unlocking-seamless-document-parsing-7a229a79b6ba">Integrating Langchain with MegaParse: Unlocking Seamless Document Parsing</a>：Ankush k Singal

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1317231750090326069)** (7 messages): 

> `文件夹创建问题, API 响应问题, Litellm 计费追踪, 日语学习应用, 本地使用 OS` 


- **注意到文件夹创建困难**：一位成员表示该工具*无法创建文件夹*，并提到生成的代码*缩进错误*，导致难以复制粘贴。
   - 他们询问是否应该在 cmd 以外的其他环境中运行。
- **API 达到免费 Token 限制**：另一位成员报告称，在 macOS **Monterey** 上下载该应用后，API 没有任何响应，且仅在**两次操作**后就达到了免费 Token 限制。
   - 这指向了该应用在该操作系统上潜在的集成或使用问题。
- **关于 Litellm 计费追踪的咨询**：一位用户询问是否有人将 OI 连接到 Litellm 代理服务器，以有效追踪计费和使用情况。
   - 他们询问了如何为集成的 Litellm 软件包启用计费追踪。
- **寻求日语学习应用**：一位成员询问是否有好的**日语**学习应用。
   - 另一位用户幽默地指出，他们可能进错了 *Discord 服务器*。
- **关于本地 OS 使用的问题**：一位用户询问是否有在本地使用 OS 的方法，表现出对本地设置的兴趣。
   - 这暗示了关于部署或本地托管解决方案的潜在讨论。


  

---


### **DSPy ▷ #[examples](https://discord.com/channels/1161519468141355160/1161519685616025600/1317245794319073291)** (5 messages): 

> `Claude Sonnet 提示词优化, DSPy 过时示例, 重构 VLM 示例` 


- **使用 DSPy 优化 Claude Sonnet 提示词**：一位用户在寻找优化 **Claude Sonnet** 提示词的方法时发现了 DSPy，并收藏了一个特定的 [Jupyter notebook](https://github.com/stanfordnlp/dspy/blob/main/examples/vlm/mmmu.ipynb)。
   - *他们提到该 notebook 最近被移到了过时示例文件夹中*，从而对其适用性产生了疑问。
- **建议谨慎对待过时示例**：另一位成员建议，在该文件夹的内容被重构之前应**谨慎使用**，表明它们可能不完全可靠。
   - *他们还指出，目前正在努力更新这些示例*，可能会提高其可用性。


  

---


### **DSPy ▷ #[colbert](https://discord.com/channels/1161519468141355160/1250300504462856265/)** (1 messages): 

nsa7211: <@1149658946982916167> ColPali 也能处理手写文档吗？
  

---

### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1317655486396633099)** (2 messages): 

> `APOLLO optimizer, LLM training memory efficiency, Multi-turn KTO` 


- **APOLLO 优化器展现出极高的显存效率**：全新的 **APOLLO 优化器**在 LLaMA 7B 训练期间实现了最佳的困惑度（perplexity），同时显著降低了显存消耗，仅需 **1.6G** 显存，而 8-bit Adam 则需要 **13G**。
   - 一个独立的 **Julia 实现** 验证了 APOLLO 的性能，确认了其在优化显存使用和训练效率方面的有效性 [查看帖子](https://bsky.app/profile/benjmurrell.bsky.social/post/3lcyfrf5b7k2u)。
- **LLM 训练中的挑战**：大型语言模型（LLMs）在使用 **AdamW 优化器**时面临严重的显存问题，通常需要昂贵的硬件或在训练期间减小 batch size。
   - 开发显存高效优化器的尝试通常涉及 SVD 操作或显著的性能权衡；然而，APOLLO 提出了一种创新方法来缓解这些挑战。
- **关于多轮 KTO 的讨论**：有人询问了**多轮 KTO** 的性能和状态，但目前尚未提供具体的细节或回复。
   - 成员们似乎对该方法在 LLM 背景下的能力和实现方式感到好奇。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2412.05270">APOLLO: SGD-like Memory, AdamW-level Performance</a>: 大型语言模型 (LLMs) 在训练期间以显存密集著称，尤其是在使用流行的 AdamW 优化器时。这种显存负担迫使研究者使用更多或更高端的 GPU，或者减少...</li><li><a href="https://zhuhanqing.github.io/APOLLO/">APOLLO: SGD-like Memory, AdamW-level Performance</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1317573115500564642)** (1 messages): 

> `Progressive Tokenization, Zero-tree Ordering, DWT Coefficients, VAE Embedding` 


- **渐进式分词 (Progressive Tokenization) 解析**：讨论集中在利用从 **VAE Embedding** 中提取的 **DWT 系数**的**零树排序 (zero-tree ordering)** 来实现**渐进式分词**。
   - 附带的视频演示了该技术的实际运行，展示了该过程的复杂性。
- **小波系数分析**：成员们研究了在所讨论的方法背景下，**level 5 wavelet** 变换如何影响分词的有效性。
   - 分析包括了实际应用以及对未来模型增强的影响，参考 [附带视频](https://cdn.discordapp.com/attachments/823813160075132991/1317573114854637680/level_5_wavelet_db5_clip_value_2.0_patch_size_1.mp4?ex=6761d015&is=67607e95&hm=8a77936be85424a1ccff2f733b8e69a5ce554860b92709f386eb634bd6d148d5&)。


  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1317547822073253899)** (1 messages): 

> `Byte Latent Transformer Patches, Large Concept Models, NLP advancements` 


- **Byte Latent Transformer Patches 性能优于 Token**：题为 [Byte Latent Transformer Patches: Scale Better than Tokens](https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/) 的论文讨论了 NLP 中的一种新方法，展示了 Byte Latent Transformer Patches 相比传统 Token 如何实现更好的扩展性。
   - 这一进展引发了关于在各种应用中增强语言建模有效性和效率的讨论。
- **在 NLP 中探索 Large Concept Models**：LCM 团队（包括 **Loic Barrault** 和 **Holger Schwenk** 等成员）正致力于通过基于句子表示空间的框架来理解语言建模。
   - 他们的研究旨在为如何在 NLP 模型中有效地构建和利用语言概念提供更深入的见解。



**提及的链接**: <a href="https://ai.meta.com/research/publications/byte-latent-transformer-patches-scale-better-than-tokens/">未找到标题</a>: 未找到描述

  

---

### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1318308461712379994)** (1 条消息): 

> `Retrieval Augmented Generation, Event Preparations, SQLite-Vec and LlamaFile, Python Development` 


- **12 月最后一场 RAG 应用活动**：明天的活动将重点介绍如何使用 **sqlite-vec** 和 **llamafile**，通过 **最基础的 Python (bare-bones Python)** 且无需任何额外依赖或安装，来创建一个 **超低依赖的检索增强生成 (RAG)** 应用。
   - 活动将由 **Alex Garcia** 主持，为参与者提供构建 RAG 应用的简明方法。
- **迎接假期休整**：本次活动是 12 月放假前的 **最后一次聚会**，强调了年底前参与的重要性。
   - 鼓励参与者 **加入会议**，作为假期的前奏，并深入了解 RAG 开发。


  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[leaderboard](https://discord.com/channels/1111172801899012102/1214705495974092810/)** (1 条消息): 

huanzhimao: 更新：它们在[这里](https://github.com/HuanzhiMao/BFCL-Result)。
  

---


---


---


{% else %}


> 为了方便邮件阅读，完整的频道明细已被截断。 
> 
> 如果您想查看完整明细，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}