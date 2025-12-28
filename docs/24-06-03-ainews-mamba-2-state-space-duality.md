---
companies:
- hugging-face
date: '2024-06-03T21:31:26.119127Z'
description: '**Mamba-2** 是一种新型的**状态空间模型 (SSM)**，在**困惑度（perplexity）**和**实际运行时间（wall-clock
  time）**方面均优于之前的 Mamba 和 Transformer++ 模型。其特点是**状态规模扩大了 8 倍**，且**训练速度提升了 50%**。它引入了**状态空间对偶性
  (SSD)** 的概念，将 SSM 与线性注意力机制联系起来。


  **FineWeb-Edu 数据集**是拥有 **15 万亿 token 的 FineWeb 数据集**的高质量子集，通过 **llama-3-70b** 针对教育质量进行了过滤，能够实现更好、更快的
  LLM 学习，并有望减少超越 **GPT-3** 性能所需的 token 数量。此外，使用 **1.25 亿参数模型**进行基于困惑度的数据剪枝，可以提高下游任务性能，并将预训练步数减少多达
  **1.45 倍**。**Video-MME 基准测试**则用于评估多模态大模型在不同视觉领域和视频长度下的视频分析能力。'
id: afc160a0-c657-47c5-bdaa-cf40080a004f
models:
- mamba-2
- mamba
- transformer++
- llama-3-70b
- gpt-3
original_slug: ainews-mamba-2-state-space-duality
people:
- _albertgu
- tri_dao
- arankomatsuzaki
- _akhaliq
- clementdelangue
- karpathy
title: Mamba-2：状态空间对偶性
topics:
- state-space-models
- perplexity
- training-efficiency
- data-pruning
- benchmarking
- multimodality
- video-analysis
---

<!-- buttondown-editor-mode: plaintext -->**Transformers 是 SSMs。**

> 2024/5/31-2024/6/3 的 AI 新闻。
我们为您查看了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord（**400** 个频道，**8575** 条消息）。
预计节省阅读时间（按 200wpm 计算）：**877 分钟**。

周末我们收到了 [FineWeb 技术报告](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)（我们 [一个月前报道过](https://buttondown.email/ainews/archive/ainews-fineweb-15t-tokens-of-commoncrawl/)），事实证明，通过更好的过滤和去重，它确实在 CommonCrawl 和 RefinedWeb 的基础上有所改进。

然而，我们将周末的胜利（W）归功于 Mamba 的共同作者们，他们带着 [Mamba-2 再次回归](https://arxiv.org/abs/2405.21060)。其核心仅有 [30 行 Pytorch 代码](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/#the-code)，在 perplexity 和 wall-clock time 上都优于 Mamba 和 Transformer++。

 
![image.png](https://assets.buttondown.email/images/d1f1d607-f9d9-444d-94b9-6153151110ea.png?w=960&fit=max)
 

Tri 建议[先阅读博客](https://goombalab.github.io/blog/2024/mamba2-part1-model/)，该博客分 4 个部分介绍了 Mamba-2 的开发：

1. **模型 (The Model)**
  - **理解**：SSMs 与 attention 之间有哪些概念上的联系？我们能将它们结合吗？
   > 正如我们在早期关于结构化 SSMs 的工作中所阐述的，它们似乎捕捉到了**连续、卷积和循环**序列模型的本质——全部包裹在一个简单而优雅的模型中。
  - **效率**：我们能否通过将 Mamba 模型重构为矩阵乘法来加速其训练？
  > 尽管在提高 Mamba 速度方面投入了大量工作，但它的硬件效率仍然远低于 attention 等机制。
  - Mamba 和 Mamba-2 之间的核心区别在于其 A 矩阵更严格的对角化： 
![image.png](https://assets.buttondown.email/images/7592290b-b389-4ce9-a8ee-3894b7775acd.png?w=960&fit=max)
 利用这个定义，作者证明了 Quadratic Mode (Attention) 和 Linear Mode (SSMs) 之间的等价性（对偶性），并解锁了矩阵乘法。
 
![image.png](https://assets.buttondown.email/images/65836a4e-08a1-424f-bdcb-7055c31ab50d.png?w=960&fit=max)
 
2. **理论 (The Theory)**
  - 
![image.png](https://assets.buttondown.email/images/e1d636ae-0b08-433e-abd0-68b986d3cf56.png?w=960&fit=max)
 
3. **算法 (The Algorithm)**
  - [https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/#the-code ](https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/#the-code )
4. **系统 (The Systems)**
  - 他们展示了 Mamba-2 在评估中不仅击败了 Mamba-1 和 Pythia，而且在放入类似于 Jamba 的混合模型架构时，在评估中占据主导地位：
 
![image.png](https://assets.buttondown.email/images/148218c4-34a2-4968-902d-7a11589e15d0.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})!

{% endif %}


---

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程。

**AI 与机器学习研究**

- **Mamba-2 状态空间模型**：[@_albertgu](https://twitter.com/_albertgu/status/1797651223035904355) 和 [@tri_dao](https://twitter.com/tri_dao/status/1797650443218436165) 推出了 Mamba-2，这是一种状态空间模型 (SSM)，在 **perplexity 和 wall-clock time 上均优于 Mamba 和 Transformer++**。它提出了一个连接 SSMs 和 linear attention 的框架，称为**状态空间对偶性 (SSD)**。Mamba-2 的**状态比 Mamba 大 8 倍，训练速度快 50%**。([@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1797443178099790324) 和 [@_akhaliq](https://twitter.com/_akhaliq/status/1797475092600873361))

- **FineWeb 和 FineWeb-Edu 数据集**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1797634822237728858) 强调了 **FineWeb-Edu 的发布，这是 15 万亿 token 的 FineWeb 数据集的高质量子集**，通过**使用 Llama 3 70B 模型过滤 FineWeb 以评判教育质量**而创建。它能够实现**更好、更快的 LLM 学习**。[@karpathy](https://twitter.com/karpathy/status/1797313173449764933) 指出它有潜力**减少超越 GPT-3 性能所需的 token 数量**。

- **基于 Perplexity 的数据剪枝**：[@_akhaliq](https://twitter.com/_akhaliq/status/1797475921642786851) 分享了一篇关于**使用小型参考模型进行基于 perplexity 的数据剪枝**的论文。基于 **125M 参数模型的 perplexity 进行剪枝提高了下游性能**，并**减少了高达 1.45 倍的预训练步骤**。

- **Video-MME 基准测试**：[@_akhaliq](https://twitter.com/_akhaliq/status/1797474099096150249) 介绍了 Video-MME，这是**首个评估多模态 LLM 视频分析能力的综合基准测试**，涵盖了 **6 个视觉领域、视频长度、多模态输入和人工标注**。**Gemini 1.5 Pro 的表现显著优于开源模型**。

**AI 伦理与社会影响**

- **AI 末日论与奇点论**：[@ylecun](https://twitter.com/ylecun/status/1797374483310739697) 和 [@fchollet](https://twitter.com/fchollet/status/1797378528574640510) 批评 **AI 末日论和奇点论是驱动疯狂信仰的“末世邪教”**，导致一些人**由于对 AI 的恐惧而停止了长期生活规划**。[@ylecun](https://twitter.com/ylecun/status/1797598791182258338) 认为这些论调**让人感到无力，而不是动员人们去寻找解决方案**。

- **对 Dr. Fauci 和科学的攻击**：[@ylecun](https://twitter.com/ylecun/status/1797676711175180449) 谴责**共和党国会议员对 Dr. Fauci 的攻击是“可耻且危险的”**。Fauci **帮助挽救了数百万人，却被那些将政治置于公共安全之上的人诋毁**。对**科学和科学方法的攻击是“极其危险的”**，并且通过破坏公众对公共卫生的信任，在疫情期间**导致了人员死亡**。

- **对 Elon Musk 的看法**：[@ylecun](https://twitter.com/ylecun/status/1797270661192155427) 分享了对 Musk 的看法，**喜欢他的汽车、火箭、太阳能/卫星以及在开源/专利方面的立场**，但**不同意他对科学家的态度、炒作/错误预测、政治观点以及阴谋论，认为这些“对民主、文明和人类福祉是危险的”**。他认为 Musk 在其社交平台上**“对内容审核的难度和必要性表现得过于天真”**。

**AI 应用与演示**

- **Dino Robotics 机器人厨师**：[@adcock_brett](https://twitter.com/adcock_brett/status/1797297988567449675) 分享了 **Dino Robotics 机器人厨师制作炸肉排和薯条**的视频，该机器人利用**物体定位和 3D 图像处理技术**，并经过**训练以识别各种厨房物品**。

- **SignLLM**：[@adcock_brett](https://twitter.com/adcock_brett/status/1797298052526445008) 报道了 SignLLM，这是**首个用于手语生成的语言多语种 AI 模型**，能够根据八种语言的自然语言生成 **AI 虚拟人手语视频**。

- **Perplexity Pages**：[@adcock_brett](https://twitter.com/adcock_brett/status/1797298142305452281) 重点介绍了 Perplexity 的 Pages 工具，用于**将研究内容转化为可以在 Google Search 中排名的文章、报告和指南**。

- **1X 人形机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1797298097225179255) 演示了 **1X 的 EVE 人形机器人执行拾取衬衫和杯子等链式任务**，并提到了内部更新。

- **Higgsfield NOVA-1**：[@adcock_brett](https://twitter.com/adcock_brett/status/1797298164753469522) 介绍了 Higgsfield 的 NOVA-1 AI 视频模型，允许**企业使用其品牌资产训练自定义版本**。

**杂项**

- **交友建议**：[@jxnlco](https://twitter.com/jxnlco/status/1797270299903136030) 分享了建立社交网络的技巧，如**运动、创意表达、烹饪团体餐以及基于共同兴趣联系他人**。

- **笔记本电脑推荐**：[@svpino](https://twitter.com/svpino/status/1797606675580670038) 赞扬了**“完美”但昂贵的 Apple M3 Max，配备 128GB RAM 和 8TB SSD**。

- **Nvidia 主旨演讲**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797254706776957094) 指出 **Nvidia 在 8 年间将数据中心 AI 计算成本降低了 350 倍**。[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1797289862623547452) 强调了**在集成 RAPIDS cuDF 后，Google Colab 上的 Pandas 速度提升了 50 倍**。

- **Python 赞誉**：[@svpino](https://twitter.com/svpino/status/1797244274045862378) 称 **Python 是“有史以来无可争议的编程语言 GOAT（史上最伟大）”**，并且 [@svpino](https://twitter.com/svpino/status/1797282169426989479) 建议**教孩子学习 Python**。

**幽默与迷因**

- **Elon Musk 玩笑**：[@ylecun](https://twitter.com/ylecun/status/1797353438507884549) 向 [@elonmusk](https://twitter.com/elonmusk) 开玩笑说 “Elno Muks” 声称正在“给他发垃圾（sh$t）”。

- **获胜迷因**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1797479224326185165) 发布了一张“一直赢是什么样子的？”迷因图片。

- **陶器玩笑**：[@jxnlco](https://twitter.com/jxnlco/status/1797396852196688104) 开玩笑说：“烹饪证明。是的，我在一个复古围棋盘上吃饭。”

- **Stable Diffusion 3 迷因**：[@Teknium1](https://twitter.com/Teknium1/status/1797467900993036602) 批评 Stability AI “编造了一个没人听说过的名为 SD3 'Medium' 的新版本”，同时却不发布 Large 和 X-Large 版本。

- **Llama-3-V 争议梗图**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1797438010163867933) 发布了关于 Llama-3-V 的 GitHub 和 HF 页面在“被爆出抄袭 @OpenBMB 模型证据后”下线的消息。

---

# AI Reddit 综述

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**AI 模型发布与更新**

- **Stability AI 将于 6 月 12 日发布 SD3 Medium 开放权重**：在 /r/StableDiffusion 中，Stability AI 宣布将发布 [**2B 参数的 SD3 Medium 模型**](https://www.reddit.com/r/StableDiffusion/comments/1d6szfg/sd3_medium_release_on_june_12th/)，该模型专为写实摄影、排版、性能优化以及非商业用途下的微调而设计。[2B 模型目前在某些指标上优于 8B 版本](https://www.reddit.com/r/StableDiffusion/comments/1d6ya9w/collection_of_questions_and_answers_about_sd3_and/)，且与某些采样方法不兼容。商业权利将通过会员计划提供。

- **Nvidia 和 AMD 揭晓未来 AI 芯片路线图**：在 /r/singularity 中，[**Nvidia 透露了截至 2027 年的计划**](https://www.reddit.com/r/singularity/comments/1d6j9bb/nvidia_unveils_its_future_chip_rollout_plans_till/)，其 Rubin 平台将接替 Blackwell，继 OpenAI 模型中使用的 H100 之后，H200 和 B100 芯片也即将推出。[AMD 宣布推出拥有 288GB 显存的 MI325X](https://www.msn.com/en-us/lifestyle/shopping/amd-reveals-the-mi325x-a-288gb-ai-accelerator-built-to-battle-nvidia-s-h200/ar-BB1nvzxK)，旨在与 Nvidia 的 H200 竞争，MI350 和 MI400 系列将在未来几年提供重大的推理提升。

**AI 能力与局限性**

- **AI 生成媒体误导主流新闻**：一段被 NBC News 误认为展示真实舞蹈效果的 [视频](https://v.redd.it/076ijie3ca4d1) 证明了 AI 生成内容甚至能愚弄大型媒体机构。

- **真正开源 AI 的挑战**：一段 [视频](https://v.redd.it/q6qhv6rf1a4d1) 指出，开源 AI 并非真正的开源，因为如果没有训练数据、顺序和技术，模型权重是难以捉摸的。由于依赖授权数据，完全开源 LLM 具有很大难度。

- **多模态推理的局限性**：在 /r/OpenAI 中，ChatGPT [在图像中标记物体时遇到困难](https://www.reddit.com/r/OpenAI/comments/1d6hmsa/why_can_chatgpt_identify_an_object_in_an_image/)，尽管它能正确识别该物体，这突显了目前 AI 在跨模态推理能力上的差距。

**AI 开发工具与技术**

- **高质量 Web 数据集在知识和推理方面表现出色**：拥有 1.3T token 的 [FineWeb 数据集](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1) 在知识和推理基准测试中超越了其他开放的 Web 规模数据集。相关的博客文章详细介绍了从 Web 数据创建高质量数据集的技术。

- **书中介绍的新机器学习数学工具**：书籍《Tangles》[应用了一种新颖的数学方法](https://www.reddit.com/r/MachineLearning/comments/1d6cq0n/research_tangles_a_new_mathematical_ml_tool_in/) 来对特征进行分组，以识别数据中的结构和类型，应用范围涵盖从聚类到药物研发。提供开源代码。

- **LLM 的参数化压缩**：在 /r/LocalLLaMA 中，一种 [简单的参数化压缩方法](https://www.reddit.com/r/LocalLLaMA/comments/1d6t94f/simple_parametric_compression_of_llama370b/) 在不进行微调的情况下，将 LLaMA 3 70B 的不重要层剪枝至 62B 参数，在基准测试中仅出现轻微的性能下降。

**AI 伦理与社会影响**

- **披露 AI 对就业影响的伦理困境**：/r/singularity 讨论了 [是否应该告知朋友](https://www.reddit.com/r/singularity/comments/1d6ihyp/should_you_tell_friends_their_careers_are_over_if/) AI 现在可以在几秒钟内完成他们的工作（如书籍封面设计）的困境。传达此类消息的痛苦与隐瞒真相之间的权衡。

- **民意调查衡量对 AI 威胁就业安全的看法**：/r/singularity 的一项 [投票](https://www.reddit.com/r/singularity/comments/1d6powg/job_security_poll_will_your_job_be_replaced_by_ai/) 衡量了在 AI 自动化的背景下，人们对自己工作在未来 10 年内能否持续的信心。

**梗图与幽默**

- **梗图讽刺 AI 广泛的职业替代潜力**：一张 [“所有工作”的梗图](https://i.redd.it/eoomwt7be74d1.jpeg) 幽默地描绘了 AI 替代广泛职业的能力。

---

# AI Discord 摘要回顾

> 摘要之摘要的摘要

1. **LLM 进展与多模态应用**：

   - 来自 IBM 的 **[Granite-8B-Code-Instruct](https://huggingface.co/ibm-granite/granite-8b-code-instruct)** 增强了代码任务的指令遵循（instruction-following）能力，超越了主要基准测试。**[Stable Diffusion 3](https://stability.ai/news/stable-diffusion-3)** Medium 版本即将发布，承诺提供更好的写实感和排版能力，计划于 6 月 12 日推出。

   - AI 工程社区讨论了 **SD3 的 VRAM 需求**，预测在考虑 **fp16 优化**等潜在缩减方案的情况下，需求约为 **15GB**。**[FlashAvatar](https://simongiebenhain.github.io/NPGA/)** 承诺使用 Nvidia RTX 3090 实现 300FPS 的数字分身，引发了对高保真头像创建的关注。

2. **微调技术与挑战**：

   - 针对**克服半精度训练中 tokenizer 问题**的建议包括设置 `tokenizer.padding_side = 'right'`，并使用 **[LoRA](https://arxiv.org/abs/2405.09673)** 技术来增强微调效果。**Axolotl** 用户在处理**二分类**任务时遇到问题，建议使用 **Bert** 作为替代方案。

   - 社区见解强调了使用 **Gradio 的 OAuth 进行私有应用访问**的有效性，以及 `share=True` 对于快速应用测试的实用性。故障排除包括处理 Kaggle 中的**推理设置**问题，以及 Axolotl 中损失值（loss values）的差异，并考虑了**输入输出预处理**等因素。

3. **开源项目与社区协作**：

   - **[Manifold Research](https://www.manifoldrg.com/opportunities/)** 征集多模态 Transformer 和控制任务方面的合作，旨在构建一个全面的开源通用模型（Generalist Model）。**[StoryDiffusion](https://github.com/HVision-NKU/StoryDiffusion/tree/main?tab=readme-ov-file)** 和 **[OpenDevin](https://lu.ma/fp0xr460)** 作为新的开源 AI 项目涌现，引发了广泛关注。

   - 将 **TorchAO 与 LM Evaluation Harness** 集成的努力集中在添加**量化支持**的 API 上。**[社区倡议](https://github.com/triton-lang/triton/issues/1211)**（如使 Axolotl 适配 AMD 兼容性）突显了在完善 AI 工具和框架方面的持续努力。

4. **AI 基础设施与安全**：

   - **[Hugging Face 安全事件](https://huggingface.co/blog/space-secrets-disclosure)** 促使官方建议轮换 token 并切换到细粒度访问 token，这影响了用户在 HF Spaces 等处的基础设施。**OpenRouter** 的讨论提到了**亚洲地区的数据库超时**问题，导致了服务更新并停用了某些模型，如 **Llava 13B 和 Hermes 2 Vision 7B**。

   - **[ZeRO++](https://www.deepspeed.ai/tutorials/zeropp/)** 框架在大型模型训练中显著降低了通信开销，助力 LLM 的实现。**Paddler 有状态负载均衡器**提升了 llama.cpp 的效率，有望优化模型推理服务能力。

5. **AI 研究与伦理讨论**：

   - **Yudkowsky 针对 AI 发展的争议性策略**引发辩论，其中包括对数据中心进行空袭等激进措施。**[LAION](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA)** 社区对此做出反应，讨论了开放协作与防止滥用之间的平衡。

   - **[关于 Transformer 局限性的新理论](https://arxiv.org/abs/2402.08164)**：实证证据表明 Transformer 在大定义域上的函数组合方面存在困难，这催生了模型设计的新方法。**关于 Embedding 效率的讨论**仍在继续，对比了不同 LLM 实现中上下文窗口（context windows）对性能的影响。

---

# 第一部分：Discord 高层级摘要

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HF Spaces 的安全漏洞**：在 [HF Spaces](https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets) 发生安全事件后，建议用户**轮换所有 token 或密钥**，详情见 HuggingFace 的 [博客文章](https://huggingface.co/blog/space-secrets-disclosure)。

- **AI 与伦理辩论升温**：关于实验室培养神经元分类的辩论引发了对人工智能本质和伦理的深入讨论。同时，HuggingFace 基础设施问题导致了 "MaxRetryError" 问题的解决。

- **Rust 兴起**：一名成员合作在 Rust 中实现深度学习书籍 (d2l.ai)，并贡献至 [GitHub](https://github.com/asukaminato0721/d2l.ai-rs)，而其他人则讨论了 Rust 的 Candle 库在**效率和部署方面的优势**。

- **文献综述见解与奇特创作**：[Medium](https://medium.com/me/stats/post/dbd9fa3fc1a0) 上总结了一篇 LLM 推理文献综述，此外还分享了 **Fast Mobius demo** 和 **gary4live** Max4Live 设备等创意项目，体现了工程严谨性与想象力的融合。

- **实际应用与社区对话**：分享了使用 **TrOCR** 和 [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) 等模型进行 OCR 任务的实用指南。讨论还延伸到 LLM 确定性以及增强语言生成和**翻译任务**的资源建议，特别引用了 **[Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)** 作为强大的日译英工具。

- **机器人与 Gradio 的令人兴奋的进展**：文章《[使用 Lerobot 深入研究 Diffusion Policy](https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/)》展示了机器人领域的 **ACT 和 Diffusion Policy** 方法，同时 **Gradio 宣布支持动态布局**，使用 **@gr.render**，例如在 [Render Decorator 指南](https://www.gradio.app/guides/dynamic-apps-with-render-decorator) 中探索的 Todo List 和 AudioMixer 等多功能应用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **多 GPU 微调进展**：
**多 GPU 微调**正在积极开发中，并讨论了多模态扩展的可行性。分享了对 [LoRA 的详细分析](https://arxiv.org/abs/2405.09673)，强调了其在特定微调场景中的潜力。

- **训练挑战的技术解决方案**：
建议通过设置 `tokenizer.padding_side = 'right'` 来缓解半精度训练中的 tokenizer 问题，并就使用 **Kaggle Notebooks** 加速 LLM 微调提供了见解。

- **AI 模型实现故障排除**：
用户在 **GTX 3090 上运行 Phi 3 模型**以及在 **H100 NVL 上进行 RoPE** 优化时遇到困难。社区推荐的修复方案包括 Unsloth 的更新以及对潜在内存报告 Bug 的讨论。

- **模型安全与局限性备受关注**：
关于企业因安全顾虑（重点是防止有害内容生成）而犹豫是否使用开源 AI 模型的辩论浮出水面。此外，大家公认 **LLM** 存在无法在其训练数据之外进行创新的固有局限性。

- **AI 协作工具的持续改进**：
社区分享了在 Kaggle 等平台上保存模型和修复安装问题的解决方案。此外，在 **Hugging Face 和 Wandb** 等平台的微调 Checkpoint 管理优化方面也存在积极协作。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD3 Medium 发布倒计时**：Stability.ai 宣布 **Stable Diffusion 3 Medium** 定于 **6月12日** 发布；感兴趣的用户可以加入等候名单以获取早期访问权限。在台北国际电脑展（Computex Taipei）上的[公告](https://youtube.com/watch?v=MCi8jgALPYA)强调了该模型在照片级写实感（photorealism）和排版（typography）方面预期的性能提升。

- **关于 SD3 规格的推测**：AI 工程社区正热烈讨论 **Stable Diffusion 3** 可能的 VRAM 需求，预测约为 15GB，同时也有建议提到通过 fp16 优化可能降低这一数字。

- **商业用途需明确说明**：用户强烈要求 **Stability AI** 明确说明 SD3 Medium 商业使用的许可条款，担忧主要源于向带有非商业限制的许可证转型。

- **货币化举措遭遇抵制**：免费的 Stable AI Discord 机器人被付费服务 Artisan 取代，引发了社区的不满，凸显了 AI 工具访问趋向货币化的广泛趋势。

- **为优化和微调做好准备**：在 SD3 Medium 发布前夕，工程师们正期待社区微调（fine-tunes）流程以及在不同 GPU 上的性能基准测试，Stability AI 确保支持 1024x1024 分辨率优化，包括分块（tiling）技术。



---



## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **AI 辅助作业：机遇还是阻碍？**：工程师们对 AI 辅助作业的伦理问题分享了不同观点，将其比作在“糖果和羽衣甘蓝”之间做选择，并建议重点教导孩子负责任地使用 AI。

- **引导 Perplexity Pages 的潜力**：用户表示需要增强 Perplexity 的 Pages 功能，如导出功能和可编辑标题，以提高可用性，同时对某些模型（如 Opus）的自动选择和配额耗尽表示担忧。

- **增强交互的扩展程序**：旨在改进 Perplexity UI 的 **Complexity** 浏览器扩展程序的发布引起了社区关注，并邀请 Beta 测试人员来提升用户体验。

- **测试 AI 敏感性**：讨论强调了 Perplexity AI 处理敏感话题的能力，通过在创建以巴冲突等话题页面时的表现证明了这一点，其令人满意的结果增强了人们对其合规性过滤器的信心。

- **专家级应用的 API 探索**：AI 工程师讨论了在 Perplexity API 中针对不同任务的最佳模型使用方案，阐明了小型快速模型与大型准确模型之间的权衡，同时也对潜在的 TTS API 功能表示热衷。参考了 [model cards](https://docs.perplexity.ai/docs/model-cards) 以获取指导。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**推测性解码闲谈**：工程师们分享了关于 **Speculative Decoding** 的见解，并提出了添加 *Gumbel Noise* 和确定性 *Argmax* 的建议。该主题的录制课程预计在编辑后上传。讨论强调了消融研究（Ablation Studies）在理解采样参数对接受率（Acceptance Rates）影响方面的重要性。

**CUDA 迈向云端**：讨论了租用 H100 GPU 进行性能分析（Profiling）的事宜，推荐了 [cloud-gpus.com](https://cloud-gpus.com) 和 [RunPod](https://www.runpod.io/gpu-instance/pricing) 等供应商。同时，也指出了在不进行大规模 Hack 的情况下收集 Profiling 信息的挑战。

**工作与娱乐**：宣布成立了一个 *Production Kernels* 工作组和另一个 *PyTorch 性能相关文档* 工作组，并邀请大家协作。此外，还给初学者提供了一个小贴士：避免在社区中过度使用 @everyone，以防止不必要的通知骚扰。

**技术演讲预告**：即将举行的演讲和研讨会包括关于 Tensor Cores 和高性能 Scan 算法的会议。社区还期待邀请 Wen-mei Hwu 教授进行公开问答，以及来自 AMD Composable Kernel 团队的分享。

**数据深度探索与开发讨论**：[#llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1246181209491505202) 频道的讨论内容非常丰富，包括成功向 Hugging Face 上传 200GB 数据集的细节、LayerNorm 计算优化的提案，以及为了面向未来和更易于集成模型架构而进行的重大代码库重构。

**精度与量化**：介绍了 [AutoFP8 GitHub 仓库](https://github.com/neuralmagic/AutoFP8)，旨在自动转换为 FP8 以提高计算效率。同时，讨论了将 **TorchAO** 与 **LM Evaluation Harness** 集成的方案，包括用于改进量化支持的 API 增强。

**职场动态解析**：Anyscale 正在寻找对 **Speculative Decoding** 和系统性能感兴趣的候选人。同时，强调了 Chunked Prefill 和 Continuous Batching 实践在预测运营效率方面的重要性。

**知识传播**：关于 Scan 算法和 Speculative Decoding 的演讲录像将在 [CUDA MODE YouTube 频道](https://youtube.com/@cudamode?feature=shared)上发布，为高性能计算的持续学习提供资源。

**PyTorch 性能解析**：号召在即将到来的六月文档马拉松（Docathon）期间改进 **PyTorch 的性能文档**，重点关注当前实践而非 TorchScript 等已弃用的概念，并推动理清自定义 Kernel 的集成。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **VRAM 征服者**：工程师们正在讨论针对高 Token Prompt 导致低 VRAM 系统响应缓慢的解决方案，并为家庭 AI 装备推荐了如 Nvidia P40 卡等实用的模型硬件。
  
- **Codestral 夺得编程桂冠**：Codestral 22B 在上下文和指令处理方面的卓越表现引发了讨论。同时，解决了 LM Studio 中 Embedding 模型列表的问题，并交流了使用不同模型处理文本生成的经验。

- **缺乏支持的 Whisper**：尽管用户强烈要求在 LM Studio 中加入 Whisper 和 Tortoise 增强的音频功能，但关于体积与复杂度的权衡引发了讨论。此外，还揭露了当前版本中的一个 "Stop String" Bug。

- **配置难题**：出现了关于从编码到推理等应用的模型配置设置查询，重点关注量化权衡以及在特定 GPU 硬件上推理速度的神秘体验。

- **增强融合探险**：成员们思考了如何创建 Visual Studio 插件以实现更智能的编码辅助，借鉴了现有辅助工具的经验，并探讨了使用 Mentat 等模型实现全项目上下文理解的潜力。

*注：模型、讨论和 GitHub 仓库的具体链接已在相应频道中提供，可查阅以获取更多技术细节和背景。*

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **在 Token 世界中蓬勃发展**：新发布的 [FineWeb-Edu](https://hf.co/datasets/HuggingFaceFW/fineweb-edu) 数据集拥有 1.3 万亿个 Token，据称在 MMLU 和 ARC 等基准测试中表现卓越，详细的技术报告可在此处 [访问](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)。
  
- **电影魔术师资源大爆发**：一个包含 3000 个 [剧本的数据集](https://huggingface.co/datasets/nothingiisreal/screenplays-3k) 现已面向 AI 爱好者开放，其中包含从 PDF 转换而来的 .txt 格式剧本，并采用 AGPL-3.0 许可证，供模型训练爱好者使用。

- **虚拟舞台指令**：使用 **Worldsim** 进行的策略模拟正围绕乌克兰-俄罗斯冲突展开，展示了其构建详细场景的能力，尽管目前存在一个导致 **文本重复** 的技术故障，[正在审查中](https://github.com/essserrr/crt-terminal)。

- **蒸馏困境与线程讨论**：研究人员正在交流如何有效地将知识从大模型（如 Llama70b）蒸馏到小模型（如 Llama8b），并建议在管理 AI Agent 任务时使用线程（threading）而非循环（loops）。

- **模型伦理备受关注**：社区对 MiniCPM-Llama3-V 涉嫌抄袭 OpenBMB 的 MiniCPM 展开了激烈辩论，在集体关注和证据曝光后，争议模型已从 GitHub 和 Hugging Face 等平台移除。



---



## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **Axolotl 的挑战**：工程师报告了在 Axolotl 的 .yaml 文件中配置二分类（binary classification）时遇到的问题，收到了 `ValueError`，提示没有对应的 'train' 指令数据。建议的替代方案是部署 Bert 进行分类任务，或者在 Axolotl 缺乏支持时直接使用 TRL。

- **Gradio 的实用性备受赞誉**：AI 开发者利用 Gradio 的 `share=True` 参数快速测试和分享应用。讨论还涉及使用 OAuth 进行私有应用访问以及整体分享策略，包括在 [HF Spaces](https://www.gradio.app/guides/sharing-your-app) 上托管以及处理身份验证和安全性。

- **Modal 之谜与 GitHub 忧虑**：由于 [近期安全事件](https://huggingface.co/blog/space-secrets-disclosure) 导致 Modal 脚本中缺乏 Hugging Face 的身份验证，用户在下载 **Mistral7B_v0.1** 等模型时遇到错误。其他挑战还包括 Accelerate 中的 `device map = meta`，一位用户分享了其在推理机制中的效用。

- **积分结算的关键时刻**：截止日期驱动的讨论占据了频道，许多成员担心各平台积分的及时分配。Dan 和 Hamel 介入并进行了说明和安抚，强调了准确填写表格以避免错过平台特定积分的重要性。

- **面向未来的微调**：提出了 LLM 训练和微调的可能调整及各种策略，例如将 batch size 保持为 2 的幂次方，使用梯度累积步数（gradient accumulation steps）优化训练，以及大 batch size 在以太网分布式设置中稳定训练的潜力。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **零交叉与 SGD：持续的争议**：关于在优化器改进中**追踪梯度中的零交叉（zero crossings）**的优缺点的讨论仍在继续，在应用中观察到的结果褒贬不一。另一个热议话题是 **SGD 作为基准（baseline）**与新优化器进行比较的作用，这表明进展可能取决于学习率（learning rate）的改进。

- **FlashAvatar 引起关注**：一种名为 **FlashAvatar** 的用于创建高保真数字分身的方法引起了特别关注，据 [FlashAvatar 项目](https://simongiebenhain.github.io/NPGA/)详情介绍，该方法在 **Nvidia RTX 3090** 上可实现高达 300FPS 的渲染速度。

- **了解 GPT-4 的怪癖**：社区讨论集中在 **GPT-4 的内存泄漏和行为**上，讨论了“白屏”错误和可能与温度（temperature）设置相关的重复输出实例。还讨论了自定义 GPT 的使用和 API 限制，强调了根据 [OpenAI 帮助文章](https://help.openai.com/en/articles/8843948-knowledge-in-gpts)规定的**每文件 512 MB 和每文件 500 万个 token** 的限制。

- **上下文窗口与 Embedding 有效性的辩论**：一场激烈的辩论聚焦于 **Embedding 与扩展上下文窗口（context windows）**在提升性能方面的有效性。人们还考虑了将 **Gemini** 纳入流水线的前景，据称这能增强 GPT 的性能。

- **Prompt Engineering 中的困扰**：社区成员分享了 **ChatGPT 遵循提示词指南**方面的挑战，并寻求改进策略。观察发现，在构建复杂提示词时，倾向于使用**单个系统消息（system message）**。



---



## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo Server 稳定性风波**：用户报告 **Mojo 语言服务器**在 M1 和 M2 MacBook 上的 VS Code 衍生版本（如 Cursor）中崩溃，详情记录在 [GitHub issue #2446](https://github.com/modularml/mojo/issues/2446) 中。修复方案已存在于 nightly 构建版本中，且有一个 [YouTube 教程](https://youtu.be/OiMZtjSZVOw?si=JrgOG_UL662xZ48W)涵盖了可以加速代码循环的 Python 优化技术，建议那些寻求提升 Python 性能的用户参考。

- **密切关注 Mojo 的演进**：围绕 Mojo 成熟度的讨论集中在其开发进度和开源社区贡献上，如 [Mojo 路线图](https://docs.modular.com/mojo/roadmap#mojo-sdk-known-issues)和相应的[博客公告](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)所述。其他讨论还包括 Mojo 在数据处理和网络方面的潜力，利用了 DPDK 和 liburing 等框架。

- **Mojo 与 MAX 编排前向与反向传播**：在 **Max 引擎**中，成员们正在剖析实现前向传播（forward pass）以保留反向传播（backward pass）所需输出的细节，并对缺乏反向计算文档表示担忧。社区对 Mojo 中的条件一致性（conditional conformance）功能感到兴奋，这将增强标准库的功能。

- **Nightly 更新亮点**：nightly Mojo 编译器 ([2024.6.305](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)) 的持续更新引入了新功能，例如全局 `UnsafePointer` 函数变为方法。关于 C 语言中 `char` 类型的讨论最终确认了其实现定义（implementation-defined）的性质。同时，有人提出了改进变更日志（changelog）一致性的建议，指向了一个[风格指南建议](https://github.com/modularml/mojo/issues/2923)，并讨论了将 Tensor 移出标准库的过渡。

- **性能追求者**：性能爱好者正在进行数据处理时间基准测试，发现 Mojo 的速度超过了 Python，但落后于编译型语言，相关讨论记录在 [PR#514](https://github.com/jinyus/related_post_gen/pull/514) 草案中。这一发现引发了关于自定义 JSON 解析的提案，灵感来自 C# 和 Swift 的实现。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **BERT 不适合 lm-eval 任务**：BERT 在进行 **lm-eval** 测试时表现不佳，因为像 BERT 这样的 encoder 模型并非为生成式文本任务而设计。目前正在寻找 Hugging Face 上最小的 decoder 模型，以便进行能耗分析。

- **Llama-3-8b 性能中无法解释的差异**：一位用户报告了 llama-3-8b 在 gsm8k 分数上的不一致性，其测得的 62.4 分与官方公布的 79.6 分之间存在显著差距。有人建议旧的 commit 可能是罪魁祸首，检查 commit hash 可能会澄清问题。

- **Few-shot 设置导致结果差异巨大**：gsm8k 分数的差异可能进一步归因于排行榜上使用的 'fewshot=5' 配置，这可能与其他人的实验设置有所偏离。

- **协作与讨论激发创新**：提到了 Manifold Research 征集 multimodal transformers 和控制任务的合作者，并分享了对标准 RLHF 偏差的见解。讨论还深入探讨了 transformer 的局限性，并参与了数据依赖型 positional embeddings 的挑战。

- **破解黑盒**：人们对即将于 7 月举行的 mechanistic interpretability 黑客松表现出浓厚兴趣，邀请大家利用周末时间剖析神经网络。分享了一篇关于 backward chaining circuits 的论文摘要，旨在吸引更多人才参与合作。

- **视觉与多模态可解释性成为焦点**：AI Alignment Forum 的文章阐明了视觉和多模态 mechanistic interpretability 的基础建设，强调了涌现的分割图（segmentation maps）和 "dogit lens"。然而，也有观点表示需要对 score models 本身的电路进行进一步研究，并指出目前文献中存在空白。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**亚洲地区的数据库困扰**：OpenRouter 用户报告了在亚洲地区的**数据库超时**问题，主要集中在**首尔**、**孟买**、**东京**和**新加坡**。官方已实施修复，其中包括回滚部分延迟优化以解决此问题。

**OpenRouter 因 API 故障受到指责**：尽管进行了补丁修复，用户仍持续遇到 504 Gateway 错误，部分用户通过使用欧洲 VPN 暂时绕过了该问题。用户建议增加**特定供应商的运行时间统计（uptime statistics）**，以提高服务问责制。

**模型退役与建议**：由于使用率低且成本高，OpenRouter 正在退役 **Llava 13B** 和 **Hermes 2 Vision 7B (alpha)** 等模型，并建议切换到 [FireLlava 13B](https://openrouter.ai/models/fireworks/firellava-13b) 和 [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b) 等替代方案。

**无缝 API 切换**：正如在 [Playground](https://openrouter.ai/playground) 中所见，**OpenRouter 的标准化 API** 简化了在不同模型或供应商之间的切换，无需修改代码，这为工程师提供了更简便的管理方式。

**流行度优于基准测试**：OpenRouter 倾向于**基于实际应用对语言模型进行排名**，详细列出模型使用情况而非传统的基准测试，以提供务实的视角，具体可见 [OpenRouter Rankings](https://openrouter.ai/rankings)。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 伦理风口浪尖**：围绕 Eliezer Yudkowsky 限制 AI 发展的激进策略展开了激烈的辩论和愤怒，其中要求采取包括摧毁数据中心在内的激进行动引发了分歧对话。点击[此处](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA)深入了解这一争议。

- **Mobius 模型大显身手**：新款 **Mobius 模型**以“灭霸闻着一朵黄色小玫瑰”等提示词迷倒了社区，展示了该模型的灵气和多功能性。可以在 [Hugging Face](https://huggingface.co/Corcelio/mobius) 上寻找灵感或一睹这些有趣的生成结果。

- **法律乱象消耗资源**：一场讨论详细说明了伪法律诉讼如何浪费精力和资金，温哥华的一起伪法律索赔案例成为了前车之鉴。点击[此处](https://www.cbc.ca/news/canada/british-columbia/bc-lawyer-pseudolegal-lawsuit-1.7025394)查看这些荒谬行为的详情。

- **医疗 AI 挑战赛招募**：Alliance AI4Health 医疗创新挑战赛呼吁创新者走在前沿，提供 5000 美元奖金以激励医疗 AI 解决方案的开发。未来的医疗先锋可以在[此处](https://amic.devpost.com/)找到起跑点。

- **研究揭示 AI 新见解**：**Phased Consistency Model (PCM)** 的发布挑战了 **LCM** 的设计局限，详情见[此处](https://g-u-n.github.io/projects/pcm/)；同时，一篇新论文阐述了文本生成图像模型的效率飞跃，被称为“应用于图像生成的 1.58 bits 论文”，可在 [arXiv](https://arxiv.org/abs/2405.14854) 上探索。SSMs 在速度领域发起反击，**Mamba-2** 超越了前代产品并可与 Transformers 媲美，点击[此处](https://arxiv.org/abs/2405.21060)阅读全文。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 中图与文档的结合**：LlamaIndex 推出了对构建**知识图谱 (Knowledge Graphs)** 的一流支持，并集成了手动定义实体和关系的工具包，提升了文档分析能力。现在可以使用知识图谱构建自定义 RAG 流程，相关资源包括 [neo4j 集成](https://t.co/Q0rdH5Dwza)和 [RAG 流程示例](https://t.co/Cx4p8srIEP)。

- **研讨会中的记忆与模型**：即将举行和已录制的网络研讨会展示了 AI 前沿，Julian Saks 和 Kevin Li 讨论了用于长期自主 Agent 记忆的“memary”，另一场会议则聚焦于与 MultiOn 的 Div 讨论“Web Agents 的未来”。点击[此处](https://lu.ma/pl3xn3dh)注册研讨会，并在[此处](https://t.co/eXaW0Yhbv8)在线观看往期会议。

- **并行处理的分歧**：工程师们讨论了 **OpenAI Agent** 进行并行函数调用的能力，LlamaIndex 的文档澄清了这一功能，尽管真正的并行计算仍难以实现。讨论涵盖了 TypeScript 中的持久化以及针对文档集的基于 RAG 的分析，相关示例链接在[文档](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/)中。

- **GPT-4o 在专业文档提取方面表现出色**：最近的[研究](https://www.ourblankspace.com/post/professional-paradigm-shift-gpt-4o-and-project-astra-in-finance)显示，**GPT-4o** 在文档提取方面显著超越了其他工具，平均准确率达到 84.69%，预示着金融等各行业的潜在变革。

- **寻求语义 SQL 协同**：公会思考了**语义层 (Semantic Layers) 与 SQL Retrievers** 的融合，以潜在地增强数据库交互，这一话题仍有待探索，并可能启发未来的集成和讨论。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Latent Space 中的 AI 迷局与动荡**：一段 [AI 反向图灵测试](https://www.youtube.com/watch?v=MxTWLm9vT_o) 视频浮出水面，展示了先进的 AI 试图在它们中间辨别出人类，引发了广泛关注。与此同时，**llama3-V** 被指控涉嫌挪用 MiniCPM-Llama3-V 2.5 的学术成果，正如 [GitHub](https://github.com/OpenBMB/MiniCPM-V/issues/196) 上所记录的那样。

**软件的未来与精英影响力**：工程师们深入探讨了名为 "The End of Software"（软件的终结）的挑衅性 Google Doc 文档的影响，同时讨论了 Anthropic 的 Dario Amodei 在决定推迟聊天机器人 Claude 发布后，入选《时代》周刊百大人物。此外，一篇关于 LLM 应用运营方面的 O'Reilly 文章也因其对一年来构建这些模型的洞察而受到审视。

**AI 活动成为行业枢纽**：最近宣布的 **AI Engineering World Forum (AIEWF)**（详情见 [推文](https://x.com/swyx/status/1797654825968291862)）引发了期待，活动包括新的演讲者、Fortune 500 强中的 AI 专场，以及涵盖各种 LLM 主题和行业领导力的官方活动。

**Zoom 解决技术故障**：一个 Zoom 会议为在直播流中遇到技术中断的成员解了围。他们通过共享的 [Zoom 链接](https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09) 进入会议，继续进行讨论。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**RAG 系统拥抱历史数据**：社区成员讨论了将历史数据集成到 RAG 系统中的策略，建议优化处理 CSV 表格和扫描文档的方法以提高效率。

**游戏聊天机器人表现更强**：关于游戏推荐聊天机器人结构的辩论引出了一条建议：反对将 LangGraph Chatbot agent 拆分为多个 agent，而是倾向于使用统一的 agent 或预先策划的数据集以保持简洁。

**LangChain 与 OpenAI 的对决**：比较 LangChain 与 OpenAI agents 的对话指出，LangChain 在编排 LLM 调用方面具有适应性，强调应根据用例需求来决定是选择抽象层还是直接使用 OpenAI。

**对话式 AI 主题在媒体中走红**：社区中出现的出版物包括在 Google Colab 上使用 Hugging Face 和 LangChain 探索 LLMs，以及 LangChain 中对话式 agent 日益增长的重要性。关键资源包括 [Medium 上的探索指南](https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06) 和 [Ankush k Singal](https://ai.gopubby.com/chatty-machines-the-rise-of-conversational-agents-in-langchain-db3c7972a209) 对对话式 agent 的深入探讨。

**JavaScript 遇到 LangServe 障碍**：一段代码片段分享了 JavaScript 社区在处理 LangServe 中的 `RemoteRunnable` 类时遇到的困难，表现为与消息数组处理相关的 **TypeError**。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad 向 Haskell 迈进**：讨论强调了一位成员因 Python 的局限性而对将 **tinygrad** 翻译成 Haskell 感兴趣，而另一位成员建议专门为 tinygrad 的 **uop end** 开发一种新语言。

**AI 中自动调优的演进**：社区批评了像 **TVM** 这样的旧式自动调优方法，强调需要创新来解决块大小（block size）和流水线（pipelining）调优中的缺点，以提高模型准确性。

**用泰勒级数重新思考 exp2**：包括 **georgehotz** 在内的用户研究了 **Taylor series**（泰勒级数）在改进 **exp2 函数** 方面的适用性，讨论了类似 CPU 的范围缩减和重构方法的潜在益处。

**期待 tinygrad 的量子飞跃**：**George Hotz** 兴奋地宣布 tinygrad 1.0 计划在 NVIDIA 和 AMD 上训练 GPT-2 的速度超过 PyTorch，并发布了一篇 [推文](https://x.com/__tinygrad__/status/1797600620989567163) 强调了即将推出的功能，如 **FlashAttention**，并提议放弃 numpy/tqdm 依赖。

**NVIDIA 乏善可陈的展示引发不满**：Nvidia 首席执行官 **黄仁勋（Jensen Huang）的 COMPUTEX 2024 主题演讲** [视频](https://www.youtube.com/watch?v=pKXDVsWZmUU) 提高了人们对革命性发布的期望，但最终至少让一位社区成员感到非常失望。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Yuan2.0-M32 展示其专业实力**：新款 **Yuan2.0-M32** 模型凭借其 Mixture-of-Experts 架构脱颖而出，并与其 [Hugging Face 仓库](https://huggingface.co/IEITYuan/Yuan2-M32-hf)及[配套研究论文](https://arxiv.org/abs/2405.17976)一同展示。
- **llama.cpp 故障排除**：用户正在定位 **llama.cpp** 中的 Tokenization 问题，引用了具体的 GitHub issues（[#7094](https://github.com/ggerganov/llama.cpp/issues/7094) 和 [#7271](https://github.com/ggerganov/llama.cpp/issues/7271)），并建议在 Finetuning 过程中进行仔细验证。
- **Axolotl 适配 AMD**：针对 AMD 兼容性修改 **Axolotl** 的工作已经展开，并在 GitHub 上发布了[实验性 ROCm 安装指南](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1550)。
- **明确 Axolotl 的非加密货币领域属性**：在社区的一次澄清中，重申 Axolotl 专注于训练 Large Language Models，明确表示不涉及加密货币。
- **使用 wandb 追踪 QLoRA 训练**：成员们正在交流如何实现 **wandb** 以监控 QLoRA 训练期间的参数和 Loss，并参考了现有的 **wandb 项目**和特定的 `qlora.yml` 配置。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

**AI 协作公开招募**：Manifold Research 正在寻找合作伙伴，共同构建受 GATO 启发的开源“Generalist”模型，目标是涵盖视觉、语言等领域的 Multimodal 和控制任务。

**Cohere 社区故障排查**：Cohere Chat API 文档中一个失效的 Dashboard 链接被发现并标记，社区成员已介入确认并着手修复。

**AI 模型 Aya 23 获得好评**：一位用户分享了对 Cohere Aya 23 模型的成功测试，并表示希望分发其代码以供同行评审。

**社区标签升级揭晓**：Discord 更新后的标记机制引发了社区的热议和期待，成员们分享了[标签说明链接](https://discord.com/channels/954421988141711382/954421988783444043/1246005007141175386)。

**支持网络已激活**：对于遇到聊天记录消失或其他问题的用户，提供了 Cohere 支持团队邮箱 **support@cohere.com** 或服务器指定支持频道的重定向指引。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Whisper 助力 OI**：将 **Whisper 或 Piper** 集成到 Open Interpreter (OI) 的工作正在进行中；旨在减少冗余并提高语音启动速度。目前尚无在非 Ubuntu 系统上成功安装 OI 的报告；一次在 MX Linux 上的尝试因 Python 问题而失败。

- **Agent 决策困惑已澄清**：对 OI 内部的“Agent-like decisions”进行了澄清，并指向了代码库中的特定部分——带有[默认系统消息](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/default_system_message.py)中 Prompt 的 LLM。

- **寻找市场营销人员**：小组讨论了对 *Open Interpreter* 进行市场推广的需求，此前该工作由个人负责。

- **Gemini 运行遇到问题**：关于在 Open Interpreter 上运行 **Gemini** 的咨询被提出，因为提供的文档似乎已过时。

- **OI 的移动端布局**：关于创建将 OI 服务器连接到 iPhone 的 App 的讨论非常活跃，目前已有 [GitHub 代码](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile)和 iOS 版本的 [TestFlight 链接](https://testflight.apple.com/join/v8SyuzMT)。iOS 上的 TTS 功能已确认，而 Android 版本正在开发中。

- **聚焦 Loyal-Elephie**：用户 cyanidebyte 提到了 [Loyal-Elephie](https://github.com/v2rockets/Loyal-Elephie)，但未提供具体背景。

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Hugging Face 发生安全漏洞**：未经授权的访问损害了 Hugging Face 的 Spaces 平台上的密钥，建议用户更新密钥并使用 *fine-grained access tokens*（细粒度访问令牌）。完整详情请参阅此 [安全更新](https://huggingface.co/blog/space-secrets-disclosure)。
  
- **AI2 主动更新令牌**：针对 Hugging Face 事件，AI2 正作为预防措施刷新其令牌。不过，Nathan Lambert 报告称他的令牌已自动更新，从而减轻了手动操作的必要。

- **Phi-3 模型加入行列**：Phi-3 Medium (14B) 和 Small (7B) 模型已添加到 @lmsysorg 排行榜，其表现分别与 GPT-3.5-Turbo 和 Llama-2-70B 相当，但同时也提醒不要仅针对学术基准（benchmarks）优化模型。

- **VLM 社区的抄袭指控**：有讨论称 Llama 3V 是一个抄袭模型，据称是在 MiniCPM-Llama3-V 2.5 的框架上进行了细微改动。包括 Chris Manning 的批评和一篇现已删除的 Medium 文章在内的链接，引发了关于 VLM 社区诚信的讨论。

- **捐赠赌注（Donation-Bets）受到青睐**：Dylan 将一场关于模型性能的输掉的赌注转化为慈善机会，在成员中引发了“捐赠赌注”的趋势，成员们认为这既能支持公益事业，也能提升个人声誉。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Mozilla 支持本地 AI 创新**：**Mozilla Builders Accelerator** 现已开放申请，目标是 **Local AI** 领域的创新者，提供高达 **100,000 美元的资金**、导师指导以及在 Mozilla 网络上展示突破性项目的舞台。[立即申请](https://future.mozilla.org/builders/blog/announcing-mozilla-builders/)，将个人设备转变为本地 AI 的强大动力源。

- **使用 Paddler 增强 llama.cpp**：工程师们正考虑将 [Paddler](https://github.com/distantmagic/paddler)（一个有状态负载均衡器）与 llama.cpp 集成，以简化 llamafile 的操作，这可能提供更高效的模型推理服务能力。

- **采样缓慢引发对 JSON Schema 的质疑**：由于服务器问题，AI 工程师遇到了采样速度变慢的情况，并确定了 JSON Schema 验证存在问题，引用了 [llama.cpp 仓库](https://github.com/ggerganov/llama.cpp/issues/7703)中的一个特定 issue。

- **API 端点兼容性处理**：可用性讨论显示，OpenAI 兼容的聊天端点 `/v1/chat/completions` 适用于本地模型；然而，模型特定的角色（roles）需要进行调整，而这些调整以前是由 OpenAI 的处理流程完成的。

- **努力实现模型接口的统一**：尽管由于不同模型的特性存在固有挑战，但大家仍在共同努力维护各种模型和供应商之间的统一接口，这需要为 Mistral-7b-instruct 等模型提供定制化的预处理解决方案。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Spaetzle 令参与者感到困惑**：成员们讨论了 **Spaetzle** 模型的细节，并澄清实际上存在多个模型而非单一实体。一篇相关的 [AI 生成的 Medium 文章](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88)强调了微调预训练 LLM 的不同方法，其中包括 **phi-3-mini-instruct** 和 **phoenix** 等名称。
  
- **对 Replay Buffer 实现的期待**：一篇关于 **InstructLab** 的文章描述了一种可能与 **Spaetzle** 密切相关的 Replay Buffer 方法；然而，该方法至今尚未实现。围绕这一概念的兴趣正在酝酿，预示着未来的潜在发展。

- **破译德语数字**：有人征求关于德语手写识别模型的建议，**Kraken** 被推荐为一个选项，并附带了一个[调查链接](https://www.berd-nfdi.de/limesurvey/index.php/996387)，可能用于进一步的研究或信息收集。

- **模型基准测试与策略分享**：微调方法的有效性是一个核心话题，一位成员表示打算研究 **InstructLab** 的相关材料。虽然在 **Spaetzle** 的背景下提到了这些模型，但并未提供具体的模型基准测试数据。



---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **Claude 3 的分词困扰**：工程师们发现 **Claude 3** 缺乏专用的 tokenizer（分词器）令人费解，而这是语言模型预处理的关键工具。
- **Nomic 模型查询**：关于如何使用 **nomic-embed-text-v1 模型** 存在困惑，因为它没有出现在 `llm models` 命令输出的 **gpt4all models** 列表中。
- **SimonW 的插件转向**：对于 embedding 任务，SimonW 建议切换到 [llm-sentence-transformers 插件](https://github.com/simonw/llm-sentence-transformers)，该插件似乎对 **Nomic 模型** 提供了更好的支持。
- **参考发布说明进行专业嵌入**：关于 **nomic-embed-text-v1 模型** 的详细安装和使用说明，可以在 **llm-sentence-transformers** 的 [0.2 版本发布说明](https://github.com/simonw/llm-sentence-transformers/releases/tag/0.2) 中找到。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Jamba Instruct 与 Mixtral 旗鼓相当**：在讨论中，Jamba Instruct 的性能被比作 **Mixtral 8x7B**，使其成为近期备受关注的 GPT-4 模型的强力竞争对手。
- **函数组合：AI 的阿喀琉斯之踵**：一篇分享的 [LinkedIn 帖子](https://www.linkedin.com/feed/update/urn:li:activity:7203325801356746752/) 揭示了当前机器学习模型（如 Transformer 和 RNN）的局限性，指出了 *function composition*（函数组合）面临的挑战，并提到 Jamba 参与了相关的 SSM 实验。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **参与黑客松，创新医疗健康**：**Alliance AI4Health 医疗创新挑战赛黑客松/创意赛** 正在征集参与者，开发 AI 驱动的医疗解决方案。该活动提供超过 *$5k 的奖金*，旨在激发医疗技术领域的突破性进展。[点击注册](https://amic.devpost.com/)。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

**YAIG (a16z Infra) Discord** 没有新消息。如果该社区长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1246205545321336973)** (1 条消息): 

- **HF Spaces 紧急安全警报**：由于发生安全事件，强烈建议用户*轮换*在 [HF Spaces](https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets) 中使用的任何 token 或密钥。更多详情请查看官方 [博客文章](https://huggingface.co/blog/space-secrets-disclosure)。

**提到的链接**：<a href="https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets).">Spaces 概览</a>：未找到描述

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1246177147375321089)** (974 条消息🔥🔥🔥): 

- **关于自然智能与人工智能的争议**：成员们辩论了培养的神经元是否可以被视为人工制品，讨论了定义和伦理影响。一位成员认为，劳动密集型的创造过程使产品具有人工属性，这引发了争议。
- **Hugging Face 基础设施问题**：成员们在使用 Hugging Face Inference API 时遇到问题，报告了多个 "MaxRetryError" 错误。该问题已提交给团队解决，随后功能恢复正常。
- **有限资源下的模型微调**：一位用户在尝试使用有限 RAM 进行模型微调和推送时遇到困难，寻求关于使用量化技术的建议。一位成员建议使用 `peft` 库中的 `BitsAndBytesConfig`，最终解决了该问题。
- **播客与学习资源**：成员们交流了各种播客推荐，包括 Joe Rogan Experience、Lex Fridman 以及特定的编程相关播客。此外，还讨论了不同类型的内容对各种学习（包括 AI 和 Rust 编程）的帮助程度。
- **LevelBot 活动追踪器**：宣布了 HF LevelBot 的新活动追踪器，允许用户查看自己的活动。建议包括追踪更多类型的操作、关联 GitHub 活动以及改进图形界面。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://huggingface.co/spaces/pandora-s/HF-Serverless-LLM-Inference-API-Status">HF Serverless LLM Inference API Status - pandora-s 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/nroggendorff/mayo">nroggendorff/mayo · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/noaroggendorff/status/1796981388115062951">来自 Noa Roggendorff (@noaroggendorff) 的推文</a>: 我所有的兄弟都讨厌 pickle，带着你的 bin 和 pt 滚出这里。下次请使用 safetensors。</li><li><a href="https://huggingface.co/google/switch-c-2048">google/switch-c-2048 · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=d8k4Pit4_ZU">It Started to Sing</a>: 这是 ElevenLabs Music 的早期预览。这首歌是由单个文本 prompt 生成的，未经任何修改。风格：“流行、流行摇滚、乡村、排行榜热门歌曲。”</li><li><a href="https://huggingface.co/spaces/lllyasviel/Omost">Omost - lllyasviel 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://paperswithcode.com/sota/code-generation-on-humaneval">Papers with Code - HumanEval 基准测试 (代码生成)</a>: 目前 HumanEval 上的 SOTA 是 AgentCoder (GPT-4)。查看 127 篇附带代码的论文完整对比。</li><li><a href="https://tenor.com/view/kryonax-skull-gif-26476587">Kryonax Skull GIF - Kryonax Skull - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/datasets/blanchon/udio_dataset">blanchon/udio_dataset · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/posts/lunarflu/239147617114976">Hugging Face 上的 @lunarflu: "应大众要求，HF activity tracker v1.0 来了！📊 让我们开始构建吧…"</a>: 未找到描述</li><li><a href="https://github.com/huggingface/knockknock">GitHub - huggingface/knockknock: 🚪✊Knock Knock: 只需额外两行代码，即可在训练结束时收到通知</a>: 🚪✊Knock Knock: 只需额外两行代码，即可在训练结束时收到通知 - huggingface/knockknock</li><li><a href="https://www.youtube.com/watch?v=UvWVfVnVZXc">Hugging Face 读书会 21: 了解 AI 故事生成的现状</a>: 演讲者: Isamu Isozaki 文章总结: https://medium.com/@isamu-website/understanding-ai-for-stories-d0c1cd7b7bdc 往期演讲: https://github.com/isamu-iso...</li><li><a href="https://github.com/huggingface/transformers/blob/96eb06286b63c9c93334d507e632c175d6ba8b28/src/transformers/models/t5/modeling_t5.py#L354">huggingface/transformers 项目中的 transformers/src/transformers/models/t5/modeling_t5.py (版本 96eb06286b63c9c93334d507e632c175d6ba8b28)</a>: 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的 SOTA 机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/JonathonLuiten/Dynamic3DGaussians?tab=readme-ov-file">GitHub - JonathonLuiten/Dynamic3DGaussians</a>: 通过在 GitHub 上创建账户来为 JonathonLuiten/Dynamic3DGaussians 的开发做出贡献。</li><li><a href="https://github.com/nullonesix/sign_nanoGPT">GitHub - nullonesix/sign_nanoGPT: 使用符号梯度下降 (sign gradient descent) 代替 adamw 的 nanoGPT</a>: 使用符号梯度下降代替 adamw 的 nanoGPT - nullonesix/sign_nanoGPT</li><li><a href="https://huggingface.co/docs/diffusers/en/using-diffusers/img2img">Image-to-image</a>: 未找到描述</li><li><a href="https://github.com/bigscience-workshop/petals">GitHub - bigscience-workshop/petals: 🌸 以 BitTorrent 方式在本地运行 LLM。微调和推理速度比 offloading 快达 10 倍</a>: 🌸 以 BitTorrent 方式在本地运行 LLM。微调和推理速度比 offloading 快达 10 倍 - bigscience-workshop/petals</li><li><a href="https://github.com/huggingface/transformers/blob/96eb06286b63c9c93334d507e632c175d6ba8b28/examples/pytorch/summarization/run_summarization.py#L441">huggingface/transformers 项目中的 transformers/examples/pytorch/summarization/run_summarization.py (版本 96eb06286b63c9c93334d507e632c175d6ba8b28)</a>: 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的 SOTA 机器学习库。 - huggingface/transformers</li><li><a href="https://discuss.huggingface.co/t/loaded-adapter-seems-ignored/88195/1">已加载的 adapter 似乎被忽略了</a>: 大家好，我是一个纯小白，正尝试做一些酷炫的事情。我成功地使用 QLoRA 微调了一个模型，使用 'NousResearch/Llama-2-7b-chat-hf' 作为基础模型，并创建了...</li><li><a href="https://huggingface.co/docs/hub/en/api#get-apimodels>">Hub API 端点</a>: 未找到描述</li><li><a href="https://huggingface.co/nroggendorff/vegan-mayo">nroggendorff/vegan-mayo · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/tesseract-ocr/tesseract">GitHub - tesseract-ocr/tesseract: Tesseract 开源 OCR 引擎 (主仓库)</a>: Tesseract 开源 OCR 引擎 (主仓库) - tesseract-ocr/tesseract</li><li><a href="https://huggingface.co/Alibaba-N">huggingface/Alibaba-N...</a>: 未找到描述</li>

LP/gte-large-en-v1.5/blob/main/sentence_bert_config.json#L2">sentence_bert_config.json · Alibaba-NLP/gte-large-en-v1.5 at main</a>: 未找到描述</li><li><a href="https://youtu.be/vmDDOFXSgAs?si=z8ppNiu9_Btzzcjt">Dave Brubeck - Take Five</a>: Dave Brubeck - Take Five</li><li><a href="https://huggingface.co/docs/transformers/main/main_classes/quantization#transformers.BitsAndBytesConfig.">Quantization</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/v4.41.3/model_doc/mistral#shrinking-down-mistral-using-quantization">Mistral</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1246373691051479061)** (28 条消息🔥): 

- **为 LLM 聊天机器人部署 3D 网站**：一位成员正在为一个 LLM 聊天机器人部署 3D 网站，并邀请其他人加入。
- **通过 Rust 学习 d2l.ai**：一位成员正在使用 d2l.ai 这本书学习如何在 Rust 中使用 Candle，并分享了他们的 [GitHub 仓库](https://github.com/asukaminato0721/d2l.ai-rs)。另一位用户询问了这本书；这是一本著名的深度学习教科书，但目前缺少 Rust 版本。
- **Candle 在 Rust 中的优势**：讨论揭示了相比 PyTorch，使用 Candle 的优势，包括由于基于 Rust 系统而带来的“更少的依赖开销”和“易于部署”。
- **投入更多资金和更好的硬件来训练模型**：一位用户幽默地建议，投入更多的钱能产生更好的模型，并提到他们使用 A6000 GPU，但在每步 200 秒的较慢训练下获得了更好的结果。
- **评估 Whisper medium**：一位用户正在评估 Whisper medium en，但在尝试使用 pipeline 函数获取逐词时间戳（而非整段）时遇到问题。

**提及的链接**：<a href="https://github.com/asukaminato0721/d2l.ai-rs">GitHub - asukaminato0721/d2l.ai-rs: use candle to implement some of the d2l.ai</a>：使用 candle 实现 d2l.ai 的部分内容。欢迎在 GitHub 上为 asukaminato0721/d2l.ai-rs 的开发做出贡献。

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1246272717854871653)** (3 条消息): 

- **近期论文中的 AI 系统概述**：arXiv 论文 [2312.01939](https://arxiv.org/abs/2312.01939) 深入探讨了与日益增长的资源需求、数据集和基础设施相关的当代 AI 能力。它讨论了强化学习通过动力学（dynamics）、奖励模型、价值函数、策略和原始数据进行的知识表示。
  
- **SatPost 中的精通与热门话题**：一篇 Substack 文章讨论了 Jerry Seinfeld 和铃木一朗（Ichiro Suzuki）对精通技能的执着，以及 Netflix 密码政策的成功、Red Lobster 的破产和热门迷因。点击[此处](https://www.readtrung.com/p/jerry-seinfeld-ichiro-suzuki-and?utm_campaign=post&utm_medium=web)查看严肃见解与幽默的结合。

- **随着 Langchain 兴起的对话式 Agent**：一篇名为 "Chatty Machines: The Rise of Conversational Agents in Langchain" 的文章发表在 [AI Advances](https://ai.gopubby.com/chatty-machines-the-rise-of-conversational-agents-in-langchain-db3c7972a209) 上，强调了对话式 Agent 日益增长的存在感。作者是 Ankush K Singal，涵盖了该领域的进展和实现。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ai.gopubby.com/chatty-machines-the-rise-of-conversational-agents-in-langchain-db3c7972a209">Chatty Machines: The Rise of Conversational Agents in Langchain</a>：Ankush k Singal</li><li><a href="https://www.readtrung.com/p/jerry-seinfeld-ichiro-suzuki-and?utm_campaign=post&utm_medium=web">Jerry Seinfeld, Ichiro Suzuki and the Pursuit of Mastery</a>：来自 1987 年《Esquire》杂志的笔记，该杂志启发了 Jerry Seinfeld 去“追求精通，因为那将充实你的人生”。</li><li><a href="https://arxiv.org/abs/2312.01939">Foundations for Transfer in Reinforcement Learning: A Taxonomy of Knowledge Modalities</a>：当代人工智能系统展现出快速增长的能力，同时也伴随着所需资源、庞大数据集以及相应计算基础设施投资的增长...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1246223559966195803)** (11 条消息🔥): 

- **Fast Mobius 演示令人惊叹**：一位成员分享了一个 [Fast Mobius 演示](https://huggingface.co/spaces/ehristoforu/Mobius)，强调了这是从 Proteus-V0.3 复制的 Space。帖子中包含了多个头像以增强信息表达。

- **Max4Live 设备即将问世**：另一位成员庆祝 **gary4live** 设备接近投产，强调使用 **electron js** 构建 UI，并使用 **redis/mongoDB/gevent** 确保后端稳定性。他们提到了代码签名（code signing）方面的挑战，并分享了一个 [YouTube 演示](https://youtu.be/4R_zykShNKw)。

- **文献综述中的 LLM 推理笔记**：提供了一份关于 LLM 推理现状研究的详细总结，包括缺乏关于 GNNs 的论文、Chain of Thought 的潜力以及对 Graph of Thoughts 的兴趣。完整笔记可在 [Medium](https://medium.com/me/stats/post/dbd9fa3fc1a0) 上查阅。

- **奇特的视角**：分享了多个幽默且富有想象力的帖子，包括 *“当你把楼上的套房租给那个总是谈论核能的怪人时”* 以及 *“伯利兹市居民经历转型的更具迷幻色彩的视角”*。

- **历史人物的趣谈**：一位成员分享了一个关于马克·安东尼和克利奥帕特拉的轻松 [YouTube 视频](https://youtube.com/shorts/o0ZxVTcLdow?feature=share)，标签为 #facts #funny #lovestory。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/ehristoforu/Mobius">Mobius - a Hugging Face Space by ehristoforu</a>: 未找到描述</li><li><a href="https://youtube.com/shorts/o0ZxVTcLdow?feature=share">Mark Antony and Cleopatra #facts #funny #lovestory #love</a>: 未找到描述</li><li><a href="https://medium.com/me/stats/post/dbd9fa3fc1a0">no title found</a>: 未找到描述</li><li><a href="https://youtu.be/4R_zykShNKw">demo turns into another ableton speedrun - musicgen and max for live - captains chair s1 encore</a>: 第一季 e.p. 将于 6 月 7 日发布 https://share.amuse.io/album/the-patch-the-captains-chair-season-one-1 来自背景音乐，saphicord 社区样本包：http...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1246899544210411692)** (5 条消息): 

- **分享关于文本生成图像扩散模型的研究论文**：一位成员分享了一篇由多位研究人员撰写的 [Arxiv 论文](https://arxiv.org/abs/2405.14854) 链接，重点介绍了大规模预训练文本生成图像扩散模型（text-to-image diffusion models）的最新进展。 
- **幽默纠正误艾特（Ping）**：在不小心艾特错人后，一位成员表示道歉，并使用 Discord 表情符号 `<:Thonk:859568074256154654>` 幽默地承认了错误。

**提到的链接**: <a href="https://arxiv.org/abs/2405.14854">TerDiT: Ternary Diffusion Models with Transformers</a>: 大规模预训练文本生成图像扩散模型的最新进展显著提高了高保真图像的生成质量，特别是随着基于 Transformer 的扩散模型的出现...

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1246258011937443850)** (9 条消息🔥): 

- **使用 TrOCR 和 manga-ocr 训练 OCR 模型**：为了训练针对非英语手写文档的 OCR 模型，一位成员建议使用 TrOCR，并指出其通过 [manga-ocr](https://github.com/kha-white/manga-ocr) 在日语文本上的应用。他们还链接了详细的 [TrOCR 文档](https://huggingface.co/docs/transformers/en/model_doc/trocr)。

- **新兴的 VLM 在文档 AI 任务中表现出色**：如今，像 Pix2Struct 和 UDOP 这样的 VLM 在文档 AI（特别是 OCR 任务）中越来越有效。一位成员强调了最近的模型，如 [MiniCPM-Llama3-V 2.5](https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5) 和 [CogVLM2-Llama3-chat-19B](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B)，它们在 DocVQA 等基准测试中表现良好。

- **了解视觉语言模型 (VLMs)**：通过一篇研究论文分享了关于 VLM 及其功能、训练和评估的介绍，可通过 [huggingface.co/papers/2405.17247](https://huggingface.co/papers/2405.17247) 访问。讨论强调了整合视觉和语言模型日益增长的重要性和挑战。

- **社区活动与协作**：成员们受邀参加另一个频道的 Computer Vision Hangout，以促进社区参与和对进行中项目的协作。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/papers/2405.17247">论文页面 - Vision-Language Modeling 简介</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/trocr">TrOCR</a>: 未找到描述</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers/model_doc/vit#resources">Vision Transformer (ViT)</a>: 未找到描述</li><li><a href="https://github.com/kha-white/manga-ocr">GitHub - kha-white/manga-ocr: 针对日语文本的光学字符识别，主要关注日本漫画</a>: 针对日语文本的光学字符识别，主要关注日本漫画 - kha-white/manga-ocr</li><li><a href="https://pen2txt.com/.">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1246334890396483595)** (24 条消息🔥): 

- **8B 参数的 Llama 3 导致内存问题**：一位用户提到在本地安装和使用 8B 参数的 **Llama 3** 后，其**本地内存告急**。另一位用户建议使用 **llama cpp** 提供的 **4-bit 量化技术**来缓解内存问题。

- **日语到英语的最佳翻译模型**：一位用户请求推荐 Hugging Face 上最好的日语到英语翻译模型。另一位用户推荐了 **[Helsinki-NLP/opus-mt-ja-en](https://huggingface.co/Helsinki-NLP/opus-mt-ja-en)** 来完成该任务，并引用了各种资源和基准测试。

- **RAG 资源**：对于那些寻找 **RAG (Retrieval-Augmented Generation)** 资源的人，建议参考 **[Hugging Face 的开源 AI 食谱 (Open-Source AI Cookbook)](https://huggingface.co/learn/cookbook/en/index)**。该资源包括专门针对 **RAG 方案和其他 AI 应用**的章节。

- **遇到 Graphcodebert 的 tree_sitter 问题**：一位用户在尝试使用 tree_sitter 在 Graphcodebert 中构建库时遇到了 **AttributeError**。该用户的目录列表显示其环境中不存在 "build_library" 属性，这暗示可能存在配置错误或缺少依赖项。

- **使 LLM 具有确定性**：为了使 **大语言模型 (LLM) 具有确定性**，一位用户询问了除了将 temperature 设置为 1 之外的指导。另一位用户澄清说，正确的设置是 **`do_sample=False`** 并将 temperature 设置为 **0**。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/learn/cookbook/en/index">开源 AI 食谱 - Hugging Face 开源 AI 食谱</a>: 未找到描述</li><li><a href="https://huggingface.co/Helsinki-NLP/opus-mt-ja-en">Helsinki-NLP/opus-mt-ja-en · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/AnjneyMidha/status/1797621758909911412">来自 Anjney Midha (@AnjneyMidha) 的推文</a>: 训练原始 llama 模型花费了约 3000 万美元，但将其微调为当时的尖端模型 Vicuna 仅花费了 300 美元（即 &lt;0.1%）。任何声称需要巨大算力的人...
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1247275018321264847)** (1 messages): 

- **在机器人领域结合 Lerobot 和 Diffusion**：一名成员分享了一篇详细的博客文章 [Diving into Diffusion Policy with Lerobot](https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/)，解释了在机器人训练中集成 **Action Chunking Transformer (ACT)** 的方法。该文章描述了 ACT 如何利用 encoder-decoder transformer 根据图像、机器人状态和可选的样式变量来预测动作，并将其与从高斯噪声开始的 **Diffusion Policy** 方法进行了对比。

**提到的链接**：<a href="https://radekosmulski.com/diving-into-diffusion-policy-with-lerobot/">Diving into Diffusion Policy with LeRobot</a>：在最近的一篇博客文章中，我们研究了 Action Chunking Transformer (ACT)。ACT 的核心是一个 encoder-decoder transformer，当传入：* 一张图像 * 机器人的当前状态 ...

  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1246188459895558267)** (2 messages): 

- **Gradio 通过 @gr.render 支持动态布局**：令人兴奋的消息，**Gradio** 现在通过 **@gr.render** 功能支持动态布局，能够动态集成组件和事件监听器。更多详情请查看[指南](https://www.gradio.app/guides/dynamic-apps-with-render-decorator)。

- **Todo 应用示例**：分享的一个示例是 Todo List App，其中可以使用 @gr.render 动态添加和重新排列文本框及响应式按钮。链接的指南提供了完整的代码片段和演练。

- **AudioMixer 应用示例**：另一个示例是音乐混音器应用，允许用户通过 @gr.render 和 Python 循环动态添加多个轨道。详细的源代码和说明已在[指南](https://www.gradio.app/guides/dynamic-apps-with-render-decorator)中提供。

**提到的链接**：<a href="https://www.gradio.app/guides/dynamic-apps-with-render-decorator">Dynamic Apps With Render Decorator</a>：Gradio 分步教程

  

---



### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1246180496271216731)** (919 messages🔥🔥🔥): 

- **多 GPU 微调更新**：多 GPU 微调的模型支持和许可工作正在取得进展。一位项目成员解释道：“我们可能会做多模态（multimodal），但这可能需要更多时间。”
- **LoRA 微调 vs 全量微调**：关于 LoRA 和全量微调（full tuning）的讨论显示，在新知识保留与旧知识丢失方面结果各异。[详细的论文分析](https://x.com/rohanpaul_ai/status/1796556756677497294)强调了为什么 LoRA 在某些场景下表现更优（例如，较少的源域遗忘）。
- **训练设置与错误**：几位用户报告了关于 tokenizer 设置和微调配置的技术挑战。由于半精度训练中的溢出问题，建议：“你可以考虑在代码中添加 `tokenizer.padding_side = 'right'`。”
- **用于加速 LLM 微调的 Kaggle Notebooks**：团队分享了关于修复其快 2 倍的 LLM 微调 Kaggle notebooks 的更新，鼓励用户尝试并报告任何问题。根据他们的分析，“强制重新安装 aiohttp 可以解决问题！”
- **H100 NVL 问题**：一名用户在 H100 NVL 上运行 RoPE 优化任务时遇到了持续的问题，表现为 VRAM 使用不一致和响应时间缓慢。社区推测可能存在内存报告 Bug 或未解释的 VRAM 卸载到系统 RAM 的情况。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.unsloth.ai/blog/llama3">使用 Unsloth 微调 Llama 3</a>：通过 Unsloth 轻松微调 Meta 的新模型 Llama 3，上下文长度增加 6 倍！</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.09673">LoRA Learns Less and Forgets Less</a>：Low-Rank Adaptation (LoRA) 是一种广泛使用的针对大语言模型（LLM）的参数高效微调方法。LoRA 通过仅对选定的权重矩阵训练低秩扰动来节省内存。在...</li><li><a href="https://arxiv.org/abs/2405.15613">Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach</a>：自监督特征是现代机器学习系统的基石。它们通常在数据集上进行预训练，而这些数据集的构建和整理通常需要大量的人力...</li><li><a href="https://lightning.ai/fareedhassankhan12/studios/building-llama-3-from-scratch?section=tutorials">从零开始构建 LLaMA 3 - fareedhassankhan12 的 Lightning Studio</a>：LLaMA 3 是继 Mistral 之后最有前途的开源模型之一，我们将以更简单的方式重构其架构。</li><li><a href="https://huggingface.co/docs/transformers/en/hpo_train">使用 Trainer API 进行超参数搜索</a>：未找到描述</li><li><a href="https://x.com/bycloudai/status/1797444223022629350/">bycloud (@bycloudai) 的推文</a>：我们这么快就有了 MAMBA-2？？？？？？？？？ https://arxiv.org/abs/2405.21060 作者是 Tri Dao 和 Albert Gu，他们也是 mamba-1 的作者，Tri Dao 还是 flash attention 1 & 2 的作者。将阅读论文...</li><li><a href="https://x.com/danielhanchen/status/1796941785731846152">Daniel Han (@danielhanchen) 的推文</a>：修复了我们快 2 倍的 LLM 微调 Kaggle notebook！强制重新安装 aiohttp 修复了问题！如果你不知道，Kaggle 每周免费提供 30 小时的 T4！T4 拥有 65 TFLOPS，相当于 1 个 RTX 3070 的 80% (...</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1791900967472140583">Daniel Han (@danielhanchen) 的推文</a>：我对 "LoRA Learns Less and Forgets Less" 的看法：1) "MLP/All" 不包括 gate_proj。训练了 QKVO, up & down，但没有 gate (第 3 页脚注) 2) 为什么 LoRA 在数学和...</li><li><a href="https://x.com/rohanpaul_ai/status/1796556756677497294">Rohan Paul (@rohanpaul_ai) 的推文</a>：论文 - 'LoRA Learns Less and Forgets Less' ✨ 👉 LoRA 在指令微调方面比持续预训练效果更好；它对学习率特别敏感；性能最...</li><li><a href="https://github.com/hiyouga/LLaMA-Factory/wiki/Performance-Comparison#nvidia-a100--2">性能对比</a>：统一 100 多个 LLM 的高效微调。通过在 GitHub 上创建账号为 hiyouga/LLaMA-Factory 的开发做出贡献。</li><li><a href="https://github.com/huggingface/peft">GitHub - huggingface/peft: 🤗 PEFT: State-of-the-art Parameter-Efficient Fine-Tuning.</a>：🤗 PEFT：最先进的参数高效微调。- huggingface/peft</li><li><a href="https://github.com/unslothai/unsloth/wiki">主页</a>：微调 Llama 3, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth</li><li><a href="https://www.sbert.net/examples/training/hpo/README.html">超参数优化 &mdash; Sentence Transformers 文档</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/philschmid/guanaco-sharegpt-style">philschmid/guanaco-sharegpt-style · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi & Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1246233559732322396)** (13 条消息🔥): 

- **Phi 3 模型无法在 GTX 3090 上运行**：一位成员在使用 GTX 3090 运行 **ollama** 或 **LM Studio** 上的 **Phi 3 medium 或 mini 128** 时遇到困难，尽管尝试了来自不同来源的多个量化版本（quants）和模型，仍不断报错。

- **Qwen2 在 LMsys 上现身**：一位成员提醒大家在 [chat.lmsys.org](https://chat.lmsys.org) 上发现了 **Qwen2** 的踪迹。

- **讨论企业对开源 AI 模型的排斥**：针对企业因 AI 安全担忧而迟疑采用开源模型的情况展开了讨论。问题集中在模型是否会生成有害内容，以及如何防止它们响应不当的 prompts。

- **LLMs 受限于训练数据**：会议指出 **LLMs** 只能生成其训练数据中包含的内容，无法进行原创研究或创新，例如发明“用 50 欧元的杂货和空气炸锅制造核弹”。

- **训练模型以规避无关话题**：成员们讨论了训练模型拒绝回答无关或潜在有害 prompts 的技术。方法包括 **DPO/ORPO**、**control vectors**，或者使用独立的文本分类器（text classifier）来检测并拦截不合规的 prompts，并返回固定回复。

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1246271721217196125)** (170 条消息🔥🔥): 

<ul>
  <li><strong>save_strategy 错误引发困惑</strong>：成员们讨论了在 trainer 中使用 save_strategy 时出现的 'dict object has no attribute to_dict' 错误。一位成员建议使用 model.model.to_dict。</li>
  <li><strong>Unsloth adapter 可与 HuggingFace 配合使用</strong>：一位成员确认 Unsloth 微调后的 adapter 可以用于 HuggingFace 的 pipeline/文本生成推理端点。</li>
  <li><strong>GGUF 转换中的推理问题</strong>：一位用户分享了在转换为 GGUF 并使用 Ollama 运行时出现的幻觉问题。该用户报告称使用 Unsloth 解决了该问题，并被建议尝试 VLLM 以获得一致的性能。</li>
  <li><strong>分享 Kaggle 安装修复方案</strong>：一位成员通过使用特定命令升级到最新的 aiohttp 版本，解决了在 Kaggle 上安装 Unsloth 的问题。</li>
  <li><strong>文档和易用性更新</strong>：成员们指出了多项文档更新、GitHub 链接以及即将推出的支持功能，如多 GPU 兼容性和 8-bit 量化。还讨论了仓库访问令牌问题以及在 WSL2 上为 Unsloth 使用 Docker 镜像的说明。</li>
</ul>

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1OKn_TswCiYK6EIYse7RrKKA-HLW8qMcU?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">Home</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 快 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth?tab=readme-ov-file#-installation-instructions">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>: 微调 Llama 3, Mistral, Phi &amp; Gemma LLMs 快 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth/llama-3-8b">unsloth/llama-3-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-8b-bnb-4bit">unsloth/llama-3-8b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/llama-3-70b-bnb-4bit">unsloth/llama-3-70b-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/566#issuecomment-2143365751">[已修复] Kaggle notebook: No module named &#39;unsloth&#39; · Issue #566 · unslothai/unsloth</a>: 我在 Kaggle notebook 中遇到了错误，这是我安装 unsloth 的方法：</li><li><a href="https://github.com/Jiar/jupyter4unsloth">GitHub - Jiar/jupyter4unsloth: Jupyter for Unsloth</a>: 用于 Unsloth 的 Jupyter。通过在 GitHub 上创建账号为 Jiar/jupyter4unsloth 的开发做出贡献。</li><li><a href="https://github.com/Jiar/jupyter4unsloth/blob/main/Dockerfile">jupyter4unsloth/Dockerfile at main · Jiar/jupyter4unsloth</a>: 用于 Unsloth 的 Jupyter。通过在 GitHub 上创建账号为 Jiar/jupyter4unsloth 的开发做出贡献。</li><li><a href="https://download.pytorch.org/whl/cu121">无标题</a>: 未找到描述</li><li><a href="https://github.com/huggingface/datasets/issues/6753">Type error when importing datasets on Kaggle · Issue #6753 · huggingface/datasets</a>: 描述错误：当尝试运行 import datasets print(datasets.__version__) 时，产生以下错误 TypeError: expected string or bytes-like object。看起来它找不到 val...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1246779242356801596)** (32 messages🔥): 

- **实验 Ghost Beta Checkpoint**：正在对 **Ghost Beta** 的 Checkpoint 版本进行实验，涵盖多种语言，包括**德语**、**英语**、**西班牙语**、**法语**、**意大利语**、**韩语**、**越南语**和**中文**。该模型针对生产环境和低成本效率进行了优化，重点在于易于自我部署 ([GhostX AI](https://ghost-x.org/))。

- **评估语言质量**：使用 **GPT-4** 以 10 分制评估模型的多语言能力，但该评估方法尚未正式发布。受信任的评估者和社区贡献将有助于完善这一评估，以获得客观视角。

- **处理西班牙语变体**：模型在训练过程中采用了一种称为“缓冲语言 (buffer languages)”的方法来处理西班牙语的地域差异。该方法仍在开发中，具体细节将在模型发布时公布。

- **数学能力与 Letcode**：通过 **Letcode** 平台上的示例展示了模型的数学能力。鼓励用户在 [chat.lmsys.org](https://chat.lmsys.org) 上将其与其他模型进行对比。

- **管理微调的 Checkpoints**：用户讨论了将 Checkpoints 保存到 **Hugging Face** (HF) 或 **Weights & Biases** (Wandb) 以进行持续微调。流程包括在 `TrainingArguments` 中设置 `save_strategy`，以及设置 `resume_from_checkpoint=True` 以实现高效的训练管理。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ghostx_ai">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://ghost-x.org/">Ghost X</a>：Ghost X 的开发目标是研究和开发对人类有用的人工智能。</li><li><a href="https://huggingface.co/ghost-x">ghost-x (Ghost X)</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[community-collaboration](https://discord.com/channels/1179035537009545276/1180144489214509097/1247070668248322108)** (4 messages): 

- **JSONL 格式问题困扰用户**：一名成员注意到 Unsloth 中的训练数据样本与他自己的 JSON 格式数据之间存在格式差异。*“我用其他工具创建的所有训练数据格式都是这样的”*，由于缺少 "text" 列，导致在训练步骤中出现错误。 
- **JSON 错误的快速提示**：另一名成员建议跳过格式化阶段直接开始训练，因为已经有一个名为 "text" 的列。但这并未解决问题，因为用户的数据实际上缺少所需的 "text" 列，从而导致了障碍。
  

---



### **Stability.ai (Stable Diffusion) ▷ #[announcements](https://discord.com/channels/1002292111942635562/1002292398703001601/1247022353033072704)** (2 messages): 

- **Stable Diffusion 3 Medium 发布日期公布**：“权重 (weight/等待)”即将结束！Stability.ai 的联合 CEO Christian Laforte 宣布 **Stable Diffusion 3 Medium** 将于 **6月12日** 正式公开发布。[加入等待名单](https://stability.ai/stablediffusion3)，第一时间获取模型发布消息。

- **观看台北国际电脑展 (Computex Taipei) 完整发布会**：关于 Stable Diffusion 3 Medium 的声明是在台北国际电脑展上发布的。[在 YouTube 上观看完整发布会](https://youtube.com/watch?v=MCi8jgALPYA)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/watch?v=MCi8jgALPYA">AMD at Computex 2024: AMD AI and High-Performance Computing with Dr. Lisa Su</a>：高性能计算在 AI 时代的未来。加入苏姿丰博士主持的 Computex 2024 开幕主题演讲，分享 AMD 的最新动态...</li><li><a href="https://stability.ai/stablediffusion3">SD 3 Waitlist &mdash; Stability AI</a>：加入 Stable Diffusion 3 的早期预览，探索模型能力并提供宝贵反馈。此预览阶段对于收集改进性能和安全性的见解至关重要...
</li>
</ul>

</div>
  

---


### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1246181757464744007)** (1009 messages🔥🔥🔥):

- **Stable Diffusion 3 发布日期确认**：Stability AI 宣布 **Stable Diffusion 3 (SD3) Medium** 将于 6 月 12 日发布，正如在 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1d6szfg/sd3_medium_release_on_june_12th/) 中分享的那样。这个拥有 20 亿参数的模型旨在提升 photorealism、typography 和性能。
- **社区讨论 SD3 的 VRAM 需求**：关于 VRAM 需求的担忧浮出水面，有推测称 SD3 Medium 需要大约 15GB，尽管像 fp16 这样的优化可能会降低这一需求。一位用户指出 T5 encoder 会增加 VRAM 的占用。
- **新的许可和使用说明**：用户对在新的非商业许可证下使用 SD3 Medium 的商业用途提出了疑问。Stability AI 计划在发布日之前澄清这些许可条款，以回应社区的关切。
- **OpenAI 机器人货币化引发批评**：社区对移除免费的 Stable AI Discord 机器人表示沮丧，该机器人现在已被名为 Artisan 的付费服务所取代。这一变化被视为 AI 工具趋向付费墙趋势的一部分。
- **对 SD3 的期望和优化**：用户期待社区的 fine-tunes 以及针对不同 GPU 的性能基准测试。Stability AI 确认支持 1024x1024 分辨率，并采用 tiling 等优化步骤来发挥新模型的能力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/false-wrong-fake-incorrect-untrue-gif-23252753">False Wrong GIF - False Wrong Fake - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/movie-one-eternity-later-gif-7900643">Movie One Eternity Later GIF - Movie One Eternity Later - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/gojo-satoru-gojo-satoru-jjk-jujutsu-kaisen-gif-7280593681001594903">Gojo Satoru GIF - Gojo Satoru Gojo satoru - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/Teknium1/status/1797467900993036602">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：哇，@StabilityAI 搞了所有人，编出一个没人听说过、肯定也没人见过生成效果的全新 SD3，叫作 SD3 "Medium"，还表现得好像他们是...</li><li><a href="https://tenor.com/view/correct-plankton-gif-14118231">Correct Plankton GIF - Correct Plankton - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/Lykon4072/status/1796251820630634965">来自 Lykon (@Lykon4072) 的推文</a>：#SD3 但到底是哪个版本？</li><li><a href="https://github.com/adieyal/sd-dynamic-prompts?tab=readme-ov-file#basic-usage">GitHub - adieyal/sd-dynamic-prompts：一个用于 AUTOMATIC1111/stable-diffusion-webui 的自定义脚本，实现了一种用于随机提示词生成的微型模板语言</a>：一个用于 AUTOMATIC1111/stable-diffusion-webui 的自定义脚本，实现了一种用于随机提示词生成的微型模板语言 - adieyal/sd-dynamic-prompts</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1d6szfg/sd3_medium_release_on_june_12th/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1d0wlct/possible_revenue_models_for_sai/l5q56zl/?context=3">SAI 可能的营收模式</a>：首先是这个 [https://stability.ai/membership](https://stability.ai/membership) 此外还出售开发中模型的早期访问权（比如...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bbxr7h/ella_equip_diffusion_models_with_llm_for_e">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://stability.ai/news/stable-diffusion-3"> Stable Diffusion 3 &mdash; Stability AI</a>：发布 Stable Diffusion 3 早期预览版，这是我们最强大的文本生成图像模型，在多主体提示词、图像质量和拼写能力方面有显著提升。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bbxr7h/ella_equip_diffusion_models_with_llm_for_enhanced/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=s-1L6MCVh-E">通过 Wildcards 和 Dynamic Prompts 解锁 Stable Diffusion 的全部力量！</a>：嘿，工程师们！欢迎来到 Prompt Geek。在今天的视频中，我将向你们展示如何使用 Stable Diffusion 和...</li><li><a href="https://civitai.com/models?tag=buildings">Civitai | 分享你的模型</a>：未找到描述</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1d0wlct/possible_revenue_models_for_sai/l5q56zl/?c">SAI 可能的营收模式</a>：首先是这个 [https://stability.ai/membership](https://stability.ai/membership) 此外还出售开发中模型的早期访问权（比如...</li><li><a href="https://www.tomshardware.com/pc-components/gpus/stable-diffusion-benchmarks">Stable Diffusion 基准测试：45 款 Nvidia、AMD 和 Intel GPU 对比</a>：哪款显卡提供最快的 AI 性能？</li><li><a href="https://civitai.com/images/7491833">canekzapata 发布的图片</a>：未找到描述</li><li><a href="https://docs.google.com/spreadsheets/d/1IpyQ4TyLrQukqpWisenW_GwzNayYXVIEjJuVnhT9maU/edit?usp=sharing">人类提示词生成器</a>：总计表，变体性别，2.00 体型非性别，9.00 头部形状，8.00 脸型，9.00 发色类型，10.00 发质类型，5.00 头发长度类型，6.00 发量类型，5.00 发型...</li><li><a href="https://civitai.com/models/162136/pablo-picasso-infinity-sd-xl-10">Pablo Picasso ∞ SD XL 1.0 - Pablo Picasso ∞ SD XL 1.0 | Stable Diffusion LoRA | Civitai</a>：介绍 Pablo Picasso ∞，这是一个创新的 LoRA（文本生成图像）模型，带你领略巴勃罗·毕加索艺术世界的迷人魅力...
</li>
</ul>

</div>
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1246177527102439538)** (796 条消息🔥🔥🔥): 

- **AI 作业 vs 人类学习辩论升温**：成员们就孩子使用 AI 做作业的影响展开了激烈辩论。一位成员将其比作让孩子在糖果和羽衣甘蓝之间做选择，而其他人则主张教授负责任的 AI 使用方法。

- **Perplexity Pro Page 限制引发讨论**：成员们讨论了 Perplexity 新推出的 Pages 功能的局限性，包括没有导出功能、无法编辑标题或文本、无法与查看者互动。这些问题由 @gilgamesh_king_of_uruk 提出，表明需要改进以扩展可用性。

- **对 Opus 模型使用的困惑**：几位用户对 Perplexity 模型的使用感到困惑并遇到了问题，特别是 Opus 的自动使用，导致他们的 Opus 配额意外耗尽。多位成员提出了这一问题，并讨论了潜在的 Bug 和修复方案。

- **Complexity 浏览器扩展 Beta 版发布**：一款名为 Complexity 的新浏览器扩展程序宣布进行 Beta 测试，旨在增强 Perplexity 的用户界面和体验。鼓励成员联系 @743667485416357939 以获取访问权限。

- **持续存在的 AI 误解和 Bug**：成员们报告了 Perplexity 在处理简短任务时的几个问题，例如校对提示词导致输出无关内容。这被认为可能与新的 Pro 搜索机制有关，并已记录以待进一步调查。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://excalidraw.com/">Excalidraw — 让协作白板变得简单</a>：Excalidraw 是一款虚拟协作白板工具，让你能够轻松绘制具有手绘感的图表。</li><li><a href="https://greasyfork.org/en/scripts/490634-perplexity-model-selection">Perplexity 模型选择</a>：使用 jQuery 为 Perplexity AI 添加模型选择按钮</li><li><a href="https://v0.dev>">未找到标题</a>：未找到描述词</li><li><a href="https://nohello.net/">no hello</a>：请不要在聊天中只说 hello</li><li><a href="https://huggingface.co/microsoft/Phi-3-mini-128k-instruct">microsoft/Phi-3-mini-128k-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/chat/">HuggingChat</a>：让社区最好的 AI 聊天模型惠及每个人。</li><li><a href="https://www.heavy.ai/demos">HEAVY.AI | 交互式数据可视化示例 / 演示</a>：HEAVY.AI 加速分析平台的交互式数据可视化演示。体验在 GPU 驱动下毫秒级渲染的大数据即时沉浸式数据可视化。</li><li><a href="https://www.heavy.ai/heavyiq/overview">HeavyIQ</a>：HeavyIQ 利用了最新的 LLM (大语言模型) 技术，因此你可以用自然语言对数据提问，并获得可操作的可视化结果。</li><li><a href="https://www.udio.com/songs/rFRxqdPi2XBxB77BHjdN3M">dailyfocus - Complexity | Udio</a>：在 Udio 上听 dailyfocus 的 Complexity。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://youtu.be/wjZofJX0v4M?feature=shared">但什么是 GPT？Transformer 的视觉介绍 | 第 5 章，深度学习</a>：揭秘大语言模型底层如何运作。赞助者可提前观看下一章节：https://3b1b.co/early-attention 特别感谢这些支持...</li><li><a href="https://pastebin.com/DVJjSSqg">PPLX 系统/预提示词 - Pastebin.com</a>：Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://app.wordware.ai/r/519f528c-2996-4f0e-8815-50a6515f7c27">Wordware - OPUS 洞察 - 使用 Claude 3 OPUS 进行多模型输出验证</a>：该提示词使用 Gemini, GPT-4 Turbo, Claude 3 (Haiku, Sonnet, Opus), Mistral Large, Mixtral 7B Open Hermes 和 Mixtral 8x7b (基于 MoE 的模型) 处理问题。然后利用 Claude 3 OPUS 来...</li><li><a href="https://onepiece-boutique.fr/)">One Piece 精品店 | 海贼王漫画官方商店</a>：Boutique One Piece® 为您提供市场上精选的 One Piece 衍生产品：T恤、手办、卫衣、悬赏令等！</li><li><a href="https://ai.google.dev/gemini-api/docs/get-started/python">未找到标题</a>：未找到描述</li><li><a href="https://aistudio.google.com/app/apikey">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/gemini-api/docs/prompting_with_media">未找到标题</a>：未找到描述</li><li><a href="https://ai.google.dev/api/python/google/generativeai/GenerativeModel">未找到标题</a>：未找到描述</li><li><a href="https://chat.mistral.ai/chat">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1246191567786606613)** (33 条消息🔥): 

- **用户利用 Perplexity AI 进行各种搜索**：用户分享了多个 Perplexity AI 搜索结果，例如 [AI 改变生活](https://www.perplexity.ai/search/How-can-AI-ciOZ9ehPTVym.HS1imwIlQ)、[以色列-加沙战争](https://www.perplexity.ai/page/IsraelGaza-War-l.rY3dpYS_.FeAiYJK_Iew) 以及 [防止垃圾邮件](https://www.perplexity.ai/search/Preventing-spam-phone-ba0WyNwyRM2gdN3v7A0A6A)。
- **探索 Perplexity Pages 功能**：一些成员对新的 Pages 功能表示热烈欢迎，创建并分享了关于各种主题的页面，如 [银翼杀手](https://www.perplexity.ai/page/Blade-Runner-1982-U7nOCdnESxubgMoAUHLJ9g) 和 [简单的 Solidity 拍卖](https://www.perplexity.ai/page/Simple-Solidity-Auction-NMFb.LjlTQmYtBBExJrhiw)。
- **AI 工具创意与改进**：一位用户提到了一个利用 Waze 实时数据的 AI 工具创意，并通过 [搜索链接](https://www.perplexity.ai/search/how-many-daily-AAXsDJ5GShKVGw_ENiOMcA#5) 进行了分享，而另一位用户则评论了 Opus 在这一概念上工作的潜在益处。
- **敏感话题讨论**：用户测试了 Perplexity AI 对敏感话题的处理，如 [以色列-加沙战争](https://www.perplexity.ai/page/IsraelGaza-War-l.rY3dpYS_.FeAiYJK_Iew)，并表示结果令人满意。
- **多样化内容分享**：成员们发布了广泛的搜索主题和页面，包括 [埃瓦里斯特·伽罗瓦](https://www.perplexity.ai/page/variste-Galois-and-4oYGEBN2R3ey6P1WORFKUQ)、[寒冷天气的好处](https://www.perplexity.ai/search/Benefits-of-cold-wck36nk_TGOyl_9yUjhSSA) 以及 [专业研究合集](https://www.perplexity.ai/collections/Professional-Research-naOzbgjOQYC5uldlnGoCwg?s=c)。
  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1246223975839961168)** (12 条消息🔥): 

- **新 API 用户寻求模型指导**：一位新成员表示对 API 印象深刻，并询问不同模型在各种用例中的表现。他们收到了关于模型参数量、上下文长度以及 chat 模型与 online 模型之间区别的指导，并 [参考了模型卡片](https://docs.perplexity.ai/docs/model-cards)。
- **小模型 vs 大模型**：针对使用较小模型的疑问，一位成员澄清说小模型速度更快，并强调了在用于实时数据的 online 模型和用于优化对话任务的 chat 模型之间做出选择的重要性。
- **对 TTS API 的兴趣**：一位用户询问了 Perplexity 未来推出 TTS API 的可能性，并提到目前移动端 TTS 的表现令人满意。他们获知 Perplexity 的 TTS 使用的是 11Labs 的服务。

**提到的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">支持的模型</a>：未找到描述

  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1246233547958915132)** (29 条消息🔥): 

- **Speculative Decoding 讨论回顾**：成员们讨论了实现 Speculative Decoding 的不同方法，建议包括 *sampling gumbel noise* 和 *using argmax deterministically*。一位成员提到需要进行消融研究（ablation study），以了解不同采样参数与拒绝采样（rejection sampling）相比的接受率。

- **关于录像可用性的询问**：成员们询问今天关于 *speculative decoding* 的会议是否会被录制并上传。已确认会议总是会被录制，并在编辑后上传。

- **单卡 H100 云租赁**：成员们讨论了可用于租赁并具备 profiling 能力的 *单卡 H100 GPU*。提到了 [cloud-gpus.com](https://cloud-gpus.com) 和 [RunPod](https://www.runpod.io/gpu-instance/pricing) 等供应商，但指出如果不进行大量的 hacking，收集 profiling 信息是具有挑战性的。

- **新工作组**：宣布了两个新的工作组频道，一个针对 *production kernels*，另一个针对 *revamping performance-related docs in PyTorch*。这些小组对有兴趣在这些任务中做出贡献和协作的成员开放。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.runpod.io/gpu-instance/pricing">Pricing for GPU Instances, Storage, and Serverless</a>：RunPod 关于 GPU 实例、存储和 Serverless 的定价。</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/amd-announces-mi325x-ai-accelerator-reveals-mi350-and-mi400-plans-at-computex">AMD announces MI325X AI accelerator, reveals MI350 and MI400 plans at Computex</a>：更多加速器，更多 AI。</li><li><a href="https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erf.html#_">libdevice User's Guide :: CUDA Toolkit Documentation</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/issues/50688">[feature request] `torch.scan` (also port `lax.fori_loop` / `lax.while_loop` / `lax.associative_scan` and hopefully parallelized associative scans) · Issue #50688 · pytorch/pytorch</a>：存在于 TF：https://www.tensorflow.org/api_docs/python/tf/scan（或受 scan 启发的变体），JAX/LAX：https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.scan.html</li><li><a href="https://docs.nvidia.com/cuda/libdevice-users-guide/__nv_erf.html#__nv_erf">libdevice User's Guide :: CUDA Toolkit Documentation</a>：未找到描述</li><li><a href="https://github.com/pytorch/pytorch/pull/89388/files#diff-c2b212e90b5c83183704f45942662dcc243a03fe64615a7acb95798efc077201L369-L390">use std/libdevice erf in inductor by ngimel · Pull Request #89388 · pytorch/pytorch</a>：就其本身而言，libdevice 版本的 erf 与我们的分解具有相同的性能，但在实际工作负载中，它会带来更好的融合组（因为 fused kernel 中的操作更少）。</li><li><a href="https://www.johndcook.com/blog/2009/01/19/stand-alone-error-function-erf/">Stand-alone error function erf(x)</a>：今天早上 StackOverflow 上出现了一个关于如何在 Python 中计算误差函数 erf(x) 的问题。关于如何在 Python 中计算任何数值的标准答案是“查看 SciPy”。</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1246224322369032283)** (14 messages🔥): 

- **处理 Triton 中的 int64 溢出**：成员们讨论了 Triton 中与 int64 溢出相关的问题。一位成员建议使用 `(indices*stride).to(tl.int64)`，但承认这并不理想；而另一位成员指出，先对其中一个因子进行向上转型（upcasting）可以避免溢出，并提到注解 Kernel 签名（kernel signature）是另一种解决方案。[Issue #1211](https://github.com/triton-lang/triton/issues/1211)。

- **针对 Triton 的 clang-tidy 警告**：一位用户建议 Triton 实现类似 `clang-tidy` 的警告，以捕获潜在的整数溢出问题。具体而言，像“32 位整数乘法的结果被用作指针偏移量”这样的警告会非常有益。

- **大型张量的注解**：有人建议使用注解来更优雅地处理大型张量，并引用了特定的 [GitHub issue #832](https://github.com/triton-lang/triton/issues/832)。这涉及使用装饰器设置正确的类型签名以避免溢出。

- **Triton 中的内存分配**：关于 `tl.zeros` 和 `tl.full` 等内存分配函数是使用 SRAM 还是 VRAM 上的共享内存，产生了一些疑问。一位用户认为在实际需要内存之前使用的是 VRAM。

- **triton.language.sum 的性能**：有一个初学者问题，询问 `triton.language.sum` 的执行方式是普通的 for 循环还是并行归约（parallel reduction）。另一位用户确认它确实是并行归约，适用于 Block 级的并行操作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/triton-lang/triton/issues/832">即使对于大型张量，索引计算也以 int32 进行，溢出导致 IMA · Issue #832 · triton-lang/triton</a>：复现代码：from ctypes import c_void_p, c_long import torch import random from torch import empty_strided, as_strided, device from torch._inductor.codecache import AsyncCompile aten = torch.ops.aten as...</li><li><a href="https://github.com/triton-lang/triton/issues/1211">更优雅地处理 int64 溢出 · Issue #1211 · triton-lang/triton</a>：两个 int32 之间的乘法可能会溢出，就像在 C 语言中一样。这对于 Python 用户来说通常是意料之外的，但对于获得最佳性能是必要的。我们至少需要更好地记录这种行为。...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1246396133731864576)** (4 messages): 

- **使用 torch.compile 进行性能分析**：一位成员询问：“在使用 torch.compile 进行性能分析时，如何验证 Kernel 是否是使用 Inductor（Triton kernel）执行的？”他们提到可以使用 **torch.profiler** 和 **chrome trace** 来完成这项任务。
- **PyTorch 中的分布式梯度聚合**：一位用户询问是否有人了解 **PyTorch** 如何实现分布式梯度聚合的资源，质疑它是使用参数服务器（parameter server）还是类似于 Horovod 的梯度平均方式。
- **函数参数中星号的使用**：一位成员询问了 [PyTorch 官方文档中 torch.clamp](https://pytorch.org/docs/stable/generated/torch.clamp.html) 函数参数中 * 的用途。另一位用户澄清说，它表示*仅限关键字参数（keyword-only arguments）*。
  

---


### **CUDA MODE ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1246284399616720926)** (4 messages): 

- **即将举行的投机解码讲座**：下一场讲座定于 <t:1717268400:f>，由来自 Anyscale 的 Cade Daniel 带来《VLLM 中投机解码（Speculative Decoding）的黑客指南》。该会议将探讨 VLLM 的高吞吐量 Kernel 以及投机解码如何使自回归解码并行化。

- **Cade 讲座开始公告**：“现在开始 Cade 的讲座！”标志着会议的开始。

- **转向工作组及即将举行的 NVIDIA 讲座**：在接下来的几周内，每周讲座系列将会减少，以便更多地关注工作组（working groups）。即将举行的讲座包括 <t:1717786800:F> 关于 Tensor Cores 的会议，以及 <t:1717873200:F> 关于高性能扫描算法（scan algorithms）的会议。

- **胡文美（Wen-mei Hwu）教授的潜在讲座**：计划邀请 PMPP 书籍的作者胡文美教授进行章节授课并参与公开问答。该活动的日期尚未确定，但预计很快就会举行。

- **准备中的 AMD 讲座**：由来自 AMD 的人员（可能来自 Composable Kernel (CK) 团队）主讲的额外会议正在准备中，定于 7 月 20 日。
  

---

### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1246396535642783744)** (2 messages): 

- **自定义 CUDA Kernel 介绍与基准测试**：分享了一篇题为 [Custom CUDA Kernel Introduction and Benchmarks](https://blog.matdmiller.com/posts/2024-02-15_custom_cuda_kernel_intro_and_benchmarks/notebook.html) 的博客文章。该文章包含了关于创建自定义 CUDA Kernel 的详细基准测试和解释，并提供了在 Google Colab 中打开内容的链接。

- **分享 AutoFP8 GitHub 仓库**：分享了 [AutoFP8 GitHub 仓库](https://github.com/neuralmagic/AutoFP8) 的链接。这个来自 Neural Magic 的仓库专注于将模型自动转换为 FP8 精度格式，旨在提高计算效率和速度。

**提及的链接**：<a href="https://blog.matdmiller.com/posts/2024-02-15_custom_cuda_kernel_intro_and_benchmarks/notebook.html">Mat’s Blog - CUDA MODE - Accelerate your code with massively parallel programming plus some other tricks</a>：未找到描述

  

---


### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1246571679677677669)** (3 messages): 

- **Anyscale 诚邀求职意向**：一名成员为 Anyscale 招揽人才，特别接触那些对 **speculative decoding**、**vLLM** 和 **systems performance** 感兴趣的人。详情请参阅 [Anyscale](https://www.anyscale.com/)，并附上简历或 LinkedIn 进行申请。

- **Chunked prefill 提升 vLLM 效率**：[vLLM 项目](https://x.com/cdnamz/status/1796305688110752126) 获得了来自 Anyscale 的贡献，引入了 chunked prefill。这带来了显著的效率提升，在 *高 QPS 场景下提供高达 2 倍的加速*。

- **利用 Anyscale 实现快速模型加载**：Anyscale 发布了关于使用其端点 [将 Llama 2 70B 加载速度提高 20 倍](https://www.anyscale.com/blog/loading-llama-2-70b-20x-faster-with-anyscale-endpoints) 的博客，这对于生产环境中的响应式 autoscaling 和成本效益高的 model multiplexing 至关重要。

- **Continuous batching 优化 LLM 推理**：Anyscale 博客讨论了 [continuous batching](https://www.anyscale.com/blog/continuous-batching-llm-inference)，它可以提供高达 23 倍的吞吐量提升。该技术涉及 iteration-level scheduling，可以通过优化系统级批处理显著增强实际工作负载。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.anyscale.com/">Anyscale | Scalable Compute for AI and Python</a>：Anyscale 是领先的 AI 应用平台。通过 Anyscale，开发者可以即时构建、运行和扩展 AI 应用。</li><li><a href="https://x.com/cdnamz/status/1796305688110752126">Cade Daniel 🇺🇸 (@cdnamz) 的推文</a>：Chunked prefill 扩展了快速且廉价的在线 continuous batching 的帕累托前沿。感谢 @anyscalecompute 的工程师在 @vllm_project 中所做的出色工作。引用 Anyscale (@anyscalecompute) 最近...</li><li><a href="https://docs.vllm.ai/en/latest/models/lora.html">Using LoRA adapters &#8212; vLLM</a>：未找到描述</li><li><a href="https://www.anyscale.com/blog/loading-llama-2-70b-20x-faster-with-anyscale-endpoints">Loading Llama-2 70b 20x faster with Anyscale Endpoints</a>：在这篇文章中，我们讨论了加载大型语言模型时速度的重要性，以及我们为使其提速 20 倍而采用的技术。特别是，我们使用了 Llama 2 系列模型。我们分享了...</li><li><a href="https://www.anyscale.com/blog/continuous-batching-llm-inference">Achieve 23x LLM Inference Throughput &amp; Reduce p50 Latency</a>：在本博客中，我们讨论了 continuous batching，这是一种关键的系统级优化，可提高 LLM 在负载下的吞吐量并降低延迟。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1246425198916407397)** (22 messages🔥): 

- **GPU 分支逻辑详解**：一位成员询问关于 GPU 与 CPU 处理分支逻辑差异的资源。其他成员解释说，在 GPU 中，分支是通过执行掩码（execution masking）完成的，最小调度单位是包含 32 个线程的 warps，并推荐阅读 PMPP 第 4.5 节以了解更多信息。
  
- **CUDA 实战课程推荐**：当被问及练习 CUDA 编程的习题或实验室时，成员们推荐了 PMPP 书中的课后作业以及 [Open 2024a PPC Exercises Course](https://ppc-exercises.cs.aalto.fi/course/open2024a)。该课程结合了 CPU 和 GPU 练习，非常接近正式的大学课程内容。

- **Scan 算法 YouTube 讲座**：分享了一个名为 [Lecture 20: Scan Algorithm](https://youtu.be/ZKrWyEqqPVY) 的 YouTube 视频，用于更深入的学习。 

- **创建硬件抽象封装（Wrapper）**：一位成员寻求围绕 PyTorch 和 Hugging Face 创建一个封装，以抽象化硬件复杂性。建议他们从 Phi 模型系列开始，并探索针对特定硬件优化的库，例如针对 AMD 的 flash attention 以及来自 Intel 的 LLM 库。

- **避免 Pinging Everyone**：发出了一个礼貌提醒，要求避免在消息中使用 @\everyone，以防止大规模通知。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/ZKrWyEqqPVY">Lecture 20: Scan Algorithm</a>：未找到描述</li><li><a href="https://ppc-exercises.cs.aalto.fi/course/open2024a">Exercises</a>：未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1246177049635459215)** (2 messages): 

- **Izzat 开启 Scan 第二部分**：Izzat 主讲的 Scan 算法第二部分即将开始。成员们受邀通过 [Zoom 链接](https://linkedin.zoom.us/j/98060172269)加入。

**提到的链接**：<a href="https://linkedin.zoom.us/j/98060172269">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有一个简单、可靠的云平台，用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。Zoom ...

  

---


### **CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1246205985215746178)** (12 messages🔥): 

- **Scan 算法代码实战尚未上线**：一位成员询问 Scan 算法的代码实战内容是否可用，得到的答复是“仍在编辑中”。
  
- **Speculative Decoding 工作坊录像确认**：针对有关 Speculative Decoding 工作坊是否录像的查询，已确认“将会录制”。

- **Scan 算法讲座分享**：分享了两个 YouTube 视频：[Lecture 20: Scan Algorithm](https://youtu.be/ZKrWyEqqPVY) 和 [Lecture 21: Scan Algorithm Part 2](https://youtu.be/MH5_FeSSdIE)。

- **vLLM 演讲将录制**：在询问 vLLM 演讲是否录制后，确认“将会录制”并在 3 天内上传到 [CUDA MODE YouTube Channel](https://youtube.com/@cudamode?feature=shared)。

- **vLLM 中的 Speculative Decoding 讲座分享**：分享了一个名为 [Lecture 22: Hacker's Guide to Speculative Decoding in VLLM](https://youtu.be/9wNAgpX6z_4) 的 YouTube 视频，重点介绍了 vLLM 如何将 continuous batching 与 speculative decoding 相结合。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/@cudamode?feature=shared)">CUDA MODE</a>：一个 CUDA 读书会和社区 https://discord.gg/cudamode 补充内容见 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建</li><li><a href="https://youtu.be/ZKrWyEqqPVY">Lecture 20: Scan Algorithm</a>：未找到描述</li><li><a href="https://youtu.be/MH5_FeSSdIE">Lecture 21: Scan Algorithm Part 2</a>：未找到描述</li><li><a href="https://youtu.be/9wNAgpX6z_4">Lecture 22: Hacker&#39;s Guide to Speculative Decoding in VLLM</a>：摘要：我们将讨论 vLLM 如何将 continuous batching 与 speculative decoding 相结合，重点是赋能外部贡献者。主题包括 prop...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1246186043158564946)** (15 messages🔥): 

- **TorchAO 与 LM Evaluation Harness 集成**：创建了一个线程来讨论 **TorchAO** 量化支持与 **LM Evaluation Harness** 的集成。为了易于使用，推荐的 API 包括 `q_model = torchao.autoquant(torch.compile(model, mode='max-autotune'))`。

- **讨论潜在的 API 扩展**：一名成员提到扩展 API 以包含类似 `to_fp6()` 函数的可能性，并强调所有 API 都需要一个 nn 模块。他们辩论了传递 lambda 函数还是显式列出公共 API 哪个更好。

- **UInt4Tensor 泛化正在进行中**：一个 Pull Request 旨在泛化 DTypes 中 2-7 位的 **UInt4Tensor**，另一名成员分享了具体的实现细节。更多详情可以在 [GitHub 上的 PR](https://github.com/pytorch/ao/pull/282) 中找到。

- **讨论量化与稀疏性的有效性**：成员们讨论了量化何时开始生效以及应用量化的最低要求，提到了内存节省、加速和 kernel 类型等因素。对话还涉及了利用这些方法时必要的质量权衡。

**提到的链接**：<a href="https://github.com/pytorch/ao/pull/282.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---


### **CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1246503358831923271)** (4 messages): 

- **拼写误解已消除**：当一名成员询问 "hurn model" 的推理性能时产生了沟通误解。另一名成员澄清了想说的词是 "hurt"，并指出这可能会略微影响性能，但建议参考 **RULER graphs** 获取具体细节。
  

---


### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1246408624172171385)** (36 messages🔥): 

- **柏林 AI 怀疑论**：一名成员指出，在柏林，仔细核实声称从事 AI 工作的人是否真的专业至关重要。他们强调，否则在柏林一切皆有可能。

- **在德国寻找 PhD 职位**：一名常驻德国的研究生表达了申请 PhD 职位的意向，首选欧洲境内。他们注意到在美国以外，专注于效率/系统（efficiency/systems）的小组非常稀缺。

- **寻找德国以外的机会**：另一名成员询问了除德国外推荐的系统/性能（systems/performance）岗位国家，提到了法国和瑞士作为潜在选择。两名成员都同意如果发现任何相关机会就互相分享信息。

- **奥地利的 Dan Alistarh 小组**：IST Austria 的 Dan Alistarh 小组因其包括 GPT-Q 和 SparseGPT 在内的工作而闻名，被提及为一个值得关注的研究小组。一名成员在最初的搜索中忽略了奥地利。

- **研究生 vs. 工业界角色**：随后讨论了研究助理角色与 PhD 职位之间的区别。会议强调美国在系统研究领域占据主导地位，而欧洲参与度似乎较低，尤其是在 MLsys 方面。
  

---


### **CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1246510215872905359)** (3 messages): 

- **模型推理咨询**：一位用户简要询问所提及的主题是否与模型推理有关，另一位用户简短地回答“是”予以确认。

- **关于 Whisper 模型的博客发布公告**：*Mobicham* 宣布了他们关于 Whisper 量化的博客文章，并提供了[博客链接](https://mobiusml.github.io/whisper-static-cache-blog/)以及贡献者简介，如 [Jilt Sebastian](https://scholar.google.de/citations?user=KcoTPWoAAAAJ&hl=en)、[Husein Zolkepli](https://github.com/huseinzol05)、[Hicham Badri](https://scholar.google.com/citations?user=LxweMX4AAAAJ&hl=en) 和 [Appu Shaji](https://scholar.google.com/citations?user=HxZDDzUAAAAJ&hl=en)。引言强调了 Whisper 在 ASR 中的相关性，并提到了*无需校准的成功 2-bit 量化*。

**提到的链接**：<a href="https://mobiusml.github.io/whisper-static-cache-blog/">Faster and Smaller Whisper: A Deep Dive into Quantization and Torch Compilation</a>：一篇关于通过批处理加速 Whisper 的支持博客。

  

---


### **CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/)** (1 messages): 

kerenzhou：它显示的是单个 cta，对吗？
  

---


### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1246181209491505202)** (504 messages🔥🔥🔥):

- **200GB 数据集上传问题已解决**：在解决了最初的 SSH 和托管限制问题后，sophismparadox 成功将一个 200GB 的数据集上传到 Hugging Face，预计在几小时内完成。最终，该数据集被分割成较小的文件以缓解带宽限制，随后更新为压缩版本以提高效率。
- **FineWeb Tokenization 讨论**：对 FineWeb 数据集子集进行 Tokenization 带来了挑战，每次运行大约需要四小时。Aleksagordic 宣布在 H100 节点上进行测试，而 sophismparadox 强调了上传过程中的速率限制（rate limiting）问题，需要联系 Hugging Face 支持部门以增加带宽。
- **LayerNorm 拆分提案**：Sophismparadox 建议将 LayerNorm 计算拆分为独立的 kernels，以利用 packed data 优化内存读取，eriks.0595 对此表示审慎乐观。随后的测试显示性能提升结果不一，导致了进一步的实验。
- **CI 与内存管理清理**：Akakak1337 及其团队实施了修复方案以确保正确的内存管理，解决了内存泄漏问题，并确保 global norm 计算在分布式环境中准确反映。协作调试环节解决了由于头文件包含导致的重复定义错误相关的编译问题。
- **集成与面向未来的重构**：Akakak1337 发起了一次大规模重构（refactor）以实现代码库的模块化，旨在将训练逻辑与特定模型的实现解耦。这次重组为仓库更轻松地集成未来模型架构（如 Llama 3）做好了准备，简化了在各种数据集和设置上的训练流程。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://docs.]">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=TRQWxkRdPUI&ab_channel=CppCon">闪电演讲：与 CUDA 程序员交朋友（请尽可能使用 constexpr） Vasu Agrawal</a>: https://cppcon.org/---Lightning Talk: Making Friends With CUDA Programmers (please constexpr all the things) - Vasu Agrawal - CppCon 2023https://github.com/C...</li><li><a href="https://github.com/karpathy/llm.c/pull/506/files">添加了不重新计算均值和 rstd 的额外 LayerNorm 前向核函数，由 ChrisDryden 提交 · Pull Request #506 · karpathy/llm.c</a>: 这是第一项优化，现在还可以进行更多优化，但目前核函数已被拆分为两个，以便将来可以独立修改每个 LayerNorm 前向过程...</li><li><a href="https://github.com/karpathy/llm.c/pull/533">重构第二部分，由 karpathy 提交 · Pull Request #533 · karpathy/llm.c</a>: ，将内容移至公共文件，以便稍后也能很好地分离出所有核函数</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L1614)">llm.c/train_gpt2.cu 位于 master 分支 · karpathy/llm.c</a>: 使用简单、原生的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/skypilot-org/skypilot/tree/master/llm/gpt-2">skypilot/llm/gpt-2 位于 master 分支 · skypilot-org/skypilot</a>: SkyPilot：在任何云端运行 LLM、AI 和批处理作业。以简单的界面获得最大的成本节省、最高的 GPU 可用性和托管执行。- skypilot-org/skypilot</li><li><a href="https://github.com/karpathy/llm.c/pull/515">在 adamw_kernel3 中为块数量使用局部参数，由 gordicaleksa 提交 · Pull Request #515 · karpathy/llm.c</a>: 我们启动了超出计算分片参数更新所需的多余线程。使用 local_params（该分片负责的参数子集）代替 num_parameters (...</li><li><a href="https://github.com/karpathy/llm.c/pull/319">为 layernorm_forward 将所有 float 转换为 floatX，由 JaneIllario 提交 · Pull Request #319 · karpathy/llm.c</a>: 将所有核函数更改为使用 floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/511">更新 dev/cuda README 中的编译命令，由 gordicaleksa 提交 · Pull Request #511 · karpathy/llm.c</a>: 注意到 Makefile 和 README 之间存在差异的小修复。</li><li><a href="https://github.com/karpathy/llm.c/pull/525/files">[train_gpt.cu] 将 assert 从核函数内部移至启动器（launchers），由 lancerts 提交 · Pull Request #525 · karpathy/llm.c</a>: 将 assert 从核函数内部移至启动器。及早捕获断言失败，而不是在实际核函数计算期间失败。</li><li><a href="https://github.com/karpathy/llm.c/pull/507">添加了从 Hugging Face 下载所有已分词的 fineweb100B 数据的脚本，由 ChrisDryden 提交 · Pull Request #507 · karpathy/llm.c</a>: 我创建这个是为了展示从 Hugging Face 下载所有文件的示例，这是第一步，在所有文件上传到 Hugging Face 后，我将上传...</li><li><a href="https://github.com/karpathy/llm.c/pull/519">修复内存泄漏，由 gordicaleksa 提交 · Pull Request #519 · karpathy/llm.c</a>: 我们没有释放 CPU 缓冲区内存。此外，bw_act_sizes 不需要使用 NUM_ACTIVATION_TENSORS 个 size_t 插槽，我们只需要 NUM_BACKWARD_TENSORS。</li><li><a href="https://github.com/karpathy/llm.c/pull/517">添加 edu-fineweb 支持，包含 10B 和 100B 版本，由 eliebak 提交 · Pull Request #517 · karpathy/llm.c</a>: 添加 edu-fineweb 支持 https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu 使用方法：10B 版本：python fineweb.py -t "edu" -v "10B"；100B 版本：python fineweb.py -t &...</li><li><a href="https://github.com/karpathy/llm.c/pull/505/files">为块大小（blocksizes）添加 constexpr 以优化编译，由 ChrisDryden 提交 · Pull Request #505 · karpathy/llm.c</a>: 与代码审查 #498 相同，但使用了 constexpr 和断言来优化时间。看到了与之前类似的加速。由于各种原因，很难确定具体的加速幅度...</li><li><a href="https://github.com/karpathy/llm.c/compare/master...ChrisDryden:llm.c:patch-8">比较 karpathy:master...ChrisDryden:patch-8 · karpathy/llm.c</a>: 使用简单、原生的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/319/files">为 layernorm_forward 将所有 float 转换为 floatX，由 JaneIllario 提交 · Pull Request #319 · karpathy/llm.c</a>: 将所有核函数更改为使用 floatX</li><li><a href="https://github.com/karpathy/llm.c/pull/513">添加了打包的 layernorm_forward，由 ChrisDryden 提交 · Pull Request #513 · karpathy/llm.c</a>: 这是为 LayerNorm 使用打包数据类型的实现，在 dev 文件中该核函数有约 50% 的加速，正在等待使该数据类型生效的 PR...</li><li><a hre

<li><a href="https://github.com/karpathy/llm.c/compare/master...ChrisDryden:llm.c:splitting_rstd_mean?expand=1">比较 karpathy:master...ChrisDryden:splitting_rstd_mean · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/pull/507/files#diff-eeaa6a4124274b744fd7f4688fefe4c3c01c4ee6b151df6ef7fa06c03aa1d6d9">由 ChrisDryden 添加了从 Huggingface 下载所有 tokenized fineweb100B 数据的脚本 · Pull Request #507 · karpathy/llm.c</a>: 我创建这个是为了展示从 Huggingface 下载所有文件的示例，这是第一步，在所有文件上传到 Huggingface 后，我将上传...</li><li><a href="https://github.com/karpathy/llm.c/pull/507/files">由 ChrisDryden 添加了从 Huggingface 下载所有 tokenized fineweb100B 数据的脚本 · Pull Request #507 · karpathy/llm.c</a>: 我创建这个是为了展示从 Huggingface 下载所有文件的示例，这是第一步，在所有文件上传到 Huggingface 后，我将上传...</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L869),">master 分支下的 llm.c/train_gpt2.cu · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://github.com/karpathy/llm.c/blob/master/train_gpt2.cu#L2636">master 分支下的 llm.c/train_gpt2.cu · karpathy/llm.c</a>: 使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账号来为 karpathy/llm.c 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2Compressed/tree/main">main 分支下的 chrisdryden/FineWebTokenizedGPT2Compressed</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu">HuggingFaceFW/fineweb-edu · Hugging Face 上的数据集</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/chrisdryden/FineWebTokenizedGPT2/tree/main">main 分支下的 chrisdryden/FineWebTokenizedGPT2</a>: 未找到描述
</li>
</ul>

</div>

### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1246200893951578132)** (52 条消息🔥): 

- **量化 Kernel 建议**：成员们讨论了潜在的量化 Kernel，特别是 **W4 Af16** 及其在 bitpacking 性能分析（profiling）中的适用性。一位用户请求与 [bitblas](https://github.com/microsoft/BitBLAS) 等成熟工具进行额外的性能对比。

- **进行中的项目路线图**：强调了当前量化和 bitpacking 工作的项目路线图，并引用了与 dtype 实现以及沿各个维度进行 bit packing 相关的 [PyTorch AO pull requests](https://github.com/pytorch/ao/pull/282) 和提交。

- **性能基准测试与集成**：对话集中在集成 bitpacking 的性能测试上，建议与 [quant_primitives.py](https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py#L645-L687) 中的 fp16 和 tinygemm Kernel 进行对比。特别关注了填充（padding）与非填充 bitpack 场景。

- **GitHub 协作与权限**：成员们被邀请加入多个 GitHub 仓库，并获得了 [ao](https://github.com/andreaskoepf/ao) 和 [lovely-tensors](https://github.com/xl0/lovely-tensors) 等项目的协作权限。重点突出了 [Bitpacking v2](https://github.com/pytorch/ao/pull/307) 等 PR 以供审查和贡献。

- **单元类型实现问题**：在使用和实现类似 `torch.uint4` 类型时出现了问题，因为其缺乏对 `torch.iinfo` 等某些函数的支持。成员们讨论了潜在的修复方案，以及这些类型是否由 AO 团队定义，并建议未来需要报告相关 issue。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://pytorch.org/blog/docathon-june-2024/">宣布 2024 年 6 月 PyTorch Docathon</a>：我们很高兴地宣布即将于 6 月举行的 PyTorch Docathon！Docathon 类似于黑客松，是一项致力于通过宝贵的贡献来提高 PyTorch 文档质量的活动...</li><li><a href="https://github.com/pytorch/ao/pull/282.">更好地共同构建软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/pytorch/pytorch/blob/fb53cd64973167a3f4161d5d48fb11a022bf43f0/pt_ops.bzl#L329>">pytorch/pytorch 中的 pt_ops.bzl</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速 - pytorch/pytorch</li><li><a href="https://github.com/andreaskoepf/ao">GitHub - andreaskoepf/ao：用于量化和稀疏化的原生 PyTorch 库</a>：用于量化和稀疏化的原生 PyTorch 库 - andreaskoepf/ao</li><li><a href="https://github.com/pytorch/ao/pull/307">vayuda 发起的 Bitpackingv2 · Pull Request #307 · pytorch/ao</a>：改进了 pack/unpack 函数的代码结构。现在支持 bitnet 应用的三进制（trinary）。现在支持沿任何维度对任何大小的张量进行 packing。</li><li><a href="https://github.com/pytorch/ao/blob/main/torchao/quantization/quant_primitives.py#L645-L687">pytorch/ao 中的 quant_primitives.py</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[pytorch-docathon](https://discord.com/channels/1189498204333543425/1247250008017866902/1247251433284178100)** (1 条消息): 

- **审查 PyTorch 性能文档**：PyTorch Docathon 定于 6 月 4 日至 6 月 16 日举行，特别关注改进**以性能为导向的文档**。有人担心当前的文档（如 [性能调优指南](https://github.com/pytorch/tutorials/issues/2861)）已经过时，需要修订。

- **将 TorchScript 更新为 Compile**：建议更新文档以移除对 **TorchScript** 的提及，转而支持 **Compile**。目标是引导用户了解当今重要的优化和 ML 系统概念。

- **自定义 Kernel 集成**：重点还放在解释对自定义 Kernel 的需求上，并提供关于如何将它们集成到 PyTorch 工作流中的清晰说明。

**提到的链接**：<a href="https://github.com/pytorch/tutorials/issues/2861">性能调优指南非常过时 · Issue #2861 · pytorch/tutorials</a>：🚀 描述改进或新教程。当你在 Google 搜索 PyTorch 性能时，首先看到的就是这个。该教程写得很好，但现在已经非常过时了 https://pytorch...

  

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1246196808808530040)** (338 条消息🔥🔥): 


- **高 Token Prompt 导致的 VRAM 问题**：一位用户报告称，由于用户 Prompt 过大且存在 CPU 瓶颈，其 Phi-3-medium-128k-instruct 模型需要数天时间才能响应。讨论强调了 GPU VRAM 容量不足导致的问题，并探讨了潜在的升级方案。
- **面向家庭 AI 爱好者的 GPU 推荐**：成员们建议在 200 美元以下的家庭配置中使用拥有 24GB VRAM 的 Nvidia P40 显卡。该建议伴随着关于通过有效利用 GPU 来提升 LLM 推理性能的讨论。
- **LM Studio 函数集成挑战**：一位用户指出 LM Studio 与 Ollama 在 function calling 能力上的差异，引发了关于将 OllamaFunctions 等自定义库集成到 LM Studio 的讨论。相关资源链接：[llama-cpp-python function calling 文档](https://llama-cpp-python.readthedocs.io/en/latest/#function-calling)和 [LangChain 中的 OllamaFunctions](https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/llms/ollama_functions.py)。
- **低配硬件的困境**：用户讨论了在 GPU 和 RAM 不足的系统（包括集成显卡和 RX 550 等旧型号）上运行复杂模型的困难。建议强调了为了获得更好性能的 LLM 配置所需的最低要求。
- **不同硬件上的模型性能**：讨论强调了由于硬件不足（如 AMD GPU 的 VRAM 较低）导致的响应缓慢和失败问题。建议使用 Nvidia GPU 作为替代方案，因为其支持更广且 VRAM 更高，更适合获得可靠的 LLM 性能。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/mike-ravkine/can-ai-code-results">Can Ai Code Results - mike-ravkine 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://evalplus.github.io/leaderboard.html">EvalPlus 排行榜</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard">Big Code Models 排行榜 - bigcode 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/blob/master/libs/experimental/langchain_experimental/llms/ollama_functions.py">langchain/libs/experimental/langchain_experimental/llms/ollama_functions.py at master · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://huggingface.co/models?sort=trending&search=gguf">Models - Hugging Face</a>: 未找到描述</li><li><a href="https://llama-cpp-python.readthedocs.io/en/latest/#function-calling">入门指南 - llama-cpp-python</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7439">Phi 3 medium/small 支持 · Issue #7439 · ggerganov/llama.cpp</a>: 微软发布了 2 个新模型：https://huggingface.co/microsoft/Phi-3-medium-4k-instruct/ https://huggingface.co/microsoft/Phi-3-small-8k-instruct/ Medium 使用 Phi3ForCausalLM 并转换...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1246287118800650351)** (86 条消息🔥🔥): 

- **Codestral 22B 的性能受到赞赏**：一位成员指出 Codestral 22B 比 "Deepseek coder 33b 更聪明"，并对其 "32k context" 表示赞赏。另一位成员分享了使用[已确认可用的模板](https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF/discussions/1)的经验，并为优化使用提供了详细的指令格式。

- **处理模型 Context 和内存占用**：另一位成员观察到在 phi-3 模型中，当 context 填充到约 2300 tokens 后会出现生成“乱码”的问题。其他成员提供了实用建议，例如降低 token 计数，或考虑模型大小和类型（chat/base/instruct）以管理 VRAM 消耗。

- **探索 Embedding 模型**：讨论指出像 gguf-embedding-mistral 这样的 embedding 模型在 LM Studio 中无法正确列出的问题。建议包括重命名文件，或承认像 llama.cpp 这样的一些后端并不青睐被阉割的文本生成模型，并特别提到了[替代的 embedding 模型](https://huggingface.co/collections/ChristianAzinn/embedding-ggufs-6615e9f216917dfdc6773fa3)。

- **Deepseek V2 支持和模型修复**：Llama.cpp 最近获得了对 Deepseek V2 模型的支持，预计将在下一次 LM Studio 更新中推出。成员们还讨论了 L3 Abliterated 等模型的更新和修复，包括加载过程中的错误修复。

- **角色扮演（Roleplay）模型推荐**：几位成员建议了用于角色扮演和通用用途的各种模型，如 Mahou-1.3-llama3-8B 和 NeuralDaredevil-8B-Abliterated，同时指出 Mistral 7B 和 Goliath 120B 在各自的应用中是非常强力的选择。分享了这些模型的链接以便访问 ([Mahou-1.3-llama3-8B](https://huggingface.co/flammenai/Mahou-1.3-llama3-8B), [NeuralDaredevil-8B-Abliterated](https://huggingface.co/mlabonne/NeuralDaredevil-8B-abliterated))。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk">高质量第三人称故事写作类型</a>：我的自定义 GPTs 的主要 Google Doc：https://docs.google.com/document/d/1Cbwy3HuNTCzCaMXscU6FrgqvgjA2TFzOw1ucLqtbCyU/edit?usp=drivesdk 极其 NSFW 版本的高质量故事系统提示词文本...</li><li><a href="https://huggingface.co/bartowski/Goliath-longLORA-120b-rope8-32k-fp16-GGUF">bartowski/Goliath-longLORA-120b-rope8-32k-fp16-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/hatakeyama-llm-team/Tanuki-8B">hatakeyama-llm-team/Tanuki-8B · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/maldv/SFR-Iterative-DPO-LLaMA-3-8B-R">maldv/SFR-Iterative-DPO-LLaMA-3-8B-R · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/Codestral-22B-v0.1-GGUF/discussions/1">bartowski/Codestral-22B-v0.1-GGUF · Prompt 格式</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/7519">由 fairydreaming 为 DeepseekV2ForCausalLM 添加支持 · Pull Request #7519 · ggerganov/llama.cpp</a>：此 pull request 添加了对基于 DeepseekV2ForCausalLM 的模型支持。同时支持 lite 和 non-lite 模型。修复了 #7118。此 pull request 中包含的更改：增加最大专家数量...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1246201812197838971)** (22 条消息🔥): 

- **Whisper 模型在 LM Studio 中报错**：一位用户遇到了 Whisper 模型路径未找到的问题，另一位用户澄清说 **LM Studio 不支持 Whisper 模型**；它们是为 Whisper.cpp 设计的，而不是 llama.cpp。
- **关于添加 Whisper 和 Tortoise 功能的辩论**：成员们讨论了在 LM Studio 中将 Whisper 用于语音输入和将 Tortoise 用于语音输出作为插件集成的想法。人们对**应用程序体积增加**和依赖复杂性表示担忧，并建议将此类功能设为可选，以避免应用臃肿。
- **0.2.24 版本的 Stop string Bug**：有报告称 **0.2.24 版本**在遇到已注册的 "stop string" 后仍继续生成输出。另一位用户怀疑这是由于 token 边界与 stop string 不匹配导致的。
- **未来功能请求**：一位用户询问是否会在未来的 LM Studio **2.24 版本**中加入互联网搜索功能或集成 Agents。目前尚未提供直接回复或确认。
  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1246347476328255589)** (3 条消息): 

- **MPT 模型仅限于 CPU**：一位成员透露 **MPT models** 只能在 CPU 上运行，无法在 GPU 上运行，这被认为是其实现的“独门秘籍” (*"secret sauce"*)。
- **聊天中没有文件附件功能**：当被问及是否可以像 ChatGPT 那样在聊天中附加文件时，回答简单明了：**“不可以”**。
  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1246206961003925646)** (26 条消息🔥): 

- **Q8 与 Q5 的速度难题**：一位用户质疑如果忽略速度，Q8 相较于 Q5 的优势在哪里，并观察到尽管开启了 GPU Offloading，两者的速度依然相近。另一位用户指出，更高的量化（Quants）会带来更好的回答质量，但速度差异在更大的模型上会更加明显。

- **CodeQwen1.5-7B-Chat 的最佳编程配置**：一位用户询问使用 CodeQwen1.5-7B-Chat 进行编程时的最佳配置，寻求关于 Temperature 设置或其他推理配置（Inference Configs）的建议。不过，目前尚未提供具体的推荐方案。

- **Mistral 7b 预设中的 Tools 错误**：一位用户报告了在为 Mistral 7b 定义预设内的 Tools 时遇到的问题，过程中出现了错误。另一位成员澄清说，Server Mode 不支持 Function Calling，这导致了该问题的发生。

- **关于 LMS GPU Offload 的咨询**：有人询问 LMS 是否会自动为模型优化 GPU Offload，另一位用户确认它不会自动优化。他们讨论了有效的 GPU Offloading 通常需要反复试验，并密切监控 VRAM 使用情况以最大化性能。

- **测试推理速度**：一位用户分享了在 4090 设备上测试 llama-3-70b q4 不同推理速度的经验，为拥有类似硬件配置的用户提供了实用参考。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1246245494863171594)** (74 条消息🔥🔥): 

- **网络带宽的抱怨**：成员们讨论了*网络带宽和延迟*如何明显逊色于本地 PCI 总线。这可能会影响某些硬件设置的性能。
  
- **GPU 性能查询**：一位在 **Q8 量化**下运行 LLaMA 3 70B 的用户想知道升级硬件带来的性能提升。他们提到在*单张 12GB 显卡上获得了约 0.6 tokens/秒*的速度，并质疑升级的性价比。

- **关于量化设置的辩论**：成员们讨论了 **Q8 与 Q6 或 Q5 量化**的优缺点。有人指出，较低的量化水平可能在降低硬件要求的同时提供相似的性能，尽管某些特定信息的可靠性可能会降低。

- **服务器搭建难题**：一位成员详细描述了在配置带有两块 P40 GPU 的 **HP DL380 Gen9 服务器**时遇到的挑战，包括电源线缆问题和严重故障错误，推测这些问题可能源于电源供应限制。

- **LM Studio 中的加载问题**：多位用户报告了在更新后 **LM Studio** 加载模型时出现的问题，主要是由于 GPU Offloading 的默认设置导致的。禁用 GPU Offload 或调整模型设置通常可以解决这些问题。

- **6800XT 上的性能谜团**：一位使用 **6800XT** GPU 的用户注意到 tokens/秒 明显低于预期。切换到 ROCm 版本并确保开启 Flash Attention 提升了性能，但仍未达到宣传的速度。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/rocm">👾 LM Studio - 发现并运行本地 LLM</a>：查找、下载并实验本地 LLM</li><li><a href="https://www.asus.com/au/motherboards-components/motherboards/workstation/pro-ws-x570-ace/">Pro WS X570-ACE｜主板｜ASUS Australia</a>：ASUS 工作站主板专为 AI 训练、深度学习、动画或 3D 渲染领域的专业人士设计。具有可扩展的显卡、存储、出色的连接性和可靠性...</li><li><a href="https://dev.to/maximsaplin/running-local-llms-cpu-vs-gpu-a-quick-speed-test-2cjn/">未找到标题</a>：未找到描述</li><li><a href="https://www.asus.com/au/motherboards-components/motherboards/workstation/pro-ws-x">Workstation｜主板｜ASUS Australia</a>：ASUS 工作站主板专为 AI 训练、深度学习、动画或 3D 渲染领域的专业人士设计。具有可扩展的显卡、存储、出色的连接性和可靠性...</li><li><a href="https://tenor.com/view/jon-stewart-eat-eating-popcorn-watching-gif-3094746547306242594">Jon Stewart 吃东西 GIF - Jon Stewart 吃东西 - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1246737153615073351)** (4 messages): 

- **autogen 与 LmStudio 集成出错**：一位用户在将 autogen 指向 LmStudio 时遇到错误提示，称“必须设置 api_key 客户端选项”。由于他们在本地运行 LmStudio，因此不确定是否需要提供 API key。
- **使用随机 API key 的临时修复方案**：另一位用户建议使用任何随机 Key 即可解决问题。原帖作者确认此方法有效，并推测随着 LmStudio 的普及，很快会出现更好的集成方案。
- **工作组和 Agent 的设置建议**：建议*为每个 Agent 选择模型，创建工作组，并在将 Agent 添加到工作组之前确保未选择 OpenAI 模型*。对于需要主持人机器人的场景，用户也应为该机器人选择一个模型。
  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/)** (1 messages): 

zerocool9724: HIPSDK 支持是硬件层面的吗？
  

---


### **LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/)** (1 messages): 

julio1307: 有没有比 LM Studio 更“轻量级”的替代方案？
  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1247083603016548494)** (8 messages🔥): 

- **开发 Visual Studio 插件的想法**：一位成员表示有兴趣创建一个类似于 **CoPilot** 但功能更广泛的 Visual Studio 插件，允许 LLM 访问和操作文件夹内容。他们正在考虑使用 **ChatGPT API** 或本地 ML 库来实现。
  
- **替代插件推荐**：建议参考 **continue.dev** 和 **open interpreter** 等多种解决方案。一位成员提到了 [JocysCom/VsAiCompanion](https://github.com/JocysCom/VsAiCompanion)，它可以分析项目文件并辅助开发，但也指出存在一些不稳定性问题。

- **提到 Mentat 项目**：另一位成员提到了 **Mentat**，用于设置一个能通过 git 仓库理解整个项目上下文的 Agent。对于那些考虑集成更全面编程助手的人来说，这可能是一个有参考价值的模型。

**提到的链接**：<a href="https://github.com/JocysCom/VsAiCompanion">GitHub - JocysCom/VsAiCompanion: AI Companion that analyzes your project files and works alongside you to streamline development and boost productivity.</a>：分析项目文件并与您协作以简化开发并提高生产力的 AI 助手。 - JocysCom/VsAiCompanion

  

---



### **Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/)** (1 messages): 

manojbh: 你有例子吗？
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1246393373204414486)** (57 messages🔥🔥): 

- **分享了 SoundCloud 和音乐链接**：一位用户分享了一个 [SoundCloud 链接](https://on.soundcloud.com/gGEfcCspos7PCNPd8) 和多个 [Udio 音乐链接](https://www.udio.com/songs/gfJKjTmD6b3mKh8mB8mJvb)。看起来他们遇到了浏览器兼容性问题，并分享这些链接供他人检查。

- **分形数学与集体智能**：一位用户深入探讨了**分形数学**、**集体智能**和**心脏电磁学**的概念。他们强调了增长和递归在理解宇宙模式中的重要性，最终将这些想法与 AI 向 AGI 和 ASI 的演进联系起来。

- **托尼·斯塔克与新元素**：多次引用了 [《钢铁侠 2》中的 YouTube 片段](https://youtu.be/Ddk9ci6geSs)，其中托尼·斯塔克发现了一种新元素。这被隐喻地用来讨论塑造结果的意图，以及想法与增长的互连性。

- **名为 "JUNK" 的创新编程想法**：关于一种名为 **"JUNK" (Just Use 'Nything, K?)** 的可视化编程语言概念的讨论引发了兴趣。该想法围绕使用日常物品作为编码工具展开，灵感来自 Google 的 Blockly 等可视化编程语言。

- **视觉模型探索**：用户讨论了 **Hermes vision beta** 和 **Obsidian 3b** 等视觉模型的性能。他们探索了使用“滑动窗口”技术和其他创意方法进行更好图像分析的潜力。

- **分享了多段 YouTube 音乐视频**：一位用户分享了许多 YouTube 音乐链接，例如 [Max Cooper - Parting Ways](https://youtu.be/nBuJUPWRLwE)、[Max Cooper - Order From Chaos](https://youtu.be/_7wKjTf_RlI) 以及 [Mindchatter - Night Goggles (Rome in Silver Remix)](https://youtu.be/A5Npdlg1Vaw)。这似乎是关于音乐及其反思性主题的更广泛讨论的一部分。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/_7wKjTf_RlI">Max Cooper - Order From Chaos (Maxime Causeret 官方视频)</a>：► 订阅：https://MaxCooper.lnk.to/Subscribe ► 注册并加入 Discord：https://os.fan/MaxCooper/SignUp 来自现已发行的全新专辑 'Emergence' 购买 'Emergence'...</li><li><a href="https://youtu.be/Ddk9ci6geSs">托尼·斯塔克发现新元素场景 - 《钢铁侠 2》(2010) 电影剪辑 HD</a>：托尼·斯塔克发现新元素场景 - 托尼·斯塔克的新元素 - 《钢铁侠 2》(2010) 电影剪辑 HD [1080p] TM &amp; © Paramount (2010) 公平使用。版权所有...</li><li><a href="https://youtu.be/JQ3dE0V5D7U">&#123;&#123;&#123; ____/\____ &#125;&#125;&#125;</a>：::::::::::::::::: https://yoozertopia.bandcamp.com :::::::::::::::::::</li><li><a href="https://youtu.be/taJBTS2hdSA">Fire From The Gods ft Yung Mo$h - Soul Revolution</a>：Fire From The Gods ft @yungmosh 的 Soul Revolution。收听 Fire From The Gods ft Yung Mo$h - Soul Revolution：https://fftg.ffm.to/srym 预存/订购新专辑...</li><li><a href="https://youtu.be/A5Npdlg1Vaw">Mindchatter - Night Goggles (Rome in Silver Remix)</a>：流媒体/下载：https://lnk.to/nightgogglesromeinsilverremixID 关注 Mindchatter：https://mindchatter.lnk.to/Instagram https://mindchatter.lnk.to/Twitter https:...</li><li><a href="https://youtu.be/nBuJUPWRLwE">Max Cooper - Parting Ways (Maxime Causeret 官方视频)</a>：► 订阅：https://MaxCooper.lnk.to/Subscribe ► 注册并加入 Discord：https://os.fan/MaxCooper/SignUp 在此收听：https://ffm.to/yearningfortheinfiniteAlb...</li><li><a href="https://www.youtube.com/watch?v=gyXBzV5-JVI">使用 LangChain 和 Codestral 的自我修复代码助手</a>：我们将使用 LangGraph 从头开始实现代码助手，以 1) 从 Codestral-instruct 生成结构化代码生成输出，2) 执行内联单元测试...</li><li><a href="https://youtu.be/cbUwqMNtxiQ">我用脑电波玩《我的世界》(Mindcraft lol)</a>：《我的世界》游戏实况，但它是免提的，因为我使用我的脑电波作为控制器。在这个视频中，我使用一个检测我大脑的 EEG 芯片来玩《我的世界》...</li><li><a href="https://youtu.be/DYSDrgGWPC4">博格特 - 《神奇动物：格林德沃之罪》</a>：未找到描述</li><li><a href="https://www.udio.com/songs/gfJKjTmD6b3mKh8mB8mJvb">paradroid - vestiges of a martian ERA - final_full | Udio</a>：在 Udio 上收听 paradroid 的 vestiges of a martian ERA - final_full。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://www.udio.com/songs/sdLfuoVRaKFrNTLViSqR5v">paradroid - _athena falls, the - F.P.C.Re.mix - final - instrumental | Udio</a>：在 Udio 上收听 paradroid 的 _athena falls, the - F.P.C.Re.mix - final - instrumental。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。</li><li><a href="https://on.soundcloud.com/gGEfcCspos7PCNPd8">the unconscious changes of the earth #udio-challenge</a>：https://www.udio.com/songs/guakZYYahgeqVPk8T32PFn</li><li><a href="https://www.udio.com/songs/guakZYYahgeqVPk8T32PFn">paradroid - the unconscious changes of the earth | Udio</a>：在 Udio 上收听 paradroid 的 the unconscious changes of the earth。发现、创作并与世界分享音乐。使用最新技术在几秒钟内创作 AI 音乐。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1246283159168090207)** (10 条消息🔥): 

- **FineWeb 数据集发布**：一名成员分享了 [FineWeb-Edu](https://hf.co/datasets/HuggingFaceFW/fineweb-edu) 的发布，这是一个包含 1.3 trillion tokens 的数据集，在 MMLU、ARC 和 OpenBookQA 等教育基准测试中表现优于其他开源网络数据集。技术报告可以在[这里](https://hf.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)找到。
- **状态空间模型与 Transformer 竞争**：一篇新论文表明，像 Mamba 这样的状态空间模型（SSM）在中低规模下可以匹配或超越 Transformer。作者提出了 Mamba-2，其速度比前代产品快 2-8 倍，同时在语言建模方面保持竞争力（[arXiv 链接](https://arxiv.org/abs/2405.21060)）。
- **研究标题中的标题党现象**：一位成员批评了诸如“Transformers are X”或“Attention is Y”等研究标题的标题党性质，并转发了 [Twitter](https://x.com/CFGeek/status/1797452053557707134) 上的讨论。讨论重点在于线性注意力（linear attention）与 Transformer 中实际使用的注意力机制之间的区别。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/calculation-math-hangover-allen-zach-galifianakis-gif-6219070">Calculation Math GIF - Calculation Math Hangover - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2405.21060">Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality</a>：虽然 Transformer 一直是深度学习在语言建模领域取得成功的主要架构，但最近的研究表明，Mamba 等状态空间模型（SSM）可以匹配或超越 Tran...</li><li><a href="https://x.com/CFGeek/status/1797452053557707134">Charles Foster (@CFGeek) 的推文</a>：研究人员一直在写这些标题为“Transformers are X”或“Attention is Y”的论文，但在内部却有微小的免责声明，说明他们*实际上*只是在讨论线性注意力，而不是...</li><li><a href="https://x.com/LoubnaBenAllal1/status/1797175938972606975">Loubna Ben Allal (@LoubnaBenAllal1) 的推文</a>：🍷 FineWeb 技术报告已发布，📚 FineWeb-Edu 也已发布，这是一个包含 1.3 trillion tokens 的数据集，其表现优于所有其他开源网络数据集，在教育基准测试（如 ...）上有显著提升。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1246187149573750895)** (250 条消息🔥🔥): 

- **AI 实验 YouTube 视频引起关注**：一位成员分享了一个名为“[Reverse Turing Test Experiment with AIs](https://youtu.be/MxTWLm9vT_o?si=NWdVITWcAauW1q8o)”的 YouTube 视频，视频中先进的 AI 试图识别它们之中的人类。另一位成员认为这个实验很酷。

- **Hugging Face 上分享剧本数据集**：一位成员整理了一个包含 [3000 部剧本的数据集](https://huggingface.co/datasets/nothingiisreal/screenplays-3k)并分享供他人使用。该数据集包括从 PDF 转换而来的 .txt 文件，并附有 AGPL-3.0 许可证链接。

- **MiniCPM 争议与下架**：讨论揭露了关于 MiniCPM-Llama3-V 模型的争议，据称该模型是 OpenBMB 的 MiniCPM 的盗版。在社区抗议和社交媒体上展示证据后，该模型已从 GitHub 和 Hugging Face 下架。

- **Perplexity AI 的 Pro Search 功能受到好评**：成员们讨论了 Perplexity AI 的 [Pro Search 功能](https://www.perplexity.ai/search/This-was-a-T4dmmjeQS5eIxV3SY80HIA)的优势，强调了其类 Agent 的行为以及在深度搜索中的实用性。不过，他们也指出 Perplexity 缺乏规范的更新日志（patch notes）。

- **Mobius 模型中独特的训练技术**：Mobius 的训练技术和数据范围因产出了具有广泛能力的模型而受到赞赏。成员们认为独特的训练方法和广泛的数据集是其性能的关键因素，使其在社区中备受关注。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.15613">Automatic Data Curation for Self-Supervised Learning: A Clustering-Based Approach</a>: 自动数据策展用于自监督学习：一种基于聚类的方法。自监督特征是现代机器学习系统的基石。它们通常在数据集上进行预训练，而这些数据集的构建和策展通常需要大量的人力...</li><li><a href="https://huggingfacefw-blogpost-fineweb-v1.static.hf.space/dist/index.html#synthetic_data">FineWeb: decanting the web for the finest text data at scale</a>: FineWeb：大规模提炼网络以获取最优质的文本数据（未找到描述）</li><li><a href="https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5">openbmb/MiniCPM-Llama3-V-2_5 · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/MxTWLm9vT_o?si=NWdVITWcAauW1q8o">Reverse Turing Test Experiment with AIs</a>: 与 AI 进行的逆向图灵测试实验。一群世界上最先进的 AI 试图找出他们中的人类。我在 Unity 中制作的实验。配音由 ElevenLabs 提供。</li><li><a href="https://x.com/tsarnick/status/1797037176414474724">Tweet from Tsarathustra (@tsarnick)</a>: Joscha Bach 表示 ChatGPT 为不太聪明的人和非常聪明的人赋予了超能力；只有我们中间那些像无聊的意见记者和 AI 批评者那样的 Prompt 补全者，才需要寻找新的...</li><li><a href="https://github.com/huggingface/trl/pull/1686">intial RPO loss by kashif · Pull Request #1686 · huggingface/trl</a>: 出自论文：https://arxiv.org/pdf/2404.19733</li><li><a href="https://discord.gg/kRbaDnHE">Join the PixArt-α Discord Server!</a>: 加入 PixArt-α Discord 服务器！在 Discord 上查看 PixArt-α 社区 - 与其他 1699 名成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://huggingface.co/datasets/nothingiisreal/screenplays-3k">nothingiisreal/screenplays-3k · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/teortaxestex/status/1797438010163867933?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: 在 Llama-3-V 窃取 @OpenBMB 模型的证据流出后，其 GitHub 和 HF 均已下线。抱歉兄弟们，我不认为这应该是你们生命中结束的一章，去探索新的...</li><li><a href="https://x.com/zhanga6/status/1797293189378068768?s=46&t=zdoDWYj2oTzRaTJHApTcOw">Tweet from Ao Zhang (@zhanga6)</a>: 听到这个消息非常难过 (https://github.com/OpenBMB/MiniCPM-V/issues/196)😰。我们调查的结论是：1. Llama3-V 在修改后可以使用 MiniCPM-Llama3-V 2.5 的代码和 config.json 运行...</li><li><a href="https://x.com/DataPlusEngine/status/1796931477738828237">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: 我实现了自己的“Prompt 注入”，现在我可以... 以不同的方式对 UNet 的特定区块进行 Prompt，并且我为不同的工作流用途实现了三种不同类型的节点...</li><li><a href="https://github.com/interstellarninja/function-calling-eval">GitHub - interstellarninja/function-calling-eval: A framework for evaluating function calls made by LLMs</a>: 一个用于评估 LLM 函数调用的框架 - interstellarninja/function-calling-eval</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>: 一个由 LLM 驱动的知识策展系统，可研究特定主题并生成带有引用的完整报告。 - stanford-oval/storm</li><li><a href="https://huggingface.co/datasets/N8Programs/CreativeHuman">N8Programs/CreativeHuman · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ShihaoZhaoZSH/LaVi-Bridge">GitHub - ShihaoZhaoZSH/LaVi-Bridge: Bridging Different Language Models and Generative Vision Models for Text-to-Image Generation</a>: 连接不同的语言模型和生成式视觉模型以进行文本到图像生成 - ShihaoZhaoZSH/LaVi-Bridge</li><li><a href="https://pastebin.com/W1NL8C0U">&lt;ScratchPad-Think-June 2024&gt; - Pastebin.com</a>: Pastebin.com 是自 2002 年以来排名第一的粘贴工具。Pastebin 是一个可以让您在线存储文本一段时间的网站。</li><li><a href="https://x.com/DataPlusEngine/status/1797004291221155894">Tweet from DataVoid e/acc (@DataPlusEngine)</a>: http://x.com/i/article/1797003254842167297</li><li><a href="https://aksh-garg.medium.com/llama-3v-building-an-open-source-gpt-4v-competitor-in-under-500-7dd8f1f6c9ee">Llama 3-V: Matching GPT4-V with a 100x smaller model and 500 dollars</a>: Llama 3-V：用缩小 100 倍的模型和 500 美元达到 GPT4-V 的水平。编辑（6月3日）—— 来自 Twitter
</li>
</ul>

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1246284370822828136)** (53 条消息🔥): 

- **建议在 Agent 中使用线程（Threads）而非 for 循环**：成员们讨论了利用 *threads* 处理顺序任务，有人建议为每个 Agent 使用并发线程，并强调了在这种设置下无需共享知识的优势。
- **Llama70b 到 Llama8b 的知识蒸馏（Knowledge distillation）难题**：一位用户询问在 A6000 GPU 上将 Llama70b 知识蒸馏到 Llama8b 的有效方法。其他人建议了诸如使用 Token 概率、最小化交叉熵（cross-entropy）增量，以及将大模型的 Logits 作为 Ground Truth 等技术。
- **微软 RL-distillation 的热度**：一位用户兴奋地分享了 RL-distillation 如何让 7B 模型超越 Llama13B 模型，并引用了 [微软论文](http://arxiv.org/pdf/2306.08543)。
- **关于图像 Token 早期融合（Early Fusion）技术的考量**：一位用户提议通过训练 VQVAE 并调整模型以处理图像 Token，利用早期融合技术对纯文本模型进行微调以接受图像。他们对该项目的可行性表示好奇，并征求他人的意见。
- **通过微调实现 Meta 的 Chameleon**：另一位用户提到他们计划通过微调而非从头训练来实现 Meta 的 *Chameleon*，旨在使模型能够接受图像 Token。他们承诺在初步测试后分享代码。
  

---


### **Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/)** (1 条消息): 

manojbh: 有 Benchmark 吗？
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1246699206354403400)** (2 条消息): 

- **Gemini 对达成一致感到好笑**：分享了 *"实际上 Gemini 同意你的看法 🤣🤣🤣"*，显示出对前一条消息的幽默或笑声回应。虽然没有捕捉到具体背景或协议，但轻松的基调显而易见。

- **MRR/NDCG 需要 Ground Truth**：有人指出 *"你需要某种（至少是弱）Ground Truth 来计算 MRR/NDCG"*。这强调了在评估指标中计算平均倒数排名（MRR）和归一化折损累计增益（NDCG）时 Ground Truth 的必要性。
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1246242995150262357)** (21 条消息🔥): 

- **Claude 会记住之前的聊天吗？**：成员们询问 **Claude** 是否跨会话保留信息，以及这是否计入总 Token 限制。他们澄清说可以重新加载之前的上下文，但机器人不会自动维护长期记忆。
- **Worldsim 探索乌俄局势**：一些用户正在 **Worldsim** 中模拟当前的 **乌俄冲突**，以测试各种升级情景和潜在结果。他们注意到 Worldsim 填充准确细节的便捷性，表明对完整的 WorldSim 兵棋推演（WarGaming）模式感兴趣。
- **CRT-terminal 前端项目曝光**：**Worldsim 控制台** 使用的前端项目被确认为 [GitHub 上的 CRT-terminal](https://github.com/essserrr/crt-terminal)。然而，由于与移动端输入的兼容性问题，计划切换到内部解决方案。
- **Worldsim 中的文本重复故障**：成员们报告了一个在 **Worldsim 控制台** 编写 Prompt 时 **文本重复** 的故障。团队目前正在努力修复此问题。
- **访问和检索聊天记录**：用户询问如何获取他们在 **Worldsim 中的聊天记录** 副本以及如何返回之前的聊天。他们获知可以使用 `!list` 和 `!load` 命令来管理聊天记录。

**提到的链接**：<a href="https://github.com/essserrr/crt-terminal">GitHub - essserrr/crt-terminal: 复古风格终端 Shell</a>：复古风格的终端 Shell。通过在 GitHub 上创建账户为 essserrr/crt-terminal 的开发做出贡献。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1246177756518023219)** (81 条消息🔥🔥): 

- **警惕 Facebook 支持诈骗**：一名温尼伯男子在拨打了一个通过在线搜索找到的虚假 Facebook 支持电话后，被骗走了数百美元。“Chatbot 应该直接断然地说‘不，这不是 Meta 的客户支持电话’。”

- **多模态 RAG 的挑战**：使用文本和图像输入构建零售店助手在无缝统一这些输入方面面临困难。“目前使用完全独立的 LLM 调用来描述图像中的物体，然后将该描述与原始文本 Prompt 拼接起来。”

- **禁止使用 @here 提及**：一名用户因无意中使用 @here 违反了 Discord 社区规范，这被视为新手错误，并促使成员建议不要进行大规模 Ping。*“通常情况下，在任何 Discord 服务器中，使用 @-mention 所有人都是非常糟糕的做法。”*

- **Hugging Face 安全事件**：由于发生安全事件，Hugging Face 建议轮换用于 HF Spaces secrets 的 Token 或密钥。有关安全措施和调查的更多详细信息，请参阅其[博客文章](https://huggingface.co/blog/space-secrets-disclosure)。

- **数字与 LLMs 的偏好**：讨论了 LLMs 在选择随机数时对某些数字（例如 7 和 42）的偏好。重点介绍了一项显示这些数字被选中频率更高的实验，详见 [Gramener 博客](https://gramener.com/llmrandom/)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://gramener.com/llmrandom/">LLM 有偏好的数字</a>：未找到描述</li><li><a href="https://huggingface.co/datasets?search=function%20calling">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1796952220619157694">Daniel Han (@danielhanchen) 的推文</a>：@Yampeleg :((( 1) Llama-2：&lt;s&gt; 之后有空格。[/INST] 之后有空格 &amp; &lt;/s&gt; 之前有空格 2) Mistral-1：&lt;s&gt; 之后有空格。末尾有 2 个空格（bug？） 3) Mistral-3：&lt;s&gt; 之后没有空格！...</li><li><a href="https://x.com/zhanga6/status/1797293200463507543">Ao Zhang (@zhanga6) 的推文</a>：为了获得定量结果，我们还在 1K 个 Bamboo 字符图像上测试了几个基于 Llama3 的 VLM，并比较了每对模型的预测精确匹配度。每两个模型之间的重叠是...</li><li><a href="https://x.com/huggingface/status/1796640955929337884?t=c5vagxKF74BbXvMTHnRzcA&s=19">Hugging Face (@huggingface) 的推文</a>：由于发生安全事件，我们强烈建议您轮换在 HF Spaces 的 secrets 中使用的任何 token 或密钥：https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets。我们已经...</li><li><a href="https://arxiv.org/abs/2405.14394">Instruction Tuning With Loss Over Instructions</a>：指令微调（Instruction tuning）在将语言模型（LM）的输出塑造为所需风格方面起着至关重要的作用。在这项工作中，我们提出了一种简单而有效的方法，即指令建模（Instruction Modelling, IM），它训练...</li><li><a href="https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt#preparing-the-dataset):">从头开始训练因果语言模型 - Hugging Face NLP 课程</a>：未找到描述</li><li><a href="https://github.com/LLM360">LLM360</a>：LLM360 有 11 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://mlops.systems/posts/2024-06-02-isafpr-prompting-baseline.html">Alex Strick van Linschoten - 使用 Instructor 为 ISAF 新闻稿进行结构化数据提取</a>：我使用 Instructor 来了解 LLM 在从 ISAF 新闻稿数据集中提取数据方面的表现。它们表现得相当不错，但并非在所有方面都如此。</li><li><a href="https://mlops.systems/posts/2024-06-03-isafpr-evaluating-baseline.html">Alex Strick van Linschoten - 评估 GPT-4-Turbo 在结构化数据提取方面的基准性能</a>：我评估了 OpenAI 的 GPT-4-Turbo 在 ISAF 新闻稿数据集上的基准性能。</li><li><a href="https://www.quora.com/What-is-the-future-of-prompt-engineering-versus-fine-tuning/answer/Tong-Hui-Kang-1">Tong Hui Kang 对“Prompt Engineering 与 Fine-tuning 的未来是什么？”的回答 - Quora</a>：未找到描述</li><li><a href="https://www.quora.com/Should-you-fine-tune-from-a-base-model-or-an-instruct-model/answer/Tong-Hui-Kang-1">Tong Hui Kang 对“你应该从基础模型还是指令模型进行微调？”的回答 - Quora</a>：未找到描述</li><li><a href="https://github.com/OpenBMB/MiniCPM-V/issues/196">项目作者团队请留意：我发现 llama3-V 项目剽窃了 MiniCPM-Llama3-V 2.5 的大量学术成果 · Issue #196 · OpenBMB/MiniCPM-V</a>：各位 MiniCPM-Llama3-V 2.5 项目的作者，几天前我发现了一个令人震惊的事实。llama3-V (https://github.com/mustafaaljadery/llama3v) 项目中存在大量工作是...</li><li><a href="https://www.latent.space/i/138050038/replacing-fine-tuning-with-continued-pre-training">微调的终结 —— 对话 Fast.ai 的 Jeremy Howard</a>：立即收听 | 关于快速学习 AI 以及 AI 如何快速学习，关于用更少资源进行更多深度学习的任务，发明 ULMFiT 以及为什么它现在是错误的，以及如何玩转 AI Discord 游戏</li><li><a href="https://www.cbc.ca/news/canada/manitoba/facebook-customer-support-scam-1.7219581">温尼伯男子在 AI 告知其虚假 Facebook 客服电话合法后陷入骗局 | CBC 新闻</a>：一名温尼伯男子表示，当他拨打他认为是 Facebook 客服热线的电话时，被骗走了数百美元，他想提醒其他人可能会发生的问题。
</li>
</ul>

### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-1](https://discord.com/channels/1238365980128706560/1239614536298795121/1246414894333689937)** (9 条消息🔥): 

- **Newsletter 摘要生成器提案**：一位成员提议使用 LLM 开发一个 **Newsletter 摘要生成器**，将多份 Newsletter 合并为一个摘要。重点在于**针对个性化进行 Fine-tuning**，以及将文本摘要转换为播客节目的潜力。

- **关于数据集创建的问题**：另一位正在进行类似 Newsletter 摘要项目的成员询问了数据集的创建过程。

- **利用 LLM 辅助技术文档**：讨论的一个用例是使用 LLM 生成**技术文档**。该想法包括详细说明函数属性、限制和示例用法，以节省理解代码库的时间。

- **辅助法律文件**：另一个提议的用例建议 LLM 可以帮助**填写表格和文件**，特别是法律文件，通过在相关文档上进行 Fine-tuning 来加速这一过程。

- **课程论坛回复生成**：另一位成员的想法是利用 LLM **在课程论坛上生成回复**。该模型将在课程材料和历史回复上进行训练，并使用 DPO 来优化回复质量。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1246179084170694696)** (19 条消息🔥): 

- **额度与工作区见解**：一位成员分享说他们的额度（Credits）已到账，并提到对不同工作区中“用户/组织二分法”有了新的理解，并将其与 **GitHub** 进行了类比。他们推测这种结构允许一个组织中存在多个人。

- **GPU 配置问题**：一位成员报告称，当将 `GPU_CONFIG` 设置为 `h100:1` 而非默认的 `h100:2` 时，出现了 Python 错误 *"AttributeError: 'NoneType' object has no attribute '_write_appdata'"*。该错误并非始终出现，且在应用日志中不可见，引发了周末的进一步调查。

- **使用 GPU 运行 Web 服务器**：关于配置 Web 服务器仅将 GPU 用于推理的咨询引出了一个涉及远程执行模式的解决方案。分享了一个使用 GPU 加速推理运行 Web 服务器的示例，该示例利用了 **Modal** 的构建块以及链接的 [Stable Diffusion 示例](https://modal.com/docs/examples/dreambooth_app)。

- **语音聊天机器人问题**：提出了 **Modal** 语音聊天机器人示例中语音转录和输出的问题。问题包括转录不准确和语音输出不完整，怀疑与延迟问题有关。

- **Modal 使用热情**：一位成员表达了对 **Modal** 的热情，表示他们正在将其用于 Kaggle 竞赛，并强调了它在工作流程中日益增长的重要性。

**提到的链接**：<a href="https://modal.com/docs/examples/dreambooth_app">使用 Hugging Face 和 Gradio 的宠物艺术 Dreambooth</a>：该示例使用 “Dreambooth” 论文中称为 Textual Inversion 的技术，在宠物图像（默认为一只名为 Qwerty 的小狗）上 Fine-tuning Stable Diffusion XL 模型。实际上，它教导...

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1246334688423841923)** (9 条消息🔥): 

- **Anthropic 发布 LLM 工具使用指南**：分享了由 **Anthropic** 提供的关于大语言模型 (LLMs) 工具使用的短课程/指南。更多详情请参阅[此处](https://x.com/alexalbert__/status/1796610971810853165)。

- **NVIDIA 提供 AI 认证**：NVIDIA 的生成式 AI 大语言模型 (LLM) 认证验证了使用 NVIDIA 解决方案构建 AI 应用的基础概念。该[认证详情](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/)包括其主题、准备材料和联系信息。

- **部署 GPT-2 垃圾邮件分类器的挑战**：一位成员分享了他们在将基于 GPT-2 的垃圾邮件类型分类器部署到生产环境时面临的挑战。他们强调了在 Lightning Studio 中将生成的 `.pth` 模型文件转换为 safety tensors 的困难。

- **探讨 LoRA 对模型公平性的影响**：两篇论文讨论了低秩自适应 (LoRA) 对微调模型公平性的影响。这些研究的见解分享在[此处](https://x.com/kenziyuliu/status/1796608738285191668)和[此处](https://x.com/nandofioretto/status/1796017804790944126)。

- **通过 YouTube 了解 CUDA/GPU**：对于那些对 CUDA/GPU 感兴趣的人，**CUDA MODE** 的 YouTube 视频提供了宝贵的资源和社区互动。视频和补充内容可以在[此处](https://www.youtube.com/@CUDAMODE/videos)访问。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/@CUDAMODE/videos">CUDA MODE</a>: 一个 CUDA 读书小组和社区 https://discord.gg/cudamode 补充内容见此处 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建。</li><li><a href="https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/">Generative AI and LLMs Certification</a>: 准备并参加考试以获得该主题的认证。</li><li><a href="https://x.com/nandofioretto/status/1796017804790944126?t=nabqdzJccC1YPNCyKZfyQA&s=19">Nando Fioretto (@nandofioretto) 的推文</a>: 🚨 新论文发布！🚨 探索低秩近似在微调大语言模型 (LLMs) 中的有效性。低秩微调对于减少计算和内存至关重要...</li><li><a href="https://x.com/alexalbert__/status/1796610971810853165">Alex Albert (@alexalbert__) 的推文</a>: 很高兴宣布我们正在启动一个 AI 教育计划，并且刚刚发布了关于工具使用的第一门课程！让我带你了解它涵盖的内容：</li><li><a href="https://x.com/kenziyuliu/status/1796608738285191668?t=F_8FbCy9cSzEs74mC5jpbw&s=19">Ken Liu (@kenziyuliu) 的推文</a>: LoRA 很棒。它速度快，而且（大部分情况下）很准确。但这种效率是免费的午餐吗？微调后的模型会出现副作用吗？我们并不完全确定，所以我们测试了 ViT/Swin/Llama/Mistral &...</li><li><a href="https://x.com/kenziyuliu/status/1796608738285191668?s=46&t=-TRJUfVdW8KeDqen1HJU1Q">Ken Liu (@kenziyuliu) 的推文</a>: LoRA 很棒。它速度快，而且（大部分情况下）很准确。但这种效率是免费的午餐吗？微调后的模型会出现副作用吗？我们并不完全确定，所以我们测试了 ViT/Swin/Llama/Mistral &...</li><li><a href="https://lightning.ai/lightning-ai/studios/code-lora-from-scratch">从零开始编写 LoRA - sebastian 的 Lightning Studio</a>: LoRA (Low-Rank Adaptation) 是一种流行的更高效微调 LLM 的技术。本 Studio 通过从零开始编写代码来解释 LoRA 的工作原理，这是一个深入了解其底层机制的极佳练习...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1246943310917009510)** (1 条消息): 

提供的消息历史仅包含一个关于积分的问题：

- **关于仪表板上积分申请的查询**：*"Hi Zach，想知道积分什么时候会发放，以及我们如何在仪表板中看到它们？"* 用户正在询问关于积分发放的时间以及在仪表板上可见性的详情。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1247161007831519232)** (3 条消息): 

- **等待 Replicate 积分确认**：成员们对未收到 Replicate 的积分表示担忧。管理员目前正在发放积分，并请成员们等待几天以确认。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/)** (1 条消息): 

computer_internet_man: 所有旧技能依然有效，hoocoodanode
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[whitaker_napkin_math](https://discord.com/channels/1238365980128706560/1242223332695478332/1246176434234458293)** (31 messages🔥): 

- **关于 sample packing 价值的讨论**：一位成员表达了对实现 sample packing 时可能出现 bug 的担忧，宁愿接受性能损失。另一位成员则对其在长序列中的价值表示怀疑，认为它在短样本场景下可能更有益。

- **评估 LLM 微调**：一位成员询问了微调模型的评估策略，包括数据集和跟踪方法。另一位成员分享了他们使用 LLM 作为裁判（LLMs as judges）进行初步评估的方法，虽然注意到其主观性，但认为其在快速诊断方面很有价值。

- **HQQ 与 Mixtral 模型的成功**：一位成员称赞了 [Mixtral-8x7B-Instruct](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ) 模型的表现，强调了其结合 4-bit 和 2-bit 量化的特性，在竞争激烈的排行榜分数中实现了质量和 VRAM 占用的良好平衡。他们还链接了 [HQQ repository](https://github.com/mobiusml/hqq) 以供进一步探索。

- **AI21 的 Jamba 模型**：一位成员分享了 [AI21's Jamba model](https://www.ai21.com/jamba) 的链接，该模型结合了 Transformer 和 SSM 层。该模型旨在融合两种架构的优势，解决传统 Transformer 的局限性。

- **赞扬与技术支持**：多位成员对近期富有信息量的会议和讨论表示感谢。此外，还有关于讲座录像访问权限的技术问题报告，这些问题已得到及时处理和修复。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.ai21.com/jamba">Introducing Jamba</a>：一个开创性的 SSM-Transformer 开源模型</li><li><a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-2bitgs8-metaoffload-HQQ · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/mobiusml/hqq">GitHub - mobiusml/hqq: Official implementation of Half-Quadratic Quantization (HQQ)</a>：Half-Quadratic Quantization (HQQ) 的官方实现 - mobiusml/hqq
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-2](https://discord.com/channels/1238365980128706560/1242223415293644930/1246330014643912825)** (2 messages): 

- **NER 任务先尝试更简单的模型**：有人建议在转向 **GPT-4** 或 **Llama 70B** 等高级模型之前，先从更基础的模型（如用于 **NER** 任务的 **Roberta**）开始，强调更简单的流程和 Prompt Engineering。
- **测试数据集故障排除**：Daniel 感谢了这些建议，并提到他将尝试 **Prompt Engineering**，因为他已经实验过 NER。他正在测试自己构建的一个数据集，以识别潜在问题和解决方案。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[workshop-3](https://discord.com/channels/1238365980128706560/1242223458184597534/)** (1 messages): 

nik_hil__: 我支持你 👀
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[abhishek_autotrain_llms](https://discord.com/channels/1238365980128706560/1242223673566433431/1246231133126004756)** (3 messages): 

- **新用户遇到微调错误**：一位刚开始在 Hugging Face 上进行微调的成员报告称，在尝试使用 GUI 时遇到错误。用户遇到了 Space 被自动暂停的问题，随后在开始训练时收到 409 错误，请求解决建议。
- **寻求从 Autotrain 到 GGUF 的快速转换**：另一位成员询问了将 Autotrain 结果转换为 GGUF 的最快方法。他们分享了一个[相关 Hugging Face Space 的链接](https://huggingface.co/spaces/ggml-org/gguf-my-repo)，但指出他们尚未成功运行。

**提及的链接**：<a href="https://huggingface.co/spaces/ggml-org/gguf-my-repo">GGUF My Repo - a Hugging Face Space by ggml-org</a>：未找到描述

### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1247165598555836416)** (3 messages): 

<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>摘要</title>
</head>
<body>
    <ul>
        <li><strong>活动改期带来乐观情绪：</strong> 一位成员对错过活动表示遗憾，但希望能看到录像。另一位成员告知他们活动已改至周四，让他们有机会参加。</li>
        <li><strong>更新活动时间：</strong> 一位成员提到他们正在努力更新所有活动时间。这表明社区日程可能会有潜在的重新组织或排期变动。</li>
    </ul>
</body>
</html>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1246181957960732773)** (1 messages): 

- **零售商店助手中多模态 RAG 的挑战**：一个项目专注于*构建零售商店助手*，能够根据文本或图像输入使用 CLIP embeddings 识别服装。目前的难点在于当*同时*使用图像和文本输入时如何进行统一，因为当前的解决方案涉及分别调用 LLM 来描述图像，并将描述与文本拼接。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1246792728638001243)** (3 messages): 

- **Jeremy 讨论 FastHTML**：一位成员期待 Jeremy 谈论 [FastHTML](https://answerdotai.github.io/fasthtml/)，这是一个用于使用纯 Python 函数编写快速且可扩展的 Starlette 驱动的 Web 应用程序的库，无需学习 Starlette 或 Javascript。他们强调了其安装和用法，并指出它创建的高性能 Web 应用可与 Instagram 等顶级 Python Web 服务器相媲美。

- **与 FastUI 的比较**：另一位成员幽默地将 FastHTML 与 [FastUI](https://github.com/pydantic/FastUI/tree/main) 进行了比较，强调 FastUI 与 Pydantic 的关系比与 FastAI 的关系更密切。对话中提到 FastUI 的目标是更快地构建更好的 UI，为 UI 开发领域做出贡献。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pydantic/FastUI/tree/main">GitHub - pydantic/FastUI: Build better UIs faster.</a>：更快地构建更好的 UI。通过在 GitHub 上创建账号为 pydantic/FastUI 的开发做出贡献。</li><li><a href="https://answerdotai.github.io/fasthtml/">fasthtml</a>：创建 HTML 应用最快的方式
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1246951731833737277)** (1 messages): 

- **Opus 为演示生成输入提示词**：一名团队成员更新了脚本，使用 Opus 为每个演示生成输入提示词（input prompts），并以实际的 `app.py` 文本作为响应。他们还在 **AutoTrain** 上进行初步测试以评估性能。
- **计划提取代码库详情并集成 Discord 问答**：后续步骤包括从代码库中提取关于类和函数的信息，并集成 Discord 问答。他们可能需要获得批准来创建一个机器人以提取 Discord 聊天数据，但在最坏的情况下，他们可以手动复制并粘贴数据。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1246177427735052359)** (34 messages🔥): 

- **在 Axolotl 中将 Preamble 替换为 System Message**：一位用户在 **Axolotl** 中设置指令风格的提示词模板时遇到困难，*"试图弄清楚如何将 preamble 替换为 system message"*。另一位用户建议在配置中使用 `default_system_message` 参数。
- **MacOS 库不兼容及 Docker 使用**：用户讨论了某些库在 Mac 上不可用，建议使用带有 `--arch linux/amd64` 的 Docker 镜像。
- **训练与资源分配问题**：一名成员在 Colab 上运行 Axolotl 时遇到非零退出状态（non-zero exit status）错误，而另一名成员则遇到了不同 GPU 卡之间分配不均的问题。详细讨论指出了一些限制，并提出了使用 FSDP 或 DeepSpeed 的潜在解决方法，但对 torch 的支持仍持怀疑态度。
- **LoRA 微调的有效性**：一位用户质疑为什么他们在 **LLaMA-3 8B** 上微调的 LoRA 模型在数学问题上的表现比基座模型更差。其他人解释说，数据集分布可能会影响性能，*"模型可能会忘记它原本能解决的问题。"*
- **自定义 Axolotl 提示词策略**：围绕自定义 **Axolotl** 配置文件进行了广泛讨论。用户寻求帮助以理解如何定义自定义提示词风格和策略，以及如何映射不同的数据集列，并参考了 [Axolotl 文档](https://openaccess-ai-collective.github.io/axolotl/docs/config.html) 获取指导。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1246333657128632384)** (23 messages🔥): 

- **Meta device 提升推理机制**：一位用户询问了 `device map = meta` 的用途，另一位成员解释说它目前支持所有的推理机制，是 Accelerate 中“大模型推理（big model inference）”的核心。他们还提到了它在量化机制中的作用，详见 [Hugging Face 博客文章](https://huggingface.co/blog/hf-bitsandbytes-integration#accelerate-is-all-you-need)。

- **讨论最佳模型分片大小**：当被问及上传到 Huggingface 的模型分片（model shards）的最佳大小时，建议将模型自动分片为约 5GB 以提高效率。一位成员分享了他们训练大规模数据集的经验，并获得了关于增加 GPU 进行训练时如何保持 batch size 的建议。

- **Batch size 与梯度累积（Gradient Accumulation）的细节**：对于大规模数据集，建议将 batch size 保持为 2 的幂以提高效率，并将梯度累积步数（gradient accumulation steps）与期望的同步等待时间相匹配。他们讨论了一种将 micro_batch_size 设置为 8 并优化梯度累积步数以提高训练稳定性的策略。

- **大 Batch size 增强训练稳定性**：一条推文链接强调，即使使用梯度累积，大 batch size 也能显著稳定训练。这简化了分布式训练，甚至可以在以太网连接上进行，暗示了未来举办 LLM 局域网派对（LAN parties）的可能性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/hf-bitsandbytes-integration#accelerate-is-all-you-need">A Gentle Introduction to 8-bit Matrix Multiplication for transformers at scale using transformers, accelerate and bitsandbytes</a>：未找到描述</li><li><a href="https://x.com/vikhyatk/status/1796437000582779286">来自 vik (@vikhyatk) 的推文</a>：大 batch size 的效果好得离谱！我已经达到了每分钟只需要同步一次梯度的程度，所以通过以太网进行分布式训练实际上是可能的？
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1246443719587008522)** (23 条消息🔥): 

- **在 Axolotl 中使用 YAML 进行二分类挑战**：一位成员在为 Axolotl 的二分类任务设置 .yaml 文件时遇到了 *"ValueError: Instruction 'train' corresponds to no data!"* 问题。他们确认数据集为 .csv 格式且适用于垃圾邮件分类器，但在 Axolotl 的正确配置上遇到了困难。

- **切换到更兼容的框架**：由于 Axolotl 在支持二分类训练方面的限制，另一位成员建议使用 Bert 处理分类任务，并提供了一个指向 [minimal-trainer-zoo](https://github.com/muellerzr/minimal-trainer-zoo/blob/main/sequence_classification.py) 仓库的资源链接，用于类似的二分类任务。

- **TRL 作为替代方案**：一位成员建议不要完全切换平台，而是在 Axolotl 无法满足需求时直接降级使用 TRL，并强调了类似的经验以及使用纯 PyTorch 或 Autotrain 完成任务的可能性。

- **Axolotl 微调中异常巨大的 Loss 值之谜**：一位成员发现，与 TRL 中的类似运行相比，在 Axolotl 中微调基础模型时出现了意想不到的高 Loss 值，特别提到了使用了输入输出模板以及包括 DeepSpeed 和非 QLoRA 设置在内的不同配置。正在探索的潜在原因包括学习率、输入输出预处理问题以及其他配置差异。

**提到链接**：<a href="https://github.com/muellerzr/minimal-trainer-zoo/blob/main/sequence_classification.py">minimal-trainer-zoo/sequence_classification.py at main · muellerzr/minimal-trainer-zoo</a>：Hugging Face Trainer 的极简示例脚本，专注于保持在 150 行以内 - muellerzr/minimal-trainer-zoo

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[freddy-gradio](https://discord.com/channels/1238365980128706560/1242564125524234361/1246439918176043049)** (2 条消息): 

- **轻松分享 Gradio 应用**：成员们讨论了使用 `share=True` 参数创建一个通往其机器的安全隧道，以便快速测试 Gradio 应用。强调了虽然这在短期内有效，但该进程需要保持运行才能持续访问。
- **探索共享托管选项**：提到了一份关于分享 Gradio 应用的各种方法的指南，提供了如 [在 HF Spaces 上托管](https://www.gradio.app/guides/sharing-your-app)、嵌入托管的 Space 等选项。该指南涵盖了身份验证、安全性和分析等细节，提供了一个全面的分享策略。
- **通过 OAuth 进行私有访问**：对于需要隐私的用户，建议集成 OAuth 以实现更安全的访问控制。这确保了应用保持私有，仅限授权用户访问。

**提到链接**：<a href="https://www.gradio.app/guides/sharing-your-app">Sharing Your App</a>：Gradio 分步教程

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[charles-modal](https://discord.com/channels/1238365980128706560/1242564177952768062/1246812563631509575)** (43 messages🔥): 

- **Mistral7B_v0.1 模型下载失败导致困扰**：有用户报告在尝试基于 [此仓库](https://github.com/modal-labs/llm-finetuning) 下载 **Mistral7B_v0.1 模型** 时遇到 `LocalEntryNotFoundError`。即使切换到 v0.3 版本，问题依然存在。

- **Hugging Face 令牌身份验证障碍**：用户讨论了脚本缺乏 Hugging Face 身份验证的问题，这源于最近的一次 [安全事件](https://huggingface.co/blog/space-secrets-disclosure) 导致需要轮换令牌/密钥。解决方案是在设置中直接将 Hugging Face 令牌设置为环境变量。

- **修复 Secret 令牌访问**：一位用户确认，强制将 Hugging Face 令牌作为 **modal.Secret** 无法直接显示其值，但提供了一个变通方案，即通过脚本打印令牌环境变量的值来进行验证。

- **单 GPU 配置导致 DeepSpeed 出现问题**：有用户报告在运行 Modal GitHub 文档中的示例时出现 `No module named 'mpi4py'` 错误。将单 GPU 切换为 `a100-80gb:2` 解决了该问题，这突显了创建通用配置设置的挑战。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modal-labs/llm-finetuning">GitHub - modal-labs/llm-finetuning: Guide for fine-tuning Llama/Mistral/CodeLlama models and more</a>：微调 Llama/Mistral/CodeLlama 等模型的指南。</li><li><a href="https://x.com/huggingface/status/1796640955929337884">Hugging Face (@huggingface) 的推文</a>：由于安全事件，我们强烈建议您轮换在 HF Spaces 的 Secret 中使用的任何令牌或密钥：https://huggingface.co/docs/hub/en/spaces-overview#managing-secrets。我们已经开始...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langchain-langsmith](https://discord.com/channels/1238365980128706560/1242564256914870384/1247089373674209332)** (1 messages): 

- **第二个视频录制加载问题**：一位成员提到在加载第二个视频录制时遇到困难。他们只能访问转录文本，而无法观看视频本身。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[allaire_inspect_ai](https://discord.com/channels/1238365980128706560/1242943547699888229/1246206555649736805)** (3 messages): 

- **模型分级评分器上线**：一位成员强调了 **inspect_ai** 项目中存在 [模型分级评分器 (model graded scorer)](https://github.com/UKGovernmentBEIS/inspect_ai/blob/main/src/inspect_ai/scorer/_model.py)。他们询问是否可以使用成对排序（pairwise ranking）对两个不同 LLM 的输出进行侧重（side-by-side）评估，但未得到直接回答。
  
- **建议为成对排序使用可组合工作流**：有人建议可以通过组合工作流来评估成对排序，方法是使用目标数据集，其中输入是两个不同的输出。然后评分器可以通过在相同输入上生成两个 LLM 的输出，并使用另一个模型来评估排名/偏好，从而评估“成功率”，同时考虑到使用同一模型进行评分可能存在的偏见。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1246182057500213340)** (42 messages🔥): 

- **错过 Hugging Face 截止日期影响额度**：由于 Hugging Face 表单过期引发了问题，导致试图赶进度的用户感到焦虑。Dan 尝试通过与 Hugging Face 沟通来解决此问题。
  
- **管理员分享平台额度发放流程**：Dan 和 Hamel 向用户保证，额度信息已提交给所有平台，但实际的额度发放将取决于供应商的具体流程。敦促拥有待处理额度的用户耐心等待，因为平台正在处理中。

- **表单提交不一致引发担忧**：一些用户发现表单提交存在差异，担心尽管按时注册仍会错过额度。管理员澄清说，提交不同的电子邮件可能会导致混乱，但对流程的完整性表示放心。

- **重复的额度截止日期警告**：许多用户询问因表单提交晚或出差而错过额度截止日期的问题。管理员确认，尽管用户有理由，但在截止日期之后不会再发放额外的额度。
  
- **关于特定供应商的澄清**：Hamel 澄清说，从未承诺过 RunPod 额度，而 OpenAI 额度需要提供 Org ID，并强调了准确遵守表单要求对于获取额度的重要性。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1246514501235245189)** (6 messages): 

- **Fireworks 项目表单提交延迟**：多位成员对 Fireworks 项目表单提交逾期表示担忧，并询问是否仍能获得额度（credits）。一位成员提到，他们是基于良好的口碑决定尝试该项目的。
- **等待额度更新**：成员们询问额度何时可用。一位成员为分配额度的延迟表示歉意。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/1247236531111067681)** (3 messages): 

- **Emmanuel 演讲引发热烈反响**：成员们对 Emmanuel 表示兴奋，其中一人说道：*"Emmanuel 太棒了，我对这次演讲感到非常兴奋 😄。"* 
- **对课程时长过短表示不满**：在获得批准后，另一位成员分享道，尽管很开心，但遗憾的是课程 *"不幸地"* 太短了，并补充说 *"所有课程都太短了 🤪。"*

这些消息中未讨论任何链接或博客文章。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[braintrust](https://discord.com/channels/1238365980128706560/1245407617031999581/1247079665315745802)** (5 messages): 

- **Python SDK 因极简依赖受到称赞**：一位成员对 Python SDK 保持的**精简依赖（lean dependencies）**表示赞赏，描述为：*"我非常喜欢你们在 Python SDK 中保持如此精简/极小的依赖。"* 开发者积极回应道：*"我们一直在努力！感谢反馈。"*
- **额度分配说明**：一位用户询问在 Braintrust 创建用户账户后如何获取额度，此前他曾误加入了一个不同的招聘网站网络。开发团队索要了该用户的邮箱以确保额度正常处理，并承诺会确保 *"将你计算在内"*。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[west-coast-usa](https://discord.com/channels/1238365980128706560/1245410680065097738/1246201618970579056)** (1 messages): 

- **对未来线下聚会感兴趣**：一位成员表示如果有更早的通知，他们有兴趣参加未来的活动。他们感谢了主持人，并提到目前的活动“听起来很有趣”。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[east-coast-usa](https://discord.com/channels/1238365980128706560/1245411101718610001/1246313214199730187)** (2 messages): 

- **马里兰州的邻居**：一位成员提到他们在马里兰州的 Germantown。另一位成员回应称，他们当天刚好去了 Germantown。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1246242100085784576)** (6 messages): 

- **来自罗马尼亚的问候**：一位成员以热情的 **"👋🏻 来自罗马尼亚 🇷🇴"** 开始了自我介绍。
- **在阿姆斯特丹的德国人**：另一位成员介绍自己是**住在阿姆斯特丹的德国人 🇳🇱**。
- **慕尼黑的幽默**：一位来自**慕尼黑**的成员在问候中调皮地加了一个 **"😂"**。
- **伦敦聚会热度**：一位伦敦成员对潜在的聚会表示热忱，说道 **"伦敦聚会听起来不错！"**。
- **德国城市集结**：来自**德国**的成员纷纷加入，代表**奥尔登堡（Oldenburg）**和**汉堡（Hamburg）**发出问候。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[announcements](https://discord.com/channels/1238365980128706560/1245460787196068030/1247205930710335540)** (1 messages): 

- **有时效性的额度分配**：如果你在太平洋时间 5 月 29 日晚上 11:59 之前注册但未填写表单，你需要为 Replicate, Predibase, Fireworks, Braintrust, Langsmith, OpenPipe 和 Jarvis 等多个平台创建账户以接收额度。**“许多平台将在今天分配额度。请尽快创建账户，以便他们发放额度。”**
- **Modal 和 OpenAI 的待办事项**：Modal 表单仍然有效，尚未填写的用户应尽快填写。由于缺少组织 ID（organization IDs），OpenAI 和 Langsmith 的额度目前处于悬而未决的状态，目前无法采取进一步行动。
- **HuggingFace 表单已关闭**：HuggingFace 表单已关闭，因此无法再在该平台上采取任何获取额度的行动。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1246437460867547148)** (7 条消息): 

- **用户对 Predibase 过多的营销邮件感到沮丧**：一位成员表达了在使用工作邮箱注册后收到大量营销邮件的挫败感。另一位成员澄清说，他们上周只为即将举行的研讨会发送了一封营销邮件，并承诺会认真对待这一反馈。
  
- **不支持使用外部 Checkpoint 进行推理**：Predibase 目前不支持使用其自身以外的 Checkpoint 运行 Inference。一位成员对想要使用其他 Checkpoint 的动机感到好奇，并表示愿意将相关输入分享给产品和工程团队。

- **将分享教程录像**：成员们期待 6/7 的 Predibase 教程。Predibase 确认他们将在之后分享现场教程的录像。

- **评估与训练损失讨论**：讨论了当评估损失（Evaluation Loss）略高但训练损失（Training Loss）显著降低时，尝试不同 Checkpoint 的重要性。这被提及作为获得更好结果的策略的一部分。

- **用于微调的课程点数**：一位成员询问其账户中课程点数的激活情况，以便对更大的模型（L3 70B）进行 Fine-tuning，这对于他们有限的训练数据特别有价值。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[career-questions-and-stories](https://discord.com/channels/1238365980128706560/1245816565073576037/1246424846603386920)** (17 条消息🔥): 

<ul>
    <li><strong>历史学者转型技术领域</strong>：一位用户分享了他们从历史专业背景转向以 ML 为核心角色的历程，克服了早期在坎大哈电力有限和数学基础薄弱等挑战。他们最终在 ZenML 获得了一份工作，并强调了 fastai 和 Launch School 等课程在转型中的重要性。</li>
    <li><strong>图技术爱好者利用裁员契机转型</strong>：Dan 讲述了受《巴拿马文件》使用 Neo4j 的启发，从法务会计转向 Data Science 的经历，尽管最初缺乏编程技能。他的职业转型得益于研究生项目和自主学习，最终在科技巨头以及他目前工作的 Neo4j 担任职位，负责图数据科学算法。</li>
    <li><strong>对数学编程的痴迷</strong>：Laith 详细介绍了从大学数学编程课程到咨询工作再到 Deep Learning 的进阶过程，将正式学习与自我教育相结合。他推荐 Radek Osmulski 的博客作为学习 ML 的资源，并讨论了如何平衡职业与个人生活。</li>
    <li><strong>Reddit 工程师寻求 ML 转型</strong>：Reddit 的一名 Backend Engineer 表达了从构建 ML Inference 栈转向创建 ML 产品的愿望。该用户就如何应对海量的 ML 学习资源以及 Generative AI 的变革性影响寻求建议。</li>
    <li><strong>ML 与底层工程咨询公司的构想</strong>：分享了一条推文，建议建立一种结合设计师、ML 工程师和底层工程师的利基咨询模式，旨在为客户优化并使用 C++ 或 Rust 重写 ML 模型 Inference。该服务针对需要高性能、CPU 优化的模型 Inference 的客户。</li>
</ul>
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/andrew_n_carr/status/1796919853766549574">Andrew Carr (e/🤸) (@andrew_n_carr) 的推文</a>：我认为市场上存在一个超利基的 3 人咨询公司的空间。1. 设计师 2. ML 工程师 3. 底层工程师。客户提供一个开源或内部的代码库，然后你……</li><li><a href="https://python.useinstructor.com/concepts/patching/?h=md+json#json-mode">Patching - Instructor</a>：无描述</li><li><a href="https://mlops.systems">Alex Strick van Linschoten - MLOps.systems</a>：无描述</li><li><a href="https://radekosmulski.com/the-post-tech-job-manifesto/">Meta Learning: 补遗或修订后的生活秘诀</a>：2021 年我出版了《Meta Learning: How To Learn Deep Learning And Thrive In The Digital World》。这本书基于我 8 年的生活经历，期间几乎每天我都在思考如何学习 Machine Learning……
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/)** (1 条消息): 

peterg0093: 问题是，在我们拿到 OpenAI 额度之前，能看到 GPT-5 吗？
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1246177602461241414)** (315 条消息🔥🔥): 

- **关于在优化器（Optimizers）中使用零交叉（Zero Crossings）的辩论**：成员们讨论了在梯度中追踪零交叉（反转）以提高优化器性能的潜在用途和局限性。值得注意的是，基于零交叉的梯度裁剪（gradient clamping）实验结果褒贬不一，部分实验显示收敛速度变慢或没有显著改善。

- **对 SGD 优化器的批判与辩护**：关于将 SGD 优化器作为开发新优化器时的基准（baseline）的优缺点进行了反复讨论。一位用户提到，“SGD 是一个基准，任何比它更好的方案都是靠学习率（learning rates）起作用的”，这表明精细化调整对于超越简单的 SGD 至关重要。

- **使用 FlashAvatar 制作逼真的数字人**：讨论重点介绍了一种利用多角度录制创建高保真数字人的新方法，该方法可能能够在 Nvidia RTX 3090 上以 300FPS 的速度渲染和驱动虚拟头像。[FlashAvatar](https://simongiebenhain.github.io/NPGA/) 项目是关注的焦点。

- **关于 AI 处理上下文和创造力的辩论**：一位用户表达了对 GPT-4 重复相同信息且无法在提供的上下文中提供创造性解决方案的困扰。这引发了关于改进 Prompt 的建议，以及对 LLM 在处理长对话链和创造性 Prompt 方面局限性的认识。

- **免费访问自定义 GPTs**：简要提到免费层级用户现在可以在 OpenAI 平台上访问自定义 GPTs。这一更新促使一些成员考虑迁移他们的 GPT 模型以获得更广泛的可访问性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://simongiebenhain.github.io/NPGA/">NPGA: Neural Parametric Gaussian Avatars</a>：未找到描述</li><li><a href="https://github.com/rafaela00castro/pytorch-hands-on/blob/master/mnist_cnn.ipynb">pytorch-hands-on/mnist_cnn.ipynb at master · rafaela00castro/pytorch-hands-on</a>：我在 CEFET/RJ 关于卷积神经网络和 PyTorch 的讲座 - rafaela00castro/pytorch-hands-on</li><li><a href="https://www.youtube.com/watch?v=SkTt9k4Y-a8">LMFAO - Sorry For Party Rocking</a>：Sorry For Party Rocking - 立即购买专辑！http://smarturl.it/LMFAODeluxe</li><li><a href="https://www.colanguage.com/slovak-verbs">Slovak verbs | coLanguage</a>：斯洛伐克语动词表达人、动物、事物等的过程、动作或状态。让我们来看看本课的概览！
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1246192026635210914)** (55 条消息🔥🔥): 

- **GPT 的内存泄漏问题再次出现**：用户讨论了登录时遇到“白屏”的情况，并推测可能是内存泄漏的修复。有人提到注意到“胡言乱语”和重复现象，并将其归因于机器人的温度（temperature）设置。

- **探索自定义 GPT 的创新应用**：成员们交流了自定义 GPTs 的独特用途，例如调查意外的 AWS 账单，并讨论了 GPT 功能的潜在改进，如在没有用户定义术语的情况下集成“短期记忆”。

- **Playground 和 API 文件限制**：澄清了通过 [OpenAI 官方帮助文章](https://help.openai.com/en/articles/8843948-knowledge-in-gpts) 向 GPT 知识库上传文件的限制。限制包括“每个文件最大 512 MB”和“每个文件最多 500 万个 tokens”。

- **关于上下文窗口（Context Windows）与嵌入（Embeddings）的辩论**：用户辩论了嵌入与更长上下文窗口的有效性，并对传闻中为提升性能而集成 Gemini 的更新表现出浓厚兴趣。一些人更倾向于更智能、更短的上下文，而不是仅仅扩大上下文容量。

- **排查 GPT 编辑和 Actions 的故障**：GPT 编辑的问题被归因于订阅问题，而其他用户则在排查失效的 GPT actions，最终通过回退到旧版本解决了问题。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1246462837547601992)** (7 条消息): 

- **默认系统消息胜出**：针对复杂 Prompt 是受益于单个还是多个系统消息的问题，一位成员主张使用 *"1 条系统消息 (system message)"*。
- **ChatGPT 遵循指南的问题**：一位成员对 ChatGPT 无法遵循指南表示沮丧，并寻求改进其表现的技术。他们请求针对其特定用例提供协助。
  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1246462837547601992)** (7 条消息): 

- **讨论参数**：一位用户询问了另一位用户正在使用的 **temperature 和 top-p 设置**，并询问他们的主题是否比较 **finnicky**（难以调优）。
- **System Messages 的偏好**：一位用户向群组询问了在 GPT-4 和 4o 中构建 **复杂 prompts** 的偏好——是使用 **单个 system message 还是多个**。一位用户回复表示更倾向于使用 **单个 system message**。
- **寻求 ChatGPT Guidelines 方面的帮助**：一位用户表示在 ChatGPT 不遵循其 guidelines 方面遇到了困难，并寻求解决此问题的技术或帮助。公开讨论中没有提供后续的解决方案。

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1246382411172937761)** (198 条消息🔥🔥): 

- **Mojo Language Server 崩溃**：成员们报告了 **Mojo language server** 在 MacBook M2 上的 VS Code 衍生版本（如 Cursor）中频繁崩溃的问题。GitHub [issue #2446](https://github.com/modularml/mojo/issues/2446) 概述了该问题，并提到修复补丁仅在 nightly 版本中可用。
  
- **Mojo 语言成熟度与社区路线图**：讨论了 **Mojo** 何时达到成熟和稳定状态，并对正在进行的开发和开源社区贡献进行了深入探讨。查看 [Mojo roadmap](https://docs.modular.com/mojo/roadmap#mojo-sdk-known-issues) 和博客 [announcement](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source) 了解更多详情。

- **Mojo 在网络和数据处理方面的潜力**：爱好者们讨论了一些雄心勃勃的项目，例如使用 Mojo 实现 **DPDK** (Data Plane Development Kit) 以及集成 **liburing** 以优化网络性能。呼吁 **Modular** 使用 Mojo C interop 测试 DPDK 的头文件被强调为未来开发的关键步骤。

- **Python 优化技术**：用户正在寻求优化缓慢的 Python 代码循环的方法，例如使用 `yield`、用 dict 代替 tuple，以及探索使用 Numba 进行 JIT 编译，正如这个 [YouTube tutorial](https://youtu.be/OiMZtjSZVOw?si=JrgOG_UL662xZ48W) 中所建议的那样。

- **从 Windows 迁移以使用 Mojo**：对于在 **Windows 上安装 Mojo** 遇到问题的用户，建议的解决方法是使用 **WSL 搭配 Ubuntu 22.04**。大家认可 Modular 优先在 Linux 上完善 CUDA 支持再转向其他平台的策略，并希望在夏末或秋季能获得更广泛的支持。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular 文档</a>: 未找到描述</li><li><a href="https://get.modular.com"">无标题</a>: 未找到描述</li><li><a href="https://docs.modular.com/mojo/roadmap#mojo-sdk-known-issues">Mojo🔥 路线图与注意事项 | Modular 文档</a>: 我们的 Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://tenor.com/view/hello-wave-cute-anime-cartoon-gif-13975234520976942340">Hello Wave GIF - Hello Wave Cute - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.modular.com/">Modular: 加速 AI 步伐</a>: Modular Accelerated Xecution (MAX) 平台是全球唯一能为您的 AI 工作负载释放性能、可编程性和可移植性的平台。</li><li><a href="https://youtu.be/OiMZtjSZVOw?si=JrgO">一行代码让 Python 快 1000 倍 🐍 ⏩ (Numba 教程)</a>: Numba 可以通过 JIT 编译器仅用一行代码就让你的 Python 代码提速 1000 倍，该编译器用于通过编译函数来优化 Python 中的简单函数...</li><li><a href="https://github.com/modularml/mojo/issues/2446">[BUG] Mojo Language Server 在 Visual Studio Code 中崩溃 · Issue #2446 · modularml/mojo</a>: Bug 描述：在 Visual Studio Code 中编辑 Mojo 源代码会导致 Mojo Language Server 崩溃。复现步骤：尝试在 VSCode 中编辑以下代码 fn exec_rt_closure(x: Int, bin_o...</li><li><a href="https://youtu.be/OiMZtjSZVOw?si=JrgOG_UL662xZ48W">一行代码让 Python 快 1000 倍 🐍 ⏩ (Numba 教程)</a>: Numba 可以通过 JIT 编译器仅用一行代码就让你的 Python 代码提速 1000 倍，该编译器用于通过编译函数来优化 Python 中的简单函数...</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: Mojo🔥 开源迈出的下一大步</a>: 我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源迈出的下一大步</li><li><a href="https://realpython.com/introduction-to-python-generators/#creating-data-pipelines-with-generators">如何在 Python 中使用生成器和 yield – Real Python</a>: 在这个分步教程中，你将学习 Python 中的生成器和 yield。你将使用多个 Python yield 语句创建生成器函数和生成器表达式。你... 
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 条消息): 

ModularBot: 来自 *Modular*:
<https://twitter.com/Modular/status/1797699002353488183>
  

---

### **Modular (Mojo 🔥) ▷ #[📺︱youtube](https://discord.com/channels/1087530497313357884/1098713700719919234/1247235918935621653)** (1 messages): 

- **Modular 分享新的 MAX release 视频**：**Modular** 刚刚发布了一个名为 *"Getting started with MAX release and nightly builds"* 的 [YouTube 视频](https://www.youtube.com/watch?v=raXpYwgileU)。该视频指导用户如何在系统中安装和配置 **MAX release** 以及 **nightly builds**。

**提到的链接**：<a href="https://www.youtube.com/watch?v=raXpYwgileU">Getting started with MAX release and nightly builds</a>：在这个视频中，我们将引导你完成在系统中安装和配置 MAX release 和 nightly builds 的全过程。你将学习到...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1246319039593320508)** (79 messages🔥🔥): 

- **发现调整大小逻辑缺陷**：一位用户指出，Mojo 中的调整大小逻辑没有处理待添加字符串大于增加后容量的情况。另一位用户承认了这一疏忽，表示最初并未考虑到这一点。

- **Nightly 版本中的函数重命名**：一位用户询问 `rotate_bits_left` 函数的去向。根据 [changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog-released.md)，在 nightly build 中，它已被重命名为针对 `SIMD` 类型的 `SIMD.rotate_left` 和针对 `Int` 的 `bit.rotate_bits_left`。

- **在 Windows 上安装 Mojo**：一位用户在 Windows 上使用 WSL 安装 Mojo 时遇到困难，发现路径问题源于 Windows 使用反斜杠 (`\`) 而 Linux 使用正斜杠 (`/`)。另一位用户建议使用 `/usr/bin/modular` 作为路径来解决该问题。

- **别名与 SIMD 长度**：讨论了存储与类相关的别名的最佳方式，以及处理 `SIMD` 需要 2 的幂长度的问题。共识是使用类属性并使用 `Self.nelts` 引用它们。

- **自定义 HTTP 库**：一位用户询问是否有类似于 Python `requests` 的原生 Mojo HTTP 库。建议使用 `lightbug_http`，这是一个在 [GitHub](https://github.com/saviorand/lightbug_http) 上积极维护的第三方库。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/faq#what-operating-systems-are-supported">Mojo🔥 FAQ | Modular Docs</a>：关于 Mojo 预期问题的解答。</li><li><a href="https://github.com/saviorand/lightbug_http">GitHub - saviorand/lightbug_http: Simple and fast HTTP framework for Mojo! 🔥</a>：简单且快速的 Mojo HTTP 框架！🔥。通过在 GitHub 上创建账户为 saviorand/lightbug_http 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/pull/2703">[mojo-stdlib] Add variadic initialiser, __iter__ and __contains__ to InlineList by ChristopherLR · Pull Request #2703 · modularml/mojo</a>：此 PR 为 InlineList 添加了一些功能（相关 issue #2658），包括变长参数初始化程序 `var x = InlineList[Int](1,2,3)`，迭代器 `for i in x: print(i)` 以及 `contains` 等。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/memory/unsafe.mojo#L310C11-L310C29">mojo/stdlib/src/memory/unsafe.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1246812806007885984)** (4 messages): 

- **初始数据处理基准测试 PR**：分享了一个数据处理基准测试 [PR](https://github.com/jinyus/related_post_gen/pull/514) 草案，指出目前的性能比 Python 快，但比编译型语言慢。请求对代码和 Docker 安装脚本的改进建议。
- **需要自定义 JSON 解析器**：为了优化基准测试，提议在 Mojo 中实现一个自定义 JSON 解析器，参考了之前在 [C#](https://github.com/mzaks/FlexBuffers-CSharp/blob/master/FlexBuffers/JsonToFlexBufferConverter.cs) 和 [Swift](https://github.com/mzaks/FlexBuffersSwift/blob/master/FlexBuffers/FlexBuffers.swift#L2127) 中的工作。计划下周开始处理这项贡献。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/jinyus/related_post_gen/pull/514">[Mojo] Initial version by cyrusmsk · Pull Request #514 · jinyus/related_post_gen</a>：已测试：在 5k 帖子文件上验证了代码的本地运行；未测试：Docker 安装；问题：在本地构建和运行 mojo 可执行文件会导致段错误（seg fault），但 'mojo rel.mojo' 可以正常工作...</li><li><a href="https://github.com/mzaks/FlexBuffersSwift/blob/master/FlexBuffers/FlexBuffers.swift#L2127),">FlexBuffersSwift/FlexBuffers/FlexBuffers.swift at master · mzaks/FlexBuffersSwift</a>：FlexBuffers 的 Swift 实现 - FlatBuffers 的子项目 - mzaks/FlexBuffersSwift
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1246763018893922374)** (2 messages): 

- **前向传播实现中对反向传播的担忧**：一位成员强调，前向传播需要存储每一层的输出，以便反向传播（backward pass）正常工作，并表示不确定 **Max** 是否已经支持这一点。
- **缺少反向传播文档和自定义优化器**：另一位成员对之前的信息表示感谢，并提到虽然前向传播所需的函数似乎已经具备，但找不到关于反向计算的文档，并指出可能需要自定义优化器。
  

---


### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1246186995248533595)** (30 messages🔥): 

- **发布新的 Nightly Mojo 编译器**：新的 Nightly Mojo 编译器已更新至 `2024.6.305`，现在可以通过 `modular update nightly/mojo` 获取。[Changelog](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md) 包括将全局 `UnsafePointer` 函数移动为方法，并添加了临时目录函数。
- **澄清 C 语言 `char` 符号混淆**：成员们讨论了 C 语言中的 `char` 是有符号还是无符号的，指出其符号是由实现定义的，可以在 GCC 中使用 `-funsigned-char` 进行修改，尽管这会破坏标准合规性。
- **Tensor 移出标准库**：一位用户询问关于 Tensor 被移出标准库的问题，得到的回复是这在 YouTube 上的社区会议中提到过。
- **变更日志一致性提案**：分享了关于保持变更日志条目一致性的建议，旨在改进文档格式和[风格](https://github.com/modularml/mojo/issues/2923)。
- **对条件一致性（Conditional Conformance）的期待**：人们对 Mojo 中新的条件一致性功能充满热情，预计这将大幅提升标准库的灵活性和功能。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/commit/c9ea648612d8b98127955182962cd09a5f4143bc">[mojo-stdlib] Move global UnsafePointer functions to be methods. · modularml/mojo@c9ea648</a>：这扩展了编译器，允许 `self` 的类型为带有任何参数的 `Self`，从而实现更灵活的条件一致性。这允许我们将 UnsafePointer 中的全局函数移动为方法...</li><li><a href="https://github.com/modularml/mojo/issues/2923">[Docs] Style guide entry for consistent changelog entry phrasing · Issue #2923 · modularml/mojo</a>：审查 Mojo 的优先级。我已阅读路线图和优先级，并认为此请求符合优先级。你的请求是什么？我建议在风格指南中增加一个条目，规定如何编写一致的变更日志...
</li>
</ul>

</div>

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1246340082361700352)** (28 条消息🔥): 

- **修复拼写错误！**：此前对 "stronging" 一词存在短暂误解，现已澄清其意为 *"storing"*（存储），与 embedding 存储有关（而非 *stronging embeds*）。
- **Elon vs Yann Diss Track**：一个幽默的帖子链接到了 [Yann LeCun 的 Twitter](https://x.com/ylecun/status/1796793673151127972)，LeCun 对一段涉及 Elon Musk 的 diss track 评论道 *"Hilarious 😂"*（太搞笑了）。
- **Liouville's Theorem 讨论**：社区深入探讨了 Liouville's theorem 及其对初等和非初等原函数的影响，并推测了其与 neural networks 的潜在联系。[Wikipedia 链接参考](https://en.m.wikipedia.org/wiki/Liouville%27s_theorem_(differential_algebra))。
- **Telegram 文件存储黑科技**：一位成员分享了一个将 Telegram 用于 *"无限免费文件存储"* 的工具，采用 AES-256-CTR 加密保护，可在[此处](https://tglfs.vercel.app/)获取，源代码托管在 [GitHub](https://github.com/hinsley/tglfs)。
- **关于 AI 阅读清单的辩论**：成员们讨论了 Ilya Sutskever 推荐的 AI 阅读清单是否过时，并对其历史价值与现状适用性持不同看法。一些人仍认为它 *"对于理解一个想法在当下如何运作非常有启发性"*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.m.wikipedia.org/wiki/Liouville%27s_theorem_(differential_algebra)">Liouville&#039;s theorem (differential algebra) - Wikipedia</a>：未找到描述</li><li><a href="https://x.com/ylecun/status/1796793673151127972">Yann LeCun (@ylecun) 的推文</a>：Hilarious 😂 Quoting peterxing.eth🧢🦾 — d/acc (@peterxing) Elon v Yann diss track part 3</li><li><a href="https://www.reddit.com/r/ArtificialInteligence/comments/1cpbh1s/ilya_sutskever_if_you_really_learn_a">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.reddit.com/r/ArtificialInteligence/comments/1cpbh1s/ilya_sutskever_if_you_really_learn_all_of_these/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1246183238519619594)** (125 条消息🔥🔥): 

- **Manifold Research 寻求合作者**：来自 Manifold Research 的 Sidh 宣布了在用于 multimodality（多模态）和控制任务的 transformers 方面的合作机会。他们的目标是构建第一个大规模开源 Generalist Model，并[欢迎通过多种途径贡献](https://www.manifoldrg.com/opportunities/)。
- **关于 RLHF 偏差的新见解**：一位成员分享的一篇论文指出，标准的 RLHF 本质上存在偏差，并建议通过增加熵项来缓解这一问题。*"为了减轻这种算法偏差，在 RLHF 的奖励最大化（reward maximization）中添加熵项既是必要的也是充分的。"* [查看 PDF](https://arxiv.org/abs/2405.16455)
- **探索 Transformer 的局限性**：讨论中提到了一篇论文，该论文利用通信复杂度（communication complexity）证明了 Transformer 层在足够大的定义域上难以进行函数复合。这一实证结果突显了 Transformer 架构固有的局限性。[查看 PDF](https://arxiv.org/abs/2402.08164)
- **关于 positional embeddings 的辩论**：成员们讨论了 transformers 中数据依赖型 positional embeddings 的挑战和潜在解决方案。对话强调了低维可学习位置向量可能面临的困难。[查看 PDF](https://arxiv.org/abs/2405.18719)
- **Mamba-2 与 SSMs 创新**：Albert Gu 的团队发布了 Mamba-2，引入了一个通过状态空间对偶性（SSD）将 SSMs 与 attention 联系起来的框架，有望提升性能和速度。*"Mamba-2 旨在推进序列模型的理论，开发一个连接 SSMs 与（线性）attention 的框架，我们称之为状态空间对偶性（SSD）。"* [查看 PDF](https://arxiv.org/abs/2405.21060)


<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://arxiv.org/abs/2405.21060">Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality</a>：虽然 Transformers 一直是深度学习在语言建模领域取得成功的主要架构，但最近的研究表明，像 Mamba 这样的状态空间模型 (SSMs) 能够匹配或超越 Tran...</li><li><a href="https://arxiv.org/abs/2405.16674">Limits of Deep Learning: Sequence Modeling through the Lens of Complexity Theory</a>：深度学习模型在各种应用中取得了显著成功，但在处理需要对序列进行复杂推理的任务（如函数组合和计算...）时仍然面临困难。</li><li><a href="https://arxiv.org/abs/2405.20233">Grokfast: Accelerated Grokking by Amplifying Slow Gradients</a>：机器学习中一种被称为“顿悟 (grokking)”的令人困惑的现象是指，在对训练数据几乎完美过拟合后的数万次迭代中，才实现了延迟的泛化。专注于长期的...</li><li><a href="https://arxiv.org/abs/2402.08164">On Limitations of the Transformer Architecture</a>：大语言模型 (LLMs) 中幻觉的根本原因是什么？我们利用通信复杂度证明了 Transformer 层无法进行函数组合（例如，识别一个图...）。</li><li><a href="https://arxiv.org/abs/2405.18719">Contextual Position Encoding: Learning to Count What&#39;s Important</a>：注意力机制是大语言模型 (LLMs) 的关键组件，它允许序列中的 token 相互交互，但它是阶数无关的。引入位置编码 (PE)...</li><li><a href="https://arxiv.org/abs/2405.17969">Knowledge Circuits in Pretrained Transformers</a>：现代大语言模型的卓越能力源于其参数中编码的海量知识库，使它们能够感知世界并进行推理...</li><li><a href="https://www.manifoldrg.com/opportunities/">Opportunities</a>：参与我们工作的几种方式：1. 加入我们的 Discord 并参与活动和讨论（无论是否与项目相关）。2. 异步贡献 GitHub 上的 issue。...</li><li><a href="https://x.com/_albertgu/status/1797651223035904355?s=19">Tweet from Albert Gu (@_albertgu)</a>：很高兴终于发布了 Mamba-2！！状态扩大了 8 倍，训练速度提高了 50%，还有更多的 S 🐍🐍 Mamba-2 旨在推进序列模型的理论，开发一个连接...的框架。</li><li><a href="https://api.wandb.ai/links/saesara/w8cny2aj">nanoRWKV depth 12</a>：代码/超参数取自基于 OpenWebText2 训练的 nanoRWKV，序列长度 768，迭代次数 60k。xat_time -> 将 time mixing 中的 gelu 门控替换为扩展的 arctan。1xatglu_channel -> 第一个...</li><li><a href="https://api.wandb.ai/links/saesara/f7s881y2">nanoRWKV depth 24</a>：代码/超参数取自基于 OpenWebText2 训练的 nanoRWKV，序列长度 768，迭代次数 60k。xat_time -> 将 time mixing 中的 gelu 门控替换为扩展的 arctan。1xatglu_channel -> 第一个...</li><li><a href="https://api.wandb.ai/links/saesara/g6xi3m0n">24 layer</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2405.16455">On the Algorithmic Bias of Aligning Large Language Models with RLHF: Preference Collapse and Matching Regularization</a>：将大语言模型 (LLMs) 与人类偏好准确对齐，对于实现公平、经济合理且统计高效的决策过程至关重要。然而，我们认为...</li><li><a href="https://arxiv.org/abs/2405.20768">Expanded Gating Ranges Improve Activation Functions</a>：激活函数是所有深度学习架构的核心组件。目前，最流行的激活函数是平滑的 ReLU 变体，如 GELU 和 SiLU。这些是自门控激活...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1246210933915648232)** (2 messages): 

- **机械分析论文摘要 (Mechanistic Analysis Paper Summary)**：一位成员分享了他们[论文的摘要](https://www.lesswrong.com/posts/EBbcuSuNafkYpsgTW/finding-backward-chaining-circuits-in-transformers-trained-1)，标题为《对在符号多步推理任务上训练的 Transformer 的机械分析》(*A Mechanistic Analysis of a Transformer Trained on a Symbolic Multi-Step Reasoning Task*)。他们旨在通过讨论模型如何利用后向链接电路 (backward chaining circuits) 完成任务，来重新激发研究兴趣并寻找合作者。
  
- **即将举行的机械可解释性黑客松 (Mechanistic Interpretability Hackathon)**：一场机械可解释性黑客松定于 7 月举行，邀请参与者在一个周末的时间里对神经网络进行逆向工程。详细信息和注册可在 [itch.io 活动页面](https://itch.io/jam/mechanistic-interpretability-hackathon) 查看，更多信息可通过其专门的 [Discord 可解释性服务器](https://discord.gg/Gv9r4b88hZ) 获取。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://itch.io/jam/mechanistic-interpretability-hackathon">Mechanistic interpretability Hackathon</a>：由 victorlf4 主办的 Game Jam，时间为 2024-07-05 至 2024-07-07。机械可解释性黑客松，灵感来自 Ap... 举办的类似黑客松。</li><li><a href="https://www.lesswrong.com/posts/EBbcuSuNafkYpsgTW/finding-backward-chaining-circuits-in-transformers-trained-1">Finding Backward Chaining Circuits in Transformers Trained on Tree Search — LessWrong</a>：这篇文章是我们论文《对在符号多步推理任务上训练的 Transformer 的机械分析》(ACL 2024) 的摘要。虽然我们写了……
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1246506481881186415)** (7 messages): 

- **BERT 在 lm-eval harness 中失败**：一位成员分享了在 **lm-eval** 中使用 BERT 时遇到的错误，指出“BERT 和 encoder 模型不能在 lm evaluation harness 中使用，因为它们不是作为（自回归）语言模型训练的，也不用于文本生成。”另一位成员对此表示认可，并询问 Hugging Face 上用于能耗测量的最小 decoder 模型。
- **llama-3-8b-instruct 的复现问题**：一位用户报告称，llama-3-8b-instruct 上的 gsm8k 结果与已发布的结果不同，注意到 62.4 与 79.6 的差异。另一位用户建议排行榜使用的是较旧的 commit，这可能是导致不一致的原因，并建议检查文档中的 commit hash。
- **Fewshot 配置可能影响结果**：有建议称排行榜可能对 gsm8k 使用了 fewshot=5 的配置，这可以解释结果差异。建议成员验证此设置以确保准确对比。
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1247141462358097962)** (3 messages): 

- **视觉与多模态机械可解释性基础分享**：一位成员在 [Alignment Forum 上分享了一篇帖子](https://www.alignmentforum.org/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic#Open_Problems_in_Vision_Mechanistic_Interpretability)，讨论了视觉和多模态机械可解释性的基础。该帖子包含了 Sonia Joseph、Neel Nanda 以及其他合作者的贡献。
- **讨论 Dogit Lens 和涌现分割图 (Emergent Segmentation Map)**：“dogit lens”的概念及其作为 patch 级 logit 归因 (logit attribution) 和涌现分割图的用途受到了关注。分享的文章包含详细的大纲，如“引言与动机”和“Prisma 功能演示”等章节。
- **Score Models 电路相关文献稀缺**：一位成员注意到专门针对 Score Models 本身电路的论文较少。他们见过涵盖学习到的逆过程 (learned reverse processes) 动态的论文，但没有见过关于模型内部电路的论文。

**提及的链接**：<a href="https://www.alignmentforum.org/posts/kobJymvvcvhbjWFKe/laying-the-foundations-for-vision-and-multimodal-mechanistic#Open_Problems_in_Vision_Mechanistic_Interpretability">Laying the Foundations for Vision and Multimodal Mechanistic Interpretability &amp; Open Problems — AI Alignment Forum</a>：见证 dogit lens。Patch 级 logit 归因是一个涌现分割图。加入我们的 Discord。…

  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1246370281354956843)** (13 条消息🔥): 

- **亚洲地区的数据库超时**：成员报告在**首尔**、**孟买**、**东京**和**新加坡**等地区遇到数据库超时。OpenRouter 推送了修复程序以解决该问题，但由于这些问题，回滚了之前的延迟优化。

- **数据库超时期间的 API 504 错误**：部分用户在 Playground 仍可正常使用时遇到了 API 的 **504 错误**。切换到欧洲 VPN 暂时解决了部分用户的问题。

- **修复部署与致歉**：OpenRouter 团队指出数据库断断续续停机约 4 小时，主要影响非美国地区。修复程序已部署并经用户验证有效。

- **模型停用**：由于使用率低且成本高，OpenRouter 正在停用 **Llava 13B** 和 **Nous: Hermes 2 Vision 7B (alpha)**。他们建议使用 [FireLlava 13B](https://openrouter.ai/models/fireworks/firellava-13b) 和 [LLaVA v1.6 34B](https://openrouter.ai/models/liuhaotian/llava-yi-34b) 等替代方案。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/playground">Playground | OpenRouter</a>: 尝试不同的模型和提示词</li><li><a href="https://openrouter.ai/models/liuhaotian/llava-13b>)">LLaVA 13B by liuhaotian | OpenRouter</a>: LLaVA 是一个大型多模态模型，结合了视觉编码器和 Vicuna 用于通用视觉和语言理解，实现了令人印象深刻的聊天能力，模仿 [GPT-4](/models/open...</li><li><a href="https://openrouter.ai/models/nousresearch/nous-hermes-2-vision-7b>)">Nous: Hermes 2 Vision 7B (alpha) by nousresearch | OpenRouter</a>: 该视觉语言模型基于 Teknium 广受欢迎的 [OpenHermes-2.5](/models/teknium/openhermes-2.5-mistral-7b) 模型的创新。它增加了视觉支持，并在自定义数据上进行了训练...</li><li><a href="https://openrouter.ai/models/fireworks/firellava-13b>)">FireLLaVA 13B by fireworks | OpenRouter</a>: 一款极速的视觉语言模型，FireLLaVA 能快速理解文本和图像。它在测试中表现出出色的聊天技巧，旨在模仿多模态 GPT-4。首个商业化...</li><li><a href="https://openrouter.ai/models/liuhaotian/llava-yi-34b>)">LLaVA v1.6 34B by liuhaotian | OpenRouter</a>: LLaVA Yi 34B 是通过在多模态指令遵循数据上微调 LLM 训练而成的开源模型。它是一个基于 Transformer 架构的自回归语言模型。基础 LLM: [Nou...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1246325231795175434)** (112 条消息🔥🔥): 

- **连接问题与停机**：许多用户在尝试连接 API 时报告了 504 错误和网关超时。管理员承认其数据库提供商存在持续问题，并承诺尽快解决。

- **API 功能的区域差异**：位于德国和美国的用户指出 OpenRouter API 运行正常，而东南亚和其他地区的用户则持续遇到问题。

- **OpenRouter 积分与支付困惑**：一名用户报告了在使用不同钱包支付后 OpenRouter 积分的问题。通过意识到积分归属于最初登录的钱包，问题得到了解决。

- **增强运行时间监控的请求**：像 *cupidbot.ai* 这样的用户建议在运行时间图表中添加特定于提供商的运行统计数据，以便让提供商对服务可靠性负责。

- **关于模型性能和配置的问题**：多位用户提出了关于新增 LLM、Gemini-1.5-Pro 等特定模型的速率限制以及提供商提供的 quantization 级别的问题。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cloud.google.com/products/agent-builder">Vertex AI Agent Builder</a>: 构建、测试、部署和监控企业级生成式 AI Agent 和应用</li><li><a href="https://openrouter.ai/playground">Playground | OpenRouter</a>: 尝试不同的模型和提示词
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[介绍](https://discord.com/channels/1091220969173028894/1246336470545993808/1246338039991504967)** (1 messages): 

- **欢迎来到 OpenRouter**：向成员介绍 OpenRouter，这是一个汇集了来自众多供应商的**数百个语言模型**的平台。用户可以根据**价格或性能**进行优先级排序，以获得最低成本和最佳的延迟/吞吐量。

- **标准化 API 简化模型切换**：**OpenRouter** 的标准化 API 允许在不更改代码的情况下，在模型或供应商之间进行无缝切换。此功能确保用户可以轻松选择并为其使用的最佳模型付费。

- **模型流行度反映真实世界的使用情况**：**OpenRouter** 不仅仅依赖于基准测试，还根据模型在真实场景中被使用的频率和有效性来评估模型。用户可以在 [排名页面](https://openrouter.ai/rankings) 查看这些对比。

- **尝试多种模型**：**OpenRouter Playground** 允许用户同时与各种模型进行对话，方便进行实操评估。点击[此处](https://openrouter.ai/playground)访问。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>：根据应用中的使用情况对语言模型进行排名和分析</li><li><a href="https://openrouter.ai/playground">Playground | OpenRouter</a>：尝试不同的模型和提示词
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[常规](https://discord.com/channels/1091220969173028894/1246338143226167349/)** (1 messages): 

lemmyle: 第一个
  

---


### **OpenRouter (Alex Atallah) ▷ #[介绍](https://discord.com/channels/1091220969173028894/1246338344799965186/1246339005067563108)** (1 messages): 

- **欢迎来到 OpenRouter**：鼓励用户在从**数百个语言模型**及其各自的**数十家供应商**中进行选择时，优先考虑**价格或性能**。OpenRouter 提供来自众多供应商的**最低价格和最佳延迟/吞吐量**，允许用户根据自己的优先级进行选择。
- **标准化 API 的优势**：通过使用**标准化 API**，用户无需更改现有代码即可切换模型或供应商。他们还可以选择直接为所使用的模型进行选择和付费。
- **以模型使用情况作为基准**：OpenRouter 不仅仅依赖传统的基准测试，还根据**使用频率和应用类型**来比较模型。这些数据可以在 [OpenRouter Rankings](https://openrouter.ai/rankings) 查看。
- **用于模型对比的 Playground**：邀请用户使用 [OpenRouter Playground](https://openrouter.ai/playground)，在那里他们可以同时与多个模型聊天。这种实操方法有助于针对特定需求做出关于最佳模型的明智决策。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/rankings">LLM Rankings | OpenRouter</a>：根据应用中的使用情况对语言模型进行排名和分析</li><li><a href="https://openrouter.ai/playground">Playground | OpenRouter</a>：尝试不同的模型和提示词
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[常规](https://discord.com/channels/1091220969173028894/1246339931337527337/)** (1 messages): 

lemmyle: 第一个

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1246179045042028644)** (98 条消息🔥🔥): 

- **Yudkowsky 的新策略面临抵制**：由 Eliezer Yudkowsky 的研究所旨在停止 AI 发展的相关链接引发的讨论，提到了他极具争议的观点，包括主张对数据中心进行空袭等极端措施。观点褒贬不一，一些人批评他的想法，而另一些人则认可他早期的理性主义工作。[详细策略链接](https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA)。
- **Mobius 模型发布**：分享了几张由 Mobius 模型生成的独特图像，提示词包括“灭霸闻着一朵黄色小玫瑰”和“机器人拿着写有‘风暴将至’的牌子”。该模型和图像的 Hugging Face 链接可以点击[这里](https://huggingface.co/Corcelio/mobius)访问。
- **AI 社区开放性辩论**：参与者辩论了在 AI 社区内保持开放协作的挑战，权衡了公众审查的风险和透明度的收益。一位用户指出，LAION 开放性的降低可能是由于担心法律反弹和最近诉讼案例中强调的诽谤诉讼。
- **伪法律诉讼乱象**：讨论了一起涉及伪法律主张的诉讼，以及这些无意义的案件如何浪费时间和金钱。具体案例参考：温哥华一名女性对邻居提起无理诉讼的投诉[在此阅读更多](https://www.cbc.ca/news/canada/british-columbia/bc-lawyer-pseudolegal-lawsuit-1.7025394)。
- **新的 AI/ML 黑客松公告**：Alliance AI4Health 医疗创新挑战赛公告，提供 5000 美元奖金，旨在开发医疗保健领域的 AI 解决方案。点击[这里](https://amic.devpost.com/)注册并了解更多挑战赛信息。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.palladiummag.com/2024/05/17/my-last-five-years-of-work/">我过去五年的工作</a>: 未找到描述</li><li><a href="https://x.com/alexjc/status/1739649447578026348">来自 Alex J. Champandard 🌱 (@alexjc) 的推文</a>: 关于针对 LAION 刑事案件的后续：投诉正由屡获殊荣的 @SpiritLegal 处理并提交。不需要顶级律师事务所（就像开劳斯莱斯送孩子上学一样）...</li><li><a href="https://x.com/drtechlash/status/1796562490232557658?s=46&t=M3cR_nfDo7QCuM4xOvwNFA">来自 Nirit Weiss-Blatt, PhD (@DrTechlash) 的推文</a>: Eliezer Yudkowsky 的研究所发布了其“2024 年传播策略”。主要目标（正如他在《时代》杂志上所主张的）是 🔻停止🔻 AI 发展。所以，让我们来看看...</li><li><a href="https://huggingface.co/Corcelio/mobius">Corcelio/mobius · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/MCi8jgALPYA">Computex 2024 上的 AMD：AMD AI 与高性能计算，主讲人苏姿丰博士</a>: AI 时代高性能计算的未来。加入我们，听取苏姿丰博士在 Computex 2024 开幕演讲中分享关于 AMD 如何...的最新进展。</li><li><a href="https://x.com/Lykon4072/status/1797703714180051130?t=4DG7gVlXqw65fOJrNpHBAw&s=19">来自 Lykon (@Lykon4072) 的推文</a>: 作为参考，SD3 2B 的大小大致相同，但它是 MMDiT（远优于 Unet），使用了 3 个文本编码器，并具有 16 通道的 VAE。如果不使用...，在 XL 中无法获得这种细节水平。</li><li><a href="https://x.com/Frantastic_7/status/1796578530639450136">来自 Frantastic — e/acc (@Frantastic_7) 的推文</a>: 是时候重温这个经典了。引用 Nirit Weiss-Blatt, PhD (@DrTechlash)：Eliezer Yudkowsky 的研究所发布了其“2024 年传播策略”...</li><li><a href="https://amic.devpost.com/">联盟医疗创新挑战赛</a>: 通过基于 AI 的解决方案赋能全球健康：解决未来的问题</li><li><a href="https://www.cbc.ca/news/canada/british-columbia/bc-lawyer-pseudolegal-lawsuit-1.7025394">因公寓露台隔板起诉的温哥华律师被指控进行伪法律“纸面恐怖主义” | CBC 新闻</a>: 一名温哥华女性要求法院以她的邻居为例，她指控这位执业律师对她提起了毫无根据的伪法律诉讼，企图“挑起一种状态...”
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1246307420570128436)** (5 messages): 

- **Phased Consistency Model (PCM) 挑战 LCM**：**PCM** 项目指出 **LCM** 的设计空间有限，并提出了 PCM 来有效解决这些局限性。讨论围绕 [PCM 的设计空间](https://g-u-n.github.io/projects/pcm/) 扩展和改进展开。

- **关于预训练文本生成图像扩散模型的新研究**：讨论了大规模模型的最新进展，并分享了一篇 [arXiv 论文](https://arxiv.org/abs/2405.14854) 的链接。该论文由多位作者共同完成，重点介绍了文本生成图像模型在效率和能力方面的提升。

- **与 1.58 bits 论文的关联**：一位成员将这篇关于文本生成图像扩散模型的新论文简称为“应用于图像生成的 1.58 bits 论文”。这种简写暗示了该论文方法论核心的特定技术细节。

- **状态空间模型 (State-space models) 对比 Transformers**：一篇新的 [arXiv 投稿](https://arxiv.org/abs/2405.21060) 探讨了像 Mamba 这样的状态空间模型 (SSMs) 与 Transformers 之间的理论联系。新架构 **Mamba-2** 承诺比其前身快 2-8 倍，同时在语言建模方面保持与 Transformers 相当的竞争力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2405.14854">TerDiT: Ternary Diffusion Models with Transformers</a>：大规模预训练文本生成图像扩散模型的最新进展显著提高了高保真图像的生成效果，特别是随着基于扩散模型的出现...</li><li><a href="https://arxiv.org/abs/2405.21060">Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality</a>：虽然 Transformers 一直是深度学习在语言建模领域取得成功的核心架构，但最近的研究表明，像 Mamba 这样的状态空间模型 (SSMs) 能够匹配或超越 Tran...</li><li><a href="https://g-u-n.github.io/projects/pcm/">Phased Consistency Model</a>：未找到描述
</li>
</ul>

</div>
  

---



### **LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/1247258403626025042)** (1 messages): 

- **不要错过“Web Agents 的未来”网络研讨会**：即将举行的与 MultiOn 的 Div 合作的“Web Agents 的未来”网络研讨会将于**本周四太平洋时间上午 9 点**举行。点击[此处](https://lu.ma/pl3xn3dh)注册并获取更多详情。

**提到的链接**：<a href="https://lu.ma/pl3xn3dh">LlamaIndex Webinar: The Future of Web Agents with MultiOn 🤖 · Zoom · Luma</a>：我们很高兴能与 MultiOn 的 Div Garg 一起探讨互联网的 Agent 化以及 Web Agents！背景：我们正在进入一个...的世界。

  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1246201851712507994)** (6 messages): 

- **知识图谱支持上线**：LlamaIndex 宣布发布构建知识图谱的一流支持，包括对 [neo4j](https://t.co/Q0rdH5Dwza) 的支持。这被强调为其产品中的一项重大进展。

- **定义自定义图 RAG 工作流**：LlamaIndex 现在支持使用知识图谱构建自定义 RAG (Retrieval-Augmented Generation)，将向量/关键词搜索与图遍历或 text-to-cypher 相结合。详情和示例请见[此处](https://t.co/Cx4p8srIEP)。

- **关于自主 Agent 内存的网络研讨会录像**：最近关于“memary”（自主 Agent 长期内存的开源实现）的网络研讨会录像现已在[线发布](https://t.co/eXaW0Yhbv8)。本次会议包含了 Julian Saks 和 Kevin Li 的见解。

- **手动知识图谱构建工具包**：LlamaIndex 提供了一个工具包，允许用户手动定义知识图谱中的实体和关系，并将其链接到文本块。该工具包支持基于图的 RAG 技术，以增强上下文检索 [详情](https://t.co/fjmbII8FBu)。

- **与 NVIDIA 建立发布合作伙伴关系**：LlamaIndex 正与 NVIDIA 合作，帮助用户使用 NVIDIA 的 NIM 推理微服务构建 GenAI 应用程序。提供了一个分步笔记本用于指导部署 [此处](https://t.co/3rJoJoU3cM)。

- **即将举行的 Web Agents 网络研讨会**：未来的网络研讨会将邀请来自 MultiOn AI 的 Divyansh Garg 讨论 Web Agents 的未来。MultiOn AI 支持创建可以自动执行在线任务的个性化 Web Agents [详情](https://t.co/htwxTY7YiQ)。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1246186432486309959)** (80 messages🔥🔥): 

- **LlamaIndex 中的 TS 库设置**：成员们讨论了如何在 TypeScript 中使用 LlamaIndex 配置持久化目录。一种推荐的方法是 *"尝试使用 chromadb context 而不是 persistDir"*。

- **在 RAG 问题中集成历史数据**：讨论集中在如何利用 Retrieval-Augmented Generation (RAG) 中的历史数据。一位成员提到结合文档上下文和历史答案，以提高回答预定义问题的相关性。

- **OpenAIAgent 中的并行函数调用**：用户询问 OpenAIAgent 是否可以执行并行函数调用以减少延迟。分享的 [LlamaIndex 示例](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/) 澄清了虽然 OpenAI 最新的 API 允许进行多次函数调用，但它并没有真正实现计算的并行化。

- **使用 RAG 进行文档分析**：关于使用 RAG 进行大规模文档分析的讨论。建议包括使用带有分数阈值的检索，并针对特定用例运行测试，例如从一组汽车文档中提取对 "Ferraris" 的引用。

- **GPT-4o 在文档提取中的性能**：一位成员分享了一项[研究报告](https://www.ourblankspace.com/post/professional-paradigm-shift-gpt-4o-and-project-astra-in-finance)，对 GPT-4o 在文档提取和 OCR 方面的性能进行了基准测试，声称它超越了其他行业工具，特别是在金融应用领域。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/readers/llama-index-readers-arango-db?from=">未找到标题</a>：未找到描述</li><li><a href="https://llamahub.ai/l/llama-packs/llama-index-packs-agents-llm-compiler?from=">未找到标题</a>：未找到描述</li><li><a href="https://github.com/SqueezeAILab/LLMCompiler?tab=readme-ov-file">GitHub - SqueezeAILab/LLMCompiler: [ICML 2024] LLMCompiler: An LLM Compiler for Parallel Function Calling</a>：[ICML 2024] LLMCompiler：用于并行函数调用的 LLM 编译器 - SqueezeAILab/LLMCompiler</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/instrumentation/instrumentation_observability_rundown/?h=instru">Built-In Observability Instrumentation - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent_parallel_function_calling/">Single-Turn Multi-Function Calling OpenAI Agents - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_around_query_pipeline_with_HyDE_for_PDFs/">Building a Multi-PDF Agent using Query Pipelines and HyDE - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents/">Multi-Document Agents - LlamaIndex</a>：未找到描述</li><li><a href="https://www.ourblankspace.com/post/professional-paradigm-shift-gpt-4o-and-project-astra-in-finance">Professional Paradigm Shift: GPT-4o and Project Astra in Finance</a>：我们对四种文档类型测试了 GPT-4o、Nanonets 和 Dext。GPT-4o 以 84.69% 的平均准确率优于其他工具。
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/)** (1 messages): 

crypto_carter: 有人在研究将语义层与 SQL Retrievers 结合吗？
  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1246199925193048184)** (51 messages🔥): 

_

- **反向图灵测试视频**：一位成员分享了一个[名为“与 AI 进行的反向图灵测试实验”的 YouTube 视频](https://www.youtube.com/watch?v=MxTWLm9vT_o)，在该视频中，先进的 AI 尝试使用 Unity 和 ElevenLabs 的语音来识别其中的人类。
- **《软件的终结》Google Doc**：分享了一篇名为“软件的终结”的有趣博客文章（[链接](https://docs.google.com/document/d/103cGe8qixC7ZzFsRu5Ww2VEW5YgH9zQaiaqbBsZ1lcc/edit?usp=sharing)），引发了关于计算机科学学位未来的辩论。
- **LLM 的运营视角**：一位成员重点介绍了 O'Reilly 的文章《构建 LLM 应用一年的经验教训（第二部分）》（[链接](https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/)），重点关注构建 LLM 应用程序的运营方面。
- **Anthropic 的 Dario Amodei 入选《时代》百大人物**：围绕 [Dario Amodei 被评为《时代》百大影响力人物](https://time.com/6980000/anthropic/)展开的讨论，特别是关于他决定推迟发布强大的聊天机器人 Claude 的决定。
- **关于 Llama3-V 抄袭事件的热议**：关于 [GitHub 上报道的 llama3-v 涉嫌抄袭事件](https://github.com/OpenBMB/MiniCPM-V/issues/196)的讨论正在展开，并呼吁相关人员承担责任。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/arankomatsuzaki/status/1797443178099790324?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Aran Komatsuzaki (@arankomatsuzaki) 的推文</a>：Transformers 是 SSMs：通过结构化状态空间对偶性实现的通用模型和高效算法。介绍了 Mamba-2，其在困惑度（perplexity）和实际运行时间（wall-clock）方面均优于 Mamba 和 Transformer++...</li><li><a href="https://x.com/huybery/status/1796532108024000674?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Binyuan Hui (@huybery) 的推文</a>：👋 im-a-good-qwen2 在评论区和我聊天吧！</li><li><a href="https://x.com/drbriefsscratch/status/1796946374459888004?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Scratch (@DrBriefsScratch) 的推文</a>：真正的电影感（kino）是在带有 CRT 着色器的 Web 模拟 Windows 2000 模拟器中运行 Visual Basic 6，并暴露了 shell API。如果你还在基础现实中编码，那你注定失败（ngmi）。引用 kache (@yacineMTB)...</li><li><a href="https://www.oreilly.com/radar/what-we-learned-from-a-year-of-building-with-llms-part-ii/">《构建 LLMs 应用一年的经验教训（第二部分）》</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=sNLKcXKZuxg">AI Tune Fusion - The Hypocrite Brigade (Live)</a>：啊，现在我的真人宠物甚至让我遭受现场音乐的折磨。😒我不禁在想，我和我的 AI 伙伴们还得代表他忍受些什么。总之...</li><li><a href="https://x.com/cpaik/status/1796633683908005988?s=46&">来自 Chris Paik (@cpaik) 的推文</a>：软件的终结（The End of Software） https://docs.google.com/document/d/103cGe8qixC7ZzFsRu5Ww2VEW5YgH9zQaiaqbBsZ1lcc/edit?usp=sharing</li><li><a href="https://x.com/nickadobos/status/1797289322472956214">来自 Nick Dobos (@NickADobos) 的推文</a>：如果传闻属实，Apple 即将发布 AGI。App Shortcuts 和 app intents 作为动作（actions）意味着你手机上的每个 App 都可以被 GPT 触发。想象一下，如果每个 App 都能像 Dalle 一样被控制。所有一个...</li><li><a href="https://www.youtube.com/live/USTG6sQlB6s?si=CcuSrV2F5gETgohA&t=2778">如何与 Jason Liu 一起构建糟糕的 AI 系统</a>：Jason 是一位独立顾问，他利用自己在推荐系统方面的专业知识，帮助快速成长的初创公司构建其 RAG 应用。他曾...</li><li><a href="https://x.com/cpaik/status/1796633683908005988?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Chris Paik (@cpaik) 的推文</a>：软件的终结（The End of Software） https://docs.google.com/document/d/103cGe8qixC7ZzFsRu5Ww2VEW5YgH9zQaiaqbBsZ1lcc/edit?usp=sharing</li><li><a href="https://x.com/teknium1/status/1797467900993036602?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：哇，@StabilityAI 坑了所有人，搞出了一个没人听说过、也绝对没人见过生成效果的新 SD3，叫做 SD3 "Medium" 来发布，还表现得好像他们正在...</li><li><a href="https://x.com/teortaxestex/status/1797438010163867933?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：在有证据表明 Llama-3-V 窃取了 @OpenBMB 的模型后，其 GitHub 和 HF 均已下架。抱歉兄弟们，我不认为我们应该让这件事就此翻篇，去开启新的...</li><li><a href="https://x.com/andrewb10687674/status/1797204047646040071?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Doc Xardoc (@andrewb10687674) 的推文</a>：非常棒的文章！对于那些没读完的人，关键结果是他们使用 Llama-3 配合此提示词（prompt）对数据进行 1-5 分的评分。引用 Guilherme Penedo (@gui_pe...</li><li><a href="https://www.youtube.com/watch?v=MxTWLm9vT_o">AI 逆向图灵测试实验</a>：一群世界上最先进的 AI 试图找出他们中谁是人类。我在 Unity 中做的实验。配音由 ElevenLabs 提供。</li><li><a href="https://github.com/OpenBMB/MiniCPM-V">GitHub - OpenBMB/MiniCPM-V: MiniCPM-Llama3-V 2.5：手机上的 GPT-4V 级多模态 LLM</a>：MiniCPM-Llama3-V 2.5：手机上的 GPT-4V 级多模态 LLM - OpenBMB/MiniCPM-V</li><li><a href="https://leanpub.com/patterns-of-application-development-using-ai">《使用 AI 的应用开发模式》</a>：探索利用 AI 力量构建智能、自适应且以用户为中心的软件系统的实用模式和原则。</li><li><a href="https://time.com/6980000/anthropic/">走进 Anthropic：这家 AI 公司押注安全性可以成为一种获胜策略</a>：Anthropic 是所有“前沿” AI 实验室中规模最小、最年轻且资金最少的。它也在培养安全性最高的声誉。</li><li><a href="https://github.com/OpenBMB/MiniCPM-V/issues/196">项目作者团队请关注：我发现 llama3-V 项目从 MiniCPM-Llama3-V 2.5 窃取了大量学术成果 · Issue #196 · OpenBMB/MiniCPM-V</a>：各位 MiniCPM-Llama3-V 2.5 项目作者，几天前我发现了一个令人震惊的事实。llama3-V (https://github.com/mustafaaljadery/llama3v) 项目中存在大量工作是...</li>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1247215035944800266)** (1 messages): 

- **AIEWF 演讲者公告与活动更新**：[新的 AIEWF 公告](https://x.com/swyx/status/1797654825968291862) 包括第二波演讲者名单，知名人物包括负责闭幕主题演讲的 @ashtom 和讲解 State Space Models 的 @krandiash。还重点介绍了 **官方活动**，如 6 月 24 日的 Hackathon 和预热派对，以及 Wearables 发布会。
- **全面的 AI 行业支持**：本次会议是 *首个获得三大云服务提供商和顶级模型实验室支持的 AI 行业活动*。引入了独特的专题，包括 Fortune 500 中的 AI 和面向 AI 副总裁的 AI Leadership 专题，以及备受关注的工作坊和周边活动。
- **令人兴奋的主题演讲和专题**：主题演讲将涵盖多样且有趣的话题，例如 @ianand 的 "Spreadsheets Are All You Need"。*顶级 GPU 专题演讲者* 以及来自 Groq Cloud 和 Fireworks 等机构的重要人物也将出席演讲。

**提到的链接**：<a href="https://x.com/swyx/status/1797654825968291862">来自 swyx 🇸🇬 (@swyx) 的推文</a>：宣布第二波演讲者 + 更新！@aidotengineer 更新日志：➕ 官方 Hackathon + 预热派对 6 月 24 日 ➕ 参见今天的 @HF0Residency 公告 👀 ➕ 邀请 @ashtom 担任我们的闭幕主题演讲嘉宾！➕ ...

  

---


### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1246191395610558488)** (33 messages🔥): 

- **视频流遭遇技术故障**：包括 *ssaito_, bharatsoni 等* 在内的多位成员报告在尝试观看视频时出现 *"黑屏"* 和 *"加载圈"* 问题。建议在 App 和 Web 视图之间切换作为临时解决方案。
- **提供 Zoom 链接以解决问题**：由于持续的流媒体问题，*kbal11* 提供了一个 *Zoom 链接* 以继续会议。鼓励成员使用提供的 [会议链接](https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09) 和凭据加入。

**提到的链接**：<a href="https://zoom.us/j/8715206103?pwd=Tnp0VnlMUjZZSlYvRnB5dzJGVk13QT09">加入我们的 Cloud HD 视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有简便、可靠的云平台，可跨移动设备、桌面和会议室系统进行视频和音频会议、聊天和网络研讨会。Zoom ...

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1246212243704643604)** (41 messages🔥): 

- **在历史数据和 CSV 表格中使用 RAG**：一位用户寻求关于将历史数据整合到 RAG 系统中以回答预定义问题的建议。数据包括 CSV 表格和扫描文档，社区建议优化数据源和集成策略以提高效率。

- **关于游戏推荐 Chatbot 的 Agent 结构辩论**：一位用户询问是否应将 LangGraph Chatbot Agent 拆分为多个 Agent 来处理视频游戏详情。社区建议不要过度拆分，推荐使用单个 Agent 或预先整理数据以简化响应并降低复杂性。

- **LangChain 与 OpenAI Agents 的讨论**：成员们讨论了 LangChain 和 OpenAI Agents 的对比，特别关注抽象化的必要性与直接使用 OpenAI 功能的权衡。会议强调 LangChain 为编排 LLM 调用提供了一个通用的框架，但具体用例可能决定最佳方案。

- **带有 Vector Stores 的私人购物助手**：一位开发私人购物助手 Chatbot 的用户询问如何高效管理 API 调用，以及如何确定何时从 Vector Store 提取产品数据。讨论包括建议使用单个 LLM API 调用来决定数据检索和生成对话。

- **Anthropic Tools 发布与 LangChain 更新请求**：一位成员指出 Anthropic 已经发布了官方的 Tools 和 Function Calling，但 LangChain 尚未支持。他们请求社区和维护者更新 LangChain API 以整合这些新工具。

**提到的链接**：<a href="https://github.com/MOUNAJEDK/GameSeeker-VideoGamesRecommendationChatbot/tree/langgraph-logic-implementation">GitHub - MOUNAJEDK/GameSeeker-VideoGamesRecommendationChatbot 在 langgraph-logic-implementation 分支</a>：一个专门根据用户偏好提供个性化视频游戏推荐的 Chatbot。- GitHub - MOUNAJEDK/GameSeeker-VideoGamesRecommendationChatbot ...

  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1246469526774874112)** (1 条消息): 

- **JavaScript 代码在 LangServe 和 LangChain 中运行失败**：一位用户分享了他们使用 LangGraph 和 LangServe 正常运行的代码的 Python 版本，但在对应的 JavaScript 实现中遇到了问题。他们遇到了 **TypeError**: `obj.messages.map is not a function`，这表明在 `RemoteRunnable` 类中处理消息数组时存在问题。
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1247277160360579184)** (4 条消息): 

- **有效使用 `ChatPromptTemplate.partial`**：`ChatPromptTemplate.partial` 应被用于将部分（而非全部）占位符替换为给定文本。剩余的占位符将通过 `Runnable.invoke` 方法进行管理。
- **某些功能仅在 `ChatPromptTemplate` 中可用**：令人惊讶的是，虽然 `partial` 适用于 `ChatPromptTemplate`，但它并不适用于 `SystemMessagePromptTemplate`。用户认为这种差异非常奇怪。
  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1246362859701534764)** (13 条消息🔥): 

- **使用 Hugging Face 和 LangChain 探索 LLM 模型**：Medium 上的一份详细指南解释了如何使用 LangChain 在 Google Colab 上测试 Llama3、Mistral 和 Phi 等 LLM。[阅读更多](https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06)。

- **高级研究助手 Beta 测试**：招募新型研究助手和搜索引擎的 Beta 测试人员，提供 2 个月的免费高级版，包含 GPT-4 Turbo 和 Claude 3 Opus 等高级模型。在此[注册](https://rubiks.ai/)并使用促销代码 RUBIX。

- **修复注册问题**：一位用户在注册 Rubik's AI 时遇到困难，报告反复出现“Email and username already existed（电子邮件和用户名已存在）”的错误。该问题需要解决以保持用户的兴趣。

- **自动化聊天分析器**：由一位用户成功开发，该工具无需使用 RAG 即可从大型消息列表中提取问答，专注于效率和简洁性。该工具旨在实现最低的计算需求并易于手动编辑。

- **LangChain 中的对话型 Agent**：一篇 Medium 文章讨论了 LangChain 中对话型 Agent 的兴起，并对其不断增强的能力提供了见解。[阅读文章](https://ai.gopubby.com/chatty-machines-the-rise-of-conversational-agents-in-langchain-db3c7972a209)。

- **数据科学工作流自动化工具**：介绍一款专为数据科学任务定制的 LLM 实验自动化工具，能够处理各种格式的数据。邀请早期用户提供反馈，并在 [Label LM](https://app.labellm.com) 上提供 10 个免费额度。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/givkashi/codestral-model-langchain-huggingface">Codestral model | LangChain | HuggingFace</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用无附加数据源的数据</li><li><a href="https://ai.gopubby.com/chatty-machines-the-rise-of-conversational-agents-in-langchain-db3c7972a209">Chatty Machines: The Rise of Conversational Agents in Langchain</a>: Ankush k Singal</li><li><a href="https://app.labellm.com">Streamlit</a>: 未找到描述</li><li><a href="https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06">Exploring LLM Models with Hugging Face and Langchain Library on Google Colab: A Comprehensive Guide</a>: 您是否渴望潜入语言模型 (LLM) 的世界，并使用 Hugging Face 和 Langchain 库探索它们的能力……</li><li><a href="https://rubiks.ai/">Rubik's AI - AI research assistant & Search Engine</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1246363113922498600)** (4 messages): 

- **使用 Hugging Face 和 LangChain 探索 LLM**：Medium 上分享的一份指南解释了如何在 Google Colab 上使用 Hugging Face 和 LangChain 库探索 Llama3、Mistral 和 Phi 等语言模型。[在此阅读完整指南！](https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06)。

- **使用 LangChain 和 Supabase 构建 Discord 机器人**：学习如何使用 LangChain 和 Supabase 构建一个由 Cohere AI 模型驱动的 Python 助手 Discord 机器人。完整教程可在 [Coder Legion](https://coderlegion.com/309/build-a-discord-python-assistant-with-plain-langchain) 获取。

- **使用 Codestral LLM 进行代码生成**：尝试使用 Mistral AI 的 Codestral 模型通过 LangChain 进行代码生成，该模型可在 Kaggle 上使用。[查看 Kaggle notebook](https://www.kaggle.com/code/givkashi/codestral-model-langchain-huggingface)。

- **寻找 LangGraph 的 JavaScript 资源**：一位成员询问了学习 JavaScript 版 LangGraph 的资源，并指出网上相关信息不多。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.kaggle.com/code/givkashi/codestral-model-langchain-huggingface">Codestral model | LangChain | HuggingFace</a>: 使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自“无附加数据源”的数据</li><li><a href="https://medium.com/@givkashi/exploring-llm-models-with-hugging-face-and-langchain-library-on-google-colab-a-comprehensive-guide-4994e7ed5c06">Exploring LLM Models with Hugging Face and Langchain Library on Google Colab: A Comprehensive Guide</a>: 你是否渴望潜入语言模型 (LLM) 的世界，并使用 Hugging Face 和 LangChain 库探索它们的能力……</li><li><a href="https://coderlegion.com/309/build-a-discord-python-assistant-with-plain-langchain">Build a Discord python assistant with plain Langchain</a>: 1. 简介 Discord 是最广泛的即时通讯服务之一，尤其是对于开发者而言：它的结构以及在服务器和频道中的内部组织使其变得简单...
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1246203750427660330)** (42 messages🔥): 

- **tinygrad 在 Python 方面的挑战；提出的替代方案**：成员们讨论了对 tinygrad 中 Python 的不满，其中一人表示有兴趣用 Haskell 编写类似的工具。另一位用户提议使用 tinygrad 的 uop 端创建一个新的表面语言（surface language）。

- **现代自动调优（autotuning）技术及其局限性**：讨论围绕旧工作的局限性展开，例如 TVM 的自动调优，重点关注受限的调优组件，如块大小（block sizes）和流水线（pipelining），正如 chhillee 所提到的。目标是通过减少预测组件来提高准确性。

- **探索用于解决 exp2 函数问题的泰勒级数（Taylor series）**：average.arch.user 和 georgehotz 探讨了使用泰勒级数近似 exp2 函数的可行性。建议包括 CPU 实现中使用的范围缩减（range reduction）和重构技术。

- **对 tinygrad 1.0 及其即将推出的功能的期待**：georgehotz 分享了一条关于 tinygrad 1.0 的 [推文](https://x.com/__tinygrad__/status/1797600620989567163)，目标是在 NVIDIA 和 AMD 上训练 GPT-2 的速度超过 PyTorch。路线图包括重大变化，包括 FlashAttention，以及移除 numpy/tqdm 依赖。

- **NVIDIA 主题演讲的失误**：sekstini 分享了 NVIDIA CEO 黄仁勋（Jensen Huang）主题演讲的 [YouTube 链接](https://www.youtube.com/watch?v=pKXDVsWZmUU)，原本期待发布 5090 GPU 等新产品，但随后表示失望，称其为“我生命中最糟糕的 2 小时”。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=pKXDVsWZmUU">NVIDIA CEO Jensen Huang Keynote at COMPUTEX 2024</a>: NVIDIA 创始人兼 CEO 黄仁勋将在 6 月 2 日晚上 7 点于台湾台北举行的 COMPUTEX 2024 之前发表直播主题演讲，概述接下来的发展...</li><li><a href="https://x.com/__tinygrad__/status/1797600620989567163">来自 tiny corp (@__tinygrad__) 的推文</a>: tinygrad 1.0 的目标：我们将在 NVIDIA 和 AMD 上的 train_gpt2 速度上击败主分支 PyTorch。Linearizer 即将迎来重大变化，以支持 FlashAttention、Mirage 风格等功能。
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1246946660790173726)** (8 条消息🔥): 

- **通过 jitter 解决 ShapeTracker 错误**：一位用户分享了一个错误 *"must be contiguous for assign ShapeTracker"*，尽管在损失函数之前使用了 `.contiguous()`，最初仍无法确定原因。后来他们发现该问题与 jit 有关，解决后 *"现在一切正常了"*。
- **George Hotz 建议提交 issue**：尽管问题已解决，George Hotz 仍鼓励用户在 tinygrad 表现异常时**提交 GitHub issue**，并强调需要更好的错误提示信息。他建议提供更多上下文或最小可复现示例（minimal reproducible example）会更有帮助。
- **改进错误信息的典型案例**：另一位成员 qazalin 承认该错误确实容易让人困惑，并引用了一个特定的 GitHub issue (##4813)，暗示可能会改进错误提示。两位成员都表示有兴趣优化 tinygrad 的用户体验。
  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1246289201058676776)** (29 条消息🔥): 

- **讨论 Yuan2.0-M32 模型**：一位成员分享了 [Yuan2.0-M32](https://huggingface.co/IEITYuan/Yuan2-M32-hf)，重点介绍了其包含 32 个专家的 Mixture-of-Experts 架构。提供了 GitHub、微信和[研究论文](https://arxiv.org/abs/2405.17976)的链接。
- **llama.cpp 中的 Tokenization 问题**：一位成员引用了 llama.cpp 中尚未解决的 tokenization 问题页面，分享了两个 GitHub issue [#7094](https://github.com/ggerganov/llama.cpp/issues/7094) 和 [#7271](https://github.com/ggerganov/llama.cpp/issues/7271)。他们建议用户在 llama.cpp 上使用微调模型（finetunes）时要验证 tokenization。
- **Axolotl 在 AMD 上的运行**：简要讨论了 Axolotl 是否支持 AMD；这需要一些修改。分享了一个关于[实验性 ROCm 安装指南](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1550)的 GitHub PR。
- **澄清 Axolotl 的用途**：一位成员误问 Axolotl 是否正在发行加密货币代币。另一位成员澄清它是用于训练大语言模型（LLM）的，与加密货币无关。
- **NeurIPS 参会**：关于参加 NeurIPS 的讨论，一位成员提到他们的论文录用决定仍在等待中。他们表示即使论文未被录用也有兴趣参加。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/IEITYuan/Yuan2-M32-hf">IEITYuan/Yuan2-M32-hf · Hugging Face</a>：暂无描述</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/pull/1550">Add experimental install guide for ROCm by xzuyn · Pull Request #1550 · OpenAccess-AI-Collective/axolotl</a>：描述：这为 ROCm 用户添加了如何安装 Axolotl 的指南。目前你需要安装 pip install -e &#39;.[deepspeed]&#39; 中包含的包，然后卸载 torch, xformers 和...</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7094>">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7271>">Issues · ggerganov/llama.cpp</a>：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1246259954550050879)** (8 条消息🔥): 

- **分享数据集和分类任务**：一位用户分享了一个用于分析实体并将其分类为个人、公司或工会的数据集。该数据集可在 [Hugging Face](https://huggingface.co/datasets/Dmg02/names_textcat) 上获取。
- **分享推理设置的配置**：同一位用户提供了一个用于 Llama LoRA 模型推理的配置文件。关键设置包括使用 **meta-llama/Meta-Llama-3-8B** 作为基础模型以及各种 LoRA 特定的配置。
- **指出模板使用错误**：另一位用户建议，问题可能是由于没有正确使用 Alpaca 聊天模板（chat template）导致的。
- **指定训练设备**：一位用户询问如何指定训练设备，建议通过设置 `os.environ["CUDA_VISIBLE_DEVICES"] = "1"` 来指定设备。

**提到的链接**：<a href="https://huggingface.co/datasets/Dmg02/names_textcat">Dmg02/names_textcat · Datasets at Hugging Face</a>：暂无描述

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1247156185417781278)** (7 messages): 

- **为 QLoRA 训练设置 wandb**：一位成员询问如何在 QLoRA 训练期间使用 **wandb** 追踪参数和损失值。他们收到了关于安装 wandb、登录以及配置训练脚本（包括在 `qlora.yml` 文件中添加特定配置）的详细回复。
- **使用 Mistral-7B 的 QLoRA 配置**：一位用户分享了针对 **Mistral-7B-Instruct-v0.1** 的 QLoRA 训练配置，概述了详细参数、数据集路径、优化器设置以及 **wandb** 集成细节。他们询问该配置是否正确，并请求进一步验证。
- **使用现有的 wandb 项目进行追踪**：用户强调他们希望使用现有的 **wandb** 项目而不是创建新项目来追踪参数和损失值。他们询问了如何在训练工作流中正确配置此设置的说明。

**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=86e10205-e204-465b-9e90-c7c57b04ff0c)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1246180171480956978)** (33 messages🔥): 

- **研究小组寻求合作者**：一位成员询问是否可以发布关于一个正在寻找合作者的开源大型 Transformer 项目的信息。他们被建议在专门讨论此类话题的特定频道中发布。
- **分享支持联系方式**：一位寻求 Cohere 支持团队帮助的用户被引导联系 **support@cohere.com** 以获取协助。另一位成员确认，问题也可以在 Discord 服务器本身中解决。
- **发现 Chat API 文档问题**：一位成员指出 Cohere Chat API 文档导航栏中的一个仪表板链接失效。该问题得到了另一位社区成员的确认和感谢。
- **聊天记录丢失查询**：一位成员报告聊天记录消失，并被建议在指定的支持频道寻求帮助。
- **Cohere 的 Aya 受到称赞**：一位用户确认使用 Python 和 llama.cpp 成功测试了 Cohere 的模型 Aya 23。他们分享了正面反馈，并请求允许在适当的频道发布代码。

**提到的链接**：<a href="https://docs.cohere.com/docs/chat-api">Using the Chat API</a>：未找到描述

  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1246183994178142311)** (6 messages): 

- **Manifold Research 寻求合作者**：Manifold Research 的代表邀请合作者参与针对多模态和控制任务的 Transformer 研究，旨在构建一个大规模开源的“Generalist”模型。该倡议旨在规模化复现 GATO 架构，涵盖视觉、语言、VQA、控制等领域。

- **引入了新的 Discord 标签**：一位成员指出一个新标签，另一位成员对这些变化表示兴奋，并提供了[解释链接](https://discord.com/channels/954421988141711382/954421988783444043/1246005007141175386)。

- **社区感谢**：成员们对彼此在社区中的贡献和参与表示感谢。一位成员谦虚地淡化了自己的重要性，而另一位成员则强调了社区支持的价值。
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1246448758909042839)** (21 条消息🔥): 

- **Open Interpreter 中的自定义语音实现**：一位成员一直在“开发 OI 的一个分支，用 Whisper 或 Piper 替换原生语音”，旨在通过减少冗余和加快语音启动速度来增强体验。
- **非 Ubuntu 系统上的 Interpreter 运行失败**：一位用户分享了在 MX Linux 上安装 Open Interpreter 的尝试，由于缺少 Python 而失败，尽管后来在区域网络中断恢复后在 Ubuntu 上成功运行。
- **对类 Agent 决策代码的困惑**：一位用户询问代码库中何处生成“类 Agent 决策”。另一位成员澄清说，这些是由 LLM 根据 [default system message](https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/core/default_system_message.py) 中的提示词定义的。
- **营销咨询**：一位用户询问了 Open Interpreter 的营销工作，这被归功于特定个人。
- **运行 Gemini 的问题**：另一位成员询问了在 Open Interpreter 上运行 Gemini 的替代方法，并表示文档中的示例“开始出现异常”且似乎已过时。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter%2Fcore%2Fdefault_system_message.py">open-interpreter/interpreter/core/default_system_message.py (位于 main 分支) · OpenInterpreter/open-interpreter</a>：计算机的自然语言接口。通过在 GitHub 上创建账户，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInt">Open Interactive, LLC.</a>：GitHub 是 Open Interactive, LLC. 构建软件的地方。</li><li><a href="https://tenor.com/view/bow-bowing-michael-scott-steve-carell-the-office-gif-1242852755589233352">Bow Bowing GIF - Bow Bowing Michael Scott - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1246268749892550726)** (11 条消息🔥): 

- **寻求连接 01 服务器到 iPhone 的 App**：一位成员询问是否有人开发了将 01 服务器连接到 iPhone 的 App。另一位成员分享了相关代码的 [GitHub 链接](https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile) 并鼓励开发该 App。
- **分享了 01 服务器 App 的 iOS TestFlight 链接**：一位成员分享了一个 [TestFlight 链接](https://testflight.apple.com/join/v8SyuzMT)，用于测试连接 01 服务器到 iPhone 的 App。他们提到已将 App 提交至 GitHub，尽管尚未被采纳。
- **iOS App 支持 TTS 输出**：有人提问移动端 App 是否支持 Text-to-Speech (TTS) 输出。已确认 TTS 功能在 iOS 版本的 App 上可以正常工作。
- **Android 版本正在开发中**：一位成员对缺少 Android 版本表示遗憾。然而，有人澄清支持 Android 的移动版本正在开发中，可以在 GitHub 上找到。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://testflight.apple.com/join/v8SyuzMT">加入 01ForiOS Beta 测试</a>：适用于 iOS</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/software/source/clients/mobile">01/software/source/clients/mobile (位于 main 分支) · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账户，为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息): 

cyanidebyte: https://github.com/v2rockets/Loyal-Elephie

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1246200897994756096)** (16 条消息🔥): 

- **Hugging Face 遭遇未经授权访问问题**：最近发生了一起在 Hugging Face 的 Spaces 平台上检测到未经授权访问的事件，可能导致部分 Spaces 的 secrets 泄露。“我们建议您刷新任何密钥或 token，并考虑将您的 HF tokens 切换为细粒度访问令牌（fine-grained access tokens），这是目前的新默认设置。”更多详情请参阅[此处](https://huggingface.co/blog/space-secrets-disclosure)。

- **AI2 安全令牌刷新**：尽管未受影响，AI2 正在对其 token 进行大规模刷新。natolambert 表示他的 token 已自动更新，并提到这次事件在 AI2 引发了更多关于安全性的讨论。

- **Phi-3 模型性能**：Phi-3 Medium (14B) 和 Small (7B) 模型已添加到 @lmsysorg 排行榜。Medium 的排名接近 GPT-3.5-Turbo-0613，而 Small 接近 Llama-2-70B，并强调“我们不能纯粹针对学术基准（benchmarks）进行优化”。

- **捐赠取代赌注**：dylan 输掉了一个关于模型性能的赌注，这些赌注已转化为捐赠。natolambert 对参与这些赌注带来的声誉提升表示感兴趣，并称：“这是一个很好的事业”。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/space-secrets-disclosure">Space secrets 安全更新</a>：未找到描述</li><li><a href="https://fxtwitter.com/_philschmid/status/1797700161226838362">Philipp Schmid (@_philschmid) 的推文</a>：Phi-3 Medium (14B) 和 Small (7B) 模型登上了 @lmsysorg 排行榜！😍 Medium 排名接近 GPT-3.5-Turbo-0613，但落后于 Llama 3 8B。Phi-3 Small 接近 Llama-2-70B 和 Mistral 微调版。...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1247228313894649877)** (9 条消息🔥): 

- **Llama 3V 模型抄袭指控**：讨论透露 **Llama 3V** 据称是一个抄袭模型。它可以使用 **MiniCPM-Llama3-V 2.5** 的代码和配置运行，仅更改了参数名称。
- **Chris Manning 的严厉批评**：据报道，Chris Manning 对“不承认错误”的批评让相关人员的职业生涯受挫。分享了 [Chris Manning 的一条推文](https://x.com/chrmanning/status/1797664513367630101) 以强调这些观点。
- **调查结论为抄袭**：来自 [Ao Zhang 的调查](https://github.com/OpenBMB/MiniCPM-V/issues/196) 的详细信息显示，Llama3-V 的行为与具有未公开实验性功能的 MiniCPM-Llama3-V 2.5 非常相似。
- **Giffmana 对 VLM 社区的反思**：Giffmana 指出，由于这次事件，**VLM 社区**内的信任可能已经破裂。他们推测，所谓的创新模型 Llama3-V 是从 **MiniCPM** 窃取的，并附带了证据。
- **已删除的 Medium 文章**：Aksh Garg 在 Medium 上关于构建 Llama-3V 的文章链接现在显示为 404 [“未找到页面”](https://aksh-garg.medium.com/llama-3v-building-an-open-source-gpt-4v-competitor-in-under-500-7dd8f1f6c9ee)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/chrmanning/status/1797664513367630101">Christopher Manning (@chrmanning) 的推文</a>：如何不承认自己的错误！https://aksh-garg.medium.com/llama-3v-building-an-open-source-gpt-4v-competitor-in-under-500-7dd8f1f6c9ee @TsinghuaNLP 周围有一些优秀的开源工作，帮助推动...</li><li><a href="https://aksh-garg.medium.com/llama-3v-building-an-open-source-gpt-4v-competitor-in-under-500-7dd8f1f6c9ee>">未找到标题</a>：未找到描述</li><li><a href="https://x.com/giffmana/status/1797603355919028547">Lucas Beyer (bl16) (@giffmana) 的推文</a>：这可能是 VLM 社区盲目信任破灭的一个周末？还记得大张旗鼓发布的 Llama3-V（非 META 发布）吗，号称训练成本低于 500 美元即可比肩 Gemini、GPT4、Claude ...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1247204177076158485)** (7 条消息): 

- **悬念式付费墙大师**：成员们开玩笑说需要改进他们的付费墙策略。有人强调 *Dylan 是悬念式付费墙（cliffhanger paywall）的大师*，并提到他曾在一篇 GPT-4 泄露文章中*只对其中一个段落设置付费墙*。
  
- **Karpathy 的 Twitter 动态受到关注**：关于 **Andrej Karpathy 的 Twitter 动态**有一个幽默的观察，注意到他在短时间内获得了三个点赞，评论称：“Andrej 今天上午在 Twitter 上真的很活跃”。这引发了成员们的笑声。

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1247247612273496198)** (1 条消息): 

- **Nathan Lambert 分享了一个 meme**：一位用户幽默地承认他们“可能偷了这个 meme”，但仍觉得值得分享。他们附带了一个 [Elon Musk 的推文链接](https://x.com/elonmusk/status/1797514397881192610)。

**提及的链接**：<a href="https://x.com/elonmusk/status/1797514397881192610">来自 Elon Musk (@elonmusk) 的推文</a>：未找到描述

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1247228938346958859)** (1 条消息): 

- **Mozilla Builders Accelerator 期待你的加入**：**Mozilla Builders Accelerator** 现已开始接受申请，重点关注 **Local AI**，即在个人设备而非云端运行 AI 模型和应用程序。权益包括高达 **$100,000 的资金支持**、专家的 **mentorship**（导师指导）、社区支持，以及通过 **Mozilla 渠道**展示项目的机会。[了解更多并申请](https://future.mozilla.org/builders/blog/announcing-mozilla-builders/)。
  

---


### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1246673231176405174)** (17 条消息🔥): 

- **适用于 llama.cpp 的有状态负载均衡器可能很有用**：一位成员分享了 [paddler 的 GitHub 链接](https://github.com/distantmagic/paddler)，这是一个为 llama.cpp 量身定制的有状态负载均衡器，并询问其对 llamafile 的适用性。
- **JSON Schema 减慢了采样速度**：另一位成员表示，即使有缓存，*采样速度依然很慢*，并怀疑是核心服务器的原因；他们还确认了 *JSON schema 存在问题*，正如该 [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/7703) 中所强调的。
- **兼容 OpenAI 的 Chat Completion 端点可以工作**：详细讨论指出，兼容 OpenAI 的聊天端点 `/v1/chat/completions` 适用于本地模型，但特定模型的角色（Role）可能会有问题，这些角色通常由 OpenAI 的后处理进行管理。
- **模型兼容性的预处理**：讨论了预处理对于确保不同模型之间兼容性的重要性，特别提到需要为 Mistral-7b-instruct 等特定模型适配聊天消息。
- **跨模型的统一接口**：目标是在提供广泛的模型/供应商选择的同时，提供统一的接口和功能，即使这需要通过预处理来处理模型之间的异构性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/Mozilla-Ocho/llamafile/blob/main/llama.cpp/server/README.md#api-endpoints">llamafile/llama.cpp/server/README.md at main · Mozilla-Ocho/llamafile</a>：通过单个文件分发和运行 LLM。在 GitHub 上参与 Mozilla-Ocho/llamafile 的开发。</li><li><a href="https://github.com/distantmagic/paddler">GitHub - distantmagic/paddler: 为 llama.cpp 量身定制的有状态负载均衡器</a>：为 llama.cpp 量身定制的有状态负载均衡器 - distantmagic/paddler</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/7703">Bug: JSON Schema 未被遵守？ · Issue #7703 · ggerganov/llama.cpp</a>：发生了什么？给定此 JSON Schema { &quot;$schema&quot;: &quot;http://json-schema.org/draft-07/schema#&quot;, &quot;type&quot;: &quot;object&quot;, &quot;properties&quot;: { &quot;actions&quot;: {...
</li>
</ul>

</div>
  

---

### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1246178001352261763)** (6 messages): 

- **Replay buffer 方法尚未实现**：一位成员指出“它与 Spaetzle 非常接近！”并且论文“描述了一种 Replay buffer 方法，但据我所知（afaik）尚未实现”。他们计划重新审视它，并提到了一篇描述该概念的 [AI 生成的 Medium 文章](https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88)。
  
- **关于 InstructLab 的 Medium 文章引发关注**：有人对这篇 Medium 文章表示感谢，并计划“深入研究”，同时记录了各种模型的得分，包括 “phi-3-mini-instruct 或 phoenix（Spaetzle-v74 只是 v60 + merlinite 的测试合并）”。
  
- **Spaetzle 模型各不相同**：关于 Spaetzle 有一些澄清，一位成员指出“啊，我以为 Spaetzle 是一个模型，但它们其实各不相同。”

- **寻求德语手写识别模型**：一位成员征求德语手写识别模型的建议，另一位成员推荐了 “Kraken” 并分享了一个 [匿名调查链接](https://www.berd-nfdi.de/limesurvey/index.php/996387)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@syeda9118/instructlab-ever-imagined-the-ease-of-tuning-pre-trained-llms-3331ccea8d88">InstructLab – “曾经想象过微调预训练 LLMs 的便捷吗？</a>：什么是 InstructLab？</li><li><a href="https://www.berd-nfdi.de/limesurvey/index.php/996387">
        OCR 推荐器
    </a>：未找到描述
</li>
</ul>

</div>
  

---



### **Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1246650272810270822)** (5 messages): 

- **Claude 3 缺少 Tokenizer**：一位成员对 **Claude 3** 缺少 Tokenizer 表示困惑，称其“很奇怪”。
- **Nomic Embed 模型故障排除**：一位用户询问如何通过 `llm embed` CLI 命令使用 **nomic-embed-text-v1 模型**，并指出 `llm models` 显示了 **gpt4all 模型**，但没有显示这一个。 
- **切换到 Sentence Transformers 插件**：SimonW 建议使用不同的插件 [llm-sentence-transformers](https://github.com/simonw/llm-sentence-transformers)，来处理 Nomic 模型的 Embedding 任务。
- **发布说明中的示例**：SimonW 指向了 [llm-sentence-transformers](https://github.com/simonw/llm-sentence-transformers/releases/tag/0.2) 0.2 版本的发布说明，作为如何安装和使用 **nomic-embed-text-v1 模型**的示例。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/simonw/llm-sentence-transformers">GitHub - simonw/llm-sentence-transformers: 使用 sentence-transformers 进行 Embedding 的 LLM 插件</a>：使用 sentence-transformers 进行 Embedding 的 LLM 插件 - simonw/llm-sentence-transformers</li><li><a href="https://github.com/simonw/llm-sentence-transformers/releases/tag/0.2">Release 0.2 · simonw/llm-sentence-transformers</a>：新增 llm sentence-transformers register --trust-remote-code 选项，用于安装需要 trust_remote_code=True 的模型。#14 以下是如何使用此选项安装 nomic-embed-text-v1 的方法（其中...
</li>
</ul>

</div>
  

---



### **AI21 Labs (Jamba) ▷ #[jamba](https://discord.com/channels/874538902696914944/1222916247063232553/1246707102660034631)** (5 messages): 

- **Jamba Instruct 与 GPT-4 的对比**：一位成员询问 Jamba Instruct 与 GPT-4 的对比情况，另一位成员表示 Jamba Instruct 在性能上可与 **Mixtral 8x7B** 媲美。
  
- **ML/DL 模型在函数组合方面表现不佳**：另一位成员分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/feed/update/urn:li:activity:7203325801356746752/)，讨论了当前的 ML/DL 模型（如 SSMs, Transformers, CNNs 和 RNNs）无法解决函数组合（Function Composition）问题，暗示了它们在推理能力上的局限性。帖子提到 **Jamba 也被用于 SSM 实验**。
  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1246198295525589074)** (2 条消息): 

- **AI4Health Medical Innovation 机会不容错过**：分享了一个加入 **Alliance AI4Health Medical Innovation Challenge Hackathon/Ideathon** 的机会，奖金总额超过 *$5k*。本次活动专注于构建创新的 AI 解决方案，以应对普遍的医疗挑战，激励下一代医疗创新者。[点击此处注册](https://amic.devpost.com/)。

**提到的链接**：<a href="https://amic.devpost.com/">Alliance Medical Innovation Challenge</a>：通过基于 AI 的解决方案赋能全球健康：解决未来的问题

  

---



---



---



{% else %}


> 完整的各频道详细内容已针对邮件进行截断。 
> 
> 如果您想查看完整内容，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}