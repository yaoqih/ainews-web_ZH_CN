---
companies:
- stripe
- perplexity-ai
- mistral-ai
- hugging-face
- cerebras
- anthropic
- weights-biases
- google
- vllm-project
date: '2024-11-20T00:43:00.876368Z'
description: '以下是该文本的中文翻译：


  **Stripe** 推出了其 **Agent SDK**，为美国 Pro 会员实现了如 **Perplexity Shopping** 般的 AI 原生购物体验，其特点是通过
  **Perplexity 商家计划**提供一键结账和免运费服务。**Mistral AI** 发布了 **Pixtral Large 124B** 多模态图像模型，该模型目前已在
  **Hugging Face** 上线，并由 **Le Chat** 支持图像生成。**Cerebras Systems** 为 **Llama 3.1 405B**
  提供了公共推理端点，具备 128k 上下文窗口和高吞吐量。**Claude 3.6** 展现出优于 **Claude 3.5** 的改进，但仍存在细微的“幻觉”现象。**Bi-Mamba**
  1-bit 架构提升了大语言模型（LLM）的效率。**wandb SDK** 已预装在 **Google Colab** 中，同时 **Pixtral Large**
  已集成到 **AnyChat**，并得到 **vLLM** 的支持以实现高效的模型使用。'
id: 6e45bfbf-a041-4008-bf35-a867965a8c93
models:
- pixtral-large-124b
- llama-3.1-405b
- claude-3.6
- claude-3.5
original_slug: ainews-perplexity-starts-shopping-for-you
people:
- patrick-collison
- jeff-weinstein
- mervenoyann
- sophiamyang
- tim-dettmers
- omarsar0
- akhaliq
- aravsrinivas
title: Perplexity 开始为你购物。
topics:
- multi-modal
- image-generation
- inference
- context-windows
- model-performance
- model-efficiency
- sdk
- ai-integration
- one-click-checkout
- memory-optimization
---

<!-- buttondown-editor-mode: plaintext -->**Stripe SDK 就够了吗？**

> 2024/11/18-2024/11/19 AI 新闻快报。我们为您查看了 7 个 Reddit 分区、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord 社区（**217** 个频道，**1912** 条消息）。为您节省了约 **253 分钟** 的阅读时间（以 200wpm 计算）。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 来参与 AINews 讨论！

就在 Stripe 发布其 Agent SDK 仅仅两天后（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-stripe-lets-agents-spend-money-with/)），Perplexity 现已面向美国 Pro 会员[推出其应用内购物体验](https://www.perplexity.ai/hub/blog/shop-like-a-pro)。这是首个大规模的 AI 原生购物体验，比起 Amazon，它更接近（做得很好的）Google Shopping。示例展示了你可以使用自然语言进行查询，而这在传统的电子商务 UI 中是很难实现的：


![image.png](https://assets.buttondown.email/images/4d1c2002-36b5-4515-8d9c-4b0711888664.png?w=960&fit=max)



![image.png](https://assets.buttondown.email/images/de725fe5-8927-4000-b94d-67fecb011da7.png?w=960&fit=max)


全新的 "Buy With Pro" 计划提供与 "精选商家"（！稍后详细说明）的一键结账和免费送货服务。

[Snap to Shop](https://x.com/GregFeingold/status/1858559783340560391) 也是一个很棒的视觉电商创意... 但它在非 Perplexity 员工手中的实际准确性仍有待观察。


![image.png](https://assets.buttondown.email/images/c956746f-00ad-4888-b918-fe645f2c1fdb.png?w=960&fit=max)


Buy With Pro 计划几乎肯定与新的 **Perplexity Merchant Program** 挂钩，这是一种标准的**免费**“以数据换推荐”的价值交换。

[Patrick Collison](https://x.com/patrickc/status/1858910030030139618) 和 [Jeff Weinstein](https://x.com/jeff_weinstein/status/1858916112089706821) 都迅速指出了 Stripe 的参与，尽管两人都没有直接说明 Perplexity Shopping 使用的就是 Stripe 刚刚发布的那个 Agent SDK。

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

**AI 模型发布与性能**

- **Mistral 的多模态图像模型**：[@mervenoyann](https://twitter.com/mervenoyann/status/1858561096015053250) 宣布发布 **拥有 124B 参数的 Pixtral Large**，现在 [@huggingface](https://twitter.com/mervenoyann/status/1858560560322732137) 已支持该模型。此外，[@sophiamyang](https://twitter.com/sophiamyang/status/1858574087146160427) 分享了 **@MistralAI** 现在支持在 **Le Chat 上生成图像**，由 **@bfl_ml** 提供支持，并免费开放。
  
- **Cerebras Systems 的 Llama 3.1 405B**：[@ArtificialAnlys](https://twitter.com/ArtificialAnlys/status/1858594969927512476) 详细介绍了 **Cerebras 的 Llama 3.1 405B 公共推理端点**，拥有 **969 output tokens/s** 和 **128k 上下文窗口**。这一性能比中位数供应商快 **10 倍以上**。定价设定为每 1M input tokens **$6**，每 1M output tokens **$12**。

- **Claude 3.5 和 3.6 的增强**：[@Yuchenj_UW](https://twitter.com/Yuchenj_UW/status/1858572694930993339) 讨论了 **Claude 3.5** 如何被 **Claude 3.6** 超越，后者虽然更具说服力，但表现出**更微妙的幻觉**。像 [@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1858885296621850836) 这样的用户已经开始**调试输出**以维持对模型的信任。

- **Bi-Mamba 架构**：[@omarsar0](https://twitter.com/omarsar0/status/1858878654736199850) 介绍了 **Bi-Mamba**，这是一种专为更高效的 **LLM** 设计的 **1-bit Mamba 架构**，在显著减少内存占用的同时，实现了与 **FP16 或 BF16** 模型相当的性能。

**AI 工具、SDK 与平台**

- **Google Colab 上的 Wandb SDK**：[@weights_biases](https://twitter.com/weights_biases/status/1858582707179016384) 宣布 **wandb Python SDK** 现在已**预装在每个 Google Colab 笔记本中**，允许用户跳过 `!pip install` 步骤直接导入。

- **AnyChat 集成**：[@_akhaliq](https://twitter.com/_akhaliq/status/1858650300493504966) 强调 **Pixtral Large** 现在已在 **AnyChat** 上可用，通过集成 **ChatGPT** 和 **Google Gemini** 等多个模型来增强 **AI 灵活性**。

- **vLLM 支持**：[@vllm_project](https://twitter.com/vllm_project/status/1858568598123671676) 通过简单的 `pip install -U vLLM` 引入了对 **Pixtral Large** 的支持，使用户能够高效地运行该模型。

- **Perplexity Shopping 功能**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1858560970223911122) 详细介绍了 **Perplexity Shopping** 的发布，该功能与 **@Shopify** 集成，提供 **AI 驱动的产品推荐**和**多模态购物体验**。

**AI 研究与基准测试**

- **nGPT 论文与基准测试**：[@jxmnop](https://twitter.com/jxmnop/status/1858627599981048211) 分享了关于 **nGPT** 论文的见解，强调了其声称比 **GPT** 训练速度快 **4-20 倍**。然而，由于**基准测试存在问题**，社区在**复现结果**方面面临挑战。

- **VisRAG 框架**：[@JinaAI_](https://twitter.com/JinaAI_/status/1858693566857703906) 介绍了 **VisRAG**，这是一个通过**多模态推理**解决 **RAG 瓶颈**来增强**检索工作流**的框架，其性能优于 **TextRAG**。

- **用于 LLM 评估的 Judge Arena**：[@clefourrier](https://twitter.com/clefourrier/status/1858862476281909537) 展示了 **Judge Arena**，这是一个用于**比较模型裁判 (model-judges)** 的工具，旨在对**复杂生成内容**进行细致评估，帮助研究人员选择合适的 **LLM 评估器**。

- **Bi-Mamba 的效率**：[@omarsar0](https://twitter.com/omarsar0/status/1858878654736199850) 讨论了 **Bi-Mamba** 如何实现**与全精度模型相当的性能**，标志着 **LLM** **低比特表示 (low-bit representation)** 的一个重要趋势。

**AI 公司合作伙伴关系与公告**

- **Google Colab 与 Wandb 合作伙伴关系**：[@weights_biases](https://twitter.com/weights_biases/status/1858582707179016384) 宣布与 **@GoogleColab** 合作，确保 **wandb SDK** 可供用户直接使用，从而简化工作流集成。

- **凯悦 (Hyatt) 与 Snowflake 的合作伙伴关系**：[@RamaswmySridhar](https://twitter.com/RamaswmySridhar/status/1858562698315002343) 分享了 **@Hyatt** 如何利用 **@SnowflakeDB** 来**统一数据**、**减少管理时间**并**快速创新**，从而提高**运营效率**和**客户洞察**。

- **Figure Robotics 的招聘与部署**：[@adcock_brett](https://twitter.com/adcock_brett/status/1858713940525806019) 多次讨论了 **Figure** 致力于**交付数百万台人形机器人**、**招聘顶尖工程师**以及**部署自主车队**的承诺，展示了 **AI 机器人**领域的重大**规模化努力**。

- **Hugging Face 增强功能**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1858556414488096997) 强调 **@huggingface** 现在提供**帖子互动情况**的可视化，增强了该平台作为 **AI 新闻和更新**中心的角色。

**AI 活动与工作坊**

- **AIMakerspace Agentic RAG 工作坊**：[@llama_index](https://twitter.com/llama_index/status/1858659057185550535) 宣传了将于 **11 月 27 日**由 **@AIMakerspace** 主办的**直播活动**，重点关注使用**开源 LLM** 构建**本地 Agentic RAG 应用**，并由 **Dr. Greg Loughnane** 和 **Chris "The Wiz" Alexiuk** 主持动手实践环节。

- **SambaNova 与 Hugging Face 开源 AI 之夜**：[@_akhaliq](https://twitter.com/_akhaliq/status/1858710777865093192) 宣布了一场定于 **12 月 10 日**举行的**开源 AI 活动**，届时将汇聚**硅谷的 AI 思想家**，促进 **@Sambanova** 与 **@HuggingFace** 之间的合作。

- **新加坡 DevDay**：[@stevenheidel](https://twitter.com/stevenheidel/status/1858634765567668626) 分享了参加 **2024 年最后一场 DevDay**（位于**新加坡**）的兴奋之情，并强调了与**其他受邀演讲者交流**的机会。

**梗/幽默**

- **对 AI 的误解与挫败感**：[@transfornix](https://twitter.com/transfornix/status/1858895038140293567) 表达了对**缺乏动力**和**大脑迷雾**的沮丧。同样，[@fabianstelzer](https://twitter.com/fabianstelzer/status/1858855105858036031) 分享了对 **AI 工作流**和意外结果的轻松调侃式挫败感。

- **关于 AI 和技术的幽默见解**：[@jxmnop](https://twitter.com/jxmnop/status/1858895357209403510) 幽默地质疑为什么 **Transformer 实现错误**会导致一切崩溃，反映了开发者的常见挫败感。此外，[@idrdrdv](https://twitter.com/vikhyatk/status/1858791746605318309) 开玩笑说**范畴论 (category theory)** 让新人望而却步。

- **轻松有趣的互动**：诸如 [@swyx](https://twitter.com/swyx/status/1858690687530922482) 分享关于 **oauth 需求**的幽默评论，以及 [@HamelHusain](https://twitter.com/HamelHusain/status/1858664296487764400) 参与轻松的对话，展示了社区风趣的一面。

- **对 AI 发展的反应**：[@aidan_mclau](https://twitter.com/aidan_mclau/status/1858792340451979646) 对在社交网络上看到他人做出了幽默的回应，[@giffmana](https://twitter.com/giffmana/status/1858819137163718699) 则分享了关于 AI 互动的笑料。

**AI 应用与用例**

- **AI 在文档处理中的应用**：[@omarsar0](https://twitter.com/omarsar0/status/1858875645943480663) 介绍了 **Documind**，这是一个用于从 PDF 中提取结构化数据的 **AI 驱动工具**，并强调了其**易用性**和 **AGPL v3.0 License**。

- **AI 在金融回测中的应用**：[@virattt](https://twitter.com/virattt/status/1858652975214014715) 描述了使用 **@LangChainAI** 进行编排来**回测 AI 金融 Agent** 的方案，并概述了评估投资组合收益的**四个步骤**。

- **AI 在购物与电子商务中的应用**：[@AravSrinivas](https://twitter.com/AravSrinivas/status/1858560970223911122) 展示了 **Perplexity Shopping**，详细介绍了**多模态搜索**、**Buy with Pro** 以及**与 @Shopify 的集成**等功能，旨在**简化购物体验**。

- **AI 在医疗沟通中的应用**：[@krandiash](https://twitter.com/krandiash/status/1858583470551036202) 强调了**与 @anothercohen 的合作**，旨在利用 AI **改善医疗沟通**，并重点介绍了修复破碎系统的努力。

**AI 社区与综合讨论**

- **AI 好奇心与学习**：[@saranormous](https://twitter.com/saranormous/status/1858752050760138837) 强调，**真正的技术好奇心**是一种强大且**难以伪造的特质**，鼓励 AI 社区内的持续学习与探索。

- **AI 模型开发中的挑战**：[@huybery](https://twitter.com/huybery/status/1858733666798608454) 和 [@karpathy](https://twitter.com/karpathy/status/1858688510842335635) 讨论了**模型训练的挑战**，包括**延迟问题**、**Layer Normalization** 以及**模型监管**对于构建值得信赖的 AI 系统的重要性。

- **AI 在社会科学与伦理中的应用**：[@BorisMPower](https://twitter.com/BorisMPower/status/1858766322773192893) 思考了 **AI 在社会科学中的革命性潜力**，主张在假设检验中使用 **in silico 模拟**（计算机模拟）取代传统的真人访谈。

- **AI 在软件工程中的应用**：[@inykcarr](https://twitter.com/andrew_n_carr/status/1858632378664841735) 和 [@HellerS](https://twitter.com/heller_s/status/1858702928208613624) 参与了关于 **LLM Prompt Engineering** 的讨论，强调了通过有效利用 AI 实现 **10 倍生产力提升**的“超能力”。

---

# AI Reddit 热点回顾

## /r/LocalLlama 综述

**主题 1. Mistral Large 2411：期待与发布详情**

- **[Mistral Large 2411 和 Pixtral Large 将于 11 月 18 日发布](https://github.com/mistralai/platform-docs-public/compare/main...doc/v0.0.100)** ([Score: 336, Comments: 114](https://reddit.com/r/LocalLLaMA/comments/1gu7cm8/mistral_large_2411_and_pixtral_large_release_18th/)): **Mistral Large 2411** 和 **Pixtral Large** 定于 **11 月 18 日**发布。
  - **许可与使用方面的担忧**：针对 Mistral 模型限制性的 **MRL License** 存在大量讨论，用户对不明确的许可条款以及 Mistral 对商业用途咨询缺乏回应表示沮丧。一些人认为，虽然该许可证允许研究使用，但它使商业应用和微调模型的分享变得复杂。
  - **基准测试对比**：据报道，**Pixtral Large** 在 **MathVista (69.4)** 和 **DocVQA (93.3)** 等多项基准测试中表现优于 **GPT-4o** 和 **Claude-3.5-Sonnet**，但用户注意到缺乏与 **Qwen2-VL** 和 **Molmo-72B** 等其他领先模型的对比。此外，基于基准测试表中的潜在拼写错误或泄露，有人猜测可能存在 **Llama 3.1 505B** 模型。
  - **技术实现与支持**：用户讨论了将 **Pixtral Large** 与 **Exllama** 集成以提高 VRAM 效率和张量并行（Tensor Parallelism）的潜力，并确认 **Mistral Large 2411** 不需要对 **llama.cpp** 进行更改即可支持。此外，还提到了一种新的指令模板（Instruct Template）可能会增强模型的可控性（Steerability），这与社区建议的 Prompt 格式化方案不谋而合。

- **[mistralai/Mistral-Large-Instruct-2411 · Hugging Face](https://huggingface.co/mistralai/Mistral-Large-Instruct-2411)** ([Score: 303, Comments: 81](https://reddit.com/r/LocalLLaMA/comments/1gu7k6a/mistralaimistrallargeinstruct2411_hugging_face/)): 该帖子讨论了 **Mistral Large 2411**，这是一个在 **Hugging Face** 的 **mistralai/Mistral-Large-Instruct-2411** 仓库中提供的模型。帖子正文未提供更多细节或背景。
  - 用户讨论了 **Mistral Large 2411 的性能**，指出在各种任务中结果参差不齐。**Sabin_Stargem** 提到在 NSFW 叙事生成方面取得了成功，但在背景设定（lore）理解和掷骰子点数任务中表现失败。**ortegaalfredo** 发现整体有轻微改进，但在编程任务中更倾向于使用 **qwen-2.5-32B**。
  - 关于 **模型的许可和分发** 存在争议。**TheLocalDrummer** 和其他人对 MRL 许可证表示担忧，**mikael110** 对 Apache-2 发布的终结感到惋惜。**thereisonlythedance** 尽管对许可证有抱怨，但出于经济必要性，仍对 Mistral 提供的本地模型访问表示赞赏。
  - 技术讨论涉及 **模型部署和量化**。**noneabove1182** 分享了 **Hugging Face** 上 GGUF 量化的链接，并提到缺乏与之前版本对比的评估（evals）。**segmond** 对缺乏评估数据表示怀疑，并指出在编程测试中与 **large-2407** 相比性能略有下降。


- **[Pixtral Large Released - Vision model based on Mistral Large 2](https://mistral.ai/news/pixtral-large/)** ([Score: 123, Comments: 27](https://reddit.com/r/LocalLLaMA/comments/1gu7l8s/pixtral_large_released_vision_model_based_on/)): **Pixtral Large** 已作为基于 **Mistral Large 2** 的 **视觉模型（vision model）** 发布。帖子中未提供有关模型规格或能力的进一步细节。
  - **Pixtral Large 的视觉设置**：该模型并非基于 **Qwen2-VL**；相反，它使用了 **Qwen2-72B 文本 LLM** 和自定义视觉系统。**7B 变体** 使用 **Olmo 模型** 作为基础，其表现与 Qwen 基础模型相似，表明了其数据集的稳健性。
  - **技术要求与能力**：运行该模型可能需要大量的硬件资源，例如 **4x3090 GPU** 或配备 **128GB RAM 的 MacBook Pro**。该模型的视觉编码器更大（1B 对比 400M），表明它可以处理至少 **30 张高分辨率图像**，尽管“高分辨率（hi-res）”的具体定义尚不明确。
  - **性能基准与对比**：Pixtral Large 的性能仅与 **Llama-3.2 90B** 进行了对比，后者因其规模而被认为表现欠佳。与 **Molmo-72B** 和 **Qwen2-VL** 在 **Mathvista**、**MMMU** 和 **DocVQA** 等数据集上的对比显示出不同的结果，表明其与当前最先进水平（state-of-the-art）的对比仍不全面。


**主题 2. Llama 3.1 405B 推理：Cerebras 的突破**

- **[Llama 3.1 405B now runs at 969 tokens/s on Cerebras Inference - Cerebras](https://cerebras.ai/blog/llama-405b-inference)** ([Score: 272, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1gun7zd/llama_31_405b_now_runs_at_969_tokenss_on_cerebras/)): Cerebras 在其推理平台上以 **969 tokens/s** 的速度运行 **Llama 3.1 405B**，实现了一个性能里程碑。这展示了 Cerebras 在高效处理大规模模型方面的能力。
  - 用户注意到 **405B 模型** 目前仅在付费层级向企业开放，而 **Openrouter** 以显著降低的价格提供该模型，尽管速度较慢。**128K 上下文长度** 和全 16 位精度被强调为 Cerebras 平台的关键特性。
  - 讨论强调 **Cerebras 的性能提升** 更多归功于软件改进而非硬件变更，一些用户指出了 **WSE-3 集群** 的使用以及潜在的替代方案，如 **8x AMD Instinct MI300X 加速器**。
  - 人们对高速推理的应用场景表现出兴趣，例如 **Agent 工作流** 和 **高频交易**，在这些场景中，对大型模型的快速处理可以提供优于传统慢速方法的显著优势。


**主题 3. Raspberry Pi 上的 AMD GPU：Llama.cpp 集成**

- **在 Raspberry Pi 5 上通过 Vulkan 为 llama.cpp 提供 AMD GPU 支持** ([Score: 144, Comments: 49](https://reddit.com/r/LocalLLaMA/comments/1gucux2/amd_gpu_support_for_llamacpp_via_vulkan_on/))：作者一直在 **Raspberry Pi 5** 上集成 **AMD 显卡**，并已成功在 Pi OS 上实现了 Linux `amdgpu` 驱动。他们为多款 AMD GPU 编译了支持 **Vulkan** 的 `llama.cpp`，并正在收集基准测试结果，详情可见[此处](https://github.com/geerlingguy/ollama-benchmark/issues/1)。他们正在寻求社区对额外测试的建议，并计划评估低端 AMD 显卡的性价比和能效比。
  - 几位用户建议使用 **ROCm** 代替 **Vulkan** 以获得更好的 AMD GPU 性能，但指出由于兼容性有限，在 ARM 平台上支持 **ROCm** 具有挑战性。虽然有人建议使用 **hipblas** 等替代方案，但其设置过程非常复杂，参考 [Phoronix 文章](https://www.phoronix.com/news/ARM64-AMDKFD-HSA-Compute)。
  - 讨论中涉及了针对 ARM CPU 的**量化优化**，特别是使用 `llama.cpp` 的 `4_0_X_X` 量化级别，以利用 `neon+dotprod` 等 ARM 特定指令。这可以提高搭载 **BCM2712** 的 Raspberry Pi 5 等设备的性能，使用如 `-march=armv8.2-a+fp16+dotprod` 的编译标志。
  - 在 **RX 6700 XT** 上使用 **Vulkan** 版 `llama.cpp` 的基准测试结果显示了令人期待的性能指标，但也凸显了功耗问题，测试期间平均约为 **195W**。讨论还涉及了使用 GPU 执行 AI 任务与 CPU 相比的效率，Raspberry Pi 设置在待机时仅消耗 **11.4W**。

**主题 4. txtai 8.0：精简版 Agent 框架发布**

- **[txtai 8.0 发布：极简主义者的 Agent 框架](https://medium.com/neuml/whats-new-in-txtai-8-0-2d7d0ab4506b)** ([Score: 60, Comments: 9](https://reddit.com/r/LocalLLaMA/comments/1guovi1/txtai_80_released_an_agent_framework_for/))：**txtai 8.0** 已作为专为极简主义者设计的 **Agent 框架**发布。此版本专注于简化 AI 应用的开发和部署。
  - **txtai 8.0** 引入了一个新的 Agent 框架，该框架与 **Transformers Agents** 集成并支持所有 **LLMs**，提供了一种精简的方法来部署现实世界的 Agent，而无需不必要的复杂性。更多详情和资源可在 [GitHub](https://github.com/neuml/txtai)、[PyPI](https://pypi.org/project/txtai/) 和 [Docker Hub](https://hub.docker.com/u/neuml) 上找到。
  - txtai 8.0 中的 **Agent 框架**通过工具使用和规划展示了决策能力，如 [Colab](https://colab.research.google.com/github/neuml/txtai/blob/master/examples/67_Whats_new_in_txtai_8_0.ipynb) 上的详细示例所示。该示例展示了 Agent 如何使用 'web_search' 和 'wikipedia' 等工具来回答复杂问题。
  - 用户询问了该框架的能力，包括是否支持 **Agent 的函数调用 (function calling)** 和 **视觉模型 (vision models)**。这些问题凸显了用户对于将 txtai 的功能扩展到包含视觉分析等更高级特性的兴趣。

## 其他 AI Subreddit 摘要

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Flux vs SD3.5：尽管存在技术权衡，社区仍更青睐 Flux**

- **Flux 与 SD3.5 的现状如何？** ([Score: 40, Comments: 99](https://reddit.com/r/StableDiffusion/comments/1gv18ac/what_is_the_current_state_of_flux_vs_sd35/))：社区正在对 **Stable Diffusion 3.5** 和 **Flux** 进行比较，据报道，自一个月前发布以来，对 **SD3.5** 的最初热情有所下降。该帖子寻求澄清可能导致用户回归 **Flux** 的 **SD3.5** 潜在技术问题，尽管提问中未提供具体的技术对比。
  - **SD3.5** 面临重大的采用挑战，原因是它在 **Forge** 上不可用，且与 **Flux** 相比 **微调 (finetune)** 能力有限。用户报告称 **SD3.5** 擅长艺术风格和更高分辨率，但在解剖结构（尤其是手部）方面表现不佳。
  - 社区测试显示 **Flux** 在 **img2img** 任务和写实人物生成方面更胜一筹，而 **SD3.5** 提供了更好的负面提示词 (negative prompt) 支持和更快的处理速度。**SD3.5** 缺乏高质量的微调模型和 **LoRAs**，限制了其广泛采用。
  - 高级用户建议结合两者的优势，例如使用 **SD3.5** 进行初步的创意生成，然后使用 **Flux** 进行解剖结构优化。**Flux** 于 **8月** 发布，并凭借大量可用的微调模型维持了更强大的社区支持。

- **通过提示词减少 Flux “同质化面孔”的方法（摘要与长文）** ([Score: 33, Comments: 5](https://reddit.com/r/StableDiffusion/comments/1gumazr/ways_to_minimize_flux_same_face_with_prompting/)): 该帖子为减少 **Flux 图像生成**中的“同质化面孔 (same face)”问题提供了**技术建议**，推荐的关键策略包括将 **CFG/Guidance** 降低至 **1.6-2.6**，避免使用 "man"/"woman" 等通用词汇，并加入关于族裔、体型和年龄的具体描述。作者分享了[一组示例图像](https://www.dropbox.com/scl/fi/cfhyip06fak52nr1o6mxr/AntiSameFaceExamples.zip?rlkey=f7tuvuybh1bmyityt2w13hxqf&st=3aepm0bc&dl=0)来演示这些技术，并提供了一个描述“*轮廓分明的黎巴嫩女性*”在厨房场景中的示例提示词，同时解释了模型训练偏差和常见的提示词模式是如何导致同质化面孔问题的。
  - **Flux** 中**较低的 Guidance 设置** (1.6-2.6) 以牺牲提示词遵循度为代价换取照片写实感和多样性，而一些用户为了在复杂描述中获得更好的提示词遵循度，仍维持默认的 **3.5 CFG**。
  - 社区此前曾记录过针对“**sameface**”问题的类似解决方案，包括一个通过**国籍**、**姓名**和**头发特征**随机化提示词的 [auto1111 扩展](https://github.com/artyfacialintelagent/CloneCleanser)。
  - 出于安全考虑，用户建议不要下载随机的 **.zip 文件**，并建议使用 **Imgur** 等替代图像托管平台来分享生成的示例。


**主题 2. O2 机器人技术突破：宝马工厂速度提升 400%**

- **[Figure 02 现已成为在宝马工厂工作的自主机群，近几个月速度提升了 400%](https://v.redd.it/ariij7t9jw1e1)** ([Score: 180, Comments: 63](https://reddit.com/r/OpenAI/comments/1gv48sq/figure_02_is_now_an_autonomous_fleet_working_at_a/)): **Figure 02** 机器人目前作为自主机群在**宝马工厂**运行，近几个月其操作**速度提升了 400%**。
  - 根据 [宝马新闻稿](https://www.press.bmwgroup.com/deutschland/article/detail/T0444264DE/erfolgreicher-testeinsatz-humanoider-roboter-im-bmw-group-werk-spartanburg)，**Figure 02** 机器人目前在需要充电前可运行 **5 小时**，单台成本为 **$130,000**。这些机器人在实现 **400% 速度提升**的同时，**可靠性也提升了 7 倍**。
  - 多位用户指出，这些机器人的**进步速度**超过了人类的能力，具有**全天候 24/7 运行的潜力**，且无需休息或福利。批评者则指出，与专用机械臂相比，目前在效率上仍存在局限。
  - 讨论集中在**经济可行性**上，一些人主张为了自动化应重新设计整个工厂，而不是保留适应人类的空间。机器人需要工厂照明和维护成本，但不需要暖气或保险。


**主题 3. Claude vs ChatGPT：企业用户体验讨论**

- **我应该升级到 ChatGPT Plus 还是 Claude AI？帮我决定！** ([Score: 33, Comments: 69](https://reddit.com/r/ClaudeAI/comments/1gurzrr/should_i_upgrade_to_chatgpt_plus_or_claude_ai/)): **数字营销**专业人士对比了 **ChatGPT Plus** 和 **Claude AI** 在内容创作和技术辅助方面的表现，重点关注处理**内容构思**（占用途的 75%）和 **Linux 技术支持**（占用途的 25%）。近期出现了对 **Claude 可靠性**和**模型降级**的担忧，用户报告了**停机**和模型变更中的**透明度问题**，引发了关于 Claude 作为付费服务可行性的质疑。
  - **OpenRouter** 和 **TypingMind** 等第三方应用成为直接订阅的热门替代方案，提供了在不同模型间切换的灵活性，且成本可能低于每月 20 美元的方案。用户强调了在同一处维持上下文并集成多个 API 的能力。
  - **Claude 最近的变化**引发了关于**审查制度**加强和**使用限制**（免费档的 5 倍限制）的批评，尤其影响了西班牙语用户和技术任务。用户报告 Claude 以“*伦理原因*”拒绝任务，并经历了显著的模型行为变化。
  - **o1-preview** 模型因其集成的**思维链 (chain of thought)** 能力和处理复杂数学的能力而受到高度赞赏，而 **Google Gemini 1.5 Pro** 则因其 **1,000,000 Token 上下文窗口**以及与 [Google Workspace](https://gemini.google/advanced/) 的集成而备受关注。

- **Claude 的服务器快崩溃了！** ([Score: 86, Comments: 24](https://reddit.com/r/ClaudeAI/comments/1gv1nik/claudes_servers_are_dying/)): **Claude** 用户报告了持续的**服务器容量问题**，频繁的高需求通知导致工作流中断。用户对服务中断表示沮丧，并要求 **Anthropic** 团队升级基础设施。
  - 用户报告 **Claude** 的最佳使用时间是当**印度**和**加利福尼亚**都不活跃时，多位用户确认他们根据这些时区规划工作以避开过载问题。
  - 几位用户建议放弃 **Claude 网页界面**，转而使用**基于 API 的解决方案**，其中一位用户详细介绍了他们从使用网页界面到通过自定义实现和 **Open WebUI** 管理 **100 多个 AI 模型**的历程。
  - 用户对**简短的回答**和 *"Error sending messages. Overloaded"* 通知表示不满，尽管成本更高，一些人仍推荐 **OpenRouter API** 作为替代方案。


**主题 4. CogVideo 封装器更新：重大重构与 1.5 支持**

- **Kijai 更新了 CogVideoXWrapper：支持 1.5！重构了简化管道并进行了额外优化。（但会破坏旧的工作流）** ([Score: 69, Comments: 23](https://reddit.com/r/StableDiffusion/comments/1gv571o/kijai_has_updated_the_cogvideoxwrapper_support/)): **CogVideoXWrapper** 迎来了重大更新，支持 **CogVideoX 1.5 模型**，其特点包括代码清理、将 **Fun-model** 功能合并到主管道中，并添加了 **torch.compile** 优化以及 **torchao 量化**。此次更新对旧工作流引入了破坏性变更，包括从采样器组件中移除宽度/高度、将 **VAE** 从模型中分离、支持 **fp32 VAE** 以及用 **FasterCache** 替换 **PAB**，同时在 [ComfyUI-CogVideoXWrapper](https://github.com/kijai/ComfyUI-CogVideoXWrapper) 保留了旧版本的遗留分支。
  - 在 **RTX 4090** 上的测试显示，**CogVideoX 1.5** 以 **720x480** 分辨率处理 **49 帧**、**20 步**大约需要 **30-40 秒**，在相同帧数下，该模型明显比以前的版本更快。
  - **2B 模型**需要大约 **3GB VRAM** 用于存储，外加额外的推理内存，在 **512x512** 分辨率下的测试显示，包括 **VAE 解码**在内的峰值 VRAM 占用约为 **6GB**。
  - **Alibaba** 发布了更新版本的 [CogVideoXFun](https://github.com/aigc-apps/CogVideoX-Fun#cogvideox-fun-v11-5b-control)，截至 **2024.11.16**，新增了对 **Canny**、**Depth**、**Pose** 和 **MLSD** 条件的**控制模型**支持。


---

# AI Discord 摘要

> 由 O1-preview 提供的摘要之摘要的总结

**主题 1：尖端 AI 模型竞相宣示主导地位**

- [**Cerebras 凭借 Llama 3.1 创下速度纪录**](https://x.com/artificialanlys/status/1858594969927512476?s=46): Cerebras 声称其 Llama 3.1 405B 模型达到了惊人的 **969 tokens/s**，比平均供应商快 **10 倍**以上。批评者认为这是“苹果与橘子”的不对等比较，指出 Cerebras 在 batch size 为 1 时表现出色，但在处理大 batch 时落后。
- [**Runner H 冲向 ASI，表现超越竞争对手**](https://x.com/hcompany_ai/status/1858907033921069449): **H Company** 宣布了 Runner H 的 Beta 版本，声称突破了 scaling laws 的限制，向人工超智能（ASI）迈进。据报道，Runner H 在 **WebVoyager 基准测试**上[超越了 Qwen](https://www.hcompany.ai/blog/a-research-update)，展示了卓越的导航和推理能力。
- [**Mistral 发布具有 128K 上下文窗口的 Pixtral Large**](https://mistral.ai/news/pixtral-large/): Mistral 推出了 Pixtral Large，这是一个基于 Mistral Large 2 的 **124B** 多模态模型，通过 **128K 上下文窗口**可处理超过 **30 张高分辨率图像**。它在 **MathVista** 和 **VQAv2** 等基准测试中达到了业界领先的性能。

---

**主题 2：AI 模型努力应对局限性与 Bug**

- [**Qwen 2.5 模型在训练中表现不稳定**](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct)：用户报告在训练 Qwen 2.5 时结果不一致，而切换到 Llama 3.1 时错误则会消失。该模型似乎对特定配置非常敏感，这给开发者带来了困扰。
- [**AI 在井字棋中失误并遗忘规则**](https://discord.com/channels/974519864045756446)：成员们观察到像 GPT-4 这样的 AI 模型在处理井字棋等简单游戏时表现挣扎，无法封堵对手招式并在游戏途中丢失进度。LLM 作为状态机的局限性引发了关于需要更好游戏逻辑框架的讨论。
- [**Mistral 模型陷入死循环**](https://console.mistral.ai/)：用户在使用 Mistral Nemo 等模型通过 OpenRouter 输出时遇到了死循环和重复输出的问题。调整温度（temperature）设置未能完全解决该问题，表明模型输出存在更深层的问题。

---

**主题 3：创新研究照亮 AI 前景**

- [**Neural Metamorphosis 实现网络即时变形**](https://arxiv.org/abs/2410.11878)：Neural Metamorphosis 论文通过学习连续权重流形引入了自变形神经网络，允许模型在不进行重新训练的情况下调整大小和配置。
- [**LLM2CLIP 利用 LLM 增强 CLIP**](https://arxiv.org/abs/2411.04997)：微软发布了 LLM2CLIP，利用大语言模型增强 CLIP 处理长且复杂标题的能力，显著提升了其跨模态性能。
- [**AgentInstruct 生成海量合成数据**](https://arxiv.org/abs/2407.03502)：AgentInstruct 框架自动创建了 **2500 万个** 多样化的提示词-响应对（prompt-response pairs），使 **Orca-3** 模型在 AGIEval 上的表现提升了 **40%**，超越了 GPT-3.5-turbo 等模型。

---

**主题 4：AI 工具演进与工作流优化**

- [**Augment 为开发者加速 LLM 推理**](https://www.augmentcode.com/blog/rethinking-llm-inference-why-developer-ai-needs-a-different-approach)：Augment 详细介绍了他们在优化 LLM 推理方面的方法，通过提供完整的代码库上下文（这对开发者 AI 至关重要）并克服延迟挑战，以确保快速且高质量的输出。
- [**DSPy 引入 VLM 支持进军视觉领域**](https://x.com/karthikkalyan90/status/1858609018228355414)：DSPy 宣布对 Vision-Language Models 提供 Beta 版支持，并在教程中展示了如何从图像（如网站截图）中提取属性，标志着其能力的重大扩展。
- [**Hugging Face 通过 Pipelines 简化视觉模型**](https://x.com/mervenoyann/status/1858537240596451472)：Hugging Face 的 pipeline 抽象现在支持视觉语言模型，使得以统一方式处理文本和图像变得前所未有的简单。

---

**主题 5：社区动态与重大举措**

- [**Roboflow 获 4000 万美元融资以强化 AI 视觉**](https://fortune.com/2024/11/19/exclusive-roboflow-vision-ai-startup-raises-40-million-series-b/)：Roboflow 在 B 轮融资中额外筹集了 **4000 万美元**，用于增强视觉 AI 的开发者工具，旨在医疗和环境等行业部署应用。
- [**Google AI 工作坊将在黑客松中释放 Gemini 潜力**](https://lu.ma/agents-hackathon-googleai)：11 月 26 日的一场特别 Google AI 工作坊将在 LLM Agents MOOC 黑客松期间向开发者介绍如何基于 Gemini 进行构建，包括现场演示以及与 Google AI 专家的直接问答。
- [**LAION 发布 1200 万个用于 ML 的 YouTube 样本**](https://laion.ai/blog/laion-disco-12m/)：LAION 宣布推出 LAION-DISCO-12M，这是一个包含 **1200 万个** YouTube 链接及元数据的数据集，旨在支持音频和音乐领域基础模型的研究。

---

---

# 第一部分：Discord 高层级摘要

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Muon 优化器表现不如 AdamW**：讨论强调，由于不合适的学习率和调度技术，**Muon 优化器**的表现明显逊于 **AdamW**，引发了对其优越性主张的怀疑。
  
  - 一些成员指出，使用更好的超参数可以改善对比结果，但针对未调优基准线的批评依然存在。
- **Neural Metamorphosis 引入自变形网络**：关于 [Neural Metamorphosis (NeuMeta)](https://adamdad.github.io/neumeta/) 的论文提出了一种通过直接学习连续权重流形来创建自变形神经网络的新方法。
  
  - 这可能允许针对任何网络规模和配置进行即时采样，并引发了关于利用小模型更新来实现更快训练的讨论。
- **SAE 特征转向推动 AI Safety**：Microsoft 的合作者发布了一份关于 **SAE 特征转向**的报告，展示了其在 **AI Safety** 方面的应用。
  
  - 研究表明，转向 **Phi-3 Mini** 可以增强拒绝行为，同时强调需要探索其优势和局限性。
- **Cerebras 收购推测**：讨论集中在为什么像 **Microsoft** 这样的大公司尚未收购 **Cerebras**，推测这可能是因为它们有潜力与 **NVIDIA** 竞争。
  
  - 一些成员回忆起 **OpenAI** 在 2017 年前后曾有兴趣收购 **Cerebras**，暗示了其在 AI 领域的持久影响力。
- **经济可行性担忧下 Scaling Laws 依然重要**：**Scaling laws** 仍被视为模型的基本属性，但在经济上，进一步推行扩展已变得不可行。
  
  - 一位成员幽默地指出，如果你没有达到 *GPT-4 或 Claude 3.5* 的预算水平，可能还不需要担心收益递减问题。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4-turbo 更新引发性能审查**：成员们正在询问 **gpt-4-turbo-2024-04-09** 更新，并注意到其此前**出色的表现**。
  
  - 一位用户对更新后模型思考能力的**不一致性**表示沮丧。
- **NVIDIA 的 Add-it 提升 AI 图像编辑**：讨论重点介绍了顶尖的 AI 图像编辑工具，如 'magnific' 和 NVIDIA 的 **_**[**Add-it**](https://research.nvidia.com/labs/par/addit/) **_**，后者允许根据文本提示添加对象。
  
  - 成员们对这些新兴工具的**可靠性**和实际**可访问性**表示**怀疑**。
- **Temperature 设置影响 AI 创造力**：在井字棋（Tic Tac Toe）讨论中，较高的 **Temperature** 设置会导致 AI 回复的**创造力**增加，这可能会阻碍其在基于规则的游戏中的表现。
  
  - 参与者注意到，在 **Temperature 0** 时，由于其他影响因素，AI 的回复虽然一致，但并非*完全相同*。
- **LLMs 作为游戏状态机面临挑战**：用户指出，当 **LLMs** 被用于表示井字棋等游戏中的**状态机**时，会表现出**不一致性**。
  
  - 大家一致认为，需要比单纯依赖 LLMs 更有效地处理游戏逻辑的框架。
- **难度参数增强 AI 游戏表现**：参与者讨论了引入**难度参数**来改进 AI 游戏表现，例如让 AI **提前思考几步**。
  
  - 随着用户对长时间的 AI 对话感到**疲劳**，进一步的讨论被暂停。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 2.5 模型问题**：用户报告在使用 ORPO trainer 训练 **Qwen 2.5** 模型时出现**结果不一致**的情况，并指出切换到 **Llama 3.1** 后解决了这些错误。
  
  - 讨论集中在**模型类型**的变化是否通常会影响训练结果，见解表明此类调整可能不会显著影响结果。
- **基于人类反馈的强化学习 (RLHF)**：社区探索了集成 **PPO (RLHF)** 技术，表明映射 **Hugging Face** 组件可以简化该过程。
  
  - 成员们分享了开发**奖励模型 (reward model)** 的方法论，为有效实施 **RLHF** 提供了一个支持性框架。
- **多轮对话微调**：提供了关于为**多轮对话**格式化数据集的指导，建议使用 **EOS tokens** 来指示响应终止。
  
  - 强调利用适合**多轮格式**的数据（如 **ShareGPT**）来增强训练效果。
- **Aya Expanse 支持**：确认了对 Cohere 的 [Aya Expanse](https://huggingface.co/CohereForAI/aya-expanse-8b) 模型支持，解决了成员关于其集成的咨询。
  
  - 讨论未深入探讨更多细节，主要集中在对 **Aya Expanse** 兼容性的积极确认上。
- **语言模型中的合成数据**：一场讨论强调了**合成数据 (synthetic data)** 对于加速**语言模型**开发的重要性，并引用了论文 [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502)。
  
  - 该论文探讨了**模型崩溃 (model collapse)** 问题，并强调在使用**合成数据**时需要进行细致的质量和多样性管理。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **Pipeline 抽象简化视觉模型**：@huggingface transformers 中的 [pipeline abstraction](https://x.com/mervenoyann/status/1858537240596451472) 现在支持**视觉语言模型 (vision language models)**，简化了推理过程。
  
  - 此次更新使开发者能够在统一的框架内高效处理视觉和文本数据。
- **Diffusers 引入 LoRA Adapter 方法**：**Diffusers** 为支持 **LoRA** 的模型添加了两个新方法：`load_lora_adapter()` 和 `save_lora_adapter()`，方便与 LoRA checkpoints 直接交互。
  
  - 这些新增功能消除了加载权重时对先前命令的需求，提升了工作流效率。
- **精确取消学习 (Exact Unlearning) 揭示 LLM 中的隐私漏洞**：最近一篇关于将**精确取消学习**作为**机器学习模型**隐私机制的论文揭示了其在**大语言模型 (LLMs)** 应用中的不一致性。
  
  - 作者强调，虽然取消学习可以管理训练期间的数据删除，但模型仍可能保留未经授权的知识，如**恶意信息**或错误信息。
- **RAG Fusion 变革生成式 AI**：一篇文章讨论了 **RAG Fusion** 作为**生成式 AI** 的关键转变，预测了 AI 生成方法的重大变革。
  
  - 它探讨了 **RAG 技术** 的影响及其在各种 AI 应用中的预期集成。
- **Augment 为开发者优化 LLM 推理**：Augment 发布了一篇 [博客文章](https://www.augmentcode.com/blog/rethinking-llm-inference-why-developer-ai-needs-a-different-approach)，详细介绍了他们通过提供**全代码库上下文 (full codebase context)** 来增强 LLM 推理的策略，这对于开发者 AI 至关重要，但也带来了延迟挑战。
  
  - 他们概述了旨在提高推理速度和质量的优化技术，确保为客户提供更好的性能。

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Mochi 在排行榜上表现优于 CogVideo**：成员们讨论了 **Mochi-1** 目前在排行榜上表现优于其他模型，尽管其 Discord 社区似乎并不活跃。
  
  - **CogVideo** 因其功能和更快的处理速度而受到欢迎，但在纯 **text-to-video** 任务中，仍被认为逊色于 Mochi。
- **Stable Diffusion 初学者的顶级模型选择**：建议新用户探索 **Auto1111** 和 **Forge WebUI**，作为 Stable Diffusion 的入门友好选项。
  
  - 虽然 **ComfyUI** 提供了更多控制权，但其复杂性可能会让新手感到困惑，这使得 **Forge** 成为一个更具吸引力的选择。
- **在 GGUF 和大型模型格式之间做出选择**：**stable-diffusion-3.5-large** 与 **stable-diffusion-3.5-large-gguf** 之间的区别在于 GPU 处理数据的方式，GGUF 允许进行更小的、分块的处理。
  
  - 鼓励拥有更强大配置的用户使用基础模型以获得速度优势，而 VRAM 有限的用户可以探索 GGUF 格式。
- **推出 AI 驱动的新闻内容创作软件**：一位用户介绍了一款能够监控新闻话题并生成 AI 驱动的社交媒体帖子的软件，强调了其在 **LinkedIn** 和 **Twitter** 等平台上的实用性。
  
  - 该用户正在为这项服务寻找潜在客户，并强调了其在房地产等行业的应用能力。
- **社区成员对 WebUI 的偏好**：社区分享了关于不同 **WebUI** 的经验，指出 **ComfyUI** 在工作流设计方面的优势，特别是对于熟悉音频软件的用户。
  
  - 一些人对 **Gradio** 的表单填写特性表示不满，呼吁提供更用户友好的界面，同时也承认了 Forge 强大的优化能力。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **OpenAI o1 模型现在支持 streaming**：OpenAI 的 [o1-preview](https://platform.openai.com/docs/api-reference/streaming) 和 o1-mini 模型现在支持 **streaming**（流式传输），允许在所有付费使用层级进行开发。主分支默认集成了此功能，增强了**开发者能力**。
  
  - Aider 的主分支在发布新版本时会提示更新，但开发者环境不保证自动更新。
- **Aider API 兼容性与配置**：有成员担心 Aider 的默认输出限制被设置为 **512 tokens**，尽管通过 OpenAI API 支持高达 **4k tokens**。成员们讨论了调整配置的方法，包括利用 `extra_params` 进行自定义设置。
  
  - 强调了将 Aider 与本地模型和 Bedrock（如 **Anthropic Claude 3.5**）连接时的问题，需要正确格式化的元数据 JSON 文件以避免冲突和错误。
- **Anthropic API 速率限制变更引入分级限制**：Anthropic 取消了每日 token 限制，在不同层级引入了新的基于分钟的输入/输出 token 限制。此更新可能需要开发者升级到更高层级以获得更高的速率限制。
  
  - 用户对层级结构表示怀疑，认为这是激励用户为了获得更多访问权限而增加支出的策略。
- **Pixtral Large 发布增强 Mistral 性能**：Mistral 发布了 **Pixtral Large**，这是一个基于 **Mistral Large 2** 构建的 **124B** 多模态模型，在 **MathVista**、**DocVQA** 和 **VQAv2** 上达到了 **state-of-the-art** 性能。它可以处理超过 **30 张高分辨率图像**，具有 **128K context window**，并可在 [API](https://console.mistral.ai/) 中作为 `pixtral-large-latest` 进行测试。
  
  - Elbie 提到希望看到 Pixtral Large 的 **Aider benchmarks**，并指出虽然之前的 **Mistral Large** 表现出色，但并未完全满足 Aider 的要求。
- **qwen-2.5-coder 的困境及与 Sonnet 的对比**：用户报告称 OpenRouter 的 **qwen-2.5-coder** 有时无法提交更改或进入死循环，可能是由于设置参数不正确或内存压力。它在 **architect mode** 下的表现比常规模式更差。
  
  - 与 **Sonnet** 的对比表明，根据初步经验，qwen-2.5-coder 可能无法达到 Sonnet 的效率，这引发了关于影响性能的训练因素的讨论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **7900XTX 显卡性能**：一位用户报告称 **7900XTX** 在处理文本时效率很高，但在使用专为 AMD 设计的 **amuse** 软件处理图形密集型任务时会出现明显减速。另一位用户询问了针对 **7900XTX** 图形性能测试的具体模型。
  
  - 用户正在积极评估 **7900XTX** 在不同工作负载下的能力，指出其在文本处理方面的优势，同时强调了在图形密集型应用中的挑战。
- **基于 Llama 3.2 的角色扮演模型**：一位用户寻求适用于角色扮演的优质 **NSFW/Uncensored**、基于 **Llama 3.2** 的模型推荐。另一位成员回应称，通过适当的搜索可以找到此类模型。
  
  - 讨论强调了在 **HuggingFace** 上查找特定角色扮演模型的难度，建议需要更好的搜索策略。
- **LM Studio 的远程服务器使用**：一位用户寻求将 LM Studio 指向远程服务器的配置建议。建议包括使用 **RDP** 或 **openweb-ui** 以增强用户体验。
  
  - 一位用户表示有兴趣利用 **Tailscale** 远程托管推理后端，并强调了在不同设置中保持性能一致性的重要性。
- **Windows vs Ubuntu 推理速度**：测试显示，一个 1b 模型在 Windows 上的速度为 **134 tok/sec**，而 Ubuntu 以 **375 tok/sec** 的表现远超前者，表明存在巨大的性能差异。一位成员建议，这种差异可能是由于 Windows 中不同的电源计划造成的，并建议切换到高性能模式。
  
  - 社区正在研究导致操作系统之间推理速度差异的因素，将电源管理设置视为潜在原因。
- **AMD GPU 性能挑战**：讨论强调，虽然 AMD GPU 提供了**高效的性能**，但受限于软件支持，使其在某些应用中缺乏吸引力。一位成员指出，由于与各种工具的兼容性问题，使用 AMD 硬件通常感觉像是一场艰苦的战斗。
  
  - 参与者对 AMD GPU 的软件兼容性表示沮丧，强调需要改进支持以充分利用 AMD 硬件的能力。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **O1 流式传输现已上线**：[OpenAIDevs](https://x.com/OpenAIDevs/status/1858609150999359559) 宣布 **OpenAI** 的 **o1-preview** 和 **o1-mini** 模型现在支持真正的流式传输功能，所有付费层级的开发者均可使用。
  
  - 此次更新解决了之前“伪”流式传输方法的局限性，社区对最新流式传输功能的进一步明确表示出兴趣。
- **Gemini 模型遇到速率限制**：用户报告在利用 Google 的 `Flash 1.5` 和 `Gemini Experiment 1114` 时频繁出现 503 错误，这表明这些较新的实验性模型可能存在速率限制问题。
  
  - 社区讨论强调了资源耗尽错误，成员建议 OpenRouter 提供更好的沟通以减轻此类技术中断的影响。
- **Mistral 模型面临无限循环问题**：关于 **Mistral** 模型（如 `Mistral Nemo`）和 `Gemini` 在与 OpenRouter 配合使用时出现无限循环和重复输出的问题被提出。
  
  - 建议包括调整 Temperature 设置，但用户承认解决这些技术挑战具有复杂性。
- **自定义提供商密钥需求激增**：多位用户请求访问**自定义提供商密钥 (custom provider keys)**，突显了利用它们进行多样化应用的浓厚兴趣。
  
  - 在 beta-feedback 频道中，用户还对 **beta 自定义提供商密钥**和自带 **API keys** 表示了兴趣，表明了向更具定制化的平台集成发展的趋势。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 中的音轨分离**：一位成员在 [#use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1308172671149936745) 频道询问了在录音过程中获取独立人声音轨的方法。
  
  - 这凸显了用户对 **音频管理工具** 的持续关注，并提到了用于说话人分离和 mp4 录音共享的 [Simli_NotebookLM](https://github.com/jjmlovesgit/Simli_NotebookLM)。
- **创新的视频创作解决方案**：分享了关于 [Somli](https://www.somli.com) 视频创作工具以及以极具竞争力的价格使用 [D-ID Avatar studio](https://www.d-id.com/avatar-studio) 的讨论。
  
  - 成员们交流了使用数字人（Avatar）制作视频的步骤，并为感兴趣的人提供了相关的编程实践。
- **利用 NotebookLM 增强文档组织**：一位成员表示有兴趣利用 **NotebookLM** 来汇编和组织世界观构建（world-building）文档。
  
  - 该请求强调了 NotebookLM 通过有效管理大量笔记来简化 **创作过程** 的潜力。
- **在 NotebookLM 中创建定制化课程**：一位英语教师分享了他们使用 **NotebookLM** 开发针对学生兴趣的阅读和听力课程的经验。
  
  - 该方法将工具提示作为微型课程，通过实际的语言场景增强学生的语境理解。
- **利用 NotebookLM 从代码生成播客**：一位用户讨论了尝试使用 **NotebookLM** 从代码片段生成播客的实验。
  
  - 随着成员们探索各种生成技术，这展示了 NotebookLM 从多样化数据输入创建内容的通用性。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LLM2CLIP 提升 CLIP 的文本处理能力**：[LLM2CLIP](https://microsoft.github.io/LLM2CLIP) 论文利用大语言模型通过高效处理更长的描述（captions）来增强 **CLIP** 的多模态能力。
  
  - 这种集成利用微调后的 LLM 来引导视觉编码器，显著提高了 **CLIP** 在跨模态任务中的性能。
- **Neural Metamorphosis 实现自变形网络**：[Neural Metamorphosis (NeuMeta)](https://arxiv.org/abs/2410.11878) 引入了一种通过从连续权重流形（weight manifold）中采样来创建自变形神经网络的范式。
  
  - 该方法允许为各种配置动态生成权重而无需重新训练，强调了流形的平滑性。
- **AgentInstruct 自动化大规模合成数据创建**：**AgentInstruct** 框架从原始数据源生成了 2500 万个多样化的提示-响应对（prompt-response pairs），以促进 **Generative Teaching**。
  
  - 在使用该数据集进行训练后，**Orca-3** 模型在 AGIEval 上的表现比之前的模型（如 **LLAMA-8B-instruct** 和 **GPT-3.5-turbo**）提高了 **40%**。
- **LLaVA-o1 增强视觉语言模型的推理能力**：[LLaVA-o1](https://arxiv.org/abs/2411.10440) 为视觉语言模型引入了结构化推理，使其能够在复杂的视觉问答任务中进行自主的多阶段推理。
  
  - **LLaVA-o1-100k** 数据集的开发为推理密集型基准测试的精度提升做出了显著贡献。
- **合成数据生成策略讨论**：讨论强调了 **合成数据生成** 在训练鲁棒 AI 模型中的重要性，并引用了 **AgentInstruct** 等框架。
  
  - 参与者强调了大尺度合成数据集在实现基准测试性能提升方面的作用。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **在 PyTorch 中集成 Triton CPU 后端**：分享了一个 [GitHub Pull Request](https://github.com/pytorch/pytorch/pull/133408)，旨在将 **Triton CPU** 作为 **PyTorch** 中的 Inductor 后端，目标是利用 **Inductor-generated kernels** 对新后端进行压力测试。
  
  - 此次集成旨在评估 Triton CPU 后端的性能和鲁棒性，从而增强 **PyTorch** 内部的计算能力。
- **PyTorch FSDP 内存分配洞察**：成员们讨论了在保存操作期间，**FSDP** 分配是如何发生在设备内存的 `CUDACachingAllocator` 中，而非 CPU 上。
  
  - 未来的 **FSDP** 版本预计将改进分片技术，通过消除对参数进行 **all-gathering** 的需求来减少内存分配，发布目标定于今年年底或明年年初。
- **Liger Kernel 蒸馏损失的增强**：针对在 **Liger Kernel** 中实现新的**蒸馏损失函数**提出了一个 [GitHub issue](https://github.com/linkedin/Liger-Kernel/issues/371)，概述了支持各种对齐和蒸馏层的动机。
  
  - 讨论强调了通过引入多样化的**蒸馏层**来改进模型训练技术的潜力，旨在提高性能和灵活性。
- **优化寄存器分配策略**：讨论强调了**寄存器分配**中的**溢出 (spills)** 会严重影响性能，主张增加**寄存器**利用率以缓解此问题。
  
  - 成员们探索了诸如定义和重用单个寄存器瓦片（tiles）以及平衡资源分配以最小化溢出的策略，特别是在添加额外的 **WGMMAs** 时。
- **解决 FP32 MMA 中的 FP8 对齐问题**：发现了一个挑战，即 **FP32 MMA** 中的 **FP8** 输出线程片段所有权（thread fragment ownership）与预期输入不符，如[此文档](https://arxiv.org/pdf/2407.08608)所述。
  
  - 为了在不通过 warp shuffle 降低性能的情况下解决这种不匹配，对共享内存张量采用了静态布局置换（static layout permutation），以实现高效的数据处理。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Runner H Beta 版发布，迈向 ASI**：**H Company** 宣布发布 **Runner H** 的 Beta 版本，标志着超越当前缩放法则（scaling laws）、迈向**人工超级智能 (ASI)** 的重大进展。[H Company 的推文](https://x.com/hcompany_ai/status/1858907033921069449)强调了这一里程碑。
  
  - 该公司强调，通过这次 Beta 版发布，他们不仅是在推出一款产品，还在开启 AI 发展的新篇章。
- **Pixtral 论文揭示先进技术**：**Sagar Vaze** 讨论了 [Pixtral 论文](https://arxiv.org/abs/2410.07073)，特别引用了 **第 4.2 节**、**第 4.3 节**和**附录 E**。该论文深入探讨了与当前研究相关的复杂方法论。
  
  - **Sagar Vaze** 提供了见解，指出这些详细讨论为小组当前的项目提供了宝贵的背景信息。
- **Runner H 在基准测试中超越 Qwen**：如 [WebVoyager 论文](https://arxiv.org/abs/2401.13919)所述，**Runner H** 在使用 **WebVoyager 基准测试**时表现出优于 **Qwen** 的性能。
  
  - 这一成功突显了 **Runner H** 通过创新的自动评估方法在现实场景评估中的优势。
- **树搜索方法的进展**：一份[最新报告](https://arxiv.org/abs/2411.11694)强调了在 **Jinhao Jiang** 和 **Zhipeng Chen** 等研究人员的共同努力下，**树搜索（tree search）**技术取得了显著进展。
  
  - 这些改进增强了大型语言模型（LLM）的推理能力。
- **探索 Q* 算法的基础**：重新审视了 **Q*** 算法，引发了关于其在当前 AI 方法论中基础性作用的讨论。
  
  - 成员们表达了怀旧之情，承认该算法对当今 AI 技术产生的深远影响。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Cerebras 的 Llama 3.1 推理速度**：Cerebras 声称其 **Llama 3.1 405B** 的推理速度达到 [969 tokens/s](https://x.com/artificialanlys/status/1858594969927512476?s=46)，比中位数供应商基准快 **10 倍**以上。
  
  - 批评者认为，虽然 **Cerebras** 在 Batch Size 为 1 的评估中表现出色，但在更大 Batch Size 下性能会有所下降，建议对比时应考虑这些差异。
- **OpenAI 增强语音功能**：**OpenAI** 宣布在 [chatgpt.com](https://chatgpt.com) 上为**付费用户**推出语音功能更新，旨在让演示变得更容易。
  
  - 此次更新允许用户通过演示学习发音，突显了对增强用户交互的持续关注。
- **Roboflow 获得 4000 万美元 B 轮融资**：**Roboflow** 额外筹集了 **4000 万美元**，用于增强其在医疗和环境等各个领域的视觉 AI 应用开发工具。
  
  - CEO Joseph Nelson 强调了他们的使命是赋能开发者有效地部署视觉 AI，并强调了在数字世界中“看见”的重要性。
- **关于小语言模型的讨论**：社区辩论了小语言模型 (**SLM**) 的定义，建议指出 **1B 到 3B** 规模的模型属于小型。
  
  - 共识是较大的模型不符合此分类，并指出了基于在消费级硬件上运行能力的区分。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **AIMakerspace 领导本地 RAG 工作坊**：加入 [AIMakerspace](https://twitter.com/AIMakerspace) 于 **11 月 27 日**举办的活动，学习使用开源 LLM 构建本地 RAG 应用，重点关注 **LlamaIndex Workflows** 和 **Llama-Deploy**。
  
  - 该活动提供实战培训和构建稳健本地 LLM 栈的深入见解。
- **LlamaIndex 在 Microsoft Ignite 上与 Azure 集成**：**LlamaIndex** 在 [#MSIgnite](https://twitter.com/hashtag/MSIgnite) 上展示了其与 **Azure** 集成的端到端解决方案，涵盖 **Azure OpenAI**、**Azure AI Embeddings** 和 **Azure AI Search**。
  
  - 鼓励参会者联系 **@seldo** 了解此全面集成的更多细节。
- **将聊天历史集成到 RAG 系统中**：一位用户讨论了如何利用 **Milvus** 和 **Ollama** 的 LLM，通过自定义索引方法将聊天历史整合到 RAG 应用中。
  
  - 社区建议修改现有的聊天引擎功能，以增强与其工具的兼容性。
- **使用 SQLAutoVectorQueryEngine 实现引用**：提出了关于使用 **SQLAutoVectorQueryEngine** 获取行内引用及其与 **CitationQueryEngine** 潜在集成的咨询。
  
  - 顾问建议将引用工作流分开，因为实现引用逻辑本身非常直接。
- **评估 RAG 系统中的检索指标**：针对 RAG 系统中缺乏用于评估检索指标的 Ground Truth 数据表达了担忧。
  
  - 社区成员被要求提供方法论或教程，以有效解决这一测试挑战。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 0.10.0 发布，包含 1200 个 Commit**：团队宣布发布 [tinygrad 0.10.0](https://github.com/tinygrad/tinygrad/releases/tag/v0.10.0)，包含 **1200 多个 Commit**，重点在于**最小化依赖**。
  
  - tinygrad 现在同时支持**推理**和**训练**，并立志构建硬件，近期已**筹集资金**。
- **ARM 测试失败及解决**：用户报告了 **aarch64-linux** 架构上的测试失败，具体是在测试期间遇到了 **AttributeError**。
  
  - 这些问题在不同架构上均可复现，潜在的解决方案包括在 `test_interpolate_bilinear` 中集成 `x.realize()`。
- **Kernel Cache 测试修复**：通过添加 `Tensor.manual_seed(123)` 实现了对 `test_kernel_cache_in_action` 的修复，确保测试套件通过。
  
  - **ARM 架构**上仅剩一个问题，相关解决方案正在讨论中。
- **在 tinygrad 中调试 Jitted 函数**：设置 **DEBUG=2** 会导致进程在底部行持续输出，表明其正在运行。
  
  - tinygrad 中的 Jitted 函数仅执行 **GPU Kernels**，因此内部的 print 语句不会产生可见输出。

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **分词训练损害单词识别**：一位成员指出，单词 **'strawberry'** 在训练过程中被分词（tokenized），这干扰了其在 **GPT-4o** 和 **Google’s Gemma2 27B** 等模型中的识别，揭示了不同系统面临的类似挑战。
  
  - 这一分词问题影响了模型准确识别某些单词的能力，引发了关于通过更好的训练方法来改进单词识别的讨论。
- **Cohere 研究工具 Beta 计划**：[Cohere 研究原型 Beta 计划](https://forms.gle/Teis9VwM6eZP6nxVA) 的报名将于今晚 **东部时间午夜** 截止，入选者可提前体验专为研究和写作任务设计的新工具。
  
  - 鼓励参与者提供 **详细反馈** 以帮助塑造工具的功能，重点是创建复杂的报告和摘要。
- **配置 Command-R 模型语言设置**：一位用户询问如何为 **Command-R 模型** 设置 **Preamble**，以确保使用 **保加利亚语** 回答，并避免与 **俄语** 术语混淆。
  
  - 他们提到使用 **API 请求构建器** 进行自定义，表明模型响应中需要更清晰的语言区分。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **开发分支面临稳定性问题**：一位成员报告称，**开发分支 (development branch)** 目前处于 [进行中状态](https://link.to.commit)，`interpreter --version` 显示为 **1.0.0**，表明 **UI** 和功能可能出现了退化。
  
  - 另一位成员自愿解决这些问题，并指出最后的提交记录为 **9d251648**。
- **寻求技能生成方面的帮助**：**Open Interpreter** 用户请求在技能生成（skills generation）方面提供帮助，提到预期的文件夹为空，并寻求后续操作指导。
  
  - 建议遵循与教学模型相关的 [GitHub 指南](https://github.com/openinterpreter/01)，未来的版本计划整合此功能。
- **UI 简化收到褒贬不一的反馈**：围绕最近的 **UI 简化** 展开了讨论，一些成员更喜欢之前的设计，表示对旧界面感到更舒适。
  
  - 开发者确认了反馈，并询问用户是否更青睐旧版本。
- **Claude 模型的问题引发担忧**：有报告指出 **Claude 模型** 出现故障；临时切换模型解决了问题，这引发了对 **Anthropic** 服务可靠性的担忧。
  
  - 成员们询问这些问题是否在不同版本中持续存在。
- **Ray Fernando 在最新播客中探索 AI 工具**：在 [YouTube 视频](https://www.youtube.com/watch?v=9DAZP1MdcbQ) 中，Ray Fernando 讨论了能增强构建过程的 **AI 工具**，重点介绍了 **10 个助力快速构建的 AI 工具**。
  
  - 这集名为“10 个真正产生效果的 AI 工具”的视频为对工具利用感兴趣的开发者提供了宝贵的见解。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **DSPy 引入 VLM 支持**：新的 [DSPy VLM 教程](https://x.com/karthikkalyan90/status/1858609018228355414) 现已发布，重点介绍了处于 Beta 阶段的 **VLM 支持**，用于从图像中提取属性。
  
  - 该教程利用 **网站截图** 演示了有效的属性提取技术。
- **DSPy 与非 Python 后端的集成**：成员报告称，将 DSPy 编译的 JSON 输出与 **Go** 集成时准确率有所下降，这引发了对 Prompt 处理复现的担忧。
  
  - 有建议提出使用 **inspect_history** 方法来创建针对特定应用定制的模板。
- **DSPy 中的成本优化策略**：讨论了 DSPy 如何通过 **Prompt 优化** 以及可能使用小语言模型作为代理来降低 Prompt 成本。
  
  - 然而，对于 **长上下文限制** 存在担忧，需要采取上下文剪枝和 **RAG** 实现等策略。
- **长上下文 Prompt 的挑战**：强调了在长文档解析中带有大量上下文的 Few-shot 示例效率低下的问题，并批评了对模型在大规模输入中保持连贯性的依赖。
  
  - 提议包括将处理过程分解为更小的步骤，并最大限度地提高每个 Token 的信息量，以解决上下文相关的问题。
- **DSPy Assertions 与 MIRPOv2 的兼容性**：针对即将发布的 2.5 版本中 DSPy Assertions 与 **MIRPOv2** 的兼容性提出了疑问，并参考了过去的兼容性问题。
  
  - 这表明人们对这些功能在框架内将如何演变和集成持续关注。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral Large 引入 Pixtral 模型**：社区成员表示有兴趣尝试最新的 **Mistral Large** 和 **Pixtral** 模型，并寻求经验丰富用户的专业建议。
  
  - 讨论反映了正在进行的实验，以及对这些 AI 模型性能见解的渴求。
- **MI300X 训练现已投入运行**：使用 **MI300X** 进行的训练现已投入运行，多项上游变更确保了性能的一致性。
  
  - 一位成员强调了上游贡献在维持训练过程可靠性方面的重要性。
- **bitsandbytes 集成增强**：有人对在不使用 **bitsandbytes** 时仍需在训练期间导入它的必要性提出了质疑，建议将其设为可选。
  
  - 提出了一项建议，即实现一个上下文管理器来抑制导入错误，旨在提高代码库的灵活性。
- **Axolotl v0.5.2 发布**：新的 [Axolotl v0.5.2](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.2) 已经发布，具有多项修复、增强的单元测试和升级的依赖项。
  
  - 值得注意的是，该版本通过解决 `pip install axolotl` 问题，修复了 v0.5.1 版本的安装问题，为用户提供了更顺畅的更新体验。
- **Phorm Bot 弃用担忧**：有人对 **Phorm Bot** 可能被弃用提出了疑问，有迹象表明它可能出现了故障。
  
  - 成员们推测，该问题源于机器人在迁移到新组织后仍引用过时的仓库 URL。

 

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Max Graph 集成增强知识图谱**：有人询问 **Max Graph** 是否可以增强传统的 **Knowledge Graphs**（知识图谱），将其作为 Agentic [RAG tools](https://arxiv.org/pdf/2404.16130) 之一用于统一 **LLM inference**。
  
  - Darkmatter 指出，虽然 **Knowledge Graphs** 充当数据结构，但 **Max Graph** 代表了一种计算方法。
- **MAX 提升图搜索性能**：关于利用 **MAX** 提升图搜索性能的讨论显示，目前的能力需要将整个图复制到 **MAX** 中。
  
  - 提出了一种潜在的变通方案，涉及将图编码为一维字节 **Tensor**，尽管内存需求可能会带来挑战。
- **区分图类型及其用途**：一位用户指出了各种图类型之间的区别，指出 **MAX computational graphs** 与计算相关，而 **Knowledge Graphs** 存储关系。
  
  - 他们进一步解释说，**Graph RAG** 利用知识图谱增强检索，而 **Agent Graph** 描述了 **Agent** 之间的数据流。
- **Max Graph 的 Tensor 依赖性受到关注**：Msaelices 质疑 **Max Graph** 是否从根本上与 **Tensor** 绑定，并注意到其 API 参数受限于 **TensorTypes**。
  
  - 这引发了在继续进行实现咨询之前查阅 API 文档的建议。

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **Gemini Google AI 工作坊**：参加于 **PT 时间 11/26 下午 3 点**举行的 [Google AI 工作坊](https://lu.ma/agents-hackathon-googleai)，重点讨论在 LLM Agents MOOC Hackathon 期间使用 **Gemini** 进行构建。活动包括 **Gemini** 的现场演示，以及与 **Google AI 专家**进行直接支持的互动问答。
  
  - 参与者将深入了解 **Gemini** 以及 **Google** 的 AI 模型和平台，利用最新技术增强 Hackathon 项目。
- **第 10 讲公告**：第 10 讲定于今日 **PST 时间下午 3:00** 举行，并提供[直播](https://www.youtube.com/live/f3KKx9LWntQ)以供实时参与。本节课将介绍 Foundation Models 开发中的重大更新。
  
  - 所有课程材料，包括直播链接和作业，均可在[课程网站](http://llmagents-learning.org/f24)上获取，确保核心资源的集中访问。
- **Percy Liang 的演讲**：斯坦福大学副教授 **Percy Liang** 将发表题为“**Foundation Models 时代的开源与科学**”的演讲。他强调，尽管目前存在可访问性限制，但开源对于推动 AI 创新至关重要。
  
  - Liang 强调了社区资源对于开发强大的开源模型的必要性，以促进该领域的共同进步。
- **实现非英语模型的 State of the Art**：*Tejasmic* 询问了关于如何使**非英语模型**达到 **State of the Art** 性能的策略，特别是在**数据点较少**的语言中。
  
  - 有建议提议将该问题提交至专门频道，工作人员正在那里积极审阅类似咨询。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Flex Attention 块分数复制**：一名成员报告了在尝试复制 Flex Attention 的 `score_mod` 函数中的 **Attention Scores** 时出现错误，导致 *Unsupported: HigherOrderOperator* 变异错误。
  
  - 另一名成员确认了这一限制，并引用了 [Issue](https://github.com/pytorch-labs/attention-gym/issues/19) 以获取更多详情。
- **Attention 分数提取技巧**：成员们讨论了由于无法访问 **SDPA 内部机制**，使用 Vanilla Attention 复制 **Attention Scores** 所面临的挑战，并建议修改 **Gemma 2 Attention 类**可能提供一种变通方法。
  
  - 有人分享了一个 [GitHub Gist](https://gist.github.com/drisspg/c66d79d51b5dd1895a552cef0820ba2e)，详细介绍了一种在不使用 Triton Kernel 的情况下提取 Attention Scores 的技巧，尽管这偏离了标准的 **Torchtune** 实现。
- **Vanilla Attention 变通方案**：据透露，由于缺乏对 **SDPA 内部机制**的访问，使用 Vanilla Attention 复制 **Attention Scores** 是不可行的，这引发了对替代方案的探索。
  
  - 一名成员建议修改 **Gemma 2 Attention 类**可能会提供解决方案，因为它更易于进行 Hack 修改。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **LAION-DISCO-12M 发布，包含 1200 万个链接**：LAION 宣布推出 **LAION-DISCO-12M**，这是一个包含 **1200 万个**公开 YouTube 样本链接及其元数据的集合，旨在支持**通用音频和音乐**的基础机器学习研究。该计划的更多详情见其[博客文章](https://laion.ai/blog/laion-disco-12m/)。
  
  - [LAION 的推文](https://x.com/laion_ai/status/1858751486265622934)中也强调了这一发布，重点介绍了该数据集在增强音频相关 Foundation Models 方面的潜力。
- **音频研究的元数据增强**：LAION-DISCO-12M 集合中包含的**元数据**旨在促进音频分析领域 Foundation Models 的研究。
  
  - 几位开发者对公告中强调的潜在用例表示兴奋，强调了在**音频机器学习领域**需要更好的数据。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Transformer Lab Demo 今日开启**：今天的 [Transformer Lab Demo](https://discord.com/channels/1089876418936180786/1089876419926032396/1308466715360890920) 展示了 **Transformer 技术** 的最新进展。
  
  - 鼓励成员加入并参与讨论，以探索这些进步。
- **元数据过滤（Metadata Filtering）会议提醒**：一场关于 [元数据过滤](https://discord.com/events/1089876418936180786/1300483739872399411) 的会议定于明天举行，由 [#1262961960157450330](https://discord.com/channels/1089876418936180786/1262961960157450330) 频道的专家主持。
  
  - 参与者将获得有关 AI 中有效数据处理实践的见解。
- **Refact AI 讨论自主 AI Agent**：[Refact AI](https://discord.com/events/1089876418936180786/1300459081181429810) 将于本周四介绍如何构建 **自主 AI Agent** 以端到端地执行工程任务。
  
  - 他们还将回答与会者的提问，提供互动学习的机会。

 

---

**Alignment Lab AI Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**LLM Finetuning (Hamel + Dan) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道详细摘要和链接

{% if medium == 'web' %}

### **Eleuther ▷ #**[**general**](https://discord.com/channels/729741769192767510/729741769738158194/1308169088371003392) (73 条消息🔥🔥):

> - `Cerebras 技术推测`
> - `Pre-NeurIPS 聚会`
> - `超参数调优工具`
> - `用于无审查 LLM 的 NovelAI 服务`
> - `Llama-2 70B 性能`

- **关于 Cerebras 收购的疑问**：讨论集中在为什么像 Microsoft 这样的大公司还没有收购 Cerebras，推测可能是因为它们有潜力与 NVIDIA 竞争。
  
  - 一些成员回忆起 OpenAI 在 2017 年左右曾有兴趣收购 Cerebras，暗示了其在 AI 领域的持久影响力。
- **EleutherAI 爱好者聚会**：分享了一份在 Dolores Park 举行的 Pre-NeurIPS 聚会邀请，鼓励参加者 RSVP 以获取零食和饮料，并与 AI 爱好者交流。
  
  - 此次聚会旨在在即将到来的活动前联系 EleutherAI 社区成员，促进关于 AI 和休闲话题的讨论。
- **超参数调优工具建议**：一位参与者询问超参数调优工具的建议，引发了对 HEBO 和 Grid Search 的推荐，后者更易于教学演示。
  
  - 一位成员指出，尽管 Grid Search 效率较低，但其图形表示可能更具美感。
- **使用 NovelAI 探索无审查 LLM**：成员们讨论了用于创作成人内容的无审查 LLM 选项，推荐了 NovelAI，并重点关注其端到端加密的隐私特性。
  
  - 潜在用户得到了关于隐私实践的保证，强调不记录日志并安全处理生成的故事。
- **来自 Cerebras WSE 的性能见解**：Cerebras Wafer Scale Engine (WSE) 因其令人印象深刻的能力而受到关注，据报道其训练速度与推理速度相当甚至更快。
  
  - 有人对性能声明的有效性表示担忧，并认为需要独立基准测试来证实该系统的效率。

**提到的链接**：

- [Cerebras Now The Fastest LLM Inference Processor; Its Not Even Close](https://www.forbes.com/sites/karlfreund/2024/11/18/cerebras-now-the-fastest-llm-inference-processor--its-not-even-close/)：该公司攻克了 Llama-3.1 405B 基础模型的推理，并彻底碾压了对手。
- [Breaking the Molecular Dynamics Timescale Barrier Using a Wafer-Scale System](https://arxiv.org/abs/2405.07898)：分子动力学 (MD) 模拟改变了我们对纳米尺度的理解，推动了材料科学、计算化学以及包括生物物理在内的多个领域的突破……
- [Cerebras - Wikipedia](https://en.m.wikipedia.org/wiki/Cerebras#Deployments)：未找到描述
- [AI Friends @ Dolores Park (pre Neurips gathering) · Luma](https://lu.ma/fi3edk93)：如果你感兴趣请 RSVP！AI 朋友们 - 让我们在 Dolores Park 见面。距离上次在旧金山为 EleutherAI 成员（及朋友）举办的聚会已经太久了 🌁 随着……
- [‎Gemini - Challenges and Solutions for Aging Adults](https://gemini.google.com/share/6d141b742a13)：由 Gemini 创建
- [Anon's Entry Level /lmg/ Guide For Clueless Newbies](https://rentry.org/lmg-spoonfeed-guide)：又名：大勺子喂饭指南，张开嘴说“啊——”。0. 基础入门：我将按顺序为你提供要求和链接。你被要求阅读所有的安装说明……
- [Models - Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending)：未找到描述

---

### **Eleuther ▷ #**[**research**](https://discord.com/channels/729741769192767510/747850033994662000/1308197395624099840) (46 条消息🔥):

> - `Muon 优化器对比`
> - `Neural Metamorphosis`

- **Muon 优化器表现不如 AdamW**：讨论指出，由于学习率和调度技术不当，Muon 优化器的表现明显逊于 AdamW，这导致人们对其优越性的主张产生怀疑。
  
  - 一些成员指出，使用更好的超参数可以改善对比结果，但针对未调优基线的批评依然存在。
- **Neural Metamorphosis 引入了自变形网络**：关于 [Neural Metamorphosis (NeuMeta)](https://adamdad.github.io/neumeta/) 的论文提出了一种通过直接学习连续权重流形来创建自变形神经网络的新方法。
  
  - 这可能允许对任何网络规模和配置进行即时采样，并引发了关于利用小模型更新实现更快训练的讨论。
- **nGPT Bug 已修复及 AI2 的新项目**：一位成员分享了 [nGPT](https://github.com/NVIDIA/ngpt/issues/1) 中的 Bug 已被修复，NVIDIA 目前正在进一步开发这种归一化 Transformer 模型。
  
  - 此外，AI2 已开始在他们的 Olmo 项目中尝试复现 nGPT，突显了在改进优化方法方面的持续努力。
- **对数据可用性的担忧**：几位成员表示，**数据缺乏**是各种模型和方法的主要障碍，并讨论了优化激活和有效的上采样（upscaling）。
  
  - 有建议认为，与其关注外部分辨率，不如通过更好的建模来增强内部分辨率，从而实现改进。

**提到的链接**：

- [MARS: Unleashing the Power of Variance Reduction for Training Large Models](https://arxiv.org/abs/2411.10438)：训练深度神经网络——以及最近的大型模型——需要高效且可扩展的优化器。像 Adam、AdamW 及其变体这样的自适应梯度算法一直是这一领域的核心...
- [Neural Metamorphosis](https://adamdad.github.io/neumeta/)：未找到描述
- [Huizhuo Yuan (@HuizhuoY) 的推文](https://x.com/HuizhuoY/status/1858634508230115381)：对于 GPT2 small 上的 Muon，我们尝试了 2e-2, 6e-3, 3e-3 和 6e-4。对于 GPT2 small 上的 MARS，我们尝试了相同的一组学习率。对于 medium 和 large，我们分别按 1/2, 1/3 比例下调了学习率...
- [Geometric Optimisation on Manifolds with Applications to Deep Learning](https://arxiv.org/abs/2203.04794)：我们设计并实现了一个 Python 库，旨在帮助非专家以高效、可扩展且易于集成到数据科学家工作流中的方式使用这些强大的工具...
- [NVIDIA/ngpt](https://github.com/NVIDIA/ngpt/issues/1)：归一化 Transformer (nGPT)。通过在 GitHub 上创建账户来为 NVIDIA/ngpt 的开发做出贡献。

---

### **Eleuther ▷ #**[**scaling-laws**](https://discord.com/channels/729741769192767510/785968841301426216/1308171824085078117) (12 messages🔥):

> - `LLM 中的 Scaling laws`
> - `LLM 预训练的可扩展性`
> - `Scaling 中的财务考量`
> - `AI 能力预测`
> - `观测性 Scaling laws 研究`

- **Scaling Laws 尚未消亡**：*Scaling 并没有死——它永远不会被终结*，因为它是模型的根本属性，但从经济角度来看，进一步推进 Scaling 已变得不可行。
  
  - 一位成员幽默地指出，如果你还没有达到 *GPT-4 或 Claude 3.5* 的预算水平，可能还不需要担心收益递减（diminishing returns）的问题。
- **关于 LLM 预训练角色的讨论**：目前正在讨论将工作重点转向 **LLM 预训练的可扩展性（scalability）和性能**，许多人认为尽管存在潜在的 Scaling 限制，这些领域仍然至关重要。
  
  - 有人担心 Scaling 是否已经实际上“完成”，用户在思考重点是否应该转移到现有模型的扩展性和性能上。
- **关于 Scaling 的多元观点**：一位成员指出了围绕 Scaling 叙事中的张力：虽然 *Scaling 在技术上仍然有效*，但普通用户能感知到的性能提升正在放缓。
  
  - 来自 The Information 和 Bloomberg 的文章链接强调了开发更先进模型过程中持续存在的挑战。
- **对观测性 Scaling Laws 的咨询**：一位成员寻求关于 *观测性 Scaling laws（observational scaling laws）* 和预测性评估的论文及研究员推荐，并强调了这对于政府情境感知（situational awareness）的重要性。
  
  - 从背景来看，英国政府的 AI Safety Institute 正在努力关注能力预测，这表明各界对 Scaling laws 的趋势有着广泛兴趣。
- **关于模型局限性的疑问**：在关于 Scaling 的讨论中，另一位成员对 *语言模型在某些任务上的提升* 是否纯粹是规模（scale）的函数表示怀疑。
  
  - 他们声称熟悉几篇在 Scaling 背景下讨论这些局限性的论文。

**提到的链接**：[Scaling realities](https://www.interconnects.ai/p/scaling-realities)：两种说法都是真实的。Scaling 仍然有效。OpenAI 等公司仍然过度承诺了他们的保证。

---

### **Eleuther ▷ #**[**interpretability-general**](https://discord.com/channels/729741769192767510/1052314805576400977/1308259110625738795) (1 messages):

> - `SAE 特征引导 (feature steering)`
> - `AI 安全研究`
> - `Phi-3 Mini 模型性能`
> - `与 Microsoft 的合作`
> - `越狱 (Jailbreak) 鲁棒性`

- **SAE 特征引导推动 AI 安全**：Microsoft 的合作者发布了一份关于 **SAE 特征引导** 的报告，展示了其在 **AI 安全** 方面的应用。
  
  - 研究表明，引导 **Phi-3 Mini** 可以增强拒绝行为（refusal behavior），同时强调需要探索其优势和局限性。
- **报告对比 Anthropic 的发现**：这份新报告被认为是 **Anthropic** 最近发现的有价值补充，扩展了其研究的实际应用。
  
  - 感谢贡献者提供的 **Top-k SAE 实现**，该实现支持了这项工作。
- **Phi-3 拒绝机制的探索**：摘要指出，在推理时引导模型激活是更新模型权重以实现安全行为的一种具有成本效益的替代方案。
  
  - 研究结果包括：虽然 **特征引导** 增强了针对 **越狱尝试 (jailbreak attempts)** 的鲁棒性，但它可能会对整体 Benchmark 性能产生负面影响。
- **研究通过多种格式共享**：该研究已公开，包含 [预印本 (preprint)](https://arxiv.org/abs/2411.11296) 和 [直接 PDF](https://arxiv.org/pdf/2411.11296) 的链接。
  
  - 鼓励参与此研究主题的讨论，并特别引用了 [此推文线程](https://x.com/KyleDevinOBrien/status/1858698819904696447) 以供进一步探讨。

**提到的链接**：

- [来自 Kyle O'Brien (@KyleDevinOBrien) 的推文](https://x.com/KyleDevinOBrien/status/1858698819904696447)：使用 SAE 特征引导语言模型是一种很有前景的、由可解释性驱动的 AI 安全方法。然而，其优势和局限性仍有待探索。我们研究了引导 Phi-3 进行拒绝行为并测量...
- [使用稀疏自编码器引导语言模型拒绝行为](https://arxiv.org/abs/2411.11296)：部署语言模型的负责任做法包括引导模型识别并拒绝回答被认为不安全的提示，同时遵循安全提示。实现这种行为...

---

### **Eleuther ▷ #**[**lm-thunderdome**](https://discord.com/channels/729741769192767510/755950983669874798/1308259865155600405) (65 messages🔥🔥):

> - `lm_eval config.json issue`
> - `pawsx and headqa errors`
> - `glue evaluation metrics`
> - `headqa performance update`
> - `RWKV model preparation`

- **lm_eval 在 config.json 上遇到困难**：有用户报告 `lm_eval` 尝试下载 `config.json`，但在模型仓库的 main 分支中无法找到该文件。
  
  - 临时的解决方法包括指定本地模型目录，但 `lm_eval` 需要与 OpenAI 兼容的服务器运行在同一台机器上。
- **pawsx 和 headqa 遇到错误**：一位成员在 **0.4.5** 版本中运行 `pawsx` 和 `headqa` 时遇到问题，面临与任务组和配置相关的意外错误。
  
  - 讨论指出这可能是一个 bug，可能的解决方法是使用 **0.4.4** 版本，但结果各异。
- **澄清 glue 评估指标**：成员们讨论了在 glue 任务中聚合指标的需求，重点是对 **cola** 的 **mcc** 等非准确率指标进行平均。
  
  - 结论是，分别对每个指标进行宏平均（macro-averaging）可能是预期的做法，同时由于指标不同而排除 **cola**。
- **更新 headqa 功能**：项目更新表明 **headqa** 现在应该可以正常工作，尽管之前有错误报告。
  
  - 用户报告遇到了与 Arrow 类型相关的新错误，但通过更新 **datasets** 库解决了此问题。
- **RWKV 模型评估准备**：频道对话暗示正在为发布基于海量数据集训练的最新 **RWKV-6** 模型做准备。
  
  - 成员们希望相对于之前的模型运行保持评估方法的一致性，并参考了历史论文。

 

---

### **OpenAI ▷ #**[**ai-discussions**](https://discord.com/channels/974519864045756446/998381918976479273/1308160954634207332) (87 messages🔥🔥):

> - `Model Updates`
> - `AI Image Editing Tools`
> - `Scrolling Issues on Devices`
> - `Using Python for Infographics`
> - `LLM Application Evaluation`

- **关于 GPT 模型更新的问题**：成员们正在询问 GPT 的当前版本，特别是寻求关于 'gpt-4-turbo-2024-04-09' 更新的细节，有人声称其之前的表现非常出色。
  
  - 一位用户对模型预期思考能力的不一致表示沮丧。
- **顶尖 AI 图像编辑工具**：讨论围绕顶级 AI 图像编辑工具展开，提到了 'magnific' 以及 NVIDIA 最近开发的 ***Add-it***，后者允许根据文本提示添加对象。
  
  - 成员们对知名度较低的工具表示怀疑，质疑其可靠性和实际访问权限。
- **Chromebook 上的滚动问题**：一位用户报告在 Chromebook 上滚动聊天记录时出现问题，引发了关于技术限制和 RAM 问题的讨论。
  
  - 注意到其他人在不同设备上可以正常滚动，这表明可能是特定于 Chromebook 的技术问题。
- **使用 Python 制作信息图**：成员们对通过 ChatGPT 和 Python 生成信息图感到好奇，讨论建议使用 Google Colab 等在线资源来执行 Python 代码。
  
  - 一位用户提到在实验时达到了数据分析限制，表明该工具存在学习曲线。
- **评估 LLM 应用**：针对 RAG 等 LLM 应用的常见问题进行了对话，这些问题主要源于开发者对其局限性的误解。
  
  - 基础的 embedding RAG 与传统内容索引之间存在重要区别，突显了知识差距。

 

**Link mentioned**: [Add-it](https://research.nvidia.com/labs/par/addit/): no description found

 

---

### **OpenAI ▷ #**[**gpt-4-discussions**](https://discord.com/channels/974519864045756446/1001151820170801244/1308173441416822814) (10 条消息🔥):

> - `Game Bots 行为`
> - `Temperature 对 AI 的影响`
> - `Tic Tac Toe GPT 策略`

- **Game Bots：停机挫败感**：一位用户表达了对 Game Bots 偶尔停机并忽略指令的沮丧，称这种情况令人*愤怒*。
  
  - 这凸显了游戏过程中 Bot 行为潜在的可靠性问题。
- **Temperature 影响 AI 创造力**：在关于 Tic Tac Toe 的讨论中，有人指出 AI 模型中较高的 Temperature 会导致创造力增加，但这在基于规则的游戏中可能是有害的。
  
  - 正如一位参与者所说，在这些受限的环境中，*表现出创造力和随机性并不是好事*。
- **低 Temperature 带来一致的响应**：据分享，由于随机性有限，Temperature 设置为 0 通常会导致 AI 针对相同的 Prompt 生成相似的响应。
  
  - 然而，也有人指出，由于其他影响因素，响应并不会*完全相同*。
- **理解 Temperature 的影响**：用户解释说，Temperature 会影响 AI 选择准确度较低的 Token 的可能性，较高的值会增加此类选择的机会。
  
  - 这一见解对于理解基于 Temperature 设置的 AI 输出变异性至关重要。

 

---

### **OpenAI ▷ #**[**prompt-engineering**](https://discord.com/channels/974519864045756446/1046317269069864970/1308172960884068362) (44 条消息🔥):

> - `AI 在 Tic Tac Toe 中的表现`
> - `LLM 与状态机的挑战`
> - `用户对 AI 策略的沮丧`
> - `游戏日志与移动追踪`
> - `AI 游戏玩法中的难度参数`

- **用户在 AI 拦截策略上遇到困难**：用户报告称 AI 在 Tic Tac Toe 中不能一致地拦截对手的移动，导致游戏体验不佳。有人建议明确指令，例如为 AI 和用户更严格地定义角色。
  
  - 一位成员指出，特定的措辞（如使用 'You will be X and I will be O'）提高了 AI 的表现。
- **AI 的表现在游戏过程中下降**：据观察，随着游戏的进行，AI 的有效性会降低，导致移动执行出错。用户讨论了潜在原因，如注意力分散和在较长游戏中缺乏上下文管理。
  
  - 用户担心如果没有结构化的追踪或状态管理，AI 可能会在游戏过程中失去其战略能力。
- **游戏日志作为解决方案**：围绕记录移动是否能刷新 AI 的战略理解展开了讨论。这种日志记录可能有助于在整个游戏中保持一致性，但也有人提出了关于通过排列组合增加复杂性的疑问。
  
  - 参与者承认，虽然这对于像 Tic Tac Toe 这样的游戏可能有效，但并不一定能克服 LLM 固有的局限性。
- **LLM 在游戏机制方面的局限性**：成员们指出了使用 LLM 表示游戏中状态机的挑战，认为它们容易出现不一致。共识是，虽然这是一项创造性的尝试，但可能不会产生可靠的结果。
  
  - 对话强调了需要能够有效处理游戏逻辑的合适框架，而不是仅仅依赖 LLM。
- **探索 AI 难度和策略**：难度参数的想法作为改进 AI 游戏玩法的一种潜在策略出现，并建议可以要求 AI 提前思考几步。然而，由于用户感到疲劳，关于这一概念的进一步讨论被推迟了。
  
  - 随着疲劳感的产生，一位用户提到了在应对失眠的同时沉浸在以 AI 为中心的讨论中所带来的精神负担。

 

---

### **OpenAI ▷ #**[**api-discussions**](https://discord.com/channels/974519864045756446/1046317269069864970/1308172960884068362) (44 条消息🔥):

> - `AI Tic-Tac-Toe Bot`
> - `阻挡策略问题`
> - `模型局限性`
> - `状态机表示`
> - `难度参数`

- **AI 在 Tic-Tac-Toe 中难以进行阻挡**：用户注意到他们的 AI Bot 经常无法正确阻挡对手的移动，导致在游戏过程中可能失败。
  
  - 调整 Prompt 以指定阻挡优先级虽然提高了性能，但正如多位用户提到的，挑战依然存在。
- **游戏过程中感知的 AI 性能下降**：一位参与者观察到，随着游戏的进行，AI 在决策方面的有效性似乎有所下降。
  
  - 有人担心是否可以通过移动日志（move logging）或其他方法来维持或提高 AI 的性能。
- **LLM 作为状态机的局限性**：一位用户对使用 LLM 建模状态机表示怀疑，认为很可能会出现错误和不一致。
  
  - 尽管如此，一些人认为将 LLM 集成到游戏逻辑中的练习非常有趣，且有助于理解它们的能力。
- **未来移动计算的挑战**：讨论强调了引入指定 AI 应提前考虑多少步的 Prompt 对于改进游戏玩法的重要性。
  
  - 关于 AI 在 Tic-Tac-Toe 中保持竞争力所需的最佳难度参数出现了疑问。
- **AI 讨论期间用户的睡眠不足**：一位用户谈到了长时间参与 AI 对话带来的疲劳，并质疑自己对 Tic-Tac-Toe 的专注度。
  
  - 这引发了关于管理疲劳以及在长时间游戏过程中游戏 AI 潜在复杂性的旁注评论。

 

---

### **Unsloth AI (Daniel Han) ▷ #**[**general**](https://discord.com/channels/1179035537009545276/1179035537529643040/1308160837277450360) (135 messages🔥🔥):

> - `Qwen 2.5 model issues` (Qwen 2.5 模型问题)
> - `Unsloth Training FAQs` (Unsloth 训练常见问题)
> - `Multiple turn conversation fine-tuning` (多轮对话微调)
> - `Utilizing Azure for training` (利用 Azure 进行训练)
> - `Reinforcement Learning from Human Feedback (RLHF)` (来自人类反馈的强化学习)

- **Qwen 2.5 的结果不一致**：几位用户报告了在使用 ORPO 训练器训练 **Qwen 2.5** 模型时遇到的问题，并指出切换到 **Llama 3.1** 后解决了这些错误。
  
  - 用户讨论了持续报错的潜在原因，其中一位建议模型类型的更改通常不应影响结果。
- **多轮对话微调**：回复者提供了关于如何为多轮对话格式化数据集的指导，建议使用 **EOS tokens** 在每次回答后停止。
  
  - 强调了使用适合多轮格式的数据（如 **ShareGPT**）是有效训练的关键。
- **探索使用 Azure 进行 AI 训练**：一位用户询问了使用 **Azure 的 GPUs** 进行模型微调的可行性，因为他们有大量的额度可用。
  
  - 确认了利用 Azure 资源将大有裨益，特别是考虑到像 Mac Mini M4 Pro 这样的本地机器不支持训练。
- **实施 RLHF 技术**：记录了关于集成 **PPO (RLHF)** 的讨论，表明映射 Hugging Face 组件可以简化此过程。
  
  - 社区分享了关于开发奖励模型的各种方法的见解，为新手提供了一个支持性的框架。
- **模型响应一致性担忧**：一位用户表达了对模型在调整 temperature 设置后仍持续产生相同输出的担忧。
  
  - 建议尝试进一步提高 temperature，以探索响应的多样性。

**Links mentioned**:

- [GGUF Editor - a Hugging Face Space by CISCai](https://huggingface.co/spaces/CISCai/gguf-editor): 未找到描述
- [rombodawg/Rombos-Coder-V2.5-Qwen-7b · Hugging Face](https://huggingface.co/rombodawg/Rombos-Coder-V2.5-Qwen-7b): 未找到描述
- [Huggingface GGUF Editor · ggerganov/llama.cpp · Discussion #9268](https://github.com/ggerganov/llama.cpp/discussions/9268): Huggingface GGUF 编辑器 🎉 看看我的最新项目 🌍✨ 一个专门为编辑 GGUF 元数据设计的强大编辑器，并能直接从任何 Huggingface 仓库下载结果...
- [unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit): 未找到描述
- [unsloth/Qwen2.5-7B-Instruct · Hugging Face](https://huggingface.co/unsloth/Qwen2.5-7B-Instruct): 未找到描述
- [Dynamic Deep Learning | Richard Sutton](https://www.youtube.com/watch?v=75jr5E4OzEE): ICARL 研讨会系列 - 2024 冬季动态深度学习，Richard Sutton 的研讨会 —————————————————— 摘要：尽管取得了巨大成功，当前的深度学习方法...
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): 请参阅下面的列表以获取我们所有的 Notebooks：
- [GitHub - huggingface/trl: Train transformer language models with reinforcement learning.](https://github.com/huggingface/trl): 使用强化学习训练 Transformer 语言模型。 - huggingface/trl
- [trl dpo AttributeError: 'generator' object has no attribute 'generate' · Issue #2292 · huggingface/trl](https://github.com/huggingface/trl/issues/2292): trl dpo AttributeError: 'generator' object has no attribute 'generate' print('start training...') if list(pathlib.Path(training_args.output_dir).glob("checkpoint-\*"))...
- [Google Colab](https://colab.research.google.com/drive/1tEd1FrOXWMnCU9UIvdYhs61tkxdMuKZu?usp=sharing): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1XamvWYinY6FOSX9GLvnqSjjsNflxdhNc?usp=sharing#scrollTo=p31Z-S6FUieB): 未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/1gof0o1/a_team_from_mit_built_a_model_that_scores_619_on/): 未找到描述
- [microsoft/orca-agentinstruct-1M-v1 · Datasets at Hugging Face](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1): 未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**help**](https://discord.com/channels/1179035537009545276/1179777624986357780/1308163033675075687) (41 条消息🔥):

> - `Using CSV for Chat Models` (在对话模型中使用 CSV)
> - `Aya Expanse Support` (Aya Expanse 支持)
> - `Model Finetuning Issues` (模型微调问题)
> - `Container Installation Problems` (容器安装问题)
> - `Unsloth Trainer Compatibility` (Unsloth Trainer 兼容性)

- **Using CSV for Chat Models**: 一位成员询问是否有类似于 Titanic CSV 示例的 Notebook，用于以对话方式使用 CSV。
  
  - 该讨论串中未提供具体的回复或解决方案。
- **Aya Expanse Support**: 关于是否支持 Cohere 的 [Aya Expanse](https://huggingface.co/CohereForAI/aya-expanse-8b) 模型的咨询得到了另一位成员的肯定确认。
  
  - 讨论未包含更多细节或疑虑。
- **Model Finetuning Issues**: 一位用户报告了使用量化微调模型时遇到的挑战，并询问是否可以在保存 Q4 格式的同时保存其他位宽格式。
  
  - 讨论引导至资源分享和对当前保存方法的确认，但未给出解决方案。
- **Container Installation Problems**: 一位成员在基于 `nvidia/cuda:12.3.0-base-ubuntu20.04` 构建的容器中安装 `bitsandbytes==0.43.1` 时遇到困难。
  
  - 其他成员建议了可能的变通方法，但未直接解决安装问题。
- **Unsloth Trainer Compatibility**: 有人提出了在为 Unsloth 使用 ORPO trainer 时 'Trainer.tokenizer' 被弃用的问题，引发了关于更新的讨论。
  
  - 确认了该 trainer 仍然有效，并建议更新 Unsloth 以获得向后兼容性。

**提到的链接**:

- [Google Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sharing#scrollTo=yqxqAZ7KJ4oL): 未找到描述
- [Home](https://github.com/unslothai/unsloth/wiki): 以 2-5 倍的速度和减少 80% 的显存微调 Llama 3.2, Mistral, Phi, Qwen 2.5 和 Gemma LLM - unslothai/unsloth
- [Hugging Face – The AI community building the future.](https://huggingface.co/settings/tokens): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing#scrollTo=8ywYGU2bLW1o): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1T5-zKWM_5OD21QHwXHiV9ixTRR7k3iB9?usp=sharing): 未找到描述
- [Google Colab](https://colab.research.google.com/drive/1Ys44kVvmeZtnICzWz0xgpRnrIOjZAuxp?usp=sha): 未找到描述
- [Unsloth Notebooks | Unsloth Documentation](https://docs.unsloth.ai/get-started/unsloth-notebooks): 查看下方列表以获取我们所有的 Notebook：
- [Google Colab](https://colab.research.google.com/drive/1T-YBVfnphoVc8E2E854qF3jdia2Ll2W2?usp=sharing): 未找到描述

---

### **Unsloth AI (Daniel Han) ▷ #**[**research**](https://discord.com/channels/1179035537009545276/1257011997250424842/1308281236581515275) (1 条消息):

> - `Synthetic Data in Language Models` (语言模型中的合成数据)
> - `AgentInstruct`

- **Synthetic Data's Role in Language Model Development**: 讨论集中在**合成数据**对于加速**语言模型**开发的重要性，正如论文 [AgentInstruct: Toward Generative Teaching with Agentic Flows](https://arxiv.org/abs/2407.03502) 中所强调的那样。
  
  - 该论文探讨了**模型崩溃**和人工策展的需求，强调有效利用合成数据需要细致的质量和多样性管理。
- **Post-Training Data Creation for Skill Teaching**: 作者建议将合成数据用于**训练后 (post-training)**，即由强大的模型创建数据，以向其他模型传授新的技能或行为。
  
  - 这种创新方法旨在利用合成数据的潜力，同时减轻与模仿现有模型相关的担忧。

---

### **HuggingFace ▷ #**[**announcements**](https://discord.com/channels/879548962464493619/897387888663232554/1308511513740578848) (1 条消息):

> - `视觉模型的 Pipeline 抽象`
> - `Diffusers 中的新方法`
> - `具备扩展上下文的 Qwen 2.5`
> - `Pixtral Large 支持`
> - `Open LLM Leaderboard 上的 CO2 计算`

- **Pipeline 抽象现在支持视觉模型**：@huggingface transformers 中的 [pipeline 抽象](https://x.com/mervenoyann/status/1858537240596451472) 现在支持 **视觉语言模型 (Vision Language Models)**，简化了推理过程。
  
  - 这一增强功能使开发者能够更轻松地同时处理视觉和文本数据。
- **Diffusers 引入新的适配器方法**：为支持 **LoRA** 的模型在 **Diffusers** 中添加了两个新方法：`load_lora_adapter()` 和 `save_lora_adapter()`。
  
  - 这些方法允许直接与 LoRA 检查点 (checkpoints) 交互，而无需使用之前的命令来加载权重。
- **Qwen 2.5-Turbo 快速处理 1M Token**：新发布的 **Qwen 2.5-Turbo** 模型拥有 **100 万 Token** 的上下文长度，推理速度提升了 **4.3 倍**。
  
  - 此次升级确保了更快的处理速度，将首个 Token 的响应时间 (time to first token) 从 **4.9 分钟缩短至仅 68 秒**。
- **Pixtral Large 已集成至 transformers**：[Pixtral Large](https://x.com/mervenoyann/status/1858576496425644482) 现在已在 Hugging Face transformers 中获得原生支持，丰富了模型库。
  
  - 感谢 @art_zucker，这一支持增强了 transformers 框架的通用性。
- **LLM 排行榜中的 CO2 排放追踪**：Open LLM Leaderboard 已更新，包含 **CO2 计算**，允许用户评估模型评估对环境的影响。
  
  - [在此查看排行榜](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)，在选择模型时做出更可持续的选择。

**提到的链接**：

- [merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1858537240596451472)): @huggingface transformers 的 pipeline 抽象现在支持视觉语言模型，方便推理 🫰🏻
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1858772629424898438)): 在支持 LoRA 的模型上发布了 Diffusers 🧨 中的两个新方法 —— `load_lora_adapter()` 和 `save_lora_adapter()`。这有助于更直接地与 LoRA ckpt 和模型进行交互，而无需通过...
- [Vaibhav (VB) Srivastav (@reach_vb) 的推文](https://x.com/reach_vb/status/1858514911023792443)): 100 万 Token 且速度极快！🔥 在下方的 @Gradio space 中试用。引用 Qwen (@Alibaba_Qwen)：在 Qwen2.5 发布后，我们听到了社区对处理更长上下文的需求....
- [merve (@mervenoyann) 的推文](https://x.com/mervenoyann/status/1858576496425644482)): Pixtral Large 已在 @huggingface transformers 中获得支持 💗 感谢 @art_zucker 🎩
- [Sayak Paul (@RisingSayak) 的推文](https://x.com/RisingSayak/status/1857326079867838886)): Mochi-1 现在在 `diffusers` 中获得原生支持。查看原始模型卡以获取所有详细信息。提醒：它是 Apache 2.0 协议！感谢 @genmoai 团队公开他们的出色工作...
- [Alina Lozovskaya (@ailozovskaya) 的推文](https://x.com/ailozovskaya/status/1857071017866240028)): 🌱 Open LLM Leaderboard 上的 CO₂ 计算！你现在可以查看每个模型评估的 CO₂ 排放量！追踪哪些模型更环保，并做出可持续的选择🌍 🔗 排行榜：https://hug...
- [Daniel van Strien (@vanstriendaniel) 的推文](https://x.com/vanstriendaniel/status/1857012848695677345)): 很高兴看到 @huggingface 趋势榜前 10 名的数据集中有两个源自开放许可内容。

---

### **HuggingFace ▷ #**[**general**](https://discord.com/channels/879548962464493619/879548962464493622/1308163355113947187) (139 条消息🔥🔥):

> - `Gradio API 配额问题`
> - `Hub-Stats 数据集与排名`
> - `合成数据生成`
> - `NeurIPS 会议`
> - `使用 Hugging Face Hub 进行 Zero-Shot 分类`

- **Gradio API 用户面临配额限制**：用户报告在使用 Gradio API 进行音乐生成时遇到配额耗尽的问题，特别是作为 PRO 用户登录时，尽管他们获得的限制比免费用户更高。
  
  - 讨论包括关于 token 使用情况的见解，以及在 Python 代码中获取 PRO 状态以最大化使用限制的方法。
- **Hub-Stats 数据集的新见解**：最近更新的 [Hub-Stats 数据集](https://huggingface.co/datasets/cfahlgren1/hub-stats) 现在包含社交帖子，使用户能够查看自己的排名，一些用户的排名高达第 37 位和第 29 位。
  
  - 讨论涉及提升参与度和 yapping 指标，并对数据集中可能出现的竞争性方面进行了幽默调侃。
- **开源合成数据生成工具**：一位用户询问了类似于 G-LLaVa 的开源实现，并讨论了现有的替代方案，如 WizardLM 的 EvolInstruct。
  
  - 成员们对用于合成数据生成的多模态工具表现出兴趣，强调了社区对共享资源的需求。
- **NeurIPS 会议参会咨询**：用户表达了参加 NeurIPS 会议的兴趣，并询问如何查找会议期间周边活动的信息。
  
  - 这引发了关于 NeurIPS 在机器学习社区中重要性的简短讨论，显示了社区的参与度。
- **关于 Zero-Shot 分类 API 使用的澄清**：一位用户需要明确 Hugging Face Hub 的 endpoint 客户端是否可以接受多个输入进行 zero-shot 分类，并将其与常规的 post 请求进行了对比。
  
  - 有人呼吁为 zero-shot 分类提供潜在的 batch 函数，强调了当前客户端能力的局限性。

**提到的链接**：

- [Quickstart](https://huggingface.co/docs/huggingface_hub/quick-start)：未找到描述
- [zero-gpu-explorers/README · use authentication in huggingface Gradio API!!!(hosting on ZeroGPU)](https://huggingface.co/spaces/zero-gpu-explorers/README/discussions/129)：未找到描述
- [microsoft/orca-agentinstruct-1M-v1 · Datasets at Hugging Face](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)：未找到描述
- [huggingface_hub/src/huggingface_hub/inference/_client.py at main · huggingface/huggingface_hub](https://github.com/huggingface/huggingface_hub/blob/main/src/huggingface_hub/inference/_client.py)：Huggingface Hub 的官方 Python 客户端。 - huggingface/huggingface_hub
- [google-bert (BERT 社区)](https://huggingface.co/google-bert)：未找到描述
- [FacebookAI (Facebook AI 社区)](https://huggingface.co/FacebookAI)：未找到描述
- [apple (Apple)](https://huggingface.co/apple)：未找到描述
- [google (Google)](https://huggingface.co/google)：未找到描述
- [meta-llama (Meta Llama)](https://huggingface.co/meta-llama)：未找到描述

---

### **HuggingFace ▷ #**[**today-im-learning**](https://discord.com/channels/879548962464493619/898619964095860757/1308178964107362347) (3 条消息):

> - `EMA Scaling`
> - `Rust 编写的神经网络`

- **通过视频探索 EMA Scaling**：一位用户正在观看 [标题为 "Scaling EMA" 的 YouTube 视频](https://www.youtube.com/watch?v=9qtRfIP8Kx8) 以理解 **EMA Scaling** 的概念。
  
  - 该视频可能提供有关该主题的见解，并鼓励通过点赞和评论进行互动。
- **Rust 神经网络协作请求**：一位成员正在寻求构建 **Rust 神经网络** 的帮助，并希望展示其性能基准测试。
  
  - 他们正在联系社区，看是否有人具备相关专业知识或兴趣来协作完成这个项目。

 

**提到的链接**：[Scaling EMA](https://www.youtube.com/watch?v=9qtRfIP8Kx8)：点赞 👍。评论 💬。订阅 🟥。🏘 Discord: [https://discord.gg/pPAFwndTJdhttps://arxiv.org/pdf/2307.13813.pdfhttps://huggingface.co/papers/2307.13813#machi](https://discord.gg/pPAFwndTJdhttps://arxiv.org/pdf/2307.13813.pdfhttps://huggingface.co/papers/2307.13813#machi)...

 

---

### **HuggingFace ▷ #**[**cool-finds**](https://discord.com/channels/879548962464493619/897390579145637909/1308376275462783009) (5 messages):

> - `Exact Unlearning in LLMs`
> - `HuggingFace Feature Update`
> - `RAG Fusion in Generative AI`
> - `Voice Data Augmentation for Whisper`
> - `Chat Collapsing Feature`

- **Exact Unlearning in LLMs**: 最近的一篇论文讨论了 **exact unlearning** 作为一种隐私机制，允许用户根据要求从 **machine learning models** 中撤回其数据，并强调了其在 **Large Language Models** 应用中的不一致性。
  
  - 作者认为，虽然 unlearning 可以有效地管理训练阶段，但它并不能阻止模型拥有不允许的知识，例如 **malicious information** 或不准确的信息。
- **HuggingFace 引入 Chat Collapsing**: **HuggingFace** 推出了一项功能，当聊天记录过长时会折叠对话，从而消除了对无限滚动的需求。
  
  - 这一更新通过在长时间聊天中更轻松地访问最新对话，增强了可用性。
- **RAG Fusion 变革 Generative AI**: 一篇文章强调 **RAG Fusion** 是 **generative AI** 中的一个重大范式转变，表明它有望改变 AI 生成的格局。
  
  - 该文章深入探讨了其影响以及在各种应用中使用 **RAG techniques** 的未来。
- **语音数据增强提升 Whisper 性能**: 语音数据增强显著提高了在 **Arabic language** 上训练的 **Whisper small** 的准确性，从而降低了 **Word Error Rate (WER)**。
  
  - 这一增强标志着在微调模型以更好地完成特定语言任务方面取得的成就。

 

**Link mentioned**: [UnUnlearning: Unlearning is not sufficient for content regulation in advanced generative AI](https://arxiv.org/abs/2407.00106): Exact unlearning 最初是作为一种隐私机制引入的，允许用户根据要求从 machine learning models 中撤回其数据。不久之后，人们提出了不精确方案来缓解……

 

---

### **HuggingFace ▷ #**[**i-made-this**](https://discord.com/channels/879548962464493619/897390720388825149/1308240958147788861) (12 条消息🔥):

> - `Augment Inference Engine`
> - `LLM Ranking Arena`
> - `Response Generator Challenges`
> - `Qwen2-VL on ONNX Runtime`
> - `PTA-1 GUI Element Localization`

- **Augment 解决 LLM 推理延迟问题**：Augment 发布了一篇文章，概述了他们构建推理引擎的方法，强调**全代码库上下文（full codebase context）**对于开发者 AI 至关重要，尽管这会带来延迟问题。他们讨论了如何优化 LLM 推理，以为客户提高速度和质量。
  
  - 文章阐明，提供更多相关的上下文可以显著提高代码处理的质量。
- **用于 LLM 比较的 Judge Arena 正式发布**：一个名为 [Judge Arena](https://huggingface.co/spaces/AtlaAI/judge-arena) 的新平台已上线，旨在帮助用户根据 LLM 作为“裁判”的表现进行排名，从而促进社区参与。访问者可以运行测试并投票选出他们认为最有效的 LLM。
  
  - 该计划旨在利用众包反馈来确定哪些模型在语言评估方面表现出色，类似于 LMSys 成功的 Chatbot Arena。
- **响应生成中的挑战**：一位用户讨论了使用 **GPT-2** 和 **GPT-XL** 生成准确响应时遇到的困难，并指出所创建的训练数据在产生有效输出方面效果不佳。他们希望在改进响应和推荐生成器方面获得帮助。
  
  - 目前的方法主要涉及利用 mostly.ai 获取训练数据，但仍需要进一步改进以提升生成器的能力。
- **Qwen2-VL 在 ONNX Runtime 上运行**：经过不懈努力，一位用户成功在 *onnxruntime-web* 上部署了 **Qwen2-VL**，并指出由于部分进程在 *wasm* 上运行，目前运行速度较慢。大多数操作利用了 *webgpu*，这表明性能仍有提升潜力。
  
  - 这一实现表明在 Web 环境中运行复杂模型取得了进展，增强了可访问性。
- **PTA-1 改进 GUI 元素定位**：最近发布的 **PTA-1** 模型能有效定位屏幕截图上的 GUI 元素，从而在本地计算机上以低延迟实现快速自动化。它仅使用 **270M 参数**，却实现了优于大型模型的性能。
  
  - 该模型的输入由屏幕截图和目标元素的描述组成，通过生成边界框（bounding boxes）实现精确测量。

**提到的链接**：

- [Rethinking LLM Inference: Why Developer AI Needs a Different Approach](https://www.augmentcode.com/blog/rethinking-llm-inference-why-developer-ai-needs-a-different-approach)：来自 Augment Code 的技术博客文章，解释了他们为针对代码的 AI 应用优化 LLM 推理的方法。文章详细介绍了他们如何实现卓越的延迟和吞吐量……
- [AskUI/PTA-1 · Hugging Face](https://huggingface.co/AskUI/PTA-1)：未找到描述
- [GitHub - MaloLM/whisper-3-speach-to-text: A simple python program for audio files transcription using Whisper model.](https://github.com/MaloLM/whisper-3-speach-to-text?tab=readme-ov-file)：一个使用 Whisper 模型进行音频文件转录的简单 Python 程序。 - MaloLM/whisper-3-speach-to-text
- [Judge Arena - a Hugging Face Space by AtlaAI](https://huggingface.co/spaces/AtlaAI/judge-arena)：未找到描述
- [Judge Arena: Benchmarking LLMs as Evaluators](https://huggingface.co/blog/arena-atla)：未找到描述
- [streamerd/diplo-ai · Hugging Face](https://huggingface.co/streamerd/diplo-ai)：未找到描述
- [GitHub - streamerd/diplo-ai: Suite of data and scripts that can classify diplomatic statements into one of 62 predefined categories and generate diplomatic responses or recommendations based on the classified statement.](https://github.com/streamerd/diplo-ai)：一套能够将外交声明分类为 62 个预定义类别之一，并根据分类声明生成外交响应或建议的数据和脚本。 - stre...

---

### **HuggingFace ▷ #**[**reading-group**](https://discord.com/channels/879548962464493619/1156269946427428974/1308271333188173845) (9 messages🔥):

> - `GPU 性能对比`
> - `水冷问题`
> - `NVIDIA vs AMD 在 AI 领域的对比`
> - `Radeon Instinct MI50 和 MI60`

- **Crazypistachecat 的 GPU 配置**：目前 **Crazypistachecat** 正在使用 **3 块 RX 6800**，因为第四块显卡出现技术问题导致系统崩溃。
  
  - 他们打算在其他地方测试这块有问题的显卡，并提到它目前装有水冷头 (water block)。
- **NVIDIA vs AMD 在 AI 工作负载中的表现**：在回答关于 **GPU 性能** 的问题时，有人提到 **NVIDIA** 在 AI 硬件和软件方面通常处于领先地位。
  
  - 对于预算有限的用户来说，寻找具有 **16GB VRAM** 的 **NVIDIA 显卡** 是一个经济的选择。
- **GPU 的高性价比选择**：Crazypistachecat 选择使用 RX 6800 GPU 是因为其 **性价比**，因为它们价格低廉且支持 ROCm。
  
  - 他们强调像 **3090** 这样的选项太贵了，买一块 **3090** 的钱可以买两块 **6800**。
- **探索 Radeon Instinct 选项**：Crazypistachecat 表示有兴趣转向使用 **AMD Radeon Instinct MI50 和 MI60** GPU。
  
  - 这一转变表明人们越来越多地考虑 AMD 产品线中的其他高性能替代方案。

 

---

### **HuggingFace ▷ #**[**core-announcements**](https://discord.com/channels/879548962464493619/1014557141132132392/1308350211323592786) (1 messages):

> - `LoRA 模型支持`
> - `LoRA 的新方法`

- **两种新方法增强 LoRA 支持**：最近的更新宣布发布了 **两种新方法**，以便在模型层级更好地支持 **LoRA**。
  
  - 这一改进旨在优化用户应用程序中的性能和集成。
- **社区对 LoRA 改进感到兴奋**：成员们对 LoRA 的 **新方法** 表现出极大的热情，强调了它们对现有工作流的潜在影响。
  
  - 一位成员提到：*“这确实能提高我们模型的效率和易用性。”*

 

---

### **HuggingFace ▷ #**[**computer-vision**](https://discord.com/channels/879548962464493619/922424143113232404/1308225337209131049) (1 messages):

> - `视频目标检测`
> - `油气压裂现场分析`
> - `目标检测中的标签挑战`

- **寻求视频目标检测解决方案**：一位用户正在寻找一种 **简便的方法来对油气压裂现场 (Oil and Gas Frac site)** 的视频流进行目标检测，但在现有标签方面遇到了问题。
  
  - 他们提到该视频用于测试目的，并表示如有必要愿意为协助支付费用。
- **现有目标检测标签的挑战**：同一位用户指出，在上传视频进行测试时，**网上找到的标签** 无法正常工作。
  
  - 这些挑战凸显了对针对特定工业应用定制的、更易于获取且有效的 **目标检测解决方案** 的需求。

 

---

### **Stability.ai (Stable Diffusion) ▷ #**[**general-chat**](https://discord.com/channels/1002292111942635562/1002292112739549196/1308162162241572894) (134 条消息🔥🔥):

> - `Mochi 与 CogVideo 性能`
> - `针对初学者的模型推荐`
> - `为 Stable Diffusion 使用不同的 WebUI`
> - `GGUF 格式 vs. Large 模型`
> - `AI 驱动的新闻内容创作软件`

- **Mochi 与 CogVideo 的性能对比**：成员们讨论到，尽管 **Mochi-1** 的 Discord 社区看起来不太活跃，但它目前在排行榜上的表现优于其他模型。
  
  - *CogVideo* 因其功能和更快的处理速度而受到关注，但在纯 text-to-video 任务中，与 Mochi 相比仍被认为稍逊一筹。
- **对 Stable Diffusion 新用户的建议**：建议新用户尝试 **Auto1111** 和 **Forge WebUI**，作为 Stable Diffusion 的入门友好选项。
  
  - 虽然 *ComfyUI* 提供了更多控制权，但其复杂性可能会让新手感到困惑，这使得 Forge 成为一个更具吸引力的选择。
- **Stability 的最佳模型格式**：**stable-diffusion-3.5-large** 与 **stable-diffusion-3.5-large-gguf** 之间的区别在于 GPU 处理数据的方式，GGUF 允许进行更小的分块处理。
  
  - 建议配置较高的用户使用基础模型以获得速度优势，而 VRAM 有限的用户可以尝试 GGUF 格式。
- **用于自动化内容生成的 AI**：一位用户介绍了一款能够监控新闻话题并生成 AI 驱动的社交媒体帖子的软件，强调了它在 **LinkedIn** 和 **Twitter** 等平台上的实用性。
  
  - 该用户正在为这项服务寻找潜在客户，并强调了其在房地产等领域的应用能力。
- **各种 WebUI 的偏好与体验**：社区分享了关于不同 **WebUI** 的经验，指出 **ComfyUI** 在工作流设计方面的优势，特别是对于熟悉音频软件的用户。
  
  - 一些人对 **Gradio** 的表单填写式交互表示不满，呼吁更具用户友好性的界面，同时也承认了 Forge 强大的优化能力。

**提到的链接**：

- [InstantX/SD3.5-Large-IP-Adapter · Hugging Face](https://huggingface.co/InstantX/SD3.5-Large-IP-Adapter)：未找到描述
- [Video Generation Model Arena | Artificial Analysis](https://artificialanalysis.ai/text-to-video/arena?tab=Leaderboard)：通过在不知道提供商的情况下选择你喜欢的视频来比较 AI 视频生成模型。

---

### **aider (Paul Gauthier) ▷ #**[**general**](https://discord.com/channels/1131200896827654144/1131200896827654149/1308174799435993182) (83 条消息🔥🔥):

> - `OpenAI o1 models update`
> - `Issues with qwen-2.5-coder`
> - `Kubernetes editing with Aider`
> - `Anthropic rate limits changes`
> - `Model performance comparison`

- **OpenAI o1 模型现在支持 streaming**：OpenAI 的 o1-preview 和 o1-mini 模型现在已支持 streaming，允许在所有付费使用层级进行开发。main 分支默认支持此功能，增强了开发者的能力。
  
  - Aider 的 main 分支在发布新版本时会提示更新，但开发者环境不保证会自动更新。
- **qwen-2.5-coder 的挑战**：用户报告 OpenRouter 的 qwen-2.5-coder 有时无法提交更改或陷入死循环，引发了对解决方案的讨论。一些成员建议这可能是由于设置参数不正确或内存压力导致的。
  
  - 有关于 qwen-2.5-coder 在不同模式下性能的推测，报告指出其在 architect 模式下的表现不如常规模式。
- **探索使用 Aider 编辑 Kubernetes manifest**：讨论了使用 Aider 通过“autoscale service X”等命令编辑 Kubernetes manifest 的内容。用户正在考虑该模型在处理特定 Kubernetes 字段方面的熟练程度，以及使用参考文档进行引导的潜力。
  
  - 建议包括围绕 manifest 类型创建 wrappers，以帮助访问相关文档。
- **Anthropic API rate limits 的变更**：Anthropic 宣布取消每日 Token 限制，并在各层级引入了新的基于分钟的输入/输出 Token 限制。这一更新可能会迫使开发者为了减少与较低层级相关的速率限制而进行升级。
  
  - 一些用户表示怀疑，认为这种层级结构是鼓励消费以获取更高访问权限的一种手段。
- **通用模型性能讨论**：用户分享了关于 qwen 模型能力的见解，将其与 Sonnet 进行对比，并讨论了可能影响性能的训练因素。根据初步经验，轶事证据表明 qwen 的效率可能无法与 Sonnet 相提并论。
  
  - 参与者热衷于探索不同模型在特定任务中的有效性，期待未来的更新和改进。

**提到的链接**：

- [Model warnings](https://aider.chat/docs/llms/warnings.html#unknown-context-window-size-and-token-costs): aider 是你终端里的 AI 配对编程工具
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1858609150999359559?t=Ar_0GTXm6-fnr7HzZH_mIw&s=19): OpenAI o1-preview 和 o1-mini 现在支持 Streaming。🌊 https://platform.openai.com/docs/api-reference/streaming 我们已向所有付费使用层级的开发者开放了这些模型的访问权限...
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1858942428809662718): 在此处对比当前限制与新限制：https://docs.anthropic.com/en/api/rate-limits
- [来自 Alex Albert (@alexalbert__) 的推文](https://x.com/alexalbert__/status/1858942416595939499): 给 @AnthropicAI 开发者的好消息：明天我们将取消每日 Token (TPD) 限制，并将每分钟 Token (TPM) 拆分为 Anthropic API 各层级独立的输入/输出限制。
- [PEP 541 Request: aider · Issue #3296 · pypi/support](https://github.com/pypi/support/issues/3296#issuecomment-2484206735): 申请认领项目 aider: https://pypi.org/project/aider/ 你的 PyPI 用户名 paul-gauthier: https://pypi.org/user/paul-gauthier/ 申请理由：该项目的所有发布版本均发生在...

---

### **aider (Paul Gauthier) ▷ #**[**questions-and-tips**](https://discord.com/channels/1131200896827654144/1133060505792159755/1308163238487134209) (47 条消息🔥):

> - `Aider and OpenAI API limitations` (Aider 与 OpenAI API 限制)
> - `Connecting Aider with local models` (将 Aider 连接到本地模型)
> - `Benchmark test skips` (基准测试跳过)
> - `Using extra_params in Aider` (在 Aider 中使用 extra_params)
> - `Aider with Bedrock models` (Aider 与 Bedrock 模型)

- **Aider 在应对 OpenAI API 限制方面存在困难**：有成员担心 Aider 的默认输出限制被设置为 **512 tokens**，尽管 API 提供了很大的上下文容量。
  
  - 成员们讨论了调整配置的必要性，但指出即使将 **max new tokens 设置为 4k**，问题依然存在。
- **将 Aider 连接到本地模型时的问题**：几位成员分享了在使用 OpenAI 兼容 API 时，Aider 与本地模型之间的兼容性困扰，并建议可能需要配置 metadata JSON 文件。
  
  - 讨论包括需要正确格式化这些 JSON 文件，以避免在使用过程中出现冲突和错误。
- **难以识别基准测试中跳过的测试**：一位成员询问如何检查旧的基准测试运行文件夹，以确定是否有测试因超时而被跳过。
  
  - 结论是，目前没有一种直接的方法来验证之前基准测试运行中跳过的测试。
- **利用 extra_params 进行 Aider 配置**：讨论明确了 Aider 中的 `extra_params` 允许添加自定义参数（包括 headers），以增强 API 交互。
  
  - 提到该功能目前还不支持环境变量插值，但已经引入了全局 extra_params 功能以实现更广泛的设置。
- **Aider 与 Bedrock 模型的兼容性**：提到特定的 Bedrock 模型（如 **Anthropic Claude 3.5**）具有区域依赖性，如果与 AWS 配置不一致，可能会导致错误。
  
  - 成员们分享了解决这些问题以顺利运行 Aider 的见解，但仍有人在获取有效的 token 输出方面面临挑战。

**提到的链接**：

- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html#model-settings)：配置 LLM 的高级设置。
- [Advanced model settings](https://aider.chat/docs/config/adv-model-settings.html#global-extra-params)：配置 LLM 的高级设置。

---

### **aider (Paul Gauthier) ▷ #**[**links**](https://discord.com/channels/1131200896827654144/1268910919057149974/1308214796122062849) (3 条消息):

> - `Pixtral Large Release` (Pixtral Large 发布)
> - `Deno Project Discussion` (Deno 项目讨论)
> - `Mistral Large 2 Performance` (Mistral Large 2 性能)
> - `Aider Benchmarking` (Aider 基准测试)

- **Pixtral Large 发布，具备前沿级性能**：Mistral 宣布发布 **Pixtral Large**，这是一个基于 **Mistral Large 2** 构建的 **124B** 多模态模型，在 **MathVista**、**DocVQA** 和 **VQAv2** 上拥有顶尖性能。
  
  - 它可以使用 **128K 上下文窗口**处理超过 **30 张高分辨率图像**，并已在 [API](https://console.mistral.ai/) 中作为 `pixtral-large-latest` 提供测试。
- **对 Deno 未来的担忧**：引发了一场关于 **Deno 项目**及其近期**对 NodeJS 批评**的讨论，人们对其与传统解决方案相比的生存能力表示怀疑。
  
  - 有观点认为，一个旨在用于**网站**的开发工具不应与一个能够构建和引入外部二进制文件的**生产代码**运行时竞争。
- **对 Aider 与 Pixtral Large 兼容性的关注**：*Elbie* 表示希望看到新 Pixtral 模型的 **Aider 基准测试**，并指出虽然之前的 **Mistral Large** 非常出色，但并不完全符合 Aider 的需求。
  
  - 如果新模型表现相似且兼容，它将显著增强 Aider 的能力。

**提到的链接**：[Pixtral Large](https://mistral.ai/news/pixtral-large/)：Pixtral 成长了。

---

### **LM Studio ▷ #**[**general**](https://discord.com/channels/1110598183144399058/1110598183144399061/1308177086288105533) (65 条消息🔥🔥):

> - `7900XTX 图形性能`
> - `Llama 3.2 角色扮演模型`
> - `LM Studio 远程服务器使用`
> - `托管本地 LLM`
> - `LM Studio 中的模型更新`

- **7900XTX 在图形任务中表现吃力**：一位用户报告称，**7900XTX** 在处理文本时非常高效，但在处理图形密集型任务时会出现明显减速，特别是在使用专为 AMD 设计的 **amuse** 软件时。
  
  - 另一位用户询问了在 **7900XTX** 上测试图形性能的具体模型。
- **寻找用于角色扮演的 NSFW 模型**：一位用户请求推荐基于 **Llama 3.2** 的优质 **NSFW/Uncensored** 角色扮演模型，另一位用户回答说通过适当的搜索可以找到。
  
  - 随后讨论了在 **HuggingFace** 上寻找这些模型的难度。
- **LM Studio 的远程服务器访问**：一位用户寻求关于将 LM Studio 指向远程服务器的建议，建议包括使用 **RDP** 或 **openweb-ui** 以获得更好的体验。
  
  - 一位用户表示有兴趣使用 **Tailscale** 远程托管推理后端（inference backends），并强调了性能的一致性。
- **托管本地 LLM 的讨论**：用户交流了关于托管 **本地 LLM** 的想法，提到 LM Studio 提供了该功能的大部分能力，并建议将 **SillyTavern** 作为额外的 UI。
  
  - 一位用户指出使用本地 SSD 存储的好处，并澄清虽然这有助于缩短初始加载时间，但推理速度（inference speed）不受影响。
- **在 LM Studio 中更新模型**：一位用户询问是否可以在不删除已下载模型的情况下对其进行更新，另一位用户确认模型在更新时必须重新下载。
  
  - 这引发了关于 LM Studio 内部模型处理细节和故障排除的进一步讨论。

**提到的链接**：

- [Sideload models - Advanced | LM Studio Docs](https://lmstudio.ai/docs/advanced/sideload)：使用在 LM Studio 之外下载的模型文件
- [microsoft/orca-agentinstruct-1M-v1 · Datasets at Hugging Face](https://huggingface.co/datasets/microsoft/orca-agentinstruct-1M-v1)：未找到描述

---

### **LM Studio ▷ #**[**hardware-discussion**](https://discord.com/channels/1110598183144399058/1153759714082033735/1308159914014933062) (67 条消息🔥🔥):

> - `Windows vs Ubuntu 推理速度`
> - `AMD GPU 性能挑战`
> - `RTX 4090 配置选项`
> - `AMD W7900 基准测试结果`

- **Windows vs Ubuntu 推理速度揭晓**：测试显示，在 Windows 上，一个 1b 模型达到了 **134 tok/sec**，而 Ubuntu 以 **375 tok/sec** 的速度遥遥领先，表明存在显著的性能差距。
  
  - *一位成员提出，这种差异可能是由于 Windows 中不同的电源方案造成的，建议切换到高性能模式。*
- **AMD GPU：优秀的硬件，糟糕的软件**：讨论强调，虽然 AMD GPU 提供了**高效的性能**，但受限于软件支持的匮乏，使其在某些应用中缺乏吸引力。
  
  - *一位成员指出，由于与工具的兼容性问题，使用 AMD 硬件通常感觉像是一场艰苦的战斗。*
- **RTX 4090 配置引起关注**：发烧友们提到了将多块 **RTX 4090** 直接连接到主板的令人印象深刻的配置，这大幅增强了他们的基准测试能力。
  
  - *另一位成员幽默地提到，因为买不起机箱，他们只能把设备安装在木板上。*
- **AMD W7900 对标 Nvidia 的基准测试**：一位用户试图重新评估 **AMD W7900**，回忆起之前的基准测试显示，在使用某些提示词（prompts）时，其性能略慢于 **3090**。
  
  - *成员们同意在聊天中分享他们的测试结果，强调了协作进行基准测试的努力。*
- **推测加强版双 AMD CPU 配置**：一位成员描述了使用 **双 128 核 Bergamo AMD CPU 配置** 以及水冷 RTX 4090 的计划，旨在打造一个既强大又耗资源的稳健配置。
  
  - *他们幽默地指出，由于传统机箱的预算限制，他们正将硬件安装在木板上。*

 

**提到的链接**：[Don't ask to ask, just ask](https://dontasktoask.com/)：未找到描述

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**announcements**](https://discord.com/channels/1091220969173028894/1092729520181739581/1308484005917888572) (1 条消息):

> - `Activity page outage` (Activity 页面故障)

- **Activity 页面故障调查**：**Activity 页面** 遭遇故障，目前正在对该问题进行持续调查。
  
  - 更新消息称，该页面已于 **美国东部时间下午 12:31** 恢复运行。
- **Activity 页面恢复更新**：团队沟通称，**Activity 页面** 的故障已解决，服务已于 **美国东部时间下午 12:31** 恢复。
  
  - 在短暂的中断后，用户现在可以正常访问该页面。

 

---

### **OpenRouter (Alex Atallah) ▷ #**[**general**](https://discord.com/channels/1091220969173028894/1094454198688546826/1308171524871688296) (119 条消息🔥🔥):

> - `O1 Preview and Streaming Support` (O1 Preview 和 Streaming 支持)
> - `Gemini Model Issues` (Gemini 模型问题)
> - `Mistral API Limitations` (Mistral API 限制)
> - `OpenRouter Error Reports` (OpenRouter 错误报告)
> - `Developer Requests and Suggestions` (开发者请求与建议)

- **O1 Streaming 现已可用**：讨论显示，OpenAI 的 `o1-preview` 和 `o1-mini` 模型现在支持真正的 Streaming 功能，这一点已由 [OpenAIDevs](https://x.com/OpenAIDevs/status/1858609150999359559) 确认。这一变化为所有付费使用层级的开发者开放了访问权限。
  
  - 成员们提到了过去“伪” Streaming 方法的局限性，并表示希望更新说明能更加清晰。
- **Gemini 模型频繁报错**：用户报告在使用 Google 的 `Flash 1.5` 和 `Gemini Experiment 1114` 时出现高错误率，特别是 503 错误，这表明可能存在 Rate Limiting 问题。一些社区成员推测这可能与新实验模型的 Bug 有关。
  
  - 此外，与资源耗尽相关的错误也很常见，这促使人们建议 OpenRouter 改进沟通机制。
- **Mistral 模型限制**：有用户反映在 OpenRouter 上使用 Mistral 模型时遇到问题，特别是无限循环和重复输出。这似乎是多个模型（包括 `Mistral Nemo` 和 `Gemini`）中反复出现的模式。
  
  - 社区成员建议进行调整，例如降低 Temperature 设置，但也承认解决这些技术难题存在挑战。
- **OpenRouter Dashboard 错误**：用户在访问 OpenRouter 设置面板时遇到问题，特别是在 Brave 浏览器中，理由是缺少 Redis 配置参数。Alex 回应了这些担忧，确认正在调查并宣布面板已恢复上线。
  
  - 其他用户在 Chrome 中也注意到了类似问题，突显了不同浏览器之间的差异。
- **开发者功能请求**：社区成员讨论了 OpenRouter 平台的潜在增强功能，包括为代码输出添加“复制”按钮的请求，以及查看账户活动的功能。这些请求反映了用户对提高易用性和功能性的期望。
  
  - 这些建议受到了好评，一些成员表示实现这些功能是合理的。

**提及的链接**：

- [Large Enough](https://mistral.ai/news/mistral-large-2407/): 今天，我们发布了 Mistral Large 2，这是我们旗舰模型的新一代产品。与前代产品相比，Mistral Large 2 在代码生成、数学和推理能力方面有了显著提升……
- [来自 OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/OpenAIDevs/status/1858609150999359559): OpenAI o1-preview 和 o1-mini 现在支持 Streaming。🌊 https://platform.openai.com/docs/api-reference/streaming 我们已向所有付费使用层级的开发者开放了这些模型的访问权限……
- [Models | OpenRouter](https://openrouter.ai/models): 在 OpenRouter 上浏览模型
- [生成式 AI 如何通过 LearnLM 扩展好奇心和理解力](https://blog.google/outreach-initiatives/education/google-learnlm-gemini-generative-ai/): LearnLM 是我们全新的基于 Gemini 的模型系列，旨在提供更好的学习和教学体验。
- [Self Report Among Us GIF - Self report Among us Troll - 发现并分享 GIF](https://tenor.com/view/self-report-among-us-troll-fia-agent-gif-3619993576443358983): 点击查看 GIF
- [Cerebras 现为最快的 LLM 推理处理器；差距巨大](https://www.forbes.com/sites/karlfreund/2024/11/18/cerebras-now-the-fastest-llm-inference-processor--its-not-even-close/): 该公司攻克了 Llama-3.1 405B 基础模型的推理难题，并取得了压倒性优势。
- [司法部将推动 Google 出售 Chrome 以打破搜索垄断 - Bloomb…](https://archive.md/vePVT): 未找到描述
- [未找到标题](https://console.mistral.ai/): 未找到描述

---

### **OpenRouter (Alex Atallah) ▷ #**[**beta-feedback**](https://discord.com/channels/1091220969173028894/1277894087755829278/1308396748586221600) (5 条消息):

> - `Custom Provider Keys`
> - `Beta Custom Provider Keys`
> - `Bring Your Own API Keys`

- **关于 Custom Provider Keys 的请求大量涌现**：多位用户请求访问 **custom provider keys**，表达了在各种应用中使用它们的浓厚兴趣。
  
  - 一位用户提到，*“我想申请访问 custom provider keys”*，显示了对这些资源的明确需求。
- **对 Beta Custom Provider Keys 的兴趣**：一位用户专门请求访问 **beta custom provider keys**，强调了对最新可用功能的渴望。
  
  - 这种措辞反映了一个日益增长的趋势，即用户寻求早期访问或在项目中测试新功能。
- **Bring Your Own API Keys**：另一位用户提到希望申请访问并自带 **API keys**，表明了对更具定制化解决方案的推动。
  
  - 这反映了平台内向用户定义集成的转变。
- **对 Custom Provider Key 访问权限的赞同**：一位用户通过简单的“+1”表示支持，强调了社区对 Key 访问请求的支持。
  
  - 这种认可可能表明用户在这些 Key 的重要性上正逐渐达成共识。

---

### **NotebookLM Discord ▷ #**[**use-cases**](https://discord.com/channels/1124402182171672732/1124403655819415592/1308172671149936745) (20 条消息🔥):

> - `Audio Track Separations`
> - `Video Creation Tools`
> - `NotebookLM for Document Organization`
> - `Teaching with NotebookLM`
> - `Podcast Experimentation with Code`

- **关于分离语音音频轨道的咨询**：一位成员询问是否有人找到了在录音过程中获取两个独立语音音轨的方法。
  
  - 这反映了用户对在 NotebookLM 等平台中管理音频复杂细节工具的持续兴趣。
- **探索视频制作解决方案**：关于视频制作工具的讨论引导分享了使用 [Somli](https://www.somli.com) 的虚拟形象（Avatar）的方法，价格具有竞争力。
  
  - 分享了如何使用 D-ID Avatar studio 构建视频的步骤，并为感兴趣的人提到了相关的编码。
- **使用 NotebookLM 进行文档整理**：一位成员表达了使用 NotebookLM 有效汇编和组织世界观设定（world-building）文档的兴趣。
  
  - 对更好笔记组织的需求暗示了 NotebookLM 在简化创作流程方面的潜力。
- **使用 NotebookLM 创建定制课程**：一位英语老师分享了他们使用 NotebookLM 开发针对学生兴趣的阅读和听力课程的经验。
  
  - 该方法将工具提示作为微型课程，增强了在实际场景中对语言语境的理解。
- **尝试从代码生成播客**：一位用户对 NotebookLM 在面对代码片段时可能生成的播客类型表示好奇。
  
  - 这种实验展示了 NotebookLM 在从多样化数据输入生成内容方面的多功能性。

**提到的链接**：

- [The best games of 2024](https://play.google.com/store/apps/editorial?id=mc_bestof2024_games_fcp)：未找到描述
- [Starfish English Lessons](https://lessons.starfishenglish.com/)：未找到描述
- [GitHub - jjmlovesgit/Simli_NotebookLM: A project to take an audio file and separate it into speakers and play it with avatars and save the recording as an mp4 for sharing on social etc. Ideal for Deep Dive podcasts from Google NotebookLM](https://github.com/jjmlovesgit/Simli_NotebookLM)：一个获取音频文件并将其分离为不同发言者，使用虚拟形象播放，并将录音保存为 mp4 以便在社交媒体等平台分享的项目。非常适合 Google NotebookLM 的 Deep Dive 播客...

---

### **Notebook LM Discord ▷ #**[**general**](https://discord.com/channels/1124402182171672732/1124402182909857966/1308172382174969896) (101 条消息🔥🔥):

> - `NotebookLM UI 困惑`
> - `数据源限制`
> - `使用 NotebookLM 进行学习`
> - `移动端访问与应用更新`
> - `Podcast 生成功能`

- **用户对 NotebookLM UI 感到困惑**：许多用户对 NotebookLM 中 “GENERATE” 功能的位置表示困惑，一些人没有意识到需要点击右下角的 “notebook guide” 按钮才能访问该功能。
  
  - 用户强调，UI 中某些按钮和功能的命名可能会导致对其功能的误解。
- **数据源可用性问题**：一位用户报告了在 NotebookLM 中访问引用（citations）时遇到的问题，源内容显示为问号框而不是可读内容，这表明存在格式问题。
  
  - 建议包括将文档转换为不同格式，以解决这些显示问题。
- **NotebookLM 辅助学习**：一些用户在阅读笔记和幻灯片时通过生成 Podcast 来辅助学习，尽管也有人对音频摘要的可靠性表示担忧。
  
  - 一位用户提到，他们会预先准备好摘要，然后通过收听 Podcast 来加深学习时的理解。
- **移动端访问及即将到来的改进**：用户询问了在移动端访问 NotebookLM 的最佳方式，建议在专门的移动端 App 发布之前先保存快捷方式。
  
  - 移动端网页访问的改进预计很快就会推出，而 App 的开发仍在计划中。
- **Podcast 生成限制**：用户发现每个笔记本（notebook）只能生成一个 Podcast，这导致在尝试创建多个 Podcast 而不删除之前的内容时产生困惑。
  
  - 讨论了一些变通方法，包括将多个源合并到单个文档中以优化输出。

 

**提到的链接**：[AI Note Taking & Transcribe & Summarizer | AI Notebook App](https://ainotebook.app/)：为大学生讲座提供转录和 AI 摘要生成。专注于 YouTube Video Summarizer、PDF Summarizer、Article Summarizer。保存关键见解并使用学习指南进行复习...

 

---

### **Nous Research AI ▷ #**[**general**](https://discord.com/channels/1053877538025386074/1149866623109439599/1308163156056739880) (78 messages🔥🔥):

> - `Recent AI/ML Research Papers`
> - `AI Model Feedback`
> - `Opportunities for High School Students in AI/ML`
> - `Security Concerns in Job Postings`
> - `Forge API Access Requests`

- **探索新的 AI/ML 研究论文**：成员们讨论了多篇具有影响力的 AI/ML 研究论文，主题包括 'Cut Your Losses in Large-Vocabulary Language Models' 和 'LLMs Can Self-Improve in Long-context Reasoning'。这展现了对该领域前沿进展的持续关注，并反映了最近的发展动态。
  
  - 一些研究人员表示有兴趣将这些论文的研究成果作为自己项目和实验的基础。
- **关于 Hermes AI 响应的反馈**：一位用户对 Hermes AI 关于速度的表达方式提出了担忧，认为这可能会导致对 AI 情感或意识的误解。这类响应应当重新评估，以保持对 AI 能力本质的清晰认识。
  
  - 这引发了关于 AI 交互中此类语言适用性的讨论，突显了用户对感知到的“自我意识（sentience）”的敏感性。
- **给有志于 AI/ML 的学生的建议**：一位高中生寻求关于如何在 AI/ML 领域脱颖而出的建议，而另一位更年轻的成员分享了他们撰写关于 inference time compute 论文的经验。对话转向了在不同 AI 领域的专业化以及从基础知识开始学习。
  
  - 建议强调避免使用高度抽象的框架，转而掌握核心原理，并推荐了学习资源和个人实践项目。
- **招聘信息中的安全问题**：展开了一场关于招聘信息相关诈骗风险的讨论，特别是在 web3 领域。一位用户强调了在与潜在可疑邀约互动时应采取的预防措施。
  
  - 这次交流指向了在线工作机会中更广泛的信任问题，建议用户保持警惕并优先考虑安全。
- **Forge API 访问请求**：几位成员表达了访问 Forge API 的兴趣，一些参与者已经加入了 waitlist 并通过私信请求访问权限。对话强调了在他们的项目中进行集成的需求日益增长。
  
  - 鼓励用户联系以获取访问请求，同时确认该 API 目前处于小规模 beta 阶段。

**提及的链接**：

- [chatgpt2dataset.py](https://gist.github.com/archit-spec/02fb6fc6b7b7d310fcd208cd1514abba)：GitHub Gist：立即分享代码、笔记和片段。
- [来自 GitHub 的推文 - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others](https://x.com/Alpha7987))：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等 - FixTweet/FxTwitter
- [研究更新](https://nousresearch.typeform.com/FORGEAPI)：使用 Typeform 将数据收集变成一种体验。创建精美的在线表单、调查、测验等。免费试用。

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308166943072718918) (5 messages):

> - `LLM2CLIP`
> - `Neural Metamorphosis`
> - `AgentInstruct`

- **LLM2CLIP 增强视觉表示**：该论文提出了 **LLM2CLIP**，它利用 LLM 的能力，通过优化文本处理（特别是在管理复杂图像描述方面）来改进 **CLIP** 模型，详见[研究](https://arxiv.org/abs/2411.04997)。
  
  - 该方法通过利用微调后的 LLM 作为 CLIP 视觉编码器的引导，显著提升了 CLIP 在跨模态任务中的性能。
- **Neural Metamorphosis 创新神经网络训练**：论文介绍了 **Neural Metamorphosis (NeuMeta)**，描述了一种通过学习模型的**连续权重流形（continuous weight manifold）**来创建自变形神经网络的方法，允许在不重新训练的情况下为各种配置采样权重，详见[此处](https://arxiv.org/abs/2410.11878)。
  
  - 这种新范式利用神经隐式函数作为超网络（hypernetworks）来动态生成权重，增强了模型的灵活性。
- **AgentInstruct 自动化合成数据生成**：**AgentInstruct** 框架自动产生多样化且高质量的合成数据，通过利用原始数据源创建大规模数据集（包括用于语言模型训练的惊人的 2500 万个配对），促进了**生成式教学（Generative Teaching）**，详见[论文](https://arxiv.org/abs/2407.03502)。
  
  - 使用这些数据进行后期训练后，**Orca-3** 模型在各项基准测试中相比其前代产品有了实质性的性能提升，例如在 AGIEval 上实现了 **40% 的提升**。

 

**提到的链接**：[Neural Metamorphosis](https://arxiv.org/abs/2410.11878)：本文介绍了一种名为 Neural Metamorphosis (NeuMeta) 的新学习范式，旨在构建自变形神经网络。与为不同架构构建独立模型相反……

 

---

### **Nous Research AI ▷ #**[**research-papers**](https://discord.com/channels/1053877538025386074/1104063238934626386/1308166943072718918) (5 messages):

> - `LLM2CLIP`
> - `Neural Metamorphosis`
> - `AgentInstruct`
> - `Cross-modal representation`
> - `Synthetic data generation`

- **LLM2CLIP 增强视觉表示**：论文提出了 [LLM2CLIP](https://microsoft.github.io/LLM2CLIP)，利用大语言模型通过实现对更长描述（captions）的高效处理来提升 CLIP 的多模态能力。
  
  - *通过在描述空间中微调 LLM*，该方法在跨模态任务和视觉编码器性能方面取得了显著进展。
- **Neural Metamorphosis 引入自变形网络**：[Neural Metamorphosis (NeuMeta)](https://arxiv.org/abs/2410.11878) 提供了一种学习范式，通过直接从连续权重流形中采样来产生自变形神经网络。
  
  - 这种创新方法允许在不重新训练模型的情况下为未见过的配置生成权重，重点在于所学习流形的平滑性。
- **AgentInstruct 自动化合成数据生成**：AgentInstruct 框架自动化了用于训练语言模型的合成数据创建过程，从原始数据源生成了 2500 万个多样化的提示（prompts）和响应。
  
  - 当使用该数据集对 Mistral-7b 进行后期训练时，生成的 Orca-3 模型展示了显著的基准测试提升，超越了 LLAMA-8B-instruct 和 GPT-3.5-turbo 等模型。

 

**提到的链接**：[Neural Metamorphosis](https://arxiv.org/abs/2410.11878)：本文介绍了一种名为 Neural Metamorphosis (NeuMeta) 的新学习范式，旨在构建自变形神经网络。与为不同架构构建独立模型相反……

 

---

### **Nous Research AI ▷ #**[**reasoning-tasks**](https://discord.com/channels/1053877538025386074/1264666760972472481/1308455137915834469) (2 messages):

> - `LLaVA-o1`
> - `Vision-Language Models`
> - `Reasoning capabilities`
> - `Visual question-answering`
> - `Inference-time scaling`

- **LLaVA-o1 在 VLMs 中引入了结构化推理**：该论文介绍了 **LLaVA-o1**，这是一种新型的 **Vision-Language Model**，旨在实现**自主多阶段推理**，从而提高在复杂视觉问答任务中的性能。它涵盖了摘要、视觉解释、逻辑推理和结论生成等阶段。
  
  - 作者声称，通过开发一个包含来自不同 VQA 来源样本的综合数据集 **LLaVA-o1-100k**，在推理密集型任务的精度上取得了显著提升。
- **与现有推理模型的比较**：在讨论中，一位成员询问 LLaVA-o1 的方法是否与现有推理模型中使用的技术相似。这体现了对 AI 中不同推理方法进行交叉对比分析的兴趣。

 

**提到的链接**：[LLaVA-o1: Let Vision Language Models Reason Step-by-Step](https://arxiv.org/abs/2411.10440?utm_source=tldrai)：正如 OpenAI 的 o1 等模型所示，大型语言模型在推理能力方面已经取得了实质性进展，特别是通过 **inference-time scaling**。然而，目前...

 

---

### **GPU MODE ▷ #**[**general**](https://discord.com/channels/1189498204333543425/1189498205101109300/1308193411811246142) (2 messages):

> - `Colexicographical Order`
> - `Cutlass/Cute API Behavior`

- **理解坐标中的反字典序 (Colexicographical Order)**：一位成员讨论了他们在坐标映射背景下对 **colexicographical order** 的困惑，指出对于像 **(A, (B, C))** 这样的坐标，预期的迭代顺序应该是内层 **A**、**B**，外层 **C**。
  
  - 他们注意到这与 **Cutlass/Cute** 的 API 行为存在差异，后者似乎遵循不同的提取顺序，从而引发了进一步的疑问。
- **Cutlass/Cute API 混淆**：另一位成员指出了在 **Cutlass/Cute** 中观察到的迭代顺序与其 API 描述之间的一致性问题，特别是在 `get<I0,I1,...,IN>(x)` 方法上。
  
  - 他们强调 API 中的提取方法暗示了一种由内而外的迭代序列，这与他们对 **colexicographical process** 的理解相矛盾。

 

**提到的链接**：[cutlass/media/docs/cute/01_layout.md at main · NVIDIA/cutlass](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute/01_layout.md)：用于线性代数子程序的 **CUDA** 模板。通过在 **GitHub** 上创建账号为 NVIDIA/cutlass 的开发做出贡献。

 

---

### **GPU MODE ▷ #**[**triton**](https://discord.com/channels/1189498204333543425/1189607595451895918/1308173641233731668) (5 messages):

> - `Triton spelling`
> - `Triton CPU backend`
> - `GitHub Pull Request`

- **关于 Triton 拼写的笑谈**：一位成员幽默地评论道：*“几乎拼不对 triton”*，引起了聊天中其他人的笑声。
  
  - 这种轻松的交流突显了社区的同志情谊和共同经历。
- **引入 Triton CPU 后端**：一位成员分享了一个 [GitHub Pull Request](https://github.com/pytorch/pytorch/pull/133408)，旨在将 **Triton CPU** 作为 **Inductor** 后端添加到 **PyTorch** 中。
  
  - 此次集成的目标是使用 **Inductor 生成的 kernel** 来对新的 **Triton CPU** 后端进行压力测试。

 

**提到的链接**：[Add Triton CPU as an Inductor backend by int3 · Pull Request #133408 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/133408)：来自 ghstack 的堆栈（最早的在底部）：-> Add Triton CPU as an Inductor backend #133408。目标是使用 Inductor 生成的 kernel 来对新的 Triton CPU 后端进行压力测试。抄送 @XilunWu @H...

 

---

### **GPU MODE ▷ #**[**torch**](https://discord.com/channels/1189498204333543425/1189607750876008468/1308165148724428912) (17 条消息🔥):

> - `DCP 保存机制`
> - `FSDP 内存分配`
> - `State Dict 分析`
> - `Transformer Block 自动包装策略`
> - `FSDP 改进的未来`

- **DCP 保存功能见解**：一位成员分享了一个基于 [DCP 教程](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html) 的简单脚本，以说明他们对 `dcp.save` 的使用。
  
  - 他们强调，由于张量被复制到磁盘，Checkpoint 过程涉及 CPU 内存分配，这在分布式训练期间可能会带来挑战。
- **FSDP 的设备内存分配**：讨论澄清了保存期间看到的分配实际上来自 `CUDACachingAllocator`，并且发生在设备内存（Device Memory）上，而非 CPU 内存。
  
  - 成员们对尽管有预期但仍出现的大量内存分配表示困惑，并将其归因于参数“unflattening”（去扁平化）所需的新分配。
- **State Dict 内存使用**：探讨了 `get_state_dict` 函数在内存使用中的参与情况，观察到它在 all-gather 过程中需要为扁平化参数进行分配。
  
  - 一位成员指出，这一过程的效率可能取决于所使用的自定义自动包装策略（auto-wrap policy）。
- **Transformer Block 自动包装审查**：讨论揭示了 FSDP 中的 `FlatParameter` 概念是指每个 Transformer Block 的原子通信单元，这会影响使用 `get_state_dict` 时保留的内存。
  
  - 在 Block 级别捕获模型权重和优化器状态，意味着需要理解 Transformer Block 级别的内存分配复杂性。
- **FSDP 的未来增强**：成员们确认，即将推出的 FSDP 版本将致力于改进 Sharding 技术，使其不需要对参数进行 all-gathering，从而减少内存分配。
  
  - 预计发布时间定于今年年底或明年年初，有望在内存管理方面取得重大进展。

**提到的链接**：

- [Getting Started with Distributed Checkpoint (DCP) — PyTorch Tutorials 2.5.0+cu124 documentation](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)：未找到描述
- [pytorch/torch/distributed/checkpoint/filesystem.py at e80b1b2870ad568aebdbb7f5205f6665f843e0ea · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/e80b1b2870ad568aebdbb7f5205f6665f843e0ea/torch/distributed/checkpoint/filesystem.py#L169)：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能 - pytorch/pytorch
- [Rethinking PyTorch Fully Sharded Data Parallel (FSDP) from First Principles](https://dev-discuss.pytorch.org/t/rethinking-pytorch-fully-sharded-data-parallel-fsdp-from-first-principles/1019#flatparameter-4)：鉴于一些兴趣，我正在分享一份关于 PyTorch Fully Sharded Data Parallel (FSDP) 设计的笔记（最初为内部编写）。这涵盖了大部分内容但并非全部（例如，它排除了 autograd 和 CUDA 缓存...）

---

### **GPU MODE ▷ #**[**beginner**](https://discord.com/channels/1189498204333543425/1191300313928433664/1308246678612213840) (8 条消息🔥):

> - `重写 Aten 内核`
> - `算子融合 (Kernel fusion) 的好处`
> - `Torch.compile 的局限性`

- **使用 Triton 和 CUDA 重写 Aten 内核**：当使用 Triton 或 CUDA 内核时，我们实际上是在替换 **Torch Aten** 内核，后者为了处理各种用例而设计得更为通用。
  
  - 这允许针对特定问题约束进行定制化优化。
- **算子融合 (Kernel Fusion) 的好处**：算子融合通过合并多个操作来最小化内存读写，从而提高性能，这是 **Dynamo** 和 **TorchInductor** 等框架的核心优势。
  
  - 例如，融合线性层（Linear）和激活函数（Activation）操作可以防止中间激活值被写入主内存，从而显著加快处理速度。
- **Torch.compile 的自动融合**：在简单情况下，**torch.compile** 可以自动处理算子融合，从而可能减少手动编写融合内核的需求。
  
  - 然而，局限性依然存在，因为 torch.compile 由于其硬编码性质和它所识别的特定代码结构，无法优化所有模式。

---

### **GPU MODE ▷ #**[**youtube-recordings**](https://discord.com/channels/1189498204333543425/1198769713635917846/1308519601155473550) (1 messages):

> - `FP8 and FP32 MMA alignment`
> - `Warp shuffle performance issues`
> - `Static layout permutation`

- **FP32 MMA 中的 FP8 不匹配**：讨论围绕 FP32 MMA 中 FP8 的输出线程片段 (thread fragment) 所有权如何与预期输入不匹配展开，特别是在 [第 8 页，图 3 和 图 4](https://arxiv.org/pdf/2407.08608) 中有所说明。
  
  - 在同一位置需要来自不同线程所有权的元素突显了处理过程中的数据处理问题。
- **Warp Shuffle 对片段对齐并不理想**：通过 warp shuffle 修复线程所有权对齐的朴素方法，如果应用于片段中的每个元素，可能会降低性能。
  
  - 这一担忧强调了在管理线程所有权的同时，需要高效的数据处理以确保最佳运行。
- **使用静态布局排列提高效率**：为了防止性能下降，采用了共享内存 (shared memory) 张量的静态布局排列 (static layout permutation)，以在不产生性能开销的情况下对齐线程所有权。
  
  - 这种方法允许更合适的数据排列，以匹配后续操作的预期输入要求。

 

---

### **GPU MODE ▷ #**[**off-topic**](https://discord.com/channels/1189498204333543425/1215328286503075953/1308359019198545984) (8 messages🔥):

> - `Travel Tips`
> - `Flight Searching Tools`
> - `OpenAI o1 Technical Discussion`
> - `YouTube Content`
> - `Discount Travel Strategies`

- **OpenAI o1 技术教程视频**：查看标题为 ["Speculations on Test-Time Scaling (o1)"](https://www.youtube.com/watch?v=6PEJ96k1kiw) 的 YouTube 视频，了解 OpenAI o1 背后的技术背景教程，幻灯片可在 [GitHub](https://github.com/srush/awesome-o1) 上获得。这次演讲是与 Daniel Ritter 共同编写的，深入探讨了 LLM 中“大 (large)”的含义。
  
  - 该视频承诺以简洁的形式加深对 OpenAI 创新的理解。
- **关于更便宜机票的建议**：一位成员分享了寻找便宜航班的有效技巧，推荐了 [Google Flights](https://www.google.com/travel/flights) 和 [Skiplagged](https://skiplagged.com/) 等工具。关键策略包括设置航班提醒、使用 VPN 避免价格上涨，以及考虑使用旅行奖励信用卡。
  
  - 对话强调，使用奖励卡在预订航班时可以显著省钱，并辅以个人案例说明。

 

**提到的链接**：[Speculations on Test-Time Scaling (o1)](https://www.youtube.com/watch?v=6PEJ96k1kiw)：关于 OpenAI o1 技术背景的教程。与 Daniel Ritter 共同编写的演讲。幻灯片：[https://github.com/srush/awesome-o1Talk](https://github.com/srush/awesome-o1Talk)：LLM 中的“大 (large)”是...

 

---

### **GPU MODE ▷ #**[**liger-kernel**](https://discord.com/channels/1189498204333543425/1275130785933951039/1308415406897365022) (8 messages🔥):

> - `Strange Model Outputs`
> - `Liger Kernel Distillation Loss`
> - `Kaggle Collaborations`

- **Qwen2.5 模型的奇怪输出**：一位用户报告说，在 Kaggle 上成功安装 **Liger Kernel** 后，使用 **Qwen2.5 模型** 时出现了奇怪的结果，产生了无意义的输出。
  
  - 相比之下，切换到 **AutoModelForCausalLM** 则产生了一个连贯的解方程分步解释，突显了初始模型的潜在问题。
- **Liger Kernel 功能实现请求**：一位用户指出关于在 **Liger Kernel** 中实现新的**蒸馏损失函数 (distillation loss functions)** 的问题，引发了对额外功能的兴趣。
  
  - [GitHub 上的 issue](https://github.com/linkedin/Liger-Kernel/issues/371) 中提供了详细信息，概述了支持各种对齐和蒸馏层的动机。
- **Liger Kernel 的 Kaggle 协作**：一位用户表示有兴趣分享 Kaggle notebook 以帮助排查 **Liger Kernel** 的问题，并提到了在竞赛中使其有效运行所面临的挑战。
  
  - 他们正在寻找 Kaggle 账号以方便分享，这表明了社区在努力解决共同问题。

 

**提到的链接**：[[RFC] Liger FlexChunkLoss: Alignment and Distillation loss · Issue #371 · linkedin/Liger-Kernel](https://github.com/linkedin/Liger-Kernel/issues/371)：🚀 功能、动机和构想。我们希望支持各种对齐和蒸馏损失函数。参考关于 ORPO 的 PR：#362 进度：Alignment ORPO #362 CPO #382 DPO #378 SimPO #386 IRPO .....

 

---

### **GPU MODE ▷ #**[**self-promotion**](https://discord.com/channels/1189498204333543425/1288557096404516945/1308469330161438850) (1 messages):

> - `NVIDIA Virtual Connect with Experts`
> - `CUDA Core Compute Libraries`

- **加入 NVIDIA CUDA 库专家小组**：不要错过 **2024年11月22日，星期五** **太平洋时间上午10点** 举行的 **NVIDIA Virtual Connect with Experts** 活动，本次活动将聚焦于 **CUDA Core Compute Libraries**。
  
  - 专家小组将涵盖包括 **Thrust**、**CUB** 和 **libcudacxx** 在内的主题，更多详情请访问其 [GitHub 页面](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts)。
- **宣传即将举行的活动**：鼓励参与者通过 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:share:7263302292756410369/) 和 [Facebook](https://www.facebook.com/photo.php?fbid=1118567573609376&set=a.747927134006757&type=3) 等社交媒体平台与朋友和同事分享活动详情。
  
  - 公告强调了社区参与，并邀请所有对 **CUDA** 开发感兴趣的人加入。
- **在 X 上与 CUDA 开发者交流**：X 上的一篇帖子分享到，参与者可以在 **11月22日** **太平洋时间上午10点至11点30分** 的虚拟活动期间，直接与 **CUDA 开发者** 小组取得联系。
  
  - 该帖子提供了信息链接，并邀请观众在[此处](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts?ncid=so-twit-910119)了解更多关于活动的信息。

**提及的链接**：

- [accelerated-computing-hub/connect-with-experts at main · NVIDIA/accelerated-computing-hub](https://github.com/NVIDIA/accelerated-computing-hub/tree/main/connect-with-experts): NVIDIA 策划的与通用 GPU 编程相关的教育资源合集。 - NVIDIA/accelerated-computing-hub
- [no title found](https://www.facebook.com/photo.php?fbid=1118567573609376&set=a.747927134006757&type=3): 未找到描述
- [来自 NVIDIA HPC Developer (@NVIDIAHPCDev) 的推文](https://x.com/NVIDIAHPCDev/status/1857521459725299810): 直接与 NVIDIA 的 #CUDA 开发者小组联系。👀 我们将讨论 CUDA Core Compute Libraries，如 Thrust、CUB 和 libcudacxx。加入我们的虚拟 CUDA 活动 📆 11月22日 1...

---

### **GPU MODE ▷ #**[**🍿**](https://discord.com/channels/1189498204333543425/1298372518293274644/1308173453760790588) (16 messages🔥):

> - `Scheduler Development`
> - `Modal Integration`
> - `Remote Authentication`

- **调度器实现成功**：一名成员开发了一个调度器并进行了快速修复，解决了 Modal 尝试运行机器人代码而不是用户提交文件的问题，目前正在推送新分支。
  
  - 据报告，该调度器是他们目前最快的一个，展示了性能的提升。
- **关于计算资源的讨论**：一名成员询问了如何在 Modal 中使用自己的计算资源，得到的解释是 Modal 仅为其用户提供计算资源，未来可能针对大型企业开放。
  
  - 另一名成员提议为更好的 GPU 添加 5000 美元的 credits 额度，强调支持升级。
- **Modal 的远程身份验证**：一名成员询问了如何从 Heroku 等远程机器验证 Modal 身份而无需浏览器验证，寻求使用环境变量的方法。
  
  - 建议复制 `.modal.toml` 文件或设置 `MODAL_TOKEN_ID` 和 `MODAL_TOKEN_SECRET` 等环境变量，该方案已成功实施。
- **Modal 的 CLI Token 设置**：另一名成员指出通过 CLI 设置 token 的命令为 `modal token set`，提供了另一种身份验证方法。
  
  - 该方法为从各种环境集成 Modal 的用户增加了身份验证过程的灵活性。

 

**提及的链接**：[modal branch by msaroufim · Pull Request #25 · gpu-mode/discord-cluster-manager](https://github.com/gpu-mode/discord-cluster-manager/pull/25): 仍然有一个恼人的 bug，即 Modal 尝试运行机器人代码本身而不是 train.py。不过日志语句显示文件名和内容是正确的，而且我知道一个玩具示例...

 

---

### **GPU MODE ▷ #**[**thunderkittens**](https://discord.com/channels/1189498204333543425/1300872762163728550/1308176003289321533) (2 条消息):

> - `Register Allocation`
> - `Spill Prevention Strategies`
> - `Nsight Compute Profiling`

- **Register Allocation 中的 Spills 是有害的**：就性能而言，*Spills 是非常有害的*，强调了为消费者有效地增加 **registers** 的重要性。
  
  - 一位成员指出，在 kernel 内部进行 **register allocation** 时需要谨慎，以防止性能下降。
- **Register Allocation 策略**：一位成员询问了关于谨慎进行 **register allocation** 的策略，并详细介绍了他们定义单个 register tile 并重用它以减少 spills 的经验。
  
  - 他们报告说，只有在添加额外的 **WGMMAs** 后才会出现 spills，这表明必须在资源利用率和性能之间取得平衡。
- **利用 Nsight Compute 进行 Profiling**：讨论了 **Nsight Compute profile** 是否能为优化 register allocation 策略提供见解。
  
  - 成员们请求关于在 profile 中具体查看哪些内容的建议，以增强 **TK** 中的实现。

---

### **Interconnects (Nathan Lambert) ▷ #**[**news**](https://discord.com/channels/1179127597926469703/1179128538679488533/1308310759155040270) (7 条消息):

> - `Runner H Navigation Skills`
> - `Pixtral Paper Discussion`
> - `Runner H Performance Evaluation`
> - `Runner H Beta Release`
> - `Comparisons with Qwen`

- **Runner H 的导航技能非常出色**：@laurentsifre 强调 **Runner H** 的导航技能由其内部 **VLM** 驱动，该模型以其在 UI 元素定位方面的 state-of-the-art 能力而闻名，同时比竞争模型更小、更便宜。
  
  - 这突显了 Runner H 在实际应用中的技术优势。
- **来自 Pixtral 论文的见解**：@Sagar_Vaze 指出，[Pixtral 论文](https://arxiv.org/abs/2410.07073)中对相关点进行了详细讨论，特别是参考了 **第 4.2 和 4.3 节**以及附录 E。
  
  - 这些资源为研究背景下讨论的复杂性提供了宝贵的见解。
- **Runner H 在与对手的竞争中表现强劲**：一份研究更新详细介绍了 **Runner H 0.1 agents** 在 **WebVoyager benchmarks** 上相对于竞争对手的表现，表明其在真实场景评估中取得了成功。
  
  - 正如 [原始 WebVoyager 论文](https://arxiv.org/abs/2401.13919)中所述，这是通过自动评估方法实现的。
- **Runner H Beta 版本发布标志着新进展**：在 **Runner H** Beta 版本发布的公告中提到，这一进展正在突破停滞的 **scaling laws** 的限制，向 **artificial super intelligence (ASI)** 迈进。
  
  - 他们宣称，随着这次发布，他们不仅是在推出一款产品，而是**开启了 AI 的新篇章**。
- **Runner H 与 Qwen 的对比**：值得注意的是，**Runner H** 确实将其性能与 **Qwen** 进行了对比，反映了当前 AI 发展中的竞争格局。
  
  - 这种对比对于理解行业内 AI 模型的有效性至关重要。

**提到的链接**：

- [来自 H Company (@hcompany_ai) 的推文](https://x.com/hcompany_ai/status/1858907033921069449)：随着 Runner H Beta 版本的发布，我们正在突破停滞的 scaling laws 的限制，向 artificial super intelligence (ASI) 迈出一步。通过 Runner H，我们不仅仅是在介绍...
- [来自 Sagar Vaze (@Sagar_Vaze) 的推文](https://x.com/Sagar_Vaze/status/1858880536959148343)：@TheXeophon 我们在 Pixtral 论文中对这一点进行了详细讨论：https://arxiv.org/abs/2410.07073 参见第 4.2 和 4.3 节，特别是附录 E。
- [Runner H state-of-the-art 结果背后的技术](https://www.hcompany.ai/blog/a-research-update)：H 在网络上首次实现了自主 agents 的承诺。我们对 Runner H 与竞争对手在 WebVoyager benchmarks 上的评估证明，该技术在真实世界中处于排行榜顶端...
- [来自 Laurent Sifre (@laurentsifre) 的推文](https://x.com/laurentsifre/status/1858918590960775359)：Runner H 的导航技能由我们的内部 VLM 驱动，它在 UI 元素定位方面处于 state-of-the-art 水平，同时与其它基础模型相比，其体积更小、服务成本更低...

---

### **Interconnects (Nathan Lambert) ▷ #**[**random**](https://discord.com/channels/1179127597926469703/1183121795247779910/1308301424005742663) (52 messages🔥):

> - `Tulu 讨论`
> - `Google 员工视角`
> - `Grok 更新`
> - `Threelu 命名创意`
> - `时区挑战`

- **Tulu 之夜计划引发热议**：*Tulu tonight* 成了讨论的热门话题，多位成员就他们的计划和时区开起了玩笑。有人提到他们的“今晚”正好是别人的“明天”，展现了群组内轻松幽默的氛围。
  
  - 尽管对话很轻松，但一些人对深夜会议表示抗拒，这演变成了关于集体职业倦怠（burnout）的一个长期笑话。
- **对 Google 文化的复杂情感**：成员们分享了他们在 Google 的不同经历，强调虽然员工个人都很友好，但组织层面似乎存在问题。有人评论道，*the org is broken somehow*（这个组织在某种程度上出了问题），暗示了公司结构内部的脱节。
  
  - 其他人也表达了类似的观点，认为尽管拥有优秀的员工，Google 的整体运营仍面临挑战。
- **关注 Grok 更新**：一位成员对当晚缺少 Grok 更新感到遗憾，表示第二天不得不处理两个更新。这引发了对 Grok 持续开发和预期的关注。
  
  - 更新的缺席引发了调侃，表明虽然错过了更新，但群组成员总能找到让对话保持趣味的方法。
- **关于 Threelu 的命名建议**：关于是将项目命名为“Threelu”还是“Tulululu”等替代方案，展开了一场轻松的辩论。一位成员认为 *Tulu* 比 *Threelu* 看起来更漂亮，反映了讨论的幽默本质。
  
  - 命名讨论在聊天中引发了梗图和怀旧情结，包括过去针对上下文模型的 *Clong* 等笑话。
- **共鸣时区难题**：成员们交流了跨多个时区工作的困难，其中一人回忆了与 *+9 时区* 的人协作的经历。这突显了科技行业从业者面临的普遍挑战。
  
  - 分享此类经历时的情谊强调了尽管地理位置不同，社区成员之间仍有着紧密的联系。

**提到的链接**：

- [来自 undefined 的推文](https://x.com/JustinLin610)：未找到描述
- [来自 Binyuan Hui (@huybery) 的推文](https://x.com/huybery/status/1858732706743677372)：@KMatiDev1 两个都可以。我只是不想被忽视，比如在新模型发布时经常被排除在对比之外。
- [来自 Yaroslav (@512x512) 的推文](https://x.com/512x512/status/1858795479171297788)：抱歉，今晚没有 Grok 更新，这意味着明天会有两个。

---

### **Interconnects (Nathan Lambert) ▷ #**[**reads**](https://discord.com/channels/1179127597926469703/1214764639397617695/1308288583357956227) (3 messages):

> - `树搜索收益`
> - `Tree of Thoughts`
> - `Q* 算法`

- **关于树搜索收益的重要报告**：最近的一份 [报告](https://arxiv.org/abs/2411.11694) 讨论了树搜索（tree search）方法的显著进展，展示了该领域此前未曾观察到的收益。
  
  - 报告将这些改进归功于包括 **Jinhao Jiang** 和 **Zhipeng Chen** 在内的多元化作者团队的协作努力。
- **对 Tree of Thoughts 的反思**：一位成员回忆起“过去的好时光”，提到了 **Tree of Thoughts** 方法，并指出了其产生的积极影响。
  
  - 这激发了人们重新审视那些塑造了当前方法论的过往策略的兴趣。
- **对 Q* 算法的怀旧**：提到 **Q*** 算法引发了成员们对其在 AI 技术讨论中奠基性作用的怀旧之情。
  
  - 这突显了人们对历史算法的持续认可，因为它们为当前的持续发展做出了贡献。

**提到的链接**：[技术报告：通过奖励引导的树搜索增强 LLM 推理能力](https://arxiv.org/abs/2411.11694)：近期，test-time scaling 引起了研究界的极大关注，这主要归功于 OpenAI 发布的 o1 模型所取得的实质性进展。通过分配更多的计算资源……

---

### **Latent Space ▷ #**[**ai-general-chat**](https://discord.com/channels/822583790773862470/1075282825051385876/1308168517383749652) (44 messages🔥):

> - `Cerebras 推理性能`
> - `OpenAI 语音功能`
> - `Roboflow B 轮融资`
> - `Small Language Models`

- **Cerebras 的 Llama 3.1 推理速度**：Cerebras 声称能以 **969 tokens/s** 的速度提供 Llama 3.1 405B 推理，比中位数供应商基准快 **10 倍**以上。
  
  - 批评者认为，虽然 **Cerebras** 在 batch size 1 的评估中表现出色，但在较大 batch size 下性能会有所下降，建议对比时应考虑这些差异。
- **OpenAI 增强语音功能**：OpenAI 宣布在 [chatgpt.com](https://chatgpt.com) 上为**付费用户**推出语音功能更新，旨在让演示变得更轻松。
  
  - 此次更新允许用户通过演示学习发音，突显了其持续关注增强用户交互的趋势。
- **Roboflow 获得 4000 万美元 B 轮融资**：Roboflow 额外筹集了 **4000 万美元**，用于增强其在医疗、环境等多个领域的视觉 AI 应用开发者工具。
  
  - CEO Joseph Nelson 强调了他们的使命是赋能开发者有效部署视觉 AI，并强调了在数字世界中“视觉”的重要性。
- **关于小型语言模型的讨论**：社区讨论了小型语言模型 (**SLM**) 的定义，有建议认为 **1B 到 3B** 参数规模的模型属于小型。
  
  - 大家的共识是较大的模型不符合这一分类，并指出了基于在消费级硬件上运行能力的区分标准。

**提到的链接**：

- [OpenAI Developers (@OpenAIDevs) 的推文](https://x.com/openaidevs/status/1858609150999359559?s=46)：OpenAI o1-preview 和 o1-mini 现已支持 Streaming。🌊 https://platform.openai.com/docs/api-reference/streaming 并且我们已向所有付费使用层级的开发者开放了这些模型的访问权限...
- [Artificial Analysis (@ArtificialAnlys) 的推文](https://x.com/artificialanlys/status/1858594969927512476?s=46)：Cerebras 能够以 969 output tokens/s 的速度提供 Llama 3.1 405B，并宣布很快将提供公共推理端点 🏁 我们已独立基准测试了一个私有端点...
- [独家：视觉 AI 初创公司 Roboflow 筹集 4000 万美元 B 轮融资](https://fortune.com/2024/11/19/exclusive-roboflow-vision-ai-startup-raises-40-million-series-b/)：据 Fortune 独家获悉，Roboflow 已完成由 GV 领投的 4000 万美元 B 轮融资。
- [推理，快与慢](https://www.latent.space/p/inference-fast-and-slow)：当 System 1/System 2 的类比不再足够时：LLM 推理的 6 种类型。
- [Goddammit! GIF - Ryan Reynolds Goddammit Damn](https://tenor.com/view/ryan-reynolds-goddammit-damn-hitmans-bodyguard-hitmans-bodyguard-gifs-gif-8352668)：点击查看 GIF。
- [swyx (@swyx) 的推文](https://x.com/swyx/status/1679241722709311490)：我建议对 LLM 的权重分级采用 T 恤尺码标准：XXLLM: ~1T (GPT4, Claude2, PanGu) XLLM: 300~500B (PaLM, PaLM2) LLM: 20~100B (GPT3, Claude, UL2) ~~涌现区~~ MLM: 7~14B (T5, LLaMA, MPT) SL...
- [OpenAI (@OpenAI) 的推文](https://x.com/OpenAI/status/1858948388005572987)：另一个高级语音功能更新——现已在 http://chatgpt.com 桌面端向所有付费用户推出。这样你就可以轻松学习如何用语音完成整个演示...
- [Joseph Nelson (@josephofiowa) 的推文](https://x.com/josephofiowa/status/1858977542629454143?s=46)：Roboflow 额外筹集了 4000 万美元以继续推进计算机视觉的发展，视觉能力是体验世界的基础，但软件几乎没有利用好视觉。我们正在加大投入...
- [我是 Groq 的忠实付费客户，但他们在 Cerebras 面前没有竞争力... | Hacker News](https://news.ycombinator.com/item?id=42179927)：未找到描述。
- [Tim Dettmers (@Tim_Dettmers) 的推文](https://x.com/tim_dettmers/status/1858977311569440955?s=46)：澄清一下这个基准测试。这是一个不对等的比较。- Cerebras 在 batch size 1 时很快，但在 batch size n 时很慢。- GPU 在 batch size 1 时较慢，但在 batch size n 时很快。我...

---

### **LlamaIndex ▷ #**[**blog**](https://discord.com/channels/1059199217496772688/1187460979064324127/1308217967032668201) (2 messages):

> - `Local Agentic RAG Application`
> - `LlamaIndex Workflows`
> - `Llama-Deploy`
> - `LlamaIndex Azure Integration`
> - `Microsoft Ignite`

- **构建你自己的本地 RAG 应用**：在 **11月27日** 加入我们的朋友 [AIMakerspace](https://twitter.com/AIMakerspace)，学习如何使用开源 LLM 设置用于报告生成的“私有化部署” LLM 应用栈，内容涵盖 **LlamaIndex Workflows** 和 **Llama-Deploy**。
  
  - 该活动承诺提供实战培训，并分享构建稳健本地应用的见解。
- **LlamaIndex Azure 解决方案在 Ignite 亮相**：我们很高兴宣布本周在 [#MSIgnite](https://twitter.com/hashtag/MSIgnite) 上推出了集成 **LlamaIndex** 与 **Azure** 的端到端解决方案，其特点是包含 Azure Open AI、Azure AI Embeddings 和 Azure AI Search。
  
  - 如果你正在参加 Microsoft Ignite，请联系 **@seldo** 了解有关这一强大集成的更多细节。

---

### **LlamaIndex ▷ #**[**general**](https://discord.com/channels/1059199217496772688/1059201661417037995/1308189840365322281) (34 messages🔥):

> - `Document Processing in S3`
> - `RAG App Functionality`
> - `SQLAutoVectorQueryEngine Citations`
> - `Chat History in RAG`
> - `Iterating on Prompts`

- **S3 文档处理问题**：一位用户报告了在使用 S3 读取器处理 PDF 文档时遇到困难，收到了关于文件路径的错误。
  
  - 另一位用户建议修改 PDF 读取器并使用自定义的 `PDFParser` 来克服兼容性问题。
- **实现 RAG 功能的挑战**：一位用户询问如何在使用 Milvus 和 Ollama 的 LLM 的 RAG 应用中集成聊天历史功能，并提到了他们自定义的索引方法。
  
  - 他们被引导至可以根据其现有工具进行修改的 Chat Engine 功能。
- **SQLAutoVectorQueryEngine 的引用功能**：一位用户询问如何使用 SQLAutoVectorQueryEngine 获取行内引用，以及它是否可以与 CitationQueryEngine 集成。
  
  - 鉴于实现引用逻辑的简单性，建议他们将引用的 Workflow 分开。
- **使用检索质量指标测试 RAG**：一位参与者对缺乏用于测试其 RAG 系统检索指标质量的 Ground Truth 数据表示担忧。
  
  - 他们向社区寻求有效应对这一挑战的方法论或教程。
- **高效迭代 Prompt**：一位用户询问在将 Prompt 集成到服务中后如何高效迭代的建议，并表示目前使用 Jupyter notebooks 感觉效率低下。
  
  - 他们正在寻求更好的方法或工具来改进这方面的 Workflow。

**提到的链接**：

- [no title found](https://www.llamaindex.ai/bl): 未找到描述
- [Add neo4j generic node label (#15191) · run-llama/llama_index@77bd4c3](https://github.com/run-llama/llama_index/commit/77bd4c3fc6db725ffe04dbf778b1d7a3f9e63baa): 未找到描述
- [Workflows - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/workflow/#workflows)): 未找到描述
- [Pandas - LlamaIndex](https://docs.llamaindex.ai/en/stable/api_reference/query_engine/pandas/): 未找到描述
- [Milvus Vector Store - LlamaIndex](https://docs.llamaindex.ai/en/stable/examples/vector_stores/MilvusIndexDemo/): 未找到描述
- [Module Guides - LlamaIndex](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/modules/): 未找到描述

---

### **LlamaIndex ▷ #**[**ai-discussion**](https://discord.com/channels/1059199217496772688/1100478495295017063/1308352164938383441) (5 messages):

> - `RAG systems`
> - `Agent invocation strategies`
> - `Preventing spam in channels`

- **理解 RAG 系统响应机制**：一位用户询问 RAG 系统如何决定是再次搜索向量 Embedding 还是优化之前的答案，并以在表格中显示叶子列表为例。
  
  - 该问题强调了需要一种机制来区分新查询和对现有响应的优化。
- **利用工具调用的 Agent**：另一位成员澄清说，Agent 会根据聊天历史选择调用工具（如向量数据库搜索），并利用来自各种 LLM 的训练协议。
  
  - 一些 LLM 采用 **ReAct**、**chain-of-code** 和 **chain-of-abstraction** 等 Prompt 策略来促进工具调用。
- **防止频道垃圾信息的提醒**：一位成员警告不要在频道中发布垃圾信息，并敦促过多的提问可能会导致被封禁。
  
  - 这一提醒鼓励成员保持尊重且专注的对话，避免不必要的干扰。

---

### **tinygrad (George Hotz) ▷ #**[**general**](https://discord.com/channels/1068976834382925865/1068976834928193609/1308229528870453359) (25 条消息🔥):

> - `tinygrad 0.10.0 发布`
> - `不同架构上的测试失败`
> - `会议权限与笔记`
> - `Kernel Cache in Action 测试`
> - `ARM 上的插值测试`

- **tinygrad 0.10.0 发布**：团队宣布发布 [tinygrad 0.10.0](https://github.com/tinygrad/tinygrad/releases/tag/v0.10.0)，其中包含超过 1200 次 commit，更新重点在于最小化依赖。
  
  - 除了简洁性，tinygrad 现在同时支持推理和训练，并抱有未来构建硬件的愿景，近期已完成融资。
- **ARM 上的间歇性测试失败**：有用户报告称，虽然在 `x86_64-linux` 上更新运行顺畅，但在 `aarch64-linux` 上遇到了大量测试失败，特别是在测试期间出现了 `AttributeError`。
  
  - 经过调查，确认该问题在不同架构间均可复现，目前正在讨论可能的解决方案。
- **会议权限仅限 Red 成员**：关于更新会议笔记的讨论，George Hotz 授予了 @zibokapi 作为 'red' 成员的发布权限。
  
  - 该权限对于在会议录制频道添加笔记是必要的。
- **修复 test_kernel_cache_in_action 问题**：一名用户发现，在 `test_kernel_cache_in_action` 的某些操作前添加 `Tensor.manual_seed(123)` 解决了之前失败的测试。
  
  - 该修复经确认可使整个测试套件通过，仅剩 ARM 架构上的一个遗留问题。
- **插值测试失败诊断**：`aarch64-linux` 上剩余的失败追溯到了插值测试，特别是 `test_interpolate_bilinear`，并分享了相关引用以供进一步检查。
  
  - 提议集成 `x.realize()` 可能会增强现有的测试实现。

**提到的链接**：

- [tinygrad](https://pypi.org/project/tinygrad/)：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！ <3
- [test_kernel_cache_in_action: fix test by GaetanLepage · Pull Request #7792 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/pull/7792)：修复了 c100f3d 中引入的回归问题。self = &lt;tinygrad.engine.realize.CompiledRunner object at 0xfffedac82a80&gt; p = Program(name=&#39;E_\\x1b[34m4194304\\x1b[0m\\x1b[90m_\\x1b[0m\\x1b[...
- [Release tinygrad 0.10.0 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/releases/tag/v0.10.0)：一次重大的底层更新。自 0.9.2 以来超过 1200 次 commit。代码量 9937 行。发布亮点：使用 VIZ=1 展示重写过程，尝试 0 python 依赖！从 numpy ran 切换...
- [tinygrad/tinygrad/engine/realize.py at v0.10.0 · tinygrad/tinygrad](https://github.com/tinygrad/tinygrad/blob/v0.10.0/tinygrad/engine/realize.py#L143-L156>)：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！ ❤️ - tinygrad/tinygrad
- [add support and tests for nearest modes in interpolate, adapt uint8 b… · tinygrad/tinygrad@7de4eac](https://github.com/tinygrad/tinygrad/commit/7de4eac8f7c0db52550f1fad36904c20eafe60d4)：…ilinear 到 torch 实现 (#6308) \* 在 interpolate 中添加 `nearest` 模式。匹配已知存在 bug 的 pytorch `nearest`。相关 TestsOps。添加 `nearest-exact` 模式到 interpol...
- [default threefry (#6116) · tinygrad/tinygrad@c100f3d](https://github.com/tinygrad/tinygrad/commit/c100f3d40618ff7f19ded78eee89d8a0dc253135)：未找到描述

---

### **tinygrad (George Hotz) ▷ #**[**learn-tinygrad**](https://discord.com/channels/1068976834382925865/1070745817025106080/1308173384210972733) (4 条消息):

> - `DEBUG 输出`
> - `Jitted 函数行为`

- **理解 DEBUG=2 输出**：一名成员询问了设置 **DEBUG=2** 时的输出以及它是否在 **master** 分支上。另一名成员指出，当进程运行时，底部行会不断重复。
  
  - *“如果底部行持续滚动，说明它正在运行”* 表明尽管有重复，代码仍在正常运作。
- **关于 Jitted 函数的澄清**：讨论表明 **jitted function** 仅执行 GPU kernel，这意味着在其内部的 print 语句不会有可见输出。这让大家更清楚地了解了这些函数内部的调试机制。

 

---

### **Cohere ▷ #**[**discussions**](https://discord.com/channels/954421988141711382/954421988783444043/1308243193992712292) (13 条消息🔥):

> - `AI 模型中的 Tokenized Training`
> - `名称与符号识别`
> - `语言模型训练挑战`
> - `APIs 与语言响应设置`

- **Tokenized Training 对单词识别的影响**：一位成员指出，单词 **'strawberry'** 在训练期间被分词（tokenized），由于被拆分为更小的组件，影响了对其的识别。
  
  - 这一问题在 **GPT-4o** 和 **Google’s Gemma2 27B** 等模型中均有观察到，凸显了不同系统面临的相似挑战。
- **关于名称识别能力的讨论**：一位成员对 **Aya** 在名称和音节识别方面的能力表示好奇，暗示其在该领域可能具有优势。
  
  - 讨论指出，由于 Tokenization 的原因，模型可能难以准确识别某些单词。
- **频道内的早安问候**：成员们交换了早安问候，多位用户对 **prabhakar171** 的 'gm' 进行了回应。
  
  - 这种轻松的互动反映了讨论区用户之间的情谊。
- **API 请求中的语言设置**：一位用户询问如何配置 **command-r model** 以使用**保加利亚语**响应，同时避免与**俄语**术语混淆。
  
  - 他们提到使用 API 请求生成器进行此类自定义，表明需要更清晰的语言区分。

 

---

### **Cohere ▷ #**[**announcements**](https://discord.com/channels/954421988141711382/996880279224451154/1308467341390123080) (1 条消息):

> - `Cohere 工具的 Beta 计划`
> - `研究与写作工具`
> - `用户反馈的重要性`

- **加入我们 Beta 计划的最后召集！**：[Cohere 研究原型 Beta 计划](https://forms.gle/Teis9VwM6eZP6nxVA)的报名将于今晚 **ET 时间午夜**截止，该计划提供了一款专为研究和写作任务设计的新工具的早期访问权限。
  
  - 鼓励参与者提供**详细反馈**，以帮助塑造工具的功能，重点关注复杂报告和摘要等任务。
- **文本交付成果创作者的机会**：该 Beta 计划面向经常创建报告、摘要或博客文章等**文本交付成果（text-based deliverables）**的人群。
  
  - 测试者将有机会影响工具的开发，确保其能有效辅助实际任务。

 

**提到的链接**：[研究原型 - 早期 Beta 报名表](https://forms.gle/Teis9VwM6eZP6nxVA)：感谢您有兴趣参与我们研究原型的 Beta 测试阶段——这是一款旨在帮助用户处理研究和写作任务的工具，例如：创建复杂的报告、执行...

 

---

### **Cohere ▷ #**[**questions**](https://discord.com/channels/954421988141711382/1168411509542637578/1308419069648568370) (6 条消息):

> - `API 中的速率限制 (Rate limits)`
> - `模型的语言设置`

- **关于 API 速率限制的说明**：一位成员询问 [文档](https://docs.cohere.com/v2/docs/rate-limits) 中设定的速率限制是按 **API key** 强制执行还是按 **account** 执行。
  
  - 讨论旨在明确速率限制的机制，但目前尚未得出明确答案。
- **保加利亚语模型配置**：一位用户询问如何设置 **preamble** 或 command-r 模型的变体，以确保其专门以**保加利亚语**响应并避免与**俄语**单词混淆。
  
  - 他们提到为此目的使用了 **API request builder**，表明模型需要更清晰的语言处理能力。

 

---

### **OpenInterpreter ▷ #**[**general**](https://discord.com/channels/1146610656779440188/1147665339266650133/1308436529303257158) (8 条消息🔥):

> - `开发分支状态`
> - `Open Interpreter 技能生成`
> - `UI 简化`
> - `Claude 模型问题`

- **开发分支似乎处于 WIP 状态**：一位成员注意到开发分支（或 beta 版）似乎处于 [WIP（在研）状态](https://link.to.commit)，因为 `interpreter --version` 显示为 **1.0.0**，这表明 UI 和功能可能出现了回归（regression）。
  
  - 另一位成员提出帮助修复任何问题，并指出最后的提交版本号为 **9d251648**。
- **Open Interpreter 技能生成需要帮助**：一位成员请求在 **Open Interpreter** 中生成技能方面的协助，提到预期的文件夹是空的，并寻求如何继续的建议。
  
  - 有建议参考 [GitHub](https://github.com/openinterpreter/01) 上关于教学模型的说明，因为未来的版本也将利用这一功能。
- **UI 简化引发褒贬不一的反应**：关于最近的 **UI 简化** 展开了讨论，一位成员表示更喜欢旧设计，并指出他们已经习惯了旧版。
  
  - 开发者确认了关于 UI 更改的反馈，并询问用户是否更喜欢旧版本。
- **Claude 模型问题受到关注**：一位成员报告了 **Claude** 模型崩溃的问题，指出切换模型解决了该问题，并对 **Anthropic** 服务表示担忧。
  
  - 这引发了关于这些问题是否在不同版本中反复出现的询问。

---

### **OpenInterpreter ▷ #**[**ai-content**](https://discord.com/channels/1146610656779440188/1149229778138824765/1308469168647049227) (12 条消息🔥):

> - `10 个 AI 工具播客`
> - `Tool Use 播客`
> - `服务器礼仪`
> - `社区参与`

- **Ray Fernando 分享 AI 工具见解**：在最近的一期 [YouTube 视频](https://www.youtube.com/watch?v=9DAZP1MdcbQ)中，Ray Fernando 讨论了他如何利用 **AI 工具** 来增强他的构建流程，重点介绍了 **10 个有助于快速构建的 AI 工具**。
  
  - 这段对话在名为“10 个真正产生效果的 AI 工具”的视频中有详细介绍，为对工具使用感兴趣的开发者提供了宝贵的见解。
- **关于 Tool Use 播客的讨论**：成员们表达了对 **Tool Use 播客** 的喜爱，强调了引人入胜的内容以及联合主持人 Mike 和 Ty，同时讨论了服务器中提及（mentions）的使用。
  
  - 会议指出，@everyone 标签的使用应保留给重要的通信，以尊重社区成员的时间和注意力。
- **鼓励职业转型**：一位成员分享了在 **Tool Use 播客** 第 5 集中了解到 Mike 的编程历程是如何*令人深受鼓舞*的，因为他也是在较晚的年龄经历了职业转型。
  
  - 这个个人故事引起了其他人的共鸣，说明了播客对考虑重大生活转变的个人的影响。
- **重视社区参与**：社区成员强调了按自己的节奏参与内容的重要性，对分享的见解表示感谢，并鼓励保持尊重的沟通。
  
  - 大家对平衡社区参与与个人节奏及注意力管理的需求达成了共识。
- **紧跟新内容**：一位成员表达了对 **Tool Use 播客** 的喜爱，表示在繁忙的日程中，提醒功能对于收听最新剧集非常有帮助。
  
  - 这种情绪反映了社区内对帮助他人了解相关内容的协调者的普遍赞赏。

**提到的链接**：

- [10 AI Tools That Actually Deliver Results (ft. Ray Fernando) - Ep 14](https://www.youtube.com/watch?v=9DAZP1MdcbQ)：加入我们，与从 Apple 工程师转行为直播主的 Ray Fernando 进行深入对话，共同探索 AI 驱动的工具和转录的世界...
- [GitHub - gregpr07/browser-use: Make websites accessible for AI agents](https://github.com/gregpr07/browser-use)：让网站可供 AI Agent 访问。通过在 GitHub 上创建账户为 gregpr07/browser-use 的开发做出贡献。

---

### **DSPy ▷ #**[**show-and-tell**](https://discord.com/channels/1161519468141355160/1202371242519441499/1308178069692874754) (1 条消息):

> - `DSPy VLM tutorial`
> - `Attribute extraction from images`

- **DSPy 引入 VLM 支持**：现在已提供新的 **DSPy tutorial**，重点介绍了最近在 beta 版中新增的 **VLM support**，用于从图像中提取属性。
  
  - 该教程使用网站截图来演示如何有效利用此功能，详情见此 [Twitter thread](https://x.com/karthikkalyan90/status/1858609018228355414)。
- **属性提取与图像处理变得简单**：在教程中，展示了从图像（特别是 **网站截图**）中提取有用 **attributes** 的实际案例。
  
  - 正如作者在其信息丰富的推文串中所解释的，这标志着 DSPy 能力的重大增强。

 

**提到的链接**：[来自 Karthik Kalyanaraman (@karthikkalyan90) 的推文](https://x.com/karthikkalyan90/status/1858609018228355414)：🧵DSPy 最近在 beta 版中增加了对 VLM 的支持。关于使用 DSPy 从图像中提取属性的简短线程。在这个例子中，我们将看到如何从网站截图中提取有用的属性...

 

---

### **DSPy ▷ #**[**general**](https://discord.com/channels/1161519468141355160/1161519469319946286/1308209610699833345) (17 条消息🔥):

> - `DSPy integration with non-Python`
> - `Cost reduction with DSPy`
> - `Challenges with long-context prompts`
> - `Testing DSPy code for React agents`
> - `DSPy assertions compatibility with MIRPOv2`

- **DSPy 输出在非 Python 后端遇到困难**：一位成员在将 DSPy 编译后的 JSON 输出与 **Go** 集成时遇到了准确率下降的问题，并对复制 Prompt 处理逻辑表示担忧。
  
  - 另一位成员建议使用 **inspect_history** 方法来构建针对特定应用定制的模板。
- **使用 DSPy 的成本节约策略**：成员们讨论了 DSPy 如何帮助降低 Prompt 成本，特别是通过 Prompt 优化以及可能采用小语言模型作为代理。
  
  - 然而，人们对长上下文限制表示担忧，并认为需要上下文剪枝和 RAG 实现等策略来提高效率。
- **长上下文 Prompt 优化困境**：讨论强调了在长文档解析中，大上下文 few-shot 示例的低效性，并批评了对模型在超长输入下保持连贯性的依赖。
  
  - 一位成员提出将处理过程分解为更小的步骤，并最大化每个 token 的信息量，作为缓解上下文相关问题的潜在解决方案。
- **寻求 React Agent 的 DSPy 示例**：一位成员正在寻找 DSPy 代码示例，以测试一个旨在提高 React Agent 性能的工具包装器，特别是针对那些已经出现问题的 Agent。
  
  - 他们澄清说，他们正在寻找那些运行不佳的 Agent 示例，特别是在处理意外输出和多轮对话方面。
- **MIRPOv2 兼容性查询**：一位成员询问了即将发布的 2.5 版本中 DSPy assertions 与 MIRPOv2 的兼容性，并提到了之前的兼容性问题。
  
  - 该查询表明人们对这些功能在框架内的演进和集成持续关注。

 

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**general**](https://discord.com/channels/1104757954588196865/1104757955204743201/1308343801395548210) (14 messages🔥):

> - `Mistral Large / Pixtral models`
> - `MI300X training`
> - `bitsandbytes integration`
> - `Web3 platform job openings`

- **探索 Mistral Large / Pixtral 模型**：成员们讨论了尝试最新 **Mistral Large** 和 **Pixtral** 模型的兴趣，并向有经验的人寻求见解。
  
  - 对话反映了社区对这些 AI 模型的持续好奇和实验精神。
- **MI300X 训练已投入运行**：使用 **MI300X** 的训练已成功运行了相当长一段时间，据报道多项更改已提交至上游（upstreamed）。
  
  - 一位成员强调了上游贡献的重要性，以确保训练过程中的一致性能和可靠性。
- **Bitsandbytes 需要导入调整**：有人担心即使在训练期间没有主动使用 **bitsandbytes**，也必须导入它，建议将其设为可选。
  
  - 一位成员提议使用上下文管理器（context manager）来抑制导入错误，从而增强代码库的灵活性。
- **bitsandbytes 的 ROCm 兼容性**：一位成员指出，在 **bitsandbytes** 主仓库的一个分支中已经提供了 **ROCm 支持**。
  
  - 另一位成员提到了一个提供 **bnb 支持** 的 fork，有助于在 ROCm 兼容平台上进行集成。
- **Web3 平台寻求团队成员**：一个 **Web3 平台** 正在招聘多个职位，包括开发人员、版主和 Beta 测试人员，薪资具有竞争力。
  
  - 该团队营造了友好的环境，且无需经验，对社区中的潜在候选人很有吸引力。

**提到的链接**：

- [在 Runpod MI300X 上设置 axolotl+FA2+BnB+liger-kernel 的 Bash 脚本](https://gist.github.com/DocShotgun/c67c1220a82506133e7b1f4886260ab6)：用于在 Runpod MI300X 上设置 axolotl+FA2+BnB+liger-kernel 的 Bash 脚本 - axolotl_ROCm_setup_v2.sh
- [GitHub - arlo-phoenix/bitsandbytes-rocm-5.6: 适用于 PyTorch ROCm 兼容的 8-bit CUDA 函数](https://github.com/arlo-phoenix/bitsandbytes-rocm-5.6)：适用于 PyTorch ROCm 兼容的 8-bit CUDA 函数。可以通过在 GitHub 上创建账户来为 arlo-phoenix/bitsandbytes-rocm-5.6 的开发做出贡献。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-dev**](https://discord.com/channels/1104757954588196865/1104758010959634503/) (1 messages):

faldore: <@257999024458563585> 你实现这个了吗？

[https://arxiv.org/abs/2410.05258](https://arxiv.org/abs/2410.05258)

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**announcements**](https://discord.com/channels/1104757954588196865/1113462842436354149/1308497601322287124) (1 messages):

> - `Axolotl v0.5.2 release`
> - `Optimizer support`
> - `Upgraded dependencies`
> - `FSDP gradient accumulation fix`

- **Axolotl v0.5.2 发布，包含重大修复**：新发布的 [v0.5.2](https://github.com/axolotl-ai-cloud/axolotl/releases/tag/v0.5.2) 包含大量修复、改进的单元测试以及对底层依赖项的各种升级。
  
  - 值得注意的是，该版本解决了困扰上一个版本 v0.5.1 的 `pip install axolotl` 安装问题。
- **新增优化器支持功能**：Axolotl v0.5.2 引入了对 **schedule-free optimizers** 和 **ADOPT optimizer** 的支持，以提升性能。
  
  - 这包括对上游 [transformers](https://github.com/huggingface/transformers/releases/tag/v4.46.3) 库中 **FSDP+gradient accumulation**（梯度累积）问题的修正。
- **升级核心组件增强性能**：在此版本中，**liger** 和 **datasets** 等核心组件得到了显著升级。
  
  - 此外，**autoawq** 的集成也是 v0.5.2 版本的亮点之一。
- **撤回旧版本以解决安装问题**：**v0.5.1** 版本因通过 `pip install axolotl` 安装时存在问题而被撤回（yanked），现已在 v0.5.2 中修复。
  
  - 此举确保了用户可以顺利过渡到更稳定的 v0.5.2，而不会遇到安装障碍。
- **对未来更新的期待**：公告暗示地平线上还有更多改进和功能，称“更多内容即将推出！”。
  
  - 这表明了对增强整体用户体验的持续开发和承诺。

---

### **OpenAccess AI Collective (axolotl) ▷ #**[**axolotl-phorm-bot**](https://discord.com/channels/1104757954588196865/1225558824501510164/1308444799568838739) (2 messages):

> - `Phorm Bot 弃用`
> - `仓库 URL 问题`

- **Phorm Bot 可能已弃用**：一名用户询问 **phorm bot** 是否已弃用，暗示它可能已经损坏。
  
  - 另一名成员推测，这可能是因为它**指向了旧的仓库 URL**，在迁移到新组织后该 URL 尚未更新。
- **仓库 URL 问题**：讨论透露 **phorm bot** 的**仓库 URL** 已过时。
  
  - 有人指出，在迁移到新组织后，必要的**切换从未发生**，这可能导致了该 bot 的问题。

 

---

### **Modular (Mojo 🔥) ▷ #**[**max**](https://discord.com/channels/1087530497313357884/1212827597323509870/1308183326992896121) (12 messages🔥):

> - `Max Graphs 与知识图谱 (Knowledge Graphs)`
> - `使用 MAX 进行图搜索`
> - `Graph RAG 系统`
> - `Mojo 与 Max Agent 实现`

- **Max Graphs 促进知识图谱集成**：有人询问 **Max Graphs** 是否能增强传统的**知识图谱 (Knowledge Graphs)**，以统一 **LLM 推理**，将其作为 agentic [RAG 工具](https://arxiv.org/pdf/2404.16130)之一。
  
  - Darkmatter 指出，虽然**知识图谱**作为数据结构，但 **Max Graph** 代表的是一种计算方法。
- **尝试使用 MAX 加速图搜索**：关于利用 **MAX** 提升图搜索性能的讨论显示，目前的能力需要将整个图复制到 **MAX** 中。
  
  - 提出了一种潜在的变通方案，涉及将图编码为一维字节张量 (1D byte tensors)，尽管内存需求可能会带来挑战。
- **澄清图类型及其用途**：一名用户指出了各种图类型之间的区别，表明 **MAX 计算图** 与计算相关，而**知识图谱**存储关系。
  
  - 他们进一步解释说，**Graph RAG** 利用知识图谱增强检索，而 **Agent 图 (Agent Graph)** 描述了 Agent 之间的数据流。
- **Max Graph 的张量依赖担忧**：Msaelices 质疑 **Max Graph** 是否从根本上与张量绑定，并指出其 API 参数受限于 **TensorTypes**。
  
  - 这促使人们建议在进行实现咨询之前先审阅 API 文档。

 

**提到的链接**：[GitHub - microsoft/graphrag: A modular graph-based Retrieval-Augmented Generation (RAG) system](https://github.com/microsoft/graphrag)：一个模块化的基于图的检索增强生成 (RAG) 系统 - microsoft/graphrag

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**hackathon-announcements**](https://discord.com/channels/1280234300012494859/1280236929379602493/1308500279007776871) (1 messages):

> - `Google AI 工作坊`
> - `Gemini 集成`
> - `黑客松见解`

- **关于 Gemini 的 Google AI 工作坊**：欢迎参加 **11/26 下午 3 点 (PT)** 举行的特别 [Google AI 工作坊](https://lu.ma/agents-hackathon-googleai)，重点是在 LLM Agents MOOC 黑客松期间使用 **Gemini** 进行构建。
  
  - 活动包括 Gemini 的现场演示，以及与 **Google AI 专家**进行直接咨询的互动问答环节。
- **解锁 Gemini 的能力**：参与者将深入了解 **Gemini** 以及 **Google** 的 AI 模型和平台套件的潜力。
  
  - *不要错过这个机会*，利用尖端技术增强你的黑客松项目。

 

**提到的链接**：[Workshop with Google AI: Building with Gemini for the LLM Agents MOOC Hackathon · Luma](https://lu.ma/agents-hackathon-googleai)：Google AI 工作坊：为 LLM Agents MOOC 黑客松使用 Gemini 进行构建。关于工作坊：加入我们在 LLM Agents MOOC 举办的独家工作坊……

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-announcements**](https://discord.com/channels/1280234300012494859/1280369709623283732/1308176543134257328) (1 条消息):

> - `Lecture 10 Announcement`（第 10 讲公告）
> - `Percy Liang's Presentation`（Percy Liang 的演讲）
> - `Open-Source Foundation Models`（开源 Foundation Models）
> - `Course Logistics`（课程事务安排）

- **第 10 讲于 PST 时间下午 3:00 开始**：第 10 讲定于今天 **PST 时间下午 3:00** 举行，[此处提供直播](https://www.youtube.com/live/f3KKx9LWntQ)。
  
  - 本次讲座将展示 Foundation Models 领域的最新进展。
- **Percy Liang 讨论 AI 的开放性**：斯坦福大学计算机科学副教授 Percy Liang 将发表题为 **“Open-Source and Science in the Era of Foundation Models”**（Foundation Models 时代的开源与科学）的演讲。他强调，尽管目前获取途径有限，但开源对于推动 AI 创新至关重要。
  
  - Liang 强调需要社区资源来开发强大的开源模型。
- **课程资源已上线**：所有必要材料，包括直播链接和作业，均可在 [课程网站](http://llmagents-learning.org/f24) 访问。该网站集中了顺利完成课程所需的关键资源。
  
  - 鼓励学生访问该网站以获取最新更新和资料。
- **与课程工作人员沟通**：如有问题、反馈或疑虑，参与者可直接在指定频道与课程工作人员沟通。这确保了每个人都能及时获得有关课程相关问题的回复。
  
  - 通过适当的渠道进行互动将促进更顺畅的沟通。

 

**提到的链接**：[CS 194/294-196 (LLM Agents) - Lecture 10, Percy Liang](https://www.youtube.com/live/f3KKx9LWntQ.)：未找到描述

 

---

### **LLM Agents (Berkeley MOOC) ▷ #**[**mooc-lecture-discussion**](https://discord.com/channels/1280234300012494859/1282734248112947210/1308173583113125888) (7 条消息):

> - `Non-English models`（非英语模型）
> - `State of the art performance`（SOTA 性能）
> - `Low data point challenges`（低数据点挑战）

- **实现非英语模型的 SOTA 性能**：*Tejasmic* 询问了如何针对 **非英语模型**（尤其是 **低数据量** 的语言）实现 **State of the Art (SOTA)** 性能的方法。
  
  - 有建议提议在专门的频道中提出该问题，因为工作人员正在那里积极审查内容。
- **最佳讲座赞誉**：另一位成员 *zbrn_07683* 称其为 **“最好的讲座”**，表明了总体上的积极反响。
  
  - 这体现了社区对讲座内容的参与感和认可。

 

---

### **Torchtune ▷ #**[**general**](https://discord.com/channels/1216353675241590815/1216353675744641096/1308395875265020017) (9 条消息🔥):

> - `Flex Attention Limitations`（Flex Attention 的局限性）
> - `Attention Score Hacks`（Attention Score 的 Hack 方法）
> - `Vanilla Attention Strategies`（Vanilla Attention 策略）

- **Flex Attention 无法复制 scores**：一位成员在尝试将 Flex Attention 的 `score_mod` 函数中的 Attention Scores 复制到全局变量时遇到错误，提示 *Unsupported: HigherOrderOperator* 变异错误。
  
  - 另一位成员确认了 Flex Attention 中存在此限制，并指出可以在 [此处](https://github.com/pytorch-labs/attention-gym/issues/19) 查看该问题。
- **可能的 Vanilla Attention 变通方案**：讨论显示，由于缺乏对 SDPA 内部机制的访问，使用 Vanilla Attention 复制 Attention Scores 也是不可行的。
  
  - 一位成员指出，修改 Gemma 2 的 Attention 类可能提供一种解决方案，并强调其更具可操作性（hackable）。
- **发现 Score 提取的 Hack 方法**：一位成员分享了一个 [GitHub Gist](https://gist.github.com/drisspg/c66d79d51b5dd1895a552cef0820ba2e)，详细介绍了一种在不使用 Triton Kernels 的情况下获取 Attention Scores 的 Hack 方法。
  
  - 进一步的讨论承认，虽然这个 Hack 方法有效，但它偏离了标准的 torchtune 实现。

**提到的链接**：

- [Repro.py](https://gist.github.com/drisspg/c66d79d51b5dd1895a552cef0820ba2e)：GitHub Gist：即时分享代码、笔记和片段。
- [Issues · pytorch-labs/attention-gym](https://github.com/pytorch-labs/attention-gym/issues/19.)：处理 Flex-Attention 的有用工具和示例 - Issues · pytorch-labs/attention-gym

---

### **LAION ▷ #**[**announcements**](https://discord.com/channels/823813159592001537/826154622644649985/1308315538547544064) (1 messages):

> - `LAION-DISCO-12M`
> - `YouTube samples for ML`

- **LAION-DISCO-12M 发布，包含 1200 万个链接**：LAION 宣布推出 **LAION-DISCO-12M**，这是一个包含 **1200 万个链接**的集合，指向公开可用的 YouTube 样本并配有 metadata，旨在支持针对**通用音频和音乐**的基础 Machine Learning 研究。
  
  - 这一举措的更多细节见其 [blog post](https://laion.ai/blog/laion-disco-12m/)。
- **音频研究的 Metadata 增强**：LAION-DISCO-12M 集合中包含的 **metadata** 旨在促进音频分析领域 **Foundation Models** 的研究。
  
  - 几位开发者对公告中强调的潜在用例表示兴奋，并强调了在**音频 Machine Learning 领域**对更好数据的需求。

 

**提到的链接**：[来自 LAION (@laion_ai) 的推文](https://x.com/laion_ai/status/1858751486265622934)：我们宣布推出 LAION-DISCO-12M - 一个包含 1200 万个链接的集合，指向公开可用的 YouTube 样本并配有 metadata，以支持通用音频领域 Foundation Models 的基础 Machine Learning 研究...

 

---

### **Mozilla AI ▷ #**[**announcements**](https://discord.com/channels/1089876418936180786/1089876419926032396/1308466715360890920) (1 messages):

> - `Transformer Lab Demo`
> - `Metadata Filtering`
> - `Refact AI`
> - `Autonomous AI Agents`

- **Transformer Lab 演示即将开始**：今天的 **Transformer Lab** 演示即将开始，展示 Transformer 技术的最新进展。
  
  - 鼓励成员加入并参与精彩的讨论。
- **Metadata 过滤会议提醒**：明天，一场关于 [metadata 过滤](https://discord.com/events/1089876418936180786/1300483739872399411) 的会议将由 <@533894367354552330> 在频道 <#1262961960157450330> 主持。
  
  - 参与者可以获得关于 AI 中有效数据处理实践的宝贵见解。
- **Refact AI 将于周四登场**：在**周四**，[Refact AI](https://discord.com/events/1089876418936180786/1300459081181429810) 将讨论构建 **Autonomous AI Agents** 以端到端地执行工程任务。
  
  - 他们还将回答与会者的提问，这提供了一个互动学习的绝佳机会。

 

---

---

---

---

---

{% else %}

> 完整的频道细分内容已为邮件格式进行截断。
> 
> 如果您想查看完整细分，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
> 
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}