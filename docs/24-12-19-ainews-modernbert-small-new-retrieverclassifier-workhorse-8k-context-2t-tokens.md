---
companies:
- answerdotai
- lightonio
- hugging-face
- google-deepmind
- openai
- meta-ai-fair
- figure
date: '2024-12-20T03:27:55.084640Z'
description: '**Answer.ai/LightOn** 发布了 **ModernBERT**，这是一款更新的仅编码器（encoder-only）模型，支持
  **8k token 上下文**，在包含代码的 **2 万亿 token** 数据集上训练而成。该模型拥有 **1.39 亿/3.95 亿参数**，在检索、自然语言理解（NLU）和代码任务中表现出顶尖（SOTA）性能。其特点是采用了**交替注意力（Alternating
  Attention）**层，融合了全局和局部注意力。


  **Gemini 2.0 Flash Thinking** 在 Chatbot Arena 中首次亮相即登顶榜首，而 **O1 模型**在推理基准测试中取得了最高分。**Llama**
  的下载量突破了 **6.5 亿次**，在 3 个月内翻了一番。**OpenAI** 推出了具有语音功能的桌面端应用集成。**Figure** 交付了其首批商用人形机器人。机器人仿真领域也取得了显著进展，新的物理引擎
  **Genesis** 声称其运行速度比实时快 **43 万倍**。'
id: 65a7ef70-6cea-4cb1-a00a-26d16f926569
models:
- modernbert
- gemini-2.0-flash-thinking
- o1
- llama
original_slug: ainews-modernbert-small-new-retrieverclassifier
people:
- jeremyphoward
- alec-radford
- philschmid
- drjimfan
- bindureddy
title: ModernBert：新款小型检索/分类利器，支持 8k 上下文，训练量达 2T tokens。
topics:
- encoder-only-models
- long-context
- alternating-attention
- natural-language-understanding
- reasoning
- robotics-simulation
- physics-engine
- humanoid-robots
- model-performance
- model-releases
---

<!-- buttondown-editor-mode: plaintext -->**Encoder-only 模型就是你所需要的一切。**

> 2024/12/18-2024/12/19 AI 新闻。我们为你检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord（**215** 个频道，**4745** 条消息）。预计节省阅读时间（以 200wpm 计算）：**440 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论了！

正如他几个月来[一直预告](https://www.latent.space/p/answerai)的那样，Jeremy Howard 和 Answer.ai/LightOn 团队[今天发布了 ModernBERT](https://x.com/jeremyphoward/status/1869786023963832509?s=46)，更新了 2018 年的经典 BERT：


![image.png](https://assets.buttondown.email/images/5b764d0e-7bc1-48a7-b422-f0f0f5bbf3bd.png?w=960&fit=max)


[HuggingFace 博客文章](https://huggingface.co/blog/modernbert)详细介绍了为什么这很有用：

- **Context (上下文)**：旧的 BERT 只有约 500 token 的上下文；ModernBERT 拥有 8k。
- **Data (数据)**：旧的 BERT 基于较旧/较少的数据；ModernBERT 在 2T 数据上进行训练，包括“大量的代码”。
- **Size (规模)**：如今的 LLM 普遍 >70B，伴随着相应的成本和延迟问题；ModernBERT 仅有 139M (base)/395M (large) 参数。
- **同等规模下的 SOTA 性能**：在所有检索/NLU/代码类别中击败了像 DeBERTaV3 这样的常规 Kaggle 冠军。 
![image.png](https://assets.buttondown.email/images/c482db4e-ec78-450f-ae5b-6adcf65faf1d.png?w=960&fit=max)

- **真实世界的变长长上下文**：现实世界中的输入大小各不相同，因此这是我们努力优化的性能——即“variable”列。如你所见，对于变长输入，ModernBERT 比所有其他模型都快得多。 
![image.png](https://assets.buttondown.email/images/7d022871-e0e3-4747-983e-ac08212db3e9.png?w=960&fit=max)

- **Bidirectional (双向)**：Decoder-only 模型被特别限制不能“向后看”，而 BERT 可以填补空白：

```py
import torch
from transformers import pipeline
from pprint import pprint

pipe = pipeline(
    "fill-mask",
    model="answerdotai/ModernBERT-base",
    torch_dtype=torch.bfloat16,
)

input_text = "One thing I really like about the [MASK] newsletter is its ability to summarize the entire AI universe in one email, consistently, over time. Don't love the occasional multiple sends tho but I hear they are fixing it."
results = pipe(input_text)
pprint(results)
```

[论文](https://arxiv.org/pdf/2412.13663)中披露的众多有趣细节之一是 **Alternating Attention** 层——以 Noam Shazeer 在 Character 所做的相同方式混合全局和局部注意力（[我们的报道在此](https://buttondown.com/ainews/archive/ainews-shazeer-et-al-2024/)）：


![image.png](https://assets.buttondown.email/images/84a44194-4862-4937-97cb-d0fc54187399.png?w=960&fit=max)



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有回顾由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型发布与性能**

- [@drjwrae 宣布](https://twitter.com/drjwrae/status/1869806618025832788)发布 **Gemini 2.0 Flash Thinking**，该模型基于 2.0 Flash 模型构建，旨在提升推理能力。
- [@lmarena_ai 报告](https://twitter.com/lmarena_ai/status/1869793847548817563) **Gemini-2.0-Flash-Thinking** 在 Chatbot Arena 的所有类别中均首次亮相即位列第一。
- [@bindureddy 指出](https://twitter.com/bindureddy/status/1869542214734663795)新的 **O1 model** 在推理方面得分 91.58，并在 Livebench AI 上排名第一。
- [@answerdotai 和 @LightOnIO 发布了](https://twitter.com/reach_vb/status/1869791808030708054) **ModernBERT**，其上下文长度高达 8,192 tokens，并提升了性能。

**重大公司新闻**

- [@AIatMeta 分享道](https://twitter.com/AIatMeta/status/1869775975917257037)，**Llama 的下载量已超过 6.5 亿次**，在 3 个月内翻了一番。
- [@OpenAI 推出了](https://twitter.com/gdb/status/1869811511616778280)桌面端应用集成，支持 Xcode、Warp、Notion 等应用，并具备语音功能。
- [@adcock_brett 宣布](https://twitter.com/adcock_brett/status/1869863378975658441) **Figure** 已向商业客户交付了首批人形机器人。
- [Alec Radford 从 OpenAI 离职](https://twitter.com/iScienceLuvr/status/1869852854728700166)的消息被公开。

**技术进展**

- [@DrJimFan 讨论了](https://twitter.com/DrJimFan/status/1869795912597549137)机器人仿真领域的进展，强调了大规模并行化和生成式图形学的趋势。
- [@_philschmid 分享了](https://twitter.com/_philschmid/status/1869639246434246966)关于 **Genesis** 的细节，这是一个全新的物理引擎，声称比实时仿真快 43 万倍。
- [@krandiash 概述了](https://twitter.com/krandiash/status/1869828879856349488)在扩展 AI 模型上下文窗口和内存方面面临的挑战。

**梗与幽默**

- [@AmandaAskell 调侃了](https://twitter.com/AmandaAskell/status/1869584124627066977)物种通过 FOMO（错失恐惧症）进行繁衍的现象。
- [@_jasonwei 分享了](https://twitter.com/_jasonwei/status/1869618956333645940)被女朋友吐槽的经历，她将他的演讲比作电影《降临》（Arrival）中的场景。
- [@karpathy 发布了](https://twitter.com/karpathy/status/1869522720377221291)关于他每天下午 3:14 拍照的 PiOclock 传统。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Bamba：推理高效的混合 Mamba2 模型**

- **[Bamba：推理高效的混合 Mamba2 模型 🐍](https://huggingface.co/blog/bamba)** ([评分: 60, 评论: 14](https://reddit.com/r/LocalLLaMA/comments/1hhodui/bamba_inferenceefficient_hybrid_mamba2_model/)): **Bamba** 是一个基于 **Mamba2** 的**推理高效混合模型**。帖子标题暗示该模型关注性能差距和新的基准测试，尽管正文中未提供更多细节。
  - **基准测试差距**：讨论指出，由于训练数据以及在训练阶段加入了与基准测试对齐的指令数据集，**Bamba** 模型在数学基准测试中表现出与其他线性模型类似的差距。一个具体的例子是，通过添加 **metamath** 数据，**GSM8k score** 从 **36.77 提升到了 60.0**。
  - **方法论的开放性**：评论者对 **Bamba** 模型在训练和量化过程中的透明度表示赞赏，并对即将发布的论文表示期待，该论文承诺将提供有关数据源、比例和消融实验技术的详细见解。
  - **模型命名幽默**：网友们对 **Bamba**、**Zamba** 等模型的命名惯例进行了轻松的交流，并提供了 **Hugging Face** 上相关论文和模型的链接 ([Zamba-7B-v1](https://huggingface.co/Zyphra/Zamba-7B-v1), [Jamba](https://huggingface.co/papers/2403.19887), [Samba](https://huggingface.co/papers/2406.07522))。


**主题 2. Genesis：生成式物理引擎的突破**

- **[新款物理 AI 简直疯狂（开源）](https://v.redd.it/15c7r7rjxq7e1)** ([评分: 1350, 评论: 147](https://reddit.com/r/LocalLLaMA/comments/1hhmebr/new_physics_ai_is_absolutely_insane_opensource/)): 该帖子讨论了一个名为 **Genesis** 的 **开源物理 AI**，强调了其令人印象深刻的生成能力和物理引擎功能。由于缺乏详细的文字描述，链接的视频可能提供了关于其功能和应用的更多见解。
  - **怀疑与担忧**：许多评论者对该项目表示怀疑，将其与 **Theranos** 和 **Juicero** 等过度炒作的技术进行比较，并暗示其所属机构和“开源”声明可能被夸大了。**MayorWolf** 等人怀疑视频的真实性，认为其中包含创意剪辑，且开源部分可能仅限于 **Blender** 等现有工具中已有的功能。
  - **技术讨论**：一些用户讨论了技术层面，例如使用 **Taichi** 进行高效的 GPU 模拟，以及与 **Nvidia** 的 **Omniverse** 的潜在相似之处。**AwesomeDragon97** 指出了模拟中关于水滴粘附的一个缺陷，表明物理引擎需要进一步完善。
  - **项目合法性**：分享了该项目的 [网站](https://genesis-embodied-ai.github.io/) 和 [GitHub 仓库](https://github.com/Genesis-Embodied-AI/Genesis) 链接，一些用户注意到有顶尖大学参与，并认为它可能是真实的。其他用户如 **Same_Leadership_6238** 强调，虽然它看起来好得令人难以置信，但它是开源的，值得进一步调查。


- **Genesis 项目：一个能够生成由物理模拟平台驱动的 4D 动态世界的生成式物理引擎** ([评分: 103, 评论: 13](https://reddit.com/r/LocalLLaMA/comments/1hhl1m0/genesis_project_a_generative_physics_engine_able/)): **Genesis 项目** 推出了一种 **生成式物理引擎**，能够使用物理模拟平台创建 **4D 动态世界**，该项目历时 24 个月开发，由 20 多个研究实验室共同贡献。该引擎采用纯 Python 编写，运行速度比现有的 GPU 加速栈快 **10-80 倍**，并在模拟速度上取得了重大突破，比 **实时速度快约 430,000 倍**。它是开源的，旨在为机器人和物理 AI 应用自主生成复杂的物理世界。
  - **生成式物理引擎** 允许进行模拟，使包括软体机器人在内的机器人能够以远快于现实世界试验的速度进行实验和改进动作，这可能会彻底改变机器人和物理 AI 应用。
  - 对 **模拟和动画的影响** 是巨大的，使拥有 **NVIDIA 4090** 等消费级硬件的个人能够为现实世界应用训练机器人，而这在以前仅限于拥有大量资源的实体。
  - 由于其令人印象深刻的宣称，人们对该技术的能力存在怀疑，用户表示希望亲自测试该引擎以验证其性能。


**主题 3. Slim-Llama ASIC 处理器的效率飞跃**

- **[[Slim-Llama 是一款 LLM ASIC 处理器，能够处理 30 亿参数，而功耗仅为 4.69mW —— 我们很快就会了解到更多关于这一潜在 AI 游戏规则改变者的信息]](https://www.techradar.com/pro/slim-llama-is-an-llm-asic-processor-that-can-tackle-3-bllion-parameters-while-sipping-only-4-69mw-and-we-shall-find-out-more-about-this-potential-ai-game-changer-in-february-2025)** ([Score: 240, Comments: 25](https://reddit.com/r/LocalLLaMA/comments/1hhn2r0/slimllama_is_an_llm_asic_processor_that_can/)): **Slim-Llama** 是一款 **LLM ASIC 处理器**，能够处理 **30 亿参数**，而功耗仅为 **4.69mW**。关于这一潜在的 AI 硬件重大进展的更多细节预计很快就会公布。
  - 人们对 **Slim-Llama** 的性能持怀疑态度，担心其 **3000ms 延迟**以及其 **5 TOPS（能效比 1.3 TOPS/W）** 功耗效率的实用性。批评者认为，**500KB 内存**不足以在不使用外部内存的情况下运行 **1B 模型**，而外部内存会增加能耗（[来源](http://ssl.kaist.ac.kr/bbs/board.php?bo_table=HI_systems&wr_id=39)）。
  - **Slim-Llama** 仅支持 **1 比特和 1.5 比特模型**，被视为一种学术上的探索而非实际解决方案。由于其 **4.69mW** 的极低功耗，在 **可穿戴设备**、**IoT 传感器节点**和节能型 **工业应用** 中具有潜在用途。一些评论者对未来通过改进 **4-bit quantization** 和更好的软件支持来实现更多用例表示期待。
  - 讨论内容包括该芯片采用 **三星 28nm CMOS 工艺**，**芯片面积为 20.25mm²**，并对其在 **5nm 或 3nm** 等更先进工艺上的潜在表现感到好奇。此外，还有关于在“基于 **SLUT** 的 BMM 核心”上运行 **Enterprise Resource Planning** 模拟的玩笑，突显了该芯片的新颖性和小众吸引力。


**主题 4. Gemini 2.0 Flash Thinking Experimental 发布**

- **[[Gemini 2.0 Flash Thinking Experimental 现已在 Google AI Studio 免费开放（10 RPM，1500 次请求/天）]](https://i.redd.it/xbibsmke7u7e1.png)** ([Score: 73, Comments: 10](https://reddit.com/r/LocalLLaMA/comments/1hhxkyk/gemini_20_flash_thinking_experimental_now/)): **Gemini 2.0 Flash Thinking Experimental** 现已在 **Google AI Studio** 中免费提供，允许用户每分钟进行 10 次请求，每天 1500 次请求。界面包含用于回答“你现在是谁？”等查询的系统指令，并允许调整模型选择、token 数量和 temperature 设置。
  - 一位用户幽默地描述了一个 **thinking process**（思考过程）示例，模型计算了 "strawberry" 中 "r" 的出现次数，但指出了一处拼写错误，突显了模型的逐步推理能力。
  - 人们对利用 **Gemini 2.0 Flash Thinking** 的输出来训练其他思考模型的潜力感到好奇，这表明了对模型改进和开发的兴趣。

## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. Gemini 2.0 Flash Thinking 发布，性能超越旧模型**

- **[Gemini 2.0 Flash Thinking (推理模型，免费)](https://reddit.com/r/OpenAI/comments/1hhygng/gemini_20_flash_thinking_reasoning_free/)** ([Score: 268, Comments: 95](https://reddit.com/r/OpenAI/comments/1hhygng/gemini_20_flash_thinking_reasoning_free/)): **Gemini 2.0 Flash** 是 **Google** 推出的一款推理模型，现已在 [aistudio.google.com](http://aistudio.google.com) 免费开放，每天提供高达 **1500 次免费请求**，**知识截止日期为 2024 年**。作者认为它令人印象深刻，特别是它可以通过系统提示词（system prompts）进行引导，并指出在图像处理、通用问题和数学等任务上，它的表现与 **OpenAI 的 GPT-3.5** 相当甚至更好，同时批评了 OpenAI 产品的成本和限制。
  - 用户对 **Gemini 2.0 Flash** 的性能印象深刻，注意到它在 **数学** 方面优于其他模型，并且能够展示其推理过程，一些人认为这非常了不起。普遍观点认为它超越了 **OpenAI 的产品**，用户开始质疑支付 **ChatGPT Plus** 费用的价值。
  - 讨论涉及 **Google 的战略优势**，这得益于其具有成本效益的基础设施，特别是其 **TPUs**，这使得他们能够免费提供该模型，而 **OpenAI** 的模型则昂贵且封闭。这种成本优势被视为 Google 在 AI 领域潜在的长期胜利。
  - 一些用户表达了对 Google AI 产品 **UI/UX 改进** 的渴望，认为更友好的用户界面可以增强其吸引力。对话还涉及 Gemini 缺乏网页搜索功能，以及 AI Studio 中自定义指令的潜力，这增强了用户对模型响应的控制。

- **[O1 的完整 LiveBench 结果现已公布，表现非常亮眼。](https://i.redd.it/uyqg3gekap7e1.png)** ([得分: 267, 评论: 85](https://reddit.com/r/OpenAI/comments/1hhgd0v/o1s_full_livebench_results_are_now_up_and_theyre/)): **OpenAI 的 "o1-2024-12-17" 模型**在 **LiveBench 结果**中处于领先地位，特别是在 **Reasoning**（推理）和 **Global Average**（全球平均）得分方面表现卓越。该表格对比了多个模型在 **Coding**（编程）、**Mathematics**（数学）和 **Language**（语言）等指标上的表现，竞争对手包括 **Google**、**Alibaba** 和 **Anthropic**。
  - 关于 **O1 模型的定价和性能**存在大量讨论。一些用户认为 O1 比 Opus 更贵，因为存在“隐形思维 Token (thought tokens)”，导致每百万输出 Token (**mTok output**) 的成本超过 **200 美元**；而另一些人则声称价格相同，但由于推理 Token (reasoning tokens) 的存在导致成本累积 ([来源](https://livebench.ai))。
  - **O1 的能力和访问权限**也引发了争论，有人指出 O1 Pro API 尚未开放，且当前的 O1 模型使用了 "reasoning_effort" 参数，这会影响其性能和定价。该参数表明 O1 Pro 可能是一个具有更高推理强度的更高级版本。
  - 与 **Gemini 2.0 Flash** 等其他模型的**对比**非常普遍，Gemini 因其性价比和扩展潜力而受到关注。一些人推测 Gemini 的效率归功于 Google 的 TPU 资源，并对未来 1-2 年内实现“开箱即用的 AGI (in-the-box-AGI)”持乐观态度。


- **[Artificial Analysis 发布的 AI 竞赛历程](https://i.redd.it/280qbvkqqo7e1.jpeg)** ([得分: 157, 评论: 12](https://reddit.com/r/OpenAI/comments/1hhdzhd/the_ai_race_over_time_by_artificial_analysis/)): 来自 **Artificial Analysis** 的报告全面回顾了 AI 竞赛，重点关注了 **OpenAI、Anthropic、Google、Mistral** 和 **Meta** 的 AI 语言模型演进。一张折线图展示了随时间变化的“前沿语言模型智能 (Frontier Language Model Intelligence)”，使用“Artificial Analysis 质量指数”对比了 **2022 年第四季度至 2025 年第二季度**的模型质量，突出了 AI 发展的趋势和进步。[完整报告在此](https://artificialanalysis.ai/downloads/ai-review/2024/Artificial-Analysis-AI-Review-2024-Highlights.pdf)。
  - **Gemini 2.0** 被认为在各方面都优于目前的 **GPT-4o 模型**，并且可以在 **Google AI Studio** 上免费使用。
  - 关于时间线有一个修正：**GPT-3.5 Turbo** 在 **2022 年**尚未推出；当时可用的是 **GPT-3.5 Legacy**。


**主题 2. NotebookLM 引入交互式播客功能**

- **Notebook LM 交互测试版。令人震撼。** ([得分: 272, 评论: 69](https://reddit.com/r/OpenAI/comments/1hhlsyx/notebook_lm_interaction_beta_mindblown/)): **Google** 悄然激活了 **NotebookLM** 中的**交互功能**，允许用户与生成的播客进行互动。该帖子表达了对这一新功能的兴奋，称其“令人震撼 (mindblowing)”。
  - 用户讨论了 **NotebookLM** 中的**交互功能**，指出它允许就上传的源材料与 AI 进行实时对话。然而，目前的交互仍停留在表面，用户表达了对更深层次对话能力以及相比 **ChatGPT** 更好的 Prompt 响应的期望。
  - 该功能需要创建一个新的笔记本并添加来源以生成音频概览。音频准备就绪后即可开始交互，但一些用户注意到它缺乏保存或下载交互式播客的功能，且可用性可能因地区而异。
  - 对于 **Google** 在 AI 领域的进展，反应褒贬不一，一些用户对 Google 在 AI 竞赛中的地位表示怀疑，而另一些人则指出了该功能在学习方面的实用性，同时也有人将其与 **OpenAI** 最近的更新进行了对比，部分人认为 OpenAI 的更新不尽如人意。


---

# AI Discord 回顾

> 由 o1-2024-12-17 生成的摘要之摘要的总结

**主题 1. 激烈的模型大战与大胆的价格削减**  

- [**Gemini 2.0 闪亮登场**](https://x.com/NoamShazeer/status/1869789881637200228)：用户称赞 *“Gemini 2.0 Flash Thinking”* 展示了显式的思维链（chain-of-thought），并在推理任务中击败了旧模型。包括 [lmarena.ai 的提及](https://x.com/lmarena_ai/status/1869793847548817563)在内的多项测试显示，它在性能排行榜上名列前茅，引发了公众的热烈讨论。  
- [**OpenRouter 在史诗级对决中大幅降价**](https://openrouter.ai/gryphe/mythomax-l2-13b)：**MythoMax** 和 **QwQ** 等供应商降价超过 7%，[mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo) 降价 12.5%。观察人士称之为 *“持续的价格战”*，因为 AI 供应商正在争夺用户采用率。  
- [**Databricks 为增长狂揽 100 亿美元**](https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation)：该公司以惊人的 620 亿美元估值完成了一轮巨额融资，计划使年化营收运行率（revenue run rate）超过 30 亿美元。利益相关者将这一增长归功于企业级 AI 需求的飙升和 60% 的年增长率。

**主题 2. 多 GPU 与微调热潮**  

- [**Unsloth 筹备 GPU 魔法**](https://unsloth.ai/blog/llama3-3)：多 GPU 支持将于第一季度上线，团队正在测试企业定价和销售体系重组。他们确认 Llama 3.3 需要大约 41GB 的 VRAM 才能进行妥善的 fine-tune（微调）。  
- [**SGLang 与 vLLM 的性能对决**](https://lmsys.org/blog/2024-12-04-sglang-v0-4)：vLLM 在原始吞吐量（throughput）上胜出，而 SGLang 在结构化输出和调度方面表现出色。工程师们权衡了利弊，指出 SGLang 灵活的模块化方法适用于某些工作流。  
- [**量化（Quantization）拯救世界**](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu)：讨论区热议使用 4-bit 或 8-bit 量化来缩小内存占用。贡献者强调 *“RAG 加上量化”* 是资源受限任务的高效路径。

**主题 3. Agent、RAG 与 RLHF 的突破**  

- [**Agentic Systems 竞相发展**](https://www.anthropic.com/research/building-effective-agents)：Anthropic 的 *“Agentic Systems 之年”* 蓝图概述了可组合模式，引发了对 2025 年重大飞跃的推测。研究人员将这些设计与经典搜索进行了比较，并指出开放式思维模式可以超越简单的检索。  
- [**异步 RLHF 助力更快训练**](https://arxiv.org/abs/2410.18252)：一篇论文提出了 off-policy RLHF，将生成与学习解耦，以加速语言模型的精炼。社区在追求效率的过程中争论 *“我们可以容忍多少程度的 off-policyness？”*。  
- [**多 Agent LlamaIndex 释放 RAG 潜力**](https://t.co/lbhFDbSabS)：开发者正从单 Agent 转向多 Agent 架构，每个 Agent 专注于一个专门的子任务，以实现稳健的检索增强生成（RAG）。他们使用 Agent 工厂来协调任务，确保对大型语料库有更好的覆盖。

**主题 4. AI 编程工具成为焦点**  

- [**Cursor 0.44.4 升级**](https://www.cursor.com/downloads)：此次发布引入了 “Yolo mode” 并改进了 Agent 命令，详见 [changelog](https://www.cursor.com/changelog)。早期采用者注意到在大型项目中代码编辑速度更快，任务处理能力更强。  
- [**GitHub Copilot Chat 推出免费版**](https://x.com/code/status/1869449373995708703)：Microsoft 宣布了一个无需信用卡的层级，甚至可以调用 *“Claude 以获得更强的能力”*。开发者为免费的实时代码建议欢呼，尽管有些人仍然更喜欢传统的 diff 编辑来进行版本控制。  
- [**Windsurf vs. Cursor 大对决**](https://www.builder.io/blog/windsurf-vs-cursor)：用户比较了协作编辑、大文件处理和性能。许多人提到 Cursor 在复杂重构中的一致性，而一些人则欣赏 Windsurf 在处理较小任务时灵活的 UI。

**主题 5. 新鲜的库与开源探索**  

- [**Genesis AI 幻化物理现实**](https://x.com/zhou_xian_/status/1869511650782658846)：一个新的生成式引擎可以模拟 4D 世界，*速度比实时快 430,000 倍*。机器人爱好者对在 RTX4090 上仅需 26 秒的训练运行感到惊叹，该项目展示在 [Genesis-Embodied-AI/Genesis 仓库](https://github.com/Genesis-Embodied-AI/Genesis)中。  
- [**ModernBERT 亮相**](https://huggingface.co/blog/modernbert)：这个 *“主力模型”* 与旧版 BERT 相比，提供了扩展的上下文以及改进的分类或检索能力。社区测试者确认了其在 RAG 工作流中具有更好的性能和更简单的优化。  
- [**Nomic 在浏览器中映射数据**](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers)：其数据映射系列的最后一篇文章展示了可扩展的 embeddings（嵌入）和降维技术如何使大规模数据集的可视化变得大众化。读者称赞它是探索性分析的颠覆者。

---

# PART 1: 高层级 Discord 摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 筹备 Multi-GPU 魔法**：**Unsloth** 的 Multi-GPU 支持计划于第一季度推出，团队正在进行最终测试并微调定价细节。
   - 他们还暗示将针对企业兴趣改进**销售流程**，尽管其企业版 Beta 仍处于测试阶段。
- **Llama 3.3 提升动力**：根据 [Unsloth 博客](https://unsloth.ai/blog/llama3-3)，**Llama 3.3** 模型微调大约需要 41GB 的 VRAM。
   - 参与者报告称，与早期版本相比，该版本性能更高，这归功于在大数据集上进行精心训练周期的收益。
- **SGLang vs. vLLM：速度大比拼**：许多人认为 **vLLM** 在处理繁重的生产任务时超过了 **SGLang**，但 [SGLang v0.4](https://lmsys.org/blog/2024-12-04-sglang-v0-4) 在结构化输出和调度技巧方面看起来很有前景。
   - 社区成员认为 **vLLM** 在吞吐量方面更强，而 SGLang 则吸引那些优化模块化结果的用户。
- **RAG 遇见量化 (Quantization)**：当资源紧张时，**Retrieval-Augmented Generation (RAG)** 表现为直接微调的一种更智能的替代方案，通常采用分块数据和 embeddings 进行上下文检索。
   - 用户称赞**量化**（参见 [Transformers 文档](https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu)）可以在不完全牺牲性能的情况下缩小内存占用。
- **LoRAs、合并与指令微调 (Instruction Tuning) 警告**：将 **Low Rank Adapters (LoRAs)** 与基础模型合并（可能保存为 [GGUF 选项](https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf)）需要仔细平衡参数以避免不必要的失真。
   - 一篇 [指令微调论文](https://arxiv.org/abs/2402.05119) 强调了部分训练如何退化核心知识，突显了在没有彻底验证的情况下合并多种技术的风险。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor 0.44.4 发布，增强 Agent 功能**：Cursor **0.44.4** 引入了改进的 Agent 功能、**Yolo mode**，可在此处 [下载](https://www.cursor.com/downloads)。
   - 工程师们称赞其更快的命令执行和更好的任务处理能力，并引用了 [更新日志 (changelog)](https://www.cursor.com/changelog) 以获取详细分解。
- **抛硬币：O1 vs Sonnet 3.5**：用户将 **O1** 的成本定在每次请求约 40 美分，并将其收益与 **Sonnet 3.5** 进行了比较。
   - 一些人认为 Sonnet 3.5 “足够好”，而另一些人则质疑 O1 的额外成本是否物有所值。
- **构建之争：Framer vs. DIY 代码**：一场生动的讨论对比了用于快速建站的 **Framer** 与完全自定义的代码。
   - 一些人称赞其节省了时间，而另一些人则更喜欢对性能和灵活性进行完全控制。
- **Gemini-1206 引起好奇**：成员们对 **Gemini-1206** 表现出兴趣，但关于其能力的具体证据仍然稀缺。
   - 其他人则继续专注于使用 **Sonnet 3.5** 进行编码，因为他们缺乏关于 Gemini-1206 的广泛数据。
- **大学还是创业：大对决**：有人认为**常春藤盟校 (Ivy League)** 的资历提供了人脉优势，而另一些人则倾向于跳过学校去构建现实世界的产品。
   - 观点各异，个人成功案例表明任何路径都可以产生重大突破。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **Cline 与 Gemini 联手获胜**：多位成员称赞 [Cline v3](https://x.com/sdrzn/status/1869470308442452478) 结合 **Gemini 2.0** 带来了更流畅的编码体验和大型任务处理能力。
   - 他们指出，这优于其他配置，主要是由于更快的迭代和更稳定的重构能力。
- **Windsurf vs Cursor 摊牌**：比较参考了关于协作编辑和文件处理等功能的 [直接对比分析](https://www.builder.io/blog/windsurf-vs-cursor)。
   - 观点似乎存在分歧，但许多人认为 Cursor 更一致的性能是重度代码工作流中的关键优势。
- **额度结转 (Credit Rollover) 保证**：用户确认 [Codeium 付费计划](https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits-but-not-premium-flow-action-credits) 中的 **flex credits 可以结转**，确保不会突然中断。
   - 一些参与者对支付后不会丢失额度感到宽慰，强调了稳定订阅模型的重要性。
- **Claude vs Gemini 模型讨论**：社区成员权衡了 **Claude Sonnet**、**Gemini** 和其他 AI 模型之间的性能差异，并参考了 [Aider LLM Leaderboards](https://aider.chat/docs/leaderboards)。
   - 他们强调需要上下文提示和详尽的文档，以充分利用每个模型的编码潜力。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Gemini 2.0 展现“大声思考”技巧**：Google 推出了 [Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1869789881637200228)，这是一个实验性模型，通过训练**显式思维链 (explicit chain-of-thought)** 来增强聊天机器人任务中的推理能力和速度。
   - 社区成员引用了 [Denny Zhou 关于经典 AI 依赖搜索的立场](https://x.com/denny_zhou/status/1869771028693713284)，暗示 **Gemini** 的开放思考模式可能超越朴素的检索方案。
- **OpenAI 语音模式开启联动**：OpenAI 在**语音模式**中推出了 **Work with Apps** 功能，实现了与 **Notion** 和 **Apple Notes** 等应用的集成，正如其 [12 Days of ChatGPT 网站](https://openai.com/12-days/?day=11)所预告的那样。
   - 成员们称这是将 **ChatGPT** 与现实世界生产力连接起来的简单但重要的一步，一些人希望高级语音功能能够助力日常任务。
- **Chollet 关于 “o1 争端” 震惊 LLM 圈**：François Chollet 将 **o1** 标记为 LLM 比作将 **AlphaGo** 称为“卷积网络 (convnet)”，在 [X](https://x.com/fchollet/status/1869612195425972715) 上引发了激烈争论。
   - 社区成员指出这与之前的 *Subbarao/Miles Brundage 事件* 类似，要求澄清 **o1** 架构的呼声进一步加剧了这场风波。
- **FineMath：LLM 算术能力的巨大提升**：来自 [@anton_lozhkov](https://x.com/anton_lozhkov/status/1869771053146464507) 的链接展示了 **FineMath**，这是一个专注于数学的数据集，包含超过 **50B+ tokens**，有望比传统语料库带来更大提升。
   - 参与者认为这是复杂代码数学任务的一大飞跃，并提到将 *FineMath 与主流预训练合并* 以处理高级计算。
- **RLHF 书籍：找错别字，赢免费副本**：GitHub 上提到了一个 **RLHF** 资源，发现错别字或格式错误的志愿者有资格获得该书的免费副本。
   - 积极的贡献者发现以这种方式完善 **reinforcement learning** 基础知识压力较小，称这一过程既有趣又对社区有益。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **OpenAI 第 11 天提升 ChatGPT**：**12 Days of OpenAI** 的第 11 天为 **ChatGPT** 引入了新方法，并举行了 [YouTube 直播](https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz)，重点展示了高级代码协作。
   - 工程师现在可以在 AI 的协助下扩展日常开发周期，尽管**手动复制操作**仍然是必要的。
- **ChatGPT 与 XCode 集成**：参与者讨论了将代码从 **ChatGPT** 直接复制到 **XCode** 中，从而简化 iOS 开发任务。
   - 这一步带来了便利，但实际的代码插入仍取决于用户发起的触发操作。
- **Google 的 Gemini 2.0 备受瞩目**：Google 发布了 **Gemini 2.0 Flash Thinking** 实验性模型，其大胆的性能宣称引起了好奇。
   - 一些参与者在模型处理**字母计数任务**出错后对其可靠性表示怀疑，引发了对其真实实力的质疑。
- **使用 ChatGPT 演示 YouTube 克隆**：成员们探索了使用 **ChatGPT** 构建类似 YouTube 的体验，涵盖了前端和后端解决方案。
   - 虽然前端任务看起来很简单，但服务器端设置需要通过终端指令执行更多步骤。
- **AI 自动化在工程领域升温**：对话集中在 AI 全面自动化软件开发的前景上，这正在重塑对人类工程师的需求。
   - 虽然许多人认识到潜在的时间节省，但也有人怀疑炒作是否超前于实际的突破。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **FSDP vs Tensor Parallel Tangle**: 在 Eleuther，参与者对比了 **Fully Sharded Data Parallel (FSDP)** 与 **Tensor Parallelism**，并参考了 [llama-recipes](https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/config_utils.py#L95) 的实际实现。
   - 他们争论了 FSDP 中较高的通信开销，并将其与基于 Tensor 方法的直接并行操作优势进行了权衡，一些人对多节点扩展限制表示担忧。
- **NaturalAttention Nudges Adam**: 一位成员在 [GitHub](https://github.com/jeroaranda/naturalattention) 上重点介绍了一个新的 **Natural Attention Optimizer**，它通过基于 Attention 的梯度调整来修改 Adam，并由 [Natural_attention_proofs.pdf](https://github.com/jeroaranda/naturalattention/blob/main/papers/Natural_attention_proofs.pdf) 中的证明提供支持。
   - 他们声称性能有显著提升，尽管有人指出 [natural_attention.py](https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py) 中的代码可能存在 Bug，并建议在复现结果时保持谨慎。
- **Diffusion vs Autoregressive Arm-Wrestle**: 一场关于图像和文本领域中 **diffusion** 和 **autoregressive** 模型对比的讨论展开了，重点讨论了效率权衡和离散数据处理。
   - 一些人认为 diffusion 在图像生成方面领先，但在需要 Token 级控制的任务中可能会受到 autoregressive 方法的挑战。
- **Koopman Commotion in NNs**: 成员们辩论了将 **Koopman theory** 应用于神经网络的问题，参考了 [Time-Delay Observables for Koopman: Theory and Applications](https://arxiv.org/abs/1810.01479) 和 [Learning Invariant Subspaces of Koopman Operators--Part 1](https://arxiv.org/abs/2212.07358)。
   - 他们质疑将 Koopman 方法强加于标准框架的正当性，认为如果底层数学与现实世界的激活行为不符，可能会误导研究人员。
- **Steered Sparse AE OOD Queries**: 在可解释性讨论中，爱好者们探索了 **steered sparse autoencoders (SAE)**，以及对重构质心进行余弦相似度检查是否能有效衡量 Out-of-Distribution (OOD) 数据。
   - 他们报告称，调整一个激活值通常会影响其他激活值，这表明存在很强的相互依赖性，并提醒在解释基于 SAE 的 OOD 分数时要谨慎。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity's Referral Program Boosts Sign-Ups**: 多位用户确认 [Perplexity 提供推荐计划](https://www.perplexity.ai/settings/account)，为那些邀请新用户注册的人提供奖励。
   - 爱好者们旨在招募整个团体，加速平台的覆盖范围，并激发了关于用户增长的讨论。
- **You.com Imitation Raises Accuracy Concerns**: 社区成员讨论了 **You.com** 使用基于搜索的系统指令复制回答的情况，并质疑其输出质量。
   - 他们指出，依赖直接的模型调用通常会产生更精确的逻辑，揭示了面向搜索的问答解决方案中潜在的差距。
- **Game Descriptions Overwhelm Translation Limits**: 一位尝试将长列表转换为法语的用户遇到了大小限制，显示了 **Perplexity AI** 的文本处理约束。
   - 他们寻求关于将内容分割成更小块的建议，希望能绕过复杂翻译任务中的这些限制。
- **Magic Spell Hypothesis Sparks Curiosity**: 一份发布的 [文档](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA) 描述了 **Magic Spell Hypothesis**，将高级语言模式与科学界新兴概念联系起来。
   - 研究人员和社区成员评估了其可信度，赞扬了在结构化实验中测试边缘理论的尝试。

---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 取得进展**：12/19，**Gemini 2.0 Flash Thinking** 推出了 `gemini-2.0-flash-thinking-exp-1219` 变体，正如 [Jeff Dean 的推文](https://x.com/JeffDean/status/1869789813232341267)所言，该模型在 Agent 工作流中表现出更强的推理能力。
   - 初步测试显示其性能优于 O1 和 DeepSeek，部分社区成员对其升级后的输出质量表示赞赏。
- **Aider 与 MCP 深度集成**：用户实现了 **Aider** 与 **MCP** 的集成以简化 Jira 任务，参考了 [Sentry 集成服务器 - MCP Server 集成](https://mcpserver.cloud/server/server-sentry)。
   - 他们讨论了在 MCP 配置中用其他模型替换 Sonnet 的方案，这为错误追踪和工作流自动化提供了极高的灵活性。
- **OpenAPI 孪生使用引发热议**：社区成员探索了在本地运行 **Ollama** 的同时，在 Hugging Face 上运行 **QwQ**，并明确了 Hugging Face 要求使用其自身的 API 以实现无缝模型切换。
   - 他们发现需要在模型名称中注明服务来源，以防止在多 API 配置中产生混淆。
- **Copilot Chat 迎来更新**：根据 [GitHub 的公告](https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/)，**GitHub Copilot Chat** 推出了免费的沉浸模式，提供实时代码交互和更精准的多文件编辑。
   - 虽然用户对增强的聊天界面表示赞赏，但一些人仍然倾向于使用传统的 **diff 编辑**，以控制成本并保持工作流的可预测性。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 与 Supabase 实现即时设置**：**Bolt** 与 **Supabase** 的集成已正式上线，提供更简单的“一键连接”，如 [这条来自 StackBlitz 的推文](https://x.com/stackblitz/status/1869715661444043245)所示。它消除了手动步骤，让工程师能更快地统一服务并减少开销。
   - 用户称赞了这种简便的设置方式，指出它缩短了数据驱动型应用的开发周期，并提供了无摩擦的开发者体验。
- **Figma 导出受阻与 .env 文件问题**：用户报告了 **.env 文件** 重置导致 Firebase 配置中断的问题，刷新后锁定尝试失败，并引发“项目超出支持的总 Prompt 大小”错误。
   - 此外，目前无法直接从 **Figma** 上传，迫使设计师依赖截图，同时他们也呼吁提供更强大的“从设计到开发”的集成功能。
- **冗余代码优化与 Public 文件夹设置**：社区成员询问 **Bolt** 是否可以分析代码中的冗余块，旨在减少大规模应用中的 Token 消耗。他们还需要关于构建 **public 文件夹** 以托管图像的指导，反映出对项目结构的困惑。
   - 一些人建议提供更直白的文档来解决文件夹设置的不确定性，表明在配合 Bolt 工作时需要更简单的参考资料。
- **会话故障与 Token 消耗困扰**：频繁的会话超时和强制页面刷新导致许多人在 **Bolt** 中丢失聊天记录，增加了挫败感和 Token 成本。开发团队正在调查这些身份验证问题，但实时中断的情况仍然存在。
   - 用户希望修复这些问题以减少冗余输出，并控制 Token 的过度支出，寻求项目工作流的稳定性。
- **社区汇聚编写指南与集成方案**：参与者计划为 **Bolt** 编写一份更广泛的指南，并提供一个用户仪表板用于提交和审核资源。对话涉及了 **Stripe** 集成、高级 Token 处理以及与多种技术栈的协同。
   - 他们还展示了 [Wiser - 知识共享平台](https://boltwiser.levyandco.net/)，暗示了在共享内容和更完善的开发者体验方面有更深层次的扩展。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **Interactive Mode 覆盖全员**：开发团队确认 **Interactive Mode** 已覆盖 100% 的用户，并对音频概览（audio overviews）进行了显著改进。
   - 爱好者们称赞了其带来的创意可能性，并分享了更流畅部署的第一手经验。
- **用于自动 NPC 的 MySQL 数据库钩子**：一位游戏主持人（game master）询问如何将大型 **MySQL** 数据库连接到 **NotebookLM**，以实现非玩家角色（NPC）响应的自动化。
   - 他们强调了积累十年的 RPG 数据，并寻求管理动态查询的方法。
- **播客作者调整录音设置**：成员们讨论了交互式播客功能不存储对话的问题，这迫使他们必须为外部听众进行独立的音频采集。
   - 一个简洁的“播客风格提示词”引发了人们对 **QWQ model** 评论中更快速、更直率见解的兴趣。
- **AI 生成的太空 Vlog 震撼观众**：一位用户展示了由 AI 渲染的为期一年的宇航员隔离 Vlog，链接见 [此 YouTube 链接](https://youtu.be/_ys7FchEkak?feature=shared)。
   - 其他人注意到由 NotebookML 输出驱动的每日动画上传，展示了持续的内容生产能力。
- **更新后的 UI 获得好评**：用户们赞赏 **NotebookLM** 界面的翻新，称其在项目导航方面更加灵敏和便捷。
   - 他们渴望测试新的布局，并称赞了整体精致的外观。



---



## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SDXL 的 Ubuntu 步骤**：一些成员分享了在 **Ubuntu** 上运行 **SDXL** 的技巧，建议使用来自 [stable-diffusion-webui-forge](https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh) 的 shell 脚本以简化设置。
   - 他们强调了系统知识对于避免性能瓶颈的重要性。
- **ComfyUI 崩溃问题**：工程师们抱怨尽管尝试修复采样问题，**ComfyUI** 仍持续出现错误和焦黑的输出。
   - 他们建议使用 **Euler** 采样并配合调优后的去噪水平（denoising levels），以减少瑕疵结果。
- **AI 图像通往完美的道路充满坎坷**：一些人认为，由于当前的挑战，**AI 生成的图像**和**视频**到 2030 年也不会达到完美。
   - 另一些人则反驳称，技术的飞速跨越可能会更早地带来精致的输出。
- **关于 P=NP 的量子争论**：一场激烈的聊天集中在如果 **P=NP** 成为现实，**quantum computing**（量子计算）是否还有相关性。
   - 怀疑者指出从量子态中提取现实世界价值存在困难，并引用了实际执行中的复杂性。
- **Civitai.com 又挂了？**：多位用户注意到 **civitai.com** 频繁宕机，导致模型访问变得困难。
   - 他们推测反复出现的服务器问题是导致多次停机的原因。



---



## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **GPU 闪烁与电感啸叫**：用户抱怨退货的 **RX 6750XT** 存在*离谱的电感啸叫（coil whine）*，此外 **VRChat** 对显存的渴求促使一些人选择了 **4090s**。
   - 他们还对下一代 **RTX 50** 系列显卡可能更高的价格表示担忧，同时对比了 **7900 XTX**。
- **Triton 在 AMD 上的尝试**：社区成员在 **RX 7900** 等 **AMD GPU** 上测试了 **Triton** kernel，注意到性能仍落后于 **PyTorch/rocBLAS**。
   - 他们还发现 **warp-specialization** 在 **Triton 3.x** 中被移除，这促使他们探索其他的优化方案。
- **CARLA 冲入 UE 5.5**：**CARLA 0.10.0 版本**引入了 [Unreal Engine 5.5 特性](https://carla.org/2024/12/19/release-0.10.0/)，如 **Lumen** 和 **Nanite**，提升了环境真实感。
   - 与会者还称赞了 [Genesis AI](https://genesis-embodied-ai.github.io/) 的水滴演示，预见了其与 **Sim2Real** 的协同效应，并引用了 [Waymo 的合成数据方法](https://waymo.com/research/embedding-synthetic-off-policy-experience-for-autonomous-driving-via-zero/)用于自动驾驶。
- **MatX 的 HPC 招聘热潮**：**MatX** 宣布公开招聘 **low-level compute kernel authors** 和 **ML performance engineers**，旨在构建 **LLM accelerator ASIC**。
   - [职位列表](https://grnh.se/2b337cb08us)强调了一个高信任度的环境，比起长时间的测试，该环境更看重*大胆的设计决策*。
- **Alma 的 40 选项基准测试大杂烩**：两人组发布了 **alma**，这是一个 Python 包，可以在单个函数调用中检查超过 **40 种 PyTorch 转换选项**的 throughput（吞吐量）。
   - 根据 [GitHub](https://github.com/saifhaq/alma) 的描述，它通过*隔离进程*优雅地处理失败，并很快将扩展到 **JAX** 和 **llama.cpp**。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Anthropic Agents 发力**：Anthropic 发布了 [Building effective agents](https://www.anthropic.com/research/building-effective-agents)，介绍了 **AI agentic systems** 的模式，并预告 2025 年将迎来重大里程碑。
   - 他们强调了可组合的工作流，并引用了一条关于“代理系统之年（year of agentic systems）”的推文，旨在进行高级设计。
- **Gemini 2.0 提速**：多条推文（包括 [lmarena.ai 的提及](https://x.com/lmarena_ai/status/1869793847548817563) 和 [Noam Shazeer 的公告](https://x.com/NoamShazeer/status/1869789881637200228)）称赞 **Gemini 2.0 Flash Thinking** 在所有类别中均名列前茅。
   - 该模型训练以“出声思考（think out loud）”，从而实现更强的推理能力，并超越了早期的 Gemini 版本。
- **Databricks 融资 100 亿美元**：他们宣布了价值 **100 亿美元** 的 [J 轮融资](https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation)，由 Thrive Capital 领投，估值达到 **620 亿美元**。
   - 他们预计年营收运行率（revenue run rate）将突破 **30 亿美元**，并报告了由 **AI** 需求引发的 60% 的增长。
- **ModernBERT 登场**：一个名为 **ModernBERT** 的新模型[被推出](https://huggingface.co/blog/modernbert)，作为一种具有扩展上下文和改进性能的“主力军”选择。
   - 诸如 [Jeremy Howard 的提及](https://x.com/jeremyphoward/status/1869786023963832509) 等参考资料显示，人们正尝试将其应用于检索和分类，引发了从业者之间的讨论。
- **Radford 告别 OpenAI**：GPT 原始论文的作者 Alec Radford [离开了 OpenAI](https://x.com/steph_palazzolo/status/1869848094009110826)，去追求独立研究。
   - 这一变动引发了关于 **OpenAI** 在行业内未来走向的猜测。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 的 Vision 版本**：OpenInterpreter 1.0 现在包含 **vision** 支持，可通过 [GitHub](https://github.com/OpenInterpreter/open-interpreter.git) 和 `pip install git+https://github.com/OpenInterpreter/open-interpreter.git@development` 进行安装。
   - 实验表明 `--tools gui` 命令在桥接不同模型或 API 时运行良好，用户还提到了本地或基于 SSH 的用法。
- **Server 模式引发执行疑问**：成员们询问 **server mode** 如何处理命令执行，讨论任务是在本地运行还是在服务器上运行。
   - 他们提到了使用 SSH 进行更简单的交互，并建议增加一个前端以改进工作流。
- **Google Gemini 2.0 受到关注**：一位用户对 **Google Gemini 2.0** 在 OS 模式下的多模态任务表现出兴趣，希望其具备高效的命令执行能力。
   - 他们将其与现有配置进行了比较，并想知道它是否能与其他系统有效竞争。
- **清理安装与 O1 困惑**：一些用户在多次配置后遇到了 **OpenInterpreter** 安装问题，促使他们删除标志以进行全新设置。
   - 同时，一位 O1 频道的用户抱怨文档不清晰，即使在阅读了官方参考资料后仍寻求直接指导。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **Safetensors 故障困扰 LM Studio**：用户在加载模型时遇到 **Safetensors header is unexpectedly large: bytes=2199142139136** 错误，被迫重新下载 **MLX 版本的 Llama 3.3** 以修复可能的损坏问题。
   - 讨论中提到了文件兼容性冲突，一些用户建议在未来的下载中进行仔细的文件检查。
- **移动端梦想：iOS 迎来聊天功能，Android 仍在等待**：一款名为 **3Sparks Chat** ([link](https://apps.apple.com/us/app/3sparks-chat/id6736871168)) 的 iOS 应用可连接到 Mac 或 PC 上的 LM Studio，为本地 LLM 提供手持界面。
   - 成员们对缺乏 Android 客户端表示失望，社区纷纷请求替代方案。
- **AMD 24.12.1 的困扰**：**AMD 24.12.1** 驱动在通过 LM Studio 加载模型（连接到 llama.cpp rocm 库）时引发了系统卡顿和性能下降。
   - 降级驱动程序解决了一些配置中的问题，**7900XTX** GPU 的稳定性也成为了关注焦点。
- **LM Studio 的视觉模型希望**：关于**图像输入模型**的查询引出了对 [mlx-community/Llama-3.2-11B-Vision-Instruct-4bit](https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit) 的提及，突显了集成视觉功能的早期尝试。
   - 用户报告了在 Windows 上的加载问题，引发了关于模型与本地硬件兼容性的讨论。
- **Apple Silicon vs. 4090 GPU 对决**：社区成员质疑 **Mac Pro 和 Ultra 芯片** 是否因内存带宽优势而优于 **30 或 4090** 显卡。
   - 基准测试引用指向了 [llama.cpp GitHub discussion](https://github.com/ggerganov/llama.cpp/discussions/4167)，用户证实 4090 在实际测试中仍保持着更快的指标。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **降价震动 LLM 市场**：今天早上 [gryphe/mythomax-l2-13b](https://openrouter.ai/gryphe/mythomax-l2-13b) 降价 7%，[qwen/qwq-32b-preview](https://openrouter.ai/qwen/qwq-32b-preview) 降价 7.7%，[mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo) 降价 12.5%。
   - 社区成员戏称“持续的价格战”正在加剧供应商之间的竞争。
- **众包 AI 技术栈备受关注**：风险投资公司发布了各种生态系统图谱，但人们正推动一种真正的**众包**和**开源**方法，如[此 GitHub 项目](https://github.com/daytonaio/ai-enablement-stack)所示。
   - 一位用户请求对提议的逻辑提供反馈，鼓励社区“为动态开发者资源做出贡献”。
- **DeepSeek 加速代码学习**：开发者使用 **DeepSeek V2** 和 **DeepSeek V2.5** 解析整个 GitHub 仓库，报告称在项目范围的优化方面有显著提升。
   - 然而，一位用户警告说“它可能无法处理高级代码生成”，但仍对其注释能力表示赞赏。
- **对编程式 API Key 的呼吁**：讨论中提到允许在请求中隐式发送 **provider API key**，以简化集成。
   - 一位用户表示“我很想看到一个编程式版本”，以提高开发者的整体便利性。



---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **GitHub Copilot 推出免费版**：Microsoft 推出了新的 [GitHub Copilot 免费层级](https://x.com/code/status/1869449373995708703)，立即面向所有用户开放。
   - 令人惊喜的是，它包含了 **Claude** 以提供更强大的能力，且无需信用卡。
- **Granite 3.1-8B-Instruct 备受青睐**：开发者们称赞了 [Granite 3.1-8B-Instruct 模型](https://huggingface.co/ibm-granite/granite-3.1-8b-instruct)在长上下文任务中的强劲表现。
   - 它能为实际案例提供快速结果，IBM 在 [GitHub](https://github.com/ibm-granite/granite-3.1-language-models) 上提供了代码资源。
- **LM Studio 支持本地 LLM 选择**：[LM Studio](https://lmstudio.ai/) 简化了离线运行 Llama、Mistral 或 Qwen 模型的过程，同时支持从 Hugging Face 下载文件。
   - 用户还可以快速与文档进行对话，吸引了那些希望使用离线方式的人群。
- **微调中使用统一指令引发辩论**：关于在问答数据集中对每个 prompt 使用相同指令的做法引发了疑问。
   - 有人提出警告，由于重复使用，这可能会导致**模型性能欠佳**。
- **Genesis 项目凭借生成式物理学大放异彩**：[Genesis 引擎](https://x.com/zhou_xian_/status/1869511650782658846)构建 **4D 动力学世界**的速度比实时快 430,000 倍。
   - 它是[开源的](https://github.com/Genesis-Embodied-AI/Genesis)，运行在 Python 环境中，在单张 RTX4090 上将机器人训练时间缩短至仅 26 秒。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Mojo 中的负索引之争**：关于在 **Mojo** 中采用负索引（negative indexing）引发了激烈讨论，一些人认为它是错误诱因，而另一些人则认为这是 **Python** 中的标准做法。
   - 反对者更倾向于使用 `.last()` 方法以避免开销，并警告负偏移量可能带来的性能问题。
- **Dict 中 SIMD 键引发崩溃**：基于 **SIMD** 的结构体键（struct keys）中的一个严重 Bug 触发了 **Dict** 使用中的段错误（segmentation faults），详见 [GitHub Issue #3781](https://github.com/modularml/mojo/issues/3781)。
   - 缺失 **scaling_cur_freq** 导致了这些崩溃，促使官方在 **6 周**窗口内进行修复。
- **Mojo 在 Android 上“野蛮生长”**：爱好者们尝试通过基于 Docker 的黑客手段在原生 Android 上运行 **Mojo**，尽管这被标记为“完全不支持”。
   - 许可规则禁止发布 Docker 镜像，但本地自定义构建仍然可行。
- **Python 集成探索 SIMD 支持**：参与者讨论了将 **SIMD** 和条件一致性（conditional conformance）与 Python 类型合并，平衡对整数和浮点数据的分别处理。
   - 他们强调了 **ABI** 约束和未来的位宽扩展，激发了对跨语言交互的兴趣。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **合成数据解释指南受到关注**：一位贡献者正在编写关于**合成数据（synthetic data）**生成方式的解释指南，并征求社区对难点领域的意见。
   - 他们计划重点介绍创建方法以及对高级模型性能的影响。
- **DataBricks 速率限制辩论**：参与者指出吞吐量费用高昂，呼吁在 DataBricks 中加入**速率限制器（rate limiter）**以防止过度使用。
   - 一些人建议使用 [LiteLLM 代理层](https://example-url-for-lighter-llm.com)进行用量跟踪，同时也参考了 [Mosaic AI Gateway](https://docs.databricks.com/en/ai-gateway/index.html) 作为补充方案。
- **dspy.Signature 作为类使用**：一位用户询问关于以类（class）形式返回 **dspy.Signature** 的问题，旨在获得结构化输出而非原始字符串。
   - 他们希望定义明确的字段以提高清晰度并实现潜在的类型检查。
- **预置吞吐量让钱包缩水**：一场对话揭露了 DataBricks 中**预置吞吐量（provisioned throughput）**在保持激活状态时的高昂费用。
   - 成员们建议使用“**缩减至 0（scale to 0）**”功能，以遏制空闲期间的成本。
- **LiteLLM 接入 DataBricks**：与会者讨论了是将 **LiteLLM 代理**嵌入 DataBricks notebook 还是独立运行。
   - 他们一致认为，考虑到环境控制和资源需求，整合这两种方法是可行的。

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **LlamaIndex 的 Multi-Agent 改造**：一篇文章描述了从 **single agent** 到 **multi-agent system** 的飞跃，并提供了 **LlamaIndex** 中的实际代码示例，参考[此链接](https://t.co/lbhFDbSabS)。
   - 它还阐明了 **agent factories** 如何管理多个协同工作的任务。
- **Vectara 的 RAG 动态**：更新展示了 **Vectara** 的 RAG 优势，包括数据加载和基于流式的查询，参考[此链接](https://t.co/traVaQiUt3)。
   - 它强调了 RAG 方法的 Agentic 使用，并对托管环境中的 reranking 提出了见解。
- **Vercel 的 AI 调查呼吁**：敦促社区成员填写 **Vercel** 的 State of AI Survey，详见[此处](https://t.co/O3sYZ6L9Gq)。
   - 他们计划收集有关开发者经验、挑战以及未来 AI 改进目标领域的数据。
- **用于 PDF 处理的 Vision Parse**：介绍了一个新的开源 Python 库 [Vision Parse](https://github.com/iamarunbrahma/vision-parse)，用于使用先进的 Vision Language Models 将 PDF 转换为结构良好的 Markdown。
   - 参与者赞扬了其简化文档处理的潜力，并欢迎为集体成长而进行的开源努力。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Nomic 的数据映射系列马拉松结束**：**Data Mapping Series** 的最后一篇在 [Nomic 的博客文章](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers)中重点介绍了针对 embeddings 和非结构化数据的**可扩展图形**。
   - 这个分为六部分的系列展示了 **dimensionality reduction** 等技术如何使用户能够在 Web 浏览器中可视化海量数据集。
- **BERT 和 GGUF 故障已修复**：在一次 commit 破坏了功能后，用户在从 Huggingface 加载 **Nomic 的 BERT** embedding 模型时遇到了问题，但现在修复程序已上线。
   - 社区成员还指出了 **.GGUF** 文件中的 chat template 问题，并承诺在即将发布的版本中提供更新版本。
- **Code Interpreter 和系统加载器亮点**：一个 [pull request](https://github.com/nomic-ai/gpt4all/pull/3173) 提议了一个基于 jinja template 构建的 code interpreter 工具，用于运行高级代码任务。
   - 同时，用户请求一个更方便的系统消息加载器，以避免手动复制粘贴大量的上下文文件。
- **GPT4All 设备规格确认**：关于 **GPT4All** 系统要求的问题引导至一个详细说明[硬件支持](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md)的链接。
   - 重点强调了重要的 CPU、GPU 和内存细节，以确保稳定的本地 LLM 体验。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **TinyChat 安装纠纷**：一位用户在设置 TinyChat 时遇到问题，报告缺少 **tiktoken** 等组件、系统冻结 30 秒，以及关于本地网络设备的令人费解的提示。
   - George Hotz 谈到了在 TinyGrad 中编写 **tiktoken** 替代方案，并将 **8GB RAM** 标记为限制因素。
- **Mac 滚动方向异常**：一位用户抱怨运行 TinyChat 翻转了他们 Mac 上的**滚动方向**，然后在应用关闭后恢复。
   - George Hotz 称这种行为令人莫名其妙，承认这是一个奇怪的故障。
- **Bounty 推动与布局讨论**：贡献者讨论了推动 tinygrad 发展的 **bounty** 奖励，强调测试和改进是关键驱动力。
   - 一位用户提到了布局符号 (layout notation) 的复杂性，并链接到 [view merges 文档](https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit)和 [viewable_tensor.py](https://github.com/unknownusername504/MicroGrad/blob/main/micrograd/tensors/viewable_tensor.py) 以获取更深层次的背景。
- **#learn-tinygrad 中的 Scheduler 查询**：一位参与者询问为什么 scheduler 在 expand 或 unsafe pad ops 之前使用 **realize**，但未得到明确解释。
   - 小组没有完全展开讨论其原因，使该话题保持开放以供进一步探索。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Ikuo 令人印象深刻 & 礼仪随之而来**：Ikuo618 介绍了自己，他拥有六年的 **DP**、**NLP** 和 **CV** 经验，并重点展示了他的 **Python**、**TensorFlow** 和 **PyTorch** 技能。
   - 随后出现了一个温馨提示，建议成员不要在不同频道重复发布消息，以保持对话流程整洁。
- **平台功能疑问**：一位用户询问了平台上某个功能的可用性，一名成员确认该功能尚未上线。
   - 询问者表示感谢，并以一个笑脸符号愉快地结束了对话。
- **Cohere 密钥与速率限制揭晓**：Cohere 提供评估和生产 **API** 密钥，详见 [API 密钥页面](https://dashboard.cohere.com/api-keys) 和 [定价文档](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work)。
   - 速率限制包括：试用版每分钟 **20 次调用**，生产版在 **Chat** 端点每分钟 **500 次调用**，**Embed** 和 **Classify** 共享不同的配额。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **Torchtune 预告 Phi 4 与角色**：在 [Torchtune 官方文档页面](https://pytorch.org/torchtune/stable/api_ref_models.html) 中，成员确认 **Torchtune** 目前仅支持 **Phi 3**，但欢迎对 **Phi 4** 的贡献。
   - 他们引入了 Discord 上的 **Contributor** 角色，并指出 **Phi 3** 和 **Phi 4** 之间的差异极小，以简化新的 **pull requests**。
- **异步 RLHF 飞速发展**：**Asynchronous RLHF** 将生成和学习分离，以实现更快的模型训练，详见 [“Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models”](https://arxiv.org/abs/2410.18252)。
   - 该论文探讨了 *我们可以容忍多大程度的离策性 (off-policyness)*，在不牺牲性能的情况下追求速度。
- **训练后 (Post-Training) 势头强劲**：[Allen AI 博客](https://allenai.org/blog/tulu-3) 强调，在预训练之后，**post-training** 对于确保模型安全地遵循人类指令至关重要。
   - 他们概述了指令微调步骤，并专注于在专业化的同时保留中间推理等能力。
- **指令微调的平衡木**：**InstructGPT** 风格的策略可能会无意中削弱某些模型能力，特别是如果专业任务掩盖了更广泛的用途。
   - 在处理 *诗意或通用指令* 的同时保持 **coding** 熟练度，成为了需要维持的微妙平衡。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **LLM Agents 黑客松倒计时**：黑客松的 **提交截止日期** 为 **12/19 11:59 PM PST**，参赛作品需通过 [官方提交表单](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform) 提交。
   - 社区正处于 *最后修复* 的待命状态，确保每个人在时间截止前都有公平的机会。
- **最后时刻的 LLM 问题支持**：参与者可以在聊天中提出 **最后时刻的问题**，以获得同伴的快速反馈。
   - 组织者敦促开发者及时完成检查，避免在最后关头进行 *疯狂的合并 (merges)*。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Axolotl AI Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**LAION Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1319100971997073449)** (352 条消息🔥🔥): 

> `Unsloth 的 Multi-GPU 支持、Llama 3.3 微调、SGLang vs. vLLM、销售策略、FFT 支持` 


- **Unsloth 将在 Q1 提供 Multi-GPU 支持**：Unsloth 的 Multi-GPU 支持已在计划中，预计将于第一季度（Q1）发布，目前正在进行测试。
   - 团队在完善该功能的同时，正在评估定价和许可方案。
- **微调 Llama 3.3 的要求**：根据 Unsloth 博客，微调 Llama 3.3 大约需要 41GB 的 VRAM。
   - 该模型在经过适当微调后，与之前的版本相比显示出显著的性能提升。
- **SGLang 与 vLLM 性能对比**：社区讨论了 SGLang 和 vLLM，共识是 vLLM 通常在生产推理任务中提供更好的吞吐量。
   - SGLang 被认为在结构化输出方面很有用，而 vLLM 在其他领域提供了更强的性能。
- **销售策略与产品可用性**：随着对企业级解决方案兴趣的增长，用户呼吁 Unsloth 提供更精简的销售流程。
   - 虽然企业版产品仍处于 beta 阶段，但团队旨在评估需求并相应调整其销售策略。
- **FFT 引擎支持**：Unsloth 目前不支持 FFT，但用户可以手动实现。
   - 讨论强调，与其他训练引擎相比，利用 FFT 可以提供显著的性能改进。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmsys.org/blog/2024-07-25-sglang-llama3/">使用 SGLang 运行时实现更快的开源 Llama3 服务 (对比 TensorRT-LLM, vLLM) | LMSYS Org</a>: &lt;p&gt;在 LMSYS.org，我们运行 &lt;a href=&quot;https://chat.lmsys.org/&quot;&gt;Chatbot Arena&lt;/a&gt; 平台已有一年多，为数百万用户提供服务。我们深知...</li><li><a href="https://lmsys.org/blog/2024-12-04-sglang-v0-4/">SGLang v0.4：零开销批调度器、缓存感知负载均衡器、更快的结构化输出 | LMSYS Org</a>: &lt;p&gt;我们很高兴发布 &lt;a href=&quot;https://github.com/sgl-project/sglang&quot;&gt;SGLang v0.4&lt;/a&gt;，其具有显著的性能改进和新功能：...</li><li><a href="https://unsloth.ai/blog/contpretraining">使用 Unsloth 进行持续 LLM 预训练</a>: 使用 Unsloth 通过 Llama 3、Phi-3 和 Mistral 进行持续预训练，让模型学习一种新语言。</li><li><a href="https://unsloth.ai/blog/llama3-3">使用 Unsloth 微调 Llama 3.3</a>: 微调 Meta 的 Llama 3.3 (70B) 模型，其性能优于 GPT 4o，通过 Unsloth 开源实现 2 倍提速！对初学者友好。现已支持 Apple 的 Cut Cross Entropy 算法。</li><li><a href="https://huggingface.co/tiiuae/Falcon3-10B-Instruct">tiiuae/Falcon3-10B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/rombodawg/Rombos-LLM-70b-Llama-3.3">rombodawg/Rombos-LLM-70b-Llama-3.3 · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/basics/continued-pretraining">持续预训练 | Unsloth 文档</a>: 又名持续微调。Unsloth 允许你进行持续预训练，以便模型学习新语言。</li><li><a href="https://huggingface.co/unsloth/Llama-3.3-70B-Instruct">unsloth/Llama-3.3-70B-Instruct · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1319042969818632222)** (117 条消息🔥🔥): 

> `Adapters vs Models, 微调挑战, 微调学习资源, 指令微调的局限性, 模型合并技术` 


- **理解 Adapters 和模型**：Adapters，特别是 Low Rank Adapters (LoRAs)，通过修改模型参数的一个很小的子集，实现了灵活的组合，而无需对整个模型进行重新训练。
   - 为了组合它们，可以使用 [GGUF 选项](https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf) 保存模型以简化推理，或者将它们作为独立文件进行管理。
- **应对微调障碍**：微调并非简单的点击按钮；它需要理解底层过程，以避免诸如灾难性遗忘（catastrophic forgetting）之类的问题。
   - 成员们强调，成功的微调取决于找到调整的正确平衡点，并且通常涉及多个重新训练周期。
- **推荐的学习资源**：建议参考 [DeepLearning.ai](https://www.deeplearning.ai/) 和 Hugging Face 文档，以深入学习微调和模型训练。
   - 参与者强调了除了微调技术之外，拥有扎实的基础理解的重要性。
- **关于指令微调的见解**：一篇富有洞察力的论文指出，指令微调（Instruction Tuning）通常无法增强模型知识，甚至可能导致知识退化。
   - 成员们指出，对外部数据集的依赖可能会降低回答质量，从而再次强调了微调所涉及的复杂性。
- **探索模型合并技术**：尝试合并模型可能会产生不同的结果，因为保持平衡对于克服各种技术之间的权衡至关重要。
   - 合并技术（包括结合基础指令和 LoRA 调整）需要仔细管理，以避免精度损失等常见陷阱。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/basics/saving-and-using-models/saving-to-gguf">Saving to GGUF | Unsloth Documentation</a>: 将模型保存为 16bit 的 GGUF 格式，以便你可以将其用于 Ollama, Jan AI, Open WebUI 等工具！</li><li><a href="https://huggingface.co/settings/tokens">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2402.05119">A Closer Look at the Limitations of Instruction Tuning</a>: 指令微调 (IT) 是使用指令-响应对训练大型语言模型 (LLMs) 的过程，已成为将预训练基座 LLMs 转换为开放式...</li><li><a href="https://civitai.com/models/555285/miyabi-hoshimi-zenless-zone-zero">Miyabi Hoshimi (星見雅) (星见雅) - Zenless Zone Zero (绝区零) (絕區零) (ゼンレスゾーンゼロ) - booru | Stable Diffusion LoRA | Civitai</a>: 在 facebook.com/Kaiseir patreon.com/Serkai https://ko-fi.com/kaiseir 支持我。权重：1.0 触发词：Appearance: miyabihoshimi, &amp;lt;lora:miya...</li><li><a href="https://youtu.be/3UQ7GY9hNwk?si=FdoeDFWvqVzv9TMY"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1319086227516817468)** (468 条消息🔥🔥🔥): 

> `Fine-tuning LLMs, RAG (Retrieval-Augmented Generation), Quantization, 使用 Google Colab 和 Kaggle 进行模型训练, 模型的 JSON 数据格式化` 


- **Fine-tuning LLMs 的挑战**：一位用户强调了由于硬件限制在 Fine-tuning LLMs 时遇到的困难，特别是使用 TinyLlama 处理 1GB JSON 格式的大型数据集时。
   - 尽管困难重重，但在修复环境和更好地理解训练过程方面取得了进展。
- **引入 RAG 以增强学习**：强调了 Retrieval-Augmented Generation (RAG) 的重要性，认为它可能比直接进行 Fine-tuning 更有效，尤其是在针对特定任务使用较小模型时。
   - 参与者讨论了使用数据分块 (chunking) 和 Embedding 等技术来提高模型性能并降低初始训练的复杂性。
- **利用 Quantization 提高资源利用率**：讨论了 Quantization 技术作为降低模型训练时内存和计算成本的一种方式，支持 4-bit 或 8-bit 表示等更大的模型尺寸。
   - 建议用户使用正确的 Quantization 设置，以避免在训练期间导致本地机器崩溃。
- **利用在线平台进行训练**：推荐将 Google Colab 和 Kaggle 作为无需巨额费用即可获取 GPU 资源的替代方案，特别是对于本地计算能力有限的用户。
   - 尽管对使用云平台存在抵触情绪，但参与者承认它们在初始学习和模型测试方面的实用性。
- **处理 JSON 数据格式**：正确格式化 JSON 数据被认为是模型训练成功的关键步骤，但参与者在处理大型数据集时面临挑战。
   - 改进 JSON 文件的结构和格式被认为是有效利用 RAG 和为 Fine-tuning 工作做准备的必要条件。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>: Unsloth 新手？从这里开始！</li><li><a href="https://obsidian.md/">Obsidian - 磨砺你的思维</a>: Obsidian 是一款私密且灵活的笔记应用，能够适应你的思维方式。</li><li><a href="https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/LORA.md">mlx-examples/llms/mlx_lm/LORA.md at main · ml-explore/mlx-examples</a>: MLX 框架中的示例。通过在 GitHub 上创建账户，为 ml-explore/mlx-examples 的开发做出贡献。</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLMs</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。</li><li><a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 以 2-5 倍的速度和减少 70% 的内存 Fine-tuning Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs</a>: 以 2-5 倍的速度和减少 70% 的内存 Fine-tuning Llama 3.3, Mistral, Phi, Qwen 2.5 &amp; Gemma LLMs - unslothai/unsloth
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1319031452230221824)** (706 条消息🔥🔥🔥): 

> `Cursor 0.44.4 发布，O1 对比 Sonnet 3.5 性能，网站生成器对比自定义代码，Gemini-1206 能力，大学对初创公司的作用` 


- **Cursor 0.44.4 发布**：频道讨论了最近发布的 Cursor 0.44.4 版本，详细介绍了包括 Agent 增强和 Yolo 模式在内的多项新功能和改进。
   - 用户报告称 0.44.4 中的 Agent 性能更佳，指出其运行命令和处理任务的效率更高。
- **关于 O1 和 Sonnet 3.5 的讨论**：对话集中在 O1（每次请求价格约为 40 美分）及其与 Sonnet 3.5 相比的价值，用户分享了关于两者有效性的不同看法。
   - 一些用户认为 Sonnet 3.5 已能满足需求，对 O1 是否物有所值表示怀疑。
- **关于网站开发工具的看法**：关于使用 Framer 等网站生成器与从零开始编码的辩论兴起，强调了节省时间与成本之间的权衡。
   - 虽然一些人欣赏网站生成器的效率，但另一些人认为自定义编码提供了更多的灵活性和控制力。
- **Gemini-1206 的能力**：有人询问了用户使用 Gemini-1206 的体验，一些人对其功能和潜在益处表示感兴趣。
   - 然而，其他人仍然关注 Sonnet 3.5 等成熟模型在编程任务中的表现。
- **大学对初创公司的重要性**：讨论涉及了大学教育（特别是常春藤盟校）与追求创业项目之间的价值对比。
   - 参与者辩论了在创业背景下正规教育的必要性，并将其与实践经验和成功进行了权衡。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://svelte.dev/docs/kit/introduction">Introduction • Docs • Svelte</a>：未找到描述</li><li><a href="https://www.cursor.com/downloads">Downloads | Cursor - AI 代码编辑器</a>：选择您的平台以下载最新版本的 Cursor。</li><li><a href="https://svelte.dev">Svelte • 为所有人准备的 Web 开发</a>：未找到描述</li><li><a href="https://docs.astral.sh/uv/">uv</a>：未找到描述</li><li><a href="https://forum.cursor.com/t/i-can-not-delete-history-of-composer-and-chat/36026">我无法删除 composer 和聊天历史</a>：描述 Bug，我无法删除 composer 和聊天历史。重现步骤：创建一个新的 composer 和聊天。删除按钮仅清除聊天历史，而不是整个聊天。它无法被...</li><li><a href="https://simonwillison.net/2024/Dec/16/webdev-arena/">WebDev Arena</a>：来自 [Chatbot Arena](https://lmarena.ai/) 团队（前身为 LMSYS）的新排行榜，这次专注于评估不同模型在“Web 开发”方面的表现——尽管它...</li><li><a href="https://x.com/_philschmid/status/1869639246434246966?s=46">来自 Philipp Schmid (@_philschmid) 的推文</a>：WTF？！全新的开源物理 AI 引擎简直疯狂！🤯 Genesis 是一款结合了超快速模拟与生成能力的新型物理引擎，旨在为机器人技术创建动态 4D 世界...</li><li><a href="https://x.com/btibor91/status/1869160332712960345">来自 Tibor Blaho (@btibor91) 的推文</a>：这是 ChatGPT 任务和自动化（“jawbone” - 开发中）的预览。引用 Tibor Blaho (@btibor91)：还记得 “Jawbone” 吗？它是 ChatGPT “Tasks” 的代号...</li><li><a href="https://github.com/richards199999/Thinking-Claude/blob/main/model_instructions/v5.1-extensive-20241201.md">Thinking-Claude/model_instructions/v5.1-extensive-20241201.md at main · richards199999/Thinking-Claude</a>：让你的 Claude 能够思考。通过在 GitHub 上创建账户为 richards199999/Thinking-Claude 的开发做出贡献。</li><li><a href="https://youtu.be/oFfVt3S51T4?si=MtUVbzYc6H231xyJ"> - YouTube</a>：未找到描述</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新和改进。</li><li><a href="https://cursor.directory/">Cursor Directory</a>：为您的框架和语言寻找最佳的 Cursor 规则
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1319049018655113227)** (65 条消息🔥🔥): 

> `Flex credits 结转, 在 Windows 中使用 repoprompt, 从 GitHub 集成功能, Codeium 扩展问题, Windsurf 用户体验` 


- **Flex credits 结转说明**：一位成员确认 **flex credits 会结转**，确保用户在支付后保留其额度。
   - 另一位成员也证实了这一点，提到他们在支付后使用量已重置。
- **寻求 Windows 上的 repoprompt 替代方案**：一位用户询问 Windows 上是否有 **repoprompt** 的等效工具，表现出对其环境中类似功能的兴趣。
   - 虽然没有提供直接的替代方案，但成员们鼓励探索各种选项并测试不同的配置。
- **使用 ChatGPT 集成 GitHub 功能**：一位成员表达了在使用 ChatGPT 将功能从一个 GitHub 分支集成到另一个分支时遇到的挑战，并询问是否有相关指南。
   - 建议包括寻找特定的 YouTube 频道和指南，以简化集成过程。
- **VSCode 中的 Codeium 扩展问题**：用户报告了 **Codeium 扩展** 在 VSCode 的 Jupyter notebooks 中不再支持自动补全的问题，而以前是可以的。
   - 一位成员还提到通过降级扩展版本来解决服务器断开连接的问题。
- **对 Windsurf 用户体验的不满**：一位用户对 Windsurf 处理大文件的方式表示沮丧，特别提到了它总是删除相同的代码行。
   - 他们认为 **cascade** 功能需要改进，但表现出不愿通过官方支持渠道提交 bug 工单。


  

---

### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1319044599091564596)** (509 条消息🔥🔥🔥): 

> `Windsurf 性能问题，Cline + Gemini 使用体验，Codeium 支持与功能，模型对比，AI 工具中的额度管理` 


- **更新后的 Windsurf 性能问题**：多位用户反映 Windsurf 最近几天的功能性有所下降，出现了文件编辑问题以及使用过程中的频繁报错。
   - 用户对软件的可靠性感到越来越沮丧，促使一些人考虑 Cursor 等替代工具。
- **Cline + Gemini 使用的正面反馈**：一些用户提到，与 Windsurf 相比，将 Cline 与 Gemini 2.0 结合使用能获得更好的编码结果，且功能运行更流畅。
   - 用户赞赏 Cline 的效率，特别是在重构和处理大型代码任务时表现出色且没有问题。
- **询问 Codeium 的支持与改进**：用户表达了希望 Codeium 提供更快速响应支持的愿望，并反映了现有问题缺乏近期更新或修复。
   - 社区热切希望看到与当前用户需求相匹配的改进和功能，以提升功能性。
- **AI 模型及其有效性的对比**：关于不同模型（包括 Claude Sonnet 和 Gemini）性能差异的讨论，强调了它们在特定任务中的效率各不相同。
   - 用户指出需要上下文信息和完善的文档来增强 AI 模型的使用效能。
- **关于额度管理和成本的担忧**：用户对 Windsurf 中的额度消耗及其对用户体验的影响表示担忧，特别是在处理大型任务时。
   - 用户正在评估不同方案的性价比，以及在 AI 工具中使用额度的影响。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits-but-not-premium-flow-action-credits">付费计划与额度使用 - Codeium 文档</a>: 未找到描述</li><li><a href="https://x.com/sdrzn/status/1869470308442452478">来自 Saoud Rizwan (@sdrzn) 的推文</a>: 很高兴分享 Cline v3 🎉 你现在可以自动批准所有操作，限制 API 请求的最大数量，并在任务完成时接收桌面通知！Cline 现在还使用灵活的 diff 编辑来...</li><li><a href="https://docs.codeium.com/windsurf/usage#what-happens-when-you-run-out-of-premium-user-prompt-credits">付费计划与额度使用 - Codeium 文档</a>: 未找到描述</li><li><a href="https://stateofai.tools/">2024 年 AI 工具现状 - 开发者调查</a>: 通过分享你的经验，帮助塑造 AI 加速开发的未来。</li><li><a href="https://www.builder.io/blog/ai-dev-skill">为什么 AI 让开发技能变得更有价值，而不是更低</a>: AI 不会取代开发者，它会让开发者更有价值。让我们来看看开发者的工作是如何演变的，以及它如何影响团队</li><li><a href="https://www.mcpservers.ai/servers/modelcontextprotocol/Sequential%20Thinking">MCP 服务器</a>: 浏览最大的 Model Context Protocol 服务器库。与他人分享你创建的 Model Context Protocol 服务器。</li><li><a href="https://docs.codeium.com/context-awareness/overview">概览 - Codeium 文档</a>: 未找到描述</li><li><a href="https://aider.chat/docs/leaderboards/">Aider LLM 排行榜</a>: LLM 代码编辑能力的定量基准测试。</li><li><a href="https://www.builder.io/blog/windsurf-vs-cursor">Windsurf vs Cursor：哪款 AI 代码编辑器更好？</a>: 对比 Windsurf 和 Cursor AI 驱动的 IDE：功能、用户体验和工作流效率。哪款最适合你？</li><li><a href="https://zed.dev/releases/stable/0.166.1">Zed - 面向未来的编辑器</a>: Zed 是一款高性能、多人的代码编辑器，由 Atom 和 Tree-sitter 的创作者打造。</li><li><a href="https://codeium.com/contact">联系方式 | Windsurf 编辑器与 Codeium 扩展</a>: 联系 Codeium 团队获取支持，并了解更多关于我们企业级产品的信息。</li><li><a href="https://github.com/modelcontextprotocol/servers/tree/main/src/git">modelcontextprotocol/servers 仓库中的 git 服务器</a>: Model Context Protocol 服务器。通过在 GitHub 上创建账号为 modelcontextprotocol/servers 做出贡献。</li><li><a href="https://youtu.be/54RUAzPYEeY"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1319163554619850834)** (193 条消息🔥🔥): 

> `Gemini 2.0 Flash Thinking, OpenAI 更新，研究员离职，搜索引擎竞争，推理模型`

- **Gemini 2.0 Flash Thinking 发布**：Google 推出了 [Gemini 2.0 Flash Thinking](https://x.com/NoamShazeer/status/1869789881637200228)，这是一款实验性模型，旨在推理时显式展示其思考过程，并承诺提升性能。
   - 该模型旨在结合速度与增强的推理能力，可能使 Google 在 AI chatbot 领域占据强有力的地位。
- **OpenAI 推出新的聊天功能**：OpenAI 在其 12 Days of ChatGPT [网站](https://openai.com/12-days/?day=11)上宣布，语音模式现已支持 “Work with Apps”，允许与 Notion 和 Apple Notes 等应用集成。
   - 这标志着 OpenAI 在增强其系统用户交互和功能方面又迈出了一步。
- **OpenAI 的重要人员离职**：著名研究员 @AlecRad 已离开 OpenAI，他被公认为 GPT、Whisper 和 DALL-E 等模型开发的关键人物。
   - 此次离职引发了人们对 OpenAI 未来领导层和发展方向的担忧。
- **搜索引擎的竞争格局**：@amir 报道称，Google 正在将其 Gemini chatbot 直接整合到搜索结果中，标志着搜索向对话式 AI 模式的战略转变。
   - 这引发了关于 Kagi 等竞争服务如何吸引寻求低商业化搜索体验的用户的问题。
- **关于 AI 模型推理的讨论**：参与者讨论了推理模型的有效性，强调如果模型能够无误地有效输出推理过程，那么自我修正（self-correction）可能并非必要。
   - 这突显了对于 AI 如何实现推理及其与传统搜索方法区别的持续探索。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/tsarnick/status/1869500847488692256">来自 Tsarathustra (@tsarnick) 的推文</a>：前 OpenAI 首席研究官 Bob McGrew 表示，o1 实际上就是 GPT-5，因为它代表了比 GPT-4 高出 100 倍的算力提升，而 GPT-4.5 的发布将是一个有趣的揭示，展示预训练...</li><li><a href="https://x.com/zhou_xian_/status/1869511650782658846">来自 Zhou Xian (@zhou_xian_) 的推文</a>：你对生成式模型所喜爱的一切——现在由真实的物理驱动！宣布 Genesis 项目——经过 24 个月、涉及 20 多个研究实验室的大规模研究合作——一个生成式...</li><li><a href="https://x.com/denny_zhou/status/1869771028693713284">来自 Denny Zhou (@denny_zhou) 的推文</a>：树搜索（Tree search）作为经典 AI 的核心思想，与真正的智能或推理几乎没有关系，无论搜索能很好地解决哪些有趣的益智游戏（例如 24 点游戏）。搜索仅仅是一种工具的使用。S...</li><li><a href="https://arxiv.org/abs/2412.13663">更聪明、更好、更快、更长：一种用于快速、内存高效且长上下文微调与推理的现代双向编码器</a>：Encoder-only Transformer 模型（如 BERT）在检索和分类任务中，相比于更大的 Decoder-only 模型，提供了极佳的性能与尺寸权衡。尽管作为...</li><li><a href="https://x.com/testingcatalog/status/1869810186648740153">来自 TestingCatalog News 🗞 (@testingcatalog) 的推文</a>：重大新闻 🚨：OpenAI 在语音模式中引入了 “Work with Apps” 支持，同时支持 Notion、Apple Notes 等应用 👀 引用 OpenAI (@OpenAI) 第 11 天：一种与 ChatGPT 协作的新方式 http...</li><li><a href="https://x.com/altryne/status/1869571717368267092">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：@arcprize o1 preview 达到了 21% 🔥 这真是一个巨大的飞跃</li><li><a href="https://x.com/wintermoat/status/1869784711121514620">来自 Alphabetting (@wintermoat) 的推文</a>：@TheXeophon 现在就在 AI Studio 中！</li><li><a href="https://x.com/Presidentlin/status/1869745206842794047">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：@Alibaba_Qwen 他们做了一个推理 VL 模型？</li><li><a href="https://x.com/NoamShazeer/status/1869789881637200228">来自 Noam Shazeer (@NoamShazeer) 的推文</a>：我们一直在“思考”如何改进模型的推理能力和可解释性。介绍 Gemini 2.0 Flash Thinking，这是一个经过训练可以“大声思考”的实验性模型，从而带来了更强的推理性能...</li><li><a href="https://x.com/amir/status/1869837622627184865">来自 Amir Efrati (@amir) 的推文</a>：新消息：Google 实际上正在将其 Gemini 聊天机器人直接添加到搜索结果中——“AI Mode”。创新者困境依然存在，但这表明 Google 正在认真对待对话式聊天机器人的性能...</li><li><a href="https://x.com/btibor91/status/1869784134224359709">来自 Tibor Blaho (@btibor91) 的推文</a>：最适用于：多模态理解、推理、编程。用例：对最复杂的问题进行推理、展示模型的思考过程、解决困难的代码和数学问题。知识截止日期：A...</li><li><a href="https://www.interconnects.ai/p/openais-o1-using-search-was-a-psyop">OpenAI 的 o1 使用“搜索”是一个心理战（PSYOP）</a>：如何将 OpenAI 的 o1 模型理解为一个古怪、奇妙且漫长的思维链（Chain of Thought）</li><li><a href="https://fxtwitter.com/arcprize/status/1869551373848908029">来自 ARC Prize (@arcprize) 的推文</a>：o1 在 ARC-AGI 半公开评估（100 个任务）中的验证性能：o1, Low: 25% ($1.5/任务)；o1, Medium: 31% ($2.5/任务)；o1, High: 32% ($3.8/任务)</li><li><a href="https://x.com/natolambert/status/1869802093856612657">来自 Nathan Lambert (@natolambert) 的推文</a>：加油 Google，给我看看你们内部实验的测试时扩展（test time scaling）图表。这是 RL 可信度所需的文档。</li><li><a href="https://x.com/justinlin610/status/1869793885540757715?s=46">来自 Junyang Lin (@JustinLin610) 的推文</a>：非常抱歉让大家久等了。今晚不会发布。我们仍需改进此次发布的内容。很快就会回来。</li><li><a href="https://x.com/anton_lozhkov/status/1869771053146464507?t=J1oHcOrr0APg0r9b1mP3tQ&s=19">来自 Anton Lozhkov (@anton_lozhkov) 的推文</a>：介绍 📐FineMath：拥有 50B+ token 的最佳开源数学预训练数据集！数学对 LLM 来说仍然具有挑战性，通过在 FineMath 上进行训练，我们看到了比其他数学数据集更显著的提升，特别是...</li><li><a href="https://x.com/swishfever/status/1869774920164778170">来自 fishy business (@swishfever) 的推文</a>：数据挖掘出的字符串：“模型产生的想法是实验性的。” “p6ntest-ai-llm-prompt-config-thinking-model-disclaimer” 引用 Logan Kilpatrick (@OfficialLoganK) 🤔</li><li><a href="https://x.com/amir/status/1869847852308205935">来自 Amir Efrati (@amir) 的推文</a>：新闻：另一位 OpenAI 关键研究员 @AlecRad 离职。他是 GPT 论文的第一作者，对 Whisper 和 Dall-E 的开发起到了至关重要的作用...</li><li><a href="https://aistudio.google.com/">https://aistudio.google.com/</a></li>

u/2/prompts/new_chat?pli=1">没有找到标题</a>: 没有找到描述</li><li><a href="https://x.com/lmthang/status/1869797423763341448">来自 Thang Luong (@lmthang) 的推文</a>: @GoogleDeepMind 今年的最后一次发布？不确定 :) 但很高兴能成为发布 Gemini 2.0 Flash Thinking 团队的一员，这是一个既聪明又快速的模型！欢迎来到思考时代....</li><li><a href="https://x.com/vikhyatk/status/1869605301596631191">来自 vik (@vikhyatk) 的推文</a>: 每个人都在发关于这个的 🤯 推文，但我敢说没人试过。因为我试了，它说 genesis 模块上不存在 generate 方法。引用 Allen T. (@Mr_AllenT) 这是最...</li><li><a href="https://github.com/googleapis/python-genai/blob/3e42644784304d45d0b0bfdc8279958109650576/google/genai/tests/models/test_generate_content_thought.py">python-genai/google/genai/tests/models/test_generate_content_thought.py at 3e42644784304d45d0b0bfdc8279958109650576 · googleapis/python-genai</a>: Google Gen AI Python SDK 为开发者提供了一个接口，用于将 Google 的生成式模型集成到他们的 Python 应用程序中。这是一个早期版本。API 可能会发生变化。请...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1319190817877004298)** (16 messages🔥): 

> `o1 模型讨论, Chollet 的类比, Subbarao/Miles Brundage 事件, Francois Chollet 的坏脾气, Interconnects 参与度` 


- **Chollet 对 o1 作为 LLM 的看法**：François Chollet 声称将 o1 标记为“一个 LLM”类似于将 AlphaGo 称为“一个 convnet”，这引发了成员之间的辩论。
   - 虽然有些人引用 AlphaGo 对 MCTS 和神经网络的依赖来挑战 Chollet，但许多人对 o1 的运行原理表示困惑。
- **Francois Chollet 的坏脾气名声**：成员们幽默地注意到 Chollet 在讨论 o1 模型及其与既有模型的比较时表现出的坏脾气。
   - 评论强调了对 o1 更好清晰度的渴望，并建议 OpenAI 的人应该向 Chollet 解释其功能。
- **回顾 Subbarao/Miles Brundage 事件**：讨论提到了 Subbarao/Miles Brundage 事件，强调了 o1 主要作为语言模型运行的观点。
   - 一位成员引用了这一事件，认为这反映了社区对模型部署更广泛的误解。
- **征集关于“Oneshotting Turbonormies”的梗图**：一位成员表示需要一个与“oneshotting turbonormies”相关的梗图（Meme），这表明讨论中持续存在的梗文化。
   - 有人对在需要时无法快速找到该梗图表示沮丧。
- **与 Interconnects 的互动**：成员们讨论了阅读 Interconnects 内容的价值，并建议向 Chollet 回复相关讨论的链接。
   - 对话强调了对紧跟社区内快节奏辩论的幽默看法。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/tszzl/status/1869681557340086602)">来自 roon (@tszzl) 的推文</a>: @rao2z @Miles_Brundage 但已部署的产品如何工作或模型如何推理并不是一个真正的科学问题。o1 只是一个语言模型</li><li><a href="https://x.com/fchollet/status/1869612195425972715?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 François Chollet (@fchollet) 的推文</a>: 将 o1 之类的东西称为“一个 LLM”，其准确性大约相当于将 AlphaGo 称为“一个 convnet”</li><li><a href="https://fxtwitter.com/fchollet/status/1869854758443557020">来自 François Chollet (@fchollet) 的推文</a>: 对于那些不理解的人——AlphaGo 是一个 MCTS 搜索过程，它为了计算单次走棋而对两个独立的 convnets 进行了数千次调用。像 o1 pro 这样的东西也是，据我们所知...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1319036036470407319)** (167 条消息🔥🔥): 

> `Stripe Tax 实施、Substack 营收模式与税务担忧、税务申报的 CPA 建议、数字服务与 VAT 合规、国际税务挑战` 


- **Stripe Tax 作为安全网**：讨论强调了为数字服务启用 [Stripe Tax](https://stripe.com/tax) 以简化税务合规的重要性，特别是对于接近营收阈值的 Substack 创作者。
   - *开启此功能可以避免日后与税务机关产生潜在的麻烦。*
- **关于 Substack 税务处理的困惑**：参与者对 Substack 如何处理税务表示不确定，讨论集中在 Substack 是否被视为负责代扣代缴税款的市场运营商。
   - Nate 指出，由于款项直接进入 Substack 的 Stripe 账户，这使得创作者的税务情况变得复杂。
- **向大型 Substack 创作者学习**：Nate 注意到，即使是规模较大的 Substack 创作者似乎也缺乏对纳税义务的了解，这表明该领域的创作者中可能存在一种普遍趋势。
   - 这引发了关于申报收入和税收时问责制和责任等更广泛问题的讨论。
- **CPA 与税务建议**：几位成员建议咨询 CPA，以获取有关处理税务要求的指导，特别是针对数字服务业务。
   - Nate 提到他伴侣的母亲是一位 CPA，并表示有兴趣收集更多建议以确保合规。
- **国际税务挑战**：讨论了在欧洲管理 VAT 的挑战，以及个人或企业如何在国际背景下应对潜在的税务责任。
   - 一位成员幽默地指出，不合规可能会导致严重后果，这表明了这些税务问题的严肃性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://app.hex.tech/533fe68e-dcd8-4a52-a101-aefba762f581/app/b9dc830a-bd3f-4dc9-8495-3be64e735ce2/latest">vLLM 2024 分析</a>：Hex 是一个用于数据科学和分析的现代数据平台。协作式笔记本、精美的数据应用和企业级安全。</li><li><a href="https://help.kagi.com/kagi/faq/sales-tax-vat.html">账单 / 销售税 / VAT 常见问题解答 | Kagi 文档</a>：未找到描述</li><li><a href="https://support.substack.com/hc/en-us/articles/12282257442580-Does-Substack-integrate-with-Stripe-Tax">Substack 是否集成了 Stripe Tax？</a>：是的！如果您启用了 Stripe Tax，此功能可以帮助在 Substack 的销售点确定并计算正确的税额。在特定地区的交易会自动计算税费...</li><li><a href="https://www.regs2riches.com/p/substack-ed-against-sales-tax">🃏 substack-ed 针对销售税？</a>：💸 即使是颠覆者也有纳税义务</li><li><a href="https://open.substack.com/pub/faq/p/setting-up-vat-tax-compliance-for?r=68gy5&utm_medium=ios">为您的 Substack 设置 VAT 税务合规</a>：Substack 现在让欧洲作者能够轻松跟踪纳税义务</li><li><a href="https://stripe.com/tax">Stripe Tax | 通过单一集成实现税务自动化</a>：使用 Stripe Tax 自动化税务合规。通过单一集成轻松计算、收取和申报全球支付的销售税、VAT 和 GST。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1319190812940308551)** (2 条消息): 

> `游戏节目中的交互式 AI、社交媒体反应` 


- **ChatGPT 登上游戏节目：一个搞笑的转折**：一位成员开玩笑说在《谁想成为百万富翁》游戏期间拨打 **1-800-ChatGPT**，展示了 AI 在流行文化中日益增长的影响力。
   - 这个幽默的引用反映了 AI 正在不断融入日常场景和娱乐活动。
- **病毒式 AI 推文**：**voooooogel** 的一条推文引起了关注，虽然具体内容尚不明确，但暗示了引发讨论的 AI 相关内容。
   - 这种互动突显了社交媒体平台上围绕 AI 话题的好奇心和参与度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/Yuchenj_UW/status/1869799400681419122">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：当你参加《谁想成为百万富翁》并决定拨打 “1-800-ChatGPT” 时：</li><li><a href="https://x.com/voooooogel/status/1869529374829207884">来自 thebes (@voooooogel) 的推文</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/)** (1 messages): 

kevin_nejad: 这种行为纯粹源于 RL 训练，这很有趣（但并不显而易见）。
  

---


### **Interconnects (Nathan Lambert) ▷ #[lectures-and-projects](https://discord.com/channels/1179127597926469703/1223784028428177510/1319390383649853560)** (7 messages): 

> `RLHF Book, 拼写错误修正, 基础复习` 


- **通过 RLHF 复习缓解压力**：一位成员表示，在休假前花时间阅读 **RLHF Book** 并观看 YouTube 上的**长讲座**时，感到工作压力*减轻了*。
   - 他们发现复习 RLHF 的基础知识非常*治愈*，并强调了其重要性。
- **征集拼写错误修正**：有人提议向帮助修正拼写错误或格式问题的个人赠送 **RLHF Book** 的**免费副本**。
   - 另一位成员踊跃报名参与，表示凭借自己的英语水平非常适合这项工作。
- **社区参与 RLHF**：一位 RLHF 新手表示有兴趣帮助纠正书中的错误，并提到他们很喜欢*免费*的东西。
   - 社区参与意愿强烈，成员们都意识到了这一协作机会。
- **GitHub 上的 RLHF 资源**：一位成员指出 **RLHF Book** 的资源都可以在 **GitHub** 上找到，非常方便获取。
   - 这种开放性促进了社区的协作与贡献。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/)** (1 messages): 

natolambert: 是的。学生们走过来想合影的样子太可爱了 ❤️
  

---


### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1319361588431425627)** (1 messages): 

> `12 Days of OpenAI, ChatGPT 工作增强` 


- **ChatGPT 第 11 天活动启动**：**12 Days of OpenAI** 的第 11 天介绍了一种与 **ChatGPT** 协作的新方式，详情见 [YouTube 直播会话](https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz)。
   - 在活动期间，可以通过在 <id:customize> 中领取 <@&1261377106890199132> 身份组来*保持关注*。
- **参与 OpenAI 更新**：鼓励参与者通过关注最新进展和机会来参与 **12 Days of OpenAI** 活动。
   - 该计划让成员能够提升在使用 **ChatGPT** 及相关工具时的体验。



**提到的链接**: <a href="https://www.youtube.com/live/g_qxoznfa7E?si=q71WyiHuioBSGzvz"> - YouTube</a>: 未找到描述

  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1319038880808173598)** (310 条消息🔥🔥): 

> `ChatGPT 与集成，Google 的 AI 进展，YouTube 克隆项目，软件工程自动化，AI 基准测试与能力` 


- **ChatGPT 新增与 XCode 的集成**：用户讨论了 ChatGPT 如何通过允许用户直接将文本复制并粘贴到 XCode 中来促进代码开发，从而增强工作流。
   - 虽然此功能提供了便利，但仍需要用户手动输入，例如启动复制操作。
- **Google 发布实验性 AI 模型**：聊天参与者注意到 Google 最近发布了 Gemini 2.0 Flash Thinking 实验性模型，强调了其性能和公众的兴趣。
   - 对该模型的准确性存在怀疑，特别是在简单的任务中，如计算单词中的字母数量。
- **使用 ChatGPT 创建 YouTube 克隆版**：成员们对使用 ChatGPT 创建 YouTube 克隆版的前景充满热情，讨论了该模型处理前端和后端编码的能力。
   - 挑战在于后端构建所需的更复杂的终端操作，这被认为是过程中的一个复杂点。
- **AI 驱动下的软件工程未来**：参与者推测 AI 的进步如何可能自动化整个软件工程任务，从而影响对人类工程师的需求。
   - 自动化被视为既令人兴奋又令人担忧，这取决于尽管有 AI 能力，任务的复杂程度如何。
- **AI 性能基准测试**：社区提出了关于 AI 基准测试和模型性能的问题，特别是关于 Google 的新产品与现有产品的对比。
   - 参与者对模型的能力表示兴趣，但也持有怀疑态度，强调了关于 LLM 效率的持续讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/AlignAGI/Alignment/">GitHub - AlignAGI/Alignment: Promoting global awareness and action for ethical AI alignment and safeguarding humanity against AI self-replication risks. Includes research, frameworks, and open-source resources.</a>: 促进全球对伦理 AI 对齐的认识和行动，保护人类免受 AI 自我复制风险。包括研究、框架和开源资源。 - AlignAGI/Alig...</li><li><a href="https://www.youtube.com/watch?v=v-EYzZCLF48"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/AeMvOPkUwtQ?feature=shared"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1319055729478467654)** (8 条消息🔥): 

> `编辑 GPTs，项目文件夹限制，支持渠道，Pro Package 工具问题` 


- **编辑 GPTs 仍是一个谜团**：关于**编辑 GPTs** 的能力存在困惑，一名用户坚持认为他们可以编辑，而另一名用户则报告无法编辑。
   - *jerzjorge23* 表示，自最近的项目发布以来，他们只能创建新的 GPTs。
- **项目文件夹的限制**：*7_vit_7* 提到，由于潜在的附件文件会导致冲突，无法将 GPTs 移动到**项目文件夹**中。
   - *jerzjorge23* 澄清说他们并不是尝试移动文件，而只是想编辑它们。
- **寻求支持渠道**：*armantlrp* 询问了有关工具可用性方面的**支持渠道**。
   - 他们指出，包括 **canvas**、搜索和图片在内的多种工具在 Web 和 MacOS 版本上都无法使用。
- **Pro Package 工具问题持续存在**：*armantlrp* 的 **Pro Package** 工具已经连续几天无法使用。
   - 这引起了社区对可能影响 **Pro Package** 用户功能的持续问题的担忧。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1319078222301888605)** (146 条消息🔥🔥): 

> `FSDP 和 Tensor Parallelism, EleutherAI Token 争议, Natural Attention 优化器, 调试训练模型, Attention 中的 Causal Masking` 


- **FSDP 与 Tensor Parallelism 辩论**：成员们讨论了 **Fully Sharded Data Parallel (FSDP)** 与 **Tensor Parallelism** 之间的区别，指出 FSDP 在跨 GPU 维持操作的同时对参数进行分片（sharding）。
   - 一些人对 FSDP 的效率表示怀疑，因为与直接的 Tensor Parallel 实现相比，它的通信开销（communication overhead）更高。
- **揭穿 EleutherAI Token 谣言**：EleutherAI 没有任何关联的加密货币，成员们警告他人警惕近期出现的与非官方 Token 相关的诈骗。
   - 社区强调，投资此类 Token 类似于参与庞氏骗局。
- **Natural Attention 优化器介绍**：一位成员分享了关于一种新型 **Attention Informed Optimizer** 的见解，该优化器利用来自 Attention 机制的梯度来调整 Adam 优化算法。
   - 据称该优化器能显著提高性能，尽管结果的收敛情况引发了对实现中可能存在 Bug 的担忧。
- **模型训练调试中的挑战**：参与者讨论了模型训练中的故障排除问题，特别关注了一位参与者结果中异常低的 Loss 值。
   - 建议包括仔细检查 Causal Masking 函数，因为错误的实现可能导致误导性的训练指标。
- **Attention 中 Causal Masks 的重要性**：成员们强调了在 Attention 机制中使用 Causal Masks 的必要性，以防止未来的 Token 影响当前的预测。
   - 有人指出，忽视这一组件可能会导致模型性能和输出出现极端差异。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2310.01889">Ring Attention with Blockwise Transformers for Near-Infinite Context</a>: Transformer 已成为许多最先进 AI 模型的首选架构，在广泛的 AI 应用中展示了卓越的性能。然而，内存需求...</li><li><a href="https://pump.fun/coin/5CCtDehQTswpWzeYdxUWz7VS3bCrwV9o8ZfUyKgJpump">Eleuther Ai (eAI) - Pump</a>: 一家专注于人工智能可解释性、对齐和伦理的非营利研究实验室。</li><li><a href="https://github.com/jeroaranda/naturalattention">GitHub - jeroaranda/naturalattention</a>: 在 GitHub 上通过创建账户为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/papers/Natural_attention_proofs.pdf">naturalattention/papers/Natural_attention_proofs.pdf at main · jeroaranda/naturalattention</a>: 在 GitHub 上通过创建账户为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/meta-llama/llama-recipes/blob/main/src/llama_recipes/utils/config_utils.py#L95">llama-recipes/src/llama_recipes/utils/config_utils.py at main · meta-llama/llama-recipes</a>: 用于使用可组合的 FSDP 和 PEFT 方法微调 Meta Llama 的脚本，涵盖单/多节点 GPU。支持用于摘要和问答等应用的默认及自定义数据集。</li><li><a href="https://github.com/huggingface/transformers/blob/v4.47.1/src/transformers/models/gpt2/modeling_gpt2.py#L195>">transformers/src/transformers/models/gpt2/modeling_gpt2.py at v4.47.1 · huggingface/transformers</a>: 🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的最先进机器学习库。 - huggingface/transformers</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: 在 GitHub 上通过创建账户为 jeroaranda/naturalattention 的开发做出贡献。</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>: 一个用于语言模型 Few-shot 评估的框架。 - EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/jeroaranda/naturalattention/blob/main/natural_attention.py#L43>">naturalattention/natural_attention.py at main · jeroaranda/naturalattention</a>: 在 GitHub 上通过创建账户为 jeroaranda/naturalattention 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1319031842422259774)** (123 条消息🔥🔥): 

> `微软研究伦理, Koopman 理论与神经网络, Diffusion vs Autoregressive 模型, ML 研究中的剽窃担忧, 研究提交与监督`

- **对 Microsoft Research 伦理的担忧**：讨论强调了 Microsoft Research (MSR) 在伦理实践方面的问题，包括最近对其论文在未引用的情况下抄袭他人工作的指控。
   - 之前的争议（如 Phi 方法论）以及诚信度低的案例被提及，引发了对 MSR 整体伦理文化的质疑。
- **关于 Koopman Theory 应用的辩论**：成员们辩论了在神经网络背景下使用 Koopman theory 的有效性，一些人认为这种应用似乎很牵强，且没有产生明显的收益。
   - 有人对这类方法的底层理论依据表示担忧，认为它们可能会在无意中误导研究人员。
- **Diffusion 与 Autoregressive 模型之争**：针对 Diffusion 模型与 Autoregressive 方法在各种模态下的优缺点展开了讨论，特别是它们的效率以及对离散数据集的适用性。
   - 虽然 Diffusion 模型目前在图像生成领域占据主导地位，但人们对其在其他任务中相较于 Autoregressive 技术的长期可行性存在推测。
- **机器学习研究中的抄袭问题**：几位成员对近期机器学习研究论文中明显的抄袭行为表示担忧，特别是那些来自 MSR 等知名机构的论文。
   - 呼吁在研究实践中建立问责制和透明度，强调需要公众对不道德行为进行抵制。
- **研究提交与监管**：关于研究机构中不同监管结构的讨论引发了对研究诚信及其争议处理影响的质疑。
   - 成员们指出，MSR 的去中心化监管可能导致了伦理失范，并将其与其他机构中观察到的更中心化的方法进行了对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/zhou_xian_/status/1869511650782658846">来自 Zhou Xian (@zhou_xian_) 的推文</a>：你对生成模型所喜爱的一切——现在由真实的物理引擎驱动！宣布 Genesis 项目——经过 24 个月、涉及 20 多个研究实验室的大规模研究合作——一个生成式...</li><li><a href="https://tsb0601.github.io/metamorph/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/cloneofsimo/status/1869807463186472970?s=46">来自 Simo Ryu (@cloneofsimo) 的推文</a>：好吧，Lucas 提出了一个观点，也许它有效是因为那是一个 12 层的网络。所以我把它做成了 96 层网络（带有 768 个隐藏维度，哈哈）并进行了长达 10 小时的参数扫描。令我惊讶的是，差距反而扩大了...</li><li><a href="https://arxiv.org/abs/2112.00114">展示你的工作：用于语言模型中间计算的草稿本（Scratchpads）</a>：大型预训练语言模型在可以“一次性完成”的任务上表现得非常好，例如生成逼真的文本或合成计算机程序。然而，它们在...</li><li><a href="https://arxiv.org/abs/2305.20050">让我们逐步验证</a>：近年来，大型语言模型在执行复杂多步推理的能力上有了很大提高。然而，即使是最先进的模型仍然经常产生逻辑错误。这...</li><li><a href="https://arxiv.org/abs/1810.01479">用于 Koopman 的时延观测器：理论与应用</a>：非线性动力系统在科学和工程中无处不在，但这些系统的分析和预测仍然是一个挑战。Koopman 算子理论通过将...绕过了其中的一些问题。</li><li><a href="https://arxiv.org/abs/2212.07358">学习 Koopman 算子的不变子空间——第 1 部分：证明字典近似子空间不变性的方法论</a>：Koopman 算子将非线性动力学建模为作用于非线性函数（作为状态）的线性动力系统。这种非标准状态通常被称为 Koopman 观测器，通常被近似为...</li><li><a href="https://arxiv.org/abs/2206.07137">对可学习、值得学习且尚未学习的点进行优先训练</a>：在网络规模的数据上进行训练可能需要数月时间。但大部分计算和时间都浪费在已经学过或不可学习的冗余和噪声点上。为了加速训练，我们引入了 Reducib...</li><li><a href="https://github.com/Genesis-Embodied-AI/Genesis/tree/main/genesis/assets/meshes">Genesis/genesis/assets/meshes 分支 main · Genesis-Embodied-AI/Genesis</a>：一个用于通用机器人和具身智能（Embodied AI）学习的生成式世界。- Genesis-Embodied-AI/Genesis</li><li><a href="https://github.com/KellerJordan/modded-nanogpt?tab=readme-ov-file#muon-optimizer>">GitHub - KellerJordan/modded-nanogpt: 5 分钟内完成 NanoGPT (124M)</a>：5 分钟内完成 NanoGPT (124M)。通过在 GitHub 上创建账号为 KellerJordan/modded-nanogpt 的开发做出贡献。</li><li><a href="https://x.com/kellerjordan0/status/1869752568026689767>">来自 Keller Jordan (@kellerjordan0) 的推文</a>：我想向 Microsoft Research 最近发表的以下论文提出 Muon 的引用请求：Ma 等人 (2024)。SWAN：预处理 SGD 在 LLM 训练上实现了 Adam 级别的性能...</li><li><a href="https://x.com/hi_tysam/status/1869756590661992919>">来自 Fern (@hi_tysam) 的推文</a>：Keller 表现得很礼貌（这是恰当的）。但在我看来，这似乎是相当恶劣的抄袭，这在 @MSFTResearch 身上发生很奇怪。它非常公然地抄袭了来自 Shamp... 的概念。</li><li><a href="https://x.com/HessianFree/status/1869781347696550178>">来自 Omead Pooladzandi (@HessianFree) 的推文</a>：看到 @MSFTResearch 的人做出这样的事情真的很令人沮丧。梯度白化（Gradient whitening）并不是新鲜事。参见 Amari 1998, LeCun 1998，但我们都知道 Keller 在 Newton-Schulz 上一直做着出色的工作。</li><li><a href="https://x.com/evaninwords/status/1869767632636854570>">来自 Evan Walters (@evaninwords) 的推文</a>：他们拼错了 newton-schulz，一次也没提到 muon，还漏掉了其他论文，比如多年前促使我尝试正交化梯度的这一篇 https://arxiv.org/abs/2202.07052（他们...</li><li><a href="https://x.com/xidulu/status/1869754635453681723>).">来自 Xidulu (@xidulu) 的推文</a>：我以前不知道 MSR 的因果推理小组也在研究优化 🤣 引用 Keller Jordan (@kellerjordan0)：我想向以下新发表的论文提出 Muon 的引用请求...</li><li><a href="https://x.com/YouJiacheng/status/1869780973862408641>)">来自 YouJiacheng (@YouJiacheng) 的推文</a>：我怀疑 `GradNorm` 在“梯度白化（GradWhitening）”之后实际上是一个空操作（除了使最大奇异值 < sqrt(3) 以便 NS 收敛）。-- 平均值（顺便说一下，我认为它应该是 1_m 而不是...</li><li><a href="https://openreview.net/forum?id=0NMzBwqaAJ">并非所有 Token 都是预训练所需要的</a>：P...</li>

之前的语言模型预训练方法统一对所有训练 token 应用 next-token prediction loss。挑战这一常规，我们认为“并非语料库中的所有 token 都是...”
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1319284179208507414)** (5 条消息): 

> `Independence of Neural Network Activations, Pre-image Reconstruction Methods, Steered vs Unsteered Sparse Autoencoders, Out-of-Distribution (OOD) Evaluation` 


- **调查激活的独立性**：一位用户询问了同一层内**神经网络激活的独立性**（independence of neural network activations），表示在寻找相关分析时遇到挑战。有观点指出，*更高的模型非线性*往往会降低中间层的独立性。
- **原像重构的挑战**：该用户详细介绍了使用 MNIST 对 CNN 进行**原像重构**（pre-image reconstruction）的实验，发现对一个激活的编辑会影响其他激活。在比较两种原像方法时，*激活变化的关联性*表明激活之间存在一定程度的依赖关系。
- **关于 Sparse Autoencoder 特征的见解**：该用户对 **Sparse Autoencoder 特征**进行了类似的实验，观察到特征之间缺乏独立性。这强化了这样一个观点：神经网络中的激活可能并不像传统假设的那样独立运作。
- **衡量 SAE 重构的分布外 (OOD) 情况**：另一位用户寻求评估受控（steered）Sparse Autoencoders 中 **OOD 程度**的最佳实践。他们询问 *steered 与 unsteered 激活质心之间的余弦相似度*是否可以作为一种可行的测量策略。


  

---


### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1319035817699840000)** (254 条消息🔥🔥): 

> `Perplexity AI updates, You.com features, Gemini models, Student discounts, Referral systems` 


- **Perplexity AI 推荐系统已确认**：一位用户确认 [Perplexity 确实有一个推荐系统](https://www.perplexity.ai/settings/account)，可以让通过链接注册的用户受益。
   - 另一位用户对吸引更多人加入充满热情，表示他们的整个兄弟会都可能加入。
- **You.com 性能与模型对比**：用户对 You.com 的回答质量表示担忧，认为由于搜索界面的原因，答案可能无法达到直接使用模型的性能。
   - 用户讨论了所使用的实际模型的价值，而不仅仅是通过系统指令模拟回答。
- **学生可以使用 .edu 邮箱获得免费 Pro 权限**：有报告称学生通过 .edu 邮箱登录获得了免费的 Pro 访问权限，尽管一些用户在过程中遇到了问题。
   - 一位用户分享了[返校季促销](https://www.perplexity.ai/backtoschool)的链接，强调了潜在的福利。
- **对新超人电影的期待**：分享了新超人电影预告片的细节，引发了用户的复杂反应和兴奋。
   - 这一随机公告被描述为令人惊喜，表明用户参与度超出了 AI 讨论范围。
- **翻译游戏描述的挑战**：一位用户在让 Perplexity AI 将整个游戏描述列表翻译成法语时遇到困难，AI 在处理了几个后就停止了。
   - 用户在寻求如何管理 AI 处理大型数据集时的限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://notebooklm.google.com/)">未找到标题</a>：未找到描述</li><li><a href="https://x.com/apostraphi/status/1869612493989163410?s=46">来自 Phi Hoang (@apostraphi) 的推文</a>：我们的命运在上方</li><li><a href="https://x.com/pplxsupply/status/1869134944418890157?s=46">来自 Perplexity Supply (@PPLXsupply) 的推文</a>：设计细节</li><li><a href="https://www.youtube.com/watch?v=g_qxoznfa7E"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1319031608585617428)** (7 条消息): 

> `EU Funds Starlink Rival, Plants Cry, Law of the Few, Magic Spell Hypothesis, Tornado Alley` 


- **欧盟资助 Starlink 竞争对手**：一段 [YouTube 视频](https://www.youtube.com/embed/FBX4lu3LIEI) 讨论了 **EU** 如何资助 **Starlink** 的竞争对手，并探讨了其对全球互联网接入的潜在影响。
   - 该视频还涵盖了这一举措对卫星互联网服务中 **connectivity** 和 **competition** 的影响。
- **植物表现出哭泣行为**：关于 **plants cry**（植物哭泣）的原因话题浮出水面，讨论了关于这一迷人现象的最新发现及其对植物生物学的影响。
   - 读者参与讨论的资料指出，植物的反应可能类似于情感状态，并反映其环境压力水平。
- **理解个别人物法则 (Law of the Few)**：提到了 **Law of the Few**，正如研究人员所讨论的，这表明少数人可以影响更大的人群。
   - 相关链接说明了这一社会原则如何应用于技术和营销策略，从而增强病毒式增长的潜力。
- **魔咒假设 (Magic Spell Hypothesis) 综述**：一份文档讨论了 **Magic Spell Hypothesis**，概述了其核心论点以及与当前科学辩论的相关性。
   - [链接](https://www.perplexity.ai/page/the-magic-spell-hypothesis-n5tkbs1JR4OGww9A25c9ZA) 提供了关于其理论应用以及学术界批评的见解。
- **探索龙卷风走廊 (Tornado Alley)**：最近一项针对 **Tornado Alley** 的调查研究了定义该地区龙卷风发生情况的地理和气象数据。
   - 讨论强调了针对居住在脆弱地区人群的安全措施和准备策略，正如在[宝贵资源](https://www.perplexity.ai/search/what-is-tornado-alley-MKPYqZvsQg6x1TtVvhmARQ)中所分享的那样。



**提到的链接**：<a href="https://www.youtube.com/embed/FBX4lu3LIEI">YouTube</a>：未找到描述

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1319038525579985006)** (222 messages🔥🔥): 

> `Gemini models, Aider integration, MCP functionality, OpenAI access issues, Jira task automation` 


- **Gemini 2.0 Flash Thinking 发布**：新模型 `gemini-2.0-flash-thinking-exp-1219` 推出，展示了在推理和响应质量方面的提升潜力，特别是在 Agent 工作流中。
   - 初步测试表明，与 O1 和 DeepSeek 等现有模型相比，其性能更快且输出质量更高。
- **Aider 与 MCP 集成**：用户讨论了在 Aider 中设置 MCP 功能，并成功将其集成到创建和管理 Jira 更新等任务中。
   - 一些用户指出，虽然 Sonnet 常用，但在 MCP 设置中使用其他模型也具有潜力。
- **从 EC2 访问 OpenAI**：一位用户询问了从 EC2 服务器访问 OpenAI 服务的问题，确认运行顺畅，未报告任何问题。
   - 澄清了最初的担忧，表明这与个人设置有关，而非普遍问题。
- **任务自动化中的模型偏好**：用户确定了在工作流中处理特定任务（如 commit 信息和摘要）时对弱模型（weak model）的偏好。
   - 讨论强调了结合不同模型以在任务管理中获得最佳性能的灵活性。
- **测试其他模型**：有关于 Qwen 等各种模型的能力及其在代码任务和调试中表现的咨询。
   - 用户表示有兴趣尝试这些模型，以便更好地集成到他们的工作流自动化中。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://livebench.ai/#/">LiveBench</a>：未找到描述</li><li><a href="https://chatapi.akash.network/documentation">Akash Chat API</a>：未找到描述</li><li><a href="https://x.com/JeffDean/status/1869789813232341267">Jeff Dean (@JeffDean) 的推文</a>：介绍 Gemini 2.0 Flash Thinking，这是一个显式展示思考过程的实验性模型。基于 2.0 Flash 的速度和性能，该模型经过训练，通过思考来增强其推理能力...</li><li><a href="https://aider.chat/docs/llms/openai-compat.html">OpenAI 兼容 API</a>：aider 是你终端里的 AI 结对编程工具</li><li><a href="https://aider.chat/docs/faq.html#how-are-the-aider-wrote-xx-of-code-stats-computed">FAQ</a>：关于 aider 的常见问题。</li><li><a href="https://tenor.com/view/good-morning-gif-24191255">Good Morning GIF - Good Morning - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://openrouter.ai/docs/provider-routing',">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://mcpserver.cloud/server/server-sentry">Sentry Integration Server - MCP Server Integration | MCPHub</a>：用于错误追踪和性能监控的 Sentry.io 集成。将 Sentry Integration Server 与 Model Context Protocol 集成，以增强 AI 能力并实现无缝的模型交互。</li><li><a href="https://openrouter.ai/deepseek/deepseek-chat">DeepSeek V2.5 - API, Providers, Stats</a>：DeepSeek-V2.5 是结合了 DeepSeek-V2-Chat 和 DeepSeek-Coder-V2-Instruct 的升级版本。通过 API 运行 DeepSeek V2.5</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">高级模型设置</a>：为 LLM 配置高级设置。</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html#example-model-settings">高级模型设置</a>：为 LLM 配置高级设置。
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1319103139366043699)** (11 messages🔥): 

> `使用多个 OpenAPI 服务，Gemini Flash 2.0 问题，Architect 模式功能，模糊添加文件，项目规划模型` 


- **高效组合 OpenAPI 服务**：一位用户最初寻求关于如何使用两个不同的 OpenAPI 服务的指导，特别是 **Hugging Face** 上的 **QwQ** 和本地 **Ollama**。
   - 他们后来意识到 Hugging Face 有自己的 API，且调用方法需要在模型名称中指定。
- **Gemini Flash 2.0 修改问题**：一位成员报告了 **Gemini Flash 2.0** 持续存在的问题，指出它通常会修改错误的实例，通常是第一个。
   - 另一位成员建议使用 AI comments 功能作为权宜之计。
- **—watch-files 是否适用于 Architect 模式？**：有人询问 **—watch-files** 选项是否与 **architect** 模式兼容。
   - 回复指出，在正确使用该选项时，系统会提示进行调整。
- **聊天中的模糊文件添加**：一位用户询问了一种模糊添加文件的方法，而不需要每次都指定完整路径，并分享了一个示例输出。
   - 他们发现，必须提交文件，Aider 才能在 **/add** 命令中自动建议它们。
- **Aider 客户端推荐硬件**：有人提出了关于运行 Aider 客户端的合适硬件的问题，有报告称 LLM 已经完成，但客户端在组装响应时出现延迟。
   - 另一位成员回应称，这种延迟不应该发生，表明设置可能存在问题。


  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1319055993748983818)** (9 messages🔥): 

> `GitHub Copilot Chat, Aider Composer VSCode Extension, Diff Edits 偏好` 


- **GitHub Copilot Chat 沉浸模式发布**：GitHub 宣布了 [Copilot Chat](https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/) 的增强功能，包括沉浸式聊天体验以及针对用户需求定制的更智能、更快速的响应。
   - 现在支持与代码库的实时交互，允许立即回答编码问题并促进轻松的代码生成。
- **Aider Composer 扩展评测**：对 Aider Composer VSCode Extension 的评测强调了新的 diff 接受视图，它取代了之前的 git diff 视图，但指出它不会提交到 git，限制了撤销能力。
   - 该扩展的主要优势在于它使用已安装的 Aider 版本，增强了用户对编码过程的控制。
- **GitHub Copilot 自发布以来的改进**：成员们讨论了 GitHub Copilot 自最初免费发布以来的进展，现在提供了 Claude Sonnet 集成和多文件编辑功能。
   - 尽管如此，一位用户批评免费层级仍然提供有限的访问权限，导致对成本效益与传统 diff edits 相比的担忧。
- **对 Diff Edits 的偏好**：在比较 GitHub Copilot 的功能时，用户表现出对传统 diff edits 的偏好，认为它们更有效且经济可行。
   - 一位成员对 Copilot 的改进表示满意，但仍然主张 diff edits 在编码工作流中的持久效用。



**提及的链接**：<a href="https://github.blog/changelog/2024-12-18-announcing-github-copilot-free/">Announcing GitHub Copilot Free · GitHub Changelog</a>: Announcing GitHub Copilot Free

  

---


### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1319276345595002981)** (1 messages): 

> `Bolt Supabase 集成` 


- **Bolt 与 Supabase 集成上线**：**Bolt<>Supabase 集成**正式上线并对所有人开放，显著简化了流程。
   - *无需手动设置*：只需点击、连接即可完成，让用户更容易上手。
- **无缝连接转换**：用户现在可以通过简单的步骤连接到 **Supabase**，从而毫不费力地将其应用程序与 **Bolt** 集成。
   - 此次集成旨在简化开发人员的工作流程并消除复杂的设置过程。



**提及的链接**：<a href="https://x.com/stackblitz/status/1869715661444043245">Tweet from StackBlitz (@stackblitz)</a>: 📢 Announcing: Supabase&lt;&gt;Bolt integration!No manual setup: just click, connect, and it&#39;s done!

  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1319155617444659251)** (14 条消息🔥): 

> `Bolt 项目设置，.env 文件问题，从 Figma 直接上传，应用程序审查流程` 


- **.env 文件重置问题**：用户报告了 **.env 文件**重置的问题，导致其 Firebase 设置出现错误。一位成员指出，**锁定 .env 文件**有助于防止在会话期间发生更改，但遇到了刷新后文件被覆盖的问题。
   - *This project exceeds the total supported prompt size*（此项目超过了支持的总 Prompt 大小）被认为是用户因该问题而面临的常见故障。
- **目前尚无法从 Figma 直接上传**：有用户询问是否可以上传 Figma 文件让 Bolt 生成代码，但已确认**目前不支持直接上传**。建议的方法是使用**截图**作为变通方案。
   - 该方法已被多次请求，表明用户对改进与 Figma 等设计工具集成的需求。
- **查找 Bolt 应用程序中的冗余**：用户询问是否有办法让 Bolt **审查应用程序中的冗余**并高效清理。反馈表明，当前流程可能会消耗不必要的 Token，且未能提供有效的清理方案。
- **为项目创建 public 文件夹**：分享了关于为项目创建 **public 文件夹**并向其中添加图片以供 Bolt 使用的说明。用户对如何有效执行这些步骤以及该文件夹的位置表示困惑。
   - 对文件夹设置的澄清表明，用户仍在寻求关于项目结构的更清晰指导。


  

---


### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1319032336825712640)** (182 条消息🔥🔥): 

> `Bolt 问题与反馈，社区支持与资源，Supabase 集成，功能与 Token 使用，Bolt 用户体验` 


- **用户遇到停机和会话问题**：多位用户报告登录 Bolt 困难，以及会话超时需要刷新页面，导致聊天记录丢失。
   - 虽然团队已知晓并正在努力解决身份验证问题，但许多人对项目受到的影响和 Token 消耗表示沮丧。
- **项目社区协作**：成员们讨论了为 Bolt 创建实用指南，利用社区贡献并专注于在项目开发过程中互相支持。
   - 协作内容包括计划建立用户仪表板以上传和批准指南，这体现了社区的积极努力。
- **Supabase 集成与未来功能**：讨论强调了 Supabase 与 Bolt 的集成，并强调了其重要性，同时还提到了 Stripe 集成的未来计划和改进的 Token 管理。
   - 用户渴望了解如何在现有项目中最好地利用 Supabase 以及不同可用模式的功能。
- **产品功能与 Token 消耗反馈**：许多用户对构建应用程序时的 Token 消耗表示不满，认为冗余输出往往导致 Token 使用过度。
   - 有建议提出应改进应用程序输出的审查流程，以管理冗余并优化 Token 使用。
- **技术栈探索与开发挑战**：成员们讨论了移动应用开发的推荐技术栈，特别关注与 Supabase 的兼容性以及与 Bolt 的整体功能配合。
   - 一些用户在成功构建符合预期的项目时遇到了挑战，从而引发了关于如何有效利用 Bolt 的疑问。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://gyazo.com/2d33c95cc8f2f94179e04c14b6fdc1b2">Gyazo</a>:  </li><li><a href="https://boltwiser.levyandco.net/">Wiser - 知识共享平台</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1319355634138878023)** (1 条消息): 

> `音频概览的交互模式` 


- **音频概览交互模式现已面向所有人开放！**：团队已成功向 **100% 的用户**推出了 **Interactive Mode for Audio Overviews** 的改进。
   - 鼓励用户尝试该功能，如果之前发现其无响应，建议重新访问。
- **令人兴奋的音频功能推出**：许多 **NotebookLM 工程师**努力工作，以增强音频功能概览的**性能**。
   - 此次更新旨在为所有用户提供更流畅的体验。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1319031602226921512)** (17 条消息🔥): 

> `NotebookML 视频生成, 交互式 Podcast 功能, Podcast 编辑工作流, 将 MySQL 数据库连接到 NotebookLM, YouTube 内容创作` 


- **AI 生成的视频探索太空中的孤独感**：一位用户分享了一个 AI 生成的视频 vlog，记录了一名在太空中被隔离一年的宇航员的经历，通过[此 YouTube 链接](https://youtu.be/_ys7FchEkak?feature=shared)展示了孤独感对身心的折磨和创造力的体现。
   - 另一位用户对该视频评价道：*这是对心灵崩溃过程的扣人心弦的刻画*。
- **Podcast 交互内容未保存**：一位用户澄清说，交互式 Podcast 功能不会将交互内容保存为 Podcast 的一部分，因此有必要为外部听众分别录制这两部分。
   - 这引发了关于工作流的问题，促使一位用户寻求关于 Podcast 创建过程的进一步说明。
- **YouTube 频道展示动画视频**：另一位用户指出，某位成员的内容创作非常高产，几乎每天都会上传各种视频，包括使用 NotebookML 输出制作的动画视频，可通过[此处](https://youtu.be/ub0J93QuUH4?feature=shared)访问。
   - 观众的反馈表达了对内容的赞赏，并指出了如此频繁的上传背后的创意需求。
- **寻求连接 MySQL 到 NotebookLM 的帮助**：一位 Game master 寻求关于如何将其庞大的 MySQL 数据库连接到 NotebookLM 的建议，以便在长期运行的 RPG 会话中自动生成 NPC 反应。
   - 他们强调了自己拥有超过 10 年的游戏主持经验和庞大的玩家群体，这表明了其中涉及的复杂性。
- **旨在提高效率的 Podcast 风格 Prompt**：一位用户分享了一个旨在使 Podcast 对话更加简洁和直接的 Prompt，重点是对一个与视频加速计算相关的 QWQ 模型进行评论。
   - 提供的音频摘录旨在通过鼓励快节奏、干脆利落的对话来增强 Podcast 的表达风格。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/ub0J93QuUH4?fe"> - YouTube</a>: 未找到描述</li><li><a href="https://youtu.be/_ys7FchEkak?feature=shared"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1319033007268560906)** (144 条消息🔥🔥): 

> `Notebook LM Interactive Mode, Audio Overview 发音, 跨 Notebook 功能, 用户对新 UI 的反馈, AI 在故事创作中的实验性应用` 


- **用户反馈 Notebook LM Interactive Mode 的推出情况**：许多用户正在讨论他们对新 **Interactive Mode** 的体验，并指出虽然有些人已经获得了访问权限，但其他人仍在等待该功能的全面推出。
   - 尽管初期面临一些挑战，但用户对该模式提供的创意可能性感到兴奋。
- **Audio Overviews 中的发音问题**：一位用户报告称 Notebook LM 错误地发音了 **Shane Gostisbehere** 这个名字，反复将其与另一个名字混淆，凸显了发音方面的挑战。
   - 开发团队正在积极调查此问题，并鼓励用户提供音频样本以便更好地理解。
- **关于跨多个 Notebook 使用功能的疑问**：一位用户询问是否可以在为不同模块创建的多个 Notebook 之间共享功能和内容。
   - 已确认用户必须将所有来源上传到同一个 Notebook 中，因为目前尚不支持跨 Notebook 功能。
- **对新用户界面的正面反馈**：多位用户对最近更新的 **Notebook LM UI** 表示赞赏，认为其非常实用且用户友好。
   - 团队收到了积极的肯定，用户们渴望探索新的功能和特性。
- **AI 在故事创作中的创意应用**：一位用户分享了他们使用 AI 进行故事创作的兴奋之情，详细介绍了一个为设定在 Cyberpunk 未来的 TTRPG 生成角色的实验。
   - 他们强调了 Notebook LM 如何在保持忠于故事素材的同时，成功适应各种叙事挑战。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?">Upgrading to NotebookLM Plus - NotebookLM Help</a>：未找到描述</li><li><a href="https://support.google.com/notebooklm/answer/15678219?hl=en#:~:text=NotebookLM%20vs%20NotebookLM%20Plus%20User%20Limits">Upgrading to NotebookLM Plus - NotebookLM Help</a>：未找到描述</li><li><a href="https://youtu.be/tVURtFDvyFc?si=8PTHE9BdAKrJ2f0N"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=YS4rdvcfqEU"> - YouTube</a>：未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1hhyv8r/notebook_lm_hosts_have_full_on_sex_warning/">Reddit - Dive into anything</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1319032634587742220)** (102 messages🔥🔥): 

> `在 Ubuntu 上运行 SDXL，ComfyUI 问题，AI 图像和视频质量，量子计算对话，Civitai 网站问题` 


- **在 Ubuntu 上运行 SDXL 的建议**：几位成员讨论了在 **Ubuntu** 上运行 **SDXL** 的技巧，建议包括使用 **Forge UI** 以及利用 shell 启动文件来简化设置。
   - *Nuuideas* 指出，对系统缺乏了解可能会阻碍 **ComfyUI** 的性能。
- **ComfyUI 的持续性问题**：尽管尝试了故障排除，仍有投诉称 **ComfyUI** 存在恼人的错误，且在使用某些采样方法时会产生“烧焦”的图像。
   - *Nuuideas* 建议使用 **Euler** 采样并保持最佳的去噪（denoising）设置以获得更好的效果。
- **AI 图像和视频的预期与现实**：讨论了 **AI 生成的图像**和**视频**是否已达到完美，*earnstar* 断言由于诸多挑战，即使到 2030 年它们也不会完美。
   - *Eyaura* 表示反对，声称 AI 技术的快速进步可能会更早带来改进。
- **关于量子计算未来的辩论**：对话围绕**量子计算**展开，特别是关于证明 **P=NP** 等问题的含义，*Nuuideas* 对量子算法的实用性表示担忧。
   - *Earnstar* 强调了从量子态中提取有用结果的挑战。
- **Civitai 网站功能**：*Wallykz* 报告了访问 **civitai.com** 的问题，其他成员确认了宕机情况，并指出该网站经常离线。
   - *Crystalwizard* 提到该网站经常出现服务器问题，导致无法访问。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.tomshardware.com/news/how-to-manage-virtual-memory-pagefile-windows-10,36929.html">How To Manage Virtual Memory (Pagefile) In Windows 10</a>：按照这些简单的步骤在 Windows 10 中手动管理虚拟内存（页面文件）大小。</li><li><a href="https://tenor.com/view/hello-well-hello-home-alone-christmas-marvin-gif-15846293">Hello Well Hello GIF - Hello Well Hello Home Alone - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=AbB33AxrcZo"> - YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=NB9K4CoYSIM&ab_channel=AIRevolution"> - YouTube</a>：未找到描述</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/blob/main/webui-user.sh">stable-diffusion-webui-forge/webui-user.sh at main · lllyasviel/stable-diffusion-webui-forge</a>：通过在 GitHub 上创建账户为 lllyasviel/stable-diffusion-webui-forge 的开发做出贡献。</li><li><a href="https://youtu.be/S9L2WGf1KrM"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1319052977977298964)** (58 messages🔥🔥): 

> `电感啸叫，GPU 性能与选择，瓶颈辩论，VRChat VRAM 需求，下一代 GPU 定价` 


- **电感啸叫再次袭来**：一位用户对退货的 RX 6750XT 发出的**荒谬电感啸叫**表示担忧，导致了糟糕的体验。
   - 另一位成员幽默地建议，电感啸叫声可能大到可以“播放音乐”。
- **决定 GPU 选择**：讨论围绕选择预算友好的 GPU 展开，建议中提到 **7900 XTX** 是相对于 **NVIDIA** 显卡的一个不错选择。
   - 共识倾向于等待下一代 GPU，因为预期价格会很高。
- **瓶颈论引发辩论**：一位用户辩称**瓶颈（bottlenecking）**并不存在，而其他人则指出较弱的 CPU 会延迟向 GPU 交付帧。
   - 辩论突显了关于 CPU 性能对整体 **FPS** 影响的不同观点。
- **VRChat 对显存的渴求**：提到了 VRChat 的 **VRAM 需求**，暗示它可能会迅速消耗可用内存，导致性能问题。
   - 用户注意到，由于这些需求，许多玩家选择了 **4090**。
- **对下一代 GPU 的担忧**：有人担心未来的 **RTX 50** 系列价格可能比现有产品**高得离谱**。
   - 尽管存在担忧， AMD 承诺以更低成本提供具有竞争力的性能，这带来了一定程度的审慎乐观。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1319122047825350748)** (4 messages): 

> `tl.dot 输入形状要求，AMD GPU 性能对比 PyTorch，Nvidia Hopper warp-specialization 移除，Triton 性能优化` 


- **tl.dot 的输入形状需要 >= 16**：**tl.dot** 的输入形状应满足 **M >= 16, N >= 16, 且 K >= 16**，这主要是由于 Tensor Core 的硬件要求。
   - 有用户询问，当 M、N 或 K 小于 **16** 时，**tl.dot** 是否可以默认使用 *CUDA cores* 进行计算。
- **寻找更快的 AMD GPU 核函数**：一位用户询问是否有人发现过在 **RX 7900** 等 **AMD GPU** 上运行速度比 **PyTorch/rocBLAS** 更快的核函数。
   - 另一位用户指出，到目前为止，**Triton** 的性能尚未超越其 **BLAS** 实现，特别是在 **Navi31** 架构上。
- **Nvidia Hopper 的 warp-specialization 特性被移除**：一位用户发现 **Nvidia Hopper** 的 **warp-specialization** 特性在 **Triton 3.x** 中已被移除。
   - 他们询问在此更改后，使用 **Triton** 获得更好性能的可能技术。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1319297531951185930)** (5 messages): 

> `cudaMemcpy 性能，CUTLASS tma_store_wait 函数行为，TMA 操作文档` 


- **探索 cudaMemcpy 的更快速替代方案**：一位成员询问是否有比使用 **cudaMemcpy** 更快的方法来将小尺寸数据（如 12 字节）复制到设备内存，据报道 **cudaMemcpy** 大约需要 **1-2us**。
   - 这引发了关于 CUDA 编程中内存传输潜在优化方案的讨论。
- **tma_store_wait 可能会自动完成**：一位成员观察到，在 CUTLASS 中使用 **tma_store_wait** 执行 TMA-store 操作后，可能不需要手动等待，因为它似乎会自动完成。
   - 这表明其行为类似于 **expect_tx**，引发了关于其处理操作效率的讨论。
- **需要文档确认**：针对 TMA 操作的讨论，一位成员请求提供文档，以澄清该功能是否如最初预期的那样受支持。
   - 该请求强调了准确且易于获取的文档对于开发实践的重要性。


  

---


### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/)** (1 messages): 

0x000ff4: 有人在为 Keras/PyTorch 做贡献吗？
  

---

### **GPU MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1319327988533166122)** (21 messages🔥): 

> `Genesis AI, Sim2Real Technology, CARLA Simulator Update, Synthetic Data Generation for Autonomous Driving, Dexterous Task Applications` 


- **Genesis AI 引起关注**：社区对 [Genesis](https://genesis-embodied-ai.github.io/) 及其潜在应用表现出极大的兴趣，特别是强调了令人印象深刻的水滴演示。
   - 一位成员评论道，*'超级酷的东西，'* 展示了 AI 新工具的吸引力。
- **探索 Sim2Real 概念**：讨论转向了 [Sim2Real](https://www.example.com)，重点关注其将技能从模拟环境转移到现实世界应用的能力，并强调了烹饪和组装等任务。
   - 一位用户询问，*'我想知道它在灵巧任务（dexterous tasks）上的表现如何，'* 表明了对其具体功能的兴趣。
- **CARLA Simulator 迎来重大升级**：团队庆祝了 **CARLA 0.10.0 版本**的发布，该版本通过迁移到 **Unreal Engine 5.5** 增强了视觉保真度，并引入了 [Lumen](https://dev.epicgames.com/documentation/en-us/unreal-engine/lumen-technical-details-in-unreal-engine) 和 [Nanite](https://dev.epicgames.com/documentation/en-us/unreal-engine/nanite-virtualized-geometry-in-unreal-engine) 等高级功能。
   - 此次更新包括*升级的环境和资产*，展示了渲染技术的进步。
- **Synthetic Data 生成讨论浮现**：关于 **Waymo** 数据处理方式的推测，成员指出 *Waymo 可能在真实驾驶数据之外也生成 Synthetic Data*。
   - 分享了相关文章的链接，包括关于在自动驾驶中嵌入 Synthetic Data 的[这项研究](https://waymo.com/research/embedding-synthetic-off-policy-experience-for-autonomous-driving-via-zero/)。
- **自动驾驶中 Synthetic Data 的未来**：成员们详细阐述了未来的进步可能会使大部分模拟数据变为 Synthetic Data，并可能集成像 **Genesis** 这样的工具。
   - 对话以对这类框架的扩展性好奇结束，并指出了它们在生成精确车辆动力学方面的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://carla.org/2024/12/19/release-0.10.0/">CARLA 0.10.0 发布，支持 Unreal Engine 5.5！</a>：迁移至 Unreal Engine 5.5，全新的资产，升级的 Town 10，重新建模的车辆，露天矿地图</li><li><a href="https://arxiv.org/html/2406.09386v1">SimGen: Simulator-conditioned Driving Scene Generation</a>：未找到描述
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/1319261477802213438)** (1 messages): 

> `Image Analysis, User Concerns` 


- **围绕图像分析展开讨论**：一位成员引用了频道中的一张图片，引发了关于其内容和相关性的讨论。
   - 虽然没有引用关于图片的具体细节，但互动表明它引起了小组的注意。
- **对图像的幽默互动**：该成员的回复包含了一句轻松的评论，表示对分享的图片内容感到有趣。
   - 这种幽默暗示了聊天中活跃的气氛，有助于提升讨论的整体愉悦度。


  

---

### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1319053856751095948)** (1 条消息): 

> `MatX hiring, LLM accelerator ASIC development, Low level compute kernel author roles, ML performance engineer roles, In-person work culture` 


- **加入 MatX，共同构建 LLM 加速器 ASIC**：MatX 正在招聘包括 **底层计算 Kernel 作者**、**编译器**和 **ML 性能工程师**在内的职位。感兴趣的候选人可以在其 [职位列表](https://grnh.se/2b337cb08us) 中找到更多信息。
   - 他们重视**效率**和**高质量解决方案**，欢迎从应届毕业生到资深专业人士的各类申请者。
- **MatX 鼓励创新式问题解决**：团队强调需要*考虑新方法*，经常为了更适合其场景的更好替代方案而放弃传统方法。
   - 他们优先考虑基于深度理解做出重大决策，这表明透彻的推理往往比广泛的测试更重要。
- **强调高信任的团队环境**：MatX 倡导一种植根于在其高信任团队中邀请并包容多元观点的文化。
   - 支持性的团队合作至关重要，因为他们相信这对于共同应对复杂挑战至关重要。



**提到的链接**：<a href="https://grnh.se/2b337cb08us">MatX</a>：&lt;header&gt;&lt;h2&gt;MatX: 为 LLM 提供更快的芯片&lt;/h2&gt;&lt;/header&gt;&lt;div id=&quot;maincontent&quot;&gt;&lt;h3&gt;加入我们！&lt;/h3&gt;&lt;ul&gt;&lt;li&gt;无论我们是在工作...

  

---


### **GPU MODE ▷ #[sparsity-pruning](https://discord.com/channels/1189498204333543425/1247663759434977453/1319229746688819251)** (1 条消息): 

> `Sparsity Design, Sparsifier Functionality, Sparsify Kernel Optimization, Demo for Sparsify Usage` 


- **理解 Sparsifier 的作用**：有人提出了关于 **Sparsifier** 功能的疑问，特别是它是否负责确定 **sparsity pattern**（稀疏模式）。
   - 得到的澄清是 **Sparsifier** 确实决定了模式，但其输出如何通过 **sparsify_** 与 Kernel 优化过程交互受到了质疑。
- **Sparsifier 与 Sparsify 之间的交互**：用户询问 **sparsify_** 函数在运行期间是否消耗 **Sparsifier** 的输出。
   - 理解这种交互对于优化设计中的稀疏性至关重要，并寻求进一步的指导。
- **请求 Sparsify 使用演示**：有人请求提供关于 **sparsify_** 使用的演示，强调了对实际示例的需求。
   - 该演示将提供关于如何在实际场景中有效实现 **sparsity design** 的见解。


  

---


### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1319282562580877314)** (1 条消息): 

> `alma Python Package, Model Benchmarking, PyTorch Conversion Options` 


- **开源 alma：基准测试利器**：两位开发者刚刚开源了他们的项目 **alma**，这是一个 Python 包，旨在通过单个函数调用对超过 **40 种 PyTorch 转换选项**的速度进行基准测试。
   - 它具有*优雅的错误处理*和*隔离进程*等特性，以实现更安全的测试，旨在简化 CI 集成。
- **alma 的未来集成计划**：未来的开发目标是增加更多转换选项，并与 *JAX*、*llama.cpp* 和 *VLLM* 集成，以增强通用性。
   - 创建者邀请用户通过 GitHub 分享想法，强调通过社区参与来扩展功能。
- **真实世界性能示例**：示例输出显示了令人印象深刻的结果，**EAGER** 模式在 CUDA 设备上实现了 **282395.70 samples/second** 的吞吐量。
   - 在 *EXPORT+EAGER* 模式下，性能略有提升，达到 **305974.83 samples/second**，展示了该包的效率。



**提到的链接**：<a href="https://github.com/saifhaq/alma">GitHub - saifhaq/alma</a>：通过创建一个账户来为 saifhaq/alma 的开发做出贡献。

  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/)** (1 条消息): 

kimishpatel: 这正是我来这里的目的 🙂
  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1319135025979199549)** (5 messages): 

> `Cost of GPUs on Vast AI, Generative Flow Networks, ARC Prize Daily Puzzle, Training Smaller Models, Synthesizing Riddles` 


- **Vast AI 提供廉价 GPU 选项**：一位成员指出 [Vast AI](https://vast.ai) 上的 GPU 非常实惠，特别提到 **3070** 是个人使用的性价比之选。
   - 另一位成员分享了他们的经验，表示之前只检查过 **Lambda** 和 **Runpod** 的 GPU 选项。
- **探索用于数据集生成的 Generative Flow Networks**：讨论围绕 **Generative Flow Networks** 展开，认为它是**合成数据集生成**的一种有前景的方法，特别是在 oracle rewards 成本高昂的场景下。
   - 一位成员分享了关于该主题的[论文](https://arxiv.org/pdf/2106.04399)，强调了减少 **(problem, reward) pairs** 标注的潜力。
- **解决 ARC Prize 每日谜题**：一位成员庆祝成功解决了 ARC Prize 每日谜题，强调了 **12pm UTC** 的每日挑战需要对输入进行排序。
   - 他们对 autoregressive models 表示怀疑，指出其在设计上除非采用某些先验推理，否则在排序方面存在局限性。
- **使用较小模型进行训练以提高效率**：一位成员提到了在 **24G** 显存上可训练的小型模型进行迭代的实用性，认为这比大型模型更具效率优势。
   - 这与之前关于探索低成本 GPU 选项以获得最佳训练结果的讨论相呼应。
- **谜题合成中的挑战**：一位成员反思了寻找生成谜题的正确表示形式的困难，强调了对 **input boards** 和变换（transformations）的需求。
   - 他们强调了确保所有相关的变换参数都能从提供的示例中推导出来的重要性。



**提到的链接**：<a href="https://arcprize.org/play">ARC Prize - Play the Game</a>：对人类简单，对 AI 很难。尝试一下 ARC-AGI。

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1319033351595757648)** (87 messages🔥🔥): 

> `AI Agentic Systems, Gemini 2.0 Flash Thinking, Databricks Funding, ModernBERT Release, Alec Radford Departure from OpenAI` 


- **AI Agentic Systems 正在兴起**：Anthropic 分享了关于 Agentic Systems 成功实现的见解，并强调使用简单、可组合的模式进行构建，暗示 2025 年将是该领域的关键一年。
   - Anthropic 的博客文章强调了 AI 中 Agent 和 workflows 的最佳实践以及不断演变的定义。
- **Gemini 2.0 Flash Thinking 占据主导地位**：Gemini 2.0 Flash Thinking 的推出展示了其推理能力，在各个类别中均名列前茅，并在多项任务中超越了其前代产品。
   - 根据报告，该模型显式地展示了其思考过程，更有效地提高了推理性能。
- **Databricks 获得巨额融资**：Databricks 宣布了由 Thrive Capital 领投的 J 轮融资，筹集了 100 亿美元，估值达到 620 亿美元，并预计收入运行率（revenue run rate）将突破 30 亿美元。
   - 这标志着公司的重大势头，反映了主要由 AI 需求驱动的 60% 同比增长。
- **ModernBERT 的发布引发关注**：ModernBERT 的推出在 AI 社区引起了极大关注，它比旧模型有所改进，具有更长的 context 和增强的性能。
   - 关于其特性和潜在应用的讨论突显了对其集成到现有 workflows 中的期待。
- **Alec Radford 离开 OpenAI**：OpenAI GPT 开发的关键人物 Alec Radford 即将离职去从事独立研究，这引发了关于该组织未来的疑问。
   - 这一人事变动引发了在行业近期其他变化的背景下对 OpenAI 发展方向的猜测。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/vikhyatk/status/1869605301596631191">来自 vik (@vikhyatk) 的推文</a>：每个人都在发布关于这个的 🤯 推文，但我敢说没人真正试过。因为我试了，它显示 `genesis` 模块上不存在 `generate` 方法。引用 Allen T. (@Mr_AllenT) 这是最...</li><li><a href="https://x.com/justinlin610/status/1869793885540757715?s=46">来自 Junyang Lin (@JustinLin610) 的推文</a>：非常抱歉让大家久等了。今晚不会有任何进展。我们仍需为这次发布完善一些细节。很快就会回来。</li><li><a href="https://apply.ai.engineer">AI Engineer Summit</a>：年度最高质量的 AI 技术盛会。面向 AI Engineers 和领导者，2025 年 2 月 20 日至 21 日。</li><li><a href="https://x.com/presidentlin/status/1869745206842794047?s=46">来自 Lincoln 🇿🇦 (@Presidentlin) 的推文</a>：@Alibaba_Qwen 他们做了一个推理 VL 模型？</li><li><a href="https://genesis-embodied-ai.github.io/">Genesis</a>：未找到描述</li><li><a href="https://www.anthropic.com/research/building-effective-agents">构建高效 agents</a>：一篇为开发者提供构建高效 AI agents 的建议和工作流的文章。</li><li><a href="https://x.com/elevenlabsio/status/1869462840941461941">来自 ElevenLabs (@elevenlabsio) 的推文</a>：认识一下 Flash。我们最新的模型，生成语音仅需 75ms + 应用和网络延迟。你从未体验过如此快速且类人的 TTS。</li><li><a href="https://x.com/lmarena_ai/status/1869793847548817563?s=46">来自 lmarena.ai (原 lmsys.org) (@lmarena_ai) 的推文</a>：来自 Chatbot Arena 的重磅消息⚡🤔 @GoogleDeepMind 的 Gemini-2.0-Flash-Thinking 在所有类别中首次亮相即登顶第一！相比 Gemini-2.0-Flash 的飞跃：- 总榜：#3 → #1 - 总榜（风格控制）：#4 → #...</li><li><a href="https://x.com/steph_palazzolo/status/1869848094009110826">来自 Stephanie Palazzolo (@steph_palazzolo) 的推文</a>：来自 @erinkwoo 的重大 OpenAI 人事消息：OpenAI 原始 GPT 论文的第一作者 Alec Radford 将离职去从事独立研究。https://www.theinformation.com/briefings/senior-op...</li><li><a href="https://x.com/swyx/status/1869825047051022464">来自 swyx (@swyx) 的推文</a>：这就是幕后与像 @benghamine 这样的 AGIs 合作的样子。想知道 @recraftai、@GeminiApp 或 @xai 何时能匹配这种工作流。引用 jason liu (@jxnlco) 噢，天哪，这太重磅了...</li><li><a href="https://x.com/_sholtodouglas/status/1869798291535446383">来自 Sholto Douglas (@_sholtodouglas) 的推文</a>：我非常喜欢这个问题中的思考，这是一个跳出框框思考的绝佳例子。随着模型变得越来越强，认真对待它们将继续是理解当前一代...的正确方式。</li><li><a href="https://www.luzia.com/en">Luzia：点击即达的智能助手</a>：轻松且免费地获取 AI 的力量。Luzia (我是 Luzia) 帮助你在日常生活中处理成千上万的任务，无论是在工作、学校、社交时刻还是追求你的爱好...</li><li><a href="https://www.databricks.com/company/newsroom/press-releases/databricks-raising-10b-series-j-investment-62b-valuation">Databricks 正在以 620 亿美元估值筹集 100 亿美元的 J 轮投资</a>：融资由新投资者 Thrive Capital 领投。公司预计年化营收将突破 30 亿美元，并在第四季度实现正向自由现金流。</li><li><a href="https://x.com/noamshazeer/status/1869789881637200228?s=46">来自 Noam Shazeer (@NoamShazeer) 的推文</a>：我们一直在 *思考* 如何提高模型的推理能力和可解释性。介绍 Gemini 2.0 Flash Thinking，这是一个经过训练可以“大声思考”的实验性模型，从而带来了更强的推理表现...</li><li><a href="https://huggingface.co/blog/modernbert">终于有了 BERT 的替代品：介绍 ModernBERT</a>：未找到描述</li><li><a href="https://x.com/odysseyml/status/1869417873938219360?s=46">来自 Odyssey (@odysseyml) 的推文</a>：今天是 @odysseyml 的大日子。我们正在分享 Explorer，我们的第一个生成式世界模型。我们认为世界模型是 AI 的下一个前沿，能够实现美妙的新事物。为了帮助塑造这一领域，Ed ...</li><li><a href="https://x.com/ehuanglu/status/1869549996045160558?s=46">来自 el.cine (@EHuanglu) 的推文</a>：AI 语音正在接管一切！ElevenLabs 刚刚发布了 Flash 2.5，能够几乎瞬间从文本生成逼真的电影对话。从现在起，不要相信你听到的一切！</li><li><a href="https://x.com/altryne/status/1869835859727393234?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：我在来自 @AIExplainedYT simple bench 的 10 个挑战性问题上评估了 o1-2024-12-17（包含所有 3 种推理强度）和 gemini-2.0-flash-thinking-exp-1219，并得到了一些令人惊讶的结果！Flash thin...</li><li><a href="https://x.com/alexalbert__/status/1869812081597526079?s=46">来自 Alex Albert (@alexalbert__) 的推文</a>：2025 年将是 agentic systems 之年。拼图正在逐渐完整：computer use、MCP、改进的工具</li>

使用。是时候开始考虑构建这些系统了。在 Anthropic，我们正...</li><li><a href="https://x.com/officiallogank/status/1869789820308074837?s=46">来自 Logan Kilpatrick (@OfficialLoganK) 的推文</a>：就在你以为一切都结束时... 我们推出了 Gemini 2.0 Flash Thinking，这是一个全新的实验性模型，它解锁了更强的推理能力并展示其思考过程。该模型会进行规划（通过...</li><li><a href="https://x.com/zhou_xian_/status/1869511650782658846?s=46">来自 Zhou Xian (@zhou_xian_) 的推文</a>：你所喜爱的生成式模型的一切——现在由真实的物理驱动！发布 Genesis 项目——经过 20 多个研究实验室参与的为期 24 个月的大规模研究合作——一个生成式...</li><li><a href="https://x.com/hamelhusain/status/1869808528258679057?s=46">来自 Hamel Husain (@HamelHusain) 的推文</a>：对于那些好奇 ModernBert 如何融入 RAG 的人，一个很好的起点是 @bclavie 的入门指南“超越 RAG 的基础”。他谈到了各种类型的 encoder，何时使用它们，以及不同的 t...</li><li><a href="https://x.com/_sholtodouglas/status/1869796444502462527?s=46">来自 Sholto Douglas (@_sholtodouglas) 的推文</a>：感受一下我们最近在思考的东西 :) 试一试吧！它还有些原始，我们预计它会有一些不完善之处——但它代表了在 test time compute 上的惊人算法进展。一个...</li><li><a href="https://x.com/jeremyphoward/status/1869786023963832509?s=46">来自 Jeremy Howard (@jeremyphoward) 的推文</a>：我直入主题。我们训练了 2 个新模型。像 BERT，但是更现代。ModernBERT。不是什么炒作的 GenAI 玩意儿，而是一个真正的“劳模”模型，用于检索、分类等。真正的实用...</li><li><a href="https://x.com/daytonaio/status/1869727933046112578">来自 Daytona.io (@daytonaio) 的推文</a>：🚀 绘制 AI 开发的未来！这是你的 AI Enablement Stack 全面指南——从基础设施到自主 Agent——一个社区驱动的开源努力。深入探讨...</li><li><a href="https://youtu.be/a0bEU83P8g8?si=9V0yJeqtWnhVicKI"> - YouTube</a>：未找到描述
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1319073093133537395)** (67 条消息🔥🔥): 

> `OpenInterpreter 1.0 更新, 在 server 模式下运行命令, Google Gemini 2.0 多模态, 本地 vs 服务器命令执行, OS 模式功能` 


- **OpenInterpreter 1.0 支持视觉模型**：1.0 分支支持具有视觉功能的模型，允许用户通过 [GitHub](https://github.com/OpenInterpreter/open-interpreter.git) 使用 `pip install git+https://github.com/OpenInterpreter/open-interpreter.git@development` 进行安装。
   - 实验表明 `--tools gui` 命令是可用的，可以根据需要连接到不同的模型或 API。
- **Server 模式操作问题**：一位用户询问当 OI 作为服务器运行时命令是如何执行的，想知道它们是在本地运行还是在服务器上运行。
   - 有人指出，一些用户在常规模式下运行它并使用 SSH 进行访问，但他们正在考虑集成前端以提高效率。
- **询问 Google Gemini 2.0 的能力**：一位用户对 Google Gemini 2.0 多模态能力的表现表示好奇，特别是在 OS 模式下。
   - 大家对新模型与现有系统的对比很感兴趣，特别是关于其命令执行功能。
- **使用 OI 控制本地机器**：关于 OpenInterpreter 控制本地系统能力的讨论揭示了鼠标和代码执行功能方面的局限性。
   - 用户报告称，要使预期的 OS 模式功能完全运作仍存在问题。
- **清理 OpenInterpreter 安装**：由于在使用 OpenInterpreter 时遇到的问题，特别是进行了多次配置后，用户提出了需要进行干净安装的担忧。
   - 用户讨论了删除某些 flag 并调整命令以解决错误和安装过程中的不确定性。


  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1319336699368706160)** (1 条消息): 

> `O1 频道探索, 理解文档` 


- **寻求关于 O1 功能的澄清**：一位成员在探索 **O1** 频道后，表示需要一个更简单的解释来了解其工作原理。
   - 他们承认阅读了文档，但仍然觉得自己像个 *noob*（菜鸟），并感谢任何提供的帮助。
- **文档帮助不够**：同一位成员指出，尽管他们努力阅读了文档，但文档并未提供必要的清晰度。
   - 他们正在寻找直接的指导，以便快速上手 **O1**。


  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1319032281507041310)** (62 条消息🔥🔥): 

> `LM Studio 模型加载问题、LM Studio 移动端访问、GPU 驱动问题、LM Studio 图像输入模型、AMD 驱动已知问题` 


- **LM Studio 模型加载错误**：一位用户报告在尝试加载模型时遇到错误提示 ```Safetensors header is unexpectedly large: bytes=2199142139136```，这表明可能存在兼容性问题或文件损坏。
   - 另一位用户确认 Llama 3.3 的 MLX 版本也出现了该提示，导致他们不得不重新下载模型以期望解决问题。
- **从移动设备连接到 LM Studio**：成员们讨论了通过移动端使用 LM Studio 的方法，一位用户分享了一个名为 **3Sparks Chat** 的 iOS 应用，它可以连接到 PC 或 Mac 上的 LM Studio 服务器。
   - 然而，由于目前还没有可用的移动端解决方案，对 Android 版本的需求落空了。
- **最新 AMD 驱动程序的问题**：用户详细说明了 **AMD 24.12.1 驱动程序** 的问题，据报道该驱动在 LM Studio 加载模型时会导致系统卡顿，这表明它与 llama.cpp rocm 库存在更广泛的冲突。
   - 建议包括降级到之前的驱动版本，以缓解部分用户遇到的性能问题。
- **LM Studio 的图像输入模型**：一位用户询问了适用于 LM Studio 的图像输入模型（特别是针对 PC 用户），并获得了关于 **mlx-community/Llama-3.2-11B-Vision-Instruct-4bit** 模型的信息，但该模型面临若干加载问题。
   - 讨论中涉及了模型兼容性，并对其他格式不支持 Windows 运行时（runtime）表示了担忧。
- **通用硬件配置讨论**：几位用户交流了他们的硬件规格细节，特别是在使用 LM Studio 时 **7900XTX** GPU 的兼容性，以及配置差异如何影响性能。
   - 一位用户指出，尽管配置相似，但体验却有所不同，这表明硬件性能或驱动交互可能存在差异。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://apps.apple.com/us/app/3sparks-chat/id6736871168">‎3sparks Chat</a>：无论您是使用 LM Studio 还是 Ollama 在 Mac、PC 或 Linux 上本地运行 LLM，还是访问 OpenAI API 的强大功能，3Sparks Chat 都是您的首选移动客户端。随时随地与您的 LLM 聊天...</li><li><a href="https://huggingface.co/mlx-community/Llama-3.2-11B-Vision-Instruct-4bit">mlx-community/Llama-3.2-11B-Vision-Instruct-4bit · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1319398696961577052)** (3 条消息): 

> `Silicon 芯片性能、基准测试对比` 


- **高端 Silicon 芯片速度质疑**：一位成员询问 **prompt processing**（提示词处理）在 **高端 Silicon 芯片**（max, pro, ultra）上是否因为更强的 **memory bandwidth**（内存带宽）而更快。
   - 另一位成员指出，这些芯片的速度不如 **30/4090** 型号。
- **获取 Llama.cpp 基准测试**：一位成员分享说 **llama.cpp** 在其 GitHub 讨论页面上维护了每个模型的基准测试。
   - 详情可以在这个 [GitHub discussion](https://github.com/ggerganov/llama.cpp/discussions/4167) 中找到。


  

---

### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1319338173586997330)** (1 messages): 

> `Price reductions, Market competition` 


- **Gryphe 降价 7%**：[gryphe/mythomax-l2-13b](https://openrouter.ai/gryphe/mythomax-l2-13b) 的价格今早下降了 **7%**，延续了市场降价的趋势。
   - *这是 AI 模型竞争格局中持续价格战的一部分*。
- **Qwen 降价 7.7%**：随着价格战升温，[qwen/qwq-32b-preview](https://openrouter.ai/qwen/qwq-32b-preview) 出现了另一次 **7.7% 的大幅降价**。
   - *这些调整反映了领先 AI 供应商之间的激烈竞争*。
- **Mistral-Nemo 降价 12.5%**：[mistralai/mistral-nemo](https://openrouter.ai/mistralai/mistral-nemo) 降价了 **12.5%**，表明了其积极的定价策略。
   - 这反映了**日益加剧的市场动态**，各公司都在争夺客户关注度和市场份额。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/gryphe/mythomax-l2-13b>)">MythoMax 13B - API, Providers, Stats</a>: Llama 2 13B 性能最高且最受欢迎的微调版本之一，具有丰富的描述和角色扮演能力。#merge。通过 API 运行 MythoMax 13B</li><li><a href="https://openrouter.ai/qwen/qwq-32b-preview>)">QwQ 32B Preview - API, Providers, Stats</a>: QwQ-32B-Preview 是由 Qwen 团队开发的专注于 AI 推理能力的实验性研究模型。作为预览版，它展示了极具前景的分析能力，同时也存在一些...</li><li><a href="https://openrouter.ai/mistralai/mistral-nemo>)">Mistral Nemo - API, Providers, Stats</a>: 由 Mistral 与 NVIDIA 合作构建的 12B 参数模型，具有 128k token 上下文长度。该模型是多语言的，支持英语、法语、德语、西班牙语、意大利语、葡萄牙语、中文...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1319300285755293787)** (1 messages): 

> `AI Ecosystem Maps, Crowdsourced AI Enablement Stack` 


- **对众包 AI Enablement Stack 的需求**：许多 VC 机构都发布了他们的 AI 生态系统图谱，但市场需要一个真正**众包**且**开源的 AI enablement stack**。
   - 该倡议旨在让开发者了解该使用哪些工具，确保他们不会在项目中浪费时间。更多详情请见 [GitHub](https://github.com/daytonaio/ai-enablement-stack)。
- **征求关于 AI Enablement 逻辑的反馈**：目前正在公开征集关于此 AI enablement 方法的**逻辑和结构**的贡献与反馈。
   - 目标是为开发者创建一个最新的资源，鼓励社区投入与协作。



**提到的链接**: <a href="https://github.com/daytonaio/ai-enablement-stack">GitHub - daytonaio/ai-enablement-stack: A Community-Driven Mapping of AI Development Tools</a>: 社区驱动的 AI 开发工具图谱 - daytonaio/ai-enablement-stack

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1319034462800973918)** (62 条消息🔥🔥): 

> `DeepSeek 模型, OpenRouter 问题, 模型与 API 讨论, 数据管理, 用户体验反馈` 


- **DeepSeek 模型学习探索**：用户正在尝试使用 **DeepSeek-v2** 和 **DeepSeek V2.5** 进行编程辅助，强调了输入整个 GitHub 仓库以更好地理解复杂项目的好处。
   - 一位用户提到 **DeepSeek** 如何帮助进行代码优化和注释，而另一位用户则警告不要将其用于高级代码创建。
- **OpenRouter 用户支持挑战**：多位用户报告了 **OpenRouter** 的问题，包括意外的账户问题以及支持团队针对余额丢失给出的回复不明确。
   - 用户沮丧情绪显而易见，其中一人寻求关于余额消失的解释，强调了改进支持团队沟通的必要性。
- **API 与模型能力讨论**：出现了关于 **o1 reasoning_effort** 参数可访问性的提问，表明用户对模型能力和接口的关注。
   - 用户还讨论了不同模型的效用以及敏感任务（特别是涉及医疗数据）中隐私设置的重要性。
- **OpenRouter 功能用户体验**：参与者分享了他们对界面及其在各种用途下的适用性的看法，并提出了一些改进用户导航的建议。
   - 讨论涉及界面标签、清晰度以及在 AI 应用中需要更精简的用户体验。
- **社区互动与幽默**：成员们参与了轻松的闲聊，讨论了关于用户简介和在线人设趣味性的笑话。
   - 社区氛围显得十分友好，用户在进行关于平台的严肃咨询之余，也参与了有趣的评论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/terms#_4_-payment">OpenRouter</a>: LLM 的统一接口。为您的提示词寻找最佳模型和价格。</li><li><a href="https://openrouter.ai/docs/oauth">OAuth PKCE | OpenRouter</a>: 通过 OAuth 进行安全用户认证。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[beta-feedback](https://discord.com/channels/1091220969173028894/1277894087755829278/1319147059042979862)** (1 条消息): 

> `编程化功能请求, Provider API 集成` 


- **编程化功能实现请求**：一位成员表示有兴趣看到特定功能的**编程化版本**，并强调了在请求中传递 **provider API key** 的能力。
   - *“我很想看到这个功能的编程化版本”* 突显了对增强 API 集成功能的渴望。
- **对 API Key 功能的兴趣**：该成员再次强调了在请求中隐式传递 **provider API key** 的需求，以简化访问并提升用户体验。
   - 这表明开发者对满足灵活性和效率需求的 API 功能有着更广泛的兴趣。


  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1319039562802266133)** (47 messages🔥): 

> `GitHub Copilot 免费层级, Granite 3.1-8B-Instruct 模型, 用于本地 LLM 的 LM Studio, Model Context Protocol 测试, Gemini Flash Thinking 实验版` 


- **GitHub Copilot 现已向所有人免费开放**：宣布推出全新的 [GitHub Copilot 免费层级](https://x.com/code/status/1869449373995708703)，即刻可用，无需试用或订阅。
   - 用户无需提供信用卡即可享受此优惠，且有趣的是，它还包含了 Claude 以增强功能。
- **Granite 3.1-8B-Instruct 令用户印象深刻**：用户对 **Granite 3.1-8B-Instruct 模型**感到兴奋，该模型针对长上下文任务进行了微调，在实际应用中表现出色。
   - 相关模型资源可在 [Granite GitHub](https://github.com/ibm-granite/granite-3.1-language-models) 和 [文档](https://www.ibm.com/granite/docs/)中找到。
- **LM Studio 提供便捷的模型访问**：[LM Studio](https://lmstudio.ai/) 允许用户在本地运行 LLM，与文档聊天，并从 Hugging Face 下载模型文件。
   - 它支持 Llama 3.2、Mistral 和 Qwen 2.5 等架构，满足那些需要离线功能的用户需求。
- **实验 Model Context Protocol**：一位用户计划用 Bash 实现一个快速服务器，以测试 **Model Context Protocol**，尽管最初持保留意见。
   - 该实验旨在评估该协议在现实环境中的实际价值。
- **Gemini Flash Thinking 表现惊艳**：Gemini 2.0 Flash Thinking 针对星球大战中“Hello there!”的梗给出了风趣的回应，突显了其上下文相关性。
   - 最终的回应巧妙地融入了文化细微差别和角色特征，展示了该模型引人入胜的能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/code/status/1869449373995708703">来自 Visual Studio Code (@code) 的推文</a>: 宣布 GitHub Copilot 免费！GitHub Copilot 的全新免费层级，今天在 @code 中面向所有人开放。无需试用。无需订阅。无需信用卡。在我们的博客中了解更多：http://aka.ms/copilot...</li><li><a href="https://huggingface.co/ibm-granite/granite-3.1-8b-instruct">ibm-granite/granite-3.1-8b-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=ZooojV4ZDMw"> - YouTube</a>: 未找到描述</li><li><a href="https://lmstudio.ai/">LM Studio - 发现、下载并运行本地 LLM</a>: 在你的电脑上本地运行 Llama, Mistral, Phi-3。
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1319041270777249862)** (2 messages): 

> `Agent 消息格式化, 微调数据集一致性` 


- **Agent 消息缺乏句子分隔**：一位成员注意到 Agent 的最新消息在句子之间缺少句号，这表明存在格式上的奇特之处。
   - 他们将此行为与 **gpt-4o** 进行了对比，确认后者不存在同样的问题。
- **在微调中使用统一的指令**：一位成员询问了在由“Question”和“Answer”对组成的微调数据集中使用相同指令的影响。
   - 他们的担忧集中在与多样化指令相比，这种方法是否会导致**模型性能欠佳**。


  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1319241909524238367)** (2 messages): 

> `Genesis Project, Generative Physics Engine, Open Source Robotics Simulation` 


- **Genesis Project 以真实物理彻底改变机器人技术**：[Genesis 项目](https://x.com/zhou_xian_/status/1869511650782658846)已发布，这是一个能够创建 **4D 动力学世界**的生成式物理引擎，显著增强了机器人和具身智能（Physical AI）应用。
   - 该项目采用纯 Python 开发，其仿真速度比实时快达 **430,000 倍**，在单张 RTX4090 上训练机器人运动策略的时间缩短至仅 **26 秒**。
- **Genesis 物理引擎开源获取**：Genesis 物理引擎已[完全开源](https://github.com/Genesis-Embodied-AI/Genesis)，邀请社区合作和贡献以增强其功能。
   - 它集成了先进的物理求解器来模拟整个物理世界，旨在实现机器人技术的完全**自动化数据生成**过程。
- **Genesis 机器人运动教程**：一份全面的[教程](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/locomotion.html)解释了如何利用 Genesis 物理引擎训练机器人运动策略。
   - 该训练过程展示了引擎的高效性，比 Isaac Gym 等现有的 GPU 加速方案快 **10-80 倍**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/zhou_xian_/status/1869511650782658846">来自 Zhou Xian (@zhou_xian_) 的推文</a>: 你所喜爱的关于生成式模型的一切 —— 现在由真实物理驱动！宣布 Genesis 项目 —— 经过涉及 20 多个研究实验室、为期 24 个月的大规模研究合作 —— 一个生成式...</li><li><a href="https://github.com/Genesis-Embodied-AI/Genesis">GitHub - Genesis-Embodied-AI/Genesis: 一个用于通用机器人和具身智能学习的生成式世界。</a>: 一个用于通用机器人和具身智能学习的生成式世界。 - Genesis-Embodied-AI/Genesis
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1319209126903152700)** (37 messages🔥): 

> `Mojo Indexing and Casting, SIMD Keying in Dict, Running Mojo on Android, Python Integration Ideas, Negative Indexing Debate` 


- **关于 Mojo 索引实践的辩论**：围绕在 **Mojo** 中使用 **Int** 进行索引展开了讨论，观点在负索引是否应集成到默认实现中，还是使用 `.last()` 等替代方案即可之间产生了分歧。
   - *Darkmatter* 认为负索引通常是编程错误，声称它会引入不必要的运行成本，而其他人则强调了它在 **Python** 等语言中的普遍用法。
- **SIMD 结构体和 Dict 中的 Bug**：提到了 **Mojo** 中一个关于缺失 **scaling_cur_freq** 的重大 Bug，当使用基于 SIMD 的结构体作为 **Dicts** 的键时会导致段错误（segmentation faults），同时也影响了基准测试。
   - 该 Bug 记录在 [GitHub Issue #3781](https://github.com/modularml/mojo/issues/3781) 中，详细说明了重现步骤，并寻求在建议的 **6 周窗口期**内解决。
- **在原生 Android 上运行 Mojo**：一些成员讨论了在原生 Android 上运行 **Mojo** 的可能性，提到了通过 Docker 容器中的 **Magic** 进行设置，尽管这被认为是“完全不受支持的”。
   - 有人指出，虽然可以自行设置，但许可规则禁止创建可公开分发的 Docker 镜像。
- **Python 集成考量**：有关于为 Mojo 创建 Python 类型的咨询，特别是研究了 **SIMD** 的集成和条件一致性（conditional conformance），以实现对各种数据类型的支持。
   - 由于 ABI 的要求，人们对维持整型和浮点型的独立处理表示担忧，而支持任意位宽整数的想法则受到了热烈欢迎。
- **索引的安全性和效率**：关于索引的讨论引发了安全担忧，讨论了从 **UInt** 到 **Int** 的隐式类型转换的影响，以及检查负索引相关的性能成本。
   - *Darkmatter* 建议，虽然可以实现重载，但它们会使现有的类型转换规则复杂化，并可能引入歧义。



**提到的链接**: <a href="https://github.com/modularml/mojo/issues/3781">[BUG] 如果使用基于 SIMD 的结构体作为 Dict 的键，则会出现段错误 · Issue #3781 · modularml/mojo</a>: Bug 描述 当使用包含足够大 SIMD 的结构体作为 Dict 的键时，会遇到段错误。重现步骤 执行以下代码: from collections impor...

  

---

### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1319059635214487624)** (28 条消息🔥): 

> `Synthetic Data Primer, Rate Limiting in DataBricks, DSPy Signature Outputs, Provisioned Throughput Costs, LiteLLM Proxy Layer` 


- **正在编写合成数据解释指南**：一名成员正在编写一份关于**合成数据（Synthetic Data）**基础知识的解释指南，涵盖其创建方式、用途以及对模型能力的影响。
   - 他们正在征求社区对合成数据特定问题或感兴趣领域的反馈。
- **DataBricks 中的速率限制解决方案**：一名成员讨论了由于吞吐量分配产生的高昂成本，在 DataBricks 中实现**速率限制器（rate limiter）**的可能性。
   - 另一位成员建议使用 **LiteLLM** 代理层来实现速率限制和预算控制等功能。
- **关于 DSPy Signature 类输出的问题**：一位用户询问了将 **dspy.Signature** 作为类类型而非字符串生成的示例，并表示有兴趣使用 DSPy 框架。
   - 他们正在探索直接返回带有指定字段的 signature 的可行性。
- **对预置吞吐量成本的担忧**：一位成员讲述了他们在 DataBricks 中因预置吞吐量（Provisioned Throughput）产生**高昂成本**的经历，引发了对不必要费用的担忧。
   - 他们明确了启用 **scale to 0** 选项的重要性，以便在不使用时避免产生费用。
- **在 DataBricks 中部署 LiteLLM**：讨论了 **LiteLLM proxy** 是否可以部署在 DataBricks notebook 中，还是需要单独的 VM。
   - 一位成员确认 LiteLLM 可以在受控环境中与服务一起管理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.databricks.com/product/pricing/foundation-model-serving">Mosaic AI Foundation Model Serving</a>：未找到描述</li><li><a href="https://docs.databricks.com/en/ai-gateway/index.html">Mosaic AI Gateway</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1319064531049386046)** (3 条消息): 

> `Multi-agent systems, Vectara RAG capabilities, AI journey survey` 


- **使用 LlamaIndex 构建多智能体系统**：一篇文章讨论了如何使用 LlamaIndex 从**单智能体（single agent）**演进到协同的**多智能体系统（multi-agent system）**，并提供了实际代码示例。
   - 文章强调了**智能体工厂（agent factories）**在此转型中的重要性，详见全文 [此处](https://t.co/lbhFDbSabS)。
- **解锁 Vectara 的 RAG 能力**：探索如何利用 **Vectara** 强大的 **RAG 能力**，包括数据加载以及带有流式传输和重排序（reranking）选项的查询。
   - 该文章介绍了构建**智能体 RAG 应用（agentic RAG applications）**的方法，同时突出了 Vectara 托管服务的全部功能，详见 [此处](https://t.co/traVaQiUt3)。
- **参与 Vercel 的 AI 现状调查**：一项行动呼吁邀请社区成员通过 @vercel 的 **State of AI Survey** 分享他们在 **AI 历程**中的进展。
   - 参与者可以通过访问 [此链接](https://t.co/O3sYZ6L9Gq) 为理解 AI 领域现状做出贡献。



**提到的链接**：<a href="https://t.co/O3sYZ6L9Gq">State of AI Developer Survey</a>：分享您的经验、挑战和见解，帮助塑造 AI 驱动创新的未来。

  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1319241784211013642)** (23 条消息🔥): 

> `HuggingFaceEmbedding 模型加载、Azure OpenAI embedding 速率限制、TextNode 插入错误` 


- **HuggingFaceEmbedding 无法从本地加载**：一位用户在尝试从本地存储加载 HuggingFace embedding 模型时遇到问题，并收到关于使用 mean pooling 创建新模型的警告。
   - 另一位用户澄清说，只需提供模型名称，系统会先检查缓存文件夹，而不会进行不必要的下载。
- **Azure OpenAI embedding 速率限制的解决方案**：一位用户报告了 Azure OpenAI embedding 模型的持续速率限制错误，并寻求解决此问题的建议。
   - 建议包括增加 max retries（最大重试次数）并减慢文档摄取速度，以避免速率限制问题。
- **关于插入 TextNodes 的困惑**：一位用户在尝试将 TextNodes 插入索引时遇到 `AttributeError`，提示缺少 `get_doc_id` 属性。
   - 建议指出插入节点的正确方法是 `insert_nodes`，且逐个处理节点可能有助于缓解速率限制。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/huggingface/.">使用 HuggingFace 的本地 Embeddings - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/7879">[问题]：构建索引时持续出现速率限制错误 · Issue #7879 · run-llama/llama_index</a>：问题验证。我已在文档和 Discord 中搜索过答案。问题：我正使用基础代码对一个约 10 行的单个文本文件进行索引，代码来自 llama_index import ...
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1319351653492850729)** (1 条消息): 

> `Vision Parse，PDF 转 Markdown` 


- **用于 Markdown 转换的 Vision Parse 库发布**：一位成员分享了 [Vision Parse](https://github.com/iamarunbrahma/vision-parse) 的发布，这是一个开源 Python 库，利用先进的 Vision Language Models 将 PDF 文档转换为格式良好的 Markdown 内容。
   - *State-of-the-art* 技术旨在通过**出色的格式化**选项提升转换体验。
- **对开源贡献的热情**：社区对 Vision Parse 的发布表现出极大的热情，强调了其在简化开发者文档处理方面的潜力。
   - 成员们讨论了*开源项目*在促进技术领域创新与协作方面的重要性。


  

---


### **Nomic.ai (GPT4All) ▷ #[announcements](https://discord.com/channels/1076964370942267462/1090471714888102009/1319137980471836713)** (1 条消息): 

> `数据映射系列、可扩展图形、Embeddings、降维、非结构化数据` 


- **数据映射系列最终篇发布**：**Nomic 团队**宣布发布**数据映射系列**的最后一篇，重点关注用于管理 Embeddings 和非结构化数据的**可扩展图形**，阅读地址见[此处](https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers)。
   - 该系列详细介绍了 **Embeddings** 和**降维**等机器学习概念如何赋能用户在 Web 浏览器中可视化海量数据集。
- **六部分数据映射探索**：最新文章总结了旨在阐明 **Nomic Atlas 平台**在**非结构化数据可视化**方面背后技术的六部分系列。
   - 鼓励读者查看该系列的前几部分，涵盖了[数据地图](./data-mapping)、[Embeddings](./embeddings-are-for-so-much-more-than-rag) 和[降维](./see-your-data-with-dimensionality-reduction)等基础知识。



**提到的链接**：<a href="https://www.nomic.ai/blog/posts/why-are-web-browsers-the-best-data-browsers">数据地图第四部分：为什么 Web 浏览器是最好的数据浏览器？</a>：为什么 Web 浏览器是最好的数据浏览器？

  

---

### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1319055038013771806)** (17 条消息🔥): 

> `Nomic BERT 问题, Code Interpreter Pull Request, 加载 System Messages, GGUF 文件问题, 设备需求` 


- **Nomic BERT Embedding 模型问题**: 用户报告了从 Huggingface 加载 **Nomic's embedding model** 时出现错误，原因是最近的一个 [commit](https://huggingface.co/nomic-ai/nomic-bert-2048/commit/ba22e9d89df6236d83c3daa26cc8dd78a130c3f2) 破坏了功能。幸运的是，该问题现已修复。
- **Code Interpreter 工具的 Pull Request**: 一个名为 [Code interpreter by manyoso](https://github.com/nomic-ai/gpt4all/pull/3173) 的 Pull Request 正在进行中，旨在添加一个基于 jinja 模板的 Code Interpreter 工具。成员们表示有兴趣关注其进展。
- **关于加载 System Messages 的讨论**: 一位用户询问是否可以增加一个 `load` 按钮来从文本文件加载 System Messages，并对频繁的复制粘贴表示困扰。由于用户拥有许多用于设置上下文的文本文件，因此对该功能有明显需求。
- **GGUF 文件兼容性问题**: 讨论涉及多个 **.GGUF** 文件的 chat templates 损坏问题，提到了 **Llama-3.3-70B-Instruct-Q4_K_M** 和 **Qwen2-72B-Instruct.Q4_K_M** 等文件。这些文件的修复方案承诺将在下一个版本中发布。
- **GPT4All 的设备需求**: 一位用户询问了 **GPT4All** 的设备需求，另一位成员分享了官方 [system requirements](https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md) 的链接。该文档概述了运行 GPT4All 所需的规格。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/nomic-ai/gpt4all/blob/main/gpt4all-chat/system_requirements.md">gpt4all/gpt4all-chat/system_requirements.md at main · nomic-ai/gpt4all</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可商用。 - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3173">Code interpreter by manyoso · Pull Request #3173 · nomic-ai/gpt4all</a>: 这是基于 jinja PR 的 Code Interpreter 工具调用的 WIP。这是我目前为 Qwen2.5-Coder-7B 使用的最新 jinja 模板:&#123;&#123;- &amp;#39;&amp;lt;|im_start|&amp;gt;system\n&amp;#39; }...
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1319057454881050714)** (16 messages🔥): 

> `TinyChat 安装问题、Tiktoken 替代方案讨论、滚动方向 Bug 报告、Bounty 项目参与、布局符号见解` 


- **TinyChat 安装面临障碍**：在尝试设置 TinyChat 后，一位用户报告了缺少 **tiktoken** 等依赖项的问题，并在安装过程中经历了 **~30 秒** 的系统冻结。
   - 他们还注意到一个关于*在本地网络上寻找设备*的奇怪提示，并质疑其必要性。
- **Tiktoken 需要定制化的替代方案**：George Hotz 承认需要 **tiktoken 的替代方案**，并提出了是否可以直接在 TinyGrad 中编写该方案的问题。
   - 他将 **8GB RAM** 的限制作为讨论的一个关键点。
- **滚动方向意外切换**：一位用户报告了一个奇怪的问题，在运行 TinyChat 后，其 Mac 上的**滚动方向**发生了反转，在终止应用程序后恢复正常。
   - George Hotz 对此问题表示惊讶，确认这确实令人费解。
- **Bounty 项目参与策略**：Chenyu 提到悬赏的目标是**推进项目**，并强调要与那些通过测试和改进增加价值的贡献者互动。
   - 他们指出，以测试和优化讨论形式进行的贡献对于推动进度至关重要。
- **关于布局符号（layout notation）的讨论**：一位用户分享了关于布局符号虽然强大但很**复杂**的想法，并指出文档中图形表示的有效性。
   - 他们强调，**补集部分（complement section）**通过描述所有未被选中的元素提供了一个独特的视角，这与传统的 mask 不同。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/1RRuhAW-I5u_6ssbm1eLqVWSK_PYFoARIVBYwYZwmcM4/edit?usp=drivesdk">View Merges</a>：相关代码：https://github.com/unknownusername504/MicroGrad/blob/main/micrograd/tensors/viewable_tensor.py  View 的目标是什么？移动内存是一项昂贵的操作。如果你有...</li><li><a href="https://github.com/tinygrad/tinygrad/issues/8194))">Issues · tinygrad/tinygrad</a>：你喜欢 PyTorch？你喜欢 MicroGrad？你一定会爱上 TinyGrad！❤️ - Issues · tinygrad/tinygrad
</li>
</ul>

</div>
  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/)** (1 messages): 

khaner2162: 你好，
为什么调度器（scheduler）要 `# realize before expand or unsafe pad ops`（在 expand 或 unsafe pad 操作之前进行 realize）？
  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1319302126484062301)** (6 messages): 

> `Ikuo618 的介绍、频道礼仪提醒` 


- **Ikuo618 自我介绍**：Ikuo618 分享了他的背景，他是一名资深 AI 开发者，在 **DP**、**NLP** 和 **CV** 领域拥有超过 **6 年** 的 AI 模型构建和部署经验。
   - 他强调了自己在 **Python** 方面的专长，以及熟练使用 **TensorFlow** 和 **PyTorch** 开发智能系统的能力。
- **关于重复发帖的礼仪提醒**：向一位用户发出了提醒，要求不要在多个频道重复发布相同消息，以保持聊天内容的整洁。
   - 这旨在引导大家在参与讨论时遵守频道礼仪。


  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1319356136650178612)** (2 messages): 

> `平台可用性` 


- **平台状态确认**：一位成员询问某项特定功能是否在平台上可用，另一位成员确认该功能**尚未上线平台**。
   - 询问的成员用笑脸表达了对确认的感谢。
- **确认过程中的用户互动**：这次互动展示了友好的交流，一名用户确认了平台功能的缺失。
   - 这一回应突显了用户之间的积极互动，一方对确认表示了感谢。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1319117786731118674)** (3 messages): 

> `Cohere API pricing, API keys types, Rate limits for endpoints` 


- **Cohere API 提供免费和付费密钥**：Cohere 提供两种类型的 API keys：**评估密钥 (evaluation keys)** 是免费的，但使用次数有限；**生产密钥 (production keys)** 是付费的，限制要少得多。
   - 用户可以在 [API keys 页面](https://dashboard.cohere.com/api-keys) 创建这些密钥，并在 [定价文档 (pricing docs)](https://docs.cohere.com/v2/docs/how-does-cohere-pricing-work) 中查看定价详情。
- **Cohere API 的详细速率限制 (rate limits)**：Cohere API 为每个端点 (endpoint) 设置了特定的速率限制，试用限制显著低于生产限制；例如，**Chat** 端点对试用用户限制为 **每分钟 20 次调用**，而对生产用户为 **每分钟 500 次**。
   - 其他端点如 **Embed** 和 **Classify** 也有不同的限制，所有端点的累计限制为 **每月 1,000 次调用**。



**提及的链接**：<a href="https://docs.cohere.com/docs/rate-limits">API Keys and Rate Limits — Cohere</a>：此页面描述了 Cohere API 针对生产和评估密钥的速率限制。

  

---


### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/)** (1 messages): 

ikuo618: hi..................!
  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/)** (1 messages): 

benny0917: 产品看起来很棒 <@799853279017173033> 恭喜！
  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1319068909693042708)** (6 messages): 

> `Torchtune Phi 4 Support, New Contributor Role, Implementation Differences Between Phi 3 and Phi 4` 


- **Torchtune 目前缺乏 Phi 4 支持**：一位成员询问关于使用 **Torchtune** 支持 **Phi 4** 的情况，得到的确认是目前仅支持 **Phi 3**，并欢迎针对 Phi 4 的贡献。
   - 成员们表达了对潜在贡献以启用 **Phi 4** 支持的兴趣。
- **引入新的 Contributor 角色**：Discord 上推出了新的 **Contributor** 角色，以表彰那些为所有人改进 **Torchtune** 的社区成员。
   - 该计划旨在认可贡献，并建立 **GitHub** 与 **Discord** 用户名之间的联系。
- **预计 Phi 4 的差异极小**：关于 **Phi 3** 和 **Phi 4** 之间的实现差异展开了讨论，一位成员指出它们看起来差异非常小。
   - 分享的一张图片似乎支持了这一观点，引发了对这些变化的进一步好奇。



**提及的链接**：<a href="https://pytorch.org/torchtune/stable/api_ref_models.html">torchtune.models &mdash; torchtune 0.4 documentation</a>：未找到描述。

  

---

### **Torchtune ▷ #[papers](https://discord.com/channels/1216353675241590815/1293438210097025085/1319088822167801936)** (2 messages): 

> `异步 RLHF、训练后技术、模型安全与鲁棒性` 


- **RLHF 中的异步方法**：传统的 RLHF 方法**计算效率较低**；然而，正如 [online but off-policy RLHF](https://arxiv.org/abs/2410.18252) 研究中所建议的，将生成与学习分离可以实现更快、异步的模型训练。
   - 该研究强调了一个关键问题：在确保高效学习且不牺牲性能的前提下，*我们能容忍多大程度的离策性 (off-policyness)*。
- **模型训练后 (Post-Training) 的重要性**：模型在预训练阶段之后需要进行**训练后处理**，以确保其安全并能有效地遵循人类指令，正如 [Allen AI 博客](https://allenai.org/blog/tulu-3) 中所讨论的。
   - 这一过程涉及指令微调和从人类反馈中学习，以避免在专业化过程中削弱核心能力。
- **指令微调的挑战**：最初受 **InstructGPT** 启发的训练后方法，可能会随着教授更多专业技能而导致某些模型能力的下降。
   - 在增强 **coding** 等能力与保留 **poetry 和指令遵循** 技能之间寻找平衡，仍然是一个复杂的挑战。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2410.18252">Asynchronous RLHF: Faster and More Efficient Off-Policy RL for Language Models</a>：RLHF 的主流范式是在线且同策 (on-policy) 的 RL：同步地从大语言模型 (LLM) 策略中生成，使用奖励模型进行标注，并利用 LLM 的反馈进行学习...</li><li><a href="https://allenai.org/blog/tulu-3">Tülu 3 opens language model post-training up to more tasks and more people  | Ai2</a>：Tülu 3 是一个领先的指令遵循模型系列，提供完全开源的数据、代码和配方，旨在作为现代训练后技术的全面指南。
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1319370418674597938)** (1 messages): 

> `黑客松提交截止日期、LLM Agents 黑客松、最后提醒、项目提交、最后时刻提问` 


- **黑客松提交最后召集！**：已发布提醒，黑客松的**提交截止日期**为今晚 **11:59 PM PST** (12/19)。
   - *请确保您的项目已通过* [LLM Agents Hackathon 提交表单](https://docs.google.com/forms/d/e/1FAIpQLSe3Y5BMGJFdI3PUIM1rtEEGI5u5kxesVxPnjb5rD4iAgSOeVw/viewform)完成提交！
- **支持最后时刻的提问**：鼓励参与者在频道中提出任何**最后时刻的问题**以寻求帮助。
   - 社区正在集结力量，确保每个人都能在截止日期前*完美收官*。


  

---


---


---


---


---


---


---


{% else %}


> 完整的频道逐项解析已针对邮件进行截断。
> 
> 如果您想查看完整内容，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}