---
companies:
- cohere
- anthropic
- openai
- microsoft
- stability-ai
- opera-software
- meta-ai-fair
- google-deepmind
- mistral-ai
date: '2024-04-04T22:21:15.996359Z'
description: '**Cohere** 推出了 **Command R+**，这是一个拥有 **1040亿参数的稠密模型**，具备 **128k 上下文长度**，专注于
  **RAG（检索增强生成）**、**工具调用**以及涵盖 **10 种主要语言**的**多语言**能力。它支持**多步工具调用**，并开放了研究权重。**Anthropic**
  为 **Claude** 引入了**工具调用测试版（beta）**，支持超过 **250 种工具**，并发布了用于实际应用的新指南（cookbooks）。**OpenAI**
  升级了其微调 API，并分享了来自 Indeed、SK Telecom 和 Harvey 的案例研究，以推广自助微调和定制模型训练。**微软**在量子计算领域取得突破，将**错误率降低了
  800 倍**，并实现了迄今为止最实用的量子比特。**Stability AI** 发布了 **Stable Audio 2.0**，提升了音频生成的质量和控制力。**Opera
  浏览器**增加了对 **Meta 的 Llama**、**谷歌的 Gemma** 和 **Vicuna** 等大语言模型的本地推理支持。Reddit 上的讨论聚焦于
  **Gemini 的大上下文窗口**、**GPT-3.5-Turbo** 模型规模的分析，以及使用 **Mistral** 和 **Gemma** 等本地 7B
  模型进行的 **Claude 3** 与 **ChatGPT** 之间的对战模拟。'
id: c2eed8e3-e60a-4b34-b7b5-4f7d26aa5c66
models:
- c4ai-command-r-plus
- claude-3
- gpt-3.5-turbo
- gemini
- mistral-7b
- gemma-2
- claude-3-5
- llama-3
- vicuna
original_slug: ainews-cohere-command-r-anthropic-claude-tool-use
people: []
title: Cohere Command R+、Anthropic Claude 工具使用、OpenAI 微调
topics:
- tool-use
- multilingual-models
- rag
- fine-tuning
- quantum-computing
- audio-generation
- local-inference
- context-windows
- model-size-analysis
- model-comparison
---

<!-- buttondown-editor-mode: plaintext -->> 2024年4月3日至4月4日的 AI 新闻。我们为您检查了 5 个 subreddits、[**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **26** 个 Discord 社区（**385** 个频道，**5656** 条消息）。预计节省阅读时间（按 200wpm 计算）：**639 分钟**。

今天非常忙碌。

1. 获得了[至少 5 亿美元融资](https://twitter.com/steph_palazzolo/status/1773095998555898305)的 Cohere 推出了上个月 [Command R](https://twitter.com/aidangomez/status/1767264315550163024?t=6FDPaNxZcbSsELal6Sv7Ug) 的快速跟进版本 Command R+（[官方博客](https://txt.cohere.com/command-r-plus-microsoft-azure/)，[权重](https://huggingface.co/CohereForAI/c4ai-command-r-plus)）。这是一个拥有 104B 参数的稠密模型，具有 128k 上下文长度，专注于 RAG、tool-use 和多语言（“[10 种核心语言](https://x.com/cohere/status/1775878865631498360)”）用例。权重对研究开放，但 Aidan 表示，如果你想获得授权（而不是支付其 [每百万 token $3/$15 的价格](https://x.com/SelfInfinity/status/1775881659058946416)），“[直接联系即可](https://x.com/aidangomez/status/1767264324626559249)”。它现在支持 [Multi-Step Tool use](https://x.com/cohere/status/1775878859033858346)。
 
![image.png](https://assets.buttondown.email/images/823c645a-957f-47cf-bbb5-bd814cd4114e.png?w=960&fit=max)
 
2. 获得了 [27.5 亿美元融资](https://www.maginative.com/article/amazon-completes-massive-4-billion-investment-in-ai-startup-anthropic/) 的 Anthropic 按照之前的承诺[推出了 tool use 的 Beta 版本](https://twitter.com/AnthropicAI/status/1775979802627084713)（[官方文档](https://docs.anthropic.com/claude/docs/tool-use)）。详尽的文档包含许多值得注意的特性，最显著的是[宣传其处理超过 250 个工具的能力](https://twitter.com/swyx/status/1775993946935906645)，这实现了一种与以往截然不同的 function calling 架构。这大概归功于过去一年中上下文长度和召回率（recall）的提升。更多详情请参阅其 3 本新的 cookbook：

  - [在 Claude 中使用计算器工具](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/calculator_tool.ipynb)
  - [使用客户端工具创建客户服务 Agent](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/customer_service_agent.ipynb)
  - [使用 Claude 和 tool use 提取结构化 JSON](https://github.com/anthropics/anthropic-cookbook/blob/main/tool_use/extracting_structured_json.ipynb)

3. OpenAI（据我们所知，上个月没有进行任何融资）为原本非常 MVP 式的微调体验[增加了一系列备受欢迎的升级](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program)，并发布了与 Indeed、SK Telecom 和 [Harvey](https://openai.com/customer-stories/harvey) 合作的三个案例研究，基本上是在说“你现在可以更好地 DIY，但我们也欢迎你来找我们微调和训练你的模型”。

 
![image.png](https://assets.buttondown.email/images/75751b5a-56c6-4d64-a4c8-a64dab0c10b8.png?w=960&fit=max)
 

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取功能尚未实现，但即将推出。

**AI 技术进展**

- **量子计算突破**：在 /r/singularity 中，微软在量子计算领域取得了突破，将错误率降低了 800 倍，并拥有迄今为止最实用的量子比特，[**这是量子计算能力的重大进步**](https://v.redd.it/wp1djpidpbsc1)。
- **Stable Audio 2.0 发布**：在 /r/StableDiffusion 中，Stability AI 推出了 Stable Audio 2.0，通过提升质量和控制力[**增强了音频生成能力**](https://stability.ai/news/stable-audio-2-0)。
- **浏览器集成大语言模型**：在 /r/LocalLLaMA 中，Opera 浏览器已添加对[**在本地运行 Meta 的 Llama、Google 的 Gemma 和 Vicuna 等大语言模型**](https://www.reddit.com/r/LocalLLaMA/comments/1buu5v1/opera_has_added_local_llm_inference_to_their/)的支持，使其更易于访问。

**模型能力与对比**

- **Gemini 的超大上下文窗口**：在 /r/ProgrammerHumor 中，一张图片强调了 [**Gemini 的 Context Window 远大于其他模型**](https://i.redd.it/o2appsoeyasc1.png)，从而能够实现更强的上下文理解。
- **GPT-3.5-Turbo 模型规模分析**：在 /r/LocalLLaMA 中，分析表明 [**GPT-3.5-Turbo 可能是一个 8x7B 模型，规模与 Mixtral-8x7B 相似**](https://www.reddit.com/r/LocalLLaMA/comments/1bv9kag/gpt35turbo_is_most_likely_the_same_size_as/)。
- **Claude 3 vs ChatGPT 对战模拟**：在 /r/LocalLLaMA 中，一段视频展示了 Claude 3 与 ChatGPT 在“街头霸王”风格战斗中的对比，[**使用了 Mistral 和 Gemma 等本地 7B 模型**](https://v.redd.it/joikvgj2l9sc1)。

**AI 研究与教育**

- **斯坦福 Transformers 课程向公众开放**：在 /r/StableDiffusion 中，斯坦福大学的 CS 25 Transformers 课程向公众开放，[**邀请顶尖研究人员讨论架构、应用等方面的突破**](https://www.reddit.com/r/StableDiffusion/comments/1bve5kp/stanford_cs_25_transformers_course_open_to/)。
- **股票预测研究的挑战**：在 /r/MachineLearning 中，一场讨论探讨了[**为什么股票预测研究论文通常无法转化为现实世界的生产应用**](https://www.reddit.com/r/MachineLearning/comments/1bv0cu7/why_stock_prediction_papers_arent_put_to/)。
- **检索增强生成辩论**：在 /r/MachineLearning 中，关于 [**检索增强生成（RAG）是否只是美化版的 Prompt Engineering**](https://www.reddit.com/r/MachineLearning/comments/1busp41/d_is_rag_just_glorified_prompt_engineering/) 引发了辩论。

**AI 工具与应用**

- **GPT-4-Vision 用于在线模仿**：在 /r/singularity 中，一段视频演示了[**如何使用 GPT-4-Vision 在电子邮件或任何网站上一键模仿用户本人**](https://v.redd.it/h1g82xgyi8sc1)。
- **自动视频精彩片段检测**：在 /r/singularity 中，展示了一个[**通过自定义搜索词自动在长视频中寻找精彩片段**](https://v.redd.it/halkizhh4asc1)的工具。
- **Daz3D AI 驱动图像生成**：在 /r/StableDiffusion 中，Daz3D 与 Stability AI 合作推出 [**Daz AI Studio，用于根据文本生成风格化图像**](https://www.reddit.com/r/StableDiffusion/comments/1bvb88n/daz3d_partnering_with_sai/)。

**AI 梗与幽默**

- **Gemini 上下文窗口梗**：在 /r/ProgrammerHumor 中，一张[**幽默图片描绘了“Gemini 的 Context Window 比其他任何人都大”**](https://i.redd.it/o2appsoeyasc1.png)。
- **《超级银河战士》恶搞预告片**：在 /r/singularity 中，使用 Dalle3 和 GPT 制作了一个 [**《超级银河战士》（Super Metroid）恶搞电影预告片**](https://v.redd.it/galfcs7gzasc1)。
- **卧室二维码梗**：在 /r/singularity 中，分享了一张[**卧室二维码梗图**](https://i.redd.it/arudc6zav8sc1.png)。

# AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和 Flow Engineering。

**Cohere Command R+ 发布**

- **新型开源模型**：[@cohere](https://twitter.com/cohere/status/1775878850699808928) 发布了 Command R+，这是一个拥有 104B 参数、128k 上下文长度的模型，其权重针对非商业用途开放，并具备强大的多语言和 RAG 能力。该模型已在 [Cohere playground](https://twitter.com/cohere/status/1775878883268509801) 和 [Hugging Face](https://twitter.com/osanseviero/status/1775882744792273209) 上线。
- **针对 RAG 工作流优化**：Command R+ 针对 [RAG 进行了优化](https://twitter.com/aidangomez/status/1775878606108979495)，具备拆解复杂问题的 **multi-hop** 能力和强大的 **tool use**（工具调用）能力。它已与 [@LangChainAI](https://twitter.com/cohere/status/1775931339361149230) 集成，用于构建 RAG 应用程序。
- **多语言支持**：Command R+ 在包括英语、法语、西班牙语、意大利语、德语、葡萄牙语、日语、韩语、阿拉伯语和中文在内的 10 种语言中表现出 [强劲性能](https://twitter.com/seb_ruder/status/1775882934542533021)。[@JayAlammar](https://twitter.com/JayAlammar/status/1775928159784915229) 指出，其 **tokenizer** 对 **阿拉伯语和其他非英语语言** 更为高效，所需的 **token** 更少，从而降低了成本。
- **定价与可用性**：[@cohere](https://twitter.com/cohere/status/1775878850699808928) 提到 Command R+ 在可扩展市场类别中处于领先地位，**助力企业投入生产**。它已在 Microsoft Azure 上可用，并将很快登陆其他云服务提供商。[@JayAlammar](https://twitter.com/JayAlammar/status/1775881793796726808) 补充道，凭借 **multi-hop** 能力，它将 RAG 提升到了一个新高度。
- **LangChain 集成**：[@hwchase17](https://twitter.com/hwchase17/status/1775922998853414961) 和 [@LangChainAI](https://twitter.com/LangChainAI/status/1775889394916049230) 宣布推出 `langchain-cohere` 软件包，以提供 **chat models** 和 **model-specific agents** 等集成功能。[@cohere](https://twitter.com/cohere/status/1775931339361149230) 对该集成在 **adaptive RAG** 方面的应用感到兴奋。
- **Hugging Face 与性能**：[@osanseviero](https://twitter.com/osanseviero/status/1775882744792273209) 指出该模型已在 Hugging Face 上提供，并附带了 playground 链接。[@seb_ruder](https://twitter.com/seb_ruder/status/1775882934542533021) 强调了其 **10 种语言的多语言能力**。[@JayAlammar](https://twitter.com/JayAlammar/status/1775928159784915229) 提到了针对 **阿拉伯语等语言的 tokenizer 优化**，以降低成本。
- **微调与效率**：[@awnihannun](https://twitter.com/awnihannun/status/1775942513653924049) 展示了 **在 M2 Ultra 上使用 MLX 通过 QLoRA 对 Command R+ 进行 fine-tuning**。[@_philschmid](https://twitter.com/_philschmid/status/1775894028707639357) 总结了这个 104B 模型的特点：开放权重、支持 RAG 和 **tool use** 以及多语言支持。

**DALL-E 3 Inpainting 发布**

- **新功能**：[@gdb](https://twitter.com/gdb/status/1775780196517548335) 和 [@model_mechanic](https://twitter.com/model_mechanic/status/1775590691487556064) 宣布 DALL-E 3 **inpainting**（局部重绘）现已对所有 ChatGPT Plus 订阅用户开放。这允许用户根据文本指令 **编辑和修改图像的局部**。
- **使用方法**：[@chaseleantj](https://twitter.com/chaseleantj/status/1775493065677169147) 提供了一个指南：涂抹需要替换的区域，输入描述更改的 **prompt**；为了获得最佳效果，请勿涂抹所有文字。目前仍存在一些 **局限性**，例如无法在空白区域生成文字。

**用于高效 Transformer 的 Mixture-of-Depths**

- **方法**：[@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1775743788402479323) 分享了 Google 的 Mixture-of-Depths 方法，用于在 Transformer 模型中 **动态分配计算资源**。它通过限制每一层 **self-attention/MLP** 中的 **token** 数量来强制执行总计算预算。
- **优势**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1775927231463706773) 解释说，这种方法通过将更多计算资源分配给难以预测的 **token**，而不是像标点符号这样容易预测的 **token**，从而最大限度地减少计算浪费。计算支出在 **总体上是可预测的，但在 token 级别是动态且上下文敏感的**。

**RAG 与 Agent 进展**

- **自适应 RAG 技术**：[Adaptive RAG](https://twitter.com/LangChainAI/status/1775917799065653250) 和 [Corrective-RAG](https://twitter.com/llama_index/status/1775912690529288556) 等新论文提出根据查询复杂度动态选择 RAG 策略。相关实现已在 [LangChain](https://twitter.com/LangChainAI/status/1775569294241472810) 和 [LlamaIndex](https://twitter.com/llama_index/status/1775912690529288556) 的 cookbook 中提供。
- **基于 RAG 的应用**：基于 RAG 的应用示例包括 AI 驱动的知识库 [Omnivore](https://twitter.com/jerryjliu0/status/1775691578994278719)，以及用于扩展复杂推理的 [Elicit 任务分解架构](https://twitter.com/labenz/status/1775894599179157840)。将 RAG 与 [tool use](https://twitter.com/hwchase17/status/1775922998853414961) 相结合，可以构建更具 Agent 特性的系统。

**开源模型与框架**

- **Anthropic 越狱**：[@AlphaSignalAI](https://twitter.com/AlphaSignalAI/status/1775561052325077218) 分享了 Anthropic 关于“many-shot jailbreaking”的研究，该研究通过构建良性对话来**绕过 LLM 安全措施**。它利用大上下文窗口来生成通常会被规避的响应。
- [@AssemblyAI](https://twitter.com/AssemblyAI/status/1775527556042629437) 推出了 **Universal-1，一个多语言语音识别模型**，在 1250 万小时的数据上训练而成。它在准确率和幻觉率方面优于 Whisper 等模型。
- **开源模型与数据集**：新的开源模型包括来自 01.AI 的 [Yi](https://twitter.com/rohanpaul_ai/status/1775924341923860594)、来自清华大学的 [Eurus](https://twitter.com/rohanpaul_ai/status/1775458159865323810)、来自 AI21 Labs 的 [Jamba](https://twitter.com/maximelabonne/status/1775511912773566733) 以及来自 AssemblyAI 的 [Universal-1](https://twitter.com/AssemblyAI/status/1775527556042629437)。来自 [Hugging Face](https://twitter.com/rohanpaul_ai/status/1775506872520355870) 的大型 OCR 数据集推动了文档 AI 的研究。
- **高效推理技术**：[BitMat](https://twitter.com/rohanpaul_ai/status/1775608234180558851) 减少了量化模型的内存占用。[Mixture-of-Depths](https://twitter.com/arankomatsuzaki/status/1775743788402479323) 在 Transformer 中动态分配计算资源。[HippoAttention](https://twitter.com/rohanpaul_ai/status/1775923372242726995) 和 [MoE 优化](https://twitter.com/rohanpaul_ai/status/1775944589230170350) 加速了推理过程。
- **便捷的模型部署**：[Hugging Face](https://twitter.com/_philschmid/status/1775885996435087449) 降低了托管推理的价格，而 [Koyeb](https://twitter.com/llama_index/status/1775688909042954723) 和 [SkyPilot](https://twitter.com/skypilot_org/status/1775931821257314745) 简化了在任何云平台上部署模型的过程。

**梗与幽默**

- 一段由 AI 生成的[忧郁女孩演唱 MIT License](https://twitter.com/goodside/status/1775713487529922702) 的视频在网上疯传。
- 人们对 [Apple 的 AI 雄心](https://twitter.com/Teknium1/status/1775748185203634388) 进行了猜测，并开玩笑说 [AI 将取代软件工程师](https://twitter.com/bindureddy/status/1775538983688450480)。
- 还有一些梗在调侃 [AI 炒作](https://twitter.com/bindureddy/status/1775920627603657142) 和 [大语言模型的局限性](https://twitter.com/fchollet/status/1775636345190588689)。

# AI Discord 回顾

> 摘要之摘要的摘要

1. **LLM 进展与集成**：
   - [Cohere 发布 Command R+](https://txt.cohere.com/command-r-plus-microsoft-azure/)，这是一款 **104B 参数的多语言 LLM**，专为企业级使用而优化，具备先进的检索增强生成 (RAG) 和多步工具能力，引发了人们对其与其他模型性能对比的关注。
   - [JetMoE-8B](https://research.myshell.ai/jetmoe) 代表了一个成本低于 **10 万美元** 的里程碑，仅使用 **2.2B 激活参数** 就超越了 Meta AI 的 LLaMA2 性能。
   - 关于将 **HQQ 等 LLM 与 gpt-fast** 集成的讨论，探索了 **4/3 bit 量化** 方法，例如 [Mixtral-8x7B-Instruct 量化模型](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ)。

2. **优化 LLM 推理与训练**：
   - [Mixture-of-Depths (MoD)](https://arxiv.org/abs/2404.02258) 使 Transformer 能够**在序列中动态分配计算资源**，相比均匀分布，这可能提高效率。
   - [Visual AutoRegressive (VAR) 建模](https://arxiv.org/abs/2404.02905) 重新定义了自回归图像生成，在质量和速度上均优于 Diffusion Transformers。
   - 根据《The Era of 1-bit LLMs》论文，**BitMat** 等技术提供了 [高效的 1-bit LLM 实现](https://github.com/astramind-ai/BitMat)。

3. **LLM 评估与基准测试**：
   - 评估 LLM 情感智能的新基准：[Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html) 和 [Judgemark](https://eqbench.com/judgemark.html)，使用相关性指标进行衡量。
   - **COMET 分数** 突显了 [Facebook WMT21 模型的翻译实力](https://github.com/CrispStrobe/llm_translation)，其最高分达到了 **0.848375**。
   - 关于 AI 产品**系统化评估实践**的讨论，[Hamel Husain 的文章](https://hamel.dev/blog/posts/evals/) 被视为具有开创性。

4. **开源 AI 框架与工具**：
   - [LlamaIndex](https://www.llamaindex.ai/) 发布了**使用 MistralAI 构建 RAG 系统的指南 (cookbooks)**，包括路由和查询分解。
   - [Koyeb](https://t.co/weFs0waN4o) 通过连接 GitHub 仓库部署 Serverless 应用，实现 **LLM 应用的轻松全球扩展**。
   - [SaladCloud](https://bit.ly/3TFIsKt) 提供**针对 AI/ML 工作负载的托管容器服务**，以避免高昂的云成本。
   - [transformer-heads GitHub 仓库](https://github.com/center-for-humans-and-machines/transformer-heads) 提供了通过附加新的模型头来**扩展 LLM 能力**的工具。

---

# 第一部分：Discord 高层摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **谨慎对待高分辨率**：从业者讨论了 **Stable Diffusion 模型在放大时的最佳设置**，主张使用 35 步以及特定的 Upscalers 和 Control Nets 来减轻图像失真。正如 [Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction](https://arxiv.org/abs/2404.02905) 中所述，更高的分辨率（尤其是 2k）会导致更长的生成时间并可能出现问题。

- **AI 在创意过程中的角色**：关于 AI 对创意产业日益增长的影响进行了热烈讨论，特别是思考其取代好莱坞和游戏开发中某些角色的潜力。小组探讨了像 SDXL 这样的 AI 工具如何改变就业格局，可能会提高产出高质量内容的准入门槛。

- **针对性 Lora 训练技术**：为了训练用于生成特定服装（如束身衣）图像的 Lora，建议使用该物品在排除无关细节后的多角度拍摄照片。其目的是帮助 AI 专注于核心元素，从而避免在输出中引入不必要的特征。

- **成本、投资者与 AI 市场动态**：公会讨论了 Stability AI 的战略障碍——在吸引投资、丰富数据集和开发新模型之间取得平衡。对话围绕在计算成本上升和模型研究兴趣波动的背景下，企业数据集货币化策略的创新展开。

- **随机闲聊依然活跃**：在技术讨论之余，成员们进行了随意的闲聊，包括文化引用和问候。有人分享了一个滑稽歌曲的非话题链接，展示了社区在技术参与之外轻松的一面。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**API 兴奋感与支付难题**：一位 **Perplexity API** 用户遇到了令人困惑的支付问题，其交易状态卡在“Pending”且未更新账户余额。与此同时，讨论围绕 API 的潜力以及在 Pro Subscription 与 pay-as-you-go API 之间的选择展开，观点倾向于在初始业务构思阶段选择订阅模式，因为其成本具有可预测性。

**模型混搭狂热**：用户深入探讨了**模型偏好**，倾向于在更高的消息数量和足够的 context window 之间寻找平衡。他们还应对了复杂编程语言（如 Rust）带来的模型限制挑战，并采用了自定义的“metaprompt”策略来实现结构化输出。

**内容分享注意事项**：提醒在 Discord 上发布内容时，确保将 **threads** 设置为可分享，以便促进更广泛的社区参与。

**对 Sonar 模型来源链接的渴望**：有关于 **sonar-medium-online** 模型返回带有数据的来源链接能力的咨询，但该功能实现的明确时间表仍未确定。

**LLM Leaderboard 的奇闻与疑问**：**LLM Leaderboard** 引发了关于模型排名的分析性讨论，其中夹杂着对模型名称误用的幽默调侃，指出了 system prompts 的清晰度对于提升 AI 性能的重要性。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DALL·E 迎来创意新功能**：**DALL·E** 现在在 Web、iOS 和 Android 平台上推出了*编辑套件（Editing Suite）*，用于图像编辑和风格灵感，增强了整个 **ChatGPT** 平台的创意潜力。与此同时，**Fine-tuning API** 引入了新的仪表板、指标和集成，方便开发者构建自定义模型，详情见[近期博客文章](https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program)。

- **AI 的存在主义沉思**：工程师们围绕 AI 展开了激烈的对话，探讨了 AI 是否在“思考”的问题，广泛共识是否认 AI 具有意识，认为其本质是复杂的数据模式执行。实时讨论中的分歧还涉及纠正公众中普遍存在的 AI 与 AGI 的误解，并以训练 LLM 执行目标导向序列的提议告终。

- **定制化还是复杂性**：在 **GPT-4 讨论**中，工程师们争论了 *Custom GPTs* 的益处、**DALL·E** 图像特定新功能的实用性，并提出了关于**数据保留政策**的问题——确保即使是被删除的聊天记录也会保留一个月。

- **提示词完美化的前奏**：技术人员钻研了将 Markdown 翻译成各种语言时遇到的问题，并建议在 *AI role play* 期间使用额外的上下文来优化 AI 的理解。还讨论了推进文本生成和在使用 LLM 时确保文档完整性的策略，建议使用“continue”等方法来扩展回复。

- **提示词精准度的耐心**：当成员们努力解决 **Markdown 翻译问题**并寻求构建有效提示词的建议时，他们被引导至重新设计的 **[#1019652163640762428](https://discord.com/channels/channel_id/1019652163640762428)** 获取资源。关于提示词效力的见解（特别是在角色扮演场景中）也浮出水面，强调了提供清晰上下文以塑造 AI 回复的重要性。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**CmdR 即将加入 Unsloth 行列**：向 Unsloth AI 添加 **CmdR** 支持的工作正在进行中，社区正热切期待其在当前任务完成后集成。这一预期与计划于 4 月 22 日发布的开源 **CPU optimizer** 相关联，旨在为 GPU 资源有限的用户提升 AI 模型的易用性。

**Continue 自动补全的界面创新**：[Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) 扩展推出了一项实验性的预发布 **tab autocomplete** 功能，旨在通过在开发环境中直接咨询语言模型，简化 VS Code 和 JetBrains 中的编码流程。

**错误消除与优化对话**：AI 工程师分享了针对命名相关 tokenizer 错误的解决方案，并讨论了 **model.save_pretrained_merged** 和 **model.push_to_hub_merged** 函数，以便在 Huggingface 上无缝保存和共享模型。尽管在 GemmaForCausalLM 中遇到了 `AttributeError`，但已引导用户通过更新 Unsloth 来解决。

**保存与服务器端设置中的障碍**：用户在处理 GGUF 转换和 Docker 设置时遇到了挑战，解决了诸如 `python3.10-dev` 依赖项以及在不同平台上进行 finetuning 期间内存错误的变通策略。

**即将深入探索 Unsloth Studio 的下一迭代**：由于当前的 Bug 修复，Unsloth Studio 的发布更新定于下月中旬，以确保与 **Google Colab** 的持续兼容性，并为利用 Studio 能力的开发者提供改进。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Stable Audio 达到新高度**：Stability AI 推出了 **Stable Audio 2.0**，能够利用单个 prompt 创建高质量的长篇音乐曲目。访问 [stableaudio.com](http://stableaudio.com) 测试模型，并在其 [blog post](https://bit.ly/3xlwmP5) 中了解更多详情。

**AssemblyAI 表现优于 Whisper**：**AssemblyAI** 发布了 **Universal-1**，这是一款语音识别模型，其准确率比 Whisper-3 高出 13.5%，且 hallucinations 减少了高达 30%。该模型仅需 38 秒即可处理一小时的音频，可在 [AssemblyAI's playground](https://www.assemblyai.com/playground) 进行试用。

**使用 ChatGPT Plus 增强您的图像**：ChatGPT Plus 用户现在可以修改 DALL-E 生成的图像和 prompt，该功能已在 Web 和 iOS 平台上可用。其 [help article](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e) 提供了完整的操作指南。

**作为可扩展微服务的 AI Agent**：讨论集中在利用事件驱动架构构建可扩展的 AI Agent，并引用了 Actor Model 作为灵感，同时展示了一个用于协作反馈的 Golang 框架。

**Opera One 直接下载 AI**：Opera 集成了让用户在本地运行 LLM 的能力，首先从开发者流（developer stream）中的 Opera One 开始，利用 Ollama 框架，详见 [TechCrunch](https://techcrunch.com/2024/04/03/opera-will-now-allow-users-download-and-use-llms-locally) 的报道。

**DSPy 成为焦点**：成员们评估了 **DSPy** 在优化基础模型 prompt 方面的表现，重点关注模型迁移和优化，同时警惕 API 速率限制。对 **Devin** 的详细研究揭示了众多 AI 项目机会，涵盖了从语音集成的 iOS 应用到文档重构计划等多种应用，引起了浓厚兴趣。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **LoRA 提升了 Mistral？**：工程师们讨论了在 **Mistral 7B** 上采用 **Low-Rank Adaptation (LoRA)** 以增强特定任务性能，并计划创新超出标准方法的句子切分和标注技术。
- **网络爬虫的困境与胜利**：**可扩展网络爬虫**的实际问题是热门话题，讨论涉及反爬措施和 JavaScript 渲染等障碍。然而，大家在 **Common Crawl** 的效用以及囤积高质量数据集的神秘存档组织方面达成了共识。
- **学习资源扩展**：分享的资源包括 **lollms 结合 Ollama Server** 的指南、来自 **Intellifusion** 的廉价 AI 芯片，以及 Hugging Face 的数据集加载工具 **Chug**。同时，CohereForAI 新推出的**多语言 104B LLM** 引起了关注，OpenAI 探索性的 **GPT-4 微调定价**也成为了讨论焦点。
- **LLM 创新处于前沿**：工程师们交流了关于**语言模型剪枝**的见解，特别是剪枝 25% 的 **Jamba 模型**，以及 Google 提倡 Transformer 学习动态分配计算资源的论文，这引发了对投机采样（speculative decoding）与 Google 方法的深入分析。
- **多样化的微调对话**：成员们介绍了针对推理优化的 **Eurus-7b-kto**，辩论了 **BitNet-1.58** 中用于三元编码的“*按比例除法*”，商讨了 Hermes-Function-Calling 的实现问题，考虑了 **QLoRA** 的 VRAM 效率，并注意到 **Genstruct 7B** 在指令生成方面的实力。
- **Project Obsidian 中的故障排除**：正在为 "llava" 项目中的 **ChatML** 进行快速修复，并打算解决 **Hermes-Vision-Alpha** 的问题，但具体细节尚不明确。
- **Finetuning Subnet Miner 意外**：**finetuning-subnet** 仓库中的一个矿工脚本错误指向了可能存在的**依赖缺失**问题。
- **RAG 数据集讨论**：关于 **Glaive** 的 RAG 样本数据集以及 grounded mode 和正确引用标记（包括 XML 指令格式）等方法的讨论，强调了未来的应用。还重点介绍了 RAG 响应中的过滤建议以及 Cohere 的 RAG 文档。
- **WorldSim 中的复制难题与指令探索**：WorldSim 令人困惑的复制粘贴机制、对移动端性能的担忧，以及指向详尽 [WorldSim Command Index](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4) 的链接，带来了生产力技巧以及在越狱 Claude 模型和 ASCII 艺术谜团中的文化片段。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**Mojo 在行动**：工程师们分享了 [Mojo](https://github.com/) 现在可以在 Snapdragon 685 CPU 等 Android 设备上运行，并讨论了将 [Mojo 与 ROS 2](https://github.com/ros2-rust/ros2_rust) 集成，强调了 Mojo 相比 Python 的内存安全性，特别是在 Python 的 GIL 限制 [Nvidia Jetson 硬件性能](https://developer.nvidia.com/embedded/jetson-modules)的机器人领域。

**性能突破与最佳实践**：显著的库性能提升被提及，执行时间缩短至分钟级，超越了之前的 Golang 基准测试。建议采用预设字典容量等优化方法，并鼓励字符串专用排序算法的设计者与 Mojo 的最新版本保持一致，详见 [mzaks/mojo-sort](https://github.com/mzaks/mojo-sort/blob/main/multi_key_quicksort/sort.mojo)。

**从解析器到 FASTQ**：引入了功能完备的新型 FASTQ 解析器 `BlazeSeq🔥`，提供了一个符合 BioJava 和 Biopython 基准测试的 CLI 兼容解析器。他们实现的缓冲行迭代器承诺增强文件处理能力，预示着将转向稳健的文件交互未来标准，该项目已在 [GitHub](https://github.com/MoSafi2/BlazeSeq) 上展示。

**Mojo 合并狂热**：关于模型合并和 Mojo 中条件一致性（conditional conformance）的创新想法使用了 **@conditional** 注解来实现可选的 trait，而 Mojo 主题毛绒玩具等周边创意也激发了社区的热情。讨论了内存管理优化，检查了 Mojo 标准库 nightly 版本中 `Optional` 返回值的潜在变化。

**Modular 更新大礼包**：[Max⚡ 和 Mojo🔥 24.2 版本发布](https://modul.ar/discord) 带来了开源的标准库和支持社区贡献的 nightly 构建版本。解决了 24.3 版本中的 Docker 构建问题，同时持续的开发讨论建议将条件一致性和错误处理策略纳入未来的路线图考虑。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**ROCm 带来的强力提升**：AMD 硬件在使用 **ROCm preview** 进行工程优化后，速度从 13 tokens/second 大幅提升至 65 tokens/second，凸显了合适的软件接口对 AMD GPU 的巨大潜力。

**Mixtral 并非 Mistral 的失误**：**Mixtral** 作为一种 **MOE** 模型，其独特的身份是将 8 个 7B 模型的实力结合成一个 56B 的强大模型，这反映了与标准 **Mistral** 7B 不同的战略方法。同时，在拥有 24GB **VRAM** 的 **NVIDIA 3090** GPU 上运行 Mixtral 8x7b 可能会遇到速度瓶颈，但这仍是一个可行的尝试。

**LM Studio 0.2.19 引入 Embeddings**：实验室刚推出的 **LM Studio version 0.2.19 Preview 1** 现在支持**本地 embedding models**，为 AI 从业者开辟了新的可能性。尽管目前的预览版缺乏 **ROCm support**，但 Windows、Linux 和 Mac 用户可以从提供的链接获取各自的构建版本。

**工程师解决模型异常行为**：关于 AI 模型给出奇怪且与任务无关的响应的讨论揭示了模型训练中潜在的失误，这标志着一个需要调试能力的编程困境。

**CrewAI 遭遇 JSONDecodeError**：在使用 **CrewAI** 时遇到 **JSONDecodeError** 表明 JSON 格式可能存在错误，这是 AI 工程师必须正确处理的难题，以避免危及数据解析过程。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**Transformers 席卷斯坦福**：斯坦福 CS25 关于 **Transformers** 的研讨会向公众开放实时旁听和录播，由行业专家主导关于 **LLM** 架构和应用的讨论。感兴趣的人士可以通过 [Zoom](https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09) 参与，访问[课程网站](https://web.stanford.edu/class/cs25/)，或在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) 上观看录像。

**对效率声明持怀疑态度**：社区对 [期刊发表](https://arxiv.org/abs/2311.12224) 中提到的 Free-pipeline Fast Inner Product (FFIP) 算法的性能声明表示怀疑，该算法承诺通过在 AI 硬件架构中减少一半的乘法运算来提高效率。

**CUDA 难题与代码冲突**：一名成员在 **H100** GPU 上使用 **LM eval harness** 排除 **RuntimeError with CUDA** 故障时，发现 `apex` 是问题所在，建议升级到 **CUDA 11.8** 并进行其他调整以保证稳定性。

**下一代 AI 训练技术备受推崇**：一篇 [arXiv 论文](https://arxiv.org/abs/2404.02258) 介绍了 **Transformers** 中的动态 **FLOP** 分配，可能通过偏离均匀分布来优化性能。此外，**AWS** 和 **Azure** 等云服务支持高级训练方案，其中明确提到了 **AWS** 的 **Gemini**。

**弹性与容错指南**：分享了使用 **PyTorch** 建立容错和弹性任务启动的详细信息，文档可在 [PyTorch elastic training 快速入门指南](https://pytorch.org/docs/stable/elastic/quickstart.html) 中找到。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **代码中的 AI 伦理**：一个名为 [ConstitutionalAiTuning](https://github.com/steffen74/ConstitutionalAiTuning) 的工具允许对语言模型进行微调以反映伦理原则，利用 JSON 文件输入原则，旨在让伦理 AI 更加普及。
- **JAX 中的类型博弈**：JAX 的类型提升（type promotion）语义根据操作顺序显示出不同的结果，正如 numpy 和 Jupyter 数组类型所演示的那样——将 `np.int16(1)` 和 `jnp.int16(2)` 与 `3` 相加，会根据操作序列产生 `int16` 或 `int32`。
- **模型训练的困惑**：一场讨论研究了模型的最佳文本输入配置，辩论了在 SD3 模型领域中序列拼接（sequence concatenation）、T5 token 扩展和微调技术的优劣。
- **法律节奏与 AI**：使用受版权保护的材料来训练 AI（例如 Suno 音乐 AI 平台）引发了对随之而来的法律风险和内容所有者潜在诉讼的担忧。
- **AI 创新者的财务动荡**：Stability AI 面临财务逆风，正在应对巨额的云服务支出，据 [Forbes 文章](https://www.forbes.com/sites/kenrickcai/2024/03/29/how-stability-ais-founder-tanked-his-billion-dollar-startup/?sh=2e53d2e3e630) 详细报道，这些支出可能会超过其营收能力。

在**研究**领域：

- **对于 LDM 而言，尺寸并不总是最重要的**：一项发表在 [arXiv 论文](https://arxiv.org/abs/2404.01367) 中的研究揭示，当推理预算保持不变时，大型潜扩散模型（LDMs）并不总是优于小型模型。
- **新型优化器即将问世**：一条 [Twitter 预热](https://twitter.com/aaron_defazio/status/1775521495298588956) 暗示 AI 社区应该密切关注一种新型优化器。
- **VAR 模型革新图像生成**：根据一篇 [arXiv 论文](https://arxiv.org/abs/2404.02905)，新提出的视觉自回归（VAR）模型在图像生成方面表现出优于 Diffusion Transformers 的效能，在质量和速度上都有所提升。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**完美补丁**：OpenAccess AI Collective 的 **axolotl** 仓库中一个值得注意的 **GitHub bug** 被迅速消除，提交历史可通过 [GitHub Commit 5760099](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a) 查看。同时，一个 **README 目录不匹配**的问题被标记并促成了清理工作。

**数据集与模型对话**：关于训练 Mistral 7B 模型**最佳数据集**的咨询引向了对 **OpenOrca 数据集**的推荐，而关于**微调实践**的辩论则倾向于优先处理“补全（completion）”而非“指令（instructions）”的策略。讨论强调了在拥有高质量指令样本时，**简单微调（SFT）**相较于持续预训练（CPT）的效力。

**机器人服务中断**：**Axolotl 帮助机器人**遇到障碍并下线，引发了成员们的一阵调侃，但事件背后的具体细节尚未披露。该机器人此前一直在提供关于 **Qwen2** 与 **Qlora** 集成的指导，并解决与 Docker 环境中**数据集流式传输**和**多节点微调**相关的挑战。

**AI 对话**：该集体的 **general 频道**充满了技术讨论——从 [Chaiverse](https://console.chaiverse.com/) 等**快速模型反馈服务**，到在 [transformer-heads GitHub 仓库](https://github.com/center-for-humans-and-machines/transformer-heads) 中发现的为 Transformer 模型添加 Head 的新资源。**CohereForAI** 发布了一个拥有 1040 亿参数的巨型 **C4AI Command R+** 模型，其专业能力已在 [Hugging Face 上揭晓](https://huggingface.co/CohereForAI/c4ai-command-r-plus)，引发了关于运行超大模型的财务影响的对话。

**基础设施创新**：SaladCloud 最近推出的针对 AI/ML 工作负载的全托管容器服务被认为是一个显著的切入点，为开发者在应对高昂的云成本和 GPU 短缺方面提供了优势，并为大规模推理提供了实惠的价格。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**AI 拼写检查变得触手可及**：一位成员分享了使用 **LlamaIndex** `Ollama` 软件包纠正拼写错误的 Node.js 代码，展示了名为 ‘mistral’ 的 AI 模型如何修复用户错误（例如将 "bkie" 修正为 "bike"），该模型可以在本地通过 `localhost:11434` 运行，无需第三方服务。

**Llama 的烹饪代码食谱**：为 AI 爱好者推出了一系列全新的烹饪主题指南，演示了如何使用 **MistralAI** 构建 RAG、agentic RAG 和基于 Agent 的系统，包括 routing 和 query decomposition。点击[此处](https://t.co/7KCqujf9sd)获取你的 AI 食谱。

**LlamaIndex 中的探索与困惑**：社区讨论提出了对缺乏 **knowledgegraphs** 流水线支持、不清晰的 **graphindex** 和 `graphdb` 集成等问题的担忧，多位成员在查询 **OpenSearch** 和在 **llama_index** 中实现 ReAct agents 时遇到困难。

**AI 讨论演进至文本之外**：关于利用 Reading and Asking Generative (RAG) 技术增强图像处理潜力的讨论引起了关注，讨论的应用范围从 CAPTCHA 解决方案到确保漫画等视觉叙事的连续性。

**扩展 AI 部署变得便捷**：Koyeb 平台因其能够轻松扩展 LLM 应用而受到关注，它可以直接连接你的 GitHub 仓库，在全球范围内部署 serverless 应用，无需管理基础设施。点击[此处](https://t.co/weFs0waN4o)查看该服务。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**仓库可见性选择**：HuggingFace 引入了默认仓库可见性设置，为企业提供 **public**、**private** 或 **private-by-default** 选项。Julien Chaumond 在[这条推文](https://twitter.com/julien_c/status/1772688542289822073)中描述了该功能。

**自定义 Quarto 发布**：HuggingFace 现在支持使用 **Quarto** 发布，详见 Gordon Shotwell 的[推文](https://twitter.com/gshotwell/status/1772661727856914720)，更多信息可在 [LinkedIn](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29) 上找到。

**摘要难题与策略**：各频道的用户讨论了使用 GPT-2 和 Hugging Face pipeline 进行摘要时的挑战，包括无效的 length penalties 以及寻找能最大化效率和结果质量的 prompt crafting 方法，即使在 CPU-only 环境下也是如此。

**AI 圈的创新与互动**：社区对包括 **Octopus 2**（一个支持 function calls 的模型）以及 Salt 推出的新 **multi-subject image node pack** 在图像处理方面的进展感到兴奋。社区还强调了学术讨论和资源，例如 **RAG for interviews** 的潜力和生产环境 prompt 中的 latency-reasoning 权衡，分享于 [Siddish 的推文](https://x.com/siddish_/status/1772345589511901368?s=20)。

**扩散模型对话探讨深度**：AI 工程师探索了 diffusion models 的创意实现，讨论了用于各种数据条件的带有 cross-attention 的 **DiT**，并考虑将 **Stable Diffusion** 修改用于立体图到深度图转换等任务，参考了 [DiT 论文](https://arxiv.org/html/2312.04557v1) 以及 [Dino v2 GitHub](https://github.com/facebookresearch/dinov2) 和 [SD-Forge-LayerDiffuse GitHub](https://github.com/layerdiffusion/sd-forge-layerdiffuse) 等资源。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**寻求赞美还是功能改进？**：Discord 从搞怪的鱼形 logo 切换到更精致的设计引发了成员之间的辩论，并导致了关于将横幅与新审美相匹配的讨论。George Hotz 对 logo 的更改似乎让一些人对旧版感到怀念。

**深度分片优化**：George Hotz 和社区成员探讨了 **优化技术和跨 GPU 通信**，面临着启动延迟和数据传输方面的挑战。他们研究了 cudagraphs 的使用、peer-to-peer 限制以及 NV 驱动程序的作用。

**Tinygrad 性能里程碑**：在分享的性能基准测试中，透露出 Tinygrad 在 **单块 4090 GPU 上达到了每秒 53.4 个 tokens**，与 gpt-fast 相比达到了 83% 的效率。George Hotz 表示目标是进一步提升 Tinygrad 的性能。

**Intel 硬件指日可待**：关于 **Intel GPU 和 NPU 内核驱动程序** 的讨论审视了各种可用驱动程序，如 'gpu/drm/i915' 和 'gpu/drm/xe'，并期待 NPU 与 CPU 搭配时可能带来的性能和能效提升。

**有益的神经网络教育热潮**：社区发现 **Tinygrad 教程** 是神经网络新手的宝贵起点，并推荐了 JAX Autodidax 教程，其中包含一个 [动手实践的 Colab notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/autodidax.ipynb)。将 ColabFold 或 OmegaFold 适配到 Tinygrad 的兴趣激增，同时也在学习 PyTorch 权重迁移方法。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **OpenRouter 采用 JSON 对象支持**：已确认 **OpenAI** 和 **Fireworks** 等模型支持 'json_object' 响应格式，可以通过 [OpenRouter 模型页面](https://openrouter.ai/models) 上的提供商参数进行验证。

- **使用 Claude 3 Haiku 寻找合适的语调**：虽然 **Claude 3 Haiku** 模型在角色扮演中表现参差不齐，但建议提供多个示例可能会获得更好的结果。然而，使用 **jailbreak (jb) 调整** 对于显著改善输出是明智的。

- **Claude 越狱的小众服务器**：寻找包含 NSFW 提示词的 **Claude 模型** 越狱方法的模型用户讨论了相关资源，指出 SillyTavern 和 Chub 的 Discord 服务器是首选之地，并提供了如何使用 pancatstack jb 等工具进行导航的指导。

- **仪表板更新展示 OpenRouter 额度**：**OpenRouter 仪表板** 的最新更新包括一个新的指定额度显示位置，可通过 `/credits` 端点访问。然而，特定模型（如 **DBRX** 和 **Midnight Rose**）的功能问题引发了对其支持兼容性的担忧。

- **审核纠纷影响 OpenRouter API 的拒绝率**：报告强调了自我审核版本的 **Claude 模型** 拒绝率很高，这可能与过度保护的“安全”提示词有关。还提到了整合更好的提供商，以帮助提高 **Midnight Rose** 等模型服务的稳定性。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **安装成功庆典与跨平台明确性**：一位工程师为在 Windows 机器上运行该 **软件** 感到欣慰，并确认该软件在 PC 和 Mac 平台上均可运行。详细的安装说明和指南可以在项目的文档中找到。

- **持续存在的 Termux 困境**：讨论指出在安装过程中 `chroma-hnswlib` 存在反复出现的问题，尽管有报告称其已被移除。成员们被建议将详细的技术支持咨询迁移到指定的支持频道。

- **讨论 Hermes-2-Pro 的 Prompt 实践**：活跃的对话强调了根据 **Hermes-2-Pro** 模型卡片的建议调整系统提示词（system prompts）的必要性。这对于优化模型性能和解决某些用户认为负担沉重的冗长输出至关重要。

- **特定平台的怪癖**：多位成员遇到并分享了在不同操作系统上运行 01 软件的挑战解决方案——从 Ollama 中的快捷命令、Linux 中的包依赖，到 Windows 11 上的 `poetry` 问题。

- **Cardputer 开发进行中**：技术讨论集中在将 **M5 Cardputer** 实现并推进到 open-interpreter 项目中。链接了 GitHub 仓库和各种工具，如用于安全隧道的 ngrok 和用于神经 TTS 系统的 rhasspy/piper 以供参考。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Command R+ 凭借 128k Token 上下文引起轰动**：一款名为 **Command R+** 的新型可扩展 LLM 因其高达 128k 的 Token 上下文窗口以及通过精细化 RAG 承诺减少幻觉而备受关注。尽管由于缺乏对比数据，人们对其与其他模型的性能对比仍持好奇态度，但爱好者可以通过 [实时演示](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 测试其能力。

- **类 ChatGPT 商业模型面临审查**：对于 ChatGPT 及其类似模型能否满足企业需求，人们持怀疑态度，讨论指向可能需要定制化开发的解决方案才能真正满足业务需求。

- **学术界为高性价比的 JetMoE-8B 欢呼**：**JetMoE-8B** 的发布在学术界受到称赞，因为它成本低廉（低于 10 万美元），且仅使用 2.2B 激活参数就实现了令人印象深刻的性能。更多详情可在其 [项目页面](https://research.myshell.ai/jetmoe) 查看。

- **Snorkel 与模型效能辩论升温**：Nathan Lambert 发布了一条具有暗示性的 [推文](https://twitter.com/natolambert/status/1775899591814300024)，预告了对当前 AI 模型（如使用 RLHF 的模型）有效性的分析，从而引发了围绕备受争议的 **Snorkel** 框架的讨论。

- **斯坦福 CS25 吸引 Transformer 爱好者**：AI 工程师对斯坦福大学的 CS25 课程表现出浓厚兴趣，该课程重点讨论了 Transformer 研究专家的见解，课程表可在 [此处](https://web.stanford.edu/class/cs25/#schedule) 获取，并有机会通过该课程的 YouTube 频道获取深度见解。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **矩阵大小至关重要**：一名成员通过为大矩阵优化 **matmul kernel** 取得了进展，解决了处理 1024x1024 以上尺寸时遇到的 CPU 缓存挑战。
- **攻克编译器难题**：编译器的增强令成员们欢欣鼓舞，这预示着代码性能将有显著提升。
- **ROCm 的硬性要求**：为了在 Windows 上成功部署 **llamafile-0.7**，成员们确认需要 **ROCm 5.7+** 版本。
- **动态 SYCL 讨论**：关于在 **llamafile** 中处理 SYCL 代码的辩论产生了一个由社区驱动的解决方案，涉及条件编译，但指出其与 MacOS 不兼容。
- **Windows 上的性能困惑**：在 Windows 上构建 **llamafile** 的尝试遇到了涉及 Cosmopolitan 编译器的复杂问题，同时还讨论了需要一个 `llamafile-bench` 程序来衡量每秒 Token 数（tokens per second）以及 RAM 对性能的潜在影响。感兴趣的各方被引导至 [The Register](https://www.theregister.com/2024/04/03/llamafile_performance_gains/) 上的一篇文章，该文章强调了性能提升，以及 [GitHub 上关于 Cosmopolitan 的讨论](https://github.com/jart/cosmopolitan/issues/1010)。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**加密货币聊天机器人热潮招募开发者**：有人正在寻找具有 LLM 训练经验的开发者，以创建一个模拟人类对话的聊天机器人，并利用实时加密货币市场数据。目标是实现能够反映最新市场变化的细微讨论。

**无需 MathpixPDFLoader 的数学符号提取**：用户正在寻求从 PDF 中提取数学符号的 MathpixPDFLoader 替代方案，以寻找有效处理这一特定任务的新方法。

**LangChain LCEL 逻辑课程**：一次讨论澄清了 LangChain 表达式语言 (LCEL) 中 '|' 运算符的使用，该运算符将 Prompt 和 LLM 输出等组件链接成复杂的序列。更多细节可在 [LCEL 入门指南](https://python.langchain.com/docs/expression_language/get_started) 中进一步探索。

**语音应用展现 AI 能力**：新推出的语音应用（如 [CallStar](https://callstar.ai)）引发了关于其交互性和设置的讨论，这些应用由 RetellAI 等技术驱动，并获得来自 Product Hunt 和 Reddit 平台的社区支持。

**LangChain 快速入门演练困扰**：在分享 [LangChain 快速入门指南](https://python.langchain.com/docs/get_started/quickstart) 时，一名用户提供了将 LangChain 与 OpenAI 集成的示例代码，但遇到了表示资源缺失的 `NotFoundError`。请求社区的技术专家协助排查此问题。

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **积微成著，效率尽显**：引用了 [BitMat](https://github.com/astramind-ai/BitMat) GitHub 仓库，该项目推广了 1-bit Large Language Models (LLMs) 的**高效实现**，与《The Era of 1-bit LLMs》中提出的方法保持一致。
- **Triton 与 Torch 的新地平线**：提议建立一个新频道用于贡献 **Triton visualizer**，以促进协作。Torch 团队正在调整 autotune 设置，转向 **max-autotuning**，并解决包括 Tensor Core 利用率和计时方法在内的 benchmarking 痛点——他们的工作记录在 [keras-benchmarks](https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17) 中。
- **CUDA 内容与课程**：针对热衷于学习 CUDA 编程的工程师，推荐了 [CUDA MODE YouTube channel](https://www.youtube.com/@CUDAMODE)，该频道拥有丰富的讲座和支持性社区，旨在降低 CUDA 学习曲线。
- **模型集成的飞跃**：新成员 mobicham 和 zhxchen17 发起了关于将 **HQQ** 与 **gpt-fast** 集成的讨论，重点关注 **Llama2-7B (base)**，并深入研究了使用 [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ) 等模型进行 4/3 bit 量化。
- **Triton 的可视化增强**：在关于 **Triton visualizations** 的讨论中，出现了增加方向箭头、将操作细节整合到视觉效果中，以及可能将项目移植到 JavaScript 以增强交互性的建议，尽管也有人对这些功能的实际效用表示担忧。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

**AI 对话的新方法**：在反思对话式 AI 术语时，一位成员建议使用 “turns” 而不是 “responses” 来描述对话中的初始消息，这一决定源于对 `logs.db` 数据库的探索，并意外地与术语 [database 'turns'](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554) 形成了双关。

**AI 产品评估获得认可**：成员们一致认为 [Hamel Husain 关于 AI 评估的文章](https://hamel.dev/blog/posts/evals/) 非常重要，该文章概述了为 AI 创建特定领域评估系统的策略，被认为对初创企业具有开创性意义。

**SQL 查询助手插件寻求透明度与控制**：有人提议让 Datasette SQL 查询助手插件的评估过程**可见且可编辑**，旨在增强用户交互以及对评估过程的控制。

**研读 Prompt 管理的未来**：关于 AI Prompt 管理的最佳实践正在引发辩论，潜在模式包括**本地化、中间件和微服务**，这暗示了将 AI 集成到大型系统中的不同方法。

**高分辨率 API 细节示例**：Cohere LLM 搜索 API 的详细 JSON 响应受到关注，提供了一个可以使 AI 开发者受益的细粒度示例，正如在分享的 [GitHub comment](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135) 中所展示的那样。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **情感智能基准测试 (Benchmarking Emotional Smarts)**：新推出的 [Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html) 和 [Judgemark](https://eqbench.com/judgemark.html) 基准测试旨在评估语言模型的情感智能，其中 Judgemark 通过相关性指标提出了严苛的测试。利用评分的标准差来区分模型如何使用 0-10 评分量表（相比 0-5 评分系统）来指示更细微的判断差异。
  
- **创意写作的评测日 (Judgment Day for Creative Writing)**：Creative Writing 基准测试的有效性归功于其 **36 个具体的评审标准**，强调了细化参数对模型评估的重要性。提供的详尽文档回答了关于这些基准测试标准的问题，展示了透明度并允许更好的模型评估。

- **衡量情感与质量 (Sizing Up Sentiment and Quality)**：关于最佳量表的讨论表明，情感分析最适合 -1 到 1 的范围，而质量评估则更倾向于 0-5 或 0-10 的更宽量表，这有助于模型传达更细微的观点。这些见解强调了针对特定判断领域定制评估指标的必要性。

- **COMET 在测试中表现出色 (COMET Blazes Through Testing)**：**COMET** 评估分数显示 Facebook WMT21 模型表现优异，其无参考（reference-free）评分采用了 **wmt22-cometkiwi-da** 方法，相关的实用脚本可在 [llm_translation GitHub repository](https://github.com/CrispStrobe/llm_translation) 获取。尽管如此，由于可能存在不准确性，建议保持谨慎，强调了在评估模型输出时需要持续警惕。

- **攀登无参考评估的高峰 (Scaling the Peaks of Reference-Free Evaluation)**：对模型准确性的呼吁强调了 COMET 评分结果并非绝对，并邀请标记显著的差异——这种做法承认了模型优化和验证的迭代本质。记录到的最高 COMET 分数为 0.848375，展示了当前语言模型在翻译任务中的先进能力。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI 爱好者关注医疗保健 (AI Enthusiasts Eye Healthcare)**：社区对医疗保健领域 AI 的参与度正在上升，标志着 AI 技术跨学科应用的增加。

- **利用 Mixture-of-Depths (MoD) 方法演进 LLM**：**Mixture-of-Depths (MoD)** 技术的引入被强调为一种允许语言模型动态分配计算资源的方式，有可能提高效率。该方法及其功能在 [arXiv](https://arxiv.org/abs/2404.02258) 上的一篇论文中有详细介绍。

- **彻底改变 AI 处理数学的方法 (Revolutionizing AI's Approach to Math)**：在讨论改进 AI 解决数学问题的策略时，有人建议训练 AI 将应用题转换为可解方程比直接计算更有效。这种方法利用了 **Python** 和 **Wolfram Alpha** 等成熟工具的强大功能来进行实际计算。

- **论文库再添新成员 (Another Paper Added to the Trove)**：社区正在分享更多资源，[一篇新论文](https://arxiv.org/abs/2404.02684) 已添加到社区知识库中，但尚未提供进一步的背景信息。



---

# PART 2: 渠道详细摘要与链接



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1225006104375726170)** (910 messages🔥🔥🔥): 

- **Stable Diffusion 模型与放大 (Upscaling)**：用户讨论了创建逼真高分辨率图像的最佳实践，建议使用较低的步数（steps）、latent upscaling 以及使用 hi-res fix 以避免图像失真。建议的设置包括使用 dpmpp ancestral Karras 或 exponential 采样器的 35 步，并配合 control nets。2k 等更高分辨率具有挑战性，通常会导致生成时间延长和潜在的图像畸变（[相关讨论](https://arxiv.org/abs/2404.02905)）。

- **AI 的未来与内容创作 (The Future of AI and Content Creation)**：关于 AI 对各种创意产业影响的辩论非常激烈，推测了 AI 取代好莱坞和电子游戏行业传统角色的潜力。参与者讨论了像 SDXL 这样的 AI 模型是否会让某些艺术家职位变得多余，以及不断发展的技术如何可能提高技能下限（skill floor），从而需要更少的努力来生成高质量内容。

- **针对特定物品的 Lora 训练 (Lora Training for Specific Items)**：一位用户询问了关于训练 Lora 以生成穿着特定物品（如紧身胸衣）的人物图像。给出的建议包括使用该物品不同角度的图像，理想情况下移除背景和面部，以防止 AI 在生成的图像中包含非预期的元素。

- **经济考量与 AI**：参与者讨论了 Stability AI 面临的挑战，例如说服投资者以及专注于数据集与开发新模型之间的权衡。对话涵盖了将数据集面向企业变现的潜力，以应对感知到的对研究模型兴趣下降的问题，以及高昂算力成本的影响。

- **杂项聊天**：互动包括涉及文化话题的轻松交流、普通的问候、对问候的回应，以及与讨论主旨无关的随机陈述。还有一位用户分享了一个无关的恶搞歌曲链接。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了视觉自回归建模（VAR），这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率预测”...</li><li><a href="https://huggingface.co/RunDiffusion/Juggernaut-XL-v9/blob/main/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors">Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors · RunDiffusion/Juggernaut-XL-v9 at main</a>：未找到描述</li><li><a href="https://sdxl.replicate.dev/">SDXL – A settings guide by Replicate</a>：或者我是如何学会制作奇怪的猫的</li><li><a href="https://remix.ai/">Remix</a>：创建、分享和重混 AI 图像与视频。</li><li><a href="https://leonardo.ai/">Home v2</a>：利用我们的 AI 图像生成器改变您的项目。以无与伦比的速度和风格生成高质量的 AI 图像，提升您的创意愿景</li><li><a href="https://www.reddit.com/r/3Frame">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=yvOXZ6SV2Rk">Stable Radio 24/7</a>：Stable Radio，一个 24/7 全天候直播流，专门播放由 Stable Audio 生成的曲目。在 stableaudio.com 上探索模型并开始免费创作</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Optimizations">Optimizations</a>：Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/continue-revolution/sd-webui-animatediff/blob/master/docs/features.md#controlnet-v2v">sd-webui-animatediff/docs/features.md at master · continue-revolution/sd-webui-animatediff</a>：适用于 AUTOMATIC1111 Stable Diffusion WebUI 的 AnimateDiff - continue-revolution/sd-webui-animatediff</li><li><a href="https://github.com/comfyanonymous/ComfyUI">GitHub - comfyanonymous/ComfyUI: The most powerful and modular stable diffusion GUI, api and backend with a graph/nodes interface.</a>：最强大且模块化的 Stable Diffusion GUI、API 和后端，具有图表/节点界面。 - comfyanonymous/ComfyUI</li><li><a href="https://forms.gle/9i4jM9BQu9bVVAAF6">Survey Form - 5day.io</a>：作为一名刚入职几年的年轻专业人士，对于证明自己和寻找每个人都在谈论的神秘工作与生活平衡，总有一种隐约的焦虑。有时...</li><li><a href="https://www.reddit.com/r/3FrameMovies/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://civitai.com/models/133005/juggernaut-xl?modelVersionId=348913">Juggernaut XL - V9 + RunDiffusionPhoto 2 | Stable Diffusion Checkpoint | Civitai</a>：有关业务咨询、商业许可、定制模型和咨询，请通过 juggernaut@rundiffusion.com 联系我。Juggernaut 已上线...</li><li><a href="https://github.com/ZHO-ZHO-ZHO/ComfyUI-SegMoE">GitHub - ZHO-ZHO-ZHO/ComfyUI-SegMoE: Unofficial implementation of SegMoE for ComfyUI</a>：SegMoE 的非官方 ComfyUI 实现。通过在 GitHub 上创建账户，为 ZHO-ZHO-ZHO/ComfyUI-SegMoE 的开发做出贡献。</li><li><a href="https://m.soundcloud.com/pelusitalachicafideo/never-gonna-give-you-up-rick-astley-minions-ver">Never Gonna Give You Up - Rick Astley [Minions Ver.]</a>：在桌面和移动设备上流式传输 Pelusita,la chica fideo 制作的 Never Gonna Give You Up - Rick Astley [小黄人版]。在 SoundCloud 上免费播放超过 3.2 亿首曲目。
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1225023684968710144)** (756 条消息🔥🔥🔥):

- **API 使用详解**：用户对 Perplexity API 的功能和相关成本感到好奇。会议澄清了 API 在自动化任务方面非常强大，对于希望将特定服务集成到其应用程序中的开发人员来说至关重要。成本效益和使用情况取决于项目的范围和处理的数据量。
- **Pro 订阅与 API 的优劣对比**：关于每月支付 20 美元订阅 Perplexity 还是使用按需付费（pay-as-you-go）的 API 更有利，存在一番争论。对于创意生成和初创业务，由于易用性和成本管理，建议倾向于选择订阅模式。
- **模型偏好讨论**：在涉及使用体验时，用户更倾向于拥有更多的消息条数和体面的 Context Window（上下文窗口），而不是较少的次数配上更大的上下文。Perplexity 的 AI 能力正被用于一系列任务，并具有绕过限制的灵活性。
- **通知与新 UI 元素更新**：有人提到新闻通知不够直观或沟通不力，建议公司更具战略性地使用 Discord 的公告频道。一些用户对 Android 应用缺乏更新表示担忧。
- **集成限制与模型能力**：讨论了在使用 Rust 等复杂语言与 AI 协作时的局限性，强调包括 Opus 在内的 AI 模型在创建可编译代码方面仍有困难。一些用户正在应用变通方法，例如开启新对话（new threads）来管理大型对话，以便更好地进行上下文管理。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platform">Perplexity 将在其 AI 搜索平台上尝试广告形式</a>：Perplexity 的首席商务官 Dmitry Shevelenko 告诉 Adweek，公司正在考虑在其平台中添加赞助商建议问题。如果用户继续搜索更多信息...</li><li><a href="https://www.theverge.com/2024/4/1/24118154/perplexity-will-try-a-form-of-ads-on-its-ai-search-platfo">Perplexity 将在其 AI 搜索平台上尝试广告形式</a>：Perplexity 的首席商务官 Dmitry Shevelenko 告诉 Adweek，公司正在考虑在其平台中添加赞助商建议问题。如果用户继续搜索更多信息...</li><li><a href="https://docs.perplexity.ai/docs/getting-started">pplx-api 入门指南</a>：未找到描述</li><li><a href="https://fontawesome.com/icons/brain-circuit?f=classic&s=thin">Brain Circuit 经典细体图标 | Font Awesome</a>：细体风格的 Brain Circuit 图标。使用最新的超轻设计为您的项目增色。现已在 Font Awesome 6 中推出。</li><li><a href="https://fontawesome.com/icons/image?f=classic&s=regular">Image 经典常规图标 | Font Awesome</a>：常规风格的 Image 图标。使用随和、易读的图标使您的设计更平滑。现已在 Font Awesome 6 中推出。</li><li><a href="https://www.tomsguide.com/ai/apple-reveals-realm-new-ai-model-could-make-siri-way-faster-and-smarter">苹果发布 ReALM —— 新 AI 模型可能让 Siri 变得更快更智能</a>：ReALM 可能是 Siri 2.0 的一部分</li><li><a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述</li><li><a href="https://tenor.com/view/ralph-wiggum-simpsons-hi-bye-gif-16529059407582436389">Ralph Wiggum Simpsons GIF - Ralph Wiggum Simpsons Hi - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bo">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/aravsrinivas/status/1775632536934486160?s=46">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：很有趣。</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/11s44ry/discord_bot_public_releaseintroducing/?rdt=64126">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=JV4JbYK-TIg">1111Hz 与宇宙连接 - 接收宇宙指引 - 吸引魔法与治愈能量 #2</a>：1111Hz 与宇宙连接 - 接收宇宙指引 - 吸引魔法与治愈能量 #2 这个频道致力于治愈您的心灵、灵魂、身体...</li><li><a href="https://gist.github.com/cjanietz/703a88924e50e1a30cb6ffc52bc52bd9">Perplexity 模型选择用户脚本</a>：Perplexity 模型选择用户脚本。GitHub Gist：即时分享代码、笔记和片段。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1225019849672228904)** (15 条消息🔥):

- **探索 Fritz Haber 的影响**：一位成员强调了 **Fritz Haber** 的贡献，例如通过 *Haber-Bosch 过程* 实现了粮食产量的增加。他复杂的遗产包括诺贝尔奖、参与化学战、个人悲剧以及反纳粹情绪。[阅读关于 Fritz Haber 的遗产](https://www.perplexity.ai/search/Who-is-Fritz-kSg0wtgUSombH0qTxfuzzg)。

- **LLM Leaderboard 的奥秘**：一位用户研究了 **LLM Leaderboard**，讨论了模型指标和排名，并发现了 *“95% CI”* 的含义，尽管遇到了一些有趣的模名错误。[探索 LLM Leaderboard 评论](https://www.perplexity.ai/search/LLM-Leaderboard-Review-4C4F5TQuQSSxnYBWVfEZgg)。

- **通过 AI 理解美**：多位成员分享了他们对美这一概念的好奇心，并使用 **Perplexity AI** 获取有关该主题的见解。[深入探讨美的本质](https://www.perplexity.ai/search/Why-is-beauty-IIA2.dXGSCOwM5aXlcbuVA)。

- **发起关于独裁统治的讨论**：一个聊天引导用户使用 **Perplexity AI** 查询独裁统治是如何自然产生的，引发了对威权政体起源的智力探索。[调查独裁统治的出现](https://www.perplexity.ai/search/Dictatorship-naturally-arises-qRK3sToeRYqDa3Y_oP6Ztw)。

- **可共享内容的提醒**：一位成员被提醒在从 Discord 频道发布链接时，要确保其 **thread** 设置为可共享。这可以确保其他人能够查看并参与分享的内容。[使 Discord threads 可共享](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1225014380295491667)** (42 messages🔥): 

- **关于 Perplexity API Sonar 模型来源链接的困惑**：一位用户询问 **sonar-medium-online** 模型何时能够随数据返回来源链接，但未收到关于该功能何时上线的明确时间表。
- **额度难题：Perplexity 中的待处理支付**：一位成员报告了购买 API 额度时的问题；交易显示为“Pending（待处理）”，且未反映在账户余额中。另一位成员要求其发送账户详情以便解决，这表明采用的是逐案排查的方法。
- **Realms, ReALM 与 Apple 的困扰**：用户发现当询问 Apple 的 ReALM 时，机器人会感到困惑，有人建议简化 system prompt 可能会获得更好的性能，因为复杂性似乎会导致混乱。
- **用于组织输出的自定义 GPT "metaprompt"**：一位用户分享了他们创建自定义 GPT 的实验，该 GPT 利用 "metaprompt" 旨在高效地构建响应，主要侧重于提供带有清晰引用的准确信息。
- **Search API 定价困惑**：一位成员质疑了 Search API 相对于语言模型的定价，讨论了 1000 次在线模型请求的成本效益，另一位成员澄清说，这并不等同于 1000 次单独搜索，而是每次请求可以包含多次搜索。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://....)"">未找到标题</a>：未找到描述</li><li><a href="https://tenor.com/view/were-on-the-same-boat-here-mickey-haller-lincoln-lawyer-we-have-a-common-problem-we-have-the-same-issue-gif-9336579479687485405">我们在同一条船上 Mickey Haller GIF - 我们在同一条船上 Mickey haller Lincoln lawyer - 发现并分享 GIF</a>：点击查看 GIF
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1225128391813103706)** (2 messages): 

<ul>
  <li>
    <strong>DALL·E 获得编辑套件</strong>：成员们获悉，现在可以在网页端、iOS 和 Android 的 ChatGPT 中编辑 DALL·E 图像，并在 DALL·E GPT 中创建图像时获得风格灵感。
  </li>
  <li>
    <strong>Fine-tuning API 升级</strong>：Fine-tuning API 引入了新的仪表板、指标和集成。开发者现在在与 OpenAI 构建自定义模型时拥有更多控制权和新选项，详情见最近的博客文章：<a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">Fine-tuning API 和自定义模型计划</a>。
  </li>
</ul>

**提到的链接**：<a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">介绍 Fine-tuning API 的改进并扩展我们的自定义模型计划</a>：我们正在添加新功能，以帮助开发者更好地控制微调，并宣布与 OpenAI 构建自定义模型的新方法。

  

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1225007451527712788)** (494 条消息🔥🔥🔥): 

- **理解 AI 与意识**：讨论围绕 AI 的认知过程与人类思维的本质区别展开，辩论 LLM 等 AI 是否具备“思考”能力，还是仅仅是执行复杂数据模式的高级算法。多位参与者认为 AI 缺乏意识，而是一种对类人行为的模拟。
  
- **定义感知力的复杂性**：感知力（Sentience）和意识是热门话题，探讨了神经活动研究揭示的动物主观体验。对话指出，辨别不同生命形式的感知力存在困难，且仅基于类人行为来定义意识具有挑战性。

- **AI 误区与预期**：部分讨论强调了公众对 AI 的误解，即许多人将所有形式的 AI 等同于科幻小说中经常描绘的 AGI（通用人工智能）概念。讨论强调需要明确区分各种形式的 AI 与当前技术的现实。

- **实时讨论动态**：关于 AI 的辩论经常导致参与者之间的摩擦，展示了关于 AI 能力、意识和伦理考量的广泛观点和意见。一些人推荐了 YouTube 视频等额外资源来强化自己的观点。

- **潜在的 AI 用途与开发思路**：一位用户建议使用面向目标的序列（如 `success <doing-business> success`）来训练语言模型，用于包括下棋或制定商业策略在内的各种应用，并推论了在推理过程中提供新信息时的交互可能性。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/China_brain">China brain - Wikipedia</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=R9OHn5ZF4Uo">How AIs, like ChatGPT, Learn</a>: 我们身边的所有算法（如 ChatGPT）是如何学习并完成工作的？脚注：https://www.youtube.com/watch?v=wvWpdrfoEv0 感谢我的支持者...</li><li><a href="https://www.asciiart.eu/food-and-drinks/bananas">ASCII Art Bananas - asciiart.eu</a>: 一个包含香蕉及其他相关食物和饮料的 ASCII 艺术画作的大型集合。</li><li><a href="https://www.lesswrong.com/posts/vJFdjigzmcXMhNTsx/simulators">Simulators — LessWrong</a>: 感谢 Chris Scammell, Adam Shimi, Lee Sharkey, Evan Hubinger, Nicholas Dupuis, Leo Gao, Johannes Treutlein, 和 Jonathan Low 对草案的反馈...
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1225060200810156115)** (46 条消息🔥): 

- **Custom GPT vs. 基础模型**：频道成员正在讨论使用 Custom GPT 相比基础 ChatGPT 模型的优势。虽然有些人更喜欢使用 Custom GPT 构建复杂提示词的便利性，但其他人认为基础 ChatGPT 模型已足以满足需求，并质疑在仅靠 Prompt Engineering 就能奏效的情况下使用 Custom GPT 的必要性。

- **DALL·E 获得新功能**：DALL·E 已更新，增加了风格建议和图像局部重绘（Inpainting）功能，允许用户编辑 DALL·E 生成图像的特定部分。对于希望利用这些功能的 Plus 计划用户来说，这些信息可能特别有用。

- **模型性能比较**：成员们就各种 GPT 模型和系统的性能进行了交流，一些成员注意到在特定领域，某些模型可能优于其他模型。对话显示出一种细致的理解，即模型性能会根据用例和个人测试而有很大差异。

- **利用 AI 处理 Wiki 数据**：一位成员正在寻求建议，关于如何让 GPT 解释并回答来自包含 Wiki 数据库转储的 XML 文件中的问题。他们表示 GPT 很难根据 XML 文件中的数据提供准确的回答。

- **数据保留问题**：用户询问了 OpenAI 的数据保留政策，特别是删除对话后的情况。回复指出，OpenAI 上删除的对话通常会保留约一个月，尽管删除后用户会立即看不到这些对话。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1225080073271382066)** (27 条消息🔥):

- **翻译困扰**：一名成员在将 Markdown 内容翻译成多种语言（尤其是阿拉伯语）时遇到不一致的问题。尽管尝试调整 Prompt，例如添加“仅返回带有 Markup 的翻译文本，不要原始文本”，但结果参差不齐，部分回复仍未翻译。

- **寻找 Prompt Library**：一位成员询问 **Prompt Library** 的位置，另一位成员迅速通过 Channel ID 将其引导至重命名后的频道。

- **完善 Apple Watch 专家 Prompt**：一位用户寻求建议，希望改进 Prompt 以便在以 Apple Watch 专家身份提问时从 Bot 获得更好的回复。另一位成员建议尝试不同版本的 Prompt，甚至可以使用模型来评估 Prompt 的清晰度以及是否存在潜在的 Hallucinations。

- **Dalle-3 Prompt Engineering 位置查询**：一位用户询问应在何处进行 Dalle-3 Prompt Engineering，是在通用的 Prompt-Engineering 频道还是特定的 Dalle 线程中。一名成员建议这取决于个人选择，但在 Dalle 专用频道可能会获得更集中的帮助。

- **延长文本回复**：一名成员对“使文本更长”的命令不再有效感到沮丧。另一名成员推荐了一种解决方法，包括复制之前的 GPT 回复，开启新的 Chat，然后使用“继续”作为 Prompt。

- **LLM 草拟文档问题**：一名成员就 LLM 在根据 Template 草拟文档时无法返回某些章节的问题寻求帮助，即使这些章节已经过修改。他们正在寻找一种解决方案，以确保所有修改过的章节都能包含在输出中。

---

**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1225080073271382066)** (27 messages🔥): 

- **带有 Markup 的翻译困扰**：一名成员尝试了各种 Prompt 公式，以保留 Markdown Markup 并正确翻译内容（包括专有名词和链接）。尽管优化了 Prompt，但仍面临维持 Markdown 格式和收到未翻译文本的问题，并对翻译的不一致性表示沮丧。

- **寻找 Prompt Library**：当被问及在哪里可以找到 Prompt Library 时，一名成员被引导至重命名为 [#1019652163640762428](https://discord.com/channels/channel_id/1019652163640762428) 的频道，指明了资源所在位置。

- **提高 AI 角色扮演的 Prompt 效能**：在关于提升专家角色扮演 Prompt 质量的讨论中，一名成员建议让 AI 评估 Prompt 的清晰度和一致性。他们讨论了除了“Roleplay”之类的单个关键词外，整个 Prompt 上下文对影响 AI 回复风格的重要性。

- **Dalle-3 Prompt Engineering 讨论位置**：一名成员询问应在何处讨论 Dalle-3 Prompt Engineering——是在 api-discussions 频道还是 Dalle 专用线程。他们被告知可以自行选择，不过在专门的 Dalle 线程中可能会得到更专注的回复。

- **扩展迭代文本生成**：在遇到“使文本更长”命令未按预期生成新内容的问题后，另一名成员建议复制 AI 的回复，启动新的 Chat，然后使用“继续”一词来扩展对话。

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1225004211947966485)** (306 messages🔥🔥): 

- **CmdR 支持即将到来**：讨论表明，在修复 Inference 问题后，为 Unsloth 添加 **CmdR** 支持的工作正在进行中。大家对进展感到兴奋，讨论暗示在当前任务完成后将有一个完成时间表。

- **对自动优化器的期待**：Unsloth 计划推出一项对 **GPU poor** 群体非常重要的全新开源功能，即将正式发布，并于 **4 月 22 日发布新版本和公告**。该功能旨在通过 CPU 优化来改善 AI 的可访问性，支持更广泛的模型，如 **Command R, Mixtral 等**。

- **性能问题解答**：用户参与了关于内存优化、使用 Unsloth 节省 70% VRAM 以及 Inplace Kernel 执行的技术讨论。对话重点关注了*不同模型上的 Data Layout 结果*以及 *Unsloth In-place 操作*在减少内存占用方面的有效性。

- **关于 Gemma 2B 模型的讨论与解惑**：支持在 Notebooks 中根据提供的说明更改为 Gemma 2B 模型，并澄清了下载 4-bit 与 16-bit 模型的区别，保证精度损失通常在 0.1-0.5% 之间。

- **职位发布与招聘伦理讨论**：关于开设求职频道的请求引发了关于无薪实习伦理以及对实习生技能期望的辩论。共识强调了为任何工作提供**经济报酬**的重要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/15g">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1-uKmQzhh8ftxEdipiqGu4sVdRb8MgWv2?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo=FqfebeAdT073">Google Colaboratory</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth)</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#manually-saving-to-gguf">Home</a>：2-5倍速，减少70%显存的 QLoRA 和 LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/myshell-ai/JetMoE">GitHub - myshell-ai/JetMoE: Reaching LLaMA2 Performance with 0.1M Dollars</a>：用10万美元达到 LLaMA2 性能。通过在 GitHub 上创建账号为 myshell-ai/JetMoE 的开发做出贡献。</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>：Linear Attention Sequence Parallelism (LASP)。通过在 GitHub 上创建账号为 OpenNLPLab/LASP 的开发做出贡献。</li><li><a href="https://github.com/toranb/sloth/blob/master/sftune.py">sloth/sftune.py at master · toranb/sloth</a>：使用 unsloth 的 python sftune, qmerge 和 dpo 脚本 - toranb/sloth</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1546dvc/24gb_vram_on_a_budget/">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp">GitHub - ggerganov/llama.cpp: LLM inference in C/C++</a>：C/C++ 环境下的 LLM 推理。通过在 GitHub 上创建账号为 ggerganov/llama.cpp 的开发做出贡献。</li><li><a href="https://github.com/pytorch-labs/ao/pull/95#issuecomment-2028912362">GaLore and fused kernel prototypes by jeromeku · Pull Request #95 · pytorch-labs/ao</a>：内核与工具原型。当前：GaLore。GaLore 内存高效训练的融合内核（fused kernels）初始实现。TODO：triton。用于量化训练的可组合 triton 内核...
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1225229765791973467)** (5 messages): 

- **Unsloth Studio 重构进行中**：由于持续存在的 Bug，Unsloth AI 团队推迟了新版本 Unsloth Studio 的发布。初步版本可能在下月中旬提供，现有的 Unsloth 软件包将保持与 **Colab** 的兼容。

- **新 Tab 自动补全功能进入预发布阶段**：VS Code 和 JetBrains 的 [Continue](https://marketplace.visualstudio.com/items?itemName=Continue.continue) 扩展现已提供 **Tab 自动补全** 的预发布实验性功能。Continue 的开源 autopilot 允许通过询问有关突出显示代码的问题并内联引用上下文，从而更轻松地使用任何 LLM 进行编码，其[文档](https://continue.dev/docs)中通过动画 GIF 进行了展示。

**提到的链接**：<a href="https://marketplace.visualstudio.com/items?itemName=Continue.continue">Continue - Claude, CodeLlama, GPT-4, and more - Visual Studio Marketplace</a>：Visual Studio Code 扩展 - 用于软件开发的开源 autopilot - 将 ChatGPT 的功能带入你的 IDE

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1225000119657566218)** (248 messages🔥🔥): 

- **Tokenizer 问题**：用户遇到的错误是由于 Tokenizer 中模型命名不正确，导致无法正确写入，进而引发执行问题。他们已自行解决该问题。

- **成功的模型保存与 Huggingface 推送**：用户讨论了使用 `model.save_pretrained_merged()` 和 `model.push_to_hub_merged()` 保存模型，重点是正确设置模型保存和 Huggingface 推送的命名参数。相关建议包括将占位符替换为 Huggingface 用户名/模型名，并从 Huggingface 设置中获取 Write 令牌（token）。

- **Gemma 上的推理问题**：一位用户遇到了与 `GemmaForCausalLM` 对象缺少 `layers` 属性相关的 `AttributeError`，该问题已通过 Unsloth 的更新修复，要求用户在个人机器上重新安装该软件包。

- **GGUF 转换与 Docker 环境的挑战**：用户分享了将模型转换为 GGUF 格式时遇到的问题，以及一个 Docker 环境产生错误的案例，该错误通过安装 `python3.10-dev` 得到了解决。

- **微调挑战与解决方案**：讨论内容包括在 Colab 中微调 Gemma 模型、在 Sagemaker 上使用 24GB GPU 时遇到 `OutOfMemoryError` 的补救措施、转换后 GGUF 单词拼写的奇特现象，以及关于使用修改后的参数恢复训练的见解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/settings/tokens">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/adding-accuracy-precision-recall-and-f1-score-metrics-during-training/16419/2">在训练期间添加 accuracy, precision, recall 和 f1 score 指标</a>：嗨，你可以定义自己的计算指标函数并将其传递给 trainer。这是一个计算指标的示例。从 sklearn.metrics 导入 accuracy_score 定义 accuracy 指标函数...</li><li><a href="https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0">TinyLlama/TinyLlama-1.1B-Chat-v1.0 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat/">deepseek-ai/deepseek-vl-7b-chat · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF">qwp4w3hyb/deepseek-coder-7b-instruct-v1.5-iMat-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/danielhanchen/model_21032024">danielhanchen/model_21032024 · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/deepseek-coder-6.7B-instruct-GGUF">TheBloke/deepseek-coder-6.7B-instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://docs.wandb.ai/guides/integrations/huggingface">Hugging Face Transformers | Weights &amp; Biases 文档</a>：Hugging Face Transformers 库使 BERT 等最先进的 NLP 模型以及 mixed precision 和 gradient checkpointing 等训练技术变得易于使用。W&amp;B 集成增加了丰富的...</li><li><a href="https://huggingface.co/docs/trl/main/en/sft_trainer#trl.trainer.ConstantLengthDataset">Supervised Fine-tuning Trainer</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-oom-or-crashing">主页</a>：快 2-5 倍，减少 70% 显存，QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth/wiki#saving-models-to-16bit-for-vllm">主页</a>：快 2-5 倍，减少 70% 显存，QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/oobabooga/text-generation-webui">GitHub - oobabooga/text-generation-webui: 一个用于 Large Language Models 的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。</a>：一个用于 Large Language Models 的 Gradio Web UI。支持 transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。- oobabooga/text-generation-webui</li><li><a href="https://github.com/unslothai/unsloth/pull/300">由 eabdullin 修复 GemmaModel_fast_forward_inference · Pull Request #300 · unslothai/unsloth</a>：在快速推理时，Gemma 模型报错 'GemmaCausalLM' has no attribute 'layers'。对此的快速修复。</li><li><a href="https://github.com/abetlen/llama-cpp-python">GitHub - abetlen/llama-cpp-python: llama.cpp 的 Python 绑定</a>：llama.cpp 的 Python 绑定。通过在 GitHub 上创建一个账号来为 abetlen/llama-cpp-python 做出贡献。</li><li><a href="https://github.com/unslothai/unsloth/">GitHub - unslothai/unsloth: 快 2-5 倍，减少 70% 显存，QLoRA &amp; LoRA 微调</a>：快 2-5 倍，减少 70% 显存，QLoRA &amp; LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1225086904501014601)** (86 条消息🔥🔥): 

- **Stable Audio 2.0 发布**：Stability AI 推出了 **Stable Audio 2.0**，能够根据单个提示词生成长达三分钟、采样率为 44.1 kHz 立体声、具有连贯音乐结构的高质量完整曲目。用户可以在 [stableaudio.com](http://stableaudio.com) 免费体验该模型，并在此处阅读博客文章 [here](https://bit.ly/3xlwmP5)。

- **AssemblyAI 的新语音模型超越 Whisper-3**：AssemblyAI 发布了 **Universal-1**，该模型声称比 Whisper-3 准确率高 13.5%，且 hallucinations 减少多达 30%，能够在 38 秒内处理 60 分钟的音频，尽管它目前仅支持 20 种语言。可以在 [assemblyai.com](https://www.assemblyai.com/playground) 的免费游乐场中进行测试。

- **在 ChatGPT Plus 中编辑 DALL-E 图像**：ChatGPT Plus 现在允许用户在 Web 端和 iOS 应用上编辑 DALL-E 图像及其对话提示词。说明和用户界面详情可以在[此处](https://help.openai.com/en/articles/9055440-editing-your-images-with-dall-e)找到。

- **slono 发起的 AI 框架讨论**：Slono 分享了关于将 AI Agent 构建为微服务/事件驱动架构以实现更好可扩展性的想法，引用了类似于 Actor Model of Computation 的理念，并为其 Golang 框架寻求反馈或帮助。

- **Opera 支持下载并运行本地 LLM**：Opera 现在允许用户在本地下载并运行大语言模型 (LLM)，首先面向拥有开发者流更新的 Opera One 用户开放。该浏览器正在使用开源的 Ollama 框架，并计划从各种来源添加更多模型供用户选择。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://dev.to/devteam/join-us-for-the-cloudflare-ai-challenge-3000-in-prizes-5f99">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/horseracedpast/status/1775757613000507736?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 horseboat (@horseracedpast) 的推文</a>: Bengio 真的在 2013 年就写过这个了 ↘️ 引用 AK (@_akhaliq) Google 发布 Mixture-of-Depths 在基于 Transformer 的语言模型中动态分配计算 基于 Transformer 的语言模型...</li><li><a href="https://deluxe-fairy-96e9ff.netlify.app/">React App</a>: 未找到描述</li><li><a href="https://x.com/theseamouse/status/1775743110774931846?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>: @fouriergalois @GoogleDeepMind 兄弟，带早期退出的 MoE。整个图都下移了，这相当于节省了 10 倍的计算量... 兄弟们</li><li><a href="https://openai.com/blog/introducing-improvements-to-the-fine-tuning-api-and-expanding-our-custom-models-program">介绍 Fine-tuning API 的改进并扩展我们的自定义模型计划</a>: 我们正在添加新功能，以帮助开发者更好地控制 Fine-tuning，并宣布与 OpenAI 一起构建自定义模型的新方法。</li><li><a href="https://overcast.fm/+HaNOG0VjE/19:08">孩子们还需要学习编程吗？(Practical AI #263) &mdash; Changelog Master Feed &mdash; Overcast</a>: 未找到描述</li><li><a href="https://x.com/theseamouse/status/1775782800362242157?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Hassan Hayat 🔥 (@TheSeaMouse) 的推文</a>: 为什么 Google DeepMind 的 Mixture-of-Depths 论文以及更广泛的动态计算方法很重要：大部分计算都被浪费了，因为并非所有 Token 的预测难度都相同</li><li><a href="https://techcrunch.com/2024/04/03/opera-will-now-allow-users-download-and-use-llms-locally">Opera 允许用户在本地下载并使用 LLM | TechCrunch</a>: Opera 今天表示，现在将允许用户在桌面上本地下载并使用大语言模型 (LLM)。</li><li><a href="https://overcast.fm/+_C9f-UYD4">与来自 LangChain 的 Harrison Chase 探讨开源 AI 应用开发 &mdash; No Priors: Artificial Intelligence | Machine Learning | Technology | Startups &mdash; Overcast</a>: 未找到描述</li><li><a href="https://x.com/StabilityAI/status/1775501906321793266?s=20">来自 Stability AI (@StabilityAI) 的推文</a>: 推出 Stable Audio 2.0 —— 一个能够从单个提示词生成长达三分钟、具有连贯音乐结构的高质量完整曲目（44.1 kHz 立体声）的新模型。探索...</li><li><a href="https://x.com/nickadobos/status/1775638457412722757?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Nick Dobos (@NickADobos) 的推文</a>: 新版 DALL-E 太棒了，我的天。比我尝试过的任何其他东西都更具可控性。我用 3 个提示词做了一个应用原型。哇！！甚至还搞定了标签栏和布局</li><li><a href="https://x.com/cohere/status/1775878850699808928?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 cohere (@cohere) 的推文</a>: 今天，我们推出 Command R+：一款最先进的、针对 RAG 优化的 LLM，旨在处理企业级工作负载并支持全球商业语言。我们的 R 系列模型家族现已推出...</li><li><a href="https://x.com/sherjilozair/status/1775765404528615798?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Sherjil Ozair (@sherjilozair) 的推文</a>: 这怎么发表的？🤔 ↘️ 引用 AK (@_akhaliq) Google 发布 Mixture-of-Depths 在基于 Transformer 的语言模型中动态分配计算 基于 Transformer 的语言模型...</li><li><a href="https://x.com/andersonbcdefg/status/1775751252330385807?s=20">来自 Ben (e/sqlite) (@andersonbcdefg) 的推文</a>: 太神奇了。“你喜欢 MoE？如果我们把其中一个专家设为恒等函数会怎样。” 砰的一声，节省了 50% 的 FLOPs 🤦‍♂️ ↘️ 引用 Aran Komatsuzaki (@arankomatsuzaki) Google 发布 Mixture-of-Depths...</li><li><a href="https://x.com/gblazex/status/1775558982645547236?s=20">来自 Blaze (Balázs Galambosi) (@gblazex) 的推文</a>: 哇。当 OpenAI API 还停留在 Whisper-2 时，@AssemblyAI 发布了甚至超越 Whisper-3 的产品：比 Whisper-3 准确率高 13.5% + 幻觉减少多达 30% + 处理 60 秒音频仅需 38 秒...</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09">加入我们的云高清视频会议</a>: Zoom 是现代企业视频通信的领导者，拥有简单、可靠的云平台，可跨移动端、桌面端和会议室系统进行视频和音频会议、聊天及网络研讨会。Zoom ...</li><li><a href="https://www.reddit.com/r/computervision/comments/1bvaak0/stanford_cs_25_transformers_course_open_to/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://lu.ma/paperclub3">SDxPaperClub · Luma</a>: SDx 论文俱乐部。将要展示的论文是 [待定]，作者 [待定] Twitter | Discord | LinkedIn</li><li><a href="https://hlfshell.ai/posts/representation-engineering/">Representation Engineering</a>

Engineering and Control Vectors - Neuroscience for LLMs</a>: tl;dr 最近的一篇论文以类似于神经科学的方式研究了大型语言模型 (LLM) 对刺激的反应，揭示了一种用于控制和理解 LLM 的诱人工具。我在这里写道...</li><li><a href="https://github.com/Paitesanshi/LLM-Agent-Survey">GitHub - Paitesanshi/LLM-Agent-Survey</a>: 通过在 GitHub 上创建账户，为 Paitesanshi/LLM-Agent-Survey 的开发做出贡献。</li><li><a href="https://abyssinian-molybdenum-f76.notion.site/237e9f7515d543c0922c74f4c3012a77?v=0a309e53d6454afcbe7a5a7e169be0f9">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://youtu.be/5q0GN2M1d2c?si=zRsm4Jye_YO8jfBz">Multimodal AI: Antonio Torralba</a>: Antonio Torralba，MIT 电气工程与计算机科学系及 CSAIL 教授，谈论视觉感知和语言模型。Torralba 的演讲是...的一部分</li><li><a href="https://www.amazon.com/Foundations-Computer-Adaptive-Computation-Learning/dp/0262048973">未找到标题</a>: 未找到描述</li><li><a href="https://mitpress.ublish.com/ebook/foundations-of-computer-vision-1-preview/12791/Cover">eReader</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1225158181316067422)** (356 条消息🔥🔥): 

- **详细总结入门**：成员们讨论了使用 **DSPy** 优化基础模型 Prompt 的细节，重点关注其在模型迁移和针对任意指标进行优化方面的功效。Eric 分享了他的演示，参与者们以掌声对他的见解表示认可。
  
- **Devin 引起关注**：对话转向了 **Devin** 的多重影响，成员们分享了可以使用这一备受瞩目的 AI 模型尝试的各种项目想法。
  
- **关于优化调用的热点话题**：俱乐部识别了 **DSPy** 的优化技术，并对 **.compile()** 函数调用期间由于 **DSPy** 发起的大量调用而导致的 API 速率限制 (rate limits) 表示担忧。
  
- **务实的编程考量**：关于 **DSPy** 与其他方法/框架的实际用例、其在不同语境下的优势，以及如何减轻模型迁移过程中的 Prompt 债务 (prompt debt) 等问题被提出。
  
- **技术和任务推测**：使用 **Devin** 的潜在应用建议从集成语音 API 的 iOS 应用到 **DSPy** 文档重写不等，展示了社区将 AI 应用于各种挑战的广泛兴趣。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://colab.research.google.com/drive/1KZR1sGTp_RLWUJPAiK1FKPKI-Qn9neUm?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://app.sli.do/event/bNV6mo3BFGhe8Bqzb1tonb/live/questions">加入 Slido：输入 #code 进行投票和提问</a>：参与实时投票、测验或问答。无需登录。</li><li><a href="https://arxiv.org/abs/2402.17764">1-bit LLM 时代：所有大语言模型都是 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://eugeneyan.com/writing/abstractive/">生成式摘要的评估与幻觉检测</a>：基于参考、上下文和偏好的指标，自一致性以及捕捉幻觉。</li><li><a href="https://arxiv.org/abs/2310.03714">DSPy：将声明式语言模型调用编译为自改进流水线</a>：ML 社区正在迅速探索提示语言模型 (LM) 并将其堆叠到解决复杂任务的流水线中的技术。不幸的是，现有的 LM 流水线通常是...</li><li><a href="https://eugeneyan.com/writing/evals/#summ">有效与无效的 LLM 特定任务评估</a>：用于分类、摘要、翻译、版权重复和毒性的评估。</li><li><a href="https://www.spotery.com/">你是人类吗？</a>：未找到描述</li><li><a href="https://eugeneyan.com/writing/evals/#summarization-consistency-relevance-length">有效与无效的 LLM 特定任务评估</a>：用于分类、摘要、翻译、版权重复和毒性的评估。</li><li><a href="https://hamel.dev/blog/posts/prompt/#dspy">- 去你的，给我看 Prompt。</a>：通过拦截 API 调用快速理解难以捉摸的 LLM 框架。</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/examples/knn.ipynb">stanfordnlp/dspy 项目 main 分支下的 dspy/examples/knn.ipynb</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://github.com/seanchatmangpt/dspygen">GitHub - seanchatmangpt/dspygen：一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。</a>：一个为 DSPy (Demonstrate, Search, Predict) 项目提供的 Ruby on Rails 风格框架，适用于 GPT、BERT 和 LLama 等语言模型。 - seanchatmangpt/dspygen</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy：DSPy：用于编程（而非提示）基础模型的框架</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://x.com/HamelHusain/status/1774999027538612652?s=20">来自 Hamel Husain (@HamelHusain) 的推文</a>：@swyx 一个人 + 一小群狂热粉丝
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1225485539952689162)** (2 条消息): 

- **探索用于增强 Mistral 的 LoRA**：有人建议在 **Mistral 7B** 之类模型的基础上创建 **LoRA** (Low-Rank Adaptation)，以在特定任务中实现卓越性能。
- **计划进行高级拆分和标记**：该方法已确认处于规划阶段，任务不仅涉及拆分句子，还涉及根据特定分类法对每个句子进行拆分和标记。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1225106882281340989)** (9 条消息🔥): 

- **网络爬虫难题**：在讨论**可扩展网络爬虫**时，成员们承认了获取大规模高质量数据集的挑战，并指出由于需要无头浏览器、绕过反爬措施以及渲染现代 JavaScript 框架，复杂性和成本有所增加。

- **秘密档案**：一位成员暗示存在拥有大量高质量数据的**存档小组**，这表明存在一个存档广泛数据集的隐秘社区。

- **寻找数据囤积者**：针对有关**存档小组**的问题，另一位参与者阐明了出于原则存档数据的人与单纯的数据囤积者之间的区别。

- **知识收集者的数据搜寻**：一位成员建议将 **Common Crawl** 作为对网络爬虫和尖端数据收集感兴趣的人的资源。

- **永恒歌单更新**：一条轻松的消息，一位成员提到为自己的葬礼选择了一首**新歌**，这代表了个人兴趣，也是从硬核技术讨论中的一种放松。
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1224997466638651403)** (10 messages🔥): 

- **Ollama Server 上的 Lollms**：分享了一个关于安装和在 Ollama Server 上使用 lollms 的 [YouTube 教程](https://www.youtube.com/watch?v=RuQSQmolXGE)，旨在引导观众完成安装过程。
- **来自中国的更便宜 AI 芯片**：[云天励飞 (Intellifusion) 的 DeepEyes](https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus) AI 盒子售价约 140 美元，提供 48 TOPS 的 AI 性能，旨在为 AI 应用中的高端硬件提供具有成本效益的替代方案。
- **时间的精度**：一位成员引用了维基百科上的 [ISO 8601 标准](https://en.wikipedia.org/wiki/ISO_8601)，详细说明了以不同格式（包括 UTC 和偏移量）表达当前日期和时间的精确方式。
- **GitHub 上的数据集加载器**：Hugging Face 推出了 [Chug](https://github.com/huggingface/chug)，这是一个包含最小化分片数据集加载器、解码器以及用于多模态文档、图像和文本数据集工具的仓库。
- **CohereForAI 的多语言 LLM**：CohereForAI 宣布发布 C4AI Command R+，这是一个支持 10 种语言的多语言 104B LLM，扩展了其开源权重产品线，可以在其 [Hugging Face 页面](https://huggingface.co/CohereForAI/c4ai-command-r-plus)上找到。
- **GPT-4 微调定价策略**：OpenAI 正在针对 GPT-4 微调进行实验性定价，以了解质量、安全性和使用情况，详情见最近的一篇[博客文章](https://openai.com/gpt-4-ft-experimental-pricing)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/chinese-chipmaker-launches-14nm-ai-processor-thats-90-cheaper-than-gpus">中国芯片制造商推出 14nm AI 处理器，比 GPU 便宜 90% —— 140 美元的芯片采用旧工艺规避美国制裁</a>：如果有办法规避制裁，你知道中国一定会这么做。</li><li><a href="https://openai.com/gpt-4-ft-experimental-pricing">GPT-4 微调</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/ISO_8601">ISO 8601 - 维基百科</a>：未找到描述</li><li><a href="https://x.com/cohereforai/status/1775878631715217522?s=46&t=stOPrwZiN_fxSK0RuC8Flg">来自 Cohere For AI (@CohereForAI) 的推文</a>：宣布 C4AI Command R+ 开源权重，这是一个具有 RAG、工具使用和 10 种语言多语言能力的尖端 104B LLM。此版本基于我们的 35B 模型构建，是我们致力于让 AI 普及的一部分...</li><li><a href="https://www.youtube.com/watch?v=RuQSQmolXGE">安装并释放 Ollama Server 上 lollms 的力量：一个有趣的技术教程 🚀</a>：🌟 嘿 YouTube 的家人们！🤓 我非常激动地向大家展示我的最新视频！在这个启发性的教程中，我将带你完成安装过程...</li><li><a href="https://github.com/huggingface/chug">GitHub - huggingface/chug：用于多模态文档、图像和文本数据集的最小化分片数据集加载器、解码器和工具。</a>：用于多模态文档、图像和文本数据集的最小化分片数据集加载器、解码器和工具。 - huggingface/chug
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1225025576503345152)** (150 messages🔥🔥): 

- **语言模型剪枝探索**：一位成员正在尝试创建一个剪枝后的 **Jamba 模型**（尺寸缩小了 25%）。他们正在使用自定义的层剪枝脚本，讨论其方法论，并提到了一篇相关的[关于层剪枝的研究论文](https://arxiv.org/abs/2403.17887)，该论文探讨了在不显著影响性能的情况下减少层数的策略。

- **LLM 的动态计算分配**：成员们讨论了一篇 [Google 论文](https://arxiv.org/abs/2404.02258v1)，该论文指出 Transformer 可以学习在序列中动态分配计算资源。对话围绕其在更高效的预训练和推理方面的潜力展开，并将其与 Speculative Decoding 方法进行了比较，讨论了对模型重训练的影响。

- **关于 Speculative Decoding 的讨论**：Speculative Decoding 技术得到了解释和审查，一位参与者强调了它与 Google 动态计算方法的区别。成员们还交流了 GPU 中的内存管理以及用于加速响应的 Batching 技术。

- **Cohere 发布 Command R+ 模型**：Command R+ 是 **Cohere** 推出的一款针对 Retrieval Augmented Generation (RAG) 优化的新模型，社区对其进行了分享和简要讨论。该模型旨在商业应用中扩展 LLMs，提供多语言支持和高级引用等功能。

- **神经推理探索**：Discord 用户讨论了 [GitHub 上的 neurallambda 项目](https://github.com/neurallambda/neurallambda)，该项目尝试将 lambda calculus 与基于 Transformer 的 LLMs 相结合。这种神经符号（neurosymbolic）方法可能在 AI 推理领域具有开创性意义。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">Introducing Command R+: A Scalable LLM Built for Business</a>: 介绍 Command R+：专为业务构建的可扩展 LLM。Command R+ 是一款最先进的、针对 RAG 优化的模型，旨在处理企业级工作负载，并首先在 Microsoft Azure 上可用。今天，我们推出 Command R+，我们最强大的...</li><li><a href="https://arxiv.org/abs/2404.02684">Cross-Architecture Transfer Learning for Linear-Cost Inference Transformers</a>: 线性成本推理 Transformer 的跨架构迁移学习：最近，提出了多种架构，通过改变 self-attention 块的设计来实现线性成本推理，从而提高 Transformer 语言模型的效率...</li><li><a href="https://lupantech.github.io/inter-gps/">Inter-GPS</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.02258v1">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: Mixture-of-Depths：在基于 Transformer 的语言模型中动态分配计算。基于 Transformer 的语言模型在输入序列中均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算）分配给特定的...</li><li><a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>: 深层网络不合理的无效性：我们对流行的开源预训练 LLM 系列进行了一种简单的层剪枝策略的实证研究，发现在不同的问答基准测试中，性能几乎没有下降，直到...</li><li><a href="https://arxiv.org/abs/2404.02893">ChatGLM-Math: Improving Math Problem-Solving in Large Language Models with a Self-Critique Pipeline</a>: ChatGLM-Math：通过自我批判流水线提高大语言模型的数学解题能力。大语言模型（LLMs）已表现出对人类语言的卓越掌握，但在需要数学解题的现实应用中仍然面临困难。虽然许多策略和数据集...</li><li><a href="https://huggingface.co/danielus/MermaidSolar-Q4_K_S-GGUF">danielus/MermaidSolar-Q4_K_S-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2404.02078">Advancing LLM Reasoning Generalists with Preference Trees</a>: 利用偏好树推进 LLM 推理通用模型：我们推出了 Eurus，这是一套针对推理优化的语言大模型（LLMs）。Eurus 模型基于 Mistral-7B 和 CodeLlama-70B 进行微调，在开源模型中取得了最先进的结果...</li><li><a href="https://arxiv.org/html/2404.02258v1">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>: 未找到描述</li><li><a href="https://huggingface.co/learn">Hugging Face - Learn</a>: 未找到描述</li><li><a href="https://course.fast.ai/">Practical Deep Learning for Coders - Practical Deep Learning</a>: 程序员实用深度学习 - 实用深度学习：一门为有一定编程经验、想学习如何将深度学习和机器学习应用于实际问题的人设计的免费课程。</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/neurallambda/neurallambda">GitHub - neurallambda/neurallambda: Reasoning Computers. Lambda Calculus, Fully Differentiable. Also Neural Stacks, Queues, Arrays, Lists, Trees, and Latches.</a>: 推理计算机。Lambda 演算，完全可微。还包括神经栈、队列、数组、列表、树和锁存器。 - neurallambda/neurallambda</li><li><a href="https://huggingface.co/datasets/glaiveai/glaive-code-assistant-v3">glaiveai/glaive-code-assistant-v3 · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/e-p-armstrong/augmentoolkit">GitHub - e-p-armstrong/augmentoolkit: Convert Compute And Books Into Instruct-Tuning Datasets</a>: 将计算和书籍转换为指令微调数据集 - e-p-armstrong/augmentoolkit</li><li><a href="https://www.reddit.com/r/Oobabooga/s/ApIzWEdZu7">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://huggingface.co/TroyDoesAI/MermaidMistral">TroyDoesAI/MermaidMistral · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi">But what is a neural network? | Chapter 1, Deep learning</a>: 神经网络究竟是什么？ | 第一章，深度学习。什么是神经元，为什么会有层，其背后的数学原理是什么？资助未来的项目：https://www.patreon.com/3blue1brown 编写/互动...</li><li><a href="https://www.youtube.com/watch?v=wjZofJX0v4M&t=430s&pp=ygULM2JsdWUxYnJvd24%3D">But what is a GPT?  Visual intro to Transformers | Chapter 5, Deep Learning</a>: GPT 究竟是什么？Transformer 的视觉入门 | 第五章，深度学习。Transformer 及其先决条件介绍。赞助者可提前观看下一章：https://3b1b.co/early-attention 特别感谢这些支持...</li><li><a href="https://www.youtube.com/watch?v=fNk_zzaMoSs&list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab">Vectors | Chapter 1, Essence of linear algebra</a>: 向量 | 第一章，线性代数的本质。从基础开始线性代数系列。资助未来的项目：https://www.patreon.com/3blue1brown 同样有价值的支持方式是...
</li>
</ul>

</div>
  

---

**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1225008200940781670)** (58 messages🔥🔥): 

- **BitNet 讨论**：辩论了 **BitNet-1.58** 中按 "scale" 进行除法的问题，用户质疑其必要性，并表示这可能会阻碍三元编码（ternary encoding）的优势。然而，有人指出在训练和缩放输出时**保持 FP16** 可能有利于数值稳定性。
- **Eurus 模型引起关注**：**Eurus-7b-kto** 是 OpenBMB 推出的一款针对推理优化的 LLM，已配合其微调数据集 **[UltraInteract_sft](https://huggingface.co/datasets/openbmb/UltraInteract_sft)** 和 **[UltraInteract_pair](https://huggingface.co/datasets/openbmb/UltraInteract_pair)** 进行了测试，并建议将 SOLAR 应用于该模型以寻求潜在改进。
- **代码库中的 Function Calling**：Hermes-Function-Calling 代码库中报告了实现差异，涉及*函数调用和编码标准的问题*。该问题中特别提到了 **langchain 的 convert_to_openai_tool()** 的用法。
- **QLoRA 受到关注**：**QLoRA** 作为一种近期的 LLM 微调方法被提及，认为其可能比 LoRA 更高效，在仅需**一半 VRAM** 的情况下提供类似的性能提升。
- **用于指令生成的 Genstruct**：简要讨论了 **NousResearch** 的指令生成模型 **Genstruct 7B** 的实用性和多样性，强调了其基于原始文本语料库为微调数据集创建多样化指令格式的潜力。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>: no description found</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: no description found</li><li><a href="https://github.com/NousResearch/Hermes-Function-Calling/issues/14">This Repo needs some refactoring for the function calling to work properly · Issue #14 · NousResearch/Hermes-Function-Calling</a>: Guys i think there is some issue with the way things are implemented currently in this repo biggest of which is regarding coding standard currently you guys use convert_to_openai_tool from langchai...</li><li><a href="https://arxiv.org/abs/2401.03462">Soaring from 4K to 400K: Extending LLM&#39;s Context with Activation Beacon</a>: The utilization of long contexts poses a big challenge for LLMs due to their limited context window size. Although the context window can be extended through fine-tuning, it will result in a considera...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1225294779420508240)** (2 messages): 

- **LLava 的 ChatML 已修复**：一名成员分享了成功解决 "llava" 项目中 **ChatML** 相关问题的情况。目前没有关于问题本质或解决方式的进一步解释或细节。
- **Hermes-Vision-Alpha 的潜在修复**：同一名成员表示打算着手解决 **Hermes-Vision-Alpha** 的问题。未提供这些问题的性质或具体修复方案的细节。
  

---


**Nous Research AI ▷ #[bittensor-finetune-subnet](https://discord.com/channels/1053877538025386074/1213221029359657021/1225323698710642708)** (2 messages): 

- **Finetuning Miner 报错**：一名成员在运行 **finetuning-subnet** 代码库中的 `miner.py` 脚本时遇到错误。得到的协助指出问题可能在于**缺失依赖项**。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1225083842071630014)** (34 messages🔥):

- **Glaive 的数据生成贡献**：[Glaive](https://huggingface.co/datasets/glaiveai/rag_sample) 创建了一个示例数据集，以协助 RAG 应用的数据生成，展示了将多个文档整合到回复中的能力。
- **RAG Grounding 概念澄清**：Sahilch 解释了 RAG 中的 **grounded** 模式如何区分模型何时应排他性地使用文档上下文，以及何时应将自身知识与文档融合，从而为响应生成过程增加了粒度。
- **RAG 命令与引用标记**：Interninja 讨论了正确引用标记的重要性，建议使用 JSON 格式进行引用可能会更有利，并分享了针对带有 RAG 功能的新 [CommandR+ 的 XML 指令格式](https://x.com/LangChainAI/status/1775917799065653250?s=20)，其中包括复杂的跨步查询，并使用 `<co: doc>` 标签来引用文档。
- **Cohere 的 RAG 文档**：Bjoernp 强调了 RAG 与 function calling 结合的潜力，分享了 [Cohere RAG 文档链接](https://docs.cohere.com/docs/retrieval-augmented-generation-rag)，并讨论了其可接受使用政策（Acceptable Use Policy）中关于合成数据生成的含义。
- **RAG 应用中的检索过滤**：Iriden 提倡在 RAG 的检索和响应之间增加一个过滤步骤，这在实践中取得了成功，特别是当用户参与选择过程以获得更精炼的结果时。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/LangChainAI/status/1775917799065653250?s=20">来自 LangChain (@LangChainAI) 的推文</a>：结合 Cohere 新型 Command-R+ 的 Adaptive RAG。Adaptive-RAG（SoyeongJeong97 等人）是最近的一篇论文，它结合了 (1) 查询分析和 (2) 迭代式答案构建，以无缝处理查询...</li><li><a href="https://docs.cohere.com/docs/retrieval-augmented-generation-rag">检索增强生成 (RAG) - Cohere 文档</a>：未找到描述</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI 可接受使用政策</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit">RAG/长上下文推理数据集</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/glaiveai/rag_sample">glaiveai/rag_sample · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1225008774318788659)** (96 messages🔥🔥): 

- **复制粘贴的怪癖**：成员们讨论了桌面端与移动端相比在复制粘贴上的困难，注意到网站上的每个字符都被包裹在 `<span>` 中，导致操作具有挑战性。一位成员提到编写了一个 Python 程序来生成相应的 HTML 代码进行“粘贴”，但最初导致了网站崩溃。
- **WorldSim 运行变慢的担忧与解决方案**：讨论强调了对网站在长时间使用后变慢的担忧，尤其是在移动端。建议的解决方案包括从存档重新加载，同时指出尽管原始 WorldSim 缺乏变体版本中的便捷功能，但其性能表现最佳。
- **分享 WorldSim 系统提示词 (System Prompts)**：分享了 WorldSim 的系统提示词，并澄清该提示词已通过 Twitter 帖子公开，[Pastebin](https://pastebin.com/Gj7CpdSE) 上也发布了一个更易于复制的版本。
- **WorldSim 命令汇编**：分享了一个更新后的 WorldSim 命令索引链接，其中包含供用户参考的[高级命令](https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)，并引发了关于用于遣散人格实体的 "sublimate" 命令的讨论。
- **Claude 模型越狱与对 ASCII 艺术的困惑**：用户尝试使用 Claude 模型绕过预设提示词，并在 labs.perplexity.ai 上报告了成功结果。另一位用户询问 WorldSim 生成的女性面部 ASCII 艺术，揭示了它代表 Nous girl 的徽标。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/karan4d/status/1768836844207378463?s=20">mephisto 🤡7 (@karan4d) 的推文</a>：我正在开源 worldsim，当然，我提供了 worldsim 的 sysprompt 和初始化对话：sysprompt：&lt;sys&gt;助手今天处于 CLI 模式。人类正在直接与模拟器交互...</li><li><a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4">Notion – 集笔记、任务、维基和数据库于一体的一站式工作空间。</a>：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一站式工作空间。</li><li><a href="https://worldsim.notion.site/WorldSim-Command-Index-961c8849f61e4558949716b1dfd5f9fa?pvs=4)">Notion – 集笔记、任务、维基和数据库于一体的一站式工作空间。</a>：一款将日常工作应用融为一体的新工具。它是为您和您的团队打造的一站式工作空间。</li><li><a href="https://tenor.com/view/friends-ross-geller-david-schwimmer-tv-series-american-sitcom-gif-17315839">《老友记》Ross Geller GIF - Friends Ross Geller David Schwimmer - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/feel-me-think-about-it-meme-gif-7715402">Feel Me Think About It GIF - Feel Me Think About It 迷因 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/oscars-standing-ovation-clap-clapping-applause-gif-5089552">起立鼓掌 GIF - 奥斯卡起立鼓掌 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://pastebin.com/0fjwccgM">WorldSim 超级英雄宇宙扩展命令集 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/aLLKvkqq">WorldSim 叙事创作扩展命令集 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。</li><li><a href="https://pastebin.com/Gj7CpdSE">Karan4D 的 WorldSim System Prompt 开源 - Pastebin.com</a>：Pastebin.com 自 2002 年以来一直是排名第一的文本粘贴工具。Pastebin 是一个可以在线存储文本一段时间的网站。
</li>
</ul>

</div>
  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1225213583210713170)** (16 条消息🔥): 

- **分享 GitHub Workflow 示例**：一位用户提供了一个与 *modular auth* 和 *Mojo packaging* 相关的 GitHub workflow 示例。最初[分享的链接](https://github.com/Moosems/Fastq_Parser/blob/main/.github/workflows/package.yml)无法访问，但随后跟进了一个复制粘贴的 workflow 代码片段。

- **寻找调试器和编辑器**：一位成员询问了除 VSCode 之外的编辑器（特别是 *neovim*）是否提供调试器和 LSP。

- **提供 Discord 解决方案链接**：针对一位用户遇到的问题，另一位成员引导他们查看之前在 Discord 消息中发布的解决方案，但该解决方案的链接不完整。

- **社区直播通知**：一位用户指出即将举行的“Modular 社区直播”缺乏通知。提供了[直播链接](https://www.youtube.com/watch?v=PL71FV2KKHE)，讨论内容为“MAX 24.2 新特性”。

- **请求 Mojo 完成路线图**：一篇源自 Mojo 频道的帖子被分享到 general 频道，请求提供 Mojo 项目“完成”的详细路线图，并与 Taichi 或 Triton 进行比较。另一位用户通过分享 [Mojo 开发路线图链接](https://docs.modular.com/mojo/roadmap)解决了这个问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/roadmap">Mojo🔥 路线图与已知局限 | Modular 文档</a>：Mojo 计划摘要，包括即将推出的功能和需要修复的问题。</li><li><a href="https://www.youtube.com/watch?v=PL71FV2KKHE">Modular 社区直播 - MAX 24.2 新特性</a>：MAX 24.2 现已发布！加入我们即将举行的直播，我们将讨论 MAX 的所有新特性——开源 Mojo 标准库、MAX Engine 支持...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1225111389652123720)** (4 条消息):

- **推文提醒**：Modular 在 Twitter 上分享了一条 [推文](https://twitter.com/Modular/status/1775549728400572660)。
- **Modular 的 Twitter 更新**：Modular 官方 Twitter 账号发布了另一条 [推文](https://twitter.com/Modular/status/1775583583530524987)。
- **推文分享环节**：查看这条最近的 Modular [推文](https://twitter.com/Modular/status/1775926484869541894) 以获取最新见解。
- **另一条值得关注的推文**：Modular 继续其 Twitter 连更，发布了这条 [帖子](https://twitter.com/Modular/status/1775946487186555225)。
  

---


**Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1225139017121529927)** (4 条消息): 

- **提议将 Mojo 与 ROS 2 集成**：一名成员建议将 [Mojo](https://github.com/) 与 [ROS 2](https://github.com/ros2) 集成，认为 Mojo 的内存安全实践可以减少 ROS 2 中的 Bug。他们强调了通过 [ros2-rust](https://github.com/ros2-rust/ros2_rust) 实现的 Rust 支持，并提到 ROS 2 正在采用一种新的中间件 [zenoh-plugin-ros2dds](https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds)，该中间件也是用 Rust 编写的。

- **ROS 2 社区主要使用 Python 而非 Rust**：有人指出，大多数 ROS 2 社区成员具有偏好 Python 的研究背景，通常不使用 Rust。这一贡献反映了社区在机器人和 AI 相关项目中的整体编程偏好。

- **Python 在机器人领域的局限性导致向 C++ 转型**：该成员分享了他们在 ROS 方面的经验，指出虽然 Python 在机器人初始开发中很方便，但通常速度太慢，导致在严肃应用中需要用 C++ 重写系统。

- **Mojo 在 Nvidia Jetson 硬件上的机遇**：该成员指出 Mojo 有潜力利用 [Nvidia Jetson hardware](https://developer.nvidia.com/embedded/jetson-modules)，该硬件在机器人领域的使用日益增加，但其性能受到 Python 全局解释器锁 (GIL) 的限制。

**提到的链接**：<a href="https://github.com/ros2-rust/ros2_rust">GitHub - ros2-rust/ros2_rust: Rust bindings for ROS 2</a>：ROS 2 的 Rust 绑定。通过在 GitHub 上创建账号为 ros2-rust/ros2_rust 的开发做出贡献。

  

---


**Modular (Mojo 🔥) ▷ #[tech-news](https://discord.com/channels/1087530497313357884/1151359062789857280/1225083116880662580)** (3 条消息): 

- **自动化 Docker 构建修复即将到来**：宣布了针对 24.3 版本的修复，解决了 **automated docker builds** 的问题。

- **社区为 Docker 修复欢呼**：关于 24.3 版本中自动化 Docker 构建修复的公告受到了社区的积极响应。

- **分享了 Modular 认证示例**：一名成员提供了 GitHub 上 **modular authentication** 示例的链接，可以在 [这里](https://github.com/Moosems/Fastq_Parser/blob/main/.github/workflows/package.yml) 查看。
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1225150299598098484)** (277 条消息🔥🔥): 

- **探索条件一致性 (Conditional Conformance)**：成员们讨论了如何在 Mojo 中实现 [conditional conformance](https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md)，使用了受 Swift 和 Rust 启发的 `trait` 和 `struct` 语法。提议的解决方案包括使用 `@conditional` 注解来指示结构体中可选的 trait 实现。
- **External Call 字符串问题**：一位用户在向 `external_call` 传递字符串参数时遇到困难，因为 StringLiteral 是编译时且不可变的。建议使用从 Mojo 字符串中提取的 C 风格以 null 结尾的字符指针，类似于聊天中分享的示例。
- **在 Android 上运行 Mojo 程序**：一位用户展示了在 Android 上运行的 Mojo，具体是在 Snapdragon 685 CPU 上。这引起了大家的兴趣，并有人询问了 CPU 详情以及请求 `lscpu` 的输出。
- **周边商品可能性**：有人提出了关于未来是否会提供 Mojo 主题周边商品的问题，引发了团队成员探索这一想法的回应。Mojo 吉祥物的毛绒玩具和手机壳被提及作为潜在商品。
- **错误处理讨论**：用户推测了 Mojo 中错误处理的可能性，讨论了类似于 Python 传统 `try-except` 块的假设性错误处理语法和多态错误解析。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://docs.modular.com/mojo/stdlib/utils/variant">variant | Modular 文档</a>：定义了一个 Variant 类型。</li><li><a href="https://gist.github.com/lsh/f47fb85015d4197522d9c614e2a0f7de">一个可以是所有权 List 或与 Buffer 配合使用的 Bytes 类型</a>：一个可以是所有权 `List` 或与 `Buffer` 配合使用的 `Bytes` 类型 - bytes_ref_or_owned.mojo</li><li><a href="https://gist.github.com/modularbot/0613c95485ee838e00dc7289b81efa2c">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/modularbot/a6c43d73ec9532fb8a7fcf258f3c02ab">playground.mojo</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/src/utils/variant.mojo">mojo/stdlib/src/utils/variant.mojo at main · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2152">[BUG] 在装饰器中使用 inout 错误地导致了奇怪的编译器错误 · Issue #2152 · modularml/mojo</a>：Bug 描述 MRE：# 正确的实现应该是 &quot;fn decorator(borrowed func: fn() -&gt; None) -&gt; fn() escaping -&gt; None:&quot; fn decorator(inout func: fn() -&gt; None) -&gt; fn() es...</li><li><a href="https://github.com/modularml/mojo/issues/2144">[BUG] 最新 nightly 分支上测试失败 · Issue #2144 · modularml/mojo</a>：Bug 描述 我获取了 upstream/nightly 分支的最新更新并运行了测试，因为我想解决 1 个问题，但该分支上有 2 个测试失败。这是输出：Successfully crea...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1225176876079911101)** (3 条消息): 

- **Logger 库已更新**：**logger 库**已获得更新，现在允许使用任意参数和关键字参数记录消息。提供的示例展示了如何通过改进的函数调用记录信息、警告、错误和致命消息。

- **介绍 BlazeSeq**：`BlazeSeq🔥` 已发布，它是对 `MojoFastTrim` 的完整重写，作为一个功能完备的 FASTQ 解析器，匹配 BioJava 和 Biopython 的测试套件；它可用于 CLI 或作为未来应用的基础。基准测试和使用示例可在 [GitHub](https://github.com/MoSafi2/BlazeSeq) 上找到。

- **用于改进文件处理的缓冲行迭代器**：一个新实现包含了一个缓冲行迭代器，类似于 Rust 的 buffer_redux crate，能够处理来自文件或内存源的不完整行和大于缓冲区的行。该迭代器被推崇为在相关功能集成到标准库之前，项目的稳健解决方案。

**提到的链接**：<a href="https://github.com/MoSafi2/BlazeSeq">GitHub - MoSafi2/BlazeSeq</a>：通过在 GitHub 上创建账户为 MoSafi2/BlazeSeq 开发做出贡献。

  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1225086228442124418)** (11 条消息🔥): 

- **解锁新等级！**：一名用户因在 ModularBot 系统中晋升至 3 级而受到祝贺，这表明了其在社区内的参与和贡献。
- **库性能提升**：一位用户报告了使用某库带来的显著性能改进，执行时间降至 10m35s，而之前使用 Golang 实现的更快基准测试为 96s。
- **有用 Shell 脚本的链接**：所讨论库的作者分享了一篇 [Medium 文章](https://mzaks.medium.com/poor-persons-package-management-in-mojo-8671aa6e420a)，描述了一个用于轻松安装该库的 shell 脚本。
- **库优化技巧**：建议在实例化字典时设置容量，以减少重新分配和重新哈希，从而进一步优化性能。
- **排序算法仍待更新**：提到了一种专门用于字符串的排序算法，可能提供更好的性能，位于 [mzaks/mojo-sort](https://github.com/mzaks/mojo-sort/blob/main/multi_key_quicksort/sort.mojo)，但尚未针对新版本的 Mojo 进行更新。
  

---


**Modular (Mojo 🔥) ▷ #[📰︱newsletter](https://discord.com/channels/1087530497313357884/1157711273111339088/1225099690790355036)** (2 条消息):

- **Max⚡ 和 Mojo🔥 24.2 正式发布**：Modular 宣布发布 **Max⚡ 和 Mojo🔥 24.2**，同时开源了其标准库并推出了 nightly 构建版本。该更新获得了社区的积极参与，约有 50 个 pull requests 被开启，10 个已合并；鼓励贡献者在 [Discord](https://modul.ar/discord) 上进行探索和提问。
- **投身 Mojo🔥 开源运动**：一篇名为《[*The Next Big Step in Mojo🔥 Open Source*](https://www.modular.com/blog/the-next-big-step-in-mojo-open-source)》的新博文详细介绍了 Mojo🔥 开源计划的最新进展。
- **探索 Mojo🔥 24.2 的新特性**：正如 [Mojo 发布博客](https://www.modular.com/blog/max-24-2-is-here-whats-new)和后续文章《[*What’s new in Mojo 24.2*](https://www.modular.com/blog/whats-new-in-mojo-24-2-mojo-nightly-enhanced-python-interop-oss-stdlib-and-more)》中所述，Mojo🔥 24.2 的发布带来了增强的 Python 互操作性等功能。
- **探索 Mojo🔥 中的高阶函数**：读者受邀了解 Mojo🔥 中的高阶函数（Higher Order Functions），[Twitter](https://twitter.com/Modular/status/1) 上提供了一个预告链接。不过，该链接似乎不完整。

**提到的链接**：<a href="https://www.modular.com/newsletters/modverse-weekly-issue-28">Modverse Weekly - 第 28 期</a>：欢迎阅读第 28 期 Modverse 通讯，涵盖专题报道、Max 平台、Mojo 和社区活动。

  

---


**Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1225054381028675634)** (17 条消息🔥): 

- **标准库解析错误的解决**：一位在 **nightly** 分支上工作的成员报告了 stdlib 中的解析错误，但提到能够毫无问题地构建 stdlib。有人担心这是否值得警惕。
- **WSL 中的 FileCheck 问题已解决**：一位成员在运行测试时遇到了 `FileCheck command not found` 错误，但在社区的帮助下，通过使用 `dpkg -S llvm | grep FileCheck` 找到了正确的目录（`/usr/lib/llvm-14/bin`）并将其添加到路径中，问题得以解决。
- **无需担心不支持的测试**：在排查 `FileCheck` 安装问题后，该成员报告了 7 个不支持的测试，另一位成员确认这是正常的，因为这些测试是特定于平台的。
- **优化 Mojo 的 Optional Value 方法**：讨论了 Mojo 的 `Optional` 在 `value()` 方法中返回 Reference（引用）而非 copy（副本）的可能性，参考了[当前的实现](https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo)，并建议该改进是可行的。
- **引用相关问题对新贡献者的易上手性**：在考虑将“从 `Optional` 返回引用”作为一个“good first issue”时，成员们一致认为，对于不熟悉 lifetimes 的新贡献者来说，处理引用可能并不友好，因为正确的推断需要经验丰富的开发者对函数参数进行底层调整（plumbing）。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L117-L118).">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/src/collections/optional.mojo#L106.">mojo/stdlib/src/collections/optional.mojo at nightly · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1225003195835617280)** (171 条消息🔥🔥):

- **理解 LLM 多任务处理与扩展**：详细讨论揭示了使用单个 LLM 进行多任务处理可能会因为共享 VRAM 和 RAM 等资源而导致性能下降。建议通过在不同服务器上并发运行独立模型，并利用队列系统分发请求来获得更好的性能。
- **思考本地与云端 LLM 的使用**：参与者辩论了运行本地 LLM 与 GPT-4 等云端解决方案的优劣。一些人更倾向于本地模型，因为其输出未经审查，且能够利用强大的硬件而受云端限制。
- **为 AI 爱好者提供的模型建议**：多位用户推荐了用于编程和通用场景的特定模型，重点介绍了 *Hermes-2-Pro-Mistral-7B Q8* 和 *Goliath 120B Longlora Q3KS* 等。用户讨论了 VRAM 和系统配置如何影响不同 LLM 的性能和适用性。
- **技术问题与解决方案探讨**：成员们解决了常见错误，并提供了涉及 GPU offloading 设置和 C Redistributable 安装的解决方案。讨论明确了 LM Studio 无法执行网页搜索，且必须安装最新驱动程序以实现高效的 GPU 利用。
- **功能更新与社区参与**：宣布了 LM Studio 即将支持 text embeddings，同时有人询问关于运行多 GPU、通过 AnythingLLM 与文档交互以及创建具有上下文感知能力的 Discord 机器人等问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>：未找到描述</li><li><a href="https://useanything.com/">AnythingLLM | 终极 AI 商业智能工具</a>：AnythingLLM 是专为组织打造的终极企业级商业智能工具。为您的 LLM 提供无限控制、多用户支持、内外向工具等...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1225119288382197871)** (39 条消息🔥): 

- **System Prompt 训练可自动激活**：讨论了一个概念，即使用带有复杂 System Prompt 的大型模型生成的输出来训练较小的 LLM。这将有效地将 System Prompt 嵌入到较小模型中，从而无需为其占用上下文空间，尽管该过程在时间和金钱上可能成本较高。
- **模型响应中的难题**：提出了一个关于模型提供与输入查询无关的、奇怪的任务导向响应的问题。这表明预设行为存在混乱，可能与模型的训练有关。
- **Mixtral 与 Mistral 的区别澄清**：区分了 Mistral 和 Mixtral 模型；Mixtral 是一种混合专家模型 (MOE)，将 8 个 7B 模型组合成一个等效于 56B 参数的模型，而 Mistral 是标准的 7B 模型。
- **大模型，小硬件**：讨论了在拥有 24GB VRAM 的 NVIDIA 3090 GPU 上运行 Mixtral 8x7b 模型，指出虽然可行，但运行速度较慢。此外，LM Studio 无法在 Raspberry Pi 上运行，但像 tinyllama 这样的小型模型可能会被编译以在这些设备上运行。
- **新模型与支持开发讨论**：分享了具有先进能力的 104B 参数 C4AI Command R+ 模型和新推出的 Eurus-7b 模型的链接。还有讨论指出 llamacpp 需要更新以支持其中一些较新的模型。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/openbmb/Eurus-7b-kto">openbmb/Eurus-7b-kto · Hugging Face</a>：未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1bvniaz/command_r_cohere_for_ai_104b/ky12kw5/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>：未找到描述</li><li><a href="https://plainenglish.io/community/direct-preference-optimization-dpo-a-simplified-approach-to-fine-tuning-large-language-models">直接偏好优化 (DPO)：一种微调大型语言模型的简化方法</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/christopherthompson81/quant_exploration">christopherthompson81/quant_exploration · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/6491/files>">由 Carolinabanana 添加 Command R Plus 支持 · Pull Request #6491 · ggerganov/llama.cpp</a>：更新了张量映射，为 GGUF 转换添加了 Command R Plus 支持。
</li>
</ul>

</div>
  

---

**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1225000508557754408)** (8 条消息🔥): 

- **Embedding Models Not Supported, Until Now**：Embedding 模型此前不支持，直到现在：一位成员询问关于在 LM Studio 中使用 Embedding 模型的问题，并提到下载了一个 GGUF Embedding 模型。官方澄清，此前不支持 Embedding 模型，但 0.2.19 版本已引入文本 Embedding 支持，Beta 版可在[此处](https://discord.com/channels/1110598183144399058/1166577236325965844/1225221755937886208)获取。

- **LM Studio (Linux) Update Notification Issues**：LM Studio (Linux) 更新通知问题：一位用户报告 Linux 版 LM Studio 不会通知更新，观察到尽管运行的是 0.2.17 版本，但 0.2.18 已发布且 0.2.19 Beta 版也已存在。

- **Linux In-App Update Mechanism Still Pending**：Linux 应用内更新机制仍待完善：针对更新通知问题，官方强调 Linux 缺乏应用内更新机制是该平台仍被视为 "Beta" 版的原因之一。

- **Enthusiasm for Linux Development**：对 Linux 开发的热情：成员们对 LM Studio 在 Linux 上的开发表现出热情，包括提供 .deb 软件包的可能性。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1225045530199982172)** (23 条消息🔥): 

- **Switching to ROCm Yields Speed Boost**：切换到 ROCm 带来速度提升：利用 ROCm 预览版，AMD 硬件的速度从 13 tokens/秒显著提升至 65 tokens/秒，表明 AMD 系统在正确的软件接口下可以大幅超出预期。

- **GPU Market Fluctuations Noted**：观察到 GPU 市场波动：近期观察到 **GP100 GPU** 价格上涨，成本从约 350 美元升至 650-700 美元，预示着市场趋势的波动。

- **TSMC Disruption May Impact Prices**：TSMC 中断可能影响价格：彭博社一篇关于大地震导致 TSMC 生产线撤离的文章暗示 GPU 和 Mac 的价格可能会上涨。

- **CUDA vs. ROCM vs. OpenCL Performance Layers**：CUDA vs. ROCm vs. OpenCL 性能层级：据估计，NVIDIA CUDA 的速度大约是 ROCm 的两倍，而 ROCm 的速度大约是 OpenCL 或 DirectML 的五倍。

- **Mixed GPU Configurations for Inference**：推理中的混合 GPU 配置：虽然由于软件不兼容，无法在单一配置中混合使用 NVIDIA 和 AMD GPU，但运行多个 LM Studio 实例以分别利用每张显卡进行不同的推理任务是一个可行的解决方案。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-04-03/tsmc-evacuates-production-lines-after-major-taiwan-quake">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/186phti/m1m2m3_increase_vram_allocation_with_sudo_sysctl/">Reddit - Dive into anything</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1225221755937886208)** (19 条消息🔥): 

- **Introducing LM Studio 0.2.19 Preview 1 with Embeddings**：推出支持 Embeddings 的 LM Studio 0.2.19 Preview 1：LM Studio 0.2.19 Preview 1 现在支持**本地 Embedding 模型**，例如通过其类 OpenAI 的 `POST /v1/embeddings` 端点和 LLama.cpp 更新支持 `nomic-embed-text-v1.5-GGUF`。[Windows](https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-1.exe)、[Linux](https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage) 和 [Mac](https://releases.lmstudio.ai/mac/arm64/0.2.19/beta/LM-Studio-darwin-arm64-0.2.19-Preview-1.zip) 预览版构建已开放下载。
  
- **Separate ROCm Version for Compatibility**：兼容性所需的独立 ROCm 版本：对于需要 **ROCm 支持**的用户，将提供一个独立版本；当前构建中不包含该支持。

- **Beta Version Confusion Clarified**：Beta 版本混淆澄清：LM Studio Beta 构建中显示的版本反映的是当前正式发布的版本，而非 Beta 迭代版本，为了清晰起见，版本号仅在正式发布时才会增加。

- **No Support for GPU over IP Yet**：尚不支持 GPU over IP：LM Studio 目前不支持跨不同机器使用多个 GPU，即 **GPU over IP**。

- **Chat Feature with Documents Still Pending**：文档对话功能仍待上线：LM Studio 尚未实现“与文档对话”的功能，但建议使用 LM Studio 服务器模式配合 AnythingLLM 作为替代方案。
<div class="linksMentioned">

<strong>提及的链接</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/tree/main">nomic-ai/nomic-embed-text-v1.5-GGUF at main</a>: 未找到描述</li><li><a href="https://lmstudio.ai/docs/welcome">Welcome | LM Studio</a>: LM Studio 是一款用于在电脑上运行本地 LLM 的桌面应用程序。</li><li><a href="https://blog.nomic.ai/posts/nomic-embed-text-v1">Introducing Nomic Embed: A Truly Open Embedding Model</a>: Nomic 发布了一款序列长度为 8192 的文本嵌入器（Text Embedder），其性能优于 OpenAI 的 text-embedding-ada-002 和 text-embedding-v3-small。</li><li><a href="https://huggingface.co/nomic-ai/nomic-embed-text-v1">nomic-ai/nomic-embed-text-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://releases.lmstudio.ai/windows/0.2.19/beta/LM-Studio-0.2.19-Setup-Preview-1.exe">no title found</a>: 未找到描述</li><li><a href="https://releases.lmstudio.ai/linux/0.2.19/beta/LM_Studio-0.2.19-Preview-1.AppImage">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1225532162145517568)** (1 条消息): 

- **Autogen Studio 输出被截断**：一位成员报告称，在将 LM Studio 与 Autogen Studio 配合使用时，推理结果仅包含 1 或 2 个 token，正在寻求获取完整补全响应的解决方案。
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1225408139508322384)** (1 条消息): 

- **关于运行时“记忆”保留的问题**：一位成员询问如何在同一运行时内实现“记忆”保留，目前仅在文件分析中成功实现。在如何跨 Bot 交互维持状态方面存在理解空白。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1225450306993786910)** (9 条消息🔥): 

- **LM Studio 在 AMD GPU 上激活**：一位成员详细介绍了在同时包含 RTX 2060 和 7900 XTX GPU 的系统中，让 **LM Studio** 在 **AMD GPU** 上运行的经验。
- **ROCm 与 OpenCL 性能咨询**：一位参与者询问了 **ROCm 和 OpenCL** 之间的速度差异，并提到尽管进行了配置尝试，但仍无法在 **6700XT** GPU 上加载模型。
- **共享 ROCm 版本的系统规格**：一位成员分享了他们的系统规格，显示使用了 **AmdROCm** GPU 类型，并指出在 Windows 10 平台上拥有 15.94 GB RAM，其中 11.86 GB VRAM 未被使用。
- **驱动问题导致 ROCm 无法在低系列 AMD GPU 上运行**：据提到，AMD 的驱动问题导致 **ROCm 版本无法在 6700 系列或更低型号的 GPU 上运行**，这表明问题的解决取决于 AMD 的干预。
- **ROCm 在其他平台上的性能优于 OpenCL**：一位成员详细介绍了他们使用 **KoboldAI** 的 ROCm 分支的正面体验，观察到与 **LMStudio + OpenCL** 相比，性能从 12T/s **显著提升**至 33T/s。
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1225180539972227167)** (24 条消息🔥): 

- **CORS 可能有所帮助**：一位成员建议开启 **CORS**（跨源资源共享）以可能解决某个问题，但未说明具体针对的是什么问题。
- **LM Studio 实现资源**：建议查看一篇关于在 CrewAI 中实现 LM Studio 的文章，标题为“Implementing LM Studio in CrewAI”，可在 [Tayyib Ali 的 Medium](https://medium.com/@tayyibali4300/implementing-lm-studio-in-crewai-270cc577acee) 上阅读。
- **CrewAI 日志级别和显示问题**：一位成员讨论了 CrewAI 的日志功能，提到可以将 **verbose** 设置为 **1 或 2** 以获得不同级别的日志详情，并对 LM Studio 预期位置未出现日志表示担忧。
- **排查 LM Studio 日志缺失问题**：在关于 LM Studio 缺失日志的排错讨论中，一位成员指出他们在 LM Studio 中没有看到任何处理过程，但确认 CrewAI 在其端运行正常。
- **CrewAI 中的 JSONDecodeError**：一位成员在使用 CrewAI 时遇到了“**json.decoder.JSONDecodeError**”并寻求帮助；该错误表明 JSON 字符串未正确终止。
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1225015134947250258)** (194 条消息🔥🔥):

- **Transformers 课程公开**：斯坦福大学关于 Transformers 的 CS 25 研讨会现已向公众开放，可以实时旁听或观看录播。讨论的主题包括 LLM 架构及其在各个领域的应用，并由知名行业专家授课。[在 Zoom 上加入](https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09)，查看[课程网站](https://web.stanford.edu/class/cs25/)，或在 [YouTube](https://www.youtube.com/playlist?list=PLoROMvodv4rNiJRchCzutFw5ItR_Z27CM) 上观看往期课程。

- **来自一线的面试技巧**：资深工程师建议，对于高级职位，应侧重于高层级技能和工作信心，而非单纯的编码测试。为了评估基础 Python 技能，有些人甚至使用简单的编程练习，以确保候选人不会过度依赖 ChatGPT 等工具。

- **数学难题**：一位成员寻求帮助，以理解一篇非公开工作论文中的数学问题，引发了关于集合二分查找以及在学术论文中定义变量重要性的讨论。

- **斯坦福课程启动**：斯坦福大学的一门 Transformers 课程现已通过 Zoom 向公众开放，该课程邀请了客座研究员并涵盖了深度学习模型，同时还开设了相应的 [Discord 服务器](https://discord.gg/2vE7gbsjzA) 供更广泛的社区讨论。

- **来下围棋**：成员们交换了用户名和链接来玩围棋（Go）。选项包括用于通信对弈的 Online Go Server (OGS) 以及在 [Infinite Go](https://infinite-go.com) 上提供的自定义版本。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://blog.eleuther.ai/nyt-yi-34b-response/">Yi-34B, Llama 2, and common practices in LLM training: a fact check of the New York Times</a>：关于 Yi-34B 和 Llama 2 的澄清。</li><li><a href="https://infinite-go.com">Infinite Go</a>：未找到描述</li><li><a href="https://x.com/DanHendrycks/status/1769452537302929682?s=20">Dan Hendrycks (@DanHendrycks) 的推文</a>：https://x.ai/blog/grok-os Grok-1 已开源。发布 Grok-1 增加了 LLM 在社会中的扩散速度。民主化访问有助于我们更好地应对该技术的影响...</li><li><a href="https://www.regulations.gov/comment/NTIA-2023-0009-0246">Regulations.gov</a>：未找到描述</li><li><a href="https://www.regulations.gov/document/NTIA-2023-0009-0001/comment">Regulations.gov</a>：未找到描述</li><li><a href="https://github.com/EleutherAI/the-pile/issues/75">Legal Contracts · Issue #75 · EleutherAI/the-pile</a>：这里是从证券交易委员会收集的法律合同。https://drive.google.com/file/d/1of37X0hAhECQ3BN_004D8gm6V88tgZaB/view?usp=sharing 原始大小约为 38 GB，包含...</li><li><a href="https://stanford.zoom.us/j/99922151759?pwd=dW5CcUtVYkNybGZGY0hMWUZtVkZBZz09).">加入我们的云高清视频会议</a>：Zoom 是现代企业视频通信的领导者，拥有易于使用、可靠的云平台，可用于跨移动设备、桌面和会议室系统的视频和音频会议、聊天和网络研讨会。</li><li><a href="https://www.youtube.com/watch?v=XfpMkf4rD6E&ab_channel=StanfordOnline)">Stanford CS25: V2 I Introduction to Transformers w/ Andrej Karpathy</a>：2023 年 1 月 10 日，Transformers 简介。Andrej Karpathy：https://karpathy.ai/。自 2017 年推出以来，Transformers 彻底改变了自然语言处理...</li><li><a href="https://discord.gg/2vE7gbsjzA)">Discord | 你的聊天与聚会场所</a>：Discord 是通过语音、视频和文本进行交流的最简单方式。在这里交谈、聊天、聚会，并与你的朋友和社区保持联系。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1225089176609882203)** (51 条消息🔥): 

- **高效模型架构创新**：一种名为 T-GATE 的新方法表明，在理解图像的粗略语义后，文本到图像（text-to-image）扩散模型中的交叉注意力（cross-attention）可能是多余的，这有望加快处理速度（[GitHub 上的 T-GATE](https://github.com/HaozheLiu-ST/T-GATE)）。然而，提供的样本尚未完全说服社区其有效性。
  
- **硬件优化突破还是徒劳？**：关于潜在硬件改进的参考资料（如 FFIP 算法）声称具有显著的效率提升，将一半的乘法转换为廉价的加法（[期刊发表](https://arxiv.org/abs/2311.12224)）。社区对此持怀疑态度，思考这些看似好得令人难以置信的说法是否存在猫腻。

- **Dynamic Allocation of FLOPs in Transformers**：一篇 [arXiv 论文](https://arxiv.org/abs/2404.02258) 介绍了一种让 Transformer 在序列中动态分配计算资源的方法，这可能优化性能并允许预定义的计算预算。这种方法不同于均匀的 FLOPs 分布，提出了一种更具选择性且可能更高效的资源分配方式。

- **Discussions on Large Language Models**：关于大规模语言模型 (HLB-GPT) 的对话探讨了 Mixture of Experts (MoE) 工作的后续研究以及具体的设计选择。一个专门的线程 ([HLB-GPT MoE and MoD Thread](https://discord.com/channels/729741769192767510/1169741769232089169/1225497424869724180)) 用于详细交流，以避免干扰主频道。

- **Contentious Data Crawling Practices**：讨论中出现了与抓取 Discord 等平台相关的挑战和潜在违规行为。虽然理论上可行，但这违反了服务条款 (Terms of Service)，并可能导致账号被封禁。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“次尺度预测”或“次分辨率预测”...</li><li><a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算资源）分配给特定的...</li><li><a href="https://arxiv.org/abs/2404.01475">Are large language models superhuman chemists?</a>：大语言模型 (LLM) 因其处理人类语言和执行未经显式训练的任务的能力而受到广泛关注。这与化学领域相关...</li><li><a href="https://x.com/cem__anil/status/1775282571070591220?s=20">Cem Anil (@cem__anil) 的推文</a>：我们最清晰的发现之一是，上下文学习 (in-context learning) 通常遵循作为演示次数函数的简单幂律。我们很惊讶没有发现这一点被明确阐述...</li><li><a href="https://arxiv.org/abs/2311.12224">Fast Inner-Product Algorithms and Architectures for Deep Neural Network Accelerators</a>：我们引入了一种名为自由流水线快速内积 (FFIP) 的新算法及其硬件架构，改进了 Winograd 提出的一种尚未被充分探索的快速内积算法 (FIP)...</li><li><a href="https://www.youtube.com/watch?v=rJIwO31uv5c">Louis Castricato - RLAIF, User Autonomy, and Controllability (Eleuther / Synthlabs)</a>：来自康奈尔科技学院开源生成式 AI 研讨会的演讲。网站：https://www.louiscastricato.com/ 幻灯片：https://drive.google.com/file/d/14Qldg0E1c...</li><li><a href="https://github.com/HaozheLiu-ST/T-GATE/">GitHub - HaozheLiu-ST/T-GATE: T-GATE: Cross-Attention Makes Inference Cumbersome in Text-to-Image Diffusion Models</a>：T-GATE：交叉注意力使文本到图像扩散模型中的推理变得繁琐 - HaozheLiu-ST/T-GATE</li><li><a href="https://github.com/trevorpogue/algebraic-nnhw">GitHub - trevorpogue/algebraic-nnhw: AI acceleration using matrix multiplication with half the multiplications</a>：通过乘法次数减半的矩阵乘法实现 AI 加速 - trevorpogue/algebraic-nnhw
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1225093201749086392)** (7 条消息): 

- **Countdown to MATS Stream Applications**：申请 Neel Nanda 的 MATS stream 的截止日期还有不到 10 天。感兴趣的申请人可以在提供的 [Google Docs 链接](https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn)中找到详情和常见问题解答。

- **Attention to Neural Networks**：分享了一个名为 **atp_star** 的 GitHub 仓库，它提供了 AtP* 的 PyTorch 和 NNsight 实现，该实现源自 Kramar 等人 2024 年的 DeepMind 论文。该仓库可以在 GitHub 上的 [koayon/atp_star](https://github.com/koayon/atp_star) 找到。

- **Saprmarks Tweets**：一位成员分享了 [@saprmarks 的 Twitter 帖子](https://twitter.com/saprmarks/status/1775513423402692685)链接，但在提供的消息中未讨论其内容。

- **Gratitude for Sharing Code**：在提供了 GitHub 仓库链接后，关于最新 AtP* 论文开源实现的查询已得到解决并表示感谢。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://docs.google.com/document/d/1p-ggQV3vVWIQuCccXEl1fD0thJOgXimlbBpGk6FI32I/edit#heading=h.y0ohi6l5z9qn">Neel Nanda MATS Stream - 录取程序 + FAQ</a>：未找到描述</li><li><a href="https://github.com/koayon/atp_star">GitHub - koayon/atp_star: AtP* 的 PyTorch 和 NNsight 实现 (Kramar et al 2024, DeepMind)</a>：AtP* 的 PyTorch 和 NNsight 实现 (Kramar et al 2024, DeepMind) - koayon/atp_star
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1225065760783859712)** (17 条消息🔥): 

- **CUDA 错误排查**：一名成员在 H100 上运行旧版 LM eval harness 时遇到了 **CUDA 运行时错误 (RuntimeError)**，而该版本在 A100 上可以运行，这指向了 `flash attention` 的潜在问题。一些建议指出升级到 **CUDA 11.8** 可能有所帮助，但真正的罪魁祸首被确定为 `apex`。通过使用 `.contiguous()` 函数的隔离测试以及转向单 GPU 解决了该问题。

- **Colab 中 top_p 参数无法识别**：另一名成员在 **Google Colab** 中尝试在 LM eval harness 命令中设置 **`top_p=1`** 时遇到了参数无法识别的错误。建议指出该问题可能是由于参数列表中的空格导致的。

**提到的链接**：<a href="https://colab.research.google.com/drive/1pDByKcCu3vQzy58iz8uSmUm806LQtG8v#scrollTo=mTSKBJlVjaB-">Google Colaboratory</a>：未找到描述

  

---


**Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1225121548147490837)** (3 条消息): 

- **PyTorch 中的容错与弹性作业启动**：一位用户分享了 PyTorch 文档链接，用于设置 **容错和弹性作业 (fault-tolerant and elastic jobs)**，并详细说明了启动这些作业所需的命令。该过程涉及对节点、每个节点的训练器数量、最大重启次数以及 rendezvous 端点的特定设置，如 [PyTorch 弹性训练快速入门指南](https://pytorch.org/docs/stable/elastic/quickstart.html) 所示。

- **高级训练方案的云支持**：另一位成员提到，像 **AWS** 和 **Azure** 这样的云服务支持高级作业训练方案，AWS 在去年发布了一个名为 **Gemini** 的产品。

**提到的链接**：<a href="https://pytorch.org/docs/stable/elastic/quickstart.html">快速入门 &mdash; PyTorch 2.2 文档</a>：未找到描述

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1225016603159302245)** (158 条消息🔥🔥): 

- **苏格拉底式导师与宪法 AI (Constitutional AI)**：提到了一个名为 [ConstitutionalAiTuning](https://github.com/steffen74/ConstitutionalAiTuning) 的包，它允许将 LLM 微调为遵循个人伦理原则的苏格拉底式导师。它需要一个包含原则的 JSON 文件，并利用这些原则构建改进后的回答以进行模型微调，旨在为技术背景较少的人员简化流程。
- **JAX 类型提升与语义澄清**：关于 [JAX 类型提升语义 (type promotion semantics)](https://jax.readthedocs.io/en/latest/type_promotion.html) 的讨论围绕 JAX 在操作过程中如何进行类型提升展开。代码片段展示了这种行为，例如 `np.int16(1) + jnp.int16(2) + 3` 的结果是 `int16`，而 `3 + np.int16(1) + jnp.int16(2)` 的结果是 `int32`。
- **SD3 模型输入配置辩论**：针对 SD3 等模型的文本输入设置进行了广泛的技术讨论，建议采用替代方法来连接序列，并讨论了在微调期间扩展 T5 token 同时限制 CLIP 使用的潜在好处。
- **AI 与版权侵权的法律风险**：一段对话强调了使用受版权保护的材料训练 AI 系统的法律风险，提到了 Suno 音乐 AI 平台以及来自唱片公司可能的法律后果。
- **GPU 基础设施成本与 Stability AI 的财务挑战**：讨论了 Stability AI 面临的财务挑战，包括他们难以承担来自云服务的高额基础设施成本以及可能无法支付这些费用的情况，正如 [Forbes 的一篇揭秘文章](https://www.forbes.com/sites/kenrickcai/2024/03/29/how-stability-ais-founder-tanked-his-billion-dollar-startup/?sh=2e53d2e3e630) 所述。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.01292">Measuring Style Similarity in Diffusion Models</a>: 生成模型现在被图形设计师和艺术家广泛使用。之前的研究表明，这些模型在生成过程中会记住并经常复制其训练数据中的内容。因此……</li><li><a href="https://www.weco.ai/blog/technical-report">Introducing Weco AIDE</a>: 您的机器学习 AI Agent</li><li><a href="https://www.972mag.com/lavender-ai-israeli-army-gaza/">‘Lavender’: The AI machine directing Israel’s bombing spree in Gaza</a>: 以色列军队利用一个几乎没有人类监督且对伤亡政策宽松的 AI 目标系统，将数万名加沙人标记为暗杀嫌疑人，+972 和 Local C...</li><li><a href="https://www.theregister.com/2024/04/03/stability_ai_bills/">Stability AI reportedly ran out of cash to pay its AWS bills</a>: 据报道，这家生成式 AI 宠儿正面临支付 9900 万美元计算费用却仅产生 1100 万美元收入的局面</li><li><a href="https://tenor.com/8a9w.gif">Ian Malcolm GIF - Ian Malcolm Jurassic - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://www.youtube.com/watch?v=_D3GACF-Bsk">Galileo</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=kJirMpbvBrM">Editing DALL·E Images in ChatGPT</a>: 您现在可以在网页端、iOS 和 Android 端的 ChatGPT 中编辑 DALL·E 图像。</li><li><a href="https://www.musicbusinessworldwide.com/suno-is-a-music-ai-company-aiming-to-generate-120-billion-per-year-newton-rex/">Suno is a music AI company aiming to generate $120 billion per year. But is it trained on copyrighted recordings? &#x2d; Music Business Worldwide</a>: Ed Newton&#x2d;Rex 发现 Suno 创作的音乐与经典版权作品有着惊人的相似之处……</li><li><a href="https://www.youtube.com/watch?v=5pidokakU4I">Axis of Awesome - 4 Four Chord Song (with song titles)</a>: 澳大利亚喜剧团体 'Axis Of Awesome' 在 2009 年墨尔本国际喜剧节上表演的一段短剧。视频由 Network Ten Australia 提供。...</li><li><a href="https://github.com/steffen74/ConstitutionalAiTuning/">GitHub - steffen74/ConstitutionalAiTuning: A Python library for fine-tuning LLMs with self-defined ethical or contextual alignment, leveraging constitutional AI principles as proposed by Anthropic. Streamlines the process of prompt generation, model interaction, and fine-tuning for more responsible AI development.</a>: 一个用于微调 LLMs 的 Python 库，具有自定义的伦理或上下文对齐功能，利用了 Anthropic 提出的 Constitutional AI 原则。简化了 Prompt 生成、模型交互和微调的过程，以实现更负责任的 AI 开发。
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1225099630627258398)** (10 messages🔥): 

- **扩展 Latent Diffusion Models (LDMs)**: 一篇 [arXiv 论文](https://arxiv.org/abs/2404.01367) 详细研究了 LDMs 的采样效率扩展特性。研究发现，在相同的推理预算下，较小的模型通常优于较大的模型。
- **分享了审核相关的 GIF**: 一名成员发布了一个来自 Tenor.com 的 [审核相关 GIF](https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646)，可能表示对离题或不当消息采取了行动。
- **关于快钱的玩笑**: 用户们开玩笑说因为消息被审核而错过了学习如何“在 72 小时内赚 5 万美元”的机会，并对毒品走私进行了猜测和梗引用。
- **新优化器的预告**: Drhead 分享了一条 [Twitter 帖子](https://twitter.com/aaron_defazio/status/1775521495298588956)，暗示即将发布一个新的 Optimizer。
- **Visual AutoRegressive (VAR) 模型表现更优**: 一篇 [arXiv 论文](https://arxiv.org/abs/2404.02905) 介绍了 VAR，这是一种新的图像自回归建模范式，已证明在图像生成的多个维度（包括质量和速度）上优于 Diffusion Transformers。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>: 我们提出了视觉自回归建模 (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率...”</li><li><a href="https://arxiv.org/abs/2404.01367">Bigger is not Always Better: Scaling Properties of Latent Diffusion Models</a>: 我们研究了潜在扩散模型 (LDMs) 的缩放特性，重点关注其采样效率。虽然改进的网络架构和推理算法已被证明能有效...</li><li><a href="https://tenor.com/view/discord-mod-moderation-ban-mod-ban-gif-9351874248631360646">Discord Mod Moderation Ban GIF - Discord mod Moderation ban Mod ban - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1225055189980020776)** (67 messages🔥🔥): 

- **探索多样化的 AI 数据集**：一位成员列出了他们拥有的几个数据集，大小从 "pairs" 的 106G 到 "PheMT" 的 994k 不等，其中几个数据集涉及欧盟语言的翻译。一些数据集（如 "pairs" 和 "WikiMatrix"）被认为不太可靠，需要指标和截断点来进行质量评估。

- **RP-LLMs 的快速反馈**：[Chaiverse](https://console.chaiverse.com/) 提供的一项新服务允许对 RP-LLM 模型进行快速反馈，在 15 分钟内提供模型评估。它旨在利用人类偏好提供最快、最准确的反馈，并通过非公开评估数据集避免“针对测试进行训练”。

- **为 AI/ML 工作负载揭秘 SaladCloud**：SaladCloud 承诺通过提供全托管的容器服务，帮助开发者避免高昂的云成本和 GPU 短缺，该服务开放了数千个消费级 GPU 的访问权限，价格低至 $0.00/hr，专为大规模推理而构建。

- **让为 Transformer 模型添加 Head 变得更容易**：分享了 [transformer-heads 的 GitHub 仓库](https://github.com/center-for-humans-and-machines/transformer-heads)，该仓库提供了用于为 Transformer 模型附加、训练、保存和加载新 Head 的工具，这对于那些希望扩展 LLM 能力的人来说非常有益。

- **CohereForAI 的巨型模型 C4AI Command R+**：创作者发布了一个名为 [C4AI Command R+](https://huggingface.co/CohereForAI/c4ai-command-r-plus) 的模型，这是一个拥有 1040 亿参数的多语言模型，具备检索增强生成 (RAG) 和处理复杂任务的多步工具使用能力。运行此类大型模型的成本仍然是一些成员关注的问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: 暂无描述</li><li><a href="https://bit.ly/3TFIsKt">Salad - GPU Cloud | 10k+ GPUs for Generative AI</a>: 节省高达 90% 的云账单。轻松部署 AI/ML 生产模型。每美元可获得多出 600% 的图像和 10 倍的推理。立即免费试用 SaladCloud。</li><li><a href="https://github.com/center-for-humans-and-machines/transformer-heads">GitHub - center-for-humans-and-machines/transformer-heads: Toolkit for attaching, training, saving and loading of new heads for transformer models</a>: 用于为 Transformer 模型附加、训练、保存和加载新 Head 的工具包 - center-for-humans-and-machines/transformer-heads</li><li><a href="https://github.com/OpenNLPLab/LASP/tree/main">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>: 线性注意力序列并行 (LASP)。通过在 GitHub 上创建账号为 OpenNLPLab/LASP 开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1225159571052695584)** (4 messages): 

- **GitHub Bug 已修复**：应用了一个修复程序来解决某个问题，并将提交推送到 GitHub，可在 [GitHub Commit 5760099](https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a) 查看。

- **目录不匹配警报**：观察到 **README 的目录 (Table of Contents)** 与其 Markdown 标题不符，表明需要进行清理。

- **为了清晰起见进行对比分析**：建议将当前的目录和 Markdown 标题并排查看，以便更好地发现不一致之处。

- **训练配置中的标题错误**：识别出 `config/train` 中使用的标题存在问题，指出其不正确并提出了潜在的修正建议。

**提及的链接**：<a href="https://github.com/OpenAccess-AI-Collective/axolotl/commit/5760099bd4605e4c4fb444890bd473cb200c5f1a">fix toc · OpenAccess-AI-Collective/axolotl@5760099</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1225164644633542818)** (11 条消息🔥): 

- **寻找高分辨率图像**：一名成员询问在哪里可以找到大量可供抓取的 **4K 和 8K 图像** 集合，但在随后的讨论中没有提供任何来源或建议。
- **需要带有 UI 反馈的部署方案**：有人征求关于**用于部署模型并获取专家反馈的优质 UI** 的建议，但该线程中未提出任何建议。
- **探索用于训练的非指令文本数据**：一名成员讨论了使用播客转录等**非指令文本数据**来训练模型，使其生成符合训练数据风格的文本，并参考了 *MistralAI*，询问其他人是否在进行类似的实验。
- **微调实践的顺序**：在一次策略讨论中，大家达成共识，在微调时应该在“指令（instructions）”之前训练 **'completion'**，这对于增加模型的特定领域知识特别有用。
- **微调技术与效率**：在关于微调技术的交流中，成员们指出，对于特定领域的训练，有时**简单微调 (SFT)** 和提示工程（prompt engineering）比持续预训练 (CPT) 更有效。有人提到，指令样本的质量和多样性（即使数量较少）通常比数量更多但质量较低的数据能带来更好的性能。

---

**OpenAccess AI Collective (axolotl) ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/1225511933306736722)** (2 条消息): 

- **Mistral 7B 的最佳数据集**：一名成员询问在 Ubuntu 22.04 上使用 axolotl 训练 **Mistral 7B 模型** 的推荐数据集。另一名成员建议使用 **OpenOrca 数据集**，因为它在全方位用途中都很有用。

---

**OpenAccess AI Collective (axolotl) ▷ #[announcements](https://discord.com/channels/1104757954588196865/1113462842436354149/1225531833970856049)** (1 条消息): 

- **新的 Discord Bot 集成上线！**：一个新的 Discord Bot 集成已设置完毕，可以直接回答来自 OpenAccess AI Collective 的问题。鼓励成员们测试该 Bot 并在指定频道留下反馈。

---

**OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1225300453051469834)** (62 条消息🔥🔥): 

- **使用 Qlora 微调 Qwen2**：一份详细的回答阐明了使用 **Qlora** 微调 **Qwen2** 的步骤，例如在配置文件中设置 `base_model` 和 `adapter`，使用 4-bit 精度，并指定优化器设置。提供了一个示例配置文件以协助该过程。
- **Axolotl 中的数据集流式传输**：Axolotl 支持使用**本地数据集进行流式传输 (streaming)**，这与之前可能暗示相反的文档理解不同。步骤包括在 `.yml` 文件中配置 `pretraining_dataset` 并指向 Hugging Face 数据集路径。
- **使用 Docker 进行多节点微调**：提出了使用 Docker 进行**多节点微调**的指南，例如设置 **accelerate** 配置、在模型上配置 FSDP 设置，并确保所有机器共享相同的 Axolotl commit 和模型配置文件。
- **Checkpoints 与混合精度的问题**：一名成员在 **Mixtral** 模型上同时使用 **Qlora** 和 **FSDP** 时，遇到了尝试展平具有不同数据类型的张量时的 `ValueError`。解决方案涉及在操作前确保张量的数据类型统一。
- **Axolotl Bot 离线**：**Axolotl bot** 经历了宕机，导致成员们通过幽默的回复表达不满。聊天记录中未提供停机的原因或解决方案。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/patrickpain-patricksomuchpain-patrickfleas-spongebobpain-spongebobsomuchpain-gif-18151897">Patrickpain Patricksomuchpain GIF - Patrickpain Patricksomuchpain Patrickfleas - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=a2fc4740-1a5c-4766-8cbb-7769186bae94)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=f8d0cb5a-e9cd-4dcf-a16f-39197690a56b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://github.com/openaccess-ai-collective/axolotl#dataset)">GitHub - OpenAccess-AI-Collective/axolotl: Go ahead and axolotl questions</a>: 尽管提问（axolotl questions）。通过在 GitHub 上创建账号，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=8b13862e-c141-4ebd-973a-e8f61032dce3)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=1608c74f-8ed6-4f25-8861-c69c9ff61737)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=7db2702b-b0e3-424e-af79-012c04808de0)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=d8e13d9b-7b9a-45e1-8c8d-ebad9a63158a)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=46b832c9-3b42-4a74-9886-711b4821502f)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。</li><li><a href="https://phorm.ai/query?projectId=43cf5c5b-941f-461a-9055-b02788a2e364&threadId=a31dec35-31c9-4260-bc7f-1d79610360aa)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>: 更快地理解代码。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[announcements](https://discord.com/channels/1059199217496772688/1073670729054294197/)** (1 条消息): 

jerryjliu0: 网络研讨会将在 15 分钟后开始！^^
  

---


**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1225096488317620346)** (6 条消息): 

- **革新您的知识管理**：全新的 **LLM 驱动、自组织的数字图书馆**不仅仅是一个聊天系统；它是一个专为专业人士和团队设计的 AI 驱动工具，用于创建、组织和注释他们的数据。点击[这里](https://t.co/nbvRS0Cc9Q)探索。

- **东京高级 RAG 见面会**：参加 **4/18 日本标准时间 (JST) 晚上 7-9 点**在东京举行的闪电演讲之夜，演讲嘉宾包括 @hexpode、Diego 和 Sudev，他们将讨论 RAG 应用，活动由 Rakuten 主办。详情和报名请见[这里](https://t.co/ovCozxNaTt)。

- **轻松在全球部署 LLM 应用**：Koyeb 的界面通过连接您的 GitHub 仓库，在零基础设施设置的情况下全球部署 serverless 应用，从而便捷地扩展 LLM 应用。点击[这里](https://t.co/weFs0waN4o)查看 Koyeb。

- **根据问题复杂度定制 RAG**：@SoyeongJeong97 的 "Adaptive RAG" 论文探讨了针对不同复杂度问题的定制化 RAG 技术，解决了速度与特异性之间的权衡。点击[这里](https://t.co/SZQppddC95)了解更多。

- **使用 LlamaIndex + MistralAI Cookbook 进行烹饪式编码**：探索系列 Cookbook，指导用户使用 **MistralAI** 构建 RAG、agentic RAG 和基于 agent 的系统，包括路由和查询分解。点击[这里](https://t.co/7KCqujf9sd)获取您的食谱。

**提到的链接**：<a href="https://t.co/nbvRS0Cc9Q">IKI AI – Intelligent Knowledge Interface</a>：面向专业人士和团队的智能图书馆和知识助手。

  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1225055713068585071)** (112 条消息🔥🔥):

- **探索 GraphIndex 的局限性**：一位成员对在 **llama_index** 中处理 **knowledgegraphs** 时缺乏 pipeline 支持表示困惑，指出关于从 `graphdb` 创建 `graphindex` 或 `docstore` 的作用缺乏清晰的文档。他们注意到，虽然 vectorindex 拥有用于重新索引节点的 pipeline 和 docstore，但 **graphindex** 似乎需要自定义代码。
- **寻求递归查询引擎文档**：一位成员找不到关于 **ragas** 配合递归查询引擎（recursive query engine）的文档，引发了关于 **langchain** 与 **ragas** 之间潜在问题以及从 **ragas.metrics** 导入函数困难的讨论。
- **查询现有的 OpenSearch 索引**：一位 LlamaIndex 新成员询问了如何查询现有的 **OpenSearch index**。他们提供了设置客户端和存储库的详细步骤，但对流程不确定，随后自行发现了 `VectorStoreIndex.from_vector_store` 方法。
- **寻找 LlamaIndex Agent 示例**：参与者讨论了创建 **llama_index agents** 的各个方面，包括生成深度响应的复杂性、持久化节点耗时异常长的问题，以及 ReAct agents 的正确用法。
- **处理 LlamaIndex 实现中的问题**：成员们就一系列 **llama_index** 实现话题寻求建议，包括在元数据中进行语义相似度匹配的可能性、集成 SQL 数据库以实现聊天机器人功能，以及在使用 **elastic search** 作为 vector db 存储时遇到的异步操作错误。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://llamahub.ai/l/tools/llama-index-tools-bing-search?from=">未找到标题</a>：未找到描述</li><li><a href="https://llamahub.ai/?tab=llama_datasets">Llama Hub</a>：未找到描述</li><li><a href="https://www.llamaindex.ai/blog/introducing-llama-datasets-aadb9994ad9e">介绍 Llama Datasets 🦙📝 — LlamaIndex，LLM 应用的数据框架</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型（LLMs）。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llama_dataset/uploading_llama_dataset/?h=dataset">向 LlamaHub 贡献 LlamaDataset - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/?h=similarity#similaritypostprocessor">节点后处理器模块 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/index_structs/struct_indices/SQLIndexDemo/">Text-to-SQL 指南（查询引擎 + 检索器） - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/evaluation/dataset_generation/?h=from_documents#llama_index.core.evaluation.DatasetGenerator.from_documents">数据集生成 - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/readers/simple_directory_reader/?h=simpledirector#llama_index.core.readers.file.base.SimpleDirectoryReader">简单目录读取器 - LlamaIndex</a>：未找到描述</li><li><a href="https://youtu.be/yGejxO1xYmo?si=22UtE4T0RVXbqYOy">创建可信 AI 的工作流与工具 | 与 Clara Shih 一起向 AI 提问更多</a>：Clara 与三家最热门 AI 公司的创始人/CEO 坐下来交谈——Aravind Srinivas (Perplexity AI)、Jerry Liu (LlamaIndex) 和 Harrison Chase (LangChain)...</li><li><a href="https://github.com/run-llama/llama_index/blob/f03db8da9301e2a1f2a1783338464bec7e7a859e/llama-index-legacy/llama_index/legacy/agent/react/prompts.py#L27">run-llama/llama_index 中的 llama_index/llama-index-legacy/llama_index/legacy/agent/react/prompts.py</a>：LlamaIndex 是适用于你 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/issues/905#issuecomment-1484288684">我在哪里定义向量库相似度搜索返回的 top_k 文档？ · Issue #905 · run-llama/llama_index</a>：调用查询函数时，如何指定我希望检索器传递给 LLM 的 k 值是多少？或者我需要在调用查询函数之前指定它吗？llm_predictor = LLMPredictor(llm=ChatOp...</li><li><a href="https://github.com/run-llama/llama-hub/">GitHub - run-llama/llama-hub: 由社区制作的 LLM 数据加载器库 —— 配合 LlamaIndex 和/或 LangChain 使用</a>：由社区制作的 LLM 数据加载器库 —— 配合 LlamaIndex 和/或 LangChain 使用 - run-llama/llama-hub
</li>
</ul>

</div>
  

---


**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1225126740515623075)** (6 条消息):

- **AI 方式的拼写检查**：一位成员分享了一段 Node.js 代码，利用 LlamaIndex 的 `Ollama` 软件包，通过名为 ‘mistral’ 的模型来纠正用户提交文本中的拼写错误。他们指出该服务可以在本地运行并处理错误，脚本演示了将 "bkie" 纠正为 "bike"，尽管 prompt 中讽刺地将 "misspelled" 给拼错了。
- **无需第三方服务的本地 AI**：同一位用户确认 `Ollama` 软件包是本地运行的 AI 服务器的客户端/封装器，建议通过 `ollama run mistral` 命令在 `localhost:11434` 上进行本地操作。
- **感谢仁慈的 AI**：一位成员在自己的代码示例中自报拼写错误事件后，幽默地称赞了 AI 的宽容本性，赞扬了 AI 理解并正确处理预期输入的能力。
- **通过 Reading and Asking (RAG) 增强图像处理**：讨论围绕将 Reading and Asking Generative (RAG) 技术用于图像处理任务的潜力展开，其实际应用包括破解 CAPTCHAs 或在连环画等视觉叙事中保持连贯性。
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1225114862640959720)** (3 messages): 

- **自定义您的仓库可见性**：使用 HuggingFace 的企业用户现在可以将默认的 **Repo visibility** 设置为公开、私有或默认私有。更多详情可以在此 [Twitter 线程](https://twitter.com/julien_c/status/1772688542289822073)中找到。
- **在 HuggingFace 上使用 Quarto 发布**：**Quarto** 推出了新的发布选项，使用户能够轻松地在 HuggingFace 上部署网站。发布指南请见[此处](https://twitter.com/gshotwell/status/1772661727856914720)和[此处](https://www.linkedin.com/feed/update/urn:li:activity:7178422723503673345?updateEntityUrn=urn%3Ali%3Afs_feedUpdate%3A%28V2%2Curn%3Ali%3Aactivity%3A7178422723503673345%29)。
- **HuggingFace Hub 企业版页面上线**：探索全新的 **HF Hub Enterprise** 页面，这是一个提供量身定制的企业解决方案的地方。公告和详情请见[此处](https://x.com/victormustar/status/1772742275744850137)。
- **企业仓库的精细化访问控制**：通过新的精细化访问控制功能，更好地控制您组织的仓库。详情可见此 [Twitter 帖子](https://twitter.com/Thom_Wolf/status/1770504033452573077)。
- **Major TOM 接入 Sentinel-1**：Major TOM 的扩展现在包括 MajorTOM-Core 中的 **Sentinel-1** 数据，拓展了空间观测能力的视野。了解更多关于该版本的信息请点击[此处](https://x.com/mikonvergence/status/1772912287709331612)。
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1225005073986490408)** (48 messages🔥): 

- **寻找用于游戏测试的 AI**：一位成员询问是否有适合测试游戏的机器学习 AI，暗示对适用于游戏开发和质量保证的工具感兴趣。
- **生成简洁摘要**：一位用户在使用 Hugging Face 的 **summarization pipeline** 时遇到困难，注意到 `text_length_penalty` 似乎无效，而 `max_length` 似乎只是截断了文本。讨论围绕模型输出长度以及如何实现更短的摘要展开，建议包括使用 `max_new_tokens`、检查模型配置或拆分输入样本。
- **多 GPU 系统设置故障排除**：有人咨询 **PCIe 插槽速度 (x4/x8)** 对本地大语言模型 (LLMs) 多 GPU 系统性能的影响。
- **部署和使用 HuggingFace 模型**：关于部署模型以及在涉及 **AWS Inferentia Instance** 的模型部署中使用 `predict` 函数的咨询，寻求关于正确方法以及此论坛是否为讨论此类问题的正确场所的澄清。
- **图像生成模型微调建议**：有人寻求关于微调图像生成模型以创建特定风格肖像画的建议，并询问加入绘画图像是否有助于实现这一目标；有人建议尝试使用 **IP adapter Face ID**。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://t.me/Jttfoxoffcial1">JTT FOX OFFICIAL</a>：您可以立即联系 @Jttfoxoffcial1。</li><li><a href="https://huggingface.co/docs/transformers/main/en/generation_strategies#customize-text-generation">Text generation strategies</a>：未找到描述</li><li><a href="https://github.com/huggingface/cookbook">GitHub - huggingface/cookbook: Open-source AI cookbook</a>：开源 AI 食谱。通过在 GitHub 上创建账号，为 huggingface/cookbook 的开发做出贡献。</li><li><a href="https://github.com/unslothai/unsloth#installation-instructions---conda">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>：速度快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1225412463906787429)** (1 条消息): 

- **速度与智能之间的平衡**：一位成员强调了生产环境 Prompt 中**延迟与推理之间的权衡**，指出没有推理的 Prompt 响应快但质量差，而加入推理则响应更智能但更慢。他们提出了一个技巧：在用户输入时，预先对最可能的场景进行推理。[在此探索该想法](https://x.com/siddish_/status/1772345589511901368?s=20)。

**提及的链接**：<a href="https://x.com/siddish_/status/1772345589511901368?s=20">来自 Siddish (@siddish_) 的推文</a>：不带推理的流式传输 -> 愚蠢的响应 🥴 带推理的流式传输 -> 响应缓慢 😴 一个 LLM 小技巧：在用户花时间输入时，主动推理最可能的场景。

  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1225158243765063801)** (5 条消息): 

- **Apple 展示技术实力**：一条消息提到 Apple 声称其最新模型比 **OpenAI 的 GPT-4** 更强大。
- **3blue1brown 依然是数学视频大师**：一位成员对 3blue1brown 持续制作教育视频表示赞赏，特别是早期的神经网络系列。
- **Visual AutoRegressive Modeling 优于 Diffusion Transformers**：一篇新论文 [Visual AutoRegressive modeling (VAR)](https://arxiv.org/abs/2404.02905) 引入了图像自回归学习的范式转变，在图像生成质量和推理速度方面超越了 Diffusion Transformers。
- **Chain of Thoughts 提升 AI 推理能力**：论文 [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) 讨论了在使用 Chain of Thought Prompting 时，LLM 在复杂推理任务中的显著性能提升。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>：我们提出了 Visual AutoRegressive modeling (VAR)，这是一种新的生成范式，它将图像上的自回归学习重新定义为从粗到细的“下一尺度预测”或“下一分辨率...”</li><li><a href="https://arxiv.org/abs/2201.11903">Chain-of-Thought Prompting Elicits Reasoning in Large Language Models</a>：我们探讨了生成思维链（一系列中间推理步骤）如何显著提高 LLM 执行复杂推理的能力。特别是，我们...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1225112064494927942)** (20 条消息🔥):

- **Octopus 2: The Tentacles of Functionality**: 一个关于 **Octopus 2** 的 Demo 已被分享，该模型具备调用函数的能力，其在设备端（on-device）的潜力引发了热烈讨论。[查看 Octopus 2 的 Space](https://huggingface.co/spaces/Tonic/Octopus)，但请注意尝试时可能需要长达 1500 秒的渲染时间。
- **通过本地处理克服音乐障碍**: 成员们讨论了在本地而非云服务上运行音乐模型的优势。他们探讨了对硬件优化的预期，并分享了一个 [Youtube Demo](https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb) 来庆祝一次成功的流水线实验。
- **使用 Salt AI 让图像栩栩如生**: 利用 Salt 新推出的 **multi-subject image node pack**（多主体图像节点包），一系列创新工作流已经发布，包括身体和面部区域检测以及换脸技术。[在 GitHub 上了解更多关于多主体图像处理的信息](https://github.com/getSaltAi/SaltAI_Multisubject)。
- **在 TED 舞台分享 AI 的影响**: 一位社区成员分享了一段 TED 演讲，重点介绍了社区参与和 AI 的进步。演讲可以在 [YouTube](https://www.youtube.com/watch?v=d8icTgtZeQg&t) 上观看，表达了对电影制作过程中社区支持的感谢。
- **PyTorch Geometric 迎来 CornellTemporalHyperGraphDataset**: CornellTemporalHyperGraphDataset 的 Pull Request 已成功合并到 PyTorch Geometric 中，可以通过从 `master` 分支下载立即访问。[在此查看 PR](https://github.com/pyg-team/pytorch_geometric/pull/9090)，准备将其整合到你的工作流中。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.producthunt.com/posts/metaforms-ai"> Metaforms AI - OpenAI + Typeform = 用于反馈、调查和研究的 AI | Product Hunt</a>: Metaforms 是 Typeform 的 AI 继任者。构建全球最强大的反馈、调查和用户研究表单，通过生成式 AI 收集关于用户的改变生活的见解。训练于...</li><li><a href="https://huggingface.co/spaces/Tonic/Octopus/">Octopus - Tonic 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://telegram.me/int_gem_bot">Int Bot</a>: 你可以立即联系 @int_gem_bot。</li><li><a href="https://github.com/getSaltAi/SaltAI_Multisubject">GitHub - getSaltAi/SaltAI_Multisubject</a>: 通过在 GitHub 上创建账户来为 getSaltAi/SaltAI_Multisubject 的开发做出贡献。</li><li><a href="https://youtube.com/shorts/Jm2xq2oNJ3E?si=MGkXSq0ZCiGM0gbb">没人写过的歌 #music #newmusic #song #timelapse #photography #musicvideo #viral #art</a>: 未找到描述</li><li><a href="https://github.com/pyg-team/pytorch_geometric/pull/9090">feat: 由 SauravMaheshkar 添加 `CornellTemporalHyperGraphDatasets` · Pull Request #9090 · pyg-team/pytorch_geometric</a>: 参考: #8501 #7312 评审请求: @rusty1s @wsad1 此 PR 旨在添加由带时间戳的单纯形组成的超图数据集，其中每个单纯形是一组节点。随论文发布 ...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1225205399297327244)** (5 条消息): 

- **MLE 面试准备的 RAG 资源**: 一位成员正在为技术面试寻找深入学习 **Retrieval-Augmented Generation (RAG)** 的资源。他们请求社区推荐优质的学习材料。

- **在 WSL Ubuntu 上配置 RAG 的困扰**: 一位 AI 新手寻求在 **WSL Ubuntu 24.04** 上使用 **Llama2** 配置 **RAG** 的帮助，并提到了在配置 **privategpt** 时遇到的困难。

- **录制下次演示以供参考**: 一位社区成员无法参加下次演示，寻求帮助进行录制。他们打算将链接放在 GitHub 上以便将来访问。

- **潜在的 OBS 录制方案**: 针对录制请求，另一位成员表示他们正在考虑使用 **OBS** 录制演示。
  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1225092089998606467)** (8 条消息🔥):

- **Batch Size 与模型性能**：较大的 Batch Size 与模型性能的提升相关，特定测试（尤其是医疗数据）显示了改进，尽管这种改进可能是微小或不显著的。然而，超过 2 个 Batch 的累加可能会对性能产生负面影响，这可能是由于 Batch Normalization 的问题导致的。
- **寻找深度学习伙伴**：一位成员提到他们正在寻找合作伙伴，共同在 **Deep Learning** 和 **Natural Language Processing** (NLP) 领域进行协作。
- **Batch Size 对学习动态的影响**：成员们分享了关于 Batch Size 的不同经验；一位成员发现较小的 Batch 对他们的小模型效果更好，而另一位成员提出了较大 Batch 可能会跳过局部最小值（local minima），但较小 Batch 更耗时的问题。
- **Learning Rate (LR) Schedulers 作为解决方案**：建议使用 **LR schedulers**（如 cyclic 或 cosine）来解决在使用较大 Batch Size 时遇到的局部最小值问题，在训练过程中平衡探索（exploration）与利用（exploitation）阶段。
- **关于在 HuggingFace 上更新自定义数据集的问题**：一位成员询问，手动更新用于在 HuggingFace 上 fine-tuning 预训练模型的自定义数据集，是否需要重新上传，还是模型会自动更新。
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1225029720949653544)** (21 messages🔥): 

- **GPT-2 在摘要任务上停滞不前**：一位用户在为文本摘要任务训练 GPT-2 时遇到了验证指标停滞的问题。他们建议 HuggingFace 平台为特定任务提供简洁的模型训练示例，以避免在互联网上四处搜寻。

- **针对 CPU 上 LLM 的 Prompt 构建**：一位成员正在寻找一个开源的 Large Language Model，能够从 HTML 中提取产品名称和价格等结构化数据，并询问合适的 Prompt。该用户指定使用 16GB RAM 的纯 CPU 环境来部署模型。

- **用 BERT 进行时间序列预测？**：有人对使用 PEFT 等方法对 BERT 进行 fine-tuning 以进行时间序列预测感兴趣。当被索要代码示例或 Notebook 以指导此过程时，一位用户提供了帮助。

- **模型 Fine-tuning 中的上下文长度限制**：关于 Babbage-002 模型的上下文长度是否可以在训练期间更改的咨询得到了解释：在 fine-tuning 期间它是不可变的，但在从头开始训练时可以修改。

- **使用免费模型增强聊天机器人回复**：一位正在开发集成 Google Books API 的聊天机器人的用户正在寻找免费的语言模型来提升回复质量，确保回答更具对话性且更完整。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1225105529903251536)** (11 messages🔥): 

- **寻找带有 Cross-Attention 的 DiT**：一位成员询问是否有经过修改、使用 cross-attention 来对文本、图像或其他数据类型进行条件控制的 **DiT (Diffusion Transformer)**。另一位成员提到 HF Diffusers 上的 **DiT** 是 class-conditioned（类别条件化）的，并链接了论文 ([DiT Paper](https://arxiv.org/html/2312.04557v1))。
- **条件策略中的成本考量**：对话强调，像 **DiT** 这样的公开扩散模型是按类别条件化的，而不是使用 cross-attention，以保持较低的成本。一位成员建议，与 *SD3 linear* 相关的修改可能更实用。
- **定制 SD 用于立体图像到深度图的转换**：一位成员表示需要将立体图像转换为深度图，发现现有模型效果不足。他们提议可能需要为此任务修改 **Stable Diffusion (SD)**。
- **带有自定义通道的 SD Fine-Tuning 限制**：关于对超过 3 个通道的 **Stable Diffusion** 进行 fine-tuning 的咨询引出了一项建议：可能需要对 **SD architecture** 进行微调，而不是从头开始训练。
- **深度估计的替代方法**：建议研究 **Dino v2** 进行深度估计训练，并考虑将 **LoRA** 用于立体图像，同时分享了相关的 GitHub 资源 ([Dino v2 GitHub](https://github.com/facebookresearch/dinov2), [Depth Estimation Notebook](https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb))。另一位成员指出了 **ControlNet** 的相关工作，其中使用了 4 通道图像，并链接到了相关仓库 ([SD-Forge-LayerDiffuse GitHub](https://github.com/layerdiffusion/sd-forge-layerdiffuse))。
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1225087752924696637)** (86 messages🔥🔥):

- **Logo 升级与告别小鱼**：Discord Logo 由 George Hotz 进行了更新，这在成员中引发了复杂的情绪——一些人喜欢这种专业感，而另一些人则为失去搞怪的小鱼 Logo 感到惋惜（该 Logo 仍保留在横幅上）。随后讨论了是否也要更新横幅。
  
- **优化与跨 GPU 通信**：话题转向机器学习模型的优化和分片（sharding），George Hotz 等人讨论了启动延迟对小算子（small kernels）性能的影响，以及 GPU 之间数据传输的挑战——包括 cudagraphs、P2P 限制，以及使用 NV 驱动程序可能带来的改进。
  
- **Tinygrad 性能雄心**：分享了性能测试结果，显示出令人期待的前景，例如在单张 **4090 GPU** 上使用 BEAM=4 达到 **53.4 tok/s**，达到了 gpt-fast 性能的 83%。George Hotz 强调了很快要用 tinygrad 超越这些结果的雄心。

- **Intel GPU 与 NPU 内核驱动**：讨论了关于 Intel GPU 和 NPU 内核驱动的技术细节，提到了多种可用驱动，如 'gpu/drm/i915'、'gpu/drm/xe' 和 'accel/ivpu'。还交流了在 CPU 配合下利用 NPU 可能带来的性能和能效提升。

- **坚持 Tinygrad 开发重点**：在技术讨论中，George Hotz 重申了该频道用于 tinygrad 相关讨论的目的，并提供了 tinygrad GitHub 仓库链接以及一份提问指南。这强化了在频道内保持主题讨论的目标。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/tinygrad/tinygrad/blob/master/docs/env_vars.md">tinygrad/docs/env_vars.md at master · tinygrad/tinygrad</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/tinygrad/tinygrad">GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</a>：你喜欢 pytorch？你喜欢 micrograd？你会爱上 tinygrad！❤️ - GitHub - tinygrad/tinygrad: You like pytorch? You like micrograd? You love tinygrad! ❤️</li><li><a href="http://www.catb.org/~esr/faqs/smart-questions.html">提问的智慧 (How To Ask Questions The Smart Way)</a>：暂无描述
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1225058022502568027)** (23 messages🔥): 

- **Tinygrad 教程赢得赞誉**：用户发现 Tinygrad 的快速入门指南非常直观，并称赞其对初学者的帮助；这激发了他们进一步深入研究神经网络领域的动力。
- **JAX 教程受到关注**：一位成员分享了 JAX Autodidax 教程的链接，通过 [动手实践的 Colab notebook](https://colab.research.google.com/github/google/jax/blob/main/docs/autodidax.ipynb) 深入探讨了 JAX 核心系统的运作方式。
- **Tinygrad 用于蛋白质折叠的咨询**：Camelcasecam 讨论了使用 Tinygrad 实现 ColabFold 或 OmegaFold 的可能性，询问潜在的性能提升，同时也表现出学习如何将 PyTorch 权重迁移到 Tinygrad 的兴趣。
- **生物领域技术的协作努力**：在将 OmegaFold 适配到 Tinygrad 的背景下，具有生物科学背景的用户表达了组队开展该项目的热情，认为协作可以产生更好的结果。
- **使用 Tinygrad 探索性能调试**：Alveoli3358 分享了他们在运行 DEBUG=2 的 Tinygrad 时解读性能输出的学习笔记，表示有兴趣计算 MNIST 示例所需的总 FLOPS/内存，以估算理论训练时间。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/profiling.md">tinygrad-notes/profiling.md at main · mesozoic-egg/tinygrad-notes</a>：tinygrad 教程。通过在 GitHub 上创建一个账户来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://jax.readthedocs.io/en/latest/autodidax.html">Autodidax: 从零开始实现 JAX 核心 — JAX 文档</a>：暂无描述
</li>
</ul>

</div>
  

---



**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1225030105441501194)** (83 messages🔥🔥): 

- **JSON 对象支持说明**：用户确认支持 'json_object' 响应格式的模型主要是 OpenAI 和 Fireworks 的端点。他们建议通过查看模型页面上的提供商参数来检查支持情况 ([OpenRouter models](https://openrouter.ai/models))。

- **Claude 3 Haiku 的 Roleplaying 疑虑**：Claude 3 Haiku 模型在 Roleplay 方面的评价褒贬不一，建议使用 Self-moderated 版本并输入多个示例 (Few-shot) 以获得更好的输出。然而，为了提高性能，建议进行 Jailbreak (jb) 调整。

- **Discord 上的 Claude Jailbreaking 资源**：用户讨论了 Claude Jailbreaks 并分享了资源，包括 SillyTavern 和 Chub 的 Discord 服务器，在那里可以找到 Jailbreak 列表和 NSFW 提示词。用户被引导至易于访问的 Jailbreaks（如 pancatstack jb），并获得了关于如何获取 NSFW 角色的建议。

- **OpenRouter Credits 位置与模型问题**：成员们讨论了 OpenRouter Dashboard 的最新变化，包括查看 Credits 的新位置，现在位于 `/credits` URL。此外，还对 DBRX 和 Midnight Rose 等某些模型的功能及其对特定特性的支持表示了担忧。

- **OpenRouter API 的审核与响应问题**：用户注意到即使是 Self-moderated 版本的 Claude 模型也有很高的拒绝率，并推测存在额外的“安全”提示词。有报告称响应不及时，并提到通过引入更好的提供商来提高 Midnight Rose 等模型的服务稳定性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://prnt.sc]">未找到标题</a>：未找到描述</li><li><a href="https://prnt.sc/_ba2eY63AJNA">Screenshot</a>：使用 Lightshot 捕获</li><li><a href="https://sillytavern.app/">SillyTavern - 面向高级用户的 LLM 前端</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1225036657842196580)** (17 messages🔥): 

- **安装成功**：一名成员在 Windows PC 上成功安装软件后表示满意：*刚刚在我的 Windows PC 上安装并运行了。太棒了*。
- **Termux 故障**：讨论了 `chroma-hnswlib` 的障碍；一名成员指出，尽管据称已被移除，但该问题在安装过程中仍然存在。他们寻求处理此问题的建议。
- **转移到支持频道**：针对上述问题，讨论被引导至另一个频道，建议将详细的技术支持话题移至更合适的位置。
- **支持与鼓励**：社区内就发帖进行了相互鼓励和感谢，重点在于相信提出的每个问题都是宝贵的学习经验。
- **确认多平台能力**：澄清了某些软件的兼容性，确认其在 PC 和 Mac 上均可运行，并参考了文档和置顶消息中的安装说明及指南。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://discord.gg/wNJZsJgQ?event=1221828294811586572">Discord - 与好友和社区聊天的新方式</a>：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的好友和社区保持紧密联系。</li><li><a href="https://docs.openinterpreter.com/getting-started/setup">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1225019016196784208)** (55 messages🔥🔥): 

- **Hermes-2-Pro 最佳实践**：成员们正在讨论 **Hermes-2-Pro** 的使用，以及按照模型卡建议更改系统提示词的重要性。
- **01 服务器中的快捷键烦恼**：一名用户表示 01 软件的输出过于冗长，希望在 Ollama 中能有类似 `ctrl+c` 的本地键盘快捷键，以便在不退出整个服务器的情况下中断 LLM 输出。
- **01 软件在 Linux 上的复杂情况**：用户分享了在各种 Linux 发行版上运行 01 软件时的故障排除和变通方法。提到了包依赖、系统消息以及硬件兼容性（如音频 ALSA lib 错误）等问题。
- **发现 Windows 11 上的 Poetry 问题**：一名用户报告在使用 Windows 11 上的 `Poetry` 时遇到问题，指出 `CTRL+C` 和音频录制存在故障。
- **Cardputer 讨论与开发**：参与者讨论了将 *M5 Cardputer* 用于 OpenInterpreter 项目的开发和潜力，包括实现细节以及正在进行的 GitHub 仓库链接。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenInterpreter/01/issues/219">Ubuntu 21+ is not supported [wayland] · Issue #219 · OpenInterpreter/01</a>: 一些依赖项使用 x11，与 wayland 不兼容 https://github.com/Kalmat/PyWinCtl?tab=readme-ov-file#linux-notice https://github.com/asweigart/pyautogui/issues?q=is%3Aissue+is%3Aopen...</li><li><a href="https://github.com/Clinteastman/c0mputer">GitHub - Clinteastman/c0mputer: Porting open-interpreter to the M5 Cardputer</a>: 将 open-interpreter 移植到 M5 Cardputer。通过在 GitHub 上创建账号来为 Clinteastman/c0mputer 的开发做出贡献。</li><li><a href="https://github.com/m5stack/M5Unified/tree/develop">GitHub - m5stack/M5Unified at develop</a>: M5Stack 系列的统一库。通过在 GitHub 上创建账号来为 m5stack/M5Unified 的开发做出贡献。</li><li><a href="https://ngrok.com/docs/getting-started/?os=linux">Quickstart | ngrok documentation</a>: 本快速入门将使用 ngrok agent 来部署您的应用程序</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper: A fast, local neural text to speech system</a>: 一个快速、本地的神经文本转语音系统。通过在 GitHub 上创建账号来为 rhasspy/piper 的开发做出贡献。</li><li><a href="https://dashboard.ngrok.com/get-started/setup/linux">ngrok - Online in One Line</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1225438495749967913)** (37 messages🔥): 

- **Command R+ 发布**: Command R+ 正式亮相，这是一款针对企业级应用优化的高度可扩展 LLM，具有先进的 RAG 功能以减少幻觉、支持多语言并改进了 Tool Use。它拥有 128k tokens 的上下文窗口，模型权重已在 [Cohere 平台](https://txt.cohere.com/command-r-plus-microsoft-azure/) 发布供研究使用。
  
- **Command R+ 备受关注**: 新的 Command R+ 模型拥有 104B 参数并展示了 RAG 能力，由于缺乏对比数据，其相对于其他模型的表现引发了疑问，目前已有 [实时 Demo](https://huggingface.co/spaces/CohereForAI/c4ai-command-r-plus) 可供实验。

- **审视 ChatGPT 企业版**: 人们对类 ChatGPT 模型在商业应用中的有效性持怀疑态度，强调真正的企业级应用可能需要高度定制的解决方案，超出了目前“企业定制”模型所能提供的范围。

- **模型评估面临挑战**: 讨论涉及评估 Command R+ 等模型的复杂性和潜在偏见，强调了像 AssistantBench 这样结构化基准测试对于更透明评估的重要性。

- **JetMoE-8B：学术界的高性价比里程碑**: 成本低于 10 万美元，且在推理过程中仅需 2.2B 激活参数，其性能便超过了 Meta AI 的 LLaMA2。JetMoE-8B 代表了学术研究中高性价比且易于获取的 LLM 的重要一步，详见其 [项目页面](https://research.myshell.ai/jetmoe)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://research.myshell.ai/jetmoe">JetMoE</a>: 未找到描述</li><li><a href="https://txt.cohere.com/command-r-plus-microsoft-azure/">Introducing Command R+: A Scalable LLM Built for Business</a>: Command R+ 是一款最先进的 RAG 优化模型，旨在处理企业级工作负载，并首先在 Microsoft Azure 上可用。今天，我们推出了 Command R+，我们最强大的...</li><li><a href="https://huggingface.co/CohereForAI/c4ai-command-r-plus">CohereForAI/c4ai-command-r-plus · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/jetmoe/jetmoe-8b">jetmoe/jetmoe-8b · Hugging Face</a>: 未找到描述</li><li><a href="https://openai.com/blog/openai-partners-with-scale-to-provide-support-for-enterprises-fine-tuning-models">OpenAI partners with Scale to provide support for enterprises fine-tuning models</a>: OpenAI 的客户可以利用 Scale 的 AI 专业知识来定制我们最先进的模型。</li><li><a href="https://fxtwitter.com/aidangomez/status/1775878606108979495?s=46">Tweet from Aidan Gomez (@aidangomez)</a>: ⌘R+ 欢迎 Command R+，我们专注于可扩展性、RAG 和 Tool Use 的最新模型。与上次一样，我们将发布研究用途的权重，希望它们对大家有用！https:/...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1225458505583034408)** (3 messages): 

- **Nathan 挑起话题**: Nathan Lambert 在一条 [推文](https://twitter.com/natolambert/status/1775899591814300024) 中暗示了争议，称“希望这不会演变成一场 Drama...”

- **点名 Snorkel**：继 Nathan 的评论之后，一条回复要求对 **Snorkel** 发表犀利见解，认为这是一个值得讨论的话题。

- **“所有模型都很差”文章预告**：Nathan Lambert 预告了一篇题为“all these models are bad”的即将发表的文章，似乎在批评当前的模型，包括那些集成了 RLHF (Reinforcement Learning from Human Feedback) 的模型。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1225175436900962397)** (21 messages🔥): 

- **《金融时报》的锁定宝藏**：一位成员分享了 [Financial Times](https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1) 的无限文章访问优惠，暗示需要付费订阅才能获取高质量新闻。链接中包含挂锁和产品图标，象征着锁定内容，暗示了内容访问与数字产品供应之间的对比。
- **对商业模式的怀疑**：在简短的交流中，一位成员表示担心传统的商业模式可能会阻碍 "genAI" 的成功，暗示现有运营中可能存在的僵化以及他们所谓的“产品自杀”前景。
- **技术政治讨论并不讨喜**：成员们分享了知名人士 Ben Horowitz 和 Marc Andreessen 关于 [技术政治讨论的链接](https://x.com/pmarca/status/1775691027363639634?s=20)，但反响平平，评论从勉强愿意听取政治见解到完全否定其内容价值不等。
- **CS25 课程讲座专题**：对话透露，被认作 **nato** 的成员将为 CS25 课程进行讲座，表达了对这一承诺的期待和物流方面的考虑。
- **斯坦福大学的 CS25 课程吸引 AI 爱好者**：分享了斯坦福大学 CS25 课程的细节，显示了与 Transformer 研究专家（包括泰斗和行业专业人士）的一系列深入讨论。感兴趣的人士被引导至 [时间表](https://web.stanford.edu/class/cs25/#schedule)，并被敦促关注该课程的 YouTube 频道以获取更多见解。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/pmarca/status/1775691027363639634?s=20">来自 Marc Andreessen 🇺🇸 (@pmarca) 的推文</a>：请观看并享受！Ben @bhorowitz 和我花了两个小时讨论华盛顿特区及其他地区的技术政治和政策。回答了许多 X 上的问题并阐述了观点。🇺🇸🚀💪</li><li><a href="https://web.stanford.edu/class/cs25/#schedule">CS25: Tranformers United!</a>：讨论 Transformer 在不同领域的最新突破</li><li><a href="https://www.ft.com/content/2f4bfeb4-6579-4819-9f5f-b3a46ff59ed1">谷歌考虑对 AI 驱动的搜索收费，这是商业模式的重大转变</a>：未找到描述
</li>
</ul>

</div>
  

---



**Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1225003496080937002)** (57 messages🔥🔥): 

- **内核可扩展性突破**：一位成员讨论了改进 matmul 内核以高效处理大矩阵的 Prompt，克服了当矩阵大小超过 1024x1024 时 CPU 缓存的限制。
- **实现编译器魔法**：分享了让编译器转换代码的喜悦，这据信带来了性能提升。
- **Llamafile 的 ROCm 版本说明**：对于 Windows 上的 llamafile-0.7，需要 **ROCm 5.7+**，这表明已经考虑了对不同版本 ROCm（包括 5.7 和 6.0.2）的支持。
- **SYCL 代码传奇继续**：关于如何在 llamafile 中处理 SYCL 代码的激烈讨论，引出了查看 `llamafile/metal.c` 和 `llamafile/cuda.c` 以进行 DSO 动态加载的建议。一位社区成员通过实现 SYCL 支持的条件编译做出了贡献，使其能在 Windows 和 Linux 上运行，但不能在 Mac 上运行。
- **Llamafile 性能及 Windows 上 Cosmopolitan 的问题**：一位成员尝试在 Windows 上构建 llamafile，但遇到了 Cosmopolitan 编译器的问题。分享了一篇强调 llamafile 性能提升的文章，并讨论了对 `llamafile-bench` 程序进行每秒 Token 数基准测试的需求。建议更多的 RAM 和更快的 RAM 可以提高性能，因为 CPU 或内存限制尚未被确定为瓶颈。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.theregister.com/2024/04/03/llamafile_performance_gains/">Llamafile LLM 驱动程序项目提升了 CPU 核心性能</a>：狠狠地压榨 LLaMA 的性能</li><li><a href="https://rocm.docs.amd.com/projects/install-on-windows/en/latest/how-to/install.html">安装 HIP SDK — HIP SDK Windows 安装</a>：未找到描述</li><li><a href="https://huggingface.co/models?library=gguf">Models - Hugging Face</a>：未找到描述</li><li><a href="https://github.com/jart/cosmopolitan/issues/1010">execve() 应该在 Windows 上对 #! 进行 polyfill · Issue #1010 · jart/cosmopolitan</a>：复制自 bellard/quickjs#197：#!/bin/qjs console.log("Hello"); 当从 bash 作为脚本调用时不起作用：$ ./test.qjs ./test.qjs: line 2: syntax error near unexpected token `&...
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1225043803778449528)** (36 条消息🔥): 

- **寻找加密货币领域的经验丰富聊天机器人开发者**：一位用户正在寻找在训练 LLM 以及将其与包含加密货币市场新闻和信息的实时数据库集成方面有经验的开发者，旨在构建一个类人聊天机器人。
- **从 PDF 中提取数学符号**：一位用户询问除了 MathpixPDFLoader 之外，是否有其他从 PDF 文件中提取数学符号的替代方案，更倾向于能处理此任务的其他方法。
- **寻求 LangChain 社区联系**：一位用户正在寻找 LangChain 的社区经理或开发者倡导者（Developer Advocate），以寻求集成方面的帮助，并获得了一个贡献集成的链接：[贡献集成指南](https://python.langchain.com/docs/contributing/integrations)。
- **关于在 JS 中使用 Langchain 进行机器人开发的初学者咨询**：一位刚开始在 JavaScript 中使用 Langchain 的用户正在寻求关于创建预约安排并与数据库交互的机器人的指导。资深用户推荐了 Sequelize（一个 Node.js 的 ORM）用于数据库交互，并附带了链接：[Sequelize GitHub 仓库](https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a)。
- **LCEL 链式调用难题与探讨**：成员们讨论了 LCEL（LangChain Expression Language）中 '|' 运算符的用途，该运算符用于链接 Prompt 和 LLM 输出等组件。提供了一个进一步阅读的链接：[LCEL 入门](https://python.langchain.com/docs/expression_language/get_started)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/docs/expression_language/get_started">入门 | 🦜️🔗 Langchain</a>：LCEL 使得从基础组件构建复杂链变得容易，并且</li><li><a href="https://python.langchain.com/docs/contributing/integrations">贡献集成 | 🦜️🔗 Langchain</a>：开始之前，请确保你已具备贡献代码指南中列出的所有依赖项。</li><li><a href="https://github.com/langchain-ai/langchain/discussions/19957">何时使用 Outputparsers、tools 和/或 LangSmith Evaluators 来测试 LLM 输出？ · langchain-ai/langchain · Discussion #19957</a>：我正在为一个简单的任务开发一个简单的 LCEL 链，脑海中浮现了这个问题。想象一下我有一个包含 2 个 Prompt 和 2 个 Output Parser 的简单 LCEL 链，用于“强制”...</li><li><a href="https://github.com/brianc/node-postgres/tree/master">GitHub - brianc/node-postgres: 适用于 node.js 的 PostgreSQL 客户端。</a>：适用于 node.js 的 PostgreSQL 客户端。通过创建账户为 brianc/node-postgres 的开发做出贡献。</li><li><a href="https://github.com/sequelize/sequelize/tree/9e141880230a7f2a9a8c1e66a31f29fea7b5a65a">GitHub - sequelize/sequelize 在 9e141880230a7f2a9a8c1e66a31f29fea7b5a65a 版本</a>：适用于现代 Node.js 和 TypeScript 的功能丰富的 ORM，支持 PostgreSQL（支持 JSON 和 JSONB）、MySQL、MariaDB、SQLite、MS SQL Server、Snowflake、Oracle DB (v6)、DB2 以及 DB2 for IBM i。 - ...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1225024321806667787)** (2 条消息): 

- **CI 失败难题**：一位成员就一个 PR 中的 **持续集成（CI）失败** 寻求帮助，该 PR 旨在即使在使用嵌套 API 路由的情况下，也能从正确的路由提供 Playground 服务。涉及的 PR 是 [从正确的路由提供 Playground PR #580](https://github.com/langchain-ai/langserve/pull/580)，本地测试已通过 Python 3.10。

- **聊天 Playground 演示**：分享了一个教程视频，展示了如何将 **Agent 与 Langserve 的新聊天 Playground** 结合使用。详细的演示（包括处理初始困难和展示 Langsmith）可以在 [YouTube](https://www.youtube.com/watch?v=stWiNP1o2_g) 上找到，最终代码在描述中提供。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=stWiNP1o2_g">全新的 Langserve Chat Playground 与 Agents | 编程演示</a>：在这次技术深度解析中，我们将带你领略 LangChain 和 LangServe 框架的精彩世界。在 17 分钟内，我们将为你呈现一个全面的...</li><li><a href="https://github.com/langchain-ai/langserve/pull/580">WIP: 如果 APIrouters 相互嵌套，则从正确的路由提供 playground，由 StreetLamb 提交 · Pull Request #580 · langchain-ai/langserve</a>：更新 playground 测试，以检查 index.html 中正确的 playground 资源路径。#578
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1225540697554419803)** (1 条消息): 

- **关于 Agents 搜索 PDF 的问题**：一位成员指出，他们的 Agent 在处理每个查询时都会搜索 PDF。他们怀疑代码中的 *system_prompt* 是原因，并寻求有关如何修改它的建议。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1225013268561788998)** (5 条消息): 

- **发布多个语音应用**：用户宣布发布了几个新的语音应用，包括 [CallStar](https://callstar.ai/)，并请求大家参与讨论并支持发布。该套件包括 CallJesus 和 CallPDF 等专业应用，并附带了 Product Hunt 和 Reddit 的投票链接。

- **语音应用交互性咨询**：针对语音应用的发布，一位用户询问了应用响应式设计背后的文档。原作者推荐了他们使用的 **RetellAI** 技术。

- **专为金融定制的 AllMind AI**：一个名为 [AllMind AI](https://allmindinvestments.com/) 的新 LLM 被引入社区，它专注于金融分析和研究。该工具旨在为用户提供洞察和全面的金融数据，并已在 [Product Hunt](https://www.producthunt.com/products/allmind-ai) 上亮相。

- **Galaxy AI 的免费高级 API 服务**：Galaxy AI 提供 **免费 API 服务**，允许访问包括 **GPT-4** 和 **Gemini-PRO** 在内的各种高级 AI 模型。该服务兼容 OpenAI 格式，方便项目集成，并支持 Langchain。更多详情请[立即尝试](https://discord.com/invite/BSphj69773)，更多信息请访问其[主页](https://galaxyapi.onrender.com)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://galaxyapi.onrender.com">Galaxy AI - Swagger UI</a>：未找到描述</li><li><a href="https://callstar.ai/">CallStar</a>：与角色和名人的 AI 语音通话</li><li><a href="https://allmindinvestments.com/">AllMind AI</a>：未找到描述</li><li><a href="https://www.producthunt.com/products/allmind-ai"> AllMind AI - 产品信息、最新更新和 2024 年评论 | Product Hunt</a>：AllMind AI 是一款专为金融分析和研究设计的新型大语言模型。该 LLM 通过为用户提供洞察和实时... 彻底改变了金融研究。</li><li><a href="https://calljesus.ai/">Call Jesus</a>：与耶稣进行真实的 AI 语音聊天</li><li><a href="https://callpdf.ai/">CallPDF</a>：通话任何 PDF - 真实的 AI 语音聊天</li><li><a href="https://calltube.ai/">CallTube</a>：通话任何 YouTube 视频 - 真实的 AI 语音聊天</li><li><a href="https://callwebsite.ai/">Call Website</a>：通话任何网站 - 真实的 AI 语音聊天</li><li><a href="https://callhackernews.com/">Call Hacker News</a>：Hacker News 的 AI 语音界面</li><li><a href="https://www.producthunt.com/posts/callstar"> CallStar - 与角色、YouTube 视频和 PDF 进行真实的 AI 语音通话 | Product Hunt</a>：下一代 AI 语音通话！与名人聊天，通过语音理解你的文档，并探索灵性。使用一流的 AI 语音让 AI 对话感觉真实且个性化。通话 PDF、YouTube...</li><li><a href="https://www.reddit.com/r/SideProject/comments/1bumj6s">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://news.ycombinator.com/item?id=39914442">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1225149749418659880)** (1 条消息):

- **LangChain 快速入门指南之旅**：一位用户分享了 [LangChain 快速入门指南](https://python.langchain.com/docs/get_started/quickstart) 的链接，该指南详细介绍了如何设置 LangChain, LangSmith 和 LangServe，以及如何使用提示词模板、模型、输出解析器和 LangChain Expression Language 来构建和追踪简单的应用程序。
- **示例代码与遇到的错误**：同一位用户发布了 Python 示例代码，展示了如何使用 `ChatOpenAI` 和 `ChatPromptTemplate` 类将 LangChain 与模型集成。然而，他们在运行代码时遇到了错误代码为 `404` 的 `NotFoundError`，表明无法找到资源，并就此问题寻求帮助。

**提及的链接**：<a href="https://python.langchain.com/docs/get_started/quickstart">Quickstart | 🦜️🔗 Langchain</a>：在此快速入门中，我们将向您展示如何：

  

---



**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1225367465639415828)** (3 messages): 

- **BitMat 亮相**：分享了一个指向 [BitMat GitHub 仓库](https://github.com/astramind-ai/BitMat) 的链接，该仓库提供了 "The Era of 1-bit LLMs" 论文中所提方法的**高效实现**。
- **Triton Visualizer 协作**：提议为 **Triton visualizer** 的贡献者开设一个新频道，以促进该项目的协作。
- **LASP 带来的闪电速度**：提供了另一个指向 [LASP 的 lightning_attention.py 文件](https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py) 的 GitHub 链接，涉及 **Linear Attention Sequence Parallelism (LASP)**。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/OpenNLPLab/LASP/blob/main/lasp/lightning_attention.py">LASP/lasp/lightning_attention.py at main · OpenNLPLab/LASP</a>：Linear Attention Sequence Parallelism (LASP)。通过在 GitHub 上创建账号来为 OpenNLPLab/LASP 的开发做出贡献。</li><li><a href="https://github.com/astramind-ai/BitMat">GitHub - astramind-ai/BitMat: An efficent implementation of the method proposed in &quot;The Era of 1-bit LLMs&quot;</a>："The Era of 1-bit LLMs" 中所提方法的高效实现 - astramind-ai/BitMat
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1225300703925502022)** (4 messages): 

- **切换到 max-autotune 编译**：一位成员建议将编译模式设置为 **max-autotune** 而不是 reduce-overhead，分享了他们从中获益的经验，并对 torch 团队在 [keras-benchmarks](https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17) 中可能发现的其他问题表示关注。
- **识别 torch 基准测试问题**：torch 团队最关心的问题包括**未利用 Tensor Cores** 以及启用 `torch.compile` 的不一致性。他们还指出了一些基准测试（如 SAM）的问题、可修复的 graph breaks，以及没有使用 cuda syncs 的不当计时方法，他们将在随后的详细回复中解决这些问题。

**提及的链接**：<a href="https://github.com/haifeng-jin/keras-benchmarks/blob/main/benchmark/torch_utils.py#L17">keras-benchmarks/benchmark/torch_utils.py at main · haifeng-jin/keras-benchmarks</a>：通过在 GitHub 上创建账号来为 haifeng-jin/keras-benchmarks 的开发做出贡献。

  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/)** (1 messages): 

iron_bound: : 在这里插入一段关于科学可重复性的吐槽 :
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1225047565670547496)** (3 messages): 

- **具有 Python 和 Rust 背景的 CUDA 学习路径**：一位具有 Python 和 Rust 经验的成员询问关于学习 **CUDA programming** 基础知识的建议。  
- **推荐 CUDA MODE YouTube 讲座**：另一位成员建议从 [名为 CUDA MODE 的 YouTube 频道](https://www.youtube.com/@CUDAMODE) 上的 CUDA 讲座开始，该频道还在 Discord 上提供**读书会和社区**，并在 GitHub 上提供补充内容。

**提及的链接**：<a href="https://www.youtube.com/@CUDAMODE">CUDA MODE</a>：一个 CUDA 读书会和社区 https://discord.gg/cudamode 补充内容见此处 https://github.com/cuda-mode 由 Mark Saroufim 和 Andreas Köpf 创建。

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1225462650423480483)** (1 messages): 

- **极速注意力机制**：**Triton "lightning_attention" 内核**被提及为一种高效解决方案，从而不再需要 FlashAttention 仓库插件（该插件负责跨设备拆分数据）。更多详情可在 [LASP GitHub 仓库](https://github.com/OpenNLPLab/LASP) 中找到。

**提到的链接**: <a href="https://github.com/OpenNLPLab/LASP">GitHub - OpenNLPLab/LASP: Linear Attention Sequence Parallelism (LASP)</a>: Linear Attention Sequence Parallelism (LASP)。通过在 GitHub 上创建账号，为 OpenNLPLab/LASP 的开发做出贡献。

---

**CUDA MODE ▷ #[hqq](https://discord.com/channels/1189498204333543425/1225499037516693574/1225500024029581503)** (19 messages🔥): 

- **CUDA MODE 社区简介**：新成员 **mobicham** 和 **zhxchen17** 加入了 CUDA MODE Discord，并受到了社区的热烈欢迎。
- **HQQ 与 GPT-fast 的集成**：zhxchen17 提议创建一个 demo 分支，展示 **HQQ** 如何与 **gpt-fast** 集成，包括一个独立的依赖分支、一个用于转换量化权重的脚本，以及用于协作审查的基准测试（benchmarking）。
- **专注于 Llama2 模型和量化**：Mobicham 建议由于现有的基准测试，集成应专注于 **Llama2-7B (base)**，并询问是否希望探索 4-bit 以下的更低位量化。zhxchen17 确认正在研究 **4/3 bit quantization**，并对 [Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ 模型](https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ) 表现出特别兴趣。
- **任务重点的确认与澄清**：在一些困惑之后，mobicham 澄清了将 **Llama2 HQQ** 转换为 **gpt-fast format** 的目标，并强调具有合适 group size 的 4-bit HQQ 可以带来显著的速度提升。
- **潜在的 Group-Size 限制和 API 考量**：讨论了 `torch.ops.aten._weight_int4pack_mm` 可能存在的 group-size 限制，以及将模型转换为 **GPT-fast format** 的 API 设计空间，zhxchen17 表示 **torchao team** 更适合定义 API 设计。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ">mobiuslabsgmbh/Mixtral-8x7B-Instruct-v0.1-hf-attn-4bit-moe-3bit-metaoffload-HQQ · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/int4mm.cu#L912">pytorch/aten/src/ATen/native/cuda/int4mm.cu at main · pytorch/pytorch</a>: Python 中具有强 GPU 加速的 Tensors 和动态神经网络 - pytorch/pytorch</li><li><a href="https://github.com/meta-llama/llama">GitHub - meta-llama/llama: Inference code for Llama models</a>: Llama 模型的推理代码。通过在 GitHub 上创建账号，为 meta-llama/llama 的开发做出贡献。
</li>
</ul>

</div>

---

**CUDA MODE ▷ #[triton-viz](https://discord.com/channels/1189498204333543425/1225499141241573447/1225499224251039804)** (13 messages🔥): 

- **视觉指示器的建议**：一位成员提议在可视化中添加**箭头或视觉指示器**来表示方向，并提供了一个快速模型（mock-up）来阐述该想法；然而，他们也提到并非每个元素都需要箭头，只需足以传达概念即可。
- **将操作细节集成到视觉效果中**：同一位成员分享了一个代码片段，强调了以视觉方式表示操作的建议，例如展示像 **10** 这样的操作数是如何添加到输入中的，类似于代码中 kernel 的运行方式。
- **对当前可视化实用性的担忧**：另一位成员对在当前可视化中添加索引的实用性以及它是否真的有助于理解表示担忧。
- **使用交互式元素进行调试的想法**：有人建议在可视化中加入**交互式元素**（如悬停查看数值），这可能对调试非常有利。
- **为增强交互性可能转向 JavaScript**：有人提到，为了在可视化中实现鼠标悬停（mouseover）等交互功能，可能需要将**项目移植到 JavaScript**。

---

**Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1225504686820298752)** (17 messages🔥):

- **评估 AI 改进**：成员们讨论了 [Hamel Husain 关于系统性 AI 改进的文章](https://hamel.dev/blog/posts/evals/) 的价值，称其具有“极高的价值”，并有可能启发多家公司的创立。
- **Datasette 插件增强**：提出了为 Datasette SQL 查询助手插件构建评估（evaluations）的想法，使其对用户既**可见**又**可编辑**。
- **Prompt 困境**：一位成员思考 Prompt 是否应该留在代码中，目前倾向于“是”，但推测从长远来看这可能不可持续。
- **演进中的 Prompt 管理实践**：概述了未来 AI Prompt 管理的潜在模式：**本地化、中间件和微服务模式**，反映了将 AI 服务集成到大型应用中的不同策略。
- **详细 API 响应数据的重要性**：提到了 Cohere LLM 搜索 API，强调了响应中提供的详细程度，并附带了一个显示 JSON 输出的 issue 评论链接：[Cohere API JSON 数据](https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://hamel.dev/blog/posts/evals/">- Your AI Product Needs Evals</a>：如何构建特定领域的 LLM 评估系统。</li><li><a href="https://github.com/simonw/llm-command-r/issues/2#issuecomment-2037420135">Support for the web search connector · Issue #2 · simonw/llm-command-r</a>：如果你在 API 调用中添加这个：diff --git a/llm_command_r.py b/llm_command_r.py index 7a334cd..e49c599 100644 --- a/llm_command_r.py +++ b/llm_command_r.py @@ -43,6 +43,8 @@ class CohereMessages(...
</li>
</ul>

</div>
  

---


**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1225067602288574554)** (1 条消息): 

- **对话术语的调整**：一位成员在探索 `logs.db` 时分享了关于对话术语的发现，并提到对于对话中的初始消息，“response”一词可能并不贴切。他们强调“speaker turn”或“turn”更合适，并决定将其应用的表命名为 `turns`，对这个意外产生的双关语感到有趣。
  

---



**DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1225373204206456872)** (10 条消息🔥): 

- **情感智能基准测试发布**：发布了两个新的排行榜：[Creative Writing EQ-Bench](https://eqbench.com/creative_writing.html)，评估 LLM 的情感智能；以及 [Judgemark](https://eqbench.com/judgemark.html)，衡量模型评判创意写作的能力。Judgemark 被描述为一项涉及相关性指标和成本考虑的*硬*测试；这些基准测试可以通过 EQ-Bench 流水线运行。
- **质量评分 - 寻找最佳平衡点**：在评估使用不同评分量表（从 -10 到 10、0-10、0-1 等）时，发现对于情感分析，-1 到 1 的量表效果很好，而对于质量判断，0-5 或 0-10 的量表更受欢迎，因为模型倾向于使用它们对数字含义的固有理解。
- **基于细节评判的创意写作**：创意写作基准测试的成功归功于使用了 **36 个定义狭窄的评分标准**。基于广泛标准（如“为这个故事打 0-10 分”）的评分导致区分度较弱。
- **基准测试标准已记录**：关于基准测试评分标准的问题通过指向包含标准的评判输出链接得到了解答。提供的示例是 EQ-bench 结果中的 [gemini-ultra.txt](https://eqbench.com/results/creative-writing/gemini-ultra.txt)。
- **微调评分量表**：模型间得分的标准差被用作衡量问题或标准区分能力的指标，通过这一过程，确定 0-10 评分系统最为有效。模型倾向于充分利用 0-10 范围，据推测这比 0-5 系统增加了粒度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://eqbench.com/creative_writing.html">EQ-Bench Creative Writing 排行榜</a>：未找到描述</li><li><a href="https://eqbench.com/judgemark.html">EQ-Bench Judgemark 排行榜</a>：未找到描述
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1225196411839189022)** (3 条消息): 

- **COMET 分数公布**：一位成员分享了展示各种语言模型性能的 **COMET** 分数，其中 **Facebook WMT21** 模型表现突出。最高分为 0.848375，对应文件名为 *Capybara_de_wmt21_scored.jsonl*。

- **Reference-Free Evaluation**：提到的分数是 **reference-free COMET scores**，具体使用了 **wmt22-cometkiwi-da**。提到了与评估相关的其他资源和脚本，可在 [llm_translation on GitHub](https://github.com/CrispStrobe/llm_translation) 获取。

- **Accuracy Caveats**：发布的模型结果具有参考意义，但并非绝对。成员指出当模型停止续写时可能存在不准确性，并请求在发现重大错误时予以通知。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/cstr/wmt21-dense-24-wide-en-x-st/">cstr/wmt21-dense-24-wide-en-x-st · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co">Hugging Face – 构建未来的 AI 社区。</a>：未找到描述</li><li><a href="https://github.com/CrispStrobe/llm_translation/">GitHub - CrispStrobe/llm_translation</a>：通过在 GitHub 上创建账号来为 CrispStrobe/llm_translation 的开发做出贡献。
</li>
</ul>

</div>
  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1225271024485007420)** (2 messages): 

- **AI in Healthcare 迎来新声音**：一位参与者表达了他们在 **AI 医疗领域** 的参与，表明该医疗技术领域的社区成员正在增加。

- **利用 Mixture-of-Depths (MoD) 创新 LLM**：分享了一种名为 **Mixture-of-Depths (MoD)** 的新方法；它允许语言模型动态分配计算资源 (compute)，并能够动态跳过单个专家 (expert) 的使用。论文及其摘要可通过 [PDF on arXiv](https://arxiv.org/abs/2404.02258) 获取。

**提到的链接**：<a href="https://arxiv.org/abs/2404.02258">Mixture-of-Depths: Dynamically allocating compute in transformer-based language models</a>：基于 Transformer 的语言模型在输入序列上均匀分布 FLOPs。在这项工作中，我们证明了 Transformer 可以学会动态地将 FLOPs（或计算资源）分配给特定的...

  

---


**Skunkworks AI ▷ #[finetuning](https://discord.com/channels/1131084849432768614/1131669354912678028/1225272891013337089)** (1 messages): 

- **数学问题的分解策略**：一位成员提到，与其让 AI 直接进行数学计算，不如训练它将应用题分解为方程式。这些方程式随后可以使用外部计算器（如 **Python** 或 **Wolfram Alpha**）来求解。
  

---


**Skunkworks AI ▷ #[papers](https://discord.com/channels/1131084849432768614/1156310031768232007/)** (1 messages): 

carterl: https://arxiv.org/abs/2404.02684
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 messages): 

jinastico: <@748528982034612226>