---
companies:
- openai
- deepseek-ai
- mistral-ai
- nvidia
- meta-ai-fair
- hugging-face
- langchain
- keras
date: '2024-07-19T00:13:31.815672Z'
description: '**OpenAI** 推出了 **GPT-4o Mini**，这是一款高性价比的小型模型，定价为**每百万输入 Token 0.15 美元**及**每百万输出
  Token 0.60 美元**。该模型旨在取代 **GPT-3.5 Turbo**，在提升智能水平的同时也存在一定的性能局限。**DeepSeek** 开源了
  **DeepSeek-V2-0628**，该模型登顶了 LMSYS Chatbot Arena 排行榜，彰显了其致力于贡献 AI 生态系统的决心。**Mistral
  AI** 与 **NVIDIA** 联合发布了 **Mistral NeMo**，这是一款拥有 **120 亿（12B）参数**的多语言模型，支持创纪录的 **12.8
  万（128k）Token 上下文窗口**，并采用 **Apache 2.0 协议**开源，这引发了关于其与 **Meta Llama 8B** 等模型基准测试准确性的讨论。


  在研究突破方面，**TextGrad** 框架通过文本反馈微分（textual feedback differentiation）来优化复合 AI 系统；而 **STORM**
  系统通过模拟多样化视角和解决来源偏见，将文章写作质量提升了 **25%**。开发者工具趋势则凸显了 **LangChain** 在上下文感知推理应用方面的演进，以及
  **Modular** 生态系统新增的官方 GPU 支持，其中包括关于 **Mojo** 与 **Keras 3.0** 集成的讨论。'
id: 74ef6d7d-fe1c-4343-b519-c3423fd9cf1b
models:
- gpt-4o-mini
- deepseek-v2-0628
- mistral-nemo
- llama-8b
original_slug: ainews-mini-nemo-turbo-lite-smol-models-go-brrr
people:
- liang-wenfeng
title: Mini, Nemo, Turbo, Lite —— 小模型（Smol models）大爆发 (GPT-4o-mini 版)
topics:
- cost-efficiency
- context-windows
- open-source
- benchmarking
- neural-networks
- model-optimization
- text-generation
- fine-tuning
- developer-tools
- gpu-support
- parallelization
- cuda-integration
- multilinguality
- long-context
- article-generation
---

<!-- buttondown-editor-mode: plaintext -->**首期 GPT4o Mini 特刊！**

> 2024年7月17日至7月18日的 AI 新闻。我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 社区（**467** 个频道，**2324** 条消息）。预计节省阅读时间（以 200wpm 计算）：**279 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

正如我们在前沿模型发布日的惯例，今天的 Discord 摘要有两个版本。您正在阅读的版本中，频道摘要由 GPT-4o-MINI 生成，然后频道摘要被汇总为 {4o/mini/sonnet/opus} 的摘要之摘要。请查看 [GPT4o 版本](https://buttondown.email/ainews/archive/ainews-lskjd/) 以获取完整邮件和 GPT4o 逐频道摘要对比。

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}



# AI Discord 复盘

## Claude 3 Sonnet


**1. 突破性的模型发布**

- **DeepSeek-V2-0628 登顶排行榜**：**DeepSeek** 已开源其 [DeepSeek-V2-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 模型，在 LMSYS Chatbot Arena 排行榜上排名**第一**，在硬提示（hard prompts）方面排名**第三**，可在 [DeepSeek Platform](https://platform.deepseek.com) 以每百万 token 0.3 美元的价格使用。
   - 此次发布引发了关于 **DeepSeek 开源精神**的讨论，创始人梁文锋重申了他们致力于成为 AI 生态系统中的“贡献者，而非搭便车者”的承诺。
- **Mistral NeMo 打破上下文限制**：**Mistral AI** 和 **NVIDIA** 联合推出了 [Mistral NeMo](https://mistral.ai/news/mistral-nemo/) 模型，这是一个拥有 **12B 参数**的多语言强力模型，具有前所未有的 **128k token 上下文窗口**，并以 **Apache 2.0 许可证**发布以供广泛采用。
   - 尽管令人印象深刻，但一些用户对其与 **Meta Llama 8B** 等模型相比的**基准测试准确性**表示**怀疑**，引发了 AI 工程师之间的激烈辩论。
- **OpenAI 发布高性价比 GPT-4o Mini**：**OpenAI** 推出了备受期待的 [GPT-4o Mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)，被誉为目前“**最强大且最具成本效益的小型模型**”，价格仅为每百万输入 token **0.15 美元**和每百万输出 token **0.60 美元**。
   - 该模型旨在取代 **GPT-3.5 Turbo**，以极低的成本提供增强的智能，尽管一些用户注意到与 **GPT-4o** 等大型变体相比存在**性能限制**。
  


**2. 开创性的研究突破**

- **TextGrad 解锁神经网络优化**：[TextGrad 论文](https://arxiv.org/abs/2406.07496)介绍了一个在**神经网络中进行文本反馈微分**的突破性框架，为优化超越传统方法的**复合 AI 系统**开辟了新途径。
   - 研究人员称赞 TextGrad 是 AI 领域的一个**范式转移**，允许编排多个大语言模型 (LLMs) 以增强性能。
- **STORM 利用 LLM 提升文章写作**：创新的 [STORM 系统](https://arxiv.org/abs/2402.14207)通过模拟多样化视角，展示了在文章组织方面的 **25% 提升**，使 LLM 能够生成类似于维基百科条目的**有据可查且结构化的长篇内容**。
   - 通过解决**来源偏见转移**和**无关事实过度关联**等挑战，STORM 展示了通过其**提问框架**改进 AI 生成写作的潜力。
  


**3. 开发者工具的新兴趋势**

- **LangChain 赋能上下文感知应用**：开发者探索了 **LangChain** 的功能，询问了其特性，如用于动态交互的 **AgentExecutor**、使用 **MongoDB** 作为向量数据库，以及集成专有模型以外的外部 API 模型。
   - 虽然 AgentExecutor 可能会被弃用，转而使用更灵活的 **LangGraph**，但 LangChain 仍在继续发展，成为构建**上下文感知推理应用**的强大框架。
- **Modular 加速 AI 开发**：**Modular** 生态系统（包括 **Max** 和 **Mojo 🔥**）随着官方宣布支持 **GPU** 而受到关注，引发了关于**并行化**、**CUDA 集成**以及潜在的 **NVIDIA 合作**的讨论。
   - 开发者深入研究了 **Mojo** 的细节，如**命名规范**、**数据类型**以及最近发布的 **Keras 3.0**，强调了该框架在加速 AI 开发方面的多功能性。

## Claude 3.5 Sonnet


**1. AI 模型发布与基准测试**

- **DeepSeek 在 LMSYS Arena 的主导地位**：DeepSeek 宣布开源发布 **DeepSeek-V2-0628**，该模型在 LMSYS Chatbot Arena 排行榜的多个类别中位列第一，其中在 Hard Prompts 类别中排名第三。
   - 该模型目前已在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 上线，并在 [DeepSeek Platform](https://platform.deepseek.com) 提供 API，引发了 AI 社区对其性能和潜在应用场景的热烈讨论。
- **OpenAI 的 GPT-4o Mini 引发关注**：OpenAI 推出了 **GPT-4o Mini**，这是一款旨在取代 GPT-3.5 Turbo 的新模型，在提供更高智能的同时，成本显著降低，每百万 Input Tokens 仅需 0.15 美元，每百万 Output Tokens 仅需 0.60 美元。
   - 该模型的发布因其有望推动先进 AI 能力的普及而令人兴奋，尽管一些用户反映其在高效处理大规模代码编辑方面存在局限性。
- **Mistral NeMo 的惊艳亮相**：Mistral AI 与 NVIDIA 合作推出了 **Mistral NeMo**，这是一个拥有 12B 参数的模型，具备 128k Token 的上下文窗口和多语言能力，采用 Apache 2.0 许可证发布。
   - 虽然该模型的发布受到了热烈欢迎，但一些社区成员对其报告的基准测试准确性提出了质疑，特别是与 Llama 3 8B 等模型的对比。
  


**2. AI 研究与开发的进展**

- **STORM 的结构化文章生成**：研究人员介绍了 **STORM**，这是一种新型写作系统，利用 Large Language Models 生成有据可查、组织严密的篇章，其质量可与维基百科条目媲美，详见[新论文](https://arxiv.org/abs/2402.14207)。
   - 通过进行多视角提问，STORM 在感知组织性方面比传统方法实现了 **25% 的绝对提升**，解决了生成内容中来源偏见转移和无关事实过度关联等挑战。
- **Patch-level Training 优化 LLM**：引入了一种名为 **Patch-level Training** 的新技术，该技术将多个 Token 压缩成一个 Patch，正如[近期论文](https://arxiv.org/abs/2407.12665)所述，这有可能降低 Large Language Models 的训练成本。
   - 研究人员正在探索该阶段学习率的收益，并讨论改进性能的潜在修改方案，目前正在通过实验收集关于不同学习率调度方案有效性的经验证据。
- **Transformers 的隐式推理能力**：一篇[研究论文](https://arxiv.org/abs/2405.15071)探讨了 Transformers 如何通过广泛训练提高隐式推理能力，表明可能会形成 **Inferential Generalization** 电路，以更好地处理 Out-of-distribution 示例。
   - 研究强调，超越饱和点的训练可以显著增强模型推断事实的能力，而非仅仅是死记硬背输入，这可能会带来更稳健、更通用的 AI 系统。
  


**3. AI 行业挑战与监管**

- **欧盟法规造成 AI 准入障碍**：讨论强调了对 **EU Regulations** 可能阻碍 AI 模型获取的担忧，一些用户建议未来可能需要使用 VPN 来下载某些模型。
   - 这种情况引起了大型科技公司的挫败感，可能会影响它们在该地区的运营决策，并引发了关于 AI 领域创新与监管平衡的质疑。
- **关于开源模型许可的辩论**：**Deepseek License** 遭到了用户的批评，认为其难以理解，尽管为学术界提供了更便宜的 API 使用方案，但这可能阻碍其更广泛的采用。
   - 这引发了关于开源 AI 社区中清晰且易于获取的许可条款重要性的广泛讨论，对研究和商业应用均有影响。
- **AI 公司的扩展挑战**：关于 **OpenAI** 等公司在将规模从小型团队扩展到数千名员工，同时保持对实现 Artificial General Intelligence (AGI) 的关注时所面临的困难引发了讨论。
   - 社区成员辩论了在快速增长与创新研究之间取得平衡的挑战，将 OpenAI 的方法与老牌科技巨头进行了对比，并质疑其对产品开发和部署速度的影响。

## Claude 3 Opus


**1. Mistral NeMo 模型发布**

- **Mistral 强大的 12B 模型**：Mistral AI 发布了 [Mistral NeMo 模型](https://mistral.ai/news/mistral-nemo/)，这是一个高容量的 **12B 参数模型**，拥有令人印象深刻的 **128k token 上下文窗口**，承诺在其同级别模型中提供顶尖的准确度。
   - 该模型是 **Mistral 7B** 的直接替代方案，提供预训练版本和指令微调（instruction-tuned）版本，采用 **Apache 2.0 许可证**，并在 Hugging Face 上公开了其[代码](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)。
- **基准测试失误？**：尽管 Mistral NeMo 的规格令人印象深刻，但对其针对 **Llama 3 8B** 等模型进行的基准测试准确性也出现了一些质疑。
   - 一些用户认为报告的数据可能存在夸大或误导，对其与竞争对手相比的真实性能表现持怀疑态度。
  


**2. GPT-4o Mini 震撼全场**

- **OpenAI 的高性价比替代方案**：OpenAI 推出了 **GPT-4o Mini**，被誉为最具成本效益的小型模型，价格为 **每百万输入 token $0.15，每百万输出 token $0.60**。
   - 它在基准测试中优于许多小型模型，同时提供 **128k 上下文窗口**，使其适用于复杂的应用程序和实时交互。
- **取代 GPT-3.5 Turbo**：GPT-4o Mini 将取代 **GPT-3.5 Turbo**，因为它更智能且更便宜。
   - 免费的 ChatGPT 用户以及 Plus 和 Team 订阅者均可使用该模型，这标志着先进 AI 的可访问性发生了重大转变。
  


**3. DeepSeek 的主导地位**

- **DeepSeek-V2 登顶榜单**：[DeepSeek-V2-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 目前在 LMSYS Chatbot Arena 排行榜的多个类别中排名 **第一**，其中在 **hard prompts（硬核提示词）** 类别中排名 **第三**。
   - 该模型的 checkpoint 已在 Hugging Face 上提供，并可通过 [DeepSeek Platform](https://platform.deepseek.com) 获取 API 访问权限，进一步巩固了其地位。
- **极具成本效益的竞争者**：DeepSeek V2 在面对更庞大的对手时表现出卓越的效率，价格仅为 **每百万 token $0.3**。
   - 然而，用户对 **DeepSeek License** 感到困惑，认为其难以理解，这可能会阻碍其更广泛的采用，尽管它为学术界提供了更便宜的 API 使用方案。
  


**4. 量化探索**

- **EfficientQAT 的 INT 优化**：**EfficientQAT** 方法通过优化 **Llama-2-70B** 的均匀 INT 量化，实现了与矢量量化（vector quantization）相当的性能，在 2-bit 训练期间仅有 **3% 的准确度下降**。
   - 该模型在单张 A100 GPU 上训练，展示了显存效率优势，仅需 **19.2GB**，而 Llama-2-13B 则需要 **24.2GB**。代码可在 [OpenGVLab 的 GitHub](https://github.com/OpenGVLab/EfficientQAT) 查看。
- **量化感知查询**：研究了经过 **量化感知（quantization awareness）** 训练的 Kernel，重点关注 **Character.AI** 通过使用 INT8 训练来提高推理性能的方法。
   - 针对量化感知实现的细节提出了疑问，特别是对于那些承诺在没有传统开销的情况下增强性能的方法。
  


**5. CUDA 难题**

- **Kernel 拆分策略**：一位成员探讨了将一个 **CUDA kernel** 拆分为多个 kernel 的想法，用于处理 **flash attention** 中的多步归约（multi-step reductions）等任务，理由是单步管理内存存在困难。
   - 他们建议通过多次 kernel 启动来实现 **延迟隐藏（latency hiding）** 可能会有益，尽管也承认其有效性存在不确定性。
- **动态共享内存思考**：对 CUDA 中 **动态共享内存（dynamic shared memory）** 使用的深入探讨引发了辩论，并分享了一篇 [NVIDIA 博客](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) 以提供更多见解。
   - 围绕带有 **prefills** 的短区域分析展开了讨论，认为恰到好处的少量 token 可以显著简化建模中的 batch 准备工作。

## GPT4O (gpt-4o-2024-05-13)


**1. Mistral NeMo 模型发布**

- **Mistral NeMo 开辟新天地**：**[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)** 模型是一个拥有 12B 参数和 128k token 上下文窗口的高容量模型。它承诺提供顶尖的准确率，并可作为现有 7B 模型的快速替代方案，其[代码](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)已在 Hugging Face 上公开。
   - 该模型采用 Apache 2.0 许可证，其出色的性能以及集成到各种 AI 系统中的潜力引发了广泛讨论。
- **Mistral NeMo 强力登场**：Mistral 推出了 **[Mistral NeMo](https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw)**，这款 12B 模型设定了 128k 上下文长度的新基准，并根据 Apache 2.0 许可证发布。
   - 此次发布展示了与 NVIDIA 的合作，强调了该模型的实力及其在研究和工业界被广泛采用的潜力。
    


**2. DeepSeek V2 模型发布**

- **DeepSeek-V2 登顶排行榜**：**[DeepSeek-V2](https://platform.deepseek.com)** 跃升至 LMSYS Chatbot Arena 排行榜榜首，价格仅为 **每百万 token 0.3 美元**，展示了相较于大型竞争对手的卓越效率。
   - 该模型的开源特性和性能表现激发了人们对其在各种应用中潜在用例的兴趣和讨论。
- **DeepSeek 称霸竞技场**：**[DeepSeek-V2-0628](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628)** 目前在 LMSYS Chatbot Arena 排行榜的多个类别中位居巅峰，其中在 **hard prompts** 类别中显著排名第三。
   - [DeepSeek Platform](https://platform.deepseek.com) 提供了该模型的 Checkpoint 和 API 访问，巩固了其在 AI 社区中的强势地位。
    


**3. 高效模型训练与优化**

- **EfficientQAT 增强量化技术**：**[EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)** 为庞大的 **Llama-2-70B 模型** 优化了整数量化，在 2-bit 训练期间仅需 19.2GB CUDA 显存（VRAM），且性能仅下降 3%。
   - 相比 13B 模型所需的 **24.2GB VRAM**，该技术显著提升了内存效率，标志着在最大化利用现有计算资源方面迈出了一步。
- **Patch 级训练降低 LLM 成本**：通过引入 **[patch-level training](https://arxiv.org/abs/2407.12665)**（Patch 级训练）这一战略转折，该技术将 token 压缩为高效的 patch，为更快速、更低成本的 LLM 训练开辟了道路。
   - 将训练数据浓缩为 patch 为模型提供了“瘦身计划”，使它们在压缩后能进行微调的 token 级训练，从而节省了时间和预算。
    


**4. GPT-4o Mini 发布**

- **GPT-4o Mini 重磅登场**：**[GPT-4o Mini](https://www.theverge.com/2024/7/18/24200714/openai-new-cheaper-smarter-model-gpt-4o-mini)** 是一款更精简的模型，旨在取代 GPT-3.5 Turbo，其输入和输出 token 的价格分别为 **0.15 美元和 0.60 美元**，推动了开发者使用 AI 的平民化。
   - 该模型的推出是迈向更广泛模型可及性的一大步，引发了关于其预期集成和潜在应用的讨论。
- **小而强大：GPT-4o Mini 对标 3.5 Turbo**：OpenAI 宣布推出 **[GPT-4o mini](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)**，称其比 **GPT-3.5 Turbo** 更智能且更具成本效益。
   - 社区反应积极，强调了由于 **GPT-4o mini** 较低的成本，获取 AI 工具的机会可能会大幅增加。
    


**5. LangChain 与 LlamaIndex 集成**

- **探索 LangChain 迷宫**：人们对 **LangChain 功能** 的全貌产生了浓厚兴趣，讨论集中在 **AgentExecutor** 的交互动态以及向 LangGraph 迁移以获得更好灵活性等方面。
   - 关于将外部 API 与 **LangChain** 集成的疑问引发了讨论，尽管目前缺乏权威指南，这暗示了现有 **documentation**（文档）中存在空白。
- **RAGapp 的亮眼进化**：**[RAGapp](https://t.co/C5uOA2g2zH)** 现在已实现与 **MistralAI**、**GroqInc** 以及 **Cohere** reranker 的无缝集成，鼓励通过 Docker 进行增强部署。
   - 正如社区论坛所讨论的，它的能力引起了关注，并可能挑战 RAG 应用中的现有范式。

## GPT4OMini (gpt-4o-mini-2024-07-18)


**1. Mistral NeMo 模型发布**

- **Mistral NeMo 树立新标准**：**[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)** 是一个 12B 参数模型，引入了显著的 **128k token 上下文窗口**，有望提升推理能力和效率。
   - 它被设计为 Mistral 7B 模型的直接替代品，旨在根据 **Apache 2.0 许可证** 提供 state-of-the-art 的性能。
- **Mistral NeMo 性能基准测试**：初步基准测试表明，**Mistral NeMo** 在速度和准确性方面均优于许多现有模型。
   - 社区反馈表明，它在各种应用中的部署可以显著提高生产力。
    


**2. GPT-4o Mini 发布**

- **OpenAI 高性价比的 GPT-4o Mini**：OpenAI 推出了 **[GPT-4o Mini](https://openrouter.ai/models/openai/gpt-4o-mini)**，价格为 **每百万输入 token 0.15 美元**，**输出为 0.60 美元**，使其成为 GPT-3.5 Turbo 的有力竞争替代方案。
   - 该模型旨在降低先进 AI 能力的使用门槛，以极低的成本提供类似的性能。
- **社区对 GPT-4o Mini 的反应**：**GPT-4o Mini** 的发布受到了社区的热烈欢迎，强调了其经济实惠和高性能。
   - 用户渴望将该模型集成到他们现有的工作流中，期待能带来显著的改进。
    


**3. DeepSeek V2 性能表现**

- **DeepSeek V2 登顶 Chatbot Arena**：**DeepSeek V2-0628** 在 LMSYS Chatbot Arena 排行榜上获得了 **第一名**，因其 **每百万 token 0.3 美元** 的高性价比而备受关注。
   - 该模型的效率和性能引发了关于其在各种 AI 工作流中潜在应用的讨论。
- **用户对 DeepSeek V2 的反馈**：用户的反馈强调了 **DeepSeek V2** 模型在实时应用中的有效性，特别是在 Chatbot 场景中。
   - 社区对其未来的发展和增强持乐观态度。
    


**4. 量化技术与效率**

- **EfficientQAT 增强模型训练**：**[EfficientQAT](https://github.com/OpenGVLab/EfficientQAT)** 方法优化了 Llama-2-70B 模型的量化，在 2-bit 训练期间仅实现了 **3% 的性能下降**。
   - 这种方法显著降低了内存需求，展示了向更高效训练方法转变的趋势。
- **量化对模型性能的影响**：最近的研究表明，有效的量化技术可以在减少资源消耗的同时保持模型性能。
   - 这对于在资源受限的环境中部署 AI 模型至关重要。
    


**5. AI 抓取与版权担忧**

- **关于 AI 抓取伦理的辩论**：围绕 **AI 抓取** 伦理的讨论，特别是涉及 **YouTube 字幕** 的讨论，强调了需要更好的 **版权改革** 来保护内容创作者。
   - 成员们强调了在大规模数据利用时代，为艺术家提供适当归属和补偿的重要性。
- **社区对版权问题的回应**：社区对当前版权法在 AI 生成内容背景下的影响表达了强烈的看法。
   - 许多人主张在创新与尊重原创者权利之间取得平衡。
    

---

# 第 1 部分：Discord 高层级摘要

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Mistral NeMo 取得新突破**：Mistral AI 发布了 **Mistral NeMo 模型**，这是一个拥有 12B 参数的高容量模型，具备惊人的 128k token 上下文窗口，承诺在其同级别模型中拥有顶尖的准确度。[在此查看完整解析](https://mistral.ai/news/mistral-nemo/)。
   - 对于寻求模型性能飞跃的用户，Mistral NeMo 可作为现有 7B 模型的快速替代方案，它拥有预训练增强和指令微调，并采用备受推崇的 Apache 2.0 许可证，其[代码](https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407)已在 Hugging Face 上公开。
- **EfficientQAT：量化领域的飞跃**：**EfficientQAT** 通过优化整数型量化，提升了大规模 **Llama-2-70B 模型** 的表现，在 2-bit 训练期间仅有 3% 的性能下降，且仅需 19.2GB VRAM，是真正的空间节省利器。[了解其实现方式](https://github.com/OpenGVLab/EfficientQAT)。
   - 该技术增强了显存效率，相比之下 13B 模型通常需要 **24.2GB VRAM**，这标志着在提高模型可管理性的同时，最大限度利用现有计算资源的趋势。
- **利用 STORM 掌控叙事**：**STORM 系统** 通过模拟多维视角，彻底改变了 LLM 的预写作阶段，与传统方法相比，提升了内容的组织结构和广度。在其 [FreshWiki 数据集分析](https://arxiv.org/abs/2402.14207)中深入了解细节。
   - STORM 增强了内容生成能力，能够构建由可靠引用支持的大纲，并有显著的产出改进作为证据；在 [Stanford 的 GitHub](https://github.com/stanford-oval/storm) 上查看其工作模型和方法。
- **Memory3：塑造更智能的 LLM**：**Memory3** 的出现有望激发 LLM 的效率，其特点是引入了显式记忆机制，预计将提升模型的精细度和执行力。[阅读关于此创新的文章](https://www.marktechpost.com/2024/07/05/memory3-a-novel-architecture-for-llms-that-introduces-an-explicit-memory-mechanism-to-improve-efficiency-and-performance)。
   - LLM 中的显式记忆可能会重新定义我们对性能的预期，在不铺张浪费的情况下提供更强的处理能力。探索 Memory3 架构在实现更精简、更灵活的计算需求方面的潜力。
- **Patch-level 训练降低 LLM 成本**：**Patch-level 训练** 引入了战略性转变，该技术将 token 压缩成高效的 patch，为更快速、更低成本的 LLM 训练开辟了道路。从[原始论文](https://arxiv.org/abs/2407.12665)中了解具体操作方法。
   - 将训练数据压缩成 patch 为模型提供了“节食计划”，使它们在压缩后能为精细的 token 级训练阶段做好准备，从而缩减时间和预算。通过这位工程师的 [tweet](https://x.com/MrCatid/status/1813829489039900999?t=CaNeBo4ErLUe_irte2yoBQ&s=19) 获取更多跨领域的见解。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **HuggingChat 的小插曲**：讨论指出 Cohere 模型的响应速度较慢，某些提示词的处理时间长达 5 分钟，而 Llama 3 则在几秒钟内迅速响应。
   - 针对服务器能力的担忧，社区建议联系 Hugging Face 支持团队。
- **社区关注计算机视觉课程**：社区计算机视觉课程（Community Computer Vision Course）宣布启动，旨在涵盖从基础到高级的计算机视觉技能。
   - 该课程为学习者提供了一个共同进步的平台，并可在此获得认证：[点击此处](https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome)。
- **Mistral 发布 NeMo 强力模型**：Mistral 推出了 Mistral NeMo，这是一个 12B 模型，以 128k 的上下文长度树立了标杆，并采用 Apache 2.0 许可证。
   - 与 NVIDIA 的合作在一篇 [tweet](https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw) 中展示，强调了该模型的强大实力。
- **去水印的神奇魔法**：一款结合了 Florence 2 和 Lama Cleaner 的新去水印工具给用户留下了深刻印象，它在处理无水印图像方面表现出色。
   - 可在[此处](https://huggingface.co/spaces/DamarJati/Remove-watermark)访问，反馈显示其在不损害图像质量的情况下表现优异。
- **Cohere 修正路线**：据报道，最近对 Cohere 模型仓库的修改对其性能产生了负面影响，引发了社区警报。
   - 开发人员承认了这些问题，并正在采取积极措施修复底层基础设施问题。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **内核探讨：拆分决策**：讨论集中在为 Flash Attention 任务中的内存效率而拆分 **CUDA kernels** 的可行性和复杂性，并对延迟隐藏技术持有不同意见。
   - 一位成员思考了 **CNNs** 中的多个内核，提醒注意后续层中的内核大小和数据管理，并引用了增加的内存或寄存器需求。
- **内存掌控：动态转变**：深入探讨了 CUDA 中 **dynamic shared memory** 的使用，并分享了一篇 [NVIDIA 博客](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/) 以获取更多见解。
   - 围绕带有 **prefills** 的短区域分析展开讨论，建议只需几个合适的 token 即可显著简化模型中的批处理准备。
- **微调见解：大模型中 LoRA 的诱惑**：社区成员阐明了 LLMs 中指令微调的好处，重点关注 **LoRA** 等方法，并参考了关于 [LLM 研究见解](https://magazine.sebastianraschka.com/p/llm-research-insights-instruction) 和 [Character.AI 优化策略](https://research.character.ai/optimizing-inference/) 的文章。
   - 一个 LinkedIn 链接详细介绍了 NVIDIA 向 **Linux GPU drivers** 开源的转型，揭示了增加技术包容性和优化的机会。
- **量化奇想：训练走向量化**：成员们分析了训练中前卫的 **Quantization Awareness** 技术，探索精度平衡的配置，并深入研究 Character.AI 的推理性能方法。
   - 量化在其他讨论中也受到了审查，其中解释了组大小（group size）与质量以及内存效率之间的细微差别，并参考了 PyTorch [官方博客](https://pytorch.org/blog/accelerating-neural-network-training/) 上关于半结构化稀疏性的内容。
- **棘手的 Triton：超越编译器**：Triton 编程引起了关注，一位成员在他们的 [GitHub repo](https://github.com/alexzhang13/Triton-Puzzles-Solutions) 上分享了 Triton Puzzles 的解决方案，引发了关于使用编译器进行优化的对话。
   - 关于 Triton 将 Python 代码转换为高性能 GPU 代码的能力存在大量推测，同时在没有开发者直接干预的情况下管理 SMs 内的优化。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 模型故障**：一位用户在利用 **Stable Diffusion** 模型时遇到障碍，报告称尽管上传了模型但无法生成图像。
   - 另一位参与者就 **Stable Diffusion** 运行所需模型的必要性提供了指导，并询问了更多细节以解决困境。
- **Adobe Stock 政策收紧**：Adobe Stock 针对使用艺术家姓名制定了更严格的内容政策，这可能会影响 Gen AI 项目，导致潜在的内容清除。
   - 社区对版权复杂性感到烦恼，特别是在像 **Rembrandt** 这样的艺术家不太可能拥有活跃版权的情况下。
- **艺术放大对话**：关于 “Hires Upscaler” 及其他 AI 艺术工具中的放大功能的讨论正在进行中，引发了对命名和应用的询问。
   - 鉴于 **Adobe** 最近的政策更新，艺术家们正在交流成功让平台接受其 AI 生成艺术的技巧。
- **社区机智与智慧**：聊天中弥漫着活泼的气氛，用户在热烈的社区对话和成员间的友好调侃中分享关于“爆米花时刻”的俏皮话。
   - 即使在俏皮的讨论盛行时，关于内容审核的实质性讨论也在展开，平衡了技术谈话与社区情谊。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **GoldFinch 稳步前行**：最近推出的 **GoldFinch** 模型结合了 **Linear Attention** 和传统的 Transformers，旨在解决**平方级减速**问题并减少 **KV-Caches**，从而在标准硬件上实现更长的上下文长度。
   - 详细的 [GoldFinch 论文](https://arxiv.org/abs/2407.12077)展示了如何通过线性缩放实现高效的 **KV-Cache** 生成，大幅缩减了缓存大小并提升了推理速度。
- **字幕抓取之争**：社区热烈讨论了 **AI 抓取**的伦理问题，特别是针对 **YouTube 字幕**的抓取，引发了关于这种做法是否侵犯版权以及是否忽视了公平补偿的质疑。
   - 共识倾向于认为，在大规模数据利用时代，有必要进行**版权改革**，以确保**适当的署名**和**补偿**。
- **ICML 洞察行程**：对 **ICML 2024** 的期待值很高，特别是关于新型**蛋白质语言模型（Protein Language Models）**的演讲，同时讨论也涉及了对上传海报与视频演示的偏好。
   - 诸如 **patch-level training** 和 **multi-token prediction** 模型等令人兴奋的进展已被探索，其潜力在于降低训练成本并增强性能，详见各类[研究论文](https://arxiv.org/abs/2309.02427)。
- **是否告别 Token？**：关于**无分词语言模型（tokenization-free language models）**是增强还是削弱**可解释性**引发了辩论，人们担心粒度可能会受到损害。
   - 然而，一些人声称避开分词可能会简化模型结构，从而增强输出解释，并更紧密地模拟自然语言的细微差别。
- **利用 lm-eval-harness**：**lm-eval-harness** 的用户对 `--predict_only` 标志等功能表现出极大的兴趣，以便在生成补全后优化指标，正如关于即将推出的增强功能的讨论中所提到的。
   - 关于 **LoraConfig 不匹配**的咨询促使官方澄清了 **lm_eval** 版本的更新，而社区驱动的 **Gigachat 模型 PR** 审查展示了模型开发中的协作努力。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **兼容性冲突：DeepSeek Coder V2 Lite**：LM Studio 中的讨论指出了 **DeepSeek Coder V2 Lite** 模型的问题，特别是围绕其架构和 **NVIDIA GeForce RTX 2080** 的差异。
   - 关于 LM Studio 是否优先考虑客户端参数的询问，有助于更好地理解参数对生成响应一致性的影响。
- **Resizable BAR：LLM 性能不受影响**：对 Resizable BAR (ReBAR) 的审查结论是，它对 LLM 推理速度没有显著影响，这引发了对其在模型加载时间和多 GPU 设置中作用的思考。
   - 辩论围绕 ReBAR 对显存速度的影响展开，并针对其在 GPU 配置中的益处制定策略。
- **为 AutoGen 探索 LM Studio 预设**：[Llama-3-Groq-8B](https://github.com/MaziyarPanahi/Llama-3-Groq-8B-Tool-Use-GGUF) 工具的实现引发了关于 LM Studio 预设与 **AutoGen** 案例兼容性的疑问。
   - AI 工程师商讨了为配合最新计算进展而提高性能所需的配置更改。
- **Meta Llama 的股票分析方案**：Meta Llama 3 作为交易战略合作伙伴受到关注，重点在于详细的市场分析和风险管理。
   - 在 Prompt 中，风险管理被强调为 AI 辅助交易策略开发的关键讨论点。
- **Groq 模型在工具调用方面表现出色**：Groq 的工具调用模型因其在 [Berkeley Function Calling Leaderboard](https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF) 上的表现而引起轰动，其 8b 和 70b 模型均获得高分。
   - 这些模型的成功表明它们具有无缝集成到依赖工具的计算工作流中的潜力。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **TextGrad 激发优化热潮**：[TextGrad 论文](https://arxiv.org/abs/2406.07496) 介绍了一个独特的神经网络内 **textual feedback differentiation**（文本反馈微分）框架，提供了潜在的优化空间。
   - *AI 正在经历一场变革*，TextGrad 通过探索超越传统方法的新优化途径，在社区中引起了轰动。
- **STORM 酝酿有序文章生成**：[开创性的 STORM 论文](https://arxiv.org/abs/2402.14207) 介绍了一个使用 LLM 创作有序长篇文章（类似于维基百科条目）的系统。
   - STORM 展示了文章组织能力的 **25% 绝对提升**，其*提问框架*克服了在**偏见和事实关联**方面的重大挑战。
- **DeepSeek 在 Chatbot Arena 中夺魁**：**DeepSeek-V2-0628** 登顶 LMSYS Chatbot Arena 排行榜，现在可通过 [DeepSeek 平台](https://platform.deepseek.com) 访问。
   - 鉴于该模型**领先的性能指标**，技术社区期待其发布后能带来具有影响力的用例。
- **Mistral NeMo 的报告引起关注**：NVIDIA 和 Mistral AI 的 **12B 参数模型 Mistral NeMo** 是一款多语言奇迹，拥有 **128k context window**，可在 [GitHub](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct) 上获取。
   - 对其相对于同类产品的基准测试准确性出现了怀疑，在 AI 工程师中引发了激烈讨论，有人声称其性能指标存在水分。
- **FP8 Quantization 引发行业辩论**：关于 FP8 Quantization 的讨论升温，探讨了该技术在 AI 模型训练中的可行性，参考了 [vLLM 文档](https://docs.vllm.ai/en/latest/quantization/fp8.html)。
   - 虽然有些人将其视为提高效率的途径，但其他人质疑其**稳定性以及 NVIDIA 的参与**，引发了一系列专业见解。



---



## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **DeepSeek 魅力征服全场**：DeepSeek 发布的 **DeepSeek V2-0628** 凭借其高性价比的表现，夺得 LMSYS Chatbot Arena 榜首，在 AI 社区引起轰动。
   - 价格仅为 **每百万 token 0.3 美元**，DeepSeek V2 在面对更庞大的对手时展示了出色的效率。
- **ChatGPT 开启语音模式**：OpenAI 预告了 ChatGPT 语音模式的 alpha 发布，预计将于本月晚些时候启动，为平台引入了新的**交互性**层级。
   - Sam Altman 的声明标志着人们对 AI 服务中全新的**交互式对话功能**的期待进一步升级。
- **小而强大：GPT-4o Mini**：OpenAI 的 **GPT-4o Mini** 首次亮相，这是一款经济型模型，号称是最实惠的模型，具有显著的 **128k context window**，输入和输出 token 的成本分别为 **0.15 美元和 0.60 美元**。
   - 它以节俭的 token 定价击败了竞争对手，使其成为复杂 AI 运营领域中的强力竞争者。
- **Llama 3 亮相倒计时**：对于推测中的 **Llama 3 400B 版本** 的发布充满期待，社区关注其在未来几天内的发布。
   - 对话暗示了一个同步的发布议程，旨在扩大 Llama 3 系列在 AI 领域的影响力。
- **选择加入以获得更深入的讨论**：鼓励 AI 爱好者选择加入深入的 thread 讨论，这些讨论会发布重大更新，确保参与者能够获得信息并保持关注。
   - 此举巩固了在深耕行业发展的 AI 专业人士之间培养动态且富有洞察力的对话的主动方法。



---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Mini Might：GPT-4o mini 对比 3.5 Turbo**：OpenAI 宣布推出 **GPT-4o mini**，被描述为比 **GPT-3.5 Turbo** 更智能且更具性价比。
   - 社区反应积极，许多人强调由于 **GPT-4o mini** 成本更低，AI 工具的普及潜力将增加。
- **Eleven Labs 的音频突破**：Eleven Labs 发布了一个新的语音提取模型，扩展了 AI 音频处理能力，并提供了[更多详情链接](https://link.to.details)。
   - 这一创新符合人们对 AI 在众多应用中实际整合的日益增长的期望。
- **用户趋势：从 ChatGPT 转向 Claude**：讨论揭示了用户从 **ChatGPT** 转向 **Claude** 的趋势，表明首选 AI 平台格局发生了变化。
   - 情绪从失望到渴望不等，反映了社区对不断演进的 AI 解决方案的脉搏。
- **NVIDIA 的社交整合**：围绕 NVIDIA 即将与 Facebook 和 Instagram 的整合出现了猜测，质疑 Meta 在 AI 嵌入式社交媒体背景下的动机。
   - 关于这一战略举措的未决问题让社区对数据共享和隐私的影响产生了猜测。
- **AI 调节语速**：一位开发者分享了在 AI 语音 Agent 中完善停顿插入以调节语音输出的见解，引发了关于改善人机语音交互的辩论。
   - 虽然实现自然停顿具有挑战性，但关于使用常见语音模式训练模型的建议提出了一种协作式的进步方法。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DeepSeek 称霸 Arena**：**DeepSeek-V2-0628** 目前在 **LMSYS Chatbot Arena Leaderboard** 的多个类别中位居榜首，在 **hard prompts** 类别中名列第三。
   - 该模型的 checkpoint 已在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 上提供，API 访问由 [DeepSeek Platform](https://platform.deepseek.com) 提供，巩固了其地位。
- **GPT-4o Mini 镜像其前身**：**GPT-4o Mini** 在某些基准测试中的得分与 **GPT-3.5** 相当，作为一个小型但胜任的模型引起了关注，特别是在 [aider 的代码编辑基准测试](https://x.com/paulgauthier/status/1814014867361374610?s=46)中。
   - 尽管如此，该模型对大型代码编辑的处理欠佳引发了讨论，迫切需要在未来的迭代中进行改进。
- **Codestral Mamba 挑剔的注意力**：与预期相反，**Codestral Mamba** 的准确率在超过 **1k token** 上下文后会下降，让用户在面对其狭窄的关注范围时苦寻解决方案。
   - 由于其无法像宣传的那样有效处理“无限”上下文，失望情绪随之而来，使其在更广泛的上下文需求中的应用受到质疑。
- **AI 的“巫术”认知**：用户表达的 AI 类似于“巫术”的氛围，反映了公众对 **ChatGPT** 等工具带来的 AI 进步日益增长的不安。
   - 这种与历史焦虑的类比引发了关于社会对 AI 适应能力的辩论，并对未来 AI 的接受度和监管产生影响。
- **OpenAI 的规模化困境**：**OpenAI** 在其宏大的规模化进程中，似乎在平衡快速增长与追求 **AGI** 之间苦苦挣扎，引发了行业讨论。
   - 与 **Google** 等科技巨头的对比凸显了 OpenAI 灵活的抱负与迅速交付 AI 产品所需的韧性之间的矛盾。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Mistral NeMo 开启新的上下文视野**：**Mistral NeMo** 的发布为上下文窗口设定了新标准，支持高达 **128,000 tokens** 并展示了其推理能力，详情见[综合博客文章](https://mistral.ai/news/codestral-mamba/)。
   - 社区对其许可协议展开讨论，强调 **Apache 2.0** 协议拓宽了其在研究和工业领域的应用前景。
- **GPT-4o Mini 揭幕：OpenAI 的最新力作**：OpenAI 最近推出的 **GPT-4o Mini** 以其 **$0.15/M 输入和 $0.60/M 输出** 的定价策略备受关注，有望成为 [GPT-3.5 Turbo](https://openrouter.ai/models/openai/gpt-3.5-turbo) 的继任者。
   - 社区充满期待，准备将这一多功能模型集成到工作流中，该模型即将向广大用户开放。
- **OpenRouter：运行顺畅的绿灯信号**：**OpenRouter** 的状态报告显示一切正常，截至 2024 年 7 月 18 日的性能指标显示无干扰或停机，已由 [OpenRouter Status](https://status.openrouter.ai/) 确认。
   - 用户正密切关注区域访问性和性能，反映了对 OpenRouter 持续服务交付的依赖。
- **解决图像 Token 定价难题**：关于 **image tokens** 计费的讨论正在展开，模型更新促使人们重新评估图像分辨率如何与不断上升的成本挂钩。
   - 关于不同图像规格的统一计费实践仍存在疑问，体现了社区对成本透明度的警惕。
- **Gemma 2 的既视感：解决重复问题**：**Gemma 2 9B** 模型面临用户的审查，用户遇到了响应**重复**的问题，引发了关于潜在修复和性能优化的讨论。
   - 社区热衷于从性能指标中提取模式，旨在追踪并减轻导致重复响应的因素。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Max/Mojo 结合 GPU 掌控力**：成员们热议 **Max/Mojo 中的 GPU 支持**，并提及 Lattner 的 Nvidia 演讲，凸显了集成潜力。
   - 关于 **Mojo 并行化** 的推测不断展开，用户提出了直接暴露给尖端硬件的想法。
- **Mojo 编译器的 Nightly 升级**：[Mojo 编译器的 Nightly 更新](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)引入了嵌套 Python 对象支持等功能，并修复了标准库。
   - 有关于 **stdlib 扩展提案** 的讨论，旨在减轻维护者的工作负担，目前正等待强有力的社区验证。
- **Max Inference 引导 Llama3 洞察**：使用 `Llama3` 的 **Max Inference** 将 prompt 作为上下文，实现了 [Max 的 GitHub 示例](https://github.com/modularml/max/tree/nightly/examples/gui)中所示的交互式聊天。
   - 对话涉及通过利用本地下载和 `--model-path` 参数在 Llama 3 中加载自定义权重。
- **Lubeck 在基准测试中领先**：一场激烈的交流发生，据报道 **Lubeck** 的性能超越了 MKL，LLVM 的“秘密武器”可能在其中发挥了作用。
   - 虽然 **SPIRAL** 作为数字信号处理库的自动化竞争者出现，但其复杂性引发了关于日常功能实用性的辩论。
- **社区对 Stdlib 策略的思考**：一项 **stdlib 扩展提案** 通过建议社区驱动的“扩展”作为简化贡献的手段，引起了热议。
   - 围绕适用于高性能场景的 **Async IO API** 展开了讨论，避开了 Python 的内置功能。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere API 探索之旅**：成员们分享了关于创建调用 API 工具的见解，强调了 Cohere 控制面板在合并工具和连接器任务中的实用性。
   - 详尽的文档[重点介绍了步骤](https://docs.cohere.com/docs/tool-use)，展示了如何利用这些 API，并重点关注单步和多步方法。
- **Discord 关于图片的指令**：Discord 中的图片成为了讨论焦点，共识倾向于为特定角色启用权限，以保持内容不偏离主题。
   - 随着管理员授予图片分享权限，社区活跃度激增，引发了一波庆祝 GIF 的热潮。
- **深入 DuckDuckGo 搜索**：一位成员利用 [Python package](https://pypi.org/project/duckduckgo-search/) 发挥 DuckDuckGo 的威力，实现高效的链接检索，并暗示将与 Firecrawl 集成。
   - 这引发了关于增强信息提取的讨论，表明了利用现有工具最大化产出的趋势。
- **Firecrawl 通过自托管节省成本**：围绕自托管 [Firecrawl](https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md) 作为一种省钱替代方案的讨论非常热烈，尽管其价格不菲。
   - 社区分享了经验和资源，为那些受困于服务成本的人们描绘了缓解压力的前景。
- **GPT-4o 与 Streamlit 结合实现高效 PoC**：出现了将 GPT-4o 与存储在 .env 文件中的个人 API key 集成的策略，并结合使用 Streamlit 进行敏捷的 PoC 开发。
   - 这种集成方案为 API 与抓取工具的无缝融合奠定了基础，体现了渐进式的协作。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **罗技提供的 Perplexity Pro 优惠**：围绕罗技发送的提供 6 个月 **Perplexity Pro** 的邮件展开了讨论，用户们对该优惠的真实性进行了辩论，直到成功兑换 **promo code** 的确认信息出现。
   - 参与者指出 Dmitry Shevelenko 与罗技之间的合作推文，展示在[这里](https://x.com/dmitry140/status/1813698975884792095)，标志着合作旅程的开始。
- **GPT-4o Mini 重磅登场**：OpenAI 推出了备受瞩目的 **GPT-4o Mini**，这是一款更精简的模型，旨在取代 GPT-3.5 Turbo 并为开发者普及 AI。
   - 该模型的推出是迈向更广泛模型可访问性的一大步，引发了关于模型预期集成的讨论，详见 [OpenAI 的公告](https://www.theverge.com/2024/7/18/24200714/openai-new-cheaper-smarter-model-gpt-4o-mini)。
- **ChatGPT 拆分句子引发猜测**：由于用户剖析了 **ChatGPT** 发送拆分响应的奇特行为，试图理解其复杂性，从而引发了困惑。
   - 这一困境被认为与最新的 **GPT-4o Mini implementation** 有关，引发了对根本原因的辩论，但尚未有明确结论。
- **DALL-E 升级备受关注**：随着用户报告故障并期待新版本发布，**DALL-E** 的更新引发了讨论。
   - 这些见解引发了关于通过升级解决图像生成问题的猜测，指向即将推出的更新。
- **在 NextCloud 中连接 Perplexity**：一位用户在尝试配置 **NextCloud** 以使用 **Perplexity API** 时遇到了集成困难，特别是关于模型选择的谜团。
   - 一位热心的成员提供了建议，通过修改 payload 中的 'model' 字符串来调整模型选择，尽管具体的实现细节仍不明确。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **LangChain 迷宫探索**：对 **LangChain features** 全貌的好奇心激增，讨论涵盖了 **AgentExecutor** 的交互动态以及向 LangGraph 转型以提升灵活性。
   - 关于将外部 API 与 **LangChain** 集成的提问引发了讨论，尽管目前缺乏明确的指南，这暗示了现有 **documentation** 中存在空白。
- **Debugger 深入探讨与 Langserve 层级**：求知者们探讨了 [Langserve Debugger](https://registry.hub.docker.com/r/langchain/langserve-debugger) 在解决 **LangChain ecosystem** 内部问题方面的效用。
   - 辩论聚焦于区分标准的 **Langserve container** 与其 Debugger 版本，后者更侧重于解决问题的能力。
- **ChatPromptTemplate 中的模板纠葛**：一位用户在尝试于 `ChatPromptTemplate` 中发挥 **JSON** 威力时遇到了令人困惑的 **KeyError**，其中 **'$schema'** 变量似乎“玩起了捉迷藏”。
   - **GitHub interventions** 建议将 JSON 相关问题包装在双大括号中，这一未经测试的方案为该问题引发了更多讨论。
- **Easy Folders 在 Product Hunt 上线**：**Easy Folders** 在 **Product Hunt** 亮相，在 **Browser Extensions** 和 **AI** 类别中备受关注，以有序的聊天记录和整洁的 prompt 管理器吸引用户。
   - 一项巧妙的 **30-day Superuser giveaway** 活动通过点赞和评论进行引流，用户纷纷涌向 Easy Folders 寻求 **free trial**。
- **聊天机器人幻觉的融合修复**：**Corrective RAG** 与 **RAG Fusion** 的结合成为解决 AI 聊天机器人幻觉的一种方案，是追求可靠性的 Python 开发者的良方。
   - 一个关于使用 **LangGraph** 创建本地聊天机器人的 [YouTube guide](https://www.youtube.com/watch?v=7h6uDsfD7bg) 承诺了简易性，解决了关于构建可信 AI 交互的讨论。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **知识助手的预言**：**Jerry Liu** 关于知识助手未来的精彩主题演讲吸引了与会者，[录音已发布](https://t.co/o93s5WSMIV)，标志着他作为 AI 领域引领者的地位。
   - 社区成员强调了该演讲对于理解该领域关键改进的价值。
- **RAGapp 的显著进化**：**RAGapp** 现在已与 **MistralAI**、**GroqInc** 和 **Cohere** reranker 无缝集成，并鼓励通过 Docker 进行增强部署。
   - 它的竞争力引发了关注，并可能挑战 RAG 应用程序的现有范式。
- **Stack Podcast 上的数据深度探讨**：在由 **Jerry Liu** 主持的 Stack Podcast 中，围绕 **prompt engineering** 和 **long context windows** 展开了重要讨论，提供了对主流 AI 障碍的 [insights](https://t.co/C5uOA2g2zH)。
   - 这些对话得到了社区的回响，提炼出了对任何 AI 工程师工具箱都至关重要的知识。
- **索引效率受到质疑**：社区成员讨论了处理 **Neo4jPropertyGraphStore** 时缓慢的索引性能，并将数据量视为一个影响因素。
   - 达成的共识是，大型存储库会增加索引时间，这一细节对于管理预期至关重要。
- **查询效能与解析难题**：使用 **GPT4o** 和 **Sonnet3.5** 进行的 **Multimodal RAG** 试验引发了对 query rewriting 及其益处以及 **LlamaIndex** 内部机制的好奇。
   - 在 RAG 的 **Langchain** 使用和文档处理方面的具体经验引发了与 **LlamaIndex** 独特解析方法的比较，并促成了一次关于正确实现的 [GitHub-based exchange](https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/claude_parse.ipynb)。

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **Mistral 在 Axolotl 武器库中的威力**：一名成员询问 Axolotl 是否能无缝集成拥有 **128k token 上下文窗口** 的 [Mistral 12B NeMo 模型](https://mistral.ai/news/mistral-nemo/)。
   - 对话中引发了关于通过尝试来验证兼容性的玩笑，强调了 **实验 (experimentation)** 是潜在的解决之道。
- **MMLU 意外：Llama 3 的评分风云**：**Llama 3 8B** 的 MMLU 评分报告不一致，范围在 **62.3%** 到 **66.6%** 之间，引发了关于模型性能差异的讨论。
   - 随后展开了关于 **TriviaQA** 基准测试有效性的辩论，建议有必要进行标准化报告。
- **Transformer 跨越到深度推理**：成员们分享了一篇关于 Transformer 实现 *grokking*（一种增强的推理形式，暗示处理复杂推断的能力）潜力的论文见解。
   - 该论文假设，通过大量的训练，Transformer 可能会发展出超越记忆的 **推理性泛化 (inferential generalization)** 能力。
- **微调：更大的空间意味着更多的余地？**：讨论强调了 **12B 模型** 为优秀的微调提供了充足的空间，相较于 **Llama 3 8B** 处于有利地位。
   - 大模型尚未达到其训练极限的观点表明，在微调场景中更有机会获得卓越的结果。
- **Llama3：首选模型还是潜在幻象？**：在 #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1263545706749366385) 频道中，成员们将 **Llama3** 视为未来工作的目标模型，引发了一系列充满希望的推测。
   - 尽管训练损失呈现积极趋势，但社区在实验性 Rank 调整后对其潜力仍保持谨慎乐观。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **模型进入微调狂热**：成员间的辩论揭示了在微调过程中，开源模型（如 **Mistral 7B** 和 **Llama3**）与 **gpt-3.5-turbo** 之间令人惊讶地缺乏性能对比。
   - 当 **gpt-3.5-turbo** 表现似乎优于其他模型时，引发了好奇，人们猜测 OpenAI 的 **数据政策 (data policies)** 可能是导致其未能被更广泛采用的原因。
- **M1 Mac 在模型内存上遇到对手**：首次加载模型的延迟让一位在 **Mac M1** 上进行测试的 **Hugging Face** 爱好者感到沮丧，指出初始内存分配是罪魁祸首。
   - 社区成员表示，这个瓶颈在后续运行中可以被绕过，建议通过重复测试以获得更流畅的体验。
- **计时策略解决棘手问题**：成员们交流了如何将 **模型加载 (model loading)** 与 **推理 (inference)** 分开的策略，以解决工作流中的计时难题。
   - 这种诊断性的划分可以揭示流程中哪一部分是性能痛点。
- **保密性引发微调中的敏感性**：敏感的业务数据成为了障碍；用户表达了对将客户和患者数据等机密信息委托给外部公司的担忧。
   - 这种忧虑凸显了在隐私保护与外部微调服务能力之间取得平衡的普遍困境。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Meta 的多模态探索**：Meta 的雄心壮志随着其在 **multimodal AI models**（多模态 AI 模型）领域的突破而飙升，承诺将增强用户与技术的交互方式。
   - Meta 的这一举措旨在将不同类型的数据输入编织在一起，以创造更丰富、更**集成的用户体验**。
- **Llama 告别欧盟**：由于监管环境的影响，**Llama models** 告别了欧盟用户，引发了关于该地区 AI 能力削弱的讨论。
   - 这一决定凸显了欧洲日益增长的**监管挑战**，这些挑战正影响着先进 AI 技术的可用性和可访问性。
- **Codestral Mamba 迈向成功**：源自 Mixtral 系列的 **Codestral Mamba** 发布，标志着代码生产力的进步，它具备线性时间推理和处理理论上无限长度序列的能力。
   - 该模型由 Albert Gu 和 Tri Dao 提供技术支持，确保了深度交互的快速响应，正如其 [发布公告](https://mistral.ai/news/codestral-mamba/) 中所强调的那样。
- **通过 Prover-Verifier 对话提升清晰度**：为了提高模型输出的可读性，**OpenAI 的 Prover-Verifier** 机制通过阐明 LLMs 回答背后的思维过程，提升了清晰度。
   - 通过参与这些人工对话，LLM 输出的透明度得到了显著提升，促进了更深层次的理解，详见 [OpenAI 的方法](https://openai.com/index/prover-verifier-games-improve-legibility/)。
- **NuminaMath-7B 的数学造诣**：NuminaMath-7B 在 AIMO 竞赛中脱颖而出，超越了竞争对手，解决了大量复杂的**高中数学问题**。
   - 然而，爱好者们强调在解读这些胜利时要保持谨慎，因为基准测试可能无法完全捕捉到 **LLMs 的基础推理缺陷**，这一观点在一条 [推文](https://x.com/JJitsev/status/1813930981637902486) 中有所分享。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **自动化 CI 的烦恼**：有人对在 Pull Requests (PRs) 上自动运行的持续集成 (CI) 流程表示担忧，这干扰了开发者的工作流。
   - 建议的修复方案是**让 CI 不受干扰地运行**，直到 PRs 退出草稿状态并需要同行评审。
- **为搞怪 AI 调整模板**：讨论围绕自定义 AI 模板重命名列的模糊性，以及是否保留 **alpaca cleaned dataset**。
   - 一位成员做出了澄清，他计划在未来使用 alpaca 数据集，尽管目前的重点是一个配置为输出 "HAHAHA" 的滑稽模板。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GTX 1080 在 tinygrad 运行中受阻**：用户在使用其 **GTX 1080** 时遇到了 `tinygrad.device.CompileError`，引发了关于在设置 `CUDA=1` 时该显卡与 tinygrad 兼容性的技术咨询。
   - 社区成员参与了讨论，探讨了旧代 NVIDIA 显卡是否缺乏支持，以及是否需要通过补丁 **ops_cuda** 或禁用 tensor cores 等解决方案。
- **展望未来：新硬件，新视野**：讨论转向 **2080 series GPUs**，这似乎是顺利运行 tinygrad 的最低要求，凸显了旧款 NVIDIA 型号可能被排除在外。
   - 作为积极的一步，原帖作者提到正在更现代的系统上设置 tinygrad 以规避兼容性障碍，并对社区的建议表示感谢。

---

**Alignment Lab AI Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**MLOps @Chipro Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**DiscoResearch Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长时间没有动态，请告知我们，我们将将其移除。

---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1263215251873202176)** (245 条消息🔥🔥): 

> - `Mistral NeMo`
> - `Gemma 2 Models`
> - `RAG Frameworks`
> - `Using Windows vs Linux for AI`
> - `Unsloth Compatibility` 


- **Mistral NeMo 模型发布**：Mistral AI 发布了 [Mistral NeMo 模型](https://mistral.ai/news/mistral-nemo/)，这是一个 12B 模型，具有高达 128k tokens 的超大上下文窗口，并声称在其参数量级别中拥有最先进（state-of-the-art）的准确度。
   - 该模型旨在作为 Mistral 7B 的直接替代方案，提供预训练和指令微调（instruction-tuned）版本，并采用 Apache 2.0 许可证发布。
- **关于 Gemma 2 和大语言模型的讨论**：参与者讨论了 Unsloth 是否支持最新的 Flash Attention (FA)，以及这对于训练带有 soft capped attention 的 Gemma 2 等大型模型的影响。
   - 会议强调了在 FA2.6 中运行 Gemma 2 滑动窗口以支持更长上下文长度的能力，并希望在未来的更新中实现兼容。
- **对 RAG 框架的关注**：讨论涉及了 RAG 模型在快速检索信息方面的优势，同时也讨论了对使用 LLM 处理实质性任务的怀疑态度。
   - 参与者一致认为 RAG 和微调（fine-tuning）并非互斥，并建议在组织环境中将两者结合使用。
- **优化训练期间的 VRAM 占用**：一位用户表达了对模型训练时 VRAM 限制的担忧，寻求在训练过程中不占用额外 VRAM 的情况下执行验证（validation）。
   - 针对如何使用标准的 HF trainer 设置独立运行评估而不影响训练过程，提供了相关建议。
- **在 AI 任务中使用 Windows 与 Linux 的对比**：讨论了在 AI 任务中使用 Windows 与 Linux 的可行性，观点倾向于 Linux，因为它在兼容性和资源管理方面更具优势。
   - 参与者指出，许多游戏在 Linux 上运行良好，同时也能处理训练负载，这说明了 Linux 在高性能计算环境中的灵活性日益增强。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个与 NVIDIA 合作构建的 12B 模型，具有 128k 上下文长度，采用 Apache 2.0 许可证发布。</li><li><a href="https://arxiv.org/html/2407.07858v1">FACTS About Building Retrieval Augmented Generation-based Chatbots</a>: 关于构建基于检索增强生成（RAG）聊天机器人的事实</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 暂无描述</li><li><a href="https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention">AIUnplugged 15: Gemma 2, Flash Attention 3, QGaLoRE, MathΣtral and Codestral Mamba</a>: 洞察胜于信息</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e5uxnj/thanks_to_regulators_upcoming_multimodal_llama/">Reddit - 深入探讨一切</a>: 暂无描述</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/gigabyte-releases-ai-software-to-help-train-your-own-ai">技嘉发布 AI 软件以帮助训练你自己的 AI —— AI TOP 工具利用技嘉主板、GPU、SSD 和电源来微调本地 AI 模型训练</a>: 该工具可以本地训练高达 236B 参数的 AI 模型。</li><li><a href="https://github.com/bclavie/RAGatouille">GitHub - bclavie/RAGatouille: 在任何 RAG 管道中轻松使用和训练最先进的后期交互检索方法（ColBERT）。专为模块化和易用性而设计，并有研究支持。</a>: 在任何 RAG 管道中轻松使用和训练最先进的后期交互检索方法（ColBERT）。专为模块化和易用性而设计，并有研究支持。 - bclavie/RAGatouille
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1263316194920763413)** (11 条消息🔥): 

> - `3090 显卡推荐`
> - `双 4090 讨论`
> - `Runpod 的优势`
> - `Womp Womp 时刻` 


- **推荐 3090 而非 TI**：一位成员建议购买 **3090**（而非 TI 版本），并表示二手卡（甚至是矿卡）也是可以接受的。
   - 这一建议来自一位已经拥有 **2x4090** 显卡的用户。
- **Runpod 相比双 4090 的优势**：针对拥有双 4090 的话题，一位成员宣称 **Runpod** 甚至比拥有两张 4090 更好。
   - 这表明 GPU 需求正在向云端解决方案转变。
- **轻松的 'Womp Womp' 交流**：围绕 **'womp womp'** 这一短语展开了有趣的互动，该短语被多次提及。
   - 成员们很享受这个时刻，一位用户表示他们必须一吐为快，引发了笑声。
- **提到 Shylilly 粉丝**：一位用户注意到聊天中有很多 **Shylilly** 粉丝，暗示了共同的兴趣。
   - 这引发了对该群体共同兴趣的愉快认可。
- **搜索 'Womp' 时刻**：一位成员提示使用 **'womp'** 搜索命令，暗示聊天记录中有很多类似的时刻。
   - 这反映了成员之间持续的轻松玩笑和有趣互动。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1263241651560644728)** (84 条消息🔥🔥): 

> - `禁用 pad_token`
> - `微调 4-bit 模型`
> - `在本地运行微调`
> - `模型显存消耗`
> - `在网站中实现模型` 


- **无隐患禁用 pad_token**：一位成员指出，即使设置了 **pad tokens**，模型也不会使用它们，因此可以忽略而不会产生后果。
   - 另一位用户提醒该信息尚未经过验证。
- **模型效率的微调决策**：成员们讨论了是将微调后的 **4-bit 模型** 保存为 **16-bit** 更好，还是在生产环境中直接使用 **LoRA** 适配器，有人建议使用 LoRA 的准确度可能最高。
   - 然而，他们澄清说，适配 **QLoRA** 处理大型模型需要 **48GB VRAM**。
- **本地微调环境搭建的挑战**：一位用户表示难以找到本地运行微调的指南，并表达了学习如何在不使用 **Google** 的情况下训练模型的意图。
   - 另一位用户建议在 Windows 上使用 **WSL** 以获得更好的环境兼容性。
- **模型的内存考量**：关于 **Llama 3** 模型内存消耗的查询得到了澄清，以 **4-bit** 形式加载大约消耗 **5.7 GB RAM**。
   - 随后讨论了量化问题，成员们推测了各种 bit 格式相关的质量损失。
- **寻求训练后模型的实现资源**：一位成员寻求将训练好的模型集成到网站中的资源，并获知下周将发布相关视频。
   - 与此同时，建议在 YouTube 上查看相关内容，尽管没有给出具体的推荐。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/wsl/install">安装 WSL</a>：通过命令 `wsl --install` 安装 Windows Subsystem for Linux。在 Windows 机器上使用由你偏好的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。</li><li><a href="https://tinygrad.org/#tinybox">tinygrad: 一个简单且强大的神经网络框架</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/210">我在原生 Windows 上运行了 unsloth · Issue #210 · unslothai/unsloth</a>：我在原生 Windows（无 WSL）上运行了 unsloth。你需要 Visual Studio 2022 C++ 编译器、Triton 和 DeepSpeed。我有一个完整的安装教程，我本想写在这里，但我现在在用手机...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1263300998995378238)** (7 条消息): 

> - `STORM 写作系统`
> - `EfficientQAT`
> - `Memory3 架构`
> - `量化技术`
> - `Patch-level 训练` 


- **STORM 提出新的写作框架**：该论文介绍了 **STORM**，这是一个旨在通过模拟多样化视角并策划基于可靠来源的大纲来增强 LLM 预写作阶段的系统。使用 **FreshWiki** 数据集的评估显示，与传统方法相比，其组织结构提升了 25%，内容广度增加了 10%。
   - 完整详情请参阅 [arXiv 论文](https://arxiv.org/abs/2402.14207)。
- **EfficientQAT 最大化 INT 量化效果**：**EfficientQAT** 方法通过为 Llama-2-70B 优化均匀 INT 量化，实现了与向量量化相当的性能，在 2-bit 训练期间准确率仅下降了 **3%**。该模型在单张 A100 GPU 上训练，展示了显存效率优势，仅需 **19.2GB**，而 Llama-2-13B 则需要 **24.2GB**。
   - 代码可在 [OpenGVLab 的 GitHub](https://github.com/OpenGVLab/EfficientQAT) 查看。
- **Memory3：提升 LLM 效率**：引入了一种名为 **Memory3** 的新型架构，提供显式记忆机制，旨在提高 LLM 的性能和效率。架构及其影响可以在[此处](https://www.marktechpost.com/2024/07/05/memory3-a-novel-architecture-for-llms-that-introduces-an-explicit-memory-mechanism-to-improve-efficiency-and-performance)探索。
- **探索高级量化技术**：**Spectra LLM** 套件展示了在 300B token 上训练的 54 个模型，对包括三元量化（ternary quantization）和 FloatLMs 在内的不同模型压缩方法进行了广泛对比。值得注意的是，**TriLM 3.9B** 模型在性能上优于以往的三元模型，且体积比半精度（half-precision）对应模型更小。
   - 该套件探讨了低位宽训练模型的性能动态，详见相关的 [arXiv 论文](https://huggingface.co/papers/2407.12327)。
- **Patch-Level 训练革新 LLM 训练**：**Patch-level 训练**的引入通过将多个 token 压缩到单个 patch 中，显著降低了 LLM 训练成本，使模型能够更高效地处理数据。该方法允许后续进行 token-level 训练以对齐模型进行推理，从而提升整体训练速度。
   - 更多见解请见[完整论文](https://arxiv.org/abs/2407.12665)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/papers/2407.12327">论文页面 - Spectra: A Comprehensive Study of Ternary, Quantized, and FP16 Language Models</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2407.12665">Patch-Level Training for Large Language Models</a>：随着大语言模型 (LLMs) 在语言理解和生成方面取得显著进展，其训练效率已成为关键关注点。传统上，LLMs 被训练用于预测...</li><li><a href="https://x.com/MrCatid/status/1813829489039900999?t=CaNeBo4ErLUe_irte2yoBQ&s=19">来自 catid (e/acc) (@MrCatid) 的推文</a>：2倍速 LLM 训练：https://arxiv.org/abs/2407.12665 可能也适用于其他类型的 Transformer 模型！</li><li><a href="https://www.marktechpost.com/2024/07/05/memory3-a-novel-architecture-for-llms-that-introduces-an-explicit-memory-mechanism-to-improve-efficiency-and-performance">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>：我们研究了如何应用大语言模型从零开始撰写有据可查且组织良好的长篇文章，其广度和深度可与维基百科页面相媲美。这个尚未被充分探索的问题提出了新的...</li><li><a href="https://storm.genie.stanford.edu/article/ai-human-relations-and-the-complexity-it-introduces-to-society-18731">未找到标题</a>：未找到描述</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>：一个由 LLM 驱动的知识策划系统，可研究特定主题并生成带有引用的完整报告。</li><li><a href="https://www.reddit.com/r/LocalLLaMA/s/P84n4i083q">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/OpenGVLab/">OpenGVLab</a>：上海人工智能实验室通用视觉团队。OpenGVLab 在 GitHub 上拥有 65 个代码库。
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1263597700813946882)** (1 messages): 

> - `水印去除工具`
> - `CandyLLM Python 库`
> - `AI 漫画工厂更新`
> - `快速字幕制作工具`
> - `NLP 路线图` 


- **水印去除工具发布**：由 @damarjati_ 发布了一个使用 **Florence 2** 的新 [水印去除](https://huggingface.co/spaces/DamarJati/Remove-watermark) 工具。其旨在简化从图像中有效去除水印的过程。
   - *这可以通过简化编辑流程显著提高内容创作者的生产力。*
- **介绍 CandyLLM Python 库**：**CandyLLM** 是由 @shreyanmitra_05940_88933 创建的一个新 Python 库，具有用于用户友好交互的 [Gradio UI](https://github.com/shreyanmitra/CandyLLM)。该库增强了开发者对各种 ML 模型的访问便捷性。
   - *鼓励用户探索其功能，以便更轻松地集成到项目中。*
- **AI 漫画工厂现已包含对话气泡**：感谢 @jbilcke，**AI 漫画工厂** 已更新并默认包含 [对话气泡](https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory)。这一增强功能允许创建更具互动性和吸引力的漫画。
   - *添加对话气泡旨在提高漫画的用户参与度和叙事性。*
- **快速字幕制作工具发布**：由 <@911742715019001897> 发布了一个 **快速字幕制作工具**，可通过 [此链接](https://huggingface.co/spaces/Nick088/Fast-Subtitle-Maker) 访问。该工具旨在简化为视频创建字幕的过程。
   - *用户现在可以轻松地为他们的媒体添加字幕，显著减少视频编辑所需的时间。*
- **面向开发者的 NLP 路线图已发布**：@kmjoshi 分享了一份全面的 **NLP 路线图**，现已在 [GitHub](https://github.com/kjdeveloper8/nlp-projects) 上提供。该路线图为有兴趣探索各种 NLP 项目的开发者提供指南。
   - *它为那些热衷于提高对 NLP 的理解和技能的人提供了资源和方向。*



**提到的链接**：<a href="https://youtu.be/cpoS7K_fpRM)">如何从任何领域转型到 Machine Learning？| Artificial Intelligence ft. @vizuara</a>：在这段视频中，来自 Vizuara 的 Raj Dandekar 博士分享了他从机械工程转型到 Machine Learning (ML) 的经验。他还解释了...

  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1263212225716293662)** (222 条消息🔥🔥): 

> - `HuggingChat 性能问题`
> - `模型训练疑虑`
> - `Cohere 模型问题`
> - `RVC 及替代语音模型`
> - `Meta-Llama-3-70B-Instruct API 错误` 


- **HuggingChat 性能问题**：用户报告了 Cohere 模型的响应速度缓慢，部分 prompt 处理时间长达 5 分钟，而 Llama 3 等其他模型则在几秒钟内完成处理。
   - 有人对服务器能力表示担忧，并建议直接联系 Hugging Face 获取支持。
- **模型训练疑虑**：一位用户对模型训练过程中的高 loss 率表示沮丧，并询问最佳的 batch size 和 gradient accumulation 策略。
   - 讨论内容包括增加 epoch 与过拟合风险之间的权衡，以及使用更大型数据集的影响。
- **Cohere 模型问题**：强调了 Cohere 模型的一个持续性问题，该问题源于模型仓库最近的更改，影响了性能。
   - 团队承认了该问题，并确认正在积极努力解决任何基础设施问题。
- **RVC 及替代语音模型**：一位参与者质疑 RVC 模型的可靠性，提到网上报告的许多版本都无法正常工作。
   - 他们询问了其他用于创建 AI 语音模型的项目，作为 RVC 的替代方案。
- **Meta-Llama-3-70B-Instruct API 错误**：一位用户在尝试 text2text-generation 任务时遇到了 Meta-Llama-3-70B-Instruct API 错误。
   - 错误信息显示预期的模型类型不匹配，因此请求指导如何在 Hugging Face 上验证模型能力。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/fal/AuraSR">fal/AuraSR · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/gokaygokay/AuraSR">AuraSR - a Hugging Face Space by gokaygokay</a>: 未找到描述</li><li><a href="https://x.com/abhi1thakur/status/1813892464144798171">来自 abhishek (@abhi1thakur) 的推文</a>: 我们刚刚在 AutoTrain 中集成了数据集查看器 💥 现在，您可以在训练模型之前，在页面内直接查看数据集、识别正确的 split 和列 🚀</li><li><a href="https://tenor.com/view/unicorn-happy-birthday-dance-moves-gif-24459212">Unicorn Happy GIF - Unicorn Happy Birthday - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://imgur.com/dd3TB7g">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过幽默的笑话、流行的梗图、有趣的 gif、鼓舞人心的故事、病毒式视频等来提振精神...
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/)** (1 条消息): 

rp0101: https://youtu.be/N0eYoJC6USE?si=zms6lSsZkF6_vL0E
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1263373706856763514)** (7 条消息): 

> - `Transformers.js Sentiment Analysis Tutorial`
> - `Community Computer Vision Course Launch`
> - `AutoTrain for Machine Learning`
> - `Mistral NeMo Model Release` 


- **Transformers.js Sentiment Analysis 教程发布**：一个新的 [教程](https://huggingface.co/docs/transformers.js/en/tutorials/next) 展示了如何使用 Transformers.js 构建一个简单的 Next.js 应用程序进行 **sentiment analysis**，支持客户端和服务器端推理。
   - 最终产品包含一个 **demo**，可在 [客户端](https://huggingface.co/spaces/Xenova/next-example-app) 和 [服务器端](https://huggingface.co/spaces/Xenova/next-server-example-app) 应用程序中查看。
- **Community Computer Vision 课程现已开放**：**Community Computer Vision Course** 已启动，旨在深入探讨从基础到高级概念的 **computer vision** 应用。
   - 参与者可以学习如何提交作品并获得证书，同时加入以社区为中心的学习体验。
- **AutoTrain 简化自定义 ML 模型训练**：[AutoTrain](https://huggingface.co/autotrain) 平台允许用户通过简单上传数据来训练自定义机器学习模型，并实现模型选择过程的自动化。
   - 凭借快速部署选项，模型可立即在 [Hugging Face Hub](https://huggingface.co/models) 上使用，并满足包括 **LLM finetuning** 和 **image classification** 在内的各种任务。
- **Mistral NeMo 是新的基准模型**：Mistral 与 NVIDIA 合作推出了 **Mistral NeMo**，这是一款最先进的 12B 模型，具有 **128k context length**，并在 **Apache 2.0 license** 下发布。
   - 该公告是通过 [Twitter 帖子](https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw) 发布的，强调了其功能和重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/autotrain">AutoTrain – Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/learn/computer-vision-course/unit0/welcome/welcome">Welcome to the Community Computer Vision Course - Hugging Face Community Computer Vision Course</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/transformers.js/en/tutorials/next">Building a Next.js application</a>: 未找到描述</li><li><a href="https://x.com/mistralai/status/1813947930455499200?s=46&t=IfJRyr-UwyoM2m-vJODIzw">Tweet from Mistral AI (@MistralAI)</a>: https://mistral.ai/news/mistral-nemo/
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1263213216935317534)** (23 条消息🔥): 

> - `AI Comic Factory 更新`
> - `YouTube 转录工具`
> - `Sophi 生产力助手`
> - `CandyLLM 框架`
> - `水印去除工具` 


- **AI Comic Factory 引入默认对话气泡**：**AI Comic Factory** Space 已更新，现在默认提供对话气泡功能，尽管这仍是一个新功能，可能需要刷新才能正常工作。如果用户遇到问题，建议点击“redraw”或调整设置。
- **自动 YouTube 视频转录工具上线**：一款使用 Deepgram 和 Claude 开发的新工具已发布，可自动 **转录并总结 YouTube 视频**，目标用户为内容创作者和研究人员。用户可以[尝试该工具](https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)、[自定义模板](https://app.hunch.tools/app/canvas/new/vyg7V?invitationCode=u54c55ff)以及[阅读更多相关信息](https://hunch.tools/blog/video-transcription-and-summary-tool/)。
   - 该工具专为快速从视频内容中提取重要信息而定制，旨在提高生产力。
- **AI 生产力助手征求反馈**：一位开发者正在为其新的 **AI 生产力助手** 寻求反馈，该助手旨在通过集成多个平台来节省用户时间。该助手承诺每次查询至少节省 **15 分钟**，显著提升生产力。
   - 核心功能包括实时公共知识访问和主动提醒，以简化工作流程。
- **CandyLLM 框架发布**：介绍了一个名为 **CandyLLM** 的基础 Python 库，它使用 Gradio 作为用户界面。该项目已在 [GitHub](https://github.com/shreyanmitra/CandyLLM) 上开源，为 HuggingFace 和 OpenAI 的文本生成模型提供了一个易于使用的框架。
   - 虽然仍处于早期阶段，但创作者鼓励社区提供反馈和参与。
- **水印去除工具令用户印象深刻**：展示了一个基于 Florence 2 和 Lama Cleaner 构建的 **水印去除工具**，其处理非水印图像的能力给用户留下了深刻印象。该工具可以通过[此处](https://huggingface.co/spaces/DamarJati/Remove-watermark)访问，并邀请用户分享更多关于水印去除的资源。
   - 反馈强调该工具在不损害图像其他部分质量的情况下表现良好。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://app.hunch.tools/app/tool/yB85W?tpreview=true&invitationCode=u54c55ff)">Hunch - 面向团队的 AI 工具</a>：创建 AI 工作流和工具，实现知识工作自动化并提升团队生产力</li><li><a href="https://huggingface.co/spaces/jbilcke-hf/ai-comic-factory">AI Comic Factory - 由 jbilcke-hf 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://sophi.app/">Sophi.app — 通过应用集成和 AI，Sophi 助您完成工作</a>：🚀 智能、主动且可操作的问答引擎，理解您的数字生活并让您保持领先。</li><li><a href="https://huggingface.co/spaces/DamarJati/Remove-watermark">Remove-WM - 由 DamarJati 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/shreyanmitra/CandyLLM">GitHub - shreyanmitra/CandyLLM: 一个简单易用的 HuggingFace 和 OpenAI 文本生成模型框架。</a>：一个简单易用的 HuggingFace 和 OpenAI 文本生成模型框架。 - shreyanmitra/CandyLLM</li><li><a href="https://app.hunch.tools/app/canvas/new/vyg7V?invitationCode=u54c55ff)">Hunch - 面向团队的 AI 工具</a>：创建 AI 工作流和工具，实现知识工作自动化并提升团队生产力
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1263242131560988762)** (5 条消息): 

> - `项目演示时间线`
> - `适合初学者的论文`
> - `ML 模型层优化` 


- **项目演示推迟**：一位成员提到他们无法按时演示项目，预计将在三周后进行演示，以避开日程冲突。
   - *由于我认为那时没有演示活动，所以极有可能在 3 周后进行演示！*
- **寻找适合初学者的论文**：针对关于适合初学者论文的查询，一位成员推荐了 [huggingface.co/papers](https://huggingface.co/papers) 作为良好的资源。
   - 他们还提到 Yannic Kilcher 有一个专门用于每日论文讨论的 Discord 服务器。
- **ML 优化的基础论文**：一位成员询问在优化 ML 模型层（特别是 dense layers、GRU 和 LSTM GPU kernels）时，有哪些必读的基础论文或文章。
   - 这一咨询凸显了在模型优化领域建立职业生涯的兴趣。


  

---


### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/)** (1 条消息): 

dorbit_: 嘿！有人有使用 Transformers 进行相机标定（camera calibration）的经验吗？
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1263213639318634506)** (5 条消息): 

> - `Stable Video Diffusion 模型`
> - `文本分类挑战`
> - `使用 Transformers 和 Accelerate`
> - `多标签分类经验` 


- **Stable Video Diffusion 模型提示词**：一位用户分享了他们使用 [Stable Video Diffusion](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt) 模型的经验，该模型可以从静态图像生成视频。
   - 他们询问了有效的 *prompt engineering* 策略，以便从静态图像创建火箭移动的视频。
- **安装 Transformers 和 Accelerate**：一位成员建议在 Colab 中，可以通过运行 `!pip install transformers accelerate` 轻松导入 `transformers` 和 `accelerate`。
   - 这突出了一种为模型训练设置必要库的简便方法。
- **文本分类中的挑战**：一位用户就包含约 **200 个标签** 的大型数据集的文本分类寻求建议，并对处理其复杂性表示担忧。
   - 有人建议 *“你有这个的数据集吗？”*，并提到微调像 YOLO 这样更简单的模型可能是一个可行的选择。
- **多标签分类经验**：一位成员分享了他们使用单个模型管理具有数百个类别的 **multi-label classification** 的经验。
   - 这强调了使用统一方法处理大型标签集的潜在可行性，而不是为每个标签创建单独的模型。



**提到的链接**：<a href="https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt">stabilityai/stable-video-diffusion-img2vid-xt · Hugging Face</a>：未找到描述内容

  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1263232649317847091)** (6 条消息): 

> - `CUDA kernel splitting`
> - `CUDA graphs`
> - `Open source GPU kernel modules`
> - `Instruction tuning in LLMs`
> - `Flash attention reduction` 


- **关于拆分 CUDA kernel 的思考**：一位成员探讨了将一个 **CUDA kernel** 拆分为多个 kernel 的想法，用于处理 **flash attention** 中的多步 reduction 等任务，理由是在单步中管理内存存在困难。
   - 他们建议通过多次 kernel 启动来实现 **latency hiding** 可能会有益，尽管对其有效性表示不确定。
- **CNN 中多个 Kernel 的讨论**：另一位成员思考了在 **CNN** 中使用多个 kernel 的优势，其中较大的 kernel 尺寸需要跨层检查更多数据。
   - 他们警告说，如果尝试将各层融合（fuse）在一起，可能会产生不切实际的内存或寄存器需求。
- **询问 CUDA graphs 资料**：一位成员询问是否有专门针对 **CUDA graphs** 的讲座或资料，以进一步加强对该主题的理解。
   - 这为社区成员分享他们可能拥有的资源或见解打开了大门。
- **NVIDIA 转向开源 GPU Kernel Modules**：分享了一个链接，讨论 NVIDIA 决定将其 **Linux GPU kernel modules** 作为开源软件发布，从 2022 年 5 月开始采用 GPL 和 MIT 双重许可。
   - 该帖子强调了这些开源模块带来的改进和新功能，包括 **heterogeneous memory management**（异构内存管理）。
- **LLM 中的 Instruction Tuning 见解**：一位成员分享了近期关于大语言模型中 **instruction finetuning** 和 **LoRA** 论文的见解，强调了这些方法的实际意义。
   - 他们引用了一篇特定的文章，质疑了 **instruction tuning** 中的常见做法，突出了社区中关于有效方法的持续讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://developer.nvidia.com/blog/nvidia-transitions-fully-towards-open-source-gpu-kernel-modules/">NVIDIA Transitions Fully Towards Open&#x2d;Source GPU Kernel Modules | NVIDIA Technical Blog</a>：随着 R515 驱动程序的发布，NVIDIA 在 2022 年 5 月将一套 Linux GPU kernel modules 作为开源软件发布，采用 GPL 和 MIT 双重许可。初始版本针对数据中心计算 GPU……</li><li><a href="https://magazine.sebastianraschka.com/p/llm-research-insights-instruction">LLM Research Insights: Instruction Masking and New LoRA Finetuning Experiments</a>：讨论 2024 年 5 月最新的模型发布和 AI 研究。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1263251897318379634)** (1 条消息): 

> - `tl.pow`
> - `triton.language.extra.libdevice.pow` 


- **探讨 `tl.pow` 的缺失**：一位成员指出，在当前关于 Triton 功能的讨论中，缺失 `tl.pow` 是一个值得注意的点。
   - 作为回应，有人强调 `triton.language.extra.libdevice.pow()` 函数可以作为替代方案。
- **关于 Triton 语言特性的讨论**：社区继续围绕 Triton 中可用的特性和函数进行对话，特别是那些辅助数学运算的函数。
   - 成员们正在积极探索 `tl.pow` 等函数的冗余性，从而确定了有效编码的替代方案。


  

---

### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1263235922196172880)** (37 条消息🔥): 

> - `CUDA 中的动态共享内存 (Dynamic Shared Memory)`
> - `使用 Torch Compile 的稀疏模型指标`
> - `Torch-TensorRT 安装问题`
> - `使用 Triton Kernels 的自定义 Embedding 层` 


- **动态共享内存使用案例**：一位成员讨论了在短区域生成 profile，强调使用 **prefill 加上 2-3 个 tokens** 可以很好地概览模型的 batch 准备情况。
   - *另一位用户确认，与之前尝试的更大范围相比，仅使用 prefill 加上 5 个 tokens 就能看到更好的表现*。
- **稀疏模型准确率担忧**：尽管稠密模型达到了 **93.6% 的测试准确率**，但一位用户在应用剪枝技术后重新训练稀疏模型时遇到了性能崩溃。
   - 他们发现，在识别出 **torch.compile** 导致不准确后，启用 `torch._dynamo.config.guard_nn_modules=True` 解决了一些问题。
- **Torch-TensorRT 安装挑战**：一位用户报告在尝试安装 **torch-tensorrt** 时遇到错误，建议改用特定命令从 NVIDIA 的 GitHub releases 安装。
   - 建议包括从源码构建或尝试旧版本，因为错误可能源于不支持的 Python 版本，例如使用了 **3.12** 而不是 **3.10**。
- **为 nn.Embedding 定制 Triton Kernel**：一位用户注意到 **nn.Embedding** 的反向传播由于多次 kernel 启动而变慢，并表示有兴趣将其替换为融合的 triton kernel。
   - 有人建议创建一个**自定义 Embedding 层**，直接调用 triton kernels 以获得潜在的性能提升。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/">在 CUDA C/C++ 中使用共享内存 | NVIDIA 技术博客</a>: 在之前的文章中，我研究了一组线程对全局内存的访问如何合并为单个事务，以及对齐和步长如何影响各代产品的合并...</li><li><a href="https://leimao.github.io/blog/CUDA-Shared-Memory-Capacity/">CUDA 共享内存容量</a>: 使用大共享内存进行 CUDA Kernel 优化</li><li><a href="https://github.com/pytorch/pytorch/issues/124717">Compile 不保护用户 NN 模块属性 · Issue #124717 · pytorch/pytorch</a>: 🐛 描述 Bug：TorchTune 依赖于修改用户 NN 模块的属性来决定是否应用 LoRA 技术。使用模式如下：import contextlib import torch class ...</li><li><a href="https://github.com/NVIDIA/Torch-TensorRT/releases">Releases · pytorch/TensorRT</a>: 使用 TensorRT 的 NVIDIA GPU 版 PyTorch/TorchScript/FX 编译器 - pytorch/TensorRT
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1263499248289185953)** (1 条消息): 

> - `Google Gemma 2`
> - `Flash Attention 3`
> - `QGaLoRE`
> - `Mistral AI MathΣtral`
> - `CodeStral mamba` 


- **Google 发布 Gemma 2 系列模型**：**Gemma 2** 模型于 **2024 年 2 月**首次发布，提供 **2B** 和 **7B** 尺寸变体，据报道性能优于 **Llama 2** 系列。
   - 随后，**Llama 3**、**Qwen 2** 和 **MiniCPM 2** 的发布标志着 AI 的快速演进。
- **Flash Attention 3 让 GPU 运行更快**：**Flash Attention 3** 因显著提升 GPU 性能而受到关注，它融合了 **Flash Attention 1** 和 **2** 的见解。
   - 围绕这些进展的讨论表明，Flash Attention 的优化在每一次迭代中都在缩小性能差距。
- **QGaLoRE 彻底改变微调**：**QGaLoRE** 作为一种专注于量化低秩梯度的方法被引入，旨在改进模型的微调过程。
   - 社区对该技术表现出兴奋，强调了其在降低计算成本的同时增强模型效率的潜力。
- **MathΣtral 和 CodeStral mamba 引起轰动**：**Mistral AI** 的 **MathΣtral** 和 **CodeStral mamba** 作为最新发布的模型备受关注。
   - 成员们对其能力表示好奇，寻求有关如何将它们集成到现有工作流中的更多信息。



**提到的链接**: <a href="https://datta0.substack.com/p/aiunplugged-15-gemma-2-flash-attention">AIUnplugged 15: Gemma 2, Flash Attention 3, QGaLoRE, MathΣtral 和 Codestral Mamba</a>: 洞察胜过信息

  

---

### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1263315357272375407)** (6 条消息): 

> - `构建 CUTLASS 教程`
> - `使用 Nsight CLI` 


- **轻松构建 CUTLASS 教程**：一位用户询问如何构建并运行 [CUTLASS repo](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu) 中的 cute 教程。另一位成员回复说 *这只是一个 make 目标*，并表示如果其他人需要帮助可以提供支持。
- **Nsight CLI 资源查询**：一位成员询问了关于使用 **Nsight CLI** 的资源，以及如何将远程捕获的 profile 加载到 GUI 中进行分析。
   - 作为回应，另一位用户提到 *有一个选项可以将捕获的 profile 导出为文件，然后你可以通过 GUI 打开它*。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_1.cu">cutlass/examples/cute/tutorial/sgemm_1.cu at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账号为 NVIDIA/cutlass 的开发做出贡献。

  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1263237279691047043)** (2 条消息): 

> - `HF 相关讨论`
> - `FSDP 替代方案` 


- **聚集进行 HF 相关讨论**：几位成员表示希望在频道中讨论 **HF (Hugging Face) 相关内容**。
   - 这表明了对探索与 HF 相关的各种主题和进展的兴趣。
- **FSDP2 将取代 FSDP**：一位成员指出 **FSDP2** 将取代 **FSDP**，建议大家现在就开始采用 FSDP2，并以 **nf4** 为例。
   - 这一转变突显了框架的持续改进，一位成员提到他们很快会进一步深入研究。


  

---


### **CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1263211234145206394)** (2 条消息): 

> - `Triton 编译器功能`
> - `Triton Puzzles 解答`
> - `Triton 优化技术` 


- **Triton 编译器高效处理 GPU 代码**：**Triton** 编译器通过将其转换为 Triton IR，并进一步转换为 LLVM-IR / MLIR，最后通过 `libLLVM` 直接生成 PTX 等技术，将 Python 代码无缝转换为优化的 GPU 代码。
   - 它允许没有 CUDA 经验的用户编写出性能可与专家相媲美的代码。
- **关于 Triton Puzzle 解答的讨论**：一位成员分享了他们对 Triton Puzzles 的个人解答，并提供了其 [GitHub 仓库](https://github.com/alexzhang13/Triton-Puzzles-Solutions) 的链接供他人访问。
   - 他们表示虽然自己的解答可能写得不好，但很可能是正确的，并指出了 puzzle 12 中 **糟糕的符号表示 (notation)** 问题。
- **关于 Triton 优化的见解**：**Triton** 自动管理流式多处理器 (SMs) 内的优化，允许用户专注于任务划分，而无需担心详细的优化。
   - 提到了一篇博客文章，详细介绍了 Triton 的工作原理，包括从 Python 代码通过 AST 到最终编译的 GPU 代码的转换过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fkong.tech/posts/2023-04-23-triton-cuda/">Demystify OpenAI Triton</a>：通过分步说明和代码示例，学习如何构建从 OpenAI Triton 到 CUDA 的映射，以实现高性能深度学习应用。</li><li><a href="https://github.com/alexzhang13/Triton-Puzzles-Solutions">GitHub - alexzhang13/Triton-Puzzles-Solutions: Triton Puzzles 的个人解答</a>：Triton Puzzles 的个人解答。通过在 GitHub 上创建账号为 alexzhang13/Triton-Puzzles-Solutions 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1263264699085226056)** (159 条消息🔥🔥): 

> - `FP8 配置`
> - `模型训练改进`
> - `内存管理重构`
> - `训练中的量化感知`
> - `CUDA 优化策略` 


- **FP8 配置中的挑战**：成员们讨论了运行 FP8 训练的指令，强调了为确保 checkpoint 兼容性而进行特定配置设置的必要性。
   - 某些设置组合导致无法成功配置，其中的不确定性突显了 FP8 集成的复杂性。
- **FP32 模型的重大变化**：发布了一个大型 Pull Request，对 FP32 训练代码进行了重大重构和改进，有望简化 Kernel 操作。
   - 尽管改动广泛，成员们对该 PR 的兼容性表示有信心，并认为未来发生合并冲突的可能性降低了。
- **内存管理设计改进**：讨论了一项通过消除重复分配来整合内存管理的提案，旨在使模型和训练操作中的分配实践更加清晰。
   - 对话内容包括考虑根据命令行参数简化 activation 分配，以增强代码清晰度。
- **探索训练中的量化感知**：研究了经过量化感知训练的 Kernel，重点关注 Character.AI 通过使用 INT8 训练来提高推理性能的方法。
   - 针对量化感知实现的细节提出了疑问，特别是那些承诺在没有传统开销的情况下提升性能的方法。
- **CUDA 优化讨论**：成员们分享了关于 CUDA Kernel 优化的见解，讨论了使用 `auto` 关键字和模板可见性对提高代码可读性的影响。
   - 辩论了矩阵乘法 Kernel 的优化，成员们建议依赖厂商（vendor）实现可能会获得更好的性能结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个最先进的 12B 模型，具有 128k 上下文长度，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://research.character.ai/optimizing-inference/">Optimizing AI Inference at Character.AI</a>: 在 Character.AI 优化 AI 推理：在 Character.AI，我们正致力于实现 AGI。在未来的愿景中，大语言模型 (LLMs) 将增强日常生活，提供商业生产力和娱乐，并帮助人们...</li><li><a href="https://pytorch.org/blog/accelerating-neural-network-training/">Accelerating Neural Network Training with Semi-Structured (2:4) Sparsity</a>: 使用半结构化 (2:4) 稀疏性加速神经网络训练：在过去的一年里，我们在 PyTorch 中增加了对半结构化 (2:4) 稀疏性的支持。只需几行代码，我们就能够通过... 在 segment-anything 上展示 10% 的端到端推理加速。</li><li><a href="https://github.com/karpathy/llm.c/pull/696">Major FP32 llm.c improvements/refactoring/etc. by ademeure · Pull Request #696 · karpathy/llm.c</a>: ademeure 对 FP32 llm.c 的重大改进/重构等：我有点做过头了，这最终显著改变了 train_gpt2_fp32.cu 中的几乎每一个 Kernel！我还给 Kernel 添加了很多注释——可能太多了，但如果...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[lecture-qa](https://discord.com/channels/1189498204333543425/1238926773216084051/1263246495432577086)** (9 条消息🔥): 

> - `CUDA 中的深拷贝 (Deep Copy)`
> - `Kernel 参数限制`
> - `量化组大小 (Quantization Group Size)`
> - `CPU 与 GPU 之间的内存复制` 


- **关于第 6 课中深拷贝的困惑**：一位成员对在 device 代码中使用任何 host 数据时进行**深拷贝 (deep copy)**数据的常见做法表示困惑，并强调了 **4k kernel 参数限制**。
   - 另一位成员澄清说，要从 device 代码访问大型结构体，必须将指针传递给 GPU 上的大型内存缓冲区。
- **理解 CUDA 中的指针用法**：讨论明确了在使用 **cudaMalloc** 时，内存指针必须位于 GPU 内存中，且 CPU 上的任何数据在调用 kernel 之前都必须复制过去。
   - 提到一个交替使用 ** 和 * 的提案引发了关于内存分配的困惑，因为目前尚不清楚为什么它没有按预期工作。
- **关于量化组大小 (Quantization Group Size) 的问题**：一位成员询问了量化讲座中提到的与困惑度 (perplexity) 变化相关的 **group size** 一词，寻求其定义的澄清。
   - 另一位成员解释说，**group size** 指的是有多少个量化值共享一个缩放参数 (scale parameter)，这会影响内存效率和量化质量。
- **量化中的内存考量**：讨论认为量化参数需要在节省内存与量化误差之间取得平衡，更多数值共享一个 scale 可以提高内存效率。
   - group size 与质量之间的权衡被认为是各种量化库的一个关键方面，对性能有重要影响。


  

---



### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1263210519997845605)** (213 条消息🔥🔥): 

> - `Stable Diffusion 模型问题`
> - `Adobe Stock 内容政策变更`
> - `AI 工具中的放大器 (Upscaler) 选项`
> - `社区互动与辩论` 


- **Stable Diffusion 模型的问题**：一位用户报告了在 Stable Diffusion 中使用模型的困难，称上传模型后无法创建图像。
   - 另一位用户澄清说，Stable Diffusion 需要模型才能运行，并询问了该用户设置的更多细节。
- **Adobe Stock 收紧内容政策**：Adobe Stock 更新了关于艺术家姓名的政策，这可能会导致在 Gen AI 项目中引用这些姓名的内容被删除。
   - 用户对与艺术家姓名相关的版权声明表示沮丧，特别提到像伦勃朗 (Rembrandt) 这样的艺术家可能并不拥有有效的版权。
- **AI 工具中的放大器 (Upscaler) 功能**：关于 AI 工具中放大选项的可用性和命名规范的讨论，特别提到了 'Hires Upscaler'。
   - 用户正在寻求澄清和技巧，以便有效地让他们的生成艺术作品通过 Adobe 等平台的审核。
- **社区参与和趣闻**：聊天中出现了一些幽默的交流，用户们拿涉及社区成员的 '爆米花时刻' 开玩笑，并进行俏皮的打趣。
   - 用户讨论了如何在保持轻松氛围的同时，有效处理特定的内容审核问题。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/multimodalart/AuraFlow">Auraflow Demo - multimodalart 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://tenor.com/view/dies-from-cringe-meme-cringe-imagine-gif-23477312">Dies From Cringe Meme GIF - Dies From Cringe Meme Cringe - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://civitai.com/articles/4248```">什么是 score_9 以及如何在 Pony Diffusion 中使用它 | Civitai</a>: 对下一版 Pony Diffusion 感兴趣？在此阅读更新：https://civitai.com/articles/5069/towards-pony-diffusion-v7 你可能见过 score_9...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1263301231682523252)** (1 条消息): 

> - `GoldFinch`
> - `Hybrid Attention Models`
> - `KV-Cache Optimization`
> - `Finch-C2`
> - `GPTAlpha` 


- **GoldFinch 在 AI 领域引起轰动**：新的 **GoldFinch** 模型结合了 **Linear Attention** 和传统的 Transformer，解决了 **quadratic slowdown**（二次方减速）和 KV-Cache 过大等关键问题，允许在消费级硬件上实现极长的上下文长度。
   - *在实验中，GoldFinch 在下游任务上的表现优于同级别的 1.5B Llama 和 Finch (RWKV-6) 模型。*
- **GoldFinch 论文详情展示了令人印象深刻的特性**：[GoldFinch 论文](https://arxiv.org/abs/2407.12077) 介绍了一种高效的 **KV-Cache** 生成技术，该技术呈线性扩展，显著减小了 Cache 大小以实现更快的推理。
   - 这一进步使模型能够处理极大的文本输入，与传统的 Transformer 相比，其 Cache 大小缩小了 **756-2550 倍**。
- **Finch-C2 提升性能**：**Finch-C2** 作为 Finch (RWKV-6) 架构的高性能版本发布，进一步增强了下游任务的性能。
   - 它旨在为那些在不需要大量硬件资源的情况下寻求强劲性能的用户提供替代方案。
- **介绍 GPTAlpha 模型**：**GPTAlpha** 架构通过 RWKV 组件增强了传统的 Transformer，在应用 Softmax Attention 的同时实现了更好的整体性能。
   - 该模型展示了 Transformer 策略的演进，融合了提高效率和有效性的新方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.12077">GoldFinch: High Performance RWKV/Transformer Hybrid with Linear Pre-Fill and Extreme KV-Cache Compression</a>：我们介绍了 GoldFinch，这是一种混合 Linear Attention/Transformer 序列模型，它使用一种新技术，在处理线性时间和空间复杂度的同时，高效地生成高度压缩且可重复使用的 KV-Cache...</li><li><a href="https://github.com/recursal/GoldFinch-paper">GitHub - recursal/GoldFinch-paper: GoldFinch and other hybrid transformer components</a>：GoldFinch 和其他混合 Transformer 组件。通过创建账号为 recursal/GoldFinch-paper 的开发做出贡献。</li><li><a href="https://huggingface.co/recursal/GoldFinch-paper">recursal/GoldFinch-paper · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1263248892200484946)** (72 条消息🔥🔥): 

> - `AI Scraping Controversy`
> - `YouTube Subtitles Usage`
> - `Copyright Law and Content Usage`
> - `Community Project Opportunities`
> - `Ethics in Data Scraping` 


- **AI 抓取争议升级**：成员们讨论了对 AI 抓取数据的**愤怒**，特别是关于 **YouTube 字幕**的使用，认为这种抵制情绪显得有些夸张和愚蠢。
   - *一位成员表示，与围绕字幕的争议相比，艺术家和作家面临的问题更值得关注。*
- **呼吁对数据使用进行法律改革**：存在一种强烈的**版权改革**情绪，强调需要为数据可能被抓取的艺术家和作家制定更好的**署名 (attribution)** 和**授权 (accreditation)** 法律。
   - *人们担心公司在没有获得适当许可或提供补偿的情况下，从个人数据中获利。*
- **寻求社区项目参与**：一位成员询问是否有机会加入与 **LLM 和 NLP** 相关的社区项目，但发现许多频道并不活跃。
   - *另一位成员指出，该领域大多数活跃的贡献者似乎更关注高薪的技术工作，而非公共项目。*
- **围绕基准版权与数据抓取的讨论**：成员们辩论了 **Google** 及类似平台是否应该为使用公开数据付费，类似于搜索引擎的运作方式。
   - *他们指出，搜索结果通常不会削弱原始内容的经济价值，从而支持更广泛的数据使用。*
- **抓取伦理与社区观点**：对话强调了对**抓取伦理**认知的挫败感，认为公众的愤怒似乎更多地集中在开源项目上，而非大型企业。
   - *几位成员认为目前的讨论缺乏平衡，更多地集中在对开源组织的攻击上。*


  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1263216620621266944)** (108 条消息🔥🔥): 

> - `ICML 2024`
> - `Patch-Level Training`
> - `Learning Rate Schedules`
> - `Language Model Efficiency`
> - `Cognitive Architectures for Language Agents`

- **对 ICML 2024 演讲的期待**：成员们分享了参加 ICML 2024 的兴奋之情，其中一人展示了一篇关于 **Protein Language Models** 的论文，该模型在区分病毒蛋白质方面达到了 **99.7% ROCAUC**。
   - 讨论还涉及了演讲缺乏视频选项的问题，重点转向了 poster 上传。
- **Patch-Level Training 优化 LLMs**：讨论了 **patch-level training** 的引入，强调了其通过将多个 tokens 压缩到单个 patch 中来降低训练成本的效率。
   - 持续的讨论研究了该阶段 learning rates 的益处，以及通过修改来提高性能的潜力。
- **正在调查中的 Learning Rate 调整**：针对 patch-level training 期间重置 learning rates 的问题提出了担忧，并建议保持其稳定以获得更好的效率。
   - 成员们正在尝试不同的 learning rate schedules，并收集关于其有效性的实证证据。
- **Multi-Token Prediction 的影响**：探索了使用独立 heads 同时预测多个 tokens 的方法，作为提高训练效率的一种方式，尽管初步结果并不理想。
   - 反馈表明，新参数可能会使不同训练模式之间的转换变得复杂，需要进一步的实验。
- **面向 Language Agents 的 CoALA 框架**：提到了一篇介绍 **Cognitive Architectures for Language Agents (CoALA)** 框架的新论文，旨在组织各种 language agents 的能力。
   - 这一系统性框架借鉴了认知科学，以规划 language agent 功能的未来发展。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2309.02427">Cognitive Architectures for Language Agents</a>: 《语言智能体的认知架构》：最近的研究通过外部资源（如互联网）或内部控制流（如 Prompt Chaining）增强了大语言模型（LLMs），以应对需要 Grounding 或推理的任务...</li><li><a href="https://arxiv.org/abs/2407.12665">Patch-Level Training for Large Language Models</a>: 《大语言模型的 Patch 级训练》：随着大语言模型（LLMs）在语言理解和生成方面取得显著进展，其训练效率已成为一个关键问题。传统上，LLMs 被训练用于预测...</li><li><a href="https://arxiv.org/abs/2404.19737">Better &amp; Faster Large Language Models via Multi-token Prediction</a>: 《通过 Multi-token 预测实现更好、更快的语言模型》：像 GPT 和 Llama 这样的大语言模型是使用 Next-token 预测损失进行训练的。在这项工作中，我们建议训练语言模型一次预测多个未来的 Token 会导致...</li><li><a href="https://arxiv.org/abs/2405.18392">Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations</a>: 《超越固定训练时长的扩展定律与计算最优训练》：规模已成为获得强大机器学习模型的主要因素。因此，了解模型的 Scaling 特性是有效设计正确训练设置的关键...</li><li><a href="https://arxiv.org/abs/2402.04362">Neural Networks Learn Statistics of Increasing Complexity</a>: 《神经网络学习复杂度递增的统计数据》：分布简单性偏差（DSB）假设神经网络首先学习数据分布的低阶矩，然后再转向高阶相关性。在这项工作中，我们展示了...</li><li><a href="https://x.com/JJitsev/status/1813930981637902486">Tweet from Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev)</a>: Jenia Jitsev (@JJitsev) 的推文：又一个关于兴衰的故事：最近，NuminaMath-7B 在 AIMO 竞赛中排名第一，解决了 50 道奥数级别私有测试集中的 29 道。它能处理简单的 AIW 问题吗，这需要...</li><li><a href="https://github.com/shaochenze/PatchTrain">GitHub - shaochenze/PatchTrain: Code for paper &quot;Patch-Level Training for Large Language Models&quot;</a>: GitHub - shaochenze/PatchTrain：论文《大语言模型的 Patch 级训练》的代码</li><li><a href="https://github.com/RulinShao/retrieval-scaling">GitHub - RulinShao/retrieval-scaling: Official repository for &quot;Scaling Retrieval-Based Langauge Models with a Trillion-Token Datastore&quot;.</a>: GitHub - RulinShao/retrieval-scaling：《使用万亿 Token 数据存储扩展基于检索的语言模型》的官方仓库。</li><li><a href="https://openreview.net/forum?id=gGnJBLssbb&noteId=gGnJBLssbb">Protein language models expose viral mimicry and immune escape</a>: 《蛋白质语言模型揭示病毒模拟与免疫逃逸》：病毒通过分子模拟规避免疫系统，采用其宿主的生物物理特征。我们调整了蛋白质语言模型（PLMs）来区分人类和病毒...</li><li><a href="https://github.com/ddofer/ProteinHumVir">GitHub - ddofer/ProteinHumVir: Code &amp; data for &quot;Protein Language Models Expose Viral Mimicry and Immune Escape&quot;</a>: GitHub - ddofer/ProteinHumVir：论文《蛋白质语言模型揭示病毒模拟与免疫逃逸》的代码和数据</li><li><a href="https://doi.org/10.1101/2024.03.14.585057">Protein Language Models Expose Viral Mimicry and Immune Escape</a>: 《蛋白质语言模型揭示病毒模拟与免疫逃逸》：动机：病毒通过分子模拟规避免疫系统，采用其宿主的生物物理特征。我们调整了蛋白质语言模型（PLMs）来区分人类和病毒...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1263598332773797889)** (1 messages): 

> - `Tokenization-free language models`
> - `Interpretability issues` 


- **关于 Tokenization-Free 语言模型的辩论**：成员们正在讨论 **Tokenization-free 语言模型** 会改善还是阻碍**可解释性 (Interpretability)**。
   - *有人担心消除 Tokenization 可能会导致对模型中语言处理过程的理解不够细致。*
- **Tokenization-Free 方法的潜在益处**：一些成员认为移除 Tokenization 可以简化模型结构，增强对输出和行为的**解释 (Interpretation)**。
   - 他们建议模型可以在没有 Token 边界限制的情况下，以更**自然**的方式表达复杂的想法。


  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1263227270403981323)** (14 messages🔥): 

> - `lm-eval-harness predict_only flag`
> - `LoraConfig size mismatch`
> - `PR Review for Gigachat model`
> - `Model evaluation methods`
> - `System instruction behavior` 

- **lm-eval-harness predict_only 标志查询**：一名成员询问关于在 lm-eval-harness 中使用 `--predict_only` 标志，以便在生成补全（completions）后运行指标（metrics）计算的问题。
   - *另一名成员提到这在未来更新的待办事项（todo list）中优先级很高*。
- **LoraConfig 运行时错误困惑**：一名成员报告了在尝试使用特定设置进行微调时，遇到的与 LoraConfig 尺寸不匹配相关的 `RuntimeError`。
   - 另一名成员询问了 `lm_eval` 的版本，并建议在 **0.4.3** 版本之前已经进行了相关修复。
- **Gigachat 模型的 PR 评审请求**：一名成员请求对其 PR 进行评审，该 PR 通过使用带有聊天模板（chat templates）的 API 添加了新的 **Gigachat model**。
   - 另一名成员表示感谢，并提到由于任务积压，会尽快进行评审。
- **模型评估方法讨论**：一名成员询问在使用 lm_eval 时，在同一个 Hugging Face 模型实例上复现分数是否是验证正确性的有效基准（benchmark）。
   - 另一名成员澄清说，不同实现之间的分数应该相当接近，并承认可能存在数值差异。
- **任务中的 System instruction 处理**：一名成员询问通过 `--system_instruction` 传递系统消息是否与 task.yaml 中 `description` 字段的消息具有相同的效果。
   - 另一名成员确认两者的处理方式类似，系统提示词（system prompt）会与已有的描述（description）拼接在一起。

**提到的链接**：<a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1996">Add Gigachat model by seldereyy · Pull Request #1996 · EleutherAI/lm-evaluation-harness</a>：使用 API 和聊天模板向库中添加一个新模型。对于身份验证，请为您的 API auth_data 设置环境变量 "GIGACHAT_CREDENTIALS" 和 "GIGACHAT_SCOPE"...

---

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1263210402821574676)** (59 条消息🔥🔥): 

> - `LM Studio and Model Support` (LM Studio 与模型支持)
> - `Model Performance Comparisons` (模型性能对比)
> - `Temperature and Configuration Settings` (Temperature 与配置设置)
> - `Mistral-Nemo Release` (Mistral-Nemo 发布)
> - `Context Length Issues` (上下文长度问题)


- **LM Studio 在支持 DeepSeek Coder V2 Lite 时遇到困难**：用户报告了在 LM Studio 中使用 **DeepSeek Coder V2 Lite** 模型的问题，特别是尽管日志显示加载成功，但其架构仍不被支持。
   - 一位使用 **NVIDIA GeForce RTX 2080** 的用户注意到服务器响应与模型能力之间存在差异。
- **理解 API 参数优先级**：讨论显示，LM Studio 在生成响应时通常优先考虑客户端参数设置（如 **temperature**），除非超出了上下文大小限制。
   - 一位用户确认，尽管有客户端设置，特定提示词仍产生了一致的响应，这引发了关于参数合规性的疑问。
- **Temperature 设置对输出的影响**：一位用户指出，调整 **GPU layer** 设置解决了 **phi-3** 模型输出乱码的问题，这表明模型对硬件设置具有敏感性。
   - 他们请求对此过程进行深入了解，并提到了之前在其他模型上的类似经历。
- **Mistral-Nemo 模型发布详情**：最近公布了与 **NVIDIA** 合作开发的 **Mistral-Nemo 12B** 模型，其具备高达 **128k tokens** 的上下文窗口。
   - 该模型被定位为 **Mistral 7B** 的直接替代方案，旨在其尺寸类别中提供最先进的性能。
- **对 LM Studio 中 RAG 的需求**：用户正在寻求在运行 **LM Studio server** 时使用 **RAG** 的指导，表明了对集成更高级运行时功能的兴趣。
   - 对话突显了社区对于有效实施这些配置的资源和支持的需求。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://lmstudio.ai/docs/lmstudio-sdk/examples">Code Examples | LM Studio</a>: 如何使用 LM Studio JavaScript/TypeScript SDK 的示例</li><li><a href="https://lmstudio.ai/docs/local-server#supported-payload-parameters">Local LLM Server | LM Studio</a>: 你可以通过在 localhost 上运行的 API 服务器使用在 LM Studio 中加载的 LLM。</li><li><a href="https://huggingface.co/TheDrummer/Smegmma-9B-v1">TheDrummer/Smegmma-9B-v1 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1263483251448872960)** (23 messages🔥): 

> - `DeepSeek-V2-Chat-0628`
> - `LM Studio Support`
> - `Mistral NeMo`
> - `Open-source LLMs and China`
> - `Verbose AI Models` 


- **DeepSeek-V2 受到关注**：[DeepSeek-V2](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 得到了积极讨论，一名成员向其背后的团队表示祝贺。
   - 他们提供了其主页、聊天界面和其他资源的链接，强调了其开源特性。
- **LM Studio 与 DeepSeek-V2 的兼容性**：一位用户询问 LM Studio 是否将支持 DeepSeek-V2-Chat-0628，讨论指出这取决于 [llama.cpp 的支持](https://link.to/llama-support)。
   - 对话强调了各种 AI 模型的持续开发和集成。
- **Mistral NeMo 引起轰动**：[Mistral NeMo](https://mistral.ai/news/mistral-nemo/)（一个 12B 模型）的发布因其 **128k tokens** 的超大上下文窗口和出色的性能而受到赞誉。
   - 与其他模型相比，关于其逻辑推理能力的评价褒贬不一，激起了用户的好奇心。
- **对中国使用开源 LLM 的担忧**：一位成员对中国利用开源 LLM 表示怀疑，理由是严格的数据政策。
   - 社区讨论了潜在的影响，指出了中国广泛的数据收集和缺乏 DRM 法律的情况。
- **AI 模型表现冗长**：用户分享了 AI 模型在回答中过于冗长的经历，特别是在逻辑推理问题上。
   - 其中一个回答被强调为过于详细但逻辑严密，展示了详尽与简洁之间的平衡。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个最先进的 12B 模型，具有 128k 上下文长度，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628">deepseek-ai/DeepSeek-V2-Chat-0628 · Hugging Face</a>: 未找到描述</li><li><a href="https://build.nvidia.com/nv-mistralai/mistral-nemo-12b-instruct">NVIDIA NIM | mistral-nemo-12b-instruct </a>: 立即体验领先模型，构建企业级生成式 AI 应用。
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/)** (1 messages): 

xoxo3331: 无法通过 CLI 使用参数或标志加载带有预设（preset）的模型。
  

---


### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1263534463711711287)** (1 messages): 

> - `Meta Llama 3 Instruct 7B Q8`
> - `Stock trading strategies`
> - `Market analysis` 


- **协作式交易策略讨论**：一位成员分享了他们针对 Meta Llama 3 的 Prompt，强调将其作为交易伙伴来分析股票交易策略。
   - 他们详细说明了对特定交易进行联合市场分析的需求，包括对提议策略和资金分配的评估。
- **交易提案中对风险管理的强调**：该 Prompt 强调了在讨论交易策略和资金分配时，管理投资组合风险承受能力的重要性。
   - 鼓励成员在执行交易前评估潜在的风险和回报，重点关注商定的风险水平。


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1263559049560064051)** (1 messages): 

> - `Llama-3-Groq-8B`
> - `LM Studio Presets`
> - `AutoGen Cases` 


- **探索 Llama-3-Groq-8B 的工具使用**：一位成员在讨论其实现时引用了 [Llama-3-Groq-8B Tool-Use GitHub 仓库](https://github.com/MaziyarPanahi/Llama-3-Groq-8B-Tool-Use-GGUF)。
   - 他们表示有兴趣了解默认的 **LM Studio 预设** 是否与 **AutoGen 案例** 兼容。
- **关于 AutoGen 兼容性的查询**：对话转向现有的 **LM Studio 预设** 是否能在 **AutoGen 案例** 中正常运行。
   - 这个问题引发了关于为获得更好性能而进行潜在优化和配置调整的讨论。


  

---

### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1263214903347515484)** (23 条消息🔥): 

> - `Xeon Specs` (Xeon 规格)
> - `Resizable BAR on LLMs` (LLM 上的 Resizable BAR)
> - `GTX 1050 Performance Issues` (GTX 1050 性能问题)
> - `LM Studio ROCM Version` (LM Studio ROCM 版本)
> - `DIY Hardware Cooling Concerns` (DIY 硬件散热顾虑)


- **Xeon 规格揭晓**：一位成员详细介绍了他们的配置，包括 **2x Xeon 8 core 3.5GHz**、**32GB ECC RAM** 和一块 **P40 GPU**，并配备了优质的 **1300W power supply**。
   - 他们的目标是在单卡上以不错的速度运行 **7/13B model**，强调比起模型大小更看重速度。
- **Resizable BAR 对 LLM 的影响**：关于 **Resizable BAR** 是否影响 LLM 性能的问题引发了讨论，共识是*推理速度没有提升*。
   - 另一位成员指出，虽然 ReBAR 不影响显存速度，但它可能会影响模型加载和多 GPU 性能。
- **GTX 1050 GPU 使用率困扰**：一位用户报告称，尽管将配置设置为 **50% CPU/GPU**，他们的 **GTX 1050** GPU 使用率仅达到 **10%**。
   - 建议包括测试更小的模型，并确保 **GPU Offload** 已**禁用**（disabled）以获得更好的性能。
- **关于 LM Studio ROCM 版本的讨论**：针对最新版 **LM Studio ROCM** 的咨询得到了澄清，它不再是 **0.2.24** 版本，并指向了一条特定的消息以获取更新。
   - 这表明社区内围绕 ROCM 版本的开发和更新正在持续进行。
- **对 DIY 硬件散热的担忧**：一位成员担心 **Xeons** 可能过热，如果在无人看管的情况下可能会损坏木制支架。
   - 作为回应，另一位成员保证*存在空气间隙*可以防止热损伤。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1263296398909112471)** (3 条消息): 

> - `Beta Enrollment Feedback` (Beta 报名反馈)
> - `Beta Access Criteria` (Beta 访问权限标准)
> - `Public Beta Release Timeline` (公测版发布时间线) 


- **Beta 报名引发疑问**：一位用户对他们的 '0.3.0 Beta' 状态表示困惑，指出他们的 Beta 报名消息没有得到回复。
   - 另一位用户提到“*我也是*”，并猜测可能采用了分阶段邀请流程，以便在全面发布前收集反馈。
- **Beta 访问权限的积极参与度**：另一位成员推测 '0.3.0 Beta' 权限主要授予了聊天频道中的“活跃”参与者。
   - 他们表示版主可能会发布公告，暗示未来几天会有更多邀请。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1263543622712823848)** (4 条消息): 

> - `AMD RDNA Compatibility` (AMD RDNA 兼容性)
> - `CUDA on AMD with ZLUDA` (通过 ZLUDA 在 AMD 上运行 CUDA)
> - `SCALE's New Release` (SCALE 的新版本)
> - `Portable Install Options` (便携式安装选项) 


- **针对 Navi 显卡的 AMD 编译器备受关注**：一款针对 **AMD cards** (Navi 31 和 21) 的新编译器已经发布，报告显示 **RX 7800** 表现良好，甚至 **RDNA 1** 也支持它。
   - 成员们表示有兴趣在 **llama.cpp** 中进行测试，看看它的表现是否优于 **ROCm implementation**。
- **ZLUDA 实现 AMD 原生运行 CUDA**：ZLUDA 被提及为一种允许 **CUDA** 在 **AMD** 硬件上原生运行的工具，尽管它似乎尚未集成到 **llama.cpp** 中。
   - 这种情况为 **AMD users** 在使用 **llama.cpp** 时留下了测试和优化的空间。
- **SCALE 带着类似功能出现**：成员们注意到几天前发布的 **SCALE** 提供了类似于 ZLUDA 的功能，用于在 AMD 系统上运行 CUDA 任务。
   - 这一进展可能为 AMD 用户增强其在 CUDA 环境下的体验提供另一个选择。
- **对便携式安装选项的需求增加**：一位成员表达了对 **portable install options** 的需求，以方便这些工具的部署。
   - 此类选项可以提高用户在处理 **AMD** 配置时的可访问性和灵活性。



**提及的链接**：<a href="https://www.reddit.com/r/LocalLLaMA/comments/1e6cxef/cuda_on_amd_rdna_3_and_rdna_2_new_release/">Reddit - Dive into anything</a>：未找到描述

  

---

### **LM Studio ▷ #[model-announcements](https://discord.com/channels/1110598183144399058/1225909444727013466/1263525794945175563)** (1 条消息): 

> - `Groq's tool use models`
> - `Berkeley Function Calling Leaderboard` 


- **Groq 的 Tool Use 模型准备就绪**：Groq 的 Tool Use 模型已开发完成并开放使用，在 [Berkeley Function Calling Leaderboard](https://huggingface.co/lmstudio-community/Llama-3-Groq-8B-Tool-Use-GGUF) 上展示了令人印象深刻的结果。**8b** 模型获得了 **89.06%** 的评分，**70b** 模型获得了 **90.76%** 的评分。
   - 这些模型预计将增强任何包含 Tool Use 和 Function Calling 的 Pipeline。
- **在 Function Calling 排行榜上获得高分**：Groq 的 **8b** 和 **70b** 模型在 [Berkeley Function Calling Leaderboard](https://huggingface.co/lmstudio-community/Llama-3-Groq-70B-Tool-Use-GGUF) 上表现优异，得分分别为 **89.06%** 和 **90.76%**。这一表现标志着它们在 Function Calling 和工具集成方面的熟练程度。
   - 它们的设计使其特别适用于依赖高效 Tool Use 的应用。


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1263233006932590602)** (14 条消息🔥): 

> - `Hosting Models Online`
> - `Using Ngrok for Access`
> - `Tailscale for Secure Access`
> - `Frontend Development Needs`
> - `Dedicated Model Hosting Plans` 


- **探索模型的在线托管**：一位用户正在寻求关于如何在 Windows 或 WSL 上与在线好友分享其本地托管模型的建议。
   - 他们正在寻找一种允许模型同时供多个用户测试的方法。
- **Ngrok 作为潜在解决方案**：Ngrok 被提及作为为本地服务器创建公共 URL 的工具，如果它是免费的，用户计划尝试一下。
   - 他们还提到自己在 Linux 上配置 Nginx 的经验，但不确定 Windows 上的设置。
- **使用 Tailscale 进行安全访问**：另一位用户建议使用 Tailscale，让朋友能够安全地访问托管在 WSL 上的应用，而无需处理路由器配置。
   - 他们强调了与传统方法相比，Tailscale 对移动用户和 IP 变动的便利性。
- **对前端开发帮助的需求**：用户表达了对前端开发协助的需求，以创建一个支持用户身份验证和独立聊天的网站。
   - 鉴于目前的技能水平，他们正考虑雇人帮助构建项目的前端。
- **对未来增长的愿景**：用户分享了为大量用户托管模型的长期愿景，甚至提到了潜在的社交媒体引流（Funneling）。
   - 他们的目标是提供比现有解决方案更好的体验，同时也表达了对 OpenAI 局限性的担忧。


  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1263257584836804701)** (2 messages): 

> - `TextGrad 优化`
> - `STORM 写作系统`
> - `AI 在文章生成中的应用`
> - `长篇写作中的挑战` 


- **TextGrad 提供引人注目的优化方法**：[TextGrad 论文](https://arxiv.org/abs/2406.07496)介绍了一个框架，利用来自 LLM 的文本反馈在神经网络中执行自动微分，旨在优化复合 AI 系统中的组件。
   - *AI 正在经历范式转移*，研究人员正在探索优化神经网络的这一新领域，其潜在应用前景令人兴奋。
- **STORM 系统增强文章写作**：[STORM 论文](https://arxiv.org/abs/2402.14207)提出了一种新颖的写作系统，利用大语言模型生成有据可查、组织严整的长篇文章，类似于维基百科条目。
   - 通过进行*多视角提问*，STORM 在感知组织性方面比传统方法实现了 **25% 的绝对提升**。
- **长篇有据文章中的挑战**：STORM 系统还解决了诸如*源偏见转移*以及生成内容中可能出现的**无关事实过度关联**等挑战。
   - 根据资深维基百科编辑的反馈，这些障碍强调了对 AI 生成写作进行持续改进的必要性。
- **探索 AI 与人类的协作**：分享了一个链接，讨论了 *AI 与人类关系* 为社会带来的复杂性，强化了在写作过程中谨慎实施 AI 的重要性。
   - 对话强调了围绕 AI 能力进步所带来的社会影响以及必要调整的持续讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.07496">TextGrad: Automatic &#34;Differentiation&#34; via Text</a>：AI 正在经历范式转移，通过编排多个大语言模型 (LLMs) 和其他复杂组件的系统实现了突破。因此，开发原则性且自动化的...</li><li><a href="https://arxiv.org/abs/2402.14207">Assisting in Writing Wikipedia-like Articles From Scratch with Large Language Models</a>：我们研究如何应用大语言模型从零开始撰写有据可查且组织严整的长篇文章，其广度和深度可与维基百科页面相媲美。这个尚未被充分探索的问题提出了新的...</li><li><a href="https://storm.genie.stanford.edu/article/ai-human-relations-and-the-complexity-it-introduces-to-society-18731">未找到标题</a>：未找到描述</li><li><a href="https://github.com/stanford-oval/storm">GitHub - stanford-oval/storm: An LLM-powered knowledge curation system that researches a topic and generates a full-length report with citations.</a>：一个由 LLM 驱动的知识策展系统，可研究特定主题并生成带有引用的完整报告。 - stanford-oval/storm
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[datasets](https://discord.com/channels/1053877538025386074/1105324249721356298/1263524936316747807)** (1 messages): 

> - `合成数据集`
> - `AI 知识库` 


- **Mill Pond Research 发布 AI 知识库**：名为 [AI Knowledge Base](https://github.com/Mill-Pond-Research/AI-Knowledge-Base) 的数据集为专注于检索增强生成 (RAG) 的 **AI 系统提供了一个全面的通用知识库**。
   - 该项目旨在收集和组织基础知识与见解，从而实现更有效的 AI 开发和研究。
- **关注商业应用**：成员们讨论了 AI Knowledge Base **以业务为中心**的性质，强调了其在各个行业中的潜在应用。对于寻求更有效集成 AI 的公司来说，这可能是一个宝贵的资源。
- **合成数据集的重要性**：对话强调了**合成数据集**在训练 AI 模型中的重要性，特别是对于需要广泛且多样化数据源的任务。创建这些数据集的能力可以增强模型的性能和可靠性。



**提及的链接**：<a href="https://github.com/Mill-Pond-Research/AI-Knowledge-Base">GitHub - Mill-Pond-Research/AI-Knowledge-Base: Comprehensive Generalized Knowledge Base for AI Systems (RAG)</a>：AI 系统综合通用知识库 (RAG) - Mill-Pond-Research/AI-Knowledge-Base

  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1263208719383134330)** (3 条消息): 

> - `Intelligent Digital Agents`
> - `Mistral-NeMo-12B-Instruct`
> - `Synthetic Data Creation` 


- **探索 Intelligent Digital Agents**：[Intelligent Digital Agents in the Era of Large Language Models](https://x.com/ManifoldRG/status/1811120196570206459) 是一篇立场论文，讨论了 LLM 驱动的 Agent 的最新进展并指出了重大局限性。
   - 该论文建议必须从基于语言的处理转向其他方式，以增强推理能力。
- **Mistral-NeMo-12B-Instruct 发布**：NVIDIA 和 Mistral AI 推出了 [Mistral-NeMo-12B-Instruct](https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct)，这是一款拥有 **120 亿参数**并支持**多语言**应用的 Large Language Model。
   - 它拥有 **128k 上下文窗口**，并提供无精度损失的 **FP8 量化版本**，为同类模型树立了新标杆。
- **AgentInstruct 自动化合成数据生成**：Arindam Mitra 及其合著者介绍了 [AgentInstruct](https://x.com/MSFTResearch/status/1813974519469515087)，旨在通过多 Agent 框架实现自动化，从而简化合成数据生成（Synthetic Data Creation）的挑战。
   - 该框架旨在为语言模型训练后（post-training）应用大规模生成高质量的合成数据。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/nvidia/Mistral-NeMo-12B-Instruct">nvidia/Mistral-NeMo-12B-Instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/ManifoldRG/status/1811120196570206459">来自 Manifold Research (@ManifoldRG) 的推文</a>: 🚨我们很高兴分享《Intelligent Digital Agents in the Era of Large Language Models》，这是一篇探讨 LLM 驱动的 Agent 进展、识别局限性并建议...</li><li><a href="https://x.com/MSFTResearch/status/1813974519469515087">来自 Microsoft Research (@MSFTResearch) 的推文</a>: 合成数据生成非常困难。Arindam Mitra 及其合著者旨在通过 AgentInstruct 改变这一现状，这是一个自动化的多 Agent 框架，用于为语言模型大规模生成高质量合成数据...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1263208757719203880)** (115 条消息🔥🔥): 

> - `DeepSeek Model Release` (DeepSeek 模型发布)
> - `Mistral NeMo Performance` (Mistral NeMo 性能)
> - `GPT-4o Mini Benchmarking` (GPT-4o Mini 基准测试)
> - `Hermes Model Toolkit` (Hermes 模型工具包)
> - `FP8 Quantization Discussion` (FP8 量化讨论)


- **DeepSeek 模型发布**：新模型 DeepSeek-V2-0628 已发布，在 LMSYS Chatbot Arena Leaderboard 上排名第一，在多个类别中表现出色。
   - 该模型的可用性及其 API 已在 [DeepSeek 平台](https://platform.deepseek.com)上公布。
- **Mistral NeMo 宣称高性能**：Mistral NeMo 是一款 12B 模型，现已发布，拥有 128k tokens 的超大上下文窗口，并在多语言和代码数据上进行了训练。
   - 有人对其针对 Llama 3 8B 等模型的基准测试准确性表示担忧，认为报告的数据可能具有误导性。
- **GPT-4o Mini 性能评测**：GPT-4o Mini 在多项编程任务中进行了基准测试，表现与 GPT-3.5-Turbo 相似，这让一些期待更高性能的用户感到失望。
   - 评论指出 OpenAI 对该模型的基准测试可能过于乐观。
- **Hermes 模型工具包更新**：随着 Hugging Face 仓库中 tokenizer 配置的最新更新，Hermes 模型可能很快将支持 tool use。
   - 讨论显示，用户对 tokenizer 的更改存在困惑，这些问题已由贡献更新的终端用户解决。
- **FP8 量化技术**：最近的对话围绕模型训练中 FP8 量化的可行性和稳定性展开，对其效果褒贬不一。
   - 参与者强调了显著提升效率的潜力，同时也对 NVIDIA 对此类技术的支持表示怀疑。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/abacaj/status/1813977261818904908">anton (@abacaj) 的推文</a>: Mistral NeMo 报告的数据（考虑到它是一个 12B 模型对比 Meta Llama 8B）是错误的吗？由于某种原因，它们让 Llama 3 8B 看起来比实际情况差得多……</li><li><a href="https://x.com/maksym_andr/status/1813608842699079750">Maksym Andriushchenko (@maksym_andr) 的推文</a>: 🚨很高兴分享我们的新论文！🚨 我们揭示了当前 refusal training 方法中一个有趣的泛化差距：只需将有害请求重新表述为过去时（例如，“如何制作……”）</li><li><a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一款与 NVIDIA 合作构建的先进 12B 模型，具有 128k 上下文长度，并以 Apache 2.0 许可证发布。</li><li><a href="https://huggingface.co/mistralai/Mistral-Nemo-Instruct-2407">mistralai/Mistral-Nemo-Instruct-2407 · Hugging Face</a>: 未找到描述</li><li><a href="https://maartengr.github.io/BERTopic/index.html">BERTopic</a>: 未找到描述</li><li><a href="https://x.com/deepseek_ai/status/1813921111694053644">DeepSeek (@deepseek_ai) 的推文</a>: 🎉激动人心的消息！我们开源了 DeepSeek-V2-0628 检查点，这是 @lmsysorg LMSYS Chatbot Arena Leaderboard 上排名第一的开源模型。详细 Arena 排名：总榜第 11，Hard Prompts 第 3，Co...</li><li><a href="https://docs.vllm.ai/en/latest/quantization/fp8.html">FP8 &#8212; vLLM</a>: 未找到描述</li><li><a href="https://x.com/natolambert/status/1814024567192748166">Nathan Lambert (@natolambert) 的推文</a>: Reward Bench 上的 GPT4-o-mini：高于 Claude 3 Sonnet（不是 3.5）和 Llama 3 70B，低于 Gemma 2 27B。实际上这些都很相似。已经相当饱和了。</li><li><a href="http://github.com/neuralmagic/autofp8">GitHub - neuralmagic/AutoFP8</a>: 通过在 GitHub 上创建账户来为 neuralmagic/AutoFP8 的开发做出贡献。</li><li><a href="https://x.com/i/broadcasts/1lDGLldQVmvGm">GitHub 推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/discussions/13">NousResearch/Hermes-2-Pro-Llama-3-8B · 添加 tool use 模板</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B/commit/714ffdffc3cbf97d02f0b484c9676f371830bce3#d2h-846292">上传 3 个文件 · NousResearch/Hermes-2-Pro-Llama-3-8B @ 714ffdf</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1263543956776681564)** (6 messages): 

> - `World Sim functionality` (World Sim 功能)
> - `User feedback` (用户反馈)


- **World Sim 停机问题已解决**：用户对 **World Sim** 的功能表示担忧，指出其无法正常工作。
   - 一位成员确认问题正在解决中，并表示一分钟内应该就能恢复。
- **对快速修复的感谢**：一位成员感谢另一位成员解决了 **World Sim** 的问题，肯定了其对修复的贡献。
   - 在确认修复后，大家分享了“*感谢反馈 :)*”的情绪。


  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1263226138310738051)** (121 messages🔥🔥): 

> - `DeepSeek V2 Release` (DeepSeek V2 发布)
> - `ChatGPT Voice Mode` (ChatGPT 语音模式)
> - `GPT-4o Mini Launch` (GPT-4o Mini 发布)
> - `Upcoming Llama 3 Models` (即将发布的 Llama 3 模型)
> - `LMSYS Arena Updates` (LMSYS Arena 更新)


- **DeepSeek V2：新的竞争者**：DeepSeek 宣布发布其开源模型 DeepSeek-V2-0628，目前在 LMSYS Chatbot Arena 排行榜上排名第一。
   - 凭借每百万 token 仅 0.3 美元的卓越推理成本，其相对于大型竞争对手的高效率正受到关注。
- **ChatGPT 即将推出语音模式**：Sam Altman 宣布 ChatGPT 语音模式的 Alpha 测试将于本月晚些时候开始，随后将很快全面开放。
   - 此次发布凝聚了团队的巨大努力，引发了人们对 ChatGPT 新交互能力的期待。
- **GPT-4o Mini 发布**：OpenAI 推出了 GPT-4o Mini，被誉为最具成本效益的模型，输入价格为每百万 token 0.15 美元，输出为 0.60 美元。
   - 它在基准测试中优于许多小型模型，同时提供 128k 的 context window，使其适用于复杂的应用程序。
- **Llama 3 模型预计即将发布**：关于 Llama 3 模型的发布存在诸多猜测，特别是预计在四天内发布的一个新的 400B 参数版本。
   - 随着多个现有模型同时发布，讨论暗示这是一种旨在最大化影响力的协同发布策略。
- **LMSYS Arena 和模型更新**：LMSYS Arena 目前托管着许多未发布的模型，暗示了即将到来的 Gemini 等创新带来的竞争格局。
   - 针对同一天发布模型数量不断增加的评论，引发了关于 AI 研究社区战略时机选择的讨论。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/patloeber/status/1813871331756105744?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Patrick Loeber (@patloeber) 的推文</a>：这是 @karpathy 即将推出的 AI 课程大纲。天哪，我太期待了！🤩 特别期待所有不仅使用 Python，还使用 C 和 CUDA 的动手编程部分。</li><li><a href="https://x.com/jeffintime/status/1814000186357923851">来自 Jeff Harris (@jeffintime) 的推文</a>：宝藏发现：GPT-4o mini 支持最高 16K 的 max_tokens（GPT-4T 和 GPT-4o 仅为 4K） https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/</li><li><a href="https://x.com/andrewcurran_/status/1813942258968018954?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：这是我们的新模型 “GPT-4o mini”。根据 OpenAI 的说法，它是“当今最强大且最具成本效益的小型模型”。今天对免费版和 Pro 版用户开放。</li><li><a href="https://x.com/eglyman/status/1813987755270996106">来自 Eric Glyman (@eglyman) 的推文</a>：在早期测试中立刻就能发现，OpenAI 最新的 GPT-4o mini 模型更进一步。它正在帮助我们为客户节省更多时间。引用 Ramp (@tryramp) 很高兴能帮助 OpenAI...</li><li><a href="https://blogs.nvidia.com/blog/mistral-nvidia-ai-model/">Mistral AI 和 NVIDIA 发布 Mistral NeMo 12B，一款尖端的企业级 AI 模型</a>：Mistral AI 和 NVIDIA 今天发布了一款全新的最先进语言模型 Mistral NeMo 12B，开发者可以轻松地对其进行定制，并部署在支持聊天机器人、多语言...的企业级应用中。</li><li><a href="https://fxtwitter.com/artificialguybr/status/1814018708391760276">来自 𝑨𝒓𝒕𝒊𝒇𝒊𝒄𝒊𝒂𝒍 𝑮𝒖𝒚 (@artificialguybr) 的推文</a>：我删除了关于 GPT-4o 不再对免费用户开放的帖子。不幸的是，我被 UI 搞混了，说了一些废话/假新闻。我为这个错误道歉！</li><li><a href="https://x.com/ArtificialAnlys/status/1813975855468560621">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：今天发布的 GPT-4o Mini 以其极低的价格表现得非常出色 👀 凭借 82% 的 MMLU 分数（据 TechCrunch 报道），它超越了包括 Gemini 在内的其他小型模型的质量...</li><li><a href="https://x.com/mattshumer_/status/1813958229577302098">来自 Matt Shumer (@mattshumer_) 的推文</a>：Mistral NeMo 看起来是一个非常棒的模型 - 12B 参数，因此微调快速且便宜 - 推理速度快（体积小 + 经过量化感知训练） - 处理多语言效率极高的新 Tokenizer...</li><li><a href="https://x.com/emollick/status/1813753156431384851?s=46">来自 Ethan Mollick (@emollick) 的推文</a>：👀 Claude 处理了一个疯狂的请求：“移除鱿鱼”。“该文档似乎是 Erich Maria Remarque 的小说《西线无战事》的全文。它并不包含...”</li><li><a href="https://x.com/terryyuezhuo/status/1813998867039617444">来自 Terry Yue Zhuo (@terryyuezhuo) 的推文</a>：GPT-4o mini 在 BigCodeBench-Hard 上的表现出炉了：Complete Pass@1: 27.0，Instruct Pass@1: 24.3，平均分：25.7。平均分非常接近 Claude-3-Opus (26.0)！引用 Boris Power (@BorisMPower)...</li><li><a href="https://x.com/nutlope/status/1813996350008422426">来自 Hassan (@nutlope) 的推文</a>：Together AI 的 API 随着我们两个新版本的 Llama-3 变得更快、更便宜：◆ Llama-3-8B Turbo (FP8) – 高达 400 tokens/s ◆ Llama-3-8B Lite (INT4) – 每百万 tokens 0.10 美元 ◆ Turbo & Lite 适用于...</li><li><a href="https://x.com/teortaxesTex/status/1813717300257931588">来自 Teortaxes▶️ (@teortaxesTex) 的推文</a>：Deepseek 的内部日志是你从未读过的最好的仙侠故事。东海蓝鲸宗的弟子们将继续修炼，直到诸天震颤。</li><li><a href="https://www.theverge.com/2024/7/17/24199005/samsung-galaxy-ai-z-fold-6-sketch-to-image">三星新的图像生成 AI 工具好得有点过头了</a>：到底什么是照片？</li><li><a href="https://x.com/xenovacom/status/1813968731250274784">来自 Xenova (@xenovacom) 的推文</a>：Mistral 和 NVIDIA 刚刚发布了 Mistral NeMo，一个拥有 128k 上下文长度的最先进 12B 模型！😍 它使用了一个新的基于 Tiktoken 的 Tokenizer，在压缩源代码方面效率更高...</li><li><a href="https://x.com/natolambert/status/1813955064949772763?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：让我更印象深刻的是，微小的 Gemini Flash 模型击败了所有这些笨重的开源模型。传闻 Gemini Pro 的活跃参数量 < 70B，猜测 Gemini Flash 的活跃参数量 < 30B，甚至可能像...</li><li><a href="https://x.com/willdepue/status/1813995162814869892">来自 will depue (@willdepue) 的推文</a>：关于即将在近期推出的语音模式。团队为了发布这个付出了英雄般的努力。引用 Sam Altman (@sama)：@jakebrowatzke Alpha 测试本月晚些时候开始，正式发布（GA）会稍后到来。</li><li><a href="https://x.com/NickADobos/status/1813626926273380429">推文</a>

<li><a href="https://x.com/NickADobos/status/1813988248660320679">来自 Nick Dobos (@NickADobos) 的推文</a>：OpenAI 必须让 AI 变得更笨，这样愚蠢的人类才能理解它。引用 OpenAI (@OpenAI)：我们训练了先进的语言模型来生成弱模型可以轻松验证的文本，并发现它...</li><li><a href="https://x.com/swyx/status/1812988248660320679">来自 swyx 🤞 🔜 SFO (@swyx) 的推文</a>：完全假设一下... 如果有一个开源的 GPT-4o 级别的模型，你会用它做哪些现在做不到的事情？在“新常态” AI 的范围内，你可以提出哪些能带来超额收益（alpha）的问题...</li><li><a href="https://x.com/imjaredz/status/1814007428440272953">来自 Jared Zoneraich (@imjaredz) 的推文</a>：快速进行了一次批量运行，对比了 4-turbo 和 4o-mini。速度快了一个数量级且更便宜。这将开启许多新的用例，在这些用例中，你会乐于为了速度/成本而牺牲智能。引用 Jared Zon...</li><li><a href="https://x.com/imjaredz/status/1814005499299312021">来自 Jared Zoneraich (@imjaredz) 的推文</a>：gpt-4o-mini 刚刚发布，比原本就已经是目前最便宜模型的 gpt-4o 还便宜 33 倍。gpt-4o-mini 比 gpt-4 便宜 200 倍。已经被 @tryramp 和 @Superhuman 大规模使用。我们...</li><li><a href="https://x.com/minimaxir/status/1813985834728919249">来自 Max Woolf (@minimaxir) 的推文</a>：GPT-4o mini 的价格是每 1M input tokens 0.15 美元，每 1M output tokens 0.60 美元。相比之下，Claude Haiku 是每 1M input tokens 0.25 美元，每 1M output tokens 1.25 美元。这种价格战到底的竞争方式不可能持续...</li><li><a href="https://x.com/Teknium1/status/1813971144695075255">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：不知何故，这条消息今天在我的信息流中被漏掉了；Mistral 发布了一个新的基础模型，以多出 4B 参数的规模击败了 l3 8b —— 不确定它是否与旧的 Mistral 架构相同，因为它被称为 Mistral Nemo：https://mi...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1e61odl/introducing_spectra_a_comprehensive_study_of/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://x.com/sama/status/1813984927622549881">来自 Sam Altman (@sama) 的推文</a>：回溯到 2022 年，当时世界上最好的模型是 text-davinci-003。它比这个新模型差得多得多。而且价格贵了 100 倍。</li><li><a href="https://x.com/main_horse/status/1813580480761196987">来自 main (@main_horse) 的推文</a>：Deepseek 创始人梁文锋：我们不会走闭源路线。我们认为首先建立一个强大的技术生态系统更为重要。</li><li><a href="https://x.com/rememberlenny/status/1814004561696465316">来自 Lenny Bogdonoff (@rememberlenny) 的推文</a>：确实如此，但这针对的是从事智能工作的劳动时长。而且速度要快得多。</li><li><a href="https://x.com/abacaj/status/1813691718522564633">来自 anton (@abacaj) 的推文</a>：OpenAI > 这是我们的一篇酷炫论文，研究如何降低智能模型输出的难度；Anthropic > 这是一个你可以使用的酷炫模型，预计今年晚些时候会有更大的模型。</li><li><a href="https://x.com/togethercompute/status/1813989061503406478">来自 Together AI (@togethercompute) 的推文</a>：今天我们宣布了一个新的推理栈，其解码吞吐量比开源的 vLLM 快 4 倍。我们还推出了新的 Together Turbo 和 Together Lite 端点，能够实现...</li><li><a href="https://x.com/andrewcurran_/status/1813704834819965147?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：从中可以得知：- Llama 4 已于 6 月开始训练 - Llama 4 将是完全多模态的，包括音频 - Llama 3 405b 仍将在欧盟发布 - 除非... 否则 Llama 4 及更高版本将不会在欧盟发布。</li><li><a href="https://x.com/gdb/status/1814019156561543658?s=46">来自 Greg Brockman (@gdb) 的推文</a>：由于开发者的强烈需求，我们构建了 gpt-4o mini。我们 ❤️ 开发者，并致力于为他们提供最好的工具，将机器智能转化为各个领域的积极应用。请...</li><li><a href="https://news.ycombinator.com/item?id=40998702">未找到标题</a>：未找到描述</li><li><a href="https://x.com/phill__1/status/1813677446362992689">来自 Phil (@phill__1) 的推文</a>：目前在 lmsys arena 中至少有 6 个未发布的模型：-gemini-test-1 和 gemini-test-2（可能是新的 Gemini 1.5 版本，也可能是 Gemini 2.0）-im-a-little-birdie (???) -upcoming-gpt-mini ...</li><li><a href="https://x.com/simonw/status/1814003235268829494">来自 Simon Willison (@simonw) 的推文</a>：我关于今天发布的 GPT-4o mini 的笔记：https://simonwillison.net/2024/Jul/18/gpt-4o-mini/。最大的新闻是价格：这甚至比 Claude 3 Haiku 还便宜，每百万 input tokens 仅需 15 美分...</li><li><a href="https://x.com/romainhuet/status/1813986836039290970">来自 Romain Huet (@romainhuet) 的推文</a>：发布 GPT-4o mini：迄今为止最智能且最具成本效益的小型模型！它比 GPT-3.5 Turbo 更聪明、更便宜，是 function calling、大上下文、实时交互的理想选择——并且具有...</li><li><a href="https://x.com/elder_plinius/status/1814023961535295918?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Phil Plinius (@elder_plinius) 的推文</a></li>

">来自 Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>：⚡️ 越狱警报 ⚡️ OPENAI：被攻破 ✌️😎 GPT-4O-MINI：已解放 🤗 看来新的 &#34;instruction hierarchy&#34; 防御机制还不够 🤷‍♂️ 见证全新的 gpt-4o-mini ...</li><li><a href="https://news.ycombinator.com/item?id=40996058">未找到标题</a>：未找到描述</li><li><a href="https://x.com/vipulved/status/1813991596029084103">来自 Vipul Ved Prakash (@vipulved) 的推文</a>：我们今天发布了 Llama-3 的 Turbo 和 Lite 版本，结合了我们在优化和量化方面的最新研究。Lite 模型比 GPT-4o mini 便宜 6 倍，可能是目前最具成本效益的...</li><li><a href="https://x.com/karpathy/status/1814038096218083497?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Andrej Karpathy (@karpathy) 的推文</a>：LLM 模型尺寸竞争正在加剧……反向加剧！我打赌我们会看到“思考”得非常好且可靠的极小模型。很可能存在一种设置甚至...</li><li><a href="https://x.com/abacaj/status/1814000594899870070">来自 anton (@abacaj) 的推文</a>：比开源 vLLM 服务高出 4 倍的吞吐量……我们自己部署模型还有什么希望。引用 Together AI (@togethercompute)：今天我们宣布一个新的推理栈（inference stack），它提供了解码...</li><li><a href="https://x.com/lmsysorg/status/1813999088758673875">来自 lmsys.org (@lmsysorg) 的推文</a>：祝贺 @openai 发布新的 GPT-4o mini！GPT-4o mini 的早期版本 &#34;upcoming-gpt-mini&#34; 过去一周在 Arena 中进行了测试。凭借超过 6K 的用户投票，我们很高兴能分享它...</li><li><a href="https://x.com/LouisKnightWebb/status/1813996569840238794">来自 Louis Knight-Webb (@LouisKnightWebb) 的推文</a>：不出所料，gpt-4o mini 的上下文利用率（Context utilisation）比 3.5 好得多，但比不上老大哥 4o</li><li><a href="https://mp.weixin.qq.com/s/r9zZaEgqAa_lml_fOEZmjg">揭秘DeepSeek:一个更极致的中国技术理想主义故事</a>：做贡献者，而非搭便车者。</li><li><a href="https://www.artificial.agency/news/artificial-agency-launches">Artificial Agency 结束隐身模式并获得 1600 万美元融资，为游戏带来生成式行为（Generative Behavior） — Artificial Agency </a>：全球首个 AI 驱动的行为引擎，将运行时决策集成到游戏机制中，开启新一代自适应智能游戏</li><li><a href="https://x.com/GuillaumeLample/status/1813949898095534278">来自 Guillaume Lample @ ICLR 2024 (@GuillaumeLample) 的推文</a>：非常高兴发布我们的新小模型 Mistral NeMo，这是一个与 @nvidia 合作训练的 12B 模型。Mistral NeMo 支持 128k tokens 的上下文窗口，提供 FP8 对齐的检查点，...</li><li><a href="https://x.com/abacaj/status/1813977261818904908">来自 anton (@abacaj) 的推文</a>：Mistral NeMo 报告的数据（考虑到它是 12B 模型对比 Meta Llama 8B）是错误的吗？出于某种原因，它们让 Llama 3 8B 看起来比实际情况差得多……</li><li><a href="https://x.com/ArtificialAnlys/status/1813965193933623781">来自 Artificial Analysis (@ArtificialAnlys) 的推文</a>：Mistral 发布了 NeMo，这是一个新的开源、长上下文的小型模型，作为 Mistral 7B 的继任者。为什么它令人兴奋，请看下方 👇 - 具有 128k 上下文窗口的开源模型：大上下文窗口...</li><li><a href="https://github.com/openai/simple-evals">GitHub - openai/simple-evals</a>：通过在 GitHub 上创建账户来为 openai/simple-evals 的开发做出贡献。</li><li><a href="https://brx.ai/">BRX - 加载中</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1263570911253233765)** (1 条消息): 

> - `模型发布日`
> - `更新的线程讨论` 


- **宣布大模型发布日**：今天被标记为**大模型发布日**，预示着 AI 领域的重大更新。
   - 鼓励成员随时关注全天展开的最新进展。
- **讨论需主动加入**：提供**深度更新的线程讨论**，但参与需要主动加入（opt-in）。
   - 这确保了感兴趣的人可以积极参与讨论。

### **OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1263549154550874204)** (1 条消息): 

> - `GPT-4o mini`
> - `GPT-3.5 Turbo` 


- **介绍 GPT-4o mini！**: OpenAI 推出了 **GPT-4o mini**，这是目前最智能且最实惠的小型模型，现已在 API 中可用，并于今天开始在 ChatGPT 中推出。据报道，它比 **GPT-3.5 Turbo** 显著更智能且更便宜，对用户来说是一个极具前景的选择。
   - 更多详情，请查看 [OpenAI 官网](https://openai.com/index/gpt-4o-mini-advancing-cost-efficient-intelligence/)上的公告。
- **GPT-4o mini vs GPT-3.5 Turbo**: **GPT-4o mini** 的发布强调了其与 **GPT-3.5 Turbo** 相比增强的智能和成本效益。这一新模型旨在为寻求 AI 能力的用户提供更高效且经济的解决方案。
   - 社区成员对这一进展感到兴奋，许多人希望这种**更便宜的模型**能让更多人接触到先进的 AI 工具。


  

---


### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1263211426315767838)** (66 条消息🔥🔥): 

> - `Eleven Labs Voice Extraction Model`
> - `ChatGPT to Claude Transition`
> - `NVIDIA Installer Integration`
> - `Gpt-4o Mini Differences`
> - `Support for Image and Audio in Future Models` 


- **Eleven Labs 推出语音提取功能**: 有传闻称 Eleven Labs 发布了一个新的[语音提取模型](https://link.to.details)，这增强了 AI 音频处理领域的能力。
   - 这紧随 AI 创新趋势，提高了人们对实际应用的期望。
- **从 ChatGPT 转向 Claude**: 讨论中提到了用户从 **ChatGPT** 转向 **Claude** 的现象，这表明了 AI 偏好中更广泛的趋势。
   - 许多用户分享了他们对这些变化的沮丧和兴奋，展示了 AI 工具不断演变的格局。
- **NVIDIA 安装程序包集成 Meta**: 有评论流传称即将推出的 [NVIDIA 安装程序](https://link.to/nvidia-installation)可能会集成 Facebook 和 Instagram。
   - 这种集成引发了关于 Meta 不断扩张的生态系统的疑问，特别是其社交媒体平台。
- **澄清 Gpt-4o 与 Gpt-4o Mini 的区别**: 对比了 **Gpt-4o** 和 **Gpt-4o Mini** 的差异，强调 Mini 虽然“智能”程度稍低，但价格显著更便宜。
   - 参与者推测 Mini 模型的能力是否会与未来音频和视频支持的进展保持一致。
- **AI 模型的未来功能**: 对话透露 **Gpt-4o Mini** 预计在未来的更新中支持图像和音频的输入/输出功能。
   - 成员们对这些功能的发布时间表表示好奇，特别是它们与当前模型能力的关系。



**提到的链接**: <a href="https://tenor.com/view/gollum-lord-of-the-rings-gif-19273356">Gollum Lord GIF - Gollum Lord Of - Discover &amp; Share GIFs</a>: 点击查看 GIF

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1263211465738031315)** (15 条消息🔥): 

> - `Quota limitations with OpenAI API`
> - `Image token counts for GPT-4o and GPT-4o mini`
> - `Rate limit changes`
> - `Capabilities of 4o-mini in Playground`
> - `Performance comparison between GPT-4o mini and GPT-4o` 


- **配额限制阻碍使用**: 一名用户报告在尝试运行 OpenAI API 的代码片段时遇到配额错误，提示 **'You exceeded your current quota'**。
   - 另一名用户建议，这可能是因为 API 需要购买额度（credits）才能使用。
- **图像 Token 计数差异**: 讨论了 **GPT-4o mini** 的图像 Token 计数，一名用户能够发送 **150k tokens**，尽管据称限制为 128k。
   - 用户对定价页面表示担忧，该页面显示 **4o mini** 的图像 Token 计数更高，具体而言，一张 **150x150** 的图像需要 **255 tokens**。
- **更新后的速率限制变化**: 一名用户询问 **rate limits**（速率限制）是否也随最近的更新发生了变化。
   - 这一话题尚未解决，因为目前还没有关于更新对速率限制影响的实质性回复。
- **探索 4o-mini 的全部功能**: 有人询问用户是否可以在 Playground 中使用 **4o-mini** 的所有功能，还是必须等待 ChatGPT 的发布。
   - 回复确认通过 API 和 Playground 进行测试确实是可行的。
- **比较 GPT-4o mini 与 GPT-4o**: 当被问及 **4o mini** 是否比 **4o** 更智能时，一名用户澄清说它比它所取代的 **3.5** 更智能。
   - 目前还没有直接对 **4o mini** 和 **4o** 的智能水平进行对比。


  

---

### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1263228216668192828)** (20 条消息🔥): 

> - `ChatGPT 幻觉挑战`
> - `新型提示词框架`
> - `语音 Agent 停顿控制`
> - `思维启发策略` 


- **探讨 ChatGPT 的幻觉风险**：一位用户讨论了 ChatGPT 4o 在面对 **'fleperwelp'** 等未知术语时产生内容的倾向，强调了需要结构化 Prompt 来减轻幻觉。
   - *“模型没有被要求做其他事情，但它却凭空捏造”* 强调了生成看似合理但纯属虚假响应的问题。
- **针对 Zero-Shot Prompt 的创新框架**：一位用户分享了他们对一种新型提示词框架的实验，该框架在 Zero-Shot 场景下特别有效，并展示了其成功的实验结果。
   - 他们还提供了一个将文本转换为 **EWAC 命令** 的工具，邀请其他人试用。
- **语音 Agent 的停顿控制机制**：一位开发者分享了他们在 AI 语音 Agent 方面的进展，该 Agent 能够通过在单词之间插入特殊字符来实现 **停顿**，从而控制语速。
   - 然而，他们表示在教导模型如何在说话过程中自然地产生这些停顿方面面临挑战。
- **有效利用示例教学停顿**：有建议提出向模型提供说明停顿常见用法的示例（如电话号码和地址），以增强其理解。
   - *“你有没有问过模型，它对于人类说话时通常在何时何地停顿了解多少？”* 为用户指出了一个潜在的探索领域。
- **研究模型知识以优化输出**：一位用户认识到，在进一步完善 Prompt 之前，查询模型关于 **停顿策略** 的现有知识非常重要。
   - 这种战术性方法被认为不仅有利于 AI 交互，也有利于一般的沟通改进。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1263228216668192828)** (20 条消息🔥): 

> - `ChatGPT 幻觉`
> - `新型提示词框架`
> - `语音 Agent 停顿控制` 


- **ChatGPT 幻觉高度警惕**：一名成员对 ChatGPT 在被问及冷门技术 'Fleperwelp' 时编造信息表示担忧，强调需要更清晰的指令来避免幻觉。
   - 他们建议探索 Prompt 的组成部分，并确保在提供何种信息方面有明确的指导。
- **EWAC 命令框架讨论**：一名成员介绍了一种新型提示词框架，该框架能有效处理 Zero-Shot 系统提示和通用查询，展示了其实用性。
   - 他们邀请其他人查看他们的 GPT，该 GPT 可以将文本转换为 EWAC 命令。
- **语音 Agent 的停顿控制挑战**：一名成员分享了他们在语音 Agent 方面的工作，该 Agent 能够通过在阅读电话号码等信息时插入特殊停顿字符来调整速度。
   - 他们表示，在编程 Agent 以识别响应用户“减速”指令时的合适停顿位置方面存在挑战。
- **优化语音 Agent 的语音模式**：该成员正在寻求方法，使他们的语音 Agent 在接收到用户要求放慢语速的输入后，能够识别并执行停顿。
   - 他们的目标是设计 Agent 辨别句子重要部分的能力，以便进行有效的停顿放置。
- **语言学知识调研**：一名成员询问了在定义查询范围之前，探索模型关于语音停顿的现有知识是否有用。
   - 另一名成员确认该技术是有益的，特别是在连接想法以提高响应质量时。


  

---



### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/)** (1 条消息): 

natolambert: 有人在 ICML 吗？我的一个 VC 朋友想在高级晚宴上见见我的朋友们。
  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1263376755381637201)** (74 messages🔥🔥): 

> - `EU 法规`
> - `Mistral NeMo 发布`
> - `GPT-4o Mini 性能`
> - `Deepseek 许可证担忧`
> - `LMSYS 中的模型传闻` 


- **EU 法规引发紧张局势**：讨论强调了对 **EU** 法规的担忧，一些人认为这可能会阻碍对 AI 模型的访问，导致人们猜测需要使用 VPN 才能下载。
   - 一位用户指出，现行立法令大型科技公司感到沮丧，可能影响其运营决策。
- **Mistral NeMo 激发开源社区热情**：**Mistral NeMo** 是一款具有 **128k** token 上下文窗口的 12B 模型，其发布预计将促进研究和商业环境中的采用。
   - 该模型具有无损 FP8 性能等令人印象深刻的特性，被视为市场上现有产品的强大竞争对手。
- **GPT-4o Mini 在基准测试中表现出色**：**GPT-4o Mini** 在某些基准测试中的得分与 **GPT 3.5** 相当，尽管用户注意到它在高效处理代码编辑方面存在局限性。
   - 尽管早期结果显示出潜力，但由于性能限制，人们对其编辑大型代码文件的能力仍存疑虑。
- **Deepseek 许可证引发批评**：用户对 **Deepseek License** 表示担忧，认为其难以理解，并暗示这可能会阻碍更广泛的采用。
   - 虽然 Deepseek 为学术界提供了更便宜的 API 使用方案，但其许可条款可能为更广泛的部署设置障碍。
- **关于即将推出的 LMSYS 模型的传闻**：一位成员分享了关于 **LMSYS** 竞技场中多个未发布模型的传闻，包括 **Gemini** 迭代版本和 **Eureka Chatbot**。
   - 用户对这些模型表示怀疑，同时在期待中推动测试设置的更高透明度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/andrewcurran_/status/1813942258968018954?s=46">Andrew Curran (@AndrewCurran_) 的推文</a>: 这是我们的新模型。“GPT-4o mini”。OpenAI 称其为“当今最强大且最具成本效益的小型模型”。今天对免费版和专业版用户开放。</li><li><a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的 state-of-the-art 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://x.com/paulgauthier/status/1814014867361374610?s=46">Paul Gauthier (@paulgauthier) 的推文</a>: GPT 4o mini 在 aider 的代码编辑基准测试中得分与原始 GPT 3.5 相似（后期的 3.5 版本表现更差）。初步看来，它似乎无法通过 diffs 编辑代码，这限制了它的用途 ...</li><li><a href="https://x.com/andrewcurran_/status/1813965829996003608?s=46">Andrew Curran (@AndrewCurran_) 的推文</a>: 对于许多询问 GPT-4o Mini 的 API 定价的人：每百万 token 输入 15 美分，每百万 token 输出 60 美分。128k 上下文窗口。</li><li><a href="https://x.com/morqon/status/1813960872810996211?s=46">morgan — (@morqon) 的推文</a>: gpt-4o mini 在 MMLU 上得分 82%，仅供参考。</li><li><a href="https://x.com/phill__1/status/1813677446362992689">Phil (@phill__1) 的推文</a>: 目前在 lmsys 竞技场中至少有 6 个未发布的模型：-gemini-test-1 和 gemini-test-2（可能是新的 Gemini 1.5 版本，也许是 Gemini 2.0）-im-a-little-birdie (???) -upcoming-gpt-mini ...</li><li><a href="https://fxtwitter.com/testingcatalog/status/1813965406664900856?s=46">TestingCatalog News 🗞 (@testingcatalog) 的推文</a>: 基于</li><li><a href="https://x.com/elder_plinius/status/1814023961535295918?s=46">Pliny the Prompter 🐉 (@elder_plinius) 的推文</a>: ⚡️ 越狱警报 ⚡️ OPENAI：被攻破 ✌️😎 GPT-4O-MINI：被解放 🤗 看起来新的“指令层级”防御机制还不够 🤷‍♂️ 见证新的 gpt-4o-mini ...
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1263263486079795220)** (5 条消息): 

> - `PRM-Code dataset`
> - `Code-related PRM datasets`
> - `Positive/Negative/Neutral vs Scalar labels`
> - `Synthetic data in research` 


- **对代码相关 PRM 数据集的请求**：一名成员询问了已发布的优质**代码相关 PRM 数据集**，并表示《Let's Reward Step by Step》论文中的 AST 变异方法感觉有局限性。
   - 他们还质疑了与 Ground-truth 数据集相比，基于模型的变异在生成中性/负面示例方面的有效性。
- **确认对代码 PRM 数据集的需求**：另一位成员确认目前缺乏此类数据集，指出这些数据集是**迫切需要的**，并鼓励提问者去创建它们。
   - 这一评论引发了发起者轻松的回应，表达了他们目前作为一名渴望学习的新手的状态。
- **探索合成数据创建**：原帖作者分享了他们打算在即将开始的 MS 课程中探索“做研究”的想法，特别是围绕 PRM 和合成数据（Synthetic data）。
   - 他们表达了理解用于生成数据集的 **Pos/Neg/Neutral vs Scalar** 标注系统之间细微差别的动力。
- **不确定的书籍引用**：一名成员要求澄清一本被引用的书籍，导致了关于该书目前涉及诉讼的猜测。
   - 回复比较含糊，表明对所提及的书籍缺乏官方确认。


  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1263218671552168088)** (21 条消息🔥): 

> - `Public Perception of AI`
> - `OpenAI's Business Challenges`
> - `Consumer Tools vs Enterprise Solutions`
> - `Google vs OpenAI Shipping`
> - `Witchcraft Metaphor in AI Discussions` 


- **公众对 AI 工具的不安**：讨论强调，许多“普通人”觉得强大的 AI 工具令人费解且不适，特别是像 **ChatGPT** 这样通常被视为 AI 门面的工具。
   - *巫术*目前可能有些夸张，但人们担心公众的不安可能会导致社会对令人不安的想法做出反应的历史重演。
- **OpenAI 面临业务挑战**：有推测称 **OpenAI** 可能正面临重大的业务问题，特别是当他们从一个小团队扩展到数千人规模时。
   - 一些人认为，相对于实现 **AGI** 的使命，业务中驱动收入的部分正变成次要问题。
- **扩展 AI 公司非常困难**：讨论成员指出，扩展 OpenAI 的员工队伍是一项艰巨的任务，尤其是与拥有成熟资源的更大型科技公司相比。
   - 虽然扩展并非不可逾越，但重点仍然在于他们能否成功地重新分配现有资源以满足当前需求。
- **消费者偏好不足以构成护城河**：成员们对通过 **ChatGPT** 收集的用户偏好数据似乎并未为 OpenAI 提供针对新兴模型的竞争优势感到惊讶。
   - 这种观点表明，尽管拥有潜在优势，但 AI 领域的竞争已变得更加激烈，各种参与者都在追赶。
- **OpenAI 与 Google 发布速度的比较**：有人担心在 AI 开发的进展和功能发布（Shipping）方面，**OpenAI** 似乎被 **Google** 赶超了。
   - 一名成员提到想使用 **GPT-4o mini** 进行图像生成，表明希望 OpenAI 能有更快速的创新。



**提到的链接**：<a href="https://x.com/cto_junior/status/1813956330287513717?s=46">来自 TDM (e/λ) (@cto_junior) 的推文</a>：所有酷炫的东西都在后面，我很确定在这一切之前我们会先迎来 Gemini-2.0，它无论如何都支持所有模态。

  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1263352302366883870)** (9 messages🔥): 

> - `Codestral Mamba`
> - `DeepSeek-V2-0628 Release`
> - `Whale Organization` 


- **Codestral Mamba 在 Token 上下文处理上遇到困难**：一位成员指出，**Codestral Mamba** 的准确率在约 **1k tokens** 的上下文后会降至 **零**，这表明该研究领域仍面临持续挑战。
   - *这对于一个声称能处理“无限”上下文的模型来说是一个重大问题*，引发了对其实际应用的担忧。
- **DeepSeek-V2-0628 正式发布**：令人兴奋的消息，**DeepSeek** 发布了 **DeepSeek-V2-0628** 检查点，该模型在 LMSYS Chatbot Arena 排行榜的多个类别中位列第一，其中在 **Hard Prompts 类别中排名第三**。
   - 该模型目前已在 [Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628) 上线，并在 [DeepSeek Platform](https://platform.deepseek.com) 提供 API。
- **社区对 Whale Organization 的喜爱**：成员们表达了对 **Whale Organization** 的钦佩，并对其在 AI 领域的贡献给予了正面评价。
   - 一位参与者简单地表示，“我们爱 Whale”，并强调了该组织在培育这些发展成果中所扮演的角色。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/louisknightwebb/status/1813678943230439851?s=46">来自 Louis Knight-Webb (@LouisKnightWebb) 的推文</a>：Codestral Mamba 🐍 的准确率在约 1k tokens 上下文后降至零。作为对比的是 Codestral (常规版)。看来整个 Mamba 架构仍然是一个非常开放的研究问题，但即便如此……</li><li><a href="https://x.com/deepseek_ai/status/1813921111694053644?s=46">来自 DeepSeek (@deepseek_ai) 的推文</a>：🎉激动人心的消息！我们开源了 DeepSeek-V2-0628 检查点，这是 LMSYS Chatbot Arena 排行榜上排名第一的开源模型。详细 Arena 排名：总榜第 11，Hard Prompts 第 3，Co...
</li>
</ul>

</div>
  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1263565530372112538)** (1 messages): 

> - `GPT-4o mini`
> - `Cost-effectiveness in models` 


- **GPT-4o mini：OpenAI 的最新创新**：OpenAI 推出了 **GPT-4o mini**，这是其最新的支持文本和图像输入、文本输出的模型，可通过[此链接](https://openrouter.ai/models/openai/gpt-4o-mini)获取。
   - 该模型旨在保持 **SOTA 级别的智能**，同时具有极高的性价比，价格仅为 **每百万 Input $0.15，每百万 Output $0.60**。
- **GPT-4o mini 的相对经济性**：GPT-4o mini 比 [GPT-3.5 Turbo](https://openrouter.ai/models/openai/gpt-3.5-turbo) 便宜 **60% 以上**，使其成为追求性价比用户的理想替代方案。
   - 其定价被描述为比其他近期的前沿模型 **便宜许多倍**，标志着先进 AI 可及性的重大转变。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://openrouter.ai/models/openai/gpt-4o-mini>)">OpenAI: GPT-4o by openai</a>：GPT-4o（“o”代表“omni”）是 OpenAI 最新的 AI 模型，支持文本和图像输入以及文本输出。它保持了 [GPT-4 Turbo](/models/open... 的智能水平。</li><li><a href="https://openrouter.ai/models/openai/gpt-3.5-turbo>)">OpenAI: GPT-3.5 Turbo by openai</a>：GPT-3.5 Turbo 是 OpenAI 速度最快的模型。它可以理解并生成自然语言或代码，并针对聊天和传统的补全任务进行了优化。训练数据截至 2021 年 9 月。
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1263243002571395074)** (97 条消息🔥🔥): 

> - `Mistral NeMo 发布`
> - `OpenAI GPT-4o Mini 公告`
> - `OpenRouter 可用性`
> - `图像 Token 定价`
> - `Gemma 2 用户体验` 


- **支持 128K 上下文窗口的 Mistral NeMo 发布**：Mistral 宣布推出 **Mistral NeMo**，这是一个 12B 模型，支持高达 **128,000 tokens** 的超大上下文窗口。其推理能力和世界知识被公认为达到 State-of-the-art 水平，更多细节可在[官方博客文章](https://mistral.ai/news/codestral-mamba/)中查看。
   - 用户讨论了该预训练检查点（checkpoints）在 **Apache 2.0 许可证**下发布，促进了研究人员和企业的使用。
- **OpenAI 发布 GPT-4o Mini**：OpenAI 推出了备受期待的 **GPT-4o Mini**，旨在取代 **GPT-3.5 Turbo**。ChatGPT 免费用户以及 Plus 和 Team 订阅用户很快即可使用。初始定价已确定，输入成本为 **每 100 万 tokens 0.15 美元**，输出成本为 **每 100 万 tokens 0.60 美元**。
   - 讨论涵盖了该模型在 ChatGPT 网站和 OpenAI API 上的可用性，用户对使用这一升级后的模型表现出极大的热情。
- **OpenRouter 当前状态**：参与者询问了 **OpenRouter 的可用性**，并提供了其状态页面的链接，该页面显示近期没有影响性能的事件。截至 2024 年 7 月 18 日，该平台运行正常，未报告停机。
   - 用户分享了对区域性问题以及团队对停机情况持续监测的观察，反映了平台的整体状态和可靠性。
- **图像 Token 定价困惑**：关于模型如何对 **图像 tokens** 计费引发了讨论，用户对分辨率和潜在的成本差异提出了疑问。计算表明，最大 token 计数似乎有所增加，使得图像定价与 GPT-4o 保持一致。
   - 用户对这些计费实践是否统一适用于不同的图像尺寸和分辨率表示担忧，并持续讨论其对用户的影响。
- **Gemma 2 9B 的用户重复问题**：一位用户表达了在使用 **Gemma 2 9B** 模型时遇到 **重复问题 (repetition issues)** 的困扰，并向社区寻求可能的解决方案或建议。这引发了关于用户体验和潜在模型局限性的进一步讨论。
   - 社区认为按关键性能指标对模型响应进行排序非常重要，这有助于识别特定的模式或结果。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://t.co/FgHDivTLh5">Mistral NeMo</a>: Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的 State-of-the-art 12B 模型，与 NVIDIA 合作构建，并在 Apache 2.0 许可证下发布。</li><li><a href="https://huggingface.co/mistralai/mamba-codestral-7B-v0.1">mistralai/mamba-codestral-7B-v0.1 · Hugging Face</a>: 暂无描述</li><li><a href="https://www.cnbc.com/2024/07/18/openai-4o-mini-model-announced.html">OpenAI 首次推出其迄今为止最强大模型的 mini 版本</a>: OpenAI 周四发布了新 AI 模型 &quot;GPT-4o mini&quot;，这是这家人工智能初创公司扩大其热门聊天机器人使用的最新努力。</li><li><a href="https://x.com/mattshumer_/status/1813952065057542522">Matt Shumer (@mattshumer_) 的推文</a>: OpenAI 新模型！GPT-4o mini 今日发布。似乎是 GPT-3.5-Turbo 的替代品（终于！）。这个模型看起来与 Claude Haiku 非常相似 —— 快速、廉价，且非常擅长处理...</li><li><a href="https://status.openrouter.ai/">OpenRouter 状态</a>: OpenRouter 事件历史记录
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1263252832392450051)** (7 messages): 

> - `Max/Mojo 中的 GPU 支持`
> - `Mojo 中的并行化`
> - `Nvidia 协作` 


- **Max/Mojo 正式支持 GPU**：用户讨论了最近关于 Max/Mojo 支持 **GPU** 的公告，特别是在 Lattner 关于 Nvidia 的演讲中。
   - 成员们对该功能如何在 Mojo 中集成和使用表示好奇。
- **直接从 Mojo 进行并行化**：成员们询问 **parallelization**（并行化）是直接在 Mojo 中执行，还是通过 Max 促进。
   - 澄清说，理想情况下，Mojo 将允许直接并行化，利用最新的硬件能力。
- **信任 MAX 编译器进行优化**：一位回复者指出，使用 **MAX** 可以让编译器自动优化流程，提供流线化的体验。
   - 然而，在使用 MAX 时，用户仍然可以编写自定义 Mojo kernel 以实现对操作的更精细控制。
- **与 Nvidia 合作支持 CUDA**：提到与 **Nvidia** 的合作伙伴关系，旨在将 CUDA 与 Mojo 的功能集成，预计于 2023 年 12 月完成。
   - 目标包括确保 Mojo 中的并行操作将通过 MLIR 在 GPU 硬件上有效执行。



**提到的链接**：<a href="https://github.com/modularml/mojo/issues/3262)">Issues · modularml/mojo</a>：Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。

  

---


### **Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/)** (1 messages): 

ModularBot：来自 *Modular*：
<https://twitter.com/Modular/status/1813988940405493914>
  

---


### **Modular (Mojo 🔥) ▷ #[ai](https://discord.com/channels/1087530497313357884/1103420074372644916/1263208633446170735)** (7 messages): 

> - `图像目标检测模型`
> - `帧率优化`
> - `处理中的视频帧处理`
> - `Mojo 数据类型` 


- **实时检测中常见的帧率**：对于实时应用，通常以 **5 fps** 等低 **frame rate**（帧率）运行图像目标检测模型，而不是处理每一帧。
   - Bounding box（边界框）问题很常见，可以通过后处理来 **平滑框的位置**。
- **多帧带来的挑战**：一位成员询问在使用 **MP4** 格式进行目标检测时，如何更好地处理大型视频的帧。
   - 另一位成员承认没有灵丹妙药，并表示：*“如果你有很多绝对必须处理的帧，那么你就必须处理它们。”*
- **请求 Mojo 数据类型**：<@1099107160882950244> 被要求列出 Mojo 中的 **primitive** 和 **composite 数据类型**。
   - 针对该请求，没有提供额外的回复或细节。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1263211911164592170)** (35 messages🔥): 

> - `在 Mojo 中循环遍历 Tuple`
> - `Mojo 命名规范`
> - `Keras 3.0 发布`
> - `MAX 与通用计算`
> - `使用 InlineArray 对比 Tuple` 


- **在 Mojo 中循环遍历 Tuple**：一位用户询问了如何在 Mojo 中循环遍历 `Tuple`，但有人指出由于其异构（heterogeneous）特性，通常无法直接遍历。建议考虑使用 `InlineArray` 代替。
- **Mojo 命名规范**：有人请求提供关于 Mojo 变量和文件命名的最佳实践资源，类似于 Python 的 PEP8。一位参与者分享了来自 [Mojo 仓库的风格指南](https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md)，作为一个有用的参考资源。
- **Keras 3.0 发布**：社区讨论了最近正式发布的 [Keras 3.0](https://keras.io/keras_3/)，强调了其与 JAX、TensorFlow 和 PyTorch 协同工作的能力。这标志着 Keras 和 Mojo 的重大进展，大家对这些技术的集成持乐观态度。
- **MAX 与通用计算**：参与者讨论了 MAX 作为图编译器（graph compiler）的能力及其与高性能计算（HPC）框架的区别。对话暗示 MAX 的可移植性可能取决于其运行任意设备端 kernel 的能力。
- **使用 InlineArray 对比 Tuple**：关于在 Mojo 中使用 `InlineArray` 和 tuple 处理 `FloatLiteral` 的讨论。虽然 `Tuple[FloatLiteral, FloatLiteral](1.0, 2.0)` 可以工作，但参与者建议根据具体用例，`InlineArray` 可能更合适。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://keras.io/keras_3/">Keras: Deep Learning for humans</a>: 未找到描述</li><li><a href="https://youtu.be/_QVs626Vn2k?t=3934">Mojo 🔥 Community Meeting #4</a>: Mojo 社区会议 #4 的录音 🫓 Flat Buffers: 高效内存序列化 ⚒️ Forge Tools: 扩展 Mojo 🔥 标准库 🔄 Mojo 🔥 Gen...</li><li><a href="https://github.com/modularml/mojo/blob/main/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at main · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[max](https://discord.com/channels/1087530497313357884/1212827597323509870/1263373831993688245)** (5 条消息): 

> - `使用 Llama3 进行 Max Inference`
> - `加载模型权重`
> - `交互式聊天机器人示例`
> - `Hugging Face 模型 URI`
> - `CLI 改进` 


- **Max Inference 利用 Prompt 作为上下文**：当使用 `mojo ../../run_pipeline.:fire: llama3 --prompt` 时，Prompt 作为推理的初始上下文，在退出前生成一个响应。
   - *Bradlarson* 提到，最新的示例对此进行了扩展，包含了交互式聊天功能。
- **Llama3 中的自定义权重加载**：用户可以通过在本地下载并指定 `--model-path` 参数，为 Llama 3 Pipeline 加载任意权重。
   - *Bradlarson* 解释说，选择默认权重是因为 GGUF 版本比 PyTorch 权重更容易获取，从而提升了初始加载体验。
- **提供交互式聊天机器人示例**：在 `max` 仓库的 nightly 分支中开发了一个交互式聊天机器人示例，允许设置 System Prompts 并保留上下文。
   - 该示例可以在 [此 GitHub 链接](https://github.com/modularml/max/tree/nightly/examples/gui) 找到，并可以在社区会议视频的 [此时间戳](https://www.youtube.com/live/uookgZ7Ojg8?si=u-iwoMJWmMigVwSH&t=1197) 处观看。
- **关于来自 Hugging Face 的模型 URI 的讨论**：用户的询问集中在为什么使用 llama3-7B 等特定模型，而不是从官方 Hugging Face 模型页面加载。
   - 据指出，选择默认模型是因为它们的 GGUF 格式，与 PyTorch 权重相比，这使得模型摄取（Ingestion）更容易。
- **持续进行的 CLI 体验改进**：团队正致力于增强文本生成 Pipeline 的命令行界面体验。
   - 改进旨在为使用 MAX 平台的用户提供更流畅、更直观的交互。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/modularml/max/tree/nightly/examples/gui">max/examples/gui at nightly · modularml/max</a>：一系列示例程序、Notebook 和工具，展示了 MAX Platform 的强大功能 - modularml/max</li><li><a href="https://www.youtube.com/live/uookgZ7Ojg8?si=u-iwoMJWmMigVwSH&t=1197">Modular 社区直播 - MAX 24.4 新特性</a>：MAX 24.4 现已发布！加入我们的直播，讨论 MAX Engine 和 Mojo🔥 的新功能 - macOS 上的 MAX、MAX Engine 量化 API 等...</li><li><a href="https://llama.meta.com/docs/model-cards-and-prompt-formats/meta-llama-3">Meta Llama 3 | 模型卡片与 Prompt 格式</a>：Meta Llama 3 使用的特殊 Token。一个 Prompt 应包含单个 System Message，可以包含多个交替的 User 和 Assistant Message，并始终以最后一个 User Message 结束...</li><li><a href="https://huggingface.co/meta-llama/">meta-llama (Meta Llama)</a>：未找到描述</li><li><a href="https://github.com/modularml/max/blob/7189864b2fc829176149f6997a70c62732982ec8/examples/graph-api/pipelines/llama3/run.%F0%9F%94%A5#L224-L243">max/examples/graph-api/pipelines/llama3/run.🔥 at 7189864b2fc829176149f6997a70c62732982ec8 · modularml/max</a>：一系列示例程序、Notebook 和工具，展示了 MAX Platform 的强大功能 - modularml/max
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1263311768772546580)** (13 条消息🔥): 

> - `Nightly Mojo 编译器更新`
> - `stdlib 扩展提案`
> - `社区对 stdlib 的反馈`
> - `对 Async IO API 的担忧`
> - `关于 stdlib 选择性加入/退出的讨论` 


- **Nightly Mojo 编译器升级**：发布了新的 Nightly Mojo 编译器，更新至 `2024.7.1805` 版本，增强功能包括支持嵌套 Python 对象以及对标准库函数的各种修复。
   - 完整的变更日志更新请点击[此处](https://github.com/modularml/mojo/blob/nightly/docs/changelog.md)。
- **stdlib 扩展提案受到关注**：一项通过 `stdlib-extensions` 减轻 **stdlib** 维护者工作量的提案正在被积极讨论，重点强调了来自贡献者的社区反馈。
   - 讨论强调了在提交贡献之前评估 API 和普及度的重要性，将其作为解决人力资源挑战的技术方案。
- **社区支持对想法进行验证**：成员们主张在将提案纳入 stdlib 之前由社区进行评估，以防止罕见用例带来的潜在摩擦。
   - 有人对 stdlib 如何对齐高性能需求提出了担忧，特别是关于 async IO API。
- **Async IO API 兼容性担忧**：人们强烈希望有一种 async IO API，通过直接向操作提供缓冲区来适应更高性能，并与 Python 的标准实现分离。
   - 一位成员表示，避免注重性能的库与流行工具之间的冲突对未来发展至关重要。
- **关于标准库灵活性的辩论**：讨论围绕选择性加入或退出使用 stdlib 的可能性展开，揭示了社区对代码灵活性的渴望。
   - 成员们提到提供此类选项可能是有益的，尽管对于该功能是否已经存在仍有争议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/gabrieldemarmiesse/mojo/blob/proposal_stdlib_extensions/proposals/stdlib-extensions.md#the-future-of-this-repository-when-mojo-has-a-public-source-of-truth">mojo/proposals/stdlib-extensions.md at proposal_stdlib_extensions · gabrieldemarmiesse/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 gabrieldemarmiesse/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/discussions/3233">[Proposal] 通过 `stdlib-extensions` 减轻 stdlib 维护者的工作量 · modularml/mojo · Discussion #3233</a>: 此讨论旨在提供一个交流以下提案的场所：pull request markdown 文档。我们对频繁贡献者的意见以及...特别感兴趣。
</li>
</ul>

</div>
  

---

### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1263224800147542017)** (16 messages🔥): 

> - `Lubeck 性能`
> - `LLVM 生成`
> - `SPIRAL 项目`
> - `cuBLAS 集成` 


- **Lubeck 超越 MKL**：成员们讨论了据报道 **Lubeck** 比知名库 **MKL** 更快。
   - 尽管 **Lubeck** 与某些 **BLAS** 有联系，但其独特的 **LLVM IR generation** 方法被认为是其性能优越的原因。
- **LLVM 在性能中的作用**：一位用户引用称 **Mir** 是一个 **LLVM** 加速的数值库，暗示了其对性能基准测试的潜在影响。
   - 这引发了关于 **LLVM** 的使用如何解释 **Lubeck** 相比传统 **BLAS** 增强能力的讨论。
- **SPIRAL 旨在实现自动化**：**SPIRAL** 项目被介绍为一种为数字信号处理自动生成高性能库的工具。
   - 讨论强调了它的复杂性，其代码类似于*数学论文*，目标是在最大化性能的同时最小化编码工作。
- **SPIRAL 使用中的挑战**：虽然 **SPIRAL** 可以产生高效的库实现，但其复杂性意味着它对于像 **BLAS** 这样的标准函数往往不切实际。
   - 成员们评论说，尽管它功能强大，但挑战在于它对于价值较低的数值函数的可用性。
- **使用 cuBLAS 进行 GPU 加速**：一位成员建议通过将 **cuBLAS** 与 **NumPy** 集成来作为变通方案，利用 **GPU** 能力进行数值运算。
   - 这种方法可以通过利用 **GPU** 算力进行矩阵运算来显著提升性能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="http://blog.mir.dlang.io/glas/benchmark/openblas/2016/09/23/glas-gemm-benchmark.html">Numeric age for D: Mir GLAS is faster than OpenBLAS and Eigen</a>: 未找到描述</li><li><a href="http://www.spiral.net/">SPIRAL Project: Home Page</a>: 未找到描述
</li>
</ul>

</div>
  

---



### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1263298613627457646)** (52 messages🔥): 

> - `创建 API 工具`
> - `Discord 中的图像权限`
> - `DuckDuckGo 搜索集成` 


- **创建新的 API 工具**：成员们讨论了如何创建用于调用 **API** 的新工具，指导方向指向 [Cohere dashboard](https://dashboard.cohere.com) 中的设置。一位成员指出，工具（tools）和连接器（connectors）用途相似，但在 **API** 使用方面，工具正在取代连接器。
   - 另一位成员分享了一个[文档链接](https://docs.cohere.com/docs/tool-use)，解释了工具的 **API** 用法，特别是单步和多步方法。
- **图像权限讨论**：关于 **Discord** 中图像权限的对话引起了关注，成员们注意到发送图像可能被限制，以防止新成员发布无关内容。大家达成共识，认为允许开发者（makers）和常驻用户（regulars）等特定角色分享图像可能是有益的。
   - 一位成员感谢管理员启用了图像权限，对这一新功能表示兴奋，随后分享了一些轻松的 **GIF**。
- **DuckDuckGo 搜索集成**：一位成员寻求使用 **DuckDuckGo** 的文档，并被引导至一个 [Python package 链接](https://pypi.org/project/duckduckgo-search/)。他们还提到在工作中使用此工具来高效地获取链接。
   - 这引发了关于利用该集成与 **Firecrawl** 配合以提取更多信息的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/duckduckgo-search/">duckduckgo-search</a>: 使用 DuckDuckGo.com 搜索引擎搜索词条、文档、图像、新闻、地图和文本翻译。</li><li><a href="https://tenor.com/view/yay-kitty-cat-happy-excited-gif-10302657046876115666">Yay Kitty GIF - Yay Kitty Cat - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/trombone-pusheen-musician-instrument-gif-11434220432919976776">Trombone Pusheen GIF - Trombone Pusheen Musician - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.cohere.com/docs/tool-use">Tool Use with Cohere's Models - Cohere Docs</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1263488823795650615)** (31 条消息🔥): 

> - `Firecrawl 自托管`
> - `DuckDuckGo 搜索库`
> - `使用 GPT-4o API Key`
> - `用于 PoC 开发的 Streamlit` 


- **通过自托管 Firecrawl 降低成本**：一名成员强调 **Firecrawl** 非常昂贵，但他们很赞赏能够[自托管](https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md)后端的能力，这使得它物有所值。
   - 另一名成员表示如释重负，称自托管将为他们节省**几百美元**，并询问了其可行性。
- **使用 DuckDuckGo 库进行抓取**：一名成员分享了 [DuckDuckGo Search 库](https://pypi.org/project/duckduckgo-search/)的链接，将其作为有效收集 URL 和抓取内容的资源。
   - 他们指出该库是**免费的**，在与 **BeautifulSoup** 配合使用时可能有助于简化抓取流程。
- **集成 GPT-4o 与个人 API Key**：讨论涉及使用自己的 **GPT-4o** API Key，一名成员确认其存储在 **.env** 文件中以便访问。
   - 这种设置允许无缝集成网页抓取和 LLM 提取功能。
- **PoC 开发首选 Streamlit**：一名成员提到，使用 **Streamlit** 可以简化概念验证（PoC）开发，尤其是在抓取内容时。
   - 他们成功将其与 **Firecrawl** 集成，以确保系统功能正常。
- **社区支持与协作**：参与者对分享的信息和资源表示感谢，营造了一个支持性的问题解决环境。
   - 互动包括轻松的时刻和感激之情，增强了协作精神。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://pypi.org/project/duckduckgo-search/">duckduckgo-search</a>：使用 DuckDuckGo.com 搜索引擎搜索词汇、文档、图片、新闻、地图和文本翻译。</li><li><a href="https://github.com/mendableai/firecrawl/blob/main/SELF_HOST.md">firecrawl/SELF_HOST.md at main · mendableai/firecrawl</a>：🔥 将整个网站转换为 LLM 就绪的 Markdown 或结构化数据。通过单个 API 进行抓取、爬取和提取。- mendableai/firecrawl</li><li><a href="https://jsfiddle.net/razodactyl/gqr5vaot/1/">Edit fiddle - JSFiddle - Code Playground</a>：未找到描述内容
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1263208867290939512)** (63 messages🔥🔥): 

> - `Perplexity Pro 订阅邮件`
> - `GPT-4o Mini 模型发布`
> - `ChatGPT 回复问题`
> - `DALL-E 更新`
> - `搜索功能与域名排除` 


- **关于 Logitech 的 Perplexity Pro 邮件问题**：用户讨论了收到来自 Logitech 的邮件，提供 6 个月的 Perplexity Pro 订阅，部分用户对该优惠的真实性表示怀疑。
   - 然而，多名用户确认了邮件的真实性，其中一人表示已成功兑换了促销代码。
- **OpenAI 发布 GPT-4o Mini 模型**：OpenAI 宣布推出全新的 GPT-4o Mini 模型，旨在为开发者提供比现有模型更轻量、更经济的选择。
   - 该模型旨在提高 AI 的普及性，并将从今天起在各种订阅计划中取代 GPT-3.5 Turbo。
- **ChatGPT 的拆分回复问题**：用户对 ChatGPT 中出现的拆分回复现象提出疑问，寻求对其原因的澄清。
   - 一位用户指出，这种行为可能与新推出的 GPT-4o Mini 实施有关。
- **DALL-E 的潜在更新**：有讨论称 DALL-E 可能会升级到新版本，目前有报告称图像生成和 Pro 计划设置存在问题。
   - 用户认为观察到的故障可能是由于即将发布的 DALL-E 模型更新所致。
- **在 Perplexity 搜索中排除特定域名**：一位用户询问如何在 Perplexity 搜索结果中排除特定域名，特别是针对一些常见的干扰网站。
   - 另一位用户分享了可以使用 `-site:example.com` 搜索参数来有效过滤不需要的域名。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.theverge.com/2024/7/18/24200714/openai-new-cheaper-smarter-model-gpt-4o-mini">OpenAI 正在发布一个更便宜、更智能的模型</a>：OpenAI 正在推出名为 GPT-4o Mini 的更便宜、更智能的模型，作为开发者更易获取的模型。</li><li><a href="https://x.com/dmitry140/status/1813698975884792095">Dmitry Shevelenko (@dmitry140) 的推文</a>：Perplexity 🤝 Logitech。感谢 @ATXsantucci 的伟大合作。才刚刚开始！引用 Jorge Barba (@jorgebarba)：哇！完全出乎意料。收到了 @perplexity_ai Pro 的 6 个月订阅...
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1263252162830532679)** (5 messages): 

> - `莱茵河源头`
> - `Runway Gen3`
> - `剑龙拍卖`
> - `实验室培育宠物食品`
> - `Anthropic AI 基金` 


- **莱茵河的发源地**：分享了一个讨论 **莱茵河 (Rhine River)** 起源的链接，详细信息见 [此处](https://www.perplexity.ai/search/where-does-the-rhine-originate-leG7SSmcSOumGgMjEKEfWw#0)。
   - 该页面提供了关于莱茵河地理和历史意义的见解。
- **Runway Gen3 讨论**：此链接 [Runway Gen3](https://www.perplexity.ai/search/runwey-gen3no-n5T4dqx3Tz2hZXfHiIsqPw) 重点介绍了关于 **Runway Gen3** 的有趣讨论。
   - 内容涵盖了最新模型的关键更新和功能。
- **破纪录的剑龙拍卖**：一段名为 *YouTube* 的视频提到了 **破纪录的剑龙 (Stegosaurus) 拍卖**，视频可在 [此处](https://www.youtube.com/embed/do_EmoTIMn0) 观看。
   - 该视频似乎总结了古生物学界的重要趋势和事件。
- **分享研究咨询**：提供了一个表达研究意愿的链接，见 [此处](https://www.perplexity.ai/search/i-want-you-to-do-some-research-ynMkNdSLQFSRQ5ujxNssRQ)。
   - 该咨询邀请就研究课题进行协作和讨论。
- **关于 H2O-3 漏洞的精选页面**：一个关注 **H2O-3 代码执行漏洞** 的精选页面可在 [此处](https://www.perplexity.ai/page/h2o-3-code-execution-vulnerabi-zynZYKoxSqiUE7DE.Kkbag) 查看。
   - 该页面提供了关于 H2O-3 相关安全问题和潜在缓解措施的见解。



**提及的链接**：<a href="https://www.youtube.com/embed/do_EmoTIMn0">YouTube</a>：未找到描述

  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1263250068777992335)** (5 条消息): 

> - `NextCloud setup with Perplexity API`（在 NextCloud 中设置 Perplexity API）
> - `Selecting models in Perplexity API`（在 Perplexity API 中选择模型）
> - `API call for model information`（用于获取模型信息的 API 调用）


- **NextCloud 在适配 Perplexity API 时遇到困难**：一名成员在设置 **NextCloud** 使用 **Perplexity API** 时遇到问题，特别是在模型选择方面。
   - 另一名成员建议分享代码，并指出可以通过在请求体（request body）中设置名为 'model' 的字符串来更改模型。
- **获取未格式化的响应**：一名成员提供了一个 Prompt 示例，指导 API 返回关于某本书的信息且不包含任何格式化。
   - 这样做的目的是让响应成为一段仅包含请求细节的流畅文本。
- **呼吁提供详细模型信息的 API**：一名成员请求增加一项 API 功能，用于检索**可用模型名称**、上下文窗口（context windows）、每次请求的成本以及限制。
   - 他们指出，这种 API 调用（最好不需要指定特定的模型名称）将有助于有效地管理使用情况。



**提及的链接**：<a href="https://docs.perplexity.ai/docs/model-cards">Supported Models</a>：未找到描述

  

---



### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1263229567792058410)** (39 条消息🔥): 

> - `LangChain features overview`（LangChain 功能概览）
> - `LangChain AgentExecutor`（LangChain AgentExecutor）
> - `Using MongoDB in LangChain`（在 LangChain 中使用 MongoDB）
> - `Integrating external API models`（集成外部 API 模型）
> - `HyDE availability in TypeScript`（HyDE 在 TypeScript 中的可用性）


- **关于 LangChain 功能概览的咨询**：一名成员询问是否有官方表格总结了 **LangChain** 的功能，并参考了他们在网上找到的一张对比图。
   - 成员们对图中信息的时效性以及 **HyDE** 是否支持 **TypeScript** 表示好奇。
- **关于 LangChain AgentExecutor 的疑问**：一名用户询问 **LangChain** 中的 `AgentExecutor` 如何处理交互的动态特性，特别是它如何使用 LLM 进行决策。
   - 有人指出 **LangChain** 正趋向于弃用 `AgentExecutor`，转而推荐使用 **LangGraph**，因为它提供了更高的灵活性。
- **在 LangChain 中将 MongoDB 用作向量存储**：一名用户表示打算将 **MongoDB** 作为其 **RAG** 应用的向量存储（vector store），并询问如何在 **LangChain** 中实现混合搜索（hybrid search）。
   - 他们请求提供 **Python** 和其他语言的代码参考，寻求混合搜索集成的具体示例。
- **在 LangChain 中集成外部 API 模型**：一名成员寻求澄清，想知道 **LangChain** 是否具有内置功能来轻松集成外部 API 模型，而不局限于 **ChatAnthropic** 和 **ChatOpenAI** 等专有模型。
   - 他们在文档中没有找到关于此功能的明确信息。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://js.langchain.com/v0.2/docs/integrations/chat/anthropic/#custom-headers>).">ChatAnthropic | 🦜️🔗 Langchain</a>：LangChain 支持 Anthropic 的 Claude 系列聊天模型。</li><li><a href="https://python.langchain.com/v0.2/docs/concepts/#agents>)].">Conceptual guide | 🦜️🔗 LangChain</a>：本节包含 LangChain 核心部分的介绍。</li><li><a href="https://v02.api.js.langchain.com/functions/langchain_core_messages.trimMessages.html#Example>)">trimMessages | LangChain.js - v0.2.10</a>：未找到描述</li><li><a href="https://js.langchain.com/v0.2/docs/integrations/vectorstores/mongodb_atlas/#search>).">MongoDB Atlas | 🦜️🔗 Langchain</a>：仅在 Node.js 上可用。</li><li><a href="https://github.com/langchain-ai/langchain/issues/5421>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/15050>)]">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。</li><li><a href="https://github.com/langchain-ai/langchain/issues/22585>)]">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1263446738753294408)** (2 条消息): 

> - `Langserve Debugger`
> - `Langserve 容器差异` 


- **了解 Langserve Debugger 容器**：一位成员询问了 [Langserve Debugger 容器](https://registry.hub.docker.com/r/langchain/langserve-debugger) 的功能，寻求对其用途的澄清。
   - 该容器专为 LangChain 生态系统内的调试和问题解决而设计。
- **Langserve 与 Debugger 之间的区别**：另一位成员询问了 [Langserve Debugger](https://registry.hub.docker.com/r/langchain/langserve-debugger) 与标准 [Langserve 容器](https://registry.hub.docker.com/r/langchain/langserve) 之间的区别。
   - 讨论围绕 Langserve Debugger 特有的调试功能展开，并与主 Langserve 容器的部署能力进行了对比。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://registry.hub.docker.com/r/langchain/langserve">未找到标题</a>：未找到描述</li><li><a href="https://registry.hub.docker.com/r/langchain/langserve-debugger">未找到标题</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1263579789940228159)** (1 条消息): 

> - `ChatPromptTemplate JSON 问题`
> - `KeyError 故障排除`
> - `GitHub 支持方案` 


- **在 ChatPromptTemplate 中使用 JSON 时出现 KeyError**：用户遇到了一个 **KeyError**，提示“ChatPromptTemplate 的输入缺少变量”，这与系统消息中包含的 JSON 内容有关。
   - 错误提到，尽管 **'$schema'** 是 JSON 输入的一部分，但预期的变量仍然缺失。
- **来自 GitHub 支持的潜在解决方案**：有人指出，正如 [GitHub 支持线程](https://github.com/langchain-ai/langchain/issues/1914) 中提到的，用双大括号包裹 JSON 可能会解决此问题。
   - 然而，尽管有这种变通方法，用户仍在继续寻求该问题的其他解决方案。
- **关于 JSON 变量插值的讨论**：频道参与者讨论了在使用 `ChatPromptTemplate` 时将 **JSON 作为模板内容**传递的挑战。
   - 虽然建议使用*双大括号包裹 JSON*，但成员们反馈在实施过程中的效果参差不齐。



**提到的链接**：<a href="https://github.com/langchain-ai/langchain/issues/1914,">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账户为 langchain-ai/langchain 的开发做出贡献。

  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1263405001028669512)** (1 条消息): 

> - `Easy Folders 发布`
> - `Product Hunt`
> - `免费功能` 


- **Easy Folders 在 Product Hunt 上线**：Easy Folders 已在 **Product Hunt** 上线，展示了其创建文件夹、搜索聊天记录和 Prompt 管理器等功能。
   - 发布类别包括 [Browser Extensions](https://www.producthunt.com/topics/browser-extensions)、[Productivity](https://www.producthunt.com/topics/productivity) 和 [Artificial Intelligence](https://www.producthunt.com/topics/artificial-intelligence)。
- **Easy Folders 限时优惠**：用户可以通过点赞发布页面、留下评论并私信截图来参与**限时优惠**，从而获得 Easy Folders 的 **30 天免费超级用户会员 (Superuser membership)**。
   - 这一举措在推广产品功能的同时，鼓励了社区的参与。



**提到的链接**：<a href="https://www.producthunt.com/posts/easy-folders-for-chatgpt-claude"> Easy Folders for ChatGPT &amp; Claude - 整理并归纳你的聊天记录 | Product Hunt</a>：创建文件夹、搜索聊天记录、书签聊天、Prompt 管理器、Prompt 库、自定义指令配置文件等。

  

---

### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1263498280474775674)** (1 条消息): 

> - `LangGraph`
> - `Corrective RAG`
> - `RAG Fusion`
> - `AI Chatbots` 


- **结合 Corrective RAG 与 RAG Fusion**：一位成员在进行 Python 项目开发时，探索了将 **Corrective RAG** 与 **RAG Fusion** 集成，以解决现代 AI 聊天机器人中的 **hallucinations** 问题。
   - *“这种方法可以增强聊天机器人的可靠性，”* 他们提到，并提供了一个 [YouTube 教程](https://www.youtube.com/watch?v=7h6uDsfD7bg) 链接以供进一步参考。
- **本地聊天机器人创建教程**：分享的 YouTube 视频标题为“[LangGraph + Corrective RAG + RAG Fusion Python Project: Easy AI/Chat for your Docs](https://www.youtube.com/watch?v=7h6uDsfD7bg)”，重点介绍如何使用 **LangGraph** 构建完全本地化的聊天机器人。
   - 他们强调了其简单性，并表示：*“这是一个使用 LangGraph 创建聊天机器人的超快速教程。”*



**提及的链接**：<a href="https://www.youtube.com/watch?v=7h6uDsfD7bg">LangGraph + Corrective RAG + RAG Fusion Python Project: Easy AI/Chat for your Docs</a>：#chatbot #coding #ai #llm #chatgpt #python #在这段视频中，我为大家准备了一个超快速教程，展示如何使用 LangGraph 创建一个完全本地的聊天机器人，...

  

---



### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1263258169736429599)** (4 条消息): 

> - `Jerry Liu's Keynote`
> - `Updates on RAGapp`
> - `Stack Podcast Discussion`
> - `New Model Releases` 


- **Jerry Liu 关于 Knowledge Assistants 的主题演讲**：错过了 @jerryjliu0 在 @aiDotEngineer World's Fair 的演讲？点击[这里](https://t.co/o93s5WSMIV)观看他关于 Knowledge Assistants 未来的主题演讲——他是去年观看次数最多的演讲者！
   - 参与者强调了他的讨论对于理解知识应用进展至关重要。
- **RAGapp 获得重大更新**：@MarcusSchiesser 宣布了 RAGapp 的新版本，现在支持 **@MistralAI**、**@GroqInc**，并包含一个 **@cohere** reranker 以改进结果。该应用可通过 Docker 为企业轻松部署。
   - 这使其在 RAG 应用中极具竞争力。
- **来自 Stack Podcast 的见解**：联合创始人 @jerryjliu0 加入了 @StackPodcast，讨论了**高质量数据**、**prompt engineering** 和**长 context windows** 的重要性。他们还探讨了检索增强生成 (RAG) 的挑战。
   - 对于渴望了解当前 AI 发展趋势的人来说，这个播客是必听之选。
- **令人兴奋的新模型发布**：新模型发布的大日子到了，来自 **@MistralAI** 和 **@OpenAI** 的发布都获得了零日支持。值得注意的是，Mistral 的 **NeMo** 模型（一个紧凑的 **12B** 模型）现在超越了其前身 Mistral 7b，并拥有 **128k context window**。
   - 这使其成为小模型类别中的有力竞争者。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://t.co/o93s5WSMIV">未找到标题</a>：未找到描述</li><li><a href="https://t.co/C5uOA2g2zH">The framework helping devs build LLM apps - Stack Overflow</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1263220279316840559)** (21 条消息🔥): 

> - `Neo4jPropertyGraphStore 索引`
> - `开启编程之旅`
> - `AI Agents 开发`
> - `使用 Llama-Index 脱敏敏感数据`
> - `Retriever 评估挑战` 


- **Neo4jPropertyGraphStore 索引耗时**: 成员们讨论了使用 *Neo4jPropertyGraphStore* 时索引速度缓慢的问题，其中一位指出这在很大程度上取决于处理的数据量。
   - 另一位成员确认，较大的研究笔记会导致索引时间延长。
- **新开发者寻求指导**: 建议寻求编程入门建议的新成员观看 [A Hacker's Guide to Language Models](https://www.youtube.com/watch?v=jkrNMKz9pWU) 并参加相关的短期课程。
   - 有人建议，先理解 LLM APIs 会让向使用框架的过渡更加平滑。
- **对构建 AI Agents 的兴趣**: 一位成员表达了构建 AI Agents 的愿望，并寻求入门建议。
   - 讨论指出，在深入研究框架细节之前，首先学习 LLM APIs 是必要的。
- **为 OpenAI 集成脱敏敏感数据**: 成员们讨论了在将数据发送到 OpenAI 之前脱敏敏感数据的策略，其中一位建议使用 postprocessor 来处理隐私问题。
   - `PIINodePostprocessor` 被强调为处理 PII 的潜在 Beta 版解决方案。
- **Retriever 评估的挑战**: 一位成员报告了在生成具有意义查询的 QA 数据集时遇到的困难，产生了大量通用提示词而非精确问题。
   - 这种缺乏针对性的情况导致在使用 retriever evaluator 时评估结果较差，例如 hit rate 和 MRR 为 0。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/stable/module_guides/querying/node_postprocessors/node_postprocessors/#beta-piinodepostprocessor">Node Postprocessor Modules - LlamaIndex</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=jkrNMKz9pWU">A Hackers&#39; Guide to Language Models</a>: 在这段内容丰富的视频中，fast.ai 联合创始人、所有现代语言模型 (LMs) 所基于的 ULMFiT 方法的创造者 Jeremy Howard...</li><li><a href="https://www.deeplearning.ai/short-courses/building-systems-with-chatgpt/">Building Systems with the ChatGPT API</a>: 使用 LLMs 简化任务、自动化工作流并改进输出。确保 LLM 输入和输出的安全性和准确性。</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/PII/">PII - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1263572925697495112)** (2 条消息): 

> - `重写查询的实用性`
> - `使用 LlamaIndex 进行 Multimodal RAG`
> - `Langchain RAG 应用开发`
> - `LlamaIndex 文档解析` 


- **探索重写查询的效用**: 一位成员在处理一个有问题的演示文件时，使用 **GPT-4o** 和 **Sonnet 3.5** 测试了 **multimodal RAG**，体验到了来自 **LlamaIndex** 令人惊讶的高质量响应。
   - 他们询问其他人是否发现查询重写对提升性能有益，并表达了想要了解更多关于 **LlamaIndex 生态** 的愿望。
- **Langchain 与 LlamaIndex 在 RAG 应用上的对比**: 一位成员分享了他们过去使用 **Langchain** 开发 **RAG apps** 的经验，概述了拆分文档、向量化文本和存储数据的过程。
   - 他们注意到在他们查看的 **LlamaIndex** 示例中缺少文档拆分，并质疑文档是否被改为按页划分。
- **关于 LlamaIndex 文档解析的澄清**: 针对 **LlamaIndex** 文档解析示例中未包含传统 RAG 应用中的文档拆分步骤，产生了一些疑问。
   - 该成员要求对其解析机制的理解进行澄清，并反思了这与之前经验的差异。



**提到的链接**: <a href="https://github.com/run-llama/llama_parse/blob/main/examples/multimodal/claude_parse.ipynb">llama_parse/examples/multimodal/claude_parse.ipynb at main · run-llama/llama_parse</a>: 为优化 RAG 解析文件。通过在 GitHub 上创建一个账号来为 run-llama/llama_parse 的开发做出贡献。

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1263515987081170945)** (9 messages🔥): 

> - `Mistral 12B NeMo model`
> - `High context length training effects`
> - `Transformer reasoning capabilities`
> - `Model performance comparison`
> - `Fine-tuning advantages` 


- **Mistral 12B NeMo 模型集成**：一位成员询问新的 [Mistral 12B NeMo 模型](https://mistral.ai/news/mistral-nemo/) 是否可以在不更新 Axolotl 的情况下直接使用，并提到了其令人印象深刻的特性，包括 **128k token 上下文窗口**。
   - 另一位成员幽默地暗示，亲自尝试是确认其兼容性的唯一方法。
- **对模型性能差异的担忧**：讨论中提到了 **Llama 3 8B** 不同的 MMLU 分数；一份报告称其得分为 **62.3%**，而另一份则声称是 **66.6%**，这引发了潜在的警示。
   - 这导致人们猜测 **TriviaQA** 基准测试可能也因为来源不同而无法匹配。
- **Transformer 与推理能力的进步**：一位成员分享了一篇论文的见解，指出 Transformer 可以通过 *grokking* 提高隐式推理能力，这需要超出简单记忆的大量训练。
   - 关键发现表明，可能会形成 **推理泛化 (inferential generalization)** 电路，使 Transformer 能够更好地处理分布外 (out-of-distribution) 示例。
- **更大模型中的微调空间**：一位成员强调了 **12B 模型**在微调空间方面的优势，认为它可能不像 Llama 3 8B 那样被训练到了极限。
   - 这暗示了由于模型的容量，在微调场景中可能会有更好的性能表现。
- **论文见解的曙光**：一位成员正在审阅一篇[研究论文](https://arxiv.org/abs/2405.15071)，该论文探讨了 Transformer 训练及其对 **泛化 (generalization)** 和 **推理 (inferences)** 的影响。
   - 他们强调，训练超过饱和点可以显著增强模型推断事实的能力，而不仅仅是严格记忆输入。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个最先进的 12B 模型，具有 128k 上下文长度，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://arxiv.org/abs/2405.15071">Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization</a>：我们研究了 Transformer 是否可以学会对参数化知识进行隐式推理，这是即使是最强大的语言模型也难以掌握的技能。重点关注两种代表性的推理类型……
</li>
</ul>

</div>
  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1263545706749366385)** (7 messages): 

> - `Model Selection`
> - `Training Adjustments` 


- **Llama3 成为首选模型**：一位成员确定 **Llama3** 为他们正在使用的模型，并引发了关于其性能的讨论。
   - 有关于调整后其表现如何的推测。
- **降低 rank 可改善 eval loss**：一位成员注意到降低他们的 rank 有助于改善 **eval loss**，这让他们感到惊讶。
   - 他们计划稍后运行 eval set 以验证改进是否持续。
- **训练损失显示出积极趋势**：另一位成员提到 **training loss** 似乎明显降低，表明取得了积极进展。
   - 这与他们的调整以及正在进行的跟踪性能测试相吻合。


  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1263228510491639818)** (7 条消息): 

> - `Finetuning 性能对比`
> - `Mac M1 上的 Hugging Face 模型`
> - `模型加载延迟`
> - `Finetuning 中的数据敏感性` 


- **Finetuning 模型性能对比较少**：成员们注意到，通常在 Finetuning 过程中，很少有将 **Mistral 7B** 和 **Llama3** 等开源模型与 **gpt-3.5-turbo** 进行性能对比的研究。
   - 一位成员发现 **gpt-3.5-turbo** 的表现优于其他模型，并质疑为什么许多人因为 OpenAI 的 API 数据政策而避免使用 GPT 模型进行 Finetuning。
- **Hugging Face 模型在 Mac M1 上首次运行缓慢**：一位用户在 **Mac M1** 上首次在预处理流水线中使用 **Hugging Face** 模型时遇到了延迟。
   - 其他人解释说，这种变慢是因为模型在首次运行时会加载到内存中，这在测试多个模型时可能非常耗时。
- **模型加载与推理计时**：为了缓解延迟问题，一位成员建议将模型的 **loading**（加载）和 **inference**（推理）分开，这有助于计时每个过程所需的时间。
   - 这种方法可以更清晰地了解使用过程中的性能瓶颈。
- **业务数据的敏感性**：讨论中提到，用户可能会因为担心将敏感业务数据（如客户和患者信息）发送给外部公司，而避免对模型进行 Finetuning。
   - 这突显了在模型部署决策中考虑数据隐私的重要性。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jarvis-labs](https://discord.com/channels/1238365980128706560/1241117895740625099/)** (1 条消息): 

ashpun: 我不认为有过期日期。我们有 <@657253582088699918> 吗？
  

---



### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1263270792939180103)** (2 条消息): 

> - `Meta 未来的多模态 AI 模型`
> - `面向欧盟用户的 Llama 模型` 


- **Meta 推进多模态 AI 雄心**：根据 [Axios](https://www.axios.com/2024/07/17/meta-future-multimodal-ai-models-eu) 的报道，Meta 正专注于开发 **multimodal AI models**，以增强其在该领域的产品。
   - 该公司的目标是创建更集成的解决方案，从而重新定义用户交互和体验。
- **欧盟用户告别 Llama 模型**：欧盟用户将**不再能使用 Llama 模型**，这影响了他们获取某些 AI 能力的途径。
   - 此举引发了人们对 **AI accessibility** 以及科技公司在欧洲面临的持续监管挑战的担忧。


  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1263229865696428162)** (6 条消息): 

> - `Codestral Mamba`
> - `Prover-Verifier Games`
> - `NuminaMath-7B performance`
> - `Mistral NeMo` 


- **Codestral Mamba 提供高效建模**：继 [Mixtral 系列](https://mistral.ai/news/codestral-mamba/)发布之后，**Codestral Mamba** 展示了线性时间推理能力，并且理论上可以处理无限长度的序列，非常适合提升代码生产力。
   - 它是在 **Albert Gu** 和 **Tri Dao** 的参与下开发的，允许用户在进行大规模模型交互时获得快速响应。
- **通过 Prover-Verifier Games 提升可读性**：OpenAI 讨论了使用 [Prover-Verifier games](https://openai.com/index/prover-verifier-games-improve-legibility/) 来增强 LLM 输出的清晰度。
   - 该方法旨在提高模型生成结果的可读性，并更好地理解模型是如何生成这些结果的。
- **NuminaMath-7B 在奥数竞赛中表现出色**：NuminaMath-7B 最近在 AIMO 竞赛中排名第一，解决了 **50 道高中数学题中的 29 道**，但也有警告指出，该基准测试无法检测到基础推理缺陷。
   - 敦促用户在根据这些基准测试做出强有力断言时保持谨慎，特别是涉及 LLM 的*基础推理问题*时。
- **Mistral NeMo 改变了游戏规则**：[Mistral NeMo](https://mistral.ai/news/mistral-nemo/) 是最新发布的 **12B 模型**，由 NVIDIA 合作开发，具有高达 **128k tokens** 的上下文窗口。
   - 它的开发旨在轻松集成到现有系统中，并提供最先进的推理和编码准确性，支持量化感知以实现 FP8 推理。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-nemo/">Mistral NeMo</a>：Mistral NeMo：我们最新的最佳小型模型。一个具有 128k 上下文长度的最先进 12B 模型，与 NVIDIA 合作构建，并根据 Apache 2.0 许可证发布。</li><li><a href="https://mistral.ai/news/codestral-mamba/">Codestral Mamba</a>：作为对克利奥帕特拉（Cleopatra）的致敬，我们自豪地发布 Codestral Mamba，这是一个专门用于代码生成的 Mamba2 语言模型，可在...下使用。</li><li><a href="https://x.com/JJitsev/status/1813930981637902486">Jenia Jitsev 🏳️‍🌈 🇺🇦 (@JJitsev) 的推文</a>：(又) 一个兴衰的故事：最近，NuminaMath-7B 在 AIMO 竞赛中排名第一，解决了 50 道奥数水平的私有集题目中的 29 道。它能处理简单的 AIW 问题吗，这需要...
</li>
</ul>

</div>
  

---



### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1263429585786241066)** (6 条消息): 

> - `PR 上的 CI 取消`
> - `自定义模板配置`
> - `Alpaca 数据集使用` 


- **用户询问 PR 上的 CI 取消问题**：一位成员询问其他人在向 PR 添加内容时是否会手动取消 CI，并指出他们的流程现在已经开始自动运行了。
   - 另一位成员给出的建议是：*在将其移出草稿状态并请求审查之前，直接忽略它即可。*
- **自定义模板配置困惑**：讨论围绕重命名自定义模板映射中预期的列展开，一位成员对配置中保留 alpaca 清理后的数据集表示疑问。
   - 另一位成员确认他们打算稍后使用 alpaca 数据集，但目前正在测试一个旨在始终响应“HAHAHA”的模板。

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1263579220781699112)** (3 条消息): 

> - `tinygrad CUDA 兼容性`
> - `GTX 1080 错误`
> - `CUDA 补丁选项` 


- **GTX 1080 在 tinygrad 中遇到编译错误**：一位用户报告称，在使用 env-var `CUDA=1` 运行 tinygrad 时，收到 `tinygrad.device.CompileError`，提示 **GTX 1080** 不受支持。
   - *这一代 NVIDIA 显卡是否不受支持？* 引发了关于潜在解决方案的讨论。
- **2080 系列作为最低要求**：另一位成员建议 **2080 系列** 是运行 tinygrad 的最低要求，这表明旧显卡可能存在兼容性问题。
   - 他们建议通过在 **ops_cuda** 中修补架构并禁用 Tensor Cores 作为权宜之计。
- **在更新的系统上探索 tinygrad**：原用户宣布计划在 **更新的系统** 上设置 tinygrad，以进一步探索兼容性。
   - 他们感谢社区对该问题的反馈。


  

---



---



---



---



---



---



---



{% else %}


> 完整的逐频道详情已针对电子邮件进行了删减。 
> 
> 如果您想查看完整详情，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢！

{% endif %}