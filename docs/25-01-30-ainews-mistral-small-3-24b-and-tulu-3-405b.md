---
companies:
- mistral-ai
- ai2
- sakana-ai
- alibaba_qwen
- deepseek
- ollama
- llamaindex
date: '2025-01-31T00:08:47.548368Z'
description: '**Mistral AI** 发布了 **Mistral Small 3**，这是一个拥有 **240 亿参数**的模型，针对低延迟的本地推理进行了优化，在
  **MMLU 上的准确率达到 81%**，可与 **Llama 3.3 70B**、**Qwen-2.5 32B** 以及 **GPT4o-mini** 竞争。**AI2**
  发布了 **Tülu 3 405B**，这是一个基于 **Llama 3** 的大型微调模型，采用了基于可验证奖励的强化学习（RVLR）技术，性能可与 **DeepSeek
  v3** 媲美。**Sakana AI** 推出了 **TinySwallow-1.5B**，这是一款采用 **TAID** 技术、专为端侧使用的日语语言模型。**阿里巴巴
  Qwen** 发布了 **Qwen 2.5 Max**，该模型在 **20 万亿 token** 上进行了训练，性能与 **DeepSeek V3**、**Claude
  3.5 Sonnet** 和 **Gemini 1.5 Pro** 相当，并更新了 API 定价。这些发布突显了开源模型、高效推理和强化学习技术方面的进展。'
id: 43d26a15-9c4e-4391-9331-9f5e01ce3300
models:
- mistral-small-3
- tulu-3-405b
- llama-3
- tiny-swallow-1.5b
- qwen-2.5-max
- deepseek-v3
- claude-3.5-sonnet
- gemini-1.5-pro
- gpt4o-mini
- llama-3-3-70b
original_slug: ainews-mistral-small-3-24b-and-tulu-3-405b
people:
- clementdelangue
- dchaplot
- reach_vb
title: Mistral Small 3 24B 和 Tulu 3 405B
topics:
- reinforcement-learning
- model-fine-tuning
- local-inference
- model-performance
- model-optimization
- on-device-ai
- instruction-following
- api
- training-data
- natural-language-processing
---

<!-- buttondown-editor-mode: plaintext -->**开源模型就是我们所需要的一切。**

> 2025年1月29日至1月30日的 AI 新闻。我们为您检查了 7 个 subreddit、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **34** 个 Discord 服务（**225** 个频道，**7312** 条消息）。预计节省阅读时间（按每分钟 200 字计算）：**744 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

命运弄人，风投支持的 Mistral（至今已融资 14 亿美元）和非营利组织 AI2 今天分别发布了一个小型 Apache 2 模型和一个大型模型，但它们的规模与你根据融资背景所预期的正好相反。

首先是 **Mistral Small 3**，通过他们标志性的 [磁力链接](https://x.com/mistralai/status/1884967826215059681?s=46) 发布，幸好还有 [博客文章](https://twitter.com/dchaplot/status/1884975427460206649)：


![image.png](https://assets.buttondown.email/images/cd4040a3-58ca-4099-aa33-53aa0dea68ab.png?w=960&fit=max)


这是 Mistral 产品线在 2025 年的一次非常出色的更新，针对本地推理进行了优化——尽管人们注意到其效率图表的 x 轴变化比 y 轴更快。网络侦探已经 [对比](https://x.com/espadrine/status/1885004488206856638) 了它与 Mistral Small 2 的架构差异（基本上是扩大了维度，但减少了层数和注意力头以降低延迟）：


![image.png](https://assets.buttondown.email/images/eae4fd25-550c-4c9a-9a3d-3a61c7749baa.png?w=960&fit=max)


他们关于使用场景的段落提供了有趣的信息，解释了为什么他们认为发布这个模型是值得的：


![image.png](https://assets.buttondown.email/images/b9be05a2-4e6e-48f0-82ce-ae125dc34e31.png?w=960&fit=max)


接下来，AI2 发布了 **Tülu 3 405B**，这是他们对 Llama 3 的大型微调版本，使用了来自 [Tulu 3 论文](https://arxiv.org/abs/2411.15124) 的可验证奖励强化学习（RVLR）方案，使其在某些维度上能与 DeepSeek v3 竞争：


![image.png](https://assets.buttondown.email/images/b5a54590-c17c-4025-95c9-821fb1502f6b.png?w=960&fit=max)


不幸的是，发布时似乎没有任何托管 API，因此很难试用这个 [庞大](https://x.com/soldni/status/1885004141564731717) 的模型。

---

{% if medium == 'web' %}

**目录**

[TOC] 

{% else %}

**目录** 和 **频道摘要** 已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}

---

# AI Twitter 回顾

> 所有摘要均由 Gemini 2.0 Flash 完成

**模型发布与更新**

- **Sakana AI 发布了 TinySwallow-1.5B**，这是一个使用其新方法 **TAID (Temporally Adaptive Interpolated Distillation)** 训练的小型日语语言模型，在其尺寸类别中实现了 state-of-the-art 的性能。该模型可以完全在设备端运行，甚至可以在 Web 浏览器中运行。目前已提供演示版供试用，以及 [模型](https://twitter.com/SakanaAILabs/status/1884770664353325399) 和 [GitHub 仓库](https://twitter.com/SakanaAILabs/status/1884770664353325399)。还提供了一个包含模型权重的 [自包含 Web 应用](https://twitter.com/SakanaAILabs/status/1884880970790343001)，可用于本地执行。
- **Mistral AI 发布了 Mistral Small 3**，这是一个拥有 24B 参数的模型，采用 Apache 2.0 许可证，提供 Base 和 Instruct 两个版本。其设计目标是低延迟，速度达 **150 tokens/s**，**MMLU 准确率为 81%**。它被定位为 **Llama 3.3 70B**、**Qwen-2.5 32B** 和 **GPT4o-mini** 的竞争对手。该模型已在 [la Plateforme、HF 和其他供应商](https://twitter.com/omarsar0/status/1884972996575609092) 上线，[博客文章](https://twitter.com/dchaplot/status/1884975427460206649) 提供了详细信息。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1884994066129039671) 也关注了此次发布，并指出了 [Base 模型](https://twitter.com/ClementDelangue/status/1884994066129039671) 和 [Instruct 模型](https://twitter.com/ClementDelangue/status/1884994066129039671) 的可用性。[Ollama](https://twitter.com/ollama/status/1884970144562381165) 和 [llama.cpp](https://twitter.com/reach_vb/status/1885007847135609224) 也已发布了对其的支持。
- **Alibaba_Qwen 发布了 Qwen 2.5 Max**，这是他们迄今为止最大的模型，在 **20 万亿 tokens** 上训练而成，其性能可与 **DeepSeek V3**、**Claude 3.5 Sonnet** 和 **Gemini 1.5 Pro** 相媲美，**Artificial Analysis 质量指数为 79**。他们还发布了 [Qwen2.5-VL Cookbooks](https://twitter.com/Alibaba_Qwen/status/1884809286288810231)，这是一系列展示 Qwen2.5-VL 各种用例的 Notebook，包括计算机使用、空间理解、文档解析、Mobile Agent、OCR、通用识别和视频理解。[该模型的 API 已更新](https://twitter.com/Alibaba_Qwen/status/1884995327318782086)，价格为每百万 Input Token 1.6 美元，每百万 Output Token 6.4 美元。
- **Allen AI 发布了 Tülu 3 405B**，这是一个开源的 Post-training 模型，其性能超越了 **DeepSeek-V3**，证明了他们包含 **RVLR (Reinforcement Learning from Verifiable Rewards)** 在内的方案可以扩展到 405B，并表现出与 **GPT-4o** 相当的水平。[@ClementDelangue](https://twitter.com/ClementDelangue/status/1885004067547557987) 也注意到了这次发布，强调了这些模型在 HF 上的可用性。[@reach_vb](https://twitter.com/reach_vb/status/1884969597473886248) 称其为一个“精心准备”的发布，并指出它在比 DeepSeek V3 小 40% 的情况下实现了超越。
- **DeepSeek-V3 被 Tülu 3 超越**，[@Tim_Dettmers](https://twitter.com/Tim_Dettmers/status/1885024960118202538) 指出这是通过 **405B Llama Base** 实现的，且 **扎实的 Post-training** 发挥了重要作用。他强调了该方案完全开源特性的重要性。
- **DeepSeek R1 Distill** 已在 [Together AI](https://twitter.com/togethercompute/status/1885008866259460474) 上免费提供。Together AI 还为该模型提供了一个 [100% 免费的 API 端点](https://twitter.com/togethercompute/status/1885008864422264997)。

**工具、基准测试与评估**

-   **LangChain** 在 LangSmith 中为标注队列（annotation queues）引入了[批量视图](https://twitter.com/LangChainAI/status/1885003940661743999)，允许用户管理用于模型训练的大型数据集。他们还添加了[瀑布图](https://twitter.com/LangChainAI/status/1884987434645041482)来可视化 traces，以便发现瓶颈并优化响应时间。此外还发布了一个关于如何评估文档提取流水线（document extraction pipelines）的[视频](https://twitter.com/LangChainAI/status/1885012449352704123)。
-   [@awnihannun](https://twitter.com/awnihannun/status/1884812911572566027) 指出，**Qwen 2.5 模型可以在笔记本电脑上使用 mlx-lm 生成或微调代码**，并报告称 7B 模型在 **M4 Max** 上运行非常快，使用的是 mlx-lm 代码库（1.6万行）作为上下文。此外还提供了一份[关于高效重新计算 prompt cache 的指南](https://twitter.com/awnihannun/status/1884813199347986790)。
-   [@jerryjliu0](https://twitter.com/jerryjliu0/status/1884777070292762723) 分享了 **LlamaReport** 的预览，这是一个能从非结构化数据创建复杂、多章节报告的 Agent。
-  [@AravSrinivas](https://twitter.com/AravSrinivas/status/1884891721181298847) 指出，**来源（sources）和推理链（reasoning traces）**对 AI 产品的 UX 和信任度产生了巨大影响。他还表示，Perplexity 将使**手机（Android）上的原生助手**更可靠地完成任务。他向所有拥有 .gov 邮箱的美国政府雇员免费提供为期一年的 Perplexity Pro。
-  [@_akhaliq](https://twitter.com/_akhaliq/status/1884785023569527264) 在 ai-gradio 上提供了支持 DeepSeek 模型的 Perplexity Sonar Reasoning。他们还发布了 [Atla Selene Mini](https://twitter.com/_akhaliq/status/1884795448139166067)，这是一个通用评估模型。
-  [@swyx](https://twitter.com/swyx/status/1884775744917967191) 在多个模型上运行了他们的报告 Agent，并得出结论：**Gemini 2.0 Flash** 在摘要报告方面比 **O1** 更高效，且价格**便宜 200 倍**。
-   [@karpathy](https://twitter.com/karpathy/status/1885026028428681698) 解释了一个关于 LLM 的教科书类比，将 **pretraining、supervised finetuning 和 reinforcement learning** 分别比作教科书阐述、例题讲解和练习题。


**AI Infrastructure and Compute**

-   [@draecomino](https://twitter.com/draecomino/status/1885022313260998953) 指出，**Cerebras** 让 AI 再次实现即时响应，**DeepSeek R1 70B** 的 **首个 token 响应时间（time to first token）仅为 1 秒**。
-   [@cto_junior](https://twitter.com/cto_junior/status/1884823329477177687) 指出，**2000 块 H100 足以在一个财季内完成 15T tokens 的稠密 70B 模型训练**，成本约为 **1000 万美元**。他还提到 [Yotta 拥有 4096 块 H100 的访问权限](https://twitter.com/cto_junior/status/1884823329477177687)。
-   [@fchollet](https://twitter.com/fchollet/status/1885040378170269889) 表示，AI 领域 **5000 亿美元的数字是虚假的**，估计最多 **1500 亿美元** 才是现实的。
-  [@mustafasuleyman](https://twitter.com/mustafasuleyman/status/1885042373757198811) 认为技术倾向于变得**更便宜、更高效**。他还认为 AI 正在从模仿学习（imitation learning）转向奖励学习（reward learning）。
-   [@teortaxesTex](https://twitter.com/teortaxesTex/status/1884824912818225308) 指出，**R1 的发布让许多人得出结论“你可以直接构建东西”**。他们表示 DeepSeek 在**算力（compute power）较少**的情况下实现了这一目标。
-  [@shaneguML](https://twitter.com/shaneguML/status/1885053092913406236) 指出，测试时算力扩展（test-time compute scaling）有利于像 **Cerebras 和 Groq** 这样快速推理芯片初创公司。

---

# AI Reddit Recap

## /r/LocalLlama Recap

**主题 1：Mistral Small 3 发布：具备与更大型模型竞争的实力**

- **[Mistral Small 3](https://i.redd.it/kj3s0jvr35ge1.png)** ([分数：643，评论：205](https://reddit.com/r/LocalLLaMA/comments/1idny3w/mistral_small_3/)): **Mistral Small 3** 在 **@MistralAI** 发布于 2025 年 1 月 30 日的一条推文中被提及，推文中包含一个可能指向该版本资源或细节的 URL。该推文已获得 998 次查看，显示出外界对该主题的关注。
  - **Mistral Small 3** 是一款拥有 **24B 参数的模型**，采用 **Apache 2.0 许可证**发布，针对低延迟和高效率进行了优化，处理速度达 **150 tokens per second**。它以强大的语言任务处理和指令遵循能力著称，在相同硬件上比 **Llama 3.3 70B** 等大型模型快三倍以上，并在 **MMLU** 上实现了超过 **81% 的准确率**。
  - 用户们非常欣赏针对小型模型的 **human evaluation chart**（人类评估图表），强调了使模型符合人类视角而非仅仅关注 Benchmark 的重要性。该模型可以针对法律、医疗和技术支持等多个领域进行 Fine-tune，并适用于 **RTX 4090** 或配备 **32GB RAM 的 Macbooks** 等设备上的本地推理（Local Inference）。
  - 社区对 **Apache 2.0 许可证** 充满热情，因为这允许广泛的分发和修改。此外，用户还讨论了该模型与 **Qwen 2.5 32B** 和 **GPT-4o-mini** 等其他模型的性能对比，以及在不同硬件配置下的速度和效率，有用户报告在 **RTX 8000** 上速度为 **21.46 tokens/s**，在 **M1 Max 64GB** 上为 **24.4 tokens/s**。


- **[DeepSeek 创始人专访：我们不会走闭源路线。我们相信建立强大的技术生态系统更为重要。](https://thechinaacademy.org/interview-with-deepseek-founder-were-done-following-its-time-to-lead/)** ([分数：298，评论：41](https://reddit.com/r/LocalLLaMA/comments/1idtkll/interview_with_deepseek_founder_we_wont_go/)): **DeepSeek** 的创始人强调了他们坚持开源的承诺，认为建立强大的技术生态系统优先于闭源策略。采访表明，这种方法对于 AI 社区的创新与协作至关重要。
  - **OpenAI 与 DeepSeek**：讨论中流露出对 OpenAI 最初开源意图的怀疑，并将其与 DeepSeek 当前的开源策略进行了对比。用户担心一旦完成用户习惯培养，是否会像 OpenAI 那样转向闭源。
  - **对冲基金策略**：有人对 DeepSeek 的财务策略提出猜测，部分用户认为他们的运作方式类似于对冲基金，通过发布开源模型来影响市场估值，这种策略被描述为一种*基于信息的市场操纵*。
  - **技术好奇心**：社区对 DeepSeek 的技术表现出明显兴趣，特别是关于他们的 **FP8 训练代码**。用户表达了获取该代码以加速家庭端训练的愿望，强调了技术社区利用开源进展开展个人项目的兴趣。


- **[Mistral 新开源模型](https://i.redd.it/5nnsoy4295ge1.png)** ([分数：128，评论：7](https://reddit.com/r/LocalLLaMA/comments/1idokcx/mistral_new_open_models/)): **Mistral** 发布了两个新模型：**Mistral-Small-24B-Instruct** 和 **Mistral-Small-24B-Base-2501**，并配备了包含搜索栏和排序选项的用户界面更新。这些模型属于 23 个可用模型系列的一部分，其中 Instruct 模型获得了 50 个赞，Base 模型获得了 23 个赞。
  - **Mistral Small 3** 因其与 **Llama 3.3 70B** 和 **Qwen 32B** 等大型模型的竞争力而受到关注，在相同硬件上速度快 **3 倍以上**且完全开源。它被认为是 **GPT-4o-mini** 等专有模型的优秀开源替代方案。更多详情可以在[这里](https://mistral.ai/news/mistral-small-3/)找到。
  - 评论中有人对 **Base** 和 **Instruct** 模型之间的区别感到好奇，尽管具体细节尚未详述。


**主题 2. Nvidia 削减 RTX 40/50 GPU 的 FP8 训练性能**

- **Nvidia 在 RTX 40 和 50 系列 GPU 上将 FP8 训练性能减半** ([Score: 401, Comments: 93](https://reddit.com/r/LocalLLaMA/comments/1ideaxu/nvidia_cuts_fp8_training_performance_in_half_on/)): 根据 Nvidia 新发布的 Blackwell GPU 架构白皮书，**Nvidia** 据报道已将 **RTX 40** 和 **50 系列 GPU** 的 **FP8 训练性能**削减了一半。其中 **4090** 型号在 FP8（使用 FP32 累加）下的性能从 **660.6 TFlops** 降至 **330.3 TFlops**。这一变化可能会阻碍在 Geforce GPU 上进行 AI/ML 训练，反映了自 Turing 架构以来，在保持 Quadro 和数据中心 GPU 全速性能的同时，限制消费级显卡性能的一贯模式。
  - 许多评论者认为，**RTX 40 和 50 系列 GPU** 中报道的 **FP8 训练性能**减半可能是文档中的笔误，并参考了 **Ada Lovelace 论文**，其中曾将 FP8/FP16 累加与 FP8/FP32 混淆。一些人建议使用新旧驱动程序进行测试，以验证性能是否确实被更改。
  - 有指控称 **Nvidia** 从事反消费者行为，并提到可能通过**芯片蚀刻（chip etching）**和固件限制来约束性能。讨论还涉及了法律行动的可能性，将此情况与之前的案例进行对比，如苹果 iPhone 降速和解案以及 Nvidia GTX 970 虚假广告罚款案。
  - 用户强调了 **CUDA** 对机器学习任务的重要性，并指出了在 **Apple Silicon** 等非 Nvidia 硬件上遇到的困难。讨论还触及了 AI/ML GPU 市场的不健康状态，并对比了 **Quadro** 和数据中心 GPU 的全速性能，而这些性能在消费级 GPU 中并未得到体现。


**主题 3. DeepSeek R1 性能：在本地设备上表现出色**

- **DeepSeek R1 671B 在本地游戏配置上无需 GPU 即可达到超过 2 tok/sec！** ([Score: 165, Comments: 57](https://reddit.com/r/LocalLLaMA/comments/1idseqb/deepseek_r1_671b_over_2_toksec_without_gpu_on/)): 该帖子讨论了在不使用 GPU 的情况下，通过一台配备 **96GB RAM** 和 **Gen 5 x4 NVMe SSD**（用于内存缓存）的**游戏配置**，在 **DeepSeek R1 671B** 模型上实现了 **2.13 tokens/秒** 的速度。作者建议，投资多个 NVMe SSD 可能是运行大模型的一种高性价比替代方案，可以取代昂贵的 GPU，因为他们的配置显示 CPU 和 GPU 占用率极低，突显了家庭配置更好的性价比潜力。
  - 用户讨论了 **2.13 tokens/秒** 速率的实用性和局限性，一些人表示至少需要 **5 tokens/秒** 才能有效使用，另一些人则指出 **2k context** 对于编程等某些应用来说是不够的。
  - 人们对通过将 **NVMe SSD** 堆叠成 RAID 配置或使用加速卡来提高性能表现出兴趣，并建议花费约 **$1,000** 理论上可以达到 **60 GBPS**，从而提升运行大模型的速度和性能。
  - 对详细复现步骤和特定命令用法的请求表明社区有兴趣尝试类似的配置。一位用户分享了一个 [包含 llama.cpp 命令和日志的 gist](https://gist.github.com/ubergarm/0681a59c3304ae06ae930ca468d9fba6)，以帮助他人理解和复现该设置。

- **你*到底*在用 R1 做什么？** ([Score: 106, Comments: 134](https://reddit.com/r/LocalLLaMA/comments/1idgrh4/what_are_you_actually_using_r1_for/)): 作者质疑 **DeepSeek R1 模型** 的实际效用，指出它们专注于推理并生成冗长的思考过程，即使是针对简单问题也是如此。他们对匆忙将这些模型用于日常任务表示怀疑，认为它们可能更适合解决复杂问题，而不是像 **GPT-4o** 那样的常规交互。
  - 用户强调了 **DeepSeek R1** 在各种技术任务中的效用，例如编程、数学解题和数据分析。**Loud_Specialist_6574** 和 **TaroOk7112** 发现它在编程方面特别有用，**TaroOk7112** 指出它有能力在第一次尝试时就无误地将脚本转换为更新的版本。**No-Statement-0001** 描述了一个复杂问题，R1 提供了一个涉及处理 Docker 信号的 shell 脚本解决方案。
  - 几位用户提到了该模型在**创意和理论应用**中的有效性。**Automatic_Flounder89** 和 **Acrolith** 分别指出了它在理论实验和创意写作中的用途，而 **a_beautiful_rhind** 则赞赏它的角色扮演能力。**Dysfu** 将其用作数学助教，通过避免直接给出答案来增强学习体验。
  - **AaronFeng47** 和 **EmbarrassedBiscotti9** 讨论了 R1 面临的挑战，例如代码中的逻辑错误和偶尔忽略规范，但也承认其在复杂任务中的潜力。**AaronFeng47** 将其与其他模型的体验进行了对比，发现 R1 不如 **o1-preview** 可靠。


**主题 4. 马克·扎克伯格谈 Llama 4 进展与策略**

- **马克·扎克伯格谈 Llama 4 训练进展！** ([Score: 154, Comments: 85](https://reddit.com/r/LocalLLaMA/comments/1id6gcj/mark_zuckerberg_on_llama_4_training_progress/)): **马克·扎克伯格** 强调了 **Meta** 在 **Llama 4** 上的进展，突出了其凭借**多模态能力**在 AI 领域领先的潜力，以及 2025 年即将到来的惊喜。他还讨论了 **Ray-Ban Meta AI 眼镜** 的成功以及重大基础设施投资计划，预计 **Meta AI** 将成为超过 **10 亿人** 使用的领先个性化助手。
  - 人们对 **Llama 4** 的**模型大小**和配置表现出浓厚兴趣。用户表示需要能够适应各种硬件能力的任务模型，并建议提供 **1B**、**3B**、**7B** 直至 **630B** 的中间尺寸，以适应不同的 VRAM 容量，避免 **7B** 和 **800B** 模型之间的断档。
  - 围绕 **Meta 多模态能力** 的讨论突出了对原生全模态的兴奋，期待模型在文本、推理、视觉理解和音频方面表现出色。用户渴望支持**音频/文本**、**图像/文本**和**视频**能力的模型，这对于语音助手和视觉合成等应用至关重要。
  - 评论反映了对 **Meta** 时间表和战略决策的怀疑。担忧包括 **Llama 4** 的延迟发布、对训练后微调的关注以及模型尺寸范围有限的可能性。辩论还涉及 **Meta** 的 AI 发展在隐私以及与其他科技巨头竞争背景下的更广泛影响。


## 其他 AI 版块回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. DeepSeek-R1 的影响：技术与竞争分析**

- **无炒作 DeepSeek-R1 阅读清单** ([Score: 196, Comments: 9](https://reddit.com/r/MachineLearning/comments/1ideupn/no_hype_deepseekr1_reading_list/)): 作者分享了从他们的研究论文俱乐部汇编的**阅读清单**，重点关注通向 **DeepSeek-R1** 的 **AI/ML** 基础论文。旨在提供对该技术的深入理解，他们邀请读者在 [Oxen.ai 的博客](https://www.oxen.ai/blog/no-hype-deepseek-r1-reading-list) 上探索该清单。
  - 讨论了带有 Attention 的**低秩矩阵**方法，并提出了一个问题：是否可以使用现有权重将其回溯应用到现有模型中。
  - 表达了加入**研究论文俱乐部**的兴趣，并询问了更多关于如何参与的信息。
  - 分享了对**阅读清单**的正面反馈，并期待即将举行的 **Paper Club** 会议。

- **[d] 为什么 "Knowledge distillation" 现在突然被贴上了盗窃的标签？** ([Score: 256, Comments: 87](https://reddit.com/r/MachineLearning/comments/1idjtta/d_why_is_knowledge_distillation_now_suddenly/)): **Knowledge distillation** 正被争议性地贴上盗窃的标签，尽管它是一种通过模仿输出来近似变换的方法。该帖子认为这种标签是站不住脚的，因为架构和训练方法不同，且该过程并不一定会复制原始的变换函数。
  - 几位评论者强调了 **copyright law**（版权法）与 **Terms of Service (TOS) 违规**之间的区别，强调虽然使用 OpenAI 模型的输出可能违反 TOS，但在 copyright law 下并不等同于盗窃。**ResidentPositive4122** 指出，OpenAI 的文档阐明他们并不对 API 生成内容主张版权，只是规定使用此类数据训练其他模型违反了 TOS。
  - 围绕 **OpenAI 对潜在 TOS 违规的反应**的讨论表明，这是一种维持其地位的战略举措，**proto-n** 认为 OpenAI 对 DeepSeek 的指控是断言其在 AI 领域影响力和重要性的一种方式。**batteries_not_inc** 和其他人认为，OpenAI 的反应更多是出于不满而非法律依据。
  - 辩论还涉及 AI **监管与伦理**的更广泛主题，**H4RZ3RK4S3** 等人讨论了 **EU regulations** 的影响以及对**美国和中国技术实践**的不同看法。**KingsmanVince** 和 **defaultagi** 对美国和中国的做法都表示怀疑，表明了伦理考量和公众认知的复杂格局。


- **[OpenAI 与 Microsoft 的现状：昨天 vs 今天](https://i.redd.it/k79xgml9d6ge1.jpeg)** ([Score: 154, Comments: 27](https://reddit.com/r/OpenAI/comments/1idtwy7/state_of_openai_microsoft_yesterday_vs_today/)): **DeepSeek-R1** 现已集成到 **Microsoft Azure services** 中，这标志着与此前涉及指控从 **OpenAI API** 泄露数据的争议相比发生了转变。最近在 **Azure AI Foundry** 和 **GitHub** 上的发布突显了该平台的可靠性和能力，与 **Reuters** 此前报道的安全担忧形成鲜明对比。
  - **DeepSeek-R1** 现已在 **Azure** 上可用，用户表示有兴趣将其作为 API 选项进行测试。人们对 Microsoft 的动机表示怀疑，一些人认为他们正在利用之前的争议牟利。
  - 该模型是 **free and open source** 的，这是其获得广泛支持的关键原因，尽管一些用户并不理解模型与其应用之间的区别。
  - 讨论中提到了 Microsoft 历史上的 **"embrace, extend, and extinguish"**（拥抱、扩展、再消灭）策略，暗示了对其支持 **DeepSeek-R1** 背后真实意图的担忧。


**主题 2. Copilot 的 AI 模型集成与用户反馈**

- **[o1 现在在 Copilot 中免费提供](https://i.redd.it/r074j9gia1ge1.png)** ([Score: 253, Comments: 56](https://reddit.com/r/OpenAI/comments/1idamb3/o1_now_available_free_of_charge_in_copilot/)): **Copilot** 现在向所有用户免费提供 **OpenAI 的 reasoning model (o1)**，正如 Mustafa Suleyman 在 Twitter 上宣布的那样。该公告展示了一个关于洋流的对话，说明了 o1 提供详细回答的能力，并突出了用户参与度指标。
  - 大多数用户对 **Copilot** 表示不满，称其为 Microsoft 产品中“最差”的 AI，多条评论强调了与 **错误答案** 和集成不佳相关的问题。有一种观点认为 Copilot 的质量已经恶化，尤其是自 **去年 8 月** 左右做出更改以来。
  - 一些用户推测，Copilot 感知质量下降的原因是 **Microsoft 和 OpenAI** 的战略决策，旨在引导用户回到 OpenAI 订阅，或为未来的“虚拟员工”等产品收集数据。**Microsoft 持有 OpenAI 49% 的股份**被认为是这些战略中的一个重要因素。
  - 技术问题被归咎于**超长的 system prompts** 和出于“安全原因”的 **prompt injections**，这些因素干扰了模型性能。重点似乎在于企业用户，因为尽管产品质量下降，但**公司**在使用 Copilot 处理其数据时感到更放心。


**主题 3. ChatGPT 最新更新：用户体验与技术变革**

- **[ChatGPT 获得了一些不错的增量更新](https://i.redd.it/gzqfir8465ge1.png)** ([Score: 171, Comments: 61](https://reddit.com/r/OpenAI/comments/1ido8jq/chatgpt_got_some_nice_incremental_updates/)): 截至 **2025年1月29日**，ChatGPT 的 **GPT-4o 模型** 已获得增量更新，包括扩展了训练数据范围以提供更相关的知识、增强了图像分析能力，并提升了在 STEM 相关查询中的表现。此外，该模型现在对 emoji 的响应更加热情。
  - 用户对 **GPT-4o** 的 **增量更新** 持怀疑态度，认为 **OpenAI** 解除了之前的限制是为了推销更高价格的订阅层级，一些用户注意到响应质量正回归到最初的水平。讨论中还提到对 **o3-mini** 的期待，将其视为应对当前局限性的潜在短期方案。
  - 新更新中 **emojis** 的使用引发了分歧，一些用户欣赏增强的格式化效果，而另一些人则认为这过于冗余且具有干扰性，尤其是在专业语境下。一位用户将当前的 emoji 使用情况与 **Copilot** 的早期版本进行了对比。
  - 讨论了 **"Think" 按钮** 功能，部分用户已获得访问权限，并指出它有潜力为 **GPT-4o** 增加推理链。然而，人们担心这可能会如何影响消息限制，特别是对于配额有限的用户。


---

# AI Discord 回顾

> 由 Gemini 2.0 Flash Exp (gemini-2.0-flash-exp) 生成的摘要之摘要

**1. DeepSeek 的崛起：速度、泄露与 OpenAI 的竞争**

- **DeepSeek 超出预期，泄露数据**：DeepSeek 模型，尤其是 **R1**，展示了强大的推理和创意潜力，足以与 **OpenAI 的 o1** 匹敌，但 [Hacker News](https://news.ycombinator.com/item?id=42871371) 上的一次数据库暴露泄露了用户数据，引发了隐私担忧。尽管如此，[许多人认为它在创意任务](https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/)和[代码](https://www.codeium.com/changelog)方面的表现正超越 OpenAI。
- **R1 性能表现各异**：根据[此文档](https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1)，**DeepSeek R1 1.58B** 在基础硬件上运行缓慢（**3 tokens/s**），需要 **160GB VRAM** 或快速存储才能获得更好的吞吐量，但也有人报告在高端 GPU 上达到了 **32 TPS**。用户还反映量化版本在指令遵循方面可能会遇到困难。
- **OpenAI 与 DeepSeek 激烈交锋**：虽然有人注意到 **OpenAI** [指责 DeepSeek](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA) 使用其训练数据，但他们也[在内部使用 DeepSeek 进行数据检索](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA)。这种[竞争已经白热化](https://www.youtube.com/watch?v=3QuWqjJ1ZjM)，并引发了关于审查、开放访问和数据收集实践的质疑。

**2. 小模型掀起大波澜：Mistral 与 Tülu**

- **Mistral Small 3 表现亮眼**：新的 [**Mistral Small 3**](https://mistral.ai/news/mistral-small-3/)（**24B** 参数，**81% MMLU**）因其低延迟和本地部署能力而受到赞誉。根据[官方网站](https://mistral.ai/news/mistral-small-3/)，它的运行速度比竞争对手快 **3倍**，在性能和资源消耗之间找到了平衡点，并采用 Apache 2.0 许可。
- **Tülu 3 击败顶尖模型**：[**Tülu 3 405B**](https://allenai.org/blog/tulu-3-405B) 是一个拥有 **405B** 参数的开源权重模型，在基准测试中超越了 **DeepSeek v3** 和 **GPT-4o**。这得益于其 **可验证奖励强化学习 (RLVR)** 方法，并提供了[公开的后训练配方](https://allenai.org/blog/tulu-3-405B)。
- **量化权衡讨论**：开发者正在尝试模型量化，指出这虽然减小了模型大小和 VRAM 占用，但也可能**降低指令遵循能力**，促使用户评估其在各种任务中的有效性。

**3. RAG 与工具：LM Studio 与 Agent 工作流**

- **LM Studio 支持 RAG**：[**LM Studio 0.3.9**](https://lmstudio.ai/blog/lmstudio-v0.3.9) 现在支持通过本地文档附件实现 **RAG**，详见[文档](https://lmstudio.ai/docs/basics/rag)。这允许在聊天会话中使用上下文窗口内的文档。此外，它现在还支持 **Idle TTL** 和**自动更新 (auto-update)**，提升了运行效率。
- **Aider 通过只读存根 (Read-Only Stubs) 实现本地化**：出于隐私考虑，用户正在探索将 **Aider** 与 **Ollama** 等本地模型集成的方法。新的 [YouTube 视频](https://youtu.be/XE6v_RGe0-U)重点介绍了使用**只读存根 (read-only stubs)** 来管理大型代码库。
- **LlamaIndex 集成高级 Agent**：**LlamaIndex 的 “Mastering AI Agents Workshop”** 介绍了用于多 Agent 系统的高级 **AgentWorkflow** 概念，并展示了利用 LlamaIndex 构建的强大架构，详见[此处](https://t.co/UKIClalkKG)。

**4. 硬件与性能：GPU 与优化**

- **Blackwell 的性能提升**：根据 [NVIDIA 文档](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md)，采用 **sm_120a** 的新 **Blackwell** 架构将彻底改变 GPU 性能，为消费级 GPU 提供更强的计算能力。讨论指出，在新款 **RTX 5090** 上，**FP4** 任务的速度可能提升 **5 倍**，尽管部分测试显示增幅仅为 **2 倍**。
- **PyTorch 2.6 性能调节选项**：新发布的 [**PyTorch 2.6**](https://pytorch.org/blog/pytorch2-6/) 为 **Python 3.13** 添加了 `torch.compile` 支持，在 **X86** 上引入了 **FP16**，并使用了 **Manylinux 2.28**，但停止了对 **Conda** 分发的支持。
- **GPU 价格与可用性**：用户注意到新款 **5090** GPU 非常难买，迅速售罄；而 **Jetson Nano** 的价格已从约 **$250** 的标价飙升至 **$500-$700**。

**5. 融资、伦理与社区热点**

- **Dario Amodei 的 AI Safety 投资遭到批评**：社区成员对 **Dario Amodei** 投入 **$10亿** 推动 **AI Safety** 的大胆举动表示怀疑，有人将其言论贴上“欺诈性营销”的标签，并[对大规模 AI 筹款活动提出质疑](https://www.nature.com/articles/s41593-023-01514-1)。
- **软银对 OpenAI 的百亿美金豪赌**：据报道，**软银 (SoftBank)** 计划向 **OpenAI** 注入 **$150亿-$250亿** 的巨额投资，这是其对 AI 及其未来潜力的又一次重大押注，[进一步增加了其现有的投入](https://x.com/firstadopter/status/1884794211091759444)。
- **跨平台社区互动**：成员们积极分享发现并提出问题，对各种 AI 模型、框架和工具表现出强烈的参与度，包括在多个 Discord 频道中讨论不同方法如何影响该领域。


---

# PART 1: Discord 高层级摘要




## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **DeepSeek R1 的速度与障碍**：**DeepSeek R1 1.58B** 在受限硬件上的运行速度约为 **3 tokens/s**。[这份官方文档](https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1)建议使用 **160GB VRAM** 或高速存储以获得更高的吞吐量。
   - 社区成员指出了在 **Windows** 上可能存在的问题，并推荐使用 **Linux** 以获得更好的量化性能。
- **Mistral-Small 24B 强势登场**：新分享的 [Mistral-Small-24B-Instruct-2501-GGUF](https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF) 采用 **Apache 2.0 许可证**，虽然权重不公开，但承诺降低延迟。
   - 贡献者引用了 [Mistral 官网](https://mistral.ai/news/mistral-small-3/)的数据，称其 **MMLU** 达到 **81%**，认为它是开源选项中极具竞争力的补充。
- **Unsloth 开启在线 DPO 火花**：一位用户确认，在应用部分硬编码以处理内存限制后，使用 [Unsloth 仓库](https://github.com/unslothai/unsloth/issues/1494)成功运行了**在线 DPO (online DPO)**。
   - 他们分享了一篇关于降低 DPO 内存消耗的 [LinkedIn 帖子](https://www.linkedin.com/posts/keith-truongcao-7bb84a23b_reduce-online-dpo-memory-consumption-with-activity-7290108099607097344-jzaO)，并征求实际反馈。
- **MusicGen 微调尝试**：一位新手目标是使用 `.WAV` 和 `.TXT` 文件微调 **facebook/musicgen-medium** 或 **musicgen-small**，重点关注 [本指南](https://github.com/volcengine/verl) 中提到的 epoch 和 batch size。
   - 他们考虑利用 **vllm** 进行生成，同时也研究了 **Unsloth** 和 **GRPOTrainer**，以寻求稳定的微调路径。
- **vllm vs. Unsloth：殊途同归还是分道扬镳？**：社区成员将 **vllm** 的 Neural Magic 优势与 **Unsloth** 的量化方法进行了比较，不确定未来在 **Red Hat** 旗下两者是否会趋于一致。
   - 有人提议进行部分集成以减少 GPU 空闲时间，而另一些人则认为由于速度差异，这两种方法各具特色。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **O1 vs. R1 之争与 Perplexity 的切换困扰**：用户质疑 Perplexity Pro 中 **O1 vs. R1** 的可靠性，注意到尽管选择了 O1，系统仍会默认切换到 R1。许多人认为 **O1** 提供了更好的推理能力，但最近的可靠性问题引发了担忧。
   - 在 [Aravind Srinivas 的一条推文](https://x.com/aravsrinivas/status/1884801300027589007)中，承诺为 **Pro 用户**提供**每日 500 次 DeepSeek R1** 查询，但其一致性仍存疑，部分用户称其“稳定得令人恼火”。
- **阿里巴巴的竞争追赶者模型**：阿里巴巴推出了一个**新模型**以加强其竞争地位，可能会重新调整市场动态。更多详情见[此链接](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg)，重点介绍了用于提升用户体验速度的高级算法。
   - 社区成员期待进一步的增强，有人暗示可能与现有的开源框架产生**协同效应**，尽管尚未发布官方声明。
- **DeepSeek 获得关注并颠覆数据检索**：[OpenAI 澄清了其对 DeepSeek 的使用](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA)，赞扬了其处理复杂数据集的查询能力。许多人称赞 **DeepSeek** 稳定的隐私功能，尽管他们也注意到了偶尔的停机。
   - Deepinfra 的 [DeepSeek-R1 Demo](https://deepinfra.com/deepseek-ai/DeepSeek-R1) 被引用为可以完成与 **OpenAI-O1** 类似的任务，引发了关于 **Token** 使用和性能优势的激烈辩论。
- **Sonar-Reasoning 令人惊讶的不足**：**sonar-reasoning** 模型 API 的测试者质疑其在实际应用中的表现，寻求相比其他模型的改进细节。一些人报告称出现了**冗长且重复的回答**，浪费了 Token 并忽略了新的 Prompt。
   - 其他人认为它在某些任务中仍然表现出色，但在 **Playground** 中的直接对比表明，该模型的“思考”能力在 API 响应中可能有所减弱。
- **GPT-4、Sonnet 和 Gemini 的对决**：在一场持续的辩论中，用户讨论了 **GPT-4**、**Sonnet** 和 **Gemini 2.0** 在高级查询（包括微积分和编程任务）中的表现。**Sonnet** 因更自然的文本表达而获得赞誉，而 GPT-4 和 Gemini 在原始准确性方面仍是强力竞争者。
   - 一些人强调，将 **Sonnet** 与 O1 结合使用可以为**复杂任务提供更清晰的输出**，促使人们放弃部分 Claude 订阅并重新思考付费墙。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **DeepSeek 的动态双雄**：Windsurf 为 Pro 级账户引入了 [DeepSeek R1](https://x.com/windsurf_ai/status/1885077046663217230) 和 **DeepSeek V3**，每条消息需要不同的 Credits。
   - 开发者强调了 R1 首次作为 **Coding Agent** 的使用，并参考 [Changelog](https://www.codeium.com/changelog) 获取更多更新。
- **Cascade 的快速修复**：社区成员报告了**输入延迟的减少**，以及修复了 Cascade 面板在重新加载时自动重新打开的问题。
   - 他们还讨论了通过 `@web` 和 `@docs` 实现的新网页搜索功能，指向基于 URL 的上下文处理。
- **DeepSeek vs. Sonnet 对决**：用户比较了 **DeepSeek** 和 **Claude 3.5 Sonnet** 的成本效益和性能，许多测试者更青睐 R1。
   - 其他人描述 **Sonnet** 会不断地编辑文件，而 R1 在编程任务中表现出稳定的行为。
- **额度困惑得到澄清**：成员们争论 **DeepSeek R1** 每条消息是消耗 0.25 还是 0.5 个 Credits，理由是文档说明不一。
   - 他们指向 [Codeium Docs](https://docs.codeium.com/windsurf/usage) 和[支持页面](https://codeium.com/support)以获取准确详情。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **DeepSeek 敢于与 OpenAI 决斗**：在公会讨论中，参与者强调 **DeepSeek R1** 在创意任务上优于 **OpenAI** 的 o1，并引用了[一个 Raspberry Pi 演示](https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/)以及来自 **Gemini Pro** 和 **Grok** 的潜在竞争。
   - 有人在[一段 YouTube 评论视频](https://www.youtube.com/watch?v=3QuWqjJ1ZjM)中声称 *“DeepSeek 审查结果”*，引发了关于**数据收集**和开放访问的推测。
- **OneClickPrompts 助力快速设置**：介绍了一款名为 **OneClickPrompts** 的新工具，用于构建可个性化的多部分 Prompt，并分享了一个 GIF，展示了其在重复任务中的简化用法。
   - 用户称赞了该扩展的**模块化方法**，但指出 *“智能 Prompt 组合”* 对于获得更深层次的结果仍然至关重要。
- **Ollama 微调获得关注**：一位用户寻求为特定领域任务**微调 Ollama** 的方法，引发了对未来扩展或官方工作流的期待。
   - 其他人提到了 GitHub 上零散的参考资料，并补充说简化的流程可以解锁 **Ollama** 中 *“下一级别的适应性”*。
- **GPT 的记忆与 Context Windows 冲突**：成员们批评 **GPT** 的记忆在漫长的对话中会丢失关键细节，引发了对 **DeepSeek** 等开源项目更大 Context Window 的重新关注。
   - 他们认为**不一致的回忆**阻碍了生产环境的使用，并呼吁将 *“稳定的上下文保留”* 作为未来的必备功能。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **LM Studio 0.3.9 势头强劲**：**LM Studio 0.3.9** 增加了 **Idle TTL**、API 响应中独立的 **reasoning_content** 以及运行时的**自动更新**，官方安装程序见[此处](https://lmstudio.ai/download)。
   - 社区成员认可了改进的内存管理，并提到 **Hugging Face** 模型下载的自动更新过程更加简单，参考了[文档](https://lmstudio.ai/docs/api/ttl-and-auto-evict)。
- **RAG 推出在 LM Studio**：**LM Studio** 现在支持在聊天会话中附加本地文档进行 **RAG**，详见[文档](https://lmstudio.ai/docs/basics/rag)。
   - 用户观察到，如果文档符合模型的 Context，则可以完整包含，这激发了利用本地参考资料的兴趣。
- **DeepSeek 的 GPU 性能飙升**：讨论显示，在 **GTX 1080** 和 **Ryzen 5 3600** 上，**DeepSeek** 模型可达到 **6-7 tokens/sec**，重点在于 VRAM 管理以防止减速。
   - 其他人报告称，**i9-14900KF**、**128GB RAM** 和双 **RTX 4090** 的配置在 **70B** 模型上达到了 **30-40 tokens/sec**，强调了将整个模型放入 GPU 显存的重要性。
- **Jetson Nano 的价格令人咋舌**：成员们注意到 **Jetson Nano** 的价格达到了 **$500-$700** 或处于缺货状态，使其与标准 GPU 相比吸引力下降。
   - 少数人发现了 **$250** 左右的报价，但许多人更倾向于选择性能更优的常规硬件。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **DeepSeek R1 飙升与数据库泄露**：成员们报告 **DeepSeek R1** 在 4090 GPU 上达到了约 **32 TPS**，在赞扬其性能的同时也指出了量化版本的问题。[Hacker News 上的一个爆料](https://news.ycombinator.com/item?id=42871371)揭露了 **DeepSeek** 的数据库暴露，引发了用户隐私警报。
   - 一些参与者对依赖可能存在数据泄露的服务表示怀疑，称“隐私噩梦”是他们探索本地解决方案的原因。
- **O3 Mini 热度与量化怪癖**：许多人对 **O3 Mini** 表现出兴趣，认为它是一个更快、更小的替代方案，期待其体验优于现有的庞大模型。他们讨论了 **quantization**（量化）如何阻碍性能和指令遵循能力，有人称这是一种棘手的权衡。
   - 少数人开玩笑说正焦急等待 **O3 Mini** 来解决他们的模型困扰，而其他人则分享了之前量化版本的不同结果，强调了缩小模型规模的不可预测性。
- **Aider 转向本地化与只读存根 (Read-Only Stubs)**：出于隐私考虑，用户探索将 **Aider** 与 **Ollama** 等本地模型集成，期望一种避免将数据发送给第三方的解决方案。一个新的 [YouTube 视频](https://youtu.be/XE6v_RGe0-U)展示了旨在更高效处理大型代码库的 **read-only stubs**。
   - 一些人在使用多个端点（如 Azure AI）时遇到困惑，但发现 [advanced model settings](https://aider.chat/docs/config/adv-model-settings.html) 的参考资料很有帮助，另一些人则称赞 *stubs* 是加强代码修改控制的受欢迎举措。
- **O1 Pro 辩论引发定价讨论**：几位开发者支持将 **O1 Pro** 用于编程任务，但批评其成本和使用限制。他们将这些因素与本地开源模型进行权衡，指出审查担忧偶尔会阻碍生产力。
   - 少数参与者形容 **O1 Pro** 尽管价格昂贵，仍是强大的编程盟友，而另一些人则坚持使用本地模型，以摆脱潜在政策变化的影响。

---

## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **DeepSeek R1 席卷西方**：Windsurf 宣布 **DeepSeek R1** 和 V3 现已上线 [tool calling capabilities](https://x.com/windsurf_ai/status/1885077046663217230)（工具调用能力），首次实现了 R1 在编程 Agent 模式下运行。
   - 用户注意到它完全托管在西方服务器上，并参考了 [Cursor 社区论坛](https://forum.cursor.com/latest) 进行持续讨论。
- **Chat 与 Composer 中的 Token 纠葛**：一些用户对 **10k token context** 设置表示困惑，报告在 Chat 和 Composer 中难以追踪使用情况。
   - 他们质疑 Beta 设置是否真的提供了扩展上下文，或者消息是否在没有警告的情况下被截断。
- **MCP 设置势头强劲**：一种 Bash 脚本方法允许人们快速添加 **MCP server** 配置，如[此 GitHub 仓库](https://github.com/daniel-lxs/mcp-server-starter)所示。
   - 开发者分享了 [MCP Servers 网站](https://www.mcpservers.ai/)，鼓励尝试将不同的服务器与 Cursor 协同使用。
- **模型安全风暴预警**：针对 ML 模型中潜在的隐藏代码执行产生了担忧，引用了[一篇关于 Hugging Face 模型静默后门的帖子](https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/)。
   - 一些人建议使用 [protectai/modelscan](https://github.com/protectai/modelscan) 扫描本地环境，以发现任何可疑的负载。
- **本地 vs 托管大对决**：关于自托管与依赖 **DeepSeek R1** 等解决方案的辩论异常激烈，涉及隐私和成本的权衡。
   - 虽然本地爱好者希望有更好的离线模型，但其他人则指出托管服务器随着发展带来的性能优势。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Nous x Solana Sunset Soirée**: 即将举行的纽约 **Nous x Solana** 活动收到了大量参加请求，重点讨论 AI 模型的分布式训练。
   - 参与者期待**现场演示**和专门的 Q&A，并希望与新的 **Psyche** 方法产生协同效应。
- **Mistral & Tülu Tussle**: 社区成员对 [Hugging Face](https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501) 上的 **Mistral-Small-24B-Instruct-2501** 和[这条推文](https://x.com/allen_ai/status/1884966600039915809)中的 **Tülu 3 405B** 感到兴奋，两者都定位为高性能的小规模 LLM。
   - 几位成员提到了 [R1-Zero 的博客分析](https://arcprize.org/blog/r1-zero-r1-results-analysis)进行基准测试对比，引发了关于哪种模型真正卓越的辩论。
- **Psyche Paves Paths for Distributed Gains**: **Psyche** 分布式训练框架旨在通过模块化系统处理大规模 RL，因其扩展模型训练的能力而受到赞誉。
   - 一条[推文](https://x.com/Teknium1/status/1884740956911718853)展示了对该框架开源的兴奋之情，重点关注 GitHub 的可访问性和可能的共识算法路线图。
- **China's Ten Titans Tower Over Europe's Models**: 根据[这条推文](https://x.com/deedydas/status/1884786839913111931)，一次聊天透露**中国**拥有 10 个可与欧洲顶级模型（包括 **Mistral**）匹敌的一流 AI 模型。
   - 参与者指出，**美国**仅拥有五个主要的 AI 实验室——**OpenAI**、**Anthropic**、**Google**、**Meta** 和 **xAI**——凸显了激烈的全球竞争。
- **CLIP-Driven Generation Gains Ground**: 一位成员询问了关于 **CLIP** 嵌入的**自回归生成**，这通常用于引导 **Stable Diffusion**。
   - 他们强调直接由 CLIP 驱动的生成过程缺乏参考资料，表明了将多模态输入与解码任务合并的兴趣。

---

## [Yannick Kilcher](https://discord.com/channels/714501525455634453) Discord

- **Dario’s Daring $1B Venture**: 社区成员讨论了 **Dario Amodei** 及其投入 **10 亿美元**推动 **AI Safety** 的举动，对他博客文章中的财务透明度和雄心勃勃的声明提出了质疑。他们对一些人标记为“欺诈性营销”的行为表示不安，反映了对大规模 AI 筹款活动的更深层怀疑。
   - 几位技术专家认为，将如此巨额资金投入广泛的安全倡议可能会忽视其他紧迫的 AI 研究，而其他人则坚持认为这可以催化更负责任的 AI 发展。
- **Mistral’s Middling-Sized Marvel**: 新发布的 [**Mistral Small 3**](https://mistral.ai/news/mistral-small-3/) 拥有 **24B** 参数，在 MMLU 上达到 **81%**，运行速度比大型竞争对手快 **3 倍**。开发者赞扬了它的本地部署能力，称其在性能和资源效率之间找到了平衡点。
   - 爱好者将其与 **Llama 3.3 (70B)** 等模型进行了对比，认为 Mistral 精简的设计可能会推动更易于获取的专业化解决方案。
- **Tülu 3 405B Triumph**: AI2 的研究人员发布了 [**Tülu 3 405B**](https://allenai.org/blog/tulu-3-405B)，拥有惊人的 **405B** 参数，并在多个基准测试中击败了 **DeepSeek v3** 和 **GPT-4o**。其 **Reinforcement Learning from Verifiable Rewards (RLVR)** 方法提升了模型在测试环境中的准确性和一致性。
   - 参与者注意到了该模型的训练配方和权重开放政策，认为这可能为更大胆的开放研究合作提供动力。
- **Framework Face-Off: LlamaIndex vs PydanticAI vs LangChain**: 开发者报告称 **PydanticAI** 界面整洁且有内部温度设置，但遗憾其经常出现损坏的 JSON 输出。[**LlamaIndex**](https://x.com/llama_index) 产生了更一致的结构化数据，而 **LangChain** 因其基于管道的架构使错误追踪复杂化而受到批评。
   - 其他人强调某些 UI 中的高 CPU 或 GPU 占用是一个痛点，促使人们呼吁开发具有强大日志记录和性能指标的精简 Agent 工具。
- **Prospective Config’s Bold Brainchild**: 一篇 [**Nature Neuroscience** 论文](https://www.nature.com/articles/s41593-023-01514-1)引入了 **prospective configuration** 作为**超越反向传播 (backpropagation)** 的学习基础，引发了对下一代神经训练的新推测。该方法声称提高了效率，并更好地与生物过程保持一致。
   - 社区讨论表明其可能与 **RL** 方法产生协同效应，而一些人则质疑在技术飞跃如此之快的领域，该方法是否可能承诺过度。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Tülu 3 击败巨头**：**Tülu 3 405B** 的发布展示了相比 **DeepSeek v3** 和 **GPT-4o** 更优越的性能，详见[其博客](https://allenai.org/blog/tulu-3-405B)。
   - 爱好者们强调了其**开源训练后配方 (open post-training recipes)**，对其可扩展性和巨大的 405B 参数量感到兴奋。
- **Mistral Small 3 掌控极低延迟**：**Mistral Small 3** 作为一款 24B 参数模型首次亮相，具有低延迟特性，据称可以在典型硬件上流畅运行（[详情点击](https://mistral.ai/news/mistral-small-3)）。
   - 社区反馈赞扬了其**知识密集型 (knowledge-dense)** 架构，将其定位为本地生成式 AI 任务的强力竞争者。
- **DeepSeek 泄露引发安全担忧**：[Wiz Research 透露](https://x.com/wiz_io/status/1884707816935391703)一个可公开访问的 DeepSeek 数据库泄露了密钥和聊天记录。
   - 讨论集中在**隐私问题**上，引发了对 AI 基础设施采取更严格控制措施的呼吁。
- **软银向 OpenAI 注入数十亿美元**：有报道称 **SoftBank** 计划向 OpenAI 投资 **150-250 亿美元**，以补充其现有的超过 150 亿美元的承诺。
   - 分析师认为这是对 AI 的又一次巨大赌注，提高了本已激烈的融资竞赛的筹码。
- **DeepSeek v3 专家实现并行化**：DeepSeek v3 中新的 **Mixture-of-Experts (MoE)** 设计使用了 **sigmoid gating** 和 **dropless load balancing**，让多个专家能够在没有直接竞争的情况下做出响应（[论文](https://arxiv.org/abs/2401.06066)）。
   - 贡献者讨论了微调这些专家层并应用 **MTP** 来一次预测两个 token，引发了关于推理加速的猜测。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Deepseek 困境：OpenAI 的双标伦理**：社区成员注意到 **OpenAI** 在批评 **Deepseek 训练** 的同时，却讽刺地使用来自类似来源的数据，这引发了对其动机的质疑。他们怀疑 **OpenAI** 利用法律声明在拥挤的领域中塑造自信的形象。
   - 一些参与者认为 **Deepseek** 的争论凸显了潜在的虚伪，加深了对 **OpenAI** 是否真正保护合作者利益的怀疑。
- **RL 启示：更少的工具，更多的天赋**：LLM 爱好者发现，使用**强化学习 (RL)** 可以减少工具使用指令的大小，让模型在最少的指导下掌握核心技能。他们担心过度依赖特定工具可能会损害核心解决问题的能力。
   - 通过平衡 **RL** 与选择性的工具接触，他们希望保留模型的推理能力，而不至于陷入机械的工具依赖。
- **Hyperfitting 热潮：微小数据带来巨大收益**：新结果显示，在极小的数据集上进行 **hyperfitting** 可以大幅提升开放式文本生成，在人类偏好评分中从 **4.9%** 跃升至 **34.3%**。一篇[论文](https://openreview.net/forum?id=Ij9ilPh36h)证实了这些显著的改进，促使人们重新审视传统的过拟合 (overfitting) 担忧。
   - 批评者争论这种狭窄的训练是否会损害更广泛的泛化能力，但许多人对这些令人惊讶的文本质量提升表示欢迎。
- **批判热潮：微调优于盲目模仿**：研究人员提出了**批判微调 (Critique Fine-Tuning, CFT)**，教模型识别和纠正带噪声的回答，而不仅仅是模仿正确的解决方案。据[这篇论文](https://arxiv.org/abs/2501.17703)记录，他们在六个数学基准测试中报告了 **4–10%** 的性能提升。
   - 社区对教模型批判错误可能比标准监督微调 (SFT) 产生更稳健的推理表示乐观。
- **后门传闻与 Llama2 配置困惑**：[这篇论文](https://arxiv.org/abs/2409.03077)中出现了关于不可检测的**后门模型 (backdoored models)** 的新警告，对传统的基于损失 (loss-based) 的检测策略提出了质疑。同时，开发者在设置 gated MLP 维度时，对 [Llama2 配置](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26)中 **32768** 这个数字的意义提出了疑问。
   - 一些人指出这个数字不能被 **3** 整除，导致重置为 **11008**，并引发了关于如何干净地导出模型配置的进一步讨论。

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **DeepSeek 的双重蒸馏模型发布**：OpenRouter 推出了 [DeepSeek R1 Distill Qwen 32B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b) 和 [DeepSeek R1 Distill Qwen 14B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b)，两款模型均承诺以每百万 Token **$0.7–$0.75** 的价格提供接近大模型的性能。
   - 据报道，14B 版本在 AIME 2024 上获得了 **69.7** 分，两款模型均可通过 OpenRouter **Discord** 访问。
- **Subconscious AI 与 Beamlit 的重大进展**：Subconscious AI 在其[官网](https://www.subconscious.ai)展示了**因果推理（causal inference）**和**市场模拟**的潜力，强调“保证人类级别的可靠性”。
   - 与此同时，[Beamlit](https://beamlit.com) 推出了免费 Alpha 版，可将 AI Agent 的交付速度提升高达 **10 倍**，并提供 GitHub 工作流和可观测性工具。
- **OpenRouter 价格争议与速率限制吐槽**：用户对 OpenRouter 收取 **5%** 的费用展开讨论，部分原因归结于底层的 **Stripe** 成本。
   - 其他用户报告了使用 Google Gemini 时频繁出现 **429 RESOURCE_EXHAUSTED** 错误，建议使用个人 API Key 以避免超时。
- **Mistral Small 3 与 Tülu 3 预告**：通过 [推文](https://x.com/allen_ai/status/1884966600039915809) 宣布，**Mistral Small 3**（24B，81% MMLU）和 **Tülu 3**（405B）均承诺进行扩展训练并提供更快的吞吐量。
   - 社区讨论认为，这些新发布的模型与 DeepSeek 搭配使用可能会在速度和准确性上获得更大提升。

---

## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **Bolt 重大的二进制文件突破**：根据 [bolt.new 的推文](https://x.com/boltdotnew/status/1885019780840653183)，Bolt 停止生成二进制资产，从而显著减少了**数十万个** Token 的消耗，并提升了输出质量。
   - 社区成员对转向**外部资产**的做法表示赞赏，认为这加快了执行速度，并在社区讨论中称其为“重大的性能飞跃”。
- **社区系统提示词（System Prompt）惊喜**：开发者讨论转向了**项目和全局系统提示词**，一位用户将其用于变更日志（changelog）更新，并希望看到更多扩展的创意用途。
   - 有技巧提示分享特定文件并确认视图正确，展示了超越日常任务的更深层使用潜力。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **ComfyUI 获得更清晰的局部重绘（Inpainting）控制**：部分参与者分享了手动局部重绘的方法，引用了 [Streamable 上的示例](https://streamable.com/d3ww4l)，展示了在 **ComfyUI** 中通过高级 ControlNet 设置实现精准修饰。
   - 他们赞赏这种针对特定调整的灵活性，而非仅仅依赖自动化方法。
- **硬件热潮：GPU 讨论升温**：用户讨论了他们的 GPU 选择，**Intel Arc A770 LE** 被认为在游戏和 AI 任务中可与 **3060** 媲美。
   - 其他用户交流了 **3080** 和 **3090** 的使用心得，重点关注 Stable Diffusion 的 VRAM 需求。
- **换脸工具 Reactor 带着过滤器重新上线**：参与者注意到 **Reactor** 曾因缺乏 NSFW 检查被移除，随后在 [GitHub](https://github.com/Gourieff/sd-webui-reactor-sfw) 上传了更安全的版本。
   - 他们还指向了 [ComfyUI 扩展](https://github.com/Gourieff/ComfyUI-ReActor)，以实现更精简的换脸功能。
- **Stable Diffusion 的 Lora 训练技巧**：成员们剖析了构建 **Lora** 的步骤，强调了风格整合和精确的面部匹配。
   - 他们讨论了结合多个参考源的方法，突出了在同步风格和特征方面的挑战。
- **5090 GPU 瞬间售罄**：新款 **5090** GPU 被瞬间抢购一空，引发了对缺货和高昂需求的沮丧情绪。
   - 人们在考虑通过融资方式购买新硬件，并对极少的库存感到失望。

---

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **Blackwell 与 sm_120a 的重大突破**：全新的 **Blackwell** 架构及其 **sm_120a** 特性使之前的 **sm_90a** 功能相形见绌，详见 [cutlass/media/docs/blackwell_functionality.md](https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md)，这预示着消费级 GPU 将拥有更强大的计算能力。
   - 社区成员讨论了 **RTX 5090** 相比 **RTX 4090** 的提升，提到在 **FP4** 任务中可能有 **5倍** 的加速，但在其他测试中仅为 **2倍**，引发了对*文档不一致*的担忧。
- **PyTorch 2.6 重磅发布**：最新推出的 **PyTorch 2.6** 增加了对 **Python 3.13** 的 `torch.compile` 支持，引入了 **X86 上的 FP16**，并使用了 **Manylinux 2.28**，详见 [PyTorch 2.6 发布博客](https://pytorch.org/blog/pytorch2-6/)。
   - 爱好者们注意到了 **Conda** 的弃用，同时赞扬了如 `torch.compiler.set_stance` 等新的性能调节参数，有人称其为分发策略中的*“重大转变”*。
- **Reasoning Gym 快速扩张**：**Reasoning Gym** 已飙升至 **33** 个数据集，并包含在 [GALLERY.md](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) 的新展示页中，展示了广泛的强化学习任务。
   - 贡献者赞扬了协作挑战，并提出了*多 Agent 协商*设置，引发了关于**解释性**和**基于逻辑**任务的讨论。
- **Mistral 在 AIx Jam 的精彩表现**：**Mistral AIx** 的参赛作品在 🤗 Game Jam 中获得第 2 名，邀请大家在 [这个 HF Space](https://huggingface.co/spaces/Mistral-AI-Game-Jam/ParentalControl) 中测试 **ParentalControl**，将 **AI** 与游戏开发结合，打造喜剧恐怖体验。
   - 他们还展示了 **Llama3-8B R1**，声称在 GSM8K 上有 **14%** 的提升，详见[这篇博客文章](https://mobiusml.github.io/r1_redistill_blogpost/)，引发了对高性价比训练的热议。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **DeepSeek 模型引发 LaTeX 讨论**：成员们热切期待 **DeepSeek** 的发布，称赞其在复杂任务中强大的数学和 **LaTeX** 能力。
   - 他们讨论了 VRAM 限制，强调在进行繁重计算时需要谨慎管理上下文窗口大小。
- **Ollama 与 GPT4All 联动实现本地增益**：一些用户确认通过将 **Ollama** 作为服务器运行，并从 **GPT4All** 调用 **OpenAI** API，成功将 **GPT4All** 连接到 **Ollama**。
   - 他们推荐参考 [GPT4All 文档](https://docs.gpt4all.io/gpt4all_api_server/home.html) 获取分步操作方法。
- **远程 LLM 接入 GPT4All**：用户测试了将远程 LLM 加载到 **GPT4All** 中，强调了设置正确 API 密钥和环境变量的必要性。
   - 他们建议在 [GitHub wiki](https://github.com/nomic-ai/gpt4all/wiki/Local-API-Server) 中改进引导说明以帮助新手。
- **AI 教育计划进入离线模式**：一位用户展示了为非洲儿童构建 AI 驱动工具的计划，引用了 [Funda AI](https://emmanuelsibanda.hashnode.dev/funda-ai-building-a-laptop-powered-by-ai-to-help-students-in-africa-learn)。
   - 他们计划使用轻量级模型和精选数据，以便在没有网络的情况下进行自主学习，弥补资源差距。
- **模型后缀 -I1- 之谜**：一位成员询问某些模型名称中的 **-I1-** 含义，但目前尚无官方确认的解释。
   - 其他人要求更清晰的标签，表明对更公开的模型文档的需求。

---

## [MCP (Glama)](https://discord.com/channels/1312302100125843476) Discord

- **Cursor 受限的 MCP 能力**：新版 **Cursor** 增加了部分 **MCP** 支持，但**环境变量**仍是一个空白，导致需要像 [env invocation](https://www.gnu.org/software/coreutils/manual/html_node/env-invocation.html) 中提到的那样，使用 `FOO=bar npx some-server` 这种命令行变通方案。
   - 社区成员寻求 **MCP** 与 **LSP** 配置结构之间更好的对齐，认为这种不匹配是阻碍广泛采用的绊脚石。
- **MCP 的 Web 客户端黑科技**：一个自托管的 Web 客户端现在可以协调多个 **MCP** 服务器和 Agent，实现本地或云端设置的平滑切换。
   - 其灵活的方法激发了人们的兴趣，尽管有些人对 **MCP** 缺乏动态 Agent 提示词（prompt）功能感到遗憾。
- **8b 模型的函数调用挫败感**：**MCP** 中的 **8b** 模型在函数调用（function-calling）和工具使用方面表现挣扎，这让依赖强大 Agent 交互的测试者感到困惑。
   - 几位贡献者建议在 *Reddit* 等论坛上进行更深入的社区讨论，希望能解决该模型的可靠性问题。
- **Hataraku 在 ShowHN 登顶**：**Hataraku** 项目在 ShowHN 上飙升至榜首，为其 [TypeScript SDK 提案](https://github.com/turlockmike/hataraku/blob/main/docs/sdk-proposal.md)和 CLI 功能注入了动力。
   - 社区成员正通过协作和试运行积极参与，旨在优化界面并提升更广泛的用户体验。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **NotebookLM 二月反馈盛会**：NotebookLM 将于 **2025 年 2 月 6 日** 举办远程聊天会议以收集用户反馈，并向参与者提供 **$75** 奖励。
   - 参与者需要[提交筛选表单](https://forms.gle/HJmCwNepsfPSdC7g7)，具备稳定的网络连接以及带视频功能的设备。
- **用英雄联盟术语转录交易策略**：一位用户将交易课程视频转换为音频，然后使用 AI 进行转录，并利用 [NotebookLM](https://link.to.notebooklm) 来理清高级材料。
   - 他们使用 LoL（英雄联盟）的引用介绍了 **Big Players**（大玩家/主力），展示了 AI 在解释复杂概念时的灵活性。
- **24 小时内解读行政命令**：NotebookLM 在不到一天的时间内总结了一份关于公共教育隐私的新行政命令（Executive Order），并附带了深入的 [YouTube 评论](https://youtu.be/8RFYmgYn7P4)。
   - 这一演示引发了关于将该工具应用于政策简报和深入分析的讨论。
- **剖析 DeepSeek R1：GRPO 与 MoE**：一期 NotebookLM 播客介绍了 **DeepSeek R1**，重点讲解了 **GRPO** 和 **Mixture of Experts** (MoE) 以解释其架构。
   - 听众观看了包含基准测试和快速演示的[完整讨论](https://youtube.com/watch?v=zVDmKv3hWzk)，并针对性能提升提出了疑问。
- **学习指南生成缓慢与语言失效**：一些用户在生成学习指南时面临 10-30 分钟的延迟，即使只有一个源文件也是如此。
   - 其他人则抱怨多语言处理能力较差（例如**韩语**和**日语**）以及短暂的 **Gemini 2.0 Flash** 故障，同时寻求更严格的来源引用规则。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **分支提升与重定向汇总**：**分支变更**已完成，所有 Pull Request 均已重定向，确保了代码集成过程的顺畅。
   - 团队成员如有疑问可以随时提问，凸显了项目对公开沟通的重视。
- **为 Mojo LSP 推进 NeoVim 支持**：开发者讨论了如何通过 [nvim-lspconfig](https://github.com/neovim/nvim-lspconfig) 启用 **Mojo LSP**，但在设置过程中遇到了一些奇怪的问题。
   - 少数人报告仅获得部分成功，表明需要更深入的调试来实现稳定的工作流。
- **Mojo 1.0：速度与稳定性的对决**：Chris Lattner 强调 **Mojo 1.0** 应该融合 GPU 优化与直接执行，以实现速度最大化。
   - 参与者认为，在追求顶级性能指标的竞争中，必须平衡即时的可靠性。
- **向后兼容性大检查**：成员们担心新版本 Mojo 中的**破坏性变更**（breaking changes）可能会阻碍用户升级。
   - 他们强调了对旧版本库的支持，以保持势头并培养稳定的用户群。
- **Mojo 中的反射与性能探讨**：对话集中在用于数据序列化的**反射**（reflection）上，并指出部分反射功能已初步实现。
   - 与会者还推动进行大规模集群的**基准测试**（benchmarking），并提到了 Chris Lattner 的 [Mojo🔥: a deep dive on ownership](https://youtu.be/9ag0fPMmYPQ) 视频。

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **小巧但迅速：Mistral 3**：根据[官方详情](https://mistral.ai/news/mistral-small-3/)，**Mistral Small 3** 作为一款拥有 24B 参数的模型发布，采用 **Apache 2.0 license**，具备 **81% MMLU** 和 **150 tokens/sec** 的性能。
   - 它具有更少的层数和更大的词汇量，其在社交媒体上非传统的 **FF-dim/model-dim ratio** 引发了社区兴趣。
- **DeepSeek 数据库泄露暴露秘密**：据 [Wiz Research](https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak) 报道，**DeepSeek** 一个配置错误的 **ClickHouse database** 导致了重大数据泄露，包括聊天记录和密钥。
   - 他们在披露后迅速修复了泄露，这引发了人们对 **AI** 数据处理整体安全性的担忧。
- **Riffusion 推出 FUZZ**：Riffusion 推出了 **FUZZ**，这是一款旨在免费提供高质量输出的新型生成式音乐模型，[在此分享](https://x.com/riffusionai/status/1884984941081198954)。
   - 早期采用者称赞其旋律效果，并指出该服务仅在 GPU 资源充足时免费。
- **OpenAI API 延迟受到关注**：讨论中提到了 **OpenRouter** 和 [Artificial Analysis](https://artificialanalysis.ai/providers/openai)，作为追踪 OpenAI API 可能出现的 **latency**（延迟）激增的方法。
   - 一些人认为响应率正常，但社区成员建议保持谨慎并持续检查。
- **ElevenLabs 获得 1.8 亿美元融资**：**ElevenLabs** 在由 **a16z & ICONIQ** 领投的 C 轮融资中筹集了 **1.8 亿美元**，这一里程碑[在此宣布](https://x.com/matistanis/status/1885011065018163224)。
   - 观察人士认为，这是对 **AI voice** 技术未来及其巨大市场潜力的有力认可。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **赛道预告吸引 LLM Agents 观众**：参与者正在等待有关 **LLM Agents** MOOC 的 **application**（应用）和 **research**（研究）赛道的更多细节，组织者承诺很快会分享。
   - 社区成员重复着“请保持关注！”的消息，渴望听到官方公告。
- **报名故障导致确认停滞**：几个人注意到他们提交了 Google Forms 报名表但未收到回复，特别是那些寻求 **PhD** 机会的人。
   - 他们要求提供最终录取详情并加快回复速度，以便管理自己的日程。
- **Quiz 1 查询与私人存档**：成员们确认 **Quiz 1** 已在课程网站上线（参考教学大纲），一些人正在寻找之前 **LLM Agent** 课程的旧测验解答。
   - 他们分享了一个 [Quizzes Archive](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit)，并提醒注意隐藏的答案和过时的浏览器提示。
- **证书困惑仍在继续**：许多人正在等待早期课程的 **certificates**（证书），官方指南承诺很快发布。
   - 组织者表示，即将发布的公告将澄清本学期奖励的处理流程。
- **讲座发布与无障碍目标**：成员们敦促尽快上传 **第 1 讲**，认为只需“5 分钟”，但编辑团队提到了 **Berkeley** 的字幕要求。
   - 他们指出可以通过 [课程网站](https://llmagents-learning.org/sp25) 观看直播，而精修版本需等待无障碍措施完成后发布。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **Agent 实战：通过 LlamaIndex 掌握 AI**：**Mastering AI Agents Workshop** 介绍了用于多 **Agent** 框架的高级 **AgentWorkflow** 概念，如[此链接](https://t.co/UKIClalkKG)所示。
   - 与会者探索了使用 **LlamaIndex** 的稳健架构方法，引发了关于最佳实践的新讨论。
- **BlueSky 助力：LlamaIndex 扩展版图**：**LlamaIndex** 团队正式入驻 **BlueSky**，在[此链接](https://t.co/GK4L8Sb2N6)中展示了新的曝光度。
   - 贡献者期待与该平台开展更广泛的互动，激发更多围绕 AI 发展的活动。
- **o1 的奇特支持：部分流式传输与争论**：成员们注意到 **LlamaIndex** 通过 `pip install -U llama-index-llms-openai` 增加了 `o1` 兼容性，尽管某些功能仍未实现。
   - 他们引用了一个 [OpenAI 社区帖子](https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043)，该帖子确认 **OpenAI** 尚未完全启用流式传输（streaming），这引发了用户的不满。

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **GPU 探戈：P2P Patch 对阵 Proxmox**：在 #general 频道中，参与者讨论了在多块 **NVIDIA GPU** 上使用 **P2P patch** 的情况，并权衡了 **Proxmox** 与裸机 (baremetal) 设置以获得最佳 **IOMMU** 支持。
   - 一些用户倾向于使用裸机以绕过预期的 **hypervisor** 限制，而另一些用户则报告称，如果配置得当，**Proxmox** 也能胜任。
- **Tiny Boxes 联手实现 VRAM 梦想**：成员们探讨了可以互连多少台 **Tiny Boxes**，并想知道是否可以在它们之间共享 **VRAM** 以进行 HPC 级的推理。
   - 他们注意到缺乏直接的 VRAM 池化机制，建议使用高速 **NIC** 进行基于网络的扩展，以实现分布式性能。
- **Token 吞吐量：15/秒到 100 个请求**：估算显示每个模型的容量为 **15 tokens/sec**，如果每个模型以 **14 tokens/sec** 运行，则潜在扩展能力可达 **100 个请求**。
   - 这说明了在受控条件下分布请求如何维持接近峰值的速度，为 HPC 设计讨论提供了支持。
- **为本地 (On-Prem) LLM 选购服务器**：一位用户询问在企业环境中托管 LLM 的推荐**物理服务器**，凸显了对本地部署解决方案的广泛兴趣。
   - 社区成员讨论了成本效益、功耗以及处理大规模部署的 GPU 扩展空间。
- **Tinygrad 中的 Block/Fused 代码**：在 #learn-tinygrad 频道，有人请求 blocked/fused 程序的**示例代码**，以演示如何加载和写入 **tensor blocks**。
   - 其他人解释说，通过减少开销和合并步骤，按块 (blocks) 执行操作可以显著**提升性能**。



---



## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command R 提升上下文能力**：一位用户分享了将 **command-r7b** 与**蒸馏框架 (distillation frameworks)** 集成的困扰，提到使用 *ollama* 生成合成数据，并指出这些工具现有支持中的空白。他们强调 **Command R** 是一款具有 **128,000-token** 上下文长度和检索增强生成 (RAG) 能力的大语言模型，并引导他人查看 [Models Overview](https://docs.cohere.com/v1/docs/models)、[The Command R Model](https://docs.cohere.com/v1/docs/command-r) 以及 [Command R Changelog](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale)。
   - 贡献者们关注 **Command R** 即将发布的版本，强调了增强的决策和数据分析能力。他们还讨论了弥合框架集成差距的问题，希望在未来的迭代中实现更顺畅的合成数据工作流。
- **关于 AI “毯子”的辩论**：一些成员形容 **AI 模型**是冰冷的，开玩笑说一条“毯子”可以给它们带来温暖。他们认为这反映了一种将无感情的机器拟人化的俏皮尝试。
   - 另一些人坚持认为 **AI** 不需要温暖或感情，引发了一场关于什么定义了人工智能系统中真实共情的快速争论。这段打趣凸显了人们对 **AI 情感感知**持续的好奇心。



---



## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **Proxy Patch 与 DSPy 辩论**：一位用户询问如何向 `dspy.LM` 适配器添加 **proxy**，并引用了在 `gpt3.py` 中集成 `http_client` 的 [GitHub PR #1331](https://github.com/stanfordnlp/dspy/pull/1331)。如果没有代理支持，他们无法在其托管端点上使用 **dspy 2.6**。
   - 另一位用户指出了代理用法如何与 [dspy/clients/lm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53) 代码引用保持一致。他们还质疑在 `litellm` 中是否可以配置 **SSL context** 以实现稳定连接。
- **LiteLLM 与 DSPy：支持阵容**：一位新成员询问 DSPy 支持哪些 **LLM**，从而引出了对 [LiteLLM 文档](https://docs.litellm.ai/docs/providers) 的提及。该文档引用了 **OpenAI**、**Azure** 和 **VertexAI** 提供的服务。
   - 对话还涉及了在 `http_client` 中为高级配置指定 **SSL context** 的挑战。参与者注意到，这些参数设置在默认的 DSPy 文档中并未完全解释。



---

## [Axolotl AI](https://discord.com/channels/1104757954588196865) Discord

- **KTO vs Axolotl：紧急对决**：成员们指出了在为 **KTO** 任务集成 **Axolotl** 时面临的挑战，并表示迫切需要确认可行性和解决方案路径。
   - 他们表示已准备好协助审查代码并完成任务，强调希望保持项目进度。
- **Mistral 发布 24B 模型**：拥有 **24B 参数** 的新 [Mistral-Small-24B-Base-2501 模型](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) 引发了追求高性能小型 **LLM** 成员的兴奋。
   - 此次发布凸显了 **Mistral AI** 的开源承诺，并暗示将推出更多商业变体以满足特定需求。
- **Mistral 性能之谜**：一位成员承认目前缺乏对 **新 Mistral 模型** 的实际操作经验，导致性能声明尚未得到证实。
   - 对话建议未来进行用户测试，以收集真实世界的运行结果，并深入了解该模型在实践中的表现。
- **冬季学期超负荷**：繁忙的 **冬季学期** 时间表被描述为非常紧凑，影响了一位成员的贡献能力。
   - 这可能会推迟协作任务，促使其他人协调时间表并分担责任。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Farm Friend 之谜**：一位用户表达了对去年 **Farm Friend** 的喜爱，并注意到它目前在讨论中缺席。
   - 社区成员对其命运保持好奇，因为该讨论串中没有透露进一步的更新。
- **陈词滥调的评论引发趣闻**：对 **cliché reviews** 的轻松提及引发了俏皮的玩笑，随附的一张图片突显了这个笑话。
   - 虽然没有提供更深层的背景，但这次交流为社区增添了趣味时刻。
- **解读 '01'**：一位用户解释说 '01' 与 **OpenAI** 无关，澄清了之前对话中的困惑。
   - 这一言论平息了猜测，并确认了该沟通误会纯属巧合。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **通过 DCP 开关提升 Checkpoints 性能**：成员们澄清说 **DCP checkpointing** 默认是关闭的，但可以通过在配置中设置 `enable_async_checkpointing=True` 来激活，从而实现异步写入。
   - 他们指出，该功能目前仅限于 **full_finetune_distributed**，这可能会限制在其他配置下的使用。
- **推动更广泛的 Checkpoint 支持范围**：一些人想知道为什么 **async checkpointing** 不能支持所有配置，暗示需要未来的更新。
   - 目前尚未提供明确的时间表，成员们希望看到更广泛的集成，以简化大规模微调过程。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **本地 Img2Vid 热潮**：一位用户询问了最佳的本地 **img2vid** 工具，引发了围绕性能需求和 **GPU** 利用率的讨论。
   - 其他人分享了他们的经验，强调了 AI 工程工作流中快速安装和 *清晰文档* 的重要性。
- **ltxv 受到青睐**：另一位成员推崇 **ltxv** 作为本地 **img2vid** 任务的首选，理由是其使用简单。
   - 他们暗示了未来的测试和改进，希望看到更多社区驱动的基准测试和扩展的模型支持。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **Simba 引发 Databricks 特征工程热潮**：Simba Khadder 将于 **太平洋时间 1 月 30 日上午 8 点** 举办一场关于在 **Databricks** 上构建 **feature pipelines** 的 **MLOps Workshop**，并提供了直接报名链接 [此处](https://buff.ly/40Ej4Z6)。
   - 与会者可以从 **Unity Catalog** 集成和直接的 **Q&A** 中汲取最佳实践，该活动对 **Data Engineers**、**Data Scientists** 和 **Machine Learning Engineers** **免费** 开放。
- **Databricks 拥抱地理空间分析**：**美国东部时间 2025 年 1 月 30 日下午 1:00**，**Databricks** 将举办一场关于高级 **geospatial analytics** 的免费会议，可在 [Eventbrite](https://www.eventbrite.com/e/doi-geospatial-analytics-with-databricks-tickets-1111902653769) 报名。
   - 与会者将看到 **spatial data** 如何在 **Databricks** 上进行处理，为那些寻求更深层数据工程见解的人延续了早先研讨会的势头。

---

## [Gorilla LLM (Berkeley Function Calling)](https://discord.com/channels/1111172801899012102) Discord

- **BFCL 数据向 HF 数据集靠拢**：一位参与者询问了使 **BFCL data** 符合 **Hugging Face** 数据集指南所需的步骤，寻求确保合规性的蓝图。
   - 未提供示例或文档，使得关于如何调整 *metadata schema* 或格式的讨论处于开放状态。
- **未出现其他主题**：对话仅限于关于实现 **Hugging Face** 合规性的单一查询。
   - 没有进一步的细节出现，其他人对潜在的解决方案保持沉默。

---

**Mozilla AI Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会沉寂时间过长，请告知我们，我们将将其移除。

---

# 第二部分：分频道详细摘要与链接


{% if medium == 'web' %}

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1334252515176222791)** (1053 条消息🔥🔥🔥): 

> `DeepSeek R1 performance, Mistral Small 24B, Fine-tuning strategies, Quantization, Unsloth capabilities` 


- **DeepSeek R1 1.58B 运行缓慢但连贯**：用户报告称，虽然 DeepSeek R1 1.58B 模型运行正常，但速度非常慢，由于硬件限制，通常只能达到每秒约 3 个 tokens。
   - 为了获得最佳性能，建议使用更大的 VRAM 和更快的存储解决方案。
- **Mistral Small 24B 发布**：Mistral Small 24B 已上传至 Hugging Face，在保持延迟优化的同时，提供了与更大模型竞争的性能。
   - 它在 Apache 2.0 许可证下开源，尽管其权重是闭源的，但这引起了开发者的兴趣。
- **多任务 Fine-tuning 而不遗忘**：专家建议不要进行顺序 Fine-tuning（A -> B -> C），因为这通常会导致对先前任务的灾难性遗忘。
   - 相反，建议将所有目标任务合并到单个 Fine-tuning 阶段，以保持已学习的特性。
- **Quantization 与模型大小**：对话讨论了动态 Quantization 的使用及其在保持大型模型性能的同时减少内存占用的潜力。
   - Quantization 技术可以帮助大型模型在较小的硬件上运行，尽管它们可能需要仔细的实现。
- **受限硬件的最佳模型**：对于配备 16GB RAM 和 8GB VRAM 的 RTX 4060 PC，用户询问适合其配置的最佳模型。
   - 他们被引导去尝试运行 DeepSeek 或 Mistral 模型的精简版本，具体取决于与硬件限制的兼容性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseeks-ai-breakthrough-bypasses-industry-standard-cuda-uses-assembly-like-ptx-programming-instead">DeepSeek 的 AI 突破在某些功能上绕过了行业标准的 CUDA，转而使用 Nvidia 类似汇编的 PTX 编程</a>: 极致优化并非易事。</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-unsloth-bnb-4bit">unsloth/Llama-3.2-3B-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF">unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-GGUF">unsloth/Mistral-Small-24B-Instruct-2501-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit">unsloth/Mistral-Small-24B-Instruct-2501-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit">unsloth/Llama-3.2-3B-Instruct-unsloth-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/OpenWebUI/status/1884719609552752801">来自 Open WebUI (@OpenWebUI) 的推文</a>: 🚀 感谢 @UnslothAI，你现在可以通过 llama.cpp 在 Open WebUI 上运行 1.58-bit DeepSeek-R1（非蒸馏版）了！💻⚡️（已在 M4 Max, 128GB RAM 上测试） 📝 在他们的博客文章中深入了解细节：htt...</li><li><a href="https://github.com/Jiayi-Pan/TinyZero">GitHub - Jiayi-Pan/TinyZero: DeepSeek R1-Zero 的简洁、易用的复现</a>: DeepSeek R1-Zero 的简洁、易用的复现 - Jiayi-Pan/TinyZero</li><li><a href="https://huggingface.co/docs/trl/main/en/grpo_trainer">GRPO Trainer</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>: 以下是我们所有 Notebook 的列表：</li><li><a href="https://docs.unsloth.ai/get-started/all-our-models>">Unsloth 文档</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/issues/1494">在 Unsloth 和 openInstruct 中为成功运行 Online DPO 所做的更改 · Issue #1494 · unslothai/unsloth</a>: 正如在 unsloth reddit 帖子中所承诺的 https://www.reddit.com/r/LocalLLaMA/comments/1hqkeyn/comment/m4rbtto/?utm_source=share&amp;utm_medium=web3x&amp;utm_name=web3xcss&amp;utm_term=1&amp...</li><li><a href="https://github.com/EvolvingLMMs-Lab/open-r1-multimodal">GitHub - EvolvingLMMs-Lab/open-r1-multimodal: 为 open-r1 添加多模态模型训练的分支</a>: 为 open-r1 添加多模态模型训练的分支 - EvolvingLMMs-Lab/open-r1-multimodal</li><li><a href="https://forum.devtalk.com/t/deepseek-671b-running-on-a-cluster-of-8-mac-mini-pros-with-64gb-ram-each/185709">在由 8 台各配备 64GB RAM 的 Mac Mini Pro 组成的集群上运行 DeepSeek (671B)</a>: 这太酷了！DEEPSEEK-V3 在 M4 MAC 上：在 Apple Silicon 上实现极速推理。我们刚刚见证了令人难以置信的事情：最大的开源语言模型在 Apple Silicon 上大显身手。我们...
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1334272372538081452)** (16 messages🔥): 

> `文本转图像服务器、模型训练问题、前端缺陷、敏感话题调整` 


- **文本转图像服务器推荐**：一位用户建议使用专门的文本转图像服务器，如 **Stable Diffusion** 或 **Midjourney**，并表示有成千上万个此类任务可用的服务器。
   - 回复显示，大家在图像生成应使用合适平台这一点上达成了共识。
- **模型训练的数据集担忧**：一位成员指出 **R1** 并非在 **cot traces** 上训练的，强调了当前模型数据集的局限性。
   - 这提高了人们对正确训练数据对于有效模型性能重要性的认识。
- **影响输出的前端缺陷**：一位用户观察到前端可能存在缺陷，质疑为什么在切出标签页后会发生变化。
   - 另一位成员确认，模型检测系统很可能会根据前端行为更改输出。
- **围绕某些话题的敏感性**：讨论了敏感话题，特别是与在不同语境下发生的变化相关的话题。
   - 成员们评论说，话题的正确性可能取决于敏感问题的细微差别，特别是涉及 **China** 的问题。


  

---


### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1334259246765965443)** (202 messages🔥🔥):

> `DeepSeek R1 模型、微调挑战、量化技术、模型的系统要求、推理与性能` 


- **DeepSeek R1 动态模型配置**：用户讨论了在专用服务器上运行 DeepSeek R1 Dynamic 1.58-bit 模型，并建议使用 160GB VRAM 等合适硬件以获得最佳性能。
   - 有人担心在 Windows 上使用该模型，并建议切换到 Linux，因为其性能和兼容性更好。
- **微调 Mistral 的挑战**：一位用户报告在微调 Mistral 7B 模型后出现了奇怪的重复输出，引发了关于可能存在过拟合或数据集质量问题的疑问。
   - 另一位用户建议检查微调时使用的 Chat Template，认为这可能是潜在原因。
- **关于模型量化与性能的疑问**：讨论内容包括询问 R1 32B 模型是否可以在 8GB RTX 4060 上运行，并确认在进行适当量化后是可以实现的。
   - 用户对 DeepSeek R1 8B 和 GPT-4 等模型之间的性能对比表示好奇。
- **用户经验与故障排除**：参与者分享了模型安装的个人经验，强调了有效运行 DeepSeek 所需的各种配置。
   - 建议包括使用专用服务器而非个人硬件，并避免在 Windows 上运行大型模型。
- **Unsloth 的动态量化技术**：Unsloth 使用的动态量化方法被认为是减小模型体积且不牺牲性能的重要因素，关于其有效性的讨论仍在继续。
   - 参与者寻求关于有多少模型支持该技术的澄清，并分享了相关资源以便进一步学习。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://ollama.com/Huzderu/deepseek-r1-671b-1.73bit">Huzderu/deepseek-r1-671b-1.73bit</a>：合并了 GGUF Unsloth 的 DeepSeek-R1 671B 1.73bit 动态量化版本</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_Nemo_(12B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://www.datacamp.com/blog/llm-evaluation">LLM 评估：指标、方法论、最佳实践</a>：学习如何使用关键指标、方法论和最佳实践来评估大语言模型 (LLMs)，从而做出明智的决策。</li><li><a href="https://ollama.com/download">在 macOS 上下载 Ollama</a>：下载适用于 macOS 的 Ollama</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb#scrollTo=LjY75GoYUCB8">Google Colab</a>：未找到描述</li><li><a href="https://arxiv.org/html/2404.14047v1">低比特量化的 LLaMA3 模型表现如何？一项实证研究</a>：未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/wsl/wsl-config#main-wsl-settings">WSL 中的高级设置配置</a>：关于在 Windows Subsystem for Linux 上运行多个 Linux 发行版时，用于配置设置的 wsl.conf 和 .wslconfig 文件的指南。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Conversational.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Mistral_v0.3_(7B)-Alpaca.ipynb">Google Colab</a>：未找到描述</li><li><a href="https://unsloth.ai/blog/dynamic-4bit">Unsloth - Dynamic 4-bit 量化</a>：Unsloth 的 Dynamic 4-bit 量化选择性地避免对某些参数进行量化。这在保持与 BnB 4bit 相似的 VRAM 使用量的同时，大大提高了准确性。</li><li><a href="https://docs.unsloth.ai/">欢迎 | Unsloth 文档</a>：初识 Unsloth？</li><li><a href="https://huggingface.co/collections/unsloth/unsloth-4-bit-dynamic-quants-67503bb873f89e15276c44e7">Unsloth 4-bit Dynamic Quants - unsloth 集合</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF/tree/main">unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF at main</a>：未找到描述</li><li><a href="https://huggingface.co/learn/nlp-course/chapter7/6?fw=pt">从头开始训练因果语言模型 - Hugging Face NLP 课程</a>：未找到描述</li><li><a href="https://docs.unsloth.ai/basics/inference">推理 | Unsloth 文档</a>：学习如何运行你的微调模型。</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic#running%20r1">运行 DeepSeek-R1 Dynamic 1.58-bit</a>：DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_Coder_(14B)-Conversational.ipynb#scrollTo=ekOmTR1hSNcr).">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing">首页</a>：以 2-5 倍的速度和减少 70% 的显存微调 Llama 3.3, Mistral, Phi-4, Qwen 2.5 和 Gemma LLMs - unslothai/unsloth</li><li><a href="https://docs.unsloth.ai/basics/datasets-101">数据集入门 | Unsloth 文档</a>：学习创建微调数据集的所有要点！</li><li><a href="https://docs.unsloth.ai/get-started/installing-+-updating">安装 + 更新 | Unsloth 文档</a>：学习在本地或在线安装 Unsloth。
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1334285194143858850)** (2 条消息): 

> `Online DPO, AI 内存消耗, Unsloth 项目` 


- **成功实现 Online DPO**：一位用户宣布他们成功在 **Unsloth** 上实现了 **online DPO**，并承认其仓库中存在硬编码。
   - 他们请求社区就其实现相关的任何潜在问题提供反馈。
- **关于减少内存消耗的 LinkedIn 帖子**：该用户分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/keith-truongcao-7bb84a23b_reduce-online-dpo-memory-consumption-with-activity-7290108099607097344-jzaO?utm_source=share&utm_medium=member_android)，讨论了**减少 online DPO 内存消耗**的策略。
   - 这篇文章可能为那些在 AI 领域研究类似问题的人提供宝贵的见解。


  

---

### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1334527778837758026)** (13 条消息🔥): 

> `微调 MusicGen，基于 RL 的训练框架，Unsloth 与 vllm 的对比，vllm 中的 Neural Magic，vllm 与 Unsloth 之间的协作` 


- **寻求微调 MusicGen 的支持**：一位成员正寻求使用包含 `.WAV` 和 `.TXT` 文件的自定义数据集来微调 **facebook/musicgen-medium** 或 **facebook/musicgen-small** 模型，并希望在创建专注于 epoch 和 batch size 等参数的训练工具包方面获得帮助。
   - 他们强调了自己是新手，并对训练过程中的任何帮助表示感谢。
- **关于基于 RL 的训练框架的讨论**：成员们讨论了如 [verl](https://github.com/volcengine/verl) 和 Hugging Face 的 **GRPOTrainer** 等基于 RL 的框架，并指出他们倾向于利用 **vllm** 进行生成，利用 Hugging Face Transformers 进行训练。
   - 有人好奇与在两项任务中都使用 **Unsloth** 相比，这种方法是否是最佳的长期策略。
- **关于 Unsloth-patched 模型速度的担忧**：一位成员询问 **Unsloth-patched 模型** 在生成速度上比 **vllm** 慢多少，并考虑硬件利用效率是否能平衡速度差异。
   - 讨论强调，即使 Unsloth 的生成性能没有超过 vllm，由于减少了 GPU 空闲时间，接近的性能表现仍然是有益的。
- **vllm 背后的 Neural Magic**：**vllm** 因其性能而备受关注，这主要由 Neural Magic 驱动，特别是在其被 **Red Hat** 收购之后。
   - 社区对未来的合作以及这两个模型是否能有效地协同工作表示不确定。
- **vllm 与 Unsloth 之间潜在的协作**：成员们思考让 **vllm** 和 **Unsloth** 共同运行是否可行，或者其中一个是否必然会阻碍另一个。
   - 有人提出了关于结合这两个框架（而不是二选一）的潜在益处和协同效应的问题。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1334252988273000459)** (993 条消息🔥🔥🔥): 

> `Perplexity Pro 功能、模型性能对比、O1 和 R1 的问题、DeepSeek 功能、AI 用于学术支持` 


- **关于 O1 和 R1 功能的困惑**：用户报告称，尽管在 Perplexity Pro 中选择了 O1，系统仍默认使用 R1，这让追求性能一致性的用户感到沮丧。
   - 值得注意的是，用户认为 O1 相比 R1 提供了更好的推理能力，但最近表现得不够稳定。
- **关于学习用 AI 模型的讨论**：在关于 AI 模型的对话中，用户辩论了 GPT-4、Sonnet 和 Gemini 2.0 在学习方面的效果，特别是在微积分和编程领域。
   - 许多用户表示更倾向于使用 Sonnet，因为它生成的文本更自然，并且在处理复杂任务时与 O1 配合能提供更高的清晰度。
- **DeepSeek 的访问与可靠性**：用户讨论了 DeepSeek 与 Perplexity 的可靠性，强调 Perplexity 更稳定且隐私功能更好，而 DeepSeek 经常出现宕机。
   - 一位用户表示，他们成功为 `.gov` 邮箱设置了 Pro 服务账号，展示了其在组织内部的潜在用途。
- **对 AI 平台的偏好**：大家达成共识，虽然 Perplexity 提供了实用的功能，但用户也受到同时使用多个 AI 平台（包括 ChatGPT）灵活性的吸引。
   - 一些用户取消了 Gemini 和 Claude 的订阅，转而选择 Perplexity 和 ChatGPT 提供的优势。
- **关于 AI 使用的问题**：用户就哪些 AI 模型最适合特定任务（如图像生成和文档处理）提出了问题，意见各不相同。
   - 社区分享了使用各种模型的经验，并根据个人用例和性能预期给出了建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://deepinfra.com/deepseek-ai/DeepSeek-R1">deepseek-ai/DeepSeek-R1 - Demo - DeepInfra</a>：我们推出了 DeepSeek-R1，它在 RL 之前加入了冷启动数据。DeepSeek-R1 在数学、代码和推理任务上实现了与 OpenAI-o1 相当的性能。在 Web 上试用 API。</li><li><a href="https://tenor.com/view/whoa-shaking-head-windy-wind-blown-away-gif-15465608">Whoa Shaking Head GIF - Whoa Shaking Head Windy - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/austinpowers-drevil-mikemyers-evillaugh-ohhh-gif-1374158791848398246">Austinpowers Drevil GIF - Austinpowers Drevil MikeMyers - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/aravsrinivas/status/1884801300027589007?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：所有 Perplexity Pro 用户现在每天可获得 500 次 DeepSeek R1 查询（无审查且提示词不会发送到中国）。免费用户每天可获得 5 次查询。引用 Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：所有 Perplexity Pro 用户现在每天可获得 500 次 DeepSeek R1 查询（无审查且提示词不会发送到中国）。免费用户每天可获得 5 次查询。引用 Aravind Srinivas (@AravSrinivas) 100 daily De...</li><li><a href="https://inference.cerebras.ai/">Cerebras Inference</a>：未找到描述</li><li><a href="https://tenor.com/view/synths-and-sounds-ecstacy-ecstatic-climax-peak-gif-24235783">Synths And Sounds Ecstacy GIF - Synths And Sounds Ecstacy Ecstatic - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/aravsrinivas/status/1884913220734812449?s=61">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>：Perplexity Android App 的 Pro 搜索应支持 DeepSeek R1（开启 Pro 开关即可看到该选项）。尝试前请先从 Play Store 更新应用。</li><li><a href="https://www.cplx.app/">Complexity</a>：每个人都梦寐以求的 Perplexity.ai 增强版。</li><li><a href="https://www.reddit.com/r/perplexity_ai/comments/1idrac5/this_logic_is_unbelievable/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://api-docs.deepseek.com/quick_start/token_usage">Token 与 Token 使用 | DeepSeek API 文档</a>：Token 是模型用来表示自然语言文本的基本单位，也是我们计费的单位。它们可以被直观地理解为“字符”或“单词”。...
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1334275173553143940)** (12 条消息🔥): 

> `DeepSeek 与 OpenAI，Alibaba 新模型，末日时钟更新，Nike 蛇纹红鞋，近地小行星发现` 


- **OpenAI 声称 DeepSeek 被用于数据检索**：OpenAI 澄清其工具 **DeepSeek** 被用于搜索数据，强调了其在查询中的效能。更多信息可以在[详细文章](https://www.perplexity.ai/page/openai-claims-deepseek-used-it-3WNYRWivRdm90JDznlWCPA)中找到。
   - 该网站指出，在分析复杂数据集时，**DeepSeek** 显著增强了搜索能力。
- **Alibaba 在市场竞争中推出新模型**：Alibaba 最近发布的**新模型**旨在增强其在技术领域的竞争优势，预示着市场动态的潜在转变。完整见解请访问[此链接](https://www.perplexity.ai/search/alibaba-tvrdi-da-novi-model-na-5wnBBcUuTOmmpYaT6mfkLg)。
   - 该模型结合了专为效率设计的先进算法，可能重塑用户体验。
- **Nike 推出引人注目的蛇纹红鞋**：新发布的 **Nike Snakeskin Red Shoes** 因其出众的设计和限量发售而引起轰动，吸引了球鞋爱好者的关注。关于这些鞋子的细节可以在[这里](https://www.perplexity.ai/search/nike-snakeskin-red-shoes-cnf5iibDQcq18a2Jtv8WIQ#2)探索。
   - 许多粉丝渴望在售罄前抢购一双，显示出此次发布的火热程度。
- **近地小行星的发现引发关注**：最近发现的一颗**近地小行星**在科学家和太空爱好者中引发了兴奋，关于其特征的细节正在浮出水面。通过[此链接](https://www.perplexity.ai/page/near-earth-asteroid-discovered-yhHY75OOT4ujnsalxMwelw)深入了解研究结果。
   - 研究此类小行星对于理解生命起源和行星形成具有深远意义。



**提到的链接**：<a href="https://www.youtube.com/embed/mgMh2Kp7Uwo">YouTube</a>：未找到描述

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1334253531305218048)** (4 条消息): 

> `Sonar-Reasoning 模型性能，响应质量问题，重复回答担忧` 


- **Sonar-Reasoning 模型需要评估**：成员们正在测试新的 **sonar-reasoning model API**，质疑其与其他模型相比的性能以及改进之处。
   - *它真的更好吗？* 成员们正在寻求关于特定功能领域改进的见解。
- **响应中的思考能力下降**：成员们观察到，该模型似乎不像在 Playground 中那样有效地“思考”，导致了挫败感。
   - 一位用户指出，即使给出了指令，模型仍会返回冗长的推理，不必要地消耗了 Token。
- **相似问题的重复回答**：一位用户报告了一个 Bug，即当被问及相似问题时，模型会重复提供相同的答案，而忽略新的查询。
   - 这引发了对模型区分相似 Prompt 能力的担忧，导致了令人沮丧的用户体验。


  

---

### **Codeium (Windsurf) ▷ #[announcements](https://discord.com/channels/1027685395649015980/1027688115592237117/1334636642513715272)** (1 条消息): 

> `Cascade 模型更新，DeepSeek-R1 和 DeepSeek-V3，输入延迟降低，Web Search 功能，Changelog 洞察` 


- **Cascade 迎来 DeepSeek-R1 和 V3**：Windsurf 现在为 Pro 和 Pro Ultimate 用户支持 **DeepSeek-R1** 和 **DeepSeek-V3**，每条消息和 tool call 都有特定的额度消耗。
   - 这是一个显著的升级，因为 R1 首次可以在 coding agent 中使用。
- **Cascade 的显著修复和改进**：最近的更新包括进一步降低长 Cascade 对话的 **输入延迟**，并修复了防止 Cascade 面板在重新加载时重新打开的问题。
   - 此外，**@docs** 部分增加了更多选项，并改进了 Tab to Jump 快速设置的配置。
- **Cascade 引入 Web search 功能**：Cascade 现在可以自动或通过 **@web** 和 **@docs** 等特定命令进行 **web searches**，增强了其能力。
   - 用户可以输入 URL 作为上下文，使其在访问博客文章、文档和公开 GitHub 文件时非常有效。
- **分享 Changelog 洞察**：发布了详细的 [changelog](https://www.codeium.com/changelog)，提供了对 Windsurf 最新版本中所有更改的深入了解。
   - 鼓励用户查看完整更新以全面了解各项增强功能。
- **加入 Cascade 对话**：邀请社区成员加入专门的 Discord 频道，参与有关 Cascade 相关更新的持续讨论。
   - 鼓励大家积极参与，提供反馈并分享新功能的使用体验。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/windsurf_ai/status/1885077046663217230">来自 Windsurf (@windsurf_ai) 的推文</a>：DeepSeek R1 和 V3 现在已在 Windsurf 中可用，完全托管在西方服务器上。我们在 R1 中实现了 tool calling，使其首次能在 coding agent 中使用。</li><li><a href="https://www.codeium.com/changelog">Windsurf Editor Changelogs | Windsurf Editor 和 Codeium 扩展</a>：Windsurf Editor 的最新更新和更改。
</li>
</ul>

</div>
  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1334285824371593227)** (65 条消息🔥🔥): 

> `Codeium 问题, DeepSeek vs Sonnet, Windsurf 功能请求, Cascade 性能, Android 虚拟设备` 


- **Windsurf 用户遇到问题**：多位用户报告了 Windsurf 的问题，提到 **Claude 3.5 Sonnet** 会不断地循环编辑文件并提示错误。
   - 一位用户被建议下载诊断日志，并通过[此链接](https://codeium.com/support)向支持团队报告问题。
- **DeepSeek 通常比 Sonnet 更受青睐**：讨论中对比了 **DeepSeek** 和 **Sonnet**，用户提到 DeepSeek 更便宜，且被部分人认为更好。
   - 一位用户指出，在测试 **R1** 后，其性能似乎与 **Sonnet** 旗鼓相当，引发了辩论。
- **Windsurf 和 Cascade 的功能请求**：一位用户询问了类似于 **VSCode** 和 **Cursor** 的自动提交信息功能，并引用了已提交的功能请求。
   - 另一位用户指出，可以在 **Cascade** 界面内简化提交过程，以提高工作流效率。
- **Cascade 的性能改进**：用户讨论了 Cascade 最近的增强功能，这些功能提高了深度对话期间的 Prompt 速度，并解决了之前的响应缓慢问题。
   - 一位用户确认更新已解决之前的挑战，鼓励其他人查看 [changelog](https://codeium.com/changelog)。
- **在 Windsurf 项目中使用 Android 虚拟设备**：一位用户询问如何在其 Windsurf 项目中使用 **Android Virtual Device**，另一位用户推荐了 `toroxx.vscode-avdmanager` 扩展。
   - 这一建议反映了社区在寻求集成 Android 开发工具解决方案方面的积极参与。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/usage">付费计划与额度使用 - Codeium 文档</a>: 未找到描述</li><li><a href="https://codeium.com/changel">页面未找到 | Windsurf 编辑器与 Codeium 扩展</a>: Codeium 是受开发者喜爱且受企业信赖的 AI 代码助手平台。也是首个 Agentic IDE —— Windsurf 的构建者。</li><li><a href="https://codeium.com/support">支持 | Windsurf 编辑器与 Codeium 扩展</a>: 需要帮助？联系我们的支持团队获取个性化协助。</li><li><a href="https://codeium.canny.io/feature-requests/p/auto-commit-message">自动提交信息 | 功能请求 | Codeium</a>: 从已提交的文件上下文中生成提交信息</li><li><a href="https://codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器与 Codeium 扩展</a>: Windsurf 编辑器的最新更新和变化。
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1334251746096185384)** (707 条消息🔥🔥🔥): 

> `DeepSeek R1 实现, Windsurf 性能与问题, AI 模型对比, 定价与额度, Windsurf 与 DeepSeek 用户体验` 


- **DeepSeek R1 势头强劲**：用户庆祝 DeepSeek R1 的上线，注意到其价格低廉，仅为 **每条消息 0.25 额度**，与 Claude 3.5 相比可以使用更多次。
   - 许多用户对其能力表示兴奋，而另一些用户则遇到了无效工具调用和内部错误等问题。
- **Windsurf 性能担忧**：一些用户报告 DeepSeek R1 有时无法正确应用更改，在尝试修复代码时，对模型的响应感到沮丧。
   - 关于进一步优化的必要性以及模型如何处理某些命令执行的讨论正在进行中。
- **AI 模型对比：R1 vs. Claude 3.5**：用户参与了 DeepSeek R1 和 Claude 3.5 的对比，指出 R1 通常为类似任务提供更低成本的选择。
   - 重点讨论了如何将 R1 用于项目规划，而建议将 Sonnet 用于代码执行。
- **定价与额度系统说明**：用户讨论了与 Windsurf 相关的定价模型以及各种模型的额度消耗，澄清了 **DeepSeek R1 的定价为每条消息 0.5 额度**。
   - 额度系统引起了一些困惑，但总体上明确了针对成本效益使用不同模型的好处。
- **Windsurf 与 DeepSeek 的用户体验**：多位用户分享了他们使用 Windsurf 和 DeepSeek 的经验，强调了在构建项目和有效处理 Prompt 方面的成功与挑战。
   - 尽管存在一些问题，许多用户对 DeepSeek 可能为他们的工作流带来的潜在改进和价值持乐观态度。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.codeium.com/windsurf/memories#cascade-auto-generated-memories">Cascade Memories</a>: 未找到描述</li><li><a href="https://gitbutler.com/">GitButler | Git Branching, Refined</a>: 一个在现有工作流之上支持多分支并行的 Git 客户端。</li><li><a href="https://docs.codeium.com/windsurf/memories#workspace-rules">Cascade Memories</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/memories#cascade-auto-generated-">Cascade Memories</a>: 未找到描述</li><li><a href="https://docs.codeium.com/best-practices/prompt-engineering#prompt-engineering">Prompt Engineering - Codeium Docs</a>: 未找到描述</li><li><a href="https://docs.codeium.com/windsurf/memories#global-rules">Cascade Memories</a>: 未找到描述</li><li><a href="https://codeium.com/security">Security and Privacy | Windsurf Editor and Codeium extensions</a>: Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首款 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://gist.github.com/ykka/31f1059764a4be7d4f2f0e2e700da3f5">Windsurf VSCodeVIM Keyboard Shortcuts</a>: Windsurf VSCodeVIM 键盘快捷键。GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://docs.codeium.com/windsurf/usage">Paid Plan and Credit Usage - Codeium Docs</a>: 未找到描述</li><li><a href="https://status.codeium.com/">Codeium Status</a>: 未找到描述</li><li><a href="https://x.com/EHuanglu/status/1885024173862613485">Tweet from el.cine (@EHuanglu)</a>: 中国 AI 再次获胜，阿里巴巴刚刚秘密发布了 Qwen 2.5 Max AI，它将 AI 视频生成提升到了一个新高度，而且……它是免费的。10 个示例：1. 一名女子在拥挤的街道上与一名男子争吵</li><li><a href="https://x.com/OpenRouterAI/status/1884672717460271176?s=19">Tweet from OpenRouter (@OpenRouterAI)</a>: 我们的数据显示，OpenAI o1 的思考 tokens 数量是 DeepSeek R1 的 3 倍 👀 o1-mini 的思考量多出 50%。请参阅下方时间线中的“avg”列：</li><li><a href="https://x.com/EHuanglu/status/1865475635474493665">Tweet from el.cine (@EHuanglu)</a>: 今天，AI 永远改变了。Hunyuan AI 已经打破了恐怖谷效应。它如此真实，令我震惊。你能看出它是 AI 吗？因为我看不出来！14 个示例：</li><li><a href="https://codeium.com/support">Support | Windsurf Editor and Codeium extensions</a>: 需要帮助？联系我们的支持团队以获取个性化协助。</li><li><a href="https://www.reddit.com/r/Codeium/comments/1idq65e/logging_in_to_windsurf_in_an_enterprise_network/">Reddit - Dive into anything</a>: 未找到描述</li><li><a href="https://codeium.com/live/next-js">Chat with NextJS | Windsurf Editor and Codeium extensions</a>: 使用 Codeium Live 与 NextJS 对话。Codeium 是开发者喜爱、企业信赖的 AI 代码助手平台。同时也是首款 Agentic IDE —— Windsurf 的开发者。</li><li><a href="https://www.youtube.com/watch?v=cBzc5r-FNW0">Use Obsidian (BEST Markdown editor) for note taking and tech docs!</a>: 在这段视频中，我将向你展示我最喜欢的 Markdown 工具 Obsidian（一个免费的第二大脑和知识库程序）。我将展示如何编写我的技术文档……</li><li><a href="https://stratechery.com/2025/deepseek-faq/">DeepSeek FAQ</a>: DeepSeek 彻底颠覆了人们对 AI 以及与中国竞争的预期。它是什么，为什么它很重要？
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1334251723430301778)** (474 条消息🔥🔥🔥): 

> `DeepSeek vs. OpenAI Models, AI Detectors and Education Solutions, Creative AI Model Performance, Open Source AI Developments, AI Context Windows and Usability` 


- **DeepSeek 在创意任务上优于 OpenAI**：用户注意到 DeepSeek R1 在创意任务上的表现优于 OpenAI 的 o1，后者最近表现不佳。
   - 这种性能转变凸显了 Gemini Pro 和 Grok 等 AI 模型之间日益激烈的竞争。
- **学术界对 AI 检测器可靠性的担忧**：讨论围绕 AI 检测器的不准确性展开，这导致了对学生的不公正后果，包括学术环境中的误解。
   - 有建议提议使用 Google Docs 等工具来追踪草稿，这可能是一个更可靠的解决方案。
- **关于取消家庭作业以提高学习效果的思考**：有人提议用更积极的课堂学习和测验取代传统的家庭作业，以避免利用 AI 作弊。
   - 这种方法建议转变教育策略，以增强参与度并减少对 AI 完成作业的依赖。
- **开源 AI 及其创新潜力**：大家达成共识，认为像 DeepSeek 这样的开源模型通过允许更广泛的访问和协作，可以推动 AI 的重大进步。
   - 参与者认为，与大型科技公司使用的封闭系统相比，这将鼓励更多创新。
- **AI 模型之间的 Context Window 差异**：用户辩论了各种 AI 模型中 Context Window 的大小和可用性，特别强调了 DeepSeek 在有效处理长文本输入方面的优势。
   - 对话强调了不同模型处理上下文的方式和用户体验，许多人根据自己的需求表达了偏好。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chat.qwenlm.ai/">Qwen Chat</a>: 未找到描述</li><li><a href="https://www.globalnerdy.com/2025/01/29/running-deepseek-r1-raspberry-pi/">Running DeepSeek R1 on a Raspberry Pi : Global Nerdy</a>: 我印象深刻——事实证明你可以在 Raspberry Pi 上运行 DeepSeek R1 的本地副本！上面的照片显示了当前热门的 LLM 正在我的 Raspberry Pi 500 上运行，这简直...</li><li><a href="https://www.youtube.com/watch?v=3QuWqjJ1ZjM">DeepSeek is Copying Existing Al Data, Censoring Results, and Collecting Your Data for China</a>: 当你用现有的（充满错误的）AI 训练你的 AI，代表政府进行审查，并挖掘用户数据时，你会得到什么？你会得到 Dee...</li><li><a href="https://www.nvidia.com/en-us/project-digits/">NVIDIA Project DIGITS: The World’s Smallest AI Supercomputer. </a>: 立即预订。</li><li><a href="https://bsky.app/profile/jeffgeerling.com/post/3lgt3a76mws2v">Jeff Geerling (@jeffgeerling.com)</a>: 顺便提一下……左边有一个 16GB 的 Pi 5。
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1334266029370769409)** (68 条消息🔥🔥): 

> `Next Generation AI Instructions, Memory Function in GPT Models, API and Custom GPT Limitations, OpenAI's Model Release Intentions, Fine-tuning Ollama Models` 


- **下一代 AI 有望遵循指令**：一位用户建议，与其进行文本操作，AI 模型应该执行一个执行该操作的工具，以确保不对响应进行任何修改。
   - 通过创建一个使用 Markdown 格式化响应的链接处理器函数，可以通过 Custom GPT 功能获得更一致的结果。
- **关于 Memory 功能的讨论**：用户分享了关于 GPT 模型中 Memory 功能的经验，指出虽然 Memory 旨在增强上下文感知，但往往无法做到，特别是在冗长的讨论中。
   - 有人担心，随着讨论的延长，关键细节可能会滚出上下文，从而影响模型召回重要信息的能力。
- **API 和 Custom GPT 范围挑战**：一位用户报告了在一家铁路公司的项目中使用 API 时遇到的挑战，理由是 Custom GPT 在抓取链接时的行为不一致。
   - 尽管尝试了多种 API 解决方案，但链接传输的可靠性仍然是一个重大问题。
- **关于 OpenAI 模型发布意图的辩论**：关于 OpenAI 是否恶意发布模型出现了各种观点，特别是考虑到减缓竞争对手模型进度的努力。
   - 针对 GitHub 上发布的模型的有效性和完整性提出了疑问，以及它们是否被故意破坏以误导开发者。
- **对微调 Ollama 模型的兴趣**：一位用户询问如何微调 Ollama，可能是寻求改进模型输出的指导。
   - 这与针对特定应用定制 AI 模型的广泛兴趣相一致。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1334314125794344991)** (25 条消息🔥): 

> `AI Problem-Solving Limitations, Issues with Visual Recognition, Prompt Construction Tools, Understanding of Math Puzzles` 


- **AI 难以解决问题**：一位成员提到他们是高智商人士，但仍然无法解决某些问题，引发了关于认知测试及其性质的讨论。
   - 另一位成员强调了他们在创造性解决问题方面的能力，突出了某些 AI 挑战的复杂性。
- **对 AI 行为的挫败感**：有人担心 AI 不遵守预期的输出特征，特别是在保持适当的字数和输出长度方面。
   - 一位成员指出，低质量的输出会影响未来的响应，这表明交互设计中存在缺陷。
- **关于数学谜题的讨论**：一位用户质疑了一个涉及猫头鹰和棕榈树的具体问题的性质，思考其认知测试方面。
   - 有人分享了一个讨论该数学问题的聊天链接，澄清了它涉及求解方程组。
- **Prompt 构建的新工具**：一位成员介绍了 'OneClickPrompts'，这是一个旨在帮助从多个部分构建 Prompt 的扩展，以便快速访问常用 Prompt。
   - 附带的 GIF 提供了该工具运行方式的视觉展示，增强了对易用性的了解。
- **Vision 模型改进**：讨论指出了 AI 之前在视觉识别能力方面的局限性，当时它难以区分某些图形元素。
   - 成员们分享了用特定问题训练模型的经验，表明持续的反馈可能已经带来了改进。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1334314125794344991)** (25 条消息🔥): 

> `AI 问题解决的挑战、AI 回复长度与质量、视觉模型局限性、OneClickPrompts 扩展、社交媒体上的代数讨论` 


- **高智商但在解决问题时遇到困难**：一位成员质疑，尽管自己是高智商个体，但在解决某些问题时仍面临限制，并强调了自己在创造性问题解决方面的能力。
   - 这引发了围绕猫头鹰/棕榈树问题等特定任务以及所面临的认知挑战性质的讨论。
- **对 AI 输出长度一致性的需求**：成员们对 AI 在输出长度和质量上的不一致表示沮丧，特别是当它不符合预期标准时。
   - 有人指出，AI 倾向于认为次优输出是可以接受的，这会对未来的生成产生负面影响。
- **视觉模型的辨识挑战**：关于 AI 视觉模型的担忧被提出，该模型难以区分图像中的元素，例如地面与人物上方的线条。
   - 历史讨论表明，特定问题在几个月前就已被诊断出来，这表明 AI 学习随时间推移有潜在的增强空间。
- **OneClickPrompts 辅助高效 Prompt 创建**：介绍了一个名为 OneClickPrompts 的新扩展，旨在帮助用户通过多个部分构建 Prompt 以实现快速访问。
   - 分享了一个演示该功能的 GIF，提供了对其能力的直观理解。
- **代数谜题日益流行**：一位成员注意到 TikTok 等平台上代数讨论盛行，强调了社交媒体在促进数学交流方面的力量。
   - 其他人评论了 AI 在解决相对简单的代数问题时的感知有效性，尽管可能存在不准确之处。


  

---


### **LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1334641669437657189)** (1 条消息): 

> `LM Studio 0.3.9 特性、Idle TTL 功能、API 响应中的推理内容、LM 运行时自动更新、Hugging Face 嵌套文件夹支持` 


- **LM Studio 0.3.9 的激动人心特性**：LM Studio 0.3.9 引入了多项新功能，包括用于高效管理 API 模型内存的 **Idle TTL**，以及支持从 Hugging Face 仓库的**嵌套文件夹**下载模型。
   - 更新可通过应用内更新或从[此处](https://lmstudio.ai/download)获取，并包含多项旨在提升用户体验的错误修复。
- **引入用于智能内存管理的 Idle TTL**：通过 **Idle TTL**，用户可以为 API 模型设置生存时间并自动驱逐旧模型，从而增强 LM Studio 中内存资源的利用。
   - 该功能的详细文档可在 [docs](https://lmstudio.ai/docs/api/ttl-and-auto-evict) 中找到，供用户进一步优化使用。
- **独立的推理内容功能上线**：API 响应中新增的 **reasoning_content** 字段允许用户单独访问推理细节，类似于 DeepSeek 的 API，可通过“设置”开启。
   - 这一实验性功能增强了在聊天补全过程中收集的信息，紧密贴合开发者的需求。
- **运行时的新自动更新功能**：LM Studio 现在支持运行时的**自动更新**以简化更新流程，减少了用户需要手动更新多个组件的麻烦。
   - 此功能默认启用，但可以在“应用设置”中进行调整。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/docs/api/ttl-and-auto-evict">Idle TTL and Auto-Evict | LM Studio Docs</a>: 可选在一定时间（TTL）后自动卸载空闲模型</li><li><a href="https://lmstudio.ai/blog/lmstudio-v0.3.9">LM Studio 0.3.9</a>: Idle TTL、运行时自动更新、支持 HF 仓库嵌套文件夹，以及聊天补全响应中独立的 `reasoning_content`
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1334256243472203778)** (308 条消息🔥🔥): 

> `DeepSeek 模型、LM Studio 特性、LM Studio 中的 RAG、模型性能与推理、API 与 UI 讨论`

- **关于 DeepSeek 模型兼容性的讨论**：用户报告了加载 DeepSeek 模型时的问题，特别是与 pre-tokenizer 类型相关的错误消息。建议包括将 LM Studio 更新到最新版本，并通过 CTRL + SHIFT + R 验证运行时更新。
   - 有用户提到了指示模型词表问题的错误消息，促使其他人鼓励通过版本更新来解决。
- **LM Studio 中的 RAG 能力**：用户询问了 LM Studio 通过附加文档提供上下文来支持检索增强生成 (RAG) 的能力。文档显示 LM Studio 确实支持 RAG，允许用户将文档文件附加到聊天会话中。
   - 关于 RAG 的澄清指出，如果文档内容符合模型的上下文窗口，则可以完整添加以增强对话。
- **模型性能与推理能力**：围绕各种模型推理能力的讨论强调，特定模型支持高级推理，而其他模型则不支持。影响性能的因素包括模型大小以及它们是否能完全装入 GPU 显存。
   - 用户请求推荐擅长推理的模型，其中某些模型被认为在处理任务（特别是编程任务）时能提供更好的逻辑能力。
- **定制化与 UI 调整**：用户对 LM Studio 自定义主题和 CSS 以增强 UI 灵活性的潜力表示感兴趣。LM Studio 团队承认这是他们计划在未来实现的功能。
   - 关于应用程序结构的额外讨论也随之出现，一些人注意到客户端并非开源，但 CLI 工具是可用的。
- **对 LM Studio 进展的总体赞赏**：用户对 LM Studio 的进步表示总体满意，并注意到功能和用户体验方面的改进。对话突显了社区对改进本地 LLM 应用和集成高级功能的浓厚兴趣。
   - 在技术讨论中，大家对利用 Qwen-2.5 等强大模型来突破本地 LLM 的能力边界充满了共同的热情。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mistral.ai/news/mistral-small-3">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://huggingface.co/openbmb/MiniCPM-o-2_6-gguf">openbmb/MiniCPM-o-2_6-gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://www.anandtech.com/show/21111/amd-unveils-ryzen-7040u-series-with-zen-4c-smaller-cores-bigger-efficiency">AMD 发布采用 Zen 4c 的 Ryzen Mobile 7040U 系列：核心更小，效率更高</a>: 未找到描述</li><li><a href="https://x.com/MistralAI/status/1884967826215059681">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://tenor.com/view/eye-of-sauron-lotr-lord-of-the-rings-gif-16715227">索伦之眼 指环王 GIF - 索伦之眼 指环王 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/better-call-saul-call-saul-its-showtime-folks-gif-8557719">风骚律师 Showtime 时刻 GIF - 风骚律师 Showtime 时刻 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/kryonax-skull-gif-26476587">Kryonax 骷髅 GIF - Kryonax 骷髅 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://lmstudio.ai/docs/basics/rag">与文档对话 | LM Studio 文档</a>: 如何将本地文档作为额外上下文提供给 LLM</li><li><a href="https://tenor.com/view/fallout-tv-fallout-codsworth-fallout-prime-fallout-amazon-gif-14576962590525720544">辐射电视剧 Codsworth GIF - 辐射电视剧 辐射 Codsworth - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.anandtech.com/show/21111/amd-unveils-ryzen-7040u-series-with-zen-4c-smaller-cores-bigger">AMD 发布采用 Zen 4c 的 Ryzen Mobile 7040U 系列：核心更小，效率更高</a>: 未找到描述</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://youtu.be/yFKOOK6qqT8?si=EgnAQF3mVXWcElgH">在本地运行 Deepseek R1 671b AI LLM 是 ChatGPT 杀手！</a>: Deepseek R1 671b 本地设置与运行指南 https://digitalspaceport.com/running-deepseek-r1-locally-not-a-distilled-qwen-or-llama/ 768GB RAM 或 VR...</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/256#issuecomment-2620673643">从 Hugging Face 获取结果时出错，请稍后再试 · Issue #256 · lmstudio-ai/lmstudio-bug-tracker</a>: 使用最新版本，无法从 Hugging Face 搜索和下载模块，出现以下错误：从 Hugging Face 获取结果时出错，请稍后再试。如何设置 lm stu...</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: 👾 LM Studio CLI</a>: 👾 LM Studio CLI。通过在 GitHub 上创建账号为 lmstudio-ai/lms 的开发做出贡献。</li><li><a href="https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/256#issuecomment-262">从 Hugging Face 获取结果时出错，请稍后再试 · Issue #256 · lmstudio-ai/lmstudio-bug-tracker</a>: 使用最新版本，无法从 Hugging Face 搜索和下载模块，出现以下错误：从 Hugging Face 获取结果时出错，请稍后再试。如何设置 lm stu...</li><li><a href="https://lmstudio.ai/docs/api">将 LM Studio 作为本地 LLM API 服务器 | LM Studio 文档</a>: 使用 LM Studio 在 localhost 上运行 LLM API 服务器
</li>
</ul>

### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1334257962298118235)** (203 messages🔥🔥): 

> `DeepSeek 模型性能, Jetson Nano 讨论, 编程模型选择, AI 硬件配置, 编程的 Temperature 设置` 


- **DeepSeek 模型性能见解**：用户报告称，在 **GTX 1080** 和 **Ryzen 5 3600** 的配置下运行 DeepSeek 模型，无论线程池大小或 GPU offload 设置如何，速度约为 **6-7 tokens/秒**。
   - 调整模型大小并确保适配 VRAM 至关重要，因为超出 VRAM 会显著降低性能。
- **关于 Jetson Nano 价格的讨论**：讨论了 **Jetson Nano** 高达 **$500-$700** 的价格区间，导致许多人考虑使用真正的 GPU 等替代方案。
   - 参与者指出 Jetson Nano 似乎处于缺货状态，但一些卖家标价约为 **$250**。
- **为编程选择合适的模型**：对比了 **32B** 和 **70B** 等较小模型的性能，并指出两者都能有效处理复杂的编程任务。
   - 用户表示，虽然较小模型表现尚可，但建议查看 Hugging Face 等平台上的 Benchmark 以衡量预期性能。
- **优化 AI 工作站硬件**：**i9-14900KF** 搭配 **128GB RAM** 和双 **RTX 4090 GPU** 的配置，在合适的量化（quantizations）下，可以以 **30-40 tokens/秒** 的速度运行 **DeepSeek 70B** 模型。
   - 用户注意到，确保模型适配可用 VRAM 对于保持最佳性能非常重要。
- **设置 Temperature 以获得更好的 AI 输出**：参与者强调了设置模型 Temperature 的重要性，建议数值在 **0.5-0.7** 之间，以防止编程提示词中出现过度重复。
   - 较低的 Temperature 可以增强输出的连贯性，尤其是在使用 **DeepSeek** 等模型进行编程任务时。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://gyazo.com/b1903225526b9ea0039cdd6674d8ced8">Gyazo</a>:  </li><li><a href="https://www.theregister.com/2024/07/15/amd_ai_pc_goal/">AMD 表示未来的 PC 将以 100T/s 的速度运行 30B 参数模型</a>：他们将需要极大的内存带宽——更不用说容量了——才能实现这一目标</li><li><a href="https://tenor.com/view/whale-swallow-eat-nom-hungry-gif-17097355">Whale Swallow GIF - Whale Swallow Eat - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.tomshardware.com/pc-components/cpus/amd-launches-ryzen-ai-300-and-200-series-chips-for-laptops">AMD 为笔记本电脑发布 Ryzen AI 300 和 200 系列芯片</a>：Ryzen AI 展翅高飞。</li><li><a href="https://www.laptopmag.com/laptops/asus-zenbook-s16-um5606-ryzen-ai-9">华硕 Zenbook S16 UM5606 (Ryzen AI 9) 评测：我认为 AMD 刚刚干掉了 MacBook Air</a>：一款带有集成显卡且能玩游戏的笔记本？太疯狂了。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1334252552786808965)** (430 messages🔥🔥🔥): 

> `DeepSeek R1 性能, O1 Pro 使用, Aider 与本地模型的集成, O3 Mini 发布, 量化对模型的影响`

- **关于 DeepSeek R1 速度与性能的辩论**：用户讨论了 DeepSeek R1 的性能，指出该模型在不同硬件配置上的运行速度差异显著，例如在 4090 GPU 上达到了约 32 TPS。
   - 模型的 Quantized 版本因速度慢和 Instruction-following 能力差而受到批评，引发了对其实用性的担忧。
- **O1 Pro 作为 Coding 工具**：一些用户表示 O1 Pro 非常适合新项目的 Coding 以及对现有代码库进行修改，这引发了关于定价及其对重度用户整体价值的辩论。
   - 尽管有其优势，但与 Local Models 相比，使用成本和审查带来的限制也引起了讨论。
- **Aider 与 Local Models 的集成**：由于数据隐私问题，特别是使用托管在中国的模型时，将数据发送到 DeepSeek 等外部服务引起了关注。
   - 用户正在探索如何利用 Ollama 等 Local Models 进行 Aider 应用，而不损害敏感数据。
- **对 O3 Mini 发布的热切期待**：参与者正热切期待 O3 Mini 的发布，一些人推测它可能会增强他们的 AI 模型体验，并解决现有选项的一些性能缺陷。
   - 社交平台上分享了关于等待 O3 Mini 的幽默评论，它被视为在持续寻求更好模型性能过程中的潜在 Game-changer。
- **模型 Quantization 的影响**：讨论显示 Quantization 会显著影响模型性能，从而引发了关于模型大小与输出质量之间平衡的问题。
   - 参与者分享了使用不同 Quantized 版本模型的经验，注意到不同配置下质量和 Instruction-following 成功率的差异。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://discordapp.com/channels/1131200896827654144/1133060684540813372/1332845238267412480">Discord - 充满乐趣与游戏的群聊</a>: Discord 是玩游戏和与朋友闲聊的好地方，甚至可以建立全球社区。自定义你自己的空间来交流、玩耍和聚会。</li><li><a href="https://x.com/AravSrinivas/status/1884801300027589007">来自 Aravind Srinivas (@AravSrinivas) 的推文</a>: 所有 Perplexity Pro 用户现在每天可获得 500 次 DeepSeek R1 查询（无审查，且 Prompt 不会传往中国）。免费用户每天可获得 5 次查询。引用 Aravind Srinivas (@AravSrinivas) 每天 100 次...</li><li><a href="https://one.npr.org/?sharedMediaId=nx-s1-5279550:nx-s1-5343701-1">&#x1f50a; 立即收听：OpenAI 吹捧新的政府合作伙伴关系并支持 AI 基础设施</a>: NPR One 上的 All Things Considered | 7:59</li><li><a href="https://www.youtube.com/channel/UCC0O5FKSMcjzrvOUa08hS_A/videos">Michael Automates</a>: 我帮助成千上万的人学习如何自动化他们的加密货币，以增加成功机会。全自动地从上涨中获益并防范下跌风险。这帮助了 95% 的投资者...</li><li><a href="https://tenor.com/view/nalog-gif-25906765">Nalog GIF - Nalog - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://x.com/wiz_io/status/1884707816935391703">来自 Wiz (@wiz_io) 的推文</a>: 突发：内部 #DeepSeek 数据库公开泄露 🚨 Wiz Research 发现了 "DeepLeak" —— 一个属于 DeepSeek 的公开可访问的 ClickHouse 数据库，暴露了高度敏感的信息...</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>: R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。成本比 o1 低 14 倍。</li><li><a href="https://x.com/allen_ai/status/1884966600039915809">来自 Ai2 (@allen_ai) 的推文</a>: 这是 Tülu 3 405B 🐫，我们的开源后训练模型，性能超越了 DeepSeek-V3！Tülu 3 家族的最后一名成员证明了我们的方案（包括强化学习...）</li><li><a href="https://aider.chat/docs/scripting.html">脚本化 aider</a>: 你可以通过命令行或 Python 对 aider 进行脚本化操作。</li><li><a href="https://x.com/MistralAI/status/1884968836606136636">来自 Mistral AI (@MistralAI) 的推文</a>: 推出 Small 3，我们迄今为止最高效、最通用的模型！预训练和指令版本，Apache 2.0，24B，81% MMLU，150 tok/s。无合成数据，因此是任何推理任务的绝佳基础 - 祝大家...</li><li><a href="https://aider.chat/docs/config/adv-model-settings.html">高级模型设置</a>: 为 LLM 配置高级设置。</li><li><a href="https://openrouter.ai/perplexity/sonar-reasoning">Sonar Reasoning - API、提供商、统计数据</a>: Sonar Reasoning 是由 Perplexity 提供的一种推理模型，基于 [DeepSeek R1](/deepseek/deepseek-r1)。它允许开发者利用内置网络搜索的长思维链。运行 Sonar Reas...</li><li><a href="https://github.com/Aider-AI/aider/pull/3074/files.">共同构建更好的软件</a>: GitHub 是人们构建软件的地方。超过 1.5 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-llama-70b">DeepSeek R1 Distill Llama 70B - API、提供商、统计数据</a>: DeepSeek R1 Distill Llama 70B 是基于 [Llama-3.3-70B-Instruct](/meta-llama/llama-3) 的蒸馏大语言模型。通过 API 运行 DeepSeek R1 Distill Llama 70B</li><li><a href="https://github.com/marketplace/models/azureml-deepseek/DeepSeek-R1">DeepSeek-R1 · GitHub Models · GitHub</a>: 使用 DeepSeek-R1 创建 AI 驱动的应用程序</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1:free">DeepSeek R1 (免费) - API、提供商、统计数据</a>: DeepSeek R1 来了：性能与 [OpenAI o1](/openai/o1) 相当，但已开源并具有完全开放的推理 Token。它拥有 671B 参数，推理过程中有 37B 激活。运行...</li><li><a href="https://unsloth.ai/blog/deepseekr1-dynamic">运行 DeepSeek-R1 Dynamic 1.58-bit</a>: DeepSeek R-1 是最强大的开源推理模型，性能与 OpenAI 的 o1 模型相当。运行由 Unsloth 提供的 1.58-bit Dynamic GGUF 版本。</li><li><a href="https://x.com/jacksonhinklle/status/1884686222356079075">来自 Jackson Hinkle 🇺🇸 (@jacksonhinklle) 的推文</a>: 🚨🇨🇳🇺🇸 突发：中国击败了美国 🇨🇳 DeepSeek 击败了 OpenAI 🇺🇸🇨🇳 比亚迪击败了特斯拉 🇺🇸🇨🇳 华为击败了苹果 🇺🇸🇨🇳 华为击败了美国电信 🇺🇸🇨🇳 阿里巴巴击败了亚马逊 🇺🇸🇨🇳...</li><li><a href="https://aimlapi.com/">通过单一 AI API 访问 200 多个 AI 模型 | AIMLAPI.com</a>: 通过低延迟和高可扩展性的 AI API 访问超过 200 个 AI 模型。与 OpenAI 相比，最高可节省 80%。快速、经济高效，非常适合高级机器学习项目。AI Playground。</li><li><a href="https://fourthievesvinegar.org/">Four Thieves Vinegar Col

lective</a>: 身体维修权（Right to Repair–for Your Body）。四贼醋集体（Four Thieves Vinegar Collective）是一个无政府主义集体，致力于为那些有需要但无法获得药物和医疗技术的人提供获取途径...</li><li><a href="https://github.com/bytedance/UI-TARS?">GitHub - bytedance/UI-TARS</a>: 通过在 GitHub 上创建账号，为 bytedance/UI-TARS 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1334274075027312777)** (45 messages🔥): 

> `Aider 上下文包含, Azure AI 部署问题, 模型配置挑战, 在不同模式下使用 Aider, Aider 中的文件创建提示` 


- **用户询问 Aider 的上下文包含功能**：一位用户询问 Aider 是否有自动包含与当前编辑文件相关文件的功能，以获得更好的 Prompt 效果。
   - 有人提到可以手动将文件读取到聊天中，另一位成员讨论了在 architect 模式下修改文件。
- **对 Azure AI 部署端点的困惑**：一位成员表示难以确认 Azure R1 部署需要哪些多个端点和密钥，并在连接 Aider 时遇到错误。
   - 建议包括查看 GitHub issues 寻找解决方案，以及尝试 Aider 内部备选的 'azure_ai' 实现进行测试。
- **在 Aider 中配置模型**：用户讨论了为 Aider 中的各种命令设置不同模型的方法以优化性能，特别是默认模型与特定模型的使用。
   - 一位成员建议在常规使用时保持一个良好的 coding 模型，仅在处理复杂任务时才切换到智能模型。
- **在 chat-only 模式下运行 Aider**：一位用户询问是否可以仅将 Aider 用于聊天而不涉及任何代码，以便专注于项目相关的讨论。
   - 另一位成员建议使用 '/reset' 命令来防止代码被添加到 Prompt 中。
- **文件创建提示导致困惑**：用户报告 Aider 间歇性地尝试创建带有随机名称或代码片段的文件，令人感到沮丧。
   - 有评论指出 Aider 的编辑器有时会误解输入，导致产生不想要的文件创建提示。



**提及的链接**：<a href="https://aider.chat/docs/config/adv-model-settings.html#model-settings">Advanced model settings</a>：为 LLM 配置高级设置。

  

---

### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1334313324015521802)** (11 条消息🔥): 

> `DeepSeek 数据库泄露, Aider 只读存根 (Read-Only Stubs), Aider Awesome GitHub 仓库, Pull Request 改进, Bash 一行命令 (One-Liners)` 


- **DeepSeek 数据库泄露敏感信息**：据报道 **DeepSeek** 数据库泄露了包括用户聊天记录在内的敏感信息，详情在 [Hacker News](https://news.ycombinator.com/item?id=42871371) 上进行了讨论。
   - *用户对此次泄露导致的数据隐私和安全问题表示担忧。*
- **关于 Aider 功能的新 YouTube 视频**：最近一段名为 [“导航大型代码库：Aider 的只读存根解决方案”](https://youtu.be/XE6v_RGe0-U) 的视频讨论了 Aider 中通过只读存根 (read-only stubs) 增强 AI 编程的方案。
   - 该视频重点介绍了旨在改善 AI 与大型代码库交互的新草案功能。
- **为 Aider 收集一行命令和提示词建议**：“Aider Awesome” 是一个由成员提议的 GitHub 仓库，用于收集专门针对 aider-chat 的有用一行命令 (one-liner) 和提示词 (prompt) 建议，旨在提升用户体验 [GitHub - hux/aider-awesome](https://github.com/hux/aider-awesome)。
   - 对该仓库的反馈包括提高内容可读性的建议。
- **Aider Awesome 的 Pull Request 合并**：一位用户指出一个 [pull request](https://github.com/hux/aider-awesome/pull/1) 旨在提高 Aider Awesome 仓库的可读性。
   - 该 pull request 已成功合并，贡献者们对这一过程表示满意。
- **关于 Bash 一行命令的讨论**：一位成员表达了在 **bash** 中使用 **one-liners** 的偏好，强调了它们作为单条命令的高效性。
   - 对话强调了在脚本编写中使用简洁命令策略的简单性和有效性。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/hux/aider-awesome">GitHub - hux/aider-awesome: Prompts and helpers that work well with aider-chat</a>：与 aider-chat 配合良好的提示词和辅助工具 - GitHub - hux/aider-awesome: Prompts and helpers that work well with aider-chat</li><li><a href="https://youtu.be/XE6v_RGe0-U">Navigating Large Codebases: Aider&#39;s Read-Only Stub Solution (re-upload)</a>：在 Aider 中通过只读存根增强 AI 编程。在本集中，我们深入探讨了 AI 助手 Aider 的一个新草案功能，该功能允许包含 r...</li><li><a href="https://github.com/hux/aider-awesome/pull/1">Update README.md by alexanderkjeldaas · Pull Request #1 · hux/aider-awesome</a>：未找到描述内容
</li>
</ul>

</div>
  

---


### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1334251977583886399)** (456 条消息🔥🔥🔥): 

> `DeepSeek R1, MCP 支持, Chat 和 Composer 中的 Token 使用情况, 本地模型, AI 模型安全风险` 


- **DeepSeek R1 添加新功能**：Windsurf 宣布 DeepSeek R1 和 V3 现已在其 composer 功能中可用，完全托管在西方服务器上。
   - 该更新包括 tool calling 能力，允许 R1 首次在 coding agent 模式下使用。
- **Token 使用情况可见性问题**：用户对 chat 和 composer 中缺乏 Token 使用情况的透明度表示担忧，并对 10k token 的上下文限制感到困惑。
   - 关于启用更长上下文限制的 beta 设置是否有效存在疑问。
- **MCP 服务器配置**：用户讨论了使用 bash 脚本配置 MCP 服务器，从而在 Cursor 中轻松添加各种 JSON 设置。
   - 这种方法允许用户运行不同的 MCP 服务器，而无需每次进行大量配置。
- **AI 模型中的潜在安全风险**：人们对机器学习模型中的潜在安全风险表示担忧，包括 payload 中隐藏代码执行的可能性。
   - 建议使用 modelscan 等工具来检查序列化攻击，以确保运行本地模型时的安全性。
- **本地 vs. 托管 AI 模型**：讨论强调了运行本地模型与 DeepSeek R1 等托管方案相比的挑战，隐私考量影响了用户的偏好。
   - 虽然一些用户希望有更好的本地模型集成，但其他人对性能和可靠性仍持怀疑态度。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://x.com/windsurf_ai/status/1885077046663217230">Windsurf (@windsurf_ai) 的推文</a>：DeepSeek R1 和 V3 现已在 Windsurf 中可用，完全托管在西方服务器上。我们在 R1 中实现了 tool calling，使其首次能够用于 coding agent。</li><li><a href="https://dev.to/krisplatis/auto-add-missing-imports-on-file-save-in-vs-code-1b89">未找到标题</a>：未找到描述</li><li><a href="https://www.securityweek.com/unprotected-deepseek-database-leaked-highly-sensitive-information/">未受保护的 DeepSeek 数据库泄露聊天记录及其他敏感信息</a>：属于中国 AI 公司 DeepSeek 的一个未受保护的数据库泄露了高度敏感的信息，包括聊天记录、密钥和后端数据。</li><li><a href="https://jfrog.com/blog/data-scientists-targeted-by-malicious-hugging-face-ml-models-with-silent-backdoor/">数据科学家成为带有静默后门的恶意 Hugging Face ML 模型的目标</a>：Hugging Face 是基于模型的攻击目标吗？查看攻击机制的详细解释以及识别真实威胁所需的操作 &gt;</li><li><a href="https://forum.cursor.com/latest">Cursor - 社区论坛</a>：讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://aider.chat/2025/01/24/r1-sonnet.html">R1+Sonnet 在 aider 的多语言基准测试中创下 SOTA</a>：R1+Sonnet 在 aider 多语言基准测试中创下了新的 SOTA。与 o1 相比，成本降低了 14 倍。</li><li><a href="https://forum.cursor.com/t/sonnet-3-5-stops-working/46053">Sonnet 3.5 停止工作</a>：当我启用 OpenAI API key 但未启用 Anthropic 时，它仍然尝试对服务器进行自定义 API 调用。我希望它只运行 OpenAI 模型而不运行 Anthropic。如果我禁用 OpenAI，它可以在 Anthropic 上运行...</li><li><a href="https://forum.cursor.com/t/cursor-does-not-send-files-to-claude/43948/6">Cursor 不向 Claude 发送文件</a>：今天我突然失去了在 Cursor 中通过 @ 共享任何文件的能力。我检查了日志，但没看到任何调试或错误信息。我尝试了新会话、重启机器，甚至升级...</li><li><a href="https://forum.cursor.com/t/o3-mini-support/46324">O3-mini 支持</a>：OpenAI 将在大约 30 分钟内发布 o3-mini，我们如何能立即使用这个模型？请给我一些使用 OpenRouter/直接 API 调用的技巧，以便我知道发布时该怎么做。</li><li><a href="https://www.mcpservers.ai/">MCP 服务器</a>：浏览最大的 Model Context Protocol 服务器库。与他人分享你创建的 Model Context Protocol 服务器。</li><li><a href="https://www.hailuo.ai/">海螺 AI - 您的智能解决方案终极 AI 助手</a>：探索海螺 AI，这是您在 AI 搜索、视觉、语音聊天等各个领域提供先进解决方案的首选 AI 助手。体验快速、准确的信息检索和...</li><li><a href="https://x.com/imrat/status/1884904074379759872?s=46">Imrat (@imrat) 的推文</a>：Cursor 0.45.6 - 增加了 MCP 支持 - 迫不及待想试试！</li><li><a href="https://status.deepseek.com/">DeepSeek 服务状态</a>：未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1idzrdl/o3_releasing_tomorrow/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/daniel-lxs/mcp-server-starter">GitHub - daniel-lxs/mcp-server-starter</a>：通过在 GitHub 上创建账户，为 daniel-lxs/mcp-server-starter 的开发做出贡献。</li><li><a href="https://www.cursor.com/changelog">更新日志 | Cursor - AI 代码编辑器</a>：新的更新和改进。</li><li><a href="https://github.com/protectai/modelscan">GitHub - protectai/modelscan: 防范模型序列化攻击</a>：防范模型序列化攻击。通过在 GitHub 上创建账户，为 protectai/modelscan 的开发做出贡献。</li><li><a href="https://github.com/microsoft/BitNet">GitHub - microsoft/BitNet: 1-bit LLMs 的官方推理框架</a>：1-bit LLMs 的官方推理框架。通过在 GitHub 上创建账户，为 microsoft/BitNet 的开发做出贡献。</li><li><a href="https://github.com/modelcontextprotocol/servers">GitHub - modelcontextprotocol/servers: Model Context Protocol 服务器</a>：Model Context Protocol 服务器。通过在 GitHub 上创建账户，为 modelcontextprotocol/servers 的开发做出贡献。</li><li><a href="https://github.com/supabase-community/mcp-supabase">GitHub - supabase-community/mcp-supabase: 一组将 LLMs 连接到 Supabase 的 MCP 服务器</a>：一组将 LLMs 连接到 Supabase 的 MCP 服务器 - supabase-community/mcp-supabase</li><li><a href="https://pureinsights.com/blog/2024/1-bit-llms-the-future-of-efficient-ai/">1-Bit LLMs：高效 AI 的未来？ - Pureinsights</a>：本博文解释了关于 1-bit LLMs 的初步研究，以及它们在生产既有效又高效的 AI 模型方面的潜力。</li><li><a href="https://www.minim">未找到标题</a>：未找到描述</li>

axi.com/en/news/minimax-01-series-2">MiniMax - Intelligence with everyone</a>: 未找到描述</li><li><a href="https://openrouter.ai/minimax/minimax-01">MiniMax-01 - API, 提供商, 统计数据</a>: MiniMax-01 结合了用于文本生成的 MiniMax-Text-01 和用于图像理解的 MiniMax-VL-01。它拥有 4560 亿个参数，其中 45。通过 API 运行 MiniMax-01
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1334252639428284528)** (295 条消息🔥🔥): 

> `Nous x Solana 活动、新模型发布、Psyche 与分布式学习、Mistral Small 模型发布、社区对 AI Agents 的见解` 


- **Nous x Solana 活动引发热议**：即将在纽约举行的 Nous x Solana 活动参与人数众多，由于名额有限，导致出现了大量入场申请。
   - 与会者对围绕 AI 模型分布式训练基础设施的潜在讨论表现出极大的热情。
- **对模型发布的兴奋**：社区成员正在热烈讨论新发布的模型，包括 Mistral Small，该模型声称在小型语言模型中树立了新标杆。
   - 许多参与者希望了解其可用性，并将其与现有模型进行性能对比。
- **Psyche 推出分布式学习基础设施**：Nous 宣布了 Psyche，这是一个用于开放 AI 模型的分布式训练网络，旨在通过模块化系统促进大规模 Reinforcement Learning (RL) 过程。
   - 该项目因其在创新 AI 训练方法论方面的潜力而获得了积极反馈。
- **终端用户协作与开源进展**：关于 Psyche 开源的讨论正在进行中，未来计划包括潜在的共识算法和其他资源。
   - 社区显然希望通过 GitHub 获得更好的可访问性，并询问了有关 Psyche 的特定频道。
- **围绕与 Nous 相关的 AI Agents 的讨论**：成员们表示有兴趣了解目前是否有任何 AI Agents 与 Nous 相关联，以及是否有相关列表。
   - 社区继续探索 Nous 生态系统内 AI 发展的意义。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/allen_ai/status/1884966600039915809">来自 Ai2 (@allen_ai) 的推文</a>：这是我们的开源后训练模型 Tülu 3 405B 🐫，其性能超越了 DeepSeek-V3！Tülu 3 家族的最后一名成员证明了我们的方案（包括 Reinforcement...）</li><li><a href="https://arcprize.org/blog/r1-zero-r1-results-analysis">R1-Zero 和 R1 结果与分析</a>：对 DeepSeek R1 的分析</li><li><a href="https://tenor.com/view/popcorn-minions-popcorn-day-laugh-gif-5026739">小黄人吐爆米花 GIF - Popcorn Minions Popcorn Day - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://arxiv.org/abs/2402.03804">ReLU$^2$ 获胜：为稀疏 LLMs 发现高效激活函数</a>：稀疏计算通过动态跳过非活跃神经元的计算，为低资源场景下的 Large Language Models (LLMs) 推理提供了一个极具吸引力的解决方案。虽然传统...</li><li><a href="https://x.com/BlinkDL_AI/status/1884768989743882276">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：我提出了 ZeroCoT：一种从零开始引导 CoT 的简单方法。让我知道你的想法 🙂 我很快会在 RWKV 上尝试这个。引用 BlinkDL (@BlinkDL_AI) 让我们干掉 Attention。RWKV-7 "Goose"...</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/Teknium1/status/1884740956911718853?t=0NwHRMjFT001dlRoRvAPUw&s=19">来自 Teknium (e/λ) (@Teknium1) 的推文</a>：@ylecun https://x.com/Teknium1/status/1883955152442515637 引用 Teknium (e/λ) (@Teknium1) 今天 Nous 宣布了 Psyche 的到来——一个分布式网络和训练框架，一个基础设施层...</li><li><a href="https://team.doubao.com/en/special/doubao_1_5_pro">无标题</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/open-thoughts/OpenThoughts-114k">open-thoughts/OpenThoughts-114k · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/robertcsordas/moe_layer">GitHub - RobertCsordas/moe_layer: sigma-MoE 层</a>：sigma-MoE 层。通过在 GitHub 上创建账号来为 RobertCsordas/moe_layer 的开发做出贡献。</li><li><a href="https://github.com/ibm-granite/granite-3.0-language-models/blob/main/paper.pdf">ibm-granite/granite-3.0-language-models 主分支下的 paper.pdf</a>：通过在 GitHub 上创建账号来为 ibm-granite/granite-3.0-language-models 的开发做出贡献。</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 现已在 Azure AI Foundry 和 GitHub 上可用 | Microsoft Azure 博客</a>：DeepSeek R1 可通过 Microsoft Azure AI Foundry 和 GitHub 的模型目录获取，使企业能够无缝集成先进的 AI。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1334647219881640029)** (1 条消息): 

> `Autoregressive generation on CLIP embeddings, Multimodal inputs, Stable Diffusion generation` 


- **探索基于 CLIP Embeddings 的 Autoregressive Generation**：一名成员询问了在 **CLIP embeddings** 上进行 **autoregressive generation** 的可行性，这些 embeddings 用于将多模态输入投影到单个潜空间（latent space）中。
   - 他们注意到专门针对这种生成方法的信息较少，并强调其更常用于 **Stable Diffusion** 生成中的引导（guidance）。
- **理解 CLIP 与多模态集成**：讨论围绕 **CLIP** 如何有效地将各种模态整合到统一方法中的基础知识展开。
   - 尽管它在引导模型方面有所应用，但对话指出需要更多关于利用 CLIP 进行生成过程的见解。


  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1334360329152630936)** (2 条消息): 

> `China's AI Models, AI Race, Top-tier Models` 


- **中国拥有 10 个顶级 AI 模型**：讨论透露，中国的 AI 版图包括 **10 个从零开始训练的顶级模型**，其能力与欧洲最大的模型（包括 **Mistral**）相当或更高。
   - 一位成员强调，*DeepSeek 并不是中国唯一的优秀 AI 模型*，展示了美国以外地区发展的广度。
- **竞争格局中的美国 AI 实验室**：美国仅拥有 **5 个主要的 AI 实验室**——**OpenAI**、**Anthropic**、**Google**、**Meta** 和 **xAI**——在 AI 领域的这一规模上具有竞争力。
   - 这一简报强调 **AI 竞赛正处于白热化阶段**，并对 AI 开发的全球领导地位产生影响。



**提到的链接**：<a href="https://x.com/deedydas/status/1884786839913111931">Deedy (@deedydas) 的推文</a>：DeepSeek 并不是中国唯一的优秀 AI 模型。有 10 个顶级模型都是从零开始训练的（等于或优于欧洲 / Mistral 最大的模型）。美国只有 5 个实验室——OpenAI, Anth...

  

---

### **Yannick Kilcher ▷ #[general](https://discord.com/channels/714501525455634453/986699377257119794/1334269194270670909)** (170 messages🔥🔥): 

> `Reinforcement Learning vs. Deep Learning, DeepSeek 进展, LLM 学习策略, Pretraining 与 fine-tuning 框架, LLM 训练的教育类比` 


- **Reinforcement Learning 的细微差别**：Red_code 认为现代 **Reinforcement Learning (RL)** 应该利用现有知识并增强推理能力，超越传统的试错法（trial-and-error）。
   - Zickzack 反驳道，虽然 representation learning 并非新鲜事，但 RL 的独特视角使其能够更有效地解决 credit assignment 和 memory 问题。
- **DeepSeek 的潜力**：人们对 **DeepSeek** 最近的性能提升感到兴奋，讨论集中在它与经典模型相比能更高效地取得成果的能力。
   - Red_code 指出，他们的目标是利用 prospective configuration 的理念来优化推理和 representation learning。
- **LLM 中的学习策略**：Albert_lum 分享了关于将 **prior knowledge** 集成到 RL 中的见解，强调适当的学习策略可以增强 RL 的能力。
   - 对话强调了区分 DL 和 RL 的重要性，以及两者如何互补。
- **LLM 的教育框架**：Erkinalp 通过对比教科书结构引入了一个理解 LLM 训练的框架，概述了三种主要的信息类型：背景（background）、演示（demonstration）和练习题（practice problems）。
   - 他强调，虽然 LLM 已经广泛接触了前两种类型，但引入练习题代表了有意义学习的新前沿。
- **社区参与和幽默**：成员们表达了对社区的喜爱，并评论了他们对阅读 AI 相关研究论文的兴趣。
   - Albert_lum 幽默地建议把不喜欢的人称为 'segmentation fault'，展现了讨论中轻松愉快的氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/MistralAI/status/1884967826215059681?t=VAZyVB2GkDpC_KOy_2mZCQ&s=19">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://x.com/atroyn/status/1884695801773060502">来自 anton (𝔴𝔞𝔯𝔱𝔦𝔪𝔢) (@atroyn) 的推文</a>: 我在 @aixventureshq 听 @chrmanning 关于 DeepSeek R1 的演讲。让我们看看发生了什么。</li><li><a href="https://x.com/karpathy/status/1885026028428681698">来自 Andrej Karpathy (@karpathy) 的推文</a>: 我们必须带 LLM 去上学。当你打开任何教科书时，你会看到三种主要的信息类型：1. 背景信息 / 阐述。教科书中解释概念的核心内容。...</li><li><a href="https://x.com/karpathy/status/1883941452738355376">来自 Andrej Karpathy (@karpathy) 的推文</a>: 对于这篇关于 V3 的早期帖子，我没有太多要补充的，我认为它也适用于 R1（这是最近的、思考能力的等效模型）。我想说的是，Deep Learning 有一个传奇般的 ra...</li><li><a href="https://youtu.be/lYWIkwvaUIg?t=150">Kristin Bauer - True Blood 第 5 季第 10 集: «Gone, Gone, Gone» [完整版]</a>: Like: http://facebook.com/kristinbauerfans Actress: Kristin Bauer 饰演 Pam TV Serie: True Blood Season: number 5 Episode: number 10: «Gone, Gone, Gone»**No copyrig...</li><li><a href="https://youtu.be/W_jPWjzzuaw?t=20">Kristin Bauer - True Blood 第 5 季第 9 集: « Everybody Wants To Rule The World» [完整版]</a>: Like: http://facebook.com/kristinbauerfans Actress: Kristin Bauer 饰演 Pam TV Serie: True Blood Season: number 5 Episode: number 9: « Everybody Wants To Rule The W...</li><li><a href="https://www.nature.com/articles/s41593-023-01514-1">在可塑性之前推断神经活动，作为超越反向传播的学习基础 - Nature Neuroscience</a>: 本文介绍了 ‘prospective configuration’，这是一种神经网络学习的新原理，它不同于 backpropagation，在学习效率上更高，且更具一致性...
</li>
</ul>

</div>
  

---

### **Yannick Kilcher ▷ #[paper-discussion](https://discord.com/channels/714501525455634453/1045297868136779846/1334253123488841738)** (39 条消息🔥): 

> `OpenAI 指控, AI 技术担忧, Dario Amodei 的博客文章, AI Safety 资金, 每日论文讨论` 


- **针对 OpenAI 的指控引发关注**：成员们讨论了针对 OpenAI 的指控，有人认为这更像是一场“抹黑行动（smear job）”而非建设性批评。持续的讨论指向了重大网络攻击的背景，表明**政府情绪**可能正在影响公众认知。
   - *“OpenAI 陷入了困境，”* 一位成员评论道，暗示在面临审查时，其法律手段带有一种紧迫感。
- **Dario Amodei 遭到抨击**：在持续的讨论中，Dario Amodei 受到批评，言论将其贴上 AI 领域“最明显的骗子”之一的标签。关于他最近为 **AI Safety** 筹集 **10 亿美元**资金的评论被一些成员视为可疑。
   - *“他比 Scam Altman 还没底线（less diversified）”* 这种情绪反映了对其行为意图的怀疑。
- **对 AI 代码质量的担忧**：关于 AI 在专业软件开发中有效性的讨论出现，一位成员断言**模型质量低于要求**。其他人附和道，虽然像 Claude 这样的模型可以生成代码，但它们经常使任务**过度复杂化**，且无法维持上下文。
   - *“让 Claude 输出正确代码通常比你自己动手写还要费劲，”* 这句话概括了许多人对当前 AI 能力的挫败感。
- **参与每日讨论**：几位成员反思了他们在每日讨论中的参与程度，表示由于定期参与，他们通常对正在进行的话题了如指掌。一个幽默的评论建议，讨论确实让参与者在知识储备方面感觉像 *hackers*。
   - 一位新成员表达了参加未来论文复现（paper reviews）的好奇心，表明这里对渴望倾听和学习的新手有着友好的氛围。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.12370">Parameters vs FLOPs: Scaling Laws for Optimal Sparsity for Mixture-of-Experts Language Models</a>: 扩展语言模型的容量已被证明是提高性能和解锁新能力的可靠方法。容量主要由两个维度定义：...</li><li><a href="https://tenor.com/view/hair-flip-duhh-gif-26170789">Hair Flip GIF - Hair Flip Duhh - Discover &amp; Share GIFs</a>: 点击查看 GIF
</li>
</ul>

</div>
  

---


### **Yannick Kilcher ▷ #[agents](https://discord.com/channels/714501525455634453/1269724655405498429/1334540563667488858)** (7 条消息): 

> `PydanticAI, LlamaIndex, LangChain, 模型性能, 未来 Agent 框架` 


- **PydanticAI 领先但结果滞后**：在探索 **PydanticAI** 时，用户发现其 API 最为友好，具有内部温度设置，但结果经常产生损坏的 JSON 响应。
   - *检查服务器请求被认为具有挑战性*，尽管 PydanticAI 的界面很吸引人，但目前从 **LlamaIndex** 获得的结构化输出效果最好。
- **LangChain 的前车之鉴**：一位成员警告不要使用 **LangChain**，称其“愚蠢的管道语法”使故障排除变得复杂，尤其是在出现问题时。
   - 相比之下，推荐使用 LlamaIndex，因为它性能更好、麻烦更少且数据丢失更少。
- **低端模型的挣扎**：另一位成员强调了使用 **Llama3.2** 等低端模型时面临的**挑战**，发现很难提取上下文并输出结构化数据。
   - 他们注意到，观察服务器端输出有助于他们优化 Prompt 并改善模型交互。
- **Logfire UI 性能问题**：有人投诉 Logfire UI 在浏览器闲置时 **CPU/GPU 占用率过高**，而关闭标签页后占用率显著下降。
   - 这一观察突显了 UI 潜在的低效及其对系统资源的影响。
- **Agent 框架改进愿望清单**：为未来的框架提出了一个愿望清单，包括**检查网络流量**的能力以及访问有关模型和使用指标的元数据。
   - 关键建议包括用于在模型之间平滑转换的模型池（model pool）机制，以及针对各种 Prompt 衡量响应质量的能力。


  

---

### **Yannick Kilcher ▷ #[ml-news](https://discord.com/channels/714501525455634453/853983317044756510/1334291066131710044)** (64 条消息🔥🔥): 

> `DeepSeek 知识产权争议、欧盟 AI 战略反应、Mistral Small 3 发布、Tülu 3 405B 发布、多语言训练挑战` 


- **OpenAI 指控 DeepSeek 窃取知识产权**：在最近的一场争议中，OpenAI 和 AI 主管 David Sacks 指控 [DeepSeek](https://www.youtube.com/watch?v=hpwoGjpYygI) 窃取其技术来训练新的 R1 模型，引发了巨大反响。
   - 这种情况引发了在快速发展的市场中，关于 AI 技术所有权和伦理使用的疑问。
- **关于欧盟企业使用 AI 的辩论**：欧盟委员会透露，目前仅有 **13.5%** 的欧盟企业使用 AI，这促使人们呼吁制定新的 AI 战略以增强各行业的采用率。
   - 成员们表示怀疑，认为提高 AI 开发水平应该是首要任务，而不仅仅是增加使用率。
- **Mistral Small 3 发布，规格令人印象深刻**：[Mistral Small 3](https://mistral.ai/news/mistral-small-3/) 是一款经过延迟优化的 24B 参数模型，提供了极具竞争力的性能和效率，据报道在 MMLU 基准测试中实现了超过 **81% 的准确率**。
   - 该模型专为本地部署设计，在相同硬件上比 Llama 3.3 70B 等更大型的竞争对手快 **3 倍**，且性能更优。
- **Tülu 3 405B 表现优于竞争对手**：[Tülu 3 405B](https://allenai.org/blog/tulu-3-405B) 的发布展示了开源权重模型（open-weight models）的进步，在与 DeepSeek v3 和 GPT-4o 的竞争中表现出色。
   - 其 **Reinforcement Learning from Verifiable Rewards (RLVR)** 框架的实施显著提升了模型性能。
- **多语言 AI 训练的挑战**：讨论围绕欧洲 AI 采用率低的问题展开，强调关于多语言模型训练的对话往往会引发对 GDPR 挑战的辩论。
   - 关于仅关注少数主要语言是否足以进行 AI 开发存在分歧，一些人主张提供更广泛的语言支持。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/fedora-tipshat-mlady-melady-athiest-gif-7191305">Fedora Tipshat GIF - Fedora Tipshat Mlady - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak">Wiz Research 发现暴露的 DeepSeek 数据库泄露敏感信息，包括聊天记录 | Wiz 博客</a>: DeepSeek 的一个公开可访问数据库允许对数据库操作进行完全控制，包括访问内部数据的能力。泄露内容包括超过一百万行的日志字符串...</li><li><a href="https://www.wheresyoured.at/deep-impact/">Deep Impact</a>: 原声带：The Hives — Hate To Say I Told You So。在过去的一周左右，尤其是周末，整个生成式 AI 行业陷入了混乱。这不会是一篇冗长的技术性...</li><li><a href="https://x.com/EU_Commission/status/1884635063054106770">来自欧盟委员会 (@EU_Commission) 的推文</a>: “只有 13.5% 的欧盟企业正在使用 AI。这必须改变。今年我们将为我们的大陆启动一项广泛的 AI 战略，包括一项‘应用 AI’倡议，以推动工业界采用人工智能...”</li><li><a href="https://allenai.org/blog/tulu-3-405B">扩展 Tülu 3 后训练配方以超越 DeepSeek V3 的性能 | Ai2</a>: 介绍 Tülu 3 405B，这是首次将完全公开的后训练配方应用于最大的开源权重模型。</li><li><a href="https://slator.com/meta-rolls-out-multimodal-llama-3-2-but-not-in-europe/">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=hpwoGjpYygI">OpenAI 称 DeepSeek 偷走了我们的技术...</a>: 使用 PostHog 构建更好的应用 https://posthog.com/fireship。OpenAI 和 AI 主管 David Sacks 指控 DeepSeek 窃取其知识产权来训练其新的 R1 模型...</li><li><a href="https://www.reddit.com/r/ABoringDystopia/comments/1ht7fft/used_meta_ai_to_edit_a_selfie_now_instagram_is/">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1334275589800202290)** (65 条消息🔥🔥): 

> `Tülu 3 405B 发布、Mistral Small 3 公告、DeepSeek 数据库暴露、OpenAI 在华盛顿的演示、软银与 OpenAI 的投资洽谈`

- **Tülu 3 405B 发布惊喜**：[Tülu 3 405B 的发布](https://allenai.org/blog/tulu-3-405B) 展示了开源 post-training 方案的可扩展性和有效性，其性能优于 **Deepseek v3** 和 **GPT-4o**。
   - “*Wow nature is healing*”（哇，大自然正在愈合）是大家共同的心声，反映了人们对 Tülu 团队创新成果的兴奋之情。
- **Mistral Small 3 承诺高效**：Mistral 宣布发布 **Mistral Small 3**，这是一个专为本地部署设计的 24B 参数模型，在低延迟下实现了 state-of-the-art 的性能。
   - 主要特点包括高度的**知识密集（knowledge-dense）**，且对各种生成式 AI 任务都非常有效，使其成为甚至在消费级硬件上部署的理想选择。
- **DeepSeek 敏感数据泄露**：[Wiz Research](https://x.com/wiz_io/status/1884707816935391703?s=61) 发现 DeepSeek 的一个公开可访问数据库暴露了敏感用户数据，包括密钥（secret keys）和聊天记录。
   - 对隐私和数据安全影响的担忧引发了关于 AI 平台所需**控制措施（control measures）**的讨论。
- **OpenAI 在华盛顿展示新技术**：Sam Altman 和 Kevin Weil 向华盛顿的美国政府展示了新技术，预计这次演示将引起重大反响。
   - OpenAI 之前的演示在历史上都引起了极大的关注，这表明这次活动也可能如此。
- **SoftBank 的投资意向**：有报道称，**SoftBank** 正在谈判，除了之前的承诺外，还将直接向 OpenAI 追加 **150-250 亿美元**的投资。
   - 这一举动标志着在竞争激烈的环境中，人们对大规模 AI 创业公司的兴趣和信心日益增强。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://x.com/MistralAI/status/1884967826215059681">Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://www.theguardian.com/technology/2025/jan/29/deepseek-blocked-some-app-stores-italy-questions-data-use">DeepSeek 在意大利部分应用商店被封禁，数据使用问题引发质疑</a>: 意大利和爱尔兰监管机构希望了解聊天机器人收集的数据如何被中国政府使用</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>: Apache 2.0, 81% MMLU, 150 tokens/s</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-30/deepseek-s-ai-restricted-by-hundreds-of-companies">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://www.bloomberg.com/news/articles/2025-01-30/deepseek-s-ai-restricted-by-hundreds-of-companies-within-days">Bloomberg - Are you a robot?</a>: 未找到描述</li><li><a href="https://x.com/hamishivi/status/1884990994883768721">Hamish Ivison (@hamishivi) 的推文</a>: 这是与 Tulu 3 团队所有人共同努力的一个有趣的副项目。特别鸣谢 @vwxyzjn（在训练和基础设施方面做了很多工作）和 @ljvmiranda（协助 DPO 数据生成...）</li><li><a href="https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-2501">mistralai/Mistral-Small-24B-Instruct-2501 · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/wiz_io/status/1884707819737223591?s=61">Wiz (@wiz_io) 的推文</a>: 这意味着任何人都可以访问包含实际聊天消息、内部机密、服务数据的日志，并可能在窃取数据的同时提升服务器内的权限。</li><li><a href="https://x.com/techikansh/status/1884961297709572170">Techikansh (@techikansh) 的推文</a>: Theo，你有关于 o3-mini 的禁令消息吗？</li><li><a href="https://x.com/AndrewCurran_/status/1884950312525725736">Andrew Curran (@AndrewCurran_) 的推文</a>: Sam Altman 和 Kevin Weil 今天上午在华盛顿向新政府进行演示。据 Axios 报道，他们还在展示将于第一季度发布的新技术。最后 ...</li><li><a href="https://x.com/hamishivi/status/1884990987757596832">Hamish Ivison (@hamishivi) 的推文</a>: 来自 tulu 团队的小假期项目 :) 将 Tulu 的配方扩展到 405B 效果非常好！我们主要将其视为 open-instruct 可以扩展到大规模训练的确认 —— 更多令人兴奋的...</li><li><a href="https://x.com/firstadopter/status/1884794211091759444">tae kim (@firstadopter) 的推文</a>: FT（金融时报）：据多位直接了解谈判情况的人士透露，“SoftBank 正洽谈直接向 OpenAI 投资 150-250 亿美元，此外还承诺向 Stargate 投入超过 150 亿美元。”</li><li><a href="https://x.com/wiz_io/status/1884707816935391703?s=61">Wiz (@wiz_io) 的推文</a>: 突发：内部 #DeepSeek 数据库公开泄露 🚨 Wiz Research 发现了 “DeepLeak” —— 一个属于 DeepSeek 的公开可访问 ClickHouse 数据库，暴露了高度敏感的信息...</li><li><a href="https://x.com/btibor91/status/1884756371058634762">Tibor Blaho (@btibor91) 的推文</a>: ChatGPT 中更新的 GPT-4o 模型已确认 - “ChatGPT 中的 GPT-4o 更新（2025 年 1 月 29 日）” - 拥有更新的知识（2024 年 6 月），对上传图片的更深层理解和分析，智能...</li><li><a href="https://allenai.org/blog/tulu-3-405B">将 Tülu 3 后训练配方扩展至超越 DeepSeek V3 的性能 | Ai2</a>: 介绍 Tülu 3 405B，这是完全开放的后训练配方在最大 open-weight 模型上的首次应用。</li><li><a href="https://x.com/Mobius_Labs/status/1885010791344062704">Mobius Labs (@Mobius_Labs) 的推文</a>: 我们使用 logits 蒸馏改进了 DeepSeek R1 蒸馏模型 —— 在 GSM8K 上实现了 +4-14% 的提升，而每次训练运行仅花费 3-18 美元！🚀 现在已在 Hugging Face 上线 —— 高效运行它们...</li><li><a href="https://x.com/jiayi_pirate/status/1882839370505621655?mx=2">Jiayi Pan (@jiayi_pirate) 的推文</a>: 我们在 CountDown 游戏中复现了 DeepSeek R1-Zero，它确实有效。通过 RL，3B 基础 LM 自主发展出了自我验证和搜索能力。你可以体验到那个“啊哈”时刻...</li><li><a href="https://x.com/AndrewCurran_/status/1884971368531587573">Andrew Curran (@AndrewCurran_) 的推文</a>: 开始了。引用 Andrew Curran (@AndrewCurran_) 的话：Sam Altman 和 Kevin Weil 今天上午在华盛顿向新政府进行演示。据 Axios 报道，他们还在展示新的...</li><li><a href="https://x.com/altryne/status/1884778839009796411">Alex Volkov (Thursd/AI) (@altryne) 的推文</a>: 扎克伯格在财报电话会议上的亮点：- Llama 4 & Llama 4 mini (d...</li>

（带预训练的版本）- 确认了推理版 LLaMas！- Llama 4 将是原生多模态的 —— 它是一个全能模型（omni-model）—— 并且它将拥有...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1334276178210590852)** (28 条消息🔥): 

> `Meta 的法律挑战、V3 许可问题、对模型部署的担忧、许可证对 AI 发展的影响` 


- **Meta 面临许可方面的焦虑**：成员们对 **Meta 的脆弱性** 表示担忧，特别是在其 Llama 模型的版权主张方面，尤其是如果他们转而部署 **DeepSeek** 的话。
   - *一位成员指出，如果在许可协议下找到合理的理由，DeepSeek 可能会有动力在法律上挑战 Meta。*
- **V3 许可证不是 MIT**：关于 V3 的许可存在混淆；据指出这是一种限制性许可证，被比作“准 OSS”框架，可能会限制自由。
   - *有人指出，为了获得法律上干净的 V3 版本，必须进行重复工作，这既繁琐又令人担忧。*
- **法律条款产生责任**：讨论强调，超出 MIT/Apache 范围的 **“不作恶”条款** 是有问题的，因为它们可能会产生开放式的法律责任。
   - *一位成员幽默地提到 JSON 许可证因复杂的法律语言而导致问题，反映了法律条款不可预测的本质。*
- **V3 的审查风险**：有人担心发布 **V3 finetune** 可能会导致无意的违规，特别是在涉及像 **天安门广场** 这样敏感话题的讨论时。
   - *一位成员强调，即使某种违规行为仅在某个国家属于犯罪，许可协议也可能导致诉讼，无论所在地在哪里。*
- **许可证简直疯狂**：几位成员分享了他们对许可证这一既充满压力又引人入胜的主题的感受，强调了它们在 AI 领域的复杂性。
   - *一位成员指出，许可证可能会引发永久性的责任状态，使合规几乎变得不可能。*


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1334271624660389961)** (26 messages🔥): 

> `DeepSeek R1 Launch, Speculations on Model Performance, Quantization in GPT-4, Updates on Tulu 3 Paper, Emerging Reasoning Models` 


- **DeepSeek R1 在 Azure AI Foundry 上线**：DeepSeek R1 现已在 [Azure AI Foundry](https://azure.microsoft.com/en-us/products/ai-foundry) 和 GitHub 的模型目录中可用，将其产品组合扩展到超过 **1,800 个模型**，包括各种 AI 类型。
   - *正如博客所述*，此次发布使企业能够无缝集成先进 AI，同时确保安全性和负责任的 AI 承诺。
- **对模型可靠性的担忧**：关于 OpenAI 模型生成结果是否可靠的推测仍在继续，特别是与**低精度 Quantization** 相关的问题。
   - 一些成员将 **GPT-4** 输出质量的波动归因于 Quantization 问题，这可能损害了模型的性能。
- **Tulu 3 论文更新引发关注**：Tulu 3 论文最近进行了更新，社区成员观察到这种响应速度后感到非常兴奋。
   - *一位用户指出*，“目睹 arXiv -> 媒体的管道如此疯狂”，反映了信息的快速传播。
- **新兴 Reasoning Models 技术讨论**：加拿大的团队正准备进入 **smol reasonoor 赛道**，探索将 tool use 和 RAG 集成到 Reasoning Models 中，这被认为是值得关注的。
   - 然而，他们在推理过程中的具体实现技术细节尚不明确，这引起了开发者的一些挫败感。
- **关于 FP8 支持的推测**：有传言称 **Ascend 910c** 可能不支持原生 FP8，这引发了对未来模型训练能力的质疑。
   - 这一推测一直是讨论的话题，社区成员分享了他们对性能影响的看法。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/johnschulman2/status/1884740922744983715">来自 John Schulman (@johnschulman2) 的推文</a>：我认为这个术语是由 @bobmcgrewai 在 ChatGPT 早期创造（或推广）的。出于历史原因，进行 ChatGPT 微调的团队之前被称为 RL 团队，Bob 建议...</li><li><a href="https://x.com/1vnzh/status/1884899043047887243">来自 Ivan Zhang (@1vnzh) 的推文</a>：chat 这是真的吗，帮兄弟在异乡庆祝 CNY</li><li><a href="https://x.com/amir/status/1885012737614635280">来自 Amir Efrati (@amir) 的推文</a>：你们现在基本上可以免费制作自己的类 o1 Reasoning Model 了。</li><li><a href="https://fxtwitter.com/erykbanatt/status/1884857074833584269">来自 Eryk (@erykbanatt) 的推文</a>：注意到 Tulu 3 论文刚刚更新了 👀</li><li><a href="https://azure.microsoft.com/en-us/blog/deepseek-r1-is-now-available-on-azure-ai-foundry-and-github/">DeepSeek R1 现已在 Azure AI Foundry 和 GitHub 上线 | Microsoft Azure 博客</a>：DeepSeek R1 通过 Microsoft Azure AI Foundry 和 GitHub 上的模型目录提供，使企业能够无缝集成先进 AI。
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[memes](https://discord.com/channels/1179127597926469703/1187551504995987576/1334275598637338624)** (9 messages🔥): 

> `Teortaxes 评论，Deepseek R1 训练泄露，AME(R1)CA 版本，Mistral Small 3 架构，数据可视化热情` 


- **Teortaxes 带有转折的自我描述**：一位成员幽默地评论道，**Teortaxes** 似乎主要是在描述他自己，但指出 **Žižek 声音**让一切变得永远更好了。
   - *“齐泽克（Zizek）的声音让一切永远变得美好。”*
- **泄露的 Deepseek R1 训练片段**：[Aidenybai 的推文](https://x.com/aidenybai/status/1884826039723114901) 展示了一段展示 **Deepseek R1** 训练过程的**泄露视频**。
   - 社区对泄露内容中分享的见解表达了极大的热情。
- **为美国价值观推出的 AME(R1)CA**：[Tylercosg 的推文](https://x.com/tylercosg/status/1884747401744855467) 介绍了 **AME(R1)CA**，这是 **Deepseek R1** 的一个版本，旨在符合**美国价值观**并远离 CCP 的影响。
   - 他们的口号承诺为那些担心 **CCP** 影响的人提供解决方案。
- **为延迟优化 Mistral Small 3**：[Dchaplot 的推文](https://x.com/dchaplot/status/1884975429561487815) 强调 **Mistral Small 3 架构**专门针对延迟进行了优化。
   - 聊天中讨论了 **Mistral** 在此背景下不寻常的轴选择。
- **对数据可视化的热情**：一位成员在回应关于 Mistral 的持续讨论时表示，**数据可视化**是他们的热情所在。
   - *“数据可视化是我的热情所在。”*


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/dchaplot/status/1884975429561487815">来自 Devendra Chaplot (@dchaplot) 的推文</a>: Mistral Small 3 架构针对延迟进行了优化。2/N</li><li><a href="https://x.com/aidenybai/status/1884826039723114901">来自 Aiden Bai (@aidenybai) 的推文</a>: deepseek r1 训练的泄露视频</li><li><a href="https://x.com/tylercosg/status/1884747401744855467">来自 Tyler Cosgrove (@tylercosg) 的推文</a>: 担心 CCP 的影响但仍想使用 deepseek r1？别再担心了！向 AME(R1)CA 问好，这是一个转向美国价值观并远离那些讨厌的中国共产...的 r1 版本。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[rl](https://discord.com/channels/1179127597926469703/1208183216843005962/1334316467998429194)** (11 messages🔥): 

> `Tulu3 数据准备，verl 对比 GRPOTrainer，open-instruct 实现，HF GRPO 限制，open-instruct 中的 LoRA 支持` 


- **Tulu3 数据准备的最佳实践**：一位成员询问了关于 **Tulu3** 数据准备的特殊注意事项，以便在后训练（post-training）后优化 LLM 的 RLHF。
   - 另一位成员建议关注**感兴趣的领域**、**评估（evals）设置**，并确保**偏好数据（pref data）的稳定生成**。
- **比较 verl 和 Huggingface 的 GRPOTrainer**：一位成员表示对 **verl** 和 **Huggingface 的 GRPOTrainer** 的实际应用非常感兴趣，询问两者孰优孰劣。
   - 他们目前正在使用 verl，但发现它有一些**不完善之处（rough edges）**，促使他们评估是继续投入还是寻找更好的替代方案。
- **澄清 GRPO 限制**：讨论强调 **HF GRPO** 仅支持**每次更新 1 个梯度步（grad step）**，缺乏 **PPO** 固有的裁剪逻辑（clipping logic）。
   - 成员们辩论了这一限制的影响，其中一位成员引用了 **TRL 代码**进行澄清。
- **open-instruct 中的 LoRA 支持**：一位成员询问了 open-instruct 实现中 **LoRA 支持**的优先级，并根据当前的训练方法推测其优先级不高。
   - 大家都对是否会考虑 LoRA 支持表示好奇，并反思了其与即将进行的训练运行的相关性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1334263494211665972)** (65 messages🔥🔥): 

> `DeepSeek Math Paper, Mixture-of-Experts (MoE), Multi Token Prediction (MTP), DeepSeek v3 Architecture, Inferences and Experts Balancing` 


- **DeepSeek 论文标志着 RL 的突破**：**DeepSeek 数学论文**被广泛认为是 **Reinforcement Learning (RL)** 的重大突破，并在 v2 中引入了 **GRPO**，尽管 v2 论文主要因 **MLA** 而闻名。
   - 讨论指出，在当前 RL 进展的总体背景下，这项工作具有重要意义。
- **MTP 在 Speculative Decoding 方面引起关注**：**MTP** 被强调为 DeepSeek v3 的关键特性，它能以 **85-90% 的接受率**预测 **2 个 token**，这一点被许多框架所忽视。
   - 成员们对其在训练和推理阶段的作用表示好奇，特别是它与正则化（regularization）的关系。
- **DeepSeek v3 中的 MoE 创新**：DeepSeek v3 采用了 **Sigmoid 门控**而非 Softmax，允许专家并行运行而互不竞争，同时引入了**无丢弃负载均衡 (dropless load balancing)**。
   - 该架构在专家层之外还包含一个额外的通用层，改变了多用途专家在模型中发挥作用的视角。
- **探索专家模型中的负载均衡**：成员们讨论了在 v3 框架中平衡用于专家平衡的**辅助损失 (auxiliary losses)** 所面临的挑战，并对实际实现的细节提出了疑问。
   - 对话强调了关于这些组件如何影响模型性能，以及它们是否真正提升了推理速度的困惑。
- **关于 AI 演进的对话**：一位成员分享了对 AI **快速演进**及其给研究人员带来的压力的见解，这反映在他们焦虑的行为中。
   - 讨论提供了影响 AI 发展的社会动态背景，强调了该领域内感受到的紧迫感。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.06066">DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models</a>：在大语言模型时代，Mixture-of-Experts (MoE) 是一种在扩展模型参数时管理计算成本的有前途的架构。然而，传统的 MoE 架构如...</li><li><a href="https://www.hyperdimensional.co/p/novus-ordo-seclorum">Novus Ordo Seclorum</a>：关于 DeepSeek 的思考</li><li><a href="https://ghost.oxen.ai/no-hype-deepseek-r1-reading-list/">No Hype DeepSeek-R1 Reading List</a>：DeepSeek-R1 是 AI 开源模型生态系统迈出的一大步，其最新模型在多项指标上与 OpenAI 的 o1 展开竞争。目前有很多炒作和杂音...</li><li><a href="https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py#L898">modeling_deepseek.py · deepseek-ai/deepseek-moe-16b-base at main</a>：无描述</li><li><a href="https://cameronrwolfe.substack.com/p/moe-llms?utm_source=post-email-title&publication_id=1092659&post_id=154340424&utm_campaign=email-post-title&isFreemail=true&r=764e6&triedRedirect=true&utm_medium=email)">Mixture-of-Experts (MoE) LLMs</a>：从底层理解 DeepSeek、Grok 和 Mixtral 等模型...</li><li><a href="https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/deepseek_v3.py">vllm/vllm/model_executor/models/deepseek_v3.py at main · vllm-project/vllm</a>：一个用于 LLM 的高吞吐量且显存高效的推理与服务引擎 - vllm-project/vllm
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1334279230447554666)** (2 messages): 

> `Science Phrasing, Opinion on OAI, Metaphors in AI Discourse` 


- **“凭直觉做科学”的批评**：一位成员表示，AI 讨论的某些方面感觉太像“凭直觉做科学 (science by smell)”，暗示某些评估缺乏严谨性。
   - 这一观点表明，人们希望看到更清晰、更具体的指标，而不是模糊的评估。
- **OAI 中的“苦药”与“鲜味”**：另一位成员评论说，OAI 呈现了“最大剂量的苦药 (bitter pills)”，并将其与作为“鲜味 (umami)”的验证器 (verifiers) 进行对比，这暗示在艰难的局面中存在一种增味组件。
   - 这个隐喻强调了 OpenAI 环境中艰难真相与回报性见解的二元性。


  

---

### **Interconnects (Nathan Lambert) ▷ #[policy](https://discord.com/channels/1179127597926469703/1325523782806274089/1334256165944692848)** (13 messages🔥): 

> `Training Techniques in AI Models, Concerns about Data Sources, Deepseek Speculations, OpenAI Output Usage, Tulu Dataset in Training` 


- **蒸馏在 SOTA 性能中的作用**：讨论围绕 **distillation**（蒸馏）技术是否对 V3 中看到的 **SOTA** 结果有所贡献展开，并引发了关于在不使用蒸馏数据的情况下，**MLA/8 bit training** 等方法有效性的疑问。
   - 一位成员推测，如果性能与蒸馏挂钩，那么用于基座模型的训练策略可能需要重新评估。
- **困惑度（Perplexity）数值表明训练效果强劲**：有人指出，来自大规模数据集的困惑度数值看起来非常强劲，这表明所进行的 **training** 是有效的。
   - 成员们表示怀疑，认为如果没有坚实的训练基础，很难取得如此出色的结果。
- **关于 Deepseek 方法论的推测**：对于 **Deepseek** 是否使用 **ChatGPT** 进行数据过滤，大家看法不一。成员们注意到文风的相似性可能暗示了显著的蒸馏过程，但这似乎尚未得到证实。
   - 尽管存在各种理论，但有人建议使用 OpenAI 的输出结果具有中等可能性。
- **对 Deepseek 的谨慎态度**：参与者对 Deepseek 表现出一种**不信任**的态度，担心关于其能力的假设可能源于与竞争相关的**财务**焦虑。
   - 一些理论认为，无关的功能可能会影响 OpenAI endpoint 的使用。
- **对 Tulu 数据集影响的兴趣**：一位成员对在训练的 **SFT** 阶段利用 **Tulu data** 的前景表示热切期待，表明了其在社区中的价值。
   - 其他人承认 **ShareGPT4V** 是开源 **VLM** 领域中一个值得关注的数据集，称其为经典参考。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1334354896220323904)** (31 messages🔥): 

> `OpenAI's training ethics, RL methods and tool usage, Pythia language model sampling, Concerns over model performance, Tool dependency in LLMs` 


- **OpenAI 与 Deepseek 的伦理困境**：OpenAI 强调与 **Deepseek training** 相关的问题被认为是荒谬的，尤其是考虑到他们有使用那些他们旨在取代的人的数据的历史。
   - 成员们对 OpenAI 的法律主张以及在竞争领域中表现出胜任能力的动机表示怀疑。
- **探索 RL 方法对 LLM 工具使用的益处**：人们意识到使用 **reinforcement learning (RL)** 可以通过简短解释工具来减少对大规模数据集的需求，从而允许模型自主学习。
   - 有人担心如何保持平衡，以确保 LLM 不会过度依赖特定工具。
- **Pythia 模型的采样概率**：讨论围绕从高斯分布中采样经过训练的 **Pythia** 语言模型的概率展开，并承认了局部体积估计。
   - 强调了专注于表现出特定行为的网络“邻域”的概念，以完善分析。
- **蒸馏模型中的性能差异**：成员们注意到，尽管投入了大量的训练和资源，其他开源模型仍未能达到 **GPT-4o** 蒸馏模型的下游性能。
   - 有推测认为，像 **Alpaca** 这样的后训练（post-training）方法论参与了增强模型能力的过程。
- **AI 中工具依赖的危险**：社区成员讨论了 LLM 在解决问题时过度依赖某些工具的潜在风险，建议随机的工具可用性可能是一个好策略。
   - 分享了关于智能模型如何学习有效地应用工具，同时仍保留基本问题解决能力的思考。



**提及的链接**：<a href="https://www.overleaf.com/read/krhxtvkxjywb#416acf">Overleaf, Online LaTeX Editor</a>：一个易于使用的在线 LaTeX 编辑器。无需安装，支持实时协作、版本控制、数百个 LaTeX 模板等。

  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1334253288597753887)** (178 messages🔥🔥): 

> `Hyperfitting in LLMs, Critique Fine-Tuning, Backdoor Detection, Sampling Neural Networks, Generalization vs Memorization` 

- **LLM 中的过度拟合（Hyperfitting）**：讨论了模型在特定基准测试上表现过好而无法泛化到其他任务的现象。
- **批判性微调（Critique Fine-Tuning）**：探讨了通过让模型对自身输出进行批判和修正来提升性能的方法。
- **后门检测**：研究了如何识别神经网络中可能被恶意利用的隐藏触发器。
- **神经网络采样**：关于如何从复杂的参数空间中有效采样模型的研究。
- **泛化与记忆的对比**：深入探讨了模型是真正理解了底层逻辑，还是仅仅记住了训练数据。

- **Hyperfitting 增强了 LLM 文本生成**：最近的讨论强调，在小数据集上进行 **hyperfitting** 可以显著提高 LLM 的 **open-ended text generation** 能力，这与传统认知相反。
   - 例如，一个模型的人类偏好得分从 **4.9%** 攀升至 **34.3%**，尽管存在潜在的过拟合，但其表现已与更大规模的模型相当。
- **引入 Critique Fine-Tuning**：Critique Fine-Tuning (CFT) 鼓励模型学习并批判噪声响应，而不仅仅是模仿正确响应，从而实现了性能的持续提升。
   - 该方法在六个数学基准测试中得到了验证，显示出比传统的 supervised fine-tuning 提升了 **4-10%**。
- **对 backdoor 检测影响的担忧**：ARC 的 backdoor 论文指出，无法检测到的带有 backdoor 的模型可能与常规模型非常相似，随着模型规模变大，会导致潜在的 loss 不匹配。
   - 这引发了关于 loss functions 在区分 backdoor 模型和标准模型方面有效性的疑问。
- **神经网络中的采样技术**：围绕提议的通用 **Absolute Unit NN** 架构的讨论研究了由于扩展挑战，整体性能可能如何受到损害。
   - 批评者对这种方法的实用性表示担忧，特别是在 generalization 与 memorization 方面。
- **评估 CE-loss 作为训练指标**：大家达成共识，认为使用 **cross-entropy loss** (CE-loss) 作为衡量 LLM 能力的训练指标可能不足以衡量现实世界的性能。
   - 参与者质疑为什么该指标仍在使用，并强调缺乏评估模型能力的有效替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/BlinkDL_AI/status/1884768989743882276">来自 BlinkDL (@BlinkDL_AI) 的推文</a>：我提出了 ZeroCoT：一种从零开始引导 CoT 的简单方法。让我知道你的想法 🙂 我很快会在 RWKV 上尝试这个。引用 BlinkDL (@BlinkDL_AI) 让我们干掉 attention。RWKV-7 &#34;Goose&#...</li><li><a href="https://openreview.net/forum?id=Ij9ilPh36h">Hyperfitting 现象：为...锐化并稳定 LLM</a>：本文介绍了在极小数据集上过拟合预训练大语言模型 (LLMs) 的反直觉泛化结果。在 open-ended text generation 的设置下，它...</li><li><a href="https://arxiv.org/abs/2501.17703">Critique Fine-Tuning：学习批判比学习模仿更有效</a>：Supervised Fine-Tuning (SFT) 通常用于训练语言模型以模仿针对给定指令的注释响应。在本文中，我们挑战了这一范式并提出了 Critique Fine-Tuning...</li><li><a href="https://www.overleaf.com/read/krhxtvkxjywb#416acf">Overleaf，在线 LaTeX 编辑器</a>：一个易于使用的在线 LaTeX 编辑器。无需安装，支持实时协作、版本控制、数百个 LaTeX 模板等。</li><li><a href="https://arxiv.org/abs/2409.03077">Backdoor 防御、可学习性和混淆</a>：我们通过攻击者和防御者之间的博弈引入了针对 backdoors 的防御形式化概念。在这个博弈中，攻击者修改一个函数，使其在特定的输入上表现不同...</li><li><a href="https://arxiv.org/abs/2405.14722">DAPE：用于长度外推的数据自适应位置编码</a>：Positional encoding 在 Transformer 中起着至关重要的问题，显著影响模型性能和长度泛化。先前的研究引入了绝对位置编码 (APE) 和相对...</li><li><a href="https://arxiv.org/abs/2406.11235">QTIP：使用格状结构和不相干处理的量化</a>：训练后量化 (PTQ) 通过将权重转换为低精度数据类型来减少 LLM 的内存占用。由于 LLM 推理通常受内存限制，PTQ 方法可以提高推理速度...</li><li><a href="https://arxiv.org/abs/2306.16830">深度神经网络的采样权重</a>：我们为全连接神经网络的权重和偏置引入了一种概率分布，并结合了一种高效的采样算法。在监督学习背景下，无需迭代优化...</li><li><a href="https://youtu.be/1GCf29FPM4k?si=osypqodwU_B1QmXD">最长的时间 - Numberphile</a>：Don Page 的一篇论文声称使用了物理学家计算过的最长有限时间——这是宇宙重置自身所需的时间！？更多信息...</li><li><a href="https://gwern.net/aunn">Absolute Unit NNs：适用于一切的基于回归的 MLPs · Gwern.net</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1334384668706996245)** (2 messages): 

> `DeepSpeed training issues, Intermediate dimension adjustments for gated MLPs, Llama2 config parameters` 


- **删除 torch_extensions 以修复训练问题**：一位用户建议从缓存文件夹中删除 `torch_extensions` 目录，以解决加载模型后导致训练无法开始的问题，参考了 [此 issue](https://github.com/microsoft/DeepSpeed/issues/2816)。
   - 据报道，这个简单的修复方法行之有效，为类似问题提供了潜在的解决方案。
- **在 Gated MLP 中设置中间层维度**：一种配置带有 Gated MLP 模型的理论是将中间层维度（intermediate dimension）设置为期望值的 **3 倍**，然后在导出期间重置它，以避免 Hugging Face 导出时出现问题。
   - 这种变通方法在两个测试模型中都奏效了，尽管用户承认可能还需要进一步检查。
- **Llama2 配置值说明**：用户注意到 Llama2 配置中的 **32768** 这个值没有解释，且不能被 **3** 整除，当应用关于 Gated 配置的考量时，这会导致它调整为 **11008**。
   - 这一见解基于对 [Llama2 config](https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26) 的引用，用户对这一理解持开放态度并欢迎指正。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/microsoft/DeepSpeed/issues/2816">Training stuck after loading the model?  · Issue #2816 · microsoft/DeepSpeed</a>：问题：加载模型后训练未开始。DS_REPORT (base) ext_abdul.waheed@p4-r69-a:~$ nvcc --version nvcc: NVIDIA (R) CUDA compiler driver Copyright (c) 2005-2020 NVIDIA Corporation...</li><li><a href="https://github.com/EleutherAI/gpt-neox/blob/main/configs/llama2/7B.yml#L26)">gpt-neox/configs/llama2/7B.yml at main · EleutherAI/gpt-neox</a>：基于 Megatron 和 DeepSpeed 库在 GPU 上实现模型并行自回归 Transformer - EleutherAI/gpt-neox
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1334323384485417051)** (1 messages): 

> `DeepSeek R1 Distill Qwen 32B, DeepSeek R1 Distill Qwen 14B` 


- **推出 DeepSeek R1 Distill Qwen 32B**：新模型 [DeepSeek R1 Distill Qwen 32B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b) 提供了与体量更大的 **R1 Llama 70b Distill** 类似的轻量级性能，输入和输出价格均为 **$0.7/M**。
   - 感兴趣的用户可以通过 [Discord 频道](https://discord.gg/fVyRaUDgxW) 申请该模型的访问权限。
- **发布 DeepSeek R1 Distill Qwen 14B**：[DeepSeek R1 Distill Qwen 14B](https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b) 现已上线，承诺更小的体积和更快的处理速度，同时在 **AIME 2024 上得分为 69.7**。
   - 该模型输入和输出的价格均为 **$0.75/M**，同样可以通过 [Discord](https://discord.gg/fVyRaUDgxW) 获取。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-32b)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格</li><li><a href="https://openrouter.ai/deepseek/deepseek-r1-distill-qwen-14b>)">OpenRouter</a>：LLM 的统一接口。为您的提示词寻找最佳模型和价格
</li>
</ul>

</div>
  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1334573785751490611)** (6 条消息): 

> `Subconscious AI 的能力、Beamlit 的平台特性、Discord 互动` 


- **Subconscious AI 变革决策过程**：正如其[官网](https://www.subconscious.ai)所述，Subconscious AI 正在通过**先进的 AI 驱动研究**、市场模拟和因果推理建模（causal inference modeling）彻底改变**决策过程**。
   - 他们强调其平台能帮助企业和政策制定者深入洞察消费者行为和市场趋势，并强调其因果模型具有**保证达到人类水平的可靠性**。
- **Beamlit 旨在加速 Generative AI 开发**：[Beamlit](https://beamlit.com) 的联合创始人 Mathis 分享道，他们的平台允许开发者使用类似于 AI Agents 版 Vercel 的简单命令行界面，将 **AI agents 的交付速度**提升高达 **10 倍**。
   - 他们推出了**免费公开测试版 (free public alpha version)**，邀请用户提供反馈并探索集成 GitHub 工作流和可观测性工具等功能。
- **Discord 社区互动**：一名成员对 Subconscious AI 表示了兴趣，并加入了其 **Discord** 以获取更多信息。
   - 这突显了当前以社区为导向的对话趋势，旨在促进新兴 AI 技术与潜在用户之间建立更深层次的联系。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.subconscious.ai">subconscious.ai</a>: 未找到描述</li><li><a href="https://beamlit.com:">Beamlit</a>: 未找到描述</li><li><a href="https://docs.beamlit.com/Get-started#quickstart">未找到标题</a>: 未找到描述
</li>
</ul>

</div>

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1334257353628979240)** (180 条消息🔥🔥): 

> `OpenRouter 价格担忧、DeepSeek R1 模型限制、Google AI Studio 速率限制、供应商问题与停机、新模型发布公告` 


- **OpenRouter 价格引发辩论**：用户质疑为什么 OpenRouter 对转发 API 请求收取 **5%** 的费用，有人认为相对于提供的服务，这个比例过高。
   - 另一位用户打趣道：*“你得去跟 Stripe 谈谈”*，暗示这可能涉及潜在的底层费用。
- **DeepSeek R1 的上下文窗口生成问题**：多位用户报告了 **DeepSeek R1** 模型的问题，包括由于超出上下文限制导致生成超时，从而无法检索响应。
   - 一位用户确认，若要查看模型的推理过程，需要在 API 请求中传递 `include_reasoning` 参数。
- **Google AI Studio 频繁出现速率限制错误**：用户在 Google AI Studio 中查询 **Gemini** 模型时遇到了 **429 RESOURCE_EXHAUSTED** 错误，表明配额已耗尽。
   - 速率限制由 Google 强制执行，建议用户接入自己的 Key 以提高吞吐量。
- **供应商状态波动与停机**：一些用户报告 OpenRouter 的 API 持续出现 **404 错误**，特别是在尝试访问 chat completions 端点时。
   - 停机归因于不同供应商的容量波动，**Nebius** 和 **Avian** 因服务不稳定被点名。
- **即将发布的 AI 模型引发关注**：用户讨论了关于新 AI 模型的公告，如 **Mistral Small 3** 和 **Tülu 3**，展示了在各种能力上的性能提升。
   - 社区热切期待将这些新模型集成到 OpenRouter 中，因为它们展现了强大的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/allen_ai/status/1884966600039915809">来自 Ai2 (@allen_ai) 的推文</a>：这是 Tülu 3 405B 🐫，我们的开源后训练模型，其性能超越了 DeepSeek-V3！Tülu 3 家族的最后一名成员证明了我们的方案（包括强化学习...）</li><li><a href="https://x.com/MistralAI/status/1884968836606136636">来自 Mistral AI (@MistralAI) 的推文</a>：隆重推出 Small 3，我们迄今为止最高效、最通用的模型！预训练和指令版本，Apache 2.0 协议，24B 参数，81% MMLU，150 tok/s。无合成数据，是任何推理任务的绝佳基础...</li><li><a href="https://openrouter.ai/docs/">快速入门 | OpenRouter</a>：开始使用 OpenRouter 进行构建</li><li><a href="https://openrouter.ai/docs/quick-start">快速入门 | OpenRouter</a>：开始使用 OpenRouter 进行构建</li><li><a href="https://x.com/risphereeditor/status/1885041914191192573">来自 Risphere (@risphereeditor) 的推文</a>：Fireworks AI 现在是美国最快的 DeepSeek 供应商。DeepSeek-V3 和 DeepSeek-R1 现在以每秒 30 个 token 的速度运行。祝贺 @FireworksAI_HQ 团队！</li><li><a href="https://openrouter.ai/docs/integrations#automatic-fallback">集成 | OpenRouter</a>：在 OpenRouter 中使用您自己的供应商 Key</li><li><a href="https://openrouter.ai/docs/provider-routing">供应商路由 | OpenRouter</a>：跨多个供应商路由请求</li><li><a href="https://www.reddit.com/r/bing/comments/110eagl/the_customer_s">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://openrouter.ai/docs/parameters#include-reasoning">参数 | OpenRouter</a>：为请求配置参数</li><li><a href="https://openrouter.ai/google/gemini-flash-1.5-8b">Gemini Flash 1.5 8B - API、供应商、统计数据</a>：Gemini Flash 1.5 8B 针对速度和效率进行了优化，在聊天、转录和翻译等短提示任务中提供增强的性能。通过 API 运行 Gemini Flash 1.5 8B</li><li><a href="https://www.reddit.com/r/bing/comments/110eagl/the_customer_service_of_the_new_bing_chat_is/#lightbox)">Reddit - 深入探索</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[announcements](https://discord.com/channels/364486390102097930/671536649301131325/1334579954968563772)** (1 条消息): 

> `Bolt 二进制资产生成，Token 节省，外部资产利用` 


- **Bolt 停止生成二进制资产**：Bolt 现在避免生成二进制资产，从而显著**节省 Token 和时间**，并增强输出质量。
   - 这一变化实现了更高效的处理并提升了整体性能。
- **实现显著的 Token 节省**：Bolt 的最新改进通过优化资产利用方式，节省了**数十万个 Token**。
   - 这一增强使操作速度提升了**几个数量级**，简化了整个流程并改善了用户体验。
- **利用外部资产**：Bolt 的 Agent 现在使用外部资产，而不是从头开始创建，从而实现了更高效的 Token 使用。
   - 成员们对这一战略转变表示兴奋，它提高了操作速度和结果质量。



**提到的链接**：<a href="https://x.com/boltdotnew/status/1885019780840653183">来自 bolt.new (@boltdotnew) 的推文</a>：更多的 Token 节省已上线！Bolt 的 Agent 现在利用外部资产，而不是让 LLM 从头开始创建新资产。这节省了数十万个 Token——并且速度提升了几个数量级...

  

---


### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1334324876600676353)** (10 条消息🔥): 

> `Bolt 的末尾补零问题，文件更新偷懒，雇主注册表单更新，System Prompt 的社区用例，Supabase 注册错误排查` 


- **Bolt 在处理末尾补零时遇到困难**：用户在尝试为 CBTO 输入诸如 **2.7980** 之类的数字时表示沮丧，因为 Bolt 会对其进行错误的自动格式化。尽管用户要求 Bolt 准确显示输入的数据，但它未能做到。
   - 一位成员在分享上下文图片的同时，寻求管理这种自动格式化困扰的技巧。
- **需要修复文件更新“偷懒”问题**：用户对更新期间需要保持文件其余部分不变的持续问题表示担忧。一位成员遇到了反复出现的语法错误，表明需要解决执行中的“偷懒”行为。
   - 一位用户评论了更新后的改进，指出虽然一些“偷懒”问题有所减轻，但仍有一段路要走。
- **更新雇主注册表单**：一位成员发现雇主注册表单中缺少 **First Name** 和 **Last Name** 字段。他们强调了正确数据映射的重要性，以确保顺利集成到用户 Profile 中。
   - 解决这一差距的建议包括确认匹配的文件名和视图，特别是在进行多次更新时。
- **探索 System Prompt 的社区用例**：用户表示有兴趣了解社区如何利用新的 **Project and Global System Prompt**。目前，一位成员将其用于让 Bolt 更新 Changelog，但也渴望听到其他启发性的应用。
   - 另一位成员建议在排查问题时分享特定文件并确保视图正确，以产生富有成效的结果。
- **排查 Supabase 注册错误**：强调了围绕 **Supabase 请求失败**的持续问题，一位用户在注册时遇到了 **500** 状态错误。他们建议创建一个专门的排查小组，以促进对特定应用程序错误的讨论。
   - 一位成员建议利用 AI 工具，通过分享错误详情、代码和相关截图来获取建议，从而更有效地解决问题。

  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1334263240020070420)** (170 条消息🔥🔥): 

> `Supabase 集成问题，Token 使用疑虑，Fork 项目的挑战，Supabase functions 的 CORS 问题，React 中的 SEO 元数据处理` 


- **项目 Fork 后的 Supabase 集成问题**：用户在将 Fork 的项目连接到 Supabase 时遇到问题，由于 .env 文件未随之复制，导致出现错误且用户仪表板不可用。
   - 参与者指出，在问题解决之前，建议在开发过程中使用 local storage 处理数据，以避免消耗 Token。
- **Token 使用和订阅困惑**：关于 Token 是每日重置还是每月重置，以及未使用的 Token 如何管理存在困惑。用户澄清称，按月订阅的未使用 Token 不会结转。
   - 几位用户对高 Token 消耗率表示担忧，特别是在遇到需要多次 Prompt 才能解决的问题时。
- **Fork 项目的挑战**：用户在 Fork 项目后重新建立 Supabase 连接时面临困难，建议手动复制 .env 文件以实现正确集成。
   - 建议创建 GitHub issues 来跟踪已知问题，并强调了妥善处理项目备份的重要性。
- **调用 Supabase functions 时的 CORS 问题**：一位用户报告在尝试从前端应用调用 Supabase function 时遇到 CORS 错误，阻碍了开发进度。
   - 参与者建议，API 调用应通过带有 Relay Request 的 Node 后端或通过 Edge function 进行，以避免此类问题。
- **React 应用中的 SEO 元数据处理**：一位用户正在寻求关于如何在 React 应用中为不同页面实现服务端 SEO 元数据的建议，并指出通常的方法并不奏效。
   - 讨论中提到了使用替代方案，因为默认的 helmet 方法似乎无法为社交媒体分享获取正确的元数据。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://zp1v56uxy8rdx5ypatb0ockcb9tr6a-oci3--5173--5ab5ceac.local-credentialless.webcontainer-api.io'">未找到标题</a>: 未找到描述</li><li><a href="https://tenor.com/view/frozen-freezing-cold-shivering-glace-gif-5390133">Frozen Freezing GIF - Frozen Freezing Cold - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://boltsync.mystify.tech/">BoltSync - 使用 Bolt 进行 GitHub 仓库管理</a>: 使用 Bolt Prompts 修改你的 GitHub 仓库，并通过 BoltSync 将更改同步回 GitHub。通过 AI 驱动的仓库管理简化你的开发工作流。</li><li><a href="https://x.com/boltdotnew/status/1843668731681267801">来自 bolt.new (@boltdotnew) 的推文</a>: 你现在可以在 bolt.new 中打开公共仓库了 🙌 怎么做？对于任何 GitHub URL，只需在前面加上 "http://bolt.new" 即可！（下方有发布说明！）</li><li><a href="https://github.com/stackblitz/bolt.new/issues">stackblitz/bolt.new</a>: Prompt、运行、编辑和部署全栈 Web 应用程序 - stackblitz/bolt.new</li><li><a href="https://showmeyourbolt.io/">Show Me Your Bolt</a>: 未找到描述</li><li><a href="https://boltnew.dev/apps">Bolt.new 构建者中心</a>: 未找到描述</li><li><a href="https://imbuiltwithai.com/">分享你的 AI 项目 - I'm Built With AI</a>: 未找到描述</li><li><a href="https://youtu.be/tlu5e0TxSzo?si=cCaDQFroJ8_1MNwT&t=77">如何上传文件和文件夹到 GitHub：GitHub 初学者指南</a>: 将项目上传到 GitHub 并不复杂。在本视频中，我们将向你展示两种简单的方法，将你的文件和文件夹放入 GitHub 仓库...</li><li><a href="http://bolt.new/github.com/strapi/strapi">bolt.new</a>: 未找到描述</li><li><a href="https://lucide.dev/icons/">Lucide Icons</a>: 由社区制作的美观且一致的图标工具包。
</li>
</ul>

</div>
  

---

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1334253510270779493)** (178 messages🔥🔥): 

> `ComfyUI Performance and Features, Hardware Discussions for AI Workloads, Reactor Tool for Face Swapping, Stable Diffusion Lora Training, Availability of New GPUs` 


- **ComfyUI 的局部重绘（Inpainting）手动控制**：用户讨论了 ComfyUI 中局部重绘和 ControlNet 集成涉及的手动流程，强调了特定调整所需的灵活性。
   - 一位用户表示，他们更倾向于手动控制，以便更好地利用模型的能力，而不是仅仅依赖自动化方法。
- **Stable Diffusion 的硬件规格**：对话围绕有效运行 Stable Diffusion 的 GPU 规格展开，用户分享了关于 3080 和 3090 等各种型号能力的经验。
   - 一位用户讨论了使用 Intel Arc A770 LE 的经验，以及它在游戏和 AI 任务中与 3060/3060TI 相当的性能。
- **Reactor 工具的移除与替代方案**：一位用户询问了 Reactor 工具被移除的情况，指出该工具因缺乏 NSFW 过滤器而被下架，不过后来在添加了安全防护措施后重新上传。
   - 分享了更新版 Reactor 的链接，该版本已面向 auto1111 和 ComfyUI 用户提供，支持换脸（Face Swap）功能。
- **为 Stable Diffusion 训练 Lora**：用户讨论了为 Stable Diffusion 训练 Lora 的过程，以及在确保特定特征匹配的同时集成风格的重要性。
   - 一位用户寻求关于结合特定面部和风格参考的工作流的澄清，并强调了他们最近遇到的挑战。
- **新款 GPU 的可用性**：新款 5090 GPU 的迅速售罄引发了围绕市场需求和可用性的讨论，一些用户对供应有限表示失望。
   - 对话还包括对购买技术产品的融资选项的看法，以及对难以获取新硬件的普遍市场挫败感。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://streamable.com/d3ww4l">观看 inpaint3432342 (online-video-cutter.com) | Streamable</a>：未找到描述</li><li><a href="https://tenor.com/view/wall-gif-24534315">Wall GIF - Wall - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://streamable.com/5edusn">观看 inpaint 23432432 (online-video-cutter.com) | Streamable</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=pvxUHpf1pxQ">海滨咖啡馆空间的甜美爵士乐与海洋之声 ~ 提升情绪的正能量 Bossa Nova 音乐</a>：海滨咖啡馆空间的甜美爵士乐与海洋之声 ~ 提升情绪的正能量 Bossa Nova 音乐，逃离到我们最新的梦幻海岸度假胜地...</li><li><a href="https://github.com/Gourieff/sd-webui-reactor-sfw">GitHub - Gourieff/sd-webui-reactor-sfw: (SFW 友好型) 适用于 StableDiffusion WebUI (A1111, SD.Next, Cagliostro) 的快速简单换脸扩展</a>：(SFW 友好型) 适用于 StableDiffusion WebUI (A1111, SD.Next, Cagliostro) 的快速简单换脸扩展 - Gourieff/sd-webui-reactor-sfw</li><li><a href="https://github.com/Gourieff/comfyui-reactor">GitHub - Gourieff/ComfyUI-ReActor: 适用于 ComfyUI 的快速简单换脸扩展节点 (SFW)</a>：适用于 ComfyUI 的快速简单换脸扩展节点 (SFW) - Gourieff/ComfyUI-ReActor
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1334373376462032956)** (1 messages): 

> `Decompression Time, Loading Weights from Disk` 


- **关于解压与直接加载的好奇**：一位成员表示有兴趣了解解压（Decompression）过程与直接从磁盘加载相比需要多少时间。
   - 他们质疑直接从磁盘加载是否比解压方法更有效率。
- **性能对比咨询**：该成员的询问还表明需要评估直接加载权重（Weights）与解压过程之间的性能差异。
   - 这突显了人们对优化模型加载时间以提高效率的广泛兴趣。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1334633334818476182)** (1 messages): 

> `Triton Tensor Indexing, Using tl.gather, InterpreterError` 


- **Triton 无法对张量列进行索引**：用户尝试使用 `x[:, 0]` 从张量中提取单列，但遇到了 **InterpreterError**，提示 `unsupported tensor index: 0`。
   - *这突显了 Triton 在张量索引能力方面的局限性。*
- **使用 tl.gather 的效率担忧**：用户考虑使用 `tl.gather` 并将索引张量设置为全零，作为提取该列的变通方法。
   - 然而，他们对这种方法与直接索引相比的效率表示担忧。


  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1334300650166550629)** (18 messages🔥): 

> `Blackwell architecture features, sm_X features compatibility, Performance comparisons: RTX 5090 vs RTX 4090, PTX ISA documentation, Tensor Operations discussion` 


- **Blackwell 的 sm_120a 特性说明**：成员们讨论了在 **sm_120a** 等架构中的 `a` 标识表示这些特性在未来不会获得支持，这对于那些既需要前向兼容性又需要特定特性的人来说至关重要。
   - **sm_90a** 是第一个引入这种区分的架构，现在 **Blackwell** 在消费级平台中也出现了这种情况。
- **sm_X 架构兼容性**：据指出，**sm_120** 意味着比 **sm_100** 更强的计算能力，但 'a' 变体可能会在未来的支持中省略某些特性。
   - 架构讨论深入探讨了 **sm_90a** 与其他迭代版本之间的差异，这些迭代并不保证是特性的超集。
- **RTX 5090 与 RTX 4090 的性能对比**：一位成员对性能差距提出疑问，指出 **RTX 5090** 上的 **FP4 with FP32** 比 **RTX 4090** 上的 **FP8** 快约 **5x**，但某些其他基准测试显示仅有 **2x** 的优势。
   - 有人对 NVIDIA 文档中关于性能声明的潜在准确性表示担忧，并指出了过去的差异。
- **PTX ISA 的重要资源**：讨论强调了 **PTX ISA documentation** 是一个宝贵的资源，特别是对于理解 **sm_100a** 和 **sm_101a** 等特定架构特性。
   - 成员们指出，该文档提供了关于指令和架构能力的至关重要的见解。
- **Tensor Operations 与 RTX 架构**：成员们讨论了某些张量指令的缺失，指出 **Blackwell** 引入了以前架构（如 **RTX 5 series**）所不具备的张量功能。
   - 特别是张量内存和 **tcgen05** 等操作的创新被强调为最新架构中的重大进步。



**提到的链接**：<a href="https://github.com/NVIDIA/cutlass/blob/main/media/docs/blackwell_functionality.md">cutlass/media/docs/blackwell_functionality.md at main · NVIDIA/cutlass</a>：用于线性代数子程序的 CUDA 模板。通过在 GitHub 上创建账户为 NVIDIA/cutlass 的开发做出贡献。

  

---

### **GPU MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1334312304283746424)** (2 messages): 

> `PyTorch 2.6 release, FP16 support on X86, Deprecating Conda, Manylinux 2.28 build platform` 


- **PyTorch 2.6 发布，带来令人兴奋的新特性**：我们很高兴地宣布 **PyTorch® 2.6** 正式发布，其增强功能包括 `torch.compile` 对 **Python 3.13** 的兼容性，以及新的性能调节参数 `torch.compiler.set_stance`。
   - 此版本还包含 **X86 CPU 上的 FP16 支持**，增强了性能敏感型应用的能力。
- **宣布停止支持 Conda**：随着 **PyTorch 2.6** 的发布，官方决定停止在 **Conda** 上发布更新；详情请参阅 [弃用公告](https://github.com/pytorch/pytorch/issues/138506)。
   - 建议用户转向其他安装方法，这标志着分发策略的转变。
- **PyTorch 采用新的构建平台**：此版本中的实验性 Linux 二进制文件附带 **CUDA 12.6.3**，并使用 **Manylinux 2.28 构建平台**，确保了跨不同系统的兼容性。
   - 对于有兴趣从源码构建的用户，二进制文件配置了 **CXX11_ABI=1**，以便更好地进行集成。
- **社区对 PyTorch 2.6 的热情**：社区成员对 **PyTorch 2.6** 的新特性表示了极大的热情，一位用户表示他们“对此感到非常兴奋！！！”。
   - 这种兴奋反映了用户对新版本将为他们的工作流带来的能力的强烈期待。



**提及的链接**：<a href="https://pytorch.org/blog/pytorch2-6/">PyTorch 2.6 发布博客</a>：我们很高兴宣布 PyTorch® 2.6（发布说明）发布！此版本为 PT2 带来了多项改进：torch.compile 现在可以与 Python 3.13 一起使用；新的性能相关参数...

  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1334567811585933393)** (1 messages): 

> `GPU Kernel Engineers, GPU Compiler Engineers, Next Gen ML Compiler, Job Openings` 


- **招聘 GPU Kernel 和编译器工程师**：我们正在寻求 **GPU kernel** 和 **GPU 编译器工程师**，提供**优厚的薪酬**和**股权激励**。
   - 该项目旨在构建一个将 **AI 集成到编译流程**中的**下一代 ML 编译器**，并得到了业界知名人士的支持。
- **AI 编译领域的绝佳机会**：团队正在寻找在 **Triton**、**CUDA** 和 **HIP** 方面具有专业知识的人才，共同为 ML 应用设计尖端解决方案。
   - 欲了解更多详情，请访问 [Mako Dev](https://jobs.mako-dev.com/GPU-Kernel-Engineer-144546aebc36805f9ba3f0b27aafa492) 的职位发布，但请注意该网站正在更新中。



**提及的链接**：<a href="https://jobs.mako-dev.com/GPU-Kernel-Engineer-144546aebc36805f9ba3f0b27aafa492">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一个将日常工作应用融合在一起的新工具。它是为您和您的团队提供的一体化工作空间。

  

---


### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1334417723081490485)** (6 messages): 

> `C++ versions, CUDA compatibility` 


- **选择合适的 C++ 版本**：大多数讨论表明 **C++20** 是开发的良好起点，特别是当使用像 **Cutlass** 和 **Thundekittens** 这样可能需要更新标准的库时。
   - 然而，一位成员提到在针对 **Windows** 开发时使用 **C++17**，而另一位成员表示在 **Linux** 上更倾向于使用带有反射（reflection）特性的 **C++26**。
- **C++20 与 CUDA 的潜在问题**：有人担心，如果你需要旧版本的 **CUDA**（例如仍与 **PyTorch** 兼容的 **CUDA 11.8**），使用 **C++20** 可能会导致复杂化。
   - 这强调了将 C++ 版本与你打算使用的库和框架保持一致的重要性。


  

---

### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1334451783241826386)** (7 messages): 

> `RTX 5090 可用性、自制餐食、特色餐盘` 


- **关于 RTX 5090 在欧洲销售的咨询**：一名成员询问了 **RTX 5090** 的可用性，并指出 NVIDIA 官网显示欧洲地区尚未开始销售。
   - *德国或其他欧洲国家的成员买到了吗？*
- **美味餐食描述**：一位成员分享了自制餐食的详细描述，包括**三文鱼饼、炸土豆**以及配有希腊酸奶的**自制华夫饼**。
   - 该帖子还附带了一张图片，引发了关于其视觉上酷似**鸡蛋**的讨论。
- **对餐食的视觉误解**：针对餐食描述，一位成员幽默地指出，照片中的食物乍一看像一个**巨大的鸡蛋**。
   - 另一位成员表示同意，认为罐装桃子让它看起来更像鸡蛋了。
- **关于特色餐盘的讨论**：一位成员在餐食讨论中提到了一款**特色餐盘**，特别是提到了 **Tuberculosis Sanatorium 96**（96 号结核病疗养院）。
   - 原贴作者确认了该餐盘的趣味性质，为餐食展示增添了迷人的色彩。


  

---


### **GPU MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/1334371446679670825)** (1 messages): 

> `线下活动、Discord 频道更新` 


- **计划举办更多线下活动**：一位成员表达了今年为服务器举办**更多线下活动**的意向，并将在本频道提供更新。
   - 他们强调了通过这些活动促进**社区参与**的承诺。
- **Discord 频道通知**：一位成员分享了一个 Discord 消息链接，指出遗漏了关于其中一个频道的**通知**。
   - 这显示了跟踪频道更新和讨论的重要性。


  

---


### **GPU MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1334566904433737808)** (4 messages): 

> `ROOK 博客文章、进度更新、WoW 模组项目` 


- **ROOK：国际象棋 AI 的新视野**：一篇新博客文章介绍了 **ROOK (Reasoning Over Organized Knowledge)**，这是一套旨在解决国际象棋战略推理问题的语言模型，超越了传统的搜索算法。
   - 该项目包含三个 Transformer 模型：**ROOK-CLF**、**ROOK-LM** 和 **RookWorld-LM**，旨在以类人的方式捕捉国际象棋知识。[点击此处阅读完整详情](https://laion.ai/notes/rook/)。
- **久违的用户问候**：一位成员通过询问对方目前的近况或是否在休息，表达了与另一位成员重新取得联系的兴奋之情。
   - 另一位成员幽默地回应了自上次联络以来度过的时间，并接受了这种轻松的调侃。
- **WoW 模组项目**：提到一位成员可能参与了《魔兽世界》（WoW）的 **Modding 项目**，突显了他们的创造性尝试。
   - 社区似乎非常钦佩该成员在游戏领域的投入和才华。



**提到的链接**：<a href="https://laion.ai/notes/rook/">ROOK: Reasoning Over Organized Knowledge | LAION</a>: &lt;p&gt;人工智能领域长期以来一直将战略推理任务作为衡量和推进 AI 能力的基准。国际象棋凭借其...

  

---


### **GPU MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1334305197161185383)** (1 messages): 

> `实现新的内核语言、低精度内核、学习的未来收益` 


- **质疑内核语言的实现时间**：一位成员询问现在实现新的内核语言（Kernel Languages）是否浪费时间。
   - 他们指出，为低精度内核学习一种新的内核语言具有**显而易见的未来收益**。
- **探索低精度内核的用途**：讨论强调了低精度内核在减少计算开销和提高效率方面的重要性。
   - 一位参与者强调，采用新的内核语言可以在特定应用中带来**性能提升**。


  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1334298082497658880)** (15 messages🔥): 

> `Mistral AIx Game Jam 结果, Parental Control 游戏, 语音命令功能, Flash Attention 实现, Llama3-8B R1 模型改进` 


- **Mistral AIx Game Jam 荣获铜奖！**：该团队在 **Mistral AIx 🤗 Game Jam 中获得第 2 名**，并力争赢得社区奖（Community Award）。他们鼓励大家[尝试这款游戏](https://huggingface.co/spaces/Mistral-AI-Game-Jam/ParentalControl)并提供反馈。
   - 该游戏结合了 **AI** 与 **游戏开发**，使用 Mistral AI、Godot 等工具打造了引人入胜的体验。
- **游戏项目融入恐怖元素**：游戏强调生存感，要求玩家在视频通话期间管理混乱的环境并确保婴儿的安全。玩家可以使用 **语音命令** 与婴儿互动，从而产生有趣的结果。
   - 开发者选择了 **恐怖氛围** 来反映育儿的压力，引发了玩家的幽默感和参与度。
- **CUDA 实现 Flash Attention**：一位用户在 [GitHub](https://github.com/akshat-sj/flashattention) 上分享了他们的第一个 CUDA 项目，展示了用原生 CUDA 实现的 **Flash Attention**。他们希望社区能对其工作提供反馈。
   - 该项目包含一张截图以及关于参与 Flash Attention 开发的详细信息，展示了他们在 CUDA 编程方面的进展和学习成果。
- **优化的 Llama3-8B 模型发布**：团队发布了一个新的 **Llama3-8B R1** 重蒸馏模型，详细介绍了其高性价比的方法，在 GSM8K 基准测试中实现了高达 **14% 的性能提升**。该模型已在 Hugging Face 上线，支持通过 HQQ 进行高效运行。
   - 他们的公告包含了一篇 [博客文章](https://mobiusml.github.io/r1_redistill_blogpost/) 的链接，讨论了他们在每次训练运行仅花费 3 到 18 美元的情况下取得成功的细节。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Mobius_Labs/status/1885010791344062704">Mobius Labs (@Mobius_Labs) 的推文</a>：我们利用 Logits 蒸馏改进了 DeepSeek R1 蒸馏模型——在 GSM8K 上实现了 +4-14% 的提升，而每次训练运行仅花费 $3-18！🚀 现已在 Hugging Face 上线——高效运行它们...</li><li><a href="https://github.com/akshat-sj/flashattention">GitHub - akshat-sj/flashattention: 原生 CUDA 实现的 Flash Attention</a>：原生 CUDA 实现的 Flash Attention。通过在 GitHub 上创建账号来为 akshat-sj/flashattention 的开发做出贡献。</li><li><a href="https://x.com/amtellezfdez)">GitHub - FixTweet/FxTwitter 的推文</a>：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - FixTweet/FxTwitter
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1334366946690601025)** (3 messages): 

> `CUDA 版本与 TK 内核, Nvidia P100 GPU 支持` 


- **CUDA 版本显著影响 TK 内核性能**：一位成员指出，他们在 **CUDA 12.4** 下测试 Flash Attention Hopper 仅获得 **550 TFLOPS**，显著低于预期的 **600 TFLOPS**，这引发了对不同 CUDA 版本性能差异的担忧。
   - 他们询问这是否正常，并提到 **CUDNN SDPA** 在类似设置下可以达到 **590 TFLOPS**。
- **为 Nvidia P100 GPU 提供入门支持**：一位用户表示有兴趣通过添加对 **Nvidia P100 GPU** 的支持来为 **Thunderkittens** 做出贡献，旨在实现在 Google Colab 上的使用。
   - 他们邀请其他成员通过私信（DM）联系以获取适配协助。


  

---


### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1334258466054869137)** (87 messages🔥🔥): 

> `Reasoning Gym 数据集, 生命游戏（Game of Life）挑战, 协作问题解决, Codenames 游戏机制, 谋杀之谜环境`

- **Reasoning Gym 中可用数据集的增加**：Reasoning Gym 现在拥有 **33** 个数据集，并在 [GITHUB 仓库](https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md) 中建立了一个新的简单数据集展示廊。这标志着在为强化学习提供多样化推理挑战方面取得了重大进展。
   - 鼓励贡献者提交新的数据集和想法，以进一步扩大平台的范围。
- **提议用于推理任务的交互式游戏**：围绕扩展 RL 环境的讨论包括了 **协作问题解决** 和 **多 Agent 协商** 任务的想法，允许需要 LLM 交互的复杂场景。建议的场景（如 **团队协作编程**）旨在促进多个 Agent 之间的协调。
   - 这些建议旨在增强 Reasoning Gym 的能力，引入需要更深层次推理和社交互动的多方面挑战。
- **创新的 Game of Life 推理挑战**：提议了一项涉及 Conway 的 **Game of Life** 的新挑战，模型需要预测初始随机配置的演变。这项任务的灵感来自于利用 LLM 进行 **解释性推理** 挑战的想法。
   - 该挑战包括根据定义的规则确定给定的棋盘设置是否会导致停止或非停止状态。
- **将 Codenames 机制整合到推理任务中**：讨论了将 **Codenames** 游戏作为一个潜在任务，LLM 根据选定的单词提供提示，以制定其响应策略。这可以突出模型如何利用共享的认知关联在双方进行操作。
   - 讨论反映了利用现有游戏创建引人入胜且有意义的推理环境的持续努力。
- **谋杀之谜作为多轮环境**：考虑实现一个 **谋杀之谜** 环境，允许在不需要地下城主的情况下进行交互。这种设置侧重于基于逻辑的排除，并可能导致对多轮 Agent 交互的进一步探索。
   - 动态交互框架的潜在使用可以极大地增强此类游戏中的问题解决场景。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://en.wikipedia.org/wiki/Category:Logic_puzzles">Category:Logic puzzles - 维基百科</a>：未找到描述</li><li><a href="https://x.com/karpathy/status/1884676486713737258">来自 Andrej Karpathy (@karpathy) 的推文</a>：致开源界的朋友们：我认为你们能做的杠杆效应最高的事情，就是帮助构建高度多样化的 RL 环境，以帮助诱导 LLM 的认知策略。建立一个类似 gym 的东西。这是 ...</li><li><a href="https://github.com/open-thought/reasoning-gym/issues/24">添加 WordLadder 游戏数据集 (word golf) · Issue #24 · open-thought/reasoning-gym</a>：创建一个带有配置和单元测试的单词梯（word ladder）游戏数据集类。在问题中包含简单示例以定义响应格式。GitHub 上有很多单词梯的实现，例如...</li><li><a href="https://en.wikipedia.org/wiki/Recreational_mathematics">娱乐数学 - 维基百科</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/List_of_recreational_number_theory_topics">娱乐数论话题列表 - 维基百科</a>：未找到描述</li><li><a href="https://vimeo.com/1022776731?autoplay=1&muted=1&stream_id=Y2xpcHN8MjI4Mjc5MjI0fGlkOmRlc2N8eyJyZW1vdmVfdm9kX3RpdGxlcyI6ZmFsc2V9">OARC - C3PO demo 1</a>：这是 Leonardo Borch 在 Vimeo 上发布的 "OARC - C3PO demo 1"，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://vimeo.com/1036385433?autoplay=1&muted=1&stream_id=Y2xpcHN8MjI4Mjc5MjI0fGlkOmRlc2N8eyJyZW1vdmVfdm9kX3RpdGxlcyI6ZmFsc2V9">聊天中的 OARC 贪吃蛇渲染，html</a>：这是 Leonardo Borch 在 Vimeo 上发布的 "OARC in chat snake render, html"，Vimeo 是高质量视频及其爱好者的家园。</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/38d64649f525363f4525db02d27a993ed8fbd72b/reasoning_gym/cognition/rubiks_cube.py#L38">reasoning-gym/reasoning_gym/cognition/rubiks_cube.py · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建账号来为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/23">由 Miserlou 添加 “量子锁” 谜题 · Pull Request #23 · open-thought/reasoning-gym</a>：在你面前有一些按钮、一盏灯和一个数字。每当你按下按钮时，灯会在红色和绿色之间切换。每个按钮都会对数字执行数学运算，但是...</li><li><a href="https://youtu.be/JheGL6uSF-4?si=EE1aKCt4C3MiYxXM">我制作了一个维基百科图谱... 这是我的发现</a>：我所有视频的代码：https://github.com/sponsors/adumb-codes/ 获取图谱海报：https://adumb.store/ Twitter：https://twitter.com/adumb_codes 一次深度的...</li><li><a href="https://github.com/NousResearch/Open-Reasoning-Tasks">GitHub - NousResearch/Open-Reasoning-Tasks：一个面向 LLM（及更多领域）的推理任务综合仓库</a>：一个面向 LLM（及更多领域）的推理任务综合仓库 - NousResearch/Open-Reasoning-Tasks</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/28">由 Miserlou 添加 BF (Brainf*ck) 挑战 · Pull Request #28 · open-thought/reasoning-gym</a>：添加了第一个“代码”挑战：BF！这是一个代码执行挑战，而不是代码生成挑战。共有三个难度级别。级别 1 是简单的打印字符串。级别 2 使用...</li><li><a href="https://github.com/ironman5366/ai-murder-mystery-hackathon">GitHub - ironman5366/ai-murder-mystery-hackathon：游戏开始了</a>：游戏开始了。通过在 GitHub 上创建账号来为 ironman5366/ai-murder-mystery-hackathon 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/mlabonne/agentic-datagen">Agentic 数据生成的兴起</a>：未找到描述</li><li><a href="https://github.com/open-thought/reasoning-gym/blob/main/GALLERY.md">reasoning-gym/GALLERY.md · open-thought/reasoning-gym</a>：程序化推理数据集。通过在 GitHub 上创建账号来为 open-thought/reasoning-gym 的开发做出贡献。</li><li><a href="https://github.com/open-thought/reasoning-gym/pull/30">由 Miserlou 添加康威生命游戏模拟 · Pull Request #30 · open-thought/reasoning-gym</a>：（基于 BF 分支开发，抱歉。如果需要可以进行 cherry pick！）以可配置的方式添加了康威生命游戏：`config = GameOfLifeConfig(seed=42, size=1, ...`</li><li><a href="https://github.com/Leoleojames1/Agent_Chef">GitHub - Leoleojames1/Agent_Chef：🍲Agent Chef🥘 是我用于数据集精炼、结构化和生成的强大工具。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和清理其微调数据，消除数据污染和低质量知识库。此外，它还将提供模板和框架。</a>：🍲Agent Chef🥘 是我的强大工具</li>

用于数据集的精炼、结构化和生成。通过利用程序化和合成数据集生成技术，Agent Chef 将使用户能够精炼和....</li><li><a href="https://huggingface.co/datasets/Borcherding/OARC_Commander_v001">Borcherding/OARC_Commander_v001 · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1334303351793520751)** (74 条消息🔥🔥): 

> `DeepSeek models, Running models with GPT4All, Integrating Ollama with GPT4All, Local document management, AI education tools` 


- **DeepSeek 模型性能**：成员们讨论了即将发布的 DeepSeek，表达了对其在数学和 LaTeX 支持方面性能的期待。
   - 一些人指出，由于 VRAM 限制，使用 DeepSeek 处理复杂任务可能需要有效地管理 context size。
- **GPT4All 与 Ollama 的集成**：用户确认可以通过将 Ollama 作为服务器运行并在 GPT4All 中使用 OpenAI API 来连接两者。
   - 有关于此集成文档的咨询，一些成员成功找到了相关资源。
- **在 GPT4All 中加载远程 LLM**：讨论包括了如何将远程 LLM 加载到 GPT4All GUI 的步骤，并建议确保 API keys 的正确设置。
   - 成员们提议提供更清晰的文档，以帮助新用户有效地访问远程模型。
- **开发 AI 教育工具**：一位成员分享了他们为非洲儿童构建 AI 驱动教育工具的倡议，强调了离线访问和本地化内容。
   - 他们计划使用轻量级 AI 模型和一系列精选资源，以便在无需互联网连接的情况下促进自主学习。
- **模型量化差异**：一位成员寻求关于模型命名规范的澄清，特别是名称中带有和不带有 '-I1-' 的模型之间的区别。
   - 目前尚未找到明确答案，这表明需要更好的透明度或关于模型规格的文档。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.gpt4all.io/gpt4all_api_server/home.html:">GPT4All</a>: GPT4All 文档 - 在您的硬件上高效运行 LLM</li><li><a href="https://hastebin.skyra.pw/tucewahaca.kotlin">Hastebin</a>: 未找到描述</li><li><a href="https://emmanuelsibanda.hashnode.dev/funda-ai-building-a-laptop-powered-by-ai-to-help-students-in-africa-learn">Funda AI - building a laptop powered by AI to help students in Africa learn</a>: FundAI 提供 AI 驱动的笔记本电脑，帮助非洲学生学习，专注于考试、逻辑思维和技术技能，且不依赖互联网</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Local-API-Server">Local API Server</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可商用。 - nomic-ai/gpt4all</li><li><a href="https://github.com/nomic-ai/gpt4all/wiki/Frequently-Asked-Que">Home</a>: GPT4All: 在任何设备上运行本地 LLM。开源且可商用。 - nomic-ai/gpt4all</li><li><a href="https://lmstudio.ai/">LM Studio - Discover, download, and run local LLMs</a>: 在本地计算机上运行 Llama, Mistral, Phi-3。</li><li><a href="https://anythingllm.com/desktop">Download AnythingLLM for Desktop</a>: 下载终极“全能”聊天机器人，允许您在单个桌面应用程序中私密地使用任何 LLM、embedder 和向量数据库。100% 私密。</li><li><a href="https://github.com/nomic-ai/gpt4all/pull/3440">Support for deekseek thinking in the gui. by manyoso · Pull Request #3440 · nomic-ai/gpt4all</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **MCP (Glama) ▷ #[general](https://discord.com/channels/1312302100125843476/1312302100125843479/1334296931471593503)** (47 条消息🔥): 

> `MCP Server 集成, 自托管 Web 客户端, Cursor MCP 支持, MCP 环境变量, Function Calling 问题` 


- **Cursor 增加 MCP 支持但存在限制**：成员们对 Cursor 增加 MCP 支持感到兴奋，尽管目前在添加环境变量方面存在限制。
   - 有人建议使用类似 `FOO=bar npx some-server` 的语法来设置变量，指出了潜在的变通方法。
- **自托管 Web 客户端成为焦点**：一位用户分享了关于其自托管 Web 客户端的见解，该客户端可以管理多个 MCP Server 和 Agent，并允许自动交接。
   - 这种方法保证了无论是在本地还是在云端都能无缝运行，展示了托管的灵活性。
- **关于 MCP 中 Function Calling 的讨论**：成员们讨论了一个 8b 模型在 Function Calling 和工具使用方面遇到的困难。
   - 成员们表示有兴趣确保用户（特别是在 Reddit 等平台上）能更好地集成和理解 MCP。
- **MCP 尚无动态 Agent Prompt**：一位成员表示，虽然动态 Agent Prompt 尚未实现，但系统配置可以简单地通过 Prompt 来定义。
   - 因此，用户无需复杂的设置即可自定义 Agent 行为，从而可能提高易用性。
- **MCP 与 LSP 配置结构的比较**：有人担心 MCP 没有采用与 Language Server Protocol (LSP) 相同的配置结构，后者允许 Server 向 Client 请求配置。
   - 这种结构上的差异被视为当前 MCP 实现中的一个限制。



**提到链接**: <a href="https://www.gnu.org/software/coreutils/manual/html_node/env-invocation.html">env 调用 (GNU Coreutils 9.6)</a>: 未找到描述

  

---


### **MCP (Glama) ▷ #[showcase](https://discord.com/channels/1312302100125843476/1315696461316358175/1334266918034735306)** (12 条消息🔥): 

> `Hataraku SDK 提案, TypeScript CLI 开发, 协作开发, 用户测试反馈` 


- **Hataraku 项目在 ShowHN 上势头强劲**：一个名为 **Hataraku** 的项目在 ShowHN 上排名 **#1**，引发了 Hacker News 社区的讨论和支持请求。
   - 鼓励参与者贡献想法并参与有关该项目的更广泛讨论。
- **Moonlife 的 TypeScript CLI 正在开发中**：Moonlife 正在积极开发 Hataraku 项目的 **TypeScript** 版本，并已开始仓库工作，进展顺利。
   - CLI 功能已经可以运行，但仍需要进一步抽象以完善该工具。
- **Hataraku TypeScript 实现的协作**：Saqadri 提议与 Moonlife 合作，特别是在完善 CLI 或讨论 TypeScript 版本的潜在改进方面。
   - Moonlife 确认他们已经 fork 了现有代码，以利用必要的开发基础设施。
- **界面开发进入最后阶段**：Moonlife 表示创建界面是最后一个重要步骤，核心功能进展顺利。
   - 正在寻求社区其他成员的反馈，并邀请通过私信分享见解。
- **用户测试与反馈机会**：Neil 表示有兴趣测试新界面，并强调了他们作为具有复杂工作流的用户经验，可以提供有用的反馈。
   - 这一询问反映了社区在确保不断发展的 Hataraku 项目易用性方面的持续参与。



**提到链接**: <a href="https://github.com/turlockmike/hataraku/blob/main/docs/sdk-proposal.md">hataraku/docs/sdk-proposal.md at main · turlockmike/hataraku</a>: 一个用于构建 AI 驱动开发工具的自主编码 Agent 和 SDK - turlockmike/hataraku

  

---

### **Notebook LM Discord ▷ #[announcements](https://discord.com/channels/1124402182171672732/1182376564525113484/1334657995463856169)** (1 条消息): 

> `NotebookLM 可用性研究，用户体验反馈` 


- **NotebookLM 寻求用户反馈**：NotebookLM UXR 正在组织远程聊天会议，以了解用户对产品的初步体验以及目前的使用方式。参与者将获得 **$75**（或等值金额）作为对其见解的感谢。
   - 有兴趣的人士可以填写 [筛选表单](https://forms.gle/HJmCwNepsfPSdC7g7) 申请参加定于 **2025 年 2 月 6 日** 进行的 60 分钟会议。
- **即将进行的可用性研究详情**：参与者需要高速互联网连接、活跃的 Gmail 账户以及具备视频和音频功能的设备。这项研究专注于为未来的产品增强收集反馈，强调用户需求的重要性。



**提及的链接**：<a href="https://forms.gle/HJmCwNepsfPSdC7g7">参与即将举行的 Google UXR 研究！</a>：你好，我正通过一份简短的问卷与你联系，以核实你参加即将举行的 Google 可用性研究的资格。这项研究是一个对当前正在...提供反馈的机会。

  

---


### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1334451854028832789)** (5 条消息): 

> `使用 AI 进行学习，NotebookLM Audio Overview，DeepSeek R1，用于理解的转录，用不同术语解释概念` 


- **AI 改变交易课程内容**：一位用户分享了他们如何将交易课程视频转换为音频，使用 AI 进行转录，并利用 [NotebookLM](https://link.to.notebooklm) 为同行澄清复杂话题。
   - 一种令人印象深刻的方法是使用《英雄联盟》（League of Legends）的术语来解释 **Big Players** 的概念，展示了 AI 在信息重构方面的多功能性。
- **NotebookLM 以创纪录的速度剖析行政命令**：AI 在总结复杂内容方面的高效性得到了证实，正如在关注公共教育隐私的新行政命令发布 **24 小时** 内所做的评论。
   - 听众可以观看详细的 [YouTube 视频](https://youtu.be/8RFYmgYn7P4?si=r9k0LVu_hOksnA4i)，获取该行政命令影响的客观概述。
- **NotebookLM Podcast 拆解 DeepSeek R1**：NotebookLM Podcast 讨论了 **DeepSeek R1**，用简单的术语解释了其 **GRPO** 和 **Mixture of Experts** 等特性，使复杂的 AI 技术变得通俗易懂。
   - 听众可以在[此处参与完整讨论](https://youtube.com/watch?v=zVDmKv3hWzk)，其中包括基准测试分析和快速演示。
- **Audio Overview 中的对话未被录制**：针对 Audio Overview 交互模式（Interactive Mode）下进行的对话是否持久保存的问题，确认了这些对话不会保存在可下载的录音中。
   - 这凸显了当前设计在捕捉动态讨论期间用户交互方面的局限性。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://youtube.com/watch?v=zVDmKv3hWzk">Is DeepSeek R1 the New ChatGPT Killer? NotebookLM Explains! 🔥</a>：🚀 DeepSeek V3 和 R1 解析（使用 NotebookLM）！DeepSeek R1 正在 AI 世界引起轰动，今天我们将使用 NotebookLM 对其进行全面拆解！...</li><li><a href="https://youtu.be/8RFYmgYn7P4?si=r9k0LVu_hOksnA4i">Objective NotebookLM review of an Executive Order focused on public education privacy &amp; patriotism</a>：特朗普总统签署的一项新行政命令专注于“结束 K-12 教育中的激进洗脑”。Google 的 NotebookLM Audio Overview 工具对此进行了评论...
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1334278115119071374)** (52 条消息🔥): 

> `NotebookLM 功能与性能、音频生成反馈、Gemini 更新、用户体验问题、播客见解` 


- **NotebookLM 生成速度缓慢**：用户报告点击“study guide”按钮后的生成时间差异很大，根据涉及的来源数量，预计耗时在 **10 到 30 分钟** 之间。
   - 一些用户发现，即使是单一来源也可能耗时出乎意料地长，引发了对性能一致性的担忧。
- **Audio Overviews 在其他语言中表现不佳**：一位培训师报告说，在测试 **韩语** 和 **日语** 时，Audio Overviews 表现不佳，表明多语言支持存在问题。
   - 参与者指出了困难，并表达了对改进这些语言功能的渴望，同时询问了他人的经验。
- **Gemini 2.0 Flash 更新导致故障**：在 **Gemini 2.0 Flash** 更新后，用户遇到了临时故障，引发了对其对性能影响的讨论。
   - 虽然功能随后恢复，但人们认为此次更新导致了一些用户面临的问题。
- **寻求更严格的来源利用规则**：一些用户正在探索将回答严格限制在上传来源的方法，寻求对 NotebookLM 更明确的指令。
   - 反馈表明，虽然用户可以添加 Prompt 以更好地遵守来源，但输出有时会包含外部参考，这使预期的二元回答变得复杂。
- **包含 NotebookLM 见解的播客**：一档由 NotebookLM 创始工程师参与的播客提供了关于该平台历史和增长的见解，引起了用户的兴趣。
   - 听众对未来的功能表示好奇，但指出对话中缺乏具体的细节分享。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://support.google.com/notebooklm/answer/15678219?hl=en">Upgrading to NotebookLM Plus - NotebookLM Help</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=Cr7J2PLo2fw">A Conversation with NotebookLM&#39;s Founding Engineer</a>: Google 的 NotebookLM 已成为处理文本最引人注目的 AI 工具之一。在这次对话中，该项目的创始工程师 Adam Bignell...</li><li><a href="https://youtube.com/watch?v=zVDmKv3hWzk">Is DeepSeek R1 the New ChatGPT Killer? NotebookLM Explains! 🔥</a>: 🚀 使用 NotebookLM 解释 DeepSeek V3 和 R1！DeepSeek R1 正在 AI 界引起轰动，今天我们将使用 NotebookLM 来全面解析它！...</li><li><a href="https://www.tiktok.com/t/ZT22DHefp/">TikTok - Make Your Day</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1334270047979442309)** (1 条消息): 

> `分支变更、Pull Requests` 


- **分支变更已完成**：**分支变更**现已完成，所有未完成的 **Pull Requests** 已成功**重定向 (retargeted)**。
   - 鼓励团队成员就这些更新提出任何问题。
- **Pull Requests 更新**：所有开启的 **Pull Requests** 已根据最近的分支变更进行了重定向。
   - 此调整旨在简化工作流程并促进更顺畅的集成。


  

---

### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1334338934632747019)** (48 条消息🔥): 

> `NeoVim LSP 集成, Mojo 1.0 讨论, 向后兼容性担忧, Mojo 中的 Reflection (反射), Mojo 性能基准测试` 


- **将 Mojo LSP 集成至 NeoVim**：成员们讨论了如何为 NeoVim 添加 Mojo LSP 支持，并参考了 [nvim/lspconfig GitHub](https://github.com/neovim/nvim-lspconfig)。
   - 然而，有人反映提议的解决方案并未按预期工作。
- **定义 Mojo 1.0：速度 vs. 稳定性**：Clattner 强调了发布一个有意义的 Mojo 1.0 的必要性，将其描述为一种非常适合快速执行和 GPU 利用的语言。
   - 讨论突显了实现即时可用性与确保长期稳定性和兼容性之间的矛盾。
- **对向后兼容性的担忧**：成员们对缺乏向后兼容性表示担忧，这可能会因为潜在的破坏性变更（breaking changes）而阻碍新版本的采用。
   - 普遍共识强调，确保与旧版库的兼容性对于生态系统的繁荣至关重要。
- **Mojo 中 Reflection (反射) 的重要性**：鉴于反射功能在数据序列化等用例中的重要性，关于是否应在 Mojo 1.0 中包含反射功能展开了辩论。
   - 人们担心缺乏反射会影响可用性，但也有人指出目前已经实现了一些反射功能。
- **Mojo 性能基准测试**：成员们讨论了在大型计算集群上对 Mojo 进行基准测试的必要性，以有效评估其性能。
   - 其观点是，确保在大内存机器上的稳健性能将简化配置较低用户的开发工作。



**相关链接**: <a href="https://youtu.be/9ag0fPMmYPQ),">Mojo🔥：与 Chris Lattner 深入探讨 Ownership</a>：了解关于 Mojo 中 Ownership 的一切，与 Modular CEO Chris Lattner 的深度探讨。如果您有任何问题，请务必加入我们友好的...

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1334295061328035841)** (38 条消息🔥): 

> `Mistral Small 3, DeepSeek 数据库泄露, Riffusion 的新模型, OpenAI API 延迟监控, ElevenLabs C 轮融资` 


- **Mistral Small 3 发布，参数惊人**：Mistral AI 发布了 **Mistral Small 3**，这是一个拥有 24B 参数的模型，在 **MMLU 上准确率达到 81%**，性能为 **150 tokens/sec**，现已根据 Apache 2.0 许可证开源。
   - 显著改进包括更少的层数和词汇量（vocabulary size）的大幅提升，社交媒体上分享了与其他模型的详细对比。
- **DeepSeek 面临重大数据库泄露**：发现了一个属于 **DeepSeek** 的公开 ClickHouse 数据库，泄露了包括聊天记录和密钥（secret keys）在内的敏感内部数据。
   - 该问题由 **Wiz Research** 发现后进行了负责任的披露并迅速得到修复，引发了人们对 AI 行业数据安全的担忧。
- **Riffusion 推出生成式音乐模型 FUZZ**：Riffusion 推出了 **FUZZ**，这是一款旨在生成**高质量音乐**的生成式模型，只要 GPU 资源充足，他们将免费提供。
   - 该公告突显了生成式音乐模型的持续发展和能力，表明该领域正在积极创新。
- **讨论监控 OpenAI API 延迟**：对 OpenAI API 潜在**延迟增加**的担忧促使人们讨论第三方监控解决方案，如 **OpenRouter** 和 **Artificial Analysis**。
   - 虽然初步检查显示延迟正常，社区成员交流了关于可用工具的见解，以便更好地长期衡量 API 性能。
- **ElevenLabs C 轮融资 1.8 亿美元**：ElevenLabs 完成了由 **a16z 和 ICONIQ** 领投的 **1.8 亿美元 C 轮**融资，强调了他们致力于增强 AI 能力的承诺。
   - 这一重大融资轮次标志着投资者对 AI 语音技术及其潜在市场影响的强大信心。


<div class="linksMentioned">

<strong>相关链接</strong>:

<ul>
<li>

<li><a href="https://x.com/allen_ai/status/1884966600039915809">来自 Ai2 (@allen_ai) 的推文</a>：这是 Tülu 3 405B 🐫，我们的开源后训练模型，其性能超越了 DeepSeek-V3！作为 Tülu 3 家族的最后一名成员，它展示了我们的方案（包括 Reinforcement...）</li><li><a href="https://block.github.io/goose/">codename goose | codename goose</a>：你的开源 AI Agent，无缝自动化工程任务。</li><li><a href="https://mistral.ai/news/mistral-small-3/">Mistral Small 3</a>：Apache 2.0 协议，81% MMLU，150 tokens/s</li><li><a href="https://x.com/dchaplot/status/1884975434519245021">来自 Devendra Chaplot (@dchaplot) 的推文</a>：Mistral Small 3 Instruct 模型的性能。在 HF 下载：https://huggingface.co/mistralai/Mistral-Small-24B-Instruct-25014/N</li><li><a href="https://x.com/eliebakouch/status/1884979232280813856">来自 elie (@eliebakouch) 的推文</a>：新的 Mistral (并不那么) Small，RoPE -> 100M 对应 32k 上下文长度 - 这次没有 SWA。作为参考，Qwen 1M 上下文长度的 RoPE 基数为 10M。我很期待。</li><li><a href="https://artificialanalysis.ai/providers/openai">OpenAI - 质量、性能与价格分析 | Artificial Analysis</a>：对 OpenAI 模型在质量、价格、输出速度、延迟、上下文窗口等关键指标上的分析。</li><li><a href="https://x.com/riffusionai/status/18849">来自 ok, i'll byte 🇺🇦 (@cw) 的推文</a>：我经历过的最短飞行：从圣托里尼到米科诺斯岛，仅 20 分钟。手机被允许正常使用。</li><li><a href="https://x.com/stochasticchasm/status/1885005454369272002">来自 stochasm (@stochasticchasm) 的推文</a>：新的 Mistral Small 相比之前的版本，模型维度（model dim）更窄，前馈网络维度（ff dim）翻倍，且少了 16 层？这相当于 ff_dim/model_dim 的比例达到了 6！这太奇怪了。</li><li><a href="https://x.com/espadrine/status/1885004488206856638">来自 Thaddée Tyl (@espadrine) 的推文</a>：Mistral Small 2 到 3 的变化：• 为了降低延迟，从 55 层减少到 40 层 • 词汇量从 33K 增加到 131K • Embedding 从 6K 减少到 5K • Attention Heads 从 48 减少到 32 • 10 倍的 rope_theta • SYSTEM_PROMPT token https://mistral...</li><li><a href="https://x.com/sophiamyang/status/1884970987441316268">来自 Sophia Yang, Ph.D. (@sophiamyang) 的推文</a>：🚀 发布 @MistralAI Small 3，我们迄今为止最高效、最通用的模型！✅ 24B 参数 ✅ 81% MMLU ✅ 150 tokens/sec ✅ Apache 2.0 许可证 ✅ 预训练与指令微调（无合成数据——非常适合重新...）</li><li><a href="https://x.com/kagigz/status/1884670976656630059">来自 Katia Gil Guzman (@kagigz) 的推文</a>：在 OpenAI DevDay 上，我们通过一个可以用语音导航的交互式太阳系演示介绍了 Realtime API。很多人询问它是如何构建的，所以我们已经将其开源了——包含...</li><li><a href="https://x.com/TheRealAdamG/status/1884971520348283217">来自 Adam.GPT (@TheRealAdamG) 的推文</a>：https://help.openai.com/en/articles/6825453-chatgpt-release-notes#h_caaeddc37e ChatGPT 昨天进行了一些不错的增量更新。积少成多。</li><li><a href="https://x.com/riffusionai/status/1884984941081198954?s=46">来自 Riffusion (@riffusionai) 的推文</a>：隆重推出 FUZZ —— 一款与众不同的生成式音乐模型。个性化、全长、高质量且无限。只要我们的 GPU 还能撑住，我们就让这款乐器保持免费。FUZZ 的精华...</li><li><a href="https://x.com/matistanis/status/1885011065018163224">来自 Mati Staniszewski (@matistanis) 的推文</a>：今天，ElevenLabs 开启了新篇章——我们完成了由 a16z 和 ICONIQ 领投的 1.8 亿美元 C 轮融资，旨在为每个 AI Agent 提供声音。</li><li><a href="https://x.com/iScienceLuvr/status/1884736091619537346">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：任何认为 DeepSeek 是凭空出现的人都应该看看这张图。图中的每个模型都发布了权重、代码和详细论文。这是一支拥有强大过往业绩且...</li><li><a href="https://x.com/alibaba_qwen/status/1884809286288810231?s=46">来自 Qwen (@Alibaba_Qwen) 的推文</a>：发布 Qwen2.5-VL Cookbooks！🧑‍🍳 一系列展示 Qwen2.5-VL 使用案例的 Notebook，包括本地模型和 API。示例包括计算使用、空间理解、文档解析...</li><li><a href="https://www.wiz.io/blog/wiz-research-uncovers-exposed-deepseek-database-leak">Wiz Research 发现 DeepSeek 数据库暴露，泄露敏感信息，包括聊天记录 | Wiz 博客</a>：一个属于 DeepSeek 的公开可访问数据库允许对数据库操作进行完全控制，包括访问内部数据的能力。此次暴露涉及超过一百万行日志字符串...</li><li><a href="https://x.com/altryne/status/1884778839009796411?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：扎克伯格在财报电话会议上的要点：- Llama 4 和 Llama 4 mini（已完成预训练）- 确认了推理版 Llama！- Llama 4 将是原生多模态的 --</li>

这是一个全能模型（omni-model）——它将拥有...</li><li><a href="https://openrouter.ai/openai/o1-preview">o1-preview - API, Providers, Stats</a>: OpenAI 最新且最强大的模型系列，o1 旨在响应前花更多时间思考。o1 模型针对数学、科学、编程和其他 STEM 相关任务进行了优化...</li><li><a href="https://x.com/mistralai/status/1884967826215059681?s=46">来自 Mistral AI (@MistralAI) 的推文</a>: magnet:?xt=urn:btih:11f2d1ca613ccf5a5c60104db9f3babdfa2e6003&dn=Mistral-Small-3-Instruct&tr=udp%3A%2F%http://2Ftracker.opentrackr.org%3A1337%2Fannounce&tr=http%3A%2F%http://2Fopen.tracker.cl%3A1337%2F...</li><li><a href="https://github.com/openai/openai-realtime-solar-system">GitHub - openai/openai-realtime-solar-system: 展示如何使用 OpenAI Realtime API 通过 tool calling 导航 3D 场景的演示</a>: 展示如何使用 OpenAI Realtime API 通过 tool calling 导航 3D 场景的演示 - openai/openai-realtime-solar-system</li><li><a href="https://archive.is/KiSYM">OpenAI 声称有证据表明中国的 DeepSeek 使用其模型进行训练...</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1334255715757461595)** (8 条消息🔥): 

> `赛道信息、报名回复、Quiz 1 发布、LLM Agents Quiz 仓库、证书更新` 


- **赛道信息待定**：成员们对在**应用（application）**和**研究（research）**背景下讨论的**赛道（tracks）**表示好奇，组织者承诺后续将提供更多信息。
   - *请保持关注！* 以获取有关这些赛道内容的更新。
- **报名确认延迟**：多位参与者反映已填写 Google Forms 报名表，但尚未收到关于其状态的任何回复。
   - 他们渴望获得更新，特别是关于申请 **PhD** 机会的相关信息。
- **Quiz 1 可用性**：一位成员询问了 **Quiz 1** 的发布情况，已确认该测验位于课程网站的教学大纲（syllabus）部分。
   - 关于**首批课程证书**的细节仍在待定中，建议成员等待后续更新。
- **寻找往届测验答案**：一位参与者请求访问之前 **LLM Agent 课程**的测验答案仓库。
   - 分享了一个 [包含测验存档的 Google 文档](https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing) 链接，但备注指出浏览器版本已过时。
- **证书尚未发放**：已确认课程的**证书**尚未发放，预计很快会有更多信息。
   - 鼓励成员保持关注，因为本学期证书的具体要求稍后将公布。



**提到的链接**：<a href="https://docs.google.com/document/d/1pYvOxt2UWwc3z4QlW2Di5LQT-FJPWZ419mxJT7pFPsU/edit?usp=sharing">测验存档 - LLM Agents MOOC</a>：注意：正确答案在黑框中（黑底黑字）。用光标选中方框以显示正确答案（如果难以查看，也可以将文本复制到新浏览器中...）。

  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-lecture-discussion](https://discord.com/channels/1280234300012494859/1282734248112947210/1334334453803384883)** (5 条消息): 

> `讲座上传、伯克利无障碍政策、通过网站访问讲座` 


- **第一讲上传延迟**：一位成员请求今天上传**第一讲**，认为团队只需 **5 分钟**的工作量。
   - 另一位成员指出，由于**伯克利（Berkeley）的政策**，该过程涉及大量的编辑和**字幕制作（captioning）**。
- **伯克利的无障碍要求**：针对在没有完整**无障碍设施（accessibility accommodations）**（如字幕）的情况下发布视频表示了担忧。
   - 团队成员强调了耐心的重要性，因为他们正在处理这些公开发布的要求。
- **在线访问讲座录像**：提醒成员可以在 [网站](https://llmagents-learning.org/sp25) 上通过直播链接观看讲座录像。
   - 澄清说，虽然由于正在制作字幕，剪辑版尚未公开，但仍可以通过提供的链接观看。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1334599628737351691)** (2 messages): 

> `AI Agent 工作坊，LlamaIndex 入驻 BlueSky` 


- **精通 AI Agents 工作坊**：参加 @seldo 的全面工作坊，学习如何使用 **LlamaIndex** 构建高级 **AI agents** 和 **多智能体系统 (multi-agent systems)**！工作坊将涵盖 **AgentWorkflow** 以及创建稳健多智能体框架的基础知识，并从[此处](https://t.co/UKIClalkKG)提供实战经验。
   - 参与者将深入研究 **Workflows**，这是增强 Agent 能力所必需的核心构建块，确保对多智能体系统架构有深入理解。
- **LlamaIndex 登陆 BlueSky**：LlamaIndex 已正式加入 **BlueSky**！关注他们的动态，了解他们在这个新兴平台上的新探索：[链接](https://t.co/GK4L8Sb2N6)。
   - 与社区互动，发现 **BlueSky** 上关于 AI 发展和创新的有趣讨论。



**提到的链接**：<a href="https://t.co/GK4L8Sb2N6">LlamaIndex (@llamaindex.bsky.social)</a>：将 LLMs 连接到你的数据的框架。

  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1334570774157328517)** (10 messages🔥): 

> `LlamaIndex 对 o1 的支持，o1 流式传输问题，OpenAI 模型能力` 


- **LlamaIndex 声称支持 o1**：一位用户询问为什么 **LlamaIndex** 仅支持 `o1-preview`，但随后指出在通过 `pip install -U llama-index-llms-openai` 更新后，`o1` 确实已获得支持。
   - 然而，一些人注意到完整功能可能尚不可用。
- **o1 模型缺乏流式传输支持**：用户担心 **o1** 模型没有对流式传输 (streaming) 的妥善支持，而这在 `o1-preview` 和 **o1-mini** 中是成功的。
   - 错误消息显示 **o1** 中存在不支持的流式传输值，引发了进一步讨论。
- **研究揭示了 OpenAI 的局限性**：经过进一步研究，结论是 **OpenAI** 尚未完全开发出 **o1** 模型的所有能力。
   - 相关的 [社区讨论](https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043?utm_source=chatgpt.com#:~:text=Streaming%20of%20the,for%20this%20model.) 强调了这些局限性。
- **关于 o1 的奇怪支持体验**：成员们评论了来自 **OpenAI** 的 **o1** 模型相关的奇怪支持体验，指出许多功能均不受支持。
   - 这导致了试图利用新模型的用户之间的困惑和沮丧。



**提到的链接**：<a href="https://community.openai.com/t/streaming-support-for-o1-o1-2024-12-17-resulting-in-400-unsupported-value/1085043?utm_source=chatgpt.com#:~:text=Streaming%20of%20the,for%20this%20model.">o1 (o1-2024-12-17) 的流式传输支持（导致 400 "Unsupported value" 错误）</a>：你好，似乎为 o1-preview 和 o1-mini 添加了流式传输支持（参见公告 OpenAI o1 流式传输现已可用 + 1-5 层的 API 访问）。我确认这两者对我来说都有效。然而...

  

---

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1334349996170154066)** (11 messages🔥): 

> `NVIDIA GPU 与 Hypervisor，互连 Tiny Box，VRAM 共享技术，Tiny Box 性能，LLM 物理服务器选择` 


- **GPU 设置需要测试**：一位成员强调了在使用 **P2P patch** 时，针对多 NVIDIA GPU 配置进行**测试**的重要性。
   - 他们询问其他成员是由于 IOMMU 的限制而使用像 **Proxmox 这样的 hypervisor**，还是选择 baremetal 安装。
- **对 Tiny Box 互连的好奇**：一位成员思考了可以互连多少个 **Tiny Box**，并询问了在讨论可实现的推理性能时，如何在它们之间共享 **VRAM**。
   - 另一位成员指出目前缺乏一种**无缝的方法**来共享 VRAM，但建议使用高速 **NIC/connectx** 网卡进行基于网络的推理，这可以很好地扩展。
- **推理性能评估**：评估表明，如果一个模型可以处理 **15 tokens/sec**，理论上在扩展时可以以略低的速度（每个 14 tok/sec）处理 **100 个请求**。
   - 这突出了在明确定义的条件下，分布式请求的潜在性能特征。
- **为 Tiny Box 探索 MLX**：关于使用 **MLX** 聚合 Tiny Box 能力的讨论引发了一些关于其在此背景下具体作用的困惑。
   - 对 **Apple Silicon 张量库** 的引用表明，对于 **MLX** 在其设置中的适用性存在一些不同的解读。
- **寻求物理服务器建议**：一位成员表示有兴趣购买一台**物理服务器**在本地托管 LLM 以供企业使用，并寻求理想选择的建议。
   - 这表明企业环境中对大规模模型自托管解决方案的兴趣日益浓厚。


  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1334411118583484416)** (1 messages): 

> `分块/融合程序示例代码，张量操作` 


- **寻找分块/融合代码示例**：一位用户询问是否有在 tinygrad 中实现**分块/融合程序 (blocked/fused programs)** 的**优秀示例代码**。
   - 他们特别要求提供演示如何**加载/写入张量块**以高效执行操作的示例。
- **关于张量块操作的讨论**：对话围绕如何在 tinygrad 中通过**逐块处理张量**来高效执行操作展开。
   - 成员们强调了**融合操作 (fusing operations)** 对提高性能和最小化资源使用的重要性。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1334549568259489926)** (3 messages): 

> `AI 情感反应，AI 人格化，AI 模型的感知` 


- **用户觉得 AI 模型很冷漠**：一位用户表示 **AI 模型** 在交互中显得有些冷漠。
   - 这种情绪引发了其他人的玩笑，说需要一条*毯子*来给它们取暖。
- **机器不需要温暖**：另一位成员指出 **AI 模型** 是机器，最终不需要任何温暖或人类情感。
   - 这一评论进一步推动了关于将 AI 视为更具情感实体的轻松讨论。


  

---


### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1334344851092930580)** (1 messages): 

> `支持工单，Discord 频道沟通` 


- **已创建支持工单**：一位成员创建了一个 [支持工单 (support ticket)](https://discord.com/channels/954421988141711382/1334344003994386474/1334344003994386474) 以寻求帮助，确保问题得到记录和跟踪。
   - 这再次强调了在 Discord 频道中保持清晰沟通对于高效解决问题的重要性。
- **后续沟通的重要性**：跟进支持工单对于保持清晰的沟通和高效解决问题至关重要。
   - 成员们讨论了确保支持渠道保持活跃和响应的最佳实践。


  

---

### **Cohere ▷ #[cmd-r-bot](https://discord.com/channels/954421988141711382/1168578374038470656/1334297971516248124)** (8 条消息🔥): 

> `command-r7b, Command R model, distillation frameworks` 


- **用户在 command-r7b 和 distillation frameworks 方面遇到困难**：一位成员表示在让 **command-r7b** 响应 **distillation frameworks**（如合成数据生成）时遇到困难，并询问关于使用 **ollama** 的建议。
   - 这表明在各种 **frameworks** 与 command-r7b 模型的集成支持方面可能存在差距。
- **关于 Command R 能力的见解**：在后续回复中，机器人提供了 **Command R** 的概述，详细介绍了其作为针对对话任务和检索增强生成（retrieval-augmented generation）优化的 LLM 的特性。
   - Command R 具有 **128,000 token 的上下文长度**，支持复杂工作流的工具使用（tool use），并在即将发布的版本中进行了旨在改进决策和数据分析的增强。
- **进一步学习的资源**：机器人包含了关于 **models** 概览、**Command R** 具体细节及其检索增强生成能力的补充阅读链接。
   - 这些资源可以为成员提供关于 Command R 功能和性能的更深入见解：[Models Overview](https://docs.cohere.com/v1/docs/models), [The Command R Model](https://docs.cohere.com/v1/docs/command-r), [Command R Changelog](https://docs.cohere.com/v1/changelog/command-r-retrieval-augmented-generation-at-production-scale)。


  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1334320254872059985)** (6 条消息): 

> `Adding proxy to dspy.LM adapter, Supported LLMs in DSPy, Setting litellm client with http_client, Documentation references, LiteLLM model support` 


- **为 dspy.LM 适配器添加代理（Proxy）**：一位用户询问如何为 `dspy.LM` 适配器添加 **proxy**，并引用了 [GitHub PR](https://github.com/stanfordnlp/dspy/pull/1331) 中的相关添加。该功能之前在已弃用的 `gpt3.py` 模块中实现，引发了对兼容性的担忧。
   - 另一位用户提到，由于其托管端点的代理要求，他们无法使用 **dspy 2.6**。
- **DSPy 支持的 LLMs**：一位新手询问 DSPy 支持哪些 **LLMs**，促使一名成员分享了 [LiteLLM 文档](https://docs.litellm.ai/docs/providers) 的链接，该文档详细介绍了各种模型提供商。
   - 该文档列出了对 **OpenAI**、**Azure** 和 **VertexAI** 等模型的支持。
- **使用 http_client 设置 litellm 客户端**：一位用户表示难以在 DSPy 参数中找到关于使用 **SSL context** 的 `http_client` 设置 **litellm client** 的信息。他们提到现有文档中未指明此设置。
   - 讨论继续引用了 [dspy/lm.py](https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53) 文件中的特定行，强调了框架细节。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.litellm.ai/docs/providers">Providers | liteLLM</a>: 了解如何在 LiteLLM 上部署和调用来自不同提供商的模型</li><li><a href="https://github.com/stanfordnlp/dspy/pull/1331">http client added to GPT3 by mjaliz · Pull Request #1331 · stanfordnlp/dspy</a>: 在 openai 上设置 http_client 以支持 http_proxy</li><li><a href="https://github.com/stanfordnlp/dspy/blob/main/dspy/clients/lm.py#L53">dspy/dspy/clients/lm.py at main · stanfordnlp/dspy</a>: DSPy：用于编程（而非提示）语言模型的框架 - stanfordnlp/dspy
</li>
</ul>

</div>

### **Axolotl AI ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1334251809111543951)** (6 messages): 

> `Axolotl for KTO, New Mistral model, User tasks and feature requests, Winter semester calendar, Mistral AI open source commitment` 


- **在 KTO 中使用 Axolotl 面临挑战**：*如果我们不能在 KTO 中使用 Axolotl，那运气就太糟了*，强调了 Axolotl 集成的紧迫性。
   - 成员们对可行性表示担忧，其中一人询问任务是否可以完成，并表示愿意协助 Review。
- **对新 Mistral 模型发布的兴奋**：一位成员分享了关于新 [Mistral-Small-24B-Base-2501 模型](https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501) 发布的兴奋之情，该模型拥有 **24B 参数**，在小型 LLM 中排名很高。
   - 会上提到，还将有针对特定能力的额外商业模型，强调了 **Mistral AI 对开源的承诺**。
- **对 Mistral 模型性能的不确定性**：当被问及新的 Mistral 模型是否有效时，一位成员承认：*我有一段时间没训练了，所以不太清楚*。
   - 这表明缺乏近期对该模型的实际操作经验，从而引发了关于用户体验的对话。
- **繁忙的冬季学期日程**：一位成员提到冬季学期日程繁忙，说道：*抱歉，今年冬季学期我的日程表排得非常满*。
   - 这可能会影响他们在未来几个月参与协作任务的可用性。



**提及的链接**：<a href="https://huggingface.co/mistralai/Mistral-Small-24B-Base-2501">mistralai/Mistral-Small-24B-Base-2501 · Hugging Face</a>：未找到描述

  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1334599850293067837)** (3 messages): 

> `Farm Friend, Cliche Reviews` 


- **关于 Farm Friend 的询问**：一位成员表达了他们对 **Farm Friend** 的喜爱，提到去年很喜欢它，但现在似乎找不到了。
   - 社区对该项目的现状感到好奇。
- **Meme 分析与陈词滥调的评论**：另一位成员幽默地评论了社区内的 **陈词滥调评论 (cliché reviews)**，引发了轻松的回应。
   - 分享的一张图片可能说明了这种情绪，增强了讨论的俏皮基调。
- **澄清 01 的含义**：一位成员澄清了他们之前关于 “01” 的消息，明确指出它与 **OpenAI** 无关。
   - 这一评论表明对话中可能存在误解或沟通不畅。


  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1334468225999573012)** (2 messages): 

> `DCP Checkpointing, Config Settings` 


- **配置中的 DCP Checkpointing 状态**：一位成员提出了一个问题，即当前的任何配置中是否启用了 **DCP checkpointing**。
   - 另一位成员指出，目前尚未启用 checkpointing，但如果在配置中设置 **enable_async_checkpointing=True** 即可激活，尽管目前仅适用于 `full_finetune_distributed`。
- **Checkpointing 与全量微调的集成**：**checkpointing** 功能目前主要仅集成到 `full_finetune_distributed` 配置中。
   - 这意味着即使启用了异步 checkpointing，其功能也可能无法在所有配置中使用，从而限制了它的用途。


  

---


### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1334303344071675985)** (2 messages): 

> `img2vid tools, ltxv` 


- **讨论最佳本地 img2vid 工具**：一位用户询问目前可用的最佳本地 **img2vid** 工具。
   - 另一位成员表达了对 **ltxv** 的偏好，认为它是潜在的首选。
- **用户对 ltxv 的偏好**：对 **ltxv** 的偏好被作为 img2vid 应用的一个值得注意的提及而分享。
   - 这表明人们对提供有效视频生成能力的本地工具的兴趣日益增加。


  

---

### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1334275103957057691)** (1 messages): 

> `MLOps Workshop, Feature Store on Databricks, Q&A Session, Data Engineering, Geospatial Analytics` 


- **Databricks 上的 MLOps Workshop 现已上线！**：加入我们的创始人 **Simba Khadder**，参加 **1 月 30 日上午 8 点 (PT)** 举行的“MLOps Workshop: Building a Feature Store on Databricks”实操演示。
   - 该研讨会涵盖在 Databricks 上构建和部署**生产级 feature pipelines**，不要错过[在此注册](https://buff.ly/40Ej4Z6)的机会！
- **真实世界用例和最佳实践**：Simba 将指导参与者充分利用 **Databricks** 和 **Unity Catalog**，讨论建立 feature store 的最佳实践。
   - 最后将有一个 **Q&A** 环节，允许与会者直接就所呈现的主题进行交流。
- **面向 AI/ML 爱好者的免费活动**：本次研讨会专为 **Data Engineers**、**Data Scientists** 和 **Machine Learning Engineers** 设计，欢迎任何对 AI 和 ML 感兴趣的人参加。
   - 活动**免费**，方便任何希望提升该领域技能的人参加。
- **即将举行的 Geospatial Analytics 活动**：请在日历上标记 **2025 年 1 月 30 日下午 1:00 (EST)** 的 **Geospatial Analytics with Databricks**。
   - 这是另一个参与高级分析主题的免费机会，可在 [Eventbrite](https://www.eventbrite.com/e/doi-geospatial-analytics-with-databricks-tickets-1111902653769?aff=erelexpmlt) 上注册。



**提及的链接**：<a href="https://buff.ly/40Ej4Z6">MLOps Workshop: Building a Feature Store on Databricks</a>：加入我们与 Featureform 创始人进行的 1 小时网络研讨会，了解如何通过使用 Featureform 和 Databricks 来增强您的数据能力！

  

---


### **Gorilla LLM (Berkeley Function Calling) ▷ #[discussion](https://discord.com/channels/1111172801899012102/1111353033352294440/)** (1 messages): 

glitchglitchglitch: 我们需要做什么才能让 bfcl 数据符合 hf datasets 的规范？
  

---


---


---


{% else %}


> 完整的逐频道明细已针对邮件进行截断。 
> 
> 如果您想查看完整明细，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预先感谢！

{% endif %}