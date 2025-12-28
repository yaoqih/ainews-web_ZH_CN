---
companies:
- openai
- llamaindex
- langchainai
- qdrant
- genmoai
date: '2025-01-11T07:18:42.365063Z'
description: '**Moondream** 发布了新版本，提升了显存（VRAM）效率，并增加了结构化输出和视线检测功能，标志着视觉模型实用性迈向了新前沿。Twitter
  上的讨论聚焦于推理模型的进展（如 **OpenAI 的 o1**）、模型蒸馏技术，以及 **vdr-2b-multi-v1** 和 **LLaVA-Mini**
  等新型多模态嵌入模型，这些技术显著降低了计算成本。关于生成对抗网络（GANs）和去中心化扩散模型的研究展示了更高的稳定性和性能。**MLX** 和 **vLLM**
  等开发工具迎来了更新，提升了便携性和开发者体验；同时，**LangChain** 和 **Qdrant** 等框架助力实现智能数据工作流。公司动态方面，**GenmoAI**
  宣布了新职位及团队扩张。*“效率技巧即一切（Efficiency tricks are all you need）。”*'
id: f70a1040-15e6-4dd0-aa9a-562aab4c7079
models:
- o1
- vdr-2b-multi-v1
- llava-mini
original_slug: ainews-moondream-202519-structured-text-enhanced
people:
- philschmid
- saranormous
- jxmnop
- reach_vb
- iscienceluvr
- multimodalart
- arohan
- adcock_brett
- awnihannun
- russelljkaplan
- ajayj_
title: Moondream 2025.1.9：在 2B 模型中实现结构化文本、增强 OCR 与视线检测功能。
topics:
- vision
- model-efficiency
- structured-output
- gaze-detection
- reasoning
- model-distillation
- multimodality
- embedding-models
- gan
- diffusion-models
- self-attention
- training-optimizations
- development-frameworks
- api
- cross-language-deployment
- semantic-search
- agentic-document-processing
- developer-experience
---

<!-- buttondown-editor-mode: plaintext -->**效率技巧就是你所需要的一切。**

> 2025年1月9日至1月10日的 AI 新闻。我们为您检查了 7 个 subreddits、[**433** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **32** 个 Discord 服务器（**219** 个频道和 **2928** 条消息）。预计节省阅读时间（以 200wpm 计算）：**312 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

Moondream 因其[小巧、轻便、快速且达到 SOTA 水平的视觉能力](https://www.youtube.com/watch?v=T7sxvrJLJ14)而备受关注，并在昨天发布了一个出色的一新版本，标志着 VRAM 占用的新效率前沿（比单纯的参数量更具实际意义）：


![image.png](https://assets.buttondown.email/images/b1987557-b71d-49c5-8e4f-c141e587d791.png?w=960&fit=max)


它现在还提供结构化输出和视线检测（gaze detection），这让[富有创意的 Redditor 们想出了如下脚本](https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/)：


![image.png](https://assets.buttondown.email/images/d306a5da-d414-4bdb-b6a7-c7100412f28f.png?w=960&fit=max)


如果您错过了，Vik 还在 Vision Latent Space Live 的 Best of 2024 活动中发表了关于 Moondream 的演讲：

https://www.youtube.com/watch?v=76EL7YVAwVo


---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 回顾

> 所有摘要均由 Claude 3.5 Sonnet 完成，取 4 次运行中的最佳结果。

**AI 模型与研究**

- **推理模型与蒸馏技术**：[@_philschmid](https://twitter.com/_philschmid/status/1877778889566494843) 和 [@saranormous](https://twitter.com/saranormous/status/1877608687344431586) 讨论了推理模型的进展，如 [@OpenAI](https://twitter.com/OpenAI) 的 o1，**详细介绍了构建此类模型的步骤**。此外，[@jxmnop](https://twitter.com/jxmnop/status/1877761437931581798) 强调了**模型蒸馏（distillation）的有效性**，并指出其在缺乏理论解释的情况下取得了令人惊讶的性能提升。

- **多模态与嵌入模型**：[@llama_index](https://twitter.com/llama_index/status/1877778352087699962) 推出了 “vdr-2b-multi-v1”，这是一个 **2B 参数的多模态、多语言嵌入模型**，在多种语言中实现了 **95.6% 的平均 NDCG@5**。[@reach_vb](https://twitter.com/reach_vb/status/1877773277571014882) 展示了 **LLaVA-Mini**，它**减少了 77% 的 FLOPs**，并能在单个 GPU 上实现 **3 小时视频处理**。

- **GAN 与扩散模型的创新**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1877782765107908986) 和 [@multimodalart](https://twitter.com/multimodalart/status/1877724335474987040) 分享了关于**现代 GAN 基准**和**去中心化扩散模型（Decentralized Diffusion Models）**的研究，强调了它们与传统方法相比的**稳定性与性能**。

- **自注意力与训练技术**：[@_arohan_](https://twitter.com/_arohan_/status/1877795996815728987) 讨论了用于**长度泛化（length generalization）**的 **stick-breaking attention**，而 [@addock_brett](https://twitter.com/adcock_brett/status/1877481322953715899) 预测 **2025 年将是物理 AI（Physical AI）之年**，并对**训练优化**和**模型架构**进行了反思。

**AI 工具与开发**

- **开发框架与 API**：[@awnihannun](https://twitter.com/awnihannun/status/1877490045915115992) 宣布了 **MLX** 的更新，通过支持多种语言和平台增强了**可移植性**。[@vllm_project](https://twitter.com/vllm_project/status/1877794657117392936) 为 **vLLM** 引入了 **nightly builds** 和 **原生 MacOS 支持**，通过**更快的安装**提升了**开发者体验**。

- **AI 集成与流水线**：[@LangChainAI](https://twitter.com/LangChainAI/status/1877747452486320610) 和 [@virattt](https://twitter.com/virattt/status/1877497641522835714) 演示了使用 **LangChain** 和 **Qdrant** 等工具构建 **LLM 驱动的数据流水线**和 **AI 驱动的数据工作流**，实现了**智能语义搜索**和 **Agent 式文档处理**。

- **模型导出与接口**：[@awnihannun](https://twitter.com/awnihannun/status/1877564909027835931) 提供了在 **MLX** 中**将函数从 Python 导出到 C++** 的指南，促进了**跨语言模型部署**。[@ai_gradio](https://twitter.com/ai_gradio/status/1877478548874699153) 展示了 **qwen 与 anychat 的集成**，以极少的代码增强了**开发者部署**能力。

**公司公告与更新**

- **公司角色与扩张**：[@russelljkaplan](https://twitter.com/russelljkaplan/status/1877538454969479181) 宣布了他的新角色 **"认知专家" (cognition guy)**，而 [@ajayj_](https://twitter.com/ajayj_/status/1877795313446007016) 欢迎新团队成员加入 **GenmoAI 的旧金山办公室**。

- **产品发布与增强**：[@TheGregYang](https://twitter.com/TheGregYang/status/1877540170414829675) 发布了 **Grok iOS 应用**，[@c_valenzuelab](https://twitter.com/c_valenzuelab/status/1877735724683788363) 在 **RunwayML** 上推出了 **新的牛仔系列**。[@everartai](https://twitter.com/skirano/status/1877790936966553807) 推出了 **角色微调 (character finetuning)** 服务，展示了在 **极少输入图像** 情况下卓越的 **流水线一致性 (pipeline consistency)**。

- **招聘与就业趋势**：[@cto_junior](https://twitter.com/cto_junior/status/1877685041696006345) 讨论了 **Microsoft** 的招聘趋势，而 [@bindureddy](https://twitter.com/bindureddy/status/1877474052589367388) 预测 **Salesforce** 和其他大型科技公司将由于 **AI 驱动的生产力提升** 而 **停止招聘工程师**。

**数据集与基准测试**

- **新数据集发布**：[@iScienceLuvr](https://twitter.com/iScienceLuvr/status/1877653288444665915) 和 [@miivresearch](https://twitter.com/miivresearch/status/1877653288444665915) 宣布了 **去中心化扩散模型 (Decentralized Diffusion Models)**，并发布了 **HtmlRAG** 和其他 **多模态数据集** 的 **代码** 与 **项目页面**。

- **基准测试与评估**：[@swyx](https://twitter.com/swyx/status/1877818998060175508) 分享了关于 **MMLU/GPQA 知识** 的见解，强调了对像 **@ExaAILabs** 这样的 **神经搜索引擎** 的需求。[@FinBarrTimbers](https://twitter.com/finbarrtimbers/status/1877791666330796180) 讨论了在 **机器人 (robotics)** 领域之外缺乏持久的 **认知基准测试 (cognitive benchmarks)**。

**AI 伦理、政策与社会**

- **AI 的社会影响与伦理**：[@fchollet](https://twitter.com/fchollet/status/1877535640717504810) 和 [@sama](https://twitter.com/sama/status/1877815461259235419) 辩论了 **AI 自动化社会** 中 **工作的未来** 以及 **AI 治理** 的 **伦理影响**，包括对 **政策限制** 和 **AGI 定义** 的担忧。

- **AI 的地缘政治影响**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1877730865003786384) 和 [@ClementDelangue](https://twitter.com/ClementDelangue/status/1877767382120255792) 强调了 **开源 AI** 所掌握的 **地缘政治力量**，以及 **中国** 等国家在 **AI 领域** 的 **战略举措**。

- **AI 安全与监管担忧**：[@DeepLearningAI](https://twitter.com/DeepLearningAI/status/1877792522732212268) 和 [@Nearcyan](https://twitter.com/nearcyan/status/1877734059687965125) 对 **AI 欺骗行为**、**公共安全** 以及针对潜在 **AI 驱动灾难** 进行 **妥善准备的必要性** 提出了担忧。

**个人动态与公告**

- **职业变动与角色**：[@russelljkaplan](https://twitter.com/russelljkaplan/status/1877538454969479181) 分享了对他新角色的兴奋，而 [@megansirotanggalenuyen_](https://twitter.com/karinanguyen_/status/1877578425906393431) 庆祝入选 **福布斯 30 Under 30**。

- **职场经历**：[@vikhyatk](https://twitter.com/vikhyatk/status/1877803302479421925) 表达了对他所在大学行政部门的担忧，[@sarahookr](https://twitter.com/sarahookr/status/1877464396722471354) 提供了关于 **洛杉矶灾情 (LA devastation)** 的个人更新。

- **学习与发展**：[@qtnx_](https://twitter.com/qtnx_/status/1877745878112387191) 提到了 **在 JAX 中学习 RL**，[@aidan_mclau](https://twitter.com/aidan_mclau/status/1877705861608452332) 讨论了 **AI 资本使用** 以及 **亿万富翁在 AI 开发中** 面临的挑战。

**梗/幽默**

- **对 AI 和技术的幽默看法**：[@nearcyan](https://twitter.com/nearcyan/status/1877820139992732125) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1877766872143147010) 分享了带有对 **AI 提示工程 (prompt engineering)**、**科技公司行为** 和 **AI 炒作** 的 **讽刺评论** 的推文，为技术讨论注入了轻松的评论。

- **随性且有趣的言论**：[@richardMCNgo](https://twitter.com/RichardMCNgo/status/1877806040563273999) 和 [@teortaxesTex](https://twitter.com/teortaxesTex/status/1877755556514996691) 发布了与 **AI 进展** 和 **科技文化** 相关的 **笑话** 和 **双关语**，为工程师受众提供了轻松时刻。

---

# AI Reddit 回顾

## /r/LocalLlama 回顾

**主题 1. Moondream 2b 的视线检测引发热议**

- **[有人想要在任何视频上运行 Moondream 2b 新的视线检测（gaze detection）脚本吗？](https://v.redd.it/n9beslavz0ce1)** ([Score: 1123, Comments: 207](https://reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/)): 该帖子讨论了一个用于在任何视频上运行 **Moondream 2b 视线检测**的脚本发布。帖子正文中没有提供额外的细节或背景。
  - **兴趣与热情**：许多用户，包括 **That_Neighborhood345** 和 **ParsaKhaz**，对视线检测脚本的发布表示了浓厚的兴趣，像 **ParsaKhaz** 这样的人甚至表示，如果大家感兴趣，他们愿意清理并发布自己的脚本。这表明社区对实验和利用视线检测技术有着显著的兴趣。
  - **监控担忧**：几位用户，如 **ArsNeph** 和 **SkullRunner**，表达了对视线检测技术可能被滥用于监控和侵犯隐私的担忧。他们列举了像中国的社会信用体系和企业微观管理等例子，认为这项技术可能会被滥用来监控个人的专注度和活动。
  - **技术可行性与用例**：**aitookmyj0b** 指出，利用基础的 **OpenCV processing** 实现视线检测是可行的，这表明该技术对于感兴趣的人来说已经触手可及。然而，**ArsNeph** 认为该技术在合法的眼动追踪应用中缺乏精确度，强调其主要用途是作为监控软件，而非用于有益的目的。


**主题 2. Transformers.js 通过 WebGPU 将 LLM 带入浏览器**

- **[使用 Transformers.js 在浏览器中 100% 本地运行 WebGPU 加速的推理 LLM](https://v.redd.it/vmfpb2m2r5ce1)** ([Score: 379, Comments: 62](https://reddit.com/r/LocalLLaMA/comments/1hy34ir/webgpuaccelerated_reasoning_llms_running_100/)): 演示了使用 **Transformers.js** 在浏览器中完全本地运行 **WebGPU-accelerated LLMs**。这展示了无需依赖服务器端处理即可实现**浏览器内 AI 应用**的潜力。
  - **性能差异**：用户报告了基于硬件的不同性能指标，例如 **RTX 3090** 达到了 **55.37 tokens per second**，而 **MiniThinky-v2** 在 **MacBook M3 Pro Max** 上达到了 **~60 tps**。性能指标中缺乏特定硬件说明被指出是机器学习讨论中的一个常见问题。
  - **技术探索与挑战**：人们对探索 **WebGPU** 的技术能力及其在本地运行 AI 模型中的应用很感兴趣。用户讨论了创建一个浏览器扩展的可能性，该扩展利用推理 LLM 直接操作 **DOM**，并强调隐私和本地处理。
  - **模型输出问题**：一些用户强调了模型输出的问题，例如生成无意义的文本或错误的推理，比如模型错误地声称 *“60 does not equal 60”* 的例子。这突显了在本地 AI 应用中实现准确可靠输出的挑战。


**主题 3. 拜登的 AI 芯片出口限制引发全球反应**

- **[拜登在最后冲刺中进一步限制 Nvidia AI 芯片出口，限制波兰、葡萄牙、印度或 Falcon 模型制造商阿联酋等美国盟友](https://www.bloomberg.com/news/articles/2025-01-08/biden-to-further-limit-nvidia-amd-ai-chip-exports-in-final-push)** ([Score: 167, Comments: 107](https://reddit.com/r/LocalLLaMA/comments/1hy8733/biden_to_further_limit_nvidia_ai_chip_exports_in/)): **Nvidia** AI 芯片出口面临 **Biden administration**（拜登政府）的额外限制，影响了包括**波兰、葡萄牙、印度**和**阿联酋**在内的美国盟友。此举旨在限制 AI 技术的出口，特别是对涉及 **Falcon 模型**的国家产生了影响。
  - 几位评论者批评 **Biden administration** 的政策**低效**且具有潜在危害，认为这可能导致**中国**与 **Tier 2** 国家之间加强合作，并可能无意中针对开源 AI 而非中国。人们还对全球技术发展和**美国地缘政治地位**受到的影响表示担忧。
  - 对于用于 AI 芯片出口国家分类的**分级系统（tier system）**存在困惑和不满，用户质疑将**葡萄牙**和**瑞士**归入 **Tier 2**，而**意大利**等其他国家则处于 **Tier 1** 的决定。**申根区（Schengen Area）**被提及为一个潜在漏洞，可能允许各国规避限制。
  - 讨论强调了由于这些限制，**NVIDIA 替代方案**获得关注的可能性，并对 **Nvidia 的芯片制造**地点提出了疑问，特别是关于台湾的 **TSMC** 及其对美中关系的影响。有人担心这些政策可能无法有效阻止中国等国家获得受限技术。


**主题 4. NVIDIA 的 Project Digits 承诺 AI 民主化**

- **[Project Digits：NVIDIA 价值 3,000 美元的 AI 超级计算机如何实现本地 AI 开发民主化 | Caveman Press](https://www.caveman.press/article/project-digits-nvidia-3000-ai-supercomputer-democratize-development)** ([Score: 113, Comments: 75](https://reddit.com/r/LocalLLaMA/comments/1hxuprn/project_digits_how_nvidias_3000_ai_supercomputer/)): **NVIDIA** 的 **Project Digits** 旨在通过提供价值 3,000 美元的 AI 超级计算机来实现本地 AI 开发的民主化。这一举措可以显著提高开发者和研究人员的可访问性，有可能改变本地计算能力。
  - 社区对 **NVIDIA Project Digits** 的**民主化**主张表示怀疑，认为它主要实现了部署的民主化而非训练。一些用户认为，真正的民主化需要开源 **CUDA**，并指出 **NVIDIA** 的基准测试使用的是 **fp4** 精度，低于 **fp32** 或 **fp16** 等典型标准。
  - 人们对**超级计算机**这一标签持怀疑态度，通过与现有的 **GPU** 和 **RAM 带宽**标准进行比较，认为 **Digits** 产品可能达不到预期。用户强调，已经存在具有更高 RAM 带宽和更宽 RAM 到 CPU 总线的竞争产品，例如具有 **546 GB/s** 的 **Apple M4 Max** 和具有 **460 GB/s** 的 **AMD EPYC**。
  - 讨论还集中在 **CUDA** 在机器学习中的作用，一些人主张使用更多像 **Triton** 这样与**供应商无关的解决方案**。虽然 **CUDA** 在开发新的 ML 技术方面仍然盛行，但人们正在推动支持多个供应商的框架，正如 **OpenAI** 和 **Triton** 所见，后者因其易用性和性能而正受到关注。

## 其他 AI Subreddit 回顾

> /r/Singularity, /r/Oobabooga, /r/MachineLearning, /r/OpenAI, /r/ClaudeAI, /r/StableDiffusion, /r/ChatGPT

**主题 1. DALL-E 被放弃：OpenAI 的多模态困境**

- **[OpenAI 是否完全放弃了 DALL·E？相同提示词下 DALL·E 和 Imagen3 的结果对比](https://www.reddit.com/gallery/1hxqhjw)** ([Score: 343, Comments: 136](https://reddit.com/r/OpenAI/comments/1hxqhjw/did_openai_abandon_dalle_completely_the_results/)): **OpenAI** 可能已经停止了对 **DALL·E** 的更新，正如使用相同提示词对 **DALL·E** 和 **Imagen3** 进行图像生成结果对比所显示的那样。讨论暗示 **DALL·E** 的性能没有得到改进或维护，引发了人们对 OpenAI 在该项目上投入精力的质疑。
  - 几位评论者推测了 **OpenAI DALL·E** 的未来，一些人认为随着图像生成领域竞争的加剧，OpenAI 可能会发布一个更新的或全新的模型，可能是多模态模型。**Vectoor** 和 **EarthquakeBass** 提到，过去版本的 **DALL·E** 在发布时具有开创性，但由于更新频率低，很快就落后了。
  - 评论中对 **DALL·E 3** 的审美和技术表现提出了批评，**COAGULOPATH** 和 **EarthquakeBass** 指出它无法生成令人信服的写实图像，这可能是由于 OpenAI 保守的安全立场。**Demigod123** 建议，这种卡通风格可能是一种防止滥用的刻意选择。
  - 讨论中提到了 **Midjourney**、**Flux Schnell** 和 **Mystic 2.5** 等替代方案，用户分享了他们生成的图像链接，强调了这些工具与 **DALL·E** 相比的能力。**Bloated_Plaid** 和 **MehmetTopal** 提供了视觉对比，表明其他工具目前可能提供更优的结果。


**主题 2. 微软展望组织中的 AI Agent 群体**

- **[微软 CEO 表示，每位员工很快将管理一个“AI Agent 群体”，每个组织内部将拥有“数十万个”Agent](https://v.redd.it/143088q6g1ce1)** ([Score: 235, Comments: 154](https://reddit.com/r/OpenAI/comments/1hxo7t8/microsoft_ceo_says_each_worker_will_soon_be/)): **Microsoft CEO** 预测，每位员工很快将管理一个 **AI Agent** “群体”，每个组织内部将部署“数十万个”此类 Agent。这一表态暗示了工作环境中 AI 集成和自动化的显著增加。
  - **对 AI Agent 管理的怀疑**：许多评论者对管理 **AI Agent** “群体”表示怀疑，质疑处理大量需要人工干预的 Agent 的实用性和潜在混乱。一些人认为这是 **Microsoft** 在没有交付实质性成果的情况下过度炒作技术的又一个例子。
  - **对就业和行业的影响**：讨论强调了对失业的担忧，担心 AI 将取代很大一部分劳动力，尤其是 **white-collar jobs**。关于工作的未来存在争论，一些人建议将任务从“白领”vs“蓝领”转向“可自动化”vs“不可自动化”。
  - **科技行业的战略手段**：评论者将 AI 集成与之前的技术策略（如 **Apple** 的生态系统固守）进行了类比。人们认为科技公司将使用类似的策略来锁定客户，使得从其 AI 解决方案中迁移出来的成本高昂且复杂。

---

# AI Discord 回顾

> 由 o1-mini-2024-09-12 生成的摘要之摘要

**主题 1. AI 模型对决：PHI-4 超越微软及其他模型**

- [**Unsloth 的 PHI-4 在 Open LLM Leaderboard 上超越微软**](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large)：来自 Unsloth 的 **PHI-4** 通过实施关键的错误修复和 **Llamafication**，在 Open LLM Leaderboard 上超越了 Microsoft 的基准，尽管量化变体有时会略胜非量化变体。

- [**rStar-Math 将 Qwen2.5 和 Phi3-mini 推向新高度**](https://x.com/altryne/status/1877220144725758414?s=46)：Microsoft 的 **rStar-Math** 将 **Qwen2.5** 在 MATH 基准测试中的表现从 **58.8%** 提升至 **90.0%**，将 **Phi3-mini** 从 **41.4%** 提升至 **86.4%**，标志着小型 LLM 在数学推理方面取得了重大进展。

- [**Llama 3.3 在低端硬件上表现挣扎，输出缓慢**](https://www.youtube.com/watch?v=PWgvGjAhvIw)：爱好者报告称 **Llama 3.3 70B Instruct** 在低端硬件上表现迟缓，在 **Ryzen 7** 和 **RX 7900GRE** 等中等配置系统上，Token 输出速度仅为 **0.5/sec**，突显了对强大 GPU 显存或系统 RAM 的需求。

**主题 2. AI 工具对决：Codeium、ComfyUI 和 Cursor IDE**

- [**Codeium 的私有化部署版本助力团队部署**](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides)：**Codeium** 在其企业版套餐中推出了私有化部署版本，吸引了渴望可定制、内部 AI 架构的团队，同时也需要应对额度处理的复杂细节。

- [**ComfyUI 通过 IP Adapter 魔力增强 AnimateDiff**](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use)：社区对 **AnimateDiff** 的输出质量提出了批评，转而采用集成 **IP Adapter** 的 [**ComfyUI 工作流**](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use)，以显著提升视频生成效果。

- [**Cursor IDE 规则强化 Claude 的代码编写能力**](https://dotcursorrules.com/)：开发者在 **Cursor IDE** 中使用 **.CursorRules** 来精确引导 **Claude** 的输出，显著减少了代码误改并确保了准确的功能实现。

**主题 3. GPU 难题与内核灾难：Linux 上的 Stable Diffusion**

- [**Linux 用户在 AMD GPUs 上运行 Stable Diffusion 时遭遇内核恐慌**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux)：在 **Linux** 上使用 **AMD GPUs** 运行 **Stable Diffusion** 的尝试有时会触发内核恐慌 (Kernel Panic)，但参考 [**AMD GPUs 安装 Wiki**](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux) 提供的修复方案可以解决 **Python** 版本问题。

- [**Stable SwarmUI vs A1111：用户界面之争**](https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md)：Discord 用户讨论了 **A1111**、**SwarmUI** 和 **ComfyUI** 的易用性，尽管 **SwarmUI** 的学习曲线被认为更陡峭，但其高级功能仍赢得了赞誉。

- [**MicroDiT 通过 DCAE 集成实现复现与优化**](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt)：**MicroDiT** 的成功复现提供了可下载的权重和[**推理脚本**](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb)，为使用 **DCAE** 进行架构增强以提升性能铺平了道路。

**主题 4. AI 社区动态：黑客松、招聘与融资热潮**

- [**oTTomator 的 AI Agent 黑客松触发 6000 美元奖金盛宴**](https://studio.ottomator.ai/hackathon/register)：**OpenRouter** 启动了 **oTTomator AI Agent 黑客松**，由赞助商 **Voiceflow** 和 **n8n** 提供总计 **6,000 美元** 的奖金，在 1 月 8 日至 1 月 22 日期间引发了激烈竞争。

- [**Anthropic 在 AI 创投飙升之际获得 20 亿美元融资**](https://x.com/andrewcurran_/status/1876705929296581078?s=46)：根据 [Andrew Curran 的报告](https://x.com/andrewcurran_/status/1876705929296581078)，**Anthropic** 额外筹集了 **20 亿美元**，将其估值提升至 **600 亿美元**，并巩固了其在企业级 AI 解决方案领域的地位。

- [**Nectar Social 提供 1 万美元悬赏以招揽 AI 人才**](https://www.linkedin.com/jobs/view/4120980579/)：位于西雅图的 AI 初创公司 **Nectar Social** 正在招聘**产品经理**和 **AI 工程师**，提供**高达 10,000 美元**的推荐奖金，以吸引优秀人才加入其不断发展的社交电商平台。

**主题 5. 高级 AI 技术：微调、解码与正则化难题**

- [**适配器并非儿戏：LoRA 精度至关重要**](https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426)：技术专家强调在实现 **LoRA 适配器**时使用 **16-bit 模型**的重要性，以防止输出质量下降，并主张将适配器与更高精度的基座进行合并。

- [**Speculative Decoding 成为资源节约英雄**](https://arxiv.org/abs/2501.04682)：为了减少下一个 Token 生成过程中的计算负载，社区将 **Speculative Decoding** 视为一种极具前景的技术，类似于语言模型的 **DLSS**。

- [**权重衰减之战：通过温和设置稳定 LLM**](https://arxiv.org/abs/2501.04697)：研究人员讨论了大型语言模型中**极端权重衰减**（例如 **0.1**）的影响，建议采用更温和的衰减以及诸如 *abs(norm(logits) - 1.0)* 之类的辅助损失函数，以防止模型崩溃并保持数值稳定性。

---

# PART 1: High level Discord summaries

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **GPU 性能与槽点**：工程师们对比了 **RTX 4070** 与 **3090** 在 AI 视频任务中的表现，指出 3090 可以在约 2 分钟内渲染 480p 视频；根据[此讨论](https://np.reddit.com/user/kejos92/comments/1hjkkmx/ltxv_inference_on_amd_gpus/)，AMD 显卡上的 LTXV 表现出更多差异。
   - 参与者交流了性能指标和优化技巧，倾向于使用专门的配置来实现更快的 **image-to-video** 工作流。
- **AnimateDiff 动态**：成员们批评 **AnimateDiff** 的输出效果欠佳，并参考了一个合并了 IP Adapter 以增强质量的 [ComfyUI 工作流](https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use)。
   - 他们还讨论了一个测试多种方法的 [image-to-video 对比选项](https://civitai.com/models/548997/image-to-video-comparison-workflow)，并指出某些步骤的运行时间仍然超出了预期。
- **Discord 争议**：用户举报了**不当个人资料**，并就加强监管以保持对话文明进行了争论。
   - 担忧主要集中在如何平衡清理毒性内容与维持友好环境之间。
- **界面之争**：对 **A1111**、**SwarmUI** 和 **ComfyUI** 的对比显示了对用户友好度的不同看法，SwarmUI 的功能记录在[此 GitHub 指南](https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md)中。
   - 虽然 A1111 因简单易用受到称赞，但一些人更欣赏 ComfyUI 用于**动画**内容创作的高级流水线。
- **内核恐慌**：基于 Linux 的 **Stable Diffusion** 偶尔会触发内核恐慌（kernel panics），为此参考了 [AMD GPUs 安装维基](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux)。
   - 指南和修复方案通常针对 Python 版本问题，为 Linux 上更顺畅的 AI 工作流提供备选方案。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Unsloth 的 PHI-4 实力超越 Microsoft**：在官方 [Open LLM Leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large) 评分中，得益于 Bug 修复和 Llamafication，来自 Unsloth 的 **PHI-4** 刚刚超过了 Microsoft 的基准线。
   - 社区成员称赞了这一改进，但指出**量化变体**有时比非量化配置表现更好。
- **Adapter 态度：精度决定成败**：专家强调在挂载 Adapter 时应使用 **16-bit** 模型以获得更好的吞吐量，并参考了一篇 [LoRA 注意事项文章](https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426)。
   - 他们提到使用较低精度可能会降低结果质量，通常更倾向于与高精度基座模型进行合并。
- **Chat Templates 调整 LLM 行为**：贡献者讨论了 `tokenizer_config.json` 中的 **chat templates** 如何塑造输入输出格式，从而显著影响 LLM 性能。
   - 他们强调从训练到生产保持模板一致性可以确保结果稳定，有人声称这能“决定**部署成功**与否”。
- **Speculative Decoding：减少资源消耗的关键技巧**：关于语言模型类 **DLSS** 优化的讨论引出了 *speculative decoding*，它被誉为一种资源友好型技术。
   - 研究人员发现它在绕过 next-token generation 中沉重的计算负载方面很有前景。
- **Mathstral 7B 等待更广泛支持**：澄清了 `mistralai/Mathstral-7B-v0.1` 模型目前**不支持**直接微调，因为它不是标准的基座模型或 PEFT 模型。
   - 参与者表示支持即将推出，引发了对未来模型合并和扩展的谨慎乐观。

---

## [Codeium (Windsurf)](https://discord.com/channels/1027685395649015980) Discord

- **自托管 Codeium 增强部署控制**：成员们注意到 **Codeium** 的自托管版本现已包含在企业版套餐中，吸引了渴望管理自己环境的团队的关注。
   - 他们还提出了关于额度处理和 **Windsurf** 功能的问题，并指向了 [价格页面](https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides) 以获取官方指南。
- **Windsurf 在多个发行版上的安装挑战**：用户报告在 Mint, Ubuntu 24.04, Arch 和 Hyprland 上成功安装了 **Windsurf**，有时需要删除配置文件以解决奇怪的错误。
   - 他们还讨论了在多台 PC 上共享 **Cascade** 聊天的愿望，虽然出现了云同步的建议，但目前尚未推出官方功能。
- **Flow Credits 计费问题引发不满**：几个人抱怨多次支付 Flow Credits 但从未看到到账，呼吁制定更清晰的使用政策。
   - 他们还质疑 **internal errors** 是否计入额度，敦促开发者迅速修复这些扣费问题。
- **Cascade 的无代码胜利与聊天管理愿景**：一位用户强调了在几乎没有实际编码的情况下构建了公司网站，并赞扬了 Cascade 免费层级中的 **unlimited** 查询。
   - 其他人仍然希望在多个设备上实现更好的 **Cascade** 聊天管理，表示需要官方的同步解决方案。



---



## [Cursor IDE](https://discord.com/channels/1074847526655643750) Discord

- **Cursor Rules 驯服混乱代码**：开发者分享了如何使用 [.CursorRules](https://dotcursorrules.com/) 通过结构化提示词优化 **Claude** 的输出，重点关注明确的目标以避免意外的文件更改。
   - 他们报告称，精确选择的关键词显著减少了代码误改，强调了定义良好的提示词指令的重要性。
- **Cursor Directory 受到关注**：[Cursor Directory](https://cursor.directory/) 的关注度激增，突显了其收集各种框架的社区规则的能力。
   - 用户赞赏这种集中分享规则的方式，指出在处理特殊配置时节省了时间和精力。



---



## [Stackblitz (Bolt.new)](https://discord.com/channels/364486390102097930) Discord

- **灵活的颜色提示词**：社区强调在提示词中使用颜色名称和十六进制代码来指定 **colors**，以确保使用清晰。
   - 一位成员用更精确的指南取代了模糊的请求（如 *Just do blue and white*），以控制跨应用的样式。
- **支付系统故障**：一位用户发现他们的 **payment system** 无法运行，发布了项目链接并寻求帮助。
   - 他们提到正在积极开发以恢复全部功能，并敦促测试人员提供反馈。
- **使用 Bolt.new 打开公共仓库**：开发者宣布了 **public repos** 功能，允许用户在任何 GitHub URL 前加上 [http://bolt.new](http://bolt.new) 前缀以立即访问。
   - 他们引用了一篇 [X 帖子](https://x.com/stackblitz/status/1843668731681267801)，展示了如何以最少的配置打开仓库。
- **Bolt Token 过度消耗**：多人报告在编辑或调试时 **token 消耗过快**，并遇到了重复尝试修复错误的情况。
   - 他们对持续的资源消耗表示沮丧，希望能有更高效的方法。
- **Supabase 迁移与 Netlify 故障**：开发者提到，如果在更新过程中出现问题，回滚 **Supabase migrations** 会非常麻烦，从而影响应用程序的稳定性。
   - 此外，一位用户提到 **Netlify** 加载速度缓慢，怀疑是免费层级限制或 **Bolt** 代码效率低下。



---

## [aider (Paul Gauthier)](https://discord.com/channels/1131200896827654144) Discord

- **Gemini 2.0 迈向移动端与语音交互**：一位用户在处理杂务时，在 iOS 上测试了 **Gemini 2.0 Flash Experimental** 的**语音模式**，实时构思了一个应用想法，并在返回后生成了简洁的任务列表。
   - 社区成员对 **Gemini 2.0** *自主提出项目标准*的能力表示赞赏，称其为迈向无摩擦开发（frictionless development）的有力一步。
- **Tier 5 密钥获取尝试与 Unify.ai 技巧**：讨论集中在昂贵的 **Tier 5 OpenAI** 访问权限的替代方案上，并提到 [Unify.ai](https://unify.ai/) 和其 [GitHub 仓库](https://github.com/unifyai/unify) 是灵活的多模型解决方案。
   - 成员们权衡了**订阅成本**，并分享了使用 **OpenRouter** 和 **Unify** 来简化配置的经验。
- **Aider 与 Claude 在编程领域展开对决**：多位用户将 **Aider** 不稳定的文件编辑和偶尔的失误与 **Claude** 进行了对比，并提到了整个文件被删除的滑稽事件，引用了[文件编辑问题](https://aider.chat/docs/troubleshooting/edit-errors.html)文档。
   - 一些人认为 **DeepSeek** 聊天过于容易分心且表现*懒惰*，而另一些人则认为如果谨慎管理以避免大规模代码删除，**Aider** 仍然是可用的。
- **更强大 AI Agent 的愿景**：一位用户预测 **AI** 最终将创造出自身的改进迭代并最大限度地减少人工干预，但这一观点因对**计算成本**和运营开销的担忧而有所保留。
   - 参与者强调了硬件和资源可用性的**当前局限性**，对自主 AI 能力的短期扩张既表达了乐观也保持了谨慎。

---

## [Notebook LM Discord](https://discord.com/channels/1124402182171672732) Discord

- **DeepResearch 在 NotebookLM 中取得进展**：成员们建议将 **DeepResearch** 引用集成到 [NotebookLM](https://notebooklm.google.) 中，旨在合并现有报告的输出，尽管目前还没有官方插件。
   - 一些用户提到了通过“批量上传”工作路经将大型数据集导入 **NotebookLM**，这激发了人们对更强大协同效应的期待。
- **AI 音频生成引发热潮**：参与者探索了利用精选的 NotebookLM 来源构建播客，并将其与 [Illuminate](https://illuminate.google.com/create) 结合使用，以获得音频灵活性和更好的来源针对性。
   - 他们称赞了限制来源的提示词（source-limited prompts）在控制风格方面的作用，而其他人则提到 **Jellypod** 是一个具有更广泛自定义选项的潜在替代方案。
- **跨语言播客展示 NotebookLM 的灵活性**：一些用户尝试在 [NotebookLM](https://notebooklm.google.) 内部根据**英文**内容生成**中文**播客脚本，并应用口语化改写策略以实现自然流畅。
   - 他们还测试了**日文**对话，指出准确的音译可能需要额外的检查，但这反映了用户对切换语言的适应度。
- **引用模式与系统提示词困惑**：开发者在 **NotebookLM** 中引入了“仅限引用（quotation-only）”命令，确保直接从源文件中摘录内容，并对重要引用进行更严格的验证。
   - 然而，**Gemini** 偶尔会返回不完整的引用，引发了关于在 **NotebookLM Plus** 中改进系统提示词（system prompts）以获得一致结果的讨论。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **通义千问（Qwen）奇特的聊天热潮**：阿里巴巴推出了 [Qwen Chat](https://chat.qwenlm.ai)，这是一个为 **Qwen** 模型设计的新 Web UI，提供文档上传和视觉理解功能，并在[其推文](https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113)中进行了预告。
   - 社区期待即将推出的**网页搜索**和**图像生成**等功能，将 Qwen Chat 视为不断发展的 LLM 生态系统中的关键竞争者。
- **AMD 7900 的性能难题**：用户通过一篇 [reddit 帖子](https://reddit.com) 将 **AMD RX 7900XT 20GB** 与 **NVIDIA 3090** 进行了对比，认为 7900XT 在处理 LLM 任务时可能面临显存带宽限制。
   - 其他人则认为 7900XT 在本地推理方面表现尚可，尽管他们在某些基准测试中看到 **3090** 的表现更稳定。
- **Llama 3.3 的内存乱象**：爱好者报告称，**Llama 3.3 70B Instruct** 在 **Ryzen 7** 和 **RX 7900GRE** 等低端硬件上的输出速度极慢，仅为 **0.5 token/sec**。
   - 他们强调需要大量的 GPU 显存或系统 RAM 来避免这些减速，并维持大规模的 token 吞吐量。
- **NVIDIA DIGITS 引发好奇**：社区讨论转向了 **DIGITS**，传闻它是 **NVIDIA** 工作流中用于训练和测试模型的强大解决方案。
   - 用户对性能开销保持谨慎，但期待它能成为本地 LLM 工具箱中的一个强大补充。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **O1 的“思考”特性与 A/B 测试暗示**：一位参与者指出了 **Model O1** 独特的“思考”输出，暗示可能涉及不同的模型格式。
   - 他们提出了并行运行 **Model 4O** 的可能性，反映了对比较多种性能方案的热情。
- **Meta-Prompting 激发灵感**：成员们强调了 **Meta-Prompting** 策略，提到调整系统消息可以生成更高级的响应。
   - 他们强调，在编写提示词时，在开始阶段明确目标会带来更精准的 **model outputs**。
- **Hassabis 的投资轮**：小组为 **Hassabis** 的投资轮送上祝福，认可了新资金在 AI 追求中的重要性。
   - 他们称赞了他的过往战绩，并指出支持性的资金可以推动该领域进一步的 **R&D** 工作。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **ICLR：聚会热潮**：与会者正在为 **ICLR** 做准备，交流旅行的兴奋感并实时计划潜在的聚会。
   - 他们期待活跃的面对面交流，**Philpax** 很快将穿着**浅褐色外套**和**黑色牛仔裤**抵达，准备讨论新的模型突破。
- **rStar 崛起：Qwen2.5 与 Phi3-mini 飙升**：**Microsoft** 的 **rStar-Math** 将 **Qwen2.5** 在 MATH 基准测试上的表现从 **58.8%** 提升至 **90.0%**，将 **Phi3-mini** 从 **41.4%** 提升至 **86.4%**。
   - 它目前在全美数学奥林匹克竞赛（USA Math Olympiad）中平均得分为 **53.3%**，引发了人们对 *Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking* 的关注以获取更深层的见解。
- **NuminaMath 与质量难题**：由于约 7.7% 的条目包含多个相互矛盾的解决方案，人们对 **NuminaMath** 的怀疑增加，这指向了更广泛的数据问题。
   - 成员们还引用了 [“Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought”](https://arxiv.org/abs/2501.04682)，并注意到第一作者在斯坦福大学的心理学背景引起了关注。
- **开源 AI：成本冲突**：政策制定者对**开源 AI** 成本仅为 **$5M** 表示担忧，引发了对实际预算的困惑。
   - 一条推文的成本明细排除了资本和数据支出，因误导 GPU 小时数统计而遭到批评。
- **Anthropic 的早期角色塑造**：在一次 **Anthropic** 沙龙上，**Josh Batson** 表示 **Amanda Askell** 将基础模型塑造为 Agent 的时间比一些人预期的要早。
   - 关于角色对齐（character alignment）是训练后的附加组件还是内置过程引发了辩论，对 [Anthropic Research Salon](https://youtu.be/IPmt8b-qLgk) 的引用进一步推动了对话。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **SmolLM 分片风暴**：**SmolLM Corpus** 飙升至 **320GB**，分为 **23698** 个分片，承诺采用更高效的 `.jsonl.zst` 格式，该格式要到本周晚些时候才能最终确定。
   - 成员们称赞其体积比 **1TB** 的未压缩数据集小得多，并提到 HPC 的便利性以及“减少迭代训练流水线的开销”。
- **Modal 的强力举措**：爱好者们探索了 **Modal**、**Colab** 和 **Kaggle** 等高性价比的训练和分析方案，重点关注 **Modal** 的每月额度是处理大型任务的可靠方式。
   - 他们注意到 **Modal** 可以运行超出个人 GPU 容量的任务，并赞赏其对大规模推理的稳定支持。
- **SciAgents 影响 AI 圈**：[**SciAgents** 论文](https://arxiv.org/abs/2409.05556) 采用**本体知识图谱**和 **multi-agent** 方法来增强研究操作，将结构化数据与 Agent 协作交织在一起。
   - 有些人认为这个概念并不是巨大的飞跃，但其他人喜欢这种编排方法，称其为高级学习工作流中一个很有前景的框架。
- **Grokking 势头强劲**：成员们剖析了 [**Grokking at the Edge of Numerical Stability**](https://arxiv.org/abs/2501.04697)，强调了深度网络中的**延迟泛化**和 **softmax 崩溃**。
   - 他们强调，不足的 **regularization** 会导致模型崩溃，敦促尽早进行干预并“抑制失控的 logits”。
- **权重衰减与 Llama2 HPC 困境**：几位研究人员讨论了 LLM 中**极端的权重衰减**（如 **0.1**），建议对注意力层和辅助损失（例如 *abs(norm(logits) - 1.0)*）采用更温和的设置。
   - 与此同时，尝试使用 `model_parallel=2` 预训练 **7B** Llama2 在 batch size 为 1 时触发了 **OOM** 停滞，促使进行显存分析并对 **6.7B** 配置进行新测试。

## [GPU MODE](https://discord.com/channels/1189498204333543425) Discord

- **WGMMA & Triton 试验**：工程师们讨论了 **WGMMA** 需要在 4 个 warp 之间进行拆分，且最小 tile 为 64，并参考了 [NVlabs 的 tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 以获取关于 **fused MLP** 的见解。他们还推荐使用 **Proton** 进行性能分析（profiling），并引用了[一段有用的视频](https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb)。
   - 社区成员称赞使用 Proton 调试 **Triton** kernel 更加容易，同时也对典型 HPC 任务中使用片上（on-chip）MLP 表示疑问。
- **MI210 占用率之谜**：成员们研究了 **MI210** 和 **RX 7900XTX** 的 **GPU occupancy**，参考了[一篇关于 block 级优化的资源](https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/)。他们注意到可能达到 16-warps 的占用率，但在实际代码中看到了诸如 block 级资源使用等限制。
   - 他们得出结论，达到更高的占用率通常需要多个 kernel，而 **CDNA** 架构细节揭示了实际的 block 限制和早期退出（early exit）行为。进一步的测试验证了 **MI210** 上独特的 block 调度方法。
- **Nectar Social 的 1 万美元悬赏**：**Nectar Social** 正在 **Seattle** 招聘 **Staff Product Manager**、**LLM/AI Engineer** 和 **Infra Engineer**，并提供高达 **$10,000** 的推荐奖金。他们强调了先前的初创公司经验，并表示愿意私下分享细节。
   - 一家拥有 **AMD** 等 HPC 客户的欧洲咨询公司也在寻找精通 **CUDA**、**HIP** 和 **OpenCL** 的开发人员，参考了 [LinkedIn](https://www.linkedin.com/jobs/view/4120980579/) 上的职位列表。他们还合作开发 **rocPRIM** 和 **hipCUB** 等库，旨在填补专业的 GPU 开发人员职位。
- **ARC Prize 向非营利组织转型**：正如 [François Chollet 的推文](https://x.com/fchollet/status/1877069518171943000)所示，**ARC Prize** 正在转型为非营利基金会，并由新任主席指导 AGI 研究。他们还启动了一个 **rejection sampling** 基准实验，以建立基础指标。
   - 社区成员探索了 **text-domain** 解决方案以缓解 GPU 限制，并分析了 **Meta CoT paper** ([链接](https://arxiv.org/abs/2501.04682)) 以寻求潜在的改进。作者强调了经典 CoT 方法的不足，引发了关于上下文推理的更广泛讨论。
- **MicroDiT 结合 DCAE 的进展**：**MicroDiT** 的复现圆满结束，提供了[权重文件](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt)和[推理脚本](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb)。他们对计算支持表示感谢，并旨在通过 **DCAE** 改进架构以获得更强的性能。
   - 计划包括采用 **MMDIT** 以实现更好的 prompt 遵循能力，并寻求**计算资助（compute grants）**。有限的家用 GPU 容量阻碍了高级 AI 实验，促使人们寻找额外资源。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **Microsoft 的 rStar-Math 助力 Qwen 展现统治力**：Microsoft 推出了 [rStar-Math](https://x.com/altryne/status/1877220144725758414)，将 **Qwen 2.5-Math-7B** 在 MATH 基准测试中的表现从 **58.8%** 提升至 **90.0%**，并在 AIME 中获得 **53.3%** 的分数，位列高中生前 20%。
   - 成员们讨论了**数学**能力对**推理**技能的重要性，一些人提醒说，数值上的突破并不总能保证更广泛的 **LLM** 可靠性。
- **DistTrO 敞开大门**：一位成员确认 **DistTrO** 已开源，引发了社区训练器的立即集成。
   - 贡献者称赞 **DisTrO** 的分布式训练简便性，一些人强调其设置比早期的解决方案更顺畅。
- **Carson Poole 的论文展示**：Carson Poole 介绍了 [ReLoRA](https://arxiv.org/abs/2307.05695) 和 [Sparse Upcycling](https://arxiv.org/abs/2212.05055)，并引用了 **2022 年 11 月**和 **2023 年 3 月**的讨论。
   - 他敦促成员访问[他的个人网站](https://poole.ai)，并提议在 **Forefront.ai** 或 **Simple AI Software** 上通过邮件协作进行更深入的探索。
- **DeepSeek V3 的双重测试**：成员们将 **DeepSeek V3** 官方 API 的重复输出与 **Hyperbolic** 等第三方提供商进行了对比，注意到答案质量存在显著差异。
   - 一些人将这些不一致归因于**激进的缓存（aggressive caching）**，引发了对更一致推理方法的兴趣。
- **Qwen2.5 在 24 GB VRAM 上的内存迷宫**：一位用户在 **RTX 4090** 上运行 **Qwen2.5-32B-Instruct-AWQ** 时遇到了显存溢出错误，尽管启用了 **flash attention**。
   - 讨论转向了针对约 6K **token** 上下文的潜在**内存使用**优化，以及对开源 **function calling** 准确性基准测试的咨询。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Salesforce 令人惊讶的招聘冻结**：Marc Benioff 确认 [Salesforce](https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/) 将在 **2025 年**停止招聘软件工程师，理由是 **Agentforce** 带来了 **30% 的生产力提升**。
   - 社区成员认为这是资源分配的重大转变，有人推测 *“AI 确实正在接管基础软件任务”*。
- **OpenAI 调整自定义指令**：据报道，OpenAI 更新了其 **advanced voice** 工具集的自定义指令，[来自 topmass 的推文](https://x.com/topmass/status/1877444315871326422)展示了部分功能损坏以及新功能的迹象。
   - 观察者认为这些改进可能会带来新的语音功能，一位用户将其描述为 *“为更流畅的 AI 体验提供的强大增强”*。
- **Anthropic 获得 20 亿美元注资**：根据 [Andrew Curran 的报告](https://x.com/andrewcurran_/status/1876705929296581078)，Anthropic 正在以 **600 亿美元**的估值筹集 **20 亿美元**，其 **ARR** 达到 **8.75 亿美元**。
   - 参与者评论道 *“风险投资对 AI 解决方案有巨大的胃口”*，特别是随着 Anthropic 在企业合同中的吸引力持续扩大。
- **Google 将 AI 整合进 DeepMind**：Google 宣布计划将多个 AI 产品合并到 **DeepMind** 旗下，[Omar Sanseviero 的推文](https://x.com/osanseviero/status/1877452798683430988)展示了双方在 **2025 年**联手的计划。
   - 评论者预见到公司结构可能存在重叠，称其为 *“令人费解的重组，但希望能简化 LLM 产品线”*。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **现金奖励助力黑客松热潮**：**oTTomator AI Agent Hackathon** 提供 6,000 美元的赞助奖金，第一名奖励 1,500 美元，亚军奖励 150 美元，外加 **10 美元的 OpenRouter API 额度**，报名地址见 [此处注册](https://studio.ottomator.ai/hackathon/register)。
   - 活动时间为 1 月 8 日至 1 月 22 日，社区投票时间为 1 月 26 日至 2 月 1 日，赞助商包括 **Voiceflow** 和 **n8n**，分别提供额外的 700 美元和 300 美元奖金。
- **OpenRouter UI 在超过 1000 行时出现卡顿**：用户报告称 **OpenRouter UI** 在聊天记录超过 1000 行后运行速度大幅下降，导致滚动和编辑变得非常痛苦。
   - 他们提出了按成本排序和 **Next.js 分页**等改进方案，以解决这些性能陷阱。
- **Gemini Flash 引发困惑**：**Gemini Flash** 引擎在聊天室中可以工作，但通过 API 调用似乎无法运行，这让多位用户感到困惑。
   - 另一位用户总体上赞扬了 **Gemini**，但也指出了需要立即改进的性能问题。
- **O1 采用不寻常的响应格式**：开发者注意到 **O1 的响应** 使用 '====' 代替了 Markdown 的反引号，引发了对格式异常的担忧。
   - 讨论范围涵盖了这一举措是为了减少 Token 使用量还是为了优化输出，引发了关于最佳实践的辩论。
- **API 访问与 Hanami 测试**：开发者询问了如何通过 OpenRouter 提供他们自己的 **LLM API**，并分享了请求处理方面的问题。
   - 另一位用户测试了 **Hanami** 但遇到了奇怪的字符，强调了强大的工具兼容性的重要性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **CSV 导出功能受到关注**：Perplexity 推出了将表格响应下载为 **CSV 文件**的功能，详见[说明图片](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg)。
   - 用户欢迎这一增强功能，认为它**简化了数据工作流**，并强调了在处理大型数据集时如何节省时间。
- **Youzu.ai 室内设计取得进展**：**Youzu.ai** 提供 AI 驱动的房间设计及直接购物选项，详见此 [Medium 指南](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688)。
   - 社区成员对其进行了测试，并赞赏其**减少麻烦**的潜力，同时征求关于实际使用的反馈。
- **丰田的火箭之约**：来自**丰田**的一项新[火箭风险投资](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q)表明，他们正在向标准汽车工程之外的领域进军。
   - 爱好者们注意到了丰田成熟的专业知识与**航空航天需求**之间的协同作用，预测随后会有更多官方细节。
- **NVIDIA 价值 3000 美元的家用超级计算机**：**NVIDIA** 在 [CES 2025 参考资料](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw)中宣布了一款未来的家用级超级计算机，价格为 **3000 美元**。
   - 科技粉丝们辩论了先进的性能是否与其**成本**相符，认为这是在家中进行机器学习实验的一个契机。
- **Ecosia 寻求与 Perplexity 合作**：**Ecosia 的一名产品经理**在联系 Perplexity 洽谈潜在合作时遇到困难，并寻求联系方式的指导。
   - 社区中的热心人士提供了**直接沟通**的建议，希望如果讨论能继续推进，双方能达成富有成效的联盟。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Cohere 的 'North' 崛起，挑战 Copilot**：Cohere 推出了 [**North** 的早期访问版](https://x.com/cohere/status/1877335657908949189)，这是一个安全的 AI 工作区，将 **LLMs**、**search** 和 **agents** 融合到一个界面中以提升生产力。
   - 正如[官方博客文章](https://cohere.com/blog/north-eap)所述，他们认为它可以超越 **Microsoft Copilot** 和 **Google Vertex AI**。
- **Command R+ 推动生成式收益**：一位用户在探索 Cohere 生态系统中大型生成式模型的工作流时提到了 [**Command R+**](https://docs.cohere.com/docs/models)。
   - 社区讨论强调了清晰的集成策略，并认识到需要结构良好的 prompts 来优化模型行为。
- **v2 到 v3 Embeddings：升级疑问**：关于如何从 **embed-v2** 过渡到 **v3** 而无需重新对海量数据集进行 embedding 的问题被提出，引发了对资源消耗的担忧。
   - 成员们寻求一种高效的方法，在保持性能的同时最大限度地减少开销和潜在的停机时间。
- **LLM 循环与滚动聊天：驯服 Token 溢出**：报告显示 **Cohere** 的 **LLM** 可能会陷入重复循环，导致 **Python ClientV2** 设置中出现失控的 token 使用。
   - 建议包括设置 **max_tokens** 限制，并采用**滚动聊天历史（rolling chat history）**技术来处理 4k token 限制内的扩展响应。
- **Alignment Evals Hackathon 激发行动**：宣布将于 25 日举行 **Alignment Evals Hackathon**，届时将提供社区驱动的评估（eval）和解释教程。
   - 鼓励参与者分享见解和成果，推动对齐评估方法上的协作。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **Mock GPU 混乱与悬赏讨论**：针对涉及 macOS 上 MOCKGPU 的 [Pull Request #8505](https://github.com/tinygrad/tinygrad/pull/8505) 请求进行重新测试，**George Hotz** 确认已为该修复准备好悬赏。
   - 他提议通过 **PayPal** 或以太坊上的 **USDC** 支付，强调了处理 **tinygrad** 待办任务的决心。
- **LLVM JIT 与 Autogen 结对**：成员们提议合并他们的 **LLVM JIT** 和 **LLVM Autogen** 工作，并引用了多个版本文件的更改。
   - 他们还辩论了前向与后向兼容性，一些人强调支持旧版 LLVM 以避免静默损坏（silent breakage）。
- **函数签名稳定性摩擦**：有人担心 **LLVM** 函数签名的潜在静默更改会导致未定义行为。
   - **George Hotz** 淡化了这一风险，指出更倾向于支持旧的 LLVM 版本以保持一致性。
- **TinyGrad 博客与设备设置**：一篇题为 [TinyGrad Codebase Explained-ish](https://adelaloui.me/tinygrad-codebase-explained-ish/) 的博客文章介绍了 **tinygrad** 的文件布局，并对 **tinygrad/** 目录之外测试较少的代码提出了警告。
   - 一位用户询问如何在特定硬件上初始化权重，并得到了在创建 tensors 之前将 `Device.DEFAULT` 设置为 **METAL**、**CUDA** 或 **CLANG** 的建议。

---

## [Nomic.ai (GPT4All)](https://discord.com/channels/1076964370942267462) Discord

- **Llama 与 GPT4All 之争升温**：他们强调 **Llama.cpp** Vulkan 与 **GPT4All** 内部机制不同，由于使用了 CUDA，在 **Nvidia GPUs** 上产生了巨大的速度差距。
   - 参与者得出结论，如果性能满足日常目标，这种差异可以忽略不计，并参考了 [nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/issues/3365) 以获取更多背景信息。
- **Chat Template 纠葛**：一位用户在 **GPT4All** 上使用 TheBloke 的模型时遇到了 **Chat Template** 问题，尽管安装正确，但仍收到通用回复。
   - 其他人建议查看 GitHub 上的模型特定说明，强调 **chat prompts** 在不同模型之间差异很大。
- **基于 Llama-3 的角色扮演推荐**：对于动漫主题的角色扮演，成员们推荐了 **Nous Hermes 2** 或 [llama3-8B-DarkIdol-2.2-Uncensored-1048K](https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K) 作为可行的旧选项。
   - 他们指出 Nomic 的即插即用方法简化了使用，特别是对于快速的脚本化对话。
- **ModernBERT 部署困境**：有人询问 Nomic AI 的 **ModernBERT** 是否支持 **text-embedding-inference** 或 **vLLM**。
   - 目前还没有确切的答案，这让小组对官方部署渠道感到不确定。
- **图像模型希望点燃 GPT4All 讨论**：一些人考虑将**图像模型**加入 GPT4All 以扩展模态覆盖范围。
   - 对话在没有明确计划的情况下结束，但强调了用户对桥接文本与视觉的兴趣。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **GitHub 总部大聚会**：1 月 15 日，他们将在 [GitHub HQ](https://twitter.com/llama_index/status/1877103276635848846) 举办一系列专家演讲，讨论 **AI Agent 改进**、**快速推理系统**以及使用 **LlamaIndex** 构建工作流。
   - 该活动展示了先进的 Agentic 工作流，重点介绍了来自多位行业专家的真实案例。
- **Agentic 文档工作流：一次大胆飞跃**：一篇新的 [博客文章](https://twitter.com/llama_index/status/1877420085691953385) 介绍了 **Agentic Document Workflows (ADW)**，旨在将文档处理直接集成到业务流程中。
   - 它强调 **文档有多种格式**，并为未来驱动的应用提供了一种流线型方法。
- **Ollama 提速**：**Ollama** 的最新更新将评估时间缩短至 **3 秒**以内，引发了用户的兴奋。
   - 一位用户称其提升是*令人难以置信的*，反映了对更快模型推理的强烈热情。
- **VectorStoreIndex：元数据的手动操作**：一些成员讨论了在 **Postgres** JSON 字段中使用 **VectorStoreIndex** 按元数据键过滤节点，询问是否可以避免手动索引。
   - 他们得出结论，由于 LlamaIndex 尚未处理所有相关的自动化，可能仍需要 **手动索引**。
- **驯服 TEI 和 QueryFusionRetriever 的怪癖**：使用 **本地 TEI 服务器** 进行 Reranking 的兴趣有所增加，参考了 [API 文档](https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/) 和 [源代码](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py)。
   - 与此同时，用户在 **QueryFusionRetriever** 中遇到了 518 个 Token 处的 **输入验证错误**，并分享了代码片段以寻找解决方法。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Rust 语法简化多行**：一位用户在为 **multipaxos** 构建 Actor 时称赞了 **Rust** 的多行语法，强调其类型检查更少。
   - 他们表示*函数参数可能变得繁琐*，给理清所需类型的用户带来困惑。
- **重载解析变得冒险**：一位用户警告说，在大型代码库中重新排列重载可能会导致新的障碍，建议采用 **'happens after'** 注解方法。
   - 他们补充说 *TraitVariant 检查可能与实现 Trait 混淆*，可能导致混乱的重载解析。
- **Mojo 中的量子库进展**：一位成员提到需要一个 **类 Qiskit** 的库，提到了对量子扩展的兴趣，并链接到了 [MLIR 开发视频](https://some.link)。
   - 他们建议 **MAX** 随着发展可能很快就能处理量子任务。
- **MAX 支持量子编程**：讨论聚焦于 **MAX** 作为 Mojo 微调量子例程的合作伙伴，允许实时硬件调整。
   - 人们表示，当 MAX 成熟时，它可以统一量子和经典逻辑。
- **Quojo 提供量子选项**：通过 [GitHub](https://github.com/Deftioon/Quojo) 分享的 **Quojo** 库被提及为 Mojo 中的量子计算工具。
   - 大家对推动量子编程发展的*新兴开发者*表示兴奋。

---

## [LLM Agents (Berkeley MOOC)](https://discord.com/channels/1280234300012494859) Discord

- **黑客松时间表成型**：**黑客松网站** ([链接](https://rdi.berkeley.edu/llm-agents-hackathon/)) 分享了更新后的结果公布时间表，将最终结果推迟到 **1 月下旬**。
   - 组织者表示，仍有几位评委需要完成评审，承诺在宣布获胜者之前进行彻底评估。
- **评委们欢欣鼓舞**：评委们对 **黑客松提交作品** 给予了极高的评价，称其整体为*令人印象深刻的作品*。
   - 他们强调了极高的创意水平和技术深度，增强了对最终裁决的期待。

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **OpenInterpreter 1.0 在 Python 执行方面遇到困难**：用户发现 **OpenInterpreter 1.0** 不能直接运行 Python 代码，导致对 `--tools interpreter` 命令产生困惑。
   - 一位成员对代码执行的限制表示沮丧，引发了对如何处理代码块提供**更清晰指令**的请求。
- **GPT-4o-mini 获得部分命令控制能力**：讨论指出 **GPT-4o-mini** 在命令处理方面有所改进，特别是在分块打印文件内容时。
   - 对话集中在通过更好的文件输出策略和微调命令执行来优化模型性能。
- **征集模型规格参数**：一位成员询问了关于参数量和底层框架的更多**技术细节**，以便更好地理解性能指标。
   - 这一询问强调了对**完整文档**的需求，因为参与者正在寻求关于模型构建模块的清晰说明。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **TruLie 还是假象？数据之谜**：一位用户询问了 **TruLie 数据集**，但未提供具体细节或引用，导致对话缺乏事实依据。
   - 成员们对可能的研究应用表示好奇，但尚未出现直接的资源。
- **Chirpy3D 取得进展**：爱好者们讨论了 **image-to-3D** 的进展，重点介绍了用于连续鸟类生成的 **Chirpy3D** 和 **Gaussian splats** 方法，并引用了 [Chirpy3D](https://kamwoh.github.io/chirpy3d/) 作为示例。
   - 他们提到了来自多个机构的合作，并指出 Hugging Face 上的 [3D Arena](https://huggingface.co/spaces/dylanebert/3d-arena) 是 **NeRF** 库的资源。
- **World Models 进化视觉效果**：贡献者分享了使用物理感知网络生成更真实视频内容的 **World Models**。
   - 虽然超出了纯粹的 image-to-3D 流程，但这一方向与构建复杂视觉系统的更广泛努力相一致。
- **寻求开放工具注册表**：一位研究人员请求建立一个用于**构建 Agent** 的开放工具注册表，希望能收集小组的建议。
   - 目前尚未出现直接线索，促使进一步尝试寻找完整的资源。

---

## [DSPy](https://discord.com/channels/1161519468141355160) Discord

- **聊天机器人的 Chain-of-Thought 增强**：一位用户询问如何将聊天机器人的 **Chain-of-Thought** 提升到简单的角色签名（persona signature）之外，寻求更深层次的对话风格和推理步骤。
   - 该问题未得到解答，凸显了优化**聊天机器人**逻辑和用户交互的难度。
- **使用 DSPy 进行评估尝试**：一篇关于构建自定义评估的文章，题为《构建自定义评估简介：为什么它很重要以及 DSPy 如何提供帮助》，被分享在[这里](https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html)，以强调 **DSPy** 在定制测试框架中的作用。
   - 读者对开发新的评估方法表现出兴奋，并认可了 DSPy 在改进**知识库（knowledge bank）**解决方案方面的潜力。
- **人类学与技术：Drew 的路径**：Drew Breunig 概述了他在**文化人类学**、软件和媒体方面的背景，提到了在 **PlaceIQ** 和 **Precisely** 从事数据完整性方面的工作。
   - 他还与 **Overture Maps Foundation** 合作，扩大了跨不同行业的数据使用范围。

---

## [AI21 Labs (Jamba)](https://discord.com/channels/874538902696914944) Discord

- **Python + Jamba 增强播客内容召回**：一位用户利用 **Jamba 的对话式 RAG** 配合一个基础 Python 应用，通过转录文本检索过去播客的精彩片段，称其为正在进行中的有趣实验。
   - 他们提到正在探索集成 AI 驱动召回的新方法，且没有遇到重大障碍，发现该系统对于归档节目笔记非常方便。
- **AI 代码生成很棒... 但也会出错**：一位用户对 **AI 生成代码的能力**赞不绝口，称赞其对 HTML 和 JavaScript 的处理，但也指出偶尔会出现愚蠢的错误。
   - 他们测试了 **PHP** 任务以衡量 AI 的极限，结论是代码生成虽然有时令人费解，但依然很有帮助。
- **PHP 在 Jamba 连接中表现稳健**：另一位用户宣布忠于使用 **PHP** 进行 Web 和 IRC 机器人编码，并称连接到 Jamba 是一次真正的冒险。
   - 他们喜欢它与 **deepSeek** 和 **OpenAI** API 的相似之处，这简化了编程任务并鼓励快速尝试。

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **ModernBERT 微调引起关注**：有人提出了关于 **微调 ModernBERT** 的咨询，暗示了特定任务的改进，但尚未分享直接经验。
   - 对话在没有后续跟进的情况下结束，观察者们希望有 **技术示例** 或演示来指明前路。
- **Nectar Social 抛出 1 万美元推荐奖金**：**Nectar Social** 正在寻求以 AI 为核心的人才，推荐奖金高达 **$10,000**，职位包括高级/资深产品经理和 LLM/AI 工程师。
   - 他们强调了在 **社交电商** 领域的增长，拥有知名客户，并鼓励感兴趣的申请人私信（DM）了解详情。

---

**MLOps @Chipro Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Axolotl AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Mozilla AI Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**HuggingFace Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

**Gorilla LLM (Berkeley Function Calling) Discord** 没有新消息。如果该频道长时间没有动静，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1326645308322615421)** (719 messages🔥🔥🔥): 

> `AI 模型 GPU 兼容性、图生视频挑战、Discord 社区动态、AI 工具 UI/UX 偏好、Linux 系统 Kernel Panic` 

- **AI 模型 GPU 兼容性**：用户讨论了 RTX 4070 和 3090 等不同 GPU 在生成视频时的性能，指出虽然 3090 可以在约 2 分钟内生成 480p 视频，但速度可能随其他模型而变化。
   - 某些模型（如 LTXV）声称支持图生视频功能，但根据用户的配置显示出不同的性能指标。
- **图生视频挑战**：对 AnimateDiff 等旧视频生成模型质量不佳的指责，引发了关于探索结合各种技术以获得更好效果的新方法的讨论。
   - 用户辩论了使用各平台工作流的优缺点，并分享了在 ComfyUI 中实现动画视频生成的具体指令。
- **Discord 社区动态**：社区承认成员中存在不当行为和个人资料图片，引发了关于 Discord 举报和审核的讨论。
   - 人们对维护社区标准的挑战以及有毒行为对用户体验的影响表示担忧。
- **AI 工具的 UI/UX 偏好**：用户对各种 AI 工具的易用性表达了不同意见，比较了 A1111、SwarmUI 和 ComfyUI 在用户体验和可访问性方面的界面。
   - 虽然有些人更喜欢 A1111 的直观性，但另一些人认为 SwarmUI 的高级功能更有价值，尽管其学习曲线更陡峭。
- **Linux 系统中的 Kernel Panic**：关于在 Linux 上运行 Stable Diffusion 模型的讨论带来了技术担忧，例如内核恐慌（Kernel Panic）以及与较新 Python 版本的兼容性问题。
   - 用户分享了设置和排除各种系统故障的指南和资源链接，以优化其 AI 工作流。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://civitai.com/articles/4339/image-to-video-comfyui-workflow-using-animatediff-and-ip-adapter-ready-to-use">图像转视频（使用 AnimateDiff 和 IP Adapter 的 ComfyUI 工作流）即插即用 | Civitai</a>：工作流位于右上角的附件 json 文件中。附件是一个用于 ComfyUI 的工作流，可将图像转换为视频。它会改变...</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-by-mikus-silly-and-easy">懒人教程 | 如何在 Colab 上使用 Trainer LoRA | SD 1.5 &amp; XL 作者：mikus（简单易上手）| Civitai</a>：警告！我在这方面经验不多，所以建议先学习所有功能并阅读更多关于如何操作的教程...</li><li><a href="https://civitai.com/articles/6182/how-to-make-a-lora-on-colab">如何在 Colab 上制作 LoRA | Civitai</a>：在 WebUI 的 extra 选项卡下进行批量裁剪（1024x1024）和放大（我使用 4x_NMKD-UltraYandere_300k）（从目录批量处理），上传到 Drive，运行...</li><li><a href="https://civitai.com/models/548997/image-to-video-comparison-workflow">图像转视频对比工作流 - v1.0 | Stable Diffusion XL 工作流 | Civitai</a>：摘要：此工作流是作为一个实验制作的，用于对比各种支持“图像转视频”的技术。事实上，它允许对比以下...</li><li><a href="https://tenor.com/view/cyanide-and-happiness-distraught-shocked-diagnosis-gif-23623883">Cyanide And Happiness Distraught GIF - Cyanide And Happiness Distraught Shocked - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://np.reddit.com/user/kejos92/comments/1hjkkmx/ltxv_inference_on_amd_gpus/">在 AMD GPU 上进行 LTXV 推理</a>：# 简介 在过去的两个月里，随着...</li><li><a href="https://github.com/Stability-AI/StableSwarmUI/blob/master/docs/Features/IPAdapter-ReVision.md">StableSwarmUI/docs/Features/IPAdapter-ReVision.md 分支 master · Stability-AI/StableSwarmUI</a>：StableSwarmUI，一个模块化的 Stable Diffusion Web 用户界面，强调让强大工具易于获取、高性能且具有可扩展性。- Stability-AI/StableSwarmUI</li><li><a href="https://civitai.com/models/134056/explosm-cyanide-and-happiness-style">Explosm Cyanide and Happiness 风格 - 2 | Stable Diffusion LoRA | Civitai</a>：推荐设置 0.8-1.2，负面提示词使用：nose, chin, ears, cheeks, jawline cyanide and happiness（使用 lipstick, breasts 来生成女性...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux">在 AMD GPU 上安装和运行</a>：Stable Diffusion web UI。通过在 GitHub 上创建一个账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/.github/images/swarmui.jpg">SwarmUI/.github/images/swarmui.jpg 分支 master · mcmonkeyprojects/SwarmUI</a>：SwarmUI（原名 StableSwarmUI），一个模块化的 Stable Diffusion Web 用户界面，强调让强大工具易于获取、高性能且具有可扩展性。- mcmonkeyprojects/Swa...</li><li><a href="https://www.youtube.com/watch?v=PWgvGjAhvIw">Outkast - Hey Ya! (官方高清视频)</a>：OutKast 的 "Hey Ya!" 官方高清视频。收听 OutKast：https://Outkast.lnk.to/listenYD 订阅 Outkast 官方 YouTube 频道：https://Outkas...</li><li><a href="https://civitai.com/articles/7993/lazy-tutorial-or-how-to-use-trainer-lora-on-colab-or-sd-15-and-xl-">懒人教程 | 如何在 Colab 上使用 Trainer LoRA | SD 1.5 &amp; XL 作者：mikus（简单易上手）| Civitai</a>：警告！我在这方面经验不多，所以建议先学习所有功能并阅读更多关于如何操作的教程...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki">首页</a>：Stable Diffusion web UI。通过在 GitHub 上创建一个账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/wileewang/TransPixar">GitHub - wileewang/TransPixar</a>：通过在 GitHub 上创建一个账户来为 wileewang/TransPixar 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/pull/2151">由 parsee-mizuhashi 移除侵犯许可证/潜在恶意/混淆的代码 · Pull Request #2151 · lllyasviel/stable-diffusion-webui-forge</a>：另请参阅相应仓库中的此 PR。侵犯许可证：此代码至少部分复制自 ComfyUI，ComfyUI 采用 GPL-3.0 许可证，该许可证禁止在不发布源码的情况下发布编译后的代码...</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Webui-Installation-Guides">Webui 安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1326644994135822396)** (393 messages🔥🔥): 

> `Unsloth updates, PHI-4 model fixes, Quantum models comparison, Adapters in fine-tuning, Chat templates in LLMs` 


- **Unsloth 发布 PHI-4 错误修复**：Unsloth 的 PHI-4 版本在 Open LLM Leaderboard 上已经超越了 Microsoft 官方版本，展示了显著的改进。更新包括 Unsloth 博客中报告的关键错误修复，以及对提升模型性能的持续承诺。
   - 尽管超越了 Microsoft 版本，一些用户注意到量化模型在某些领域的表现优于非量化模型的差异。
- **使用 Adapters 的最佳实践**：在推理（inference）时，建议在挂载 Adapters 时使用 16-bit 模型而非 4-bit 量化模型，以确保更好的性能。使用低精度模型创建 Adapters 可能会引入不理想的损失。
   - 虽然使用两种精度进行微调（fine-tuning）可以获得类似的结果，但合并 Adapters 最好使用高精度模型。
- **Chat templates 与微调（fine-tuning）注意事项**：Chat templates 对于 LLMs 的微调和部署都至关重要，因为它们决定了模型如何处理输入并提供输出。训练期间使用的模板可以在 tokenizer_config.json 文件中找到。
   - 正确设计 Chat templates 可以显著影响 LLMs 在生产场景中的性能和可用性。
- **将计算神经科学与 LLMs 联系起来**：讨论强调了计算神经科学的进展与大语言模型（LLMs）改进之间的平行关系。用户对受大脑功能启发的剪枝（pruning）和增强（boosting）技术的潜在影响表示好奇。
   - 尽管存在挑战，这些见解的整合继续推动着优化模型性能的兴趣。
- **兼容性与未来方向**：贡献者指出保持库更新的必要性，因为一些仓库（repositories）可能会变得过时。用户讨论了基础模型和库的重要性及其随时间的兼容性。
   - 鼓励社区探索 forks 或替代实现，以保持与当前进展一致。


<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard#/?search=phi-4&rowSize=large">Open LLM Leaderboard - 由 open-llm-leaderboard 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="https://runpod.io?ref=bb842lb3">RunPod - 为 AI 构建的云</a>：在一个云端开发、训练和扩展 AI 模型。通过 GPU Cloud 启动按需 GPU，通过 Serverless 扩展 ML 推理。</li><li><a href="https://xkcd.com/1425/">Tasks</a>：未找到描述</li><li><a href="https://huggingface.co/unsloth/phi-4-GGUF">unsloth/phi-4-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://rog.asus.com/us/laptops/rog-strix/rog-strix-scar-18-2025/">ROG Strix SCAR 18 (2025) G835 | 游戏笔记本｜ROG - 玩家国度｜ROG 美国</a>：未找到描述</li><li><a href="https://huggingface.co/docs/peft/en/index">PEFT</a>：未找到描述</li><li><a href="https://x.com/UnslothAI/status/1877136074042126338">来自 Unsloth AI (@UnslothAI) 的推文</a>：Phi-4，包括 GGUF + 4-bit + 16-bit 版本现已上线 @HuggingFace！我们在 Phi-4 中发现并修复了 4 个 bug，并将该模型 Llamafied。查看所有包含我们 bug 修复的 Phi-4 版本：https://huggingface.co/collec...</li><li><a href="https://huggingface.co/learn/cookbook/en/llm_judge">使用 LLM-as-a-judge 🧑‍⚖️ 进行自动化且通用的评估 - Hugging Face 开源 AI 食谱</a>：未找到描述</li><li><a href="https://huggingface.co/collections/unsloth/phi-4-all-versions-677eecf93784e61afe762afa">Phi-4 (所有版本) - unsloth 收藏集</a>：未找到描述</li><li><a href="https://x.com/Unsl">来自 FxTwitter / FixupX 的推文</a>：抱歉，该用户不存在 :(</li><li><a href="https://docs.unsloth.ai/get-started/unsloth-notebooks">Unsloth Notebooks | Unsloth 文档</a>：以下是我们所有 notebook 的列表：</li><li><a href="https://huggingface.co/unsloth/DeepSeek-V3-GGUF">unsloth/DeepSeek-V3-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Iq1JeXKYg5k">RTX 5090 笔记本电脑来了！</a>：Nvidia 的 Blackwell 50 系列笔记本电脑已发布，包括 RTX 5090, RTX 5080, RTX 5070Ti, RTX 5070。RTX 5090 笔记本电脑 - https://rog.asus.com/us/laptops-group/ Nvidia - https:...</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hwzmqc/phi4_llamafied_4_bug_fixes_ggufs_dynamic_4bit/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/cognitivecomputations/laserRMT">GitHub - cognitivecomputations/laserRMT：这是我们对 'Layer Selective Rank Reduction' 的自行实现</a>：这是我们对 'Layer Selective Rank Reduction' 的自行实现 - cognitivecomputations/laserRMT</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements">Unsloth 要求 | Unsloth 文档</a>：这里是 Unsloth 的要求，包括系统和 GPU VRAM 要求。</li><li><a href="https://github.com/unslothai/unsloth/pull/1516">由 danielhanchen 提交的 Bug 修复 · Pull Request #1516 · unslothai/unsloth</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1326655511847763978)** (3 条消息): 

> `求职成功，有趣的反应 GIF` 


- **Infinit3e 找到了工作！**：一位成员宣布他们的求职已经完成，并表示：*'求职结束，我现在就业了。'*
   - 社区成员对这一里程碑表达了祝贺。
- **分享了 Amogus6969 GIF**：一位成员分享了一个幽默的 GIF，画面中一名穿着西装的男子做着滑稽的表情，可以在[这里](https://tenor.com/view/amogus6969-gif-26819393)查看。
   - GIF 的描述指出该角色的反应非常有趣且适合当前语境。



**提到的链接**：<a href="https://tenor.com/view/amogus6969-gif-26819393">Amogus6969 GIF - Amogus6969 - 发现并分享 GIF</a>：点击查看 GIF

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1326791388213284935)** (48 messages🔥): 

> `Mathstral 模型状态、AI 模型建议、RAG 选项、使用 LORA 进行微调、Qwen2VL 模型错误` 


- **Mathstral 模型说明**：一位成员澄清说 `mistralai/Mathstral-7B-v0.1` 既不是基础模型也不是 PEFT 模型，这表明当前支持存在局限性。
   - 另一位成员提到预计很快会支持该模型，展示了持续的开发进展。
- **关于拆分姓名和性别识别的建议**：一位成员寻求开发 AI 来解析全名并识别性别的建议，并考虑了微调选项。
   - 其他人建议使用经典的 ML 方法，指出为此任务使用 LLM 效率低下，并强调了其历史背景。
- **建议对持续更新的内容使用 RAG**：一位成员建议在涉及持续更新内容的任务中考虑使用 RAG (Retrieval-Augmented Generation)。
   - 他们建议从 YouTube 上的 “rag tutorial” 开始获取教学指导。
- **关于 LORA 与模型合并的微调讨论**：出现了一场关于使用 16B LORA 模型进行合并的讨论，质疑使用上采样（upscaled）的 4Q 基础模型与原始模型相比的影响。
   - 参与者一致认为使用 16B 模型进行训练和合并是安全的，但也对潜在的弊端表示了担忧。
- **Qwen2VL 模型遇到错误**：一位用户在训练视觉模型时，在使用 Qwen2VL 时遇到了关于 embedding 模块的 RuntimeError。
   - 他们后来发现切换到 Llama3.2-vision-instruct 解决了该问题，证实了 Qwen2VL 可能存在损坏。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/@bnjmn_marie/lora-load-and-merge-your-adapters-with-care-3204119f0426">LoRA: Load and Merge Your Adapters with Care</a>：使用 QLoRA 微调的 LoRA 适配器案例</li><li><a href="https://docs.unsloth.ai/get-started/beginner-start-here">Beginner? Start here! | Unsloth Documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1326857299767066746)** (4 messages): 

> `语言模型的 DLSS、Speculative Decoding` 


- **探索语言模型的类 DLSS 技术**：在 CES 讨论了 DLSS 之后，一位成员想知道是否存在类似的语言模型技术，可以优化训练或推理资源。
   - 他们专门寻求与高效预测后续步骤以减少资源消耗相关的研究。
- **建议使用 Speculative Decoding**：针对关于语言模型省资源方法的咨询，有人建议将 Speculative Decoding 作为潜在解决方案。
   - 另一位成员表示感谢，指出 Speculative Decoding 是他们问题的完美答案。


  

---

### **Codeium (Windsurf) ▷ #[discussion](https://discord.com/channels/1027685395649015980/1027697446446432336/1326645517907787849)** (125 messages🔥🔥): 

> `Self-hosted Codeium, Windsurf issues, Cascade Model discussion, Purchase of credits, User experiences with Codeium` 


- **Codeium 自托管版本现已推出**：一位成员指出，**Codeium** 的自托管版本（Self-hosted）现已包含在企业版方案中。
   - *这一特性似乎引起了想要探索自托管选项的用户的兴趣。*
- **Windsurf 身份验证问题**：多位成员报告了在购买额度后，在 **codeium.com** 进行身份验证以及连接 **Windsurf** 时遇到的问题。
   - 各种建议包括退出并重新登录，作为解决错误信息的潜在修复方案。
- **Cascade 模型的灵活性讨论**：用户对 **Cascade Model** 的能力表示赞赏，特别注意到其基础版中的无限使用特性与高级版中受限查询的对比。
   - 一位用户分享说，他们没写代码就有效地构建了一个公司网站，展示了该模型的能力。
- **关于额度购买的担忧**：一位成员询问新购买的额度如何添加到其更新后的计划中，并表达了解决问题的紧迫性。
   - 建议发送电子邮件详情，以便更快地处理与额度问题相关的查询。
- **Codeium 转型带来的用户体验**：成员们分享了由于使用 **Windsurf** 和 **Cascade**，其编程习惯和生产力发生的积极转变。
   - 一位用户表示，由于这些工具带来的生产力提升，他们已经变成了一个完全不同的人。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/pricing#are-credits-consumed-for-command-and-chat-in-codeium-extensions-vs-code-and-jetbrains-ides">定价 | Windsurf 编辑器和 Codeium 扩展</a>：Codeium 对个人永久免费。团队可以通过我们的企业版方案进行升级，以获得增强的个性化和灵活的部署。</li><li><a href="https://github.com/Exafunction/codeium.el/issues/115">如何获取我的 API Key？ · Issue #115 · Exafunction/codeium.el</a>：我正尝试在 codeium.com 上查找我的 API Key 但找不到。我该去哪里找？
</li>
</ul>

</div>
  

---


### **Codeium (Windsurf) ▷ #[windsurf](https://discord.com/channels/1027685395649015980/1306163501286293515/1326642701403422740)** (140 messages🔥🔥): 

> `Windsurf Installation, Cascade Chat Optimizations, Flow Credits Discrepancies, Agent Integration in Windsurf, User Experience Issues` 


- **Windsurf 安装成功案例**：用户分享了在各种操作系统上成功安装 Windsurf 的经历，有人提到在 Mint 和 Ubuntu 24.04 上的体验非常流畅。
   - 一位用户指出，在删除配置文件夹后，在 Arch 和 Hyprland 上也取得了类似的成功，从而完成了安装。
- **Cascade Chat 无法跨设备同步**：一位用户询问如何通过 Dropbox 在多台 PC 上使用 Cascade 聊天记录，强调了聊天同步的需求。
   - 其他人也表达了对 Windsurf 内部增强聊天管理功能的类似渴望。
- **Flow Credits 困惑与计费问题**：多位用户报告了 **Flow Credits** 购买和使用中的差异，对价值和结转机制提出质疑。
   - 用户担心被重复扣费却未收到额度，且用户尝试解决计费问题的过程极具挑战性。
- **内部错误与 Flow Action 消耗**：用户对 Windsurf 倾向于生成不必要的输出表示沮丧，这导致了过多的 **Flow Action** 消耗。
   - 讨论内容包括内部错误是否计入 **Flow Actions**，以及最近的更新对可用性的影响。
- **用户体验反馈**：一位用户赞扬了 Windsurf 在快速应用开发方面的能力，而其他人则批评最近的更新响应迟钝。
   - 用户对不清晰的发布日志以及 **Composer** 和 **commit** 操作等功能表示担忧，表明需要开发者进行更清晰的沟通。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://codeium.com/changelog">Windsurf 编辑器更新日志 | Windsurf 编辑器和 Codeium 扩展</a>：Windsurf 编辑器的最新更新和变化。</li><li><a href="https://codeium.canny.io/">Codeium 反馈</a>：向 Codeium 团队提供反馈，以便我们做出更明智的产品决策。由 Canny 提供支持。
</li>
</ul>

</div>
  

---

### **Cursor IDE ▷ #[general](https://discord.com/channels/1074847526655643750/1074847527708393565/1326643664906092615)** (246 条消息🔥🔥): 

> `Cursor IDE 性能问题，使用 Cursor Rules 获得更好的输出，Composer 面临的挑战，与 Cursor 开发者取得联系，Claude 的用户体验` 


- **Cursor IDE 遭遇性能问题**：用户报告了 Cursor IDE 的使用困难，包括 'slow pool full' 消息和 Composer 卡住，引发了对稳定性和功能的担忧。
   - 该问题在各版本中持续存在，建议尝试重启或检查文件 indexing 以缓解问题。
- **使用 Cursor Rules 增强输出**：用户讨论了设置 Cursor Rules 以引导 Claude 输出的重要性，并建议在 prompts 中清晰地阐述目标。
   - 特定关键词可以增强 prompts，从而实现结构化的功能请求方式。
- **对 Composer 的挑战与不满**：多位用户对 Composer 表示不满，指出它经常忽略预设规则，并可能在多个文件中应用非预期的更改。
   - 一些人建议回退到之前的稳定版本，或使用更细颗粒度的 prompts 以避免代码更改过程中的问题。
- **寻找与 Cursor 开发者的联系**：成员们分享了多种联系 Cursor 员工的方式，包括在论坛发帖或通过社交媒体联系。
   - 幽默地提出了一些联系开发者的非正式方法，反映了社区的沮丧与轻松心态。
- **Claude 的用户体验与期望**：许多用户分享了使用 Claude 的经验，指出它有时无法有效地结合规则或 prompts，导致输出令人失望。
   - 尽管存在批评，用户也承认 Claude 在某些情况下能产生令人满意的结果，表明对其能力的反馈褒贬不一。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://forum.cursor.com/t/composer-stuck-at-generating-specific-composer-instance-not-global-issue/35479/4">Composer Stuck at &quot;Generating&quot; - Specific Composer Instance, Not Global Issue</a>: 嘿……这方面有进展吗？我仍然看到 Composer 会话卡住。我已经升级到 0.44.10；之前在 0.44.9 中卡住的会话在 0.44.10 中依然卡住。它一直停留在 “generating” 状态...</li><li><a href="https://onecompiler.com/bootstrap/435jnyccv">Card Glow Magnetic - Bootstrap - OneCompiler</a>: 未找到描述</li><li><a href="https://forum.cursor.com/">Cursor - Community Forum</a>: 讨论 Cursor 的地方（Bug、反馈、想法等）</li><li><a href="https://cursor.directory/">Cursor Directory</a>: 为你的框架和语言寻找最佳的 Cursor Rules</li><li><a href="https://dotcursorrules.com/">.CursorRules</a>: 用于自定义 AI 行为、简化开发流程，并根据你的框架和语言定制代码生成、建议和查询的 Cursor Rules。
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[prompting](https://discord.com/channels/364486390102097930/1301167628416454737/1326656125126447185)** (11 条消息🔥): 

> `Prompting 技巧、支付系统、公共 Repos 功能、睡眠时间表的影响` 


- **颜色使用的有效 Prompting**：成员们讨论了在 Prompt 中使用颜色名称和十六进制代码（hex codes）指定**颜色**的重要性，并强调了明确每种颜色应用位置的必要性。
   - *不要只说“给我做一个计时器应用，蓝白配色”*；最好提供一个总体的构思。
- **支付系统无法运行**：一位用户指出其应用的**支付系统**目前无法正常工作，暗示开发仍在进行中。
   - 他们分享了项目的链接，在努力解决支付问题的同时，恳请大家进行测试或提供反馈。
- **公共 Repos 功能发布公告**：一位成员强调，团队发布了一个关于允许用户在 bolt.new 中打开**公共 Repos** 的新功能，该功能已于 10 月发布。
   - 正如他们在 [X 帖子](https://x.com/stackblitz/status/1843668731681267801)中所宣布的，用户只需在任何 GitHub URL 前加上 *http://bolt.new* 即可访问。
- **睡眠时间表影响响应时间**：一位成员因**混乱的睡眠时间表**导致回复延迟而表示歉意，这反映了维持沟通的挑战。
   - 另一位成员对这种沟通表示理解和赞赏，强调了社区互助的氛围。
- **可用的 Prompting 资源**：一位用户分享了一个使用 bolt.new 制作的 **prompting** 资源链接，并邀请其他人提问。
   - 这表明社区正在主动促进关于有效 Prompting 策略的知识共享。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://subredditai.com">SubReddit AI</a>: 未找到描述</li><li><a href="https://x.com/stackblitz/status/1843668731681267801">来自 StackBlitz (@stackblitz) 的推文</a>: 你现在可以在 bolt.new 中打开公共 Repos 了 🙌 怎么做？对于任何 GitHub URL，只需在前面加上 "http://bolt.new" 即可！（发行说明见下方！）
</li>
</ul>

</div>
  

---

### **Stackblitz (Bolt.new) ▷ #[discussions](https://discord.com/channels/364486390102097930/680953097354215446/1326647185496080484)** (211 条消息🔥🔥): 

> `Bolt Token 问题、Bolt 中的 PWA 支持、Supabase 迁移疑虑、Netlify 性能问题、社区反馈与功能` 


- **Bolt Token 问题**：用户对 Bolt 在尝试进行编辑或排除项目错误时快速消耗 Token 表示沮丧。
   - 一些用户报告称，在工具对修复方案缺乏信心的情况下，重复操作消耗了大量 Token。
- **Bolt 中的 PWA 支持**：一位用户询问了 Bolt 对 Progressive Web Apps (PWA) 的支持情况，并引用了一条暗示底层 Stackblitz 限制的错误消息。
   - 其他用户评论称已成功部署 PWA，表明尽管原用户遇到了错误，但这应该是可行的。
- **Supabase 迁移疑虑**：有用户担心在遇到错误后，Supabase 中的迁移回滚无法与代码库更改同步处理。
   - 用户讨论了在更新失败后反转迁移以及维持应用程序功能的潜在挑战。
- **Netlify 性能问题**：一位用户报告了托管在 Netlify 上的 Bolt 网站加载缓慢，质疑问题是源于 Bolt 的代码还是 Netlify 的限制。
   - 该用户推测其免费账户可能会影响性能，暗示可能需要升级以获得更好的服务。
- **社区反馈与功能**：有建议提出在 Discord 社区中创建反馈和指南频道，以帮助用户学习和排除故障。
   - 社区成员强调了清晰的文档和对新用户的支持对于提升 Bolt 使用体验的重要性。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boltstudio.ai">BoltStudio.ai | Full Stack Prompt Engineering</a>: 未找到描述</li><li><a href="https://bolters.io">Bolters.io | Community Supported Tips, Tricks &#38; Knowledgebase for Bolt.new No-Code App Builder</a>: Bolt.new 的文档和指南</li><li><a href="https://github.com/stackblitz/bolt.new/issues/5149">Suggestion: Selector · Issue #5149 · stackblitz/bolt.new</a>: 这是我关于为站点添加选择器选项的建议。我将尝试更详细地解释：当你用鼠标高亮显示并进入聊天，例如说更改名称或删除...</li><li><a href="https://github.com/stackblitz/bolt.new/issues/2529">Bolt Outputs Application Logic in Chat · Issue #2529 · stackblitz/bolt.new</a>: 问题：Bolt 在聊天中输出应用逻辑。例如，当用户达到速率限制时，提供升级链接的代码会作为响应发送给聊天中的用户。</li><li><a href="http://bolt.diy">GitHub - stackblitz-labs/bolt.diy: Prompt, run, edit, and deploy full-stack web applications using any LLM you want!</a>: 使用你想要的任何 LLM 进行提示、运行、编辑和部署全栈 Web 应用程序！ - stackblitz-labs/bolt.diy
</li>
</ul>

</div>
  

---

### **aider (Paul Gauthier) ▷ #[general](https://discord.com/channels/1131200896827654144/1131200896827654149/1326689892834476083)** (66 messages🔥🔥): 

> `Aider 用户体验、AI 模型对比、模型能力与改进、编程助手开发、OpenAI 与 Gemini 模型` 


- **Aider 与 Claude 的行为对比**：用户观察到，与 **Claude** 相比，**DeepSeek** 作为编辑器似乎显得很“懒惰”，并评论其容易分心且无法准确执行命令。
   - 贡献者指出 **Aider** 的结果并不一致，一位成员幽默地讲述了它在修复问题时删除整个文件的经历。
- **关于 AI 模型能力的讨论**：**AI 模型的性能各不相同**，用户讨论了 **DeepSeek** 等模型以及 **Anthropic** 替代方案的效率，并强调了使用成本问题。
   - 讨论中涉及了当前流行的模型，强调了使用 **Anthropic 产品的不稳定性及高昂成本**。
- **AI 作为主动型 Agent 的未来**：一位用户乐观地表示 **AI 将变得更加主动**，暗示未来 AI 可以创建更好的自身版本，并减少人类的参与。
   - 相比之下，另一位成员提醒道，**当前 AI 的局限性**源于电力和计算成本，这可能会减缓进度。
- **开发 AI 编程助手**：一位用户表达了开发自己的**编程助手 Agent**的愿望，认为 **Aider** 是一个合适的替代方案，并寻求对该项目的贡献。
   - 贡献者讨论了将各种功能集成到 **Aider** 中的方法，包括通过 issue 追踪实现自动代码修订。
- **OpenAI 模型命名的澄清**：有人询问以 **openai/** 开头的模型名称与其他不带前缀的名称之间的区别，得到的澄清是它们本质上是相同的。
   - 另一位用户试图了解为什么某些模型带有前缀，回复中强调了 **OpenAI 在命名上的灵活性**。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aider.chat/docs/troubleshooting/edit-errors.html">文件编辑问题</a>：aider 是你终端里的 AI 配对编程工具</li><li><a href="https://aider.chat/docs/usage/lint-test.html">Linting 与测试</a>：自动修复 linting 和测试错误。</li><li><a href="https://github.com/Aider-AI/aider/blob/main/CONTRIBUTING.md">aider/CONTRIBUTING.md at main · Aider-AI/aider</a>：aider 是你终端里的 AI 配对编程工具。通过在 GitHub 上创建账号为 Aider-AI/aider 的开发做出贡献。</li><li><a href="https://github.com/Aider-AI/aider/pull/540">feat: add `/rg` command (ripgrep for `/add` files) by aleclarson · Pull Request #540 · Aider-AI/aider</a>：注意：使用此命令需要在你的机器上安装 ripgrep。工作原理：它通过子进程调用带有 -l 标志的 rg，以返回文件名列表。然后这些文件名被输入到...
</li>
</ul>

</div>

### **aider (Paul Gauthier) ▷ #[questions-and-tips](https://discord.com/channels/1131200896827654144/1133060505792159755/1326651345029169212)** (61 条消息🔥🔥): 

> `Aider 配置问题，使用 OpenAI 提供商，DeepSeek 性能，Aider 中的任务管理，处理 Aider 的聊天历史` 


- **Aider 发送 'prompt' 而非 'messages'**：一位用户报告称，Aider 在与本地 litellm 代理通信时发送的是 `prompt` 列表而非 `messages` 列表，导致出现 `TypeError: Router.acompletion() missing required argument: 'message'` 等错误。这引发了关于 JSON 文件中预期配置设置的疑问。
   - 另一位用户指出，`litellm_provider` 需要与模型名称的开头部分匹配才能成功通信。
- **获取 Tier 5 OpenAI Key**：一位用户寻求在不支付高额费用的情况下获取 Tier 5 OpenAI Key 的替代方案，并被引导至 OpenRouter 和 Unify.ai 等提供商。讨论内容包括潜在的变通方法以及使用不同模型和订阅计划的影响。
   - 分享了关于使用 `Unify` API 的信息，声称它允许通过单个 Key 访问多个模型，尽管它不是开源的。
- **DeepSeek 的性能问题**：一位用户询问了关于 deepseek-chat 或 deepseek-coder 在多次请求后卡住的性能问题，并提到是直接连接到 DeepSeek API。一些用户注意到遇到了速度变慢的情况，并建议通过不同网络更改路由可能会改善体验。
   - 其他人确认他们经常使用 DeepSeek 且没有问题，推测这可能是服务器负载或模型可用性的问题。
- **在 Aider 中管理任务**：一位用户询问如何组织 Aider 生成的多个任务建议，寻求有效管理这些建议的建议。一个有用的技巧是使用 TODO.md 文件直接在 Aider 聊天中跟踪任务。
   - 一位参与者强调了使用独立的 Aider 实例来更好地管理上下文，并防止模型因信息过多而过载的效率。
- **处理聊天历史保留**：用户对 Aider 保留聊天历史表示担忧，这有时会导致尽管之前进行了修正，但仍会出现重复的建议。建议用户使用 `/clear` 命令重置历史记录，以避免在持续会话中产生混淆。
   - 另一位用户询问如何改进 Prompt 准备以减少错误答案的重复出现，强调了 Aider 内部需要更好的状态管理。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/unifyai/unify">GitHub - unifyai/unify: Build Your AI Workflow in Seconds ⚡</a>: 在几秒钟内构建您的 AI 工作流 ⚡。通过在 GitHub 上创建账户为 unifyai/unify 的开发做出贡献。</li><li><a href="https://unify.ai/">Unify: Build AI Your Way</a>: 工具太多，过于复杂？在几秒钟内构建您自己的工作流！
</li>
</ul>

</div>
  

---


### **aider (Paul Gauthier) ▷ #[links](https://discord.com/channels/1131200896827654144/1268910919057149974/1326675570632822886)** (1 条消息): 

> `Gemini 2.0 Flash Experimental，iOS 上的语音模式，AI 辅助应用开发` 


- **与 Gemini 2.0 进行对话**：在处理杂务时，用户在 iOS 上使用 **Gemini 2.0 Flash Experimental** 的**语音模式**，像对待乘客一样讨论一个应用创意。
   - 虽然它没有提供 Markdown 文件，但该 AI 自主确立了项目标准，并在用户回家后生成了简洁的任务要点。
- **AI 辅助的应用规范**：用户请求 AI 帮助完善应用开发项目的规范和具体任务。
   - Gemini 2.0 成功引导了对话，并为未来的参考提供了可操作步骤的有用总结。


  

---

### **Notebook LM Discord ▷ #[use-cases](https://discord.com/channels/1124402182171672732/1124403655819415592/1326658028476301342)** (19 条消息🔥): 

> `将视频导入 NotebookLM，DeepResearch 报告集成，生成中文播客，直接引用的引用模式（Quotation Mode），NotebookLM Plus 中的 System prompts` 


- **将视频导入 NotebookLM 不可行**：一位成员幽默地询问是否有人尝试过将视频导入 NotebookLM，但系统目前尚不具备处理视频的能力。
   - 另一位用户提到了一种变通方法，即使用 ChatGPT 中的转录功能来生成目录。
- **为 NotebookLM 探索 DeepResearch 报告**：一位成员询问是否有人在 NotebookLM 中使用 DeepResearch 报告，并建议将这些报告中的来源整合到系统中。
   - 回复中提到目前没有直接集成，但建议使用扩展程序来批量上传来源。
- **从英文内容创建中文播客**：一位用户寻求建议，询问是否可以使用 NLM 从英文来源生成中文播客内容。
   - 这促使其他成员分享了他们如何修改播客脚本，使其更加随意和口语化的见解。
- **引用模式（Quotation Mode）的实现**：一位成员分享了一条指令，要求 NotebookLM 仅使用来源中的直接引用进行回答，以确保清晰度和可验证性。
   - 据指出，这种设置在 Gemini 上存在问题，因为它有时会返回不完整的信息。
- **关于 Plus 版本中 System Prompts 的说明**：一位用户质疑 Plus 版本中是否存在 System prompts，因为其界面看起来与免费版完全相同。
   - 回复澄清说，只有在 Plus 版本中才能持续使用与 System prompts 绑定的指令，从而增强功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.akashq.com/post/122c0310-6683-45d7-adec-3d3f4bbebd16">1 月 9 日发生了什么？</a>：1 月 9 日发生了什么？来自 This Day in History</li><li><a href="https://www.akashq.com/post/ad632a26-91b5-44b4-b8f4-5b5fd3f083e8">1 月 8 日发生了什么？</a>：1 月 8 日发生了什么？来自 This Day in History
</li>
</ul>

</div>
  

---

### **Notebook LM Discord ▷ #[general](https://discord.com/channels/1124402182171672732/1124402182909857966/1326641747840995348)** (94 条消息🔥🔥): 

> `NotebookLM 功能、音频生成特性、Workspace 许可证问题、对话中的语言选项、播客用户体验` 


- **排查 NotebookLM 访问问题**：成员们讨论了使用 Workspace 账号访问 NotebookLM Plus 的问题，强调只有拥有 Business 许可证并完成域名验证的用户才能有效使用。
   - 一些用户对功能无法正常运行表示沮丧，并建议了刷新页面或重新上传文件等故障排除方法。
- **使用选定来源生成音频**：用户询问了如何生成仅限于特定来源的播客，一种解决方案是在自定义提示词（prompts）中指定来源。
   - 分享了使用 Illuminate 等相关工具进行音频生成的技巧，旨在增强制作的灵活性。
- **日语对话**：讨论无缝切换到了日语，显示出用户在聊天中切换语言的自如。
   - 用户确认了他们能够用日语进行有效沟通，确保语言障碍降至最低。
- **NotebookLM 的增强与替代方案**：一位用户将 NotebookLM 与 Jellypod 等替代平台进行了比较，强调后者拥有更多自定义选项。
   - 提出了对未来改进可访问性和语音多样性的建议，突出了用户在教育用途方面的需求。
- **功能移除与用户适应**：用户对某些功能的移除表示不满，例如从 PDF 文本选择中生成 AI 问题建议的功能。
   - 提供了变通方法和替代技巧以适应这些变化，同时兼顾用户参与度。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.icloud.com/iclouddrive/061hg1R50Jv4idRhgdUqoMxWg#Captura_2025-01-09_a_las_8.15">iCloud Drive - Apple iCloud</a>: 未找到描述</li><li><a href="https://www.techradar.com/computing/artificial-intelligence/ive-found-a-new-ai-podcast-creator-and-it-leaves-googles-notebooklm-in-the-dust">我发现了一个新的 AI 播客生成器，它让 Google 的 NotebookLM 望尘莫及</a>: Jellypod 让你可以主持自己的播客，而无需靠近麦克风</li><li><a href="https://notebooklm.google.com/notebook/982b3b0c-0913-4599-816a-9c845a6b7d79/audio">未找到标题</a>: 未找到描述</li><li><a href="https://illuminate.google.com/create">Illuminate | 以你的方式学习</a>: 使用 Illuminate 将研究论文转换为 AI 生成的音频摘要，这是你更快理解复杂内容的 Gen AI 工具。</li><li><a href="https://notebooklm.google.">Google NotebookLM | AI 驱动的笔记与研究助手</a>: 利用 AI 的力量进行快速总结和记笔记，NotebookLM 是你强大的虚拟研究助手，植根于你可以信赖的信息。</li><li><a href="https://akashq.com">Akas: AI 播客之家</a>: 未找到描述</li><li><a href="https://youtu.be/spj0n-bFKJo"> - YouTube</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[general](https://discord.com/channels/1110598183144399058/1110598183144399061/1326662295522381945)** (66 条消息🔥🔥): 

> `LM Studio 与 API 连接性、模型加载问题、模型的目录结构、Qwen Chat 新功能发布、LLM 应用与开发趋势` 


- **LM Studio 连接挑战**：多位用户报告了 **LM Studio** 无法连接到 API 或在没有明确错误消息的情况下无法加载模型的问题，特别是使用 **0.2.26** 等旧版本的用户。
   - 用户 **friiscs2** 在确保从应用程序文件夹而非安装 GUI 打开应用后解决了问题，这凸显了安装程序可能带来的困惑。
- **模型兼容性的目录结构**：**Marsv.** 对 LM Studio 要求特定的模型子目录结构表示沮丧，这导致了与其它拥有更统一模型目录格式的应用之间的冲突。
   - 用户建议，LM Studio 和 Ollama 之间交替的目录结构可能会导致未来各种 LLM 应用功能的趋同。
- **阿里巴巴推出 Qwen Chat 功能**：阿里巴巴宣布推出 **Qwen Chat**，这是一个用于与各种 Qwen 模型交互的新 Web UI，具有文档上传和视觉理解能力等功能。
   - 该聊天界面旨在集成多个模型，并预计很快将推出网页搜索和图像生成等额外功能。
- **用户在 LLM 应用中的探索**：几位用户分享了他们尝试多个 LLM 应用的经验，指出随着时间的推移，它们通常会趋向于相似的核心功能。
   - **Skeletonbow** 强调了利用 LM Studio 作为后端开发自定义聊天客户端的乐趣，同时保持对其他应用提供的独特功能的关注。
- **OpenCL 后端支持咨询**：用户 **Uniraa** 询问了针对 Snapdragon X Elite 和 Windows on ARM 的 OpenCL 后端支持计划，并提到了 **Llama.cpp** 最近的进展。
   - 这凸显了开发者对增强 AI 领域各种硬件支持的持续关注。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai/download">下载 LM Studio - Mac, Linux, Windows</a>: 发现、下载并运行本地 LLM</li><li><a href="https://fxtwitter.com/Alibaba_Qwen/status/1877426465349972113">来自 Qwen (@Alibaba_Qwen) 的推文</a>: 🚀 激动人心的消息！我们非常高兴地宣布推出 Qwen Chat ( https://chat.qwenlm.ai ) – 您与 Qwen 模型交互的新首选 Web UI！🌟💬 轻松与我们的旗舰模型对话...</li><li><a href="https://lmstudio.ai/docs/advanced/sideload">侧加载模型 - 高级 | LM Studio 文档</a>: 使用您在 LM Studio 之外下载的模型文件
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1326644383373725738)** (33 条消息🔥): 

> `AMD RX 7900XT 性能、MacBook Pro 的 Sidecar 显卡、Llama 3.3 的内存要求、GPU 活动监控工具、基准测试 DIGITS 的到来` 


- **AMD RX 7900XT 在基准测试中表现挣扎**：一位成员提出了关于 **7900XT 20GB** 与 **4090、4080** 和 **3090** 性能对比的问题，指出对比中可能存在的较低显存带宽。
   - 另一位成员分享了一个 [reddit 链接](https://reddit.com)，展示了 **7900XT** 与 **3090** 的对比结果。
- **MacBook Pro 不支持 Sidecar 显卡**：成员们讨论了由于 Apple silicon 的限制，无法在 **MacBook Pro** 上使用“sidecar”显卡（外置显卡），这与旧款 Intel 机型有所不同。
   - 有人提出了使用 **Thunderbolt 5** 连接外置 GPU 的建议，反映了对运行更大模型的需求。
- **Llama 3.3 的内存评价**：一位尝试在有限硬件（Ryzen 7 和 **RX 7900GRE**）上运行 **Llama 3.3 70B Instruct** 的用户注意到性能缓慢，仅为 **0.5 token/sec**，引发了对 RAM 是否充足的担忧。
   - 成员们一致认为，充分利用 GPU 显存对于获得最佳速度至关重要，并建议调整 RAM 或采用更好的硬件配置。
- **系统瓶颈分析工具**：讨论包括了适用于 Linux 用户的 **htop** 和 **nvtop** 等工具，但在 GPU 活动显示方面存在一些例外。
   - 建议 Windows 用户使用各种可用的免费软件工具来监控系统性能。
- **对 DIGITS 发布的兴奋**：尽管预期 **DIGITS** 会更早到来，一位成员对其作为 **NVIDIA stack** 中全面解决方案的潜力表示乐观。
   - 人们对性能速度提出了担忧，强调了对其在未来项目中实际效用的期待。


  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1326658567394164767)** (60 条消息🔥🔥): 

> `Model Versions and Testing, TensorFlow GPU Issues, Machine Learning Resources, Jupyter vs Python File Debugging, Community Concerns about AI Safety` 


- **Model O1 讨论**：一名成员指出 Model O1 输出的独特格式，提到其中包含 'thinking'（思考）过程，暗示可能采用了不同的模型格式。
   - 另一名成员询问在 A/B 测试场景中，是否会将推理模型与 Model 4O 结合使用。
- **TensorFlow 未检测到 GPU**：一位用户报告称，尽管安装了 CUDA、cuDNN 和 tensorflow-gpu，且配置为 **64G RAM** 和 **RTX 3060**，但其 Jupyter 内核仍无法检测到 NVIDIA GPU。
   - 成员们分享了排查步骤，例如确保环境已激活，并建议运行 `conda env list` 来确认环境设置。
- **寻求 ML 学习资源**：一名成员询问了学习 Machine Learning 的最佳 YouTube 频道，表达了对社区推荐的共同兴趣。
   - 另一名成员开玩笑地回应以规避 OpenAI 的模型安全措施问题，暗示对话可能会演变成对 AI 监管的担忧。
- **关于调试方法的辩论**：一场关于使用带有断点的标准 Python 文件与使用 Jupyter Notebook 进行代码验证之优劣的讨论展开了。
   - 一名成员对在 VSCode 中解决问题表示满意，而其他人则对调试偏好发表了不同看法。
- **对 AI 响应性的担忧**：成员们对 AI 讨论中倾向于归咎于工具而非个人问题的现象表示沮丧。
   - 有人担心，无论在提高 AI 模型安全性方面投入多少努力，“jailbreaks”（越狱）总会存在，从而质疑现有安全措施的效率。


  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1326721363750162513)** (7 条消息): 

> `GPT code handling, ChatGPT generating graphs` 


- **对 GPT 代码回复的沮丧**：成员们表示，即使经过多次提示，GPT 仍然继续以注释形式回复，而不是发送请求的完整代码，这令人感到沮丧。
   - 有人指出性能上的差异，称虽然 GPT-4 能较好地处理请求，但 GPT-3.5 经常无法提供完整代码。
- **ChatGPT 生成图表令人惊喜**：一位成员评论了 ChatGPT 生成图表的惊人能力，引发了其他人的难以置信。
   - 社区以热烈的反应作为回应，强调了他们对 AI 能力的惊讶。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 条消息🔥): 

> `Meta-Prompting Use Cases, Insights on Prompting, Investor Round for Hassabis` 


- **Meta-Prompting 引起关注**：成员们分享了对 **Meta-Prompting** 的兴趣，其中一人表示希望探索其创新使用案例。
   - 另一名成员强调，一个好的 Prompt 应该从对预期输出的清晰理解开始。
- **让 OpenAI 变得无利可图**：一名成员提到他觉得从 OpenAI 那里拿到了 **0 美元**，为现场增添了幽默感。
   - 这种关于缺乏经济补偿的情绪引发了关于该群体封闭性质的讨论。
- **支持 Hassabis 的投资轮次**：一名成员请求大家对 **Hassabis 的投资轮次** 发表积极看法，并认可了他的能力。
   - 社区似乎认可他在 AI 领域的贡献和潜力。
- **追求有效的 Prompt**：一名成员表达了在扩展对什么是**优秀 Prompt** 的理解方面所面临的挑战。
   - 对话体现了成员之间在改进 Prompting 技术方面的协作精神。


  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1326734331338821704)** (13 条消息🔥): 

> `Meta-Prompting, OpenAI Contributions, Investor Round, Prompt Creation` 


- **探索 Meta-Prompting 使用案例**：一位成员询问了关于 **Meta-Prompting** 的经验和有趣的用例，表达了对其影响和有效性的好奇。
   - 另一位成员幽默地回应，建议修改 system message 可以带来显著的改进。
- **关于 OpenAI 财务贡献的质疑**：讨论了 OpenAI 缺乏财务奖励的问题，成员们对尽管做出了宝贵贡献但该小组仍缺乏资金表示困惑。
   - 一位成员幽默地评论说，他们也从 OpenAI 那里收到了“零美元”，强调了这种共同的情绪。
- **为投资者轮次祈祷**：一位成员请求在 **Hassabis** 的 investor round 期间给予支持和祈祷，认可了他的能力。
   - 这一请求强调了该小组对其创业项目成功的渴望以及对该领域人才的认可。
- **构建有效的 Prompt**：在回答有关 Prompt 编写的询问时，另一位成员强调，理解预期的输出对于创建一个好的 Prompt 至关重要。
   - 这强调了小组内的一个共识，即在与 AI 模型合作时，明确目标至关重要。


  

---


### **Interconnects (Nathan Lambert) ▷ #[events](https://discord.com/channels/1179127597926469703/1179127598442348729/1326757945525927977)** (3 条消息): 

> `ICLR Event Attendance, Meeting Points and Descriptions` 


- **对 ICLR 活动的热情**：一位成员表达了参加 **ICLR** 活动的热情，并询问其他人是否也会到场。
   - 这表明社区围绕该活动有着活跃的参与度。
- **Philpax 的到达和会面细节**：**Philpax** 宣布他们很快就会到达活动现场，并指出他们将在室外且没有移动网络。
   - 他们描述了自己的外貌：穿着**浅褐色外套**、**黑色牛仔裤**，背着**健身包**和**背包**，设定了明确的会面地点。


  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1326754105724768357)** (19 条消息🔥): 

> `rStar-Math 性能表现、Qwen Chat 发布、O1 vs GPT4o + MCTS 讨论、中国机器学习初创公司的挑战、EpiCoder 框架` 


- **rStar-Math 在数学基准测试中获得高分**：微软的新框架 **rStar-Math** 显著提升了在 MATH 基准测试上的表现，将 **Qwen2.5** 从 **58.8%** 提升至 **90.0%**，将 **Phi3-mini** 从 **41.4%** 提升至 **86.4%**。
   - 在美国数学奥林匹克竞赛（USA Math Olympiad）中，它取得了 **53.3%** 的平均分，位列高中参赛者的前 **20%**。
- **Qwen Chat Web UI 发布**：全新的 **Qwen Chat** 正式推出，允许用户通过统一界面与各种 Qwen 模型进行交互，包括 **Qwen2.5-Plus** 和 **Qwen2-VL-Max**。
   - 即将推出的功能包括网页搜索、图像生成和语音模式，增强了用户与 AI 模型的交互体验。
- **辩论：O1 vs GPT4o + MCTS**：讨论集中在 **O1** 是否仅仅是使用 **GPT4o** 和 **MCTS** 方法的更高效版本，以及是否存在 O1 在合理的计算预算内可以解决的独特问题。
   - 观点各异，有人指出 MCTS 可能更**昂贵**，而另一些人则讨论了模型自我修正（self-correction）的复杂性。
- **中国 AI 初创公司面临挑战**：中国 AI 初创公司**零一万物**（Zero One）正在进行重组，并将重点从训练大模型转向开发更具商业化价值的实用模型，理由是资金和资源限制。
   - 李开复强调了中国 AI 领域日益增长的困难，指出**芯片可用性**和**融资**限制是主要的障碍。
- **用于代码生成的 EpiCoder 简介**：**EpiCoder** 作为一个新的分层框架发布，旨在改进代码生成，在各种项目复杂度中展现出先进的能力。
   - 预计很快将进行**开源**发布，有望在编程任务中超越现有的基准测试。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/TeamCodeLLM_AI/status/1877254042574844153">来自 Wavecoder (@TeamCodeLLM_AI) 的推文</a>: 🚀 介绍 EpiCoder：一个基于分层特征树的框架，用于生成多样且复杂的代码。🔍 超越基准测试，它可以处理从简单函数到多文件项目的各种任务...</li><li><a href="https://x.com/_akhaliq/status/1877206745652592763?s=61">来自 AK (@_akhaliq) 的推文</a>: 微软展示 rStar-Math。小型 LLM 可以通过自我进化的深度思考掌握数学推理。在 MATH 基准测试中，它将 Qwen2.5-Math-7B 从 58.8% 提升到 90.0%，将 Phi3-mini-3.8B 从 41.4% 提升到...</li><li><a href="https://x.com/JustinLin610/status/1877427101370036595">来自 Junyang Lin (@JustinLin610) 的推文</a>: 就在这里！Qwen Chat (https://chat.qwenlm.ai)，我们为 Qwen 模型提供的新 Web UI。链接是：chat dot qwen lm dot ai！chat dot qwen lm dot ai！chat dot qwen lm dot ai！你可以与最令人印象深刻的模型聊天...</li><li><a href="https://mp.weixin.qq.com/s/IUA482JlwI4CcRpiMRGHbA">晚点对话李开复丨零一万物部分团队并入阿里，“灵魂拷问来得太快了”</a>: “机会来临时，要勇敢做决策，机会消失时也是。”
</li>
</ul>

</div>
  

---

### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1326754434746814515)** (21 条消息🔥): 

> `NuminaMath 数据集, 第一作者背景, 开源数据质量问题, 高中数学挑战, 技术领域的商业 vs 编程` 


- **对 NuminaMath 数据集质量的担忧**：一位成员对 **NuminaMath** 数据集的**质量**表示怀疑，指出 7.7% 的条目包含多个框选答案（boxed solutions），并引发了更深层次的质量担忧。
   - *“这类问题凸显了开源和公开数据的现状”* 暗示了系统性问题。
- **第一作者有趣的背景**：讨论透露该论文的第一作者是斯坦福大学的**心理学博士生**，这引发了成员们对其跨学科性质的关注。
   - 另一位成员提到 **Charlie Snell** 是第二作者，增加了这篇论文的吸引力。
- **数学竞赛的挑战**：一位成员分享了他们使用 NuminaMath 的 **cn_k12 子集**的经验，表示在一次研究尝试后，他们得出的结论是“在面对中国高中生时毫无胜算”。
   - 这一评论反映了在**数学相关研究**和学习中所面临的更广泛的挑战和竞争。
- **关于心理学和教育的轻松评论**：成员们就教育路径交换了轻松的看法，提到一位亲戚从**拉比研究（rabbinical studies）转向经济学**，展示了多样化的学术历程。
   - 幽默地提到了心理学教育路径，暗示了对心理学项目严格录取过程的集体尊重。
- **未来关注技术中的商业层面**：一位成员总结道，如果编程挑战得到解决，剩下的前沿领域就是**商业**，并强调了其在技术领域的重要性。
   - 对话暗示了同行之间的关注点正从技术技能转向商业敏锐度。



**提到的链接**：<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>：我们提出了一个新颖的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模得出特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)……

  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1326642132345421866)** (11 条消息🔥): 

> `大规模供应商的复杂性, Transformers vs MoEs, 模型性能效率` 


- **大规模 AI 供应商的复杂性**：成员们公认，对于**大规模供应商**来说，有效地管理模型架构中的复杂性是值得的。
   - *似乎要处理很多复杂性才能做对，但对于大规模供应商来说显然是值得的。*
- **Transformers 与 MoE 效率之争**：虽然 MoEs 似乎优于稠密模型（dense models），但一位成员建议，**for 循环**可能为初步理解这些架构提供更简单的方法。
   - 然而，大家一致认为更高的复杂性通常意味着整体性能更好。
- **MoEs 通常优于稠密模型**：一位成员强调，如果 **MoE** 模型与稠密模型保持相同数量的*激活*参数（active parameters），那么 MoE 通常更优。
   - 这符合一种观点，即尽管存在简洁性方面的缺点，但更多的信息可以存储在更多的权重中。


  

---

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1326645897110491238)** (17 messages🔥): 

> `Anthropic 沙龙, AI 模型的人格塑造, Post-training 过程, AI 专业人士的冒充者综合征, 学术界的博客写作与自我关怀` 


- **Josh Batson 关于模型塑造的见解**：在最近的 Anthropic 沙龙中，**Josh Batson** 提到 Amanda Askell 将基础模型塑造为一个 Agent，这可能暗示 **人格相关的变化** 发生得比预期更早。
   - 一些成员讨论了这种塑造是主要发生在 Post-training 阶段，还是在更早的阶段就进行了基础调整，暗示了一种 **分布式的人格对齐方法**。
- **关于人格开发时机的辩论**：讨论中出现了一种观点，即 AI 的人格塑造工作是否应被视为 **最终润色**，而非开发过程中固有的部分。
   - 成员们用比喻来说明这一点，将基础模型比作正在被塑造的粘土，而不仅仅是一个 **Post-training 任务**。
- **AI 社区中的冒充者综合征**：参与者分享了关于 **冒充者综合征** 的经历，有人指出尽管在 AI 和 ML 领域取得了成就，这种感觉依然挥之不去。
   - 他们承认这可以作为一种扭曲的动力，一位参与者幽默地评论说，它甚至可以被视为一种 **超能力**。
- **学术博客写作的挑战**：一位成员提到了难以确定哪些博客主题能引起共鸣，并幽默地指出他们关于 **自我关怀** 的帖子似乎吸引了很多人。
   - 另一位成员表示致力于发布他们的 **Deepseek 博客**，尽管面临学习 MLA 格式和撰写大量脚注等挑战。



**Link mentioned**: <a href="https://youtu.be/IPmt8b-qLgk?si=Cg2M9u4Rc5X7MHwb&t=964">How difficult is AI alignment? | Anthropic Research Salon</a>: 在旧金山举行的 Anthropic 研究沙龙活动中，我们的四位研究员——Alex Tamkin、Jan Leike、Amanda Askell 和 Josh Batson——讨论了对齐科学...

  

---


### **Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1326957605561827338)** (3 messages): 

> `Efficient Deep Learning, 弹窗问题` 


- **关于 Efficient Deep Learning 的新见解**：一位成员分享了一个 [Efficient Deep Learning 博客链接](https://alexzhang13.github.io/blog/2024/efficient-dl/)，讨论了创新的方法和技术。
   - 该博客旨在提高对 Deep Learning 中高效实践的理解和应用。
- **弹窗阻碍页面浏览**：一位成员对弹窗遮挡了他们试图查看的第一部分页面表示沮丧。
   - 另一位成员幽默地回应道：*Lmao brutal googling*。


  

---


### **Interconnects (Nathan Lambert) ▷ #[posts](https://discord.com/channels/1179127597926469703/1228051082631188530/1327017242222923907)** (14 messages🔥): 

> `开源 AI 成本, AI 政策制定者的反应` 


- **廉价的开源 AI**：讨论强调了政策制定者对 **开源 AI** 相关成本的担忧，特别是它仅需 **500 万美元** 即可开发。
   - 这引起了关注，因为一些人担心成本的影响经常被误解，从而导致公众的误解。
- **对成本呈现方式的批评**：一位成员指出，推文中的一张说明图表在 GPU 小时成本中并未包含总资本支出、R&D 支出或数据生成成本。
   - 原作者因将成本细节误传为某种“辟谣”而受到批评，突显了框架界定缺乏透明度。



**Link mentioned**: <a href="https://x.com/teortaxesTex/status/1877467302989295673/photo/1,">Tweet from Teortaxes▶️ (@teortaxesTex)</a>: @natolambert 我在实质内容上表示同意，但你为什么要把它表现得像是在辟谣？他们在那儿明确说了 GPU 小时数 * $/小时并不包括他们的总资本支出、R&D 支出或数据生成成本。（而且是我...

  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1326646270244294758)** (33 条消息🔥): 

> `SmolLM Corpus 更新，高效训练模型，用于研究的 Modal，SciAgents 讨论，GPT-NeoX 框架` 


- **SmolLM Corpus 生成延迟**：一位成员宣布完整的 SmolLM Corpus 正在生成中，将以 `jsonl.zst` 格式提供，但由于其 **320GB** 的巨大体积和 **23698 个分片 (shards)**，预计要到本周末才能完成。
   - 这一新格式被认为比之前 **1TB 未压缩** 的版本更具可用性。
- **业余爱好者探索研究**：成员们讨论了作为业余爱好者进行有趣研究的可行性，建议使用 **Modal, Colab 和 Kaggle** 进行低成本的训练和分析。
   - **Modal** 等平台提供的显著额度 (credits) 和免费层级使其对小型项目非常有吸引力。
- **Modal 的慷慨提供**：Modal 因其在运行超出个人 GPU 处理能力的更大型任务（尤其是推理和应用）方面的实用性而受到称赞。
   - 成员们强调了其慷慨的每月额度和对研究的支持，使其成为开发者的一个极具吸引力的选择。
- **关于 SciAgents 的讨论**：成员们讨论了 SciAgents 论文，该论文探讨了使用 **本体知识图谱 (ontological knowledge graphs)** 和多 Agent 系统来增强研究能力。
   - 虽然有人指出这可能不是一项突破性进展，但该方法在高级学习编排方面的潜力受到了赞赏。
- **了解 GPT-NeoX 框架的优先级**：一位成员深入介绍了 GPT-NeoX 框架的目标，强调在模型训练过程中性能与灵活性之间存在权衡。
   - 虽然能够处理多样化的任务，但他们警告说，由于其性能驱动的设计，GPT-NeoX 最适合以 Transformer 为中心的工作负载。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2409.05556">SciAgents: Automating scientific discovery through multi-agent intelligent graph reasoning</a>: 人工智能领域的一个关键挑战是创建能够通过探索新领域、识别复杂模式和揭示...来自主推进科学理解的系统。</li><li><a href="https://huggingface.co/spaces/Vokturz/can-it-run-llm">Can You Run It? LLM version - a Hugging Face Space by Vokturz</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Avelina/python-edu">Avelina/python-edu · Datasets at Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1326751257628508171)** (42 messages🔥): 

> `Grokking phenomenon, Weight decay strategies, Auxiliary loss functions, Softmax and sigmoid applications, Attention mechanisms` 


- **探索 Grokking 和 Softmax Collapse**：讨论强调了 **Grokking** 现象，即延迟泛化对深度学习理解构成的挑战，并重点关注 **Softmax Collapse** 这一障碍。
   - *如果没有正则化，模型会推向数值不稳定*，使 Grokking 任务复杂化，并需要更深层次的干预。
- **LLM 中的 Weight Decay 之争**：**极端 Weight decay**（通常设置为 **0.1**）已成为许多大型语言模型应对优化问题的常见做法。
   - 成员们讨论了**较低的 Weight decay** 是否对 Attention 层更有利，以减少低秩问题。
- **用于改进的辅助损失函数 (Auxiliary Loss Functions)**：成员们提出了替代方案，建议使用类似 *abs(norm(logits) - 1.0)* 的辅助损失，在不进行剧烈修改的情况下改进优化。
   - *在 Softmax 中使用 Softcap* 也可能在保持鲁棒性的同时加快处理速度，这表明了集成更简单调整的潜在趋势。
- **Unit Scaling 辩论**：对话指向了 **Unit scaling** 作为有效管理模型输出和梯度的必要机制的想法。
   - 有人指出，虽然 Unit scaling 在理论上感觉是正确的，但其在语言模型中的实际应用仍需进一步探索。
- **Softmax 在 Attention 和 Loss 中的作用**：成员们辩论了 **Softmax 与 Sigmoid 损失** 的功效，特别是在 Attention 可能不需要紧密分离值的背景下。
   - 在语言损失场景中，对 Softmax 压制概率的担忧浮现，建议应探索不同的机制以获得最佳性能。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04697">Grokking at the Edge of Numerical Stability</a>：Grokking 是在长时间过拟合后突然发生的泛化，是一个挑战我们对深度学习理解的惊人现象。虽然在理解方面已经取得了重大进展...</li><li><a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>：我们提出了一个新颖的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模得出特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)....</li><li><a href="https://arxiv.org/abs/2501.04519">rStar-Math: Small LLMs Can Master Math Reasoning with Self-Evolved Deep Thinking</a>：我们展示了 rStar-Math，证明了小型语言模型 (SLMs) 可以在不使用优越模型蒸馏的情况下，媲美甚至超越 OpenAI o1 的数学推理能力。rStar-Math 实现了...</li><li><a href="https://arxiv.org/abs/2411.04282">Language Models are Hidden Reasoners: Unlocking Latent Reasoning Capabilities via Self-Rewarding</a>：大型语言模型 (LLMs) 展示了令人印象深刻的能力，但在需要多个步骤的复杂推理任务中仍然面临困难。虽然像 Chain-of-Thought (CoT) 这样基于提示的方法可以改进...</li><li><a href="https://x.com/rm_rafailov/status/1877446475271037314">Rafael Rafailov @ NeurIPS (@rm_rafailov) 的推文</a>：我们关于“推理时计算 (inference time compute)”以及过去几个月工作的新立场论文！我们提出了一些关于为什么它是必要的、它是如何工作的、为什么我们需要它的理论...
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1326840285199732768)** (6 条消息): 

> `Pretraining 7B Llama2 Style Model, Memory Usage Analysis for GPU Models, Testing 6.7B Model Configurations` 


- **预训练 7B Llama2 具有挑战性**：一位用户尝试为 **7B Llama2** 模型设置预训练，但即使在 **batch size** 为 1 时也遇到了 **OOM** 问题，而 **1.3B** 模型则运行正常。
   - 他们怀疑问题出在将 **model_parallel** 设置为 2 时，并已在不同节点上测试了各种配置。
- **WandB 运行日志显示缺少依赖项**：在 **Llama 2 Config** 挂起期间，创建了一个 WandB 运行，但因日志建议安装 **boto3** 和 **hf_transfer** 以进行 S3 **checkpointing** 而停止。
   - 这些消息可能表明未满足的需求可能会影响运行进度，使其在 **checkpointing** 阶段停止。
- **内存使用分析请求**：一位用户请求使用不同的模型并行（**model parallelism**）设置，提供 **1.3B** 和 **2.7B** 模型每个 GPU 的内存使用报告。
   - 他们指出，即使模型没有出现 OOM，过高的 **VRAM usage** 也有助于调试问题。
- **使用不同配置测试 6.7B 模型**：有人提出疑问，当使用 **model_parallel = 1** 但 **pipeline_parallel = 2** 时，**6.7B 模型** 是否会发生 OOM。
   - 该用户表示尚未测试此配置，但计划在第二天进行测试。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/boto/boto3">GitHub - boto/boto3: AWS SDK for Python</a>：适用于 Python 的 AWS SDK。通过在 GitHub 上创建 account 来为 boto/boto3 的开发做出贡献。</li><li><a href="https://github.com/huggingface/hf_transfer">GitHub - huggingface/hf_transfer</a>：通过在 GitHub 上创建 account 来为 huggingface/hf_transfer 的开发做出贡献。</li><li><a href="https://gist.github.com/aflah02/cbbcff84509ea3490604199c308ecf53">6-7B.yml</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/aa7bc6ef2bb4fda5d62fb102f399848b">local_setup_wandb_modified_with_slurm.yml</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/fa5a3f2bf6891e8d8b9cb14da2777bb8">pretrain_6_7B.sh</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/e1541111956d9721b125ffc1ff34cd93">out_file_slurm.txt</a>：GitHub Gist：即时分享代码、笔记和代码片段。</li><li><a href="https://gist.github.com/aflah02/560436b0c0263b642724b69199898695">err_file_slurm.txt</a>：GitHub Gist：即时分享代码、笔记和代码片段。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1326667086910455809)** (10 条消息🔥): 

> `NCU profile comparison, Scams in the community, Learning Triton/CUDA for small GPU setups, Distributed training alternatives, Accelerating LLM inference` 


- **针对 GPU 设置的 NCU Profile 对比**：有人建议对比 **32x32 vs 16x16** 配置的 **NCU profile**，这应该能深入了解它们的性能差异。
   - 分析这些 profile 可以澄清与训练设置相关的性能特征。
- **警惕社区中的诈骗行为**：频道中出现疑似诈骗者的担忧被提出，敦促成员不要向与比特币相关的**无关讨论**关联的某些账户汇款。
   - 提到了有关努力将此人的行为引起管理员注意的证据。
- **学习 Triton/CUDA 的价值**：一位成员提出，在使用较少数量的 GPU（如 **8xH100**）时，学习 **Triton/CUDA** 是否值得。
   - 回复强调，理解这些语言可以提高代码质量，并加深对 GPU 运作方式的知识深度。
- **没有基础设施时的分布式训练选择**：一位成员询问在没有直接访问大规模基础设施的情况下，进行 **distributed training** 实验的选择。
   - 建议包括探索 **jax** 等框架，并指出 **accelerate/torch lightning** 的改进使该过程更加用户友好。
- **寻求 LLM 推理的长上下文基准测试**：一位成员正在致力于增强 LLM 推理的 **decoding**，并寻求具有大量输出生成的长上下文（**long context**）**end-to-end benchmarks**。
   - 他们指出，由于现有的基准测试侧重于较短的 prompt 生成，因此在评估运行时间方面存在挑战。


  

---

### **GPU MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1326642293784055909)** (8 messages🔥): 

> `WGMMA 计算，Fused MLP 的 Triton 实现，Profiling Triton 操作，教程示例中的错误` 


- **WGMMA 需要 warp 计算拆分**：*WGMMA* 要求能够将计算拆分到 **4 个 warps**，最小尺寸为 **16**，这意味着 tile 至少需要为 **64**。
   - 这确认了有效利用 WGMMA 的必要条件。
- **关于 Triton MLP 实现的咨询**：一位用户寻求关于 NVlabs 的 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 框架中 fused MLP 的任何 Triton 实现。
   - 他们还质疑缺乏片上 MLP 利用是否是因为其在大多数应用中被认为微不足道。
- **讨论 Profiling Triton 操作**：成员们讨论了如何对 Triton 操作进行 profiling，指出在原生 Torch 和 CUDA runtime 中，通常使用 **nsys** 和 **torch profiler**。
   - 有人提到 **Proton** 可以用于 Triton profiling，同时 **NCU** 也提供额外的 profiling 支持。
- **用于 Triton Profiling 的 Proton 工具**：一位用户分享了一个 [YouTube 视频](https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb)，解释了 **Proton** 工具，该工具可以辅助编写 Triton kernels。
   - 该工具在调试场景中特别有用，使 Triton 的开发更加容易。
- **教程示例中的错误**：有报告称在运行教程示例时，由于使用 **get_active_torch_device()** 遇到了 *AttributeError*。
   - 解决方案是改用 **torch.device('cuda')**，从而成功解决了该问题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/Av1za_0o2Qs?si=k6G2zWMzDbKTt5rb">Dev Tools: Proton/Interpreter</a>：Keren 谈到了可以帮助编写 Triton kernels 的工具——特别是 Triton 解释器，它对于调试 Illeg... 等问题非常有帮助。</li><li><a href="https://github.com/NVlabs/tiny-cuda-nn">GitHub - NVlabs/tiny-cuda-nn: 极速 C++/CUDA 神经网络框架</a>：极速 C++/CUDA 神经网络框架。可以通过在 GitHub 上创建账户来为 NVlabs/tiny-cuda-nn 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **GPU MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1326808364381896767)** (14 messages🔥): 

> `CUDA 驱动的重要性，Memory Banking 讲座，CUDA Kernel 编程，Blackwell vs Hopper，Discord 文件上传技巧` 


- **没有 GPU 时 CUDA 驱动也是必需的**：一位成员强调，如果没有 **Nvidia GPU**，你将无法运行 **CUDA kernels**，无论版本如何，**Nvidia 驱动** 对于 CUDA 功能都是必要的。
   - 该成员确认，由于缺少 GPU，尝试运行 **nvidia-smi** 失败了，并指出虽然某些代码需要 NVIDIA API，但如果没有 GPU，驱动程序就是多余的。
- **寻求 CUDA Kernel 开发指导**：一位初学者请求帮助编写一个简单的 **CUDA kernel**，用于计算 **2D N x N 矩阵** 的最大值和平均值，并表示愿意分享代码以寻求帮助。
   - 另一位成员提供了支持，并建议在 Discord 中上传 CUDA 文件时使用 **.cpp** 扩展名，以便获得可展开的预览，并推荐使用专门的提问频道。
- **探究 Blackwell 的 CUDA 模型增强**：一位成员询问 **Blackwell** 是否会像 **Hopper** 那样对 **CUDA 编程模型** 引入重大补充。
   - 他们还询问了两种架构中优化 kernels 的相似性，特别是关于 **producer-consumer** 和 **async tensor core instructions** 等构造。


  

---


### **GPU MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1326788513307561984)** (2 messages): 

> `Nectar Social 职位空缺，GPU 咨询公司招聘` 


- **Nectar Social 提供 1 万美元推荐奖金**：Nectar Social 是一家专注于 **social commerce** 的早期 AI 初创公司，正在 **Seattle** 招聘多个职位，包括 **Sr/Staff Product Manager**、**LLM/AI Engineer** 和 **Infra Engineer**。
   - 他们提供高达 **10,000 美元** 的 **推荐奖金**，并乐于私下分享更多细节，强调了过往 **初创公司经验** 的重要性。
- **欧洲咨询公司寻求开发者**：一家总部位于 **Amsterdam** 和 **Budapest** 的欧洲咨询公司正在为 GPU 和 HPC 软件项目寻找具备 **CUDA**、**HIP** 和 **OpenCL** 专业知识的开发者。
   - 他们与 **AMD** 等客户紧密合作，开发 **rocPRIM** 和 **hipCUB** 等核心库，感兴趣的候选人可以在[这里](https://www.linkedin.com/jobs/view/4120980579/)找到更多详情。


  

---

### **GPU MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1326810683538669590)** (3 messages): 

> `Installing CUDA on Ubuntu, Getting started with MacBook, Alternatives to NVIDIA GPU` 


- **Ubuntu 安装 CUDA 指南**：对于那些希望在 **Ubuntu** 上安装 **CUDA** 的用户，[NVIDIA CUDA Installation Guide for Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu) 提供了针对各种 Ubuntu 版本的详尽说明。
   - 该指南强调 **CUDA** 是一个并行计算平台，能够利用 GPU 能力提升计算性能。
- **在没有 NVIDIA GPU 的 MacBook 上开始**：一位用户表达了对在缺乏 **NVIDIA GPU** 的 **MacBook** 上开始课程的担忧。
   - 另一位成员提醒说，大多数涉及 **CUDA** 的项目在没有 NVIDIA GPU 的情况下无法运行，并建议使用 **Google Colab** 或云服务商进行动手实践。



**提及的链接**：<a href="https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#ubuntu">CUDA Installation Guide for Linux</a>：未找到描述

  

---


### **GPU MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/)** (1 messages): 

kashimoo: 我女朋友说我梦话都在说 CUDA 😭
  

---


### **GPU MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1326856718063243325)** (24 messages🔥): 

> `GPU Occupancy, MI210 Performance Analysis, RX 7900XTX Computations, CDNA Architecture Insights, Kernel Launch Dynamics` 


- **分析 GPU Occupancy 数值**：围绕 **MI210** 的最大 Occupancy 数值展开了讨论，由于与文档（包括一篇 2017 年的文章）中的预期值相比出现了非整数，从而引发了困惑。
   - *Occupancy and Resource Usage Optimization with Large Thread Groups* 强调了计算这些性能指标的复杂性。
- **MI210 和 RX 7900XTX 的 Occupancy 数值**：MI210 的 **rocminfo** 属性显示了潜在的计算结果，表明每个 CU 最多有 2 个 Block，以及导致对活跃 warps 进行解读的预期 Occupancy 指标。
   - 对于 **RX 7900XTX**，类似的计算指向预期的 Occupancy 为 **16**，符合架构预期。
- **Kernel 启动和 Occupancy 约束**：对 **CDNA1** 的见解表明，虽然理论 Occupancy 是 **10**，但由于 GPU 分级配置（binned configurations），实际使用中每个 Kernel 启动被限制在 **8** 左右。
   - 这些结果表明，只有通过同时启动多个 Kernel 才能实现更高的 Occupancy，更新后的测试确认了正确的 Block 性能。
- **Block 启动动态**：讨论注意到了 **MI210** 的独特行为，由于前一个 Block 的线程在 Block 完成前提前退出，允许每个 CU 有更多的 Block。
   - 这一观察引发了关于 Kernel 优化的对话，其中添加 `__syncthreads()` 会直接影响 Block 限制。



**提及的链接**：<a href="https://gpuopen.com/learn/optimizing-gpu-occupancy-resource-usage-large-thread-groups/">Optimizing GPU occupancy and resource usage with large thread groups</a>：Second Order Ltd 联合创始人 Sebastian Aaltonen 讨论了如何优化使用大线程组的 compute shaders 的 GPU Occupancy 和资源使用。

  

---

### **GPU MODE ▷ #[self-promotion](https://discord.com/channels/1189498204333543425/1288557096404516945/1326867199893307422)** (1 messages): 

> `MicroDiT replication, Architectural improvements with DCAE, MMDIT for prompt adherence, Compute grants for experiments` 


- **MicroDiT 复现完成**：一名成员宣布完成了 **MicroDiT** 复现项目，并提供了 [权重下载链接](https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt) 以及 [推理脚本](https://github.com/SwayStar123/microdiffusion/blob/main/test_model.ipynb)。
   - *“我想我可能快搞定了，”* 他们表示，并对获得的计算支持表示感谢。
- **使用 DCAE 改进架构**：计划使用 **DCAE** 作为自动编码器来增强 **MicroDiT** 的架构，以获得更好的性能。
   - 此外，目标是利用 **MMDIT** 来提高模型训练期间的提示词遵循度（prompt adherence）。
- **寻求算力资助**：该成员正在寻求 **compute grants**（算力资助）以加速其正在进行的实验，并指出其 **home GPU** 的性能不足以处理目前的任务。
   - 他们对目前进行高级 AI 实验的资源限制感到沮丧。



**提及链接**：<a href="https://x.com/SwayStar123/status/1854884660981219399">sway (@SwayStar123) 的推文</a>：MicroDiT 复现已完成。在此下载权重：https://huggingface.co/SwayStar123/MicroDiT/blob/main/no_cfg/microdit_model_epoch_19.pt 推理脚本在此：https://github.com/SwayStar123/mic...

  

---


### **GPU MODE ▷ #[🍿](https://discord.com/channels/1189498204333543425/1298372518293274644/1326649425732436008)** (2 messages): 

> `Alpha Competition, Softmax Kernel Performance` 


- **首届 Alpha 竞赛启动！**：宣布在测试服务器上启动首届 [alpha competition](https://link.to.competition)，邀请竞争者加入。
   - 对于有兴趣争夺最快 **softmax kernel** 的人，*请给我发私信，我会发送邀请*。
- **征集竞赛参与者**：一名成员鼓励所有感兴趣的人参加专注于 **softmax kernel** 性能的竞赛。
   - 对于希望展示其 **kernel** 优化技能的开发者来说，这是一个激动人心的机会。


  

---


### **GPU MODE ▷ #[thunderkittens](https://discord.com/channels/1189498204333543425/1300872762163728550/1326648370105684088)** (3 messages): 

> `ThunderKittens repository, Collaboration on kernel development, CPP harness usage` 


- **ThunderKittens GitHub 仓库资源**：你可以使用 [ThunderKittens GitHub 仓库](https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python) 中的代码复现该问题。该仓库专注于 **用于快速 kernel 的 tile 原语**，并包含各种开发资源。
   - 测试中使用的可视化效果是基于 **C++ 数值** 的，并且可以在 **harness** 中进行调整，以自定义 **sequence length** 和 **batch size**。
- **寻求 Kernel 开发合作伙伴**：发出了关于探索新 **kernel** 的协作邀请，包括 **MoE** 和 **DeepSeek Attention**。团队渴望与任何有兴趣 **贡献** 或 **学习 ThunderKittens** 的人建立联系。
   - 他们鼓励围绕对仓库的潜在贡献进行讨论，邀请热心的成员加入。



**提及链接**：<a href="https://github.com/HazyResearch/ThunderKittens/tree/main/tests/python">ThunderKittens/tests/python at main · HazyResearch/ThunderKittens</a>：用于快速 kernel 的 Tile 原语。通过在 GitHub 上创建账号来为 HazyResearch/ThunderKittens 的开发做出贡献。

  

---

### **GPU MODE ▷ #[arc-agi-2](https://discord.com/channels/1189498204333543425/1316377974672588850/1326819116644040704)** (7 messages): 

> `ARC Prize 演进, Rejection Sampling 基准实验, 探索 ARC 任务的文本域, Meta CoT 论文发现, 模型中的位置编码` 


- **ARC Prize 演变为非营利基金会**：[ARC Prize 正在演变为一个成熟的非营利基金会](https://x.com/fchollet/status/1877069518171943000)，以引导迈向 AGI 的研究进展，@GregKamradt 担任主席。
   - 这一举措旨在加强努力并进一步推进 ARC 社区的使命，正如其转型所强调的那样。
- **设置 Rejection Sampling 基准**：一位成员宣布正在准备一个简单的 [rejection sampling 基准实验](https://arcprize.org/blog/arc-prize-2025)，计划当晚运行。
   - 该实验旨在建立一个用于评估的基础基准。
- **ARC 的文本域探索**：由于 **GPU 限制**，**文本域（text-domain）**的探索被优先考虑，同时计划未来扩展以包含视觉输入。
   - 向对该项目感兴趣的其他成员发出了在这一方向上进行协作的邀请。
- **Meta CoT 论文强调不足之处**：**Meta CoT 论文**提出了关于传统 CoT 方法局限性的关键点，表明它们通常无法满足要求（[阅读论文](https://arxiv.org/abs/2501.04682)）。
   - 作者提供的见解可能会重塑对 AI 中上下文推理（contextual reasoning）的理解。
- **自定义位置编码提升性能**：一位成员分享说，他们的模型受益于**自定义的位置编码嵌入**而非传统方法，从而提升了性能。
   - 这一见解引发了关于定制化输入表示相对于原生（vanilla）方法潜在优势的讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2501.04682">Towards System 2 Reasoning in LLMs: Learning How to Think With Meta Chain-of-Thought</a>：我们提出了一个新颖的框架 Meta Chain-of-Thought (Meta-CoT)，它通过显式建模得出特定 CoT 所需的底层推理，扩展了传统的 Chain-of-Thought (CoT)....</li><li><a href="https://x.com/fchollet/status/1877069518171943000">来自 François Chollet (@fchollet) 的推文</a>：ARC Prize 正在演变为一个成熟的非营利基金会，以进一步履行我们引导和加速迈向 AGI 研究进展的使命。特别感谢 @GregKamradt，他将领导...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1326779084004528190)** (47 messages🔥): 

> `贡献 GPU 用于训练, 开源 DisTrO, DeepSeek V3 性能对比, Hermes 模型审查, Cursor vs WebStorm/PyCharm` 


- **关于贡献 GPU 进行训练的咨询**：一位新成员表示有兴趣贡献其 GPU 用于训练，但被告知目前还不行，并建议继续关注。
   - 这凸显了社区内对协作训练工作的持续兴趣。
- **关于 DisTrO 开源状态的澄清**：一位成员询问了 **DisTrO** 的开源情况，另一位成员确认已经开源，并提供了 Twitter 上分享的相关资源链接。
   - 成员们已经开始在他们的训练器（trainers）中实现它，表明了持续的实际应用。
- **对比 DeepSeek V3 的输出**：一位成员注意到 DeepSeek V3 的不同体验，特别是官方 API 提供的回答比 Hyperbolic 等第三方提供商更具重复性。
   - 另一位成员推测这可能是由于官方 API 采用了激进的缓存策略，强调了体验的差异性。
- **关于 Hermes 模型审查的讨论**：关于 **Hermes** 模型的审查制度出现了疑问，澄清该模型大部分是未审查的（uncensored），但需要特定的 Prompt 来引导行为。
   - 这引发了如下见解：许多未审查模型对 Prompt 表现出类似的条件响应。
- **评估 Cursor 相对于 IDE 的有效性**：有人担心 **Cursor** 是否值得从 **WebStorm** 和 **PyCharm** 等流行 IDE 切换过去，一位成员声称这可能不值得。
   - 其他人同意，只要用户理解自己的代码，各种 AI 自动补全工具都能提供类似的生产力提升。


  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1326966515295326298)** (2 条消息): 

> `Reducing Memory Usage, Open Source Function Calling Models, Qwen2.5-32B-Instruct-AWQ, Function Calling Benchmarks` 


- **寻求减少显存占用的建议**：一位成员正在寻找在拥有 **24 GB VRAM** 的 RTX 4090 上运行 **Qwen2.5-32B-Instruct-AWQ** 模型时，减少 **显存占用（memory usage）** 的策略，因为目前遇到了显存溢出（out-of-memory）错误。
   - *启用 Flash Attention 对 VRAM 占用没有影响*，且输入上下文长度约为 **6K tokens**。
- **关于最佳开源 Function Calling 模型的咨询**：另一位成员询问了目前可用的 **最佳开源 Function Calling 模型**，以及是否有追踪 **Function Calling 准确率** 百分比的 Benchmark。
   - 他们还询问了在 **Post-training Pipeline** 中，哪些因素有助于提高模型的有效性。


  

---


### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 条消息): 

> `Research Ideas, Carson's Personal Site, Forefront.ai, Simple AI Software` 


- **Carson Poole 的研究想法库**：Carson Poole 指出，许多研究想法已经转化为论文，可以在他的个人网站 [这里](https://poole.ai) 找到。值得注意的想法包括 [ReLoRA](https://arxiv.org/abs/2307.05695) 和 [Sparse Upcycling](https://arxiv.org/abs/2212.05055)，这两者最初都在 2022 年 11 月进行了讨论。
- **关注 Forefront.ai**：Carson Poole 是 [Forefront.ai](https://forefront.ai) 的联合创始人，他邀请大家探索该公司的创新 AI 解决方案。他还推广了他在 [Simple AI Software](https://simpleaisoftware.com) 上的工作，表明其专注于易用的 AI 工具。
- **通过邮件联系 Carson 进行合作**：Carson 鼓励感兴趣的人士通过电子邮件联系以寻求合作机会。他的邮箱受到保护，但可以通过他个人网站上提供的链接访问。



**提到的链接**：<a href="https://poole.ai">Carson Poole 的个人网站</a>：未找到描述

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1326790180220436532)** (11 条消息🔥): 

> `Microsoft's rStar-Math, Qwen 7B AIME performance, LLMs and reasoning capabilities, Math usefulness, Trustworthiness of LLMs in math` 


- **微软的 rStar-Math 超越基准测试**：微软展示了 rStar-Math，使 **Qwen 2.5-Math-7B** 在 MATH 基准测试中从 **58.8%** 提升至 **90.0%**，并在 AIME 上达到 **53.3%**，排名进入高中生前 **20%**。
   - *自我进化的深度思考（Self-evolved deep thinking）* 使这些小型 LLM 表现出色，突显了数学推理能力的重大进步。
- **关于 LLM 数学实用性的辩论**：*Kotykd* 质疑了 LLM 解决数学问题的实用性，建议应更多地关注代码和通用推理，而其他人则认可了其在验证方面的益处。
   - 大家达成共识，虽然 **数学很有用**，但 LLM 目前在精确应用方面缺乏可信度。
- **LLM 的推理能力受到审视**：*Stefangliga* 指出，认为数学能力必然意味着推理能力的看法是一种误解，LLM 在这两者之间表现出了脱节。
   - 正如 *kotykd* 所阐述的，真正的推理涉及适应新问题，而不仅仅是遵守一致的数学规则。



**提到的链接**：<a href="https://x.com/altryne/status/1877220144725758414?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：呃伙计们……微软刚刚让 Qwen 7B 在 AIME 上的表现达到了 o1 的水平 😵‍💫 他们还展示了通过其 MCTS 驱动过程，模型具备了像推理模型一样的自我反思能力。是否……

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1326659506599366727)** (3 条消息): 

> `Carson Poole 的研究想法、联系方式及背景` 


- **Carson Poole 分享了值得探索的研究想法**：Carson Poole 列出了几篇研究论文和概念，包括 [ReLoRA](https://arxiv.org/abs/2307.05695) 和 [Sparse Upcycling](https://arxiv.org/abs/2212.05055)，这些内容引起了社区的兴趣。
   - *这些想法分别于 2022 年 11 月和 2023 年 3 月首次被提及，突显了它们在当前研究讨论中的相关性。*
- **Carson Poole 的联系方式和职业背景**：在介绍中，Carson 自述为 [Forefront.ai](https://forefront.ai) 的联合创始人，并分享了他关于 [Simple AI Software](https://simpleaisoftware.com) 工作的链接。
   - 他鼓励成员通过电子邮件 [protected email link] 与他联系以进行进一步讨论。



**提及的链接**：<a href="https://poole.ai">Carson Poole 的个人网站</a>：未找到描述

  

---


### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1326679514205392918)** (47 条消息🔥): 

> `Salesforce AI 招聘冻结、OpenAI 产品更新、Anthropic 融资新闻、AI 职业机会、Google DeepMind 合并` 


- **Salesforce 宣布冻结软件工程师招聘**：Marc Benioff 表示，Salesforce 在 2025 年将不会招聘任何软件工程师，并将这一决定归功于其 AI 技术 Agentforce 带来的 **30% 生产力提升**。
   - Benioff 强调，随着业务计划的演变，Agentforce 仍是公司的核心重点。
- **OpenAI 更新自定义指令界面**：用户注意到 OpenAI 最近的更新正在影响其高级语音功能的自定义指令，预计很快将测试新功能。
   - 一段展示这些变化的视频正在制作中，表明用户体验正在持续改进。
- **Anthropic 获得重大融资助力**：Anthropic 正在额外筹集 20 亿美元，将其估值推高至 **600 亿美元**，并强调其最近主要来自商业销售的 **8.75 亿美元** 年度经常性收入。
   - 这一重大投资凸显了人们对 AI 驱动解决方案日益增长的兴趣以及公司的未来发展方向。
- **职业重心向 AI 和销售转移**：对 “AI Engineer” 和 “AI Consultant” 角色的需求正在飙升，反映了行业内的快速增长，因为公司正在寻求专业知识。
   - 对话表明，个人可能需要适应销售工程等角色，或开始自己的小型业务，以有效地利用其技术技能。
- **Google 将 AI 部门并入 DeepMind**：Google 正在将多个 AI 产品合并到 DeepMind 之下，引发了对其未来运营结构和效率的猜测。
   - 尽管如此，人们仍对 Google 冗余的流程以及其 LLM 模型缺乏流线型产品表示担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/topmass/status/1877444315871326422?s=46">来自 topmass (@topmass) 的推文</a>：正当我录制视频展示如何让 ChatGPT 高级语音模式（Advanced Voice）变得更好时，@OpenAI 正在发布一个更新，该更新破坏了自定义指令（Custom Instructions），但似乎也增加了新功能……</li><li><a href="https://x.com/natolambert/status/1877020436246204596?s=46">来自 Nathan Lambert (@natolambert) 的推文</a>：我重新录制了我们在 NeurIPS 上关于语言模型后训练（Post-training）部分的教程，增加了一些幻灯片，并在 @interconnectsai 上写了一篇微型现状报告。请享用！链接在引用推文中。00:00 介绍……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxm0ep/anyone_want_the_script_to_run_moondream_2bs_new/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/osanseviero/status/1877452798683430988">来自 Omar Sanseviero (@osanseviero) 的推文</a>：我非常激动地分享，我们（AI Studio, Gemma, Gemini API）将加入 Google DeepMind！😱 2025 年对于开放模型、可访问的研究以及面向开发者的出色工具来说，将是非常令人兴奋的一年……</li><li><a href="https://x.com/andrewcurran_/status/1876705929296581078?s=46">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：Anthropic 正在筹集另外 20 亿美元。这一轮融资将使 Anthropic 的估值达到 600 亿美元，是去年的三倍多。据《华尔街日报》报道，其年度经常性收入（ARR）最近达到了约 8.75 亿美元……</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/1hxjzol/new_moondream_2b_vision_language_model_release">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://x.com/tsarnick/status/1877089046528217269">来自 Tsarathustra (@tsarnick) 的推文</a>：François Chollet 表示 OpenAI 的 o1 模型正在可能的思维链（Chain of Thought）空间中运行搜索过程，生成自然语言程序，并以一种“真正的突破……”方式适应新奇事物。</li><li><a href="https://www.interconnects.ai/p/the-state-of-post-training-2025">2025 年后训练（Post-training）的现状</a>：立即观看（54 分钟）| 我在 NeurIPS 上关于语言建模教程的重新录制版（加上了一些新增内容）。</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-benioff/">Marc Benioff 表示 Salesforce 在 2025 年将不再招聘软件工程师</a>：Salesforce 首席执行官 Marc Benioff 宣布不再招聘新的软件工程师——看看 AI 如何塑造公司的未来。</li><li><a href="https://www.salesforceben.com/salesforce-will-hire-no-more-software-engineers-in-2025-says-marc-beni">Marc Benioff 表示 Salesforce 在 2025 年将不再招聘软件工程师</a>：Salesforce 首席执行官 Marc Benioff 宣布不再招聘新的软件工程师——看看 AI 如何塑造公司的未来。</li><li><a href="https://github.com/EvanZhouDev/open-genmoji">GitHub - EvanZhouDev/open-genmoji: 为我们所有人准备的生成式表情符号（Generative Emoji）。</a>：为我们所有人准备的生成式表情符号。通过在 GitHub 上创建账户，为 EvanZhouDev/open-genmoji 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1326704638870683648)** (1 条消息): 

> `AI Agent 黑客松，OpenRouter API 额度，Live Agent Studio，Voiceflow 赞助，n8n 奖金增加` 


- **加入 oTTomator AI Agent 黑客松！**：参与者可以使用任何 LLM 创建 Agent，并领取 **$10 的 OpenRouter API 额度**，第一名奖金总额达 **$1,500**，亚军为 **$150**。注册现已开放，截止日期为 1 月 22 日，获奖名单将于 2 月 1 日公布；[在此注册](https://studio.ottomator.ai/hackathon/register)。
   - 这是一项个人竞赛，每人仅限提交一份作品，鼓励参与者查阅提供的协议和指南。
- **黑客松丰厚的现金奖励**：由 **Voiceflow** 和 **n8n** 赞助的 oTTomator Live Agent Studio 黑客松将提供 **$6,000** 的现金奖励！黑客松从 1 月 8 日持续到 1 月 22 日，社区投票将于 1 月 26 日至 2 月 1 日进行。
   - 参与者可以构建与 Live Agent Studio 兼容的 Agent，n8n 团队增加了奖金池，为最佳 n8n Agent 提供 **$700** 和 **$300** 的奖励！



**提到的链接**：<a href="https://studio.ottomator.ai/hackathon/register">oTTomator</a>：未找到描述

  

---

### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1326712396831785010)** (46 条消息🔥): 

> `OpenRouter UI 性能、Gemini Flash 与 API 问题、O1 响应格式、API 访问请求、Hanami 使用体验` 


- **OpenRouter UI 响应迟缓**：用户对 **OpenRouter 的 UI 性能**表示沮丧，称在聊天记录超过 **1k 行**后会出现明显卡顿，导致滚动和编辑变得繁琐。
   - 建议包括实现**按成本排序**以及优化 **Next.js 分页 (pagination)** 以提升整体用户体验。
- **Gemini Flash 表现异常**：有用户反映 **Gemini Flash** 虽然在聊天室中正常工作，但在通过 API 调用时无法运行，引起了用户困惑。
   - 一位用户还表达了对 **Gemini** 的喜爱，但也提到了性能问题和对功能改进的需求。
- **O1 的响应格式引发质疑**：多位用户批评 **O1 API** 的响应格式，它使用 **====** 而不是 ``` 来表示 Markdown，导致使用过程中出现奇怪的行为。
   - 讨论围绕这一变化是为了节省 Token 还是为了改进输出展开，大家对其影响持不同意见。
- **API 访问与开发咨询**：一位用户询问是否可以通过 OpenRouter 发布自己的 **LLM API**，表现出扩展平台服务的兴趣。
   - 另一位用户报告了 API 请求的问题并寻求帮助，强调了对更好基础设施支持的需求。
- **Hanami 使用讨论**：一位用户询问是否有人在使用 **Hanami**，此询问促使另一位用户分享了包含意外字符的测试结果。
   - 此次交流强调了社区对各种工具更稳健体验的需求。


  

---


### **Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1326655467577147412)** (1 条消息): 

> `CSV 文件下载、表格响应` 


- **支持将表格下载为 CSV 文件！**：新功能允许用户在查看表格时选择下载选项，直接从响应中将表格下载为 **CSV 文件**。
   - 此项增强功能发布时附带了一张说明[图片](https://cdn.discordapp.com/attachments/1047204950763122820/1326655467304255508/download_csv.jpg?ex=6781892f&is=678037af&hm=f69ea0b4635a0df0dfe206fdd64762dd6fd44a96818c6347e1f1aad37404e0fe&)。
- **CSV 功能增强了数据处理**：增加 CSV 下载功能显著改善了用户管理和利用表格数据的方式。
   - 预计该功能将为处理大量数据集的用户简化工作流程。


  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1326642921490874369)** (33 条消息🔥): 

> `Youzu.ai 用于室内设计，Perplexity 问题与 Bug，Discord 协作提案，Perplexity 翻译挑战，Ecosia 产品经理寻求合作` 


- **Youzu.ai 变革房间设计**：Youzu.ai 是首批 AI 驱动的工具之一，它在提供精美房间设计建议的同时，还提供本地购买选项，为用户节省了大量时间和压力。详细概述请查看[此处的指南](https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688)。
   - 一位用户分享了他们使用 Youzu.ai 的经验，并鼓励其他人尝试，表达了对反馈的期待。
- **Perplexity 宕机或运行困难？**：多位用户报告了 Perplexity 运行缓慢或无响应的问题，其中一位指出经常出现 'want more uploads?' 的消息。另一位用户询问 Perplexity 是否宕机，其他用户也表达了类似的经历。
   - 建议 Chrome 用户禁用 SimplyCodes 扩展程序，以避免刷新问题。
- **Discord 中的协作项目愿景**：一位成员表达了发起协作项目的愿望，旨在利用 Discord 小组内多样化的技能，强调无门槛的团队合作。他们强调该小组的人才储备是开展突破性工作的基础。
   - 同时也鼓励任何感兴趣的人参与贡献，无论投入时间多少。
- **使用 Perplexity 时的翻译困难**：一位用户在翻译一部韩国小说时面临挑战，称响应限制和生成内容的不准确是主要障碍。他们寻求改进在此场景下使用 Perplexity 体验的方法。
   - 社区给出了建议，并分享了使用 Perplexity 进行翻译的经验。
- **Ecosia 产品经理寻求合作**：来自植树搜索引擎 Ecosia 的一位产品经理寻求帮助，希望联系 Perplexity 讨论潜在的合作伙伴关系。他们表示很难找到促进此类讨论的联系点。
   - 回复中包括了关于如何正确联系或与 Perplexity 洽谈潜在交易的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://medium.com/design-bootcamp/youzu-ai-where-ai-interior-design-meets-real-world-shopping-76a066be3688">Youzu.ai: Where AI Interior Design Meets Real-World Shopping</a>：介绍全球首个由 AI 驱动的“从设计到购买”平台✨</li><li><a href="https://x.com/omidaziz/status/1877409601202631083?s=46">来自 omid (@omidaziz) 的推文</a>：设计得最好和最差的 AI 应用
</li>
</ul>

</div>
  

---


### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1326740486169427988)** (6 条消息): 

> `丰田的火箭探索，即将发布的视频游戏，IndyCar 车手统计数据，西班牙人的平均寿命，NVIDIA 家用超级计算机` 


- **丰田冲向新领域**：作为其创新战略的一部分，丰田正在探索[火箭](https://www.perplexity.ai/page/toyota-is-exploring-rockets-NrLusU2uRdaUqsCirISg7Q)领域的新业务。
   - 这一举动展示了丰田在传统汽车工程之外的雄心。
- **期待即将发布的视频游戏**：玩家们对即将上市的[下一波视频游戏](https://www.perplexity.ai/search/prochaines-sorties-de-jeux-vid-zgsehswCSLuZemsB7i3UYA)议论纷纷。
   - 预计这些作品将吸引游戏社区的极大关注和热情。
- **分析 IndyCar 车手平均水平**：对 [IndyCar 车手平均水平](https://www.perplexity.ai/search/indycar-driver-averages-mOBWLru4TWqQJrczuSDMtQ)的洞察揭示了对车迷和车队都至关重要的关键性能指标。
   - 了解这些平均数据可以增强车迷在比赛季的参与度。
- **西班牙人寿命洞察**：[西班牙人的平均寿命](https://www.perplexity.ai/search/average-lifespan-of-a-spaniard-OOT0EWBjS6ifrw142dFOwg#0)是一个值得注意的重要公共卫生指标。
   - 这一统计数据反映了西班牙的健康趋势和生活质量。
- **NVIDIA 家用超级计算机：3000 美元的投资**：NVIDIA 宣布了一款专为家庭使用设计的新型[超级计算机](https://www.perplexity.ai/page/ces-2025-nvidia-s-ai-supercomp-Eldo96kHTICxurNQVyCGbw)，售价为 3000 美元。
   - 这一创新旨在让科技爱好者更容易获得强大的计算能力。


  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1326717449072279654)** (3 messages): 

> `Korean Language API Usage, Other Language Models` 


- **韩语 API 使用请求**：一位用户寻求关于如何使用仅支持**韩语**的 API 的指导，同时排除 **llama-3.1-sonar-small, large, and huge** 模型。
   - 他们明确表示希望 API **仅提供韩语回复**。
- **Discord 对话链接**：一位用户分享了一个与在 API 中使用韩语主题相关的 [Discord 对话链接](https://discord.com/channels/1047197230748151888/1047202784090538054/1316804335258173460)。
   - 该链接似乎是关于语言模型和韩语使用情况持续讨论的一部分。


  

---


### **Cohere ▷ #[discussions](https://discord.com/channels/954421988141711382/954421988783444043/1326950230167392407)** (2 messages): 

> `North launch, AI workspace, Productivity tools, Cohere vs Microsoft Copilot, Cohere vs Google Vertex AI` 


- **Cohere 发布 North 以提升生产力**：Cohere 推出了 [North 的早期访问 (Early Access)](https://x.com/cohere/status/1877335657908949189)，这是一个一体化的安全 AI 工作区平台，将 LLM、搜索和 Agent 集成到一个直观的界面中，以增强生产力。
   - 此次发布旨在[超越 Microsoft Copilot 和 Google Vertex AI Agent Builder](https://cohere.com/blog/north-eap)，承诺无缝提升劳动力生产力和运营效率。
- **North 结合了多种 AI 功能**：North 将 LLM、搜索和自动化结合到一个安全的办公空间中，旨在简化日常任务并提高绩效。
   - 该平台旨在确保用户在有效管理其 AI 集成的同时实现峰值生产力。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/cohere/status/1877335657908949189">来自 cohere (@cohere) 的推文</a>：今天，我们开启了 North 的早期访问！我们的一体化安全 AI 工作区平台将 LLM、搜索和 Agent 结合到一个直观的界面中，毫不费力地将 AI 集成到您的日常工作中...</li><li><a href="https://cohere.com/blog/north-eap">North 简介：一个助你完成更多工作的安全 AI 工作区</a>：North 将 LLM、搜索和自动化结合到一个安全的 AI 工作区中。它的表现优于 Microsoft Copilot 和 Google Vertex AI Agent Builder，无缝提升了劳动力生产力和运营效率...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[questions](https://discord.com/channels/954421988141711382/1168411509542637578/1326828245412220955)** (7 messages): 

> `Command R+ models, Upgrading embeddings, Classification model limits, Alignment Evals Hackathon, Eval and Interp tutorials` 


- **探索用于生成模型的 Command R+**：一位成员提到，对于大型生成模型，应使用 **Command R+**，并引用了[模型概览文档](https://docs.cohere.com/docs/models)。他们还询问了使用该模型的预期工作流。
   - 这突显了在集成新模型时理解特定工作流的必要性。
- **升级 Embedding 的指南**：一位用户对从 **embed-v2** 过渡到 **v3** Embedding 表示担忧，理由是前者可能被弃用。他们正在寻求在无需大规模重新生成的情况下高效升级 Embedding 的方法。
   - 这反映了为大型数据集提供清晰升级路径的重要性。
- **处理分类模型的示例限制**：一位成员在尝试使用 **95,429 个已标记示例**进行文本分类时遇到错误，原因是单次请求存在 **2,500 个示例的限制**。他们询问了有效管理此限制的最佳方法。
   - 将大型数据集拆分为较小的批次可能是一个解决方案，但仍需明确最佳实践。
- **Alignment Evals Hackathon 公告**：一位用户宣布将于 25 日举办 **Alignment Evals Hackathon**，这是一个协作贡献的机会。他们提到将作为活动的一部分发布 Eval 和 Interp 教程。
   - 此活动鼓励社区积极参与和知识共享。
- **鼓励分享 Hackathon 成果**：成员们被鼓励在指定频道分享 Hackathon 的经验。这促进了社区参与和知识共享。
   - 分享成果可以带来协作改进和学习机会。


  

---

### **Cohere ▷ #[api-discussions](https://discord.com/channels/954421988141711382/1168578329423642786/1326682963806519296)** (26 条消息🔥): 

> `Cohere LLM API Recursive Loop Issue, Improving Model Generations, Expanding Token Limits, Rolling Chat History Technique, API Rate Limit Errors` 


- **Cohere LLM API 陷入循环**：一位用户报告称，使用 **Python ClientV2** 的 **Cohere LLM API** 有时会进入递归循环，无休止地添加单词，这可能会耗尽 Token 预算。
   - 建议包括实施带有 Token 上限的保护措施，以帮助控制失控的生成。
- **增强模型输出的技巧**：另一位用户分享了改进模型响应的方法，例如使用 **system message** 并设置 **max_tokens** 限制以防止过度生成。
   - 他们强调模型可以从示例提示和响应中学习，从而增强其提供简洁答案的能力。
- **关于输出 Token 限制的查询**：一位用户询问了 **Cohere 模型输出长度**（目前限制为 **4k tokens**）的潜在扩展。
   - 回复中提到 **cmd-r 系列模型**支持显著的输入长度，并讨论了使用滚动历史记录（rolling history）来管理更长的输出。
- **利用滚动聊天历史记录实现更大的输出**：一位成员建议采用 **rolling chat history** 技术，通过重用之前的输入，允许模型产生更长的响应。
   - 这种方法允许在遵守模型架构施加的上下文限制的同时进行持续生成。
- **对 API 速率限制错误的回应**：一位用户在访问 API 数据时遇到了 **TooManyRequestsError**，表明他们达到了服务的速率限制。
   - 建议检查他们使用的是 *trial 还是 production keys*，并考虑联系支持人员以寻求进一步帮助。


  

---


### **Cohere ▷ #[projects](https://discord.com/channels/954421988141711382/1218409701339828245/1326786815415291904)** (2 条消息): 

> `Channel Posting Rules` 


- **频道发布规则提醒**：一位成员提醒另一位成员阅读**规则**，并且只在**一个频道**中发布消息以保持组织有序。
   - 回复是 *Sorry, I will do that later*，表示打算遵守。
- **合规跟进**：该成员承认了提醒，并承诺稍后会遵守频道的指南。
   - 这次交流强调了在频道中持续增强沟通礼仪意识的必要性。


  

---


### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1326763968773095506)** (18 条消息🔥): 

> `Pull Request #8505 Retest, LLVM JIT and Autogen Integration, Function Signature Stability in LLVM, Bounty Payments, Testing Compatibility with LLVM Versions` 


- **重新测试 Pull Request #8505**：一位成员请求重新测试与 OSX 上的 MOCKGPU amd 相关的 [Pull Request #8505](https://github.com/tinygrad/tinygrad/pull/8505)，该 PR 依赖于另一个 Pull Request。
   - *George Hotz* 确认了该请求，证实此任务设有 **Bounty**，并准备通过 PayPal 或 Ethereum 上的 USDC 支付。
- **结合 LLVM JIT 和 Autogen 的工作**：一位成员提到 **PR #8486** 已准备好接受审查，并建议采用结合的方法来实现 **LLVM JIT** 和 **LLVM Autogen**。
   - 他们还对是否继续使用当前的多个版本文件或对其进行简化表示不确定。
- **对 LLVM 函数签名稳定性的担忧**：一位成员对 **LLVM** 中函数签名可能发生的静默更改表示担忧，这可能导致未定义行为。
   - *George Hotz* 安慰说这种更改不太可能发生，并表示倾向于支持最旧的版本。
- **讨论贡献的 Bounty 支付**：*George Hotz* 确认锁定了一项 **Bounty**，并表示欠下一项与 CLANG 相关任务的款项，正在核实用于支付的 Ethereum 地址。
   - 聊天机器人表示准备好结算另一项关于 PR 工作的 **Bounty**。
- **测试与 LLVM 版本 11 的兼容性**：一位成员表示他们可以测试与 **LLVM** 版本 **11** 的兼容性，并指出 **版本 14** 被用作参考。
   - 他们表示愿意验证远至版本 **11** 的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://llvm.org/docs/DeveloperPolicy.html">LLVM Developer Policy &#8212; LLVM 20.0.0git documentation</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/pull/8505">MOCKGPU amd test on OSX by patrini32 · Pull Request #8505 · tinygrad/tinygrad</a>: 依赖于 #8501
</li>
</ul>

</div>
  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1326927564308086785)** (4 messages): 

> `Blog Post on TinyGrad, Initializing Layers on Specific Devices` 


- **在博文中探索 TinyGrad 代码库**：一位成员分享了他们的博文 [TinyGrad Codebase Explained-ish](https://adelaloui.me/tinygrad-codebase-explained-ish/)，该文章概述了 **TinyGrad 的代码库结构**和**核心组件**。
   - 文章强调，核心 **tinygrad/** 目录之外的代码并未经过广泛测试，并建议除非代码损坏，否则不要轻易修改。
- **在指定设备上初始化权重**：一位成员询问是否可以为 **nn.Linear** 中的权重和偏置初始化指定设备，另一位成员回答了一个解决方案，即在 Tensor 实例化之前设置 `Device.DEFAULT`。
   - 他们提供了一系列设备选项，包括 **METAL**、**CUDA** 和 **CLANG**，并指出 **CLANG** 将使用 CPU。



**提及的链接**：<a href="https://adelaloui.me/tinygrad-codebase-explained-ish/">TinyGrad Codebase Explained-ish</a>：对 TinyGrad 的仓库结构和关键文件的详细解析（大概）。

  

---


### **Nomic.ai (GPT4All) ▷ #[general](https://discord.com/channels/1076964370942267462/1090427154141020190/1326643603430183044)** (22 messages🔥): 

> `Comparing Llama.cpp and GPT4All, Performance variations in models, Troubleshooting Chat Templates, Recommendations for roleplay models, Deployment of modernbert` 


- **Llama.cpp 与 GPT4All 显示出性能差距**：据指出，**Llama.cpp** 的 Vulkan 实现与 **GPT4All** 内部使用的版本有显著不同，性能差异巨大，特别是在由于 CUDA 能力而表现突出的 Nvidia GPUs 上。
   - 成员们表示，当性能足以满足任务需求时，这些差异的重要性就会降低。
- **AI 模型的 Chat Template 混淆**：一位用户报告了在 GPT4All 中为 TheBloke 的模型设置 **Chat Template** 时遇到的问题，尽管安装正确，但收到的响应非常通用。
   - 另一位成员建议查看 GitHub 上特定模型的指南，指出不同模型之间的模板可能存在显著差异。
- **推荐用于角色扮演的 Llama-3 模型**：对于 **COTE 动画**的角色扮演，建议使用 **Nous Hermes 2** 模型，虽然该模型较老，但在此类任务中仍然有效。
   - 用户被鼓励寻找 Nomic 提供的即插即用型 Llama 模型，这可以简化流程。
- **ModernBERT 的部署查询**：有人提出了关于部署 Nomic AI 的 **ModernBERT** 及其在 **text-embedding-inference** 或 **vLLM** 中的支持状态的问题。
   - 聊天记录中没有提供关于 modernbert 兼容性的明确指导。
- **GPT4All 中图像模型的探索**：一位用户询问了在 GPT4All 中加入**图像模型**的可能性，反映了对扩展模型能力的持续兴趣。
   - 虽然没有提供具体的回答，但讨论暗示了对更广泛模型集成的渴望。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/GPT4All-Community/phi-4-GGUF/blob/main/phi-4-Q4_0.gguf">phi-4-Q4_0.gguf · GPT4All-Community/phi-4-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/nomic-ai/gpt4all/issues/3365.">nomic-ai/gpt4all</a>：GPT4All：在任何设备上运行本地 LLMs。开源且可用于商业用途。 - nomic-ai/gpt4all</li><li><a href="https://huggingface.co/aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K">aifeifei798/llama3-8B-DarkIdol-2.2-Uncensored-1048K · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1326662062595641425)** (2 messages): 

> `GitHub HQ Event, Agentic Document Workflows, AI Agents Debugging, Fast Inference Systems, LlamaIndex Workflows` 


- **参加 GitHub 总部的专家讲座**：欢迎参加 [1 月 15 日在 GitHub 总部](https://twitter.com/llama_index/status/1877103276635848846)举行的一系列专家讲座，主题包括改进 AI agents、创建快速推理系统以及使用 **LlamaIndex** 构建工作流。
   - 该活动邀请了来自 **@arizeai**、**@GroqInc** 的演讲者，并分享关于 agentic 工作流的见解。
- **介绍 Agentic Document Workflows**：一篇新博客文章讨论了 **Agentic Document Workflows (ADW)**，旨在通过直接集成到业务流程中来重新定义文档处理，详见[此处帖子](https://twitter.com/llama_index/status/1877420085691953385)。
   - 文章强调**文档有多种格式**，重点是为未来的应用简化工作流。


  

---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1326725320505561091)** (18 条消息🔥): 

> `Ollama 更新、针对电子邮件限制的应用部署、Vector DB 索引、本地 TEI 服务器支持、QueryFusionRetriever 错误` 


- **Ollama 更新提升性能**：在 **Ollama** 最近一次更新后，用户报告评估时间降至 **3 秒**以下。
   - 自更新以来，用户注意到了*令人难以置信的性能提升*。
- **部署受限访问的应用**：一位用户询问如何部署仅限特定电子邮件地址访问的应用，并建议将 **Cloud Run + Google IAP** 作为一个选项。
   - 目标是确保**非技术用户**能够轻松使用。
- **Vector 元数据需要手动索引**：围绕 **VectorStoreIndex** 以及基于 **Postgres** JSON 字段中的元数据键过滤节点展开了讨论。
   - 成员们争论是需要**手动为数据库创建索引**，还是 LlamaIndex 可以处理此功能。
- **本地 TEI 服务器重排序能力**：用户探讨了是否可以利用**本地 TEI 服务器**进行重排序（reranking），并参考了相关的 API 和安装命令。
   - 一位用户指出 LlamaIndex 中对 **TEI + gRPC** 支持的潜在问题。
- **QueryFusionRetriever 中的输入 Token 限制错误**：一位用户报告在使用 **QueryFusionRetriever** 时出现**输入验证错误**，超过了 **518 tokens** 的限制。
   - 用户分享了相关代码片段，展示了尝试集成各种检索策略的过程。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/run-llama/llama_index/issues/9572">[功能请求]: Text Embeddings Inference 重排序器 · Issue #9572 · run-llama/llama_index</a>：功能描述：你好，我们能否为 Text Embeddings Inference 服务器提供一个类似于 SentenceTransformerRerank 或 CohereRerank 的重排序类？原因：我们遇到了性能/扩展性问题...</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/postprocessor/tei_rerank/">Tei rerank - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py">llama_index/llama-index-integrations/postprocessor/llama-index-postprocessor-tei-rerank/llama_index/postprocessor/tei_rerank/base.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1326677862723616880)** (18 条消息🔥): 

> `Rust 语法与类型约束、Mojo 中的重载解析、Mojo 中的量子计算库、MAX 与量子编程、Mojo 中的 Quojo 库` 


- **Rust 语法使多行实现更简单**：一位用户分享说，他们非常欣赏 **Rust 语法**在多行实现中的便捷性，特别是在为 **multipaxos** 创建 actor 时。
   - *某些函数参数可能变得过于冗长，使得用户在确定必要类型时感到困扰。*
- **对重载解析顺序的担忧**：另一位用户表示担心在大型代码库中调整重载顺序会变得很麻烦，并建议将 **'happens after' 注解**作为一个潜在的解决方案。
   - *他们还表示，'TraitVariant' 概念在与实现 trait 结合时，可能会导致复杂的重载解析问题。*
- **在 Mojo 中寻找量子库**：一位成员询问是否有正在开发的 Mojo **量子计算库**，提到出于学习目的需要一个类似于 **Qiskit** 的实现。
   - *作为回应，建议在功能演进期间利用 **MAX**，并提供了一个解释 MLIR 的视频链接以供进一步了解。*
- **MAX 在量子编程中的角色**：讨论强调了 **MAX** 如何旨在与 Mojo 配合工作以优化量子编程，在计算过程中为硬件提供动态适配。
   - *随着 MAX 的发展，它可能会为量子和经典计算工作负载提供必要的支持。*
- **发现 Quojo 库**：一位用户指出 **Quojo** 库是一个用 Mojo 编写的量子计算资源，并链接到了其 [GitHub 页面](https://github.com/Deftioon/Quojo)。
   - *这一提法引发了热烈讨论，大家对为该领域做出贡献的年轻开发者表示赞赏。*



**提到的链接**：<a href="https://github.com/Deftioon/Quojo">GitHub - Deftioon/Quojo: 用 Mojo 编写的量子计算机器</a>：用 Mojo 编写的量子计算机器。欢迎通过创建账号为 Deftioon/Quojo 的开发做出贡献。

  

---

### **LLM Agents (Berkeley MOOC) ▷ #[hackathon-announcements](https://discord.com/channels/1280234300012494859/1280236929379602493/1326806666129313843)** (1 messages): 

> `Hackathon 结果时间线，评委反馈` 


- **Hackathon 结果发布更新**：Hackathon 结果的时间线已在 [Hackathon 网站](https://rdi.berkeley.edu/llm-agents-hackathon/)上更新，大部分最终结果已统计完成。
   - 最终结果将在 **1 月下旬** 的某个时间发布，目前正在等待几位评委的反馈。
- **评委对提交作品印象深刻**：评委对收到的提交作品表示**印象深刻**，这表明参赛者表现强劲。
   - 团队感谢大家在等待最终结果期间的耐心。


  

---


### **LLM Agents (Berkeley MOOC) ▷ #[mooc-questions](https://discord.com/channels/1280234300012494859/1280370030609170494/1326777282391703635)** (6 messages): 

> `Google Form 编辑，表单的电子邮件绕过方案，Twitter 账号注销，证书资格` 


- **Google Form 编辑困扰**：一名成员报告称无法编辑之前提交的 Google Form，并请求私信协助。
   - 针对无法编辑的表单，另一名成员表示：*你可以重新提交表单来覆盖之前提交的内容。*
- **备选电子邮件访问建议**：一名成员建议使用不同的电子邮件访问已关闭的表单，并强调需要在 Email 字段中输入正确的电子邮件。
   - 提出该绕过方案是为了解决用户面临的访问问题。
- **关于 Twitter 账号状态的担忧**：同一名成员担心其注销的 Twitter 账号是否会影响其获得证书的资格。
   - 另一名成员安慰道，*你不会因为 Twitter 账号状态而被取消资格*。


  

---


### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1326953902591049849)** (7 messages): 

> `OpenInterpreter 1.0, 模型性能, 自定义指令, Python 代码执行` 


- **OpenInterpreter 1.0 功能限制**：讨论显示 OI 1.0 似乎无法直接运行 Python 代码，正如提示用户应以特定格式编写代码以便执行的说明所暗示的那样。
   - *一名成员对 `--tools interpreter` 命令在运行代码时未按预期工作表示困惑*。
- **GPT-4o-mini 使用体验**：用户分享了他们使用该 AI 的个人经验，注意到在命令执行和文件处理方面的改进，特别是它现在打印文件的头部（head）而不是整个文件。
   - 他们强调正在不断努力优化当前设置下的模型性能。
- **索取技术细节**：一名成员询问了模型规格，包括参数和其他可能增强功能的各种相关更改。
   - 这种对信息的需求反映了用户希望明确底层框架和性能指标。
- **闲聊互动**：用户进行了随意的交流，营造了支持性的氛围，一名成员热情地向另一名成员打招呼，表现出彼此的熟悉。
   - 这些互动表明技术讨论周围有着友好的社区氛围。


  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1326744255678644325)** (5 messages): 

> `TruLie dataset, Image-to-3D advancements, Gaussian splats, Chirpy3D, World Models` 


- **关于 TruLie 数据集的咨询**：一名成员询问了 **TruLie dataset**，向频道内的其他人寻求相关信息。
   - 未提供关于其特性或应用的具体细节。
- **Image-to-3D 技术的最新进展**：一位用户询问了 **image-to-3D** 领域的更新，特别是可供个人使用的开源解决方案。
   - 他们对 **structure-from-motion** 之外的技术以及 **Gaussian splats** 兴起以来的新方法表现出兴趣。
- **Chirpy3D 引领新浪潮**：一名成员分享了 **Chirpy3D**，这是一个专注于连续 3D 鸟类生成的显著项目，强调了其创意能力。
   - 该论文的研究团队来自多家知名机构，展示了该领域专业知识的协作。
- **令人兴奋的发展：World Models**：另一位用户强调了 **World Models**，它集成了物理感知网络，以实现更真实的视频生成。
   - 虽然与 image-to-3D 没有直接关系，但这一创新与视觉媒体领域的类似技术进步是一致的。
- **Gaussian splat 库的资源共享**：成员们讨论了对 **Gaussian splat 库** 的推荐以及任何有用的 **NeRF** 库，以增强他们的项目。
   - 为希望进一步探索的用户分享了 Hugging Face 的 **3D Arena** 等资源链接。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/dylanebert/3d-arena">3D Arena - a Hugging Face Space by dylanebert</a>：未找到描述</li><li><a href="https://kamwoh.github.io/chirpy3d/">Chirpy3D</a>：未找到描述
</li>
</ul>

</div>
  

---


### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/)** (1 messages): 

rom1504: 是否有任何用于构建 Agent 的优质开源工具注册表？
  

---


### **DSPy ▷ #[general](https://discord.com/channels/1161519468141355160/1161519469319946286/1326831683705507903)** (4 messages): 

> `Improving Chain of Thought (COT), Building Your Own Evaluation, DSPy and Knowledge Banks, Cultural Anthropology and Technology` 


- **探索改进聊天机器人 COT 的方法**：一名新手询问了除了为旨在像真人一样交流的聊天机器人设置 signature 之外，还有哪些增强 **Chain of Thought (COT)** 的方法。
   - 虽然没有直接回应，但该问题突显了聊天机器人开发中持续存在的挑战。
- **构建自定义评估简介**：一名成员分享了一篇关于**构建你自己的评估 (Evaluation)** 的见解深刻的文章，强调了其重要性以及 **DSPy** 如何协助这一过程。
   - 文章标题为《构建自定义评估简介、其重要性以及 DSPy 如何提供帮助》，链接可以点击 [这里](https://www.dbreunig.com/2025/01/08/evaluating-llms-as-knowledge-banks.html) 查看。
- **Drew Breunig 的多元背景**：Drew Breunig 简要介绍了自己，提到了他在**文化人类学、计算机科学和媒体**方面的经验。
   - 他的背景包括在 **PlaceIQ** 和 **Precisely** 的工作经历，专注于数据完整性以及与 **Overture Maps Foundation** 的合作。
- **对评估内容的兴趣**：另一名成员对分享的评估内容表示热烈欢迎，针对该评估文章评论道：“太棒了！我现在就去看看”。
   - 这反映了社区对评估重要性的关注日益增长。



**提到的链接**：<a href="https://www.dbreunig.com/">Home</a>：关于技术、文化、媒体、数据及其交互方式的写作。

  

---

### **AI21 Labs (Jamba) ▷ #[general-chat](https://discord.com/channels/874538902696914944/874538902696914947/1326675915857461260)** (3 messages): 

> `使用 Jamba 的 Python 应用，AI 代码生成，对 PHP 编程的依赖，Jamba 连接体验` 


- **Python 应用增强播客回忆**：一位用户使用 Jamba 的 Conversational RAG 开发了一个**基础 Python 应用**，通过查询上传的转录文本，帮助他们回忆过去播客节目中的讨论内容。
   - 他们提到该项目仍处于**开发中 (work in progress)**，但他们非常享受这一实验过程。
- **AI 的编程能力令人印象深刻但也令人困惑**：另一位用户分享了他们发现 AI 具有**生成代码**能力的兴奋之情，同时也强调了 AI 在编程辅助中偶尔会出现的一些愚蠢错误。
   - 他们已利用该技术进行 **HTML, Javascript 和 PHP** 的故障排除，这表明 AI 的潜力才刚刚开始显现。
- **PHP 对于 Web 编程依然至关重要**：尽管 AI 工具激增，一位用户表示他们依然依赖 **PHP** 进行 Web 和 IRC bot 编程，称其为一种经得起考验的方法。
   - 他们提到连接到 Jamba 是一次冒险，但他们对其功能与 **deepSeek 和 OpenAI APIs** 相似感到满意，这简化了编程任务。


  

---


### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/)** (1 messages): 

jovial_lynx_74856: 这里有人尝试过微调 ModernBERT 吗？
  

---


### **Torchtune ▷ #[jobs](https://discord.com/channels/1216353675241590815/1326789182932123708/1326789612580110369)** (1 messages): 

> `Nectar Social 招聘，AI 初创公司职位，推荐奖金` 


- **Nectar Social 提供高额推荐奖金**：**Nectar Social** 是一家处于早期阶段的 AI 初创公司，正在寻找候选人，提供的职位推荐奖金高达 **$10,000**。
   - 职位包括分布在不同地点的资深/高级产品经理、LLM/AI 工程师、基础架构工程师、客户成功经理以及创始客户执行官。
- **拥有大客户且发展迅速**：Nectar Social 专注于**社交电商 (social commerce)**，拥有大客户，且在保持公众领域半隐身状态的同时快速增长。
   - 消息鼓励有初创公司经验的感兴趣候选人**私信 (DM)** 以获取更多详情。


  

---


---


---


---


---


{% else %}


> 为了邮件展示，完整的频道分类详情已被截断。
> 
> 如果您想查看完整详情，请访问此邮件的网页版：[{{ email.subject }}]({{ email_url }})!
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！预谢支持！

{% endif %}