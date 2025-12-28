---
companies:
- coqui
- mozilla
- hugging-face
- google
date: '2024-01-04T06:56:46.257833Z'
description: '来自 Mozilla 机器学习小组的知名开源文本转语音项目 **Coqui** 已正式关闭。**HuggingFace** Discord
  频道中的讨论对 **sdxl** 声称的 `3 倍提速` 持怀疑态度，认为这种提升更多是由于 `torch.compile`、移除 `fp16` 和 `attention`
  等技术手段，而非 **diffusers 0.25** 的新特性。


  用户确认 **HuggingFace 用户令牌 (token)** 可以在多台机器上通用，但出于安全考虑，建议使用不同的令牌。**Learning Loss Minimization
  (LLM) 排行榜** 此前曾短暂出现故障，但目前已确认恢复正常。


  此外，有人分享了一个 Kaggle 笔记本，演示了如何使用 PyTorch 从零开始构建 Transformer 架构。同时，一个包含 1.5 万张鞋子、凉鞋和靴子图像的新数据集也已发布，用于多类别分类任务。关于
  Common Crawl 网页抓取流程工作原理的相关解释也一并被分享。'
id: 5aa32ff4-41c3-4ddf-8f02-7d76d46ca933
models:
- sdxl
- diffusers-0.25
original_slug: ainews-132024-rip-coqui
people: []
title: 2024年1月3日：愿 Coqui 安息
topics:
- text-to-speech
- performance-optimization
- token-management
- transformer-architecture
- image-datasets
- web-crawling
- pytorch
- leaderboards
---

> Meta：自昨天以来进行了更多调整。我们调低了重复的 OpenAI 错误报告的频率，并优化了提示词以实现更好的总结。

Coqui 是从 Mozilla ML 团队中幸存下来的领先开源文本转语音（text to speech）方案之一，[于今日关闭](https://twitter.com/_josh_meyer_/status/1742522906041635166)。其公告推文优美且感人。

---

**目录**

[TOC] 


## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- **“快如闪电”的 sdxl 质疑其自身速度**：正如用户 `@aifartist` 所指出的，**sdxl** 宣称的 `3X faster`（快 3 倍）性能依赖于特定技术，如使用 `torch.compile` 以及移除 `fp16` 和 `attention`，这让人怀疑 **diffusers 0.25** 的特性在这一性能提升中究竟扮演了什么角色。
- **“分享即关爱”也延伸到了 HuggingFace 用户令牌**：根据 `@osanseviero` 的说法，*HuggingFace user token* 确实可以在多台运行中的机器上使用，但为了操作安全，建议使用不同的令牌。
- **学习损失最小化 (LLM) 排行榜在玩捉迷藏**：`@lee0099` 最初询问 LLM 排行榜无法运行的问题已无意义，因为随后发现排行榜运行正常。
- **从零开始创建 Transformer**：`@torres8552` 分享了一个 [Kaggle notebook](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook)，深入探讨了如何使用 PyTorch 从头构建用于语言翻译任务的 Transformer 架构。
- **鞋子、凉鞋和靴子在图像数据集 T 台上亮相**：`@andysingal` 介绍了一个包含 1.5 万张鞋子、凉鞋和靴子图片的 [数据集](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)，旨在推动其在深度神经网络多分类任务中的应用。
- **Common Crawl 的网络爬虫奥秘揭晓**：`@cakiki` 解释了 Common Crawl 的工作原理，包括强大的计算机、URL 列表以及用于网页抓取和索引的“蜘蛛”软件，满足了 `@exponentialxp` 的好奇心。此外，还通过 [Common Crawl 代码库](https://github.com/commoncrawl) 的 GitHub 链接邀请大家进一步探索。

**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (85 条消息🔥🔥): 
        
- **提速后的 sdxl 受到质疑**：用户 `@aifartist` 对 **sdxl** 的一些*性能宣称*（如 `3X faster`）表示怀疑。他们注意到这些宣称似乎严重依赖于并非 **diffusers 0.25** 特有的方法，例如使用 `torch.compile` 以及移除 `fp16` 和 `attention`。他们请求澄清 **diffusers 0.25** 的哪些特定功能真正提升了性能。
- **在多台机器上使用 HuggingFace 用户令牌**：`@dizzyme` 询问一个 *HuggingFace user token* 是否可以用于两台或更多运行中的机器。`@osanseviero` 确认可以，但建议使用不同的令牌通常会更安全。
- **Arch Linux 上的 Python 命令问题**：用户 `@gez_gin` 在 Arch Linux 上遇到终端将 `from` 报告为未知命令的问题。`@cakiki` 指出 `from` 是 Python 关键字，并建议 `@gez_gin` 先运行 Python 以进入 Python REPL。
- **学习损失最小化 (LLM) 排行榜故障**：`@lee0099` 询问 LLM 排行榜无法运行的问题。随后，他们更新称问题似乎已解决。
- **关于 MoE Frankenmodels 的困惑**：`@kquant` 就其提交到 Open LLM 排行榜的条目寻求帮助。他们提交了两个条目——其中一个被错误地标记为 adapter——并请求管理员帮助删除错误条目，仅保留正确的“原始”条目。他们已经好几天没睡觉了，并为错误带来的不便表示歉意。

**提到的链接**：

- [Diffusers Gallery - 由 huggingface-projects 创建的 Hugging Face Space](https://huggingface.co/spaces/huggingface-projects/diffusers-gallery)
- [solarc-moe-10.7bx4.Q6_K.gguf · TheBloke/SOLARC-MOE-10.7Bx4-GGUF at main](https://huggingface.co/TheBloke/SOLARC-MOE-10.7Bx4-GGUF/blob/main/solarc-moe-10.7bx4.Q6_K.gguf)
- [Kquant03/CognitiveFusion-4x7B-bf16-MoE · Hugging Face](https://huggingface.co/Kquant03/CognitiveFusion-4x7B-bf16-MoE)
- [Open LLM Leaderboard - 由 HuggingFaceH4 创建的 Hugging Face Space](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [Meet](https://meet.google.com/dmn-wvxn-wpr)：Google 提供的实时会议。使用您的浏览器，...

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (8 messages🔥): 
        
- **@neuralink 在端到端 FP8 训练方面取得进展**：表示他们已经实现了 19% 的端到端 FP8 训练，这标志着他们在 3D parallelism（3D 并行）方面的工作取得了值得关注的进展。
- **@duplaja 在优化时发现 SpeechT5 的细微差别**：分享了他们在 SpeechT5 工作上的更新，重点是创建自定义 handler 以及解决数字和长字符串分页的问题。他们发现，在较低配置的 AWS GPU T4 上使用多个实例更具成本效益，并在此分享了他们可运行的 [handler.py](https://huggingface.co/Dupaja/speecht5_tts/blob/main/handler.py)。
- **@farlin9000 通过 Luis Serrano 重新学习 ML 基础知识**：分享了 Luis Serrano 关于神经网络深度学习的 [YouTube 视频](https://www.youtube.com/watch?v=BR9h47Jtqyw)，用于复习 ML 基础。Farlin9000 最初对激活函数和概率感到困惑，但随后理解了真值分类（truth classification）的原理。


**提到的链接**：

[A friendly introduction to Deep Learning and Neural Networks](https://www.youtube.com/watch?v=BR9h47Jtqyw)：神经网络和深度学习的友好介绍...


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (23 messages🔥): 
        
- **从零开始探索 Transformers**：用户 `@torres8552` 分享了一个 [Kaggle notebook](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook)，展示了如何使用 PyTorch 从零开始构建用于语言翻译任务的 Transformer 架构，并在 OpusBook 数据集上进行了训练。
- **鞋子 vs 凉鞋 vs 靴子图像数据集**：`@andysingal` 介绍了一个新的图像 [数据集](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)，包含 15,000 张鞋子、凉鞋和靴子的图像。非常适合使用 CNNs 等深度神经网络进行多分类任务。
- **在鞋子/凉鞋/靴子数据集上使用 resnet-50 的演示**：`@andysingal` 展示了一个使用 resnet-50 处理该图像数据集的 [notebook](https://github.com/andysingal/PyTorch-ML/blob/main/notebooks/resnet-50.ipynb)。
- **Augmentoolkit 介绍**：`@heralax` 开发了 Augmentoolkit，这是一个由 LLM 驱动的全本地 [数据集生成工具](https://github.com/e-p-armstrong/augmentoolkit)。它可以将纯文本转换为多轮对话，用于微调 instruct-tuned 模型。
- **在不同数据集上使用 Augmentoolkit**：`@andysingal` 表示有兴趣将 Augmentoolkit 应用于基于指令的数据集，例如 [Kaggle](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset) 上的数据集。`@heralax` 解释说，可以通过修改 notebook 中的几个单元格来实现，但代码会根据数据集结构而有所不同。


**提到的链接**：

- [Transformer From Scratch With PyTorch🔥](https://www.kaggle.com/code/lusfernandotorres/transformer-from-scratch-with-pytorch/notebook)：在 Kaggle 上探索并运行机器学习代码...
- [Question-Answer Dataset](https://www.kaggle.com/datasets/rtatman/questionanswer-dataset)：你能用 NLP 回答这些问题吗？
- [llama_index/examples/paul_graham_essay/data/paul_graham_essay.txt at main · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/main/examples/paul_graham_essay/data/paul_graham_essay.txt)：LlamaIndex（原 GPT Index）是一个数据框架...
- [Andyrasika/ShoeSandalBootimages · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/ShoeSandalBootimages)
- [PyTorch-ML/notebooks/resnet-50.ipynb at main · andysingal/PyTorch-ML](https://github.com/andysingal/PyTorch-ML/blob/main/notebooks/resnet-50.ipynb)：为 andysingal/PyTorch-ML 的开发做出贡献...


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (6 messages): 
        
- **实时参与 vs 异步讨论**：`@swyxio` 询问了讨论的形式，表示实时活动需要提前通知。`@lunarflu` 澄清说，由于社区的全球性，讨论**通常是异步且仅限文本的**。
- **博客文章讨论建议**：`@lunarflu` 建议在**每篇博客文章下进行讨论**，类似于论文的讨论形式，但也承认目前该功能尚不可用。
- **每周论文讨论活动**：针对讨论形式的查询，`@lunarflu` 提议创建一个**每周论文讨论活动**，包括开始时间和范围。
- **个人演讲号召**：`@lunarflu` 鼓励成员准备讨论用的演示文稿，并表示一旦确定日期，将创建全服务器范围的活动。
- **确认演讲日程**：响应 `@lunarflu` 的号召，`@dhruvdh` 承诺在**周五**前准备好一份演示文稿。

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (5 messages): 
        
- **关于在数据集中打开图像的困惑**：`@xcykim_56659` 询问了如何打开数据集中的图像内容，以及如何从用于预训练 CVT 模型的 ImageFolder PIL 对象中获取图像数据。随后，`@xcykim_56659` 自行解决了该疑问并报告了成功。
- **目标检测排行榜中的 FPS 计算查询**：`@anasuna` 对 [Object Detection Leaderboard](https://huggingface.co/spaces/hf-vision/object_detection_leaderboard) 上的每秒帧数 (fps) 计算表示怀疑，指出这些数值似乎过低。
- **在连续值上训练 CV 模型**：`@tony_assi` 表示有兴趣寻找相关资源，以利用与连续数值（而非离散标签）配对的图像来训练计算机视觉 (CV) 模型。

**提及的链接**：

[Open Object Detection Leaderboard - a Hugging Face Space by hf-vision](https://huggingface.co/spaces/hf-vision/object_detection_leaderboard)


### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (4 messages): 
        
- **Common Crawl 的网页索引说明**：`@exponentialxp` 询问了 Common Crawl 是如何收集网页数据的，`@cakiki` 解释说该过程涉及**强大的计算机、URL 列表以及被称为“蜘蛛 (spider)”的软件**来对这些网站进行抓取和索引，其功能类似于 Google 和 Bing 等搜索引擎。
- **邀请探索 Common Crawl 的代码库**：`@cakiki` 提供了 GitHub 上 [Common Crawl 代码库](https://github.com/commoncrawl)的链接，供感兴趣的 `@exponentialxp` 探索。

**提及的链接**：

[Common Crawl Foundation](https://github.com/commoncrawl): Common Crawl 提供了一个网页存档...


        

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **与 Mistral-7B 的提示词探戈**：`@cognitivetech` 思考了使用 **Mistral-7b** 进行 system prompts 的[两种](https://discord.com/channels/1234/1212/12121)[方式](https://discord.com/channels/1234/1212/12121)，速度和质量的一致性是潜在的挑战。
- **解读 Ooba 的谜团**：`@cognitivetech` 分享了一个来自 [Ooba 的模板](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml)但发现其令人困惑。
- **将 AI 实验室搬回家**：`@quantumpioneer.` 询问了为本地 AI 实验室设置进行[实验](https://discord.com/channels/1234/1212/12121)所需的硬件先决条件。
- **继续训练还是重新训练**：`@maxdipper` 探讨了利用之前训练过的 uncensored 模型进行追加训练的方法，以此作为从头开始重新训练的成本效益替代方案。
- **使用 Mixtral/Mistral 进行数据挖掘**：`@unknownperson2156` 寻求关于使用 Mixtral 或 Mistral 等 LLM 提取预定义问题数据的用户体验反馈。
- **关于 Mistral 8x7B 的宏大梦想**：`@mysterious2078` 正在寻找关于 **Mistral 8x7B 模型**的文档或论文。
- **解放本地运行环境**：`@michaelwechner` 分享了在 Mac M1 本地以及使用 [Ollama](https://github.com/jmorganca/ollama) 和 [Scaleway](https://www.scaleway.com/en/mac-mini-m2-pro/) 云端成功运行 Mistral 7B 的经验。
- **应对虚拟环境限制**：`@Idellarus` 详细描述了在受限的虚拟桌面环境中运行模型的困难，`@duck` 确认了其实际可行性。
- **vLLM vs TGI，一个 Mixtral 的故事**：`@andersruge` 询问了 vLLM 和 TGI 对性能指标的影响，`@casper_ai` 进行了简洁的回答。
- **全民纳米聊天机器人**：`@daain` 简要介绍了在有限资源下部署实时聊天机器人的选项，包括 API 以及像 Phi-2 或 TinyLlama-1.1B-Chat-v1.0 这样的小型模型。
- **GPU 狩猎季节**：`@comcyber_12802` 询问了微调 Mistral 7B 的 GPU 规格，`@le_mess` 推荐了 RTX 3090，并给出了约 1 小时的训练时间估算。
- **Mistral，开源之谜**：`@darshansharma_` 澄清了 **Mistral** 确实是开源的，`@refik0727` 验证了这一事实。
- **AGI 即将到来？**：`@poltronsuperstar` 发起挑战，预测 AGI 将在数周到数月内出现，并指出“观察-构建-培养”系统标志着无代码 AI 时代的到来，但同时也阐明了最终模型将具有“绝对天才”的特质。
- **定义 AGI 的探索**：用户 `@.tanuj.` 邀请社区分享他们对 **通用人工智能 (AGI)** 的理解；这确实是一个值得承担的挑战。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (61 messages🔥🔥): 
        
- **探索 Mistral-7b 的 System Prompts**：`@cognitivetech` 寻求关于使用 Mistral-7b 的 System Prompts 的建议，并尝试了两种格式，取得了不同程度的成功 [`#1`](https://discord.com/channels/1234/1212/12121) 和 [`#2`](https://discord.com/channels/1234/1212/12121)。在修改 Prompts 时，速度和质量的一致性似乎是存在的问题。 
- **来自 Ooba 的 Prompt 实现模板**：`@cognitivetech` 分享了 [Ooba 用于实现 prompts 的模板](https://github.com/oobabooga/text-generation-webui/blob/main/instruction-templates/Mistral.yaml)，尽管觉得它令人困惑 [`#1`](https://discord.com/channels/@cognitivetech/1212/12121)。 
- **本地 AI 实验的硬件**：`@quantumpioneer.` 询问了用于运行本地 AI 实验的 PC 配置的硬件规格和功耗要求 [`#1`](https://discord.com/channels/1234/1212/12121)。 
- **Uncensored Model 后的额外训练**：`@maxdipper` 询问在 Uncensored Model 之上添加额外内容训练是否有更便宜的方法，并将其与从头开始训练 Uncensored Model 进行了比较 [`#1`](https://discord.com/channels/1234/1212/12121)。 
- **使用 Mixtral 或 Mistral 进行线索收集**：`@unknownperson2156` 询问了使用 Mixtral 或 Mistral 进行数据或信息收集的用户体验，特别是将预定义的问答数据作为与 LLM 的对话 [`#1`](https://discord.com/channels/1234/1212/12121)。

**提到的链接**：

- [mistralai (Mistral AI_)](https://huggingface.co/mistralai)
- [app.py · openskyml/mixtral-46.7b-chat at main](https://huggingface.co/spaces/openskyml/mixtral-46.7b-chat/blob/main/app.py)
- [‎Riff Runner: Heavy Metal](https://apps.apple.com/us/app/riff-runner-heavy-metal/id6468704254)：在 Riff Runner 中释放重金属的力量，...
- [Riff Runner Metal (Pre-Release - Google Play 上的应用)](https://play.google.com/store/apps/details?id=app.titangen.games.ga008b)


### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (2 messages): 
        
- **对 Edge Computing 的兴趣**：`@kagevazquez` 对 Edge Computing 表现出热情，表示：“不，但 Edge Computing 听起来很棒”。
- **关于 Mistral 8x7B 文档的查询**：`@mysterious2078` 寻求关于 **Mistral 8x7B model** 的任何可用文档或论文。


### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (34 messages🔥): 
        
- **在本地运行 LLM**：`@michaelwechner` 分享了他在 Mac M1 上使用 [Ollama](https://github.com/jmorganca/ollama) 本地运行 Mistral 7B，以及通过 [Scaleway](https://www.scaleway.com/en/mac-mini-m2-pro/) 使用 Apple Mac mini M2 Pro 在云端运行的经验。讨论还延伸到 Ollama 和其他类似工具是否是 llama.cpp 的封装（wrappers）。 
- **虚拟桌面上的部署限制**：`@kartik.07` 讨论了在无法安装新软件或第三方工具的虚拟桌面上本地运行模型的挑战。`@duck` 确认运行 Inference 需要某种类型的软件，在有此类限制的情况下可能无法实现。
- **为 Mixtral 比较 vLLM 和 TGI**：针对 `@andersruge` 关于 vLLM 和 TGI 性能基准测试的查询，`@casper_ai` 强调 vLLM 通常更快，因为它优先考虑优化，而 TGI 主要关注减少 Time to First Token。 
- **为实时 Chatbot 应用缩减规模**：`@daain` 建议了在资源有限的情况下部署实时 Chatbot 的选项，例如使用 API、选择较小的模型（如 Phi-2 或 TinyLlama-1.1B-Chat-v1.0），或利用 NVidia Jetson Nano。

**提到的链接**：

- [GitHub - jmorganca/ollama: 在本地启动并运行 Llama 2 和其他大型语言模型](https://github.com/jmorganca/ollama)：在本地启动并运行 Llama 2 和其他大型语言模型...
- [在 Scaleway 的 Mac M2 16GB 上使用 Ollama 运行 Mistral 7B](https://medium.com/@michael.wechner/run-mistral-7b-using-ollama-on-a-mac-m2-16gb-at-scaleway-d640a4bd2158)：我最近在我的...上使用 Ollama 安装了 Mistral 7B。

### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (5 messages): 
        
- **GPU 推荐请求**：`@comcyber_12802` 询问了针对约 5000 个问答对数据集进行 Mistral 7B **finetuning** 的最低 GPU 要求。`@le_mess` 建议使用 RTX 3090，并提到它可以在大约 1 小时内完成该数据集的训练，并表示愿意通过私信提供进一步帮助。
- **投入时间学习**：在得到 GPU 推荐后，`@comcyber_12802` 表示在继续操作前，打算投入更多时间来更好地理解 **RAG**, **QLoRA**, **Axolotl**, **Peft** 等 **Agent**，并对 `@le_mess` 的帮助表示感谢。
- **无关对话**：`@akshay_1` 对某个未指明的来源评论说，这相当于告诉别人“去 Google 搜索”，对此 `@duck` 在可能显得冒犯的情况下表达了歉意。


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (13 messages🔥): 
        
- **Poltronsuperstar 对无代码 AGI 平台的看法**：用户 `@poltronsuperstar` 建议建立一个由 **LLM** 驱动的**无代码平台**，该平台包含多种类型的 **Agent**；由一个通用型 **Agent** 统筹各种专业型 **Agent**。重点在于拥有智能的高层决策，而非仅仅关注实现细节。
- **Agent 间通信与上下文数据存储**：`@poltronsuperstar` 阐明 **Agent** 应该**直接通信并通过共享上下文进行交流**。建议将文件作为存储高度变化数据的理想工具，强调了在稍微改造过的 GitHub 仓库中文件系统、facets 和历史记录的效率。
- **AGI 即将来临？**：在一个大胆的预测中，`@poltronsuperstar` 预言 **AGI 的到来就在几周到几个月内**。引用 GPT-4 级别的 **LLM** 作为可能的上限，并承认这一时间表在某种程度上依赖于直觉。
- **AGI：简单但天才**：虽然预测 **AGI** 的解释会相当简单（类似于 **GAN**），但 `@poltronsuperstar` 声明，解释的简单性并不会削弱最终模型是“*绝对天才*”的事实。
- **定义 AGI**：用户 `@.tanuj.` 提出了一个重要问题：“*大家如何定义 AGI？*”，试图了解聊天社区中存在的各种定义。


### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (6 messages): 
        
- **关于 Mistral 开源状态的提问**：`@darshansharma_` 询问 **Mistral** 是否开源，`@refik0727` 确认它是开源的。
- **发起开放讨论**：`@lerela` 鼓励在频道上进行公开提问。
- **请求 MISTRAL_API_KEY**：`@carloszela` 提到他正在为 **Mistral AI** 在 **langchain4j** 中添加 Java 库，并寻求一个 **MISTRAL_API_KEY** 演示。 
- **Medium 性能咨询**：`@_definitely_not_sam_` 询问其他用户是否也遇到了 Medium 性能缓慢的问题，但未见回复。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **LAION 的儿童色情内容污染困境**：`@chad_in_the_house` 提到了一篇 [斯坦福论文](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s)，揭露了 LAION 数据集中存在的儿童色情内容，引发了关于责任归属和数据集净化的紧迫讨论。`@progamergov`、`@.undeleted` 和 `@peacekeeper8310` 进一步讨论了披露规范，并提出了这背后可能存在的反 FOSS AI 议程以及企业监管俘获的动机。
- **解码 LAION 难题**：在对 LAION 争议数据集日益增长的担忧中，`@thejonasbrothers` 和 `@chad_in_the_house` 讨论了可能的缓解方案、在彻底根除与降低到可接受程度之间的权衡，以及该问题对抓取和存储可能受污染数据的合法性认知的影响。
- **剖析 SISR 的噪声挑战**：`@vrus0188` 指出了一篇 [研究论文](https://arxiv.org/abs/2312.17526)，概述了基于深度学习的单图像超分辨率 (SISR) 固有的训练初期噪声如何使获得最佳结果变得复杂。
- **图像生成精细化的创新**：`@vrus0188` 分享了 HandRefiner 和 ElasticDiffusion，分别介绍了用于修复畸形数字手部渲染和无需训练的任意尺寸图像生成的策略。项目地址：[HandRefiner](https://github.com/wenquanlu/HandRefiner) 和 [ElasticDiffusion](https://github.com/MoayedHajiAli/ElasticDiffusion-official)。
- **边界建模与文档推理的进展**：`@thejonasbrothers` 重点介绍了一个利用边界注意力在图像边界建模方面表现出色的 [可微模型](https://arxiv.org/abs/2401.00935)，以及一种新的 [DocLLM 方法](https://arxiv.org/abs/2401.00908)，该方法通过结合边界框信息与空间布局结构来提升文档理解能力。
- **受好奇心启发的机器人技术**：`@vrus0188` 推荐了一个 [YouTube 视频](https://www.youtube.com/watch?v=Nnpm-rJfFjQ)，展示了如何开发能够体现好奇心元素的机器人。

**LAION 频道总结**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (110 条消息🔥🔥): 
        
- **LAION 因不良内容陷入困境**：`@chad_in_the_house` 讨论了最近的一篇 [斯坦福论文](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s)，该论文发现 LAION 数据集中存在儿童色情内容，迫使 LAION 将其下架。社区对此表示担忧，并讨论了使用 Common Crawl 等替代方案。
  
- **关于负责任披露及其影响的辩论**：用户 `@progamergov`、`@.undeleted` 和 `@peacekeeper8310` 评估了斯坦福研究人员的方法，有人指出，在不给 LAION 预先缓解机会的情况下公开问题可能被视为鲁莽，不符合安全领域的负责任披露规范。此外，他们指出了反 FOSS AI 议程和寻求监管俘获的企业利益的可能性。

- **重新思考策略——更多的尽职调查？**：`@thejonasbrothers` 和 `@chad_in_the_house` 辩论了该问题的潜在解决方案，承认非法图像的易变性以及 100% 无污染数据集的不可能性。他们主张采取折中方案——如果已经进行了移除 NSFW 内容的尽职调查，则可能使数据集合法化。

- **内容责任的复杂性**：用户 `@thejonasbrothers` 指出，责任最终必须由托管非法内容的人承担，而不是包含潜在“有害字符串”的 LAION。然而，持续的困境引发了关于抓取、保存以及可能分发潜在受污染数据的合法性问题。

- **清除麻烦数据的难题**：鉴于 LAION 数据库最近出现的问题，用户 `@chad_in_the_house` 和 `@thejonasbrothers` 探讨了移除所有问题内容的复杂性。他们承认彻底根除可能是不可能的，但将其减少到可接受的程度可能是次优选择。然而，曝光 LAION 数据集问题的论文可能会无意中为在互联网上定位非法内容提供路线图，使问题进一步复杂化。

**提到的链接**：

- [Electronic Tip Form | FBI](https://tips.fbi.gov/home)
- [nvidia/parakeet-rnnt-1.1b · Hugging Face](https://huggingface.co/nvidia/parakeet-rnnt-1.1b)
- [Another Hit Piece on Open-Source AI](https://www.youtube.com/watch?v=bXYLyDhcyWY&t=1421s)：斯坦福研究人员在 L... 中发现问题内容。

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (8 messages🔥): 
        
- **图像超分辨率优化中的噪声障碍**：
    - `@vrus0188` 介绍了一篇[研究论文](https://arxiv.org/abs/2312.17526)，强调了在基于深度学习的 **Single Image Super-Resolution (SISR)** 早期训练步骤中，固有噪声所带来的挑战。该研究强调需要进一步审视 SISR 过程的病态 (ill-posed) 性质。
    
- **HandRefiner 旨在改进图像生成**：
    - `@vrus0188` 分享了一个名为 [HandRefiner](https://github.com/wenquanlu/HandRefiner) 的 GitHub 仓库。该项目提出了一种方法——基于 Diffusion 的 Conditional Inpainting——用于修复生成图像中畸形的手部。
  
- **ElasticDiffusion 提供无需训练的图像生成**：
    - `@vrus0188` 介绍了 GitHub 仓库中的 [ElasticDiffusion](https://github.com/MoayedHajiAli/ElasticDiffusion-official)，提供了一种全新的 **PyTorch implementation**，用于无需训练的任意尺寸图像生成。
    
- **改进图像边界的可微模型架构**：
    - `@thejonasbrothers` 推荐了一项[研究](https://arxiv.org/abs/2401.00935)，展示了一个采用 boundary attention 的 **differentiable model**，该模型在提供卓越的抗噪能力、亚像素精度以及处理原生分辨率图像的适应性的同时，能够出色地对边界进行建模。
    
- **DocLLM：视觉文档推理的创新方法**：
    - `@thejonasbrothers` 讨论了一篇介绍 **DocLLM** 的[论文](https://arxiv.org/abs/2401.00908)。**DocLLM** 是传统 Large Language Models (LLM) 的轻量级扩展，它仅将 attention 委派给 bounding box 信息以整合空间布局结构，从而避开了昂贵的 image encoders。此外，它还定制了一个预训练目标，有助于填充文本段落。发布者还提供了论文中的直接[引用](https://discord.com/channels/782201995011817493/874797969064538123/912727896369774618)。

- **受好奇心启发的机器人开发**：
    - `@vrus0188` 标记了一个标题为 "This Curious Robot Should Be Impossible!" 的 [YouTube 视频](https://www.youtube.com/watch?v=Nnpm-rJfFjQ)。

**提到的链接**：

- [Boundary Attention: Learning to Find Faint Boundaries at Any Resolution](https://arxiv.org/abs/2401.00935)：我们提出了一个显式地……的 differentiable model。
- [Noise-free Optimization in Early Training Steps for Image Super-Resolution](https://arxiv.org/abs/2312.17526)：最近基于深度学习的 single image super-reso...
- [DocLLM: A layout-aware generative language model for multimodal document understanding](https://arxiv.org/abs/2401.00908)：企业文档如表格、发票、收据……
- [This Curious Robot Should Be Impossible!](https://www.youtube.com/watch?v=Nnpm-rJfFjQ)：❤️ 查看 Weights & Biases 并注册……
- [GitHub - wenquanlu/HandRefiner](https://github.com/wenquanlu/HandRefiner)：为 wenquanlu/HandRefiner 的开发做出贡献……
- [GitHub - MoayedHajiAli/ElasticDiffusion-official: The official Pytorch Implementation for ElasticDiffusion: Training-free Arbitrary Size Image Generation](https://github.com/MoayedHajiAli/ElasticDiffusion-official)：ElasticDiffusion 的官方 Pytorch Implementation……

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **微调困境**：`@l_teto_l` 询问使用 Manticore 数据集**微调 LLAMMA 2** 是否能获得更好的结果，引发了多位用户的积极讨论，并分享了见解和相关资源链接。
- **Mixtral 漏洞追踪**：`@bratao` 分享了一份 [bug report](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942)，指出了 **Mixtral 微调** 中的一些问题。尽管如此，他们观察到即使在应用了建议的修复方案后，Mixtral instruct 的表现依然更好。
- **归因探索**：`@yamashi` 发起了一场关于如何精准定位对输出影响最大的 token 的讨论，建议使用反向传播（backpropagation）和输入梯度分析（input gradient analysis）。多位用户推荐了 **ooba** 等工具。
- **基准测试抨击**：`@yamashi` 批评了 **medmcqa** 和 **pubmedqa** 等基准测试存在单词不完整和分布偏斜的问题，引发了关于更好评估方法的讨论。
- **Triton Kernels 悬赏**：`@caseus_` 宣布了一项 [$2400 的悬赏](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038)，旨在提高用于 FFT 的 Triton kernels 的速度和显存效率。
- **学习率的平衡艺术**：`@nafnlaus00` 讨论了最优学习率（learning rates）、评估损失（evaluative loss）和训练损失（training loss），强调了它们对模型性能的影响，并强调了保持平衡比例的重要性。
- **Dropout 之辩**：`@nafnlaus00` 分享了关于确定最有效 dropout 率的见解以及正在进行的元参数微调（metaparameter tuning）过程。
- **Axolotl 的超参数魔力**：`@giftedgummybee` 提到在 Axolotl 中使用自动超参数微调（autohyperparam tuning），引起了大家的兴趣。
- **合并多个 PR 时跳过工作流**：`@caseus_` 建议在连续合并多个 PR 时使用 `[skip ci]` 标签以减少工作流运行次数，并引用了 [GitHub 文档](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs) 中的相关概念。
- **解析 Grouped GEMM 与 Grouped Experts**：`@caseus_` 和 `@casper_ai` 深入探讨了 Grouped GEMM 与 grouped experts 之间的联系，并分享了一个对比的 [GitHub 链接](https://github.com/imoneoi/openchat/compare/master...moe)。
- **应对非英语微调**：`@muhammad_ichsan` 讨论了针对非英语语言（印尼语）微调 **Mistral** 的挑战，得到了 `@nanobitz` 等成员关于扩大分词器（tokenizer enlargement）和文本指令方面的建议。
- **在多 GPU 上进行大模型训练**：`@b_ryan0` 寻求在多块 GPU 上训练大模型（如 codellama 34b）的策略。`@noobmaster29` 建议使用 `zero3` 和微批次（micro-batching）的解决方案。
- **解决 Axolotl 的非 GPU 开发问题**：`@kcaverly` 询问了 Axolotl CLI 的可行非 GPU 开发环境配置，`@noobmaster29` 建议在 runpod 上租用经济实惠的设备。
- **提升非英语性能**：`@noobmaster29` 分享了一篇 [学术论文](https://arxiv.org/pdf/2401.01055.pdf)，旨在提高 Mistral 等模型在非英语环境下的表现。
- **期待 Shearing Mistral 代码**：`@dangfutures` 请求在搞定 shearing mistral 代码后进行分享。
- **量化 Token 效果的探索**：`@nosa_.` 建议测试增加 token 数量是否能通过使用 **SlimPajama** 等大规模数据集来提升 **Sheared-LLaMA** 的能力。
- **非版权内容使用的法律指南**：`@dctanner` 发起了一场关于使用无许可证限制内容的讨论，以避免任何法律后果，特别是在最近的版权案件之后。
- **对 Bluemoon 质量的质疑**：`@xzuyn` 警告不要单独使用 **bluemoon**，因为其内容质量较低，并主张在版权限制范围内使用分类书籍数据集。

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (42 messages🔥): 
        
- **微调困境**：`@l_teto_l` 询问使用用于 Manticore 的数据集**微调 LLAMMA 2** 是否会产生出色的结果。这引发了一场讨论，多位用户分享了他们的见解和相关链接。
- **Mixtral 微调 Bug**：`@bratao` 分享了[一份关于 Mixtral 微调的 Bug 报告](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942)，但补充说即使在应用了某些修复后，**Mixtral instruct** 的表现仍然更好。
- **Tokens 贡献分析**：`@yamashi` 发起了一场关于如何确定哪些 Token 对输出贡献最大的有趣对话，建议使用反向传播并查看输入中每个 Token 的梯度。其他用户如 `@nanobitz` 提到了 **ooba** 等可能提供此功能的工具。
- **对 Benchmark 的批评**：`@yamashi` 对 **medmcqa** 和 **pubmedqa** 等 Benchmark 明显的缺点表示沮丧，指出它们有时不提供完整的单词，且分布往往偏斜，需要更仔细的评估。
- **优化 Triton Kernels 的悬赏**：`@caseus_` 发布了关于[为 FFT 优化 Triton Kernels 的 2400 美元悬赏](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038)的公告，寻求在速度和内存效率方面的改进。


**提到的链接**：

- [CLadder: A Benchmark to Assess Causal Reasoning Capabilities of Language Models](https://arxiv.org/abs/2312.04350)：评估语言模型因果推理能力的 Benchmark...
- [Question · Issue #6 · pratyushasharma/laser](https://github.com/pratyushasharma/laser/issues/6#issuecomment-1874828714)：你好，感谢发布此代码。这段代码是否...
- [ Incorrect implementation of auxiliary loss  · Issue #28255 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28255#issuecomment-1874241942)：系统信息 transformers 版本：4.37.0.dev0 平台...
- [[BOUNTY] Optimized Triton Kernels for full fine tunes · Issue #1038 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/issues/1038)：🔖 功能描述 我们已经看到了营销...
- [HellaSwag or HellaBad? 36% of this popular LLM benchmark contains errors](https://www.surgehq.ai/blog/hellaswag-or-hellabad-36-of-this-popular-llm-benchmark-contains-errors)：我们分析了流行的 LLM Benchmark HellaSwag，并且...
- [Fix load balancing loss func for mixtral by liangxuZhang · Pull Request #28256 · huggingface/transformers](https://github.com/huggingface/transformers/pull/28256)：此 PR 做了什么？修复了 #28255 在提交前...


### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (10 messages🔥): 
        
- **平衡学习率和损失比率**：`@nafnlaus00` 讨论了学习率 (LR)、评估损失 (eval loss) 和训练损失 (train loss) 之间的关系，建议观察它们的比率，因为这会影响模型性能。他们指出：“*取决于你的 LR。观察 eval loss 和 train loss 之间的比率，即它对记忆训练数据的专注程度。*”他们还提到，评估损失和训练损失之间的理想偏差不应超过 5-10%。
- **确定理想的 Dropout 率**：`@nafnlaus00` 分享了关于最佳 Dropout 率的见解，指出：“*我一直使用 0.25 的 Dropout，但我认为更低可能更好。但我认为高于 0.07 可能是最好的。*”他们承认仍在进行元参数调优，以找到适合其情况的最佳 Dropout 和 LR。
- **Axolotl 中的自动超参数调优**：`@giftedgummybee` 评论了在 Axolotl 中使用自动超参数调优 (autohyperparam tuning)，引发了社区成员的好奇。
- **合并多个 PR 时跳过 Workflow 运行**：`@caseus_` 建议在连续合并多个 PR 时使用 `[skip ci]` 标签，以减少 Workflow 运行。他们分享了来自 GitHub 文档的相关链接（[Skipping workflow runs - GitHub Docs](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs)）。
- **Grouped Experts 和 MOE**：`@caseus_` 和 `@casper_ai` 讨论了 Grouped GEMM 和 Grouped Experts 之间的关系，后者表示：“*就我所见，Grouped GEMM = Grouped Experts*”。`@caseus_` 还强调了 GitHub 上的一个对比链接（[Comparing master...moe · imoneoi/openchat](https://github.com/imoneoi/openchat/compare/master...moe)）以进一步举例说明。

**提到的链接**：

- [Skipping workflow runs - GitHub Docs](https://docs.github.com/en/actions/managing-workflow-runs/skipping-workflow-runs)
- [Comparing master...moe · imoneoi/openchat](https://github.com/imoneoi/openchat/compare/master...moe)：OpenChat：推进开源语言模型...

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (50 条消息🔥): 
        
- **非英语微调困境**：用户 `@muhammad_ichsan` 表示在印尼语 Wikipedia 数据集上微调 **Mistral** 时遇到困难，指出训练损失（training loss）停滞不前。`@nanobitz` 建议增加 tokenizer 中的 token 数量，向模型喂入大量 token，然后再进行指令微调（instruction tune）。鉴于 `@muhammad_ichsan` 报告了英语查询出现灾难性遗忘（catastrophic forgetting）的情况，`@noobmaster29` 还建议在全量微调（FFT）期间混入英语数据。[Wikipedia 数据集链接](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id)
- **Mistral Vicuna1.1 格式化**：`@le_mess` 分享了他们为 **Vicuna1.1** 创建的聊天模板，`@nanobitz` 建议在将其设为单行时添加 `\n`。 
- **跨 GPU 训练大模型**：`@b_ryan0` 询问了在多个 GPU 上训练像 codellama 34b 这样的大模型的方案，`@noobmaster29` 提供了使用 `zero3` 和 micro-batching 的解决方案。
- **Axolotl 的非 GPU 开发**：`@kcaverly` 询问了关于 Axolotl CLI 的“贫显卡”（GPU-poor）开发环境配置，`@noobmaster29` 建议在 runpod 上租用设备以保证性价比。
- **提升非英语性能**：`@noobmaster29` 分享了一篇学术论文 (https://arxiv.org/pdf/2401.01055.pdf)，这可能对那些寻求提升 Mistral 等模型非英语表现的人有所帮助。 


**提到的链接**：

[wikimedia/wikipedia at main](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.id)


### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (7 条消息): 
        
- **索取 Shearing Mistral 代码**：`@dangfutures` 请求在研究清楚后分享 shearing mistral 的代码。
- **关于 Token 数量的假设**：`@nosa_.` 建议测试一个假设，即增加 token 投入可以进一步提升 **Sheared-LLaMA** 的能力，这将非常有趣。
- **关于数据充分性的辩论**：在测试上述假设的背景下，`@nosa_.` 和 `@xzuyn` 一致认为 **SlimPajama** 可能提供了足够大的数据集进行测试。
- **关于使用无版权内容讨论**：`@dctanner` 对使用无许可证限制的内容进行持续预训练（continued pre-training）表示关注，以避免潜在的法律问题，特别是考虑到纽约时报（NYTimes）案件的最新进展。
- **对 Bluemoon 数据集质量的担忧**：由于可能存在内容质量问题，`@xzuyn` 建议不要单独使用 **bluemoon**，并建议收集一个不会带来版权挑战的书籍数据集。


        

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **渴望西班牙语界面**：用户 `@juaniespeche` 表达了对 Perplexity **西班牙语 UI** 的需求，并指出该 AI 已经能够用西班牙语进行准确回复。
- **Perplexity 定价困惑**：`@archient` 请求澄清在使用多个模型时 **Perplexity 的 token 定价**。`@icelavaman` 和 `@ok.alex` 澄清说，Perplexity 在 **预付额度系统 (prepaid credits system)** 下运行，总成本是基于处理的 token 对每个模型产生的累计金额。
- **渴望直接与模型交流**：`@saltrockr` 询问了是否可以在不进行互联网搜索的情况下直接与模型交互。`@reflext` 建议为此目的使用 Perplexity 的 **写作模式 (writing mode)**。
- **试用期支付出现意外状况**：`@ava12138` 和 `@boredkarma` 讨论了在验证 Perplexity Pro 7 天试用支付时遇到的困难，观察到不同卡种接受情况的不一致性。
- **Phind 与 Perplexity 之间显著的 UI 相似性**：`@neuralspace` 和 `@reflext` 讨论了 Phind 和 Perplexity UI 之间明显的相似之处。`@reflext` 认为，考虑到中心搜索栏的设计惯例，这种相似性是不可避免的。
- **感谢 Perplexity AI 的帮助**：`@hei_veno` 对 Perplexity AI 在开发培训内容方面的显著帮助给予了正面反馈，尽管由于保密原因无法分享细节。`@aontoni` 和 `@whiterickruben` 也分别分享了 Perplexity AI 协助完成大学项目和备考的经验。
- **通过文章和视频展示 Perplexity AI 的概况**：`@nayka3473` 提供了一篇他们撰写的关于 Perplexity 和其他 AI 聊天平台的文章链接，以及一段 [YouTube 视频](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh)，标题为：“顶级 AI 聊天平台排名：Phind, ChatGPT, Claude, Gemini Pro, Poe 等！”。
- **思考 Perplexity App 的角色**：`@archient` 提出了一个有趣的问题，关于 Perplexity App 中的配置文件 (profile) 与 API 中的系统角色 (system role) 之间的关联。
- **呼吁加入 Solar 10.7b 模型**：`@arcinarci` 建议在 Perplexity 的模型范围中加入 “*solar 10.7b 模型*”。

**Perplexity AI 频道总结**

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (65 条消息🔥🔥): 
        
- **需要西班牙语用户界面**：用户 `@juaniespeche` 表达了对 Perplexity **西班牙语界面** 的渴望，并指出该 AI 已经能够有效地用西班牙语回复。
- **API 定价澄清**：`@archient` 询问了使用多个模型时 **Perplexity 的 token 定价**。`@icelavaman` 解释说，总成本将是基于处理的 token 对每个模型成本的总和。关于使用计费的进一步询问促使 `@icelavaman` 和 `@ok.alex` 澄清 Perplexity 通过 **预付额度系统 (prepaid credits system)** 运行。
- **与模型直接对话**：`@saltrockr` 寻求一种在不涉及互联网搜索的情况下直接查询模型的方法。`@reflext` 建议使用 Perplexity 的 **写作模式 (writing mode)**。
- **试用期支付问题**：`@ava12138` 和 `@boredkarma` 讨论了 Perplexity Pro 7 天试用支付验证方法的问题，注意到哪些卡被接受存在不一致性。
- **Phind 与 Perplexity 之间的 UI 相似性**：`@neuralspace` 和 `@reflext` 讨论了 Phind 和 Perplexity 用户界面之间的相似性。`@reflext` 表示，考虑到中心搜索栏的设计类型，这种相似性是不可避免的。

**提到的链接**：

- [Perplexity - AI Companion](https://chrome.google.com/webstore/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo)：浏览时随心提问
- [Perplexity - AI Search](https://chrome.google.com/webstore/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol)：升级你的默认搜索引擎
- [Getting Started with pplx-api](https://docs.perplexity.ai/docs/getting-started)：pplx-api 入门指南
- [Perplexity - AI Search](https://chromewebstore.google.com/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol?pli=1)：升级你的默认搜索引擎

### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (5 条消息): 
        
- **用户对 Perplexity AI 的反馈**：用户 `@hei_veno` 提到 Perplexity AI 在开发培训内容方面提供了很大帮助，但由于工作保密原因无法分享详细信息。
- **资源推荐**：`@aontoni` 分享了一个他们认为很有帮助的[链接](https://www.perplexity.ai/search/is-it-recommended-4cG8AoJaSnWId74QGXo7Cg?s=u)，但未说明更多细节。
- **Perplexity AI 协助处理 MS Access**：`@aontoni` 随后说明了 Perplexity AI 如何帮助他们理解大学项目中 MS Access 的窗体（form）与查询（query）之间的关系。
- **Perplexity AI 对考试很有帮助**：用户 `@whiterickruben` 提到 Perplexity AI 帮助他们协助朋友准备即将到来的考试。
- **关于包含 Perplexity 在内的 AI 聊天平台的文章**：`@nayka3473` 写了一篇关于 Perplexity 和其他 AI 聊天平台的文章，并通过此[链接](https://medium.com/towards-artificial-intelligence/aichatplatforms-7be703c1f21d?sk=98b1f2335efa58013585aa64c2ebc29a)分享。他们还分享了一个 [YouTube 视频](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh)，标题为：“Ranking top AI Chat Platforms: Phind, ChatGPT, Claude, Gemini Pro, Poe and more!”并征求反馈。

**提到的链接**：

- [The Rise of AI: comprehensive list of top AI Chat Platforms](https://medium.com/towards-artificial-intelligence/aichatplatforms-7be703c1f21d?sk=98b1f2335efa58013585aa64c2ebc29a)：2023 年顶级 AI 聊天平台
- [Ranking top AI Chat Platforms: Phind, ChatGPT, Claude, Gemini Pro, Poe and more!](https://youtu.be/kjagVUqNHZ8?si=EzNHygYBWONu1Kvh)：发现我们 2023 年排名顶级的 AI 聊天平台...


### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (2 条消息): 
        
- **关于 Perplexity App 的 Profile 与 API System Role 的问题**：`@archient` 问：“*Perplexity App 中的 Profile 是否与 API 中的 System Role 相同？*”。
- **对 Solar 10.7b 模型的请求**：`@arcinarci` 询问了是否可能提供“*solar 10.7b model*”。


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **ChatGPT 缺失 img2img 功能**：`_typedef` 询问了关于 img2img 模型的问题，`@solbus` 澄清说 **ChatGPT 目前不支持直接的 img2img 功能**。然而，DALL·E 开发者在一次 AMA 中暗示未来的“图像参考（image references）”可能会引入 img2img 功能。[AMA 链接](https://discord.com/channels/974519864045756446/1173674915153592320/1174040158111273070)
- **使用 Actions 进行 API 集成的便利性**：`@iamhere6321` 称赞了 **Actions 在连接外部 API 时的易用性和有效性**。相反，`@niko3757` 更倾向于更高的灵活性和创建新线程的能力。
- **对 GPT-4 效率下降的担忧**：看到 GPT-4 的效率下降，`@caesarrzk` 寻求改进建议，`@my5042` 建议使用 Custom GPT 和“you are chatgpt”指令以获得更好的输出。
- **ChatGPT 性能和注册问题**：`@wolf.lover` 表达了 **ChatGPT 延迟和错误**的问题，`@zeromaid` 在**注册过程中遇到了问题**。
- **GPT-4 事实准确性担忧**：`@wesego` 对 GPT-4 在根据附件文档生成文本时的真实准确性表示担忧，`@niko3757` 建议使用互连的 API 或 CI。
- **向 ChatGPT 教授不可变语法**：`@facebreaker.` 询问如何教 ChatGPT 一种**不可变的固定语法或结构**，以获得更具体且可复现的响应。
- **在 GPT 协助下进行文件审查**：`@jferrari_75079` 寻求一个项目的帮助，该项目由 GPT 审查/总结文件内容，并提供操作建议（删除、归档或保存）。
- **创建不含建议的最新投资文章**：`@komal0887` 寻求帮助来优化 Prompt，以生成仅包含最新投资信息的文章，特别是不包含任何建议或评估性句子。他们正在使用 **gpt-3.5-turbo-instruct 模型**来完成此任务。
- **模仿对话风格的聊天机器人**：`@emaanios` 询问了关于能够模仿提供的对话风格的聊天机器人，用于他们的语言生成机器人研究。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (13 messages🔥): 
        
- **ChatGPT 目前尚无直接的 img2img 功能**：`@_typedef` 询问 txt2img 的模型是否与 img2img 相同。`@solbus` 澄清说，目前 **ChatGPT 并不具备直接的 img2img 功能**。它会识别上传的图像 (img2txt)，然后可以用于在随后的 txt2img 步骤中生成类似的图像。不过，Solbus 引用了一次 AMA，其中 **DALL·E 开发者暗示了未来可能推出“图像参考 (image references)”**，这可能会引入某种形式的 img2img 功能。[AMA 链接](https://discord.com/channels/974519864045756446/1173674915153592320/1174040158111273070)已分享，但可能需要存档访问权限才能查看。 
- **关于 Image to Image 的通用查询**：`@_typedef` 随后澄清，他们之前关于 img2img 功能的问题是通用的，并非专门针对 OpenAI。
- **无上下文的 URL**：`@jaicraft` 分享了一个 [URL](https://g.co/bard/share/cfec5f03f662)，没有任何前置或后置上下文。
- **数字疲劳**：用户 `@mad_cat__` 表达了疲劳感，觉得在 Discord 频道中穿梭很困难。不过，他们也提到了对自己工作的兴奋。

**提到的链接**：

[‎Steve Jobs Unveils Siri Chat](https://g.co/bard/share/cfec5f03f662)：使用 Bard 创建。


### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (20 messages🔥): 
        
- **OpenAI Actions 的易用性**：`@iamhere6321` 称赞了使用 **Actions** 连接 **外部 API** 的**配置简便性**和有效性，称其为一种很有前景的方法。`@niko3757` 分享了另一种观点，更倾向于具有更多灵活性且能创建新线程 (threads) 的 Assistants。 
- **遇到注册问题**：用户 `@zeromaid` 报告了平台上的注册问题，收到消息称 **“目前无法注册，请稍后再试。”** 他们重申了该问题，表示无法注册。 
- **ChatGPT 性能问题**：`@wolf.lover` 报告了 **ChatGPT 的性能问题**，指出它变得卡顿并在 Firefox 中导致错误。他们对需要切换聊天感到担忧，尽管已经在当前的聊天上花费了大量时间。
- **使用 Assistants 的优势**：在与 `@iamhere6321` 的讨论中，`@niko3757` 列举了使用 Assistants 优于自定义 GPTs 的几个优点。其中包括无限的 Actions、将多个 Actions 打包成一个的能力、触发新线程以及增强模型中的知识嵌入等好处。尽管强调了这些优势，`@niko3757` 也指出这些功能是有成本的。
- **寻求提升 GPT-4 准确性的帮助**：`@wesego` 询问是否有人成功让 GPT-4 在编写文本时准确遵循附件文档中的事实信息。他们注意到 AI 生成的故事与事实准确性之间存在差异。`@niko3757` 建议放弃 CustomGPT，尝试互连的 APIs，可能还会涉及持续集成 (CI)。
- **施加固定语法和结构的挑战**：`@facebreaker.` 寻求关于如何教 **ChatGPT 使用不可变的固定语法/结构** 的指导。他们遇到了语法变化和质量随时间下降的问题，并希望使模型的响应具有可复现性并符合其特定需求。 
- **切换 User-Agent 后出现的问题**：`@vova5963` 开玩笑说在频繁切换 **User-Agent** 后被 Mouser 封锁了，并提到这让他们可以在不被封锁的情况下观看 YouTube。

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (12 messages🔥): 
        
- **优化文章生成的 Prompt**：用户 `@komal0887` 寻求帮助优化一个 Prompt，该 Prompt 根据从不同 URL 提取的文本生成文章。生成的文章应仅包含最新信息，且不应包含投资建议、行动号召（call-to-action）或评价性语句。用户使用的是 **gpt-3.5-turbo-instruct model**。

- **关于 GPT-4 变懒的问题**：`@caesarzzk` 对 GPT-4 随着时间的推移似乎变得越来越“懒”表示担忧，它会尽可能省略输出代码或分析，有时甚至在理解力上表现挣扎。`@my5042` 建议在 Custom GPT 中使用诸如 "you are chatgpt" 之类的指令以获得更好的结果。

- **编写准确的故事**：`@wesego` 询问关于如何编写准确故事的指导。

- **关于 System Prompts 的问题**：`@itsnp` 询问是否可以在频道中提出关于 **system prompts** 的疑问。

- **模仿对话风格的聊天机器人**：`@emaanios` 询问是否存在可以根据提供的聊天记录模仿对话风格的聊天机器人，用于他们在语言生成机器人方面的研究。

- **寻求使用 GPT 进行文件管理的帮助**：`@jferrari_75079` 寻求一个项目的协助，在该项目中 GPT 将检查每个文件、子文件夹和图像，并就删除、归档还是保存提供建议。任务还包括让 GPT 提供文件内容的简短摘要。用户报告称，他们早期的尝试导致 GPT 根据文件最后修改日期等表面因素做出决定。


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (12 messages🔥): 
        
- **优化投资文章 Prompt**：`@komal0887` 表示需要协助优化提供给 `gpt-3.5-turbo-instruct model` 的 Prompt，用于根据从不同财经新闻 URL 提取的文本生成文章。他们希望输出仅包含最新信息，且不含建议或评价性语句。
- **提高 GPT-4 的效率**：`@caesarzzk` 注意到 GPT-4 的效率越来越低，并征求改善这一困境的建议。`@my5042` 建议使用 Custom GPT 并添加指令 "you are chatgpt" 以获得更好的输出。
- **用于简洁性和彻底性的递归检查器**：针对一个未定义的问题，`@madame_architect` 提出了一种解决方案，涉及一种递归检查器技能，以确保在写作中实现全面性与简洁性之间的平衡。
- **模仿对话风格的聊天机器人**：`@emaanios` 询问专门设计用于根据提供的聊天记录模仿对话风格的聊天机器人，`@beanz_and_rice` 确认了它们的存在。
- **寻求 GPT 审查文件的帮助**：`@jferrari_75079` 寻求帮助，让 GPT 彻底检查文件，并根据内容决定是删除、归档还是保存。他们还希望 GPT 提供每个文件内容的简短摘要。据指出，GPT 此前是根据文件最后修改日期等表面因素做出决定的。


        

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord Summary

- **DPO 全关乎分布**：@gabriel_syme 关注 **Differential Privacy Offsetting (DPO)** 如何更多地与分布（distribution）而非样本（samples）相关。
- **Lion 优化器的表现**：@marthinwurer 阐明了 **lion optimizer** 的功能，强调由于其每一步的权重变化固定，因此不允许出现大的损失峰值（loss spikes）。
- **寻找图像打标工具**：@frazermc 正在寻找一个轻便的 **image captioner** 来处理 50 万张图像，并表示倾向于非 LM 增强的选项。他分享了一个 [Awesome-Multimodal-Large-Language-Models repository](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) 仓库作为参考。
- **陷入混合专家模型 (MoE)**：@michaelmelons 询问是否有人实验过具有不同参数大小专家的 **Mixture of Experts (MoE)**，包括简单和复杂架构的专家。
- **Transformer 学习算法与协作提议**：@stellaathena 提议围绕一项名为 [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028) 的研究以及 Transformer 的组合能力之谜展开协作。
- **Pythia-70m 表现不佳**：@micpie 报告称 **Pythia-70m** 模型在基准测试中表现严重不佳，准确率降至 **0.002**。富有洞察力的 `@hailey_schoelkopf` 提出，*fp16* 自动数据类型的浮点精度（floating point precision）可能是原因，调整为 `float32` 可能会纠正该问题。


**Eleuther Channel Summaries**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (18 messages🔥): 
        
- **Lion Optimizer 防止大幅 Loss 尖峰**：`@marthinwurer` 观察到使用 **Lion Optimizer** 的实际好处，特别是没有出现大幅的 Loss 尖峰，因为权重在每一步只改变固定量，而不是梯度的倍数。
- **LLM 翻转回答逻辑**：`@sk5544` 寻求社区关于论文或研究的建议，以解释为什么 **LLM** 在被问到“你确定吗？”时会翻转其回答。
- **寻找高效的 Image Captioner**：`@frazermc` 分享了需要一个 **Image Captioner** 来处理 50 万张图片的需求，理想情况下不是基于 LM 增强的。他们分享了一个关于 [Multimodal Large Language Models 的 GitHub 仓库](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation) 供参考。 
- **Huggingface Datasets 中高效的序列偏移**：`@.the_alt_man` 分享了基于 **Huggingface Datasets** 进行序列偏移 (*Shift Sequence*) 的代码，但指出 `torch -> list -> jax.Array` 的开销太重，并询问是否有更好的方法在 Huggingface 原生完成此预处理。 
- **在 Google Colab 中运行 lm-evaluation-harness**：`@lee0099` 询问是否可以在 Google Colab 中运行 **lm-evaluation-harness**，`@hailey_schoelkopf` 确认这是可行的，并分享了关于如何操作的 [GitHub 指南](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb)。
- **在 LSTM 中实现数据控制的遗忘门**：`@sentialx` 询问如何在 LSTM 中实现 **数据控制的遗忘门 (Data-Controlled Forget Gate)**，`@wonkothesensible` 建议参考 **RWKV** 以获取灵感。
- **对 Pythia LLM 分析的赞赏**：`@swyxio` 认可并强调了 **Pythia 团队** 的工作，分享了 [@rasbt 的 Twitter 线程](https://fxtwitter.com/rasbt/status/1734920232173539796)，该线程赞扬了 Pythia 对 LLM 的全面分析。


**提及的链接**：

- [lm-evaluation-harness/examples/lm-eval-overview.ipynb at main · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/examples/lm-eval-overview.ipynb)：一个用于自回归模型 Few-shot 评估的框架...
- [Sebastian Raschka (@rasbt) 的推文](https://fxtwitter.com/rasbt/status/1734920232173539796)：我正在回顾今年我最喜欢的论文，并且...
- [GitHub - BradyFU/Awesome-Multimodal-Large-Language-Models at Evaluation](https://github.com/BradyFU/Awesome-Multimodal-Large-Language-Models/tree/Evaluation)：:sparkles::sparkles: 最新的论文和数据集...


### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (17 messages🔥): 
        
- **DPO 分布关注点**：`@gabriel_syme` 指出与 DPO 的联系更多地集中在**分布 (Distribution)** 而非样本上。
- **关于 Theorem 5.4 的讨论**：`@salmon_lemon` 对 *Theorem 5.4* 表示困惑。`@sumo43` 提供了一些见解，建议通过成功优化生成器，其输出将变得与数据相似，并将 lambda 解释为学习率参数。
- **图像模型的概念擦除**：`@voxs` 询问是否有人做过 **图像模型的概念擦除 (Concept Erasure)**，随后表示他们找到了一些相关资源。
- **Mobile ALOHA 模仿学习系统**：`@ai_waifu` 发布了 [Mobile ALOHA](https://mobile-aloha.github.io/resources/mobile-aloha.pdf) 的链接，这是一个低成本的全身远程操作系统，专为模仿机器人中的移动操作任务而开发。`@thatspysaspy` 赞赏了演示并询问其鲁棒性，而 `@ai_waifu` 讨论了成本效益，并声称大规模生产可以显著降低成本。
- **具有可变参数大小的混合专家模型**：`@michaelmelons` 询问是否有人尝试过在大规模下使用具有不同参数大小专家的 **MoE (Mixture of Experts)**，包括简单和更复杂架构的专家。

**提及的链接**：

[Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation](https://mobile-aloha.github.io/)：由 Zipeng Fu*, Tony Z. Zhao* 和 Chelsea Finn 在 S...

### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (2 messages): 
        
- **Transformer 算法协作提案**：用户 `@stellaathena` 讨论了与 [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028) 第一作者进行协作的可能性。他们探讨了**组合性（compositionality）**以及**以 RASP(-L) 表示的任务信息论复杂度**等话题，并表示有兴趣了解为什么 Transformer 无法实现完美泛化。
- **对协作的积极回应**：用户 `@dashiell_s` 表示有兴趣加入拟定的协作。


### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (15 messages🔥): 
        
- **Pythia-70m 在测试中表现大幅下滑**：用户 `@micpie` 注意到 **Pythia-70m** 模型在基准测试中表现不佳，导致准确率仅为 **0.002**，而之前的测试结果为 **0.609** [查看消息](https://discord.com/channels/824728003853975583/967159777337462845/969868847195512896)。
- **浮点精度可能是问题所在**：`@hailey_schoelkopf` 建议该问题可能是由于模型在 HF 中使用 auto dtype 以 fp16 运行导致的。通过将 dtype 调整为 `float32`，测试返回了更合理的结果 [查看消息](https://discord.com/channels/824728003853975583/967159777337462845/969868919384489032)。
- **更多 Pythia 模型受到影响**：该问题似乎特定于 v1 版本的 **Pythia** 模型，且在较小模型中更为普遍。根据 `@hailey_schoelkopf` 的说法，启用 torch autocast 可能会有所帮助 [查看消息](https://discord.com/channels/824728003853975583/967159777337462845/969868974884458564)。
- **加载本地数据集困难**：`@micpie` 在加载 JSON 格式的本地数据集时遇到问题。`@hailey_schoelkopf` 建议使用 `dataset_path: json` 和 `dataset_kwargs: { data_dir: /path/to/benchmark_0-2 }` 作为临时解决方案，但指出他们将进行更改以恢复原始功能 [查看消息](https://discord.com/channels/824728003853975583/967159777337462845/969872582121836656)。 
- **等待恢复原始功能的更改**：尽管有了加载本地数据集的临时方案，`@micpie` 仍选择等待功能修复的实现，这样他们就不必调整大约 400 个配置文件 [查看消息](https://discord.com/channels/824728003853975583/967159777337462845/969874635933319178)。


        

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **使用 Ayenem 的项目进行 Token 修复 (Healing)**：`@ayenem` [发布了一个名为 TokenHealer 的项目](https://github.com/Ayenem/TokenHealer/tree/main)，该项目可以裁剪并重新生成 prompt，以与模型的 tokenizer 保持协调。这提高了模型的补全能力及其对尾随空格/标点符号的鲁棒性。关于 TokenHealer 解决的问题的更多背景信息可以在[这篇文章](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69)中找到。
- **MidJourney 的 API 障碍**：`@kevmodrome` 询问 **MidJourney** 是否可以通过 Discord 以外的 API 使用。`@jevonm` 澄清说目前它是 Discord 独占的。
- **寻找用于音频分析的 AI**：`@zf0` 对能够进行音频分析（而非仅仅是视频帧分析）的聊天模型感到好奇。`@swyxio` 建议探索 riffusion 风格的方法或 Meta 的 Seamless 模型。
- **Coqui 的关闭在 AI 社区引起反响**：`@swyxio` 传播了 [Coqui 关闭的消息](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww)。Coqui 曾是一家开源语音技术机构。
- **GPT-4 总结 AI/ML 论文**：`@intheclouddan` 重点介绍了一个 [emergentmind.com 上的工具](https://www.emergentmind.com/)，该工具利用 GPT-4 来总结 AI/ML 论文。
- **LLM Paper Club 将讨论 InsightPilot**：`@swyxio` 和 `@eugeneyan` 宣布将在[即将举行的 LLM Paper Club](https://lu.ma/llm-paper-club) 中讨论 **InsightPilot**。InsightPilot 是一个由 LLM 驱动的自动化数据探索系统。
- **Mixture of Experts (MoEs) 即将成为讨论焦点**：据 `@swyxio` 透露，下周的 LLM Paper Club 将讨论一篇关于 "Mixture of Experts" 的论文，这是开源 AI 社区的一个热门话题。博客文章链接在[这里](https://huggingface.co/blog/moe)。
- **记录 LLM Paper Club**：`@swyxio` 强调了在 Paper Club 期间做笔记的必要性，并征求关于 Discord 笔记机器人工具的建议。

**Latent Space 频道总结**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (17 messages🔥): 
        
- **Ayenem 发布 TokenHealer**: 用户 `@ayenem` [介绍了 TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main)，这是一个裁剪并重新生成 Prompt 以对齐模型 Tokenizer 的项目。这提高了模型的补全能力以及对尾随空格/标点的鲁棒性。同时分享了一篇[相关博客文章](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69)，为 TokenHealer 解决的问题提供了更多背景信息。
- **MidJourney 平台查询**: 用户 `@kevmodrome` 询问 **MidJourney** 是否可以通过 Discord 之外的任何 API 使用。`@jevonm` 回复称目前仅能通过 Discord 访问。
- **关于音频分析聊天模型的查询**: `@zf0` 询问是否有可以分析音频而不仅仅是视频帧的聊天模型。`@swyxio` 建议研究 "riffusion 风格的方法" 或 Meta 的 Seamless 模型。
- **Coqui 宣布关闭**: `@swyxio` 分享了开源语音技术组织 [Coqui 关闭的消息](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww)。 
- **总结 AI/ML 论文的新工具**: `@intheclouddan` 关注到了 [emergentmind.com 上的一个工具](https://www.emergentmind.com/)，该工具使用 GPT-4 来总结 AI/ML 论文。


**提及的链接**:

- [来自 Josh Meyer 🐸💬 (@_josh_meyer_) 的推文](https://fxtwitter.com/_josh_meyer_/status/1742522906041635166?s=46&t=90xQ8sGy63D2OtiaoGJuww): Coqui 即将关闭。这是一个令人遗憾的消息...
- [来自 Sam (@Sam_Awrabi) 的推文](https://fxtwitter.com/Sam_Awrabi/status/1742324900034150646?s=20): 1. AI 资金主要集中在模型层...
- [AI/ML 研究解析 | Emergent Mind](https://www.emergentmind.com/): 随时了解重要的全新 AI/ML arXiv 研究...
- [GitHub - Ayenem/TokenHealer](https://github.com/Ayenem/TokenHealer/tree/main): 通过创建...为 Ayenem/TokenHealer 的开发做出贡献。
- [Prompt 设计的艺术：Prompt 边界与 Token Healing](https://towardsdatascience.com/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38?gi=e8510357db69): 了解标准的贪婪 Tokenization 如何引入...


### ▷ #[ai-event-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/) (1 messages): 
        
- **与领军人物讨论 InsightPilot**: `<@187636841988620288>` 将在[此处](https://lu.ma/llm-paper-club)引导关于 **InsightPilot**（用于数据分析的 Copilot）的讨论。
- **LLM Paper Club**: 该活动是每周一次的 LLM 论文研讨，重点关注**核心思想**、其**相关性**以及阅读后的任何**开放性问题**。
- **目前暂无即将举行的场次**: 该系列目前没有即将举行的场次，但建议定期查看更新的日程表。
- **论文选择机制**: 研讨的论文会提前一周决定，详细信息将在 `#llm-paper-club` 频道分享。
- **申请 Discord 通知**: 鼓励用户申请在 `<@&1107197669547442196>` 中被提及，以便接收与见面会相关的 Discord 通知。


**提及的链接**:

[LLM Paper Club (现已移至 Discord) · Luma](https://lu.ma/llm-paper-club): 每周一次的 LLM 论文研讨，从...开始。

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (14 条消息🔥): 
        
- **InsightPilot：LLM 赋能的自动化数据探索系统**：`@swyxio` 分享了关于 InsightPilot 的论文详情，这是一个基于 LLM 的自动化数据探索系统，旨在简化数据探索流程。论文可以通过[此链接](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/)查看。
- **加入 InsightPilot 讨论**：`@eugeneyan` 邀请成员通过[此 Discord 链接](https://discord.gg/zuZp95ya)加入关于使用 LLM 分析数据的讨论。
- **下一篇预告：Mixture of Experts (MoEs)**：`@swyxio` 提供了下周论文的链接，主题是“Mixture of Experts”，这是开源 AI 社区的热门话题。博客文章链接在[这里](https://huggingface.co/blog/moe)。
- **未来论文考虑：Self-Play Fine-Tuning (SPIN)**：`@swizec` 建议考虑将关于 Self-Play Fine-Tuning (SPIN) 的论文纳入未来的讨论。提议的论文可以在[此链接](https://arxiv.org/abs/2401.01335)找到。
- **论文俱乐部笔记记录**：`@swyxio` 表示在论文俱乐部会议期间需要良好的笔记记录，并正在寻求 Discord 笔记机器人工具的建议。

**提到的链接**：

- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models](https://arxiv.org/abs/2401.01335)：通过利用人类标注数据的力量...
- [加入 /dev/invest + Latent Space Discord 服务器！](https://discord.gg/zuZp95ya)：查看 /dev/invest + Latent Space 社区...
- [Mixture of Experts Explained](https://huggingface.co/blog/moe)
- [InsightPilot: An LLM-Empowered Automated Data Exploration System - Microsoft Research](https://www.microsoft.com/en-us/research/publication/insightpilot-an-llm-empowered-automated-data-exploration-system/)：探索数据在数据分析中至关重要，因为它...

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **考虑使用 LLM 改写 Anki 卡片**：`@thebaghdaddy` 在 `#collaboration` 频道分享了使用 **LLM** 改写 Anki 卡片的兴趣，以获得更好的*信息泛化能力*。
- **探索用于流畅叙事的 Multi-agent 系统**：`@yikesawjeez` 提议建立一个 Multi-agent 系统，包括一个“编排器（orchestrator）”、一个“状态管理器（state manager）”和一个在异步文字角色扮演素材上训练的小模型，以**管理叙事创作**。
- **通过“目标（Objectives）”引导剧情**：`@yikesawjeez` 进一步建议在系统中加入由“DM”检查的“目标”部分，这有助于将*剧情引导*至预定方向。
- **旨在打破 AI 叙事循环**：`@yikesawjeez` 指出了 AI 内容生成的一个常见问题——*重复的叙事循环*。建议的解决方案：同时修改玩家和模型的文本以打破循环。
- **长上下文模型辅助叙事管理**：`@yikesawjeez` 认为管理叙事的长上下文模型可以从剧情展开的示例中受益，从而获得精确的 *few-shot 指导*。
- **Search + Search RAG API 开启 Beta 测试**：`@emrgnt_cmplxty` 在 `#rag` 频道宣布发布新的 Search + Search RAG API，邀请积极的贡献者进行 *Beta 测试*并提供用户应用反馈。该模型也是**开源**的。
- **社区对新 API 的兴趣**：`@yikesawjeez` 表现出查看此新 API 的浓厚兴趣，并索要了**链接**。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018/) (5 条消息): 
        
- **使用 LLM 改写内容**：`@thebaghdaddy` 表达了利用 **LLM** 改写 Anki 卡片的兴趣，目标是提高信息泛化能力。
- **用于叙事创作的 Multi-agent 系统**：`@yikesawjeez` 详细介绍了他们运行 Multi-agent 系统来管理叙事创作的想法。提议的系统包括一个“编排器”、“状态管理器”和一个在异步文字角色扮演素材上训练的小模型，共同协作将**叙事信息压缩成可管理的部分**。
- **目标驱动的叙事管理**：`@yikesawjeez` 还提到了增加一个由“DM”检查的“目标”部分的可能性，以引导剧情向特定方向发展。
- **避免叙事循环**：`@yikesawjeez` 强调了应对 AI 生成的叙事循环的挑战，即类似的响应会触发重复的文本。他们建议修改玩家的消息和模型的响应来打破循环。
- **用于叙事管理的长上下文模型**：`@yikesawjeez` 建议管理叙事的长上下文模型可以从剧情如何展开的示例中受益，从而实现有针对性的 few-shot 指导。

### ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/) (3 messages): 
        
- **Search + Search RAG API 测试版发布**: `@emrgnt_cmplxty` 宣布发布新的 Search + Search RAG API，并询问社区是否可以进行快速的 **Beta 测试**并提供反馈，特别是该 API 对他们的应用是否有用。
- **开源模型**: `@emrgnt_cmplxty` 提到这个新推出的 API 背后的模型是**开源的**。
- **请求新 API 链接**: 用户 `@yikesawjeez` 表示感兴趣并询问该新 API 的**链接**。

        

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord Summary

只有一个频道有活动，因此无需总结...

- **GPT-4 Turbo 与 GPT-4 的对比**: 用户 `@philipmay` 询问关于 **GPT-4 Turbo (gpt-4-1106-preview)** 与常规 **GPT-4** 性能对比的评价。
- **Turbo 在对话中表现出色**: `_jp1_` 指出，根据个人印象，**GPT-4 Turbo** 在“便捷提示词”或普通对话以及涉及长上下文的任务中甚至可能优于 **GPT-4**。
- **Turbo 在复杂任务中表现不佳**: 然而，`_jp1_` 还提到，**GPT-4 Turbo** 在面对*复杂指令*（例如按特定顺序执行的一系列自定义任务）时似乎表现不佳。
- **编程场景具有挑战性**: `@mister_poodle` 表示，在编程场景下，**GPT-4 Turbo** 即使在明确指示的情况下也经常难以实现完整代码；而在 **GPT-4** 中，除非处理极长的上下文，否则这种问题较少发生。
- **GPT-4 的整体性能**: `@mister_poodle` 观察到，自发布以来，**GPT-4 Turbo** 和 **GPT-4** 的性能似乎都有所下降。

        

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord Summary

- **热烈欢迎物理学专家**: 新成员 `@ddt1909`（又名 **Daniel**）分享了他在 **ML/计算机视觉** 方面的经验，以及他目前正在进行的利用 LLM 为企业提取信息的项目。他是受播客推荐加入该服务器的。
- **Phi-Tuning 表现不佳**: `@benxh` 描述了他在 Phi-Tuning 方面**大多是负面的体验**，并提醒社区注意该模型调整参数的困难。
- **Hugging Face 模型：表现平平**: `@benxh` 发现 **Hugging Face 上提供的微调模型表现乏善可陈**，表明可能存在未识别的问题，引发了关于预训练模型质量控制和预期的深入讨论。

**Alignment Lab AI 频道总结**

### ▷ #[join-in](https://discord.com/channels/1087862276448595968/1143791237669855302/) (1 messages): 
        
- **新成员介绍**: `@ddt1909` 介绍了自己，他叫 **Daniel**，拥有物理学背景，自 2017 年以来一直从事 **ML/计算机视觉** 工作。他目前正在构建一个基于 LLM 的企业信息提取产品。他决定加入该服务器是受 `@660097403046723594` 在播客中推荐的影响。


### ▷ #[phi-tuning](https://discord.com/channels/1087862276448595968/1151623997121908816/) (3 messages): 
        
- **Phi-Tuning 的负面体验**: 用户 `@benxh` 对 Phi-Tuning 表示不满，因为他们**大多是负面的体验**。
- **Hugging Face 上微调模型表现乏善可陈**: `@benxh` 还指出 **Hugging Face 上的微调模型表现平平**，而且似乎存在一个未识别的问题。


        

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord Summary

只有一个频道有活动，因此无需总结...

- **寻找分析型数据库资源**: 用户 `@pranay01` 表达了对学习*最先进的分析型数据库/大规模分析系统*的兴趣，并询问该关注谁，同时表达了对用户 `<@1016864328189759488>` 的欣赏。
- **专家的资源推荐**: 用户 `@andypavlo` 向 `@pranay01` 推荐了一个关于该主题的即将开课的课程，并提供了[课程页面链接](https://15721.courses.cs.cmu.edu/spring2024/)。
- **非 CMU 人员的可访问性**: `@pranay01` 随后询问是否有该课程的旧版本可供访问，以及非卡内基梅隆大学（CMU）的学生是否可以选修这些课程。

**提到的链接**:

[CMU 15-445 :: 高级数据库系统 (2024 春季)](https://15721.courses.cs.cmu.edu/spring2024/): 卡内基梅隆大学

        

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **NEJM Image Challenge 数据集现已开放**：`onuralp.` 在 [GitHub](https://github.com/cx0/nejm-image-challenge) 上分享了 NEJM Image Challenge 数据集，并指出对于已有模型的用户无需进行数据清洗。他暗示计划在本周分享 **gpt4v 结果**，并欢迎任何关于模型微调或其他修改的建议。

**Skunkworks AI 频道摘要**

### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 条消息): 
        
pradeep1148: https://www.youtube.com/watch?v=O6RPmtuGKMM


### ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/) (1 条消息): 
        
- **NEJM Image Challenge 数据集已分享**：`onuralp.` 在 [GitHub](https://github.com/cx0/nejm-image-challenge) 上发布了 NEJM Image Challenge 数据集，并提到对于已经部署模型的用户，无需进行数据清洗。他还提到计划在本周上传 **gpt4v 结果**，并欢迎任何关于模型更改或其他修改的建议。

**提到的链接**：

[GitHub - cx0/nejm-image-challenge: NEJM Image Challenge dataset and experiments](https://github.com/cx0/nejm-image-challenge): NEJM Image Challenge 数据集与实验。继续...


        

---
Datasette/LLM (@SimonW) Discord 没有新消息。如果该公会长期保持沉默，请告知我们，我们将将其移除。