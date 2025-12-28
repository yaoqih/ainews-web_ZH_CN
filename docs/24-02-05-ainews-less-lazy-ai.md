---
companies:
- openai
- hugging-face
- nous-research
- h2oai
- apple
date: '2024-02-06T00:50:28.809972Z'
description: '2024年初的 AI Discord 摘要涵盖了各种社区讨论和进展。亮点包括对 **20** 个服务器、**308** 个频道和 **10449**
  条消息的分析，估计节省了 **780 分钟** 的阅读时间。


  关键话题包括集成 PubMed API 的 **Polymind 插件谜题**、使用 **HamSter v0.2** 进行角色扮演、**Axolotl** 训练中的显存（VRAM）挑战、**FLAN-T5**
  的微调技巧以及创新的**模型合并**策略。


  **Nous Research AI** 社区讨论了 GPT-4 的文采（lyricism）问题、使用 `llama.cpp` 的量化技术、使用 **miqu-1-120b-GGUF**
  等模型进行的 **frankenmerging**（拼接式合并）、对 **Qwen2** 的期待，以及 `text-generation-webui` 和 **ExLlamaV2**
  等工具。


  **LM Studio** 社区报告了一个 Bug，即应用程序在关闭 UI 界面后仍继续运行，目前的解决方法是强制终止进程。这些讨论反映了 AI 模型训练、部署和交互中持续存在的挑战与创新。'
id: 01eabd5a-e8bb-41e3-981b-58cc9467e3b3
models:
- hamster-v0.2
- flan-t5
- miqu-1-120b-gguf
- qwen2
- axolotl
original_slug: ainews-less-lazy-ai
people:
- philschmid
title: '“Less Lazy AI” 可以翻译为：


  *   **更勤奋的人工智能**

  *   **没那么懒的人工智能**

  *   **拒绝偷懒的人工智能**'
topics:
- model-merging
- fine-tuning
- quantization
- vram-optimization
- plugin-development
- chatbot-memory
- model-training
- bug-reporting
- api-compatibility
---

 

尽管如此，“偷懒”（laziness）并不是一个定义明确的技术术语。令人沮丧的是，OpenAI 已经发现并修复了一个问题，但却没有分享具体细节。

---

**目录**

[TOC] 


# 第一部分：高层级 Discord 摘要




## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **Polymind 插件难题**：@doctorshotgun 正在通过集成 PubMed API 的插件来增强 Polymind，以提升文章搜索能力。搜索结果的排序带来了开发复杂性。

- **AI 模型角色扮演装备**：用户分享了他们使用 AI 模型进行角色扮演的经历，指出 [PotatoOff](https://huggingface.co/PotatoOff/HamSter-0.2) 的 **HamSter v0.2** 是进行详细、不受限角色扮演的选择。同时，在训练 qlora dpo 等模型时，显存（VRAM）占用显著是一个常见挑战，其中 **Axolotl** 中的 `use_reentrant` 标志设置为 `False` 是导致 VRAM 消耗的关键因素。

- **定制 FLAN-T5 训练技巧**：在寻求训练代码生成模型时，@Naruto08 被引导考虑 FLAN-T5 等模型，并可参考 [Phil Schmid 的微调指南](https://www.philschmid.de/fine-tune-flan-t5)等资源。同时，@rolandtannous 提供了 DialogSum 数据集，作为在 p3.2xlarge AWS EC2 实例上进行微调尝试的可行资源。

- **模型合并大师**：@maldevide 介绍了一种分区层 **model merging** 策略，采用创新方法处理丢弃率为 92% 的 kvq 以及丢弃率为 68% 的分区层。该方法和配置已在 [GitHub Gist](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b) 上公开分享。

- **本地聊天机器人配置难题**：@aletheion 正在寻找一种为离线聊天机器人集成本地模型数据库查询的方法，@wildcat_aurora 建议考虑使用 [h2ogpt](https://github.com/h2oai/h2ogpt) 作为解决方案。此外，@vishnu_86081 正在探索使用 ChromaDB 在聊天机器人应用中实现特定角色的长期记忆。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- **应对 GPT-4 在歌词创作上的局限**：成员们讨论了 GPT-4 在生成准确歌词方面的问题，指出使用 *perplexity with search* 的效果优于 GPT-4 编造歌词的倾向。

- **LLM 量化路线图**：话题包括模型量化策略，例如使用 `llama.cpp` 进行量化过程，并讨论了 Mixtral 等模型在 4bit 精度下可能需要高达 40GB VRAM 的高效显存使用知识需求。

- **创新的模型合并方案**：社区重点关注了 **frankenmerging** 技术，发布了如 **miqu-1-120b-GGUF** 和 **MergeMonster** 等模型，并触及了如 **emulated fine-tuning (EFT)** 等新方法，考虑将基于 RL 的框架用于语言模型教育的各个阶段。

- **对新兴模型的期待与推测**：关于即将发布的 **Qwen2** 模型的讨论非常热烈，预测其将拥有显著的基准测试实力。偏好微调（Preference tuning）的讨论提到了 KTO、IPO 和 DPO 方法，并引用了一篇 Hugging Face 博客文章，该文章认为 IPO 与 DPO 相当，且比 KTO 更有效。

- **增强 AI 交互与测试的工具和框架**：提到的解决方案包括用于模型实验的 `text-generation-webui`，用于 OpenAI API 兼容服务器的 `ExLlamaV2`，以及用于自托管 LLM 聊天机器人测试的 [Lone-Arena](https://github.com/Contextualist/lone-arena)。此外，社区还注意到了一项关于 `llama.cpp` 中潜在支持 Apple Neural Engine 的 GitHub 讨论。



---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **幽灵进程！LM Studio 无法完全关闭**：用户报告称，在关闭 UI 后，**LM Studio** 仍继续在任务管理器中运行。建议的解决方法是**强制终止进程**并报告该 Bug。

- **AVX 指令集导致的 CPU 问题**：部分用户因处理器缺乏 **AVX2 支持**而遇到错误。社区指出 **LM Studio** 需要 AVX2，但某个 Beta 版本可能兼容仅支持 AVX 的 CPU。

- **Windows 11 上的 AMD GPU 计算探索**：对于希望在 Windows 11 上配合 LM Studio 使用 **AMD GPU** 的用户，支持 **ROCm 的 Beta 版本** LM Studio 是必不可少的。有用户报告在禁用内置显卡后，成功运行了 **AMD Radeon RX 7900 XTX**。

- **Whisper 模型与 Llama 的结合引发关注**：将 **Whisper 和 Llama 模型**与 LM Studio 集成是一个热门话题，用户被引导至 **Hugging Face** 上的特定模型以及 **Continue.dev** 等其他资源，以利用 LLM 进行编程。

- **LM Studio 中的持久进程与异常指标**：用户在 **LM Studio 的 Windows Beta 构建版本**中遇到了问题，包括 CPU 使用率数据不准确以及关闭后进程依然驻留。社区讨论中随后出现了要求改进 LM Studio 中 GPU 控制能力的呼声。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **LLama3 与 Mistral 集成见解**：社区成员推测了 **LLama3** 与其他模型在架构和训练数据上的差异，同时 **Mixtral** 在处理特殊字符和长文本方面的有效性也是热门话题。讨论还涉及了 **OpenHermes 2.5** 与 **Mistral** 之间的性能对比，特别是长上下文中的 "lost in the middle" 问题。此外，成员们还交流了在 Prompt 中处理 Markdown 的细节，以及使用 [GuardrailsAI](https://www.guardrailsai.com/) 和 [Instructor](https://jxnl.github.io/instructor/) 等工具进行故障排除的经验。

- **模型托管与开发难题**：考虑到可靠性和成本效益，[Hugging Face](https://hf.co/) 和 [Perplexity Labs](https://perplexity.ai/) 等服务被视为 AI 托管的可选方案。关于 LMM 的 CPU 推理讨论提出了不同模型大小和量化方法的适用性，其中 **Mistral 的量化**被重点提及。新用户被引导使用 [Gradio](https://www.gradio.app/) 和 [Hugging Face 的托管模型](https://huggingface.co/chat)等工具，以便在没有强大硬件的情况下部署模型。

- **微调焦点与财务现状**：针对能源市场分析等特定领域的微调问题得到了解答，强调了其可行性，但也指出了由于 Mistral 资源有限而存在的现有约束。社区探讨了 **Mistral** API 开发的当前局限性，并将高昂的推理成本和团队规模视为关键因素。

- **在创意领域展示 AI**：用户展示了 AI 辅助小说写作等应用，并对 AI 生成的叙事进行了评述。建议使用 *Claude* 等工具来提升 AI 写作体验，因为它具有更长的上下文容量。此外，[YouTube 视频](https://www.youtube.com/watch?v=N5lDUZRI8sc)中展示了 **ExLlamaV2** 在本地 GPU 上快速推理的能力。

- **从随机评论到平台特性**：一位 Y Combinator 创始人向社区征求关于**在 LLM 领域进行开发**时所遇挑战的见解。轻松一点的话题是，频道中意外出现了一些像国旗表情符号之类的俏皮消息。与此同时，在 **la-plateforme** 频道中，讨论了 **mistral-medium** 的流式传输问题与 **mistral-small** 行为不一致的情况，并提出了基于响应长度进行丢弃的临时解决方案。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- **与 ControlNet 作者的协作困境**：@pseudoterminalx 表达了与 ControlNet 创作者协作时面临挑战的*沮丧*，指出其过于关注推广 AUTOMATIC1111，而忽视了对社区集成工作的支持。这反映了其他工程师在实现过程中普遍感到的困难。

- **关于数据集实践的伦理辩论**：针对斯坦福研究人员在 LAION 数据集方面的行为提出了伦理质疑，暗示其在获得资金支持后转向以商业优先，可能影响公共开发和资源获取。

- **比较 AI 领域的先驱**：一场讨论比较了 Stability AI 与 NVIDIA 等巨头的策略。对话质疑了小型实体在采用与行业领导者类似的方法时，其创新能力是否受限。

- **关于 NVIDIA 显卡的硬件讨论**：工程社区就各种 NVIDIA 显卡（特别是 4060 ti 和 3090）是否适合 AI 模型训练进行了活跃交流，并考虑了 VRAM 需求和预算因素。

- **对 Stability AI 下一步行动的推测**：对 Stability AI 即将推出的模型的期待正在升温，@thejonasbrothers 对此类进展背景下长期项目的竞争力和可行性表示担忧。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

- **Falcon-180B 的 Demo 运行困难**：像 @nekoli. 这样的用户报告了 Hugging Face 上 Falcon-180B Demo 的问题，观察到全站性问题或 Demo 的特定故障。尽管分享了链接和建议，但解决情况似乎并不一致。

- **语言模型部署与使用咨询**：出现了关于使用 AWS Inferentia2 和 SageMaker 部署 Mistral 7B 等 LLM 的咨询，以及如何通过 HuggingFace 上的免费额度使用 API 访问 LLM，尽管随后没有链接相关的教学资源。

- **Spaces 卡住与基础设施困扰**：有报告称某个 Space 处于永久构建状态，以及 Hugging Face 潜在的更广泛基础设施问题影响了 Gradio 等服务。一些用户提供了排错建议。

- **关于 AI 在安全领域角色的辩论**：用户对 Deepfake 技术的滥用表示担忧，例如涉及虚假首席财务官（CFO）的诈骗。这突显了在开发和部署 AI 系统时伦理考量的重要性。

- **综合跨学科的社区见解**：讨论涵盖了广泛的话题，包括对奠基性论文 "Attention Is All You Need" 的钦佩、**Whisper** 在语音识别中 Speaker Diarization 的进展、开发以隐私为中心的音频摘要内部工具，以及用户参与各种 Hugging Face 社区活动（如博客撰写、活动和技术协助）。

- **Hugging Face 社区创新成果**：Hugging Face 社区分享了大量作品，从提议的语言模型机器人伦理框架到 [Autocrew for CrewAI](https://github.com/yanniedog/autocrew) 等项目、黑客辅助聊天机器人、基于推文的预测表情符号 Spaces，以及发布用于驱动专业领域模型的 [Hercules-v2.0 dataset](https://huggingface.co/datasets/Locutusque/hercules-v2.0)。

- **视觉与 NLP 领域的探索**：用户对寻找资源和协作项目表现出极高热情，例如带时间戳的视频摘要、LLM 伦理框架、拼写检查和语法模型，以及通过 [规划文档](https://docs.google.com/document/d/1fP2FIrCifWcLGdTBmqeogdCdZJOwxqPfEyO-HA76_qc/edit?usp=sharing)、[教程](https://huggingface.co/blog/mlabonne/merge-models) 和 [Colab notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing) 等资源追求北欧语言模型的模型合并（Model Merging）。

- **Diffusion 讨论中的诈骗警报与技术挑战**：一条诈骗信息被标记删除，详细讨论了一个关于 `AutoModelForCausalLM` 的 GitHub Issue，并分享了 Stable Video Diffusion 模型的 [许可协议](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main) 以讨论权重访问权限，这些都反映了社区在维护诚信和解决复杂 AI 问题方面的努力。

- **计算机视觉领域的参与**：出现了关于使用 Synthdog 生成伪数据、寻找用于 Zero-shot 视觉任务的当前模型，以及创建用于视觉 LLM 训练的滑动拼图数据集的问题，表明社区正在积极寻求 AI 的新颖方法。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 摘要

- **Local LLMs 在 GPT-4 遭受批评之际引发关注**：工程师们讨论了 GPT-4 的潜在替代方案，强调了 **Local LLMs**（如 LM Studio 和 perplexity labs）是可行的选择。用户对 GPT-4 的错误表示担忧，并探索了其他模型（如 codellama-70b-instruct）的性能。

- **GPT-4 故障引发工程师猜测**：关于 **@ 提及问题**和 GPT 行为异常（包括记忆丧失）的报告浮出水面，表明 GPT-4 系统可能存在不一致性。用户群还在应对缺失的功能（如点赞选项）和缓慢的 Prompt 响应时间。

- **Prompt Engineering 困扰专业人士**：AI 工程师对 ChatGPT 在故事创作中过度使用伦理准则表示沮丧，并建议避开 AI 语言模式，以在 AI 交流中保持类人交互。使用更稳定的 GPT 版本以保持指令一致性的建议也受到青睐。

- **托管 LLMs 的硬件障碍**：深入探讨运行 Local LLMs 的硬件配置，揭示了工程师在处理系统需求，特别是关于 RAM 与 VRAM 的争论。社区还对不同硬件配置下 AI 性能信息的可靠性表示怀疑。

- **AI 助手定制难题**：针对特定需求用户（如为自闭症用户生成类人语音）优化 GPT 交流以及避免名字拼写错误的策略进行了详细讨论。此外，一些用户遇到了意外的内容政策违规消息，并推测存在内部问题。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 摘要

- **工程师的 GPU 故障排除**：为了解决 RunPod 上的 GPU 错误，`@dangfutures` 推荐使用命令 `sudo apt-get install libopenmpi-dev pip install mpi4py`。此外，`@nruaif` 表示在 Llama70 上进行 LoRA 或 QLoRA 需要 80GB VRAM，通过冻结 MoE 层可以在 8 个 A6000 GPU 上实现 Mixtral FFT。

- **扩展雄心引发怀疑与乐观**：在一份新的 [Notion 文档](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)中，OpenBMB 的一个 2B 参数模型声称具有与 Mistral 7B 相当的性能，这在工程师中引发了怀疑和兴奋。

- **微调困境与代码配置调整**：`@cf0913` 遇到了微调后 EOS token 被当作 pad token 使用的问题，根据 `@nanobitz` 的建议通过编辑 tokenizer 配置得以解决。此外，`@duke001.` 寻求关于确定每个 epoch 训练步数的建议，并将 sequence length packing 作为一种潜在策略。

- **适配新架构**：提出了在 M1 MacBook Air 上运行 axolotl 包的问题，`@yamashi` 回复称将提交一个 PR 以使用 MPS 代替 CUDA。讨论还围绕在 M3 Mac 等新硬件上实现高级算法展开。

- **差分隐私带来的内存问题**：`@fred_fups` 在将差分隐私优化（DPO）与 qlora 结合使用时遇到了内存不足问题，`@noobmaster29` 确认 DPO 消耗大量内存，在 24GB RAM 下仅允许 microbatch size 为 1。

- **RunPod 初始化错误与配置担忧**：`@nruaif` 分享了来自 RunPod 的日志，显示了弃用的配置和错误，包括缺失的 `_jupyter_server_extension_points` 函数和错误的 `ServerApp.preferred_dir` 设置。`@dangfutures` 建议探索社区版本以获得更可靠的性能。



---

## [CUDA MODE (Mark Saroufim)](https://discord.com/channels/1189498204333543425) Discord 总结

**CUDA 热点**: CUDA 对 OpenCL 的主导地位归功于其广泛的普及和 Nvidia 的支持；Python 仍然是 GPU 计算的一个可行选择，在高级编程的易用性与 kernel 编写的细节之间提供了平衡，详见 [CUDA MODE GitHub repository](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing)。成员们还讨论了编译器优化对 CUDA 性能的影响，强调了代码中微小细节的重要性，同时提倡通过 [`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn) 等共享资源进行扎实的 CUDA 学习。

**PyTorch 解析器动态**: 分享了如何通过指定编译层来高效使用 `torch.compile` API 的技巧，如 [gpt-fast repository](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314) 所示。开发者们对更精细地控制 Torch 编译器的行为有着浓厚兴趣，[PyTorch documentation](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html) 提供了相关指导。在 PyTorch 的偏好中，TensorFlow 也得到了认可，主要是因为 Google 的硬件和定价。

**讲座预热**: 随着 CUDA MODE 关于计算和内存架构的第四次讲座的预告，人们的期待与日俱增。讲座材料可以在一个被戏称为“命名越来越不准确”的 **lecture2 repo** 中找到。该讲座承诺将深入探讨 blocks、warps 和内存层次结构的细节。

**招聘市场动态**: Aleph Alpha 和 Mistral AI 正在寻找 CUDA 专家，职位涉及将语言模型研究整合到实际应用中。专注于 GPU 优化和自定义 CUDA kernel 开发的职位正在招聘中，详见 Aleph Alpha [job listing](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en) 和 Mistral AI 的 [opportunity](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48)。

**CUDA 初学者集结**: Rust 在底层图形编程中获得了一些关注，讨论转向了其在 CUDA 编程中的可行性，引发了对 Rust 编写 CUDA GPU 项目的兴趣，例如用于 shaders 的 [rust-gpu](https://github.com/embarkstudios/rust-gpu)。Rust 神经网络领域正在升温，[Kyanite](https://github.com/KarelPeeters/Kyanite) 和 [burn](https://github.com/tracel-ai/burn) 等项目点燃了编码热情。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **TimesFM 训练澄清**: 分享了一个修正后的 **TimesFM model training** 序列，以强调基于模型描述的非重叠输出路径。同时，关于处理 LLMs 中长上下文的对话聚焦于 [YaRN paper](https://arxiv.org/abs/2309.00071)，并提出了一种名为 "liturgical refinement" 的自动编码方法。

- **MoE-Mamba 表现出色**: 根据[最近的一篇论文](https://arxiv.org/abs/2401.04081)，"MoE-Mamba" SSM 模型以更少的训练步数超越了其他模型。讨论了提高 AI 效率的策略，例如添加 router loss 以平衡 MoE 模型中的 experts，以及通过 [Encodec paper](https://arxiv.org/abs/2210.13438) 中的技术稳定梯度。

- **可解释性术语定义**: 在可解释性领域，指出了“方向”（direction，编码单语义含义的向量）与“特征”（feature，单个神经元的激活）之间的区别。

- **组织大规模协作**: 确认了关于大规模测试等主题的会议时间表，定于 **英国时间 2 月 6 日星期二下午 5 点**，其中提到了 **Slurm** 作为排队大量作业的工具。

- **多模态 MoE 模型探索**: 讨论转向将 MoEs 与 VLMs 及扩散模型结合以构建多模态系统，旨在实现更深层的语义和生成整合，并研究了 RNNs、CLIP、fast DINO 或 fast SAM 等替代方案。

- **GPT-NeoX "gas" 参数弃用**: GPT-NeoX 的一项更新涉及弃用 `"gas"` 参数，因为它被发现不起作用且与 `"gradient_accumulation_steps"` 重复，并警告过去的配置可能在无意中使用了较小的 batch sizes。相关的 [pull request](https://github.com/EleutherAI/gpt-neox/pull/123) 正在审查中。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **多语言 Perplexity**：用户对 Perplexity AI 的多语言能力表现出浓厚兴趣，并讨论了其在中文和波斯语方面的熟练程度。关于 Copilot 在模型性能中的作用，用户分享了截然不同的体验，但对其确切收益尚未达成共识。

- **批评客户服务**：用户 `@aqbalsingh` 在**邮箱修改流程**和 **iPhone 应用功能**方面遇到困难，导致其取消了 Premium 账户。他们与 `@otchuda` 一同表达了对 Perplexity AI 支持服务水平的不满。

- **通过 YouTube 表达的兴奋与分析**：由 *@arunprakash_*、*@boles.ai* 和 *@ok.alex* 制作的 YouTube 视频分析并评论了为什么用户可能更青睐 Perplexity AI 而非其他 AI 解决方案，视频标题如“我为了 PERPLEXITY 3.0 放弃了 BARD、ChatGPT 和 CLAUDE！”

- **分享搜索成功案例**：用户分享了影响他们决策的 Perplexity AI 搜索结果，例如升级到 Pro 订阅或协助解决复杂问题，突出了 Perplexity 搜索功能提供的实用性和可操作的见解。

- **Mixtral 的商业化困惑**：在 **#[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203350676214784010)** 频道中，人们对 **Mixtral 的定价**持续关注，当前费率为**每 1M input tokens 0.14 美元，每 1M output tokens 0.56 美元**。社区对 API 的 **pplx-web 版本**表现出兴趣，引发了关于 Perplexity AI 商业机会的讨论，尽管目前尚未披露官方计划。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **寻求阿拉伯语 AI 对话解决方案**：成员们讨论了处理阿拉伯语内容的技术方案，由于大多数技术与语言无关，建议使用阿拉伯语 LLM 和 embeddings。针对 [embedding-ada](https://www.sbert.net/examples/training/multilingual/README.html) 不支持的语言（如阿拉伯语），提到了 *aravec* 和 *word2vec* 等特定替代方案。

- **经济高效的 Agent 托管建议**：针对每次调用成本为 5 美分的研究型 Agent，建议包括托管本地 LLM 以控制成本，以及在 DigitalOcean 等公司的服务器上部署 [ollama](https://ollama.com/) 等服务。

- **面向 LLM 爱好者的书籍和学习资源**：一本名为《LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs》的新书发布，提供了涵盖 LangChain 使用案例和部署的实战指南，可在 [Amazon](https://amzn.eu/d/dqQJzV1) 购买。此外，还分享了一个包含丰富教程的 LangChain [YouTube 播放列表](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno)。

- **CastMate 助力交互式播客跨越式发展**：**CastMate** 正式推出，使听众能够使用 LLM 和 TTS 技术与播客节目进行交互。分享了一个 [Loom 演示](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74)，并可通过 [TestFlight 链接](https://testflight.apple.com/join/9khwQ1vD)获取 **iPhone beta** 版进行测试。

- **应对 LangChain 的初期障碍**：用户报告在按照 LangChain 教程操作时遇到错误和过时信息，这表明改进文档和支持材料的潜在方向。错误范围从直接遵循 YouTube 教程步骤到 LangChain 快速入门指南中 Ollama 模型的问题不等。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **RAG 痛点解决**：@wenqi_glantz 与 @llama_index 合作，解决了生产级 RAG 开发中的 12 个挑战，并在速查表中提供了完整解决方案，详见其 [Twitter 帖子](https://twitter.com/llama_index/status/1753829824484065625)。

- **DataStax 助力的黑客松**：@llama_index 感谢 `@DataStax` 主办并提供黑客松活动的餐饮支持，并在 [Twitter](https://twitter.com/llama_index/status/1753845015833686132) 上分享了更新。

- **Mac 上的本地多模态开发**：LlamaIndex 与 Ollama 的集成现在支持本地多模态应用开发，可用于结构化图像提取和图像描述等任务，详见 [第一天集成推文](https://twitter.com/llama_index/status/1753875735776018786)。

- **深入探索 RAG 中的递归检索**：`@chiajy` 探讨了 RAG 系统中的递归检索，并在其 Medium 文章 [Advanced RAG and the 3 types of Recursive Retrieval](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba) 中分享了三种技术：基于页面的（Page-Based）、以信息为中心的（Information-Centric）和以概念为中心的（Concept-Centric）。

- **混合检索因动态调整和贡献而受到赞誉**：@cheesyfishes 确认 **Hybrid Retriever** 的 **alpha** 参数可以动态修改；@alphaatlas1 建议使用混合检索加重排序（re-ranking）流水线，重点介绍了 **BGE-M3 模型**，并呼吁对稀疏检索（sparse retrieval）方法做出贡献，详见 [Hugging Face 上的 BGE-M3](https://huggingface.co/BAAI/bge-m3)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **请求 GPT API 联邦**：`@tiagoefreitas` 对带有 API 的 GPT 商店表示兴趣，希望 **@LangChainAI** 在 OpenGPTs 中实现联邦功能，以便通过 API 在不同服务器上使用 GPT。
- **拥抱开源模型而非传统写作**：开源模型的动态输出（如 **mlewd mixtral**）因能提升内容创作的乐趣和生产力，比传统写作更受推崇。
- **专业技术问答的兴起**：`@kaycebasques` 强调了 Sentry 的倡议，这是为开发者创建专业技术问答资源这一日益增长趋势的一部分，增强了信息的获取便利性。
- **Ollama Llava 的性能赞誉**：`@ashpreetbedi` 分享了在本地运行 **Ollama Llava** 时令人印象深刻的推理速度，表明其在消费级硬件上具有强劲性能。
- **备受关注的技术职业选择**：面对科技行业的多种路径，`@mr.osophy` 的职业困境体现了在 ML Engineering 的个人兴趣与眼前工作机会之间的权衡。

**相关链接**：

- 未提供关于 OpenGPTs 联邦的具体链接。
- 有关 AI 模型合并概念的见解，请参考：[Arcee and mergekit unite](https://blog.arcee.ai/arcee-and-mergekit-unite/)。
- 了解 Sentry 等专业技术问答平台的作用，请访问：[Sentry Overflow](https://technicalwriting.tools/posts/sentry-overflow/)。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **游戏炼金术揭示哈希秘密**：有一种理论认为，游戏中生成新组合的*意外延迟*可能是由于哈希机制导致的，即当预生成的组合池发生哈希未命中（hash miss）时，才会创建新元素。
- **可视化游戏词汇的谱系**：参与者有兴趣为游戏中的词汇组合创建可视化谱系以获得更深入的见解，可能会使用 *embeddings* 来绘制合成路径。
- **使用书签脚本（Bookmarklet）掌控局面**：提供了一个 *JavaScript 书签脚本*，它利用游戏的 `localStorage` 来导出和自动保存合成物品，使玩家能够直接在游戏体验中跟踪他们合成的所有材料。
- **Llama 2 AI 引擎揭秘**：正如开发者在推文中披露的那样，为游戏中创意元素组合提供支持的 AI 是 *Llama 2*，由 TogetherAI 提供。
- **元素顺序影响合成成功率**：研究发现，游戏中元素的组合顺序会影响合成结果，某些组合只有在物品按特定顺序叠加时才能成功，且服务器会记住尝试过的序列，以防止在后续尝试中反转。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **德语语言模型得到提升**：@johannhartmann 报告称，通过利用 **German dpo** 和 **laserRMT**，**mt-bench-de** 的分数有所提高，并一直使用 *dare_ties* 合并德语 7B 模型。尽管分享了 [资源链接](https://github.com/mayflowergmbh/intel_orca_dpo_pairs_de)，但特定性能变化的原因（包括数学能力的下降）仍不清楚。
- **LLM 上下文处理的研究探索**：@nsk7153 正在寻找关于能够管理长上下文提示词（long-context prompts）的大型语言模型（LLMs）的研究材料，并分享了一个包含当前发现的 [Semantic Scholar 搜索结果](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance)。
- **推出用于微调的 GermanRAG**：@rasdani 宣布发布 **GermanRAG** 数据集，该数据集专为微调**检索增强生成（Retrieval Augmented Generation）**模型而设计，并提供了 [GitHub 仓库](https://github.com/rasdani/germanrag) 以供访问和贡献。
- **斯堪的纳维亚基准测试的热情投射到德语模型**：@johannhartmann 表示有兴趣开发一个类似于 [ScandEval](https://scandeval.com/mainland-scandinavian-nlg/) 的基准测试，用于评估德语语言模型的性能。
- **即将推出的德国托管服务**：在 #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) 频道中，flozi00 提到他们目前正在努力提供德国托管服务。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

- **深入研究 Mistral-7B Open-Orca 的训练数据**：@njb6961 寻求关于使用其“经过筛选的精选 GPT-4 增强数据子集”复现 **Mistral-7B Open-Orca** 的细节。确定的数据集 [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca) 包含约 500,000 条 GPT-4 补全（completions），旨在实现高效训练。
- **数据集发现与确认**：@ufghfigchv 确认 [SlimOrca 数据集](https://huggingface.co/datasets/Open-Orca/SlimOrca) 是用于 **Mistral-7B Open-Orca** 的训练数据。该模型的**训练配置（training configuration）**应该可以在模型仓库的 **config** 子目录中找到。
- **商业联系难题**：@tramojx 请求关于**列表和营销提案**的营销联系方式，但在提供的消息历史中未得到回应。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 总结

- **AI 讨论中的不同视角**：对话涉及了嵌入（embedding）的对比方法，考虑使用**全文文本嵌入（whole document text embeddings）**与视觉嵌入技术。讨论围绕重新实现编码器/解码器（encoder/decoder）模型的潜力展开，并对该任务的具体涉及内容表示好奇。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **BentoML 简化模型部署**：@robotums 报告了使用 **BentoML** 部署模型的流畅体验，特别是使用 AWS 上的 VLLM 后端，并将其过程描述为“非常简单，只需运行 bento 即可”。
- **DSPy 框架提升语言模型编程**：@sourya4 强调了 [DSPy](https://github.com/stanfordnlp/dspy) 的发布，这是一个斯坦福大学发起的项目，旨在改变基础模型（foundation models）的编程方式。补充的 [YouTube 视频](https://www.youtube.com/watch?v=Dt3H2ninoeY) 进一步深入介绍了 DSPy 在创建自我改进的 LM 流水线方面的能力。

---

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

- **AIEF 保加利亚分会引起关注**：**AIEF Bulgaria Chapter** 举办了其**第二次月度聚会**，共有 90 人参加。活动涵盖了多个主题的“闪电演讲（Lightning Talks）”，并提供了丰富的社交机会。
- **多元化的闪电演讲激发兴趣**：关于 **QR Code 艺术、编织过去、LMMs (Large Language Models)、Zayo** 以及**在 AI 时代构建具有护城河的业务**的策略等演讲成为亮点。分会的 YouTube 频道很快将发布完整的录像。
- **聚焦 ChatGPT 实施策略**：Iliya Valchanov 带来的 **“ChatGPT 采用方法论”** 环节分享了将 ChatGPT 集成到业务流程中的见解，相关资源已通过 [Google Slides 文档](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx) 共享。
- **在社交媒体分享成功**：AIEF 保加利亚负责人 **@yavor_belakov** 在 LinkedIn 上分享了聚会的精彩瞬间，展示了 AIEF 相关 **AI 工程社区** 的活力与进步。
- **展示技术创新的演示文稿**：聚会演示文稿的幻灯片（包括 QR Code 艺术、历史编织、LLM 命令行工具、通过 Zayo 重塑员工管理以及 AI 领域稳健的商业模式）凸显了 AIEF 社区内的技术多样性和创新能力。

---

# 第二部分：分频道详细摘要与链接

### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1203252408356970547) (1738 条消息🔥🔥🔥): 

- **插件开发历险记**：用户 `@doctorshotgun` 正在为 [Polymind](https://link.to.polymind) 开发插件，旨在利用 PubMed 的 API 改进其文章搜索功能。他们目前正在整合 `pymed` 来构建和解析搜索查询，但在搜索结果的排序和相关性方面遇到了挑战。

- **探索 Miqu**：包括 `@nextdimension` 和 `@netrve` 在内的多位用户讨论了本地 LLM 模型 miqu-1-70b 的实用性。虽然一些人认为它很有用，但另一些人反映其生成结果不尽如人意，这可能归因于其生成参数设置。

- **对 Mixtral Instruct 的兴趣**：关于响应效率和质量的讨论正在进行中。用户如 `@doctorshotgun` 指出，在 70B 模型上处理大型 RAG 上下文时，响应速度较慢。

- **BagelMIsteryTour 崭露头角**：BagelMIsteryTour-v2-8x7B-GGUF 模型获得好评。`@ycros` 将其成功归功于 Bagel 模型与 Mixtral Instruct 的融合。根据用户测试，该模型非常适合角色扮演 (RP) 和通用问答任务。

- **Oobabooga vs Silly Tavern**：用户 `@parogar` 对 Oobabooga（可能是一个本地 LLM 运行器）的 API 更改表示沮丧，因为这阻碍了 Silly Tavern 的连接。他们正在寻找恢复到更具兼容性的旧版本的方法。

**提到的链接**：

- [Download Data - PubMed](https://pubmed.ncbi.nlm.nih.gov/download/#annual-baseline)：PubMed 数据下载页面。
- [Blades Of Glory Will Ferrell GIF - Blades Of Glory Will Ferrell No One Knows What It Means - Discover &amp; Share GIFs](https://tenor.com/view/blades-of-glory-will-ferrell-no-one-knows-what-it-means-provocative-gif-5313360)：点击查看 GIF。
- [movaxbx/OpenHermes-Emojitron-001 · Hugging Face](https://huggingface.co/movaxbx/OpenHermes-Emojitron-001)：未找到描述。
- [modster (mod ster)](https://huggingface.co/modster)：未找到描述。
- [BagelMIsteryTour-v2-8x7B-Q4_K_S.gguf · Artefact2/BagelMIsteryTour-v2-8x7B-GGUF at main](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF/blob/main/BagelMIsteryTour-v2-8x7B-Q4_K_S.gguf)：未找到描述。
- [NEW DSPyG: DSPy combined w/ Graph Optimizer in PyG](https://www.youtube.com/watch?v=rqR3LeR09gc)：DSPyG 是一种基于 DSPy 并结合了 PyG 图论见解的新型优化方案。展示了具有图优化的多跳（Multi Hop）RAG 实现的真实案例。
- [Artefact2/BagelMIsteryTour-v2-8x7B-GGUF at main](https://huggingface.co/Artefact2/BagelMIsteryTour-v2-8x7B-GGUF/tree/main)：未找到描述。
- [Terminator (4K) Breaking Into Skynet](https://www.youtube.com/watch?v=CNZXYAkmFpM)：终结者 (4K) 潜入天网。
- [AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA](https://youtu.be/7OUMZoHpVnM?feature=shared)：#meme #memes #funny #funnyvideo
- [152334H/miqu-1-70b-sf · Hugging Face](https://huggingface.co/152334H/miqu-1-70b-sf)：未找到描述。

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1203259027690229821) (678 条消息🔥🔥🔥): 

- **关于模型性能和偏好的讨论**：用户分享了使用各种 AI 模型进行角色扮演的经验，提到了 **goliath 120b**、**mixtral models** 以及 **limaRP** 和 **sensual nous instruct** 等变体。**@potatooff** 建议使用 [HamSter v0.2 模型](https://huggingface.co/PotatoOff/HamSter-0.2) 进行无审查角色扮演，并配合详细的角色卡，使用带有 chat-instruct 的 Llama2 提示词模板。
  
- **DPO 与模型训练的技术深度探讨**：进行了一场关于 DeeperSpeed (DPO) 占用大量 VRAM 及其对训练 AI 模型影响的技术对话。多位用户讨论了在 GPU 上适配 qlora dpo 等模型时遇到的困难。正如 **@doctorshotgun** 所解释的，这是因为 Axolotl 中 **gradient_checkpointing_kwargs** 的 `use_reentrant` 设置默认为 `False`，他们建议更改此设置以减少 VRAM 占用。

- **寻求优化角色卡的建议**：**@johnrobertsmith** 寻求关于优化 AI 角色扮演角色卡的建议，得到的建议包括将角色描述保持在 200 tokens 左右，并使用 lorebooks 来记录复杂的细节（如世界法术）。**@mrdragonfox** 分享了一个角色卡示例，并支持使用 lorebooks 以获得更好的角色定义。

- **探索各种模型的 VRAM 消耗**：包括 **@c.gato**、**@giftedgummybee** 和 **@kalomaze** 在内的用户讨论了某些 AI 模型的资源密集特性，特别是在使用 DPO 时，并分享了由于 DPO 缓存需求所需的重复而导致大量消耗的经验。

- **杂谈与笑话**：在以技术和性能为中心的讨论中，也有一些轻松时刻，用户们开玩笑说吵赢了 AI (**@mr.devolver**)，以及对发现的物体发出“很臭”的随机吐槽 (**@kaltcit** 和 **@stoop poops**)。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1111983596572520458/1112690728531918948/118465737669188415)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Chub](https://www.chub.ai/lorebooks/sr_misterioso/advanced-personality-traits-658eaf07)：查找、分享、修改、转换并对对话式大语言模型 (LLM) 的角色和其他数据进行版本控制。曾用名/别名 Character Hub, CharacterHub, CharHub, CharaHub, Char Hub。
- [PotatoOff/HamSter-0.2 · Hugging Face](https://huggingface.co/PotatoOff/HamSter-0.2)：未找到描述
- [cognitivecomputations/dolphin-2_6-phi-2 · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2_6-phi-2)：未找到描述
- [LoneStriker/Mixtral-8x7B-Instruct-v0.1-LimaRP-ZLoss-6.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/Mixtral-8x7B-Instruct-v0.1-LimaRP-ZLoss-6.0bpw-h6-exl2)：未找到描述
- [LoneStriker/limarp-miqu-1-70b-5.0bpw-h6-exl2 · Hugging Face](https://huggingface.co/LoneStriker/limarp-miqu-1-70b-5.0bpw-h6-exl2)：未找到描述
- [相比 4.36.2，Mixtral qlora 训练的 VRAM 占用显著增加？ · Issue #28339 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28339#issuecomment-1879894108)：系统信息：环境为 Runpod 容器，python 3.10，单张 A100 80gb，transformers 4.37.0dev (3cefac1)，使用 axolotl 训练脚本 (https://github.com/OpenAccess-AI-Collective/ax...

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1203424550285742172) (18 messages🔥): 

- **选择合适的代码生成模型**：`@Naruto08` 正在寻求关于使用 `[INST] {prompt} [/INST]` 格式的自定义数据集训练代码生成模型的建议。他们拥有 24GB 显存，并希望确保选择正确的模型和训练方法。
  
- **特定模型微调咨询**：用户 `@709986_` 询问模型 *em_german_mistral_v01.Q5_0.gguf* 是否可以进行微调，但未提供关于预期结果或微调过程的具体细节。

- **在有限资源下微调 Flan-T5**：`@tom_lrd` 询问了微调 flan-t5 模型的数据集大小和硬件要求，`@rolandtannous` 回应了在 AWS 实例上对 flan-t5-base 进行 LoRA 微调的经验，并分享了位于 [Huggingface 上的 DialogSum 数据集](https://huggingface.co/datasets/knkarthick/dialogsum) 的相关链接。

- **FLAN-T5 的易用微调**：`@rolandtannous` 分享了 FLAN-T5 基础模型易于微调的细节，考虑到其大小（约 900MB-1GB），并指出了 Phil Schmidt 在相关实验中使用了配备 NVIDIA V100 的 p3.2xlarge AWS EC2 实例。他们还提供了一份关于使用 SAMSUM 数据集进行对话摘要的 [FLAN-T5 微调综合指南](https://www.philschmid.de/fine-tune-flan-t5)。

- **澄清 Huggingface 上的 "Uncensored" 模型**：`@thisisloading` 询问了 Huggingface 上的 "uncensored"（无审查）模型，引发了关于如何移除此类模型对齐（alignment）过程的讨论，详见 Eric Hartford 的博客文章：["Uncensored Models"](https://erichartford.com/uncensored-models)。该过程类似于从基础模型中“外科手术式”地移除对齐组件，从而通过微调实现进一步的定制。

**提到的链接**：

- [Uncensored Models](https://erichartford.com/uncensored-models)：我发布这篇文章是因为很多人问我是如何做到的，所以我将进行解释。https://huggingface.co/ehartford/WizardLM-30B-Uncensored https://huggingface.co/ehartford/WizardLM-13B-Uncensore...
- [Fine-tune FLAN-T5 for chat &amp; dialogue summarization](https://www.philschmid.de/fine-tune-flan-t5)：了解如何使用 Hugging Face Transformers 为聊天和对话摘要微调 Google 的 FLAN-T5。

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1203391772223602708) (5 messages): 

- **创新的模型合并技术**：`@maldevide` 详细介绍了一种 *model merging*（模型合并）的新方法，将层划分为桶（buckets）并分别进行合并，对 **kvq** 采用了独特的处理方式，涉及 100% 的合并权重，但丢弃率（drop rate）高达 92%。
- **分区层合并结果**：按照这种新方法，`@maldevide` 提到如果分为四个分区，每个分区将以 **68% 的丢弃率**进行合并，并表示这一特定丢弃率产生了显著影响。
- **对新方法的兴趣**：`@alphaatlas1` 对 `@maldevide` 的合并方法表示感兴趣，并请求查看配置或自定义代码。
- **获取新模型合并代码**：`@maldevide` 通过提供其配置的 [GitHub Gist](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b) 链接响应了请求，允许他人查看并可能使用所述技术。

**提到的链接**：

[tinyllama-merge.ipynb](https://gist.github.com/maldevide/08829eada04ad9bd78e46c1a3787d42b)：GitHub Gist：即时分享代码、笔记和代码片段。

  

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1203477948209045505) (6 条消息): 

- **本地模型查询探索**: `@aletheion` 正在寻求帮助，希望实现一个功能，让 Chatbot 可以在本地/向量数据库中执行查询操作以提供答案，同时保持全程离线。他们表示愿意使用现有的框架或解决方案。
  
- **建议使用 h2ogpt 实现本地机器人**: `@wildcat_aurora` 分享了一个 [h2ogpt 的 GitHub 仓库](https://github.com/h2oai/h2ogpt)，该项目通过本地 GPT 提供私密问答和摘要功能，支持 100% 隐私保护，并宣传兼容多种模型，这可能是解决 `@aletheion` 疑问的一个方案。

- **API 困惑解开**: `@sunija` 对 Ooba 的 API 表示沮丧，因为尽管文档暗示不需要，但它仍要求提供 "messages" 字段，但随后意识到是自己的错误，并自我反思了对发起 Web 请求的厌恶。

- **模型评估成功**: `@london` 报告称，在另一位用户要求提交后，Code-13B 和 Code-33 模型在 EvalPlus 及其他平台上的评估中取得了成功。

- **聊天机器人应用旨在实现特定角色的长期记忆**: `@vishnu_86081` 正在寻求关于为他们的 Chatbot 应用设置 ChromaDB 的指导，该应用允许用户与多个角色聊天，目标是使用向量数据库存储和检索特定角色的消息，以实现长期记忆。

**提到的链接**:

[GitHub - h2oai/h2ogpt: Private Q&amp;A and summarization of documents+images or chat with local GPT, 100% private, Apache 2.0. Supports Mixtral, llama.cpp, and more. Demo: https://gpt.h2o.ai/ https://codellama.h2o.ai/](https://github.com/h2oai/h2ogpt): 私密文档+图像问答与摘要，或与本地 GPT 聊天，100% 隐私，Apache 2.0。支持 Mixtral, llama.cpp 等。演示地址：https://gpt.h2o.ai/ https://codellama.h2o.ai/ -...

  

---



### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1203326927276875866) (56 条消息🔥🔥): 

- **GPT-4 的歌词怪癖**: `@cccntu` 讨论了 GPT-4 在准确生成歌词方面的局限性，提到结合搜索使用 Perplexity 的效果比 AI 更好，因为 AI 往往会捏造内容。
- **Greentext 生成挑战**: `@euclaise` 认为 4chan 的 Greentext 格式可能由于缺乏训练数据而难以让 AI 学习，而 `@teknium` 分享了一个片段，展示了 AI 尝试模仿涉及 Gaia's Protector 的 Greentext 叙事，突显了捕捉特定故事讲述风格的挑战。
- **征集印度语 AI 创新者**: `@stoicbatman` 邀请从事印度语言 AI 开发的开发者和科学家申请由 IIT 提供的 GPU 计算资源和基础设施支持，以推进区域语言研究。
- **Llama2 在 4chan 数据上进行了预训练？**: `@stefangliga` 声称 4chan 的内容实际上是 Llama2 预训练集的一部分，反驳了其可能被刻意排除的假设。
- **苹果被指控为 AR/VR 开发制造障碍**: `@nonameusr` 批评了苹果对其技术生态系统的做法，认为该公司的限制性做法（如仅为了列出应用就收取年费）以及 Vision Pro 缺乏沉浸式 VR 游戏是 AR/VR 进步的障碍。

**提到的链接**:

- [Skull Issues GIF - Skull issues - Discover &amp; Share GIFs](https://tenor.com/view/skull-issues-gif-13031152103567454559): 点击查看 GIF
- [Join the Bittensor Discord Server!](https://discord.gg/JkRGPEPY): 在 Discord 上关注 Bittensor 社区 - 与 20914 名其他成员一起享受免费的语音和文字聊天。
- [Watch A Fat Cat Dance An American Dance Girlfriend GIF - Watch a fat cat dance an American dance Girlfriend Meme - Discover &amp; Share GIFs](https://tenor.com/view/watch-a-fat-cat-dance-an-american-dance-girlfriend-meme-gif-6193372123771306115): 点击查看 GIF
- [4chan search](https://4chansearch.com/?q=%3Ebe+me&s=1): 未找到描述
- [ExLlamaV2: The Fastest Library to Run LLMs](https://www.youtube.com/watch?v=N5lDUZRI8sc): 一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 https://github.com/turboderp/exllamav2 https://colab.research.google.com/github...
- [Indic GenAI Project](https://forms.gle/7iZnQjU9rwCr7wF9A): 我们正在召集所有从事生成式 AI 并为印度语言构建模型的开发者、科学家和其他人员。为了帮助研究社区，我们正在汇集最优秀的...
- [DarwinAnim8or/greentext · Datasets at Hugging Face](https://huggingface.co/datasets/DarwinAnim8or/greentext): 未找到描述
- [Llama GIF - Llama - Discover &amp; Share GIFs](https://tenor.com/view/llama-gif-21325230): 点击查看 GIF

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1203302897522057286) (42 messages🔥): 

- **拥抱 EFT 革命**：@euclaise 分享了一篇介绍 **emulated fine-tuning (EFT)** 的论文，这是一种利用基于 RL 的框架独立分析语言模型从预训练和微调阶段获得的知识的新技术。该论文挑战了对预训练和微调模型知识与技能相互作用的理解，提议以新方式潜在地结合它们 ([阅读论文](https://arxiv.org/abs/2310.12962))。
- **Frankenmerge 正式落地**：@nonameusr 介绍了 **miqu-1-120b-GGUF**，这是一个基于 *miqu-1-70b* 构建的 frankenmerged 语言模型，灵感来自 Venus-120b-v1.2、MegaDolphin-120b 和 goliath-120b 等其他大型模型，并强调了 [CopilotKit](https://github.com/CopilotKit/CopilotKit) 的支持 ([在 Hugging Face 上探索](https://huggingface.co/wolfram/miqu-1-120b))。
- **GPU 上的 FP6 量化**：@jiha 讨论了一种名为 **TC-FPx** 的大型语言模型六比特量化新方法，并询问了其实现和对比性能。@.ben.com 指出大多数任务的最佳精度及其在特定用例中的实际益处 ([查看摘要](https://huggingface.co/papers/2401.14112))。
- **模型中的梅赛德斯-奔驰**：@gabriel_syme 推测了正在讨论的新模型的潜在规模，用户们在猜测即将发布的 **Qwen 2** 模型及其与 **Wen-72B** 等前作相比的性能。该话题的讨论包括对模型规模和 benchmark 性能的预期。
- **领域内的新合并工具**：@nonameusr 展示了 **MergeMonster**，这是一种用于合并基于 Transformer 的语言模型的无监督算法，具有实验性的合并方法，并在合并每一层前后进行评估 ([在 GitHub 上发现](https://github.com/Gryphe/MergeMonster))。

**提到的链接**：

- [An Emulator for Fine-Tuning Large Language Models using Small Language Models](https://arxiv.org/abs/2310.12962)：广泛使用的语言模型 (LMs) 通常通过扩展两阶段训练流水线来构建：使用超大规模、多样化文本数据集的预训练阶段和微调（有时是...）
- [wolfram/miqu-1-120b · Hugging Face](https://huggingface.co/wolfram/miqu-1-120b)：未找到描述
- [Paper page - FP6-LLM: Efficiently Serving Large Language Models Through FP6-Centric Algorithm-System Co-Design](https://huggingface.co/papers/2401.14112)：未找到描述
- [Binyuan Hui (@huybery) 的推文](https://x.com/huybery/status/1754163638259388525?t=cDduW8-dHQD_fekk1_Qajg&s=33)：静待花开 🌸
- [GitHub - Gryphe/MergeMonster: An unsupervised model merging algorithm for Transformers-based language models.](https://github.com/Gryphe/MergeMonster)：一种用于基于 Transformers 的语言模型的无监督模型合并算法。

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1203261031900647424) (550 messages🔥🔥🔥): 

- **量化 Emoji**：成员 `@agcobra1` 和 `@n8programs` 参与了一场关于如何使用 `llama.cpp` 量化模型的教学环节。该过程包括克隆模型、使用 `git lfs pull` 拉取大文件，然后使用 `convert.py` 脚本进行转换并使用 `./quantize` 进行量化。

- **Qwen2 发布期待**：Qwen 模型团队暗示将发布 Qwen2，预计它将成为 benchmark 中的有力竞争者，甚至可能超越 Mistral medium 的性能。`@bratao` 分享了一个暗示 Qwen2 即将揭晓的 GitHub 链接。

- **关于未来数字接口的讨论**：`@nonameusr` 和 `@n8programs` 深入探讨了关于脑机接口潜在未来的推测性对话，想象了思想可以直接与数字系统交互而无需传统输入方法的场景。

- **文本生成 UI 和 API 易用性**：`@light4bear` 推荐使用 text-generation-webui 来轻松实验模型，而 `@.ben.com` 提供了一个与 OpenAI API 兼容的 ExLlamaV2 服务器实验，用于测试下游客户端。

- **偏好微调的实验与比较**：`@dreamgen` 询问了 KTO、IPO 和 DPO 方法在对齐语言模型方面的实际比较。随后引用了一篇 Hugging Face 博客文章，讨论了修正后的 IPO 实现结果，显示在偏好设置中 IPO 与 DPO 持平且优于 KTO。

**提到的链接**：

- [来自 Binyuan Hui (@huybery) 的推文](https://fxtwitter.com/huybery/status/1754163638259388525?t=cDduW8-dHQD_fekk1_Qajg&s=33)：静待花开 🌸
- [Google Colaboratory](https://colab.research.google.com/drive/1P646NEg33BZy4BfLDNpTz0V0lwIU3CHu)：未找到描述
- [CodeFusion: 用于代码生成的预训练扩散模型](https://arxiv.org/abs/2310.17680)：想象一下，如果一个开发者只能修改他们最后一行代码，那么在函数正确之前，他们需要从头开始编写多少次？用于代码生成的自回归模型...
- [cxllin/StableHermes-3b · Hugging Face](https://huggingface.co/cxllin/StableHermes-3b)：未找到描述
- [movaxbx/OpenHermes-Emojitron-001 · Hugging Face](https://huggingface.co/movaxbx/OpenHermes-Emojitron-001)：未找到描述
- [NousResearch/Nous-Capybara-3B-V1.9 · Hugging Face](https://huggingface.co/NousResearch/Nous-Capybara-3B-V1.9)：未找到描述
- [tsunemoto/OpenHermes-Emojitron-001-GGUF · Hugging Face](https://huggingface.co/tsunemoto/OpenHermes-Emojitron-001-GGUF)：未找到描述
- [wolfram/miquliath-120b · Hugging Face](https://huggingface.co/wolfram/miquliath-120b)：未找到描述
- [Social Credit GIF - Social Credit - 发现并分享 GIF](https://tenor.com/view/social-credit-gif-23329982)：点击查看 GIF
- [使用直接偏好优化方法对 LLMs 进行偏好微调](https://huggingface.co/blog/pref-tuning)：未找到描述
- [来自 AI Breakfast (@AiBreakfast) 的推文](https://x.com/aibreakfast/status/1754008072828158416?s=46)：Google 的 Gemini Ultra 刚刚确认将于周三发布。Ultra 在 8 项基准测试中的 7 项击败了 GPT-4，并且是第一个在 MMLU（大规模多任务语言理解）上超越人类专家的模型...
- [text-generation-webui/requirements.txt at main · oobabooga/text-generation-webui](https://github.com/oobabooga/text-generation-webui/blob/main/requirements.txt)：一个用于 Large Language Models 的 Gradio Web UI。支持 Transformers, GPTQ, AWQ, EXL2, llama.cpp (GGUF), Llama 模型。 - oobabooga/text-generation-webui
- [GitHub - PygmalionAI/aphrodite-engine: PygmalionAI 的大规模推理引擎](https://github.com/PygmalionAI/aphrodite-engine)：PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号来为 PygmalionAI/aphrodite-engine 的开发做出贡献。
- [关于使用 Qwen1.5-Chat 系列更新 MT-Bench 排行榜的请求 · Issue #3009 · lm-sys/FastChat](https://github.com/lm-sys/FastChat/issues/3009)：你好 LM-Sys 团队，我们想展示 Qwen1.5-7B-Chat, Qwen1.5-14B-Chat 和 Qwen1.5-72B-Chat 在 MT-Bench 上的生成结果和自测分数。能否请你们帮忙验证并...
- [GitHub - bjj/exllamav2-openai-server: 一个基于 ExLlamaV2 的 OpenAI API 兼容 LLM 推理服务器。](https://github.com/bjj/exllamav2-openai-server)：一个基于 ExLlamaV2 的 OpenAI API 兼容 LLM 推理服务器。 - GitHub - bjj/exllamav2-openai-server: 一个基于 ExLlamaV2 的 OpenAI API 兼容 LLM 推理服务器。
- [Notion – 集合笔记、任务、维基和数据库的一体化工作空间。](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)：一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间。
- [GitHub - lucidrains/self-rewarding-lm-pytorch: MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现](https://github.com/lucidrains/self-rewarding-lm-pytorch)：MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现 - GitHub - lucidrains/self-rewarding-lm-pytorch: MetaAI 提出的 Self-Rewarding Language Model 训练框架的实现...
- [在 Apple Neural Engine 上部署 Transformers](https://machinelearning.apple.com/research/neural-engine-transformers)：Apple 每年构建的机器学习 (ML) 模型中，有越来越多正在部分或全部采用 [Transformer…
- [openbmb/MiniCPM-V · Hugging Face](https://huggingface.co/openbmb/MiniCPM-V)：未找到描述
- [GitLive](https://git.live/)：在任何 IDE 中进行实时代码协作
- [由 Saibo-creator 提交的上下文无关文法约束解码（ebnf 接口，兼容 llama-cpp） · Pull Request #27557 · huggingface/transformers](https://github.com/huggingface/transformers/pull/27557)：这个 PR 做了什么？此 PR 为库添加了一个新功能（上下文无关文法约束解码）。虽然已经有一个关于此功能的 PR (#26520) 正在进行中，但这个 PR 采用了不同的...
- [Neural Engine 支持 · ggerganov/llama.cpp · Discussion #336](https://github.com/ggerganov/llama.cpp/discussions/336)：如果能利用 Neural Engine 就太酷了。即使速度没有快很多，我相信它仍然会更节能。

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1203253416688881735) (90 条消息🔥🔥): 

- **Hermes 模型混淆已澄清**：`@teknium` 澄清了 **Nous Hermes 2 Mixtral** 与 **Open Hermes 2 和 2.5** 之间的区别。后者是 7B Mistrals 模型，其中 Open Hermes 2.5 增加了 100,000 条代码指令（code instructions）。
- **Mixtral 的显存占用问题**：`@teknium` 和 `@intervitens` 讨论了 Mixtral 模型需要的 VRAM 大约是 7B 模型的 8 倍，在 4bit 精度下约为 40GB。`@intervitens` 随后提到，通过 8bit cache 和优化设置，3.5 bpw 在全上下文（full context）下可以适配。
- **Prompt 探讨**：`@tempus_fugit05` 收到来自 `@teknium` 和 `.ben.com` 关于他们在 Nous SOLAR 模型中使用的 prompt 格式的修正，指出其使用了错误的 prompt 模板。
- **MoEs 中的专家混淆解释**：`.ben.com` 解释了在 MoEs 中，专家（experts）是如何根据 router 的指令按比例混合的，并强调虽然专家是按层（per-layer）选择的，但它们的输出必须在最终混合中正确累加。
- **用于 LLM 聊天机器人测试的 Lone-Arena**：`.ben.com` 分享了 [Lone-Arena](https://github.com/Contextualist/lone-arena)，这是一个 GitHub 上的自托管聊天机器人竞技场代码库，用于个人测试 LLMs。

**提到的链接**：

- [来自 Geronimo (@Geronimo_AI) 的推文](https://x.com/geronimo_ai/status/1753685586634797113?s=46)：phi-2-OpenHermes-2.5 https://huggingface.co/g-ronimo/phi-2-OpenHermes-2.5 ↘️ 引用 Teknium (e/λ) (@Teknium1) 今天我有一个重大宣布。用于创建 Open Hermes 2.5 和 Nous 的数据集...
- [NousResearch/Nous-Hermes-2-SOLAR-10.7B · Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-SOLAR-10.7B#prompt-format)：未找到描述
- [teknium/OpenHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)：未找到描述
- [GitHub - daveshap/SparsePrimingRepresentations: 记录一些 SPR 内容的公开仓库](https://github.com/daveshap/SparsePrimingRepresentations)：记录一些 SPR 内容的公开仓库。通过在 GitHub 上创建账号来为 daveshap/SparsePrimingRepresentations 的开发做出贡献。
- [GitHub - Contextualist/lone-arena: 自托管 LLM 聊天机器人竞技场，你自己是唯一的裁判](https://github.com/Contextualist/lone-arena)：自托管 LLM 聊天机器人竞技场，你自己是唯一的裁判 - GitHub - Contextualist/lone-arena: Self-hosted LLM chatbot arena, with yourself as the only judge

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1203252881361342464) (225 条消息🔥🔥): 

- **持久的幽灵进程**：用户 `@nikofus` 报告称，即使关闭了 LM Studio UI，它仍继续显示在任务管理器中并占用 CPU 资源。为了解决这个问题，`@heyitsyorkie` 建议强制结束该进程并在特定频道中提交 bug 报告。

- **LM Studio 的“幽灵控制”**：`@vett93` 询问为什么在窗口关闭后 LM Studio 仍活跃在任务管理器中。`@heyitsyorkie` 解释说这是一个已知 bug，目前的解决方案是手动结束进程。

- **AVX 指令集的困扰**：用户 `@rachid_rachidi` 和 `@sica.rios` 因其处理器不支持 AVX2 指令而遇到错误。`@heyitsyorkie` 澄清说 LM Studio 需要 AVX2 支持，但对于仅支持 AVX 的 CPU，可以提供 beta 版本。

- **寻求 ROCm 支持**：`@neolithic5452` 询问如何在 Windows 11 环境下让 LM Studio 在 AMD 7900XTX GPU 上使用 GPU 计算，而不仅仅是 CPU。`@quickdive.` 建议使用支持 ROCm 的特殊 beta 版本 LM Studio 以获得 AMD GPU 计算能力，该版本可在频道的置顶消息中找到。

- **集成传闻**：`@lebonchasseur` 对在 LM Studio 中结合使用 Whisper 和 Llama 模型的经验表示兴趣，而 `@muradb` 询问了合适的视觉模型。用户被引导至 Llava，特别是 Hugging Face 模型页面上的一个版本。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1138544400771846174/1201187492414619791)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与您的朋友和社区保持紧密联系。
- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html)：未找到描述
- [jartine/llava-v1.5-7B-GGUF · Hugging Face](https://huggingface.co/jartine/llava-v1.5-7B-GGUF)：未找到描述
- [Advanced Vector Extensions - Wikipedia](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)：未找到描述
- [HuggingChat](https://huggingface.co/chat/)：让社区最好的 AI 聊天模型惠及每个人。
- [503 Service Unavailable - HTTP | MDN](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status/503)：超文本传输协议 (HTTP) 503 Service Unavailable 服务器错误响应代码表示服务器尚未准备好处理请求。
- [teknium/openhermes · Datasets at Hugging Face](https://huggingface.co/datasets/teknium/openhermes)：未找到描述
- [Yet Another LLM Leaderboard - a Hugging Face Space by mlabonne](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard)：未找到描述
- [GitHub - by321/safetensors_util: Utility for Safetensors Files](https://github.com/by321/safetensors_util)：Safetensors 文件实用工具。通过在 GitHub 上创建账号，为 by321/safetensors_util 的开发做出贡献。
- [Terminator Terminator Robot GIF - Terminator Terminator Robot Looking - Discover &amp; Share GIFs](https://tenor.com/view/terminator-terminator-robot-looking-flex-cool-robot-gif-978532213316794273)：点击查看 GIF
- [Pinokio](https://pinokio.computer/)：AI 浏览器

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1203271600414269470) (149 条消息🔥🔥): 

- **针对特定电脑配置的模型推荐**：`@mesiax.` 询问了在拥有 32GB RAM 和 12GB VRAM 且能充分利用 GPU 的电脑上表现最佳的模型。虽然 `@wolfspyre` 提供了一些建议，但最终建议他们通过实践开始测试和学习，因为不存在一劳永逸的解决方案。

- **模型更新与通知**：用户 `@josemanu72` 询问当新版本发布时是否需要手动更新模型。`@heyitsyorkie` 澄清说更新是一个手动过程，因为 LLM 会创建一个全新的模型，而不是更新现有模型。

- **VS Code 与 IntelliJ 插件对比**：`@tokman` 表示相比 VS Code 更偏好 IntelliJ，并在发现 VS Code 的一个有用扩展后，询问 IntelliJ 是否有类似的插件。`@heyitsyorkie` 提到了一种可能的变通方法，即通过服务器模式支持本地模型的 IntelliJ 插件。

- **Continue 集成与使用**：`@wolfspyre` 讨论了 [Continue.dev](http://continue.dev/) 的好处，它方便了在 IDE 中使用任何 LLM 进行编码，`@dagbs` 指向了一个可能是集成讨论空间的频道。

- **关于图像生成模型的查询**：`@kecso_65737` 寻求图像生成模型的推荐。`@fabguy` 建议使用 Stable Diffusion (SDXL)，但指出它在 LM Studio 上不可用，`@heyitsyorkie` 强调了这一点，同时提到了 Automatic1111 以方便在 LM Studio 之外使用。

**提到的链接**：

- [Continue](http://continue.dev/)：未找到描述
- [NeverSleep/MiquMaid-v1-70B-GGUF · Hugging Face](https://huggingface.co/NeverSleep/MiquMaid-v1-70B-GGUF)：未找到描述
- [Don't ask to ask, just ask](https://dontasktoask.com/)：未找到描述
- [⚡️ Quickstart | Continue](https://continue.dev/docs/quickstart)：Continue 入门指南
- [aihub-app/ZySec-7B-v1-GGUF · Hugging Face](https://huggingface.co/aihub-app/ZySec-7B-v1-GGUF)：未找到描述
- [Replace Github Copilot with a Local LLM](https://www.youtube.com/watch?v=F1bXfnrzAxM)：如果你是一名程序员，你可能听说过或已经在利用 GitHub Copilot。最近的进展使得运行你自己的 LLM 进行代码补全的能力成为可能...
- [John Travolta GIF - John Travolta - Discover &amp; Share GIFs](https://tenor.com/view/john-travolta-gif-25290651)：点击查看 GIF
- [christopherthompson81/quant_exploration · Datasets at Hugging Face](https://huggingface.co/datasets/christopherthompson81/quant_exploration#quants)：未找到描述

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1203451447698784286) (11 条消息🔥): 

- **模型下载疑云**：Stochmal 在下载模型时遇到问题，遇到了“失败”消息，且没有重试或恢复下载过程的选项。
- **Apple Silicon VRAM 难题**：`@musenik` 报告称，即使分配了 90GB 的 VRAM，模型 **Miquella 120B q5_k_m.gguf** 在 Apple Silicon 上的 **LM Studio** 中仍无法加载，而在 Faraday 上却能成功加载。
- **LM Studio vs. Faraday**：`@yagilb` 分享了一个假设，即 **LM Studio** 可能会尝试在 macOS 上将整个模型加载到 VRAM 中，这可能会导致问题，并暗示未来的更新将解决此问题。
- **寻找隐藏的开销**：`@musenik` 建议调查 LM Studio 在加载模型时是否存在潜在的不必要开销，因为 Faraday 在加载相同模型时带有一个 VRAM 开关且功能正常。
- **请求支持断点续传**：`@petter5299` 询问 LM Studio 未来是否会增加“恢复下载功能”，并对网络中断后下载重新开始表示沮丧。
  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1203271437276946432) (217 条消息🔥🔥): 

- **寻求通用模型建议**：用户 `@mesiax.` 询问了在拥有 32GB RAM 和 12GB VRAM 的 PC 上本地运行的最佳性能模型，希望利用 GPU 进行所有处理。其他用户并未给出具体的模型推荐，而是将对话转向了关于 GPU、RAM 速度以及运行 LLM 的 PCIe 带宽等详细硬件讨论。
- **RAM 速度与 GPU VRAM 之争**：包括 `@goldensun3ds` 在内的用户讨论了 RAM 速度对运行大型模型的影响，考虑从 DDR4 3000MHz 升级到 4000MHz 或更高。对话围绕系统权衡展开，例如升级 RAM 与增加 GPU 的对比，并触及了硬件兼容性和性能预期。
- **P40 GPU 讨论引发好奇与担忧**：`@goldensun3ds` 和 `@heyitsyorkie` 等成员辩论了 Nvidia Tesla P40 GPU 是否适合运行大型模型（如 120B Goliath）。提出的问题包括驱动程序兼容性、与较新 GPU 搭配时潜在的瓶颈，以及 P40 缺乏对未来模型更新支持的问题。
- **提及 Ryzen CPU 和 DDR5 RAM**：`@666siegfried666` 和 `.ben.com` 的讨论简要指出了某些 Ryzen CPU 和 DDR5 RAM 在本地模型推理方面的优势，尽管 X3D 缓存的有效性和 Navi 集成 NPU 仍存在争议。
- **探索可行的高 VRAM 配置**：`@quickdive.` 和 `@heyitsyorkie` 等用户研究了不同 GPU 设置（包括 P40、3090 和 4090）在深度学习任务中的潜力。共识倾向于使用更高 VRAM 的 GPU，以避免瓶颈并提高性能。

**提到的链接**：

- [Rent GPUs | Vast.ai](https://vast.ai)：通过最佳的云端 GPU 租赁服务，将您的云计算成本降低 3-5 倍。Vast.ai 简单的搜索界面允许公平比较所有提供商的 GPU 租赁价格。
- [B650 UD AC (rev. 1.0) Key Features | Motherboard - GIGABYTE Global](https://www.gigabyte.com/Motherboard/B650-UD-AC-rev-10#kf)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/13omfzw/comment/jl52q44/)：未找到描述
- [Nvidia's H100 AI GPUs cost up to four times more than AMD's competing MI300X &mdash; AMD's chips cost $10 to $15K apiece; Nvidia's H100 has peaked beyond $40,000: Report](https://www.tomshardware.com/tech-industry/artificial-intelligence/nvidias-h100-ai-gpus-cost-up-to-four-times-more-than-amds-competing-mi300x-amds-chips-cost-dollar10-to-dollar15k-apiece-nvidias-h100-has-peaked-beyond-dollar40000)：花旗银行表示，AMD 今年有望通过其 Instinct MI300 GPU 产生数十亿美元收益。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/13n8bqh/my_results_using_a_tesla_p40/)：未找到描述
- [EVGA GeForce RTX 3090 FTW3 ULTRA HYBRID 24GB GDDR6X Graphic Card 843368067106 | eBay](https://www.ebay.com/itm/276294867784?epid=24042942228&hash=item4054751b48:g:2OcAAOSwhsZlrDOb)：未找到描述

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1203379442148450405) (42 条消息🔥): 

- **图像分析能力存疑**：`@syslot` 确认 **Llava-v1.6-34b** 运行良好，而 `@palpapeen` 表示尽管安装了 vision adapter 且能在聊天中发送图片，但在使其分析图片时遇到困难。对于 `@palpapeen` 来说，该配置在 Llava1.5 7B 上有效，但在 Llava1.6 34B 上无效。

- **关于模型与处理器兼容性的讨论**：`@vic49.` 提到了 GitHub 上讨论的一个问题：使用 GGUF 格式将模型和处理器分离会导致 GGUF 无法利用 1.6 版本的更高分辨率。

- **Windows 11 上 AMD 的 ROCm 路径困境**：`@sierrawhiskeyhotel` 在 Windows 11 上使用 AMD 硬件时遇到了 “Model error”，但最终通过关闭集成显卡并使用 GPU Preference 设置解决了问题，确认成功使用了 AMD Radeon RX 7900 XTX。

- **对更多 GPU 控制权的诉求**：在讨论了 ROCm 配置故障排除和 GPU 利用率后，`@fabguy`、`@heyitsyorkie` 和 `@yagilb` 一致认为，能够更多地控制使用哪块 GPU 将会非常有益，社区内已针对此问题进行了讨论。

- **新的 Windows Beta 构建版本及报告的问题**：`@yagilb` 分享了一个新的 Windows Beta 构建版本链接，该版本改进了 LM Studio 显示 RAM 和 CPU 计数的方式。`@fabguy` 报告了 CPU 使用率指标不一致以及关闭应用后进程残留的问题，而 `@heyitsyorkie` 建议该进程 Bug 并不容易复现。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1195858490338594866/1197335753307467906)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1203750081648660500/1203750)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1202562325257650186)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1203750081648660500/1203750081648660500)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。

  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1203364915042189353) (4 条消息): 

- **POST 请求错误**：`@merpdragon` 分享了一个 Pastebin 链接，其中包含他们在向 `/v1/chat/completions` 发送 POST 请求时遇到的错误。[共享日志](https://pastebin.com/NSB1H6az)显示在处理关于儿童驾驶汽车的 Prompt 时出现了问题。
- **LM Studio 与 Autogen 设置问题**：`@j_rdie` 提到已将 LM Studio 与 Autogen 配合设置，确认了 Token 和模型验证通过，但面临模型无法直接输出的问题，仅在 Autogen 测试期间有输出。
- **Autogen 入门指南**：`@samanofficial` 询问如何开始使用 Autogen，`@dagbs` 提供了一个 Discord 频道链接以获取进一步指导。然而，从消息中无法辨别该链接的具体内容或指令。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1110598183144399058/1167546228813336686/1200572588850688101)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [[2024-02-03 16:41:48.517] [INFO] Received POST request to /v1/chat/completions w - Pastebin.com](https://pastebin.com/NSB1H6az)：Pastebin.com 是自 2002 年以来排名第一的文本存储工具。Pastebin 是一个可以在线存储一段时间文本的网站。

  

---

### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1203686204965650513) (2 条消息): 

- **本地 LLM 设置需要环境变量或代码修改**：用户 `@tok8888` 发布了一个代码片段，说明对于本地设置，必须设置 `openai_api_key` 环境变量，或者修改代码直接包含 API key。他们展示了一个将 API key 设置为 `"foobar"` 并修改 `ChatOpenAI` 初始化的示例。

- **关于使用 LM Studio 进行 App Store 优化的咨询**：用户 `@disvita` 询问群组成员如何利用 LM Studio 进行 App Store 优化，但在提问中未提供进一步的背景或细节。
  

---



### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1203257040374603866) (278 条消息🔥🔥): 

- **Llama3 预测及 Mixtral 技巧征集**：用户 `@frosty04212` 想知道 Llama3 是否会采用相同的架构但使用不同的训练数据，而 `@sheldada` 则在寻求有效提示 Mixtral 的技巧，并提到了一些奇怪的结果。`@ethux` 询问了 Mixtral 的使用场景，是通过 API、私有化部署（self-hosted）还是其他方法。
  
- **Mistral 的字符难题**：`@cognitivetech` 提出了 Mistral 处理特殊字符的问题，指出在处理学术文本时，某些字符如竖线 (`|`) 等存在问题。他们讨论了超过 10,000 个字符的输入挑战，以及不同字符和模型变体下结果的可变性。

- **模型性能讨论**：`@cognitivetech` 和 `@mrdragonfox` 交流了 OpenHermes 2.5 与 Mistral 在模型推理时间上的观察，注意到使用不同工具时的差异。他们还触及了被称为 "lost in the middle" 的现象，即在处理长上下文中间的相关信息时出现的性能问题。

- **有志于图像模型开发的开发者交流**：用户 `@qwerty_qwer` 向任何开发图像生成模型的人提供 6 亿张高质量图像，引发了与 `@i_am_dom` 和 `@mrdragonfox` 关于从零开始训练模型的可行性和计算挑战的讨论。

- **Function Calling 功能请求及 Office Hours 批评**：`@jujuderp` 哀叹 Mistral API 缺乏 function calling 和 JSON 响应模式，并引用了一篇社区帖子；而 `@i_am_dom` 对 office hours 环节提出了批评，将其与 Google 在 Bard Discord 上的做法进行了比较，并指出 Mistral AI 缺乏信息丰富的回应。

**提到的链接**：

- [GroqChat](https://groq.com/)：未找到描述
- [Introduction | Mistral AI Large Language Models](https://docs.mistral.ai/)：Mistral AI 目前提供两种访问 Large Language Models 的方式：
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)：虽然最近的语言模型能够将长上下文作为输入，但关于它们如何有效利用更长上下文的了解相对较少。我们分析了语言模型在两个任务上的性能...
- [Backus–Naur form - Wikipedia](https://en.wikipedia.org/wiki/Backus%E2%80%93Naur_form)：未找到描述
- [Let&#39;s build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)：我们按照论文 &quot;Attention is All You Need&quot; 以及 OpenAI 的 GPT-2 / GPT-3 构建了一个 Generatively Pretrained Transformer (GPT)。我们讨论了...
- [[Feature Request] Function Calling - Easily enforcing valid JSON schema following](https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515/14)：现在已经有一个非常弱的版本。模型可以被强制遵守 JSON 语法，但不能遵循特定的 schema，因此它仍然相当无用。我们仍然需要验证返回...
- [[Feature Request] Function Calling - Easily enforcing valid JSON schema following](https://community.openai.com/t/feature-request-function-calling-easily-enforcing-valid-json-schema-following/263515)：嗨，我非常激动地看到新的 function calling 功能，但很快就失望地发现它不能保证有效的 JSON。这让我特别惊讶，因为我个人已经实现了...

  

---

### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1203350592387416076) (45 条消息🔥): 

- **探索 AI 托管选项**：用户 `@i_am_dom` 建议将 [Hugging Face](https://hf.co/) 作为免费且可靠的 AI 模型托管服务；随后，他们还提到了 [Perplexity Labs](https://perplexity.ai/) 作为另一个托管选项。`@ashu2024` 对这些信息表示感谢。

- **探索适用于 CPU 推理的最佳模型**：`@porti100` 征求了关于在 CPU 上运行小型 LLM 并结合 RAG 的建议，`@mrdragonfox` 推荐了 7b 模型，但警告说在 CPU 上运行会很慢。随后展开了关于低端系统性能差异以及各种 7b 量化模型效率的简短讨论。

- **强调 Mistral 卓越的量化表现**：`@cognitivetech` 分享了他们的经验，认为 [Mistral 的量化](https://github.com/cognitivetech/llm-book-summarization) 表现优于其他模型，尤其是自 0.2 版本以来。他们强调需要在理想条件下测试完整模型，以获得准确的评估。

- **执行语言对 AI 性能的影响**：`@cognitivetech` 报告了使用 Go 和 C++ 代替 Python 时性能的显著差异，而 `@mrdragonfox` 则认为，由于底层操作是用 C++ 编写的，接口语言不应严重影响结果。

- **Mistral AI 入门**：新成员 `@xternon` 询问在笔记本电脑配置不足以运行模型的情况下如何使用 Mistral AI，得到的建议是使用 [Gradio](https://www.gradio.app/) 构建演示 Web 界面，或使用 [Hugging Face 的托管模型](https://huggingface.co/chat) 获得便捷的浏览器体验。`@adriata3` 指出了本地 CPU 使用的选项，并推荐了包含 Mistral 代码示例的 GitHub 仓库，同时提到 Kaggle 也是一个潜在的免费资源。

**提到的链接**：

- [Gradio](https://www.gradio.app/)：构建并分享令人愉悦的机器学习应用。
- [HuggingChat](https://huggingface.co/chat)：让每个人都能使用社区最好的 AI 聊天模型。
- [客户端代码 | Mistral AI 大语言模型](https://docs.mistral.ai/platform/client/)：我们提供 Python 和 Javascript 的客户端代码。

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1203277393687613490) (17 条消息🔥): 

- **Mistral 在 Markdown 处理上的失误**：`@drprimeg1` 遇到了 **Mistral Instruct AWQ** 在给定带有 Markdown 格式的 Prompt 时，无法在 JSON 格式内输出内容的问题。[他们目前的分类方法可以在这里找到](https://paste.ofcode.org/eRm2UGSzZfGvMqTNypdN2c)，但模型返回的是占位符而非实际内容。

- **模型中的 Markdown 混乱**：`@ethux` 建议 `@drprimeg1` 的问题可能是由 **Markdown 格式化** 引起的，并指出模型尝试输出 JSON，但最终显示的是 Markdown 语法。

- **使用 GuardrailsAI 引导 Prompt 有效性**：`@ethux` 提供了一个解决方案，推荐使用 **GuardrailsAI** 作为确保正确输出格式的工具，并提到它具有强制输出和失败重试的能力。他们还附带了该工具的引用链接 [GuardrailsAI](https://www.guardrailsai.com/)。

- **关于 Teacher Forcing 的讨论**：`@ethux` 提到 **GuardrailsAI** 通过提供错误原因及纠正方法的预定义示例，实现了一种 Teacher Forcing。

- **Instructor 介绍**：作为结构化输出生成的另一个建议，`@ethux` 分享了 **Instructor** 的链接，这是一个由 OpenAI 的 Function Calling API 和 Pydantic 数据验证驱动的工具，被描述为简单且透明。更多见解和该工具的社区可以通过 [Instructor 官网](https://jxnl.github.io/instructor/) 访问。

**提到的链接**：

- [您的企业级 AI 需要 Guardrails](https://www.guardrailsai.com/)：您的企业级 AI 需要 Guardrails。
- [欢迎使用 Instructor](https://jxnl.github.io/instructor/)：未找到描述。
- [Paste ofCode](https://paste.ofcode.org/eRm2UGSzZfGvMqTNypdN2c)：未找到描述。

  

---

### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1203389040699777035) (17 messages🔥): 

- **能源市场 Fine-tuning 指导**：`@tny8395` 询问了关于训练一个用于自动化能源市场分析模型的问题，`@mrdragonfox` 告知可以针对此类特定用途进行 Fine-tuning。
- **频道说明与反垃圾信息警告**：`@mrdragonfox` 引导 `@tny8395` 在当前频道讨论 Fine-tuning，并提醒他们发送垃圾信息（spamming）不会得到额外的回应。
- **Mistral 与 Fine-tuning API 开发**：`@a2rette` 询问 Mistral 是否计划开发 Fine-tuning API。`@mrdragonfox` 回应称，由于推理成本和团队规模较小，目前存在局限性，结论是目前“还没有”。
- **Mistral 的资源现状**：`@mrdragonfox` 提供了 Mistral 运营规模的背景信息，解释说尽管获得了资金，但行业的高成本和约 20 人的小团队使得某些开发工作具有挑战性。
- **寻求 Mistral 结合 Together AI 的 Fine-tuning 信息**：`@andysingal` 询问了关于 Mistral 结合 Together AI 进行 Fine-tuning 的资源，但未收到直接回复。
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1203356848753872928) (7 messages): 

- **ExLlamaV2 在 YouTube 上亮相**：`@pradeep1148` 分享了一个名为 [“ExLlamaV2: The Fastest Library to Run LLMs”](https://www.youtube.com/watch?v=N5lDUZRI8sc) 的 YouTube 视频，重点介绍了一个用于在本地 GPU 上运行 LLM 的快速推理库。他们还提供了该项目的 GitHub 链接和 Google Colab 教程。
- **AI 辅助小说创作**：`@caitlyntje` 描述了他们使用 AI 创作小说的过程，包括生成大纲、章节摘要，然后对每一章进行迭代以确保一致性、风格和细节。由于其 MacBook 在 Token 处理上的限制，该过程分阶段进行。
- **AI 辅助写作过程中的仔细监控**：在后续讨论中，`@caitlyntje` 提到在使用 AI 写作时，为了保持逻辑流和时间线的连贯，必须进行仔细的监督。
- **模型容量建议**：针对 `@caitlyntje` 提到的限制，`@amagicalbook` 建议尝试 *Claude*，据称它可以处理高达 200k 的 Token 上下文。
- **对 AI 生成的《碟形世界》叙事的批评**：`@swyxio` 是 Terry Pratchett 的粉丝，他批评一段 AI 生成的《碟形世界》（Discworld）叙事未能捕捉到女巫等标志性角色的精髓，导致他停止了阅读。

**提到的链接**：

[ExLlamaV2: The Fastest Library to Run LLMs](https://www.youtube.com/watch?v=N5lDUZRI8sc)：一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库。https://github.com/turboderp/exllamav2 https://colab.research.google.com/github...

  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1203827731201134622) (2 messages): 

- **YC 创始人寻求 LLM 挑战**：用户 `@znrp` 是 Y Combinator 的创始人，正在寻求了解社区成员在 **LLM 领域构建产品**时面临的挑战。他们欢迎通过私信进行简短交流。
- **Random 频道中的旗帜飞扬**：用户 `@gafty` 通过一条包含 **罗马尼亚国旗** 和搞怪表情的简单消息表达了他们的兴奋或顽皮。
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1203468704340377670) (4 messages): 

- **流传输中断问题**：`@jakobdylanc` 遇到了 **mistral-medium** 在流式响应中不发送最后一个空块（empty chunk）的问题，这与在 **mistral-small** 中观察到的预期行为不同。
- **权宜之计，而非完整解决方案**：针对流传输问题，`@drones_flier` 建议丢弃长度低于特定值的响应作为临时解决方案，尽管他们指出这可能并不适用于所有用例。
  

---

### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1203295648007331840) (361 条消息🔥🔥): 

- **对 Fooocus 的挫败感**：`@pseudoterminalx` 表达了对与 ControlNet 作者合作以及将他们的模型适配到其他平台的不满，描述了协作中的困难，以及对方过于关注推广 AUTOMATIC1111。讨论了这些开发者是否愿意考虑社区需求的担忧。
- **对 ControlNet 采用及斯坦福研究人员的担忧**：`@astropulse` 和 `@pseudoterminalx` 等几位用户分享了在项目中实现 ControlNet 的困境，指出缺乏信息和支持。围绕与 LAION 数据集相关的斯坦福研究人员的伦理和行为展开了讨论，暗示其存在商业优先的心态，且在获得资金成功后缺乏公开开发。
- **关于科技巨头和 AI 模型训练的辩论**：`@pseudoterminalx`、`@thejonasbrothers` 和 `@drhead` 讨论了 Stability AI 的做法与 NVIDIA 的一致性，涉及追随科技巨头脚步的策略，并质疑小型实体的独立创新能力。
- **针对 AI 模型的显卡讨论**：在一系列交流中，`@ninyago` 和 `@vrus0188` 等用户讨论了各种 NVIDIA 显卡型号（如 4060 Ti 和 3090）运行 AI 模型的适用性，并考虑了 VRAM 和预算。
- **对 Stability AI 新模型发布的推测**：包括 `@thejonasbrothers` 和 `@vrus0188` 在内的几位用户谈论了 Stability AI 即将发布的新模型，`@thejonasbrothers` 为自己进行了六个月的项目感到惋惜（考虑到新模型的能力），并对不得不与 Stability AI 这样拥有大量资源的对手竞争表示失望。

**提到的链接**：

- [Model Memory Utility - 由 hf-accelerate 提供的 Hugging Face Space](https://huggingface.co/spaces/hf-accelerate/model-memory-usage)：未找到描述
- [Open LLM Leaderboard - 由 HuggingFaceH4 提供的 Hugging Face Space](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)：未找到描述
- [来自 Angry Tom (@AngryTomtweets) 的推文](https://fxtwitter.com/AngryTomtweets/status/1753905168599462175)：Apple 发布 Vision Pro 仅 1 天，人们就为之疯狂。这里有 10 个你不容错过的疯狂案例：1. Apple Vision Pro Deepfake 应用概念   
- [来自 Emad (@EMostaque) 的推文](https://x.com/EMostaque/status/1751264392828653974?s=20)：正在和朋友们尝试一个刚出炉的 @StabilityAI 实验性基础模型，感觉像是另一个 Stable Diffusion 时刻，说实话。这是什么巫术 🧙🏽‍♂️🪄✨ 坐稳了 🍑 ↘️ Quo...
- [PNY GeForce RTX 3090 24GB XLR8 Gaming REVEL EPIC-X RGB 三风扇版 #3 | eBay](https://www.ebay.de/itm/296205530951?hash=item44f739b747%3Ag%3A-SAAAOSwejNlv9ed&amdata=enc%3AAQAIAAAAwPuosBH1RVyMFwV2oqWRFeFtqoUbeNTquMPzgrcjK6fXWpO0U1%2F1kBogvxyue34J9hm%2Ba3q5hOJzxF3R53qZ7xvcmty4FW11KI9O1dgI7Yg19oqkUZzKDzitEtkoRG%2BaKmWuj3O5zoTjw83mBIMAN5Nal4ssU3VPmXEG57H6NCpRGffCX7agsUYiP62MnjiNlMdQjN%2Ff9QrSoA9oG5mQcOS5qRHF9VJN1lHf6YG7auZGUXRSiViiaOH8siM%2FsyvPWA%3D%3D%7Ctkp%3ABk9SR7KF9N-uYw&LH_All=1)：未找到描述

  

---


### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1203439431168827442) (3 条消息): 

- **表情符号胜过千言万语**：用户 `@nodja` 发布了一对有趣的 👀，没有任何随附文本，让我们都悬在半空。
- **Hugging Face 上的 Qwen-VL-Max 引起关注**：`@nodja` 分享了 [Qwen-VL-Max 的 Hugging Face Space](https://huggingface.co/spaces/Qwen/Qwen-VL-Max) 链接，但有人指出它是 [Qwen/Qwen-VL-Plus](https://huggingface.co/spaces/Qwen/Qwen-VL-Plus) 的副本，具有相同的模型头像图片。
- **澄清与撤回**：在分享链接后不久，`@nodja` 紧接着发了一条简单的 "nevermind"（没关系），表明之前的消息可能是误发。

**提到的链接**：

[Qwen-VL-Max - 由 Qwen 提供的 Hugging Face Space](https://huggingface.co/spaces/Qwen/Qwen-VL-Max)：未找到描述

  

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1203333705632849980) (289 条消息🔥🔥): 


<ul>
  <li><strong>关于 Falcon-180B 及其 Demo 无法运行的讨论：</strong> 用户如 <code>@nekoli.</code> 报告了访问 <strong>HuggingFace</strong> Demo 页面（如 Falcon-180B）时遇到的问题，这表明可能存在全站性问题或特定的 Demo 故障。虽然分享了 Falcon-180B 的链接和建议，但成功率各不相同。</li>
  <li><strong>关于 LLM 部署和使用的提问：</strong> <code>@rishit_kapoor</code> 询问了关于通过 AWS Inferentia2 和 SageMaker 部署 Mistral 7B 的教程，而 <code>@_sky_2002_</code> 则寻求有关在 HuggingFace 上使用免费额度通过 API 调用 LLM 的信息。</li>
  <li><strong>Spaces 的技术协助：</strong> <code>@dongd.</code> 就一个卡在构建（building）状态的 Space 寻求帮助，<code>@not_lain</code> 提供了排障建议。对话涉及了依赖项问题以及 “factory rebuild” 的功能。</li>
  <li><strong>Hugging Face 基础设施问题：</strong> <code>@lolskt</code> 和 <code>@wubs_</code> 指出了 Hugging Face 可能存在的基础设施问题，这可能会影响 Gradio 和其他服务，同时用户分享了通过切换硬件来解决问题的方法。</li>
  <li><strong>AI 对安全的影响：</strong> <code>@aifartist</code> 针对一则关于 deepfake 首席财务官（CFO）参与诈骗的新闻报道，反思了 deepfake 技术的潜在影响，并对该技术可能带来的破坏性用途表示担忧。</li>
</ul>

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/879548962464493619/1019296127847239751/1203736452060684338)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、闲逛，并与你的朋友和社区保持紧密联系。
- [财务人员在与 deepfake “首席财务官”视频通话后支付了 2500 万美元 | CNN](https://www.cnn.com/2024/02/04/asia/deepfake-cfo-scam-hong-kong-intl-hnk/index.html)：未找到描述
- [非工程师指南：训练 LLaMA 2 聊天机器人](https://huggingface.co/blog/Llama2-for-non-engineers)：未找到描述
- [Falcon-180B Demo - lunarflu 创建的 Hugging Face Space](https://huggingface.co/spaces/lunarflu/falcon-180b-demo-duplicate)：未找到描述
- [捂脸真的 GIF - 捂脸压力山大 - 发现并分享 GIF](https://tenor.com/view/facepalm-really-stressed-mad-angry-gif-16109475)：点击查看 GIF
- [Falcon-180B Demo - tiiuae 创建的 Hugging Face Space](https://huggingface.co/spaces/tiiuae/falcon-180b-demo)：未找到描述
- [unalignment/toxic-dpo-v0.1 · Hugging Face 数据集](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.1)：未找到描述
- [Vizcom](https://www.vizcom.ai/)：Vizcom 是一款专为设计和创意专业人士打造的 AI 驱动创意工具。它提供了一种变革性的概念绘图方法，使用户能够将草图转化为令人印象深刻的实景...
- [tiiuae/falcon-180B · Hugging Face](https://huggingface.co/tiiuae/falcon-180B)：未找到描述
- [Superagi Sam - Tonic 创建的 Hugging Face Space](https://huggingface.co/spaces/Tonic/superagi-sam/)：未找到描述
- [LLaMa Chat | 文本生成机器学习模型 | Deep Infra](https://deepinfra.com/chat)：探索 LLaMa Chat 演示，让你与 llama 70b, llama 13b, llama 7b, codellama 34b, airoboros 30b, mistral 7b 等模型进行对话！
- [Space 无法启动 - 未找到日志](https://discuss.huggingface.co/t/space-wont-start-logs-not-found/54149/2)：你好 @155elkhorn，你能分享更多细节吗？你有公开的 Space 链接可以分享吗？谢谢
- [由 rgargente 更新 bonus unit1 链接 · Pull Request #485 · huggingface/deep-rl-class](https://github.com/huggingface/deep-rl-class/pull/485)：Bonus unit 1 笔记本重复了。bonus unit 1 的链接指向旧版本，没有更新 Huggy.zip 的下载链接。此 PR 更新了链接并移除了旧笔记本...
- [afrideva/phi-2-uncensored-GGUF · Hugging Face](https://huggingface.co/afrideva/phi-2-uncensored-GGUF)：未找到描述
- [D5648R DDR5-4800 64GB ECC Reg Server Memory 4800MHz PC5-38400 CL40 1.1V RDIMM 288-pin Memory](https://www.centralcomputer.com/samsung-m321r8ga0bb0-cqk-64gb-ddr5-ecc-registered-4800mhz-pc5-38400-cl40-1-1v-rdimm-288-pin-memory.html?srsltid=AfmBOoqj9W5RhnchoAOI9d03r2E8ODlO4UBD0K1j4MJ8eymSRDEsn64V1uU)：常规信息产品类型 RAM 模块品牌名称 Samsung 制造商 Samsung 产品名称 64GB DDR5 SDRAM 内存模块制造商部件号 M32...

  

---

### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1203257438590210079) (5 条消息): 

- **“Attention” 论文令人震撼**：`@sardarkhan_` 在阅读了开创性的 *"Attention Is All You Need"* 论文后表示非常惊讶，这对他对该主题的理解产生了重大影响。
- **语音识别的进化**：`@myke420247` 成功尝试使用 **Whisper** 和用于说话人日志（speaker diarization）的 **pyannote** 将公司通话录音从 wav 转换为文本，获得的结果优于 2018 年的 **Google 付费服务**。
- **构建音频摘要工具**：`@n278jm` 正在创建一个内部工具来总结咨询会议的音频录音，分享了代码尝试，并强调了仅在本地处理以确保隐私的承诺。
  

---


### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1203259884980801596) (11 条消息🔥): 

- **情感分析深度探讨**：`@andysingal` 重点介绍了一个结合 Hugging Face 和 Deepgram 的情感分析详细教程。它展示了如何创建图表以了解情感随时间的变化，并包含了情感分析图表的视觉效果。[Sentiment Analysis with Hugging Face and Deepgram](https://deepgram.com/learn/sentiment-analysis-with-hugging-face-and-deepgram)

- **在 Hugging Face Hub 上发布博客**：`@lunarflu` 鼓励 `@imcoza1915` 在 Hugging Face Hub 上撰写社区博客文章，以提高其工作的曝光度，并链接到了 [Hugging Face Blog Explorers](https://huggingface.co/blog-explorers) 社区。

- **Agent-Helper Langchain 的发布**：`@4gentbur3k` 分享了一个 Hugging Face 博客文章链接，讨论了将 Hugging Face 的 transformers 与 Langchain 集成以用于高级 NLP 应用。该文章展示了结合这些工具如何改进语言理解和生成。[Agent Helper Langchain Blog Post](https://huggingface.co/blog/Andyrasika/agent-helper-langchain-hf)

- **Art Forge Labs AI 艺术生成**：`@wubs_` 对 Art Forge Labs 的一篇文章表示惊讶，该文章详细介绍了 AI 驱动的艺术生成在速度和质量方面的重大提升，但未提供所提内容的有效 URL。

- **模型微调资源列表**：`@andysingal` 分享了创建模型微调资源列表的进度更新，并提供了该列表正在汇编的 GitHub 仓库链接。可通过 [llm-course GitHub repository](https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md) 访问微调列表。

**提到的链接**：

- [Uniting Forces: Integrating Hugging Face with Langchain for Enhanced Natural Language Processing](https://huggingface.co/blog/Andyrasika/agent-helper-langchain-hf)：未找到描述
- [blog-explorers (Blog-explorers)](https://huggingface.co/blog-explorers)：未找到描述
- [llm-course/llama_finetune/README.md at main · andysingal/llm-course](https://github.com/andysingal/llm-course/blob/main/llama_finetune/README.md)：通过在 GitHub 上创建账户为 andysingal/llm-course 的开发做出贡献。
- [Sentiment Analysis with Hugging Face and Deepgram | Deepgram](https://deepgram.com/learn/sentiment-analysis-with-hugging-face-and-deepgram)：情感分析图表提供了一种直观的信息化方式来跟踪和理解情感随时间的变化。这些数据可视化可以提供洞察...
- [no title found](https://www.artforgelabs.com/post/art-forge-labs-unveils-revolutionary-speed-and-quality-enhancements-in-ai-art-generation)：未找到描述

  

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1203280012309041173) (9 条消息🔥): 

- **伦理 LLM 框架提案**：`@lunarflu` 提出了一个关于**通用语言模型机器人伦理框架**的想法，该框架可能在全球范围内被接受，引发了发人深省的讨论。
- **CrewAI 迎来 AutoCrew**：`@_yannie_` 分享了他们的 GitHub 项目 [Autocrew](https://github.com/yanniedog/autocrew)，这是一个为 CrewAI 自动创建团队和任务的工具，配有吸引人的仓库图片和描述。
- **准备贡献的新人**：用户 `@__codenerd__` 向聊天室介绍了自己，表达了在社区展示其最佳作品的热情。
- **黑客数字助手亮相**：`@n278jm` 在 HuggingFace 上介绍了一个面向黑客的聊天助手——名为 *Your personal hacker helper*，旨在协助分析黑客需求和工具输出。
- **根据推文预测表情符号**：`@pendrokar` 改编了 `@748130998935617676` 的 TorchMoji，制作了一个 HuggingFace Space，可以根据英文文本预测后续可能出现的表情符号，使用了 2017 年的 10 亿条推文数据集，可在 [HuggingFace 上的 DeepMoji](https://huggingface.co/spaces/Pendrokar/DeepMoji) 访问。
- **Hercules-v2.0 数据集发布**：`@locutusque` 宣布发布 **Hercules-v2.0**，这是一个为专业领域模型提供支持的综合数据集，并分享了该数据集在 [Open LLM Leaderboard](https://huggingface.co/Locutusque/Hercules-2.0-Mistral-7B) 上的表现，同时包含了关于敏感内容的警告。
- **Artforge Labs 的 AI 图像生成器**：`@wubs_` 推出了 Artforge Labs，这是一项 AI 图像生成服务，提供无限次图像创建、无风险试用和月度订阅。它旨在与 MidJourney 竞争，基于 SDXL Turbo 模型，可在 [artforgelabs.com](https://artforgelabs.com) 探索。

**提到的链接**：

- [DeepMoji - a Hugging Face Space by Pendrokar](https://huggingface.co/spaces/Pendrokar/DeepMoji)：未找到描述
- [Locutusque/hercules-v2.0 · Datasets at Hugging Face](https://huggingface.co/datasets/Locutusque/hercules-v2.0)：未找到描述
- [Locutusque/Hercules-2.0-Mistral-7B · Hugging Face](https://huggingface.co/Locutusque/Hercules-2.0-Mistral-7B)：未找到描述
- [GitHub - yanniedog/Autocrew: Automatically create a crew and tasks for CrewAI](https://github.com/yanniedog/autocrew)：自动为 CrewAI 创建团队和任务。通过在 GitHub 上创建账户为 yanniedog/Autocrew 的开发做出贡献。
- [Penne Tester - HuggingChat](https://hf.co/chat/assistant/65bee444cf81c30b367f2dd7)：在 HuggingChat 中使用 Penne Tester 助手
- [HermeticCoder - HuggingChat](https://hf.co/chat/assistant/65bf5960a77e83076fb013ec)：在 HuggingChat 中使用 HermeticCoder 助手
- [Home | Art Forge Labs](https://artforgelabs.com)：生成、精炼和绘画。由 Art Forge Labs 的 AI 图像生成器驱动。Art Forge Labs

  

---


### HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1203284774526132234) (9 条消息🔥): 

- **组织 HuggingFace 活动**：`@lunarflu` 为即将举行的活动（时间待定）创建了一个占位符，并分享了链接 [加入活动](https://discord.gg/huggingface?event=1203285706949009448)。还专门创建了一个新频道，用于以结构化的方式提问。
- **建立录音分享协议**：`@chad_in_the_house` 将录制会议的 Google Drive 链接发送给了 `@811235357663297546`，表示计划将其发布在频道中，并可能上传到 YouTube。
- **会议录音的 Drive 链接**：`@lunarflu` 向小组分享了之前会议的 [Google Drive 录音](https://drive.google.com/file/d/1R6hQnEISYT8eGSDwHO-Hwi57rdtT4Q_-/view?usp=sharing)。这可能成为未来更广泛分享的 YouTube 频道的基础。

**提到的链接**：

- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.gg/huggingface?event=1203285706949009448)：Discord 是通过语音、视频和文字进行交流的最简单方式。与您的朋友和社区聊天、聚会并保持紧密联系。
- [trimmed_pres_v2.mkv](https://drive.google.com/file/d/1R6hQnEISYT8eGSDwHO-Hwi57rdtT4Q_-/view?usp=sharing)：未找到描述

  

---

### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1203481959368757248) (12 条消息🔥): 

- **慢点，快手！**：HuggingMod 提醒 `@548866140697264129` 降低消息频率：这是为了保持**聊天环境安宁**的温和提醒。
- **发出诈骗警报**：`@meatfucker` 标记了一个正在传播的经典 Discord 诈骗并建议将其删除，以此提醒 **HuggingFace 审核团队**。
- **CogVLM 的技术问题**：用户 `@furkangozukara` 就 GitHub 上详细描述的一个问题寻求帮助，特别是关于 AutoModelForCausalLM 和 Half-Char dtype 错误。发布的 [GitHub issue](https://github.com/huggingface/transformers/issues/28856) 提供了更多关于他们困境的见解。
- **了解 AI 模型的许可协议**：`@pseudoterminalx` 分享了 **Stable Video Diffusion** 模型许可协议的链接，并询问是否可以访问 Diffusers 权重，强调了 AI 研究领域的*许可证合规性*。
- **Epochs 与训练困境**：`@bitpattern` 对脚本日志中显示的漫长训练时间提出疑问，`@pseudoterminalx` 建议减少 Epochs 数量，暗示训练过程中可能存在 **Overfitting（过拟合）或低效**。

**提到的链接**：

- [stabilityai/stable-video-diffusion-img2vid-xt-1-1 at main](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main)：未找到描述
- [When using AutoModelForCausalLM, THUDM/cogagent-vqa-hf and load_in_8bit I get this error : self and mat2 must have the same dtype, but got Half and Char · Issue #28856 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28856)：系统信息 Microsoft Windows [版本 10.0.19045.3996] (c) Microsoft Corporation。保留所有权利。G:\temp Local install\CogVLM\venv\Scripts&gt;activate (venv) G:\temp Local install\CogVLM\venv\S...

  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1203334038706851890) (7 条消息): 

- **探索使用 Synthdog 生成伪造数据**：`@swetha98` 正在寻求关于在 **Donut 模型中使用 Synthdog** 创建伪造文档图像的指导，但找不到该过程所需的脚本或图像。
- **寻找最新的 One-shot 模型**：`@ursium` 询问是否有比 **CIDAS/Clipseg_Rd64_refined** 更先进的替代方案，以获得伪影更少的 Zero-shot 视觉模型，并指出这个已有一年历史的模型可能已经过时。
- **快速消息警报**：`@HuggingMod` 在频道中出现快速发布消息后，温和地提醒一位用户慢下来。
- **用于视觉 LLM 的滑动拼图数据集**：`@harsh_xx_tec_87517` 宣布发布了一个专为训练视觉 LLM 设计的滑动拼图数据集，并分享了 [Hugging Face 上的数据集](https://huggingface.co/datasets/Harshnigm/puzzles-for-vision-llm) 和用于生成此类数据集的 [GitHub 源代码](https://github.com/Harshnigam6/puzzle_llm_dataset_generation)，寻求社区反馈。
- **模型在拼图数据集上表现不佳**：针对 `@gugaime` 的提问，`@harsh_xx_tec_87517` 提到他们目前只实现了数据集生成器，ChatGPT-4 和 LLaMA 等模型无法解决该拼图，这促使他们进一步开展 LLaMA 的 Fine-tuning 工作。


**提到的链接**：

- [Harshnigm/puzzles-for-vision-llm · Datasets at Hugging Face](https://huggingface.co/datasets/Harshnigm/puzzles-for-vision-llm)：未找到描述
- [GitHub - Harshnigam6/puzzle_llm_dataset_generation: In this repo, we implement a method to generate synthetic dataset to train a vision LLM to learn how to reconstruct a puzzle.](https://github.com/Harshnigam6/puzzle_llm_dataset_generation)：在此仓库中，我们实现了一种生成合成数据集的方法，用于训练视觉 LLM 学习如何重构拼图。- GitHub - Harshnigam6/puzzle_llm_dataset_generation...

  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1203309242476658708) (9 messages🔥): 

- **寻求视频摘要方面的知识**：`@karam15.` 正在寻找与**带有时间戳的视频摘要**相关的研究论文、模型或 GitHub 仓库。请求提供相关资源的建议和参考。
- **关于在 AWS 上部署 Mistral 7B 的咨询**：`@rishit_kapoor` 正在寻求关于使用 **AWS Inferentia2 和 SageMaker** 部署 **Mistral 7B** 的教程或材料。该查询被发布了两次，表明对该话题有浓厚兴趣。
- **探索用于拼写检查和语法的模型**：`@.bexboy` 正在寻找适用于拼写检查和语法改进的**可微调模型**。征求关于有效模型或工具的指导。
- **招募模型合并（Model Merging）合作者**：`@birger6875` 邀请社区加入 **model merging** 实验，特别是针对北欧语言。他们提供了一份[规划文档](https://docs.google.com/document/d/1fP2FIrCifWcLGdTBmqeogdCdZJOwxqPfEyO-HA76_qc/edit?usp=sharing)、一份[教程](https://huggingface.co/blog/mlabonne/merge-models)、一个 [Colab notebook](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing)，并提到了一个专门用于模型合并讨论的 Discord 频道。
- **寻找贡献机会**：`@Nicks🤙🏾` 表达了对参与项目的兴趣，并正在考虑开启这段旅程的“第一步”。

**相关链接**：

[merge-crew (Merge Crew)](https://huggingface.co/merge-crew)：未找到描述

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1203481959368757248) (12 messages🔥): 

- **发帖过快提醒**：HuggingMod 警告 `@548866140697264129` 在频道中发帖速度过快，请减慢速度。
- **警惕诈骗**：`@meatfucker` 向管理员（由 <@&897381378172264449> 表示）举报了一条典型的 Discord 诈骗消息，并建议将其删除。
- **排除 GitHub Issue 故障**：`@furkangozukara` 就 GitHub 上发布的关于 `AutoModelForCausalLM` 的问题寻求帮助，引用了一个关于数据类型（Half 和 Char）的错误。用户链接了该 [Issue](https://github.com/huggingface/transformers/issues/28856)。
- **寻找 Stable Video Diffusion 的模型权重**：`@pseudoterminalx` 正在寻找 diffusers 权重，并分享了一个 Stable Video Diffusion 模型的链接，该模型需要接受许可协议才能访问。
- **优化训练时间**：`@bitpattern` 和 `@pseudoterminalx` 讨论了关于大量 epoch 导致训练时间过长的问题，`@pseudoterminalx` 建议减少 epoch 数量或调整命令行参数以优化训练计划。

**相关链接**：

- [stabilityai/stable-video-diffusion-img2vid-xt-1-1 at main](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1/tree/main)：未找到描述
- [在使用 AutoModelForCausalLM, THUDM/cogagent-vqa-hf 和 load_in_8bit 时遇到错误：self and mat2 must have the same dtype, but got Half and Char · Issue #28856 · huggingface/transformers](https://github.com/huggingface/transformers/issues/28856)：系统信息 Microsoft Windows [Version 10.0.19045.3996] (c) Microsoft Corporation. All rights reserved. G:\temp Local install\CogVLM\venv\Scripts&gt;activate (venv) G:\temp Local install\CogVLM\venv\S...

  

---

### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1203293331443028009) (77 messages🔥🔥): 

- **探索本地 LLM 以提升性能**：`@mistermatty` 讨论了 GPT-4 的缺点，包括 "Conversation key not found" 错误和表现欠佳。他们对本地 LLM 作为替代方案表现出兴趣，并收到了 `@7877` 关于使用 LM Studio 和 perplexity labs 免费使用 LLM 的建议。
  
- **本地 LLM 与 GPT-4 的基准测试对比**：`@kotykd` 认为没有开源 LLM 能与 GPT-4 相比，即使是像 Mixtral 8x7b 这样需要大量 RAM 的模型，在大多数领域也表现不如 GPT-3.5。

- **codellama-70b-instruct 的性能亮点**：`@mistermatty` 分享了他们在 perplexity labs playground 上使用 codellama-70b-instruct 的积极体验。这次互动促使他们考虑搭建一套用于本地运行同类 LLM 的配置。

- **运行 LLM 的硬件难题**：包括 `@mistermatty`、`@kotykd`、`@johnnyslanteyes` 和 `@michael_6138_97508` 在内的多位参与者深入讨论了本地运行大型 LLM 的硬件需求，涉及 RAM 与 VRAM 的重要性、系统推荐以及笔记本电脑可能存在的散热问题。

- **AI 和 LLM 信息的可靠性**：`@johnnyslanteyes`、`@michael_6138_97508` 和 `@aipythonista` 的对话表明了对各种硬件上 AI 性能信息可靠性的怀疑，强调了亲身实践以及批判性评估 YouTube 等来源的重要性。

**提到的链接**：

[Beyond Consciousness in Large Language Models: An Investigation into the Existence of a “Soul” in…](https://takk8is.medium.com/beyond-consciousness-in-large-language-models-an-investigation-into-the-existence-of-a-soul-in-83d32002c3a0): 作者：David Côrtes Cavalcante 发布日期：2024年2月3日 © CC BY Creative Commons Attribution

  

---


### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1203350765171908608) (59 messages🔥🔥): 

- **关于 @mention 问题的提醒**：`_odaenathus` 报告了在自定义 GPT 中使用 `@` 系统的问题，观察到即使是以前可以协作的 GPT 现在也不再配合，且该问题表现并不一致。
- **GPT 健忘还是坏了？**：`blckreaper` 提到他们的 GPT 遇到了困难，例如遗忘文件和突然中断故事，并对为了调试这些问题而浪费消息额度感到沮丧。
- **点赞功能的谜团**：有用户注意到点赞（thumbs up）功能消失了，`johnnyslanteyes` 澄清说，该功能仅在消息需要重新生成时才会出现，目的是告知系统新生成的回复是否更优，以便进行内容优化。
- **Prompt 和注销问题**：`rahulhere` 表示在使用 OAuth 身份验证后 GPT 无法注销，并询问为什么 "Starting Action" 需要很长时间。
- **需要 GPT 的搜索和排名功能**：`astron8272` 和 `killlemon.eth` 等用户正在寻找对 GPT 进行排名的方法，以便高效完成语言学习等特定任务，并咨询了用于市场调研的 GPT Agent，以及更便捷的 GPT Agent 搜索功能。
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1203283846720917545) (51 messages🔥): 

- **关于自定义 GPT 指令的担忧**：`@gameboy2936` 寻求关于设置自定义 GPT 指令的建议，以确保机器人能以类人化的细微差别进行交流，而不会退回到 AI 写作风格（例如使用过于华丽的辞藻）。他们的机器人应避免使用 AI 风格的短语，并保持一致的人类说话模式。
  
- **Stealth2077 遇到过度审核**：`@stealth2077` 抱怨 GPT-4 倾向于在回复中加入伦理考量，即使这在用户的叙事场景中显得不合时宜。尽管尝试指令 AI 忽略此类主题，但问题仍未解决。
  
- **Madame_Architect 强调 Assistant Model 稳定性的重要性**：`@madame_architect` 建议在稳定的 GPT 版本上使用 Assistant Model，而不是不可预测的 Preview Model，以保持指令的一致性。她指出过度审核正在影响输出质量。
  
- **避免名字拼写错误的 Prompt 精确化策略**：用户 `@titaniumsporks` 和 `@snovov` 讨论了 ChatGPT 拼错名字导致对话偏离轨道的问题，并建议提供更精确的指令以及使用合适的平台。
  
- **用户报告突然出现内容政策违规提示**：`@papa_jhon.` 对看似无害的 Prompt 却收到意外的政策违规提示表示沮丧，`@lugui` 也遇到了类似情况，并认为这可能是内部问题，稍后可能会自行恢复。

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1203283846720917545) (51 条消息🔥): 

- **为自闭症用户探索人类语言**：用户 `@gameboy2936` 寻求关于自定义 GPT 沟通风格的帮助，以创造更具人性化的交互，并分享了详细的指令示例。 

- **Assistant 与 Custom GPT 的稳定性对比**：`@madame_architect` 建议从 Custom GPT 切换到 Assistant 模型，原因是预览模型的不稳定性，以及在“不断变化的目标”上进行 prompt engineering 的挫败感。

- **创意写作受困于过度审核**：用户 `@stealth2077` 和 `@madame_architect` 讨论了在 ChatGPT 试图遵守伦理考量时，如何保持对写作的创意控制所面临的挑战，`@stealth2077` 认为这对其特定用例具有限制性。

- **故事叙述一致性的编辑技巧**：`@johnnyslanteyes` 向 `@stealth2077` 提供了一个技巧，即通过突出显示特定的故事章节来引导 ChatGPT 进行编辑，同时也解决了叙事内容中被强加价值观的问题。

- **政策违规响应与 AI 的问题**：用户 `@papa_jhon.` 和 `@lugui` 交流了针对无害提示词出现的意外政策违规响应，认为这可能是一个内部问题，后续可能会得到解决。
  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1203256672643194911) (119 条消息🔥🔥): 

- **“cuda argument error”故障排除**：`@dangfutures` 建议运行 ```sudo apt-get install libopenmpi-dev pip install mpi4py``` 以解决 RunPod 上的 GPU 问题。
- **微调的内存与策略**：关于内存需求，`@nruaif` 表示在 Llama70 上进行 LoRA 或 QLoRA 需要 80GB VRAM。他们还指出，冻结 MoE 层可以让 8 个 A6000 GPU 处理 Mixtral FFT，而使用 LoRA 进行微调的速度虽然只有一半，但样本效率更高。
- **RunPod GPU 租赁探索**：`@yamashi` 考虑投资一台机器，在闲置时租给 RunPod，而 `@casper_ai` 指出向 RunPod 出租需要大量的 GPU。`@le_mess` 建议单台机器可以使用 vast.ai。
- **组件偏好**：`@yamashi` 得出结论，LoRA 优于 QLoRA，`@nanobitz` 提到在使用 LoRA 进行微调时，选择针对 router 和 attention 进行层更新。
- **新兴的量化与模型训练方法**：`@dangfutures` 和 `@casper_ai` 讨论了使用 AWQ 和 Marlin Quant 进行训练的潜力，承认随着最近的进展这是可能的，`@casper_ai` 计划对各种量化方法的运行速度进行 benchmark。

**提到的链接**：

- [AMD + 🤗: Large Language Models Out-of-the-Box Acceleration with AMD GPU](https://huggingface.co/blog/huggingface-and-optimum-amd)：无描述
- [Kind request for updating MT-Bench leaderboards with Qwen1.5-Chat series · Issue #3009 · lm-sys/FastChat](https://github.com/lm-sys/FastChat/issues/3009)：Hi LM-Sys 团队，我们想展示 Qwen1.5-7B-Chat、Qwen1.5-14B-Chat 和 Qwen1.5-72B-Chat 在 MT-Bench 上的生成结果和自测评分。能否请您帮忙验证...
- [twitter.co - Domain Name For Sale | Dan.com](https://twitter.co)：我在 Dan.com 上发现了一个正在出售的极品域名。快来看看吧！

  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1203397418901577768) (88 条消息🔥🔥): 

- **性能攀升至新高度**：`@casper_ai` 等人讨论了 OpenBMB 的一个新型 2B 参数模型，其性能可能与 Mistral 7B 相当，并对声称的 benchmarks 表示怀疑和惊讶。在 [Notion 页面](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20) 中分享的详细探索强调了优化 model training 的重要性。

- **高级算法的实现**：`@yamashi` 赞扬了实现其词义消歧 (WSD) 算法的潜力，指出这似乎很简单，且可能比当前方法更有效。

- **Axolotl 的 Mac 兼容性查询**：`@yamashi` 讨论了在新款 M3 Mac 上运行 Axolotl 的情况，遇到了模型默认使用 CPU 而非 GPU 的问题，正在等待 torch 和 transformers 对 Mac 的 half-precision support，并提交了一个 pull request 以帮助其他有兴趣在 Mac 上运行的用户。

- **微调技术辩论**：`@casper_ai` 和 `@c.gato` 就应用大模型的训练策略进行了详细对话，特别是 MiniCPM 发现中提到的 supervised finetuning (SFT) 和不同的训练阶段，并指出这些方法可能无法直接应用于 finetuning。

- **探索数据利用最大化**：`@dreamgen` 等人对 OpenBMB 新训练策略中使用的广泛数据表示感兴趣，特别是对大模型的影响以及 fine-tuning 方法可能需要的类似实验设置。

**提到的链接**：

- [GitHub - nektos/act: Run your GitHub Actions locally 🚀](https://github.com/nektos/act)：在本地运行你的 GitHub Actions 🚀。通过在 GitHub 上创建账号为 nektos/act 的开发做出贡献。
- [Notion – The all-in-one workspace for your notes, tasks, wikis, and databases.](https://shengdinghu.notion.site/MiniCPM-Unveiling-the-Potential-of-End-side-Large-Language-Models-d4d3a8c426424654a4e80e42a711cb20)：一款将日常工作应用整合为一的新工具。它是为您和您的团队打造的一站式工作空间。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[other-llms](https://discord.com/channels/1104757954588196865/1104758057449308220/) (1 条消息): 

cf0913: https://huggingface.co/chatdb/natural-sql-7b
  

---


### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1203590582463758386) (18 条消息🔥): 

- **Batch Size 计算难题**：`@duke001.` 寻求关于如何确定训练期间每个 epoch 的 step 数量的建议，并对理论计算与 wandb 上的实际观察之间的差异感到困惑。`@nanobitz` 建议研究 sequence length packing，并提到了 `save_per_epoch` 选项以帮助按比例保存 checkpoint。

- **Finetuning 中的 EOS Token 身份危机**：`@cf0913` 遇到了一个问题，在 finetuning 后，EOS token 似乎充当了 deepseek-coder-instruct 的 pad token，导致需要手动调整。`@nanobitz` 建议修改 tokenizer config 来交换 token，`@cf0913` 确认这可以正常工作。

- **MacBook Air M1 上的 Axolotl 支持请求**：`@mini_09075` 在尝试安装 axolotl 包时遇到错误，原因是 M1 Apple 芯片缺乏 CUDA 支持。`@yamashi` 回复提到了他们的基础 PR，该 PR 可能会用 MPS 替代 CUDA，但警告称目前不建议直接使用。

- **过时分支困扰用户**：为了在本地机器上进行 Medusa training，`@mini_09075` 使用了一个过时的分支，但很快意识到这可能行不通，`@yamashi` 询问为什么要使用过时分支也暗示了这一点。

**提到的链接**：

[GitHub - ctlllll/axolotl: Go ahead and axolotl questions](https://github.com/ctlllll/axolotl.git)：Go ahead and axolotl questions。通过在 GitHub 上创建账号为 ctlllll/axolotl 的开发做出贡献。

  

---

### OpenAccess AI Collective (axolotl) ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/1203354689488552057) (7 messages): 

- **DPO 的困扰**：`@fred_fups` 报告了在运行 **differential privacy optimization (DPO)** 时遇到的困难，特别是与常规的 **Mistral qlora** 相比，在使用 **qlora** 时遇到了显存溢出 (OOM) 问题。
- **DPO 的高内存需求**：`@noobmaster29` 确认 DPO 确实消耗显著更多的内存，并指出 24GB RAM 在 **context size 为 2048** 时仅支持 **microbatch size 为 1**。
- **替代方案建议**：针对 DPO 的问题，`@dreamgen` 建议尝试 **unsloth**，因为 sample packing 在 DPO 中无法工作。
- **关于各种优化技术的咨询**：`@dreamgen` 询问是否有人有其他优化方法的经验，并列举了 **KTO, IPO** 等作为潜在的实验对象。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[runpod-help](https://discord.com/channels/1104757954588196865/1162430527215763569/1203710549947457586) (5 messages): 

- **Runpod 上 OpenBSD SSH 的初始化问题**：`@nruaif` 分享了一个日志片段，显示 OpenBSD Secure Shell 服务器已成功启动，但遇到了关于 `notebook_shim` 中缺少 `_jupyter_server_extension_points` 函数的弃用警告。
- **弃用配置警告**：`@nruaif` 提供的日志还包含一个关于 `jupyter-server 2.0` 中 `ServerApp.preferred_dir` 配置弃用的 FutureWarning，建议改用 `FileContentsManager.preferred_dir`。
- **Runpod Docker 配置错误**：同一日志提到了一个严重错误，发现 `/workspace` 位于根内容目录之外，导致初始化期间遇到错误的配置。
- **间歇性 Secure Cloud 问题**：`@dreamgen` 指出类似问题在 Secure Cloud 环境中经常发生，并建议这可能并不总是与使用 network volumes 有关。
- **对 Runpod 问题的沮丧**：`@dangfutures` 对 Runpod 表示不满，建议像社区版本之类的替代方案往往能产生更好的结果。
  

---

### CUDA MODE (Mark Saroufim) ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1203449210842775623) (25 条消息🔥): 

- **CUDA vs OpenCL 讨论**：用户 `@Voudrais` 质疑了为何更倾向于使用 CUDA 而非 OpenCL。`@andreaskoepf` 回应称 CUDA 的优势包括普及度高以及 Nvidia 的强力支持，并表示欢迎所有人加入该小组，无论其对并行编程的偏好如何。
- **Python 优于 CUDA 或 OpenCL？**：由 `@vim410` 发起的关于将 Python 作为 GPU 计算语言的对话。`@andreaskoepf` 分享了来自 [CUDA MODE GitHub 仓库](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing) 的资源列表，承认了向高级编程推进的趋势，同时也指出直接编写 Kernel 依然具有持续的相关性。
- **课程仓库重命名与整理**：`@andreaskoepf` 对 CUDA MODE 的课程内容进行了重组，将课程合并到一个现在名为 "lectures" 的 GitHub 仓库中。与 `@jeremyhoward` 的讨论涉及了旧链接重定向以及更新视频描述以适应新仓库结构的考量。
- **机器学习可视化分享**：`@latentzoo` 分享了一个与 tiny-cuda-nn 的 Fully Fused MLP 相关的可视化内容，并附带了 [推文链接](https://x.com/mallocmyheart/status/1753512787891139008?s=20)。`@andreaskoepf` 建议该图片可能来自一段关于 Tensor Cores 的 YouTube 视频，并补充说应该只分享相关的视频内容。
- **升级到新的开发机器**：`@andreaskoepf` 发起了关于升级到新开发机器的讨论，考虑到像 Lambda 工作站这类预装机的高昂成本，他考虑逐个零件缓慢组装。该帖子开启了社区对系统构建感兴趣的可能性。

**提到的链接**：

- [GitHub - cuda-mode/resource-stream: CUDA 相关新闻和材料链接](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#python-gpu-computing)：CUDA 相关新闻和材料链接。通过在 GitHub 上创建账号为 cuda-mode/resource-stream 的开发做出贡献。
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2)：lecture 2 - 2024-01-20。通过在 GitHub 上创建账号为 cuda-mode/lecture2 的开发做出贡献。
- [GitHub - cuda-mode/profiling-cuda-in-torch](https://github.com/cuda-mode/profiling-cuda-in-torch)：通过在 GitHub 上创建账号为 cuda-mode/profiling-cuda-in-torch 的开发做出贡献。
- [Hayden (@mallocmyheart) 的推文](https://x.com/mallocmyheart/status/1753512787891139008?s=20)：刚刚理解了这里实际发生了什么，以及它与 tiny-cuda-nn 的 Fully Fused MLP 的关系。
- [自定义您的 Lambda Vector | Lambda](https://shop.lambdalabs.com/gpu-workstations/vector/customize)：未找到描述。
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2/?tab=readme-ov-file)：lecture 2 - 2024-01-20。通过在 GitHub 上创建账号为 cuda-mode/lecture2 的开发做出贡献。
- [GitHub - cuda-mode/lectures: CUDA MODE 课程材料](https://github.com/cuda-mode/lectures)：CUDA MODE 课程材料。通过在 GitHub 上创建账号为 cuda-mode/lectures 的开发做出贡献。
- [Tensor Cores 简述](https://youtu.be/yyR0ZoCeBO8?si=_PTd7rVNgjokgQC9&t=20)：这段视频简要介绍了 NVIDIA GPU 内部的 Tensor Core 技术，以及它对最大化深度学习性能的重要性。

---

### CUDA MODE (Mark Saroufim) ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1203259271224234004) (99 条消息🔥🔥): 

- **灰度转换速度与精度的发现**：`@artste` 尝试了多种将 RGB 转换为灰度的方法，发现整数运算虽然快但不精确。通过使用 float 查找表（lookup table）实现了速度与精度之间的最佳平衡，在达到与基准测试几乎相同结果的同时，速度快了约 2.8 倍（[记录实验详情的 Notebook](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb)）。

- **批处理提升性能**：当通过水平堆叠图像来模拟 Batch 时，`@artste` 发现对于 Batch Size 为 16 的图像，所进行的优化使得灰度转换过程比未优化的情况快了高达 3.98 倍。

- **编译器细节对 CUDA 性能产生显著影响**：`@andreaskoepf` 等人的讨论表明，看似微小的改动（例如添加 'f' 来表示 32-bit float）会极大地影响 GPU 上的运行时间，这强调了 GPU 优化的复杂性。

- **成员间共享的 CUDA 工具和仓库**：成员们分享了各种资源和工具，例如 Godbolt（一个 CUDA 在线编译器/探索器），以及多个 GitHub 仓库——包括用于快速神经网络框架的 [`tiny-cuda-nn`](https://github.com/NVlabs/tiny-cuda-nn)，以及来自 [`cuda-mode/lectures`](https://github.com/cuda-mode/lectures) 的 CUDA 讲座——这些资源促进了 CUDA 的学习和实验。

- **使用 PyTorch 学习和调试 CUDA**：`@edd0302` 寻求关于管理 CUDA 和 PyTorch 项目的建议，而 `@jeremyhoward` 等人讨论了使用 PyTorch 编译 CUDA 代码的特性，强调了诸如强制重新编译等挑战，以及 PyTorch 开发团队可能考虑的改进（`@marksaroufim` 表示愿意接受改进建议）。

**提到的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1189498204333543425/1189607726595194971/1202057241868640308): Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [Tensor Core 分析](https://youtu.be/xjjN9q2ym6s): 一段分析 Nvidia Volta Tensor Core 架构组成的视频。参考资料：Pu, J., et. al. "FPMax: a 106GFLOPS/W at 217GFLOPS/mm2 Single-Precision ...
- [Hayden (@mallocmyheart) 的推文](https://x.com/mallocmyheart/status/1753512787891139008): 刚刚理解了这里实际发生的事情，以及它与 tiny-cuda-nn 的全融合 MLP (fully fused mlp) 的关系。
- [lecture2/lecture3/cuda_rgb_to_gray_refactor.ipynb at cuda_rgb_to_gray_refactor_notebook · artste/lecture2](https://github.com/artste/lecture2/blob/cuda_rgb_to_gray_refactor_notebook/lecture3/cuda_rgb_to_gray_refactor.ipynb): 第 2 课 - 2024-01-20。通过在 GitHub 上创建账号为 artste/lecture2 的开发做出贡献。
- [flash-attention/flash_attn at main · Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention/tree/main/flash_attn): 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号为 Dao-AILab/flash-attention 的开发做出贡献。
- [GitHub - cuda-mode/lecture2: lecture 2 - 2024-01-20](https://github.com/cuda-mode/lecture2/): 第 2 课 - 2024-01-20。通过在 GitHub 上创建账号为 cuda-mode/lecture2 的开发做出贡献。
- [GitHub - NVlabs/tiny-cuda-nn: Lightning fast C++/CUDA neural network framework](https://github.com/NVlabs/tiny-cuda-nn): 极速 C++/CUDA 神经网络框架。通过在 GitHub 上创建账号为 NVlabs/tiny-cuda-nn 的开发做出贡献。
- [torch.utils.cpp_extension — PyTorch 2.2 文档](https://pytorch.org/docs/stable/cpp_extension.html): 未找到描述
- [GitHub - ROCm/flash-attention: Fast and memory-efficient exact attention](https://github.com/ROCm/flash-attention): 快速且内存高效的精确 Attention。通过在 GitHub 上创建账号为 ROCm/flash-attention 的开发做出贡献。
- [Compiler Explorer - CUDA C++ (NVCC 12.3.1)](https://godbolt.org/z/odb3191vK): #include &lt;stdint.h&gt; // 在此处输入代码，或加载示例。__global__ void square(uint32_t* v, size_t vn, uint32_t* r) { auto tid = blockDim.x * blockIdx.x + threadIdx.x; i...
- [具有多分辨率哈希编码的即时神经图形基元 (Instant Neural Graphics Primitives)](https://arxiv.org/abs/2201.05989): 由全连接神经网络参数化的神经图形基元在训练和评估时可能成本很高。我们通过一种通用的新型输入编码降低了这一成本，该编码允许使用较小的...
- [Compiler Explorer](https://godbolt.org/): 未找到描述
- [GitHub - cuda-mode/lectures: Material for cuda-mode lectures](https://github.com/cuda-mode/lectures/): cuda-mode 讲座材料。通过在 GitHub 上创建账号为 cuda-mode/lectures 的开发做出贡献。

---

### CUDA MODE (Mark Saroufim) ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1203834178269880361) (5 messages): 

- **快速高效的 PyTorch 代码技巧**：`@tantara` 分享了来自 **gpt-fast** 仓库的 [PyTorch 代码片段链接](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314)，并建议在使用 `torch.compile` API 时，指定编译层可能会有所帮助。
- **Torch 编译器细粒度控制揭秘**：`@marksaroufim` 提供了额外的见解，提到了 `torch.compiler.disable()` 的使用，并推荐了关于控制 `torch.compile` 的 PyTorch [细粒度 API 文档](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html)。

- **TensorFlow：替代方案之争**：`@Voudrais` 幽默地建议使用 TensorFlow 代替 PyTorch，这促使 `@andreaskoepf` 创建了一个专门讨论 TensorFlow 的频道。他承认 Google 的加速器资源和极具竞争力的价格优势，但也提醒要注意平台锁定（lock-in）问题。

**提到的链接**：

- [gpt-fast/generate.py at main · pytorch-labs/gpt-fast](https://github.com/pytorch-labs/gpt-fast/blob/main/generate.py#L307-L314)：在少于 1000 行 Python 代码中实现简单且高效的 PyTorch 原生 Transformer 文本生成。- pytorch-labs/gpt-fast
- [TorchDynamo APIs for fine-grained tracing &mdash; PyTorch main documentation](https://pytorch.org/docs/main/torch.compiler_fine_grain_apis.html)：未找到描述

  

---


### CUDA MODE (Mark Saroufim) ▷ #[announcements](https://discord.com/channels/1189498204333543425/1189640399476764692/1203420809482338415) (2 messages): 

- **新的 CUDA 课程即将到来**：`@andreaskoepf` 宣布 **CUDA MODE - 第 4 课：计算与内存架构简介** 即将开始，重点关注 PMPP 书籍的第 4 和第 5 章，涵盖 blocks、warps 和内存层级。

- **课程笔记已发布**：`@tvi_` 提到，包含第 4 和第 5 章讨论的即将到来的课程笔记可以在仓库中找到，他幽默地称该仓库为“命名越来越不准确的 lecture2 仓库”。
  

---


### CUDA MODE (Mark Saroufim) ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1203267016748826624) (2 messages): 

- **Aleph Alpha 正在寻找 CUDA 高手**：`@piotr.mazurek` 分享了一个招聘职位，**Aleph Alpha** 正在为其产品团队招聘资深专业人士。具体而言，该角色涉及将语言模型的研究转化为实际应用，影响财富 2000 强公司和政府，职位详情见[此处](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en)。

- **Mistral AI 招募 GPU 魔法师**：`@megaserg.` 强调了 **Mistral AI** 的一个机会，正在寻找 **在 GPU 上进行大语言模型推理（serving）和训练** 的专家。该职位涉及编写自定义 CUDA kernels，并最大限度地发挥 H100 等高端 GPU 的潜力，职位发布在[此处](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48)。

**提到的链接**：

- [AI Engineer - Large Language Models (m/f/d) | Jobs at Aleph Alpha GmbH](https://alephalpha.jobs.personio.de/job/1329474?language=en&display=en)：Aleph Alpha 成立于 2019 年，其使命是为强 AI 时代研究和构建基础技术。其国际科学家、工程师和创新者团队致力于研究……
- [Mistral AI - GPU programming expert](https://jobs.lever.co/mistral/399978d0-b442-4591-b677-8cc03ee24a48)：Mistral AI 正在招聘一名专家，负责在 GPU 上高速进行大语言模型的推理和训练。该角色将涉及编写底层代码，以充分利用高端 GPU（H...

  

---

### CUDA MODE (Mark Saroufim) ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1203390140077314068) (10 条消息🔥): 

- **C++ 与 C 在 CUDA 中的对比**：`@evil_malloc` 询问 C++ 是否是学习 CUDA/Triton 的先决条件，`@_tvi_` 回复称精通 C++ 并非必须，但具备一定的熟悉度是有益的，特别是在将 CUDA 与 PyTorch 结合使用时。

- **寻求掌握 C++**：`@umaiskhan` 寻求关于有效学习 C++ 的建议，`@stefangliga` 推荐了 [LearnCpp.com](https://www.learncpp.com/)，这是一个提供大量教程和示例的免费资源。

- **Rust 在 CUDA 编程中的现状**：`@greystark.` 询问了目前 Rust 对 CUDA 编程的支持情况，`@andreaskoepf` 指出目前缺乏活跃项目，但分享了用于 Rust GPU 着色器（shaders）的 [rust-gpu](https://github.com/embarkstudios/rust-gpu) 仓库。

- **探索 Rust 与 CUDA**：`@andreaskoepf` 进一步向 `@greystark.` 及其他感兴趣的人推荐了 [Rust-CUDA](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html)，该项目提供了使用 Rust 编写支持 CUDA 的 GPU crate 的指南。

- **Rust 神经网络开发**：`@andreaskoepf` 随后分享了更多与利用 CUDA 进行神经网络开发相关的活跃 Rust 仓库，提到了 [Kyanite](https://github.com/KarelPeeters/Kyanite) 和 [burn](https://github.com/tracel-ai/burn) 等值得探索的项目。

**提到的链接**：

- [Getting Started - GPU Computing with Rust using CUDA](https://rust-gpu.github.io/Rust-CUDA/guide/getting_started.html)：未找到描述
- [Learn C++ – Skill up with our free tutorials](https://www.learncpp.com/)：未找到描述
- [GitHub - EmbarkStudios/rust-gpu: 🐉 Making Rust a first-class language and ecosystem for GPU shaders 🚧](https://github.com/embarkstudios/rust-gpu)：🐉 使 Rust 成为 GPU 着色器的一流语言和生态系统 🚧 - GitHub - EmbarkStudios/rust-gpu
- [GitHub - KarelPeeters/Kyanite](https://github.com/KarelPeeters/Kyanite)：通过在 GitHub 上创建账号为 KarelPeeters/Kyanite 的开发做出贡献。
- [GitHub - tracel-ai/burn: Burn is a new comprehensive dynamic Deep Learning Framework built using Rust with extreme flexibility, compute efficiency and portability as its primary goals.](https://github.com/tracel-ai/burn)：Burn 是一个使用 Rust 构建的全新综合性动态深度学习框架，其主要目标是极高的灵活性、计算效率和可移植性。

  

---


### CUDA MODE (Mark Saroufim) ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1203788105916612649) (2 条消息): 

- **矩阵乘法耗时对比**：`@antoniooooooooooooo` 分享了他们 1024x1024 矩阵乘法的**计时结果**：**CPU 为 5,308,033μs**，**GPU 原始版本为 131,237μs**，**GPU 行优先版本为 43,896μs**，以及 **GPU 列优先版本为 32,179μs**。他们询问这些耗时之间的关系是否合理。

- **寻求理论与代码解答**：`@antoniooooooooooooo` 询问是否有关于 PMPP 书中练习题的**更多理论解答**资源以及包含**代码实现**的仓库。
  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1203343936584814612) (88 条消息🔥🔥): 

- **模型训练疑问与误解**：围绕 **TimesFM 模型训练** 的讨论由 `@Hawk` 和 `@mrgonao` 等用户进行了澄清。Hawk 最初质疑了训练过程的效率，随后提供了一个修正后的序列（`input:1-32 -> output 33-160 -> input 1-160 -> output 161-288`），并根据模型描述得出结论：输出 patch 之间不应存在重叠。

- **寻求关于长上下文 LLM 的见解**：用户 `@nsk7153` 询问了关于处理 LLM 长上下文的研究，`@stellaathena` 分享了一篇 [YaRN 论文](https://arxiv.org/abs/2309.00071) 作为回应，这是一种扩展上下文窗口长度的计算高效方法。

- **一种新颖训练方法的提议**：用户 `@worthlesshobo` 发起了关于自动编码（autoencoding）的深入讨论，并提出了一种被称为“liturgical refinement”的方法。他们建议采用交替冻结和解冻 Encoder-Decoder 模型组件的技术，以期获得更有效的表示。

- **关于模型融合与约束的想法**：用户 `@win100` 推测通过融合来自不同模型（A 和 B）的模型张量来改进预训练，这与 FuseLLM 项目的概念一致。`@!BeastBlaze` 提供了关于 LeMDA 论文中采取的相关方法的见解，该方法专注于增强特征嵌入（feature embeddings）。

- **LLM Web UI 的开发**：用户 `@318yang_` 宣布开发并部署了一个用于大语言模型（LLM）的 Web UI：[simple-ai.io](https://simple-ai.io)，这是一个开源项目，社区可以在自己的项目中使用。他们还提到计划将 Ollma 集成到这个新 UI 中以支持本地运行。

**提到的链接**：

- [未找到标题](https://simpel-ai.io~~)：未找到描述
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)：旋转位置嵌入（RoPE）已被证明能有效编码基于 Transformer 的语言模型中的位置信息。然而，这些模型在超过序列长度 t 时无法泛化...
- [来自 Valeriy M., PhD, MBA, CQF (@predict_addict) 的推文](https://x.com/predict_addict/status/1754134502895460421?s=20)：谷歌的一篇推销时间序列预测“基础模型”的新论文，既是初学者错误的典型，又采用了具有误导性的“基准测试”。在图 6 中，作者...
- [处理长上下文提示的大语言模型 | Semantic Scholar](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20large%20context%20prompts&sort=relevance)：一个利用人工智能方法提供高度相关结果和便捷筛选工具的学术搜索引擎。
- [simple ai - chat](https://simple-ai.io)：未找到描述
- [Yes I Am Gru GIF - Yes I Am Gru Steve Carell - 发现并分享 GIF](https://tenor.com/view/yes-i-am-gru-steve-carell-despicable-me2-yes-thats-me-gif-16561733)：点击查看 GIF
- [Neuronpedia](https://www.neuronpedia.org/)：AI 安全游戏与开放数据参考
- [Google Colaboratory](https://colab.research.google.com/drive/1ET3R_JkckEJ-LxJpd05PjbfGK-TyiPNF#scrollTo=RYadyNjP1-UC)：未找到描述

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1203277277354393680) (32 messages🔥): 

- **MoE-Mamba 表现优于同类模型**：在关于[最近一篇论文](https://arxiv.org/abs/2401.04081)的讨论中，`@afcruzs` 分享了一个 Arxiv 链接，介绍了一种名为 "MoE-Mamba" 的 SSM 模型，该模型在更少的训练步数下表现优于多种 state-of-the-art 模型。
- **力求专家模型的均衡**：`@catboy_slim_` 暗示 Mixture of Experts (MoE) 模型可能由于层分配不平衡而导致效率损失，并建议考虑增加额外的 router loss 来恢复平衡。
- **计算概念与折纸结合**：`@digthatdata` 发布了一篇 [Quantamagazine 文章](https://www.quantamagazine.org/how-to-build-an-origami-computer-20240130/)，将计算与折纸联系起来，引发了关于其与 AI 中 in-context learning 潜在联系的简短交流。
- **获取 Mamba Checkpoints**：在讨论 SSM 可能存在的问题后，`@woog` 表示有兴趣获取 Mamba 模型的 checkpoints，`@random_string_of_character` 指出这些可以通过请求获得，也可以在 [Hugging Face's Model Hub](https://huggingface.co/Zyphra/BlackMamba-1.5B) 上找到。
- **梯度稳定化与 Encodec**：围绕梯度稳定化展开了对话，`@nostalgiahurts` 引用了 [Encodec 论文的方法](https://arxiv.org/abs/2210.13438) 来处理多种损失类型，该方法引入了一种归一化机制，用于在训练期间平衡梯度。

**提到的链接**：

- [How to Build an Origami Computer | Quanta Magazine](https://www.quantamagazine.org/how-to-build-an-origami-computer-20240130/)：两位数学家证明，折纸在原则上可以用于执行任何可能的计算。
- [来自 Guillaume Bellec (@BellecGuill) 的推文](https://x.com/BellecGuill/status/1750814799615725793)：@francoisfleuret 这个微型 PyTorch 库正是为此而生。https://github.com/guillaumeBellec/multitask 最初是为了避免在微调辅助损失系数上浪费时间。对于你的情况，只需...
- [MoE-Mamba: Efficient Selective State Space Models with Mixture of Experts](https://arxiv.org/abs/2401.04081)：State Space Models (SSMs) 已成为序列建模领域的有力竞争者，挑战了 Transformers 的主导地位。与此同时，Mixture of Experts (MoE) 显著提高了...
- [Repeat After Me: Transformers are Better than State Space Models at Copying](http://arxiv.org/abs/2402.01032)：Transformers 是序列建模的主导架构，但人们对使用不依赖于序列长度的固定大小 latent state 的模型（我们称之为...）越来越感兴趣。
- [In-context Reinforcement Learning with Algorithm Distillation](https://arxiv.org/abs/2210.14215)：我们提出了 Algorithm Distillation (AD)，这是一种通过使用因果序列模型建模训练历史，将 Reinforcement Learning (RL) 算法蒸馏到神经网络中的方法。Algorithm...
- [High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438)：我们介绍了一种利用神经网络的最先进的实时、高保真音频编解码器。它由一个流式 encoder-decoder 架构组成，具有量化的 latent space，并以端到端的方式进行训练...
- [来自 Quentin Anthony (@QuentinAnthon15) 的推文](https://fixupx.com/QuentinAnthon15/status/1753584827100778965)：从可解释性的角度来看，如果你想研究 Mamba，我们在我们的数据集上训练了一个纯 Mamba-350M。我们还在 Pile 数据集上训练了原始的 Mamba-370M！所有 checkpoints 和数据集均可用...
- [GitHub - Zyphra/BlackMamba: Code repository for Black Mamba](https://github.com/Zyphra/BlackMamba)：Black Mamba 的代码仓库。通过在 GitHub 上创建账户为 Zyphra/BlackMamba 的开发做出贡献。

  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1203818349914619954) (1 messages): 

- **澄清可解释性中的 "Direction" 与 "Feature"**：`@pinconefish` 提出了一个关于可解释性术语的问题，询问 "direction" 是否指代 embedding space 中编码 monosemantic 含义的向量。他们指出，"direction" 可能有助于区分单个神经元的激活（也称为 "feature"）与 embedding space 中的向量，这在讨论模型中不同层级的语义含义时非常有用。
  

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1203792742321233981) (5 messages): 

- **日程已锁定**：`@asuglia` 与 `@981242445696221224` 及 `@1072629185346019358` 确认，**周二 6 日下午 5 点（英国时间）**是一个合适的会议时段。
- **准备邀请**：`@hailey_schoelkopf` 同意了提议的会议时间，并请求通过私信（DMs）获取邮箱地址，以便向 `@asuglia` 和 `@1072629185346019358` 发送邀请。
- **讨论大规模测试策略**：`@mistobaan` 询问了大规模测试的方法，提到了使用带有 worker 的队列或长时间单机运行等选项。
- **使用 Slurm 进行扩展测试**：针对 `@mistobaan` 的提问，`@.johnnysands` 提到他们通过排队大量作业来利用 **Slurm** 管理规模测试。
- **探索 Prompt 预览**：`@Goyim` 寻求关于预览特定任务 Prompt 的可能性，以及提交给模型的 **multiple_choice** Prompt 格式的见解。
  

---


### Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1203380953758834759) (9 messages🔥): 

- **探索多模态方法的 MoE**：`@martianulcrizat` 对创建多模态系统的 Mixture of Experts (MoE) 模型指导表现出兴趣，暗示将 Transformer 扩散模型与 LLaMA 等 VLM（视觉语言模型）集成。
- **寻求更深层的语义与生成集成**：`@martianulcrizat` 讨论了通过采用 MoE 框架，在 VLM 内部实现语义理解与生成能力之间更紧密集成的潜力。
- **寻找 VLM 与扩散模型的结合技术**：`@martianulcrizat` 询问了除了涉及 QFormer、Adaptor 层和带有连续 token 表示的 cross-attention 等传统方法之外，将 VLM 与扩散模型结合的其他方法。
- **认可分享的集成方法论文**：`!BeastBlaze` 认可了 `@martianulcrizat` 分享的论文的相关性，这些论文可能有助于 VLM 与扩散模型的集成。
- **结合 VLM 与扩散模型的替代简化方案**：`!BeastBlaze` 提到了新的文献（虽然目前不易获得），这些文献表明使用简单的 RNN 和 CBOW 即可实现与 CLIP 等大型模型类似的结果，从而实现像 fast DINO 或 fast SAM 这样更精简的方法。
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1203809119258021928) (2 messages): 

- **澄清 "gas" 参数功能**：`@catboy_slim_` 注意到一个 [pull request](https://github.com/EleutherAI/gpt-neox/pull/123) 旨在删除 `"gas"` 参数，并指出该参数已失效且与 `"gradient_accumulation_steps"` 重复。他们警告说，过去使用非 1 的 `"gas"` 值的运行，其有效 Batch size 可能比预期的要小。
- **即将对 "gas" 参数进行审查**：`@tastybucketofrice` 回复称，他们今天将审查有关 `"gas"` 参数的问题。
  

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1203260159967756328) (96 条消息🔥🔥): 

- **语言障碍？没问题！**：`@bondesign` 询问是否可以使用中文输入，随后的一条中文消息也表达了同样的兴趣。`@cookie_74700` 同样询问了是否可以说波斯语，`@mares1317` 提供的链接回复强调了 Perplexity 的多语言能力。

- **关于 Copilot 作用的困惑**：`@oudstand` 分享了关于使用 Copilot 似乎能提升模型性能的观察。与此同时，`@dlysltrading` 等人遇到了 Copilot 的问题，通过刷新网页得以解决；`@stocktown` 则引发了关于在开启 Copilot 时使用 write mode 的合理性的讨论。

- **处理 Perplexity 的客户服务问题**：`@aqbalsingh` 对更改账户邮箱以及 iPhone 应用上缺少上传按钮表示沮丧，并因此取消了其 premium 账户。尽管得到了 Perplexity 的回复，该用户仍对支持服务的响应速度感到失望。

- **Perplexity AI Discord 集成之苦**：`@otchuda` 哀叹缺少 Discord 集成以获得快速的 Perplexity 响应，这引发了讨论。`@icelavaman` 和 `@ok.alex` 分享了相关链接，但确认目前没有计划将文本形式的 Perplexity 机器人重新带回 Discord。

- **探索 API keys 和使用场景**：`@elanutta` 询问了关于使用 Perplexity 账户生成 OpenAI 的 API keys 的问题，而 `@glisteningsunlight` 报告并自行解决了在尝试让 Perplexity 总结 PDF 时的延迟问题。此外，`@felirami`、`@general3d` 和 `@maverix.` 还就使用配额以及 ChatGPT Plus 与 Perplexity Pro 之间的产品对比展开了讨论。

**提到的链接**：

- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1118264005207793674/1203626274132590612)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047204950763122820/1175478858816950343)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1111786888626438245/1193465802259709992)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649527299055688/1197892547276705843)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649527299055688/1202651487747047515)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1047649619695390740/1202652559463153796)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [Discord - 与好友及社区聊天的新方式](https://discord.com/channels/1047197230748151888/1183781616515031124/1184427097272365106)：Discord 是通过语音、视频和文字进行交流的最简单方式。在这里聊天、聚会，与你的好友和社区保持紧密联系。
- [什么是 Perplexity Copilot？](https://blog.perplexity.ai/faq/what-is-copilot)：浏览 Perplexity 博客，获取文章、公告、产品更新以及优化体验的技巧。随时掌握最新动态，充分利用 Perplexity。
- [图像与媒体](https://blog.perplexity.ai/faq/images-media)：浏览 Perplexity 博客，获取文章、公告、产品更新以及优化体验的技巧。随时掌握最新动态，充分利用 Perplexity。

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1203317593008767010) (14 条消息🔥): 

- **突出 Perplexity AI 的视频**：`@arunprakash_`、`@boles.ai` 和 `@ok.alex` 都分享了展示 Perplexity AI 的 YouTube 视频，讨论了其优势和功能，并解释了用户为何可能选择它而非其他 AI 选项。视频标题分别为《我们真的需要 Perplexity AI Pro 订阅吗？》、《Perplexity 和 Play.HT 不容小觑！外加 Hindenburg 评测！》以及《我为了 PERPLEXITY 3.0 放弃了 BARD、ChatGPT 和 CLAUDE！》，可以分别在这些 [YouTube 链接](https://www.youtube.com/watch?v=eqSkH_p8CZ8)、[Play.HT 评测 YouTube 链接](https://youtube.com/live/LKQTETpxS_8?feature=share) 和 [Perplexity 3.0 评测 YouTube 链接](https://youtu.be/mFfS4BYCCgw?si=ysvRx4_yU5FxU0Qe) 中找到。

- **分享启发性的 Perplexity 搜索**：用户 `@rocktownarky`、`@bwatkins`、`@maverix.`、`@gamezonebull`、`@epic9713` 和 `@darkspider1987` 分享了他们 Perplexity AI 搜索结果的直接链接，这些结果提供了有价值的见解，促使了 Pro 订阅决策，并帮助进行了复杂的决策。分享的结果可以通过提供的 Perplexity AI 搜索[链接](https://www.perplexity.ai/search/46ad8505-d3e4-4cc1-ab64-624b18a9affc)、[Prop 1 决策链接](https://www.perplexity.ai/search/Help-me-decide-UEJrUP.XROOH.V5_3HU1wQ?s=c)、[“what is”链接](https://www.perplexity.ai/search/what-is-the-DOn_dKY5RTuDw.SjxeoP8Q)、[如何获取链接](https://www.perplexity.ai/search/How-to-get-pOdKd44XSh6F3wyxadfBbQ?s=u) 以及 [AI 图像运动链接](https://www.perplexity.ai/search/AI-image-motion-RVRGLZHARw2nQBkJuA12Eg?s=c#c039b3c1-2cf2-444a-a523-5f251ffe61c8) 访问。

- **公开搜索分享技巧**：`@me.lk` 建议 `@maverix.` 和 `@gamezonebull` 通过点击右上角的分享按钮来确保他们的搜索线程是公开的，而 `@noremac258` 注意到 `@darkspider1987` 的搜索结果无法查看，这表明了将搜索设为公开以便社区分享的重要性。

- **无描述的重定向链接**：`@ok.alex` 发布了一个 [Discord 重定向链接](https://discord.com/channels/1047197230748151888/1054944216876331118/1203759744217653278)，但未对其指向的内容提供说明。

- **使用 Perplexity 度过的高效周末**：`@johnweisenfeld` 分享了一篇 [LinkedIn 帖子](https://www.linkedin.com/posts/weisenfeldj_about-pasco-high-school-activity-7160105805193641984-zMSr?utm_source=share&utm_medium=member_desktop)，讲述了得益于 Perplexity 的高效周末，并提到了使用其他 AI 服务时的困难，同时称赞 OpenAI 帮助启动了一个代码项目。

**提及的链接**：

- [Discord - 与朋友和社区聊天的新方式](https://discord.com/channels/1047197230748151888/1054944216876331118/1203759744217653278)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，与你的朋友和社区保持紧密联系。
- [🔥 我为了 PERPLEXITY 3.0 放弃了 BARD、ChatGPT 和 CLAUDE！😱 #震撼](https://youtu.be/mFfS4BYCCgw?si=ysvRx4_yU5FxU0Qe)：在这段视频中，我将展示 Perplexity AI 的最新功能。🚨 巨额折扣、免费培训和我的课程：https://wasuniverse.com ✅ 获取免费...
- [我们真的需要 Perplexity AI Pro 订阅吗？](https://www.youtube.com/watch?v=eqSkH_p8CZ8)：在这段视频中探索 Perplexity AI 的创新功能！发现其集成的 Reddit 和 YouTube 搜索、专注搜索模式、来源排除、AI 比较...
- [Perplexity 和 Play.HT 不容小觑！外加 Hindenburg 评测！](https://youtube.com/live/LKQTETpxS_8?feature=share)：我们关注 AI 搜索、AI 语音以及更传统的录音方式。此外，还有 Apple Podcast 转录功能！

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1203350676214784010) (7 messages): 

- **Mixtral 定价咨询**：`@paul16307` 询问了 Mixtral 的定价，因为 Perplexity 从 API 定价中移除了 13b。`@icelavaman` 回复了当前费率：**每 1M input tokens $0.14，每 1M output tokens $0.56**。
- **潜在的 Rate Limit 提升**：`@aiistheonlyway` 询问如何快速提升 Rate Limit，但在提供的消息中未收到回复。
- **对 Mixtral 未来定价的好奇**：在了解定价详情后，`@paul16307` 询问 Mixtral 未来的定价是否会更低，但未收到回复。
- **请求 pplx-web API 版本**：`@makadoro_95229` 建议 Perplexity 提供 **pplx-web 版本**的 API，这将提供类似于网站的搜索结果，有助于为其他网站创建聊天助手（chat assistants）。`@defektivex` 支持这一想法，提到许多人也请求过此功能，并对未来推出类似的 API 表示期待。
- **Perplexity AI 与 Siri 的集成**：`@out_a_time6794` 询问如何设置 Perplexity AI 与 Siri 配合使用，作为查询的快捷指令（shortcut），对话中没有后续回复。
  

---



### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1203313070848614440) (34 messages🔥): 

- **寻求适用于阿拉伯语内容的合适技术**：`@mukhtor` 咨询了用于阿拉伯语内容聊天的技术。`@lhc1921` 建议使用阿拉伯语 LLM 和 embeddings，并提到大多数技术是语言无关的（language-agnostic），而 `@hro_ffs_why_cant_i_use_my_name` 提到 [embedding-ada](https://www.sbert.net/examples/training/multilingual/README.html) 主要支持“法语、英语、德语、西班牙语和葡萄牙语”，但列出了 *aravec* 和 *word2vec* 作为潜在的替代方案。
  
- **自主 Agent 的高性价比托管**：`@charly8323` 寻求关于托管高性价比自主研究 Agent 的建议，希望将成本控制在每次调用 5 美分的收入结构以下。`@engineered.mind` 推荐使用本地 LLM 以控制成本，`@truethinker` 建议在 DigitalOcean 等服务器上部署 [ollama](https://ollama.com/)。

- **新书预告**：`@mehulgupta7991` 宣布发布新书《LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs》，详细介绍了如何使用 LangChain 构建各种应用，该书已在 [Amazon](https://amzn.eu/d/dqQJzV1) 上架。

- **使用 LangChain 进行长文档翻译的效率技巧**：`@o3omoomin` 询问了使用 LangChain 翻译长文档的高效方法，以避免 token 限制。该用户正在探索将其分割成更小的 chunks，并寻求更流线化处理的示例代码。

- **托管与 Fine-Tuning 挑战讨论**：多位成员（包括 `@lhc1921`、`@nrs` 和 `@sullynaj`）讨论了在云端托管和 Fine-Tuning 模型的挑战。建议包括使用本地模型、Google Colab 和 Cohere embeddings，以及使用相关数据集训练阿拉伯语模型的潜在策略。

**提到的链接**：

- [no title found](https://amzn.eu/d/dqQJzV1)：未找到描述
- [GitHub - BBC-Esq/Nvidia_Gpu_Monitor: Realtime Monitor of Nvidia GPU Metrics with NVML Library](https://github.com/BBC-Esq/Nvidia_Gpu_Monitor)：使用 NVML 库实时监控 Nvidia GPU 指标
- [GitHub - facebookresearch/contriever: Contriever: Unsupervised Dense Information Retrieval with Contrastive Learning](https://github.com/facebookresearch/contriever)：Contriever：使用对比学习的无监督密集信息检索

  

---

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1203322976418930748) (7 条消息): 

- **介绍用于交互式播客体验的 CastMate**：`@darrelladjei` 分享了 **CastMate** 的发布，这是一个允许用户收听并与他们喜爱的播客节目互动的平台，其特点是使用 LLM 和人类质量的 TTS 生成内容。他们提供了一个 [Loom 演示](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74)并征求反馈，同时还提供了 **iPhone beta** 版：[TestFlight 链接](https://testflight.apple.com/join/9khwQ1vD)。

- **关于 Artificial Agents 的 GUI 讨论**：用户 `@clickclack777` 询问正在使用哪种 GUI，这促使 `@robot3yes` 提到了他们在 **Agent IX** 上的工作，这是一个旨在与机器人交互的侧边项目。

- **新书预告：生成式 AI 应用指南**：`@mehulgupta7991` 宣布了他们的处女作，《*LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs*》，该书涵盖了涉及 LangChain 从基础到高级的各种用例。他们分享了一个失效的 Amazon [链接](https://amzn.eu/d/dqQJzV1)，该链接显示的是验证码验证页面。

- **认识作者与数据科学家**：在后续消息中，`@mehulgupta7991` 介绍自己是一名曾在 DBS Bank 任职的数据科学家，并分享了他们的 "Data Science in your Pocket" Medium 和 YouTube 频道。他们特别指向了一个 LangChain [YouTube 播放列表](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno)以获取教程。

- **寻求目标设定助手的建议**：`@mark_c_` 征求关于创建目标设定助手的架构建议，该助手用于管理长期和短期目标并协助每周调度，并提到了他们作为前程序员的背景。他们有兴趣从 Prompt Engineering 开始，但预见到需要更复杂的 Workflow。

- **用于 AI 增强投资尽职调查的工具**：`@solo78` 介绍了一个利用 LangChain 对平台和公司进行深度尽职调查的投资工具项目。他们分享了详细介绍项目历程的 Medium 博客文章，并寻求社区的意见：[Medium 文章](https://medium.com/@bsouleymane78/06a8c6c375ff)。

**提到的链接**：

- [Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno)：该播放列表包含了所有关于 LangChain 的教程，LangChain 是一个使用 LLM 构建生成式 AI 应用的框架。
- [Developing an IA tool for Investing platform Due Diligence using LLM and RAG](https://medium.com/@bsouleymane78/06a8c6c375ff)：了解我使用生成式 AI 和 Python 开发 AI 工具以对投资平台进行尽职调查的历程。
- [Loom | Free Screen &amp; Video Recording Software](https://www.loom.com/share/c7a82509eaca450c814fae77c5db7a1d?sid=67313ae9-fca0-4a55-b536-a93b711a9d74)：使用 Loom 录制屏幕和摄像头的快速视频。清晰、轻松地解释任何事情——并跳过会议。混合办公场所的必备工具。
- [no title found](https://amzn.eu/d/dqQJzV1)：未找到描述

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1203343677993525288) (5 条消息): 

```html
<ul>
  <li><strong>Next.js 和 LangChain 构建 SMART 作品集</strong>：用户 <code>@flo_walther</code> 分享了一个 <a href="https://www.youtube.com/watch?v=1LZltsK5nKI">YouTube 视频</a>，关于如何使用 <strong>Next.js 14, Langchain, Vercel AI SDK</strong> 等技术构建 SMART 作品集网站，重点介绍了一个可以在你的数据上进行训练的 AI 聊天机器人。</li>
  <li><strong>教程困扰</strong>：<code>@stuartjatkinson</code> 表达了挫败感，因为 YouTube 上 LangChain 教程中的步骤已经发生了变化，或者直接照着做会产生错误。</li>
  <li><strong>LangChain 入门障碍</strong>：<code>@treym1112</code> 在按照 LangChain 官网上的快速入门教程操作时遇到错误，特别是在使用 <strong>Ollama 模型</strong>时，导致了关于缺少 'verbose' 属性的 <em>AttributeError</em>。</li>
  <li><strong>LangChain 指南发布</strong>：<code>@mehulgupta7991</code> 宣布在 <a href="https://amzn.eu/d/dqQJzV1">Amazon</a> 上发布了他们的新书《<em>LangChain in your Pocket: Beginner's Guide to Building Generative AI Applications using LLMs</em>》，并将其描述为一本涵盖多种用例和 LangServe 部署的实战指南。</li>
  <li><strong>结识数据科学家和内容创作者</strong>：<code>@mehulgupta7991</code> 分享了他们在 DBS Bank 担任数据科学家的职业背景，并提到了他们的 Medium+YouTube 频道“<em>Data Science in your Pocket</em>”，该频道包含约 600 个教程，其中包括一个 <a href="https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ">LangChain 播放列表</a>。</li>
</ul>
```

**提到的链接**:

- [Langchain](https://youtube.com/playlist?list=PLnH2pfPCPZsKJnAIPimrZaKwStQrLSNIQ&si=UGdxhD6LwRSgQzno): 该播放列表包含所有关于 LangChain 的教程，LangChain 是一个用于使用 LLM 构建生成式 AI 应用程序的框架。
- [构建 SMART 作品集网站 (Next.js 14, Langchain, Vercel AI SDK, ChatGPT API, Tailwind CSS)](https://www.youtube.com/watch?v=1LZltsK5nKI): 你可以构建的最酷的作品集网站，以此打动招聘人员和朋友！它拥有一个基于你的数据训练的 AI 聊天机器人。该 AI 可以回答任何问题...
- [未找到标题](https://amzn.eu/d/dqQJzV1): 未找到描述

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1203389816608264294) (5 条消息): 

- **RAG 开发挑战已解决**：`@wenqi_glantz` 详细介绍了构建生产级 RAG 时的 12 个痛点，并与 `@llama_index` 一起为每个痛点提供了完整的解决方案列表，可在新发布的速查表 (cheatsheet) 中找到。公告和解决方案可以在他们的 [Twitter 帖子](https://twitter.com/llama_index/status/1753829824484065625)中找到。
- **DataStax 黑客松动态**：黑客松于上午 9 点开始，@llama_index 感谢 `@DataStax` 提供场地和食物。有关该活动的见解分享在他们的 [Twitter 更新](https://twitter.com/llama_index/status/1753845015833686132)中。
- **MacBook 上的多模态模型**：`@llama_index` 宣布了他们与 Ollama 的新集成，支持开发本地多模态应用程序，如结构化图像提取和图像标注。更多详情请见他们的 [第一天集成推文](https://twitter.com/llama_index/status/1753875735776018786)。
- **多语言嵌入优化技术**：Iulia Brezeanu 在 @TDataScience 上发表的一篇新文章讨论了如何为多语言 RAG 选择合适的嵌入模型，解决了基准测试中的语言偏见问题。该文章有助于引导用户使用未针对英语优化的模型，并通过 [LlamaIndex 的推文](https://twitter.com/llama_index/status/1754185891118239894)分享。
- **Discord 迎来 LlamaIndex 的 Slack 机器人**：`@llama_index` 发布了其深受欢迎的 Slack 机器人的 Discord 版本。感兴趣的用户可以通过其 [推文公告](https://twitter.com/llama_index/status/1754257239525982685)中分享的链接进行访问。
  

---

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1203309190773477456) (19 messages🔥): 

- **寻求 AI 解释能力的澄清**：`@meowmeow008` 正在探索 AI 如何解释 SQL 查询及后续请求（如百分比计算），并对 AI 能力可能存在的误解表示好奇。
- **Azure AI 表现不一**：`@aldrinjoseph` 在从 Azure OpenAI 3.5 Turbo 切换到 Azure OpenAI 3.5 Turbo 16K 时遇到问题，后者生成的答案超出了给定的上下文范围。
- **LlamaIndex 比 LangChain 更可靠**：`@7leven` 表达了对 LlamaIndex 优于 LangChain 的偏好，批评后者经常破坏其文档，而集成前者带来的麻烦更少。
- **无需重新实例化即可调整 Hybrid Retriever**：在关于 **Hybrid Retriever** 的讨论中，`@cheesyfishes` 向 `@7leven` 确认，可以在 Python 代码中动态调整 **alpha** 参数，而无需重新实例化。
- **RAG 应用开发与聊天历史集成**：`@jameshume` 正在寻求关于如何将聊天历史记录整合到应用中的指导，该应用利用了包括自定义 `VectorDBRetriever` 和 `CondenseQuestionChatEngine` 在内的多个组件；`@dirtikiti` 解释了一种在新的 Prompt 中跟踪并包含聊天历史记录的简单方法。

**提到的链接**：

[Usage Pattern - LlamaIndex 🦙 0.9.44](https://docs.llamaindex.ai/en/stable/module_guides/deploying/chat_engines/usage_pattern.html#available-chat-modes)：未找到描述

  

---


### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1203387117682819072) (7 messages): 

- **递归检索的探索**：`@chiajy` 分享了关于在开发可深入挖掘非结构化数据的**自学习 RAG 系统**中进行递归或迭代检索的见解。他们分享了一篇 Medium 文章，详细介绍了三种递归检索技术：基于页面的（Page-Based）、以信息为中心的（Information-Centric）和以概念为中心的（Concept-Centric），详见 [Advanced RAG and the 3 types of Recursiv...](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba)。

- **对递归检索技术文章的赞赏**：用户 `@jerryjliu0` 对 `@chiajy` 关于 RAG 系统中递归检索的文章表示赞赏，称其为“好文章！”

- **展示 LlamaIndex 对比研究**：`@andysingal` 分享了一篇文章，比较了来自 Jina AI、Nomic AI 和 FlagEmbedding 的 Embedding 技术，并讨论了它们与 LlamaIndex 的集成。这篇文章题为《揭秘 LlamaIndex 的力量》（Unveiling the Power of LlamaIndex），探讨了这些技术在 AI 中的协同作用：[Unveiling the Power of Llamaindex](https://medium.com/ai-advances/unveiling-the-power-of-llamaindex-jina-vs-nomic-ai-vs-flagembedding-557158d7ad1e)。

- **引入 BGE-M3 Embedding 模型**：`@alphaatlas1` 介绍了 **BGE-M3 embedding model**，强调了其多功能、多语言和多粒度的特性。该模型可以执行稠密检索（dense retrieval）、多向量检索（multi-vector retrieval）和稀疏检索（sparse retrieval），支持 100 多种语言，并能处理高达 8192 个 token 的各种输入粒度，在 Hugging Face 上有详细说明：[BGE-M3 on Hugging Face](https://huggingface.co/BAAI/bge-m3)。

- **RAG 检索流水线建议**：`@alphaatlas1` 建议在 RAG 检索中使用混合检索加重排序（re-ranking）流水线，以利用各种方法的优势来获得更高的准确度。他们提到 BGE-M3 模型简化了 Embedding 检索，因为不需要为查询提供额外的指令，并邀请社区为稀疏检索方法做出贡献。

**提到的链接**：

- [Advanced RAG and the 3 types of Recursive Retrieval](https://medium.com/enterprise-rag/advanced-rag-and-the-3-types-of-recursive-retrieval-cdd0fa52e1ba)：我们探讨了 3 种类型的递归检索——基于页面的、以信息为中心的和以概念为中心的检索。
- [BAAI/bge-m3 · Hugging Face](https://huggingface.co/BAAI/bge-m3)：未找到描述
- [Unveiling the Power of Llamaindex: Jina vs Nomic AI vs FlagEmbedding](https://medium.com/ai-advances/unveiling-the-power-of-llamaindex-jina-vs-nomic-ai-vs-flagembedding-557158d7ad1e)：作者 Ankush k Singal

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1203260107606073384) (29 messages🔥): 

- **探索 GPT 商店和 API**：`@tiagoefreitas` 询问了是否有类似于 OpenRouter 的 GPT 商店，以及在公共服务器下的 OpenGPTs API，并希望 **@LangChainAI** 能在 OpenGPTs 中实现联邦（federation）。他们阐明，联邦将允许在管理自己服务器的同时，通过 API 使用来自其他服务器的 GPT。

- **开放模型优于传统写作**：`@slono` 批评了传统的写作方式，强调了使用像 **mlewd mixtral** 这样的随机模型（stochastic models）工作的乐趣和效率。讨论表明，相比标准的写作方法，人们更倾向于开放模型的动态输出。

- **Sentry 深入 Q&A**：`@kaycebasques` 指出了 Q&A 解决方案的一种趋势，并以 Sentry 的做法为例，Sentry 为 20 多种编程语言和框架创建了庞大的 Q&A 资源库。这表明了一个向专业化技术 Q&A 平台发展的更广泛趋势。

- **Llava 推理速度令人印象深刻**：`@ashpreetbedi` 分享了在 MacBook 上本地运行 **Ollama Llava** 时关于**推理速度（inference speed）**的积极体验，有助于社区了解该工具的性能。

- **技术职业生涯的十字路口**：`@mr.osophy` 表达了对于接受一份与其 ML Engineering 兴趣无关的工作的犹豫，在“成为理想职位的更好候选人”与“眼前但无关的工作机会”之间进行权衡。这一困境凸显了技术专业人士在将职业变动与个人抱负及财务约束挂钩时面临的挑战。

**提到的链接**：

- [Arcee and mergekit unite](https://blog.arcee.ai/arcee-and-mergekit-unite/)：几个月前，我在语言模型训练领域偶然发现了一种名为 Model Merging 的创新技术。这种 SOTA 方法涉及将两个或多个 LLM 融合为一个单一的...
- [Sentry Overflow](https://technicalwriting.tools/posts/sentry-overflow/)：就在刚才，当我搜索如何检查目录是否存在的 Bash 脚本语法时（因为我的记忆力像一只焦躁的吉娃娃，这已经是第 63 次搜索了），我发现了一些有趣的...

  

---


### Latent Space ▷ #[llm-paper-club-east](https://discord.com/channels/822583790773862470/1200029657744027658/) (1 messages): 

swyxio: 查看摘要
  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1203390847085969418) (28 messages🔥): 

- **从预生成池中提取的组合**：`@dbreunig` 理论上认为，游戏中的“疯狂组合”之所以会有脉冲和延迟，是因为当发生哈希未命中（hash miss）时会生成新的组合，这意味着过去的组合被存储并重复使用。
- **有趣的游戏机制激发好奇心**：`@chrisamico` 和 `@cameron_y` 表达了可视化游戏词汇组合谱系的愿望，推测 **embeddings** 可能为合成路径提供洞察。
- **利用合成代码**：`@madacol` 提供了一个 **JavaScript bookmarklet**，它可以与游戏的 `localStorage` 交互，以导出并自动保存合成的物品和发现，通过检索所有已合成的原料，为游戏体验增添了新的维度。
- **由 LLM-2 驱动的游戏**：`@madacol` 澄清了游戏巧妙组合背后的 AI，指出创作者使用的是 **llama 2**，正如 @nealagarwal 的推文所披露，并由 TogetherAI 提供支持。
- **合成方向很重要**：`@madacol` 发现，在这个游戏中，组合元素的顺序会影响结果，某些结果只有在特定物品放在其他物品之上时才能成功，并指出服务器对尝试过的组合的记忆禁止在尝试后反转顺序。

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1203298413458759722) (8 条消息🔥): 

- **提升德语模型性能**：`@johannhartmann` 讨论了在使用 **German dpo** 和 **laserRMT** 后 mt-bench-de 分数的提升，重点是使用 dare_ties 合并德语 7B 模型。
- **对神秘改进方法的好奇**：应 `@philipmay` 的详细信息请求，`@johannhartmann` 提供了 [German dpo 的链接](https://github.com/mayflowergmbh/intel_orca_dpo_pairs_de) 和 [laserRMT 的链接](https://github.com/cognitivecomputations/laserRMT)，但承认并不完全理解性能变化的原因，特别是数学能力的下降。
- **寻求长上下文 LLMs 的研究**：`@nsk7153` 询问了关于处理长上下文的大语言模型 (LLMs) 的研究，并分享了一个 [Semantic Scholar 搜索链接](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance)，其中包含他们查阅过的资料。
- **介绍 GermanRAG 数据集**：`@rasdani` 自豪地发布了用于微调 **Retrieval Augmented Generation** 的 GermanRAG 数据集。他们分享了该数据集的 [GitHub 链接](https://github.com/rasdani/germanrag)，并鼓励进行定制和增强。
- **对斯堪的纳维亚语言模型的羡慕**：`@johannhartmann` 发现了用于斯堪的纳维亚自然语言生成的 [ScandEval benchmark](https://scandeval.com/mainland-scandinavian-nlg/)，并表示希望德语也能有类似的基准。

**提到的链接**：

- [Mainland Scandinavian NLG](https://scandeval.com/mainland-scandinavian-nlg/)：一个自然语言理解基准测试。
- [large language models for handling long context prompts | Semantic Scholar](https://www.semanticscholar.org/search?q=large%20language%20models%20for%20handling%20long%20context%20prompts&sort=relevance)：一个利用人工智能方法提供高度相关结果和新颖过滤工具的学术搜索引擎。
- [Discord - A New Way to Chat with Friends &amp; Communities](https://discord.com/channels/1178995845727785010/1182877486854451271/1201826534114218034)：Discord 是通过语音、视频和文字进行交流的最简单方式。聊天、聚会，并与你的朋友和社区保持紧密联系。
- [GitHub - rasdani/germanrag: GermanRAG - a German dataset for finetuning Retrieval Augmented Generation](https://github.com/rasdani/germanrag)：GermanRAG - 一个用于微调 Retrieval Augmented Generation 的德语数据集。
- [GitHub - cognitivecomputations/laserRMT: This is our own implementation of &#39;Layer Selective Rank Reduction&#39;](https://github.com/cognitivecomputations/laserRMT)：这是我们对 'Layer Selective Rank Reduction' 的自行实现。

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 条消息): 

flozi00: 我目前正在努力提供德语托管服务。
  

---



### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1203436983083466794) (5 条消息): 

- **关于 `mistral-7B open-orca` 训练数据的咨询**：`@njb6961` 对复制 **mistral-7B open-orca** 表现出兴趣，并询问用于训练的 `curated filtered subset of most of our GPT-4 augmented data`（我们大部分 GPT-4 增强数据的精选过滤子集）是否会发布。
- **寻找特定数据集**：`@njb6961` 推测所讨论的数据集可能是 [SlimOrca](https://huggingface.co/datasets/Open-Orca/SlimOrca)，它包含约 50 万条 GPT-4 补全数据，并经过精选以在更少算力的情况下提高性能。
- **数据集确认**：`@ufghfigchv` 确认 [SlimOrca 数据集](https://huggingface.co/datasets/Open-Orca/SlimOrca) 确实是所使用的子集，并提到模型的 **training configuration** 应该在模型仓库的 **config** 子目录中。
- **请求营销联系人**：`@tramojx` 联系了管理员，寻求 **listing and marketing proposal** 的联系方式，但在现有的消息记录中未见回复。

**提到的链接**：

[Open-Orca/SlimOrca · Datasets at Hugging Face](https://huggingface.co/datasets/Open-Orca/SlimOrca)：未找到描述。

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=N5lDUZRI8sc
  

---

### Skunkworks AI ▷ #[bakklava-1](https://discord.com/channels/1131084849432768614/1163141825092145182/1203924122434801674) (1 messages): 

- **关于文档嵌入（Document Embeddings）与视觉嵌入（Vision Embeddings）的咨询**：`@epinnock` 联系了 `@far_el`，咨询关于创建一个带有完整文档文本嵌入的 **llava** 版本，并将其与视觉嵌入方法进行对比。他们认为这项任务可能是对 encoder/decoder 模型的局部重新实现，并寻求关于该任务除此之外可能涉及的内容的澄清。
  

---



### LLM Perf Enthusiasts AI ▷ #[reliability](https://discord.com/channels/1168579740391710851/1169378117865963580/1203812625641635841) (1 messages): 

- **使用 BentoML 轻松部署**：用户 `@robotums` 提到在 AWS 上使用 VLLM 后端通过 **BentoML** 成功部署了开源软件模型。他们形容这个过程非常直接：“这非常简单，你只需要运行 bento 即可。”
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1203453017672777842) (1 messages): 

- **介绍 DSPy，一个语言模型编程框架**：用户 `@sourya4` 介绍了 `[DSPy](https://github.com/stanfordnlp/dspy)`，这是一个斯坦福大学的项目，旨在对基础模型进行编程——而不仅仅是提示（prompting）。他们还分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Dt3H2ninoeY)，题为 "SBTB23: Omar Khattab, DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines"，描述了在提示语言模型及其流水线集成方面的进展。

**提到的链接**：

- [SBTB23: Omar Khattab, DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines](https://www.youtube.com/watch?v=Dt3H2ninoeY)：ML 社区正在迅速探索提示语言模型（LMs）以及将其堆叠成解决复杂任务的流水线的技术。遗憾的是...
- [GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models](https://github.com/stanfordnlp/dspy)：DSPy：用于编程——而非提示——基础模型的框架 - GitHub - stanfordnlp/dspy: DSPy: The framework for programming—not prompting—foundation models

  

---

### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1203289597669539890) (1 条消息): 

- **AIEF 保加利亚分会活动火热**：AIEF 保加利亚首个分会的负责人 `@yavor_belakov` 强调了 **第二次 AIEF BG 每月聚会，共有 90 人参加**，并引入了“闪电演讲（Lightning Talks）”。社交、披萨和知识交流是活动的核心。
- **闪电演讲洞察现已发布**：分享了近期活动的演讲内容，可以预览 **QR Code Art、Weaving The Past、LMMs、Zayo** 以及 **在 AI 时代构建具有防御性的业务** 等主题。完整录像将发布在他们的 YouTube 频道上。
- **探索 ChatGPT 落地应用**：演讲中包含 Iliya Valchanov 关于 **“ChatGPT Adoption Methodology”** 的演示，详情可见分享的 [Google Slides 文档](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx)。
- **LinkedIn 窗口展现 AIEF 保加利亚进展**：`@yavor_belakov` 还在 LinkedIn 上发布了活动亮点，展示了 **AIEF foundation 社区** 的实力和活力。

Google Slides 演示文稿和 LinkedIn 帖子的链接未完整提供，因此无法准确分享。

**提到的链接**：

- [Yavor_Belakov_QR_Code_Art.pptx](https://docs.google.com/presentation/d/1yVJISPqnkaM8RjF8pBjEnJO9XXVm78v7): 1 索非亚 2024年2月1日 Meetup #2 AIE.F AI Engineer Foundation | 欧洲 | 保加利亚分会 下午 6 点 – 7 点 社交与披萨 下午 7 点 – 9 点 闪电演讲 下午 9 点 – 10 点 更多社交 AIEF Meetup #2: Lighting Ta...
- [Dani_Matev_Weaving_The_Past.pptx](https://docs.google.com/presentation/d/1c8qVQJ5SmYGGSlm5-Ae80XxutDF2qwaf): Knitting（编织）通过用针或机器将羊毛或其他纱线的环扣在一起制作（衣服、毯子等）。大家好，感谢能有这次演讲机会。现在，在...之前...
- [Dimo_Michev_LLMs.pptx](https://docs.google.com/presentation/d/19f01za6w5eZQPI3sYhNr2ILIFrsRXYW3): Python 版 LLM 软件包，用于查询 Large Language Models 的命令行和 Python 软件包，您的现场私有 LLM 工具 https://llm.datasette.io/en/stable/index.html#
- [Nicole_Yoan_Zayop.pptx](https://docs.google.com/presentation/d/1P9_mU30ed9nuLgN2esahU7svtsNuKRL8): Zayo 通过对话式 UX 重新想象员工管理 2024/02/01
- [Georgi_Stoyanov_How to Build a Defensible Business in the Age of AI.pptx](https://docs.google.com/presentation/d/1sNj3Q6Fd4jYjVVvjJBen_IgA-nbpm4x4): 1 如何在 AI 时代构建具有防御性的业务
- [Iliya_Valchanov_ChatGPT Adoption Methodology.pptx](https://docs.google.com/presentation/d/1XPMlt-qlZLagrvk4trNEI16ZSOPHRVGx): 1 ChatGPT Adoption Methodology Iliya Valchanov