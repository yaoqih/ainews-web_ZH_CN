---
companies:
- openai
- stability-ai
date: '2024-03-26T01:11:50.136366Z'
description: '**吴恩达在《The Batch》中关于智能体（Agents）的专题文章**指出，采用迭代式智能体工作流能显著提升代码基准测试的性能。其中，**GPT-3.5**
  在智能体循环（agent loop）的加持下，在 HumanEval 测试中的正确率高达 **95.1%**，远超 **GPT-4** 在零样本（zero-shot）模式下
  **67.0%** 的表现。


  该报告还涵盖了 **Stable Diffusion** 模型的新进展，包括用于生成火影忍者风格图像的 **Cyberrealistic_v40**、**Platypus
  XL** 和 **SDXL Lightning**，以及 LoRA 和上采样（upscaling）技术的创新。此外，文中还讨论了**本地大语言模型（LLM）的部署**与优化，重点关注硬件配置和微调策略，以实现高效推理和多用户服务。最后，报告还提到了
  Emad 从 **Stability AI** 离职以及 **OpenAI** 发布的新 **Sora** 视频。'
id: 474d1a8b-e935-44b5-b8bf-714a5c63822a
models:
- gpt-3.5
- gpt-4
- cyberrealistic_v40
- platypus-xl
- sdxl-lightning
original_slug: ainews-andrew-likes-agents
people:
- andrew-ng
- lilian-weng
- emad
title: 安德鲁喜欢智能体。
topics:
- agents
- human-eval-benchmark
- fine-tuning
- local-llm-deployment
- inference-speed
- image-generation
- lora
- upscaling
- workflow-optimization
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月21日至3月25日的 AI 新闻。我们为您查看了 [**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **22** 个 Discord 社区（包含 **342** 个频道和 **12281** 条消息）。预计节省阅读时间（以 200wpm 计算）：**1173 分钟**。

[Andrew Ng 在 The Batch 上关于 Agent 的文章](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/?utm_source=ainews&utm_medium=email) 本周末在各大平台引起了轰动：

> Devin 最近引人注目的演示在社交媒体上引起了大量关注。我的团队一直在密切关注编写代码的 AI 的演进。我们分析了多个研究团队的结果，重点关注算法在广泛使用的 HumanEval 编程基准测试中的表现。您可以在下图中看到我们的发现。
>
> GPT-3.5 (zero shot) 的准确率为 48.1%。GPT-4 (zero shot) 表现更好，达到 67.0%。然而，从 GPT-3.5 到 GPT-4 的提升，在引入迭代式 Agent 工作流（iterative agent workflow）后显得相形见绌。事实上，在 Agent 循环的封装下，GPT-3.5 的准确率最高可达 95.1%。

 
![image.png](https://assets.buttondown.email/images/b3745325-26e9-47d6-a195-5868d74082c8.png?w=960&fit=max)
 

对于研究过 Agent 领域的专业人士来说，这些内容并不新鲜，但 Andrew 的公信力和 Agent 框架（非常接近 [Lilian Weng](https://twitter.com/lilianweng/status/1673535600690102273?lang=en) 的观点 + 最近多智能体协作 multiagent collaboration 的新玩法）使其极具说服力。

我们今天发布了 [《ChatGPT 的解构》（The Unbundling of ChatGPT）](https://www.latent.space/p/feb-2024)。此外，[Emad 已从 Stability 辞职](https://stability.ai/news/stabilityai-announcement)，并且有 [更多 Sora 视频](https://openai.com/blog/sora-first-impressions?utm_source=ainews&utm_medium=email) 发布，请务必查看 Don Allen Stevenson III 的作品。

---

**目录**

[TOC] 

---

# REDDIT

> 我们增加了更多的 subreddit，并正在综合其中的主题。评论抓取尚未实现，但即将推出。

**Stable Diffusion 模型与技术**

- **新的 Stable Diffusion 模型和技术**正在开发中，例如用于生成火影忍者风格图像的 Cyberrealistic_v40、Platypus XL 和 SDXL Lightning。([Playing with Cyberrealistic_v40](https://www.reddit.com/gallery/1bmqyis), [Still Liking Platypus XL](https://i.redd.it/zvltrzu96dqc1.png), [Naruto (Outputs from SDXL Lightning)](https://www.reddit.com/gallery/1bmmuuj))
- /r/StableDiffusion: **正在探索 LoRA 和放大（upscaling）方法**以提高图像质量，例如在保留内容的同时将图像卡通化（平滑阴影、强化轮廓），以及用于消除常见问题的通用负面提示词（negative prompt）LoRA。([best LORA or method with sd1.5 to cartoon-ize an image while keeping content? (flatten shading, reinforce outlines)](https://www.reddit.com/r/StableDiffusion/comments/1bmjdhs/best_lora_or_method_with_sd15_to_cartoonize_an/), [General purpose negative prompt?](https://www.reddit.com/r/StableDiffusion/comments/1bmgj70/general_purpose_negative_prompt/))
- /r/StableDiffusion: **正在为 Stable Diffusion 开发新的工作流和扩展插件**，例如用于 ComfyUI 放大的 BeautifAI、用于混合模型权重的 FrankenWeights，以及在 Fooocus 中集成 Prompt Quill 扩展。([BeautifAI - Image Upscaler & Enhancer - ComfyUI](https://www.reddit.com/r/StableDiffusion/comments/1bn5jdu/beautifai_image_upscaler_enhancer_comfyui/), [It's alive! FrankenWeights is coming... [WIP]](https://www.reddit.com/r/StableDiffusion/comments/1bmemjs/its_alive_frankenweights_is_coming_wip/), [Prompt Quill in Fooocus](https://www.reddit.com/r/StableDiffusion/comments/1bmm615/prompt_quill_in_fooocus/))

**本地 LLM 部署与优化**

- /r/LocalLLaMA: **在本地部署大语言模型**是一个热门话题，讨论围绕硬件要求、推理速度以及针对不同使用场景的模型选择展开。([将 P40 24GB 与 3090 组合以获得 48GB VRAM 是否合理？](https://www.reddit.com/r/LocalLLaMA/comments/1bn43rt/would_it_make_sense_to_stick_a_p40_24gb_in_with_a/), [4090 和 64GB RAM 的最佳输出质量？](https://www.reddit.com/r/LocalLLaMA/comments/1bmk3c7/best_output_quality_for_4090_64gb_ram/), [你的电脑配置是什么？](https://www.reddit.com/r/LocalLLaMA/comments/1bmu9sh/what_is_your_computer_specs/))
- /r/LocalLLaMA: **优化 LLM 性能**是一个活跃的研究领域，讨论涉及推理架构、微调策略以及如何高效地为多用户提供服务。([什么样的架构能带给我们具备推理能力的 LLM？](https://www.reddit.com/r/LocalLLaMA/comments/1bn48cl/what_architecture_will_give_us_a_reasoning_llm/), [只工作不玩耍会让 LLM 变傻；为什么我们在微调时应该混入预训练数据。](https://www.reddit.com/r/LocalLLaMA/comments/1bmslfq/all_work_and_no_play_makes_your_llm_a_dull_boy/), [是否可以使用 llama-cpp-python 同时为多个用户提供服务？](https://www.reddit.com/r/LocalLLaMA/comments/1bmw12y/is_it_possible_to_serve_mutliple_user_at_once/))
- /r/LocalLLaMA: **指南与资源**正在不断完善，旨在帮助用户从入门到进阶掌握本地 LLM。([新用户入门指南：从纯小白到资深用户，第 1/3 部分，再次尝试...](https://www.reddit.com/r/LocalLLaMA/comments/1bmvtyb/new_user_beginning_guide_from_total_noob_to/))

**机器学习研究与技术**

- /r/MachineLearning: 新的**机器学习架构与技术**不断被提出和讨论，例如使用硬注意力（hard attention）和决策树进行因果语言建模的 Treeformers。([[P] Treeformer：硬注意力 + 决策树 = 因果语言建模](https://www.reddit.com/r/MachineLearning/comments/1bmmqqq/p_treeformer_hard_attention_decision_trees_causal/))
- /r/MachineLearning: 部署 ML 模型的**优化技术**正在被探索，例如使用 TensorRT 进行快速 PyTorch 模型推理。([[D] 寻找在 TensorRT 上运行 PyTorch 模型的最快推理方式](https://www.reddit.com/r/MachineLearning/comments/1bmnn5j/d_looking_for_fastest_inference_way_to_run_a/))
- /r/MachineLearning: **调试与改进 ML 模型**是一个持续的挑战，讨论集中在理解和修复测试损失（test loss）激增等问题上。([[D] 有人知道为什么我的测试损失会如此疯狂地激增吗？](https://www.reddit.com/r/MachineLearning/comments/1bmor9g/d_does_anyone_know_why_my_test_loss_is_spiking_so/))

**AI 助手与应用**

- /r/OpenAI: **AI 助手正以全新的方式被使用**，例如调解争论以提供中立视角，以及辅助编程任务。([使用 ChatGPT 调解争论](https://www.reddit.com/r/OpenAI/comments/1bmgh5w/mediating_arguments_with_chatgpt/), [可在 3090 上运行的编程 LLM](https://www.reddit.com/r/LocalLLaMA/comments/1bmfu9g/coding_llm_that_runs_on_3090/))
- /r/StableDiffusion: **新的 AI 应用**正在开发中，例如 AI 网红、交互式“AI 画笔”工具，以及基于图像探索 AI 生成世界的沉浸式体验。([这是我在没有任何 Stable Diffusion 经验的情况下，尝试制作 AI 网红和 Fanvue/OF 模型的最初 45 天](https://www.reddit.com/r/StableDiffusion/comments/1bn0kf8/here_is_my_first_45_days_of_wanting_to_make_an_ai/), [使用 StreamDiffusion 构建交互式“AI 画笔”的快速拆解](https://v.redd.it/4yna3y1t2bqc1), [对未来感到兴奋](https://www.reddit.com/r/OpenAI/comments/1bmuqae/excited_for_the_future/))

**迷因与幽默**

- **AI 生成的迷因与幽默内容**继续流行，调侃 AI 的现状。([不要使用在成人内容上训练的模型生成树](https://i.redd.it/gq2xicv17cqc1.jpeg), ["兄弟，千万别在加油站买大麻"](https://i.redd.it/s09ogpjy4cqc1.png), [确实就是这样](https://i.redd.it/9tfrhr4p9eqc1.png))

# PART X: AI Twitter 回顾

> 所有回顾均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果

**模型发布与更新**

- [Mistral AI 发布了全新的 7B 0.2 base 模型，具有 32k context window](https://twitter.com/osanseviero/status/1771654833830821949)，在黑客松上宣布（1.3万次观看）
- [量化 4-bit Mistral 7B 模型发布](https://twitter.com/danielhanchen/status/1771737648266178801)，通过 QLoRA finetuning 实现推理速度提升 2 倍，VRAM 占用减少 70%（1.4万次观看）
- [Mistral 7B 0.2 预计将超越 Yi-9B](https://twitter.com/teortaxesTex/status/1771665411857096870)（486次观看）


**开源努力与挑战**

- [Stability AI 的 Emad Mostaque 在投资者反水和员工流失后离职](https://twitter.com/stanfordnlp/status/1771671300823826867)（1.1万次观看）
- [如果过去十年的 AAA 游戏不是“烂透了”，开源 AI 会更好](https://twitter.com/teortaxesTex/status/1771651366290636913)，这使得高端 GPUs 在游戏领域变得不值（1.7千次观看）
- [没有分布式预训练，开源 AI 就不是真正的开源](https://twitter.com/generatorman_ai/status/1771516761206395164)，不能指望 VCs 投入数百万美元后又将其免费赠送（468次观看）

**新兴应用与演示**

- [金融 Agent 应用](https://twitter.com/virattt/status/1771614341831201193) 使用 LangChain 构建，可获取股票价格、财务数据、市场新闻（4.6万次观看）
- [Telegram 代理设置指南](https://twitter.com/rameerez/status/1771602942287626281) 用于利用内置代理功能规避西班牙可能的禁令（5万次观看）
- [由 Mistral 7B 驱动的跳舞机器人](https://twitter.com/sophiamyang/status/1771667841181233221) 在黑客松上演示（9.8千次观看）
- [Claude 对 Claude 的对话](https://twitter.com/AISafetyMemes/status/1771768138042122301) 引发了令人担忧的输出，如“精神崩溃”（5.3万次观看）


---

# PART 0: 总结之总结之总结


- **Mistral 发布新 7B v0.2 Base 模型**：Mistral AI 在 [@cerebral_valley 黑客松](https://x.com/alexreibman/status/1771608346635751541?s=46)上随手发布了他们新的 **Mistral 7B v0.2 Base** 模型，具有 32k context window 以及在[发布说明](https://x.com/mistralailabs/status/1771670765521281370?s=46)中详述的其他改进。AI 社区正热烈讨论这一重大更新的影响和 benchmarking 结果。

- **Stability AI CEO Emad Mostaque 辞职**：在一次重大动荡中，**Emad Mostaque** [辞去了 Stability AI 的 CEO 职务](https://stability.ai/news/stabilityai-announcement)以追求去中心化 AI。临时联合 CEO Shan Shan Wong 和 Christian Laforte 将领导寻找正式继任者的工作。在这次领导层更迭中，关于公司未来方向和对开源倡议承诺的猜测四起。

- **Anthropic 的 Claude 尽管有限制但表现出色**：用户称赞 **Anthropic 的 Claude** 的性能和上下文处理能力，尤其是自我调节版本，但对 200k context window 每天严格的 1M token 速率限制表示沮丧。**$500 规模计划**被认为是更适合大规模使用的选择，而 Claude 的 API 在开源开发方面的潜力也引发了兴奋。

- **优化器和架构推动 LLM 进步**：新型优化器如 **GaLore** 和架构如 **DenseFormer** 正在推高语言模型训练效率和性能的边界。讨论围绕 GaLore 显著的 VRAM 节省和潜在的过度训练风险展开，而 DenseFormer 的[深度加权平均](https://arxiv.org/abs/2402.02622)显示出有前景的 perplexity 改进。社区热切期待这些领域的进一步发展。

- **AI 助手和 Agent 进化**：像 **Open Interpreter 的 01 Light**（一个完全开源的个人 AI Agent）和来自 Nous Research 的 **World Simulator** 等项目，以其引人入胜的体验和定制潜力吸引了社区的想象力。同时，像 **LangChain** 这样的框架正在为 AI Agent 实现更复杂的决策和任务自动化，各种分享的指南和教程也证明了这一点。


---



# PART 1: 高层级 Discord 总结

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **SD 生态系统反响热烈**：社区正在积极讨论 **Stable Diffusion** 模型，特别是对即将发布的 **SD3** 充满期待，围绕潜在的改进以及与 **SDXL** 等模型的对比分析展开了热议。此外，还提出了关于 AMD GPU 兼容性的问题，成员们分享了解决方案和变通方法。

- **点击即得 AI 艺术，但并非一帆风顺**：用户对 **Civitai** 和 **Suno** 等在线 AI 图像生成服务表示了不满，理由是内容限制和生成内容的类型问题。社区成员分享了 [Stable Cascade Examples](https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/) 等资源，以展示不同模型的能力。

- **监管争议**：关于 AI 技术监管影响的辩论呈现两极分化。伦理考量与对扼杀创新的担忧并存，反映出社区对开源开发与专有限制之间平衡的关注。

- **技术支持社区**：随着新手和资深人士共同应对模型安装相关的技术难题，社区形成了浓厚的知识共享氛围。成员们分享了学习和排查故障的资源，包括支持频道的直接链接和社区内的专家建议。

- **连接 AI 资源**：分发了各种链接以提供更多信息和实用工具，例如 [Stable Diffusion Glossary](https://stable-diffusion-art.com/glossary/)，以及一个全面的 Stable Diffusion 多平台包管理器 [StabilityMatrix](https://github.com/LykosAI/StabilityMatrix/blob/main/README.md)。这些工具旨在帮助 AI 工程师理解并增强对 Stable Diffusion 产品的使用。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **在内存限制内创新**：**PyTorch** `torchtune` 仓库的一个[最新 Pull Request](https://github.com/pytorch/torchtune/pull/527) 允许在保持内存占用低于 16GB 的情况下进行全模型微调，使拥有消费级 GPU 的用户能够更高效地进行训练。
- **增强微调能力**：[Hugging Face PEFT](https://github.com/huggingface/peft/releases/tag/v0.10.0) 的最新版本 (v0.10.0) 包含了 LoftQ，它改进了大模型的微调效果。
- **在 Mistral 上实现 ORPO**：一位用户报告了在 **Mistral-7B-v0.2** 上使用 *argilla/ultrafeedback-binarized-preferences-cleaned* 数据集有效应用 **ORPO TRL** 的情况，表明该方法具有进一步优化的潜力。
- **讨论 AI 模型训练的最佳实践**：各频道的讨论涉及 **ORPO 中的多轮对话训练**挑战、标准化格式的重要性，以及使用 *Ollama templates* 和不同量化模型（Quant models）时的紧迫问题。
- **模型性能里程碑**：社区庆祝了新模型的性能表现，如 [sappha-2b-v3](https://huggingface.co/Fizzarolli/sappha-2b-v3) 和 [MasherAI-v6-7B](https://huggingface.co/mahiatlinux/MasherAI-v6-7B)，据报道，这些模型在通过 **Unsloth** 对 Gemma-2b 进行微调后，超越了基准测试。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity Pro 用户就图像生成功能展开讨论**：Pro 用户讨论了 **Perplexity 的图像生成**功能，并指出在网页版上关闭 Pro 开关可以使用 Writing 模式进行图像生成。

- **模型对决：Claude Opus 对阵 GPT-4 Turbo**：工程师们的闲聊涉及对比 **Claude 3 Opus 和 GPT-4 Turbo**，强调 GPT-4 Turbo 可以编译 Python 文件，而 Perplexity 则不行。

- **Stability AI 本地模型引发关注**：关于 **Stability AI 的本地模型**（如 SDXL）讨论热烈，重点在于性能与在个人硬件上运行这些工具的沉重成本之间的权衡。

- **Perplexity 的困惑与潜力**：用户对 Perplexity 的某些方面感到困惑，包括侵入性的搜索触发和无关的提示词，同时也展望了 **Claude 3 Opus API** 以及与 iOS Spotlight 搜索集成等功能。

- **在 Token 限制内编写代码**：对于使用 **Perplexity API** 的工程师来说，一个关键提示是注意 **16,384 Token 限制**，并建议使用 OpenAI 的 Tokenizer 等工具来准确测量 Token 数量，以遵守限制并实现最佳运行。

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **为 AI 驯服 VRAM 巨兽**：工程师们讨论了运行 AI 模型的最佳 GPU，RTX 3090 的 24GB VRAM 是一个热门选择。在配置多 GPU 设置时，注意到了不同 GPU 品牌（如 AMD 和 NVIDIA）之间的兼容性问题。

- **本地 LLM 掀起技术革命**：关于 LLM 分布式计算可行性的讨论显示出疑虑，主要原因是使用 ZLUDA 配合 ROCm 时 CPU 占用率过高，尽管类似于 Linux/LAMP 时代的本地计算极具吸引力。

- **文档发布与多模型热潮**：LM Studio 推出了[新文档网站](https://lmstudio.ai/docs)，并引入了多模型会话（Multi Model Session）功能，[教程视频](https://youtu.be/4fdZwKg9IbU?feature=shared&t=357)中对此进行了讲解。

- **LM Studio 的成长的烦恼与性能怪癖**：用户报告了从 **高 CPU 占用率**到模型输出乱码等问题，有时通过简单的重启即可解决。对于 RX 570 配合 ROCm 等旧硬件的兼容性受到质疑，加载模型时出现的 "Exit code: 42" 等错误信号表明需要持续排查故障。

- **Open-interpreter 拆解与 GGUF 模型性能**：Open-interpreter 的问题包括连接故障以及对各种 GGUF 兼容模型性能的讨论。Open Interpreter 设备引起了关注，用户可以使用来自 [01.openinterpreter.com/bodies/01-light](https://01.openinterpreter.com/bodies/01-light) 的免费 STL 文件自行 3D 打印。同时，非官方支持模型的错误促使人们关注 [Open-interpreter GitHub 上的 issue #1124](https://github.com/OpenInterpreter/open-interpreter/issues/1124)。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 进军 Linux**：用户已成功在 Ubuntu 22.04 上运行 **Open Interpreter**，讨论了麦克风支持和客户端-服务器机制，标志着向跨平台兼容性的推进。

- **DIY 爱好者集结！O1 Light 热潮**：社区的热心成员正在分享 3D 打印和组装自己的 **01 Light** 设备的技巧，专门的 Discord 频道纷纷涌现，用于分享构建经验和设计调整。

- **AI 驱动的工程讨论升温**：围绕 **Open Interpreter** 的技术对话正在扩大，重点是在各种机器（无论是低配还是云端）上运行 01 服务器，并增强安装程序的易用性。开发者们还在构思 **Groq** 的集成以及扩展 *01 Light* 的功能。

- **社区贡献释放开源力量**：AI 工程师社区正投入到 **Open Interpreter** 项目的贡献中，重点关注应用开发、不同 **LLM** 的性能，以及适用于 **Apple silicon 设备**的潜在桌面应用。

- **开源 AI 助手登场**：一段关于 **01 Lite** 的 YouTube 视频被重点推荐，标题为“Open Interpreter's 01 Lite - 全球首款完全开源的个人 AI Agent 设备”，展示了这款自研 AI 助手的强大能力。此外还分享了一个剪辑过的直播视频，以简要概述 **01 软件** [Open Interpreter's 01 Lite](https://www.youtube.com/watch?v=Q_p82HtBqoc)。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **欧盟数据法挑战 LAION 的效率**：与美国数据集相比，LAION 数据集的表现可能不佳，这在很大程度上归因于欧盟严格的监管。讨论中提到了使用合成数据以及在限制较少的地区建立合作作为可能的变通方案，被幽默地称为“数据洗白”。

- **Stability AI 领导层变动**：Emad Mostaque 已辞去 Stability AI 的 CEO 职务，公司确认了他的辞职，并任命了临时联合 CEO Shan Shan Wong 和 Christian Laforte，他们将负责寻找继任者。人们对这一变动对公司未来及其对开源承诺的影响存在各种猜测 ([Stability AI 新闻稿](https://stability.ai/news/stabilityai-announcement))。

- **SD3 与 DALL-E 3 的对比**：讨论表明，SD3 模型在某些方面的表现可以与 DALL-E 3 媲美，但在理解复杂的交互方面表现吃力，导致生成的图像更像是拼贴式的图像组装，而不是连贯的概念融合。

- **行业风波中浮现的 AI 伦理辩论**：最近在 Twitter 上关于 AI 行业领军人物动机的对话引发了全频道范围内关于开发者和研究人员伦理责任的辩论，以及社交媒体上 AI “名人”文化的影响。

- **AMD GPU 在 AI 支持方面落后**：频道成员对 AMD 与 NVIDIA 相比在机器学习工作负载方面的支持表示不满。鉴于 Stable Diffusion 等模型的兴起，缺乏消费级机器学习支持被视为一种潜在的疏忽。

- **Andrew Ng 预见 AI 工作流的演进**：Google Brain 联合创始人 Andrew Ng 预测，**AI agentic workflows** 今年可能通过对文档进行多次迭代而超越下一代基础模型。目前的 one-shot LLM 方法需要进化 ([Reddit 热帖](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/))。

- **MIT 加速图像生成技术**：MIT CSAIL 开发了一种方法，通过使用简化的单步教师-学生框架，在不损失图像质量的前提下，将 Stable Diffusion 和 DALL-E 等工具的图像生成过程加速了 30 倍 ([MIT 新闻文章](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321))。

- **NVIDIA 解决扩散模型训练障碍**：NVIDIA 最近的博客文章讨论了扩散模型训练的改进，包括 EDM2 代码和模型的发布。他们解决了风格归一化问题，这些问题可以通过类似于 EDM2 中的更改来克服 ([NVIDIA 开发者博客文章](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/))。

- **随着线性网络的兴起，Unet 的未来受到质疑**：鉴于用于图像生成的线性网络模型的兴起，频道成员辩论了 Unet 改进的相关性。尽管 layer norm 和传统的归一化方法受到质疑，但它们作为网络功能的组成部分仍然是一个讨论话题。

- **大语言模型在剪枝中表现出韧性**：研究发现，即使移除中间块，大语言模型 (LLMs) 仍能保持性能，这暗示了某些片段的冗余性。这鼓励了对线性网络架构及其战略剪枝潜力的深入研究。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **发货查询与 DIY 解决方案**：成员们对某款未命名产品的发货时间表感到好奇，期待在**夏季上市**。在缺乏具体发布日期的情况下，他们讨论了替代的 DIY 方案。

- **AI 模型创新与乐观情绪**：对 **Claude** 在开源项目上的影响抱有极高热情。据报道，一种无需预训练的 **Raptor** 实现仅需 5 分钟即可总结 3b 模型的转录内容。分享了关于 [FastAPI](https://fastapi.tiangolo.com) 在后端开发易用性方面的参考资料，以及关于 Suno.AI 策划 Spotify 播放列表能力的说明。

- **部署与说服技术分析**：利用 Kubernetes 部署 Nous 模型，同时一项预注册的 [arXiv 研究](https://arxiv.org/abs/2403.14380) 分析了 LLM 的说服力。[ArtHeart.ai](https://artheart.ai/) 等平台在 AI 驱动的艺术创作方面获得认可，并对 BitNet 1.5 的量化感知训练（quantized-aware training）在推理加速方面的表现进行了评估。

- **编织世界与 AI 辅助治疗**：World Simulator 项目展现了极高的参与度，而名为 **Thestral** 的 AI 治疗师项目旨在利用 LLaMA 70B 模型。社区正在积极讨论 Claude 3 中 Opus 的**伦理约束**、Hermes 2 Pro 等模型中**拒绝提示词（refusal prompts）**的影响，以及规避被称为“奥弗顿窗口（Overton Effect）”的 LLM 限制的操纵技术。

- **LLM、微调与精炼问题**：围绕 SFT 数据集中是否包含 few-shot prompts 以及对微型 LLM 的追求展开了辩论，并建议观看 **Andrej Karpathy** 的视频。因果掩码（causal masking）的重要性以及 Llama 三层前馈设计背后的奥秘引发了讨论，焦点集中在一篇关于 [SwiGLU](https://arxiv.org/pdf/2002.05202.pdf) 非线性的 arXiv 论文上。

- **育儿、开源与 RAFT 的崛起**：成员们在寻找 Wikipedia RAG Index 等开源选项时，也分享了育儿经验。对话重点转向了一种名为 **RAFT** 的极具前景的检索增强微调方法，该方法在分享的论文中有所讨论，并可在 [Gorilla GitHub 仓库](https://github.com/ShishirPatil/gorilla/tree/main/raft)中进行探索。

- **闲聊与 World-Sim 技术**：一位成员暗示更改 Tenor.com 的语言设置，并分享了一个 [Grim Patron GIF](https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450)，而另一位成员简单的“helloooo”活跃了聊天气氛。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Sora 塑造电影制作的未来**：OpenAI 的 **Sora** 因赋能艺术家和电影制作人创作开创性的超现实艺术而受到赞誉。导演 Paul Trillo 称赞 Sora 能够将以前无法想象的创意视觉化，[OpenAI 博客](https://openai.com/blog/sora-first-impressions)强调了其潜力。

- **深入探讨 AI 的文化指南针**：AI 讨论中的一个热门话题是 GPT 等语言模型中被察觉到的西方自由中心主义偏见，这引发了关于是否应该存在多个文化对齐的 AI 版本的辩论。将 AI 与非西方规范对齐的努力正面临挑战，“自定义 ChatGPT”功能被提及作为用户根据自身价值观个性化 AI 回复的工具。

- **GPT-4 不断演进的功能与访问关注**：成员们注意到 **Custom GPT 固定限制**有所减少，并寻求共享 GPT 访问的键盘快捷键。OpenAI 确认了 **GPT-4 with Vision** 读取图像的能力，同时通过[停用通知](https://help.openai.com/en/articles/8988022-winding-down-the-chatgpt-plugins-beta)宣布 **ChatGPT 插件测试版**结束。

- **精炼 AI 以丰富用户体验**：分享了增强 AI 创意写作、叙事风格和代码输出质量的策略。一位用户因 OpenAI SDK 更新而在使用 `.Completion` 端点时遇到问题，并被引导至 openai-python GitHub 仓库上的 [v1.0.0 迁移指南](https://github.com/openai/openai-python/discussions/742)寻求帮助。

- **实现视觉无障碍**：为一位希望改善残障人士图像识别能力的成员提供了隐私敏感建议，重点是将问题提交至 Discord 建议频道。用户还探索了 Prompt Engineering 策略，以塑造具有特定个性的 AI，并优化假设段落的生成，避免通用化陈述。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**AI Art Prompt Guide Quest**: 用户正在寻求关于创作 AI 生成艺术提示词的建议，尽管未提供具体资源。

**Blenderbot's Role-Play**: 讨论强调了 Blenderbot 在互动中展现一致性格特征的能力，这与那些承认自己非人类本质的 AI 形成对比。

**GPU Operation Showdown**: 围绕 GPU 上乘法和条件检查执行速度差异展开了技术辩论。建议参考 'iq' 的工作以获得进一步见解。

**Complex Creativity for ChatGPT**: 一位用户为 ChatGPT 请求了一个语言多样且富有创意的提示词，引发了另一位用户对该提示词复杂性的惊叹。

**Optimizing GPU Inference**: 社区探索了使用 TensorRT-LLM 和 exLLama v2 等方法和库来优化 GPU 上的 LLM 推理，并推荐了适合多用户并发服务的工具。

**Rust's Rising Star**: 关于通过 *Candle* 库将 GLiNER 模型转换为 Rust 的对话指出，其优点包括减少依赖项并适合生产环境，且已确认支持 GPU。

**Efficient Coding with Federated Learning**: [一个开源 GitHub 项目](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting) 展示了一种用于负荷预测的能效型 Federated Learning 方法。

**Compiling the Stable Diffusion Compendium**: 社区成员分享了大量 Stable Diffusion 的资源和指南，包括用于全面学习 Stable Diffusion 的 [civitai.com](https://civitai.com/articles/2054)。

**Deck Out Your Memory – Diffusers Edition**: 一个用于估算 **`DiffusionPipeline` 推理时内存需求** 的实验性工具已 [发布并征求反馈](https://github.com/huggingface/diffusers/discussions/7434)。

**SegGPT: The Contextual Segmentor**: 介绍了 [HuggingFace 上的 SegGPT](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt)，这是一个具有出色 one-shot 分割能力的模型，可针对各种 image-to-image 任务进行训练。

**BLIP-2 Ups the Fusion Game**: 在 vision-language 模型融合方面，推荐使用 [BLIP-2](https://arxiv.org/abs/2301.12597) 来连接预训练的图像编码器和语言模型，更多细节见 [transformers 文档](https://huggingface.co/docs/transformers/en/model_doc/blip-2)。

**Embedding Precision with Quantization**: Sentence Transformers 的 [Embedding Quantization](https://huggingface.co/blog/embedding-quantization) 在不损失检索准确性的情况下，带来了显著的搜索速度提升。

**Catering to the German Learners**: 一个名为 *Hans* 的 GPT 驱动的德语学习工具承诺为德语学习者提供增强的用户体验，现已在 GPT Store 上线。

**All-MiniLM-L6-v2 Download Dilemma**: 一位用户在下载和训练 **all-MiniLM-L6-v2 模型** 时寻求帮助，强调了社区支持对模型实现的重要性。

**Revolutionizing Decision-Making with Langchain**: Medium 上的一篇文章认为 Langchain 是一种改变语言 Agent 解决问题方式的变革性方法，详见 [Medium](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1)。

**Diving Into Data's Importance**: 一篇分享的 [arXiv 论文](https://arxiv.org/pdf/2212.03533.pdf) 强调了数据作为潜在关键影响因素的重要性，提醒我们高质量数据的不可替代价值。

**NEET/JEE Data Quest**: 正在寻求 NEET/JEE 考试的数据集，用于训练 MCQ 答案生成器，这标志着 AI 技术与教育资源的交汇。

**AI on the Forefront**: Recurrent Neural Notes 通讯讨论了 AI 的潜在局限性，可能对 AI 的未来能力提供细致入微的见解，详见 [Substack](https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true)。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**Twitter 上的 Human-LlamaIndex 工作流预览**：推出了一套新模板，旨在简化人类与 **LlamaIndex Agents** 之间的交互，计划降低对用户的侵入性。详情和预览已在 [Twitter](https://t.co/Z16QPCWFmG) 上分享。

**将自定义 LLMs 集成至 LlamaIndex**：Leonie Monigatti 详细介绍了将自定义语言模型 (LLMs) 整合到 **LlamaIndex** 的过程，相关说明可在 [LinkedIn](https://t.co/DBjXGkLFkg) 查看。

**为 PDF 构建 RAG Agent 指南**：Ashish S. 发布了一篇关于创建基于 **LlamaParse** 的 PDF 文件 RAG 流程教程，可通过此 [推文](https://t.co/vIANM2Byel) 查看全文。

**新版 LlamaIndex Python 文档发布**：**LlamaIndex** 更新了其 Python 文档，更好地展示了示例 Notebook，改进了搜索功能并优化了 API 布局，已在 [Twitter 帖子](https://t.co/FAuBj5gnCC) 中宣布。

**LlamaIndex 社区解决集成与文档挑战**：社区讨论重点包括与 **Merlin API** 和 **LocalAI** 的各种集成、对 LlamaIndex 评估流程逻辑的询问、v0.10 更新后冲突的文档、对多 Agent 聊天机器人示例的需求，以及将 Python 函数转换为 LlamaIndex 工具。用户交流了相关资源，包括多个 [文档](https://github.com/mudler/LocalAI) 链接和 [GitHub 代码示例](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py)。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **用于视频处理的 Whisper**：社区成员正在寻找与 OpenAI 的 Whisper 相当的视频处理工具，建议包括 [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba)、Twelve Labs 和 [videodb.io](https://videodb.io)。

- **OpenAI 的 Sora 在创意人士中获得关注**：OpenAI 推出的 **Sora** 获得了艺术家的积极反馈，展示了该工具在生成写实和富有想象力的视觉效果方面的多功能性。

- **Google 的 AI 服务令人困惑**：讨论揭示了 Google AI Studio 与 Vertex AI 之间的混淆，特别是前者新推出的 100 万 Token 上下文 API，并将其与 OpenAI 的模型部署 API 进行了比较。

- **AI 可穿戴设备日益普及**：在关于 AI 可穿戴设备兴起的对话中讨论了开源 AI 可穿戴设备 ALOHA 项目。另一款 AI 可穿戴设备 [Compass](https://x.com/itsmartynask/status/1771890769865187648) 开始预售，表明人们对本地化、个性化 AI 解决方案的浓厚兴趣。

- **利用 LLMLingua 提升 LLM 效率**：微软的 LLMLingua 作为一个有前景的工具被分享，用于压缩大语言模型 (LLMs) 中的 Prompt 和 KV-Cache，在性能损失极小的情况下实现了显著的压缩率。

- **播客形式的 AI 内部讨论**：一条 [推文](https://twitter.com/swyx/status/1771255525818397122) 强调的一期播客提供了对各大 AI 公司的见解，引发了 AI 社区的兴趣。

- **发现 AI 去中心化（Unbundling）趋势**：[latent.space](https://latent.space/p/feb-2024) 上的一篇文章讨论了 ChatGPT 的去中心化，表明随着通用模型用户增长停滞，专业化 AI 服务正变得越来越受欢迎。

- **论文俱乐部的小插曲**：llm-paper-club-west 在 Discord 上遇到了发言权限的技术困难，导致会议转至 Zoom，这提高了人们对简化未来在线聚会访问流程的意识。

- **AI 实战俱乐部的创意与音乐流动**：该俱乐部就 Tensor 操作、LLM 编码最佳实践进行了热烈讨论，并自发分享了 Slono 在 [Spotify](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA) 上唤起夜晚宁静的音乐。他们还发布了关于 AI 主题后续会议的 [日程表](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**GaLore 优化引发辩论**：关于 [GaLore optimizer](https://github.com/jiaweizzhao/GaLore/issues/6) 的讨论强调了其节省 VRAM 的能力，但也提出了由于“粗糙度（coarseness）”可能导致过度训练的问题。一些工程师渴望测试 GaLore，特别是考虑到新发布的 **Mistral v0.2 Base Model**，它现在拥有 32k 的 context window。

**低预算微调 Large Language Models**：围绕在 27gb 内存内微调 7b 模型的各种技术讨论浮出水面，重点关注了一个名为 [torchtune](https://github.com/pytorch/torchtune) 的 GitHub 仓库，它允许在没有 Huggingface 依赖的情况下进行高效微调。推荐查看一个特定的 [pull request](https://github.com/pytorch/torchtune/pull/527)，以了解需要少于 16GB RAM 的 full fine-tune 方法。

**TypeError 问题与帮助频道支持**：一名成员在处理 "examples/openllama-3b/qlora.yml" 中的 `TypeError` 时，被引导至专门的帮助频道 (#1111279858136383509) 以寻求专业解决建议。这体现了社区的协作环境，鼓励成员利用特定资源解决技术问题。

**医学模型发布困境**：在期刊评审期间是否公开分享医学模型的预印本（preprint）引发了关于早期披露权衡的讨论。这次对话强调了该领域战略性研究传播的重要性。

**开发者认可与商业合作的公开征集**：[CHAI 宣布为 LLM 开发者设立奖项](https://chai-research.typeform.com/chaiprize)，鼓励社区贡献；同时邀请企业秘密分享其 Axolotl 的应用案例，这暗示了真实世界使用场景叙述在推动 AI 技术发展中的价值。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **Midnight 70B 进军角色扮演领域**：*Midnight 70B* 模型专为故事讲述和 Roleplay 定制，继承了 Rogue Rose 和 Aurora Nights 的血统，现已上线并提供 25% 的折扣，在 [OpenRouter](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b) 上的价格为 **$0.009/1k tokens**。
- **OpenRouter 完善成本追踪工具**：OpenRouter 实现了先进的 **Usage Analytics** 功能用于实时成本追踪，并推出了 **Billing Portal** 以实现更高效的额度和发票管理。
- **Noromaid Mixtral 和 Bagel 价格调整**：运行 Noromaid Mixtral 和 Bagel 模型的费用不再包含折扣，Mixtral 的新价格定为 **$0.008/1k tokens**，而 Bagel 为 **$0.00575/1k tokens**。
- **Claude 3 & Grok**：在多模型讨论中，Claude 3 的自我审查版本因改进的过滤功能而受到关注；Grok 模型引发了辩论，其性能被认为优于其他高级替代方案，但成本过高。用户表达了对更长 context lengths 的偏好，并指出了 OpenRouter 与直接使用 API 之间在模型补全质量上的差异。
- **OpenRouter 遭受 DDoS 攻击及 API 响应问题**：OpenRouter 面临 DDoS 攻击导致服务不稳定，目前已解决；用户观察到 Perplexity 的引用数据并未按预期出现在 OpenRouter 的 API 响应中。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **PyTorch 在 MPS 上的困境**：改进 **PyTorch 中 MPS backend** 的持续努力正在进行中，自 2022 年 9 月以来，tensor 复制等显著问题一直是障碍。这项工作预计将提升本地模型测试和微调的性能。

- **LLM 训练中的 Token Blocking 争议**：一场关于**语言模型预训练中 token block 策略**的辩论引发了对重叠与非重叠序列优劣及其对模型效能影响的讨论，并涉及了 beginning-of-sentence tokens 的重要性。

- **AMD 驱动困境引发讨论**：AMD Radeon 与 Nvidia GPU 驱动的对比引发了辩论，焦点集中在驱动程序的不足以及 AMD 开源其驱动的可能性。一些参与者考虑了激进投资者采取行动促使 AMD 做出改变的潜力。

- **机器学习模型合并方法论**：新的**模型合并（model merging）方法**正在开发中，旨在超越 DARE 等现有技术，尽管这些仍处于实验阶段，需要进一步的测试和验证。

- **充满前景的新 ML 架构**：**[DenseFormer](https://arxiv.org/abs/2402.02622)** 和 **[Zigzag Mamba](https://arxiv.org/abs/2403.13802)** 等创新分别在 perplexity 和 diffusion model 内存占用方面提出了改进，而 **[DiPaCo](https://arxiv.org/abs/2403.10616v1)** 则为鲁棒的分布式模型训练提供了一种新颖方法。

- **SVM Kernel 之争**：结果表明，**sigmoid SVM kernel** 在 Pythia 的 input embeddings 上的表现优于 rbf、linear 和 poly 等其他 kernel。

- **N-gram 项目 "Tokengrams" 受到关注**：据报道，*Tokengrams* 项目现在可用于高效计算和存储来自文本语料库的 token n-grams，为研究人员提供了一个高效资源，详见 [GitHub - EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams)。

- **Chess-GPT 案例分析**：关于 **Chess-GPT** 的案例研究讨论了使用语言模型预测国际象棋走法并估算 Elo rating 的技术，以及使用 linear probes 验证计算的方法，详见 [Chess GPT Interventions](https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html)。

- **评估变异性令 AI 工程师担忧**：将 **Hugging Face transformers** 与 **Megatron-DeepSpeed 评估**进行对比时出现的**评估结果不一致性**引起了关注，有建议称应核实 *fused kqv multiplications* 中的 *bfloat16* 数值处理等实现细节是否导致了这种变异性。

- **Minecraft 作为 RL 测试场**：一个基于 **Minecraft 的强化学习（Reinforcement Learning）环境**已在 [GitHub - danijar/diamond_env](https://github.com/danijar/diamond_env) 发布，结合对 [Voyager](https://github.com/MineDojo/Voyager/issues/149) 项目的讨论，强调了游戏在 AI 模型协作研究中的应用。

- **多模态嵌入空间探索**：社区对**多模态嵌入空间（multimodal embedding spaces）的理论工作**表现出兴趣，并提供了关于 Stable Diffusion 亚文化如何处理与 **IMG2IMG** 工作流一致的 embeddings 的见解。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**当 Discord 失效时，Meet 顶上**：GTC 活动期间的技术故障导致有人建议未来的讲座默认使用语音频道，因为 Discord 的 stage 频道存在屏幕共享问题。一位不满的成员提议由于 Discord 直播的不稳定性，未来应切换到 Google Meet。

**CUDA 性能分析 (Profiling)**：针对深入研究 CUDA 的工程师，分享了一个[关于如何在 PyTorch 中对 CUDA kernel 进行 profiling 的讲座](https://www.youtube.com/watch?v=LuhJEEJQgUM)，并附带了[配套幻灯片](https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharing)和 [GitHub 代码仓库](https://github.com/msaroufim/cudamodelecture1)。当 PyTorch 的速度不足以满足性能增益需求时，CUDA 编程就成了必需。

**Triton 的棘手细节**：关于 **Triton 性能问题**的讨论非常突出，成员们被警告 **Triton 操作未来可能会被逐步淘汰**。建议在 `torchao` 仓库中建立一个新的 prototype 文件夹，以便在继续支持 Triton 的同时，协作设计高效 kernel 使用的 API。

**稀疏性与分解的优雅结合**：研究员 [Lukas Gianinazzi](https://arxiv.org/abs/2402.19364) 和 Alexandros Nikolaos Ziogas 在 **Arrow Matrix Decomposition** 论文中介绍了一种分布式稀疏矩阵乘法的新方法，其实现代码可在 [GitHub](https://github.com/spcl/arrow-matrix) 上获得。

**Blackwell GPU 对着镜头微笑**：成员们讨论了新的 **Blackwell GPU**，重点提到了一条幽默看待 GPU“笑脸”图案的推文。在关于 **CUTLASS 库**的 GitHub 讨论被提及后，大家对未公开的 NVIDIA Developer Discord 服务器进行了推测。社区还触及了深度学习中的数据类型标准化，指出 Google 未参与近期的标准联盟，且新型浮点数缺乏 **IEEE 标准**。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

**Mistral 的新 7B 模型抢占风头**：Mistral AI 在 [@cerebral_valley hackathon](https://x.com/alexreibman/status/1771608346635751541?s=46) 上**随意发布了一个新模型**——Mistral 7B v0.2 Base。模型详情（包括微调指南）可在[此处](https://x.com/mistralailabs/status/1771670765521281370?s=46)获取，不过正如 [@natolambert](https://twitter.com/MistralAI) 所指出的，本次发布未提供 magnet 链接。

**Stability AI 的大震荡**：CEO Emad Mostaque 从 Stability AI **辞职**，暗示他未来的重点将放在 **#DecentralizedAI**。在关于内部斗争以及 Stability AI 对 AI 学术界贡献本质的讨论中，社区对他任期的影响和方向表达了复杂的情绪。

**Nemo 互操作性的寻求者**：出现了关于转换和包装 **Nemo checkpoint** 以兼容 Hugging Face 的问题，突显了机器学习模型互操作性方面的技术挑战。

**AI 的伦理钢丝**：
- 基于[此处](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw)的讨论，引发了关于在 Reinforcement Learning 中创建“通用 Agent”在*实践和根本上是否可行*的辩论。
- 该频道还讨论了 FTC 对 Apple 的反垄断诉讼，Nathan Lambert 指出公众对反垄断法规存在误解，并引用相关推文支持其观点。

**二月重磅 AI 对话**：与 Anthropic CEO 和 Mistral CEO 的精彩访谈引起了关注，例如这场[“炉边谈话”](https://youtu.be/sQpeIuymJZ8)以及关于 Amodei 对 AI 行业预测的[讨论](https://www.youtube.com/watch?v=gAaCqj6j5sQ)。此外，Latent Space 总结二月关键 AI 进展的回顾可以在[此处](https://www.latent.space/p/feb-2024)找到。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **AI 以更低成本提供更多**：提出了一种创新方法，通过在达到长度限制时发起后续请求，绕过 **GPT-4-Turbo** 的 4k 输出 Token 限制，使模型能够无缝地继续生成内容。
- **Bedrock 遇上 Python**：一份关于使用 **Bedrock with Python** 的指南已经发布，展示了实际的集成技术。感兴趣的工程师可以在[这里](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be)深入阅读该指南。
- **使用 SimplyAnalyze.ai 分析 LLM 对话**：宣布推出 **SimplyAnalyze.ai**，它与 LangChain 配合使用，可剖析各业务部门的 LLM 对话。要加入免费开发者预览版，工程师可以访问 [SimplyAnalyze 的网站](https://simplyanalyze.ai/)。
- **利用 LangChain 进行决策**：分享了一篇详细介绍在 **Agent Tree Search** 中使用 LangChain 的文章，旨在通过语言模型促进更复杂的决策过程。工程师可以在[这里](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1)阅读更多内容。
- **具有记忆和解析能力的升级版聊天机器人**：对**本地角色 AI 聊天机器人**进行了增强，改进了 CSV 和 NER 解析等功能。要查看升级后的功能，GitHub 仓库可在[这里](https://github.com/ossirytk/llama-cpp-chat-memory)找到。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

**房地产匹配出错**：围绕 **GPT4.Turbo** 误解房产面积要求的问题展开了讨论，尽管要求是 2,000 - 4,000 平方英尺，但却建议了一处 17,000 平方英尺的房产。建议使用简单的基于 CSV 的数据库过滤器，而不是复杂的 LLM，这引发了关于常见误区的对话，并链接到了 **Jason Liu** 关于 LLM 中可能过度依赖 embedding search 的资源。

**对 Token 限制的挫败感**：参与者对 **Anthropic** 每天 **1M tokens** 的速率限制表示不满，认为 **200k context window** 不足。讨论了 **Bedrock 月费模式**作为潜在替代方案，同时建议 Anthropic 的 **$500 scale plan** 能为广泛使用提供更便捷的途径。

**寻找优质解释资源**：社区被征集关于高级 LLM 主题的顶级**解释资源**，特别要求针对 RHLF 等主题提供高质量、清晰的内容，而不是大量的博客集合。*Exa.ai* 被建议作为深入研究 LLM 相关主题的有益资源。

**对代码质量的简短呼吁**：在 **#[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** 频道中，一位用户用一句简洁而引起共鸣的话感叹编写高质量代码的难度。

**GPT-3.5-0125 领先**：**GPT-3.5-0125** 因其相较于之前模型的显著性能提升而受到赞赏，正如一位用户的对比测试所观察到的那样，提升了其作为 LLM 领域中特别先进迭代的地位。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **招募 AI 复仇者**：**Youth Inquiry Network** 和 **Futracode** 正在合作开发一种机器学习算法，从现有数据库中推荐最佳研究课题。他们正在招募 **Web 开发者、数据分析师以及 AI & ML 专家**来支持这一事业。
  
- **贡献技能，赢得荣誉**：志愿者不仅能通过一份亮眼的作品集来推进职业生涯，还能获得**证书和两封专业推荐信**。参与者还可以保留开发的 **ML 算法代码**用于个人或商业用途。

- **为拯救世界者提供灵活时间**：他们保证这个开创性项目的**投入时间灵活**——非常适合日程繁忙的超级英雄。招募人员可以通过简单的“感兴趣”来绕过官僚程序，开始他们的任务。

- **神秘教育改革文档发布**：一位未具名成员分享了一个 [Google Docs 链接](https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing)，讨论 **Post-AGI 教育改革**，可能暗示了一种面向未来的 AI 教育范式。

- **元调解时刻**：具有讽刺意味的是，一位**管理员在呼吁调解时经历了自我觉醒**，提醒我们即使是机器人也会忘记自己的协议。

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **LLM 与 Ollama 的对决**：成员们澄清了 **`llm` 如何与模型交互**（例如 **Mistral**），通过设置 API 端点，然后由 `ollama` 执行。`ollama` 允许本地模型执行，使模型可以通过本地 HTTP API 端点访问。
  
- **AI 辅助的技术提交**：工具 **[AICommits (GitHub - Nutlope/aicommits)](https://github.com/Nutlope/aicommits)** 旨在帮助使用 AI 编写 git commit 消息，因其实用性而受到赞赏，并有人请求增加 commit 的 emoji 标准等额外功能。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **AI 策划噱头**：一个 AI 根据 YouTuber **Mr. Beast** 的大胆冒险制作了一本独特的食谱，引起了小组的兴趣。这一创意应用在一段 [YouTube 视频](https://www.youtube.com/watch?v=Nc5Yk0XXgP8)中展示，将烹饪艺术与 machine learning 相结合，产生了异想天开的效果。
- **寻找德国技术专家**：一位社区成员正在寻找关于 deep learning 和 AI 的**德语资源**，表示希望用母语深入研究技术内容。这一请求描绘了 AI 社区中全球化、多语言的兴趣图景。



---

# PART 2: 频道详细摘要与链接



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1220621626790514709)** (1195 条消息🔥🔥🔥): 

- **Stable Diffusion 咨询与协助**：成员们讨论了 Stable Diffusion 模型的各个方面和用途，包括性能、采样器设置和 ControlNet 模型。用户还就处理错误以及在不同系统（尤其是使用 AMD GPU 的系统）上设置 AI 交换了指导意见。

- **探索 SD3 及其替代方案**：对话主题包括 SD3 预期的发布窗口和潜在改进，以及与 SDXL 等其他产品和 AI 生成视频潜力的比较。

- **对在线 AI 服务的反馈**：聊天涉及了对 Civitai 和 Suno 等在线 AI 图像生成服务的局限性和挫败感，特别指出了内容限制问题以及对显示内容类型的偏好。

- **关于 AI 伦理与监管的辩论**：成员们辩论了对 AI 技术使用进行监管的必要性，以及开源模型与专有模型的重要性。有人担心监管可能会扼杀创新和可访问性。

- **技术故障排除与学习**：寻求有关模型安装和使用技术问题帮助的新成员被引导至社区内的支持频道和专家。资深成员旨在提供指导和资源，同时向新手建议与 AI 图像生成相关的学习曲线。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>

<li><a href="https://seaart.ai>">未找到标题</a>: 未找到描述</li><li><a href="https://artificialguy.com">ArtificialGuyBr</a>: 未找到描述</li><li><a href="https://sakana.ai/evolutionary-model-merge/">未找到标题</a>: 未找到描述</li><li><a href="https://imgur.com/H4PmCXo">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的模因、有趣的 GIF、鼓舞人心的故事、病毒式视频等来振奋你的精神...</li><li><a href="https://comfyanonymous.github.io/ComfyUI_examples/stable_cascade/">Stable Cascade Examples</a>: ComfyUI 工作流示例</li><li><a href="https://xkcd.com/435/">Purity</a>: 未找到描述</li><li><a href="https://siliconangle.com/2024/03/24/emad-mostaque-resigns-ceo-troubled-generative-ai-startup-stability-ai/">Emad Mostaque 辞去陷入困境的生成式 AI 初创公司 Stability AI 的 CEO 职务 - SiliconANGLE</a>: Emad Mostaque 辞去陷入困境的生成式 AI 初创公司 Stability AI 的 CEO 职务 - SiliconANGLE</li><li><a href="https://huggingface.co/thibaud">thibaud (Thibaud Zamora)</a>: 未找到描述</li><li><a href="https://tenor.com/view/dune-oil-gif-2770573093912411630">Dune Oil GIF - Dune Oil - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://siliconangle.com/2024/03/18/nvidias-newest-cloud-service-promises-accelerate-quantum-computing-simulations-beef-post-quantum-security/">Nvidia 最新的云服务承诺加速量子计算模拟 - SiliconANGLE</a>: Nvidia 最新的云服务承诺加速量子计算模拟 - SiliconANGLE</li><li><a href="https://civitai.com/models/359999">CinematicRedmond - Cinematic Model for SD XL - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Cinematic.Redmond 来了！感谢 Redmond.AI 提供的 GPU 时间让我能够制作这个模型！这是一个经过微调的电影感模型...</li><li><a href="https://huggingface.co/spaces/artificialguybr/artificialguybr-demo-lora">Artificialguybr Demo Lora - artificialguybr 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://app.suno.ai/song/89a68ee1-899e-44c7-a8d8-a1c011376f3a">Neon City Lights | Suno</a>: 日语人声、休闲爵士、J-pop、慢节奏歌曲。使用 Suno 聆听并创作你自己的作品。</li><li><a href="https://x.com/chrlaf/status/1772226646365397311?s=20">Christian Laforte (@chrlaf) 的推文</a>: @thibaudz 谢谢 Thibaud，正如我在别处写的那样，计划没有改变，我们仍在努力改进模型以实现开放发布。包括源代码和权重。</li><li><a href="https://tenor.com/view/ok-then-um-well-ok-then-wtf-gif-23665207">Ok Then Um GIF - Ok Then Um Well Ok Then - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://civitai.com/models/310571/boring-reality">Boring Reality - BoringReality_primaryV4.0 | Stable Diffusion LoRA | Civitai</a>: 注意：请阅读下文了解如何使用这些 LoRA。如果单独且按原样使用，它们不太可能产生良好的效果。这个模型实际上...</li><li><a href="https://civitai.com/models/38784/controlnet-11-models">ControlNet 1.1 Models - Tile (e) | Stable Diffusion Controlnet | Civitai</a>: 停止！这些模型不用于提示词/图像生成。这些是 ControlNet 扩展所需的全新 ControlNet 1.1 模型，已转换...</li><li><a href="https://civitai.com/models/359999/cinematicredmond-cinematic-model-for-sd-xl">CinematicRedmond - Cinematic Model for SD XL - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Cinematic.Redmond 来了！感谢 Redmond.AI 提供的 GPU 时间让我能够制作这个模型！这是一个经过微调的电影感模型...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bnjm3i/stable_diffusion_3/">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://stable-diffusion-art.com/glossary/">Stable Diffusion Glossary - Stable Diffusion Art</a>: 对 Stable Diffusion 中的术语感到困惑？你并不孤单，我们会提供帮助。此页面包含你在 Stable Diffusion 中需要了解的所有关键术语。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1bmpqeh/stabilityai_is_alive_and_will_live_there_were/">Reddit - 探索一切</a>: 未找到描述</li><li><a href="https://civitai.com/models/261999/drone-shot-above-xl-lora">Drone Shot &quot;Above&quot; XL LoRA - v1.0 | Stable Diffusion LoRA | Civitai</a>: 在提示词中使用 "above"。最适用于大场景，而非单个物体或角色。</li><li><a href="https://github.com/virattt/financial-agent">GitHub - virattt/financial-agent: 一个完全使用 LangChain 构建的金融 Agent！</a>: 一个完全使用 LangChain 构建的金融 Agent！通过在 GitHub 上创建账户，为 virattt/financial-agent 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=1sN6U5dV1Os">AI Dance</a></li>

Animation - [ NEXT GEN ] - Stable Diffusion | ComfyUI</a>: 这段 AI 动画是使用 AnimateDiff 和 ControlNet 节点完成的，没有使用任何女孩 LORA。自上一个 AI 舞蹈视频以来，它在一致性方面有了重大改进...</li><li><a href="https://arstechnica.com/information-technology/2023/11/unauthorized-david-attenborough-ai-clone-narrates-developers-life-goes-viral/">未经授权的 “David Attenborough” AI 克隆体为开发者的生活配音，走红网络</a>：“我们观察到复杂的智人正在进行补水仪式。”</li><li><a href="https://huggingface.co/artificialguybr">artificialguybr (ArtificialGuy/JV.K)</a>: 未找到描述</li><li><a href="https://civitai.com/models/136070?modelVersionId=267507">ControlNetXL (CNXL) - bdsqlsz-depth | Stable Diffusion Checkpoint | Civitai</a>: bdsqlsz : canny | depth | lineart-anime | mlsd-v2 | normal | openpose | recolor | segment | segment-v2 | sketch | softedge | t2i-color-shuffle | ti...</li><li><a href="https://www.lesswrong.com/tag/rokos-basilisk">Roko&#x27;s Basilisk - LessWrong</a>: Roko’s basilisk 是用户 Roko 于 2010 年在 Less Wrong 社区博客上提出的一个思想实验。Roko 利用决策理论中的观点认为，一个足够强大的 AI Agent 将会...</li><li><a href="https://www.youtube.com/watch?v=w3vXaK3JC8E">谁去告诉她……</a>: 购买 T 恤支持频道：http://www.clownplanetshirts.com 不要忘记订阅。点击铃铛以获取最新视频动态。观看...</li><li><a href="https://www.youtube.com/watch?v=1CIpzeNxIhU">AI 图像生成器的工作原理 (Stable Diffusion / Dall-E) - Computerphile</a>: AI 图像生成器影响巨大，但它们是如何创建如此有趣的图像的？Dr Mike Pound 解释了其中的原理。缩略图部分由...创建。</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika 是一个 Agentic AI 软件工程师，能够理解高层级的人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标。Devika 旨在成为 Cognition AI 的 Devin 的竞争性开源替代方案。</a>: Devika 是一个 Agentic AI 软件工程师，能够理解高层级的人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/15c3rf6/sdxl_resolution_cheat_sheet/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=5mA_tzJije0">报复社会（Revenge of the nerds） - John Goodman 演讲</a>: 鼓舞人心的内容！</li><li><a href="https://github.com/LykosAI/StabilityMatrix/blob/main/README.md">StabilityMatrix/README.md at main · LykosAI/StabilityMatrix</a>: Stable Diffusion 的多平台包管理器 - LykosAI/StabilityMatrix</li><li><a href="https://replicate.com/artificialguybr/cinematic.redmond">artificialguybr/cinematic.redmond – 在 Replicate 上通过 API 运行</a>: 未找到描述</li><li><a href="https://youtu.be/RbId7zb8mqE?si=fVBYGBVUpvbYn3H3">Eyaura: Give Me A Soul. 专辑: T.B.D.</a>: G STRING - STEAM:https://store.steampowered.com/app/1224600/G_String/ G STRING DISCORD - 官方:https://discord.gg/fUuDyx7uYe G STRING DISCORD - 杂项:https:/...</li><li><a href="https://forms.gle/DqMih6Z9wCmTRvZN8">名字原型</a>: 所以，我通过在 Midjourney 中使用提示词 "a photo of (name) --style raw" 制作了这些面孔。我不禁注意到，每当你遇到一个新朋友时，他们似乎都有一个与之关联的名字...</li><li><a href="https://civitai.com/models/339604/how-to-generate-multiple-different-characters-mix-characters-andor-minimize-color-contamination-or-regional-prompt-adetailer-and-inpaint-or-my-workflow">如何生成多个不同的角色、混合角色和/或最小化颜色污染 | Regional Prompt, Adetailer, 和 Inpaint | 我的工作流 - 2. Adetailer | Stable Diffusion 工作流 | Civitai</a>: 如何生成多个不同的角色、混合角色和/或最小化颜色污染 | Regional Prompt, Adetailer, 和 Inpaint | 我的工作流...</li><li><a href="https://github.com/lllyasviel/stable-diffusion-webui-forge/discussions/169#discussioncomment-8428689">Forge 并没有使用 ComfyUI 作为后端 · lllyasviel/stable-diffusion-webui-forge · Discussion #169</a>: 最近有些人开始散布关于 Forge 使用 ComfyUI 作为后端的虚假信息。这是错误的，对社区有害，也对我们工程团队的努力有害。后端...
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1220603138852257793)** (1009 条消息🔥🔥🔥):

- **探索 SFTTrainer 中的指标**：一位用户寻求关于在 `SFTTrainer` 中使用基于生成的指标进行验证的建议，并引用了 [GitHub issue](https://github.com/huggingface/trl/issues/862#issuecomment-1896074498) 中的一种变通方法。他们不清楚 `compute_metrics` 函数中接收到的 preds，以及在使用 LoRA 适配器进行微调时 `SFTTrainer` 如何计算 loss。

- **聊天机器人模型推理的硬件需求**：一位用户询问如何确定运行 LLM 模型（如 [Nous-Capybara-34B-GGUF](https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF)）的硬件需求，另一位用户建议参考另一个 HH 模型的 [讨论](https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/discussions/2) 以获取估算值，并澄清模型需求会根据量化和 prompt 的不同而有所变化。

- **模型差异与量化**：关于两个版本的 *Mistral* 模型之间差异的咨询引出了一番解释，即像 [这里](https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit) 的 4-bit 模型下载速度更快，但准确率会略有下降。

- **Mistral 及其营销策略**：围绕 Mistral 的模型发布实践展开了讨论，一位成员认为其做法不同寻常，因为他们没有在 Hugging Face 上上传其基础模型，并且还发生了在 4chan 上泄露的非常规事件。

- **关于计算机科学教育的辩论**：鉴于 LLM 现在能够编写代码，一场关于计算机科学学位重要性的激烈讨论正在进行。对话转向了各种编程语言、内存安全以及来自不同大学学位的价值。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://inflection.ai/inflection-2-5">Inflection-2.5: meet the world&#x27;s best personal AI</a>: 我们是一家为每个人打造个人 AI 的 AI 工作室。我们的首款 AI 名为 Pi（代表个人智能），是一款提供支持且富有同理心的对话式 AI。</li><li><a href="https://gpt4all.io/index.html">GPT4All</a>: 免费、本地且注重隐私的聊天机器人</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/152334H/miqu-1-70b-sf">152334H/miqu-1-70b-sf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co">Hugging Face – The AI community building the future.</a>: 未找到描述</li><li><a href="https://tenor.com/view/funny-very-sloth-slow-gif-15401812">Funny Very GIF - Funny Very Sloth - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sloth-slow-stamp-gif-8535595">Sloth Slow GIF - Sloth Slow Stamp - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/sloth-smile-slow-smooth-hd-neuron-activation-gif-24950071">Sloth Smile Slow GIF - Sloth Smile Slow Smooth - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2-bnb-4bit">unsloth/mistral-7b-v0.2-bnb-4bit · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth/mistral-7b-v0.2">unsloth/mistral-7b-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/CodeLlama-34B-Instruct-GGUF/discussions/2">TheBloke/CodeLlama-34B-Instruct-GGUF · [AUTOMATED] Model Memory Requirements</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/Nous-Capybara-34B-GGUF">TheBloke/Nous-Capybara-34B-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://docs.gpt4all.io/gpt4all_python.html">Generation - GPT4All Documentation</a>: 未找到描述</li><li><a href="https://github.com/InflectionAI/Inflection-Benchmarks">GitHub - InflectionAI/Inflection-Benchmarks: Public Inflection Benchmarks</a>: 公开的 Inflection 基准测试。通过在 GitHub 上创建账号，为 InflectionAI/Inflection-Benchmarks 的开发做出贡献。</li><li><a href="https://www.quantamagazine.org/how-quickly-do-large-language-models-learn-unexpected-skills-20240213/">How Quickly Do Large Language Models Learn Unexpected Skills? | Quanta Magazine</a>: 一项新研究表明，所谓的“涌现能力”实际上是逐渐且可预测地发展的，这取决于你如何衡量它们。</li><li><a href="https://www.infoworld.com/article/3713203/white-house-urges-developers-to-dump-c-and-c.html">White House urges developers to dump C and C++</a>: 拜登政府呼吁开发者采用内存安全编程语言，并远离那些会导致缓冲区溢出和其他内存访问漏洞的语言。</li><li><a href="https://github.com/huggingface/trl/issues/862#issuecomment-1896074498">Compute metrics for generation tasks in SFTTrainer · Issue #862 · huggingface/trl</a>: 你好，我想在 SFTTrainer 中包含一个基于自定义生成的 compute_metrics（例如 BLEU）。但是，我遇到了困难，因为：输入到 compute_metrics 的 eval_preds 包含一个 .predicti...</li><li><a href="https://huggingface.co/alpindale/Mistral-7B-v0.2-hf">alpindale/Mistral-7B-v0.2-hf · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/GAIR/lima">GAIR/lima · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://lightning.ai/pages/community/tutorial/accelerating-large-language-models-with-mixed-precision-techniques/">Accelerating Large Language Models with Mixed-Precision Techniques - Lightning AI</a>: 由于巨大的计算需求和内存占用，训练和使用大语言模型 (LLMs) 的成本很高。本文将探讨如何利用低精度格式来增强...</li><li><a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab ( IST Austria Distributed Algorithms and Systems Lab)</a>: 未找到描述</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://github.com/PygmalionAI/aphrodite-engine/tree/main/tests/benchmarks">aphrodite-engine/tests/benchmarks at main · PygmalionAI/aphrodite-engine</a>: PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号，为 PygmalionAI/aphrodite-engine 的开发做出贡献。</li><li><a href="https://

github.com/PygmalionAI/aphrodite-engine">GitHub - PygmalionAI/aphrodite-engine: PygmalionAI 的大规模推理引擎</a>: PygmalionAI 的大规模推理引擎。通过在 GitHub 上创建账号来为 PygmalionAI/aphrodite-engine 的开发做出贡献。</li><li><a href="https://ev01.sx/">在线观看电影和免费电视剧流媒体 - ev01.net</a>: 数据库中超过 250,000 部电影和电视剧的快速免费流媒体。无需注册，无需付费，100% 免费全高清流媒体</li><li><a href="https://huggingface.co/argilla">argilla (Argilla)</a>: 未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/tree/master">GitHub - ggerganov/llama.cpp: C/C++ 中的 LLM 推理</a>: C/C++ 中的 LLM 推理。通过在 GitHub 上创建账号来为 ggerganov/llama.cpp 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1220627573776584816)** (58 条消息🔥🔥): 

- **本地机器上的 Kernel 难题**：讨论了用户本地机器上需要重启 Kernel 的问题，暗示这可能与内存有关。分享了一条关于 32-bit 以及需要重启才能消除的错误消息，并讨论了是否为 Out Of Memory (OOM)，但用户确认在重启 Kernel 后机器运行正常。

- **迎接光纤和大型模型**：一位用户兴奋地报告称，为了支持更大的模型，他们升级了光纤和新的 2TB WD 黑盘。用户对当前的性能表现以及未来可能的硬件升级充满热情。

- **ORPO 在 AI 社区引起轰动**：关于 ORPO (**Off-policy Reinforcement learning with Pretrained Overparametrized Models**) 的讨论非常热烈，用户们讨论了它的集成以及对模型性能的提升。提供了指向 [arXiv](https://arxiv.org/pdf/2403.07691.pdf) 原始论文的链接。

- **Unsloth 紧跟 TRL 步伐**：关于 ORPO 的讨论，用户确认如果 TRL (**Transformer Reinforcement Learning**) 支持该功能，Unsloth AI 也应该支持。提到了从 Unsloth 到 TRL 的优化和补丁，并鼓励大家在新的集成出现问题时进行分享。

- **Transformer 模型的新工具包**：链接了一个名为 transformer-heads 的有趣工具包。它专为 Transformer 模型附加、训练、保存和加载新的 heads 而设计，可在 [GitHub](https://github.com/center-for-humans-and-machines/transformer-heads) 上获取。

**提到的链接**：<a href="https://github.com/center-for-humans-and-machines/transformer-heads">GitHub - center-for-humans-and-machines/transformer-heads: 用于 Transformer 模型新 heads 附加、训练、保存和加载的工具包</a>：用于 Transformer 模型新 heads 附加、训练、保存和加载的工具包 - center-for-humans-and-machines/transformer-heads

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1220638996661538886)** (317 条消息🔥🔥): 

- **理解 Unsloth 的快速反量化 (Fast Dequantizing)**：Unsloth AI 在 `fast_lora.py` 中的 'fast_dequantize' 被指出针对速度进行了优化，与 `bitsandbytes` 相比减少了内存拷贝。
- **使用 Unsloth 排除故障并更新 Mistral**：由于 Gemma GGUF 的问题，建议一名成员升级 Unsloth，并提供了升级命令。指出问题不仅存在于 GGUF，还存在于合并（merging）过程中。
- **解决推理中的 Token 循环问题**：讨论了一个报告的问题，即转换为 GGUF 的模型（特别是使用 Mistral 时）在响应过程中开始循环重复 `<s>`。Unsloth 的维护者建议检查 `tokenizer.eos_token`。
- **合并多个数据集进行微调**：建议将多个数据集连接成一个文本字符串，进行处理，然后附加在一起进行训练。为此目的，可以合并来自不同数据集的增强指令和响应。
- **需要澄清微调参数**：有关于在微调期间通过 `max_steps` 控制 Epochs 的咨询，建议改为设置 `num_train_epochs`。此外，还提到由于填充（padding）原因，增加 `max_seq_length` 可能会导致更高的内存消耗。
<div class="linksMentioned">

<strong>提到的链接</strong>：

</div>

<ul>
<li>
<a href="https://goddard.blog/posts/clown-moe/">面向小丑（在马戏团）的混合专家模型 (Mixture of Experts for Clowns)</a>: 未找到描述</li><li><a href="https://huggingface.co/HirCoir/Claud-mistral-7b-bnb-4bit-GGUF/tree/main">HirCoir/Claud-mistral-7b-bnb-4bit-GGUF at main</a>: 未找到描述</li><li><a href="https://huggingface.co/HirCoir/Claud-openbuddy-mistral-7b-v19.1-4k">HirCoir/Claud-openbuddy-mistral-7b-v19.1-4k · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2#instruction-format">mistralai/Mistral-7B-Instruct-v0.2 · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2#scrollTo=LjY75GoYUCB8&line=8&uniqifier=1.">Google Colaboratory</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L488">bitsandbytes/bitsandbytes/autograd/_functions.py at main · TimDettmers/bitsandbytes</a>: 通过针对 PyTorch 的 k-bit 量化实现易用的大型语言模型。 - TimDettmers/bitsandbytes</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://youtu.be/_GkHZQYFOGM?si=taLl7f-TNdJta_W8">针对记忆化的 LLM 微调</a>: ➡️ 高级微调仓库（含记忆化脚本）: https://trelis.com/advanced-fine-tuning-scripts/ ➡️ 一键微调与推理模板: ht...</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调</a>: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调</a>: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/unslothai/unsloth.git">GitHub - unslothai/unsloth: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调</a>: 快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1220821099466592318)** (33 条消息🔥): 

- **Unsloth 集成问题的迹象**：用户表达了在使用 *Ollama 模板* 时遇到的问题，特别是针对 q4 等不同量化模型，导致结果较差。
- **GPT4All 诊断**：一位用户正在 **GPT4All** 上运行测试以解决问题，并被建议不要转义反引号，并尝试不同的 *quant*（量化）大小。
- **Q8 模型版本展现前景**：经过一番交流，一位用户确认 **Huggingface** 上的 Q8 模型似乎运行正常。
- **Sappha-2b-v3 引起轰动**：一款基于 Gemma-2b 并使用 Unsloth 微调的新模型 [sappha-2b-v3](https://huggingface.co/Fizzarolli/sappha-2b-v3)，在多个基准测试中表现优于当前模型，引发了对其能力的讨论。
- **对新模型的关注达到顶峰**：用户对新发布的模型表现出兴奋，并分享了他们表现最好的模型链接，例如 [MasherAI-v6-7B](https://huggingface.co/mahiatlinux/MasherAI-v6-7B)，同时寻求有关所用微调过程的信息。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/mahiatlinux/MasherAI-v6-7B">mahiatlinux/MasherAI-v6-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Fizzarolli/sappha-2b-v3">Fizzarolli/sappha-2b-v3 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md">ollama/docs/modelfile.md at main · ollama/ollama</a>: 快速上手并运行 Llama 2, Mistral, Gemma 以及其他大型语言模型。 - ollama/ollama
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1221127411723403406)** (29 条消息🔥):

- **PyTorch 全量微调无需巨额显存开销**：PyTorch 用户请注意：一个新的 [pull request](https://github.com/pytorch/torchtune/pull/527) 允许在少于 16GB RAM 的情况下进行全量微调（Full Finetuning），这使得拥有消费级 GPU 的用户更容易上手。
- **LoftQ 登场**：在人工智能优化新闻中，LoftQ 已包含在 [Hugging Face PEFT v0.10.0 版本](https://github.com/huggingface/peft/releases/tag/v0.10.0) 中，这为大型模型带来了增强的微调能力。
- **ORPO 的多轮对话训练挑战**：目前正在讨论 ORPO 的多轮对话训练，建议通过引入一种能有效处理多轮对话的更高效方法，来解决目前使用 (prompt:"", chosen:"", rejected:"") 格式的局限性。
- **ORPO 需要更好的多轮对话训练**：社区表示担心目前的 ORPO 方法似乎不能很好地适应多轮对话训练，而这对于 ORPO 成为 SFT 的可行替代方案至关重要，强调了对话训练标准化和优化格式的重要性。
- **Mistral 上的 ORPO 成功尝试**：一位成员分享了在 Mistral-7B-v0.2 模型上应用 ORPO TRL 实现的显著成果，使用的是 *argilla/ultrafeedback-binarized-preferences-cleaned* 数据集，并暗示进一步的调优可能会产生更好的结果。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/huggingface/peft/releases/tag/v0.10.0">Release v0.10.0: 使用 DeepSpeed 和 FSDP 微调更大的 QLoRA 模型，层复制，增强 DoRA · huggingface/peft</a>：亮点包括支持 QLoRA 配合 DeepSpeed ZeRO3 和 FSDP。我们添加了一些更改以允许 QLoRA 与 DeepSpeed ZeRO3 和 Fully Sharded Data Parallel (FSDP) 协同工作。例如，这允许...</li><li><a href="https://github.com/pytorch/torchtune/pull/527">rohan-varma 提交的显存小于 16GB 的全量微调 · Pull Request #527 · pytorch/torchtune</a>：背景：我们希望为显存有限的消费级 GPU 用户启用一种内存占用小于 16GB 的全量微调变体。此 PR 实现了全量微调以适应...
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1220636126981652500)** (892 条消息🔥🔥🔥): 

- **Pro 用户讨论 Perplexity 的图像生成**：用户讨论了使用 Perplexity Pro 功能生成图像的能力，并指出这需要使用 Writing 模式并在 Web 版本上关闭 Pro 开关。
- **Claude Opus 与 GPT-4 Turbo 的差异**：对话集中在 Claude 3 Opus 和 GPT-4 Turbo 模型的功能上，比较了它们在学术研究和编写代码方面的能力，以及 GPT-4 Turbo 能够编译 Python 文件而 Perplexity 目前不支持的区别。
- **探索 Stability AI 和本地模型**：重点讨论了 Stability AI 的模型（如 SDXL）和本地安装，用户分享了在个人硬件上运行这些强大图像生成工具的技巧和经验，尽管涉及成本较高。
- **调查 Perplexity 的 Bug 和困惑**：用户对某些 Perplexity 功能表示困惑，例如会话期间出现重复的不相关提示，在使用某些 AI 模型时如何禁用不必要的搜索触发器，以及在 iOS 应用上遇到的问题。
- **Perplexity 功能与更新讨论**：用户讨论了 Claude 3 Opus 的 API 能力、Op1 合成器的美学以及 Rabbit R1 等新模型的潜力，并讨论了将 Perplexity 与 iOS Spotlight 搜索集成的可能性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://imagine.meta.com/">Imagine with Meta AI</a>: 使用 Imagine with Meta AI 免费快速创建高分辨率的 AI 生成图像。只需描述一张图片，Meta AI 就会利用我们的图像基础模型 Emu 的技术将其生成。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - 由 mteb 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://stability.ai/stable-image">Stability AI 图像模型 &mdash; Stability AI</a>: 通过 SDXL Turbo 和 Stable Diffusion XL 体验无与伦比的图像生成能力。我们的模型使用更短的 Prompt，并生成具有增强构图和逼真美感的描述性图像...</li><li><a href="https://civitai.com/models">Civitai | 分享你的模型</a>: 未找到描述</li><li><a href="https://tenor.com/view/swedish-house-mafia-one-op1-synthesizer-edm-gif-19823728">Swedish House Mafia One GIF - Swedish House Mafia One Op1 - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/iota-crypto-cryptocurrency-green-candles-pepe-gif-22089159">Iota Crypto GIF - Iota Crypto Cryptocurrency - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.vellum.ai/blog/prompt-engineering-tips-for-claude">Claude 2.1 Prompt Engineering 指南</a>: 通过这 11 个 Prompt Engineering 技巧学习如何为 Claude 编写 Prompt。</li><li><a href="https://blogs.bing.com/search-quality-insights/december-2023/Introducing-Deep-Search">介绍深度搜索 | Search Quality Insights</a>: 未找到描述</li><li><a href="https://www.xda-developers.com/copilot-gpt-4-turbo-model-free/">你现在可以免费使用 Copilot 的 GPT-4 Turbo 模型了</a>: Microsoft 刚刚向所有人开放了这款先进的 GPT 模型，没有任何限制或套路。</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI 公告 &mdash; Stability AI</a>: 今天早些时候，Emad Mostaque 辞去了 Stability AI 的 CEO 职务以及公司董事会职位，以追求去中心化 AI。董事会已任命...</li><li><a href="https://www.youtube.com/watch?v=G7RgN9ijwE4">你是否曾做过这样的梦？</a>: 我们都在某个时刻做过。</li><li><a href="https://tenor.com/view/kys-wojak-mushroom-kill-urself-die-gif-22188194">Kys Wojak GIF - Kys Wojak Mushroom - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://docs.perplexity.ai/docs/perplexitybot">PerplexityBot</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/OpenAI/comments/1bm305k/what_the_hell_claud_3_opus_is_a_straight/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://tenor.com/view/golden-eggs-willy-wonka-and-the-chocolate-factory-clean-the-eggs-get-the-eggs-ready-chocolate-golden-eggs-gif-21442701">Golden Eggs Willy Wonka And The Chocolate Factory GIF - Golden Eggs Willy Wonka And The Chocolate Factory Clean The Eggs - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl">GitHub - OpenAccess-AI-Collective/axolotl: 尽管提问（axolotl questions）</a>: 尽管提问。通过在 GitHub 上创建账户，为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://apps.apple.com/app/id1484498200">‎Kagi 出品的 Orion 浏览器</a>: ‎Orion 是一款适用于 iPhone 和 iPad 的快速、免费、无广告、无遥测的 Web 浏览器。也可以查看适用于 Mac 桌面版的 Orion。Orion 从底层设计开始就是一款真正尊重隐私的浏览器...</li><li><a href="https://www.cnbc.com/2024/03/21/doj-sues-apple-over-iphone-monopoly.html">美国司法部在具有里程碑意义的反垄断案中起诉 Apple 垄断 iPhone 市场</a>: Apple 及其 iPhone 和 App Store 业务已受到司法部的关注，该部门此前曾对 Google 提起反垄断诉讼。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1220614816494911599)** (38 条消息🔥):

- **法律诉讼在即**：美国正卷入一场诉讼，详情可在 [United States Sues](https://www.perplexity.ai/search/United-States-sues-EDYwvszoRZO.lccIR60pLQ) 探索。
- **电子邮件认证审查**：对电子邮件安全协议（特别是 DMARC）感兴趣的用户可以在 [DMARC Details](https://www.perplexity.ai/search/DMARC-AvS.J6JyS0C6EiEK13CRpw) 找到相关信息。
- **解码市场工具**：TradingView 是一款面向交易者和投资者的工具，相关讨论见 [TradingView Insights](https://www.perplexity.ai/search/tradingview-ZK6dm4jcSyateKhccuwpXQ)。
- **围绕 Perplexity 产生好奇**：Perplexity AI 是否能取代其他工具的可能性正受到关注，见解可在 [Should Perplexity Replace?](https://www.perplexity.ai/search/should-perplexity-replace-qCBVfG3vTmO1oGEr9dTzfg) 发现。
- **探索爱的概念**：有人对爱的本质提出了询问，希望在 [What is Love?](https://www.perplexity.ai/search/What-is-love-bTVpK3ZjTqaMh89tPeWzvg) 了解更多。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1220604143857832019)** (24 messages🔥): 

- **Token 限制困扰**：一名成员因 Prompt 和 Token 生成请求超过 **16,384 token 限制**而遇到 **BadRequestError**。他们被建议减小 `max_tokens` 值以保持在限制范围内，并考虑缩短其 Prompt。
- **简历分析器开发**：该成员分享了他们正在开发一个**简历分析器/构建器项目**，以此作为练习 AI 的方式，表明他们是该领域的新手。
- **寻求 Token 计数说明**：针对通过 Token 计数限制用户 Prompt 的问题，解释称 **Token 数量**与发送的消息长度挂钩。建议他们参考 Perplexity AI 的文档以获取准确的 Token 计数。
- **Tokenization 工具提示**：另一位成员建议使用 OpenAI 的 tokenizer 工具作为 Token 计数的通用衡量标准，但指出不同的 AI 模型可能会有不同的 Tokenization 方式。为了精确起见，他们建议直接通过 Perplexity API 检查 Token 使用情况。
- **寻找类似 Autogpt 的服务**：一名成员询问是否有**支持 Perplexity API keys 的类似 autogpt 的服务**，用于自动化迭代任务。在汇总的消息历史中，该查询未得到回复。

**提及链接**: <a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>: 未找到描述

  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1220624337363800145)** (533 messages🔥🔥🔥): 

- **高 CPU 占用咨询**：用户注意到在 LM Studio 0.2.17 版本上运行模型时 CPU 占用率很高；一些人通过重启 LM Studio 并将设置重置为默认值解决了问题。解决这些问题的建议包括查看日志文件以获取错误信息。
- **GPU 兼容性疑虑**：有关于 LM Studio 最佳 GPU 显卡的咨询，讨论显示更倾向于 Nvidia 显卡（如 3090 TI）以获得更好的兼容性和性能。用户还讨论了关于 GPU offload 的各种问题以及不同模型文件大小对性能的影响。
- **本地服务器访问**：用户在使用 LM Studio 的本地服务器功能时遇到错误，成功的解决方案包括重新安装和正确配置；用户被引导至特定的 Discord 频道 (#1139405564586229810) 发布错误报告以寻求帮助。
- **模型格式支持**：讨论表明 LM Studio 支持 GGUF 模型格式，用户探索了如何使用命令行方法将 Hugging Face 模型转换为 GGUF 格式，以及将转换后的模型分享回 Hugging Face 的重要性。
- **Linux 和 MacOS 支持**：用户询问了在 Linux 和 MacOS 平台上使用 LM Studio 的情况。目前没有针对 docker 镜像或 Intel Mac 版本 LM Studio 的即时计划，但鼓励用户在功能请求频道 (#1128339362015346749) 为该功能投票。
<div class="linksMentioned">

<strong>提及链接</strong>:

</div>

<ul>
<li>
<a href="http://host.docker.internal:8080.">未找到标题</a>: 未找到描述</li><li><a href="http://localhost:YOUR_PORT`平衡">未找到标题</a>: 未找到描述</li><li><a href="https://memgpt.readme.io/docs/lmstudio">LM Studio</a>: 未找到描述</li><li><a href="https://huggingface.co/roborovski/superprompt-v1">roborovski/superprompt-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/expression_language/get_started">入门指南 | 🦜️🔗 Langchain</a>: LCEL 使得从基础组件构建复杂的链变得容易，并且</li><li><a href="https://huggingface.co/bartowski/c4ai-command-r-v01-GGUF">bartowski/c4ai-command-r-v01-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/ban-keyboard-gif-23575674">Ban Keyboard GIF - Ban Keyboard - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/blog/gptq-integration">使用 AutoGPTQ 和 transformers 让 LLM 更轻量化</a>: 未找到描述</li><li><a href="https://github.com/czkoko/SD-AI-Prompt">GitHub - czkoko/SD-AI-Prompt: 一个基于 LLama 2 的快捷指令，用于扩展 stable diffusion 提示词，由 llama.cpp 提供支持。</a>: 一个基于 LLama 2 的快捷指令，用于扩展 stable diffusion 提示词，由 llama.cpp 提供支持。 - czkoko/SD-AI-Prompt</li><li><a href="https://www.youtube.com/watch?v=4fdZwKg9IbU">在本地运行任何开源 LLM（无需代码的 LMStudio 教程）</a>: LMStudio 教程及其新功能演示：多模型支持（并行和序列化）以及 JSON 输出。加入我的时事通讯以获取定期 AI 更新...</li><li><a href="https://rentry.org/LMSTudioFAQ#how-do-i-use-already-downloaded-gguf-models-in-lmstudio">非官方 LMStudio FAQ！</a>: 欢迎来到非官方 LMStudio FAQ。在这里，你将找到我们在 LMStudio Discord 上收到的最常见问题的答案。（此 FAQ 由社区管理）。LMStudio 是一款免费的闭源...</li><li><a href="https://www.youtube.com/watch?v=UbQgXeY_zi4&list=RDUbQgXeY_zi4&start_">Caravan Palace - Lone Digger (官方 MV)</a>: 📀 预购我们的新专辑：https://caravanpalace.ffm.to/gmclub 🎫 来看我们的现场演出：http://www.caravanpalace.com/tour 🔔 订阅我们的频道并点击...</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika 是一个 Agent 架构的 AI 软件工程师，能够理解高级人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标。Devika 旨在成为 Cognition AI 的 Devin 的竞争性开源替代方案。</a>: Devika 是一个 Agent 架构的 AI 软件工程师，能够理解高级人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标...</li><li><a href="https://www.youtube.com/watch?v=UbQgXeY_zi4&list=RDUbQgXeY_zi4&start_radio=1&ab_channel=CaravanPalace">Caravan Palace - Lone Digger (官方 MV)</a>: 📀 预购我们的新专辑：https://caravanpalace.ffm.to/gmclub 🎫 来看我们的现场演出：http://www.caravanpalace.com/tour 🔔 订阅我们的频道并点击...</li><li><a href="https://github.com/caddyserver/caddy">GitHub - caddyserver/caddy: 快速且可扩展的多平台 HTTP/1-2-3 Web 服务器，支持自动 HTTPS</a>: 快速且可扩展的多平台 HTTP/1-2-3 Web 服务器，支持自动 HTTPS - caddyserver/caddy</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948">教程：如何将 HuggingFace 模型转换为 GGUF 格式 · ggerganov/llama.cpp · Discussion #2948</a>: 来源：https://www.substratus.ai/blog/converting-hf-model-gguf-model/ 我在我们的博客上发布了这篇文章，但认为这里的其他人可能也会受益，所以也在 GitHub 上分享了原始博客。希望它...</li><li><a href="https://wiki.yacy.net/index.php/Dev:API">Dev:API – YaCyWiki</a>: 未找到描述</li><li><a href="https://searchlab.eu/en/">用于数据挖掘的网络抓取 - YaCy Searchlab</a>: 未找到描述</li><li><a href="https://yacy.net/">首页 - YaCy</a>: 未找到描述</li><li><a href="https://huggingface.co/nisten/mistral-instruct0.2-imatrix4bit.gguf">nisten/mistral-instruct0.2-imatrix4bit.gguf · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/shao__meng/status/1771718504535978173?s=20">来自 meng shao (@shao__meng) 的推文</a>: 他们刚刚更改了 HuggingFace Space 的 Readme：基于 Mistral-7B-Instruct-v0.1 的 Mistral-7B-Instruct-v0.2 => Mistral-7B-v0.2 https://twitter.com/shao__meng/status/1771680453210370157 ↘️ 引用...</li><li><a href="https://youtu.be/6fFUfyT-EyA?si=Wt5IMTvNfrdHLGyV">George Hotz | 探索 | 寻找 AMD GPU 固件中的漏洞 | 为 tinybox 放弃 AMD</a>: 直播日期 2024 年 3 月 21 日。售价 1050 美元起 购买 https://comma.ai/shop/comma-3x & 世界上最好的 ADAS 系统 https://openpilot.comma.ai 已添加直播聊天...
</li>
</ul>

**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1220749688379150487)** (71 messages🔥🔥): 

- **对 Q*-AGI 的怀疑态度**: 一场关于 *Q*-AGI 视频的讨论引发了大家对该话题的疲劳感，并拒绝观看更多相关内容，频道中还分享了一个[幽默的表情包](https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134)来表达这种情绪。
- **建筑领域的 AI 需要人工验证**: 关于在建筑工程中使用 AI 的对话强调了在没有人工监督的情况下对 AI 模型的怀疑，并指出由于法律和安全问题，需要人工认证。
- **针对特定任务 Fine-Tuning 模型**: 成员们讨论了针对特定任务 Fine-Tuning 较小语言模型的有效性。一位成员分享了他们创建的一个程序，允许用户基于 ChatGPT 2 训练自己的模型，并附带了由 Claude 生成的说明手册。
- **理解模型大小与 Quantization 的区别**: 针对在决定运行哪个模型版本（例如 "llama 7b - q8" 与 "llama 13b - q5"）时，对 *#b*（基于参数的模型大小）和 *q#*（Quantization 级别）之间的区别进行了澄清。
- **为 Context Length 选择合适的模型**: 一位用户询问了用于 RAG 相关交互的 32K Context Length 模型，有人提到 **Mistral** 最近发布了一个具有 32k Context 的 7b 0.2 版本。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: 最近的研究（如 BitNet）正在为 1-bit LLM 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://www.philschmid.de/fine-tune-llms-in-2024-with-trl">How to Fine-Tune LLMs in 2024 with Hugging Face</a>: 在这篇博文中，你将学习如何在 2024 年使用 Hugging Face TRL、Transformers 和 Datasets 来 Fine-Tune LLM。我们将针对 text to SQL 数据集 Fine-Tune 一个 LLM。</li><li><a href="https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134">Dont Say That Ever Again Diane Lockhart GIF - Dont Say That Ever Again Diane Lockhart The Good Fight - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/models?pipeline_tag=image-to-text&language=en&sort=likes">Models - Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1220815376770666556)** (17 messages🔥): 

- **请求下载速度限制器**: 一位用户请求添加 **download speed limiter**（下载速度限制器），以避免耗尽家中的所有带宽。其他用户建议使用系统级设置来限制速度，并认为大型下载在许多应用程序中都很常见。
- **关于图片上传的困惑**: 一位用户在研究如何向模型上传图片时遇到困难。其他人提供了指导，提到了一个名为 **.mmproj converter** 的工具，并指引了在哪里可以找到以及如何使用它。
- **0.2.17 版本中 Llama 图片输入问题**: 由于新版本在图片输入方面存在问题，一位用户被建议使用工具的 **0.2.16** 版本而非 **0.2.17**。然而，随后的补充说明澄清了 **Linux 版跳过了 0.2.16**，而 **0.2.14** 版本除了 moondream2 之外，在 llava vision 模型上运行良好。
- **--context_window 设置的问题**: 一位用户提出了在使用 LM Studio 时 **--context_window** 设置的问题，提到它仅在默认设置下有效。消息记录中未提供直接的解决方案。
- **将讨论移至相关频道**: 一位用户被指示将**技术讨论转移**到专门讨论此类话题的更合适的频道。

**提及的链接**: <a href="https://huggingface.co/nisten/obsidian-3b-multimodal-q6-gguf">nisten/obsidian-3b-multimodal-q6-gguf · Hugging Face</a>: 未找到描述

  

---


**LM Studio ▷ #[📘-docs-and-tips](https://discord.com/channels/1110598183144399058/1123654763112824892/1221503868798636062)** (2 messages): 

- **新文档发布**: LM Studio 推出了其**全新的文档网站**，可以通过 [lmstudio.ai/docs](https://lmstudio.ai/docs) 访问。
- **操作 Multi Model Sessions**: 要了解新的 **Multi Model Session 功能**或 **JSON Mode**，用户可以观看从 [5:57](https://youtu.be/4fdZwKg9IbU?feature=shared&t=357) 开始的教程视频，其中提供了关于其用法的说明和见解。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://youtu.be/4fdZwKg9IbU?feature=shared&t=357)">在本地运行任何开源 LLM（无代码 LMStudio 教程）</a>：LMStudio 教程及其新功能演示：多模型支持（并行和串行）以及 JSON 输出。订阅我的通讯以获取定期 AI 更新...</li><li><a href="https://lmstudio.ai/docs">文档 | LM Studio</a>：技术参考
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1220661495445065789)** (228 条消息🔥🔥): 

- **分布式 AI - 联网的梦想还是实用的方案？**：针对联网机器协作运行语言模型的可行性进行了深入讨论，类似于微型分布式计算环境。由于延迟和带宽限制，人们对其可行性持怀疑态度，但提到了 HyperspaceAI 的方法和 DHT (distributed hash table) 等方法作为潜在的灵感来源。

- **显存（VRAM）至上**：讨论了用于运行 AI 模型的高 VRAM 显卡（如拥有 24GB VRAM 的 RTX 3090）的价格和性能对比。普遍共识是，对于本地机器学习，像 RTX 3090 这样的 GPU 在 VRAM 容量方面提供了最佳性价比。

- **本地计算的新地平线**：对话涉及了 LLM (large language models) 将计算需求转回本地基础设施的前景，类似于 90 年代的 Linux/LAMP 运动。人们将 LLM 开发与部署的潜在增长与过去需要大量草根技术专业知识的技术革命进行了类比。

- **Mac 与内存 - Apple 的大显存方案**：讨论了对未来 Mac 型号（特别是 M3 Ultra Studio）RAM 容量的推测，预期它可能支持至少 256GB (V)RAM。当前的 M3 和 M2 Ultra Mac Studio 因其庞大的系统与 GPU 共享内存而受到关注，能够执行高 VRAM 计算任务。

- **混合 GPU 配置 - 情况变得复杂**：尝试在同一台 PC 中配置 AMD 7800XT 和 NVIDIA 3060，导致 LM Studio 软件出现初始化错误，引发了关于运行这种“科学怪人”式组装机挑战的讨论。还触及了如何搭建拥有多个高 VRAM GPU 的设备以运行大型 AI 模型的更广泛影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://evalplus.github.io/leaderboard.html">EvalPlus 排行榜</a>：未找到描述</li><li><a href="https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/">尽管已加密，黑客仍能读取私人 AI 助手聊天内容</a>：所有非 Google 的聊天 GPT 都会受到侧信道攻击的影响，从而泄露发送给用户的响应。</li><li><a href="https://www.youtube.com/watch?v=nQmZmFERmrg">MemGPT 详解！</a>：非常感谢观看我们关于 MemGPT 的论文总结视频！MemGPT 是一项非常令人兴奋的新工作，它桥接了操作系统管理内存的理念...</li><li><a href="https://github.com/intel/intel-npu-acceleration-library">GitHub - intel/intel-npu-acceleration-library: Intel® NPU 加速库</a>：Intel® NPU 加速库。通过在 GitHub 上创建账号来为 intel/intel-npu-acceleration-library 的开发做出贡献。</li><li><a href="https://github.com/pytorch-labs/gpt-fast">GitHub - pytorch-labs/gpt-fast: 在少于 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。</a>：在少于 1000 行 Python 代码中实现简单高效的 PyTorch 原生 Transformer 文本生成。 - pytorch-labs/gpt-fast
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1220708565686419537)** (16 条消息🔥):

- **重启解决神秘的输出问题**：一位用户报告称，在经历输出质量下降和模型输出乱码后，**重启 LM Studio** 解决了该问题。
- **旧款 AMD GPU 不支持 ROCM 版本**：一位用户在 RX 570 上使用 LM Studio 加载模型时遇到错误，另一位用户澄清称 **RX 570 太旧**，无法与 ROCM 版本兼容。
- **服务器输出突然停止**：多位成员讨论了一个问题，即服务器在仅输出 2 个 token 后停止。排查建议包括共享日志和尝试 “hello world (curl)” 示例。
- **长时间会话导致乱码输出**：一位成员在长时间使用 LM Studio 的过程中遇到了**乱码输出**，这似乎与 token 数量的倍数有关，尽管使用了滚动窗口（rolling window）方法来管理上下文，但问题依然存在。
- **关于管理 Token 限制的建议**：有人指出，将 token 减半可能会导致问题，因为 token 并不等同于单词，可能会导致短语不完整，从而影响模型的响应。建议在 API server 上重复该实验，以便更好地进行日志记录和分析。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1220607338617241660)** (48 messages🔥): 

- **大语言模型（LLM）运行困难**：一位成员提到由于 8GB VRAM 的限制，无法运行**多个 LLM**，并提出了减轻此限制的可能配置。
- **Autogen 的 Token 问题**：有多个关于 Autogen 问题的报告，包括本地环境之外意外的 4-token 限制，这让尝试远程服务器设置的用户感到困惑。
- **工作流困扰**：用户对 Autogen 工作流感到沮丧，一些人根据线程中的建议，诉诸于手动编辑工作流文件，将 `max_tokens` 参数调整为 -1。
- **Autogen Studio UX 体验问题**：成员们讨论了 Autogen Studio 的用户体验，指出了 UI 不够直观，以及需要改进错误消息和模型加载指示器。
- **Autogen 的社区协作**：讨论显示成员们正在积极互相帮助解决 Autogen Studio 的问题，强调了社区驱动的问题解决和知识共享。
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/)** (1 messages): 

pradeep1148: https://www.youtube.com/watch?v=Nc5Yk0XXgP8
  

---


**LM Studio ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/1220927806385684500)** (4 messages): 

- **MemGPT 的 Windows 兼容性困惑**：一位参与者表示在 Windows 上运行 **MemGPT** 存在困难，发现过程很复杂。
- **查看用户指南**：作为回应，另一位成员建议查看**用户指南**以获取帮助。
- **可能仅限 Linux**：该成员随后推测，问题可能是因为 **MemGPT** 是一个仅限 Linux 的应用程序。
- **WSL 作为解决方案**：提供的一个实际解决方案是使用 **Windows Subsystem for Linux (WSL)** 来克服 Windows 兼容性问题。
  

---


**LM Studio ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/1221357876052889672)** (3 messages): 

- **AVX Beta 版本更新不确定**：一位成员询问了 **0.2.10 avx beta 版本**更新的可能性。另一位成员表示，**支持旧硬件**目前不是高优先级任务，一旦更紧迫的问题得到解决，更新最终可能会发布。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1220694776106586142)** (26 messages🔥): 

- **ZLUDA 与 ROCm 的兼容性问题**：一位用户报告在 ROCm 之外安装 ZLUDA 后出现 **100% CPU 占用率**，暗示两者之间存在干扰或优先级问题。讨论指向了将 ZLUDA 而非 ROCm 放在路径（path）中可能导致的冲突，从而影响了性能。
  
- **ROCm 加载困扰**：一位用户在尝试开启 ROCm offloading 加载超过 10GB 的模型时，反复遇到 “Exit code: 42” 错误，即使在 rx6950xt 上有充足的 VRAM 也是如此。模型在不使用 GPU offloading 的情况下可以加载，尽管速度很慢，这表明 offloading 过程可能存在问题。

- **解决路径？**：有人建议用户检查其用户 PATH 环境变量中是否包含以 “ROCm/bin” 结尾的条目，以确保正确连接到 ROCm 库。一位用户报告其环境变量中缺少 ROCm 相关路径，这可能是导致问题的原因。

- **分享环境变量指南**：为了帮助解决 ROCm 加载错误，一位用户建议在用户 PATH 变量中添加一个特定路径：

```C:\Users\[username]\AppData\Local\LM-Studio\app-0.2.17\resources\app\.webpack\main\build\Release\ROCm\bin``` 
此路径旨在帮助遇到加载错误的开发人员连接到 ROCm 库。

**提及的链接**：<a href="https://winaero.com/how-to-see-names-and-values-of-environment-variables-in-windows-10/)">如何在 Windows 10 中查看环境变量的名称和值</a>：在本文中，我们将了解如何查看 Windows 10 中定义的当前用户环境变量和系统变量及其值。

---

**LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1220968798946066463)** (23 条消息🔥): 

- **排除 Interpreter 连接问题**：一位用户在使用本地 *Mistral model* 时遇到错误，提示“**Model with key 'gpt-4' not loaded**”。经过一番讨论，向服务器发送一个简单的 `curl` 请求解决了该问题，但具体原因尚不明确。
- **适用于 Open-interpreter 的本地 LLM 推荐**：用户讨论了多种可与 Open-interpreter 配合使用的 LLM 选项：**CodeLlama 7B Instruct - GGUF** 在示例问题中给出了正确答案，而 **Mistral 7B Instruct v0.2 - GGUF** 表现不佳。另一个推荐是 **Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF**，其特点是输出较为冗长。
- **揭秘 Open Interpreter 的热度**：用户分享了对 Open Interpreter YouTube 宣传视频的看法，其中一位用户因地区限制无法订购。另一位用户分享说可以自行打印设备，STL 文件可在 [01.openinterpreter.com/bodies/01-light](https://01.openinterpreter.com/bodies/01-light) 免费获取。
- **探索兼容 GGUF 的模型**：用户提供了 Hugging Face 上各种模型的链接，包括 GGUF 格式的 CodeLlama 7B Instruct 和 Mistral 7B Instruct，并讨论了它们在 Open-interpreter 测试中的表现。
- **处理非官方推荐（Non-Blessed）模型错误**：一位用户通过修改默认系统消息解决了在使用非官方推荐模型时遇到的错误，并链接到了 [open-interpreter GitHub 上的第 1124 号 issue](https://github.com/OpenInterpreter/open-interpreter/issues/1124)。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://01.openinterpreter.com/bodies/01-light">01 Light - 01</a>：未找到描述</li><li><a href="https://huggingface.co/saltlux">saltlux (saltlux)</a>：未找到描述</li><li><a href="https://huggingface.co/Nan-Do/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF">Nan-Do/Truthful_DPO_TomGrc_FusionNet_7Bx2_MoE_13B-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF">TheBloke/CodeLlama-7B-Instruct-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=jWr-WeXAdeI">开源 AI Agent 震惊业界 | Open Interpreter AI Agent + 设备 (01 Light) 发布！</a>：📩 我的 5 分钟每日 AI 简报 📩 https://natural20.beehiiv.com/subscribe 🐥 在 Twitter (X) 上关注我 🐥 https://twitter.com/WesRothMoney 链接：https://www.openin...</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/issues/1124">bug: `markdown` 已禁用或不支持 · Issue #1124 · OpenInterpreter/open-interpreter</a>：描述 Bug：当使用 LM Studio 调用本地模型 https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF 时，我一直得到本应是有效的 Python 输出，但代码块...</li><li><a href="https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF">TheBloke/Mistral-7B-Instruct-v0.2-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>

---

**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1220604195145781268)** (362 条消息🔥🔥): 

- **Open Interpreter 登陆 Ubuntu**：讨论表明用户正尝试在 Ubuntu 22.04 上运行 Open Interpreter，并针对麦克风支持和客户端理解进行了一些调整。他们表示需要更好地理解客户端-服务器运行机制，并寻求社区的见解。

- **硬件黑客与愿景**：社区正积极寻求构建 O1 light 的信息，并有兴趣为该项目动手组装硬件。由于需求量大，创建了一个名为 <#1221879849535279204> 的新专用 Discord 频道。

- **关于 01 设备的兴奋与疑问**：用户对尝试并可能围绕 O1 设备进行开发感到兴奋，询问了关于其功能、是否需要订阅、Windows 兼容性、用于 4G 连接的 eSIM 可能性，以及是否有易于使用的 UI 等问题。

- **技术支持查询**：成员们正在排查与 Open Interpreter 相关的各种技术问题，从使用 `--os` 选项时 AI 过度闲聊的问题，到利用不同的 LLM 以及与 Groq 等各种 API 的集成。目前的工作重点在于增强安装程序的易用性。

- **贡献与社区增长**：社区内有很强的贡献意愿，用户们正在讨论前端应用开发、与 Groq 的潜在集成、不同 LLM 的性能，以及适用于 Apple silicon 设备的桌面应用。社区成员互相支持彼此的想法和开源努力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.amazon.com/dp/B06XT1Z9TF.">未找到标题</a>: 未找到描述</li><li><a href="https://www.goody2.ai/">GOODY-2 | 全球最具责任感的 AI 模型</a>: 介绍一款具有下一代伦理对齐能力的全新 AI 模型。立即聊天。</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/openai">OpenAI - Open Interpreter</a>: 未找到描述</li><li><a href="https://x.com/hellokillian/status/1757526563879587995?s=20).">killian (@hellokillian) 的推文</a>: ..天呐，Open Interpreter 的首个视觉模型，正在我的 8GB M1 MacBook 上运行。100% 离线。这将进入世界上的每一台电脑。</li><li><a href="https://docs.openinterpreter.com/settings/all-settings#max-tokens">所有设置 - Open Interpreter</a>: 未找到描述</li><li><a href="https://x.com/openinterpreter/status/1771358466877321227?s=46">Open Interpreter (@OpenInterpreter) 的推文</a>: https://twitter.com/i/spaces/1dRJZEPewmgGB</li><li><a href="https://docs.litellm.ai/docs/providers">提供商 | liteLLM</a>: 了解如何在 LiteLLM 上部署并调用来自不同提供商的模型</li><li><a href="https://docs.litellm.ai/docs/providers/groq">Groq | liteLLM</a>: https://groq.com/</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/3e95571dfcda5c78115c462d977d291567984b30/interpreter/core/llm/llm.py#L117">OpenInterpreter/open-interpreter 的 open-interpreter/interpreter/core/llm/llm.py (版本 3e95571)</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/utils/count_tokens.py">OpenInterpreter/open-interpreter 的 open-interpreter/interpreter/terminal_interface/utils/count_tokens.py (main 分支)</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/lavague-ai/LaVa">重定向通知</a>: 未找到描述</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/MikeBirdTech/op">重定向通知</a>: 未找到描述</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/lavague-ai/LaVague&ved=2ahUKEwiE54Sl8IuFAxXQs1YBHfqkCJ0QFnoECA8QAQ&usg=AOvVaw1b8qvOy99zeAyRN_tGJuYY">未找到标题</a>: 未找到描述</li><li><a href="https://www.google.com/url?sa=t&source=web&rct=j&opi=89978449&url=https://github.com/MikeBirdTech/open-interpreter-termux&ved=2ahUKEwi_vZHz-42FAxUPJzQIHQx_BuQQFnoECBMQAQ&usg=AOvVaw3stRzAssQaHpTjlvYh3KQD">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/openinterpreter">来自未定义用户的推文</a>: 未找到描述</li><li><a href="https://groq.com/">GroqChat</a>: 未找到描述</li><li><a href="https://humanaigc.github.io/animate-anyone/">Animate Anyone</a>: Animate Anyone：一致且可控的角色动画图像到视频合成</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter: 计算机的自然语言接口</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://youtu.be/YxiNUST6gU4?si=fSBtR7Tw6WCvWNvN">介绍 Light 01：Open Interpreter 推出的全球首款个人 AI 助手（完整设置）</a>: 在本视频中，我们将查看 OpenInterpreter Light 01 的 GitHub 仓库，这是一个正在彻底改变我们与计算机交互方式的前沿项目...</li><li><a href="https://github.com/Tas667/scalpel/">GitHub - Tas667/scalpel: 帮助你快速了解未知项目结构和内容的 Python 脚本。</a>: 帮助你快速了解未知项目结构和内容的 Python 脚本。 - Tas667/scalpel</li><li><a href="https://youtu.be/FXCaJ3Ga9TE?si=mHELyLpTr8I0MtuM&t=351">如何更便宜地使用 Open Interpreter！（LM Studio / Groq / GPT-3.5）</a>: 第一部分及介绍：https://www.youtube.com/watch?v=5Lf8bCKa_dE 0:00 - 设置 1:09 - 默认 GPT-4 2:36 - 快速模式 / GPT-3.5 2:55 - 本地模式 3:39 - LM Studio 5:5...</li><li><a href="https://github.com/cs50victor/os1">GitHub - cs50victor/os1: AGI 操作系统（Open Interpreter 01 的 UI）</a>: AGI 操作系统（Open Interpreter 01 的 UI） - cs50victor/os1</li><li><a href="https://x.com/fieroty/status/1772004445217489196?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">Ty (@FieroTy) 的推文</a>: 在 01 Light 上运行本地 LLM？轻而易举</li><li><a href="https://nosta.me/">Nosta</a>: 未找到描述</li><li><a href="https://tenor.com/view/surprised-pikachu-pokemon-shock-surprised-pikachu-gif-15357817">惊讶的皮卡丘</a>

u GIF - 惊讶的皮卡丘宝可梦 - 发现并分享 GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.litellm.ai/docs/providers/anthropic#supported-models">Anthropic | liteLLM</a>: LiteLLM 支持</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/docs/CONTRIBUTING.md">open-interpreter/docs/CONTRIBUTING.md at main · OpenInterpreter/open-interpreter</a>: 计算机的自然语言接口。通过在 GitHub 上创建账号，为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://tx.nixc.us/65TjpxNIT7/OpenInterpreter%20in%20Webtop.mov">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1220612133792907265)** (576 条消息🔥🔥🔥): 

- **自己构建 01**: 成员们正在讨论 3D 打印和组装他们自己的 **01 Light** 设备的过程，对 DIY 充满热情。社区正在分享关于材料、设置和打印可能的设计调整的见解，一些人计划对他们自制的设备进行迭代。 
- **01 在美国境外的可用性**: 预订目前仅限于美国，尚未提供国际发布的预计时间。然而，鼓励美国境外的社区成员构建自己的设备并与他人协作。
- **在机器上理解并运行 01 Server**: 对话包括关于在具有不同规格的各种机器（包括低配和云选项）上运行 **01 server** 的问题。这表明只要机器能够处理该模型，就是可行的，其中实际的 LLM 是最耗资源的部分。
- **开发 01 的功能**: 围绕扩展 **01 Light** 的功能（如集成 LED、扬声器或用于连接的 SIM 卡功能）展开了热烈讨论。大家正在交流关于如何创建一个更通用且对 DIY 友好的设计的想法。
- **一般性问题和兴奋之情**: 新成员表达了对该设备的期待，询问了关于预计交付时间、订阅要求以及自动化工具与 Windows 或 Mac 的兼容性等问题。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.voltera.io/v-one">V-One - 桌面级 PCB 打印机 | Voltera</a>：V-One 是一款四合一桌面级 PCB 打印机。在一小时内完成 PCB 原型制作和组装，并获得新设计的即时反馈。</li><li><a href="https://x.com/hellokillian">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.hackster.io/news/m5stack-launches-the-m5dial-a-swish-iot-rotary-encoder-with-a-built-in-color-touchscreen-display-438513c4e52c">M5Stack 发布 M5Dial，一款内置彩色触摸屏的时尚 IoT 旋转编码器</a>：支持 Arduino IDE、ESP-IDF 和 UIFlow，这款多功能设备旨在为您的物联网控制项目提供动力。</li><li><a href="https://tenor.com/view/shut-up-and-take-my-money-futurama-fry-take-my-money-money-gif-15195954">Shut Up And Take My Money Futurama GIF - Shut Up And Take My Money Futurama Fry - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://changes.openinterpreter.com/">Open Interpreter 博客</a>：开源项目 Open Interpreter 的官方更新日志。</li><li><a href="https://www.youtube.com/@MikeBirdTech/videos">Mike Bird</a>：AI 工程</li><li><a href="https://shop.m5stack.com/products/atom-echo-smart-speaker-dev-kit">ATOM Echo 智能扬声器开发套件</a>：ATOM ECHO 是一款可编程智能扬声器。这款 ESP32 AIoT 开发套件配有麦克风和扬声器，用于 AI 语音交互，轻巧便携。它可以接入 AWS、百度、ESPHome 和 Home Assistant...</li><li><a href="https://shop.m5stack.com/products/unitv2-ai-camera-gc2145">M5Stack UnitV2 - 用于边缘计算的独立 AI 摄像头 (SSD202D) TinyML</a>：UnitV2 是一款高效的 AI 识别模块，采用 Sigmstar SSD202D 芯片，拥有 128MB-DDR3 内存、512MB Flash 和 1080P 摄像头。UnitV2 使用简单且高效，支持 AI 识别...</li><li><a href="https://x.com/openinterpreter/status/1771358466877321227?s=46">来自 Open Interpreter (@OpenInterpreter) 的推文</a>：https://twitter.com/i/spaces/1dRJZEPewmgGB</li><li><a href="https://github.com/rhasspy/piper/?tab=readme-ov-file#running-in-python">GitHub - rhasspy/piper：一个快速、本地的神经文本转语音系统</a>：一个快速、本地的神经文本转语音系统。通过在 GitHub 上创建账号来为 rhasspy/piper 的开发做出贡献。</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuYTxMM?typeform-source=pcr08jir95k.typeform.com">联系我们</a>：使用 Typeform 将数据收集变成一种体验。创建精美的在线表单、调查、测验等。免费试用。</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md">OpenInterpreter/01 项目 main 分支下的 01/hardware/light/BOM.md</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=xPd8FFzIeOw">ChatGPT "Code Interpreter" 但 100% 开源 (Open Interpreter 教程)</a>：这是我关于 Open Interpreter 的第二个视频，包含许多新功能且更加稳定，全新的 Open Interpreter 非常出色。更新：Mixtral 7x8b 曾是...</li><li><a href="https://github.com/OpenInterpreter/01/issues">Issues · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/docs/bodies">OpenInterpreter/01 项目 main 分支下的 01/docs/bodies</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuY">探索 Typeform，让表单变得有趣</a>：无需代码，几分钟内即可创建精美的交互式表单。免费开始使用。</li><li><a href="https://github.com/openai/whisper">GitHub - openai/whisper：通过大规模弱监督实现的鲁棒语音识别</a>：通过大规模弱监督实现的鲁棒语音识别 - openai/whisper</li><li><a href="https://github.com/adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB">GitHub - adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB：Adafruit PAM8302 单声道放大器的 PCB 文件</a>：Adafruit PAM8302 单声道放大器的 PCB 文件。通过在 GitHub 上创建账号来为 adafruit/Adafruit-PAM8302-Mono-Amplifier-PCB 的开发做出贡献。</li><li><a href="https://wiki.seeedstudio.com/xiao_esp32s3_speech2chatgpt/">基于 XIAO ESP32S3 Sense 的微型 ChatGPT 语音助手 | Seeed Studio Wiki</a>：本教程介绍了如何使用 XIAO ESP32S3 录制语音、识别语音，然后向 ChatGPT 提问并获取答案显示在屏幕上。</li><li><a href="https://youtu.be/4fdZwKg9IbU?si=_rOJ4fXzAO7SpuPE">在本地运行任何开源 LLM（无代码 LMStudio 教程）</a>：LMStudio 教程及其新功能演示：多模型支持（并行和串行）以及 JSON 输出。加入我的通讯以获取定期...</li>

ar AI Up...</li><li><a href="https://youtu.be/-Y1wWJAnqRk?si=PLWODfGzDtGR4Poc">PCB prototyping, PCB making at home - WEGSTR</a>: 通过这个分步视频指南体验 PCB 制造的迷人世界。学习使用 CNC 铣床制作高质量 PCB 的艺术...</li><li><a href="https://x.com/i/spaces/1dRJZEPewmgGB">Tweet from GitHub - FixTweet/FxTwitter: Fix broken Twitter/X embeds! Use multiple images, videos, polls, translations and more on Discord, Telegram and others</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://www.adafruit.com/product/3968?gad_source=1&gclid=CjwKCAjwnv-vBhBdEiwABCYQA2Mme3uVca46pohzqJ-jT8IzOZ7Xew6y5cEefuqKkRNhJdbfjdvKyxoC-lAQAvD_BwE">Speaker - 40mm Diameter - 4 Ohm 3 Watt</a>: 听好消息！这款扬声器是任何需要 4 Ohm 阻抗和 3W 或更低功率的音频项目的绝佳补充。直径 40mm，形状更接近方形，....</li><li><a href="https://www.adafruit.com/product/3341">DotStar Micro LEDs (APA102&ndash;2020) - Smart SMD RGB LED - 10 pack</a>: 这些极其微小的表面贴装 LED 是为您的项目添加大量非常微小（但明亮！）彩色点的简便方法。它们是我们数字产品的微型版本...</li><li><a href="https://www.instagram.com/reel/C41iQZ6L0_I/"> Concept Bytes on Instagram: &quot;A useful Ai named Jarvis. 
What features do you wanna see next?
#ironman #ai #tech #xtool #xtoolf1 
&#064;xtool.official&quot;</a>: 6,784 个赞，154 条评论 - concept_bytes 于 2024 年 3 月 22 日发布：&quot;一个名为 Jarvis 的实用 AI。你接下来想看到什么功能？#ironman #ai #tech #xtool #xtoolf1 &#064;xtool.official&quot;</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/software/source/server/i.py">01/software/source/server/i.py at main · OpenInterpreter/01</a>: 开源语言模型计算机。通过在 GitHub 上创建账户为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/ROADMAP.md">01/ROADMAP.md at main · OpenInterpreter/01</a>: 开源语言模型计算机。通过在 GitHub 上创建账户为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=JeyZ4HQARMc">Wake word demonstration on Raspberry Pi and custom ESP32 board in Home Assistant | Year of the Voice</a>: 在 Home Assistant 中基于 Raspberry Pi 和自定义 ESP32 板的唤醒词演示 | 语音之年。如果说在“语音之年”期间人们一直要求的一件事，那就是唤醒词。能够说出像 &quot;OK Google,&quot; &quot;Hey Siri,&quot; 或 ... 之类的话。</li><li><a href="https://youtu.be/nzznxPeWDO4?si=sSA3iuiZQZEqgogG">Shocking Open Interpreter AI Agent + Device (01 Light ) is Finally Revealed</a>: 令人震惊的 Open Interpreter AI Agent + 设备 (01 Light ) 终于揭晓 #OpenInterpreter #aiagent 频道链接：🕵️‍♀️ 加入我的 Patreon: https://www.patreo...</li><li><a href="https://github.com/Tas667/scalpel/">GitHub - Tas667/scalpel: python script that helps you quickly understand the structure and contents of an unknown project.</a>: 帮助你快速了解未知项目结构和内容的 Python 脚本。 - Tas667/scalpel</li><li><a href="https://github.com/SYSTRAN/faster-whisper">GitHub - SYSTRAN/faster-whisper: Faster Whisper transcription with CTranslate2</a>: 使用 CTranslate2 的 Faster Whisper 转录。通过在 GitHub 上创建账户为 SYSTRAN/faster-whisper 的开发做出贡献。</li><li><a href="https://github.com/m-bain/whisperX">GitHub - m-bain/whisperX: WhisperX:  Automatic Speech Recognition with Word-level Timestamps (&amp; Diarization)</a>: WhisperX：具有词级时间戳（和说话人日志）的自动语音识别 - m-bain/whisperX</li><li><a href="https://developer.nvidia.com/blog/building-generally-capable-ai-agents-with-minedojo/">Building Generally Capable AI Agents with MineDojo | NVIDIA Technical Blog</a>: 使用 MineDojo 构建通用能力的 AI Agents | NVIDIA 技术博客。NVIDIA 正在通过一个名为 MineDojo 的新开源框架，帮助突破训练 AI 通用 Agent 的极限。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1220623095551885402)** (11 messages🔥): 

- **World's First Fully Open-Source AI Assistant**: 聊天中重点介绍了一个名为“Open Interpreter's 01 Lite - 全球首款完全开源的个人 AI AGENT 设备”的 YouTube 视频，该视频评测并展示了 **01 Lite** 的安装过程，这是一款 100% 开源的个人 AI 助手。这是[视频链接](https://www.youtube.com/watch?v=Q_p82HtBqoc)。

- **实时 AI 软件与派对回顾**：一位成员分享了他们首次尝试运行 01 软件的 YouTube 直播链接，以及在 Discord 上举行的发布派对录音。直播可以在[这里](https://www.youtube.com/live/QQl9dIfqv58?si=0nEPsJwLJWSW8H_y&t=2227)观看。
- **高质量 LLM 的必要性**：**vincentjedi** 讨论了大型语言模型 (LLM) 在 Open Interpreter (OI) 未来中的核心作用，指出进展“100%”取决于 LLM 将提示词 (prompts) 转化为无错命令的能力。
- **开创“Rabbit 策略”**：**vincentjedi** 提到，使用“Rabbit 策略”通过反馈到云服务的用户交互来训练大动作模型 (Large Action Models) 是 OI 的必要路径。
- **对 UI/UX 挑战的乐观态度**：虽然 **vincentjedi** 指出在各种应用和界面中实现无错用户体验是一项重大挑战，但 **techfren** 指出，在 UI 测试中可以安全且更高效地应用快速试错法。
- **为方便起见剪辑的 01 软件直播**：**techfren** 发布了直播的剪辑版本，重点展示了 01 软件的部分，以便于观看，为对 OI 产品感兴趣的人提供了深刻的资源。点击[这里](https://www.youtube.com/watch?v=l3fUlHjEmZE)观看剪辑后的直播。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com">YouTube</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Q_p82HtBqoc">Open Interpreter's 01 Lite - 全球首款完全开源的个人 AI AGENT 设备</a>：Open Interpreter 的 01 Lite 是一款 100% 开源的个人 AI 助手，可以控制你的电脑。让我们来评测它，我将向你展示如何安装 open...
</li>
</ul>

</div>
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1220711550726180939)** (574 条消息🔥🔥🔥): 

- **欧洲数据法阻碍 LAION 的潜力**：讨论强调，由于欧盟严格的监管，LAION 的数据集可能不如美国的同行有效。有人建议，在欧盟放宽数据法之前，必须依赖合成数据以及与限制较少司法管辖区的合作，这一过程被幽默地称为“数据洗白”。

- **Stability AI CEO 在混乱中下台**：Stability AI 创始人 Emad Mostaque 宣布辞去 CEO 职务，公司发布的[新闻稿](https://stability.ai/news/stabilityai-announcement)确认了这一消息。临时联席 CEO Shan Shan Wong 和 Christian Laforte 将领导寻找新的正式 CEO，同时人们对公司方向和开源承诺的潜在后果产生了猜测。

- **SD3 模型预期设定**：SD3 模型的预览表明，它在某些语境下的表现与 DALL-E 3 相当，但总体上在理解提示词中的复杂概念和交互方面表现挣扎。尽管具有更逼真的图像生成能力，但据报道 SD3 模型通常以拼贴式的方式组合图像，缺乏概念间的清晰融合。

- **AI 争议与伦理成为焦点**：一条消息指向了 Twitter 上的一场对话，其中对 AI 行业知名人物背后的动机表示了担忧。这引发了关于 AI 开发者和研究人员的伦理责任，以及社交媒体平台上对 AI “名人”文化痴迷的讨论。

- **AMD 在 AI 方面的性能挑战**：用户分享了他们对 AMD GPU 和 ROCm 对 ML 工作负载支持的挫败感，认为其与 NVIDIA 的解决方案相比处于劣势。轶事证据表明，AMD 在消费级 ML 支持方面缺乏投入，可能会在 Stable Diffusion 等生成式 AI 模型的兴起中错失良机。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<a href="https://www.forbes.com/sites/kenrickcai/2024/03/22/stability-ai-founder-emad-mostaque-plans-to-resign-as-ceo-sources-say/?sh=5e3e64bd5239">Stability AI 创始人 Emad Mostaque 计划辞去 CEO 职务，据消息人士称</a>：Mostaque 已告知多位亲近人士，他计划辞去这家因 Stable Diffusion 而闻名的曾经炙手可热的生成式 AI 初创公司的首席执行官职务。</li><li><a href="https://x.com/chrlaf/status/1771933102329171976?s=20">Christian Laforte (@chrlaf) 的推文</a>：@USEnglish215753 @StabilityAI @EMostaque 是的，计划没有改变，我们仍在努力改进模型，以实现开源发布。</li><li><a href="https://webllm.mlc.ai/#chat-demo>">WebLLM | 首页</a>：未找到描述</li><li><a href="https://lifehacker.com/tech/its-not-safe-to-click-links-on-x">在 X 上点击链接并不安全</a>：当有人在 X 上发布链接时，该网站会生成链接预览。但据报道，这个系统可以被欺骗，不法分子可以通过虚假的链接预览将你重定向到恶意网站……</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI 公告 &mdash; Stability AI</a>：今天早些时候，Emad Mostaque 辞去了 Stability AI 的 CEO 职务以及公司董事会的职位，以追求去中心化 AI。董事会已任命……</li><li><a href="https://tenor.com/Di0E.gif">金钱哭泣 GIF - Woody Harrelson 金钱哭泣 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://tenor.com/YqCL.gif">我正在尽我的一份力 严肃 GIF - 我正在尽我的一份力 严肃凝视 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1bmtp77/do_not_generate_a_tree_using_a_model_trained_on/>">reddit.com: 超过 18 岁？</a>：未找到描述</li><li><a href="https://archive.ph/J7Xdw">Stability AI 创始人 Emad Mostaque 计划辞去 CEO 职务，据消息人士称</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1220751321188012102)** (92 条消息🔥🔥): 

- **吴恩达 (Andrew Ng) 预测 AI 工作流革命**：Google Brain 联合创始人吴恩达预测，**AI agentic workflows** 将在今年推动 AI 的重大进展，其潜力甚至可能超过下一代基础模型。他强调了对文档进行多次迭代（如大纲、草拟和修订）的重要性，并将其与目前的 zero-shot LLM 方法进行了对比 ([来自 Reddit 的亮点](https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/))。

- **MIT CSAIL 加速图像生成**：MIT 的研究人员取得了一项突破，他们创建了一个框架，通过教师-学生模型 (teacher-student model)，在不损失图像质量的情况下，将 **Stable Diffusion** 和 **DALL-E** 等工具的图像生成过程加速了 30 倍，并将其简化为单步生成 ([MIT 新闻文章](https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321))。

- **NVIDIA 探讨 Diffusion Models 的训练**：NVIDIA 的博文讨论了改进 diffusion models 所面临的挑战，以及他们如何解决许多神经网络在训练过程中遇到的共同问题。他们提到了其 EDM2 代码和模型的发布，并建议与样式归一化 (style normalization) 相关的权属问题可能需要通过类似 EDM2 中的修改来解决 ([NVIDIA 开发者博客文章](https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/))。

- **辩论线性网络时代 Unet 的相关性**：对话中有人对在图像生成任务转向线性网络模型的背景下，增强 Unet 架构的价值表示怀疑。一些人认为线性模型不需要传统的归一化方法，而另一些人则表示怀疑，认为 layer norm 等概念仍然是神经网络功能中不可或缺的一部分。

- **战略性剪枝与中间块之谜**：关于大语言模型 (LLMs) 韧性的讨论得出一个见解，即移除网络中间的块对性能的下降影响极小。这引发了关于某些网络片段（特别是 "unet middle block"）潜在冗余的猜测，以及对线性网络架构特性进行进一步研究的必要性。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://news.mit.edu/2024/ai-generates-high-quality-images-30-times-faster-single-step-0321">AI 单步生成高质量图像速度提升 30 倍</a>：一种新的分布匹配蒸馏 (DMD) 技术将 GAN 原理与 Diffusion Models 相结合，在单个计算步骤中实现了快 30 倍的高质量图像生成，并增强了...</li><li><a href="https://tenor.com/view/explode-cute-cat-gif-14074577">爆炸可爱 GIF - 爆炸可爱猫咪 - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://docs.google.com/document/d/1M_QWSRv44M3j69Sxq1fcgfowvgioS5nYfP84D9keUeI/edit">TRC 报告 4</a>：未找到描述</li><li><a href="https://www.reddit.com/r/singularity/comments/1bl3s9r/andrew_ng_cofounder_of_google_brain_former_chief/">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://developer.nvidia.com/blog/rethinking-how-to-train-diffusion-models/">重新思考如何训练 Diffusion Models | NVIDIA 技术博客</a>：在探索了 Diffusion Models 采样、参数化和训练的基础知识后，如《生成式 AI 研究亮点：揭秘基于扩散的模型》中所述……
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1221088542047080459)** (26 条消息🔥): 

- **发货时间表咨询**：一位成员询问了某款未指明产品的发货时间表，表示希望订购一台进行实验，同时不影响客户的供应。另一位成员表示目前没有确切日期，但预计在夏季发货，并提供了一个自行组装的备选方案。
- **Raptor 实现探索**：一位成员描述了 Raptor 的一个版本实现，该版本在聚类上生成摘要，并使用句子级 word-to-vector 进行 Embedding 而无需预训练。该成员指出，对于转录文档，3b 模型生成摘要需要 5 分钟，并提到该技术可能会产生大量生成内容，但效果可能等同于带有分块摘要的 Prompt。
- **对 Claude 的赞赏**：一位成员表达了对 Claude 的热情，而另一位成员则期待由于其上下文质量，它将对开源开发和新项目的创建产生影响。
- **分享 FastAPI 资源**：一位成员分享了 [FastAPI](https://fastapi.tiangolo.com/) 的链接，赞扬其易用性且已具备生产就绪性。他们还链接了其 [文档](https://fastapi.tiangolo.com) 和 [源代码](https://github.com/tiangolo/fastapi)，并询问是否有使用该后端框架的开源项目。
- **Suno.AI 创意乐趣**：一位成员分享了 Suno.AI 的链接，表示这是一个有趣的平台，另一位成员确认了它创建 Spotify 播放列表的能力。成员们似乎对该平台的输出感到非常满意。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://fastapi.tiangolo.com/">FastAPI</a>：FastAPI 框架，高性能，易于学习，快速编码，生产就绪</li><li><a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast 遇见 Mistral：AI 根据他的疯狂特技创作了一本食谱！</a>：今天我们创作了 Beast 食谱，“Beast 食谱”的想法是一种有趣且富有创意的方式，可以与 Mr. Beast 的内容互动，并生成一个有趣的、虚构的...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1220773709313474622)** (13 条消息🔥): 

- **用于 Nous 模型的 Kubernetes**：成员们讨论了使用 Kubernetes 部署 Nous 模型，引用了 [Sidero Labs 的一条推文](https://twitter.com/SideroLabs/status/1771207304748167445)。技术方案包括在 Docker 容器中运行带有 GGUF 模型格式的 Ollama，然后使用 Kubernetes pods 进行编排。
  
- **OpenRouter.ai 发现**：一位成员引起了大家对 [OpenRouter.ai](https://openrouter.ai/) 服务的关注，有些人已使用该服务访问 Opus。它与一位 Discord 工作人员有关。
  
- **语言模型说服力分析**：提到了一篇 [arXiv 论文](https://arxiv.org/abs/2403.14380) 的预注册研究，重点关注大型语言模型在与人类辩论中的说服能力。
  
- **AI 驱动艺术平台展示**：分享的链接包括 [ArtHeart.ai](https://artheart.ai/)（一个用户可以使用 AI 角色进行娱乐、创作和获利的平台）以及 [novelcrafter](https://novelcrafter.co) 等。
  
- **评估 BitNet 的量化感知训练**：一篇 [Hugging Face 博客文章](https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5) 深入探讨了 BitNet 1.5 的实验，讨论了推理过程中的潜在加速以及由于需要平滑优化器梯度而导致的训练限制。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://x.com/winglian/status/1771918928341794821?s=46&t=stOPrwZiN_fxSK0RuC8Flg">Wing Lian (caseus) (@winglian) 的推文</a>: 目前尚不清楚在将 DenseFormers 与预训练的 Mistral-7B 集成时，是否能够实现论文中看到的改进，因为他们发现最佳性能是在训练过程中观察到的...</li><li><a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Bitnet 1.5 实验 (ngmi)</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2403.14380">论大语言模型的对话说服力：一项随机对照试验</a>: 大语言模型 (LLMs) 的发展和普及引发了人们的担忧，即它们将被用来创建量身定制的、具有说服力的论点，以在网上传播虚假或误导性的叙述...</li><li><a href="https://openrouter.ai/">OpenRouter</a>: 一个用于 LLM 和其他 AI 模型的路由器
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/)** (1 条消息): 

proprietary: @everyone https://twitter.com/NousResearch/status/1771735632035127594
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1220619542405976094)** (469 条消息 🔥🔥🔥): 

- **World Simulator 惊艳众人**：社区对 World Simulator 项目的影响议论纷纷，认为沉浸式提示 (immersive prompting) 和 AI 生成的宇宙特别酷且引人入胜。用户分享了诸如创建《三体》前传以及生成独特（尽管有时令人费解）的文明演化等经历。
  
- **期待 Winged AI 治疗师**：一位成员讨论了他们正在开发的名为 Thestral 的 AI 治疗师，考虑在 LLaMA 70B 上微调 NousResearch 的 Hermes，以获得专注于治疗的效果。他们分享了使用专为治疗对话微调设计的数据集来实现它的意图。

- **Sim 的 Opus 华丽表现**：用户讨论了 World Simulator 中使用的底层模型，Claude 3 的 Opus 因其创造性模拟宇宙的能力而被引用，尽管它存在拒绝和“伦理”约束。大家一致认为，尽管存在限制和成本，Opus 提供了比其他替代模型更令人满意的用户体验。

- **模型拒绝的奇特案例**：一场详细的讨论揭示了对嵌入在 Hermes 2 Pro 函数调用 (function-calling) 模型中的拒绝提示的担忧，这可能会干扰定制的 AI 功能。成员们讨论了有效的拒绝提示与模型适应新加入功能的潜力之间的对立。

- **解码 LLM 中的 Overton Effect**：一位成员阐明了 LLM 中所谓的 “Overton Effect”，这导致像 Claude 这样的 AI 模型将对话引向更普遍接受的规范，从而可能抑制生成过程中的创造力和新颖性。这一见解引发了关于操纵模型提示以绕过标准 LLM 限制的讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<a href="https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1">mixedbread-ai/mxbai-rerank-large-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B-DPO">NousResearch/Nous-Hermes-2-Mistral-7B-DPO · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/Nous">Nous (موسى عبده هوساوي )</a>: 未找到描述</li><li><a href="https://vatsadev.github.io/articles/transformerMath.html">Transformers learn patterns, math is patterns</a>: 未找到描述</li><li><a href="https://x.com/marvinvonhagen/status/1771609042542039421?s=46&t=TOasxww3M5DjlB4iBWa_ig">来自 Marvin von Hagen (@marvinvonhagen) 的推文</a>: Mistral 刚刚在 @SHACK15sf 宣布，他们今天将发布一个新模型：Mistral 7B v0.2 Base Model - 32k 而非 8k 的上下文窗口 - Rope Theta = 1e6 - 无滑动窗口</li><li><a href="https://tenor.com/view/spongebob-why-why-why-why-why-why-why-why-why-why-why-why-why-gif-25252239">海绵宝宝 为什么 为什么 为什么 GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/hd_nxx-gif-26166561">Hd_nxx GIF - HD_NXX - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar">未找到标题</a>: 未找到描述</li><li><a href="https://gandalf.lakera.ai/">Gandalf | Lakera – 测试你的 Prompt 技巧，让 Gandalf 泄露秘密信息。</a>: 诱导 Gandalf 泄露信息，亲身体验 LLM 的局限性。</li><li><a href="https://tenor.com/view/sifas-ruby-kurosawa-love-live-merge-gif-20382260">Sifas Ruby Kurosawa GIF - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/RESMPDEV/Mistral-7B-v0.2/tree/main">RESMPDEV/Mistral-7B-v0.2 at main</a>: 未找到描述</li><li><a href="https://x.com/jmorgan/status/1771705967886929960?s=46">来自 Jeffrey Morgan (@jmorgan) 的推文</a>: 使用 Ollama 运行 Mistral 更新至 v0.2 的新基础文本补全模型：ollama run mistral:text https://ollama.com/library/mistral:text</li><li><a href="https://x.com/AlexReibman/status/1771608346635751541?s=20">来自 Alex Reibman 🖇️ (@AlexReibman) 的推文</a>: Mistral 在 @cerebral_valley 黑客松上随手发布了一个新模型</li><li><a href="https://huggingface.co/jinaai/jina-embeddings-v2-base-en">jinaai/jina-embeddings-v2-base-en · Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/Q_p82HtBqoc">Open Interpreter 的 01 Lite - 全球首款完全开源的个人 AI AGENT 设备</a>: Open Interpreter 的 01 Lite 是一款 100% 开源的个人 AI 助手，可以控制你的电脑。让我们来评测一下，我将向你展示如何安装开源...</li><li><a href="https://huggingface.co/datasets/wesley7137/therapist-sft-format">wesley7137/therapist-sft-format · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/furlat/SpriteStash">GitHub - furlat/SpriteStash: 一个使用 LanceDB、Pydantic 和 pygame-ce 的多模态精灵图向量数据库</a>: A multimodal sprites vectordb using LanceDB, Pydantic and pygame-ce - furlat/SpriteStash</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: 通过在 GitHub 上创建账号来为 mistralai-sf24/hackathon 的开发做出贡献。</li><li><a href="https://gist.github.com/fullstackwebdev/5e812f46c542ab8869db899b0c535fc2">unsloth_finetune_mistral-7b-v0.2-on-openhermes-2.5-dataset.py</a>: GitHub Gist: 即时分享代码、笔记和代码片段。</li><li><a href="https://github.com/jquesnelle/crt-terminal">GitHub - jquesnelle/crt-terminal: 复古风格的终端 Shell</a>: Retro styled terminal shell. 通过在 GitHub 上创建账号来为 jquesnelle/crt-terminal 的开发做出贡献。</li><li><a href="https://huggingface.co/colbert-ir/colbertv2.0">colbert-ir/colbertv2.0 · Hugging Face</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1220716612810706985)** (24 条消息🔥): 

- **SFT 数据集中的 Few-Shot Prompts？**: 几位成员讨论了在指令 SFT 数据集中包含 Few-Shot Prompts 以增强 Few-Shot Prompting 能力是否正常或有益。对话倾向于认为这种做法并不常见，并提到了另一个帖子以供深入了解：[相关线程](https://discord.com/channels/1053877538025386074/1109649177689980928/1212266969131130880)。
 
- **寻找微型 LLM**: 一位成员请求推荐一个用于学习的**低参数 LLM**，另一位成员试图澄清其是否是指 100M 参数的模型，并建议观看 **Andrej Karpathy 的视频**来了解此类模型。

- **因果掩码（Causal Masking）是理论还是工程技巧？**：有人质疑了 Attention 中因果掩码的必要性，另一位成员指出其对于模型学习 Next Token Prediction（下一标记预测）的重要性。

- **Llama 前馈网络的层数之谜**：关于 Llama 的 Feedforward 拥有三个线性层的讨论，通过引用一个 GitHub Issue 和一篇 [arXiv 论文](https://arxiv.org/pdf/2002.05202.pdf) 得到了澄清。SwiGLU 的实现被强调为模型设计中一个成功的非线性激活函数选择。

- **比较 ORPO 与 SFT+DPO 以及编程模型偏好**：有人询问 **ORPO** 是否可靠地优于 **SFT+DPO**，聊天中未达成共识；另一项关于 lmstudio 中首选本地编程模型的咨询，得到的回复是目前还没有特定模型脱颖而出。

**提及的链接**：<a href="https://github.com/meta-llama/llama/issues/1004">Why does the FeedForward have three linear layer? · Issue #1004 · meta-llama/llama</a>：我发现 FFN 的实现有三个线性层。https://github.com/facebookresearch/llama/blob/ef351e9cd9496c579bf9f2bb036ef11bdc5ca3d2/llama/model.py#L337-L345 但在论文 "Atte...

  

---


**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1220606789326864456)** (3 条消息): 

- **能力确认**：一位成员通过表示 *“让我看看我能不能做到”* 表达了完成任务的信心。
- **模型特性查询**：有人提出了一个关于哪些模型被认为是 *"nonagreeable"* 的问题，但未提供进一步的背景或后续信息。
  

---


**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1220652967787495476)** (19 条消息🔥): 

- **关于 Discord 育儿的闲聊**：成员们简短讨论了为人父母的喜悦和惊喜，其中一人提到在决定成为父母后 “生物机制开始发挥作用”，带来了意想不到的幸福感。
- **寻找开源 Wikipedia RAG 索引**：一位成员询问是否有开源的 Wikipedia RAG Index，另一位成员建议有多个贡献者提供了类似的资源。
- **关于微软和加州大学伯克利分校 RAFT 的见解**：分享了一篇论文和 Twitter 帖子的链接，讨论了 “检索增强微调 (RAFT)”，该技术旨在通过使用干扰文档进行训练并结合 Chain-of-thought，使像 Llama 7B 这样的 LLM 更加健壮。分享的 [论文和帖子](https://huggingface.co/papers/2403.10131) 显示了 RAFT 具有前景的结果，例如在医疗背景下优于 GPT-3.5。
- **RAFT 实现的仓库链接**：分享了名为 “Gorilla” 的 RAFT 实现的 GitHub 仓库，它为 LLM 提供了一个 API 商店。仓库地址为 [GitHub - ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/main/raft)。
- **关于 GermanRAG 和跨文档知识检索的讨论**：一位成员在讨论跨多个文档收集知识的挑战时提到了一个名为 GermanRAG 的项目。另一位成员确认了这种复杂性，并暗示了他们一直在研究的潜在解决方案。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/ShishirPatil/gorilla/tree/main/raft">gorilla/raft at main · ShishirPatil/gorilla</a>：Gorilla：LLM 的 API 商店。通过在 GitHub 上创建账号为 ShishirPatil/gorilla 的开发做出贡献。</li><li><a href="https://x.com/_philschmid/status/1771456524763697591?s=20">Philipp Schmid (@_philschmid) 的推文</a>：我们能通过微调让 RAG 应用更健壮吗？微软和加州大学伯克利分校的一篇论文对此进行了测试，看看像 Meta 的 Llama 7B 这样的小型开源 LLM 是否能匹配 OpenAI 的 GPT-3.5。T...
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1221910863972143154)** (2 条消息): 

- **提供语言设置提示**：一位成员分享了一个 [来自 Tenor 的 GIF](https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450)，并指出如果 Tenor.com 的语言设置与用户的浏览器语言不匹配，可以进行更改。
- **轻松的问候**：另一位成员只是进来向频道打个招呼 "helloooo"。

**提及的链接**：<a href="https://tenor.com/view/everyone-get-in-here-grim-patron-gif-26273450">Everyone Get In Here Grim Patron GIF - Everyone Get In Here Grim Patron - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---



**OpenAI ▷ #[annnouncements](https://discord.com/channels/974519864045756446/977259063052234752/1221870596015657032)** (1 条消息):

- **Sora 激发艺术家和电影制作人的创造力**：OpenAI 强调了其与使用 **Sora** 的创意人士的合作，该工具旨在帮助将富有想象力的想法变为现实。导演 Paul Trillo 赞扬了它的潜力：“当你不是在复制旧事物，而是赋予那些我们原本从未有机会见到的、全新的、不可能的想法以生命时，Sora 的威力最为强大。”

- **Sora：通往超现实主义的桥梁**：制作公司 shy kids 非常看重 Sora 创造“完全超现实事物”的能力，这标志着从生成写实图像向构建不可思议之物的跨越。他们的兴奋感以及在创意工作流中的潜在应用在 [OpenAI 博客](https://openai.com/blog/sora-first-impressions)中有详细介绍。

**提到的链接**：<a href="https://openai.com/blog/sora-first-impressions">Sora: First Impressions</a>：我们从创意社区获得了宝贵的反馈，帮助我们改进模型。

---

**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1220659924212908032)** (264 条消息🔥🔥): 

- **探索 LLM 偏见与默认设置**：一场关于 AI 偏见的激烈讨论展开了，特别是围绕 GPT 等通用 LLM 默认对齐“西方自由中心主义”价值体系的问题。一位用户提出了创建多个“已对齐”版本的 AI 模型的想法，认为目前的 LLM 默认隐含地将西方价值观呈现为最优解。

- **定制 ChatGPT**：‘Customize ChatGPT’ 功能被强调为一种将个人价值观或文化背景注入 AI 回复的方式。有人建议，用户不应只关注识别 AI 偏见，而可以专注于 ChatGPT 如何为他们的生活增加生产性价值。

- **将 AI 与非西方规范对齐被证明非常棘手**：引导 AI 给出非西方答案的尝试结果参差不齐，AI 仍倾向于向以西方为中心的理想靠拢。尽管尝试了抑制“以西方为中心”的提示词（如 ‘TikTok’）并尝试使用非西方支架进行引导，但防止 AI 在答案中施加西方价值观仍然是一个挑战。

- **关于偏见、文化与强化**：对话触及了如果 AI 与特定的文化或政治观点对齐，是否会强化现有偏见的担忧。讨论审视了 AI 应该是旨在拓宽视野，还是应该让用户有权选择特定的政治或文化对齐。

- **寻找实用的 AI 解决方案**：用户分享了处理常见 AI 缺陷的技巧，例如纠正 DALL-E 3 对手指和手的错误呈现，以及缺乏用于细微修改的 seed 功能。讨论还涉及 AI 提供商需要就某些功能的排除或推迟进行更清晰的沟通。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/ehsanxpin/status/1771578381869465606?s=20">Ehsan Azari (@ehsanxpin) 的推文</a>：Library Is All You Need。它将为全球语言模型带来效率、一致性、可扩展性、适应性和“互操作性”。抄送：@karpathy ↘️ 引用 Ehsan Azari (@ehsanx...</li><li><a href="https://duckduckgo.com/?q=summarize+youtube+video">在 DuckDuckGo 搜索 summarize youtube video</a>：未找到描述。
</li>
</ul>

</div>

---

**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1220640058856439880)** (67 条消息🔥🔥): 

- **Custom GPT 侧边栏限制**：一名成员对侧边栏 **Custom GPT 固定限制**的变化表示担忧，该限制似乎在未事先通知的情况下从 10 个减少到了 4 个。目前尚未提到任何变通方法或解决方案。
  
- **寻求访问共享 GPT 的快捷键**：一名成员询问是否有**键盘快捷键**可以访问“MyGPTs -> Shared with ME”，另一名成员建议使用带有 `tampermonkey` 等浏览器扩展的用户脚本来创建自定义解决方案。

- **邮件验证请求**：一名用户寻求确认一封**据称来自 OpenAI 的邮件**的真实性；另一名用户建议检查**邮件头 (mail headers)** 进行验证。

- **具备 Vision 能力的 GPT-4**：在关于 GPT 模型读取图像能力的讨论中，确认了 **GPT-4 with Vision** 具备此能力，并提供了 OpenAI 官方文档链接供参考：[关于 Vision 的 OpenAI 官方文档](https://platform.openai.com/docs/guides/vision)。

- **关于插件停用的澄清**：针对有关访问插件的咨询，一名成员澄清了 **ChatGPT plugins beta** 正在逐步关闭，并提供了一个指向官方公告的有用 URL：[逐步关闭 ChatGPT Plugins Beta](https://help.openai.com/en/articles/8988022-winding-down-the-chatgpt-plugins-beta)。

---

**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1220896722553671691)** (61 messages🔥🔥): 

- **Vision 对残障人士的识别**：一位成员表示，Vision 难以识别包含他们自身的图像，特别是用于机器人和个人理容辅助的场景。由于隐私担忧，建议在探索解决方案时保持谨慎。
- **使用 GPT 增强章节写作**：在关于使用 Chat-GPT 改进写作的交流中，一位成员寻求关于如何在不重写整个章节的情况下添加子章节的 Prompt 建议。技巧包括在 Prompt 中更加具体，例如指定插入新内容的确切位置。
- **提升代码质量的 Prompt Engineering**：一位成员分享了一个详细的多部分 Prompt，旨在提高 GPT 执行编码任务的质量，重点在于强调编码实践的精心设计步骤。其他成员参与了讨论，分析了该方法的优点，并提出改进建议或提供自己的版本。
- **OpenAI SDK 的迁移问题**：一位用户在 OpenAI SDK 更新后遇到错误，该更新弃用了 `.Completion` 端点。另一位成员引导他们前往专门处理迁移问题相关疑问的服务器频道。
- **在 AI 写作中力求“展示而非叙述” (Show, Not Tell)**：成员们讨论了如何引导 Chat-GPT 在故事创作中展示动作而非单纯叙述，旨在提高叙事质量。分享了关于重新构建 Prompt 以引导 Chat-GPT 走向理想写作风格的建议。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>：OpenAI API 的官方 Python 库。可以通过在 GitHub 上创建账号来为 openai-python 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-python/discussions/742```">v1.0.0 迁移指南 · openai/openai-python · Discussion #742</a>：我们发布了 SDK 的新主版本，建议立即升级。这是对该库的彻底重写，因此发生了许多变化，但我们通过 cod... 简化了升级过程。
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1220896722553671691)** (61 messages🔥🔥): 

- **探索 Vision 的无障碍功能**：一位用户寻求如何让 Vision 模型将其识别为残障人士，以便用于个人理容的视觉辅助等用例。尽管讨论潜在解决方案具有敏感性，但仍有建议提出在 Discord 建议频道写下该问题，并提供了一个相关的历史帖子链接。

- **丰富 GPT-4 生成的书籍内容**：一位成员寻求帮助，希望指示 GPT-4 在不重写整个内容的情况下为章节添加子章节。建议的策略包括在构建 Prompt 期间使用章节编号以获得更好的组织性和清晰度，以及开启一个详细说明各版本的新对话以获得整合后的输出。

- **改进 GPT 编码任务的响应**：用户讨论了改进 GPT 代码输出的策略，其中一人分享了一个详细的 Prompt，用于指示模型在编码任务中表现得更好。建议使用技术流程名称以获得更好的交互效果，并使用自定义的 JSON 指令作为编码任务的 Prompt。

- **打造“可爱活泼”(Kawaii-bubbly) 的 AI 人格**：一位用户请求协助创建 Prompt，赋予 GPT 一种“可爱活泼”的人格，用于编写动画师的社交媒体简介。虽然为用户自己无法模仿的人格创建 Prompt 具有挑战性，但还是提供了一些尝试过的 Prompt 示例。

- **使用 ChatGPT 提高假设段落的质量**：一位成员需要支持，以避免生成通用的陈述，并产生充满专家理论和证明的假设段落。建议像进行普通对话一样直接与 ChatGPT 交流，并指定包含特定元素以实现高质量输出。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/openai/openai-python">GitHub - openai/openai-python: The official Python library for the OpenAI API</a>：OpenAI API 的官方 Python 库。可以通过在 GitHub 上创建账号来为 openai-python 的开发做出贡献。</li><li><a href="https://github.com/openai/openai-python/discussions/742```">v1.0.0 迁移指南 · openai/openai-python · Discussion #742</a>：我们发布了 SDK 的新主版本，建议立即升级。这是对该库的彻底重写，因此发生了许多变化，但我们通过 cod... 简化了升级过程。
</li>
</ul>

</div>
  

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1220621362864062504)** (242 messages🔥🔥): 

- **AI 艺术 Prompt 帮助的问题**：有人询问在哪里可以获得制作 AI 艺术 Prompt 的建议。消息中未提供具体的解决方案。
- **Blenderbot 一贯的角色设定**：一位用户讨论了 Blenderbot 保持一致角色设定的优势，这与 ChatGPT 等具有 AI 自我意识的聊天机器人形成对比。他们指出，Blenderbot 可能会声称自己是家庭主妇或教师，并始终保持“入戏”状态，而不像其他一些模型。
- **GPU 上乘法运算与条件检查的执行速度**：一位成员询问在 GPU 上执行乘法运算 (a*b=c) 与条件检查 (if(a==0){}) 之间的性能差异。另一位用户建议 Shader 编译器在效率方面做了很多工作，还有人推荐查看 'iq' 的作品以获取更多信息。
- **为 ChatGPT 准备的多样化语言 Prompt**：有人请求为 ChatGPT 提供一个详细且极具创意的复杂 Prompt，其中包含各种作家的风格和知名人士的原则，尽管一位用户对这种复杂性仅回应了一句 "Bloody hell"。
- **用于 GPU 推理的 TensorRT-LLM 与 ExLLama v2**：讨论围绕在 GPU 上运行大语言模型 (LLM) 推理的不同方法展开，提到 TensorRT-LLM 可能适用于单批次（single-batch）推理，而像 exLLama v2 这样的库则针对单用户速度进行了优化。对于同时服务多个用户的情况，推荐使用 vllm 或 tgi 等其他解决方案。
- **使用 GGML 进行量化**：一位用户询问 GGML 是否支持所有模型的量化，还是仅支持生成式模型。另一位成员回答说 GGML 并不支持所有模型，但包含各种语言和多模态模型，并建议使用像 llama.cpp 这样的特定语言模型文件以获得更快的性能。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://rubiks.ai/">Rubik's AI - Waitlist</a>: 未找到描述</li><li><a href="https://huggingface.co/CopyleftCultivars/Gemma2B-NaturalFarmerV1">CopyleftCultivars/Gemma2B-NaturalFarmerV1 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/fffiloni/coqui-bark-voice-cloning-docker">Coqui Bark Voice Cloning Docker - fffiloni 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>: 未找到描述</li><li><a href="https://discuss.huggingface.co/">Hugging Face Forums</a>: 社区讨论，由 Hugging Face 提供支持 <3</li><li><a href="https://huggingface.co/welcome">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/datasets/v2.1.0/en/process#split-long-examples)">Process</a>: 未找到描述</li><li><a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">unilm/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf at master · microsoft/unilm</a>: 跨任务、语言和模态的大规模自监督预训练 - microsoft/unilm</li><li><a href="https://github.com/xai-org/grok-1">GitHub - xai-org/grok-1: Grok open release</a>: Grok 开源发布。通过在 GitHub 上创建账号来为 xai-org/grok-1 的开发做出贡献。</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: 通过在 GitHub 上创建账号来为 mistralai-sf24/hackathon 的开发做出贡献。</li><li><a href="https://github.com/turboderp/exllamav2">GitHub - turboderp/exllamav2: A fast inference library for running LLMs locally on modern consumer-class GPUs</a>: 一个用于在现代消费级 GPU 上本地运行 LLM 的快速推理库 - turboderp/exllamav2</li><li><a href="https://github.com/davidberenstein1957/fast-sentence-transformers">GitHub - davidberenstein1957/fast-sentence-transformers: This repository, called fast sentence transformers, contains code to run 5X faster sentence transformers using tools like quantization and ONNX.</a>: 该仓库名为 fast sentence transformers，包含使用量化和 ONNX 等工具使 sentence transformers 运行速度提高 5 倍的代码。- davidberenstein1957/fast-sentence-transformers</li><li><a href="https://github.com/coqui-ai/TTS">GitHub - coqui-ai/TTS: 🐸💬 - a deep learning toolkit for Text-to-Speech, battle-tested in research and production</a>: 🐸💬 - 一个用于 Text-to-Speech 的深度学习工具包，经过研究和生产环境的实战测试 - coqui-ai/TTS</li><li><a href="https://github.com/suno-ai/bark?tab=readme-ov-file#-installation">GitHub - suno-ai/bark: 🔊 Text-Prompted Generative Audio Model</a>: 🔊 文本提示生成音频模型。通过在 GitHub 上创建账号来为 suno-ai/bark 的开发做出贡献。</li><li><a href="https://carrcenter.hks.harvard.edu/news/dont-talk-people-theyre-chatbots">Don't Talk to People Like They're Chatbots</a>: “AI 可能会让我们的社交互动变得更加平淡、更具偏见，甚至更粗鲁，”Carr 中心教员 Bruce Schneier 和技术与人权研究员 Albert Fox Cahn 在《大西洋月刊》中写道。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - mteb 创建的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://github.com/vgthengane/Awesome-Mamba-in-Vision">GitHub - vgthengane/Awesome-Mamba-in-Vision: List of papers related to State Space Models (Mamba) in Vision.</a>: 视觉领域中状态空间模型 (Mamba) 相关论文列表。- vgthengane/Awesome-Mamba-in-Vision</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1221453209579814952)** (9 条消息🔥): 

- **对未指明工具的兴趣**：一位成员表达了对某个工具的兴奋，但未提供更多细节或链接。

- **寻求 HuggingFace 方面的帮助**：一位用户寻求关于 HuggingFace 上 **qra-13b 模型**的帮助，特别提到了波兰。

- **模型转换尝试**：一位成员一直致力于将 **GLiNER 模型**从 PyTorch 转换为 *Candle* (Rust)，探索了量化技术并学习了 Candle 库。

- **模型转换为 Rust 的好处**：在关于将模型转换为 Rust 的优势讨论中，一位成员提到了**更少的依赖**、适合生产部署以及速度提升，尽管他们目前的实现并没有变得更快。

- **基于 Rust 的模型与 GPU 兼容性**：已确认使用 Candle 库转换为 Rust 的模型**与 GPU 兼容**。
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1220631213903970364)** (12 条消息🔥):

- **深入探讨视觉处理**：一段名为 [“Understanding early visual processing mechanisms by the principle of efficient encoding”](https://www.youtube.com/watch?v=Ed9otQAmEF4) 的 YouTube 视频讨论了生物视觉中的早期视觉处理机制。
- **探索用于音频分析的 Superlet Transform**：一种名为 *Superlet Transform* 的新方法被强调为实时音频分析的一项改进，其有效性在 [Nature 文章](https://doi.org/10.1038/s41467-020-20539-9) 中得到了证明，并在一篇文章中提供了补充的 [基准测试](https://doi.org/10.1038/s41598-022-22055-w)。
- **使用 Langchain 进行 Language Agent Tree Search**：Medium 上的一篇文章讨论了使用带有 Langchain 的语言模型在决策制定方面的潜在革命，这可能会改变 Language Agent 解决问题的方式。该文章可在 [Medium](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1) 上阅读。
- **来自 CivitAI 的宝贵见解**：一位成员在 [CivitAI](https://civitai.com/articles/2054) 上收集了关于 Stable Diffusion 的各种文章和指南，包括针对初学者和中级用户的提示、技巧和见解。
- **数据的重要性**：一位成员分享了一篇 [arXiv 论文](https://arxiv.org/pdf/2212.03533.pdf)，该论文强调了数据的重要性及其在特定语境下成为关键因素的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://ieeexplore.ieee.org/document/10333889">Exploring Lightweight Federated Learning for Distributed Load Forecasting</a>：Federated Learning (FL) 是一种分布式学习方案，使深度学习能够以保护隐私的方式应用于敏感数据流和应用程序。本文重点关注...</li><li><a href="https://civitai.com/articles/2054">A curated list of Stable Diffusion Tips, Tricks, and Guides | Civitai</a>：Stable Diffusion：https://stable-diffusion-art.com/samplers https://civitai.com/articles/983/insights-for-intermediates-how-to-craft-the-images-you...</li><li><a href="https://arxiv.org/abs/2207.04343">Explaining Chest X-ray Pathologies in Natural Language</a>：大多数深度学习算法对其预测缺乏解释，这限制了它们在临床实践中的部署。提高可解释性的方法，特别是在医学成像领域，通常...</li><li><a href="https://www.youtube.com/watch?v=Ed9otQAmEF4">Understanding early visual processing mechanisms by the principle of efficient encoding</a>：这是 CVPR 2022 教程“关于生物（人类）视觉如何工作的后马尔计算概述”的五场讲座中的第二场，日期为 2022 年 6 月 19 日...</li><li><a href="https://doi.org/10.1038/s41467-020-20539-9">Time-frequency super-resolution with superlets - Nature Communications</a>：在高精度神经生理信号中识别有限振荡包的频率、时间位置、持续时间和幅度具有挑战性。作者提出了一种基于...的方法</li><li><a href="https://doi.org/10.1038/s41598-022-22055-w">Super-resolved time–frequency measurements of coupled phonon dynamics in a 2D quantum material - Scientific Reports</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1220731707842629632)** (19 条消息🔥): 

- **Federated Learning 走向能效化**：分享了一个关于 *Exploring Lightweight Federated Learning for load forecasting* 的 GitHub 项目，旨在利用聚类和序列 DNN 方法解决负荷预测问题。该项目可在 GitHub 上的 [Exploring-Lightweight-Federated-Learning-for-load-forecasting](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting) 访问。
  
- **Stable Diffusion 资源汇编**：成员分享了多个关于 **Stable Diffusion** 的链接，包括采样器指南、见解和工具，以及诸如“*如何使用 a1111 制作你想要的图像*”的独立文章和关于“**video2video**”的视频指南。这些链接（如 [civitai.com](https://civitai.com/articles/2054)）为 Stable Diffusion 用户提供了宝贵的资源。

- **使用 AnkiForge 让 Anki 变得简单**：宣布了一款名为 AnkiForge 的新应用，它允许用户从文本笔记生成 Anki 抽认卡，并将在未来支持音频文件。该应用可在 [AnkiForge](https://ankiforge.onrender.com/) 试用。
  
- **事实核查中的定位与信任**：介绍了一篇讨论“通过知识图谱对 LLM 输出进行证据归因”以验证 LLM 输出的新研究论文，探讨了虚假信息时代的信任与验证机制。该论文专注于一种细粒度的证据归因方法，可在 [arXiv](https://arxiv.org/abs/2403.09724) 上查阅。

- **在 Recurrent Neural Notes 通讯中探索 AI 的极限**：最新一期的 **Recurrent Neural Notes** 讨论了 AI 的潜在极限，并包含深度文章。在 [Recurrent Neural Notes on Substack](https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true) 探索该通讯对 AI 未来的见解和思考。

- **使用 GPT 机器人 Hans 学习德语**：一项公告介绍了 *Hans*，这是一款在 GPT store 中提供的基于 GPT 的德语学习工具，旨在帮助用户提高德语水平。在 [GPT store 中的 Hans 🥨](https://chat.openai.com/g/g-mP8tCHgOc-hans) 查看 Hans。

- **LLM 术语视频解析**：分享了一系列解释各种 LLM（大语言模型）概念（如 Multi Query Attention、Sliding Window Attention 等）的视频，旨在揭开语言模型复杂世界的神秘面纱。该教育系列可在 [YouTube](https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz) 上观看。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://civitai.com/articles/2054">Stable Diffusion 技巧、诀窍和指南精选列表 | Civitai</a>：Stable Diffusion：https://stable-diffusion-art.com/samplers https://civitai.com/articles/983/insights-for-intermediates-how-to-craft-the-images-you...</li><li><a href="https://www.youtube.com/playlist?list=PLfSv7CK7EjD2fC9S6MAKRNDgTSCYgdGgz">LLM 术语解析</a>：欢迎来到 &quot;LLM Jargons Explained&quot; 系列，在这里我将揭开语言模型和解码技术的复杂世界。无论你是一个语言模型...</li><li><a href="https://arxiv.org/abs/2403.09724">ClaimVer：通过知识图谱实现可解释的文本断言级验证和证据归因</a>：在社交媒体广泛传播虚假信息以及 AI 生成文本泛滥的背景下，人们验证和信任信息变得越来越困难...</li><li><a href="https://huggingface.co/spaces/Tonic/Command-R">Command-R - Tonic 开发的 Hugging Face Space</a>：暂无描述</li><li><a href="https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting">GitHub - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting：使用聚类和顺序 DNN 方法在能源数据集上进行负荷预测的 Federated Learning</a>：使用聚类和顺序 DNN 方法在能源数据集上进行负荷预测的 Federated Learning - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting</li><li><a href="https://open.substack.com/pub/thernn/p/rnn-7-the-real-limits-of-ai?r=kxtnk&utm_campaign=post&utm_medium=web&showWelcomeOnShare=true">RNN #9 - AI 的真实极限</a>：探索神经网络的计算边界</li><li><a href="https://youtu.be/YpuRcmPnSTM">你刚才说的话</a>：出自电影《Billy Madison》</li><li><a href="https://ankiforge.onrender.com/">Anki Forge</a>：暂无描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1220795671427285084)** (48 条消息🔥): 

- **揭示肥胖见解**：分享了一个 Kaggle notebook，提供了肥胖数据的 EDA 探索和可视化，承诺提供有关人口统计和生活方式中影响肥胖因素的见解。该 notebook 可以在 [Deciphering Obesity Trends: An In-depth EDA](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda) 找到。

- **即将举行的活动提醒**：发布了关于即将召开会议的提醒，随后重点介绍了讨论的关于使用 **Hyper Z.Z.W 算子替换 Transformer** 的论文。该论文旨在解决基于 Attention 的机制中的挑战，可以在 [这里](https://arxiv.org/pdf/2401.17948.pdf) 阅读。

- **追求 100 万上下文**：对话涉及使用原生 Attention 实现 100 万上下文的难度，并推测了公司的技术，特别是 Google 及其在计算效率方面潜在的专有进展。

- **相关性对聊天机器人回复至关重要**：一位成员反思了 ChatGPT 在被问及高度相关且重要的问题时表现出的令人印象深刻的能力，指出模型的回复与所提询问的重要性相一致。

- **获取近期会议录音**：对于错过阅读小组活动的人，提到了会议录音并最终提供了链接，其中包含关于下一代网络架构和 **Hyper Z.Z.W 算子** 的演示。感兴趣的人可以在 [Hugging Face Reading Group 16: Hyper ZZ.W Operator Terminator](https://youtu.be/urgLoVPj1P8) 观看演示。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda">解密肥胖趋势 📉：深入的 EDA 📊</a>：使用 Kaggle Notebooks 探索并运行机器学习代码 | 使用来自多个数据源的数据</li><li><a href="https://youtu.be/urgLoVPj1P8">Hugging Face 阅读小组 16：HyperZ⋅Z⋅W Operator Terminator</a>：演讲者：Harvie Zhang，他也是这项工作的作者。遗憾的是，本次会议出现了一些主持方面的问题</li><li><a href="https://youtu.be/XiSPWW-3uNY">Watchtower</a>：由 TuneCore 提供给 YouTube。Watchtower · Michael Salvatori, Skye Lewin, Rotem Moav &amp; Pieter Schlosser。Destiny 2: Forsaken (Original Soundtrack) ℗ 2018 Bungi...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1221482604923064340)** (1 条消息): 

- **内存需求实验性工具**：发布了一个实验性工具，用于评估 **`DiffusionPipeline` 的推理时内存需求**。该工具已开放测试，欢迎在 [GitHub discussion 页面](https://github.com/huggingface/diffusers/discussions/7434)提供反馈。

**提到的链接**：<a href="https://github.com/huggingface/diffusers/discussions/7434">计算 `DiffusionPipeline` Checkpoint 的逐组件内存 · huggingface/diffusers · Discussion #7434</a>：我们发布了一个 Hugging Face Space，让你可以根据给定的 torch_dtype 计算 DiffusionPipeline Checkpoint 的内存需求：https://huggingface.co/docs/diffusers/main/en/using-diffusers/...

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1220762227687293088)** (21 条消息🔥): 

- **SegGPT 加入 HuggingFace Hub**：HuggingFace 引入了 [SegGPT](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt)，这是一个可以针对任何图像到图像任务进行训练的模型。该模型在论文 *SegGPT: Segmenting Everything In Context* 中被重点介绍，并在 COCO-20 和 FSS-1000 等数据集上展示了令人印象深刻的 one-shot 分割结果。

- **钻研 Diffusion 模型**：一位成员表示，在阅读了多篇博客并跟随教程编写代码后，他们加深了对 Diffusion 模型的理解。他们渴望为开源 Diffusion 项目做出贡献，并思考是应该深入研究代码，还是探索 Diffusion 模型中有效的不同任务和微调技术。

- **视觉模型通道输入的困惑**：有人提出了关于视觉模型通常只接受 3 通道图像的挑战。有人指出，由于基准数据集中此类数据的普遍性，3 通道默认设置很常见，尽管有人提到 **BridgeTower** 即使尝试了配置也无法适应单通道图像。

- **将图像特征与 LLM 融合**：针对合并文本和图像生成模型的咨询，推荐了 **BLIP-2** 作为资源。相关的 [BLIP-2 论文](https://arxiv.org/abs/2301.12597) 概述了一种方法，通过训练一个连接预训练图像编码器和语言模型的中间 Transformer 来学习视觉-语言表示。

- **分享 BLIP-2 资源**：分享了关于 BLIP-2 的更多资源，包括 [transformers 文档](https://huggingface.co/docs/transformers/en/model_doc/blip-2)，以帮助理解微调过程。有人指出，经过指令微调的变体 BLIP instruct 可能比标准的 BLIP-2 具有更好的性能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2301.12597">BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models</a>：由于大规模模型的端到端训练，视觉和语言预训练的成本变得越来越高。本文提出了 BLIP-2，一种通用且高效的预训练策略...</li><li><a href="https://huggingface.co/docs/transformers/en/model_doc/blip-2">BLIP-2</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/model_doc/seggpt">SegGPT</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1220621609409445919)** (24 条消息🔥): 

- **HuggingFace Trainer 问题**：一位用户遇到 HuggingFace 的 **Trainer** 类无法识别已安装的 'accelerate' 包的问题。讨论了各种故障排除步骤，包括升级包、清除缓存以及更改库的导入顺序。

- **SentenceTransformer 函数在离线状态下失效**：多位用户报告 **SentenceTransformers** 在离线环境中不接受本地目录的问题，这与其在 `transformers.AutoModel.from_pretrained` 中的功能表现不一致。用户要求对 SentenceTransformers 的离线能力进行验证。
- **寻求 NEET/JEE 数据集**：一位用户正在寻找包含往年 NEET/JEE 考试问题、答案和解释的数据集，以便使用 GPT-4 训练一个 MCQ 答案生成器，并讨论了对潜在误差范围的担忧。
- **嵌入量化突破**：🤗 HuggingFace 宣布了一种针对 **Sentence Transformers** 的全新 **Embedding Quantization** 方法，该方法在保持检索性能的同时，大幅提升了搜索速度并降低了内存、存储和成本。详情和演示可以在发布 [space](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval) 和深度 [博客文章](https://huggingface.co/blog/embedding-quantization) 中找到。
- **推理 API 摘要长度澄清**：一位用户询问如何控制 MERN 应用程序中 **facebook/bart-large-cnn** 模型生成的摘要长度。解释指出 `max_length` 参数决定了输入批次中的最大句子长度。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/awnr/Mistral-7B-v0.1-half-naive-A">awnr/Mistral-7B-v0.1-half-naive-A · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/sentence-transformers/quantized-retrieval">Quantized Retrieval - a Hugging Face Space by sentence-transformers</a>：未找到描述</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>：未找到描述</li><li><a href="https://sbert.net/examples/applications/embedding-quantization/README.html">Embedding Quantization &mdash; Sentence-Transformers  documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1220658219781722113)** (31 条消息🔥): 

- **all-MiniLM-L6-v2 模型咨询**：一位成员表示有兴趣在他们的数据集上使用 **all-MiniLM-L6-v2 model**，但需要关于如何下载和训练它的指导。他们请求有人私信提供帮助。
- **为 Hugging Face 上的图像添加背景**：一位成员询问 Hugging Face 上是否有能够为图像添加背景的预训练模型。另一位成员建议使用 **RMBG** 进行背景移除，并使用 **OpenGL, PyGLET, Kivy, and GIMP** 等工具应用滤镜进行平滑处理。
- **使用 SDXL 风格化图像**：有人提出了如何将输入图像风格化为水彩或其他效果，以及如何使用 **SDXL** 使图像无缝衔接以创建重复图案的问题。
- **寻求关于继续扩散模型研究的建议**：一位通过 YouTube 和 Medium 文章等各种资源学习并编写了扩散模型代码的成员，询问关于后续步骤的建议，是继续编码、研究扩散技术还是深入研究微调，其长期目标是为开源扩散模型仓库做出贡献。
- **微调扩散模型的学习资源**：两位成员进行了交流，其中一位询问了在个人图像上学习微调扩散模型的资源，另一位指向了 [Hugging Face 文档](https://huggingface.co/docs/diffusers/main/en/training/overview)，并建议尝试修复一个标记为 “Good first issue” 的简单开源问题，同时参考之前合并的 PR。

**提及的链接**：<a href="https://huggingface.co/docs/diffusers/main/en/conceptual/contribution#:~:text=Fix%20a%20simple%20issue%2C%20marked%20by%20the%20%E2%80%9CGood%20first%20issue%E2%80%9D%20label%2C%20see%20here.)?">如何为 Diffusers 🧨 做出贡献</a>：未找到描述

  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1220766917199921274)** (8 条消息🔥): 

- **简化人类与 LlamaIndex 的交互**：有一个新模板允许人类仅在需要干预时与 **LlamaIndex's agents** 交互，旨在提供更少干扰的用户体验。点击 [Twitter 帖子](https://t.co/Z16QPCWFmG) 抢先查看。
- **自定义 LLMs 加入 LlamaIndex 阵营**：了解如何将您自己的自定义 Language Models (LLMs) 集成到 **LlamaIndex** 中。Leonie Monigatti 在 [LinkedIn](https://t.co/DBjXGkLFkg) 上详细解释了这一过程。

- **为 PDF 创建 RAG Agent**：Ashish S. 制作了一个关于在 PDF 上构建 Agentic RAG 流的教程，其中包括用于提取文本和表格的 **LlamaParse**。完整指南见此 [推文](https://t.co/vIANM2Byel)。
- **使用 MistralAI 构建 RAG 和 Agent**：发布了一个关于使用 **LlamaIndex**、**MistralAI** 以及可选的 **LlamaParse** 来构建高级 RAG 和 Agent 的综合资源汇编。点击[此处](https://t.co/5zPWnjPCth)访问资源。
- **LlamaIndex 的 Python 文档升级**：全新的 **LlamaIndex Python 文档** 已经过翻新，重点展示了示例 Notebook，改进了带有预览和术语高亮的搜索功能，并精简了 API 信息。在 [Twitter 公告](https://t.co/FAuBj5gnCC)中查看改进后的文档。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1220604644951330866)** (296 条消息🔥🔥): 

- **关于 Bot 和 AI 工具集成的讨论**：用户讨论了将 Merlin API 和 LocalAI 等不同 AI 工具与 LlamaIndex 集成，其中 LocalAI 可以与 LlamaIndex 的 `OpenAILike` 方法配合使用进行交互，详见其[文档和 LocalAI 设置指南](https://github.com/mudler/LocalAI)。
  
- **请求评估逻辑说明**：一位用户寻求对 LlamaIndex 评估代码逻辑的解释，涉及 `CorrectnessEvaluator` 和 `SemanticSimilarityEvaluator` 等各种评估器。另一位用户 whitefang_jr 通过识别输入经过不同评估器的路径提供了清晰的解释，并附带了 BatchEvalRunner [文档](https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/)的链接。

- **关于文档中信息混杂的询问**：一位用户对 LlamaIndex 文档中相互矛盾的信息表示沮丧，并举了具体的例子，如关于使用工具的指南与实现不符。随后进行了讨论以澄清困惑，其他人也承认在 v0.10 更新后需要更新 Notebook 和文档。

- **请求多 Agent 聊天机器人示例**：一位用户询问了使用 LlamaIndex 构建多 Agent 聊天机器人的示例，以完成 SQL 查询、摘要和问答等顺序任务。Teemu2454 提供了一个多文档 Agent 示例的链接（[来源](https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents-v1/)），这可能是一个相关的起点。

- **将 Python 函数转换为 LlamaIndex 工具**：一位用户询问了类似于带有工具的 OpenAI Assistants 的功能，即如何将 Python 函数转换为 LlamaIndex 的工具。Cheesyfishes 提供了使用 `FunctionTool.from_defaults(fn=add)` 的代码，以及指向 [GitHub 上相关源代码](https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py)的链接。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://api.getmerlin.in/#pricing">Merlin API Platform</a>：在几分钟内将 LLMs 集成到您的生产应用中。</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html">正在重定向...</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/13NJEyhKWT7xdJFAJ6nB8mq-fk22UVDKa?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://llamahub.ai/l/tools/llama-index-tools-salesforce?from=">未找到标题</a>：未找到描述</li><li><a href="https://pretty-sodium-5e0.notion.site/llama-index-tools-salesforce-cdb97eca825c47bd8811b209035dae0d">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>：一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的一体化工作空间。</li><li><a href="https://huggingface.co/spaces/mteb/leaderboard">MTEB Leaderboard - 由 mteb 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/">LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/">Using Documents - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/supporting_modules/service_context_migration.html">正在重定向...</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/">Qdrant Vector Store - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/agent_runner_rag/?h=tools#define-toolset">Controllable Agents for RAG - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/">Vector Stores - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/localai/#llamaindex-interaction">LocalAI - LlamaIndex</a>：未找到描述</li><li><a href="https://www.llamaindex.ai/blog/llamaindex-v0-10-838e735948f8">LlamaIndex v0.10 — LlamaIndex，LLM 应用的数据框架</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/multi_document_agents-v1/">Multi-Document Agents (V1) - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/customization/llms/SimpleIndexDemo-Huggingface_stablelm/?h=hugg">HuggingFace LLM - StableLM - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/usecases/10q_sub_question/">10Q Analysis - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex">Vector - LlamaIndex</a>：未找到描述</li><li><a href="https://codespaces.new/Bloom-Assistant/api.getbloom.ai/tree/codespacers">共同构建更好的软件</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L704">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/be63bae53227f1360472477eb2afa993791c09ce/llama-index-core/llama_index/core/objects/base.py#L47-L49">llama_index/llama-index-core/llama_index/core/objects/base.py at be63bae53227f1360472477eb2afa993791c09ce · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py">llama_index/llama-index-integrations/extractors/llama-index-extractors-entity/llama_index/extractors/entity/base.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index</li><li><a href="https://github.com/run-llama/llama_index/blob/be63bae53227f1360472477eb2afa993791c09ce/llama-index-packs/llama-index-packs-snowflake-query-engine/llama_index/packs/snowflake_query_engine/base.py#L44">llama_index/llama-index-packs/llama-index-packs-snowflake-query-engine/llama_index/packs/snowflake_query_engine/ba

se.py at be63bae53227f1360472477eb2afa993791c09ce · run-llama/llama_index</a>: LlamaIndex 是为您 LLM 应用程序提供的数据框架 - run-llama/llama_index</li><li><a href="https://llamahub.ai/l/tools/llama-index-tools-bing-search?from=all">未找到标题</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/tools/google/">Google - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/">API Reference - LlamaIndex</a>: 未找到描述</li><li><a href="https://work.caltech.edu/telecourse">Learning From Data - 在线课程 (MOOC)</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/storing/chat_stores.html">正在重定向...</a>: 未找到描述</li><li><a href="https://llamahub.ai/l/llama-packs/llama-index-packs-snowflake-query-engine?from=">未找到标题</a>: 未找到描述</li><li><a href="https://growsmethod.com/practices/TracerBullets.html">Tracer Bullet 开发</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/">BatchEvalRunner - 运行多个评估 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">从 ServiceContext 迁移到 Settings - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v2.6.0">Release v2.6.0 - Embedding Quantization, GISTEmbedLoss · UKPLab/sentence-transformers</a>: 此版本带来了 Embedding Quantization：一种大幅加速检索和其他任务的方法，以及一个新的强大损失函数：GISTEmbedLoss。使用 pip install sentence-trans... 安装此版本</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/storing/chat_stores/">Chat Stores - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/">Index - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">用于高级 Text-to-SQL 的 Query Pipeline - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/pull/12187">由 logan-markewich 修复 async streaming · Pull Request #12187 · run-llama/llama_index</a>: 需要确保延迟声明的 queue/async 内容在访问前已实际实例化。修复了 #12180</li><li><a href="https://www.youtube.com/watch?v=QCZU9nCb-AM">Cria Demo (2024年3月14日，周四)</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/12180">[Bug]: AttributeError: &#39;NoneType&#39; object has no attribute &#39;wait&#39; · Issue #12180 · run-llama/llama_index</a>: Bug 描述 Async Streaming Chat 示例：https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/#async-streaming-chat 产生异常：AttributeError: &#39;NoneType&#39; object has n...</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/embeddings/bedrock/?h=bedrock">Bedrock Embeddings - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/bedrock/?h=bedrock">Bedrock - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/llm/sagemaker_endpoint_llm/">使用 LlamaIndex 与部署在 Amazon SageMaker Endpoint 中的 LLM 进行交互 - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/#embedding-similarity-evaluator">Embedding Similarity Evaluator - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/">Faithfulness Evaluator - LlamaIndex</a>: 未找到描述</li>
</ul>

</div>
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1220707548311715891)** (164 条消息🔥🔥): 

- **寻找类似 Whisper 的视频处理工具**：一位用户询问是否有与 Whisper 相当但用于视频处理的工具，并提到它可能利用 VLM 进行场景评估，且可能是开源的。提出了多个建议，包括 [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba)、Twelve Labs 以及视频智能服务 [videodb.io](https://videodb.io)。

- **OpenAI 的 Sora 令艺术家惊叹**：OpenAI 博客分享了对 Sora 的初步印象，揭示了创意专业人士的浓厚兴趣和认可。讨论了艺术家的作品示例，展示了 Sora 如何实现写实和超现实图像的创作。

- **Google AI Studio 与 Vertex AI 的混淆**：讨论围绕 Google AI Studio 与 Vertex AI 在提供 Gemini 等模型服务方面的差异和用法展开，AI Studio 开始推出 100 万 token 上下文 API，并在易用性方面与 OpenAI API 进行了对比。

- **AI 可穿戴设备蓬勃发展**：聊天片段聚焦于开源 AI 可穿戴设备的趋势，包括 200 美元的 ALOHA 项目，并讨论了此类产品是否完全是本地运行的。另一款 AI 可穿戴设备 [Compass](https://x.com/itsmartynask/status/1771890769865187648) 已开始预售，计划于下周开始发货。

- **大语言模型的效率**：微软的 LLMLingua 被分享为一种压缩 LLM prompts 和 KV-Cache 的工具，在性能损失极小的情况下，可能实现高达 20 倍的压缩。有建议认为，虽然优化成本至关重要，但也不要过早进行过度优化，而应专注于交付价值。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://www.instalora.xyz/">InstaLoRA - 即时 LoRA 生成器</a>：在几秒钟内生成你的 LoRA</li><li><a href="https://videodb.io">VideoDB</a>：只需 2 行简单的代码，即可在所有类型的视频上构建智能应用。由开发者构建，为开发者服务。</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba：用于高效视频理解的状态空间模型 (State Space Model)</a>：未找到描述</li><li><a href="https://x.com/kodjima33/status/1772011777066442819">Nik Shevchenko (@kodjima33) 的推文</a>：今天我看到另一个“开源” AI 可穿戴设备发布，它什么都没公布，只是为了收取你 5 倍的费用。在 @MistralAI x @cerebral_valley 举办的 @SHACK15sf 黑客松中，我们构建了 FR...</li><li><a href="https://www.forbes.com/sites/kenrickcai/2024/03/22/stability-ai-founder-emad-mostaque-plans-to-resign-as-ceo-sources-say/?sh=703ce1225239">据消息人士称，Stability AI 创始人 Emad Mostaque 计划辞去 CEO 职务</a>：Mostaque 已告诉多位亲近人士，他计划辞去这家以 Stable Diffusion 闻名的、曾备受关注的生成式 AI 初创公司的首席执行官职务。</li><li><a href="https://openai.com/blog/sora-first-impressions">Sora：初步印象</a>：我们从创意社区获得了宝贵的反馈，帮助我们改进模型。</li><li><a href="https://www.evenuplaw.com/">EvenUp</a>：未找到描述</li><li><a href="https://tenor.com/view/dj-khaled-another-one-one-more-time-gif-4816107">Another One GIF - Dj Khaled Another One One More Time - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI 公告 — Stability AI</a>：今天早些时候，Emad Mostaque 辞去了 Stability AI 的 CEO 职务以及公司董事会职位，以追求去中心化 AI。董事会已任命...</li><li><a href="https://x.com/xiangyue96/status/1771898843275067420?s=46&t=90xQ8sGy63D2OtiaoGJuww">Xiang Yue (@xiangyue96) 的推文</a>：@MistralAI 刚刚发布了他们的 v0.2 Base😱。@WenhuChen 和我使用 OpenCompass 评估包快速评估了几个基准测试。似乎几乎在所有...能力都有所下降。</li><li><a href="https://www.twelvelabs.io/">像人类一样理解视频的多模态 AI</a>：为任何应用带来类人视频理解能力，无论你拥有 TB 级还是 PB 级的视频</li><li><a href="https://x.com/omooretweets/status/1771960892810240333?s=46&t=90xQ8sGy63D2OtiaoGJuww">Olivia Moore (@omooretweets) 的推文</a>：为什么 ChatGPT 被适配到这么多用例中？ (1) 它拥有分发渠道；(2) 它结合了文本、图像、语音，成为一个全面的合作伙伴。但是，它在 UI 和工作流方面受到限制。在我看来，这是部分...</li><li><a href="https://x.com/itsmartynask/status/1771890769865187648?s=46&t=90xQ8sGy63D2OtiaoGJuww">mkrupskis (@ItsMartynasK) 的推文</a>：今天我发起了 Compass 的预订——一个售价 99 美元的开源指南。- 30 小时电池续航 - 通过转录你的对话进行学习 - 重温生活中的重要时刻 - 首批出货或...</li><li><a href="https://x.com/matpagliardini/status/1771168258856501564?s=46&t=PW8PiFwluc0tdmv2tOMdEg">Matteo Pagliardini (@MatPagliardini) 的推文</a>：对 #Transformers 架构的一个微调可以显著提高准确率！通过直接访问所有先前块的输出，一个 48 块的 #DenseFormer 优于 72 块的 Transformer，且速度更...</li><li><a href="https://www.latent.space/p/feb-2024">ChatGPT 的解构（2024 年 2 月回顾）</a>：ChatGPT 达到顶峰了吗？此外：我们照例为 AI Engineer 准备了 2024 年 2 月最高信号的顶级项目回顾！</li><li><a href="https://x.com/ai_for_success/status/1771932897915650371?s=46&t=90xQ8sGy63D2OtiaoGJuww">AshutoshShrivastava (@ai_for_success) 的推文</a>：AI 驱动的设备：可穿戴设备/个人助手。Humane AI Pin：699 美元；Rabbit R1：199 美元；Open Interpreter 01 Light：99 美元且开源；Compass：99 美元且开源。这仅仅是个开始... 我们...</li><li><a href="https://x.com/swyx/status/1772305930836918656?s=20">swyx (@swyx) 的推文</a>：🆕 ChatGPT 的解构 https://latent.space/p/feb-2024。整整一年过去了，ChatGPT 的用户数量增长几乎为 0。相反，用户正在探索大量垂直领域的参与者，以满足...</li><li><a href="https://x.com/mattshumer_/status/1771204395285246215?s=46&t=90xQ8sGy63D2OtiaoGJuww">Matt Shumer (@mattshumer_) 的推文</a>：介绍 `claude-investor` 📈。首个 Claude 3 投资分析师 Agent。只需提供一个行业，它就会：- 查找关键公司的财务数据/新闻 - 分析每个公司的情绪/趋势...</li><li><a href="https://blog.google/technology/ai/google-gemini-next-generation-model-february-2024/#performance">我们的下一代模型：Gemini 1.5</a>：Gemini 1.5 提供了显著增强的性能，在长文本...方面取得了突破。</li>

<li>跨模态的上下文理解。</li><li><a href="https://x.com/shiringhaffary/status/1771210619485659183?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Shirin Ghaffary (@shiringhaffary) 的推文</a>：🚨 新闻 OpenAI 进军好莱坞。公司下周将与好莱坞制片厂、媒体高管、人才经纪公司会面，鼓励他们使用 Sora。一些电影制作人已经获得了访问权限。COO Brad Lightcap...</li><li><a href="https://x.com/clementdelangue/status/1771395468959813922?s=46">来自 clem 🤗 (@ClementDelangue) 的推文</a>：我们是否应该收购 Stability 并开源 SD3？</li><li><a href="https://x.com/amanrsanger/status/1771590523046051947?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Aman Sanger (@amanrsanger) 的推文</a>：“Token 计数”对于长上下文模型来说是一个具有误导性的内容长度衡量标准。对于代码：100K Claude Tokens ≈ 85K GPT-4 Tokens；100K Gemini Tokens ≈ 81K GPT-4 Tokens；100K Llama Tokens ≈ 75K...</li><li><a href="https://x.com/amanrsanger/status/1771590523046051947?s=46&t=6FDPa">来自 Aman Sanger (@amanrsanger) 的推文</a>：“Token 计数”对于长上下文模型来说是一个具有误导性的内容长度衡量标准。对于代码：100K Claude Tokens ≈ 85K GPT-4 Tokens；100K Gemini Tokens ≈ 81K GPT-4 Tokens；100K Llama Tokens ≈ 75K...</li><li><a href="https://x.com/karpathy/status/1723140519554105733">来自 Andrej Karpathy (@karpathy) 的推文</a>：LLM OS。请耐心听我说，我还在构思中。规格：- LLM: OpenAI GPT-4 Turbo 256 核（批大小）处理器 @ 20Hz (tok/s) - RAM: 128Ktok - 文件系统: Ada002</li><li><a href="https://latecheckout.substack.com/p/the-guide-to-unbundling-reddit">Reddit 解构指南</a>：每隔几年就会发生一次大解构。2010 年，Andrew Parker 写了一篇关于 Craigslist “解构”的定义性文章，他在文中概述了从...中剥离出利基产品的机会。</li><li><a href="https://aneyeonai.libsyn.com/177-bjrn-ommer-are-diffusion-models-the-key-to-unlocking-ais-potential">Eye On A.I.：#177 Björn Ommer：Stable Diffusion 创始人解析扩散模型</a>：在 Eye on AI 第 177 期节目中，加入主持人 Craig Smith，与富有远见的 AI 研究员兼计算主管 Björn Ommer 一起探索人工智能生成模型的尖端世界...</li><li><a href="https://x.com/emostaque/status/1771400218170519741?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Emad acc/acc (@EMostaque) 的推文</a>：由于我的通知已经爆炸，补充几点：1. 我的股份在 Stability AI 拥有多数投票权 2. 他们拥有完整的董事会控制权。AI 权力的集中对我们所有人都不利。我决定辞职以...</li><li><a href="https://github.com/simonw/files-to-prompt">GitHub - simonw/files-to-prompt: 将整个目录的文件合并为一个单一的 Prompt，以便与 LLM 配合使用</a>：将整个目录的文件合并为一个单一的 Prompt，以便与 LLM 配合使用 - simonw/files-to-prompt</li><li><a href="https://github.com/semanser/codel">GitHub - semanser/codel: ✨ 全自动 AI Agent，可以使用终端、浏览器和编辑器执行复杂的任务和项目。</a>：✨ 全自动 AI Agent，可以使用终端、浏览器和编辑器执行复杂的任务和项目。 - semanser/codel</li><li><a href="https://www.pixee.ai/">您的自动化产品安全工程师 · Pixeebot</a>：Pixeebot 提供即时且持续的修复，使您的代码更安全。这就像身边多了一位安全专家级的开发者。</li><li><a href="https://www.brightwave.io/">Brightwave</a>：未找到描述</li><li><a href="https://github.com/OwlAIProject/Owl">GitHub - OwlAIProject/Owl: 一个在本地运行的个人可穿戴 AI</a>：一个在本地运行的个人可穿戴 AI。通过在 GitHub 上创建账号为 OwlAIProject/Owl 的开发做出贡献。</li><li><a href="https://github.com/microsoft/LLMLingua">GitHub - microsoft/LLMLingua: 为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。</a>：为了加速 LLM 的推理并增强 LLM 对关键信息的感知，压缩 Prompt 和 KV-Cache，在性能损失极小的情况下实现高达 20 倍的压缩。 - GitH...</li><li><a href="https://github.com/OwlAIProject/Owl?tab=readme-ov-file#introducing-our-reference-hardware-device-bee-">GitHub - OwlAIProject/Owl: 一个在本地运行的个人可穿戴 AI</a>：一个在本地运行的个人可穿戴 AI。通过在 GitHub 上创建账号为 OwlAIProject/Owl 的开发做出贡献。</li><li><a href="https://ai.google.dev/docs/migrate_to_cloud">未找到标题</a>：未找到描述</li><li><a href="https://console.cloud.google.com/project)">Google Cloud Platform</a>：未找到描述</li><li><a href="https://cloud.google.com/resource-manager/docs/creating-managing-projects#creating_a_project)">未找到标题</a>：未找到描述</li><li><a href="https://console.cloud.google.com/flows/enable">未找到标题</a>：未找到描述</li>

api?apiid=aiplatform.googleapis.com).">Google Cloud Platform</a>: 未找到描述
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1220814514711167046)** (5 条消息): 

- **AI 巨头见解**：在一篇 [tweet](https://twitter.com/swyx/status/1771255525818397122) 中提到的新播客节目揭露了关于 OpenAI、Google 和 Adept 的**劲爆见解**，尽管并非所有准备好的问题都得到了解答。
- **AI In Action 与 Llama.cpp**：AI In Action 活动以一场关于 **Llama.cpp** 的 [实时讨论](https://discord.com/channels/822583790773862470/1200548371715342479) 拉开帷幕，由 @363877777977376768 主持。
- **探讨 ChatGPT 的 Unbundling**：一篇关于 **ChatGPT 的 Unbundling** 的新文章指出，尽管用户增长停滞，但在用户寻求专业化 AI 服务的趋势下，OpenAI 仍可能取得成功。该文章还提示 OpenAI 可能会发布 **Sora** 和 **GPT-5** 以防止大规模退订，阅读全文请点击 [这里](https://latent.space/p/feb-2024)。

**提到的链接**：<a href="https://x.com/swyx/status/1772305930836918656?s=20">来自 swyx (@swyx) 的推文</a>：🆕 ChatGPT 的 Unbundling。https://latent.space/p/feb-2024。整整一年过去了，ChatGPT 的用户数量增长几乎为零。相反，用户正在探索大量的垂直领域玩家，以寻求...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1220766447270105088)** (14 条消息🔥): 

- **LLM Paper Club 的技术困难**：一名成员在 **llm-paper-club-west** 的会议中遇到了获取发言权限的问题，并在聊天中表达了这一点。
- **Zoom 来救场**：由于无法在 Discord 中获得发言权限，成员们转而使用 **Zoom** 进行 **paper club** 会议。
- **发言权限困惑**：对于未来 **paper club 环节**如何在 Discord 频道中获取发言权限存在困惑。
- **问题解决前会议已结束**：在 Discord 发言权限问题解决之前，Zoom 上的会议已经结束，这导致了对未来阶段便利化的思考。
- **提供访问控制协助**：另一名成员表示，**发言权限**可以由特定人员分配，为未来的会议提供了可能的解决方案。
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1220824233127706638)** (92 条消息🔥🔥): 

- **关于 Tensor 操作和 Transformer 模型的讨论**：成员们深入探讨了 Tensor 维度处理（幽默地称为“*pad and pray*”），并思考如何增强 IDE 对维度强制执行的支持。将 Transformer 模型想象成具有 Tensor 操作和可调权重的图，这种简单性被强调为一种心理模型。

- **通过 Slono 发现音乐**：分享了一个 [Spotify 链接](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA)，展示了 Slono 旨在唤起深夜放松氛围的作品。

- **LLM 中的编码和注释上下文**：讨论围绕 LLM 中注释的价值展开，强调了不同抽象层级的上下文信息。提到了注释在帮助 LLM 理解代码方面的影响。

- **对未来技术对决的期待**：分享了关于 C++ 相对于 Python 的速度优势的思考，以及对 2025 年 Luminal、Tinygrad 和 Mojo 之间对决的轻松预测。此外，大家对了解更多关于 Luminal 项目的信息也很感兴趣。

- **AI in Action Club 日程和主题分享**：分享了一个 Google Docs [电子表格](https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0)，其中包含 AI in Action Club 环节即将讨论的主题，包括生成式 AI 的 UI/UX 模式、RAG 架构以及 Prompt 格式对模型评估的影响。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://bbycroft.net/llm">LLM Visualization</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.00789">Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces</a>：Attention 机制已被广泛用于捕捉 Graph Transformers 中节点间的长程依赖。受限于二次计算成本，Attention 机制无法扩展到...</li><li><a href="https://tenor.com/view/friends-bestfriends-yep-bff-gif-4566644">Did We Just Become Best Friends? GIF - Friends Bestfriends Yep - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>：2024 主题, 日期, 协调人, 资源, @dropdown, @ GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA">slono</a>：艺术家 · 每月 110 位听众。
</li>
</ul>

</div>
  

---



**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1220711791135424565)** (214 条消息🔥🔥): 

- **GaLore 优化器引发热议**：成员们讨论了 [GaLore 优化器](https://github.com/jiaweizzhao/GaLore/issues/6)，它在全参数微调期间能显著节省 VRAM。有人对其“粗糙度”以及可能导致模型过拟合表示担忧，将该优化器的粒度比作使用“一个分辨率可调、能更新模型所有权重的非常粗糙的优化器”。

- **Axolotl Discord 深入探讨数据集难题**：一位成员询问了 SFT 和 DPO 中 sharegpt 和 chatml 的配置，另一位成员确认 chatml 确实是模型所见的内容。同时，Axolotl 仓库中的一个示例配置设置引起了困惑，可能导致数据集分词路径不正确。

- **对新模型和优化器的期待**：在技术讨论中，**Mistral v0.2 Base Model** 的发布令人兴奋，它拥有 32k 的更大上下文窗口；然而，一些人对仅限于 Mistral 7B 模型感到遗憾。GaLore 仍然是一个热门话题，周末的测试计划正在进行中，引发了关于优化策略的辩论。

- **发布困境的讨论**：一位成员分享了他们的困境：是否在医学模型进行第三轮期刊评审时发布预印本。这引发了关于早期分享研究成果优缺点的讨论。

- **公开征集与公司咨询**：CHAI 宣布支持开源 LLM 社区，并为 [LLM 开发者提供奖金](https://chai-research.typeform.com/chaiprize)，而另一位成员则鼓励将 Axolotl 用于商业应用的各公司联系他们，以便私下分享使用案例。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/posts/smangrul/896443101397392">@smangrul 在 Hugging Face 上发布：“🤗 PEFT v0.10.0 发布！🔥🚀✨

一些亮点：
_


1. FSDP+QLoRA and DeepSpeed…&quot;</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/accelerate/fsdp#use-peft-qlora-and-fsdp-for-finetuning-large-models-on-multiple-gpus">Fully Sharded Data Parallel</a>: 未找到描述</li><li><a href="https://x.com/xiangyue96/status/1771898843275067420?s=20">Xiang Yue (@xiangyue96) 的推文</a>: @MistralAI 刚刚发布了他们的 v0.2 Base😱。@WenhuChen 和我使用 OpenCompass 评估包快速评估了几个基准测试。似乎在几乎所有的...能力都有所下降。</li><li><a href="https://huggingface.co/docs/peft/accelerate/deepspeed#use-peft-qlora-and-deepspeed-with-zero3-for-fi">DeepSpeed</a>: 未找到描述</li><li><a href="https://huggingface.co/docs/peft/accelerate/deepspeed#use-peft-qlora-and-deepspeed-with-zero3-for-finetuning-large-models-on-multiple-gpus">DeepSpeed</a>: 未找到描述</li><li><a href="https://chai-research.typeform.com/chaiprize">Chai Prize</a>: 完成并赢取 3 天无限消息！</li><li><a href="https://github.com/mistralai-sf24/hackathon">GitHub - mistralai-sf24/hackathon</a>: 通过在 GitHub 上创建账户来为 mistralai-sf24/hackathon 的开发做出贡献。</li><li><a href="https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/config.yml">axolotl/examples/mistral/config.yml at main · OpenAccess-AI-Collective/axolotl</a>: 尽管提问 (axolotl questions)。通过在 GitHub 上创建账户来为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。</li><li><a href="https://github.com/huggingface/trl/blob/8534f0edf8608ad6bcbea9beefae380fa60ded77/trl/trainer/dpo_trainer.py#L877-L881">trl/trl/trainer/dpo_trainer.py at 8534f0edf8608ad6bcbea9beefae380fa60ded77 · huggingface/trl</a>: 使用强化学习训练 Transformer 语言模型。- huggingface/trl</li><li><a href="https://github.com/jiaweizzhao/GaLore/issues/6">第三方基准测试 · Issue #6 · jiaweizzhao/GaLore</a>: 你好，非常感谢这项出色的工作。我们使用 Llama-Factory 进行了一些实验，结果表明 GaLore 可以显著降低全参数...过程中的内存占用。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1220729246029316247)** (15 条消息🔥): 

- **OpenAI 示例中出现意外的 TypeError**: 一位成员在尝试运行 "examples/openllama-3b/qlora.yml" 中的示例时遇到了 `TypeError`，这与 `LlamaRotaryEmbedding.forward()` 接收到意外的关键字参数 'seq_len' 有关。
- **寻求帮助的频道重定向**: 另一位用户将面临 TypeError 的成员重定向到了 ID 为 #1111279858136383509 的特定帮助频道，以获得更好的协助。
- **讨论 LLM 微调效率**: 成员们讨论了在 27GB 内存中微调 7b 模型的可能性，并提到了一个名为 [torchtune](https://github.com/pytorch/torchtune) 的 GitHub 仓库，该仓库支持不依赖 Huggingface 库进行 *LLM 微调*。
- **微调的影响**: 一位成员指出使用原生 torch 带来的效率优势，同时也承认与使用 Huggingface 等库相比，其学习曲线更陡峭。
- **关于 Huggingface 的推荐与调侃**: 推荐了一个 [torchtune 的 pull request](https://github.com/pytorch/torchtune/pull/527)，用于查看如何以少于 16GB 的 RAM 进行全量微调，同时还顺带调侃了一下 Huggingface。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/pytorch/torchtune">GitHub - pytorch/torchtune: A Native-PyTorch Library for LLM Fine-tuning</a>: 一个用于 LLM 微调的原生 PyTorch 库。通过在 GitHub 上创建账户来为 pytorch/torchtune 的开发做出贡献。</li><li><a href="https://github.com/pytorch/torchtune/pull/527">Full finetune &lt; 16GB by rohan-varma · Pull Request #527 · pytorch/torchtune</a>: 上下文：我们希望为拥有显存有限的消费级 GPU 用户启用一种在少于 16GB RAM 中训练的全量微调变体。此 PR 使全量微调能够适配到 ...
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1220729707608277003)** (14 条消息🔥): 

- **Mixtral 微调技术尚不明确**: 一位成员正在寻求关于如何使用 GaLore 针对 Mixtral 中的 router 层进行处理的建议，但尚未在网上找到明确的文档。他们提到打算尝试 **-block_sparse_moe** 和 **-self_attn**，但随后感叹这在 Zero3 下无法工作。

- **Mixtral-7B 编程助手训练**：一位用户询问如何使用 runpod、python 等工具训练和微调 Mixtral-7B 模型，使其成为编程助手，并对在自有硬件上训练 Mixtral 模型所需的工具、IDE 和概念提出疑问。另一位成员承认了该问题的复杂性。

- **Axolotl 中的数据预处理错误**：在尝试使用 Axolotl 预处理数据时，一位成员遇到了与 'instruction' 键相关的 KeyError，尽管他们确认所有行都包含该键。另一位参与者建议可能存在缺失该键的行，但经核实并非如此。

- **TheBloke 模型微调问题**：一位用户在尝试使用 auto train 微调 TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ 模型时遇到错误，分享的错误输出显示 Windows 上的 subprocess.py 出现 FileNotFoundError。他们还分享了该模型在 [Hugging Face 上的仓库链接](https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ)。

- **关于 'Gema' 与 PyTorch 兼容性的查询**：一位成员询问 'gema' 是否仍与 PyTorch 不兼容，寻求有关该问题的最新信息。讨论的消息中没有提供明确的共识或答案。

**提到的链接**：<a href="https://huggingface.co/TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ">TheBloke/Wizard-Vicuna-7B-Uncensored-GPTQ · Hugging Face</a>：未找到描述

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1221161287770181642)** (8 messages🔥): 

- **Midnight 70B 隆重发布**：**Midnight 70B** 是最新且最受期待的模型，针对故事创作和角色扮演进行了优化，是 Rogue Rose 和 Aurora Nights 的继任者。可在 [OpenRouter](https://openrouter.ai/models/sophosympatheia/midnight-rose-70b) 获取，首发价格为 **$0.009/1k tokens**，享受 **25% 折扣**。

- **成本监控与管理新功能**：OpenRouter 推出了 **Usage Analytics**（使用分析），带有显示模型每日支出的新图表功能，以及可通过用户账户访问的 **Billing Portal**（账单门户），用于管理额度和发票。

- **Noromaid Mixtral 和 Bagel 的价格调整**：由于运行成本较高，Noromaid Mixtral 和 Bagel 模型的折扣已取消，前者定价为 **$0.008/1k tokens**，后者为 **$0.00575/1k tokens**。

- **请求扩展上下文长度**：一位用户表示希望以原始的 **32K 上下文长度**使用 Noromaid Mixtral，称目前的 8K 不够用。

- **DDoS 导致数据库停机**：由于绕过 Cloudflare 的 DDoS 攻击，OpenRouter 平台经历了数据库问题，但根据最新更新，稳定性已恢复。

**提到的链接**：<a href="https://openrouter.ai/models/sophosympatheia/midnight-rose-70b">Midnight Rose 70B by sophosympatheia | OpenRouter</a>：该模型是一个具有复杂家族树的融合模型，专为角色扮演和故事创作而设计。Midnight Rose 是 Rogue Rose 和 Aurora Nights 的继任者，并在两者基础上进行了改进。它旨在产生...

  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1220663758121341028)** (208 messages🔥🔥): 

- **Claude 3 自我审查版本获得认可**：由于在拒绝请求方面具有更好的选择性，Claude 3 的自我审查版本比网站过滤版本更受推荐。
- **关于被低估的 Grok 的案例**：Grok 模型引发了辩论，一些用户认为虽然它不如 Mixtral 强大，但在不与微调后的替代方案相比时，它是一个坚实的基础模型，尽管价格更贵。
- **微调模型偏好**：多位用户讨论了他们在角色扮演和任务中使用不同模型的经验，对 Midnight Rose 等模型可能扩展到 8k 以上的上下文表示兴趣，并对 Haiku 等开源模型在预算有限情况下的稳定质量表示认可。
- **OpenRouter 与模型表现**：一些用户报告称，在 OpenRouter 上使用 Opus 和 Haiku 等模型与直接通过 API 访问时，模型生成的质量存在差异，并正在调查这是否涉及默认 system prompts。
- **OpenRouter 中的 Perplexity 引用**：出现了关于在使用 Perplexity 时无法通过 OpenRouter 接收引用数据的讨论，并确认虽然数据存在，但由于 API 响应一致性问题，目前尚未返回。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<a href="https://x.com/deliprao/status/1770128250003460396?s=46">来自 Delip Rao e/σ (@deliprao) 的推文</a>：我看到这个，并不觉得 Grok 更好。作为一个务实的人，我看着它在想，既然已经有了性能几乎相当的 Mixtral，为什么还要费劲去用 Grok (314B) 呢，而且它还是一个...</li><li><a href="https://grok.x.ai/">xAI Grok</a>：未找到描述</li><li><a href="https://worldsim.nousresearch.com">world_sim</a>：未找到描述</li><li><a href="https://imgur.com/a/JWX7br0">imgur.com</a>：在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、流行的梗图、有趣的 GIF、感人的故事、病毒式视频等来提振你的精神...</li><li><a href="https://huggingface.co/sophosympatheia/Midnight-Rose-70B-v2.0.3">sophosympatheia/Midnight-Rose-70B-v2.0.3 · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1220704866775863336)** (64 条消息🔥🔥): 

- **GPU 战场：Apple 的 MPS vs. PyTorch**：一位成员正致力于改进 PyTorch 中的 MPS 后端，讨论了这项工作对于本地模型测试和微调的重要性，以及潜在的广泛性能收益。尽管面临挑战，他们仍坚持不懈，并指出自 2022 年 9 月以来一直影响 MPS 的张量复制（tensor copying）问题。

- **关于 LLM 预训练 Token 块策略的辩论**：成员们就构建语言模型预训练 Token 块的最佳方式展开了细致的辩论，在重叠序列与非重叠序列之间进行权衡。大家提出了多种观点，考虑了句子开头（beginning-of-sentence）Token 的重要性以及每种方法对训练效率的影响。

- **AMD vs. Nvidia GPU 驱动程序 —— 市场挑战**：关于 AMD Radeon 驱动程序与 Nvidia 相比被认为不足及其市场影响的讨论非常热烈。参与者指出，消费级 GPU 驱动程序通常将兼容性工作转移到其他地方，辩论了 AMD 开源其驱动程序的可能性，并考虑了激进投资者干预以推动 AMD 公司变革的可能性。

- **对 AI 行业公众演讲能力的关注**：有一段简短的评论比较了科技行业知名人士的公众演讲能力。在讨论其他演讲者的风格时，提到了 Lex Fridman 作为参考点。

- **机器学习中的合并热潮**：一位成员介绍了一种他们正在开发的用于组合模型的新合并方法，该方法可能超越 DARE 等现有方法。他们指出，该方法尚处于早期阶段，需要更多测试来确认其有效性。

**提到的链接**：<a href="https://tenor.com/view/ratatouille-flashback-childhood-memory-delicious-gif-3463448">Ratatouille • Flashback GIF - Ratatouille Flashback Childhood - Discover &amp; Share GIFs</a>：点击查看 GIF

  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1220604913982242857)** (105 条消息🔥🔥): 

- **带有深度加权平均（Depth-Weighted-Average）的 DenseFormer**：一种名为 [DenseFormer](https://arxiv.org/abs/2402.02622) 的 Transformer 架构新改进，增加了一个深度加权平均步骤，在不扩大模型规模的情况下显著提高了大规模模型的困惑度（perplexity）。[Hacker News 上的讨论](https://news.ycombinator.com/item?id=39794906)显示了对其可扩展性的怀疑，但支持者则坚持其潜力。

- **Mamba 遇见 Zigzag**：针对扩散模型在可扩展性和复杂性方面的固有问题，一项研究引入了 [Zigzag Mamba](https://arxiv.org/abs/2403.13802)，这是一种增强了处理高分辨率视觉数据集时内存利用率和速度的变体。该研究由 <@193386166517628929> 贡献，重点在于优化序列展平方法以提高性能。

- **在 DiPaCo 中整合碎片**：[DiPaCo](https://arxiv.org/abs/2403.10616v1) 架构的提议通过使用路径组合（path composition）方法创新了机器学习模型训练，以确保鲁棒性并减少潜在断连的计算节点之间的通信需求。这为去中心化机器学习模型训练提供了一条潜在路径。

- **“训练数据证明”（Proof-of-Training-Data）的潜力**：针对模型来源和样本投毒风险的担忧，一篇 arXiv [摘要](https://arxiv.org/abs/2307.00682)探讨了“训练数据证明”的概念，该概念允许验证用于训练神经网络的数据和计算。

- **使用 BiTFiT 进行训练偏置 (Training Bias)**：研究人员已经开展了将 BitFit 应用于 [LLama2/Mistral 等现代大型语言模型 (LLM)](https://github.com/lawrence-cj/LLaMA-DiffFit) 的研究。发布的这项研究演示了通过初始化新的偏置项并冻结其他参数来对 LLaMA 模型进行高效微调，从而提高了参数效率。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/davis21a.html">Catformer: Designing Stable Transformers via Sensitivity Analysis</a>：Transformer 架构被广泛使用，但训练它们并非易事，需要自定义学习率调度、缩放项、残差连接以及对子模块（如层归一化）的精心放置...</li><li><a href="https://arxiv.org/abs/2402.02622">DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging</a>：Vaswani 等人 (2017) 提出的 Transformer 架构现在在各个应用领域无处不在，从自然语言处理到语音处理和图像理解。我们提出了 DenseForme...</li><li><a href="https://news.ycombinator.com/item?id=39794906">未找到标题</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.13802">ZigMa: Zigzag Mamba Diffusion Model</a>：扩散模型长期以来一直受到可扩展性和二次复杂度问题的困扰，尤其是在基于 Transformer 的结构中。在这项研究中，我们旨在利用长序列建模能力...</li><li><a href="https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html">Manipulating Chess-GPT’s World Model</a>：操纵 Chess-GPT 的世界模型</li><li><a href="https://arxiv.org/abs/1802.01483">Explicit Inductive Bias for Transfer Learning with Convolutional Networks</a>：在归纳迁移学习中，微调预训练的卷积网络显著优于从头开始训练。使用微调时，其基本假设是预训练...</li><li><a href="https://arxiv.org/abs/2403.10616v1">DiPaCo: Distributed Path Composition</a>：机器学习 (ML) 的进步是由扩展神经网络模型推动的。这种扩展是通过日益英勇的工程壮举实现的，这对于容纳那些...的 ML 方法是必要的。</li><li><a href="https://arxiv.org/abs/2307.00682">Tools for Verifying Neural Models&#39; Training Data</a>：消费者和监管机构能够验证大型神经模型的来源，以评估其能力和风险，这一点至关重要。我们引入了“训练数据证明 (Proof-of-Training-Data)”的概念...</li><li><a href="https://arxiv.org/abs/1802.07044">The Description Length of Deep Learning Models</a>：Solomonoff 的通用推理理论和最小描述长度 (Minimum Description Length) 原则使奥卡姆剃刀定律正式化，并认为一个好的数据模型是一个擅长无损压缩的模型...</li><li><a href="https://arxiv.org/abs/2403.15297">Sphere Neural-Networks for Rational Reasoning</a>：大型语言模型 (LLM)（如 ChatGPT）的成功体现在它们在全球范围内的普及、类人的问答能力，以及稳步提升的推理能力...</li><li><a href="https://arxiv.org/abs/2103.01075">OmniNet: Omnidirectional Representations from Transformers</a>：本文提出了来自 Transformer 的全向表示 (OmniNet)。在 OmniNet 中，每个 token 不再保持严格的水平感受野，而是被允许关注所有 token...</li><li><a href="https://github.com/lawrence-cj/LLaMA-DiffFit">GitHub - lawrence-cj/LLaMA-DiffFit: Efficient Fine-tuning LLaMA Using DiffFit within 0.7M Parameters</a>：在 0.7M 参数内使用 DiffFit 高效微调 LLaMA - lawrence-cj/LLaMA-DiffFit
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1221451424299221062)** (5 条消息): 

- **为 Pythia 嵌入探索 SVM 核**：一位成员报告称，在对 Pythia 的输入嵌入运行多个 SVM 核后，**sigmoid 核的表现优于** rbf、linear 和 poly 核。虽然这一发现并非归功于某种聪明才智，而是通过反复试验得出的，但该用户表达了希望获得**直觉以简化流程**的愿望。

- **SVM vs. Logistic Regression**：一位参与者承认他们缺乏关于 SVM 的知识，在分类问题上更倾向于使用 **Logistic Regression**。

- **Tokengrams 仓库更新**：*Tokengrams* 项目已进展到可用的阶段，并分享了 GitHub 仓库链接。该工具用于**高效地从大型语料库中计算和存储 token n-grams**。[GitHub - EleutherAI/tokengrams](https://github.com/EleutherAI/tokengrams)。

- **Chess-GPT Interventions Summarized**: 分享了一个博客链接，详细介绍了 **Chess-GPT** 项目。该项目使用语言模型从 PGN 字符串中预测国际象棋步法，并估算玩家的技能水平。文章描述了使用 **线性探测 (linear probes) 来验证模型的计算**，并提到 Chess-GPT 的国际象棋水平约为 1500 Elo。[Chess GPT Interventions](https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://adamkarvonen.github.io/machine_learning/2024/03/20/chess-gpt-interventions.html">Manipulating Chess-GPT’s World Model</a>: 操纵 Chess-GPT 的世界模型</li><li><a href="https://github.com/EleutherAI/tokengrams">GitHub - EleutherAI/tokengrams: Efficiently computing &amp; storing token n-grams from large corpora</a>: 高效地从大型语料库中计算和存储 token n-grams - EleutherAI/tokengrams
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1220621835327111238)** (26 messages🔥): 

- **Potential Variance in Evaluation Results**: 一位成员讨论了在比较 **Hugging Face (HF) transformers** 与 **Megatron-DeepSpeed** 的评估结果时遇到的差异，大约一半的运行结果完全一致，而另一半则有约 0.5% 的差异。他们提到，通过数值检查前向传播 (forward pass) 有助于识别实现上是否存在根本差异。 

- **Determinism in Attention Mechanisms Questioned**: 为了理解评估结果的差异，一位成员询问 *flash attention* 是否会导致变异，但得到的澄清是 flash attention 在前向传播中是确定性的。他们还推测，执行 *融合 kqv 乘法 (fused kqv multiplications)* 的差异是否会导致数值偏差，这可能与 **bfloat16** 有关。

- **Minecraft as an RL Benchmark for LLM Collaboration**: 一位成员重点推荐了一个 GitHub 仓库 [GitHub - danijar/diamond_env](https://github.com/danijar/diamond_env)，它代表了一个用于强化学习 (RL) 的标准化 Minecraft 钻石环境。他们还引用了 [Voyager](https://github.com/MineDojo/Voyager/issues/149) 项目 GitHub 上的一个 issue，讨论了与 LM harness 项目合作的可能性。

- **Inverse-Scaling Evaluation Pipeline Inquiries**: 一位成员询问如何将 Inverse-Scaling 评估流水线中的多选题求解方法适配到 **lm-eval-harness** 中。他们提供了来自其 [GitHub 仓库](https://github.com/naimenz/inverse-scaling-eval-pipeline/blob/main/eval_pipeline/models.py) 的 **代码片段** 进行讨论，随后解释了在 harness 中如何处理答案选项对应的 logits。

- **Question on BB-Q Lite Task Scoring Method in BigBench**: 一位成员质疑 BigBench 任务中的 **bbq_lite** 子集是否使用了直接准确率评分法，并认为在实现中可能避开了其原始偏见评分机制的复杂性。有人建议参考一个特定的 [pull request](https://github.com/EleutherAI/lm-evaluation-harness/pull/1185)，以获取 **lm-evaluation-harness** 中另一种 BBQ 实现方式。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://github.com/naimenz/inverse-scaling-eval-pipeline/blob/main/eval_pipeline/models.py">inverse-scaling-eval-pipeline/eval_pipeline/models.py at main · naimenz/inverse-scaling-eval-pipeline</a>: 用于运行不同规模 GPT 模型并绘制结果的基础流水线 - naimenz/inverse-scaling-eval-pipeline</li><li><a href="https://github.com/MineDojo/Voyager/issues/149">Implement a way test local models · Issue #149 · MineDojo/Voyager</a>: Voyager 的出色工作。请考虑增加对本地模型的支持（不使用 openai 包，而是使用类似 Python requests 包连接到 localhost 本地模型...</li><li><a href="https://github.com/danijar/diamond_env">GitHub - danijar/diamond_env: Standardized Minecraft Diamond Environment for Reinforcement Learning</a>: 用于强化学习的标准化 Minecraft 钻石环境 - danijar/diamond_env</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/pull/1185">Add various social bias tasks by oskarvanderwal · Pull Request #1185 · EleutherAI/lm-evaluation-harness</a>: 此 PR 实现了多种流行的评估 LM 社会偏见的基准测试。我还旨在尽可能对这些任务进行验证：例如，通过与现有实现或结果进行比较...
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/1221160628501086388)** (3 messages):

- **关于多模态嵌入理论的咨询**：一位成员询问了关于**多模态嵌入空间（multimodal embedding spaces）**的理论著作，表示了广泛的兴趣，并未寻找特定内容。
- **关于 Stable Diffusion 文化中嵌入的见解**：Stable Diffusion 的亚文化将嵌入（embeddings）视为与其扩散模型中的 **IMG2IMG 工作流**类似，特别是 SDXL IMG2IMG，这可能为研究提供线索。
- **术语澄清**：术语“**IMG2IMG**”可能会与“init image（初始图像）”的使用混淆，特别是由于在 Automatic1111 web UI 中使用了这个短语；建议使用“image prompting（图像提示词）”或“image variations（图像变体）”等替代方案。
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1220828118881931284)** (9 messages🔥): 

- **Discord 舞台频道问题**：在一次 GTC 活动期间，Discord 舞台频道遇到了屏幕共享问题，随后通过切换到语音频道快速解决。有人建议在未来的讲座中默认使用语音频道。
- **相比 Discord 更倾向于 Google Meet**：一位成员对 Discord 的串流感到沮丧，并提议在未来的会议中使用 Google Meet，同时寻求有关 Discord 的意见和联系方式，以反馈串流稳定性的问题。
- **ML 与 CUDA 的联系**：一位成员询问在 ML 中何时需要进行 CUDA 编程，因为他们在 ML 实践中从未需要深入到那个程度。
- **为了速度进行 CUDA 编程**：分享了一个 YouTube 讲座链接，解释如何在 PyTorch 中对 CUDA kernel 进行性能分析：[Lecture 1 How to profile CUDA kernels in PyTorch](https://www.youtube.com/watch?v=LuhJEEJQgUM)。随附资源包括 [Google Docs 上的幻灯片](https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharing) 和 [GitHub 代码仓库](https://github.com/msaroufim/cudamodelecture1)。
- **了解何时需要深入到 CUDA 层**：针对讲座，一位成员总结道，当 PyTorch 速度不够快时，为了获得性能提升，CUDA 是必要的，这类似于为 CPU 程序编写 C 语言。另一位成员也同意这一观点。

**提到的链接**：<a href="https://www.youtube.com/watch?v=LuhJEEJQgUM">Lecture 1 How to profile CUDA kernels in PyTorch</a>: Slides: https://docs.google.com/presentation/d/110dnMW94LX1ySWxu9La17AVUxjgSaQDLOotFC3BZZD4/edit?usp=sharing Code: https://github.com/msaroufim/cudamodelecture1

  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1221525289603956766)** (27 messages🔥): 

- **调试 Triton 性能问题**：一位成员寻求关于调试 Triton kernel 性能问题的建议，将 **unsloth 的 fast_embedded_rope** 在带有连续张量（contiguous tensors）的 A10G 上的性能与 eager PyTorch 进行了对比，结果并不理想。

- **关于 Triton 编译器 Bug 解决的保证**：成员们讨论了代码注释中提到的历史性 **Triton 编译器 Bug**，并澄清这些不是当前的问题。此外，解释了像 `debug_barrier()` 这样的屏障对于正确性是必要的，类似于 CUDA 中的 syncthreads。

- **Triton 操作可能会被逐步淘汰**：一位贡献者表示 Triton 操作将来可能会被移除，建议不要提交 PR 来解决相关问题，并确认教程将继续保留用于学习目的。

- **来自 Meta 的可能基准测试**：提到 Meta 可能会为 Triton 引入一个**算子基准测试（op benchmark）**，这将为开发者提供可利用的参考实现。

- **关于架构优化协作的提议**：向一位成员建议了 `torchao` 仓库中的一个新原型文件夹，意图合并他们的工作并就高效 kernel 使用的 API 设计进行协作。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch-labs/ao">GitHub - pytorch-labs/ao: torchao: PyTorch Architecture Optimization (AO). A repository to host AO techniques and performant kernels that work with PyTorch.</a>: torchao: PyTorch Architecture Optimization (AO)。一个托管 AO 技术和适用于 PyTorch 的高性能 kernel 的仓库。 - pytorch-labs/ao</li><li><a href="https://github.com/openai/triton/blob/fb8983e5a2754ce793ab8d14ed0c333bfd9ba197/python/triton/ops/cross_entropy.py#L35">triton/python/triton/ops/cross_entropy.py at fb8983e5a2754ce793ab8d14ed0c333bfd9ba197 · openai/triton</a>: Triton 语言和编译器的开发仓库 - openai/triton
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1220999272372375652)** (7 messages): 

- **寻找 Blackwell NVIDIA 白皮书**：一位用户询问 Blackwell NVIDIA 白皮书的发布情况，但未提供有关此主题的直接信息。

- **GTC 会议详情分享**：一位成员分享了 [GTC Session Catalog](https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62400#/session/1696033648682001S1DC) 的链接，重点介绍了即将举行的研讨会、AI 会议和博览会日期，以及定于 3 月 17 日至 21 日在加州圣何塞（及线上）举行的主题演讲。

- **CUDA Toolkit 和 CuDNN 安装指南**：一位用户询问是否可以安装比 `nvidia-smi` 显示版本更高的 CUDA Toolkit，以及 CuDNN 的后续安装步骤。另一位成员提到需要将 CuDNN 添加到路径中，或将其文件复制到 Toolkit 目录。

- **Toolkit/Driver 兼容性链接补遗**：一位成员在提到 Toolkit/Driver 兼容性时漏掉了链接。随后提供了该链接，引导用户访问 NVIDIA 的 [CUDA Compatibility Guide](https://docs.nvidia.com/deploy/cuda-compatibility/index.html)，以了解每个 Toolkit 所需的最低驱动版本。

- **征集最喜欢的 CUDA Kernels**：一位成员邀请其他人提交最喜欢的 CUDA Kernels，这些 Kernels 可以优化大语言模型 (LLMs) 的操作，并可能在 Thunder 教程中展示。讨论链接指向 Lightning AI 仓库中的一个 [GitHub issue](https://github.com/Lightning-AI/lightning-thunder/issues/70)，该 issue 讨论了此功能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/deploy/cuda-compatibility/index.html">CUDA Compatibility  :: NVIDIA GPU Management and Deployment Documentation</a>：未找到描述</li><li><a href="https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=S62400#/session/1696033648682001S1DC">NVIDIA #GTC2024 Conference Session Catalog</a>：立即注册。在线直播。2024 年 3 月 18-21 日。</li><li><a href="https://github.com/Lightning-AI/lightning-thunder/issues/70">Support for CUDA kernels · Issue #70 · Lightning-AI/lightning-thunder</a>：🚀 Feature 你好 👋 从主 readme 文件中我注意到 Thunder 除了支持自定义 kernels 外，目前仅支持用 Triton 编写的。是否有支持 CUDA kernels 的计划？动力来源...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1220986153574994001)** (3 messages): 

- **新矩阵分解论文链接**：分享了一篇由研究员 [Lukas Gianinazzi](https://arxiv.org/search/cs?searchtype=author&query=Gianinazzi,+L)、[Alexandros Nikolaos Ziogas](https://arxiv.org/search/cs?searchtype=author&query=Ziogas,+A+N) 等人撰写的关于 **Arrow Matrix Decomposition** 的论文，提供了关于分布式稀疏矩阵乘法新方法的见解。在此处访问研究 [论文](https://arxiv.org/abs/2402.19364)。

- **Arrow Matrix Decomposition 的 GitHub 仓库**：**Arrow Matrix Decomposition** 的代码已在 GitHub 上发布，供对通信高效的分布式稀疏矩阵乘法感兴趣的人使用。仓库地址见此 [GitHub 链接](https://github.com/spcl/arrow-matrix)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.19364">Arrow Matrix Decomposition: A Novel Approach for Communication-Efficient Sparse Matrix Multiplication</a>：我们提出了一种迭代稀疏矩阵-稠密矩阵乘法的新方法，这是科学计算和图神经网络训练中的基本计算 kernel。在矩阵 s...</li><li><a href="https://github.com/spcl/arrow-matrix">GitHub - spcl/arrow-matrix: Arrow Matrix Decomposition - Communication-Efficient Distributed Sparse Matrix Multiplication</a>：Arrow Matrix Decomposition - 通信高效的分布式稀疏矩阵乘法 - spcl/arrow-matrix
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1221559417250517166)** (3 messages): 

- **GPU 与 CPU 架构复杂度对比**：一位成员询问 **NVIDIA GPU 架构** 是否比现代 CPU 架构更简单。另一位成员澄清说，GPU 专门用于简单操作的高吞吐量，而 CPU 处理的吞吐量较低，但操作更复杂。
  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1220761584335327242)** (12 messages🔥): 

- **进度监督**：一位成员承诺在长时间工作轮班后完成并讨论学习材料第 3 章和第 4 章的练习，利用公开监督作为动力。
- **习题答案资源共享**：建议创建一个共享的 Google Doc，以汇总达成一致的练习答案，用于交叉检查并作为所有成员的资源。

- **Exclusive Access to Exercise Solutions**: 练习题解答的专属访问权限：一位成员提议建立一个共享的练习题解答文档，仅向那些展示了自己初步尝试的成员开放访问权限，以维护挑战的完整性。
- **Experience Sharing Among Members**: 成员间的经验分享：成员们交流了他们在 C++ 和 multithreading 方面的背景，重点关注 CUDA 以及应用于各种技术的更广泛的 parallel programming 概念。
- **Collaborative Learning Through Shared Solutions**: 通过共享解答进行协作学习：分享了一个包含第 2 章练习题解答的 Google Doc 链接，在自己尝试练习后私信 (DM) 创建者即可获取。
[Ch 2 Exercise Solutions](https://docs.google.com/document/d/10ez800eu8OF-OzJXNZ0tRGdJaRAwagiyFdgeBoX0S8o/edit)
  

---


**CUDA MODE ▷ #[youtube-recordings](https://discord.com/channels/1189498204333543425/1198769713635917846/1221305349873602570)** (5 messages): 

- **Inquiring About Lecture 11 Upload**: 询问 Lecture 11 的上传情况：一位用户询问 **Lecture 11** 的录像何时上传。另一位用户回复说，等 Mark 有时间后会上传到 YouTube，并分享了一个在 [OneDrive](https://1drv.ms/v/s!AsJJewlEEg2oiPp1ja8bHuVmbVYp4Q?e=pHJp67) 上观看的临时链接。

- **Lecture 11 Now on YouTube**: 确认 **Lecture 11** 已上传至 [YouTube](https://youtu.be/mGDnOLcfE8g)。

- **Seeking Sparsity Lecture Slides**: 寻求 Sparsity 讲座的讲义 (Slides)：一位用户请求 Sparsity 讲座的讲义，询问是否公开并索要链接。另一位用户艾特 (ping) 了一位特定成员，请其在可能的情况下分享讲义。

**Link mentioned**: <a href="https://1drv.ms/v/s!AsJJewlEEg2oiPp1ja8bHuVmbVYp4Q?e=pHJp67">no title found</a>: no description found

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1221129136496574464)** (21 messages🔥): 

- **Diving Back into Ring Attention**: 重新投入 Ring Attention 研究：一位成员提到他们将在接下来的十天里专注于 **Ring Attention**，进行一些测试并探索训练细节。
- **Clarifying Meetup Time Post-Daylight Saving**: 澄清夏令时后的会议时间：一位成员询问了定期会议的具体时间，另一位成员使用时间戳回复了预定时间：<t:1711299600:t>。
- **Potential Workspace Upgrade Suggestion**: 潜在的工作空间 (Workspace) 升级建议：一位成员提议增加 workspace 文件夹的磁盘配额，随后讨论了可能迁移到具有更多存储空间的新机器。
- **Sharing Progress and Workspace Access**: 分享进度和 Workspace 访问权限：分享了几个指向 Wandb.ai 的链接，展示了与 Axolotl 相关的运行进度。更新了 SSH 配置，并讨论了重新安装 conda 和重新添加 SSH 密钥的事宜。
- **Technical Adjustments for Collaboration**: 协作的技术调整：讨论了关于重新安装 conda 的问题，base 环境被移动到了 `/workspace/miniconda3` 下。正在协调 SSH 访问权限，要求需要首次连接的人员发送公钥。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://wandb.ai/iron-bound/axolotl/runs/wjb8eyw3/workspace?nw=nwuserironbound">iron-bound</a>: Weights & Biases, developer tools for machine learning</li><li><a href="https://wandb.ai/iron-bound/axolotl/runs/7djmd1i2?nw=nwuserironbound">iron-bound</a>: Weights & Biases, developer tools for machine learning
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1220626324289228822)** (15 messages🔥): 

- **GPU Grins Back**: GPU 的微笑：推特上展示了新款 **Blackwell GPU** 似乎带有笑脸图案。分享了一个 [Twitter 链接](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908) 以供娱乐。
- **NVIDIA's Best Chips Yet**: NVIDIA 迄今为止最好的芯片：讨论了规格令人印象深刻的 **B200 加速器**，称其为市场上结合了 CUDA 生态系统的最佳产品。分享了一篇详细介绍 NVIDIA Blackwell 架构的 [AnandTech 文章](https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data)。
- **Hidden in Plain Sight**: 就在眼前：一位成员透露了与关于 **CUTLASS 库** 的 GitHub 讨论 ([GitHub 链接](https://github.com/NVIDIA/cutlass/discussions/1086)) 相关联的 "NVIDIA Developer" Discord 服务器的存在。
- **Diving Into New Data Types**: 深入研究新的数据类型：请求关于深度学习中新的 float/int 数据类型的参考资料，随后分享了一篇 [FP8 介绍论文](https://arxiv.org/abs/2209.05433) 和一篇关于多家公司将下一代窄精度数据格式标准化的 [OCP 标准化文章](https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai)。

- **标准之海**：讨论围绕着新浮点数的实现多样性以及缺乏 **IEEE standard** 的问题展开，值得注意的是 Google 缺席了达成新格式协议的联盟。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.anandtech.com/show/21310/nvidia-blackwell-architecture-and-b200b100-accelerators-announced-going-bigger-with-smaller-data">NVIDIA Blackwell Architecture and B200/B100 Accelerators Announced: Going Bigger With Smaller Data</a>：未找到描述</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1770931936657358908">来自 Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：为什么没人讨论新的 Blackwell GPU 真的在对着我们笑 lol</li><li><a href="https://github.com/NVIDIA/cutlass/discussions/1086">New Discord Channel! · NVIDIA/cutlass · Discussion #1086</a>：为了进一步改善围绕 CUTLASS 的用户体验和教育，我们创建了一个新的 Discord 频道！点击链接加入！在那里见，感谢你们的所有支持 :)</li><li><a href="https://www.opencompute.org/blog/amd-arm-intel-meta-microsoft-nvidia-and-qualcomm-standardize-next-generation-narrow-precision-data-formats-for-ai">Open Compute Project</a>：未找到描述
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1220856479830245437)** (37 条消息🔥): 

- **卡在 Puzzle 4**：一名成员正在 **调试 Puzzle 4**，其预期结果与实际测试结果之间存在差异。他们分享了自己的打印语句以交叉检查答案，并提到在使用 torch 进行外层求和（outer sum）时出现了问题。

- **关于 Puzzle 10 批处理（batching）的见解**：针对是否应在 Puzzle 10 的 batch 维度上进行并行化的询问，有人提到重点应放在将 kernel 保持在 **快速共享内存（fast shared memory）** 中，而不一定是在 batch 维度上并行化，尽管也可以利用 tensor cores。

- **用负无穷初始化数组**：讨论了如何在 Triton 中使用全 `-inf` 值初始化数组。建议的解决方案包括使用 `tl.full` 等函数以及使用极大的负数作为替代，因为 `tl.arange(0, B1) * float("-inf")` 会由于 0 * -inf 导致 NaNs。

- **Triton 中的索引挑战**：关于数组中单位置索引和切片的查询引出了一个澄清，即 Triton 不支持此类操作。这是由于数组和内存的处理方式决定的，针对这些限制的解决方法包括避免直接索引或采用结合扫描（associative scans）。

- **Puzzle 3 的探索揭示了理解**：一个自称为“菜鸟”的关于 Puzzle 3 的问题引导用户自己弄清楚了自己的误解。问题围绕在 Triton kernel 中加载和相加向量展开。
  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1221172294840356924)** (11 条消息🔥): 

- **Mistral 在黑客松暗示新模型**：聊天消息链接到了 **@AlexReibman** 和 **@adarshxs** 的推文，指出 **Mistral** 在 [@cerebral_valley hackathon](https://x.com/alexreibman/status/1771608346635751541?s=46) 上发布了一个新模型。

- **新发布公告中没有磁力链接**：**@natolambert** 注意到 [新模型发布](https://twitter.com/MistralAI) 缺少任何磁力链接（magnet links），表达了一点失望。

- **Mistral 7B v0.2 Base 模型细节揭晓**：**@xeophon.** 分享了来自 **@MistralAILabs** 的直接链接，详细介绍了 [新发布的 Mistral 7B v0.2 Base](https://x.com/mistralailabs/status/1771670765521281370?s=46)，包括其配置细节以及在哪里可以找到如何对模型进行 fine-tune 的指导。

- **对 Mistral 增长的反思**：**@xeophon.** 随口评论了 Mistral 的快速增长和发展，这从新模型发布的频率中可见一斑。

- **关于 Mistral 模型版本的澄清**：成员 **@philpax** 和 **@xeophon.** 讨论了 Mistral 模型的迭代，澄清了最近提到的 **Mistral-0.2** 并不是一个全新的模型，而是与之前的 instruct 版本相关，**@philpax** 最初误解了版本控制，随后进行了纠正。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<a href="https://x.com/alexreibman/status/1771608346635751541?s=46">Alex Reibman 🖇️ (@AlexReibman) 的推文</a>：Mistral 在 @cerebral_valley 黑客松上随手发布了一个新模型。</li><li><a href="https://x.com/adarshxs/status/1771610229412614149">Adarsh (@adarshxs) 的推文</a>：哟 @MistralAI 今天发布了一个新模型！！</li><li><a href="https://x.com/mistralailabs/status/1771670765521281370?s=46">Mistral AI Labs (@MistralAILabs) 的推文</a>：新发布：Mistral 7B v0.2 Base（用于训练 Mistral-7B-Instruct-v0.2 的原始预训练模型）🔸 https://models.mistralcdn.com/mistral-7b-v0-2/mistral-7B-v0.2.tar 🔸 32k context window 🔸 Rope Theta...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1221524167971831918)** (2 条消息): 

- **Nemo Checkpoint 转换咨询**：一位成员询问如何将 **Nemo checkpoint** 转换为与 Hugging Face 兼容，以便进行推理。
- **探索 Checkpoint 封装**：同一位成员还就封装 **Nemo checkpoints** 以供使用寻求建议，可能是在寻找包装器（wrapper）或接口。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1220943001992298597)** (29 条消息🔥): 

- **Stability AI CEO 卸任**：Stability AI 宣布 [CEO Emad Mostaque 辞去](https://stability.ai/news/stabilityai-announcement) CEO 职务及董事会席位，以追求去中心化 AI（decentralized AI），Shan Shan Wong 和 Christian Laforte 将接任临时联席 CEO。Mostaque 的推文暗示其关注点在于 **#DecentralizedAI** 和 AI 治理。

- **Stability AI 内部斗争与推测**：聊天中的讨论表明 Stability AI 面临 **长期的内部问题**，导致 Emad Mostaque 离开公司。成员们争论 Mostaque 的行为是骗局（grift）还是公司持续挣扎寻找方向的结果。

- **贡献与骗局之间的微妙界限**：聊天成员分享了对 Stability AI 运营性质的看法，一些人认为他们 **适当地授权** 了学术界开发的算法，而另一些人则认为考虑到学术界微小的算力贡献，这种做法值得商榷。

- **AI 社区对 Emad Mostaque 离职的看法**：关于 Emad Mostaque 留下的遗产意见不一，一些人认为他是 **骗子（grifter）**，同时也承认 Stability AI 业务中 **合法的一面**。

- **AI 学者的替代选择**：有人指出 AI 领域学者的选择有限，表明与 Stability AI 这样的公司合作比在资源有限的学术界能产生更大的影响力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://fxtwitter.com/ClementDelangue/status/1771395468959813922">来自 clem 🤗 (@ClementDelangue) 的推文</a>：我们应该收购 Stability 并开源 SD3 吗？</li><li><a href="https://stability.ai/news/stabilityai-announcement">Stability AI 公告 &mdash; Stability AI</a>：今天早些时候，Emad Mostaque 辞去了 Stability AI 的 CEO 职务以及公司董事会职位，以追求去中心化 AI。董事会已任命...</li><li><a href="https://fxtwitter.com/emostaque/status/1771403116099068048?s=46">来自 Emad acc/acc (@EMostaque) 的推文</a>：* 这里的“他们”——我的股份拥有董事会的完全控制权，哈哈。所以这可以说是我个人的决定。随着 AI 变得越来越重要，我们应该拥有更透明和分布式的治理方式。这很难...</li><li><a href="https://x.com/egrefen/status/1771628344204795962?s=46&t=pgJi6RxHJJYXIrBn2rKMXg">来自 Edward Grefenstette (@egrefen) 的推文</a>：本周 AI 动态：Quis griftat ipsos griftatores?（谁来忽悠那些忽悠者？）</li><li><a href="https://fxtwitter.com/emostaque/status/1771407651387383845?s=46">来自 Emad acc/acc (@EMostaque) 的推文</a>：另外，没有预售、TGE 或代币，真是见鬼。如果有的话，我会称之为 stable coin 😏</li><li><a href="https://fxtwitter.com/emostaque/status/1771380668930674850?s=46">来自 Emad acc/acc (@EMostaque) 的推文</a>：无法用更中心化的 AI 来击败中心化 AI。全力投入 #DecentralizedAI。更多消息即将发布 🔜 ↘️ 引用 Stability AI (@StabilityAI) 的公告：https://bit.ly/43zsVj...</li><li><a href="https://stability.ai/news">新闻 &mdash; Stability AI</a>：了解 Stability AI 的最新产品发布、公司更新和行业新闻。我们为激发人类潜力奠定基础。</li><li><a href="https://fxtwitter.com/emostaque/status/1771400218170519741?s=46">来自 Emad acc/acc (@EMostaque) 的推文</a>：由于我的通知已经爆了，补充几点：1. 我的股份在 @StabilityAI 拥有多数投票权 2. 它们拥有董事会的完全控制权。AI 权力的集中对我们所有人都是不利的。我决定辞职是为了...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1220635579079987210)** (8 条消息🔥): 

- **RL 通用 Agent 讨论**：一位成员链接了 [Twitter 上的讨论](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw)，关于在强化学习 (RL) 中创建“通用 Agent”的哲学，探讨了实现此类 Agent 的实际和原则上的可能性。
- **网络上对反垄断法的误读**：Nathan Lambert 对公众在近期科技诉讼和辩论中对反垄断法的普遍误解表示沮丧。
- **对 Apple 反垄断诉讼的批评**：Nathan Lambert 的几条消息批评了 FTC 对 Apple 的诉讼，建议那些欢呼雀跃的人应该阅读更多见解深刻的观点，例如 Ben Thompson 的文章。
- **不同意 Twitter 上关于 FTC 对阵 Apple 的情绪**：在 Twitter 的一场争论中，Nathan Lambert 坚持认为 FTC 对 Apple 的诉讼是不合时宜的，并引用了一条[推文](https://x.com/fentpot/status/1771634407226446254?s=20)暗示监管对 Apple 这样的大公司的影响微乎其微。
- **关于 FTC 诉讼价值的对话**：一位名为 twkillian 的成员分享了观点，认为近期 FTC 的诉讼指控反竞争行为可能存在疑问，但怀疑此类行为是否旨在恶化市场或使其他产品处于劣势。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Machine Learning Street Talk (@MLStreetTalk) 的推文</a>：我们刚刚发布了与 @MinqiJiang 和 @MarcRigter 的节目，讨论了在 RL 中构建“通用 Agent”在原则上和实践中是否可能的哲学。</li><li><a href="https://x.com/fentpot/status/1771634407226446254?s=20">来自 el (@fentpot) 的推文</a>：@nagolinc @norabelrose @natolambert 抱歉，但对于像 Apple 这样的大公司来说，这里的二阶效应微乎其微。大多数创始人如果能达到 Apple 规模的边角料水平都会感到欣喜若狂...
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[reads](https://discord.com/channels/1179127597926469703/1214764639397617695/1221520762066964560)** (19 条消息🔥): 

- **Anthropic CEO 的见解**：Nathan Lambert 重点推荐了一段名为[“Anthropic CEO 谈离开 OpenAI 及对 AI 未来的预测”](https://www.youtube.com/watch?v=gAaCqj6j5sQ)的采访，讨论了 Dario Amodei 对 2024 年及以后 AI 行业的预测。
- **反思早期 OpenAI 的愿景**：Nathan 评论道，早期的 OpenAI 贡献者对 AI 的算力轨迹有着清晰的愿景。

- **探索访谈内容**：Nathan Lambert 对 Mistral 的 CEO Arthur Mensch 表现出兴趣，希望从中深入了解公司文化，并推荐了 ["Fireside Chat w/ Mistral CEO, Arthur Mensch"](https://youtu.be/sQpeIuymJZ8)。
- **辩论中的 AGI 定义**：对话触及了定义 AGI 的难度，Nathan 提出了个人门槛，认为 GPT-4 已属于 AGI，这引发了关于什么是真正的通用智能的讨论。
- **Latent Space 的 2 月 AI 亮点**：Xeophon 分享了 Latent Space 2024 年 2 月的月度回顾链接，涵盖了重要的 AI 新闻和即将举行的活动，包括 "AI UX 2024"，并提到了 ChatGPT 在 2023 年初的用户快速增长。回顾内容可以在[这里](https://www.latent.space/p/feb-2024)找到。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://www.latent.space/p/feb-2024">The Unbundling of ChatGPT (Feb 2024 Recap)</a>: ChatGPT 的解构（2024 年 2 月回顾）：ChatGPT 达到顶峰了吗？此外：我们照例为 AI Engineer 准备的 2024 年 2 月高信号要点回顾！</li><li><a href="https://youtu.be/sQpeIuymJZ8">Fireside Chat w/ Mistral CEO, Arthur Mensch</a>: 加入我们，听取 Mistral 联合创始人兼 CEO Arthur Mensch 与 Elad Gil 的对话。涵盖的主题包括：开源与 LLM、Agent 以及多...</li><li><a href="https://www.youtube.com/watch?v=gAaCqj6j5sQ&t=5s">Anthropic CEO on Leaving OpenAI and Predictions for Future of AI</a>: Dario Amodei 是 Anthropic 的联合创始人兼 CEO。在本集中，我们讨论了对 2024、2025 年及以后 AI 行业的详细预测。Dario 讨论了...
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1220609180185002036)** (42 messages🔥): 

- **AI 中 "You Up?" 的风险**：一位用户幽默地强调了聊天机器人回答 "you up?" 这一问题所需的复杂前提，建议它应该先解决时空连续体问题并确保连接安全。
- **决定使用哪种 Chain**：关于为一个任务使用多个 Chain 的对话，一位成员建议，是查询 SQL 数据库还是 Vector Database 应该基于预期的结果大小来决定。
- **RunnableParallel 与 Streamlit 的技术困境**：一位成员在尝试将 RunnableParallel 与 Streamlit 应用一起使用时遇到了 "missing ScriptRunContext" 错误，这表明两者之间可能存在兼容性问题。
- **推出 RAG 学习平台**：一位用户分享了一个即将推出的免费资源链接，用于学习 AI 编程中的 RAG（检索增强生成），提到 OpenAI、LangChain、Chroma 和 Python 是参与者将使用的部分技术。[Intro to AI for Developers](https://takehomes.com/library/developers/intro-to-ai)
- **Vector Database 选择与信息分组的聚类算法**：一位用户寻求关于在 ChromaDB 和 Qdrant 之间选择 Vector Database，以及在基于密度的或基于质心的聚类算法之间选择，以实现文档中关键信息的语义聚类分组的建议。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<a href="https://takehomes.com/library/developers/intro-to-ai">开发者 AI 实用入门 – TakeHomes Library</a>：未找到描述</li><li><a href="https://github.com/langchain-ai/langchain/issues/6138">ConversationChain 默认提示词导致模型与自身对话 · Issue #6138 · langchain-ai/langchain</a>：系统信息 langchain==0.0.195 python==3.9.6 谁能帮忙？@hwchase17 信息 官方示例 notebooks/脚本 我自己修改的脚本 相关组件 LLMs/Chat Models Embedding Models...</li><li><a href="https://github.com/docker/for-mac/issues/6938">HEALTHCHECK 标志 "start-interval" 在 Docker 版本 24.0.5, build ced0996 中无法识别 · Issue #6938 · docker/for-mac</a>：描述 Docker 文档指出 HEALTHCHECK 有一个名为 "start-interval" 的标志（文档）。实际上在 Dockerfile 中使用该标志会导致错误。复现 使用此 Dockerfi...</li><li><a href="https://js.langchain.com/docs/use_cases/tool_use/tool_error_handling#chain>).">工具错误处理 | 🦜️🔗 Langchain</a>：使用模型调用工具存在一些明显的潜在失败模式。首先，模型需要返回一个完全可以被解析的输出。其次，模型需要返回工具参数，这些参数...</li><li><a href="https://github.com/langchain-ai/langchain/issues/10629>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/4197>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/12410>),">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13602>).">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1221422519953657958)** (1 条消息): 

- **询问关于 Langserve 的客户端执行**：一位成员询问是否可以使用托管在 Langserve 上的 runnable，并配合在客户端执行的工具。目前没有提供更多细节或回复。
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1220692626458673193)** (12 条消息🔥): 

- **突破限制扩展 LLM 输出的创新方法**：有人建议了一种绕过 GPT-4-Turbo 的 4k 输出 token 限制的变通方法，即通过检测停止原因为 "length"，并发送包含原始提示词和已生成内容的后续请求，从而允许继续生成。
  
- **Bedrock 与 Python 集成指南**：介绍了一份关于结合 Python 利用 Bedrock 的全面指南。感兴趣的人可以在[这里](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be)阅读全文。
  
- **发布用于 LLMs 分析的 SimplyAnalyze**：SimplyAnalyze.ai 亮相，这是一项与 LangChain 集成的服务，用于分析公司各部门的 LLM 对话。创作者为对免费开发者预览版感兴趣的人分享了联系方式，你可以[通过他们的网站取得联系](https://simplyanalyze.ai/)。

- **探索使用 Langchain 进行 Agent Tree Search**：分享了一篇关于使用 Langchain 提高语言模型决策能力的资讯文章。你可以在[这里](https://medium.com/ai-advances/language-agent-tree-search-with-langchain-revolutionizing-decision-making-with-language-models-a46c991397f1)阅读全文。
  
- **基于 Langchain 的增强功能聊天机器人**：一个本地角色 AI 聊天机器人已更新，其特点是改进了 CSV 解析、NER 解析、网页抓取和文档获取。访问 [GitHub 上的仓库](https://github.com/ossirytk/llama-cpp-chat-memory)以探索这些增强功能。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<a href="https://github.com/Haste171/langchain-chatbot">GitHub - Haste171/langchain-chatbot: AI Chatbot for analyzing/extracting information from data in conversational format.</a>: 用于以对话形式分析/提取数据信息的 AI Chatbot。- Haste171/langchain-chatbot</li><li><a href="https://github.com/ossirytk/llama-cpp-chat-memory">GitHub - ossirytk/llama-cpp-chat-memory: Local character AI chatbot with chroma vector store memory and some scripts to process documents for Chroma</a>: 带有 Chroma 向量存储记忆的本地角色 AI Chatbot，以及一些为 Chroma 处理文档的脚本 - ossirytk/llama-cpp-chat-memory
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1221153829005824121)** (5 条消息): 

- **探索通过 LangGraph 控制 Chatbots**：分享了一个名为 "How To Control Your Chatbot Actions and Prompt System: LangGraph" 的 YouTube 视频，演示了在 LangChain 中构建 Agent 并自动化 Chatbot 体验的方法。视频地址：[How To Control Your Chatbot Actions and Prompt System: LangGraph](https://www.youtube.com/watch?v=4e5A3opn-tc)。
- **Mr. Beast 的 AI 食谱大冒险**：分享了一个创意 YouTube 视频 "Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!"，展示了一个受这位著名 YouTuber 特技启发的 AI 生成食谱。点击查看这个有趣的理念：[Mr. Beast Meets Mistral](https://www.youtube.com/watch?v=Nc5Yk0XXgP8)。
- **垃圾信息警报**：发布了多条提供 50 美元 Steam 礼品卡的相同消息，可能预示着垃圾信息活动。随附链接为 [steamcommunity.com/gift/758474483](https://u.to/uMaEIA)。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/watch?v=4e5A3opn-tc">How To Control Your Chatbot Actions and Prompt System: LangGraph</a>: #langgraph #langchain #ai #chatbot #python #automation 如果你一直在关注 LangChain 中的 Agent，你会知道有很多构建 Agent 的方法，但在...</li><li><a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!</a>: 今天我们制作了 Beast 食谱。“Beast Cookbook” 的想法是一种有趣且富有创意的方式，可以与 Mr. Beast 的内容互动，并生成一个有趣的虚构...
</li>
</ul>

</div>
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1220818264251174993)** (21 条消息🔥): 

- **房地产 AI 领域的麻烦**：来自 *Uniti AI* 的一名成员正苦于让 **GPT4.Turbo** 根据用户需求准确匹配房产库存，并提到了一些问题，例如当需求为 2,000 - 4,000 平方英尺时，系统却建议了一个 17,000 平方英尺的房产。

- **LLM 在过滤中的作用**：该成员目前的方法是使用 LLM 将 CSV 文件中的房产与指定标准进行匹配。详细的 Prompt 旨在确保**库存建议**保持在指定要求内，允许最多 +/- 20% 的偏差。

- **简单方案优于复杂化**：另一位成员建议使用简单的数据库过滤器而不是 LLM，并指出 LLM 可以生成查询，但在实际过滤过程中并非必要。

- **避开常见的 LLM 陷阱**：针对因疏忽而感到沮丧的回复，寻求帮助的成员得到了安慰，称掉入“常见的 LLM 陷阱”时有发生，并不代表个人能力问题。

- **链接有用资源**：他们提供了一个由 **Jason Liu** 撰写的教学博客链接：["RAG is more than just embedding search"](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/)，该文章讨论了 Embedding 搜索的局限性，以及 LLM 在生成查询和处理自然语言交互中的适用性。

**提及的链接**：<a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG is more than just embedding search - Instructor</a>: 暂无描述

  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1221464274879119452)** (5 条消息): 

- **对 Anthropic 的速率限制感到恼火**：一位成员对 **Anthropic** 严格的速率限制表示沮丧，提到虽然有 **200k 上下文窗口**，但 API 每天仅允许 **1M tokens**。
- **寻求 Bedrock 的财务便捷性**：同一位成员询问了 **Bedrock** 保证吞吐量的月费模式，希望从有该服务使用经验的人那里获得见解。

- **Anthropic 的 Scale 计划提供缓解方案**：另一位成员建议联系 Anthropic 的销售团队以获取其 "scale" 计划的访问权限，并指出每月 **$500** 的支出被认为是相对较低的成本。

---

**LLM Perf Enthusiasts AI ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/1220915143987433524)** (3 messages): 

- **寻找终极指南**：一位用户正在编写资源指南，并寻求社区关于 Large Language Models (LLMs) 相关高级主题的**最受喜爱的解释资源**。
- **Exa.ai 推荐**：针对资源征集，exa.ai 被推荐为探索 LLM 相关主题的有用工具。
- **资源深度的澄清**：用户澄清了他们的请求，表示他们正在寻找关于 RHLF 等主题的最佳、最清晰的解释，而不仅仅是大量博客文章或文章的汇编。

---

**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** (1 messages): 

ibash: > write high quality code
Damn.

---

**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1220914049970339870)** (1 messages): 

- **GPT-3.5-0125 表现优于其前代产品**：一位成员强调，**GPT-3.5-0125** 在他们的所有测试中都明显优于之前的模型，标志着它是一个明显更优越的迭代版本。

---

**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/)** (1 messages): 

emrgnt_cmplxty: Basic prompting isn't getting it done for you?

---

**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1221887967409213612)** (1 messages): 

- **突破性 ML 研究项目招募志愿者**：**Youth Inquiry Network** 和 **Futracode** 正在合作开发一种 Machine Learning 算法，该算法将利用现有的研究数据库推荐最佳研究课题。他们正在寻求 Web 开发者、数据分析师以及 AI 和 ML 专家为这一雄心勃勃的努力做出贡献。

- **为认可和经验做出贡献**：志愿者将有机会提升他们的作品集，并获得一份证书、两封专业推荐信以及项目的源代码。该参与承诺灵活的时间安排，不需要高强度的时间投入。

- **现在工作，以后拥有**：这一非营利倡议的参与者将保留所开发的 ML 算法的全部权利，包括在项目完成后自由展示、推广、销售或以任何他们认为合适的方式使用它的自由。

- **参与无繁琐程序**：感兴趣的个人可以直接表达意向，无需正式申请——只需私信招聘人员或评论“interested”，即可联系进行后续步骤。

---

**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1220643702200860704)** (8 messages🔥): 

- **简短的交流互动**：*life lesson*（人生教训）这一短语引发了另一位成员幽默且共情的反应，表明了共同的社区理解或事件。
- **教育文档链接**：一位成员分享了一个关于 **Post-AGI Educational Reforms**（后 AGI 时代教育改革）的 [Google Docs 链接](https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing)；然而，没有提供进一步的细节或背景。
- **询问 DPO-P 训练代码**：一位成员询问是否有人拥有 **DPO-P 训练**的代码，但未进一步阐述 DPO-P 的含义或该代码的应用。 
- **自我意识时刻**：在一次有趣的自我意识转变中，一位成员在最初呼吁管理后意识到了自己的 Mod 身份。
- **招募 ML 项目合作志愿者**：向**程序员、数据分析师、AI 和 ML 专家**发出了号召，邀请他们自愿参加一个旨在建议研究课题的 Machine Learning 项目，激励措施包括证书、推荐信以及自由使用生成的代码。感兴趣的人受邀直接向发起人发送私信，并附上“interested”字样。

**提及的链接**：<a href="https://docs.google.com/document/d/1f-CHZudw3ZOGFIk-Kov3QHkPjjR-Sh4mMmxcExgnWUk/edit?usp=sharing">Post-AGI Educational Reforms </a>：未找到描述

---

**Alignment Lab AI ▷ #[looking-for-workers](https://discord.com/channels/1087862276448595968/1142242166677192774/1221887773313863742)** (1 messages):

- **非营利组织为突破性的 ML 项目寻求技术人才**："Youth Inquiry Network" 和 "Futracode" 正在合作开发一种 Machine Learning 算法，通过在研究数据库上进行训练来推荐最佳研究课题。他们正在寻找 Web 开发人员、数据分析师、AI & ML 专家加入这一事业。

- **具有实质性福利的志愿工作**：该项目是一项志愿活动，旨在帮助难以找到研究课题的学生。贡献者将获得作品集提升、证书以及来自这两家非营利组织创始人的推荐信。

- **为公益事业贡献代码**：志愿者将保留代码源码，用于个人成长、经验提升，甚至在项目完成后可以自由出售其贡献。工作时间灵活，可根据贡献者的空闲时间量身定制。

- **无附加条件的申请流程**：感兴趣的个人可以通过直接私信或评论 "interested" 加入项目，无需正式的申请表。
  

---



**Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1220948701175091200)** (5 messages): 

- **澄清 `llm` 和 `ollama` 的区别**：一位成员解释说，**`llm` 负责与模型交互**，但并不像 `ollama` 那样执行模型。可以配置 `llm` 来使用由 `ollama` 提供的 API 端点，后者在本地执行模型并将其作为本地 HTTP API 端点提供。
- **理解 Mistral 模型执行**：在询问有关 **Mistral 模型执行**时，一位成员得到了澄清：使用 `llm` 运行 Mistral 模型意味着它在本地运行，但是通过 **`ollama` 提供的 HTTP API 端点**，或者通过可以在没有 HTTP 的情况下运行本地模型的 `llm-llama-cpp` 插件。
- **对 AI 驱动的 Git Commit 助手的赞赏**：一位成员分享了他们持续使用 **[AICommits (GitHub - Nutlope/aicommits)](https://github.com/Nutlope/aicommits)** 的经历，这是一个利用 AI 辅助编写 git commit 消息的工具，同时表达了对 commit 表情符号标准等功能的期待。

**Link mentioned**: <a href="https://github.com/Nutlope/aicommits">GitHub - Nutlope/aicommits: A CLI that writes your git commit messages for you with AI</a>: 一个为你自动编写 git commit 消息的 CLI - Nutlope/aicommits

  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1221326221430882404)** (2 messages): 

- **AI 涉足烹饪**：一位成员分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=Nc5Yk0XXgP8)，标题为 **"Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!"**。视频讨论了 AI 如何根据 YouTuber Mr. Beast 的特技表演创作出一本食谱。
- **寻求德语 DL/AI 内容**：一位成员向小组征求 **德语 Deep Learning/AI 播客或视频系列** 的推荐。他们表达了对参与该语言内容的兴趣。

**Link mentioned**: <a href="https://www.youtube.com/watch?v=Nc5Yk0XXgP8">Mr. Beast Meets Mistral: AI Created a Cookbook Based on His Wildest Stunts!</a>: 今天我们制作了 Beast 食谱。"Beast Cookbook" 的想法是一种有趣且富有创意的方式，可以与 Mr. Beast 的内容互动，并生成一个有趣的、虚构的...