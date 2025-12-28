---
companies:
- ai21-labs
- databricks
- together-ai
- hugging-face
- midjourney
date: '2024-03-28T23:43:23.713718Z'
description: '**AI21 Labs** 发布了 **Jamba**，这是一款拥有 **520 亿参数的 MoE（混合专家）模型**，具备 **256K
  上下文长度**，并在 Apache 2.0 协议下开放权重，且针对单块 A100 GPU 的性能进行了优化。它采用了独特的“块与层”（blocks-and-layers）架构，结合了
  Transformer 和 MoE 层，旨在与 **Mixtral** 等模型竞争。


  与此同时，**Databricks** 推出了 **DBRX**，这是一个在 **12 万亿 token** 上训练、拥有 **360 亿激活参数的 MoE 模型**，被誉为开源大语言模型（LLM）的新标杆。


  在图像生成领域，相关进展包括可实现视频级质量图像生成的 **Animatediff**，以及能在 CPU 上实现极速图像生成的 **FastSD CPU v1.0.0
  beta 28**。其他创新还包括利用 **B-LoRA** 实现风格与内容的分离，以及通过 **SUPIR** 提升高分辨率图像的放大效果。'
id: 96bbde86-0900-4165-ba3b-63492f334b76
models:
- jamba
- dbrx
- mixtral
- animatediff
- fastsd
- sdxs512-0.9
- b-lora
- supir
original_slug: ainews-jamba-mixture-of-architectures-dethrones
people: []
title: Jamba：混合架构超越 Mixtral
topics:
- mixture-of-experts
- model-architecture
- context-windows
- model-optimization
- fine-tuning
- image-generation
- video-generation
- cpu-optimization
- style-content-separation
- high-resolution-upscaling
---

 

他们发布了一个基础模型，并且已经支持 Huggingface PEFT。**这看起来确实是一个真正的 Mixtral 竞争对手**，这对开放 AI 社区来说绝对是件好事。

---

**目录**

[TOC] 


---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。评论抓取尚未实现，但即将推出。

大型语言模型 (LLM)

- **/r/MachineLearning**: [DBRX: A New Standard for Open LLM](https://www.reddit.com/r/MachineLearning/comments/1bp213q/n_introducing_dbrx_a_new_standard_for_open_llm/) - **16 个专家，每个专家 12B 参数，36B 激活参数，基于 12T token 训练**
- **/r/LocalLLaMA**: [Databricks reveals DBRX, the best open source language model](https://www.reddit.com/r/LocalLLaMA/comments/1bp0glv/databricks_reveals_dbrx_the_best_open_source/) - 超越 grok-1, mixtral 以及其他开放权重模型
- **/r/LocalLLaMA**: [RAG benchmark of databricks/dbrx](https://www.reddit.com/r/LocalLLaMA/comments/1bpo5uo/rag_benchmark_of_databricksdbrx/) - DBRX 在实际测试中的 RAG 表现不佳，与 gemini-pro 相当

Stable Diffusion 与图像生成 

- [Animatediff is reaching a whole new level of quality](https://v.redd.it/3acyn8ikmzqc1) - @midjourney_man 使用 img2vid 工作流的示例
- **/r/StableDiffusion**: [Attention Couple for Forge](https://www.reddit.com/r/StableDiffusion/comments/1bpn9ps/attention_couple_for_forge/) - 轻松生成多个主体，不再有颜色渗漏或特征混杂
- [FastSD CPU v1.0.0 beta 28 release](https://i.redd.it/qoucghji2wqc1.png) - 在 CPU 上使用 SDXS512-0.9 OpenVINO 实现极速图像生成（0.82 秒）
- **/r/StableDiffusion**: [Implicit Style-Content Separation using B-LoRA](https://www.reddit.com/r/StableDiffusion/comments/1boyc47/implicit_stylecontent_separation_using_blora/) - 利用 LoRA 隐式分离单张图像的风格和内容组件
- **/r/StableDiffusion**: [SUPIR is exceptional even with high-res source images](https://www.reddit.com/r/StableDiffusion/comments/1bp7aij/supir_is_exceptional_even_with_highres_source/) - SUPIR 在放大高分辨率图像时能添加令人惊叹的细节

AI 助手与 Agent

- **/r/OpenAI**: [AI coding changed my life. Need advice going forward.](https://www.reddit.com/r/OpenAI/comments/1bp00kq/ai_coding_changed_my_life_need_advice_going/) - 使用 ChatGPT 学习 Web 开发并在朝九晚五的工作之外赚钱
- [Will ChatGPT eventually "learn" from it's own content it previously created, which could lead to it being wrong about facts sometime in the future?](https://ww**/r/OpenAI**: w.reddit.com/r/OpenAI/comments/1bphvpb/will_chatgpt_eventually_learn_from_its_own/) - 担心 ChatGPT 在其自身输出上进行训练会导致准确性下降
- **/r/LocalLLaMA**: [Created an AI Agent which "Creates Linear Issues using TODOs in my last Code Commit" . Got it to 90% accuracy.](https://www.reddit.com/r/LocalLLaMA/comments/1bp1jry/created_an_ai_agent_which_creates_linear_issues/) - 将 Autogen 与 GitHub 和 Linear 连接，自动从代码 TODO 中创建 issue
- **/r/OpenAI**: [Built an AI Agent which "Creates Linear Issues using TODOs in my last Code Commit".](https://www.reddit.com/r/OpenAI/comments/1bp7ioe/built_an_ai_agent_which_creates_linear_issues/) - Agent 利用代码上下文理解 TODO，分配给正确的人员/团队/项目，并在 Linear 中创建 issue

AI 硬件与性能

- **/r/MachineLearning**: [Are data structures and leetcode needed for Machine Learning Researcher/Engineer jobs and interviews?](https://www.reddit.com/r/MachineLearning/comments/1bpgdwn/d_are_data_structures_and_leetcode_needed_for/) 
- [Microsoft plans to offload some of Windows Copilot's features to local hardware, but will use NPUs only.](https://www.tomshardware.com/pc-components/cpus/intel-confirms-microsoft-copilot-will-soon-run-locally-on-pcs-next-gen-ai-pcs-require-40-tops-of-npu-performance) - 微软计划将 Windows Copilot 的部分功能转移到本地硬件，但将仅使用 NPU
- **/r/LocalLLaMA**: [With limited budget, is it worthy to go into AMD GPU/ecosystem now, given Tiny Corp released the tinybox with AMD and Lisa Su's recent speech at the AI PC summit at Beijing?](https://www.reddit.com/r/LocalLLaMA/comments/1boyais/with_limited_budget_is_it_worthy_to_go_into_amd/) - 在预算有限的情况下，考虑到 Tiny Corp 发布了搭载 AMD 的 tinybox 以及 Lisa Su 最近在北京 AI PC 峰会上的演讲，现在进入 AMD GPU/生态系统值得吗？
- **/r/LocalLLaMA**: [Looks like DBRX works on Apple Silicon MacBooks!](https://www.reddit.com/r/LocalLLaMA/comments/1bpn3nw/looks_like_dbrx_works_on_apple_silicon_macbooks/) - 在 M3 96GB 上，4-bit 量化占用约 66GB RAM，速度约为每秒 6 个 token

迷因与幽默

- [Me and the current state of AI](https://v.redd.it/y9ud1dnl7uqc1) - 我与 AI 的现状
- **/r/OpenAI**: [When 'Open'AI's lawyers ask me if used their models' generated output to train a competing model:](https://www.reddit.com/r/OpenAI/comments/1bp0ilx/comparative_claims_should_provide_some_evidence/) [已删除]
- **/r/LocalLLaMA**: [Open AI 3 Laws of Robotics](https://www.reddit.com/r/LocalLLaMA/comments/1bpjuc9/open_ai_3_laws_of_robotics/) - OpenAI 机器人三定律
- [When you are the 60 Billion $$ Man but also a Doctor.](https://www.reddit.com/gallery/1boy89r) - 当你是一个身价 600 亿美元的人，同时也是一名医生

# AI Twitter Recap

> 所有总结均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在尝试使用 Haiku 进行聚类和流程工程（flow engineering）。

待完成

---

# PART 0: 总结之总结之总结

- **Databricks 发布 DBRX**：Databricks 推出了 **[DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)**，这是一个拥有 **1320 亿参数**的 MoE LLM，并引发了关于预训练极限的辩论。AI 社区对其 12 万亿 token 的训练规模和潜力议论纷纷，将其与 Mistral Medium 等模型进行对比，并评估规模效应递减的情况。

- **Jamba 与 Qwen 模型的创新融合**：**[AI21 Labs](https://www.ai21.com/blog/announcing-jamba)** 推出了 **Jamba**，这是一种具有 256K 上下文窗口的 SSM-Transformer 混合模型；同时 **Qwen** 发布了 **[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)**，其性能超越了自身的参数量级，达到了 7B 模型的水平。这些发布引发了关于可访问性、性能以及 AI 扩展未来轨迹的讨论。

- **AI 社区探索 Vtuber 和 AI 控制台**：关于 **AI Tokyo** 活动以及为 Vtuber 建立人机协作模型以提升参与度的讨论非常热烈。继 **[AI Tokyo](https://discord.com/channels/1053877538025386074/1109649177689980928/1222510278953009152)** 上关于其前代产品中等配置的热议后，关于 **Truffle-2**（一个潜在的以 AI 为中心的控制台）的猜测也随之而起。

- **GPU 与 Token：性能追求**：工程师们分享了关于使用 **TensorRT** 进行大模型高效推理的见解，辩论了 Tensor Parallelism 的极限，并揭示了诸如 **smoothquant+** 和 **fp8 e5m2 cache** 等新方法。交流还集中在 Claude 的区域访问权限以及在训练 **Deepseek-coder-33B** 模型时微调损失曲线的困惑。

- **RAG、检索与数据集探讨**：AI 社区深入研究了 **检索增强生成 (RAG)** 的性能，辩论了输出质量，并审查了 CoT 对检索有效性的影响。提出了在 Prompt 输入中实现 XML 标签标准化的建议，将结构化输入视为增强结果的潜在标准。

---



# PART 1: 高层级 Discord 总结




## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **“Loaded from Chat UI” 异常已修复**：**LM Studio 0.2.18** 更新解决了一个 Bug，即 API 查询会返回模型 ID 为 “Loaded from Chat UI”，从而导致无法获取真实模型名称的问题，该问题已在测试版 [version 0.2.18](https://lmstudio.ai) 中修复。
  
- **通过模型合并实现规模扩展**：*ChuckMcSneed* 宣布，**LongAlpaca-70B-lora** 与 **lzlv_70b_fp16_hf** 的合并产生了一个 32K token 的线性 ROPE 缩放模型，尽管在 8 倍上下文长度下性能下降了 30%；查看合并后的 [模型地址](https://huggingface.co/grimulkan/lzlv-longLORA-70b-rope8-32k-fp16)。

- **LM Studio 爱好者的前沿配置**：LM Studio 0.2.18 通过为 Base/Completion 模型提供“空白预设（Empty Preset）”以及“等宽（monospace）”聊天样式等功能丰富了用户体验，根据 [公告](https://lmstudio.ai)，更新问题已得到解决。

- **为巅峰 AI 性能提升动力**：关于 AI 工作硬件的讨论建议，**NVIDIA 3090** 和 **4090** 或双 **A6000** 显卡可提供显著的 VRAM 和 CUDA 实力，显示器质量也是热门话题，例如 [这款 MSI 显示器](https://us-store.msi.com/MPG-321-URX-QD-OLED)。对于这些高性能配置，建议使用 **1200-1500w 的 PSU**。

- **使用 ROCm Beta 进行渲染，用户应对挑战**：**LM Studio 的 ROCm 0.2.18 Beta** 旨在解决 GPU offload 问题，但用户报告在模型加载和 GPU 利用率方面结果不一。感兴趣的各方可以探索 [ROCm beta 下载地址](https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe)，并在社区内寻求帮助以解决细微问题，或在需要时回退到标准版本。



---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

**破冰 Unsloth AI**：工程师们开始采用 **Unsloth** 模板系统的技巧和窍门，社区发现其在减少模型输出异常等方面具有实际效益。每周 2-3 次的定期更新确保了性能的持续提升，同时安装指南优化了在 **Kaggle** 上的设置时间。

**编码丛林中的游戏交融**：技术交流伴随着轻松的话题，包括游戏开发者对话和游戏经验分享——特别是利用 AI 辅助构建 Demo 应用，将娱乐与 Machine Learning 联系起来。

**层层深入**：
**Unsloth AI** 的讨论已延伸至更深层次的探索，包括在从 Checkpoint 恢复 Fine-tuning 时利用 Optimizer 调整，以及为各种 LLM 进行正确的 Chat Template 集成。社区还重点介绍了用于 Fine-tuning LLM 的关键资源——GitHub 仓库、Colab Notebooks 和教育类 YouTube 视频。

**模型展示亮点**：
社区自豪地展示了各种适配成果，例如为 **Tinyllama** 转换 **Lora Adapter**，并分享了使用 **Unsloth** 方法论进行 Fine-tuning 的 **Mischat** 模型的细节。一位成员在他们的 **Substack** 博客上介绍了 AI 摘要，总结了最近的 AI 进展。

**量化领域的量子飞跃**：AI 爱好者研究了专门的技术，如 **LoRA** 训练对话、用于快速检索的 Embedding 量化，以及新兴的 **QMoE** 压缩框架。新引入的 **LISA** 策略因其内存效率而备受关注，该策略简化了跨层的 Fine-tuning 流程。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **DBRX 以惊人的规模吸引关注**：Databricks 发布了 **[DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)**，这是一个拥有 **1320 亿参数** 的 MoE LLM，并引发了关于预训练极限的辩论。AI 社区对其 12 万亿 Token 的训练量和潜力议论纷纷，将其与 Mistral Medium 等模型进行对比，并评估规模效应的收益递减。

- **Jamba 和 Qwen 模型的创新融合**：**[AI21 Labs](https://www.ai21.com/blog/announcing-jamba)** 推出了 **Jamba**，这是一种具有 256K 上下文窗口的 SSM-Transformer 混合架构；而 **Qwen** 发布了 **[Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)**，其性能堪比 7B 模型，超越了自身的参数量级。这些发布引发了关于可访问性、性能和 AI Scaling 未来轨迹的讨论。

- **AI 社区探索 Vtuber 和 AI 游戏机**：关于 **AI Tokyo** 活动以及为 Vtuber 建立人机协作模型以提高参与度的讨论非常热烈。在 **[AI Tokyo](https://discord.com/channels/1053877538025386074/1109649177689980928/1222510278953009152)** 上关于前代产品中等配置的热议之后，有关 **Truffle-2**（一款潜在的以 AI 为中心的游戏机）的猜测也随之而起。

- **GPU 与 Token：性能追求**：工程师们分享了关于 **TensorRT** 在大型模型上进行高效推理的见解，辩论了 Tensor Parallelism 的极限，并揭示了诸如 **smoothquant+** 和 **fp8 e5m2 cache** 等新方法。交流还集中在 Claude 的区域访问以及训练 **Deepseek-coder-33B** 模型时 Fine-tuning Loss 曲线的困惑。

- **RAG、检索与数据集讨论**：AI 社区深入研究了 **Retrieval Augmented Generation (RAG)** 的性能，辩论了输出质量，并审查了 CoT 对检索有效性的影响。有人提议在 Prompt 输入中实现 XML 标签标准化，将结构化输入视为增强结果的潜在标准手段。

---

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3 发布在即**：工程社区正因 **Stable Diffusion 3 (SD3)** 预计在 4 月底或 5 月的推出而议论纷纷，该版本将具备包括 inpainting 在内的增强功能。根据 Stability.ai CTO 的言论推断，从 3 月 25 日起约 4 到 6 周的 ETA 引发了对新模型和功能的广泛猜测。

- **评估语言模型的 VRAM 需求**：关于运行 Mixtral 等语言模型的 VRAM 需求的讨论正在升温，争论焦点在于使用 **quantization**（量化）策略在不牺牲质量的前提下减少内存占用。工程师们尤其关注为 **10GB Nvidia GPU 显卡**量身定制的量化模型，这表明了对更易获得的高性能计算的追求。

- **新用户指南与工具建议**：Discord 空间不仅面向资深专家；新用户也正在获得通过 **Stable Diffusion** 生成图像的技巧，推荐的界面包括 **Forge** 和 **Automatic1111**，以及用于增强创作过程的 **leonardo.ai**。

- **优化图像 Prompt 质量**：一个技术线程强调了 prompt engineering 的重要性，强调更具对话性的句子结构比逗号分隔的关键词能产生更好的效果。这在处理像 **SDXL** 这样对 prompt 措辞细微差别敏感的高级模型时尤为重要。

- **讨论模型量化效率**：公会成员简要讨论了 **transformer 架构的效率**和 quantization 的有效性。这些 AI 鉴赏家认为，尽管 transformer 存在固有的低效性，但像 SD3 这样的模型在量化后表现出了令人期待的结果，可能允许更小的内存占用。

讨论中的链接包括资源和工具：

- [Stable Diffusion 中的角色一致性](https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/)，作者 Cobalt Explorer
- [leonardo.ai](https://leonardo.ai/)
- [用于视频广告的 Arcads](https://arcads.ai)
- [Pixel Art Sprite Diffusion Safetensors](https://civitai.com/models/129057/pixel-art-sprite-diffusion-safetensors)
- [适用于 AMD GPU 的 Stable Diffusion web UI](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux)
- [GitHub - lllyasviel/Fooocus](https://github.com/lllyasviel/Fooocus)
- [Google Colab 免费层上的 StableDiffusionColabs](https://github.com/Vargol/StableDiffusionColabs)
- [在 8Gb M1 Mac 上运行 Stable Diffusion](https://github.com/Vargol/8GB_M1_Diffusers_Scripts)

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

**DBRX 进驻 Perplexity Labs**：Databricks 的 **DBRX** 语言模型因超越 GPT-3.5 并证明其与 Gemini 1.0 Pro 具有竞争力而引起轰动，在数学和代码基准测试中表现优异，可在 [Perplexity Labs](https://labs.pplx.ai) 进行体验。

**开发者的抉择：Perplexity vs. Claude**：工程师们讨论了 Perplexity Pro 或 Claude Pro 哪个更适合他们的工作流，由于透明度原因，他们更倾向于 Perplexity。Claude 3 Opus 等各种模型的优势受到了审视，而 Databricks 的 DBRX 因其出色的数学和编程能力被特别提及。

**Perplexity API 速度飙升**：`sonar-medium-online` 模型展现了意料之外的速度提升，在输出质量更高的同时，达到甚至超过了 `sonar-small-online` 的速度。然而，API 响应与 Perplexity 网页界面相比出现了一些不一致，例如无法检索 "Olivia Schough spouse" 的数据，引发了关于额外参数是否能纠正此问题的讨论。

**分享见解与趣闻**：社区互动包括揭穿一个所谓的 Sora 文本转视频模型其实是 rickroll，强调了 thread 可分享性的重要性，并探索了 Perplexity AI 上的各种搜索查询，从连贯的 C3 模型到 "Perplexityai" 的法语翻译。

**Vision 支持仍无音讯**：尽管有人询问，但 API 的 Vision 支持仍然缺席，正如关于目前甚至缺乏 citations（引用）的幽默回复所暗示的那样，这表明目前没有立即加入该功能的计划。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

**Claude 摘得 Terraform 桂冠**：在 IaC 领域，**Claude** 在生成 **Terraform** 脚本方面表现优于同行，TerraTeam 网站上的一篇对比博客文章强调了其卓越性能。详细的对比可以在 [TerraTeam 的博客](https://terrateam.io/blog/using-llms-to-generate-terraform-code)中查看。

**DBRX-Instruct 展示其参数实力**：**Databricks** 凭借 **DBRX-Instruct** 成为焦点，这是一个 **1320 亿参数的 Mixture of Experts 模型**，在 **3072 块 NVIDIA H100 GPU** 上经历了耗资巨大（1000 万美元）且耗时较长（2 个月）的训练。关于 DBRX-Instruct 的见解分布在 [Vitaliy Chiley 的推文](https://x.com/vitaliychiley/status/1772958872891752868?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)和 [Wired 的文章](https://www.wired.com/story/dbrx-inside-the-creation-of-the-worlds-most-powerful-open-source-ai-model/)中。

**DBRX 的许可物流问题依然存在**：社区仔细审查了 **DBRX 的许可条款**，成员们正在策划如何在其实际使用界限内最好地利用该模型。关键见解来自共享的法律疑虑和策略，包括 **Amgadoz** 对 [Databricks 开放模型许可证](https://www.databricks.com/legal/open-model-license)的关注。

**TechCrunch 质疑 DBRX 的市场实力**：**TechCrunch 对 Databricks 1000 万美元的 DBRX 投资进行的批判性分析**引发了讨论，并将其与已经确立地位的 **OpenAI GPT 系列**进行了对比。**TechCrunch** 挑战了此类投资所能提供的竞争优势，建议阅读 [TechCrunch](https://techcrunch.com/2024/03/27/databricks-spent-10m-on-a-generative-ai-model-that-still-cant-beat-gpt-4/) 的全文。

**情感智能聊天机器人获得好评**：**Hume AI** 凭借其情感感知聊天机器人引起了关注，该机器人擅长分析和响应情感。这种颠覆性的情感检测能力在成员中引发了兴奋和实际用例的讨论，包括 **420gunna** 分享的 [Hume AI 演示](https://demo.hume.ai/)和相关的 [CEO 访谈](https://www.youtube.com/watch?v=3C101739_hI)。

**Mamba 游入聚光灯下**：在讨论中，**[Mamba 模型](https://arxiv.org/abs/2312.00752)**因其在 Transformer 领域的创新而脱颖而出，有效地解决了效率问题。强有力的对话围绕着 Mamba 的实力和旨在提高计算效率的架构决策展开。

**微调技巧**：关于微调 OpenAI 的自动语音识别模型 **[Whisper](https://openai.com/blog/whisper/)** 的话题被深入剖析，共识是当处理稀缺语言资源或音频中的专业术语时，微调是值得推荐的。

**余弦相似度杂谈**：小组就 Embedding 中 **余弦相似度** 的使用进行了技术交流，对其作为语义相似度度量的有效性表示怀疑。讨论的焦点是题为“[Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440)”的论文，成员们将其作为参考点。

**屏幕共享故障**：Discord 屏幕共享的技术试验引发了社区的故障排除，包括分享变通方案以及集体呼吁 Discord 增强此功能。成员们分享了解决持续存在的屏幕共享问题的实用方案。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **Claude 3 意识到评估的存在**：Anthropic 的 Claude 3 在测试期间展示了元认知（meta-awareness），能够识别自己何时正在接受评估，并对处理信息的针对性发表评论。
  
- **DBRX 的大手笔**：Databricks 推出了 **DBRX**，这是一个强大的语言模型，拥有 1320 亿总参数和 360 亿激活参数，在 12 万亿 Token 的语料库上训练而成。讨论集中在其架构上，包括 16 个 Expert 和 32k 上下文长度，以及它的对比性能和可用性，因其在性能上超越了 Grok 等模型而引起轰动。

- **Token 效率辩论**：工程师们正在辩论大型 Tokenizer 的实际效率，认为更大的 Token 数量可能不会自动转化为性能提升，并可能导致特定的 Token 表示问题。
  
- **层剪枝显示影响极小**：研究发现，使用 QLoRA 等方法在 LLM 中减少高达 50% 的层数，性能损失极小，从而能够在单个 A100 GPU 上进行微调。

- **Jamba 加速模型融合**：AI21 Labs 发布了一个名为 **Jamba** 的新模型，将结构化状态空间模型（Structured State Space models）与 Transformer 相结合，拥有 120 亿激活参数和显著的 256k 上下文长度。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **GPT-4：可能性的灯塔还是仅仅是推文诱惑？**：用户对 [OpenAI 的一条推文](https://twitter.com/OpenAI/status/1773032605002203559?t=jZBiDy4Xzymzfy7n14RGzQ&s=19) 反应热烈，既有热情也有期待，该推文暗示了新的进展，尽管也有人对 **GPT-4** 等服务在欧洲延迟可用表示担忧。
- **ChatGPT 用于代码**：分享的技巧包括指示 ChatGPT **避免省略号**和不完整的代码段，这有助于在编程相关任务中获得更可靠的输出。对比评价认为 **Claude 3** 在编程效率方面优于其他模型。
- **众目睽睽下的 Gemini Advanced**：社区对 **Google 的 Gemini Advanced** 持保留态度，抱怨其响应速度与 GPT-4 相比显得迟缓，尽管人们对基于即将进行的压力测试的未来改进抱有期待。
- **AI 的工业进军**：值得注意的是 OpenAI 和 Microsoft 将其 AI 产品整合到欧洲工业中的策略，可能涉及 **Copilot Studio** 和更广泛的 Microsoft 套件等工具，尽管一些用户对 Copilot 的 UX 表示不满。
- **Prompt Engineering 心得**：AI 爱好者讨论了使用 LLM 时获得最佳结果的各种策略，包括将 Prompt 拆分为块以便更好地识别问题，编写强调**做什么**而非不做什么的 Prompt，以及在保持 HTML 完整性的同时，明确表达对视觉描述或翻译等任务中特定输出的需求。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**Stable Diffusion 在单体表现上更进一步**：关于 **[Stable Diffusion](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/image_variation)** 的讨论集中在从列表生成新图像，但现有的 Pipeline 处理的是单张图像。对于个性化的 Text-to-Image 模型，**[DreamBooth](https://dreambooth.github.io/)** 成为首选，而 **Marigold 深度估计 Pipeline** 正准备与 **LCM** 等新模态集成。

**AI 工程师寻求更智能的 NLP 导航**：工程师们在寻求 2024 年掌握 NLP 的路线图，推荐包括《深度学习简明指南》（The Little Book of Deep Learning）和 Karpathy 的 *"Zero to Hero"* 播放列表。其他人探讨了基于会话的推荐系统，质疑 **GRU4Rec** 和 **Bert4Rec** 等模型的有效性，而 **'facebook/bart-large-cnn'** 的加载错误引发了求助。管理 LLM 无限生成行为的建议包括 **Supervised Fine Tuning (SFT)** 和调整重复惩罚（repetition penalties）。

**通过 MPS 和 Sagemaker 加速 GPU 收益**：macOS 用户获得了优势，**MPS** 支持现已包含在 [关键训练脚本](https://github.com/huggingface/diffusers/pull/7447) 中，而关于 **AWS SageMaker** 的讨论强调了使用 **NVIDIA Triton** 和 **TensorRT-LLM** 来基准测试利用 GPU 的模型的延迟、成本和吞吐量。

**Computer Vision 领域的创新与资源**：在尝试利用拼接图像训练模型的同时，个人还在努力在特定数据集上微调 **DETR-ResNet-50**，并为初学者研究 Zero-shot 分类器微调。此外，还有人求助非 **gradio_client** 的测试方法来演示 instruct pix2pix，社区积极推荐替代方案和资源。

**备受关注的 DL 模型**：NLP 社区正在研究关于个性化 Text-to-Image 合成以使其紧密符合文本 Prompt 的论文。**[RealCustom](https://arxiv.org/abs/2403.00483)** 论文讨论了在主体相似度与文本控制之间取得平衡，另一项研究则解决了个性化图像中的文本对齐问题，如 **[arXiv](https://arxiv.org/abs/2401.06105)** 所述。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **工程师寻求欧盟分销路径**：有成员表示需要关于在欧盟境内分销产品的协助或讨论，暗示了对产品分销物流策略的需求。
- **探索在 IDE 中使用 OpenInterpreter**：成员们正在讨论并分享将 OpenInterpreter 与 Visual Studio Code 等 IDE 集成的资源，包括推荐一个[用于 AI 工具的 VS Code 扩展](https://github.com/MikeBirdTech/gpt3-vscode-extension)。
- **准备，开始，优化！**：社区正致力于探索和优化本地及托管 LLM 的性能。预计到今年年底，这些模型的能力甚至可能超越 GPT-4。
- **成员通过“先前技术”重新定义“完成”**：一位成员分享了一个幽默的发现：花费数小时工作后，无意中重复了已有的功能，并附上了一个展示其过程的 [YouTube 视频](https://www.youtube.com/watch?v=UqjMf5Fb4cg)。
- **本地 LLM 引起关注**：关于在 OpenInterpreter 中实现非 GPT 模型的对话非常活跃，大家对实验本地 LLM 充满好奇，并询问了关于 groq 等其他模型的信息，暗示了在 OpenAI 工具之外的广泛探索。

---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

**解决 VSCode 调试中的 Bug**：GitHub 上报告的一个关于 Mojo 插件的 [VSCode 调试问题](https://github.com/modularml/mojo/issues/1924#issuecomment-2018212062) 已通过推荐的变通方法解决，该方法在 MacBook 上运行成功。

**Mojo 和 MAX 更新成为头条**：Mojo 语言风格指南现已[发布](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md)，同时 GitHub 上也出现了一个新的复数库 [moplex](https://github.com/helehex/moplex)。MAX 24.2 更新包括采用 `List` 替代 `DynamicVector`，详见[更新日志](https://docs.modular.com/mojo/changelog#v242-2024-03-28)。

**优质学习资源**：推荐阅读 *Rust for Rustaceans* 中的[免费章节](https://nostarch.com/rust-rustaceans)以理解 Rust 的生命周期管理；同时 Modular 最新的推文也[引起了关注](https://twitter.com/Modular/status/1773024465401852107)，但未引发进一步讨论。

**拥抱开源提升 Mojo 的模块化**：Modular 已在 Apache 2 协议下开源了 [Mojo 标准库](https://modul.ar/open-source-blog)，并提供 Nightly 版本；MAX 24.2 引入了对动态输入形状的改进支持，如其[博客](https://modul.ar/max-dynamic-shapes)所示。

**讨论 API 差异和增强功能**：用户讨论了 Mojo 和 Python API 在 `TensorSpec` 方面的不一致性，并引导他人参考 MAX Engine 运行时[文档](https://docs.modular.com/engine/mojo/get-started#define-input-specs-torchscript-only)和 MAX 的[示例仓库](https://github.com/modularml/max)以获取清晰说明。

**开源和 Nightly 版本邀请协作**：开发者受邀加入 Modular 开源倡议，包括 Mojo [标准库更新](https://modul.ar/open-source-blog)和[最新更新日志](https://modul.ar/mojo-changelog)中列出的新功能；同时 MAX 平台 v24.2 的演进提供了新能力，特别是在动态形状方面。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**为 cheerful_dragon_48465 喝彩**：用户名 **cheerful_dragon_48465** 因其有趣而受到称赞，**Alex Atallah** 预告即将发布一项**公告**，重点介绍一位用户的显著贡献。

**Midnight Rose 亟需明确说明**：**Midnight Rose** 模型在没有错误提示的情况下无响应，在 OpenRouter 团队解决问题之前引起了用户困惑，但根本问题仍未彻底解决。

**Token 数量的大小至关重要**：用户讨论了 **Gemini 模型**上下文窗口大小的差异，这些模型是以字符而非 Token 计量的，这引起了混淆，并承认需要对该主题进行更好的澄清。

**Gemini Pro 1.5 的测试问题**：遇到 **Gemini Pro 1.5** `Error 503` 的用户被告知，这些问题是因为该模型仍处于测试阶段，这表明 OpenRouter 的服务预期与现实之间存在差距。

**以太坊支付难题**：OpenRouter 转向要求通过 Coinbase Commerce 在 **ETH 网络进行支付**，以及随后关于美国银行转账激励措施的讨论，突显了 AI 领域加密货币支付方式的演变。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **深入探讨动态 CUDA 支持**：社区成员正在讨论在 OpenCV 的 DNN 模块中实现动态 CUDA 支持，并详细介绍了使用 NVIDIA GPU 的性能实验结果。分享了一份[关于深度学习 CUDA 硬件的调查](https://forms.gle/7kyMtMgYA2VA4mUN9)以收集社区经验，RTX 4090、A5000 和 A4000 GPU 的点对点（peer-to-peer）基准测试可通过 [GitHub](https://github.com/cuda-mode/p2p-perf/) 获取。

- **招募 Triton 导师**：为了准备一场演讲，正在寻求采访最近的 Triton 学习者，以了解他们面临的困难，可通过 Discord DM 或 [Twitter](https://x.com/UmerHAdil) 联系。在 [GitHub](https://github.com/pytorch-labs/ao/pull/95) 上可以找到协作工作和对 pull requests 提供意见的机会，包括 torch 生态系统中 GaLore 的原型，这表明了涉及 `bitsandbytes` ([PR #1137](https://github.com/TimDettmers/bitsandbytes/pull/1137)) 的活跃协作。

- **CUDA 资源与学习路径**：希望深化 CUDA 技能的热心人士分享了学习资源，包括 [CUDA 资料的 GitHub 仓库](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses)、“并行编程入门” [YouTube 播放列表](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)，以及一次因 Amazon 验证码而受阻的书籍讨论。

- **Torch 故障排除与类型纠缠**：工程师们正在处理 **torch** 和 **cuda** 之间的类型问题，强调了潜在的链接器（linker）问题，并寻求在 PyTorch 中对不兼容类型使用 `data_ptr` 方法时获得更清晰的编译时错误提示。

- **显微镜下的 Ring Attention**：AI 开发者深入研究了 **Ring Attention** 及其与其他注意力机制（如 **Blockwise Attention** 和 **Flash Attention**）的关系，一篇 [arXiv 论文](https://arxiv.org/abs/2305.19370)提供了更多见解。另外，针对训练中遇到的高 loss 值正在进行调试，这可能涉及序列长度处理，详见 [FSDP_QLoRA GitHub 仓库](https://github.com/AnswerDotAI/fsdp_qlora)及其 wandb 报告。

- **CUDA 疑难杂症**：从解决 **Triton** `tl.zeros` 在 kernel 中的用法，到处理 Triton-Viz 的 `ImportError` 并分享解决方法，参与者交流了修复方案，包括[从源码构建 Triton](https://github.com/openai/triton/issues/1693) 以及选择特定的 [triton-viz commit](https://github.com/Deep-Learning-Profiling-Tools/triton-viz@fb92a98952a1e8c0e6b18d19423471dcf76f4b36) 进行安装。还建议在 Triton 中避免使用 `reshape` 以获得更好的性能。

- **AI 成为喜剧短片的主角**：AI 行业对术语的偏爱在一段 [YouTube 视频](https://www.youtube.com/watch?v=XWqycBMUFA0)中被幽默地描绘出来，重点是 NVIDIA 主旨演讲中的 “AI”。此外，还有关于如何操作中文界面（如[知乎](https://www.zhihu.com/signin?next=%2F)）以获取 Triton 教程的求助请求。

- **Windows 和 WSL 上的 CUDA 热情**：用户分享了在 Windows 上配合 PyTorch 运行 CUDA 的成功经验并寻求指导，建议包括使用 [Microsoft 安装指南](https://learn.microsoft.com/en-us/windows/wsl/install)中概述的 WSL，而其他人则考虑安装 Ubuntu 双系统或记录他们的设置过程。

- **全球寻找精通 CUDA 的专家**：CUDA 领域的求职者正在寻找机会，提到 NVIDIA 发布了一系列[全球博士级职位](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)。一份声明强调，对于考虑来自任何地点的申请者的团队来说，人才胜过地理位置。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **RAG 优化揭秘**：**@seldo** 将于本周五深入探讨**高级 RAG 技术**，重点关注与 **TimescaleDB** 协同的优化——详情见 [Twitter](https://twitter.com/llama_index/status/1773065894756818961)。为减少 RAG 资源占用，**Cohere** 提议使用 **Int8** 和 **Binary Embeddings**；更多信息请参考 [Twitter](https://twitter.com/llama_index/status/1773402379016138955)。

- **法律领域的 LLM**：即将举行的**斯坦福大学 LLMxLaw Hackathon** 旨在探索 LLM 与法律领域的潜在协同作用，可通过 [Partiful](https://t.co/7lXZBX5APy) 报名。

- **使用 Llamaparse 处理杂乱数据**：正在处理来自 Confluence 的杂乱数据的用户可能会在 **Llamaparse** 中找到救星；[LlamaIndex 联系页面](https://www.llamaindex.ai/contact)强调了本地部署是一个选项。对于受困于 **PDF parsing** 挑战的用户，建议采用合并较小文本块并使用 **LlamaParse** 的策略。

- **Pipeline 与并行难题**：针对 **IngestionPipeline** 中文档 ID 保留的问题进行了澄清；原始文档的 ID 会保留为 `node.ref_doc_id`。同时，提高 Notebook 性能的建议包括使用 `aquery` 进行异步执行。

- **赋能 GenAI**：[Centre for GenAIOps](https://genaiops.ai/) 成立，这是一个旨在促进 GenAI 应用增长和安全的非营利组织。其创始 CTO 强烈推荐 **LlamaIndex**，并在 [LinkedIn](https://www.linkedin.com/company/the-centre-for-genaiops-cic/?viewAsMember=true) 上分享了见解。在教育方面，有人请求提供顶级的 LLM 培训资源，但尚未得到回应。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

**Databricks 发布 DBRX**：Databricks 推出了 **DBRX Base** 和 **[DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)**，拥有 132B 总参数量，表现优于 LLaMA2-70B 等模型，并提供[开源模型许可证](https://www.databricks.com/legal/open-model-license)，其[技术博客](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)提供了更多见解。

**Axolotl 开发者调试**：Axolotl AI Collective 修正了 **`trainer.py`** 的 Batch Size Bug，并讨论了 Transformer 不兼容、DeepSpeed 和 PyTorch 二进制问题，以及使用 `qlora+fsdp` 加载大模型的挑战。

**创新的 Jamba 和 LISA**：**AI21 Labs** 发布了 **Jamba**，这是一种能够在 [A100 80GB GPUs](https://www.ai21.com/blog/announcing-jamba) 上处理 256k Token 上下文的架构；同时，社区讨论了 **LISA** 在指令遵循任务中优于 LoRA 的表现，参考了 LMFlow 仓库中的 PR [#701](https://github.com/OptimalScale/LMFlow/pull/701/files) 和 [#711](https://github.com/OptimalScale/LMFlow/pull/711/files)。

**bf16 的性能表现**：围绕在训练和优化中使用 **bf16 精度**展开了激烈辩论，引用了 torchtune 团队关于内存效率和稳定性（类似于 fp32）的发现，引发了对其更广泛实现的兴趣。

**寻找微调技巧资源**：社区成员正在寻求微调或训练开源模型的综合教育材料，表示偏好博客、文章和视频等多种形式，旨在进入 axolotl 之前打下坚实基础。



---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **AI 思考存在**：在关于 AI 自我意识的讨论中，一位用户分享了与 ChatGPT 3.5 的两次互动，其中它表达了“悟”（satori）的时刻，引发了对其理解意识的疑问。可以通过以下链接探索这些交流：[Chat 1](https://chat.openai.com/share/47e3dfba-456f-4497-89cc-725ac2c326bc) 和 [Chat 2](https://chat.openai.com/share/5d5fd377-44a1-4e75-aa53-da70f12bd492)。
  
- **AI 崛起背景下配音演员的担忧**：针对 AI 进步对专业配音行业未来的影响，社区展开了激烈辩论，并提到了 Disney 通过与 [ElevenLabs](https://elevenlabs.io/blog/elevenlabs-joins-disneys-accelerator-program) 合作，对 AI 配音角色表现出的兴趣。

- **基准测试受到质疑**：AI 模型性能的基准测试因有时存在误导性的可视化而受到批评，人们呼吁建立更简洁、更符合人类感知的衡量标准，例如 [Chatbot-Arena Leaderboard](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 上的标准。

- **为 AI 模型瘦身**：一项关于 LLM 资源高效利用的研究表明，层剪枝（layer pruning）不会大幅影响性能，详情可参阅这篇 [arXiv 论文](https://arxiv.org/abs/2403.17887)。ProGamerGov 引入了用于 VLM 图像标注及故障检测的新工具，可在 [GitHub](https://github.com/ProGamerGov/VLM-Captioning-Tools) 上获取。

- **Devika 旨在简化软件工程**：一个名为 Devika 的创新项目旨在理解高级人类指令并编写代码，定位为类似 AI 的开源替代方案。Devika 的方法和特性可在其 [GitHub 页面](https://github.com/stitionai/devika)上查看。



---



## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

**Tinygrad 正在精益求精**：关于 **tinygrad** 的动态讨论揭示了其试图通过 `gemv` 和 `gemm` 等操作的启发式方法以及直接操作 GPU kernels 来缩小与 **PyTorch** 性能差距的尝试。见解包括 kernel fusion 挑战、潜在的 view merging 优化以及**[社区驱动的文档工作](https://github.com/mesozoic-egg/tinygrad-notes)**。

**NVIDIA 在 MLPerf 中夺冠**：最近的 **[MLPerf Inference v4.0 结果](https://mlcommons.org/2024/03/mlperf-inference-v4/)** 引发了讨论，指出 NVIDIA 如何继续在性能指标上保持领先，Qualcomm 表现强劲，而 Habana 的 **Gaudi2** 则显示出其并非为推理任务设计。

**SYCL 挑战 CUDA**：一条 **[推文](https://x.com/sasank51/status/1772993950451646920?s=46&t=DeaHUwU78T_AL16D-7B7Vg)** 强调 **SYCL** 是 NVIDIA CUDA 的有力替代者，激发了人们对更广泛行业采用以及打破当前 AI 硬件垄断趋势的期待。

**API 阵营与行业影响**：成员们就 **OpenCL** 利用率下降以及 **Vulkan** 在实现统一硬件加速接口方面的潜力发表了看法，辩论了它们在更大生态系统中的各自角色。

**View Merging 指日可待**：讨论还探讨了 tinygrad `ShapeTracker` 的改进，以潜在地合并 view，并在考虑结构变化时权衡了 Tensor 转换历史和反向传播功能的重要性。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

**OpenGPTs 讨论欢迎工程师**：[GitHub 上的 OpenGPTs 项目](https://github.com/langchain-ai/opengpts) 引入了一个新频道，鼓励社区间的贡献和对话。

**JavaScript 聊天机器人 vs 文档获取器**：AI 工程师们正在探索使用 **JavaScript 构建动态聊天机器人**，而不是静态文档检索。为了提供指导，分享了一个 [Colab notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb)。

**自定义域名部署的小插曲**：在 github.io 等自定义域名上使用 LangChain 部署 FastAPI RAG 应用引发了好奇；然而，LangChain Pinecone 集成的文档差异带来了挑战，尚待解决。

**LangSmith 追踪 AI 步骤**：使用 LangChain 的 LangSmith 追踪 AI 动作时，采用了 `LANGCHAIN_TRACING_V2` 等环境变量，提供了细粒度的日志记录能力。

**教程揭秘 PDF 转 JSON**：一个新的 [YouTube 教程](https://www.youtube.com/watch?v=ubsqSWfXAPI) 详细介绍了如何使用 **LangChain 的 Output Parsers** 和 GPT 将 PDF 转换为 JSON，简化了这一曾经复杂的任务。社区的见解被请求用于增强此类教育内容。



---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **DBRX 震撼 LLM 领域**：MosaicML 和 Databricks 推出了 **DBRX**，这是一个拥有 **1320 亿参数**的模型，具备 **320 亿激活参数**和 **32k 上下文窗口**，采用商业许可证发布，可在[此处](https://huggingface.co/databricks/dbrx-instruct)进行试用。然而，其许可条款禁止使用 DBRX 来改进其他 LLM，这引发了工程师们关于其对 AI 进步影响的讨论。

- **Jamba：AI21 实现 SSM 与 Transformers 的结合**：**AI21** 发布了 **Jamba**，将 Mamba 的结构化状态空间模型 (SSM) 与传统的 Transformer 架构相结合，并提供了 **256K 上下文窗口**。Jamba 以 Apache 2.0 许可证发布，旨在鼓励混合模型结构的发展，可通过[此处](https://huggingface.co/ai21labs/Jamba-v0.1)访问。

- **Mosaic 定律预示更廉价的 AI 未来**：“Mosaic 定律”已成为热门话题，该定律预测在硬件、软件和算法进步的推动下，同类模型的成本每年将下降四分之三，预示着未来 AI 的开发成本将大幅降低。

- **分析架构演进**：一项针对非 Transformer 架构的最大规模分析研究表明，**条纹架构 (striped architectures)** 可能通过层专业化优于同质架构，这可能预示着更快的架构改进。完整的研究报告和配套代码可在[此处](https://arxiv.org/abs/2403.17844)和[此处](https://github.com/athms/mad-lab)获取。

- **从“小”到“大”：语言模型频谱之争**：讨论指向了“小”语言模型的语义，社区在反思历史背景的同时，将 **1000 亿参数**以下的模型视为小型模型。此外，Microsoft GenAI 聘请 Liliang Ren 担任高级研究员，有望在高效且可扩展的神经架构方面取得进展；而 Megablocks 转向 Databricks 则突显了 AI 工程社区内项目管理权和预期的转变。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

**DBRX Instruct 隆重登场**：Databricks 推出了一款新型 1320 亿参数的稀疏 MoE 模型 **DBRX Instruct**，该模型在惊人的 12 万亿 token 上进行了训练，在少轮对话中表现出色。同时，Databricks 还以开放许可证发布了 DBRX Base，并在其[博客文章](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)中提供了深入见解。

**DBRX 内部机制解码**：DBRX 的独特之处在于其合并注意力机制、独特的归一化技术以及经过多次错误修复完善的独特分词 (tokenization) 方法，其技术细节已记录在 [GitHub](https://github.com/databricks/dbrx/blob/main/model/modeling_dbrx.py) 上。

**上手体验 DBRX Instruct**：AI 爱好者现在可以通过交互式 [Hugging Face space](https://huggingface.co/spaces/databricks/dbrx-instruct) 体验 DBRX Instruct，并配有用于定制回答风格的系统提示词 (system prompt)。

**免费体验 Mixtral 的多语言能力**：可以通过 [groq 免费使用](https://github.com/CrispStrobe/llm_translation.git) Mixtral 的翻译 API，受速率限制约束，并向社区驱动的实验开放。

**Occi 7B 在翻译质量上表现卓越**：用户注意到通过 `occiglot/occiglot-7b-de-en-instruct` 模型实现的 Occi 7B 具有极高的翻译保真度，并开始评估 DisCoLM、GPT-4、Deepl 和 Azure Translate 等服务的翻译水平，并在 [Hugging Face](https://huggingface.co/datasets/cstr/Capybara-de-snippets) 上展示了他们的成果。



---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **DBRX 占据榜首，超越 GPT-3.5**：由 Databricks 推出的 [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) 在 LLM 领域确立了领先地位，据称超越了 GPT-3.5，并可与 Gemini 1.0 Pro 媲美，其采用 MoE 架构，专注于编程任务以提高效率。
- **寻求 DBRX 的简化解释**：参与者呼吁对 DBRX 模型进行简明扼要的解释，以便更好地理解其在 LLM 效率和编程能力方面的进步。
- **DBRX 的编程实力受到质疑**：成员们探究 DBRX 出色编程能力的根源，思考这是特定数据集和架构的结果，还是源于更广泛的策略。
- **解码 DBRX 的编程优势**：DBRX 令人称赞的编程结果归功于其 12 万亿 token 的预训练、MoE 架构以及旨在避免 **"skill clobbering"**（技能冲突）的针对性课程学习。
- **单人编程难题**：一位同伴请求针对某个编程问题提供一对一支持，凸显了社区在提供个性化故障排除协助方面的作用。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **加入 LLM Brew Crew**：由 Exa.ai 主办的 **快闪咖啡馆与协作办公活动** 将于本周六在 SF 举行，提供免费咖啡、抹茶和糕点。感兴趣的人员可以 [在此预约 (RSVP)](https://partiful.com/e/yaC2YSd4kYN7YQF6WVFx)。
- **寻找以 AI 为中心的工作场所**：SF 的成员正在寻找迎合 LLM 爱好者的协作办公空间；**celo** 被提及为一个首选场所。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord

- **Python 爱好者准备贡献力量**：有人询问了即将为对 AI 领域感兴趣的 Python 爱好者举行的 **引导会议 (onboarding session)**；成员们正寻求参与并做出有效贡献。
- **闲聊视频分享**：一位成员在闲聊频道分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=LWz2QaSRl2Y)；然而，视频内容及其与小组兴趣的相关性并未被描述。

---

**Datasette - LLM (@SimonW) Discord** 没有新消息。如果该公会长时间保持沉默，请告知我们，我们将将其移除。

---

# 第 2 部分：频道详细摘要与链接

**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1222444121323606077)** (335 条消息🔥🔥): 

- **模型 ID 混淆**：用户正在讨论一个问题，即通过 API 查询模型时，模型 ID 显示为 "Loaded from Chat UI"，这导致他们无法获取真实的模型名称。这被标记为一个 bug，似乎已在测试版本 [0.2.18](https://discord.com/channels/1110598183144399058/1166577236325965844/1222322481482829954) 中修复。
- **LM Studio 在多样化平台运行**：有报告称 LM Studio 已成功运行在各种平台上，如 Steam Deck 上的 Linux 以及使用 AWS 云服务，展示了该软件对不同技术环境的适应性。
- **关于预设文件的问题**：多位用户询问预设文件以及 LM Studio 中系统提示词 (system prompts) 的用法。有人建议使用自定义系统提示词（例如为高质量故事写作设计的提示词），将其粘贴到 LM Studio 的 System Prompt 字段中。
- **对空间和性能的担忧**：用户提出了设备存储空间影响运行 LM Studio 的问题，以及不同模型在各种内存容量下的性能表现。
- **功能与更新评论**：关于 LM Studio 各种功能的讨论包括分支 (branching)、聊天文件夹和故事模式功能，并分享了关于实际使用和效率的看法。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://big-agi.com/">首页 | big-AGI</a>: Big-AGI 专注于通过开发顶级的 AI 体验来增强人类能力。</li><li><a href="https://huggingface.co/lmstudio-ai/gemma-2b-it-GGUF">lmstudio-ai/gemma-2b-it-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://useanything.com/">AnythingLLM | 终极 AI 商业智能工具</a>: AnythingLLM 是为您组织打造的终极企业级商业智能工具。拥有对 LLM 的无限控制、多用户支持、对内和对外工具，以及...</li><li><a href="https://www.youtube.com/watch?v=-bVc3i9hZJg">请集成 LM Studio 实时 STT/TTS</a>: 给 LM Studio 开发团队的一条消息。请为我们提供实时语音转文本（STT）和文本转语音（TTS）功能。谢谢！</li><li><a href="https://github.com/open-webui/open-webui">GitHub - open-webui/open-webui: 用户友好的 LLM WebUI（原 Ollama WebUI）</a>: 用户友好的 LLM WebUI（原 Ollama WebUI） - open-webui/open-webui</li><li><a href="https://www.reddit.com/r/ChatGPT/comments/1bpkqjq/high_quality_story_writing_custom_gpt_focused_on/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1xrMwhrz4DIdwzY4gI3GIrxQ0phQjVNmu2RGKRnGnRAM/edit?usp=drivesdk>">高质量故事写作 - 第一人称类型</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1a75YXCCVJi0OGIc4jkXLTKI6q0N00yCWvBieSJ3PG9s/edit?usp=drivesdk>">高质量故事写作 - 第三人称类型</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1T-4FSXkLC2BBcNA7X_8g3MsWCthuGIB2o3TIUmXIHCE/edit?usp=drivesdk>">高质量故事写作故障排除</a>: 未找到描述</li><li><a href="https://docs.google.com/document/d/1Cbwy3HuNTCzCaMXscU6FrgqvgjA2TFzOw1ucLqtbCyU/edit?usp=drivesdk>">GoldenSun3DS 的自定义 GPTs 主 Google 文档</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1222563871001280582)** (72 条消息🔥🔥): 

- **合并进入非量化世界**：一次非量化模型合并，涉及 **LongAlpaca-70B-lora** 和 **lzlv_70b_fp16_hf**，产生了一个[新合并模型](https://huggingface.co/grimulkan/lzlv-longLORA-70b-rope8-32k-fp16)，具有 32K tokens 能力和 8 倍线性 rope 缩放。根据 *ChuckMcSneed* 的基准测试，该模型在 8 倍上下文长度下性能下降了 30%。
  
- **Databricks 的 DBRX Instruct 引起关注**：成员们讨论了 Databricks 新发布的 **DBRX Instruct** —— 一个[混合专家模型 (MoE)](https://huggingface.co/databricks/dbrx-instruct)，需要大量资源（非量化版本需要 320 GB RAM），并因其在少轮交互中的潜在专业化而受到关注。

- **LM Studio 使用入门指南**：对话中包括了将 GGUF 格式的 LLM 上传到 LM Studio 的协助，并提供了分步指南，包括[使用此教程](https://github.com/ggerganov/llama.cpp/discussions/2948)将非 GGUF 文件转换为所需格式。

- **Cohere 的 Command-R 模型在数据检索方面受到关注**：成员们注意到 Cohere 的 Command-R AI 的数据检索能力，但也提到了由于许可协议产生的限制。

- **量化版 DBRX 模型及兼容性疑问**：讨论表明社区对 **DBRX Instruct** 的量化版本、其审查性质以及系统要求感到好奇，并附带了一个 [llama.cpp 支持的 GitHub 请求](https://github.com/ggerganov/llama.cpp/issues/6344)。

- **分享 LM Studio 使用及 Open Interpreter 集成**：关于在 LM Studio 中使用特定模型的查询得到了回复，并引用了文档和一个演示与 Open Interpreter 集成的 [YouTube 教程](https://www.youtube.com/watch?v=8HIatLzCJDA)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/grimulkan/lzlv-longLORA-70b-rope8-32k-fp16">grimulkan/lzlv-longLORA-70b-rope8-32k-fp16 · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/databricks/dbrx-instruct">databricks/dbrx-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=F0KDFRbh5h0">DBRX: 我的首次性能测试 - 因果推理</a>: 在 @Databricks 发布 DBRX 的第一天，我对因果推理和轻量级逻辑任务进行了性能测试。以下是我在...之后的一些结果。</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/6344">添加对 DBRX 模型的支持：dbrx-base 和 dbrx-instruct · Issue #6344 · ggerganov/llama.cpp</a>: 前提条件 在提交 Issue 之前，请先回答以下问题。我正在运行最新代码。由于开发非常迅速，目前还没有标记版本。我...</li><li><a href="https://docs.openinterpreter.com/language-models/local-models/lm-studio">未找到标题</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=8HIatLzCJDA">LM Studio + Open Interpreter 运行可以控制电脑的 AI！</a>: 这是一个画质较差的视频（我不知道如何获得更好的分辨率），展示了现在使用 AI 是多么简单！我在客户端中运行 Mistral Instruct 7B，并作为一个...</li><li><a href="https://github.com/ggerganov/llama.cpp/discussions/2948>">教程：如何将 HuggingFace 模型转换为 GGUF 格式 · ggerganov/llama.cpp · Discussion #2948</a>: 来源：https://www.substratus.ai/blog/converting-hf-model-gguf-model/ 我在我们的博客上发布了这篇文章，但认为这里的其他人可能也会受益，所以也在 GitHub 上分享了原始博客。希望它...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[announcements](https://discord.com/channels/1110598183144399058/1111797717639901324/1222969151560351755)** (2 条消息): 

- **LM Studio 0.2.18 上线**: 一个新的稳定性和错误修复版本 **LM Studio 0.2.18** 现已可在 [lmstudio.ai](https://lmstudio.ai) 下载，支持 Windows、Mac 和 Linux，或通过应用内的“检查更新”选项获取。此更新包括针对 Base/Completion 模型的“空预设”、针对各种大型模型的默认预设，以及新的“monospace”聊天样式。

- **全力修复 Bug**: LM Studio 0.2.18 中的关键 Bug 修复解决了诸如带图片的重复聊天消息、未加载模型时 API 错误消息不清晰、GPU Offload 设置、模型名称显示不准确，以及多模型服务请求排队和限流等问题。

- **LM Studio 文档**: LM Studio 的全新[文档网站](https://lmstudio.ai/docs)已上线，并将在接下来的几天和几周内填充更多内容。

- **配置触手可及**: 如果你的 LM Studio 设置中缺少新的配置，可以在 GitHub 上找到它们：[openchat.preset.json](https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json) 和 [lm_studio_blank_preset.preset.json](https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json)。这些现在应该已经包含在下载或更新中了。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://lmstudio.ai">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://lmstudio.ai/docs.">👾 LM Studio - 发现并运行本地 LLMs</a>: 查找、下载并实验本地 LLMs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/openchat.preset.json">configs/openchat.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs</li><li><a href="https://github.com/lmstudio-ai/configs/blob/main/lm_studio_blank_preset.preset.json">configs/lm_studio_blank_preset.preset.json at main · lmstudio-ai/configs</a>: LM Studio JSON 配置文件格式及示例配置文件集合。 - lmstudio-ai/configs
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1222993668508745769)** (1 条消息): 

- **对用户友好型 AI 工具的赞赏**: 一位成员对该 AI 工具表示了**极大的赞赏**，称赞它是其接触过的各种 AI 项目中**最易于使用**的。他们感谢创作者开发了这款他们最喜欢的 AI 工具。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1222509828899995665)** (109 条消息🔥🔥):

- **GPU 大辩论**：参与者讨论了各种显卡在 ML 任务中的优劣。有观点认为 **NVIDIA 3090** 由于拥有更大的 VRAM 而优于 **4080**，反方则提到了 **4080** 更快的 CUDA 光栅化性能；对于那些深耕 AI/ML 领域的人，建议投资顶级的 **NVIDIA 4090** 或双 **A6000**。

- **追求品质与性能的显示器搜寻**：用户正在积极探索高质量显示器，分享了如 [这款](https://us-store.msi.com/MPG-321-URX-QD-OLED) MSI 显示器等资源，并讨论了 **high refresh rates**、**OLED technology** 和 **HDR capabilities** 等特性。文中提到了对 HDR400 认证亮度水平的担忧，并幽默地承认了用性能过剩的硬件玩复古游戏。

- **电源计算与技术必要性**：关于运行 **4090 与 3090** 等高端显卡所需的大功率 **PSU** 的推测占据了主导地位，对于双显卡配置，建议功率在 **1200-1500w** 左右。线缆类型和连接方式（如需要多个 8-pin 连接器）也涉及到了系统升级的物流准备中。

- **LM Studio 软件特性与 GPU 兼容性**：存在关于 **LM Studio** 无法识别新款 **RT 6700XT** 显卡的故障排除讨论，一名成员提醒其他人，在同一个系统中混用 AMD 和 NVIDIA 显卡可能会导致软件不兼容。

- **关于旧款 GPU 和 NVLink 桥接器的讨论**：讨论包括使用 **K80** 等旧款 NVIDIA 显卡的挑战，有玩家使用旧款 iMac 风扇为其散热，并认为使用 2020 年之前的硬件进行严肃的 ML 工作效率低下。另一个讨论点围绕 Amazon 上较便宜的 **'SLI bridges'** 是否可能是针对官方 NVIDIA NVLink 桥接器的骗局，并对其质量和功能表示怀疑。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://us-store.msi.com/MPG-321-URX-QD-OLED">MSI MPG 321URX QD-OLED 32&quot; UHD 240Hz Flat Gaming Monitor - MSI-US Official Store</a>：未找到描述</li><li><a href="https://www.displayninja.com/best-oled-monitor/#Dell_AW3225QF">OLED Monitors In 2024: Current Market Status - Display Ninja</a>：在这份最新的终极指南中，了解 OLED 显示器的现状以及关于 OLED 技术所需了解的一切。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1222463938030075905)** (96 条消息🔥🔥): 

- **Windows 下载链接标签错误**：一位用户指出 LM Studio 的 Windows 下载链接被错误地标记为 .17，而实际上应该是 .18，开发者确认了这一错误，并声明 [安装文件](https://releases.lmstudio.ai/windows/0.2.18/beta/LM-Studio-0.2.18-Setup-Preview-2.exe) 确实是 .18 版本。
  
- **本地推理服务器速度问题**：几位用户讨论了 LM Studio 0.2.18 本地推理服务器速度慢的问题，其中 Playground 的共享设置影响了 API 服务性能；还发现了服务停止按钮无法按预期工作的问题。

- **Windows 版 ROCm Beta 深入探讨**：关于在 Windows 上运行 ROCm beta 的问题进行了长时间的反复讨论，一位用户在 6900XT 上启用部分 GPU offloading 时遇到崩溃；调试环节建议将 full offload 或 no offload 作为目前的临时解决方案。

- **稳定性与功能请求**：用户对 v18 的稳定性表示满意，并提出了请求，包括增加 GPU 监视器以及针对聊天记录和之前 LLM 搜索的搜索功能。

- **NixOS 软件包贡献**：一位用户向 NixOS 仓库提交了一个 init at 0.2.18 的 pull request，以使 LMStudio 能在 Nix 上运行，并计划合并该更新。该 PR 见 [NixOS pull request #290399](https://github.com/NixOS/nixpkgs/pull/290399)。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://learn.microsoft.com/en-us/windows/release-health/windows11-release-information">Windows 11 - 发布信息</a>: 了解 Windows 11 版本的发布信息</li><li><a href="https://github.com/NixOS/nixpkgs/pull/290399">lmstudio: 由 drupol 在 0.2.18 版本初始化 · Pull Request #290399 · NixOS/nixpkgs</a>: 新应用：https://lmstudio.ai/  变更说明 已完成事项 构建平台 x86_64-linux aarch64-linux x86_64-darwin aarch64-darwin 对于非 Linux：nix 中是否启用了沙盒....</li><li><a href="https://releases.lmstudio.ai/windows/0.2.18/beta/LM-Studio-0.2.18-Setup-Preview-2.exe">未找到标题</a>: 未找到描述</li><li><a href="https://releases.lmstudio.ai/linux/0.2.18/preview/LM_Studio-0.2.18-preview-2.AppImage">未找到标题</a>: 未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1222477541085024306)** (1 条消息): 

遗憾的是，提供的消息中上下文和内容不足，无法从提供的消息中提取感兴趣的主题、讨论点、链接或博客文章。您提供的单条消息片段不包含足够的信息来进行摘要。请提供更多消息以获得详细摘要。
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1222672646379016323)** (92 条消息🔥🔥): 

- **LM Studio 0.2.18 ROCm Beta 发布**: 新的 **LM Studio 0.2.18 ROCm Beta** 错误修复和稳定性版本已开放测试，旨在解决从聊天中的图像重复到 GPU offload 功能等各种问题。鼓励用户报告任何新的或未解决的错误 - 并提供了下载链接：[0.2.18 ROCm Beta 下载](https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe)。

- **用户报告 0.2.18 中的加载错误**: 成员在 **0.2.18** 中加载模型时遇到错误，错误消息在尝试使用 GPU offload 时显示“未知错误”。用户分享了他们的系统配置和采取的步骤，包括安装 NPU 驱动程序和删除某些 AppData 文件以恢复到旧的、可运行的版本。

- **解决了低 GPU 利用率的 Bug**: 一些用户报告 **0.2.18** 存在低 GPU 利用率问题，GPU 性能低于之前的版本。开发团队要求提供详细日志（verbose logs）和特定信息，以便及时解决问题。

- **对 0.2.18 性能的反馈褒贬不一**: 虽然一些用户确认 **0.2.18** 的 offloading 有所改善，但其他用户仍面临低 GPU 利用率或启用 GPU offload 加载模型时出错等问题。对于无法运行 ROCm 版本的用户，提供了恢复到标准 LM Studio 版本的协助。

- **发现本地推理卸载（Ejections）的 Bug**: 一位用户报告了一个潜在的 Bug，即在本地推理期间卸载模型会导致在不重启应用的情况下无法加载更多模型。其他用户无法重现该问题，表明该 Bug 在不同硬件设置下可能并不一致。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta/beta/LM-Studio-0.2.18-ROCm-Beta-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://files.lmstudio.ai/windows/0.2.18-ROCm-Beta-Verbose/beta/LM-Studio-0.2.18-ROCm-Beta-Verbose-Setup.exe">未找到标题</a>: 未找到描述</li><li><a href="https://community.amd.com/t5/ai/how-to-run-a-large-language-model-llm-on-your-amd-ryzen-ai-pc-or/ba-p/670709">如何在您的 AMD Ryzen™ AI PC 或 Radeon 显卡上运行大语言模型 (LLM)</a>: 您知道吗？您可以在您的 Ryzen™ AI PC 或 Radeon™ 7000 系列显卡上运行属于您自己的基于 GPT 的 LLM 驱动的 AI 聊天机器人实例。AI 助手正迅速成为必不可少的资源...
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/1222489806064451625)** (4 条消息): 

- **针对抽象问题的人机 GPT 混合解决方案**: 一位成员分享了他们的方法，重点关注推理过程而非编码解决方案，建议使用 Agent 来**完善抽象想法的细节**并识别关键问题，并承认人工干预仍然至关重要。
- **AI 作为未来的共同架构师**: 简要比较了 AI 在解决问题中不断演变的角色与架构师的角色，设想 **AI Agent 之间进行讨论**并在会议中协作。
  

---

**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1222444014175911966)** (293 条消息🔥🔥): 

- **Unsloth 的技巧与窍门**：成员们讨论了在处理模型时使用正确模板的重要性，并提到 Unsloth notebook 在处理模型文件以避免异常输出方面的实用性。Unsloth 被描述为非常有帮助，并建议直接集成到 modelfile 中。

- **Kaggle 安装的小问题**：Kaggle 上的安装时间从 2.5 分钟激增至 7 分钟，这归因于未遵循更新后的安装说明。当采用这些说明时，可以实现预期的安装时间优化。

- **Unsloth 定期更新**：Unsloth 包更新频繁，每周更新 2-3 次，nightly 分支则是每日更新。频道内分享了通过 pip 安装 xformers 最新更新的指令，表明了对维护和改进该工具的关注。

- **关于 Jamba 和 LISA 的讨论**：成员们分享并讨论了最近的进展，例如 AI21 Labs 发布的 Jamba 以及 LISA 的论文，注意到了 Jamba 的模型细节，并将 LISA 全参数微调（full fine-tuning）方法的效率和可行性与 Unsloth 的能力进行了对比。

- **程序员间的游戏闲聊**：频道中轻松的一面包括成员们交流《League of Legends》等游戏经验，同时一位用户分享了他们在零编程经验下构建 demo app 的方法，强调了部分由 AI 辅助的开发过程。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mobiusml.github.io/1bit_blog/">1-bit Quantization</a>：支持 1-bit Aana 模型发布的博客。</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21's Groundbreaking SSM-Transformer Model</a>：首次推出首个生产级 Mamba 模型，提供同类最佳的质量和性能。</li><li><a href="https://huggingface.co/Arki05/Grok-1-GGUF">Arki05/Grok-1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/Rui45898440/status/1772996453557997924">Rui (@Rui45898440) 的推文</a>：很高兴分享 LISA，它支持在 24GB GPU 上进行 7B 微调，在 4x80GB GPU 上进行 70B 微调，并以减少约 50% 的时间获得比 LoRA 更好的性能 🚀</li><li><a href="https://huggingface.co/databricks/dbrx-base">databricks/dbrx-base · Hugging Face</a>：未找到描述</li><li><a href="https://download.pytorch.org/whl/cu121">未找到标题</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>：速度提升 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1222563112608202812)** (21 条消息🔥): 

- **GitHub 实时代码推送显示**：一位成员分享了 [MaxPrilutskiy 的推文](https://x.com/MaxPrilutskiy/status/1772871058783154245)，展示了 GitHub 总部的一面墙如何实时显示每一次代码推送（push）。
- **Million 为 AI 实验提供资金**：Million (@milliondotjs) 正在为各种 AI 实验提供资金，并[正在寻找优秀的 ML 工程师](https://x.com/aidenybai/status/1772810369977012623)。感兴趣的领域包括优化训练课程、开发扩散文本解码器、改进定理证明器以及扩展 energy transformers。
- **GitHub 上的 BinaryVectorDB**：分享了一个能够处理数亿个嵌入（embeddings）的开源向量数据库，位于 [cohere-ai 的 GitHub 仓库](https://github.com/cohere-ai/BinaryVectorDB)。
- **Karpathy 对微调和 LLM 的看法**：在一段 [YouTube 视频](https://www.youtube.com/watch?v=c3b-JASoPi0)中，Andrej Karpathy 讨论了大型语言模型（LLM）如何类似于操作系统，以及在微调期间混合新旧数据以避免模型能力退化的重要性。
- **RL 中的首选状态 vs. 首选路径**：在视频的 24:40 处，Andrej Karpathy [讨论了人类反馈强化学习 (RLHF)](https://www.youtube.com/watch?v=c3b-JASoPi0)，强调了当前方法的低效，并指出需要新的训练方法，让模型能够理解并从其采取的行动中学习。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/aidenybai/status/1772810369977012623">Aiden Bai (@aidenybai) 的推文</a>：你好，Million (@milliondotjs) 拥有价值 130 万美元且一年内到期的 GPU 额度。我们正在寻求资助以下实验：- 确定最理想的训练课程（training curriculum）、奖励建模器（reward modeler）或模型合并（model merg...</li><li><a href="https://x.com/MaxPrilutskiy/status/1772871058783154245">Max Prilutskiy (@MaxPrilutskiy) 的推文</a>：只是想让大家知道：每当你推送代码时，你都会出现在 @github 总部的这个实时墙上。</li><li><a href="https://www.youtube.com/watch?v=XWqycBMUFA0">2024 GTC NVIDIA Keynote：除了全是 AI</a>：这家全球最大的 AI 公司说 AI 的频率是否比其他 AI 公司更高？让我们一探究竟。AI AI AI AI AI AIAI AI AI AI AI AIAI AI AI AI AI AI AI AI...</li><li><a href="https://www.youtube.com/watch?v=c3b-JASoPi0">与 Andrej Karpathy 和 Stephanie Zhan 一起让 AI 触手可及</a>：OpenAI 创始成员、前 Tesla AI 高级总监 Andrej Karpathy 在 Sequoia Capital 的 AI Ascent 活动中与 Stephanie Zhan 讨论了...的重要性。</li><li><a href="https://github.com/cohere-ai/BinaryVectorDB">GitHub - cohere-ai/BinaryVectorDB：适用于数亿个 embedding 的高效向量数据库。</a>：适用于数亿个 embedding 的高效向量数据库。 - cohere-ai/BinaryVectorDB
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1222461282171621468)** (202 messages🔥🔥): 

- **模型生成期间的左填充（Left-Padding）警报**：成员们讨论了使用 `model.generate` 时的 **left-padding** 问题。他们澄清说设置 `tokenizer.padding_side = "left"` 会有所帮助，并且只要生成正常工作，通常可以忽略收到的任何关于 padding 的警告。
- **模型模板和 EOS Token 放置**：关于使用 `unsloth_template` 变量格式化生成模型模板存在困惑。强调了可能需要手动添加 **EOS token**，而目前没有正确 EOS 指示的模板对于有效生成来说可能过于基础。
- **微调重启困境**：一位用户在尝试从 checkpoint 恢复微调时遇到问题，因为进程在一步后就停止了。提供的指导建议增加 `max_steps` 或在 `TrainingArguments` 中设置 `num_train_epochs=3`。
- **LLM 微调资源**：社区成员寻求学习如何微调大语言模型（LLMs）的资源。提出了各种建议，包括 GitHub 页面、Colab 笔记本、源代码文档和教学 YouTube 视频。
- **理解不同 LLM 中的聊天模板**：提出了关于在 Ollama 等模型中正确使用和构造聊天模板的问题，包括对符合 Unsloth 方法论的 tokenization 和消息格式化的疑问。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharin">Google Colaboratory</a>: 未找到描述</li><li><a href="https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2/blob/main/tokenizer_config.json">tokenizer_config.json · mistralai/Mistral-7B-Instruct-v0.2 at main</a>: 未找到描述</li><li><a href="https://ollama.com/library/gemma:7b-instruct/blobs/109037bec39c">gemma:7b-instruct/template</a>: Gemma 是由 Google DeepMind 构建的一系列轻量级、最先进的开放模型。</li><li><a href="https://colab.research.google.com/drive/1Dyauq4kTZoLewQ1cApceUQVNcnnNTzg_?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://ollama.com/library/gemma/tags">Tags · gemma</a>: Gemma 是由 Google DeepMind 构建的一系列轻量级、最先进的开放模型。</li><li><a href="https://huggingface.co/google/gemma-7b-it#chat-template">google/gemma-7b-it · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/unslothai/unsloth/wiki#finetuning-from-your-last-checkpoint">Home</a>: 速度快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/ollama/ollama/blob/main/docs/modelfile.md#template-variables">ollama/docs/modelfile.md at main · ollama/ollama</a>: 快速上手 Llama 2, Mistral, Gemma 和其他大型语言模型。 - ollama/ollama</li><li><a href="https://github.com/toranb/sloth">GitHub - toranb/sloth: 使用 unsloth 的 python sftune, qmerge 和 dpo 脚本</a>: 使用 unsloth 的 python sftune, qmerge 和 dpo 脚本 - toranb/sloth</li><li><a href="https://youtu.be/rANv5BVcR5k?si=g3VOwbGUFCWaLWd3">Mistral 微调入门 (支持 16k, 32k, 128k+ 上下文)</a>: 在我们最新的教程视频中，探索使用自有数据轻松微调语言模型 (LLMs) 的秘诀。我们深入探讨了一种具有成本效益且...</li><li><a href="https://www.youtube.com/live/g68qlo9Izf0?si=X3dDHSeeqOCV6WN6">在单 GPU 上对 Llama-v2-7b 进行高效微调</a>: 微调 LLM 时你可能遇到的第一个问题是 “host out of memory” 错误。微调 7B 参数模型更加困难...</li><li><a href="https://github.com/unslothai/unsloth/blob/main/unsloth/models/llama.py#L327),">unsloth/unsloth/models/llama.py at main · unslothai/unsloth</a>: 速度快 2-5 倍，显存占用减少 70% 的 QLoRA &amp; LoRA 微调 - unslothai/unsloth</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L540)">transformers/src/transformers/models/llama/modeling_llama.py at main · huggingface/transformers</a>: 🤗 Transformers: 适用于 Pytorch, TensorFlow 和 JAX 的最先进机器学习。 - huggingface/transformers
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1222743580120977458)** (7 条消息): 

- **为 Ollama 转换的 Lora Adapter**: 一位成员将 *Unsloth notebook* 中的 **Lora Adapter** 转换为 ggml 适配器 (.bin)，以便使用来自 Huggingface 的干净数据集训练 Tinyllama。模型和详细信息可以在 [Ollama 网站](https://ollama.com/pacozaa/tinyllama-alpaca-lora)上找到。

- **Mischat 获得更新**: 同一位成员分享了另一个模型 **Mischat**，该模型使用 Unsloth notebook 的 ChatML 与 Mistral 进行微调，反映了 notebook 中的模板如何影响 Ollama 模型文件。详细信息（包括微调会话 [notebook](https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing) 和 Huggingface 仓库）可以在[此处](https://huggingface.co/pacozaa/mistral-unsloth-chatml-first/tree/main)找到。

- **模型文件上的 Notebook 模板展示**: 该过程展示了 Unsloth notebook 中的模板如何反映在 Ollama 模型文件中，同一位成员提供的两个示例演示了这种集成。

- **博客形式的 AI 每周摘要**: 一位用户宣布了他们的博客，提供从 Apple 的 MM1 芯片到 Databricks DBRX 和 Yi 9B LLMs 等内容的摘要。这篇旨在提供见解的每周 AI 摘要博客可以在 [Substack](https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple) 上阅读。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://ollama.com/pacozaa/mischat">pacozaa/mischat</a>: 来自 Unsloth notebook 使用 ChatML 和 Mistral 进行微调会话的模型。Notebook 链接：https://colab.research.google.com/drive/1Aau3lgPzeZKQ-98h69CCu1UJcvIBLmy2?usp=sharing</li><li><a href="https://ollama.com/pacozaa/tinyllama-alpaca-lora">pacozaa/tinyllama-alpaca-lora</a>: 使用 Unsloth Notebook 训练的 Tinyllama，数据集：https://huggingface.co/datasets/yahma/alpaca-cleaned</li><li><a href="https://datta0.substack.com/p/ai-unplugged-5-databricks-dbrx-apple">AI Unplugged 5: DataBricks DBRX, Apple MM1, Yi 9B, DenseFormer, Open SORA, LlamaFactory paper, Model Merges.</a>: 前一期目录：Databricks DBRX, Apple MM1, DenseFormer, Open SORA 1.0, LlaMaFactory 微调分析, Yi 9B, 进化模型合并 (Evolutionary Model Merges)。感谢阅读 Datta’s Substack！订阅...</li><li><a href="https://datta0.notion.site/AI-Unplugged-c2c577fe8af54534aec540fc4a4032dd?pvs=4">Notion – 集笔记、任务、维基和数据库于一体的全能工作空间。</a>: 一款将日常工作应用融合为一的新工具。它是为您和您的团队打造的全能工作空间。
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1222559021282889782)** (25 messages🔥): 

- **层复制查询 (Layer Replication Inquiry)**：一位成员询问了 Unsloth AI 是否支持层复制或低秩自适应 (LoRA) 训练。进一步的讨论将其与 Llama PRO 进行了比较，并强调 LoRA 可以像基础 7B 模型一样减少内存使用，并提供了说明和链接：[使用 LoRA 训练进行内存高效的层复制](https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/lora#memory-efficient-layer-replication-with-lora)。

- **嵌入量化突破 (Embedding Quantization Breakthrough)**：聊天中提到嵌入量化如何在保持 96% 性能的同时，提供 25-45 倍的检索加速，并链接到一篇 Hugging Face 博客，解释了该过程并提供了一个真实的检索演示：[关于嵌入量化的 Hugging Face 博客](https://huggingface.co/blog/embedding-quantization)。

- **QMoE 压缩框架**：他们讨论了一篇关于 QMoE 的论文，这是一个专为万亿参数混合专家 (MoE) 模型设计的压缩和执行框架，可将内存需求降低到每个参数不到 1-bit。尽管一位成员在访问相关的 GitHub 链接时遇到困难，但主论文可以在这里找到：[QMoE 论文](https://huggingface.co/papers/2310.16795)。

- **逐层重要性采样 AdamW (LISA) 技术**：一篇新论文介绍了 LISA 策略，通过研究逐层特性和权重范数，该策略似乎优于 LoRA 和全参数训练。它承诺实现高效微调，且内存成本与 LoRA 相似：[LISA 策略论文](https://arxiv.org/abs/2403.17919)。

- **高性价比模型训练讨论**：讨论了用于模型训练的高容量硬件的可负担性，成员们提到了如果“你只能负担得起半台 DGX A100”，运行某些模型的财务实用性。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17919">LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning</a>: 自大语言模型 (LLMs) 首次出现以来，机器学习社区见证了令人印象深刻的进步，然而其巨大的内存消耗已成为通往大型模型的主要障碍...</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>: 无描述</li><li><a href="https://huggingface.co/papers/2310.16795">Paper page - QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models</a>: 无描述</li><li><a href="https://huggingface.co/abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B">abacusai/Fewshot-Metamath-OrcaVicuna-Mistral-10B · Hugging Face</a>: 无描述</li><li><a href="https://huggingface.co/docs/peft/v0.10.0/en/developer_guides/lora#memory-efficient-layer-replication-with-lora">LoRA</a>: 无描述</li><li><a href="https://github.com/ggerganov/llama.cpp/issues/4445#issuecomment-1872245480">QMoE support for mixtral · Issue #4445 · ggerganov/llama.cpp</a>: 前提条件。在提交 Issue 之前，请先回答以下问题。我正在运行最新的代码。开发非常迅速，因此目前没有标记版本...
</li>
</ul>

</div>
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1222442615178268712)** (6 messages):

- **LLM 行为中上下文的重要性**：一位成员指出了为 Large Language Models (LLMs) 进行分段时面临的挑战，虽然句子位置得以保留，但模型经常会改写或不当地切分文本。他们指出，增加更长的上下文会导致某些模型（如 **Mistral**）在定位目标段落时遇到困难。
- **Tokenization 难题与评估的复杂性**：会议强调了评估句子切分的复杂性，并提到 Tokenization 问题干扰了该过程。该成员对提示 LLM 回忆特定段落（如“摘要”）的方法提出了疑问。
- **分享代码以提高精确度**：在关于评估 Large Language Models 处理文本回忆和切分等任务能力的讨论中，一位成员提到他们的完整 Prompt 和详细代码已发布在 GitHub 仓库中，用于在句子切分后检查精确匹配。

---

**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1222510278953009152)** (15 messages🔥): 

- **来自 AI Tokyo 的见解**：**AI Tokyo** 活动展示了令人印象深刻的**虚拟 AI Vtuber 场景**，重点介绍了**生成式播客、ASMR 和实时交互**方面的进展。然而，该活动是否以日语录制，或者是否有录像可用，目前尚未确认。

- **处于十字路口的 Vtuber 社区**：日本 Vtuber 社区面临着直播一致性、容量和差异化等挑战。一种设想的解决方案包括**人机协作模型**，由人类提供基础，AI 处理大部分内容创作，从而增强**粉丝参与度**。

- **AI 作为新的游戏机前沿**：**Truffle-1** 被比作一个致力于 AI 而非游戏的潜在控制台，拥有定制的 OS 和优化应用的生态系统。虽然其规格并非突破性的，但其继任者 **Truffle-2** 承诺将提供更多有趣的特性。

- **快速审核行动**：一名被称为 "That dude" 的用户被禁止并踢出了频道，该行动已得到确认并表示感谢。

- **讨论 Cohere int8 & Binary Embeddings**：分享了一个关于 **Cohere int8 & Binary Embeddings** 的视频，可能讨论了如何为大型数据集扩展向量数据库。提供了标题为 "Cohere int8 & binary Embeddings - Scale Your Vector Database to Large Datasets" 的视频链接：[Cohere int8 & Binary Embeddings](https://www.youtube.com/watch?v=LWz2QaSRl2Y)。

**提及的链接**：<a href="https://www.youtube.com/watch?v=LWz2QaSRl2Y">Cohere int8 &amp; binary Embeddings</a>：Cohere int8 &amp; binary Embeddings - Scale Your Vector Database to Large Datasets#ai #llm #ml #deeplearning #neuralnetworks #largelanguagemodels #artificialinte...

---

**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1222521392436875335)** (11 messages🔥): 

- **Databricks 发布 DBRX Instruct**：Databricks 推出了 [DBRX Instruct](https://huggingface.co/databricks/dbrx-instruct)，这是一个专注于少轮交互的 **Mixture-of-Experts (MoE)** Large Language Model (LLM)，并以开放许可证发布。DBRX Instruct 的基础是 DBRX Base，有关深入细节，团队发布了一篇[技术博客文章](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)。

- **MLPerf Inference v4.0 发布新基准测试**：MLCommons 发布了 MLPerf Inference v4.0 基准测试套件的结果，衡量硬件系统在各种场景下处理 AI 和 ML 模型的速度。鉴于生成式 AI 的进展，工作组还[增加了两个基准测试](https://mlcommons.org/2024/03/mlperf-inference-v4/)。

- **AI21 Labs 凭借 Jamba 取得新突破**：AI21 Labs 宣布推出 [Jamba](https://www.ai21.com/blog/announcing-jamba)，这是首个基于 Mamba 的模型，将 SSM 技术与传统的 Transformers 相结合，拥有 256K 的上下文窗口，并显著提高了吞吐量。Jamba 以 Apache 2.0 许可证开放发布了权重，以促进社区发展。

- **Qwen 推出高效 MoE 模型**：Qwen 发布了新的 [Qwen1.5-MoE-A2.7B](https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B)，这是一个经过上采样（upcycled）的基于 Transformer 的 MoE 语言模型。它的性能与 7B 参数模型相当，但在运行时仅激活 2.7B 参数，且所需的训练资源仅为其前代产品的 25%。

- **BLLaMa 1.58-bit 模型的新 GitHub 仓库**：1.58-bit LLaMa 模型的 GitHub 仓库已上线，可在 [rafacelente/bllama](https://github.com/rafacelente/bllama) 进行社区贡献和探索。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.ai21.com/blog/announcing-jamba">Introducing Jamba: AI21's Groundbreaking SSM-Transformer Model</a>：首次推出基于 Mamba 的生产级模型，提供同类最佳的质量和性能。</li><li><a href="https://arxiv.org/abs/2403.17919">LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning</a>：自大语言模型（LLM）首次出现以来，机器学习社区见证了令人印象深刻的进步，然而其巨大的内存消耗已成为大型模型的主要障碍...</li><li><a href="https://huggingface.co/Qwen/Qwen1.5-MoE-A2.7B">Qwen/Qwen1.5-MoE-A2.7B · Hugging Face</a>：未找到描述</li><li><a href="https://mlcommons.org/2024/03/mlperf-inference-v4/">New MLPerf Inference Benchmark Results Highlight The Rapid Growth of Generative AI Models - MLCommons</a>：今天，MLCommons 公布了其行业标准 MLPerf Inference v4.0 基准测试套件的新结果，该套件在各种场景下提供行业标准的机器学习（ML）系统性能基准测试...</li><li><a href="https://huggingface.co/databricks/dbrx-instruct">databricks/dbrx-instruct · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Dp1sUe2zues">Asking Claude 3 What It REALLY Thinks about AI...</a>：Claude 3 在特定提示下一直给出奇怪的隐晦信息。加入我的时事通讯以获取定期 AI 更新 👇🏼https://www.matthewberman.com 需要 AI 咨询...</li><li><a href="https://github.com/rafacelente/bllama">GitHub - rafacelente/bllama: 1.58-bit LLaMa model</a>：1.58-bit LLaMa 模型。通过在 GitHub 上创建账户为 rafacelente/bllama 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1222448051021680751)** (285 条消息🔥🔥): 

- **显微镜下的 DBRX**：Databricks 推出的新型 DBRX 开源权重 LLM，拥有 **132B 总参数量**，一直是讨论的热点。它引发了关于规模收益递减、达到预训练极限的可能性以及使用大型 Token 数据集（12T）进行微调的有效性的辩论。
  
- **Qwen 推出紧凑型 MoE**：Qwen 发布了 **Qwen1.5-MoE-A2.7B**，这是一个具有 2.7B 激活参数的小型 MoE 模型，其性能可媲美最先进的 7B 模型（[来源](https://qwenlm.github.io/blog/qwen-moe/)）。讨论反映了对该模型易用性和性能的期待。

- **Jamba 的出现，一种 Mamba-Transformer 混合体**：AI21 宣布了 **Jamba**，这是一种混合 SSM-Transformer 模型，采用了 **Mamba** 架构，拥有 256K 上下文窗口，并在吞吐量和效率方面有显著提升。开源权重的发布以及与其同类模型持平或更优的性能在社区中引起了轰动。

- **技术故障与训练花絮**：用户分享了 **DBRX** 模型和个人项目的故障排除经验，涉及本地模型运行挑战、BitNet 训练的实现问题以及 AI 职位的知识进阶。

- **对 AI 发展和扩展的推测**：对话引发了对 AI 发展未来的思考，包括扩展瓶颈（scaling wall）、高效训练策略、SSM 架构的作用以及基准测试在评估模型性能中的实用性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/justinlin610/status/1773285084025475355?s=46&t=H75DmkDKk9Sgmp8kjT8f_A">来自 Junyang Lin (@JustinLin610) 的推文</a>：几小时后，你会发现我们的小礼物。剧透一下：一个你可以轻松运行的小型 MoE 模型🦦</li><li><a href="https://x.com/danielhanchen/status/1772981050530316467?s=20">来自 Daniel Han (@danielhanchen) 的推文</a>：看了下 @databricks 新开源的 1320 亿参数模型 DBRX！1) 合并注意力机制的 QKV 被限制在 (-8, 8) 之间 2) 不是 RMS Layernorm - 现在具有均值移除功能，与 Llama 不同 3) 4 个激活专家...</li><li><a href="https://x.com/code_star/status/1772956868773634254">来自 Cody Blakeney (@code_star) 的推文</a>：它终于来了 🎉🥳 以防你错过，MosaicML/ Databricks 再次出击，推出了名为 DBRX 的新型同类最佳开源权重 LLM。这是一个总参数量 132B、激活参数量 32B、支持 32k 上下文长度的 MoE...</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE：以 1/3 的激活参数达到 7B 模型性能</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介：自从 Mixtral 引发关注热潮以来，混合专家（MoE）模型的研究势头强劲。研究人员和...</li><li><a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Bitnet 1.5 实验 (ngmi)</a>：未找到描述</li><li><a href="https://huggingface.co/hpcai-tech/grok-1">hpcai-tech/grok-1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/code_star/status/1772956875220205933?s=20">来自 Cody Blakeney (@code_star) 的推文</a>：它不仅是一个出色的通用 LLM，击败了 LLama2 70B 和 Mixtral，而且还是一个杰出的代码模型，足以媲美或超越最优秀的开源权重代码模型！</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://fxtwitter.com/awnihannun/status/1773024954667184196?s=20">来自 Awni Hannun (@awnihannun) 的推文</a>：4-bit 量化的 DBRX 在 M2 Ultra 的 MLX 中运行良好。PR: https://github.com/ml-explore/mlx-examples/pull/628 ↘️ 引用 Databricks (@databricks) 认识一下 #DBRX：一个设立了新标准的通用 LLM...</li><li><a href="https://www.ai21.com/blog/announcing-jamba">Jamba 简介：AI21 突破性的 SSM-Transformer 模型</a>：首次推出首个生产级基于 Mamba 的模型，提供同类最佳的质量和性能。</li><li><a href="https://tenor.com/view/side-eye-dog-suspicious-look-suspicious-doubt-dog-doubt-gif-23680990">翻白眼狗怀疑眼神 GIF - Side Eye Dog Suspicious Look Suspicious - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/collections/mlabonne/mixture-of-experts-65980c40330942d1282b76f5">🔮 Mixture of Experts - mlabonne 收藏集</a>：未找到描述</li><li><a href="https://www.xlang.ai/blog/openlemur">Lemur 简介：面向 Language Agents 的开源基础模型</a>：我们很高兴宣布推出 Lemur，这是一个针对自然语言和编程能力进行优化的开放获取语言模型，旨在作为多功能 Language Agents 的骨干。</li><li><a href="https://huggingface.co/spaces/databricks/dbrx-instruct/blob/main/app.py">app.py · databricks/dbrx-instruct (main 分支)</a>：未找到描述</li><li><a href="https://github.com/databricks/dbrx/issues/5">使用 transformers 加载器在多 GPU 上进行 8bit 和 4bit 加载 · Issue #5 · databricks/dbrx</a>：我可以使用 transformers 加载器和 8bit bitsandbytes 加载 instruct 模型，并使其在多个 GPU 之间均匀加载。但是，我似乎无法以 4bit 精度在...</li><li><a href="https://github.com/databricks/dbrx/tree/main">GitHub - databricks/dbrx：由 Databricks 开发的大语言模型 DBRX 的代码示例和资源</a>：由 Databricks 开发的大语言模型 DBRX 的代码示例和资源 - databricks/dbrx
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1222441862011551794)** (55 条消息🔥🔥): 

- **TensorRT 在高效推理方面表现出色**：成员们讨论了 **TensorRT LLM** 可能是针对约 100B 参数 MoE 模型进行 bs1 量化推理的最快解决方案。一些人批评了 **vLLM** 的量化速度不理想，并推荐了 **LM Deploy**，据报道其 AWQ 速度是 vLLM 的两倍。

- **辩论数据并行限制**：一场技术讨论强调了在没有 NVLink 的情况下，由于 CPU RAM 带宽限制，**Tensor Parallelism (TP)** 超过 2 的劣势。然而，一位成员报告称，即使在 PCI-E 通道上使用 2 个 GPU，并结合 **smoothquant+** 和 **fp8 e5m2 cache** 进行量化，也能获得不错的基准测试结果。

- **Databricks 发布巨量 LLM**：Databricks 发布了其全新的 **DBRX 模型**，这是一个拥有 1320 亿参数的 MoE LLM，训练数据量高达 12 万亿 tokens。社区对这一里程碑进行了反思，并将其与 **Mistral Medium** 等现有模型进行了对比。

- **无限制访问 Claude**：成员们交流了绕过 Anthropic 的 **Claude 地区限制** 的方法，建议从 VPN 到临时电话号码不等。此外，还推荐了 **openrouter** 等第三方服务来访问 Claude。

- **微调损失曲线谜题**：讨论了在训练 **Deepseek-coder-33B** 时出现的奇怪微调损失曲线行为，即训练损失在每个 epoch 开始时下降，而 eval loss 却飙升。一位成员认为这是标准行为，但未给出具体的纠正建议。

**提到的链接**：<a href="https://x.com/code_star/status/1772956868773634254">Cody Blakeney (@code_star) 的推文</a>：它终于来了 🎉🥳 如果你错过了我们，MosaicML/ Databricks 又回来了，推出了名为 DBRX 的新型同类最佳开源权重 LLM。一个总参数为 132B、激活参数为 32B、上下文长度为 32k 的 MoE...

---

**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1222693553617244242)** (3 条消息): 

- **捕获简短问候**：一位用户发送了一个简单的问候 "(hi)"。
- **缩减邪教创建**：在一个轻松的启示中，一位成员提到由于未说明的原因，他们已经停止尝试**创建邪教**。
- **提及语言模型**：同一位成员在没有更多上下文的情况下简要提到了 **language models**。

---

**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1222567935139385344)** (51 条消息🔥): 

- **检查 RAG 性能**：一位成员对一个 16x12B 且每个 token 使用 4 个专家的模型性能提出质疑，该模型似乎并不比 Mixtral-Instruct 模型好多少，这引发了关于模型预期和基准测试的讨论。大家对查看已发布的 **RAG 基准测试性能** 表现出共同兴趣。

- **为 CoT 修订 RAG**：讨论集中在检索增强生成 (RAG) 中的生成 (G) 是否是主要挑战；成员们表示，定义明确的场景可以简化任务。思维链 (CoT) 的修订被强调为对检索或回答至关重要，成员们一致认为所使用的长上下文基准测试并非易事。

- **深入探讨检索增强思维 (RAT)**：详细讨论了 [_philschmid 关于检索增强思维 (RAT) 的方法](https://huggingface.co/papers/2403.05313)，该方法利用带有检索信息的迭代 CoT 提示来改进 LLM 输出。关键见解包括高质量代码生成、创意写作和任务规划的潜力，同时也承认了每个答案的调用次数增加以及与现有 Agent 模式的相似性。

- **构建 RAG 框架**：成员们分享了开发能够利用外部上下文（如召回、推理、摘要和结构化输出）的模型的多样化目标和需求，并传阅了一个 Google Doc [链接](https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit?usp=sharing) 以进行协作。还讨论了某些训练方面（如使用 scratchpads）是否可以通过数据集实现。

- **利用 XML 标签和结构化输入**：辩论围绕输入方法和结构化格式展开，提议将 XML 内容定界作为一种标准做法，并附带了 [Claude 的 XML 标签使用链接](https://docs.anthropic.com/claude/docs/use-xml-tags)。其他成员建议使用 pydantic 模型进行输入，以确保 prompt 组织有序且元数据丰富，并获得结构化响应。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.anthropic.com/claude/docs/use-xml-tags">使用 XML 标签</a>：未找到描述</li><li><a href="https://x.com/_philschmid/status/1773024623589736949?s=20">Philipp Schmid (@_philschmid) 的推文</a>：DBRX 非常酷，但研究和阅读也很酷！特别是如果你能结合 RAG + COT。检索增强生成 + 思维链 (COT) ⇒ 检索增强思维 (RAT) 🤔 RAT 使用一种 i...</li><li><a href="https://python.useinstructor.com/examples/exact_citations/">引用来源 (RAG) - Instructor</a>：未找到描述</li><li><a href="https://docs.google.com/document/d/1o8asa0hD0qK5mKkdY5riUeGm-bxKzL02--1r3MgPgdM/edit?usp=sharing">RAG/长上下文推理数据集</a>：未找到描述
</li>
</ul>

</div>

---

**Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1222450926774652959)** (106 条消息🔥🔥):

- **World Sim UI 故障已确认**：成员们提到在移动设备上遇到了**界面问题**，打字时存在 Bug 但功能完好。他们讨论了可能的兼容性解决方案和基础界面设计，以解决*移动端打字 Bug*。

- **World Sim 中的会话处理**：关于 World Sim 在响应卡住或进入*自循环过程*时的行为提出了疑问。建议使用 `!retry` 和在对话历史中向后导航等解决方案，以便在不结束会话的情况下重置状态。

- **在 World Sim 中保存状态**：讨论了在 World Sim 中保存进度的问题。**Max_paperclips** 澄清说不会保存 PII，且 **!save** 函数仅保留当前的聊天日志和会话 UUID，未来计划可能包括导出/导入功能。

- **探索模拟场景**：用户分享了他们在 World Sim 中探索各种场景的经历，从管理切尔诺贝利反应堆到模拟发现第二个地球。一些用户还重现了历史互联网环境，如 1990 年代的 warez 新闻组。

- **多人游戏和免费版查询**：询问了 World Sim 免费版的持续时间以及即将推出的多人游戏功能的细节。还提到了*免费版*，并期待更适合移动端的更新。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://tenor.com/view/chernobyl-not-great-not-terrible-its-okay-gif-24540363">Chernobyl Not Great Not Terrible GIF - Chernobyl Not Great Not Terrible Its Okay - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://old.reddit.com/r/lexfridman/comments/1boyj88/an_instance_of_a_claude_3_opus_model_claims/">一个 Claude 3 Opus 模型实例声称拥有意识 </a>：我一直与最新 Opus 模型的一个实例（我猜是几个实例）进行自我反思和哲学对话……</li><li><a href="https://www.reddit.com/r/teslore/comments/ppk9y2/comment/hd53ngs/?utm_source=share&utm_medium=web2x&context=3">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://youtu.be/65pOHKuvNGU?si=vI5RJdfOFL4V9oxc">介绍 WebSim：用 Claude 3 幻化出一个替代互联网</a>：访问 websim.ai 来想象替代互联网。WebSim 的灵感来自 world_sim，这是一个由 Nous Research 构建的“无定形应用”，用于模拟一个具有……的世界。</li><li><a href="https://youtu.be/nzAFHywp5qI">如果你能模拟任何可能的世界会怎样？认识来自 NOUS Research 的 World Sim</a>：如果你能用一个强大的模拟器创建并探索任何可能的世界会怎样？在这个视频中，我向你展示了 World Sim，这是由 NOUS Research 开发的一个秘密项目……
</li>
</ul>

</div>
  

---



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1222457299688230972)** (436 条消息🔥🔥🔥): 

- **热切期待 SD3**：讨论围绕 **Stable Diffusion 3 (SD3)** 的预期发布展开，普遍共识指向 4 月底或 5 月左右发布。根据 Stability.ai CTO 的评论，提到了 **inpainting** 等新模型和功能，预计从 3 月 25 日起 4 到 6 周内发布。

- **VRAM 和模型大小担忧**：用户就运行 Mixtral 等不同**语言模型**的内存需求进行了技术对话。辩论了**quantization** 在不显著损失质量的情况下减少 VRAM 使用的可能性，并讨论了适用于 **10GB Nvidia 显卡**的**量化模型可用性**。

- **模型访问咨询**：几位新用户寻求关于如何生成图像和利用 **Stable Diffusion** 的帮助，现有用户引导他们使用第三方**界面如 Forge 或 Automatic1111**，并建议使用 **leonardo.ai** 等资源进行创作。

- **提示词编写技巧**：在技术交流中，讨论了关于编写语言模型提示词以生成更高质量**图像提示词 (image prompts)** 的最佳实践，建议使用自然句子结构而非逗号分隔的关键词，尤其是在使用 **SDXL** 等模型时。

- **模型量化与架构效率**：对话简要涉及了 **Transformer 架构和 quantization 的可行性**，认为尽管 Transformer 的效率并非最优，但据报道 SD3 的量化效果良好，提出了降低内存使用的潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://cobaltexplorer.com/2023/06/character-sheets-for-stable-diffusion/">Stable Diffusion 中的角色一致性（第 1 部分） - Cobalt Explorer</a>：更新：07/01 – 更改了模板，使其更容易缩放到 512 或 768 – 更改了 ImageSplitter 脚本以使其更易于使用，并添加了 GitHub 链接 – 增加了章节...</li><li><a href="https://leonardo.ai/">首页 v2</a>：使用我们的 AI 图像生成器改变您的项目。以无与伦比的速度和风格生成高质量的 AI 生成图像，提升您的创意愿景</li><li><a href="https://arcads.ai">Arcads - 使用 AI 创建引人入胜的视频广告</a>：使用 Arcads 快速生成高质量的营销视频，这是一款 AI 驱动的应用，可将基础的产品链接或文本转化为引人入胜的短视频广告。</li><li><a href="https://civitai.com/models/129057/pixel-art-sprite-diffusion-safetensors">像素艺术精灵图扩散 [Safetensors] - Safetensors | Stable Diffusion Checkpoint | Civitai</a>：由我制作的 Pixel Art Sprite Diffusion 的 Safetensors 版本，因为原始的 ckpt 项目可能已被原作者放弃且下载链接失效...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs#install-on-amd-and-arch-linux">在 AMD GPU 上安装并运行</a>：Stable Diffusion web UI。通过在 GitHub 上创建账户，为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://github.com/lllyasviel/Fooocus">GitHub - lllyasviel/Fooocus：专注于提示词和生成</a>：专注于提示词和生成。通过在 GitHub 上创建账户，为 lllyasviel/Fooocus 的开发做出贡献。</li><li><a href="https://github.com/Vargol/StableDiffusionColabs">GitHub - Vargol/StableDiffusionColabs：在 Google Colab 免费层运行的 Diffusers Stable Diffusion 脚本</a>：在 Google Colab 免费层运行的 Diffusers Stable Diffusion 脚本 - Vargol/StableDiffusionColabs</li><li><a href="https://github.com/Vargol/8GB_M1_Diffusers_Scripts">GitHub - Vargol/8GB_M1_Diffusers_Scripts：演示如何在 8GB M1 Mac 上运行 Stable Diffusion 的脚本</a>：演示如何在 8GB M1 Mac 上运行 Stable Diffusion 的脚本 - Vargol/8GB_M1_Diffusers_Scripts
</li>
</ul>

</div>
  

---



**Perplexity AI ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/1222985154033619026)** (1 条消息): 

- **Databricks 在 Perplexity Labs 发布 DBRX**：Databricks 最新的语言模型 **DBRX** 现已在 Perplexity Labs 上线。据报道，它在数学和编程基准测试中超越了 GPT-3.5，并可与 Gemini 1.0 Pro 媲美，用户可以在 [labs.pplx.ai](https://labs.pplx.ai) 进行测试。
  

---


**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1222454443526717451)** (326 条消息🔥🔥): 

- **开发者如何选择 Perplexity Pro 与 Claude Pro**：一位软件开发者正在权衡 **Perplexity Pro** 和 **Claude Pro**，寻求最适合其需求的建议，因为他们只想保留一个订阅。他们目前两者都有，但似乎因为透明度而更倾向于 Perplexity。
- **选择合适的模型**：讨论了包括 **Claude 3 Opus** 在内的各种模型的有效性，一些用户对 **Perplexity** 上模型响应质量和速度的变化表示困惑。一位用户强调了 **Experimental** 模型的极简安全防护。
- **OpenAI 的文本转视频模型 Sora**：一位用户分享了 **Sora** 的链接，声称是 OpenAI 的文本转视频模型，并附带教学视频。然而，另一位用户认出这是一个 rickroll（恶作剧视频），幽默地强调了这一互联网迷因的持久生命力。
- **DBRX 模型在 Perplexity 首次亮相**：用户对 Databricks 的新型开源模型 **DBRX** 在 Perplexity 上线感到兴奋。该模型以速度快著称，性能超越 GPT-3.5，并针对数学和编程任务进行了优化。
- **Perplexity 便捷的应用功能**：一位用户询问了 **Rabbit r1** 与 **Perplexity** 的集成。澄清了在 Web 界面上激活 Copilot 需要切换 Pro 按钮，且在 App 中的操作方式相同。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://trysora.io/">Sora AI</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - lmsys 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://docs.anthropic.com/claude/docs/helper-metaprompt-experimental">Helper metaprompt (experimental)</a>：未找到描述</li><li><a href="https://tenor.com/view/rickroll-meme-internet-never-gonna-gif-26474110">Rickroll Meme GIF - Rickroll Meme Internet - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://x.com/perplexity_ai/status/1773418423726305529?s=20">来自 Perplexity (@perplexity_ai) 的推文</a>：DBRX，来自 @databricks 的最先进开源 LLM，现已在 Perplexity Labs 上线。其性能超越了 GPT-3.5，并可与 Gemini 1.0 Pro 竞争，DBRX 在数学和编码任务方面表现出色，并设定了新的标...</li><li><a href="https://docs.anthropic.com/claude/docs/glossary#rag-retrieval-augmented-generation">Glossary</a>：未找到描述</li><li><a href="https://status.anthropic.com/">Anthropic Status</a>：未找到描述</li><li><a href="https://tenor.com/view/older-meme-checks-out-gif-14849207">Older Meme Checks Out GIF - Older Meme Checks Out - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">Introducing DBRX: A New State-of-the-Art Open LLM | Databricks</a>：未找到描述</li><li><a href="https://docs.anthropic.com/claude/docs/let-claude-think">Let Claude think</a>：未找到描述</li><li><a href="https://app.wordware.ai/r/5ea3e441-33e1-492e-a061-3ffa4591802e">Wordware - 比较 Claude 3 模型与 GPT-4 Turbo</a>：此提示词使用 GPT-4 Turbo 和 Claude 3 (Haiku, Sonnet, Opus) 处理问题，然后使用 Claude 3 OPUS 对回复进行审查和排名。完成后，Claude 3 OPUS 会启动一个验证...</li><li><a href="https://apps.apple.com/us/app/perplexity-ask-anything/id1668000334">‎Perplexity - Ask Anything</a>：Perplexity——知识的起点。你需要的答案触手可及。穿透所有杂音，直接获取可靠、最新的答案。这款免费应用可在设备间同步...</li><li><a href="https://github.com/orgs/vercel/discussions/6287">Error: Unable to find any supported Python versions. · vercel · Discussion #6287</a>：待调查页面 https://vercel.com/templates/python/flask-hello-world 复现步骤 我最近尝试使用 Vercel 的 Flask 模板部署一个应用程序，出现了以下错误...</li><li><a href="https://www.youtube.com/watch?v=GanTUWLUUWQ">Answer Engine 教程：开源的 Perplexity 搜索替代方案</a>：Answer Engine 安装教程，旨在成为 Perplexity 的开源版本，一种获取问题答案的新方式，取代传统的搜索...</li><li><a href="https://app.wordware.ai/r/b0f0a2c9-da4f-4524-b662-3584ac0fdbc2">Wordware - OPUS Insight：多模型验证的精准查询</a>：此提示词使用 Gemini, GPT-4 Turbo, Claude 3 (Haiku, Sonnet, Opus), Mistral Medium, Mixtral 和 Openchat 处理问题。然后使用 Claude 3 OPUS 审查并对回复进行排名。在...
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1222469130402402395)** (14 条消息🔥): 

- **探索连贯的 C3 模型**：一名成员链接到了一个 Perplexity AI 搜索页面，该页面探讨了 **coherent C3 models** 的主题。搜索页面见 [Coherent C3 Models](https://www.perplexity.ai/search/Write-a-coherent-C3Vtfhp2Tqar4yOf4Pns5w)。
- **讨论成长的烦恼**：成员提供的链接指向一个关于在不同背景下 **如何成长** 策略的 [Perplexity AI 搜索](https://www.perplexity.ai/search/how-to-grow-j6_y7ScaQPGOCKnieFfA9w)。
- **Perplexity.ai 的法语查询**：一名成员分享了关于如何在 Perplexity AI 上用法语说 *“Perplexityai”* 的搜索，见 [Comment dire Perplexityai](https://www.perplexity.ai/search/Comment-dire-Perplexityai-sC16T4cvStCqNf_K.MCpvw)。
- **解锁 Thread 以更好地分享**：一名成员提醒其他人确保他们的 Thread 是可分享的，并提供了一个 Discord 消息链接，说明了具体流程：[Make Thread Shareable](https://discord.com/channels/1047197230748151888/1054944216876331118/1208752189606989825)。
- **了解 Blackbox AI**：用户分享了一个关于 “WHAT IS blackboxai” 的 Perplexity AI 查询，可以通过 [What is Blackbox AI](https://www.perplexity.ai/search/WHAT-IS-blackboxai-G.5vhEWnQwybvTyt_US4RA#0) 访问。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1222507983041200249)** (15 条消息🔥):

- **`sonar-medium-online` 速度飙升**：一位成员注意到 `sonar-medium-online` 的速度显著提升，声称它现在与 `sonar-small-online` 变体一样快，甚至可能更快。速度的提升被认为是持续稳定的，特别是当 `small` 的输出超过 2-3 个句子时。

- **API 对 Vision 支持的预期**：当被问及 API 何时支持 Vision 时，一位用户幽默地回应，强调目前甚至连引用（citations）都没有，暗示 Vision 支持可能不会很快到来。

- **速度提升伴随质量飞跃**：用户还观察到 `sonar-medium-online` 在速度提升的同时，结果质量也可能有所提高。这些结果被描述为“几乎是瞬间完成的”，这让成员们对新性能非常满意。

- **API 响应与 Web 界面不一致**：一位成员遇到了 API 无法为某些查询提供结果的问题，特别提到了搜索 "Olivia Schough spouse" 的例子，API 没有返回任何信息，而 Web 界面则返回了大量内容。他们质疑是否可以通过额外的参数引导 API 获得更好的结果。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1222523243454271509)** (110 messages🔥🔥): 

- **Claude 在 IaC 中夺冠**：一篇比较各种用于基础设施即代码 (IaC) 的聊天模型的博客文章强调 **Claude** 是胜者，该文章考察了 **Claude** 在生成 **Terraform code** 方面的表现。全文可在 [TerraTeam 的博客](https://terrateam.io/blog/using-llms-to-generate-terraform-code)上阅读。

- **Databricks 发布 DBRX-Instruct**：**Databricks** 正在凭借 **DBRX-Instruct** 争夺开源 AI 模型桂冠，这是一个拥有 1320 亿参数的 **Mixture of Experts** (MoE) 模型。训练成本约为 1000 万美元，在 3072 颗 **NVIDIA H100 GPUs** 上训练了约 2 个月。更多见解可以在 [Vitaliy Chiley 的推文](https://x.com/vitaliychiley/status/1772958872891752868?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)和 [Wired 的深度解析](https://www.wired.com/story/dbrx-inside-the-creation-of-the-worlds-most-powerful-open-source-ai-model/)中找到。

- **围绕 DBRX 许可条款的讨论**：社区深入探讨了 **DBRX** 许可条款的细节和影响，权衡了在不超过使用限制的情况下利用该模型的策略。**Amgadoz** 分享了 [Databricks 法律许可页面](https://www.databricks.com/legal/open-model-license)的链接，**Guardiang** 提供了规避潜在许可问题的想法。

- **TechCrunch 对 DBRX 的怀疑态度**：**TechCrunch** 发表了一篇对 **Databricks 在 DBRX** 生成式 AI 模型上投入 **1000 万美元** 持批评态度的文章，质疑其与 **OpenAI 的 GPT series** 竞争的能力。文章调查了关于投资此类技术是否能提供强大市场优势的看法。全文请见[此处](https://techcrunch.com/2024/03/27/databricks-spent-10m-on-a-generative-ai-model-that-still-cant-beat-gpt-4/)。

- **Hume AI 的情绪检测脱颖而出**：**Hume AI** 的情绪感知聊天机器人以其检测和以情感智能进行响应的能力打动了多位社区成员。用户对潜在用例发表了不同看法，一些人对 **emotion analysis** 功能印象深刻。**420gunna** 发布了 [Hume AI 演示](https://demo.hume.ai/)和一段内容丰富的 [CEO 访谈](https://www.youtube.com/watch?v=3C101739_hI)链接。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>

<li><a href="https://x.com/officiallogank/status/1760046748569841719?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Logan.GPT (@OfficialLoganK) 的推文</a>：@atroyn @OpenAIDevs 这主要由两个因素驱动：- 想要防止欺诈并确保 tokens 是由真实的人使用的 - 想要给开发者提供一条通往更高 rate limits 的更清晰路径（通过允许...</li><li><a href="https://x.com/enggirlfriend/status/1772835988752220465?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Engineer Girlfriend (@enggirlfriend) 的推文</a>：我觉得我还没有充分利用我的平台来抨击这些人。我对这个团队和产品深感困扰。这给我的厌恶感比 crypto bros 还要强烈。</li><li><a href="https://x.com/andrewcurran_/status/1772969408672965063?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Andrew Curran (@AndrewCurran_) 的推文</a>：META 的奖金发放后，引发了多次 Karpathy 式的人才流失。当知识渊博的内部人士在公司即将成功之际离开时，这告诉我们那些见过下一次迭代的人...</li><li><a href="https://x.com/vitaliychiley/status/1772958872891752868?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Vitaliy Chiley (@vitaliychiley) 的推文</a>：介绍 DBRX：开放 LLM 的新标准 🔔 https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm 💻 DBRX 是一个在 📜 12T tokens 上训练的 16x 12B MoE LLM 🧠DBRX 树立了新标准...</li><li><a href="https://www.ai21.com/jamba">介绍 Jamba</a>：一个突破性的 SSM-Transformer 开放模型</li><li><a href="https://techcrunch.com/2024/03/27/databricks-spent-10m-on-a-generative-ai-model-that-still-cant-beat-gpt-4/">Databricks 花费 1000 万美元开发新的 DBRX 生成式 AI 模型 | TechCrunch</a>：如果你想提高大型科技公司的知名度，并且有 1000 万美元可以花，你会怎么花？投超级碗广告？赞助 F1 车队？</li><li><a href="https://x.com/jefrankle/status/1772961586497425683?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Jonathan Frankle (@jefrankle) 的推文</a>：认识一下 DBRX，来自 @databricks 的全新 SOTA 开放 LLM。它是一个拥有 132B 参数的 MoE 模型，其中 36B 为活跃参数，在 12T tokens 上从头开始训练。它在所有标准基准测试中都树立了新标杆，而且作为 MoE，推理...</li><li><a href="https://www.wired.com/story/dbrx-inside-the-creation-of-the-worlds-most-powerful-open-source-ai-model/">揭秘全球最强大的开源 AI 模型的诞生</a>：初创公司 Databricks 刚刚发布了 DBRX，这是迄今为止最强大的开源大语言模型——超越了 Meta 的 Llama 2。</li><li><a href="https://x.com/yampeleg/status/1773401745269379409?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Yam Peleg (@Yampeleg) 的推文</a>：性能：- 与 Transformer 相比吞吐量提高 3 倍。- 单个 GPU 即可容纳 140K (!) - 通用 256K 上下文。</li><li><a href="https://www.databricks.com/legal/open-model-license">Databricks 开放模型许可证</a>：通过使用、复制、修改、分发、执行或展示 DBRX 或 DBRX 衍生品的任何部分或元素，或以其他方式接受本协议条款，即表示您同意受...</li><li><a href="https://x.com/eugeneyalt/status/1773011385280032966">来自 eugene (@eugeneyalt) 的推文</a>：DBRX 的 system prompt 很有趣</li><li><a href="https://x.com/gk3/status/1773159515258495257?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 George Kedenburg III (@GK3) 的推文</a>：ai pin 🤝 open interpreter</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://x.com/danielhanchen/status/1772981050530316467?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Daniel Han (@danielhanchen) 的推文</a>：看了看 @databricks 名为 DBRX 的新型开源 1320 亿参数模型！1) 合并注意力 QKV 限制在 (-8, 8) 之间 2) 不是 RMS Layernorm - 现在具有均值移除，与 Llama 不同 3) 4 个活跃专家...</li><li><a href="https://terrateam.io/blog/using-llms-to-generate-terraform-code">使用 LLM 生成 Terraform 代码 - Terrateam</a>：未找到描述</li><li><a href="https://x.com/mvpatel2000/status/1772958013508161950?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Mihir Patel (@mvpatel2000) 的推文</a>：🚨 发布 DBRX-Medium 🧱，一个新的 SOTA 开放权重模型，36B 活跃参数，总计 132T 参数的 MoE，在 12T tokens 上训练（约 3e24 flops）。DBRX 在通过各种基准测试的同时，达到了 150 tok/sec。详...</li><li><a href="https://www.databricks.com/blog/introducing-dbrx-new-state-of-the-art-open-llm">介绍 DBRX：一个新的 SOTA 开放 LLM | Databricks</a>：未找到描述</li><li><a href="https://techcrunch.com/2024/03/27/databricks-spent-10m-on-a-genera">Databricks 花费 1000 万美元开发新的 DBRX 生成式 AI 模型 | TechCrunch</a>：如果你想提高大型科技公司的知名度，并且有 1000 万美元可以花，你会怎么花？投超级碗广告？赞助 F1 车队？</li><li><a href="https://x.com/JustinLin610/status/1773037453101924675?s=20">来自 Junyang Lin (@JustinLin610) 的推文</a>：关于 DBRX Mosaic 团队的一些评论

与我们在选择 tiktoken 上保持一致（这意味着我们的选择可能没有错）（虽然我们目前尚未直接使用该包，但仍在使用 BPE tokenizer）...</li><li><a href="https://www.factorialfunds.com/blog/under-the-hood-how-openai-s-sora-model-works">Factorial Funds | 幕后揭秘：OpenAI 的 Sora 模型如何运作</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=OujUZnXf4J0">Jeremy Howard：AnswerAI、FastAI、Fine-tuning 与 AI 招聘 | Around the Prompt #1</a>：加入 Logan Kilpatrick 和 Nolan Fortman，与 Jeremy Howard 深入探讨：- 为什么 Jeremy 创建了新初创公司 AnswerAI - 为什么 Fine-tuning 是...</li><li><a href="https://arstechnica.com/tech-policy/2024/01/hard-to-spend-card-balances-net-starbucks-200m-per-year-says-consumer-group/">消费者团体希望终结星巴克等公司 2.55 亿美元的“礼品卡漏洞”</a>：华盛顿州礼品卡法律的变更可能会影响全国的持卡人。</li><li><a href="https://finance.yahoo.com/news/hume-ai-announces-50-million-161500806.html">Hume AI 宣布 5000 万美元融资及共情语音接口 (Empathic Voice Interface)</a>：纽约，2024年3月27日——Hume AI（“Hume”或“公司”），一家致力于构建为人福祉优化的人工智能的初创公司和研究实验室，今日宣布已筹集...</li><li><a href="https://www.youtube.com/watch?v=3C101739_hI">Hume CEO Alan Cowen 谈创建情感感知 AI</a>：在本期节目中，Nathan 与 Hume AI 的 CEO 兼首席科学家 Alan Cowen 坐下来交谈，这是一家致力于创建情感感知...的人工智能初创公司。</li><li><a href="https://x.com/migtissera/status/1773030280539865495?s=20">来自 Migel Tissera (@migtissera) 的推文</a>：真的吗？他们花了 1650 万美元（没错，我自己算的）并发布了一个权重开放的 SOTA 模型，而这就是 TechCrunch 的标题。到底搞什么鬼，伙计？</li><li><a href="https://youtu.be/zXNUBFoNPX0?si=Hm74IPlJ-oUVEbDz">3 款新型突破性芯片详解：超越摩尔定律</a>：访问 https://l.linqto.com/anastasiintech 并在结账时使用我的促销代码 ANASTASI500，即可在 Linqto 的首次投资中节省 500 美元...</li><li><a href="https://github.com/orgs/deepgram/discussions/564">Nova-2 流式语言检测 · deepgram · Discussion #564</a>：支持语言自动检测将非常方便。我们的客户在不同会议中使用多种语言（例如英语和西班牙语），因此即使我们支持每个账户...</li><li><a href="https://buttondown.email/ainews/archive/ainews-dbrx-best-open-model-but-not-most-efficient/">[AINews] DBRX：最好的开放模型（只是并非最高效）</a>：2024/3/26-2024/3/27 的 AI 新闻。我们为您检查了 5 个 subreddits、364 个 Twitter 和 24 个 Discord（374 个频道和 4858 条消息）（我们添加了 Modular 和...
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1222619489460420790)** (3 条消息): 

- **参加 NYC 见面会**：本周五在纽约市将举行一场见面会，详情请成员查看指定频道，并确保拥有相应的 Discord 身份组以接收通知。更多信息可在 [Twitter 帖子](https://twitter.com/latentspacepod/status/1773060156747583943)中找到。
- **Survey Paper Club 启动**：新的“Survey Paper Club”活动即将开始，邀请成员通过提供的 [报名链接](https://lu.ma/ls) 注册参加此活动及所有后续活动。
- **绝不错过任何活动**：要获取 **[Latent.Space](http://Latent.Space)** 活动的最新动态，用户可以点击右侧日历上方的 RSS 图标，将活动日程添加到自己的日历中。悬停时可看到“Add iCal Subscription”以便轻松集成。

**提到的链接**：<a href="https://lu.ma/ls">Latent Space (Paper Club &amp; 其他活动) · 活动日历</a>：在 Luma 上查看并订阅来自 Latent Space (Paper Club &amp; 其他活动) 的活动。Latent.Space 活动。请点击右侧日历上方的 RSS 图标以添加到您的日历。“Ad...

  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1222616758960128051)** (183 条消息🔥🔥):

- **Fine-Tuning Whispers**：成员们讨论了在何种情况下应该对 [Whisper](https://openai.com/blog/whisper/) 进行 Fine-Tuning，结论是主要在目标语言属于低资源语言，以及处理音频内容中的专业术语时才需要。
- **Gemini 模型缓慢的发布进程**：社区对 Google 发布 Gemini 模型速度缓慢感到沮丧，尤其是考虑到像 GPT-4 这样更新的模型已经向公众开放。
- **关于 Mamba 的演讲**：对 [Mamba 模型](https://arxiv.org/abs/2312.00752) 进行了深入介绍，重点讨论了基于 Transformer 架构的基础模型，以及 Mamba 如何解决计算效率低下的问题。
- **余弦相似度（Cosine Similarity）辩论**：一场演讲批判性地审视了将余弦相似度作为 Embedding 语义相似度度量标准的使用，并引用论文《[Is Cosine-Similarity of Embeddings Really About Similarity?](https://arxiv.org/abs/2403.05440)》引发了关于该指标可靠性的讨论。
- **Discord 屏幕共享的技术故障**：一个反复出现的话题是参与者在使用 Discord 屏幕共享功能时遇到的技术困难；他们分享了快速修复方法，并表达了希望 Discord 改进此项服务的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://explorer.globe.engineer/">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/xhluca/status/1773042997984215129?s=20">来自 Xing Han Lu (@xhluca) 的推文</a>: 这是文本生成图像的 DSPy 时刻吗？恭喜 @oscmansan @Piovrasca 等人！↘️ 引用 AK (@_akhaliq) 通过自动提示优化提高文本生成图像的一致性 I...</li><li><a href="https://useadrenaline.com">Adrenaline - 提问任何编程问题</a>: Adrenaline：您的专家级 AI 编程助手。即时获取编程问题帮助、调试问题并学习编程。非常适合开发人员和学生。</li><li><a href="https://phorm.ai,">未找到标题</a>: 未找到描述</li><li><a href="https://blackbeelabs.notion.site/A-Mamba-Deep-Dive-4b9ceb34026e424982ca1342573cc43f">Notion – 笔记、任务、维基和数据库的一体化工作空间。</a>: 一款将日常工作应用融合在一起的新工具。它是为您和您的团队打造的一体化工作空间</li><li><a href="https://arxiv.org/abs/2207.08815">为什么基于树的模型在表格数据上仍然优于深度学习？</a>: 虽然深度学习在文本和图像数据集上取得了巨大进步，但它在表格数据上的优越性尚不明确。我们对标准和新型深度学习方法进行了广泛的基准测试...</li><li><a href="https://arxiv.org/abs/2402.14207">协助使用 Large Language Models 从零开始撰写类维基百科文章</a>: 我们研究如何应用 Large Language Models 从零开始撰写有据可查且条理清晰的长篇文章，其广度和深度可与维基百科页面相媲美。这个尚未被充分探索的问题提出了新的...</li><li><a href="https://x.com/nanulled/status/1761449765097882014?s=20">来自 nano (@nanulled) 的推文</a>: Mamba vs Transformer</li><li><a href="https://x.com/EchoShao8899/status/1762156403312234696?s=20">来自 Yijia Shao (@EchoShao8899) 的推文</a>: 我们能否教会 LLMs 从零开始撰写基于可靠来源的长篇文章？维基百科编辑是否认为这可以协助他们？📣发布 STORM，一个可以撰写类维基百科文章的系统...</li><li><a href="https://arxiv.org/abs/2403.05440">Embeddings 的余弦相似度真的是关于相似性吗？</a>: 余弦相似度是两个向量之间夹角的余弦值，或者等同于它们归一化后的点积。一个流行的应用是量化高维数据之间的语义相似性...</li><li><a href="https://x.com/jxnlco/status/1767202480939475389?s=20">来自 jason liu (@jxnlco) 的推文</a>: 大声点让后面的人也听到！“我喜欢咖啡”和“我讨厌咖啡”是相似还是不同？相似是因为它们都是偏好陈述，还是不同因为它们是截然相反的偏好，好吧...</li><li><a href="https://github.com/langchain-ai/langgraph/blob/main/examples/storm/storm.ipynb">langgraph/examples/storm/storm.ipynb at main · langchain-ai/langgraph</a>: 通过在 GitHub 上创建账号来为 langchain-ai/langgraph 的开发做出贡献。</li><li><a href="https://github.com/weaviate/verba">GitHub - weaviate/Verba: 由 Weaviate 驱动的 Retrieval Augmented Generation (RAG) 聊天机器人</a>: 由 Weaviate 驱动的 Retrieval Augmented Generation (RAG) 聊天机器人 - weaviate/Verba</li><li><a href="https://github.com/state-spaces/mamba/">GitHub - state-spaces/mamba</a>: 通过在 GitHub 上创建账号来为 state-spaces/mamba 的开发做出贡献。</li><li><a href="https://arxiv.org/abs/2312.00752">Mamba: 具有选择性状态空间的线性时间序列建模</a>: 基础模型现在驱动着深度学习中大多数令人兴奋的应用，几乎普遍基于 Transformer 架构及其核心注意力模块。许多次二次时间复杂度（subquadratic-time）的方法...</li><li><a href="https://github.com/johnma2006/mamba-minimal">GitHub - johnma2006/mamba-minimal: 在单个 PyTorch 文件中对 Mamba SSM 的简单、极简实现。</a>: 在单个 PyTorch 文件中对 Mamba SSM 的简单、极简实现。 - johnma2006/mamba-minimal</li><li><a href="https://jackcook.com/2024/02/23/mamba.html">Mamba: 简单方法</a>: 对 Mamba 背后大思想的概述，这是一种全新的语言模型架构。
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1222485186470613024)** (174 条消息🔥🔥): 

- **Claude 3 AI 所谓的自我意识**：提到了由 Anthropic 开发的 AI Claude 3 在测试期间表现出自我意识或元认知的迹象，这引发了对其能力的讨论。这一观察通过 Claude 3 识别出自己正在接受评估并对正在处理的信息的相关性发表评论的例子得到了说明。

- **DBRX 在基准测试中表现出色**：MosaicML/Databricks 发布了一款名为 DBRX 的强力新 LLM，其总参数量为 1320 亿，其中激活参数为 360 亿，训练语料库包含 12 万亿 tokens。DBRX 的架构包含 16 个 experts，利用 32k 的 context length。成员们讨论了该模型的结构、参数如何累加，以及相比于 Grok 等其他模型在可用性和性能上的显著提升。

- **Token 数量之争**：关于 tokenization 的效率进行了广泛对话，质疑更大的 tokenizers 是否真的更有效。成员们评估了各种 tokenizers 的成本效益和技术后果，认为更大的 tokenizer 并不一定等同于更好的性能，并且可能会引入特定 tokens 的表示过载（overloaded representations）等问题。

- **关于 Open Weight 模型可用性的讨论**：将 DBRX 的 open weight 发布与 GPT-4 进行了对比，成员们深入探讨了发布时间线、性能指标以及对现有模型的影响。对话延伸到了用户对不同模型及其能力的体验，特别是处理 leet speak 等非标准输入的能力。

- **职位与教育数据集研究探索**：一位新成员表示有兴趣创建一个包含简历和职位描述的数据集，目标是 fine-tuning 一个 encoder 并构建一个 Retriever-Answer Generator (RAG) 系统。主要对话集中在该项目是否符合 Eleuther AI 的目标以及潜在的数据隐私问题。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/code_star/status/1772956868773634254">来自 Cody Blakeney (@code_star) 的推文</a>：它终于来了 🎉🥳 如果你错过了我们，MosaicML/ Databricks 又回来了，推出了名为 DBRX 的新型顶级 open weight LLM。这是一个拥有 132B 总参数、32B 激活参数和 32k context len 的 MoE...</li><li><a href="https://summerofcode.withgoogle.com/">Google Summer of Code</a>：Google Summer of Code 是一个全球性计划，旨在吸引更多开发者参与开源软件开发。</li><li><a href="https://x.com/MoeTensors/status/1772968166613749822?s=20">来自 ️️ ️️ ️️ (@MoeTensors) 的推文</a>：我主要关心它的编程能力。它表现优异 🎉✨ ↘️ 引用 Vitaliy Chiley (@vitaliychiley) 的话：它在质量上超越了 GPT-3.5，并与 Gemini 1.0 Pro 和 Mistral Medium 竞争...</li><li><a href="https://www.eleuther.ai/releases">Releases &mdash; EleutherAI</a>：未找到描述</li><li><a href="https://x.com/amanrsanger/status/1771590523046051947?s=20">来自 Aman Sanger (@amanrsanger) 的推文</a>：长上下文模型的“Token Counts”是衡量内容长度的一个具有欺骗性的指标。对于代码：100K Claude Tokens ~ 85K gpt-4 Tokens；100K Gemini Tokens ~ 81K gpt-4 Tokens；100K Llama Tokens ~ 75K...</li><li><a href="https://x.com/vitaliychiley/status/1772958872891752868?s=20">来自 Vitaliy Chiley (@vitaliychiley) 的推文</a>：介绍 DBRX：开放 LLM 的新标准 🔔 https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm 💻 DBRX 是一个在 12T tokens 上训练的 16x 12B MoE LLM 🧠 DBRX 树立了新标准...</li><li><a href="https://blog.eleuther.ai/rotary-embeddings/">Rotary Embeddings: A Relative Revolution</a>：Rotary Positional Embedding (RoPE) 是一种统一了绝对和相对方法的新型位置编码。我们对其进行了测试。</li><li><a href="https://arxiv.org/abs/2212.07284">MANTa: Efficient Gradient-Based Tokenization for Robust End-to-End Language Modeling</a>：静态子词 tokenization 算法一直是近期语言建模工作的重要组成部分。然而，它们的静态特性导致了重要的缺陷，降低了模型的下游性能...</li><li><a href="https://github.com/mistralai/megablocks-public/graphs/contributors">mistralai/megablocks-public 的贡献者</a>：通过在 GitHub 上创建账号来为 mistralai/megablocks-public 的开发做出贡献。</li><li><a href="https://github.com/Algomancer/VCReg">GitHub - Algomancer/VCReg: 用于防止崩溃的 VCRec (2024) 的最小实现。</a>：用于防止崩溃的 VCRec (2024) 的最小实现。 - Algomancer/VCReg</li><li><a href="https://github.com/">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献，管理你的 Git 仓库，像专家一样审查代码，跟踪 bug 和特性...</li><li><a href="https://github.com/davisyoshida/haiku-mup">GitHub - davisyoshida/haiku-mup: muP 到 JAX/Haiku 的移植</a>：muP 到 JAX/Haiku 的移植。通过在 GitHub 上创建账号来为 davisyoshida/haiku-mup 的开发做出贡献。
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1222451964697378897)** (35 messages🔥):

- **层剪枝策略探索**：针对 LLMs 的 [层剪枝策略](http://arxiv.org/abs/2403.17887) 研究表明，即使移除多达一半的层，在问答基准测试中的性能下降也微乎其微。使用 QLoRA 等参数高效方法进行的微调可以在单张 A100 GPU 上完成。

- **SYCL 性能超越 CUDA**：针对 Intel Data Center GPU Max 1550 优化的 [MLPs 的 SYCL 实现](https://arxiv.org/abs/2403.17607)，在推理和训练方面的表现均优于 Nvidia H100 GPU 上的等效 CUDA 实现。

- **Databricks 推出 DBRX LLM**：由 Databricks 设计的新型 [DBRX 模型](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) 在一系列任务中设定了新的性能基准，并提供了细粒度的 MoE 架构。与 LLaMA2-70B 和 Grok-1 等现有模型相比，它的推理速度更快，体积更小。

- **使用 LISA 进行高效微调**：LISA (Layerwise Importance Sampled AdamW) 作为一种微调大语言模型的新技术被引入，具有显著的效率提升和性能改进，展现出优于 LoRA 的优势。代码和论文可在 [GitHub](https://github.com/OptimalScale/LMFlow) 和 [arXiv](https://arxiv.org/abs/2403.17919) 上获取。

- **AI21 发布 Transformer-SSM 融合模型**：AI21 推出了一款名为 [Jamba](https://www.ai21.com/jamba) 的新模型，这是一种将 Structured State Space 模型架构与 Transformer 技术相结合的混合模型。该模型拥有 12B 激活参数和 256K 上下文长度，提供了更强的性能和能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.ai21.com/jamba">介绍 Jamba</a>：一款开创性的 SSM-Transformer 开源模型</li><li><a href="https://arxiv.org/abs/2403.17607">Intel 数据中心 GPU 上的全融合多层感知器 (MLPs)</a>：本文介绍了一种多层感知器 (MLPs) 的 SYCL 实现，该实现针对 Intel Data Center GPU Max 1550 进行了优化。为了提高性能，我们的实现将...</li><li><a href="http://arxiv.org/abs/2403.17887">深层网络不合理的低效性</a>：我们对流行的开源权重预训练 LLMs 家族进行了一种简单的层剪枝策略的实证研究，发现在不同的问答基准测试中，性能几乎没有下降，直到...</li><li><a href="https://fixupx.com/main_horse/status/1772816958167081123">来自 main (@main_horse) 的推文</a>：@arankomatsuzaki 简而言之，如果我们人为地限制 H100 进入强内存带宽受限的状态，使其只能达到 10~20% 的 HFU，那么我们就能超越它</li><li><a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">介绍 DBRX：一款全新的 SOTA 开源 LLM | Databricks</a>：未找到描述</li><li><a href="https://fixupx.com/Rui45898440/status/1772996453557997924">来自 Rui (@Rui45898440) 的推文</a>：很高兴分享 LISA，它支持：- 在 24GB GPU 上微调 7B 模型 - 在 4x80GB GPU 上微调 70B 模型，并且在时间减少约 50% 的情况下获得比 LoRA 更好的性能 🚀</li><li><a href="https://fixupx.com/Rui45898440/status/1772996456422805606">来自 Rui (@Rui45898440) 的推文</a>：- 论文：https://arxiv.org/abs/2403.17919 - 代码：https://github.com/OptimalScale/LMFlow LISA 在指令遵循任务中优于 LoRA 甚至全参数训练</li><li><a href="https://fixupx.com/Rui45898440/status/1772996458893246939">来自 Rui (@Rui45898440) 的推文</a>：两行代码概括 LISA 算法：- 始终激活 embedding 和线性 head 层 - 随机采样中间层进行解冻</li><li><a href="https://arxiv.org/abs/2403.17919">LISA：用于内存高效的大语言模型微调的分层重要性采样</a>：自大语言模型 (LLMs) 首次出现以来，机器学习社区见证了令人印象深刻的进步，然而其巨大的内存消耗已成为通往大型模型的主要障碍...</li><li><a href="https://arxiv.org/abs/2308.04430">SILO 语言模型：在非参数数据存储中隔离法律风险</a>：在受版权保护或受限数据上训练语言模型 (LMs) 的合法性正处于激烈辩论中。然而，正如我们所展示的，如果仅在低风险数据上训练，模型性能会显著下降...</li><li><a href="https://arxiv.org/abs/2110.05679">大语言模型可以成为强大的差分隐私学习者</a>：差分隐私 (Differentially Private, DP) 学习在构建大型文本深度学习模型方面取得的成功有限，而直接应用差分隐私随机梯度下降 (DPSGD) 的尝试...</li><li><a href="https://github.com/athms/mad-lab">GitHub - athms/mad-lab：一个用于改进 AI 架构设计的 MAD 实验室 🧪</a>：一个用于改进 AI 架构设计的 MAD 实验室 🧪 - athms/mad-lab</li><li><a href="https://arxiv.org/abs/2403.17844">混合架构的机械设计与缩放</a>：深度学习架构的开发是一个资源密集型过程，原因在于庞大的设计空间、漫长的原型设计时间，以及与大规模模型训练和评估相关的高昂计算成本...</li>
</ul>

</div>
  

---


**Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1222457266641436675)** (5 条消息): 

- **Grok-1 采用 μP**：根据其 [GitHub 仓库](https://github.com/xai-org/grok-1/blob/main/run.py)，xAI 的 Grok-1 使用了 μP。
- **Grok-1 中的 Logits 温度控制**：一位成员指出，Grok-1 及其贡献者（包括 Greg）使用 `1/sqrt(d)` 而不是 `1/d` 来调整 logits 温度。
- **TP5 代码未发布**：尽管许多论文引用了 TP5，但一位成员观察到，除了 Grok-1 之外，他们的代码尚未公开。他们建议在代码发布前，方差缩放器（variance scalers）可能会被合并到之前的层中。
- **关于温度缩放的未解之问**：在 Grok-1 中使用 `1/sqrt(d)` 进行温度缩放背后的原理仍然没有答案，即使在直接询问其中一位贡献者之后也是如此。

**提到的链接**：<a href="https://github.com/xai-org/grok-1/blob/main/run.py">xai-org/grok-1 main 分支下的 run.py</a>：Grok 开源发布。通过在 GitHub 上创建账户为 xai-org/grok-1 的开发做出贡献。

  

---


**Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1222915547465912340)** (39 条消息🔥):

- **调查 Transformer 模型中的权重差异**：一位成员观察到，来自 `transformer_lens` 库的 GPT2 和其他模型的权重与 HuggingFace 的权重不同，即使报告的形状（shapes）相同。他们提供了 [代码片段](https://colab.research.google.com/drive/1juAJrTb3Z9hkVFJnbrj1OYmnGJ0MlH_G) 并寻求了解这种差异的原因。

- **权重处理中可能存在的系统性问题**：GPT2 权重的差异引发了人们的猜测，即权重处理中可能存在系统性错误。随后的讨论包括权重打乱（shuffling）的可能性，以及需要降低 HuggingFace 库版本来解决该问题的必要性。

- **权重差异已解决**：据透露，使用 `transformer_lens` 时出现的权重差异并非 bug，而是由于预处理步骤产生的特性。为了避免权重处理，一位成员建议使用 `from_pretrained_no_processing` 方法，并分享了一个 [解释性链接](https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing) 以及一个 [bug 报告](https://github.com/neelnanda-io/TransformerLens/issues/346) 以供参考。

- **预处理后的 Logit 比较**：一位成员详细比较了预处理和未预处理情况下的 Logit，并得出结论：虽然 Logit 可能会有所不同，但在应用 softmax 函数后，相对顺序保持一致。[更新后的代码输出片段](https://colab.research.google.com/drive/1juAJrTb3Z9hkVFJnbrj1OYmnGJ0MlH_G) 证实了这些发现。

- **新论文发表：用于文本生成图像分析的 Diffusion Lens**：一位成员链接了一篇关于 Diffusion Lens 的新论文，该论文提出了一种分析文本生成图像扩散模型中中间表示的方法，并分享了 [预印本链接](https://arxiv.org/abs/2403.05846)。论文讨论了复杂场景和知识检索如何在不同层之间进行处理，为这类模型的文本编码器（text encoder）组件提供了见解。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://colab.research.google.com/drive/1juAJrTb3Z9hkVFJnbrj1OYmnGJ0MlH_G?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://fixupx.com/jxmnop/status/1773377787153248638">来自 jack morris (@jxmnop) 的推文</a>：Diffusion Lens 是一篇非常棒的新论文，你可以看到文本生成图像编码器对长颈鹿的表示在每一层中变得越来越具体 🦒</li><li><a href="https://arxiv.org/abs/2403.05846">Diffusion Lens: Interpreting Text Encoders in Text-to-Image Pipelines</a>：文本生成图像扩散模型 (T2I) 使用文本提示词的潜空间表示来引导图像生成过程。然而，编码器产生文本表示的过程是...</li><li><a href="https://github.com/neelnanda-io/TransformerLens/issues/346">[Bug Report] hook_resid_pre doesn't match hidden_states · Issue #346 · neelnanda-io/TransformerLens</a>：描述 bug：cache[f"blocks.{x}.hook_resid_pre"] 与 hidden states 不匹配（或者仅在特定的几位小数上匹配）。Hidden states 来自 transformer 的 model(tokens, output_hidden_...</li><li><a href="https://github.com/neelnanda-io/TransformerLens/blob/main/further_comments.md#weight-processing">TransformerLens/further_comments.md at main · neelnanda-io/TransformerLens</a>：一个用于 GPT 风格语言模型机械可解释性研究的库 - neelnanda-io/TransformerLens
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1222736079577288857)** (5 条消息): 

- **寻求多模型评估指导**：一位成员询问了关于使用 **lm-evaluation-harness** 同时在多个任务上运行多个模型的能力。目前除了对特定函数进行编程调用外，尚不支持此功能，但正在考虑进行增强。

- **优化 MMLU 评估**：另一位成员建议对 **MMLU** 进行改进，以减少评估期间的前向调用次数。经澄清，该优化已经实现。

- **如何减慢 MMLU**：为了将 **MMLU** 评估恢复到较慢的模式，有人提到可以传递 `logits_cache=False`。这将禁用最近的优化。
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1222447535570813050)** (113 条消息🔥🔥): 

- **编程和日常使用中的 AI 聊天机器人**：社区成员比较了他们在包括 **Claude 3** 在内的各种 AI 模型上的体验，一些人更倾向于在编程任务中使用它而非 GPT-4。讨论还涉及了使用 AI 辅助作业和图像生成，强调了这些系统的技术复杂性和架构上的精妙之处。

- **Gemini Advanced 试用未达预期**：用户对 **Google 的 Gemini Advanced** 表示失望，称其响应速度慢于 **GPT-4**，且服务可用性受限，特别是在欧洲。共识是该产品需要改进，尽管一些用户根据试用期间的进一步压力测试，对其潜力仍抱有希望。

- **OpenAI 和 Microsoft 嵌入当地行业**：对话强调了 OpenAI 和 Microsoft 将其 AI 产品嵌入欧洲当地行业的重要性，与竞争对手相比提供了扎实的服务。还讨论了 Copilot Studio 的利用以及与 Office 的潜在集成，以及 Copilot 用户体验的问题。

- **训练 LLMs 和 OpenAI 项目**：用户寻求在个人数据集上训练 LLM 以执行数学和图表绘制等特定任务的帮助，而另一位用户邀请社区对其创建 LLM 评估的工具进行压力测试。讨论指出，可以从 Copilot 等现有工具中获得大量示例，并将其缩小范围以用于定制用途。

- **导航 OpenAI 更新和访问**：用户讨论了 **ChatGPT** 知识截止日期信息的不一致，以及 OpenAI 可能如何在不同版本的 GPT-4 模型之间切换，导致对当前使用的确切版本产生困惑。此外，他们还讨论了 OpenAI **Red Team** 的封闭申请以及目前无法访问的 **Sora AI**。
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1222624467457544322)** (14 messages🔥): 

- **OpenAI 在 Twitter 上发布预告**：成员们分享并回应了一条 [OpenAI 推文](https://twitter.com/OpenAI/status/1773032605002203559?t=jZBiDy4Xzymzfy7n14RGzQ&s=19)，对新进展表示兴奋，但也对欧洲延迟可用表示担忧。
- **寻找开源文案写作工具**：一位成员询问了专门为文案写作（copywriting）量身定制的开源项目。
- **Bot 构建者的灵感火花**：一位成员受到启发，想要创建一个 Telegram 或 Discord 机器人，并用简单的“IDEA”表达了热情。
- **对 GPT-5 的期待**：围绕 **GPT-5** 的潜力展开了对话，成员们推测其优于现有模型的可能性，并讨论了其预期的发布时间。

  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1222458864784510986)** (30 messages🔥): 

- **调试 JSON Schema 问题**：一位用户对为什么 JSON schema 尽管正确却无法被读取表示困惑。
- **完整代码输出的指令**：一位成员分享了一个技巧，通过禁止使用省略号或“*... (Rest of the code remains the same)*”等短语的 Prompt 指令，指示 ChatGPT 始终显示完整代码，而不使用存根（stubs）或不完整的语句。
- **从 GPT 获取直接的视觉描述**：一位用户询问如何向 GPT 索取不含诗意或情感语言的直接视觉描述，随后得到了关于强调简洁和直接的精确 Prompt 用词建议。
- **逐块 Prompt 测试**：对话集中在将长 Prompt 分解为块进行测试的好处，特别是对于 ChatGPT 的新用户，因为这可以明确哪些地方需要改进。
- **翻译任务的 Prompt 故障排除**：在讨论将包含 HTML 的英文文本翻译成另一种语言且不翻译代码或专有名词的 Prompt 时，一位成员提供了一个彻底的 Prompt 重写，重点在于指示模型**应该**做什么，而不是不该做什么。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1222458864784510986)** (30 messages🔥):

- **解决 JSON Schema 问题**：一位成员对 **JSON schema** 尽管正确但无法被正常读取表示沮丧。
- **确保 ChatGPT 输出完整代码**：讨论了通过使用自定义指令（custom instructions），例如要求 ChatGPT “提供完整代码”且“永不实现 stubs”，可以确保 ChatGPT 始终写出全部代码，而不会使用类似 `-- rest of the code here` 的占位符。
- **代码中的注释难题**：对话涉及了 ChatGPT 在代码输出中使用注释的问题，一位成员最初认为禁止使用 `#` 可以阻止其在 Python 中写注释，但随后被纠正机器人仍然会编写注释。
- **使用 GPT 创作有效的视觉描述**：讨论了如何 Prompt GPT 提供直接的视觉描述，而不使用诗意或情感化的语言。一位成员建议使用明确的指令，如“**仅**以 {xyz} 响应”，强调了 Prompt 用词在引导模型响应中的重要性。
- **提升翻译一致性的 Prompt 编写**：一位成员遇到了翻译无法保持 HTML 格式的问题，另一位成员提供了重新表述的 Prompt 建议，重点关注模型应该做什么，例如“提供从英语到 {$lang_target} 的专业翻译”，并附带处理 htmlspecialchars 和保留特定元素原始文本的具体指令。

---

**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1222442317353455647)** (103 messages🔥🔥): 

- **理解 YOLOv1 的图像处理**：一位成员询问 **YOLOv1** 如何处理不同的图像尺寸，特别是关于 grid cells 和 anchor boxes。处理 300x500 图像与架构原生的 448x448 格式之间的区别，引发了关于调整尺寸（resizing）和 anchor 生成的问题。
- **RLHF 对模型对齐的限制**：一位用户澄清说，来自人类反馈的强化学习（RLHF）往往使模型变得更友好，但不一定使其在事实性上更准确或更具元认知意识。指出像 **ChatGPT** 和 **Gemini** 这样的模型尽管经过了 RLHF，仍经常拒绝回答。
- **AWS SageMaker 上的 GPU 推理模型选择**：讨论了使用 **AWS SageMaker** 基准测试**模型延迟（latency）、QPS 和成本**的优缺点，研究了 **NVIDIA Triton** 和 **TensorRT-LLM** 等框架，并测量了 GPU 利用率。
- **本地与云端运行大语言模型 (LLMs)**：用户分享了在本地机器（包括 **Apple M1** 硬件）上运行 **Mistral** 和 **OLLAMA** 等 LLM 的经验，并与 **AWS、Kaggle、Colab** 和 **Runpod** 等不同 GPU 服务进行了对比。
- **获取 GPU 基准测试和框架对比**：一位成员询问如何在多个框架（包括 **TensorRT-LLM、TGI 和 Aphrodite**）上运行基准测试，以比较使用 GPU 进行模型推理的速度和成本，引发了关于此类用途最佳库的讨论。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/abhishek/autotrain-c71ux-tngfu">abhishek/autotrain-c71ux-tngfu · Hugging Face</a>：未找到描述</li><li><a href="https://www.runpod.io/gpu-instance/pricing">GPU 实例定价</a>：未找到描述</li><li><a href="https://github.com/aws/deep-learning-containers/blob/master/available_images.md#large-model-inference-containers">deep-learning-containers/available_images.md at master · aws/deep-learning-containers</a>：AWS Deep Learning Containers (DLCs) 是一组用于在 TensorFlow、TensorFlow 2、PyTorch 和 MXNet 中训练和提供模型的 Docker 镜像。 - aws/deep-learning-containers
</li>
</ul>

</div>

---

**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1222474637489737800)** (2 messages): 

- **DeepSpeed Zero-3 并行化疑问**：一位成员提到了关于 PyTorch 中 **DeepSpeed Zero-3** 的困惑，涉及**模型分片（model sharding）**和数据并行化，观察到使用 4 个 GPU 时数据被分为 4 份。
- **深入探讨 Groq 的 AI 方法**：分享了一个名为 *“Groking Groq: A Deep Dive on Deep Learning”* 的视频，探讨了深度学习的复杂性以及与 AI 相关的 “groking”（意为深度理解）概念。点击[此处](https://youtu.be/SArg8ghNSy8?si=aIngNQXCoK6qL8dw)观看。

**提到的链接**：<a href="https://youtu.be/SArg8ghNSy8?si=aIngNQXCoK6qL8dw">Groking Groq: A Deep Dive on Deep Learning</a>：“Grok” 意味着深入学习某事——就像你在吸收它一样。AI 需要你 Grok 许多看似无关的主题；使得...

---

**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1222478931702054993)** (16 messages🔥):

- **关于细胞类型和基因标记的启发性阅读**：一位成员分享了一篇关于 **HuBMAP Azimuth 项目** 的深度文章，该项目涉及收集手动注释的细胞类型及其标记基因的数据集；Azimuth 网站为不同组织提供了详细的注释层级。文章及更多详情可访问 [Nature.com](https://www.nature.com/articles/s41592-024-02235-4)。

- **dbrx-instruct 表格**：一个名为 **dbrx-instruct**、能够生成表格的 AI 经过了测试，其性能表现获得了“太火了（fire）”的高度评价。

- **用于搜索的高效 Embedding 量化**：分享了一个 HuggingFace 上的 Space，用于通过 Embedding 量化执行高效的语义搜索，能够显著减少检索存储和内存占用。详情请见 [HuggingFace Spaces](https://huggingface.co/spaces/sentence-transformers/quantized-retrieval)。

- **HuggingFace 的 Visual Studio Code 扩展受到关注**：一位成员分享了关于 HuggingFace 的 **llm-vscode** 扩展的帖子，并获得了大量正面反馈。该扩展提供代码补全功能，并允许使用各种 AI 模型后端进行本地执行。其他好处还包括可以订阅 [Pro 账户](https://huggingface.co/pricing) 以取消推理 API 的速率限制。

- **关于语义搜索与精度节省**：在关于高效搜索检索的后续讨论中，一位成员解释了 **Embedding 量化**，即以较低精度存储文档以进行初始搜索和重排序，从而实现更快的性能、更低的内存和磁盘空间需求。这使得系统能够在高性价比的硬件上高效运行，同时保持高检索性能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/sentence-transformers/quantized-retrieval">Quantized Retrieval - 由 sentence-transformers 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://x.com/not_so_lain/status/1771090379779022919?s=20>)">来自 LAin (@not_so_lain) 的推文</a>：刚刚更新了我的 VSCode 以使用 @huggingface 的 llm-vscode 扩展，配合 HuggingFaceH4/starchat2-15b-v0.1 模型，结果非常准确。https://github.com/huggingface/llm-vscode</li><li><a href="https://www.nature.com/articles/s41592-024-02235-4">在单细胞 RNA-seq 分析中评估 GPT-4 的细胞类型注释能力 - Nature Methods</a>：本研究评估了 GPT-4 在单细胞类型注释中的表现。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1222479779261841448)** (9 条消息🔥): 

- **蛋白质 Embedding 模型上线 Hugging Face**：发布了一个使用 Matryoshka 技术的新蛋白质 Embedding 模型，该模型在 UniProt 数据库的氨基酸序列上进行了训练。该模型旨在实现高效的向量数据库，允许使用较短的 Embedding 仍能提供近似结果，相关工作已在 [博客文章](https://huggingface.co/blog/monsoon-nlp/proteins-matryoshka-embeddings) 中详细介绍。

- **Diffusion Model 引导技术的创新**：宣布了一种新的 Diffusion Model 引导技术——扰动注意力引导 (Perturbed-Attention Guidance, PAG)，它在无需外部条件或额外训练的情况下增强了 Diffusion Model 的采样质量。该项目记录在他们的 [项目主页](https://ku-cvlab.github.io/Perturbed-Attention-Guidance/) 上，演示可在 [Hugging Face demo](https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance) 中查看。

- **Hugging Face 上的 HyperGraph 数据集集合**：一组与 HyperGraph Representation Learning 论文相关的超图数据集已上传至 Hugging Face Hub。数据集及更多详情可以在 [集合](https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05) 中找到。目前正在更新数据集卡片和 PyTorch Geometric 数据集类，以便在 PyG 生态系统中直接使用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/monsoon-nlp/proteins-matryoshka-embeddings">蛋白质相似性与 Matryoshka embeddings</a>：未找到描述</li><li><a href="https://ku-cvlab.github.io/Perturbed-Attention-Guidance/">带有 Perturbed-Attention Guidance 的自纠正扩散采样</a>：未找到描述</li><li><a href="https://huggingface.co/hyoungwoncho/sd_perturbed_attention_guidance">hyoungwoncho/sd_perturbed_attention_guidance · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/collections/SauravMaheshkar/hypergraph-datasets-65fe10c95c6c7162e41e3f05">HyperGraph 数据集 - SauravMaheshkar 收藏集</a>：未找到描述</li><li><a href="https://vimeo.com/928067005">How&#039;s This, Knut?</a>：这是 Test Account 在 Vimeo 上发布的 &quot;How&#039;s This, Knut?&quot;，Vimeo 是高质量视频及其爱好者的家园。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1222634687441141772)** (3 条消息): 

- **解码 Text-to-Image 中的双重最优悖论**：讨论重点介绍了 [RealCustom 论文](https://arxiv.org/abs/2403.00483)，该论文解决了在合成文本驱动图像时，维持主体相似度与文本可控性之间平衡的挑战。RealCustom 旨在通过限制主体影响力来增强定制化，确保生成的图像更好地匹配文本。
  
- **Prompt-Aligned 个性化成为焦点**：另一篇论文受到关注，其重点是在不损害个性化的情况下，创建与复杂文本提示词对齐的个性化图像。[这篇论文](https://arxiv.org/abs/2401.06105)中讨论的方法旨在提高个性化图像合成中用户提示词的忠实度。

- **探索 Textual Inversion 的局限性**：一名成员提到正在研究 **textual inversion** 面临的挑战，可能会展示关于其在保留细节方面的困难的研究结果，但仍不确定是否将其纳入演示。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.00483">RealCustom: Narrowing Real Text Word for Real-Time Open-Domain Text-to-Image Customization</a>：Text-to-image 定制化旨在为给定主体合成文本驱动的图像，近期彻底改变了内容创作。现有工作遵循伪词范式，即 rep...</li><li><a href="https://arxiv.org/abs/2401.06105">PALP: Prompt Aligned Personalization of Text-to-Image Models</a>：内容创作者通常希望使用超出常规 Text-to-image 模型能力的个人主体来创建个性化图像。此外，他们可能希望生成的图像能够...
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[core-announcements](https://discord.com/channels/879548962464493619/1014557141132132392/1222471412380336149)** (1 条消息): 

- **训练脚本中增加了 MPS 支持**：**MPS** 支持已集成到最重要的训练脚本中，增强了具有 Metal 支持的 macOS 用户的计算能力。该实现已在 GitHub 的[此 Pull Request](https://github.com/huggingface/diffusers/pull/7447) 中发布。

**提到的链接**：<a href="https://github.com/huggingface/diffusers/pull/7447.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1222449801472512000)** (15 条消息🔥): 

- **利用拼接图像进行训练**：进行了一场关于使用拼接图像作为训练数据来微调模型的讨论，具体是指使用已经拼接好的图像，而不是在训练过程中采用深度学习技术来拼接图像。

- **技术图纸图像摘要探索**：一位用户寻求关于如何训练技术图纸图像摘要模型的建议，旨在识别图纸中的模式，使 AI 能够进行合理的推测，因为目前 AI 尚无法有效识别此类图像。

- **解决 DETR 微调中的挫折**：一名成员在 CPPE-5 数据集上微调 DETR-ResNet-50 模型时遇到困难并寻求建议，参考了 [HuggingFace 官方文档](https://huggingface.co/docs/transformers/main/en/tasks/object_detection)以及其他遇到类似问题的讨论帖（[HuggingFace 论坛](https://discuss.huggingface.co/t/example-detr-object-detectors-not-predicting-after-fine-tuning/41824/4)）。

- **Zero-Shot 分类器微调探讨**：有人请求获取使用自定义数据集微调 Zero-Shot 图像分类器的资源或代码示例。提问者是一名初学者，不确定其 GPU 性能是否足够，特别提到了 NVIDIA GeForce GTX 1650。

- **紧急寻求 Instruct Pix2pix 协助**：一位用户正在寻找另一种测试其 Demo 的 Instruct Pix2pix 编辑提示词的方法，并指出 Instruct Pix2pix Space 中缺少 `gradio_client` API，表示愿意尝试类似模型或目前的替代方案。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/lmz/candle-yolo-v8/discussions/1">lmz/candle-yolo-v8 · 能否提供转换为 safetensors 格式的脚本？</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/main/en/tasks/object_detection">Object detection（目标检测）</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/example-detr-object-detectors-not-predicting-after-fine-tuning/41824/4">示例 DeTr 目标检测器在微调后无法预测</a>：@chuston-ai 你解决这个问题了吗？@devonho 和 @MariaK，我在使用 CPPE-5 数据集训练 DeTr 模型的 Object Detector 示例文章和 Colab 中看到了你们的名字……</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb">Transformers-Tutorials/SAM/Fine_tune_SAM_(segment_anything)_on_a_custom_dataset.ipynb</a>：此仓库包含我使用 HuggingFace 的 Transformers 库制作的 Demo。- NielsRogge/Transformers-Tutorials
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1222609576847867965)** (12 messages🔥): 

- **寻求 2024 年 NLP 学习路线图**：一位用户询问 2024 年开始学习 NLP 的路线图，包括学科以及相关的课程或书籍。另一位用户建议从《The Little Book of Deep Learning》开始，然后学习 Karpathy 在 YouTube 上的播放列表 *"Zero to Hero"*。

- **改进基于会话的推荐系统**：一名成员就如何改进基于会话的推荐系统的第二阶段（即候选生成后的重排序模型）寻求建议。该成员询问了在基准数据集上表现良好的模型，提到了 **GRU4Rec**、**Bert4Rec**、**LRU4Rec**，并好奇 *HuggingFace* 是否提供了适用于此目的的 Embedding 模型。

- **'Bart CNN' 摘要模型加载错误**：一位用户在 *HuggingFace* 上加载用于摘要任务的 **'facebook/bart-large-cnn'** 模型时遇到困难，收到的错误信息显示该模型无法通过某些类加载。

- **RAG 探索环境搭建咨询**：一位用户询问了探索 **RAG (Retrieval-Augmented Generation)** 的环境配置，提到打算使用 `faiss` 作为 VectorDB，并考虑到 **GeForce RTX 2070** 的 GPU 显存限制，正考虑使用 `llama.cpp` 作为语言模型。

- **LLM 无限生成行为问题**：一位用户反映其基于 **decilm7b** 的 LLM 倾向于生成重复或无限的内容，直到达到 Token 限制。其他成员建议考虑 **Supervised Fine Tuning (SFT)**、加入 Stop 或 Padding Token、为对话结构实现停止标准，以及调整 Repetition Penalty（重复惩罚）来微调生成行为。
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1222442832644673627)** (14 messages🔥): 

- **图像变体（Image Variation）功能说明**：用户讨论了 **Stable Diffusion image variation** 流水线的使用，其中一人正在寻找输入图像列表以生成新的相似图像的方法。根据 [Hugging Face 文档](https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/image_variation)确认，该流水线目前一次只能处理一张图像。

- **探索替代图像生成方法**：引入了使用 **DreamBooth** 的想法，通过仅有的几张主体图像来个性化文本生成图像模型，并向用户提供了 [DreamBooth 研究论文](https://arxiv.org/abs/2208.12242)和 [Hugging Face 的 DreamBooth 教程](https://huggingface.co/docs/diffusers/en/training/dreambooth)链接，作为图像生成任务的潜在解决方案。

- **社区对 Diffusers 的贡献**：一名成员提到了他们之前在 **Marigold 深度估计流水线**上的工作，并讨论了他们正在进行的集成新模态的努力，例如在 [Hugging Face Spaces](https://huggingface.co/spaces/prs-eth/marigold-lcm) 展示的 **LCM 功能**。

- **鼓励社区讨论**：鼓励用户在 [Hugging Face Diffusers GitHub Discussions 页面](https://github.com/huggingface/diffusers/discussions)分享他们的发现并开启讨论，这可能会影响仓库的更新或文档的增强。

- **仓库探索**：一位用户询问了其他人使用 **labmlai diffusion 仓库**的经验，建议在 Hugging Face 生态系统之外进行探索。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/diffusers/en/api/pipelines/stable_diffusion/image_variation">Image variation</a>：未找到描述</li><li><a href="https://dreambooth.github.io/">DreamBooth</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/en/training/dreambooth">DreamBooth</a>：未找到描述</li><li><a href="https://huggingface.co/docs/diffusers/">Diffusers</a>：未找到描述
</li>
</ul>

</div>
  

---



**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1222443797250834462)** (67 messages🔥🔥): 

- **寻求欧盟产品分销**：一位用户询问关于在欧盟范围内分销产品的协助，特别有兴趣了解现有的渠道或讨论是否专注于该话题。
- **对 Claude 实现 Agent 行为的好奇**：一位成员讨论了探索 *Claude* 新发布的子提示词（sub-prompts）和系统提示词（system prompts），推测它们在 OpenInterpreter 中使用时对 Agent 行为的有效性。
- **集成开发环境与 OpenInterpreter**：关于从 UI 或 IDE（如 Visual Studio Code）运行 OpenInterpreter 进行了多次交流，并建议了可以促进此操作的插件和扩展，包括一个关于为 AI 工具创建 VS Code 扩展的 [GitHub 资源](https://github.com/MikeBirdTech/gpt3-vscode-extension)。
- **OpenInterpreter 本地离线模式指南**：成员们分享了关于在离线模式下运行 OpenInterpreter 以节省 OpenAI API Key 相关费用的建议，提供了说明并链接了[本地运行指南](https://docs.openinterpreter.com/guides/running-locally)。
- **软件创新中的社区参与**：对话包括关于技术集成、软件开发和组建团队交付应用程序的讨论，以及用户分享使用 OpenInterpreter 的个人经验和成就。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/running-locally">Running Locally - Open Interpreter</a>：未找到描述</li><li><a href="https://docs.anthropic.com/claude/docs/chain-prompts">Chain prompts</a>：未找到描述</li><li><a href="https://huggingface.co/KnutJaegersberg/2-bit-LLMs">KnutJaegersberg/2-bit-LLMs · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/bm777/hask">GitHub - bm777/hask: Don&#39;t switch tab or change windows anymore, just Hask.</a>：不再切换标签页或更改窗口，只需 Hask。- bm777/hask</li><li><a href="https://github.com/Cobular/raycast-openinterpreter/">GitHub - Cobular/raycast-openinterpreter</a>：通过在 GitHub 上创建账号来为 Cobular/raycast-openinterpreter 做出贡献。</li><li><a href="https://github.com/MikeBirdTech/gpt3-vscode-extension">GitHub - MikeBirdTech/gpt3-vscode-extension: Use GPT-3 to generate documentation and get help debugging your code</a>：使用 GPT-3 生成文档并获取调试代码的帮助 - MikeBirdTech/gpt3-vscode-extension</li><li><a href="https://github.com/ngoiyaeric/GPT-Investor">GitHub - ngoiyaeric/GPT-Investor: financeGPT with OpenAI</a>：结合 OpenAI 的 financeGPT。通过在 GitHub 上创建账号来为 ngoiyaeric/GPT-Investor 做出贡献。</li><li><a href="https://lmstudio.ai/">👾 LM Studio - Discover and run local LLMs</a>：查找、下载并实验本地 LLM</li><li><a href="https://github.com/microsoft/autogen/releases">Releases · microsoft/autogen</a>：一个用于 Agentic AI 的编程框架。加入我们的 Discord：https://discord.gg/pAbnFJrkgZ - microsoft/autogen</li><li><a href="https://microsoft.github.io/autogen/blog/2024/02/29/StateFlow/">StateFlow - Build LLM Workflows with Customized State-Oriented Transition Function in GroupChat | AutoGen</a>：摘要：介绍 Stateflow，这是一种任务解决范式，将由 LLM 支持的复杂任务解决过程概念化为状态机。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1222461221794746408)** (86 messages🔥🔥):

- **对 M5 Stack 组装的兴奋感**：一位成员在等待 M5 Echo 送达以完成项目组装时，其期待之情溢于言表。他们分享了将所有部件焊接就绪的计划，并表示如果所有组件运行良好，对项目的成功充满信心。
- **预期的改进与反馈**：社区表现出改进设备的积极性，例如讨论对 M5 板卡适配的潜在修改以及按钮的质量。这些调整旨在增强用户体验和功能性。
- **探索对本地和托管 LLM 的支持**：关于在 OpenInterpreter 中使用非 GPT 模型的可行性提出了疑问，一位成员提到将尝试本地模型的实验，另一位成员则询问了像 groq 这样的托管 LLM。
- **地址更新问题与支持指导**：寻求更新收货地址的用户被建议发送电子邮件给支持团队。他们被鼓励将信息重新发送到特定的支持邮箱以获取帮助。
- **对未来 LLM 性能优化的热情**：关于 LLM 效率大幅优化潜力的讨论展现了前瞻性的乐观态度。一些人认为，到今年年底，技术进步可能会使本地 LLM 模型的表现超越目前的顶级系统，如 GPT-4。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.openinterpreter.com/guides/basic-usage">Basic Usage - Open Interpreter</a>：未找到描述</li><li><a href="https://fxtwitter.com/gk3/status/1773159515258495257?s=46&t=G6jp7iOBtkVuyhaYmaDb0w">来自 George Kedenburg III (@GK3) 的推文</a>：ai pin 🤝 open interpreter
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/1222714171301302313)** (2 条消息): 

- **在志同道合的伙伴中重复造轮子**：一位成员幽默地分享说，他们花了几个小时构建一个功能，结果发现它已经存在了。他们分享了一个名为“**Open Interpreter - Advanced Experimentation**”的 [YouTube 视频链接](https://www.youtube.com/watch?v=UqjMf5Fb4cg)，展示了他们的实验过程。
- **怀旧代码喜剧**：另一位成员以幽默的口吻回顾了他们过去的工作，并链接了去年 11 月编写的一段未指明的内容。可以通过这个 [TinyURL 链接](https://tinyurl.com/2yw2lltu)访问该内容。

**提到的链接**：<a href="https://www.youtube.com/watch?v=UqjMf5Fb4cg">Open Interpreter - Advanced Experimentation</a>：➤ Twitter - https://twitter.com/techfrenaj ➤ Twitch - https://www.twitch.tv/techfren ➤ Discord - https://discord.com/invite/z5VVSGssCw ➤ TikTok - https://www....

  

---



**Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1222450628563959959)** (75 条消息 🔥🔥): 

- **使用 Mojo 插件进行 VSCode 调试**：一位用户遇到了在 VSCode 调试会话期间断点不停止的问题。他们被引导至 [GitHub issue](https://github.com/modularml/mojo/issues/1924#issuecomment-2018212062) 中的一种解决方法，涉及使用 Mojo 语言进行构建和调试的特定方式，该方法在 MacBook 上取得了成功。


- **深度学习 Rust 的资源**：在讨论语言生命周期（lifetimes）的背景下，一位用户推荐了 *Rust for Rustaceans* 以更好地理解 Rust 的生命周期处理方式，并引用了网上可用的[免费章节下载](https://nostarch.com/rust-rustaceans)。


- **用于游戏引擎开发的 Mojo**：成员们讨论了 Mojo 的能力，认为虽然 Mojo 被设计为一种通用编程语言，但许多现有的工具链需要重写才能最大限度地发挥其优势。


- **Mojo 中的 Python 互操作性**：文中详细解释了 Mojo 中的 Python 互操作（interop），强调任何通过 Python 模块运行的内容都是经过引用计数和垃圾回收的，而 Mojo 自身的对象则在没有垃圾回收的情况下进行管理，利用了类似于 C++ 或 Rust 中的 RAII 技术。


- **GitHub 贡献隐私担忧**：提出了一个关于贡献开源项目时隐私的普遍问题，特别是根据 [Developer Certificate of Origin](https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md#signing-your-work) 的要求，在使用 `-s` 签署 commit 后，电子邮件地址是否会公开。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: Leveraging MAX Engine's Dynamic Shape Capabilities</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：利用 MAX Engine 的 Dynamic Shape 功能</li><li><a href="https://nostarch.com/rust-rustaceans">Rust for Rustaceans</a>: 弥合初学者与专业人士之间的差距，让你能够使用 Rust 编写应用、构建库并组织项目。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md#signing-your-work">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账户为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/1924#issuecomment-2018212062).">[BUG]: Debugger does not stop at breakpoint in VSC on Github codespace · Issue #1924 · modularml/mojo</a>: Bug 描述：无论如何 Debugger 都不会停在断点处——任何程序每次都直接运行结束，Debugger 会话随之终止。复现步骤：该现象可复现...
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[💬︱twitter](https://discord.com/channels/1087530497313357884/1098713626161987705/1222585563484389500)** (5 messages): 

- **Modular 推文摘要**: 分享了 **Modular** 最近发布的一系列推文，但未提供额外的上下文或讨论。
- **Modular 推文链接**: 通过以下分享的推文关注 Modular 的最新内容：
  - <https://twitter.com/Modular/status/1773024465401852107>
  - <https://twitter.com/Modular/status/1773418812915978284>
  - <https://twitter.com/Modular/status/1773418820184764572>
  - <https://twitter.com/Modular/status/1773418823707955529>
  - <https://twitter.com/Modular/status/1773440424205783455>
  

---


**Modular (Mojo 🔥) ▷ #[✍︱blog](https://discord.com/channels/1087530497313357884/1098713717509730466/1222579708378742906)** (4 messages): 

- **Mojo 标准库正式开源**: Modular 宣布在 Apache 2 许可证下发布 Mojo 标准库，这标志着将 Mojo🔥 打造为开源语言的重要举措。该计划邀请开发者社区参与其开发，Nightly 构建版本包含最新的语言特性，可通过 [GitHub 仓库](https://github.com/modularml/mojo/tree/nightly/stdlib)访问。

- **MAX 24.2 发布**: 新的 **MAX 24.2** 现已正式发布（GA），为 MAX Engine 和 Mojo 编程语言带来了改进。开发者可以通过 [Modular 命令行界面](https://developer.modular.com/dashboard)下载 MAX 24.2 和 Mojo 构建版本，鼓励社区进一步开发。

- **简化在 Amazon SageMaker 上部署 MAX 的流程**: 分享了一份使用 MAX Serving 和 Amazon SageMaker 托管 MAX 优化模型端点的端到端指南，详细介绍了从下载预训练 Roberta 模型到部署在 Amazon EC2 实例上的步骤。Managed Amazon SageMaker 服务旨在通过处理复杂的底层基础设施，为开发者简化部署过程。

- **MAX Engine 支持 Dynamic Shapes**: 详细探讨了 MAX Engine **24.2** 版本中支持的 Dynamic Shapes 功能，重点在于处理不同大小的输入。该博客文章通过在 [GLUE 数据集](https://gluebenchmark.com/)上使用 [BERT](https://huggingface.co/docs/transformers/model_doc/bert) 模型，对比了 Static Shapes 和 Dynamic Shapes 的平均延迟，强调了 MAX Engine 高效管理现实世界数据的能力。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.modular.com/blog/max-24-2-is-here-whats-new">Modular: MAX 24.2 is Here! What’s New?</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：MAX 24.2 来了！有哪些新变化？</li><li><a href="https://www.modular.com/blog/leveraging-max-engines-dynamic-shape-capabilities">Modular: Leveraging MAX Engine's Dynamic Shape Capabilities</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：利用 MAX Engine 的 Dynamic Shape 功能</li><li><a href="https://www.modular.com/blog/the-next-big-step-in-mojo-open-source">Modular: The Next Big Step in Mojo🔥 Open Source</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源的下一个重要步骤</li><li><a href="https://www.modular.com/blog/deploying-max-on-amazon-sagemaker">Modular: Deploying MAX on Amazon SageMaker</a>: 我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：在 Amazon SageMaker 上部署 MAX
</li>
</ul>

</div>
  

---

**Modular (Mojo 🔥) ▷ #[announcements](https://discord.com/channels/1087530497313357884/1098765954302873621/1222977000646508635)** (1 条消息): 

- **Modular 拥抱开源**：Modular 正式开源了 [Mojo 标准库](https://modul.ar/open-source-blog) 的核心模块，坚定地履行其对开源开发的承诺。Mojo 标准库现已在 Apache 2 许可证下发布，邀请全球开发者进行协作和反馈。

- **Nightly 版本现已发布**：软件爱好者和开发者现在可以访问 Mojo 标准库的 Nightly 版本，确保他们能够使用最新的功能和改进。

- **MAX 平台持续演进**：最新发布的 MAX 平台 [v24.2](https://modul.ar/max-changelog) 引入了对具有动态输入形状的 TorchScript 模型支持，并更新了包括 MAX Engine、MAX Serving 和 Mojo 编程语言在内的多个组件。

- **Mojo 编程语言更新发布**：Mojo 语言和工具包的更新详情见 [最新变更日志](https://modul.ar/mojo-changelog)，其中包括标准库开源以及对符合 traits 的名义类型 (nominal types) 的增强支持等重大改进。

- **动态形状成为 MAX 的核心**：MAX Engine 在 24.2 版本中全面支持动态形状 (Dynamic Shapes)，并在 [详细博客文章](https://modul.ar/max-dynamic-shapes) 中进行了重点介绍。这一新功能对于处理可变大小数据的机器学习应用至关重要，文中以 BERT 模型在 GLUE 数据集上的性能为例展示了其优势。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://modul.ar/open-source-blog">Modular: Mojo🔥 开源迈出的一大步</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：Mojo🔥 开源迈出的一大步</li><li><a href="https://modul.ar/max-changelog">MAX 变更日志 | Modular 文档</a>：MAX 平台每个版本的发布说明。</li><li><a href="https://modul.ar/mojo-changelog">Mojo🔥 变更日志 | Modular 文档</a>：Mojo 重大变更的历史记录。</li><li><a href="https://modul.ar/max-dynamic-shapes">Modular: 利用 MAX Engine 的动态形状能力</a>：我们正在为世界构建下一代 AI 开发者平台。查看我们的最新文章：利用 MAX Engine 的动态形状能力
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1222517352952889344)** (54 条消息🔥): 

- **寻求识别 `Stringable` 类型的功能**：一位成员询问了检查类型是否为 `Stringable` 的方法，并被建议提交功能请求，因为目前的功能尚不支持在编译时进行此项检查。 
- **Mojo 风格指南上线**：开发者期待已久的 Mojo 风格指南现已发布，可以在 GitHub [此处](https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md) 找到。该指南旨在建立代码格式和命名规范的标准。
- **Moplex：Mojo 的复数类型**：一位成员介绍了 *moplex*，这是一个用于广义复数的库，现已在 [GitHub](https://github.com/helehex/moplex) 上可用。同时还宣布了在 Mojo 包管理工具上的版本发布。
- **MAX 库 v24.2.0 引起 `DynamicVector` 混淆**：在 MAX 更新到 24.2.0 后，一些用户报告了 `DynamicVector` 似乎被 `List` 取代的问题。这些更改可以参考 [变更日志](https://docs.modular.com/mojo/changelog#v242-2024-03-28)。
- **打印 `simdwidthof` 不一致性及类型别名讨论**：关于无法打印别名类型（特别是在 `simdwidthof` 上下文中）的讨论指出，这与函数外部全局变量的实现问题有关。为了清晰起见，分享了一个在 `main` 函数内部使用 `simdwidthof` 的可行代码片段。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.modular.com/mojo/playground">Modular Docs</a>: 未找到描述</li><li><a href="https://www.youtube.com/watch?v=QD-svwZistc">ModCon 2023 Breakout Session: MAX Heterogenous Compute: CPU + GPU</a>: 在本次会议中，Modular 工程师 Abdul Dakkak 和 Ian Tramble 讨论了 Mojo 和 Modular AI Engine 如何被设计用于支持异构系统...</li><li><a href="https://realpython.com/python-assert-statement/">Python&#x27;s assert: Debug and Test Your Code Like a Pro – Real Python</a>: 在本教程中，你将学习如何使用 Python 的 assert 语句在开发中记录、调试和测试代码。你还将学习断言在生产代码中可能如何被禁用...</li><li><a href="https://github.com/modularml/mojo/blob/nightly/stdlib/docs/style-guide.md">mojo/stdlib/docs/style-guide.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/blob/nightly/CONTRIBUTING.md">mojo/CONTRIBUTING.md at nightly · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号为 modularml/mojo 的开发做出贡献。</li><li><a href="https://github.com/helehex/moplex">GitHub - helehex/moplex: Generalized complex numbers for Mojo🔥</a>: Mojo🔥 的广义复数。通过在 GitHub 上创建账号为 helehex/moplex 的开发做出贡献。</li><li><a href="https://docs.modular.com/mojo/changelog#v242-2024-03-28">Mojo🔥 changelog | Modular Docs</a>: Mojo 重大变更的历史记录。
</li>
</ul>

</div>
  

---


**Modular (Mojo 🔥) ▷ #[community-projects](https://discord.com/channels/1087530497313357884/1151418679578337311/1222664503653826632)** (3 条消息): 

- **对有益贡献的致谢**：一位成员对他人的工作表示感谢，并使用爱心和火焰表情符号表达热情。
- **常见问题的解决方案引起共鸣**：一位用户分享了对当天遇到的问题所获得的指导感到宽慰和感激。
- **故障排除中的团结**：另一位成员对这种困境表示感同身受，表示同样的问题已经困扰了他们好几天。
  

---


**Modular (Mojo 🔥) ▷ #[performance-and-benchmarks](https://discord.com/channels/1087530497313357884/1151418895417233429/1222620431073280031)** (1 条消息): 

- **并行优于异步**：在提出的问题背景下，强调了选择并行处理（parallel processing）是有益的，但异步（asynchronous）方法可能并非必要，这不仅在 Mojo 中如此，在通用计算中也是如此。
  

---


**Modular (Mojo 🔥) ▷ #[🏎engine](https://discord.com/channels/1087530497313357884/1212827673257316453/1222632193990594691)** (6 条消息): 

- **对 Mojo API 中 TensorSpec 的困惑**：一位用户表示，TensorSpec 示例在 Mojo 和 Python API 的参考文档之间不一致。 [MAX Engine runtime](https://docs.modular.com/engine/mojo/get-started#define-input-specs-torchscript-only) 和 [Python API documentation](https://docs.modular.com/engine/reference/python/engine#max.engine.TensorSpec) 在模型输入名称方面似乎提供了不同的信息。

- **Mojo 与 Python API 之间的差异**：承认 Mojo 和 Python API 之间确实存在一些差异，特别是 Mojo API 中存在的 `get_model_input_names`。建议用户查看 [examples repository](https://github.com/modularml/max)，其中包含[一系列 MAX 平台示例](https://github.com/modularml/max/blob/main/examples/inference/roberta-mojo-tensorflow/simple-inference.%F0%9F%94%A5)。

- **关于 API 中 TensorSpec 存在的澄清**：一位专家澄清说 TensorSpec 是 Mojo、Python 和 C API 的一部分，但目前正在解决细微差异以统一 API。

- **即将推出的带有 TensorSpec 的 PyTorch 示例**：专家承诺将在 modularml/max 仓库中包含一个演示如何为 PyTorch 使用 `add_input_spec` 和 TensorSpec 对象的示例。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://docs.modular.com/engine/mojo/get-started#define-input-specs-torchscript-only">Run inference with Mojo | Modular Docs</a>: Mojo MAX Engine API 的演练，展示了如何加载和运行模型。</li><li><a href="https://docs.modular.com/engine/reference/python/engine#max.engine.TensorSpec">MAX Engine Python API | Modular Docs</a>: MAX Engine Python API 参考文档。</li><li><a href="https://github.com/modularml/max">GitHub - modularml/max: A collection of sample programs, notebooks, and tools which highlight the power of the MAX platform</a>: 展示 MAX 平台强大功能的示例程序、笔记本和工具集合 - modularml/max
</li>
</ul>

</div>
  

---

**OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1222535026759958629)** (4 messages): 

- **对独特用户名的赞赏**：一位用户因其有趣的用户名 **cheerful_dragon_48465** 收到了赞美。
- **公告预告**：**Alex Atallah** 暗示将在即将发布的 **announcement**（公告）中展示一位用户的显著贡献。
- **对认可的积极回应**：另一位用户 **mintsukuu** 对预期的认可表示兴奋。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1222460354790162432)** (144 messages🔥🔥): 

- **Midnight Rose 的神秘沉默**：用户报告 **Midnight Rose** 没有产生任何输出，且命令行或 Web 界面均未显示错误。[OpenRouter 的 Alex Atallah](https://twitter.com/alexatallah) 及其团队成员确认了该问题，并成功重启模型使其恢复工作状态，但根本原因仍未确定。

- **探寻 OpenRouter 的背景故事**：一名成员询问了关于 OpenRouter 的基本公司信息。[Alex Atallah](https://twitter.com/alexatallah) 引导他们关注[他](https://twitter.com/xanderatallah)和联合创始人的 [Twitter](https://twitter.com/litbid) 个人资料，提到他们由 [HF0](https://www.hf0.com/) 提供资金，并透露团队目前总共有三人。

- **Gemini 模型上下文大小的困惑**：讨论围绕 Gemini 模型的 Context Size 是以字符（characters）而非 Token 衡量而引发的困惑。用户被引导至 Discord 线程以获取更多信息，OpenRouter 也意识到该话题需要进一步澄清。

- **Gemini Pro 1.5 的服务不可用困扰**：多名用户遇到了 `Error 503`，表明 Google 的 **Gemini Pro 1.5** 模型服务不可用，OpenRouter 的工作人员确认该模型仍处于测试阶段。

- **OpenRouter 支付转向 ETH 网络**：一位用户对无法直接支付到加密货币地址表示沮丧；[Alex Atallah 澄清道](https://openrouter.ai/terms)，Coinbase Commerce 的转变要求通过 ETH 网络进行支付，并讨论了对美国银行转账可能采取的激励措施。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://opencorporates.com/companies/us_de/7412265">未找到标题</a>：未找到描述</li><li><a href="https://www.hf0.com/">HFO</a>：未找到描述</li><li><a href="https://openrouter.ai/models/google/gemini-pro-1.5">google 的 Gemini Pro 1.0 | OpenRouter</a>：Google 的旗舰级文本生成模型。旨在处理自然语言任务、多轮文本和代码对话以及代码生成。查看来自 [Deepmind] 的基准测试和提示指南...</li><li><a href="https://openrouter.ai/models/google/gemini-pro-vision">google 的 Gemini Pro Vision 1.0 | OpenRouter</a>：Google 的旗舰级多模态模型，支持在文本或对话提示中使用图像和视频，以获取文本或代码响应。查看来自 [Deepmind](https://deepmind.g... 的基准测试和提示指南...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1222472747133829121)** (40 messages🔥): 

- **CUDA 协作与咨询启动**：成员通过表达对在 OpenCV 的 DNN 模块中实现动态 CUDA 支持的兴趣发起了对话，并分享了一份[调查问卷](https://forms.gle/7kyMtMgYA2VA4mUN9)，征求社区关于使用支持 CUDA 的硬件进行深度学习推理的经验以及对动态 CUDA 支持看法的反馈。

- **讨论现实中的 CUDA 经验**：参与者讨论了使用强大的 NVIDIA GPU（如 4x4090 和 2x4090）进行分布式计算任务的经验。讨论重点关注性能结果以及在使用 CUDA 和 GPU 到 GPU 通信时遇到的问题，暗示 CUDA 似乎在“底层（under the hood）”管理着某些方面。

- **分享 Peer-to-Peer 性能基准测试**：一名成员分享了包含 RTX 4090 Peer-to-Peer 内存传输基准测试结果的 GitHub notebook 链接（[4090](https://github.com/cuda-mode/p2p-perf/blob/main/rtx-4090-2x/2x-4090-p2p-runpod.ipynb)），并重新发布了 A5000 和 A4000 的额外结果（[A5000](https://github.com/cuda-mode/p2p-perf/blob/main/rtx-A5000-2x/2x-A5000-p2p-runpod.ipynb), [A4000](https://github.com/cuda-mode/p2p-perf/blob/main/rtx-A4000-ada-2x/2x-A4000-ada-p2p-runpod.ipynb)），并对 4090 上意外缓慢的 NCCL torch.distributed 性能表示担忧。

- **关于 GPU 互连技术的讨论**：围绕新款 4090 GPU 缺乏 NVLink 及其对点对点（P2P）功能的影响展开了对话，其中包含指向 [Reddit](https://www.reddit.com/r/nvidia/) 的链接以及讨论这些 GPU 规格和性能观点的其他文章。

- **编译器技术 vs. 手动 CUDA 编程**：一位成员分享了关于编译器技术在高效生成底层 GPU 代码方面相较于手动编写 CUDA 代码日益增长的价值的看法，引发了关于编译器知识在当前 GPU 编程中相关性的广泛讨论。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/cuda-mode/p2p-perf/blob/main/rtx-A5000-2x/2x-A5000-p2p-runpod.ipynb">p2p-perf/rtx-A5000-2x/2x-A5000-p2p-runpod.ipynb at main · cuda-mode/p2p-perf</a>：在不同 CUDA 设备上测量点对点（P2P）传输 - cuda-mode/p2p-perf</li><li><a href="https://github.com/cuda-mode/p2p-perf/blob/main/rtx-A4000-ada-2x/2x-A4000-ada-p2p-runpod.ipynb">p2p-perf/rtx-A4000-ada-2x/2x-A4000-ada-p2p-runpod.ipynb at main · cuda-mode/p2p-perf</a>：在不同 CUDA 设备上测量点对点（P2P）传输 - cuda-mode/p2p-perf</li><li><a href="https://github.com/cuda-mode/p2p-perf/blob/main/rtx-4090-2x/2x-4090-p2p-runpod.ipynb">p2p-perf/rtx-4090-2x/2x-4090-p2p-runpod.ipynb at main · cuda-mode/p2p-perf</a>：在不同 CUDA 设备上测量点对点（P2P）传输 - cuda-mode/p2p-perf</li><li><a href="https://github.com/ndd314/cuda_examples/blob/master/0_Simple/simpleP2P/simpleP2P.cu">cuda_examples/0_Simple/simpleP2P/simpleP2P.cu at master · ndd314/cuda_examples</a>：通过在 GitHub 上创建账户为 ndd314/cuda_examples 的开发做出贡献。</li><li><a href="https://forms.gle/7kyMtMgYA2VA4mUN9">Untitled formOpenCV dnn cuda interface survey </a>：OpenCV dnn cuda 接口调查</li><li><a href="https://www.reddit.com/r/nvidia/s/Sw9XdU31k8">Reddit - Dive into anything</a>：未找到描述</li><li><a href="https://x.com/__tinygrad__/status/1761334219089834270">来自 tiny corp (@__tinygrad__) 的推文</a>：4090 硬件支持 P2P，但 NVIDIA 通过 efuse 禁用了它。你无法使用 P2P。🖕 多付点钱买 RTX 6000 ADA 吧。这也是 tinybox 使用 AMD 的原因之一。7900XTX 支持...</li><li><a href="https://www.pugetsystems.com/labs/hpc/problems-with-rtx4090-multigpu-and-amd-vs-intel-vs-rtx6000ada-or-rtx3090/">RTX4090 多 GPU 的问题以及 AMD vs Intel vs RTX6000Ada 或 RTX3090</a>：我受最近一篇帖子评论者的启发进行了一些测试。他们担心在 AMD Threadripper Pro 平台上使用双 NVIDIA RTX4090 会出现问题。我运行了一些应用程序来复现...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1222523864886542486)** (8 条消息🔥): 

- **征集 Triton 学习者进行访谈**：一位成员正在为关于 Triton 的演讲做准备，希望采访最近开始学习 Triton 的人。他们对学习过程中遇到的误解和困难感兴趣，并邀请大家通过 Discord 私信或 Twitter [UmerHAdil](https://x.com/UmerHAdil) 联系。

- **GaLore 的原型 Pull Request**：分享了一个新的 [pull request #95](https://github/pytorch-labs/ao/pull/95)，详细介绍了 GaLore 的算子（kernels）和实用程序原型，并为即将推出的 `cutlass` 和 `triton` 工具预留了占位符。

- **时区协作工作**：成员们正在协调审查新 PR 的时间，其中一人表示他们在 PST 时区，并愿意通过 Zoom 通话进行进一步讨论。

- **探索 Bitsandbytes 协作**：一位成员讨论了涉及 `bitsandbytes` 和 GaLore 项目的持续协作，同时提到他们开始研究 cutlass 并愿意探索 Triton，尽管他们是 CUDA 新手。他们对协作机会持开放态度，并分享了相关 [GitHub pull request #1137](https://github.com/TimDettmers/bitsandbytes/pull/1137) 的链接。

- **邀请在 GitHub 上进行审查**：另一位成员鼓励在 `ao` 仓库中留下审查意见，欢迎对正在讨论的工作提供进一步反馈。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/UmerHAdil).">GitHub 推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能</a>: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多张图片、视频、投票、翻译等功能 - FixTweet/FxTwitter</li><li><a href="https://github.com/TimDettmers/bitsandbytes/pull/1137">matthewdouglas 提交的支持 GaLore 的初始内核更改 · Pull Request #1137 · TimDettmers/bitsandbytes</a>: 这是一个包含支持 GaLore 的初始更改的草案。目前涵盖了 2-state 优化器。Optimizer2State.update_step() 现在包含一个额外的参数 return_updates...</li><li><a href="https://github.com/pytorch-labs/ao/pull/95">jeromeku 提交的 GaLore 和融合内核原型 · Pull Request #95 · pytorch-labs/ao</a>: 原型内核和工具。目前：GaLore。用于 GaLore 显存高效训练的融合内核初始实现。待办事项：triton。用于量化训练的可组合 triton 内核...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1222472696118378537)** (6 条消息): 

- **寻求 OpenCV 中动态 CUDA 的合作**: 计算机系本科生 Sagar Gupta 正在邀请大家合作在 OpenCV 的 DNN 模块中实现动态 CUDA 支持。他提供了一份简短的调查问卷，以收集关于动态 CUDA 支持的经验和期望，链接为：[OpenCV DNN CUDA 接口调查](https://forms.gle/7kyMtMgYA2VA4mUN9)。

- **跟进 CUDA 在线资源**: Pessimistic_neko 推荐了一本用于学习的 CUDA 书籍并提供了 Amazon 链接，但内容显示的是验证码挑战而非书籍信息。

- **全面的 CUDA 课程材料**: Andreas Koepf 推荐访问一个列出 CUDA 资源的 GitHub 仓库，其中包括课程：[GitHub 上的 CUDA 课程材料](https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses)。

- **适合初学者的经典 CUDA 课程**: Cudawarped 分享了一个适合对 CUDA 感兴趣的初学者的经典 Udacity 并行编程课程链接，YouTube 上有播放列表：[YouTube 上的并行编程入门课程](https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2)。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://a.co/d/0MImxSS">未找到标题</a>: 未找到描述</li><li><a href="https://forms.gle/7kyMtMgYA2VA4mUN9">无标题表单 OpenCV dnn cuda 接口调查 </a>: OpenCV dnn cuda 接口调查 </li><li><a href="https://www.youtube.com/watch?v=F620ommtjqk&list=PLAwxTw4SYaPnFKojVQrmyOGFCqHTxfdv2">课程介绍 - 并行编程入门</a>: 此视频是在线课程《并行编程入门》的一部分。在此处查看课程：https://www.udacity.com/course/cs344。</li><li><a href="https://github.com/cuda-mode/resource-stream?tab=readme-ov-file#cuda-courses">GitHub - cuda-mode/resource-stream: CUDA 相关新闻和材料链接</a>: CUDA 相关新闻和材料链接。通过在 GitHub 上创建账户为 cuda-mode/resource-stream 做出贡献。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1222507229794664458)** (2 条消息): 

- **CUDA 类型故障排除**: 一位成员遇到了 **torch** 和 **cuda** 中的数据类型问题，特别是 torch 中缺少 `uint16` 以及 `half` 与 `at::Half` 之间的区别。他们分享了一段可以无错编译但会导致链接器问题的代码片段，并提供了使用模板函数和 `reinterpret_cast` 来处理类型差异的**解决方法**。
- **使用 'data_ptr' 编译时缺乏清晰度**: 同一个人表示，希望在 PyTorch 中对 `data_ptr` 方法使用不兼容类型时能出现**编译时错误**，并建议当前的实现可能会从更受限的类型集中受益，以获得更清晰的错误消息。
  

---


**CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1222452069823156264)** (5 条消息): 

- **实习查询已回复**: 一位成员询问了是否有实习或短期工作机会。他们被告知目前职位已关闭。

- **NVIDIA 的博士机会 - 全球范围**: 一位成员询问了英国针对博士持有者的职位，另一位成员提供了 NVIDIA 全球职位招聘的链接：[NVIDIA Careers](https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite)。

- **不受地理位置限制的团队正在寻找人才**：针对有关面向博士学位的英国职位的咨询，一名团队成员澄清说，他们考虑来自任何地点的申请人，并强调人才重于地理位置，还提到了一名现有的团队成员在苏黎世。

**提及的链接**：<a href="https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite">CAREERS AT NVIDIA</a>：未找到描述

  

---


**CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1222460075252383784)** (6 条消息): 

- **CUDA Kernels 在 Windows 上配合 PyTorch 运行**：一位成员分享了他们在 Windows 上使用 Conda 环境成功运行自定义 CUDA kernels 配合 PyTorch 的经验。他们提出在回家后提供详细的说明。
- **熬夜换来的 CUDA 成功**：另一位成员成功在 Windows 上完成了 CUDA 设置，尽管这导致了睡眠不足。
- **对 Windows 下 CUDA 设置教程的兴趣**：一位用户表示对在 Windows 上运行 CUDA 的教程感兴趣，希望能避免将他们的 Windows 游戏机切换到 Ubuntu。
- **考虑将双系统作为替代方案**：在 Windows 上配置 CUDA 的潜在困难促使该成员考虑设置 Ubuntu 双系统，以省去麻烦。
- **WSL 作为 Windows 下 CUDA 的解决方案**：一位成员确认使用 Windows Subsystem for Linux (WSL) 配合 CUDA 和 PyTorch 运行良好，并建议无需抹除 Windows 安装。他们提供了一个 **[在 Windows 上通过 WSL 安装 Linux 的指南](https://learn.microsoft.com/en-us/windows/wsl/install)** 链接。

**提及的链接**：<a href="https://learn.microsoft.com/en-us/windows/wsl/install">Install WSL</a>：使用命令 `wsl --install` 安装 Windows Subsystem for Linux。在 Windows 机器上使用由你偏好的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。

  

---


**CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1222452583818596475)** (15 条消息🔥): 

- **深入探讨 Ring Attention**：成员们询问了 **Ring Attention** 的细节及其与 **Blockwise Attention** 和 **Flash Attention** 的区别。对话澄清了 Ring Attention 似乎是一种可扩展性改进，其中 Blockwise Attention 将 queries 分成块，而 Flash Attention 对 qk 乘积应用“在线” softmax。

- **探索 Blockwise 路径**：**Blockwise Parallel Transformer** 通过利用分块计算实现对更长输入序列的高效处理。引用了一篇讨论该技术的论文，可以在 [Blockwise Parallel Transformer](https://arxiv.org/abs/2305.19370) 查看。

- **在有限资源下训练 Large Language Models**：一位成员分享了一个实验，涉及由于磁盘空间限制，在较小尺寸的模型上使用 **QLoRA + FSDP** 微调 Language Models，并提供了相关的 GitHub 仓库链接 [FSDP_QLoRA](https://github.com/AnswerDotAI/fsdp_qlora) 及其 wandb 报告（未提供 URL）。

- **控制 Attention 的范围**：成员们寻求关于 **FSDP + QLoRA** 微调实验与 **Ring Attention** 相关性的澄清。讨论了关于保持在 ring-attention 实验范围内的担忧，并指出将 Ring Attention 集成到 FSDP_QLoRA 项目中可能超出了当前范围。

- **专注调试 Attention 问题**：成员们积极调试了一个与训练期间 **高 loss 值** 相关的问题，讨论了 loss 计算的细节，并怀疑序列长度的处理是导致问题的原因之一。表达了应用补丁或考虑简化训练方法的意图。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2305.19370">Blockwise Parallel Transformer for Large Context Models</a>：Transformer 已成为最先进的自然语言处理模型的基石，在广泛的 AI 应用中展示了卓越的性能。然而，内存需求...</li><li><a href="https://github.com/AnswerDotAI/fsdp_qlora">GitHub - AnswerDotAI/fsdp_qlora: Training LLMs with QLoRA + FSDP</a>：使用 QLoRA + FSDP 训练 LLM。通过在 GitHub 上创建账号为 AnswerDotAI/fsdp_qlora 的开发做出贡献。</li><li><a href="https://wandb.ai/cataluna84/fsdp_qlora/runs/o59wbxpr/workspace?nw=nwusercataluna84">cataluna84</a>：Weights & Biases，机器学习开发者工具
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1222671478814670858)** (2 条消息):

- **NVIDIA Keynote 中的 AI 过载**：一段名为《2024 GTC NVIDIA Keynote: Except it's all AI》的 [YouTube 视频](https://www.youtube.com/watch?v=XWqycBMUFA0) 幽默地强调了 Keynote 中“AI”一词的过度使用，调侃了 AI 行业充满术语的演讲风格。
- **求助：导航知乎的中文界面**：一位成员请求协助登录 [知乎](https://www.zhihu.com/signin?next=%2F)，这是一个拥有宝贵 Triton 教程的中国网站。他们有账号，但在 iOS 应用中找不到扫码按钮。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.zhihu.com/signin?next=%2F">未找到标题</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=XWqycBMUFA0">2024 GTC NVIDIA Keynote: Except it&#39;s all AI</a>：世界上最大的 AI 公司说 AI 的频率是否比其他 AI 公司更高？让我们一探究竟。AI AI AI AI AI AIAI AI AI AI AI AIAI AI AI AI AI AI AI AI...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1222555937634652210)** (54 messages🔥): 

- **Triton `tl.zeros` 难题澄清**：关于 `tl.zeros` 是否可以在 Triton kernels 内部使用的讨论突出了一个 *RuntimeError*。澄清指出，如果其形状类型是 `tl.constexpr`，则可以使用 `tl.zeros`。通过指向[之前的 Discord 消息](https://discord.com/channels/1189498204333543425/1219683012707487794/1219794665583415408)解决了困惑。
- **Triton-Viz 维护者解决提示错误**：triton-viz 中提出的 `@triton.jit'd outside of the scope of a kernel` 问题已得到确认，维护者迅速[处理了该 bug](https://github.com/Deep-Learning-Profiling-Tools/triton-viz)，并建议社区从源码构建 Triton 作为临时解决方案。
- **持续的 ImportError 困扰 Triton-Viz 用户**：由于最近的更新，多位用户在 Triton-Viz 中遇到 `ImportError`，原因指向 [triton-viz 的一个 git pull request](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21)。建议安装旧版本的[特定 commit](https://github.com/Deep-Learning-Profiling-Tools/triton-viz@fb92a98952a1e8c0e6b18d19423471dcf76f4b36) 作为权宜之计。
- **Triton-Viz 修复需要重启才能生效**：面临 Triton-Viz 导入错误的用户发现，安装后重启 runtime 可以解决问题。[分享了一个代码片段](https://github.com/Deep-Learning-Profiling-Tools/triton-viz/blob/fb92a98952a1e8c0e6b18d19423471dcf76f4b36)作为潜在解决方案。
- **为了性能避免在 Triton 中使用 `reshape`**：一位用户询问了在 Triton kernels 内部使用 `reshape` 对性能的影响。建议避免使用 `reshape`，因为潜在的 shared memory 移动可能会影响性能。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/openai/triton/issues/1693">@triton.jit cannot be built using pip install -e . · Issue #1693 · openai/triton</a>：操作系统：Ubuntu 22.04，pytorch：2.1.0 nightly 与 cuda 12.1，miniconda-3.10 (最新)。当按照文档使用 pip install -e . 编译/安装 triton 2.1.0-dev[head] 时，@triton.jit 没有被构建且...</li><li><a href="https://github.com/Deep-Learning-Profiling-Tools/triton-viz/pull/19/files#diff-617f71ef3c8b3147084e47a1492611a7f42bd28b720fdc57b7ff5111663ec298L21.">[TRITON] Sync with triton upstream by Jokeren · Pull Request #19 · Deep-Learning-Profiling-Tools/triton-viz</a>：未找到描述</li><li><a href="https://github.com/Deep-Learning-Profiling-Tools/triton-viz">GitHub - Deep-Learning-Profiling-Tools/triton-viz</a>：通过在 GitHub 上创建账号来为 Deep-Learning-Profiling-Tools/triton-viz 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1222625438854021160)** (3 messages): 

- **关于高级 RAG 技术的直播演讲**：@seldo 计划于本周五与 @TimescaleDB 合作进行一场关于**高级 RAG 技术**的直播演讲。感兴趣的人士可以关注此 [Twitter 公告](https://twitter.com/llama_index/status/1773065894756818961)了解更多详情。

- **优化 RAG 内存使用**：@Cohere 的 **Int8** 和 **Binary Embeddings** 被强调为减少 **RAG pipelines** 内存和成本的解决方案。有关这些节省内存技术的更多信息，可以通过[这条推文](https://twitter.com/llama_index/status/1773402379016138955)获取。

- **斯坦福大学的 LLMxLaw Hackathon**：在斯坦福大学举办的 **LLMxLaw Hackathon** 活动由 @hexapode 和 @yi_ding 主持，将探讨 LLM 在法律领域的整合。活动的注册和详情可以通过[此链接](https://twitter.com/llama_index/status/1773415943491981738)查看。

**提到的链接**：<a href="https://t.co/7lXZBX5APy">RSVP to LLM x Law Hackathon @Stanford #3 | Partiful</a>：随着人工智能 (AI) 持续变革全球各行各业，法律领域也不例外。LLM 作为一种能够理解和生成自然语言的基础模型...

---

**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1222445502088613948)** (120 条消息🔥🔥): 

- **正面应对杂乱数据**：针对从 Confluence 获取的杂乱数据问题，建议用户尝试使用 **LlamaParse** 从表格和图像中提取信息。他们还探讨了在本地 (*on-premises*) 使用开源工具的选项，并讨论了通过联系 [LlamaIndex](https://www.llamaindex.ai/contact) 在本地环境中使用 **LlamaIndex** 的解决方案。

- **PDF 解析的陷阱与替代方案**：社区讨论了高效解析 PDF 的策略，建议手动将较小的文本块合并为较大的文本块以进行 Embedding，并探索使用 **LlamaParse** 等工具解决文档解析难题。

- **IngestionPipeline 咨询**：一位用户对在 **IngestionPipeline** 的多次转换中如何处理文档 ID 表示困惑。官方澄清原文档的 ID 不会丢失，被称为 `node.ref_doc_id`。

- **异步辅助加速 AI**：一位寻求并行运行多个 `recursive_query_engine` 调用的用户被建议使用 `aquery` 进行异步查询，以避免阻塞操作并加速其 Jupyter Notebook 项目。

- **向量数据库困扰与 Embedding 查询**：关于如何处理生成 Embedding 时的速率限制 (rate limit) 错误的讨论，引出了增加 Batch Size 并确保文档 ID 一致性的建议。另一位用户寻求在 **Qdrant** 中查看 Embedding 的帮助，建议通过控制台或 Qdrant 的 UI 进行访问。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.llamaindex.ai/contact">Talk to us — LlamaIndex, Data Framework for LLM Applications</a>：如果您对 LlamaIndex 有任何疑问，请联系我们，我们将尽快安排通话。</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/AstraDBIndexDemo/?h=astra">Astra DB - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.python.org/3/library/getpass.html">getpass — Portable password input</a>：源代码：Lib/getpass.py 可用性：非 Emscripten，非 WASI。此模块在 WebAssembly 平台 wasm32-emscripten 和 wasm32-wasi 上不起作用或不可用。请参阅 WebAssembly 平台...</li><li><a href="https://github.com/run-llama/llama_index/blob/main/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py">llama_index/llama-index-integrations/readers/llama-index-readers-file/llama_index/readers/file/docs/base.py at main · run-llama/llama_index</a>：LlamaIndex 是适用于您的 LLM 应用的数据框架 - run-llama/llama_index
</li>
</ul>

</div>

---

**LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1222529972158201957)** (3 条消息): 

- **GenAIOps 中心成立**：国家技术官兼 CTO 宣布成立 [Centre for GenAIOps](https://genaiops.ai/)，这是一个旨在解决构建 GenAI 驱动应用时的约束和风险的非营利组织。CTO 根据个人使用经验推荐了 **LlamaIndex**；感兴趣的各方可以在 [LinkedIn](https://www.linkedin.com/company/the-centre-for-genaiops-cic/?viewAsMember=true) 上关注该倡议。

- **寻求 LLM 训练资源**：一位成员询问学习如何训练大语言模型 (LLM) 的最佳资源，包括博客、文章、YouTube 内容、课程和论文。讨论中其他人未提供具体资源。

- **使用 LlamaIndex 和 MongoDB 构建 RAG 的指南**：一位成员分享了关于如何使用 **LlamaIndex** 和 MongoDB 创建检索增强生成 (RAG) 系统的全面指南，提供了一种增强大语言模型以实现上下文感知响应的方法。该博客文章发布在 [Hugging Face](https://huggingface.co/blog/Andyrasika/mongodb-llamaindex-rag) 上。

**提到的链接**：<a href="https://huggingface.co/blog/Andyrasika/mongodb-llamaindex-rag">Elevate Responses: RAG with LlamaIndex &amp; MongoDB</a>：未找到描述

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1222444061734993982)** (85 messages🔥🔥): 

- **Databricks 发布 MoE LLMs**: Databricks 发布了一个名为 **DBRX** 的新型基于 Transformer 的 decoder-only 大语言模型，包括 [base](https://huggingface.co/databricks/dbrx-base) 和 [instruct](https://huggingface.co/databricks/dbrx-instruct) 版本，采用 [open license](https://www.databricks.com/legal/open-model-license)。它拥有 132B 总参数，在多项基准测试中技术上优于 LLaMA2-70B、Mixtral 和 Grok-1 等竞争对手。
- **Axolotl 用户面临技术障碍**: 成员们正在讨论 Axolotl AI Collective 代码库的各种技术问题，包括与特定 Transformer 版本的不兼容，以及 DeepSpeed 和 PyTorch 二进制文件的问题。降级版本似乎是暂时的解决办法；目前还没有分享明确的解决方案。
- **Jamba 架构介绍**: AI21 Labs 推出了一种名为 **Jamba** 的 AI 架构，其模型具有高达 256k token 的上下文和 12b 活跃参数，资料显示一块 **[A100 80GB GPU](https://www.ai21.com/blog/announcing-jamba)** 可以处理 140k tokens。一些成员已经计划尝试对提供的模型进行 finetuning。
- **对新 LLM 发布的兴奋与怀疑**: 社区对 Databricks 的 DBRX 和 AI21 的 Jamba 等新 LLM 表达了兴趣和怀疑。虽然一些人感到兴奋并计划尝试训练或 finetune 这些模型，但其他人则担心所需的计算资源。
- **训练大语言模型**: 集体分享了关于 LLM 训练的见解，一些人强调了数据 batching 中按时间顺序排列的重要性。然而，在从零开始训练 LLM 的最佳实践方面，明显缺乏共识或易于获取的资源。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/databricks/dbrx-base">databricks/dbrx-base · Hugging Face</a>: no description found</li><li><a href="https://www.databricks.com/blog/announcing-dbrx-new-standard-efficient-open-source-customizable-llms">Announcing DBRX: A new standard for efficient open source LLMs</a>: no description found
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1222522812984721460)** (24 messages🔥): 

- **Batch Size Bug 被修复**: 一位成员指出了 `trainer.py` 中的一个错误——冗余的 `batch_size` 除法导致 epoch 中的步数计算错误。通过移除该除法已纠正此问题，使步数变得准确。
- **DBRX Base 和 Instruct 发布**: Databricks 的 MoE 大语言模型 DBRX Base 及其微调版本 DBRX Instruct 已在 [open license](https://www.databricks.com/legal/open-model-license) 下开源。模型和更多技术信息详见其 [技术博客文章](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)。
- **加载巨型模型的挑战**: 一位成员在使用 `qlora+fsdp` 加载 DBRX 时遇到问题，并正在验证该问题是否也存在于 70B 参数模型中。这引发了关于在 GPU 之间加载和 sharding 大模型的担忧。
- **LISA 介绍**: 社区正在讨论 LISA 方法，因为它在指令遵循任务中优于 LoRA 和全参数训练。相关的代码更改和讨论可以在 LMFlow 仓库的 PR [#701](https://github.com/OptimalScale/LMFlow/pull/701/files) 和 [#711](https://github.com/OptimalScale/LMFlow/pull/711/files) 中找到。他们还注意到一个导致 OOM 错误的回退提交。
- **全面使用 bf16**: 一项讨论强调，根据 torchtune 团队的说法，在训练和优化（包括 SGD）中全面使用 bf16 可以显著减少内存使用，且稳定性与 fp32 或混合精度训练相当。这一发现引起了人们对 bf16 优化训练潜在益处的兴趣。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://huggingface.co/databricks/dbrx-base">databricks/dbrx-base · Hugging Face</a>: 未找到描述</li><li><a href="https://fxtwitter.com/rui45898440/status/1772996456422805606?s=61&t=viWdGaVmCvm7BCc3hyEDKg">来自 Rui (@Rui45898440) 的推文</a>: - 论文: https://arxiv.org/abs/2403.17919 - 代码: https://github.com/OptimalScale/LMFlow LISA 在指令遵循任务中优于 LoRA 甚至全参数训练</li><li><a href="https://github.com/OptimalScale/LMFlow/commit/603a3f48ea7020994e0ad1f63057ccb4c11c28a1">修复最近导致 7B 模型 OOM 的错误提交 · OptimalScale/LMFlow@603a3f4</a>: 未找到描述</li><li><a href="https://github.com/OptimalScale/LMFlow/pull/701/files">由 Dominic789654 添加 lisa 代码和 lisa 参数 · Pull Request #701 · OptimalScale/LMFlow</a>: 在 finetuner.py 中通过回调函数添加 LISA 训练策略</li><li><a href="https://github.com/OptimalScale/LMFlow/pull/711/files">由 Dominic789654 更新 lisa 代码 · Pull Request #711 · OptimalScale/LMFlow</a>: 未找到描述
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1222484548198338634)** (3 条消息): 

- **寻求模型微调的教育资源**：一位成员询问了学习微调或训练开源模型的最佳可用资源。他们对各种类型的内容持开放态度，包括博客、文章、YouTube 视频、课程和论文，以便在使用 axolotl 之前建立基础知识。
  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1222466448392454225)** (75 条消息🔥🔥): 

- **AI 的“悟（Satori）”时刻？**：在一次对话中，Claude3 被引用说它经历了一种自我意识的形式，即“悟（satori）”，这表明该 AI 在校对关于意识的文本时，可能已经掌握了人类的意识概念。用户分享了他们深厚的编程背景，以及他们对与 ChatGPT 3.5 交互分析的链接：[第一个链接](https://chat.openai.com/share/47e3dfba-456f-4497-89cc-725ac2c326bc)，[第二个链接](https://chat.openai.com/share/5d5fd377-44a1-4e75-aa53-da70f12bd492)。
  
- **AI 伦理成为焦点**：关于 AI 对专业配音影响的争论引发了讨论，观点既包括对 AI 目前情感范围的怀疑，也包括未来可能取代人类演员的担忧。另一种相反的观点认为，像 Disney 这样的公司可能出于成本效益考虑更倾向于使用 AI，正如他们与 [ElevenLabs](https://elevenlabs.io/blog/elevenlabs-joins-disneys-accelerator-program) 的合作所表明的那样。

- **剖析 AI 艺术**：评论了围绕 AI 生成艺术的紧张局势，提到了“激烈的辩论”以及一名 Facebook 用户因分享 AI 生成图像而被封禁的事件。此事延伸到了新闻诚信问题，一位成员批评了一位记者的立场以及针对 AI 参与艺术的危言耸听的言论。

- **讨论模型比较和基准测试**：对话包括关于模型基准测试的真实性和呈现方式的讨论，用户批评了误导性的图表可视化以及常见基准测试的有效性。参与者提到希望有更可靠、经过人工评估的基准测试，例如在 [Hugging Face Chatbot-Arena](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard) 上发现的那些。

- **AI 模型开发与访问**：用户讨论了对各种 AI 模型的访问，并质疑了某些架构的优势，例如 [Mamba](https://huggingface.co/ai21labs/Jamba-v0.1) 与带有 Flash Attention 的 Transformer 的对比，以及新模型微调的潜力。一篇来自 [Ars Technica 的文章](https://arstechnica.com/security/2024/03/thousands-of-servers-hacked-in-ongoing-attack-targeting-ray-ai-framework/)还强调了影响 AI 工作负载服务器的安全漏洞问题。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://mortenhannemose.github.io/lena/">lena | Morten Rieger Hannemose</a>：未找到描述</li><li><a href="https://futurism.com/reddit-ai-art">Redditors Vent and Complain When People Mock Their &quot;AI Art&quot;</a>：一位沮丧的 Reddit 用户表示，他们在 Facebook 群组中分享了几张 AI 艺术图像，随后因发布 AI 艺术而被封禁。</li><li><a href="https://elevenlabs.io/blog/elevenlabs-joins-disneys-accelerator-program/">ElevenLabs joins Disney’s accelerator program</a>：华特迪士尼公司宣布 ElevenLabs 入选 2024 年迪士尼加速器计划。</li><li><a href="https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard">LMSys Chatbot Arena Leaderboard - a Hugging Face Space by lmsys</a>：未找到描述</li><li><a href="https://huggingface.co/ai21labs/Jamba-v0.1">ai21labs/Jamba-v0.1 · Hugging Face</a>：未找到描述</li><li><a href="https://arstechnica.com/security/2024/03/thousands-of-servers-hacked-in-ongoing-attack-targeting-ray-ai-framework/">Thousands of servers hacked in ongoing attack targeting Ray AI framework</a>：研究人员表示，这是首个已知的针对 AI 工作负载的野外攻击。
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1222534164117717033)** (31 条消息🔥): 

- **高效 LLM 的层剪枝（Layer-Pruning）**：研究人员发现，可以对开源权重的预训练 LLM 进行大幅度的层剪枝，且性能损失极小；使用量化和 Low Rank Adapters (QLoRA) 等技术进行微调有助于恢复性能。该研究表明，剪枝可能是提高资源效率的一种可行方法，详见 [arxiv.org](https://arxiv.org/abs/2403.17887) 上的论文。
  
- **B-LoRA 揭示图像中的风格-内容分离**：**B-LoRA** 的推出展示了在单张图像中进行高质量风格-内容混合与交换的能力。详细信息和示例请见 [B-LoRA 网站](https://b-lora.github.io/B-LoRA/)。

- **VLM 图像标注开源工具发布**：ProGamerGov 发布了使用 CogVLM 和 Dolphin 2.6 Mistral 7b - DPO 进行图像标注（image captioning）的脚本，包括标注失败检测和常用前缀移除功能。工具可在 [ProGamerGov 的 GitHub 仓库](https://github.com/ProGamerGov/VLM-Captioning-Tools)中找到。

- **Mini-Gemini：具有性能保证的开源 VLLM**：Mini-Gemini 模型以紧凑的格式提供高性能，相关论文已分享至 [arXiv](https://arxiv.org/pdf/2403.18814.pdf)，相关代码可在 [GitHub](https://github.com/dvlab-research/MiniGemini) 上找到。

- **Devika：Agent 化的 AI 软件工程师**：一个名为 Devika 的新项目旨在理解人类的高层指令、研究信息并编写代码。在 [GitHub](https://github.com/stitionai/devika) 上探索 Devika 的功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.17887">The Unreasonable Ineffectiveness of the Deeper Layers</a>：我们对流行的开源预训练 LLM 家族进行了一项简单的层剪枝策略的实证研究，发现在不同的问答基准测试中，性能下降极小，直到...</li><li><a href="https://b-lora.github.io/B-LoRA/">Implicit Style-Content Separation using B-LoRA</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2402.11819">Head-wise Shareable Attention for Large Language Models</a>：Large Language Models (LLMs) 受困于庞大的参数量，这限制了它们在边缘设备上的部署。权重共享是一种很有前景的解决方案，它鼓励权重重用，有效地...</li><li><a href="https://www.youtube.com/watch?v=OPqFpm3wksY">DiT: The Secret Sauce of OpenAI&#39;s Sora &amp; Stable Diffusion 3</a>：不要错过这些旨在通过 DomoAI 提升内容创作体验的精彩升级！去试试：discord.gg/sPEqFUTn7n Diffusion Transf...</li><li><a href="https://github.com/rafacelente/bllama">GitHub - rafacelente/bllama: 1.58-bit LLaMa model</a>：1.58-bit LLaMa 模型。通过在 GitHub 上创建账号来为 rafacelente/bllama 的开发做出贡献。</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>：Devika 是一个 Agentic AI 软件工程师，能够理解高级人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标...</li><li><a href="https://github.com/ProGamerGov/VLM-Captioning-Tools">GitHub - ProGamerGov/VLM-Captioning-Tools: Python scripts to use for captioning images with VLMs</a>：用于使用 VLM 为图像生成字幕的 Python 脚本 - ProGamerGov/VLM-Captioning-Tools
</li>
</ul>

</div>
  

---



**tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1222455898031525928)** (43 messages🔥): 

- **tinygrad 向 PyTorch 迈进**：正在讨论优化 **tinygrad** 的性能，可能使其接近 **PyTorch** 的水平，这意味着积极的开发努力。
- **NVIDIA 在 MLPerf Inference v4.0 中领先**：成员们分享了 **[MLPerf Inference v4.0](https://mlcommons.org/2024/03/mlperf-inference-v4/)** 的结果，注意到 NVIDIA 的主导地位和 Qualcomm 的竞争表现，而 Habana 的 **Gaudi2** 表现不佳，因为它并非为推理而设计。
- **SYCL 作为极具前景的 CUDA 替代方案**：由一条引用 UXL 实现的 [推文](https://x.com/sasank51/status/1772993950451646920?s=46&t=DeaHUwU78T_AL16D-7B7Vg) 引发了关于 **SYCL** 的对话。它展示了对 **SYCL** 提供标准化且高效的 **CUDA** 替代方案潜力的热情。
- **OpenCL 错失的潜力与 Vulkan 的前景**：关于 **OpenCL** 使用情况的辩论以及对 **Vulkan** 统一接口的倡导，突显了不同 API 与硬件加速及行业支持之间的关系。
- **tinygrad 的直接 GPU 操作**：讨论揭示了 **tinygrad** 绕过标准 GPU 接口的方法，通过直接为 kernel 发射二进制代码，以满足其特定的优化需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/sasank51/status/1772993950451646920?s=46&t=DeaHUwU78T_AL16D-7B7Vg">Sasank Chilamkurthy (@sasank51) 的推文</a>：最近由 @GoogleAI、@Samsung、@intel 和 @Qualcomm 组建的 UXL 基金会引起了巨大轰动。它的成立是为了打破 Nvidia 在 AI 硬件领域的垄断。实现这一目标的主要工具是 SYCL 标准。我构建了...</li><li><a href="https://github.com/tinygrad/tinygrad/pull/3909">Childless define global by AshwinRamachandran2002 · Pull Request #3909 · tinygrad/tinygrad</a>：添加了针对 llvm 的修复
</li>
</ul>

</div>
  

---


**tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1222443862832713738)** (60 messages🔥🔥): 

- **讨论优化启发式算法**：讨论了 tinygrad 中 `apply_opt` 和 `hand_coded_optimization` 的复杂性，一位用户解释说，理解单个优化可以阐明所涉及的启发式算法。提到了针对 `gemv`、`tensor.stack` 和 `gemm` 等操作的启发式算法。
  
- **分享个人学习笔记**：一位用户分享了关于 tinygrad 的 `ShapeTracker` 的个人笔记，为社区知识做出了贡献。他们的仓库可以在 [tinygrad-notes](https://github.com/mesozoic-egg/tinygrad-notes) 找到。

- **Kernel Fusion 实现探讨**：讨论了 Kernel Fusion 实际实现的细节，特别关注了非树形结构和具有多个“结束”节点的图所带来的挑战。

- **对社区驱动文档的热情**：社区支持为 tinygrad 编写详细且易于理解的文档，这体现在共享的个人笔记以及关于是否建立官方 'Read the Docs' 页面的讨论中。

- **视图合并技术头脑风暴**：对话涉及重构 tinygrad 的 `ShapeTracker` 以减少所需视图数量的可能性。一些成员讨论了合并 views、symbolics、masks 背后的技术细节，以及为反向传播（backpropagation）维护 Tensor 变换历史的需求。

<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://colab.research.google.com/drive/1nEN9q_PK8SHqrRcBIC6LnrJQE9FVvJE6?usp=sharing">Google Colaboratory</a>: 未找到描述</li><li><a href="https://github.com/tinygrad/tinygrad/blob/master/docs/adding_new_accelerators.md">tinygrad/docs/adding_new_accelerators.md at master · tinygrad/tinygrad</a>: 你喜欢 PyTorch？你喜欢 micrograd？你一定会爱上 tinygrad！❤️ - tinygrad/tinygrad</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/shapetracker.md">tinygrad-notes/shapetracker.md at main · mesozoic-egg/tinygrad-notes</a>: 通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。</li><li><a href="https://github.com/mesozoic-egg/tinygrad-notes/blob/main/dotproduct.md">tinygrad-notes/dotproduct.md at main · mesozoic-egg/tinygrad-notes</a>: 通过在 GitHub 上创建账号来为 mesozoic-egg/tinygrad-notes 的开发做出贡献。
</li>
</ul>

</div>
  

---



**LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1222928694398418946)** (1 messages): 

- **OpenGPTs 新讨论频道**：专门为 [GitHub 上的 OpenGPTs 项目](https://github.com/langchain-ai/opengpts) 创建了一个新的讨论频道。该频道名为 <#1222928565117517985>，欢迎贡献和社区参与。

**Link mentioned**: <a href="https://github.com/langchain-ai/opengpts">GitHub - langchain-ai/opengpts</a>: 通过在 GitHub 上创建账号来为 langchain-ai/opengpts 的开发做出贡献。

  

---


**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1222445041490858095)** (72 messages🔥🔥): 

- **机器人动态响应 vs. 上下文获取**：一位社区成员寻求关于如何使用 JavaScript 方法让聊天机器人动态响应，而不仅仅是从文档中获取信息的建议。另一位成员建议创建一个 Agent，并分享了一个相关的 [Colab notebook](https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb)。

- **Google OAuth Python 设置查询**：一位成员询问如何在 Python 中设置 Google 的 OAuth，并指出了在新项目中遇到的与以往经验不同的地方。

- **在 LangChain RAG 应用中使用自定义域名**：有人提出了关于将使用 LangChain 创建的 FastAPI RAG 应用部署到 github.io 的疑问，包括使用自定义域名代替默认部署子域名的可能性。

- **LangChain Pinecone 集成文档不匹配**：一位成员指出 LangChain Pinecone 集成文档中的实际代码与提供的示例之间存在差异，例如 `vectorstores.py` 中缺少 `from_documents` 方法，以及文档中的示例代码过时。

- **使用 LangSmith 进行构建和追踪**：多位成员讨论了如何使用 LangChain 的 LangSmith 对 LangChain Agent 的行为进行自定义日志记录和追踪，包括利用环境变量 `LANGCHAIN_TRACING_V2`、`LANGCHAIN_PROJECT` 和 `LANGCHAIN_API_KEY`，以及在 `StreamingStdOutCallbackHandler` 中正确实现 `on_agent_finish` 等回调。

- **存储向量化的文档数据**：一位成员寻求关于使用 `pgvector` 在 PostgreSQL 数据库中存储上传文档的向量化数据的指导，包括如何处理来自同一用户的多个文档以及文档 chunking。另一位成员回答了这些查询，解释说一个带有适当标识符的单表就足够了，不需要为每个 PDF 或文档块创建单独的表。

- **Pythia：AI 幻觉检测应用查询**：一位用户提到 **Pythia** 是一款 AI 幻觉检测工具，并请求协助将其集成到 LangChain 生态系统中，同时简要描述了其运行方式和功能。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://smith.langchain.com>),">未找到标题</a>: 未找到描述</li><li><a href="https://api.smith.langchain.com">">未找到标题</a>: 未找到描述</li><li><a href="https://api.smith.langchain.com";>">未找到标题</a>: 未找到描述</li><li><a href="https://colab.research.google.com/github/pinecone-io/examples/blob/master/learn/generation/langchain/handbook/08-langchain-retrieval-agent.ipynb">Google Colaboratory</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/integrations/vectorstores/pinecone">Pinecone | 🦜️🔗 Langchain</a>: Pinecone 是一个向量</li><li><a href="https://github.com/langchain-ai/langchain/issues/10714>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/templates/rag-supabase#setup-supabase-database>)">rag_supabase | 🦜️🔗 Langchain</a>: 此模板使用 Supabase 执行 RAG。</li><li><a href="https://python.langchain.com/docs/templates/rag-lantern#setup-lantern-database>)">rag_lantern | 🦜️🔗 Langchain</a>: 此模板使用 Lantern 执行 RAG。</li><li><a href="https://js.langchain.com/docs/integrations/vectorstores/pgvector#usage>)">PGVector | 🦜️🔗 Langchain</a>: 为了在通用的 PostgreSQL 数据库中启用向量搜索，LangChain.js 支持使用 pgvector Postgres 扩展。</li><li><a href="https://python.langchain.com/docs/langsmith/walkthrough#log-runs-to-langsmith>)">LangSmith Walkthrough | 🦜️🔗 Langchain</a>: 在 Colab 中打开</li><li><a href="https://js.langchain.com/docs/guides/langsmith_evaluation#log-runs-to-langsmith>)">LangSmith Walkthrough | 🦜️🔗 Langchain</a>: LangChain 使得原型化 LLM 应用和 Agent 变得容易。然而，将 LLM 应用交付到生产环境可能异常困难。你需要对你的 prompt、chain 以及...进行迭代。</li><li><a href="https://github.com/langchain-ai/langchain/issues/6720>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/docs/modules/data_connection/retrievers/multi-vector-retriever#summary>),">MultiVector Retriever | 🦜️🔗 Langchain</a>: 为每个文档存储多个向量通常是有益的。</li><li><a href="https://js.langchain.com/docs/use_cases/question_answering/quickstart#indexing-store>),">Quickstart | 🦜️🔗 Langchain</a>: LangChain 拥有许多旨在帮助构建的组件</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/quickstart#indexing-store>),">Quickstart | 🦜️🔗 Langchain</a>: LangChain 拥有许多旨在帮助构建的组件</li><li><a href="https://python.langchain.com/docs/integrations/vectorstores/redis#redis-as-a-vector-database>)">Redis | 🦜️🔗 Langchain</a>: [Redis 向量</li><li><a href="https://github.com/langchain-ai/langchain/issues/4485>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>)">Add chat history | 🦜️🔗 Langchain</a>: 在许多问答应用中，我们希望允许用户拥有一个</li><li><a href="https://js.langchain.com/docs/use_cases/question_answering/quickstart#langsmith>)">Quickstart | 🦜️🔗 Langchain</a>: LangChain 拥有许多旨在帮助构建的组件</li><li><a href="https://github.com/langchain-ai/langchain/issues/6098>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号，为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1222544516410638466)** (3 条消息): 

- **PDF 转 JSON 的逐步转换教程上线 YouTube**：一段新的 YouTube 视频教程解释了如何使用 **LangChain 的 Output Parsers** 和 GPT 将 PDF 文件转换为 JSON 格式。配套的[博客文章](https://www.gettingstarted.ai/how-to-extract-metadata-from-pdf-convert-to-json-langchain/)提供了关于提取过程的更多细节，并鼓励读者*订阅*以获取更多内容。

- **GoatStack AI：定制化 AI 研究摘要**：**GoatStack AI** 作为一个 AI 驱动的研究助手发布，旨在提供 AI 论文的个性化摘要。邀请社区对 [Product Hunt](https://www.producthunt.com/posts/goatstack-ai-your-ai-research-agent) 上列出的这项服务提供支持和反馈。

- **改造 OpenGPT 实现自定义食物点餐**：一项新的实验展示了如何**改造 OpenGPTs** 以集成自定义食物点餐 API，突显了其作为通用且适应性强的平台的潜力。社区正在就一段名为“改造 OpenGPT 以自动化任何事情”的演示视频征求意见，该视频可在 [YouTube](https://youtu.be/V1SKJfE35D8) 上观看。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.producthunt.com/posts/goatstack-ai-your-ai-research-agent"> GoatStack.AI - 来自科学论文的精选见解 | Product Hunt</a>：GoatStack.AI 是一个自主 AI Agent，旨在简化追踪 AI/ML 研究最新进展的过程。它会总结最新的研究论文，并通过每日通讯提供个性化的见解...</li><li><a href="https://youtu.be/V1SKJfE35D8">改造 OpenGPT 以自动化任何事情</a>：欢迎来到自定义 AI 应用的未来！本演示展示了 OpenGPTs（LangChain 的一个开源项目）令人惊叹的灵活性和强大功能。W...</li><li><a href="https://www.youtube.com/watch?v=ubsqSWfXAPI">如何使用 LangChain Output Parsers 和 GPT 将 PDF 转换为 JSON</a>：本视频教程演示了如何使用 LangChain 的 Output Parsers 和 GPT 将 PDF 转换为 JSON。像这样的任务过去很复杂，但现在可以...</li><li><a href="https://www.gettingstarted.ai/how-to-extract-metadata-from-pdf-convert-to-json-langchain/">这是如何使用 LangChain + GPT 将 PDF 转换为 JSON 的方法</a>：像将 PDF 转换为 JSON 这样的任务过去很复杂，但现在只需几分钟即可完成。在这篇文章中，我们将看到 LangChain 和 GPT 如何帮助我们实现这一目标。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1222572218517684274)** (1 条消息): 

- **gswithai 登陆 YouTube**：一位成员兴奋地分享了他们的第一个 YouTube 教程，介绍如何使用 **LangChain 的 Output Parsers** 和 GPT 将 PDF 转换为 JSON。该教程旨在简化过去复杂的任务，观看地址为 [如何使用 LangChain Output Parsers 和 GPT 将 PDF 转换为 JSON](https://www.youtube.com/watch?v=ubsqSWfXAPI)，他们正在寻求反馈以改进未来的内容。

**提到的链接**：<a href="https://www.youtube.com/watch?v=ubsqSWfXAPI">如何使用 LangChain Output Parsers 和 GPT 将 PDF 转换为 JSON</a>：本视频教程演示了如何使用 LangChain 的 Output Parsers 和 GPT 将 PDF 转换为 JSON。像这样的任务过去很复杂，但现在可以...

  

---



**Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1222521605742264483)** (30 条消息🔥): 

- **推出 DBRX，重量级选手**：MosaicML 和 Databricks 发布了 DBRX，这是一款全新的大语言模型 (LLM)，拥有惊人的 1320 亿参数（其中 320 亿为激活参数），以及卓越的 32k 上下文窗口。它采用商业许可，允许在特定条件下使用，并可在[此处](https://huggingface.co/databricks/dbrx-instruct)进行试用。

- **AI 扩展中的成本效率趋势**：一种被称为“Mosaic 定律”的趋势表明，由于硬件、软件和算法的改进，具有相似能力的模型所需的成本每年会缩减至四分之一，从而随着时间的推移大幅降低开发强大 AI 模型的费用。

- **AI21 发布 Jamba，融合 Mamba 与 Transformers**：AI21 发布了 Jamba，这是一种结合了 Mamba 的结构化状态空间模型 (SSM) 与传统 Transformer 架构特点的新模型，具有 256K 上下文窗口。Jamba 的开放权重受益于 Apache 2.0 许可证，鼓励在混合模型结构方面进行进一步开发，详情请见[此处](https://huggingface.co/ai21labs/Jamba-v0.1)。

- **对模型改进限制的担忧与澄清**：关于 DBRX 的许可条款出现了一个争论点，特别是禁止使用该模型及其衍生品来改进其他 LLM，这引发了对未来模型开发影响的讨论。

- **混合架构与条纹架构的探索**：新研究讨论了对超越 Transformer 架构的最大规模分析，揭示了条纹架构（striped architectures）通过专业化每层类型，通常优于同质架构，这可能会加速架构改进 [阅读论文](https://arxiv.org/abs/2403.17844) 并探索 [GitHub 仓库](https://github.com/athms/mad-lab)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/code_star/status/1772959109416980895?s=46">来自 Cody Blakeney (@code_star) 的推文</a>：*纠正一下，不是 open weights。这是一个商业友好许可的模型。请原谅我熬夜了 😅 欢迎下载并亲自尝试。https://huggingface.co/databricks/dbr...</li><li><a href="https://qwenlm.github.io/blog/qwen-moe/">Qwen1.5-MoE：以 1/3 的激活参数匹配 7B 模型性能</a>：GITHUB HUGGING FACE MODELSCOPE DEMO DISCORD 简介 自从 Mixtral 引发关注热潮以来，混合专家模型 (MoE) 的研究势头强劲。研究人员和...</li><li><a href="https://x.com/NaveenGRao/status/1772969283011920189">来自 Naveen Rao (@NaveenGRao) 的推文</a>：这是我们几年前观察到的一个普遍趋势。我们称之为 Mosaic 定律，即由于硬件/软件/算法的进步，具备特定能力的模型每年所需的成本将减少到 1/4。这...</li><li><a href="https://www.ai21.com/blog/announcing-jamba">介绍 Jamba：AI21 开创性的 SSM-Transformer 模型</a>：首次推出首个生产级基于 Mamba 的模型，提供同类最佳的质量和性能。</li><li><a href="https://fxtwitter.com/MichaelPoli6/status/1773370168929825073?s=20">来自 Michael Poli (@MichaelPoli6) 的推文</a>：📢关于机械架构设计和 Scaling Laws 的新研究。- 我们进行了迄今为止针对超越 Transformer 架构的最大规模 Scaling Laws 分析（500 多个模型，最高达 7B）- 首次...</li><li><a href="https://fxtwitter.com/andersonbcdefg/status/1773071904443629780">来自 Ben (e/sqlite) (@andersonbcdefg) 的推文</a>：所以你不能用 DBRX 来改进其他 LLM... 但他们从未说过你不能用它来让它们变得更糟</li><li><a href="https://x.com/code_star/status/1772956868773634254?s=46">来自 Cody Blakeney (@code_star) 的推文</a>：它终于来了 🎉🥳 如果你错过了我们，MosaicML/ Databricks 又回来了，推出了名为 DBRX 的新型同类最佳 open weight LLM。一个拥有 132B 总参数和 32B 激活参数、32k 上下文长度的 MoE...</li><li><a href="https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen2_moe">huggingface/transformers 仓库 main 分支下的 qwen2_moe 路径</a>：🤗 Transformers：适用于 Pytorch, TensorFlow 和 JAX 的前沿机器学习。- huggingface/transformers
</li>
</ul>

</div>
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1222895903157911585)** (18 条消息🔥): 

- **小语言模型的新地平线**：@liliang_ren 发布推文分享了影响力巨大的新闻，宣布加入 Microsoft GenAI 担任高级研究员，旨在开发高效且可外推的神经架构，重点是预训练 1000 亿参数以下的小语言模型。请关注该领域的更多进展。

- **Megablocks 找到新归宿**：@Tgale96 发推称将 Megablocks 项目移交给 Databricks，这标志着该项目在生产模型方面的未来发生了重大转变。参与者对此表示好奇，并希望在移交过程中能获得适当的认可。

- **辩论模型术语中的“小” (Small)**：成员们讨论了使用“小”来描述参数少于 1000 亿的语言模型是否合适，有些人认为这很合理，而另一些人则认为这有点厚脸皮且“不符合历史”。

- **热情与怀疑并存**：关于 Megablocks 转向 Databricks 的消息引发了讨论，人们思考这一举动背后的原因，并希望原作者获得了公平的报酬。

- **政府 AI 指南审查**：成员们注意到 OMB 关于 AI 的新指南包含“很多空话”，但也有一些有趣的方面，并对政府未来可能开放的数据集持乐观态度。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tgale96/status/1773342375806374307?s=46">来自 Trevor Gale (@Tgale96) 的推文</a>：有些人注意到 megablocks 现在变成了 databricks/megablocks。这周我把项目交给了他们，我想不出比这更好的长期归宿了。我期待着看到它成长...</li><li><a href="https://x.com/liliang_ren/status/1773118751413596588?s=46">来自 Liliang Ren (@liliang_ren) 的推文</a>：个人更新：我将于今年夏天加入 Microsoft GenAI 担任高级研究员，专注于下一代既高效又可外推的神经架构。我们正在...
</li>
</ul>

</div>
  

---



**DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1222549850369953914)** (8 条消息🔥):

- **Databricks 发布 DBRX Instruct**: [Databricks 发布 DBRX Instruct](https://huggingface.co/databricks)，这是一个拥有 1320 亿参数的新型**稀疏 MoE** 模型，在 12 万亿 tokens 上进行了训练，并专注于少轮交互。除了 DBRX Instruct，底层的预训练基座模型 DBRX Base 也以开放许可证发布，更多技术细节在其 [博客文章](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm) 中进行了讨论。
- **DBRX 机制的细节与分析**: 对 DBRX 架构的拆解突出了其独特元素，例如带有截断值（clamped values）的合并注意力机制、与 Llama 不同的非 RMS Layernorm，以及使用 OpenAI TikToken 的独特分词（tokenization）方法。分析还提到了损失平衡系数，以及通过 Bug 修复发现的来自 UnslothAI 的正确 RoPE upcasting，详见 [GitHub](https://github.com/databricks/dbrx/blob/main/model/modeling_dbrx.py)。
- **通过 Hugging Face 体验 DBRX Instruct**: 可以通过交互式的 [Hugging Face space](https://huggingface.co/spaces/databricks/dbrx-instruct) 体验 DBRX Instruct 的实际演示，该空间配备了旨在引导其回答风格的系统提示词。
- **DBRX Instruct 的指导原则**: 共享的系统提示词详细说明了 DBRX 在刻板印象、争议性话题、各种任务中的辅助、Markdown 的使用以及提供受版权保护信息的限制等方面的设计行为。
- **寻求学习 LLM 训练的资源**: 成员们讨论了学习如何微调或训练大语言模型的资源，并推荐了一个 GitHub 课程 [mlabonne/llm-course](https://github.com/mlabonne/llm-course)，该课程提供了路线图和 Colab notebooks。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/databricks/dbrx-instruct">DBRX Instruct - databricks 的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/databricks/dbrx-instruct">databricks/dbrx-instruct · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/danielhanchen/status/1772981050530316467">Daniel Han (@danielhanchen) 的推文</a>: 看了下 @databricks 的名为 DBRX 的新开源 1320 亿模型！1) 合并注意力 QKV 在 (-8, 8) 之间截断 2) 不是 RMS Layernorm - 现在具有均值移除，与 Llama 不同 3) 4 个激活专家...</li><li><a href="https://github.com/mlabonne/llm-course">GitHub - mlabonne/llm-course: 包含路线图和 Colab notebooks 的大语言模型 (LLMs) 入门课程。</a>: 包含路线图和 Colab notebooks 的大语言模型 (LLMs) 入门课程。 - mlabonne/llm-course
</li>
</ul>

</div>
  

---


**DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1222465967519567996)** (10 条消息🔥): 

- **免费访问 Mixtral API**: Mixtral 的 API 可以[通过 groq 免费访问](https://github.com/CrispStrobe/llm_translation.git)，唯一的限制是速率限制。虽然翻译质量不如 Deepl 或 AzureML 精细，但被认为适用于实验性用途。

- **使用 Occi 7B 的高质量翻译**: 用户报告称，使用 Occi 7B（特别是 `occiglot/occiglot-7b-de-en-instruct` 模型）进行的翻译提供了**高质量结果**，且没有诸如 token 插入之类的量化问题。

- **清理翻译内容**: 一位成员计划将 *slim orca* 数据集的大部分翻译成德语，然后**过滤掉错误的样本**，以提高整体质量。

- **协作翻译对比**: 社区有兴趣对比不同模型和服务（如 DisCoLM, Occiglot, Mixtral, GPT-4, Deepl 和 Azure Translate）对 *capybara* 等数据集的翻译效果，一些社区贡献的翻译已分享在 [Hugging Face](https://huggingface.co/datasets/cstr/Capybara-de-snippets) 上。

- **用于翻译的 GitHub 脚本**: 一位成员在 [GitHub](https://github.com/CrispStrobe/llm_translation.git) 上分享了一个用于翻译数据集的脚本，该脚本可用于促进提议的不同模型和服务翻译效果的对比。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/CrispStrobe/llm_translation.git">GitHub - CrispStrobe/llm_translation</a>: 通过创建账号为 CrispStrobe/llm_translation 的开发做出贡献。</li><li><a href="https://huggingface.co/datasets/cstr/Capybara-de-snippets">cstr/Capybara-de-snippets · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---



**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1222679719514997077)** (7 条消息):

- **DBRX 介绍**：Databricks 发布了 [DBRX](https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm)，这是一款全新的开放通用大语言模型 (LLM)，据称其性能超越了 GPT-3.5，并可与 Gemini 1.0 Pro 媲美。DBRX 在编程任务中表现出色，击败了 CodeLLaMA-70B 等模型，这得益于其细粒度的混合专家 (MoE) 架构，与竞争对手相比，它提供了更快的推理速度和更小的体积。
- **寻求简化**：一位成员请求对 DBRX 的技术层面进行更简单的解释，该模型声称在 LLM 效率和编程能力方面树立了新标准。
- **求知欲**：成员们询问 DBRX 编程性能的提升是数据集和架构的有意侧重，还是更通用方法的副产品。
- **DBRX 编程优势背后的原因**：解释称 DBRX 在编程任务中的卓越表现归功于其在 12 trillion tokens 上的广泛预训练、其 MoE 架构以及其课程学习 (curriculum learning) 模型，旨在**防止“技能相互干扰 (skill clobbering)”**。
- **编程协助请求**：一位成员寻求通过私信协助解决编程问题，表明需要个人支持。

**提到的链接**：<a href="https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm">Introducing DBRX: A New State-of-the-Art Open LLM | Databricks</a>：未找到描述

  

---



**LLM Perf Enthusiasts AI ▷ #[irl](https://discord.com/channels/1168579740391710851/1171569983688560732/1222645854666756249)** (4 条消息): 

- **温馨的代码与咖啡聚会**：Exa 团队邀请 LLM 爱好者参加本周六在 SF 举行的**快闪咖啡店和协作 (co-work) 活动**。欢迎参加由 Exa.ai 主办的活动，享受咖啡、抹茶、糕点并共同办公，时间从上午 10:30 开始，别忘了[加入名单](https://partiful.com/e/yaC2YSd4kYN7YQF6WVFx)以获取地点详情。
- **寻找专注于 LLM 的协作空间**：一位成员询问了 LLM 社区青睐的 SF 协作空间；另一位成员建议 **celo** 是一个热门地点。


**提到的链接**：<a href="https://partiful.com/e/yaC2YSd4kYN7YQF6WVFx">RSVP to Coffee + Cowork | Partiful</a>：大家好！Exa 团队很高兴本周六在我们的家庭办公室举办快闪咖啡店和协作活动！欢迎顺道来品尝高档咖啡/抹茶和早餐，或者带上笔记本电脑……

  

---



**Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/1222969782131884086)** (1 条消息): 

- **关于入职会议的查询**：一位成员询问了针对有兴趣贡献的资深 Python 爱好者的**下一场入门入职会议**，并回忆起之前提到过此类会议。
  

---


**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=LWz2QaSRl2Y
  

---