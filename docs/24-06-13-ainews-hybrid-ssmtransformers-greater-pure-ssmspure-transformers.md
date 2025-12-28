---
companies:
- nvidia
- lamini-ai
- sakana-ai
- luma-labs
date: '2024-06-13T20:52:25.343318Z'
description: '**NVIDIA** 的 Bryan Catanzaro 重点介绍了一篇关于 **Mamba 模型**的新论文，研究表明将 Mamba
  和 Transformer 模块混合使用，其性能优于单一架构，且最佳注意力机制占比低于 **20%**。**Mixture-of-Agents (MoA)**
  架构提升了大型语言模型（LLM）的生成质量，在 **AlpacaEval 2.0** 上的得分为 **65.1%**，超过了 **GPT-4 Omni 的 57.5%**。**LiveBench
  AI 基准测试**用于评估推理、编程、写作和数据分析能力。仅包含 **7% 注意力机制**的混合模型 **Mamba-2-Hybrid** 在 MMLU 准确率上超越了
  Transformer，从 **50% 提升至 53.6%**。**GPT-4** 在温度（temperature）设为 1 时表现更好。**Qwen 72B**
  在 LiveBench AI 榜单上领跑开源模型。**LaminiAI 的记忆微调（Memory Tuning）**在 SQL 代理任务中实现了 **95% 的准确率**，优于指令微调。**Sakana
  AI 实验室**利用进化策略进行偏好优化。**Luma Labs 的 Dream Machine** 展示了先进的文本生成视频技术。**MMWorld 基准测试**用于评估多模态视频理解能力，而
  **Table-LLaVa 7B** 在多模态表格任务中可与 GPT-4V 媲美。'
id: eab4785c-c252-4c7f-9cf6-1c8e4f50c74e
models:
- mamba-2-hybrid
- gpt-4
- qwen-72b
- table-llava-7b
original_slug: ainews-to-be-named-2494
people:
- bryan-catanzaro
- bindureddy
- ylecun
- ctnzr
- corbtt
- realsharonzhou
- andrew-n-carr
- karpathy
- _akhaliq
- omarsar0
title: 混合 SSM/Transformer 架构优于纯 SSM 或纯 Transformer。
topics:
- mixture-of-experts
- benchmarking
- fine-tuning
- multimodality
- text-to-video
- model-performance
- memory-optimization
- preference-optimization
- video-understanding
- multimodal-tables
---

<!-- buttondown-editor-mode: plaintext -->**7% 的 Transformers 就足够了。**

> 2024年6月12日至6月13日的 AI 新闻。
我们为您检查了 7 个 subreddits、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **30** 个 Discord（**414** 个频道，**3646** 条消息）。
预计节省阅读时间（按每分钟 200 字计算）：**404 分钟**。您现在可以标记 [@smol_ai](https://x.com/smol_ai) 进行 AINews 讨论！

今天有很多有趣的 [image-to-video](https://x.com/karpathy/status/1801305852735115357?utm_source=ainews&utm_medium=email) 和 [canvas-to-math](https://x.com/tldraw/status/1801264226314408029) 演示在流传，但技术细节不多，所以我们转向别处，关注 [NVIDIA 的 Bryan Catanzaro](https://x.com/ctnzr/status/1801050835197026696?utm_source=ainews&utm_medium=email) 提到的关于研究 Mamba 模型的高质量[新论文](https://arxiv.org/pdf/2406.07887)：

 
![image.png](https://assets.buttondown.email/images/77bce511-0f83-4f5a-88a1-9cdc8878c4f9.png?w=960&fit=max)
 

正如 Eugene Cheah 在 Latent Space Discord 中指出的，这是继 [Jamba](https://buttondown.email/ainews/archive/ainews-jamba-mixture-of-architectures-dethrones/) 和 [Zamba](https://x.com/QuentinAnthon15/status/1780280071304937978) 之后，第三个独立发现将 Mamba 和 Transformer 块混合使用效果优于单一架构的团队。而且该论文通过实验得出结论，Attention 的最佳比例小于 20%，远非“all you need”（你所需要的全部）。

 
![image.png](https://assets.buttondown.email/images/4cf09552-b0ef-4051-8eac-70ce04b41940.png?w=960&fit=max)
 

 
![image.png](https://assets.buttondown.email/images/26f043f7-4f99-41be-b017-9443dab35eee.png?w=960&fit=max)
 

---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI 推特摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**LLM 能力与评估**

- **Mixture-of-Agents 增强 LLM 性能**：[@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) 指出，Mixture-of-Agents (MoA) 在分层架构中使用多个 LLM 来迭代增强生成质量。使用开源 LLM 的 MoA 设置在 **AlpacaEval 2.0 上得分 65.1%，而 GPT-4 Omni 为 57.5%**。
- **LiveBench AI 基准测试**：[@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) 和 [@ylecun](https://twitter.com/ylecun/status/1800897325759701489) 宣布了 LiveBench AI，这是一个包含无法被记忆的挑战的新 LLM 基准测试。它在推理、编码、写作和数据分析方面评估 LLM，旨在提供独立、客观的排名。
- **Mamba-2-Hybrid 性能超越 Transformer**：[@ctnzr](https://twitter.com/ctnzr/status/1801050835197026696) 分享了一个使用 7% attention 的 8B-3.5T 混合 SSM 模型，在相同数据集上获得了比 8B-3.5T Transformer 更好的准确率，**MMLU 从 50% 跃升至 53.6%**，同时具有相同的训练效率和更低的推理成本。
- **GPT-4 在 Temperature=1 时表现更佳**：[@corbtt](https://twitter.com/corbtt/status/1801026166020833457) 根据他们的评估发现，即使在确定性任务上，GPT-4 在 temperature=1 时也比 temperature=0 时更“聪明”。
- **Qwen 72B 领跑开源模型**：[@bindureddy](https://twitter.com/bindureddy/status/1801010849160818701) 指出，Qwen 72B 是 LiveBench AI 上表现最好的开源模型。

**LLM 训练与微调**

- **Memory Tuning 实现 95% 以上的准确率**：[@realSharonZhou](https://twitter.com/realSharonZhou/status/1801271891954696317) 宣布了 @LaminiAI Memory Tuning，它使用多个 LLM 作为 Mixture-of-Experts (MoE) 来迭代增强基础 LLM。一家财富 500 强客户的案例研究显示，**在 SQL agent 任务上准确率达到 95%，而仅靠指令微调（instruction fine-tuning）时仅为 50%**。
- **Sakana 的进化 LLM 优化**：[@andrew_n_carr](https://twitter.com/andrew_n_carr/status/1801080453426024534) 强调了 Sakana AI Lab 的工作，即使用进化策略来发现用于偏好优化（preference optimization）的新损失函数，性能超越了 DPO。

**多模态与视频模型**

- **Luma Labs Dream Machine**：[@karpathy](https://twitter.com/karpathy/status/1801305852735115357) 等人注意到了 Luma Labs 新的 Dream Machine 模型令人印象深刻的文本生成视频（text-to-video）能力，该模型可以将图像扩展为视频。
- **MMWorld 基准测试**：[@_akhaliq](https://twitter.com/_akhaliq/status/1801077422676205708) 介绍了 MMWorld，这是一个用于在多学科、多维度视频理解任务上评估多模态语言模型的基准测试。
- **用于多模态表格的 Table-LLaVa**：[@omarsar0](https://twitter.com/omarsar0/status/1801271773796716646) 分享了用于多模态表格理解的 Table-LLaVa 7B 模型，该模型与 GPT-4V 具有竞争力，并在多个基准测试中超越了现有的 MLLM。

**开源模型与数据集**

- **用于图像标注的 LLaMA-3**：[@_akhaliq](https://twitter.com/_akhaliq/status/1801076206604783888) 和 [@arankomatsuzaki](https://twitter.com/arankomatsuzaki/status/1801073353899261982) 强调了一篇论文，该论文微调了 LLaVA-1.5，使用 LLaMA-3 对 DataComp-1B 数据集中的 13 亿张图像进行重新标注（recaption），展示了对训练视觉语言模型的益处。
- **Stable Diffusion 3 发布**：[@ClementDelangue](https://twitter.com/ClementDelangue/status/1800908975409656014) 等人注意到了 Stability AI 发布了 Stable Diffusion 3，它迅速成为 Hugging Face 上排名第一的热门模型。
- **Hugging Face 收购 Argilla**：[@_philschmid](https://twitter.com/_philschmid/status/1801274502879273009) 和 [@osanseviero](https://twitter.com/osanseviero/status/1801260106702590375) 宣布，数据集创建和开源贡献领域的领先公司 Argilla 正在加入 Hugging Face，以增强数据集的创建和迭代。

---

# AI Reddit 回顾

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**Stable Diffusion 3 Medium 发布**

- **资源高效型模型**：在 /r/StableDiffusion 中，Stable Diffusion 3 Medium 权重已发布，这是一个拥有 2B 参数的模型，[**资源效率高，能够在消费级 GPU 上运行**](https://www.reddit.com/r/StableDiffusion/comments/1de2qne/announcing_the_open_release_of_stable_diffusion_3/)。
- **相比前代模型的改进**：SD3 Medium [**克服了手部和面部的常见伪影，能理解复杂的提示词，并实现了高质量的文本渲染**](https://www.reddit.com/r/StableDiffusion/comments/1de2qne/announcing_the_open_release_of_stable_diffusion_3/)。
- **新的许可条款**：在 /r/OpenAI 中，Stability AI 宣布了 SD3 的新许可条款：[**非商业用途免费，有限商业用途需支付每月 20 美元的 Creator License，全面商业用途则需定制价格**](https://www.reddit.com/r/OpenAI/comments/1debbp5/stability_ai_unveils_new_advanced_image_generator/)。
- **褒贬不一的初步反馈**：首批测试者报告了不同的体验，一些人在复现结果时遇到问题，而另一些人则对 [**提示词遵循能力、细节丰富度以及光影/色彩给出了正面评价**](https://www.reddit.com/r/OpenAI/comments/1debbp5/stability_ai_unveils_new_advanced_image_generator/)。

**SD3 Medium 的问题与局限性**

- **人体解剖结构表现不佳**：在 /r/StableDiffusion 中，用户报告 SD3 Medium [**在人体解剖结构方面表现挣扎，特别是在生成躺下或特定姿势的人像时**](https://www.reddit.com/r/StableDiffusion/comments/1deav7h/sd3_has_sd_20_level_censorship/)。关于该模型局限性的更多细致想法在[此处进一步讨论](https://www.reddit.com/r/StableDiffusion/comments/1dehg03/some_nuanced_thoughts_on_stable_diffusion_3/)。
- **严重的审查**：该模型似乎经过了 [**严格的审查，导致在生成裸露或暗示性内容时表现糟糕**](https://www.reddit.com/r/StableDiffusion/comments/1deav7h/sd3_has_sd_20_level_censorship/)。
- **难以处理艺术风格**：SD3 Medium [**难以遵循艺术风格和概念，往往会生成写实风格的图像**](https://www.reddit.com/r/StableDiffusion/comments/1dekudj/sd3_artist_styles_and_concepts/)。

**与其他模型的对比**

- **各有优劣**：SD3 Medium、SDXL 以及 Stable Cascade 和 PixArt Sigma 等其他模型的对比显示，[**在不同类型的图像（写实、绘画、风景、漫画艺术）中各有优劣**](https://www.reddit.com/r/StableDiffusion/comments/1deb7hb/comparison_of_sd3_and_stable_cascade_a_woman/)。额外的[对比图集](https://www.reddit.com/r/StableDiffusion/comments/1deccwa/image_gen_comparisons_4_sets_each_with_sd3_sdxl/)进一步强调了这些差异。
- **在特定领域表现出色**：SD3 Medium [**在某些领域优于其他模型，例如生成云朵或文本**](https://www.reddit.com/r/StableDiffusion/comments/1deb8mu/sd3_cant_produce_a_cloud_in_shape_of_a_cat_while/)，但在 [人体解剖结构等其他方面表现不足](https://www.reddit.com/r/StableDiffusion/comments/1dedjtv/sd3_is_good_at_text/)。

**社区反应与推测**

- **对发布的失望**：/r/StableDiffusion 的许多用户表达了 [**对 SD3 Medium 发布的失望，理由是解剖结构、审查制度以及缺乏艺术风格等问题**](https://www.reddit.com/r/StableDiffusion/comments/1deaahg/sd3_dead_on_arrival/)。有些人甚至 [称其为“笑话”](https://www.reddit.com/r/StableDiffusion/comments/1de9wfz/sd3_is_a_joke/)。
- **对原因的推测**：一些用户 [**推测表现不佳可能是由于权重适配中的 Bug 或模型架构造成的**](https://www.reddit.com/r/StableDiffusion/comments/1depcxv/are_we_sure_its_not_a_bug_in_adopting_the_weights/)。
- **依赖微调**：其他人建议 [**社区需要依靠微调和自定义数据集来提升 SD3 的能力，就像对待之前的模型一样**](https://www.reddit.com/r/StableDiffusion/comments/1deebnz/for_those_disappointed_with_sd3/)。

**梗图与幽默**

- **调侃缺点**：用户分享了 [**调侃 SD3 缺点的梗图和幽默图片，特别是它无法生成解剖结构正确的人体**](https://www.reddit.com/r/StableDiffusion/comments/1de9xt6/sd3_api_vs_sd3_local_i_dont_get_what_kind_of/)。有些人甚至 [讽刺地声称该模型取得了“巨大成功”](https://www.reddit.com/r/StableDiffusion/comments/1deano8/huge_success_with_sd3/)。

---

# AI Discord 摘要

> 摘要之摘要的摘要

1. **Stable Diffusion 3 面临审查但提供了替代方案**：

   - **SD3 因模型质量面临批评**：用户对 SD3 表示不满——强调了其解剖结构不准确和提示词（prompt）问题——而中型模型可以在 [Huggingface 上下载](https://huggingface.co/stabilityai/stable-diffusion-3-medium)。
   - **讨论了首选界面与工具**：**ComfyUI** 成为最受青睐的界面，并建议使用 **uni_pc** 和 **ddim_uniform** 等采样器以获得最佳性能。**Juggernaut Reborn** 和 [Playground](https://playground.com) 等替代方案因其特定功能而受到关注。

2. **提升 AI 性能与基础设施洞察**：

   - **更高的模型秩（Rank）提升 LLM 性能**：将 Rank 从 16 提升到 128 解决了 **Qwen2-1.5b** 的乱码输出问题，使其输出达到 **llama-3** 的水准。
   - **Perplexity AI 的高效 LLM 使用**：通过利用 **NVIDIA A100 GPU**、**AWS p4d 实例**和 [TensorRT-LLM 优化](https://www.perplexity.ai/hub/blog/introducing-pplx-api)实现了快速响应。

3. **微调与量化方面的创新**：

   - **使用新模型微调 LLM**：讨论涉及了使用 GPT 生成数据的法律层面，引用了 **OpenAI 的商业条款**。针对 [ToolkenGPT](https://arxiv.org/abs/2305.11554) 的实验展示了用于微调的合成数据的创新方法。
   - **CUDA 量化项目讨论**：如 [BiLLM](https://github.com/Aaronhuang-778/BiLLM) 等项目展示了大模型的快速量化，这对于高效的 AI 部署至关重要。

4. **模型管理与部署技术**：

   - **处理大型 Embedding 的策略**：针对 **170,000 个 Embedding 索引**的咨询，建议使用 **Qdrant** 或 FAISS 以实现更快的检索。针对错误查询的具体修复方案已在[此处](https://docs.llamaindex.ai)分享。
   - **Docker 与 GPU 配置故障排除**：在 WSL 上处理 Docker GPU 检测的用户通过参考官方 [NVIDIA toolkit 指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)找到了解决方案。

5. **AI 社区趋势与更新**：

   - **OpenAI 的营收里程碑与重心转移**：OpenAI 的营收翻了一番，反映了直接来自 ChatGPT 和其他服务的销售额，而非主要由 Microsoft 促成（[来源](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx)）。
   - **合作伙伴关系与会议吸引社区参与**：Aleph Alpha 和 Silo AI 联手推动欧洲 AI 发展（[阅读更多](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/)），而 [Qwak 的免费虚拟会议](https://tinyurl.com/j8z6s8ka)承诺将深入探讨 AI 机制并提供社交机会。

---

# 第一部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

**SD3 坎坷的发布**：用户对 **Stable Diffusion 3 (SD3)** 表示不满，理由是与 **SDXL** 和 **SD1.5** 相比，它存在解剖结构不准确和不遵循提示词等问题。尽管存在批评，SD3 的中型模型现在已可在 [Huggingface](https://huggingface.co/stabilityai/stable-diffusion-3-medium) 上下载，但需要填写表格才能访问。

**首选界面与采样器**：**ComfyUI** 是目前运行 **SD3** 的首选界面，用户建议不要使用 Euler 采样器。为了让 SD3 达到峰值性能，最受青睐的采样器是 **uni_pc** 和 **ddim_uniform**。

**探索替代方案**：频道参与者强调了 **Juggernaut Reborn** 和 **Divinie Animemix** 等替代模型和工具，分别用于实现更强的写实感或动漫风格。其他资源包括用于管理和部署模型的 [Playground](https://playground.com/) 和 [StableSwarm](https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#installing-on-windows)。

**保持讨论的相关性**：在有关全球政治和个人轶事的讨论偏离了技术 AI 话题后，管理员不得不引导对话回到正轨。

**大模型，大需求**：SD3 的 10GB 模型被提及为社区中备受追捧的选择，这表明尽管 SD3 发布后的评价褒贬不一，但用户仍渴望更大、更强大的模型。

---

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **通过提高 Rank 提升模型性能**：将模型 Rank 从 16 增加到 128 解决了 **Qwen2-1.5b** 在训练期间产生乱码的问题，使输出质量与 **llama-3** 的训练结果保持一致。

- **scGPT 的实际应用受限**：尽管 Prompting 和 Tokenizer 的实现很有趣，但 **scGPT**（一个用 PyTorch 编写的自定义 Transformer）被认为在学术环境之外并不实用。

- **拥抱 Unsloth 以实现高效推理**：实施 **Unsloth** 显著降低了训练和推理过程中的内存占用，为人工智能模型提供了一个更节省内存的解决方案。

- **Mixture of Agents (MoA) 令人失望**：Together AI 提出的 **MoA** 方法旨在分层语言模型 Agent，但因过于复杂且更像是一个展示品而非实用工具而受到批评。

- **推进 LLM 的 Docker 集成**：AI 工程师建议创建命令行界面 (CLI) 工具以简化工作流，并将 Notebook 与 **ZenML** 等框架更好地集成，从而在实际应用中取得实质性成果。



---



## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**SD3 革新 Stable Diffusion**：**Stable Diffusion 3 (SD3)** 已发布并带来大量增强功能——现在配备了三个强大的文本编码器（[CLIP L/14](https://huggingface.co/openai/clip-vit-large-patch14), [OpenCLIP bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k), [T5-v1.1-XXl](https://huggingface.co/google/t5-v1_1-xxl)）、一个 Multimodal Diffusion Transformer 和一个 16 通道的 AutoEncoder。SD3 实现的细节可以在 [Hugging Face 博客](https://huggingface.co/blog/sd3#summary-of-memory-optimizations)中找到。

**应对 SD3 挑战**：用户在不同平台上遇到了 **SD3** 的困难，建议包括应用 `pipe.enable_model_cpu_offload()` 以加快推理速度，并确保安装了 `sentencepiece` 等依赖项。GPU 设置技巧包括使用 **RTX 4090**、采用 fp16 精度以及确保路径格式正确。

**Hugging Face 家族扩展，迎来 Argilla**：在一个令人兴奋的事件中，Hugging Face 将 **Argilla** 纳入麾下，这一举动受到了社区的欢迎，认为其有潜力推进开源 AI 计划和新的合作。

**社区与支持在行动**：从大学（如 Hugging Face 上新创建的 [University of Glasgow 组织](https://huggingface.co/UniversityofGlasgow)）到个人贡献（如 **LLM** 的 Google Colab 教程），成员们一直在为各种 AI 项目贡献资源并寻求支持。

**通过共享资源丰富学习**：成员们正在积极交流知识，重点资源包括 [Google Colab 上的 LLM 设置教程](https://github.com/casualcomputer/llm_google_colab)、关于文本生成图像模型 **MaPO** 技术的拟议阅读小组讨论，以及一篇阐明 PCFGs 的 [NLP 学术论文](https://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf)。



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **通往 AI 专家之路**：有抱负的 AI 工程师被引导至 **Andrej Karpathy** 的资源和 [sentdex 的 YouTube 系列视频](https://www.youtube.com/watch?v=GZqYr8_Q7DE)，内容关于创建深度学习聊天机器人。讨论围绕 AI 工程职业所需的知识和技能展开。

- **GPT-4.5 Turbo 推测引发辩论**：一个被辩论的话题是泄露的 **GPT-4.5 Turbo** 提及，推测其具有 256k context window 和 2024 年 6 月的知识截止日期。这引发了对其潜在 continuous pretraining 功能的推测。

- **揭秘 ChatGPT 的存储策略**：有人建议 ChatGPT 内部更好的内存管理可能涉及集体记忆总结和清理技术，以解决当前的局限性。

- **团队合作让梦想成真**：分享了关于 ChatGPT Team 账户的关键点，强调了双倍的 Prompt 限制以及在不按年计费时多个团队席位的财务支出。

- **分解大数据**：对于管理 **300MB** 文件等大量文本数据，给出了通过 chunking 和修剪以提高实用性的建议。链接了有用的工具和指南，包括一篇包含处理大型文档实用技巧的 [论坛帖子](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2) 和一本关于通过 embeddings 处理长文本的 [Notebook](https://cookbook.openai.com/examples/embedding_long_inputs)。



---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **LLM 微调：添加新知识**：AI 爱好者讨论了微调像 "Nous Hermes" 这样的 LLM 以引入新知识，尽管成本较高。关于使用 GPT 生成的数据引发了法律辩论，用户查阅了 [OpenAI 的商业条款](https://openai.com/policies/business-terms/)；另外还提到了 [ToolkenGPT 论文](https://arxiv.org/abs/2305.11554) 中引用的合成数据生成。

- **技术故障与建议**：在 LLM 领域，用户报告了 llama-3-8b 和 mistral-7b 等模型的 **preprocessing errors**（预处理错误）。在实践方面，成员们交流了通过 `nohup` 维持 SSH 连接的技巧，相关建议可参考此 [SuperUser 线程](https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal)。

- **备受关注的创新模型框架**：模型框架引起了关注，**LangChain** 和 **LangGraph** 引发了不同的观点。[glaive-function-calling-v1](https://huggingface.co/glaiveai/glaive-function-calling-v1) 的推出引发了关于模型函数执行能力的讨论。

- **Hugging Face Spaces 中的部署与展示**：几位用户宣布了他们的 RAG 应用，例如使用 Gradio 构建并托管在 Hugging Face Spaces 上的 [RizzCon-Answering-Machine](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine)，尽管有人指出需要提高速度。

- **积分与资源追寻仍在继续**：关于 OpenPipe 等平台积分缺失以及联系人的咨询不断。未收到积分的用户分享了他们的用户名（例如 *anopska-552142*, *as-ankursingh3-1-817d86*），并提到预计在 14 号进行第二轮积分发放。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **无需人类专家的 LLM 目标发现**：一篇 [arXiv 论文](https://arxiv.org/abs/2406.08414) 详细介绍了一种由模型自身驱动的 LLM 优化算法发现方法，这可以在不需要人类专家输入的情况下简化 LLM 偏好优化。该方法采用对 LLM 进行迭代提示（iterative prompting），以根据指定指标提升性能。
  
- **MoA 超越 GPT-4 Omni**：正如一篇 [Hugging Face 论文](https://huggingface.co/papers/2406.04692) 所强调的，Mixture-of-Agents (MoA) 架构显示，结合多个 LLM 可以提升性能，在 AlpacaEval 2.0 上以 65.1% 的得分超越了 GPT-4 Omni。对于想要贡献或深入研究的 AI 爱好者，MoA 模型的实现已在 [GitHub](https://github.com/togethercomputer/moa) 上发布。

- **Stable Diffusion 3：初印象褒贬不一**：虽然 Stable Diffusion 3 在初始发布中既获得了掌声也遭到了批评，但关于 GPT-4 在较高 temperature 设置下反而表现更好的反直觉讨论，加剧了关于模型配置的辩论。相反，一位社区成员传阅了 [OpenHermes-2.5 数据集的无审查版本](https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored)，而一篇关于 [消除 MatMul 操作](https://arxiv.org/abs/2406.02528) 的论文承诺将带来显著的内存节省。

- **寻找遗失的论文**：社区成员正积极寻找一篇关于 **预训练与指令交替 (interleaving pretraining with instructions)** 的被遗忘论文，这表明社区频道内对前沿研究分享有着浓厚兴趣。

- **RAG 数据集开发持续进行**：RAG 的数据集架构（schema）仍在完善中，Marker 的文档转换工具还需要进一步优化，其中设置 min_length 可能会提高处理速度。同时，Pandoc 和 make4ht 成为处理各种文档类型的可能转换方案。

- **World-sim 项目现状**：尽管有讨论和未来重新考虑的潜力，World-sim 项目的闭源状态尚未改变。此外，要求让 world-sim AI 变得更大胆并将其适配到移动平台的呼声，反映了社区的前瞻性想法。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **对 Perplexity 搜索更新的热情**：成员们对 Perplexity AI 最近推出的 **搜索功能 (search feature)** 表现出极大的兴奋，并立即表达了对 **iOS 版本** 的兴趣。

- **马斯克与 OpenAI 的法律战告一段落**：埃隆·马斯克在法庭听证会的前一天，**撤回**了对 **OpenAI** 的诉讼。该诉讼指控 OpenAI 从使命驱动转向了利润导向，并声称其优先考虑 **Microsoft** 等投资者的利益 ([CNBC](https://www.cnbc.com/2024/06/11/elon-musk-drops-suit-against-openai-and-sam-altman.html))。

- **Perplexity AI 在大语言模型上的速度**：尽管使用了大型语言模型，**Perplexity.ai** 仍通过利用 **NVIDIA A100 GPU**、**AWS p4d 实例** 以及 **TensorRT-LLM** 等软件优化，实现了快速的响应结果 ([Perplexity API 介绍](https://www.perplexity.ai/hub/blog/introducing-pplx-api))。

- **Perplexity 上自定义 GPT 的困扰**：工程师们正面临 **Custom GPTs** 的**连接问题**；问题似乎仅限于平台的 Web 版本，因为桌面应用程序上没有相关故障报告，这表明可能存在 **API** 或特定平台的复杂问题。

- **电子邮件的环境足迹**：一封电子邮件平均排放约 **4 克二氧化碳 (CO2)**；通过优先使用**文件共享链接**而非附件，可以减轻碳排放影响 ([Mailjet 电子邮件碳足迹指南](https://www.mailjet.com/blog/email-best-practices/email-carbon-footprint/))。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

**计算强度讨论悬而未决**：一位成员询问 **计算强度 (compute intensity)** 的计算是否应仅考虑来自 **Global Memory** 数据的浮点运算。该话题仍处于讨论阶段，尚未得出结论性答案。

**精简的 Triton 3.0 安装**：出现了两种实用的 **Triton 3.0** 安装方法；一份指南详细介绍了[从源码安装](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/)，而另一种方法涉及使用 `make triton` 并配合 [PyTorch 仓库中的特定版本](https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt)。

**在 PyTorch 中优化优化器**：社区进行了一场深入的对话，讨论如何使用 **纯 PyTorch** 和 **torch.compile** 创建快速的 8-bit 优化器，并制作一个精度可与 32-bit 媲美的无缝替换方案，灵感源自 [bitsandbytes 的实现](https://arxiv.org/pdf/2110.02861)。

**量化和训练动态方面的突破**：[BiLLM 项目](https://github.com/Aaronhuang-778/BiLLM) 宣称能够对大语言模型进行快速量化，同时 **torchao** 的成员们正在辩论矩阵乘法过程中各种数值表示（从 **INT8** 到 **FP8** 甚至 **INT6**）在速度和精度之间的权衡。

**硬件对决与量化创新**：AMD 的 **MI300X** 在 LLM 推理方面展示了比 NVIDIA H100 更高的吞吐量。**Bitnet** 在重构和每日构建策略方面取得了进展，但由于一个*无关的 mx 格式测试*，仍存在一个悬而未决的构建问题。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Gemini 1.5 JSON 模式的问题**：工程师们报告称 **Gemini 1.5 flash** 在 **JSON 模式** 下表现不佳，导致输出出现间歇性问题。社区邀请用户分享对此挑战的见解或解决方案。

**Tess 登场**：**Tess 2.5 72b q3** 和 **q4 量化模型现已在 Hugging Face 上线**，为实验提供了新工具。

**AVX2 指令是必需的**：遇到直接 AVX2 错误的用户应验证其 **CPU 是否支持 AVX2 指令**，以确保与应用程序要求的兼容性。

**LM Studio 的限制与解决方案**：**LM Studio** 无法在无头 (headless) Web 服务器上运行，也不支持 safetensor 文件，但它成功采用了 **GGUF 格式**，并且可以通过 llama.cpp 等替代方案启用 **Flash Attention**。

**硬件市场波动**：**电子报废的 P40 GPU** 价格飙升，目前价格超过 200 美元；此外还有一条幽默的消息称制裁可能影响了**俄罗斯的 P40 库存**。一位社区成员分享了一套高效的**家用服务器配置**：R3700X、128GB RAM、RTX 4090 以及多种存储选项。

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **LLAMA3 70B 展示多样化才能**：**LLAMA3 70B** 表现出广泛的输出能力，在从空白文档进行提示时，生成了 60% 的 AI2 arc 格式、20% 的 wiki 文本和 20% 的代码，这表明其针对特定格式进行了微调。在另一个查询中，有关于使用滑动窗口技术对 **BERT** 进行长文本 finetuning 的指导，并指向了相关资源，如 [NeurIPS 论文及其实现](https://github.com/Sleepychord/CogLTX)。

- **Samba 表现优于 Phi3-mini**：**Microsoft 的 Samba 模型**在 3.2 万亿 token 上进行了训练，在 benchmark 中显著优于 Phi3-mini，同时保持了线性复杂度并实现了卓越的长上下文检索能力。另一场讨论深入探讨了 **Samba** 在 256k 序列中的 **passkey retrieval**，讨论了 Mamba 层和 SWA 的有效性。

- **Magpie 展露头角**：新推出的 **Magpie** 方法提示对齐的 Large Language Models (LLMs) 自动生成高质量的指令数据，从而规避了手动数据创建。沿着这一创新前沿，另一场讨论强调了将 embedding 和 unembedding 层绑定的争议性做法，并分享了来自 [LessWrong 文章](https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are) 的见解。

- **辩论归一化标准**：在社区内部，评估模型的指标引发了辩论，特别是对于具有相同 tokenizer 的模型，是按 **token 还是按 byte 归一化准确率**。分享了一个[关于 Qwen1.5-7B-Chat 测试的相关日志](https://pastebin.ai/i6qnlbg8x3)，讨论了解决 `truthfulqa_gen` 任务中空响应故障的方案。

- **Open Flamingo 展翅高飞**：一条简短的消息将成员引向 [LAION 关于 Open Flamingo 的博客文章](https://laion.ai/blog/open-flamingo/)，这可能是对其多模态模型工作的引用。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**GitHub 上的 TiDB AI 实验**：PingCap 展示了一个使用其 TiDB 数据库与 LlamaIndex 知识图谱结合的 **RAG 应用**，所有内容均作为开源代码提供，并在 GitHub 上附有 [demo](https://t.co/JKOa6ab1Uh) 和 [源代码](https://t.co/bUWs9lM1ea)。

**巴黎 AI 基础设施见面会诚邀参加**：工程师们可以参加在巴黎 Station F 举行的 **AI Infrastructure Meetup**，演讲者来自 LlamaIndex、Gokoyeb 和 Neon；详情和报名请点击[此处](https://twitter.com/llama_index/status/1801288913312760205)。

**快速查询的向量数据库解决方案**：对于包含 170,000 个 embedding 的索引，建议使用 **Qdrant** 或 FAISS Index；讨论内容包括修复与 FAISS 查询相关的 `AssertionError`，以及使用 **Chroma** 从 VectorStoreIndex 直接进行节点检索。

**从 Qdrant 检索相邻节点**：
一位用户询问如何在 Qdrant 向量库中获取法律文本的相邻节点，建议利用节点关系和最新的 API 功能进行定向节点检索。

**利用 PDF Embedding 提升 LLM-Index 能力**：一位 AI 工程师讨论了使用 LLM-Index 将 PDF 和文档嵌入到 Weaviate 中，表现出对扩展向量数据库摄取复杂数据类型的兴趣。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **Command-R 登场**：**Coral** 已更名为 **Command-R**，但 Command-R 和原始的 Coral 仍在运行，以促进模型相关任务。
- **调优还是不调优**：在追求最佳模型性能的过程中，辩论激烈，一些工程师强调 **prompt engineering** 优于参数微调，而另一些人则交换了值得注意的配置。
- **解读 Cohere 的可接受使用政策**：注意到大家在共同努力解读 **[Cohere Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)** 的细微差别，重点在于界定个人项目背景下私用与商用的区别。
- **测试密钥的波折**：社区交流了关于 **trial keys** 遇到权限问题和限制的挫败感，并将这些经历与正式版密钥用户报告的更顺畅体验进行了对比。
- **向 Fluent API 支持致敬**：对话中简要提到了对 **Fluent API** 的偏好，并对其被 Cohere 采纳表示赞赏，最近一个包含 Cohere 支持的项目发布也体现了这种祝贺基调。

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **Stable Diffusion 中的性别失衡困扰**：社区讨论了 **Stable Diffusion** 由于审查制度在生成女性图像（无论是否着装）方面存在问题，建议使用自定义 checkpoint 和基于 SD1.5 的 img2img 技术作为变通方案。点击此处查看 [讨论线程](https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67)。

- **Dream Machine 首次亮相**：**Luma AI** 的 **Dream Machine**（一款文本转视频模型）已经发布，其潜力引发了热烈讨论，尽管用户注意到它在处理复杂 prompt 时表现不稳定。在此处查看该 [模型](https://lumalabs.ai/dream-machine)。

- **AI 领域概览**：讨论了 **SD3 Large、SD3 Medium、Pixart Sigma、DALL E 3 和 Midjourney** 等模型的对比，以及 /r/StableDiffusion 子版块的重新开放和 Reddit 的 API 变更。社区正密切关注这些模型和问题，该 [Reddit 帖子](https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/) 提供了对比。

- **模型不稳定性曝光**：一项研究显示，像 GPT-4o 这样的模型在面对输入稍作修改的**爱丽丝梦游仙境（Alice in Wonderland）**场景时会发生剧烈崩溃，凸显了推理能力中的重大问题。详情可见 [论文](https://arxiv.org/abs/2406.02061)。

- **重新标注网页**：AI 生成的嘈杂网页图像标注增强即将到来，**DataComp-1B** 旨在通过更好地对齐文本描述来改进模型训练。如需进一步了解，请查看 [综述](https://www.haqtu.me/Recap-Datacomp-1B/) 和 [科学论文](https://arxiv.org/abs/2406.08478)。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **追求最佳语音转文本方案**：工程师们讨论了 **speech-to-text** 解决方案，在 **AWS Transcribe**、**OpenAI Whisper** 和 **Deepgram Nova-2** 等工具之外，寻找带有 MP3 和角色分离（diarization）的数据集。还强调了对鲁棒处理的需求，即在不使用工具的情况下处理简单响应，并管理流式响应（streaming responses）而不丢失上下文。

- **LangChain 连接链与状态**：在 **LangChain AI** 中，明确讨论了在状态管理中集成用户 ID 和线程 ID，并分享了利用 **LangGraph** 在各种交互中简洁维护状态的技巧。对于 LangChain 内的消息相似度检查，建议结合实际用例使用字符串和 embedding 距离指标。

- **为所有人简化 LLM**：展示了一个名为 [tiny-ai-client](https://github.com/piEsposito/tiny-ai-client) 的 **GitHub** 项目以简化 LLM 交互，一个 [YouTube 教程](https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy) 展示了如何使用 Docker 和 Ollama 设置 LLM 的本地执行。同时，另一位成员分享了一个 [GitHub 教程](https://github.com/casualcomputer/llm_google_colab) 介绍如何利用 15GB Tesla T4 GPU 在 **Google Colab** 上设置 LLM。

- **简化开发的代码示例与对话**：在整个讨论中，引用了各种**代码示例**和问题来辅助排查故障并简化 LLM 开发流程，包括 [聊天机器人反馈模板](https://python.langchain.com/v0.2/docs/templates/chat-bot-feedback/#usage) 等链接，以及在 LangChain 中*评估现成评估器*和*维护问答聊天历史*的方法论。

- **社区知识共享**：成员们积极分享自己的作品、方法和解决问题的策略，建立了一个包含操作指南和非平凡 LangChain 场景响应的社区知识库，肯定了工程社区的协作精神。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **Windows 的困境与对策**：工程师们讨论了 **Modular (Mojo 🔥)** 对 Windows 的支持，预计将在秋季发布，并期待[直播更新](https://m.youtube.com/watch?v=uookgZ7Ojg8)。同时，一些人已转向使用 **WSL** 作为 Mojo 开发的临时解决方案。

- **Mojo 字符串的真值特性**：有人指出 **Mojo 中的非空字符串** 被视为真值（truthy），这可能会在代码逻辑中导致意外结果。同时，可以在 **Neovim** 中设置 **Mojo LSP**，相关配置可在 [GitHub](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo) 上找到。

- **优化矩阵运算**：基准测试显示，在小型固定尺寸的**矩阵乘法**中，**Mojo** 的性能优于 **Python**，这归功于 Python 的 numpy 在此类任务中具有较高的开销。

- **循环逻辑与输入处理**：Mojo 中 `for` 循环的特殊行为导致有人建议在需要变量重新赋值的迭代中使用 `while` 循环。此外，确认了 Mojo 目前缺乏对 `stdin` 的支持。

- **Nightly 版本更新与编译器见解**：**Mojo 编译器版本 `2024.6.1305`** 发布，引发了关于更新程序的讨论，建议使用 `modular update nightly/max` 并考虑使用别名（aliases）来简化操作。讨论还涉及了编译器的局限性，以及 **ExplicitlyCopyable trait** 在避免语言中隐式复制方面的潜在益处。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **OpenAI 的间接盈利之路**：OpenAI 的营收**在没有微软帮助的情况下激增**，在过去六个月中几乎翻了一番，主要归功于 ChatGPT 等产品的直接销售，而非依赖微软的渠道，这与行业预期相反。[阅读更多](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx)。

- **AI 研究的闭环**：**Sakana AI 的 DiscoPOP** 是一种最先进的偏好优化算法，它源于 AI 驱动的发现，预示着 LLM 可以自主改进 AI 研究方法的新时代。在其[论文](https://arxiv.org/abs/2406.08414)中探索研究结果，并通过 [GitHub 仓库](https://github.com/SakanaAI/DiscoPOP)做出贡献。

- **硬件热潮与研究启示**：随着一条 [推文](https://x.com/SebastianB929/status/1800991419437367655) 的预热，人们对 Nvidia 可能即将推出的 Nemotron 充满期待；同时，Jupinder Parmar 和 Shrimai Prabhumoye 等研究人员发布了一篇探索语音建模的[论文](https://arxiv.org/abs/2402.16819)，讨论了突破性进展。

- **SSMs 仍在局中**：社区对 **Structured State Machines** (SSMs) 的延续持 50/50 的观望态度，尽管人们更倾向于混合 SSM/Transformer 架构，因为并非每个步骤都需要 Attention 层。

- **新架构横扫基准测试**：**Samba 3.8B** 问世，这是一种融合了 Mamba 和 Sliding Window Attention 的架构，展示了它在主要基准测试中可以显著超越 Phi3-mini 等模型，并提供具有线性复杂度的无限上下文长度。Samba 的强大性能详情见此[论文](https://arxiv.org/abs/2406.07522)。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **Haize Labs 挑战 AI 防护栏 (AI Guardrails)**：[Haize Labs](https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww) 发布了一份关于识别和修复 AI 故障模式的宣言，通过成功破解领先 AI 安全系统的保护机制，展示了其中的漏洞。

- **tldraw 以开源风格复刻 iPad 计算器**：[tldraw](https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww) 团队将苹果的 iPad 计算器重构为一个开源项目，展示了他们分享创新成果的承诺。

- **亚马逊对话式 AI 失误分析**：根据 [cakecrusher](https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/) 分享的文章中前员工的见解，对亚马逊对话式 AI 进展（或缺乏进展）的调查指出，其文化和运营流程优先考虑产品而非长期的 AI 开发。

- **OpenAI 财政业绩飙升**：OpenAI 的年化收入运行率已接近 [\$34 亿](https://x.com/deedydas/status/1801003523292729789)，引发了关于此类收益影响的对话，包括可持续性和支出率。

- **Argilla 与 Hugging Face 合并以优化数据集**：[Argilla](https://argilla.io/blog/argilla-joins-hugggingface) 已与 Hugging Face 合并，为推动 AI 数据集和内容生成的改进奠定了更好的协作基础。



---



## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **Open Interpreter 为 LLM 赋能**：**Open Interpreter** 被讨论作为一种将自然语言转化为直接计算机控制的手段，为未来与定制化 **LLM** 和增强型感官模型的集成提供了桥梁。

- **视觉与代码在实际应用中相遇**：社区分享了使用 **Open Interpreter** 运行视觉模型代码的经验和故障排除技巧，特别关注 `llama3-vision.py` 配置文件，以及在复杂任务中管理服务器负载的策略。

- **浏览器控制初见成效**：在一个实际应用案例中，**Open Interpreter** 成功导航浏览器查看实时体育比分，展示了用户 Prompt 的简洁性以及对服务器需求的影响。

- **Whisper STT 的 DIY 方法**：在寻找合适的 **Whisper Speech-To-Text (STT)** 库时，一位成员最终自己开发了一个独特的解决方案，体现了社区解决问题的精神。

- **针对峰值性能进行微调**：关于微调 **Open Interpreter**（如修改 *core.py*）的讨论，突显了为应对性能和服务器负载挑战以满足用户特定需求所做的持续努力。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **苹果 30 亿参数模型的突破**：苹果在 WWDC 上发布了一个 **30 亿参数** 的端侧语言模型，通过混合 **2-bit 和 4-bit 配置**（平均每权重 **3.5 bits**）的策略，实现了与未压缩模型相同的准确度。该方法优化了内存、功耗和性能；更多细节见其 [研究文章](https://machinelearning.apple.com/research/introducing-apple-foundation-models)。

- **Docker 化 AI 遭遇 GPU 障碍**：一名工程师遇到 Docker Desktop 在 Windows 11 的 Ubuntu 虚拟机上无法识别 GPU 的问题，尽管使用了 `docker run --gpus all --rm -it winglian/axolotl:main-latest` 等命令。建议的诊断步骤包括使用 `nvidia-smi` 检查 GPU 状态并确认 CUDA toolkit 的安装。

- **CUDA 困惑与 WSL 2 解决方案**：讨论转向了应该在 Windows 还是 Ubuntu 上设置 CUDA toolkit，共识倾向于在 Ubuntu 的 WSL 2 中安装。一位用户表示打算参考官方 [NVIDIA toolkit 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 在 Ubuntu WSL 上配置 CUDA。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

**OpenRouter 中的参数截断 (Param Clamping)**：Alex Atallah 明确指出，对于 OpenRouter，超过支持范围的参数（如 Temp > 1）会被截断（clamped）为 1；此外，尽管 UI 界面显示了 Min P 等参数，但这些参数实际上并未通过 UI 传递。

**Mistral 7B 延迟之谜**：用户注意到 **Mistral 7B** 变体的响应时间有所增加，并将其归因于上下文长度的变化和潜在的路由调整。这一观点得到了来自 [API watcher](https://orw.karleo.net/changes) 和 [model uptime tracker](https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime) 数据的支持。

**区块链开发者求职**：一位资深全栈兼区块链开发者正在寻找新机会，展示了该领域的丰富经验和积极参与的意愿。

**视觉模型展望**：有用户请求在 OpenRouter 中加入更多先进的视觉模型（如 cogvlm2），以增强数据集打标（captioning）能力。

---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **tinygrad 中的 RDNA3 汇编悬赏**：George Hotz 发布了 [tinygrad 中支持 RDNA3 汇编的悬赏](https://github.com/tinygrad/tinygrad/pull/3637)，邀请合作者共同完成此项增强。
- **招募 Qualcomm 内核驱动开发**：目前有一个开发“支持 HCQ graph 的 Qualcomm 内核级 GPU 驱动”的机会，目标对象是具备 Qualcomm 设备和 Linux 系统专业知识的工程师。
- **tinygrad 移动端能力确认**：已确认 **tinygrad** 可以在 **Termux** 应用中运行，展示了其对移动环境的适应性。
- **讨论：在 tinygrad 中模拟混合精度**：关于在矩阵乘法期间通过 bfloat16 和 float32 之间的转换来实现混合精度的讨论表明，当与 Tensor Core 数据类型对齐时，可能会带来潜在的速度提升。
- **张量索引与 UOp 图执行查询**：正在探索高效的张量索引技术（参考布尔索引），以及使用 `MetalDevice` 和 `MetalCompiler` 进行 UOp 图执行，重点在于使用 `compiledRunner` 简化内核执行。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord

- **AI 炒作中的现实清醒剂**：dbreunig 的一篇博客指出，AI 行业主要由接地气的务实工作组成，并将当前的 LLM 工作阶段与 2019 年左右的数据科学状况进行了对比。他的[文章](https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html)登上了 Hacker News 首页，表明人们对超越煽情预期的务实 AI 方法有着浓厚兴趣。

- **昔日 CPU，今日 GPU**：计算资源的搜索重点已从 RAM 和 Spark 容量转向 GPU 核心和 VRAM，这说明了随着 AI 发展的推进，技术需求正在发生变化。

- **LLM 部署中的 Token 经济学**：Databricks 的客户数据显示，大语言模型（LLMs）的输入输出比为 9:1，这强调了输入 Token 的成本可能比输出更关键，这对运营 LLM 的机构具有经济影响。

- **聚焦务实 AI 应用**：Hacker News 对 dbreunig 在 Databricks 峰会上的观察表示认可，突显了社区对 AI 技术演进和现实落地讨论的关注。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **欧洲 AI 协同**：Aleph Alpha 与 Silo AI 建立了战略合作伙伴关系，旨在推动开源 AI 的前沿发展，并为欧洲量身定制企业级解决方案。此次合作结合了 Aleph Alpha 的先进技术栈和 Silo AI 强大的 300 多名 AI 专家团队，旨在加速 AI 在欧洲工业领域的部署。[阅读合作详情](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/)。

---

## [Torchtune](https://discord.com/channels/1216353675241590815) Discord

- **请大家参与投票**：一位用户请求社区参与一项关于如何提供微调（finetuned）模型服务的投票，并对大家的参与表示感谢。

- **Tokenizer 正在迎来大改**：一项关于全面重构 Tokenizer 的 RFC 已被提出，旨在为模型 Tokenization 提供一个功能更丰富、更具可组合性且更易用的框架，详见 GitHub 上的 [Pull Request](https://github.com/pytorch/torchtune/pull/1082)。

---

## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **AI 爱好者的免费虚拟会议**：*Infer: Summer '24* 是一场将于 6 月 26 日举行的免费虚拟会议，将汇集 AI 和 ML 专业人士，共同讨论该领域的最新动态，包括**推荐系统 (recommender systems)** 和 **AI 在体育领域的应用**。
- **行业专家集结**：来自 Qwak 的解决方案架构师 **Hudson Buzby** 和来自 ArtifexAI 的数据科学家 **Russ Wilcox** 等资深专业人士将在会议上分享见解，他们代表了 **Lightricks、LSports 和 Lili Banking** 等公司。
- **专家实时互动**：参会者将有机会与行业领袖进行实时交流，为在 ML 和 AI 领域交换实践知识和创新解决方案提供平台。
- **动手学习机会**：预定的演讲将深入探讨 AI 系统的实际应用，如架构和用户参与策略，重点是构建稳健的预测技术。
- **与 AI 专业人士建立联系**：鼓励参与者通过[免费注册](https://tinyurl.com/j8z6s8ka)参加活动，与顶尖的 ML 和 AI 专业人士建立联系并向其学习。组织者强调，这次活动是扩大 AI 认知和行业联系的关键机会。

---

## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **不要错过 AI 开发的未来**：一场关于 AI 软件开发系统如何**赋能开发者 (amplify developers)** 的活动即将举行。更多信息和 RSVP 请点击[此处](https://discord.com/events/1089876418936180786/1242653066512175157)。
  
- **了解顶尖 ML 论文**：最新的**机器学习论文精选 (Machine Learning Paper Picks)** 已整理完毕供您阅读，请点击[此处](https://discord.com/channels/1089876418936180786/1250679657263534152/322567890986752)。

- **参与 CambAI 团队的最新动态**：参加即将与 CambAI 团队举行的活动，保持行业领先地位。点击[此处](https://discord.com/events/1089876418936180786/1250168740667195455) RSVP 参与对话。

- **领取您的 AMA 福利**：参加 AMA 的成员，请记得通过公告中提供的自定义链接领取您的 **0din 角色**，以获取有关 T-shirts 等周边礼品 (swag) 的更新。

- **通过新标签参与精选对话**：新的 `member-requested` 标签现已上线，用于向专门策划的讨论[频道](https://discord.com/channels/1089876418936180786/1231977676458168381)投稿，体现了社区驱动的内容策划。

- **为创新者提供资金和支持**：**Builders Program** 正在征集寻求 AI 项目支持和资金的成员，更多详情可通过链接的公告获取。

---

## [YAIG (a16z Infra)](https://discord.com/channels/958905134119784489) Discord

- **GitHub Codespaces 使用情况调查**：在 #[tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608/1250874031679475743) 频道发起了一项调查，以统计团队中 GitHub Codespaces 的使用情况，使用 ✅ 表示“是”，❌ 表示“否”作为回复选项。

---

**LLM Perf Enthusiasts AI Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI Stack Devs (Yoko Li) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

**AI21 Labs (Jamba) Discord** 没有新消息。如果该社区长期沉寂，请告知我们，我们将将其移除。

---

# 第 2 部分：按频道划分的详细摘要和链接

{% if medium == 'web' %}

### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1250525139192516609)** (854 条消息🔥🔥🔥):

- **成员对 SD3 质量表示失望**：用户批评了新发布的 **SD3** 的质量，经常将其与 **SDXL** 和 **1.5** 版本进行不利对比。担忧包括解剖结构不准确、模型不遵循 prompt 以及摄影风格质量下降。
- **SD3 medium 在 Huggingface 发布**：**SD3** medium 模型现在可以在 [Huggingface](https://huggingface.co/stabilityai/stable-diffusion-3-medium) 下载，尽管用户必须填写表格才能获得访问权限。git cloning 存在一些问题，10GB 的模型选项很受欢迎。
- **ComfyUI 成为运行 SD3 的首选**：**ComfyUI** 目前是运行 **SD3** 的首选界面，用户建议使用最佳的 sampler 和 scheduler 以获得最优结果。建议包括避免使用 Euler sampler，并使用 **uni_pc** 和 **ddim_uniform** 以获得更好的性能。
- **替代模型建议和工具**：成员们分享了替代方案和工具，例如用于写实风格的 **Juggernaut reborn** 和用于动漫风格的 **Divinie animemix**。其他推荐的运行模型的资源包括 [Playground](https://playground.com/) 和 [StableSwarm](https://github.com/Stability-AI/StableSwarmUI?tab=readme-ov-file#installing-on-windows)。
- **围绕全球和政治问题的离题讨论**：频道有时会偏离主题，讨论政治、国际关系和个人互动，分散了对 AI 模型和技术帮助的核心关注。管理员提醒社区成员保持话题相关性并举报不当内容。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/muajaja-the-simpsons-gif-7251407">Muajaja Risa Malvada GIF - Muajaja The Simpsons - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://playground.com/">免费 AI 图像生成器：艺术、社交媒体、营销 | Playground</a>: Playground (官方网站) 是一个免费使用的在线 AI 图像生成工具。使用它来创作艺术、社交媒体帖子、演示文稿、海报、视频、Logo 等。</li><li><a href="https://youtu.be/Di1KqPXxx2Y?si=CyYMHhaZCzVhNy4N">SD3 来了！！ComfyUI 工作流。</a>: SD3 终于支持 ComfyUI 了！Topaz Labs: https://topazlabs.com/ref/2377/ 如何支持我的频道 - 通过加入我的 Patreon 来支持我：https://www.patreon.co...</li><li><a href="https://tenor.com/view/crycat-crying-cat-crying-cat-thumbs-up-thumbs-up-ok-gif-17048449662472934214">Crycat Crying Cat GIF - Crycat Crying Cat Crying Cat Thumbs Up - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1de65iz/how_to_run_sd3medium_locally_right_now/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides">安装指南</a>: Stable Diffusion 知识库（安装、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://imgur.com/a/D0p0cUf">imgur.com</a>: 在 Imgur 发现互联网的魔力，这是一个由社区驱动的娱乐目的地。通过有趣的笑话、热门迷因、有趣的 GIF、励志故事、病毒视频等来振奋你的精神...</li><li><a href="https://github.com/RocketGod-git/stable-diffusion-3-gui/">GitHub - RocketGod-git/stable-diffusion-3-gui: 用 Python 编写的 Stable Diffusion 3 GUI</a>: 用 Python 编写的 Stable Diffusion 3 GUI。通过在 GitHub 上创建账户，为 RocketGod-git/stable-diffusion-3-gui 的开发做出贡献。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://opendata.blender.org/">Blender - 开放数据</a>: Blender Open Data 是一个收集、展示和查询硬件及软件性能测试结果的平台 —— 由公众提供。</li><li><a href="https://stability.ai/news/deepfloyd-if-text-to-image-model">Stability AI 发布 DeepFloyd IF，这是一款强大的文本转图像模型，可以智能地将文本集成到图像中 — Stability AI</a>: DeepFloyd IF 是一款最先进的文本转图像模型，以非商业、研究许可协议发布，允许研究实验室检查和实验先进的文本转图像生成...</li><li><a href="https://stability.ai/stable-assistant-gallery">Stable Assistant 画廊 — Stability AI</a>: Stable Assistant 提供前所未有的图像创作能力。探索我们的画廊，为您所能实现的一切感到惊叹！</li><li><a href="https://stability.ai/stable-assistant-gallery?utm_campaign=Stable%20Assistant&utm_content=186946249&utm_medium=social&utm_source=twitter&hss_channel=tw-1281048162602369024">Stable Assistant 画廊 — Stability AI</a>: Stable Assistant 提供前所未有的图像创作能力。探索我们的画廊，为您所能实现的一切感到惊叹！</li><li><a href="https://imgsys.org/">imgsys.org | 由 fal.ai 提供的图像模型竞技场</a>: 一个生成式 AI 竞技场，您可以在其中测试不同的提示词并选择您最喜欢的结果。查看模型排名并亲自尝试！</li><li><a href="https://civitai.com/models/490622/mobius">Mobius - v1.0 | Stable Diffusion Checkpoint | Civitai</a>: Mobius：重新定义去偏置扩散模型的最先进技术。Mobius 是一款突破了领域无关去偏置界限的扩散模型...
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1250526449300017302)** (446 messages🔥🔥🔥): 

- **Rank 提升带来更好的训练结果**："我确认 Qwen2-1.5b 训练出现乱码的问题与 Rank 过低有关。我将 Rank 从 16 切换到 128 后，输出质量达到了我训练 Llama-3 的水平。"
- **scGPT 在学术界之外可能并不实用**："那是用 Torch 实现的自定义 Transformer 😅 目前在学术界之外基本不可用……不过他们在 Prompting 和 Tokenizer 上的做法很有趣！"
- **Unsloth 的推理支持和内存效率**：一位用户询问 Unsloth 是否会影响推理，并得到确认：Unsloth 在训练和推理过程中都能显著减少内存占用。
- **Mixture of Agents (MoA) 表现平平**：用户讨论了 Together AI 的 [MoA](https://www.together.ai/blog/together-moa) 这种分层 LLM Agent 的方法，但认为它“有点华而不实”且过于复杂。
- **容器化以实现无缝流水线**："Notebook 是很好的入门工具，但在实际应用中，你需要将其与其他组件集成。开发 CLI 工具可以实现一致的工作流，并更轻松地与 ZenML 等框架集成。"

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=l8pRSuU81PU">让我们复现 GPT-2 (124M)</a>：我们从零开始复现 GPT-2 (124M)。这段视频涵盖了整个过程：首先构建 GPT-2 网络，然后优化其训练以实现真正的……</li><li><a href="https://www.together.ai/blog/together-moa">Together MoA — 开源模型的集体智慧推动 LLM 能力的前沿</a>：未找到描述</li><li><a href="https://huggingface.co/google/recurrentgemma-9b">google/recurrentgemma-9b · Hugging Face</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=zduSFxRajkE">让我们构建 GPT Tokenizer</a>：Tokenizer 是大语言模型 (LLMs) 中必不可少且无处不在的组件，它负责在字符串和 Token（文本块）之间进行转换。Tokenizer……</li><li><a href="https://tenor.com/view/smh-shaking-my-head-sneaky-gif-21653713">Smh Shaking GIF - Smh Shaking My - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">介绍 Apple 的端侧及服务器基础模型</a>：在 2024 年全球开发者大会上，我们推出了 Apple Intelligence，这是一个深度集成的个人智能系统……</li><li><a href="https://github.com/bowang-lab/scGPT/tree/main/tutorials/zero-shot">scGPT/tutorials/zero-shot at main · bowang-lab/scGPT</a>：在 GitHub 上为 bowang-lab/scGPT 的开发做出贡献。</li><li><a href="https://github.com/bowang-lab/scGPT">GitHub - bowang-lab/scGPT</a>：在 GitHub 上为 bowang-lab/scGPT 的开发做出贡献。</li><li><a href="https://github.com/bowang-lab/scGPT/tree/integrate-huggingface-model">GitHub - bowang-lab/scGPT at integrate-huggingface-model</a>：在 GitHub 上为 bowang-lab/scGPT 的开发做出贡献。</li><li><a href="https://github.com/ollama/ollama/tree/main/examples">ollama/examples at main · ollama/ollama</a>：快速上手 Llama 3, Mistral, Gemma 和其他大语言模型。- ollama/ollama</li><li><a href="https://www.ncbi.nlm.nih.gov/guide/howto/dwn-genome/">下载生物体的完整基因组</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.05955">Turbo Sparse：以极少的激活参数实现 LLM SOTA 性能</a>：利用激活稀疏性是显著加速大语言模型 (LLMs) 推理过程且不损失性能的一种极具前景的方法。然而，激活稀疏性……</li><li><a href="https://arxiv.org/abs/2406.06282">PowerInfer-2：智能手机上的快速大语言模型推理</a>：本文介绍了 PowerInfer-2，这是一个专为智能手机上的大语言模型 (LLMs) 高速推理而设计的框架，对于模型大小超过设备……</li><li><a href="https://github.com/sebdg/unsloth/tree/cli-trainer">GitHub - sebdg/unsloth at cli-trainer</a>：微调 Llama 3, Mistral, Phi & Gemma LLM，速度提升 2-5 倍，内存占用减少 80% - GitHub - sebdg/unsloth at cli-trainer</li><li><a href="https://finalspark.com/neuroplatform/">Neuroplatform - FinalSpark</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/Replete-AI/code_bagel_hermes-2.5">Replete-AI/code_bagel_hermes-2.5 · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1250525091499085875)** (190 条消息🔥🔥): 

- **Trainer 和 TrainingArguments 的困惑**：一位用户对 Huggingface 文档中 [Trainer](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) 和 [TrainingArguments](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments) 的具体细节表示困惑。他们指出，文档描述并未完全解释如何使用这些类。

- **保存为 gguf 的问题**：一位用户在尝试将模型保存为 gguf 时遇到了 `ValueError`，但在将量化方法指定为 `f16` 后成功解决。他们分享了该解决方案及语法：`model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")`。

- **未训练 Token 错误**：另一位用户在训练模型时遇到了与未训练 Token 相关的 `ValueError`，这表明 `embed_tokens` 和 `lm_head` 必须包含在训练过程中。该问题与添加新 Token 有关，这需要启用模型某些部分的训练。

- **多标签分类的数据集格式化**：一位成员就如何微调 Llama 3 以进行多标签分类以及是否可以使用相同的 Prompt 格式寻求建议。虽然回复确认了需要一致的数据集模板，但未提供针对多标签分类的具体解决方案。

- **引用 Unsloth**：用户讨论了如何在论文中引用 Unsloth，建议引用方式为："Daniel Han and Michael Han. 2024. Unsloth, Unsloth AI."，随后附上 [Unsloth GitHub 页面](https://github.com/unslothai)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/afrizalha/Kancil-V1-llama3-fp16">afrizalha/Kancil-V1-llama3-fp16 · Hugging Face</a>：未找到描述</li><li><a href="https://discuss.huggingface.co/t/how-do-you-calculate-max-steps/40177">如何计算 max steps</a>：我想了解在不手动设置时，max steps 计算背后的数学原理。我尝试通过改变 epoch、batch 和 micro-batch 来进行反推，看看是否能弄清楚...</li><li><a href="https://github.com/unslothai/unsloth/wiki#continued-pretraining--finetuning-the-lm_head-and-embed_tokens-matrices">首页</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments">Trainer</a>：未找到描述
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1250557790208917638)** (8 条消息🔥): 

- **Stable Diffusion 3 发布，带来增强功能**：*SD3 是一个 Diffusion 模型，包含三个文本编码器*（[CLIP L/14](https://huggingface.co/openai/clip-vit-large-patch14)、[OpenCLIP bigG/14](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) 和 [T5-v1.1-XXL](https://huggingface.co/google/t5-v1_1-xxl)）、一个 Multimodal Diffusion Transformer 和一个 16 通道的 AutoEncoder 模型。在 [Stable Diffusion 3 Medium space](https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium) 查看完整详情和代码。
- **社区亮点 #62**：展示了新的工具和项目，如 [F1 预测模型](https://huggingface.co/posts/Draichi/560425192506443)、[Simpletuner v0.9.7](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7) 和 [Tiny LLM client](https://github.com/piEsposito/tiny-ai-client)。此外还有 [1M+ Dalle 3 字幕数据集](https://huggingface.co/datasets/ProGamerGov/synthetic-dataset-1m-dalle3-high-quality-captions) 和 [MARS v5](https://github.com/camb-ai/mars5-tts) 等酷炫内容。
- **Argilla 加入 Hugging Face**：*今天对 Argilla、Hugging Face 和开源 AI 社区来说是伟大的一天：Argilla 正式[加入](https://huggingface.co/posts/dvilasuero/203008804842390) Hugging Face*！强调了以社区和数据为中心的开源 AI 方法。
- **社区欢迎 Argilla**：成员们祝贺 Argilla 加入 Hugging Face，并对未来的项目和合作表示期待。*"期待我的第一个 distilabel x arguilla 项目（当然是托管在 hf 上！）"*。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium">stabilityai/stable-diffusion-3-medium · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/blog/sd3">Diffusers 欢迎 Stable Diffusion 3</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1250525199183642816)** (185 条消息🔥🔥): 

- **解决 Diffusers 中的 SD3 问题**：用户在尝试使用最新的 diffusers 库更新运行 Stable Diffusion 3 (SD3) 模型时遇到了多个错误。一位用户分享说，添加 `pipe.enable_model_cpu_offload()` 显著提高了推理速度。

- **HuggingFace 上的格拉斯哥大学组织**：一位成员宣布在 HuggingFace 上创建了格拉斯哥大学（University of Glasgow）组织，邀请教职员工、研究人员和学生使用大学邮箱加入。[组织链接](https://huggingface.co/UniversityofGlasgow)。

- **LLM 会话管理错误**：一位用户寻求帮助，希望在使用 Ollama 接口和 Llama3 的本地 LLM 中维持会话。另一位成员提供了一个 Python 脚本解决方案，使用 OpenAI API 格式进行持续会话管理。

- **Generalized LoRA (GLoRA) 实现帮助**：一位致力于将 GLoRA 集成到 PEFT 库的用户请求协助解决其分叉（forked）实现中的错误。他们提供了 [GitHub 仓库链接](https://github.com/viliamvolosv/peft)和相关的[研究论文](https://arxiv.org/abs/2306.07967)。

- **Google Colab 上的 LLM 教程**：一位成员分享了一个[在 Google Colab 上设置 LLM 的教程](https://github.com/casualcomputer/llm_google_colab)，以利用免费 GPU 进行 GPU 加速和仅 CPU 的推理。他们强调该教程对遇到类似问题的其他人很有帮助。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/nroggendorff/sd3">Stable Diffusion 3 - nroggendorff 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/TheStinger/Ilaria_RVC">Ilaria RVC - TheStinger 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/UniversityofGlasgow">UniversityofGlasgow (格拉斯哥大学)</a>：未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - LanguageBind 创建的 Hugging Face Space</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2306.07967">One-for-All: 用于参数高效微调的 Generalized LoRA</a>：我们提出了 Generalized LoRA (GLoRA)，这是一种用于通用参数高效微调任务的高级方法。通过增强 Low-Rank Adaptation (LoRA)，GLoRA 采用通用 Prompt 模块来优化...</li><li><a href="https://huggingface.co/Helsinki-NLP/opus-mt-ko-en">Helsinki-NLP/opus-mt-ko-en · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/viliamvolosv/peft">GitHub - viliamvolosv/peft: 🤗 PEFT: 尖端的参数高效微调。</a>：🤗 PEFT: 尖端的参数高效微调。 - viliamvolosv/peft</li><li><a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: 关于如何在 Google Colab 上为 GPU 加速和仅 CPU 会话设置 LLM 的教程。</a>：关于如何在 Google Colab 上为 GPU 加速和仅 CPU 会话设置 LLM 的教程。 - casualcomputer/llm_google_colab</li><li><a href="https://github.com/VedankPurohit/LiveRecall">GitHub - VedankPurohit/LiveRecall: 欢迎来到 **LiveRecall**，Microsoft Recall 的开源替代方案。LiveRecall 捕获屏幕快照，并允许您使用自然语言查询进行召回，利用语义搜索技术。为了增加安全性，所有图像都经过加密。</a>：欢迎来到 **LiveRecall**，Microsoft Recall 的开源替代方案。LiveRecall 捕获屏幕快照，并允许您使用自然语言查询进行召回，利用语义搜索技术...</li><li><a href="https://github.com/continuedev/deploy-os-code-llm">GitHub - continuedev/deploy-os-code-llm: 🌉 如何为您的开发团队部署开源代码 LLM</a>：🌉 如何为您的开发团队部署开源代码 LLM - continuedev/deploy-os-code-llm</li><li><a href="https://tenor.com/view/smol-illegally-smol-cat-cute-cat-boop-gif-3484507763170497045">Smol Illegally Smol Cat GIF - Smol Illegally smol cat Cute - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://colab.research.google.com/drive/1Rlcbd3SCibgJQmzAGZW7pyiK01pwumB4?usp=sharing>">Google Colab</a>：未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1250530466365177856)** (10 条消息🔥): 

- **Distill 宝贵但已停止更新的资源**：尽管已停止更新，[Distill](https://distill.pub/) 仍提供了关于各种 ML 和 DL 主题的极具说明性的文章。亮点包括 *Understanding Convolutions on Graphs* 和 *A Gentle Introduction to Graph Neural Networks*。
  
- **企业级 LLMs 安全查询引发关注**：一位成员询问是否有人在研究企业级 LLMs 的安全性。这反映了社区对 AI 模型安全日益增长的关注。
  
- **深入探讨 NLP 和图模型学术资源**：分享了各种学术资源，包括 [关于 PCFGs 的 NLP 笔记](https://www.cs.columbia.edu/~mcollins/courses/nlp2011/notes/pcfgs.pdf) 以及 [arXiv](https://arxiv.org/abs/2209.03003) 上一种开创性的神经 ODE 模型方法。
  
- **Luma AI 的 DreamMachine 被誉为 Pikalabs 的继任者**：[Luma AI 的 DreamMachine](https://lumalabs.ai/dream-machine) 被誉为 Pikalabs 的继任者，每月提供 30 个免费视频，支持 img2vid 和 text2vid 功能。讨论了优化结果的 Prompt 技巧，建议保持 Prompt 简洁，让系统的模型自主控制。
  
- **Nodus Labs 的 ACM 论文**：一位成员分享了一篇来自 [ACM 的论文](https://dl.acm.org/doi/10.1145/3308558.3314123)，对于那些对进一步技术细节感兴趣的人来说，这可能很有启发。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2209.03003">Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow</a>: 我们提出了一种名为 Rectified Flow 的方法，这是一种极其简单的学习（神经）常微分方程（ODE）模型的方法，用于在两个经验观察到的分布 π_0 和 π_1 之间进行传输...</li><li><a href="https://distill.pub/">Distill — 关于机器学习的最新文章</a>: 关于 Machine Learning 的文章</li><li><a href="https://ciechanow.ski/archives/">Archives - Bartosz Ciechanowski</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1250643738313887877)** (11 messages🔥): 

- **Colab 教程解决 LLM 设置问题**：一位成员分享了一个[在 Google Colab 上设置 LLM 的教程](https://github.com/casualcomputer/llm_google_colab)，利用免费的 15GB Tesla T4 GPU 进行 GPU 加速和仅 CPU 的推理会话。他们希望这能帮助那些在现有解决方案和故障排除中遇到困难的人。

- **Tiny-AI-Client 简化 LLM 使用**：另一位成员介绍了一个[微型且直观的 LLM 客户端](https://github.com/piEsposito/tiny-ai-client)，支持 vision 和 tool use，旨在成为更简单用例下 langchain 的替代方案。他们表示愿意协助解决 bug，并邀请其他人尝试。

- **SimpleTuner 发布版集成 SD3**：[SimpleTuner 的新版本](https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7)发布，全面集成了 Stable Diffusion 3 的 unet 和 lora 训练。该成员分享了他们对项目的投入，为了实现这一集成而不懈努力。

- **法语 Deep Learning 笔记本**：分享了一个包含[法语 Deep Learning 笔记本](https://github.com/SimonThomine/CoursDeepLearning)的 GitHub 仓库，旨在让 Deep Learning 更加易于学习。该课程灵感来自 Andrej Karpathy 和 DeepLearning.ai 的资源，目前仍在开发中。

- **Conceptual Captions 数据集**：一位成员分享了一个[海量数据集](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext)，包含为 Google CC12M 的 1100 万张图像生成的 2200 万条高质量字幕，使用 LLaVaNext 创建。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/Hatman/pixel-prompt">Pixel Prompt - Hatman 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://huggingface.co/posts/Draichi/560425192506443">Hugging Face 上的 @Draichi："嘿 Hugging Face 社区 🤗 我很高兴分享我的最新项目..."</a>：未找到描述</li><li><a href="https://github.com/bghira/SimpleTuner/releases/tag/v0.9.7">Release v0.9.7 - stable diffusion 3 · bghira/SimpleTuner</a>：Stable Diffusion 3。使用时，在 sdxl-env.sh 中设置 STABLE_DIFFUSION_3=true，并将基础模型设置为 stabilityai/stable-diffusion-3-medium-diffusers。更新内容：训练样本加速...</li><li><a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: 关于如何在 Google Colab 上为 GPU 加速和仅 CPU 会话设置 LLM 的教程。</a>：关于如何在 Google Colab 上为 GPU 加速和仅 CPU 会话设置 LLM 的教程。 - casualcomputer/llm_google_colab</li><li><a href="https://github.com/SimonThomine/CoursDeepLearning">GitHub - SimonThomine/CoursDeepLearning: 从零开始学习 deep learning 的笔记本集合</a>：从零开始学习 deep learning 的笔记本集合 - SimonThomine/CoursDeepLearning</li><li><a href="https://github.com/piEsposito/tiny-ai-client">GitHub - piEsposito/tiny-ai-client: 具有 vision 和 tool calling 功能的微型 LLM 客户端。极尽简便。</a>：具有 vision 和 tool calling 功能的微型 LLM 客户端。极尽简便。 - piEsposito/tiny-ai-client</li><li><a href="https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext">CaptionEmporium/conceptual-captions-cc12m-llavanext · Hugging Face 数据集</a>：未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1250561970114134106)** (1 messages): 

- **讨论 MaPO 的提议**：一位成员建议邀请 MaPO 论文的作者之一进行演讲，并强调了社区内潜在的兴趣。该论文讨论了一种用于 text-to-image diffusion 模型的创新对齐技术，避免了 divergence regularization 的局限性，并且在处理 preference data 时更加灵活 [MaPO](https://mapo-t2i.github.io/)。

**提及的链接**：<a href="https://mapo-t2i.github.io/">MaPO 项目主页</a>：SOCIAL MEDIA DESCRIPTION TAG TAG

  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1250557622910980106)** (28 messages🔥): 

- **图像检索模型方面的帮助**：一位新人寻求关于图像检索最佳模型的建议，提到了在 Microsoft's Vision API 和 CLIP 上的成功经验。建议包括在实时应用中使用 **Mobilenets**，为了获得最高分使用 **OpenCLIP**，以及利用 **Faiss** 实现多功能搜索引擎功能。
- **推荐 smol-vision**：一位成员分享了 **smol-vision** 的 [GitHub 链接](https://github.com/merveenoyan/smol-vision)，该项目提供了*缩小、优化和定制前沿视觉模型的方案*。
- **频道重定向**：引导用户在更相关的频道 <#1019883044724822016> 中发布消息，以获取更深入的见解和建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://tenor.com/view/smash-gif-21365305">Smash GIF - Smash - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://github.com/merveenoyan/smol-vision">GitHub - merveenoyan/smol-vision: 缩小、优化、定制前沿视觉模型的方案。💜</a>: 缩小、优化、定制前沿视觉模型的方案。💜  - GitHub - merveenoyan/smol-vision: Recipes for shrinking, optimizing, customizing cutting edge vision models. 💜
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1250680480085311488)** (1 messages): 

- **为多标签分类微调 Llama3**：一位成员询问了为 **MultiLabel classification** 微调 **Llama3** 的方法，询问是否应该为每一行遵循[特定 Kaggle notebook](https://www.kaggle.com/code/danielhanchen/kaggle-llama-3-8b-unsloth-notebook/notebookhere)中使用的相同 prompt 格式，或者是否存在类似于 BERT 的其他方法。
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1250562702632419451)** (79 messages🔥🔥): 

- **SD3 在不同调度器之间的结果差异**：一位用户注意到在测试 **SD3** 的不同 schedulers 时，**Comfy** 和 **Diffusers** 之间得到了不同的结果。他们征求他人的意见，看是否有人有类似经历。

- **Stable Diffusion 3 Medium 模型发布**：拥有 2B 参数的 **Stable Diffusion 3 Medium** 已经发布，并可在 **Hugging Face Hub** 上获取。该模型包含了与 Diffusers 的集成，并附带了 **Dreambooth 和 LoRA 训练脚本** [博客文章](https://huggingface.co/blog/sd3#summary-of-memory-optimizations)。

- **运行 Diffusers 脚本时的常见问题**：多位用户在运行 **SD3** 的 Diffusers 脚本时遇到了与 tokenizer 和环境设置相关的错误。建议的修复方案包括安装 `sentencepiece` 并确保路径和依赖项正确。

- **使用单 GPU 训练 SD3 LoRA**：用户讨论了使用 **RTX 4090** 等单个高端 GPU 训练 **SD3 LoRA** 的可能性。建议包括调整 batch sizes、使用 fp16 精度，并验证路径是否为绝对路径且格式正确。

- **GPU 配置故障排除**：在 **Windows** 和 **Linux** 上运行脚本遇到问题的用户分享了解决方案，如确保正确的 NVIDIA 驱动安装和 **accelerate** 配置。建议使用 **Hugging Face model hub** 手动下载模型并检查依赖项。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/sd3#summary-of-memory-optimizations>">Diffusers 欢迎 Stable Diffusion 3</a>: 无描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers/tree/main">stabilityai/stable-diffusion-3-medium-diffusers at main</a>: 无描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers#using-with-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: 无描述</li><li><a href="https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/README_sd3.md#lora--dreambooth">diffusers/examples/dreambooth/README_sd3.md at main · huggingface/diffusers</a>: 🤗 Diffusers: PyTorch 和 FLAX 中用于图像和音频生成的先进扩散模型。 - huggingface/diffusers
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1250531973177413662)** (223 条消息🔥🔥): 

- **探索 AI 路径及潜在指导**：有用户询问了成为 AI 工程师的路径，建议参考 OpenAI 前联合创始人、Tesla 前 AI 总监 **Karpathy** 提供的资源。另一位用户链接了 [sentdex 的 YouTube 系列视频](https://www.youtube.com/watch?v=GZqYr8_Q7DE)，内容是关于利用深度学习创建聊天机器人。
- **关于 GPT-4.5 Turbo 模型的推测**：针对一个临时可见的页面展开了激烈讨论，该页面暗示 OpenAI 将发布具有 256k 上下文窗口且知识截止日期为 2024 年 6 月的 **GPT-4.5 Turbo**。一些用户认为这可能是一个测试页面或错误，而另一些人则推测其具备持续预训练（Continuous Pretraining）能力。
- **ChatGPT 中的记忆限制与管理**：几位用户讨论了与 ChatGPT 记忆限制相关的问题。一位成员建议通过检查并集体总结记忆来释放空间，作为一种变通方法。
- **团队账户权益与条件**：澄清了 ChatGPT Team 账户的权益和限制，包括双倍的 Prompt 限制和计费政策。一位用户分享道，ChatGPT Team 计划要求至少订阅两个席位，如果不按年计费，月度费用会更高。
- **ChatGPT 的新 UI 与行为变化**：用户观察到最新的 ChatGPT 更新中出现了新的 UI 变化和更具表现力的文本回复，引发了对模型行为改进的兴趣和推测。


**提到的链接**：<a href="https://arxiv.org/abs/1903.00161">DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs</a>：阅读理解领域近期取得了快速进展，系统在最热门的任务数据集上已能与人类媲美。然而，大量研究强调了这些系统的脆弱性……

  

---


### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1250684685957464135)** (3 条消息): 

- **模型名称澄清**：被称为 **GPT-4 Turbo** 的模型在 OpenAI API 中使用标识符 `"gpt-4-turbo"` 进行引用。另一位成员确认，基于 GPT-4 架构的模型通常使用 `"gpt-4"` 或 `"gpt-4-turbo"` 等标识符。

- **自定义 GPT 角色引发困惑**：一位成员对使用 `"gpt-4"` 和 `"gpt-4-turbo"` 等模型名称自定义 GPT 角色表示困惑。

- **GPTs 中可能实现临时消息**：一位用户询问 GPTs 是否可以拥有临时消息。该用户随后确认这是可能的，并兴奋地表示 *"确实可以！太棒了"*。
  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1250796047815540807)** (3 条消息): 

- **分块并修剪大型文本数据**：针对管理“300MB 文本数据”的问题，一位成员建议“将其分块并进行删减”。这意味着将数据分解为更小的、易于管理的部分。
- **处理大型文档的实用技巧**：分享了一个论坛帖子的链接，提供了[处理大型文档的实用技巧](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2)。该资源可能提供了在 Token 限制内处理大型文本输入的策略。
- **使用 Embeddings 处理长文本**：另一个指向 [Embedding 长输入的 Notebook](https://cookbook.openai.com/examples/embedding_long_inputs) 的链接展示了如何处理超过模型最大上下文长度的文本。该指南使用了来自 `text-embedding-3-small` 的 Embeddings，并参考了 [OpenAI Embeddings 指南](https://beta.openai.com/docs/guides/embeddings) 以供进一步学习。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2">处理大型文档（&gt;2048 tokens）的实用技巧</a>：亲爱的 Plane 先生，请与 OpenAI 团队联系，他们将回复您的具体参数。顺颂商祺，Robinson</li><li><a href="https://cookbook.openai.com/examples/embedding_long_inputs">Embedding 长度超过模型最大上下文长度的文本 | OpenAI Cookbook</a>：未找到描述
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1250796047815540807)** (3 messages): 

- **将数据分块处理 (Chunk your data)**：一位成员建议将大型文本数据文件（特别是 **300MB** 左右的文件）进行拆分，以便更好地管理。他们指出，较小的分块（chunks）更容易处理。

- **大文档处理实用技巧**：分享了一个 [社区论坛帖子](https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2) 的链接，为管理长度超过 2048 tokens 的文档提供指导。

- **OpenAI Embedding 模型限制**：分享了一个关于 [嵌入长输入 (embedding long inputs)](https://cookbook.openai.com/examples/embedding_long_inputs) 的资源，解释了 OpenAI 的 Embedding 模型具有以 tokens 衡量的最大文本长度限制。该帖子包含一个演示如何处理超长文本的 notebook，并提供了 [OpenAI Embeddings 指南](https://beta.openai.com/docs/guides/embeddings) 的链接。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://community.openai.com/t/practical-tips-for-dealing-with-large-documents-2048-tokens/17185/2">Practical Tips for Dealing with Large Documents (&gt;2048 tokens)</a>：亲爱的 Plane 先生，请与 OpenAI 团队联系，他们将回复您的具体参数。顺颂商祺，Robinson。</li><li><a href="https://cookbook.openai.com/examples/embedding_long_inputs">Embedding texts that are longer than the model&#x27;s maximum context length | OpenAI Cookbook</a>：未找到描述。
</li>
</ul>

</div>
  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1250525063736721438)** (15 messages🔥): 

- **寻找 LLM 微调的示例数据**：一位用户询问在缺乏大型数据集时如何获取微调示例。回复中未讨论具体的解决方案。

- **6 月会议录音**：澄清了 6 月的会议不会只有一份录音，而是每个活动都有各自的录音。

- **记忆实验揭示有趣结果**：一位用户分享了关于 LLM 记忆（memorization）实验的结果，显示在 10 倍示例集下有显著改进，但在更高倍数下增长微弱。分享了详细的发现以及一个包含数据集和脚本的 [GitHub 仓库](https://github.com/petergpt/Fine-Tuning-Memorisation-Experiement-GPT-35)，供进一步探索。

- **招募 LLM 专家加入团队**：发布了一个职位空缺，寻找开发和微调 LLM 的专家，以帮助创建 Hugging Face 排行榜模型，并强调需要创新的评估指标。感兴趣的候选人受邀提交简历和求职信以供考虑。

- **提振士气的提醒及新频道公告**：针对一名成员享受闲暇时间的幽默抱怨，宣布了一个新频道 <#1250895186616123534>，用于展示和演示项目。鼓励用户使用 General 语音频道进行现场演示。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/hamelhusain/status/1801309935512469938?s=46&t=6mnB9S1kF_kR1cspkH1ayQ">Hamel Husain (@HamelHusain) 的推文</a>：拥有空闲时间真是太好了！</li><li><a href="https://x.com/hamelhusain/status/1801309935512469938?s=46&t=6mnB">Hamel Husain (@HamelHusain) 的推文</a>：拥有空闲时间真是太好了！</li><li><a href="https://docs.google.com/spreadsheets/d/1Sjj5N7J7AFotEeVGL7cHqbHIy2ET8XG3xni2nG5yAGQ/edit?usp=sharing">Fine-Tuning-Memorisation-v2</a>：Results-all-v2 示例，运行 1 或 2，对象，编号，问题，预期答案，响应，提取的数字，正确，Temperature 1-50-Examples，One，Apple，Q1，Apple 的数字是多少？，65451，Apple 的数字是...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[🟩-modal](https://discord.com/channels/1238365980128706560/1241044231829848125/1250564026312691732)** (4 messages): 

- **微调 llama3 时 text-to-SQL 模板出现问题**：有成员反映在使用 `llama-3.yml` 配置和 `sqlqa.subsample.jsonl` 数据集微调 llama3 进行 text-to-SQL 时遇到问题。预处理阶段未能正确格式化数据，导致出现了 `<|begin_of_text|>` 标签，而不是 `[INST]` 和 `[SQL]` 标记。
- **关于额外额度的困惑**：几位成员讨论了因 6 月 11 日截止日期而错过额外额度（extra credits）的问题。一位成员表示跟进这么多频道非常复杂，并赞赏使用 Modal 启动应用的便捷性。
- **对 Modal 额度的感谢**：多位成员感谢收到 1000 美元额度，其中一人分享了邮箱，因为尽管在截止日期前运行了 starter app，但未收到 500 美元的额外额度。

### **LLM Finetuning (Hamel + Dan) ▷ #[learning-resources](https://discord.com/channels/1238365980128706560/1241089743933149204/1250812787740114974)** (1 条消息): 

- **寻求 LLM 输入与输出验证资源**：一位用户询问有关编写 LLM 输入和输出验证资源的建议。他们还请求推荐在该领域中受欢迎的工具。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[hugging-face](https://discord.com/channels/1238365980128706560/1241141471814488115/1250853288677478430)** (2 条消息): 

- **向 Huggingface Spaces 上传 SQLite 数据库遇到困难**：一位用户在向 Huggingface Spaces 上传 3MB 的 SQLite 文件时遇到困难，最初由于文件大小限制无法更新。他们发现的解决方案是**使用 git lfs** 来管理文件上传。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[langsmith](https://discord.com/channels/1238365980128706560/1241167367040405544/1250585502814306334)** (2 条消息): 

- **LangSmith 免费开发者计划信息**：免费开发者计划的用户享有“$0/席位成本以及每月 5K 次免费 traces”。*在你的额度耗尽之前，不会产生任何费用。*
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[berryman_prompt_workshop](https://discord.com/channels/1238365980128706560/1242223275463938221/)** (1 条消息): 

hamelh: humanloop 和 promptfoo 都很受欢迎。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[clavie_beyond_ragbasics](https://discord.com/channels/1238365980128706560/1242223963346698250/1250756995661955165)** (4 条消息): 

- **成功构建运行系统：** 一位成员在成功构建了一个可运行的系统后表达了宽慰和满足，并提到这花费了几个小时才完成。
- **分块隔离导致的库问题：** 另一条消息提出了关于 **fasthtml 库**由于分块（chunks）隔离而无法获得正确答案的问题，暗示这可能会影响性能。
- **寻求 Colbert embeddings 示例：** 一位成员询问是否有展示如何将 **Colbert embeddings** 用于检索的优秀示例，表明需要更清晰的说明或学习资源。
- **应用已部署至 Hugging Face Spaces：** 该[已部署的应用](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine)包含反馈机制，虽然运行缓慢但功能正常。应用的部署时间比构建时间更长，涵盖了 20-21 场演讲，并将反馈存储在 SQLite 数据库中。

**提到的链接**：<a href="https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine">RizzConn Answering Machine - t0mkaka 的 Hugging Face Space</a>：未找到描述。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[jason_improving_rag](https://discord.com/channels/1238365980128706560/1242224099548332132/1250554849825788107)** (2 条消息): 

- **评估类别选择方法**：一位成员讨论了在过滤文档时从 500 个类别中进行选择的挑战。他们提到曾尝试使用 GPT-3.5 和 4 来测试此方法但未获成功，目前正考虑针对不同的产品 Feed 重新使用 GPT-4.0 进行尝试，以探索潜在的改进空间。

- **ParadeDB 安装亮点**：一条消息推荐安装 ParadeDB 扩展，强调其能够*“统一操作型和分析型数据”*，并*“通过一体化搜索和分析数据库更快地获取洞察并简化数据栈”*。ParadeDB 的 **ACID 兼容**事务控制和 **Postgres 兼容性**也被列为主要优势。

**提到的链接**：<a href="https://www.paradedb.com/">ParadeDB - 用于搜索和分析的 Postgres</a>：ParadeDB 是一个基于 Postgres 构建的现代 Elasticsearch 替代方案。

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[saroufimxu_slaying_ooms](https://discord.com/channels/1238365980128706560/1242224552415596554/1250674841778917396)** (7 条消息): 

- **成员请求训练分析（Training Profiling）讲座**：一位成员询问专家是否可以举办一场关于 **training profiling** 的讲座，涵盖工具、环境搭建和日志解读。他们强调了最大化 GPU 利用率和降低训练成本的重要性，并指出*“不同框架之间存在巨大的速度差异”*。

- **讲座需求投票**：有人建议通过投票来衡量兴趣。另一位成员获得了创建投票的权限，以统计社区需求。

- **潜在讲师表示兴趣**：潜在讲师初步同意如果兴趣足够就举办讲座。他们还幽默地表示，如果社区兴趣不足，也可以开设私人课程。
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[paige_when_finetune](https://discord.com/channels/1238365980128706560/1242224662142779530/1250548667149058059)** (4 messages): 

- **Gemini 模型的方法有所不同**：一位用户寻求澄清，想知道 Gemini.Google.com 模型是否是 RAG/搜索模型的混合体，而 Gemini 1.5 Pro 则纯粹是一个语言模型。他们试图解决这些模型对同一 Prompt 输出结果不一致的问题。

- **青少年福祉框架（Adolescent Well-being Framework）的详细描述**：用户分享了来自 gemini.google.com 的青少年福祉框架（AWF）的全面细分，强调了其五个相互关联的领域，如身心健康、社区和连通性。该框架由 WHO 开发，旨在评估和促进青少年的福祉。

- **比较青少年福祉框架**：来自 aistudio.google.com 使用 Gemini 1.5 Pro 的另一个回答提供了更广泛的视角，强调了几个知名模型，如积极青少年发展的 Five Cs 模型和全校、全社区、全儿童（WSCC）模型。这个替代回答展示了物理健康、心理和情感健康以及经济福祉等多个维度，并指出这些对青少年发展至关重要。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[gradio](https://discord.com/channels/1238365980128706560/1242283474300174346/1250881489286070342)** (2 messages): 

- **在 Huggingface Spaces 上发布基于 RAG 的应用**：一位成员宣布他们使用 **Gradio** 构建了一个 **基于 RAG 的应用** 并上传到了 Huggingface Spaces。[RizzCon-Answering-Machine](https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine) 被描述为“令人耳目一新”，处理了前 20 场演讲的数据。
  
- **基于 RAG 的应用性能需要改进**：同一位成员指出，他们基于 RAG 的应用目前运行太慢，需要进行一些性能分析（profiling）以获得更好的性能。

**提到的链接**：<a href="https://huggingface.co/spaces/t0mkaka/RizzCon-Answering-Machine">RizzConn Answering Machine - a Hugging Face Space by t0mkaka</a>：未找到描述

---

### **LLM Finetuning (Hamel + Dan) ▷ #[axolotl](https://discord.com/channels/1238365980128706560/1242542198008975430/1250581077529989181)** (9 messages🔥): 

- **Llama-3-8b 预处理错误困扰用户**：一位成员报告在预处理 llama-3-8b 时遇到警告，指出：*"copying from a non-meta parameter in the checkpoint to a meta parameter in the current model, which is a no-op"*。这个错误似乎在每一层都会重复出现，在 mistral-7b 模型中也会出现。
- **VSCode 断开连接阻碍微调任务**：一位成员在通过 VSCode 中的 SSH 在深度学习机上运行微调时，经历了频繁的断开连接。另一位成员建议使用 `nohup` 或切换到在物理机上手动执行，并建议考虑使用 SLURM 以获得更好的处理效果。
- **Nohup 拯救后台进程**：成员们讨论了使用 `nohup` 在后台运行命令，即使 SSH 会话断开，进程也能继续运行。分享了一个指向 [SuperUser 线程](https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal) 的有用链接，其中包含详细说明。

**提到的链接**：<a href="https://superuser.com/questions/448445/run-bash-script-in-background-and-exit-terminal">Run Bash script in background and exit terminal</a>：是否可以启动一个命令或 Bash 脚本后退出终端且不中断命令？我的解决方案是在一天的特定时间运行 cron，但我确定有更简单的方法。

---

### **LLM Finetuning (Hamel + Dan) ▷ #[zach-accelerate](https://discord.com/channels/1238365980128706560/1242564031425024010/1250885608520286389)** (1 messages): 

- **在 FSDP 和 DeepSpeed 之间无缝切换**：分享了一篇题为《Hugging Face Accelerate 多后端故事：FSDP 和 DeepSpeed》的文章。它讨论了如何使用 Hugging Face [Accelerate](https://huggingface.co/docs/accelerate/en/index) 在 FSDP 和 DeepSpeed 之间往返切换，强调了这些后端之间的差异以及一个已合并到上游的精度相关更改。[在此阅读更多](https://huggingface.co/blog/deepspeed-to-fsdp-and-back)。

**提到的链接**：<a href="https://huggingface.co/blog/deepspeed-to-fsdp-and-back">From DeepSpeed to FSDP and Back Again with Hugging Face Accelerate</a>：未找到描述

---

### **LLM Finetuning (Hamel + Dan) ▷ #[wing-axolotl](https://discord.com/channels/1238365980128706560/1242564077151326388/1250568402360078466)** (3 messages): 

- **Axolotl 在 Modal 上运行困难，但寄希望于 Jarvis**: 一位用户分享说，尽管 Modal 据称易于使用，但他们在上面运行 Axolotl 时遇到了困难。他们计划接下来在 Jarvis 上进行尝试。

- **针对 JarvisLabs 上 Axolotl 的实用 YouTube 教程**: 另一位用户提供了一个 [YouTube 教程](https://www.youtube.com/watch?v=Y9464wasHuE&ab_channel=JarvisLabsAI)，他们认为这对于在 JarvisLabs 上运行 Axolotl 非常有用。该视频包含了 JarvisLabs 和 Axolotl GitHub 仓库的链接。

**提及的链接**: <a href="https://www.youtube.com/watch?v=Y9464wasHuE&ab_channel=JarvisLabsAI">How to run axolotl on JarvisLabs | Tutorial</a>: 在 JarvisLabs 上查看 axolotl : jarvislabs.ai/templates/axolotl；查看 axolotl github : https://github.com/OpenAccess-AI-Collective/axolotl

  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[simon_cli_llms](https://discord.com/channels/1238365980128706560/1242664474276659320/1250761957129191466)** (1 messages): 

- **寻求不通过 Ollama 访问本地模型的方法**: 一位用户询问如何通过 API 调用访问运行在服务器上的本地模型，而不是使用 Ollama，特别提到了 **TGI** 或 **vllm**。该用户引用了可能讨论此内容的博客。

- **对 DeepSeek V2 API 支持的兴趣**: 用户询问是否有支持 **DeepSeek V2 API** 的计划。这一查询表明了对增强 API 功能的浓厚兴趣。

- **在命令行中使用 FIM 模型进行自动补全**: 用户表示希望使用 **LLM CMD 配合 FIM 模型**，以获得类似于 Copilot 但在命令行中的行内自动补全建议。他们建议可以运行一次并切换到 **CMD FIM 模式**，或者直接通过前缀处理补全。

- **LLM 中的 Function calling 支持**: 有人提问当模型产生 function call token 时，**LLM 是否会支持 function calling**。这表明用户对处理代码执行的高级功能感兴趣。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[fireworks](https://discord.com/channels/1238365980128706560/1245126291276038278/1250569058261274706)** (4 messages): 

- **积分支持请求涌入**: 多位用户在填写表单后寻求积分方面的帮助。用户提供了他们的账户 ID，如 *anopska-552142*、*as-ankursingh3-1-817d86*、*standonopenstds-ff2f17* 和 *apratim941208-cc11fd*，但尚未收到积分。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[emmanuel_finetuning_dead](https://discord.com/channels/1238365980128706560/1245129595749925086/)** (1 messages): 

gitmaxd: 好问题
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[europe-tz](https://discord.com/channels/1238365980128706560/1245425547048386732/1250531245310349352)** (3 messages): 

- **Europe TZ 频道的问候**: 一位成员用 "Salutare" 在频道中打招呼，随后切换到英语，询问是否有人有兴趣使用 **Euro24 新闻**及相关数据（如球员/比赛统计数据）来 **fine-tuning 模型**。另一位成员回应道："来自德国美因茨的问候！"
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/1250525685022457916)** (5 messages): 

- **Office Hours 感谢**: "感谢大家今天参加 Office Hours！有很多很棒的问题，非常感谢大家抽时间参加！" 这一致谢突显了社区在富有成效的问答环节中的积极参与。

- **来自 AWS 博客文章的 DJL 话题**: 一位成员在 Office Hours 期间就一篇名为《使用 Amazon SageMaker 进行高效且具成本效益的多租户 LoRA 服务》的 AWS 博客文章中的 "**DJL**" 提出了疑问。该博客讨论了生成式 AI 模型的优势和用例，提到了 [BloombergGPT](https://www.bloomberg.com/company/press/bloomberggpt-50-billion-parameter-llm-tuned-finance/) 以及 vLLM 和 "optimized rolling batch scheduler" 等技术。

**提及的链接**: <a href="https://aws.amazon.com/blogs/machine-learning/efficient-and-cost-effective-multi-tenant-lora-serving-with-amazon-sagemaker/">Efficient and cost-effective multi-tenant LoRA serving with Amazon SageMaker | Amazon Web Services</a>: 在本文中，我们探索了一种利用 Amazon SageMaker 上的 LoRA 服务直接解决这些挑战的解决方案。通过使用 SageMaker large m... 中 LoRA 技术的新性能优化...

  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[openpipe](https://discord.com/channels/1238365980128706560/1245927847437008896/1250760637823451156)** (3 条消息): 

- **关于 openpipe 联系方式和额度的查询**：一名成员询问是否有人知道 openpipe 的联系人，因为他们还没有收到额度（credits）。
- **关于邮件接收情况的后续询问**：另一名成员询问第一名成员是否收到了关于额度的邮件。
- **第二轮额度发放更新**：提到第二轮额度将于 14 号（即明天）发放。
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[openai](https://discord.com/channels/1238365980128706560/1245927985123692575/1250547864334241802)** (40 条消息🔥): 

- **Fine-Tuning 可以增加新知识**：一场热烈的讨论围绕着通过 Fine-Tuning 模型来增加新知识展开，多位成员强调这一过程虽然成本高昂但非常有效。文中引用了 "Nous Hermes" 等例子，说明了通过 Fine-Tuning 成功整合多项技能的情况。

- **GPT 生成的数据与法律限制**：关于在企业内部使用 GPT 生成的数据进行训练的问题引发了辩论。讨论引用了 OpenAI 的商业条款，该条款限制使用输出结果来开发竞争性的 AI 模型，并提供了[政策链接](https://openai.com/policies/business-terms/)。

- **BLEU 分数与语法的相关性**：有人询问使用 BLEU 分数来测试语法是否合适，一些成员对其有效性表示怀疑和批评。

- **为 Fine-Tuning 生成合成数据**：社区对使用合成数据进行商业用途的 AI 模型 Fine-Tuning 的可行性和合法性感到好奇。分享了一篇关于 [ToolkenGPT 论文](https://arxiv.org/abs/2305.11554)的链接，讨论了将工具演示与 In-context learning 相结合的创新方法。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2305.11554">ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings</a>: 使用外部工具增强大语言模型 (LLMs) 已成为解决复杂问题的一种极具前景的方法。然而，传统方法通过工具演示数据对 LLMs 进行 Fine-Tuning...</li><li><a href="https://cookbook.openai.com/">OpenAI Cookbook</a>: 使用 OpenAI API 进行构建的开源示例和指南。浏览代码片段、高级技术和演练集合。分享你自己的示例和指南。</li><li><a href="https://cookbook.openai.com/examples/fine-tuned_qa/olympics-1-collect-data">Fine-Tuned Q&amp;A - collect data | OpenAI Cookbook</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LLM Finetuning (Hamel + Dan) ▷ #[pawel-function-calling](https://discord.com/channels/1238365980128706560/1250550872312643594/1250563483372617730)** (92 条消息🔥🔥): 

- **Fireworks AI logo 备受喜爱**：一位成员表达了对 **Fireworks AI logo** 的赞赏。另一位成员分享了对 function calls 的好奇，以及如何将其用于 stream evaluations 和 hallucination 管理工具。
- **模型权重中的 Delta 引起关注**：讨论了 “delta” 的概念，即模型 fine-tuning 中 instruct 权重与 base 权重之间的差异。一位成员鼓励大家尝试 model merging，称其非常“有趣”。
- **Axolotl 特殊 token 配置**：一位成员询问如何在 Axolotl 中为 function calls 添加 special tokens，其他成员确认可以通过配置实现。他们还讨论了使用 “template-free format” 来避免 chat prompt templates 的问题。
- **LangGraph 与 LangChain 的辩论**：多位成员分享了使用 **LangChain** 和 **LangGraph** 的经验，对其易用性褒贬不一。一些人称赞 LangChain 广泛的生态系统和支持，而另一些人则更喜欢 LangGraph 的定制化以及对 LangChain 的独立性。
- **Glaive Function Calling 模型介绍**：分享了 [glaive-function-calling-v1](https://huggingface.co/glaiveai/glaive-function-calling-v1) 的链接，这是一个 2.7B 参数的开源对话模型，能够进行多轮对话和智能函数执行，基于 Replit-code-v1-3b 模型。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.youtube.com/playlist?list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg">LangGraph (Python)</a>：该视频系列涵盖了如何使用 LangGraph 的代码功能，以及可能需要的常见修改。</li><li><a href="https://x.com/gitmaxd/status/1800234864329068708?s=46&t=QitgwfFVpCSQgUY0DIcTdA">来自 Git Maxd (@GitMaxd) 的推文</a>：LangGraph 学习顺序：1. LangGraph YouTube 学习系列：https://www.youtube.com/watch?v=5h-JBkySK34&list=PLfaIDFEXuae16n2TWUkKq5PgJ0w6Pkwtg 2. LangGraph 自我纠错代码助手...</li><li><a href="https://huggingface.co/glaiveai/glaive-function-calling-v1">glaiveai/glaive-function-calling-v1 · Hugging Face</a>：未找到描述</li><li><a href="https://discord.gg/scEHnRaz">加入 Fireworks.ai Discord 服务器！</a>：在 Discord 上查看 Fireworks.ai 社区 —— 与 2148 名其他成员一起交流，享受免费的语音和文字聊天。</li><li><a href="https://openaccess-ai-collective.github.io/axolotl/docs/input_output.html">Axolotl - Template-free prompt construction</a>：未找到描述</li><li><a href="https://huggingface.org/spaces/arcee-ai/mergekit-gui">mergekit-gui - arcee-ai 提供的 Hugging Face Space</a>：未找到描述</li><li><a href="https://nbsanity.com/static/d06085f1dacae8c9de9402f2d7428de2/demo.html">Llama-3 Function Calling Demo</a>：未找到描述</li><li><a href="https://clip.cafe/commando-1985/until-the-next-time/">'This was the last time. Until a next time. No chance.' - Commando</a>：[Matrix 拒绝了重返部队的提议] John Matrix：这是最后一次。Major General Franklin Kirby：直到下一次。[停顿] John Matrix：没门。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1250631155364134994)** (5 条消息): 

- **LLM 驱动的目标发现（objective discovery）突破了偏好优化（preference optimization）的界限**：一篇 [arXiv 论文](https://arxiv.org/abs/2406.08414) 探讨了如何利用 LLM 驱动的目标发现来自动寻找新的优化算法，从而实现 LLM 的离线偏好优化。这种方法通过基于性能指标迭代地提示 LLM，从而在无需人类专家干预的情况下发现偏好优化算法。
  
- **Mixture-of-Agents (MoA) 方法论汇集了多个 LLM 的集体智慧**：另一篇发表在 [Hugging Face](https://huggingface.co/papers/2406.04692) 上的论文提出了一种 Mixture-of-Agents 架构，其中多个分层配置的 LLM Agent 协同工作，以增强在各种基准测试中的表现。MoA 模型优于 GPT-4 Omni，在 AlpacaEval 2.0 上取得了 65.1% 的惊人评分，而 GPT-4 Omni 为 57.5%。

- **MoA 方法论的 GitHub 仓库**：Mixture-of-Agents 模型的实现可以在 [GitHub](https://github.com/togethercomputer/moa) 上找到。用户可以参与贡献并探索这种利用多个 LLM 的极具前景的方法。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2406.08414">Discovering Preference Optimization Algorithms with and for Large Language Models</a>: 离线偏好优化是增强和控制大语言模型 (LLM) 输出质量的关键方法。通常，偏好优化被视为一种离线监督...</li><li><a href="https://github.com/togethercomputer/moa">GitHub - togethercomputer/MoA</a>: 通过在 GitHub 上创建账号来为 togethercomputer/MoA 的开发做出贡献。</li><li><a href="https://huggingface.co/papers/2406.04692">Paper page - Mixture-of-Agents Enhances Large Language Model Capabilities</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1250537134767472810)** (152 messages🔥🔥): 

- **Stable Diffusion 3 早期评价**：[Stable Diffusion 3](https://x.com/minimaxir/status/1800921802765717754) 收到了褒贬不一的反馈，一些用户认为其令人印象深刻，而另一些用户则指出了初始阶段的问题和 Bug。一位用户提到 *“它使用 LLM 作为编码器”*，暗示了与传统模型不同的方法。

- **关于 GPT-4 Temperature 设置的辩论**：一条推文透露，即使在确定性任务上，GPT-4 在 temperature=1 时的表现也优于 temperature=0，这引发了惊讶和辩论。用户讨论了这如何不适用于微调后的 Llama 3 模型，且不同任务的结果各异。

- **无审查模型发布**：一位用户分享了他们的 [OpenHermes-2.5 数据集的无审查版本](https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored)，删除了 2,697 行受审查的内容。讨论集中在移除对齐约束的影响及其对模型响应的效果。

- **无 MatMul 的 LLM**：分享了一篇关于在大型语言模型中[消除 MatMul 操作](https://arxiv.org/abs/2406.02528)的新论文，该论文声称在十亿参数规模下具有显著的内存节省和强劲的性能。论文指出，无 MatMul 模型可以匹配最先进的 Transformer 的性能，同时减少高达 61% 的内存使用。

- **关于 AI 模型越狱的讨论**：[Haize Labs](https://x.com/haizelabs/status/1800936720990384174) 宣布了一种自动越狱 AI 模型的方法，通过诱导有害内容揭示了顶级 AI 系统中的安全违规行为。这引发了 AI 社区关于此类行为的伦理和后果的讨论。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.02528">Scalable MatMul-free Language Modeling</a>：矩阵乘法 (MatMul) 通常在大型语言模型 (LLM) 的整体计算成本中占主导地位。随着 LLM 扩展到更大的嵌入维度和上下文长度，这种成本只会不断增加……</li><li><a href="https://huggingface.co/spaces/stabilityai/stable-diffusion-3-medium">Stable Diffusion 3 Medium - stabilityai 在 Hugging Face 上的 Space</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Cm7IlwoVT4w">用《斩杀公主》测试 Neuro 的道德（第一部分）</a>：Vedal 尝试让 Neuro-sama 玩《斩杀公主》。►Twitch: http://www.twitch.tv/vedal987 ►Twitter: https://twitter.com/Vedal987 由 Paradomix 编辑</li><li><a href="https://x.com/corbtt/status/1801026166020833457">Kyle Corbitt (@corbtt) 的推文</a>：每个部署 LLM 的人都应该知道的一个疯狂事实——即使在确定性任务上，GPT-4 在 temperature=1 时也比 temperature=0 更“聪明”。老实说，在我亲自尝试之前，我也不相信这一点……</li><li><a href="https://x.com/haizelabs/status/1800936720990384174">Haize Labs (@haizelabs) 的推文</a>：今天对语言模型来说是个非常糟糕的日子。今天，我们宣布 Haize Labs 宣言。@haizelabs 对 AI 系统进行越狱（自动红队测试），以预先发现并消除任何故障……</li><li><a href="https://x.com/jeremyphoward/status/1801037736968913128">Jeremy Howard (@jeremyphoward) 的推文</a>：刚发布的新论文，展示了如何通过将蒙特卡洛树搜索 (MCTS) 与语言模型结合来大幅提高 LLM 的数学分数。不错！但是……如果我们只是简单地告诉 LLM 去……</li><li><a href="https://x.com/minimaxir/status/1800921802765717754">Max Woolf (@minimaxir) 的推文</a>：看看人们测试 Stable Diffusion 3 的情况，坦白说，这非常给力。</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1df0kau/sd3_has_been_liberated_internally_pure_text2img/">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1df0kau/sd3_has_been_liberated_internally_pure_tex">Reddit - 深入了解一切</a>：未找到描述</li><li><a href="https://x.com/corbtt/status/1801059164954775643">Kyle Corbitt (@corbtt) 的推文</a>：@eugeneyan @aidan_mclau 好吧，我现在在微调后的 Llama 3 模型上看到了截然不同的结果（这符合我之前的直觉——高温在创意任务上表现更好，低温在……）</li><li><a href="https://huggingface.co/datasets/Replete-AI/OpenHermes-2.5-Uncensored">Replete-AI/OpenHermes-2.5-Uncensored · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.05955">Turbo Sparse: Achieving LLM SOTA Performance with Minimal Activated Parameters</a>：利用激活稀疏性是显著加速大型语言模型 (LLM) 推理过程且不损害性能的一种极具前景的方法。然而，激活稀疏性……
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1250639839939002421)** (2 messages): 

- **关于交替预训练和指令的遗失论文**：一名成员请求帮助寻找**一篇最近的论文**，该论文在 finetuning 期间交替使用预训练文档和提取的指令。另一名成员表示如果找到了这篇论文，希望能得到通知。
  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1250760508274118707)** (15 messages🔥): 

- **尚未确定最终的 RAG 数据集 Schema**：关于 RAG 最终数据集 Schema 的询问显示其尚未定稿。“目前看来还没有”以及关于敲定 datagen pipeline 的讨论确认了开发仍在进行中。
  
- **嵌入文本以实现更好的验证**：有建议提出链接或展示嵌入文本（embedded texts）以避免幻觉（hallucinations）。一位成员指出，*“你可以对 embeddings 进行 ctrl+f 搜索。”*

- **使用 Marker 进行 PDF 到 Markdown 的转换**：已确认原始文本将为 Markdown 格式，因为 PDF 正通过 [Marker](https://github.com/VikParuchuri/marker) 进行转换。该工具因其准确性受到称赞，但被指出在推理时（inference time）转换速度较慢。

- **使用 Pandoc 和 make4ht 进行其他文档转换**：讨论的另一种转换各种文档类型的方法包括对 LaTeX 文件使用 [Pandoc](https://www.pandoc.org) 和 make4ht。这些工具被建议作为处理不同格式的替代方案。

- **速度优化建议**：提出了针对 Marker 的速度改进建议，例如设置 min_length 以避免不必要的 OCR。一位成员声称在 A10 GPU 上每个 worker 每秒大约可以处理 1 页，强调了并行处理的潜力。

**提到的链接**：<a href="https://github.com/VikParuchuri/marker">GitHub - VikParuchuri/marker: Convert PDF to markdown quickly with high accuracy</a>：使用 Marker 快速且高精度地将 PDF 转换为 Markdown - VikParuchuri/marker

  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1250560067032449074)** (6 messages): 

- **开源状态目前仍为关闭**：当被问及该项目是否开源时，得到的确认是：*“目前尚未开源。”* 目前没有立即改变这一状态的计划，尽管这被*“讨论过”*，并可能随着开发的继续而被重新考虑。
- **对 world-sim AI 增强功能的请求**：一位用户建议让 **world-sim AI 更加犀利（edgier）**，并提议为该应用开发一个**免费的移动端移植版**。他们认为移动版本会很*“酷”*。
  

---



### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1250530296504258685)** (136 messages🔥🔥): 

- **对新搜索功能的喜爱**：一位用户表达了对新搜索功能的兴奋，说：*“非常喜欢这个新搜索，”* 并立即请求 iOS 版本，*“现在就需要 iOS 版，拜托了。”*

- **对 Perplexity 网站功能的担忧**：一位成员询问其他人是否也遇到了 Perplexity 网站的显示问题，即每次回答后聊天界面都会向上跳动半页。

- **企业支持方面的困扰**：一位拥有企业版计划的用户对支持问题表示沮丧，提到尽管两周前发了邮件，但他们被*“丢进工单系统后就再也没有收到回复”*。建议包括联系 Alex 或查看 [支持页面](https://www.perplexity.ai/settings/account)。

- **关于 API 限制和解决方案的讨论**：成员们讨论了 Perplexity API 的局限性，一人建议分解复杂的请求，另一人建议使用较小的模型来处理任务。一位成员还提到了一款正在开发中的替代 API 产品，声称其产品可能提供更好的可扩展性和功能。

- **对 Rabbit Inc. 的不满**：在一段冗长的讨论中，一位用户分享了他们在 Rabbit Inc. 的负面经历，批评了他们的支持和隐私政策，并链接到了一个详细描述该问题的 [Reddit 帖子](https://www.reddit.com/r/rabbitinc/comments/1desu27/rabbit_inc_will_not_allow_you_to_participate_in/)。另一位成员指出，Perplexity 并不需要此类证明即可参与讨论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.reddit.com/r/rabbitinc/comments/1desu27/rabbit_inc_will_not_allow_you_to_participate_in/">Reddit - 深入了解任何事物</a>：未找到描述</li><li><a href="https://www.perplexity.ai/settings/account">Perplexity</a>：Perplexity 是一款免费的 AI 驱动问答引擎，能够为任何问题提供准确、可信且实时的回答。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1250554338917744701)** (8 条消息🔥): 

- **埃隆·马斯克撤销对 OpenAI 的诉讼**：埃隆·马斯克在法庭听证会前一天撤回了对 OpenAI 及其高管的诉讼。该诉讼声称 OpenAI 已从其使命转向以营利为目的的实体，优先考虑其最大的投资者 Microsoft，而非人道主义目标 [来源](https://www.pymnts.com/artificial-intelligence-2/2024/elon-musk-drops-lawsuit-against-openai-one-day-before-hearing/) [来源](https://www.cnbc.com/2024/06/11/elon-musk-drops-suit-against-openai-and-sam-altman.html)。

- **首颗木制卫星定于 2024 年发射**：京都大学和住友林业的研究人员展示了全球首颗木制卫星 LignoSat，预计将于 2024 年 9 月发射。该项目旨在评估木材作为太空可持续材料的潜力，并减少对环境的影响 [来源](https://www.japantimes.co.jp/news/2024/05/29/japan/science-health/world-first-wooden-satellite/) [来源](https://www.space.com/japan-september-launch-first-wooden-satellite)。

- **电子邮件的碳足迹及其环境影响**：一封电子邮件的平均碳足迹约为 4 克二氧化碳，当邮件包含大型附件时，这一数值会显著增加。使用文件共享链接代替附件可以减少二氧化碳排放 [来源](https://www.mailjet.com/blog/email-best-practices/email-carbon-footprint/) [来源](https://blog.rwth-aachen.de/itc/en/2023/12/18/co2-auswirkungen/)。

- **成语 "rock bottom" 的起源**：短语 "hit rock bottom" 指达到可能的最低水平，自 19 世纪中叶以来一直被隐喻使用。它最初描述的是矿工在挖掘时触及土壤下方坚硬岩层的经历 [来源](https://dictionary.langeek.co/en/word/212505?entry=hit+rock+bottom) [来源](https://english.stackexchange.com/questions/597487/what-is-the-origin-of-the-phrase-hit-rock-bottom)。

- **Perplexity.ai 利用 LLM 实现快速响应**：Perplexity.ai 利用 AWS p4d 实例（配备 NVIDIA A100 GPU）等尖端硬件以及 NVIDIA 的 TensorRT-LLM 等软件优化，尽管使用了大型语言模型（LLM），仍能实现快速结果。与 AWS 和 Kubernetes 的集成促进了弹性扩展，减少了停机时间和网络开销 [来源](https://www.perplexity.ai/hub/blog/introducing-pplx-api)。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/embed/eC882F-jaMw">YouTube</a>：未找到描述</li><li><a href="https://www.perplexity.ai/search/where-does-the-fpYkPFAuS.Gbido9PnphDQ">成语 &quot;rock bottom&quot; 从何而来？</a>：成语 &quot;hit rock bottom&quot; 起源于挖掘或采矿时触及坚硬基岩或底层岩石的字面含义。它已被用于...</li><li><a href="https://www.perplexity.ai/page/Worlds-First-Wooden-j0Q0FI3MS6OKEDTPzSN7Fw">全球首颗木制卫星</a>：在一项突破性进展中，京都大学和住友林业的研究人员展示了 LignoSat，这是全球首颗木制卫星，定于...</li><li><a href="https://www.perplexity.ai/page/Musk-drops-lawsuit-LLG_ToBhQ2..DzJ3e1RKJQ">马斯克撤销对 OpenAI 的诉讼</a>：OpenAI 联合创始人埃隆·马斯克在法庭听证会前一天撤回了对该公司及其高管 Sam Altman 和 Greg Brockman 的诉讼...</li><li><a href="https://www.perplexity.ai/page/Emails-Carbon-Footprint-m0T5zvUQQkC.G2wIQV0GSg">电子邮件的碳足迹</a>：研究表明，一封电子邮件的平均碳足迹约为 4 克二氧化碳当量，如果邮件包含附件，足迹会显著增加...</li><li><a href="https://www.perplexity.ai/search/how-does-perplexityai-JQsCcEwOQSyyMFTrGbw43g">Perplexity.ai 如何在 LLM 较慢的情况下快速获取结果</a>：Perplexity.ai 通过结合尖端软件和硬件，在即便使用大型语言模型 (LLM) 的情况下也能实现快速结果。关键在于...</li><li><a href="https://www.perplexity.ai/search/what-do-you-dt6P7CulQAWGTwBa2.zbzA">Perplexity</a>：Perplexity 是一款免费的 AI 驱动回答引擎，可为任何问题提供准确、可信且实时的答案。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1250598120169934848)** (14 messages🔥): 

- **Custom GPTs 的连接问题**：一名成员报告了在 [chat.openai.com](https://chat.openai.com/) 上尝试执行函数调用（function calls）时，Custom GPTs 显示 *"it couldn't connect"*（无法连接）的问题。另一名成员建议检查在创建 GPT 时是否在 Auth 面板中添加了 PPLX API key。
- **桌面端应用 vs. 网页版问题**：有用户指出 Custom GPTs 在 chat.openai.com 的网页版上无法运行，但在桌面端应用上运行正常。这暗示可能存在影响 API 调用的特定平台问题。
- **仍未解决**：尽管确认了 API key 的正确使用并尝试了推荐的解决方案，一名成员在使用 Custom GPTs 时仍然遇到错误且没有响应。另一名成员表示他们在相同条件下的设置运行良好，这表明问题可能是孤立的。
  

---



### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1250856013259870301)** (3 messages): 

- **关于计算强度指标的疑问**：一位成员询问，在实践中，是否只有对从 **Global Memory** 访问的数据进行的**浮点运算**才被计入计算强度（每字节访问的操作数），而不是对仅存在于寄存器（registers）中的数据进行的操作。随后他发了一条自我调侃的备注，幽默地承认他在一周前就打好了这段咨询，但一直没发出去。
  

---


### **CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1250854733481246801)** (2 messages): 

- **轻松安装 Triton 3.0**：一位用户分享了一个无障碍的[从源码安装 Triton 3.0](https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/)指南。这包括卸载当前的 Triton 版本、克隆仓库、安装依赖项并运行设置脚本。

- **从 PyTorch 安装 Triton 的替代方法**：另一位成员建议了一种涉及[克隆 PyTorch 仓库](https://github.com/pytorch/pytorch)并使用 `make triton` 命令的方法。特定版本可以在 [triton_version.txt 文件](https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt)中找到。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.umerha.com/smarties/2024-06-13-installing-triton-3-0/">安装 Triton 3.0.0</a>：截至 2024 年 6 月 13 日，要获取 Triton 3.0，你必须从源码安装，步骤如下：</li><li><a href="https://github.com/pytorch/pytorch">GitHub - pytorch/pytorch</a>：Python 中的张量和动态神经网络，具有强大的 GPU 加速功能。</li><li><a href="https://github.com/pytorch/pytorch/blob/main/.ci/docker/triton_version.txt">pytorch/.ci/docker/triton_version.txt</a>：PyTorch 仓库中记录 Triton 版本的文本文件。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1250629580742594581)** (13 messages🔥): 

- **关于使用 PyTorch 实现快速 8-bit 优化器的好奇**：一位成员询问是否可以仅使用**纯 PyTorch** 和 **torch.compile** 创建快速的 8-bit 优化器。他们强调了动态创建查找表（lookup tables）以及像 [bitsandbytes 实现](https://arxiv.org/pdf/2110.02861)中那样在 FP32 中进行计算的重要性。

- **无缝替换（Drop-in replacement）的担忧**：另一位成员讨论了为 8-bit 优化器制作“无缝替换”可能面临的挑战，因为与 32-bit 计算相比可能存在精度偏差。然而，他们指出在实践中，使用 bitsandbytes 的 8-bit AdamW 时没有观察到精度下降。

- **纯 PyTorch + Triton 版本的 8-bit 优化器**：一位正在进行此类实现的成员提出了一个纯 `PyTorch` + `Triton` 版本的 `bitsandbytes` 8-bit 优化器。目标是避免使用自定义 CUDA kernels。

- **bitsandbytes 的长期路线图**：另一位贡献者承认了将 bitsandbytes 与 `torch.compile` 集成以获得更好兼容性的可能性。他们指出了实现中的关键方面，如量化映射（quantization maps）和确保操作在 FP32 中执行。

**提及的链接**：<a href="https://arxiv.org/abs/1511.04561">8-Bit Approximations for Parallelism in Deep Learning</a>：开发实用的深度学习数据产品通常需要跨处理器和计算机进行并行化，以便在大数据集上实现深度学习，但通信瓶颈……

  

---

### **CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1250617944174493857)** (2 条消息): 

- **1-bit LLM 承诺快速量化**：分享了一个指向 [BiLLM GitHub 项目](https://github.com/Aaronhuang-778/BiLLM) 的链接，该项目声称可以在单张 GPU 上在 0.5 小时内完成 7B LLM 的量化。一位成员指出，“*似乎没有 fused kernel，所以（目前）没有加速*。”
- **观察到 benchmark 准确率呈上升趋势**：一位成员观察到 benchmark 准确率似乎呈上升趋势。他们对这种模式很感兴趣，但没有讨论更多细节。

**提及的链接**：<a href="https://github.com/Aaronhuang-778/BiLLM">GitHub - Aaronhuang-778/BiLLM: (ICML 2024) BiLLM: Pushing the Limit of Post-Training Quantization for LLMs</a>: (ICML 2024) BiLLM: Pushing the Limit of Post-Training Quantization for LLMs - Aaronhuang-778/BiLLM

  

---


### **CUDA MODE ▷ #[cool-links](https://discord.com/channels/1189498204333543425/1189868872887705671/)** (1 条消息): 

_shivasinghbagri: https://powerinfer.ai/v2/
  

---


### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1250737710792966214)** (31 条消息🔥): 

- **在 4090 上难以实现加速**：尽管设置了相关的 flag 并使用了多种方法（如 `torch.matmul` 和 `torch.nn.functional.linear`），一位成员在 RTX 4090 GPU 上进行 INT8 和 FP16 混合矩阵乘法时仍无法获得理想的加速。他们尝试了各种配置，但观察到性能反而变慢了。
  
- **FP8 和 INT8 矩阵乘法**：讨论显示，由于类型转换（casting）过程缓慢，FP8 x FP8 的 matmul 操作比 INT8 更慢。INT8 量化似乎能提供更好的速度和准确率，并且即使在 Pascal 等旧款 GPU 上也受支持。
  
- **INT6 量化展现出潜力**：在 Llama3-8B-instruct 模型的测试中，使用 group size 为 64 的 INT6 量化显示出极小的准确率下降。成员们指出，INT6 和 INT4 量化具有前景良好的性能，但需要小心处理以维持准确率。

- **探索 FP16/FP8 混合精度**：讨论了 FP16 x FP8 混合精度操作，强调了其潜在优势以及 Microsoft BitBLAS 中现有的支持。提到了实现和集成方面的挑战，特别是需要编译 TVM。

- **Microsoft BitBLAS 实现**：BitBLAS 支持 FP16 x FP8E4M3 等混合精度矩阵乘法，但它严重依赖 TVM 的 Python 接口。这一要求使得与其他库的集成变得复杂，降低了某些项目的易用性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/pytorch/blob/f9c9c06c4478fbfbbf6986f21410ecac37d3f63e/test/test_matmul_cuda.py#L316-L325">pytorch/test/test_matmul_cuda.py at f9c9c06c4478fbfbbf6986f21410ecac37d3f63e · pytorch/pytorch</a>：Python 中具有强大 GPU 加速功能的张量和动态神经网络 - pytorch/pytorch</li><li><a href="https://gist.github.com/mobicham/f95a09eaf2db48632f8bf571693c0884">torch_compile_mixed_mm_test.py</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/pytorch/ao/blob/6f44d259fabe1195669654e22f7f97fc028f89af/torchao/quantization/subclass.py#L371-L383">ao/torchao/quantization/subclass.py at 6f44d259fabe1195669654e22f7f97fc028f89af · pytorch/ao</a>：用于量化和稀疏化的原生 PyTorch 库 - pytorch/ao
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1250525436593705101)** (54 条消息🔥): 


- **控制 norm kernel 调用次数**：一位成员提到他们提交了一个更改，用于控制额外的 norm kernel 调用数量，并建议对 adam kernel 采用类似的策略可能会有好处。

- **探索 C++20 的可能性**：讨论了使用 `std::span<const float *, N>` 实现类似功能的可能性。一位成员指出，这可能需要修改调用端，或者利用现有的构造函数。

- **修复导致 loss 为无穷大的竞态条件**：loss 出现无穷大（infinities）的问题被确定为竞态条件（race condition）。该修复方案结合移除 atomic add，在 8xH100 上进行 profile 时，实现了完全的确定性（determinism）并略微提升了性能。

- **随机数据加载改进**：引入数据加载随机性的 PR，以确保更好的打乱（shuffling）和处理。观察了文档边界和数据批处理（batching）如何影响训练性能，并分享了文档长度的直方图以提供见解。

- **讨论批次与文档相似性**：辩论了训练期间文档批处理中的语义相似性影响。一些成员认为这有助于 in-context learning，而另一些成员则对其在泛化（generalization）方面的效果持怀疑态度。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/karpathy/llm.c/pull/573">Dataloader - introducing randomness by gordicaleksa · Pull Request #573 · karpathy/llm.c</a>：迈向完全随机的训练数据打乱... 此 PR 实现了以下内容：每个进程拥有不同的唯一随机种子，每个进程的训练数据加载器独立选择其起始 sha...</li><li><a href="https://github.com/karpathy/llm.c/pull/522">Add master weights to resume state by gordicaleksa · Pull Request #522 · karpathy/llm.c</a>：我们目前没有将 master weights 作为状态的一部分保存 -> 这会导致损失一些精度，因为否则当我们恢复训练时，必须通过从低精度向上转换来重建 master weights...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[rocm](https://discord.com/channels/1189498204333543425/1233704710389764236/1250752018293985373)** (1 条消息): 

- **AMD MI300X 性能超越 NVIDIA H100**：最近的一篇 [TensorWave 博客文章](https://www.blog.tensorwave.com/amds-mi300x-outperforms-nvidias-h100-for-llm-inference/) 分享了令人兴奋的基准测试结果，显示 AMD 的新款 MI300X 加速器在 LLM 推理方面的吞吐量比 NVIDIA 的 H100 SXM 高出 33%。这一初步成功基于 MK1 的推理软件在 Mixtral 8x7B 上运行 vLLM，表明尽管软件生态系统尚不成熟，AMD 的硬件仍是一个强劲的竞争对手。

**提到的链接**：<a href="https://www.blog.tensorwave.com/amds-mi300x-outperforms-nvidias-h100-for-llm-inference/">AMD’s MI300X Outperforms NVIDIA’s H100 for LLM Inference</a>：了解 AMD 的 MI300X 加速器在真实 AI 工作负载中是否能超越 NVIDIA 的 H100。早期结果已出炉！

  

---


### **CUDA MODE ▷ #[bitnet](https://discord.com/channels/1189498204333543425/1240586843292958790/1250549480164425828)** (13 条消息🔥): 

- **未解决的 MX 格式测试导致构建失败**：由于一个*无关的 mx 格式测试*，构建过程失败。重试测试的尝试未获成功，该问题仍未解决。
- **CoffeeVampir3 分享测试文件**：当被问及用于测试 Bitnet 的特定文件时，一位成员提供了 [bitnet_trained_to_ao_test.py](https://github.com/CoffeeVampir3/ao-bitnet/blob/main/bitnet_staging/bitnet_trained_to_ao_test.py) 的链接。
- **发布分支截止日期与 Nightly 构建**：提到任何要包含在 0.3 发布分支中的工作必须在下周二之前合并。或者，也可以选择继续使用 nightly 构建进行实验。
- **为 Bitnet 重构 Uint2**：Uint2 的重构正在进行中，预计明天完成。随着 Bitnet 的 bitpacking 现已可用，这项工作正在扫清后续开发的障碍。

### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1250538845229813781)** (71 messages🔥🔥): 

- **讨论在 Web 服务器上运行 LM Studio**：一位成员询问：“LM Studio 可以在 Web 服务器上运行吗？”另一位成员澄清道：“如果是远程 Headless 服务器，它将无法工作。”
- **加载 Qwen-2 模型出错**：用户在 LMChat 上使用 Qwen-2 模型时遇到错误。根据另一位用户的建议，解决方案是使用 ChatML 预设并启用 Flash Attention。
- **翻译模型推荐**：成员们讨论了最适合翻译的 LLM 模型。推荐了 Qwen2 和 Aya 23，尽管指出它们并不完美。
- **对 LM Studio 开发者的担忧**：一位用户对 LM Studio 开发者的匿名性表示担忧。随后有人澄清，可以在 GitHub 上找到首席开发人员和创始人。
- **通过 CLI 启用 Flash Attention**：用户寻求通过命令行界面启用 Flash Attention。据指出，LMS CLI 缺少此功能，但建议使用 llama.cpp 作为替代方案。

**提到的链接**：<a href="https://github.com/andrewyng/translation-agent">GitHub - andrewyng/translation-agent</a>：通过在 GitHub 上创建账户，为 andrewyng/translation-agent 的开发做出贡献。

  

---


### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1250572233215184896)** (8 messages🔥): 

- **Gemini 1.5 flash 在 JSON 模式下表现不佳**：“伙计们，这里有点偏离主题，但是……有人试过 Gemini 1.5 flash 吗？JSON 模式表现非常糟糕且具有随机性，会产生空字符串。”一位成员在 JSON 模式下使用 Gemini 1.5 flash 时遇到问题，正在寻求他人的反馈。
  
- **Tess 2.5 72b 量化模型在 HF 发布**：“Tess 2.5 72b q3 和 q4 量化 GGUF 已在 HF 上线。”这些新的量化模型现在可以在 Hugging Face 上获取。

- **VRAM 限制影响模型选择**：“但在只有 8GB VRAM 的情况下，你不会有太多选择。对于小型模型，也许 Llama 是一个不错的开始。”一位成员就 VRAM 限制向另一位成员提供建议，并推荐了像 Llama 这样的小型模型。

- **语法和句法问题的模型推荐**：“上次我尝试使用模型来检查内容的语法和句法问题时，”一位成员正在寻求能够准确检查语法和句法，特别是能够识别对话的模型推荐。

- **LM Studio 支持 GGUF，不支持 safetensor 文件**：“你好！LM Studio 支持多部分 safetensor 文件吗？”“不支持 safetensor，只支持 GGUF。”一位成员确认 LM Studio 仅支持 GGUF 格式，不支持 safetensor 文件。
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1250724815799713893)** (2 messages): 

- **提供 AVX2 错误的直接解决方案**：一位成员指出，遇到的错误消息是由于缺乏 AVX2 指令集导致的。这引导用户去检查其 CPU 的 AVX2 支持情况。
  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1250545180012970096)** (9 messages🔥): 

- **P40 价格飙升**：作为**电子废料的 P40** 在中国二手市场曾经大约只要 *75 美元*，但现在价格已超过 200 美元。这种飙升与全新的 **4060Ti 16G** 形成对比，后者价格约为 480 美元。
- **制裁影响俄罗斯 P40 库存**：一位成员指出，**制裁**可能会减少**俄罗斯的 P40** 库存。他们幽默地补充说，这显示了*制裁正以一种奇怪的方式发挥作用*。
- **家用服务器搭建见解**：一位用户询问了**家用设置的服务器规格**。另一位分享了他们的配置：*"Linux 工作站，配备 R3700X, 128GB RAM, RTX 4090, 2个 1TB SSD，以及一些 3.5 英寸 SATA HDD"*。


  

---

### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1250536384717127680)** (16 条消息🔥): 

- **LLAMA3 70B 生成多样化输出**：一位用户报告称，当从空白文档（`<|begin_of_text|>`）提示 **LLAMA3 70B** 时，模型会生成 *“60% AI2 arc 格式的评估数据、20% wiki 文本和 20% 代码”*。这表明 **LLAMA3** 可能针对特定格式进行了最终的 eval tuning。
  
- **请求协助调试 VLLM**：一名成员请求帮助调试 **VLLM**，另一名成员回复称自己不是专家，但研究过该仓库。

- **针对长文本微调 BERT**：一位用户寻求关于使用滑动窗口（sliding window）方法为 6000 词输入微调 **BERT** 模型的建议。他们被引导至特定资源，包括一篇 [NeurIPS 论文](https://proceedings.neurips.cc/paper_files/paper/2020/file/96671501524948bc3937b4b30d0e57b9-Paper.pdf) 及其 [实现](https://github.com/Sleepychord/CogLTX)。

- **Mixture of Millions of Memory Experts 作为 RAG 替代方案**：一位用户提到了一篇关于采用 **Mixture of Millions of Memory Experts** 进行事实记忆并减少幻觉（hallucinations）以作为 **RAG** 替代方案的 [研究论文](https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf)。另一名成员认为这种方法可能多余，因为之前可能已经尝试过。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://open-engineer.vercel.app/">Open Engineer</a>：一个免费的教育平台，通过实战项目、实用工具和基础理论提供 AI 学习。</li><li><a href="https://github.com/lamini-ai/Lamini-Memory-Tuning/blob/main/research-paper.pdf">Lamini-Memory-Tuning/research-paper.pdf at main · lamini-ai/Lamini-Memory-Tuning</a>：消除 LLM 幻觉需要重新思考泛化 - lamini-ai/Lamini-Memory-Tuning
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1250534867716935710)** (52 条消息🔥): 

- **Samba-3.8B 实现完美检索**：[微软的 Samba 模型](https://github.com/microsoft/Samba) 在 3.2 万亿 token 上训练，在主要基准测试（benchmarks）中显著优于 Phi3-mini。它在保持序列长度线性复杂度的同时，实现了长上下文检索能力。
  
- **Passkey 检索的挑战**：成员们讨论了 Samba 处理 256k 序列长度 passkey 检索的能力。对话涉及了 Mamba 层和 SWA 在分别处理长期和局部信息方面的效率，其中 rope 可能会影响结果。

- **寻找神经网络论文**：一位用户试图回忆一篇关于神经网络在 OOD 数据下行为的论文，提到了“均值回归”（mean reversion）等概念。另一位用户迅速找到了 [可能的论文](https://arxiv.org/abs/2310.00873) 并分享了链接及其他资源。

- **自合成高质量指令数据**：介绍了一种名为 [Magpie](http://arxiv.org/abs/2406.08464) 的新方法，该方法通过仅使用用户输入开始模板（start-of-user-input template）提示对齐的 LLM 来合成指令数据。这种自合成方法旨在生成大规模对齐数据，绕过手动创建数据的需求。

- **关于绑定 embedding 和 unembedding 层的讨论**：一名成员询问了绑定 embedding 和 unembedding 层的缺点。另一名成员分享了一篇 [LessWrong 帖子](https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are)，解释了现代 LLM 已经不再采用这种做法，尽管缺乏实证数据。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="http://arxiv.org/abs/2406.08478">What If We Recaption Billions of Web Images with LLaMA-3?</a>：从网络爬取的图像-文本对本质上是存在噪声的。先前的研究表明，在语义上对齐并丰富这些配对的文本描述可以显著增强模型训练...</li><li><a href="http://arxiv.org/abs/2406.08464">Magpie: Alignment Data Synthesis from Scratch by Prompting Aligned LLMs with Nothing</a>：高质量的指令数据对于对齐大语言模型（LLM）至关重要。尽管像 Llama-3-Instruct 这样的模型已经开源了权重，但它们的对齐数据仍然是私有的，这阻碍了...</li><li><a href="https://arxiv.org/abs/2406.07887">An Empirical Study of Mamba-based Language Models</a>：像 Mamba 这样的选择性状态空间模型（SSMs）克服了 Transformer 的一些缺点，例如随序列长度呈二次方增长的计算复杂度和推理时巨大的内存需求...</li><li><a href="http://arxiv.org/abs/2406.07548">Image and Video Tokenization with Binary Spherical Quantization</a>：我们提出了一种基于 Transformer 的新型图像和视频分词器，采用二进制球面量化（Binary Spherical Quantization, BSQ）。BSQ 将高维视觉嵌入投影到低维超球面上，然后应用...</li><li><a href="https://arxiv.org/abs/2106.00003">Parallelized Computation and Backpropagation Under Angle-Parametrized Orthogonal Matrices</a>：我们提出了一种在存在矩阵正交性和单位性约束的情况下并行加速学习的方法，这些约束在机器学习的多个分支中都备受关注。我们展示了如何通过一种应用...</li><li><a href="https://arxiv.org/abs/2310.00873">Deep Neural Networks Tend To Extrapolate Predictably</a>：传统观点认为，神经网络在面对分布外（OOD）输入时，其预测往往是不可预测且过度自信的。我们的工作重新评估了神经网络的这一假设...</li><li><a href="https://arxiv.org/abs/2311.14648">Calibrated Language Models Must Hallucinate</a>：最近的语言模型以惊人的频率生成虚假但听起来合理的文本。这种“幻觉”是基于语言的 AI 系统可用性的障碍，并且可能损害...</li><li><a href="https://x.com/sakanaailabs/status/1801069076003082502?s=46">Tweet from Sakana AI (@SakanaAILabs)</a>：LLM 能发明更好的训练 LLM 的方法吗？在 Sakana AI，我们正在开拓 AI 驱动的方法来自动化 AI 研究和发现。我们很高兴发布 DiscoPOP：一种新的 SOTA 偏好优化...</li><li><a href="https://arxiv.org/abs/2406.08070v1">CFG++: Manifold-constrained Classifier Free Guidance for Diffusion Models</a>：无分类器引导（CFG）是现代扩散模型中用于文本引导生成的基石工具。虽然有效，但 CFG 也有显著的缺点。例如，带有 CFG 的 DDIM 缺乏可逆性...</li><li><a href="https://www.lesswrong.com/posts/pHPmMGEMYefk9jLeh/llm-basics-embedding-spaces-transformer-token-vectors-are">LLM Basics: Embedding Spaces - Transformer Token Vectors Are Not Points in Space — LessWrong</a>：这篇文章旨在解释我在刚开始接触 Transformer 嵌入时产生的一个误解。感谢 Stephen Fowler 的...
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1250618347859607613)** (10 messages🔥): 

- **关于在相同分词器下使用按 Token 还是按字节计算 acc_norm 的辩论**：一位成员质疑在比较使用相同分词器的模型时，使用**按 Token 计算 acc_norm** 而非按字节计算是否合适。另一位成员指出“很多人都这么做”，并建议参考 GPT-3 论文以获取更多细节。
  
- **Qwen1.5-7B-Chat 的 Generate_until 问题**：一位成员报告称，在 dtype='float' 下对 Qwen1.5-7B-Chat 运行 **truthfulqa_gen** 任务时，749 个问题中有 670 个返回了空响应，并附上了 [pastebin 日志](https://pastebin.ai/i6qnlbg8x3) 供参考。另一位成员认为遗漏了 `--apply_chat_template` 可能是原因，理由是 Prompt 格式问题，并建议使用 `--fewshot_as_multiturn` 作为可能的修复方案。

**提到的链接**：<a href="https://pastebin.ai/i6qnlbg8x3">log_samples truthfulqa_gen - Pastebin.ai</a>：未找到描述

  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 messages): 

yash05880: 也许你还不知道 https://laion.ai/blog/open-flamingo/
  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1250546854442045481)** (3 messages): 

- **PingCap 展示了使用 TiDB 和 LlamaIndex 的 RAG 应用**：我们在 PingCap 的朋友们利用 LlamaIndex 的知识图谱功能，围绕他们的 TiDB 数据库构建了一个高质量的 **RAG 应用程序**。它是开源的，你可以在[这里](https://t.co/JKOa6ab1Uh)试用，或者在 [GitHub](https://t.co/bUWs9lM1ea) 上查看代码。

- **活动预告：巴黎 AI Infrastructure Meetup**：6 月 19 日星期三，欢迎在巴黎 Station F 参加 **AI Infrastructure Meetup**，届时将有来自 LlamaIndex 的 Pierre-Loic Douclet 以及来自 **Gokoyeb** 和 **Neon** 的演讲者。更多详情和注册请点击[这里](https://twitter.com/llama_index/status/1801288913312760205)。

- **Mixture-of-Agents (MoA) 增强 LLMs 能力**：**TogetherCompute** 的一项研究表明，仅使用开源 LLMs 的 **Mixture-of-Agents (MoA)** 设置可以显著增强任务处理能力。点击[这里](https://t.co/awJyjj1F2W)了解更多关于 MoA 的潜力。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://t.co/EPwlqsQ4kq">AI Infrastructure · Luma</a>: 📣 召集所有 AI 开发者和基础设施爱好者！我们很高兴地宣布将于 6 月 19 日星期三举行 AI Infrastructure Meetup！The Linux…</li><li><a href="https://t.co/JKOa6ab1Uh">TiDB.AI</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1250542964124422287)** (61 messages🔥🔥): 

- **处理大型 Embedding 索引**：一位成员在处理包含 170,000 个 Embedding 的索引查询时间时遇到困难，建议使用 **Qdrant** 或 FAISS Index 等向量数据库来加快检索速度。他们分享了详细的 FAISS 索引和查询代码，寻求解决查询过程中出现的 **AssertionError**。

- **节点被标记为 "Unknown"**：用户正在讨论为什么属性图中的某些节点显示为 "Unknown"。建议包括隐式提取器和源/父关系可能存在的问题，并推荐使用 `pip install -U llama-index-graph-stores-neo4j` 进行修复。

- **从 VectorStores 中提取节点**：一位尝试从使用 **Chroma** 构建的 **VectorStoreIndex** 中提取节点的用户遇到了空字典，并学会了使用 `vector_store._collection.get()` 来检索节点。他们寻求关于直接从 VectorStoreIndex 执行此任务的进一步帮助。

- **过滤节点检索结果**：用户讨论了过滤 `index.as_retriever().retrieve()` 结果的不同方法，重点关注 metadata 过滤器、相似度后处理器（postprocessors）以及针对 Qdrant 和 Chroma 等特定向量存储使用 `get_nodes()`。

- **Qdrant 数据库节点检索**：一位使用包含法律文本条目的 Qdrant 数据库的用户询问如何在查询期间检索相邻节点（前一篇文章和后一篇文章）。建议涉及使用节点关系以及最新 Qdrant 向量存储中的新 API 方法。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://docs.llamaindex.ai/en/latest/module_guides/deploying/agents/tools/#return-direct>).">Tools - LlamaIndex</a>: 未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-core/llama_index/core/evaluation/context_relevancy.py">llama_index/llama-index-core/llama_index/core/evaluation/context_relevancy.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>: LlamaIndex 是用于 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/#custom-node-postprocessor>))">Node Postprocessor - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/node_postprocessors/#metadatareplacementpostprocessor>))">Node Postprocessor Modules - LlamaIndex</a>: 未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/module_guides/querying/node_postprocessors/#using-with-retrieved-nodes>))">Node Postprocessor - LlamaIndex</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1250796661295550504)** (1 messages): 

- **使用 LLM-Index 将 PDF 嵌入到 Weaviate**：一位成员征求关于使用 LLM-Index 将 PDF 和文档嵌入到 Weaviate 向量数据库的建议。该消息表明这是一个正在进行的项目，但未包含更多细节或回复。

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1250525374799024209)** (56 messages🔥🔥): 

- **Coral 品牌重塑为 Command-R**：一位用户询问了 **Coral** 的状态，得到的澄清是它现在被称为 **Command-R**，但原始的 Command-R 和 Coral 仍在运行。 
- **模型使用参数和 Prompt 控制**：讨论强调了模型参数调整中的不同做法，一些用户倾向于通过 **Prompt Engineering 而非参数微调**来实现更好的控制，而另一些用户则分享了他们认为有效的特定配置。
- **关于可接受使用政策的澄清**：用户分享并询问了 **[Cohere Acceptable Use Policy](https://docs.cohere.com/docs/c4ai-acceptable-use-policy)**，寻求关于私人用途与商业用途的澄清，个人项目被允许用于向雇主展示。
- **API 的内部服务器错误**：遇到 `ApiError: status_code: 500` 错误的用户讨论了潜在原因，包括特定 Prompt 内部的错误与请求问题，其他人建议检查请求详情。
- **Trial key 限制和访问问题**：用户报告了 **Trial key** 提示权限不足并达到使用限制的问题，而其他人确认了类似经历，但指出使用 Production key 成功。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://cohere.com/terms-of-use">Terms Of Use</a>: 放心访问和使用 Cohere 的 Natural Language Processing 和 Large Language Models，确保您充分了解我们的使用条款。</li><li><a href="https://docs.cohere.com/docs/c4ai-acceptable-use-policy">C4AI Acceptable Use Policy</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1250680242222272613)** (3 messages): 

- **Glarb-Glarb 和 Fluent API 偏好**：一位成员幽默地提到了 "Glarb-Glarb"，并表达了他们对 **Fluent API** 的喜爱。
- **祝贺发布并感谢对 Cohere 的支持**：另一位成员祝贺某人的发布，并对添加 **Cohere 支持**表示感谢，最后以敬礼表情符号结束。
  

---

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1250545818918977537)** (41 条消息🔥): 

- **Stable Diffusion 在生成女性图像方面遇到困难**：讨论指出，由于严格的审查，StabilityAI 的 **Stable Diffusion** 模型在生成裸体甚至穿着衣服的女性图像时表现不佳。用户建议等待自定义 checkpoints，并使用 SD1.5 进行二次 img2img 处理以达到预期效果。[讨论链接](https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67)。

- **庆祝 Luma AI 发布新的 text-to-video 模型**：用户们祝贺一名成员发布了 Luma AI 的 **Dream Machine**，这是一个 text-to-video 模型。尽管最初存在一些服务器问题，该模型显示出巨大的潜力，不过用户注意到其性能在处理复杂 prompts 时会有所波动。[尝试 Dream Machine](https://lumalabs.ai/dream-machine)。

- **不同 AI 模型的对比**：有一个分享链接对比了 **SD3 Large, SD3 Medium, Pixart Sigma, DALL E 3 和 Midjourney**。该帖子讨论了 /r/StableDiffusion subreddit 的重新开放以及 Reddit API 变更带来的持续问题。[对比链接](https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/)。

- **对视频 tokenization 模型的关注**：一位用户询问了将视频转换为 token 序列的最佳开源模型，类似于 **VAE**。另一位用户承认他们只能提供基础视频编辑任务的帮助，如分割视频和去除水印。[推文链接](https://fxtwitter.com/blizaine/status/1801126160904098247)。

- **Luma 视频生成的潜力**：用户评论称 **Luma 的 Dream Machine** 视频生成具有潜力，尽管表现尚不稳定。关于其能力和改进空间的讨论正在进行中。

<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium/discussions/67">stabilityai/stable-diffusion-3-medium · 该模型无法创建裸体女性图像</a>: 未找到描述</li><li><a href="https://fxtwitter.com/blizaine/status/1801126160904098247">来自 Blaine Brown  (@blizaine) 的推文</a>: 来自 @LumaLabsAI 的 Dream Machine 真的让梗图活了起来！线索 🧵</li><li><a href="https://fxtwitter.com/dome_271/status/1800922604511105246">来自 dome | Outlier (@dome_271) 的推文</a>: 出发吧！很高兴分享我们的第一个 TEXT-TO-VIDEO 模型！ https://lumalabs.ai/dream-machine 引用 Luma AI (@LumaLabsAI) 介绍 Dream Machine - 下一代视频模型 ...</li><li><a href="https://www.reddit.com/r/StableDiffusion/comments/1deeqhe/sd3_large_vs_sd3_medium_vs_pixart_sigma_vs_dall_e/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://fxtwitter.com/blizaine/status/1801126279917547726">来自 Blaine Brown  (@blizaine) 的推文</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1250616787385581649)** (14 条消息🔥): 

- **AIW 研究揭示模型不稳定性**：在讨论 AIW 问题的变体时，该研究揭示了像 GPT-4o 这样的模型在面对微小变化（例如改变数值）时会出现**剧烈的崩溃**。分析强调“GPT-4o（以及所有其他模型）对 AIW 变体不具鲁棒性”，突显了推理能力方面的核心缺陷 ([source](https://arxiv.org/abs/2406.02061))。

- **DataComp-1B 改进的 Captions**：从 Haoqin Tu 个人网站分享的链接详细介绍了增强网络爬取的噪声图像-文本对文本描述的工作，通过对齐改进了模型训练 ([source](https://www.haqtu.me/Recap-Datacomp-1B/)) ([paper](https://arxiv.org/abs/2406.08478))。

- **简化的 Masked Diffusion 模型**：一篇分享的论文为 Masked Diffusion 模型提出了一个**简单且通用的框架**，在 GPT-2 规模下通过 Perplexity 评估时，表现优于之前的模型 ([source](https://arxiv.org/abs/2406.04329))。

- **重标注 (Recaptioning) 与数据集更新**：多次提及表明正在积极进行 **CC12M** 和 **GBC10M** 等数据集的重标注工作，并提供了 Hugging Face 上补充资源的链接 ([GBC10M](https://huggingface.co/datasets/graph-based-captions/GBC10M), [CC12M LLaVaNext](https://huggingface.co/datasets/CaptionEmporium/conceptual-captions-cc12m-llavanext))。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2406.08478">What If We Recaption Billions of Web Images with LLaMA-3?</a>：网络爬取的图像-文本对本质上是有噪声的。先前的研究表明，对这些配对进行语义对齐和丰富文本描述可以显著增强模型训练...</li><li><a href="https://arxiv.org/abs/2406.02061">Alice in Wonderland: Simple Tasks Showing Complete Reasoning Breakdown in State-Of-the-Art Large Language Models</a>：大语言模型 (LLMs) 经常被描述为基础模型——即能够以 few-shot 或 zero-shot 方式在各种任务和条件下进行强大迁移的模型，而...</li><li><a href="https://www.haqtu.me/Recap-Datacomp-1B/">What If We Recaption Billions of Web Images with LLaMA-3?</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2406.04329">Simplified and Generalized Masked Diffusion for Discrete Data</a>：Masked (或 absorbing) diffusion 作为自回归模型的替代方案，正被积极探索用于离散数据的生成建模。然而，该领域的现有工作一直受到不必要的...
</li>
</ul>

</div>
  

---


### **LAION ▷ #[resources](https://discord.com/channels/823813159592001537/991938328763056168/)** (1 条消息): 

sidfeels: <@&825830190600683521>
  

---


### **LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/)** (1 条消息): 

.michu7: <@&825830190600683521>
  

---


### **LAION ▷ #[paper-discussion](https://discord.com/channels/823813159592001537/1172520224797511700/)** (1 条消息): 

zuwop21: 50$ from steam
[steamcommunity.com/glft/918524](https://sc.link/HSvw7)
@everyone
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1250589784586453112)** (49 条消息🔥): 

- **寻求语音转文本数据集和解决方案**：一位成员询问关于包含 MP3 文件和带有 Diarization（说话人日志）原始转录文本的语音转文本数据集推荐。他们还提到曾使用过 AWS Transcribe、OpenAI Whisper 和 Deepgram Nova-2，并正在寻找该领域现有的最佳解决方案。

- **处理 Chain 中的空工具调用**：一位用户正在处理一个问题，即当 'tool_calls' 列表为空时，Chain 会抛出错误。他们正在寻求关于如何管理用户输入不需要使用工具的场景的建议，以确保 Chain 能够无错地处理简单的响应。

- **评估 LLM 消息相似度**：另一位用户询问了 LangChain 如何处理 LLM 消息与预期脚本消息之间的相似度测量。回复解释说，LangChain 可以使用字符串距离指标或 Embedding 距离指标来评估这种相似度，并提供了实际的代码示例。

- **LangGraph 中的状态管理设计模式**：关于在 LangGraph 中管理状态进行了深入讨论，特别关注将 user ID 和 thread ID 集成到状态中。对话集中在状态应该包含整个对话历史还是仅包含最近的交互，并提供了详细的示例和最佳实践。

- **在 LangChain 中处理人工干预和流式响应**：成员们讨论了在 LangGraph 工作流中管理人工干预以及无缝恢复对话的方法。还有关于在保留聊天历史的同时实现流式响应的问题，并提供了代码示例来演示如何在多轮对话中保持上下文。

<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://python.langchain.com/v0.2/docs/templates/chat-bot-feedback/#usage>).">Chat Bot Feedback Template | 🦜️🔗 LangChain</a>: 该模板展示了如何在没有显式用户反馈的情况下评估聊天机器人。它在 chain.py 中定义了一个简单的聊天机器人，并使用自定义评估器根据后续内容对机器人响应的有效性进行评分...</li><li><a href="https://docs.smith.langchain.com/how_to_guides/evaluation/use_langchain_off_the_shelf_evaluators#use-string-or-embedding-distance-metrics>).">Use LangChain off-the-shelf evaluators (Python only) | 🦜️🛠️ LangSmith</a>: 在深入研究此内容之前，阅读以下内容可能会有所帮助：</li><li><a href="https://github.com/langchain-ai/langchain/issues/15934>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>)">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: 这里我们重点介绍如何从旧版 LangChain Agent 迁移到 LangGraph</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/qa_chat_history_how_to/#agents>))">How to add chat history | 🦜️🔗 LangChain</a>: 在许多问答应用中，我们希望允许用户进行来回对话，这意味着应用需要某种对过去问题和答案的“记忆”，以及一些逻辑...</li><li><a href="https://js.langchain.com/v0.2/docs/how_to/migrate_agent/#basic-usage>))">How to migrate from legacy LangChain agents to LangGraph | 🦜️🔗 Langchain</a>: 这里我们重点介绍如何从旧版 LangChain Agent 迁移到 LangGraph</li><li><a href="https://github.com/langchain-ai/langchain/issues/18598>))">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1250670200982994994)** (2 messages): 

- **Tiny AI Client 简化 LLM 使用**：一位成员在 GitHub 上分享了一个名为 [tiny-ai-client](https://github.com/piEsposito/tiny-ai-client) 的项目，将其描述为“适用于简单用例的微型 LLM 客户端”，支持 OAI、Anthropic 和 Gemini 模型的 tools 和 vision 功能。他们希望这对社区中的其他人有所帮助。
- **使用 Docker 和 Ollama 在本地运行 LLM**：另一位成员制作了一个 [YouTube 视频](https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy)，演示了如何使用 Docker 和 Ollama 在本地运行 LLM。他们邀请大家对视频提供反馈，视频标题为 "Exécuter des LLMs en local avec Docker et Ollama"。

**提及的链接**：<a href="https://youtu.be/NLOY9RLMI6k?si=-OdUtYSWTJwhvtzy">Exécuter des LLMs en local avec Docker et Ollama</a>：未找到描述。

  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1250645308661039224)** (1 messages): 

- **在 Google Colab 上设置 LLM 的新教程**：一位成员在 **GitHub** 上分享了一份关于在 Google Colab 上设置 **LLM** 的*详尽教程*，充分利用了免费的 15GB Tesla T4 Colab GPU。该教程包含了 **GPU 加速**和**仅 CPU** 推理的说明，链接见[此处](https://github.com/casualcomputer/llm_google_colab)。

**提及的链接**：<a href="https://github.com/casualcomputer/llm_google_colab">GitHub - casualcomputer/llm_google_colab: A tutorial on how to set up a LLM on Google Colab for both GPU-accelerated and CPU-only session.</a>：关于如何在 Google Colab 上为 GPU 加速和仅 CPU 会话设置 LLM 的教程。- casualcomputer/llm_google_colab

  

---



### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1250569249353633812)** (7 messages): 

- **暗示 Windows 支持将于秋季发布**：一位成员询问 Windows 支持何时可用。另一位成员回答，猜测会在夏末或秋季发布，并引用了即将举行的 [YouTube 直播](https://m.youtube.com/watch?v=uookgZ7Ojg8)以获取更新。
- **WSL 作为 Windows 用户的变通方案**：一位成员报告称成功使用 WSL (Windows Subsystem for Linux) 作为在 Windows 上进行 Mojo 开发的临时变通方案。然而，提问者更倾向于使用原生 Windows 构建环境来处理 PE 格式。

**提及的链接**：<a href="https://m.youtube.com/watch?v=uookgZ7Ojg8">Modular Community Livestream - New in MAX 24.4</a>：MAX 24.4 现已发布！加入我们即将举行的直播，我们将讨论 MAX Engine 和 Mojo🔥 的新功能 —— macOS 上的 MAX、MAX Engine 量化 API 等...

  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1250553872603287633)** (31 messages🔥): 

- **Mojo 中的非空字符串为真值 (Truthy)**：一位用户发现 Mojo 中的任何非空字符串都会被评估为真值，这导致其代码中出现了一些意外行为。对此的解释是，像 `String.ASCII_LOWERCASE` 这样的常量是非空字符串，因此评估结果始终为 True。
- **Neovim 中的 Mojo LSP 配置**：有人询问如何在 Neovim 中设置 Mojo LSP。得到的回答是一个 [GitHub 链接](https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo)，确认其已预装在 Neovim 中。
- **Mojo 与 Python 的矩阵乘法性能对比**：一位用户讨论了在小型固定尺寸矩阵乘法中 Mojo 和 Python 的性能基准测试。他们发现 Mojo 的运行速度明显快于 Python，因为 Python 在处理小型计算时使用 numpy 等库会产生额外开销。
- **Mojo 中的迭代器和循环行为**：一些用户讨论了 Mojo 中循环变量的重新赋值行为，特别是 `for` 循环的使用以及在循环内部修改 `i` 的情况。建议如果需要持久修改，应使用 `while` 循环。
- **Mojo 中的 Stdin 和 Stdout**：有人提问关于 Mojo 缺乏 `stdin` 支持的问题，已确认目前尚不支持。

**提及的链接**：<a href="https://github.com/neovim/nvim-lspconfig/blob/master/doc/server_configurations.md#mojo">nvim-lspconfig/doc/server_configurations.md at master · neovim/nvim-lspconfig</a>：Nvim LSP 的快速入门配置。欢迎在 GitHub 上为 neovim/nvim-lspconfig 开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1250595066184073226)** (13 messages🔥): 

- **讨论了条件非一致性 (Conditional non-conformance)**：一位成员建议需要条件非一致性，以便“消除通过解引用可能产生的隐式复制”。另一位成员提到这可能由 ExplicitlyCopyable trait 和 CollectionElementNew 处理。
- **编译器限制排查**：当一位成员询问某个编译器错误是否为 bug 时，另一位成员澄清这很可能是编译器限制。建议的变通方法是使用 `fn (Optional[T]) capturing -> Bool`，同时假设参数永远不会为 none。
- **发布新的 nightly Mojo 编译器版本**：发布了最新的 nightly Mojo 编译器版本 `2024.6.1305`，并附带了 changelog 和 raw diff 链接。一位成员幽默地提到要记得使用 `modular update nightly/max` 而不是 `modular update nightly/mojo` 进行更新。
- **建议为更新设置别名 (Alias)**：针对更新命令的混淆，一位成员建议设置别名以简化流程。这被幽默地公认为“Actual Intelligence”（真实智能）的一个好例子。

- **讨论了条件非一致性 (Conditional non-conformance)**：一位成员建议需要条件非一致性，以便“消除通过解引用可能产生的隐式复制”。另一位成员提到这可能由 ExplicitlyCopyable trait 和 CollectionElementNew 处理。
- **编译器限制排查**：当一位成员询问某个编译器错误是否为 bug 时，另一位成员澄清这很可能是编译器限制。建议的变通方法是使用 `fn (Optional[T]) capturing -> Bool`，同时假设参数永远不会为 none。
- **发布新的 nightly Mojo 编译器版本**：发布了最新的 nightly Mojo 编译器版本 `2024.6.1305`，并附带了 changelog 和 raw diff 链接。一位成员幽默地提到要记得使用 `modular update nightly/max` 而不是 `modular update nightly/mojo` 进行更新。
- **建议为更新设置别名 (Alias)**：针对更新命令的混淆，一位成员建议设置别名以简化流程。这被幽默地公认为“Actual Intelligence”（真实智能）的一个好例子。

  

---

### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1250530820779937933)** (33 messages🔥): 

- **OpenAI 的营收在没有 Microsoft 帮助的情况下飞速增长**：OpenAI 的营收在过去六个月中几乎翻了一番，主要来自 ChatGPT 和其他 OpenAI 产品的直接销售，而非来自 Microsoft 的销售。[来源](https://www.theinformation.com/articles/openais-annualized-revenue-doubles-to-3-4-billion-since-late-2023?utm_source=ti_app&rc=c48ukx)。
- **Sakana AI 推出 DiscoPOP**：Sakana AI 推出了 DiscoPOP，这是一种由 LLM 发现并编写的新型 SOTA 偏好优化算法，展示了自动化 AI 研究的 AI 驱动方法。[查看详情](https://x.com/SakanaAILabs/status/1801069076003082502) 并探索 [论文](https://arxiv.org/abs/2406.08414) 和 [GitHub 仓库](https://github.com/SakanaAI/DiscoPOP)。
- **Mira Murati 对 OpenAI 的模型发表评论**：Mira Murati 提到，OpenAI 实验室中可用的 AI 模型并不比公开可用的模型先进多少。[来源](https://x.com/tsarnick/status/1801022339162800336)。
- **关于 Reka 收购的传闻**：传闻 Reka 将被 Snowflake 以 10 亿美元收购，但不久后，Reka 宣布与 Shutterstock 建立长期合作伙伴关系。这一推测突显了 AI 行业收购的动态性质。
- **Apple 与 OpenAI 令人惊讶的合作伙伴关系**：Apple 与 OpenAI 的合作更多侧重于在其设备上推广 OpenAI 的品牌和技术，而不是产生大量的初始收入。[详细文章](https://www.bloomberg.com/news/articles/2024-06-12/apple-to-pay-openai-for-chatgpt-through-distribution-not-cash)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.bloomberg.com/news/articles/2024-06-12/apple-to-pay-openai-for-chatgpt-through-distribution-not-cash">Bloomberg - Are you a robot?</a>：未找到描述</li><li><a href="https://x.com/amir/status/1800992276652630455">来自 Amir Efrati (@amir) 的推文</a>：新闻：OpenAI 的营收在过去 6 个月内~翻倍~。虽然许多人认为大量营收来自 Microsoft 销售 OpenAI 技术并给这家初创公司分成……恰恰相反。这……</li><li><a href="https://x.com/SakanaAILabs/status/1801069076003082502">来自 Sakana AI (@SakanaAILabs) 的推文</a>：LLM 能发明更好的训练 LLM 的方法吗？在 Sakana AI，我们正在开拓 AI 驱动的方法来自动化 AI 研究和发现。我们很高兴发布 DiscoPOP：一种新的 SOTA 偏好优化……</li><li><a href="https://x.com/tsarnick/status/1801022339162800336">来自 Tsarathustra (@tsarnick) 的推文</a>：Mira Murati 表示 OpenAI 实验室中的 AI 模型并不比公开可用的模型先进多少。
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1250734937133285448)** (4 messages): 

- **讨论了不寻常的论文提交**：一位成员分享了来自 [Cosmin Negruseri](https://x.com/cosminnegruseri/status/1800683283069767691) 的推文，并链接到了一篇 arXiv 论文 [2404.07221](https://arxiv.org/abs/2404.07221)，指出其不寻常之处。另一位成员幽默地回应，用“Hahahhahaha what”表示惊讶。

**提到的链接**：<a href="https://x.com/cosminnegruseri/status/1800683283069767691">来自 Cosmin Negruseri (@cosminnegruseri) 的推文</a>：以前没在论文里见过这个

### **Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1250550474684366898)** (6 messages): 

- **Nvidia Nemotron 热度上升**：一位成员分享了一条 [推文](https://x.com/SebastianB929/status/1800991419437367655)，暗示 Nvidia Nemotron 可能发布。这条推文引发了社区对新硬件的兴奋。

- **关于语音模型的最新研究论文**：一位成员发布了由 Jupinder Parmar 和 Shrimai Prabhumoye 等多位研究人员撰写的 [arXiv 研究论文](https://arxiv.org/abs/2402.16819) 链接。该论文探讨了语音建模方面的进展。

- **来自内部联系的见解**：另一位成员评论道：“他们的 Alignment 负责人是付费订阅者和朋友，但我觉得他不在 Discord 里。”这引发了对之前讨论项目潜在内部信息的关注。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/SebastianB929/status/1800991419437367655">来自 SebastianBoo (@SebastianB929) 的推文</a>：似乎是 Nvidia Nemotron，引用自 Xeophon (@TheXeophon) june-chatbot👀</li><li><a href="https://arxiv.org/abs/2402.16819">Nemotron-4 15B 技术报告</a>：我们介绍了 Nemotron-4 15B，这是一个拥有 150 亿参数的大型多语言语言模型，在 8 万亿个文本 Token 上进行了训练。Nemotron-4 15B 在英语、多语言评估中表现出强劲的性能...
</li>
</ul>

</div>
  

---


### **Interconnects (Nathan Lambert) ▷ #[nlp](https://discord.com/channels/1179127597926469703/1208183200099344445/1250824109898862752)** (7 messages): 

- **Samba 3.8B 横扫基准测试**：[Liliang Ren 的推文](https://x.com/liliang_ren/status/1801027052147216457) 介绍了 Samba 3.8B，这是一种结合了 Mamba 和 Sliding Window Attention 架构的模型，在 MMLU、GSM8K 和 HumanEval 等主要基准测试中显著超越了 Phi3-mini。该架构具有线性复杂度的无限上下文长度（[论文在此](https://arxiv.org/abs/2406.07522)）。
- **SSM 依然稳健**：成员们对 **SSM** (Structured State Machines) 仍在被积极开发和讨论表示欣慰和乐观。一位成员幽默地评论说，他们对 SSM 是否会继续发展持 50/50 的态度，并指出胜算尚可。
- **混合架构可能是未来**：大家达成共识，趋势正向 **SSM/Transformer 混合架构** 发展，理由是并非每一层都需要 Attention。

**提及的链接**：<a href="https://x.com/liliang_ren/status/1801027052147216457">来自 Liliang Ren (@liliang_ren) 的推文</a>：介绍 Samba 3.8B，一个简单的 Mamba+Sliding Window Attention 架构，在主要基准测试（如 MMLU、GSM8K 和 HumanEval）上大幅超越 Phi3-mini。😮 而且它具有无限的...

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1250543546579161169)** (43 messages🔥): 

- **Haize Labs 发布 AI 越狱检测**：一家名为 [Haize Labs](https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww) 的初创公司发布了一份宣言，专注于预先发现并消除 AI 系统中的故障模式。他们展示了越狱行业领先 AI 模型安全防护栏的能力，揭示了严重的安全违规行为。
  
- **tldraw 开源 iPad 计算器复刻版**：[tldraw](https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww) 分享了 Apple iPad 计算器的开源复刻版，引发了关注和赞赏。tldraw 团队虽然规模较小，但其创新的思维实验和快速的演示不断给人留下深刻印象。

- **关于 Amazon AI 困境的文章**：[cakecrusher](https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/) 分享的一篇文章讨论了 Amazon 的文化和运营如何导致其在对话式 AI 发展中落后。前员工提到了诸如复杂的构建/部署系统以及产品优先的心态阻碍了长期发展等问题。

- **OpenAI 营收飙升**：正如 @deedydas 所分享的，OpenAI 的年化营收已达到 [\$34 亿](https://x.com/deedydas/status/1801003523292729789)。随后引发了关于如此惊人的营收数据背后的盈利能力和潜在烧钱率的讨论。

- **Argilla 加入 Hugging Face**：[Argilla](https://argilla.io/blog/argilla-joins-hugggingface) 宣布与 Hugging Face 合并，承诺在生成有价值的数据集和内容方面增强协同效应。该伙伴关系旨在放大双方团队在构建优秀产品和促进 AI 领域创新方面的努力。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=sQar5NNGbw4">扩展可解释性</a>：科学与工程密不可分。我们的研究人员反思了科学与工程进步之间的紧密关系，并讨论了技术...</li><li><a href="https://www.mihaileric.com/posts/how-alexa-dropped-the-ball-conversational-ai/">未找到标题</a>：未找到描述</li><li><a href="https://x.com/tldraw/status/1800515870709706879?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 tldraw (@tldraw) 的推文</a>：我有一个想法</li><li><a href="https://x.com/tldraw/status/1800515870709706879?s=46&t=">来自 tldraw (@tldraw) 的推文</a>：我有一个想法</li><li><a href="https://x.com/haizelabs/status/1800936720990384174?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Haize Labs (@haizelabs) 的推文</a>：今天对语言模型来说是个糟糕透顶的日子。今天，我们发布了 Haize Labs 宣言。@haizelabs 对 AI 系统进行 haizes（自动红队测试），以预先发现并消除任何故障...</li><li><a href="https://x.com/alvarobartt/status/1801278221901512839?s=46">来自 Alvaro Bartolome (@alvarobartt) 的推文</a>：正如你可能已经知道的，@argilla_io 现在加入了 @huggingface 🎉 在专业层面，看到两个伟大的团队共享同样的激情是令人难以置信的。我坚信 Argilla 的使命...</li><li><a href="https://x.com/tldraw/status/1801212867061879175?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 tldraw (@tldraw) 的推文</a>：你的计算器能做到这一点吗？</li><li><a href="https://x.com/tldraw/status/1801264226314408029?s=46">来自 tldraw (@tldraw) 的推文</a>：成为数学</li><li><a href="https://x.com/realsharonzhou/status/1801271891954696317?s=46&t=PW8PiFwluc0tdmv2tOMdEg">来自 Sharon Zhou (@realSharonZhou) 的推文</a>：很高兴宣布 @LaminiAI Memory Tuning，一项新的研究突破！🎉 ◽95% 以上的准确率，将幻觉减少了 10 倍 ◽将任何开源 LLM 转换为 1M 路适配器 MoE（论文和 Lamini-1 模型权重...</li><li><a href="https://x.com/deedydas/status/1801003523292729789">来自 Deedy (@deedydas) 的推文</a>：OpenAI 的年化收入达到 34 亿美元。哇。</li><li><a href="https://x.com/the_aiju/status/1800986743832736129?s=46&t=90xQ8sGy63D2OtiaoGJuww">来自 Emily (@the_aiju) 的推文</a>：显然，你可以通过询问 LLM 是否存在各位数字之和为 9 的质数（正确答案是“不”）来对它进行 nerdsnipe（知识狙击）^^ 我尝试过的每一个模型都会发疯般地尝试它能找到的每一个质数...</li><li><a href="https://www.anthropic.com/research/engineering-challenges-interpretability">扩展可解释性的工程挑战</a>：Anthropic 是一家 AI 安全和研究公司，致力于构建可靠、可解释且可控的 AI 系统。</li><li><a href="https://x.com/liliang_ren/status/1801027052147216457?s=46&t=6FDPaNxZcbSsELal6Sv7Ug">来自 Liliang Ren (@liliang_ren) 的推文</a>：介绍 Samba 3.8B，一个简单的 Mamba + Sliding Window Attention 架构，在主要基准测试（如 MMLU、GSM8K 和 HumanEval）上大幅超越 Phi3-mini。😮 而且它具有无限的...</li><li><a href="https://github.com/microsoft/Samba">GitHub - microsoft/Samba: “Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling”的官方实现</a>： “Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling”的官方实现 - microsoft/Samba
</li>
</ul>

</div>
  

---



### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1250550501959925921)** (40 条消息🔥): 

- **澄清 Open Interpreter 的功能**：讨论明确了 **Open Interpreter** 赋予了 Large Language Models (**LLMs**) Agent 能力。一位成员解释道，“*OI 将自然语言转换为计算机控制*”，并考虑了未来与特定机器的 **LLMs** 和感官输入模型的潜在集成。
- **运行 Vision 模型**：成员们互相帮助，使用 **Open Interpreter** 同时运行代码和 Vision 模型，例如 `llama3-vision.py` 配置文件。他们讨论了下载模型以及让模型执行屏幕截图等任务的方法和问题。
- **浏览器自动化示例**：一位用户分享了他们成功让 **Open Interpreter** 通过简单命令浏览网页以获取当前独行侠队（Mavericks）比分的经历。他们强调了 Prompt 的简洁性以及可能导致延迟的服务器负载。
- **Whisper STT 库查询**：一位成员询问了一个好用/简单的 **Whisper Speech-To-Text (STT)** 库，随后提到他们创建了自己的解决方案。
- **性能与定制**：讨论涉及性能问题、服务器负载以及潜在的定制化，特别是通过修改 *core.py* 来达到预期效果。
  

---

### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1250857028201480204)** (5 messages): 

- **Apple 在 WWDC 中披露的模型参数量**：好奇 Apple 最近发布的模型参数量吗？这是一个 **30 亿参数 (3 billion parameter)** 的端侧语言模型。详细信息请参阅[此处](https://machinelearning.apple.com/research/introducing-apple-foundation-models)。
- **关于 Apple 与模型胜出的澄清**：有人询问“win”一词是指 Apple 的胜利还是某个模型的胜利。一位参与者澄清道：“我认为是指 Apple 的模型在与选定模型的对比中胜出”。
- **模型优化技术解析**：Apple 的端侧推理采用了 **low-bit palletization** 以及 2-bit 和 4-bit 混合配置策略。这种方法实现了平均 **3.5 bits-per-weight**，在优化内存、功耗和性能的同时，达到了与未压缩模型相同的准确度。

**提到的链接**：<a href="https://machinelearning.apple.com/research/introducing-apple-foundation-models">Introducing Apple’s On-Device and Server Foundation Models</a>：在 2024 年全球开发者大会（WWDC）上，我们推出了 Apple Intelligence，这是一个深度集成到……的个人智能系统。

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1250834851654013053)** (16 messages🔥): 

- **Docker Desktop 难以识别 GPU**：一名成员尝试在 Windows 11 的虚拟机（Ubuntu）上通过 Docker 运行 axolotl，但收到“未找到 GPU”的错误。他们运行了 `docker run --gpus all --rm -it winglian/axolotl:main-latest` 和 `accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml`，但均未成功。

- **建议使用 `nvidia-smi`**：另一名成员建议运行 `nvidia-smi` 来检查 GPU 状态。他们还询问了宿主系统是否安装了 CUDA toolkit。

- **在哪里安装 CUDA toolkit**：排障讨论引向了是在 Windows 还是 Ubuntu 上安装 CUDA toolkit 的问题。

- **分享 Nvidia toolkit 安装指南**：分享了 [NVIDIA 安装指南](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) 的链接，以帮助设置该工具包。

- **WSL 2 与 GPU 配置**：用户澄清他们正在使用 WSL 2，并尝试在 Ubuntu WSL 中配置 CUDA。他们表示感谢并提到将再次尝试该流程。

**提到的链接**：<a href="https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html">Installing the NVIDIA Container Toolkit &mdash; NVIDIA Container Toolkit 1.15.0 documentation</a>：未找到描述。

  

---



### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1250573286904168478)** (16 messages🔥): 

- **OpenRouter UI 限制不支持的参数**：当用户询问 OpenRouter 是否支持 Command R 的 Temp > 1 和 Min P 时，Alex Atallah 澄清说，虽然 UI 支持这些设置，但像 temp 这样的参数会被限制在 temp=1，而 Min P 则不会被传递。

- **Mistral 7B 模型的高响应延迟**：用户观察到所有 Mistral 7B 变体的响应时间都很长，并将其归因于上下文长度的变化以及模型可能的重新路由。讨论还指出了一次 [上下文长度调整](https://orw.karleo.net/changes) 以及 [模型运行时间追踪器](https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime) 显示的持续中断。

- **求职意向**：一名用户介绍自己是高级全栈与区块链开发工程师，表示拥有丰富的经验并正在寻求工作机会。

- **视觉模型需求**：另一名用户询问是否有计划添加更多视觉模型（如 cogvlm2），以获得更好的数据集标注（dataset captioning）能力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://openrouter.ai/models/mistralai/mistr">OpenRouter</a>：LLM 路由与市场</li><li><a href="https://orw.karleo.net/changes">OpenRouter API Watcher</a>：探索 OpenRouter 的模型列表和记录的变更。每小时更新一次。</li><li><a href="https://openrouter.ai/models/mistralai/mistral-7b-instruct%3Anitro/uptime">Mistral: Mistral 7B Instruct (nitro) – Uptime and Availability</a>：Mistral: Mistral 7B Instruct (nitro) 在各供应商的运行时间统计 - 一个高性能、行业标准的 7.3B 参数模型，针对速度和上下文长度进行了优化。注意：这是……
</li>
</ul>

</div>

### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1250708697714266204)** (3 messages): 

- **RDNA3 汇编悬赏热潮**：George Hotz 提到一个 [支持 RDNA3 汇编的 GitHub PR](https://github.com/tinygrad/tinygrad/pull/3637)，并邀请大家为该悬赏做出贡献。此更新旨在通过添加 RDNA3 汇编支持来增强 **tinygrad**。
- **建议设立 Qualcomm 内核级悬赏**：Hotz 还建议为“具有 HCQ graph 支持的 Qualcomm 内核级 GPU 驱动”设立悬赏，对于拥有 Qualcomm 智能手机和底层 Linux 知识的人来说，这是一个很好的起点。这为贡献 GPU 驱动开发提供了机会。
- **Tinygrad 在 Termux 中运行顺畅**：Hotz 确认 **tinygrad** 在 **Termux** 中运行良好。这意味着 tinygrad 具有通用性，可用于包括移动端在内的各种环境。

**提到的链接**：<a href="https://github.com/tinygrad/tinygrad/pull/3637">geohot 提交的 RDNA3 汇编支持 · Pull Request #3637 · tinygrad/tinygrad</a>：未找到描述

  

---


### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1250589848344199250)** (10 messages🔥): 

- **在 TinyGrad 中使用 bfloat16 和 float32 的混合精度**：一位成员询问是否可以通过在 matmul 操作期间进行类型转换（casting）来模拟混合精度。另一位成员确认这是可行的，并且可能更快（*“特别是如果它符合 tensor core 的 dtype 模式”*）。

- **使用计算出的索引 X 对 tensor 进行索引**：一位成员寻求关于访问从 kernel 操作中计算出的 tensor 索引的高效方法的建议，并提到涉及多个 kernel 的潜在效率低下问题。他们还引用了 PR#3707 中讨论的布尔索引模式。

- **在 TinyGrad 中使用 UOp graph**：一位成员分享了使用 `MetalDevice` 和 `MetalCompiler` 编译并运行 UOp graph 的代码，但需要关于执行已编译 kernel 的指导。另一位成员建议查看 `compiledRunner` 以获取更多信息。
  

---



### **Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1250871666096083005)** (6 messages): 

- **清醒的 AI 主导 Databricks 峰会：** 来自 [dbreunig](https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html) 的博客文章强调，大多数 AI 工作都是实用且接地气的，与 OpenAI 和 Google 等公司的炒作相反。他指出最佳实践才刚刚稳定，将现代 LLM 工作比作 2019 年的数据科学。
- **GPU 与 Spark 容量之争：** 当前的 AI 开发在争夺 GPU 核心和 VRAM 交换，这与之前数据工程中对 Spark 容量和 RAM 交换的争斗形成对比。这突显了该领域技术限制的演变过程。
- **高 LLM 输入输出比：** 在 Databricks，客户的 LLM 输入与输出比例为 9:1，这表明输入 token 的定价比输出 token 的定价更重要。这一比例强调了有效运营 LLM 的经济层面。
- **登上首页：** dbreunig 对 Databricks 峰会的见解登上了 Hacker News 首页，表明人们对当前 AI 领域中“清醒 AI”和实际应用挑战等主题有着广泛兴趣。

**提到的链接**：<a href="https://www.dbreunig.com/2024/06/12/sober-ai-is-the-norm.html">清醒的 AI 是常态</a>：尽管听到了很多关于人类替代品和 AGI 的炒作，但清醒的 AI 才是默认的常态。数据科学家和工程师正通过实际应用悄然改变商业智能...

  

---



### **DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1250780693030637588)** (3 messages): 

- **Aleph Alpha 和 Silo AI 合作推动欧洲 AI**：Aleph Alpha 和 Silo AI 宣布建立战略合作伙伴关系，以推进欧洲各地的开源 AI 和企业级解决方案。他们的合作旨在利用 Aleph Alpha 的技术栈和 Silo AI 拥有 300 多人 AI 团队的专业知识，增强工业企业的 AI 部署。[Aleph Alpha 和 Silo AI 合作伙伴关系](https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/)

**提到的链接**：<a href="https://aleph-alpha.com/aleph-alpha-and-silo-ai-enter-a-strategic-partnership-to-advance-open-source-ai-and-enterprise-grade-solutions-in-europe/">Aleph Alpha 和 Silo AI 建立战略合作伙伴关系，以推进欧洲开源 AI 和企业级解决方案 - ALEPH ALPHA - 为企业和政府提供的 AI</a>：为了促进生成式 AI 的采用并充分利用其在欧洲工业企业中的潜力，欧洲最大的 AI 实验室 Silo AI 和欧洲 AI 冠军 Aleph Alpha 宣布了一项长期合作...

  

---

### **Torchtune ▷ #[general](https://discord.com/channels/1216353675241590815/1216353675744641096/1250813272375038033)** (1 条消息): 

- **关于 Torchtune 用户模型推理服务（serving）的调查**：一位用户发起了一项社区投票，调查大家如何提供其微调（finetuned）模型的推理服务。他们对任何帮助表示感谢，并说道：“*感谢您的帮助！*”。
  

---


### **Torchtune ▷ #[dev](https://discord.com/channels/1216353675241590815/1236040539409879170/1250529900495110267)** (1 条消息): 

- **提议的 Tokenizer 重构获得了热烈的建议**：一位开发者分享了一个关于[大规模 Tokenizer 重构](https://github.com/pytorch/torchtune/pull/1082)的 RFC，详细说明了此次变更的重要原因。他们强调了诸如更容易添加功能、更好的组合性以及缩短新模型或更新模型 Tokenizer 的上手时间等优点。

**提到的链接**：<a href="https://github.com/pytorch/torchtune/pull/1082.">Build software better, together</a>：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/1250792339916718161)** (1 条消息): 

- **加入 Infer: Summer '24 获取 AI 洞察**：Qwak 将于 6 月 26 日举办一场**免费虚拟会议**，面向 AI 和 ML 爱好者。该活动包括与专家的实时互动以及来自该领域领导者的实用见解。
- **深刻的演讲与实用知识**：亮点包括关于构建**推荐系统（recommender systems）**和探索 **AI 在体育领域**的应用讨论。您将学习实施预测解决方案和稳健系统的技术，重点关注架构和用户参与度。
- **尊贵的演讲者阵容**：来自 **Lightricks、LSports 和 Lili Banking** 的专家将分享他们的实战经验和知识。著名演讲者包括 Qwak 的解决方案架构师 **Hudson Buzby**，以及 ArtifexAI 的数据科学家兼 AI 顾问 **Russ Wilcox**。
- **立即注册免费参加**：不要错过这个扩展您的 AI 敏锐度并与顶级行业专业人士建立联系的机会。[点击此处免费注册](https://tinyurl.com/j8z6s8ka)并参加 2024 年 6 月 26 日的活动。

**提到的链接**：<a href="https://tinyurl.com/j8z6s8ka">Infer Summer ‘24 by Qwak | The Engineering Behind AI and ML</a>：Qwak 举办的 Infer Summer ‘24 邀请 AI 领导者分享全球领先公司如何在生产中使用 ML 和 AI。请于 2024 年 6 月 26 日上午 11:00 (EDT) 参加直播。

  

---



### **Mozilla AI ▷ #[announcements](https://discord.com/channels/1089876418936180786/1089876419926032396/1250821788481359932)** (1 条消息): 

- **即将举行的 AI 软件开发系统活动**：发布了一项关于即将举行的活动“**旨在实现开发者被增强而非被自动化的未来的 AI 软件开发系统**”的公告。详情和 RSVP 链接在[这里](https://discord.com/events/1089876418936180786/1242653066512175157)。

- **Machine Learning Paper Picks #2 发布**：最新一期的 **Machine Learning Paper Picks** 已发布。点击[这里](https://discord.com/channels/1089876418936180786/1250679657263534152/1250679657263534152)查看精选论文。

- **新的 CambAI 团队活动**：发布了一个与 **CambAI 团队**合作的新活动。更多信息和 RSVP 链接可以在[这里](https://discord.com/events/1089876418936180786/1250168740667195455)找到。

- **AMA 参与者角色分配**：提醒 AMA 参与者通过 <id:customize> 链接领取 **0din 角色**。该角色是接收 T 恤发放通知所必需的。

- **精选频道的成员请求标签**：为用户添加了一个新的 `member-requested` 标签，以便为精选[频道](https://discord.com/channels/1089876418936180786/1231977676458168381)做出贡献。特别鸣谢发起此项工作的成员。

- **资助与支持的 Builders 计划**：鼓励感兴趣的成员查看 **Builders 计划**，为他们的项目寻求资金和支持。[详情请见](https://discord.com/channels/1089876418936180786/1089876419926032396/1247228938346958859)。
  

---

### **YAIG (a16z Infra) ▷ #[tech-discussion](https://discord.com/channels/958905134119784489/960713746702020608/1250874031679475743)** (1 条消息): 

- **关于 GitHub Codespaces 使用情况的调查**：一名成员发起了一项调查，询问团队是否使用 GitHub Codespaces，提示回复 ✅ 表示是，或 ❌ 表示否。此消息可能旨在评估该功能在团队中的采用和利用情况。
  

---



---



---



{% else %}


> 完整的各频道详细分析已针对电子邮件进行了截断。
> 
> 如果您想查看完整的详细分析，请访问此电子邮件的网页版：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前致谢！

{% endif %}