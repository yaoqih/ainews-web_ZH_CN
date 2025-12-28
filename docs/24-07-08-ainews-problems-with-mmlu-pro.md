---
companies:
- huggingface
- meta-ai-fair
- salesforce
- runway
- nomic-ai
- pineapple
- argil-ai
date: '2024-07-09T00:20:51.624419Z'
description: '以下是该文本的中文翻译：


  **MMLU-Pro** 作为 HuggingFace **Open LLM Leaderboard V2** 上 MMLU 的继任者正备受关注。尽管社区对其评估差异和提示词敏感性（这会影响模型性能）表示担忧——例如
  **Llama-3-8b-q8** 仅通过简单的提示词微调就实现了 **10 分的提升**。**Meta 的 MobileLLM** 研究探索了如何通过共享权重和更深层的架构，在智能手机上运行参数量低于十亿（sub-billion）的大语言模型。**Salesforce
  的 APIGen** 推出了一种针对函数调用（function-calling）任务的自动化数据集生成系统，其表现优于许多参数量更大的模型。**Runway Gen-3
  Alpha** 为付费用户发布了一款 AI 视频生成器，能够创作长达 10 秒的高逼真视频剪辑。**Nomic AI 的 GPT4All 3.0** 提供了一款开源桌面应用程序，支持数千种本地模型。具备多模态能力且能以实惠价格接入
  ChatGPT、Claude、Llama 和 Gemini 等多种大语言模型的 AI 助手正不断涌现。**Meta 3D Gen** 推动了“文本到 3D 资产”生成技术的发展，而
  Argil AI 则支持通过文本对话生成深度伪造（deepfake）视频。关于 Transformer “顿悟”（grokking）与推理的研究，凸显了在增强鲁棒推理能力方面取得的进展。'
id: 4c58d595-fd90-448e-87dc-e3d95a3f7c54
models:
- mmlu-pro
- llama-3-8b-q8
- gpt4all-3.0
- chatgpt
- claude
- llama
- gemini
- mobilellm
- runway-gen-3-alpha
- meta-3d-gen
original_slug: ainews-et-tu-mmlu-pro
people:
- wenhu-chen
- danhendrycks
- clementine
- ylecun
- adcock_brett
- svpino
- rohanpaul_ai
title: MMLU-Pro 存在的问题
topics:
- benchmarking
- prompt-engineering
- model-evaluation
- model-performance
- multimodality
- automated-dataset-generation
- video-generation
- open-source-models
- ai-assistants
- text-to-3d
- deepfake
- transformers
- reasoning
---

<!-- buttondown-editor-mode: plaintext -->**阅读 Benchmark 代码就是你所需要的一切。**

> 2024年7月5日至7月8日的 AI 新闻。
我们为你检查了 7 个子版块、[**384** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **29** 个 Discord 服务端（**462** 个频道，**4661** 条消息）。
预计节省阅读时间（以 200wpm 计算）：**534 分钟**。你现在可以标记 [@smol_ai](https://x.com/smol_ai) 来讨论 AINews！

随着 [MMLU-Pro](https://x.com/_philschmid/status/1791137274337354166) 取代已饱和的 MMLU，人们对此充满了期待。在 [Dan Hendrycks 发布他自己的更新](https://x.com/DanHendrycks/status/1804929811703591345)之前，HuggingFace 已经在 [Open LLM Leaderboard V2](https://huggingface.co/spaces/open-llm-leaderboard/blog) 中将 MMLU-Pro 确立为继任者（更多内容将在即将播出的与 Clementine 的播客中讨论）。它相比 [MMLU 有很多改进](https://x.com/WenhuChen/status/1790597967319007564)... 

 
![image.png](https://assets.buttondown.email/images/6d389db1-b599-49fb-a88b-0209f7f8a29c.png?w=960&fit=max)
 

但是... /r/LocalLlama 的好心人们一直在深入研究并发现了问题，首先是 [数学比重过高](https://www.reddit.com/r/LocalLLaMA/comments/1du52gf/mmlupro_is_a_math_benchmark/)，但今天更具毁灭性的是，MMLU-Pro 团队在评估不同模型时，在采样参数、System Prompt 以及答案提取正则表达式方面存在一些令人震惊的差异：

 
![image.png](https://assets.buttondown.email/images/8aa9d3eb-510c-49cd-a9e1-78a646bb60e4.png?w=960&fit=max)
 

就 MMLU-Pro 团队而言，他们承认了这些差异（包括模型之间的差异，以及已发表论文与代码实际执行之间的差异），但[声称他们的样本影响极小](https://github.com/TIGER-AI-Lab/MMLU-Pro/issues/5#issuecomment-2213291392)，然而社区正确地指出，[对闭源模型的额外关注和定制化使得开源模型处于劣势](https://www.reddit.com/r/LocalLLaMA/comments/1dw8l3j/comment/lbu6efr/)。

经验告诉我们，目前的模型对 Prompt Engineering 仍然高度敏感，**对 System Prompt 的简单调整就让 Llama-3-8b-q8 的性能提升了 10 个点（！！？？！）**。

 
![image.png](https://assets.buttondown.email/images/e244cd9b-9f8b-45e7-a0bb-b99db1cbc59d.png?w=960&fit=max)
 

令人失望但可以修复，维护大型 Benchmark 总是项繁杂的任务，但考虑到我们对它们的重视程度日益提高，人们本希望这些简单的变量来源能得到更好的控制。



---


{% if medium == 'web' %}


**目录**

[TOC] 

{% else %}

**目录**和**频道摘要**已移至此邮件的网页版：[{{ email.subject }}]({{ email_url }})！

{% endif %}


---

# AI Twitter 摘要

> 所有摘要均由 Claude 3 Opus 完成，取 4 次运行中的最佳结果。我们正在使用 Haiku 进行聚类和流程工程（flow engineering）。

**AI 进展**

- **Meta 的 MobileLLM**：[@ylecun](https://twitter.com/ylecun/status/1810035281472491665) 分享了一篇关于在智能手机上运行 10 亿参数以下 LLMs 的论文，使用了**增加深度、共享矩阵以及 Transformer 块之间的权重共享**等技术。
- **Salesforce 的 APIGen**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981480052916275) 重点介绍了关于**为函数调用（function-calling）任务生成 AI 训练最优数据集的自动化系统**的新研究，其表现优于尺寸为其 7 倍的模型。
- **Runway Gen-3 Alpha**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981671606735253) 宣布该 **AI 视频生成器现已向所有付费用户开放**，可根据文本和图像生成逼真的 10 秒片段。
- **Nomic AI GPT4All 3.0**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981693979201932) 分享了新的开源 LLM 桌面应用，支持**数千个在本地私密运行的模型**。

**AI Agent 与助手**

- **具备视觉和听觉的 AI 助手**：[@svpino](https://twitter.com/svpino/status/1809921844297732268) 用 Python 构建了一个**能看能听**的 AI 助手，并附带分步视频教程。
- **Pineapple 的 ChatLLM**：[@svpino](https://twitter.com/svpino/status/1810026351514321031) 发布了一款 AI 助手，每月只需 10 美元即可访问 **ChatGPT, Claude, Llama, Gemini 等模型**。

**AI 艺术与视频**

- **Meta 3D Gen**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981569857114600) 分享了 Meta 的新 AI 系统，可**根据文本提示生成高质量 3D 资产**。
- **Argil AI Deepfake 视频**：[@BrivaelLp](https://twitter.com/BrivaelLp/status/1809898328668209383) 使用 Argil AI 将 **Twitter 线程转换为 Deepfake 视频**。

**AI 研究与技术**

- **Transformers 中的 Grokking 与推理**：[@rohanpaul_ai](https://twitter.com/rohanpaul_ai/status/1809950057086530019) 分享了一篇论文，探讨 Transformers 如何通过**超越过拟合的延长“Grokking”训练来学习稳健的推理**，并在比较任务中取得成功。
- **寻找 RAG 的最佳实践**：[@_philschmid](https://twitter.com/dair_ai/status/1809878384526139782) 总结了一篇通过实验确定**检索增强生成（RAG）系统最佳实践**的论文。
- **基于 Mamba 的语言模型**：[@slashML](https://twitter.com/slashML/status/1809881609316815175) 分享了一项关于在 **3.5T Token 数据上训练的 8B Mamba-2-Hybrid 模型**的实证研究。

**机器人进展**

- **用于远程操作机器人的 Open-TeleVision**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981502702145951) 分享了来自 UCSD/MIT 的**开源系统，允许通过浏览器在数千英里外控制机器人**。
- **BMW 的 Figure-01 自主机器人**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981547551817990) 分享了 **Figure 机器人在 BMW 使用 AI 视觉自主工作**的新片段。
- **Clone Robotics 类人手**：[@adcock_brett](https://twitter.com/adcock_brett/status/1809981779194839193) 重点介绍了一家波兰初创公司，他们使用**液压肌腱肌肉构建类人肌肉骨骼机器人手**。

**AI 文化与社会**

- **对 AI 选举的担忧**：[@ylecun](https://twitter.com/ylecun/status/1810065581174931806) 反驳了关于**法国极右翼被“剥夺胜利”**的说法，指出他们只是没有赢得多数票。
- **性格盆地（Personality Basins）作为心智模型**：[@nearcyan](https://twitter.com/nearcyan/status/1810099024764289026) 分享了一篇关于使用**“性格盆地”概念作为理解人们长期行为的心智模型**的文章。
- **LLM 使用量增加**：[@fchollet](https://twitter.com/fchollet/status/1810025103054479459) 对追随者进行了调查，询问**过去 6 个月与之前相比，他们使用 LLM 助手的频率**。

**迷因与幽默**

- **顶尖少年（Cracked Kids）与伟大**：[@teortaxesTex](https://twitter.com/teortaxesTex/status/1810066303165616326) 开玩笑说，**那些真正伟大的人并不在乎“顶尖”少年们的惨痛教训**。
- **努力让 AI 运行的开发者**：[@jxnlco](https://twitter.com/jxnlco/status/1809975279802003562) 分享了一个关于**开发者努力让 AI 在生产环境中运行的艰辛**的迷因。
- **AI 狂热者与数字陪伴**：[@bindureddy](https://twitter.com/bindureddy/status/1810042560271794456) 开玩笑说 **“AI 狂热者”寻找数字陪伴和角色扮演**。

---

# AI Reddit 综述

> 涵盖 r/LocalLlama, r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence, /r/LLMDevs, /r/Singularity。评论抓取功能现已上线，但仍有很大改进空间！

**技术进展**

- **AI 模型训练成本迅速增加**：在 /r/singularity 中，Anthropic 的 CEO 表示，[耗资 10 亿美元训练的 AI 模型正在进行中，1000 亿美元的模型也即将到来](https://www.reddit.com/r/singularity/comments/1dy294l/ai_models_that_cost_1_billion_to_train_are/)，而目前最大的模型训练成本“仅”为 1 亿美元。这表明了 AI Scaling 的指数级速度。
- **小鼠寿命延长突破**：在 /r/singularity 中，[Altos Labs 利用山中因子（Yamanaka factor）重编程将小鼠寿命延长了 25% 并改善了健康寿命](https://www.reddit.com/r/singularity/comments/13uoq9o/altos_labs_extends_lifespan_of_mice_by_25_and/)，这是领先的 AI 和生物技术公司在抗衰老研究领域取得的重大成就。
- **DeepMind AI 从视频生成音频**：在 /r/singularity 中，[DeepMind 的新 AI 通过学习从视频生成音频，发现了“像素的声音”](https://www.reddit.com/r/singularity/comments/13v1lz1/deepminds_new_ai_found_the_sound_of_pixels_by/)，展示了将视觉与相关声音联系起来的高级多模态 AI 能力。

**模型发布与基准测试**

- **Llama 3 微调模型在故事创作方面表现不佳**：在 /r/LocalLLaMA 中，一位用户发现 [与 Mixtral 和 Llama 2 微调模型相比，Llama 3 微调模型在故事创作方面表现糟糕](https://www.reddit.com/r/LocalLLaMA/comments/13v0yrm/llama_3_finetunes_are_terrible_for_story_writing/)，因为 Llama 3 模型在长篇故事生成中容易偏离轨道，且不能很好地遵循 Prompt。
- **开源 InternLM2.5-7B-Chat 模型展现出强大能力**：在 /r/ProgrammerHumor 中，[开源大语言模型 InternLM2.5-7B-Chat 展示了无与伦比的推理、长上下文处理和增强的工具使用能力](https://www.reddit.com/r/ProgrammerHumor/comments/13v0yrk/internlm257bchat_an_opensource_large_language/)，推动了开源 AI 能力的边界。
- **用户对 28 个 AI 模型进行了各项任务的基准测试**：在 /r/singularity 中，[一名用户对 28 个不同的 AI 模型运行了小规模个人基准测试](https://www.reddit.com/r/singularity/comments/13v0yrj/i_ran_smallscale_personal_benchmarks_on_28/)，测试了推理、STEM、实用性、编程和审查制度。GPT-4 和 Claude 变体位居榜首，而 Llama 和 GPT-J 等开源模型紧随其后，并提供了详细的评分数据。
- **默认 MMLU-Pro 提示词不适合 Llama 3 的基准测试**：在 /r/LocalLLaMA 中，研究发现 [默认的 MMLU-Pro 系统提示词对于 Llama 3 模型的基准测试非常糟糕](https://www.reddit.com/r/LocalLLaMA/comments/1dxpns0/default_mmlupro_system_prompt_is_really_bad/)，导致结果不一致，而修改提示词可以显著提高模型在该基准测试中的表现。

**讨论与观点**

- **对 LMSYS AI 排行榜有效性的担忧**：在 /r/singularity 中，有人认为 [由于存在操纵风险和结果不一致，流行的 AI 排行榜 LMSYS 本质上存在缺陷，不应再作为基准测试使用](https://www.reddit.com/r/singularity/comments/1dxcyav/lmsys_is_inherently_flawed_and_should_not_be_used/)，强调了对替代评估方法的需求。
- **构建 AI 应用的经验教训**：在 /r/ProgrammerHumor 中，[一名用户询问了构建 AI 应用时学到的最大教训](https://www.reddit.com/r/ProgrammerHumor/comments/13v0yri/what_are_the_biggest_lessons_youve_learned_when/)。回复强调了拥有可靠的评估数据集、从托管模型开始，以及避免在无休止地调整框架或数据集上浪费时间。
- **在超级计算机上训练更大模型的潜力**：在 /r/singularity 中，[有人提出了一个问题：现代超级计算机是否能够训练比当前模型大得多的模型](https://www.reddit.com/r/singularity/comments/13v0yre/are_modern_supercomputers_capable_of_training/)。计算能力似乎已经具备，但尚不清楚是否正在秘密进行此类大规模训练。

**迷因与幽默**

- **幽默迷因图**：在 /r/singularity 中，[一张迷因图以幽默的口吻问道“Where Are Ü Now?”](https://www.reddit.com/r/singularity/comments/13v0yrl/where_are_%C3%BC_now/)，未提供进一步背景。

---

# AI Discord 摘要

> 摘要之摘要的摘要

**1. 模型架构与训练的进展**

- **Hermes 2 在基准测试中的卓越表现**：**Hermes 2** 模型及其改进版本 **Hermes 2.5** 在基准测试中展现了显著的性能提升，超越了该领域的许多其他模型。
   - 社区讨论强调，虽然 Hermes 2 表现出色，但像 **Mistral** 这样的其他模型在没有进一步预训练的情况下，很难将 context 扩展到 8k 以上。这引发了关于模型扩展（scaling）和通过合并策略（merging tactics）提升性能潜力的辩论。
- **BitNet 的二进制突破**：[BitNet](https://arxiv.org/abs/2310.11453) 引入了一种可扩展的 1-bit 权重 Transformer 架构，在显著降低内存占用和能耗的同时，实现了极具竞争力的性能。
   - 这种 1-bit 模型的创新为在资源受限的环境中部署 LLM 开启了可能性，有望使获取先进 AI 能力变得更加普及。
- **T-FREE 的 Tokenizer 变革**：研究人员推出了 [T-FREE](https://arxiv.org/abs/2406.19223)，这是一种通过字符三元组（character triplets）的激活模式对单词进行 embedding 的 tokenizer，在保持竞争力的性能的同时，将 embedding 层的大小减少了 85% 以上。
   - 这种新颖的 tokenization 方法可能会带来更高效的模型架构，从而减少训练和部署 LLM 所需的计算资源。

**2. AI 效率与部署方面的创新**

- **QuaRot 的量化探索**：[最近的研究](https://arxiv.org/abs/2404.00456) 展示了 QuaRot 在 LLM 的 4-bit 量化（quantization）方面的有效性，在显著降低内存和计算成本的情况下，实现了接近全精度的性能。
   - 这种量化技术的进步可以极大地提高 LLM 部署的效率，使得在更普通的硬件配置上运行强大的模型成为可能。
- **MInference 为长文本 LLM 提速**：微软的 [MInference 项目](https://github.com/microsoft/MInference) 旨在加速长文本（Long-context）LLM 的推理，在 **A100** GPU 上将 **延迟（latency）降低** 了多达 10 倍。
   - MInference 采用了近似和动态稀疏计算的新技术，在准确性与 **性能效率** 之间取得了平衡。该工具可以显著提高 LLM 在需要快速响应场景中的实际应用价值。
- **Cloudflare 的 AI 爬虫防护**：**Cloudflare** 推出了一项功能，允许网站阻止 AI 爬虫机器人，这可能会影响用于 AI 训练的数据收集，并引起了 AI 社区的关注。
   - 虽然有些人担心这会对 AI 发展产生影响，但另一些人认为只有积极尝试阻止 AI 的网站才会使用此功能。这一进展突显了 AI 时代数据可访问性与隐私之间日益增长的紧张关系。


---

# 第一部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stability AI 的授权迷宫**：社区正在积极讨论新的 [Stability AI 模型授权](https://stability.ai/news/license-update) 条款，重点关注对年收入超过 100 万美元的企业的影响。
   - 关于 SD3 模型在商业应用中的使用仍存在**担忧**，特别是对小型企业的影响。
- **像素完美：放大（Upscaling）之旅**：分享了一个**放大工作流**，结合了 Photoshop、[SUPIR](https://old.reddit.com/r/StableDiffusion/comments/1b50sp0/ccsr_vs_supir_upscale_comparison_portrait/) 等工具，在平衡细节和一致性的同时生成高分辨率图像。
   - 这种多步骤策略旨在解决**分块（tiling）问题**，这是图像放大中的常见瓶颈。
- **模型质量迷局**：部分成员对 SD3 模型的质量表示**失望**，并将其与前代模型进行对比，推测仓促发布可能带来的后果。
   - 未来的 8B 版本备受期待，同时还讨论了伦理考量以及 NSA 等机构感知到的影响。
- **Text2img 故障排除：VRAM 紧缺**：用户经验表明，将 ControlNet 与 text2img 结合使用时会出现减速，这与 VRAM 限制有关，需要进行**内存管理**。
   - 建议使用优化 Windows 页面文件设置和卸载（offloading）等有效缓解技术来应对减速。
- **培养创意提示词（Prompts）**：公会一直在交流关于如何更好利用提示词和外部集成（如 [github.com/AUTOMATIC1111](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings)）的见解，以增强图像生成效果。
   - 建议包括在提示词中战略性地使用语言，以及应用多种工具以获得最佳图像结果。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

- **推理耐力不尽如人意**：有报告称推理端点（inference endpoints）的初始化时间过长，表明在 GPU 可用性或特定配置设置方面存在挑战；一位成员建议评估 eu-west-1 区域的 [AWS Nvidia A10G](https://aws.amazon.com/ec2/instance-types/a10g/) 作为补救措施。
   - 关于效率的话题浮出水面，一位成员担心 GPTs Agent 在初始训练后无法学习，引发了关于当前 AI 模型适应性极限的讨论。
- **词汇表化解 AI 术语困惑**：**LLM/GenAI Glossary** 作为一份旨在让 AI 术语易于理解的综合指南发布。Prashant Dixit 分享了社区创建的[词汇表链接](https://github.com/freetoolsarebest/llm-glossary)，该表定期更新以辅助学习和贡献。
   - 该倡议旨在简化 AI 社区内的技术交流，强调了在这个充满复杂术语的领域中清晰表达的重要性。
- **AI 创作者集结 HuggingFace Space**：成员宣布的 **ZeroGPU HuggingFace Space** 提供了多种 Stable Diffusion 模型对比，包括 **SD3 Medium**、**SD2.1** 和 **SDXL**，[可供实验](https://huggingface.co/spaces/Nick088/stable-diffusion-arena)。
   - 本着 DIY 精神，**qdurllm** 作为一个结合了 **Qdrant**、**URL 抓取**和 **Large Language Models** 的工具出现，用于本地搜索和聊天，其开源格式促进了在 [GitHub](https://github.com/AstraBert/qdurllm) 上的协作探索。
- **目标检测的视觉指标**：Torchmetrics 在改进目标检测指标方面得到了认可，其应用在 [Trainer API 和 Accelerate 示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection)中得到了强调。
   - RT-DETR 模型作为一种实时目标检测产品引起了关注，它融合了卷积的效率与以 Attention 为中心的 Transformer，如这篇 [推文](https://x.com/mervenoyann/status/1807790959884665029) 所示，采用 Apache 2.0 授权。
- **sd-vae 重建中的伪影之谜**：成员们开始讨论 **sd-vae** 中出现蓝色和白色像素伪影（artifacting）是否正常，以及这对于重建结果意味着什么。
   - 参数调整的探索成为社区排除此类现象故障的共同策略，强调了完善 sd-vae 模型的协作方法。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **Perplexity 受到审视**：用户发现 Perplexity 经常返回**过时信息**，且在上下文保留方面表现不佳，在后续对话的流畅度上落后于 **GPT-4o** 和 **Claude 3.5**。
   - Pro 版本相比免费服务没有显著提升，引发了关于替代服务（如 **Merlin.ai** 和 **ChatLLM**）的讨论。
- **揭秘隐藏功能**：Perplexity 的图像生成能力让一些人感到惊讶，Pro 用户指导他人通过*自定义提示词 (custom prompt)* 选项来最大化利用该功能。
   - 技术故障讨论包括文本重叠和上下文丢失，社区倾向于使用**系统提示词 (system prompts)** 作为临时补救措施。
- **社区知识中的小众干货**：一份[地下生存指南](https://www.perplexity.ai/page/minecraft-underground-survival-hj7PsuozQ32xoJKudQqm8g)深入探讨了 Minecraft 生存方法，引发了策略交流。
   - 一位用户关于平均成本的**研究见解令人侧目**，而另一位用户则在设置新 Google 账号的挫折中寻求共鸣。
- **API 的忧与喜**：更新后的 Perplexity **API** 在处理多部分查询方面表现出潜力，但用户对延迟的 Beta 访问和漫长的处理时间感到愈发沮丧。
   - **API 与搜索页面结果**之间的关系模糊不清，令用户感到困惑，一些人觉得对多步搜索 API 的功能一无所知。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

- **MacBook M3 处理模型的能力受到赞赏**：配备 **128GB RAM** 的新款 **M3 MacBook Pro** 因其管理 **WizardLM-2-8x22B** 等大型模型的能力而受到积极关注，这使其区别于有内存限制的旧版本。
   - 尽管 M2 MacBook 无法加载 **WizardLM-2-8x22B**，但 M3 的实力巩固了 **Apple** 在为大型模型推理工作负载提供**强大解决方案**方面的地位。
- **Gemma 2 模型等待 Bug 修复**：社区讨论集中在 **Gemma 2 模型**推理缓慢和计算错误的问题上，用户期待未来的更新能解决这些问题。
   - 讨论串指出了 [Gemma 模型架构 Bug](https://github.com/ggerganov/llama.cpp/pull/8348) 的引用，表明即将到来的改进可能会解决目前的限制。
- **讨论模型量化进展**：用户交流了高级量化方法的见解，辩论了**模型性能**与**输出质量**之间的最佳平衡。
   - 分享了[量化模型](https://huggingface.co/Joseph717171/Models/tree/main)的链接，引发了关于利用 F32 和 F16 格式以获得**增强结果**的对话。
- **LM Studio 的 x64bit 安装程序疑问得到澄清**：在 LM Studio 的讨论频道中，一位用户对缺少 64 位安装程序的困惑得到了解答，解释称现有的 x86 标识也包含 64 位兼容性。
   - 这种透明度消除了误解，并突显了 **LM Studio 细致的社区互动**。
- **Fedora 40 Kinoite 与 7900XTX 的协同效应表现稳健**：部署更新后，**LM Studio** 内的**生成速度**显著提升，这证明了 **Fedora 40 Kinoite** 与 **7900XTX GPU** 配置之间的协同效应。
   - 这一进展反映了优化方面的持续进步，强调了**速度提升**是当前 AI 工具的一个重点。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **Hermes 升温，Mistral 未达预期**：关于 **Hermes 2** 与 **Hermes 2.5** 性能的辩论升温，对比了增强的基准测试结果，以及 **Mistral** 在没有进一步预训练的情况下难以扩展到 8k 以上的问题。
   - 讨论深入探讨了通过**合并策略 (merging tactics)** 改进 AI 模型的潜力；与此同时，Cloudflare 最近的功能因其拦截 AI 数据抓取机器人的能力而引起了褒贬不一的反应。
- **自定义 GPT 努力应对 Zapier**：社区成员分享了使用**自定义 GPT** 的经验，讨论了尽管遇到可靠性问题，但仍通过集成 **Zapier** 来实现任务自动化。
   - **GPT-4o** 更快的响应时间引发了关于其与 **GPT-4** 相比在质量权衡上的争论，而重复的验证要求则让用户感到沮丧。
- **内容创作与受众参与**：成员们讨论了内容创作者生成引人入胜内容的策略，增强了对特定平台建议、内容日历结构以及决定成功的关键指标的兴趣。
   - AI 工程师强调了提示词 (prompts) 在吸引人的内容创作和客户获取中的重要作用，聚焦于成员们对当前趋势创新用法的想法。

## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **Qwen 隐藏的才华被揭示**：社区成员对 **Qwen Team** 的贡献表示赞赏，强调尽管该团队创建了优秀的资源（如新的 [训练视频](https://link.to.video)），但其努力仍被低估。
   - 关于 **Qwen** 的讨论表明，人们对提供实用 AI 工具和资源的团队越来越尊重。
- **GPU 大决战：AMD vs NVIDIA**：一场关于 **AMD GPU** 与 **NVIDIA** 在 LLM 训练效率方面的技术辩论展开，指出 NVIDIA 由于卓越的软件生态系统和能效而占据主导地位。
   - 尽管 AMD 有所进步，但社区共识倾向于将 NVIDIA 作为 LLM 任务的务实选择，因为其库支持更完善。有人指出：“大多数库不支持 AMD，因此你在使用上会受到很大限制。”
- **Phi-3 使用 Alpaca 训练时的故障**：AI 工程师交流了在利用 Alpaca 数据集进行 Phi-3 训练时遇到的错误解决方案，指出所使用的 `xformers` 版本缺乏 CUDA 支持，并建议进行更新。
   - 对比了 **Llama-3** 与 **Phi 3.5 mini** 的推理速度，并讨论了提高效率的建议，例如参考 Tensorrt-llm 以获得最先进的 GPU 推理速度。
- **Kaggle 的限制激发创新**：社区讨论围绕克服 **Kaggle** 平台的磁盘空间限制展开，该限制在超过 **100GB** 后导致会话崩溃，但在崩溃前已利用 [Weights & Biases](https://wandb.ai) 保存了关键数据。
   - 这一事件突显了 AI 工程师即使在面临资源有限的情况下也在不断创新，同时也说明了在数据密集型任务中可靠 Checkpoint 的重要性。
- **赋能 AI 领域的求职者**：AI 社区成员提议创建一个专门的职位频道，以简化求职和招聘流程，这反映了行业动态增长以及对职业导向服务的需求。
   - 这一倡议展示了在不断增长的 AI 领域中，将社区努力组织并引导至职业发展的积极尝试。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **利用 LLM API 封装复杂性**：利用 **LLM-style API** 重构代码结构可以简化复杂任务；一位用户强调了编码员在系统集成中的关键作用。
   - 通过 Zeroshot LLM 提示词对 API 进行创意组合，将繁重任务转化为仅需极少努力的任务，有望大幅节省时间。
- **探索政府对 AI 的审查**：**英国政府的 Inspects AI 框架** 针对大语言模型，引发了对其潜在探索和影响的好奇。
   - 该框架已在 [GitHub](https://github.com/UKGovernmentBEIS/inspect_ai) 上开源，其在公共部门的地位凸显了审查和监管 AI 技术日益增长的趋势。
- **播客节目席卷 Hacker News**：一位用户在 **Hacker News** 上分享了一集播客（[现已登上 HN！](https://news.ycombinator.com/newest)），旨在吸引关注并提高参与度。
   - 支持性的社区成员通过点赞提高了可见度，反映了 Hacker News 上活跃且参与度高的在线讨论。
- **Fortnite 重塑趣味性**：Fortnite 旨在通过取消联动来重新吸引玩家，这源于一篇讨论游戏动态的 [Polygon 报道](https://www.polygon.com/gaming/24185789/fortnite-reload-new-game-mode)。
   - 社区通过点赞做出了 **即时反应**，PaulHoule 等用户的认可为宣传火上浇油。
- **融合 AI 思想**：随着对 **模型融合策略 (model merging strategies)** 的深入探讨吸引了爱好者，AI Engineer World Fair 的热度达到顶峰，并得到了 [GitHub 上的 mergekit](https://github.com/arcee-ai/mergekit) 等工具的支持。
   - 关于自动确定融合策略的暗示引发了辩论，尽管其智力稳健性被标记为 **存疑**。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **CUDA 证书之争**：关于招聘时 CUDA 认证与公开 GitHub CUDA 项目价值的辩论引发热议，**社区共识**倾向于公开仓库这一切实的证据。
   - `公开的、经过验证的工作总是比证书更有价值`是提出的一个核心观点，强调了**可演示技能的价值**而非证书。
- **编译前行之路**：[Lightning AI](https://boards.greenhouse.io/lightningai/jobs/6045025003) 正在寻找**编译器爱好者**，并提供与 **Luca Antiga** 并肩工作的机会。
   - [Thunder 项目](https://github.com/Lightning-AI/lightning-thunder) 的源码到源码（source-to-source）编译器旨在将 **PyTorch 模型性能提升高达 40%**，有望改变优化基准。
- **PyTorch Profiler 性能洞察**：**torch.compile 手册**被推崇为优化的“**缺失环节**”，并分享了一份阐述其作用和优势的指南。
   - 另一位成员建议使用 `torch.utils.flop_counter.FlopCounterMode` 作为 `with_flops` 的稳健替代方案，理由是其持续的维护和开发。
- **稀疏性的量子化**：CUDA 探索转向了 **2:4 稀疏模式**，并讨论了用于优化稀疏矩阵乘法 (SpMM) 的 **cusparseLT** 和 **CUTLASS** 库的对比。
   - 辩论围绕潜在的性能差异展开，普遍观点倾向于使用 **cusparseLT**，因其**优化**程度和维护情况更佳。
- **LLM 课程规划**：**LLM101n** 的构思，这是一个拟议的课程，旨在引导用户从 **micrograd** 和 **minBPE** 的基础知识走向 **FP8 精度**和**多模态训练**等更复杂的领域。
   - 讨论强调了分层学习方法，在晋升到**最先进的模型实践 (state-of-the-art model practices)** 之前先夯实基础。



---



## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **批判伙伴提升 AI 奖励模型**：探索来自 **LLM** 的合成批判的效用，Daniella_yz 的预印本揭示了在 Cohere 实习期间改进偏好学习的潜力，详见[研究报告](https://arxiv.org/abs/2405.20850)。
   - 研究表明，**CriticGPT** 不仅能辅助人类评估，还能在活跃项目中直接增强奖励模型。
- **测试时训练 (Test-Time-Training) 层打破 RNN 限制**：Karan Dalal 介绍了 **TTT 层**，这是一种在[预印本](https://arxiv.org/abs/2407.04620)中展示的新架构，用 ML 模型取代了 RNN 的隐藏状态。
   - 这种创新带来了**线性复杂度架构**，让 LLM 能够在海量 Token 集合上进行训练，TTT-Linear 和 TTT-MLP 的表现优于顶尖的 Transformer。
- **与 Dataline 进行数据对话**：[RamiAwar 开发的 Dataline](https://github.com/RamiAwar/dataline) 正式发布，该平台允许用户通过 AI 界面查询 CSV、MySQL 等多种数据库。
   - 一项名为《**LLM 的几何理解**》(The Geometrical Understanding of LLMs) 的新研究调查了 LLM 的推理能力及其自注意力图密度；更多内容请参阅[论文](https://arxiv.org/abs/2407.02678)。
- **GPT-4 基准测试热潮**：用户圈内的一个显著观察是 GPT-4 在较高温度设置下在基准测试中表现更好，尽管在本地模型上重现似乎具有挑战性。
   - 随着上下文示例 (in-context examples) 提升模型性能，人们兴奋不已，同时尽管训练复杂度较高，BitNet 架构的内存节省效率仍引发了关注热潮。
- **RAG 与现实：透视幻觉**：一段[新的 YouTube 视频](https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu)聚焦于 LegalTech 工具的可靠性，揭示了 RAG 模型产生幻觉的频率。
   - 此外，为了引用的一致性，提议使用类似维基百科的 `ref` 标签，并且 [AymericRoucher 的 RAG 教程](https://x.com/mervenoyann/status/1810291157961781671)因优化效率而受到赞誉。



---

## [Modular (Mojo 🔥)](https://discord.com/channels/1087530497313357884) Discord

- **WSL 飞跃 - 玩转 Windows 上的 Mojo**: 为安装 **Mojo** 而升级 WSL 在旧版 Windows 10 系统上遇到了小故障；[Microsoft 的 WSL 指南](https://learn.microsoft.com/en-us/windows/wsl/install)在导航升级路径时被证明非常有价值。
   - **Python 的依赖烦恼**引发了讨论，虚拟环境是首选解决方案；[GitHub 讨论帖](https://github.com/modularml/mojo/discussions/1401)也开启了关于 Mojo 简化这些问题的潜力讨论。
- **舍入大乱斗 - Mojo 数学混乱**: Mojo 中的舍入函数 bug 引起了集体抱怨；在社区对**舍入特性**的深入探讨中，强调了与 **SIMD** 的不一致性。
   - 在 int-float 的讨论中，**64 位难题**成为了核心，Mojo 对 `Int64` 和 `Float64` 的分类导致了跨操作的非预期行为。
- **栈上加栈 - 高超的 Matmul 操作**: 成员们对 Max 在 matmul 中使用 ***stack allocation***（栈分配）以提升 Mojo 性能感到惊叹，并指出缓存优化是关键的增强因素。
   - **Autotuning**（自动调优）作为一种简化 *simdwidth* 调整和块大小的理想方案浮出水面，但其实施现状仍处于反思性讨论阶段。
- **Libc 之恋 - 将旧代码链接到 Mojo**: 社区就将 **libc 函数**引入 Mojo 达成了共识；lightbug_http 在 [GitHub](https://github.com/saviorand/lightbug_http/blob/main/external/libc.mojo) 上展示了**自由链接**的实际应用。
   - 关于交叉编译能力的查询以 Mojo 目前尚不支持而告终，促使成员们提出未来可能包含的功能。
- **元组探戈 - 释放 Mojo 潜力**: Mojo 缺乏用于别名的 *tuple unpacking*（元组解包）引发了语法驱动的推测，社区成员渴望一种概念上更清晰的结构。
   - **Nightly 编译器更新**让 Mojo 玩家们紧跟代码节奏，版本 `2024.7.705` 引入了新的模块和变更。

---

## [Cohere](https://discord.com/channels/954421988141711382) Discord

- **AI-Plans 平台揭晓对齐策略**: 围绕 **AI-Plans** 展开了讨论，这是一个旨在促进对齐策略同行评审的平台，主要关注**红队测试对齐计划**。
   - 细节较少，因为用户目前尚未提供有关该项目的进一步见解或直接链接。
- **Rhea 亮眼的“保存到项目”功能点亮 HTML 应用**: Rhea 集成了一项新的“保存到项目”功能，使用户能够直接从其 **dashboards**（仪表板）存储交互式 HTML 应用程序，详见 [Rhea 平台](https://rhea.run)。
   - 这一新增功能促进了更流畅的工作流，有望激发增强的用户参与度和内容管理。
- **Rhea 注册因大小写敏感问题受阻**: Rhea 的注册过程出现了一个小问题，用户电子邮件必须以小写形式输入才能通过邮件验证，这暗示了在 **user-experience**（用户体验）考虑上可能存在的疏忽。
   - 这一发现强调了在用户界面设计中进行严格测试和反馈机制的重要性，特别是针对大小写敏感的处理。
- **Cohere 社区纽带与创投的传闻**: Cohere 社区的新面孔分享了他们的热情，兴趣集中在协同使用 **Aya** 等工具进行协作工作流和文档记录。
   - 这些介绍成为了分享经验的跳板，增强了 Cohere 的工具利用率和社区凝聚力。
- **青少年遇见技术：Rhea 开启儿童友好型 AI 编程俱乐部冒险**: 儿童编程俱乐部的成员正在寻求新视野，通过将 Rhea 易于使用的平台集成到他们的 **AI 和 HTML 项目**中，旨在激励下一代 AI 爱好者。
   - 这一举措代表了在 AI 领域培养青少年思想迈出的一步，突显了像 Rhea 这样的教育工具对于不同年龄段和技术背景的适应性。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

- **T-FREE 缩小 Tokenizer 占用空间**：[T-FREE](https://arxiv.org/abs/2406.19223) Tokenizer 的引入彻底改变了 Embedding，层大小减少了 85%，并实现了与传统模型相当的效果。
   - 该 Tokenizer 放弃了预分词（pretokenization），通过字符三元组激活模式（character triplet activation patterns）转换单词，这是迈向模型紧凑化的重要一步。
- **SOLAR 为模型扩展提供新思路**：关于 [SOLAR](https://arxiv.org/abs/2310.07999)（一种模型扩展技术）的讨论非常热烈，主要涉及其效率与从零开始训练模型的对比。
   - 虽然 SOLAR 展示了性能优势，但仍需要与从零训练的模型进行更好的对比才能得出最终结论。
- **BitNet 凭借 1-bit 权重 Transformer 实现飞跃**：[BitNet](https://arxiv.org/abs/2310.11453) 首次推出了 1-bit 权重 Transformer 架构，在性能与资源消耗之间取得了平衡，具有内存和能源友好的特性。
   - 在不大幅牺牲结果的情况下进行权重压缩，使 BitNet 的 Transformer 能够在资源受限的场景中扩大应用范围。
- **QuaRot 证明了 4-bit 量化的强大**：[QuaRot 的研究](https://arxiv.org/abs/2404.00456)表明，4-bit 量化在 LLM 中能保持接近全精度的水平，同时有效降低了内存和处理需求。
   - 在没有严重性能下降的情况下大幅削减计算成本，使 QuaRot 成为推理运行时优化的实际选择。
- **寻找 GPT-Neox 的正确 Docker 部署方式**：关于有效使用 Docker 容器部署 GPT-Neox 的咨询引发了关于 Kubernetes 可能更适合大规模任务管理的推测。
   - 虽然 Docker Compose 一直很方便，但在部署环境中，为了降低复杂性和提高效率，规模化部署更倾向于使用 Kubernetes。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **JPEG XL 夺得桂冠**：**JPEG XL** 现在被认为是领先的图像编解码器，因其在该领域优于其他格式的效率而受到认可。
   - 讨论强调了它相对于传统格式的稳健性，并考虑将其作为未来的标准用法。
- **Kolors 仓库引起关注**：[Kolors GitHub 仓库](https://github.com/Kwai-Kolors/Kolors) 因其重要的论文部分而引起了关注热潮。
   - 成员们对其技术深度表达了兴奋和幽默，预测其将对该领域产生强烈影响。
- **噪声调度引发辩论**：关于增加 100 个时间步长并转向 **v-prediction** 进行噪声调度的有效性是一个热门辩论话题，特别是为了实现零终端 SNR。
   - 在对高分辨率采样场景中测试与训练不匹配的担忧中，**SDXL 的论文**被引用作为指导。
- **Meta 的 VLM 广告面临质疑**：Meta 决定宣传 VLM 而不是发布 **Llama3VLM** 引起了不满，用户对 Meta 对 API 可用性的承诺表示怀疑。
   - 社区对 Meta 优先考虑自家产品而非广泛的 API 访问表示担忧。
- **VALL-E 2 的文本转语音突破**：**VALL-E 2** 为文本转语音系统设定了新基准，其零样本 TTS 能力在自然度和稳健性方面脱颖而出。
   - 尽管它需要显著的计算资源，但其在 LibriSpeech 和 VCTK 数据集上的结果引发了社区对复制工作的期待。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **通过 LangChain 解析 CSV**：用户探讨了在 LangChain 中处理 CSV 文件的方法，讨论了超越以往限制的现代化方法的需求。
   - **LangChain 的 utility functions** 提供了帮助，建议将模型输出转换为 JSON，并使用 `Json RedactionParser` 等工具来增强解析能力。
- **异步配置揭秘**：通过社区协作，揭开了 LangChain 中异步配置的神秘面纱，特别是使用 `astream_events` 时 `ToolNode` 内的 `ensure_config()` 方法。
   - 分享了在 `invoke` 函数中包含 `config` 的关键指导，从而简化了 **async task management**（异步任务管理）。
- **本地 LLM 实验规模扩大**：关于在配备 NVIDIA RTX 4090 GPU 的个人设备上运行 `phi3` 等小型 LLM 模型的讨论非常热烈。
   - 对于管理 70B 参数等巨型模型以及在多 GPU 设置上实现此类壮举的可行性，好奇心激增，预示着 **local LLM innovation**（本地 LLM 创新）的驱动力。
- **LangGraph Cloud 服务引发猜测**：**LangGraph Cloud** 即将到来的暗示引发了关于 LangServe API 部署是否需要第三方提供商的疑问。
   - 社区对新服务产品的期待以及部署范式可能发生的转变议论纷纷。
- **浏览器内视频分析工具引起关注**：**'doesVideoContain'** 是一款用于在浏览器内扫描视频内容的工具，凭借其对 **WebAI** 技术的使用引起了兴趣。
   - 为了推动社区参与，提供了 YouTube 演示和 Codepen 实时示例的直接链接，促进其应用。

---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **RAG 的技能库强化行动**：为了提高效率，一名成员率先将技能库与 **RAG** 集成，增强了指定操作的一致性。
   - 这一进展已与社区分享，激励了对 **RAG** 在各种 AI 应用中潜力的进一步探索。
- **OI 团队警惕守护安全边界**：OI 团队对安全性的承诺在最近的一次视频会议中受到关注，将其巩固为运营完整性的首要任务。
   - 他们的前瞻性措施正在为集体安全协议树立基准。
- **GraphRAG 有效穿梭于数据簇**：一位参与者展示了 **Microsoft** 的 **GraphRAG**，这是一款将数据聚类为社区以优化 **RAG** 用例的高级工具。
   - 实施 GraphRAG 的热情被点燃，同时还参考了来自 [@tedx_ai](https://x.com/tedx_ai/status/1808561861589139690) 的一条富有启发性的推文。
- **7 月 4 日聚会的节日基调**：OI 团队的 **4th of July** 庆祝活动增进了友谊，展示了新的演示，并培养了对未来团队聚会的期待。
   - 团队精神受到鼓舞，希望将这一庆祝活动确立为每月的常规亮点。
- **O1 单元准备 11 月推出**：时间表显示首批 1000 个 O1 单元计划于 11 月交付，反映了对其按时到达的高度期望。
   - 围绕 O1 的对话能力充满好奇，同时社区通过分享解决 Linux 'typer' 模块故障的方案提供了支持。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **支持多种货币的加密货币支付**：社区讨论集中在 **Coinbase Commerce** 处理多种加密货币支付的能力，包括通过 **Polygon** 支付 **USDC** 和 **Matic**。
   - 一位用户确认了使用 **Matic** 的无缝交易，并对其效果表示认可。
- **Perplexity API 表现不佳**：用户指出 **Perplexity API** 的性能与其网页版相比逊色不少，Payload 中缺少关键的参考链接。
   - 规避此问题的建议包括使用 **Phind** 等替代方案，或直接从 **GitHub** 和 **StackOverflow** 抓取数据。
- **预测生成式视频的发展轨迹**：一位成员询问了关于**生成式视频**在未来 **18 个月**内质量、执行速度和成本的预期轨迹。
   - 目前尚未做出明确预测，强调了此类生成媒介尚处于初期阶段。
- **OpenRouter 的定制化 AI 选项**：确认 OpenRouter 允许能够处理大量请求的用户部署自己的**微调模型 (fine-tuned models)**。
   - 这被认为是希望赋予定制化 AI 功能的开发者的福音。
- **DeepInfra vs. Novita：价格战**：OpenRouter 见证了 **DeepInfra** 和 **NovitaAI** 之间的价格竞争，它们在提供 **Llama3** 和 **Mistral** 等模型服务方面争夺领先地位。
   - 一场以 **0.001** 为单位降价的幽默战斗，使得这些模型的定价极具竞争力。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

- **自动驾驶式交易：LlamaIndex 驱动 AI 股票助手**：一个利用 **Llama Index Agent** 的 AI 交易助手在[教程视频](https://t.co/dcLG3orq0s)中展示，可执行多种股票交易任务。
   - 其能力由 [Llama Index 的 RAG 抽象](https://t.co/ocPaeLphyG)驱动，包括预测分析和交易，并展示了实际应用案例。
- **构建 RAG 数据集：更丰富问题的工具**：Giskard AI 的工具包有助于生成强大的 RAG 数据集，其[工具包文章](https://t.co/sewtQcb9b8)中展示了生成多种问题类型的功能。
   - 该工具包超越了典型的自动生成集，为[数据集创建提供了更丰富的工具](https://t.co/rQ7WxplJpF)。
- **微服务，大潜力：大规模敏捷 Agent**：**Llama-agents** 现在为可扩展、高需求的微服务提供了一套设置，详见[这篇见解深刻的文章](https://t.co/y9a3PdfW0M)。
   - 这种“Agent 与工具即服务”的模式增强了可扩展性并简化了微服务交互。
- **分析分析师：LlamaIndex 助力 10K 报告剖析**：得益于 [Llama Index 的功能](https://t.co/rOetN1zeNg)，**多文档财务分析师 Agent** 将每份文档视为一个工具，处理 10K 等财务报告的分析。
   - Pavan Mantha 展示了利用 [Llama Index 的特性](https://t.co/LJhV838EUM)进行此类分析的效率。



---

## [tinygrad (George Hotz)](https://discord.com/channels/1068976834382925865) Discord

- **红色阵营的犹豫：对 Instinct 系列的谨慎？**：一名成员对 **team red** 的 **Instinct 显卡驱动程序** 表示担忧，由于潜在的支持问题，在购买二手 Mi100 时感到犹豫。
   - 对话中提到，目前只有 7900xtx 显卡在测试中，这意味着 Instinct 显卡用户可能需要独自解决故障。
- **API 演进：构建自定义梯度**：一位用户提出了一个新的 **自定义梯度 (custom grads) API**，希望实现类似于 **jax.customvjp** 的功能，以增强用于量化训练等任务的 Tensor 操作。
   - 建议的改进目标是在 tinygrad.functions 中使用 **lazybuffers** 替换当前操作，提倡直接进行 Tensor 操作。
- **强化学习：多 GPU 指南**：寻求 Tinygrad 多 GPU 训练知识的用户被引导至 [beautiful_mnist_multigpu.py 示例](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist_multigpu.py)，该示例重点展示了模型和数据分片 (sharding) 技术。
   - 分享了使用 `shard(axis=None)` 复制模型以及使用 `shard(axis=0)` 进行数据拆分的细节，有助于实现高效的并行训练。
- **等价性参与：类 Torch 的 Tensor 之战**：关于类似于 `torch.all` 的 Tensor 比较方法的查询，通过引入 `(t1 == t2).min() == 1` 的比较方式得到解决，随后在 Tinygrad 中添加了 **Tensor.all**。
   - 这一功能对齐的进展记录在 [此 Tinygrad commit](https://github.com/tinygrad/tinygrad/commit/6856f915d6f0e10d41e8e11c8976024989d90aa7) 中，为用户提供了更简便的 Tensor 操作。
- **优化障碍：Adam 的归零效应**：有反馈称 Tinygrad 中的 **Adam 优化器** 在第二次迭代步骤后会导致权重变为 **NaNs**，这与 SGD 的稳定性形成了鲜明对比。
   - 随着工程师们寻求防止优化器破坏学习过程的解决方案，这一调试对话仍在进行中。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **MInference 的敏捷加速**：一名成员重点介绍了微软的 [MInference 项目](https://github.com/microsoft/MInference)，该项目声称可以加速长上下文 LLMs 的推理，在 **A100** 上将 **延迟 (latency)** 降低多达 10 倍。
   - **MInference** 采用了新颖的近似和动态稀疏计算技术，旨在平衡准确性与 **性能效率**。
- **Yi-1.5-9B 结合 Hermes 2.5 批量上线**：**Yi-1.5-9B-Chat** 的更新显示其使用了 **OpenHermes 2.5** 进行微调，并公开分享了在 **AGIEval Benchmark** 中表现优异的 [模型和量化版本](https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat)。
   - 该增强模型在 **4x NVIDIA A100 GPU 上训练了超过 48 小时**，其“意识”令人印象深刻，目前正计划利用 **POSE** 将其上下文长度推升至 **32k tokens**。
- **Mistral 的聊天模板难题**：关于在 Axolotl 中进行 **Mistral 微调** 时使用哪种最佳 **chat_template** 的讨论引起了关注，答案取决于数据集结构。
   - 社区共识倾向于利用 **"chatml"** 模板，并提供了 YAML 配置示例来指导成员。

---

## [LLM Finetuning (Hamel + Dan)](https://discord.com/channels/1238365980128706560) Discord

- **MLOps 策略与 FP8 难题**：社区成员分享了见解，其中一人引用了一篇关注 **MLOps 实现** 的[博客文章](https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline)，另一人讨论了在 **分布式 vllm 推理** 中遇到的 **FP8 量化** 问题。
   - 针对 **FP8 的敏感性问题** 找到了解决方案，从而修正了输出，一个 [GitHub 线程](https://github.com/vllm-project/vllm/issues/6179) 为处理类似问题的用户提供了更多背景信息。
- **剖析模型集成**：一位成员正在[评估](https://discord.com/channels/1238365980128706560/1241163904927666287/1259183509965115392) 传统工具（如 **Transformers** & **Torch**）与来自 **OpenAI** 和 **Anthropic** 的成熟模型的集成。
   - 对话集中在寻找一种既能提供有效性又能针对特定项目需求进行无缝集成的最佳方法。
- **积分申领进入冲刺阶段**：#[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1259445018905542707) 频道的讨论明确指出：**积分申领已永久关闭**，标志着该福利的终结。
   - 会议强调，这种积分积累的终止适用于所有人，没有例外，并关闭了任何未来申领的途径。
- **Replicate 积分倒计时**：#[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/) 频道的一场对话透露，**前 25 个 Replicate 积分** 的有效期为一个月，这是对用户的一个重要更新。
   - 这一限时优惠似乎是使用策略的一个关键点，特别是对于那些依赖这些初始积分开展项目的用户。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **Interconnects 机器人：改进空间**：一位用户表示 **Interconnects 机器人** 表现良好，但最近的总结输出没有显著变化。
   - 该用户主张进行显著的更新或增强，以提升 Interconnects 机器人的功能。
- **RAG 使用案例与企业讨论**：成员们讨论了检索增强生成 (**RAG**) 模型，强调了它们在企业内部不断发展的使用案例。
   - 一些参与者建议 **RAG** 可能会增强内部知识库的使用，而另一些人则回忆起该模型在 *早期 AI 热潮* 期间的炒作。
- **翻找对 RAG 的早期反思**：对话触及了围绕 **RAG** 的最初兴奋感，并对最初过高的期望表达了共鸣。
   - 交流揭示了一个共同观点：早期的炒作尚未完全转化为广泛的企业采用。
- **成本效益与知识检索：企业视角**：讨论围绕 **RAG** 如何帮助提高企业模型的成本效益展开。
   - 有人提出，此类模型通过挖掘庞大的内部知识库，可以为企业开辟新的技术途径。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

- **Buzz 赢得赞赏并预告发布**：群组中对 **Buzz** 的热情显而易见，一位成员称赞了它的能力并暗示会有更多功能推出。
   - Autometa 预告了即将发布的版本，引发了社区的好奇心。
- **聚焦 FPGA：Autometa 即将举行的会议**：Autometa 宣布计划开会讨论 **FPGA** 领域的新应用，并指出了议程中的几个关键议题。
   - 成员们受邀参与并分享他们对当前项目中 **FPGA** 多样化用途的见解。
- **敞开大门：使用 Calendly 安排协作**：为了促进关于 AI alignment 的讨论，Autometa 为社区分享了一个[公开的 Calendly 链接](https://calendly.com/alignmentlab/meeting)。
   - 该链接作为安排深入讨论的公开邀请，为协作努力提供了一个平台。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **Flash 1.5 受到关注**：成员 **jeffreyw128** 表示 **Flash 1.5** 表现异常出色。
   - 该话题未提供额外的背景信息或详细讨论。
- **等待进一步见解**：目前关于 **Flash 1.5** 的技术性能和功能的细节较少。
   - 随着该工具获得更多关注，预计随后会有社区讨论和更深入的分析。

## [AI Stack Devs (Yoko Li)](https://discord.com/channels/1122748573000409160) Discord

- **Sprite Quest: Google Image Galore**: 一位成员提到 **sprites** 素材来源于 **random Google image searches**，以满足素材收集快速且多样化的需求。
   - 重点是在不购买的情况下获取多样化的 **sprites**，而 **tilesets** 是唯一的 **paid assets**。
- **Tileset Trade: The Only Expense**: 对话透露，唯一投入资金的资产是 **tilesets**，凸显了注重成本的方法。
   - 这种区别强调了对资产的有条理选择，即 **money spent solely on tilesets**，而 **sprites obtained freely** 则通过搜索引擎获取。



---



## [MLOps @Chipro](https://discord.com/channels/814557108065534033) Discord

- **EuroPython Vectorization Talk**: 一位用户表达了他们参加 **EuroPython** 的意愿，并暗示即将有一个专注于 **vectorization** 的演讲。
   - 感兴趣的社区成员可能会参加，以深入了解 **vectorization** 在 Python 中的作用，这是 **AI engineering** 的一个重要方面。
- **Community Engagement at Conferences**: 用户提到 **EuroPython** 突显了社区在 Python 会议上的外展和活跃存在。
   - 这鼓励了 **AI and Machine Learning** 领域的 Python 从业者之间的社交和知识共享。



---



## [Mozilla AI](https://discord.com/channels/1089876418936180786) Discord

- **Google's Gem Sparkles in Size and Performance**: Google 的 [Gemma 2 9B](https://blog.google/technology/developers/google-gemma-2/) 已作为开源语言模型进入赛场，因其强劲的性能而受到关注。
   - **Despite its smaller scale**，Gemma 2 9B 挑战了 GPT-3.5 等重量级模型，适用于资源有限的环境。
- **Lambda Lift-Off: Gemma 2 Reaches Serverless Heights**: 社区通过在 AWS Lambda 上将 Google 的 Gemma 2 与 Mozilla 的 Llamafile 集成，探索了 **serverless AI inference**，如[本教程](https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda)所示。
   - 这种 serverless 方法使得在低资源设置（包括移动设备、个人电脑或本地化云服务）中高效部署 Gemma 2 9B 成为可能。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord

- **Models Fusion Forge**: 一位成员提议使用 **Hermes-2-Theta-Llama-3-70B** 作为构建 **Llama3-DiscoLeo-Instruct-70B** 模型的基础。
   - 随后的对话暗示了合并两个模型的能力以增强性能的优势。
- **Enhancement Speculations**: 工程师们考虑了模型集成的预期收益，重点关注 **Hermes-2-Theta-Llama-3-70B** 和 **Llama3-DiscoLeo-Instruct**。
   - 对话围绕通过战略性融合不同模型特性来提升 AI 能力的潜在进展展开。



---


**Torchtune Discord** 没有新消息。如果该公会长期沉默，请告知我们，我们将移除它。


---


**AI21 Labs (Jamba) Discord** 没有新消息。如果该公会长期沉默，请告知我们，我们将移除它。


---

# PART 2: Detailed by-Channel summaries and links


{% if medium == 'web' %}




### **Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1258859978572173475)** (804 messages🔥🔥🔥): 

> - `Model Licensing`
> - `Performance and Troubleshooting`
> - `Generation Techniques and Tools`
> - `Community and Ethical Concerns`
> - `Image Upscaling Techniques`

- **Stability AI 模型许可混淆**：社区正在努力理解新的 Stability AI 模型许可条款，特别是对于年收入超过 100 万美元的企业。
   - 虽然提供了一些澄清，但关于将 SD3 用于商业目的以及对小企业影响的担忧依然存在。
- **图像生成的性能问题**：用户报告在将 controlnet 与 text2img 结合使用时出现显著减速，这通常是由于 VRAM 限制导致与系统 RAM 之间发生内存交换（memory shuffling）。
   - 调整 Windows pagefile 设置并使用 offloading 策略可以缓解部分减速问题。
- **高级图像放大策略**：分享了一个涉及多个放大步骤以及 Photoshop、SUPIR 和 transformer upscalers 等软件的详细工作流，用于实现高分辨率图像。
   - 该方法避免了分块（tiling）等常见问题，旨在保持细节添加与图像一致性之间的平衡。
- **社区对模型质量和发布的反应**：社区对 SD3 模型的质量表示失望，认为其不如之前的版本，并对仓促发布表示担忧。
   - 人们对 8B 版本等改进模型充满期待，并持续讨论 NSA 介入的潜在影响及其他伦理问题。
- **技术支持与解决方案**：讨论内容包括解决特定 prompts 的问题、集成外部工具以获得更好结果，以及处理硬件限制。
   - 提供了关于在 prompts 中有效使用术语以及利用多种软件工具实现理想图像生成结果的建议。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://www.invoke.com/">Invoke | 为您的业务打造的 AI 图像生成器</a>：Invoke 是唯一的生成式创作工具和自定义 AI 模型管理器，在这里您可以保留对作品、模型和 IP 的完全控制权和所有权。</li><li><a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - Meta Learning 书籍章节总结要点</a>：未找到描述</li><li><a href="https://www.youtube.com/watch?v=Ly6USRwTHe0">Krita 生成式 AI - 配合 ControlNet</a>：使用 Stable Diffusion 在 Krita 中以最少的操作生成图像。https://github.com/Acly/krita-ai-diffusion 现在支持 ControlNet scribble 和 line art....</li><li><a href="https://www.interstice.cloud/plugin">Interstice</a>：未找到描述</li><li><a href="https://huggingface.co/tianweiy/DMD2/tree/main">tianweiy/DMD2 at main</a>：未找到描述</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI/blob/master/docs/Model%20Support.md#pixart-sigma>">SwarmUI/docs/Model Support.md at master · mcmonkeyprojects/SwarmUI</a>：SwarmUI，一个模块化的 Stable Diffusion Web-User-Interface，重点在于让高级工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/SwarmUI</li><li><a href="https://github.com/CS1o/Stable-Diffusion-Info/wiki/Installation-Guides#nvidia-automatic1111-webui-stable-diffusion-webui">安装指南</a>：Stable Diffusion 知识库（设置、基础、指南等）- CS1o/Stable-Diffusion-Info</li><li><a href="https://www.runcomfy.com/comfyui-web">ComfyUI Online - 免费 ComfyUI Web</a>：无需安装即可免费在线使用 ComfyUI，轻松构建 Stable Diffusion 工作流，并在几秒钟内生成图像。</li><li><a href="https://comfyuiweb.com/">Comfyui Web - 免费在线使用 ComfyUI</a>：未找到描述</li><li><a href="https://www.reddit.com/r/krita/comments/r5nq1y/krita_wont_allow_me_to_change_its_settings/">Reddit - 深入探索</a>：未找到描述</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Command-Line-Arguments-and-Settings">命令行参数和设置</a>：Stable Diffusion web UI。通过在 GitHub 上创建账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://research.nvidia.com/labs/dir/jedi/"> 联合图像扩散 (Joint-image Diffusion)</a>：未找到描述</li><li><a href="https://stability.ai/news/license-update">社区许可证 — Stability AI</a>：我们的新社区许可证（Community License）现在对研究、非商业和商业用途免费。只有当您的年收入超过 100 万美元且使用 Stability AI 模型时，才需要付费的企业许可证...</li><li><a href="https://github.com/mcmonkeyprojects/SwarmUI">GitHub - mcmonkeyprojects/SwarmUI: SwarmUI，一个模块化的 Stable Diffusion Web-User-Interface，重点在于让高级工具易于访问、高性能且具有可扩展性。</a>：SwarmUI，一个模块化的 Stable Diffusion Web-User-Interface，重点在于让高级工具易于访问、高性能且具有可扩展性。 - mcmonkeyprojects/SwarmUI</li><li><a href="https://civitai.com/models/74776/moebius-jean-giraud-style">Moebius (Jean Giraud) 风格 - SD XL | Stable Diffusion LoRA | Civitai</a>：Moebius，也被称为 Jean Giraud，是一位法国漫画艺术家和插画家，以其在科学领域具有影响力和远见的的作品而闻名...</li><li><a href="https://www.sca.org/">首页 - SCA.org</a>：SCA 是一个国际组织，致力于通过活动和项目研究/重现 17 世纪以前的技能、艺术、战斗、文化和历史。</li><li><a href="https://github.com/Acly/krita-ai-diffusion?tab=readme-ov-file">GitHub - Acly/krita-ai-diffusion: 在 Krita 中使用 AI 生成图像的流线型界面。支持使用可选文本提示进行局部重绘 (Inpaint) 和外延绘制 (Outpaint)，无需繁琐调整。</a>：在 Krita 中使用 AI 生成图像的流线型界面。支持使用可选文本提示进行局部重绘 (Inpaint) 和外延绘制 (Outpaint)，无需繁琐调整。 - Acly/krita-ai-diffusion</li><li><a href="https://old.reddit.com/r/StableDiffusion/comments/1b50sp0/ccsr_vs_supir_upscale_comparison_portrait/```">CCSR vs SUPIR 放大对比（人像摄影）</a>：我对 256x384 放大 8 倍至 2048x3072 进行了简单的对比。我主要将 SD 用于真实人像摄影的放大，因此面部忠实度...</li><li><a href="https://safebooru.org/index.php?page=post&s=list&tags=looking_at_viewer">Safebooru / looking_at_viewer</a>：未找到描述</li><li><a href="https://github.com/civitai/civitai/blob/feb2337c202ab82661958481de9652a4a6b3417b/src/utils/metadata/lists/words-young.json#L4">civitai/src/utils/metadata/lists/words-young.json at feb2337c202ab82661958481de9652a4a6b3417b · civitai/civitai</a>：模型、文本反转 (textual inversions) 等资源的仓库 - civitai/civitai</li><li><a href="https://github.com/civitai/civitai/blob/feb2337c202ab8">

2661958481de9652a4a6b3417b/src/utils/metadata/lists/words-nsfw.json">civitai/src/utils/metadata/lists/words-nsfw.json at feb2337c202ab82661958481de9652a4a6b3417b · civitai/civitai</a>: 一个模型、Textual Inversions 等内容的仓库 - civitai/civitai</li><li><a href="https://civitai.com/models/447902/mangled-merge-xl?modelVersionId=619849">Mangled Merge XL - v3.0 | Stable Diffusion Checkpoint | Civitai</a>: V3：我很荣幸向大家介绍 3.0 版本，这是 Mangled Merge XL 系列的下一个迭代版本。我花了一些时间研究 DARE/TIES ...</li><li><a href="https://civitai.com/models/101055?modelVersionId=128080">SD XL - v1.0 Refiner VAE fix | Stable Diffusion Checkpoint | Civitai</a>: 最初发布于 Hugging Face，经 Stability AI 许可在此分享。SDXL 由用于 Latent Diffusion 的两步流水线组成：首先...
</li>
</ul>

</div>
  

---



### **HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1258861335119270028)** (605 messages🔥🔥🔥): 

> - `Hermes 2`
> - `GPTs Agents`
> - `OpenAI's sidebars`
> - `AI 项目融资`
> - `Inference API 问题` 


- **Inference API 面临停滞问题**：几位成员报告了 Inference Endpoints 初始化时间过长的问题，潜在原因可能是 GPU 可用性问题或特定的配置设置。一位成员建议使用 [AWS Nvidia A10G on eu-west-1](https://aws.amazon.com/ec2/instance-types/a10g/) 作为替代方案。
- **GPTs Agents 在初始训练后无法学习**：一位成员分享了关于 GPTs Agents 在初始训练后无法从提供的额外信息中学习的担忧。
- **自定义 LLM 指标请求**：一位用户询问了有关 LLM 的自定义指标，如响应完整性、文本相似度和幻觉指数（Hallucination Index）。他们提到评估 Levenshtein 距离、Surprisal/Perplexity 以及特定任务相关的指标，如用于机器翻译的 BLEU 分数。
- **考虑 Regex 模式的反垃圾邮件措施**：关于改进反垃圾邮件措施的讨论包括实施 Regex 模式，以自动过滤和禁止某些词汇或短语。
- **社区对摘要功能的反馈**：社区讨论了 Discord 内置摘要功能的实用性（该功能使用 OpenAI 的 GPT-3.5），并对隐私和有效性表示了担忧。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>

<li><a href="https://www.kaggle.com/learn/intro-to-machine-learning">学习机器学习入门教程 | Kaggle</a>: 未找到描述</li><li><a href="https://www.youtube.com/@matthew_berman">Matthew Berman</a>: 人工智能 (AI)、开源、生成艺术、AI 艺术、未来主义、ChatGPT、大语言模型 (LLM)、Machine Learning、技术、编程、教程、AI 新闻等 ** 独家 Pine...</li><li><a href="https://docs.continue.dev/how-to-use-continue#ask-questions-about-your-codebase">🧑‍🎓 如何使用 Continue | Continue</a>: 在编码时通过 Continue 使用 LLM</li><li><a href="https://docs.confident-ai.com/docs/metrics-introduction#using-a-custom-llm">指标 | DeepEval - 开源 LLM 评估框架</a>: 快速摘要</li><li><a href="https://www.gradio.app/guides/getting-started-with-the-python-client">Python 客户端入门</a>: Gradio 分步教程</li><li><a href="https://huggingface.co/spaces/InstantX/InstantStyle">InstantStyle - 由 InstantX 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/nroggendorff/llava">Llama - 由 nroggendorff 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - 元学习 (Meta Learning) 书籍分章节摘要要点</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/qnguyen3/nanoLLaVA">nanoLLaVA-1.5 - 由 qnguyen3 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/LanguageBind/Video-LLaVA">Video LLaVA - 由 LanguageBind 提供的 Hugging Face Space</a>: 未找到描述</li><li><a href="https://scale.com/leaderboard">SEAL 排行榜</a>: 未找到描述</li><li><a href="https://huggingface.co/stabilityai/stable-diffusion-3-medium-diffusers">stabilityai/stable-diffusion-3-medium-diffusers · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/discord-community/HuggingMod/discussions/1">discord-community/HuggingMod · 请合并</a>: 未找到描述</li><li><a href="https://huggingface.co/settings/billing">Hugging Face – 构建未来的 AI 社区。</a>: 未找到描述</li><li><a href="https://lu.ma/yzzespyu">Block 举办的为期 4 周的 AI 学习小组：Andrej Karpathy 的 Zero to GPT Hero · Luma</a>: 注意：这是一个连续 4 周的重复活动，从 7 月 24 日开始，到 8 月 14 日结束！~ GPT 现象在很大程度上归功于……</li><li><a href="https://www.youtube.com/watch?v=WhAMvOEOWJw">一分钟 Gradio #1：动态渲染</a>: 一分钟 Gradio #1 - 快速学习 Gradio 技巧！今天，我们将讨论 Gradio 中的动态渲染（即 @gr.render 装饰器）以及它如何让...</li><li><a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">来自 Prashant Dixit (@Prashant_Dixit0) 的推文</a>: ✨开源全面的 LLM 词汇表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，别忘了给 st...</li><li><a href="https://huggingface.co/blog/not-lain/rag-chatbot-using-llama3">使用 llama3 的 RAG 聊天机器人</a>: 未找到描述</li><li><a href="https://tenor.com/view/huh-cat-cat-huh-small-cat-huh-what-gif-2593177363967991691">Huh Cat GIF - Huh Cat Cat huh - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/rip-buff-spongebob-ripping-shirt-gif-14008353">Rip Buff GIF - Rip Buff Spongebob - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://tenor.com/view/iceage-possum-peaceout-gif-2272177960667492692">Iceage Possum GIF - IceAge Possum PeaceOut - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://aclanthology.org/2023.eamt-1.19/">大语言模型是翻译质量的最先进评估器</a>: Tom Kocmi, Christian Federmann. 欧洲机器翻译协会 (EAMT) 第 24 届年会论文集. 2023.</li><li><a href="https://github.com/nroggendorff/diffusion/blob/main/zelda.ipynb">nroggendorff/diffusion 项目 main 分支下的 diffusion/zelda.ipynb</a>: 通过在 GitHub 上创建账号来为 nroggendorff/diffusion 的开发做出贡献。</li><li><a href="https://github.com/confident-ai/deepeval/blob/main/deepeval/metrics/toxicity/template.py">confident-ai/deepeval 项目 main 分支下的 deepeval/deepeval/metrics/toxicity/template.py</a>: LLM 评估框架。通过在 GitHub 上创建账号来为 confident-ai/deepeval 的开发做出贡献。</li><li><a href="https://github.com/aymeric-roucher/agent_reasoning_benchmark/">GitHub - aymeric-roucher/agent_reasoning_benchmark: 🔧 比较 Agent 系统在多个基准测试中的表现。📊🚀</a>: 🔧 比较 Agent 系统在多个基准测试中的表现。📊🚀 - aymeric-roucher/agent_reasoning_benchmark</li><li><a href="https://github.c

om/hpcaitech/Open-Sora">GitHub - hpcaitech/Open-Sora: Open-Sora: Democratizing Efficient Video Production for All</a>: Open-Sora: 为所有人实现高效视频制作的民主化 - hpcaitech/Open-Sora</li><li><a href="https://github.com/huggingface/lighteval">GitHub - huggingface/lighteval: LightEval is a lightweight LLM evaluation suite that Hugging Face has been using internally with the recently released LLM data processing library datatrove and LLM training library nanotron.</a>: LightEval 是一个轻量级的 LLM 评估套件，Hugging Face 内部一直在将其与最近发布的 LLM 数据处理库 datatrove 和 LLM 训练库 nanotron 配合使用。 - hug...</li><li><a href="https://tenor.com/view/red-kit-gif-11737462">Red Kit GIF - Red Kit - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/docs/huggingface_hub/main/en/package_reference/inference_client#huggingface_hub.InferenceApi">Inference</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1259311320008753214)** (4 条消息): 

> - `Boid AI`
> - `LLM/GenAI 术语表`
> - `使用 Scikit-Learn 的 GPA 预测器`
> - `生成式文本项目` 


- **介绍 Boid AI 概念**：一位成员介绍了 **Boid AI** 的概念，其中 'boid' 代表 'bird-oid'（类鸟体），暗示类鸟类的 AI 行为。
- **开源全面的 LLM/GenAI 术语表**：一位成员通过 GitHub 分享了一个 [全面的 LLM 术语表](https://github.com/freetoolsarebest/llm-glossary)，旨在让 AI 术语更易于理解。
   - *探索、学习并添加关于 LLMs 和 GenAI 的术语。*
- **使用 Scikit-Learn 构建 GPA 预测器**：一位成员分享了在 Kaggle 上使用 **Scikit-Learn** 创建粗略的 **GPA 预测器**，并阅读了 Geron Aurelion 的《Hands-On Machine Learning》。
   - 他们还观看了一些 [3Blue1Brown 的神经网络系列视频](https://www.youtube.com/user/3blue1brown) 以进一步学习。
- **关于生成式文本项目的建议**：一位成员就启动生成式文本项目寻求建议，在是用现有模型还是从头构建之间犹豫不决。
   - 他们提到有人建议将 Hugging Face 与 Langchain 结合使用，并寻求使用 Langchain 的理由。



**提到的链接**: <a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">来自 Prashant Dixit (@Prashant_Dixit0) 的推文</a>: ✨开源全面的 LLM 术语表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，不要忘记给 star...

  

---


### **HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1258883504171909271)** (16 条消息🔥): 

> - `Claude Artifacts`
> - `PersonaHub 数据集`
> - `脱敏/伪匿名化技术`
> - `管理员请求` 


- **Claude 专注于 artifacts 以获得令人印象深刻的结果**：一位用户推测 Claude 的出色表现可能归功于其对 'artifacts' 的关注。
- **探索 PersonaHub 数据集**：一位用户分享了 [PersonaHub 数据集](https://huggingface.co/datasets/proj-persona/PersonaHub)，该数据集旨在理解表演艺术中心和城市规划。
   - 该数据集包含诸如调度多场演出节日以及对比不同社区公共服务等场景。
- **脱敏技术影响模型质量**：来自 [TrustNLP 2023](https://aclanthology.org/2023.trustnlp-1.20/) 的一篇论文分析了用于文本分类和摘要的脱敏（Pseudonymization）技术。
   - *用伪名替换命名实体* 在某些 NLP 任务上保持了性能。
- **频繁的管理员提及和垃圾信息问题**：成员们频繁地 @ 管理员并要求封禁重复的垃圾信息，特别是提到了 'opensea'。
   - 出现了“请封禁 opensea 这个词”的呼吁，以及关于被盗用户和潜在机器人的讨论。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://aclanthology.org/2023.trustnlp-1.20/">Privacy- and Utility-Preserving NLP with Anonymized data: A case study of Pseudonymization</a>: Oleksandr Yermilov, Vipul Raheja, Artem Chernodub. 第三届可信自然语言处理研讨会论文集 (TrustNLP 2023). 2023.</li><li><a href="https://huggingface.co/datasets/proj-persona/PersonaHub">proj-persona/PersonaHub · Hugging Face 数据集</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1259013525809008680)** (24 条消息🔥): 

> - `in10search Tabs Sidepanel AI`
> - `ZeroGPU HuggingFace Space`
> - `qdurllm`
> - `AI on-call developer: merlinn`
> - `DarkWebSight` 


- **使用 in10search Tabs Sidepanel AI 进行浏览**：一款名为 **in10search Tabs Sidepanel AI** 的新浏览器侧边栏扩展程序，集成了水平标签页和 ChatGPT。更多详情请见 [GitHub](https://github.com/vtempest/in10search-chrome)。
- **用于 Stable Diffusion 模型的 ZeroGPU HuggingFace Space**：一位成员介绍了一个 **HuggingFace Space**，允许用户比较多个 **Stable Diffusion Models**，如 **SD3 Medium**、**SD2.1**、**SDXL** 等。点击[此处](https://huggingface.co/spaces/Nick088/stable-diffusion-arena)查看。
- **qdurllm：结合 Qdrant 和 LLM 的本地搜索引擎**：新推出的开源产品 **qdurllm** 结合了 **Qdrant**、**URL scraping** 和 **Large Language Models**，用于本地搜索和聊天。在其 [GitHub 仓库](https://github.com/AstraBert/qdurllm)中进一步探索。
- **AI 值班开发人员：merlinn**：名为 **merlinn** 的 AI 值班开发人员通过提供上下文信息来帮助调查生产事故。请在 [GitHub](https://github.com/merlinn-co/merlinn) 上查看并提供反馈。
- **gary4live Ableton 插件**：一款名为 **gary4live** 的有趣 Ableton 插件已在 Gumroad 发布。这是一个将趣味工作流与 AI 相结合的 max4live 设备，可在此处[免费获取](https://thecollabagepatch.com)。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/Nick088/stable-diffusion-arena">Stable Diffusion Arena - a Hugging Face Space by Nick088</a>: 未找到描述</li><li><a href="https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant">Self Reviewing Coding Assistant - a Hugging Face Space by as-cle-bert</a>: 未找到描述</li><li><a href="https://huggingface.co/datasets/Csplk/DarkWebSight">Csplk/DarkWebSight · Datasets at Hugging Face</a>: 未找到描述</li><li><a href="https://youtu.be/D4E9zAmrCQ8?si=YDTa8ZIOTNCSxG9H">ghost chords - the captain&#39;s chair, season two - episode 1</a>: 00:00 - intro01:28 - ghost chords explained02:25 - the riff03:40 - the robot joins in08:55 - the trackseason one on spotify:https://open.spotify.com/album/7h...</li><li><a href="https://wandb.ai/sauravmaheshkar/llamaindex-local-models-index/reports/Training-a-chatbot-on-personal-data-with-LlamaIndex-and-W-B--Vmlldzo4MzQzMDE3">Training a chatbot on personal data with LlamaIndex and W&B</a>: 在本文中，我们将介绍如何使用 LlamaIndex 和本地模型，并结合 Weights &amp; Biases 集成，在个人数据上创建一个聊天机器人。</li><li><a href="https://tenor.com/view/lost-lost-tv-show-desmond-desmond-hume-lost-desmond-gif-17240446">Lost Lost Tv Show GIF - Lost Lost Tv Show Desmond - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://youtu.be/NW42xY651cQ">Design ChatGPT like AI Assiatant | ML System Design | #machinelearning</a>: 我们探讨了创建类似 ChatGPT 的 AI 助手的 ML 系统设计问题。视频中展示的 AI 助手的目的是自动...</li><li><a href="https://github.com/AstraBert/qdurllm">GitHub - AstraBert/qdurllm: Search your favorite websites and chat with them, on your desktop🌐</a>: 在桌面上搜索您喜爱的网站并与之聊天🌐 - AstraBert/qdurllm</li><li><a href="https://github.com/vtempest/in10search-chrome">GitHub - vtempest/in10search-chrome: in10search Tabs Sidepanel AI   - Horizontal Tabs in Browser Sidepanel with ChatGPT</a>: in10search Tabs Sidepanel AI - 带有 ChatGPT 的浏览器侧边栏水平标签页 - vtempest/in10search-chrome</li><li><a href="https://github.com/merlinn-co/merlinn">GitHub - merlinn-co/merlinn: Open source AI on-call developer 🧙‍♂️ Get relevant context &amp; root cause analysis in seconds about production incidents and make on-call engineers 10x better 🏎️</a>: 开源 AI 值班开发人员 🧙‍♂️ 在几秒钟内获取有关生产事故的相关上下文和根因分析，让值班工程师效率提升 10 倍 🏎️ - merlinn-co/merlinn</li><li><a href="https://github.com/U-C4N/H.I.BOT/">GitHub - U-C4N/H.I.BOT</a>: 通过在 GitHub 上创建账户，为 U-C4N/H.I.BOT 的开发做出贡献。</li><li><a href="https://thecollabagepatch.com">no title found</a>: 未找到描述</li><li><a href="https://x.com/thepatch_kev/status/1810063563823907172">Tweet from thecollabagepatch (@thepatch_kev)</a>: 13 位传奇人物刚刚收到了关于 gary4live 的邮件，这是一款实现此功能的 Ableton 插件，现在就在 gumroad 下载吧 ⬇️链接 @_buildspace @_nightsweekends</li><li><a href="https://portfolio-app-raj.streamlit.app/">no title found</a>: 未找到描述
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1259056094878105600)** (22 条消息🔥): 

> - `用于目标检测的 Torchmetrics`
> - `RT-DETR 模型发布`
> - `用于视觉语言模型的 CogVLM2`
> - `零样本目标检测模型`
> - `MaskFormer 与实例分割` 


- **Torchmetrics 被推荐用于目标检测**：Torchmetrics 被建议用于目标检测指标，并在[官方示例脚本](https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection)中与 Trainer API 和 Accelerate 一起使用。
- **RT-DETR 模型发布**：[RT-DETR](https://x.com/mervenoyann/status/1807790959884665029) 是一款类 YOLO 的实时目标检测模型，结合了卷积和基于 Attention 的 Transformer。
   - 它采用 Apache 2.0 许可证，兼具两者的优势。
- **用于视觉语言模型的 CogVLM2**：[CogVLM2](https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B) 被推荐用于各种大规模视觉语言模型任务，在 TextVQA 和 DocVQA 等基准测试中表现出色。
- **零样本目标检测模型**：Transformers 库支持 OWL-ViT、OWLv2 和 Grounding DINO 等零样本目标检测模型，用于基于文本描述的目标检测。
   - 这些模型还可以执行图像引导的目标检测，如本 [demo](https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb) 所示。
- **MaskFormer 与实例分割**：在 ADE20k 等数据集上训练用于语义分割的 MaskFormer 模型，可以通过[此处](https://github.com/huggingface/transformers/tree/main/examples/pytorch/instance-segmentation)新增的官方脚本扩展到实例分割任务。
   - 建议从预训练的 COCO 模型开始，在实例分割任务上进行微调。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://segments.ai/,">Segments</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/finetune-florence2">Fine-tuning Florence-2 - Microsoft's Cutting-edge Vision Language Models</a>: 未找到描述</li><li><a href="https://huggingface.co/facebook/maskformer-swin-large-ade">facebook/maskformer-swin-large-ade · Hugging Face</a>: 未找到描述</li><li><a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">no title found</a>: 未找到描述</li><li><a href="https://huggingface.co/THUDM/cogvlm2-llama3-chat-19B">THUDM/cogvlm2-llama3-chat-19B · Hugging Face</a>: 未找到描述</li><li><a href="https://x.com/mervenoyann/status/1807790959884665029">Tweet from merve (@mervenoyann)</a>: Real-time DEtection Transformer (RT-DETR) 已登陆 @huggingface transformers 🤩，采用 Apache 2.0 许可证 😍 DETR 在实时目标检测上能击败 YOLO 吗？继续阅读 👀</li><li><a href="https://huggingface.co/spaces/andito/Florence-2-DocVQA/blob/main/app.py#L25">app.py · andito/Florence-2-DocVQA at main</a>: 未找到描述</li><li><a href="https://x.com/skalskip92/status/1808874766515818840">Tweet from SkalskiP (@skalskip92)</a>: 没有新的 VLM 了？我终于在为我的足球 AI 项目制作 YouTube 教程了；教程应该下周发布。敬请期待: https://www.youtube.com/roboflow</li><li><a href="https://github.com/NielsRogge/Transformers-Tutorials/blob/master/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb">Transformers-Tutorials/OWLv2/Zero_and_one_shot_object_detection_with_OWLv2.ipynb at master · NielsRogge/Transformers-Tutorials</a>: 此仓库包含我使用 HuggingFace 的 Transformers 库制作的 demo。 - NielsRogge/Transformers-Tutorials</li><li><a href="https://huggingface.co/facebook/mask2former-swin-small-coco-instance">facebook/mask2former-swin-small-coco-instance · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/huggingface/transformers/tree/main/examples/pytorch/object-detection">transformers/examples/pytorch/object-detection at main · huggingface/transformers</a>: 🤗 Transformers: 面向 Pytorch、TensorFlow 和 JAX 的前沿机器学习。 - huggingface/transformers
</li>
</ul>

</div>
  

---

### **HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1259012438276177941)** (7 messages): 

> - `Label Error in NLP Dataset` (NLP 数据集中的标签错误)
> - `Extending deepseek-ai model context length` (扩展 deepseek-ai 模型的上下文长度)
> - `Byte Pair Encoding Implementation in C` (C 语言实现的 Byte Pair Encoding)
> - `Comprehensive LLM/GenAI Glossary` (全面的 LLM/GenAI 术语表)


- **标签错误令用户受挫**：一位用户在处理从 .txt 文件导入的 NLP 数据集时，报告了 `ValueError: Invalid string class label ['B-COMPANY']` 错误。
   - 该问题导致错误消息频繁变化，使故障排除过程变得复杂。
- **deepseek-ai 模型上下文长度咨询**：一位用户询问是否可以在不进行微调的情况下，将 `deepseek-ai/deepseek-math-7b-rl` 模型的上下文长度从 4k 扩展到 8k。
   - 他们探索了如 vLLM 或直接通过 HF 加载等选项来实现这一扩展。
- **C 语言版 Byte Pair Encoding 发布**：Ashpun 宣布在 C 语言中实现了一个极简的 [Byte Pair Encoding 机制](https://github.com/ash-01xor/bpe.c)。
   - 博客文章即将发布，代码现已在 GitHub 上可用。
- **LLM/GenAI 术语表开源**：Prashant Dixit 推广了一个[全面的 LLM 术语表](https://x.com/Prashant_Dixit0/status/1809900514097979768)，旨在让每个人都能更轻松地理解 AI。
   - 术语会定期更新，该项目是开源的，可在 [GitHub](https://github.com/freetoolsarebest/llm-glossary) 上获取。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">来自 Prashant Dixit (@Prashant_Dixit0) 的推文</a>: ✨开源全面的 LLM 术语表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，别忘了给个 star...</li><li><a href="https://github.com/ash-01xor/bpe.c">GitHub - ash-01xor/bpe.c: 用于分词过程的简单 Byte pair Encoding 机制，纯 C 语言编写</a>: 用于分词过程的简单 Byte pair Encoding 机制，纯 C 语言编写 - ash-01xor/bpe.c
</li>
</ul>

</div>
  

---


### **HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1259958137360879778)** (1 messages): 

> - `Artifacting in sd-vae` (sd-vae 中的伪影问题)
> - `Common issues in sd-vae reconstruction` (sd-vae 重建中的常见问题)


- **sd-vae 中的伪影引发疑问**：一位成员询问在使用 **sd-vae 进行重建**时，出现蓝色和白色像素伪影是否正常。
   - 这引发了关于 **sd-vae** 中常见问题和像素伪影故障排除方法的讨论。
- **识别 sd-vae 中的常见问题**：成员们深入探讨了 sd-vae 中遇到的常见问题，重点关注像素伪影和重建质量。
   - 故障排除建议包括尝试不同的参数设置，并分享结果以获取社区反馈。


  

---


### **HuggingFace ▷ #[gradio-announcements](https://discord.com/channels/879548962464493619/1014577787039924226/1259850068085964870)** (1 messages): 

> - `Enhanced Documentation Search on Gradio` (Gradio 文档搜索功能增强)
> - `Navigation of Gradio Documentation Pages` (Gradio 文档页面的导航)


- **Gradio 增强文档搜索**：Gradio 社区宣布在其文档页面中发布了全新的[增强搜索功能](https://www.gradio.app/)，使导航和获取信息更加容易。
   - 他们邀请用户访问文档进行体验，并强调他们致力于改善用户体验。
- **快速入门和教程现在更容易访问**：改进后的搜索工具帮助用户更高效地找到快速入门指南和深入教程。
   - Gradio 鼓励用户继续发送反馈，以进一步提升体验。



**提及的链接**: <a href="https://www.gradio.app/">Gradio</a>: 构建并分享令人愉悦的机器学习应用

  

---

### **Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1258875621149380689)** (502 条消息🔥🔥🔥): 

> - `Perplexity 的问题`
> - `Pro Search 及其局限性`
> - `订阅替代方案`
> - `图像生成`
> - `技术问题与 Bug` 


- **用户面临 Perplexity 的性能问题**：多位用户提到 Perplexity 经常无法提供准确或最新的文章，尽管提示词非常精确，仍会返回过时信息。
   - 一位用户对后续提问中的上下文丢失表示沮丧，认为 GPT-4o 比 Claude 3.5 能更好地维持上下文。
- **Pro Search 的价值让部分用户失望**：一些用户认为 Pro 订阅是在浪费钱，认为其结果与免费版相比没有显著改进。
   - 尽管如此，Perplexity Pro 提供了更高级的搜索能力和频繁的更新，但部分用户认为其他替代服务以相同或更低的成本提供了更好的价值。
- **探索替代 AI 服务**：用户讨论了各种替代方案，如 Merlin.ai、Abacus.AI 中的 ChatLLM 以及 You.com，并分享了关于它们性能和可用性的褒贬不一的评价。
   - Monica.ai 以及配合 LibreChat 使用的 OpenRouter 因其丰富的功能和用户友好的界面而受到关注，成为强有力的竞争对手。
- **Perplexity 的图像生成能力**：一些用户不知道 Perplexity 可以生成图像，需要说明如何访问此功能。
   - Perplexity Pro 用户拥有图像生成权限，利用图像生成中的自定义提示词选项可以获得更好的效果。
- **Bug 和技术问题**：多位用户报告了 Perplexity 中的 Bug，例如文本重叠、上下文丢失以及生成脚本时的问题。
   - 社区建议使用系统提示词（system prompts）作为权宜之计，并强调需要更直观、更简单的功能来提升用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://apps.apple.com/us/app/deepl-translate-write/id1552407475">‎DeepL: translate &amp; write</a>：DeepL 是您首选的 AI 翻译和写作助手，提供精准的翻译、强大的语法修正和清晰的风格增强。凭借先进的 Language AI 力量，DeepL 允许您进行...</li><li><a href="https://msty.app">Msty - Using AI Models made Simple and Easy</a>：与文件聊天、理解图像，并离线访问各种 AI 模型。在一个统一的界面中使用来自 OpenAI、Claude、Perplexity、Ollama 和 HuggingFace 的模型。</li><li><a href="https://console.groq.com/playground">GroqCloud</a>：体验世界上最快的推理速度</li><li><a href="https://x.com/baronitaigas/status/1809155575340544500?s=19">来自 Baron of the Taiga (@baronitaigas) 的推文</a>：⚡️🇱🇻：拉脱维亚军队将在官方文件中开始将俄罗斯（Russia）的首字母写为小写 'r' —— 拉脱维亚国防部长公共事务官员 Sandra Brale。</li><li><a href="https://tenor.com/view/laughing-spongebob-patrick-blush-gif-4679526">大笑的海绵宝宝 GIF - Laughing Spongebob Patrick - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://gitlab.com/monnef/ailin">monnef / AIlin · GitLab</a>：AIlin 是一个将 Perplexity.ai 等 AI 服务与您的本地计算机连接的工具。</li><li><a href="https://www.threads.net/@perplexity.ai/post/C7mU3LdC6Cj?xmt=AQGzr8iLqFKizU24JG74yUmtoD5g8xMIjIC5fZLt_7B_Iw">Threads 上的 Perplexity AI (@perplexity.ai)</a>：说到升级！我们很高兴推出 Perplexity Pages，这是一种将您的研究转化为视觉精美文章的简便方法。通过格式化的图像和章节，Pages 让您可以分享...</li><li><a href="https://chatllm.abacus.ai/">Abacus.AI - </a>：Abacus.AI 是世界上第一个由 AI 而非人类大规模构建应用型 AI Agent 和系统的 AI 平台。利用生成式 AI 和其他新型神经网络技术，AI 可以构建 LLM 应用、生成式 AI...</li><li><a href="https://chromewebstore.google.com/detail/deepl-translate/cofdbpoegempjloogbagkncekinflcnj?hl=fr).">DeepL Translate</a>：使用世界上最准确的翻译器 DeepL Translate，在阅读和写作时进行翻译。</li><li><a href="https://www.mlb.com/stats/?playerPool=ALL_CURRENT)">2024 MLB 球员击球统计数据排行榜</a>：球员击球统计、MLB 本垒打王、打击率、OPS 和统计数据的官方来源。
</li>
</ul>

</div>
  

---

### **Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1258905764936814613)** (15 messages🔥): 

> - `Minecraft Underground Survival` (Minecraft 地下生存)
> - `Average Cost Research` (平均成本研究)
> - `Relational Table Considerations` (关系表注意事项)
> - `Current Redemption Programs` (当前兑换计划)
> - `Next iPad Mini Release` (下一代 iPad Mini 发布)


- **Minecraft Underground Survival Guide**: 几位用户讨论了一份详细的 [Minecraft 地下生存指南](https://www.perplexity.ai/page/minecraft-underground-survival-hj7PsuozQ32xoJKudQqm8g)，探索了在游戏的地下环境中生存的策略。
- **Average Cost Research Findings**: 一位成员分享了他们关于 [平均成本研究](https://www.perplexity.ai/search/what-is-the-average-cost-of-a-JM0.Us5FQO6c6cozzHTvOw) 的深入见解，并提到结果“令人大吃一惊”。
- **Setting Up New Google Account Issues**: 一位用户在 [设置新 Google 账号](https://www.perplexity.ai/search/i-m-trying-to-set-up-a-new-goo-iFZlMi1vQ0qcmbkCH.heEQ) 时寻求帮助，表示在过程中遇到了困难。
- **Exploring Neuromorphic Chips**: 成员们深入研究了 [类脑芯片 (Neuromorphic Chips) 的工作原理](https://www.perplexity.ai/page/how-neuromorphic-chips-work-jb7QR.G6TzGswMico3It5g)，这种芯片模拟人脑架构以实现高效处理。
- **Craft CMS Upgrade Guidance**: 一场讨论集中在 [将 Craft CMS 从 3.9.5 版本升级到 5](https://www.perplexity.ai/search/upgrade-craft-cms-3-9-5-to-5-w-D_kJzsmYTfOe3ISXp1GtFw)，涵盖了必要步骤和潜在挑战。



**Link mentioned**: <a href="https://www.youtube.com/embed/CcOK72Jmlno">YouTube</a>: no description found

  

---


### **Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1259136267703029770)** (9 messages🔥): 

> - `Online model performance` (在线模型性能)
> - `API request processing` (API 请求处理)
> - `API vs Perplexity search results` (API 与 Perplexity 搜索结果对比)
> - `Beta access delay` (Beta 访问延迟)
> - `Multi-step search in API` (API 中的多步搜索)


- **New online model shows improved performance**: 据用户分享，**Online model** 表现更好，特别是在处理多部分查询方面。
   - 与之前的版本相比，生成响应时*感觉更加稳健和精确*。
- **Issues around API request processing**: 用户对 **API 访问请求的处理时间** 提出疑问，并好奇是否有加快进度的方法。
   - 关于**通常的处理时间**或**加急请求**，目前没有提供明确的答案。
- **Disparity between API results and Perplexity search**: 用户担心 **API 结果** 与 **Perplexity.ai 搜索页面** 的结果不匹配。
   - 一位成员澄清说，API 结果与**非 Pro 搜索结果**是一致的。
- **Long wait for Beta access**: 一位用户对等待 **Beta 访问权限** 近一个月且未收到任何回复表示不满。
   - 目前没有提供解决 Beta 访问延迟的*更新或时间表*。
- **Multi-step search in Perplexity API**: 一位用户询问 Perplexity API 中是否提供 **多步搜索功能 (multi-step search feature)**。
   - 目前没有具体信息；该成员被引导至一个 **Discord 频道** 链接以获取更多潜在细节。


  

---



### **LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1258867467028271195)** (249 messages🔥🔥): 

> - `Hermes 2.5`
> - `Mistral struggles` (Mistral 的困境)
> - `Model Merging` (模型合并)
> - `Open Empathic`
> - `IPEX-LLM integration` (IPEX-LLM 集成)

- **IPEX-LLM 集成虽然繁琐但可行**：在遵循 [IPEX-LLM 快速入门指南](https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md) 后，用户报告了在将 IPEX-LLM 与 llama.cpp 集成时的不同成功情况。
   - 一些成员由于指南过时而面临困难，而另一些成员则报告通过遵循官方说明成功完成了构建。
- **MacBook M3 可处理大型模型**：用户讨论了 M2 和 M3 MacBook 的性能，特别称赞了配备 **128GB RAM** 的 M3 MacBook Pro 在处理 WizardLM-2-8x22B 等大型模型方面的表现。
   - 尽管旧型号存在内存限制问题，但 M3 被视为大型模型推理的稳健解决方案。
- **WizardLM-2-8x22B 性能测试**：由于之前有性能不佳的说法，一位成员寻求帮助，希望在具有 **32k context** 的 M2 MacBook 上测试 **WizardLM-2-8x22B-Q4_K_M** 的性能。
   - 由于内存限制，该模型加载失败，计划使用 M3 MacBook 重新尝试。
- **InternLM 模型及其视觉能力**：成员们询问了关于使用 **InternLM 模型** 进行视觉任务的情况，并指出了在 LM Studio 中的兼容性问题。
   - 虽然某些模型在 Python 中运行良好，但用户报告称在 LM Studio 中进行视觉处理需要特定的配置和适配器。
- **GLM4 模型在 llama.cpp 中的支持**：一位用户询问 LM Studio 是否会支持 **GLM4 模型**，因为 **llama.cpp** 最近增加了对它们的支持，希望能高效运行 CodeGeex 模型。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="http://host.docker.internal:PORT_NUMBER.">未找到标题</a>：未找到描述</li><li><a href="http://host.docker.internal:11434.">未找到标题</a>：未找到描述</li><li><a href="https://github.com">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿名开发者共同塑造软件未来的地方。为开源社区做贡献，管理您的 Git 仓库，像专家一样审查代码，跟踪错误和功能...</li><li><a href="https://huggingface.co/internlm/internlm-xcomposer2d5-7b">internlm/internlm-xcomposer2d5-7b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-gguf">internlm/internlm2_5-7b-chat-gguf · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/mradermacher/internlm2_5-7b-GGUF">mradermacher/internlm2_5-7b-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/THUDM/codegeex4-all-9b">THUDM/codegeex4-all-9b · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/inter">inter (Xhark Zhang)</a>：未找到描述</li><li><a href="https://huggingface.co/QuantFactory/internlm2_5-7b-chat-1m-GGUF">QuantFactory/internlm2_5-7b-chat-1m-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/ggerganov/llama.cpp/releases/tag/b3333">Release b3333 · ggerganov/llama.cpp</a>：未找到描述</li><li><a href="https://huggingface.co/bartowski/WizardLM-2-8x22B-GGUF/tree/main/WizardLM-2-8x22B-Q4_K_M.gguf">bartowski/WizardLM-2-8x22B-GGUF at main</a>：未找到描述</li><li><a href="https://github.com/intel-analytics/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_quickstart.md">ipex-llm/docs/mddocs/Quickstart/llama_cpp_quickstart.md at main · intel-analytics/ipex-llm</a>：在 Intel CPU 和 GPU（例如带有 iGPU 的本地 PC、Arc 和 Flex 等独立 GPU）上加速本地 LLM 推理和微调（LLaMA, Mistral, ChatGLM, Qwen, Baichuan, Mixtral, Gemma, Phi 等）。</li><li><a href="https://github.com/lmstudio-ai/lms">GitHub - lmstudio-ai/lms: LM Studio CLI. Written in TypeScript/Node</a>：LM Studio CLI。使用 TypeScript/Node 编写。通过创建账号参与 lmstudio-ai/lms 的开发。</li><li><a href="https://www.youtube.com/watch?v=Y08Nn23o_mY">Intro to RAG for AI (Retrieval Augmented Generation)</a>：这是一个关于检索增强生成 (RAG) 的入门视频。RAG 非常适合赋予 AI 长期记忆和外部知识，降低成本，以及更多...</li><li><a href="https://www.youtube.com/watch?v=pK8u4QfdLx0">&quot;okay, but I want Llama 3 for my specific use case&quot; - Here&#39;s how</a>：如果您想要个性化的 AI 策略来让自己和您的业务面向未来，请加入我的社区：https://www.skool.com/new-society 在 Twitter 上关注我 -...</li><li><a href="https://www.youtube.com/watch?v=zjkBMFhNj_g">[1hr Talk] Intro to Large Language Models</a>：这是一个面向普通观众的 1 小时大型语言模型入门介绍：它是 ChatGPT、Claude 和 Bard 等系统背后的核心技术组件。什么是...
</li>
</ul>

</div>
  

---

### **LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1258862244566011955)** (163 条消息🔥🔥): 

> - `Experiences with Different Model Versions`（不同模型版本的体验）
> - `Model Performance Issues`（模型性能问题）
> - `Model Quantization Discussions`（模型量化讨论）
> - `Fine-tuning and Customization`（微调与定制）
> - `Categorizing Text Prompts`（文本提示词分类）


- **多样化的模型体验与问题**：用户讨论了 Hermes、Mistral 和 Gemma 等各种模型的体验，指出了性能差异和无限循环等问题。
   - 一些用户提到了特定的硬件设置和配置来诊断或提高性能，强调了不同的 Quantization（量化）设置及其影响。
- **Gemma 2 模型面临性能 Bug**：多位用户遇到了 **Gemma 2** 模型的性能问题，包括推理缓慢和数学计算错误。
   - 社区期待在即将到来的更新中解决这些 Bug，并针对 [Gemma 模型架构问题](https://github.com/ggerganov/llama.cpp/pull/8348) 进行了专门讨论。
- **优化性能的量化技术**：对话倾向于高级 Quantization 技术，例如层量化的粒度，以在保持输出质量的同时提高模型性能。
   - 用户分享了 [量化模型](https://huggingface.co/Joseph717171/Models/tree/main) 的链接，并讨论了使用 F32 和 F16 等格式以获得更好结果。
- **文本提示词分类的挑战**：一位用户询问在 LM Studio 中对文本提示词进行分类，但被告知 LLM 在此类任务中并不高效。
   - 有建议指出可以探索用于文本分类的 BERT 模型，但 LM Studio 目前尚不支持此类模型。
- **自定义训练与微调的限制**：一位用户询问在 LM Studio 中使用特定数据集训练模型，但被纠正该平台仅支持 Inference（推理）。
   - 建议使用 Text Embeddings 以及 Hugging Face 等平台进行 Fine-tuning（微调）作为替代方案。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/gokaygokay/Florence-2">Florence 2 - a Hugging Face Space by gokaygokay</a>：未找到描述</li><li><a href="https://huggingface.co/legraphista/glm-4-9b-chat-1m-GGUF">legraphista/glm-4-9b-chat-1m-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/TheDrummer/Smegmma-Deluxe-9B-v1-GGUF">TheDrummer/Smegmma-Deluxe-9B-v1-GGUF · Hugging Face</a>：未找到描述</li><li><a href="https://huggingface.co/internlm/internlm2_5-7b-chat-gguf/tree/main">internlm/internlm2_5-7b-chat-gguf at main</a>：未找到描述</li><li><a href="https://huggingface.co/Joseph717171/Models/tree/main">Joseph717171/Models at main</a>：未找到描述</li><li><a href="https://huggingface.co/docs/transformers/tasks/sequence_classification">Text classification</a>：未找到描述</li><li><a href="https://github.com/THUDM/CodeGeeX4">GitHub - THUDM/CodeGeeX4: CodeGeeX4-ALL-9B, a versatile model for all AI software development scenarios, including code completion, code interpreter, web search, function calling, repository-level Q&amp;A and much more.</a>：CodeGeeX4-ALL-9B，一款适用于所有 AI 软件开发场景的多功能模型，包括代码补全、代码解释器、网络搜索、函数调用、仓库级问答等。</li><li><a href="https://github.com/yfzhang114/SliME?tab=readme-ov-file">GitHub - yfzhang114/SliME: ✨✨Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models</a>：✨✨Beyond LLaVA-HD: Diving into High-Resolution Large Multimodal Models - yfzhang114/SliME</li><li><a href="https://github.com/ggerganov/llama.cpp/pull/8348">llama : fix n_rot default by ggerganov · Pull Request #8348 · ggerganov/llama.cpp</a>：修复 #8246 #8251。确定默认 n_rot 参数的逻辑未考虑到 LLM_KV_ATTENTION_KEY_LENGTH 的覆盖。这导致了 Gemma2 模型的无效上下文偏移：# gemma-2-27...
</li>
</ul>

</div>
  

---


### **LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1259164948651704342)** (4 条消息): 

> - `x64bit installer for LM Studio`（LM Studio 的 x64 位安装程序）
> - `Features of LM Studio`（LM Studio 的功能）
> - `Community feedback on LM Studio`（社区对 LM Studio 的反馈）
> - `Vision-enabled models`（支持视觉的模型）
> - `Tool calling and model capabilities`（工具调用与模型能力）


- **LM Studio 安装程序关于 x64 位的困惑**：一名成员质疑 LM Studio 缺少 64 位安装程序，错误地认为 x86 不是 64 位。
- **社区对 LM Studio 的反馈**：一名成员分享了他们使用 LM Studio 的体验，称赞其对初学者友好，但也表示需要更多高级功能。
- **对 LM Studio 高级功能的呼吁**：同一名成员敦促 LM Studio 发布 Tool calling（工具调用）、针对文件上传的 RAG 以及图像生成能力的 Beta 功能，以跟上竞争对手的步伐。


  

---

### **LM Studio ▷ #[📝-prompts-discussion-chat](https://discord.com/channels/1110598183144399058/1120489168687087708/1259508784854995015)** (1 messages): 

> - `RAG applications`
> - `Optimal placement of retrieved context`
> - `System message vs final user message` 


- **RAG 应用中的最佳上下文放置**：一场关于在 **RAG applications** 中从向量数据库检索到的上下文应放置在何处的讨论——是放在 system message 还是 final user message 中。
   - 成员们正在权衡上下文放置策略的优劣，以提高系统响应的准确性和相关性。
- **System vs Final User Message 之争**：辩论集中在将上下文嵌入 **system message** 还是 **final user message** 能产生更好的性能。
   - 参与者正在考虑各种使用场景以及对用户体验的潜在影响。


  

---


### **LM Studio ▷ #[⚙-configs-discussion](https://discord.com/channels/1110598183144399058/1136793122941190258/1259277897831682058)** (3 messages): 

> - `internllm2_5 config`
> - `models for understanding PDFs`
> - `using LMStudio with Shell GPT` 


- **寻求 internllm2_5 的配置**：一位成员询问是否有人可以分享 **internllm2_5** 的优质配置。
- **寻找理解 PDF 的模型**：另一位成员询问适合理解 **PDFs** 的模型。
- **在 Shell GPT 中使用 LMStudio 的帮助**：一位成员寻求关于如何配置 **LMStudio**（而非 **Ollama**）与 [Shell GPT](https://github.com/ther1d/shell_gpt?tab=readme-ov-file) 配合使用以实现命令行 AI 生产力的帮助。
   - 他们尝试更改 `API_BASE_URL` 和 `DEFAULT_MODEL` 但没有成功，并请求进一步协助。



**提到的链接**：<a href="https://github.com/ther1d/shell_gpt?tab=readme-ov-file">GitHub - TheR1D/shell_gpt: A command-line productivity tool powered by AI large language models like GPT-4, will help you accomplish your tasks faster and more efficiently.</a>：一个由 GPT-4 等 AI 大语言模型驱动的命令行生产力工具，将帮助你更快、更高效地完成任务。 - TheR1D/shell_gpt

  

---


### **LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1258864698267406450)** (44 messages🔥): 

> - `Snapdragon Elite X Machines`
> - `RAM upgrades and costs`
> - `Unified Memory in Windows and Mac`
> - `External GPUs`
> - `Feasibility of using Quad Xeon Servers for AI` 


- **等待 Snapdragon Elite X 的 NPU 支持**：一位用户对 **Snapdragon Elite X 机器**中 **16 GB 和 32 GB** RAM 之间的价格差异表示担忧，并考虑在购买前等待 **NPU 支持**。
   - 另一位用户建议考虑使用 **M3 Max MacBook Pro**，强调其对开发和 LLM 任务的适用性。
- **Windows 中的统一内存转型**：**用户讨论了** **Windows 转向统一内存 (Unified Memory)** 的潜在好处，并与 Apple 的统一内存系统进行了比较。
   - 他们推测了即将推出的技术，提到了 **Lunar Lake** 以及目前可能支持该技术的 Qualcomm Snapdragon X 笔记本电脑。
- **用于推理的外接显卡**：一位成员询问是否可以在笔记本电脑上使用 **external GPU** 进行 LLM 推理。
   - 确认在正确的 GPU 配置下是可行的，但 **带宽瓶颈 (bandwidth bottlenecks)** 可能会是一个问题。
- **使用四路 Xeon 服务器运行 AI 的可行性**：一位用户质疑在配备 **256 GB DDR3 RAM** 的 **quad Xeon X7560** 服务器上运行 LLM 的可行性。
   - 成员们指出，由于缺乏 **AVX2** 支持以及 DDR3 RAM 的限制，这对于 LLM 任务来说是不切实际的。


  

---


### **LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1259841910793568267)** (2 messages): 

> - `Suspicious Activity in Chat`
> - `Discord Update Delays` 


- **可疑用户已快速处理**：一位成员指出 **<@302816205217988609> 看起来很可疑**。
   - 另一位成员确认该问题已处理，只是在等待 Discord 更新：*“已处理，Discord 只是需要时间来更新。”*
- **Discord 更新延迟**：Discord 在更新与可疑用户相关的更改时存在延迟。
   - 一位成员安慰说问题已经解决，但用户可能仍会看到过时的信息。


  

---

### **LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1259796930737999984)** (1 messages): 

> - `Cost Warning Suppression`
> - `LM-Studio Configuration`
> - `Messaging Bug` 


- ****抑制成本警告**：已实现日志增强**：一位用户分享了一段代码片段，通过添加自定义过滤器来消除特定消息，从而[抑制来自 **autogen.oai.client** 日志记录器的成本警告](https://example.link)。
- ****新的 LM-Studio 配置**：集成 `gemma-2b-it-GGUF` 模型**：分享了新的 **LM-Studio 配置**，其特点是使用了 `gemma-2b-it-GGUF` 模型，未启用缓存，并设置了本地服务器地址为 **http://localhost:1234/v1**。
- ****一月份的消息 Bug**：消息顺序的已知问题**：一位用户提到了之前一月份的一个 Bug，涉及以特定顺序发送 system、assistant 和 user 消息时出现的问题。


  

---


### **LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1259667570621939752)** (2 messages): 

> - `LM Studio`
> - `Generation Speed`
> - `Fedora 40 Kinoite`
> - `7900XTX` 


- **LM Studio 中创纪录的生成速度**：一位用户确认 **LM Studio** 的最新更新运行符合预期，并强调了生成速度的**疯狂**增长。
- **使用 7900XTX 进行 Fedora 40 Kinoite 测试**：一位用户提到了他们在运行 **7900XTX** GPU 的 **Fedora 40 Kinoite** 上的配置。


  

---


### **LM Studio ▷ #[🛠-dev-chat](https://discord.com/channels/1110598183144399058/1234988891153629205/1259668172928188497)** (3 messages): 

> - `Removing CPU requirement for app`
> - `Forcing the model to load into RAM`
> - `GPU offload configuration` 


- **移除打开应用的 CPU 限制要求**：一位用户询问如何移除打开应用时的**最低 CPU 要求**。
- **强制模型加载到 RAM**：一位用户询问如何强制将模型加载到 **RAM** 而不是 **VRAM**，因为在同时运行 **Stable Diffusion** 时出现了减速问题。
   - 另一位用户建议在侧边配置菜单中*禁用 GPU offload* 作为解决方案。


  

---



### **OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1258879556006449252)** (325 messages🔥🔥): 

> - `Hermes 2`
> - `Mistral struggles`
> - `Model Merging`
> - `Open Empathic`
> - `Cloudflare blocking AI bots` 


- **关于当前 AI 局限性与演进的讨论**：成员们正在讨论 **Hermes 2** 及其改进版本 **Hermes 2.5** 在基准测试中的重要性，同时也对 **Mistral** 等模型在没有进一步预训练的情况下难以扩展超过 8k 上下文表示担忧。
   - 有人建议将*合并策略（Merging tactics）*作为改进 AI 模型的潜在手段，而其他人则指出了像 **Claude 3.5** 这种 AI 的安全限制和上下文限制。
- **Cloudflare 的 AI 爬虫机器人拦截功能**：有人对 **Cloudflare** 推出允许网站拦截 AI 爬虫机器人的功能表示担忧，这可能会影响 AI 的数据收集。
   - 然而，一些人认为只有那些积极尝试拦截 AI 的人才会使用它，大多数网站不会使用。
- **关于 AGI 和 ASI 潜力的辩论**：社区正在辩论 **通用人工智能 (AGI)** 和 **人工超智能 (ASI)** 的潜力和时间表，并与 **Nvidia 的 Omniverse** 进行了比较。
   - 成员们正在权衡 AGI 的实用性和紧迫性，引用了 **Nvidia 的进展**，并讨论了像 **Safe Superintelligence Inc.** 这样的公司是否能比 **OpenAI** 或 **Google** 等老牌企业更早实现 ASI。
- **自动化的未来以及 AI 在劳动力中的角色**：参与者讨论了 AI 对工厂自动化的影响，提到了全自动化 **BMW 工厂** 的例子以及 **Tesla** 大规模生产机器人的计划。
   - 对于这些进步将如何影响人类劳动、创建“硬盘大脑”的效率以及人类与 AI 协作的平衡，存在各种担忧和见解。
- **AI 的社区与实际应用**：有人建议了实际应用方案，例如利用 **OpenAI GPT-4o 的视觉能力** 进行实时物体检测，而为了提高效率，推荐使用 **计算机视觉模型 (YOLO)** 等替代方案。
   - 成员们分享了组织社区活动和见面会以讨论这些进展的想法，并参与 **OpenAI 社区** 等论坛以进行更好的协调。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://forum.openai.com">OpenAI 论坛</a>：未找到描述</li><li><a href="https://community.openai.com">OpenAI 开发者论坛</a>：提问并获取使用 OpenAI 平台进行构建的帮助
</li>
</ul>

</div>
  

---

### **OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1258918233990500443)** (13 messages🔥): 

> - `GPT-4o vs GPT-4`
> - `Verification issues`
> - `Custom GPTs + Zapier integration` 


- **GPT-4o 被认为更快但不一定更好**：社区成员讨论了 **GPT-4o** 是否是 **GPT-4** 的更好替代品，因为它响应更快，但有人认为它牺牲了质量。
- **反复出现的验证提示问题**：多位用户报告在访问 ChatGPT 时遇到持续的 *'Verify your Human'*（验证你是人类）弹窗，这造成了极大的困扰。
- **Custom GPTs 与 Zapier 集成的挑战**：一位用户询问了使用 **custom GPTs** 结合 **Zapier** 自动化任务的经验，并指出 Zapier 的不可靠性是一个挑战。


  

---


### **OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1259848003699871797)** (3 messages): 

> - `Content Creation Tips`
> - `Increasing Engagement`
> - `Platform Optimization`
> - `Content Calendar Structure`
> - `Tracking Metrics for Success` 


- **制作互动内容的最佳 Prompt**：一位成员询问对于想要创作互动内容并增加粉丝的内容创作者来说，**哪些 Prompt 效果最好**。
   - *另一位用户回复了一个详细的请求*，向 ChatGPT 索取内容创意、互动技巧、平台特定建议、内容日历建议以及追踪成功的关键指标。
- **互动内容创作策略**：用户向 ChatGPT 提供了一个全面的请求，询问 **5-10 个新鲜的内容创意**、提升互动率的策略、平台特定建议、内容日历结构以及监控指标。
   - 该详细请求概述了关键领域，如针对 **Instagram, YouTube, 和 TikTok** 优化内容，并根据粉丝增长和互动情况追踪成功。


  

---


### **OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1259848003699871797)** (3 messages): 

> - `Content creation tips`
> - `Audience engagement strategies`
> - `Platform optimization advice`
> - `Content calendar structure`
> - `Key metrics for content success` 


- **为内容创作者编写互动 Prompt**：一位成员询问内容创作者创作互动内容并增加粉丝的最佳 Prompt，引发了各种建议和讨论。
   - 一位用户提供了一个详细的 Prompt，要求提供 **内容创意、互动技巧、平台特定建议、内容日历结构以及追踪成功的关键指标**。
- **内容创作策略的详细 Prompt**：建议的详细 Prompt 包括请求 **根据该领域的趋势话题提供 5-10 个新鲜内容创意**、提升互动的策略以及平台特定的优化建议。
   - 它还建议询问一个简单的内容日历结构，以及用于监控内容成功和粉丝增长的关键指标。


  

---

### **Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1258872092280229888)** (167 条消息🔥🔥): 

> - `Qwen 模型被低估`
> - `Martin Shkreli 现身`
> - `SLM 微调实践`
> - `Unsloth Studio Beta UI`
> - `LLM 训练中的 AMD 与 NVIDIA 之争` 


- **尽管工作出色，Qwen 团队仍被低估**：多位成员称赞了 **Qwen Team** 的努力，表达了如“*Qwen 团队被严重低估了*”之类的观点。
   - 一段新的 [Qwen 训练视频](https://link.to.video) 被认为非常出色。
- **Martin Shkreli 现身聊天室**：一名成员指出 Martin Shkreli 出现在聊天中，引发了笑声，并确认他参与了相关的 Discord 频道。
- **微调实践讨论**：关于 **finetuning** 实践的讨论强调了高质量数据集至关重要，重点在于质量而非数量：“*微调 80-90% 的时间和成本都花在数据集上*”。
- **Unsloth Studio Beta UI**：Unsloth 的 [Studio Beta UI](https://docs.unsloth.ai) 已完成 80%，它将 Colab 上的 **finetuning** 简化到只需 1-5 次点击。
   - 讨论了未来与 **Gradio UI** 集成的可能性：“*这会是一个极好的主意！！*”
- **LLM 训练中的 AMD 与 NVIDIA 之争**：**AMD GPU** 正在追赶，但由于更好的软件支持和效率，**NVIDIA** 在 LLM 训练方面仍保持领先。
   - “*大多数库不支持 AMD，因此你在使用上会受到很大限制*”。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/unsloth/gemma-2-9b">unsloth/gemma-2-9b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>: 未找到描述</li><li><a href="https://docs.unsloth.ai/">Welcome | Unsloth Docs</a>: 初识 Unsloth？从这里开始！</li><li><a href="https://huggingface.co/Replete-AI/Llama3-8B-Instruct-Replete-Adapted">Replete-AI/Llama3-8B-Instruct-Replete-Adapted · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2">blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://colab.research.google.com/drive/1WZDi7APtQ9VsvOrQSSC5DDtxq159j8iZ?usp=sharing">Google Colab</a>: 未找到描述</li><li><a href="https://colab.research.google.com/notebooks/forms.ipynb)">Google Colab</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[off-topic](https://discord.com/channels/1179035537009545276/1179039861576056922/1259047540209225769)** (18 条消息🔥): 

> - `Kaggle 磁盘空间限制`
> - `Anthropic 的控制（steering）方法`
> - `模型剪枝`
> - `AI 研究社区`
> - `LivePortrait` 


- **Kaggle 磁盘空间崩溃**：一名成员突破了 **Kaggle** 限制，在超过 **100GB** 后会话崩溃。
   - 他们在崩溃前成功将一个**重要的 checkpoint** 保存到了 [Weights & Biases](https://wandb.ai)。
- **Anthropic 控制方法咨询**：讨论了 **Anthropic's steering method**，一名成员请求提供讨论该方法的 Twitter 帖子链接。
   - 另一人确认读过关于可解释 AI（Explainable AI）是未来的文章，但因未保存而无法提供链接。
- **剪枝模型寻求帮助**：一名成员寻求帮助，希望从 ⌘R 35b 模型中**剪掉 15-20B 参数**，用于他们自己的小型模型家族项目。
   - 他们联系了另一位成员以寻求指导。
- **社区 AI 研究焦点**：一名成员正在建立一个**专注于 AI 研究的社区**，并邀请对理论工作感兴趣的人加入。
   - 该社区旨在开展重大项目，且不需要编程经验。
- **LivePortrait 令人印象深刻**：一名成员对 **LivePortrait** 表示赞赏。



**提及的链接**: <a href="https://gate-app.com/research">Gate</a>: Gate 平台

  

---

### **Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1258872426947940434)** (120 messages🔥🔥): 

> - `使用 Alpaca 数据集训练 Phi-3`
> - `Llama-3 与 Phi 3.5 mini 的推理速度与效率`
> - `训练后的 GGUF 转换问题`
> - `针对 Gemma 2 27B 的 DPO`
> - `微调模型结合 RAG 方法` 


- **使用 Alpaca 数据集训练 Phi-3**：一位用户在以 Alpaca 格式训练 Phi-3 时遇到了 `xFormers wasn't built with CUDA support` 错误，并被建议更新其使用的 `xformers` 软件包版本。
- **Llama-3 与 Phi 3.5 mini 的推理速度与效率**：一位用户指出 Llama-3 8B 的速度与 Phi 3.5 mini 相当，两者运行速度均为 280 tokens/second，且 Llama-3 使用的 VRAM 略少。
   - 另一位用户提到 Tensorrt-llm 是目前 GPU 推理速度的业界领先（state of the art）方案。
- **训练后的 GGUF 转换问题**：一位用户在尝试将训练好的模型转换为 GGUF 格式时遇到了 `FileNotFoundError`，具体表现为缺失 `tokenizer.model` 文件。
   - 建议使用 `FastLanguageModel.from_pretrained(..., force_download = True)` 重新下载模型，因为在之前的更新中 `tokenizer.model` 可能最初缺失。
- **针对 Gemma 2 27B 的 DPO**：在使用 DPO 处理 Gemma 2 27B 时，由于 Llama 模型前向操作（forward operations）期间的自动微分问题出现了错误。
   - 该问题在更新 Unsloth 后得到解决，但指出该过程现在会消耗显著更多的显存。
- **微调模型结合 RAG 方法**：一位用户询问关于将微调后的模型与 RAG（Retrieval-Augmented Generation）结合使用的问题，并得到了这是一个可行方案的肯定回复。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/ZLbVdvOoTKM?si=6v4YyWtROCGZcTVX">How to Build an LLM from Scratch | An Overview</a>：👉 需要 AI 方面的帮助？联系方式：https://shawhintalebi.com/ 这是关于在实践中使用大语言模型（LLMs）系列视频的第 6 部分。在这里，我回顾了...</li><li><a href="https://github.com/Unstructured-IO/unstructured">GitHub - Unstructured-IO/unstructured: Open source libraries and APIs to build custom preprocessing pipelines for labeling, training, or production machine learning pipelines.</a>：用于构建自定义预处理流水线的开源库和 API，适用于标注、训练或生产级机器学习流水线。 - GitHub - Unstructured-IO/unstructured...</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sharing">Google Colab</a>：未找到描述</li><li><a href="http://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth</li><li><a href="https://huggingface.co/unsloth">unsloth (Unsloth AI)</a>：未找到描述</li><li><a href="https://colab.research.google.com/drive/135ced7oHytdxu3N2DNe1Z0kqjyYIkDXp?usp=sha">Google Colab</a>：未找到描述</li><li><a href="https://github.com/unslothai/unsloth#-finetune-for-free">GitHub - unslothai/unsloth: Finetune Llama 3, Mistral, Phi &amp; Gemma LLMs 2-5x faster with 80% less memory</a>：微调 Llama 3, Mistral, Phi &amp; Gemma LLM，速度提升 2-5 倍，显存占用减少 80% - unslothai/unsloth
</li>
</ul>

</div>
  

---


### **Unsloth AI (Daniel Han) ▷ #[research](https://discord.com/channels/1179035537009545276/1257011997250424842/1259592130817429574)** (13 messages🔥): 

> - `在论坛寻求帮助`
> - `对招聘频道的的需求` 


- **别问能不能问，直接问！**：一位用户分享了一个[链接](https://dontasktoask.com/)，解释了为什么在提出问题前先询问是否有专家在场是不礼貌且低效的。
   - 核心信息是：*“不要浪费时间；直接提出你的问题，”* 这引起了一些成员的共鸣。
- **研究频道被误用引发了创建招聘频道的建议**：成员们注意到将研究频道变成求职或招聘论坛是不合适的，一位成员明确要求保持频道的主题相关性。
   - 针对寻求 AI 工作的不相关帖子，有人建议创建一个专门的招聘频道。



**提到的链接**：<a href="https://dontasktoask.com/">Don't ask to ask, just ask</a>：未找到描述

  

---



### **Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1258862345225113701)** (26 messages🔥): 

> - `LLM 编程效率`
> - `Bug 修复文档`
> - `Inspect AI 框架`
> - `Dario Amodei 的见解`
> - `Schedule-Free 优化器`

- **使用 LLM 进行高效编码**：一位用户讨论了如何通过重构代码以使用 **LLM-style APIs** 来简化复杂的编码任务，并强调了人类在沟通和集成系统中的作用。
   - 他们认为，将 **APIs 粘合在一起** 可以将耗时的任务转变为简单的 zeroshot LLM prompts，从长远来看可以节省精力。
- **深入探讨 Bug 修复文档**：一位用户分享了一个针对处理字符串别名和声明类型的[详细 bug 修复](https://github.com/go-go-golems/glazed/pull/418/files)，并增加了大量的文档和单元测试。
   - 他们强调，虽然修复花费了 2 小时，但生成的文档有助于未来的增强，并使 LLMs 更容易生成解决方案。
- **英国政府的 Inspect AI 框架**：一位用户对尝试新的 [Inspect AI framework](https://github.com/UKGovernmentBEIS/inspect_ai) 感到兴奋，该框架用于评估大语言模型。
- **Dario Amodei 关于经济影响的见解**：Anthropic 的 CEO Dario Amodei 在[最近的播客](https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880)中讨论了计算成本（占支出的 80%）和可扩展模型。
   - 他还提到了自己过去和现在玩 **Final Fantasy** 的经历，为对话增添了个人色彩。
- **Schedule-Free Optimizers 的创新**：一位研究人员报告了 **schedule-free optimizers** 的显著成果，该优化器简化了超参数调优，且开箱即用表现良好（[详情](https://x.com/Yuchenj_UW/status/1809622351543484563?s=46)）。
   - 该方法允许在没有预定义停止点的情况下进行持续学习，展示了在 AI 模型训练中广泛采用的潜力。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/OpenRouterAI/status/1810001066240413908">来自 OpenRouter (@OpenRouterAI) 的推文</a>：宣布全新的模型市场 UI ✨ 探索 180 个活跃的语言模型，每周处理 740 亿个 tokens 👇</li><li><a href="https://youtu.be/xm6jNMSFT7g?si=BnYoL-E1QXGTw23P&t=3880">Dario Amodei - Anthropic CEO | 播客 | In Good Company | Norges Bank Investment Management</a>：Anthropic CEO Dario Amodei：Claude、新模型、AI 安全与经济影响。下一代 AI 模型会有多大、多强？Anthropic 的 CEO...</li><li><a href="https://x.com/yuchenj_uw/status/1809622351543484563?s=46">来自 Yuchen Jin (@Yuchenj_UW) 的推文</a>：我在 @karpathy 的 nanoGPT 上使用 @aaron_defazio 的 Schedule-Free 优化器训练了 GPT-2 (124M)：- 设置：AdamW，学习率=0.0018（同 https://x.com/Yuchenj_UW/status/1795850420503...</li><li><a href="https://x.com/firstadopter/status/1809633896436269347?s=46">来自 tae kim (@firstadopter) 的推文</a>：Anthropic CEO Dario Amodei 在播客中表示，计算资源占其支出的 80% 以上。600 名员工的薪资支出要小得多</li><li><a href="https://x.com/norabelrose/status/1810342367972495590?s=46">来自 Nora Belrose (@norabelrose) 的推文</a>：@AiEleuther 可解释性团队正在为 Llama 3 8B 的每一层发布一套 top-k sparse autoencoders：https://huggingface.co/EleutherAI/sae-llama-3-8b-32x 我们正在开发一个自动化的...</li><li><a href="https://x.com/alexalbert__/status/1810376544734556540">来自 Alex Albert (@alexalbert__) 的推文</a>：距离参赛截止还有两天！引用 Alex Albert (@alexalbert__)：宣布 2024 年 6 月 Build with Claude 竞赛。我们将发放价值 3 万美元的 Anthropic API 额度。你只需要...</li><li><a href="https://github.com/UKGovernmentBEIS/inspect_ai">GitHub - UKGovernmentBEIS/inspect_ai: Inspect: 一个用于大语言模型评估的框架</a>：Inspect：一个用于大语言模型评估（large language model evaluations）的框架 - UKGovernmentBEIS/inspect_ai</li><li><a href="https://github.com/wesen/glazed/blob/e180e5d59031f20009c461466a2995ff28ee25a7/pkg/doc/topics/13-layers-and-parsed-layers.md">glazed/pkg/doc/topics/13-layers-and-parsed-layers.md at e180e5d59031f20009c461466a2995ff28ee25a7 · wesen/glazed</a>：一个让你的命令行工具轻松输出结构化数据的库。为你的数据锦上添花 - wesen/glazed</li><li><a href="https://aipapersoftheweek.substack.com/p/ai-papers-of-the-week-july-3rd-2024">本周 AI 论文 - 2024 年 7 月 3 日 - Kyutai Moshi, Meta 3D Gen 等</a>：涵盖的论文：AI Agents that Matter、Kyutai Moshi、Meta 3D Gen、Open-TeleVision：具有沉浸式主动视觉反馈的远程操作、PathAlign：用于组织病理学全切片图像的视觉语言模型...</li><li><a href="https://github.com/go-go-golems/glazed/pull/418/files">:ambulance: :umbrella: :books: 由 wesen 提交的 Pull Request #418：处理 layers/parameters 的字符串别名和字符串声明类型</a>：这增加了处理 glazed 的 parameters/layers/reflect 模块中字符串别名和字符串声明的代码。借此机会增加了大量文档和单元测试。</li><li><a href="https://www.nbim.no/en/publications/podcast/dario-amodei-ceo-of-anthropic-claude-new-models-ai-safety-and-economic-impact/">Anthropic CEO Dario Amodei：Claude、新模型、AI 安全与经济影响 | Norges Bank Investment Management</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1258895771055427635)** (5 条消息): 

> - `播客的 HN 帖子`
> - `Fortnite 的新游戏模式`
> - `职场沟通问题`
> - `HN 上的点赞与互动` 


- **在 Hacker News 上分享的播客剧集**：[现已登上 HN！](https://news.ycombinator.com/newest) 一位用户在 Hacker News 上分享了最近一期播客的链接，希望能获得关注。
- **Fortnite 文章的互动**：围绕一篇关于 **Fortnite 移除联动内容**以重拾乐趣的 [Polygon 文章](https://www.polygon.com/gaming/24185789/fortnite-reload-new-game-mode) 展开了讨论。
   - 该文章获得了初步关注，获得了 **1 个点赞**，并由名为 PaulHoule 的用户分享。
- **处理职场沟通问题**：HN 上另一个有趣的话题是关于[处理同事的沟通问题](https://www.nytimes.com/2024/07/07/business/work-friend-anna-holmes.html)，由 *jaredwiener 分享*。
- **HN 社区互动**：一位用户通过点赞在 HN 上分享的播客剧集来表达支持，鼓励持续参与。



**提及的链接**：<a href="https://news.ycombinator.com/newest">最新链接 | Hacker News</a>：未找到描述

  

---

### **Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1258874839163207785)** (243 条消息🔥🔥): 

> - `AI in Action`
> - `AI Engineer World Fair`
> - `LlamaFile vs. Ollama`
> - `Model Merging`
> - `Wearables and Privacy` 


- **AI Engineer World Fair 洞察**：AI Engineer World Fair 带来了许多引人注目的演讲，包括 Justine Tunney 的主题演讲、广受好评的 AI 领导力研讨会，以及关于 LLM 和 Model Merging 的有趣讨论。
   - 一位成员指出，尽管存在一些物流问题，但会议反响良好，涵盖了 **AI-generated music** 和 **Tool Use with Open-Source LLMs** 等主题，会议内容丰富且充满活力。
- **LlamaFile vs. Ollama 之争**：成员们讨论了 LlamaFile 和 Ollama 之间的区别，LlamaFile 专注于 **portability and optimization**（便携性与优化），而 Ollama 则侧重于 **compatibility with a large amount of models**（对大量模型的兼容性）。
   - 一些成员表示希望有一个适配器能结合两者的优势，并建议 **Ollama 可能会作为一个 Llama.cpp wrapper** 运行。
- **Model Merging 技术探索**：Model Merging 是一个热门话题，成员们分享了 [mergekit GitHub](https://github.com/arcee-ai/mergekit) 等资源以及 Merging 策略的最新更新。
   - 讨论了使用深度学习模型来收敛到最佳 Model Merging 策略的可能性，尽管有人指出这种方法在 **intellectually suspect**（学术严谨性上存疑）。
- **可穿戴设备隐私担忧**：针对可穿戴设备以及在活动期间录制非麦克风时段的知情同意问题，人们提出了担忧。
   - 提出了一种涉及桌面集成和可穿戴设备通知功能的解决方案，以确保用户的知情权和同意。
- **未来会议规划**：关于明年 AI Engineer World Fair 的讨论包括将活动延长一天，或增加一个包含活动的休息日。
   - 为了提升参会者体验，有人建议设立专门的 AI 女友应用分会场，并将会议日程游戏化。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.workshopsurvival.com">The Workshop Survival Guide</a>：《The Workshop Survival Guide》：学习如何设计和教授每次都能成功的教育研讨会。现已在 Amazon 上架。</li><li><a href="https://www.ivanleo.com/blog/ai-conf">AI Engineering World Fair</a>：未找到描述</li><li><a href="https://aie.compasswearable.com/events">AI Engineers World Fair Recaps - Powered by Compass</a>：通过实时转录和 AI 生成的摘要体验规模最大的技术型 AI 大会。</li><li><a href="https://x.com/latentspacepod/status/1805836033445216644">Tweet from Latent.Space (@latentspacepod)</a>：@aiDotEngineer 盛况空前！</li><li><a href="https://x.com/philip_kiely/status/1808589566921879702">Tweet from Philip Kiely (@philip_kiely)</a>：以下是我在 @aiDotEngineer World's Fair 充满活力的三天中总结出的 3 个主题：1. 开源正在缩小差距 2. 推理（Inference）无处不在 3. Evals（评估）就是一切。详情：</li><li><a href="https://x.com/RickLamers/status/1808705188024439187">Tweet from Rick Lamers (@RickLamers)</a>：模型合并（Model merging）太疯狂了，看看这个家谱图 :0</li><li><a href="https://github.com/arcee-ai/mergekit">GitHub - arcee-ai/mergekit: Tools for merging pretrained large language models.</a>：用于合并预训练大语言模型（LLM）的工具。- arcee-ai/mergekit</li><li><a href="https://docs.google.com/document/d/1TLXkcaNX6cvpiqqyo952_K2a7XTF064R44v3WL9CSbE/edit?usp=sharing">AI Engineering Worlds Fair</a>：AI Engineering Worlds Fair Thomas Dohmke 以人为本的方法 —— “co-pilot”。Copilot 帮助开发者保持在软件开发的 Flow 状态中。使信息获取民主化 —— 入职引导。Agent —— AI 洗碗机（侧面...</li><li><a href="https://docs.google.com/presentation/d/1A_yLcD6Sy1Nr_v2YesOzvtcg5yAmmrfPR2bU4dyxTzw/edit?usp=sharing">AI in action - 2024-07-05</a>：AI in action AI Engineers World Fair 回顾 2024-07-05</li><li><a href="https://codingwithintelligence.com/p/ai-engineer-world-fair-in-sf">AI Engineer World Fair in SF</a>：Coding with Intelligence 第 26 周</li><li><a href="https://x.com/intertwineai/status/1807060271828975632">Tweet from Bryan Young (@intertwineai)</a>：@aiDotEngineer 第三天回顾与总结！1/12：#AIEWF 2024 的第三天已经结束，显然我们才刚刚触及 AI 潜力的皮毛，并正在定义什么是 @aiDotEngineer。这里是...</li><li><a href="https://x.com/intertwineai/status/1806270266965889289">Tweet from Bryan Young (@intertwineai)</a>：@aiDotEngineer 第二天回顾！1/14。第二天以 @YoungPhlo_ 关于 AI 生成音乐的及时分享开始。我们一起制作了一些超酷的节奏。尽管最近 @RIAA 针对... 的诉讼...</li><li><a href="https://x.com/intertwineai/status/1805867608593645916">Tweet from Bryan Young (@intertwineai)</a>：1/5：@aiDotEngineer 的第一天和我预想的一样精彩！#AIEWF 当日的快速回顾：
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1258900955001262250)** (10 条消息🔥): 

> - `CUDA Certification vs GitHub Repos`
> - `NVIDIA Deep Learning Institute`
> - `Peak FLOPS 对比`
> - `教育支出策略` 


- **公开的 GitHub Repos 胜过 CUDA Certification**：一位用户提出了在招聘时更看重 CUDA 认证课程还是 CUDA kernel 的 GitHub 链接的问题，引发了关于公开作品与证书价值的辩论。
   - *as_ai* 表示：“公开的已证明作品总是比一份不能说明全部情况的证书更有价值。”
- **NVIDIA Deep Learning Institute 资源**：一位用户推荐了 [NVIDIA Deep Learning Institute](https://www.nvidia.com/en-us/training/) 的各种教育资源，并引用了在他们大学举办的课程的个人经验。
   - 该学院提供涵盖 AI、加速计算等领域的自学和直播培训项目——非常适合使用公司的学习预算。
- **注意差距：对比 GPU Peak FLOPS**：一位用户分享了令人惊讶的性能数据，指出 4090 Ti 的峰值达到 **93 TFLOPS**，而 A100 的单精度峰值仅为 19.5 TFLOPS。
   - *eriks.0595* 解释说，对比 Ampere 和 Ada 架构可以看出差异，正如 [Ada tuning guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) 中所述，Ada 提高了 FP32 吞吐量。
- **用于教育目的的报销策略**：一位用户幽默地建议报销一块 GPU 并声称其用于教育目的。
   - 讨论集中在如何创造性地利用公司的学习预算进行个人提升和技能进阶。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.nvidia.com/cuda/ada-tuning-guide/index.html#improved-fp32-throughput">NVIDIA Ada GPU Architecture Tuning Guide</a>：未找到描述</li><li><a href="https://www.nvidia.com/en-us/training/.">NVIDIA Deep Learning Institute and Training Solutions</a>：我们提供 AI、加速计算和加速数据科学方面的实战培训。
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1259172056692555837)** (10 条消息🔥): 

> - `torch.compile 缺失的手册`
> - `PyTorch tensors 与类型擦除 (type erasure)`
> - `图创建中的灵活性与模板对比`
> - `PyTorch profiler 与 FLOP 估算`
> - `FlopCounterMode vs with_flops` 


- **torch.compile 手册明确了用法**：一位成员分享了 [torch.compile, the missing manual](https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab) 的链接，强调了它的实用性。
- **关于 PyTorch tensors 使用类型擦除的讨论**：一位成员询问了关于为什么 **PyTorch tensors** 广泛使用类型擦除以及相比使用更多模板的优势的文档。
   - 类型擦除简化了 Python 和 C++ 前端之间的处理，并举例说明了模板需要复杂的宏或 if/else 语句所带来的挑战。
- **PyTorch profiler 的 FLOP 估算功能**：一位成员对 **PyTorch profiler** 中的 `with_flops` 参数很感兴趣，该参数可以估算模型消耗的 FLOPs，尽管这方面的文档并不完善。
   - 另一位成员建议使用 `torch.utils.flop_counter.FlopCounterMode` 进行 FLOP 计数，因为 `with_flops` 目前已不再积极开发。



**提到的链接**：<a href="https://docs.google.com/document/d/1y5CRfMLdwEoF1nTk9q8qEu1mgMUuUtvhklPKJ2emLU8/edit#heading=h.ivdr7fmrbeab">torch.compile, the missing manual</a>：torch.compile 缺失的手册。你来到这里是因为你想使用 torch.compile 让你的 PyTorch 模型运行得更快。torch.compile 是一个复杂且相对较新的软件，因此你...

  

---

### **CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1259879862991196253)** (1 条消息): 

> - `编译器爱好者职位空缺`
> - `Thunder 编译器优化项目` 


- **Lightning AI 招聘编译器爱好者**：[Lightning AI](https://boards.greenhouse.io/lightningai/jobs/6045025003) 提供了一个职位空缺，面向热爱编译器并希望与包括 Luca Antiga 在内的知名同事共事的开发者。
- **Thunder 提升 PyTorch 模型性能**：Lightning AI 的 [Thunder 项目](https://github.com/Lightning-AI/lightning-thunder) 承诺通过源码到源码（source-to-source）编译器使 **PyTorch 模型提速高达 40%**。
   - Thunder 支持同时使用不同的硬件执行器，无论是单个 GPU 还是数千个 GPU。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://boards.greenhouse.io/lightningai/jobs/6045025003">Compiler Engineer</a>: 英国英格兰伦敦</li><li><a href="https://github.com/Lightning-AI/lightning-thunder">GitHub - Lightning-AI/lightning-thunder: 让 PyTorch 模型提速高达 40%！Thunder 是一个针对 PyTorch 的源码到源码编译器。它支持同时使用不同的硬件执行器；跨越一个或数千个 GPU。</a>: 让 PyTorch 模型提速高达 40%！Thunder 是一个针对 PyTorch 的源码到源码编译器。它支持同时使用不同的硬件执行器；跨越一个或数千个 GPU。 - Lightning-AI/ligh...
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1258880966047432796)** (20 条消息🔥): 

> - `CUDA 初学者项目`
> - `在 CUDA 中使用 Python 与 C++ 的对比`
> - `开始学习 CUDA`
> - `Tinygrad 与 Teenygrad`
> - `具有 2:4 稀疏模式的 SpMM` 


- **CUDA 初学者项目想法**：一位成员提到他们想启动一个 CUDA 项目，并询问实现 Flash attention 是否合适；他们对建议和合作持开放态度。
   - 其他人提供了诸如研究 teenygrad 的想法，或者由于复杂度原因建议了一些更易于管理的项目。
- **社区推荐初学者使用 Python 学习 CUDA**：一位成员在讨论使用 Python 还是 C++ 来编写带有 CUDA 的深度学习框架，担心复杂性和性能问题。
   - 社区建议从 Python 和 CUDA Python 开始，并引用了 llama.c 或 Karpathy 的仓库作为更易理解的例子。
- **深度学习初学者的学习建议**：几位社区成员建议采用自顶向下和自底向上的方法，在深入编码之前先理解数学基础。
   - 他们强调理解简单神经网络的前向传播和反向传播是必不可少的基础工作。
- **cusparseLT 与 CUTLASS 在 SpMM 上的对比**：一位社区成员询问在具有 2:4 稀疏模式的 SpMM 中，cusparseLT 和 CUTLASS 之间是否存在性能差异。
   - 有人建议 cusparseLT 可能经过了更严格的优化和维护。
- **学习 CUDA 的资源**：一位初学者询问开始学习使用 CUDA 进行 GPU 编程的资源。


  

---

### **CUDA MODE ▷ #[torchao](https://discord.com/channels/1189498204333543425/1205223658021458100/1259004537142775848)** (19 messages🔥): 

> - `2:4 稀疏度结合 int8 量化`
> - `可定制的纯 Python 低比特优化器`
> - `非连续梯度问题`
> - `FP8 Adam 优化`
> - `CI 机器上的回归测试` 


- **2:4 稀疏度现在可以与 int8 量化结合使用**：这一新特性已悄然添加，允许 **2:4 稀疏度**与 **int8 量化**组合，并在 [Python 代码](https://github.com/pytorch/ao)中提供了简单的实现。
- **提供纯 Python 低比特优化器**：[TorchAO](https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim) 现在拥有 **8-bit** 和 **4-bit** 优化器的可定制纯 Python 实现。
- **讨论了非连续梯度问题**：关于在 **torchao** 优化器中使用 `.view()` 还是 `.reshape()` 来处理非连续梯度的议题展开了辩论。
- **FP8 Adam 优化器的实验**：一项将 **FP8 Adam** 中的自定义量化/反量化逻辑替换为硬件指令（需要 **Ada** 或 **Hopper** 架构）的实验显示出良好的前景。
- **CI 机器上的回归测试**：通过在 CI 机器上使用多个 GPU，可以使用特定的 **benchmark script** 代替测试套件，将结果打印到控制台。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/pytorch/ao/pull/486">为仅 int8 权重分量化模型启用 `model.to(device)`，由 jerryzh168 提交 · Pull Request #486 · pytorch/ao</a>：摘要：修复了 int8_wo_quantized_model.to(device) 的一些实现问题。测试计划：python test/quantization/test_quant_api.py -k test_quantized_model_to_device。评审人：订阅者：任务：...</li><li><a href="https://github.com/pytorch/ao/tree/main/torchao/prototype/low_bit_optim">ao/torchao/prototype/low_bit_optim at main · pytorch/ao</a>：为训练和推理创建并集成自定义数据类型、布局和内核 - pytorch/ao</li><li><a href="https://github.com/pytorch/ao/pull/403/files#diff-c9eb698c2226c153d926c4709d378e3349d020f4a18bc59245d60320cc317e5fR585">添加 FSDP QLoRA 测试并撤销失败的 PR，由 weifengpy 提交 · Pull Request #403 · pytorch/ao</a>：修复运行 torchtune QLoRA + FSDP2 时的错误 #380 TypeError: nf4_detach() 缺少 1 个必需的位置参数：'args' torchtune 命令 tune download meta-llama/Llama-2-7b-hf --ou...</li><li><a href="https://github.com/pytorch/ao/pull/403/files">添加 FSDP QLoRA 测试并撤销失败的 PR，由 weifengpy 提交 · Pull Request #403 · pytorch/ao</a>：修复运行 torchtune QLoRA + FSDP2 时的错误 #380 TypeError: nf4_detach() 缺少 1 个必需的位置参数：'args' torchtune 命令 tune download meta-llama/Llama-2-7b-hf --ou...
</li>
</ul>

</div>
  

---

### **CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1258943198395502684)** (13 条消息🔥): 

> - `AliExpress 周年庆促销`
> - `创意像素艺术工具`
> - `初创公司创始人的暑假`
> - `Techsupportgore Subreddit`
> - `潜在的在线诈骗` 


- **AliExpress 周年庆促销引发质疑**：成员们对 AliExpress 一项以 430 美元提供 RTX 4090 并附带批量购买激励的促销活动表示怀疑，称其令人难以置信。
   - 有评论讽刺地建议，买家收到的可能只是一张 RTX 4090 的打印照片，而不是实际产品。
- **初创公司创始人与假期无缘**：一位用户开玩笑说，住在美国却不知道什么是暑假，突显了初创公司世界中持续不断的辛劳。
   - 另一位成员幽默地指出：“初创公司创始人：什么是假期？”，强调了这种持续工作的文化。
- **Techsupportgore Subreddit 抗议 Reddit 的 API 政策**：讨论涉及以令人尴尬的技术支持瞬间闻名的 [Techsupportgore subreddit](https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_miner_rinsing_cards_off_with_a_pressure/)，该版块目前正在抗议 Reddit 的 API 政策。
   - 用户被提醒，该版块不是为了寻求技术支持，而是为了查看和发布糟糕技术实践的照片。
- **Pixel Mirror 将现实变为像素艺术**：设计师 Hakusi Katei 开发的一款名为 Pixel Mirror 的 [新工具](https://www.yankodesign.com/2024/07/04/this-crystal-fragment-turns-everything-you-see-into-8-bit-pixel-art-and-its-fascinating/) 可将现实世界的视图转换为 8-bit 像素艺术，融合了模拟与数字体验。
   - 该产品吸引了早期计算机图形学的怀旧粉丝，通过具有独特分辨率降低特性的晶体创建像素化图像。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://www.aliexpress.com/item/1005006997501364.html?src=google">ORIGINAL NEW Fast selling NVIDIA GeForce RTX 4090 Founders Edition Graphics Card 24GB - AliExpress 1420</a>: 聪明购物，更好生活！Aliexpress.com</li><li><a href="https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_miner_rinsing_cards_off_with_a_pressure/">Reddit - 深入探索一切</a>: 未找到描述</li><li><a href="https://www.yankodesign.com/2024/07/04/this-crystal-fragment-turns-everything-you-see-into-8-bit-pixel-art-and-its-fascinating/">This Crystal Fragment turns everything you see into 8-bit Pixel Art, and it’s FASCINATING - Yanko Design</a>: https://www.youtube.com/watch?v=v4VN2ZZZT9c&amp;feature=youtu.be 不可否认，现代图形分辨率已经达到了难以想象的高度。然而，仍有许多人对……有着情感上的联系。</li><li><a href="https://www.reddit.com/r/techsupportgore/comments/xorwdy/a_crypto_min">Reddit - 深入探索一切</a>: 未找到描述
</li>
</ul>

</div>
  

---


### **CUDA MODE ▷ #[irl-meetup](https://discord.com/channels/1189498204333543425/1218444432588800010/)** (1 条消息): 

fancytrevor: 想知道是否有人有旧金山 (SF) 线下聚会的建议
  

---

### **CUDA MODE ▷ #[llmdotc](https://discord.com/channels/1189498204333543425/1227345713348870156/1258911594310533120)** (179 条消息🔥🔥): 

> - `muP 实验`
> - `FP8 精度`
> - `CUDA Checkpointing`
> - `推理优化`
> - `LLM101n 课程计划` 


- **muP 实验结果喜忧参半**：团队的 muP 实验并未显著超越 baseline，对于 `attn_mult` 等超参数的结果不一，需要进一步探索。
- **FP8 精度探索**：讨论了在某些 matmuls 中使用 FP8 的情况，特别是它对最终层的益处，目前正在持续进行 FP8 使用的 benchmark 和优化工作。
- **对 NVIDIA checkpointing 工具的兴趣**：NVIDIA 新的 [cuda-checkpoint](https://github.com/NVIDIA/cuda-checkpoint) 工具及其与 CRIU 的集成以实现细粒度 checkpointing 引起了成员们的兴趣。
- **通过减小 batch sizes 进行推理优化**：[PR #671](https://github.com/karpathy/llm.c/pull/671) 将推理检查更改为使用最小 B/T 而非最大值，旨在不产生发散的情况下获得更快的性能。
- **LLM101n 课程及开发计划**：讨论了逐步 LLM 开发课程 (LLM101n) 的计划，涵盖从 micrograd、minBPE 等基础构建模块到 FP8 和多模态训练等高级主题。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/EleutherAI/pythia-1.4b-v0">EleutherAI/pythia-1.4b-v0 · Hugging Face</a>：未找到描述</li><li><a href="https://github.com/google-research/t5x/blob/0728d8429041d6c6e75077334e76eb2370c6057b/t5x/losses.py#L25-L57)">t5x/t5x/losses.py at 0728d8429041d6c6e75077334e76eb2370c6057b · google-research/t5x</a>：通过在 GitHub 上创建账户为 google-research/t5x 的开发做出贡献。</li><li><a href="https://github.com/ash-01xor/bpe.c/tree/main">GitHub - ash-01xor/bpe.c: Simple Byte pair Encoding mechanism used for tokenization process . written purely in C</a>：用于分词过程的简单字节对编码（Byte pair Encoding）机制，纯 C 语言编写 - ash-01xor/bpe.c</li><li><a href="https://github.com/karpathy/llm.c/pull/671">Faster inference by changing (B,T) to (1,t) by ademeure · Pull Request #671 · karpathy/llm.c</a>：推理完整性检查目前处理所有的 (B,T)，尽管默认只需要 (1,64)。此 PR 与之前的版本在位级别上完全一致，同时将其减少到 (1,t)，其中 t 是经过舍入的...</li><li><a href="https://github.com/NVIDIA/nccl/issues/1026">half precision reduction accumulation in fp32? · Issue #1026 · NVIDIA/nccl</a>：是否有计划修复 NCCL 以在 fp32 累加的情况下对 BFLOAT16 操作数执行规约？否则，我们无法在没有巨大损失的情况下规约梯度，并且必须使用既昂贵又... 的 fp32 通信</li><li><a href="https://github.com/ademeure/llm.c/blob/fp8_phase1/dev/cuda/advanced_copy_transpose.cu">llm.c/dev/cuda/advanced_copy_transpose.cu at fp8_phase1 · ademeure/llm.c</a>：使用简单、原始的 C/CUDA 进行 LLM 训练。通过在 GitHub 上创建账户为 ademeure/llm.c 的开发做出贡献。</li><li><a href="https://github.com/tinygrad/tinygrad/discussions/1505">Flash attention · tinygrad/tinygrad · Discussion #1505</a>：像 Flash Attention (2) 这样的东西是否由 tinygrad 通过所有的 lazy（表达式模板？）机制自动计算？</li><li><a href="https://x.com/__tinygrad__/status/1802435228570616192">来自 tiny corp (@__tinygrad__) 的推文</a>：我们落后的主要领域是 NVIDIA 的速度，特别是对于 LLM 训练，因为我们没有 flash attention 并且 softmax 表现糟糕。我们领先的主要领域是可移植性。tiny...</li><li><a href="https://developer.nvidia.com/blog/checkpointing-cuda-applications-with-criu/">Checkpointing CUDA Applications with CRIU | NVIDIA Technical Blog</a>：CUDA 的 Checkpoint 和恢复功能通过名为 cuda-checkpoint 的命令行工具公开。该工具可用于透明地对 CUDA 状态进行 checkpoint 和恢复...
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[research-papers](https://discord.com/channels/1053877538025386074/1104063238934626386/1259448514463662100)** (3 messages): 

> - `Critiques in Preference Learning`
> - `Test-Time-Training Layers` 


- **合成批判增强奖励模型**：@Daniella_yz 探讨了在 @Cohere 实习期间使用来自 **large language models** 的合成批判来改进奖励模型，详情见[其预印本](https://arxiv.org/abs/2405.20850)。
   - *除了协助人类评估（例如 CriticGPT），批判还可以直接增强偏好学习 (preference learning)*。
- **新架构取代 RNN 隐藏状态**：@karansdalal 分享了一种新架构，即 **测试时训练层 (TTT layers)**，它将 RNN 的隐藏状态替换为机器学习模型，并通过对输入 Token 进行梯度下降来压缩上下文，如[其预印本](https://arxiv.org/abs/2407.04620)中所述。
   - 这一创新实现了具有表达性内存的**线性复杂度架构**，允许在数百万或数十亿 Token 的上下文中训练 LLM，其实例 TTT-Linear 和 TTT-MLP 均达到或超过了最强的 Transformers 和 Mamba。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://x.com/Daniella_yz/status/1809720946066092097">Daniella Ye (@Daniella_yz) 的推文</a>: 除了用于协助人类评估（例如 CriticGPT），批判能否直接增强偏好学习？在我的 @Cohere 实习期间，我们探索了使用来自 large lang... 的合成批判。</li><li><a href="https://x.com/karansdalal/status/1810338845659131940?s=46">Karan Dalal (@karansdalal) 的推文</a>: 我很高兴分享一个我已经研究了一年多的项目，我相信它将从根本上改变我们处理语言模型的方式。我们设计了一种新架构，它取代了 h...
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1259264486523670663)** (2 messages): 

> - `Nous Magazine`
> - `Cryptoland`
> - `YouTube video`
> - `Fantasy Division` 


- **即将推出的 Nous Magazine 预览**：John0galt 分享了即将推出的 **Nous Magazine** 的前几页。
- **YouTube 视频探索 Cryptoland**：Iron_bound 发布了一个 [名为“Whatever Happened to Cryptoland?”的 YouTube 视频](https://www.youtube.com/watch?v=W9ggP26yH7A)，强调了加密货币世界中不可预见的事件。
   - 他们还分享了指向 [Fantasy Division](https://fantasydivision.online/References) 的链接以及相关的 [Google Docs 文档](https://docs.google.com/docu)。



**提及的链接**：<a href="https://www.youtube.com/watch?v=W9ggP26yH7A">Whatever Happened to Cryptoland?</a>：绝对没有人能预料到这一切⚔️ 你会想看看这个：https://fantasydivision.online/References：https://docs.google.com/docu...

  

---


### **Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1259435421683941388)** (2 messages): 

> - `Dataline by RamiAwar`
> - `LLM Reasoning Capabilities` 


- **使用 Dataline 与你的数据对话**：一个新的 GitHub 项目 [Dataline](https://github.com/RamiAwar/dataline) 提供了跨多个数据库（如 **CSV**、**Postgres**、**MySQL**、**Snowflake** 和 **SQLite**）的 AI 数据分析和可视化功能。
- **通过几何学探索 LLM 推理能力**：arXiv 上的一篇新论文 [The Geometrical Understanding of LLMs](https://arxiv.org/abs/2407.02678) 探讨了 LLM 的推理能力与自注意力图密度之间的联系。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2407.02678">Reasoning in Large Language Models: A Geometric Perspective</a>: 大语言模型 (LLM) 在现实世界应用中的进步关键取决于增强其推理能力。在这项工作中，我们探索了大语言模型的推理能力...</li><li><a href="https://github.com/RamiAwar/dataline">GitHub - RamiAwar/dataline: 与你的数据对话 - 在 CSV, Postgres, MySQL, Snowflake, SQLite 上进行 AI 数据分析和可视化...</a>: 与你的数据对话 - 在 CSV, Postgres, MySQL, Snowflake, SQLite 上进行 AI 数据分析和可视化... - RamiAwar/dataline
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1258872555184324688)** (211 条消息🔥🔥): 

> - `GPT4 基准测试分数`
> - `Temperature 效应`
> - `In-Context Learning 示例`
> - `Prompt Caching 成本`
> - `BitNet 训练` 


- **GPT4 在 Temperature 升高时得分更高**：一位成员报告称，GPT4 在高 Temperature 下的基准测试得分更高，但另一位成员无法在本地模型中复现这些结果。
- **In-Context Learning (ICL) 提升模型性能**：成员们讨论了增加 In-Context Learning 中示例数量的影响，一致认为更多的示例能增强模型性能。
- **BitNet 引起关注但面临训练挑战**：成员们对 BitNet 架构表示出兴趣，一些人希望使用其 1.58-bit 格式训练模型以节省显存。
- **预见生成式视频技术的快速进步**：成员们乐观地认为，在强劲动力和当前发展速度的推动下，生成式视频技术将在未来 1-1.5 年内实现实时生成。
- **获取 Fine-tuning 资源和指导**：参与者分享了 Fine-tuning 资源，并讨论了如何使用 Ada Instruct 和 Nous 的 Genstruct 7B 等模型从原始文档创建多样化、高质量的数据。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/blog/fellowship">宣布 Hugging Face Fellowship 计划</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/OLMo-Bitnet-1B">NousResearch/OLMo-Bitnet-1B · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/joey00072/experiments-with-bitnet-1-5">Bitnet 1.5 实验 (~ngmi~)</a>: 未找到描述</li><li><a href="https://arxiv.org/html/2407.03040v1">Raw Text is All you Need: Knowledge-intensive Multi-turn Instruction Tuning for Large Language Model</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://tenor.com/view/go-green-cool-swag-dance-gif-12671410">Go Green GIF - Go Green Cool - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://huggingface.co/hugg">Hugg (Yy)</a>: 未找到描述</li><li><a href="https://x.com/karan4d/status/1768836844207378463">来自 mephisto ∃ (@karan4d) 的推文</a>: 我正在开源 worldsim，当然，这是用于初始化的 worldsim 系统提示词和对话：sysprompt: &lt;sys&gt;Assistant is in a CLI mood today. The human is interfacing with the simulator direc...</li><li><a href="https://x.com/norabelrose/status/1810342367972495590?s=46">来自 Nora Belrose (@norabelrose) 的推文</a>: @AiEleuther 可解释性团队正在为 Llama 3 8B 的每一层发布一套 top-k 稀疏自编码器：https://huggingface.co/EleutherAI/sae-llama-3-8b-32x 我们正在开发一个自动...</li><li><a href="https://huggingface.co/Replete-AI/Llama3-8B-Instruct-Replete-Adapted">Replete-AI/Llama3-8B-Instruct-Replete-Adapted · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2">blockblockblock/Llama3-8B-Instruct-Replete-Adapted-bpw6-exl2 · Hugging Face</a>: 未找到描述</li><li><a href="https://github.com/microsoft/MInference">GitHub - microsoft/MInference: 为了加速长上下文 LLM 的推理，通过近似和动态稀疏计算 Attention，在保持准确性的同时，将 A100 上的预填充推理延迟降低了多达 10 倍。</a>: 为了加速长上下文 LLM 的推理，通过近似和动态稀疏计算 Attention，在保持准确性的同时，将 A100 上的预填充推理延迟降低了多达 10 倍...</li><li><a href="https://github.com/ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models">GitHub - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models: 此仓库收集了关于 LLM 可解释性的所有相关资源</a>: 此仓库收集了关于 LLM 可解释性的所有相关资源 - ruizheliUOA/Awesome-Interpretability-in-Large-Language-Models</li><li><a href="https://huggingface.co/Weyaxi/Einstein-v7-Qwen2-7B">Weyaxi/Einstein-v7-Qwen2-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://fxtwitter.com/Weyaxi/status/1809644014515154961">来自 Weyaxi (@Weyaxi) 的推文</a>: 🚀 推出基于强大的 Qwen2 7B 模型的 Einstein v7，使用多样化、高质量的数据集进行监督微调 (SFT)！ 📊 第 7 版增加了 SystemChat 和一部分 a...</li><li><a href="https://anyscale.com/blog/fine-tuning-is-for-form-not-facts)">Blog | Anyscale</a>: Anyscale 是领先的 AI 应用平台。通过 Anyscale，开发者可以立即构建、运行和扩展 AI 应用。
</li>
</ul>

</div>
  

---

### **Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1259276528160739338)** (2 messages): 

> - `使用 LLM 生成确定性报告`
> - `传统编程的集成` 


- **寻求使用 LLM 生成确定性报告的方法**：**nav10** 询问了如何使用 LLM 创建确定性报告以识别业务流程中的瓶颈，目标是达到 80% 以上的一致性率。
   - **nav10** 正在考虑结构化生成以及使用 LLM judge 进行排序的可能性。
- **关于结合传统编程与 LLM 的建议**：成员 **deoxykev** 建议用传统语言编写确定性部分的代码，并将 LLM 用于传统编程效率不高的、小型的、结构化的任务。
   - *诀窍是尽可能少地使用 LLM，而在使用时，只让它们执行受限且简单的任务。*


  

---


### **Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1259557677701271744)** (4 messages): 

> - `RAG 与幻觉`
> - `维基百科风格的引用`
> - `RAG 与 Hugging Face Agents` 


- **RAG 幻觉研究**：一段 YouTube 视频 [“无幻觉？评估领先 AI 法律研究工具的可靠性”](https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu) 讨论了一篇斯坦福大学关于各种 LegalTech 工具中幻觉程度的论文。
   - *研究 RAG 模型如何处理法律查询，可以深入了解它们在关键应用中的幻觉率和可靠性。*
- **提议使用维基百科风格的引用**：成员们讨论了使用维基百科风格的 `<ref> </ref>` 标签进行引用，理由是基础模型在预训练阶段就已熟悉这种格式。
   - 一位成员分享了一个[示例模板](https://en.wikipedia.org/wiki/Template:Ref)，以说明如何正确格式化这些引用。
- **被严重低估的 RAG 教程**：一条推文强调了 [@AymericRoucher](https://x.com/mervenoyann/status/1810291157961781671) 在 Hugging Face Cookbook 中编写的 RAG 和 Agents 教程，并指出 agentic RAG 的表现优于标准 RAG。
   - *这些教程为增强 RAG 性能提供了宝贵的见解和技术，是必读内容。*


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://youtu.be/no7EQkOiHQM?si=b35yua7rZuaEVvKu">无幻觉？评估领先 AI 法律研究工具的可靠性（论文解读）</a>：#rag #hallucinations #legaltech 深入探讨了最近的一篇斯坦福论文，该论文研究了各种集成了……的 LegalTech 工具中的幻觉程度。</li><li><a href="https://x.com/mervenoyann/status/1810291157961781671">merve (@mervenoyann) 的推文</a>：被严重低估：Hugging Face Cookbook 中由 @AymericRoucher 编写的 RAG 和 Agents 教程 📝 最新一篇是关于 agentic RAG 的，其表现优于标准 RAG，所有内容见下方 ⥥</li><li><a href="https://example.com">示例域名</a>：未找到描述</li><li><a href="https://en.wikipedia.org/wiki/Template:Ref">Template:Ref - 维基百科</a>：未找到描述
</li>
</ul>

</div>
  

---


### **Nous Research AI ▷ #[world-sim](https://discord.com/channels/1053877538025386074/1221910674347786261/1258870858672242688)** (5 messages): 

> - `WorldSIM 模拟成功`
> - `下一代模拟时代` 


- **WorldSIM 佛陀模拟迅速实现觉悟**：一位用户分享了他们创建一个植根于**佛教原则**的世界的经历，该世界在不到 3 万步的时间内演化成了**单一的觉悟种群**，称其“简直太容易了”。
   - 他们提到，由于这个模拟，在一个午休时间就耗尽了所有额度。
- **对下一代模拟时代的期待**：一位成员透露，目前的资源正投入到他们正在开发的**下一代模拟时代**中。
   - 这激发了频道中其他人的兴奋和好奇。


  

---

### **Modular (Mojo 🔥) ▷ #[general](https://discord.com/channels/1087530497313357884/1098713601386233997/1258872163432271964)** (68 条消息🔥🔥): 

> - `为安装 Mojo 将 WSL 更新至 WSL2`
> - `Python 中的依赖地狱 (Dependency Hell)`
> - `Mojo 四舍五入函数 Bug`
> - `获取 'Mojician' 徽章`
> - `Mojo 的 Int64 与 Float64 行为` 


- **为安装 Mojo 将 WSL 更新至 WSL2**: 用户讨论了为安装 Mojo 将 WSL 更新到 WSL2 相关的问题，特别是对于使用旧版 Windows 10 电脑的用户。
   - 分享了 [Microsoft 关于安装 WSL 的指南](https://learn.microsoft.com/en-us/windows/wsl) 链接，帮助一位尝试了数小时的用户解决了问题。
- **Python 中的依赖地狱噩梦**: 一位用户询问如何处理 Python 项目中冲突的依赖版本，其他用户回应称目前已知的唯一解决方案是并行安装或使用虚拟环境 (virtual environments)。
   - 针对 Mojo 或其他系统是否能解决此问题展开了有趣的讨论，并指向了一个建议改进的 [GitHub 讨论帖](https://github.com/modularml/mojo/discussions/1401)。
- **Mojo 在四舍五入函数上的困扰**: 几位用户发现了 Mojo 中 `round` 函数的多个 Bug，特别是 `int` 和 `float` 类型没有按预期进行四舍五入。
   - 在讨论不一致性时，用户发现 Mojo 中的 SIMD 四舍五入没有正确使用第二个参数，导致输出符合预期。
- **获取 'Mojician' 徽章的步骤**: 用户询问如何在服务器上获得 'Mojician' 徽章，发现需要用 Mojo 创作一些酷炫的东西并发布到 Community Posts。
- **Mojo Int64 和 Float64 类型的非预期行为**: 通过讨论，用户注意到 Mojo 对 `Int64` 和 `Float64` 类型的处理在调用四舍五入函数时会导致非预期行为。
   - Mojo 中的 `Roundable` trait 目前存在局限性，导致尽管指定了小数位数，四舍五入始终发生在零位小数。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://learn.microsoft.com/en-us/windows/wsl/install">安装 WSL</a>: 使用命令 wsl --install 安装 Windows Subsystem for Linux。在 Windows 机器上使用由你偏好的 Linux 发行版（Ubuntu, Debian, SUSE, Kali, Fedora, Pengwin...）运行的 Bash 终端。</li><li><a href="https://docs.modular.com/mojo/stdlib/builtin/math/round">round | Modular 文档</a>: roundT: Roundable -&gt; $0</li><li><a href="https://github.com/modularml/mojo/discussions/1401">自动修补第三方导入以缓解依赖地狱 · modularml/mojo · Discussion #1401</a>: 维基百科关于依赖地狱 (dependency hell) 的文章提供了很好的定义和形式列表。提议的解决方案应该能解决所有形式的依赖地狱，但为了清晰起见...
</li>
</ul>

</div>
  

---


### **Modular (Mojo 🔥) ▷ #[🔥mojo](https://discord.com/channels/1087530497313357884/1151418092052815884/1259055427094839306)** (10 条消息🔥): 

> - `Mojo 中的 __del__ 方法`
> - `Mojo 3D 图形示例`
> - `Mojo 中的常用 libc 函数`
> - `使用 Mojo 进行交叉编译`
> - `在 Mojo 中使用 partition 方法` 


- **理解 Mojo 中的 __del__ 方法**: 成员们讨论了 Mojo 如何使用 **ASAP** 策略在实例最后一次使用时调用析构函数，以及如何手动延长生命周期。
- **Mojo 与 3D 图形：OpenGL, Vulkan 和 WebGPU**: 一位成员询问了使用 Mojo 进行 3D 图形开发的示例，例如 **OpenGL**、**Vulkan** 或 **WebGPU**。
- **Mojo 中可访问的常用 libc 函数**: 成员们讨论了在 Mojo 标准库之外调用常用 **libc 函数** 的可用性。
- **Mojo 目前不支持交叉编译**: 成员们询问了是否可以使用 Mojo 进行交叉编译 (cross compilation)。



**提到的链接**: <a href="https://github.com/saviorand/lightbug_http/blob/main/external/libc.mojo">lightbug_http/external/libc.mojo at main · saviorand/lightbug_http</a>: 适用于 Mojo 的简单快速的 HTTP 框架！🔥。欢迎在 GitHub 上通过创建账号为 saviorand/lightbug_http 的开发做出贡献。

  

---

### **Modular (Mojo 🔥) ▷ #[nightly](https://discord.com/channels/1087530497313357884/1224434323193594059/1259023164164542464)** (9 messages🔥): 

> - `Mojo 编译器 Nightly 版本发布`
> - `Changelog 更新与 PR`
> - `移除 C 指针语义`
> - `表达式颜色变化`
> - `元组解包（Tuple unpacking）请求` 


- **Mojo 发布 Nightly 编译器更新**：新的 Nightly Mojo 编译器已发布，更新至 `2024.7.605` 以及随后的 `2024.7.705`，可通过 `modular update nightly/mojo` 进行更新。
   - 更新内容包括多项更改，例如在未设置 `HOME` 时为家目录提供回退方案，以及新增了一个遵循 Python 语法的 `pwd` 模块。
- **Changelog 更新对 PR 至关重要**：成员们明确表示，由 `PR` 提供的任何更改、添加或删除都应记录在 `changelog.md` 中。
   - 这确保了项目修改的清晰文档化和跟踪。
- **摆脱 C 指针语义**：移除将整数转换为指针的能力是摆脱 C 指针语义工作的一部分。
   - *Melodyogonna* 在 Changelog 中找到了这一更改的原因。
- **表达式失败颜色：红色还是黑色？**：一位用户注意到表达式失败的颜色现在显示为黑色而不是红色，并询问这是否是刻意为之。
   - 另一位用户确认在他们那边显示的仍然是红色。
- **功能请求：元组解包（Tuple Unpacking）**：Benny.n 询问是否可能为非 `def` 函数和别名（aliases）提供元组解包功能。
   - 该用户认为，*Alias a, b, c = (1, 2, 3)* 将是一个非常实用的功能。


  

---


### **Modular (Mojo 🔥) ▷ #[mojo-marathons](https://discord.com/channels/1087530497313357884/1255303604894437388/1258896353711493250)** (77 messages🔥🔥): 

> - `Matmul 算法优化`
> - `SIMD 与缓存性能`
> - `Mojo 的编译时间问题`
> - `性能自动调优（Autotuning）` 


- **在 Matmul 中选择栈分配**：一位成员讨论了在 Matmul 算法中使用栈分配（stack allocation）进行临时存储，以提高缓存局部性和性能，特别是在最内层循环中。
   - 他们的测试显示出显著的性能差异，强调了预取（prefetching）和缓存优化的重要性。
- **Graviton 3 上的对齐与 SIMD 优化**：成员们确认 Graviton 3 的缓存行大小为 64 字节，并讨论了 SIMD 指令的对齐要求。
   - 有人建议 `simdwidth` 最好是 256 字节的倍数，以避免性能问题。
- **Matmul 算法中处理小矩阵**：引入了针对小矩阵的优化，利用简单循环来最小化开销并提高性能。
- **Mojo 特化（Specializations）带来的编译时间问题**：一位用户指出，由于 Mojo 中针对不同矩阵大小和数据类型的多次特化，导致编译时间过长。
   - 建议有效处理编译时数值，以避免性能瓶颈。
- **性能优化的自动调优（Autotuning）前景**：讨论强调了自动调优在优化 `simdwidth` 和块大小（block sizes）方面的效用，目前这些工作非常耗时且不具备可移植性。
   - 成员们表示希望自动调优功能能够回归，以简化优化过程。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://github.com/gabrieldemarmiesse/Mojo-Marathons/tree/output_csv">GitHub - gabrieldemarmiesse/Mojo-Marathons at output_csv</a>: 通过在 GitHub 上创建账号来为 gabrieldemarmiesse/Mojo-Marathons 的开发做出贡献。</li><li><a href="https://github.com/modularml/mojo/issues/2053.">Issues · modularml/mojo</a>: Mojo 编程语言。通过在 GitHub 上创建账号来为 modularml/mojo 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **Cohere ▷ #[general](https://discord.com/channels/954421988141711382/954421988783444043/1258862049170161715)** (67 messages🔥🔥): 

> - `AI-Plans platform`
> - `Using the rerank API`
> - `Cohere community introductions`
> - `Meta Learning by Radek Osmulski`
> - `Dark theme for Coral Chat Interface` 


- **用于红队对抗对齐的 AI-Plans 平台**：一位用户提到正在开发 **AI-Plans**，这是一个专为红队对抗对齐（red teaming alignment）计划设计的同行评审平台。
   - 他们目前尚未提供更多细节或链接。
- **rerank API 部署遇到困难**：一名成员在使用生产密钥调用 **rerank API** 时遇到问题，尽管在本地运行正常，但在部署期间遇到了 `TypeError`。
   - 其他用户建议检查脚本，特别是数据编码，并可能通过更新 Cohere SDK 来解决差异。
- **新成员的介绍与讨论**：新用户向社区介绍了自己，表达了对**加入 Cohere 并探索其工具**的兴奋之情。
   - 例如，一位用户分享了他们对协同工作以及使用 **Aya** 进行文档和构思工作流的兴趣。
- **Radek Osmulski 的 Meta Learning**：一位用户分享了 **Radek Osmulski 的 Meta Learning** 总结，并在其博客 [此处](https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/) 提供了更详细的笔记链接。
   - 他们描述了关键要点，包括 Stack Overflow 的重要性、代码编辑器的有效使用以及学习过程中实践练习的价值。
- **Coral Chat Interface 改进建议**：用户为 **Coral Chat Interface** 提出了多项增强建议，例如实现深色模式和为消息添加编辑按钮。
   - 一位用户承认 **Cohere 正在不断发展**，并暗示即将推出的版本 2 将包含更多 UI 功能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://riteshrm.github.io/posts/Meta%20Learning%20By%20Radek%20Osmulski%20Chapter%20Wise%20Summary%20Points/">Ritesh Kumar Maurya - Meta Learning Book Chapter Wise Summary Points</a>：未找到描述</li><li><a href="https://youtu.be/gFTLmVsX3ZQ?feature=shared">The Dream Team | Crash Zone - Season 1 Episode 1</a>：网络上的一个加密消息变成了 Mike、Pi、Bec、Marcello 和 Ram 无法抗拒的谜题。顺着线索，他们见到了 Alexandra Dav...
</li>
</ul>

</div>
  

---


### **Cohere ▷ #[project-sharing](https://discord.com/channels/954421988141711382/1218409701339828245/1258888358910759006)** (78 messages🔥🔥): 

> - `Rhea Platform`
> - `AI Creation and Interaction`
> - `Organizational Accounts`
> - `User Experience Feedback`
> - `Coding Club Projects with Children` 


- **Rhea 发布“保存到项目”功能**：“保存到项目”功能现已在 [Rhea 平台](https://rhea.run)上线，允许用户直接从仪表板保存交互式 HTML 应用程序。
- **编程俱乐部利用 Rhea 探索 AI**：一位经营儿童编程俱乐部的用户很高兴能与学生一起使用 Rhea 集成 AI 和 HTML 项目，并指出该平台用户友好且富有启发性。
- **发现 Rhea 注册流程中的 Bug**：一位用户发现 Rhea 的注册流程存在电子邮件验证问题，即电子邮件地址必须以小写形式输入。
- **Rhea 组织账户功能开发中**：Rhea 正在致力于支持组织账户，这将允许多个账户在共同的组织内共享和管理项目产出，从而增强协作工作。
- **来自 Rhea 的强大 AI 功能与技巧**：用户分享了在 Rhea 中利用 GPT-4 和 Claude 等不同 AI 来调试和增强代码的技巧；还讨论了隐藏命令和即将推出的功能，以提供更丰富的用户体验。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://docs.cohere.com/docs/the-cohere-platform">The Cohere Platform - Cohere Docs</a>：未找到描述</li><li><a href="https://rhea.run">Rhea | Byte Breeze Studios</a>：未找到描述
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1259901451879059467)** (1 messages): 

> - `Top-k Sparse Autoencoders`
> - `Llama 3 8B`
> - `Automated Pipeline for SAE Features`
> - `Training SAEs for 70B Model` 


- **Llama 3 8B 的 Top-k Sparse Autoencoders 已发布**：可解释性团队发布了针对 **Llama 3 8B** 每一层的 Top-k Sparse Autoencoders，可在 [Hugging Face](https://huggingface.co/EleutherAI/sae-llama-3-8b-32x) 获取。
   - 你可以使用 [sae 库](https://github.com/EleutherAI/sae) 加载它们。
- **自动化流水线与新的训练工作**：团队正在开发一个自动化流水线来解释 **SAE 特征**，并将很快开始为 **70B 模型** 训练 SAE。
   - 有兴趣提供帮助的人员，请查看 <#1153431135414669422>。


  

---


### **Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1258885906484559992)** (33 messages🔥): 

> - `PDF markup tools`
> - `Training variable resolution ViT using IJEPA`
> - `Evaluating LlaVa LLM`
> - `Randomness of Copilot`
> - `Determinism in LLM completions` 


- **难以找到具有“搜索并全部标记”功能的 PDF 标记工具**：一位用户正在寻找具有“搜索 -> 全部标记”功能的 PDF 标记工具，并报告称只发现了像 Bluebeam 和 PDF Studio 这样昂贵的选项。
- **使用 IJEPA 训练 ViT 展现出前景**：一位用户正在使用 IJEPA 训练变分辨率 ViT，在 20 个 epoch 后在 ImageNet1k 上达到了约 30% 的准确率，并在此分享了他们的[初步报告](https://theadamcolton.github.io/image-ssl-on-a-shoestring)。
   - 他们正在寻求反馈和帮助，以改进并加速其设置。
- **使用 lm-evaluation-harness 评估 LlaVa LLM 遇到问题**：一位用户报告了在使用 lm-evaluation-harness 评估 LlaVa LLM 时出现关于无法识别配置类的错误。
   - 他们正在寻求帮助以解决此问题。
- **Copilot 在姓名选择中的随机性受到质疑**：一位成员对 Copilot 从 120 人的名单中选择 50 人进行抽奖时的随机性提出了质疑，询问 LLM 是否擅长随机性。
   - 讨论强调了 LLM 是统计模型，可能会表现出确定性行为，一些证据表明微调模型中的姓名补全范围更窄。
- **Copilot 补全的确定性**：*philpax* 指出 Copilot 似乎会产生确定性的补全，经常在不同项目中生成相同的行内建议，例如 'This is a hack, but it works'。
   - 其他成员讨论认为，即使 temperature 设置允许生成多个补全，行内补全看起来也是一致的，且可能是确定性的。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">未找到标题</a>: 未找到描述</li><li><a href="https://x.com/ptrschmdtnlsn/status/1617019805793669125">来自 Peter Schmidt-Nielsen (@ptrschmdtnlsn) 的推文</a>: Copilot 真的非常想写下注释 &#34;This is a hack, but it works&#34;。这有点令人不安。
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1259063961458835489)** (67 messages🔥🔥): 

> - `T-FREE Tokenizer`
> - `Research on Model Expansion Efficiency`
> - `The BitNet Transformer`
> - `Gradient Conflicts in Diffusion Models`
> - `Quantization in Inference`

- **T-FREE Tokenizer 提出参数削减方案**：研究人员推出了 [T-FREE](https://arxiv.org/abs/2406.19223)，这是一种通过字符三元组（character triplets）上的激活模式对单词进行 Embedding 的 Tokenizer，在保持竞争力的性能的同时，将 Embedding 层的大小显著降低了 85% 以上。
- **关于模型扩展效率的辩论**：成员们讨论了诸如 SOLAR 等模型扩展技术的效率，引用了显示性能提升但往往缺乏与从头开始训练模型对比的 [论文](https://arxiv.org/abs/2310.07999)。
- **BitNet Transformer：1-bit 模型的飞跃**：[BitNet](https://arxiv.org/abs/2310.11453) 引入了一种可扩展的 1-bit 权重 Transformer 架构，在实现极具竞争力的性能的同时，显著降低了内存占用和能耗。
- **梯度冲突减缓了 Diffusion 模型的收敛**：一篇关于 Diffusion 模型的论文 [Min-SNR-$\gamma$](https://arxiv.org/abs/2303.09556) 揭示了收敛缓慢源于冲突的优化方向，并提出根据信噪比（signal-to-noise ratios）调整 Loss 权重来解决此问题，将收敛速度提高了 3.4 倍。
- **推理中的量化展示了实际效益**：[最近的研究](https://arxiv.org/abs/2404.00456) 展示了 QuaRot 在 LLM 上进行 4-bit 量化的有效性，在显著降低内存和计算成本的情况下，实现了接近全精度（full-precision）的性能。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2407.04620">Learning to (Learn at Test Time): RNNs with Expressive Hidden States</a>: Self-attention 在长上下文中表现出色，但具有平方复杂度。现有的 RNN 层具有线性复杂度，但它们在长上下文中的表现受限于其隐藏状态的表达能力...</li><li><a href="https://arxiv.org/abs/2407.03502">AgentInstruct: Toward Generative Teaching with Agentic Flows</a>: 合成数据对于加速大型和小型语言模型的发展变得越来越重要。尽管有几个成功的用例，研究人员也对...提出了担忧。</li><li><a href="https://arxiv.org/html/2312.15166v2">SOLAR 10.7B: Scaling Large Language Models with Simple yet Effective Depth Up-Scaling</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2212.09720">The case for 4-bit precision: k-bit Inference Scaling Laws</a>: 量化方法减少了表示模型中每个参数所需的位数，以精度换取更小的内存占用和推理延迟。然而，最终的模型大小取决于...</li><li><a href="https://arxiv.org/abs/2407.02783">52B to 1T: Lessons Learned via Tele-FLM Series</a>: Large Language Models (LLMs) 代表了迈向通用人工智能（AGI）的重要一步。随着 Scaling Laws 强调了增加模型大小的潜力，学术界加强了...</li><li><a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>: 最近的研究（如 BitNet）正在为 1-bit LLM 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://arxiv.org/abs/2406.19223">T-FREE: Tokenizer-Free Generative LLMs via Sparse Representations for Memory-Efficient Embeddings</a>: Tokenizer 对于 LLM 中的信息编码至关重要，但它们的发展最近停滞不前，且存在固有弱点。主要限制包括计算开销...</li><li><a href="https://arxiv.org/abs/2310.11453">BitNet: Scaling 1-bit Transformers for Large Language Models</a>: LLM 规模的不断扩大给部署带来了挑战，并因高能耗引发了对环境影响的担忧。在这项工作中，我们介绍了 BitNet，一种可扩展的...</li><li><a href="https://arxiv.org/abs/2407.02423">On the Anatomy of Attention</a>: 我们引入了一种范畴论图示形式化方法，以便系统地关联和推理机器学习模型。我们的图示直观地展示了架构，且不失...</li><li><a href="https://arxiv.org/abs/2303.09556">Efficient Diffusion Training via Min-SNR Weighting Strategy</a>: 去噪扩散模型一直是图像生成的主流方法，然而，训练这些模型通常面临收敛缓慢的问题。在本文中，我们发现导致收敛缓慢的原因是...</li><li><a href="https://arxiv.org/abs/2404.00456">QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs</a>: 我们介绍了 QuaRot，一种基于旋转的新型量化方案，能够对 LLM 进行端到端量化，包括 4-bit 的所有权重、激活值和 KV cache。QuaRot 以一种...的方式旋转 LLM。</li><li><a href="https://arxiv.org/abs/1812.10783">Topological Constraints on Homeomorphic Auto-Encoding</a>: 在对存在于高维空间中的已知非平凡流形上的数据进行表示学习时，自然希望编码器在受限于...时是同胚的。</li><li><a href="https://arxiv.org/abs/2310.07999">LEMON: Lossless model expansion</a>: 深度神经网络（尤其是 Transformer）的扩展对于其性能的飙升至关重要，并进一步导致了基础模型中复杂推理能力的出现。</li><li><a href="https://github.com/martius-lab/hitchhiking-rotations">GitHub - martius-lab/hitchhiking-rotations: Learning with 3D rotations, a hitchhiker’s guide to SO(3) - ICML 2024</a>: Learning with 3D rotations, a hitchhiker’s guide to SO(3) - ICML 2024 - martius-lab/hitchhiking-rotations</li><li><a href="https://github.com/Mooler0410/LLMsPracticalGuide">GitHub - Mooler0410/LLMsPracticalGuide: A curated list of practical guide resources of LLMs (LLMs Tree, Examples, Papers)</a>: 一个精选的 LLM 实用指南资源列表（LLMs Tree、示例、论文）- Mooler0410/LLMsPracticalGuide
</li>
</ul>

</div>
  

---

### **Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1259864689849008199)** (3 messages): 

> - `Attention as Hypernetwork`
> - `Attention as Hypernetwork 的实证结果` 


- **将 Attention 重新表述为 Hypernetwork**：一位成员分享了一篇[论文](https://arxiv.org/abs/2406.05816)，该论文将 Attention 重新表述为 **Hypernetwork**。
   - *对我来说，似乎 W_key 和 W_value 构成了 hypernetwork。*
- **对 Attention as Hypernetwork 论文的否定**：一位成员建议忽略这篇[论文](https://arxiv.org/abs/2406.05816)，并将 hypernetwork 部分解释为 attention scores。
   - 另一位成员同意这一评估。



**提到链接**：<a href="https://arxiv.org/abs/2406.05816">Attention as a Hypernetwork</a>：Transformer 在某些情况下可以泛化到新的问题实例，这些实例的组成部分可能在训练中遇到过，但其组合方式未曾见过。什么机制...

  

---


### **Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1259571359990943824)** (2 messages): 

> - `Mech Interp 阅读清单 v2`
> - `个人偏好的论文清单`
> - `Mechanistic Interpretability`
> - `阅读清单`
> - `文献综述` 


- **极具个人观点的 Mech Interp 阅读清单 v2 发布！**：**Neelnanda** 宣布发布了[其 Mechanistic Interpretability 阅读清单的 v2 版本](https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1)，更新了清单，包含了他们最喜欢的论文、核心要点和评论。
   - *这是我两年前制作的[类似清单](https://www.alignmentforum.org/posts/SfPrNY45kQaBozwmu/an-extremely-opinionated-annotated-list-of-my-favourite)的大幅更新版本*。
- **社区对阅读清单表示感谢**：一位成员感谢 Neelnanda 为创建新阅读清单所付出的努力。
   - 该清单旨在帮助该领域的新人应对海量的 Mechanistic Interpretability 论文。



**提到链接**：<a href="https://www.alignmentforum.org/posts/NfFST5Mio7BCAQHPA/an-extremely-opinionated-annotated-list-of-my-favourite-1">An Extremely Opinionated Annotated List of My Favourite Mechanistic Interpretability Papers v2 — AI Alignment Forum</a>：这篇文章代表了我个人的犀利观点，不代表我的团队或雇主的意见。这是我两年前制作的类似清单的大幅更新版本……

  

---

### **Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1258878175455481897)** (32 条消息🔥): 

> - `LlaVa LLM 评估`
> - `lm-evaluation-harness 中的功能请求`
> - `AISI 的 Inspect 与 lm-eval harness 对比`
> - `长上下文评估基准` 


- **LlaVa LLM 评估困境**：一位成员在尝试使用 **lm-evaluation-harness** 评估 **LlaVa LLM** 时遇到了 `ValueError`，因为该模型是多模态模型，目前该 harness 尚未支持。
   - 社区建议使用 `HFLM._get_model`，并指出 **lm-evaluation-harness** 支持 `AutoModelForSeq2SeqLM` 和 `AutoModelForCausalLM` 类。
- **lm-evaluation-harness 功能请求**：有人提出了关于在 `lm-evaluation-harness` 中排除默认任务的问题，并建议为此选项添加 CLI 标志。
   - 成员们讨论了使用 `include_default` 标志的可能性，并详细说明了针对 OOM 问题的修复方案 ([GitHub Issue #1923](https://github.com/EleutherAI/lm-evaluation-harness/issues/1923))。
- **AISI 的 Inspect 与 lm-eval harness 对比**：Inspect AI 拥有强大的 UI 和设计良好的库，但与 **lm-eval harness** 相比，缺乏对本地模型经过实战检验的支持。
   - Inspect 为多次 LM 调用、Prompt Engineering 和前沿 API 模型提供了强大的支持，而 **lm-eval harness** 则专注于标准化和内置的任务逻辑。
- **提议长上下文评估基准**：创建了一个讨论串来讨论长上下文评估，如滑动窗口 PPL 和其他新任务，并建议参考 `wikitext` 的 word_perplexity 和 byte_perplexity 指标。
   - 社区成员分享了潜在基准的链接，并讨论了在长上下文评估中使用 word_perplexity 等指标 ([arXiv 论文](https://arxiv.org/abs/2402.13718))。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://arxiv.org/abs/2402.13718">$\infty$Bench: Extending Long Context Evaluation Beyond 100K Tokens</a>：处理和推理长上下文对于大语言模型 (LLMs) 的许多实际应用至关重要，例如文档理解和 Agent 构建。尽管最近取得了进展...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/issues/1923">OOM Issue · Issue #1923 · EleutherAI/lm-evaluation-harness</a>：你好！我正在运行评估，但一直遇到 OOM 错误。这是我的脚本：TASKS=&quot;mmlu&quot; BATCH_SIZE=1 NUM_SHOTS=5 MODEL=Qwen/Qwen1.5-4B API=vllm lm_eval \ --model ${API} \ --model_args pret...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/task.py#L1329).">lm-evaluation-harness/lm_eval/api/task.py at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/api/model.py#L59">lm-evaluation-harness/lm_eval/api/model.py at main · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 Few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---


### **Eleuther ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/)** (1 条消息): 

wendlerc: 有人有好的 SDXL latent downscaler 吗？我想从 128x128x4 降采样到 64x64x4。
  

---


### **Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1259707602410016768)** (1 条消息): 

> - `Docker 容器使用`
> - `GPT-Neox 部署`
> - `用于大规模作业的 Kubernetes`
> - `Docker Compose 与 Kubernetes 对比` 


- **关于使用 Docker 部署 GPT-Neox 的问题**：一位成员询问了 GPT-Neox Docker 容器的实际用法，提到取得了一些成功，但对其在大规模作业中的效果表示怀疑。
   - 他们推测 Kubernetes 可能比 Docker Compose 对此类作业更有用，并寻求他人关于实际部署实践的见解。
- **考虑使用 Kubernetes 而非 Docker Compose**：一位成员想知道在运行 GPT-Neox 的大规模作业时，Kubernetes 是否比 Docker Compose 更有益。
   - 他们询问其他人是否在实践中真正使用 Docker 容器，以及 Docker Compose 是否是首选平台。

### **LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1258890742009565244)** (41 messages🔥): 

> - `JPEG XL Image Codec`
> - `Kolors GitHub Repository`
> - `Noise Scheduling in Machine Learning`
> - `Meta VLM Ads`
> - `IJEPA Training with Variable Resolution ViT` 


- **JPEG XL 统治图像编解码器**：在询问最先进的图像编解码器时，一位成员宣布 **JPEG XL** 是目前的最优选择。
- **Kolors GitHub 仓库受到关注**：一位成员分享了 [Kolors GitHub 仓库](https://github.com/Kwai-Kolors/Kolors)，其中包含一段他们认为特别值得注意的论文章节。
   - 他们提到其内容极具冲击力，甚至可能让人“瞬间中风”（instant stroke）。
- **关于机器学习中噪声调度（Noise Scheduling）的辩论**：参与者讨论了增加 100 个时间步是否可行，指出**切换到 v-prediction** 不需要进一步的 hack 手段，并能在终端时间步实现零终端 SNR（zero terminal SNR）以达到完全噪声。
   - 引用 **SDXL 论文**（引用 20）作为指导，另一位成员指出尽管在高分辨率采样时存在**测试与训练不匹配（test-train mismatches）**，该技术依然有效。
- **Meta VLM 广告遭到批评**：一位成员质疑为什么 Meta 在投放其 VLM 的广告而不是发布 **Llama3VLM**，暗示了用户间的挫败感。
   - 人们对 API 的可用性持怀疑态度，担心它可能仍与 Meta 的特定产品绑定。
- **分享 IJEPA 训练实验**：一位成员分享了使用 IJEPA 训练可变分辨率 ViT 的[初步结果](https://theadamcolton.github.io/image-ssl-on-a-shoestring)，在 20 个 **epochs** 后在 **Imagenet1k** 上达到了 **30% 的准确率**。
   - 他们邀请大家提供反馈并进行协作，以增强这种前景广阔且资源高效的模型训练方法。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://theadamcolton.github.io/image-ssl-on-a-shoestring">未找到标题</a>：未找到描述</li><li><a href="https://palette.fm/">Colorize Photo | Try Free | Realistic Colors</a>：在几秒钟内为您的黑白图像上色。免费试用我们的在线 AI 上色工具，无需注册。</li><li><a href="https://github.com/Kwai-Kolors/Kolors">GitHub - Kwai-Kolors/Kolors: Kolors Team</a>：Kolors 团队。通过在 GitHub 上创建账号为 Kwai-Kolors/Kolors 的开发做出贡献。</li><li><a href="https://gokaygokay-kolors.hf.space/">Kolors</a>：未找到描述
</li>
</ul>

</div>
  

---

### **LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1258988195601711114)** (28 messages🔥): 

> - `VALL-E 2`
> - `Terminator 模型讨论`
> - `新的字幕模型 - CapPa`
> - `VisualKeras 工具` 


- **VALL-E 2 在文本转语音中达到人类水平**：**VALL-E 2** 是零样本 TTS 的一个里程碑，引入了 Repetition Aware Sampling 和 Grouped Code Modeling，在 LibriSpeech 和 VCTK 数据集上的鲁棒性和自然度超越了以往的模型。
   - 尽管需要大量的计算资源，但它可以使用公开数据集复现，人们希望像 lucidrains 这样的大神能够复现其代码。
- **关于 Terminator 模型有效性的辩论**：讨论强调了许多模型研究在没有进行适当计算规模比较的情况下声称具有优越性的担忧；**Terminator** 因高计算需求和缺乏扩展定律（scaling law）证据而受到严厉批评。
   - 呼吁进行科学合理的比较，在计算规模跨度内检查模型，而不是随意挑选基准测试（benchmarks）。
- **CapPa 字幕模型需要 JAX**：一个新的字幕模型 **CapPa** 已发布，[这里](https://wandb.ai/craiyon/cappa-jax/reports/CapPa-Training-vision-models-as-captioners--Vmlldzo4NDUyNDUz)展示了使用 JAX 训练它的过程。
   - 提供详细信息的 GitHub 仓库是 [visualkeras](https://github.com/borisdayma/clip-jax/blob/main/utils/demo_cappa.ipynb)。
- **VisualKeras 工具介绍**：介绍了一个可能很有帮助的工具 **VisualKeras**，用于可视化具有可定制样式选项的 Keras 神经网络架构。
   - [在 GitHub 上查看](https://github.com/paulgavrikov/visualkeras)，支持适用于不同类型神经网络的层级和图表样式可视化。


<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://wandb.ai/craiyon/cappa-jax/re">craiyon</a>: Weights & Biases，机器学习开发者工具</li><li><a href="https://github.com/paulgavrikov/visualkeras">GitHub - paulgavrikov/visualkeras: Visualkeras 是一个 Python 软件包，旨在帮助可视化 Keras（无论是独立版本还是包含在 TensorFlow 中）神经网络架构。它允许轻松定制样式以满足大多数需求。该模块支持生成层级风格的架构，非常适合 CNNs（卷积神经网络），以及图表风格的架构，非常适合包括普通前馈网络在内的大多数模型。</a>: Visualkeras 是一个 Python 软件包，旨在帮助可视化 Keras（无论是独立版本还是包含在 TensorFlow 中）神经网络架构。它允许轻松定制样式以满足大多数需求。该模块支持...</li><li><a href="https://wandb.ai/craiyon/cappa-jax/reports/CapPa-Training-vision-models-as-captioners--Vmlldzo4NDUyNDUz">CapPa: Training vision models as captioners</a>: &quot;Image Captioners are Scalable Vision Learners Too&quot; 的开源复现。由 Boris Dayma 使用 Weights &amp; Biases 制作</li><li><a href="https://github.com/borisdayma/clip-jax/blob/main/utils/demo_cappa.ipynb">clip-jax/utils/demo_cappa.ipynb at main · borisdayma/clip-jax</a>: 使用 JAX 和 🤗 transformers 训练视觉模型。通过在 GitHub 上创建账号为 borisdayma/clip-jax 的开发做出贡献。
</li>
</ul>

</div>
  

---

### **LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1258882445437308948)** (52 messages🔥): 

> - `LangChain 中的 CSV 文件处理`
> - `LangChain 工具函数`
> - `LangGraph 设置问题`
> - `在本地运行 LLM`
> - `LangChain 中的异步配置` 


- ****LangChain 中的 CSV 文件处理****：一位用户正在寻求关于使用 LangChain 处理 CSV 文件的建议，询问使用多个 CSV 文件的现代方法，并希望改进之前的局限性。
- ****LangChain 中的异步配置****：一位用户询问如何在 LangChain 的异步环境中使用 `ensure_config()` 方法，寻求在使用 `astream_events` 时如何在 `ToolNode` 中获取 `thread_id` 的指导。
   - 用户收到建议，在工具的 `invoke` 函数中包含 `config` 参数以提取 `thread_id`。
- ****LangGraph ToolNode 错误****：一位用户报告了来自 `langgraph.prebuilt` 的 `create_react_agent` 中 `ToolNode` 的错误，导致 `NameError: name 'Type' is not defined`，并请求协助排查。
   - 用户分享了他们在 GitHub 上的 notebook 链接以便进一步调查。
- ****在本地机器上运行 LLM****：用户讨论了在配备 NVIDIA RTX 4090 GPU 等高端配置的本地 PC 上运行 `phi3`、`mistral` 和 `llama3` 等较小 LLM 模型的经验。
   - 同时也提出了关于使用多 GPU 运行 70B 参数等大规模模型的可行性和性能问题。
- ****LangChain 工具函数****：一位用户寻求在 LangChain 中将模型响应转换为 JSON 格式的帮助，并被引导至关于使用 `JsonOutputParser` 以及与 `Pydantic` 集成的特定文档。
   - 用户对指导表示感谢，并确认其问题已解决。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://python.langchain.com/v0.1/docs/modules/model_io/output_parsers/types/json/">JSON parser | 🦜️🔗 LangChain</a>：此输出解析器允许用户指定任意 JSON schema，并向 LLM 查询符合该 schema 的输出。</li><li><a href="https://python.langchain.com/v0.1/docs/modules/data_connection/retrievers/self_query/">Self-querying | 🦜️🔗 LangChain</a>：前往集成页面查看具有内置自查询支持的向量存储文档。</li><li><a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Prashant Dixit (@Prashant_Dixit0) 的推文</a>：✨开源全面的 LLM 术语表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，别忘了给 st...</li><li><a href="https://github.com/Adefioye/Alpha-Agent/blob/main/financial_annual_report/financial_annual_report.ipynb">Alpha-Agent/financial_annual_report/financial_annual_report.ipynb at main · Adefioye/Alpha-Agent</a>：通过在 GitHub 上创建账号为 Adefioye/Alpha-Agent 开发做贡献。</li><li><a href="https://langchain-ai.github.io/langgraphjs/tutorials/multi_agent/agent_supervisor">Agent supervisor - LangGraph.js</a>：未找到描述</li><li><a href="https://python.langchain.com/v0.2/docs/how_to/callbacks_async/#next-steps>).">如何在异步环境中使用回调 | 🦜️🔗 LangChain</a>：本指南假设你熟悉以下概念：</li><li><a href="https://github.com/langchain-ai/langchain/issues/16425>)">Issues · langchain-ai/langchain</a>：🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 开发做贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1259264035187064885)** (1 messages): 

> - `LangServe 部署问题`
> - `LangGraph Cloud 公告` 


- **对 LangServe 部署的困惑**：一位用户对从 LangSmith 部署 LangServe 表示困惑，提到在尝试部署时只收到关于 **LangGraph Cloud** 即将推出的消息。
   - *如果我想部署我的 LangServe API，我是否必须选择第三方云提供商？* 是一个后续问题。
- **LangGraph Cloud 即将推出**：成员们注意到在尝试通过 LangSmith 部署 LangServe 时，会出现关于 **LangGraph Cloud** 即将推出的消息。
   - 这引发了关于 LangServe 部署是否需要第三方云提供商的不确定性。


  

---


### **LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1259062994109993050)** (10 messages🔥): 

> - `doesVideoContain`
> - `qdurllm`
> - `OranScribe`
> - `LLM 术语表`
> - `高级研究助手`

- **创新的 'doesVideoContain' 工具引起关注**：介绍一款名为 **'doesVideoContain'** 的新工具，它允许视频自扫描特定内容，使用 [WebAI](https://github.com/jasonmayes/doesVideoContain)，完全在浏览器的 JS 中运行。
   - 观看 [YouTube 视频演示](https://www.youtube.com/watch?v=3FrYr13RL1E) 和 [Codepen 上的实时演示](https://codepen.io/jasonmayes/pen/eYaqZZo)。
- **'qdurllm' 发布，融合 Qdrant、URL 和 LLM**：介绍 **qdurllm**：一个本地搜索引擎，使用 [LangChain 和 Sentence Transformers](https://github.com/AstraBert/qdurllm) 将 URL 内容嵌入并存储在向量数据库中。
   - 允许用户运行语义搜索，并利用 **gemma-2b-it** 等 LLM 来增强查询结果，全部在本地通过 Gradio 界面运行。
- **自我修正 AI 编程助手发布**：宣布一款结合了 [Langchain 和 GPT4-o](https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant) 的新型自我修正、自我审查 Python 编程助手，灵感来自 **Codium-AI 的 AlphaCodium**。
   - 该助手旨在通过高效地自动识别和解决问题来增强编程工作流。
- **LangGraph 的 AI Agent 现已进入 Beta 测试**：一款名为 **Devin for LangGraph** 的新工具正在寻找 Beta 测试人员，该工具旨在将访谈转化为 LangGraph 中的 AI Agent。
   - 更多详情可以在 [Streamlit](https://definitive-ai.streamlit.app/) 和 [GitHub](https://github.com/Definitive-AI/Agent-Examples) 上找到，目前正在进行私人 Beta 测试。
- **Llamapp：用于准确响应的本地 RAG**：介绍 **Llamapp**，一个在本地运行的检索增强生成器（RAG），它[结合了文档检索和 LLM 生成](https://github.com/rajatasusual/llamapp)以提供准确的响应。
   - 该工具使用自定义检索技术，并强制 LLM 遵循源数据。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">来自 Prashant Dixit (@Prashant_Dixit0) 的推文</a>：✨开源全面的 LLM 术语表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，别忘了给个 star...</li><li><a href="https://huggingface.co/spaces/as-cle-bert/self-reviewing-coding-assistant">Self Reviewing Coding Assistant - as-cle-bert 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://github.com/rajatasusual/llamapp">GitHub - rajatasusual/llamapp: 一个完全在本地运行的检索增强生成器 (RAG)，结合了文档检索和语言模型生成，以提供准确且具有上下文相关性的响应。基于 @Langchain-ai 构建</a>：一个完全在本地运行的检索增强生成器 (RAG)，结合了文档检索和语言模型生成，以提供准确且具有上下文相关性的响应。基于...</li><li><a href="https://link.medium.com/DLY6pZRE3Kb">未找到标题</a>：未找到描述</li><li><a href="https://github.com/AstraBert/qdurllm">GitHub - AstraBert/qdurllm: 在桌面上搜索你喜欢的网站并与之聊天🌐</a>：在桌面上搜索你喜欢的网站并与之聊天🌐 - AstraBert/qdurllm</li><li><a href="https://github.com/Haste171/rag-demo">GitHub - Haste171/rag-demo: RAG 的基础解释与演示</a>：RAG 的基础解释与演示。通过在 GitHub 上创建账号来为 Haste171/rag-demo 的开发做出贡献。</li><li><a href="https://rubiks.ai/">Rubik's AI - AI 研究助手与搜索引擎</a>：未找到描述</li><li><a href="https://github.com/jasonmayes/doesVideoContain">GitHub - jasonmayes/doesVideoContain</a>：通过在 GitHub 上创建账号来为 jasonmayes/doesVideoContain 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=3FrYr13RL1E">Web AI 演示：Does Video Contain - 使用 AI 让视频能够“自看”以执行有用的工作</a>：这个小型实用程序库的初始版本允许你询问处理视频内容时最常见的问题——视频中是否包含某些内容...</li><li><a href="https://x.com/jason_mayes/status/1809497359812030801">来自 Jason Mayes (@jason_mayes) 的推文</a>：💡如果你能回答处理视频内容时最常见的问题：它是否包含你想要的东西，会怎样？我用 #WebAI 制作了一个 MVP，它会和你一起观看视频并抓取关键图像...</li><li><a href="https://scribe.oranai.com/">OranScribe</a>：OranScribe 是你的终极 AI 写作流库，旨在帮助你的企业利用行业最佳实践创建内容。简化你的内容创建并制作高性能的社交媒体内容...</li><li><a href="https://definitive-ai.streamlit.app/">未找到标题</a>：未找到描述</li><li><a href="https://github.com/Definitive-AI/Agent-Examples">GitHub - Definitive-AI/Agent-Examples: Agent 生成器输出</a>：Agent 生成器输出。通过在 GitHub 上创建账号来为 Definitive-AI/Agent-Examples 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1259065615268839444)** (2 条消息): 

> - `LangGraph state`
> - `Langchain + Graph RAG + GPT-4o` 


- **探索 LangGraph state 教程**：一段名为 ["LangGraph state"](https://youtu.be/DBXdE_5Jces) 的 YouTube 视频解释了如何使用带有 State（状态）的 LangGraph，State 代表了应用的当前快照。
   - *"在本教程中，我们将使用带有 State 的 LangGraph。"*
- **集成 Langchain 与 Graph RAG 和 GPT-4o**：一段名为 ["Langchain + Graph RAG + GPT-4o Python Project"](https://www.youtube.com/watch?v=HPmO1UZwfHc&t=1s) 的 YouTube 视频概述了为你的网站创建 AI/聊天机器人的 4 步流程。
   - *"#coding #rag #llm #ai #graphrag #chatbot 💚 代码链接：https://www.patreon.com/GaoDalie_AI"*


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.youtube.com/watch?v=HPmO1UZwfHc&t=1s">Langchain + Graph RAG + GPT-4o Python 项目：为你的网站打造简单的 AI/聊天功能</a>：#coding #rag #llm #ai #graphrag #chatbot 💚 代码链接：https://www.patreon.com/GaoDalie_AI 在这段视频中，我将引导你通过 4 个步骤来构建一个...</li><li><a href="https://youtu.be/DBXdE_5Jces">LangGraph state</a>：在本教程中，我们将使用带有 State 的 LangGraph。State 是一个共享的数据结构，代表应用的当前快照。00:45 加入 Skool ...
</li>
</ul>

</div>
  

---

### **OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1258860087091396708)** (32 条消息🔥): 

> - `带有 RAG 的技能库`
> - `OI 团队优先考虑安全性`
> - `GraphRAG`
> - `7 月 4 日家庭派对`
> - `RAG 系统中的 Langchain` 


- **通过 RAG 扩展技能提供一致性**：一名成员成功运行了带有 **RAG** 的技能库，这应该会使某些操作更加一致。
- **OI 团队优先考虑安全措施**：一名成员赞扬了 OI 团队花时间进行视频会议并讨论安全措施，强调了团队将安全性作为重要优先事项的承诺。
- **引入 GraphRAG 以增强检索增强生成**：一位用户分享了 **Microsoft GraphRAG** 的详细分解和教程，该工具将数据聚类到社区中，以实现更好的 **RAG** 用例。
- **7 月 4 日家庭派对圆满成功**：OI 团队通过新的演示、新面孔和更新预览庆祝了他们的 **7 月 4 日**家庭派对，并计划在每个月的第一个星期四继续举办此类活动。
- **在 RAG 中实现 Langchain**：讨论强调了在各种项目的 RAG 系统中使用 **Langchain** 的情况，成员们对进一步探索其功能表现出浓厚兴趣。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/tedx_ai/status/1808561861589139690">来自 Ted Werbel (@tedx_ai) 的推文</a>：@Microsoft 刚刚发布了他们的 GraphRAG 实现——这是一个用 Python 构建的基于图的检索增强生成系统。这里有一个关于它如何工作以及为什么它...的简化分解。</li><li><a href="https://github.com/MTG/freesound-python">GitHub - MTG/freesound-python: freesound API 的 Python 客户端</a>：freesound API 的 Python 客户端。通过在 GitHub 上创建账户来为 MTG/freesound-python 的开发做出贡献。
</li>
</ul>

</div>
  

---


### **OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1258984807317110785)** (10 条消息🔥): 

> - `发货时间线`
> - `O1 的说话能力`
> - `文本显示选项`
> - `Google I/O 演示眼镜`
> - `Linux 模块错误` 


- **前 1000 台设备将于 11 月前发货**：截至 4 月 30 日，前 1000 台设备的发货/交付预计时间表大约为今年 11 月，不过自 4 月以来这一情况可能有所变化。
- **关于 O1 说话能力的疑问**：一位成员询问 O1 是否可以说话，得到的回复表明如果配置正确，它应该可以。
- **将眼镜用作文本显示器**：一位用户建议眼镜可能会显示文本输出，其功能可能类似于 Google 的 I/O 演示眼镜。
   - 另一位用户提到了对 Meta 的 Rayban 眼镜进行越狱以实现类似功能的可能性。
- **寻求 Linux 模块错误 'typer' 的解决方案**：一位运行 Linux 的用户正在寻求解决 'ModuleNotFoundError: No module named 'typer'' 错误的帮助，并提到尝试过 `pip install typer` 但未获成功。


  

---



### **OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1259560590649851986)** (2 条消息): 

> - `新的模型浏览器 UI`
> - `Noromaid Mixtral 弃用` 


- **OpenRouter 发布新的模型浏览器 UI**：OpenRouter 推出了[全新的模型浏览器 UI](https://x.com/OpenRouterAI/status/1810001066240413908)，具有 **16 个参数过滤器**、**类别过滤器**、上下文长度、价格等功能。
   - /models 页面现在速度显著提升，尤其是在移动设备上，使用户更容易探索每周处理 740 亿个 token 的 **180 个活跃语言模型**。
- **Neversleep 的 Noromaid Mixtral 模型被弃用**：由于使用量下降，**Noromaid Mixtral 模型**将被弃用，并将在接下来的两周内继续通过 API 运行，之后将被移除。
   - *告别 Neversleep 的 Noromaid Mixtral*，因为它在设定时间段后将返回 404。



**提到的链接**：<a href="https://x.com/OpenRouterAI/status/1810001066240413908">来自 OpenRouter (@OpenRouterAI) 的推文</a>：宣布全新的模型市场 UI ✨ 探索 180 个每周处理 740 亿个 token 的活跃语言模型 👇

  

---

### **OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1258991361336545322)** (6 messages): 

> - `Viinyx AI 发布`
> - `文本转图像 API 服务` 


- **Viinyx AI 发布提升生产力**：**Viinyx AI** 是一款浏览器扩展，通过集成 **ChatGPT**、**Anthropic** 和 **Gemini** 等多个生成式 AI 模型，实现在网页任意位置进行写作和图像创建，从而增强浏览体验。[在 Chrome 网上应用店查看](https://chromewebstore.google.com/detail/viinyx-ai-assistant-chatg/ochleehcckobncbecepoccjhpjfgepae) 以及 [官方网站](https://www.viinyx.com)。
- **寻求文本转图像 API 服务**：一位用户询问了提供类似 **OpenRouter** 模式、支持多种模型的文本转图像 API 服务推荐。**Replicate** 被提议为一个可能的选项，其他提到的还包括 **Novita** 和 **Fireworks**。


<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://chromewebstore.google.com/detail/viinyx-ai-assistant-chatg/ochleehcckobncbecepoccjhpjfgepae">Viinyx - AI Assistant (ChatGPT, GPT-4o, Claude, Gemini)</a>: 强大的全能型 AI 副驾驶，提升你的生产力。使用生成式 AI (ChatGPT, Claude, Gemini) 在任何地方进行写作和绘画。</li><li><a href="https://www.viinyx.com">来自 Viinyx AI 的推文 - 最好的全能 AI 浏览器助手</a>: Viinyx AI 浏览器扩展 - 在任何网页上使用 ChatGPT, Claude, Meta.ai, Microsoft Copilot。总结页面和视频以加速你的学习。Viinyx AI 支持 BYOK 并使用你自己的 AI 提供商...
</li>
</ul>

</div>
  

---


### **OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1259002342649626736)** (27 messages🔥): 

> - `加密货币支付`
> - `Perplexity 模型`
> - `生成式视频的未来`
> - `OpenRouter 提供商选项`
> - `模型定价竞争` 


- **探索多种加密货币支付选项**：用户讨论了 **Coinbase Commerce** 支持通过 USDC、Polygon 网络上的 Matic 以及其他加密货币进行支付。
   - 有用户指出 **Matic** 支付体验良好。
- **Perplexity 模型存在 API 限制**：**Perplexity API** 的表现不如其 Web 界面，特别是在响应中缺少参考链接。
   - 对于技术查询的总结，像 **Phind** 以及**直接抓取 GitHub 和 StackOverflow** 的替代方案可能效果更好。
- **生成式视频质量预测**：一位用户询问了未来 1-1.5 年内**生成式视频**在**质量、速度和价格**方面的发展趋势。
   - 讨论并未得出具体的预测，凸显了此类技术进步的推测性质。
- **OpenRouter 允许自定义提供商**：成员们确认，如果用户能够处理大量的请求，**OpenRouter** 允许用户提供自己微调（finetuned）的模型。
   - 这为寻求集成自定义 AI 解决方案的开发者提供了灵活性。
- **OpenRouter 上 DeepInfra 与 Novita 的价格战**：**DeepInfra** 和 **NovitaAI** 正在 **OpenRouter** 上争夺 **Llama3** 和 **Mistral** 等模型的首位，价格差异极小。
   - 用户开玩笑说，他们通过降低 **0.001** 的价格来互换排名位置，直到达到极具竞争力的阈值。


  

---

### **LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1258947144123945027)** (6 条消息): 

> - `用于股票交易的 Agentic RAG`
> - `用于 RAG 数据集生成的工具包`
> - `作为微服务的 Agents`
> - `多文档财务分析师 Agent`
> - `RAG 检索评估` 


- **用于股票交易的 Agentic RAG 📈🤖**：一段教程视频展示了如何构建一个由 [Llama Index agent/tool/RAG 抽象](https://t.co/ocPaeLphyG) 驱动的 AI 交易助手。
   - 该助手可以执行股票交易中的各种任务，如 [视频教程](https://t.co/dcLG3orq0s) 中所示。
- **用于 RAG 数据集生成的工具包**：为 RAG 创建评估数据集具有挑战性，但 [Giskard AI 提供了一个工具包](https://t.co/rQ7WxplJpF) 用于生成多样化的问答集。
   - 正如其 [文章](https://t.co/sewtQcb9b8) 中讨论的，与大多数自动数据集生成器相比，该工具包涵盖了更广泛的问题范围。
- **作为微服务的 Agents**：正如 [这篇文章](https://t.co/y9a3PdfW0M) 中解释的，Llama-agents 能够将 Agent 服务和工具服务都设置为能够处理大量请求的微服务。
   - 这种模式简化了 Agent 与工具之间的交互，将它们转变为可扩展的微服务。
- **多文档财务分析师 Agent**：将每份财务文档视为一个工具，可以构建一个 [多文档财务分析师 Agent](https://t.co/LJhV838EUM) 来分析分类文档，尤其是 10K 报告。
   - Pavan Mantha 展示了如何利用 [Llama Index 的特性](https://t.co/rOetN1zeNg) 来辅助该 Agent 的分析。
- **RAG 检索评估的重要性**：RAG 中的检索评估可能比 LLM 评估更关键；必要的步骤包括确定正确的指标并拥有统一的数据集表示，详见 [Ross A. 的这篇文章](https://t.co/7uSgwwWThM)。
   - 这些评估可以显著影响 RAG 系统的有效性和准确性，在 [这篇文章](https://t.co/xxj69nneDK) 中有进一步讨论。


---

### **LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1258866349866745969)** (21 条消息🔥): 

> - `AI application mentorship` (AI 应用导师指导)
> - `Claude 3 models in Bedrock` (Bedrock 中的 Claude 3 模型)
> - `Knowledge graphs from GitHub code` (基于 GitHub 代码构建知识图谱)
> - `Structured data queries with LlamaIndex` (使用 LlamaIndex 进行结构化数据查询)
> - `ReAct agent observations` (ReAct Agent 的观察结果)


- **寻求 AI 应用导师指导**：一名成员请求导师或指导者帮助构建 AI 应用，表示他们只需要指导，执行工作由自己完成。
   - *pwnosaurusrex* 建议从 LlamaIndex 文档中的 [5 行代码](https://docs.llamaindex.ai/en/stable/getting_started/starter_example/) 入门示例开始。
- **Bedrock 现已支持 Claude 3 模型**：有人询问 Bedrock 中对 Claude 3 模型支持的情况。
   - *whitefang_jr* 确认了 Claude 3 模型已获得支持，并分享了一个 [GitHub 链接](https://github.com/run-llama/llama_index/blob/65eb552b13664e713d3cdcf8f432e9696cabc50c/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py#L47) 作为参考。
- **基于 GitHub 代码构建知识图谱的挑战**：一名成员询问是否有人正在从 GitHub 代码仓库构建知识图谱 (Knowledge Graphs)。
   - 他们提到使用 Property Graph Store Index 进行实体提取和 Embeddings 创建，但在使用自定义 Retriever 时面临结果方面的挑战。
- **寻求使用 LlamaIndex 查询结构化数据的更好方法**：一名成员表示在跨多张表查询结构化数据 (SQL) 时遇到困难，并分享了 LlamaIndex 文档链接。
   - 他们还提到正在研究 [Vanna](https://github.com/vanna-ai/vanna) 以寻找潜在的解决方案。
- **通过响应对象访问 ReAct Agent 的中间步骤**：有人询问如何通过 Response 对象访问 ReAct Agent 的观察 (Observations)、思考 (Thoughts)、动作 (Actions) 和步骤。
   - *cheesyfishes* 回复称可以通过低级 API 实现，并分享了一个 [Google Colab 链接](https://t.co/YEGfTOkAkY)。


<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://x.com/Prashant_Dixit0/status/1809900514097979768">Prashant Dixit (@Prashant_Dixit0) 的推文</a>：✨开源全面的 LLM 词汇表✨ 探索、学习并添加关于 #LLMs 和 #GenAI 的术语。让我们让 AI 对每个人都变得简单。🚨定期添加新术语，别忘了给项目打星...</li><li><a href="https://docs.llamaindex.ai/en/stable/getting_started/starter_example/">入门教程 (OpenAI) - LlamaIndex</a>：无描述</li><li><a href="https://github.com/run-llama/llama_index/blob/65eb552b13664e713d3cdcf8f432e9696cabc50c/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py#L47">llama_index/llama-index-integrations/llms/llama-index-llms-bedrock/llama_index/llms/bedrock/utils.py</a>：LlamaIndex 是用于 LLM 应用的数据框架</li><li><a href="https://t.co/YEGfTOkAkY">Google Colab</a>：无描述
</li>
</ul>

</div>
  

---



### **tinygrad (George Hotz) ▷ #[general](https://discord.com/channels/1068976834382925865/1068976834928193609/1259197885099282633)** (7 条消息): 

> - `team red's drivers` (Red Team/AMD 驱动)
> - `Instinct cards` (Instinct 系列显卡)
> - `custom grads API` (自定义梯度 API)
> - `tinygrad functions` (tinygrad 函数)
> - `Monday team meeting` (周一团队会议)


- **Instinct 卡的信心受到质疑**：一名成员质疑 **Red Team (AMD) 驱动的信心水平** 是否足以让 Instinct 卡值得购买，并表示在获得更好的支持之前，对购买廉价的二手 Mi100 感到犹豫。
   - *另一名成员指出，目前只有 7900xtx 卡正在接受测试，选择 Instinct 卡意味着需要自力更生。*
- **自定义梯度 (custom grads) API 提案**：一名用户建议实现一个更好的 **自定义梯度 API**，类似于 **jax.customvjp**，以简化 Tensor 操作，特别是针对量化训练 (Quantization Training)。
   - 他们提出愿意负责这项改进工作，并认为 tinygrad.functions 中当前的语法并不理想，因为它操作的是 **lazybuffers** 而不是 Tensors。
- **即将召开的周一团队会议议程**：太平洋时间 **周一上午 9:40** 的会议主题包括 tinybox 更新、来自 tinybox 所有者的反馈，以及关于 [新内存调度器](https://github.com/tinygrad/tinygrad/pull/5278)、LLVM NaN 修复、`UOps.VECTORIZE`、Bug 修复和新 API 的讨论。
   - 其他讨论点还包括 **sharded llama**、**sin/exp/log 近似**、**mlperf** 以及 **其他悬赏任务 (Bounties)**，如 **std mean 单内核**、**Qualcomm 运行时**、**Apple AMX** 和 **clang mmap 运行时**。


  

---

### **tinygrad (George Hotz) ▷ #[learn-tinygrad](https://discord.com/channels/1068976834382925865/1070745817025106080/1258916976466858127)** (20 条消息🔥): 

> - `requires_grad 行为`
> - `多 GPU 训练文档`
> - `tensor 比较方法`
> - `Adam 优化器问题`
> - `Tinygrad tensor 的新方法` 


- **澄清 requires_grad 默认行为**：讨论了为什么 `tensor.py` 中的 `requires_grad` 可以是 **None**、**False** 或 **True**。None 是默认值，如果 Tensor 被放入优化器中，它会被更新为 **True**。
- **Tinygrad 多 GPU 训练简介**：对于多 GPU 训练，用户可以参考 [beautiful_mnist_multigpu.py 示例](https://github.com/tinygrad/tinygrad/blob/master/examples/beautiful_mnist_multigpu.py)。模型可以使用 `shard(axis=None)` 进行复制，数据可以使用 `shard(axis=0)` 进行拆分。
- **简化 Tinygrad 中的 Tensor 比较**：用户询问了 Tinygrad 中与 `torch.all` 等效的 Tensor 比较方法。建议使用 `(t1 == t2).min() == 1` 进行比较，随后在 [此 commit](https://github.com/tinygrad/tinygrad/commit/6856f915d6f0e10d41e8e11c8976024989d90aa7) 中添加了 **Tensor.all** 以匹配 Torch 方法。
- **Adam 优化器导致 NaNs**：一位成员报告称，在使用 Adam 优化器时，权重在第二步后变为 **NaN**，而使用 SGD 则运行正常。


  

---



### **OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1259389623155691531)** (8 条消息🔥): 

> - `模型合并`
> - `MInference`
> - `RAM 问题`
> - `Offload 配置` 


- **模型合并困扰**：一位成员询问另一位是否仍在尝试合并他们的模型。
   - 另一位成员询问了合并时使用的工具。
- **微软推出 MInference**：一位成员分享了微软 **MInference** 项目的 [GitHub 链接](https://github.com/microsoft/MInference)，该项目可加速长上下文 LLM 的推理，并将 A100 上的预填充（pre-filling）延迟降低高达 10 倍。
   - 该工具采用近似和动态稀疏计算，在提高 **A100** 预填充性能的同时保持准确性。
- **模型合并期间的 RAM 问题**：在询问 RAM 耗尽的情况后，另一位用户确认了该问题。
   - 通过为该进程**指定 CPU** 解决了该问题。



**提及的链接**：<a href="https://github.com/microsoft/MInference">GitHub - microsoft/MInference: 为了加速长上下文 LLM 的推理，采用近似和动态稀疏计算 Attention，在保持准确性的同时，将 A100 上的预填充推理延迟降低高达 10 倍。</a>

  

---


### **OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1259513609596309524)** (1 条消息): 

> - `Yi-1.5-9B-Chat 训练`
> - `Hermes-2.5 集成`
> - `基准测试结果`
> - `扩展上下文长度的未来计划` 


- **在 OpenHermes-2.5 上微调的 Yi-1.5-9B-Chat**：一位成员分享了他们在 **OpenHermes-2.5** 上训练的 **Yi-1.5-9B-Chat**，并对结果表示满意，提供了 [GGUF 版本和常用量化版本](https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat) 供试用。
   - 该模型在特定场景下显得更聪明、更具“意识”，并提到其在同类模型的 **AGIEval Benchmark** 中有显著提升。
- **Hermes-2.5-Yi-1.5-9B-Chat 的微调细节**：该微调模型是 [01-ai/Yi-1.5-9B-Chat](https://huggingface.co/01-ai/Yi-1.5-9B-Chat) 的一个版本，使用了 [teknium/OpenHermes-2.5](https://huggingface.co/datasets/teknium/OpenHermes-2.5) 数据集，在 4 台 NVIDIA A100 40GB GPU 上训练了 48:32:13 小时。
   - 模型的**序列长度**为 8192 tokens，并使用 **chat-template**: chatml 进行训练。
- **使用 POSE 进行未来改进**：计划使用 **POSE** 将模型的上下文长度扩展到 **32k tokens**。
   - 此增强旨在提高模型在处理更长上下文场景时的性能。



**提及的链接**：<a href="https://huggingface.co/juvi21/Hermes-2.5-Yi-1.5-9B-Chat">juvi21/Hermes-2.5-Yi-1.5-9B-Chat · Hugging Face</a>: 暂无描述内容

  

---

### **OpenAccess AI Collective (axolotl) ▷ #[axolotl-help-bot](https://discord.com/channels/1104757954588196865/1225300056442409040/1259560193583218910)** (5 messages): 

> - `chat_template`
> - `mistral finetuning in axolotl` 


- **关于 Mistral 微调 Chat Template 的查询**：一位成员询问在 axolotl 中进行 **Mistral finetuning** 时应该使用哪个 **chat_template**。
   - 另一位成员回答说这取决于数据集结构。
- **在 YAML 中配置 Chat Template**：建议在 Axolotl 的 Mistral 微调中使用 `"chatml"` 聊天模板。
   - 提供了一个在 YAML 格式中使用 `"chatml"` 模板的配置示例。



**提到的链接**：<a href="https://phorm.ai/query?projectId=1e8ce0ca-5f45-4b83-a0f4-9da45ce8e78b&threadId=be14de0a-8a0e-4075-90ea-a6fac1a0008b)">OpenAccess-AI-Collective/axolotl | Phorm AI Code Search</a>：更快速地理解代码。

  

---



### **LLM Finetuning (Hamel + Dan) ▷ #[general](https://discord.com/channels/1238365980128706560/1238365980128706563/1259150359318888488)** (8 messages🔥): 

> - `MLOps implementation`
> - `Distributed VLLM inference`
> - `FP8 quantization issues`
> - `Chat template challenges` 


- **MLOps 实施的战略见解**：一位用户分享了一篇[博客文章](https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline)，探讨了构建 MLOps 流水线的关键问题，强调了理解 MLOps 基础知识和高质量数据的重要性。
   - 该文章旨在指导公司通过关键考量因素成功部署 MLOps，以提高模型准确性并降低运营成本。
- **使用 FP8 进行分布式 VLLM 推理的问题**：一位用户请求帮助解决在 8xL40S GPU 上对 FP8 量化的 Llama 3 70B 模型进行分布式 vLLM 推理时遇到的性能下降和输出错误问题。
   - 经过调试，确定该问题与 autofp8 对 padding tokens 的敏感性以及对 chat templates 的处理不当有关，随后该问题已得到解决。
- **Neural Magic FP8 量化**：用户尝试使用类似于 [Neural Magic 示例](https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py) 的代码进行 FP8 量化，但在推理设置中遇到了问题。
   - 经确认，FlashAttention-2 后端不支持 FP8 KV cache，这可能是导致性能问题的原因。
- **FP8 量化和 Chat Template 问题的解决**：经过进一步调查，用户发现 autofp8 对 padding token 的敏感性和 chat templates 的错误应用是问题的根本原因。
   - 对代码进行的调整和部分重写最终解决了这些问题，实现了正确的推理操作。


<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://nik-hil.hashnode.dev/diving-deep-essential-questions-for-building-your-mlops-pipeline">Essential Questions for Your MLOps Pipeline</a>：通过解决有关数据、模型开发、部署、监控、工具和治理的关键问题，构建稳健 MLOps 流水线的指南。</li><li><a href="https://github.com/neuralmagic/AutoFP8/blob/147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a/example_dataset.py">AutoFP8/example_dataset.py at 147fa4d9e1a90ef8a93f96fc7d9c33056ddc017a · neuralmagic/AutoFP8</a>：通过在 GitHub 上创建账户为 neuralmagic/AutoFP8 的开发做出贡献。</li><li><a href="https://github.com/vllm-project/vllm/issues/6179">[Usage]: Struggling to get fp8 inference working correctly on 8xL40s · Issue #6179 · vllm-project/vllm</a>：当前环境收集环境信息... PyTorch 版本：2.3.0+cu121 是否为调试构建：False 用于构建 PyTorch 的 CUDA：12.1 用于构建 PyTorch 的 ROCM：N/A 操作系统：Ubuntu 22.04.4...
</li>
</ul>

</div>
  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[replicate](https://discord.com/channels/1238365980128706560/1241163904927666287/1259183509965115392)** (1 messages): 

> - `Replicate billing setup issues` 


- **设置账单后未添加 Replicate 额度**：一位成员表示担心在设置账单后 **Replicate credits** 没有被添加。
   - 他们提到“抱歉太晚了”，暗示可能存在延迟或配置错误。
- **对账单设置时机的担忧**：提出的另一点是账单设置的时机是否会影响额度的分配。
   - 该成员今天没有看到 Replicate 的额度，暗示可能存在 **timing issues**（时机问题）。

### **LLM Finetuning (Hamel + Dan) ▷ #[jeremy_python_llms](https://discord.com/channels/1238365980128706560/1242224309548875917/1259271877252092006)** (1 messages): 

> - `Transformers & Torch`
> - `Integrating with OpenAI/Anthropic models` 


- **探索 Transformers & Torch 的替代方案**：一位成员目前正在实验 **Transformers** 和 **Torch**，以评估它们在项目中的潜在有效性。
- **集成考虑：OpenAI/Anthropic**：正在考虑的另一个替代方案是集成来自 **OpenAI** 和 **Anthropic** 的模型。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[credits-questions](https://discord.com/channels/1238365980128706560/1243721538432270388/1259445018905542707)** (1 messages): 

> - `Credit Claims Closed`
> - `Credit Eligibility` 


- **积分申领永久关闭**：一条消息澄清了所有申领积分的表单均已关闭，**不再有人符合获得新积分的资格**。
- **积分资格更新**：该更新表明积分申领已永久关闭，且适用于所有用户，无一例外。


  

---


### **LLM Finetuning (Hamel + Dan) ▷ #[predibase](https://discord.com/channels/1238365980128706560/1245803791710687272/)** (1 messages): 

4.8.15.16.23.42_: 前 25 个积分对所有人开放，但有效期仅为 1 个月 🙂
  

---



### **Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1258882468967088220)** (2 messages): 

> - `Interconnects Bot Feedback` 


- **Interconnects Bot：少量反馈**：一位用户指出 Interconnects bot 的表现令人满意，但建议其最近的摘要没有太大变化。
- **Interconnects Bot 的潜在改进**：来自同一位用户的后续消息表达了希望 Interconnects bot 的功能有更显著的更新或改进。


  

---


### **Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1258909623163949197)** (8 messages🔥): 

> - `RAG discussions`
> - `Enterprises and RAG`
> - `RAG use cases`
> - `early AI boom`
> - `retrieval and cost efficiency` 


- **关于 RAG 的辩论**：成员们讨论了 **RAG** 及其对企业的感知效用，一些人认为这通常是由不在企业工作的人在谈论。
   - 另一位成员指出，虽然 **RAG** 可以帮助企业利用其内部知识库，但用例仍在演变。
- **早期 AI 热潮的炒作**：有人评论了早期 AI 热潮期间围绕 **RAG** 的最初**炒作**。
   - *“当时人们对此表现得很荒谬”*是大家共有的情绪。
- **企业的检索与成本效率**：一位成员强调，虽然并非所有企业都在使用 RAG，但它可以实现具有成本效益的模型和新的用例。
   - 另一位用户指出，利用内部知识库是企业理解并想要的一种技术选择。


  

---



### **Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1259210089537994833)** (6 messages): 

> - `Buzz excitement`
> - `FPGA meeting`
> - `Calendly scheduling` 


- **成员称 Buzz 非常棒**：一位成员表达了对 Buzz 的热情，随后 Autometa 暗示即将发布另一个有趣的发布。
- **Autometa 安排 FPGA 会议**：Autometa 请求安排一次会议来讨论 FPGA 主题，并提到有几个有趣的观点需要涵盖。
- **Alignment Lab 的开放 Calendly 调度**：Autometa 分享了一个 [开放的 Calendly 链接](https://calendly.com/alignmentlab/meeting) 用于安排讨论，欢迎任何有兴趣的人预约会议。



**提到的链接**：<a href="https://calendly.com/alignmentlab/meeting">meeting - Auto Meta</a>：未找到描述

  

---



### **LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/)** (1 messages): 

jeffreyw128: 哇，Flash 1.5 实际上非常好
  

---



### **AI Stack Devs (Yoko Li) ▷ #[assets](https://discord.com/channels/1122748573000409160/1176906086368935966/1259654236917202995)** (1 messages): 

> - `Google image searches for sprites`
> - `Purchased assets for tilesets` 


- **Sprite 素材来源于 Google 图片搜索**：一位成员提到所有的 sprite 都是通过随机的 Google 图片搜索获得的。
- **仅 Tileset 是购买的资产**：讨论强调唯一购买的资产是 tileset，而不是 sprite。


  

---



### **MLOps @Chipro ▷ #[events](https://discord.com/channels/814557108065534033/869270934773727272/)** (1 messages): 

jonononono: 有人去 EuroPython 吗？我要做一个关于向量化（vectorization）的演讲 👀
  

---

### **Mozilla AI ▷ #[llamafile](https://discord.com/channels/1089876418936180786/1182689832057716778/1259341165237502035)** (1 messages): 

> - `Gemma 2 9B`
> - `Small Language Models (SLMs)`
> - `Serverless AI inference` 


- **Google 的 Gemma 2 9B 表现出色**：Google 的 [Gemma 2 9B](https://blog.google/technology/developers/google-gemma-2/?ref=unremarkable.ai) 是最近发布的一款开源语言模型，因其性能和能力而备受关注。
   - [尽管体积较小](https://www.reddit.com/r/LocalLLaMA/comments/1drxhlh/gemma_2_9b_appreciation_post/?ref=unremarkable.ai)，Gemma 2 9B 的表现可与 GPT-3.5 等大型模型相媲美甚至更优，使其非常适合在资源受限的环境中部署。
- **在 AWS Lambda 上使用 Gemma 2 进行 Serverless AI 推理**：分享了一个关于在 AWS Lambda 上使用 Gemma 2 和 Mozilla 的 Llamafile 进行 [Serverless AI 推理](https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda) 的教程。
   - 这种方法有助于在手机、PC 或本地云等低资源环境中部署 Gemma 2 9B。



**提到的链接**：<a href="https://www.unremarkable.ai/serverless-ai-inference-with-gemma-2-using-mozillas-llamafile-on-aws-lambda/">Serverless AI Inference with Gemma 2 using Mozilla&#x27;s llamafile on AWS Lambda</a>：Google 的 Gemma 2 9B 是最近发布的一款开源语言模型，在我们的社区中引起了极大关注。这款轻量级模型是 Google 开发的 Gemma 系列模型的一部分...

  

---



### **DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1259476099763798038)** (1 messages): 

> - `Experiment with base models`
> - `Hermes-2-Theta-Llama-3-70B`
> - `Llama3-DiscoLeo-Instruct-70B` 


- **以 Hermes-2-Theta-Llama-3-70B 作为 Llama3-DiscoLeo-Instruct 的基础**：一位成员建议进行一项有趣的实验，使用 **Hermes-2-Theta-Llama-3-70B** 作为基础模型来创建 **Llama3-DiscoLeo-Instruct-70B**。
- **组合模型的潜在优势**：讨论暗示了将 **Hermes-2-Theta-Llama-3-70B** 与 **Llama3-DiscoLeo-Instruct** 结合以增强性能和能力的潜在**优势**。


  

---



---



{% else %}


> 完整的频道详情已针对邮件进行截断。 
> 
> 如果您想查看完整详情，请访问此邮件的网页版本：[{{ email.subject }}]({{ email_url }})！
>
> 如果您喜欢 AInews，请[分享给朋友](https://buttondown.email/ainews)！提前感谢！

{% endif %}