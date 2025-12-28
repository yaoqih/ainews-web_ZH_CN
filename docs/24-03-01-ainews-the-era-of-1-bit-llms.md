---
companies:
- hugging-face
date: '2024-03-01T22:33:03.450029Z'
description: '**“1比特大语言模型时代”**的相关研究（包括 **BitNet b1.58** 模型）引入了一种三元参数方法。该方法在性能上可媲美全精度
  Transformer 大语言模型，同时将能耗大幅降低了 **38 倍**。这一创新有望催生针对 1 比特大语言模型优化的新缩放法则和硬件设计。AI 推特（Twitter）上的讨论热点包括
  **AGI 的社会影响**、**结合多模态模型的机器人技术**、**ResLoRA 等微调技术**，以及 **Hugging Face 在 AI 安全领域的投入**。此外，生成式
  AI 的伦理考量以及 AI 社区内的幽默话题也是备受关注的议题。'
id: 851dc9c9-9602-4a29-ba5a-d534bbf64bd1
models:
- bitnet-b1.58
original_slug: ainews-the-era-of-1-bit-llms
people:
- swyx
- levelsio
- gdb
- npew
- _akhaliq
- osanseviero
- mmitchell_ai
- deliprao
- nearcyan
- clementdelangue
title: '**1比特大语言模型时代**'
topics:
- quantization
- model-optimization
- energy-efficiency
- fine-tuning
- robotics
- multimodality
- ai-security
- ethics
- humor
---

<!-- buttondown-editor-mode: plaintext -->> 2024年2月29日至3月1日的 AI 新闻。我们为您查看了 [**356** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **22** 个 Discord 社区（**351** 个频道，**6023** 条消息）。预计节省阅读时间（按 200wpm 计算）：**577 分钟**。

Quantization（量化）最极端的表现形式是 Binarization（二值化）——即除了权重的 1 bit 之外，切掉所有其他部分。TheBloke 目前将其削减至 4 bits，但性能损失通常非常剧烈。通常情况下是这样。

[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764) 这篇论文在 [HN](https://news.ycombinator.com/item?id=39535800) 和 Discord 上引起了相当大的关注。其摘要值得仔细研读（附 swyx 的评论）：

- **最近的研究（如 BitNet）正在为 1-bit Large Language Models (LLMs) 的新时代铺平道路。**（[BitNet 论文](https://arxiv.org/abs/2310.11453) 展示了如何使用二进制 BitLinear 函数作为传统矩阵乘法的直接替代方案，从而从头开始训练 1-bit 权重，能耗降低了 38 倍，且性能具有竞争力）
- 在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中 **LLM 的每一个参数（或权重）都是三值的 {-1, 0, 1}**。
- 它在 Perplexity（困惑度）和最终任务性能方面，都能与相同模型大小和训练 Token 数量的全精度（即 FP16 或 BF16）Transformer LLM 相匹配，同时在 **Latency（延迟）、Memory（内存）、Throughput（吞吐量）和能耗方面更具成本效益**。
- 更深远的是，**1.58-bit LLM 定义了训练兼具高性能和成本效益的新一代 LLM 的新 Scaling Law 和方案**。此外，它实现了一种新的计算范式，并为设计针对 1-bit LLM 优化的特定硬件打开了大门。

我们通常会进行更全面的论文解析，但现在得去参加 Dylan Patel 的节目。更多内容请关注本周末 Latent Space 的文章。

---

**目录**

我们正在尝试移除目录，因为许多人反映它并不像预期的那样有用。如果你怀念目录，请告诉我们，否则它们将永久消失。

# PART X: AI Twitter 摘要

### AI 与机器学习创新

- **AGI 与大科技巨头：** 关于 AGI 未来社会影响的讨论是一个突出主题，人们担心少数群体会获得巨大权力。[levelsio 讨论了这一潜在的未来](https://twitter.com/levelsio/status/1763282668874010813)。
- **机器人与多模态模型：** 机器人技术正随着多模态模型的扩展而取得进展，强调了人形机器人和 AI 在现实场景中互动的未来。[gdb 谈论了为机器人扩展多模态模型](https://twitter.com/gdb/status/1763296690738720859)，而 [npew 强调了与 Figure_robot 团队的合作](https://twitter.com/npew/status/1763282241558573556)。
- **大语言模型 (LLMs)：** ResLoRA 等用于微调 LLM 的新方法表明模型效率和有效性在不断演进。[_akhaliq 展示了 Low-Rank Adaption 中的 ResLoRA 恒等残差映射](https://twitter.com/_akhaliq/status/1763328999621529946)。
- **提高 AI 安全性：** 关于模型仓库安全性的讨论强调了安全持久化方法和恶意软件扫描的重要性，以防止 LLM 的恶意使用。[osanseviero 分享了 Hugging Face 在模型安全方面的努力](https://twitter.com/osanseviero/status/1763331704146583806)。
- **日常使用中的 AI 与开发者工具：** 重点关注 AI 在提升日常便利性方面的作用，以及简化 AI 研究和应用的工具开发。[levelsio 观察到了发展中国家的现代便利设施](https://twitter.com/levelsio/status/1763298959169097755)，[AravSrinivas 谈论了包括 AI 在内的各种播客话题](https://twitter.com/AravSrinivas/status/1763285261071650856)。

### AI 研究与伦理

- **AI 中的伦理考量：** 正在审查伦理在生成式 AI 开发中的作用，反映了社区对 AI 社会影响的认识。[mmitchell_ai 分享了一篇关于生成式 AI 伦理的评论文章](https://twitter.com/mmitchell_ai/status/1763284604696629252)。
- **研究影响与认可：** 围绕研究影响的对话强调了学术界谦逊和建设性参与的必要性。[deliprao 谈到了研究者的反应](https://twitter.com/deliprao/status/1763298115195408757)。

### 梗/幽默 (Memes/Humor)

- **AI 社区中的幽默：** AI 工程社区内的笑话和梗为该领域的种种怪癖提供了轻松的氛围和评论。[nearcyan 幽默地讨论了商业决策中的时机问题](https://twitter.com/nearcyan/status/1763318561072713865)，[ClementDelangue 调侃了与 Stripe 员工的对话](https://twitter.com/ClementDelangue/status/1763350424050868547)，而 [deliprao 则表达了对聊天机器人的沮丧之情](https://twitter.com/deliprao/status/1763341490246336764)。


### 总体总结

通过 Twitter 对话反映出的 AI 和技术工程社区内的讨论，涵盖了从对 AGI 社会影响的深刻担忧，到对特定 AI 模型和优化技术的详细讨论。关于 AGI 背景下未来经济格局的辩论 ([@levelsio](https://twitter.com/levelsio/status/1763282668874010813)) 代表了对技术影响力不断扩大的重大关注。同时，关于多模态模型和机器人技术的对话 ([@gdb](https://twitter.com/gdb/status/1763296690738720859)) 反映了将 AI 与现实世界应用相结合的热情。

社区显著强调提高效率和完善 AI 方法论，ResLoRA 作为大语言模型微调的一项创新被广泛讨论 ([@_akhaliq](https://twitter.com/_akhaliq/status/1763328999621529946))，而对 StackOverflow 未来地位的担忧 ([@fchollet](https://twitter.com/fchollet/status/1763306890161992143)) 则表明了在 AI 进步背景下开发者资源格局的演变。对模型安全和伦理 AI 的好奇展示了该行业对稳健且负责任开发的重视 ([@osanseviero](https://twitter.com/osanseviero/status/1763331704146583806))。

这些讨论反映了 AI 社区广泛的兴趣范围，从深层的技术关注到社会影响，表明了该领域专业人士和爱好者之间多样化的优先级和关注领域。

---

# PART 0: 总结之总结之总结

<div><h2><strong>对 AI 巨头和模型开发的期待</strong>：</h2><ul><li><strong>Allen Institute for AI</strong> 和 <strong>OpenAI</strong> 处于 AI 进步的前沿，关于 <strong>65b AI 模型</strong> 和 <strong>GPT-6</strong> 的讨论暗示了 AI 技术的未来能力。社区对这些模型的潜力充满期待，将其与 <strong>Llama 3</strong> 等现有模型进行比较，并推测它们对 AI 研究和应用的影响​​​​。</li></ul><h2><strong>法律与伦理辩论</strong>：</h2><ul><li>Elon Musk 对 OpenAI 采取的法律行动引发了关于该组织对开放 AI 技术承诺的辩论。这一争议突显了人们对主要 AI 实体伦理和治理日益增长的担忧，强调了创新、所有权与开源原则之间复杂的辩论​​。</li></ul><h2><strong>技术创新与挑战</strong>：</h2><ul><li><strong>Flipper Zero</strong> 设备以及 AI 基础设施的进步（如 <strong>Modular MAX Developer Edition</strong>）代表了 AI 和黑客社区在硬件和工具方面的重大进展。这些讨论揭示了创新、监管与伦理黑客攻击之间持续的平衡过程​​​​。</li></ul><h2><strong>训练与量化技术</strong>：</h2><ul><li>关于训练协议的深度技术讨论，包括 <strong>tinyllama</strong>、<strong>QLoRA</strong> 和<strong>量化策略</strong>的使用，反映了 AI 社区在优化 AI 模型训练和部署方面的努力。用于微调和部署量化模型的脚本和文章交流，展示了克服 AI 模型开发中技术挑战的协作方法​​​​。</li></ul><p>这些主题表明了一个充满活力的生态系统，开发者、研究人员和爱好者正致力于突破 AI 技术的界限，应对其伦理影响，并探索创新的应用和工具。</p></div>

---

# PART 1: 高层级 Discord 总结

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 摘要

- **对 AI 巨头的期待**：关于 Allen Institute for AI 据传即将推出的 **65b AI 模型** 的传闻在社区引起了巨大反响，引发了与 OpenAI 的 LLM 的对比以及对 **Llama 3** 能力的推测。讨论包括了对此类模型潜力的见解，并分享了一个[研究链接](https://arxiv.org/abs/2401.05566)。

- **马斯克的法律行动引发辩论**：Elon Musk 对 OpenAI 提起的[法律诉讼](https://www.courthousenews.com/wp-content/uploads/2024/02/musk-v-altman-openai-complaint-sf.pdf)引发了关于该组织对开放 AI 技术承诺的争议，诉讼文件揭示了 Musk 的不满。

- **Flipper Zero：黑客的宠儿与难题**：关于 Flipper Zero 硬件的对话集中在它在涉及 NFC 和 RFID 项目中的用途，以及调试 BLE 问题的挑战。用户还对该设备的定价和感知价值发表了看法，特别是在其被禁及随后价格上涨之后 [Flipper Zero 产品页面](https://flipperzero.one/)。

- **训练领域趋向量化**：技术交流深入探讨了训练协议，包括使用 **5e-6 学习率** 对 **tinyllama** 进行的实验、**QLoRA** 的有效使用，以及训练和量化模型的序列策略。分享了一个 [Colab 脚本](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) 和一篇 [Medium 文章](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e)，以辅助微调和部署量化模型。

- **角色扮演让 LLM 更生动**：关于将角色扮演引入 LLM 是否能增强其表观智能的有趣对话，观察表明，详细的角色设定 Prompt 可能有助于模型做出与现有数据集更紧密结合的改进预测。`@maldevide` 强调了创建具有说服力的对话角色的实践。

- **复杂的模型合并方法论**：模型合并频道的讨论涉及球面线性插值 (**slerp**) 与 linear ties、diffusion 和 huggingface 测试方法，以及来自 `@alphaatlas1` 的建议，即在使用 **PEFT** 时采用 **concatenation** 而非全权重合并。

- **前沿代码协作**：**Modular MAX Developer Edition** 的发布为 AI 基础设施提供了新的可能性，而 [npm](https://www.npmjs.com/package/semantic-chunking) 上的 `semantic-chunking` 软件包承诺利用 transformers.js 和 ONNX 为 LLM 提供流式文本处理。进一步的讨论探索了优化 GPU 利用率以及使用 ONNX 的 WebAssembly 后端可能带来的性能提升。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- **GPU 在模型推理方面优于 CPU**：在 #deployment 频道中，参与者强调了 **GPU**（特别是像 **RTX 4090** 这样具有更高 VRAM 的显卡）对于运行像完整的 7B Mistral 这样的大型模型至关重要。讨论还涉及了量化模型在更大上下文中的有限性能，以及特定语言模型优于多语言模型的优点。

- **微调见解仍然有限**：#finetuning 频道出现了关于在 H100 上微调 7B 模型所需时间的询问，以及关于 Mistral 7B instruct v0.2 背后的方法和数据集的推测。然而，关于 Mistral 微调过程的详细见解仍未公开。

- **展示期待与访问咨询**：#showcase 和 #random 频道的用户对即将推出的项目预览以及如何访问 Google 的 1M context AI 表现出浓厚兴趣。一位用户推荐了来自 *Deepmind* 的联系人作为潜在的访问途径。

- **Mistral API 和 AI 模型的不确定性与问题**：在 #la-plateforme 频道，用户澄清了某些功能的缺失、API 的模型不匹配问题，以及验证错误（这些错误暗示了临时的不一致性，已通过部署修复解决）。

- **评估策略的答疑时间**：来自 #office-hour 频道的一条消息强调了预定于 **CET 时间 3 月 5 日下午 5 点**进行的关于 **评估与基准测试 (evaluation and benchmarking)** 的讨论。

- **对 Le Chat 和 CroissantLLM 的改进建议**：#le-chat 频道的参与者为 **Le Chat** 提出了各种增强建议，同时对 **CroissantLLM** 表示不满，暗示可以通过微调进行潜在改进。

- **计算资源讨论占据主导**：在多个频道中，对话围绕与计算资源相关的技术讨论展开，如 VRAM 需求、推理中 GPU 优于 CPU 的重要性，以及硬件规格（如 M2 和 M3 Macs 在计算任务中的效能）。

- **提示词与失败案例数据稀疏**：#failed-prompts 和 #prompts-gallery 频道包含了一些提到失败提示词和模型不准确的消息，但缺乏可用于分析有意义 AI 开发见解的具体数据或示例。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **GPT-3 消失了**：`@temparr` 意外地找不到他们的自定义 GPTs，但 `@openheroes` 迅速指出了它们在 [OpenAI GPTs 页面](https://chat.openai.com/gpts/mine) 的 "mine" 选项卡下的位置。
- **实战胜过证书**：在 **AI 认证 vs 经验** 的博弈中，`@navs02` 的疑问得到了 `@dezuzel` 的回应，后者主张实战 AI 能力，而 `.dooz` 则认为 Andrew Ng 的课程和 Andrej Karpathy 的 YouTube 教程是制胜组合。
- **AI 水手在电子表格海洋中航行**：`@gatorsuf83` 思考如何使用 AI 将船只数据绘制成电子表格，促使 `@.braydie` 建议使用 CSV 格式，而 `@eskcanta` 则展示了一个以 [AI 生成的 Excel 表格](https://chat.openai.com/share/0581e083-d02a-4b9a-b02e-12b7997dc32f) 形式存在的宝库。
- **数字海洋上的上传故障**：地平线上出现了一波关于文件上传故障的抱怨，`@metaldrgn` 和 `@cqoker` 等航行者面临困境，引发了关于使用限制和系统中潜在 Bug 的讨论。
- **DALL-E 3 的提示词难题**：在关于图表绘制和字符限制的讨论中，`@madame_architect` 赞扬了像 Mermaid 这样的图表工具，而 `@darthgustav` 解决了 DALL-E 3 的 JSON 字符串中花括号解析的障碍，`@solbus` 则通过 [OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3) 的指引澄清了关于提示词字符限制的模糊文档。

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **余弦困惑得以澄清**：`@pseudoterminalx` 澄清了围绕 **Cosine LR schedules** 的困惑，强调许多人将简单版本误认为是真正的振荡类型。他们提倡使用更细致的 **带有衰减的 Cosine schedule**，类似于 Pytorch 分类为 "cosine annealing with decay" 的方式。

- **Ideogram 模型——神秘还是革命？**：由前 Google 工程师设计的最新发布模型 **Ideogram.ai** 引起了成员们的兴趣。尽管缺乏实质性细节，社区对其潜力议论纷纷，并将其与 Google 的 Imagen 等其他未发布的模型进行比较。

- **审美 AI 之争**：公会讨论了 AI 生成图像中 Prompt 遵循度与审美吸引力之间的平衡。`@devilismyfriend` 指出，更好的审美有时可能需要偏离精确的 Prompt 指令。

- **协作式字幕标注计划**：分享了为大型图像数据集添加字幕的技术，包括 `@pseudoterminalx` 提议为该任务提供志愿者集群。这突显了社区在构建高质量带字幕数据集方面的努力。

- **模型训练经验分享**：公会成员交流了关于模型训练和增强策略的见解，讨论了从不同分辨率到使用 CLIP 进行文本结合的主题。有人谈到汇集 text embeddings 并将其作为 register tokens 在训练期间直接添加。

**共享的关键资源**：
  
- 讨论在 Vision Transformers 中使用寄存器的论文：[Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)
- **kopyl** 的图标生成模型，可在 [Hugging Face](https://huggingface.co/kopyl/ui-icons-256) 和 [Civitai](https://civitai.com/models/327499) 上获取
- RNN 的复兴：[Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- **LM Studio 讨论预设参与度与强制响应**：`@zorian_93363` 认为 LM Studio 的预设有些欠缺，并质疑使用 system prompts 引导 AI 响应的有效性。同时，关于 LM Studio 内部一键模型下载功能的简单性及其可能带来的不必要假设引发了辩论，`@fabguy` 警告不要添加任何可能限制用户控制的功能。

- **借助 Google Coral 提升 Pi 设备上的模型性能**：建议使用 Google Coral 加速器来增强 Raspberry Pi 和 Orange Pi 5 等低功耗设备上的模型执行，有可能为紧凑型设备带来更强大的算力。

- **硬件难题：崩溃、散热器与配置**：公会成员解决了一系列硬件问题，从 `@666siegfried666` 报告的神秘系统崩溃，到为配备 AMD Ryzen Threadripper Pro 和多个 NVIDIA GPU 的高端系统寻找合适的散热解决方案和电源。用户还分享了提高 GPU 利用率的售后策略，例如在 MSI Afterburner 中解锁电压控制。

- **对认知理解的不断追求**：公会成员交流了关于商业文档分析和摘要等任务的本地模型推荐，并关注了热门的 Huggingface 模型以及具体的 Nous-Capybara-3B-V1.9 和 MiniChat-2-3B。此外，还有关于模型性能中 MoE 数量增加导致收益递减的轻松评论。

- **从 AI 游戏到商业文档分析**：有人建议将 AI 交互转化为游戏或电视节目，以诱导 AI 提问；并就建立一个针对商业文档分析优化的强力 PC 寻求建议，尽管在提供的消息中没有推荐具体的模型。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Perplexity 表现优于 ChatGPT**：在与 ChatGPT 的对比中，用户分享了 Perplexity AI 如何提供更及时的信息，类似于 Google 的服务。文中提到了一篇来自 IEEE spectrum 的文章，阐述了 Perplexity 重塑 AI 驱动搜索工具的雄心 ([Perplexity-ai disrupts traditional chatbots](https://spectrum.ieee.org/perplexity-ai))。

- **AI 工具测试**：社区成员评估了 Perplexity 和 Copilot 的总结能力，同时测试了文件上传和提取功能，重点关注输出质量。此外，还讨论了 Copilot Pro 的 code interpreter 功能，并强调该功能也对免费用户开放。

- **API 问题与初期磨合**：API 频道的对话揭示了在使用和查阅 Perplexity API 文档时遇到的挑战和困惑，包括模型对比、性能问题，以及预定于 3 月 15 日弃用的 `pplx-70b-online`。用户被引导至[入门指南](https://docs.perplexity.ai/docs/getting-started)以及 [2024 年 2 月 API 更新日志](https://docs.perplexity.ai/changelog/api-updates-february-2024)。

- **塔可与技术的碰撞**：在分享频道中，Perplexity AI 的趣味和创新用途备受关注，包括搜索最佳塔可食谱和生成播客内容。AI 在创作肖像和音频内容方面的实力也得到了展示，体现了该平台的多功能性和创意潜力。

- **对旧版模型的保留呼吁**：在更新和模型更迭之际，用户恳请不要逐步淘汰备受青睐的 `pplx-70b-online` 模型，并讨论了其相对于 `sonar-medium-online` 等新模型的优势。分享的经验强调了对模型稳定性和可靠性能的需求。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **为初级 AI 开发者推出的备忘单**：**Foundation Model Development Cheatsheet**（基础模型开发备忘单）现已发布，该项目由 `@hailey_schoelkopf` 领导，EleutherAI 及多个机构共同参与。它为新的开源模型开发者提供了全面指南，强调了流程中经常被忽视的部分，如数据集文档和许可实践。该备忘单提供论文格式（[点击此处](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf)）和交互式网站（[点击此处](https://fmcheatsheet.org/)）。

- **揭秘排行榜**：针对排行榜上出现的 `mmlu_no_train` 用户进行了澄清，这与来自 lm-eval-harness 的自动下载有关，而非真实的用户参与。general 频道的进一步讨论指向了一些资源，例如[解释多选题归一化技术的博客文章](https://blog.eleuther.ai/multiple-choice-normalization/)，以及 `@slowturtle_p` 确认的在 lm-evaluation-harness 中使用 TensorRT 等自定义代码替换模型调用的可能性。

- **量化在模型可解释性中的作用**：关于极度量化的 LLM 的可解释性出现了推测，正如[近期一篇论文](https://arxiv.org/pdf/2402.17764.pdf)所讨论的，由于权重更简单，可能会提供新的见解。同时，Transformer 难以学习对输入变化高度敏感的函数，这可能导致其偏向于低敏感度函数，并增加了我们对这些模型学习能力的认知，详见[这篇论文](https://arxiv.org/abs/2402.09963)。

- **翻译对 LLM 性能的影响**：`@marcobella` 改进了 Lambada 数据集的翻译，使多语言模型的准确率显著提升了 10%-15%，证明了高质量翻译对模型性能的重要性。修订后的翻译可在 [Hugging Face 数据集页面](https://huggingface.co/datasets/marcob/lambada_multilingual)获取。

- **深入探讨 GPT-NeoX 和 The Pile**：**GPT-NeoX** 的基础设施需要手动设置，并且确认 **The Pile** 的验证集是均匀随机采样的，且在创建前进行了去重（deduplication）。在回答有关采样方法和规范验证集创建的问题时澄清了这些细节，但有关去重的具体细节以及相对于 **Pythia** 使用数据集的时间点尚不完全明确。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **GPT-6 准备迎接新能力**：一项关于 **GPT-6** 的专利暗示了其在 Agent 和音乐生成方面的潜在进展。然而，讨论中并未透露该专利的具体细节。

- **Gemma 7B 微调技巧**：用户间分享了一个关于如何使用 Unsloth 微调 **Gemma 7B** 模型的[视频指南](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)，并附带了引用的 Google Colab 笔记本，旨在提升模型性能。

- **1-bit 模型成为焦点**：[BitNet b1.58 论文](https://arxiv.org/abs/2402.17764)介绍了一种具有高性价比性能的 1-bit Large Language Model，因其三值（ternary）和加法特性，引发了关于硬件实现的讨论。

- **基准测试揭示 Gemma 模型异常**：讨论指出了 Google **Gemma** 模型性能中的异常现象，注意到在多个基准测试中，规模较大的 Gemma 8B 模型表现不如 Gemma 2B。

- **OpenAI Five 准备迎接人类挑战**：[OpenAI 的博客文章](https://openai.com/research/openai-five-defeats-dota-2-world-champions)详细介绍了 **OpenAI Five** 的成功，它已从击败 Dota 2 机器人发展到与人类玩家协作，预示着即将举行一场全球表演赛来测试其能力。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

- **Groq 的 AI 硬件创新吸引观众**：在[卡塔尔 Web Summit](https://youtu.be/IixoaS5ckBA)上，Groq CEO Jonathan Ross 讨论了公司在 TPU 和 LPU 技术方面的进展，其对 AI 基础设施的潜在影响引起了关注。
- **突破性的 1-Bit LLM 引发褒贬不一的反应**：[新的“三值参数”论文](https://arxiv.org/abs/2402.17764)在 [Hacker News](https://news.ycombinator.com/item?id=39535800) 上引发了辩论，该论文声称 1-bit Large Language Model 的性能可与全精度 LLM 相媲美，但其在实用性和重新训练需求方面受到了质疑。
- **Banana.dev Serverless GPUs 的事后分析**：一篇详细介绍 Banana.dev Serverless GPUs 产品兴衰的[博客文章](https://blog.erikdunteman.com/banana-pivot-unpeeled)引发了关于 AI 初创公司面临挑战的讨论，强调了 AI 领域产品市场契合度（product-market fit）的复杂性。
- **AI 产品管理讨论**：公会成员交流了管理 AI 项目的心得，其中杜克大学在 Coursera 上的 AI Product Management 专项课程以及 Fullstack Deep Learning 关于 ML 团队和项目管理的讲座被列为[推荐资源](https://www.coursera.org/specializations/ai-product-management-duke)。
- **Representation Engineering 环节激发好奇心**：关于 **Representation Engineering** 的讨论揭示了其在可控性（steerability）和对齐（alignment）中的基础作用。同时，LLM Asia Paper Club 正在规划日程以适应不同时区的成员，且将 Representation Engineering 库应用于 [Colab 笔记本](https://colab.research.google.com/)的想法受到了好评。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **优化的 Hybrid Search 亮相**：LlamaIndex 推崇一种使用 **Language Models (LLMs)** 优化 Hybrid Search 效能的创新方法，通过根据查询类型自动调整 alpha 参数，该方法已通过 [Twitter](https://twitter.com/llama_index/status/1763252392639042024) 分享。

- **连接数据类型的 RAG 架构**：LlamaIndex 正在探索将结构化数据集成到 **Retrieval Augmented Generation (RAG)** 框架中，详细见解记录在 [ClickHouseDB 博客文章](https://twitter.com/llama_index/status/1763282902358585445)中。

- **部署私有 RAG 本地系统的研讨会**：LlamaIndex CEO [@jerryjliu0](https://twitter.com/llama_index/status/1763304038442192978) 宣布了一场研讨会，展示如何结合 LlamaIndex + Tonic Validate 与 Ollama 部署**本地 RAG 系统**，旨在增强数据隐私。

- **通过 OpenLLMetry 实现 LLM 应用的可观测性**：未来的 LlamaIndex 研讨会将介绍在 LLM 应用中实现可观测性的技术，根据[此公告](https://twitter.com/llama_index/status/1763364010676900080)，重点强调了详细插桩（instrumentation）的必要性。

- **展望长上下文 RAG 系统的未来**：LlamaIndex 在 [Twitter 讨论](https://twitter.com/llama_index/status/1763620476847632744)中推测了 RAG 系统在处理如 **Gemini 1.5 Pro** 等长上下文模型时的演进，暗示了检索方法论的适配调整。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **Axolotl 问卷调整以优化用户界面**：`@caseus_` 更新了他们的 [**Axolotl 终端用户问卷**](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform)，根据社区反馈减少了必填项，旨在深入了解用户与 axolotl 的交互情况。
- **TinyBox 展现 AI 实力**：`@dreamgen` 介绍了配备六块 AMD Radeon RX 7900 XTX GPU 的 [**TinyBox** 系统](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production)，强调了其为 AI 应用提供高性价比 PetaFLOPS 级性能的潜力。
- **Sophia 和 DropBP 算法的创新**：分享了二阶优化器 [**Sophia optimizer**](https://arxiv.org/abs/2305.14342) 和旨在缩短训练时间的 [**DropBP**](https://arxiv.org/abs/2402.17812) 方法，它们分别提供了优于传统 backpropagation 和 Adam 优化方法的效率。
- **Starcoder2 获得社区关注**：围绕 **Starcoder2** 的集成和支持展开了讨论和咨询，并分享了 [GitHub 仓库](https://github.com/bigcode-project/starcoder2)，凸显了对这一新兴模型的关注及其应用价值。
- **使用 Mistral 精通丹麦语**：`@le_mess` 通过合成数据集、迭代模型训练以及 [Scandeval.com](https://scandeval.com) 基准测试，利用 **7B Mistral 模型** 在丹麦语任务中取得了与 **ChatGPT 3.5** 相当的结果，强调了针对开源商业应用的手动和自动数据清洗过程。

---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **WMMA 优化 7900xtx 上的 Tensor 操作**：[MLIR/LLVM 中 WMMA 的启用](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b) 提升了 **7900xtx** 的性能，并分享了详细的指标。`@iron_bound` 的成功展示了精度格式对大矩阵尺寸的影响。
- **消除 Triton 调试中的 TypeError**：设置 `TRITON_INTERPRET` 环境变量可以解决使用 Triton 调试器时的 `TypeError`，因为在 **Triton 3.0.0 和 2.2.0** 中，关键字 'interpret' 已被弃用。
- **Ada Lovelace GPU 与 FP8 计算限制**：对话指出，尽管提供了 **FP8 intrinsics**，但 Ada Lovelace GPU 上的实际计算受到限制，特别是缺乏 `wgmma.mma_async` 是一个显著的短板。`@drisspg` 引用了一篇探讨这些计算约束的 [PyTorch 讨论](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)。
- **引入用于高效 Attention 的 BASED 架构**：介绍了一种名为 BASED 的新型基于 Attention 的语言模型架构，详见 [研究论文](https://arxiv.org/pdf/2402.18668.pdf)，该架构承诺提高效率。此外，有人指出 Hugging Face 的 Mistral 实现具有存疑的 Attention 默认设置，在 4k context 以上可能会出现问题，正如一条 [推文](https://x.com/ph_singer/status/1763538607191527540?s=20) 和提议的 [修复 PR](https://github.com/huggingface/transformers/pull/29220) 所证明的那样。
- **Ring Attention 动态引发混乱**：`@ericauld` 和 `@jamesmel` 在使用 Ring Attention 时遇到了多个问题，包括错误的梯度和指针参数错误。查阅 lucidrains 的 [仓库历史记录](https://github.com/lucidrains/ring-attention-pytorch/commits/main/) 暗示了自定义 kernel 尝试存在问题，而 GPU 资源分配冲突则通过系统重启得到了解决。

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 摘要

**LangChain 中的序列化障碍**：`@thatdc` 在使用 **langserve** 时遇到了一个问题，即他们的 Agent 仅返回最终输出，而不返回中间步骤。正在进行的 [GitHub issue #381](https://github.com/langchain-ai/langserve/issues/381) 可能包含相关信息，但尚未提供明确的解决方案。

**遏制诈骗行为**：多个频道报告了 `@skywalker09_` 发布的内容，其中包含承诺“50 美元礼品”的可疑链接，这可能是一个潜在的**诈骗 (scam)**。

**使用 LangGraph 的股票聊天机器人**：用户 `@tarikkaoutar` 在 [YouTube 视频](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s)中展示了 LangGraph 与 YahooFinance 的集成，创建了一个多 Agent 股票分析聊天机器人。

**Endoftext 简化提示工程**：`@cabreraalex` 发布了 **Endoftext**，这是一款提供建议和测试用例的 AI Prompt 编辑器，并在 [60 秒演示](https://youtu.be/PGv5ymOhaNA)中进行了展示，可在 [Endoftext 官网](https://app.endoftext.app/)访问。

**通过 Airbyte 和 Langchain 进行数据集成**：`@andysingal` 分享的一篇文章解释了 Airbyte 与 Langchain 的结合如何改进数据集成流程，并在 [Medium 文章](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad)中进行了进一步探讨。



---



## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 摘要

- **Stripe 标记预付卡而非虚拟卡**：`@fakeleiikun` 在 OpenRouter 上通过 Google Pay 使用预付卡时遇到了 *error 402 或 error 502*；`@louisgv` 提到，虽然 Stripe Radar 可能会标记像 Discovery 这样的卡，但来自受支持银行的虚拟卡可以正常工作。
- **Helicone 与 OpenRouter 结合**：`@wise_monkey_42910` 寻求将 Helicone 与 OpenRouter 集成的帮助；`@louisgv` 提供了一个 [GitHub 上的集成示例](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts)，并引导至 [Helicone 文档](https://docs.helicone.ai/getting-started/integration-method/openrouter)。
- **Token 术语整理**：在关于带 function calling 的流式传输讨论中，`@alexatallah` 解释说 `native_tokens` 代表模型自身 Tokenizer 中的 Token，并承诺更新文档以反映现有的使用指标是针对 native tokens 的。
- **OpenRouter 聊天中消除马斯克传闻**：`@alexatallah` 回应了 `@telepathyx` 关于 Elon Musk 与 OpenRouter 竞争的猜测，澄清未来可能会考虑将 Groq（而非 Grok）添加到 OpenRouter，否定了马斯克竞争的想法。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 摘要

- **Nate 与 AI 巨头的偶遇**：在一次偶然的相遇中，`@natolambert` 遇到了 AI 泰斗 Yann LeCun，尽管他错过了邀请这位著名科学家参加播客的机会。
- **Yann LeCun 分享绿色愿景**：在这次意外的会面中，`@natolambert` 与 Yann LeCun 就共同感兴趣的话题——**绿色能源 (green energy)** 进行了深入交流。
- **播客鼓励**：在听说这次偶遇后，`@philpax` 愉快地鼓励了 `@natolambert`，暗示他在未来的邀请中会通过“魅力判定 (charisma check)”。
- **家族关系调侃**：社区成员就可能存在的 Lambert 家族联系进行了打趣，`@victory` 提议添加自定义服务器表情符号，而 `@mike.lambert` 则试图调查家族联系。
- **LeCun 的孤独感与对 RL 的怀疑**：`@natolambert` 分享了与 Yann LeCun 交谈的见解——他在推动开放 AI 方面感到孤独，以及他对强化学习 (RL) 一贯的怀疑态度，这被认为是*典型的 Yann 风格 (normal yann stuff)*。

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **为 DPR 构建高质量负样本**：`@philipmay` 推荐了一种改进 Dense Passage Retrieval 数据集的策略，即通过 Language Model 生成**故意生成的错误答案**，这可能为训练目的提供更有效的负样本。

- **DiscoLM_German_7b 性能探索**：`@mab3049` 正在寻找 **DiscoLM_German_7b 模型**的最佳设置，并反馈了在复现 Demo 性能结果时遇到的挑战。

- **微调中的 Padding 困境**：`@silicagel_64242` 提出了一个关于在微调模型时使用合适 `pad_token` 的疑问，候选方案包括 `eos_token`、`unk_token` 和特定的 `"[PAD]"` token，但尚未达成共识。

- **寻找卓越的德语 RAG**：`@bjoernwerner` 正在寻找用于 Retriever-Aggregator-Generator 应用的最有效的德语 Embedding 模型，并探索了各种**单向量和多向量 Embedding** 选项。

- **MT-Bench-X 引发德语数据集搜寻**：`@crispstrobe` 关注了难以寻觅的 MT-Bench-X 数据集，指出其采用 Apache 2.0 许可证，且根据 [arxiv.org](https://arxiv.org/pdf/2402.13703v1.pdf) 的论文显示其具有处理德语任务的潜力；讨论中还提到了 MT-Bench-DE 和经过人工改进的 [MT-Bench-TrueGerman](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman) 等替代方案，认为它们是更真实的德语基准测试资源。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

- **Claude 摒弃闲聊**：用户探索了通过设置初始返回字符来规避 Claude 默认对话式开场白的策略。参考了 [Anthropic 的重写指南](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites)，其中一种技术涉及强制 Claude 以特定字符（如 `<rewrite>`）开头，以绕过不必要的引导语。

- **本地 LLM 的热情遭遇冷遇**：`@gwthompson` 请求推荐可在本地运行并用于 Datasette 增强的最佳开源大语言模型（LLM），但社区未提供具体建议。

- **寻找简洁 C API 的无果尝试**：`@florents_` 询问是否存在具有简洁 C API 的 LLM，但对话中未出现直接的推荐。

- **通过 llama.cpp 预览 LLM**：`@agarcia_me` 指出了 llama.cpp 在 Embedding 支持方面的实用性（尽管需要 C++ 编译器），并提到打算发布一个集成 LLM Embedding 的 sqlite 扩展代码，并强调了 C API 的使用。

- **带有 C 代码演示的 Embedding 指导**：`@agarcia_me` 分享了来自 [llama.cpp/examples](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp) 的详细 C 代码片段，展示了 LLM Embedding 的实现。他强调该代码是用纯 C 编写的，支持 batch size 为 1 的操作，并指出 `llama_batch` 函数封装了核心复杂性。

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **据称 Anthropic 优于 Gemini 1.5**：用户讨论了未经证实的说法，即 **Anthropic** 在上下文长度和准确性方面优于 **Gemini 1.5**。然而，参与者尚未进行个人测试来证实这些传闻。
  
- **寻求 OpenAI 增强功能**：一位成员表示需要更多关于 OpenAI 的**资源和信息**，特别是寻求在生产环境中实施 OpenAI **codeinterpreter** 的建议。

- **System Prompts 之谜**：讨论者谈到了 **system prompts** 对模型输出的重要但往往不透明的影响。会议强调，由于模型差异和研究实验室的频繁更新，Prompt 的有效性可能并不一致。

## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

- **寻求英国 AI Engineer 职位描述**：`@peterg0093` 正在寻找符合新兴标准的英国 AI Engineer 职位描述示例。`@swyxio` 分享了 Hex AI Engineer 招聘页面 [Hex Careers](https://hex.tech/careers/ai-engineer/)，作为一个展示公司文化和角色期望的潜在模板。
  
- **AIEF 可能会采用 AI-Infra 的结构**：`@swyxio` 建议 AI Engineer Foundation 可能会从类似于 [AI-Infra](https://ai-infra.fun/) 的结构化资源组织方式中受益。

- **活动组织者在 LinkedIn 上获得认可**：`@huikang` 对其参与 2024 年 2 月 24 日活动的认可表示感谢，相关内容发布在 [LinkedIn](https://www.linkedin.com/posts/ai-eng-foundation_last-saturday-on-022424-sasha-organized-activity-7169152145336782850-_TsG) 上。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **Gemma 7B 获得微调指南**：分享了一个关于使用 Unsloth 微调 **Gemma 7B** 的 [YouTube 教程](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)，并附带了一个可访问的 [Colab notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo...)。
- **OpenCodeInterpreter 发布**：一段 [YouTube 视频](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be) 介绍了 **OpenCodeInterpreter**，这是一个旨在通过 LLM 增强代码生成的开源项目。
- **Discord 内的人才挖掘**：@.papahh 向一名公会成员发出了工作邀请，建议他们查看 DM 了解更多详情。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- **神经网络可解释性的新方法**：`@camelfacts` 介绍了一篇论文，该论文通过绘制表征生态位（representational niches），利用经济学和信息论的概念，提供了一种**解释神经网络表征的新方法**。该论文正在寻求反馈，并已通过 [此链接](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1) 分享在 LessWrong 上。



---

# 第 2 部分：频道详细摘要和链接



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1212680316624314378) (1182 条消息🔥🔥🔥): 

- **对未来 AI 发布的热情参差不齐**：用户讨论了 [Allen Institute for AI](https://www.allenai.org/) 即将发布的据称 65b 的 AI 模型，并将其预期与 OpenAI 的 LLM 进行了比较。一些人推测了未来模型如 Llama 3 的表现，并附带了 [相关材料链接](https://arxiv.org/abs/2401.05566)。

- **Musk vs. OpenAI**：围绕 Elon Musk 对 OpenAI 提起的法律诉讼展开了讨论，一份 [起诉书](https://www.courthousenews.com/wp-content/uploads/2024/02/musk-v-altman-openai-complaint-sf.pdf) 显示了 Musk 对 OpenAI 运营的不满，认为其违背了开源 AI 技术的承诺。

- **黑客与硬件讨论**：Netrve 对 Flipper Zero 表现出兴趣，用于个人硬件项目及其提供的多功能能力，包括 NFC 和 RFID 使用。

- **Flipper Zero 产品闲聊**：用户分享了使用 Flipper Zero 设备的经验，从调试 Bluetooth Low Energy (BLE) 问题到其设计引发的怀旧感。讨论了 [Flipper Zero 的价格](https://flipperzero.one/) 及其相关配件，以及禁令后明显的涨价。

- **LLM 脚本聊天**：Quantscope 询问是否有人有编写利用本地 LLM 脚本的经验，引发了关于个人项目和分享 Hugging Face 库支持等资源的讨论。

**链接提及**:

- [未找到标题](https://www.amazon.ca/Rotring-500-Drafting-Pencil-0-5/dp/B001A1TN0G/ref=mp_s_a_1_5?crid=1KAN0GXZYPQ9F&dib=eyJ2IjoiMSJ9.ewJpEfa07fgtLJha4Np31X8Mo3wtqPvYsZmIMdwfI2McYeCH4TUyT_S5Nupclflsyu9iwCKyshKCCHTvWzpwWQHLtXweM6cOljB4YeRV8KTq3p1SWhDmByy8ts0N6-88ABZ5cxQp46WfiwRb2ikqKAiC8eHBogRUTpwS9a2fe2zBbGyOn3IenrxCUwbAT0XMB_kmh-IjxXTBqwqqNwJCPQ.ufZ3NQ-gq-88LR3hGsBagrP3kEc9xvQ1w-Uod9voRv0&dib_tag=se&keywords=rotring+500+0.5&qid=1709241957&sprefix=rotring+500%2Caps%2C135&sr=8-5)): 未找到描述
- [EMO](https://humanaigc.github.io/emote-portrait-alive/): EMO: Emote Portrait Alive - 在弱条件下利用 Audio2Video Diffusion Model 生成富有表现力的肖像视频
- [Cerebras](https://www.cerebras.net/): Cerebras 是快速且轻松进行 AI 训练的首选平台。欲了解更多信息，请访问 www.cerebras.net。
- [Cat Cat Meme GIF - Cat Cat meme Funny cat - Discover &amp; Share GIFs](https://tenor.com/view/cat-cat-meme-funny-cat-cat-eating-cat-eating-chips-gif-10455465908695706650): 点击查看 GIF
- [Sad GIF - Sad - Discover &amp; Share GIFs](https://tenor.com/view/sad-gif-7523306793289960933): 点击查看 GIF
- [Sleeper Agents: Training Deceptive LLMs that Persist Through Safety Training](https://arxiv.org/abs/2401.05566): 人类能够表现出策略性的欺骗行为：在大多数情况下表现得乐于助人，但在有机会追求其他目标时，其行为会变得截然不同。...
- [Futurama Bender GIF - Futurama Bender Dance - Discover &amp; Share GIFs](https://tenor.com/view/futurama-bender-dance-gif-4195777226086506084): 点击查看 GIF
- [Thanos Perfectlybalanced GIF - Thanos Perfectlybalanced - Discover &amp; Share GIFs](https://tenor.com/view/thanos-perfectlybalanced-gif-18301221): 点击查看 GIF
- [Netflix, hungry for more growth, signals more price hikes](https://arstechnica.com/gadgets/2024/01/netflix-hungry-for-more-growth-signals-more-price-hikes/): 基础无广告方案将首先从加拿大和英国的订阅者中取消。
- [How to enable Previous Versions to recover files on Windows 10 - Pureinfotech](https://pureinfotech.com/enable-previous-versions-recover-files-windows-10/): Windows 10 的“以前的版本”功能允许你使用文件资源管理器恢复文件和文件夹，本指南将介绍如何配置它。
- [Product - Chip - Cerebras](https://www.cerebras.net/product-chip/): 未找到描述
- [p1atdev/dart-v1-sft · Hugging Face](https://huggingface.co/p1atdev/dart-v1-sft): 未找到描述
- [Tweet from Arthur Mensch (@arthurmensch)](https://x.com/arthurmensch/status/1762818733016322168?s=46&t=0D9nuNBS26GijH-DPnkpgw): 鉴于我们读到了对最新公告的一些创造性解读，在此澄清几点：- 我们仍然致力于领先的 open-weight 模型！请保持一点耐心，1.5k H100s ...
- [adamo1139 (Adam)](https://huggingface.co/adamo1139): 未找到描述
- [OpenAI's Statement SHOCK the Entire Industry! AI Riots vs "Moore's Law for Everything" by Sam Altman](https://youtu.be/JEFEJsTxPCc): 订阅我的每日 AI 通讯 🔥https://natural20.beehiiv.com/subscribe [关于 AI 的新闻、研究和教程] 链接：万物摩尔定律：https://moores.sa...
- [Moore's Law for Everything](https://moores.samaltman.com/): 我们需要设计一个拥抱这种技术未来的系统，并对构成那个世界大部分价值的资产——公司和土地——征税，以便公平地分配一些...
- [GitHub - facebookresearch/nougat: Implementation of Nougat Neural Optical Understanding for Academic Documents](https://github.com/facebookresearch/nougat): 用于学术文档的 Nougat Neural Optical Understanding 的实现 - facebookresearch/nougat
- [GitHub - vosen/ZLUDA: CUDA on AMD GPUs](https://github.com/vosen/ZLUDA): 在 AMD GPU 上运行 CUDA。通过在 GitHub 上创建账号，为 vosen/ZLUDA 的开发做出贡献。
- [adamo1139/rawrr_v2 · Datasets at Hugging Face](https://huggingface.co/datasets/adamo1139/rawrr_v2): 未找到描述

---

### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1212689925678571551) (283 messages🔥🔥): 

- **角色扮演切换的乐趣**：`@lyrcaxis` 提出了戏内/戏外（in-character/out-of-character）分离的想法，以创建更具说服力的角色扮演设置，即角色扮演者通过角色进行交流，而不是直接传达想法。
- **机器人因垃圾信息被禁言**：`@dampf` 回应了 `@mrdragonfox` 关于 tldr 的询问，告知已从频道中删除了捕获的垃圾信息。
- **角色扮演能让 LLM 变得更聪明吗？**：`@superking__` 等人讨论了让 LLM 进行角色扮演是否能让它们看起来更聪明，一些实验表明，使用 instruct-driven 提示词进行角色扮演可能会产生更好或更具体的结果。
- **角色提示词带来更好的建模**：`@maldevide` 分享了一种定义角色和使用详细提示词的广泛方法，他们认为这能让 LLM 通过紧密贴合训练数据集来更好地预测后续对话。
- **谦逊的 Midori 被修改了**：`@c.gato` 幽默地讨论了创建一个 LLM，`@lisamacintosh` 观察到一个有趣的转变：一个名为 Midori 的 LLM 在被描绘成一辆 2006 年的 Honda Civic 后，开始在句子中加入 "vroom"，展示了角色建模中富有想象力且有时出人意料的结果。

**提到的链接**：

- [maldv/conversation-cixot · Hugging Face 数据集](https://huggingface.co/datasets/maldv/conversation-cixot)：未找到描述
- [Aeala (A&#39;eala)](https://huggingface.co/Aeala)：未找到描述

  

---


### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1212853189838249994) (31 messages🔥): 

- **Tinyllama 进行学习率实验**：`@maldevide` 正在对 **tinyllama** 进行测试，使用 `5e-6` 的学习率，并计划使用 **Supervised Fine-Tuning (SFT)** 处理所有数据来调节模型。

- **使用 QLoRA 的训练策略**：`@222gate` 分享了他们使用 QLoRA 的方法，然后将 **LoRA** 与 **4-bit quantized model** 合并，具体将 adapter 设置为 "qlora"，optimizer 设置为 "adamw_bnb_8bit"。

- **模型训练与量化难题**：`@orel1212` 对训练和量化模型的正确顺序感到好奇，引发了关于是在与 base models 合并之前还是之后进行量化的讨论。`@maldevide` 提到，**QLoRA 的学习参数**应在量化之前应用回 base model。

- **发现 Colab 资源**：`@orel1212` 和 `@222gate` 分享了用于训练和微调量化模型的资源，包括 [Colab 脚本](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996) 和一篇 [Medium 文章](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e) 的链接。

- **模型预训练中的验证集困境**：`@cogbuji` 询问在原始文本领域数据上预训练模型的无监督学习中，验证集是否必要。`@maldevide` 澄清说，保留验证集可以防止评分偏差（score bias），但这并非强制性的，尤其是在数据有限的情况下。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1Xu0BrCB7IShwSWKVcfAfhehwjDrDMH5m#scrollTo=70zJf1hi0huQ)：未找到描述
- [使用 peft 和 tlr 微调 GPTQ 模型](https://gist.github.com/SunMarc/dcdb499ac16d355a8f265aa497645996)：使用 peft 和 tlr 微调 GPTQ 模型。GitHub Gist：即时分享代码、笔记和片段。
- [微调与部署 LLM：PEFT 和 GPTQ！第 2/2 部分](https://dsmonk.medium.com/training-and-deploying-of-quantized-llms-with-lora-and-gptq-part-2-2-ec7b54659c9e)：如果我们从量化模型开始会怎样？

  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1212826786367545485) (5 messages): 

- **表达感谢**：`@222gate` 对分享的信息表示感谢，尽管尚不清楚他们具体指的是哪些内容。
- **关于 Slerp 或 Linear Ties 的澄清**：`@222gate` 询问讨论的是关于 **slerp (spherical linear interpolation)** 还是仅仅是 linear ties。
- **讨论中的测试方法论**：`@alphaatlas1` 回应了 `@222gate` 的问题，澄清他们的扩散测试使用了 **dare ties**，并推测 Hugging Face 的测试利用了 **task arithmetic**。
- **PEFT 合并建议**：`@alphaatlas1` 建议 `@222gate` 在进行 PEFT 合并时尝试使用 **concatenation** ("**concat**")，并认为这比全权重合并（full weight merging）更有效。
  

---

### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1212798213577248818) (6 条消息): 

- **Modular MAX 平台正式发布**：`@dirtytigerx` 分享了 [Modular MAX Developer Edition](https://www.modular.com/blog/announcing-max-developer-edition-preview) 的发布公告。该平台旨在为 AI 构建统一且高效的基础设施，使其对所有开发者都具备可用性和可扩展性。该平台承诺赋能全球开发者，并优化 AI 硬件效率及总体拥有成本。

- **JavaScript 语义分块工具发布**：`@jparkerweb` 介绍了一个 `semantic-chunking` 包，用于在不依赖重型框架的情况下，为 LLM 工作流高效地拆分长文本。该包现已在 [npm](https://www.npmjs.com/package/semantic-chunking) 上架，并利用 transformers.js 和 ONNX 进行运行。

- **GPU 优化讨论**：在 `@jparkerweb` 分享了他们的 `semantic-chunking` 方案后，`@dirtytigerx` 提到他们使用 Rust/Neon 通过 node addon 开发了一个类似工具，以实现更好的 GPU 利用率。他们暗示 `@jparkerweb` 的包也可以通过将 ONNX 后端设置为 `wasm` 来增强 GPU 支持。

- **探索 WebAssembly 以提升性能**：`@jparkerweb` 表示有兴趣按照 `@dirtytigerx` 的建议，探索用于 ONNX 的 WebAssembly (`wasm`) 后端，以潜在地提高其 `semantic-chunking` 工具的性能和效率。

**提到的链接**：

- [Modular: Announcing MAX Developer Edition Preview](https://www.modular.com/blog/announcing-max-developer-edition-preview)：我们正在为全球构建下一代 AI 开发者平台。查看我们的最新文章：Announcing MAX Developer Edition Preview。
- [semantic-chunking](https://www.npmjs.com/package/semantic-chunking)：从长文本中语义化地创建分块（适用于传递给 LLM 工作流）。最新版本：1.0.0，最后发布于：一天前。通过运行 `npm i sema...` 开始在你的项目中使用 semantic-chunking。

---

### Mistral ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/1212670228291522612) (724 条消息🔥🔥🔥): 

- **VRAM 困惑已澄清**：`@mehdi1991_` 询问了关于运行 Gemma 7B 等模型的合适服务器以及 Python 库的 VRAM 要求，包括 `@ethux` 在内的用户帮助进行了解答，解释了 VRAM 是指 GPU 的显存，并建议在运行大语言模型 (LLMs) 和库时使用 RTX 3090。

- **潜在的 Mistral AI 应用与开源模型推测**：`@pacificprime` 询问了即将推出的 Mistral 应用，而 `@blacksummer99` 和 `@mrdragonfox` 的讨论则围绕 Mistral 发布新 open-weight 模型的推测时间，以及其与 Meta 发布 Llama3 之间可能存在的联系。

- **关于 API 滥用和防范欺诈的讨论**：`@foxalabs_32486` 和 `@chrunt` 表达的担忧强调了对 API 密钥被盗导致 API 被滥用的顾虑，这会造成财务损失，并引发了可能影响公司 API 使用政策的安全讨论。

- **微软与 Mistral 被误解的联盟关系**：`@kerunix` 对微软与 Mistral AI 之间被媒体炒作而误解的关系表示担忧。用户 `@lerela` 通过引用官方澄清说明纠正了这些误解，淡化了“联盟”这一概念。

- **克隆模型讨论与技术规范**：包括 `@i_am_dom`、`@mrdragonfox` 和 `@shaman6991` 在内的多位用户参与了关于运行 Mixtral 等克隆模型的复杂性、模型切换的细节以及检索增强生成 (RAG) 系统效率的讨论。`@mrdragonfox` 还分享了关于并发查询处理和合适硬件的技术建议。

**提到的链接**：

- [What Is Retrieval-Augmented Generation aka RAG?](https://blogs.nvidia.com/blog/what-is-retrieval-augmented-generation/)：检索增强生成 (RAG) 是一种通过从外部来源获取事实来增强生成式 AI 模型准确性和可靠性的技术。
- [Client code | Mistral AI Large Language Models](https://docs.mistral.ai/platform/client/#json-mode)：我们提供 Python 和 Javascript 的客户端代码。
- [NVIDIA Chat With RTX](https://www.nvidia.com/fr-fr/ai-on-rtx/chat-with-rtx-generative-ai/)：定制并部署您的 AI 聊天机器人。
- [vLLM | Mistral AI Large Language Models](https://docs.mistral.ai/self-deployment/vllm/)：vLLM 可以使用我们提供的 Docker 镜像部署，也可以直接从 Python 包部署。
- [Klopp Retro GIF - Klopp Retro Dancing - Discover &amp; Share GIFs](https://tenor.com/view/klopp-retro-dancing-liverpool-champions-gif-19224858)：点击查看 GIF。
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-)：微软向 Mistral AI 投资 1500 万欧元，这是一家总部位于巴黎、致力于基础模型开发的 AI 初创公司。
- [Microsoft made a $16M investment in Mistral AI | TechCrunch](https://techcrunch.com/2024/02/27/microsoft-made-a-16-million-investment-in-mistral-ai/amp/)：微软向 Mistral AI 投资 1500 万欧元，这是一家总部位于巴黎、致力于基础模型开发的 AI 初创公司。
- [openchat/openchat-3.5-0106 · Hugging Face](https://huggingface.co/openchat/openchat-3.5-0106)：未找到描述。
- [Legal terms and conditions](https://mistral.ai/terms/#terms-of-use)：使用 Mistral 产品和服务的条款和条件。
- [Pixels - Actualités, vidéos et infos en direct ](https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-et)：关于 Pixels 主题的所有新闻。查阅 Le Monde 发布的 Pixels 栏目中的所有文章、报道、直播、照片和视频。
- [Pixels - Actualités, vidéos et infos en direct ](https://www.lemonde.fr/pixels/article/2024/03/01/on-a-teste-le-chat-l-etonnant-chatgpt-a-la-francaise-de-mistral-ai_6219436_4408996.html),)：关于 Pixels 主题的所有新闻。查阅 Le Monde 发布的 Pixels 栏目中的所有文章、报道、直播、照片和视频。

---

### Mistral ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/1212686973882335263) (14 messages🔥): 

- **Mistral-7B 的查询复杂度**：`@sanipanwala` 询问 Mistral-7B-v0.1 是否能处理复杂的 SQL 查询，并提供了一个涉及 `SELECT` 语句以及 `INNER JOIN` 和 `OUTER APPLY` 的示例。`@tom_lrd` 确认 Mistral 模型可以尝试任何查询，并提供了一个示例结构来测试性能。
- **特定 SQL 查询构建**：在后续讨论中，`@sanipanwala` 询问了如何自定义 SQL 查询以从表中选择特定字段，`@tom_lrd` 通过提供一个复杂的 SQL 查询示例，演示了如何向 Mistral 描述该请求。
- **数学数据集 Embedding 咨询**：`@aky6691` 向小组询问了有关 Embedding 数学数据集的经验，但未说明数学类型或预期用途。`@tom_lrd` 要求澄清“数学数据集的 Embedding”具体是指什么。
- **图像 Prompt 生成效果参差不齐**：`@h0rizons` 分享了他们的经验，认为 Mistral 的 Large 模型在为 AI 图像生成器创建 Prompt 时表现不如 GPT-4，并指出在明确指示其不要重复艺术 Prompt 时效果会有所改善。
- **Mistral 定价结构讨论**：`@jb_5579` 询问了 Mistral Large 和 Next 的 API 费率，随后 `@mrdragonfox` 分享了一个指向 Mistral 定价的完整链接。这确认了各种模型的成本，包括 Mistral Large 的输入价格为每 1M tokens $8，输出价格为每 1M tokens $24，详见 [Mistral 定价页面](https://docs.mistral.ai/platform/pricing/)。

**提到的链接**：

[Pricing and rate limits | Mistral AI Large Language Models](https://docs.mistral.ai/platform/pricing/): 按需付费

  

---


### Mistral ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/1212670570878214174) (153 messages🔥🔥): 

- **推理中 GPU 优于 CPU**：`@ethux` 提到，对于模型推理，**GPU** 起着至关重要的作用，而 **CPU** 仅在一定程度上有重要性。他们还讨论了 GPU 显存（VRAM）的重要性，指出要运行完整的 7B Mistral 模型，需要配备足够 VRAM 的 **RTX 4090**。
  
- **量化模型的缺点**：在 `@frigjord`、`@_._pandora_._` 和 `@ethux` 的对话中，他们对量化模型的性能表示不满，特别是在处理长上下文和编程任务时，确认了量化模型不如完整版本。
  
- **辩论特定语言模型的优劣**：`@frigjord` 和 `@mrdragonfox` 辩论了创建专注于特定编程语言（如 JS）的模型是否会优于多语言模型。`@mrdragonfox` 建议训练语言的多样性可以带来更好的泛化能力。

- **硬件的性价比与效率**：用户 `@frigjord` 和 `@sublimatorniq` 讨论了使用 M2 和 M3 Macs 等高规格硬件的成本和收益。`@frigjord` 指出了 M2 的速度优势，而 `@sublimatorniq` 分享了他使用 96GB 配置的经验。

- **数据清洗仍是重大挑战**：在一段冗长的讨论中，`@mrdragonfox` 强调了数据清洗所需的巨大精力，这构成了模型训练数据准备工作中的绝大部分，并分享了他用于处理此类任务的个人硬件配置。

**提到的链接**：

- [starling-lm](https://ollama.com/library/starling-lm)：Starling 是一个通过 AI 反馈强化学习（RLAIF）训练的大语言模型，专注于提高聊天机器人的帮助性。
- [Jurassic Park GIF - Jurassic Park World - Discover & Share GIFs](https://tenor.com/view/jurassic-park-world-velociraptor-clever-gif-25116052)：点击查看 GIF。
- [Tags · mixtral](https://ollama.com/library/mixtral/tags)：由 Mistral AI 发布的具有开放权重的优质混合专家（MoE）模型。

  

---

### Mistral ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/1212733431931346974) (18 条消息🔥): 

- **询问微调时长**：`@atip` 询问了在 H100 上完全微调一个 7B 模型所需的小时数，`@mrdragonfox` 回复称这取决于数据集的大小。
- **探寻 Mistral 7B 的微调秘诀**：`@pteromaple` 寻求关于 Mistral 7B instruct v0.2 微调方法和数据集的细节，但 `@mrdragonfox` 确认这些信息尚未公开，并对涉及的质量衡量和数据准备进行了推测。
- **揭开微调之谜**：`@pteromaple` 和 `@mrdragonfox` 讨论了从 Mistral 7B instruct v0.1 到 v0.2 显著提升的技术细节，并将其性能与 Google 的 Gemma 7B IT 进行了对比。
- **通过 API 进行微调**：`@claidler` 询问了是否可以通过 API 微调闭源模型，`@ethux` 指出 API 响应中的一个线索表明模型微调是未来的目标。
- **顶级微调模型查询**：`@kunpengguo` 询问 `mixtral-8x7b-instruct-v0.1` 是否是最好的 Mistral 微调模型，`@mrdragonfox` 肯定了其地位。
  

---


### Mistral ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/1212823146282098788) (2 条消息): 

- **对即将开展项目的关注**：`@akshay_1` 表示有兴趣预览 `@patagonia50` 计划开展的项目，并希望在可能时获得**抢先体验 (sneak peek)**。
  

---


### Mistral ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/1212767292195078204) (2 条消息): 

- **访问 Google 的 1M 上下文 AI**：用户 `@j673912` 询问如何访问 Google 的 1M context AI。`@dawn.dusk` 建议联系 *Deepmind* 的相关人员。
  

---


### Mistral ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/1212685615989002250) (21 条消息🔥): 

- **关于 Chat 可用性的困惑**：`@paul.martrenchar_pro` 澄清某项功能目前在通用范围内**不可用**，仅存在于 **Le Chat** 中。

- **Mistral API 模型不匹配查询**：`@ls_rageux_` 对请求 **mistral-small** 时 API 返回 **open-mixtral-8x7b** 表示困惑，这似乎揭示了模型处理中的差异。

- **Mistral 系统角色支持澄清**：`@lerela` 确认在 **Mistral Large** 中，不支持在用户角色之前添加 **system/assistant** 或 **assistant** 角色的前缀提示词。

- **Mistral 中系统角色的变通方法**：`@zerosignal_x` 询问了 medium 等模型中的系统/助手配对，而 `@not__cool` 和 `@skisquaw` 讨论了替代方法，例如在 **Mistral Large** 的用户提示词中使用**系统角色 (system role)**。

- **关于 Mistral 工具调用和响应的澄清**：`@januify` 寻求关于向 **Mistral Large** 发送请求时响应体中缺少 **`tool_calls`** 的澄清。`@lerela` 解释说，即使设置了 `tool_choice: "any"`，是否使用工具也由模型决定，并请求提供更详细的示例以便调查。

- **Mistral API 中的 ValidationError 和 Python 客户端不匹配**：`@proffessorblue` 报告了一个与 **ChatCompletionResponse** 相关的 ValidationError，这可能表明 Mistral API 与 Python 客户端之间存在暂时的不一致。`@lerela` 承认了**短暂的部署不一致**并已修复，随后 `@proffessorblue` 进一步指出需要更新 API 规范文档。
  

---


### Mistral ▷ #[office-hour](https://discord.com/channels/1144547040454508606/1192781286008422441/1212716795262402570) (1 条消息): 

- **关注评估策略日程**：`@sophiamyang` 宣布下一次 Office Hour 将于 **CET 时间 3 月 5 日下午 5 点**举行，重点讨论**评估和基准测试 (evaluation and benchmarking)**。团队渴望与参与者讨论各种评估和基准测试方法。
  

---

### Mistral ▷ #[le-chat](https://discord.com/channels/1144547040454508606/1211692704363323453/1212691647226454027) (212 条消息🔥🔥): 

- **在 CroissantLLM 中寻求 Mistral 的特质**：`@tom_lrd` 对 **CroissantLLM** 表示失望，认为它缺乏 **Mistral** 的能力。他们建议使用法语-英语 Hermes 数据集进行 finetuning 可能会有帮助，但对潜在的改进仍持怀疑态度。

- **Unsloth 快速且高效的 Finetuning**：`@foxalabs_32486` 分享了 [Unsloth 仓库](https://github.com/unslothai/unsloth)，强调其声称的 finetuning 速度快 5 倍且节省 60% 内存，而 `_._pandora_._`、`@sublimatorniq` 和 `@foxalabs_32486` 讨论了性能提升是否如广告所言那样显著。

- **Le-Chat 增强建议纷至沓来**：改进 **Le Chat** 的多项建议包括**实时 token 计数**、**设计调整**、**图像输入**以及**“NEW CHAT”按钮调整**。`@sophiamyang` 征求了反馈，引发了包括 `@_._pandora_._`、`@foxalabs_32486` 和 `@sublimatorniq` 在内的用户的热烈讨论。

- **使用 Mistral 的输出构建 Fine-tuning 数据集**：`@dedded___` 询问了使用 **Mistral Large** 创建数据集的可行性，`@mrdragonfox` 澄清说，虽然小规模数据集是一个选择，但要与大型模型竞争将是一项艰巨的任务。

- **Mistral AI 模型更新的澄清**：在关于 **Mistral 模型**的反复讨论中，`@lifeverygoode` 寻求确认模型 `78x7` 是否将保持开源，`@ethux` 确认了其开源状态，并提到将发布新模型而不是更新现有模型。

**提到的链接**：

- [Endpoints and benchmarks | Mistral AI Large Language Models](https://docs.mistral.ai/platform/endpoints/#benchmarks-results)：我们提供五个不同的 API 端点来提供具有不同价格/性能权衡的生成模型，以及一个用于嵌入模型的嵌入端点。
- [Why 2024 Will Be Not Like 2024](https://medium.com/@unravelingentertainment/why-2024-will-be-not-like-2024-8799121ee791)：在不断发展的技术和教育领域，一股革命性的力量正准备重塑我们学习、思考和……的方式。
- [Unsloth update: Mistral support + more](https://unsloth.ai/blog/mistral-benchmark#Benchmark%20tables)：我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构的模型的 QLoRA 支持！我们添加了滑动窗口注意力（sliding window attention）、初步的 Windows 和 DPO 支持，以及……
- [GitHub - jondurbin/airoboros: Customizable implementation of the self-instruct paper.](https://github.com/jondurbin/airoboros)：self-instruct 论文的可定制实现。- jondurbin/airoboros
- [GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth)：快 5 倍、省 60% 内存的 QLoRA finetuning。通过在 GitHub 上创建账户为 unslothai/unsloth 的开发做出贡献。

---

### Mistral ▷ #[failed-prompts](https://discord.com/channels/1144547040454508606/1212715819159654451/1212715907898671166) (10 条消息🔥): 

- **失败定义不明确**：`@notan_ai` 的评论含糊其辞，暗示可能存在失败的 prompt，但未提供失败场景的具体信息。
- **半途而废的数学之谜**：`@blueaquilae` 幽默地提到了一个与数学相关的失败，“数学，在 large chat 上完成了一半（双关语）”，但未提供 prompt 或失败的细节。
- **Prompt 失败确认无果**：`@blacksummer99` 提到 *Mistral next on le chat* 在特定 prompt 上失败，但未给出 prompt、预期输出或模型输出的细节。
- **检测到日期差异**：`@aiwaldoh` 对一个未命名实体的成立年份不一致表示担忧，询问是否是 "Fondée en 2016?!"，同时暗示这可能与特定网站有关。
- **网页与成立年份**：`@aiwaldoh` 补充说，虽然提到了一个引用 2023 年的网页，但在没有更多上下文的情况下，问题似乎仍未解决。
- **执着于发现**：`@_._pandora_._` 认可了 `@aiwaldoh` 在寻找差异根源方面的奉献精神，赞扬了他们的努力，`@aiwaldoh` 对此给予了肯定回复。

---

### Mistral ▷ #[prompts-gallery](https://discord.com/channels/1144547040454508606/1212717054302625843/1212717273610063902) (5 messages): 

- **提示词分享空间发布**：`@sophiamyang` 发起了 **prompts-gallery** 频道，邀请成员使用特定的格式（列出模型、提示词和输出）分享他们最优秀的提示词。

- **发布的消息不明确**：`@akshay_1` 仅发布了 "DSPy"，缺乏上下文且未遵循频道的提示词分享格式。

- **对 SudoLang 的好奇**：`@notan_ai` 对 "SudoLang" 表现出兴趣，但似乎对该频道的用途感到困惑。

- **格式不规范的提示词贡献尝试**：`@blacksummer99` 两次尝试提交名为 "Mistral next le chat" 的提示词，但未提供模型、提示词和输出等必要详情。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1212743318388088852) (44 messages🔥): 

- **GPTs 丢失问题**：`@temparr` 反映其所有的自定义 GPTs 都消失了。`@openheroes` 迅速引导他们在 [OpenAI GPTs 页面](https://chat.openai.com/gpts/mine) 的 "mine" 标签页下找回。

- **AI 认证与实战经验之争**：年轻开发者 `@navs02` 咨询了 AI 认证相关事宜，`@dezuzel` 回应时强调了真实 AI 案例比认证更重要，而 `.dooz` 则推荐了 **Andrew Ng** 的免费课程和 **Andrej Karpathy** 的 YouTube 教程，以进行实操学习并提升简历竞争力。

- **报告 Bug 赏金漏洞**：用户 `@l0k1_b24` 询问如何报告漏洞并获取赏金。`@solbus` 将其引向 [OpenAI 安全信息页面](https://chat.openai.com/.well-known/security.txt)，`@aminelg` 提醒他们在报告前应阅读 Bug 赏金计划的完整说明。

- **Lexideck Professional 操纵 CSS 代码**：`@beanz_and_rice` 赞赏了 `@darthgustav.` 的 dexideck 及其网站，后者将其归功于 Lexideck Professional 的创作，并澄清相关的 GitHub 账号可能并不代表他们本人。

- **讨论 AI 的真实性**：在一次哲学性的讨论中，`@drinkoblog.weebly.com` 质疑了人造产物的真实性，这引发了关于 "真实" 与 "人造" 定义的讨论，`@aminelg` 参与了讨论，`@eskcanta` 则引用了关于合成细菌的内容，并链接至 [《卫报》的一篇文章](https://www.theguardian.com/science/2019/may/15/cambridge-scientists-create-worlds-first-living-organism-with-fully-redesigned-dna)。

- **AI 与电子表格协作**：`@gatorsuf83` 询问如何使用 AI 将船只数据整理到电子表格中，`@.braydie` 建议采用 CSV 或 Markdown 表格的方式，并提供了高效引导 GPT 的策略。`@eskcanta` 分享了一个成功的测试案例，展示了使用 AI 生成的直接可用的 Excel 下载链接：[AI 生成的 Excel 表格](https://chat.openai.com/share/0581e083-d02a-4b9a-b02e-12b7997dc32f)。

**提到的链接**：

[全球首个具有完全重新设计 DNA 的活体生物问世](https://www.theguardian.com/science/2019/may/15/cambridge-scientists-create-worlds-first-living-organism-with-fully-redesigned-dna)：研究人员创造了改变后的合成基因组，这一举措具有潜在的医学价值。

  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1212706826852892743) (78 条消息🔥🔥): 

- **GPT 知识文件处理与网页浏览的困惑**：包括 `@darthgustav.` 和 `@yami1010` 在内的用户讨论了 GPT 是否能“读取”大型知识文件，以及使用 Python 进行搜索是否会禁用网页浏览功能。`@yami1010` 分享了[截图](https://cdn.discordapp.com/attachments/)，暗示在网页搜索功能方面存在误导性行为，引发了关于 AI 透明度的讨论。

- **OpenAI Discord 面临上传问题**：多位用户（包括分享具体经历的 `@metaldrgn` 和 `@cqoker`）报告了上传文件（尤其是图像和数据文件）时遇到的问题，导致出现错误消息且上传仅能间歇性成功。这引发了对潜在使用限制（usage caps）的担忧，并有人建议这可能是一个影响上传功能的更广泛 Bug。

- **关于文件上传限制的误解**：`@cqoker` 和 `@darthgustav.` 强调了围绕文件上传上限的困惑，重点在于使用限制是指总上传量，还是专门针对 GPT 知识文件的上传。这引发了反复讨论，试图澄清这些限制及其适用性。

- **讨论了年度使用上限但未明确**：`@cqoker` 和 `@darthgustav.` 就潜在的 10GB 使用上限进行了讨论，但无法确定这是指终身、每日还是其他时间范围，导致对该政策的进一步不确定。 

- **对透明度和模型更新的担忧**：包括 `@darthgustav.` 和 `@cqoker` 在内的用户对话反映了对缺乏清晰文档的担忧，以及由于模型不断更新而难以理解当前状态，以及这如何影响用户在使用 GPT 文件上传及其他功能时的体验。
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1212701829348200538) (108 条消息🔥🔥): 

- **寻求神秘列表的帮助**：`@remi1054` 询问了一个他们很想找到的列表；`@madame_architect` 提议在喝完早咖啡后将最新版本上传到 aiempower GitHub。
- **关于 Diagramming as Code 的讨论**：`@madame_architect` 分享了她对 Diagramming as Code 的热爱，提到 Mermaid、Mathplotlib 和 PlantUML 等工具彻底改变了她创建图表的工作流程。
- **发现 DALL-E 3 解析器的小瑕疵**：`@darthgustav` 幽默地讲述了解决 DALL-E 3 解析器中一个初级开发错误的过程，该错误导致解析器无法解释 JSON 字符串中的标准花括号。
- **微调聊天机器人响应策略**：在一次详细讨论中，包括 `@beanz_and_rice`、`@darthgustav` 和 `.braydie` 在内的用户讨论了增强 ChatGPT 和 DALL-E 提示词（Prompts）创意和有效性的策略，例如调用“创意过程（Creative Process）”或使用“chain-of-thought”推理。
- **DALL-E 文档差异引发沮丧**：在 `@darthgustav` 纠结于过时文档后，`@solbus` 引用了 DALL-E-3 提示词的正确字符限制，这引发了对参考资料不一致的短暂沮丧。

**提到的链接**：

[What's new with DALL·E-3? | OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3)：未找到描述

  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1212701829348200538) (108 条消息🔥🔥): 

- **等待 GitHub 仓库更新**：`@remi1054` 询问了一个列表的位置，`@madame_architect` 承诺在喝完早咖啡后将最新更新上传到 AIempower GitHub 仓库。
- **利用 ChatGPT 辅助绘图**：`@madame_architect` 分享了使用 Mermaid, mathplotlib 和 PlantUML 进行代码绘图的经验，而 **Darthgustav.** 提到在一次测试中发现了与转义字符相关的 DALL-E 3 JSON 字符串解析错误。
- **使用 AI 导航大型文档**：`@enkai3526` 寻求关于如何编写 Prompt 以让 ChatGPT 在大型文档中有效搜索相关文本的建议，这引发了关于使用 Code Interpreter 和基于关键词搜索方法的讨论。
- **引导 ChatGPT 进行顺序问答**：`@etheraland1way` 寻求帮助创建一个可以处理长文本及其后问题的 Prompt，`@eskcanta` 提供了一个简单的解决方案，即在文本后紧跟每个问题以获得结构化的回复。
- **DALL-E 3 Prompt 限制澄清**：**Darthgustav.** 发起了关于 DALL-E 3 Prompt 字符限制的讨论，`@beanz_and_rice` 最初对误解的限制提出了质疑，随后 `@solbus` 引用 OpenAI 官方文档进行了澄清。

**相关链接**：

[What's new with DALL·E-3? | OpenAI Cookbook](https://cookbook.openai.com/articles/what_is_new_with_dalle_3)：未找到描述

  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1212737859447160842) (295 条消息🔥🔥): 

- **Cosine LR Schedule 争议**：`@pseudoterminalx` 表达了挫败感，认为大多数人谈论的 "Cosine LR schedule" 实际上是他所认为的简化版本，既没有真正的振荡，也缺乏对峰值间距的控制。他区分了自己的 Cosine schedule 方法，该方法具有振荡特性并允许此类控制，PyTorch 将其标记为 "cosine annealing with decay"（带衰减的余弦退火）。
  
- **Ideogram 模型引发好奇**：频道内对 Ideogram.ai 表现出浓厚兴趣，这是由 `@pseudoterminalx` 介绍的一位前 Google 工程师发布的模型。该模型承诺将脱离现有架构，但细节较少，引发了对其效能以及 Google Imagen 等类似未发布模型质量的推测。

- **AI 标注质量讨论**：`@pseudoterminalx` 和 `@thejonasbrothers` 等用户辩论了 AI 模型在遵循 Prompt 和创建美学图像方面的差异。讨论中，`@devilismyfriend` 观察到，保持美感往往意味着不能严格遵守 Prompt。

- **大型图像集标注的协作努力**：`@pseudoterminalx` 和 `@thejonasbrothers` 讨论了创建高质量标注数据集的技术，`@pseudoterminalx` 提议可以访问一个志愿者集群来为大型图像集生成标注。

- **关于模型训练和增强的反思**：包括 `@thejonasbrothers`、`@pseudoterminalx` 和 `@chad_in_the_house` 在内的多位成员交流了模型训练策略的技巧和技术，包括使用数据增强（augmentations）、不同分辨率训练，以及将 CLIP 等工具的文本整合到训练中。提到了池化文本嵌入（pooling text embeddings）并将其作为 Register Tokens 添加，以及关于如何在模型训练中最佳利用有限的 CLIP Tokens 的讨论。

**相关链接**：

- [Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588)：Transformer 近期已成为学习视觉表征的强大工具。在本文中，我们识别并表征了监督和自监督特征图中的伪影...
- [panopstor/nvflickritw-cogvlm-captions · Datasets at Hugging Face](https://huggingface.co/datasets/panopstor/nvflickritw-cogvlm-captions)：未找到描述
- [ptx0/photo-concept-bucket · Datasets at Hugging Face](https://huggingface.co/datasets/ptx0/photo-concept-bucket)：未找到描述

  

---

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1212736552933789766) (23 messages🔥): 

- **图标生成模型发布**：用户 `@kopyl` 展示了一个用于生成图标的新 **SOTA 模型**，他在训练上投入了 2000 美元，在商业化尝试失败后决定将其公开。该图标模型以及使用说明和合作邀请可以在 [Diffusers by Kopyl](https://huggingface.co/kopyl/ui-icons-256) 找到。

- **RNN 焦点**：`@twoabove` 分享了一篇复兴 **Recurrent Neural Networks (RNNs)** 的论文链接，引发了关注和怀旧，讨论了一种新的线性递归架构，该论文[已在 arXiv 发布](https://arxiv.org/abs/2402.19427)。

- **论模型输入简化的优点**：在关于模型训练数据图像格式的对话中，`@nodja` 指出使用简单 BMP 格式的优势在于**避免了解码的复杂性**，否则会浪费计算资源。

- **模型蒸馏中的对比学习**：`@jh0482` 寻求关于**语言模型蒸馏学习**的论文信息，并质疑当目标是连续空间时使用 Contrastive Learning 的效果。

- **RNN 复兴的调侃**：提到 **RNNs** 时，`@thejonasbrothers` 幽默地表达了对该架构的热爱，并打趣地将这种期待拟人化为等待他们的“循环救世主”。

**提到的链接**：

- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)：循环神经网络 (RNNs) 具有推理速度快且在长序列上扩展效率高的特点，但难以训练且难以扩展。我们提出了 Hawk，一种具有门控线性递归的 RNN，...
- [I Can See It Timothee Chalamet GIF - I Can See It Timothee Chalamet Paul Atreides - Discover &amp; Share GIFs](https://tenor.com/view/i-can-see-it-timothee-chalamet-paul-atreides-dune-i-can-visualize-it-gif-18400807)：点击查看 GIF
- [kopyl/ui-icons-256 · Hugging Face](https://huggingface.co/kopyl/ui-icons-256)：未找到描述
- [UI icons - v1.0 | Stable Diffusion Checkpoint | Civitai](https://civitai.com/models/327499)：用于生成图标的 SOTA 模型。动机：我自费 2000 美元训练了这个模型。由于无法将其商业化，所以我决定分享给...

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1212670791691542538) (151 messages🔥🔥): 

- **LM Studio 预设与强制响应**：`@zorian_93363` 觉得 LM Studio 的预设有些空洞，并思考如何使用 system prompt 来强制 Assistant 的响应以特定字符串开头。
- **在低功耗设备上运行模型**：`@zorian_93363` 回复了全栈区块链开发者 `@n3kosenpai`，建议 Google Coral 加速器与 Raspberry Pi 等系统兼容，并能增强 Orange Pi 5 等设备上的模型潜力。
- **超快速 AI 聊天机器人引发关注**：`@pierrunoyt` 分享了 Groq 超快速 AI 聊天机器人的链接（未包含失效链接），而 `@nullt3r` 发现其在 [Mouser](https://eu.mouser.com/ProductDetail/BittWare/RS-GQ-GC1-0109?qs=ST9lo4GX8V2eGrFMeVQmFw%3D%3D) 上的售价高达 2 万欧元，并指出它只有 230MB 的 "RAM"。
- **LM Studio 中的模型执行问题**：`@barnley` 报告在尝试从 Huggingface 下载模型或使用搜索选项时，LM Studio 出现网络错误；`@heyitsyorkie` 建议检查潜在的互联网访问问题，如公司网络限制或 ISP/国家的封锁。
- **在应用中用 LM Studio 替换 OpenAI API**：`@veryvanya` 寻求在应用配置中将 OpenAI key 替换为 LM Studio 服务器的指导；`@heyitsyorkie` 提供了一个示例，说明如何设置 OpenAI 客户端的 `base_url` 以指向本地 LM Studio 服务器，并建议将 `api_key` 设置为 `"not-needed"`。

**提到的链接**：

- [no title found](http://192:168:0:100:1234/v1",)：未找到描述
- [GroqChat](https://groq.com/)：未找到描述
- [Continue](https://continue.dev/)：未找到描述
- [MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF · Hugging Face](https://huggingface.co/MaziyarPanahi/dolphin-2.6-mistral-7b-Mistral-7B-Instruct-v0.2-slerp-GGUF)：未找到描述
- [mradermacher/miquliz-120b-v2.0-i1-GGUF · Hugging Face](https://huggingface.co/mradermacher/miquliz-120b-v2.0-i1-GGUF)：未找到描述
- [vGPU Nvidia Tesla M10 32 GB RAM GDDR5 PCIe 3.0 x16 64 virtuelle Benutzer  | eBay](https://www.ebay.de/itm/126344098433)：未找到描述
- [NVIDIA Tesla K80 Dual GPU 24GB PCI-E Computing Accelerator - 699-22080-0200-511](https://www.gekko-computer.de/NVIDIA-Tesla-K80-Dual-GPU-24GB-PCI-E-Computing-Accelerator-699-22080-0200-511.html)：这是经过我们技术团队测试的二手商品。其技术和外观状态均良好。
- [nVidia Tesla M60 Dual GPU 16GB PCI-E Computing Accelerator - 900-2G402-0010-000](https://www.gekko-computer.de/nVidia-Tesla-M60-Dual-GPU-16GB-PCI-E-Computing-Accelerator-900-2G402-0010-000.html)：这是经过我们技术团队测试的二手商品。其技术和外观状态均良好。
- [Add support for StarCoder2 by pacman100 · Pull Request #5795 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/5795)：这个 PR 做了什么？增加了对最近发布的 StarCoder 2 模型的支持。

  

---


### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1212749005117984789) (13 messages🔥): 

- **让 AI 通过提问进行交互**：用户 `@pwrreset` 概述了让 AI 提问的两种策略：通过 prompt 引导用户，或者将交互转变为游戏或电视节目格式。
- **高性能 PC 寻求处理业务文档的 AI**：用户 `@redcloud9999` 询问在其高配机器（14900k, 192GB RAM, 1x 4090, Windows 11）上分析和编写业务文档的最佳配置。提供的消息中未推荐具体模型。
- **本地模型推荐**：用户 `@heyitsyorkie` 建议 `@redcloud9999` 下载并测试 LLM，推荐搜索 "TheBloke" 发布的 GGUF 量化版本。`@coachdennis.` 也建议查看 Huggingface 上的热门模型以获取最新且合适的选择。
- **寻找总结方案**：用户 `@tay2win` 为短期记忆系统寻求擅长总结的数据集和模型推荐，最初使用 phi-2 但发现效果不理想。用户 `@drawless111` 推荐了多个可以尝试的模型，包括 Nous-Capybara-3B-V1.9 和 MiniChat-2-3B，并建议降低小模型的 temperature 设置以改善结果。
- **关于模型专家混合的疑问**：用户 `@goldensun3ds` 询问为什么增加 Mixture of Experts 的数量通常不会增强模型性能，尽管预期应该会。该问题基本未得到正面回答，除了 `@tay2win` 提出的一个幽默比喻：厨师太多反而坏事。
  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1212866535366983720) (127 条消息🔥🔥): 

- **排除棘手的硬件崩溃故障**：`@666siegfried666` 报告了频繁的系统崩溃，且未留下任何错误日志或转储文件 (dumps)。他们讨论了各种潜在原因，如 RAM 问题、Wi-Fi 网卡（`AX200 - 网络适配器向驱动程序返回了无效值`），甚至是 PSU 线缆问题。尽管进行了广泛的硬件测试，包括 Memtest86+ 和 PSU 电压测量，确切原因仍然不明。用户 `@wolfspyre` 建议启动进入 Linux 系统作为诊断步骤，以确定是硬件问题还是驱动程序问题。

- **使用 MSI Afterburner 提升 GPU 利用率**：`@goldensun3ds` 在 MSI Afterburner 中解锁电压控制后，运行 Dolphin 2 6 Mixtral 7B Q3KM 模型的 GPU 利用率显著提高，使用双 RTX 4060 Ti GPU 达到了约每秒 15 个 token。

- **潜在的 LLM 高端系统配置**：用户 `@razdnk` 提出了一个用于语言模型工作的系统构建方案，包括华硕 Pro WS WRX90E-SAGE SE 主板、AMD Ryzen Threadripper Pro 7965WX 以及多张 NVIDIA 3090 GPU。他们正在寻求支持此类高端配置的 CPU 散热器、机箱和电源的建议。

- **用于散热管理的开放式机架配置**：`@nink1` 和 `@heyitsyorkie` 讨论了多 GPU 配置在空气冷却方面面临的挑战，并建议使用 1x 转接线 (risers) 和开放式机架或服务器机架来管理热量。`@heyitsyorkie` 提供了一个指向 [Emil Wallner 的 ML 装备](https://www.emilwallner.com/p/ml-rig)的链接，作为组装高性能 ML 硬件的有价值参考。

- **显卡组合与超频讨论**：用户分享了关于不同显卡配置的经验和咨询。`@wilsonkeebs` 询问关于 NVIDIA 4090 与 3090 混装的问题，`@ben.com` 和 `@heyitsyorkie` 建议不要对 ML 装备进行水冷，因为这会带来复杂性和维护负担。

**提到的链接**：

- [James Bond 007 GIF - James Bond 007 Voodoo - Discover &amp; Share GIFs](https://tenor.com/view/james-bond-007-voodoo-must-be-the-gif-13955810)：点击查看 GIF
- [Warhammer40k Angron GIF - Warhammer40k Angron Primarch - Discover &amp; Share GIFs](https://tenor.com/view/warhammer40k-angron-primarch-angry-ultra-angry-gif-17080819)：点击查看 GIF
- [Helluva Boss Helluva GIF - Helluva Boss Helluva Loona Helluva Boss - Discover &amp; Share GIFs](https://tenor.com/view/helluva-boss-helluva-loona-helluva-boss-blitzo-eat-this-gif-25976859)：点击查看 GIF
- [(4k) RTX 3090*4! It is a Luxury in Dreams](https://m.youtube.com/watch?v=fdtAOPyZ9z8)：这台电脑最初想安装风冷散热。后来因为原装显卡太厚无法安装，于是……
- [How I built a €25K Machine Learning Rig](https://www.emilwallner.com/p/ml-rig)：如何规划、购买、构建和存放你的 2-10 GPU 机器学习服务器和 PC。
- [NVIDIA Tesla K80 Dual GPU 24GB PCI-E Computing Accelerator - 699-22080-0200-511](https://www.gekko-computer.de/NVIDIA-Tesla-K80-Dual-GPU-24GB-PCI-E-Computing-Accelerator-699-22080-0200-511.html)：这是经过我们技术团队测试的二手商品，在技术和外观上均处于完美状态。
- [Amazon.com: StarTech.com PCI Express X1 to X16 Low Profile Slot Extension Adapter - PCIe x1 to x16 Adapter (PEX1TO162) : Electronics](https://www.amazon.com/gp/aw/d/B0039XPS5W/)：未找到描述
- [Pro WS WRX90E-SAGE SE｜Motherboards｜ASUS Global](https://www.asus.com/motherboards-components/motherboards/workstation/pro-ws-wrx90e-sage-se/)：华硕工作站主板专为 AI 训练、深度学习、动画或 3D 渲染领域的专业人士设计。具有可扩展的显卡、存储、令人印象深刻的连接性和可靠性……

---

### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1212808339847651398) (7 条消息): 

- **视觉模型下载说明**：`@hypocritipus` 询问了未来是否可以在 LM Studio 内部下载支持 Llava 的模型（包括 Vision Adapter）。他们分享了[可用模型列表](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1)以及发布说明中的[演示视频](https://x.com/LMStudioAI/status/1734640355318944190?s=20)。
  
- **双击下载模型**：`@jedd1` 回复了 `@hypocritipus`，解释称目前下载 Vision Adapter 和主模型需要两个独立的操作，LM Studio 内部目前还没有一键下载的解决方案。

- **对一键下载模型的疑虑**：`@fabguy` 评论了为仓库提供多种选项的复杂性，并表示担心一键下载功能可能会导致 LM Studio 做出不必要的假设，从而可能掩盖用户的选择。

**提到的链接**：

- [Vision Models (GGUF) - a lmstudio-ai Collection](https://huggingface.co/collections/lmstudio-ai/vision-models-gguf-6577e1ce821f439498ced0c1)：未找到描述
- [来自 LM Studio (@LMStudioAI) 的推文](https://x.com/LMStudioAI/status/1734640355318944190?s=20)：数企鹅可能很有挑战性 🧐🐧 LM Studio 0.2.9 新功能：🎉 本地和离线 Vision Models！在此演示中：由 @NousResearch 提供的虽小但令人印象深刻的 Obsidian Vision 3B。

  

---


### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/) (1 条消息): 

1sbefore: 是的，我也认为在仅用于此目的的 .py 文件中没有配置（conf）是不太常见的。
  

---



### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1212707066167296030) (166 条消息 🔥🔥): 

- **Perplexity AI 对比 ChatGPT**：针对 `@marshmodem` 的提问，`@bartleby0` 解释说 Perplexity 更像 Google，能提供最新信息，而 ChatGPT 则不同，并分享了一篇文章链接以供深入了解。
- **Perplexity 与 Copilot 摘要对比**：`@jaicraft` 测试了 Perplexity 和 Copilot 的摘要能力，结论是两项服务都能提供令人满意的结果，尽管 Copilot 在生成较长摘要时可能需要更多的 Prompt 引导。
- **Pro 版本的文档上传测试**：`@jaicraft` 和 `@dailyfocus_daily` 讨论了在 Perplexity Pro 和 Copilot Pro 上进行的文档上传和摘要提取测试，探索了输出的质量和效率。
- **Code Interpreter 集成**：`@dailyfocus_daily` 和 `@jaicraft` 交流了 Copilot Pro 的 Code Interpreter 功能，该功能也对免费用户开放。
- **AI 搜索引擎曝光**：`@.nohler` 分享了来自 [IEEE spectrum](https://spectrum.ieee.org/perplexity-ai) 的一篇文章，内容涉及 Perplexity AI 与 ChatGPT 等传统聊天机器人的方法对比，强调了 Perplexity 致力于创建 AI 驱动的搜索工具。

**提到的链接**：

- [总结此 PDF](https://copilot.microsoft.com/sl/jD0bsv95Qd2)：这是我使用 Microsoft Copilot（全球首个 AI 驱动的回答引擎）得到的答案。选择查看完整答案或亲自尝试。
- [总结此 PDF](https://copilot.microsoft.com/sl/iGdav3ZxB4S)：这是我使用 Microsoft Copilot（全球首个 AI 驱动的回答引擎）得到的答案。选择查看完整答案或亲自尝试。
- [Perplexity.ai 扭转 Google 局势，颠覆 SEO 信条](https://spectrum.ieee.org/perplexity-ai)：AI 搜索领导者将 Meta 构建的智能与初创公司的拼搏热情相结合
- [检索增强生成 (RAG) 研究：2017-2024](https://scalingknowledge.substack.com/p/rag)：RAG 文献综述，包括：REPLUG, Fusion-in-Decoder, KNN-LM, RETRO, FLARE, HyDe, SILO, WebGPT, Toolformer, Self-RAG, GRIT 等
- [Perplexity - AI Companion](https://chrome.google.com/webstore/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo)：浏览时随时提问
- [Perplexity - AI Search](https://chrome.google.com/webstore/detail/perplexity-ai-search/bnaffjbjpgiagpondjlnneblepbdchol)：升级你的默认搜索引擎

  

---

### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1212675530273464330) (9 条消息🔥): 

- **科技赋能周二塔可日**：`@bonhart5` 分享了一个关于**最佳塔可食谱**的 [Perplexity AI 查询](https://www.perplexity.ai/search/Best-taco-recipe-rA8ta05ATtabuvA91gKLVg)。
- **AI 播客创新**：`@_paradroid` 发布了一个 [48 小时 AI 播客提示词与结果](https://www.perplexity.ai/search/You-will-act-hEljiMC4SqWMacvlhk4Njw)，展示了 AI 如何生成播客内容。
- **AI 肖像创建**：`@dailyfocus_daily` 链接到了一个使用 AI 生成的 [EMO 表情肖像搜索](https://www.perplexity.ai/search/EMO-Emote-Portrait-oCaUsYWbS2CFU_Qr8ig5SQ)。
- **驾驶时的 AI 音频内容**：`@_paradroid` 讨论了 AI 生成音频的质量，提到 Perplexity 与 ElevenLabs 内容的结合在驾驶时**像播客一样令人愉悦**。相关的音频内容可以在 [Community-Projects](https://www.perplexity.ai/search/Fully-review-Richard-2GzIaJMnTw6Lo5WZFaAhDQ) 中找到。
- **AI 探索时事**：正如 `@_paradroid` 的[分享链接](https://www.perplexity.ai/search/TOPIC-The-wildfires-MAUe_J7rRYaEdeIxdSbA.Q)所示，Perplexity 的 AI 探索了**山火**话题。
  

---


### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1212673291576287242) (51 条消息🔥): 

- **关于订阅和仅使用 API 的困惑**：用户 `@monish0612` 询问如何仅订阅 API 而非 Pro 订阅。`@mares1317` 提供了一份详尽的 [Perplexity API 入门指南](https://docs.perplexity.ai/docs/getting-started)，其中包括注册信用卡和生成 API key 的步骤。

- **模型对比及可用性担忧**：`@icelavaman` 告知 `pplx-70b-online` 将于 3 月 15 日弃用，引发了关于 `sonar-medium-online` 与 `pplx-70b-online` 模型的争论。`@thedigitalcat` 和 `@lazysucker` 等用户因质量原因更青睐 `pplx-70b-online`，导致了对模型性能的讨论以及不要过早淘汰该模型的请求。

- **`sonar-medium-online` 性能问题**：包括 `@tob1724` 和 `@brknclock1215` 在内的多位用户报告了 `sonar-medium-online` 模型的异常行为，如回答不完整和缺乏时间感知能力。用户分享了不同的经历，并尝试了各种 system prompts 来缓解这些问题。

- **API 文档的请求与澄清**：`@jeffworthington` 等用户遇到了 OpenAPI 定义的问题，而 `@tom_primozic` 寻求逆向工程网站 WebSocket 协议的替代方案。`@jeffworthington` 和 `@yury.zem.` 在 API key 身份验证和免费额度可用性方面也遇到了挑战。

- **最新模型更新与用户反馈**：`@mares1317` 分享了一个[变更日志链接](https://docs.perplexity.ai/changelog/api-updates-february-2024)，详细说明了某些模型的淘汰以及新模型（如 `sonar-small-chat`）的引入。反馈仍在继续，用户讨论了 `pplx-70b-online` 和 `sonar` 模型在准确性和时效性方面的优劣及可靠性。

**提到的链接**：

- [pplx-api 入门指南](https://docs.perplexity.ai/docs/getting-started)：您可以使用 HTTPS 请求访问 pplx-api。身份验证涉及以下步骤：首先访问 Perplexity API 设置页面。注册您的信用卡以开始使用。此步骤将……
- [Phi Hoang (@apostraphi) 的推文](https://x.com/apostraphi/status/1762870577444847964?s=20)：先发布，因为已经足够好了，然后再改进。
- [2024 年 2 月 API 更新](https://docs.perplexity.ai/changelog/api-updates-february-2024)：发布我们的最新模型。我们很高兴宣布推出最新的 Perplexity 模型：sonar-small-chat 和 sonar-medium-chat，以及它们的搜索增强版本 sonar-small-online……
- [不仅仅是 OpenAI 套壳：Perplexity 转向开源](https://thenewstack.io/more-than-an-openai-wrapper-perplexity-pivots-to-open-source/)：Perplexity CEO Aravind Srinivas 是 Larry Page 的大粉丝。然而，他认为自己找到了一种不仅能与 Google 搜索竞争，还能与 OpenAI 的 GPT 竞争的方法。

### Eleuther ▷ #[announcements](https://discord.com/channels/729741769192767510/794042109048651818/1212781444720627783) (1 条消息): 

- **Foundation Model Development Cheatsheet 发布**：`@hailey_schoelkopf` 宣布发布 **Foundation Model Development Cheatsheet**（基础模型开发速查表），这是一份为新进开源模型开发者创建的指南，由 EleutherAI 及多个机构的成员共同贡献。该速查表旨在协助开发者走通整个开源模型开发流程，并关注了诸如数据集文档和许可实践等讨论较少的领域。
- **速查表是对开源模型增长的回应**：该资源的创建源于权重开放（open weights）新模型的显著增长，特别是 Pythia 模型套件以及 LLM360 的 Amber 和 AI2 的 OLMo 等项目的发布。该倡议旨在为开源模型开发领域提供更多切入点。
- **访问 Foundation Model Development Cheatsheet**：这一综合资源可以通过 [paper format](https://github.com/allenai/fm-cheatsheet/blob/main/app/resources/paper.pdf) 阅读，也可以通过 [interactive website](https://fmcheatsheet.org/) 进行探索。更多见解可参考配套的 [blog post](https://blog.eleuther.ai/fm-dev-cheatsheet/) 和 [Twitter thread](https://twitter.com/AiEleuther/status/1763219826602901518)。
  

---


### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1212679303720992768) (70 条消息🔥🔥): 

- **Mamba 序列分类咨询**：`@_michaelsh` 询问是否存在用于序列分类的预训练 Mamba 模型，`@frazermc` 澄清说虽然目前可能不存在，但在预训练 Checkpoint 之上训练一个分类头（classification head）是可行的。
- **自动下载混淆排名**：`@ad8e` 和 `@ilovescience` 澄清，排行榜上的用户 `mmlu_no_train` 似乎与 lm-eval-harness 的自动下载有关，而非真实的用户参与。
- **Harness 评估方法查询**：`@slowturtle_p` 询问了 lm-evaluation-harness 中归一化准确率分数的计算方法，`@stellaathena` 向其推荐了一篇关于[多项选择归一化的详细博客文章](https://blog.eleuther.ai/multiple-choice-normalization/)。
- **Harness 自定义代码替换**：`@maya_liv` 询问是否可以在 lm-evaluation-harness 中用自定义代码替换模型调用，`@slowturtle_p` 根据其在 TensorRT 上的个人经验确认这是可行的。
- **LLM 预训练 Loss 尖峰讨论**：`@staticpunch` 在 LLM 预训练期间遇到了异常的 Loss 尖峰，并得到了 `@lucaslingle`、`@cubic27` 等人的全面反馈，建议这可能是由于学习率过高或 Data Loader 恢复问题等因素导致的，并提出了多种优化策略，如更改随机种子或使用 Lion 等不同的 Optimizer。

**提及的链接**：

- [Multiple Choice Normalization in LM Evaluation](https://blog.eleuther.ai/multiple-choice-normalization/)：在 GPT-3/Neo/J 等自回归 LLM 上评估多项选择任务有多种方法。本文阐述了当前主流的归一化方法。
- [A Large Batch Optimizer Reality Check: Traditional, Generic...](https://openreview.net/forum?id=E9e18Ms5TeV)：近期提出了 LARS 和 LAMB 优化器，用于使用大 Batch Size 更快地训练神经网络。LARS 和 LAMB 在 Heavy-ball 的更新规则中加入了逐层归一化……
- [Oogway Master Oogway GIF - Oogway Master Oogway Kung Fu Panda - Discover & Share GIFs](https://tenor.com/view/oogway-master-oogway-kung-fu-panda-gif-26485559)：点击查看 GIF。
- [How does Groq LPU work? (w/ Head of Silicon Igor Arsovski!)](https://www.youtube.com/watch?v=WQDMKTEgQnY)：成为 Patreon 赞助者：https://www.patreon.com/theaiepiphany 👨‍👩‍👧‍👦 加入我们的 Discord 社区：https://discord.gg/peBrCpheKE。我邀请了 Groq 的芯片负责人 Igor Arsovski……
- [Issues · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/issues/978)：一个用于语言模型 Few-shot 评估的框架。- Issues · EleutherAI/lm-evaluation-harness

  

---


### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1212694151859019796) (77 条消息🔥🔥): 

- **模型崩溃之谜**：`@kramakek` 报告称一个 70B LLM 在持续预训练期间发生了崩溃，但训练 Loss 并未出现相应尖峰，表现为 Benchmark 指标下降和文本伪影。讨论指出潜在原因可能是灾难性遗忘（catastrophic forgetting），而 `@kramakek` 澄清其学习率为 1.5e-5，比其初始预训练率低 10 倍。

- **基础模型（Foundation Model）辩论升级**：`@aaron_wtr` 质疑了在生物学中使用“基础模型（foundation model）”一词的恰当性，引发了关于该术语歧义及其法律影响的讨论。`@valiant` 推测“foundation model”可能会成为一个法律术语，而 `@xylthixlm` 澄清说，行政命令中的“双重用途（dual-use）”意味着潜在的军事应用。

- **ResLoRA 增强 LoRA**：`@jckwind` 关注了 ResLoRA，这是一个增强的低秩自适应（low-rank adaptation）框架，旨在提高大型语言模型（LLM）的训练和推理效率。一些社区成员质疑 ResLoRA 的必要性，`@power_cookie` 对反向传播路径长度问题表示不确定，而 `@xylthixlm` 则解释了跳跃连接（skip connections）附近梯度流的改进。

- **下一波高效模型浪潮**：`@trre` 介绍了 BitNet b1.58，这是一种新型的 1-bit 三进制权重 LLM，声称在性能上可与全精度 LLM 媲美，但更具成本效益。`@honolouloute` 分享了一篇关于 Hyena 和 Mamba 等新兴模型的论文，而 `@random_string_of_character` 讨论了关于激活稀疏性（activation sparsity）的论文以及 PowerInfer 研究人员发布的最终检查点（checkpoints）。

- **关于数字世界模拟的思考**：`@fairy8767` 转述了 bGPT 的概念，这是一种专为下一字节预测（next byte prediction）设计的模型，用于模拟数字操作，声称在文本、音频和图像方面能与专用模型相媲美。在 `@jckwind` 分享的一段讲座引用中，Geoffrey Hinton 反思了大脑中神经活动的时间尺度，激发了在模型中为短期记忆实现可变学习率的想法，但 `@thooton_` 指出 Transformer 缺乏循环机制（recurrence），不支持这种基于 Agent 的架构。

**提到的链接**：

- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)：循环神经网络（RNN）具有快速推理能力，并在长序列上具有高效的扩展性，但它们难以训练且难以扩展。我们提出了 Hawk，一种具有门控线性循环的 RNN，...
- [KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：高效地服务大型语言模型（LLM）需要将许多请求批处理在一起，以降低每个请求的成本。然而，存储注意力键（keys）和值（values）以避免重复计算的 KV Cache...
- [ProSparse: Introducing and Enhancing Intrinsic Activation Sparsity within Large Language Models](https://arxiv.org/abs/2402.13516)：激活稀疏性（Activation sparsity）是指激活输出中存在大量贡献较弱的元素。作为使用 ReLU 激活函数的模型的一种普遍属性，它已经...
- [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)：最近的研究表明，基于注意力的语言模型在召回（recall）方面表现出色，即能够将生成内容建立在之前语境中出现的 Token 之上。然而，基于注意力的模型的效率...
- [ReLU$^2$ Wins: Discovering Efficient Activation Functions for Sparse LLMs](https://arxiv.org/abs/2402.03804)：稀疏计算通过动态跳过非活跃神经元的计算，为低资源场景下的大型语言模型（LLM）推理提供了一个极具吸引力的解决方案。虽然传统...
- [RNNs are not Transformers (Yet): The Key Bottleneck on In-context Retrieval](http://arxiv.org/abs/2402.18510)：本文研究了循环神经网络（RNN）和 Transformer 在解决算法问题背景下表示能力的差距。我们专注于理解 RNN 是否...
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)：最近的研究（如 BitNet）正在为 1-bit 大型语言模型（LLM）的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...
- [Trajectory Consistency Distillation](http://arxiv.org/abs/2402.19159)：潜在一致性模型（LCM）将一致性模型扩展到潜在空间，并利用引导一致性蒸馏技术在加速文本生成图像方面取得了令人印象深刻的性能...
- [LeoLM: Igniting German-Language LLM Research | LAION](https://laion.ai/blog/leo-lm/.)：我们自豪地推出 LeoLM（**L**inguistically **E**nhanced **O**pen **L**anguage **M**odel）...
- [Beyond Language Models: Byte Models are Digital World Simulators](https://arxiv.org/abs/2402.19155)：传统的深度学习往往忽略了字节（bytes），这是数字世界的基本单位，所有形式的信息和操作都以二进制格式进行编码和处理。受成功的启发...

- [Consistency Trajectory Models: Learning Probability Flow ODE Trajectory of Diffusion](http://arxiv.org/abs/2310.02279): Consistency Models (CM) (Song et al., 2023) 以样本质量为代价加速了基于分数的扩散模型采样，但缺乏一种自然的方式在质量和速度之间进行权衡。为了解决这一限制...
- [Paper page - ResLoRA: Identity Residual Mapping in Low-Rank Adaption](https://huggingface.co/papers/2402.18039): 未找到描述
- [Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/#learning-concrete-scores-with-score-entropy): 未找到描述
- [“What&#39;s wrong with LLMs and what we should be building instead” - Tom Dietterich - #VSCF2023](https://youtu.be/cEyHsMzbZBs?si=8iY9GdeDK6XSLxHN): Thomas G. Dietterich 是俄勒冈州立大学计算机科学名誉教授。他是机器学习领域的先驱之一。他曾...
- [Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution](https://arxiv.org/abs/2310.16834): 尽管扩散模型在许多生成建模任务中表现出色，但在自然语言等离散数据领域表现不佳。至关重要的是，标准的扩散模型...
- [GitHub - louaaron/Score-Entropy-Discrete-Diffusion: Discrete Diffusion Modeling by Estimating the Ratios of the Data Distribution (https://arxiv.org/abs/2310.16834)](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion): 通过估计数据分布比例进行离散扩散建模 (https://arxiv.org/abs/2310.16834) - louaaron/Score-Entropy-Discrete-Diffusion
- [Language Modeling by Estimating the Ratios of the Data Distribution | Aaron Lou](https://aaronlou.com/blog/2024/discrete-diffusion/): 未找到描述
- [Tweet from Aaron Lou (@aaron_lou)](https://fixupx.com/aaron_lou/status/1763242384958386306): 与 @chenlin_meng @StefanoErmon 一同发布 Score Entropy Discrete Diffusion (SEDD)。SEDD 挑战了自回归语言范式，在困惑度和质量上击败了 GPT-2！Arxiv: https://arxiv...

  

---


### Eleuther ▷ #[scaling-laws](https://discord.com/channels/729741769192767510/785968841301426216/1212728547677052958) (3 messages): 

- **关于 GIF 制作的简短交流**：`@kyo_takano` 提到了使用 `imageio`，大概与制作 GIF 动画有关。`@.the_alt_man` 表示惊讶，询问它是否是用 `imageio` *制作* 的。
- **对图像处理的好奇**：`@karatsubabutslower` 插话表示 "++ curious"，显示出对 `imageio` 讨论的兴趣。
  

---


### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1212869551084015616) (21 messages🔥): 

- **量化对模型可解释性的潜力**：`@jstephencorey` 询问了 1/1.58 bit 语言模型的可解释性，引用了[最近的一篇论文](https://arxiv.org/pdf/2402.17764.pdf)，暗示量化模型可能更具可解释性。`@woog` 指出，由于论文较新且尚未发布代码，目前可能还没有关于其可解释性的研究工作。

- **可解释性研究需要复现**：针对极度量化的 LLM 是否更具可解释性的问题，`@woog` 建议在研究其可解释性之前，复现相关模型是必要的第一步。

- **探索 Transformer 的学习敏感度**：`@dashiell_s` 分享了一篇[论文](https://arxiv.org/abs/2402.09963)，讨论了 Transformer 在学习对输入敏感的函数时的困难，导致其偏向于低敏感度，这可能解释了某些可学习性的局限性。

- **基于敏感度的函数学习理论受到好评**：`@norabelrose`、`@quintinpope` 和 `@karatsubabutslower` 对 `@dashiell_s` 强调的论文表示热赞，认可其在理解 Transformer 学习能力方面的潜在贡献。

- **将敏感度与理论计算机科学联系起来**：`@stellaathena` 详细阐述了该论文见解的重要性，将敏感度与理论计算机科学的复杂度度量联系起来，并指出低度函数对应于低敏感度。

**提到的链接**：

[Why are Sensitive Functions Hard for Transformers?](https://arxiv.org/abs/2402.09963): 实证研究已经确定了 Transformer 的一系列可学习性偏差和局限性，例如在学习计算简单的形式语言（如 PARITY）方面存在持久的困难，以及...

  

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1212693457211105310) (15 messages🔥): 

- **任务修改进度更新**：`@asuglia` 回应了 `@981242445696221224` 的提醒，并告知他们已确定正在进行的任务中需要修改的主要区域，但由于其他优先事项，编程方面的更改有所推迟。

- **Lambada 数据集增强翻译**：`@hailey_schoelkopf` 宣布 `@946388490579484732` 改进了 Lambada 数据集的翻译，质量超过了机器翻译。他们打算将其添加到 evaluation harness 中，并引用了 [Hugging Face 数据集页面](https://huggingface.co/datasets/marcob/lambada_multilingual)。

- **多语言翻译的质量问题**：`@marcobella` 指出了 Lambada 数据集在多种语言的机器翻译中存在的问题，包括错误的标点和空格。他们还提到在新的翻译中增加了荷兰语和葡萄牙语（巴西）。

- **手动验证显示性能提升**：在手动检查翻译后，`@marcobella` 发现翻译质量显著影响了模型性能，改进翻译后，多语言模型的准确率提高了 10%-15%。

- **尝试使用 GPT-4 翻译但已放弃**：`@marcobella` 曾打算使用 GPT-4 翻译文档，但由于部分文档触发了使用条款违规，不得不放弃该方法，并对这些案例进行了人工翻译。

- **关于多答案基准测试的咨询**：`@pbevan1` 为其 EQ-bench 的实现寻求单个 prompt 对应多个答案的任务示例。`@hailey_schoelkopf` 建议将 truthfulqa_mc2 作为潜在参考。

**提到的链接**：

- [marcob/lambada_multilingual · Datasets at Hugging Face](https://huggingface.co/datasets/marcob/lambada_multilingual)：未找到描述
- [GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench/tree/main_v2_1)：大型语言模型情感智能基准测试 - EQ-bench/EQ-Bench

  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1212783352684478514) (6 messages): 

- **关于 GPT-NeoX 基础设施的澄清**：`@triggerhappygandhi` 澄清说 **GPT-NeoX** 的容器需要预先设置，因为 NeoX 不做任何基础设施假设，仅提供用于多节点运行的 Slurm 脚本。

- **关于 The Pile 验证集的咨询**：`@pietrolesci` 询问了有关验证集如何从 **The Pile** 数据集中采样的细节，好奇它是按来源分层采样还是均匀采样。

- **确认 Pile 验证集采用均匀采样**：针对 `@pietrolesci` 的提问，`@hailey_schoelkopf` 分享了 **The Pile** 论文中的一段话，确认验证和测试数据都是随机均匀采样的，尽管关于上/下采样相对于验证集创建的具体时间点仍不清楚。

- **关于去重和验证集的细节**：`@hailey_schoelkopf` 告知 `@pietrolesci`，**The Pile** 论文中描述的去重过程发生在创建其验证集之前，并指出 **Pythia** 中使用的去重数据集没有规范的验证集。
  

---

### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1212725947778994177) (10 条消息🔥): 

- **GPT-6 猜测升温**：`@0xevil` 提到了一项关于 GPT-6 的专利，暗示它可能与 Agent 和音乐生成有关，尽管未提供具体细节。

- **Gemma 7B 微调指南**：`@pradeep1148` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)，解释了如何使用 Unsloth 微调 Gemma 7B 模型，并链接到了相关的 Google Colab。

- **Vocaloid 创意应用**：`@everyoneisgross` 成功使用口袋 MIKU Vocaloid 合成器创建了一个文本转语音系统，能够将英语句子转换为 Vocaloid 音标和 SysEx 命令。

- **Elon Musk 对阵 OpenAI**：`@mautonomy` 发布消息称，据报道 Elon Musk 已对 OpenAI 和 Sam Altman 提起诉讼，指控他们违反了保持非营利性质的创始协议。

- **Bittensor 注册困扰**：`_terps` 正在寻求 Bittensor 注册脚本方面的帮助，目前在获取低廉注册费方面遇到困难。

**提到的链接**：

- [来自 X News Daily (@xDaily) 的推文](https://fxtwitter.com/xDaily/status/1763464048908382253)：突发：Elon Musk 已对 OpenAI 和 Sam Altman 提起违约诉讼。该诉讼指控 Altman 等人背叛了 OpenAI 创立时关于保持非营利性质的协议...
- [使用 Unsloth 微调 Gemma 7B](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)：我们将了解如何使用 Unsloth 微调 Gemma 模型 https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollT...
- [OpenCodeInterpreter](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be)：OpenCodeInterpreter 是一套开源代码生成系统，旨在弥合大语言模型（LLM）与复杂专有系统之间的差距...

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1212746283467546684) (11 条消息🔥): 

- **1-bit Large Language Models 即将到来**：用户 `@deki04` 在发现 [BitNet b1.58](https://arxiv.org/abs/2402.17764) 论文后表示惊讶，该论文开启了 1-bit Large Language Models 的新纪元，声称其性能可与全精度模型媲美且更具成本效益。`@max_paperclips` 强调了其由于三值（ternary）和加法特性在硬件实现方面的潜力。
- **Scaling law 争论引发 Nous 研究员关注**：`@sherlockzoozoo` 提到了同一篇 [BitNet b1.58 论文](https://arxiv.org/abs/2402.17764) 中的 `Multiplicative scaling law`（乘法缩放法则），并将其与随模型规模增大而表现不佳的加法缩放进行了对比。
- **Large Language Models 基准测试引发好奇**：`@tarruda` 分享了一个包含真实世界测试的 [LLMs 新基准测试](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html)。针对 Nous Research 模型进行了额外的基准测试，可以在 [YouTube 视频对比](https://www.youtube.com/watch?v=IH2htfsciO4) 中查看。
- **Orca-Math 攻克数学文字题**：链接了一篇关于 [Orca-Math](https://arxiv.org/abs/2402.14830) 的论文，其中较小的语言模型在 GSM8K 基准测试中达到了 80% 以上的准确率，为解决问题的有效性提供了新策略。
- **WeightWatcher 检测 LLMs 中的过拟合**：`@charlesmartin14` 分享了关于 WeightWatcher 项目的博客文章，该项目利用 Double Descent（双重下降）概念帮助检测微调后 LLMs 中的过拟合，并附带了工具链接 [weightwatcher.ai](https://weightwatcher.ai)。

**提及的链接**：

- [Orca-Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830)：数学文字题求解长期以来被认为是小语言模型（SLMs）的一项复杂任务。最近的一项研究假设，要达到 80% 以上的准确率所需的最小模型尺寸...
- [Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models](https://arxiv.org/abs/2402.19427)：循环神经网络（RNNs）具有推理速度快且在长序列上高效缩放的优点，但难以训练且难以扩展。我们提出了 Hawk，一种具有门控线性递归的 RNN...
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)：最近的研究（如 BitNet）正在为 1-bit Large Language Models (LLMs) 的新纪元铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...
- [My benchmark for large language models](https://nicholas.carlini.com/writing/2024/my-benchmark-for-large-language-models.html)：未找到描述
- [Describing Double Descent with WeightWatcher](https://calculatedcontent.com/2024/03/01/describing-double-descent-with-weightwatcher/)：Double Descent (DD) 让统计学家、计算机科学家和深度学习从业者感到惊讶，但它在 80 年代的物理学文献中就已为人所知：虽然 DD 可以……
- [Mistral Large vs GPT4 - Practical Benchmarking!](https://www.youtube.com/watch?v=IH2htfsciO4)：➡️ 一键微调与推理模板：https://github.com/TrelisResearch/one-click-llms/ ➡️ Trelis Function-calling 模型（包括 OpenChat 3.5）：http...
- [GitHub - microsoft/azure-openai-dev-skills-orchestrator: Building a set of semantic kernel skills to act as a virtual developer team](https://t.co/1VYs4RU3x8)：构建一套语义内核技能，以充当虚拟开发团队 - microsoft/azure-openai-dev-skills-orchestrator

---

### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1212707717358026832) (89 条消息🔥🔥): 

- **Dota2 战场上的 AI 对决**：`@afterhoursbilly` 分享了一篇 [OpenAI 博客文章](https://openai.com/research/openai-five-defeats-dota-2-world-champions)，强调了 **OpenAI Five** 如何从击败机器人转向在 Dota 2 中与人类合作。OpenAI Five 准备进行一次全网范围的展示，以发现其优势和可利用性。
- **使用 GitHub Actions 进行教科书级控制**：`@thilotee` 宣布在 GPT4All 上为 Notre-Hermes-2-Mistral-7B-DPO 模型开启了一个 [pull request](https://github.com/nomic-ai/gpt4all/pull/2054/files)，寻求关于 system prompts 的建议，并解决了代码库中句子结束 token 化（end-of-sentence tokenization）的变化。
- **苹果暗示进军生成式 AI**：`@teknium` 推测 Tim Cook 在 2023 年关于苹果将在 GenAI 领域取得突破性进展的声明，可能最终只是一个在移动设备上运行的 3B 模型，这引发了关于 Siri 潜在改进的讨论。
- **谷歌的 Gemma 阵营出现麻烦？**：包括 `@teknium` 在内的多位用户注意到谷歌 Gemma 模型的性能异常，在各种基准测试中，像 Gemma 8B 这样较大的模型表现反而不如 Gemma 2B 等较小的模型。
- **RAG-narok 进化**：`@gabriel_syme` 可能已经破解了进化 RAG 模型的最后一步，致力于在无监督的情况下生成有趣的问题，以便在知识库中进行导航。

**提到的链接**：

- [Maxime Labonne (@maximelabonne) 的推文](https://fxtwitter.com/maximelabonne/status/1763262504883380462?s)：看起来 Gemma-7b 在 AGIEval、GPT4All 和 Bigbench 上的表现实际上不如 Gemma-2b。我以前从未见过这种情况，这个模型真的很奇怪。有什么想法吗？ 🤗 Gemmalpaca-7B: https://huggingf...
- [Maxime Labonne (@maximelabonne) 的推文](https://fxtwitter.com/maximelabonne/status/1763262504883380462?s=20)：看起来 Gemma-7b 在 AGIEval、GPT4All 和 Bigbench 上的表现实际上不如 Gemma-2b。我以前从未见过这种情况，这个模型真的很奇怪。有什么想法吗？ 🤗 Gemmalpaca-7B: https://huggingf...
- [Daniel Han (@danielhanchen) 的推文](https://x.com/danielhanchen/status/1763613620909580505?s=12&t=qi3AsKtzHSMXVDOdzD4icQ)：也许我发现了另一个 Gemma 的 bug？#Gemma 团队有人能确认 Gemma 使用的是 approx gelu 还是 exact gelu 吗？Keras=approx，Gemma_pytorch=exact，HF=exact。当比较 Keras 和 HF 时，torch.dist 得到...
- [OpenAI Five 击败 Dota 2 世界冠军](https://openai.com/research/openai-five-defeats-dota-2-world-champions)：OpenAI Five 是第一个在电竞游戏中击败世界冠军的 AI，在本周末的决赛中连续两场战胜了 Dota 2 世界冠军战队 OG。OpenAI Five 和 De...
- [pansophic/new_model_test · Hugging Face](https://huggingface.co/pansophic/new_model_test)：未找到描述
- [Philipp Schmid (@_philschmid) 的推文](https://fxtwitter.com/_philschmid/status/1763607891343225217?s=20)：Zephyr 7B Gemma 发布了！🔷🔶 我们很高兴地宣布 Zephyr Gemma，这是 @Google Gemma 7B 最好的微调版本。在 6 个基准测试中有 5 个超过了 Google Gemma Instruct，包括 MT Bench...
- [TechCrunch (@TechCrunch) 的推文](https://x.com/techcrunch/status/1762942326391906352?s=46)：Tim Cook 表示苹果今年将在 GenAI 领域“取得突破性进展” https://tcrn.ch/3Ig8TAX
- [GitHub - datadreamer-dev/DataDreamer: DataDreamer: Prompt. Generate Synthetic Data. Train & Align Models. 🤖💤](https://github.com/datadreamer-dev/DataDreamer)：DataDreamer：Prompt。生成合成数据。训练与对齐模型。 🤖💤 - datadreamer-dev/DataDreamer
- [Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus · Hugging Face 数据集](https://huggingface.co/datasets/Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus)：未找到描述
- [模型：由 ThiloteE 移除 Nous-Hermes-2-Mistral-7b-DPO 的 system prompt · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/fil)：描述你的更改。添加了“接受各种 system prompts”。移除 system prompt，修复空格。在请求审查前的检查清单。我已经对我的代码进行了自检。如果是...
- [模型：由 ThiloteE 移除 Nous-Hermes-2-Mistral-7b-DPO 的 system prompt · Pull Request #2054 · nomic-ai/gpt4all](https://github.com/nomic-ai/gpt4all/pull/2054/files)：描述你的更改。添加了“接受各种 system prompts”。移除 system prompt，修复空格。在请求审查前的检查清单。我已经对我的代码进行了自检。如果是...

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1212801067247603802) (41 messages🔥): 

- **搬家琐事**：`@slono` 在经历了充满搬家和生活挑战的两周缺席后回归，`@swyxio` 对此更新表示关注。
  
- **硬件赛道中的 AI**：`@guardiang` 分享了 Jonathan Ross 在卡塔尔 Web Summit 上的 [YouTube 视频](https://youtu.be/IixoaS5ckBA)，讨论了 Groq 在 TPU 和 LPU 技术方面的进展。
  
- **1-Bit LLM 突破**：`@nembal` 重点介绍了关于“三进制参数 (ternary parameters)”的新论文，该论文声称 1-bit 变体可以媲美全精度 LLM，这引发了 [Hacker News 讨论](https://news.ycombinator.com/item?id=39535800)，而 `@fanahova` 则对是否需要重新训练持怀疑态度。

- **Banana.dev 关停**：`@swyxio` 分享了一篇 [复盘博客](https://blog.erikdunteman.com/banana-pivot-unpeeled)，讨论了 Banana.dev 的 Serverless GPU 产品的兴衰，`@stealthgnome` 觉得故事中的某些部分格外令人感伤。

- **AI 产品管理资源**：`@swizec` 寻求有关 AI 项目产品管理的资源，`@420gunna` 推荐了杜克大学在 Coursera 上的 AI Product Management 专项课程，`@mrjose9` 则提议了 Fullstack Deep Learning 课程中 Josh Tobin 的讲座。

**提到的链接**：

- [AI Infrastructure Landscape](https://ai-infra.fun/)：未找到描述
- [no title found](https://news.ycombinator.com/item?id=39535800)：未找到描述
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)：最近的研究（如 BitNet）正在为 1-bit 大语言模型 (LLM) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...
- [Banana Pivot: Unpeeled](https://blog.erikdunteman.com/banana-pivot-unpeeled)：Erik Dunteman
- [Tweet from clem 🤗 (@ClementDelangue)](https://x.com/clementdelangue/status/1763328911365353933?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：按模型下载量、数据集下载量、Spaces 点赞数和个人资料粉丝数排名的 Hugging Face 顶级用户。恭喜榜单上的所有人，以及即将上榜的人！@mvaloatto 创建了该 Space：https://hu...
- [The Full Stack - Lecture 8: ML Teams and Project Management](https://fullstackdeeplearning.com/course/2022/lecture-8-teams-and-pm/)：构建由 ML 驱动的产品及其背后的团队
- [Jonathan Ross at Web Summit Qatar](https://youtu.be/IixoaS5ckBA?si=)：Groq CEO 兼创始人 Jonathan Ross 在 #WebSummitQatar2024 主舞台讨论如何让 AI 落地。X (原 Twitter): @WebSummitQatar Instagram: @WebSumm...
- [Jonathan Ross at Web Summit Qatar](https://youtu.be/IixoaS5ckBA?si=iTQFG-k_SQd6OP8H)：Groq CEO 兼创始人 Jonathan Ross 在 #WebSummitQatar2024 主舞台讨论如何让 AI 落地。X (原 Twitter): @WebSummitQatar Instagram: @WebSumm...
- [[AINews] Dia de las Secuelas (StarCoder, The Stack, Dune, SemiAnalysis)](https://buttondown.email/ainews/archive/ainews-dia-de-las-secuelas-starcoder-the-stack/)：2024年2月28日的 AI 新闻。我们为您检查了 356 个 Twitter 动态和 22 个 Discord（351 个频道，9043 条消息）。预计节省阅读时间（以 200wpm 计算）：860...
- [AI Product Management](https://www.coursera.org/specializations/ai-product-management-duke)：由杜克大学提供。管理 ML 产品的设计与开发。了解机器学习的工作原理，以及何时、如何使用它... 免费注册。

  

---


### Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1212700149315276810) (8 messages🔥): 

- **Representation Engineering 101 环节公告**：`@ivanleomk` 宣布 `@aimuggle` **很快**将在频道中演示 Representation Engineering 101。 
- **Swyxio 对录像表示关注**：`@swyxio` 遗憾错过了该环节，并对**录制版本**表现出兴趣。
- **Ivanleomk 建议进行第二轮**：`@ivanleomk` 提议让 `@aimuggle` 进行 Representation Engineering 101 环节的**第二轮**分享。
- **Aimuggle 考虑后续环节**：`@aimuggle` 俏皮地回应了这一建议，并提到**可能在几周内**进行第二次分享。
- **提高 RepEng 库的易用性**：`@aimuggle` 表示计划让 Representation Engineering 库能在免费层的 **Colab workbook** 中运行，以提高其易用性。
  

---

### Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1212699746758688802) (52 messages🔥): 

- **寻找最佳日程安排**：`@aimuggle` 和 `@youngphlo` 讨论了 **LLM Asia Paper Club** 的时间安排，考虑了成员所在的不同时区。提议的开始时间为 **新加坡时间晚上 8 点或 9 点**，或者可能在午餐时间，并打算在 6:05 pm 开始会议以照顾迟到者。
  
- **Representation Engineering 101**：`@ivanleomk` 介绍了 **Representation Engineering** 主题，强调了理解和操作神经网络中间表示（intermediate representations）在 steerability 和 alignment 应用中的重要性。该环节面向初学者，并鼓励公开参与。

- **追求清晰度**：用户就 **representation engineering** 概念进行了详细讨论，`@fx2y`、`@bryanblackbee` 和 `@danial.alh` 寻求关于 representation 与 embedding 之间的区别、向量差值的计算以及 control vectors 评估方法等话题的解答。

- **动态操控模型**：对话围绕 control vectors 的实际应用展开，`@fx2y` 和 `@jytan` 对多个 control vectors 的潜在堆叠以及在无需额外 fine-tuning 的情况下进行即时推理（on-the-fly inferences）感到好奇，这被确认为一种典型方法。

- **探索 Linear Representation Hypothesis**：`@healthymonkey` 询问了在讨论主题背景下 **linear representation** 的本质，引出了关于 representation 空间中的偏移如何以相反方向反映“好”与“不好”等概念含义的解释。

**相关链接**：

- [Nextra: the next docs builder](https://llm-paper-club-asia-notes.vercel.app/)：Nextra：下一代文档生成器
- [Representation Engineering Mistral-7B an Acid Trip](https://vgel.me/posts/representation-engineering/#How_do_we_make_one?_Is_it_hard?')：未找到描述
- [Representation Engineering 101](https://tana.pub/OG9hf2MA4tNS/representation-engineering-101)：未找到描述

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1212811979274846230) (6 messages): 

- **LLMs 微调 Hybrid Search**：LlamaIndex 介绍了一种使用 **LLMs** 优化 hybrid search 的方法，根据查询类别自动设置 alpha 参数。他们分享了一篇 [Twitter 帖子](https://twitter.com/llama_index/status/1763252392639042024) 详细介绍了该方法。
  
- **解决 RAG 中对结构化数据的需求**：LlamaIndex 团队推荐了 [@ClickHouseDB 的一篇博客文章](https://twitter.com/llama_index/status/1763282902358585445)，讨论了如何修改 **RAG 架构** 以在同一个向量数据库中同时处理非结构化和结构化数据。

- **本地 RAG 部署研讨会**：正如 CEO [@jerryjliu0](https://twitter.com/llama_index/status/1763304038442192978) 所宣布的，即将举行的由 @ollama 和 @tonicfakedata 主持的研讨会将展示如何构建 **本地 RAG 系统** 以增强隐私。它将演示如何使用 Ollama 部署 LlamaIndex + Tonic Validate 以维护数据隐私。

- **通过可观测性改进 LLM 应用**：@jerryjliu0 和来自 @traceloopdev 的 @nir_ga 将在 LlamaIndex 研讨会中演示如何使用 **OpenLLMetry** 为查询流水线添加可观测性（observability）。他们的 [推文](https://twitter.com/llama_index/status/1763364010676900080) 强调了对复杂查询进行 tracing 和 instrumenting 的重要性。

- **展望 Long-Context RAG**：LlamaIndex 在 [Twitter 帖子](https://twitter.com/llama_index/status/1763620476847632744) 中推测了在 **Gemini 1.5 Pro** 等 long-context LLMs 背景下 RAG 的未来，并讨论了检索技术可能发生的变化。

**相关链接**：

- [Preserve privacy using local RAG with Tonic.ai + LlamaIndex | Webinars | Tonic.ai](https://t.co/ke1XgF5Qb9)：了解如何开发本地检索增强生成（RAG）系统，并观看展示 LlamaIndex + Tonic Validate 如何使用 Ollama 进行本地部署的实操演示。
- [Embeddings & NLP](https://t.co/NFXtvm7K4z)：mixedbread.ai 提供简单的文本 embedding 生成服务，旨在增强 AI 项目的开发体验。

### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1212677950827270175) (84 messages🔥🔥): 

- **按页提取内容的挑战 (Page-Wise Content Extraction Challenge)**：`@ansumansatapathy` 询问如何使用 Llama Parse 提取按页划分的内容。`@whitefang_jr` 解释说 LlamaParse 目前不包含页码，并建议将 PDF 文件按页拆分作为变通方案。

- **LlamaParse 即将支持页码？**：`@cheesyfishes` 提到 LlamaParse 的 Markdown 输出可能会使用 `\n---\n` 作为分页符，并暗示即将推出高级输出格式。

- **澄清 Groq LLM 访问问题**：`@sridhar_10158` 在访问 `llama_index.llms` 中的 Groq 对象时遇到问题，重新安装包后解决。`@cheesyfishes` 回应了关于 Groq 可用性的查询，并建议使用最新版本的 LlamaIndex。

- **在 Query Engines 中组合 Postprocessors**：`@mysterious_avocado_98353` 得到 `@cheesyfishes` 的确认，多个 Node Postprocessor 模块（如 *MetadataReplacementPostProcessor* 和 *FixedRecencyPostprocessor*）可以在 Query Engine 中链式调用，并按列出的顺序应用。

- **澄清 Subquery QA 的实现**：`@andreipopg` 在实现 Subquery QA 时，寻求关于如何访问每个子问题的源节点以提取 QA 对的元数据和文本的帮助，但消息中似乎没有提供明确的解决方案。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1sQhOI7TN6CUfHp90Uvs8YmHmznpLp0qn?usp=sharing)：未找到描述
- [Ollama - Llama 2 7B - LlamaIndex 🦙 v0.10.14](https://docs.llamaindex.ai/en/stable/examples/llm/ollama.html)：未找到描述
- [无标题链接](https://www.secinsights.ai/)：未找到描述
- [llama_index/llama-index-packs/llama-index-packs-fuzzy-citation](https://github.com/run-llama/llama_index/tree/main/llama-index-packs/llama-index-packs-fuzzy-citation)：LlamaIndex 是一个用于 LLM 应用程序的数据框架 - run-llama/llama_index
- [GitHub - run-llama/sec-insights: 使用 LlamaIndex 的真实全栈应用程序](https://github.com/run-llama/sec-insights)：使用 LlamaIndex 的真实全栈应用程序 - run-llama/sec-insights

  

---



### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1212739586166628382) (20 messages🔥): 

- **征集 Axolotl 用户见解**：`@caseus_` 请求用户填写一份 [**调查问卷**](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform)，旨在了解终端用户如何与 Axolotl 交互。随后根据用户反馈更新了表单，减少了必填项。
- **术语调整建议**：`@nanobitz` 纠正了 `@caseus_` 问卷中使用的语言，建议使用“使用 (use)” Axolotl 而不是“购买 (buy)”。
- **TinyBox 为 AI 提供强劲动力**：`@dreamgen` 分享了关于 [**TinyBox**](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production) 的链接，这是一个使用六块 AMD Radeon RX 7900 XTX GPU 的高性能 AI 系统，旨在使 PetaFLOPS 级的 AI 性能普及化。
- **宣布下一场 Mistral AI Office Hour**：`@casper_ai` 发布了一个 Discord 邀请链接，指向即将举行的 **Mistral AI** Office Hour 信息，但未提供更多细节。
- **训练 Loss 异常问题**：`@nruaif` 报告了训练 Loss 和梯度范数，指出可能存在以 `nan` 表示的梯度值缺失问题。

**提到的链接**：

- [加入 Mistral AI Discord 服务器！](https://discord.gg/mistralai?event=1204405056825327677)：查看 Discord 上的 Mistral AI 社区 - 与其他 13789 名成员交流，享受免费的语音和文字聊天。
- [TinyBox 搭载六块 AMD 最快游戏 GPU 重新用于 AI —— 新机箱使用 Radeon 7900 XTX，零售价 1.5 万美元，现已投产](https://www.tomshardware.com/tech-industry/artificial-intelligence/tinybox-packs-a-punch-with-six-of-amds-fastest-gaming-gpus-repurposed-for-ai-george-hotzs-new-box-uses-radeon-7900-xtx-and-retails-for-dollar15k-now-in-production)：初创公司希望利用 Radeon RX 7900 XTX 提供高性能 AI 计算。
- [Axolotl 终端用户调查问卷](https://docs.google.com/forms/d/e/1FAIpQLSeyJkTk7sCYWpCNfKNNpnlMQlT9XU2nt_TJCzP4GSZBT0vrRA/viewform)：未找到描述

  

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1212789413080203304) (38 messages🔥): 

- **Sophia 优化器引起关注**：`@casper_ai` 分享了一篇关于 Sophia 的 [arXiv 论文](https://arxiv.org/abs/2305.14342) 链接。Sophia 是一种二阶优化器，据称速度是 Adam 算法的两倍，可以显著减少模型训练的时间和成本。他们还提供了一个 [Sophia 的 Jax 实现（非 Torch）](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py) 的链接。

- **丢弃反向传播，而非标准**：`@suikamelon` 介绍了 DropBP，这是一种在 [arXiv 论文](https://arxiv.org/abs/2402.17812) 中描述的新方法，它仅在反向传播（backward propagation）期间丢弃层，以保持前向传播（forward propagation）的准确性。该方法有代码支持，据报道实现了训练时间的缩减。

- **支持 StarCoder2**：`@faldore` 询问了关于 StarCoder2 的支持情况，随后分享了一个 [GitHub 仓库](https://github.com/bigcode-project/starcoder2) 并提到了一个将 StarCoder2 添加到项目中的相关 Pull Request。

- **在单 GPU 上进行 Unsloth 训练**：`@faldore` 表示有兴趣像 [Twitter 帖子](https://twitter.com/danielhanchen/status/1752707608488923614) 中提到的“在单块 H100 上进行 Unsloth 70b 训练”那样训练模型。`@caseus_` 回应称，Unsloth OSS 的限制是仅支持在单 GPU 上进行 LoRA，除非与 Axolotl 集成，而 `@giftedgummybee` 指出大多数 Axolotl 爱好者也面临同样的限制。

- **TRL 的 KTO 训练器存在问题**：`@giftedgummybee` 对 TRL 中的 KTO 训练器提出了担忧，警告其在 LoRA 配置下表现不佳，不支持 bnb 4 bit，且计算效率低下导致执行缓慢。这些观察结果得到了详细错误日志的支持，日志显示了段错误（segmentation faults）和其他兼容性警告。

**提到的链接**：

- [Sophia: A Scalable Stochastic Second-order Optimizer for Language Model Pre-training](https://arxiv.org/abs/2305.14342)：鉴于语言模型预训练的巨大成本，优化算法的非平凡改进将显著减少训练的时间和成本。Adam 及其变体...
- [DropBP: Accelerating Fine-Tuning of Large Language Models by Dropping Backward Propagation](https://arxiv.org/abs/2402.17812)：训练深度神经网络通常在正向和反向传播过程中涉及巨大的计算成本。传统的层丢弃技术在训练期间丢弃某些层...
- [levanter/src/levanter/optim/sophia.py at main · stanford-crfm/levanter](https://github.com/stanford-crfm/levanter/blob/main/src/levanter/optim/sophia.py)：使用 Named Tensors 和 Jax 构建易读、可扩展、可复现的基础模型 - stanford-crfm/levanter
- [GitHub - bigcode-project/starcoder2: Home of StarCoder2!](https://github.com/bigcode-project/starcoder2?tab=readme-ov-file#training)：StarCoder2 的主页！通过在 GitHub 上创建账号为 bigcode-project/starcoder2 的开发做出贡献。
- [GitHub - OpenLLMAI/OpenRLHF: An Easy-to-use, Scalable and High-performance RLHF Framework (Support 70B+ full tuning &amp; LoRA &amp; Mixtral &amp; KTO)](https://github.com/OpenLLMAI/OpenRLHF)：一个易于使用、可扩展且高性能的 RLHF 框架（支持 70B+ 全量微调 &amp; LoRA &amp; Mixtral &amp; KTO） - OpenLLMAI/OpenRLHF
- [Add Prodigy, SophiaG optimizers by Kimiko-AI · Pull Request #1350 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1350)：描述 https://arxiv.org/pdf/2306.06101.pdf https://arxiv.org/abs/2305.14342 动力与背景 如何测试？否 截图（如果适用）变更类型 Social Han...
- [NobodyExistsOnTheInternet/KTO-PRM-small · Datasets at Hugging Face](https://huggingface.co/datasets/NobodyExistsOnTheInternet/KTO-PRM-small)：未找到描述
- [add starcoder2 by ehartford · Pull Request #1349 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1349/files)：添加 starcoder2 描述 添加 starcoder2 动力与背景 添加 starcoder2 如何测试？我用它运行了一个 build 并且成功了 截图（如果适用）变更类型 添加 star...

---

### OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1212752354408140890) (4 messages): 

- **Community Cloud 质量差异**：`@dreamgen` 提到在 **Community Cloud** 上服务质量参差不齐，表明缺乏一致性。
- **RunPod Secure Cloud 使用查询**：`@dreamgen` 还询问是否有人正在利用 **RunPod's secure cloud**，暗示其可能不值得投入。
- **Starcoder2 兼容性检查**：`@faldore` 询问了某个实体或函数与 **starcoder2** 的兼容性，但未具体说明他们尝试操作的内容。
- **DPO 训练指南请求**：`@wizmak` 正在寻求关于如何在 **axolotl** 上使用 **DPO** 训练模型的示例或文章，表明需要教学资源。
  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1212754289269809224) (17 messages🔥): 

- **Mistral 在丹麦语任务中的出色表现**：`@le_mess` 分享了一个成功案例，指出他们的 **7B Mistral model** 通过使用**合成数据方法**，在丹麦语任务中达到了 **ChatGPT 3.5's performance**。
- **迭代模型训练取得成果**：通过训练超过 **30 iterative models**，`@le_mess` 实现了模型响应质量的持续提升，全程未使用 GPT-4，且仅用于**开源商业用途**。
- **自动化筛选中的人工参与**：最初，`@le_mess` 手动筛选了 1000 条响应，随后使用训练好的模型进行进一步的自动化筛选，以精炼输出并重新训练模型。
- **验证集的秘密揭晓！**：当 `@nanobitz` 询问评估数据集时，`@le_mess` 澄清他们实际上是指**验证数据集 (validation dataset)**，并提到他们使用了来自 [Scandeval.com](https://scandeval.com) 的基准测试。
- **基准测试基础**：`@le_mess` 确认没有创建自己的基准测试工具，而是引导用户使用他们利用的外部资源，暗示了制作自有评估数据集的复杂性。
  

---



### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1212693857360285706) (17 messages🔥): 

- **Tensor 性能调优成功**：`@iron_bound` 透露他们通过在 MLIR/LLVM 中启用 WMMA，解决了 **7900xtx** 上 Tensor 的性能问题，并分享了针对大矩阵尺寸下不同精度格式的详细性能指标。提供了一个解释更改的 commit 链接：[Tensor core fix on RDNA3](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b)。
  
- **Triton 调试器故障排除**：`@kierandidi` 在 **Triton 3.0.0 和 2.2.0** 中尝试使用 Triton 调试器时遇到了 `TypeError`，提示存在意外的关键字参数 'interpret'。`@andreaskoepf` 建议设置 `TRITON_INTERPRET` 环境变量，`@marksaroufim` 确认这是正确的方法，因为之前的方法已被弃用。

- **寻求段错误 (Segfault) 解决方案**：`@andreaskoepf` 分享了他们在 Triton 中遇到段错误的经历，通过重新排列代码行和修改 `num_warps` 的使用解决了问题，并附上了[问题代码](https://gist.github.com/andreaskoepf/833aac25c6e049e37ddadb5d0ad1ef48)和[修订版本](https://gist.github.com/andreaskoepf/4916d2a010f175b25aaa0655c8e5c9b4)的链接。

- **沉浸在 Triton 开发的一天**：`@andreaskoepf` 表达了对花一整天时间进行 Triton 开发的热情，而 `@drisspg` 询问开发是针对 Triton 代码本身还是在 Triton 环境内进行的，以了解开发工作的背景。

**提及的链接**：

- [[AMD][Navi31] Convert WMMA dot op to LLVM · joviliast/triton@f2f21c1](https://github.com/joviliast/triton/commit/f2f21c1a17437d3cef42dc31c08c82491ce4b08b#diff-c3f95d90ba556d38204257db3be8b6ae4f66f08d247ea8684ffec76432f6e05c)：添加了 dot 操作的 WMMA 转换逻辑。Signed-off-by: joviliast <iveselov.nn@gmail.com>
- [flash_attn_bias.py](https://gist.github.com/andreaskoepf/833aac25c6e049e37ddadb5d0ad1ef48)：GitHub Gist：即时分享代码、笔记和片段。
- [fash_attn_triton_working.py](https://gist.github.com/andreaskoepf/4916d2a010f175b25aaa0655c8e5c9b4)：GitHub Gist：即时分享代码、笔记和片段。

  

---

### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1212784318867312650) (12 messages🔥): 

- **FP8 Intrinsics 可用性确认**：`@zippika` 指出 **FP8 (8-bit floating-point)** Intrinsics 在 [CUDA 文档](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC)中仍然可用，需要在程序中包含头文件 `cuda_fp8.h`。

- **FP8 主要用于数据存储**：`@zippika` 强调 FP8 主要是一种“数据”格式，通常不用于实际计算。

- **cudaMallocManaged 与 malloc 的讨论**：`@vim410` 讨论了 `malloc` 和 `cudaMallocManaged` 之间的区别，并引用了一篇关于异构内存管理 (HMM) 的博客文章，指出后者优于 `malloc`，但速度不如 `cudaMalloc`。

- **Ada Lovelace GPU 上受限的 FP8 计算操作**：`@drisspg` 分享了对 FP8 计算的见解，引用了 [PyTorch 讨论](https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815)，其中提到 Ada Lovelace GPU 对 FP8 计算操作的支持有限，特别是缺乏 `wgmma.mma_async` 指令。

- **CUDA 和 PyTorch 中的统一内存 (Unified Memory)**：`@marksaroufim` 在讨论统一内存和 `cudaMallocManaged` 时分享了一个 [GitHub 链接](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L181)，并指出如果统一内存能够比 CPU Offloading 编写出更快的代码，那么对于资源受限的 GPU 配置来说，它可能被视为更好的默认选项。

**提到的链接**：

- [CUDA Math API :: CUDA Toolkit Documentation](https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__FP8__MISC.html#group__CUDA__MATH__FP8__MISC)：未找到描述
- [bitsandbytes/bitsandbytes/functional.py at main · TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/functional.py#L181)：通过 PyTorch 的 k-bit 量化实现可访问的大语言模型。- TimDettmers/bitsandbytes

  

---


### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1212979036054229032) (6 messages): 

- **BASED Attention 论文分享**：`@marksaroufim` 分享了一篇探索基于 Attention 的语言模型效率的研究论文，并介绍了 BASED 这一新架构。该论文可在 [arXiv](https://arxiv.org/pdf/2402.18668.pdf) 获取。
- **尝试滑动窗口注意力 (Sliding Window Attention)**：`@drisspg` 正在致力于为 [PyTorch 添加滑动窗口注意力偏置](https://github.com/pytorch/pytorch/pull/120143)，这可能会改善推理过程中的内存消耗问题。
- **关于摘要和 Mask 实现的讨论**：`@andreaskoepf` 提供了同一篇[论文](https://arxiv.org/abs/2402.18668)的摘要链接，`@marksaroufim` 询问是否可以通过更改缩放点积注意力 (SDPA) 中的 Mask 来实现不同的方案。
- **对 BASED Attention 的调侃**：`@marksaroufim` 以戏谑的口吻对讨论论文中 BASED Attention 的命名回应道：“based attention lmao”。
- **对 HF Transformers 中默认 Attention 的担忧**：`@marksaroufim` 惊讶地发现 Hugging Face 的 Mistral 实现默认使用 SDPA 而不使用 Sliding Window Attention，这导致在 4k 上下文以上可能会出现问题。该消息引用了一条表达担忧的推文，并链接到了[相关代码](https://github.com/huggingface/transformers/blob/main/src/transformers/models/mistral/modeling_mistral.py#L1004)和提议的 [修复 PR](https://github.com/huggingface/transformers/pull/29220)。

**提到的链接**：

- [Philipp Singer (@ph_singer) 的推文](https://x.com/ph_singer/status/1763538607191527540?s=20)：显然目前 HF Transformers 的 Mistral 实现将默认 Attention 设置为 SDPA，但没有使用 Sliding Window。我在 4k 以上的上下文观察到了奇怪的行为。所以如果我没有...
- [Simple linear attention language models balance the recall-throughput tradeoff](https://arxiv.org/abs/2402.18668)：最近的研究表明，基于 Attention 的语言模型在召回率（即在上下文中定位先前看到的 Token 的能力）方面表现出色。然而，基于 Attention 的模型的效率...
- [Add sliding window attention bias by drisspg · Pull Request #120143 · pytorch/pytorch](https://github.com/pytorch/pytorch/pull/120143)：摘要 WIP，详情请参阅 #119653

  

---

### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1212776944077377556) (34 条消息🔥): 

- **梯度混淆 (Gradient Confusion)**：`@ericauld` 询问了一个与 backward pass 相关的问题，随后 `@andreaskoepf` 澄清这似乎与错误的梯度有关。
- **执行过程错误频发**：`@ericauld` 在尝试运行测试脚本时遇到了多个问题，包括拼写错误和缺少导入，最终导致放弃了该尝试。
- **令人困扰的 Triton 消息**：`@jamesmel` 指出设置 `cuda = True` 会导致问题，并强调了一个关于 Triton 和指针参数（pointer arguments）的错误。
- **损坏代码中的提交历史线索**：`@iron_bound` 建议 lucidrains 仓库的提交历史可能暗示了其中尝试的自定义 Kernel 存在问题，链接见[此处](https://github.com/lucidrains/ring-attention-pytorch/commits/main/)。
- **GPU 错误频现**：`@andreaskoepf` 注意到 GPU 资源分配和模块缺失的异常行为，最终通过重启系统解决了问题。

**提到的链接**：

- [ring-attention-pytorch/ring_attention_pytorch/ring_flash_attention_cuda.py at df48d4d338f5b970086aec2df75e4be34080de1b · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/blob/df48d4d338f5b970086aec2df75e4be34080de1b/ring_attention_pytorch/ring_flash_attention_cuda.py#L61)：来自 Berkeley AI 的 Liu 等人对 Ring Attention 的探索 - lucidrains/ring-attention-pytorch
- [Commits · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/commits/main/)：来自 Berkeley AI 的 Liu 等人对 Ring Attention 的探索 - Commits · lucidrains/ring-attention-pytorch
- [GitHub - zhuzilin/ring-flash-attention: Ring attention implementation with flash attention](https://github.com/zhuzilin/ring-flash-attention/)：结合 Flash Attention 的 Ring Attention 实现 - zhuzilin/ring-flash-attention
- [A ring attention with flash attention kernel implementation · Issue #4 · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/issues/4#issuecomment-1969440025)：你好！感谢你在 PyTorch 中实现 Ring Attention 的工作！我刚刚尝试实现了一个 ring_flash_attn_qkvpacked_func（对应 Flash Attention 中的 flash_attn_qkvpacked_func...）
- [ring-attention-pytorch/ring_attention_pytorch/ring_flash_attention_cuda.py at df48d4d338f5b970086aec2df75e4be34080de1b · lucidrains/ring-attention-pytorch](https://github.com/lucidrains/ring-attention-pytorch/blob/df48d4d338f5b970086aec2df75e4be34080de1b/ring_attention_pytorch/ring_flash_attention_cuda.py#L349)：来自 Berkeley AI 的 Liu 等人对 Ring Attention 的探索 - lucidrains/ring-attention-pytorch

  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1212679781355888700) (30 条消息🔥): 

- **寻求 ChatVertex AI 的环境建议**：用户 `@irfansyah5572` 询问了在使用 LangChain 操作 ChatVertex AI 时的最佳环境，但未提供更多细节，也未收到回复。
- **JSON Schemas 与 LLMs 集成**：`@kamakshi08` 分享了一个[链接](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json)，解释了如何将 JSON schemas 与大语言模型（LLMs）结合使用以生成格式良好的 JSON 输出。他们随后提出了一个关于如何将此解析器与通过 `ollama` 下载的 `llava` 结合使用的问题，主要涉及多模态模型。
- **DataBricks 工作流作业中的故障排除**：用户 `@hanumantgarad_25732` 讨论了一个问题：`SQLDatabase.from_databricks` 在 Databricks notebooks 中运行正常，但在 Databricks 工作流作业中失败并抛出 `AttributeError`。用户假设该错误是由于在 notebook 环境之外缺少 `DatabricksReplContext` 对象导致的。
- **探索自定义 LangChain 工具的重试机制**：用户 `@abinandan` 寻求在发生 `ToolException` 时重试自定义 LangChain 工具的方法，并得到了 `kapa.ai` 的支持，建议采用用户分享的变通方法，包括输出已知值或针对可识别的文本抛出异常以触发重试条件。
- **关于 LangChain 安全性与能力的疑问**：`@.suzerain` 询问 LangChain 是否在其 LCEL 中采用了额外的安全保护措施，但 `kapa.ai` 未给出直接回答。用户 `@akis_21513` 链接了一个现有的 LangChain [GitHub issue](https://github.com/langchain-ai/langchain/issues/18292)，反映了他们遇到的类似问题，但聊天中未提出解决方案。
- **使用 AI 自动化 Shopify 客户支持**：`@erikk4` 描述了一个使用 AI 工具自动化处理 Shopify 相关查询的客户支持流程，并征求除 LangChain 之外的工具建议。目前没有后续讨论或进一步指导。
- **Weaviate 与 LangChain 的问题**：用户 `@dazzling_puppy_08816` 和 `@chayan_systango` 分别表达了在使用 LangChain 时遇到的问题，具体包括尝试在 VSCode 中运行以及为现有索引初始化 Weaviate，但消息中未给出解决方案。
- **在 LangChain 中实现 GPTCache 并处理批处理**：`@tawsif2781` 讨论了在调用链时结合使用 batch 的复杂性，并寻找使用 LCEL 混合这两种方法的方式；同时 `@david_zoe` 寻求实现 GPTCache 的帮助并遇到了 Onnx runtime 错误，但未获得指导。

**提到的链接**：

- [未找到标题](https://js.langchain.com>)): 未找到描述
- [JSON 解析器 | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/model_io/output_parsers/types/json): 该输出解析器允许用户指定任意 JSON schema。
- [如何在已创建的 weaviate 持久化数据上进行 RAG · langchain-ai/langchain · Discussion #15332](https://github.com/langchain-ai/langchain/discussions/15332): 大家好，我正在探索将 weaviate 与 langchain 结合使用。我已经加载了一堆 PDF 并进行了标准分块，并创建了一个 weaviate 类，如 class_obj = { "class": "WMOInfo", "...
- [自定义 Agent 类失败，提示对象没有 'is_single_input' 属性 · Issue #18292 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/18292): 检查了其他资源。我为此 issue 添加了一个非常详细的标题。我使用集成搜索查询了 LangChain 文档。我使用 GitHub 搜索查找了类似问题并...
- [Issues · langchain-ai/langchain](https://github.com/langchain-ai/langchain/issues/10714>).): 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。

  

---

### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1212755246955237459) (12 messages🔥): 

- **thatdc 提交的 Langserve 故障排除**：`@thatdc` 遇到了 **langserve** 不返回 Agent 中间步骤的问题，只返回了最终输出。他们认为问题出在 RemoteRunnable 对象的 `_invoke` 方法和 `_decode_response` 方法中，具体是在 `output = serializer.loadd(obj["output"])` 这一行。

- **veryboldbagel 建议的临时解决方案**：`@veryboldbagel` 建议在 `output_type` 中使用 `Any` 来尝试解决此问题。他们还指向了一个未解决的 [GitHub issue #381](https://github.com/langchain-ai/langserve/issues/381)，该问题与中间步骤的序列化有关，并进一步建议在 Chain 中添加额外部分来处理序列化作为变通方法。

- **API 请求调查**：`@thatdc` 分享了一个用于测试 API 的 **curl** 命令，展示了他们对 Agent 的调用，随后发布了收到的 JSON 响应，显示仅包含最终输出。

- **Agent Executor 配置详情**：`@thatdc` 发布了其 **AgentExecutor** 的配置，强调了 `return_intermediate_steps=True` 和 `streaming=True`，希望能从输出中获取中间步骤。

- **垃圾礼品链接**：`@skywalker09_` 发布了一个未经请求的所谓 50 美元 Steam 礼品链接，似乎与讨论无关，可能被视为垃圾信息。

**提及的链接**：

[Serialization issues with intermediate_steps for AgentExecutor · Issue #381 · langchain-ai/langserve](https://github.com/langchain-ai/langserve/issues/381)：我尝试了一个使用案例，其中我使用一个作为 RemoteRunnable 的 Agent Chain 来初始化 AgentExecutor。即客户端代码如下：from langchain.agents import AgentExecutor...

  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1212720596207673394) (2 messages): 

- **关于生成模板的咨询**：`@tigermusk` 询问如何创建一个类似于 [Smith Langchain 的 React Chat JSON 模板](https://smith.langchain.com/hub/hwchase17/react-chat-json) 的模板。目前还没有关于如何在 Python 代码中实现这一点的后续信息。
- **垃圾信息警报**：`@skywalker09_` 发布了一条疑似垃圾信息的消息，提供“50 美元礼品”并附带链接 [steamcommunity.com/gift/50](https://u.to/eA9sIA)。

**提及的链接**：

[LangSmith](https://smith.langchain.com/hub/hwchase17/react-chat-json)：未找到描述

  

---


### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1212986681922289674) (5 messages): 

- **使用 Endoftext 轻松编辑 Prompt**：`@cabreraalex` 介绍了 **Endoftext**，这是一个 AI 驱动的 Prompt 编辑器，可以生成建议和测试用例以优化 AI Prompt。观看他们在 [YouTube](https://youtu.be/PGv5ymOhaNA) 上的 60 秒演示，并在 [Endoftext](https://app.endoftext.app/) 尝试测试版。

- **Airbyte 与 Langchain 结合**：`@andysingal` 分享了一篇文章，介绍 Airbyte 与 Langchain 的集成如何简化数据集成和文档处理。在 [Medium 文章](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad) 中了解更多关于它们协同使用的信息。

- **SimplyAnalyze AI 发布对话分析开发者预览版**：`@petervandijck_68934` 宣布推出 **SimplyAnalyze AI**，这是一个类似于 Google Analytics 但专门用于分析对话的平台。早期采用者可以在 [SimplyAnalyze.AI](https://simplyanalyze.ai) 注册免费的一年期账户。

**提及的链接**：

- [Airbyte with Langchain: Streamlining Data Integration and Document Processing](https://medium.com/ai-advances/airbyte-with-langchain-streamlining-data-integration-and-document-processing-8593db1fc3ad)：Ankush k Singal
- [endoftext Demo - An AI-powered Prompt Editor](https://youtu.be/PGv5ymOhaNA)：endoftext 通过建议编辑、Prompt 重写和测试用例生成来帮助你编写更好的 Prompt。请访问 https://endoftext.app 查看。
- [endoftext | AI-powered prompt editor](https://app.endoftext.app/)：通过 Prompt 建议、智能重写和合成数据生成，消除 Prompt 工程中的猜测。endoftext 是一个 AI 驱动的 Prompt 编写助手，可帮助你快速改进...

  

---

### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1212826031908589578) (3 条消息): 

- **LangGraph 结合 YahooFinance**：用户 `@tarikkaoutar` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s)，演示了如何通过集成 Function Call 和 YahooFinance，利用 LangGraph 创建一个多 Agent 股票分析聊天机器人。该视频为那些希望了解 LangGraph 在不同场景下应用的人提供了重点参考。
- **可疑 Steam 链接警报**：用户 `@skywalker09_` 发布了一个声称是通过 Steam 赠送 50 美元礼品的链接 ([steamcommunity.com/gift/50](https://u.to/eA9sIA))。然而，应谨慎对待，因为这可能是一个钓鱼链接或诈骗。
- **GPTCache 实现咨询**：用户 `@david_zoe` 就 LangChain 中 GPTCache 的实现向社区寻求帮助，他在使用时遇到了 "Onnx runtime error"。他们表示有兴趣探索来自 OpenAI 或 HuggingFace 的 SafeTransformers 的 Embedding 选项，并正在寻求解决缓存问题的指导。

**提到的链接**：

[LangGraph + Function Call+ YahooFinance = Multi-Agent Application](https://www.youtube.com/watch?v=r2PvHdkaXWc&t=129s)：#chatbot #animation #trading #ai #machinelearning #datascience 在这个视频中，你将使用 LangGraph、Function call 和 C... 制作一个 AI 股票分析聊天机器人。

---

### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1212735927546421288) (45 条消息🔥): 

- **预付卡困境**：用户 `@fakeleiikun` 询问了 OpenRouter 对预付卡的支持情况，并提到在使用 Google Pay 时出现了 *error 402 或 error 502* 等问题，尽管该卡在其他网站上可以正常使用。`@louisgv` 建议，像 Discovery 这样的预付卡可能会被 Stripe Radar 标记，但来自受支持银行的虚拟卡通常是被接受的。
  
- **寻求 Helicone 集成帮助**：用户 `@wise_monkey_42910` 寻求在使用 LangChain ChatOpenAI 时将 Helicone 与 OpenRouter 集成的帮助。`@louisgv` 提供了一个指向 [GitHub 示例](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts) 的有用链接以及 [Helicone 文档](https://docs.helicone.ai/getting-started/integration-method/openrouter) 以进行正确的集成。

- **Token 问题澄清**：`@cupidbot.ai` 询问了关于带有 Function Calling 的流式传输，以及 `native_tokens_prompt` 与 `tokens_prompt` 之间的区别。`@alexatallah` 澄清说，`native_tokens` 指的是模型自身 Tokenizer 中的 Token，现有的使用指标确实是原生的，并计划相应地更新文档。

- **Elon Musk 与 OpenRouter**：当 `@telepathyx` 暗示 Elon Musk 可能会进入一个与 OpenRouter 竞争的领域时，对话发生了转折。虽然 `@louisgv` 起初感到惊讶，但 `@alexatallah` 纠正说，Groq（而非 Grok）在解决速率限制后，可能是 OpenRouter 未来潜在的补充，从而打破了 Musk 直接竞争的想法。

**提到的链接**：

- [OpenRouter - Helicone](https://docs.helicone.ai/getting-started/integration-method/openrouter)：未找到描述
- [openrouter-examples/examples/langchain/index.ts at main · OpenRouterTeam/openrouter-examples](https://github.com/OpenRouterTeam/openrouter-examples/blob/main/examples/langchain/index.ts)：集成 OpenRouter API 的示例。通过在 GitHub 上创建账户为 OpenRouterTeam/openrouter-examples 的开发做出贡献。

---

### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1212891180728188949) (16 条消息🔥): 

- **Nate 与 Yann 的偶遇**：`@natolambert` 分享了一个令人兴奋的个人动态：他见到了 Yann LeCun，尽管他没有鼓起勇气邀请他参加播客。
- **需要提升魅力值**：针对 `@natolambert` 的犹豫，`@philpax` 开玩笑地建议他下次会在“魅力检查”中成功。
- **与 Yann 讨论清洁能源话题**：与 Yann LeCun 的交谈非常吸引人，`@natolambert` 和 Yann 广泛讨论了**清洁能源**。
- **Lambert 家族的联系**：关于 Lambert 家族联系的聊天幽默地展开了，`@victory` 暗示了一个自定义服务器表情符号，而 `@mike.lambert` 试图确定 `@victory` 与哪个 Lambert 有亲戚关系，推测可能是 Nate。
- **Yann 的内部观点**：`@natolambert` 强调 Yann LeCun 看起来非常随和且开放，但在争取 AI 开放性的斗争中表达了孤独感，同时也对强化学习 (RL) 表示怀疑，他将其总结为“典型的 Yann 风格”。

---

### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1212763024054816789) (4 条消息): 

- **为 DPO 数据集生成错误答案**：`@philipmay` 建议了一种在 Dense Passage Retrieval (DPR) 数据集中创建负样本的方法，即**要求 LLM 根据给定的上下文和问题生成刻意的错误答案**。
- **DiscoLM_German_7b 设置搜索**：`@mab3049` 正在寻求关于 **DiscoLM_German_7b 演示模型**所使用设置的见解，因为他们自己的尝试未能匹配演示模型的结果。
- **Fine Tuning 中的 Padding Token 困惑**：`@silicagel_64242` 询问在 Fine Tuning 期间应使用哪个 token 作为 `pad_token`。他们遇到了相互矛盾的建议，涉及 `eos_token`、`unk_token` 以及显式的 `"[PAD]"` token。
- **为 RAG 寻找最佳德语 Embedding 模型**：`@bjoernwerner` 征求关于在特定领域的 Retriever-Aggregator-Generator (RAG) 应用中最有效的德语文本 Embedding 模型的意见，并列出了几个潜在的**单向量和多向量 Embedding** 供考虑。
  

---


### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1212877396936753214) (5 条消息): 

- **寻找难以捉摸的 MT-Bench-X**：`@crispstrobe` 正在寻找 **MT-Bench-X** 数据集，并提到根据 [arxiv.org 上的论文](https://arxiv.org/pdf/2402.13703v1.pdf) 该数据集采用 Apache 2.0 许可证。其具体兴趣是寻找在德语任务中表现良好的模型。
- **发现替代德语数据集**：`@bjoernp` 虽然没见过 MT-Bench-X，但推荐了 Hugging Face 上的 **MT-Bench-DE**，这对于寻求德语基准测试的人可能有所帮助。
- **倡导真正的德语基准测试**：`@crispstrobe` 推荐了经过人工改进的 [MT-Bench-TrueGerman](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman) 数据集，强调了真实德语基准测试的稀缺性，以及为此目的使用 GPT-4 翻译的弊端。


**提到的链接**：

- [VAGOsolutions/MT-Bench-TrueGerman · Hugging Face 数据集](https://huggingface.co/datasets/VAGOsolutions/MT-Bench-TrueGerman)：未找到描述
- [LeoLM/MT-Bench-DE · Hugging Face 数据集](https://huggingface.co/datasets/LeoLM/MT-Bench-DE)：未找到描述

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/1212858233925013587) (3 条消息): 

- **EQ-Bench 添加德语 Prompt**：用户 `@crispstrobe` 分享了一个 [GitHub Pull Request](https://github.com/EQ-bench/EQ-Bench/pull/12)，宣布 EQ-Bench 现在支持一套用于快速评估的德语 Prompt。结果显示 `gpt-4-1106-preview` 和 `gpt-3.5-turbo-0125` 等模型获得了高分，德语翻译是使用 ChatGPT-4-turbo 完成的。

- **Ollama 模型模板中可能存在的错误**：此外，`@crispstrobe` 引用了一个 [GitHub Issue](https://github.com/ollama/ollama/issues/1977)，讨论了从 ollama.ai 下载的模型模板定义中可能存在的错误，这些错误可能会影响模型性能。

- **Discord 上的持续讨论**：`@_jp1_` 指向了一个关于该话题的[广泛讨论](https://discord.com/channels/1178995845727785010/1183158791605330051/1211590899902058557)，尽管未提供讨论的具体细节。

**提到的链接**：

- [GitHub: Let’s build from here](https://github.com/)：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做贡献、管理 Git 仓库、像专业人士一样审查代码、跟踪错误和功能...
- [Build software better, together](https://github.com/EQ-bench/EQ-Bench/pull/12)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、Fork 并为超过 4.2 亿个项目做贡献。
- [ollama.ai 可下载模型模板定义中的错误 · Issue #1977 · ollama/ollama](https://github.com/ollama/ollama/issues/1977)：你好，从 https://ollama.ai 下载的模型在 TEMPLATE 定义中的一些错误正在不同程度地损害模型表现。我只是在实验时偶然发现了这一点...

### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1212690854653861938) (3 messages): 

- **Claude 的开场白怪癖**：`@justinpinkney` 分享了一种技巧，通过设置模型返回的初始字符来避免 Claude 倾向于以“Sure here's a...”之类的短语开始回答，详见 [Anthropic 的重写指南](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites)。这可以强制 Claude 从特定字符（如 `<rewrite>`）开始，从而绕过无用的介绍。
- **引导 Claude 朝正确方向发展**：`@derekpwillis` 同意绕过 Claude 开场评论的难度，并尝试强制它以 `{` 开头，尽管 Claude 经常坚持要解释其行为。

**提到的链接**：

[Ask Claude for rewrites](https://docs.anthropic.com/claude/docs/ask-claude-for-rewrites)：如果 Claude 给出的回答接近但并不完全是你想要的，你可以要求 Claude 重写。在 Slack 中，这可以简单到在 Claude 回答后告诉它“再试一次”...

  

---


### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1213152538988253265) (8 messages🔥): 

- **寻找最佳开源 LLM**：`@gwthompson` 询问了关于可以在本地通过 LLM 运行并用于 Datasette enrichment 的最佳开源模型的建议，但消息中未提供具体建议。
- **寻求简洁的 LLM C API**：`@florents_` 询问了具有简洁 C API 用于文本嵌入（text embedding）的 LLM，但未收到针对其查询的直接建议。
- **介绍带有 C API 的 Llama.cpp**：`@agarcia_me` 提到 Llama.cpp 中提供了嵌入支持，虽然需要 C++ 编译器，但提供了 C API。他们还提到打算很快分享一个用于嵌入的 sqlite 扩展代码。
- **关于 C API 使用的澄清**：在回复 `@florents_` 时，`@agarcia_me` 澄清说 `embedding.cpp` 仅使用了 `common.h` 中的几个函数，建议提取必要的函数并直接依赖 C API。
- **分享 LLM 嵌入的 C 代码片段**：`@agarcia_me` 分享了一个详细的 C 代码片段，演示如何实现 LLM 嵌入，并提到它适用于批大小（batch size）为 1 的情况，且是纯 C 语言编写，随后澄清 `llama_batch` 是该过程中最复杂的部分。

**提到的链接**：

[llama.cpp/examples/embedding/embedding.cpp at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/embedding/embedding.cpp)：C/C++ 中的 LLM 推理。通过在 GitHub 上创建账户来为 ggerganov/llama.cpp 的开发做出贡献。

  

---



### LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1213215210278752348) (3 messages): 

- **Anthropic 领先于 Gemini 1.5**：用户 `@res6969` 提到有传言称，在闭门测试中，**Anthropic** 在上下文长度能力方面优于 **Gemini 1.5**。
- **Anthropic 在准确率方面也处于领先地位**：除了上下文长度外，`@res6969` 还听说 Anthropic 与 Gemini 1.5 相比显示出**显著更好的准确率**。
- **缺乏个人测试**：尽管传闻甚广，`@res6969` 指出他们**尚未能亲自测试**这些能力。
  

---


### LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/1212764382644146186) (2 messages): 

- **寻找 OpenAI 资源**：`@res6969` 表示需要寻找更好的 OpenAI **资源**和**信息源**。
- **寻求生产级技巧**：`@res6969` 正在寻找在生产环境中实现 OpenAI **codeinterpreter** 的**资源**或指导。
  

---


### LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/1212999028678074388) (2 messages): 

- **System Prompts 中的秘密配方**：`@robertchung` 提出了 **system prompts** 及其对模型输出影响的话题，指出它们扮演着重要但有些神秘的角色，但也提到该主题缺乏可用资源。
- **模型行为受变动和更新影响**：`@jeffreyw128` 建议 system prompts 的有效性可能取决于**特定模型**和实验室进行的**持续更新**，这表明其性能存在不可预测的因素。
  

---

### AI Engineer Foundation ▷ #[general](https://discord.com/channels/1144960932196401252/1144960932657758210/1212713293693718598) (4 messages): 

- **寻找 AI Engineer 职位描述示例**：用户 `@peterg0093` 正在寻找英国 AI Engineer 的优秀职位描述示例，并热衷于在招聘过程中采用新兴的标准语言。
- **分享描述性职位示例**：`@swyxio` 提供了 Hex 的 AI Engineer 职业页面的链接，该页面提供了关于公司文化、使命和角色期望的见解，可作为职位描述的参考模板：[Hex Careers](https://hex.tech/careers/ai-engineer/)。
- **关于 AI Engineer Foundation 模型的建议**：`@swyxio` 建议 "AI Engineer Foundation" (AIEF) 可以采用类似于 [AI-Infra](https://ai-infra.fun/) 的结构化设置来组织资源。

**提及的链接**：

- [AI Infrastructure Landscape](https://ai-infra.fun/)：未找到描述
- [AI Engineer - Careers | Hex ](https://hex.tech/careers/ai-engineer/)：在生产级 AI 应用的最前沿工作。

---

### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1212946349813661769) (1 messages): 

- **对活动组织的认可**：用户 `@huikang` 对在 [LinkedIn](https://www.linkedin.com/posts/ai-eng-foundation_last-saturday-on-022424-sasha-organized-activity-7169152145336782850-_TsG) 上被提及参与上周六（2024年2月24日）组织的活动表示感谢，并强调了其在近期活动中的参与。

---

### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/1212804141626294282) (2 messages): 

- **探索 Gemma 7B 的 Finetuning**：`@pradeep1148` 分享了一个名为 "Finetune Gemma 7B with Unsloth" 的 [YouTube 视频](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)，提供了 Finetuning Gemma 模型的演练，并附带了一个 [Colab notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollTo...)。
- **OpenCodeInterpreter 介绍**：`@pradeep1148` 还发布了一个关于 "OpenCodeInterpreter" 的 [YouTube 视频](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be)，介绍了一个旨在弥合 LLM 与复杂专有系统之间差距的开源代码生成系统套件。

**提及的链接**：

- [OpenCodeInterpreter](https://www.youtube.com/watch?v=cwJKopBBnWo&feature=youtu.be)：OpenCodeInterpreter 是一套开源代码生成系统，旨在弥合 LLM 与复杂专有系统之间的差距...
- [Finetune Gemma 7B with Unsloth](https://www.youtube.com/watch?v=ikIgy0qlif8&feature=youtu.be)：我们将了解如何使用 Unsloth 对 Gemma 模型进行 Finetuning。https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing#scrollT...

---

### Skunkworks AI ▷ #[general](https://discord.com/channels/1131084849432768614/1192042724480794685/1212786676825329695) (1 messages): 

- **聊天中的招聘邀约**：用户 `.papahh` 直接联系了 `@1117586410774470818` 并提供了一份工作机会，请其查看私信了解详情。消息中未提供更多背景或信息。

---

### Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/1212862697079574591) (1 messages): 

- **神经网络生态位映射 (Niche Mapping)**：`@camelfacts` 介绍了一篇论文，该论文提出了一种通过创建表示生态位图来解释 **神经网络表示 (neural network representations)** 的新方法。该论文结合了经济学和信息理论，并已在 LessWrong 上分享以获取反馈：[What’s in the box? Towards interpretability by distinguishing...](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1)。

**提及的链接**：

[What’s in the box?! – Towards interpretability by distinguishing niches of value within neural networks. — LessWrong](https://www.lesswrong.com/posts/7tSthxSgnNxbt4Hk6/what-s-in-the-box-towards-interpretability-by-distinguishing-1)：摘要：数学模型可以描述神经网络架构和训练环境，然而出现的学习表示却具有……