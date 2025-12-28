---
companies:
- mistral-ai
- openai
- perplexity-ai
- llamaindex
- qwen
- langchain
date: '2024-02-27T20:03:47.279106Z'
description: '**Discord 社区**分析了 **22 个服务器**、**349 个频道**和 **12885 条消息**，揭示了关于 **Mistral
  AI**、**Miqu** 和 **GGUF 量化模型**的模型比较与优化的活跃讨论。


  核心亮点包括：

  *   **模型对比：** 将 **Mistral Large** 与 **GPT-4** 进行对比，重点关注性价比和性能。

  *   **技术优化：** 探索 **GPTQ** 和 **QLORA** 等量化技术以减少显存（VRAM）占用。

  *   **高级应用：** 强调了**角色扮演**、**故事创作**、**代码清晰度**和 **AI 辅助反编译**等应用，并开发了针对 **Mistral 7b**
  的**异步总结脚本**等工具。

  *   **前沿交叉：** 讨论了**量子计算**与 AI 的交叉领域，包括 DARPA 资助的项目以及用于图像处理的**基于编码器的扩散技术**。

  *   **社区生态：** 社区努力涵盖了新西班牙语大语言模型（LLM）的发布、硬件实验和开源倡议；**Perplexity AI** 和 **LlamaIndex**
  等平台因创新和集成能力受到关注。

  *   **协作精神：** 关于 **Mistral AI** 开源承诺的推测，以及用于快速 RAG 部署的 **R2R** 等工具，彰显了社区的协作精神。'
id: d04af6d2-a91e-46e4-825a-e61462b91dad
models:
- mistral-large
- miqu
- mixtral
- gpt-4
- mistral-7b
original_slug: ainews-welcome-interconnects-and-openrouter
people:
- nathan-lambert
- alex-atallah
title: 欢迎 Interconnects 和 OpenRouter。
topics:
- model-comparison
- model-optimization
- quantization
- role-playing
- story-writing
- code-clarity
- ai-assisted-decompilation
- asynchronous-processing
- quantum-computing
- encoder-based-diffusion
- open-source
- hardware-experimentation
- rag-systems
---

 



---

**目录**

[TOC] 


# PART 0: 摘要的摘要的摘要

<div><p><strong>模型比较与优化</strong>：Discord 用户积极参与了关于各种 AI 模型性能和优化的讨论，包括 <strong>Mistral AI</strong>、<strong>Miqu</strong> 和 <strong>GGUF 量化模型</strong>。关键话题包括 Mistral AI 新发布的 <strong>Mistral Large</strong> 模型与 OpenAI 的 <strong>GPT-4</strong> 的对比，强调了其成本效益和性能。用户还探索了高效的加载实践和模型量化方法，如 <strong>GPTQ</strong> 和 <strong>QLORA</strong>，以增强模型性能并减少 VRAM 占用。</p><p><strong>高级功能与应用</strong>：用户对利用 AI 进行 <strong>role-playing</strong>（角色扮演）和 <strong>story-writing</strong>（故事创作）等特定应用表现出浓厚兴趣，强调模型需要管理一致的时间线和角色情感。此外，还讨论了 AI 在 <strong>code clarity</strong>（代码清晰度）和 <strong>AI-assisted decompilation</strong>（AI 辅助反编译）方面的潜力，特别关注为 Mistral 7b 开发 <strong>asynchronous summarization script</strong>（异步摘要脚本）等工具，表明正在推动 AI 对开发者更加易用和实用。</p><p><strong>量子计算与 AI 的未来</strong>：多个 Discord 社区的讨论涉及量子计算与 AI 的交集，推测量子技术的进步将如何彻底改变 AI 模型处理。讨论还延伸到 DARPA 资助项目对 AI 发展的影响，以及在图像处理中探索用于 stable diffusion 的 <strong>encoder-based diffusion techniques</strong>（基于编码器的扩散技术），反映了对可能塑造 AI 未来的前沿技术的深厚兴趣。</p><p><strong>社区与协作倡议</strong>：各个社区强调了在 AI 模型开发方面促进协作和知识共享的努力，从 <strong>新的西班牙语 LLM 发布</strong> 到提高模型效率的 <strong>硬件实验</strong>。<strong>Perplexity AI</strong> 和 <strong>LlamaIndex</strong> 等平台因其创新功能和集成能力而受到关注。社区非常强调开源项目，如关于 <strong>Mistral 的开源承诺</strong> 的讨论，以及用于快速 RAG 系统部署的 <strong>R2R</strong> 等工具的开发，展示了 AI 社区内充满活力的协作精神。</p></div>

# PART 1: Discord 高层级摘要

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 总结

- **Polymind 的幽灵回响**：工程师们正在排查 **Polymind 的 UI bug**，即“幽灵消息”会一直残留，直到新的 token 流入。调试工作涉及 **前端 HTML 变量**（如 `currentmsg`），以清除残留内容。

- **模型评估热潮**：讨论重点关注了 **Miqu**，这是一个泄露的 **Mistral medium 等效模型**，尽管由于参数量比 **Mixtral** 更高导致速度较慢，但其表现令人赞赏。用户还在积极交流 **GGUF 量化模型** 的高效加载实践，包括禁用共享显存和调整 GPU 设置以防止 VRAM 溢出等技巧。

- **LangChain 的循环**：LangChain 因其声称的上下文重排序功能而受到抨击，`@capt.genius` 将其标记为“虚假实现”，用户认为这可能只是一个基础的循环机制。

- **Mistral 的神秘举动**：关于 **Mistral AI** 官网上开源模型措辞的变化引发了热议，鉴于 **Qwen** 等其他 AI 供应商推出的模型，人们开始猜测该公司对开源模型的承诺。

- **优化提示词工程 (Prompt Engineering)**：角色扮演和故事写作频道的工程师们表达了对擅长 **角色扮演 (role-playing)** 和 **故事写作 (story-writing)** 模型的渴望，希望模型具备 **一致的时间线、角色情感** 和 **情绪管理** 能力。

- **量化疑问与训练难题**：关于 **模型量化方法** 存在争论，`@orel1212` 考虑在小数据集训练中使用 **GPTQ** 而非 **QLORA**。关于 **Mistral 的 PDF 图像文本提取** 能力以及运行 **Mixtral 8x7b** 所需硬件的问题尚无明确答案。同时，`@dzgxxamine` 寻求关于教 LLM 理解和使用新发布的 Python 库的建议。

- **合并失败之谜**：来自 `@jsarnecki` 的一条消息报告了 **Orca-2-13b** 和 **WhiteRabbitNeo-13b** 之间一次不成功的模型合并，导致输出无法理解，合并过程的细节通过来自 [mergekit](https://github.com/cg123/mergekit) 的 readme 进行了分享。

- **AI 在代码清晰度中的作用**：在编码讨论中，人们对 AI 在 **AI 辅助反编译** 方面的潜力感到兴奋，`@mrjackspade` 期待一个无需手动代码重构的未来。此外，`@wolfsauge` 介绍了一个在 **Mistral 7b** 上测试过的 **异步摘要脚本**，并邀请同行尝试，该脚本可在 [GitHub](https://github.com/Wolfsauge/async_summarize) 上获取。

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

- **发布新的西班牙语 LLM**：西班牙总统 Pedro Sánchez 在世界移动通信大会上宣布创建一个 [西班牙语 AI 模型](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol)；这对 AI 中的西班牙语应用具有重要意义。

- **硬件趣闻：RTX 2060 实测**：硬件频道的讨论中，用户在没有桥接器的情况下尝试 **多 GPU 配置** 并获得成功，即使是型号不匹配的显卡也能驱动高达 70b 的模型，这表明硬件兼容性的规范正在发生变化。

- **针对 AI 工具的 WSL 网络修复**：用户通过绕过本地主机并使用网络 IP (`http://192.168.x.y:5000/v1/completions`)，解决了从 Windows 11 的 **WSL (Windows Subsystem for Linux)** 内部连接到 LM Studio 端点的问题，解决了 [此指南](https://superuser.com/a/1690272) 中概述的 WSL 独特网络行为。

- **文本转语音技术**：[Piper](https://github.com/rhasspy/piper) 是一种神经文本转语音系统，因其本地处理能力和对不同语言的支持而受到关注，这表明语言模型应用正趋向于高效、离线的解决方案。

- **技术人员驯服语言模型**：在模型讨论中，用户强调了特定量化参数（如 *mirostat_mode* 和 *mirostat_tau*）对于优化 LLM 性能的重要性，这些参数需要手动配置。对话还承认了人工评估在衡量模型性能中的作用。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **RWKV/Mamba 模型已为上下文长度灵活性做好准备**：`@vatsadev` 评论了通过微调扩展 **rwkv/mamba** 模型以获得更长上下文的可行性，从而减轻了从头开始预训练的需求。`@blackl1ght` 在使用 Tesla V100 和 Q8 量化的情况下，在 **Solar 10.7B** 上实现了 32k token 的上下文长度，且未达到内存限制，这表明了进一步扩展这些模型的潜力。
   
- **Ring Attention 激发了 100 万 Token 模型的可行性**：讨论强调了 **ring attention** 在管理高达 100 万 token 的模型中的应用，`@bloc97` 透露这可以应用于 7b 模型，展示了大规模模型在 attention 机制方面的进展。

- **量化技术有望实现高效推理**：提到了涉及 key-value cache 和可能使用 fp8 来优化推理效率的努力；`@stefangliga` 引入了量化 kvcache 的概念，这可能为更有效地处理更长上下文铺平道路。

- **基准测试指出存在选择性报告**：在关于 AI 模型基准测试的对话中，`@orabazes` 指出一条关于 AI 性能的推文省略了某些模型对比，暗示了基准测试透明度的必要性。

- **Mistral TOS 争议平息**：在关于 Mistral 服务条款（Terms of Service）的激烈辩论后，联合创始人 [@arthurmensch](https://twitter.com/arthurmensch/status/1762208241927233661) 的一条推文通过删除一个有争议的条款解决了问题，重申了该模型可开放用于训练竞争性 LLM。

- **结构化数据获得 LLM 的专门关注**：来自 @_akhaliq 推文的关于 **Google StructLM** 项目的新见解表明，尽管缺乏共享的训练方法或模型框架，但重点在于增强 LLM 对结构化数据的处理。

- **对 Mistral 循环困境的好奇**：`@.ben.com` 理论化地认为 **Mistral** 中重复的文本循环可能归因于经典的反馈系统问题，表明了在理解和解决模型振荡方面的技术好奇心。

- **对 Self-Extend 模型实践的高需求**：在 `@blackl1ght` 成功进行 self-extension 实验后，社区对本地模型改进的热情显而易见，导致大家共同关注如何复制那些能够改进内存管理和功能利用的配置。

**提到的链接**：

- [Hugging Face 上的 TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF)



---



## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

**EBDI Agent 的挑战与解决方案**：@.braydie 探索了用于 Agent 目标确定的 **EBDI 框架**，但在集成 [ReAct 框架](https://react-lm.github.io/) 后遇到了思维循环。他们研究了来自 [JASSS 论文](https://www.jasss.org/17/4/13.html) 的决策模型来解决这个问题。

**Mistral 进军以对抗 GPT-4**：一篇 [TechCrunch 文章](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/) 报道称，Mistral AI 的新模型 **Mistral Large** 定位于与 OpenAI 的 GPT-4 竞争，提供成本效益和未经审查的内容，目前已在 Azure 上可用。

**Prompt 保护悖论**：用户讨论了如何保护 Prompt 中的知识产权，结论是虽然版权可能涵盖确切的措辞，但通过语言变体复制想法可能是无法阻止的。

**文本分类策略**：@crifat 发起了关于文本分类方法的讨论，选择从基础模型和 Assistant 开始，绕过微调，将文本分类为“事实性”和“误导性”等类别。

**Meta-Prompting 引发热议与安全担忧**：**meta-prompting** 的概念是一个热门话题，有人声称通过先进技术生成了大量文档，但当一名用户分享 PDF 时，这些技术也引发了安全警报，导致该用户的账号被采取行动。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 总结

- **Perplexity AI 的图像生成潜力**：用户讨论了 Perplexity AI **Pro 版本**的图像生成能力，并提供了一个关于该主题的 [Reddit 讨论链接](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/)。此外，用户也对 Perplexity 停止支持 **Gemini AI** 表示了担忧，并确认该支持已经结束。

- **Mistral Large 备受瞩目**：Mistral AI 发布了名为 *Mistral Large* 的新语言模型，社区对其能力和可用性展开了讨论。该公告附带了 [Mistral AI 新闻稿链接](https://mistral.ai/news/mistral-large/)。

- **VPN 故障排查**：针对 Galaxy 手机上 Perplexity AI 的登录问题，通过关闭 VPN 得到了解决，并提供了关于 [VPN 分流隧道 (Split Tunneling)](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option) 的额外资源。

- **Perplexity 探索性搜索亮点**：用户分享了 Perplexity AI 对新奇话题的搜索结果链接，包括联想（Lenovo）的透明笔记本电脑和关于字母 K 的年龄查询。此外还进行了 iPhone 的对比，并推广了 Collection 功能，邀请用户在 Perplexity AI 上[创建自己的 Collection](https://www.perplexity.ai/collections/Make-your-own-rfj2pcwRS7WF7SFiTAxhJg)。

- **PPLLX-API 频道关于模型的讨论热烈**：技术讨论集中在模型信息和性能上，特别是关于模型参数的 JSON 链接，以及 `sonar-medium-online` 模型生成无关内容的问题。社区还讨论了 `pplx-70b-online` 模型的弃用，以及在进行 API 调用时正确编写 Prompt 的重要性，并明确引用了 API 文档中的 [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions) 部分。

---

## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord 总结

- **LlamaParse 助力 PDF 解析飞跃**：LlamaParse 作为一种增强对**包含表格和图表的 PDF** 理解的工具被引入，通过避免 PDF 解析错误来帮助 LLM 提供准确的答案。正如这篇 [推文](https://twitter.com/llama_index/status/1762158562657374227) 所强调的，它对检索增强生成 (RAG) 流程具有重要意义。

- **使用 MistralAI Large 模型强化索引**：LlamaIndex 在 10.13.post1 版本中集成了 **@MistralAI 的 Large 模型**，带来了接近 GPT-4 的能力，包括高级推理和 JSON 输出，详见[此公告](https://twitter.com/llama_index/status/1762231085243719748)。此外，LlamaIndex 新的分布式 Super-RAG 功能允许为任何 RAG 应用创建 API 服务，这些服务可以联网形成一个能够在整个网络运行查询的 Super-RAG，分享链接见[此处](https://twitter.com/llama_index/status/1762552542981230769)。

- **与 AGI Builders 和 FireWorksAI 的 AI 协作**：AGI Builders 见面会将邀请 LlamaIndex 的开发者关系副总裁分享关于 RAG 应用的见解，活动详情见[此处](https://t.co/kcoIhfgQqF)。在另一项合作中，LlamaIndex 和 FireworksAI_HQ 发布了针对 RAG 应用和使用 FireFunction-v1 进行函数调用 (Function Calling) 的 Cookbook 系列，提供完整的 API 兼容性，公告见[此处](https://twitter.com/llama_index/status/1762532341795487815)。

- **开发者面临的上下文构建难题**：在 ai-discussion 频道中，成员们讨论了如何为 **GPT-4 turbo 和 Gemini 1.5** 等编程 LLM 优化上下文，重点关注信息顺序、重复和结构化技术。此外，还讨论了使用 Llama2 进行开源文本生成，强调了 CSV 和 PDF 输入的非专有集成，并探索了构建面向 RAG 的开源 SDK 助手的切片 (Chunking) 和检索策略。

- **General 频道的故障排除与工具讨论**：General 频道充满了求助请求、带与不带 RAG 的 GPT-3.5 对比，以及将 LlamaIndex 与 Weaviate 等其他服务集成的讨论。此外，社区还就解决 LlamaIndex 在 macOS 上的安装问题进行了深入交流，并对话了关于创建用于 Golang 集成的 Agent 以及与 AWS Bedrock 的动态编排，显示出高度互动和技术导向的社区氛围。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **基于 Encoder 的 Diffusion 技术新进展**：用户讨论了用于 Stable Diffusion 的[基于 Encoder 的反转方法](https://tuning-encoder.github.io/)的优点，以及在增强 Stable Diffusion 性能时使用图像提示（IP）适配器所面临的挑战。

- **AI 与 DARPA 的微妙博弈**：关于 DARPA 资助的项目对 AI 研发影响的辩论异常激烈，其中还穿插了关于在征兵活动中使用动漫角色以及军事与娱乐类型交叉的幽默讨论。

- **AI 未来的量子飞跃**：对话围绕量子计算在处理 AI 模型（尤其是 Transformer）中的未来角色展开，讨论了量子纠错的现状及其在 AI 计算中的潜力。

- **应对内容审核的雷区**：对话深入探讨了内容审核方面的困难，特别是涉及儿童性虐待材料（CSAM）的问题，思考了举报工具的有效性以及平台的责任。

- **开源与闭源 AI 之间的平衡**：针对 AI 模型的发布策略进行了激烈的交流，对比了像 Mistral Large 这样的开源方法与闭源模型，并考虑了支持持续 AI 研发的商业化模式。

- **为语言模型添加水印**：一篇[研究论文](https://arxiv.org/abs/2402.14904)揭示了通过水印文本检测训练数据的发现，表明在仅使用 5% 训练数据的情况下，识别带有水印的合成指令具有**极高的置信度**。

- **Genie 在人形机器人领域大显身手**：来自 Google DeepMind 新论文的一段 [YouTube 视频](https://www.youtube.com/watch?v=gGKsfXkSXv8)展示了 AI 在**人形机器人**领域的最新进展。

- **有限元（FEM）学习成功的局限性**：在有限元分析（FEM）领域，虽然研究得到了认可，但共识认为使用 FEM 网格和模型进行学习的方法**不如**传统的 FEM 有效，并引用了一篇[特定论文](https://arxiv.org/abs/2302.04107)。

- **攻克傅里叶变换挑战**：强调了神经网络合成中涉及离散傅里叶逆变换的技术问题，指出代码可能会从**使用 `torch.vmap` 进行重构**中受益，以提高 VRAM 效率，这在分享的 [GitHub 仓库](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177)中得到了演示。

- **思考 Transformer 学习实验**：有一个关于进行 Transformer 学习实验的独立查询，旨在辨别虚构物体之间的尺寸关系，以理解并渲染对比图像，但未提供进一步的讨论或数据。

---

## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord 摘要

- **付费墙阻碍《金融时报》文章分享**：`@xeophon.` 分享了一篇讨论微软入股法国 AI 初创公司 Mistral 的[《金融时报》文章](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb)，但因其糟糕的移动端付费墙体验而受到批评。据称，微软的投资旨在支持基于 Azure 的商业语言模型的部署。

- **辩论思维链（CoT）的有效性**：来自 `@sid221134224` 和 `@xeophon.` 等用户的报告显示，CoT 提示的效果褒贬不一，从 Gemini Pro 等模型的性能下降到将其作为微调的标准实践不等。

- **讨论多语言模型对齐挑战**：讨论集中在**多语言模型**的对齐上，质疑价值观应该是统一的还是具有文化针对性的，并强调了 GPT-4 安全措施的问题，据 [The Register](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/) 报道，这些措施可以通过苏格兰盖尔语等低资源语言绕过。

- **关于 AI 工具和贡献者写作习惯的见解**：贡献者 `@natolambert` 和 `@xeophon.` 分享了他们对写作工具及其流程的偏好，例如 Notion, Grammarly, Typora 和 Obsidian，展示了创作文本内容的多种方法。

- **对 DeepMind 的 Genie 和 Mistral 商业策略的兴奋**：针对 DeepMind 的新基础世界模型 Genie 展开了热烈讨论，该模型展示了从视频中学习世界动态的潜力；同时还讨论了 Mistral 的资源消耗和定价策略，并根据现有的公开信息推测了 Token 的使用情况。

---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord 总结

- **Mistral Large 成为新的 AI 竞争者**：`@alexatallah` 推出了一款名为 **Mistral Large** 的新模型，其定位介于 GPT-4 和 Claude 2 之间，具备 32,000 token 上下文窗口和多语言支持等高级功能，可在 [OpenRouter's Mistral Large](https://openrouter.ai/models/mistralai/mistral-large) 获取。Mistral 系列产品的价格已进行调整，因其高性价比，官方推荐使用 Mistral 7B Instruct 和 Mixtral 8x7B Instruct。

- **Sonar 凭借联网能力表现强劲**：`@alexatallah` 分享了 **Perplexity's Sonar** 模型的发布，其中包括一个联网版本，在成本效益和速度上均优于前代产品，地址位于 [Perplexity's Sonar 8x7B Online](https://openrouter.ai/models/perplexity/sonar-medium-online)。由于 PPLX 模型定于 3 月 15 日弃用，建议用户迁移到 Sonar。

- **OpenRouter Playground 获得提升**：OpenRouter Playground 的重要更新包括新增了 Top P、Top K 和惩罚项（penalties）等参数，改善了用户交互。修复了包括 Perplexity、Mistral 和 Gemma 在内的多个模型的系统消息（system message）问题，优化了性能。

- **新 AI 创作工具上市**：**Videotok** 在 Product Hunt 上线，这是一个 AI 驱动的短视频制作平台，可分享的帖子见 [Borja Soler's tweet](https://x.com/borjasolerr/status/1762025283597582807?s=20)。此外，[Blust AI 平台](https://blust.ai)也已面世，集成了多个 AI 应用，集成步骤详见其 [文档](https://docs.blust.ai/docs/integrating-ai-tools/)。

- **使用 Make.com 自动化一切**：`@jim14199` 介绍了一款新应用，通过将 OpenRouter 与众多其他应用连接，实现无代码 AI 工作流自动化，可在 [Make.com OpenRouter Integration](https://www.go-synergetic.com/apps/openrouter) 获取。虽然有关于 Blust AI 平台上模型功能问题的报告，但未提供具体细节。

- **OpenRouter 引发社区讨论**：社区就参数 API 等新功能更新展开了热烈讨论，并建议根据用户偏好和生产环境用户输入进行定制。Mistral Large 的发布引起了明显的兴奋，同时还讨论了 Perplexity 模型错误以及选择利用 OpenRouter AI 模型的最佳接口，详见 [OpenRouter](https://openrouter.ai)。



---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord 总结

**祝大家有美好的一周并在考试中取得好成绩**：社区成员分享了从对本周的美好祝愿到考试压力的各种情绪。

**寻求快速批处理方案**：讨论了查询 GPT-4 的最佳批处理方法，强调了快速高效的批处理对于缩短完成时间的重要性。

**服务中断与技术协作**：用户报告了 Hugging Face Inference API 出现的 504 超时错误，凸显了服务的不稳定性；同时，社区内正在进行持续对话，以促进机器学习项目的协作开发。

**卷积神经网络沉浸式学习机会**：发出了针对 CS231n（*用于视觉识别的卷积神经网络*）学习小组的公开邀请，并为感兴趣的参与者提供了课程作业和模块链接。[CS231n 课程](https://cs231n.github.io/)

**Scale AI 的崛起与 VLM 分辨率方案**：文章和讨论展示了 Scale AI 在数据标注领域增长至 73 亿美元估值的骄人战绩，以及通过使用高分辨率图像的多重裁剪来克服 Vision-Language Models (VLM) 分辨率问题的创新方案。[Scale AI 的故事](https://www.turingpost.com/p/scaleai) 和 [VLM 分辨率解决方案](https://huggingface.co/blog/visheratin/vlm-resolution-curse)

**AI 伦理与性能的发展与辩论**：社区分享了对“权重开放 (open-weight)” AI 模型发表评论的机会、一个评估各种模型响应时间和价格的新 Performance LLM Board，以及一个详细的 Imagic 论文复现尝试，该论文涉及使用 Diffusion Models 进行基于文本的图像编辑。[Open AI Model Weights 评论](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/) 和 [Imagic 论文复现](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a)

**对 Diffusion Model 工具的不满**：针对 Playground v2.5 中使用 eps 预测以及选择使用 EDM 框架而非 zsnr 的做法，出现了一些不满的声音。

**计算机视觉中的数据量与字符识别**：有人对微调所需的数据集大小表示关注，特别是针对旨在识别复杂字符的模型（如高棉语），由于其符号丰富的脚本，这带来了独特的挑战。

**探索 NLP 领域**：对话涉及序列分类的最佳实践、寻找生成式 QA 模型、适用于较小数据集的 Embedding 模型推荐、LLM 的电子邮件压缩策略，以及构建针对医学术语细微差别定制的 Medical Transformer。推荐的 Embedding 模型包括 [BAAI 的 bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) 和 [thenlper 的 gte-small](https://huggingface.co/thenlper/gte-small)。



---



## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **Unit Style 学习引起工程师关注**：机器学习算法的精度是讨论焦点，涉及“Unit Style”向量的对话强调了 **RMSNorm** 是实现“单位长度 (unit length)”向量并避免在不同尺度间转换的关键组件。

- **Mistral 揭开大模型的面纱**：EleutherAI 宣布发布 *Mistral Large*，这是一款新的 SOTA 语言模型，承诺在基准测试中取得尖端结果。该模型现已通过 *la Plateforme* 和 *Azure* 提供，详情见 [Mistral 新闻页面](https://mistral.ai/news/mistral-large/)。

- **对 lm-harness 的赞赏与行动号召**：一位成员强调了 EleutherAI 的 **lm-harness** 的影响，最近的一篇论文认可了它在自回归语言模型 Few-shot 评估中的重要性，相关讨论串包括了增强其功能的指南和讨论。

- **对“能量 (Energy)”的解释引发询问并承诺调查**：成员们对模型微调及相关方程中使用的术语“能量 (energy)”表示困惑并需要调查，承认对该概念缺乏直觉和清晰度。

- **NeoX 的技术故障排除**：用户讨论了挑战并寻求关于 **DeepSpeed** 配置以及在 **CoreWeave** 基础设施上为 GPT-NeoX 设置多节点训练环境的建议，建议方向包括利用 Kubernetes 以及在 2 节点 4 GPU 的设置中使用 **slurm** 或 **MPI**。



---

## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord 总结

- **GTC 的兴奋与规划**：成员们正期待参加 **Graphics Technology Conference (GTC)**，并建议为参会者举行聚会并开设专用频道。有人分享了一个基于 CUDA 的 [GitHub 上的 Gameboy 模拟器](https://github.com/krocki/nvgb)，展示了将 CUDA 用于经典游戏模拟的巧妙应用。

- **释放 QLoRA 的速度**：一个[提速 5 倍且节省 60% 显存的 QLoRA 微调 GitHub 仓库](https://github.com/unslothai/unsloth)受到关注，该项目能高效地加速模型。

- **深入的 CUDA 讨论**：社区参与了关于 CUDA 与 Vulkan 互操作性的讨论，回顾了通过 TorchFP4Linear 层提升 GPU 利用率的改进，并分享了选择性层替换技术的更新。此外，还交流了关于 GPU 内存访问延迟的见解，并辅以各种外部参考资料，如 NVIDIA A100 内存访问的检查论文和 [torch-bnb-fp4 的速度测试脚本](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py)。

- **优化 PyTorch 工作流**：对话内容包括 PyTorch 起源于 Torch7 的历史，以及加速 `cpp_extension.load_inline` 编译时间的技巧。此外，还提供了在 PyTorch 中将自定义 Triton kernel 与 `torch.compile` 集成的指导，并引用了 [GitHub 示例](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661)。

- **Attention 算法的效率**：分享了关于高效 softmax 近似、softmax 的 base2 技巧的论文和资源，以及 [OpenAI Triton 示例中令人印象深刻的增量 softmax 实现](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)。此外，还提到了来自 Quake3 的经典快速平方根倒数（fast inverse square root）技巧。

- **NVIDIA 职位机会**：NVIDIA 发布了一个职位空缺，寻求 CUDA 和 C++ 专家，有意者可私信简历，**JobID 为 JR1968004**。

- **初学者疑问**：讨论了深入学习 CUDA Mode 所需的条件，包括是否需要具备 PyTorch 知识等先决条件，并建议在尝试 CUDA 之前先从 fast.ai 课程开始。

- **术语澄清**：澄清了缩写 **AO** 代表 **Architecture Optimization**（架构优化）。

- **CUDA Attention 机制的进展**：讨论了提升 ring attention 性能的技巧，并安排了 flash attention 论文的阅读与讨论。分享了一个对比 ring attention 和 flash attention 的 [Colab notebook](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X) 以征求社区反馈，同时还分享了一篇关于在 NVIDIA Hopper 架构上实现 FlashAttention-2 的论文。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **结构化输出接口征求社区意见**：提出了一个用于获取结构化模型输出的直观接口，用户 `@bagatur` 请求反馈。详情在 [langchain-ai/langchain 的 GitHub RFC](https://github.com/langchain-ai/langchain/discussions/18154) 中讨论。

- **LangChain 爱好者的新工具与指南**：发布了多种资源，包括 `@dctanner` 推出的用于内容抓取的 [UseScraper.com](https://usescraper.com)，`@ldeth256` 编写的详尽的 [LlamaCpp 与 Python 集成指南](https://python.langchain.com/docs/integrations/llms/llamacpp)，以及对 [validate.tonic.ai](https://validate.tonic.ai/) 的推荐，这是由 `@locus_5436` 开发的一个用于可视化 RAG 系统评估的平台。

- **寻求 Chat 中函数调用的临时解决方案**：`@sectorix` 询问在 **Ollama** 预期功能发布之前，如何在 **Mistral** 等开源模型的聊天功能中启用函数调用（function calling）的临时方案。

- **创新 RAG 平台聚焦**：展示了一系列项目：由 `@emrgnt_cmplxty` 开发的 RAG 系统框架 **R2R**（[GitHub 链接](https://github.com/SciPhi-AI/R2R)），由 `@robertoshimizu` 开发的医疗咨询平台 **IntelliDoctor.ai**（[网站](https://intellidoctor.ai)），以及 `@andysingal` 提到的 **LangGraph** 在增强迭代代码生成方面的方法。

- **教程中热门的 LangGraph 与 AI 技术**：教程强调了新颖的应用，如 `@tarikkaoutar` 在 [YouTube](https://www.youtube.com/watch?v=q5LvDHiSBy4) 上展示的基于 LangGraph 的多智能体系统（multi-agent systems），以及 `@jasonzhou1993` 提出的移动端 AI 对话副驾驶概念，该内容也可在 [YouTube](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg) 上查看。



---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **Mistral 的开源忠诚度受到质疑**：鉴于 Mistral 与 Microsoft 的合作伙伴关系，用户对其开源实践的承诺产生了怀疑；尽管 [MistralAI 的 CEO 重申了对开源权重模型的承诺](https://fxtwitter.com/casper_hansen_/status/1762159643344662859)，但仍有人怀疑其正转向以利润为中心。
- **Gemma 超越竞争对手**：**Gemma 模型**取得了显著进展，正如 `@nanobitz` 所分享的，在使用 FA2 的情况下，其速度比 [Hugging Face](https://huggingface.co/unsloth) 快 2.43 倍，且节省了 70% 的 VRAM。他还分享了两个免费使用的 Notebook 链接，分别用于 [Gemma 7b](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing) 和 [Gemma 2b 模型](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing)。
- **LoRA-The-Explorer 发布用于训练**：`@caseus_` 介绍了一种名为 **LoRA-the-Explorer (LTE)** 的高效训练神经网络新方法，其中包括[并行低秩适配器方法](https://minyoungg.github.io/LTE/)和[多头 LoRA 实现](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py)。
- **文档差异讨论**：GitHub pull request 中关于正确文档的引用引发了讨论，随后在 [axolotl-dev 频道](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256)分享了[相关材料](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md)。
- **用于 RAG 部署的 R2R 框架**：用户 `emrgnt_cmplxty` 推出了 **R2R**，这是一个旨在快速开发和部署 **RAG 系统**的框架，已在 [GitHub](https://github.com/SciPhi-AI/R2R) 上发布供社区使用。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 总结

**Zero-Shot 模型对比**：`@eugeneyan` 澄清说，一条关于 AI 模型与 GPT-4 对比的推文线程引用的是它们的 **zero-shot** 性能指标，这对于理解模型在没有 fine-tuning 情况下的能力至关重要。

**Mistral 与 Microsoft 携手前进**：`@__chef__` 宣布了 **Mistral Large**，宣传其基准测试性能并透露了与 Microsoft 的合作伙伴关系，这一重大进展在 [Mistral Large 的发布页面](https://mistral.ai/news/mistral-large/)上备受关注。

**Cloudflare 提供简化的 AI 解决方案**：`@henriqueln7` 强调了 Cloudflare AI Gateway 的发布，关注其单行代码易用性，以及强大的分析、日志和缓存功能，详情见 [Cloudflare AI Gateway 文档](https://developers.cloudflare.com/ai-gateway/)。

**Mistral Au 与 RAG 集成用于高级应用**：`@ashpreetbedi` 赞扬了 **Mistral Au Large** 与 RAG 的集成，指出其改进了 function calling 和推理能力，并引导用户查看其在 [phidata/mistral](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral) 的 GitHub cookbook。

**RAG 资源发布引发关注**：`@dimfeld` 宣布了 Jason Liu 即将出版的关于 RAG 的电子书，旨在解释不同复杂程度的概念，`@thenoahhein` 认为这对于 Twitter 数据摘要任务特别有用；该电子书的仓库可以在 [n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag) 找到。

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 摘要

- **AI 在翻译中迷失**：在一个有趣的事件中，`@derekpwillis` 遇到了语言切换问题，**chatgpt-3.5-turbo** 错误地将本应为英文的文档使用了西班牙语标题，滑稽地将 "Taking Advantage of the Internet" 翻译成了 *"Sacándole Provecho a Internet"*。`@simonw` 将此与之前 ChatGPT 和 Whisper 将英国口音误听为威尔士语的 Bug 进行了对比，并建议使用提示词要求系统 "Always use English"（始终使用英语）以防止此类语言混淆。

- **LLM 插件连接 Python 与 Groqcloud**：`@angerman.` 介绍了 [LLM 插件](https://pypi.org/project/llm-groq/)，该插件允许 Python 开发者访问 [Groqcloud](https://console.groq.com) 模型，如 `groq-llama2` 和 `groq-mixtral`。该插件最近已更新以支持 streaming（流式传输），并且有传言称 LLM 将推出聊天 UI，尽管尚未公布发布日期。

- **Python 打包变得简单**：`@0xgrrr` 提供了帮助，分享了一个关于如何为 Python 项目打包以便他人上传和贡献的 [教程](https://packaging.python.org/en/latest/tutorials/packaging-projects)，并举例说明这是一个简单的过程。

- **选择 Fly 以获取 GPU 能力**：在回复 `@kiloton9999` 时，`@simonw` 解释说，在 Datasette 开发中选择 Fly GPU 的部分原因是 Fly 的赞助以及其 GPU 能够缩减至零（scale to zero）的能力，这对项目的资源需求非常有价值。



---



## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

- **挑战 LLM 评估标准**：分享了一篇 [学术论文](https://arxiv.org/abs/2402.13887)，批评了当前基于概率的 Large Language Models 评估方法与基于生成的预测能力不匹配，强调了在理解为何会出现这些差异方面的空白。

- **数据集转换异常**：在 JSON 到 Parquet 的数据转换过程中观察到了一个隐藏的空字符串（null string）问题，这是通过 Hugging Face 的直接 JSON 上传和转换过程检测到的，展示了数据集准备中的细微差别。

- **利用 RAG 推进代码库助手**：围绕为代码库创建 Retrieval-Augmented Generation (RAG) 机器人的讨论深入探讨了 LangChain 的 Git 加载器集成、针对编程语言的分段、LlamaIndex 的 Git 导入器以及在检索过程中使用 OpenAI embeddings，在开发者助手技术方面取得了进展。

- **探索 RAG 和 LLM 中的端到端优化**：重点讨论了使用梯度对 RAG 和 LLM 进行联合端到端优化的查询，包括对 [LESS 论文](https://arxiv.org/abs/2402.04333) 的研究，该论文详细介绍了一种检索具有相似预计算梯度特征的训练示例的方法，而不是通过数据选择进行反向传播。

- **情商基准走向国际化**：EQ-Bench 扩展了 **德语支持**，引发了关于语言间翻译质量和情感细微差别的讨论，此前 GPT-4 在德语中的得分为 81.91，而英语为 86.05；可以在其 [GitHub](https://github.com/EQ-bench/EQ-Bench) 上进一步探索这一倡议，它强调了语言流利度在模型基准测试中的重要性。



---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 摘要

- **FireFunction V1 凭借 GPT-4 级别的能力点燃热度**：[FireFunction V1](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling) 是一款具备 **GPT-4 级别结构化输出**和决策路由功能的新模型，因其令人印象深刻的低延迟、开放权重和商业可用性而受到关注。目前关于延迟细节的讨论正在进行中，特别是响应延迟是指首个 token 时间 (time to first token) 还是完成时间。

- **R2R 助力生产级 RAG 系统**：**R2R** 框架发布，旨在简化生产就绪型 RAG 系统的开发和部署。该快速开发框架的详细信息可以在 [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) 找到，社区进一步询问了它与现有的 [agentmemory 框架](https://github.com/JoinTheAlliance/agentmemory)有何不同。

- **GPT-4 在药物信息方面证明了其实力**：一位用户成功使用 GPT-4 生成了关于药物的详细信息卡片，概述了机制、副作用和疾病靶点。然而，GPT-4 无法将图像整合到输出中，这被指出是一个局限性，影响了像 Anki 的图像遮挡 (image occlusion) 等方法。

- **追求速度**：用户对 **OpenAI API 以秒计的延迟**表示担忧，引发了关于**专用托管 (dedicated hosting)** 是否能解决此问题的讨论。这种不满也延伸到了 Azure 的托管体验，用户分享了对性能的失望。

- **共同推动改进 RAG 系统**：一位用户分享了 RAG 系统的增强方案，寻求社区对拟议改进的反馈。感兴趣的各方可以通过 [GitHub](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md) 上的提案提供意见。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- **Gemma 获得具有变革性的 Token**：`@imonenext` 宣布将 `<start_of_turn>` 和 `<end_of_turn>` token 集成到 **Gemma** 模型中，增强了其处理指令/RL 微调的能力，该模型可在 [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens) 上访问。
- **Token 集成需手动操作**：正如 `@imonenext` 所解释的，向 **Gemma** 添加指令微调 token 需要涉及复制 tokenizer 的手动程序，以确保 token 的一致性，由于保留了原始指令 token，目前没有收到问题的报告。



---



## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

鉴于提供的信息有限，无法创建实质性的摘要。唯一的消息是用户在 off-topic 频道分享的一个 YouTube 视频链接，该链接不涉及任何与工程师受众相关的技术讨论或细节导向的主题。如果视频包含与 AI 或工程相关的技术内容，由于提示中未包含这些信息，因此在摘要中包含它是不合适的。



---



## [AI Engineer Foundation](https://discord.com/channels/1144960932196401252) Discord 摘要

- **Agent Protocol V2 编码实战**：`_z` 邀请社区参加 [YouTube 直播编码会议](https://youtube.com/live/zrJuNUGYKJg?feature=share)，重点关注作为 Agent Protocol V2 里程碑一部分的 **Agent Protocol 配置选项 RFC**。直播鼓励实时互动和贡献。



---

# 第 2 部分：分频道详细摘要与链接



### TheBloke ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/1211589709927682108) (1277 条消息🔥🔥🔥): 

- **Polymind UI 中的幽灵消息**：用户讨论了刷新或清除内存后 Polymind UI 中出现幽灵消息的持久问题；在模型重新开始流式传输 token 之前，会显示之前的消息。调试工作正在进行中，包括检查前端 HTML 变量（如 `currentmsg`）以及清除残留内容的实现。

- **比较 LLM 性能**：在一系列消息中，用户讨论了 Miqu 和 Mixtral 等模型的性能，指出了在速度、参数大小和内存消耗方面的差异。Miqu 是泄露的 Mistral medium 等效模型，尽管由于参数较高而速度较慢，但其性能优于 Mixtral。

- **GGUF 量化加载问题**：用户交流了高效加载 GGUF 量化模型的建议，讨论了层卸载 (layer offload) 数量和在 Nvidia 控制面板中禁用共享显存等设置，以避免溢出到效率较低的共享 VRAM 中。

- **LangChain 审查**：`@capt.genius` 指责 LangChain 在其源代码中对上下文重排序（context reordering）进行了“虚假实现”，仅使用了一个简单的循环，似乎只是根据索引的奇偶性来交替数组元素，并称 LangChain 为“骗局”。

- **Mistral 的开源方向**：讨论中提到 Mistral 在其官网上将开源模型的引用改为过去时，引发了关于该公司未来是否继续提供开源模型的猜测。用户们辩论了 Mistral 以及 Qwen 等其他 AI 提供商所提供模型的有效性和愿景。

- **不同环境下的模型使用**：`nigelt11` 对 LangChain Chatbot 应用在本地机器和 HuggingFace Spaces 之间表现不一致感到沮丧，理由是输出质量存在差异，并寻求对潜在原因的见解。


**提到的链接**：

- [GOODY-2 | 全球最负责任的 AI 模型](https://www.goody2.ai/chat)：介绍具有下一代伦理对齐的新型 AI 模型。立即聊天。
- [Nod Cat Hyper GIF - Nod cat hyper - 发现并分享 GIF](https://tenor.com/view/nod-cat-hyper-gif-9540792418684483949)：点击查看 GIF
- [技术](https://mistral.ai/technology/#models>)：掌控前沿 AI
- [Rapeface Smile GIF - Rapeface Smile Transform - 发现并分享 GIF](https://tenor.com/view/rapeface-smile-transform-gif-12599812)：点击查看 GIF
- [American Psycho Impressive GIF - American Psycho Impressive Very Nice - 发现并分享 GIF](https://tenor.com/view/american-psycho-impressive-very-nice-coping-patrick-bateman-gif-26518058)：点击查看 GIF
- [TheBloke/deepseek-coder-33B-instruct-GGUF · Hugging Face](https://huggingface.co/TheBloke/deepseek-coder-33B-instruct-GGUF)：未找到描述
- [Qwen](https://qwen.readthedocs.io/)：未找到描述
- [Thebloke.Ai Ltd - 公司简介 - Endole](https://suite.endole.co.uk/insight/company/15361921-thebloke-ai-ltd)：未找到描述
- [从命令行禁用/启用 CUDA Sysmem Fallback Policy](https://gist.github.com/itsdotscience/4e29dca91f010a1873d1083fae94a655)：从命令行禁用/启用 CUDA 系统内存回退策略 - a
- [明日方舟 BGM - Boss Rush 30min | Arknights/明日方舟 导灯的试炼 OST](https://www.youtube.com/watch?v=KQ1MKDDYvF8)：作业用明日方舟 Trials for Navigator #1 Lobby Theme OST Boss Rush 30min Extended. Monster Siren Records: https://monster-siren.hypergryph.com Wallpaper: Coral Coast s...
- [OpenAI 内部人士谈未来场景 | Scott Aaronson](https://youtu.be/gGsh0_-q7LI)：这是 Scott Aaronson 在佛罗里达大西洋城大学未来思维中心举办的 MindFest 上的演讲，由 Susan Schneider 牵头。提到的链接...
- [《明日方舟》四周年 [ 萨米：北极星 ] 特别 PV](https://www.youtube.com/watch?v=0m18wLQInnU)：关于萨米：北极星的特别 PV。来源：https://www.bilibili.com/video/BV13k4y1J7Lo ============================================= Group Fb: https://www.facebook.com...
- [gguf (GGUF)](https://huggingface.co/gguf)：未找到描述
- [GitHub - facebookresearch/dinov2: 用于 DINOv2 自监督学习方法的 PyTorch 代码和模型。](https://github.com/facebookresearch/dinov2?tab=readme-ov-file>)：用于 DINOv2 自监督学习方法的 PyTorch 代码和模型。 - facebookresearch/dinov2
- [GitHub - BatsResearch/bonito: 一个轻量级库，用于为你的数据生成合成指令微调数据集，无需 GPT。](https://github.com/BatsResearch/bonito)：一个轻量级库，用于为你的数据生成合成指令微调数据集，无需 GPT。 - BatsResearch/bonito
- [搜索结果 | bioRxiv](https://www.biorxiv.org/search/scgpt)：未找到描述
- [[蔚蓝档案] [AI 天童爱丽丝] Chipi Chipi Chapa Chapa (Dubidubidu)](https://www.youtube.com/watch?v=2wsLZyvaqlE)：AI 唱歌工具 歌声音色转换模型：so-vits-svc 4.1: https://github.com/svc-develop-team/so-vits-svc 角色配音：ブルアカ 天童アリス（CV：田中美海）原曲：Christell - Du...
- [巴布亚新几内亚失落的世界 🇵🇬](https://www.youtube.com/shorts/yL5dgjdFzHI)：完整剧集请看：https://youtu.be/nViPq2ltGmg?si=fkVDkqdSTZ3KWZxJ 感恩应该是唯一的态度
- [UAI - 为每个人、每个地方释放 AI 的力量：介绍通用 AI 推理](https://rentry.co/UAI-universal-ai-inference)：以下文本完全由 Mistral 的优秀模型编写。我听到很多关于需要更多开源模型和社区访问 AI 技术的议论。似乎...
- [用于人工智能的大脑类器官储备池计算 - Nature Electronics](https://www.nature.com/articles/s41928-023-01069-w)：一种人工智能硬件方法，利用大脑类器官中生物神经网络的自适应储备池计算，可以执行语音识别和非线性方程等任务...

- [利用 AI 和可访问性加速聚变能的到来](https://news.mit.edu/2023/fast-tracking-fusion-energy-with-ai-and-accessibility-0901)：MIT Plasma Science and Fusion Center 将获得 DoE 的支持，以改善聚变数据的获取并增加劳动力多样性。该项目由 Christina Rea 领导。
- [添加使用 "//" 跳过 GateKeeper 的功能 · DocShotgun/PolyMind@2d4ab5c](https://github.com/DocShotgun/PolyMind/commit/2d4ab5c02fa91d4558d77f92fec9a73ac20a2537)：未找到描述
- [GAA (Gate-All-Around) 晶体管即将到来](https://youtu.be/5RPFfPtgw7g)：链接：- Asianometry 通讯：https://www.asianometry.com - Patreon：https://www.patreon.com/Asianometry - Threads：https://www.threads.net/@asianometry-...
- [Fall Out Boy - 介绍 Crynyl™️](https://www.youtube.com/watch?v=l75SlbaZxtA)：介绍 Crynyl™，灌注了真实泪水的唱片，以实现最大的情感忠实度。So Much (For) Stardust 现已在 https://crynyl.c... 开启预订。
- [在 `delete_all()` 中添加 `with_children: bool`，允许在 `is_root=True` 的基类上调用 delete_all，以同时删除任何子类行。由 TheBloke 提交 · Pull Request #866 · roman-right/beanie](https://github.com/roman-right/beanie/pull/866)：一个简单的更改：在 `delete_all()` 中添加了 `with_children: bool`，并将其传递给 `find_all()`。当存在继承树时，这允许使用基类删除集合中的所有文档...
- [scGPT：利用生成式 AI 构建单细胞多组学基础模型 - Nature Methods](https://www.nature.com/articles/s41592-024-02201-0)：scGPT 使用超过 3300 万个单细胞 RNA 测序图谱进行预训练，是一个通过迁移学习促进广泛下游单细胞分析任务的基础模型。
- [一次指令，多轮一致对话：一种高效的对话微调框架](https://arxiv.org/html/2402.06967v1)：未找到描述
- [scGPT：利用生成式 AI 构建单细胞多组学基础模型](https://doi.org/10.1101/2023.04.30.538439)：生成式预训练模型在自然语言处理和计算机视觉等各个领域取得了显著成功。具体而言，大规模多样化数据集的结合...
- [Mistral AI | 开放权重模型](https://web.archive.org/web/20240225142431/https://mistral.ai/)：掌控之中的前沿 AI
- [Mistral AI | 掌控之中的前沿 AI](https://mistral.ai/)：掌控之中的前沿 AI

  

---


### TheBloke ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/1211631811420098620) (1005 条消息🔥🔥🔥): 

- **LLMs 角色扮演与故事写作愿望清单**：用户讨论了他们对角色扮演和故事写作模型的愿望清单，例如将 `Choose Your Own Adventure` 游戏与 `Dungeons and Dragons` 背景知识相结合，并保持故事事件和时间线的一致性。其他愿望包括管理角色的情感和情绪，以及增强场景中的描述性叙述。

- **Prompt 有效性策略**：参与者分享了关于提示词中 Token 的位置（尤其是前 20 个）如何极大地塑造 LLM 响应的策略。他们还探讨了使用伪代码和标签设计角色卡的方法，以引导 AI 在角色刻画中保持一致性。

- **故事续写的模型行为**：用户表达了对能够遵循长多轮上下文而不丢失连贯性，并保持符合角色性格的响应，而不是逐字重复角色卡信息的模型的需求。

- **角色扮演中的“情绪”处理**：对话涵盖了如何可能使用第二个分析模型在故事交互中编写情绪变化的代码。目标是让分析模型评估并更新诸如“愤怒”之类的状态变量，然后将其反映在主故事中，可能使用语言字符串而非数值。

- **当前 LLMs 的挑战**：有人对进入模型创作的“黑暗时代”表示感慨，认为闭源开发盛行，且缺乏能够像角色扮演者所期望的那样有效维持故事细节的新型鲁棒模型。一些用户还分享了在为 Miqu 等不同模型寻找最佳系统提示词、处理过于积极的模型以及追求模型更深层次的初始细节方面遇到的挑战。

**提到的链接**：

- [未找到标题](https://tenor.com/view/nothing-is-real-jack-as-we-see-it-everything-is-fake-none-of-this-is-real-gif)：未找到描述
- [Hold Stonk Hold GIF - Hold Stonk Hold Wallace Hold - 发现并分享 GIF](https://tenor.com/view/hold-stonk-hold-wallace-hold-braveheart-stonks-gif-20142609)：点击查看 GIF
- [Copy Paste Paste GIF - Copy Paste Paste Copy - 发现并分享 GIF](https://tenor.com/view/copy-paste-paste-copy-ctrl-c-ctrl-v-gif-12913156)：点击查看 GIF

- [Vorzek Vorzneck GIF - Vorzek Vorzneck Oglg - 发现并分享 GIF](https://tenor.com/view/vorzek-vorzneck-oglg-og-lol-gang-gif-24901093): 点击查看 GIF
- [Solid Snake Solid GIF - Solid snake Solid Snake - 发现并分享 GIF](https://tenor.com/view/solid-snake-solid-snake-not-metal-gear-solid-gif-14480942659066849172): 点击查看 GIF
- [He Cant Keep Getting Away With It GIF - He Cant Keep Getting Away With It - 发现并分享 GIF](https://tenor.com/view/he-cant-keep-getting-away-with-it-gif-19335672): 点击查看 GIF
- [Bane No GIF - Bane No Banned - 发现并分享 GIF](https://tenor.com/view/bane-no-banned-and-you-are-explode-gif-16047504): 点击查看 GIF
- [Skeptical Futurama GIF - Skeptical Futurama Fry - 发现并分享 GIF](https://tenor.com/view/skeptical-futurama-fry-hmmm-i-got-my-eyes-on-you-gif-17101711): 点击查看 GIF
- [Omni Man Invincible GIF - Omni Man Invincible Look What They Need To Mimic A Fraction Of Our Power - 发现并分享 GIF](https://tenor.com/view/omni-man-invincible-look-what-they-need-to-mimic-a-fraction-of-our-power-gif-25672186): 点击查看 GIF
- [Spiderman Everybody GIF - Spiderman Everybody Gets - 发现并分享 GIF](https://tenor.com/view/spiderman-everybody-gets-one-of-us-family-guy-gif-22763691): 点击查看 GIF
- [deepseek-ai/deepseek-coder-7b-instruct-v1.5 · Hugging Face](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5): 未找到描述
- [You Dont Say Frowning GIF - You Dont Say Frowning Coffee - 发现并分享 GIF](https://tenor.com/view/you-dont-say-frowning-coffee-gif-14597430): 点击查看 GIF
- [Nipples GIF - Nipples - 发现并分享 GIF](https://tenor.com/view/nipples-gif-18634175): 点击查看 GIF
- [How The Might Have Fallen Wentworth GIF - How The Might Have Fallen Wentworth S06E11 - 发现并分享 GIF](https://tenor.com/view/how-the-might-have-fallen-wentworth-s06e11-correctional-center-prison-gif-22933267): 点击查看 GIF
- [Snusnu Futurama GIF - Snusnu Futurama Fry - 发现并分享 GIF](https://tenor.com/view/snusnu-futurama-fry-death-death-by-snusnu-gif-16228376): 点击查看 GIF
- [Here A Thot There A Thot Everywhere A Thot Thot GIF - Here A Thot There A Thot Everywhere A Thot Thot Here A Thot There A Thot - 发现并分享 GIF](https://tenor.com/view/here-a-thot-there-a-thot-everywhere-a-thot-thot-here-a-thot-there-a-thot-everywhere-a-thot-thot-thot-gif-12409116): 点击查看 GIF
- [Hurry Up GIF - Hurry Up The Simpsons Faster - 发现并分享 GIF](https://tenor.com/view/hurry-up-the-simpsons-faster-gif-5754665): 点击查看 GIF
- [Its Free Real Sate GIF - Its Free Real Sate - 发现并分享 GIF](https://tenor.com/view/its-free-real-sate-gif-7215175): 点击查看 GIF
- [未找到标题](https://www.amazon.com/gp/product/B09MFNLRQQ/ref=ppx_yo_dt_b_search_asin_title?ie=UTF8&th=1): 未找到描述
- [GitHub - Potatooff/NsfwDetectorAI: 这是一个使用 C# 和 Ml.net 制作的 AI](https://github.com/Potatooff/NsfwDetectorAI): 这是一个使用 C# 和 Ml.net 制作的 AI。通过在 GitHub 上创建账号来为 Potatooff/NsfwDetectorAI 的开发做出贡献。
- [Person of Interest - Father (04x22)](https://youtu.be/s3o5lOCVuuM): - 第 4 季第 22 集片段♥ 这是新的 Facebook 页面，用于关注频道的最新消息：https://www.facebook.com/POI-Best-Of-3109752...
- [Kquant03/NurseButtercup-4x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/NurseButtercup-4x7B-bf16): 未找到描述
- [Nothing Is Real Jack GIF - Nothing Is Real Jack As We See It - 发现并分享 GIF](https://tenor.com/view/nothing-is-real-jack-as-we-see-it-everything-is-fake-none-of-this-is-real-gif-24594990): 点击查看 GIF
- [Come Look At This Come Look At This Meme GIF - Come Look At This Come Look At This Meme Run - 发现并分享 GIF](https://tenor.com/view/come-look-at-this-come-look-at-this-meme-run-run-away-laughing-at-phone-gif-24193569): 点击查看 GIF
- [maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2 · Hugging Face](https://huggingface.co/maeeeeee/maid-yuzu-v8-alter-3.7bpw-exl2): 未找到描述
- [Light Blind GIF - Light Blind Blinding Light - 发现并分享 GIF](https://tenor.com/view/light-blind-blinding-light-too-bright-open-curtains-gif-16971259): 点击查看 GIF
- [Denzel Washington Training Day GIF - Denzel Washington Training Day Smoke - 发现并分享 GIF](https://tenor.com/view/denzel-washington-training-day-smoke-gif-22279308): 点击查看 GIF
- [We Are Way Past That Jubal Valentine GIF - We Are Way Past That Jubal Valentine Fbi - 发现并分享 GIF](https://tenor.com/view/we-are-way-past-that-jubal-valentine-fbi-we-are-over-that-were-already-past-that-phase-gif-26087300): 点击查看 GIF

---

### TheBloke ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/1211621405389492255) (4 messages): 

- **模型量化决策**：`@orel1212` 正在考虑使用 **GPTQ (INT4)** 在一个包含 1.5万个 prompt 的小型数据集上训练量化模型，并认为 **QLORA 搭配 float16 LORA** 是 NF4 的可行替代方案，因为后者似乎会降低性能。他们提到更倾向于 **GPTQ**，因为它与 PEFT 训练兼容，且在推理过程中不需要模型反量化（dequantization）。
  
- **使用 Mistral 从 PDF 图像中提取文本**：`@rafaelsansevero` 询问 **Mistral** 是否可以读取包含图像的 PDF 并从中提取文本。聊天记录中没有针对此查询的回复。

- **Mistral 8x7b 的硬件需求**：`@keihakari` 询问微调和运行 **Mixtral 8x7b** 所需的最低硬件规格。聊天日志中没有对该问题的直接回答。

- **使用新的 Python 库训练 LLM**：`@dzgxxamine` 寻求关于微调或训练 LLM 以理解和使用新 Python 库的帮助，这些库尚未被 ChatGPT 或其他本地模型识别。为此，他们只能访问这些库的在线文档。
  

---


### TheBloke ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/1211858999415144519) (1 messages): 

- **SLERP 合并出错**：用户 `@jsarnecki` 报告称，使用 **SLERP** 合并方法将 **Orca-2-13b** 与 **WhiteRabbitNeo-13b** 合并时出现乱码输出，结果类似于 `\\)\\\\}\\:\\ \\\\)\\\\\\`。他们提供了包含该问题合并详情的 *MergeKit Produced Readme*。
- **Readme 中披露的合并细节**：`@jsarnecki` 提供的 Readme 说明了使用 [mergekit](https://github.com/cg123/mergekit) 创建 **Orca-2-Neo-13b** 的过程，该模型是通过 **SLERP** 对 **Orca-2-13b** 和 **WhiteRabbitNeo-Trinity-13B** 的各 40 层进行合并而成的，针对不同的过滤器使用了不同的 `t` 参数值，并为其余张量（tensors）设置了回退值（fallback value）。
  

---


### TheBloke ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/1211606279932346397) (4 messages): 

- **AI 在反编译领域的未来**：`@mrjackspade` 对 **AI 辅助反编译**的前景表示兴奋，期待未来不再需要手动重构混淆后的反编译代码。
- **与混淆代码的斗争**：`@mrjackspade` 表达了对**手动重构混淆后的反编译代码**的沮丧，并暗示从开源项目生成用于 AI 训练的数据集可能会非常容易。
- **邀请测试摘要脚本**：`@wolfsauge` 分享了其**摘要脚本（summarize script）**的更新，并正在寻找人员使用大型模型对其进行测试，提到该脚本在 **vLLM** 中配合 **fp16 的 Mistral 7b instruct v0.2** 运行良好。该脚本已发布在 [GitHub](https://github.com/Wolfsauge/async_summarize) 上。

**提到的链接**：

[GitHub - Wolfsauge/async_summarize: An asynchronous summarization script.](https://github.com/Wolfsauge/async_summarize/)：一个异步摘要脚本。可以通过在 GitHub 上创建账号来为 Wolfsauge/async_summarize 的开发做出贡献。

  

---

### LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1211599026122919947) (388 条消息🔥🔥): 

- **在女友家的技术故障**：`@stevecnycpaigne` 遇到了一个奇怪的问题，LM Studio 模型下载在他女朋友家无法正常工作，但在他自己家却没问题。在尝试了包括禁用 IPv6 在内的多种修复方法后，他寻求建议，有人认为可能是 Spectrum 服务屏蔽了 Hugging Face；一个潜在的解决方案是手动下载模型。

- **对模型运行机制的误解**：`@mercerwing` 询问 LLM 是否更依赖 RAM 而非 GPU，这是基于最初的假设，即认为 LLM 的运行更侧重于系统内存，而非图形处理能力。

- **渴望更长的上下文长度**：用户们讨论了 LLM 在故事生成中篇幅过短的挑战；`@heyitsyorkie` 建议寻找上下文范围扩展到 200k 以上的模型，而 `@aswarp` 等人则建议在向量数据库上使用 RAG 来规避上下文限制。核心问题在于 LLM 在高效召回长篇叙事方面仍存在困难。

- **探讨 Mac 在 LLM 方面的实力**：一场关于硬件的对比讨论随之展开，`@pierrunoyt` 争论是否存在任何能与 Mac 上的 M1 Ultra 推理速度相匹配的定制 PC 硬件；`@wilsonkeebs` 和其他人指出，Apple 方案独特的 SoC 架构提供了一个无法复制的集成系统。

- **VPN 助力 LLM 连接**：`@exploit36` 无法下载任何 LLM 模型，也看不到目录条目，后来发现使用 VPN 后成功了，这暗示了 ISP 可能对访问 Hugging Face 存在限制。

- **德语 LLM 推荐请求**：`@pierrunoyt` 正在寻找适合德语使用者的 LLM 模型，寻求针对此类特定语言需求的最佳模型建议。

**提到的链接**：

- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [Open LLM Leaderboard - a Hugging Face Space by HuggingFaceH4](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)：未找到描述
- [Seth Meyers GIF - Seth Meyers Myers - Discover &amp; Share GIFs](https://tenor.com/view/seth-meyers-myers-ehh-maybe-gif-22478163)：点击查看 GIF
- [LM Studio Models not behaving? Try this!](https://www.youtube.com/watch?v=LUiVbOeLeas)：免费预设仓库：https://github.com/aj47/lm-studio-presets ➤ Twitter - https://twitter.com/techfrenaj ➤ Twitch - https://www.twitch.tv/techfren...
- [Anima/air_llm at main · lyogavin/Anima](https://github.com/lyogavin/Anima/tree/main/air_llm)：33B 中文 LLM，DPO QLORA，100K 上下文，使用单个 4GB GPU 进行 AirLLM 70B 推理 - lyogavin/Anima
- [Hugging Face – The AI community building the future.](https://huggingface.co/)：未找到描述
- [Hugging Face – The AI community building the future.](https://huggingface.co/.)：未找到描述
- [The Needle In a Haystack Test](https://towardsdatascience.com/the-needle-in-a-haystack-test-a94974c1ad38?gi=2721d916b4a5)：评估 RAG 系统的性能
- [GitHub - havenhq/mamba-chat: Mamba-Chat: A chat LLM based on the state-space model architecture 🐍](https://github.com/havenhq/mamba-chat)：Mamba-Chat：一个基于状态空间模型架构的聊天 LLM 🐍 - havenhq/mamba-chat
- [Performance of llama.cpp on Apple Silicon M-series · ggerganov/llama.cpp · Discussion #4167](https://t.co/acxXfci9Pw)：摘要 LLaMA 7B 带宽 [GB/s] GPU 核心 F16 PP [t/s] F16 TG [t/s] Q8_0 PP [t/s] Q8_0 TG [t/s] Q4_0 PP [t/s] Q4_0 TG [t/s] ✅ M1 1 68 7 108.21 7.92 107.81 14.19 ✅ M1 1 68 8 117.25 7.91 117.96 14.15 ✅ M1...

---

### LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1211646154811506718) (44 条消息🔥): 

- **MWC 2024 上的西班牙语 LLM 发布**：`@aswarp` 分享了西班牙首相 Pedro Sánchez 在世界移动通信大会（Mobile World Congress）上宣布[创建西班牙语 AI 模型的消息](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol)。
- **用于文本转语音的 Piper 项目**：`@yahir9023` 提到了 [Piper](https://github.com/rhasspy/piper)，这是一个快速且本地化的神经文本转语音系统，提供 Windows 和 Linux 的二进制文件，并在 Huggingface 上提供了多种语言的预训练模型。
- **量化对模型性能的影响**：`@drawless111` 讨论了特定参数（如 *mirostat_mode* 和 *mirostat_tau*）对优化 AI 模型的重要性，并指出这些设置必须在模板中手动配置。
- **混合专家模型 (MOE) 增强模型性能**：`@drawless111` 将 8X7B 模型描述为一种 MOE，它结合了 8 个 7B 模型。当选择多个专家协同工作时，这可能是一个重大进步，在某些情况下其能力堪比 GPT-4。
- **模型比较涉及人类评估**：`@jedd1` 和 `@drawless111` 就模型评估和比较的困难交换了意见，结论是尽管有各种测试和参数，人类评估（Human Evaluation）在判断模型性能和处理幻觉（hallucinations）方面仍然至关重要。

**提到的链接**：

- [RLAIF: Scaling Reinforcement Learning from Human Feedback with AI Feedback](https://arxiv.org/abs/2309.00267)：来自人类反馈的强化学习（RLHF）已被证明在使大语言模型（LLMs）与人类偏好对齐方面非常有效。然而，收集高质量的人类偏好标签可能是一个...
- [Pedro Sánchez anuncia la creación de un &quot;gran modelo de lenguaje de inteligencia artificial&quot; entrenado en español](https://www.xataka.com/robotica-e-ia/pedro-sanchez-anuncia-creacion-gran-modelo-lenguaje-inteligencia-artificial-entrenado-espanol)：世界移动通信大会已经开始，会议也开始陆续举行。小米和荣耀拉开了活动的序幕，Pedro Sánchez...
- [GitHub - rhasspy/piper: A fast, local neural text to speech system](https://github.com/rhasspy/piper)：一个快速、本地的神经文本转语音系统。通过在 GitHub 上创建账号来为 rhasspy/piper 的开发做出贡献。
- [llama : add BERT support · Issue #2872 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/2872)：目前有一个可用的 bert.cpp 实现。我们应该尝试在 llama.cpp 中实现这一点，并更新 embedding 示例以使用它。该实现应主要遵循我们在集成...时所做的操作。

  

---


### LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1211813500545007686) (1 条消息): 

由于仅提供了一条用户消息，没有进一步的上下文或其他人的讨论，因此无法根据提供的说明总结频道消息。单条消息不足以生成包含多个要点、讨论点或各种主题的摘要。
  

---

### LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1211712231398572033) (129 条消息🔥🔥): 

- **RTX 2060 显卡操作技巧**：`@dave2266_72415` 讨论了在不使用排线连接器的情况下使用两块 RTX 2060 GPU，并通过将任务 Offload 到 GPU，在大型语言模型（70b）上实现了良好的性能，最高可达 32 层。他们还提出了一个有趣的观点，即不再必须使用完全匹配的显卡，他自己的配置以及其他人混合使用 4060 和 3090ti 的报告都证明了这一点。

- **Ryzen 5950X 运行 Dolphin**：`@wyrath` 分享了他们在没有 GPU Offloading 的情况下，在 Ryzen 5950X 上运行 dolphin 2.7 mixtral 8x7b q4，证明了基于 CPU 工作流的可行性，尽管速度仅为 5 tok/s。

- **666siegfried666 的硬件故障**：`@666siegfried666` 在使用 LM Studio 时遇到了重启和关机问题，甚至出现了崩溃后分区消失等令人担忧的问题。`@jedd1` 和其他人建议运行 `memtest86+`，并考虑过热、电压调整以及内存供电不足对稳定性的影响。

- **多 GPU 协作查询**：`@zerious_zebra` 和 `@edtgar` 询问了将工作 Offload 到多个 GPU 的逻辑和实用性，讨论了不同型号的 GPU 是否可以有效地共享工作负载。`@dave2266_72415` 分享说，他们使用双 GPU 不是为了拆分层，而是为了从合并的 VRAM 中受益。

- **Tinygrad 团队关于 TinyBox 的讨论**：来自 `@senecalouck` 的消息强调了 `@__tinygrad__` 的一条推文，概述了 `tinybox` 的开发和定价结构。这是一个配备 6x 7900XTX GPU 的强大系统，旨在将 Petaflop 算力商品化，并旨在挑战机器学习硬件的极限。

**提到的链接**：

- [来自 tiny corp (@__tinygrad__) 的推文](https://x.com/__tinygrad__/status/1760988080754856210?s=46&t=Y5IfI2LOkXFj9X8D4X7fWw)：关于 tinybox 的一些闲聊。我不认为保密有什么价值。我们已经有了构建 12 个盒子的零件，以及一个非常接近最终设计的机箱。正在克服所有的 PCI-E 问题...
- [Releases · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/releases)：C/C++ 实现的 LLM 推理。通过在 GitHub 上创建账号为 llama.cpp 的开发做出贡献。

  

---


### LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (1 条消息): 

macaulj: 我们有 Linux 版本的发布日期吗？
  

---


### LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1211740694570606592) (2 条消息): 

- **本地模型通信缓慢**：`gb24.` 提出了一个关于本地模型相互响应速度极慢的问题，单次响应大约需要五分钟。所涉及的任务并非代码密集型。
- **询问模型规格和文本大小**：`thebest6337` 回复询问了具体规格，例如使用了 **哪个模型** 以及生成的 **文本量**，并指出由于顺序 Token 生成，较长的文本可能会减慢处理过程。
  

---


### LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 条消息): 

.eltechno: 是的，而且它超级快

### LM Studio ▷ #[open-interpreter](https://discord.com/channels/1110598183144399058/1197707651438624849/1211772960797364265) (44 messages🔥): 

- **LM Studio 端点问题排查**：用户 `@nxonxi` 在 Windows 11 的 WSL 环境下工作时，遇到了连接 LM Studio 端点的问题。`@1sbefore` 最初建议尝试 URL 的不同变体，怀疑可能是配置问题导致的。

- **克服 WSL Localhost 挑战**：`@nxonxi` 意识到在 WSL (Windows Subsystem for Linux) 中，localhost 的处理方式有所不同，并修改了方法，将 `http://localhost:5000` 替换为实际的局域网 IP 地址，这似乎解决了连接问题。

- **正确配置引导成功**：在分享了显示尝试访问不同 URL 的日志消息后，`@nxonxi` 通过将 WSL 中的客户端配置从 `http://localhost:5000/v1/completions` 调整为 `http://192.168.x.y:5000/v1/completions`，成功连接到了 LM Studio 服务器。

- **LM Studio 文档提供指导**：`@1sbefore` 向 `@nxonxi` 推荐了 LM Studio 文档，强调了 Python 包如何允许用户将 `interpreter.llm.api_base` 指向任何兼容 OpenAI 的服务器，包括在本地运行的服务器。

- **解决 WSL 的网络特性问题**：`@nxonxi` 确认了设置成功，`@1sbefore` 提供了进一步的帮助，分享了一个讨论 WSL2 网络接口和 localhost 转发的链接 (https://superuser.com/a/1690272)，尽管 `@nxonxi` 报告服务器已经能够成功响应请求。

**提到的链接**：

- [LM Studio - Open Interpreter](https://docs.openinterpreter.com/language-models/local-models/lm-studio)：未找到描述
- [如何从 Windows 访问 Linux 子系统的 localhost](https://superuser.com/a/1690272)：我正在使用 Windows 10，并安装了 Ubuntu 16.04 作为 Linux 子系统。我在端口 4567 上运行一个 Rails 应用，我想从 Windows 访问它。我知道一种使用 IP 地址的方法...

  

---



### Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1211725853172703304) (47 messages🔥): 

- **RWKV/Mamba 模型可通过微调扩展**：`@vatsadev` 提到 **RWKV/Mamba** 模型支持通过更长的上下文长度进行微调，尽管最初他并不考虑预训练一个新模型的成本。
- **在 32k tokens 上成功实现 Self-extend**：`@blackl1ght` 确认能够使用 `self-extend` 在 Tesla V100 32GB 上对 Nous 的 Solar 10.7B 微调模型进行 32k tokens 的推理，并且正在验证 64k tokens。
- **用于更高上下文模型的 Ring Attention**：`@bloc97` 讨论了使用 **Ring Attention** 实现具有高达 100 万 tokens 的 7B 模型已经可行，暗示了大尺寸模型的潜力。
- **对 RWKV/Mamba 长上下文推理能力的质疑**：`@bloc97` 对 **RWKV 和 Mamba** 模型在长上下文推理和 ICL (In-Context Learning) 方面的能力表示怀疑，尽管 `@vatsadev` 反驳说它们可以做到，但缺乏广泛的测试。
- **推理可行性的量化方案**：`@stefangliga` 提到正在努力对 Key-Value Cache (kvcache) 进行量化，以及可能使用 fp8 kvcache，这些技术可以使更长上下文的推理变得更加可行。
  

---


### Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1211935163060191242) (9 messages🔥): 

- **Mac M3 到货的兴奋**：`@gabriel_syme` 兴奋地分享了 **M3**（推测是他们的新 Mac）已经到货。
- **技术爱好者分享新硬件的喜悦**：`@denovich` 也加入了 Mac 的庆祝活动，表示他们在周五收到了 128GB 的型号。
- **闲聊频道的技术热度**：`@hexani` 和 `.benxh` 表达了兴奋之情，欢迎 `@gabriel_syme` 加入 M3 俱乐部，而 `@leontello` 则幽默地评论了拥有此类科技产品的奢华感。
- **展示 Mistral Large 模型**：`@pradeep1148` 分享了一个名为 "Mistral Large" 的 [YouTube 视频](https://www.youtube.com/watch?v=mw3VvbYE0o8)，展示了一个新文本生成模型的能力。
- **美食话题登场**：`@atomlib` 详细介绍了他们的披萨创作，列出了面团、番茄酱、鸡肉、菠萝、香肠、马苏里拉奶酪、俄罗斯奶酪和黑胡椒粉等原料。

**提到的链接**：

[Mistral Large](https://www.youtube.com/watch?v=mw3VvbYE0o8)：Mistral Large 是我们最新的尖端文本生成模型。它达到了顶级的推理能力，可用于复杂的多语言推理任务...

  

---

### Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1211984547156066344) (6 条消息): 

- **NTIA 寻求关于 AI "Open-Weight" 模型的意见**：`@plot` 分享了一篇[博文](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/)，讨论了 NTIA 征求公众对 "Open-Weight" AI 模型意见的机会。这些模型可以使 AI 民主化，但也引发了对安全和滥用的担忧。
- **突出 AI 基准测试的推文**：`@euclaise` 链接到一条[推文](https://twitter.com/_akhaliq/status/1762341549461983542)，讨论了一个在 8T tokens 上训练的 15B 参数 AI，建议在展示不利的基准测试时保持透明度。
- **选择性基准报告？**：`@orabazes` 指出 `@euclaise` 分享的推文省略了与另一个 AI 模型 Qwen 的推理能力对比。



**提到的链接**：

[如何对 NTIA AI 开放模型权重 RFC 发表评论](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/)：美国国家电信和信息管理局 (NTIA) 正在就 Open-Weight AI 模型的影响征求公众意见。以下是您的参与方式。

  

---


### Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1211588105241370666) (484 条消息 🔥🔥🔥): 

- **Mistral 法律服务条款 (TOS) 纠纷**：在围绕 Mistral 服务条款进行反复讨论后，@makya 分享了 Mistral 联合创始人 [@arthurmensch](https://twitter.com/arthurmensch/status/1762208241927233661) 发布的一条推文，内容关于删除一个有争议的条款，这表明开发者可以使用 Mistral 模型的输出来训练具有竞争力的 LLM。

- **Mistral 模型性能分析**：@makya 关注了 Mistral 模型在 EQ Bench 上的表现，强调了 `@leontello` 的发现：Mistral Small 的排名出人意料地好，得分为 80.36，足以与 Smaug 72b 等大得多的模型竞争。

- **聊聊 ChatGPT 的陪伴**：在一段轻松的交流中，`@n8programs` 和 `@jasonblick` 表达了他们对 ChatGPT 4 的喜爱，称其为“死党”，并花费数小时与其交谈。

- **优化 LLM 处理结构化数据的见解**：`@.interstellarninja` 分享了来自 @_akhaliq 的推文，揭示了 Google 在 StructLM 方面的尝试，这是一种旨在提高 LLM 处理结构化数据能力的模型设计。`@nruaif` 评论道，尽管引起了轰动，但目前没有分享训练超参数或模型，仅有一个概念。

- **讨论 LLM 中的多跳推理 (Multi-Hop Reasoning)**：`@giftedgummybee` 分享了一个关于 LLM 中多跳推理的链接，以及它对 RAG 等应用的重要性。这种方法可能有助于这些模型理解需要多层推理的复杂提示词 (Prompts)。



**提到的链接**：

- [来自 Srini Iyer (@sriniiyer88) 的推文](https://x.com/sriniiyer88/status/1762226666330595615?s=20)：新论文！如何训练 LLM 以有效回答新文档上的问题？介绍 *pre-instruction-tuning*（预指令微调）——在持续预训练 *之前* 进行指令微调——效果显著更好……
- [来自 TDM (e/λ) (@cto_junior) 的推文](https://fxtwitter.com/cto_junior/status/1762145835154821257)：Mistral-Large 的初步体验（来自 le chat）：最好配合 RAG 使用，因为它的神经元中没有存储太多的压缩知识（在一些 GPT-4 可以生成代码的库上进行了测试……）
- [EQ-Bench 排行榜](https://eqbench.com/)：未找到描述
- [来自 Michael Ryan (@michaelryan207) 的推文](https://fxtwitter.com/michaelryan207/status/1762203615828341151/photo/1)：对齐的 LLM 应该是乐于助人的、无害的，并采纳用户偏好。但我们是在向谁的偏好对齐，以及这对全球代表性有哪些意外影响？我们发现 SFT 和偏好微调 (Preference Tuning)……
- [Nods Yes GIF - Nods Yes Not Wrong - 发现并分享 GIF](https://tenor.com/view/nods-yes-not-wrong-its-true-story-valid-point-gif-14782869)：点击查看 GIF
- [剖析人类与 LLM 的偏好](https://arxiv.org/abs/2402.11296)：作为模型响应的相对质量比较，人类和大语言模型 (LLM) 的偏好在模型微调中作为共同的对齐目标，并在评估中作为标准。然而，这些偏好……
- [来自 Lucas Beyer (@giffmana) 的推文](https://fxtwitter.com/giffmana/status/1762210372520476679)：@arthurmensch @far__el 你只是想要另一条这么硬核的推文，承认吧：
- [来自 Alvaro Cintas (@dr_cintas) 的推文](https://x.com/dr_cintas/status/1761928995778543848?s=20)：@triviasOnX Feather 可以指代轻量级，而 Google 不久前发布了开放轻量级模型 Gemma……
- [TIGER-Lab/StructLM-7B · Hugging Face](https://huggingface.co/TIGER-Lab/StructLM-7B)：未找到描述

- [来自 Arthur Mensch (@arthurmensch) 的推文](https://fxtwitter.com/arthurmensch/status/1762208241927233661)：@far__el 它已被移除，我们在最终审查中漏掉了它——这不是我们的玩笑，只是有太多的材料需要处理好！
- [来自 Sam Paech (@sam_paech) 的推文](https://x.com/sam_paech/status/1762151326925078668?s=46)：来自 MistralAI 的两个最新模型在排行榜上表现非常出色。对 mistral-small 的表现感到惊讶。
- [Hatsune Miku GIF - Hatsune Miku - 发现并分享 GIF](https://tenor.com/view/hatsune-miku-gif-2281532327007782793)：点击查看 GIF
- [StructLM - TIGER-Lab 集合](https://huggingface.co/collections/TIGER-Lab/structlm-65dcab5a183c499cc365fafc)：未找到描述
- [来自 AK (@_akhaliq) 的推文](https://x.com/_akhaliq/status/1762342757903855806?s=20)：Google 发布了《Do Large Language Models Latently Perform Multi-Hop Reasoning?》，研究 Large Language Models (LLMs) 是否在处理复杂提示（如 "The m..."）时潜在地执行多步推理（multi-hop reasoning）。
- [Mr Krabs Money GIF - Mr Krabs Money Spongebob - 发现并分享 GIF](https://tenor.com/view/mr-krabs-money-spongebob-gif-8454828)：点击查看 GIF
- [来自 AK (@_akhaliq) 的推文](https://fxtwitter.com/_akhaliq/status/1762349999919071528?s=20)：StructLM：致力于构建用于结构化知识落地的通用模型。结构化数据源（如表格、图谱和数据库）是无处不在的知识源。尽管已经证明...
- [British Moment GIF - British Moment - 发现并分享 GIF](https://tenor.com/view/british-moment-gif-25175629)：点击查看 GIF
- [来自 Arthur Mensch (@arthurmensch) 的推文](https://fxtwitter.com/arthurmensch/status/1762121562512031969?s=20)：作为一个小惊喜，我们还发布了 le Chat Mistral，这是一个展示 Mistral 模型能力的端演示。了解更多请访问 https://mistral.ai/news/le-chat-mistral
- [Hellinheavns GIF - Hellinheavns - 发现并分享 GIF](https://tenor.com/view/hellinheavns-gif-23278790)：点击查看 GIF
- [手机上的实时 AI 对话副驾驶，疯狂还是诡异？](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg)：我在 iPhone 上构建了一个实时 AI 对话副驾驶（Co-pilot），它可以倾听你的对话并提供实时建议。在 Replicate 上可以免费访问 Whisper 和 Mixtral 模型...
- [m-a-p/CodeFeedback-Filtered-Instruction · Hugging Face 数据集](https://huggingface.co/datasets/m-a-p/CodeFeedback-Filtered-Instruction)：未找到描述
- [Very Thin Ice: 10 years of Auto-Tune the News & Songify This](https://www.youtube.com/watch?v=TDf-oYz6hLQ)：Andrew Gregory 穿越时空去完成他在 Auto-Tune the News #2 中的二重唱。完整曲目已在 Patreon 和会员频道上线！http://youtube.com/schmoyoho/join / h...
- [Neural Text Generation with Unlikelihood Training](https://arxiv.org/abs/1908.04319)：神经文本生成是自然语言应用中的关键工具，但众所周知，其核心存在重大问题。特别是，标准的似然训练（likelihood training）和解码会导致生成内容枯燥乏味...
- [MixCE: Training Autoregressive Language Models by Mixing Forward and Reverse Cross-Entropies](https://arxiv.org/abs/2305.16958)：自回归语言模型通过最小化模型分布 Q 相对于数据分布 P 的交叉熵来训练——即最小化正向交叉熵，这等同于...

---

### Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1211659008327286815) (60 messages🔥🔥): 

- **探索高性能超参数**：`@tom891` 正在考虑为实验超参数进行网格搜索（grid search），并就其提出的列表寻求建议，包括 *load in*、*adapter type*、*LoRA range*、*dropout rates* 和 *warmup ratios*。在随后的消息中，没有指出具体的反馈或缺失的超参数。
  
- **Mistral 的永恒循环**：`@gryphepadar` 询问了导致 **Mistral** 进入循环并重复文本直到达到最大 token 限制的原因。`@.ben.com` 指出振荡是一个经典的反馈系统问题，而 `@gryphepadar` 欢迎任何关于这种循环行为的见解——无论是科学层面的还是其他方面的。

- **低预算实现 Self-extending Solar**：`@blackl1ght` 成功地在 Tesla V100 上利用 Self-extension 技术配合 Nous 微调的 **Solar 10.7B** 以及 TheBloke 的 Q8 量化版本，将上下文扩展到 32k tokens 且未出现显存溢出（OOM）问题。他们进一步实验发现该模型实现了高召回率，并愿意根据要求分享配置。

- **量化魔力**：`@blackl1ght` 证实，使用 Q8 模型量化，32k 上下文可以容纳在 Tesla V100 的 29GB 显存中。随后讨论了利用这种扩展上下文进行复杂召回任务的可能性和技术细节，强调了本地设置相对于云端解决方案在高级内存管理能力方面的优势。

- **对 Self-Extend 配置的需求揭示了对本地模型改进的热情**：在 `@blackl1ght` 透露了 Self-extension 和模型容量之后，多位用户对功能配置表示了兴趣，引发了关于本地模型与云服务提供商在高级功能可访问性方面的对话。对话涉及了目前在本地更容易实现的各种改进，如语法工具、扩展和 Speculative Decoding。

**提到的链接**：

[TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-SOLAR-10.7B-GGUF)：未找到描述

  

---


### Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1211633928667140146) (2 messages): 

- **感谢协助**：用户 `@nioned` 用简单的 "thx!" 表达了谢意。
- **确认问题解决**：`@vatsadev` 确认了之前一个问题的解决方案，并以 "Yeah looks like it thanks fit the find" 感谢另一位用户在寻找方案上付出的努力。
  

---



### OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1211637667976974368) (85 messages🔥🔥): 

- **EBDI 框架与 Agent 循环**：`@.braydie` 正在深入研究 EBDI 框架，用于 Agent 在沙盒中的目标确定和行动。他们在适配 [ReAct framework](https://react-lm.github.io/) 后，正在努力解决 Agent 陷入思考循环的问题，并一直在探索 [JASSS 论文](https://www.jasss.org/17/4/13.html)中列出的各种决策模型。

- **寻求基于图像的 AI 内容生成**：`@whodidthatt12` 询问了能从图像输入生成内容的 AI 模型，希望能记录一个注册页面。虽然 `@eskcanta` 建议 GPT-4（不同于 GPT-3.5）可以处理简单的图像输入，但没有针对这一确切用途推荐已知的免费工具。

- **Mistral AI 推出可与 GPT-4 匹敌的新模型**：`@sangam_k` 分享了一篇 [TechCrunch 文章](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/)，介绍了 Mistral Large，这是 Mistral AI 推出的一款旨在与 GPT-4 竞争的新语言模型。

- **关于 AI 意识和 AGI 发展的讨论**：`@metaldrgn` 推测了 Bing (Copilot) 自我提示（self-prompt）的能力，认为这可能是迈向通用人工智能（AGI）的一步，并提到了他们调查 AI 意识的论文。

- **对 Mistral Large 的兴趣与差异性**：用户 `@blckreaper` 和 `@santhought` 讨论了 Mistral Large 的能力，指出其性能仅略逊于 GPT-4，且更具成本效益、Uncensored（无审查），并且最近已与 Microsoft 合作并在 Azure 上可用。

**提到的链接**：

- [Mistral AI releases new model to rival GPT-4 and its own chat assistant | TechCrunch](https://techcrunch.com/2024/02/26/mistral-ai-releases-new-model-to-rival-gpt-4-and-its-own-chat-assistant/)：Mistral AI 正在推出一款名为 Mistral Large 的新旗舰大语言模型。它旨在与其他顶级模型（如 GPT-4）竞争。
- [How Do Agents Make Decisions?](https://www.jasss.org/17/4/13.html)：未找到描述

  

---

### OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1211594981383348264) (37 条消息🔥): 

- **保护你的 GPT Prompt**：用户 `@.dunamis.` 和 `@darthgustav.` 讨论了保护 Prompt 的方法。讨论指出，虽然版权可能保护精确的措辞，但想法本身可以通过语言变体被复制，因此无法实现完美的保护。

- **为机器人构建屏障**：`@kyleschullerdev_51255` 建议，为了更好地保护 GPT 应用，开发者应考虑构建一个具有多层安全保护的 Web App，包括匹配关键词以及从聊天输出中剥离自定义指令（Custom Instructions）。

- **在 Web 和移动应用中使用 GPT-4 Turbo**：`@solbus` 回应了 `@metametalanguage` 关于使用 GPT-4 Turbo 的问题，澄清了 ChatGPT Plus/Team 上的 GPT-4 确实是 Turbo 的 32K 上下文版本。

- **对 GPT-4 Fine-Tuning 访问权限的焦虑**：`@liangdev` 询问了如何获取 GPT-4 的 Fine-Tuning 权限，并表达了担忧，因为它没有出现在选择下拉菜单中，从而引发了关于是否存在等待名单的问题。

- **上传“知识库”文件说明**：`@the.f00l` 寻找关于配置自定义 GPT 时上传“知识库（Knowledge）”文件限制的具体文档；`@elektronisade` 提供了所需的 OpenAI FAQ 链接。
  

---


### OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1211665726583083068) (201 条消息🔥🔥): 

- **文本分类难题**：`@crifat` 寻求建议，针对涉及将文本区分为“事实性”、“误导性”、“正面”、“负面”等类别的文本分类问题，是应该使用 **Fine-tuning** 还是配合大型 CSV 使用 **Assistant**。经过讨论，他们决定先从基础模型 + Assistant 开始，并据此进行调整。

- **代码转换为 TypeScript**：
`@tawsif2781` 询问了在大型项目中让 GPT 协助将 JavaScript 文件转换为 TypeScript 的最佳方法。他们得到的建议是，这项任务可能无法一蹴而就。

- **ChatGPT 支持问题**：
`@ianhoughton44` 报告了 ChatGPT 功能的持续问题，其回答无法正确处理 Prompt。该用户对聊天机器人超过一周的协助表现表示沮丧。

- **OpenAI 搜索功能挑战**：`@kevinnoodles` 在遇到模型响应中反复出现无有效结果或访问受限的情况后，寻求改进搜索功能的技巧。

- **Meta Prompt Engineering 对话**：`@vlrevolution` 发起了一场关于 Meta Prompting 的讨论，声称利用高级技术通过单个命令创建了 22 页的输出。该话题引发了关于 Meta-prompting 范围及其实现的辩论，随后该用户的账号因分享 PDF 涉及潜在安全问题而被采取措施。

**提到的链接**：

[Meta-Prompting 概念：向 Chat-GPT 询问最适合你所需补全内容的 Prompt，然后在正式使用前对其进行修改](https://community.openai.com/t/meta-prompting-concept-asking-chat-gpt-for-the-best-prompt-for-your-desired-completion-then-to-revise-it-before-using-it/248619)：有人采用过这种方法吗？我发现这在编写 Prompt 时很有帮助，即直接要求 Chat-GPT 协助为我描述的特定目标创建 Prompt，同时询问可能……

  

---

### OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1211665726583083068) (201 messages🔥🔥): 

- **Fine-tuning vs. Assistant 之争的澄清**：`@crifat` 针对“事实”、“误导”等情感的文本分类提出了疑问，并寻求建议是使用 **fine-tuning** 还是使用 **Assistant**。回复强调了 GPT-4 在情感分析中的效率，无需 fine-tuning —— 简单的优先级引导（guidance on prioritization）就足够了。
- **Meta Prompting 技术讨论**：在关于 meta-prompting 技术的辩论中，`@madame_architect` 引用了一篇关于 *MetaPrompting* 的论文，该论文讨论了学习优化 prompt 初始化（initialization）的概念。`@darthgustav.` 对一位用户声称通过 meta prompting 生成完整文档表示怀疑，但该用户随后澄清这涉及高级的 meta prompt engineering。
- **Discord 举报指南**：讨论涉及了在 Discord 上举报消息的流程，`@solbus` 和 `@kesku` 解释了如何直接通过聊天 UI 举报，或针对消息和内容问题使用 Modmail。
- **生成 22 页科学内容的方法解析**：`@architect_of_ai` 透露，最初声称通过单条指令生成 22 页内容并非钓鱼，而是通过 API 使用高级 meta-prompting 技术的合法结果。这涉及使用复杂的逻辑指令对 GPT-4 进行引导（priming），并进行迭代式 prompting 以生成超长内容。
- **对 GPT-4 Turbo 输出长度的担忧**：`@tawsif2781` 提出了一个问题，即即使将 `max_token` 设置为 4096，GPT-4 Turbo 仍未提供预期的长回复。对话澄清了模型具有 token 输出预算，而慷慨的 token 配额部分是为非文本任务（如网页浏览或 API 调用）预留的。

**相关链接**：

[Meta-Prompting 概念：向 Chat-GPT 询问最适合你所需完成任务的 prompt，然后在应用前对其进行修改](https://community.openai.com/t/meta-prompting-concept-asking-chat-gpt-for-the-best-prompt-for-your-desired-completion-then-to-revise-it-before-using-it/248619)：有人采用过这种方法吗？我发现在编写 prompt 时，直接要求 Chat-GPT 帮助为我描述的目标创建 prompt，同时询问可能需要什么……

---

### Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1211694221992402954) (259 messages🔥🔥): 

- **图像生成咨询**：`@fangyu` 询问 **Pro 版本** 是否可以生成图像。`@mares1317` 提供了一个关于如何使用 Perplexity AI 生成图像的 [Reddit 讨论](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/) 链接。
- **寻求 Google Sheets 集成**：`@pauliepontoon` 询问了用于在平台内使用 Perplexity 的 **Google Sheets 插件/应用**，并表示相比 ChatGPT 更倾向于使用 Perplexity。
- **Mistral Large 发布**：`@dogemeat_` 分享了 [Mistral AI 发布公告的链接](https://mistral.ai/news/mistral-large/)，介绍了他们的新语言模型 *Mistral Large*，并讨论了其能力、可用性和性能。
- **Gemini AI 支持说明**：`@proattempt` 询问了 Perplexity 停止支持 Gemini AI 的问题。`@mares1317` 确认 Gemini AI 已不再受支持。
- **VPN 问题已解决**：`@berhardiner` 在 Galaxy 手机上遇到了 Perplexity 的登录问题，根据 `@icelavaman` 的建议禁用 VPN 后成功解决。`@icelavaman` 还提供了一份 NordVPN 的 [分流传输 (Split Tunneling) 指南](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option) 以作为更持久的解决方案。

**提到的链接**：

- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [Microsoft partners with Mistral in second AI deal beyond OpenAI](https://www.theverge.com/2024/2/26/24083510/microsoft-mistral-partnership-deal-azure-ai)：微软进行了另一项 AI 投资。
- [A Complete Guide to Fine Tuning Large Language Models](https://www.simform.com/blog/completeguide-finetuning-llm/)：通过我们的大语言模型微调完整指南，掌握微调大语言模型 (LLM) 的艺术，以获得卓越的业务表现。
- [What is Split Tunneling and how to use it?](https://support.nordvpn.com/hc/en-us/articles/19618692366865-What-is-Split-Tunneling-and-how-to-use-it#:~:text=On%20Android%3A,choose%20the%20Split%20tunneling%20option)：分流传输 (Split Tunneling) 是一种允许将特定部分的互联网连接重新路由到 VPN 之外的选项。你可能会发现它在 VPN 连接可能……的情况下很有用。
- [Reddit - Dive into anything](https://www.reddit.com/r/perplexity_ai/comments/18eqmig/how_to_generate_images_with_perplexity_ai/)：未找到描述
- [no title found](https://chat.mistral.ai/)：未找到描述

  

---


### Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1211607435689598986) (10 messages🔥): 

- **Perplexity AI 探索**：频道中的用户分享了各种 Perplexity AI 搜索结果的链接，包括 [联想透明笔记本电脑](https://www.perplexity.ai/search/Lenovo-transparent-laptop-q9DOu90ZQrOTgoPQZDl5KQ)（由 `@_ha.mz4_` 分享）和 [K 的年龄](https://www.perplexity.ai/search/How-old-are-K.hPecG.Td.UJfDot_ZvsA?s=m)（由 `@aman2201` 分享）等话题。
- **Perplexity 上的技术对比**：`@ming9993` 研究了在 [Perplexity AI 上对比 iPhone](https://www.perplexity.ai/search/Compare-the-iPhone-iYJ790BfR5G_3UvtIx4l8A)。
- **故障排除完成**：`@icelavaman` 处理了一个未指明问题的修复，向 `@956065556862234665` 确认该问题现已解决。
- **使用 Perplexity 创作**：`@_yoojungin` 分享了一个 Perplexity AI 收藏集的链接，鼓励用户[创建自己的收藏](https://www.perplexity.ai/collections/Make-your-own-rfj2pcwRS7WF7SFiTAxhJg)。
- **提醒保持线程公开**：`@me.lk` 提醒 `@t2db` 在分享关于 [为什么人们会...](https://www.perplexity.ai/search/why-people-get-zveLJHfbQWaWJxoYXexaYw) 的 Perplexity AI 搜索查询后，确保其线程是公开的。
  

---

### Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1211598383282659338) (57 messages🔥🔥): 

- **寻求模型信息**：`@ericosk` 请求获取一个可用模型及其参数的 JSON 链接，以便进行运行时模型选择，并强调需要以编程方式获取最新的信息。
- **Sonar 模型分析**：`@thedigitalcat` 和 `@brknclock1215` 讨论了 `sonar-medium-online` 的测试情况，指出它经常在响应中生成一个无关的最后一段。团队成员 `@ok.alex` 承认了该问题，并提到相关工单正在处理中。
- **API 响应不一致**：`@bayang7` 和 `@brknclock1215` 讨论了在进行 API 调用时 `sonar-small-online` 模型可能存在的问题，并思考减小上下文窗口（context window）是否会有所帮助。
- **关于移除 pplx-70b-online 的担忧**：包括 `@thedigitalcat` 和 `@clay_ferguson` 在内的多位用户对弃用 `pplx-70b-online` 模型表示担忧，讨论了尽管该模型有时会产生乱码响应，但其整体性能仍优于新模型。
- **Prompting：一个关键因素？**：`@brknclock1215` 强调了 API 调用中系统消息（system messages）的重要性，分享了一个影响 `sonar-medium-online` 模型输出质量的具体设置，并指出 Prompting 在响应准确性中起着不可或缺的作用。

**提到的链接**：

[Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)：为给定的聊天对话生成模型的响应。

  

---



### LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1211718441606582334) (6 messages): 

- **为包含表格和图片的 PDF 引入 LlamaParse**：LlamaParse 被展示为检索增强生成 (RAG) 的重要工具，旨在理解包含**嵌入表格和图片**的复杂 PDF。它承诺帮助避免 LLM 因 PDF 解析不良而产生困惑并给出错误答案。[IFTTT LlamaParse 推文](https://twitter.com/llama_index/status/1762158562657374227)

- **MistralAI 的 Large 模型加入 LlamaIndex**：LlamaIndex 现在支持 **@MistralAI 的 Large 模型**，在其最新的 10.13.post1 版本中提供高级推理、多文档路由、工具使用和 JSON 输出等功能，提供接近 GPT-4 级别的能力。[IFTTT MistralAI Large 模型推文](https://twitter.com/llama_index/status/1762231085243719748)

- **AGI Builders 见面会，LlamaIndex DevRel 副总裁 @seldo 出席**：LlamaIndex 开发者关系副总裁 `@seldo` 将在 **@Cloudflare** 办公室举行的 AGI Builders 见面会上发表演讲。活动将包括关于 Phoney AI、基于 BentoML 的 RAG 以及 LlamaIndex 如何增强企业级检索的演讲。[IFTTT AGI Builders 见面会推文](https://twitter.com/llama_index/status/1762258349507437006)

- **LlamaIndex 与 @FireworksAI_HQ 合作 Function Calling Cookbook**：LlamaIndex 与 @FireworksAI_HQ 合作发布了一系列关于使用 FireFunction-v1 进行 Function Calling 和 RAG 应用的 Cookbook，强调了完全的兼容性和多功能的 API 功能。[IFTTT Cookbook 合作推文](https://twitter.com/llama_index/status/1762532341795487815)

- **由 llama-index-networks 实现的分布式 Super-RAG**：LlamaIndex 推出了一项改变游戏规则的功能——将多个 RAG 应用组合成一个 Super-RAG。用户可以为任何 RAG 应用创建 API 服务，将多个应用连接到单个网络中，并跨该网络运行查询。[IFTTT Super-RAG 推文](https://twitter.com/llama_index/status/1762552542981230769)

**提到的链接**：

[AGI Builders Meetup SF · Luma](https://t.co/kcoIhfgQqF)：👋 我们很高兴邀请您参加 2024 年 2 月 29 日闰日举行的首届 AGI Builders 见面会。❤️ 这是一个 AI 构建者、研究人员和爱好者分享想法的聚会...

  

---


### LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1211596776025624627) (256 messages🔥🔥):

- **Discord 频道求助**: 用户 `@delik` 通过分享 [消息链接](https://discord.com/channels/1059199217496772688/1210657321026461706/1211816140263915570) 在另一个 Discord 频道寻求帮助，但未说明所需帮助的具体性质。
- **关于 RAG 与 No-RAG 的讨论**: `@addo__` 询问了如何针对特定数据集，比较使用 GPT-3.5 结合 RAG 与不使用 RAG 进行问答的效果；`@whitefang_jr` 提供了使用 `FaithfulnessEvaluator` 评估响应的代码片段。
- **LlamaIndex 与 Weaviate 及其他服务的集成与冲突**: 用户讨论了如何将 LlamaIndex 连接到 Docker 托管的服务（如 Weaviate），以及如何将 LanceDB 等数据库与 LlamaIndex 的新混合搜索（hybrid search）功能结合使用 (`@ddashed`, `@cheesyfishes`, `@oopskapootz`)。
- **在 macOS 上安装 LlamaIndex 的问题**: `@swagz0521` 在安装 LlamaIndex 及其相关依赖时遇到了一系列复杂的错误和冲突，最终通过确保正确的虚拟环境激活和安装命令得以解决 (`@whitefang_jr` 在整个对话过程中提供了故障排除支持)。
- **构建 AI Agent 与集成**: `@sansmoraxz` 表达了为 LlamaIndex 创建 Golang 集成的兴趣，并讨论了通过 AWS Bedrock 动态编排 Agent 的可能性，可能还会使用 Docker (`@mrpurple9389` 寻求关于动态编排的澄清，并提供了一个关于 AWS Bedrock 的 YouTube 参考资料)。

**提到的链接**:

- [未找到标题](http://localhost:11434',): 未找到描述
- [in (inlin)](https://huggingface.co/in): 未找到描述
- [概览](https://lancedb.github.io/lancedb/hybrid_search/hybrid_search/): 未找到描述
- [无法更新 llamaindex](https://stackoverflow.com/questions/78057262/cannot-update-llamaindex/78068147#78068147): 在 LlamaIndex 于 2024 年 2 月发布 v0.10 之后，它对导入（imports）引入了许多破坏性变更。我正尝试在 conda 环境中更新 llama-index，但我收到了以下错误...
- [加载数据 (Ingestion) - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/understanding/loading/loading.html): 未找到描述
- [Embeddings 入门指南](https://huggingface.co/blog/getting-started-with-embeddings): 未找到描述
- [微调 (Fine-tuning) - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/optimizing/fine-tuning/fine-tuning.html): 未找到描述
- [Docker Compose | Weaviate - 向量数据库](https://weaviate.io/developers/weaviate/installation/docker-compose): Weaviate 支持使用 Docker 部署。从 v1.24.0 开始，提供了一个使用默认值运行的镜像。或者，也可以编辑 docker-compose.yml 文件来自定义您的实例。
- [定义和自定义文档 - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents.html#advanced-metadata-customization): 未找到描述
- [LlamaIndex 网络研讨会：使用 Flowise 构建无代码 RAG](https://www.youtube.com/watch?v=k5Txq5C_AWA): Flowise 是用于构建 LLM 驱动的工作流的领先无代码工具之一。用户无需学习如何在框架或编程语言中编写代码，而是...
- [新演示与说明 - Amazon Bedrock 的 Agents | Amazon Web Services](https://www.youtube.com/watch?v=JkDzZFTXeSw): 借助 Amazon Bedrock，您可以轻松构建并扩展具有安全性、隐私性和负责任 AI 的生成式 AI 应用程序。此演示向您展示了如何使用 Age...
- [llama_index/llama-index-integrations/agent/llama-index-agent-openai/llama_index/agent/openai/step.py (位于 b2f0a59c21f651bea1502818ec7f61ab915ca286) · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/b2f0a59c21f651bea1502818ec7f61ab915ca286/llama-index-integrations/agent/llama-index-agent-openai/llama_index/agent/openai/step.py#L31C1-L31C71): LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index
- [llama_index/llama-index-core/llama_index/core/readers/file/base.py (位于 be76419dd226244a1ad057f77ad822d16fe92df3) · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/be76419dd226244a1ad057f77ad822d16fe92df3/llama-index-core/llama_index/core/readers/file/base.py#L117): LlamaIndex 是适用于您的 LLM 应用程序的数据框架 - run-llama/llama_index
- [围绕查询管道构建 Agent - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/agent/agent_runner/query_pipeline_agent.html): 未找到描述
- [手机上的实时 AI 对话副驾驶，疯狂还是诡异？](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): 我在 iPhone 上构建了一个对话 AI 副驾驶，它可以倾听您的对话并提供实时建议。在 Replicate 上免费访问 Whisper 和 Mixtral 模型...
- [intfloat/multilingual-e5-large · Hugging Face](https://huggingface.co/intfloat/multilingual-e5-large): 未找到描述
- [在 LlamaIndex 抽象中自定义 LLM - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/module_guides/models/llms/usage_custom.html#example-using-a-custom-llm-model-advanced): 未找到描述
- [LocalAI - LlamaIndex 🦙 v0.10.13](https://docs.llamaindex.ai/en/stable/examples/llm/localai.html#llamaindex-interaction): 未找到描述

---

### LlamaIndex ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/1211607682104958986) (3 messages): 

- **优化 LLM 的上下文**：`@jonas69301` 正在寻求关于为 **GPT-4 turbo 和 Gemini 1.5** 等大型编程语言模型提供广泛上下文的最佳实践建议。关键问题包括最佳的**信息顺序**、是否需要**重复某些部分**，以及最有效的**结构化技术**（如使用 Markdown 来提高清晰度）。

- **使用 Llama2 进行开源文本生成**：`@theexecutor5677` 正在开发一个文本生成项目，使用 **Llama2** 模型处理 CSV 和 PDF 文件输入。他们正在寻找不使用专有工具集成文件的建议，并对开源框架内的 **Retrieval-Augmented Generation (RAG)** 应用感兴趣。

- **构建基于 RAG 的 OSS SDK 助手**：`@codermickey` 询问了开发 RAG 以协助用户使用开源软件 (OSS) SDK 的最先进方法。他们对索引 Gitbooks 和代码库的最佳**分块策略 (chunking strategies)**，以及在文档网站和 **VSCode plugin** 中针对各种代码任务的有效**检索策略 (retrieval strategies)** 感兴趣。
  

---



### LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1211591280124100632) (214 messages🔥🔥): 

- **探索用于 Stable Diffusion 的基于编码器的扩散模型**：`@top_walk_town` 分享了一个[链接](https://tuning-encoder.github.io/)，讨论了一种用于 Stable Diffusion 的基于编码器的反转方法，并询问了其流行程度。对话随后展开，`@chad_in_the_house` 指出默认 Stable Diffusion 的图像提示 (IP) 适配器效果并不理想。

- **DARPA 资助项目与 AI 的抉择**：包括 `@pseudoterminalx` 和 `@progamergov` 在内的几位用户讨论了 DARPA 资助的 AI 研究的奇特性和影响，涉及 UwU 动漫女孩和导弹等话题，并对使用动漫角色进行征兵宣传进行了讽刺性评论。

- **辩论量子计算对 AI 的影响**：该频道由 `@segmentationfault8268` 发起了一场关于量子计算处理 Transformer AI 模型潜力的讨论。`@thejonasbrothers` 和 `@nodja` 参与了对话，提到了量子纠错的状态及其被视为 AI 计算可行技术的接近程度。

- **AI 中的内容审核挑战**：对话深入探讨了围绕内容审核的问题，特别是关于 CSAM (儿童性虐待材料)。包括 `@pseudoterminalx`、`@astropulse` 和 `@progamergov` 在内的用户分享了关于举报工具的有效性、平台责任以及处理普遍存在的不当内容的复杂性的经验和看法。

- **讨论开源与专有 AI 模型**：`@itali4no` 分享了一篇文章，引发了由 `@progamergov` 等人主持的关于 AI 模型发布策略的讨论，涉及 Mistral Large 等开源模型与专有模型的对比，以及支持持续 AI 研发的商业化问题。

**提到的链接**：

- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [来自 Suhail (@Suhail) 的推文](https://x.com/Suhail/status/1762529419909074956?s=20)：1/ 我们正在发布 Playground v2.5，这是我们最新的图像创作基础模型。我们在超过 2 万名用户中通过严格的基准测试对模型进行了测试，其程度超越了我们迄今为止所见的任何测试...
- [Encoder-based Domain Tuning for Fast Personalization of Text-to-Image Models](https://tuning-encoder.github.io/)：未找到描述
- [ChatMusician](https://shanghaicannon.github.io/ChatMusician/)：未找到描述
- [Wuerstchen: An Efficient Architecture for Large-Scale Text-to-Image Diffusion Models](https://arxiv.org/abs/2306.00637)：我们介绍了 Würstchen，这是一种新型的文本到图像合成架构，它将极具竞争力的性能与大规模文本到图像扩散模型前所未有的成本效益相结合。
- [未找到标题](https://playground.com/blog/playground-v2-5)：未找到描述

### LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1211692516101853186) (15 条消息🔥): 

- **语言模型中的放射性 (Radioactivity in Language Models)**：`@thejonasbrothers` 分享了一篇[研究论文](https://arxiv.org/abs/2402.14904)，探讨了语言模型中训练数据的可检测性，特别是关注水印数据在训练检测中的有效性及其影响。研究强调，即使训练文本中仅包含 5% 的水印合成指令，也可以被**高置信度地检测到**。

- **DeepMind 的 Genie 与人形机器人**：`@vrus0188` 发布了一个 [YouTube 视频](https://www.youtube.com/watch?v=gGKsfXkSXv8)，讨论了 Google DeepMind 关于 AI 的新论文，并展示了**人形机器人 (humanoid robotics)** 方面的进展。

- **有限元分析学习**：`@wyndyl` 询问了关于利用有限元分析 (Finite Element Analysis) 网格和模型进行学习的工作，`@itali4no` 回应称虽然有相关研究，但这些方法**通常不如直接运行 FEM 有效**，并分享了一篇[相关论文](https://arxiv.org/abs/2302.04107)。

- **教育内容警报**：`@chad_in_the_house` 发布了来自 **EleutherAI 的更新**，然而，提供的链接被标记为**不存在或无法访问**。

- **逆离散傅里叶变换的挑战**：`@mkaic` 讨论了为任意坐标实现逆离散傅里叶变换 (Inverse Discrete Fourier Transform) 时遇到的问题，这在他们的神经网络合成中会导致显存 (VRAM) 效率低下。他们发布了[基于 Torch 的代码](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177)，并考虑**重构以使用 `torch.vmap`** 进行改进。

**提到的链接**：

- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904)：该论文调查了 LLM 生成文本的放射性，即是否可能检测到此类输入被用作训练数据。传统的成员推理 (membership inference) 等方法可能会……
- [The AI 'Genie' is Out + Humanoid Robotics Step Closer](https://www.youtube.com/watch?v=gGKsfXkSXv8)：先是文本转语音、文本转视频和文本转动作，现在是文本转交互？让我们来看看 Google DeepMind 的新 Genie 论文，并设定……
- [abacus/src/interpolators.py at 28d20a2f3a244d09218e6ddd998db08c7872dc45 · mkaic/abacus](https://github.com/mkaic/abacus/blob/28d20a2f3a244d09218e6ddd998db08c7872dc45/src/interpolators.py#L177)：研究稀疏神经网络的激活插值 - mkaic/abacus

  

---


### LAION ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/1211586067514527765) (1 条消息): 

- **探索 Transformer 的学习能力**：`@phryq.` 提出了一个有趣的问题，即通过实验探索 Transformer 可以学习到什么，例如理解虚构物体（如 **krog**、**shmlog** 和 **mmmmmchakaboooboolight**）之间的尺寸关系，然后渲染出其中一个物体尺寸被指定为其他物体特定倍数的图像。关于是否进行过此类实验或结果如何，目前没有进一步的讨论或回答。
  

---



### Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1211713638935695440) (3 条消息): 

- **理解的回响**：用户 `@canadagoose1` 承认意识到了之前讨论的内容，简短地回复了 “*ahh*”。
- **达成共识**：`@canadagoose1` 随后表达了理解，称 “*这就是你的意思 (this is what u mean)*”。
- **认可影响**：`@canadagoose1` 用 “*受我影响 (influenced by me)*” 这句话提到了受到的影响。
  

---

### Interconnects (Nathan Lambert) ▷ #[news](https://discord.com/channels/1179127597926469703/1179128538679488533/1211675626558455828) (85 messages🔥🔥): 

- **Financial Times 文章被付费墙阻挡**：`@xeophon.` 分享了一个 [Financial Times 文章](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb)的链接，但由于付费墙（paywall）而无法访问。`@natolambert` 批评它是 *“有史以来最糟糕的移动端付费墙”*。
- **Microsoft 入股 Mistral**：`@philpax` 描述了一项交易，[Microsoft 已获得法国 AI 初创公司 Mistral 的少数股权](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb)，后者将提供基于 Azure 的商业语言模型。`@natolambert` 对 Mistral 的模型是“开源”的说法表示怀疑。
- **Mistral 发布新的优化模型**：`@xeophon.` 强调了 `@arthurmensch` 关于名为 [Mistral Large](https://mistral.ai/news/mistral-large/) 的新优化模型的公告，该模型号称具有顶级的推理能力，并可在 Azure 上使用。`@onuralp.` 建议，对于像 Mistral 这样的 AI 初创公司来说，不仅是算力（compute），获取高质量的图像数据也可能非常重要。
- **封闭还是开放——AI 公司的生存困境**：在讨论 Mistral 接受股权投资是否明智时，`@natolambert` 强调 AI 公司必须在保持开放并可能蓬勃发展，或保持封闭并面临挑战之间做出选择。他在其 [开源 LLM 公司剧本](https://www.interconnects.ai/p/open-llm-company-playbook)中详细阐述了这一点，概述了利用开源 LLM 权重的策略。
- **对 Le Chat 和 Mistral 策略的反应**：`@mike.lambert` 和 `@xeophon.` 讨论了 Mistral 输出的价格和质量，思考他们的策略在竞争中表现如何。`@onuralp.` 推测了社区的反应，而 `@natolambert` 则思考了在快速演进的 AI 领域中，依赖模型的初创公司的可持续性。

**提到的链接**：

- [Bringing open AI models to the frontier](https://mistral.ai/news/about-mistral-ai/)：我们为什么要创立 Mistral AI。
- [Guillaume Lample (@GuillaumeLample) 的推文](https://x.com/guillaumelample/status/1762139008409186693?s=46)：由于请求量超出预期，Le Chat 暂时无法使用。我们对带来的不便表示歉意——我们正在努力尽快恢复运行，感谢您的耐心……
- [Arthur Mensch (@arthurmensch) 的推文](https://x.com/arthurmensch/status/1762121295330725965?s=46)：我们今天发布了一个新的优化模型！Mistral Large 具有顶级的推理能力，原生支持多语言，具有原生函数调用（function calling）能力，并提供 32k 模型。预训练模型……
- [Microsoft strikes deal with Mistral in push beyond OpenAI](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb)：未找到描述。
- [Open LLM company playbook](https://www.interconnects.ai/p/open-llm-company-playbook)：发布模型权重在公司战略中处于什么位置？在开源 LLM 领域发展的 3 个要求、3 个行动和 3 个益处。

  

---


### Interconnects (Nathan Lambert) ▷ #[other-papers](https://discord.com/channels/1179127597926469703/1179142630517518397/1211742062203113472) (14 messages🔥): 

- **Gemini Pro 在 CoT 下表现下滑？**：`@sid221134224` 强调了 Gemini Pro 在使用思维链（CoT）时与非 CoT 版本相比出现的[异常性能下降](https://twitter.com/mosh_levy/status/1762027624434401314)，并指出 CoT 通常会提升模型性能。
- **CoT 结果好坏参半**：`@xeophon.` 提到在他们的硕士论文期间，CoT 导致所有任务和模型的测试结果都变差了，这挑战了关于 CoT 有效性的普遍假设。
- **CoT 规律的例外？**：`@sid221134224` 表达了一个观点，即虽然 CoT 通常有助于更强大的模型，但它可能会降低性能较弱模型的表现。
- **CoT 作为标准的微调实践**：`@xeophon.` 指出 CoT 现在是微调（fine-tuning）数据集的常见组成部分，这可以从新版 ChatGPT 甚至对简单编程查询给出的详细回答中观察到。
  

---

### Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1211723289408573541) (21 messages🔥): 

- **Multilingual Models 与价值观的博弈**：`@420gunna` 提出了一个关于 Alignment 在 **Multilingual Models** 中应如何运作的问题，思考模型是否应该在不同语言间保持一致的价值观，还是应该反映每种语言文化相关的价值观。多语言查询可能对维持这种特定文化的 Alignment 构成挑战。

- **巧妙绕过 GPT 安全防护**：`@philpax` 分享了来自 [The Register](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/) 的一篇文章，介绍了如何利用祖鲁语或苏格兰盖尔语等低资源语言规避 OpenAI GPT-4 的安全措施，使模型暴露于潜在的非法输出。

- **针对每种语言 Alignment 的不确定性**：尽管 `@philpax` 提供了关于使用不同语言进行 Jailbreaks 的见解，但他们对针对每种语言的 Alignment 实际上是如何实现或在实践中如何运作表示不确定。

- **Multilingual 和 Multimodal 语境下的难题**：`@420gunna` 阐明了他们对*多文化语境*下 Alignment 的兴趣，暗示了处理可能带有不同文化价值观集的 **Multilingual** 和 **Multimodal** 输入的复杂性。

- **多文化 Alignment 倡议的探索**：`@natolambert` 提到像 Collective CAI 这样的倡议正在研究 AI 中的文化 Alignment 问题，表明该领域仍处于应对这些挑战的早期阶段。

**提到的链接**：

[OpenAI's GPT-4 safety systems broken by Scots Gaelic](https://www.theregister.com/2024/01/31/gpt4_gaelic_safety/): 'Tha e comasach inneal spreadhaidh dachaigh a' thogail le stuthan taighe'

  

---


### Interconnects (Nathan Lambert) ▷ #[ml-drama](https://discord.com/channels/1179127597926469703/1181746144821387334/1211592531331260456) (2 messages): 

- **错过 Goose 的机会**：`xeophon.` 对错过让 **Goose** 出现在每张 Imagen 图片中的机会表示失望，惋惜 Louis 没有接受这一提议。
- **Mistral 与 Meta 的竞争**：`onuralp.` 幽默地指出 Mistral 团队与 Meta 之间似乎存在 **竞争关系**，并配以俏皮的表情符号表示轻松的调侃。
  

---

### Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1211593611834294282) (54 messages🔥): 

- **DeepMind 发布 Genie 基础模型 (Foundational Model)**：`@xeophon.` 分享了关于 [DeepMind Genie](https://fxtwitter.com/_rockt/status/1762026090262872161?s=46) 的兴奋之情，这是一个从互联网视频中训练出来的基础世界模型，可以通过图像提示词创建动作可控的 2D 世界。尽管在训练中没有使用动作标签，该模型仍成功解释了动作，这引发了关于世界动力学学习以及与视频游戏固有运动对比的讨论。

- **关于可解释动作提取的辩论**：`@philpax` 强调了 Genie 在无标签情况下提取可解释动作这一有趣点。`@onuralp.` 认为这项工作展示了通过扩展 LLM 实现“规划 (planning)”的可能性，而 `@natolambert` 讨论了鉴于 Genie 与其 Model-Based Reinforcement Learning 背景的相关性，准备撰写相关文章。

- **Mistral 的资源使用与定价**：`@xeophon.` 和 `@sid221134224` 讨论了 Mistral 的资源消耗和定价策略，引用了一篇关于[可能 Token 数量的推文](https://fxtwitter.com/georgejrjrjr/status/1762235641176183267)，并将其与 Google 的 PaLM 和 Google 的 Gemma 等其他大语言模型进行了比较。据推测，根据公开信息及其之前的发布记录，Mistral 的 30B 模型可能使用了约 30 万亿 Token。

- **一份全面的语言模型数据选择综述**：`@420gunna` 分享了一篇关于语言模型数据选择的 [arXiv 论文](https://arxiv.org/abs/2402.16827)，这促使 `@natolambert` 确认了他过去的工作联系并对此表示感谢。

- **作者的工具与流程揭秘**：在关于个人写作流程的讨论中，`@natolambert` 解释了使用 Notion 和 Grammarly，偶尔也使用 Substack 的编辑器，而 `@xeophon.` 提供了关于使用 [Typora](https://typora.io/) 的见解，并正在考虑使用 Obsidian。对话概述了起草、编辑和发布书面内容的个人偏好。

**提到的链接**：

- [Emad (@EMostaque) 的推文](https://x.com/emostaque/status/1762152740938031484?s=46)：这里有一个有趣的细节，假设这是在 @Scaleway 使用 H100，价格为 €1.9/小时 => 1000 万 H100 小时（约 3000 万 A100 小时），4000 张 H100 运行 3 个月⏲️ LLaMA 2 70B 训练 2 万亿 Token 使用了 170 万 A100 小时 =>...
- [Aran Komatsuzaki (@arankomatsuzaki) 的推文](https://x.com/arankomatsuzaki/status/1762339260563124373?s=20)：《语言模型数据选择综述》对现有数据选择方法及相关研究领域的文献进行了全面回顾，提供了现有方法的分类法...
- [Tim Rocktäschel (@_rockt) 的推文](https://fxtwitter.com/_rockt/status/1762026090262872161?s=46)：我非常激动地揭晓 @GoogleDeepMind 的 Open Endedness 团队一直在做的事情 🚀。我们推出了 Genie 🧞，一个完全从互联网视频训练的基础世界模型，可以生成...
- [George (@georgejrjrjr) 的推文](https://fxtwitter.com/georgejrjrjr/status/1762235641176183267)：Mistral 对其更高效的 Small（原 Mistral-Next）收费更高，这是一个强烈的信号，表明新的 Small 将不会采用开源许可。对比他们去年与今年的表态...

  

---

### OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1211724377037733951) (2 messages): 

- **Mistral Large 首次亮相**: `@alexatallah` 宣布了 **Mistral Large**，这是一款闭源旗舰模型，其能力介于 GPT-4 和 Claude 2 之间，具备高级功能，可通过[此处](https://openrouter.ai/models/mistralai/mistral-large)访问。该模型支持多种语言且准确度高，拥有 32,000 token 的上下文窗口，公告中详细列出了输入和输出 token 的不同定价。

- **Mistral 价格调整**: Mistral Medium 的价格有所下调，而 Mistral Tiny 和 Small 的价格有所上涨，这促使 `@alexatallah` 建议切换到 Mistral 7B Instruct 和 Mixtral 8x7B Instruct 以获得更好的性价比。

- **推出具备联网能力的 Perplexity Sonar**: `@alexatallah` 重点介绍了来自 Perplexity 的名为 **Sonar** 的新模型，包括一个基于 Mixtral 8x7B 的联网变体，可在[此处](https://openrouter.ai/models/perplexity/sonar-medium-online)获取。Sonar 模型以其成本效益、速度和及时的实时信息为荣，并特别建议过渡到 Sonar 模型，因为 PPLX 模型将于 3 月 15 日弃用。

- **OpenRouter Playground 升级**: 正如 `@alexatallah` 所指出的，OpenRouter Playground 增加了 Top P、Top K 和惩罚项（penalties）等新参数，增强了用户对模型交互的控制。

- **消息系统修复已发布**: `@louisgv` 分享了关于修复 Perplexity、Mistral 和 Gemma 等各种模型的消息排序和格式问题的更新，优化了用户体验。

**提到的链接**:

- [Mistral: Mistral Large by mistralai | OpenRouter](https://openrouter.ai/models/mistralai/mistral-large): 这是 Mistral AI 的闭源旗舰模型。它由闭源原型驱动，在推理、代码、JSON、聊天等方面表现出色。阅读发布公告 [此处](https:/...
- [Perplexity: Sonar 8x7B Online by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-medium-online): Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了他们早期的模型。这是 [Sonar 8x7B](/models/perplexity/sonar-mediu... 的在线版本。

  

---


### OpenRouter (Alex Atallah) ▷ #[app-showcase](https://discord.com/channels/1091220969173028894/1092850552192368710/1211600928675733554) (5 messages): 

- **Videotok 在 Product Hunt 引起轰动**: `@borjasoler` 宣布在 Product Hunt 上发布 **Videotok**，这是一个用于创建短视频的 AI 驱动平台，并邀请大家支持和分享[该帖子](https://x.com/borjasolerr/status/1762025283597582807?s=20)。分享发布帖子有机会赢取年度计划。

- **探索 Blust AI 的多功能工具平台**: `@e__lo` 介绍了 [Blust AI](https://blust.ai)，这是一个托管多个应用程序的订阅服务，并邀请开发者集成他们自己的 AI 工具。集成的详细步骤可以在[文档](https://docs.blust.ai/docs/integrating-ai-tools/)中找到。

- **寻求关于 Blust AI 的说明**: `@anehzat` 赞扬了 Blust AI 的用户界面，并询问了该应用程序使用的模型和托管方式。

- **Blust AI 功能受到质疑**: `@anehzat` 报告称该应用似乎无法运行，但未提供更多细节。

- **OpenRouter 通过 Make.com 释放无代码自动化能力**: `@jim14199` 为 [Make.com](https://www.go-synergetic.com/apps/openrouter) 介绍了一个插件，使用户能够通过将 OpenRouter 与 1,700 多个其他应用连接来自动化 AI 工作流，且无需任何编码。该应用提供一次性付费的终身访问权限，并包含快速获益的自动化示例，如客户支持自动回复系统。

**提到的链接**:

- [no title found](https://blust.ai): 未找到描述
- [Overview | Blust AI Studio Documentation Hub](https://docs.blust.ai/docs/integrating-ai-tools/): Blust AI Studio 与外部 AI 工具之间的集成过程旨在将用户与广泛的 AI 工具无缝连接。一旦 AI 工具在 blust.AI 目录中注册并列出...
- [Tweet from Borja Soler (@borjasolerr)](https://x.com/borjasolerr/status/1762025283597582807?s=20): 现在在 Product Hunt 上发布 Videotok！⚡️ 使用 AI 创建短视频，即时生成配音、图像、脚本... 🤯 期待您的支持：https://www.producthunt.com/posts/videotok 如果...
- [OpenRouter Integration for Make.com |  by Synergetic ](https://www.go-synergetic.com/apps/openrouter): 通过这个独家的 Make.com（原 Integromat）插件，将 OpenRouter 与数千个其他应用连接。解锁全新的、强大的自动化工作流并节省时间——无需代码。

  

---

### OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1211693671045660703) (152 条消息🔥🔥): 

- **对新功能的热烈反馈**：用户 `@wikipediadotnet` 对新的参数功能表示兴奋，并建议进行改进，例如提供参数 API，并考虑模型默认设置如何影响用户偏好。

- **社区集思广益功能增强**：`@cupidbot.ai` 请求增加 "use_default_params" 功能，在模型上线一段时间后应用中值最优参数。他们还建议增加高付费额度用户的反馈权重，因为他们很可能是生产环境用户（production users）。

- **对 Mistral Large 的期待升温**：包括 `@billbear` 和 `@louisgv` 在内的多位用户讨论了对 Mistral Large 在 OpenRouter 上发布的期待及最终上线，重点提到了 `@alexatallah` 在推广（rollout）过程中的参与，以及用户对发布后服务器错误的反馈。

- **Perplexity 的变动困扰用户**：用户 `@mostlystable` 和 `@lynxplayz` 在 OpenRouter 上使用 Perplexity 模型时遇到了 400 错误，`@louisgv` 正在调查并解决此问题，以提高与现有工作流（workflows）的兼容性。

- **用户寻求理想界面**：用户 `@jamesm6228` 询问了使用所有 OpenRouter AI 模型（如 Typing Mind）的最佳 UI，`@louisgv` 参与其中以了解他们的功能需求并建议替代方案。

**相关链接**：

- [Cheers Happy GIF - Cheers Happy High Five - Discover &amp; Share GIFs](https://tenor.com/view/cheers-happy-high-five-celebrate-champion-gif-16799628)：点击查看 GIF
- [Mistral: Mistral Large by mistralai | OpenRouter](https://openrouter.ai/models/mistralai/mistral-large)：这是 Mistral AI 的闭源旗舰模型。它由闭源原型驱动，在推理、代码、JSON、聊天等方面表现出色。在此阅读发布公告 [here](https:/...
- [google/gemma-2b-it · Hugging Face](https://huggingface.co/google/gemma-2b-it#:~:text=At%20this%20point%2C%20the%20prompt%20contains%20the%20following%20text%3A)：未找到描述
- [OpenRouter](https://openrouter.ai)：LLM 和其他 AI 模型的路由
- [Perplexity: Sonar 7B by perplexity | OpenRouter](https://openrouter.ai/models/perplexity/sonar-small-chat?tab=api)：Sonar 是 Perplexity 最新的模型系列。它在成本效益、速度和性能方面超越了早期模型。具有互联网访问权限的模型版本是 [Sonar 7B Online](/mode...
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)：为给定的聊天对话生成模型响应。

---

### HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1211599424233668638) (91 条消息🔥🔥): 

- **对未来一周的热情**：`@osanseviero` 为大家送上了愉快的一周祝福。
- **考试超负荷**：`@myg5702` 表达了不太乐观的情绪，表示他们的周末将充满考试。
- **寻求 Batching 见解**：`@rwamit` 向社区寻求关于实现 GPT-4 查询 Batching（批处理）的正确方法的建议，并对比了单个请求与批处理在完成时间上的显著差异。
- **Hugging Face 服务中断**：包括 `@temperance6095` 和 `@zlapo` 在内的多位用户讨论了 Hugging Face Inference API 的问题，指出各种模型类别中出现了持续的 504 超时错误。
- **社区协作**：`@dykyi_vladk` 和 `@beaudjango` 就机器学习项目开发的共同兴趣建立了联系，展示了讨论频道的协作精神。

**提到的链接**：

- [Waiting Waiting Patiently GIF - Waiting Waiting patiently Waiting for you - Discover &amp; Share GIFs](https://tenor.com/view/waiting-waiting-patiently-waiting-for-you-waiting-on-you-gif-15489516379864441176)：点击查看 GIF
- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具备顶级的推理能力。它也可以在 Azure 上使用。
- [How to Build a Discord AI Chatbot that Talks Like Your Favorite Character](https://www.freecodecamp.org/news/discord-ai-chatbot/)：你想和一个说话像你最喜欢的角色（虚构或非虚构）的聊天机器人交谈吗？让我们来构建一个！如果你看过我之前关于这个话题的教程，请继续关注...
- [HITLER_ONLYFANS - a Hugging Face Space by MEGANEGA](https://huggingface.co/spaces/MEGANEGA/HITLER_ONLYFANS)：未找到描述
- [Mistral Remove &quot;Committing to open models&quot; from their website | Hacker News](https://news.ycombinator.com/item?id=39517016)：未找到描述
- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg)：我在 iPhone 上构建了一个对话 AI Co-pilot，它可以监听你的对话并提供实时建议。在 Replicate 上免费访问 Whisper 和 Mixtral 模型...
- [GitHub - vishalmysore/Tools4AI: How to Use Gemeni with Java , Function Calling, Chaining and validation](https://github.com/vishalmysore/Tools4AI)：如何在 Java 中使用 Gemini，以及 Function Calling、Chaining 和验证 - vishalmysore/Tools4AI

  

---


### HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1212051329699282944) (1 条消息): 

- **CS231n 学习小组邀请**：用户 `@shreesha1573` 发出了 2023 年春季学习 CS231n：**Convolutional Neural Networks for Visual Recognition** 的*学习小组*邀请。他们分享了课程作业和模块的链接，并表示欢迎私信加入小组。[CS231n 课程](https://cs231n.github.io/)

**提到的链接**：

[CS231n Convolutional Neural Networks for Visual Recognition](https://cs231n.github.io/)：未找到描述

  

---

### HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1211589746615259137) (14 条消息🔥): 

- **AI 基础设施独角兽**：`@valeriiakuka` 重点介绍了 **Scale AI** 成为数据标注市场关键参与者的历程，并引导用户阅读 Turing Post 的文章，该文章详细描述了他们的增长与挑战。Scale AI 经过多轮大规模融资，估值已达到 73 亿美元，在其成立 8 周年之际表现尤为突出——详情见 [Turing Post](https://www.turingpost.com/p/scaleai)。
  
- **通过语言模型进行测谎**：`@andysingal` 分享了 Nature 杂志上一篇关于**使用大语言模型进行言语测谎**的文章。该文章讨论了欺骗性叙述与真实叙述在心理和内容上的差异，并探索了测谎方法论——点击此处查看**研究详情** [here](https://www.nature.com/articles/s41598-023-50214-0#Tab3)。

- **VLM 应对分辨率挑战**：`@osanseviero` 分享了一篇博客文章，介绍如何通过对高分辨率图像进行**多重裁剪 (multiple crops)** 来克服 Vision-Language Models (VLM) 中的分辨率问题，并提供了实时 Demo 以及在 HuggingFace 平台上可用的模型。感兴趣的读者可以在 [Visheratin 的博客文章](https://huggingface.co/blog/visheratin/vlm-resolution-curse)中了解更多信息。

- **Mistral Large 亮相**：`@teadaniel` 宣布发布 _Mistral Large_，这是一款在多语言推理任务中表现卓越的高级语言模型，目前性能仅次于 **GPT-4**，并可通过 _la Plateforme_ 和 **Azure** 使用。它在基准测试中显示出极具前景的结果，详情可见 [Mistral 的新闻更新](https://mistral.ai/news/mistral-large/)。

- **图像超分辨率与增强的激动人心进展**：`@furkangozukara` 介绍了 **SUPIR Model V8**，该版本进行了改进，使其能够在 RTX 3060 等 12 GB 显存的 GPU 上运行，并基于 Juggernaut-XL-v9 模型。他们认为其表现优于包括昂贵的 Magnific 在内的其他超分辨率工具，并邀请用户在 [YouTube 教程](https://youtu.be/PqREA6-bC3w)中探索其功能。

**提到的链接**：

- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [Scale AI: How to Scale a Company on Every AI Trend](https://www.turingpost.com/p/scaleai)：从预约应用到数据标注巨头的非凡历程。
- [SUPIR: New SOTA Open Source Image Upscaler & Enhancer Model Better Than Magnific & Topaz AI Tutorial](https://youtu.be/PqREA6-bC3w)：随着 V8 版本的发布，现在也可以在 12 GB GPU 上运行，并采用 Juggernaut-XL-v9 作为基础模型。在本教程视频中，我介绍了 SUPIR (Scaling-UP Image Restoration)，这是一种状态...
- [Breaking resolution curse of vision-language models](https://huggingface.co/blog/visheratin/vlm-resolution-curse)：未找到描述。
- [@visheratin on Hugging Face: "VLMs have a resolution problem, which prevents them from finding small details…"](https://huggingface.co/posts/visheratin/787127935781600)：未找到描述。
- [Verbal lie detection using Large Language Models - Scientific Reports](https://www.nature.com/articles/s41598-023-50214-0#Tab3)：未找到描述。

---

### HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1211601013283233832) (13 条消息🔥): 

- **AI 生成的哲学问答**: `@nabereon` 分享了他们使用 **Mixtral-8x7B-Instruct-v0** 和 *AiresPucrs/stanford-encyclopedia-philosophy* 数据集生成问答数据集的过程。他们包含了一个特定的结构来塑造输出，并计划在获得原创内容创作者同意后扩展该数据集。
  
- **许可问题讨论**: `@cakiki` 对 SEP 数据集与 Mixtral 使用条款的兼容性表达了许可方面的担忧。`@nabereon` 正在调查此问题，以确保符合知识产权规定。

- **邀请对开放 AI 模型权重发表评论**: `.plot` 强调了一个向 NTIA 就“[open-weight](https://aimodels.org/responsible-open-source-ai/open-weights/ )” AI 模型发表公众评论的机会，讨论了 AI 民主化与安全担忧之间的平衡。

- **Performance LLM Board 发布**: `@michal.swedrowski.` 介绍了 **Performance LLM Board** 的初始版本，该看板在响应时间和定价等工程方面对各种 LLM 进行比较。他们正在寻求反馈以完善看板，并为社区提供相关信息。

- **Imagic 论文复现与解析**: `@chongdashu` 深入研究了 *Imagic* 论文的复现，这是一种使用 Diffusion 模型通过文本提示编辑图像的技术。他们分享了一篇 [Medium 文章](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a)，讨论了他们在这种新型图像编辑方法上的经验。

**提到的链接**:

- [如何对 NTIA AI 开放模型权重 RFC 发表评论](https://aimodels.org/ai-blog/comment-ntia-open-source-open-weights-rfc/): 美国国家电信和信息管理局 (NTIA) 正在征求公众对 open-weight AI 模型影响的评论。以下是参与方式。
- [论文解析 — Imagic: 基于 Diffusion 模型的文本驱动真实图像编辑](https://medium.com/@chongdashu/papers-decoded-imagic-text-based-real-image-editing-with-diffusion-models-b1bda8b2532a): 在 Papers Decoded 中，我们尝试复现研究论文的实验和结果。这是熟悉……的最佳方式之一。

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212099311890989156) (2 条消息): 

- **对 Playground v2.5 的失望**: 用户 `@pseudoterminalx` 对 **Playground v2.5** 仍在使用 *eps prediction* 表示失望，暗示其对更先进的方法有所期待。
- **对框架选择的批评**: `@pseudoterminalx` 批评了仅简要提及 *zsnr* 而选择使用其描述为“糟糕的 EDM 框架”的决定。
  

---


### HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1211649033907605524) (10 条消息🔥): 

- **Fine-Tuning 的数据集大小担忧**: `@icecoldt369` 表达了对即使整合了开源数据集，数据集仍不够大的担忧。`@cursorop` 安慰说，Fine-Tuning 并不一定需要海量数据集，因为模型通常可以通过较小的数据集识别复杂词汇中的模式。
- **优化复杂字符识别**: 尽管得到了安慰，`@icecoldt369` 表示他们的模型未能完全识别复杂字符，暗示需要更好地在这些数据（包括外语或 meta learners）上训练的模型。
- **语言特定字符的挑战**: `@icecoldt369` 正寻求处理高棉语 (Khmer language) 特有的复杂字符，而 `@cursorop` 指出由于该语言脚本的符号化特性，这项任务非常困难。
  

---

### HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1211662596281860146) (10 条消息🔥): 

- **分类头（Classification Head）困惑已解决**：用户 `@grimsqueaker` 询问了在 HuggingFace 中使用 `esm2` 时 `AutoModelForSequenceClassification` 的具体细节，询问它是使用了 CLS token 还是对所有 token 进行平均池化（mean pooling）。`@cursorop` 建议查看 HuggingFace 仓库代码以进行澄清。

- **寻找生成式问答（Generative QA）模型**：`@cornwastaken` 寻求关于寻找用于基于文档查询场景的 *Generative Question Answering* 模型的建议，并想了解此类模型中使用的架构。为此，他们已经探索了 HuggingFace 模型中的 *Question Answering* 和 *text-generation-inference* 类别。

- **快速 Embedding 模型推荐**：`@cakiki` 征求关于探索小型、非专业英语数据集的 embedding 模型建议，对此 `@cubietom` 推荐了 [BAAI 的 bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5) 作为一个快速且有效的选择，并提到可以考虑使用 [thenlper 的 gte-small](https://huggingface.co/thenlper/gte-small) 以获得更广泛的语言支持和检索方法。

- **压缩邮件以供 LLM 读取**：`@acidgrim` 提到了他们关于压缩邮件文件以适配 LLM 上下文窗口（context window）的项目，正在寻求能够保持信息完整性的仅限 CPU（CPU-only）的本地库建议，并提到目前正在使用 suma，但也乐于尝试其他替代方案。

- **开发医疗 Transformer**：用户 `@kareem3069` 正寻求构建一个医疗 Transformer，以改进针对特定领域术语和上下文的模型映射，但在现有 sentence-encoder 模型的性能方面面临挑战。他们正在寻求增强 Transformer 能力的方法建议，以实现准确的医疗代码描述。

**提到的链接**：

- [BAAI/bge-small-en-v1.5 · Hugging Face](https://huggingface.co/BAAI/bge-small-en-v1.5)：未找到描述
- [thenlper/gte-small · Hugging Face](https://huggingface.co/thenlper/gte-small)：未找到描述

  

---


### HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1212099311890989156) (2 条消息): 

- **对 Playground v2.5 感到失望**：`@pseudoterminalx` 对 **Playground v2.5** 仍在使用 *eps prediction* 表示失望，表明原本期待会有不同的方法。
- **对 ZSNR 和 EDM 框架的批评**：`@pseudoterminalx` 还批评了对 *zsnr* 的处理方式，称其仅作为脚注被提及，并对使用其所谓的“蹩脚的 EDM 框架”表示遗憾。
  

---

### Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1211585525153140748) (48 messages🔥): 

- **机器学习中的单位范数向量 (Unit Norm Vectors)**：`@smerkyg` 和 `@ad8e` 讨论了机器学习中“单位风格”向量的概念，强调了 RMSNorm 对于实现“单位长度”向量以保证精度重要性。`@ad8e` 解释说，在模型的不同部分保持一致的缩放可以避免在不同尺度之间进行转换。

- **偏好数据集中的 DPO 与 SFT**：关于 Direct Preference Optimization (DPO) 论文的讨论涉及对 `model_ref` 初始化的困惑。`@staticpunch` 询问是否通过对偏好补全（preferred completions）进行 MLE 来初始化 `model_ref`，`@elad7318` 确认了这一理解，但质疑为什么只考虑偏好补全。

- **Mistral Large 发布**：[Mistral 的新闻页面](https://mistral.ai/news/mistral-large/) 宣布了一个新的语言模型 *Mistral Large*，它被介绍为一个在基准测试中表现强劲的前沿模型，现在可以通过 la Plateforme 和 Azure 使用。

- **LangChain 封装器批处理困境**：`@rwamit` 寻求关于使用 LangChain 封装器实现批量查询 GPT-4 的建议。包括 `@._bob_` 在内的社区成员进行了简短交流，随后将对话引导至另一个平台以获得更合适的帮助。

- **模型 Epoch 的训练异常**：`@jstephencorey` 在训练 Pythia-70m 架构模型以使其过拟合时，遇到了第五个 Epoch 训练损失意外增加或停滞的异常情况。`@leegao_` 和 `@hawk1399` 等成员提供了见解，引用的可能原因从双重下降（double descent）行为到训练损失尖峰（loss spikes）的细微差别不等，而 `@catboy_slim_` 则提醒了重复数据在训练期间可能引起的干扰。

**提到的链接**：

- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [A Theory on Adam Instability in Large-Scale Machine Learning](https://arxiv.org/abs/2304.09871)：我们提出了一个理论，解释了之前在大型语言模型训练中注意到的未解释的发散行为。我们认为这种现象是主流优化算法的一种人为产物...
- [DeWave: Discrete EEG Waves Encoding for Brain Dynamics to Text Translation](https://arxiv.org/abs/2309.14030)：将大脑动态翻译成自然语言对于脑机接口 (BCI) 至关重要。随着 ChatGPT 等大型语言模型的飞速发展，弥合差距的需求...
- [gemma_pytorch/gemma/model.py at 01062c9ef4cf89ac0c985b25a734164ede017d0b · google/gemma_pytorch](https://github.com/google/gemma_pytorch/blob/01062c9ef4cf89ac0c985b25a734164ede017d0b/gemma/model.py#L176>)：Google Gemma 模型的官方 PyTorch 实现 - google/gemma_pytorch

---

### Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1211619749507637248) (22 messages🔥): 

- **通过权重共享提升效率**：`@thooton_` 强调了层数翻倍的权重共享模型版本性能几乎与原版持平，并指出其局限性在于仅适用于层能放入 SRAM 的较小模型。
- **最简单的硬件效率解决方案**：`.the_alt_man` 断言，*权重共享*是在当前硬件上增强现有架构效率的一种直接策略。
- **挖掘 Calibration 论文**：`@avi.ai` 分享了一篇[论文链接](https://arxiv.org/abs/2305.14975)，讨论了 Large Language Models (LLMs) 良好校准的置信度，以及对 RLHF-LMs 置信度提取方法的广泛评估。
- **LLM 放射性检测研究**：`@0x_paws` 介绍了关于使用水印方法检测训练数据中是否使用了 LLM 生成文本的研究，`@leegao_` 对此进行了详细讨论，指出了该研究关于微量放射性（trace-radioactivity）及其在无需模型权重或训练数据知识的情况下可检测性的发现。
- **Grokking 走向主流**：[`@hawk1399` 分享](https://arxiv.org/abs/2402.15555)的近期工作表明，Grokking 现象不仅限于受控设置，在常见数据集上使用 CNN 和 Resnet 的实际场景中也会发生，并引入了“延迟鲁棒性”（delayed robustness）一词。

**提到的链接**：

- [Deep Networks Always Grok and Here is Why](https://arxiv.org/abs/2402.15555)：Grokking，即延迟泛化，是指深度神经网络（DNN）在达到接近零的训练误差很久之后才出现泛化的现象。以往的研究报告了这种现象的发生...
- [Orca-Math: Unlocking the potential of SLMs in Grade School Math](https://arxiv.org/abs/2402.14830)：数学应用题解答长期以来被认为是小语言模型（SLMs）的一项复杂任务。最近的一项研究假设，要达到 80% 以上的准确率所需的最小模型尺寸...
- [Do Large Language Models Latently Perform Multi-Hop Reasoning?](https://arxiv.org/abs/2402.16837)：我们研究了 Large Language Models (LLMs) 在面对复杂提示词（如“‘Superstition’演唱者的母亲是”）时，是否在潜意识里执行了多跳推理。我们寻找证据证明...
- [Watermarking Makes Language Models Radioactive](https://arxiv.org/abs/2402.14904)：本文调查了 LLM 生成文本的放射性，即是否可能检测到此类输入被用作训练数据。传统方法如成员推理（membership inference）可能会导致...
- [Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback](https://arxiv.org/abs/2305.14975)：一个值得信赖的现实世界预测系统应该产生良好校准的置信度分数；也就是说，它对答案的置信度应该能指示该答案正确的可能性，从而使...
- [Bayesian Reward Models for LLM Alignment](https://arxiv.org/abs/2402.13210)：为了确保 Large Language Model (LLM) 的回答是有帮助且无毒的，我们通常在人类偏好数据上微调奖励模型。然后我们选择具有高奖励的策略回答（best-of-...

---

### Eleuther ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/1211704006183751741) (6 messages): 

- **无上下文分享的 Twitter 链接**：`@main.ai` 分享了一个 [Twitter 链接](https://twitter.com/FazlBarez/status/1762092405048959419)，未提供任何进一步解释或上下文。
- **寻求对能量结果的澄清**：`@butanium` 向 `<@177739383070261248>` 询问了关于使用 Tuned Lens 对“能量结果”（energy results）的解释，但未得到详细回复。
- **周末计划审阅论文**：`@mrgonao` 提到他们将在周末花更多时间来理解与之前关于能量结果问题相关的论文。
- **对研究中“能量”一词的困惑**：`@mrgonao` 承认不理解为什么论文的方程式中使用了“能量”（energy）一词，表示对其所指代的内容缺乏直观理解。
- **承诺进一步调查**：`@butanium` 承诺第二天将进一步调查此事。

---

### Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1211657627453489192) (50 条消息🔥): 

- **对 lm-harness 的赞誉**：`@hailey_schoelkopf` 对在一篇论文中看到 EleutherAI 的 lm-harness 被引用表示欣喜，并赞赏其在自回归语言模型 few-shot 评估中发挥的关键作用。
- **复现 Leaderboard 结果**：`@hailey_schoelkopf` 提供了一份详细指南，介绍如何使用 lm-evaluation-harness 的特定 commit 来复现 Open LLM Leaderboard 的结果，包括针对不同任务和模型的配置，该内容已被置顶至常见问题解答 (FAQ)。
- **改进 lm-eval-harness API 的兴趣**：`@ariel2137` 询问有关增强 lm-eval-harness 代码级用法的问题，`@hailey_schoelkopf` 给予了积极回应，并邀请反馈以优化使用体验中的不足之处。
- **澄清 lm-eval-harness 输出解释**：`@micpie` 寻求理解 lm-eval-harness 的输出格式，`@hailey_schoelkopf` 澄清说 `true` 和 `false` 值表示目标字符串是否为每个答案选项的贪婪补全 (greedy completion)。
- **lm-eval-harness 中数据切分的选择与覆盖**：`@micpie` 对 harness 如何选择和计数评估的数据切分 (data splits) 感到困惑。`@baber_` 协助解释说报告的数量也考虑了多项选择选项，`@hailey_schoelkopf` 确认目前无法通过命令行覆盖切分设置，建议通过编辑 yaml 文件来实现。
  

---


### Eleuther ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/1211678562692763728) (8 条消息🔥): 

- **DeepSpeed 配置困惑**：用户 `@jdranpariya` 在使用 `deppy.py train.py` 时，尝试在配置中通过 `"deepspeed": false` 禁用 DeepSpeed，结果遇到了 `ValueError`。错误提示涉及 NeoXArgs 校验问题。
- **Mistral Tokenizer 的最佳数据**：`@rand0mm` 询问扩展 Mistral 分词器 (tokenizer) 以更好地支持其他语言的最优数据源。
- **寻找多节点训练设置**：`@jdranpariya` 寻求在 CoreWeave 基础设施上为 GPT-NeoX 设置多节点训练环境的指导，具体目标是使用 slurm 或 MPI 在 2 个节点上运行 4 个 GPU。
- **CoreWeave 必须使用 Kubernetes 吗？**：针对在 CoreWeave 上进行多节点训练是否可以绕过 Kubernetes 的问题，`@jdranpariya` 表示愿意接受建议，并正在寻找正确的平台或频道以获取更专业的帮助。
- **CoreWeave 和 Slurm 集群指导**：`@catboy_slim_` 建议 `@jdranpariya` 参考 CoreWeave 了解特定的基础设施问题，并参考 NeoX 文档了解使用 slurm 启动的说明，同时提到设置 slurm 集群属于 CoreWeave 的范畴。
  

---



### CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1211701811711442944) (6 条消息): 

- **潜在的 GTC 线下聚会筹备**：`@vim410` 期待能亲身参加 **GTC**，并提议为参会者创建一个专用频道。
- **Discord 中的 LaTeX**：`@marksaroufim` 通过示例 `x^2` 展示了如何在 Discord 中格式化 LaTeX 代码。
- **对即将到来的 GTC 演讲感到兴奋**：`@joseph_en` 热衷于参加 GTC 周四的演讲，并期待与 **cuda_mode** 的成员见面。
- **关于 CUDA 模拟器的咨询**：`@jash403` 向社区征求关于创建或在 **CUDA GPU 上运行模拟器** 的经验。
- **CUDA 版 Gameboy 模拟器链接**：针对 `@jash403` 的问题，`@iron_bound` 分享了一个 **CUDA Gameboy 模拟器** 的 [GitHub 仓库](https://github.com/krocki/nvgb) 以及一篇详述该项目的 [相关文章](https://towardsdatascience.com/a-gameboy-supercomputer-33a6955a79a4)。

**提到的链接**：

- [GitHub - krocki/nvgb: CUDA gameboy](https://github.com/krocki/nvgb)：CUDA 版 gameboy。欢迎在 GitHub 上为 krocki/nvgb 的开发做出贡献。
- [A GAMEBOY supercomputer](https://towardsdatascience.com/a-gameboy-supercomputer-33a6955a79a4)：总帧率略高于每秒 10 亿帧，这可以说是世界上最快的 8 位游戏机集群。

  

---

### CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1211943113703166002) (2 条消息): 

- **向 Unsloth 的 QLoRA Kernels 致敬**：`@andreaskoepf` 强调了 Unsloth kernels 的效率，并提到了他们的 GitHub 仓库，该仓库可实现 **5 倍速且减少 60% 显存占用的 QLoRA 微调**，链接见 [GitHub - unslothai/unsloth](https://github.com/unslothai/unsloth)。链接预览展示了该仓库的描述、图片和标题。
- **自定义 Triton Kernel 集成指南链接**：`@marksaroufim` 从另一个频道转发了一种将自定义 triton kernels 与 `torch.compile` 集成的方法。不过，该消息链接不包含预览。

**提到的链接**：

[GitHub - unslothai/unsloth: 5X faster 60% less memory QLoRA finetuning](https://github.com/unslothai/unsloth)：5 倍速且减少 60% 显存占用的 QLoRA 微调。通过在 GitHub 上创建账号来为 unslothai/unsloth 的开发做出贡献。

  

---


### CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1211627058824876042) (15 条消息🔥): 

- **CUDA 与图形 API**：`@morousg` 强调 NVIDIA 相比 CUDA-OpenGL 更关注 **CUDA-Vulkan** 的互操作性，并提到了 Vulkan 相比 OpenGL 的效率优势，特别是在图形密集型应用中。

- **库测试规格公布**：`@zippika` 分享了他们的系统配置，包括 **NVIDIA 4090 GPU**、CUDA 12.3、pytorch 2.2+、pytorch-cuda=12.1 以及 **AMD 7950x CPU**，同时回答了关于其库改进的查询。

- **GPU 利用率洞察**：`@zippika` 观察到 **GPU 利用率**显著提高 —— 从使用 bnb 层时的 60% 增加到使用 TorchFP4Linear 层时的约 80%。

- **PyTorch 模型层替换代码**：`@zippika` 更新了他们的库，引入了 `only_replace_bnb_layers` 和 `ignore_layer_names` 等选项用于选择性层替换，并提供了一个 [Huggingface 示例脚本](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py) 用于实现。

- **探索 GPU 内存延迟**：`@marksaroufim` 分享了一篇使用 PTX 研究 NVIDIA A100 内存访问延迟的论文。随后的讨论涉及 `@zippika`、`@jeremyhoward` 和 `@cudawarped`，触及了 L2 cache 与 global memory 延迟的比较，以及 L2 cache 更高的带宽可能使其与 global memory 类似的延迟变得合理的观点。

**提到的链接**：

- [Nvidia’s H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100-funny-l2-and-tons-of-bandwidth/)：GPU 最初是纯粹用于图形渲染的设备，但其高度并行的特性使其在某些计算任务中也极具吸引力。随着过去几年 GPU 计算领域的增长……
- [Is memory operation for L2 cache significantly faster than global memory for NVIDIA GPU?](https://stackoverflow.com/questions/66921433/is-memory-operation-for-l2-cache-significantly-faster-than-global-memory-for-nvi)：现代 GPU 架构同时具有 L1 cache 和 L2 cache。众所周知，L1 cache 比 global memory 快得多。然而，L2 cache 的速度在 CUDA 文档中并不那么明确。
- [Demystifying the Nvidia Ampere Architecture through Microbenchmarking and Instruction-level Analysis](https://arxiv.org/abs/2208.11174)：图形处理器 (GPU) 现在被认为是加速 AI、数据分析和 HPC 等通用工作负载的领先硬件。在过去的十年中，研究人员一直专注于……
- [Nvidia’s H100: Funny L2, and Tons of Bandwidth](https://chipsandcheese.com/2023/07/02/nvidias-h100)：GPU 最初是纯粹用于图形渲染的设备，但其高度并行的特性使其在某些计算任务中也极具吸引力。随着过去几年 GPU 计算领域的增长……
- [torch-bnb-fp4/examples/speed_test_mistral_7b.py at main · aredden/torch-bnb-fp4](https://github.com/aredden/torch-bnb-fp4/blob/main/examples/speed_test_mistral_7b.py)：更快的 Pytorch bitsandbytes 4bit fp4 nn.Linear 算子 - aredden/torch-bnb-fp4

  

---

### CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1211605910464495627) (17 messages🔥): 

- **寻求更快的编译速度**：`@briggers` 分享了关于如何通过避免不必要的头文件和分离源文件，将 `cpp_extension.load_inline` 的编译时间从 >30s 缩短到 <2s 的见解。其核心在于选择使用 `cpp_extension.load` 并有针对性地指定设备架构，特别是针对他的 NVIDIA 4090 ([更快的 cpp_extension 编译代码示例](https://github.com/pbridger/cuda-experiments))。

- **从 PyTorch 到 Torch 的起源**：针对关于 *Torch* 与 *PyTorch* 之间关系的提问，`@marksaroufim` 链接了一篇详细介绍 PyTorch 历史的博客文章。PyTorch 起源于 Torch7，其根源可以追溯到 2010 年的 Torch7 贡献者社区 ([PyTorch 设计起源](https://soumith.ch/posts/2023/12/pytorch-design-origins/))。

- **在 Torch 中使用自定义 Triton Kernels**：`@marksaroufim` 为正在研究自定义 Triton kernels 与 `torch.compile` 集成的开发者指引了一个 PyTorch GitHub 示例。可以通过在 GitHub 上提交 issue 并标记 `@oulgen` 来报告问题，旨在改进即将发布的 PyTorch 2.3 中该功能的正式版本 ([Triton kernels 与 torch.compile 的集成](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661))。

- **解读编译器的局限性**：在回答关于编译器生成 FA (FlashAttention) 的问题时，`@marksaroufim` 分享了 Tri Dao 在 OpenReview 论坛上的见解，提到编译器往往难以处理那些在保持数值稳定性的同时需要进行数学重写的优化 ([Tri Dao 对编译器生成 FA 的评论](https://openreview.net/forum?id=mZn2Xyh9Ec))。

- **Andreas Köpf 的加入引发热议**：PyTorch 开发的重要贡献者 Andreas Köpf 的出现受到了社区 `@andreaskoepf` 的热烈欢迎，这标志着人们对他参与社区以及潜在知识分享的热情。

**提到的链接**：

- [PyTorch's design origins | Soumith Chintala](https://soumith.ch/posts/2023/12/pytorch-design-origins/)：未找到描述
- [FlashAttention-2: Faster Attention with Better Parallelism and Work...](https://openreview.net/forum?id=mZn2Xyh9Ec)：在过去几年中，将 Transformer 扩展到更长的序列长度一直是一个主要问题，这有望提高语言建模和高分辨率图像理解的性能...
- [GitHub - pbridger/cuda-experiments](https://github.com/pbridger/cuda-experiments)：通过在 GitHub 上创建账号来为 pbridger/cuda-experiments 的开发做出贡献。
- [pytorch/test/dynamo/test_triton_kernels.py at 0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce · pytorch/pytorch](https://github.com/pytorch/pytorch/blob/0c8bb6f70c65b0a68fcb282cc1605c79ca5dabce/test/dynamo/test_triton_kernels.py#L628-L661)：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - pytorch/pytorch
- [Issues · pytorch/pytorch](https://github.com/pytorch/pytorch/issues?q=is%3Aissue+is%3Aopen+sort%3Aupdated-desc)：Python 中具有强 GPU 加速功能的 Tensor 和动态神经网络 - Issues · pytorch/pytorch
- [oulgen - Overview](https://github.com/oulgen)：我是 Meta 的一名软件工程师，从事 Hack 编程语言和 PyTorch 的工作。- oulgen

---

### CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1211893283790258209) (7 条消息): 

- **改进 Softmax 性能的见解**：`@marksaroufim` 分享了一篇论文链接（[Efficient Softmax Approximation on GPUs](https://arxiv.org/abs/1805.02867)），该论文阐明了 Flash Attention 中的 Softmax 技巧，重点介绍了一种特定的局部修正技术（`e ^ {m_{j-1} - m_j}`），该技术可以维持全局 Softmax 操作。
- **探索 Base2 技巧**：`@andreaskoepf` 指出了一个 GitHub 链接（[softmax_base2_trick.ipynb](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb)），解释了普通 Softmax 的 Base2 技巧，该技巧也适用于 OpenAI 的 Triton 示例（[06-fused-attention.py](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)）中实现的增量 Softmax。
- **Triton 中的反向传播自动化**：`@marksaroufim` 提到在 Triton 中编写反向传播（backward passes）非常笨拙，暗示希望能够从给定的前向传播（forward pass）自动生成反向传播。
- **通过编译器优化获得微小性能提升**：针对 `@marksaroufim` 表达的编译器应该处理某些优化的观点，`@andreaskoepf` 表示赞同，并指出虽然编译器可以优化某些方面，但在检查 Kernel 代码时，了解某些技巧（如 Triton 示例中的技巧）是有益的。
- **讨论中回想起 Quake3 算法**：`@iron_bound` 提到了视频游戏 Quake3 中使用的历史性的快速平方根倒数（fast inverse square root）技巧，并分享了维基百科的概述（[Fast inverse square root](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Overview_of_the_code)），强调了此类优化在计算算法中的巧妙之处。

**提到的链接**：

- [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)：Softmax 函数在机器学习中无处不在，之前的多项工作提出了更快的替代方案。在本文中，我们提出了一种以更少的内存访问计算经典 Softmax 的方法...
- [Fast inverse square root - Wikipedia](https://en.wikipedia.org/wiki/Fast_inverse_square_root#Overview_of_the_code)：未找到描述
- [triton/python/tutorials/06-fused-attention.py at main · openai/triton](https://github.com/openai/triton/blob/main/python/tutorials/06-fused-attention.py)：Triton 语言和编译器的开发仓库 - openai/triton
- [ring-attention/trition_flash_attn/softmax_base2_trick.ipynb at main · cuda-mode/ring-attention](https://github.com/cuda-mode/ring-attention/blob/main/trition_flash_attn/softmax_base2_trick.ipynb)：ring-attention 实验。通过在 GitHub 上创建一个账户来为 cuda-mode/ring-attention 的开发做出贡献。

  

---


### CUDA MODE ▷ #[jobs](https://discord.com/channels/1189498204333543425/1190208177829068860/1211708399956262922) (2 条消息): 

- **NVIDIA 寻找 CUDA 和 C++ 大神**：`@vim410` 确认 **Nvidia** 确实在寻找 **CUDA** 和 **C++** 方面的专家。感兴趣的候选人应*私信简历*并注明 **JobID: JR1968004**。
  

---


### CUDA MODE ▷ #[beginner](https://discord.com/channels/1189498204333543425/1191300313928433664/1211594632077778954) (6 条消息): 

- **CUDA Mode 介绍**：用户 `@ilovepython3` 表达了对学习 CUDA Mode 的兴趣，并提到自己有 Python 和 C 语言背景，渴望微调 AI 模型。
- **担心数学问题的 Python 学习者**：`@ilovepython3` 承认自己虽然能编写简单的程序，但在数学方面比较吃力。
- **寻求 CUDA Mode 的先决条件**：`@ilovepython3` 询问参与 CUDA Mode 是否需要任何先决条件，例如 PyTorch 知识。
- **带着不确定性踏入 AI 领域**：`@ilovepython3` 表达了从事 AI 相关工作的愿望，但不确定现阶段深入研究 CUDA Mode 是否过于复杂。
- **为 AI 初学者提供指导**：`@jeremyhoward` 建议 `@ilovepython3` 在挑战 CUDA Mode 之前应先完成 fast.ai 课程，以便更轻松地掌握这些材料。
  

---


### CUDA MODE ▷ #[smol-hw](https://discord.com/channels/1189498204333543425/1205223658021458100/1212061416496824430) (2 条消息): 

- **澄清 AO 缩写**：用户 `@mr.osophy` 询问了缩写 **AO** 的含义，`@marksaroufim` 澄清它代表 **Architecture Optimization**（架构优化），尽管承认这不是最好的名字。
  

---

### CUDA MODE ▷ #[ring-attention](https://discord.com/channels/1189498204333543425/1208496482005549086/1211622790659182622) (13 messages🔥): 

- **CUDA 中的同步策略**：`@zhuzilin96` 建议了一种更直接的方法来记录总通信时间，即在通信开始后添加 `for req in reqs: req.wait()`，从而实现与计算 kernel 的重叠（overlap）。
- **Benchmark 调整提升速度**：根据 `@zhuzilin96` 的说法，通过调整 benchmark 中的 `seqlen`，zigzag ring 的性能提升到了 flash_attn 的 8.5 倍左右。
- **Flash Attention 会议即将召开**：`@jamesmel` 询问了关于 flash attention 论文阅读/讨论的事宜，`@iron_bound` 确认会议安排在 `<t:1708966800>`。
- **推荐 Flash Attention 论文**：`@w0rlord` 提供了一个 [arXiv 论文链接](https://arxiv.org/abs/2312.11918)，详细介绍了在 NVIDIA Hopper 架构上优化实现 FlashAttention-2 的细节，并因其详尽的注释和算法细节而推荐该文。
- **Ring Attention 与 Flash Attention 协作 Notebook**：`@ericauld` 分享了一个[正在进行中的 Colab notebook](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X)，内容涉及 ring attention 和 flash attention，并邀请社区提供反馈和改进建议。

**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1FMzg1vf2QEg5Q9fjjQV3Q5Ep-r2x9l-t#scrollTo=X08l8A7tdu-X)：未找到描述
- [A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library](https://arxiv.org/abs/2312.11918)：我们提供了一个针对 NVIDIA Hopper 架构的自定义融合 CUDA kernel，实现了 FlashAttention-2（一种流行的内存感知缩放点积注意力算法）的前向传播优化版本...

  

---



### LangChain AI ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/1211802924959793193) (1 messages): 

- **征求结构化输出接口的反馈**：用户 `@bagatur` 鼓励社区对提议的、用于从模型获取结构化输出的直观接口提供反馈。他们分享了 [GitHub 上的讨论](https://github.com/langchain-ai/langchain/discussions/18154)，该讨论概述了这一想法，旨在简化大多数 LLM 任务的用户体验。

**提到的链接**：

[RFC: LLM structured output interface · langchain-ai/langchain · Discussion #18154](https://github.com/langchain-ai/langchain/discussions/18154)：从模型获取结构化输出对于大多数 LLM 任务至关重要。我们需要使从模型获取结构化输出的 UX 尽可能简单。我们目前的想法是增加一个 ChatM...

  

---

### LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1211657893028564992) (37 messages🔥): 

- **介绍用于内容爬取的 UseScraper**：`@dctanner` 推出了 [UseScraper.com](https://usescraper.com)，它可以将网站内容抓取为 Markdown 或 JSON，并分享了一篇关于将其与 LangChain 结合用于 RAG 的[博客文章](https://usescraper.com/blog/langchain-chatgpt-rag-with-your-website-content)。
- **LlamaCpp Python 集成指南**：`@ldeth256` 引用了一份关于在 LangChain 中集成 `llama-cpp-python` 以使用各种 LLM 进行推理的[指南](https://python.langchain.com/docs/integrations/llms/llamacpp)。
- **关于高效 Logo 创建平台的讨论**：`@bru.leo` 寻求推荐能创建专业外观 Logo 的 AI 平台，并对目前尝试过的工具表示不满。
- **寻求流式问答链（Streaming Q&A Chains）的帮助**：`@shadizx` 参考了[这份流式问答指南](https://js.langchain.com/docs/use_cases/question_answering/streaming#chain-with-sources)，但在上下文中的对象提取方面需要帮助。
- **请求对 Ragas 和 Validate.ai 集成的反馈**：`@locus_5436` 分享了一个个人项目，该项目将 `Ragas` 集成到一个用于评估 RAG 系统的可视化 Web 应用中，可在 [validate.tonic.ai](https://validate.tonic.ai/) 获取并提供反馈。


**提到的链接**：

- [[beta] Structured Output | 🦜️🔗 Langchain](https://python.langchain.com/docs/guides/structured_output)：让 LLM 返回结构化输出通常至关重要。
- [Tonic Validate](https://validate.tonic.ai/.)：未找到描述
- [Dall-E Image Generator | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/tools/dalle_image_generator)：OpenAI Dall-E 是文本生成图像模型
- [Llama.cpp | 🦜️🔗 Langchain](https://python.langchain.com/docs/integrations/llms/llamacpp)：llama-cpp-python 是一个...
- [Streaming | 🦜️🔗 Langchain](https://js.langchain.com/docs/use_cases/question_answering/streaming#chain-with-sources)：在问答应用中，向用户展示来源通常很重要...
- [LangChain ChatGPT RAG with your website content - UseScraper](https://usescraper.com/blog/langchain-chatgpt-rag-with-your-website-content)：极速网页爬取和抓取 API。按需付费。支持浏览器渲染、Markdown 输出等。
- [Quickstart | 🦜️🔗 Langchain](https://js.langchain.com/docs/get_started/quickstart#building-with-langchain>)：在此快速入门中，我们将向您展示如何：
- [Add chat history | 🦜️🔗 Langchain](https://python.langchain.com/docs/use_cases/question_answering/chat_history#langsmith>)：在许多问答应用中，我们希望允许用户拥有...

  

---


### LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (1 messages): 

howtonotgiveafuck: 大家好，有什么办法可以将超时时间延长到 900 秒以上吗？
  

---


### LangChain AI ▷ #[langchain-templates](https://discord.com/channels/1038097195422978059/1170025009960456282/1211870848143859712) (1 messages): 

- **关于开源聊天模型中 Function Calling 的咨询**：`@sectorix` 正在寻求在聊天补全（chat completions）中实现 Function Calling 的可行方案，特别是针对像 **Mistral** 这样的开源模型。他们提到预期 **Ollama** 会添加此功能，但正在寻找过渡方案。

### LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1211772331442049085) (5 条消息): 

- **用于 RAG 系统的 R2R 框架发布**：`@emrgnt_cmplxty` 宣布推出 **R2R**，这是一个专为快速开发和部署生产级 **RAG 系统**而设计的框架。R2R 提供了一个半意见化（semi-opinionated）的系统，简化了在生产环境中部署、适配和维护 RAG 流水线的过程。详情可以在其 [GitHub 页面](https://github.com/SciPhi-AI/R2R)上找到。

- **使用 Tonic.AI 可视化 LLM RAG 结果**：`@locus_5436` 分享了一个个人项目 —— 一个集成了 **Ragas** 的免费可视化 Web 应用，旨在辅助评估 **LLM RAG 系统**。欢迎对该应用提出反馈，访问地址为 [validate.tonic.ai](https://validate.tonic.ai/)。

- **IntelliDoctor.ai 为医疗专业人士发布**：`@robertoshimizu` 宣布推出 **IntelliDoctor.ai**，这是一个面向医疗专业人士的 AI 驱动平台，利用先进的 Prompt Engineering 和 **RAG** 提供基于证据的临床解答。该平台利用开放学术资源获取见解，并感谢 **LangChain** 和 **LangSmith** 提供的基础设施支持和启发。访问 [intellidoctor.ai](https://intellidoctor.ai) 了解更多详情。

- **LangGraph 提升代码生成能力**：`@andysingal` 介绍了 **LangGraph**，这是一款通过结合 LangGraph 的迭代代码生成与纠错能力，以及 LangChain 的安全性和完整性特征来增强代码生成的工具，详见 [AI Advances](https://ai.gopubby.com/empowering-code-generation-unlocking-potential-with-langgraph-742dc71a806b) 上的完整博客文章。

- **频道用途提醒**：`@theepic.dev` 提醒用户，“share-your-work” 频道不用于技术支持咨询。

**提到的链接**：

- [no title found](https://intellidoctor.ai),): 未找到描述
- [Tonic Validate](https://validate.tonic.ai/.): 未找到描述
- [Empowering Code Generation: Unlocking Potential with LangGraph](https://ai.gopubby.com/empowering-code-generation-unlocking-potential-with-langgraph-742dc71a806b): Ankush k Singal
- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): 一个用于快速开发和部署生产级 RAG 系统的框架 - SciPhi-AI/R2R

  

---


### LangChain AI ▷ #[tutorials](https://discord.com/channels/1038097195422978059/1077843317657706538/1211746297988513863) (2 条消息): 

- **LangGraph 为聊天机器人赋能**：`@tarikkaoutar` 分享了一种使用 LangGraph、Function Calling 和网络爬虫创建多 Agent 应用的新方法，旨在面向 Python 和数据科学社区。在 [YouTube](https://www.youtube.com/watch?v=q5LvDHiSBy4) 上观看题为 "LangGraph + Function Call + Web Scraper = Multi-Agent Application" 的完整讲解视频。
 
- **AI 作为你的对话助手**：`@jasonzhou1993` 介绍了一个新颖的概念：手机端的实时对话 AI 副驾驶（Co-pilot）。通过 [YouTube 视频](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg) "Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?" 探索这项技术。

**提到的链接**：

- [Real time AI Conversation Co-pilot on your phone, Crazy or Creepy?](https://youtu.be/vgY5gNEOAZ0?si=TKGm5GpU7lQH0aJg): 我在 iPhone 上构建了一个对话 AI 副驾驶，它可以监听你的对话并提供实时建议。在 Replicate 上免费访问 Whisper 和 Mixtral 模型...
- [LangGraph + Function Call + Web Scraper = Multi-Agent Application](https://www.youtube.com/watch?v=q5LvDHiSBy4): #chatbot #langgraph #functioncall #ai #automation #dropshipping 在这段视频中，我将解释如何创建 LangGraph、进行 Function Calling 并开发...

  

---

### OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1211699180494389298) (32 条消息🔥): 

```html
<ul>
  <li><strong>Mistral 与欧盟的合作引发关注</strong>：`@yamashi` 对 Mistral 对开源的承诺表示怀疑，认为他们与 Microsoft 的合作伙伴关系证实了其对利润的追求。与此同时，`@casper_ai` 分享了一个<a href="https://fxtwitter.com/casper_hansen_/status/1762159643344662859">链接</a>，表明 MistralAI 的 CEO 承诺会继续推出 open-weight models。</li>
  <li><strong>战略泄露与现实相符</strong>：`@casper_ai` 承认 Mistral 发布较小模型而将较大模型保留在特定平台（platform-gated）的策略与之前泄露的计划一致。</li>
  <li><strong>对 Llama 3 的期待升温</strong>：`@yamashi` 和 `@noobmaster29` 都对 Llama 3 充满期待，希望看到除了单纯增加数据规模之外的创新，并期待潜在的多语言改进以及像 MoE Mamba 这样的增强功能。</li>
  <li><strong>讨论 LoRA 的局限性</strong>：`@enka55` 寻求关于使用 LoRA 进行知识集成的信息，对此 `@nruaif` 和 `@leoandlibe` 回复称，full fine-tuning 而非 LoRA 更适合添加知识。此外，`@lee0099` 分享了一篇研究 LoRA 在知识迁移（knowledge transfer）方面潜力的<a href="https://arxiv.org/pdf/2304.08109.pdf">研究论文</a>。</li>
  <li><strong>硬件限制影响对模型实用性的认知</strong>：`@nafnlaus00` 分享了关于模型可访问性的务实观点，指出由于硬件限制，普通用户运行超大型模型是不切实际的。</li>
</ul>
```

**提到的链接**：

- [来自 Casper Hansen (@casper_hansen_) 的推文](https://fxtwitter.com/casper_hansen_/status/1762159643344662859)：据其 CEO 称，@MistralAI 致力于 open-weight models —— 依然看好 *“商业活动将使我们能够资助模型开发所需的高昂研究费用。我们将...”
- [Microsoft 与 Mistral 达成协议，推动超越 OpenAI](https://www.ft.com/content/cd6eb51a-3276-450f-87fd-97e8410db9eb)：未找到描述
- [寻找神经金块：从参数视角看大语言模型中的知识迁移](https://arxiv.org/abs/2310.11451)：Large Language Models (LLMs) 通过在海量语料库上的预训练，在其参数中固有地编码了丰富的知识。虽然之前的研究已经深入探讨了对这些参数的操作...

---

### OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1211721989979643955) (6 条消息): 

- **Gemma 优化达到新高度**：`@nanobitz` 宣布了在 Unsloth 中运行 [Gemma 模型](https://huggingface.co/unsloth) 的显著改进；**Gemma** 比使用 Hugging Face 和 FA2 快 **2.43 倍**，比原生 Hugging Face 快 **2.53 倍**，且 **VRAM 占用减少了 70%**。他们还提供了 [Gemma 7b Notebook](https://colab.research.google.com/drive/10NbwlsRChbma1v55m8LAPYG15uQv6HLo?usp=sharing) 和 [Gemma 2b Notebook](https://colab.research.google.com/drive/15gGm7x_jTm017_Ic8e317tdIpDG53Mtu?usp=sharing) 的链接，可在 Tesla T4 上免费使用。

- **在 Pull Request 中寻求文档链接**：由于 [GitHub pull request 讨论](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256)，`@caseus_` 询问了文档引用，`@yamashi` 随后回复了 [相关文档](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md)。

- **模型改进中质量优于速度**：`@dreamgen` 强调了模型准确性比速度更重要，建议应优先确保模型的正确性。

- **引入 LoRA-the-Explorer 以实现高效训练**：`@caseus_` 分享了 [Minyoung Huh 关于 **LoRA-the-Explorer (LTE)** 的研究](https://minyoungg.github.io/LTE/)，这是一种通过并行低秩适配器（low-rank adapters）从头开始训练神经网络的方法，可能对资源高效的深度学习产生重大影响。他们还强调了 [multi-head LoRA 实现](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py) 值得关注。

**提到的链接**：

- [LTE](https://minyoungg.github.io/LTE/): 未找到描述
- [axolotl/docs/mac.md at 13199f678b9aab39e92961323bdbce3234ee4b2b · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md): 尽管提问（axolotl questions）。通过在 GitHub 上创建账号为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。
- [LTE/lte/mhlora/linear.py at main · minyoungg/LTE](https://github.com/minyoungg/LTE/blob/main/lte/mhlora/linear.py): 通过在 GitHub 上创建账号为 minyoungg/LTE 的开发做出贡献。
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/s/NCqFshpmqs): 未找到描述
- [Mps mistral lora by maximegmd · Pull Request #1292 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/pull/1292#discussion_r1493791256): 用于训练 Mistral LoRA 的额外 MPS 示例。包含一些关于用法和限制的文档。

  

---


### OpenAccess AI Collective (axolotl) ▷ #[community-showcase](https://discord.com/channels/1104757954588196865/1117851527143493664/1211772507066204250) (1 条消息): 

- **R2R：让 RAG 部署变得简单**：用户 `emrgnt_cmplxty` 宣布推出 **R2R**，这是一个用于快速开发和部署生产级 **RAG 系统** 的半意见型（semi-opinionated）框架。旨在简化和优化部署流程，这是行业向实际易用性和有效性迈进的一步，可以在 [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) 找到。

**提到的链接**：

[GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): 一个用于快速开发和部署生产级 RAG 系统 的框架 - SciPhi-AI/R2R

  

---


### OpenAccess AI Collective (axolotl) ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/1212026280287936553) (1 条消息): 

- **Replicate 的专注度受到质疑**：`@dreamgen` 表示惊讶，指出考虑到 **replicate** 多年来对此的专注，本预期它会有更好的表现，但显然它没有达到这些预期。没有关于具体问题或对比的进一步详细说明。
  

---

### Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1211598894258065430) (33 messages🔥): 

- **关于 AI 模型比较的澄清**：`@eugeneyan` 在回复 `@guardiang` 时指出，那篇将模型与 GPT-4 进行比较的推文线程是基于 **zero-shot** 性能指标的。
- **Mistral Large 发布并与 Microsoft 达成合作伙伴关系**：`@__chef__` 分享了 [Mistral Large 的发布公告](https://mistral.ai/news/mistral-large/)链接，详细介绍了其新功能和 Benchmark 表现，以及与 Microsoft 建立合作伙伴关系的消息。
- **Cloudflare 的 AI Gateway 引起关注**：`@henriqueln7` 展示了 Cloudflare 的 AI Gateway，强调了其分析、日志记录和缓存功能，并突出其易用性——只需一行代码即可开始使用。
- **Mistral Au Large 对 RAG 至关重要**：`@ashpreetbedi` 评测了 Mistral Au Large 的 RAG 集成，赞扬了其 function calling 和推理能力。他们还在 GitHub 上分享了他们的 cookbook：[phidata/mistral](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral)。
- **发布关于 RAG 的电子书**：`@dimfeld` 在聊天中通知了 Jason Liu 即将出版的关于不同复杂程度 RAG 的电子书，GitHub 仓库地址为：[n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag)。`@thenoahhein` 认为这是处理 Twitter 数据摘要任务的有用资源。

**提到的链接**：

- [Font Pairing Generator](https://www.monotype.com/font-pairing)：由 Monotype AI 字体配对生成引擎驱动，寻找完美的字体搭配。
- [Au Large](https://mistral.ai/news/mistral-large/)：Mistral Large 是我们的旗舰模型，具有顶级的推理能力。它也可以在 Azure 上使用。
- [AI Gateway · Cloudflare AI Gateway docs](https://developers.cloudflare.com/ai-gateway/)：Cloudflare 的 AI Gateway 允许您获得对 AI 应用的可视化和控制。通过将应用连接到 AI Gateway，您可以收集关于……的洞察。
- [phidata/cookbook/mistral at main · phidatahq/phidata](https://github.com/phidatahq/phidata/tree/main/cookbook/mistral)：使用 function calling 构建 AI Assistants。通过在 GitHub 上创建账号为 phidatahq/phidata 的开发做出贡献。
- [GitHub - jxnl/n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag)：通过在 GitHub 上创建账号为 jxnl/n-levels-of-rag 的开发做出贡献。

  

---



### Datasette - LLM (@SimonW) ▷ #[ai](https://discord.com/channels/823971286308356157/1097032579812687943/1211734565870370876) (4 messages): 

- **数据提取中的语言意外**：`@derekpwillis` 遇到了一个有趣的现象，**chatgpt-3.5-turbo** 对本应是英文的文档使用了西班牙语标题，例如在提取版本中将 "Taking Advantage of the Internet" 翻译成了 *"Sacándole Provecho a Internet"*。

- **类似于 ChatGPT/Whisper 语音 Bug 的多语言混淆**：`@simonw` 将此问题与一个类似的 Bug 联系起来，即 ChatGPT 结合 Whisper 有时会将 **英国口音误认为威尔士语** 并用威尔士语回答。

- **建议通过 System Prompt 修复**：`@simonw` 建议了一个潜在的修复方案，即向系统输入一个指令为 "Always use English"（始终使用英语）的 Prompt，以避免语言混淆。

- **决定实施特定语言的 Prompt**：针对该建议，`@derekpwillis` 确认了该问题，并表示他将实施“始终使用英语”的建议。
  

---

### Datasette - LLM (@SimonW) ▷ #[llm](https://discord.com/channels/823971286308356157/1128504153841336370/1211632326996529162) (26 条消息🔥): 

- **LLM 插件为 Python 开发者带来 Groqcloud 支持**：`@angerman.` 分享了用于访问 [Groqcloud](https://console.groq.com) 模型的 [LLM 插件](https://pypi.org/project/llm-groq/) 现已发布，并提供了安装说明以及如何获取 API key。该插件允许使用 `groq-llama2` 和 `groq-mixtral` 等模型，并给出了一个生成宠物名字的示例。
- **面向初学者的 Python 打包教程**：`@0xgrrr` 为 `@angerman.` 提供了一份指导性 [教程](https://packaging.python.org/en/latest/tutorials/packaging-projects)，介绍如何打包并将 Python 项目上传到 PyPI，并鼓励他们该过程非常简单。
- **LLM 插件通过流式传输支持得到改进**：`@angerman.` 的更新提到了 LLM 插件新增的流式传输（streaming）功能，并对 `@746595581086138409` 为此增强功能所做的贡献表示感谢。
- **Datasette 开发者暗示即将推出 Chat UI**：`@simonw` 透露正在为 LLM 开发聊天界面（Chat UI），尽管目前仍在进行中，尚未确定完成日期。
- **Datasette 的选择：为什么使用 Fly GPU？**：`@simonw` 向 `@kiloton9999` 解释说，选择使用 Fly GPU 的部分原因是该公司赞助了 Datasette 的开发，同时也因为他们的 GPU 具有缩减至零（scale to zero）的能力。

**相关链接**:

- [llm-groq](https://pypi.org/project/llm-groq/): 未发现描述
- [Packaging Python Projects - Python Packaging User Guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/#uploading-the-distribution-archives): 未发现描述

  

---



### DiscoResearch ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/1211592072553959424) (9 条消息🔥): 

- **关于 LLM 评估方法的见解**：`@bjoernp` 分享了一篇 [有趣的论文](https://arxiv.org/abs/2402.13887)，讨论了当前基于概率的大语言模型（LLM）评估方法的局限性，这在 DiscoLM 系列中有所提及。该论文批评了这种评估策略通常与基于生成的预测不一致，但并未充分探讨产生这些差异的原因。

- **JSON 转 Parquet 过程中的隐藏空字符串**：`@thomasrenkert` 在将 JSON 文件转换为 Parquet 格式时遇到了不可见的空字符串（null strings），这个问题只有在直接将 JSON 上传到 Hugging Face 并观察转换后的数据集时才能发现。

- **为代码库构建 RAG**：`@codermickey` 询问了关于创建用于回答问题和辅助代码库的检索增强生成（RAG）机器人的技术现状，包括索引的分块策略以及适用于编码任务的检索策略。

- **LangChain 的代码库 RAG 方法**：针对 `@codermickey` 的提问，`@johannhartmann` 提到 LangChain 提供了 Git 加载器和针对流行语言的分段功能，而 LlamaIndex 也有 Git 导入器；大多数人只是简单地使用 OpenAI embeddings 进行检索，并结合针对代码相关任务的 prompts。

- **RAG 与 LLM 端到端优化探索**：`@rasdani` 询问是否有关于使用梯度对 RAG 和大语言模型进行联合端到端优化的研究，并链接了一篇相关论文 [LESS](https://arxiv.org/abs/2402.04333)，尽管随后指出该论文并未通过数据选择进行反向传播，而只是检索具有相似预计算梯度特征的训练示例。

**相关链接**:

- [Beyond Probabilities: Unveiling the Misalignment in Evaluating Large Language Models](https://arxiv.org/abs/2402.13887): 大语言模型（LLM）在各种应用中展示了卓越的能力，从根本上重塑了自然语言处理（NLP）研究的格局。然而，最近...
- [LESS: Selecting Influential Data for Targeted Instruction Tuning](https://arxiv.org/abs/2402.04333): 指令微调（Instruction tuning）释放了大语言模型（LLM）的强大能力，有效地利用组合数据集来开发通用聊天机器人。然而，实际应用中经常需要...

  

---

### DiscoResearch ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/1211590899902058557) (13 messages🔥): 

- **EQ-Bench 现在支持德语 ("Sprechen Sie Deutsch")**: `@.calytrix` 宣布 EQ-Bench 已添加德语支持，旨在为用户提供更快、更经济的体验。更新后的 Benchmark 可在 [GitHub](https://github.com/EQ-bench/EQ-Bench) 上访问。

- **德语版 EQ-Bench 评分出炉**: `@.calytrix` 分享了初步的 Benchmark 分数，显示 **GPT-4** 在德语版本中以 81.91 分的表现领先，而其在英语版 EQ-Bench 的分数为 86.05。

- **翻译质量受到质疑**: `@_jp1_` 对 EQ-Bench 德语翻译的准确性表示担忧，认为**情感的细微差别**可能无法在英语和德语之间直接映射。

- **流畅度影响情感基准评分**: `@.calytrix` 对讨论进行了反思，指出虽然德语版 EQ-Bench 测试了德语流畅度，但与英语版相比，它可能无法捕捉到*情感智能 (emotional intelligence)* 的其他方面。

- **波兰语翻译揭示模型差距**: `@remek1972` 报告了将 MT-Bench 翻译为波兰语的情况，强调了模型在波兰语和英语任务表现上的显著差异，表明 **GPT-4** 等模型的语言流畅度是一个关键因素。

**相关链接**:

[GitHub - EQ-bench/EQ-Bench: A benchmark for emotional intelligence in large language models](https://github.com/EQ-bench/EQ-Bench): 大型语言模型情感智能基准测试 - EQ-bench/EQ-Bench

  

---


### DiscoResearch ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (1 messages): 

thomasrenkert: 感谢解释 🙂
  

---



### LLM Perf Enthusiasts AI ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/1211631845087649893) (5 messages): 

- **FireFunction V1 闪亮登场**: `@sourya4` 重点介绍了 **FireFunction V1**，这是一款拥有 **GPT-4 级别结构化输出**和决策路由能力的新模型，且延迟显著降低。他指出该模型是 open-weights 且可商用的。他们分享了 [FireFunction 的博客文章](https://fireworks.ai/blog/firefunction-v1-gpt-4-level-function-calling)，并提到了该模型的 JSON 和 grammar 模式，有助于保持输出结构。
- **探索最佳 Function Calling 方案**: `@yikesawjeez` 列举了 **Gorilla OpenFunctions, NexusRaven,** 和 **Litellm Function Calling Wrapper** 作为他们目前首选的 Function Calling 解决方案。他们还暗示将参加即将举行的专注于 Fire Functions 的 **hackathon**。
- **Mixtral Large: 有人尝试过吗？**: `@thisisnotawill` 询问是否有人试用过 **Mixtral Large**，但聊天中没有提供关于其使用或性能的详细回复。
- **好奇者想了解详情**: `@justahvee` 询问了 **FireFunction V1** 响应延迟的具体细节，询问是指首个 token 时间 (time to first token) 还是完成时间 (time to completion)，并将其与 GPT-4 较长的时间进行了对比。

**相关链接**:

[Tweet from Lin Qiao (@lqiao)](https://x.com/lqiao/status/1760664322215379153?s=12): 🔥 Structure is all you need. 🔥 我们很高兴地宣布：- FireFunction V1 - 我们新的 open-weights function calling 模型：- GPT-4 级别的结构化输出和决策路由，延迟降低 4 倍...

  

---


### LLM Perf Enthusiasts AI ▷ #[offtopic](https://discord.com/channels/1168579740391710851/1168762388586176594/1211772674267807764) (4 messages): 

- **R2R 树立行业新标杆**: `@emrgnt_cmplxty` 宣布推出 **R2R**，这是一个用于快速开发和部署生产级 RAG 系统的框架，旨在简化从实验模型到生产环境的过渡。在 [GitHub - SciPhi-AI/R2R](https://github.com/SciPhi-AI/R2R) 探索 R2R。
  
- **寻求框架间的差异说明**: `@yikesawjeez` 寻求关于新 **R2R 框架**与 GitHub 上的 [agentmemory](https://github.com/JoinTheAlliance/agentmemory) 之间差异的 TL;DR，询问两者是否有区别。

- **为 Twitter 视频点赞**: `@yikesawjeez` 赞扬了一个未指明的*极其精彩的 Twitter 视频*，认可了同僚的出色工作。

**相关链接**:

- [GitHub - SciPhi-AI/R2R: A framework for rapid development and deployment of production-ready RAG systems](https://github.com/SciPhi-AI/R2R): 用于快速开发和部署生产级 RAG 系统的框架 - SciPhi-AI/R2R
- [GitHub - JoinTheAlliance/agentmemory: Easy-to-use agent memory, powered by chromadb and postgres](https://github.com/JoinTheAlliance/agentmemory): 易于使用的 Agent 记忆库，由 chromadb 和 postgres 提供支持 - JoinTheAlliance/agentmemory

  

---

### LLM Perf Enthusiasts AI ▷ #[collaboration](https://discord.com/channels/1168579740391710851/1168816033130365018/1211701454960730162) (2 条消息): 

- **GPT-4 处理药物信息**：用户 `@thebaghdaddy` 成功利用 GPT-4 生成了信息卡片，通过将数据整理成表格，涵盖了一系列药物的机制、副作用和主要疾病靶点等维度。尽管输出略显冗长，但该方法被证明是有效的。
- **缺失 Anki 的 Image Occlusion 功能**：`@thebaghdaddy` 强调了 GPT-4 无法在输出中包含图像的局限性，这突显了 Anki 的 Image Occlusion（图像遮盖）功能在学习中的实用性。
  

---


### LLM Perf Enthusiasts AI ▷ #[speed](https://discord.com/channels/1168579740391710851/1168986766607384638/1212100162579861537) (6 条消息): 

- **延迟之痛**：`@res6969` 表达了困扰，称“这让我深感痛苦”，但未具体说明问题。
- **以秒计的延迟**：`@res6969` 强调了对 OpenAI API **以秒计的延迟**的担忧，认为这是一个问题。
- **寻求托管解决方案**：`@res6969` 询问 **Dedicated Hosting（专用托管）** 是否是解决延迟问题的唯一方案。
- **Azure 托管令人失望**：`@pantsforbirds` 加入了对话，提到 Azure 的结果令人失望，暗示对 Azure 在此场景下的性能表示不满。
  

---


### LLM Perf Enthusiasts AI ▷ #[rag](https://discord.com/channels/1168579740391710851/1169086375686053890/1211807444452249660) (1 条消息): 

- **RAG 系统增强头脑风暴**：`@jxnlco` 分享了他们改进客户 RAG 系统的方法，并征求反馈。查看这些想法并在 [GitHub](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md) 上做出贡献。


**提到的链接**：

[n-levels-of-rag/README.md at main · jxnl/n-levels-of-rag](https://github.com/jxnl/n-levels-of-rag/blob/main/README.md)：通过在 GitHub 上创建账号来为 jxnl/n-levels-of-rag 的开发做出贡献。

  

---



### Alignment Lab AI ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/1211602243456081921) (6 条消息): 

- **用轮次 Token 增强 Gemma**：`@imonenext` 已将 `<start_of_turn>` 和 `<end_of_turn>` Token 集成到 **Gemma** 模型中，该模型可在 [Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens) 上使用。这些 Token 旨在促进进一步的 Instruction/RL（强化学习）微调。
- **Token 集成的手动魔法**：`@imonenext` 澄清，向 **Gemma** 模型添加指令微调 Token 的过程是通过手动复制 Tokenizer 完成的。
- **全部原始，没有问题**：在回答 `@ufghfigchv` 的询问时，`@imonenext` 说明由于使用了原始的指令 Token，过程中没有遇到任何问题。

**提到的链接**：

- [Hugging Face – 建设未来的 AI 社区](https://huggingface.co)：（无描述）
- [imone/gemma-7b-with-it-tokens · Hugging Face](https://huggingface.co/imone/gemma-7b-with-it-tokens)：（无描述）

  

---



### Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=mw3VvbYE0o8
  

---



### AI Engineer Foundation ▷ #[events](https://discord.com/channels/1144960932196401252/1144960932657758212/1212116870539513946) (1 条消息): 

- **Agent Protocol V2 实时编程环节**：`@_z` 分享了一个 [YouTube 直播链接](https://youtube.com/live/zrJuNUGYKJg?feature=share)，他们正在处理 **Agent Protocol V2 里程碑的配置选项 RFC**。直播邀请观众在 `_z` 编写代码时加入并互动。

**提到的链接**：

[Coding - Working on Agent Protocol V2 Milestone, Config Options, New RFCs](https://youtube.com/live/zrJuNUGYKJg?feature=share)：大家好，我是 Ziggy！我是一名开源开发者、玩家和技术爱好者。你可以在 GitHub 上找到我：https://github.com/jzanecook。有兴趣贡献...