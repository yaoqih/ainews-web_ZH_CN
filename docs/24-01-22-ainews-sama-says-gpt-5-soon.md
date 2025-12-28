---
companies:
- openai
- codium
- thebloke
- amd
- hugging-face
date: '2024-01-22T20:51:23.366064Z'
description: '**山姆·奥特曼（Sam Altman）**在达沃斯论坛上强调，他的首要任务是发布新模型（很可能命名为 **GPT-5**），同时对**伊利亚·苏茨克维尔（Ilya
  Sutskever）**的任职状态表示不确定。来自 **Codium 的 Itamar** 介绍了 **AlphaCodium** 的“**流程工程**”（Flow
  Engineering）概念，引起了**安德烈·卡帕斯（Andrej Karpathy）**的关注。在 **TheBloke 的 Discord** 频道中，工程师们讨论了一种**多专业混合专家（MoE）模型**，该模型结合了七个分别专注于法律、金融和医学领域的
  70 亿参数模型。关于 **8 位微调（8-bit fine-tuning）**以及在 GPU 支持下使用 **bitsandbytes** 的讨论非常突出。讨论还涉及使用
  **Mergekit** 等工具进行**模型合并**，以及与 **Alpaca 格式**的兼容性。值得注意的是，人们对使用 **AOCL blas 和 lapack
  库**配合 **llama.cpp** 在 **AMD** 硬件上优化 AI 模型表现出了浓厚兴趣。用户尝试将 AI 用于命令行任务，而 **Mixtral MoE
  模型**经过优化，在编程能力上超过了更大规模的模型。对 **GPT-3.5**、**Mixtral**、**Gemini Pro** 和 **GPT-4** 等大语言模型的比较，主要集中在知识深度、问题解决能力和运行速度上，特别是在编程任务方面。'
id: 134bae82-928e-4c06-8299-07e9b9135c3f
models:
- gpt-5
- mixtral-7b
- gpt-3.5
- gemini-pro
- gpt-4
- llama-cpp
original_slug: ainews-ai-discords-1192024
people:
- sam-altman
- ilya-sutskever
- itamar
- andrej-karpathy
title: 奥特曼（Sam Altman）表示：GPT-5 很快就来。
topics:
- mixture-of-experts
- fine-tuning
- model-merging
- 8-bit-optimization
- gpu-acceleration
- performance-comparison
- command-line-ai
- vector-stores
- embeddings
- coding-capabilities
---

<!-- buttondown-editor-mode: plaintext -->> 我们为您检查了 **19** 个 guilds，**290** 个频道和 **4378** 条消息。预计节省的阅读时间（按 200wpm 计算）：**377 分钟**。

https://www.youtube.com/watch?v=QFXp_TU-bO8

[Sama 在达沃斯](https://www.axios.com/2024/01/17/sam-altman-davos-ai-future-interview)：

- Altman 表示，他目前的首要任务是发布新模型，可能被命名为 GPT-5。
- 令人惊讶的是，Altman 承认他“不确定” Sutskever 职位的确切状态。

另外，[来自 Codium 的 Itamar 通过 AlphaCodium 提出了 Flow Engineering 概念](https://twitter.com/itamar_mar/status/1747957348293824676)，并得到了 [Karpathy](https://x.com/karpathy/status/1748043513156272416?s=20) 的关注。

--

**目录**

[TOC]

## [TheBloke](https://discord.com/channels/1111983596572520458) Discord 总结

- **构思“瑞士军刀”式 AI**：响应 `@cos2722` 的提议，服务器上的工程师们讨论了构建一个**多专业 MOE 模型**，该模型结合了 7 个不同的 7B 参数模型，每个模型分别专注于法律、金融和医学等领域。
  
- **8-Bit 微调辩论**：`@netrve` 和 `@that_one_short_guy` 讨论了微调时使用 8-bit 优化器的必要性，后者建议确保安装了支持 GPU 的 **bitsandbytes** 以实现最佳运行。
  
- **果断处理频道垃圾信息**：一名发布违规垃圾信息的攻击性用户被立即从社区封禁，体现了迅速的审核行动。
  
- **模型合并对话**：讨论围绕合并具有共享架构的模型展开，并提供了实用建议，例如使用 **Mergekit** 或确保模型采用 **Alpaca format** 以获得更广泛的兼容性。
  
- **针对 AI 模型的 AMD 优化**：社区有兴趣测试 AI 模型在 AMD 系统上的性能，特别是配合 `llama.cpp` 使用 **AMD AOCL blas 和 lapack 库**，通过 AVX512 寄存器提高效率。
  

**TheBloke 频道总结**

### ▷ #[general](https://discord.com/channels/1111983596572520458/1111984430945402960/) (1151 条消息🔥🔥🔥):

- **用于命令行任务的 AI**：`@stoop poops` 分享了一个实验，他们让 AI 访问非交互式 bash 终端 shell 命令以观察其行为。这个被称为 'tewi' 的 AI 能够执行诸如 `cat /etc/shadow`、对路由器进行 `nmap` 扫描，甚至使用 `ssh-keygen` 等操作。
  
- **Mixtral 的编程能力**：`@rombodawg` 一直在为 Mixtral MoE 模型优化提示词（prompts），目标是在人工评估中超越 33b 模型，旨在以 13b 参数的速度实现比 GPT-3.5 更强的编程能力。
  
- **用于自主任务的 LLM**：`@selea` 询问是否有人成功地将编程 AI 用于编写网站解析器或编写游戏怪物（mob）行为脚本等任务，且无需人工监督，并暗示在有足够示例和微调（fine tuning）的情况下这是可能的。
  
- **LLM 中的嵌入式系统与向量存储 (Vector Stores)**：`@iukea` 讨论了通过向量存储和嵌入（embeddings）增强 AI 的潜力，将 GPT-4 的知识深度与其他模型进行了对比，并探讨了在实际应用中使用大模型的意义。
  
- **LLM 之间的性能比较**：包括 `@giftedgummybee`、`@iukea` 和 `@natepdx` 在内的多位用户对比了 GPT-3.5、Mixtral、Gemini Pro 和 GPT-4 等 LLM，讨论了它们在知识深度、问题解决能力、响应质量和速度方面的优势，特别是在代码相关任务的背景下。
  

**提到的链接**：

- [Squidward Spongebob GIF - Squidward Spongebob Head Bang - Discover & Share GIFs](https://tenor.com/view/squidward-spongebob-head-bang-gif-15984525)：点击查看 GIF
- [Release Smooth Sampling Test Build (koboldcpp) · kalomaze/koboldcpp](https://github.com/kalomaze/koboldcpp/releases/tag/smooth-sampling-v1)：动态温度采样是一个独特的概念，但有一点一直让我困扰：我们基本上被迫使用像 Min P 或 Top K 这样的截断策略，因为动态选择的温度由其...
- [TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-Mixtral-8x7B-DPO-GGUF)：未找到描述
- [RamAnanth1/lex-fridman-podcasts · Datasets at Hugging Face](https://huggingface.co/datasets/RamAnanth1/lex-fridman-podcasts)：未找到描述
- [How vector search and semantic ranking improve your GPT prompts](https://youtu.be/Xwx1DJ0OqCk?si=bzehk6Oxmf2o4EPl)：改进信息检索过程，从而获得生成有用 AI 响应所需的最优基准数据（grounding data）集。了解 Azure Cognitive 如何...
- [GitHub - SteveJustin1963/tec-iDADmm: tec1 MINT running a digital to analog to digital repeating loop to speed calculations, eg matrix multiplication](https://github.com/SteveJustin1963/tec-iDADmm)：tec1 MINT 运行数字到模拟到数字的重复循环以加速计算，例如矩阵乘法。
- [Releases · kalomaze/koboldcpp](https://github.com/kalomaze/koboldcpp/releases)：一个简单的单文件方式，通过 KoboldAI 的 UI 运行各种 GGML 模型。
- [GitHub - itsme2417/PolyMind: A multimodal, function calling powered LLM webui.](https://github.com/itsme2417/PolyMind)：一个多模态、由函数调用（function calling）驱动的 LLM WebUI。
- [sade-adrien/redpajama_v2_sample_100M · Datasets at Hugging Face](https://huggingface.co/datasets/sade-adrien/redpajama_v2_sample_100M)：未找到描述

### ▷ #[characters-roleplay-stories](https://discord.com/channels/1111983596572520458/1112690728531918948/) (425 条消息🔥🔥🔥):

- **LLM 集中设置**：`@firepin123` 提议创建一个类似于 Hugging Face 的开源前端设置集中平台，并配备投票系统，通过标准化设置来简化 LLM 的使用，改善用户体验，并辅助调试和 Benchmarking 过程。
- **微调技术讨论**：`@c.gato` 等人讨论了 LLM 的 DPO 和微调技术，特别是关于 `@c.gato` 的模型 Thespis-13b。`@jondurbin` 建议在进行 DPO 时使用 rmsprop 代替 adam，并注意观察学习率是否过高的迹象。
- **模型创作者提供的 RP 角色卡**：`@stoop poops` 和 `@c.gato` 讨论了模型创作者包含默认角色卡的潜在好处，前者表示更倾向于“正常”内容，出于内容敏感性排除了 ERP 卡。
- **探索用于角色扮演的 LLM**：`@netrve` 分享了使用基于 Yi-32B 的 Doctor's Nous-Capybara LimaRP 的积极体验，并表达了对其使用 DPO 的好奇，同时感叹微调像 WinterGoddess 这样的模型成本过高。
- **设置的重要性与文档**：包括 `@theyallchoppable`、`@doctorshotgun` 和 `@keyboardking` 在内的几位用户讨论了正确设置对于获得模型最佳性能的重要性，以及对更好文档和社区驱动建议的需求。

**提到的链接**：

- [Doubt Press X GIF - Doubt Press X La Noire - Discover & Share GIFs](https://tenor.com/view/doubt-press-x-la-noire-meme-x-button-gif-19259237)：点击查看 GIF
- [Kquant03/FrankenDPO-4x7B-GGUF · Hugging Face](https://huggingface.co/Kquant03/FrankenDPO-4x7B-GGUF)：未找到描述
- [Kquant03/Prokaryote-8x7B-bf16 · Hugging Face](https://huggingface.co/Kquant03/Prokaryote-8x7B-bf16)：未找到描述
- [Robert Downey GIF - Robert Downey Jr - Discover & Share GIFs](https://tenor.com/view/robert-downey-jr-tony-stark-gif-26471287)：点击查看 GIF
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/15lwtai/new_sillytavern_release_with_proxy_replacement/jvdtgr6/?context=3)：未找到描述
- [cloudyu/Mixtral_34Bx2_MoE_60B · Hugging Face](https://huggingface.co/cloudyu/Mixtral_34Bx2_MoE_60B)：未找到描述
- [moreh/MoMo-70B-LoRA-V1.4 · Hugging Face](https://huggingface.co/moreh/MoMo-70B-LoRA-V1.4)：未找到描述
- [Ayumi Benchmark ERPv4 Chat Logs](http://ayumi.m8geil.de/erp4_chatlogs/#!/index)：未找到描述
- [bagel/bagel/tune/dpo.py at main · jondurbin/bagel](https://github.com/jondurbin/bagel/blob/main/bagel/tune/dpo.py)：A bagel, with everything. 通过在 GitHub 上创建账号为 jondurbin/bagel 的开发做出贡献。
- [c-gatomon](https://wandb.ai/c-gatomon/Mayo7b/runs/q5gwug34?workspace=user-)：Weights & Biases，机器学习开发者工具
- [c-gatomon](https://wandb.ai/c-gatomon/Mayo7b/runs/6k0a9wkh?workspace=user-)：Weights & Biases，机器学习开发者工具
- [medmcqa · Datasets at Hugging Face](https://huggingface.co/datasets/medmcqa)：未找到描述
- [GBaker/MedQA-USMLE-4-options · Datasets at Hugging Face](https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options)：未找到描述
- [dataset (dataset)](https://huggingface.co/dataset)：未找到描述
- [GitHub - kbressem/medAlpaca: LLM finetuned for medical question answering](https://github.com/kbressem/medAlpaca)：为医学问答微调的 LLM。通过在 GitHub 上创建账号为 kbressem/medAlpaca 的开发做出贡献。

### ▷ #[training-and-fine-tuning](https://discord.com/channels/1111983596572520458/1112794268080283728/) (29 messages🔥):

- **超级瑞士军刀 AI 的愿景**：`@cos2722` 提出了创建一个**多专业 MOE 模型**的想法，通过结合最优秀的 7B 参数专业化模型，使其像瑞士军刀一样全能。该模型将通过整合 **DeepSeek7b**、**Open Chat 0106**、**Medicine Chat**、**Finance Chat**、**Law Chat** 以及其他三个自选模型来处理各种复杂的请求。
  
- **用于微调的 8-Bit 优化器**：`@netrve` 收到关于 **bitsandbytes** 在编译时未启用 GPU 支持的警告，并寻求关于 8-bit 支持对微调重要性的澄清。`@that_one_short_guy` 澄清说 8-bit 很少用于微调，并建议**安装带有 GPU 支持的 bitsandbytes**。
  
- **快速封禁**：`@mrdragonfox` 迅速封禁了一名滥用用户，`@netrve` 确认了这一点，并注意到该用户在**其他频道也发送了垃圾信息**。
  
- **医疗领域 MLX 微调的困境**：`@cogbuji` 分享了在医疗数据集上使用 MLX 进行指令微调（instruction fine-tuning）时遇到的挑战，结果导致输出语无伦次。他们考虑**转向自监督方法**，而不是目前使用的有监督指令微调方法。
  
- **Bagel 模型训练，并不理想**：`@jondurbin` 分享了他们的 **Bagel-1.1B 训练**的[损失图表](https://wandb.ai/jondurbin/bagel-1.1b-v0.3/runs/wxidsckq?workspace=user-jondurbin)，显示评估损失有所下降，但性能并未随之提升，判断该模型“完全脑死（braindead）”，并建议不要使用 **tinyllama**。`@sanjiwatsuki` 将其实验与损失更高的 **TinyMistral 模型**进行了对比。
  

**相关链接**：

[jondurbin](https://wandb.ai/jondurbin/bagel-1.1b-v0.3/runs/wxidsckq?workspace=user-jondurbin): Weights & Biases，机器学习开发者工具

### ▷ #[model-merging](https://discord.com/channels/1111983596572520458/1136257162717429760/) (10 messages🔥):

- **寻求模型合并工具**：`@givan_002` 询问是否有脚本或资源可以将使用开源角色扮演数据集微调的 13B 模型与其他 13B 模型进行合并。
- **关于合并相同架构模型的建议**：`@kquant` 建议 13B 模型通常可以与任何模型合并，只要它们共享相同的架构，例如 Mistral 与 Mistral 合并，Llama 与 Llama 合并。
- **确保合并格式兼容**：`@kquant` 还提到确保被合并的模型遵循相同格式的重要性。
- **Mergekit 作为模型合并方案**：`@sao10k` 建议使用 Mergekit 来满足模型合并需求。
- **Alpaca 格式的广泛兼容性**：`@sao10k` 解释说 Alpaca 格式是一种“安全的通用格式”，并强调了它在合并 13B 模型中的流行地位，即使模型并非在 Alpaca 格式下训练。

### ▷ #[coding](https://discord.com/channels/1111983596572520458/1112409939336503338/) (2 messages):

- **征集 AMD 爱好者**：`@spottyluck` 正在寻找在没有 GPU 的 AMD 系统上使用 `llama.cpp` 运行模型的人员，以测试 **AMD AOCL blas 和 lapack 库**。这可能有助于利用 AVX512 寄存器并优化性能。
- **寻找下载地址**：`@apcameron` 询问在哪里可以**下载**进行 `@spottyluck` 提到的测试所需的 AMD AOCL blas 和 lapack 库。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 总结

- **GPT-5 传闻与现实**：[Sully Omarr 的一条推文](https://fxtwitter.com/SullyOmarr/status/1747711388749852925)引发了关于 GPT-5 的讨论，包括对其影响的预测以及对多模态新颖性的怀疑。用户还讨论了依靠风险投资运行且不收取订阅费的 SAAS 初创公司的财务可持续性，并引用了[一条质疑该商业模式的推文](https://fxtwitter.com/KyleLiang5/status/1745483033895968801?t=yyvXpm2PPTVB-5BWVsPK0Q&s=19)。
  
- **代码生成创新与 AI 模型分发**：**AlphaCodium** 的推出引起了轰动，这是一款在编程竞赛中超越人类的开源代码生成工具，其方法和 GitHub 仓库已[分享](https://github.com/codium-ai/alphacodium)。讨论中还提到了将 Torrents 作为 AI 模型分发的潜在模式，建议采用去中心化的传播方式。
  
- **微调技术与自我奖励模型**：新的 Fine-Tuning 技术如 [SymNoise](https://arxiv.org/abs/2312.01523) 因其提升 LLM 性能的能力而受到关注，同时还有关于模型生成自身奖励的研究，这可能导致超人类 Agent 的出现，并暗示了 AI 训练自给自足的未来。
  
- **Meta 与 AI 领域**：关于 Meta 的 LLaMa 3 以及与 GPT-4 比较的对话反映了对 AGI 进展和策略的期待，包括 GPU 的使用以及对扎克伯格开源承诺的认可。讨论涉及了硬件资源的获取及其对模型训练能力的潜在影响。
  
- **超椭圆挑战与 AI 抱负**：围绕使用 bezier 段创建超椭圆（squircle）发起了一项数学相关的行动，[Figma 博客详细介绍了其中的奥秘](https://www.figma.com/blog/desperately-seeking-squircles/)。此外，[Teknium 的一条推文](https://fxtwitter.com/Teknium1/status/1741638013481091369)中分享的个人成长故事，为寻求在该领域增长专业知识的 AI 新人提供了灵感。
  

**Nous Research AI 频道总结**

### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (22 条消息🔥):

- **GPT-5 期待热潮**：用户 `@teknium` 分享了一条[推文](https://fxtwitter.com/SullyOmarr/status/1747711388749852925)，暗示 **GPT-5** 是 OpenAI 的下一个重大发布，而 `@max_paperclips` 预测会经历一个从最初炒作到性能削弱（nerfs）的周期。
- **对多模态炒作的怀疑**：`@teknium` 和 `@max_paperclips` 对被认为是即将推出的 GPT-5 核心的多模态（multimodality）方面表示不感兴趣，`@teknium` 称其为 "一般般"（meh），而 `@giftedgummybee` 暗示由于拥有充足的计算资源，期望其表现能令人惊艳。
- **风投资助的 SaaS 初创公司成本疑问**：用户 `@0xevil` 分享了一条[推文](https://fxtwitter.com/KyleLiang5/status/1745483033895968801?t=yyvXpm2PPTVB-5BWVsPK0Q&s=19)，质疑一家不收订阅费的 SaaS 公司产品的可持续性，导致 `@gabriel_syme` 评论说其目标是创建“伟大的门户”，而不一定是产品。
- **提议将 Torrent 作为 AI 模型的分发模式**：`@everyoneisgross` 强调了使用 Torrent 分发模型、数据和机器学习应用指令的潜力，正如 Mistral 所展示的那样。
- **对模型微调误解的沮丧**：针对 `@youngphlo` 分享的一条声称微调（finetuning）无法为 LLM 增加新知识的推文，`@teknium` 表现出明显的沮丧，断言微调确实能增加知识，`@youngphlo` 对此表示同情，认为这是一种合理的反应。

**提到的链接**：

- [Shahul Es (@Shahules786) 的推文](https://fxtwitter.com/shahules786/status/1748059074556760421)：来自 Microsoft 的 RAG 与微调（finetuning）对比研究假设微调可以将新的事实/领域特定知识注入 LLM，但这并非事实。微调不是 RAG 的替代方案。目前，只有...
- [Plink Cat GIF - Plink cat Plink Cat - 发现并分享 GIF](https://tenor.com/view/plink-cat-plink-cat-gif-1794292671885121408)：点击查看 GIF
- [Kaizhao Liang (@KyleLiang5) 的推文](https://fxtwitter.com/KyleLiang5/status/1745483033895968801?t=yyvXpm2PPTVB-5BWVsPK0Q&s=19)：@abacaj 买 10,000 个那个，然后以零服务器成本开始你的 LLM SaaS 公司。既然没有订阅，他们怎么能不快点破产？
- [2024年1月18日最新 AI 资讯](https://www.youtube.com/watch?v=POgLwYxDGYk)：AI 的最新进展 [https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/https://www.reddit.com/r/LocalLLaMA/com](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/https://www.reddit.com/r/LocalLLaMA/com)...
- [Sully (@SullyOmarr) 的推文](https://fxtwitter.com/SullyOmarr/status/1747711388749852925)：好的，这在某种程度上得到了证实：“Altman 表示他的首要任务是发布新模型，可能被称为 GPT-5”。预计 OpenAI 的最新模型将在模型能力上实现指数级飞跃。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (13 条消息🔥):

- **被描绘为《她》中“Samantha”的 AGI**：`@burnytech` 分享了 [@Schindler___ 的推文](https://fxtwitter.com/Schindler___/status/1745986132737769573)，提议了一种模仿电影《她》（Her）中 Samantha 的 AGI 架构，该架构具备动态语音、不断进化的性格特征以及外部记忆交互能力。
  
- **探索 AlphaCodium 的能力**：`@metaldragon01` 重点介绍了 [AlphaCodium](https://fxtwitter.com/itamar_mar/status/1747957348293824676)，这是一个开源代码生成工具，据称在编程竞赛中超越了大多数人类选手；`@teknium` 询问它是一个通用的编程模型，还是现有模型之上的应用层。
  
- **GitHub 项目 AlphaCodium 揭晓**：`@adjectiveallison` 发现了 [GitHub 上的 AlphaCodium](https://github.com/codium-ai/alphacodium)，这是一种通过多阶段、基于测试的迭代过程来提高 LLM 代码生成准确性的方法，引发了关于在实际应用中使用迭代方法的讨论。
  
- **利用 SymNoise 彻底改变 LLM 微调**：`@euclaise` 和 `@teknium` 讨论了一种涉及对称噪声的[新微调技术](https://arxiv.org/abs/2312.01523)，该技术可以改进 LLM，在各种模型和数据集上表现出优于以往方法的性能。
  
- **面向超强 Agent 的 Self-Rewarding Language Models**：`@metaldragon01` 发现了关于 [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020) 的研究，该模型可以生成自己的奖励，从而提高指令遵循能力，并在训练期间提供高质量的自我评估。
  

**提到的链接**：

- [SymNoise: Advancing Language Model Fine-tuning with Symmetric Noise](https://arxiv.org/abs/2312.01523)：在本文中，我们介绍了一种新型的语言模型微调技术，该技术涉及在嵌入过程中加入对称噪声。该方法旨在增强模型的功能...
- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)：我们认为，为了实现超强 Agent，未来的模型需要超强反馈以提供充足的训练信号。目前的方法通常根据人类偏好训练奖励模型...
- [Introducing ASPIRE for selective prediction in LLMs – Google Research Blog](https://blog.research.google/2024/01/introducing-aspire-for-selective.html)：未找到描述
- [GitHub - Codium-ai/AlphaCodium](https://github.com/codium-ai/alphacodium)：通过在 GitHub 上创建账号来为 Codium-ai/AlphaCodium 的开发做出贡献。
- [Tweet from Itamar Friedman (@itamar_mar)](https://fxtwitter.com/itamar_mar/status/1747957348293824676)：🚀 介绍 AlphaCodium - 首创的开源代码生成工具，在代码竞赛中超越了大多数人类选手 ⭐️ 灵感来自 DeepMind 的 AlphaCode❤️‍🔥，但击败了它...
- [Tweet from Schindler (@Schindler___)](https://fxtwitter.com/Schindler___/status/1745986132737769573)：(1/2) 关于 AGI 架构的提议。电影《她》中的 Samantha 就在这里：一个能够自由思考和交谈、持续学习和进化的自主 AI。创建...

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (338 条消息🔥🔥):

- **对社交媒体机器人的怀疑**：在系列消息中，`@gabriel_syme` 等人讨论了对 AI 在社交媒体中效用的担忧，认为虽然它可能适用于低质量的文本输出，但这并不是 AI 的最佳应用，且在用例上缺乏想象力。用户们还开玩笑说，Twitter 机器人化是唯一有效的用例。
  
- **讨论 AI Agent 的未来**：对话转向了 Agentic AI 的潜在用途，包括对未来客户服务系统的预测（`@leontello`）。`@.benxh` 补充说，减少人类在社交媒体管理中的参与可能对全人类有益，但对营销专业人员的影响表示保留。
  
- **展望面向代码的 AI 模型**：聊天参与者讨论了对 Agentic AI 在代码测试和开发中应用的期望（`@_3sphere`），以及与多模态模型集成的可能性。他们还评论了某些模型在编码中途停止的挑战，强调了对更长 Token 序列或分步处理的需求（`@teknium`）。
  
- **对齐算法比较**：`@osanseviero` 分享了比较 DPO, IPO 和 KTO 对齐算法的文章链接，结论是 DPO 似乎是整体上的最佳选择，但也承认 KTO 由于数据需求更简单而更易于扩展。用户讨论了各种评估之间的相关性，并提到了基准测试和 Elo 评分。

- **Meta 的 LLaMa 3 与 AGI 竞赛**：Meta 对 LLaMa 3 的训练引发了关于其潜在进展以及如何与 OpenAI 的 GPT-4 进行对比的讨论。对话触及了对 GPUs 等资源的战略性使用，以及 Meta 的 CEO 作为开源发展支持者的引人注目的立场 (`@gezegen`, `@_3sphere`, `@teknium`)。

**提到的链接**：

- [你比 LLM 更聪明吗？](https://d.erenrich.net/are-you-smarter-than-an-llm/index.html): 未找到描述
- [来自 Edward Beeching (@edwardbeeching) 的推文](https://fxtwitter.com/edwardbeeching/status/1747999497609961651): 在我们最新的博客文章中，我们总结了对三种最先进的对齐算法的广泛评估：DPO vs IPO vs KTO。结果展示了关键超参数之间复杂的相互作用...
- [报告索引](https://teknium1.github.io/LLM-Logbook/): 未找到描述
- [与开源大语言模型聊天](https://arena.lmsys.org/): 未找到描述
- [LMSys Chatbot Arena 排行榜 - lmsys 提供的 Hugging Face Space](https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard): 未找到描述
- [来自 Omar Sanseviero (@osanseviero) 的推文](https://fxtwitter.com/osanseviero/status/1746889044414320710): “假设是它们在彼此分离的情况下进行了多样化的训练” 这并不是专家的真正定义（MoE 应该被命名为路由稀疏模型或类似的名称...
- [站立的猫 惊讶的猫 GIF - 站立的猫 惊讶的猫 被催眠 - 发现并分享 GIF](https://tenor.com/view/standing-cat-amazed-cat-hypnotized-hypnotized-cat-gif-23851821): 点击查看 GIF
- [来自 OpenLLMLeaders (@OpenLLMLeaders) 的推文](https://fxtwitter.com/OpenLLMLeaders/status/1748081303084228663): 排行榜新增模型！模型名称 [https://hf.co/intervitens/internlm2-base-20b-llama](https://hf.co/intervitens/internlm2-base-20b-llama) 总排名：800，13B 类别排名：130，基准测试平均分：62.69，ARC: 62.97，HellaSwag: 82.15 M...
- [混合专家模型 (Mixture of Experts) 详解](https://huggingface.co/blog/moe): 未找到描述
- [来自 AK (@_akhaliq) 的推文](https://fxtwitter.com/_akhaliq/status/1748166535795847579): Meta 发布 Self-Rewarding Language Models 论文页面：[https://huggingface.co/papers/2401.10020](https://huggingface.co/papers/2401.10020) 在我们方法的三个迭代中微调 Llama 2 70B，得到的模型优于许多现有的...
- [Mark Zuckerberg 在 Instagram 上表示：“关于我们 AI 工作的一些更新。我们的长期愿景是构建通用智能，负责任地将其开源，并使其广泛可用，让每个人都能受益。我们正在将两个主要的 AI 研究工作（FAIR 和 GenAI）更紧密地结合起来以支持这一点。我们目前正在训练下一代模型 Llama 3，并且正在构建大规模的计算基础设施来支持我们未来的路线图，包括到今年年底拥有 35 万块 H100 —— 如果算上其他 GPU，总计约 60 万块 H100 等效算力。此外，对我们构建新型以 AI 为中心的计算设备（如 Ray Ban Meta 智能眼镜）的进展感到非常兴奋。更多内容即将推出。”](https://www.instagram.com/reel/C2QARHJR1sZ/?utm_source=ig_embed&ig_rid=610676e4-745b-4d79-89bd-844fd1fbd23c): 7.4 万次点赞，5,594 条评论 - zuck 于 2024 年 1 月 18 日发布：“关于我们 AI 工作的一些更新。我们的长期愿景是构建通用智能，开源...”
- [来自 OpenLLMLeaders (@OpenLLMLeaders) 的推文](https://fxtwitter.com/OpenLLMLeaders/status/1747985592464314748): 排行榜新增模型！模型名称 [https://hf.co/chargoddard/internlm2-20b-llama](https://hf.co/chargoddard/internlm2-20b-llama) 总排名：305，13B 类别排名：63，基准测试平均分：70.61，ARC: 64.68，HellaSwag: 83.16，MMLU: 6...
- [来自 Alex Volkov (Thursd/AI) (@altryne) 的推文](https://fxtwitter.com/altryne/status/1748057569816416451): 以防你不想点击进入其他网站，Zuck 的重大更新 - 开源将继续 - 正在训练 Llama 3 - AI + 元宇宙 - 将拥有 350,000 块 H100 和约 60 万块 H100 等效算力...
- [来自 Alim (@almmaasoglu) 的推文](https://fxtwitter.com/almmaasoglu/status/1748066671846138307): @Teknium1 @ylecun 我唯一的问题是他们是怎么搞到这么多（GPU）的，哈哈
- [来自 Archit Sharma (@archit_sharma97) 的推文](https://fxtwitter.com/archit_sharma97/status/1748009137991279036): @Teknium1 @huggingface 哦，从实现角度来看没问题，我还没见过仅通过 *非成对 (unpaired)* 数据就能显著提升的模型。我很想看一些实验！
- [Sparse Universal Transformer](https://arxiv.org/abs/2310.07096): Universal Transformer (UT) 是 Transformer 的一种变体，它在各层之间共享参数。经验证据表明，UT 比 Vanilla Transformer 具有更好的组合泛化能力...
- [ikawrakow 提交的 k-quants · Pull Request #1684 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/pull/1684): 内容：此 PR 增加了一系列 2-6 bit 量化方法，以及量化混合，如 #1240 和 #1256 中所建议。提供了 Scalar, AVX2, ARM_NEON 和 CUDA 实现。原因：这是...

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (44 messages🔥):

- **针对 Embeddings 的 OCR？一个勇敢新世界**：`@_3sphere` 提出了一个新颖的概念，建议对 Embedding 进行 OCR，并思考神经 JPG 何时能成为现实，将 Embedding 视为一种编解码器 (codec)。
- **数学协作解决几何挑战**：`@bernaferrari` 正在寻找数学爱好者来解决使用贝塞尔曲线段 (bezier segments) 表示超椭圆 (squircle) 的问题，正如 Figma 博客文章中所解释的那样。他们认为，一个恰当的数学表示可能会在 Hacker News 上成名并改善该领域，因为目前的方法缺乏优雅性。
- **LLMs 在几何生成中的应用**：`@gabriel_syme` 回忆了过去使用 LLMs 生成几何图形的成功经验，并指出如果当时的模型更好，迭代生成的潜力会更大。同时，`@mr.userbox020` 讨论了几何的深度以及 LLMs 在数学问题上的适用性，建议简单的 2D 向量方法可能就足够了。
- **超椭圆 (Squircle) 之旅**：`@mr.userbox020` 怀疑地对待使用 LLMs 解决 `@bernaferrari` 的超椭圆问题，敦促采用更传统的数学路径而非复杂的 LLMs，因为该问题涉及无理数和无限精度。
- **从 AI 新手到专家的旅程**：在转发的 `@Teknium1` 推文中，庆祝了一个非凡的一年转型，激励了 `@quilalove` 询问从哪里开始他们自己的 AI 领域之旅，并与他人合作开展 AI 技术知识和实现。

**提到的链接**：

- [Teknium (e/λ) (@Teknium1) 的推文](https://fxtwitter.com/Teknium1/status/1741638013481091369)：祝大家新年快乐！🥳 一年前的今天，我：- 从未训练过任何模型 - 对 AI 一窍不通 - 从未在科技行业工作过 - 在 twitter 上只有 8 个关注者？（大概）一年后...
- [苦寻超椭圆 (Squircles) | Figma 博客](https://www.figma.com/blog/desperately-seeking-squircles/)：在 1972 年的一次著名采访中，Charles Eames 回答了一系列关于设计本质的基本问题。
- [LoneStriker/Nous-Capybara-34B-8.0bpw-h8-exl2 at main](https://huggingface.co/LoneStriker/Nous-Capybara-34B-8.0bpw-h8-exl2/tree/main)：未找到描述

---

## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 总结

**GPU 探戈：聚焦 VRAM 和资源管理**：公会的工程师讨论了 **LM Studio** 中的 GPU 卸载 (GPU offload) 设置，指出将 GPU 卸载设置为 -1 会利用所有层，但可能显示较低的 GPU 利用率。建议使用 **Nvidia P40** GPU 作为高性价比的性能解决方案，并对在运行 AI 模型的同时运行游戏等密集型应用时可能出现的 VRAM 分配冲突表示担忧。

**LM Studio Beta V4 亮相**：LM Studio 的 **Beta V4 (0.2.11 release candidate)** 已经发布，其特点是包含带有 VRAM 拟合估算的权重搜索页面，并支持新的 2bit 量化 (quants)。[提供了下载链接](https://discord.com/channels/1110598183144399058/1197706175437873164)，并表示开源计划或添加插件系统正在开发中，保证 **LM Studio 将对个人使用保持免费**。

**来自硬件前线的派遣**：相关的硬件讨论包括双 RTX 3090 配置的电源供应考虑，建议使用 1200W+ 的 PSU。交流了将大型 GPU 装入小机箱的创意解决方案，强调了工程师在优化其 AI 计算装备方面的独创性。

**CrewAI：框架与性能见解**：重点介绍了 **CrewAI Multi-Agent Framework** 及其与 **LM Studio API** 的集成，并提到利用特定 Agent 执行互联网搜索等专用任务。承诺为使用 CrewAI 的多个模型提供基准测试，并在用户工作完成后提供示例代码。

**模型性能与使用**：据报告，本地模型虽然可以运行重复的函数调用，但不如 3.5T 模型令人印象深刻。Skyrim ChatGPT 模组的图像识别被视为与其他进程竞争 GPU 资源的并行任务。还出现了 **LM Studio 安装问题** 以及 **24G RAM 笔记本电脑上的未指定模型错误**，后者已转至技术支持频道以寻求进一步帮助。

**LM Studio 频道总结**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (200 条消息🔥🔥):

- **GPU Offload 与 VRAM 利用率**：`@heyitsyorkie` 解释说，在 LM Studio 中将 GPU offload 设置为 -1 会将所有层分配给 GPU 使用，尽管像 `@4b0d3` 这样的用户报告称 GPU 利用率较低。`@senecalouck` 分享说 ROCm beta 可能会为 AMD 显卡带来显著的速度提升。
  
- **在各种系统上运行 LM Studio**：`@heyitsyorkie` 和 `@dagbs` 讨论了在 Macbook M1/2/3 芯片等硬件上运行 LM Studio，并比较了不同设备间的模型性能，指出 LM Studio 最初是为 MacOS M1/2/3 设计的。
  
- **模型比较与偏好**：`@dagbs` 和 `@4b0d3` 等用户比较了包括 Dolphin 2.6 DPO 和 Laserxtral 在内的各种模型，并根据响应质量和速度讨论了偏好。`@dagbs` 进一步指出，像 Mixtral 这样的大型模型在 Q6 量化下，当上下文长度（context sizes）较高时可能会出现幻觉。
  
- **远程模型使用与推理服务器**：`@dagbs` 澄清说，虽然 LM Studio 不是无头模式（headless），但它确实有一个推理服务器（Inference Server）用于远程运行模型。然而，由于硬件要求较高，像 `@leamac51_62244` 这样的用户仍在寻求关于远程使用模型的讨论。
  
- **LM Studio 安装问题**：`@surrender` 遇到了安装后 LM Studio 无法启动的问题。`@dagbs` 建议删除 .cache/lm-studio 文件夹，并在相应的支持频道寻求更多帮助。
  

**提到的链接**：

- [HuggingChat](https://huggingface.co/chat/): 未找到描述
- [LM Studio Beta Releases](https://lmstudio.ai/beta-releases.html): 未找到描述
- [How Linux Users Install A Web Browser GIF - How Linux Users Install A Web Browser Linux Linux Users - Discover & Share GIFs](https://tenor.com/view/how-linux-users-install-a-web-browser-linux-linux-users-gif-20223386): 点击查看 GIF
- [TheBloke/WhiteRabbitNeo-33B-v1-GGUF · Not able to run this model?](https://huggingface.co/TheBloke/WhiteRabbitNeo-33B-v1-GGUF/discussions/1): 未找到描述
- [TheBloke/MegaDolphin-120b-GGUF · Hugging Face](https://huggingface.co/TheBloke/MegaDolphin-120b-GGUF): 未找到描述
- [How To Install Uncensored Mixtral Locally For FREE! (EASY)](https://www.youtube.com/watch?v=DC2te4CZXeM&list=TLPQMTgwMTIwMjTO3gv0zEnsyg&index=17,): 在这个视频中，我将为你提供关于如何在本地安装无审查 Mixtral 的终极指南！Mixtral 8x7B，一个高质量的稀疏专家混合模型（MoE）...
- [GitHub - Significant-Gravitas/AutoGPT: AutoGPT is the vision of accessible AI for everyone, to use and to build on. Our mission is to provide the tools, so that you can focus on what matters.](https://github.com/Significant-Gravitas/AutoGPT): AutoGPT 是为每个人提供可访问 AI 的愿景，供大家使用和构建。我们的使命是提供工具，让你专注于重要的事情。 - GitHub - Significant-Gravitas/AutoGPT: Aut...
- [Which devices are even supported? (HIP/ROCm) · Issue #1714 · ROCm/ROCm](https://github.com/ROCm/ROCm/issues/1714): 我是一名长期的 CUDA 开发者，想要探索 ROCm 和 HIP 开发，但发现哪些硬件支持这些工具比预想的要难。让我们看看... 这个仓库的...

### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (11 条消息🔥):

- **Skyrim ChatGPT 模组的 GPU 问题**：`@gamerred` 询问 LM Studio 是否需要在 GPU 上运行，因为 Skyrim ChatGPT 模组也在执行图像识别。`@fabguy` 认为这两个进程会竞争 GPU 资源。
- **LM 可能导致轻微的游戏卡顿**：`@dagbs` 解释说，虽然计算和 3D 渲染是分开的，但语言模型 (LLM) 在初始“思考”阶段可能会导致短暂的掉帧，但不会显著影响一般游戏体验。
- **VRAM 分配可能是一个问题**：`@fabguy` 指出真正的问题在于 VRAM 的分配，尽管 `@dagbs` 认为游戏往往会索取超出必要的资源。
- **注意推荐的图形设置**：`@ben.com` 警告说，游戏可能不会考虑到 LLM 已经占用的 GPU VRAM，因此，用户应考虑相应地减小纹理大小或其他设置。
- **游戏期间 LLM 的后台运行**：`@dagbs` 分享了在后台保持 LLM 闲置的同时运行中等图形要求游戏的个人经验，`@_anarche_` 评论说在运行某些 7B 模型时仍能在 COD 中保持高 FPS，这表明他们的配置中存在 CPU 瓶颈。

### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (6 条消息):

- **关于 Beta 版本发布状态的查询**：`@logandark` 询问了新 Beta 版本可能延迟的问题。虽然没有发布确切的更新进展，但对话表明相关工作可能仍在进行中。
- **用户遇到未指定的模型错误**：`@aindy_niu` 报告了在 24G RAM 的笔记本电脑上运行 **lm-studio** 时遇到的问题，出现了退出代码和未知错误。在给出的交流中未提供解决方案。
- **提供的技术支持频道引导**：当 `@aindy_niu` 寻求模型错误帮助时，`@dagbs` 将其引导至特定的 Discord 频道，暗示那里更适合进行技术支持。

### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (79 条消息🔥🔥):

- **预算有限下的体面性能**：`@dagbs` 和 `@heyitsyorkie` 讨论了使用 Nvidia P40 进行 AI 计算的经济可行性，指出如果有相应的配置环境，这是一个值得推荐的廉价选择，它能提供 24GB 的 VRAM 以及不错的性能，对于像 120b Goliath 这样的大型模型，甚至能通过“多张 P40 实现个位数的 tok/s”。
- **双 3090 的电源供应**：`@rouw3n` 询问了配置第二张 RTX 3090 的 PSU 需求，对此 `@heyitsyorkie` 等人建议使用 1200W+ 的电源，而 `.ben.com` 则建议通过一些调整，1000W 可能也足够。
- **在狭小空间内集成 GPU**：`@dagbs`、`.ben.com` 和 `@pefortin` 等用户分享了他们通过重新利用空间、使用 PCI 延长线或将硬件靠在其他组件上，将大型 GPU 装入较小机箱的经验，突出了构建紧凑且强大的 AI 设备（rigs）的创意解决方案。
- **实验最佳 GPU 负载**：`@ericericericericericericeric` 参与了关于针对不同模型大小实验 GPU Offload 层数设置的讨论，`@heyitsyorkie` 建议尝试调整层数并监控 VRAM 使用情况，并指出没有万能的设置。
- **提升一体机的 AI 性能**：`@jilloschwortz` 寻求提升配置为 i7 13700 和 16GB RAM 的一体机 PC 的 AI 性能。`@heyitsyorkie` 建议攒钱买一台专用设备，而 `@dagbs` 则提出了外接 GPU 连接作为一个可行的解决方案。

### ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/) (37 条消息🔥):

- **Beta V4 取得进展**：`@yagilb` 宣布 **Beta V4 (0.2.11 release candidate)** 已发布，其特点是拥有带 VRAM 占用预估的新模型搜索页面、修复了文本粘贴的 bug，以及集成了最新的 `llama.cpp` 提交。鼓励用户对新搜索页面提供反馈，[下载链接请点击此处](https://discord.com/channels/1110598183144399058/1197706175437873164)。
  
- **2bit 量化创新**：在简短的交流中，`@n8programs` 询问且 `@yagilb` 确认了 Beta V4 支持 **新的 2bit 量化 (quants)**，展现了对该更新的兴奋。
  
- **ROCm 目前仍保持独立**：`@_anarche_` 询问了 ROCm 支持情况，`@yagilb` 回复称其尚未集成，在集成工作简化之前将继续单独分享。
  
- **插件功能指日可待**：当 `@n8programs` 询问关于开源 LM Studio 以接受社区贡献或添加插件系统的可能性时，`@yagilb` 暗示相关计划正在开发中。
  
- **个人使用始终免费**：在关于 LM Studio 未来定价的猜测中，`@yagilb` 向 `@n8programs` 保证，它将维持现有模式，**对个人使用保持免费**。
  

### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (1 条消息):

yagilb: [https://discord.com/channels/1110598183144399058/1197707651438624849](https://discord.com/channels/1110598183144399058/1197707651438624849)

### ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/) (1 条消息):

- **本地模型表现尚可但并非“极佳”**：用户 `@anarche_` 评论道，他们在多次处理函数调用（function calls）方面成功使用了多个本地模型。然而，他们指出这些模型不如 3.5T 模型那样令人印象深刻。

### ▷ #[avx-beta](https://discord.com/channels/1110598183144399058/1177047883237822536/) (1 条消息):

- **错误：不允许额外的属性**：用户 `@_elchupacabras` 遇到了一个错误，提示 **"Error: must NOT have additional properties. File contains unknown property: 'min_p'"**，目前正在寻求解决方案。

### ▷ #[crew-ai](https://discord.com/channels/1110598183144399058/1197374792668545034/) (10 messages🔥):

- **CrewAI 多智能体框架实践**：`@senecalouck` 讨论了在 **CrewAI** 中利用 **LM Studio API** 配合 `@<bot-id>` 进行互联网搜索和摘要。他们实施了一种策略，将特定的 Agent 与搜索等单个工具对齐，而 Crew 中的其他成员仅使用 **LLM** 访问权限。
- **使用 CrewAI 对多个模型进行基准测试**：用户 `@_anarche_` 提到正在使用 CrewAI 进行 Benchmarking，测试了多个模型，并承诺在完成后分享所使用的 Crew 设置结果和示例代码。
- **关于 Dolphin DPO 分数的问题**：`@dagbs` 询问了 Dolphin DPO 分数旁边的星号 (\*) 的含义，并表达了 Dolphin 设置中的一个特定问题，即忘记安装 requirements。
- **Dolphin 模型的小挫折**：在回复 `@dagbs` 时，`@_anarche_` 承认 Dolphin 模型“完成了工作，但有一两个小问题”，暗示其性能存在一些不一致性。

---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord 摘要

- **Perplexity 与 Rabbit 联手**：在与 Rabbit OS 建立合作伙伴关系后，前 100,000 名购买 Rabbit R1 的用户将获得一年的免费 Perplexity Pro，通过集成 **PPLX** 在线 LLM API 提供**实时、精准的回答**。[Rabbit 的推文](https://x.com/rabbit_hmi/status/1748105199418490968?s=46&t=Hug1fgRBxMNq3A5tmmpzqw)还强调了为 r1 用户提供的自然语言搜索增强。
  
- **关于所使用 AI 模型的说明**：Perplexity 向用户保证，**Perplexity Pro** 确实采用了包括 **GPT-4** 和 **Claude 2.1** 在内的真实模型，技术细节详见其 [Technical FAQ](https://blog.perplexity.ai/technical-faq/what-models-does-copilot-use)。特别是，**Copilot** 为 Pro 用户使用 GPT-4，并由微调版的 GPT-3.5 提供支持。
  
- **独家优惠引发热议**：合作伙伴关系的揭晓引发了热潮，为 Rabbit r1 的前 100,000 名买家提供 **200 美元的免费 Perplexity Pro 额度**，这在 [Jesse Lyu 的推文](https://fxtwitter.com/jessechenglyu/status/1748138591828709421)中得到确认，并强调 Rabbit r1 上的 Perplexity 将免收订阅费。
  
- **免费 AI 工具吸引社区**：一段分享的 [YouTube 视频](https://www.youtube.com/watch?v=ZYUt4WE4Mrw)展示了“23 个你不敢相信是免费的 AI 工具”，通过一个月的免费 Skillshare 试用激励观众；而另一段视频则支持 **Perplexity AI** 作为内容创作中优于 Google 等其他工具的首选，可在[此处](https://www.youtube.com/watch?v=aphHCBSTx7Q)观看。
  
- **社区帮助与 API 交互**：突出了积极的社区互动，一位用户对高效解决支付方式问题表示感谢。然而，用户被告知某些特定信息和功能目前尚不可用，也未列入开发 Roadmap，强调了根据当前能力管理预期的必要性。
  

**Perplexity AI 频道摘要**

### ▷ #[announcements](https://discord.com/channels/1047197230748151888/1047204950763122820/) (1 messages):

- **Perplexity 与 Rabbit 达成合作**：`@ok.alex` 宣布了一项合作伙伴关系，将 PPLX 在线 LLM API 与 Rabbit R1 集成，以提供**实时、精准的回答**。前 100,000 名购买 Rabbit R1 的用户将获赠一年的 Perplexity Pro。

### ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/) (186 条消息🔥🔥):

- **Perplexity AI 模型说明**：用户如 `@charlesalan` 寻求确认 Perplexity Pro 是否使用了真正的 **GPT-4** 和 **Claude 2.1** 模型。`@icelavaman` 提供了保证和链接来澄清这些细节，确认了所用模型的真实性。
- **Copilot 模型详情**：针对 `@gpt_five` 的查询，`@icelavaman` 分享了一份 [技术常见问题解答 (Technical FAQ)](https://blog.perplexity.ai/technical-faq/what-models-does-copilot-use)，详细说明了 **Copilot 为 Pro 用户提供 GPT-4**，并由经过微调的 GPT-3.5 版本进行路由，强调了其提供深度回答的能力。
- **令人兴奋的合作伙伴关系公告**：`@otub` 透露了一项 **合作伙伴关系**，将为前 100,000 名 Rabbit r1 购买者提供 **价值 200 美元的免费 Perplexity Pro 额度**。包括 `@glap` 和 `@ok.alex` 在内的多位用户确认了这一交易，并指出该额度甚至可以延长现有的 Pro 订阅。
- **澄清 R1 与 Perplexity Pro 的关系**：`@dan9070` 引用了 `@jessechenglyu` 的一条 Twitter 帖子，确认 R1 将在 **rabbit r1 上免费提供 Perplexity，无需任何订阅**——这对该设备的早期采用者来说是一个重大福利。
- **用户参与和支持**：`@lkshrc` 和 `@yogable` 询问了如何获取 Pro Discord 访问权限，`@icelavaman` 迅速解决了该问题，展示了 Perplexity AI 服务器内的社区支持和响应能力。

**提到的链接**：

- [Chat with Open Large Language Models](https://chat.lmsys.org/)：未找到描述
- [Jesse Lyu (@jessechenglyu) 的推文](https://fxtwitter.com/jessechenglyu/status/1748138591828709421)：关键信息：1. rabbit r1 上的 Perplexity 是免费的。2. Perplexity 为前 10 万个 r1 订单提供 200 美元的免费额度作为礼品。3. rabbit r1 保持免订阅。结论：太划算了！↘️ 引用...
- [什么是 Perplexity Copilot？](https://blog.perplexity.ai/faq/what-is-copilot)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Perplexity 博客](https://blog.perplexity.ai/faq/how-does-file-upload-work.)：浏览 Perplexity 博客，获取文章、公告、产品更新和优化体验的技巧。保持关注并充分利用 Perplexity。
- [Chat Completions](https://docs.perplexity.ai/reference/post_chat_completions)：未找到描述
- [Perplexity - AI Companion](https://chromewebstore.google.com/detail/perplexity-ai-companion/hlgbcneanomplepojfcnclggenpcoldo)：浏览时随时提问
- [Copilot 使用哪些模型？](https://blog.perplexity.ai/technical-faq/what-models-does-copilot-use)：通过我们全面的常见问题解答页面深入了解 Perplexity 的技术细节。从 GPT-4 和 Claude 2 等 AI 模型的细微差别到 Token 限制和 AI 配置文件，获取简明答案以优化您的...

### ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/) (5 条消息):

- **YouTube 上的免费 AI 工具**：`@siddhj` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=ZYUt4WE4Mrw)，标题为“23 个你不敢相信是免费的 AI 工具”，展示了多种免费可用的 AI 工具。视频描述中提到了与 Skillshare 的合作，提供为期一个月的免费试用。
- **对 Riley Brown 视频的赞赏**：`@samangel7358` 对 Riley Brown 的努力表示认可，赞扬了另一个内容丰富的 AI 相关视频。
- **Rabbit 与 Perplexity AI 达成合作**：`@br0k3r81` 强调了 rabbit OS 与 Perplexity AI 之间的新合作伙伴关系，该消息通过 [@rabbit_hmi 的推文](https://x.com/rabbit_hmi/status/1748105199418490968?s=46&t=Hug1fgRBxMNq3A5tmmpzqw)分享，旨在提升 r1 用户的自然语言搜索能力。
- **Perplexity AI 服务实测**：`@almost.engineering` 发布了一个 [链接](https://www.perplexity.ai/search/Gannon-Makerspace-ernLYqi9TJiB3QNZjAH_Rw?s=c)，展示了 Perplexity AI 针对 Gannon Makerspace 相关特定内容的搜索能力。
- **对 Perplexity AI 的个人偏好**：`@oneisall_` 分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=aphHCBSTx7Q)，创作者在其中解释了为什么他们更喜欢使用 Perplexity 而非 Google、ChatGPT、BARD 和 Microsoft Copilots，特别是在内容创作方面。

**提到的链接**：

- [我使用 Perplexity 的频率高于 Google 和 ChatGPT](https://www.youtube.com/watch?v=aphHCBSTx7Q)：该视频的主要内容：“我使用 Perplexity 超过 ChatGPT、BARD 和 Microsoft Copilots 的五个主要原因，包括它在内容创作中的应用……”
- [23 个你不敢相信是免费的 AI 工具](https://www.youtube.com/watch?v=ZYUt4WE4Mrw)：现在，前 500 名使用我链接的人将获得 Skillshare 的一个月免费试用：[https://skl.sh/futurepedia11231After](https://skl.sh/futurepedia11231After) 经过 8 个月的实验……
- [来自 rabbit inc. (@rabbit_hmi) 的推文](https://x.com/rabbit_hmi/status/1748105199418490968?s=46&t=Hug1fgRBxMNq3A5tmmpzqw)：在 rabbit，我们一直在寻找顶级的 AI 服务和合作伙伴，以帮助我们的用户快速准确地完成任务。因此，我们很高兴地宣布与 @perplexity_ai 达成合作伙伴关系，以增强……

### ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/) (6 条消息):

- **对问题解决的感谢**：用户 `@rxiiia` 对 `@830126989687914527` 在解决支付方式问题上的帮助表示感谢，该问题已解决，无需重新创建支付方式。
- **鼓励社区认可**：`@Dyno` 建议使用 ⭐ 表情符号对有帮助的消息做出反应。累积五个星标会将消息发送到 ⭐│starred 频道，并为作者赢得 EXPLORER 角色。
- **请求更具体的说明**：`@dvrshil` 请求更具体的细节或指令，表示目前的帮助不足。
- **信息限制**：`@icelavaman` 直接拒绝了 `@dvrshil` 的请求，声称无法提供所请求的具体信息或细节。
- **功能不在路线图中**：用户 `@icelavaman` 告知 `@dvrshil`，所讨论的功能目前不在开发 Roadmap 上。

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- **Mistral 模型大乱斗**：**模型性能**和**模型训练**成为焦点，讨论范围从最值得使用的 7b **Mistral** 模型（如 [OpenPipe/mistral-ft-optimized-1227](https://huggingface.co/OpenPipe/mistral-ft-optimized-1227) 和 [Bagel 7B](https://huggingface.co/jondurbin/bagel-7b-v0.1)），到 **LoRA/qLoRA** 和 **Axolotl** 等模型中 **sample packing** 的挑战。用户深入探讨了**数据质量和数据集有效性**，建议使用 **RedPajamaV2** 和 **Dolma** 进行模型测试，并强调了 Meta 购入 **600,000 块 Nvidia H100 GPU** 的消息，以说明 **LLaMa 3** 等 AI 领域不断增长的计算规模。
  
- **使用 Axolotl 进行打包与部署**：在 [Axolotl 开发进展](https://github.com/OpenAccess-AI-Collective/axolotl/blob/acfc4ef7ddd15bf85c2feed2142ab7331694dd35/src/axolotl/core/trainer_builder.py#L1033)中，对话集中在更新 `flash-attn` 的**包依赖需求**、**DPOTrainer** 缺乏直接配置以及对包依赖管理的担忧。用户指出 **ColossalAI** 的 **ShardFormer** 是实现简化张量并行（tensor parallelism）的潜在一步，并对 **Unsloth** 关于训练速度和 **VRAM** 效率的**主张**真实性提出了质疑。
  
- **Qlora 与 LoRA 的进展**：有关于实现 **Qlora** 以复现特定研究结果的咨询，以及关于 **Mixtral** 中 **8bit LoRA 微调**已修复 Bug 的疑问。
  
- **数据集利用与清洗对话**：用户对 **oasst1/2 数据集**利用率不足表示惊讶，并分享了使用 **GPT-4** 和 **mistral-medium** 进行**有效数据清洗的策略**。他们讨论了训练 Token 的策略性选择（如 `<BAD>` vs `<GOOD>`），强调了 Token 选择对**模型训练结果**的影响。
  
- **RLHF 沉思录**：**rlhf** 频道的对话探讨了 **input + label + output 训练**方法相较于 DPO 的潜在稳定性，考虑到其在提高模型稳定性方面的效用，并特别提到了该方法在 FAANG 公司的应用。
  
- **Replicate 托管与 API 考量**：**replicate-help** 中的查询涉及该平台是否支持托管，并思考了如何建立到模型的 **API 连接**。
  

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (140 条消息🔥🔥):

- **模型合并者的迷宫 (Model Merger's Labyrinth)**：`@le_mess` 在 **leaderboard** 排名混乱的情况下，询问了关于使用 chatml 格式训练的最佳 7b **Mistral** 模型的建议。`@dreamgen` 和 `@bozoid.` 讨论了各种合并模型，如 **OpenPipe/mistral-ft-optimized-1227** 和 **Bagel 7B** 的独特性，同时对混合 prompt format 训练和数据质量问题表示不满。
- **Sample Packing 的难题**：`@tiendung` 询问了 sample packing 在不同类型模型（如 **LoRA / qLoRA**）中的有效性，而 `@dreamgen` 讨论了 Hugging Face 实现中可能存在的问题，特别是在 attention mask 和 positional encoding 方面。`@tiendung` 和 `@nanobitz` 探讨了 **Axolotl** 是否比 Hugging Face 的方法更正确地实现了 sample packing。
- **数据集重于模型**：`@bozoid.` 表示希望看到模型在 **RedPajamaV2** 和 **AllenAI's Dolma** 等数据集上进行测试。`@bozoid.` 和 `@nruaif` 交流了在大规模数据集上训练的挑战性，以及在不损害性能的情况下缩减模型规模的野心。
- **Meta 算力军备的力量**：`@yamashi`、`@noobmaster29` 和 `@casper_ai` 讨论了 Meta 为训练 **LLaMa 3** 而大规模采购 **600,000 块 Nvidia H100 GPUs** 的举动，强调了最前沿 AI 训练任务所涉及的巨大资源规模。
- **DPO 训练的磨难与考验**：`@c.gato` 和 `@dangfutures` 在应用 **DPO (Decentralized Parallel Optimization)** 时遇到了障碍并分享了经验。他们的对话揭示了在尝试改进模型训练过程中的不确定性和学习时刻。

**相关链接**：

- [Paper page - Self-Rewarding Language Models](https://huggingface.co/papers/2401.10020)：未找到描述
- [Inception Deeper GIF - Inception Deeper Go Deeper - Discover & Share GIFs](https://tenor.com/view/inception-deeper-go-deeper-we-need-to-go-deeper-leonardo-di-caprio-gif-16756828)：点击查看 GIF
- [Supervised Fine-tuning Trainer](https://huggingface.co/docs/trl/sft_trainer#packing-dataset--constantlengthdataset-)：未找到描述
- [jondurbin/bagel-7b-v0.1 · Hugging Face](https://huggingface.co/jondurbin/bagel-7b-v0.1)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/199y05e/zuckerberg_says_they_are_training_llama_3_on/)：未找到描述
- [OpenPipe/mistral-ft-optimized-1227 · Hugging Face](https://huggingface.co/OpenPipe/mistral-ft-optimized-1227)：未找到描述
- [teknium/OpenHermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B)：未找到描述
- [Non Contaminated Packing by nivibilla · Pull Request #1235 · huggingface/trl](https://github.com/huggingface/trl/pull/1235)：正如 #1230 中讨论的那样，我做了一个快速且粗糙的实现。还包含了一个示例 notebook（未测试）。我有空会测试。或者如果你有时间请随时测试，还有任何...
- [Packing in SFT · Issue #805 · huggingface/trl](https://github.com/huggingface/trl/issues/805)：我理解预训练中是如何允许 packing 的，但我正在寻求关于在 SFT 中如何使用 ConstantLengthDataset 对样本进行 packing 的澄清。我看到放置了一个 EOS token...

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (26 messages🔥):

- **`flash-attn` 的需求更新**：用户 `@louist4455` 指出 `flash-attn==2.3.3` 可能已经过时，LLM FT 需要更高版本。`@caseus_` 承认由于缺乏对多 GPU 支持的自动化测试，升级目前是一个手动过程。
- **DPO 清理分支中的配置查询**：`@filippob82` 询问为什么某些参数（如 `max_length` 和 `max_prompt_length`）不能在 `DPOTrainer` 中直接配置。`@caseus_` 表示对于目前使用的大多数架构，这些设置并不关键，但愿意参考 GitHub [脚本](https://github.com/huggingface/trl/blob/928d14445e31b3586ce8b73ca70ecb02dc603369/examples/scripts/dpo.py#L58-L60) 中的示例进行调整。
- **Axolotl 包依赖问题**：`@faldore` 提出了关于防止 Axolotl 安装 `cuda` 和 `torch` 的问题，他们更倾向于独立管理这些依赖。`@caseus_` 提到需要重新考虑为什么将 `bert-score` 添加为依赖项，而 `@nanobitz` 建议在 requirements 中注释掉不需要的安装项。
- **对 ShardFormer 张量并行的兴趣**：`@caseus_` 分享了 ColossalAI 的 ShardFormer 链接，暗示可能更简单地集成 Tensor Parallelism，并指向了 [ShardFormer 项目](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer) 的 GitHub 页面。
- **对 Unsloth 速度宣传的怀疑**：`@nanobitz` 分享了一篇关于 Unsloth 在模型微调性能提升和 VRAM 减少方面的 Reddit 帖子。`@caseus_` 对其营销数据表示怀疑，提到在 3090 GPU 上可以在一小时内完成训练，并澄清 Transformers 实现了 4D 注意力掩码（4d attention masks），而不是 packing 支持。

**提到的链接**：

- [argilla/distilabeled-Hermes-2.5-Mistral-7B · Hugging Face](https://huggingface.co/argilla/distilabeled-Hermes-2.5-Mistral-7B#training-details)：未找到描述
- [Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/19a7vc2/finetune_387_faster_tinyllama_600_faster_gguf/)：未找到描述
- [trl/examples/scripts/dpo.py at 928d14445e31b3586ce8b73ca70ecb02dc603369 · huggingface/trl](https://github.com/huggingface/trl/blob/928d14445e31b3586ce8b73ca70ecb02dc603369/examples/scripts/dpo.py#L58-L60)：使用强化学习训练 Transformer 语言模型。- huggingface/trl
- [ColossalAI/colossalai/shardformer at main · hpcaitech/ColossalAI](https://github.com/hpcaitech/ColossalAI/tree/main/colossalai/shardformer)：让大型 AI 模型更便宜、更快、更易于获取 - hpcaitech/ColossalAI
- [axolotl/src/axolotl/core/trainer_builder.py at acfc4ef7ddd15bf85c2feed2142ab7331694dd35 · OpenAccess-AI-Collective/axolotl](https://github.com/OpenAccess-AI-Collective/axolotl/blob/acfc4ef7ddd15bf85c2feed2142ab7331694dd35/src/axolotl/core/trainer_builder.py#L1033))：尽管向 Axolotl 提问。通过在 GitHub 上创建账户为 OpenAccess-AI-Collective/axolotl 的开发做出贡献。

### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (2 messages):

- **使用 Qlora 的计划**：用户 `@jacques_10431` 提到他们的团队计划利用 **Qlora**，尝试复制某篇特定文章的结果。
- **咨询 8bit LoRA 微调 Bug**：`@jaredquek` 询问了关于 **Mixtral** 中与 **8bit LoRA tuning** 相关的 Bug 更新（相对于 **qLoRA** 或 fft），该问题最初由用户 Caseus 提出。他们很好奇该问题是否已成功修复。

### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (6 messages):

- **OASST 数据集未被充分利用令人惊讶**：`@dreamgen` 对缺乏利用 **oasst1/2 datasets** 的模型表示惊讶，并提到在经过一些过滤后，这些数据集具有很大潜力。
- **深度学习微调咨询**：随后，`@dreamgen` 询问了关于使用 20 个样本进行训练的细节，包括使用 **DPO with QLora**、学习率以及其他具体参数。
- **倡导 GPT-4 数据清洗**：`@dreamgen` 建议投入 **GPT-4** 进行数据清洗，强调了其重要性，并将其与微调和推理的成本进行了对比。
- **通过示例明确目标**：`@dreamgen` 索要示例，以便更好地理解数据清洗的目标。
- **数据清洗与增强的经验**：在反思中，`@dreamgen` 分享到 **mistral-medium** 在某些数据清洗和增强任务中证明有时已经足够，甚至超过了 **GPT-4 Turbo**，而在其他任务中，**GPT-3.5 Turbo** 的表现优于 mistral-medium。
- **认可 GPT-4 的效率**：`.____init___` 同意 `@dreamgen` 的观点，认为使用 **GPT-4** 进行一次性数据清洗过程是合理的。

### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (5 messages):

- **DPO 与基准方法的比较**：`@dreamgen` 讨论了使用 input + label + output 训练的潜在好处，并将其与 Direct Policy Optimization (DPO) 进行对比，认为这是一种更稳定的方法，并提到其在 FAANG 公司的应用。
- **模型训练标签**：`@dreamgen` 解释了使用 `<BAD>` 与 `<GOOD>` 等 tokens 来区分训练数据中不同响应类型的概念，指出在实践中自然 tokens 可能比合成 tokens 更有效。

### ▷ #[replicate-help](https://discord.com/channels/1104757954588196865/1197414694248534116/) (2 messages):

- **托管咨询**：用户 `@dangfutures` 询问该平台是否用于托管用途。
- **API 设置的可能性**：`@noobmaster29` 认为为模型设置 API 是可能的，并考虑稍后进行尝试。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- **智能生态系统中的 AI 集成**：随着 `@.pythagoras` 等人讨论 Samsung S24 等智能手机型号中的 AI 集成，人们的热情不断高涨，并期待未来的 Pixel 手机也能具备类似的 AI 功能。围绕 Apple 生态系统与 Samsung AI 能力的辩论展开，预测 AI 将成为科技领域的默认配置。
  
- **AI 伦理辩论与文档**：AI 社区参与了关于 AI 伦理、治理和对齐（alignment）的讨论，参考了分享的 [arXiv 论文](https://arxiv.org/pdf/2310.07019.pdf) 和关于卫生领域 AI 治理的 [WHO 文档](https://iris.who.int/bitstream/handle/10665/375579/9789240084759-eng.pdf?sequence=1&isAllowed=y)。
  
- **GPT-4 社区贡献与担忧**：`@serenejay` 报告了 GPT Store 的验证问题并咨询隐私选项，同时 `@marcus_73` 为其 HopeGPT 寻求反馈，`@russellsapalmer` 警告开发者不要抄袭 GPT 应用。建议包括通过域名验证来保护隐私，并呼吁 OpenAI 监控此类活动，同时提醒关注 [OpenAI Status](https://status.openai.com) 以获取服务更新。
  
- **Prompt Engineering 策略与交流**：在 AI 模型中使用自定义指令（custom instructions）的体验各不相同；`@realgavok` 认为它们不够稳定，而 `@darthgustav.` 建议使用 XML tagging 来提高 GPT-4 的选择准确性，并分享了一个 [XML tagging 示例](https://discord.com/channels/974519864045756446/1192857176562208798/1192857176562208798) 以提供更清晰的模型引导。
  
- **XML Tagging 成为核心话题**：在 Prompt Engineering 的讨论中，`@darthgustav.` 建议 `@magiciansinc` 使用 XML tagging 以获得比 CSV 或 JSON 更好的效果。示例展示了如何通过使用结构良好的标准来优化 AI 在过滤列表时的性能。
  

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (110 条消息🔥🔥):

- **AI 增强型智能手机引发热潮**：`@.pythagoras` 对集成在三星 S24 等新型号智能手机中的 AI 工具表示出浓厚兴趣，并希望 Google 能在 Pixel 手机中跟进类似功能。其他用户分享了他们的使用体验和偏好，对话随后转向了关于三星与 Apple 优劣的泛泛讨论，以及对 AI 成为智能手机标配功能的期待。
  
- **AI 冰箱幻想激发想象力**：用户 `@.pythagoras` 幽默地预见到未来所有家电都将标榜“AI 能力”，引发了其他用户一系列创意推测，包括会聊天的冰箱以及类似自动售货机的多功能厨房电器。
  
- **AI 伦理与治理讨论**：`@clockrelativity2003` 分享了一篇讨论 AI 与判例法的 [arXiv 论文链接](https://arxiv.org/pdf/2310.07019.pdf)，以及一份关于卫生领域 AI 伦理与治理的 [WHO 文档链接](https://iris.who.int/bitstream/handle/10665/375579/9789240084759-eng.pdf?sequence=1&isAllowed=y)，引发了关于 AI Alignment（AI 对齐）及其影响的回应和讨论。
  
- **Gemini Ultra 发布时间尚不确定**：在关于 “Gemini Ultra” 发布时间的讨论中，`@la42099` 幽默地猜测它可能会在未来 30 天内推出，用户们表达了对更大 Prompt 限制和其他进步的期待。
  
- **技术生态系统辩论**：一场关于 Apple 生态系统和连续互通（Continuity）功能的激烈辩论展开，`@muyfashionista` 赞扬了 Apple 设备间无缝集成的优势。用户 `@mrcrack_` 和 `@darkangel9365` 则就 Android 和三星的功能发表了看法，提到了定制化选项并质疑了 Apple 的应用审批政策。
  

**提到的链接**：

- [老人 GIF - 孩子们 - 发现并分享 GIF](https://tenor.com/view/children-gif-5754160)：点击查看 GIF
- [医疗健康领域人工智能的伦理与治理：大型多模态模型指南](https://iris.who.int/handle/10665/375579)：未找到描述

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (38 条消息🔥):

- **serenejay 的验证困扰**：`@serenejay` 反馈在发布 GPT Store 时无法完成构建者个人资料（builder profile）的问题，尽管尝试了更换浏览器和清除缓存。他们在通过网页端使用卡片订阅后获得了成功，但由于隐私顾虑，询问是否可以不使用真实姓名。
  
- **域名验证作为解决方案**：在遇到 Google Play 验证问题后，`@rjkmelb` 建议 `@serenejay` 通过获取域名来验证其 OpenAI 账户。`@7877` 补充说，使用域名验证可以帮助隐藏真实姓名，转而显示域名。
  
- **HopeGPT 在竞赛中获胜**：`@marcus_73` 分享了他们的 GPT 模型 HopeGPT，该模型在一次以“注入希望”为主题的竞赛中获胜，并征求改进建议；提供了模型链接，并在 `@solbus` 的引导下将其分享到专门频道以增加曝光度。
  
- **关于开发者 GPT 被抄袭的警示**：用户 `@russellsapalmer` 提出了一个严重关切，指责开发者账号 tapgpts 涉嫌抄袭数百名开发者的作品，在未署名的情况下模仿名称、Logo、描述和示例 Prompt，呼吁 OpenAI 监控此类活动。
  
- **ChatGPT 停机与沟通问题**：`@c6565` 质疑为什么 ChatGPT 服务中断没有公开沟通，`@7877` 对此提供了 OpenAI 状态页面的链接，该页面详细列出了运行更新和过去的事故记录。
  

**提到的链接**：

[OpenAI 状态](https://status.openai.com)：未找到描述

### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (16 messages🔥):

- **自定义指令（Custom Instructions）效果参差不齐**：用户 `@realgavok` 观察到**禁用自定义指令**似乎能提高一致性。这引发了讨论，`@darthgustav.` 认为自定义指令的有效性在很大程度上取决于其内容和结构。
  
- **XML 标记提升 GPT-4 的选择准确率**：在给 `@magiciansinc` 的建议中，`@darthgustav.` 推荐**使用 XML 标记**来提升 GPT-4 在根据标准（例如挑选适合热带度假的城市）对列表进行排序时的性能。据称该技术优于使用 CSV 或 JSON 格式。
  
- **Darthgustav. 提供的 XML 标记示例**：为了进一步帮助 `@magiciansinc`，`@darthgustav.` 提供了一个 **XML 标记示例**，列出了各种城市及相关活动，以演示如何利用标记来增强 GPT-4 的输出。
  
- **使用 Discord 链接分享 XML 格式**：作为一种非常规做法，`@darthgustav.` 引导 `@magiciansinc` 通过 Discord 链接查看示例，具体链接为 [https://discord.com/channels/974519864045756446/1192857176562208798/1192857176562208798](https://discord.com/channels/974519864045756446/1192857176562208798/1192857176562208798)，这是所提供协助的一部分。
  
- **继续探索 XML 标记**：`@magiciansinc` 表示打算测试 XML 标记方法，`@darthgustav.` 祝他们好运，展现了 **prompt-engineering** 频道中的协作氛围。
  

### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (16 messages🔥):

- **自定义指令的一致性**：`@realgavok` 询问了自定义指令的有效性，指出禁用它们有时会获得更好的一致性。`@darthgustav.` 回应称，一致性取决于指令的内容和质量。
- **提倡使用自定义 GPTs**：`@darthgustav.` 分享了他们倾向于专门使用自定义指令或自定义 GPTs (Custom GPTs) 的偏好，暗示对其性能感到满意。
- **通过标准增强列表过滤**：`@magiciansinc` 正在寻求关于使用 GPT-4 根据特定标准过滤列表（如城市或产品）的建议。他们反映目前从模型收到的建议和解释效果不佳。
- **XML 标记带来更好结果**：`@darthgustav.` 建议 `@magiciansinc` 使用 XML 标记来注明城市的通用属性，这应该会提升 GPT-4 的性能。他们还强调了正确引导模型的重要性。
- **XML 标记示例**：当 `@magiciansinc` 索要 XML 标记示例时，`@darthgustav.` 提供了一个详细的样本，并根据他们的测试建议这种方式可能比 CSV 或 JSON 表现更好。他们还提到了一个用于生成此类数据的外部来源。

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

**Mistral 7B 的自托管与 API 奇迹**：各频道的讨论显示了对**自托管 Mistral 7B** 并在 Python 应用程序中使用它的兴趣，多位用户提供了协助和工具建议。同时也提出了关于**商业应用数据隐私**以及**量化（quantization）**影响性能的技术问题。

**长文本的困境**：用户讨论了使用 Mistral 处理长文本及 32K Token 限制的问题。虽然文档提到了这一限制，但实际的 Token 上限会根据**模型大小**和**特定任务**条件而有所不同。

**微调（Fine-Tuning）中的挫折与建议**：社区报告了在微调 **Mistral 7B** 时遇到的挑战，例如旧 Prompt 响应的残留以及在 **RTX 4090** 上的 GPU 显存困难。此外，关于在 **HF trainer** 中正确实现 Mistral 以及寻找优质 **GGUF 格式模型**也是咨询的热点。

**关于部署和工具集成的热烈讨论**：参与者交流了集成 **Deep Chat** 等工具的经验，强调了其相比 **Open Copilot** 等复杂设置的简便性。成员们还分享了与开源项目相关的个人经历以及在科技行业的国际搬迁见解。

**对有志编码者的指导及对 LLaMa 的沉思**：对初学编码者的建议指向了**哈佛大学的 CS50** 课程以及通过实践经验学习。Reddit 上关于 Meta AI 的 **LLaMa 3** 正在使用惊人的 **600,000 张 H100** 进行训练的讨论也激起了大家的好奇心。

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (31 条消息🔥):

- **关于 Mistral 7B 私有化部署的咨询**：用户 `@bot7_` 询问是否可以**私有化部署 Mistral 7B** 并将其与 Python 应用配合使用。`@kerunix` 确认这是可行的，而 `@tom_lrd` 指出这取决于用户的操作系统和硬件，并提供了一些相关工具的名称。
  
- **探讨 Mistral 处理长文本的方法**：`@lukasgutwinski` 询问了使用 Mistral 处理长文本（最多 100 页）的最佳方式，以及 Mistral Medium 和 Small 是否都具有 32K token 的上下文窗口。`@i_am_dom` 建议 **Mixtral** 在 16k token 以内表现良好，但超过该阈值后可能不稳定。
  
- **寻求便捷的 Mixtral 聊天方式**：用户 `@rod_____` 想知道是否有办法通过简单插入 API key 来与 **Mixtral** 聊天，`@jortega_17718` 回复了一个 Hugging Face endpoints 的链接。
  
- **关于 Langchain 配合 Mistral API 的说明**：`@western_01` 分享了通过 langchain 在 CrewAI 中成功使用 **Mistral API** 的经验，并纠正了之前的错误，指出默认的 API endpoint 运行完美。
  
- **32K Token 限制确认及注意事项**：`@jortega_17718` 和 `@sublimatorniq` 都讨论了 Mistral 生成端点所谓的 32K token 限制，指出虽然文档是这样写的，但在实际应用中往往达不到，特别是对于较小的模型或特定任务。
  

**提到的链接**：

[mistralai/Mixtral-8x7B-Instruct-v0.1 · Hugging Face](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)：未找到描述

### ▷ #[models](https://discord.com/channels/1144547040454508606/1154028112124846150/) (22 条消息🔥):

- **Mistral 7B 私有化部署挑战**：用户 `@bot7_` 询问如何私有化部署 Mistral 7B 并将其与 Python 应用配合使用，并因其法语背景对英语水平表示歉意。
- **寻找 Mistral 7B API**：`@rohit3389` 开始通过 GPT4All Python 库使用 "Mistral-7b-openorca.q4_0.gguf"，并想知道是否有可以在 Python 中使用的 API。
- **Mistral 模型说明**：`@tom_lrd` 回复称，要使用像 Openorca 这样的特定微调模型需要第三方服务器，因为 Mistral 的官方 API 仅提供 mistral7b-instruct 等模型。
- **理解 LLM 并寻求 API 解决方案**：`@rohit3389` 寻求更快的 API 解决方案，以避免加载 4GB 的重型模型，`@tom_lrd` 建议通过 Mistral 的 API 使用 tiny、small 和 medium 模型，虽然风格不完全一致，但性能相当或更好。
- **关于无限制（Off-Guardrails）模型的建议**：`@dizzytornado` 正在寻找适合编写具有真实、冲突性格角色脚本的 Mistral 模型，而不是快乐、和谐的场景 —— `@vhariational` 推荐了 Dolphin-Mistral 模型，并提供了其 Hugging Face 页面链接和 Discord 邀请以供进一步讨论。

**提到的链接**：

[cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo-laser)：未找到描述

### ▷ #[deployment](https://discord.com/channels/1144547040454508606/1154028168466923600/) (8 条消息🔥):

- **Mistral 7B 私有化部署咨询**：`@bot7_` 询问是否可以**私有化部署 Mistral 7B** 以配合 Python 应用使用。`@akshay_1` 确认这是可行的，并安慰了 `@bot7_` 对其英语水平的担心。
  
- **在部署方面提供帮助**：`@akshay_1` 承认私有化部署 **Mistral 7B** 的复杂性，并提出提供专业支持，请 `@bot7_` 查看私信以获取进一步帮助。
  
- **商业应用中的数据隐私担忧**：`@xxtinction` 对在处理敏感数据的商业应用中使用 **Mistral 7B** 表示担忧，询问数据是否会保持私密，还是会被 Mistral 用于训练。由于对 Mistral 的隐私政策感到困惑，他们请求文档说明。
  
- **Mistral 7B 量化技术问题**：`@lauthu` 提到在 Mistral 7B 上使用 **TensorRT-LLM W8A8 Smooth Quant** 时遇到了精度下降的问题，并询问其他人是否遇到过类似情况。

### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (7 messages):

- **关于 GGUF 格式 Mistral 7B 的咨询**：`@bot7_` 正在寻找一个**优质的 GGUF 格式 Mistral 7B**，但该查询未收到任何回复。
- **HF Trainer 中 Mistral 的问题**：`@bozoid.` 分享了关于 **HF trainer 中 Mistral 实现不正确**的担忧，这影响了微调（finetuning）期间的性能。据 `@andrewwwwme` 称，该问题尚未得到官方解决。
- **Mistral 中持续出现旧 Prompt 的回复**：`@dizzytornado` 报告了一个问题，即 Mistral 总是返回来自**旧 Prompt** 的单词，但尚未获得解决方案。
- **在 RTX 4090 上微调 Mistral 7B 的挑战**：`@kaizen0340` 询问了在 **RTX 4090 上使用 LORA 微调 Mistral 7B** 的经验，并提到了 GPU 显存方面的困难。`@enzodeg40` 回复询问 `CUDA_VISIBLE_DEVICES` 是否配置正确。

### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (104 messages🔥🔥):

- **Deep Chat 的易集成性**：`@ovi8773` 赞赏了 Deep Chat 的集成简便性——只需**一行代码**且无需注册，优于 Open Copilot 等需要更复杂配置的全栈替代方案。
- **Open Copilot 复杂的设置**：与 Deep Chat 的易用性形成对比，`@ovi8773` 评论说 Open Copilot 的设置过程非常繁琐，尽管它是一个具有可定制选项的开源项目。他们认为 Deep Chat 在开发者便利性和实现方面更胜一筹。
- **对项目贡献的赞誉**：Deep Chat 获得了 `@ethux` 的钦佩，他非常喜欢这个项目并在 GitHub 上给了星标（star），分享了对开源贡献的热情。
- **讨论全球技术中心**：对话延伸到了全球生活成本和技术中心的讨论。`@ovi8773` 和 `@ethux` 交流了关于房价、各国吸引力以及荷兰 30% 税收减免（30% tax ruling）等政策的见解。
- **技术领域的个人历程**：`@ovi8773` 分享了从软件工程（Software Engineering）职业生涯中抽身专注于开源项目的个人经历，并考虑移居其他国家。这引发了与 `@ethux` 关于搬迁利弊的讨论，特别是在技术环境和生活水平的背景下。

**提到的链接**：

- [no title found](https://funda.nl,): 未找到描述
- [30% tax ruling in the Netherlands | I amsterdam](https://www.iamsterdam.com/en/live-work-study/living/official-procedures/30-tax-ruling)：前往荷兰的高技术移民可能有资格享受 30% 的税收减免。了解有关福利和要求的所有信息。
- [GitHub - openchatai/OpenCopilot: 🤖 🔥 Let your users chat with your product features and execute things by text - open source Shopify sidekick](https://github.com/openchatai/OpenCopilot)：🤖 🔥 让您的用户通过聊天与您的产品功能互动并执行操作 - 开源的 Shopify 助手 - GitHub - openchatai/OpenCopilot。

### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (4 messages):

- **@fufuespade 寻求初学者编程建议**：用户 `@fufuespade` 询问如何开始学习编程，以及有哪些推荐给初学者的资源（如论坛或 YouTube 频道）。
- **推荐哈佛编程课程**：`@jakobdylanc` 建议初学者查看 YouTube 上的 **CS50**，这是哈佛大学的一门免费课程，拥有详尽的讲座视频。
- **@akshay_1 的实践学习法**：`@akshay_1` 建议 `@fufuespade` 通过直接实现一个想法并获得实践经验来学习编程。
- **关于 Meta LLaMa 的对话**：`@yamashi` 分享了一个 [Reddit 链接](https://www.reddit.com/r/LocalLLaMA/comments/199y05e/zuckerberg_says_they_are_training_llama_3_on/)，讨论了 Meta AI 的大语言模型 **LLaMa**，以及马克·扎克伯格关于在 600,000 块 H100 上训练 **LLaMa 3** 的评论。

**提到的链接**：

[Reddit - Dive into anything](https://www.reddit.com/r/LocalLLaMA/comments/199y05e/zuckerberg_says_they_are_training_llama_3_on/)：未找到描述

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord 总结

- **DeciTech 发布双模型惊喜**：DeciTech 发布了支持八种编程语言的 **DeciCoder-6B**，以及图像生成模型 **DeciDiffusion v2.0**，其速度比 Stable Diffusion v1.5 快 2.6 倍。可以在 [Hugging Face](https://huggingface.co/Deci/DeciCoder-6B) 上探索 DeciCoder-6B，并在 [Colab](https://colab.research.google.com/drive/1QRbuser0rfUiFmQbesQJLXVtBYZOlKpB) 或 [Hugging Face Space](https://huggingface.co/spaces/Deci/DeciCoder-6B-Demo) 进行测试。
  
- **FABBLER.AI 征集创意测试者**：FABBLER.AI 正在为一款创新的叙事故事创作工具寻找 Beta 测试人员，该工具可将故事转换为视频。请在 [YouTube](https://www.youtube.com/watch?v=J4olyiCLLRs) 上查看演示，并在 [Hugging Face Space for Proteus-V0.1](https://huggingface.co/spaces/ehristoforu/Proteus-V0.1) 探索该工具。
  
- **大型模型的 GPU 托管？欧盟用户想知道！**：一位成员正在整理一份位于欧盟的 GPU 托管提供商名单，这些提供商能够支持 13B 到 70B 的模型，用于图像转文本（image-to-text）和邮件分类等任务。该请求要求低延迟和按需使用，讨论中尚未提供具体的提供商或解决方案。
  
- **Phi-2 模型权重，警惕感叹号入侵**：在 **Phi-2 模型**更新后，一位用户在进行 FP16 推理（inference）时遇到了输出全是感叹号的问题，通过切换到 `device_map="auto"` 解决了该问题。遇到类似问题的开发者可以在[此处](https://huggingface.co/microsoft/phi-2/discussions/89)查看详情。
  
- **Computer Vision 中的语法困扰与模型查询**：一些用户在训练模型时遇到了语法错误，通过社区建议得以解决；而另一些用户寻求关于对象追踪的建议，但尚未收到回复。一位处理印度食物数据集的初学者得到了同行关于如何推进的指导。
  

**HuggingFace Discord 频道总结**

### ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/) (1 条消息):

- **DeciTech 发布 DeciCoder-6B 和 DeciDiffusion v2.0**：推出了两个新模型：DeciCoder-6B 支持八种编程语言，在 HumanEval 基准测试中优于竞争对手；DeciDiffusion v2.0 是一款图像生成模型，速度比 Stable Diffusion v1.5 快 2.6 倍。在 [DeciCoder-6B](https://huggingface.co/Deci/DeciCoder-6B) 查看详情，并在 [Colab](https://colab.research.google.com/drive/1QRbuser0rfUiFmQbesQJLXVtBYZOlKpB) 和 [Hugging Face Space](https://huggingface.co/spaces/Deci/DeciCoder-6B-Demo) 中进行尝试。
  
- **加速车辆速度估算**：@SkalskiP 介绍了一个实时车辆速度估算教程，涉及使用 YOLOv8 进行车辆检测、使用 ByteTrack 进行追踪以及距离计算的复杂性。点击[此处](https://www.youtube.com/watch?v=uWP6UjDeZvY)观看教程。
  
- **对抗语言模型中的幻觉**：一项新研究探讨了检测和编辑语言模型输出中的幻觉，并引入了一个检索增强模型 (FAVA)，其表现优于 ChatGPT 和 Llama2 Chat。在[项目网站](https://fine-grained-hallucination.github.io)上了解分类法、模型和演示。
  
- **艺术与 AI：创意伙伴关系**：@fffiloni 撰文阐述了艺术和设计在提升 AI 能力方面的关键作用，鼓励艺术家、设计师和 AI 研究人员之间的合作。在 [Hugging Face Blog](https://huggingface.co/blog/fffiloni/the-critical-role-of-art-and-design-in-advancing-a) 阅读全文。
  
- **与 Lyon NLP Group 一起拥抱法语文本**：`lyon-nlp-group` 将 Massive Text Embedding Benchmark (MTEB) 扩展到法语，助力法语文本嵌入方法的评估和比较。详细分析见[博客文章](https://huggingface.co/blog/lyon-nlp-group/french-mteb-datasets)。
  

**提到的链接**：

- [@harpreetsahota 在 Hugging Face 上发布："✌🏼今天发布了两个新模型 👇🏽 👩🏾‍💻 𝐃𝐞𝐜𝐢𝐂𝐨𝐝𝐞𝐫-𝟔𝐁…"]([https://huggingface.co/posts/harpreetsahota/814290289723145](https://huggingface.co/posts/harpreetsahota/814290289723145))：未找到描述

- [@SkalskiP 在 Hugging Face 上发布："实时车辆速度估算教程 🚗💨💨💨 TL;DR: 观看…"]([https://huggingface.co/posts/SkalskiP/421333989856413](https://huggingface.co/posts/SkalskiP/421333989856413))：未找到描述

- [@s3nh 在 Hugging Face 上发布："GPU 贫民视角：构建一个解决特定任务的 RAG。每个人都喜欢…"]([https://huggingface.co/posts/s3nh/683576905550627](https://huggingface.co/posts/s3nh/683576905550627))：未找到描述

- [@gsarti 在 Hugging Face 上发布："💥 今日推荐——语言模型的可解释性与分析：细粒度…"](https://huggingface.co/posts/gsarti/989501255639069)：未找到描述
- [打破障碍：艺术和设计在提升 AI 能力中的关键作用](https://huggingface.co/blog/fffiloni/the-critical-role-of-art-and-design-in-advancing-a)：未找到描述
- [使用 Aliyun Scheduler 在 Kubernetes 中实现 GPU 分片](https://huggingface.co/blog/NileshInfer/implementing-fractional-gpus-in-kubernetes)：未找到描述
- [将 Massive Text Embedding Benchmark 扩展到法语：数据集](https://huggingface.co/blog/lyon-nlp-group/french-mteb-datasets)：未找到描述
- [释放语言模型中 Logprobs 的力量：实用指南](https://huggingface.co/blog/Andyrasika/logprobs-transformers)：未找到描述
- [E5 - 由 Tonic 提供的 Hugging Face Space](https://huggingface.co/spaces/Tonic/e5)：未找到描述
- [Fast AI Image Upscaler 4x - 由 FumesAI 提供的 Hugging Face Space](https://huggingface.co/spaces/FumesAI/Fast-AI-Image-Upscaler-4x)：未找到描述
- [Andyrasika/VQA-Dataset · Hugging Face 数据集](https://huggingface.co/datasets/Andyrasika/VQA-Dataset)：未找到描述
- [H94 IP Adapter FaceID SDXL - 由 r-neuschulz 提供的 Hugging Face Space](https://huggingface.co/spaces/r-neuschulz/h94-IP-Adapter-FaceID-SDXL)：未找到描述
- [Proteus V0.1 - 由 ehristoforu 提供的 Hugging Face Space](https://huggingface.co/spaces/ehristoforu/Proteus-V0.1)：未找到描述

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (79 条消息🔥🔥):

- **Phi-2 模型权重问题**：用户 `@admin01234` 描述了 **Phi-2 model** 的一个问题：在更新文件后，生成的只有感叹号。提到的一个解决方案是在模型配置中将 `torch_dtype="auto"` 切换为 `device_map="auto"`。该问题及代码片段在 [此论坛帖子](https://huggingface.co/microsoft/phi-2/discussions/89) 中进行了讨论。
  
- **BERT Model Token 限制**：`@redopan706` 询问如何修改 **BERT Model** 的最大 Token 限制，对此 `@stroggoz` 建议他们 *阅读 Hugging Face 上的文档*，指出模型配置详情可以在那里找到。另一位用户 `@vipitis` 建议寻找具有更大上下文尺寸的其他预训练模型，而不是尝试重新训练或插值。
  
- **微调模型位宽问题**：`@samuelcorsan` 寻求关于将模型从 4-bit 转换为 8-bit 量化的建议。与 `@doctorpangloss` 的讨论显示，8-bit 的反向传播可能并不实用，他们建议改用 **bf16** 或 **fp32** 进行 LoRA 训练。
  
- **macOS 上的 AI 生成人像**：`@itscharliecrown` 表示希望使用个人图像训练 AI，并通过 **Stable Diffusion Web UI-UX** 生成人像。作为回应，`@doctorpangloss` 指出了在 macOS 上进行训练的可行性，但警告称其速度远低于支持 CUDA 的平台（如 Windows 或 Linux）。
  
- **Hugging Face 系统故障**：用户 `@theyruinedelise` 和 `@jo_pmt_79880` 报告了 **Hugging Face** 平台故障，遇到了 **504** 错误和网页加载问题，并幽默地暗示“饥饿的仓鼠……在啃电线”可能是导致停机的原因。
  

**提到的链接**：

- [microsoft/phi-2 · New tokens generated with FP16 inference are only exclamation marks "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"](https://huggingface.co/microsoft/phi-2/discussions/89)：未找到描述
- [GitHub - whitead/paper-qa: LLM Chain for answering questions from documents with citations](https://github.com/whitead/paper-qa)：用于从带有引用的文档中回答问题的 LLM Chain - GitHub - whitead/paper-qa: LLM Chain for answering questions from documents with citations

### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (4 条消息):

- **求知者的早睡计划**：用户 `@mastermindfill` 表示感谢，并提到计划在睡觉前**保存提供的链接以便将来使用**。在这些最后的消息中没有讨论具体的链接或主题。

### ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/) (4 条消息):

- **SDXL V2 模型发布**：`@_vargol` 表示 **h94** 已经发布了 **SDXL** 的第 2 版模型，虽然有所改进，但仍需要偏向写实主义。
- **ZavyChromaXL 推荐**：`@meatfucker` 提到在之前的 SDXL 版本上使用 **zavychromaxl models** 效果很好，尽管他们还没有尝试新版本。
- **Zavy 模型的灵活性**：继续讨论中，`@meatfucker` 指出他们成功使用 **zavychromaxl model** 实现了写实和卡通风格的输出。

### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (7 条消息):

- **FABBLER.AI 招募 Beta 测试人员**: `@piotr_fabbler.ai` 正在招募 Beta 测试人员，以试用一款旨在创建可导出为视频的叙事故事的新 AI 工具。感兴趣的用户可以联系 Piotr 以获得独特的叙事体验并提供反馈，此处提供简短的展示视频 [here](https://www.youtube.com/watch?v=J4olyiCLLRs)。
  
- **Proteus-V0.1 在 Hugging Face Spaces 上线**: `@ehristoforu` 分享了在 zerogpu 上运行的新 Hugging Face Space [Proteus-V0.1](https://huggingface.co/spaces/ehristoforu/Proteus-V0.1) 的链接。`@osanseviero` 发表了评论，表示感兴趣并询问了关于 zerogpu 的使用体验。
  
- **对模型改进的好奇**: 用户 `@merve3234` 询问了 `@ehristoforu` 的模型相比之前使用 1.5 的版本是否有改进，表明了对模型开发进展的关注。
  
- **关于显示放大图像的建议**: `@lunarflu` 称赞了 `@ehristoforu` 模型的简洁高效，并建议增加一个并排显示原始图像和放大图像的功能，以便更好地进行对比。
  
- **GitHub 上的 AI 游乐场和模型实验**: `@vishyouluck` 分享了他们的 GitHub 仓库 [vishalmysore/AI](https://github.com/vishalmysore/AI/tree/main)，该仓库作为一个使用不同模型的 AI 示例游乐场。他们邀请其他人探索并分享对仓库内容的看法。
  

**提及的链接**:

- [Proteus V0.1 - a Hugging Face Space by ehristoforu](https://huggingface.co/spaces/ehristoforu/Proteus-V0.1): 未找到描述
- [FABBLER.AI Feature Showcase](https://www.youtube.com/watch?v=J4olyiCLLRs): FABBLER.AI 功能展示
- [GitHub - vishalmysore/AI: Explore the forefront of AI innovation with this dedicated repository, housing cutting-edge examples and implementations. Dive into the latest advancements, stay ahead with groundbreaking applications, and harness the power of state-of-the-art models and techniques. Elevate your understanding of artificial intelligence through hands-on work](https://github.com/vishalmysore/AI/tree/main): 通过这个专用仓库探索 AI 创新的前沿，其中包含尖端的示例和实现。深入了解最新进展，通过突破性的应用保持领先，并利用最先进的模型和技术。通过实践提升你对人工智能的理解。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 条消息):

- **寻求加载动画模型的帮助**: 用户 `@latentspace` 询问是否可以从单个 `.ckpt` 或 `.safetensord` 文件中为新的 Stable Diffusion 版本加载动画模型。作为回应，`@sayakpaul` 建议在 GitHub 上发起讨论，并承诺会邀请相关专家参与咨询。
- **探索大模型的 GPU 托管选项**: `@johntdavies` 正在寻求建议和一份支持托管 13B 到潜在 70B 模型的 GPU 托管商完整清单，用例涵盖消息传递中的 Image-to-Text 和电子邮件分类，且优先选择总部位于欧盟的服务。他目前正在收集数据以创建提案。

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (27 条消息🔥):

- **语法错误**: 用户 `@swetha98` 在尝试训练 donut docvqa 模型时遇到错误，并分享了 traceback 日志。`@gugaime` 指出代码字符串中可能存在多余反斜杠 (`\`) 的拼写错误，并建议添加空格。
  
- **计算机视觉中的目标追踪**: `@curiousbro` 询问是否有好的 Python 计算机视觉模型用于追踪目标和收集数据，但在提供的消息记录中未收到回复。
  
- **Notebook 故障排除历程**: `@xeus69` 在运行 Notebook 和安装 `accelerate` 时遇到问题，`@meatfucker` 强调了这一细节并建议查看错误消息并确保安装了正确的版本。在 `@xeus69` 清除 Notebook 缓存后，问题最终得到解决。
  
- **初次涉足机器学习**: 新手 `@xeus69` 提到自己是初学者，并在使用 Colab 进行机器学习的初步尝试中得到了 `@meatfucker` 的帮助。讨论显示 `@xeus69` 正在处理与印度食物相关的内容，这是 `@meatfucker` 从输出目录中推断出来的。
  
- **字幕生成模型讨论**: `@merve3234` 质疑了 `@xeus69` 选择字幕生成模型而非像 KOSMOS-2 这样更具 Grounded 特性的模型，暗示了对字幕准确性的需求，这对于文档理解任务非常重要。记录中没有 `@xeus69` 对此询问的回应。

### ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/) (2 条消息):

- **Docker 中 Transformers 的缓存配置**：`@asprtnl_50418` 提供了一个代码片段，介绍如何通过设置 `TRANSFORMERS_CACHE` 环境变量来更改 Docker 容器内 Transformers 的缓存目录。他们还包含了关于在启动 Docker 容器时如何挂载卷（volume）以将本地缓存链接到容器缓存的说明。

### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (3 条消息):

- **关于加载动画 .ckpt 模型的咨询**：`@latentspace` 询问是否可以从单个 `.ckpt` 或 `.safetensord` 文件加载动画模型，并提到了用于 animatediff pipeline 的 **SD v15 和 SDXL** 版本，但未提供有关其设置或背景的更多细节。
- **GitHub 讨论建议**：`@sayakpaul` 回复了 `@latentspace`，建议在 GitHub 上发起讨论并提供了一些链接，以便他们可以标记相关的贡献者来协助解决该问题。
- **寻找 GPU 托管选项**：`@johntdavies` 寻求有关 GPU 托管服务的**讨论组或线程**建议，特别是针对在欧盟（EU）运行 13B 以及可能的 70B 模型，需求涵盖了从消息传递中图像转文本的低延迟到电子邮件分拣和回复的按需使用。他们还在寻求一份公司名单以准备提案。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 摘要

- **DougDoug 的 AI 喜剧发布**：讨论指出，YouTuber DougDoug 使用 ChatGPT 以及 ElevenLabs 的语音生成功能创建了一个带有 NSFW 元素的 AI 角色。这种以 AI 为中心的喜剧方法通过他在 [GitHub](https://github.com/DougDougGithub/Babagaboosh) 上的开源项目得以实现。
  
- **AI 讽刺法律恐慌**：一项具有争议性的《禁止 AI 欺诈法案》（No AI FRAUD Act）被认为可能违宪，引发了关于其对讽刺和喜剧类 AI 内容重大影响的讨论。Reason.com 的一篇文章提供了一份关于其影响的信息性[细目分类](https://reason.com/2024/01/17/ai-fraud-act-could-outlaw-parodies-political-cartoons-and-more/)。
  
- **语言规则语言学争论**：辩论了词典构建中规范性与描述性语言的角色，结论是词典被视为语言使用的历史记录，而非规则强制实体。
  
- **用 AI 之眼进行视频超分辨率**：针对视频超分辨率（Upscaling）对时序感知模型（temporally-aware models）的需求展开了技术讨论，涉及帧细节不一致等问题，并引用了 OpenModelDB 作为资源。
  
- **用于 TTS 的 WhisperSpeech 翻转**：重点介绍了通过反转 OpenAI 的 Whisper 模型来创建开源文本转语音系统 WhisperSpeech，并附带了相关的 GitHub [仓库](https://github.com/collabora/WhisperSpeech)。此外，关于多语言 LLM 评估的讨论以及对连续 Token 嵌入方法论文的搜索，表明了正在进行的调研和进展。
  
- **通过 SSM 解锁 Visual Vim**：一篇新的 arXiv 论文介绍了一种使用双向 Mamba 模块的视觉骨干网络 Vim，详见[此处](https://arxiv.org/abs/2401.09417)；而另一项详细研究则探讨了 LLM 的性能分析，详见[此处](https://arxiv.org/abs/2401.08671)。
  

**LAION 频道摘要**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (78 条消息🔥🔥):

- **DougDoug 的 AI 机器人直播 - 实现原理**：`@ignizherz` 等人讨论了 YouTuber DougDoug 如何创建一个带有 NSFW 元素的 AI 角色（据他称使用了 ChatGPT）。提到 DougDoug 使用了 OpenAI API 以及 ElevenLabs 进行语音合成，并且他已经在 [GitHub 上开源了一个类似项目](https://github.com/DougDougGithub/Babagaboosh)。
- **恶搞与 AI 面临困境**：多位用户（如 `@thejonasbrothers`、`@chad_in_the_house` 和 `@.undeleted`）对可能违宪的 "No AI FRAUD" 法案表示担忧和批评，该法案可能会严重限制基于第一修正案权利的恶搞和喜剧内容。分享了一篇来自 reason.com 的 [文章](https://reason.com/2024/01/17/ai-fraud-act-could-outlaw-parodies-political-cartoons-and-more/)，讨论了与该拟议监管相关的风险。
- **语言规则辩论**：`@mkaic`、`@clock.work_` 和 `@atlasunified` 等用户就规定性（prescriptive）与描述性（descriptive）语言规则展开了长时间辩论。对话探讨了语言的流动性和词典的作用，最终达成共识：词典是描述性的记录，而非规定性的法律。
- **AI 视频超分辨率（Upscaling）讨论**：`@realz` 询问了关于在不导致帧细节不一致的情况下进行视频超分辨率的合适工具，这引发了与 `@pseudoterminalx` 关于时序感知（temporally-aware）超分辨率模型的需求以及视频转码技术方面的讨论。分享了有关可用超分辨率模型的链接和信息，包括对时序因素的考量。
- **为新语言训练 WhisperSpeech**：`@__._astro_.__` 询问了在一种新语言上训练 WhisperSpeech 的要求，指出了当前支持存在的问题以及与英语相比更高的 WER（词错误率）。频道内未提供关于此类训练所需音频小时数的具体细节或估算。

**提到的链接**：

- [Soumith Chintala (@soumithchintala) 的推文](https://fxtwitter.com/soumithchintala/status/1748074223187173724)：终于可以公开谈论一些 GPU 数量了 🙃 到今年年底，Meta 将拥有 60 万张 H100 等效 GPU。尽管猜猜哪些已经部署并投入使用了 😉！
- [Is 'Irregardless' a Real Word?](https://www.merriam-webster.com/grammar/is-irregardless-a-real-word-heh-heh)：哈哈，看看你现在的表情。
- [Nerd Nerd Emoji GIF - Nerd Nerd Emoji Submarine - Discover & Share GIFs](https://tenor.com/view/nerd-nerd-emoji-submarine-location-echolocation-gif-27080631)：点击查看 GIF
- [AI fraud act could outlaw parodies, political cartoons, and more](https://reason.com/2024/01/17/ai-fraud-act-could-outlaw-parodies-political-cartoons-and-more/)：该法案范围广泛，足以针对讽刺特朗普的《周六夜现场》短剧、泰勒·斯威夫特的喜剧模仿，或者 ChatGPT 生成的安·兰德的怪异图像。
- [Sassy Justice Sassy Trump GIF - Sassy Justice Sassy Trump Reindeer Election - Discover & Share GIFs](https://tenor.com/view/sassy-justice-sassy-trump-reindeer-election-sassy-christmas-donald-trump-gif-19541431)：点击查看 GIF
- [Peggle Speedrun, but an Ai Robot threatens me with trivia](https://www.youtube.com/watch?v=HyqK2Tsujho)：我是史上最聪明的 YouTuber。在 Twitch 直播！[https://www.twitch.tv/dougdougFull](https://www.twitch.tv/dougdougFull) 完整直播录像：[https://www.youtube.com/watch?v=E8-qFR](https://www.youtube.com/watch?v=E8-qFR)_...
- [Sassy Justice with Fred Sassy (Full Episode) | Deep Fake and Deep Fake: The Movie](https://www.youtube.com/watch?v=9WfZuNceFDM)：由 Deep Fake 和 Deep Fake: The Movie 为您呈现，Fred Sassy 是一位美国消费者权益倡导者，也是当地电视台 Cheyenne 9 点新闻的记者...
- [OpenModelDB](https://openmodeldb.info/?t=video-frame)：OpenModelDB 是一个社区驱动的 AI 超分辨率模型数据库。我们的目标是提供比现有来源更好的模型查找和比较方式。
- [GitHub - DougDougGithub/Babagaboosh: App that lets you have a verbal conversation with OpenAi's GPT 4](https://github.com/DougDougGithub/Babagaboosh)：让你能与 OpenAI 的 GPT 4 进行语音对话的应用 - GitHub - DougDougGithub/Babagaboosh: App that lets you have a verbal conversation with OpenAi's GPT 4

### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (5 条消息):

- **用于文本转语音的 Whisper 逆向工程**：`@helium__` 分享了一个名为 [WhisperSpeech](https://github.com/collabora/WhisperSpeech) 的 GitHub 仓库，这是一个通过逆向 Whisper 构建的开源文本转语音（TTS）系统。
  
- **关于使用 SSMs 的视觉骨干网络新论文**：`@thejonasbrothers` 提供了一篇 [arXiv 论文](https://arxiv.org/abs/2401.09417)的链接，讨论了一种名为 **Vim** 的新视觉骨干网络，它使用双向 Mamba 模块进行图像序列表示，并在各种任务上实现了高性能。
  
- **LLM 性能分析论文作者确认**：在 `@thejonasbrothers` 的另一条消息中，他们分享了一篇由多人合著的 [arXiv 论文](https://arxiv.org/abs/2401.08671)，展示了他们与长文本语言模型（LLMs）相关的研究工作。
  
- **关于连续 Token Embedding 论文的咨询**：`@JH` 寻求帮助寻找一篇研究 LLM 中连续 Token Embedding（相对于离散 Token Embedding）的论文。
  
- **多语言 LLM 的评估方法**：`@alyosha11` 提出了一个问题，即在缺乏现有数据集的情况下，哪些评估方法对多语言 LLM 是有意义的。
  

**提到的链接**：

- [Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model](https://arxiv.org/abs/2401.09417)：最近，具有高效硬件感知设计的状态空间模型（SSMs），即 Mamba，在长序列建模方面显示出巨大潜力。构建纯粹的高效且通用的视觉骨干网络...
- [DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference](https://arxiv.org/abs/2401.08671)：随着大型语言模型（LLMs）渗透到各种应用中，其部署和扩展变得至关重要，这需要高吞吐量和低延迟的服务系统。现有框架...
- [GitHub - collabora/WhisperSpeech: An Open Source text-to-speech system built by inverting Whisper.](https://github.com/collabora/WhisperSpeech)：一个通过逆向 Whisper 构建的开源文本转语音系统。- GitHub - collabora/WhisperSpeech。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 总结

- **Hypernets，效率的新前沿？**：随后展开了关于使用 **hypernets** 记忆混合专家模型（MoE）中的权重矩阵，以可能**减少参数**并提高效率的讨论，尽管目前尚未分享结论性结果。
  
- **亚马逊以资源助力 LLM 研究**：分享了亚马逊通过 [Amazon Research Awards](https://www.amazon.science/research-awards/program-updates/amazon-research-awards-issues-winter-2024-call-for-proposals) 征集提案的消息，提供**资助和 AWS 额度**以支持 LLM 项目，这并非纯粹的推广。
  
- **跨语言评估 LLM**：对话强调了非英语语言中的 **tokenization 问题**，以及缺乏评估多语言 LLM 的数据集（目前主要使用 BLEU 作为指标）。还提到了 **Self-Rewarding Language Models** 论文以及 *Self-Rewarding* 方法，该方法推动语言模型超越了现有系统。
  
- **HELM 与 Evaluation Harness 的区别详解**：澄清了 **HELM** 与 **evaluation harness** 之间的区别——evaluation harness 处理编排问题，而 HELM 概述了评估的方法论。此外，还就如何在 **eval-harness** 框架内组织**翻译评估任务**寻求建议，这些任务可以放置在 `tasks/translations/` 目录下。
  
- **GPT-NeoX 开发者的 Pull Request 提醒**：在 **gpt-neox-dev** 频道中，一个 [pull request](https://github.com/EleutherAI/gpt-neox/pull/1125) 被重点提及，该 PR 修复了 Docker 容器中的默认设置以及 evaluate 函数的一个单元测试。计划更新 `apex` 以获得更好的 **Python** 和 **PyTorch** 兼容性，尽管其构建时间需要优化。
  
- **机器人进展与鼓励公众参与**：分享了 **Robot Kyle 2a0a** 的训练更新——目前已达到 1.4 亿步——并邀请社区成员通过访问[源代码](https://github.com/cat-game-research/NekoCatGame/tree/main/RagdollTrainer)来训练他们自己的版本。参与者可以在 [YouTube](https://youtube.com/live/mcXqta_5X-Y?feature=share) 上观看 Kyle 的实时训练课程。
  

**Eleuther 频道总结**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (28 messages🔥):

- **探索 Hypernet 效率**：用户 `@Hawk` 和 `@stellaathena` 针对在 Mixture of Experts (MoE) 场景中使用 Hypernet 来记忆权重矩阵进行了简短讨论，旨在潜在地减少参数并提高效率。
- **Amazon 推动 LLM 研究**：来自 Amazon 的 `@desik_agi` 分享了通过 [Amazon Research Awards](https://www.amazon.science/research-awards/program-updates/amazon-research-awards-issues-winter-2024-call-for-proposals) 发起的提案征集，为 LLM 项目提供资助和 AWS 促销额度，并澄清这并非推销，而是为寻求算力资源的人提供的机会。
- **Triton 自定义后端咨询**：用户 `@gabriel_syme` 正在询问是否有人具有为 Triton 设置自定义后端服务器的经验。
- **LM Evaluation Harness 查询**：`@hamelh` 正在寻求关于利用 *eval harness* 来确定哪些任务需要 logprobs 的帮助，并提供了一个 [GitHub 搜索链接](https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+%22output_type%3A+generate_until%22+language%3AYAML+path%3A%2F%5Elm_eval%5C%2Ftasks%5C%2F%2F&type=code) 以辅助理解。
- **多语言 LLM 评估讨论**：用户 `@alyosha11` 和 `@catboy_slim_` 正在探讨用于测试 LLM 多语言能力的评估指标和数据集，BLEU 被确定为标准，但目前数据集仍以英文为主。

**提到的链接**：

[Build software better, together](https://github.com/search?q=repo%3AEleutherAI%2Flm-evaluation-harness+%22output_type%3A+generate_until%22+language%3AYAML+path%3A%2F%5Elm_eval%5C%2Ftasks%5C%2F%2F&type=code)：GitHub 是人们构建软件的地方。超过 1 亿人使用 GitHub 来发现、fork 并为超过 4.2 亿个项目做出贡献。

### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (26 messages🔥):

- **非英语语言的 Tokenization 难题**：`@xylthixlm` 强调了非英语语言在 Tokenization 方面面临的挑战，特别关注中日韩 (CJK) 语言，`@stellaathena` 对此表示认同。
  
- **时间感知型 LLM 之谜**：`@bluerune` 引用了一篇未具名的论文或研究，该研究根据一条显示具有统计学意义结果的推文建议，当 LLM “认为”现在是 12 月而不是 5 月时，可能会生成更短的 Token 输出。
  
- **AlphaGeometry：AI 超越人类数学家**：`@the_alt_man` 分享了一篇关于 AlphaGeometry 的 [DeepMind 博客文章](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)，这是一个能以人类奥数金牌得主水平成功解决复杂几何问题的 AI 系统。
  
- **Self-Rewarding Language Models**：`@pizza_joe` 介绍了一篇关于 Self-Rewarding Language Models 的论文，概述了一种语言模型使用 LLM-as-a-Judge 提示词进行自我奖励的方法，其性能超过了许多现有系统。这一观点引发了由 `@xylthixlm` 等人发起的讨论，探讨了 LLM 在拥有正确调优算法的情况下，是否具备获得更高性能的充足信息。
  
- **指令微调 (Instruction Tuning) 的悖论**：`@catboy_slim_` 和 `@fern.bear` 辩论了 LLM 在微调过程中的信息保留概念，重点在于这究竟是真正的信息丢失，还是未能具体引导模型的输出。`@catboy_slim_` 提到 LoRA 权重是一种可能减轻微调过程中信息丢失的技术。
  

**提到的链接**：

- [Self-Rewarding Language Models](https://arxiv.org/abs/2401.10020)：我们假设要实现超人类 Agent，未来的模型需要超人类反馈以提供充足的训练信号。目前的方法通常根据人类偏好训练奖励模型……
- [AlphaGeometry: An Olympiad-level AI system for geometry](https://deepmind.google/discover/blog/alphageometry-an-olympiad-level-ai-system-for-geometry/)：我们的 AI 系统超越了最先进的几何问题处理方法，推进了 AI 在数学领域的推理能力。
- [来自 Rob Lynch (@RobLynch99) 的推文](https://fxtwitter.com/RobLynch99/status/1734278713762549970)：@ChatGPTapp @OpenAI @tszzl @emollick @voooooogel 惊人的结果。通过 API 调用的 gpt-4-turbo 在“认为”是 12 月时生成的回复比“认为”是其他月份时（在统计学上显著地）更短……

### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (1 messages):

jsai_51448: 什么是 mech interp、concept interp 和 dev interp？

### ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/) (9 messages🔥):

- **寻求关于 Harness 与 Helm 的澄清**：`@aloo_kachalu` 发起了关于 **evaluation harness** 与 **HELM** (Holistic Evaluation of Language Models) 对比的讨论，引发了对其功能和背后设计哲学的探讨。
  
- **解开 HELM 的困惑**：`@stellaathena` 澄清说，**evaluation harness** 侧重于在各种模型上运行评估任务的编排（orchestration）问题，而 **HELM** 则提倡一套用于执行评估的推荐方法论。
  
- **评估希腊语模型**：`@zoulr` 分享了他们使用从 ARC 等英文任务翻译而来的希腊语任务评估模型的经历，并就 **eval-harness** 仓库中特定语言任务的首选目录格式寻求建议。
  
- **组织翻译后的评估任务**：`@hailey_schoelkopf` 建议将翻译后的任务组织在 **eval-harness** 任务部分的特定翻译目录下，并提出了 `tasks/translations/` 或 `arc_multilingual/` 等方案。
  
- **分享特定的 GitHub Pull Request**：`@hailey_schoelkopf` 发布了一个关于在 **eval-harness** 仓库中将 `datasets` 依赖项固定在 2.15 版本的特定 GitHub Pull Request 链接：[Pin `datasets` dependency at 2.15](https://github.com/EleutherAI/lm-evaluation-harness/pull/1312)。
  

**提及的链接**：

- [Stanford Center for Research on Foundation Models](https://github.com/stanford-crfm/)：斯坦福基础模型研究中心（Stanford Center for Research on Foundation Models）拥有 17 个可用仓库。在 GitHub 上关注他们的代码。
- [GitHub - stanford-crfm/helm: Holistic Evaluation of Language Models (HELM), a framework to increase the transparency of language models (https://arxiv.org/abs/2211.09110). This framework is also used to evaluate text-to-image models in Holistic Evaluation of Text-to-Image Models (HEIM) (https://arxiv.org/abs/2311.04287).](https://github.com/stanford-crfm/helm)：Holistic Evaluation of Language Models (HELM)，一个旨在提高语言模型透明度的框架 ([https://arxiv.org/abs/2211.09110](https://arxiv.org/abs/2211.09110))。该框架也用于在 Holistic Evaluation of Text-to-Image Models (HEIM) 中评估文本生成图像模型 ([https://arxiv.org/abs/2311.04287](https://arxiv.org/abs/2311.04287))。
- [Pin `datasets` dependency at 2.15 by haileyschoelkopf · Pull Request #1312 · EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness/pull/1312)：似乎许多用户在升级到 `datasets` 2.16 及以上版本时遇到错误，且由于 Hugging Face Hub 上的数据集正被后台替换为 Parquet 格式。我们……

### ▷ #[multimodal-general](https://discord.com/channels/729741769192767510/795089627089862656/) (1 messages):

- **Robot Kyle 散步**：`technosourceressextraordinaire` 分享了 **Robot Kyle 2a0a** 正在平地上进行冷却训练运行的更新，这可能会改善其在坡道上的运动表现。他们提到运行步数为 2000 万步，最终将达到 1.4 亿步，并邀请其他人访问源代码并在 [NekoCatGame/RagdollTrainer](https://github.com/cat-game-research/NekoCatGame/tree/main/RagdollTrainer) 训练自己的 Kyle 版本。
  
- **欢迎观看训练直播**：提供了 Robot Kyle 2a0a 的实时训练课程，展示了如何使用 Unity Machine Learning Agents 训练机器人步行者，可在 YouTube 上的 [Live AI Robot Training](https://youtube.com/live/mcXqta_5X-Y?feature=share) 观看。
  

**提及的链接**：

- [NekoCatGame/RagdollTrainer at main · cat-game-research/NekoCatGame](https://github.com/cat-game-research/NekoCatGame/tree/main/RagdollTrainer)：一个关于 catifu 的游戏。通过在 GitHub 上参与 cat-game-research/NekoCatGame 的开发做出贡献。
- [💻 Unity 2024 ML-Agents | Live AI Robot Training | Kyle 2a0a | PyTorch | Part 11](https://youtube.com/live/mcXqta_5X-Y?feature=share)：在本视频中，我将向你展示如何使用 Unity Machine Learning Agents 工具训练机器人步行者在敌对环境中与其他步行者协作……

### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (2 messages):

- **修复承诺**：`@catboy_slim_` 确认了一个需要的修复，并表示打算尽快处理，但未具体说明是什么问题。
- **Pull Request 中的微小更改和修复**：`@catboy_slim_` 强调了一个 [pull request](https://github.com/EleutherAI/gpt-neox/pull/1125)，其中包括一些*微小更改*，例如 Docker 容器的默认输出，以及针对 evaluate 函数的单元测试修复。
- **尝试优化 Apex**：`@catboy_slim_` 正寻求更新 `apex` 版本，以确保与新版本的 Python 和 PyTorch 兼容。然而，`apex` 的构建时间过长是一个挑战，`@catboy_slim_` 计划通过在一个 fork 中对其进行精简来解决这个问题。

**提到的链接**：

[Minor changes by segyges · Pull Request #1125 · EleutherAI/gpt-neox](https://github.com/EleutherAI/gpt-neox/pull/1125)：更改了 Docker 容器的默认输出；重命名了 Docker Pythia 配置以表明其为 Docker Pythia 配置；修复了 evaluate 函数的单元测试。

---

## [LlamaIndex Discord](https://discord.com/channels/1059199217496772688) Discord 总结

- **RAG 在 LlamaIndex 黑客松中获得强力助推**：LlamaIndex 为其 RAG-A-THON 黑客松设立了 **$8,000 奖金池**，围绕 Retriever-Augmented Generation 发起竞赛，敦促参与者[注册参加该活动](https://t.co/j33mXMctJV)。活动将于 2 月 2 日至 4 日在加州圣克拉拉的 DataStax 总部举行。
  
- **新课程预警！为 LlamaIndex 学习投票**：LlamaIndex 打算制作一门在线课程，并正在进行投票以确定社区感兴趣的主题。社区成员可以在 [Twitter 投票](https://twitter.com/llama_index/status/1748035774183067750)中表达他们的偏好。
  
- **通过高级查询解锁 RAG 的全部潜力**：LlamaIndex 建议利用查询理解层来增强 Retriever-Augmented Generation (RAG)；建议的改进包括 HyDE 和迭代推理等技术。有关改进 RAG 的更多细节可以在其 [Twitter 线程](https://twitter.com/llama_index/status/1748147811944984728)中进一步探索。
  
- **社区工程师应对 LlamaIndex 的技术挑战**：从处理大型 PDF 的有效方法、在获取节点中使用元数据的复杂性，到关于 Azure Key/LlamaIndex 集成的技术建议，以及总结长文档的策略——工程师们分享了各个主题的指导。值得注意的贡献包括 `@whitefang_jr` 的 [元数据查询方法](https://github.com/run-llama/llama_index/blob/fcfab6486bc6a0eec31a983dd3056ef9cbe8ceb2/llama_index/vector_stores/postgres.py#L102) 和 `@cheesyfishes` 关于将 Azure key 与 LlamaIndex 集成的建议，详见其[文档](https://docs.llamaindex.ai/en/stable/examples/customization/llms/AzureOpenAI.html)。
  
- **使用 AI 模型处理大文档**：在寻求高效处理大型文档以创建目录和摘要，同时确保隐私的过程中，`@takuma.fusioncloud.ai` 寻求了社区帮助。`@greyman_007` 建议探索 **Zephyr 模型**，尽管未提供具体资源。
  

**LlamaIndex Discord 频道总结**

### ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/) (4 messages):

- **探索可组合检索**：LlamaIndex 讨论了高级检索系统中的可组合层级概念。[推文](https://twitter.com/llama_index/status/1748019272679649386)解释了将较短文本链接到较大文本作为检索过程的一部分。
  
- **LlamaIndex 课程兴趣调研**：LlamaIndex 正在考虑创建一门在线课程，并正在投票调查用户最想学习的重要主题。[参与投票](https://twitter.com/llama_index/status/1748035774183067750)或在回复中进一步说明。
  
- **$8,000 RAG-A-THON 黑客松公告**：LlamaIndex 宣布将其首场专注于 Retriever-Augmented Generation 技术的线下黑客松奖金翻倍至 $8,000。[注册参加活动](https://t.co/j33mXMctJV)，并注意至少有一名团队成员必须在 2 月 2 日至 4 日期间出现在加州圣克拉拉的 DataStax 总部。
  
- **通过高级查询转换增强 RAG**：LlamaIndex 建议通过加入查询理解层来改进 Retriever-Augmented Generation (RAG)，提到了 HyDE、子问题分解、迭代推理或路由等技术。[了解更多关于改进 RAG 的信息](https://twitter.com/llama_index/status/1748147811944984728)。
  

**提到的链接**：

[LlamaIndex RAG Hackathon (仅限线下)](https://t.co/j33mXMctJV)：超越聊天机器人：释放 AI Agent 的潜力

### ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/) (34 messages🔥):

- **PDF 页面来源追踪**：`@whitefang_jr` 建议 `@alvarojauna` 通过打印 `response.source_nodes` 来在 metadata 中定位页码信息，以处理大型 PDF 查询。
  
- **通过 Metadata 查询获取节点**：`@whitefang_jr` 回复 `@vozervn` 建议使用 `docstore`。随后的交流暗示在 PGVector 中通过 metadata 检索特定节点存在困难，但 `@whitefang_jr` 最终链接到了 LlamaIndex GitHub 仓库的相关部分以提供进一步指导。
  
- **关于 LlamaIndex 高级问答工具的协助**：`@risk_seeking` 询问了有关在 LlamaIndex 文档上进行问答的第三方工具，并寻求社区推荐。
  
- **Azure 密钥与 LlamaIndex 的集成**：`@cheesyfishes` 通过参考文档并建议使用 `AzureOpenAI` 以及可能用于 Header 管理的自定义 httpx 客户端，帮助 `@zubeen_` 解决了 Azure 提供的 OpenAI 密钥与 LlamaIndex 集成的问题。
  
- **长文档摘要的挑战**：`@ben25635` 寻求对一份 500 页综合报告进行摘要的指导，`@nerdai` 建议采用分层方法，先进行分段摘要，然后再构建顶层摘要。
  

**提到的链接**：

- [Azure OpenAI - LlamaIndex 🦙 0.9.33](https://docs.llamaindex.ai/en/stable/examples/customization/llms/AzureOpenAI.html)：未找到描述
- [LLM Prompt FORMATS make or break you LLM (RAG)](https://www.youtube.com/watch?v=M5i3rQfEw_A)：LLM Prompt 格式化本质上涉及输入数据或问题在提交给 LLM 或 VLM 时的结构化和呈现方式。LLM 对...的敏感性。
- [llama_index/llama_index/vector_stores/postgres.py at fcfab6486bc6a0eec31a983dd3056ef9cbe8ceb2 · run-llama/llama_index](https://github.com/run-llama/llama_index/blob/fcfab6486bc6a0eec31a983dd3056ef9cbe8ceb2/llama_index/vector_stores/postgres.py#L102)：LlamaIndex（原 GPT Index）是适用于你的 LLM 应用程序的数据框架 - run-llama/llama_index

### ▷ #[ai-discussion](https://discord.com/channels/1059199217496772688/1100478495295017063/) (3 messages):

- **寻求使用 ChatGPT 处理大文档的帮助**：`@takuma.fusioncloud.ai` 正在寻求关于如何利用 ChatGPT 处理大文档以创建目录和摘要，以及如何维护 10-12 本书集合的隐私方面的帮助。
- **建议使用 Zephyr 模型处理大文档**：`@greyman_007` 建议在 Google Colab 上结合 LlamaIndex 使用 **Zephyr 模型** 来处理 `@takuma.fusioncloud.ai` 提到的任务，但未提供更多细节或链接。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

- **AlphaCodium 在 GitHub 上亮相**：[**AlphaCodium**](https://github.com/Codium-ai/AlphaCodium) 是一款受 DeepMind AlphaCode 启发的开源代码生成工具，现已在 GitHub 上[发布](https://x.com/itamar_mar/status/1747957348293824676?s=20)，并在[专门的论文](https://arxiv.org/abs/2401.08500)中详细介绍了其流程工程（flow engineering）。
- **Karpathy 的认可与 YouTube 见解**：Andrej Karpathy 审查并认可了 AlphaCodium 的能力，更多见解可以从 [**AI Explained YouTube 视频**](https://youtu.be/dOplrIJEYBo?si=fPomG0jy4BDBkbC_)中获得。
- **关于 AlphaCodium IDE 插件的查询**：讨论中包括一个关于 AlphaCodium IDE 插件开源状态的问题，该插件被注明采用 Apache 2.0 许可。
- **Meta 的大规模 GPU 部署计划**：Meta 公布了其目标，即在今年年底前部署相当于 60 万个 H100 GPU 的算力；对话涉及了 GPU 的可用性，并提醒通过 [Tweet 链接](https://x.com/soumithchintala/status/1748074223187173724?s=46&t=6FDPaNxZcbSsELal6Sv7Ug) 邀请一位关键参与者加入讨论。
- **推荐 Gradient Dissent 播客以获取 LLM 见解**：对于那些对 **LLM 训练**和部署感兴趣的人，`@swyxio` 重点推荐收听 [Gradient Dissent 的一期播客](https://overcast.fm/+Y_EFBYrkg)，嘉宾是 EleutherAI 的 Stella Biderman。

**Latent Space 频道摘要**

### ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/) (27 messages🔥):

- **AlphaCodium 发布**：`@itamar_mar` 宣布正式推出 AlphaCodium，这是一款受 DeepMind 的 AlphaCode 启发、在编程竞赛中具有竞争力的开源代码生成工具，并[邀请用户提问](https://x.com/itamar_mar/status/1747957348293824676?s=20)。该项目已在 [GitHub 上发布](https://github.com/Codium-ai/AlphaCodium)。
- **论文讨论与咨询**：`@slono` 参与了关于 AlphaCodium 相关论文的讨论，探讨了 Prompt Engineering 的程度以及在优化 Agent 步骤上投入的精力，得到了 `@itamar_mar` 的[回复](https://arxiv.org/abs/2401.08500)，称 85% 的精力都投入到了流程设计（Flow Design）中。
- **技术社区焦点**：`@itamar_mar` 分享了 Andrej Karpathy 评测他们 AlphaCodium 工作的好消息，`@swyxio` 表示祝贺，并分享了 [Karpathy 的 Twitter](https://fxtwitter.com/karpathy/status/1748043513156272416?s=20) 链接以及相关的 [AI Explained YouTube 视频](https://youtu.be/dOplrIJEYBo?si=fPomG0jy4BDBkbC_)。
- **来自代码库的工具**：`@lightningralf` 询问了 AlphaCodium IDE 插件的开源状态，并指出 PR-Agent 采用的是 Apache 2.0 许可。
- **Meta 的 GPU 军火库揭秘**：`@guardiang` 分享了 `@soumithchintala` 的一条推文，透露 Meta 的目标是在年底前部署相当于 60 万块 H100 GPU 的算力，引发了关于 GPU 可用性的讨论，`@swyxio` 提醒某人 (`@194927177265840128`) 加入对话。

**提到的链接**：

- [Code Generation with AlphaCodium: From Prompt Engineering to Flow Engineering](https://arxiv.org/abs/2401.08500)：代码生成问题不同于普通的自然语言问题——它们需要匹配目标语言的精确语法，识别正常路径（Happy Paths）和边缘情况（Edge Cases），关注数值...
- [Andrej Karpathy (@karpathy) 的推文](https://fxtwitter.com/karpathy/status/1748043513156272416?s=20)：代码生成的 Prompt Engineering（或者更确切地说是“Flow Engineering”）正在加强。非常值得一读，它提醒了我们从原始的 prompt:ans 转向流程设计中蕴含着多少 Alpha（pass@5 从 19% 提升到 44%）...
- [Soumith Chintala (@soumithchintala) 的推文](https://x.com/soumithchintala/status/1748074223187173724?s=46&t=6FDPaNxZcbSsELal6Sv7Ug)：终于可以公开谈论一些 GPU 数量了 🙃 到今年年底，Meta 将拥有 60 万块 H100 等效 GPU。尽管猜猜哪些已经部署并投入使用了 😉！
- [Alpha Everywhere: AlphaGeometry, AlphaCodium and the Future of LLMs](https://youtu.be/dOplrIJEYBo?si=fPomG0jy4BDBkbC_)：AlphaGeometry 是迈向 AGI 的关键一步吗？甚至 DeepMind 的领导层似乎也无法达成共识。在这段视频中，我将为你简要介绍 AlphaGeometry...
- [GitHub - Codium-ai/AlphaCodium](https://github.com/Codium-ai/AlphaCodium)：通过在 GitHub 上创建账号来为 Codium-ai/AlphaCodium 的开发做出贡献。
- [GitHub 的推文 - FixTweet/FxTwitter: 修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能](https://x.com/itamar_mar/s)：修复损坏的 Twitter/X 嵌入！在 Discord、Telegram 等平台上使用多图、视频、投票、翻译等功能 - GitHub - FixTweet/FxTwitter...
- [Itamar Friedman (@itamar_mar) 的推文](https://x.com/itamar_mar/status/1747957348293824676?s=20)：🚀 介绍 AlphaCodium - 首创的开源代码生成工具，在编程竞赛中超越了大多数人类选手 ⭐️ 灵感来自 DeepMind 的 AlphaCode❤️‍🔥，但超越了它...

### ▷ #[llm-paper-club](https://discord.com/channels/822583790773862470/1107320650961518663/) (1 messages):

- **为 Pythia 论文讨论做准备**：`@swyxio` 为即将进行的 Pythia 论文讨论推荐了一个来自 Gradient Dissent 的旧但信息量很大的播客片段，其中包含 2022 年左右对 EleutherAI 的 Stella Biderman 的采访。查看它可以获得关于 **LLM 训练**和部署的见解：[Gradient Dissent Podcast](https://overcast.fm/+Y_EFBYrkg)。

**提到的链接**：

[How EleutherAI Trains and Releases LLMs: Interview with Stella Biderman — Gradient Dissent: Exploring Machine Learning, AI, Deep Learning, Computer Vision — Overcast](https://overcast.fm/+Y_EFBYrkg)：未找到描述

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- **LangChain 更新受阻于过时的文档**：`@daslav` 指出 **LangChain 文档** 已过时，特别是涉及 `from langchain import hub` 代码的问题。此外，`@Sovok` 在其 **RAG 系统** 中遇到了未解决的错误，而 `@Behlal` 在使用 NVIDIA 4090 GPU 运行 **快速入门教程检索链 (quickstart tutorial retrieval chain)** 时也遇到了问题，这表明需要对文档进行审查并提供更好的错误诊断。
  
- **LangServe 的嵌套知识**：`@veryboldbagel` 讨论了 LangServe 中嵌套信息的高级用法，主张使用 `TypedDict` 和 `pydantic` 进行精确序列化，参考 [`server.py 示例`](https://github.com/langchain-ai/langserve/blob/main/examples/passthrough_dict/server.py#L57)。这一建议与其呼吁采用最近合并的 `astream_event` 以支持 UI 流式传输的观点一致，为增强交互式系统开启了可能性。
  
- **API 与前端同步成为焦点**：根据 `@veryboldbagel` 的见解，LangServe 用户应注意 `openai_assistant` API 对复杂输入（而非简单 prompt）的要求，并研究利用 **Server-sent Events (SSE)** Web 标准在前端进行数据流式传输，参考 [Mozilla 的 SSE 指南](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events)。
  
- **MapReduce 对 LCEL 的需求及 SQL 接口搜索**：讨论参与者强调了 **MapReduce** 缺乏 **LCEL** 支持（由 `@pramodhgopalan_80290` 提出），这预示着 Chain 语言灵活性即将升级。同时，有人向 `@meq__` 推荐了一个名为 *vanna* 的工具，用于开源的 **自然语言转 SQL 查询接口 (natural language to SQL query interface)**，为直观的数据查询提供了潜在解决方案。
  
- **AI 设计与生产力的创新与查询**：AI 正在重塑设计领域，**neThing.xyz** ([neThing.xyz](https://nething.xyz/)) 和 **Langsmith** 推动了 CAD 与生成式 AI (generative AI) 的融合，`@rawwerks` 正在寻求反馈。`@_anubix` 发起了关于提高生产力工具的对话，而 `@esxr_` 赞扬了 Olama 和 Langchain 对工作流的革命性改变，并邀请同行访问其以 AI 为中心的博客 ([esxr.io](https://esxr.io))。
  

**LangChain AI 频道总结**

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (14 条消息🔥):

- **MapReduce 尚不支持 LCEL**: `@pramodhgopalan_80290` 询问为何 **MapReduce 和 Stuff Summarization** 还没有 **LCEL** (LangChain Expression Language) 版本，指出文档中仅列出了旧版 (legacy) Chain。他们发现目前正在开发所有 Chain 的 LCEL 版本，以便于修改并提供原生的流式支持。
  
- **从 DuckDuckGo 检索图像**: `@solononforever3` 询问是否可以使用 **DuckDuckGo** 工具检索图像，但未收到直接回复。
  
- **为聊天机器人嵌入 Markdown 数据**: `@xery.` 计划为基于 YouTube 的维修指南聊天机器人嵌入 400 多个 Markdown 文件，但不确定单独嵌入每个 Markdown 文件的最佳块大小 (chunk size)。
  
- **寻找开源 SQL 查询语言接口**: `@meq__` 正在寻找开源的 **自然语言转 SQL 查询接口**，并记得之前在频道中见过。`@roi_fosca` 针对该查询建议了 *vanna* 这个名字。
  
- **LangChain 文档过时**: `@daslav` 报告 LangChain 文档似乎已过时，特别提到涉及 `from langchain import hub` 的代码已不存在。
  
- **重复回答之谜**: `@seththunder` 推测前一位用户查询中出现重复回答的原因可能是由于响应的流式传输，尽管这是在利用 **markdown text splitter** 嵌入数据的背景下提出的。
  
- **寻找 LangSmith 托管和企业计划**: `@muthu1823` 请求有关托管其自身 **LangSmith** 环境的联系方式或建议，并询问企业版或定价的可用性。
  
- **RAG 系统错误困扰**: `@Sovok` 在其 RAG (Retrieval-Augmented Generation) 系统中遇到了未指明的错误，并对无法理解原因表示沮丧，提到无法打开头文件。
  
- **快速入门教程检索链问题**: `@Behlal` 报告在配备 NVIDIA 4090 GPU 和 Ubuntu 系统的机器上，使用 Ollama 和 Llama2 运行 **快速入门教程 (quickstart tutorial)** 中的检索链时出现错误。
  

**提到的链接**:

[Chains | 🦜️🔗 Langchain](https://python.langchain.com/docs/modules/chains): Chains 指的是一系列调用——无论是对 LLM、工具还是...

### ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/) (7 messages):

- **使用 `TypedDict` 和 `pydantic` 进行嵌套**：`@veryboldbagel` 在 [`server.py`](https://github.com/langchain-ai/langserve/blob/main/examples/passthrough_dict/server.py#L57) 中提供了一个在 LangServe 中使用嵌套信息的示例。他们建议使用 `TypedDict` 以获得更高的精确度，并使用 `pydantic` 进行对象序列化，并建议参考 [Custom User Types](https://github.com/langchain-ai/langserve?tab=readme-ov-file#custom-user-types) 进行继承。
  
- **引用的详细 API 实现**：`@veryboldbagel` 强调了 `openai_assistant` API 除了简单的 prompt 之外还需要额外信息，并通过 [base.py 第 98 行](https://github.com/langchain-ai/langchain/blob/ca014d5b04b1d73fd8f0fe224def98a82600c991/libs/langchain/langchain/agents/openai_assistant/base.py#L98-L98) 和 [base.py 第 79 行](https://github.com/langchain-ai/langchain/blob/ca014d5b04b1d73fd8f0fe224def98a82600c991/libs/langchain/langchain/agents/openai_assistant/base.py#L79-L79) 分享了具体的实现示例链接。
  
- **用于 Svelte 自定义 UI 的 RemoteRunnable 客户端**：`@veryboldbagel` 讨论了 Langchain-js 的 remote runnable 客户端的使用，并提供了 [API 链接](https://api.js.langchain.com/classes/langchain_runnables_remote.RemoteRunnable.html)，这有助于使用 Svelte 创建自定义 UI。
  
- **可配置的 Runnables 和模型**：在 `@veryboldbagel` 的一条消息中，解释了作为 LangChain Expression Language 一部分的可配置 runnables 的用法，并提议在 langserve 中进一步讨论，以造福社区并提高解决方案的可发现性。
  
- **在前端处理流式数据**：`@veryboldbagel` 回复了 `@hiranga.g` 关于向前端流式传输数据的问题，建议先从作为 Web 标准的 server-sent events (SSE) 开始，并在深入研究 RemoteRunnable 之前先查看使用 SSE 的示例应用程序。他们分享了一个 [Mozilla 资源](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) 供参考。
  
- **Langchain-Core 上的流式支持**：`@veryboldbagel` 指出了 Langchain-core 上最近合并的一个 RFC，该 RFC 引入了 `astream_event` 以在 UI 中提供更好的流式支持，并承诺在一周内尝试将其添加到 langserve 中。他们提供了一个 [讨论链接](https://github.com/langchain-ai/langchain/discussions/16175) 以获取更多细节。
  

**提到的链接**：

- [🛸 Streaming: RFC Adding astream_event to all Runnable objects to help with streaming use cases · langchain-ai/langchain · Discussion #16175](https://github.com/langchain-ai/langchain/discussions/16175)：大家好！我们想改进 LangChain 中的流式体验。我们正在考虑为 Runnable 接口添加一个 astream_event 方法。下面的代码来自以下 PR，且没有...
- [langchain/libs/langchain/langchain/agents/openai_assistant/base.py at ca014d5b04b1d73fd8f0fe224def98a82600c991 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/ca014d5b04b1d73fd8f0fe224def98a82600c991/libs/langchain/langchain/agents/openai_assistant/base.py#L98-L98)：⚡ 通过可组合性使用 LLM 构建应用程序 ⚡ - langchain-ai/langchain
- [langchain/libs/langchain/langchain/agents/openai_assistant/base.py at ca014d5b04b1d73fd8f0fe224def98a82600c991 · langchain-ai/langchain](https://github.com/langchain-ai/langchain/blob/ca014d5b04b1d73fd8f0fe224def98a82600c991/libs/langchain/langchain/agents/openai_assistant/base.py#L79-L79.)：⚡ 通过可组合性使用 LLM 构建应用程序 ⚡ - langchain-ai/langchain
- [langserve/examples/passthrough_dict/server.py at main · langchain-ai/langserve](https://github.com/langchain-ai/langserve/blob/main/examples/passthrough_dict/server.py#L57)：LangServe 🦜️🏓。通过在 GitHub 上创建一个帐户来为 langchain-ai/langserve 的开发做出贡献。
- [GitHub - langchain-ai/langserve: LangServe 🦜️🏓](https://github.com/langchain-ai/langserve?tab=readme-ov-file#custom-user-types,)：LangServe 🦜️🏓。通过在 GitHub 上创建一个帐户来为 langchain-ai/langserve 的开发做出贡献。

### ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/) (4 条消息):

- **neThing.xyz 借助 Langsmith 成型**: 用户 `@rawwerks` 正在利用 [Langsmith](https://langsmith.ai/) 来辅助 [neThing.xyz](https://nething.xyz/) 的追踪和评估。这是一个针对 CAD 和工程应用的文本转 3D 生成式 AI。他们欢迎对该项目的任何反馈，该项目承诺提供一种在设计领域与 AI 交互的新方式。
  
- **提升生产力的工具**: 用户 `@_anubix` 向社区询问了能显著提高日常生产力的工具。
  
- **Ollama 和 Langchain 彻底改变日常工作流**: `@esxr_` 分享了 Ollama 和 Langchain 如何极大地改变了他们的工作方式，使他们能够构建自定义解决方案。他们还为自己的用途定制了 Ollama WebUI，这显著提高了他们的生产力。
  
- **AI 爱好者记录 AI 探索博文**: `@esxr_` 提到了他们的博客 [esxr.io](https://esxr.io)，他们在那里记录自己的 AI 发现和经验，表明对 AI 及其应用的更广泛领域有着浓厚的兴趣。
  

**提到的链接**:

- [neThing.xyz - AI Text to 3D Model](https://nething.xyz/): AI 驱动的文本转 3D 模型
- [Pranav Dhoolia](https://esxr.io): 我是一名 AI 爱好者，热衷于探索广阔而有趣的人工智能领域。我将这个博客作为记录我发现的协作笔记本。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 总结

- **新的德语模型登场**: **[DiscoLM German 7b](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1)**，在 650 亿个 tokens 上进行训练，具备英语、德语和翻译任务的能力，支持 RAG 和 function calling。有人询问了 DiscoLM German 7b 的性能指标，但尚未提供具体的基准测试数据。
  
- **基准测试：情感 vs 指令**: 在 **benchmark_dev** 的讨论中，考虑增加一个**复杂推理部分**，以衡量情商和复杂的指令遵循能力。7b 模型的排名之高令人惊讶，随后展开了关于严格关注情商的基准测试标准的讨论。
  
- **需要更长的代码片段**: 在 **embedding_dev** 中观察到，代码文档检索在超过 512 个 token 限制后性能会下降，建议尝试使用 **jina encodings** 和扩展的 chunk 大小。
  
- **Axolotl 准备打磨**: 在 **discolm_german** 中讨论了即将分享的 **Axolotl** 模型的训练代码和配置，并提到了可能的训练数据/代码共享以及以 RAG 为重点的协作。用户报告的演示页面故障得到了及时关注，强调了积极的支持和运营意图。
  

**DiscoResearch 频道总结**

### ▷ #[general](https://discord.com/channels/1178995845727785010/1182877486854451271/) (4 条消息):

- **推出 DiscoLM German 7b**: `@_jp1_` 宣布发布 **[DiscoLM German 7b](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1)**，该模型在 65b tokens 上训练，专为英语、德语和翻译用途设计。该模型独特地支持 RAG 应用和实验性的 function calling 能力。
- **查看在线演示**: `_jp1_` 分享了 DiscoLM German 7b 的在线演示，可在 **[demo.discoresearch.org](https://demo.discoresearch.org/)** 进行实际体验。
- **模型变得调皮了！**: `@devnull0` 幽默地评论说，问模型 "Was geht?"（怎么了？）可能会让它崩溃，暗示在测试期间给模型输入了一些俏皮或复杂的指令。
- **性能基准测试咨询**: `@cryptossssun` 询问了 DiscoLM German 7b 的基准测试数据，寻求对其性能指标的深入了解。

**提到的链接**:

- [DiscoResearch/DiscoLM_German_7b_v1 · Hugging Face](https://huggingface.co/DiscoResearch/DiscoLM_German_7b_v1): 未找到描述
- [DiscoLM German 7b Demo](https://demo.discoresearch.org/): 未找到描述

### ▷ #[benchmark_dev](https://discord.com/channels/1178995845727785010/1183158791605330051/) (4 messages):

- **Mixtral 在新版本中性能提升**：`@.calytrix` 回应 `@_jp1_`，强调 **Mixtral** 在最新版本中比第一版本表现更出色。
  
- **对 7b 模型排名靠前感到惊讶**：`@_jp1_` 对 **Beagle** 等 7b 模型相比 **Mixtral instruct** 排名更高表示惊讶，并请求提供 Beagle 表现更优的示例。
  
- **关于基准测试标准的澄清**：`@.calytrix` 向 `@_jp1_` 澄清，虽然单个问题的分析可能无法完全代表整体性能，但评论部分（critique section）很有启发性。这些基准测试是专门为严格评估情商（emotional intelligence）而设计的，而非复杂的指令遵循（instruction following）。
  
- **基准测试方法的潜在增强**：`@.calytrix` 向 `@_jp1_` 提到，可能会在测试中加入**复杂推理部分**（complex reasoning section），以创建一个同时衡量情商和复杂指令遵循的综合评分。
  

### ▷ #[embedding_dev](https://discord.com/channels/1178995845727785010/1192471915504341062/) (1 messages):

- **代码文档检索性能下降**：`@sebastian.bodza` 观察到，在与 512 token 限制相关的**激进截断**（aggressive truncation）后，代码文档检索的**性能**有所下降。下一步将尝试使用 **jina encodings** 和更长分块大小（chunk sizes）的实验。

### ▷ #[discolm_german](https://discord.com/channels/1178995845727785010/1197630242815213618/) (13 messages🔥):

- **对开源倡议感兴趣**：`@philipmay` 对分享开源工作表示感谢，并对项目的多个方面表现出兴趣，提出了多个问题。
- **Axolotl 的训练和代码即将公开**：`@_jp1_` 确认计划分享 **Axolotl** 的训练代码/配置以及包含高级用法示例的仓库；不过，他们指出需要更多时间来整洁地呈现这些内容。
- **训练数据分享的可能性**：在回应 `@philipmay` 时，`@_jp1_` 透露分享训练数据和代码（特别是关于 **RAG** 的部分）是可能的，并强调了 `<@1048301853806448680>` 的参与，同时提到了正在进行的改进和潜在的合作。
- **处理 AI 的拒绝回答**：针对 `@maxidl` 遇到的 **Axolotl** 给出拒绝回答的情况，`@_jp1_` 承认正在努力过滤这些内容，并鼓励用户反馈以改进未来的迭代。
- **演示页面故障与恢复**：`@devnull0` 称赞了演示页面，随后报告了一个 **Cloudflare Origin DNS 错误**，但 `@_jp1_` 迅速表示问题已解决，标志着页面已恢复运行。

---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

- **对功能性数据集的需求**：用户 **@interstellarninja** 和 **@yikesawjeez** 讨论了对精细化 **function calling 数据集**的需求，以对齐 OpenAI 的规范，并强调了开源 function caller 与 OpenAI API 的兼容性要求。
  
- **探讨 LLM 推理成本动态**：虽然 **@helium0120** 寻求有关 **LLM 推理成本**趋势的数据，但 **@nisten** 对成本计算的复杂性提出了警告，指出 API 服务可能的补贴是干扰因素。
  
- **对 Lookahead Decoding 方法的审查**：**@nisten** 对 **lookahead decoding 方法**进行了批判性评估，承认其局限性，但也注意到其在代码编辑等特定场景下的有效性。贡献内容包括一个探讨该方法用于 LLM 推理加速的[详细博客文章](https://lmsys.org/blog/2023-11-21-lookahead-decoding/)链接。
  
- **无关话题交流**：用户 **pradeep1148** 分享了一个非技术性的 YouTube 视频链接，这与公会的技术和工程讨论无关。
  

**Skunkworks AI 频道摘要**

### ▷ #[general](https://discord.com/channels/1131084849432768614/1131084849906716735/) (9 messages🔥):

- **Function Calling 数据集的挑战**: `@interstellarninja` 承认了现有数据集的局限性，并表示需要一个**多样化的 Function Calling 数据集**，该数据集应与 OpenAI 的函数签名和调用模式 (schema) 保持一致。这将有助于与 OpenAI API 的兼容性，使开源 Function Caller 易于替换。
  
- **继续寻找 Function Caller**: `@yikesawjeez` 意识到现有数据集的局限性，并表示打算寻找一个更适合 OpenAI 需求的数据集。
  
- **寻求 LLM 推理成本趋势**: 用户 `@helium0120` 询问了有关 **LLM 推理成本** 随时间下降的趋势或预测的任何可用数据。
  
- **对 LLM 推理成本降低的怀疑**: `@nisten` 评论说，由于 API 服务可能在补贴这些成本，推理成本的计算具有挑战性，这使得直接的成本降低趋势存疑。
  
- **Lookahead Decoding 方法评估**: `@nisten` 批判性地评估了 **Lookahead Decoding 方法**，发现它并不像声称的那样有效，除非在某些特定场景下（例如需要对代码进行少量修改并重新输出整个代码的代码编辑）。讨论中提供了一个博客文章链接 ([Lookahead Decoding: Accelerating LLM Inference](https://lmsys.org/blog/2023-11-21-lookahead-decoding/))，该文章深入探讨了该方法加速 LLM 推理的方式。
  

**提到的链接**:

[使用 Lookahead Decoding 打破 LLM 推理的顺序依赖 | LMSYS Org](https://lmsys.org/blog/2023-11-21-lookahead-decoding/): <p><strong>摘要:</strong> 我们介绍了 <strong>lookahead decoding</strong>，这是一种新的、精确且并行的解码算法，用于加速 LLM 推理。Look...

### ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/) (1 messages):

pradeep1148: [https://www.youtube.com/watch?v=POgLwYxDGYk](https://www.youtube.com/watch?v=POgLwYxDGYk)

---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord 总结

- **切换模型以确保顺利运行**: `@jeffreyw128` 提到他们在**不同的 GPT** 之间切换以避免问题，而 `@thebaghdaddy` 正在考虑**不使用高级分析**来运行流程作为变通方法。
- **用户更倾向于使用 Instruct 模型**: `thisisnotawill` 表示正在使用来自 Anyscale 的 **Instruct 模型**，未提供额外上下文。
- **征求数据合成见解**: `@ayenem` 正在寻找有关**将数据合成模型投入生产**的资源，但社区尚未回应。
- **考虑开设 MLOps 频道**: `@ayenem` 建议创建一个 #mlops 频道，`@pantsforbirds` 认为这可能很有用，尽管他将 MLOps 称为“我痛苦的根源”。`@jeffreyw128` 质疑是否有必要设立独立的 MLOps 频道。
- **Azure 过滤器切换困扰**: `@thisisnotawill` 寻求有关如何**在 Azure 中禁用内容过滤器**的帮助，并指出该功能似乎仅限于内部使用；随后的讨论或解决方案尚未明确。

**LLM Perf Enthusiasts AI 频道总结**

### ▷ #[gpt4](https://discord.com/channels/1168579740391710851/1168582188950896641/) (3 messages):

- **切换 GPT 以避免烦恼**: `@jeffreyw128` 提到为了规避某些问题，他们选择使用**不同的 GPT**。
- **分析绕过策略**: 针对 `@jeffreyw128`，`@thebaghdaddy` 考虑了该建议，并决定**不使用高级分析**来运行他们的流程，作为一种潜在的解决方案。

### ▷ #[opensource](https://discord.com/channels/1168579740391710851/1168606773595349082/) (1 messages):

thisisnotawill: 是的，我正在使用来自 Anyscale 的 Instruct 模型。

### ▷ #[resources](https://discord.com/channels/1168579740391710851/1168760058822283276/) (1 messages):

- **寻找数据合成的智慧**: `@ayenem` 正在寻求有关**将数据合成模型投入生产**的经验或资源，如**博客、书籍或工具**。消息历史中没有提供回复。

### ▷ #[feedback-meta](https://discord.com/channels/1168579740391710851/1169009508203368549/) (3 messages):

- **MLOps 频道提案**: 用户 `@ayenem` 询问其他人是否对 #mlops 频道感兴趣，暗示社区可能对这样的空间有需求。
- **MLOps：值得一读**: `@pantsforbirds` 幽默地将 MLOps 称为“我痛苦的根源”，但表示如果创建了 #mlops 频道，他有兴趣阅读有用的帖子。
- **讨论 MLOps 频道的必要性**: 作为回应，`@jeffreyw128` 询问在 #mlops 频道中会进行哪些在当前频道中无法进行的讨论。

### ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/) (1 条消息):

- **Azure 内容过滤器困惑**：用户 `@thisisnotawill` 询问关于在 Azure 中**禁用内容过滤器**的问题，提到该选项似乎仅限内部使用。在提供的历史记录中没有提供解决方案或后续讨论。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 总结

只有一个频道有活动，因此无需总结...

imonenext: 有人有 Gemini Pro 的密钥吗？

---

## [Datasette - LLM (@SimonW)](https://discord.com/channels/823971286308356157) Discord 总结

只有一个频道有活动，因此无需总结...

- **离线 GPT-4All 模型使用的 PR**：`@cameron_y` 提交了一个 Pull Request 以支持 gpt4all 模型的离线使用，解决了即使模型已存在于本地，库仍会尝试下载模型的问题。该修复详情见 [GitHub 上的 PR #18](https://github.com/simonw/llm-gpt4all/pull/18)。

**提到的链接**：

[fix: allow local models to work without internet connection by hydrosquall · Pull Request #18 · simonw/llm-gpt4all](https://github.com/simonw/llm-gpt4all/pull/18)：动机：目前，即使模型已存在于本地，库仍会尝试下载模型，这阻碍了离线使用。修复了 #10，应用了来自 @rotterb 的代码提示和调查。变更内容...

---

YAIG (a16z Infra) Discord 没有新消息。如果该服务器沉寂时间过长，请告知我们，我们将将其移除。