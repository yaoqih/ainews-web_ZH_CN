---
companies:
- microsoft
- mistral-ai
- ollama
date: '2024-03-22T23:55:31.644920Z'
description: 'Reddit 社区 /r/LocalLlama 经常讨论**大语言模型（LLM）的微调与训练**，包括使用词典等特定数据以及规模超过 **250
  亿 token** 的合成数据集进行模型训练的教程和疑问。


  用户们还在探索 **mistral-7b** 等模型在**检索增强生成（RAG）**方面面临的挑战，以及针对 **EEG 脑电活动**的嵌入生成（embedding
  generation）。讨论内容还涉及在预算有限的情况下，如何在本地运行 **llama-2-70b** 的**硬件优化**方案，以及 **qwen-1.5**
  模型的性能基准测试。此外，社区对扩展 LLM 的功能表现出浓厚兴趣，例如将 **llama-2-7b** 转化为像 **llava** 这样的视觉模型，以及通过改进模型记忆来提升长上下文的保留能力。'
id: e716fd7e-b73f-42dd-a92e-459827abadab
models:
- llama-2-70b
- llama-2-7b
- mistral-7b
- qwen-1.5
- llava
original_slug: ainews-not-much-happened-today-2070
people: []
title: 今天没什么事。
topics:
- fine-tuning
- synthetic-data
- retrieval-augmented-generation
- embeddings
- hardware-optimization
- performance-benchmarks
- model-memory
- multimodality
---

<!-- buttondown-editor-mode: plaintext -->> 2024年3月21日至3月22日的 AI 新闻。我们为您查看了 [**364** 个 Twitter 账号](https://twitter.com/i/lists/1585430245762441216) 和 **22** 个 Discord 社区（**341** 个频道，共 **5210** 条消息）。预计为您节省阅读时间（以 200wpm 计算）：**526 分钟**。

当我们可以说一整天的新闻都可以跳过时，我们为您节省了最多的时间……如果我们错了，我们也很喜欢这种（[伪托的](https://en.wikipedia.org/wiki/Nothing_Important_Happened_Today#Production)）讽刺！

祝阅读愉快，或者查看 [Latent Space 关于 Adept 的新剧集](https://www.latent.space/p/adept)。下周我们将扩大对 Reddit 的覆盖。

---

**目录**

[TOC] 

---

# REDDIT

> 目前仅从 /r/LocalLlama 开始，我们很快会总结评论，但接下来我们已经规划好了 r/machinelearning, r/openai, r/stablediffusion, r/ArtificialInteligence。如果您发现我们遗漏了任何主要的 alpha drop 子版块，请告知我们。

## /r/LocalLlama

**Fine-Tuning 和训练 LLM：**

- **学习如何进行 Fine-Tuning（第一次），我提供了找到的教程链接，还有人推荐其他资料吗？** 一位用户正尝试学习如何对模型进行 Fine-Tuning，并整理了来自 Reddit 和 DuckDuckGo 的阅读材料。他们提出了关于在特定主题（如《赛博朋克 2077》和业务数据）上训练模型的问题，并正在寻找使用 llama.cpp 进行 Fine-Tuning 的技巧。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkqxui/learning_how_to_finetune_first_time_ive_provided/)
- **LLM 可以在词典上训练吗？如果可以，该如何操作？** 一位用户希望在本地语言词典上训练像 Gemma 这样的多语言模型，并正在为非技术外行寻找操作步骤。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkqtvs/can_llm_trained_on_a_dictionary_if_yes_how_to_do/)
- **如何生成大规模合成数据。** 一篇关于如何使用 Mixtral 模型构建超过 25B+ tokens 的大规模合成数据集的博客文章，类似于微软用于训练 Phi 模型的数据集。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk3lqc/how_to_generate_largescale_synthetic_data/)

**检索增强生成 (RAG) 和 Embeddings：**

- **[问题] RAG 中的查询没有返回 chunks，也没有结果？** 一位用户正尝试基于 mistral 7b 模型、chroma DB 和 markdown 文本作为输入数据源开发 RAG。他们正在进行自定义的 chunking 和 embedding，但在进行常规查询时，没有返回任何 chunks 或响应。他们提供了示例代码和 markdown 文件。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk9tky/question_query_in_rag_returning_no_chunks_and_no/)
- **有人研究过生成脑活动的 Embeddings 吗？** 一位用户正在处理 EEG 数据，并希望匹配类似的 EEG 信号模式。他们引用了一篇论文，并想知道是否有人在这个领域取得了成功。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk63pq/has_anyone_worked_on_generating_embeddings_on/)
- **关于理解为什么你的 RAG/LLM 组合不起作用的精彩视频。** 一位用户推荐了一个经过深入研究的视频，讨论了为什么 Fine-Tuning 加 RAG 优于单纯的 RAG、大参数模型与小参数模型的区别，以及如何将 RAG 查询中的偏差情境化。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk75fk/great_video_on_understanding_why_your_ragllm/)

**部署和优化 LLM：**

- **llama 2 70b 的硬件建议。** 一位用户的上司要求他们构建一个合适的工作站机架来本地运行 llama 模型，目标是将查询时间从目前 7b 模型的 3 分钟缩短到 10 秒以内。他们的预算在 1.5 万欧元以下，正在寻求建议。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk4j9t/hardware_suggestion_for_llama_2_70b/) 
- **一个用于测量 ollama 模型每秒 token 数 (tokens per second) 的脚本（在 Nvidia 4090 上测得 llama2:13b 为 80t/s）。** 一位用户分享了他们制作的用于测量 ollama 模型 tokens per second 的脚本。在 Nvidia 4090 上，他们在 llama2:13b 上获得了 80t/s，在 llama2:7b 上获得了 127t/s。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkl5s2/a_script_to_measure_tokens_per_second_of_your/)
- **Qwen1.5 模型的速度和内存基准测试。** 指向 Qwen1.5 模型在速度和内存占用方面的基准测试链接。[链接](https://qwen.readthedocs.io/en/latest/benchmark/hf_infer.html)

**扩展 LLM：**

- **是否可以将 LLaMA 转换为 LLaVA。** 一位用户微调了 LLaMA 2 7B 模型，并想知道是否可以在不需要单独微调 LLaVA 的情况下为其添加视觉能力。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bksyq3/is_it_possible_to_turn_llama_into_llava/)
- **模型“记忆”。** 一位用户询问是否可以改进模型的“记忆”，使其能够记住至少 5 条消息之前的内容。他们知道 Context Size（上下文窗口大小）很重要，但想知道是否还有其他方法。他们还询问是否有支持 CS 8K 的 13B 模型。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkbrp6/model_memory/)
- **推理时的深度上采样（Depth upscaling）。** 一位用户分享了一个在推理时实现深度上采样的实验，而无需实际增大模型，因此对 GPU 资源匮乏的用户非常友好。由于模型目前存在一定的重复性，因此仍需要进行微调。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkjlvu/depth_upscaling_at_inference_time/)

**应用与使用案例：**

- **说实话：真的有人在运行能赚钱的 Agent 吗？** 一位用户询问是否有人在运行能够自主赚钱的 LLM Agent，哪怕每天只有几美元。如果有人愿意分享，他们希望了解所使用的架构和模型的模糊信息。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk3fd5/lets_get_real_is_there_anybody_whos_running/)
- **如何高效地利用 LLM 并根据自己的写作风格训练出专属写作助手？** 一位用户正在寻求一种快速且高效的方法，来训练已安装的 LLM 或 chat.ml，使其写作风格与用户一致，因为仅靠 Prompting 仍然会导致输出风格像 ChatGPT。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bkg8tn/what_is_an_efficient_way_to_create_your_own/)
- **与大型 PDF 库交互。** 一位用户拥有数千篇以 PDF 格式存储的科学论文，并希望有一个聊天机器人能够回答关于整个库内容的问题，从多个 PDF 中检索信息，而无需用户指定是哪些文件。他们询问是否存在这样的工具。[链接](https://www.reddit.com/r/LocalLLaMA/comments/1bk1tte/interacting_with_a_large_pdf_library/)

---

# 第 X 部分：AI Twitter 综述

> 所有综述由 Claude 3 Opus 完成，从 4 次运行中选取最佳结果

**开源模型与框架**

- [Open-Sora 1.0](https://twitter.com/svpino/status/1769467954477859047)：开源文本转视频模型，提供完整的训练过程、数据和 Checkpoints（10 万次观看）
- [Thunder](https://twitter.com/rasbt/status/1770805633698181383)：PyTorch 的新型开源编译器，在 Llama 2 7B 等 LLM 训练任务中比常规 PyTorch 提速 40%（8.7 万次观看）
- [Jan](https://twitter.com/omarsar0/status/1770927000326201685)：开源 ChatGPT 替代方案，可在电脑本地运行，支持多种架构（5.1 万次观看）
- [LLaVA-NeXT (LLaVA-1.6)](https://twitter.com/ClementDelangue/status/1771047389983367419)：强大的开源 Vision-Language 模型，现已加入 Hugging Face Transformers 库（1 次转发）
- [Transformers 4.39](https://twitter.com/osanseviero/status/1770931570272030760)：新版本包含大量模型更新，如 Mamba, Command-R, LLaVA-NeXT, MusicGen Melody, StarCoder2, SegGPT 等（1.1 万次观看）

**计算趋势与硬件**

- [Sam Altman 认为计算力将成为未来最重要的货币](https://twitter.com/AISafetyMemes/status/1769600345171481073)，世界对日益增长的计算需求准备不足（18.1 万次观看） 
- [在 Groq 硬件上运行 Grok 可能会改变游戏规则](https://twitter.com/deliprao/status/1769492688770908207)（3.8 千次观看）
- [Nvidia 是 AGI 公司的最佳典范](https://twitter.com/far__el/status/1770958097734877352)，完全掌控了整个硬件/软件栈（6 千次观看）

**进化模型合并**

- [Sakana AI Labs 发布了模型合并的进化方法](https://twitter.com/maximelabonne/status/1770768615576408434)，通过优化参数和层排列，实现专用模型的创建（1.9 万次观看）
- [Sakana AI 的进化模型合并技术被用于创建具备数学推理能力的日语 LLM、Vision LLM 以及图像生成模型](https://twitter.com/hardmaru/status/1770789055090786354)（2 次转发）

**检索增强生成 (RAG)**

- [RAFT (Retrieval Augmented Fine-Tuning)](https://twitter.com/cwolferesearch/status/1770912695765660139)：通过在特定领域文档上进行微调来提升 LLM 在 RAG 中表现的方法，性能优于标准 RAG（2.7 万次观看）
- [结合合成数据生成的 RAG 差分隐私技术](https://twitter.com/llama_index/status/1770837291855991085)，实现了敏感数据集的知识共享（3.6 万次观看）

**新兴趋势与应用**

- [来自 Meta AI 的 SceneScript](https://twitter.com/AIatMeta/status/1770844932346920976)：一种使用端到端 Machine Learning 重建环境并表示物理空间布局的新方法（23 万次观看）
- [Suno AI 发布 v3 模型](https://twitter.com/suno_ai_/status/1770857426507399285)，可在数秒内生成广播级音质的音乐（15.2 万次观看）
- [Cohere 正在通过长文本摘要和知识助手改变保险业](https://twitter.com/cohere/status/1770817028183486824)（4 千次观看）
- [Runway 与 Musixmatch 合作](https://twitter.com/c_valenzuelab/status/1770801245445407001)，简化歌词视频的创建与定制（8 千次观看）

**提示工程作为一种职业**

- [“我还记得当时人们认为 ‘Prompt Engineering’ 会成为一种真正的职业。”](https://twitter.com/svpino/status/1770873052810883156)（100 万次观看）


---

# 第 0 部分：摘要之摘要之摘要


> 我们得出结论，Claude Opus 是顶级摘要的最佳模型，因此我们将停止 A/B/C 测试（见存档了解我们的尝试/记录）。我们将为所有 3 个及更多模型（包括 Gemini 1.5!!）提供并行运行，因为这个问题在拓扑上与我们即将推出的个性化应用相似。

- **Stable Diffusion 3 期待值升温**：Stability.ai 社区热切期待 **Stable Diffusion 3 (SD3)** 的发布，讨论了用于艺术生成的最佳 **control nets**、**AMD GPU 兼容性**，以及为硬件受限用户提供的 **cloud GPU services**。分享了故障排除技巧，例如使用 [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml) 以支持 AMD。

- **Unsloth AI 的即将推出的功能**：**Unsloth AI** 正在致力于集成 **multi-GPU support** 和用于自动数据策划的平台 UI。社区还辩论了评估框架、数据质量以及 Benchmark 透明度的重要性，例如纠正 **MMLU dataset** 中 25% 样本参考答案错误的问题。

- **OpenInterpreter 的 01 Light 发布**：**[01 Developer Preview](http://openinterpreter.com/01)** 发布，这是一款通过语音控制计算机的便携式 AI 设备，引发了热议。社区分享了组装说明、[Bill of Materials](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md) 和 [3D print designs](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model)，同时还讨论了发货和软件功能。

- **LM Studio 更新引发讨论**：**LM Studio** 的新功能（如多模型支持和 ROCm 0.2.17 Beta v3 发布）引发了围绕 **ejecting models**、**GPU offloading**、**ZLUDA interference** 和 **high CPU usage** 的故障排除讨论。社区还推荐了用于结构化 LLM 输出的 [Instructor library](https://github.com/jxnl/instructor)。

- **AI 伦理与安全担忧**：Perplexity AI 和 HuggingFace 的对话涉及了 AI 访问敏感信息的伦理问题，例如在 ['Guardrails Arena'](https://huggingface.co/spaces/lighthouzai/guardrails-arena) 实验中，以及允许拦截加密 AI 聊天机器人 Token 的安全漏洞（[详细解释](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)）。

- **新兴技术与数据集**：多个频道讨论了新的 AI 技术和数据集，例如： 
  - [DenseFormer](https://arxiv.org/abs/2402.02622) 提出 Depth-Weighted-Average 以改进 Transformer 模型
  - [Quiet-STaR](https://arxiv.org/abs/2403.09629) 用于为每个 Token 生成理由，以增强 LM 文本解释
  - [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)，一个用于 LLM 预训练的大型合成数据集
  - [ASCII Art Dataset](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM) 引发了对 ASCII 艺术扩散模型的兴趣

- **优化 AI 性能**：讨论涵盖了各种优化技术，包括：
  - [1-bit LLMs like BitNet b1.58](https://arxiv.org/abs/2402.17764) 以更高的效率匹配全精度模型
  - [Galore optimizer](https://github.com/huggingface/transformers/pull/29588) 用于大模型的内存高效微调
  - 将 **GaLore's Adam optimizer** 与 Triton 融合，以实现更快的预训练和微调
  - 最大化 Transformer 模型 GPU 性能的指南（[论文](https://arxiv.org/abs/2401.14489)）

---



# 第 1 部分：高层级 Discord 摘要

## [Stability.ai (Stable Diffusion)](https://discord.com/channels/1002292111942635562) Discord

- **Stable Diffusion 3 热潮兴起**：社区正热切期待 **Stable Diffusion 3 (SD3)** 的发布，讨论重点集中在为艺术生成选择最佳的 **ControlNets**。此外，针对 **AMD GPU 兼容性** 的见解交流也非常活跃，并为硬件受限的用户推荐了 **云端 GPU 服务**。
  
- **深入 AMD 领域**：一位在 AMD 系统上遇到 **NVIDIA 驱动 RuntimeError** 的用户得到了帮助，被引导至支持 AMD GPU 的 [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml)，并附带了详尽的安装指南。

- **显存门 (VRAM-Gate)**：技术讨论正围绕即将发布的 SD3 的 **显存 (VRAM) 需求** 展开，引发了关于在本地机器上运行资源密集型模型可行性的推测。

- **提示词工程即服务 (Prompt Engineering-as-a-Service)**：社区成员正在分享磨炼 **提示词 (Prompting)** 技巧的方法，涵盖从“部落视频”到 **D&D 战役** 视觉效果的创作，并寻找经过微调、能理解复杂角色和场景艺术详细提示词的特定模型。

- **AI 工具：是祸还是福？**：关于 AI 对就业和创造力影响的辩论正在升温，观点从谨慎到乐观不等，认为 AI 是增强人类能力的 **进化工具**。



---



## [Unsloth AI (Daniel Han)](https://discord.com/channels/1179035537009545276) Discord

- **多 GPU 支持和数据策展即将登陆 Unsloth AI**：Unsloth AI 正在积极将 **多 GPU 支持** 作为开源功能进行集成，旨在兼容 Kaggle 等平台。此外，他们正在开发一个用于自动数据策展 (Data Curation) 的平台 UI，以简化模型微调中的数据准备步骤。

- **探索 Unsloth AI 安装问题的解决方案**：用户报告了安装 Unsloth AI 时的问题，包括 'no matching distribution' 错误以及在单 GPU 限制设置下的 `RuntimeError`。还有关于 **4-bit 量化模型** 可能的 **CUDA 级别更改** 以及量化模型显存限制超过 15GB 可能导致显存溢出 (Out-of-memory) 错误的讨论。

- **Unsloth AI 社区解决多样化问题**：讨论围绕配置 **LoRA 设置**、通过调整训练参数处理显存溢出错误，以及本地保存/加载模型和 Tokenizer 的技巧展开。对缺少 `protobuf` 等依赖项的担忧以及对某些技术领域最佳模型的困惑也值得关注。

- **社区焦点：Samantha Mistral Instruct 7B**：社区成员 cognitivetech 展示了他们使用 [Samantha Mistral Instruct 7B](https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes) 的成果，专门用于总结心理学书籍。分享了模型量化方面的困难以及向 Hugging Face 上传可用版本的承诺。

- **Lightning Thunder 在 Unsloth AI 社区引起轰动**：社区成员指出了 Unsloth AI 与 Lightning Thunder 集成中可能存在的失误，指出了性能问题和实现错误的内核 (kernels)。有人呼吁在基准测试 (benchmarks) 中进行协作并准确展示 Unsloth 的能力，一些人对 [Twitter](https://twitter.com/danielhanchen/status/1770863091296632921) 上误导性的性能对比表示沮丧。



---

## [OpenInterpreter](https://discord.com/channels/1146610656779440188) Discord

- **01 Light 全球发布会**：工程师们对 [01 Developer Preview](http://openinterpreter.com/01) 的发布感到兴奋，这是一款便携式计算机语音界面设备，具备识别屏幕和使用 App 的能力。社区正在分享组装说明和 [Bill of Materials](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md)，并对发货至印度和欧盟等地区表示关注。

- **硬件爱好者大显身手**：DIY 社区成员正在讨论 3D 打印他们自己的 01 Light 版本，**设计文件[可在 Printables 获取](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model)**，源代码可在 **[OpenInterpreter GitHub](https://github.com/OpenInterpreter/01)** 找到。

- **跨时区故障排除**：广泛的故障排除主题包括在各种操作系统上设置 01，以及解决国际物流问题。有人提出了一个 Windows 兼容性的变通方案——`poetry run 01 --client --client-type linux --server-host localhost`。

- **对软件功能的关注与疑问**：成员们深入探讨了 OpenInterpreter 的软件层面，讨论了本地运行与云端运行、API keys、语言兼容性和电池寿命，突出了 AI 工程师理解产品易用性和技术规格的关键方面。

- **预告片发布**：在 #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/) 频道中只有一条简洁的消息，包含一个与 OpenInterpreter 相关的 YouTube 链接，未提供背景或内容详情。



---



## [LM Studio](https://discord.com/channels/1110598183144399058) Discord

**Hermes 2.5 稳坐头把交椅**：在添加了 [代码指令示例](https://huggingface.co/roborovski/superprompt-v1) 后，**Hermes 2.5** 在各项基准测试中表现优于 **Hermes 2**，用户正在讨论不同模型和配置对 LM Studio 性能的影响。

**解决 LM Studio 的小毛病**：成员们报告了 **LM Studio 版本 0.2.17** 的问题，包括符号链接（symlinks）无法识别，以及提示 "Model with key Mistral/Herm... not found" 的错误。此外，性能讨论还涉及异常的 **CPU usage** 以及与 AMD ROCm 和 RX 570 显卡的 **compatibility**。

**AI 伦理与安全：热烈讨论**：社区通过讨论在 [Hugging Face 的 'Guardrails Arena'](https://huggingface.co/spaces/lighthouzai/guardrails-arena) 中与模型交互，深入探讨了 AI 的伦理和安全问题，以及允许拦截加密 AI 聊天机器人 Token 的安全漏洞（[详细解释见此](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)）。

**模型掌握与多任务处理**：用户交流了在 **LM Studio** 中优化多模态模型功能的知识，处理 **VRAM limitations** 问题，并使用多模型设置来改进复杂任务。对话还包括了关于在具有特定容量的个人机器上实现 "Full GPU Offload Possible" 的模型建议。

**AMD ROCm：追求稳定还是引发风暴？**：**ROCm 0.2.17 Beta v3** 的发布反响不一，成员们报告了与 **ejecting models**、**GPU offloading**、**ZLUDA interference** 以及 **high CPU utilization** 相关的问题。尽管存在这些挑战，仍有几位报告在 **AMD GPUs** 上表现稳定，表明最新的 ROCm beta 版本可能有潜在改进。

**简化 AI 工作流**：工程师们建议探索用于语言模型工作流结构化输出的 [Instructor library](https://github.com/jxnl/instructor)，并分享了将特别微调版的 **OpenChat** 与 dolphin mistral 微调版成功集成的经验，以提高语言建模效率。



---

## [Perplexity AI](https://discord.com/channels/1047197230748151888) Discord

- **模型对决：Claude 3 Opus vs. Gemini**：用户讨论了 **Claude 3 Opus** 与 **Gemini** 之间的性能细微差别，探讨了哪款 AI 感觉更像人类。讨论还延伸到了 **Inflection-2.5** 和 **Pi.AI** 等个人 AI 模型，强调了它们的对话优势以及对其平台未来的担忧。

- **使用 Perplexity 探索 AI**：关于 **Perplexity AI** 如何进行网页搜索和图像生成的查询非常突出，表明用户对 **Unlimited Claude3 Opus** 等功能的移动端可访问性很感兴趣。咨询内容还涉及使用 **Perplexity AI** 探讨从最大的行星到 **GPT-5 发布传闻**等各种话题。

- **社区呼吁更暗的 iOS 主题**：精通技术的 Discord 用户对 **iOS App 更新**中缺乏更暗的午夜/黑色主题表示沮丧，理由是在数字环境中需要视觉舒适度。

- **达到 Token 限制！惨痛的教训**：一名 API 用户因 **6621 token 的 Prompt 和 9900 token 的输出**超过了 Perplexity 的 **16384 token 限制**而导致 BadRequestError，这凸显了在 API 请求中准确计算 Token 数量的重要性。

- **对 Cloudflare 过度验证码（CAPTCHA）的沮丧**：一位用户抱怨 Cloudflare 验证码挑战的侵入性，尤其是在使用 VPN 时，并表示即使是常规浏览也可能触发这些防御机制。

引用的技术参考资料包括 [Inflection-2.5](https://inflection.ai/inflection-2-5)、[Neuralink 首位人类受试者的见解](https://www.businessinsider.com/neuralink-first-human-trial-patient-quadriplegic-elon-musk-x-2024-3)，以及根据 [Analytics India Magazine](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/) 的说法，Perplexity 可能是一个 Google Search 封装器。[Perplexity 文档](https://docs.perplexity.ai/reference/post_chat_completions)被提及用于澄清 Token 计数。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord

- **缺失的人格预测**：对话显示了对 **"mypersonality"** 数据集的兴趣，特别是其在根据文本预测作者人格方面的应用，但对其可访问性存在担忧。

- **为卓越 AI 进行池化（Pooling）**：[Hugging Face diffusers 库](https://github.com/huggingface/diffusers/issues/7365)的 Embedding 实现受到了批评，建议修改文本 Embedding 的池化方法以提升模型性能。

- **数据集未来充满不确定性**：在新的欧盟法规背景下，**LAION-5B** 数据集的移除导致了对 Datacomp 和 DFN 等替代数据集的探索，人们怀疑 LAION 是否能克服法律障碍重新发布其数据集。

- **呼吁 OpenAI 提高透明度**：尽管 OpenAI 此前犹豫不决，但社区预期其可能会开源即将推出的模型（如 **SD3**）的训练代码，这对于追求 AI 进步的人来说是一个重要话题。

- **AI 破坏还是安全？**：成员们对研究人员关注包含敏感材料的数据集的意图持怀疑态度，思考此类行为究竟是不必要的 AI 进步阻碍，还是解决安全问题的真诚努力。

- **建议创新的图像缩放**：[arXiv](https://arxiv.org/pdf/2403.13043.pdf) 上的一项研究建议使用图像的**多尺度（multiple scales）**来增强模型效果，为视觉 AI 工程指明了潜在路径。

- **图像编码的时间技巧**：通过一篇 [arXiv 论文](https://arxiv.org/pdf/2403.13802.pdf)介绍的一种有趣方法采用了**六倍于时间戳数量**的方式对图像进行编码，尽管一些社区成员认为这更多是一种变通方案（workaround）。

- **神秘推文暗示技术趋势**：[Bindu Reddy](https://twitter.com/bindureddy/status/1770934470373421522?t=5dLWWD7d9PN0C4ZjHAY_Tw&s=19) 的一条推文被提及可能暗示了未来的发展，引发了成员们对其影响的好奇。

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord

- **探索 AI 中的“扩展思维” (Extended Mind)**：工程师们讨论了“**Extended Mind**”概念，该概念涉及存储用于联想记忆的向量，并在前向传播 (forward pass) 期间获取前 k 个向量，从而增强模型的推理和记忆能力。这场辩论基于 Phoebe Klett 的推文，而与 **Mistral** 的集成被视为一个极具前景的未来实验。

- **微调挑战与 AI 设备热潮**：一个新的 [YouTube 教程](https://www.youtube.com/watch?v=21Tc92g15pM) 提供了微调 **LLaVA 模型** 的指导；同时，讨论也集中在最新的开源 AI 设备 **01 Light** 上，旨在通过语音控制电脑，该消息分享自 [OpenInterpreter 的推文](https://twitter.com/OpenInterpreter/status/1770821439458840846)。

- **Cosmopedia 和 Quiet-STaR 引起关注**：Hugging Face 博客关于 **Cosmopedia** 的文章展示了为 AI 创建合成数据集的过程；而关于 [Quiet-STaR](https://arxiv.org/abs/2403.09629) 的论文则建议 LM 可以为每个 token 生成解释，从而增强文本解读能力。

- **AI 模型改进工作势头正盛**：工程师们在 Embedding 模型中遇到了 *BatchAllTripletLoss* 性能问题，并分享了开源 Rainfall API (RAG) 平台等项目的进展。讨论还涉及了使用手势甚至直接脑机接口进行 AI 交互的可能性。

- **量化查询与协作进展**：成员们分享了关于模型**量化** (Quantization) 的信息，包括一个用于 4-bit 量化的旧仓库 **AutoAWQ** ([GitHub 链接](https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena))，并思考了 Attention 机制中因果掩码 (causal masking) 的理论基础。

- **数据工具与技术稳步推进**：用户们对 **LanceDB** 的混合搜索能力及生成式界面表示支持，而 **Polars** 等集成技术和一个共享的 **GitHub 仓库** ([Neural-Dragon-AI/Cynde](https://github.com/Neural-Dragon-AI/Cynde)) 展示了将语义智慧与预测性机器学习模型结合的潜力。



---



## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord

- **RAG 辩论升温**：关于**检索增强生成 (RAG)** 与 AI 中基于 Agent 的模型展开了激烈辩论，一些人认为 RAG 仅仅是弥补知识缺失的权宜之计，而另一些人则支持基于 Agent 模型的复杂性和鲁棒性。

- **FastChat 的格式混乱**：**FastChat 的 alpaca** 模型因与 Stanford 的 alpaca 格式不一致而受到关注，促使人们建议提交 Pull Request 以统一格式，详见 [FastChat GitHub 仓库](https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L550)。

- **Galore 的优雅集成**：**Galore 优化器**备受关注，它在微调大模型时具有极高的显存 (VRAM) 效率，最近已[合并至 Hugging Face Transformers](https://github.com/huggingface/transformers/pull/29588)，并且能够以更少的内存占用管理全参数微调，正如 [benchmark issue](https://github.com/jiaweizzhao/GaLore/issues/6) 中所强调的那样。

- **GPT-3.5 查询引发兴趣**：由于处理病人信息等敏感数据的隐私限制绕过方案，导致 Mac 上的本地推理速度变慢，关于 **GPT-3.5** 性能和推理时间的讨论因此展开。

- **文本分类思考**：在文本分类领域，微调 LLM 以生成类别名称作为输出（而非添加分类头）的策略因其灵活性以及鼓励模型遵循**思维链 (chain of thoughts)** 的优势而受到讨论。



---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord

- **AI 进化融合方法揭晓**：Hardmaru 最近发表的一篇 [论文](https://arxiv.org/abs/2403.13187) 介绍了一种用于基础模型融合的 *自动化进化算法*，引发了关于其在无需重度训练的情况下结合开源模型并提升性能潜力的讨论。

- **AI 社区在巴黎蓬勃发展**：成员们积极分享了他们在巴黎参加 AI 见面会的经验和计划，特别是对 [Paris Retrieval Augmented Generation group](https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/) 表现出极大的热情，凸显了当地强劲的数字技术氛围。

- **Zoom 拯救论文俱乐部**：为了解决 Discord 频道中的 **发言权限 (speaker rights)** 问题，有人建议创建一个 Zoom 房间，展示了在面对技术限制时的灵活性。

- **AI 实用工具的创新与讨论**：小组深入探讨了 **llama.cpp** 的潜在 GPU 使用、“pad and pray” 张量维度解决方案，以及 bbycroft.net 提供的用于理解 Transformer 模型的视觉化工具。此外，还展望了关于音乐生成模型和处理大型代码库的讨论。

- **播客揭秘 AI 巨头**：一个包含 OpenAI、Google 和 **Adept** 等公司见解的新播客引起了关注，并辅以一篇 [Twitter 帖子](https://twitter.com/swyx/status/1771255525818397122)。一场名为 *AI In Action* 的 AI 活动重点介绍了 **Llama.cpp**，并邀请通过 [Discord 频道](https://discord.com/channels/822583790773862470/1200548371715342479) 加入。



---



## [LlamaIndex](https://discord.com/channels/1059199217496772688) Discord

**敏感数据与 AI 的安全碰撞**：LlamaIndex 博客强调了使用患者临床报告等敏感数据训练 LLM/RAG 应用的风险，并建议使用 *差分隐私 (differential privacy)* 来保护个人信息，相关见解通过 [博客推文](https://t.co/2ZipwvOwXv) 分享。

**Navarasa 2.0 拥抱多样性**：博客介绍了 Navarasa 2.0，这是针对 15 种印度语言进行微调的升级版 **Google Gemma 7B**，强调了 AI 中本地语言支持的价值，并通过 [发布推文](https://t.co/HHrfonnAr2) 进行了重点介绍。

**UX 变得更智能**：LlamaIndex 上的一款新 UX 模板旨在通过限制 Agent 仅在必要时请求人类输入，从而增强 Agent 与人类的交互，更多信息请见 [相关推文](https://t.co/Z16QPCWFmG)。

**集成难题！**：Discord 成员讨论了将各种工具与聊天机器人集成的复杂性，并遇到了诸如 “BadRequestError” 之类的问题，在激烈的讨论中分享了 [文档建议](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/) 和故障排除建议。

**文档风波**：在 MKDocs 更新期间，用户在访问 LlamaIndex 文档时遇到了困难，分享了 [新文档格式的链接](https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex)，并对 [此处](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/) 详细说明的查询管道 DAG 混淆提供了澄清。



---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord

**寻求紧凑型代码数据集**：[CodeSearchNet corpus](https://huggingface.co/datasets/code_search_net) 被考虑作为预训练数据集，但遇到了上下文长度的问题；相反，[The MiniPile](https://arxiv.org/abs/2304.08442)（一个包含 1M 文档的语料库）因其多样性和紧凑的规模被推荐，适合在性能损失最小的情况下进行预训练。

**揭秘闭源模型内部**：社区讨论了在 Claude 和 Gemini 等闭源模型中无法访问 logprobabilities 和 tokenizers 的问题，这与 OpenAI 等随时提供这些功能的平台形成鲜明对比，并推测这种限制背后的专有原因。

**最大化模型的 GPU 潜力**：一份关于最大化 Transformer 模型 GPU 运行时性能的[最新论文](https://arxiv.org/abs/2401.14489)给出了指南，包括超参数微调和高效的模型形状，可能将吞吐量提高多达 39%。

**AI 进军生物技术领域**：一篇关于 [AI 用于抗体设计](https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them)的 Ars Technica 文章引发了讨论，既展现了对 Diffusion 模型前景的兴奋，也对其实际经济应用持怀疑态度。

**缓解调试难题**：参与者在将 `megatron-deepspeed` 与 `lm-eval 0.3.0` 配合使用时遇到了问题，并提出了变通方案，如从旧版本的 `cais/mmlu` 加载，但由于辅助训练拆分重定位，这仍然存在问题，如 [Gist traceback](https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21) 所示。

---

## [HuggingFace](https://discord.com/channels/879548962464493619) Discord

**ASCII 艺术获得数据集并在 Diffusion 中发展**：随着 [ASCII Art 数据集](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM)的发布，工程师们分享了对 **ASCII Art** 的兴奋，并讨论了微调 LLM 和 Diffusion 模型以生成 ASCII 艺术。一个特别的挑战是微调语言模型以生成复杂的设计，这促使人们寻找高效的训练方法以及开发 ASCII 适配的 Diffusion 模型的想法。

**SMIT 为语言模型引入音频**：引入了一种名为 **SMIT** 的新模态集成工具，使得在语言模型中包含音频变得更加容易。**SMIT** 用于音乐生成模型的 YouTube 演示因其潜在应用引起了关注。同时，**Fluently-v4** 全球发布，为多项任务提供了单一模型解决方案。

**1-bit LLM 承诺高效能**：关于 [1-bit LLM BitNet b1.58](https://arxiv.org/abs/2402.17764) 的论文表明，其性能在与全精度模型匹配的同时，优化了成本效益。这可能会导致针对 LLM 的 1-bit 优化硬件的开发。

**各个 AI 领域的新方法和工具**：SegGPT 的引入增加了图像分割任务的工具集，承诺提供 one-shot 结果。**UniProt 项目** 的 1024 维 embeddings 准备使用 **Matryoshka embeddings** 进行重新训练，以便在蛋白质数据库中获得更好的可搜索性。一项利用数据分析对肥胖趋势进行的[深度探索](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda)为健康相关的 AI 研究树立了新典范。

**模型开发和联邦学习中的社区协作蓬勃发展**：协作需求日益增长，成员们在负荷预测的联邦学习等项目上寻求帮助，分享了用于深度代码生成的 6TB **"The Stack"** 数据集等可能性，并调用 **BERTopic** 进行现代化的主题建模。讨论了对微调模型进行量化（quantizing）的担忧以及 Huggingface 中 Trainer 类的问题，反映了共同克服技术障碍的承诺。

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord

- **关于 AI 成本及其应用的讨论**：成员们讨论了在 Discord 中添加 **Chat GPT 机器人**的成本，以及尽管配置正确但在 **Postman** 中未收到响应的痛点。关于 **Perplexity AI 作为 Google Search 封装层**的热议引发了讨论，引用了 Mohit Pandey 的文章，暗示其总结了 Google Search 的顶级搜索结果。对比了 AI 在**视频压缩**方面的潜力与深度学习超采样（DLSS），并以现有博客文章作为参考。在效率方面，一名成员声称通过将向量数据库的 Float32 嵌入转换为 Int8，实现了 **80% 的存储成本降低** [Deep Compression with AI](https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html) 和 [Perplexity article](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/)。

- **GPT-4 自定义模型与可用性查询**：关于通过 API 连接到**自定义 GPT-3 模型**的咨询引出了分享的 **Assistants API Guide**。一个为用户分配动物化身的 GPT 征求反馈，并提供了 Prompt 示例。**固定自定义 GPT 的数量突然减少到 4 个**让用户感到困惑，这可能是一个未记录的变化。讨论还涉及了将**知识文件分布在多个 GPT 中**与将 Prompt 的不同部分整合到单个 GPT 中的效率对比 [Assistants API Guide](https://help.openai.com/en/articles/8673914-gpts-vs-assistants)。

- **服务器规则和产品描述主导 Prompt Engineering 讨论**：**Rule 7** 成为焦点，在一名用户发布关于 Prompt Engineering 工作的帖子以及另一名用户尝试推广 **Prompt 链/Prompt Engineering 工具包**后，重申了禁止自我推广的准则。对于 **GPT-4 Vision 无法协助残障人士**的情况出现了挫败感，而另一位成员则寻求挑战 ChatGPT **生成自然的产品描述**，建议将任务拆分为生成特定部分可能会更有效。

- **API 频道呼应规则强化和模型限制**：与 Prompt Engineering 频道的讨论类似，API 讨论强调了 **Rule 7**，并对之前的违规行为表示歉意。**GPT-4 Vision** 在识别残障人士方面的局限性催生了关于 AI 包容性的对话。提出了在没有人工监督的情况下使用 ChatGPT 进行**自动化产品描述**的挑战，质疑 AI 生成内容的精确性。



---



## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord

- **Python 依赖难题困扰 Langchain 爱好者**：Python 版本冲突和依赖问题在 **[langchain-ai/weblangchain](https://github.com/langchain-ai/weblangchain)** 中引发了麻烦，`TypeError: Type is not JSON serializable: numpy.float64` 等错误导致程序崩溃。GitHub 上正在跟踪一个相关问题：**[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langchain/discussions/17876)**。

- **寻求序列化解决方案**：尽管使用了 Poetry 并固定了旧版本的 Starlette，`numpy` 序列化问题仍然存在，最终导致了一个新的 GitHub Issue，标题为 **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langserve/issues/551)**，旨在解决 Langchain/Langserve 的不兼容问题。

- **Token 限制引发技术讨论**：Langchain 用户正在探索处理超出模型 Token 限制的大型输出的功能，例如 OpenAI GPT-4-Turbo 的 4k 输出 Token，考虑通过发送额外请求让链（Chains）继续生成输出的方法。

- **Promptsage 旨在优化 Prompt 体系**：一个新项目 [Promptsage](https://github.com/alexmavr/promptsage) 为 LLM 提供了一种简化的 Prompt 构建和清理方法，同时具备安全和隐私护栏，专为兼容 Langchain 而设计。

- **数据分析师赞赏 AI 驱动的演进**：一篇题为《利用 Langchain、Instructor 和 Pydantic：用 AI 重新定义数据分析》的文章赞扬了集成各种工具以增强数据分析的做法。可以在 [Medium](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616) 上阅读相关见解。



---

## [OpenRouter (Alex Atallah)](https://discord.com/channels/1091220969173028894) Discord

- **西海岸用户面临延迟困扰**：**西海岸**的用户正面临请求缓慢的问题，怀疑与云服务故障有关；目前正在进行调查。

- **Gemini 1.5 Pro 引发关注与咨询**：尽管[官方文档](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning)除 1.0 版本外未提及其他信息，但关于 **Google 的 Gemini 1.5 Pro** 及其令人印象深刻的 100 万词上下文窗口的讨论非常热烈；部分成员已联系 Google 申请访问权限。

- **模型对决：C3 vs. Claude 3 vs. GPT-4**：工程师们对模型进行了辩论，**C3 Model** 因其不稳定性受到批评，而 **Claude 3** 的一个自我审核变体在内容审核方面被认为可与 **GPT-4** 媲美。

- **对 Grok AI 性能的看法分歧**：关于 **Grok AI** 的评价出现分歧，有人批评其可能训练不足且成本高昂，而另一些人则为其作为基础模型的能力辩护，认为它不应直接与 Mixtral 等经过聊天微调（chat-tuned）的模型进行比较。

- **Grok 的基准测试和公开测试引发辩论**：工程师们讨论了 **Grok AI** 基准测试的价值，并分享了试用该模型的[链接](https://grok.x.ai/)，强调通过 xAI 平台可能无需 Twitter Premium+ 即可访问。讨论还涉及哪些内容最适合评估 Grok 的性能。



---



## [CUDA MODE](https://discord.com/channels/1189498204333543425) Discord

- **用于机器学习加速的 Nanobinding**：在讨论中，**nanobind** 被推荐用于提高机器学习的效率，特别是针对 MLX。与此同时，成员们在 GTC 活动期间遇到了 Discord 舞台频道的问题，建议未来转向语音频道以避免类似问题。

- **前沿的优化器与编译器**：一位成员透露成功将 **GaLore 的 Adam 优化器** 与 Triton 融合，以提高模型的显存效率，并提供了 [GitHub pull request](https://github.com/jiaweizzhao/GaLore/pull/29) 支持。另外，[micrograd-cuda 库](https://github.com/mlecauchois/micrograd-cuda)被引入用于 CUDA 加速基于 Python 的 micrograd 扩展，而 PyTorch 编译器 [Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder) 因其在加速器上显著的性能提升而受到关注。

- **矩阵乘法、求和与标准启发**：社区分析了用于增强矩阵乘法的 *Ozaki scheme*，并得到了 Jeremy Howard 的认可，同时讨论了用于减少计算误差的 [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)。此外，IEEE 754 浮点标准被指出至关重要，并引用了一篇关于该主题的 [ITU 论文](https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf)。

- **虚拟对话规范与知识孵化**：一位成员建议使用结构化消息以提高对话清晰度，并向在另一个服务器上以此为典范的 <@272654283919458306> 致敬。在教育方面，分享了一个关于 PPAM 2022 的 [Springer 图书链接](https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes)，提供了通往并行处理当代进展的门户。

- **CUDA 知识寻求者的分享与幽默**：一位正在确认 "pmpp-book" **第 2 章练习题** 的成员建议通过私信验证答案。在轻松的一面，GTC 上发布了一个针对 Python 和 PyTorch 用户的全新 [Zero to Thunder 教程](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial)，同时有人观察到新的 Blackwell GPU 设计看起来像笑脸，引发了 [Twitter](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908) 上的趣谈。

- **Triton 的顽强排错**：在 **triton-puzzles** 频道中，社区解读了张量颜色编码，并讨论了越界指示器中潜在的误导。`tl.exp` 算子的问题引发了关于解释器模式下 NotImplementedError 的讨论，Triton puzzles 的工作取得进展，完成了 Puzzle 3 并对 Puzzle 4 进行了协作调试。



---

## [LLM Perf Enthusiasts AI](https://discord.com/channels/1168579740391710851) Discord

- **GPT4.Turbo 的匹配功能故障**：Uniti AI 正在努力解决 **GPT4.Turbo** 错误建议房产空间的问题，匹配误差非常明显，例如在请求 2,000 - 4,000 平方英尺时却提供了 17,000 平方英尺。当试图遵循特定的房产面积百分比范围时，挑战进一步加剧，这促使人们建议采用更简单的解决方案，如 **直接 SQL 查询**。

- **警惕“常见的 LLM 陷阱”**：工程师们讨论了在某些任务中过度使用 LLM 的潜在问题，而这些任务通过基础数据库查询可能效率更高。分享了 Jason Liu 关于 [Retrieval Augmented Generation (RAG)](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/) 的博客文章，强调了将 LLM 与标准数据库交互相结合如何改进日期范围提取等任务。

- **对 Claude 而言，直接集成优于 Bedrock**：在 AI 接口领域，一位用户报告称，与使用 **Bedrock** 等框架相比，与 AI 模型 **Claude** 进行 **直接集成** 更为理想，理由是可靠性和运行时间表现更好。即使是拥有优先速率限制、绕过了超过 20 万等待名单的用户，也选择了与 Claude 直接连接。

- **Jeffreyw128 和 ibash 留下神秘评论**：在讨论中，jeffreyw128 的“lol wut”和 ibash 对高质量代码编写的一字评价“Damn”等简洁消息点缀了对话，但提供的上下文或可讨论的观点有限。

- **基础 Prompting 是否不足？**：一条孤立的消息质疑了基础 Prompting 的有效性，暗示在与 AI 交互时需要更高级或更细致的技巧，特别是对于技术领域的从业者。



---



## [Interconnects (Nathan Lambert)](https://discord.com/channels/1179127597926469703) Discord

- **合成基准测试 (Synthetic Benchmarks) 的探索升温**：工程师们正在研究 **全合成基准测试 (fully synthetic benchmarks)** 以研究语言模型的能力，初创公司正在生成数据以支持这项研究。目标是通过操纵训练数据中的多样性和推理等因素，更好地理解 LLM 的能力。

- **工程师们热议合成数据和开放策展**：**合成数据和合成世界 (synthetic data and worlds)** 领域引起了广泛关注；一位工程师甚至考虑为此撰写论文。此外，有人建议采用系统化的开源数据策展方法进行模型预训练，以改进该领域的集体努力。

- **ChatGPT：学术界的新助手**：讨论强调了在学术项目中使用 **ChatGPT** 重写内容以追求 **State-of-the-art 结果**，目前正在进行一个 **Side project** 以探索进一步的应用，这表明重写任务现在已成为一种主流策略。

- **国际象棋、围棋与人类心理：AI 驱动世界中的科技巨头**：成员们思考了 AI 进步带来的心理影响，引用了卡斯帕罗夫输给 Deep Blue 等历史事件，并反思了个人对 AI 的态度。重点讨论了在 Reinforcement Learning 中创建通用 Agent 的潜力，其中包含了 **Minqi Jiang** 和 **Marc Rigter** 的见解，并通过 [MLStreetTalk 推文](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw) 进行了分享。



---



## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord

**召集所有开源爱好者**：一位社区成员正在寻求关于 **the 01**（一款完全开源的硬件设备）的合作，并在 [公开推文](https://twitter.com/OpenInterpreter/status/1770821439458840846) 中分享了细节。

---

# PART 2: Detailed by-Channel summaries and links



**Stability.ai (Stable Diffusion) ▷ #[general-chat](https://discord.com/channels/1002292111942635562/1002292112739549196/1220272884610502667)** (884 messages🔥🔥🔥):

- **模型热潮开启**：成员们正在热烈讨论 **Stable Diffusion models**，特别是对 **Stable Diffusion 3 (SD3)** 的期待，以及为艺术创作精心选择 **control nets** 和插件。关于 **AMD GPU compatibility** 的问题以及针对硬件配置较低用户的 **cloud GPU services** 建议非常普遍。
- **技术故障排除进行中**：一位成员在尝试使用 **Stable Diffusion WebUI** 时，在 AMD GPU 系统上遇到了关于 **NVIDIA drivers** 的 **RuntimeError**。他们被引导至 [lshqqytiger's fork](https://github.com/lshqqytiger/stable-diffusion-webui-directml) 以获取 AMD 支持，并获得了详细的安装步骤指南。
- **对更高质量的热捧**：讨论转向了关于不同 **Stable Diffusion models** 的 **V-RAM requirements** 的技术细节。随着即将发布的 SD3 被认为需要高 VRAM，成员们正在推测在本地运行此类大型模型的实际可行性。
- **Prompt 编写与艺术创作**：用户正在分享各种创意项目的 **prompting techniques** 和 **AI results**，例如为“部落视频”和 **D&D campaigns** 生成图像，一些人正在寻找能够理解详细 prompt 以生成角色艺术和场景的特定模型。
- **社区观点的多样性**：围绕 AI 的优缺点展开了辩论，一些人对 **AI's impact** 对就业和创造力的影响表示怀疑。与此同时，其他人强调了 **evolutionary nature of AI tools** 及其增强人类工作流程的潜力。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>

<li><a href="https://platform.stability.ai/">Stability AI - Developer Platform</a>: 未找到描述</li><li><a href="https://www.runpod.io/console/gpu-cloud">no title found</a>: 未找到描述</li><li><a href="https://app.suno.ai/song/8250b732-8f32-4be1-a38e-9d1c23f926b5/">Kitty Cat Groove | Suno</a>: 雷鬼、舞厅乐风格歌曲。在 Suno 聆听并创作你自己的作品。</li><li><a href="https://civitai.com/articles/1997/comfyui-guide-to-stacker-nodes">ComfyUI - Stacker Nodes 指南 | Civitai</a>: 这篇文章是关于 Stacker Nodes 及其在工作流中的使用方法。适用于 ComfyUI 的初学者和高级用户。Stacker nodes 是 ...</li><li><a href="https://tenor.com/view/arch-arch-linux-btw-i-use-arch-btw-monokuma-gif-26738028">Arch Arch Linux GIF - Arch Arch Linux Btw - 发现并分享 GIF</a>: 点击查看 GIF</li><li><a href="https://app.suno.ai/song/78ab7dff-a3fc-4038-90e2-d15cec8590d8/">Generative AI | Suno</a>: 流行歌曲。在 Suno 聆听并创作你自己的作品。</li><li><a href="https://app.suno.ai/song/7df3f524-39f9-4678-bee0-ba6ab4175417">Schnappi rock version | Suno</a>: 摇滚、硬核、Breakcore、摇摆电子乐风格歌曲。在 Suno 聆听并创作你自己的作品。</li><li><a href="https://app.suno.ai/song/40e83bad-3ef2-40bf-aa89-8f49a5a981c1">BABY SHARK | Suno</a>: 前卫摇滚、电吉他、电贝斯、切分音风格歌曲。在 Suno 聆听并创作你自己的作品。</li><li><a href="https://civitai.com/models/350524/jboogx-and-the-machine-learners-animatelcm-subject-and-background-isolation-via-invertmask-vid2vid-highresfix">JBOOGX &amp; THE MACHINE LEARNER&#x27;S ANIMATELCM SUBJECT &amp; BACKGROUND ISOLATION via INVERTMASK VID2VID + HIGHRESFIX - v1.0 | Stable Diffusion 工作流 | Civitai</a>: 这是我的 AnimateLCM 工作流的进化版。经过精简和整合以方便使用。该工作流最多需要 12-14GB 的 VRAM...</li><li><a href="https://civitai.com/models/38784/controlnet-11-models">ControlNet 1.1 模型 - Tile (e) | Stable Diffusion Controlnet | Civitai</a>: 停止！这些模型不用于提示词/图像生成。这些是 ControlNet 扩展所需的全新 ControlNet 1.1 模型，已转换...</li><li><a href="https://civitai.com/models/38784?modelVersionId=44756">ControlNet 1.1 模型 - Softedge | Stable Diffusion Controlnet | Civitai</a>: 停止！这些模型不用于提示词/图像生成。这些是 ControlNet 扩展所需的全新 ControlNet 1.1 模型，已转换...</li><li><a href="https://app.suno.ai/song/f435f67b-e5e3-4ef3-bd5e-c9764b90b550">Listen | Suno</a>: 工业金属风格，带有厚重的吉他和警报声般的合成器节奏。包含钢琴对位桥段。在 Suno 聆听并创作你自己的作品。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Install-and-Run-on-AMD-GPUs">在 AMD GPU 上安装与运行</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=RwLY0bsQpx4&t=1s">在最佳免费 COLAB 替代方案上运行的一键式 STABLE DIFFUSION NOTEBOOK</a>: 各位 Stable Diffusion 爱好者们，大家好！创建一个免费的 Kaggle 账户并验证你的手机号码。你可以从我的 Discord 频道免费下载该 Notebook：h...</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui/discussions/14029">Automatic1111 stable diffusion dreamartist 无法工作。我是新来的，请求帮助 🙏 :) · AUTOMATIC1111/stable-diffusion-webui · Discussion #14029</a>: *** 加载脚本错误：dream_artist_main.py Traceback (most recent call last): File &quot;D:\automatic1111\sd.webui\webui\modules\scripts.py&quot;, line 383, in load_scripts script_module = script_...</li><li><a href="https://github.com/castorini/daam">GitHub - castorini/daam: 用于解释 Stable Diffusion 的扩散注意力归因图（Diffusion attentive attribution maps）。</a>: 用于解释 Stable Diffusion 的扩散注意力归因图。 - castorini/daam</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika 是一个 Agentic AI 软件工程师，能够理解人类的高级指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标。Devika 旨在成为 Cognition AI 的 Devin 的竞争性开源替代方案。</a>: Devika 是一个 Agentic AI 软件工程师，能够理解人类的高级指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标...</li><li><a href="https://github.com/lshqqytiger/stable-diffusion-webui-directml">GitHub - lshqqytiger/stable-diffusion-webui-directml: Stable Diffusion web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建账户来为 lshqqytiger/stable-diffusion-webui-directml 的开发做出贡献。</li><li><a href="https://github.com/AUTOMATIC1111/stable-diffusion-webui">GitHub - AUTOMATIC1111/stable-diffusion-webui: Stable Diffusio</a>

n web UI</a>: Stable Diffusion web UI。通过在 GitHub 上创建一个账户来为 AUTOMATIC1111/stable-diffusion-webui 的开发做出贡献。</li><li><a href="http://www.replicate.com">Replicate</a>: 使用云端 API 运行开源机器学习模型。</li><li><a href="https://stable-diffusion-art.com/controlnet/">ControlNet: A Complete Guide - Stable Diffusion Art</a>: ControlNet 是一个通过添加额外条件来控制 Stable Diffusion 图像生成的神经网络。详情请参阅文章。</li><li><a href="https://azure.microsoft.com/en-us/free/ai-services/">立即创建您的 Azure 免费账户 | Microsoft Azure</a>: 开始使用 12 个月的免费服务、40 多项始终免费的服务以及 200 美元的额度。立即在 Microsoft Azure 创建您的免费账户。
</li>
</ul>

</div>
  

---



**Unsloth AI (Daniel Han) ▷ #[general](https://discord.com/channels/1179035537009545276/1179035537529643040/1220289632831279135)** (696 messages🔥🔥🔥): 

- **Unsloth AI 正在筹备 Multi-GPU 支持**：Unsloth AI 团队确认 Multi-GPU 支持最终将作为开源功能提供，旨在允许在 Kaggle 等平台上免费使用 Mixtral。目前的重点仍然是启动 Unsloth Studio (Beta)。

- **改进用于 Fine-Tuning 的数据清洗（Data Curation）**：Unsloth AI 正在探索创建一个高效的平台 UI，用于自动数据清洗，目标用户是那些觉得为模型 Fine-Tuning 准备数据具有挑战性的用户。该平台旨在解决数据格式化和问答准备步骤。

- **关于评估框架和数据质量的辩论**：进行了一场关于建立稳健评估框架的重要性，以及定义和获取高质量数据进行模型训练所面临挑战的漫长讨论。一个重要的部分是确保基准测试（Benchmarks）的透明度和准确性，例如纠正所使用的数据集，比如 MMLU 中有 25% 的示例参考答案是错误的。

- **尽管遇到挫折，社区支持依然坚定**：尽管之前社区中存在误导信息传播的情况，但 Unsloth AI 已经获得了显著的关注和支持，其 VRAM 减少技术已得到广泛认可。热情的社区成员对即将推出的 Multi-GPU 支持和其他功能表示期待。

- **庆祝协作与开源贡献**：提到了 OpenInterpreter 的第一批产品售罄，其利润被重新分配给开源贡献者等项目。这突显了 AI 工具社区内协作和再投资的积极趋势。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://unstructured-io.github.io/unstructured/index.html">Unstructured 0.12.6 文档</a>: 未找到描述</li><li><a href="https://inflection.ai/inflection-2-5">Inflection-2.5: 遇见世界上最好的个人 AI</a>: 我们是一家 AI 工作室，为每个人创造个人 AI。我们的第一个 AI 名为 Pi，代表个人智能（personal intelligence），是一个具有支持性和同理心的对话式 AI。</li><li><a href="https://huggingface.co/stabilityai/stablelm-2-1_6b">stabilityai/stablelm-2-1_6b · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/NousResearch/Genstruct-7B">NousResearch/Genstruct-7B · Hugging Face</a>: 未找到描述</li><li><a href="https://www.reddit.com/r/LocalLLaMA/comments/18xz9it/augmentoolkit_easily_generate_quality_multiturn/">Reddit - 深入探索</a>: 未找到描述</li><li><a href="https://huggingface.co/ISTA-DASLab">ISTA-DASLab (IST Austria 分布式算法与系统实验室)</a>: 未找到描述</li><li><a href="https://x.com/DavidSHolz/status/1770697881160179786">David (@DavidSHolz) 的推文</a>: @theashbhat 正在生成文本</li><li><a href="https://unsloth.ai/blog/mistral-benchmark">Unsloth 更新：支持 Mistral 及更多</a>: 我们很高兴发布对 Mistral 7B、CodeLlama 34B 以及所有其他基于 Llama 架构模型的 QLoRA 支持！我们添加了滑动窗口注意力（sliding window attention）、初步的 Windows 和 DPO 支持，以及...</li><li><a href="https://github.com/huggingface/trl/issues/862">SFTTrainer 中生成任务的计算指标 · Issue #862 · huggingface/trl</a>: 你好，我想在 SFTTrainer 中包含一个自定义的基于生成的 compute_metrics，例如 BLEU。但是，我遇到了困难，因为：compute_metrics 的输入 eval_preds 包含一个 .predicti...</li><li><a href="https://github.com/InflectionAI/Inflection-Benchmarks">GitHub - InflectionAI/Inflection-Benchmarks: 公开的 Inflection 基准测试</a>: 公开的 Inflection 基准测试。通过在 GitHub 上创建一个账户来为 InflectionAI/Inflection-Benchmarks 的开发做出贡献。</li><li><a href="https://datadreamer.dev/docs/latest/datadreamer.steps.html#datadreamer.steps.RankWithPrompt">DataDreamer</a>: 未找到描述
</li>
</ul>

</div>
  

---

**Unsloth AI (Daniel Han) ▷ #[random](https://discord.com/channels/1179035537009545276/1179039861576056922/1220308710187077662)** (35 条消息🔥): 

- **Unsloth AI 安装故障**：一位用户在使用 `pip` 从 nightly 版本安装 **Unsloth AI** 时遇到问题，报错称找不到符合指定要求的发行版本。该问题涉及一个名为 "kaggle-new" 的特定 extra。
  
- **单显卡训练失败**：另一位用户在将训练限制在单张 GPU 卡时遇到错误。错误信息显示设备混用导致崩溃：*RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:1 and cuda:0!*

- **量化模型的 CUDA 层级变化？**：一位用户在运行 Unsloth 的 solar-10.7b-bnb-4bit 出现问题后，询问 4bits 量化模型在 CUDA 层级是否发生了变化，该模型此前在其机器上运行正常。

- **Solar 模型的 VRAM 限制**：据观察，尽管是量化模型（理应占用更少 VRAM），Unsloth 的 solar-10.7b-bnb-4bit 仍可能超出了用户 15GB A4000 GPU 的可用 VRAM，从而导致显存溢出（out-of-memory）问题。

- **需要重启 Kernel 以避免 32-bit 警告**：注意到需要反复重启 kernel 才能避免关于 32-bit 处理的警告，尽管预期的量化模型不应触发此类警告。推测该机器可能存在内存不足的情况。

**提到的链接**：<a href="https://huggingface.co/docs/transformers/main/en/main_classes/quantization#offload-between-cpu-and-gpu">Quantization</a>：未找到描述

  

---


**Unsloth AI (Daniel Han) ▷ #[help](https://discord.com/channels/1179035537009545276/1179777624986357780/1220265072111255613)** (92 条消息🔥🔥): 

- **切换到 16-bit LoRA 配置**：成员们讨论了更改 LoRA 设置，建议将 `load_in_4bit` 设置为 `false`，或使用 `load_in_8bit` 或 `load_in_16bit` 等参数。

- **训练期间的 VRAM 消耗和显存溢出问题**：一位成员报告在评估期间出现显存溢出（OOM）错误，但在训练期间没有。建议尝试将 "adamw_8bit" 更改为 "paged_adamw_8bit"，并减小 batch size 以降低 VRAM 占用。

- **在本地保存和加载模型及 Tokenizer**：一位成员发现，为了有效地使用 `FastLanguageModel.from_pretrained()`，模型和 tokenizer 都需要保存在同一个文件夹中。

- **Unsloth 中可能缺失 `protobuf` 依赖**：一位成员担心某个特定版本的 Unsloth 可能缺少 `protobuf`，这一点得到了认可，但尚不确定是否属实。

- **物理、数学和工程领域的模型选择不明确**：一位成员就适合高级物理、数学、工程和 Python 的 AI 模型选择寻求建议，得到的推荐是查看 YouTube 视频和文章等可用资源，例如 [Towards Data Science 上的这篇文章](https://towardsdatascience.com/fine-tune-google-gemma-with-unsloth-and-distilled-dpo-on-your-computer-ca1ce8828122)。

- **Unsloth 库更新和环境管理的挑战**：多位成员遇到了与 Unsloth 更新相关的问题，并建议升级必要的库；同时有一位成员描述了环境管理的困难以及对依赖项进行大规模检修的需求。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/TimDettmers/bitsandbytes/blob/main/bitsandbytes/autograd/_functions.py#L488">bitsandbytes/bitsandbytes/autograd/_functions.py at main · TimDettmers/bitsandbytes</a>：通过 PyTorch 的 k-bit 量化实现可访问的大语言模型。- TimDettmers/bitsandbytes</li><li><a href="https://github.com/unslothai/unsloth">GitHub - unslothai/unsloth: 2-5X faster 70% less memory QLoRA &amp; LoRA finetuning</a>：速度快 2-5 倍，显存占用减少 70% 的 QLoRA 和 LoRA 微调 - unslothai/unsloth
</li>
</ul>

</div>
  

---


**Unsloth AI (Daniel Han) ▷ #[showcase](https://discord.com/channels/1179035537009545276/1179779344894263297/1220538856206504116)** (30 条消息🔥):

- **展示 Samantha Mistral Instruct 7B**：cognitivetech 重点介绍了他们在 [Samantha Mistral Instruct 7B](https://huggingface.co/cognitivetech/samantha-mistral-instruct-7b_bulleted-notes) 上的工作，该模型旨在每次处理 2000 个 token 来总结心理学书籍，并感谢了社区的支持。
- **感谢社区指导**：对在使用 Unsloth notebooks 过程中获得的帮助，以及社区在回答模型微调（fine-tuning）相关问题时提供的协助表示感谢。
- **排除模型问题**：cognitivetech 讨论了 q4 量化（quantization）出现的问题，指出它产生的是“垃圾”结果，而不像 `model-unsloth.Q8_0.gguf` 模型在总结书籍时表现完美。
- **上传可用的量化模型**：在讨论了模型故障排除后，cognitivetech 通知他们将在大约 20 分钟内向 Hugging Face 上传一个可用的 q8 版本供他人检查。
- **社区协作与测试**：cognitivetech 和 solobsd 合作在 GPT4All 等平台上测试和运行模型，分享了 Ollama 模板，并讨论了遇到问题的潜在原因。

**提到的链接**：<a href="https://huggingface.co/blog/cognitivetech/samantha-mistral-instruct-7b-bulleted-notes">Samantha Mistral Instruct 7b - Comprehensive Bulleted Notes</a>：未找到描述

---

**Unsloth AI (Daniel Han) ▷ #[suggestions](https://discord.com/channels/1179035537009545276/1180144489214509097/1220382977733955705)** (14 messages🔥): 

- **Lightning Thunder 引起关注**：一名成员分享了 [Lightning Thunder 的链接](https://t.co/P6pvGJBugB)，强调了它通过利用不同的硬件执行器来加速 PyTorch 模型的潜力。但他们指出，由于它是构建在 Triton 之上的，可能对 Unsloth AI 没有直接帮助。
- **对 Unsloth 实现方式的困惑**：一些成员担心 Lightning Thunder 没有正确实现 Unsloth，建议他们本可以咨询 Unsloth 团队以获得更好的集成效果。
- **Unsloth Kernels 的潜在误用**：一名成员指出 Lightning Thunder 在使用 Unsloth kernels 时存在问题，例如不必要的复制和转置，并强调咨询本可以避免这种处理不当。
- **呼吁合作与澄清**：有建议提出联系 Lightning Thunder 团队以纠正错误，并澄清在其演示中对 Unsloth 的使用，强调在 Benchmark（基准测试）中进行准确比较的重要性。
- **对性能对比的沮丧**：一名成员通过 [Twitter 链接](https://twitter.com/danielhanchen/status/1770863091296632921)表达了对不准确对比的沮丧，这种对比使 Unsloth kernels 看起来性能不佳，并敦促演示应反映正确的实现。

**提到的链接**：<a href="https://t.co/P6pvGJBugB">GitHub - Lightning-AI/lightning-thunder: Source to source compiler for PyTorch. It makes PyTorch programs faster on single accelerators and distributed.</a>：PyTorch 的源码到源码编译器。它使 PyTorch 程序在单个加速器和分布式环境下运行更快。- Lightning-AI/lightning-thunder

---

**OpenInterpreter ▷ #[general](https://discord.com/channels/1146610656779440188/1147665339266650133/1220288094050451487)** (254 messages🔥🔥): 

- **OpenInterpreter Discord 频道活动频繁**：成员们热烈讨论 OpenInterpreter Discord 聊天机器人的消息，由于时区差异，一些人为了跟上正在进行的讨论而难以入睡。
- **发布期待与预订查询**：成员们对 01 Light 的发布表现出极大的热情，询问预订事宜，并希望除了目前仅限美国的选项外，还能提供国际运输方案。
- **技术爱好者社区支持硬件创新**：分享了 01 Light 的 **3D 打印设计**链接，鼓励 DIY 爱好者构建自己的语言模型计算机，**设计文件可在 [Printables](https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model)** 找到，更多信息请访问 [GitHub - OpenInterpreter/01](https://github.com/OpenInterpreter/01)。
- **开发与安全讨论升温**：关于 **OpenInterpreter 开发流程**和**安全措施**的讨论非常热烈，成员们对红队（red-teaming）计划和防护措施感到好奇，并引导他人前往 **OpenInterpreter/01** [GitHub 仓库](https://github.com/OpenInterpreter/01)了解更多详情。
- **关于 Windows 支持的社区协作与疑问**：用户询问关于在 Windows 上运行 OpenInterpreter 以及是否会有官方 Windows 支持的问题，对话中没有直接确认此类支持的回复。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://x.com/hellokillian/status/1757526563879587995?s=20).">来自 killian (@hellokillian) 的推文</a>：..天哪，Open Interpreter 的首个 Vision 模型，正在我的 8GB M1 MacBook 上运行。100% 离线。这将进入世界上的每一台电脑。</li><li><a href="https://www.amazon.com/dp/B06XT1Z9TF.">未找到标题</a>：未找到描述</li><li><a href="https://x.com/hellokillian?s=21&t=G6jp7iOBtkVuyhaYmaDb0w">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://docs.openinterpreter.com/language-models/hosted-models/openai">OpenAI - Open Interpreter</a>：未找到描述</li><li><a href="https://kidger.site/thoughts/jaxtyping/">不再有形状错误！针对 Tensor/数组形状和数据类型的类型注解。</a>：TL;DR：你可以显式使用形如 def f(x: Float[Tensor, "channels"], y: Float[Tensor, "channels"]): ... 的类型注解来指定 Tensor/数组的形状和数据类型；声明...</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/ROADMAP.md">OpenInterpreter/01 的 01/ROADMAP.md</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter/blob/main/interpreter/terminal_interface/utils/count_tokens.py">OpenInterpreter/open-interpreter 的 open-interpreter/interpreter/terminal_interface/utils/count_tokens.py</a>：计算机的自然语言接口。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://x.com/altryne/status/1770835426384715803?s=46">来自 Alex Volkov (Thursd/AI) (@altryne) 的推文</a>：https://twitter.com/i/spaces/1YpKkwdyWjdKj</li><li><a href="https://youtu.be/YxiNUST6gU4?si=fSBtR7Tw6WCvWNvN">介绍 Light 01：Open Interpreter 推出的全球首款个人 AI 助手（完整设置）</a>：在这段视频中，我们将查看 OpenInterpreter Light 01 的 GitHub 仓库，这是一个正在彻底改变我们与计算机交互方式的前沿项目...</li><li><a href="https://github.com/OpenInterpreter/01">GitHub - OpenInterpreter/01：开源语言模型计算机</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/open-interpreter">GitHub - OpenInterpreter/open-interpreter：计算机的自然语言接口</a>：计算机的自然语言接口。通过在 GitHub 上创建账号来为 OpenInterpreter/open-interpreter 的开发做出贡献。</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-mod">01 Light - 第 1 版：由 01 推出的全球首款语言模型计算机 | 下载免费 STL 模型 | Printables.com</a>：01 项目展示了 01 Light v1 | 下载免费的 3D 打印 STL 模型</li><li><a href="https://github.com/patrick-kidger/jaxtyping">GitHub - patrick-kidger/jaxtyping：针对 JAX/NumPy/PyTorch 等数组形状和数据类型的类型注解及运行时检查。</a>：针对 JAX/NumPy/PyTorch 等数组形状和数据类型的类型注解及运行时检查。https://docs.kidger.site/jaxtyping/ - patrick-kidger/jaxtyping</li><li><a href="https://github.com/stanfordnlp/dspy">GitHub - stanfordnlp/dspy：DSPy：用于编程（而非提示）基础模型的框架</a>：DSPy：用于编程（而非提示）基础模型的框架 - stanfordnlp/dspy</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model">01 Light - 第 1 版：由 01 推出的全球首款语言模型计算机 | 下载免费 STL 模型 | Printables.com</a>：01 项目展示了 01 Light v1 | 下载免费的 3D 打印 STL 模型</li><li><a href="https://www.thingiverse.com/thing:6529845">01 Light - 第 1 版：由 openinterpreter 推出的全球首款语言模型计算机</a>：设计概览：01 Light 是有史以来第一台语言模型计算机。第一版 01 Light 的设计时尚且符合人体工程学，并且没有屏幕。内部有多个凹槽和...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/hardware/light">OpenInterpreter/01 的 01/hardware/light</a>：开源语言模型计算机。通过在 GitHub 上创建账号来为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[O1](https://discord.com/channels/1146610656779440188/1194880263122075688/1220277189635735563)** (286 条消息🔥🔥):

- **发布会期待**：成员们对 [01 Developer Preview](http://openinterpreter.com/01) 表达了兴奋之情。01 Light 是一款便携式语音交互设备，用于控制家用电脑，具备查看屏幕、使用应用和学习新技能的能力。
- **DIY 你的 01**：社区成员讨论了根据 [物料清单 (BOM)](https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md) 组装自己的 01 设备，并排查了发往印度和欧盟等地区的潜在运输问题，提出了本地化社区协作的可能性。
- **设置查询与故障排除**：对话解决了在各种操作系统上使用 01 的设置疑虑。Windows 用户的一个关键解决方案是运行 `poetry run 01 --client --client-type linux --server-host localhost`，这表明在使用 Linux 客户端类型设置时，该设备可以兼容 Windows。
- **批次更新与发货关注**：社区获悉预订批次已满，并对发货时间表示好奇。人们询问第 2 批及后续批次何时发货，目前尚未给出确切的承诺日期。
- **不断发展的软件讨论**：成员们询问了软件的各种功能和易用性，包括本地与云端运行、非开发人员的可访问性、API keys 以及对德语等语言的兼容性。关于软件更新和电池寿命的问题也受到了关注。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://01.openinterpreter.com/getting-started/introduction">Introduction - 01</a>：未找到描述</li><li><a href="https://01.openinterpreter.com/getting-started/setup#captive-portal">无标题</a>：未找到描述</li><li><a href="https://x.com/hellokillian">来自 undefined 的推文</a>：未找到描述</li><li><a href="https://www.openinterpreter.com/01">The 01 Project</a>：01 Project 是一个用于家用电脑的语音界面。</li><li><a href="https://01.openinterpreter.com/getting-">Introduction - 01</a>：未找到描述</li><li><a href="https://www.youtube.com/@MikeBirdTech/videos">Mike Bird</a>：AI 工程</li><li><a href="https://tenor.com/view/her-theodore-joaquin-phoenix-scarlett-johannson-samantha-gif-5203383">Her Theodore GIF - Her Theodore Joaquin Phoenix - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuYTxMM?typeform-source=pcr08jir95k.typeform.com">Contact Us</a>：使用 Typeform 将数据收集转化为一种体验。创建精美的在线表单、调查、测验等。免费试用。</li><li><a href="https://tenor.com/view/shut-up-and-take-my-money-futurama-fry-take-my-money-money-gif-15195954">Shut Up And Take My Money Futurama GIF - Shut Up And Take My Money Futurama Fry - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://fxtwitter.com/OpenInterpreter/status/1770821439458840846">来自 Open Interpreter (@OpenInterpreter) 的推文</a>：介绍 01 Developer Preview。立即订购或构建你自己的设备：http://openinterpreter.com/01 。01 Light 是一款控制家用电脑的便携式语音界面。它可以看见你的屏幕...</li><li><a href="https://github.com/OpenInterpreter/01/blob/main/hardware/light/BOM.md">01/hardware/light/BOM.md at main · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://github.com/OpenInterpreter/01/issues">Issues · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。</li><li><a href="https://0ggfznkwh4j.typeform.com/to/WfuY">Discover Typeform, where forms = fun</a>：无需代码，在几分钟内创建精美、互动的表单。免费开始使用。</li><li><a href="https://www.printables.com/model/803461-01-light-version-1-the-worlds-first-language-model">01 Light - Version 1: The World's First Language Model Computer by 01 | 下载免费 STL 模型 | Printables.com</a>：01 Project 展示 01 Light v1 | 下载免费 3D 打印 STL 模型</li><li><a href="https://www.thingiverse.com/thing:6529845">01 Light - Version 1: The World&#039;s First Language Model Computer by openinterpreter</a>：设计概述 01 Light 是有史以来第一台语言模型计算机。第一版 01 Light 的设计简洁且符合人体工程学，并且没有屏幕。内部有多个凹槽和专业...</li><li><a href="https://github.com/OpenInterpreter/01/tree/main/hardware/light">01/hardware/light at main · OpenInterpreter/01</a>：开源语言模型计算机。通过在 GitHub 上创建账号为 OpenInterpreter/01 的开发做出贡献。
</li>
</ul>

</div>
  

---


**OpenInterpreter ▷ #[ai-content](https://discord.com/channels/1146610656779440188/1149229778138824765/)** (1 条消息):

cyanidebyte: https://www.youtube.com/watch?v=Q_p82HtBqoc
  

---



**LM Studio ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/1220269548259901440)** (305 条消息🔥🔥): 

- **LM Studio 本地文档支持查询**：一位用户询问了是否可以通过 Anything-LLM 和 LM Studio 支持将本地文档用于检索增强生成 (RAG)，但随后没有提供关于该功能的进一步回答。
- **关于 LM Studio 多模型问题的担忧**：用户报告了 LM Studio 新 beta 版本中无法添加多个模型的问题，还有一名用户在将所有层卸载到 GPU 的情况下仍遇到 CPU 占用过高的问题，该问题通过重启系统得到了解决。
- **更新后 LM Studio 模型加载错误**：一位用户描述了 LM Studio 无法识别非本地模型名称并导致错误的问题，另一位用户建议已加载的模型现在会生成一个通过 GET 端点可见的静态键名 (key name)，该问题在分享的消息中未得到直接解决。
- **LM Studio Playground 中消失的图标**：另一位用户遇到了界面行为问题，即在 LM Studio 的 Playground 不同部分之间导航时，模型名称会从 UI 中弹出。文中提到了一种解决方法，建议在想要重新加载模型时仅点击一次黄色的“Reload”框。
- **在 LM Studio 中使用 Llava 进行图像分析**：一位用户询问如何将图像提供给 LM Studio Chat AI 中的 Llava 模型进行分析，回复指出必须将图像拖放到输入框中，模型才能“看到”它。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

</div>

<ul>
<li>
<a href="https://huggingface.co/roborovski/superprompt-v1">roborovski/superprompt-v1 · Hugging Face</a>: 未找到描述</li><li><a href="https://lmstudio.ai/jobs.html">正在重定向...</a>: 未找到描述</li><li><a href="https://huggingface.co/TheBloke/dolphin-2.7-mixtral-8x7b-GGUF">TheBloke/dolphin-2.7-mixtral-8x7b-GGUF · Hugging Face</a>: 未找到描述</li><li><a href="https://huggingface.co/blog/gptq-integration">使用 AutoGPTQ 和 transformers 让 LLM 更轻量化</a>: 未找到描述</li><li><a href="https://python.langchain.com/docs/expression_language/get_started">入门指南 | 🦜️🔗 Langchain</a>: LCEL 使得从基础组件构建复杂链变得简单，并且</li><li><a href="https://status.openai.com/">OpenAI 状态</a>: 未找到描述</li><li><a href="https://github.com/stitionai">stition</a>: stition 有 3 个可用的仓库。在 GitHub 上关注他们的代码。</li><li><a href="https://github.com/stitionai/devika">GitHub - stitionai/devika: Devika 是一个 Agent 化的 AI 软件工程师，能够理解高层级的人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标。Devika 旨在成为 Cognition AI 的 Devin 的竞争性开源替代方案。</a>: Devika 是一个 Agent 化的 AI 软件工程师，能够理解高层级的人类指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标...</li><li><a href="https://github.com/kalomaze/koboldcpp/releases/tag/v1.57-cuda12-oldyield">发布更快的 CPU Prompt 处理 (v1.57, CUDA 12) · kalomaze/koboldcpp</a>: 我还原了上游 llama.cpp 中导致线程让步（thread yielding）变为条件触发的更改，改为始终执行。这提高了我 CPU 上的 Prompt 处理性能，我的 CPU 具有...</li><li><a href="https://github.com/Nexesenex/kobold.cpp/releases/tag/v1.59d_b2254">发布 Kobold.CPP_Frankenstein_v1.59d_b2254_4x3bits_SOTA · Nexesenex/kobold.cpp</a>: Kobold.CPP Frankenstein v1.59 的源码和适用于 Windows 的 .exe 文件，使用 Openblas/Clblast/Vulkan 构建（小体积 .exe），以及包含 Cublas 的版本（大体积 .exe）：基于 LlamaCPP b2254 和 LostRuin 的 Kobol...</li><li><a href="https://github.com/caddyserver/caddy">GitHub - caddyserver/caddy: 快速且可扩展的多平台 HTTP/1-2-3 Web 服务器，支持自动 HTTPS</a>: 快速且可扩展的多平台 HTTP/1-2-3 Web 服务器，支持自动 HTTPS - caddyserver/caddy</li><li><a href="https://github.com/BBC-Esq/ChromaDB-Plugin-for-LM-Studio">GitHub - BBC-Esq/ChromaDB-Plugin-for-LM-Studio: 为在服务器模式下运行的 LM Studio 创建 ChromaDB 向量数据库的插件！</a>: 为在服务器模式下运行的 LM Studio 创建 ChromaDB 向量数据库的插件！ - BBC-Esq/ChromaDB-Plugin-for-LM-Studio</li><li><a href="https://github.com/czkoko/SD-AI-Prompt">GitHub - czkoko/SD-AI-Prompt: 一个基于 Llama 2 的快捷指令，用于扩展 Stable Diffusion 的 Prompt，由 llama.cpp 提供支持。</a>: 一个基于 Llama 2 的快捷指令，用于扩展 Stable Diffusion 的 Prompt，由 llama.cpp 提供支持。 - czkoko/SD-AI-Prompt</li><li><a href="https://github.com/kyegomez/BitNet">GitHub - kyegomez/BitNet: 在 PyTorch 中实现 "BitNet: Scaling 1-bit Transformers for Large Language Models"</a>: 在 PyTorch 中实现 "BitNet: Scaling 1-bit Transformers for Large Language Models" - kyegomez/BitNet
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/1220441138872455178)** (29 条消息🔥):

- **探索银行安全中 AI 的伦理边界**：成员们讨论了一个[名为 'Guardrails Arena' 的 Hugging Face Space](https://huggingface.co/spaces/lighthouzai/guardrails-arena)，用户可以在其中与模型交互以评估虚构的银行安全措施，结果显示某些模型会持续拒绝敏感信息，而其他模型则更为坦诚。
- **Guardrails 的底层机制**：对于对 'Guardrails Arena' 技术细节感兴趣的人，[Guardrails 模型 Python 脚本](https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/guardrails_models.py)和[提示词配置](https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/nemoguardrails_config/prompts.yml)提供了相关链接，深入展示了 AI 的决策策略。
- **即将推出的新推理模型**：引用了一篇讨论 'Quiet-STaR' 的新论文，这是一种通过在每个 token 处生成推理过程（rationales）来改进语言模型的泛化方法。该模型已转化为模型形式，可以在 [quietstar-8-ahead-GGUF 的 Hugging Face 仓库](https://huggingface.co/dagbs/quietstar-8-ahead-GGUF)和 [YouTube 视频](https://www.youtube.com/watch?v=9gdiqTJNeEc)中查看。
- **AI 驱动架构中的人工监督**：一次对话强调，虽然 AI 可能会设计出结构合理的建筑，但由于结构性错误的固有风险，法律和伦理上的人工监督是必要的。
- **互补的 AI 工作流，而非替代**：成员们认为 AI 应该被用作工作流加速器，辅助设计和测试等任务，而不是完全替代，并强调了持续的人工参与和验证的必要性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/spaces/lighthouzai/guardrails-arena">Guardrails Arena - lighthouzai 的 Hugging Face Space</a>：未找到描述</li><li><a href="https://tenor.com/view/dont-say-that-ever-again-diane-lockhart-the-good-fight-dont-say-that-never-say-that-again-gif-18052604895623551134">Dont Say That Ever Again Diane Lockhart GIF - Dont Say That Ever Again Diane Lockhart The Good Fight - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://huggingface.co/spaces/lighthouzai/guardrails-arena/blob/main/nemoguardrails_config/prompts.yml">nemoguardrails_config/prompts.yml · lighthouzai/guardrails-arena at main</a>：未找到描述</li><li><a href="https://huggingface.co/papers/2403.09629">论文页面 - Quiet-STaR: Language Models Can Teach Themselves to Think Before
  Speaking</a>：未找到描述</li><li><a href="https://huggingface.co/dagbs/quietstar-8-ahead-GGUF">dagbs/quietstar-8-ahead-GGUF · Hugging Face</a>：未找到描述
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/1220360925031305289)** (26 条消息🔥): 

- **LM Studio 中的符号链接问题**：**LM Studio 0.2.17 版本**的更新导致模型停止加载，在旧版本中可用的符号链接（symlinks）不再被识别。尽管尝试重新生成符号链接，用户仍遇到 *“Model with key Mistral/Hermes/Hermes-2-Pro-Mistral-7B.Q4_0.gguf not found.”* 错误。

- **语言提醒**：在一名用户发布中文消息后，Discord 成员被提醒 **English 是该服务器的主要语言**。

- **频道混淆**：有讨论建议需要更清晰的指引来告知在何处发布特定主题，因为 **反馈或帮助相关的问题** 经常被发布在错误的频道中。

- **文件交互功能请求**：一位用户表达了希望与 PDF、DOCX 或 PNG 等**文件进行对话**的愿望，并被告知使用 **Llava 模型** 支持与 PNG 图像对话。

- **总结多个 PDF 文档**：针对关于 **总结多个 PDF** 的咨询，一名成员被引导至特定频道以获取模型建议。

- **下载限速功能请求**：用户请求 **下载限速功能**，以防止大型模型下载占用全部带宽。随后讨论了 **OS 级别设置** 是否是带宽管理的更好解决方案。
  

---


**LM Studio ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/1220353404170272818)** (98 条消息🔥🔥): 

- **云端与本地 AI 硬件对比**：一位成员表达了在机器学习方面对本地硬件而非云服务的偏好，理由是成本效益和学习机会。在个人硬件上实验 AI 可以让公司在无需支付高昂云服务费用的情况下了解 AI。

- **IT 范式的转变**：成员们讨论了中心化和去中心化计算的周期性本质，预测根据趋势，在转向强大的去中心化 AI PC 之前，本地部署（on-premise）的 AI 服务器可能会变得更受青睐。

- **AI 聊天机器人的安全担忧**：随后讨论了一个安全漏洞，该漏洞允许通过侧信道攻击（side channel attack）拦截加密的 AI 聊天机器人 tokens。尽管进行了加密，攻击者仍可以推断出发送给用户的信息，这表明 OpenAI 等服务存在潜在漏洞（[此处有更详细的解释](https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/)）。

- **追求高效的 AI 开发**：对话转向了 AI 开发基础设施日益增长的复杂性和成本。高端硬件（如 GPUs、infiniband 交换机）以及对巨大电力的需求成为了经济和环境方面的担忧，对未来的预测倾向于 SaaS 解决方案。

- **为个人电脑选择合适的模型和规格**：一位成员就其配备 18GB RAM 的 M3 Pro 应该运行哪些 AI 模型寻求建议。建议寻找支持 "Full GPU Offload Possible" 的模型，并预料到由于硬件限制，在处理更高容量的模型时会存在局限性。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://evalplus.github.io/leaderboard.html">EvalPlus Leaderboard</a>：未找到描述</li><li><a href="https://ca.news.yahoo.com/hackers-spy-chats-almost-ai-133041191.html?guccounter=1&guce_referrer=aHR0cHM6Ly93d3cuZ29vZ2xlLmNvbS8&guce_referrer_sig=AQAAAA628vNFzyRtneQvrem1WDoac-lQ3-TX-faSAkTYTAnJC9GbR4hMplcovcWJLKYfRKzeKXGjwz5w4hkM4dBJp6XSEIgDvGir_0i8m4DEkXe2UOjpb_xrivCKUh4jSLjxoTviS1daIJ0mbC9fuYbZ8_kMXo_rApntCtJnL5pQsLa1">专家发现，黑客几乎可以监视任何 AI 的聊天内容</a>：小心你对 AI 聊天机器人说的话，因为显然黑客很容易就能破解。“目前，任何人都可以读取从 ChatGPT 和其他服务发送的私密聊天内容，”Yisroel ...</li><li><a href="https://arstechnica.com/security/2024/03/hackers-can-read-private-ai-assistant-chats-even-though-theyre-encrypted/">尽管已加密，黑客仍能读取私密的 AI 助手聊天内容</a>：所有非 Google 的聊天 GPTs 都会受到侧信道攻击的影响，从而泄露发送给用户的响应。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[🧪-beta-releases-chat](https://discord.com/channels/1110598183144399058/1166577236325965844/1220280552947712011)** (10 条消息🔥): 

- **寻求多模态使用的澄清**：一位用户表示，他们过于专注于让多模态 LM Studio 运行起来，以至于忘记了学习使用它的初衷。
- **模型推荐交流**：用户讨论了他们使用不同版本 Command-R 模型的经验。有人推荐了 second-state 的 q8 模型。
- **寻找具有外部存储能力的模型**：一位用户对多模型设置表现出兴趣，该设置允许模型与外部存储（如文本文件或本地 redis 实例）交互，以提高模型在涉及 Golang 和 Hugo 的复杂任务上的性能。
- **LM Studio 中的技术问题和故障排除**：一位用户分享了他们针对某个未指定模型的配置设置，该设置导致了异常行为，另一位用户建议重启 LM Studio 作为解决他们遇到的类似问题的可能方案。
- **硬件兼容性查询已解决**：一位用户在尝试于带有 RX 570 的 AMD ROCm 上运行 LM Studio 时遇到错误。另一位用户澄清说，RX 570 显卡太旧，无法与 ROCm 构建版本配合使用。
  

---


**LM Studio ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/1220273700452958260)** (10 条消息🔥): 

- **关于 LM Studio 新功能的澄清**：LM Studio 推出了支持 **multi-model**（多模型）使用的功能，允许同时将多个模型加载到 VRAM 中，以便通过 LMS 控制台或 autogen 等工具比较并利用最适合特定任务的模型。

- **Autogen 问题求助**：一位用户在他们的 Autogen 脚本中遇到了 **TypeError**，提示 `Completions.create()` 得到了一个意外的关键字参数 'request_timeout'。他们发布了错误回溯并寻求解决问题的帮助。

- **代码审查和敏感数据警示**：另一位用户建议从用户的 `config_list` 文件中删除 API key，以防止被机器人抓取，尽管该用户被告知这不是真实的 key。分享此建议是为了提醒大家养成良好的安全习惯，不要在公共论坛发布敏感数据。

- **寻求关于 VRAM 限制的建议**：一位成员询问 8GB 的 VRAM 是否被视为过低，因为他们报告在超过限制前只能运行一个语言模型 (LM)。他们想知道是否有选项可以移除或增加此限制。
  

---


**LM Studio ▷ #[langchain](https://discord.com/channels/1110598183144399058/1167546793656062063/1220537958793351220)** (2 条消息): 

- **使用 Instructor 简化语言建模**：建议成员们查看 GitHub 上的 [Instructor 库](https://github.com/jxnl/instructor)，该库旨在为语言模型工作流提供结构化输出 (structured outputs)。该库被强调为可以为用户简化流程的工具。
- **特别微调版 OpenChat 的成功**：一位成员提到他们拥有一个表现良好的特别微调版 **OpenChat**，并已成功将其与 dolphin mistral 微调版集成。
- **简短私信**：一条简短的说明，表示已发送私信以跟进对话，推测是关于上述提到的某个主题或工具。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.co">GitHub: Let’s build from here</a>：GitHub 是超过 1 亿开发者共同塑造软件未来的地方。为开源社区做出贡献，管理您的 Git 仓库，像专业人士一样审查代码，跟踪错误和功能...</li><li><a href="https://github.com/jxnl/instructor">GitHub - jxnl/instructor: structured outputs for llms</a>：为 LLM 提供结构化输出。通过创建账户为 jxnl/instructor 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LM Studio ▷ #[amd-rocm-tech-preview](https://discord.com/channels/1110598183144399058/1195858490338594866/1220367505231446138)** (23 条消息🔥): 

- **需要调整 Eject 和上下文大小 (Context Size)**：一位成员报告在加载过程中需要点击 "Eject"，然后最小化上下文大小，以防止在使用 Command-R 等支持大上下文的模型版本时出现 *out-of-memory* (内存不足) 问题。

- **关于 ROCm 0.2.17 Beta v3 的反馈**：分享了新版 **ROCm 0.2.17 Beta v3** 的链接，其中包括一份 [变更日志](https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe)，提到了针对 **GPU offloading** 相关问题的潜在修复。

- **GPU Offloading 的混合体验**：成员们讨论了在 **ROCm** 上进行 GPU offloading 的各种体验，指出了诸如 100% CPU 占用率以及可能由 **ZLUDA** 在系统路径中占先导致的混乱等问题。

- **ZLUDA 对 ROCm 的干扰**：成员们注意到，安装 **ZLUDA** 并将其加入 PATH 可能会干扰 **ROCm** 的运行，这可能解释了高 **CPU 占用率** 的问题。

- **在 AMD 硬件上的稳定表现**：多位用户报告在各种 **AMD GPU** 上成功且稳定地使用了 [ROCm 0.2.17 Beta v3](https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe)，反馈从“运行良好”到观察到显著的 **GPU 活动**不等。

**提到的链接**：<a href="https://files.lmstudio.ai/windows/0.2.17-ROCm-Beta-v3/beta/LM-Studio-0.2.17-Setup-ROCm-Beta-v3.exe">未找到标题</a>：未找到描述

  

---



**Perplexity AI ▷ #[general](https://discord.com/channels/1047197230748151888/1047649527299055688/1220273067230629950)** (340 条消息🔥🔥): 

- **Discord 用户讨论 Perplexity 和 AI 差异**：用户参与了关于不同 AI 模型性能的讨论，其中 **Claude 3 Opus** 和 **Gemini** 之间的比较非常频繁。一些人表示 Gemini 听起来更像人类，而另一些人则表示更喜欢 Opus 或 Anthropic 的顶级模型。
  
- **技术更新和故障排除**：参与者讨论了各种应用的更新、文本输入框的挑战，并为在移动端与 PC 端的使用提供互助。一些人分享了与 **iOS 应用更新**和功能相关的挫败感，例如希望有更暗的午夜/黑色主题以获得更好的视觉舒适度。

- **对 Cloudflare 的批评**：一位用户对 Cloudflare 的 CAPTCHA 验证表示不满，特别是在为了隐私使用 VPN 时，并指出这甚至会影响未开启隐私设置的用户。

- **Perplexity 的网页搜索和图像生成查询**：关于 **Perplexity AI** 如何进行网页搜索和图像生成的查询占据了显著位置。用户澄清说，虽然在移动端可能无法访问某些功能（如 **Unlimited Claude3 Opus**），但可以按照特定指令生成图像。

- **个人 AI 模型的讨论与比较**：用户分享了对 **Inflection-2.5 和 Pi.AI** 等各种个人 AI 模型的看法，强调了它们在对话用途和语音模型方面的优势。鉴于人才流失，对这些平台未来的担忧也浮出水面。
<div class="linksMentioned">

_

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://`,">未找到标题</a>：未找到描述</li><li><a href="https://inflection.ai/inflection-2-5">Inflection-2.5：遇见世界上最好的个人 AI</a>：我们是一家 AI 工作室，致力于为每个人创造个人 AI。我们的第一个 AI 名为 Pi，代表个人智能（personal intelligence），是一个具有支持性和共情能力的对话式 AI。</li><li><a href="https://www.businessinsider.com/neuralink-first-human-trial-patient-quadriplegic-elon-musk-x-2024-3">Neuralink 揭晓首位人体试验患者，一位 29 岁的四肢瘫痪者表示脑机芯片“并不完美”但改变了他的生活</a>：Elon Musk 的 Neuralink 揭晓了首位人体试验患者，这位 29 岁的四肢瘫痪者表示，脑机芯片虽然“并不完美”，但已经改变了他的生活。</li><li><a href="https://www.tradershub.ninja/">Tradershub Ninja</a>：未找到描述</li><li><a href="https://tenor.com/view/the-batman-no-selfpromo-no-self-promotion-batman-gif-24317098">The Batman No Selfpromo GIF - The Batman No Selfpromo No Self Promotion - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/thanos-talking-meme-thanos-talking-meme-thanos-speech-gif-1800590086203910493">Thanos Talking GIF - Thanos Talking Meme - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://tenor.com/view/light-thme-light-dark-theme-gif-27389075">Light Thme Light GIF - Light Thme Light Dark Theme - Discover &amp; Share GIFs</a>：点击查看 GIF</li><li><a href="https://www.reddit.com/r/singularity/s/j7YBzKr3ql">Reddit - 深入探索一切</a>：未找到描述</li><li><a href="https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/">Perplexity 极有可能是一个 Google 搜索外壳</a>：一位用户在 Reddit 的 LocalLLaMA 板块发帖称，Perplexity 总结了来自 Google 搜索前 5-10 条结果的内容。
</li>
</ul>

</div>
  

---


**Perplexity AI ▷ #[sharing](https://discord.com/channels/1047197230748151888/1054944216876331118/1220283301126733895)** (19 条消息🔥): 

- **最大行星知识查询**：一则帖子分享了关于最大行星的 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/The-largest-planet-Tmh863TjTp66BfMzKVBQEw#0)，表明可能针对该话题进行了研究或讨论。
- **推动日本 LLM 发展**：一位用户指出了 [日语语言模型开发](https://www.perplexity.ai/search/japanese-llm-development-5GPNniNXRXy2UqnfB8sjuA)，暗示了对该领域的关注或兴趣。
- **关于时间的问题**：一名成员分享了一个涉及法语短语 "Combien de temps" 的 [Perplexity AI 搜索链接](https://www.perplexity.ai/search/Combien-de-temps-kewLnab9THOciQpv4Y0Ttg)，可能代表一个与语言相关的查询。
- **GPT-5 发布传闻**：通过分享的 [Perplexity AI 链接](https://www.perplexity.ai/search/GPT5-release-rumors-NfNcO6yfRG..vG9acBEIYQ) 表达了对 GPT-5 发布的好奇或传闻关注。
- **资深 macOS 用户拥抱 Linux**：一位用户讲述了他们为了 AI/ML 学习从 macOS 切换到 Linux 的经历，并利用 [Perplexity AI](https://www.perplexity.ai/search/What-are-some-AmEb7c4LTn2GNCeiBSJmYw) 辅助学习过程，对 MX Linux 表示满意。
  

---


**Perplexity AI ▷ #[pplx-api](https://discord.com/channels/1047197230748151888/1161802929053909012/1220383326347722834)** (26 条消息🔥): 

- **Perplexity API 与 UI 引用的对比**：一名成员询问为什么 API 不像 Perplexity UI 那样提供来源和引用，暗示这可能是一个潜在的新功能。
- **Token 限制失误**：一位用户在尝试发送包含 **6621 tokens** 提示词且预期输出为 **9900 tokens** 的请求时遇到了 `BadRequestError`，这超出了 Perplexity **16384 tokens 的限制**。他们对如何相应地调整 API 调用感到困惑。
- **简历分析器挑战**：遇到 Token 限制的成员正在构建一个简历分析器/生成器作为 AI 实践项目，表明他们是该领域的新手。
- **Token 计数技巧**：另一位社区成员引导该用户查看 [Perplexity 文档](https://docs.perplexity.ai/reference/post_chat_completions)，以便准确检查其 AI 查询的 Token 数量。
- **寻求使用说明**：用户探讨了如何限制用户提示词，并得知这完全取决于内容长度，这对他们正在进行的简历项目是很有帮助的建议。

**提到的链接**：<a href="https://docs.perplexity.ai/reference/post_chat_completions">Chat Completions</a>：未找到描述

  

---



**LAION ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/1220288007031095397)** (369 条消息🔥🔥):

- **数据集困境 - 寻找 "mypersonality" 数据集**：成员们正在讨论来自 Facebook 的 **"mypersonality"** 数据集及其在基于文本预测作者性格方面的用途。该数据集的可访问性受到质疑，一名成员表示由于研究需要而了解该数据集。

- **Hugging Face 实现受到审视**：关于 [Hugging Face diffusers 库](https://github.com/huggingface/diffusers/issues/7365)中 embedding 实现潜在问题的广泛讨论展开，成员们分享了代码片段和修正方案。大家对文本 embedding 的正确池化方法（pooling method）表示关注，并建议修正现有代码以提升模型性能。

- **数据集恢复与替代方案探索**：**LAION-5B** 数据集在移除后的状态和未来是讨论的话题，鉴于欧盟立法，Datacomp 和 DFN 等新数据集被推崇为替代方案。成员们对 **LAION** 全面清除法律障碍并重新发布其数据集的能力表示怀疑，暗示这些数据集可能保持未发布状态。

- **推动 OpenAI 代码透明化**：成员们讨论了开源训练代码对 AI 进步的重要性，并对未来模型（如 **SD3**）可能的开放性表示期待，尽管之前的版本有所挫折。

- **训练技术讨论与改进**：关于在 SD2.x 等模型中**微调 text encoder** 的争论达成共识，即可能不需要对 text encoder 进行剧烈修改。曾经备受争议的关键微调方法现在被公认为一种“高级”方法，并已整合到 **Diffusers** 的官方训练脚本中。

- **对 AI 破坏指控的怀疑**：一位成员对那些对包含敏感材料（如 CSAM）的数据集表示担忧的研究人员是否真心诚意表示怀疑，推测他们的行为可能是为了阻碍 AI 的进步。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lifehacker.com/tech/its-not-safe-to-click-links-on-x">在 X 上点击链接并不安全</a>：当有人在 X 上发布链接时，该网站会生成链接预览。但据报道，这个系统可以被欺骗，恶意行为者可以通过虚假的链接预览将你重定向到恶意网站……</li><li><a href="https://www.thewrap.com/openai-to-meet-with-hollywood-studios-and-talent-agencies-next-week-on-sora-integration/">OpenAI 将于下周与好莱坞制片厂和人才机构会面商讨 Sora 整合</a>：OpenAI 下周将与好莱坞制片厂和人才机构举行会议，向电影制作人推介 Sora 的整合。</li><li><a href="https://huggingface.co/blog/xingxm/svgdreamer">SVGDreamer：基于 Diffusion Model 的文本引导矢量图形生成</a>：未找到描述</li><li><a href="https://tenor.com/bcuEi.gif">Annoyed Fuck GIF - Annoyed Fuck Frustrated - 发现并分享 GIF</a>：点击查看 GIF</li><li><a href="https://github.com/huggingface/transformers/blob/main/src/transformers/models/clip/modeling_clip.py#L99>">transformers/src/transformers/models/clip/modeling_clip.py at main · huggingface/transformers</a>：🤗 Transformers：适用于 Pytorch、TensorFlow 和 JAX 的先进机器学习库。- huggingface/transformers</li><li><a href="https://github.com/huggingface/diffusers/issues/7365">提供的 pooled_prompt_embeds 被 prompt_embeds[0] 覆盖 · Issue #7365 · huggingface/diffusers</a>：diffusers/src/diffusers/pipelines/stable_diffusion_xl/pipeline_stable_diffusion_xl.py 第 386 行 25caf24 pooled_prompt_embeds = prompt_embeds[0] 简单修复：pooled_prompt_embeds = prompt_embeds[0]...</li><li><a href="https://github.com/openai/CLIP/blob/main/clip/model.py#L364>">CLIP/clip/model.py at main · openai/CLIP</a>：CLIP (Contrastive Language-Image Pretraining)，根据图像预测最相关的文本片段 - openai/CLIP</li><li><a href="https://huggingface.co/docs/transformers/en/main_classes/output#transformers.modeling_outputs.BaseModelOutputWithPooling>">模型输出</a>：未找到描述
</li>
</ul>

</div>
  

---


**LAION ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/1220349817809670234)** (4 条消息): 

- **扩展图像尺度**：[arXiv](https://arxiv.org/pdf/2403.13043.pdf) 上讨论的一篇论文建议通过使用图像的**多个尺度**来提高模型性能。

- **使用时间编码图像**：另一篇来自 [arXiv](https://arxiv.org/pdf/2403.13802.pdf) 的论文介绍了一种使用 **6 倍时间戳数量**编码图像的方法，采用了不同的 zig-zags，一些人认为这可能是一种权宜之计，而非优雅的解决方案。

- **分形焦点**：讨论中幽默地提到了连续分形空间填充曲线，暗示了它们在解决当前编码方法方面的潜力。

- **展望未来**：分享了来自 [Bindu Reddy](https://twitter.com/bindureddy/status/1770934470373421522?t=5dLWWD7d9PN0C4ZjHAY_Tw&s=19) 的一条推文，作为前瞻性发展的指标，尽管推文的具体内容在消息中未披露。
  

---



**Nous Research AI ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/1220329148577353728)** (21 messages🔥): 

- **辩论 "Extended Mind" 概念**：成员们讨论了 "**Extended Mind**" 的概念，引用了 [Phoebe Klett 的一条推文](https://twitter.com/KlettPhoebe/status/1770480361656533449)。有人表示有兴趣将其移植到 **Mistral** 以提高易用性。
- **理解 Extended Mind 的深度**：一位成员提到 Extended Mind 似乎类似于一种**联想记忆 (associative memory)**，即通过一个独立的数据库保存信息，注意力可以像调用记忆和工具一样调用这些信息。
- **澄清 Extended Mind 的机制**：讨论澄清了 **Extended Mind** 涉及在 Forward pass 期间存储向量并获取 Top k，强调它更多是关于选择联想记忆的各个方面，而不是工具。
- **推测通过 Extended Mind 集成工具**：有关于未来对 **Extended Mind** 进行实验的讨论，推测将更深入地集成不同的工具，并探索其在影响记忆和推理方面的潜力。
- **识别 Extended Mind 的潜力与挑战**：成员们讨论了在 Extended Mind 方法中需要**加权注意力 (weighed attention)**，系统必须学习何时关注记忆。该概念与**记忆与推理 (memorizing versus reasoning)** 之间的关系也被提及为一个关注点。
  

---


**Nous Research AI ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/1220271708523728926)** (5 messages): 

- **问候与致意**：热烈欢迎新成员加入社区。
- **硬件难以跟上 AI 软件的发展**：一位成员评论了在不进行量化的情况下在本地运行 **oversized param models** 的挑战，并推测硬件最终会赶上软件的进步。
- **分享 LLaVA 模型微调教程**：分享了一个 [YouTube 视频](https://www.youtube.com/watch?v=21Tc92g15pM) 链接，该视频提供了如何微调 **LLaVA 模型** 的说明，涵盖了多模态学习和深度学习等各种主题。
- **介绍新型开源 AI 设备**：一位成员兴奋地分享了关于一款新型开源 AI 设备的 [Twitter 帖子](https://twitter.com/OpenInterpreter/status/1770821439458840846)，表达了对其由 Nous Research 模型驱动的期待。
- **对开源 AI 设备项目的赞赏**：对 Killian 及其团队在上述推文中提到的开源 AI 设备项目上取得的进展表示认可和赞赏。

**提到的链接**：<a href="https://www.youtube.com/watch?v=21Tc92g15pM">Finetune MultiModal LLaVA</a>：此视频解释了如何微调 LLaVA 模型 #llm #ml #ai #deeplearning #largelanguagemodels #python https://wandb.ai/byyoung3/ml-news/reports/How-to-Fine...

  

---


**Nous Research AI ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/1220279110694076436)** (23 messages🔥): 

- **揭开合成数据集的秘密**：Hugging Face 博客概述了生成一个庞大的合成数据集 **Cosmopedia**，以镜像 [Phi-1.5](https://arxiv.org/abs/2309.05463)。该文章强调了从昂贵的人工标注数据向合成数据集的转变，[Cosmopedia 是这一趋势的证明](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia)。

- **Prompt Engineering 的细节决定成败**：生成 **Cosmopedia** 并不依赖高性能 GPU，而是依赖详细的 **Prompt Engineering**。博客文章透露，在 Prompt 编写上投入的时间是该任务的重要组成部分。

- **Quiet-STaR 宣称可增强文本理解**：[Quiet-STaR](https://arxiv.org/abs/2403.09629) 是 STaR 的扩展，被提议作为一种技术，使语言模型学习为每个 Token 生成解释，从而改进其文本预测。论文摘要指出，LLM 有潜力在任意文本中推断未说明的推理依据 (rationales)。

- **OpenInterpreter 对 AI 设备的愿景**：通过一条 [推文](https://x.com/OpenInterpreter/status/1770821439458840846?s=20) 介绍了一款名为 **01 Light** 的新设备，它承诺作为一个便携式语音交互界面来控制电脑及其应用程序。创作者强调了其开源特性，并表示用户可以自行构建，或利用即将推出的 App 进行远程控制。

- **关于 OpenInterpreter 01 硬件必要性的辩论**：对话围绕 01 Light 是否仅仅是一个“高级麦克风”展开，一些成员指出硬件是可选的，用户可以 [在电脑上免费使用该系统](https://openinterpreter.com/01)。尽管最初存在质疑，但该开源项目及其软件的价值得到了认可。

- **在 Kubernetes 上运行 Nous 模型？**：一位用户询问了在 Kubernetes 中集成 Nous 模型的问题，并希望有一个类似于 [SideroLabs 推文](https://twitter.com/SideroLabs/status/1771207304748167445) 中提供的简单安装过程。目前没有关于 Nous 模型或其 Kubernetes 兼容性的进一步信息。

<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/blog/cosmopedia">Cosmopedia：如何为预训练 Large Language Models 创建大规模合成数据</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.09629">Quiet-STaR：语言模型可以在说话前教自己思考</a>：在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的工作通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理其实是……</li><li><a href="https://x.com/OpenInterpreter/status/1770821439458840846?s=20">来自 Open Interpreter (@OpenInterpreter) 的推文</a>：介绍 01 开发者预览版。立即订购或自行构建：http://openinterpreter.com/01 。01 Light 是一个可以控制家用电脑的便携式语音界面。它可以查看你的屏幕……
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/1220287620970446848)** (126 条消息🔥🔥): 

- **嵌入模型微调难题**：一位用户在微调 Embedding 模型时遇到了问题，提到使用 *Sentence Transformers* 的 *BatchAllTripletLoss* 时评估分数没有变化，表明模型没有在学习。此外，使用 Angle loss 会导致正样本和负样本都远离查询（query）。

- **寻求摘要生成研究**：一位用户正在寻找研究论文来测试新的摘要生成器（summarizer），请求他人提供文档。这引发了关于 Quiet-STaR（STaR 的泛化版本）论文的讨论，在该论文中，LLM 在每个 token 处生成推理过程（rationales）以解释未来的文本。

- **关于聊天机器人的讨论**：成员们讨论了将现有仓库 llm_steer 与通过“激活向量（activation vectors）”与 LLM 交互的界面进行集成。此外，还就 LLM 中逻辑推理和规划的有效性进行了辩论，特别是在采用各种方法（如 cursed model merging）时。

- **开源 RAG 平台和基准测试**：用户分享了他们的项目链接，例如一个用于 RAG 应用的开源平台，并讨论了 Mistral-Evolved-11b-v0.1 等模型的基准测试，评论了它们的性能提升。

- **探索 AI 与硬件**：一些成员质疑像 Open Interpreter 的 01 Lite 这样与 AI 相关的硬件发布的实用性，而另一些人则暗示了“意念转文字（mind to text）”技术的潜力，该技术可以通过颈部的 EMG 传感器解释内部语音。一些用户设想了未来通过手势或直接脑机接口与 AI 交互的可能性。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09629">Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking</a>: 在写作和交谈时，人们有时会停下来思考。虽然以推理为中心的研究通常将推理框架化为回答问题或完成 Agent 任务的方法，但推理其实是...</li><li><a href="https://tenor.com/view/spongebob-why-why-why-why-why-why-why-why-why-why-why-why-why-gif-25252239">Spongebob Why Why Why Why Why Why Why GIF - Spongebob Why Why Why Why Why Why Why Why - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/system_prompt.md">Abstractions/abstractions/goap/system_prompt.md at main · furlat/Abstractions</a>: 一个用于抽象 IRL 的 Pydantic 模型集合。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://hamel.dev/blog/posts/prompt/">- Fuck You, Show Me The Prompt.</a>: 通过拦截 API 调用，快速理解难以捉摸的 LLM 框架。</li><li><a href="https://github.com/furlat/Abstractions/blob/main/abstractions/goap/spatial.py#L267">Abstractions/abstractions/goap/spatial.py at main · furlat/Abstractions</a>: 一个用于抽象 IRL 的 Pydantic 模型集合。通过在 GitHub 上创建账号来为 furlat/Abstractions 的开发做出贡献。</li><li><a href="https://youtu.be/Q_p82HtBqoc">Open Interpreter&#39;s 01 Lite - WORLD&#39;S FIRST Fully Open-Source Personal AI AGENT Device</a>: Open Interpreter 推出的 01 Lite 是全球首款 100% 开源的个人 AI Agent 设备，可以控制你的电脑。让我们来评测一下，我将向你展示如何安装 open...</li><li><a href="https://gist.github.com/fullstackwebdev/4f8fc4931bd4dfba4231c8caf578e15e">graph_workflow.py</a>: GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/embed4all2graph_01.py">scratchTHOUGHTS/embed4all2graph_01.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 第二大脑草稿记忆，用于避免 self 的溢出错误。 - EveryOneIsGross/scratchTHOUGHTS</li><li><a href="https://github.com/EveryOneIsGross/scratchTHOUGHTS/blob/main/w2v2graph_01.py">scratchTHOUGHTS/w2v2graph_01.py at main · EveryOneIsGross/scratchTHOUGHTS</a>: 第二大脑草稿记忆，用于避免 self 的溢出错误。 - EveryOneIsGross/scratchTHOUGHTS</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steering-gpt-2-xl-by-adding-an-activation-vector">Steering GPT-2-XL by adding an activation vector — LessWrong</a>: 给模型的 Prompt [1] I hate you because GPT-2 I hate you because you are the most disgusting thing I have ever seen. GPT-2 + "Love" vector I hate…</li><li><a href="https://arxiv.org/abs/2310.01405">Representation Engineering: A Top-Down Approach to AI Transparency</a>: 在本文中，我们确定并描述了表征工程（RepE）这一新兴领域，这是一种借鉴认知神经科学见解来增强 AI 系统透明度的方法...</li><li><a href="https://www.lesswrong.com/posts/5spBue2z2tw4JuDCx/steer">Steering GPT-2-XL by adding an activation vector — LessWrong</a>: 给模型的 Prompt [1] I hate you because GPT-2 I hate you because you are the most disgusting thing I have ever seen. GPT-2 + "Love" vector I hate…
</li>
</ul>

</div>
  

---


**Nous Research AI ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/1220271417439027243)** (21 messages🔥): 

- **寻找 Quantization 支持**：一名成员提到了某个语言模型的 **Quantization**（量化），并建议另一名成员可能对此有见解。第二名成员承认尝试过 Quantization，并指出需要更多研究才能使其奏效。
- **Quantization 方面的协作**：应要求，一名成员分享了一个名为 **AutoAWQ** 的 4-bit Quantization 仓库。然而，他们说明这是一个过时的版本，并邀请其他人尝试修复它：[GitHub - casper-hansen/AutoAWQ](https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena)。
- **对 NousForge 的期待**：在有人询问其发布情况并提到是通过 Google 发现该聊天频道后，成员们表示 **NousForge** 尚未发布。
- **讨论指令 SFT 中的 Few-Shot Prompt**：一名成员质疑在指令 SFT（监督微调）数据集中包含 Few-Shot Prompt 的普遍性和益处，另一名成员对其普遍性给出了否定回答，随后最初的成员发现了一个相关的讨论帖，但尚未有结果报告。
- **质疑 Causal Masking 的理论基础**：一名成员询问 Attention 机制中的 Causal Masking 是否有理论依据，或者仅仅是为了工程上的便利。另一位参与者强调了 Masking 对于模型学习 **Next Token Prediction** 的必要性。

**提到的链接**：<a href="https://github.com/casper-hansen/AutoAWQ/tree/striped_hyena">GitHub - casper-hansen/AutoAWQ at striped_hyena</a>：AutoAWQ 实现了用于 4-bit 量化的 AWQ 算法，在推理过程中可实现 2 倍加速。文档：- GitHub - casper-hansen/AutoAWQ at striped_hyena

---

**Nous Research AI ▷ #[project-obsidian](https://discord.com/channels/1053877538025386074/1156472202619781140/1220535122093408356)** (3 条消息): 

- **Obsidian 的潜在改进**：分享了一个 [Baifeng Shi 的 Twitter 帖子](https://twitter.com/baifeng_shi/status/1770643896437240052)链接，暗示它可以改进 **Obsidian**。
- **确认 Obsidian 的增强**：一名成员承认分享链接中的内容确实会增强 **Obsidian**。
- **探索实现方案**：一名成员表示打算尝试针对 **Obsidian** 改进所建议的实现方案。

---

**Nous Research AI ▷ #[rag-dataset](https://discord.com/channels/1053877538025386074/1218682416827207801/1220271385264394292)** (38 条消息🔥): 

- **LanceDB 受到开发者关注**：Gabriel_syme 表达了对 **LanceDB** 的热情，强调了其**速度**、**易用性**，以及通过 SQL 类查询等生成式接口执行**混合搜索查询 (hybrid search queries)** 的能力。相比之下，Iriden 讨论了使用 **Polars** 进行传统查询，尽管其在与语言模型配合使用时语法具有挑战性。
  
- **等待数据工具更好的集成**：提到了 **Polars** 正在等待更好的集成，并指出 LanceDB 和 Polars 可以交换数据，但开发者需要手动完成集成。此外，gabriel_syme 考虑了 Polars 潜在的云原生能力。

- **托管云解决方案的可能性**：Iriden 强调了一个正在开发的 **FastAPI/Streamlit** 应用，该应用允许上传 parquet 文件并运行 Polars 表达式，并提到一旦部署到 modal.com，它就可以作为托管云解决方案。

- **在 GitHub 上分享开发工作**：Iriden 分享了一个 [GitHub 仓库](https://github.com/Neural-Dragon-AI/Cynde)，其中包含在 **Polars frames** 上异步运行 GPT 的代码，以及使用 embeddings 的机器学习模型。该仓库旨在将语义智慧 (Semantic Wisdom) 与预测模型 (Predictive Models) 相结合。

- **开发者之间的育儿讨论**：进行了一段简短而轻松的关于为人父母的交流，triggerhappygandhi 表示祝贺并谈到 Discord 上父母身份的稀缺，denovich 则根据个人经验回应了成为父母后的生物学影响。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lancedb.github.io/lancedb/basic/">快速开始</a>：未找到描述</li><li><a href="https://github.com/Neural-Dragon-AI/Cynde">GitHub - Neural-Dragon-AI/Cynde: Integrating Semantic Wisdom with Predictive Models</a>：将语义智慧与预测模型集成 - Neural-Dragon-AI/Cynde
</li>
</ul>

</div>

---

**OpenAccess AI Collective (axolotl) ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/1220337261091094569)** (213 条消息🔥🔥): 

- **RAG 与 Agent 方法的对比**：一些成员辩论了**检索增强生成 (RAG)** 与基于 Agent 模型的有效性。有人认为 RAG 是一种能力较弱的方法，仅仅是弥补缺失知识的权宜之计 (band-aid)，而具有工具使用和反思 (reflection) 能力的基于 Agent 的模型则更加健壮，尽管 RAG 在实现上可能看起来更简单。

- **FastChat 格式冲突**：**FastChat 的 alpaca** 模型提示词格式存在差异，人们担心它与 Stanford 的 alpaca 格式不一致。一名成员强调了这种差异，并指出可能需要一个 pull request 来修正 FastChat 的格式以保持一致性。

- **Galore 优化器的收益**：关于新出的 **Galore 优化器** 的讨论非常积极，指出其设置顺畅，并能为大语言模型的微调 (fine-tuning) 节省大量 VRAM。它利用每个全模型参数梯度的低秩矩阵，允许以极低的内存占用进行全参数微调。

- **GPT-3.5 性能持续关注**：频道参与者对 **GPT-3.5** 表现出兴趣，询问了各种模型大小和配置的性能及推理时间。一位用户指出，由于处理患者数据的隐私限制，在 Mac 上本地运行时推理速度并不理想。

- **数据集讨论**：进行了一段关于数据集策划和格式化的简短交流。涵盖了不同类型的数据集及其相应的模型训练格式，特别是 *sharegpt* 和 *chatml* 格式之间的区别，并确认了 Axolotl 如何解释和处理这些数据集以供模型使用。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py#L550>">FastChat/fastchat/conversation.py at main · lm-sys/FastChat</a>: 一个用于训练、部署和评估大型语言模型（LLM）的开放平台。Vicuna 和 Chatbot Arena 的发布仓库。 - lm-sys/FastChat</li><li><a href="https://github.com/jiaweizzhao/GaLore/issues/6">Third-party benchmark · Issue #6 · jiaweizzhao/GaLore</a>: 你好，非常感谢这项出色的工作。我们使用 Llama-Factory 进行了一些实验，结果表明 GaLore 在全参数...过程中可以显著减少显存占用。
</li>
</ul>

</div>
  

---


**OpenAccess AI Collective (axolotl) ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/1220479137823592480)** (6 messages): 

- **GaLore 合并至 Transformers**：[GaLore 优化器](https://github.com/huggingface/transformers/pull/29588)已合并到 **Hugging Face Transformers** 库中，期待其集成的成员们对此感到兴奋。
- **需要技术协助**：一名成员报告了在运行 "examples/openllama-3b/qlora.yml" 示例时出现的 **TypeError**，与未预料到的关键字参数 'seq_len' 有关。另一名成员将该求助请求重定向到了特定频道以获得更好的支持。

**链接提及**: <a href="https://github.com/huggingface/transformers/pull/29588">FEAT / Optim: Add GaLore optimizer by younesbelkada · Pull Request #29588 · huggingface/transformers</a>: 这个 PR 做了什么？如标题所示，添加了来自 https://github.com/jiaweizzhao/GaLore 的 GaLore 优化器。修复了：#29512。这是我目前测试 API 的方式：import torch import datasets from ...

  

---


**OpenAccess AI Collective (axolotl) ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/1220390639548633179)** (10 messages🔥): 

- **文本分类之争**：一位成员对微调语言模型（LLM）进行文本分类的常见做法提出了质疑，即教模型将类别名称作为文本输出，而不是在顶部添加分类头。针对这种方法给出的一个可能原因是其灵活性，例如可以支持在 *chain of thoughts*（思维链）上进行训练。

- **调整模型参数**：有人询问如何调整 *GaLore* 中的所有参数，特别提到了 `-mlp` 和 `self_attn`。从消息中尚不清楚该成员是否解决了他们的问题。

- **训练 Mixtral 作为编程助手**：一位用户寻求关于训练和微调 Mixtral-7B 模型以使其成为具备 *runpod* 和 *python* 等工具文档知识的编程助手的指导。他们询问了在个人硬件上训练模型所需的工具、IDE 和概念。

- **PyTorch 与 Gema**：一位成员询问在 PyTorch 上是否仍不推荐（*a no-no*）使用 *gema*。

- **预处理错误排查**：一位成员报告了在使用 `axolotl` 预处理脚本预处理数据时出现的 `KeyError: 'instruction'` 错误，并分享了他们的配置文件和数据片段。消息记录中未提供解决方案。
  

---



**Latent Space ▷ #[ai-general-chat](https://discord.com/channels/822583790773862470/1075282825051385876/1220319698483281991)** (120 messages🔥🔥): 

- **自动模型合并突破**：一位成员分享了 [Hardmaru 关于基础模型合并的自动进化算法的新论文](https://arxiv.org/abs/2403.13187)。他们讨论了这是一种结合多样化开源模型的独特方式，无需大量训练即可提升模型性能。
  
- **巴黎 AI 社区对线下聚会的热烈讨论**：成员们就法国巴黎的 AI 社区展开了热烈讨论。一些人分享了近期聚会的经验，而另一些人则表示有兴趣参加未来的聚会，如 [Paris Retrieval Augmented Generation group](https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/) 会议，强调了该地区活跃的技术社区。

- **模型缩放详解**：一位成员询问 Hugging Face 上的模型（如 `cosmo-1b`）是如何从 `Llama` 等大型模型缩减规模的。另一位成员[解释道](https://github.com/HuggingFaceTB/cosmo-1b)，较小的模型并非通过微调得到，而是从头开始训练的独立架构，只是参数规模有所缩减。

- **视频理解 AI 工具聚焦**：关于视频分析工具的讨论引出了几项推荐，包括 [Video Mamba](https://huggingface.co/blog/vladbogo/video-mamba) 和 [Twelve Labs](https://www.twelvelabs.io/)，这些工具能够利用基础模型实现先进的视频理解。

- **对开源 AI 平台的兴趣日益增长**：一位成员指出了 JanAI 项目，这是 LM Studio 的一个开源替代方案，已在 Reddit 上引起关注。另一位成员在关于 AI 平台透明度的讨论出现后，澄清了有关 LM Studio 部分开源计划的细节。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.twitch.tv/georgehotz">georgehotz - Twitch</a>: georgehotz 在 Twitch 上直播！查看他们的视频，注册聊天并加入他们的社区。</li><li><a href="https://videodb.io">VideoDB</a>: 只需 2 行简单的代码，即可在各种类型的视频上构建智能应用。由开发者构建，为开发者服务。</li><li><a href="https://arxiv.org/abs/2403.13187">Evolutionary Optimization of Model Merging Recipes</a>: 我们提出了一种进化算法的新颖应用，用于自动创建强大的基础模型。虽然模型合并已成为 LLM 开发中一种极具前景的方法，因为其……</li><li><a href="https://helixml.substack.com/p/how-we-got-fine-tuning-mistral-7b">How we got fine-tuning Mistral-7B to not suck: Helix Project Report, Feb 2024</a>: 发布 Helix v0.5，改进了文本微调并支持 OpenAI API 🎉</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba: State Space Model for Efficient Video Understanding</a>: 未找到描述</li><li><a href="https://x.com/__tinygrad__/status/1770112124871979095">Tweet from the tiny corp (@__tinygrad__)</a>: 很少有人能成功制造出这些机器。复杂性主要体现在几个方面。1) PCI-E AER 错误。很难获得可靠的 PCI-E 扩展。我们不得不定制电缆……</li><li><a href="https://x.com/theinformation/status/1770183406640373901?s=61">Tweet from The Information (@theinformation)</a>: Perplexity AI 是一家 AI 驱动的搜索引擎，因挑战 Google 而成为硅谷初创公司的宠儿。它也在悄悄使用 Google 的数据。 https://www.theinformation.com/articles/a...</li><li><a href="https://x.com/maximelabonne/status/1767124527551549860?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Maxime Labonne (@maximelabonne)</a>: ♾️AutoMerger 我制作了一个简洁的小工具，可以在 @huggingface 上自动合并模型。它在周末已经创建了几个具有竞争力的模型。以下是它的工作原理。🧵 Space: https://h...</li><li><a href="https://x.com/__tinygrad__/status/1770510742007271545">Tweet from the tiny corp (@__tinygrad__)</a>: @luka_emon 当我开始时，我不明白 AMD 的问题出在哪里。我以为是驱动程序，其实不是。tinygrad 现在直接向 GPU 提交 AQL 队列。是……</li><li><a href="https://x.com/esotericcofe/status/1770842634229014949?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from Nucleus☕️ (@EsotericCofe)</a>: 结束了</li><li><a href="https://x.com/davidsholz/status/1770601982488912281?s=46&t=90xQ8sGy63D2OtiaoGJuww">Tweet from David (@DavidSHolz)</a>: 我很乐意资助 7B 级别的开源文本扩散模型（对混合扩散/AR 持开放态度）的研究和创建。有人有兴趣参与吗？接受资助、兼职或全职……</li><li><a href="https://www.twelvelabs.io/">Multimodal AI that understands videos like humans</a>: 为任何应用带来类人视频理解能力，无论你拥有 TB 级还是 PB 级的视频。</li><li><a href="https://jan.ai/">Jan | Rethink the Computer</a>: Jan 通过在你的计算机上本地运行 LLM，将你的电脑变成一台 AI 机器。这是一个注重隐私、本地优先、开源的解决方案。</li><li><a href="https://www.youtube.com/watch?v=cvOpX75Kz4M">Deep dive: model merging</a>: 模型合并是一种日益流行的技术，它可以在不需要任何额外……的情况下，为 Transformer 模型添加或移除功能。</li><li><a href="https://github.com/simonw/files-to-prompt">GitHub - simonw/files-to-prompt: Concatenate a directory full of files into a single prompt for use with LLMs</a>: 将包含文件的整个目录连接成一个单独的 Prompt，以便与 LLM 配合使用 - simonw/files-to-prompt</li><li><a href="https://github.com/stitionai/devika?tab=readme-ov-file">GitHub - stitionai/devika: Devika is an Agentic AI Software Engineer that can understand high-level human instructions, break them down into steps, research relevant information, and write code to achieve the given objective. Devika aims to be a competitive open-source alternative to Devin by Cognition AI.</a>: Devika 是一个 Agent 架构的 AI 软件工程师，能够理解人类的高层指令，将其分解为步骤，研究相关信息，并编写代码以实现给定目标。Devika 旨在成为 Cognition AI 的 Devin 的有力开源替代方案。</li><li><a href="https://www.meetup.com/fr-FR/paris-retrieval-augmented-generation-group/">Paris RAG User Group (Retrieval Augmented Generation) | Meetup</a>: 欢迎来到巴黎 RAG！我们是一个由对 RAG 及其相关技术感兴趣的专业人士和爱好者组成的社区，这些技术可以增强大语言模型和 AI！
</li>
</ul>

</div>
  

---


**Latent Space ▷ #[ai-announcements](https://discord.com/channels/822583790773862470/1075282504648511499/1220814514711167046)** (4 条消息):

- **Adept 洞察播客发布**：一集新的播客上线，其中包含一篇关于 *OpenAI*、*Google* 和 **Adept** 见解的文章。该公告附带了一个 [Twitter 链接](https://twitter.com/swyx/status/1771255525818397122)。
- **Adept 播客团队努力获认可**：在准备 **Adept 播客** 期间，对一位成员的协助表示了感谢，尽管由于时间限制未能涵盖所有问题。
- **AI In Action：Llama.cpp**：一场名为 *AI In Action* 的 AI 主题活动即将开始，展示 **Llama.cpp**，并提供了 [Discord 频道链接](https://discord.com/channels/822583790773862470/1200548371715342479) 以供实时参与。
  

---


**Latent Space ▷ #[llm-paper-club-west](https://discord.com/channels/822583790773862470/1197350122112168006/1220766447270105088)** (10 messages🔥): 

- **未授予发言权限**：成员们注意到他们在 Discord 频道中没有 **发言权限 (speaker rights)**。

- **Zoom 救场**：一位成员提到创建一个 **Zoom 会议室** 作为沟通的替代方案，因为 Discord 的发言权限不可用。

- **日程紧凑**：一位参与者表示在 **12:45 有硬性截止时间 (hard stop)**，表明讨论时间有限。
  

---


**Latent Space ▷ #[ai-in-action-club](https://discord.com/channels/822583790773862470/1200548371715342479/1220824233127706638)** (92 messages🔥🔥): 

- **发现 Slono 的身份**：真相大白，一位名为 "slono" 的用户可能实际上并不叫这个名字。尽管感到意外，群组中还是分享了 slono 音乐的 Spotify 链接，展示了一种旨在捕捉 *长夜将尽时难以捉摸的氛围* 的风格 ([听听 slono 的音乐](https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA))。

- **填充哲学 (The Padding Philosophy)**：成员们幽默地讨论了当张量维度“对不上”时的“填充并祈祷 (pad and pray)”方法，建议维度或许应该像 Python 中的类型一样，在 IDE 端进行更严格的管理或强制执行。

- **Llama.cpp 功能与 UI 挑战**：一位用户建议 llama.cpp 潜力巨大，可以利用 GPU 处理。此外，还有关于 Discord 移动端 UI 体验不佳的反馈，特别是使用时无法最小化摄像头的问题。

- **理解 Transformer 模型**：展开了一场关于将 Transformer 模型视为具有可调权重和图操作的加权张量的概念性讨论。这促使一位成员分享了来自 bbycroft.net 的关于这些模型工作原理的可视化链接 ([LLM 可视化](https://bbycroft.net/llm))。

- **关于代码库和音乐模型的新兴讨论**：简要触及了高效阅读大型代码库所涉及的学习曲线，并对未来关于音乐生成模型的讨论表示期待。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://bbycroft.net/llm">LLM Visualization</a>: 未找到描述</li><li><a href="https://arxiv.org/abs/2402.00789">Graph-Mamba: Towards Long-Range Graph Sequence Modeling with Selective State Spaces</a>: Attention 机制已被广泛用于捕捉 Graph Transformers 中节点间的长程依赖。受限于二次计算成本，Attention 机制无法扩展到...</li><li><a href="https://tenor.com/view/friends-bestfriends-yep-bff-gif-4566644">Did We Just Become Best Friends? GIF - Friends Bestfriends Yep - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://docs.google.com/spreadsheets/d/1q5rwO4wleMTLXr1z58c2UC03QsDsGwbJY1v4UG7eEOs/edit#gid=0">AI In Action: Weekly Jam Sessions</a>: 2024 主题, 日期, 协调人, 资源, @dropdown, @ GenAI 的 UI/UX 模式, 1/26/2024, nuvic, &lt;a href=&quot;https://maggieappleton.com/squish-structure&quot;&gt;https://maggieappleton.com/squish-stru...</li><li><a href="https://open.spotify.com/artist/1rWeYVkrGXRqaD8e0kwMbc?si=xu1E7Di8T_OUpQvT46f-BA">slono</a>: 艺术家 · 每月 107 名听众。
</li>
</ul>

</div>
  

---



**LlamaIndex ▷ #[blog](https://discord.com/channels/1059199217496772688/1187460979064324127/1220396913665507329)** (4 messages): 

- **使用 IFTTT 在学习中导航隐私**：LlamaIndex 博客讨论了在不冒私有数据泄露风险（特别是针对患者临床报告）的情况下，通过 few-shot 示例改进 LLM/RAG 应用的挑战。这一担忧通过一条[链接到博客文章的推文](https://t.co/5rTwPePqV6)得到了说明。

- **Navarasa 2.0 打破语言障碍**：LlamaIndex 博客的一次更新介绍了 Navarasa 2.0，这是由 @ravithejads 对 **Google Gemma 7B** 进行微调后的版本，旨在支持 15 种印度语言。这一进展强调了将通用 AI 模型本地化以更好地服务于区域语言使用者的重要性，如[这条推文](https://t.co/HHrfonnAr2)所述。

- **医疗数据中的 Differential Privacy**：LlamaIndex 的一篇新文章讨论了在 LLMs/RAG 系统中实现 Differential Privacy，以便安全地使用敏感数据（如医疗信息），目标是在不损害个人隐私的情况下加强研究。更多见解可以在 [相关推文](https://t.co/2ZipmvOwXv) 中找到。

- **Agent-Human Interaction 中的 UX**：LlamaIndex 博客介绍了一个新模板，通过让 Agent 仅在必要时请求人工输入来优化用户体验。这种方法旨在平衡自主性与干预，更多细节见 [分享的推文](https://t.co/Z16QPCWFmG)。
  

---


**LlamaIndex ▷ #[general](https://discord.com/channels/1059199217496772688/1059201661417037995/1220266398471684147)** (184 条消息🔥🔥): 

- **Bot Tool Integration 的困扰**：成员们讨论了在创建集成 Google Search 和 code interpreter 等不同工具的聊天机器人时遇到的困难。虽然参考了 [文档](https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/)，但成员们遇到了诸如 *"BadRequestError"* 之类的错误，建议包括将工具合并到单个列表中并进行故障排除。

- **API 和文档更新**：几位用户报告了访问 LlamaIndex 文档某些页面时的问题，可能是由于网站更新到了 MKDocs。成员们提供了 [新格式文档的链接](https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex) 作为临时解决方案。

- **Query Pipeline 的困惑**：[此处](https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/) 详细介绍的一个 Query Pipeline DAG 用例让用户对路径遍历的决策过程感到困惑。解释指出，DAG 中的每个链和链接都明确定义了输入和输出的路径及交互，确保收敛到单个输出。

- **Batch Evaluation 逻辑查询**：成员们请求协助理解 LlamaIndex 中应用的评估逻辑，并特别要求对代码流程进行注释以提高清晰度。提供的直接回答详细说明了每个代码片段的功能，以及响应评估背后的逻辑，以确定 LLM 输出是否符合预期结果。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>

<li><a href="https://api.getmerlin.in/#pricing">Merlin API Platform</a>：在几分钟内将 LLMs 集成到您的生产应用中。</li><li><a href="https://colab.research.google.com/drive/13NJEyhKWT7xdJFAJ6nB8mq-fk22UVDKa?usp=sharing">Google Colaboratory</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/loading/documents_and_nodes/usage_documents/">Using Documents - LlamaIndex</a>：未找到描述</li><li><a href="https://www.llamaindex.ai/blog/running-mixtral-8x7-locally-with-llamaindex-e6cebeabe0ab">Running Mixtral 8x7 locally with LlamaIndex and Ollama — LlamaIndex, Data Framework for LLM Applications</a>：LlamaIndex 是一个简单、灵活的数据框架，用于将自定义数据源连接到大语言模型 (LLMs)。</li><li><a href="https://docs.llamaindex.ai/">LlamaIndex - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/deploying/query_engine/streaming/">Streaming - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/">Build your own OpenAI Agent - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/usecases/10q_sub_question/">10Q Analysis - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/latest/examples/llm/localai/#llamaindex-interaction">LocalAI - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/module_guides/supporting_modules/service_context_migration/">Migrating from ServiceContext to Settings - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/UKPLab/sentence-transformers/releases/tag/v2.6.0">Release v2.6.0 - Embedding Quantization, GISTEmbedLoss · UKPLab/sentence-transformers</a>：此版本带来了 Embedding 量化：一种大幅加速检索和其他任务的方法，以及一个新的强大损失函数：GISTEmbedLoss。使用 pip install sentence-trans... 安装此版本</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/vector_stores/QdrantIndexDemo/">Qdrant Vector Store - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/f5263896121721de1051ce58338a1e0ea6950ca7/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py#L704">llama_index/llama-index-integrations/vector_stores/llama-index-vector-stores-qdrant/llama_index/vector_stores/qdrant/base.py at f5263896121721de1051ce58338a1e0ea6950ca7 · run-llama/llama_index</a>：LlamaIndex 是一个用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/">Index - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/blob/4394c7f11e907c4a7c9926ae98eb53e6d60a1619/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py#L66">llama_index/llama-index-integrations/embeddings/llama-index-embeddings-huggingface/llama_index/embeddings/huggingface/base.py at 4394c7f11e907c4a7c9926ae98eb53e6d60a1619 · run-llama/llama_index</a>：LlamaIndex 是一个用于您的 LLM 应用程序的数据框架 - run-llama/llama_index</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/pipeline/query_pipeline_sql/">Query Pipeline for Advanced Text-to-SQL - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/rags">GitHub - run-llama/rags: Build ChatGPT over your data, all with natural language</a>：完全使用自然语言在您的数据上构建 ChatGPT - run-llama/rags</li><li><a href="https://github.com/run-llama/llama_index/pull/12187">fix async streaming by logan-markewich · Pull Request #12187 · run-llama/llama_index</a>：需要确保延迟声明的 queue/async 内容在访问前已实际实例化。修复了 #12180</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/batch_eval/">BatchEvalRunner - Running Multiple Evaluations - LlamaIndex</a>：未找到描述</li><li><a href="https://github.com/run-llama/llama_index/issues/12180">[Bug]: AttributeError: &#39;NoneType&#39; object has no attribute &#39;wait&#39; · Issue #12180 · run-llama/llama_index</a>：Bug 描述：Async Streaming Chat 示例：https://docs.llamaindex.ai/en/stable/examples/agent/openai_agent/#async-streaming-chat 产生异常：AttributeError: &#39;NoneType&#39; object has n...</li><li><a href="https://github.com/run-llama/llama_index/issues/12143">[Question]: benchmark for the llama_index, but the latency is so weird. · Issue #12143 · run-llama/llama_index</a>：问题验证：我已在文档和 Discord 中搜索过答案。问题：你好，我想对 LlamaIndex 系统进行性能分析。我的代码片段如下。我的 GPU 是单块 A10，24G...</li><li><a href="https://developer.twitter.com/en/products/twitter-a">developer.twitter.com/en/products/twitter-a</a></li>

pi">来自 Twitter API 的推文 | 产品</a>：使用 Twitter API 分析、学习并与推文、私信和用户互动。扩展您的访问权限以实现增长、实验和创新。</li><li><a href="https://docs.llamaindex.ai/en/stable/api_reference/indices/vector/#llama_index.core.indices.VectorStoreIndex">Vector - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/semantic_similarity_eval/#embedding-similarity-evaluator">Embedding Similarity Evaluator - LlamaIndex</a>：未找到描述</li><li><a href="https://docs.llamaindex.ai/en/stable/examples/evaluation/faithfulness_eval/">Faithfulness Evaluator - LlamaIndex</a>：未找到描述
</li>
</ul>

</div>
  

---



**Eleuther ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/1220385053410983986)** (51 messages🔥): 

- **寻找紧凑的代码数据集**：用户寻求小型预训练数据集，并考虑了包含 200 万个注释/代码对的 [CodeSearchNet 语料库](https://huggingface.co/datasets/code_search_net)，但注意到与上下文长度相关的潜在问题。
- **The MiniPile - 多样化预训练的紧凑替代方案**：[The MiniPile](https://arxiv.org/abs/2304.08442) 被建议作为一个合适的、包含 100 万个文档的多样化文本语料库，用于在较小的数据集上预训练语言模型，且性能损失极小。
- **API 在 Logprobs 上有所保留？**：讨论强调了像 Claude 和 Gemini 这样的闭源模型不提供 logprobabilities 和 tokenizers，而 OpenAI 等平台通常会提供，这可能是出于专有原因。
- **优化模型的 GPU 性能**：一份[论文提供](https://arxiv.org/abs/2401.14489)了通过考虑超参数的影响和高效的模型形状来最大化 Transformer 模型运行时性能的指南，这可能带来高达 39% 的吞吐量提升。
- **科技圈的风云变幻**：对话涉及 MS 据报道向 Inflection 支付 6 亿美元以挖走员工，并提到了一个价值不菲的 H100 集群，同时对比了各科技巨头的公开演讲风格。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2401.14489">The Case for Co-Designing Model Architectures with Hardware</a>：虽然 GPU 负责训练绝大多数最先进的深度学习模型，但在设计新的深度学习 (DL) 模型时，其架构的影响往往被忽视...</li><li><a href="https://arxiv.org/abs/2304.08442">The MiniPile Challenge for Data-Efficient Language Models</a>：预训练文本语料库日益增长的多样性使语言模型具备了跨各种下游任务的泛化能力。然而，此类多样化的数据集往往过于庞大...</li><li><a href="https://github.com/allenai/OLMo/issues/518">Something weird with Instruct Model · Issue #518 · allenai/OLMo</a>：🐛 描述 Bug。这是我正在运行的代码。目标是获取 Chat 模型生成的每个 Token 的 logprob。olmo = AutoModelForCausalLM.from_pretrained(&quot;allenai/OLMo-7B-Instruct&quo...</li><li><a href="https://huggingface.co/datasets/code_search_net?row=42">code_search_net · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://arstechnica.com/information-technology/2023/09/ai-language-models-can-exceed-png">September | 2023 | Ars Technica</a>：未找到描述</li><li><a href="https://arstechnica.com/information-technology/2023/09/ai-language-models-can-exceed-png-and-flac-in-lossless-compression-says-study/>">September | 2023 | Ars Technica</a>：未找到描述
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/1220324145301033130)** (59 messages🔥🔥): 

- **抗体设计中的 AI 创新**：一名成员分享了一篇 [Ars Technica 文章](https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them)，讨论了 AI 在创建治疗性抗体方面的进展，表达了对 Diffusion 模型在该领域潜力的兴奋。然而，另一位成员对该研究领域的实际经济应用案例表示怀疑。

- **DenseFormer 揭示激活模式**：[DenseFormer](https://arxiv.org/abs/2402.02622) 架构提出了一种简单而有效的方法，即使用深度加权平均 (Depth-Weighted-Average, DWA) 来改进大规模模型，而无需显著增加参数，引发了关于机器学习中经常被忽视的简单想法的讨论。

- **探索强化学习与 Transformer 敏感性**：[发表的论文](https://proceedings.mlr.press/v139/davis21a.html) 介绍了 *Catformer* 架构，旨在通过级联层减少敏感性来解决训练 Transformer 模型中的挑战，这种方法可以提高训练的稳定性。

- **深度注意力方法讨论**：社区成员就 Transformer 架构的历史渊源和最新创新（如 [OmniNet](https://arxiv.org/abs/2103.01075)）展开了讨论，强调了实现具有全感受野的广泛注意力机制的潜力和挑战。

- **架构变更中的新颖性与功能性**：在关于修改神经网络架构（如受 DenseNet 启发的 Transformer）的讨论中，参与者权衡了新颖性的价值与让模型修改在大规模下有效工作的实际收益。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://proceedings.mlr.press/v139/davis21a.html">Catformer: Designing Stable Transformers via Sensitivity Analysis</a>：Transformer 架构被广泛使用，但训练它们并非易事，需要自定义学习率调度、缩放项、残差连接以及对子模块（如 n...）的精心布置。</li><li><a href="https://arxiv.org/abs/2402.02622">DenseFormer: Enhancing Information Flow in Transformers via Depth Weighted Averaging</a>：Vaswani 等人 (2017) 提出的 Transformer 架构现在在从自然语言处理到语音处理和图像理解的各个应用领域都无处不在。我们提出了 DenseForme...</li><li><a href="https://arxiv.org/abs/2103.01075">OmniNet: Omnidirectional Representations from Transformers</a>：本文提出了来自 Transformer 的全向表示 (OmniNet)。在 OmniNet 中，不再维持严格的水平感受野，而是允许每个 token 关注所有 token...</li><li><a href="https://arxiv.org/abs/2312.01552">The Unlocking Spell on Base LLMs: Rethinking Alignment via In-Context Learning</a>：大型语言模型 (LLM) 的对齐微调过程通常涉及通过监督微调 (SFT) 进行指令学习，以及通过人类反馈强化学习 (RLHF) 进行偏好微调...</li><li><a href="https://www.technologyreview.com/2008/03/10/221426/enzymes-built-from-scratch/">Enzymes Built from Scratch</a>：研究人员使用一种新的计算技术设计了前所未见的催化剂。</li><li><a href="https://arstechnica.com/science/2024/03/antibodies-against-anything-ai-tool-adapted-to-make-them">Antibodies against anything? AI tool adapted to make them</a>：现在，制造抗体意味着免疫动物。但这可能会改变。</li><li><a href="https://github.com/marc-rigter/waker">GitHub - marc-rigter/waker: Official code for &quot;Reward-Free Curricula for Training Robust World Models&quot; accepted to ICLR 2024.</a>：被 ICLR 2024 接收的论文 "Reward-Free Curricula for Training Robust World Models" 的官方代码。 - marc-rigter/waker
</li>
</ul>

</div>
  

---


**Eleuther ▷ #[lm-thunderdome](https://discord.com/channels/729741769192767510/755950983669874798/1220359878275764254)** (73 条消息🔥🔥): 

- **Megatron-Deepspeed 与 lm-eval 0.3.0 之间的兼容性问题**：一位参与者指出了 `megatron-deepspeed` 评估兼容性的一个 bug。建议从旧版本的 `cais/mmlu` 加载以绕过此问题，但由于辅助训练集划分被移动，这仍然会产生问题，如提供的 [Gist traceback](https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21) 所示。

- **修改版 lm-evaluation-harness 的内部使用**：一篇 [arXiv 论文](https://arxiv.org/abs/2403.09611) 引用了使用 EleutherAI 的 lm-evaluation-harness 的内部 fork 进行多模态预训练评估。随后讨论了获取其评估框架访问权限的好处，并邀请合作将 harness 扩展到多模态模型。

- **lm-evaluation-harness 的 WandB 日志记录挑战**：一位用户报告了在使用 8 个 GPU 运行时 WandB 会记录 8 次的问题，且 GSM8K 分数打印到了终端但未记录。建议将一段日志代码移至 `post_init()` 作为临时修复，且需要额外的测试协调。

- **量化激活支持查询**：有人提出了关于 eval harness 是否支持 W8A8 等量化激活的问题，得到的澄清是量化支持是通过 Huggingface 等其他库间接实现的，这些库可能提供一些 A8 方法。

- **与 Megatron-Deepspeed 潜在的数值差异**：讨论了关于使用 Huggingface transformers 和 Megatron-Deepspeed 进行评估时存在微小数值差异的担忧。据推测，融合 KQV 乘法的差异可能是由于使用了 bfloat16，虽然 flash attention 是确定性的，但仍有必要对前向传播（forward pass）输出进行分析。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2403.09611">MM1: Methods, Analysis &amp; Insights from Multimodal LLM Pre-training</a>：在这项工作中，我们讨论了构建高性能多模态大语言模型 (MLLMs)。特别是，我们研究了各种架构组件和数据选择的重要性。通过仔细的...</li><li><a href="https://x.com/BlancheMinerva/status/1770839679580901840?s=20">Stella Biderman (@BlancheMinerva) 的推文</a>：我们非常有兴趣扩大 eval harness 的范围，以包括多模态模型、RAG 和 AI 评分设置等内容。正在寻找构建此功能的最佳方式...</li><li><a href="https://x.com/BlancheMinerva/status/1770839676435210546?s=20">Stella Biderman (@BlancheMinerva) 的推文</a>：“我们所有的多模态预训练评估都是在 @AiEleuther 的 lm-evaluation-harness 的内部 fork 中实现的” 有机会分享代码吗 @mckbrando？那将是一个巨大的...</li><li><a href="https://huggingface.co/datasets/cais/mmlu/blob/main/hendrycks_test.py">hendrycks_test.py · cais/mmlu at main</a>：未找到描述</li><li><a href="https://gist.github.com/jonabur/d99bb92be81a5af6b01f81b589b68d21">gist:d99bb92be81a5af6b01f81b589b68d21</a>：GitHub Gist：即时分享代码、笔记和片段。</li><li><a href="https://huggingface.co/docs/transformers/main_classes/quantization#transformers.QuantoConfig>">Quantization</a>：未找到描述</li><li><a href="https://github.com/MineDojo/Voyager/issues/149">Implement a way test local models · Issue #149 · MineDojo/Voyager</a>：你好，Voyager 的工作非常出色。请考虑增加对本地模型的支持（不使用 openai 包，而是使用类似 Python requests 包连接到使用 openai 接口的本地模型...</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_test.py">lm-evaluation-harness/lm_eval/tasks/hendrycks_test.py at master · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness/blob/34c9b7e40825ec998e44c5f45041953249c06a7b/lm_eval/logging_utils.py#L98-L101">lm-evaluation-harness/lm_eval/logging_utils.py at 34c9b7e40825ec998e44c5f45041953249c06a7b · EleutherAI/lm-evaluation-harness</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness</li><li><a href="https://github.com/danijar/diamond_env">GitHub - danijar/diamond_env: Standardized Minecraft Diamond Environment for Reinforcement Learning</a>：用于强化学习的标准 Minecraft 钻石环境 - danijar/diamond_env</li><li><a href="https://github.com/EleutherAI/lm-evaluation-harness">GitHub - EleutherAI/lm-evaluation-harness: A framework for few-shot evaluation of language models.</a>：一个用于语言模型 few-shot 评估的框架。- EleutherAI/lm-evaluation-harness
</li>
</ul>

</div>
  

---



**HuggingFace ▷ #[announcements](https://discord.com/channels/879548962464493619/897387888663232554/1220491333257396325)** (1 条消息): 

- **ASCII 艺术数据集发布**：通过社区成员提供的新数据集探索 ASCII 艺术，其中包含 **andreas_who_is_who.txt** 和 **ascii_history_jgs.gmi** 等文本文件。点击[此处](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM)探索该数据集及各种 ASCII 艺术家资源。

- **旋律遇见模型**：使用 **SMIT** 将音频集成到您的语言模型中，这是一个可在 [GitHub](https://github.com/Thytu/SMIT/tree/main) 上获取的模态集成工具。在 [YouTube](https://youtu.be/nQCibZE14Bo) 上观看音乐生成模型微调过程的演示。

- **一个模型统治一切**：**Fluently-v4** 全球发布，提倡针对多项任务的单一模型解决方案。有关该模型及其涉及检查和 Loras 的创建细节已在 [Hugging Face](https://huggingface.co/fluently/Fluently-v4) 上展示。

- **AI 助力开放治理**：一篇博客文章讨论了 AI（特别是 LLM）在提高政府透明度和公共记录可访问性方面的潜力。在 [kyopengov.org](https://kyopengov.org/blog/exploring-open-records-law-ai) 上回顾了 GPT-4 和 Claude 3 等 AI 技术在该领域的应用。

- **使用 SVGDreamer 进行想象**：该博客揭晓了 SVGDreamer，这是一款使用 Diffusion Model 的新型文本引导矢量图形生成工具。该工具已发表于 **CVPR2024**，允许根据文本提示创建可编辑的矢量图形——更多信息请参阅 [Hugging Face blog](https://huggingface.co/blog/xingxm/svgdreamer)。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM">Csplk/THE.ASCII.ART.EMPORIUM · Hugging Face 数据集</a>：未找到描述</li><li><a href="https://github.com/Thytu/SMIT/tree/main">GitHub - Thytu/SMIT: SMIT: A Simple Modality Integration Tool</a>：SMIT：一个简单的模态集成工具。通过在 GitHub 上创建账号来为 Thytu/SMIT 的开发做出贡献。</li><li><a href="https://youtu.be/nQCibZE14Bo">fine-tuning musicgen + making an infinite remix - special episode - captains chair 18</a>：本周 Kev 尝试使用 @bleepybloops 的艺术家列表和 Colab Notebook 快速微调 MusicGen，进行一场有史以来最奇特的合作：https://github.co...</li><li><a href="https://huggingface.co/fluently/Fluently-v4">fluently/Fluently-v4 · Hugging Face</a>：未找到描述</li><li><a href="https://kyopengov.org/blog/exploring-open-records-law-ai">Exploring Open Records Law with AI | KOGC</a>：未找到描述</li><li><a href="https://huggingface.co/blog/xingxm/svgdreamer">SVGDreamer: Text Guided Vector Graphics Generation with Diffusion Model</a>：未找到描述</li><li><a href="https://github.com/dominiquegarmier/grok-pytorch">GitHub - dominiquegarmier/grok-pytorch: pytorch implementation of grok</a>：Grok 的 PyTorch 实现。通过在 GitHub 上创建账号来为 dominiquegarmier/grok-pytorch 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/AviSoori1x/makemoe2">Sparse Mixture of Experts Language Model from Scratch: Extending makeMoE with Expert Capacity</a>：未找到描述</li><li><a href="https://github.com/andrew-m-holmes/nura">GitHub - andrew-m-holmes/nura</a>：通过在 GitHub 上创建账号来为 andrew-m-holmes/nura 的开发做出贡献。</li><li><a href="https://huggingface.co/blog/vladbogo/video-mamba">VideoMamba: State Space Model for Efficient Video Understanding</a>：未找到描述</li><li><a href="https://github.com/di37/coding-assistant-codellama-streamlit">GitHub - di37/coding-assistant-codellama-streamlit: This project demonstrates how to utilize Codellama, a local open-source Large Language Model (LLM), and customize its behavior according to your specific requirements using a Modelfile.</a>：该项目演示了如何利用本地开源 LLM CodeLlama，并使用 Modelfile 根据您的特定需求自定义其行为。 - di37/codi...</li><li><a href="https://huggingface.co/blog/JMJM/vulnerabilities-top-10-hf-models">Giskard Bot: Identifying robustness, performance and ethical vulnerabilities in the Top 10 Most Popular Hugging Face Models</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2403.10853">Just Say the Name: Online Continual Learning with Category Names Only via Data Generation</a>：在现实场景中，由于成本高昂，持续学习的大量手动标注是不切实际的。虽然之前的研究受大规模网络监督训练的影响，建议...</li><li><a href="https://huggingface.co/blog/Pclanglais/common-corpus">Releasing Common Corpus: the largest public domain dataset for training LLMs</a>：未找到描述</li><li><a href="https://huggingface.co/blog/andmholm/what-is-automatic-differentiation">What&#39;s Automatic Differentiation?</a>：未找到描述</li><li><a href="https://huggingface.co/blog/lorinma/yi-9b-divedeep">Dive Deeper into Yi-9B</a>：未找到描述</li><li><a href="https://huggingface.co/blog/hrishioa/retrieval-augmented-generation-1-basics">Better RAG 1: Advanced Basics</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/1220285666550612028)** (76 条消息🔥🔥):

- **对 Cookbook 的好奇**：一位成员询问了 HuggingFace *learn* 章节中 “cookbook” 一词的含义，但回复中未提供具体细节。
- **在 Sdxl 1.0 和 Stable Cascade 之间选择**：讨论强调 **Sdxl 1.0** 或 **Stable Cascade** 可能是整体表现最好的模型，并可以通过专门的 finetuning 在各个领域进行改进。
- **Accelerate 的 Quantization 技术**：成员们讨论了 Accelerate 的 quantization 文档中的 `load_and_quantize_model` 功能，认为它是 `load_checkpoint_and_dispatch` 的一种可能替代方案，初步测试表明这是一个可行的选择。
- **Gradio API 调用与不活跃状态**：关于通过 **Gradio Client** 的 API 调用是否会自动重启不活跃 Space 的疑问未得到明确回答。
- **协作与专业知识请求**：发出了多项关于各种主题的协助或协作请求，包括 pretraining 数据挑战、涉及 PyTorch 专业知识的项目协作以及对模型 quantization 的理解。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://huggingface.co/spaces/fffiloni/coqui-bark-voice-cloning-docker">Coqui Bark Voice Cloning Docker - fffiloni 开发的 Hugging Face Space</a>：未找到描述</li><li><a href="https://hf-mirror.com/">HF-Mirror - Huggingface 镜像站</a>：未找到描述</li><li><a href="https://github.com/suno-ai/bark?tab=readme-ov-file#-installation">GitHub - suno-ai/bark: 🔊 文本提示生成音频模型</a>：🔊 文本提示生成音频模型。通过在 GitHub 上创建账号为 suno-ai/bark 的开发做出贡献。</li><li><a href="https://www.youtube.com/watch?v=dHcxTmU6atk">Coffee Shop AI - 咖啡师追踪</a>：咖啡店使用 AI 追踪咖啡师的生产力以及顾客在店内停留的时间。伙计们，我找到了源代码，就在这里：https://...</li><li><a href="https://www.youtube.com/watch?v=00TSeKZyeXQ">t-SNE 简单原理解析</a>：清晰且仔细地解释了数据科学中的 t-SNE 方法！0:00 邻居的概念 6:25 邻居相似度 8:17 关于标准差的说明 10:48 移动...</li><li><a href="https://github.com/coqui-ai/TTS">GitHub - coqui-ai/TTS: 🐸💬 - 一个用于文本转语音的深度学习工具包，经过研究和生产环境的考验</a>：🐸💬 - 一个用于文本转语音的深度学习工具包，经过研究和生产环境的考验 - coqui-ai/TTS
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/1220368201590902835)** (1 条消息): 

- **蛋白质序列获得向量增强**：**UniProt 项目**为其数据库中的大量蛋白质发布了 [1024 维 Embeddings](https://www.uniprot.org/help/embeddings)。一位成员正考虑使用 **Matryoshka embeddings** 对这些数据进行重新训练，以获得更好的搜索能力，正如最近的一篇 [HuggingFace 博客文章](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/README.md)中所描述的那样。
<div class="linksMentioned">

<strong>提及的链接</strong>:

<ul>
<li>
<a href="https://www.uniprot.org/help/embeddings">UniProt</a>：未找到描述</li><li><a href="https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/matryoshka/README.md">sentence-transformers/examples/training/matryoshka/README.md at master · UKPLab/sentence-transformers</a>：使用 BERT 的多语言句子和图像 Embeddings - UKPLab/sentence-transformers
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[cool-finds](https://discord.com/channels/879548962464493619/897390579145637909/1220280430771703899)** (10 条消息🔥): 

- **BitNet b1.58 发布**：一篇发表在 [arXiv](https://arxiv.org/abs/2402.17764) 上的论文详细介绍了一种名为 **BitNet b1.58 的新 1-bit LLM**，声称其性能可与全精度 LLM 相媲美，同时在延迟、内存、吞吐量和能耗方面更具成本效益。这项工作可能会推动针对 1-bit LLM 优化的硬件开发。

- **AI 驱动的数据分析技术兴起**：Medium 上的一篇文章讨论了使用 **Langchain, Instructor, 和 Pydantic** 重新定义 AI 数据分析，承诺在效率和能力上有所提升。文章可在此处阅读 [here](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616)。

- **人机团队凝聚力研究**：第一篇讨论在工程背景下研究 **人机团队 (HRTs) 凝聚力** 概念框架的博士论文，可在 Cambridge Core [此处](https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/conceptual-framework-to-study-team-cohesion-in-humanrobot-teams/9A1BD1CB1FB23B998E57A1AB1A299FCB)获取。

- **PatchTST 在时间序列预测中的突破**：一篇 *Towards Data Science* 的文章介绍了 **PatchTST**，这是一种有望推动时间序列预测发展的新方法。文章可以参考[这里](https://towardsdatascience.com/patchtst-a-breakthrough-in-time-series-forecasting-e02d48869ccc)。

- **衡量 LLMs 的 ASCII 艺术能力**：一项研究在 [arXiv](https://arxiv.org/pdf/2307.16806.pdf) 论文中提出了一套可衡量的指标，用于评估 Large Language Models 生成 ASCII 艺术的能力。

- **视觉处理机制教程**：一段来自 CVPR 2022 题为“通过高效编码原理理解早期视觉处理机制”的 YouTube 视频，深入探讨了生物视觉的工作原理。讲座可以在[这里](https://www.youtube.com/watch?v=Ed9otQAmEF4)观看。

- **无详细信息的研究兴趣**：一位成员分享了来自 IEEE Xplore 的一个可能有趣的链接，但未提供有关该[文档](https://ieeexplore.ieee.org/document/10333889)内容或相关性的直接信息。在后续消息中也未提供进一步描述。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://arxiv.org/abs/2402.17764">The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits</a>：最近的研究（如 BitNet）正在为 1-bit Large Language Models (LLMs) 的新时代铺平道路。在这项工作中，我们引入了一个 1-bit LLM 变体，即 BitNet b1.58，其中每一个参数...</li><li><a href="https://ieeexplore.ieee.org/document/10333889">Exploring Lightweight Federated Learning for Distributed Load Forecasting</a>：Federated Learning (FL) 是一种分布式学习方案，使深度学习能够以隐私保护的方式应用于敏感数据流和应用。本文重点关注...</li><li><a href="https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616">Harnessing Langchain, Instructor, and Pydantic: Redefining Data Analysis with AI</a>：Ankush k Singal</li><li><a href="https://www.youtube.com/watch?v=Ed9otQAmEF4">Understanding early visual processing mechanisms by the principle of efficient encoding</a>：这是 CVPR 2022 教程“关于生物（人类）视觉如何工作的后马尔计算概述”五个讲座中的第 2 讲，日期为 2022 年 6 月 19 日...</li><li><a href="https://www.cambridge.org/core/journals/proceedings-of-the-design-society/article/conceptual-framework-to-study-team-cohesion-in-humanrobot-teams/9A1BD1CB1FB23B998E57A1AB1A299FCB">CONCEPTUAL FRAMEWORK TO STUDY TEAM COHESION IN HUMAN-ROBOT TEAMS | Proceedings of the Design Society | Cambridge Core</a>：研究人机团队凝聚力的概念框架 - 第 3 卷
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/1220332454498406400)** (23 条消息🔥): 

- **追求 ASCII 大师之路**：一位参与者正在寻找合作者，共同应对微调语言模型以生成高质量 ASCII 艺术的挑战，此前他已通过自定义 GPTs 取得了一定进展。他们分享了 [ASCII Art 数据集](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM)，并表达了开发开源 LLM 的愿望，例如能够创建复杂的“不可能几何错觉”的 ASCII 艺术。

- **Telegram 机器人发布**：介绍了一个使用 Hugging Face Mistral AI 创建的 AI 机器人，并在 Telegram 上的 [@mistralaichat_bot](t.me/mistralaichat_bot) 体验后征求反馈。开发者正在寻求扩大规模和未来项目的合作。

- **Chaiverse 的 Beta 开发者平台**：来自 Chai Research 的一位工程师宣布了他们的 Beta 开发者平台 Chaiverse，该平台对社区生成的 LLMs 进行排名，并允许开发者提交模型以获取真实用户的反馈。感兴趣的人士可以阅读 [Chaiverse 白皮书](https://www.chaiverse.com/white-paper) 了解更多关于他们使命的信息。

- **推广联邦学习**：分享了一个 [GitHub 仓库](https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting)链接，该仓库专注于使用聚类和序列 DNN 方法进行负荷预测的联邦学习。

- **ASCII 艺术聊天实验**：参与者讨论了使用 LLMs 生成 ASCII 艺术的方法和挑战，包括使用 HTML 和 CSS 进行格式化，以及向模型请求复杂 ASCII 艺术时参差不齐的结果。共识似乎是模型有时可以生成像猫这样的简单表示，但在处理更复杂的设计时会感到吃力。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM">Csplk/THE.ASCII.ART.EMPORIUM · Hugging Face 数据集</a>: 未找到描述</li><li><a href="https://github.com/ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting">GitHub - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting: 使用聚类和序列 DNN 方法在能源数据集上进行负荷预测的 Federated Learning</a>: 使用聚类和序列 DNN 方法在能源数据集上进行负荷预测的 Federated Learning - ADG4050/Exploring-Lightweight-Federated-Learning-for-load-forecasting</li><li><a href="https://console.chaiverse.com">Leaderboard</a>: 未找到描述</li><li><a href="https://www.chaiverse.com/white-paper">白皮书 | Chaiverse | Chai AI 开发者平台</a>: 探索 Chai AI 通过众包实现万亿参数 AGI 跨越的愿景。
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/1220295378755391539)** (2 条消息): 

- **快，在日历上标记好！**: 活动详情已添加，并提供了 [活动链接](https://discord.com/events/879548962464493619/1219690164339736770)；预计今天还会发布公告。
- **用数据解码肥胖**: 查看关于肥胖趋势的 [深度 EDA notebook](https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda)，其中统计分析和可视化揭示了年龄、性别和生活方式选择对这一关键健康问题的相互作用。

**提到的链接**: <a href="https://www.kaggle.com/code/muhammadibrahimqasmi/deciphering-obesity-trends-an-in-depth-eda">解读肥胖趋势 📉：深度 EDA 📊</a>: 使用 Kaggle Notebooks 探索和运行机器学习代码 | 使用来自多个数据源的数据

  

---


**HuggingFace ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/1220301340278919179)** (4 条消息): 

- **警惕不请自来的私信报价**: 一名成员警告说，有人可能会通过私信请求付费工作，并指出此人之前因类似行为被其他 Discord 服务器踢出。
- **SegGPT 模型发布**: 新的 **SegGPT** 模型已添加，能够执行各种图像到图像任务，并具有令人印象深刻的 one-shot 分割结果。SegGPT 模型及其论文可通过 [Hugging Face 文档](https://huggingface.co/docs/transformers/main/en/model_doc/seggpt) 获取。
- **对 SegGPT 的感谢**: 一名成员表示感谢，并表示有兴趣尝试新推出的 **SegGPT** 模型。

**提到的链接**: <a href="https://huggingface.co/docs/transformers/main/en/model_doc/seggpt">SegGPT</a>: 未找到描述

  

---


**HuggingFace ▷ #[NLP](https://discord.com/channels/879548962464493619/922424173916196955/1220284164666818590)** (33 条消息🔥): 

- **寻求人格预测数据**: 正在探索适用于基于文本的 **personality prediction** 研究的数据集，由于 **myPersonality** 数据集不可用。此类应用公开数据集的稀缺，给学生级研究带来了挑战，因为获取大规模数据的途径有限。

- **用 LLM 掌握 ASCII 艺术之旅**: 一项令人兴奋的尝试正在进行中，旨在微调大型语言模型 (LLM) 以使其擅长生成 **ASCII art**，并提到了一个特定数据集 [THE.ASCII.ART.EMPORIUM](https://huggingface.co/datasets/Csplk/THE.ASCII.ART.EMPORIUM)，同时寻求关于如何为 LLM 训练有效嵌入 ASCII art 的指导。

- **分享深度代码生成数据集 —— "The Stack"**: 正在分享 **"The Stack" 数据集**，这是一个包含 300 多种编程语言、容量达 6TB 的源代码宝库，可能对代码生成项目有用。用户必须同意相关条款，包括原始代码许可证和数据删除更新，详见 [此处](https://huggingface.co/datasets/bigcode/the-stack)。

- **使用基于 BERT 的算法实现主题建模现代化**: 建议查看 **BERTopic**，这是一种使用 🤗 transformers 和上下文相关 embedding 的主题建模技术，提供各种主题建模方法，详见 [此处](https://maartengr.github.io/BERTopic/index.html)。

- **解决微调模型的量化挑战**: 关于量化 LoRA 适配模型的最佳实践讨论强调了合并和最小化量化损失的效用，相关示例可在 [PEFT 文档](https://huggingface.co/docs/peft/developer_guides/lora) 中找到。

- **Troubleshooting Trainer Class Issues in Huggingface**：报告了在 Huggingface 中使用 **Trainer class** 时遇到的问题，特别是关于依赖项需要更新和加速的问题。建议包括升级库、清理缓存、调整导入顺序以及考虑重启或重新配置。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/docs/peft/developer_guides/lora">LoRA</a>：未找到描述</li><li><a href="https://huggingface.co/datasets/bigcode/the-stack">bigcode/the-stack · Datasets at Hugging Face</a>：未找到描述</li><li><a href="https://maartengr.github.io/BERTopic/index.html">Home</a>：利用 BERT 和基于类的 TF-IDF 来创建易于解释的主题。</li><li><a href="https://huggingface.co/spaces/sentence-transformers/quantized-retrieval">Quantized Retrieval - a Hugging Face Space by sentence-transformers</a>：未找到描述</li><li><a href="https://huggingface.co/blog/embedding-quantization">Binary and Scalar Embedding Quantization for Significantly Faster &amp; Cheaper Retrieval</a>：未找到描述</li><li><a href="https://sbert.net/examples/applications/embedding-quantization/README.html">Embedding Quantization &mdash; Sentence-Transformers  documentation</a>：未找到描述
</li>
</ul>

</div>
  

---


**HuggingFace ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/1220302753138610216)** (28 messages🔥): 

- **损坏的状态字典 (State Dictionary) 之苦**：一名成员在尝试使用 `model.eval()` 加载微调模型时遇到了指示状态字典损坏的 **ValueError**。目前尚不清楚该问题是否提出了解决方案或已得到解决。

- **解码 Diffusion Checkpoint 代码**：简要解释了 checkpoint 存储了模型的学习信息，随后对话转向在 HuggingFace 上搜索诸如 **sdxl 1.0 或 stable diffusion 2.1** 之类的 checkpoint。

- **使用 Diffusion 模型生成 ASCII 艺术**：围绕为 ASCII 艺术数据集创建类似 Diffusion 的模型展开了讨论。对话探讨了将 ASCII 转换为图像，但关于**开发原生运行在 ASCII 上的 Diffusion 模型**的问题仍悬而未决。

- **金融 AI 聊天机器人构建**：一位用户询问如何为具有多个访问级别和分类的金融数据构建 AI 聊天机器人。在给出的消息中没有提出具体的模型，但另一位用户提到需要先审查数据。

- **好奇者想知道**：用户提出了关于加入名为 **Zero-GPU-Explorers** 小组的问题，并寻求在使用其数据集训练 **all-MiniLM-L6-v2 模型** 方面的帮助，表达了对社区支持的渴望。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://huggingface.co/CiroN2022/ascii-art">CiroN2022/ascii-art · Hugging Face</a>：未找到描述</li><li><a href="https://ivbhatt.medium.com/asciify-b3a0c70433fa">ASCIIfy</a>：使用 OpenCV、Pillow 和 Python 3 将图像转换为 ASCII
</li>
</ul>

</div>
  

---



**OpenAI ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/1220293288939225228)** (40 messages🔥):

- **将 Chat GPT 机器人添加到 Discord 及 API 成本**：要将 Chat GPT 机器人添加到 Discord 频道，必须获取 API，这并非免费服务，而是付费服务。
- **在 Postman 中接收响应时遇到的问题**：一位社区成员在设置了 assistant、thread 和 message 后，在 Postman 上无法接收到响应，建议其查阅 [文档](https://platform.openai.com/docs/api-reference/chat) 并检查响应中的 "content" 参数。
- **Perplexity 据称是一个搜索外壳 (Search Wrapper)**：一位成员分享了一篇文章，声称 Perplexity 可能会压缩来自 Google Search 前几名结果的内容，总结前 5-10 条条目的内容。Mohit Pandey 于 2024 年 3 月 18 日在 [此处](https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/) 发表了题为 *Perplexity is Most Likely a Google Search Wrapper* 的文章。
- **思考 AI 在视频压缩中的作用**：社区讨论探讨了在视频压缩中使用 AI 的想法，并将其潜在用途与用于音频压缩的深度学习超采样 (DLSS) 和 Whisper 进行了比较。一篇现有的博客文章在 [此处](https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html) 讨论了音频压缩方面的内容。
- **转换为 Int8 嵌入 (Embeddings) 以提高存储效率**：一位成员报告称，在将 Float32 嵌入预转换为 Int8 并发送到其向量数据库后，节省了约 80% 的存储成本。他们希望 embedding-v3 模型能提供原生的 Int8 支持以简化流程，并讨论了在多模态原型中为各种任务使用 pickle、sqlite 和其他数据库的潜在可能。

<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://www.dbreunig.com/2023/11/07/extreme-compression-with-ai.html">Extreme Compression with AI: Fitting a 45 Minute Podcast into 40kbs</a>：关于技术、文化、媒体、数据及其交互方式的文章。</li><li><a href="https://clickup.com/ai">ClickUp Brain | One AI to Replace them All</a>：未找到描述</li><li><a href="https://analyticsindiamag.com/perplexity-is-most-likely-a-google-search-wrapper/">Perplexity is Most Likely a Google Search Wrapper</a>：一位用户在 Reddit 的 LocalLLaMA 板块发帖称，Perplexity 总结了来自 Google Search 前 5-10 条结果的内容。
</li>
</ul>

</div>
  

---


**OpenAI ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/1220272435434229830)** (11 条消息🔥): 

- **自定义 API 连接说明**：一位成员询问如何通过 API 连接到自定义 GPT-3 模型。Solbus 指导其使用 Assistants API，并提供了一个链接以获取进一步帮助：[Assistants API 指南](https://help.openai.com/en/articles/8673914-gpts-vs-assistants)。

- **寻求关于动物化身 GPT 的反馈**：一位名为 boouyaah 的用户分享了一个 GPT 作品，该作品可以将个人转变为动物版本的自己，并寻求关于提示词 (prompts) 的反馈：[You, but as an animal](https://chat.openai.com/g/g-SGpDLmwE9-you-but-as-an-animal)。

- **固定自定义 GPT 数量突然减少**：Jaredquek 报告了一个关于可以固定到侧边栏的自定义 GPT 数量的问题，称之前固定的 GPT 消失了，现在限制只能固定 4 个，并寻求解释或解决方法。

- **优化跨多个 GPT 的知识文件分布**：Mikejeason 提出了一个问题：将知识文件分布在针对提示词不同部分量身定制的多个 GPT 中，是否比将所有内容合并到单个 GPT 中更有效率。
  

---


**OpenAI ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/1220355607471718411)** (41 条消息🔥):

- **规则提醒收紧**：在询问关于 **prompt engineering jobs** 的问题后，用户被提醒遵守禁止自我推广、招揽或广告的 **Rule 7**。一位用户提供了指向该规则的直接[链接](https://discord.com/channels/974519864045756446/1107255707314704505/1213395523973808148)，进一步明确了规则。
- **GPT-4 Vision 对残障人士表现冷淡**：一位用户表达了挫败感，因为 **GPT-4 Vision** 在提供有关残障人士的协助时失败，反复回答“抱歉，我无法提供帮助”。
- **工具包预告引发好奇**：尽管有禁止自我推广的规则，一位用户还是提到正在开发一个 **prompt chaining/prompt engineering toolkit**，并寻找人员测试原型。
- **挑战 ChatGPT 产品描述生成器**：针对使用 ChatGPT 为目录生成产品描述（侧重于天然和有机产品）的可行性进行了详细讨论。人们对 AI 在没有人工干预的情况下准确处理任务的能力表示怀疑。
- **寻求关于收益和应用的澄清**：讨论演变为简化 ChatGPT 的任务，一位用户建议专注于根据提供的产品描述生成**收益和用途**部分，这对于 AI 来说可能是一个更易于管理的方法。
  

---


**OpenAI ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/1220355607471718411)** (41 messages🔥): 

- **Rule 7 提醒**：一位新用户因询问 prompt engineering jobs 而无意中违反了 [Rule 7](https://discord.com/channels/974519864045756446/1107255707314704505/1213395523973808148)，被提醒查看服务器规则，特别是禁止自我推广、招揽或广告的条款。
- **为失误道歉**：在注意到服务器规则后，该用户进行了**道歉**，并承诺查看规则以确保不再发生此类情况。
- **讨论 GPT-4 Vision 的局限性**：一位成员讨论了让 **GPT-4 Vision** 承认残障人士的困难，系统给出了标准的无用回复。
- **Prompt 工具包推广违规**：用户 *quixoticquiche* 因宣传其 prompt chaining toolkit 并寻求反馈而违反了 Rule 7，导致再次收到关于服务器禁止招揽规则的提醒。
- **自动化产品描述的挑战**：成员们讨论了使用 ChatGPT 自动生成“详细产品描述”的可行性；人们对在没有人工监督的情况下生成内容的准确性和可靠性表示担忧。
  

---



**LangChain AI ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/1220302688995119174)** (96 messages🔥🔥): 

- **理解 LangChain 工具摄取**：一位成员询问是否可以将数组作为输入传递给 LangChain 中的工具，得到的解释是虽然提供了通用示例，但知识库中没有关于数组输入的具体案例。
- **GraphCypherQAChain 使用案例**：一位成员就如何在 GraphCypherQAChain 中执行小写字符串比较寻求建议，但知识库中没有提供具体信息。
- **学习检索增强生成 (Retrieval-Augmented Generation)**：向那些希望通过基于项目的方法学习以 LLM 为重点的 AI 的人推荐了一个免费资源：[Intro to AI for Developers](https://takehomes.com/library/developers/intro-to-ai)。
- **对 AI 挑战的幽默看法**：在一次轻松的讨论中，成员们开玩笑说将各种框架和技术与 LangChain 集成的复杂性，暗示其难度不亚于解决时空连续体问题。
- **数据库查询的动态决策**：讨论涉及创建一个能够根据用户问题决定是查询 SQL 数据库还是向量数据库的 Agent，强调了在 LangChain 使用案例中自动决策的必要性。
<div class="linksMentioned">

<strong>Links mentioned</strong>:

<ul>
<li>
<a href="https://takehomes.com/library/developers/intro-to-ai">A Practical Introduction to AI for Developers – TakeHomes Library</a>: 未找到描述</li><li><a href="https://js.langchain.com/docs/use_cases/graph/quickstart#chain>).">Quickstart | 🦜️🔗 Langchain</a>: 在本指南中，我们将介绍在...上创建问答链（Q&amp;A chain）的基本方法。</li><li><a href="https://python.langchain.com/docs/use_cases/web_scraping#question-answering-over-a-website>)">Web scraping | 🦜️🔗 Langchain</a>: 在 Colab 中打开</li><li><a href="https://tenor.com/bgtK0.gif">Ideas Genius GIF - Ideas Genius George - Discover &amp; Share GIFs</a>: 点击查看 GIF</li><li><a href="https://github.com/langchain-ai/langchain/issues/6138">ConversationChain default prompt leads the model to converse with itself · Issue #6138 · langchain-ai/langchain</a>: 系统信息 langchain==0.0.195 python==3.9.6 谁能帮忙？@hwchase17 信息 官方示例 notebooks/脚本 我自己修改的脚本 相关组件 LLMs/Chat Models Embedding Models...</li><li><a href="https://python.langchain.com/docs/use_cases/web_scraping#asynchtmlloader>).">Web scraping | 🦜️🔗 Langchain</a>: 在 Colab 中打开</li><li><a href="https://github.com/langchain-ai/langchain/issues/7876>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/11590>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/1438>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/4561>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/9389>)">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/4197>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/12410>),">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。</li><li><a href="https://github.com/langchain-ai/langchain/issues/13602>).">Issues · langchain-ai/langchain</a>: 🦜🔗 构建上下文感知的推理应用。通过在 GitHub 上创建账号为 langchain-ai/langchain 的开发做出贡献。
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[langserve](https://discord.com/channels/1038097195422978059/1170024642245832774/1220406790341001336)** (7 条消息): 

- **Python 版本地狱再次袭来！**: 一位成员尝试更新 **[langchain-ai/weblangchain](https://github.com/langchain-ai/weblangchain)**，并遇到了依赖项和 Python 版本的问题。错误 `TypeError: Type is not JSON serializable: numpy.float64` 导致应用程序崩溃，暗示了 `numpy` 数据类型的序列化问题。
  
- **可能与现有 Issue 相关**: 序列化问题可能与 langchain-ai 的 GitHub 讨论区中讨论的已知问题 **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langchain/discussions/17876)** 有关。

- **其他组件的故障排除**: 使用 LangSmith 进行测试未发现问题，因此问题可能与 TypeScript 客户端有关，因为将 Starlette 固定到旧版本并未解决该问题。

- **Poetry 无法解决所有问题**: 一位成员建议使用 Poetry 来摆脱 Python 版本问题，但结果发现已经在使用 Poetry，且问题在最新版本的 Langchain/Langserve 中依然存在。

- **在 GitHub 上提交了 Issue**: 序列化问题导致创建了一个名为 **[TypeError: Type is not JSON serializable: numpy.float64](https://github.com/langchain-ai/langserve/issues/551)** 的 GitHub Issue，以解决与最新版本 Langchain/Langserve 的不兼容问题。
<div class="linksMentioned">

<strong>提到的链接</strong>:

<ul>
<li>
<a href="https://smith.langchain.com/public/272f4463-4bb7-4fa3-ad5d-aea31dab5c8d/r">LangSmith</a>: 未找到描述</li><li><a href="https://github.com/langchain-ai/weblangchain/">GitHub - langchain-ai/weblangchain: 基于 LangChain 的网络研究员聊天机器人。在网络上搜索来源并在生成的回答中引用它们。</a>: 基于 LangChain 的网络研究员聊天机器人。在网络上搜索来源并在生成的回答中引用它们。 - langchain-ai/weblangchain</li><li><a href="https://github.com/mieslep/weblangchain/tree/compoent_and_update">GitHub - mieslep/weblangchain at compoent_and_update</a>: 基于 LangChain 的网络研究员聊天机器人。在网络上搜索来源并在生成的回答中引用它们。 - GitHub - mieslep/weblangchain at compoent_and_update</li><li><a href="https://github.com/langchain-ai/langserve/issues/551">TypeError: Type is not JSON serializable: numpy.float64 · Issue #551 · langchain-ai/langserve</a>: 我已将问题缩小到可以在 weblangchain 仓库 https://github.com/langchain-ai/weblangchain 中复现的范围。我正尝试更新到最新版本的 LangChain/LangSmith/L...</li><li><a href="https://github.com/langchain-ai/langchain/discussions/17876">TypeError: Type is not JSON serializable: numpy.float64 · langchain-ai/langchain · Discussion #17876</a>: 检查了其他资源。我为此 Issue 添加了一个非常详细的标题。我使用集成搜索查阅了 LangChain 文档。我使用 GitHub 搜索查找了类似问题并...
</li>
</ul>

</div>
  

---


**LangChain AI ▷ #[share-your-work](https://discord.com/channels/1038097195422978059/1038097372695236729/1220289106161041459)** (5 messages): 

- **AI 驱动的数据分析增强**：一篇名为《利用 LangChain、Instructor 和 Pydantic：用 AI 重新定义数据分析》的文章详细介绍了集成多种工具如何改变数据分析。在 [Medium](https://medium.com/technology-hits/harnessing-langchain-instructor-and-pydantic-redefining-data-analysis-with-ai-6cfe0e89b616) 上阅读全文。

- **推出 Promptsage 以简化 Prompt Engineering**：一个新的周末项目 Promptsage 旨在简化 LLM 的 prompt 构建和清理，具有安全和隐私护栏，并与 LangChain 兼容。在 [GitHub](https://github.com/alexmavr/promptsage) 上探索该工具。

- **探索针对大输出的 Chain 扩展**：一位成员询问了一项 **LangChain** 功能，该功能允许 chain 通过根据“停止原因 (Stop Reason)”判断发送额外请求，从而在超过模型的 token 限制后继续生成输出。该问题强调了对有效处理超过 token 限制（如 OpenAI GPT-4-Turbo 的 4k 输出 token）的大输出的需求。

- **Python 遇上 Bedrock Anthropic Haiku**：由于 Bedrock 缺乏对 function 的支持，一份全面的指南已经创建，演示了如何利用 Python 使用 Bedrock Anthropic Haiku。感兴趣的读者可以在 [Medium](https://medium.com/@leonardo.bolanos/leveraging-bedrock-anthropic-haiku-with-python-a-comprehensive-guide-9f5e912982be) 上找到该指南。

**提到的链接**：<a href="https://github.com/alexmavr/promptsage">GitHub - alexmavr/promptsage: Promptsage 是一个内置护栏的 LLM prompt 构建器、linter 和清理器</a>：Promptsage 是一个内置护栏的 LLM prompt 构建器、linter 和清理器 - alexmavr/promptsage

  

---



**OpenRouter (Alex Atallah) ▷ #[announcements](https://discord.com/channels/1091220969173028894/1092729520181739581/1220555842441580606)** (1 messages): 

- **美国西海岸请求缓慢**：**美国西海岸**的用户正经历异常缓慢的请求。怀疑原因是云服务问题，目前正在调查中。
  

---


**OpenRouter (Alex Atallah) ▷ #[general](https://discord.com/channels/1091220969173028894/1094454198688546826/1220385619822379088)** (53 messages🔥): 

- **对 Google Gemini 1.5 Pro 的好奇**：成员们讨论了具有 100 万 token 上下文窗口的 **Google Gemini 1.5 Pro** 的发布。尽管[公开文档](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning)仅提到 1.0 版本，但一位成员提到正在与 Google 联系以获取访问权限。

- **C3 模型的不一致性困扰**：用户对 C3 的表现表示沮丧，其中一人推荐使用自我审查版本的 **Claude 3**，声称它不太可能错误地拒绝内容，甚至在这方面超过了 GPT-4。

- **辩论 Grok AI 的能力**：对话转向了开源模型 Grok，关于其质量与 Mixtral 相比的观点出现了分歧，一些人因其高昂的成本和潜在的训练不足而将其贴上**“糟糕”**的标签。然而，其他人辩护称 Grok 作为一个纯基础模型的能力很强，强调将其与对话微调 (chat-tuned) 模型进行比较是不公平的。

- **Grok 基准测试受到质疑**：关于 Grok 基准测试的实用性展开了辩论，一些人质疑与 Mixtral 等模型的对比，而另一些人则指出 Grok 的官方基准测试显示它是一个知识渊博且风趣的对话模型。

- **探索 Grok 的测试与访问**：成员们讨论了如何测试 Grok，一位成员提供了[试用链接](https://grok.x.ai/)，并澄清可以通过 xAI 平台访问 Grok，可能不需要 Twitter Premium+。他们还讨论了可用于测试的内容，例如询问政治观点或 IT 相关问题。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://grok.x.ai/">xAI Grok</a>：未找到描述</li><li><a href="https://x.ai/blog/grok">Announcing Grok</a>：未找到描述</li><li><a href="https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versioning?authuser=1#gemini-model-versions">无标题</a>：未找到描述</li><li><a href="https://x.com/deliprao/status/1770128250003460396?s=46">来自 Delip Rao e/σ (@deliprao) 的推文</a>：我看着这个，并不觉得 Grok 更好。作为一个务实的人，我看着它并怀疑既然已经有了性能几乎相似的 Mixtral，为什么还要费心去用 Grok (314B)，而且它还是一个...
</li>
</ul>

</div>
  

---



**CUDA MODE ▷ #[general](https://discord.com/channels/1189498204333543425/1189498205101109300/1220361174344798319)** (4 条消息): 

- **Nanobind 作为解决方案**：一位成员建议研究 **nanobind** 以提高 MLX 的效率，并提到根据他们的经验，这可能会有所帮助。
- **感谢有用的建议**：同一位成员在后续消息中对查看 **nanobind** 的建议表示感谢。
- **GTC 活动期间 Discord 的小插曲**：在 GTC 活动期间，成员们遇到了 Discord Stage 频道屏幕共享无法正常工作的问题。通过移动到语音频道解决了该问题，并建议未来的讲座默认使用语音频道。
  

---


**CUDA MODE ▷ #[triton](https://discord.com/channels/1189498204333543425/1189607595451895918/1220583810396328016)** (1 条消息): 

- **将 GaLore 的 Adam 与 Triton 融合**：一位成员进行了一项研究，并在 GitHub 上提交了一个 [pull request](https://github.com/jiaweizzhao/GaLore/pull/29)，详细介绍了将 **GaLore 的 Adam 优化器**与 Triton 融合的过程。通过使用 `torch.matmul` 将梯度投影到低秩的混合内核（hybrid kernel）实现了最佳效果，从而提高了模型预训练和微调期间的内存效率。

**提及的链接**：<a href="https://github.com/jiaweizzhao/GaLore/pull/29">[WIP] Fused Adam Triton Kernels by jeromeku · Pull Request #29 · jiaweizzhao/GaLore</a>：融合 GaLore Adam (WIP) 针对梯度低秩投影的各种 Adam 更新步融合实现。这是优化 GaLore Adam 优化器更新步的初步尝试。总体而言...

  

---


**CUDA MODE ▷ #[cuda](https://discord.com/channels/1189498204333543425/1189607726595194971/1220298969800900639)** (1 条消息): 

- **micrograd 获得 CUDA 加速**：一位成员分享了一个库的链接 [micrograd-cuda](https://github.com/mlecauchois/micrograd-cuda)，该库使用 CUDA 内核扩展了 Karpathy 的 micrograd 库，并添加了 2D tensor 逻辑。该 GitHub 仓库欢迎为进一步开发这个 CUDA 加速版本的 micrograd 做出贡献。

**提及的链接**：<a href="https://github.com/mlecauchois/micrograd-cuda">GitHub - mlecauchois/micrograd-cuda</a>：通过创建一个账号为 mlecauchois/micrograd-cuda 的开发做出贡献。

  

---


**CUDA MODE ▷ #[torch](https://discord.com/channels/1189498204333543425/1189607750876008468/1220510281206202439)** (3 条消息): 

- **Lightning 赋能 PyTorch**：重点介绍了 [Lightning Thunder](https://github.com/Lightning-AI/lightning-thunder)，这是一个针对 PyTorch 的源码到源码（source-to-source）编译器，旨在加速单加速器和分布式系统上的 PyTorch 程序。

- **GTC 会议公告**：成员们获悉了即将举行的 GTC 演讲，并被提醒在约 24 小时后的会议开始前向特定人员提问。

- **NVIDIA GTC 会议链接**：提到了一个与 Thunder 相关的 NVIDIA GTC 演讲，并附带了[会议目录的直接链接](https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=Thunder%20#/session/1696294424486001JD3i)，详细说明了 3 月 17 日至 21 日在加州圣何塞及线上举行的研讨会、AI 会议和博览会以及主题演讲的日期。
<div class="linksMentioned">

<strong>提及的链接</strong>：

<ul>
<li>
<a href="https://www.nvidia.com/gtc/session-catalog/?tab.allsessions=1700692987788001F1cG&search=Thunder%20#/session/1696294424486001JD3i">NVIDIA #GTC2024 会议议程目录</a>: 立即注册。在线直播。2024年3月18-21日。</li><li><a href="https://github.com/Lightning-AI/lightning-thunder">GitHub - Lightning-AI/lightning-thunder: PyTorch 的源码到源码编译器。它使 PyTorch 程序在单个加速器和分布式环境下运行更快。</a>: PyTorch 的源码到源码编译器。它使 PyTorch 程序在单个加速器和分布式环境下运行更快。 - Lightning-AI/lightning-thunder
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[algorithms](https://discord.com/channels/1189498204333543425/1189861061151690822/1220499189172273232)** (9 条消息🔥): 

- **Ozaki 方案增强矩阵乘法**：正如一篇 arXiv 论文中所解释的，[Ozaki 方案](https://arxiv.org/abs/2301.09960)优化了多精度基础线性计算，并且在固定和任意精度的矩阵乘法中比现有方法表现更快。它受益于优化的低精度操作，并在达到一定精度之前优于 Strassen 矩阵乘法。
- **探索 Kahan 求和算法**：分享了一个指向关于 [Kahan 求和算法](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)的维基百科文章链接，这是一种数值分析方法，通过维持运行补偿来显著减少求和过程中的数值误差。
- **IEEE 754 标准**：引用了一篇讨论 [IEEE 754 标准](https://www.itu.dk/~sestoft/bachelor/IEEE754_article.pdf)的 ITU 论文，这些标准对于浮点计算至关重要。
- **Jeremy 的团队对被提及表示感谢**：Jeremy Howard 对提及他们与 Ozaki 方案实现相关的工作表示感谢，并表达了未来在该领域进行改进的意愿。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://en.wikipedia.org/wiki/Kahan_summation_algorithm">Kahan 求和算法 - 维基百科</a>：未找到描述</li><li><a href="https://arxiv.org/abs/2301.09960">使用 Ozaki 方案加速多精度矩阵乘法</a>：优化的多精度基础线性计算，特别是矩阵乘法，对于解决病态问题至关重要。最近提出的 Ozaki 方案实现了精确的...
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[suggestions](https://discord.com/channels/1189498204333543425/1189868872887705671/1220448064850886677)** (5 条消息): 

- **有序对话的建议**：一位成员建议对新话题使用**标准消息**，对分支对话使用**回复**，对特定主题使用**创建线程**，强调了这对消息可见性和频道可读性的好处。他们称赞了用户 <@272654283919458306> 在另一个名为 Latent Space 的服务器上出色的线程管理。
- **对服务器技巧的认可**：一位成员对关于 Discord 对话组织的礼仪建议表示赞赏，认为这些建议作为新用户特别有帮助。
- **分享资源链接**：一位成员分享了一个 [Springer 书籍链接](https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes)，重点介绍了 PPAM 2022 会议论文集的详细信息，包括贡献者和可访问论文的目录。

**提到的链接**：<a href="https://link.springer.com/book/10.1007/978-3-031-30442-2#other-volumes">并行处理与应用数学</a>：未找到描述

  

---


**CUDA MODE ▷ #[pmpp-book](https://discord.com/channels/1189498204333543425/1194427148656721970/1220761584335327242)** (2 条消息): 

- **寻求答案验证**：一位成员完成了 'pmpp-book' 的**第 2 章练习**，正在寻找验证答案的方法。他们表达了通过 **DM（私信）交流**与他人交叉检查答案的兴趣。
  

---


**CUDA MODE ▷ #[off-topic](https://discord.com/channels/1189498204333543425/1215328286503075953/1220466248907751445)** (2 条消息): 

- **CUDA 爱好者的 Lightning 突袭**：一位成员强调了在 GTC 上推出的全新 [Zero to Thunder 教程](https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial)，目标受众是希望为非标准模型提供自定义 CUDA kernel 的 Python 和 PyTorch 爱好者。虽然它仍处于实验阶段，且可能缺少某些功能，但对于勇于冒险的人来说，这是一个诱人的尝试。

- **微笑的 GPU 引起轰动**：通过 [Twitter 链接](https://fxtwitter.com/iScienceLuvr/status/1770931936657358908) 分享的一个观察指出，新款 Blackwell GPU 似乎带有笑脸，这一有趣的发现非常逗乐。这个独特的设计特征吸引了技术爱好者的注意，并在网上引发了幽默的评论。
<div class="linksMentioned">

<strong>提到的链接</strong>：

<ul>
<li>
<a href="https://lightning.ai/lightning-ai/studios/zero-to-thunder-tutorial">zero-to-thunder-tutorial - t-vi 开发的 Lightning Studio</a>：快速上手 ⚡ Lightning Thunder。</li><li><a href="https://fxtwitter.com/iScienceLuvr/status/1770931936657358908">Tanishq Mathew Abraham, Ph.D. (@iScienceLuvr) 的推文</a>：为什么没有人讨论新款 Blackwell GPU 真的在对着我们微笑这件事，哈哈。
</li>
</ul>

</div>
  

---


**CUDA MODE ▷ #[triton-puzzles](https://discord.com/channels/1189498204333543425/1219683012707487794/1220358094777876580)** (12 条消息🔥): 

- **Tensor 颜色代码解析**：在 triton-puzzles 中，Tensor 的颜色编码得到了澄清；颜色代表源 Tensor，越界访问显示为红色。讨论中提到一个潜在的 Bug，即即使 masking 正确，**越界加载（out-of-bounds loads）也可能被错误地标记**。
  
- **Tensor 绘制顺序说明**：Triton 中 Tensor 的绘制顺序被指定为深度、行、列，而一维 Tensor 被绘制为 1,1,col。

- **Triton `tl.exp` 算子问题报告**：一位新成员在 Triton 中使用 `tl.exp(x)` 或 `x.exp()` 时遇到了 `NotImplementedError`，指出该操作在没有 numpy 实现的解释器模式（interpreter mode）下不受支持。

- **成员提交 Exp2 的 PR**：提到了 exp2 函数，可能是在 flash attention 中进行修复或实现的上下文，并已向 Triton 提交了 Pull Request。

- **完成 Puzzle 3，正在调试 Puzzle 4**：一位成员完成了 Puzzle 3，并分享了他们在 Puzzle 4 中使用 print 语句的调试过程，他们通过使用 torch 进行外和（outer sum）运算来将自己的答案与预期结果进行对比。
  

---



**LLM Perf Enthusiasts AI ▷ #[general](https://discord.com/channels/1168579740391710851/1168579740391710855/1220818264251174993)** (21 条消息🔥): 

- **空间匹配难题**：Uniti AI 的 AI 租赁 Agent 面临问题，**GPT4.Turbo** 无法准确将物业库存与用户需求匹配——例如，当用户要求 2,000 - 4,000 平方英尺的空间时，它却建议了 17,000 平方英尺的物业。
- **复杂的匹配逻辑挑战**：一个微妙的挑战是，对于 5,000 平方英尺以下的物业提供 33% 范围内的库存，而对于 5,000 平方英尺以上的物业提供 20% 范围内的库存，这使得匹配过程日益复杂。
- **简化方法的建议**：目前的方法涉及使用详细的 Prompt 来匹配经纪人的查询，但有人建议改用**常规过滤器或让 LLM 生成 SQL 查询**来提取正确的单元。
- **认识到对 LLM 的过度依赖**：对话强调了一个“常见的 LLM 陷阱”，即并非所有任务都需要 LLM，简单的数据库查询可能是更好的解决方案。思路是使用 LLM 生成查询，而不是由其直接进行过滤。
- **参考 RAG 以提高效率**：提到了 Jason Liu 关于 [检索增强生成 (RAG)](https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/) 的博客文章，说明了将 LLM 与常规数据库查询结合使用在日期范围提取等任务中的有效性。

**提到的链接**：<a href="https://python.useinstructor.com/blog/2023/09/17/rag-is-more-than-just-embedding-search/">RAG 不仅仅是嵌入搜索 - Instructor</a>：未找到描述

  

---


**LLM Perf Enthusiasts AI ▷ #[claude](https://discord.com/channels/1168579740391710851/1168582222194933860/1220596048871821382)** (5 条消息): 

- **Bedrock 与直接连接方式的对比**：一位用户提到，选择与 Claude **直接集成**可能更有效，因为 **Bedrock** 已被证明有些笨重，且在运行时间（uptime）方面不太可靠。
- **Claude 的前线访问权限**：一位成员透露，由于长达一年的开发合作伙伴关系，他们拥有 Claude 的**优先速率限制（priority rate limits）**，这使他们领先于超过 **20 万人的等候名单**。
- **选择直接连接而非 Bedrock**：尽管拥有优先访问权，该用户确认他们仍在使用**直接连接**与 Claude 交互，绕过了 Bedrock 框架。
  

---


**LLM Perf Enthusiasts AI ▷ #[jobs](https://discord.com/channels/1168579740391710851/1169107992587812864/)** (1 条消息): 

ibash: > 编写高质量代码
该死。
  

---


**LLM Perf Enthusiasts AI ▷ #[openai](https://discord.com/channels/1168579740391710851/1171903046612160632/)** (1 条消息): 

jeffreyw128: 哈哈什么鬼
  

---

**LLM Perf Enthusiasts AI ▷ #[prompting](https://discord.com/channels/1168579740391710851/1179271229593624677/)** (1 条消息): 

emrgnt_cmplxty: 基础的 Prompting 无法满足你的需求吗？
  

---



**Interconnects (Nathan Lambert) ▷ #[ideas-and-feedback](https://discord.com/channels/1179127597926469703/1179127598442348730/1220363095277309952)** (15 条消息🔥): 

- **寻求用于 LLM 研究的合成基准测试**：一位成员询问是否存在具有可控属性的全合成基准测试，用于研究 *foundation model* 的能力，特别是 LLM。
- **初创公司通过 LLM 生成数据**：有人提到，初创公司正在利用 LLM 根据他们正在研究的模型生成大量数据，从而创建合成基准测试。
- **将 LLM 能力与数据质量解耦**：讨论了通过改变训练数据中的多样性和推理存在来研究 LLM 能力起源的问题，旨在超越“能力全在数据中”的普遍认知。
- **合成数据与合成世界引起关注**：一位成员对合成数据和合成世界表达了热情，并考虑就此主题撰写论文。
- **组织开源数据策展**：一位成员建议，采用公开、系统的方法来构建预训练数据，可能有利于组织开源数据策展（Data Curation）工作。
  

---


**Interconnects (Nathan Lambert) ▷ #[ml-questions](https://discord.com/channels/1179127597926469703/1179208129083363358/1220546492830585003)** (6 条消息): 

- **寻找 SOTA**：提到了 ChatGPT 作为内容重写的工具，暗示使用语言模型进行重写是实现 **SOTA (state-of-the-art)** 结果的常用做法。
- **工作流中的微调**：一位成员讨论了使用 ChatGPT 重写内容，并进行**微小调整**以使输出符合其需求。
- **项目进行中**：提到了一个与 ChatGPT 和重写相关的侧边项目，该成员指出由于参与度有限，目前缺乏实质性的见解。
- **学术冲刺**：ChatGPT 的重写能力正被用于加速完成一个**课程项目**。
  

---


**Interconnects (Nathan Lambert) ▷ #[random](https://discord.com/channels/1179127597926469703/1183121795247779910/1220363410491969576)** (5 条消息): 

- **机器人击败人类对心理的影响**：一位成员推测了输给 AI 对人类玩家的影响，指出尽管出现了超人水平的 AI，国际象棋等游戏依然未受损伤。
- **历史性的 AI 胜利引发共鸣**：一位参与者指出，*加里·卡斯帕罗夫输给深蓝 (Deep Blue)* 对国际象棋产生了重大影响，类似于后来 AI 在围棋（Go）中的胜利。
- **个人玩家的看法**：一位用户发表意见，认为玩家受 AI 影响的程度可能因个人性情而异。
- **关于 AI 的哲学讨论**：有人分享了一个链接，内容是与 *Minqi Jiang* 和 *Marc Rigter* 讨论在强化学习（RL）中构建通用 Agent 的可能性 ([MLStreetTalk Tweet](https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw))。

**提到的链接**：<a href="https://x.com/mlstreettalk/status/1770516991943586021?s=46&t=_jodDCDeIUnWb_Td0294bw">来自 Machine Learning Street Talk (@MLStreetTalk) 的推文</a>：我们刚刚发布了与 @MinqiJiang 和 @MarcRigter 的节目，讨论了在原则上和实践中，是否可能在 RL 中构建一个“通用 Agent”的哲学。

  

---



**Alignment Lab AI ▷ #[looking-for-collabs](https://discord.com/channels/1087862276448595968/1095393077415383261/1220406585336004651)** (1 条消息): 

- **为 01 开源硬件寻求帮助**：一位成员介绍了一个名为 **the 01** 的新型开源硬件设备，并请求社区贡献。该项目的硬件和软件完全开源，详情见[这条推文](https://twitter.com/OpenInterpreter/status/1770821439458840846)。
  

---


**Alignment Lab AI ▷ #[general-chat](https://discord.com/channels/1087862276448595968/1095458248712265841/)** (1 条消息): 

venadore: 人生教训
  

---



**Skunkworks AI ▷ #[off-topic](https://discord.com/channels/1131084849432768614/1140423597454807179/)** (1 条消息): 

pradeep1148: https://www.youtube.com/watch?v=21Tc92g15pM
  

---