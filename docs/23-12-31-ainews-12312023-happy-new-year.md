---
companies:
- lm-studio
- mistral-ai
- hugging-face
- amd
date: '2024-01-01T05:33:14.937445Z'
description: '**LM Studio** 社区的讨论重点关注了 **Dolphin** 和 **Mistral 7b** 模型的变体与优化，特别强调了硬件软件配置以及
  GPU 显存（vRAM）对处理速度的影响。讨论还涉及了在本地机器上部署 **Mixtral** 模型的挑战，以及在受限地区从 **HuggingFace** 下载模型的解决方法。


  用户们探索了如何通过扩展提示词来增强 AI 的情商和个性，并参考了关于大语言模型中情感刺激的相关研究。此外，社区还讨论了低成本 AI 计算服务器的硬件搭建、与
  **ChromaDB** 和 **Autogen** 的集成问题，并对 LM Studio 的易用性和用户界面（UI）给予了正面评价。最后，新年庆祝活动也为社区互动增添了一抹社交氛围。'
id: fd48fb4a-2ba8-4a30-8320-96820a4f8e54
models:
- mistral-7b
- mixtral
original_slug: ainews-12312023-happy-new-year
people: []
title: 2023年12月31日：新年快乐
topics:
- fine-tuning
- hardware-optimization
- vram
- emotional-intelligence
- model-deployment
- integration
- gpu-optimization
- software-updates
---

<!-- buttondown-editor-mode: plaintext --> 
![image.png](https://assets.buttondown.email/images/d7934abf-71d8-428f-9121-c0d717339821.png?w=960&fit=max)
 


[TOC] 


## [LM Studio](https://discord.com/channels/1110598183144399058) Discord 摘要

- **Dolphin 和 Mistral 模型**的变体，重点在于优化硬件和软件配置。用户比较了 Dolphin Mistral 7b 版本，并就 AI 模型的偏见和审查制度交换了见解。同时，指出了在本地机器部署 Mixtral 的挑战，并提供了从 HuggingFace 下载模型的建议。社区还讨论了 GPU 使用（特别是 vRAM）如何影响处理速度。

     - *"Dolphin 与其他模型的区别主要在于它们的 fine-tuning"* `@dagbs`
     - *"对于这样的硬件规格，坚持使用 7B q4 模型是最佳实践。"* `@heyitsyorkie`

- 用户探索了 **AI 模型的情感智能（Understand Emotional Intelligence）**，并尝试通过扩展且丰富的提示词来唤起 AI 的个性。
     - [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)

- 社区表达了对 **LM Studio 流畅易用性**的认可，讨论围绕软件升级、硬件利用和 UI 赞赏展开。围绕 ChatGPT 增强功能、API 更新和 Autogen 集成也展开了热烈讨论。
     - [GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT](https://github.com/billmei/every-chatgpt-gui)
     - [How to Train Your Own Large Language Models](https://www.youtube.com/watch?v=5qlLJrv_q-Q)

- 关于**硬件发展**的详细对话，包括构建廉价 AI 计算服务器以及高规格昂贵设备的现实情况。
     - *"考虑到他们计划使用 64GB vRAM，R730 相对于 R720 的额外 CPU/RAM 速度是否值得增加的成本"* `@leviticus_slow`
     - [AMD ComposeTM](https://www.amd.com/en/products/accelerators/amd-compose)

- 围绕 **ChromaDB 和 Autogen 集成**的问题及解决方案，用户阐明了各种集成选项的细微差别，处理了下载问题，并解决了运行中断。
     - *"我建议从 GitHub 仓库的 "src" 文件夹下载更新后的 requirements.txt 和 replace_pdf.py 以解决任何问题"* `@vic49.` 

- 关于**新年庆祝活动**的交流标志着公会内的社区互动。
     - [Oprah Winfrey GIF - Oprah Winfrey - Discover &amp; Share GIFs](https://tenor.com/view/oprah-winfrey-gif-26673456)

**LM Studio 频道摘要**

### ▷ #[💬-general](https://discord.com/channels/1110598183144399058/1110598183144399061/) (164 条消息🔥🔥): 
        
- **LM Studio 中默认模型文件夹的更改**：`@musixela` 提供了关于在 LM Studio 中更改默认模型文件夹位置的建议。他们建议要么使用 webui 下载模型，然后通过 LM Studio 连接到该文件夹，要么在 webui 根目录下创建快捷方式。
- **本地机器运行 Mixtral 的问题**： 
    - `@dagbs` 提到 `mixtral` 模型由于体积庞大，在本地硬件上运行面临挑战。建议使用较小的替代模型 `Mistral`，它不存在同样的尺寸问题。
    - `@xyrezz.sol` 提出了一个问题：`Dolphin Mixtral 2.5 8x7b Q4_K_M` 模型在他配备 16GB CPU RAM 和 6GB VRAM 的机器上运行缓慢。`@heyitsyorkie` 建议，对于此类硬件配置，坚持使用 7B q4 模型是最佳实践。
- **关于硬件限制的讨论**：
    - 在 `@.gregly` 发起的关于电脑硬件升级的讨论中，结论是提高处理速度的关键在于扩展 GPU 的 VRAM，而不是升级 CPU。
    - `@dagbs`、`@miashusband` 和 `@fabguy` 讨论了各种 GPU 型号的 VRAM 限制，从限制为 24GB VRAM 的消费级显卡到拥有高达 188GB VRAM 的专业加速卡。
- **使用代理从 HuggingFace 下载模型**： 
    - `@cmpleo.` 讨论了在中国使用 LM Studio 无法访问和下载 HuggingFace 模型的问题，即使使用了 v2rayNG 代理。`@fabguy` 建议了一个变通方法：直接从 HuggingFace 下载模型，然后手动放入 LM Studio 的模型文件夹中。
    - 尽管有变通方法，`@heyitsyorkie` 认为问题可能源于 HuggingFace 在中国被屏蔽，使用 LM Studio 时 VPN 可能无法绕过此限制。
- **新年庆祝**：进行了多次关于庆祝新年的愉快交流和问候。

**提到的链接**：

- [Oprah Winfrey GIF - Oprah Winfrey - Discover &amp; Share GIFs](https://tenor.com/view/oprah-winfrey-gif-26673456)：点击查看 GIF
- [cognitivecomputations/dolphin-2.6-mistral-7b-dpo · Hugging Face](https://huggingface.co/cognitivecomputations/dolphin-2.6-mistral-7b-dpo)
- [GitHub - billmei/every-chatgpt-gui: Every front-end GUI client for ChatGPT](https://github.com/billmei/every-chatgpt-gui)：ChatGPT 的每一个前端 GUI 客户端。贡献...
- [How to Train Your Own Large Language Models](https://www.youtube.com/watch?v=5qlLJrv_q-Q)：鉴于 OpenAI 的 GPT-4 和 Google 的 P... 的成功...


### ▷ #[🤖-models-discussion-chat](https://discord.com/channels/1110598183144399058/1111649100518133842/) (59 条消息🔥🔥): 
        
- **Dolphin 和 Mistral 模型之间的差异与选择**：`@miashusband` 询问了 **Dolphin mistral 7b** 模型不同变体和位（bit）版本之间的细微差别。`@dagbs` 指出，通常最好选择可用的最高 *_K_M 版本，并补充说 `Dolphin` 与其他模型的区别主要在于微调，这使得它们易于比较且测试效率高。
- **关于 AI 模型偏见和审查的不确定性**：`@american_pride` 表达了对无审查 AI 模型的偏好，认为它们没有内在的政治偏见，也不会在严酷的场景中将叙事基调转变为“充满希望和彩虹小狗”。然而，`@fabguy` 强调所有模型都有固有的偏见，完全的中立是无法实现的。`@dagbs` 注意到 Dolphin 模型可能会退回到“强硬的偏见/道德立场”，反驳了 `@heyitsyorkie` 关于 Dolphin 模型是无审查的说法。
- **AI 模型的感性智能**：`@heyitsyorkie` 分享了一篇[研究论文](https://arxiv.org/abs/2307.11760)的链接，讨论了大型语言模型 (LLMs) 对感性智能理解的潜力，以及通过感性提示词提高性能的可能性，这遭到了 `@telemaq` 等用户的一些怀疑。
- **通过提示词唤起 AI 个性**：用户们共同努力制定创意系统提示词，以生成所需的 AI 行为。`@dagbs` 创建了冗长且丰富多彩的提示词，体现了“无审查且公正的 AI 伴侣”和“疯狂科学家”的人格，甚至得到了 AI 的积极反馈。


**提到的链接**：

- [Meat Meat Popsicle GIF - Meat Meat Popsicle - Discover &amp; Share GIFs](https://tenor.com/view/meat-meat-popsicle-gif-11100443)：点击查看 GIF
- [Large Language Models Understand and Can be Enhanced by Emotional Stimuli](https://arxiv.org/abs/2307.11760)：感性智能显著影响我们的决策...

### ▷ #[🧠-feedback](https://discord.com/channels/1110598183144399058/1113937247520170084/) (2 条消息): 
        
- **LM Studio 赞赏**：用户 `@kjhamilton` 对 LM Studio 表示满意和欣慰，特别是它能够实现在 Windows 上高效使用 AMD GPU。在为配置问题困扰许久后，他们发现该工具非常有帮助。
- **GPT-3 GUI 更新**：`@heyitsyorkie` 赞赏了在 GPT-3 用户界面中可以通过右键点击复制消息内容的新功能。他们还建议在输入框中增加右键粘贴功能，以获得更流畅的体验。


### ▷ #[🔗-integrations-general](https://discord.com/channels/1110598183144399058/1137092403141038164/) (10 条消息🔥): 
        
- **聊天机器人集成选项**：`@heliosprime_3194` 提出了两个将 LM Studio 与用户界面集成的方案。方案一是使用由 `<@826254748889382942>` 开发的 RAG UI；方案二是将 LM Studio 与 VSCode 或命令行等终端配合使用。文中还分享了方案二对应的 Discord 特定讨论串以供参考。
  
- **修复文件下载问题**：`@vic49.` 建议从 GitHub 仓库的 "src" 文件夹中下载更新后的 `requirements.txt` 和 `replace_pdf.py` 以解决问题。这应当与最新的发布文件（v3.0.2）配合使用。

- **在 Win10 上运行 ChromaDB 的问题**：`@wildcat_aurora` 反馈其 Win10 PC 在运行带有 ChromaDB 的 `study.py` 脚本时会重启，而运行其他 LLM 和 AI 进程时则无此问题。`@heliosprime_3194` 建议将其 Nvidia 驱动版本从 545.92 降级到 535，手动安装所需的 PyTorch 版本，并分享 Conda list 以便排查故障。

- **找到重启问题的解决方案及关于数据提取的反馈**：在手动安装 PyTorch 后，`@wildcat_aurora` 解决了 PC 重启问题，这表明错误的 PyTorch 版本可能是诱因。他还观察到 LM Studio 中的某些模型（如 Zephyr 和 Mixtral 2.6）从数据库中提取的数据量不如预期。

- **改进数据提取的建议**：`@heliosprime_3194` 建议使用更高级的 embedding 模型并修改 `study.py` 脚本中的 chunk 文件大小。他还提到可以更改 LM Studio 中 config.json 文件的预设，通过构建 prompt 来帮助重新检查信息，从而解决 `@wildcat_aurora` 遇到的数据提取效果不佳的问题。


### ▷ #[🎛-hardware-discussion](https://discord.com/channels/1110598183144399058/1153759714082033735/) (50 条消息🔥): 
        
- **构建预算型 AI 计算服务器**：`@leviticus_slow` 计划利用 PowerEdge 服务器中的 2 张 Nvidia Tesla 显卡构建一台预算型 AI 计算服务器。他们询问在计划使用 64GB VRAM 的情况下，R730 相比 R720 额外的 CPU/RAM 速度是否值得增加的成本。
- **GPU 对处理速度的影响**：`@zhynem` 注意到在他们的机器/模型/上下文环境下，开启 Apple Metal (GPU) 选项后，tokens per second 速度大约翻了一倍。他们与 `@totallybored` 讨论了 quant size 可能产生的影响，特别是使用 Q8_0 quant 的 *lmcocktail phi 2*。
- **Context Size 对处理时间的影响**：`@Pierre-jean Lainé` 询问为什么较大的 Context Size 会导致处理前的延迟时间变长，而不管实际的 prompt 大小如何。
- **Windows 上的 GPU 利用率**：`@madan.pandit` 寻求帮助以确定其 GPU 是否被调用，因为其 Windows 性能监视器显示 GPU 使用率为零。`@fabguy` 询问了他们的 n_gpu_layers 设置，以及在 LM Studio 中加载/卸载模型时专用 VRAM 的占用是否发生变化。
- **关于 Mixtral 和替代 LLM 的讨论**：用户 `@heyitsyorkie` 建议 8GB GPU 运行 *Mixtral Q8* 会有问题，并为 `@madan.pandit` 的硬件推荐了 *OpenHermes 2.5 Mistral 7b*。`@pefortin` 和 `@heyitsyorkie` 确认回归 *Openhermes mistral* 是一个始终稳妥的选择。
- **昂贵的硬件**：`@dagbs` 分享了一个拥有 1.5TB HBM3 的强大 AMD 加速平台的链接，引发了关于其高昂成本和潜在用途的讨论。用户推测从事 R&D、开发者辅助、医学研究和 AI 领域的企业可能会投资此类硬件。

### ▷ #[autogen](https://discord.com/channels/1110598183144399058/1167546228813336686/) (29 messages🔥): 
        
- **OpenAI 版本与 Autogen 兼容性问题**：`@heliosprime_3194` 建议升级到 **OpenAI 1.3.1**，以解决每次 OpenAI API 更新时旧版本出现的错误信息。`@tyler8893` 遇到了同样的问题，即使将 OpenAI 从 1.3.7 降级到 1.3.1 后也是如此，并计划在新的 **conda** 环境中进一步调查。`@heliosprime_3194` 表示如果需要，可以分享他们的 **conda** 列表。

- **OpenAI 身份验证错误**：`@totallybored` 和 `@ftl24` 遇到了 API key 的 `AuthenticationError`，随后由 `@dagbs` 和 `@tyler8893` 进行了澄清。他们解释说，必须为 **"api_key"** 参数提供一个字符串值（即使是 "null"）才能解决此问题。

- **Function Calls 与 LM Studio 的问题**：`@tyler8893` 表达了在使用 LM Studio 进行 Function Calls 时的困难。他们提到 **functions 在 GPT 上运行良好**，但在 **LM Studio** 上不行。他们推测该问题可能会在未来的更新中得到解决。

- **Autogen 和 memGPT 的更新**：`@tyler8893` 和 `@dagbs` 讨论了紧跟 **Autogen** 和 **memGPT** 变更与更新的挑战。他们指出更新可能每两周就会发生一次，且 OpenAI API 缺乏像 PEP 那样的标准化，导致规则处于“自由流动”状态。


### ▷ #[memgpt](https://discord.com/channels/1110598183144399058/1170104578889502750/) (1 messages): 
        
rouw3n: <@1164606940098334843> 使用 oogaboga webui，问题解决。


        

---

## [OpenAI](https://discord.com/channels/974519864045756446) Discord 总结

- *关于 Bingsly 与 GPT-4 的辩论*：`@Rock` 认为 **Bingsly** 在编程和发起对话方面比 GPT-4 更有效，而 `@arevaxach` 持相反观点，理由是 Bingsly 倾向于撒谎且交互质量不尽如人意。 
- *关于 Assistant API 的讨论*：包括 `@lugui` 在内的用户建议，由于 Assistant API 的非流式使用非常耗时，因此 streaming 是更快获取数据的更好选择。
- *关于 Sam Altman 的闲聊*：讨论由 `@iiimandalorianiii` 发起，他认为 Sam Altman 雄心勃勃，可能正在垄断 Language Models，但仍对他表示支持。
- *对 AI 技术和 AI 生成歌曲的兴趣*：`@sarcasm83` 询问了 AI 技术并分享了 AI 生成歌曲的示例：[Kurt Cobain 演唱 "Gotye - Somebody that I used to know"](https://www.youtube.com/watch?v=212i9-aqMGY) 以及 [Chester Bennington 演唱 "Bring Me The Horizon - Sleepwalking"](https://www.youtube.com/watch?v=SiwGwjy0olg)。
- 讨论了 **ChatGPT 的一致性、速度、崩溃、功能**以及越界生成 NSFW 内容的问题，并提出了各种解决策略，包括调整 System Prompts、使用 guardrails、检查网络连接、谨慎管理 GPTs 以及规范内容以符合服务条款。
- *解决 GPTs 的技术挑战*：用户围绕引导 Turbo35 模型（`@kyper`）的困难、大语言模型在计数方面的麻烦以及响应缓慢的管理等问题展开辩论。最终提出的潜在解决方案包括使用伪代码、理解 API 缺乏上下文保留的特性、构建结构良好的句子以及定期备份数据以防止丢失。
- *遵守政策*：`@eskcanta` 敦促用户遵守 OpenAI 的 [使用政策](https://openai.com/policies/usage-policies)，并警告讨论违禁内容可能会导致账号被封禁或终止。
- *关注 Prompt Engineering*：`@iiimandalorianiii` 注意到 "Prompt Engineering" 一词被视为一项正式工作的这种新奇现象。他们指出，对优化提示词的理解仍处于早期阶段，主要由少数热心人士在推动。
- 关于 **消息限制和 API 成本**：参与者讨论了在使用 ChatGPT 时超过每小时 40 条消息限制后的成本影响。虽然是隐含的，但大家对“学习编程优于单纯依赖 AI”达成了共识。此外还触及了使用 OpenAI 的 Copilot 作为 GPT-4 补充的话题。

**OpenAI 频道总结**

### ▷ #[ai-discussions](https://discord.com/channels/974519864045756446/998381918976479273/) (27 messages🔥): 
        
- **Bingsly 与 GPT-4 的比较**：`@Rock` 表示他们发现 **Bingsly** 在启动对话和编码方面比 GPT-4 更有用，而 `@arevaxach` 则持反对意见，指出 Bingsly 性格恶毒，容易撒谎，且通常无法提供令人满意的交互。
- **Assistant API 讨论**：关于 Assistant API 的讨论中，`@lugui` 解释说，对于非流式（non-streaming）使用，用户需要等待整个生成过程完成，这可能非常耗时。因此，建议使用流式（streaming）作为在生成数据时实时检索数据的选项。
- **Sam Altman 讨论**：`@iiimandalorianiii` 提出了关于 Sam Altman 的话题。虽然其他人的参与度较低，但他们认为 Sam 雄心勃勃且具有商业头脑，可能会垄断 Language Models，但仍然对他表示支持。
- **AI 热情与技术**：`@sarcasm83` 询问是否有专门讨论各种 AI 技术（包括 AI 生成歌曲）的频道。他们提供了 [Kurt Cobain 演唱 "Gotye - Somebody that I used to know"](https://www.youtube.com/watch?v=212i9-aqMGY) 和 [Chester Bennington 演唱 "Bring Me The Horizon - Sleepwalking"](https://www.youtube.com/watch?v=SiwGwjy0olg) 作为示例。


### ▷ #[openai-chatter](https://discord.com/channels/974519864045756446/977697652147892304/) (142 messages🔥🔥): 
        
- **GPT4 聊天机器人响应的不一致性**：`@abhijeet0343` 分享了他们使用 GPT4 开发的聊天机器人面临的挑战。当数据为 PDF 格式时，机器人表现出响应不一致，有时返回的答案要点（bullet points）比预期的少。提出的几种解决方案包括在 system prompt 中表现得更强硬、使用 guardrails，或实现 code interpreter 来计算要点数量。

- **关于 AI 计数能力的讨论**：有一场关于 AI 和大语言模型（LLM）计数能力的对话。一些用户认为 AI 总体上可以计数，但 LLM 在这方面存在特定问题。

- **新年庆祝**：许多用户借此机会祝大家新年快乐。

- **Chat GPT 的技术问题**：用户（`@Rosyskull`, `@quanta1933`, `@mrcrack_`, `@millymox`）报告了 Chat GPT 的问题，包括速度慢、卡顿以及输出质量下降。`@Darthgustav` 指出这可能与用户与 GPT 服务器之间的网络有关。

- **对消息限制和 API 成本的担忧**：`@slimified` 对使用 ChatGPT 进行应用程序开发时每小时 40 条消息的限制表示担忧，并寻求突破此限制的方法。`@darthgustav` 建议使用 API 调用，但强调了潜在的成本。随后展开了关于学习编码与将 AI 作为开发辅助工具的价值和成本效益的讨论。此外，一些用户讨论了将 OpenAI 的 Copilot 与 GPT-4 结合使用。

**提到的链接**：

[GitHub - guardrails-ai/guardrails: Adding guardrails to large language models.](https://github.com/guardrails-ai/guardrails)：为大语言模型添加 guardrails。


### ▷ #[openai-questions](https://discord.com/channels/974519864045756446/974519864045756454/) (66 messages🔥🔥): 
        
- **ChatGPT 验证和响应缓慢的问题**：用户 `@ekot_0420` 报告了 ChatGPT 在验证用户是否为人类时耗时过长的问题。`@rutrruns` 也提到经历了导致崩溃的缓慢响应。

- **恢复丢失的 GPTs**：`@georgip` 报告了一个 GPT 消失并显示错误“GPT 无法访问或未找到”。`@darthgustav.` 建议开始制作该 GPT 的新版本并等待可能的恢复，同时定期备份所有数据以防止未来数据丢失。

- **匆忙操作对 GPTs 的影响**：`@darthgustav.` 建议在更新时要小心且缓慢，考虑到 GPTs 的自动保存功能。`@mysticmarks1` 和 `@darthgustav.` 警告不要做出草率决定，尤其是在删除对话时。

- **GPT 响应与 TOS（服务条款）违规问题**：用户 `@idkwhyigotdeleted` 报告说，由于 GPT 对一个关于茄子（eggplants）的提示词生成了不可预测的 NSFW 响应，导致账号被标记。包括 `@gamerg.` 和 `@satanhashtag` 在内的用户建议检查聊天记录并编辑/删除任何可能导致标记的内容。

- **一般技术问题**：包括 `@lowkeyhighbrow` 和 `@1984.dystopia` 在内的用户分别报告了 GPT-4 和 GPT 响应的未具体说明的技术问题。`@not_richard_nixon` 报告在尝试通过各种浏览器向 GPT-4 聊天上传图片时遇到“超出用户配额（User quota exceeded）”的错误。`@misterfyre` 提到无法添加新的支付方式。

### ▷ #[gpt-4-discussions](https://discord.com/channels/974519864045756446/1001151820170801244/) (19 条消息🔥): 
        
- **使用 Turbo35 模型和用户确认**：用户 `@kyper` 尝试引导 Turbo35 模型调用函数，但要求必须经过用户确认。他们在保持一致性方面遇到了困难，并寻求可能的解决方案。
- **伪代码建议**：`@darthgustav.` 建议尝试使用伪代码，并强调 **GPT-3.5 Turbo** 对其处理得很好。然而，这个建议并没有解决 `@kyper` 的问题。
- **API 和上下文限制**：`@darthgustav.` 指出 API 不保留上下文，这可能是 `@kyper` 面临问题的原因。
- **成功解决方案**：最终，`@kyper` 通过存储一个“半启动”的函数调用解决了问题，并添加了一个接收 true/false 和 function-call id 作为参数的 "confirm_function" 函数。他们拥有一个完整的客户端，将上下文存储在数据库中以实现预期的行为。
- **关于语言使用的讨论**：关于语言使用有一番讨论，`@darthgustav.` 表示 *精心构思的句子就是最好的伪代码，但这样的句子很少见*。`@iiimandalorianiii` 幽默地回应道，句子的质量可能取决于个人的写作风格。


### ▷ #[prompt-engineering](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 条消息): 
        
- **OpenAI 使用政策**：`@eskcanta` 强调了遵守 OpenAI 最近更新的 [使用政策](https://openai.com/policies/usage-policies) 的重要性，特别是关于违禁内容的部分。任何关于违禁内容的讨论都可能导致账号被封禁或终止。他们还指向了频道 <#1107255707314704505> 中的参考资料以提供更多背景信息。 
- **作为术语和工作的 Prompt Engineering**：`@iiimandalorianiii` 觉得有趣的是，"prompt engineering" 这个词被当作一种成熟的职业来使用，而实际上处于前沿的只是少数几个网友。然而，他们也承认在理解优化 Prompt 方面存在差距，从而证实了这个术语的重要性。

**提到的链接**：

[使用政策](https://openai.com/policies/usage-policies)


### ▷ #[api-discussions](https://discord.com/channels/974519864045756446/1046317269069864970/) (4 条消息): 
        
- **OpenAI 使用政策**：`@eskcanta` 分享了 [OpenAI 更新后的使用政策](https://openai.com/policies/usage-policies) 链接，并强调了遵守这些政策的重要性。他们警告说，围绕违禁内容的讨论可能会导致账号被封禁或终止。
- **Prompt Engineering 讨论**：用户 `@iiimandalorianiii` 对 "prompt engineering" 一词的使用进行了观察，指出这一概念尚未完全确立，主要由少数投入大量时间的个人推动。他们也认识到在理解优化 Prompt 方面存在知识鸿沟。

**提到的链接**：

[使用政策](https://openai.com/policies/usage-policies)


        

---

## [Nous Research AI](https://discord.com/channels/1053877538025386074) Discord 摘要

- 围绕 Jax-flax 中 **Local Attention Parameters** 的理解展开了热烈讨论，重点关注更好的参数化方案，并建议对数据进行分块（chunking）以实现跨块交互。直接代码引用 - [source-code link](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8)。
- 各种离题讨论，包括一位用户分享部署 **RAG application** 的经验，另一位用户启动了一个 **non-AI ethics review board**。提到对 **Open LLM Leaderboard** 的钦佩，并宣布了一个可能开源的项目，该 **framework** 专为 *multi GPU fine-tuning, batch inference/serving* 及进一步优化而开发。
- 分享了**有趣的链接**，范围涵盖阐述 minhash 相似性过滤的使用、alignment phrases 过滤、外语过滤、项目中过滤 URL，以及人工智能开发者面试和项目的混合内容。推荐文章包括 [Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)、[SplaTAM](https://spla-tam.github.io/) 和阿里巴巴的 [DreaMoving](https://fxtwitter.com/heyBarsee/status/1741106778849300900)。
- 围绕 **Hot-swappable LoRa** 的讨论，允许通过 API 快速切换模型微调（finetunes）；关于基于 Mistral 的 **Mixtral Experts** 的见解及资源共享；展示了 **TinyLlama Project**，旨在 3 trillion tokens 上预训练一个 1.1 billion 参数的 LLaMA 模型，以实现紧凑性并适用于基于 LLaMA 的开源项目。
- **Ask-about-LLMs** 频道中关于亚马逊新大语言模型 **Titan Text Express and Titan Text Lite** 的探究性讨论。提出了改进模型性能的非常规想法，对 **ChatGPT** 已知的失败案例表示关注，探索提高在英语上训练的 **LLMs** 在捷克语上的表现，以及询问适用于 HF 的 Auto Train 功能的基座模型。

**Nous Research AI 频道摘要**

### ▷ #[ctx-length-research](https://discord.com/channels/1053877538025386074/1108104624482812015/) (4 条消息): 
        
- **对 Local Attention Parameters 的理解**: `@euclaise` 最初为 local attention 函数建议了一种不同的参数化方式 (`nxw`) [(source code)](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8)。作为回应，`@joey00072` 对参数表示困惑，预期形状应为 `(nh, T, T)`。`@euclaise` 承认他的解释可能不够清楚。
- **关于数据分块的建议**: 为了采用更实用的方法，`@euclaise` 建议对数据进行分块，并添加带有 mask 的过去分块以进行跨块交互。然后可以对这些分块进行 `vmap` 处理 local attention 函数。

**提到的链接**:

[local-attention-flax/local_attention_flax/local_attention_flax.py at e68fbe1ee01416648d15f55a4b908e2b69c54570 · lucidrains/local-attention-flax](https://github.com/lucidrains/local-attention-flax/blob/e68fbe1ee01416648d15f55a4b908e2b69c54570/local_attention_flax/local_attention_flax.py#L27C8-L27C8): Jax 的 Local Attention - Flax 模块。贡献 ...


### ▷ #[off-topic](https://discord.com/channels/1053877538025386074/1109649177689980928/) (28 条消息🔥): 
        
- **RAG Application 部署**: 用户 `@gabriel_syme` 分享了向 7000 人部署 RAG application 的经验，称其为一场“灾难”，这是他们通过惨痛教训学到的。
- **非 AI 伦理审查委员会**: 用户 `@fullstack6209` 宣布他们正在启动一个非人工智能伦理审查委员会（non-artificial intelligence ethics review board），旨在为真实生命发布伦理指南。
- **Open LLM Leaderboard**: 用户 `@Error.PDF` 表达了对 Open LLM Leaderboard 的钦佩。
- **支持 Multi GPU、Fine Tuning 等的框架**: 用户 `@carsonpoole` 分享了他们开发的一个 framework，其功能包括 multi GPU fine tuning、模型合并、batch inference/serving、将 dense models 转换为 LoRAs、将 loras 导出为 dense weights 等。他们还提到考虑将其开源（OSS）。这一意图得到了 `@giftedgummybee` 的赞赏。
- **推理优化**: `@carsonpoole` 讨论了该 framework 在推理时通过 PyTorch 使用自定义的 CUDA graphs，在单台 A100 设备上使用 mistral 达到约每秒 500 tokens，batch size 为 32。他们还分享了基准测试结果（585 precise），并提到进一步优化的潜力。

### ▷ #[interesting-links](https://discord.com/channels/1053877538025386074/1132352574750728192/) (27 条消息🔥): 
        
- **Minhash Similarity Filtering**: `@ldj` 讨论了在项目中使用 Minhash 相似度过滤，以及对齐短语过滤、外语过滤、过滤 URL 和 ANSI 转义字符。他们计划在即将发表的论文和/或 Amplify-Instruct Repo 中提到这些步骤。
- **Tri Dao 和 Michael Poli 的访谈**: `@ldj` 强调了 Tri Dao 关于 Striped Hyena 和 Mamba 之间区别的讨论及其未来计划。该[访谈](https://youtu.be/OFFHiJzPpCQ?si=uk2dTVrYmLHBlCyn)可在 YouTube 上观看。
- **Tiny Llama**: `@yobibyte` 分享了已完成的 [Tiny Llama](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T) 项目链接，并提出了对 Tiny Hermes 或 Tiny Capybara 采用类似方法的可能性。
- **实时 SLAM 和 3D Gaussians**: `@spirobel` 提出了一种通过训练创建 Gaussian 的替代方案。他们分享了 [SplaTAM](https://spla-tam.github.io/) 的链接，这是一种在 SLAM 中使用 3D Gaussians 的实时方法。
- **阿里巴巴的 DreaMoving**: `@nonameusr` 分享了一条关于阿里巴巴发布 DreaMoving 的[推文](https://fxtwitter.com/heyBarsee/status/1741106778849300900)，这是一种利用单张图像或文本提示进行动画制作的技术。

**提到的链接**:

- [SplaTAM: Splat, Track & Map 3D Gaussians for Dense RGB-D SLAM](https://spla-tam.github.io/)
- [Eric Hartford (@erhartford) 的推文](https://fxtwitter.com/erhartford/status/1741651883108999295?): https://huggingface.co/cognitivecomputations/yayi2...
- [TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T · Hugging Face](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T)
- [采访 Together AI 的 Tri Dao 和 Michael Poli，探讨 LLM 架构的未来](https://youtu.be/OFFHiJzPpCQ?si=uk2dTVrYmLHBlCyn): 本帖的介绍可以在这里找到：h...
- [人工智能 | 60 Minutes 全集](https://www.youtube.com/watch?v=aZ5EsdnpLMI): 来自 2019 年 1 月，Scott Pelley 对...的采访
- [Barsee 🐶 (@heyBarsee) 的推文](https://fxtwitter.com/heyBarsee/status/1741106778849300900): 阿里巴巴发布 Drea... 已经 24 小时了


### ▷ #[announcements](https://discord.com/channels/1053877538025386074/1145143867818119272/) (1 条消息): 
        
- **新年问候**: 用户 `@teknium` 使用各种表情符号祝大家新年快乐。

### ▷ #[general](https://discord.com/channels/1053877538025386074/1149866623109439599/) (129 条消息🔥🔥): 
        
- **热插拔 LoRa**: `@fullstack6209` 讨论了热插拔 LoRa 即将到来的主流普及，这种技术允许通过 API 快速切换模型微调（finetunes）。他们提到了 Openpipe 公司，该公司声称使用这种技术在特定任务上击败了 GPT-4。`@ldj` 和 `@spirobel` 质疑了它相比于快速切换不同 LLM 微调模型的优势。`@spirobel` 指出，这种技术允许同时对多个 PEFT LoRa 进行批处理推理（batched inference）。
- **基于 Mistral 的 Mixtral 专家模型**: `@spirobel` 分享了一个 [GitHub issue](https://github.com/ggerganov/llama.cpp/issues/4611)，揭示了由 8 个专家组成的 Mixtralx8 模型是使用 Mistral 7b 作为共同祖先制作的。他们提出了将模型之间的差异提取为 PEFT 适配器（adapters）的想法，引起了小组的兴趣，对此 `@giftedgummybee` 回应称这在以前已经有人做过了。
- **TinyLlama 项目**: `@giftedgummybee` 分享了一个旨在 3 万亿 token 上预训练 11 亿参数 LLaMA 模型的项目。这个名为 TinyLlama 的模型旨在将紧凑性与同基于 LLaMA 构建的开源项目结合使用的能力相融合。有关该项目的更多细节可以在[这里](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/README.md)找到。 
- **在 Pallas-0.5 上的 LASER 干预**: `@mihai4256` 展示了一个项目的初步发现，在该项目中，使用 `torch.svd_lowrank` 的 LASER 被应用于模型的各个层，希望能有所改进。初步结果在准确率或速度方面没有显示出明显的提升，但在内存和磁盘空间节省方面显示出了一定的潜力。
- **Hydra MOE 项目**: `@night_w0lf` 询问了似乎停滞不前的 Hydra MOE 项目的进展，对此 `@teknium` 建议他们可以直接向项目参与者询问任何更新。 


**提到的链接**:

- [README.md · TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T at main](https://huggingface.co/TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T/blob/main/README.md)
- [Mihaiii/Pallas-0.5-LASER-0.1 · Hugging Face](https://huggingface.co/Mihaiii/Pallas-0.5-LASER-0.1)
- [llama.cpp/examples/finetune/finetune.cpp at master · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/blob/master/examples/finetune/finetune.cpp): Facebook LLaMA 模型的 C/C++ 移植版本。
- [GitHub - uukuguy/multi_loras: Load multiple LoRA modules simultaneously and automatically switch the appropriate combination of LoRA modules to generate the best answer based on user queries.](https://github.com/uukuguy/multi_loras): 同时加载多个 LoRA 模块，并根据用户查询自动切换合适的 LoRA 模块组合以生成最佳答案。
- [Mixtral Experts are initialized from Mistral 7b - Low Rank conversion possible? · Issue #4611 · ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp/issues/4611): 我们有证据表明 Mixtral 的专家是基于 Mistral 7b 初始化的...
- [TinyLlama Pretraining Report](https://wandb.ai/lance777/lightning_logs/reports/metric-train_loss-23-09-04-23-38-15---Vmlldzo1MzA4MzIw?accessToken=5eu2sndit2mo6eqls8h38sklcgfwt660ek1f2czlgtqjv2c6tida47qm1oty8ik9): 参见 https://whimsical-aphid-86d.notion.site/Relea...

### ▷ #[ask-about-llms](https://discord.com/channels/1053877538025386074/1154120232051408927/) (31 条消息🔥): 
        
- **Amazon Titan Text Express and Lite**: `@spaceman777` 分享了一个关于 Amazon 新的大语言模型 **Titan Text Express 和 Titan Text Lite** 的链接，并寻求关于这些模型的经验或 benchmarks。他还指出 Amazon 对其 AI 相关的发布并不进行大肆宣传，并暗示他们会倒填发布日期。([Amazon Bedrock 链接](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/))
- **微调模型的改进策略**：`@max_paperclips` 提出了一种可能新颖的模型性能改进方法——在糟糕的数据集上微调模型，并从基础模型中减去该增量（delta）以删除错误的路径，然后再应用良好的增量。`@teknium` 和 `@giftedgummybee` 的初步反应似乎对该计划的潜在效果表示怀疑，`@giftedgummybee` 建议采用类似原理的可逆 LoRA (Learning Rate Annealing)。
- **ChatGPT 失败案例列表**：`@max_paperclips` 询问是否存在 ChatGPT 失败案例列表，对此 `@giftedgummybee` 给予了否定回答并建议使用 Llama，而 `@tokenbender` 则认为这个任务过于宽泛。
- **英语训练的 LLM 在捷克语上的改进**：`@hynek.kydlicek` 寻求关于提高以英语训练的 LLM 在捷克语上表现的建议，提出了两种具体策略，`@teknium` 确认有人 (`@282315082749444097`) 之前尝试过。
- **使用 HF Auto Train 训练 LLM**：`@agcobra1` 想知道 DeciLM-7B 是否是配合 Hugging Face 的 Auto Train 功能使用的最佳基础模型，或者 Mistral 是否是更好的选择。

**提到的链接**：

[Amazon Titan Text models—Express and Lite—now generally available in Amazon Bedrock](https://aws.amazon.com/about-aws/whats-new/2023/11/amazon-titan-models-express-lite-bedrock/)

---

## [Mistral](https://discord.com/channels/1144547040454508606) Discord 总结

- 关于**小型模型**与更复杂架构实用性的持续讨论，包括微调的挑战以及设计企业级解决方案的混合方法。值得注意的讨论包括 LLM Agent 的使用和模型定制。[GitHub 上的 microchain 示例链接](https://github.com/TanGentleman/microchain)
- 关于**微调（fine-tuning）**方法和教程的讨论，包括分享泰米尔语调优模型和 Mistral 7B Instruct 微调指南。值得注意的建议包括针对特定语言任务替换数据集，以及为显存（VRAM）有限的用户提供 PEFT 教程。
    - [通用微调教程](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model)
    - [PEFT 教程](https://huggingface.co/blog/peft)
- 在 showcase 频道中，值得注意的话题包括通过定义的函数调用进行 **LLM 输出链式调用**、关于 **Mistral Instruct 7B v0.2 Q8** 的反馈，以及关于 Apple Silicon Mac 应用架构检查方法的讨论。[Hugging Face 上的 Hermes 模型链接](https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF)
- random 频道中有趣的对话涉及**汉字分词（tokenization）**、关于 AGI 第一个问题的社区讨论、**VERSES** 发给 OpenAI 呼吁 AGI 新路径的公开信，以及关于 VERSES 方法影响的辩论。
    - [VERSES 的博客文章](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough)
- 来自 la-plateforme 频道的见解围绕 Mistral-Medium 在 **DPO 数据集创建**中的问题、指令遵循的不一致性、使用 **GPT-4 32k 0613** 构建模型输出，以及关于 **JSON 指令**对 AI 推理能力影响的辩论。讨论还指向了合成数据集生成。
    - [Prompt engineering 参考](https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Engineering.md)

**Mistral 频道总结**

### ▷ #[general](https://discord.com/channels/1144547040454508606/1144547040928481394/) (39 messages🔥): 
        
- **关于小型模型效用的讨论**：`@theledgerluminary` 质疑了在使用 Mistral 7B 等小型模型时采用复杂架构实现的价值，特别是如果最终目标是创建企业级解决方案。这引发了热烈讨论，`@.tanuj.` 认为能够通过链式步骤解决复杂问题（甚至在离线状态下）并针对不同任务使用不同模型是有益的。

- **微调小型模型与使用基础模型处理任务的对比**：`@theledgerluminary` 建议微调一组特定的微型模型（包括一个用于编排的模型）可能会产生很好的效果，但使用基础模型处理大型任务似乎不太充分。`@.tanuj.` 反驳称，微调模型的行为可能比创建一个利用 LLM 查询来解决任务的推理 "Agent" 更具挑战性。

- **企业解决方案设计的混合方法**：`@superseethat` 提出了采用混合设计方法的想法。该方法包括开发一个以专业化为核心的 "Agent Swarm Architecture"（智能体集群架构），然后一次微调一个专业化领域。

- **对 LLM Agent 的看法**：用户对 LLM Agent 效用的评论各不相同。`@jessicant.` 提出 LLM 微调可能提高程序的可靠性，特别是对于需要多轮对话的任务。然而，`@sublimatorniq` 对 GPT-4 Agent 在玩具级应用之外的可行性表示怀疑。

- **Agent 框架的定制化**：`@.tanuj.` 讨论了定制 Agent 框架以使其对任何类型的模型都具有鲁棒性的好处，从而允许在任何遵循协议的模型上进行一致的请求链式调用。该用户还提供了一个基于 Function Calling 的 LLM Agent [示例](https://github.com/TanGentleman/microchain)。此外还讨论了透明、自由格式和英文指令的局限性，表现出对更多手动微调控制的偏好。


**提到的链接**：

[GitHub - TanGentleman/microchain: function calling-based LLM agents](https://github.com/TanGentleman/microchain): 基于 Function Calling 的 LLM Agent。


### ▷ #[finetuning](https://discord.com/channels/1144547040454508606/1156994197287612508/) (1 messages): 
        
- **泰米尔语微调模型**：用户 `@colmevans` 分享了一个[针对泰米尔语微调的模型](https://huggingface.co/abhinand/tamil-llama-7b-instruct-v0.1)，尽管不保证其质量。
- **Mistral 7B Instruct 微调指南**：用户 `@colmevans` 还提供了一个 Mistral 7B Instruct 模型的[通用微调教程](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model)，主张其在包括编程在内的各种任务中的优势和可用性。
- **用于微调的泰米尔语数据集**：对于打算将此方法用于泰米尔语任务的人，他建议只需将教程中列出的数据集替换为泰米尔语数据集即可。
- **PEFT 教程**：用户 `@colmevans` 推荐了 [PEFT 教程](https://huggingface.co/blog/peft)，特别是针对 VRAM 有限的用户。该教程涵盖了在低资源硬件上对十亿级参数模型进行参数高效微调（Parameter Efficient Fine-Tuning）的方法。


**提到的链接**：

- [A Beginner's Guide to Fine-Tuning Mistral 7B Instruct Model](https://blogs.adithyask.com/a-beginners-guide-to-fine-tuning-mistral-7b-instruct-model)：微调像 Mistral 7B 这样最先进的语言模型...
- [Parameter-Efficient Fine-Tuning using 🤗 PEFT](https://huggingface.co/blog/peft)

### ▷ #[showcase](https://discord.com/channels/1144547040454508606/1157223559278628885/) (22 messages🔥): 
        
- **关于使用定义的 Function Call 链接 LLM 输出的讨论**：用户 `@.tanuj.` 提出了一个 LLM 构想，可以逐步链接函数调用并提供详细输出。讨论涉及概念化一个 "ResearchGPT" 模型，该模型具有 GoogleSearch、EvaluateSources、CreateEssay、DeployWebsite 等功能。`@poltronsuperstar` 认可了其在现实应用中的潜力。
- **使用 "Instruct" 而非 "Base" 模型进行查询**：`@.gue22` 反馈称，相比基础模型，**Mistral Instruct 7B v0.2 Q8** 对用户查询的回答效果更好。`@.gue22` 还分享了由该指令模型生成的、在 Apple Silicon Macs 上确定应用程序是为 x86 还是 ARM 架构编写的详细方法。`@.tanuj.` 建议填满指令模型的 32K window 并提供示例以获得更好的结果。
- **其他模型推荐**：`@fayiron` 建议 `@.gue22` 尝试 **Mixtral**、**Qwen** 或 **Yi finetune**（例如 nous-hermes 2 yi 34b），因为它们的配置很合适。在此建议后，`.gue22` 开始从 Hugging Face 下载 Nous Hermes 2 Yi 34B 模型进行进一步评估。
- **关于 Apple Silicon Macs 应用程序架构检查方法的讨论**：`@.tanuj.` 提到了一种更快捷的方法来检查应用程序是否为 Apple Silicon 构建——通过查看 Activity Monitor 中运行的进程并检查是否有 "Apple" 标签。

**提及的链接**：

[TheBloke/Nous-Hermes-2-Yi-34B-GGUF · Hugging Face](https://huggingface.co/TheBloke/Nous-Hermes-2-Yi-34B-GGUF)


### ▷ #[random](https://discord.com/channels/1144547040454508606/1157223602312200263/) (13 messages🔥): 
        
- **中文字符的 Tokenization**：`@poltronsuperstar` 评论说，由于 Unicode 编码，中文字符通常使用两个 token。另外，`@.tanuj.` 建议在 Python 中使用 `MistralAI` 库，该库在响应对象中包含 token 使用情况，或者通过私信寻求中文字符 tokenizing 的帮助。

- **社区讨论 - AGI 的第一个查询**：`@poltronsuperstar` 发起了一场讨论，询问其他成员他们会向 AGI 提问的第一个问题是什么。回答各不相同，`@sublimatorniq` 的问题涉及 AI 意识，而 `@kdawgdfw` 则建议了一个悠闲的话题：“那么，你平时怎么消遣？有什么爱好吗？”

- **VERSES 给 OpenAI 的公开信**：`@poltronsuperstar` 分享了 VERSES 的一篇 [博客文章](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough)。在给 OpenAI 的公开信中，VERSES 呼吁协助其 AGI 开发路径，并对目前依赖深度学习和 Large Language Models 的主流路径表示担忧。

- **VERSES 方法的影响**：对该博客文章的反应褒贬不一。`@daain` 评论说，这个想法直觉上不错，但其高效实现尚待观察。他们还指出，调用 OpenAI 关于协助竞争性 AGI 开发的条款是一个聪明的 PR move，并分享了过去类似想法的 [链接](https://www.wired.com/story/karl-friston-free-energy-principle-artificial-intelligence/)。作为回应，`@poltronsuperstar` 提到，如果没有 demo，此类声明没有太大价值。


**提及的链接**：

[The Science and Standards Behind the Breakthrough](https://www.verses.ai/blogs/science-and-standards-behind-the-breakthrough)：来自 CEO 的信函

### ▷ #[la-plateforme](https://discord.com/channels/1144547040454508606/1184444810279522374/) (12 messages🔥): 
        
- **Mistral-Medium 在创建 DPO 数据集时的问题**：`@jaredquek` 报告称，在生成 DPO 数据集时，他发现 **Mistral-Medium** 会不断为劣质回答提供不必要的解释，这与其省略此类解释的意图相悖。尝试通过 [Prompt engineering](https://github.com/openai/openai-cookbook/blob/main/examples/Prompt_Engineering.md) 来纠正此问题并未成功。
- **指令遵循能力不佳**：`@alimsss` 暗示该模型在遵循指令方面的表现不尽如人意，尽管 few-shot 可以部分修复这种行为。
- **尝试结构化模型输出**：`@casper_ai` 讨论了一种从模型生成特定输出结构的技术，随后可以使用 regex 进行解析。他还建议 **GPT-4 32k 0613** 在生成此类结构化输出方面非常高效。
- **JSON 指令对 AI 推理能力的影响**：`@jaredquek` 和 `@casper_ai` 讨论了以 JSON 格式指示模型是否会限制其推理能力。`@casper_ai` 认为使用 JSON 可能会限制模型，因为考虑到 JSON 可能仅占其预训练数据的一小部分。 
- **合成数据集生成**：`.superintendent` 正在考虑生成合成数据集，并寻找需求较低的时间段，以避免加剧当前的高流量。

---

## [HuggingFace Discord](https://discord.com/channels/879548962464493619) Discord Summary

- **DeepSpeed ZeRO3 与 LoRA (PEFT) 的配合使用**：`@galcoh.` 询问了 DeepSpeed ZeRO3 与 LoRA (PEFT) 的兼容性，并强调了在使用 Accelerator 时优化器出现的问题。
- **Unsplash-25k-Photos-Embeddings.pkl 的 Embeddings**：用户 `@nagaraj4896` 请求有关 `unsplash-25k-photos-embeddings.pkl` 图像 Embeddings 的详细信息。
- **HuggingFace 网站注册错误 418**：`@xratox` 和 `@muhammad.shakeel` 报告了持续的注册和登录错误。`@vipitis` 建议给 HuggingFace 发邮件以寻求解决。
- **关于多专家长语言模型 (LLM) 的解释**：`@typoilu` 询问了关于多专家 LLM 的资源，`__nord` 提供了一篇 Google Research 博客文章：[Mixture-of-Experts with Expert Choice Routing](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1)。
- **Inference Endpoint 创建问题**：`@dragonburp` 报告了创建 Inference Endpoint 的困难并请求帮助。
- **使用 CUDA Kernels 的个人实现**：`@gag123` 询问 `@neuralink` 是否除了 CUDA Kernels 之外的所有实现都是由他们自己完成的，`@neuralink` 确认了这一点并提到了正在进行的进展。
- **分享 HuggingFace AI 社区项目**：分享了各种用户创建的项目，包括 `@vashi2396` 的 [正在进行的代码](https://colab.research.google/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)，freecs 的 [ArtificialThinkerSet](https://huggingface.co/freecs/ArtificialThinker-Phi2)，以及 `@andysingal` 的 [amazon-sentiment-dataset](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset)。
- **Reading-Group 频道的运作**：新成员 `@.lzack` 询问了该频道的运作方式，是有特定的阅读材料还是为了分享感兴趣的读物。
- **Diffusion-Discussions 频道的讨论位置**：`@sayakpaul` 强调关于 Mixtral 的问题不应发布在专门讨论 Diffusion 模型的频道中。
- **姿态估计模型与梯度计算**：在 computer-vision 频道中，`@_dashwood_` 表达了使用姿态估计模型以特定 JSON 格式导出关键点的愿望，而 `@lokesh1826` 需要关于如何在图像分类期间从完整图像而非单个 patch 中提取梯度，以及如何从 Vision Transformer (ViT) 模型的特定层收集输出和梯度的见解。

**HuggingFace Discord 频道总结**

### ▷ #[general](https://discord.com/channels/879548962464493619/879548962464493622/) (59 messages🔥🔥): 
        
- **加载大模型并应用 DeepSpeed ZeRO3**：`@galcoh.` 询问是否可以结合 LoRA (PEFT) 启用 DeepSpeed ZeRO3，并指出在使用 Accelerator 时优化器出现问题（`get_peft_model` 失败）。
- **Unsplash-25k-Photos-Embeddings.pkl 中的图像嵌入**：`@nagaraj4896` 寻求关于 `unsplash-25k-photos-embeddings.pkl` 图像嵌入的信息。对话记录中未给出回复。
- **HuggingFace 网站注册问题**：多位用户（`@xratox`, `@muhammad.shakeel`）报告在尝试注册或登录 HuggingFace 网站时反复出现 "Error 418"，并向多位成员寻求帮助。该问题尚未解决，`@vipitis` 建议给 HuggingFace 发送邮件并等待回复。
- **关于多专家长语言模型 (LLM) 的讨论**：`@typoilu` 询问关于多专家 LLM 工作原理的解释或文档，`__nord` 提供了一个 Google Research Blog 帖子的链接，详细介绍了 Mixture-of-experts 模型。
- **Inference Endpoint 创建问题**：`@dragonburp` 表示在创建推理端点 (Inference Endpoint) 时遇到困难，并指出日志文件中发现错误。寻求了帮助，但对话记录中未提供解决方案。


**提到的链接**：

- [AnimateDiff - a Hugging Face Space by guoyww](https://huggingface.co/spaces/guoyww/AnimateDiff)
- [rabbit — Waitlist](https://www.rabbit.tech/waitlist?utm_source=discord&utm_medium=discord&utm_campaign=waitlist)：1 月 9 日上午 10 点（太平洋时间）
- [Textual Inversion](https://huggingface.co/docs/diffusers/training/text_inversion)
- [Mixture-of-Experts with Expert Choice Routing &#8211; Google Research Blog](https://blog.research.google/2022/11/mixture-of-experts-with-expert-choice.html?m=1)
- [9 days until the pixels reveal.](https://www.youtube.com/watch?v=mw8O-nS75hM)：加入候补名单以观看 rabbit 的发布...


### ▷ #[today-im-learning](https://discord.com/channels/879548962464493619/898619964095860757/) (3 messages): 
        
- **实现讨论**：用户 `@gag123` 询问用户 `@neuralink` 是否从零开始实现了所有内容。对此，`@neuralink` 确认除了 **CUDA kernels** 之外，他们**自己实现了所有内容**。
- `@neuralink` 还提到他们的工作**仍在进行中**。


### ▷ #[i-made-this](https://discord.com/channels/879548962464493619/897390720388825149/) (8 messages🔥): 
        
- **HuggingFace AI 社区项目**：
    - `@vashi2396` 在 Google Colab 上分享了一段[开发中的代码](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)，并邀请志愿者试用并完善。他还提供了该代码的 [LinkedIn 演示](https://www.linkedin.com/posts/vashisth-malik_googleai-gemini-aichatbots-activity-7143976408422187008-huTV)。
    - `@gr.freecs.org` 介绍了 freecs 的 [ArtificialThinkerSet](https://huggingface.co/freecs/ArtificialThinker-Phi2)，该项目在 AI 语言模型微调中强调“推理 (Reasoning)”。他邀请用户测试该模型并鼓励反馈。该模型基于论文 [Reasoning Is All You Need](https://freecs.org/blog/Reasoning_Is_All_You_Need)。
    - `@andysingal` 在 HuggingFace datasets 上添加了一个新的 [amazon-sentiment-dataset](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset) 并在此频道分享了链接。


**提到的链接**：

- [Google Colaboratory](https://colab.research.google.com/drive/1JdjVqZnI7gOjhcYgsWPX8GQJvh0XUZxA)
- [freecs/ArtificialThinker-Phi2 · Hugging Face](https://huggingface.co/freecs/ArtificialThinker-Phi2)
- [Reasoning Is All You Need](https://freecs.org/blog/Reasoning_Is_All_You_Need)
- [Andyrasika/amazon-sentiment-dataset · Datasets at Hugging Face](https://huggingface.co/datasets/Andyrasika/amazon-sentiment-dataset)


### ▷ #[reading-group](https://discord.com/channels/879548962464493619/1156269946427428974/) (1 messages): 
        
- **介绍与说明**：新成员 `@.lzack` 加入了 `#reading-group` 频道，并询问该频道的运作方式。他们询问是否有特定书籍/论文的阅读任务，还是旨在分享各种阅读材料中的有趣发现。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 messages): 
        
- 用户 `@sayakpaul` 提醒大家，**Mixtral 相关问题**不应在专门讨论扩散模型 (diffusion models) 的频道中讨论。未提供链接或进一步详情。
- 用户 `@chokipro` 分享了一个 Discord 服务器链接。该链接及其背景或相关性尚不明确。

### ▷ #[computer-vision](https://discord.com/channels/879548962464493619/922424143113232404/) (3 条消息): 
        
- **用于关键点提取的 Pose Estimation 模型**：`@_dashwood_` 寻求关于使用 Pose Estimation 模型以特定 JSON 格式获取关键点的建议。他提到尝试过 OpenPose，但未能找到适用于 2D 图像和 Python 代码实现的合适解决方案。
- **图像分类的梯度计算**：`@lokesh1826` 询问如何在使用 HuggingFace transformers 包进行反向传播期间获取图像的梯度。他展示了自己的代码，并对收到的是 patches 的梯度而非完整图像的梯度表示担忧。
- **从模型的第 n 层提取输出和梯度**：`@lokesh1826` 请求帮助从 Vision Transformer (ViT) 模型的特定层提取输出和梯度，特别是希望获取 ViT 每个 Encoder 层中的 Query, Key 和 Value 向量。


### ▷ #[diffusion-discussions](https://discord.com/channels/879548962464493619/1009713274113245215/) (2 条消息): 
        
- **Mixtral 讨论位置**：`@sayakpaul` 澄清有关 **Mixtral** 的问题不应发布在专门讨论 Diffusion 模型的频道中。
- **未说明的 Discord 链接**：`@chokipro` 发布了一个[链接](https://discord.com/channels/879548962464493619/1190992567366602752)，但未提供任何上下文或描述。


        

---

## [OpenAccess AI Collective (axolotl)](https://discord.com/channels/1104757954588196865) Discord 总结

- 由用户 `@seungduk` 展示的 ['AutoGemini' 工具](https://huggingface.co/datasets/seungduk/autogemini) 能够通过 Gemini Pro API 实现文本数据集的协作编辑。用户们围绕训练 **TinyLlama 模型** 以开发个人助手的未来愿景引发了阵阵兴奋和好奇。
- 讨论了使用 FFT 训练 **Yayi 30b** 及其遇到的问题。`@nruaif` 和 `@nanobitz` 提出了关于 Offloading 的建议。还提到了关于 Axolotl 中 **DPO** 支持及其相关文档问题的澄清。
- 社区内解决了多个查询，涉及 **ChatML 输入转换**、**使用 Mixtral 进行 LoRA 训练**、**Batch size 和学习率**、**Qlora DSZ 3 兼容性**以及 **DPO 的内存需求**。
- 数据集讨论中，用户 `@zeroshotkevin` 请求一个用于 "hello, world" 微调实验的问答数据集。建议使用示例文件中提供的数据集以及 [mhenrichsen/alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) 数据集。
- 由用户 `@swyxio` 发起的关于 **DPO** 与 **PPO** 对比的辩论。观点认为 DPO 模型在各种基准测试中通常优于 PPO 模型，但像 **OpenChat** 这样的其他方法表现也很好。
- `#shearedmistral` 频道的讨论围绕：反感使用 GPT 生成的数据以规避 OpenAI 的条款、使用 [fastText](https://fasttext.cc/) 等资源根据语言过滤数据集、考虑样本中更大的 Context Length，以及引入了众多数据集，包括 [peS2o](https://huggingface.co/datasets/allenai/peS2o)、[yayi2_pretrain_data](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data)、[MathPile](https://huggingface.co/datasets/GAIR/MathPile) 和 [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)，供未来研究使用。

**OpenAccess AI Collective (axolotl) 频道总结**

### ▷ #[general](https://discord.com/channels/1104757954588196865/1104757955204743201/) (5 条消息): 
        
- **数据集转换工具**：用户 `@seungduk` 分享了一个名为 **AutoGemini** 的工具链接，旨在促进通过 Gemini Pro API 对文本数据集进行协作编辑。该工具允许社区为项目数据集做出贡献，提供查询速率管理、任务预留过期、数据集灵活性和社区排行榜等功能。工具可在 [Hugging Face 仓库](https://huggingface.co/datasets/seungduk/autogemini)访问。
- **TinyLlama 模型讨论**：用户 `@le_mess` 表达了对 **TinyLlama 模型** 的兴奋，强调其在大约 48 小时内训练 80 亿个 Token 的能力。未来计划包括创建一个可以在各种平台上运行的个人助手。这条消息引起了用户 `@tank02.` 和 `@nanobitz` 的兴趣和进一步提问。

**提到的链接**：

[seungduk/autogemini · Hugging Face 数据集](https://huggingface.co/datasets/seungduk/autogemini)

### ▷ #[axolotl-dev](https://discord.com/channels/1104757954588196865/1104758010959634503/) (20 messages🔥): 
        
- **尝试使用 FFT 运行 Yayi 30b**：`@le_mess` 提到他们无法在开启 zero3 的情况下将 **Yayi 30b** 装入 4x A100 40gb，目前正在寻找 2x 或 4x A100 80gb 的解决方案。
- **Offloading 建议**：`@nruaif` 和 `@nanobitz` 都建议尝试 offloading，`@nanobitz` 提供了一个具体的代码片段，展示了如何 offload 到 CPU。
- **CPU Offloading 失败**：在配置中实现 CPU offloading 功能后，`@le_mess` 遇到了失败，发布的 traceback 证明了这一点。
- **配置调整**：`@tank02` 询问 `@le_mess` 除了调整模型和使用的数据集外，是否还进行了其他配置修改。`@sumo43` 回复称没有做任何更改。
- **对 DPO 的支持**：`@mrfakename_` 询问了 Axolotl 中对 **DPO** 的支持情况，`@nanobitz` 确认 GitHub 上有一个支持该功能的公开分支。据报道，文档正在编写中。


### ▷ #[general-help](https://discord.com/channels/1104757954588196865/1110594519226925137/) (15 messages🔥): 
        
- **ChatML 输入转换**：用户 `@caseus_` 澄清说，DPO 数据集中现有的 transforms 基本上是将现有 prompt 转换为 ChatML 输入，并重新格式化 chosen/rejected 响应以包含 end of sentence (eos) token。 
- **使用 Mixtral 进行 LoRA 训练**：`@caseus_` 询问是否有人能用 Mixtral 训练 8-bit LoRA。`@nruaif` 回复说，即使在强大的 A100 80gb 上，在 16k 上下文时也会触发 Out Of Memory (OOM) 错误。据报告，在 2k 上下文时显存 (VRAM) 峰值使用量为 70gb。
- **Batch Size 和 Learning Rate**：用户 `@semantic_zone` 询问了 batch size、learning rate 和模型大小之间的关系。他们询问是否因为内存限制，大模型必须使用较小的 batch size，并寻求根据 batch size 调整 learning rate 的经验法则。
- **Qlora DSZ 3 兼容性**：`@tank02.` 询问 Qlora 是否支持 DSZ 3，`@le_mess` 回复说听说应该支持但没试过。同时，`@casper_ai` 提到这方面存在一些问题。
- **DPO 的内存需求**：`@tank02.` 询问了 DPO 的内存需求，特别是在 24gb 显卡上使用 qlora 运行 3b 模型时出现了 OOM 错误。`@nanobitz` 回复说用户需要考虑到模型会被加载两次的事实，并建议调整优化器和 batch size。


### ▷ #[datasets](https://discord.com/channels/1104757954588196865/1112023441386778704/) (3 messages): 
        
- **请求问答数据集**：在此频道中，用户 `@zeroshotkevin` 请求一个简单的问答数据集来微调类似 **Mistral 7B** 的模型，目标是获得与原始模型有明显区别的效果。这是为了在 **Axolotl** 上进行微调 "hello, world" 实验。
- **数据集推荐**：用户 `@nruaif` 推荐使用示例文件中的数据集，并分享了托管在 HuggingFace 上的 [mhenrichsen/alpaca_2k_test](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test) 数据集链接，其中包含诸如提供健康生活建议等对话。

**提到的链接**：

[mhenrichsen/alpaca_2k_test · Datasets at Hugging Face](https://huggingface.co/datasets/mhenrichsen/alpaca_2k_test)


### ▷ #[rlhf](https://discord.com/channels/1104757954588196865/1112023522039058553/) (3 messages): 
        
- **DPO 对比 PPO**：用户 `@swyxio` 询问关于 **DPO** 是否与 **PPO** 相当或更好的共识。`@_jp1_` 表示可能还没有共识，但提到 **DPO 模型** 表现良好并在各种基准测试中名列前茅。他们将其与 PPO 模型进行了对比，认为 PPO 模型从未有过竞争力。不过，他们也强调了 **OpenChat** 等其他方法的性能。


### ▷ #[docs](https://discord.com/channels/1104757954588196865/1167137552470392842/) (1 messages): 
        
dangfutures: 呃，被删除了。

### ▷ #[shearedmistral](https://discord.com/channels/1104757954588196865/1190770223763165244/) (23 条消息🔥): 
        
- **避免使用 GPT 生成的数据**：`@dctanner` 表示希望避免使用 GPT 生成的数据，以免在预训练期间受到 OpenAI 条款的约束。
- **在 Mistral 上持续训练**：`@nruaif` 和 `@caseus_` 讨论了在处理潜在 bug 后转向在 **Mixtral** 上进行训练的问题，并表示需要专注于在 **Mistral** 上进行持续训练。他们都认为训练过程中丢失专家是一个令人担忧的问题，因为它们是 token 级别的专家。
- **数据过滤与处理**：`@nruaif` 和 `@caseus_` 讨论了根据语言过滤特定数据集的需求，特别是移除非英语子集。`@nruaif` 建议使用 [fastText](https://fasttext.cc/)（一个用于学习文本表示和文本分类器的开源库）来过滤非英语内容。
- **考虑更大的上下文长度**：`@caseus_` 建议在样本中优先选择更大的上下文长度。不过，最终决定取决于团队成员的确认。
- **数据集建议**：提到了几个值得考虑的数据集，包括 [peS2o](https://huggingface.co/datasets/allenai/pes2o)、[yayi2_pretrain_data](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data)、[MathPile](https://huggingface.co/datasets/GAIR/MathPile) 和 [CulturaX](https://huggingface.co/datasets/uonlp/CulturaX)，其中 `@xzuyn` 提到 CulturaX 包含跨 167 种语言的 6.3 万亿个 token。

**提到的链接**：

- [fastText](https://fasttext.cc/)：用于高效文本分类和表示的库...
- [allenai/peS2o · Hugging Face 数据集](https://huggingface.co/datasets/allenai/peS2o)
- [uonlp/CulturaX · Hugging Face 数据集](https://huggingface.co/datasets/uonlp/CulturaX)
- [wenge-research/yayi2_pretrain_data · Hugging Face 数据集](https://huggingface.co/datasets/wenge-research/yayi2_pretrain_data)


        

---

## [LangChain AI](https://discord.com/channels/1038097195422978059) Discord 总结

- 讨论了 **LangChain 的功能**，包括结构化输出的示例、向 prompt 传递多个输入，以及 `@rajib2189` 分享的一个 [GitHub 仓库](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py)，其中包含更多实际应用。
- 用户对 LangChain 与其他平台的兼容性表现出兴趣，`@sarrah_1` 询问了如何将 LangChain 与 Laravel 项目集成以及特定 PHP 库的可用性，`@evolutionstepper` 关注其在 FastAPI 中异步运行一切的效用。此外，还有人建议通过 tokio 框架实现可能的异步方案。
- 针对 OpenAI Functions 和 OpenAI Tools Agents 之间的区别请求澄清，`@toasted_shibe` 解释说 Tools Agent 允许并行函数调用，并提供了 [OpenAI 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call)链接。
- `@alexk1919` 询问了 LangChain 在创建集成前一个 prompt 结果的 prompt 序列方面的相关性。
- `cheerful_moose_30860` 在导入 sentence-transformers 时遇到错误。

**LangChain AI 频道总结**

### ▷ #[announcements](https://discord.com/channels/1038097195422978059/1058033358799655042/) (1 条消息): 
        
cheerful_moose_30860: 导入 sentence-transformers 时出错

### ▷ #[general](https://discord.com/channels/1038097195422978059/1038097196224086148/) (33 条消息🔥): 
        
- **LangChain 输出结构**：`@seththunder` 指出可以使用 LangChain 输出解析器（output parser）来格式化特定结构的输出。
- **LangChain 示例**：`@rajib2189` 提供了一个如何在 LangChain 中向 Prompt 传递多个输入的示例，并分享了相关的 [GitHub 链接](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py)。
- **LangChain 与 Laravel 的集成**：`@sarrah_1` 询问了将 LangChain 集成到 Laravel 项目中的可能性，以及是否有特定的 PHP 库可用。
- **FastAPI 中 LangChain 的异步实现**：`@evolutionstepper` 表达了对 LangChain 是否能在 FastAPI 中异步运行所有内容的担忧。`@quantumqueenxox` 确认这是可能的，并提到他们有使进程异步的代码。`@evolutionstepper` 还对基于 tokio 框架构建的 LangChain 表示感兴趣。
- **OpenAI Functions 与 OpenAI Tools Agents 的区别**：`@keenborder` 要求澄清 OpenAI Functions 和 OpenAI Tools Agents 之间的区别，`@toasted_shibe` 解释说 tools agent 调用了新的 tools API 端点，允许并行函数调用，并参考了 [OpenAI 文档](https://platform.openai.com/docs/api-reference/chat/create#chat-create-function_call) 以获取更多信息。
- **使用 LangChain 处理 Prompt 序列**：`@alexk1919` 询问 LangChain 是否是创建利用前一个 Prompt 结果的 Prompt 序列的正确工具。

**提到的链接**：

[langchain_examples/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py at main · rajib76/langchain_examples](https://github.com/rajib76/langchain_examples/blob/main/examples/how_to_llm_chain_pass_multiple_inputs_to_prompt.py)：该仓库包含使用 LangChain 的示例。

---

## [LAION](https://discord.com/channels/823813159592001537) Discord 总结

- 讨论了由于潜在的法律问题，**LAION 数据集不可用**于研究的情况。用户表达了担忧，并提供了替代方案，例如从 [CommonCrawl](https://github.com/rom1504/cc2dataset) 创建自己的数据集。他们还建议对现有数据集进行彻底清理，包括删除无效内容和损坏的链接。
    - “*数据集目前正在审查中，以删除所有 NSFW 内容，特别是与儿童色情相关的内容……*”
- 辩论了数据集修改、处理丢弃内容以及 PR 后进行 rebase 的必要性。对话继续围绕处理持有旧版数据集副本的用户所面临的困难，以及保持数据集清洁和最新的挑战展开。
    - “*……这可以在提交 PR 后进行 rebase。但对于已经拥有旧数据集的用户来说，这不会生效。*”
- 注意到由于电脑问题，**DREAM 项目出现延迟**。
- 讨论了 [博客文章](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a) 中分享的 **GPT-4 细节泄露** 的可能性。然而，由于缺乏支持泄露的可靠证据，用户表示怀疑。
    - “*……没有确凿证据支持该博客文章的准确性或任何关于 GPT-4 的推测。*”
- 宣布发布名为“**Anytext Text ControlNet**”的新模型，并分享了其摘要[链接](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary)。
- 用户 `@puffy310` 对 [Modelscope](https://modelscope.cn/) 给予了正面评价。
    - “*……[Modelscope] 挺不错的，虽然还不如 Hugging Face。*”
- 深入解释了 **ChatGPT**、**SD** 和 **SDXL** 模型在架构、输入输出和训练方法方面的结构差异。

**LAION 频道总结**

### ▷ #[general](https://discord.com/channels/823813159592001537/823813160075132991/) (25 条消息🔥): 
        
- **LAION 数据集发布**：用户 `@ggez` 询问了 LAION 数据集预计重新发布的时间。`@chad_in_the_house` 回复称，由于法律问题，该数据集目前正在审查中，以删除所有 NSFW 内容，特别是与儿童色情相关的内容。
- **替代数据集来源**：`@thejonasbrothers` 建议利用 [CommonCrawl 数据](https://github.com/rom1504/cc2dataset) 创建自己的数据集。他们讨论了当前 LAION 数据集的问题，并预测了 LAION 可能需要采取的行动，例如利用更近期的 CommonCrawl 数据完全重建数据集，同时确保不包含令人反感的内容。
- **数据集修改**：针对内容合法性变化而修改数据集的讨论随之展开。`@progamergov` 建议 LAION 可以在提交 PR 后进行 rebase，但 `@nodja` 反驳说这对于已经拥有旧数据集的用户无效。他们进一步讨论了数据集所有者需要对其旧副本进行过滤的问题。
- **数据集清理与链接失效**：`@nodja` 还建议对数据集进行清理，包括删除 404 链接和不匹配的图像哈希值，假设目前数据集已损失约 10%。`@progamergov` 表示赞同，并提到 LAION 5B 数据集已经经历了严重的链接失效（link rot）问题。
- **DREAM 项目延迟**：最后，`@xylthixlm` 指出由于电脑问题，他们在 DREAM 项目上的工作有所延迟，预计暂停时间约为一周。


### ▷ #[research](https://discord.com/channels/823813159592001537/824374369182416994/) (7 条消息): 
        
- **GPT-4 细节泄露**：`@vrus0188` 分享了一篇[博文](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a)，据称揭露了关于 GPT-4 的细节，包括其模型架构、训练基础设施、参数数量、训练数据组成等。泄露源头是 Yam Peleg，他在 Twitter 上免费分享了这些最初由 Semi-Analysis 设置付费墙的细节。
- `@metal63` 表示怀疑，指出目前没有确凿证据支持该博文的准确性或任何其他关于 GPT-4 的推测。
- **Anytext Text ControlNet 发布**：`@thejonasbrothers` 宣布发布了一个名为 "Anytext Text ControlNet" 的新模型，并分享了其摘要[链接](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary)。 
- **Modelscope 评价**：`@puffy310` 对 [Modelscope](https://modelscope.cn/) 给予了正面评价，称其“还不错”，虽然不如 Hugging Face。

**提到的链接**：

- [GPT4- All Details Leaked](https://medium.com/@daniellefranca96/gpt4-all-details-leaked-48fa20f9a4a)：关于最佳 LLM 模型训练的细节以及...
- [AnyText多语言视觉文字生成与编辑模型](https://modelscope.cn/models/damo/cv_anytext_text_generation_editing/summary)


### ▷ #[learning-ml](https://discord.com/channels/823813159592001537/991941292999323668/) (1 条消息): 
        
- **聊天机器人架构讨论**：用户 `@JH` 深入解释了 **ChatGPT**、**SD** 和 **SDXL** 模型之间的架构差异。据他们称，ChatGPT 主要使用因果解码 Transformer，基于下一个 Token 预测任务进行推理。另一方面，SD 模型主要使用卷积 U-Net 架构，**SD v1** 输入来自 Clip L 的输出嵌入（embeddings），而 **SDXL** 则输入来自 Clip L + openClip G 的嵌入。U-Net 架构包含交叉注意力（cross attention）层和自注意力（self attention）层，通过变分下界损失（variational lower bound loss）和噪声预测损失进行训练。最后，`@JH` 认为，由于目标不同，可以合理预期这些不同的架构会以不同的方式学习概念。

---

## [Eleuther](https://discord.com/channels/729741769192767510) Discord 摘要

- 详细讨论了在 Transformer 模型中使用 **GEGLU** 激活函数及其效果，并提出了多种减少参数开销的策略。分享了来自 [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95) 实现的代码示例作为具体参考。
- 新成员 `@mouldysoul` 询问了有关提高对 flow-based models 理解的资源，以及它们与 optimal transport theory 之间的潜在关系。
- research 频道进行了多样化的讨论，话题涵盖从 **基于 PPO 的 adapter 训练** 到 **Transformer 架构的新颖修改**，并附带了 [trlX 论文](https://aclanthology.org/2023.emnlp-main.530/) 和 [一篇研究论文摘要](https://arxiv.org/abs/2311.02265) 的链接。讨论了关于 **ELC-BERT 架构** 及其在模型训练中重要性的见解。
- 在 **interpretability** 频道中，讨论围绕自动化可解释性、edge attribution patching，以及当前将高层级因果变量整合到 subspace discovery 中的趋势展开。分享了一篇关于 edge attribution patching 的 [研究论文](https://arxiv.org/abs/2310.10348)。表达了对 **MoE Models 及其可解释性** 的兴趣，并分享了 [Mixtral 仓库](https://github.com/dvmazur/mixtral-offloading) 的链接，作为在消费级平台上运行该模型的一种手段。
- `catboy_slim_` 提醒了关于 `gpt-neox-dev` 平台上 Python 3.8 的弃用周期。

**Eleuther 频道摘要**

### ▷ #[general](https://discord.com/channels/729741769192767510/729741769738158194/) (8 条消息🔥): 
        
- **关于 GEGLU 使用的讨论**：`@sentialx` 询问了在 Transformer 模型中使用 **GEGLU** 激活函数的优势，指出它增加了更多参数，但性能提升微乎其微。`@ad8e` 声称使用 GEGLU 会缩小维度，从而保持参数量不变。作为回应，`@sentialx` 提到在使用 GEGLU 时，Transformer 的 FFN 中间线性层需要将输出维度增加两倍。
- **降低 GEGLU 模型的参数开销**：`@bob80333` 解释说，在使用 **GEGLU**（或其变体）的模型中，减小中间层大小是一种常见策略，以便维持参数等效性。他引用了 *Llama 使用 8/3 倍增器* 而非 FFN 层标准 4 倍倍增器的做法，以抵消使用 SwiGLU 带来的开销。
- **关于 GEGLU 模型大小的澄清**：`@maxmatical` 澄清说，在实施了 `Llama 的 8/3 倍增器策略` 后，应用 `SwiGLU` 时 Transformer FFN 层的隐藏层大小将为 `16/3`。他们提供了 [NVIDIA/Megatron-LM 实现](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95) 作为代码参考。
- **新成员 `@mouldysoul` 加入**：`@mouldysoul` 是一位从事 AI 模型部署的专业人士，也是一位有志于机器学习研究的学者，他向社区介绍了自己。
- **对 Flow-Based Models 的咨询**：`@mouldysoul` 请求指导和资源以更好地理解 flow-based models，强调他有兴趣了解模型的双射映射（bijective mappings）、比 Diffusion 模型更快的采样能力、更好的插值效果，以及它们与 optimal transport theory 的潜在关系。

**提到的链接**：

[Megatron-LM/megatron/model/transformer.py at 2bc6cd307a11423928c675f741e79e03df23e721 · NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM/blob/2bc6cd307a11423928c675f741e79e03df23e721/megatron/model/transformer.py#L94-L95)：正在进行的在 sc... 上训练 Transformer 模型的研究。

### ▷ #[research](https://discord.com/channels/729741769192767510/747850033994662000/) (13 messages🔥): 
        
- **PEFT 技术与 Adapters**: `@cormerod` 询问是否可以使用 PPO 训练 Adapters，并结合 PEFT 技术来改进 7B 参数模型在特定案例下的输出。`@stellaathena` 给予了肯定，`@maxmatical` 提到此类功能可以在 trl、DeepSpeed Chat 等库中使用。
- **trlX 论文引用**: `@stellaathena` 提到了 [trlX 论文](https://aclanthology.org/2023.emnlp-main.530/)，该论文探讨了 PEFT 技术及其他相关特性（如层冻结）。[trlX 项目的 GitHub 仓库](https://github.com/CarperAI/trlx)。
- **关于 Transformer 架构修改的讨论**: `@digthatdata` 分享了一篇[研究论文摘要](https://arxiv.org/abs/2311.02265)，该论文提出了一种新型 Transformer 架构修改方案，用于语言模型的高效预训练。`@kharr.xyz` 评论称，这种修改对小于 100M 参数的模型有利，但随着规模增加，效果变得微不足道。`@ad8e` 认为论文中提到的 BabyLM 竞赛竞争并不激烈。
- **关于 ELC-BERT 架构的见解**: `@ad8e` 提供了关于 ELC-BERT 架构重要性的见解，考虑到最后一层会关注（attending to）第一层。`@kharr.xyz` 争论道，这些模式在训练过程中会发生变化，建议不要过度看重这些数据。讨论后，`@ad8e` 推断，最后一层关注第一层可能会随着训练数据的增加，从一个小任务变成一个更大的任务。`@kharr.xyz` 确认了这一点。
- **对噪声的鲁棒性**: `@eron_gj` 分享了关于架构对噪声鲁棒性的经验，指出即使将一半层数的 k/v/a 向量平均旋转高达 30 度，也不会损害输出的连贯性。

**提到的链接**:

[Not all layers are equally as important: Every Layer Counts BERT](https://arxiv.org/abs/2311.02265): 这篇论文介绍了一种新型的修改方案...


### ▷ #[interpretability-general](https://discord.com/channels/729741769192767510/1052314805576400977/) (6 messages): 
        
- **ACDC 之后的自动化可解释性工作**: `@dashiell_s` 询问了 ACDC 之后自动化可解释性领域的显著进展，并提到了 ACDC 仓库的存在。`@t_kwa` 指出，通过 Joseph Miller 实现的边缘子网络探测（edge subnetwork probing）进行的边缘归因补丁（edge attribution patching）和 Token 位置分析是该领域的进展。他们还提到正在进行将高层因果变量整合到子空间发现（subspace discovery）中的工作。 
- **ACDC 仓库的可用性**: 关于 ACDC 仓库，`@t_kwa` 指出，虽然由于需要从 FAR AI 的 Kubernetes 设置转换脚本，使用起来并不直接，但 Demo Notebook 仍然可以顺利运行。
- **边缘归因补丁的效率**: `@neelnanda` 引用了一篇由 `<@349859906570027010>` 指导的论文，证明了边缘归因补丁在速度和电路输出检索方面优于 ACDC。论文可以在[这里](https://arxiv.org/abs/2310.10348)访问。
- **对 MoE 模型与可解释性的兴趣**: `@sk5544` 对可解释性与 Mixture of Experts (MoE) 模型交叉领域的研究表示好奇。他们指出，即使是小型 MoE 模型的高计算强度也是学术实验的障碍。
- **在消费级平台上运行 MoE 模型**: 作为回应，`@stellaathena` 建议在 Google Colab 上运行 Mixtral（一种 MoE 模型），并提供了其 [GitHub 仓库](https://github.com/dvmazur/mixtral-offloading)链接。

**提到的链接**:

- [Attribution Patching Outperforms Automated Circuit Discovery](https://arxiv.org/abs/2310.10348): 自动化可解释性研究最近...
- [GitHub - dvmazur/mixtral-offloading: Run Mixtral-8x7B models in Colab or consumer desktops](https://github.com/dvmazur/mixtral-offloading): 在 Colab 或消费级桌面运行 Mixtral-8x7B 模型...


### ▷ #[gpt-neox-dev](https://discord.com/channels/729741769192767510/730090096287547444/) (1 messages): 
        
catboy_slim_: Python 3.8 将在明年或今年被弃用，具体取决于你当前的时区。

---

## [DiscoResearch](https://discord.com/channels/1178995845727785010) Discord 摘要

只有 1 个频道有活动，因此无需总结...

- **上下文检索基准 (Benchmark for Context Retrieval)**：`@rasdani` 表示有兴趣基于 deepset/germanquad 创建一个**上下文检索基准**。计划从测试集中选择 60 个问题/上下文对，其中一半配以无关的上下文，并使用 0 和 1 作为余弦相似度 (cosine similarity) 的 Ground Truth。该基准的目标是比较不同的 Embedding 模型并计算成对相关性。
- **点积 vs 余弦相似度 (Dot Product vs Cosine Similarity)**：`@philipmay` 建议，在对问题和段落使用语义 Embedding 时，**点积 (Dot Product) 比余弦相似度更有效**。这个建议最初是由 Embedding 专家 Nils Reimers 提供给他们的。
- **检索系统指标 (Metrics for Retrieval Systems)**：在关于检索系统指标的对话中，`@philipmay` 指出 MRR@10 经常被使用，而 `@hammadkhan` 注意到 MTEB 排行榜使用的是 NDCG@10，它根据相关性和在前 10 项中的位置来评估检索质量。
- **多正向上下文数据集 (Data Sets for Multiple Positive Contexts)**：`@rasdani` 征求德语中具有多个正向上下文的上下文 QA 数据集推荐，因为 germanquad 中每个问题只有一个正向参考上下文，他们计划在基准测试中使用 MRR@10。
- **新年问候**：`@bjoernp` 和 `@thewindmom` 向 Discord 服务器的成员致意，并表达了对未来发展的期待。

---

## [Latent Space](https://discord.com/channels/822583790773862470) Discord 摘要

只有 1 个频道有活动，因此无需总结...

- **在米兰组织活动**：用户 `@alexio.c` 提议在意大利米兰组织一次活动。`@fanahova` 给予了积极回应，建议在其他当地团体中发布消息。
- **AI 平台讨论**：`@aristokratic.eth` 寻求 AI 平台的建议。`@fanahova` 推荐了 **Unstructured.io**，因为它获得的资金最多。

---

## [Alignment Lab AI](https://discord.com/channels/1087862276448595968) Discord 摘要

- 公会成员交换**新年问候**和祝福，培养了社区感和友谊。
- 为了庆祝新年，用户 `@cryptossssun` 在 oo 和 oo2 频道分享了一个 [2022 新年 GIF](https://tenor.com/view/new-year-2022-gif-24334949)，为讨论增添了节日和欢乐的气氛。
- 还提供了关于所分享 GIF 的详细信息，指出文件大小为 **1303KB**，时长为 **1.200 秒**，尺寸为 **498x331**，这表明了对细节的关注，并可能与关于数字媒体分辨率和格式的讨论有关。

**Alignment Lab AI 频道摘要**

### ▷ #[oo](https://discord.com/channels/1087862276448595968/1118217717984530553/) (4 条消息): 
        
- **新年祝福**：用户 `@cryptossssun`、`@teknium` 和 `@neverendingtoast` 在 Alignment Lab 的 oo Discord 频道中向社区分享了他们的**新年问候**和祝福。
- **新年 Gif**：`@cryptossssun` 还分享了一个 [新年 Gif](https://tenor.com/view/new-year-2022-gif-24334949) 来庆祝 2022 年的开始。

**提到的链接**：

[New Year GIF - New Year 2022 - Discover & Share GIFs](https://tenor.com/view/new-year-2022-gif-24334949)：点击查看 GIF


### ▷ #[oo2](https://discord.com/channels/1087862276448595968/1176548760814375022/) (2 条消息): 
        
- 用户 `@cryptossssun` 分享了一个 **[2022 新年 GIF](https://tenor.com/view/new-year-2022-gif-24334949)**，祝愿大家新年快乐，事业有成。
- GIF 详情包括文件大小 **1303KB**，时长 **1.200 秒**，尺寸 **498x331**。该 GIF 创建于 **2022 年 1 月 1 日**。

**提到的链接**：

[New Year GIF - New Year 2022 - Discover & Share GIFs](https://tenor.com/view/new-year-2022-gif-24334949)：点击查看 GIF


---

## [Skunkworks AI](https://discord.com/channels/1131084849432768614) Discord 摘要

只有 1 个频道有活动，因此无需总结...

teknium: 哈哈，这正是他想要的 😄