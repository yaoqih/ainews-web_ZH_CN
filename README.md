
# 🤖 AI News CN (全自动中文同步版)

> **自动追踪 · 智能翻译 · 每日更新**
>
> 本项目通过 GitHub Actions 自动监控 [smol-ai/ainews-web-2025](https://github.com/smol-ai/ainews-web-2025)，利用大模型将最新的 AI 技术新闻翻译为中文，并发布到 GitHub Pages。

[在线阅读最新一期](https://yaoqih.github.io/ainews-web_ZH_CN/) ## ✨ 项目亮点

* **零服务器成本**：完全基于 GitHub Actions (计算) + GitHub Pages (托管)，无需租赁服务器。
* **智能并行翻译**：
    * 自动识别 Markdown 结构，按 H1-H4 标题将长文切分为多个片段。
    * 使用 `ThreadPoolExecutor` 并发调用 LLM，大幅提升翻译速度（比单线程快 5-10 倍）。
    * 智能识别代码块，确保代码不被错误翻译。
* **模型中立**：支持任何兼容 OpenAI 格式的 API（如 **DeepSeek-V3**, **GPT-4o**, **Claude via OneAPI** 等）。
* **GitOps 工作流**：自动提交、自动构建、自动部署。

## 🛠️ 系统架构

```mermaid
graph LR
    A[🕒 定时触发 (GitHub Actions)] -->|1. 获取文件列表| B[🔍 上游仓库 (smol-ai)]
    B -->|2. 对比增量| C{本地是否存在?}
    C -- 是 --> D[跳过]
    C -- 否 --> E[⚡ Python 脚本处理]
    E -->|3. 解析 & 智能切分| F[Chunk 1...N]
    F -->|4. 并发翻译 (DeepSeek/GPT)| G[调用 LLM API]
    G -->|5. 组装 & 重建 Frontmatter| H[生成中文 Markdown]
    H -->|6. Git Push| I[📂 docs/ 目录]
    I -->|7. 自动部署| J[🌐 GitHub Pages]

```

## 🚀 快速开始

如果你想部署自己的版本，请按照以下步骤操作：

### 1. Fork 本仓库

点击右上角的 **Fork** 按钮，将本项目复制到你的 GitHub 账号下。

### 2. 配置密钥 (Secrets)

进入仓库的 `Settings` -> `Secrets and variables` -> `Actions` -> `Repository secrets`，添加以下变量：

| Secret Name | 说明 | 示例值 |
| --- | --- | --- |
| `LLM_API_KEY` | 你的大模型 API 密钥 | `sk-xxxxxxxx` |
| `LLM_BASE_URL` | API 接口地址 | `https://api.deepseek.com/v1` 或 `https://api.openai.com/v1` |

> **注意**：请确保添加在 **Repository secrets** 中，而不是 Environment secrets。

### 3. 开启 GitHub Pages

1. 进入 `Settings` -> `Pages`。
2. 在 **Source** 下选择 `Deploy from a branch`。
3. **Branch** 选择 `main`，文件夹选择 `/docs` (这很重要！)。
4. 点击 Save。

### 4. 手动触发一次

进入 `Actions` 选项卡，选择 `Auto Translate & Publish`，点击 `Run workflow` 手动运行一次，测试配置是否成功。

---

## ⚙️ 本地开发与调试

如果你想在本地运行脚本：

1. **安装依赖**
```bash
pip install -r requirements.txt

```


2. **设置环境变量**
```bash
export LLM_API_KEY="your_key"
export LLM_BASE_URL="[https://api.deepseek.com](https://api.deepseek.com)"

```


3. **运行诊断脚本** (测试 API 连接)
```bash
python debug_llm.py

```


4. **运行翻译脚本**
```bash
python translator.py

```



## 📂 核心文件说明

* `translator.py`: 核心逻辑脚本。负责下载、解析 Markdown、并发调用 LLM 进行翻译以及重新组装文件。
* `debug_llm.py`: 用于诊断 API 连接问题的工具脚本。
* `.github/workflows/daily_sync.yml`: GitHub Actions 配置文件，定义了定时任务（每天 UTC 0点）和 CI/CD 流程。
* `docs/`: 存放翻译后的 Markdown 文件和 `index.md`，也是 GitHub Pages 的发布源。

## 🙏 致谢

* 上游内容来源：[smol-ai/ainews-web-2025](https://www.google.com/url?sa=E&source=gmail&q=https://github.com/smol-ai/ainews-web-2025) (感谢 swyx 的辛勤整理)
* 翻译驱动：gemini-3-flash-preview

## 📄 License

MIT License

