# SPAR 复现说明

本文档基于 `pasa/SPAR` 当前代码整理，目标是把 SPAR 从环境准备到 Web 演示、API 调用、benchmark 批量运行的复现路径讲清楚。

SPAR 全称是 `Scholar Paper Retrieval with LLM-based Agents for Enhanced Academic Search`，核心流程是：用户输入学术检索问题，系统用 LLM 做 query 改写、意图分析和相关性判断，再调用 arXiv、OpenAlex、PubMed、Google Serper 等外部检索源获取论文，最后做重排并输出检索结果和搜索树。

## 1. 目录结构

主要文件如下：

| 文件 | 作用 |
| --- | --- |
| `demo_app_with_front.py` | 推荐优先复现的 Web/API 服务入口，启动 FastAPI 并加载 `index.html` |
| `index.html` | 前端页面 |
| `pipeline_spar.py` | SPAR 高级检索流水线，包含 `AcademicSearchTree` |
| `search_engine.py` | 多源检索、query 改写、融合判断和 rerank 逻辑 |
| `api_web.py` | arXiv、OpenAlex、PubMed、Semantic Scholar、Google Serper 等外部 API 封装 |
| `local_request_v2.py` | LLM 调用封装，默认按 OpenAI-compatible 接口访问 DeepSeek，也保留本地 Qwen 配置 |
| `global_config.py` | 全局配置，包括模型名、API key、检索源、阈值和代理 |
| `run_spr_agent.py` | benchmark 批量运行入口，但当前文件存在语法问题，需要先修复 |
| `benchmark/` | 内置测试集，包含 `AutoScholarQuery_test.jsonl` 和 `spar_bench.jsonl` |
| `figs/` | 示例图片和示例检索结果 |

## 2. 环境准备

建议使用 Python 3.10 或 3.11，新建虚拟环境：

```powershell
cd D:\Desktop\人工智能创新大赛\github-pasa\pasa\SPAR
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

项目自带的 `requirements.txt` 内容不完整，而且包含 `sqlite3`。`sqlite3` 是 Python 标准库，一般不需要也不能通过 pip 单独安装。建议先安装项目实际用到的依赖：

```powershell
pip install dataclasses func_timeout cachetools openai biopython graphviz fastapi uvicorn pydantic requests arxiv numpy tqdm
```

如果需要生成搜索树图片，还需要安装 Graphviz 系统程序：

1. 下载并安装 Graphviz: <https://graphviz.org/download/>
2. 将 Graphviz 的 `bin` 目录加入系统 `PATH`
3. 安装 Python 包：

```powershell
pip install graphviz
```

安装后可检查：

```powershell
dot -V
```

## 3. 配置模型和 API

### 3.1 默认模型调用方式

当前代码已经改为默认使用 DeepSeek API，不需要先下载本地模型。服务器启动前设置：

```bash
export DEEPSEEK_API_KEY="你的 DeepSeek key"
export GOOGLE_SERPER_KEY="你的 Google Serper key"
```

`global_config.py` 会按下面的优先级选择模型：

```python
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", DEPLOYMENT_NAME)
DEPLOYMENT_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
```

也就是说，默认模型是 `deepseek-chat`。如果要换模型，可以设置：

```bash
export DEEPSEEK_MODEL="deepseek-chat"
```

或者直接覆盖 SPAR 使用的模型配置名：

```bash
export LLM_MODEL_NAME="deepseek-chat"
```

`local_request_v2.py` 中已经预置以下 DeepSeek 配置名：

- `deepseek-chat`
- `deepseek-reasoner`
- `deepseek-v4-flash`
- `deepseek-v4-pro`

如果你仍想使用本地 Qwen，也可以把 `LLM_MODEL_NAME` 改回 `Qwen3-8B`，并启动对应的本地 OpenAI-compatible 服务。

### 3.2 DeepSeek API 配置

当前 `global_config.py` 中：

```python
API_KEY = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", "your_openai_api_key_here"))
ENDPOINT = os.getenv("OPENAI_ENDPOINT", os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
DEPLOYMENT_NAME = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
```

当前主流程实际通过 `local_request_v2.py` 的 `get_from_llm()` 调模型。DeepSeek 走 OpenAI-compatible SDK，请求地址默认是：

```bash
export DEEPSEEK_BASE_URL="https://api.deepseek.com"
```

### 3.3 检索源配置

默认配置：

```python
SEARCH_ROUTES = ["arxiv", "openalex"]
DO_REFERENCE_SEARCH = False
ENABLE_RERANK = False
```

Web 前端可选择：

- `arxiv`
- `openalex`
- `pubmed`

注意：

- `openalex` 和 `pubmed` 主要依赖公网 API。
- `arxiv` 路线会先用 Google Serper 搜索 arXiv ID，再查 arXiv 详情，因此需要 `GOOGLE_SERPER_KEY`。
- `semantic` 相关代码存在，但 Web 前端默认没有开放该选项。

如果使用 arXiv/Google 路线，需要申请 Google Serper key，并设置：

```powershell
$env:GOOGLE_SERPER_KEY="你的 Google Serper key"
```

也可以在 Web 页面里填写 Google Serper Key，后端会把它写入当前进程环境变量。

### 3.4 代理配置

`global_config.py` 默认代理是：

```python
PROXIES = {
    "http": "http://localhost:1080",
    "https": "http://localhost:1080"
}
```

如果你本机没有 1080 代理，而外部 API 可以直连，建议改成空字典：

```python
PROXIES = {}
```

否则 Semantic Scholar、OpenAlex、arXiv 等请求可能因为代理不可用而失败。

## 4. 推荐复现路线：启动 Web Demo

先进入 SPAR 目录：

```powershell
cd D:\Desktop\人工智能创新大赛\github-pasa\pasa\SPAR
.\.venv\Scripts\Activate.ps1
```

启动服务：

```powershell
python demo_app_with_front.py
```

成功后访问：

```text
http://127.0.0.1:8000
```

健康检查：

```powershell
curl http://127.0.0.1:8000/health
```

预期返回类似：

```json
{"status":"healthy","message":"Scholar Paper Search API is running"}
```

页面上可以输入学术问题，选择检索源，然后提交搜索。

建议第一次复现时使用简单模式或少量检索源，避免 LLM 和外部 API 调用过多。比如只选 `openalex`，问题可以用：

```text
retrieval augmented generation for large language models
```

## 5. API 方式复现

服务启动后，可以直接请求 `/search`。

### 5.1 简单检索

简单检索不走完整搜索树，但仍会做关键词抽取，因此仍需要 LLM 可用：

```powershell
curl -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"queries\":[\"retrieval augmented generation for large language models\"],\"sources\":[\"openalex\"],\"use_advanced_search\":false,\"max_workers\":1,\"batch_size\":5}"
```

### 5.2 高级检索

高级检索走 `AcademicSearchTree`，会进行 query 改写、意图分析、相关性判断和搜索树构建，耗时和 token 消耗都会更高：

```powershell
curl -X POST http://127.0.0.1:8000/search `
  -H "Content-Type: application/json" `
  -d "{\"queries\":[\"retrieval augmented generation for large language models\"],\"sources\":[\"openalex\"],\"use_advanced_search\":true,\"max_depth\":1,\"relevance_doc_num\":5,\"similarity_threshold\":0.5}"
```

返回字段主要包括：

| 字段 | 含义 |
| --- | --- |
| `status` | `success` 或 `error` |
| `total_papers` | 汇总论文数量 |
| `query_results` | 每个 query 对应的论文列表 |
| `all_papers` | 去重后的论文详情字典 |
| `query_source_map` | query 到检索源的映射 |
| `search_tree` | 高级检索模式下的搜索树 |

## 6. benchmark 批量复现

README 中给出的批量命令是：

```powershell
python run_spr_agent.py AutoScholarQuery
```

或：

```powershell
python run_spr_agent.py OwnBenchmark
```

两个 benchmark 映射为：

| 参数 | 数据文件 |
| --- | --- |
| `AutoScholarQuery` | `benchmark/AutoScholarQuery_test.jsonl` |
| `OwnBenchmark` | `benchmark/spar_bench.jsonl` |

但是，当前 `run_spr_agent.py` 直接运行会失败。我在本地做了静态编译检查：

```powershell
python -m py_compile .\pasa\SPAR\run_spr_agent.py
```

报错为：

```text
SyntaxError: invalid syntax
```

问题出在导入配置处，当前文件里有不完整的符号：

```python
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_
    DO_FUSION_JUDGE,
    FUSION_TEMP,
    SEARCH_ROUTE,
)
```

而 `global_config.py` 中实际存在的是：

```python
DO_REFERENCE_SEARCH
DO_FUSION_JUDGE
FUSION_TEMPLATE
SEARCH_ROUTES
```

因此 benchmark 路线需要先修复变量名，例如把导入和后续引用统一改成：

```python
from global_config import (
    LLM_MODEL_NAME,
    DO_REFERENCE_SEARCH,
    DO_FUSION_JUDGE,
    FUSION_TEMPLATE,
    SEARCH_ROUTES,
)
```

同时把：

```python
SEARCH_ROUTE
FUSION_TEMP
```

分别改成：

```python
SEARCH_ROUTES
FUSION_TEMPLATE
```

此外，`OwnBenchmark` 的 `select_file` 当前写成：

```python
"code_official/benchmark/spar_bench_select_2000.jsonl"
```

这个目录在当前 `pasa/SPAR` 下不存在，运行前需要改成类似：

```python
"./benchmark/spar_bench_select_2000.jsonl"
```

修复后再运行：

```powershell
python run_spr_agent.py AutoScholarQuery
```

输出目录形如：

```text
gen_result/AutoScholarQuery_2000_msearch_arxiv-openalex_depth2_do_reference_False_query_judge_True_fusion_AUTOMATIC_no_enddate_no_autocorrect_pasa_score_0.5
```

每个 query 会输出一个 JSON 文件，并尝试生成搜索树可视化文件。

## 7. 本地数据库缓存

README 提供了本地数据库下载方式：

```bash
mkdir -p database
wget "http://flagchat.ks3-cn-beijing.ksyuncs.com/shixiaofeng/project/SPAR/arxiv_data.db?KSSAccessKeyId=AKLTkqVnZwpfTBiiu7O6iQHnA&Expires=7753081204&Signature=gaj8%2F5rJ%2BUQWp6wSr0f5KKuJdqs%3D" -O database/arxiv_data.db
```

但当前 `local_db_v2.py` 实际使用的是：

```python
db_path = "./database/recovered.db"
```

如果你下载的是 `arxiv_data.db`，需要二选一：

1. 把下载文件改名为 `database/recovered.db`
2. 或把 `local_db_v2.py` 中的 `db_path` 改成 `./database/arxiv_data.db`

如果不下载数据库，代码会在 `database/recovered.db` 创建一个新的 SQLite 缓存库。注意需要先确保 `database` 目录存在：

```powershell
mkdir database
```

## 8. 常见问题

### 8.1 `ModuleNotFoundError`

说明依赖没有装全。建议执行：

```powershell
pip install dataclasses func_timeout cachetools openai biopython graphviz fastapi uvicorn pydantic requests arxiv numpy tqdm
```

### 8.2 `sqlite3` 安装失败

这是正常的。`sqlite3` 是 Python 标准库，不需要写在 pip 依赖里。可以从 `requirements.txt` 中移除。

### 8.3 LLM 请求失败

检查：

- 服务器启动前是否设置了 `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL` 是否是可访问的 OpenAI-compatible 地址，默认是 `https://api.deepseek.com`
- `LLM_MODEL_NAME` 是否在 `local_request_v2.py` 的 `MODEL_CONFIGS` 中存在
- `DEEPSEEK_MODEL` 是否是 DeepSeek 账号当前可用的模型名
- 如果改回本地 Qwen，本地 OpenAI-compatible 服务是否已启动，端口和模型名是否一致

### 8.4 外部检索 API 请求失败

检查：

- 网络是否能访问 OpenAlex、arXiv、PubMed、Semantic Scholar
- `PROXIES` 是否配置正确
- 使用 arXiv/Google 路线时，`GOOGLE_SERPER_KEY` 是否有效

### 8.5 Web 页面能打开但搜索返回 `error`

优先看后端终端日志。常见原因是：

- LLM 服务不可用
- 检索源 API 访问失败
- 代理配置错误
- Google Serper key 缺失
- Graphviz 未安装，导致搜索树可视化失败

### 8.6 benchmark 不能运行

当前 `run_spr_agent.py` 有语法错误和变量名不一致问题。先按第 6 节修复，再运行 benchmark。

## 9. 最小复现建议

最稳的复现顺序：

1. 安装 Python 依赖。
2. 启动本地 OpenAI-compatible LLM 服务，或修改 `local_request_v2.py` 指向可用服务。
3. 确认 `global_config.py` 中 `PROXIES` 与本机网络一致。
4. 启动：

```powershell
python demo_app_with_front.py
```

5. 打开：

```text
http://127.0.0.1:8000
```

6. 先只选 `openalex`，用简单问题测试。
7. 简单检索成功后，再打开高级检索、arXiv/Google、PubMed、Graphviz 和 benchmark。

## 10. 当前代码检查结论

我对当前仓库做了静态编译检查：

```powershell
python -m py_compile .\pasa\SPAR\demo_app_with_front.py
python -m py_compile .\pasa\SPAR\search_engine.py .\pasa\SPAR\pipeline_spar.py .\pasa\SPAR\api_web.py
python -m py_compile .\pasa\SPAR\run_spr_agent.py
```

结果：

- `demo_app_with_front.py` 可以编译。
- `search_engine.py`、`pipeline_spar.py`、`api_web.py` 可以编译，其中 `api_web.py` 有一个正则转义 warning，不阻塞运行。
- `run_spr_agent.py` 当前不能编译，需先修复后再跑 benchmark。

因此，当前最推荐的复现入口是：

```powershell
python demo_app_with_front.py
```
