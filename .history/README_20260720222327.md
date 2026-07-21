# InstitutionScanner — 机构吸筹扫描器

**Institutional Accumulation Scanner** 是一个 Python 量化工具，专门寻找处于长期熊市底部、被大资金持续放量承接、但价格尚未启动的美股和 ETF。

---

## 核心理念

传统扫描器寻找"上涨"的股票。这个扫描器寻找的是：

> **被低估的、底部横盘的、机构在悄悄买入的标的。**

对标：TLT / IGV / SOXS 等长期下跌后大资金慢慢接货的模式。

---

## 功能

- **多数据源 ticker 自动发现**：NASDAQ / NYSE / AMEX 全部股票，1000+ ETF
- **10 年历史日线数据下载**，增量缓存，断点续传
- **50+ 技术指标**：价格、成交量、资金流、趋势、波动率、Volume Profile
- **Wyckoff 阶段自动检测**：吸筹 / 拉升 / 派发 / 下跌 + 卖出高潮
- **100 分制评分系统**：趋势(20) + 成交量(25) + 吸筹(25) + 波动率(15) + 底部结构(15)
- **Top50 CSV + Top200 Parquet** 输出
- **终端格式化报告**，每个标的附带理由

---

## 快速开始

### 环境要求

- Python 3.12+
- pip
- (可选) Docker

### 安装

```bash
cd InstitutionScanner
pip install --break-system-packages -r requirements.txt
```

### 使用

```bash
# 完整扫描（股票 + ETF）
python main.py scan

# 只看 ETF
python main.py scan --etfs-only

# 只看特定标的
python main.py scan --tickers TLT,SOXS,ASHR,IGV

# 强制重新下载所有数据
python main.py scan --force-download

# 重新评分（不重新下载，速度快）
python main.py report

# 只下载数据
python main.py download

# 清除所有缓存
python main.py clean
```

### 输出文件

| 文件 | 说明 |
|------|------|
| `output/Top50.csv` | 得分最高的 50 个标的 |
| `output/Top200.parquet` | 前 200 名，Parquet 格式 |
| `cache/*.csv` | 每个 ticker 的 OHLCV 缓存 |
| `logs/*.log` | 详细运行日志 |

---

## Docker 运行

```bash
# 构建镜像
docker build -t institution-scanner .

# 方式一：docker compose
docker compose up

# 方式二：直接运行
docker run --rm \
  -v $(pwd)/cache:/app/cache \
  -v $(pwd)/output:/app/output \
  -v $(pwd)/logs:/app/logs \
  institution-scanner \
  python main.py scan --etfs-only
```

### 定时运行

取消 `docker-compose.yml` 中 `scheduler` 部分的注释即可启用每周一上午 8:00（美东时间）自动扫描。

---

## 项目结构

```
InstitutionScanner/
├── config.py          # 所有可调参数
├── main.py            # CLI 入口
├── scanner.py         # 扫描编排引擎
├── indicators.py      # 50+ 技术指标计算
├── score.py           # 100 分制评分系统
├── filters.py         # 筛选条件
├── downloader.py      # 数据下载 & ticker 发现
├── report.py          # 输出生成（CSV / Parquet / 终端）
├── requirements.txt   # Python 依赖
├── Dockerfile         # Docker 镜像
├── docker-compose.yml # Docker Compose 编排
├── cache/             # 历史数据缓存
├── output/            # 扫描结果
└── logs/              # 运行日志
```

---

## 评分维度详解

| 维度 | 满分 | 考察内容 |
|------|------|----------|
| **趋势** | 20 | MA200 下降程度、价格低于 MA200 的幅度、熊市持续时间、两年跌幅 |
| **成交量** | 25 | 持续放量天数、Volume Ratio 大小、Volume Trend、成交量稳定性 |
| **吸筹** | 25 | OBV 顶背离、A/D 线斜率、CMF 正值、MFI 区间 |
| **波动率** | 15 | ATR 压缩、布林带宽收缩、历史波动率下降 |
| **底部结构** | 15 | 距离 52 周低点距离、横盘时间、线性回归斜率、Volume Profile HVN 位置 |

---

## 自定义配置

编辑 `config.py`：

```python
MIN_PRICE = 5.0           # 最低价格
MIN_VOLUME = 200_000      # 最低日均成交量
BEAR_DECLINE_PCT = -30.0  # 熊市跌幅阈值
VOLUME_ACCUM_RATIO = 1.5  # 放量倍率
DOWNLOAD_THREADS = 8      # 下载线程数
```

---

## 性能

- 预期：8000 个标的（7000 股 + 1000 ETF）首次下载约 30-40 分钟
- 后续增量更新 + 重新扫描：10-15 分钟
- 支持断点续扫（`--resume`，默认启用）
- 支持 Docker 部署

---

## 数据源

- **行情数据**：[yfinance](https://github.com/ranaroussi/yfinance)（Yahoo Finance）
- **Ticker 列表**：NASDAQ Trader FTP、Wikipedia S&P 500、ETFdb
- 全部免费，无需 API Key

---

## 注意事项

1. **首次运行**会下载所有 ticker 的 10 年历史数据，耗时较长，请耐心等待。
2. **yfinance 有速率限制**，项目内置了批次延迟和重试机制。如果遇到 429 错误，增大 `DOWNLOAD_BATCH_PAUSE` 参数。
3. **Docker 环境**中，cache 和 output 目录已挂载到宿主机，不会丢失数据。
4. **增量化**：第二次运行只下载新增的 K 线，速度极快。
5. **免责声明**：本工具仅供研究和学习使用，不构成任何投资建议。过往表现不代表未来收益。

---

## License

MIT
