# OME-TIFF 断点续传入湖与双染料解混 API

单进程服务，把网络易中断的大幅 OME-TIFF 以 **8 MiB 分片、真实 S3 multipart upload**
续传进本地 MinIO 数据湖；合并后核验 OME 元数据与 TIFF 页目录，**坏图不得入湖**；
分析程序只能读到完整原件。在此之上提供**双染料探测器串色解混**：对已入湖的
双通道 CYX uint16 图像逐像素做非负最小二乘（NNLS），输出带染料通道名的
两通道 float32 OME-TIFF，原件保持不变。

技术栈：FastAPI · SQLite（WAL）· tifffile · NumPy（NNLS 向量化实现，无 scipy 依赖）
· boto3（S3 multipart）· 本地 MinIO · Uvicorn

---

## 1. 快速开始

```bash
# 启动持久化 MinIO + API（含故障注入开关）
./scripts/run_demo.sh
```

或手动分步：

```bash
docker compose up -d --build
# 生成无隐私的合成 OME-TIFF 样例（大图/小图/坏图/双通道 CYX 解混样例）
docker compose exec api python scripts/generate_samples.py /sample-data
# 运行 8 个端到端场景（含双染料解混）
docker compose exec api python scripts/demo.py
```

入口：

| 入口 | 地址 | 凭据 |
| --- | --- | --- |
| Swagger UI（可操作全流程） | http://localhost:8000/docs | — |
| OpenAPI Schema | http://localhost:8000/openapi.json | — |
| MinIO S3 API | http://localhost:9000 | minioadmin / minioadmin |
| MinIO 控制台 | http://localhost:9001 | minioadmin / minioadmin |

停止与清盘：

```bash
docker compose down        # 停止，保留数据卷（续传/恢复状态仍在）
docker compose down -v     # 连同 MinIO 与 SQLite 持久化卷一起清除
```

---

## 2. API 全流程

### 2.1 创建上传（按样本号，声明总字节数）

```bash
curl -X POST http://localhost:8000/uploads \
  -H 'Content-Type: application/json' \
  -d '{"sample_id":"S-2026-001","filename":"slide.ome.tiff","total_bytes":25169920}'
```

响应给出分片规划（`part_size` 固定 8 MiB，末片按总字节取余可短）：

```json
{
  "upload_id": "ab12...", "object_key": "samples/S-2026-001/ab12.../slide.ome.tiff",
  "total_bytes": 25169920, "part_size": 8388608,
  "expected_parts": 4, "status": "open"
}
```

### 2.2 查询已接纳片号（断点续传决策）

```bash
curl http://localhost:8000/uploads/$UID/parts
# -> received_parts / missing_parts / bytes_received
```

### 2.3 上传 / 重传分片（请求体即原始分片字节）

```bash
curl -X PUT --data-binary @part-003.bin \
  http://localhost:8000/uploads/$UID/parts/3
```

- 非末片必须恰好 **8 MiB**，末片按声明大小取余；
- **同片同字节重传 → 200 幂等**（直接返回，不再次请求 S3）；
- **同片异内容 → 409**，已接纳分片绝不被覆盖；
- 不同片号可并发上传，服务按 `upload_id` 串行化同一任务。

### 2.4 完成（齐全后才能合并）

```bash
curl -X POST http://localhost:8000/uploads/$UID/complete
```

分片缺号、字节合计与声明不符都会返回 409。合并成功后自动：

1. `head_object` 核对合并对象大小 == 声明总字节；
2. 下载到本地临时文件，用 tifffile 逐页遍历 **TIFF 页目录（IFD）**，
   校验 **OME-XML 元数据**、维度与 dtype；
3. 若创建时给了 `content_sha256`，再做整图哈希校验；
4. 全部通过才置 `completed` 并开放查询；任何失败 → 删除远端对象、置 `invalid`，坏图不入湖。

成功响应直接带回提取的尺寸：

```json
{"status":"completed","width":2048,"height":2048,"pages":3,"dtype":"uint16","axes":"ZYX"}
```

### 2.5 放弃（与完成竞争，只有一种生效）

```bash
curl -X POST http://localhost:8000/uploads/$UID/abort
```

- 通过 SQLite CAS 状态流转保证与 `complete` 竞争时只有一种生效；
- 先落库 `aborted`（此后迟到分片在入口即被 **410** 拒绝），
  再 `AbortMultipartUpload` **清除远端全部分片**，不留残件。

### 2.6 仅完成任务开放的查询

```bash
curl http://localhost:8000/uploads/$UID/dimensions   # 尺寸/页数/dtype/axes
curl -O http://localhost:8000/uploads/$UID/object    # 流式下载完整原件
```

`open / completing / aborted / invalid` 状态访问均返回 409/410，分析程序无法读到半成品。

---

## 3. 双染料解混 API（探测器串色消除）

### 3.1 适用输入（约束在服务端强制校验）

| 约束 | 要求 | 违反时 |
| --- | --- | --- |
| 上传状态 | 仅 `completed` 的原件可提交 | 409 `not_ready` |
| 序列数 | 单序列（single series） | 422 `unmixing_error` |
| 轴序/形状 | 轴序恰好 `CYX`、恰好 2 个通道 | 422 |
| 像素类型 | `uint16` | 422 |
| 解码体积 | 像素缓冲 ≤ **16 MiB**（即 2×H×W×2 字节） | 422 |
| 响应矩阵 | 2×2、全部有限、**条件数 ≤ 1000** | 422 |
| 有效像素 | 不允许全部像素饱和 | 422 |
| 原件读取 | 湖中对象必须可读 | 502 `storage_error` |

任何失败都给出明确错误码与中文原因，**不写入结果对象、不返回残缺产物**，
失败任务以 `failed` 状态留痕（含 `error_message`）。

### 3.2 提交解混任务

`response_matrix` 为 2×2 响应矩阵：**行 = 两路探测器**（与原件通道顺序一致），
**列 = 两种染料**（与 `dye_names` 顺序一致），即 `A[i,j]` 表示第 j 种染料
在第 i 路探测器上的响应。模型 `y = A·x`，逐像素求非负最小二乘解 `x ≥ 0`。

```bash
curl -X POST http://localhost:8000/uploads/$UID/unmix \
  -H 'Content-Type: application/json' \
  -d '{"dye_names":["Dye-A","Dye-B"],
       "response_matrix":[[1.0,0.25],[0.4,1.0]]}'
```

成功响应（HTTP 201）：

```json
{
  "job_id": "9f3c...", "upload_id": "ab12...", "status": "completed",
  "dye_names": ["Dye-A","Dye-B"],
  "response_matrix": [[1.0,0.25],[0.4,1.0]],
  "result_object_key": "unmixed/S-2026-001/ab12.../9f3c.../unmixed.ome.tiff",
  "result_bytes": 131596, "width": 128, "height": 128,
  "saturated_pixels": 2, "valid_pixels": 16382, "error_message": null
}
```

### 3.3 饱和规则与 NNLS 语义

- 像素两路原始读数中**任一路达到 65535** 即判定探测器饱和，该像素的
  **两路解混结果都记为 NaN**（float32 IEEE 754），其余像素正常求解；
- 两路变量的 NNLS 枚举 4 个活动集（零解 / 两个单染料解 / 双染料最小二乘解），
  全部像素向量化批量计算，取残差最小且非负的可行解，结果严格 `≥ 0`；
- 若所有像素都饱和（无有效像素），返回 422，不产出结果文件。

### 3.4 查询与下载

```bash
curl http://localhost:8000/uploads/$UID/unmix          # 该上传的全部解混任务
curl http://localhost:8000/unmix-jobs/$JOB             # 单任务状态/统计
curl -OJ http://localhost:8000/unmix-jobs/$JOB/result  # 下载结果 OME-TIFF
```

下载产物保证：

- 两通道、`CYX`、**float32** OME-TIFF，OME-XML 中两个 `Channel` 的
  `Name` 分别为提交的两种染料名称；
- 饱和像素两路均为 NaN，可被下游分析程序按缺失值处理；
- 结果是独立对象（键前缀 `unmixed/`），**原件对象始终只读、逐字节不变**。

### 3.5 矩阵错误的明确报错

```text
含 NaN/Inf        -> 422 响应矩阵含非有限数（NaN/Inf），无法解混
条件数 > 1000     -> 422 响应矩阵条件数 ... 超过上限 1000，解混不稳定
形状非 2×2        -> 422 响应矩阵必须是 2×2
全部像素饱和      -> 422 ... 无有效像素可解混
原件读取失败      -> 502 storage_error
```

---

## 4. 状态机与崩溃恢复

```
                 PUT parts(幂等/409)
   open ┌──────────────────────────────┐
    │   │                              │
    │   ▼                              │
    │  (CAS open→completing)           │ POST abort (CAS open/completing→aborted)
    │ completing                       │
    │   │ 合并 S3 → 下载 → tifffile 核验
    │   ├─ 通过 → completed（开放原件/尺寸）
    │   └─ 失败 → invalid（删远端对象，不开放）
    └──→ aborted（清远端分片，拒迟到写入）
```

启动时 `recover_on_startup()` 对每个 `open / completing` 任务核实并续完：

| 崩溃窗口 | 重启后的行为 |
| --- | --- |
| 分片已写 S3、SQLite 未落库 | 与 S3 `ListParts` 对账；本地无哈希的片不冒充已接纳，按缺片返回，客户端**只补缺片**，不重传整图 |
| `CompleteMultipartUpload` 已成功、回执丢失/进程退出 | `HeadObject` 发现对象已存在且大小正确 → **跳过重复合并**，直接下载核验续完，不重复发布 |
| 核验阶段崩溃 | 重新下载核验；坏图删对象置 `invalid` |
| 无数据库归属的远端 multipart | 启动时作为孤儿上传 `AbortMultipartUpload` 清除 |
| 客户端重试 `complete` | `completed` 直接幂等返回；`completing` 走恢复式收尾，永久卡不住 |

任务在 SQLite 与 MinIO 数据卷中持久化，`docker compose down` 后再 `up` 仍可续传/恢复。

---

## 5. 故障演示（8 个场景）

`ENABLE_FAULTS=true` 时开放 `POST /dev/faults` 注入两个真实崩溃点
（SIGTERM，容器 `restart: unless-stopped` 自动拉起，等价主机重启）：

| 场景 | 验证点 |
| --- | --- |
| 1 小图全流程 | 合并核验后尺寸/原件才开放 |
| 2 大图中断续传 | 4 分片（末片 1199 字节），按 `missing_parts` 续传，原件逐字节一致 |
| 3 幂等与冲突 | 同片重传 200、异内容 409 且不污染 |
| 4 放弃 | 远端分片清除、迟到写入 410、无残件 |
| 5 `crash_after_part` | 第 3 片写 S3 后崩溃，重启只补缺片即完成 |
| 6 `crash_after_merge` | 合并后崩溃，重启续完；重试 complete 幂等、湖中仅一个对象 |
| 7 坏图 | 缺 OME-XML 的 TIFF 返回 422、对象删除、查询不开放 |
| 8 双染料解混 | NNLS 解混成功、饱和像素两路 NaN、float32 结果带染料通道名可下载、病态矩阵 422 |

```bash
docker compose exec api python scripts/demo.py
```

生产环境设置 `ENABLE_FAULTS=false`，故障端点返回 403。

无 Docker 的环境可用内存 S3 在真实 uvicorn HTTP 上跑通同一套演示脚本：

```bash
python scripts/smoke_local.py
```

---

## 6. 测试

```bash
python -m pytest          # 52 项：上传服务/HTTP 语义 + 解混矩阵/约束/NNLS/产物
```

覆盖：多分片并发、同片幂等/409、片长校验、完成/放弃 CAS 竞争（多轮）、
放弃清残件与迟到写入拒绝、无 OME 元数据/截断 TIFF/SHA256 不符三类坏图拦截、
两类崩溃窗口恢复、孤儿 multipart 清理、complete 幂等；
以及解混的矩阵有限性/条件数/形状校验、非 CYX/非两通道/非 uint16/超 16 MiB/
缺 OME 拦截、NNLS 对已知混合数据的还原与非负性、饱和像素两路 NaN、
全部饱和报错、原件读取失败 502、失败不留结果对象、产物 OME-TIFF 通道名与下载。

## 7. 目录结构

```
app/
  config.py            # 环境变量配置、8MiB/16MiB/条件数等常量
  db.py                # SQLite WAL schema、CAS 状态流转、解混任务表
  storage.py           # boto3 真实 S3 multipart 适配 + 对象写入
  image_validation.py  # 入湖 OME-XML + IFD 页目录核验、尺寸提取
  unmixing.py          # 矩阵校验、CYX/uint16 严格加载、向量化 NNLS、OME-TIFF 编码
  unmix_service.py     # 解混任务编排：下载原件 -> 解混 -> 结果回流入湖
  faults.py            # 故障注入（SIGTERM）
  service.py           # 上传生命周期/幂等/并发/放弃/启动恢复
  main.py              # FastAPI 路由与错误映射
scripts/
  generate_samples.py  # 无隐私合成 OME-TIFF（含双通道 CYX 解混样例）
  demo.py              # Docker 内 8 场景端到端演示
  smoke_local.py       # 无 Docker 本地 HTTP 冒烟
tests/                 # pytest 套件（内存 S3 模拟 multipart 语义）
docker-compose.yml     # 持久化 MinIO + API
```

## 8. 关键设计约束

- 单进程 + 每 `upload_id` 一把 `RLock`：同任务分片/完成/放弃/解混串行，不同任务不互斥；
- 网络 IO（S3 调用）不持数据库事务，SQLite 事务只覆盖本地状态变更；
- 所有“终态发布”以 SQLite 状态为准，S3 对象存在但状态未完成时一律不可读；
- 对象键含 `upload_id`，同一样本反复上传不会互相覆盖；解混结果独立成键，
  原件对象在整个解混流程中只读，任何失败路径都不修改或删除原件；
- 解混为纯 NumPy 向量化计算（两变量 NNLS 仅 4 个候选活动集），不引入 scipy，
  依赖体积与镜像构建保持轻量。
