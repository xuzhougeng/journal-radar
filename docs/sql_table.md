# 数据库表说明（Journal Monitor）

本文档概述 SQLite 数据库中的表及其在代码中的主要用途，便于排查数据流与功能对应关系。

> 入口模型定义：`app/models.py`

## subscriptions
**功能**：订阅源配置（RSS/Crossref 等）。

| 字段         | 字段说明                                  |
| ------------ | ----------------------------------------- |
| id           | 主键，自增                               |
| name         | 订阅名称（用于展示与分组）                |
| source_type  | 订阅类型：如 `rss`、`crossref`           |
| config       | 订阅源配置 JSON（如 RSS URL、ISSN 等）   |
| enabled      | 是否启用订阅                             |
| created_at   | 创建时间                                 |
| updated_at   | 更新时间                                 |

**代码中的作用**：
- 订阅源创建/编辑/删除/导入导出：`app/web/api.py`
- 页面展示与筛选：`app/web/routes.py`、`app/web/templates/subscriptions.html`
- 调度执行时读取启用的订阅：`app/runner.py`（`run_check`）
- 与 `entries` 关联（`Subscription.entries`）

## entries
**功能**：每次抓取到的条目（论文/文章）元数据。

| 字段             | 字段说明               |
|------------------|-----------------------|
| id               | 主键，自增            |
| subscription_id  | 所属订阅（外键）      |
| fingerprint      | 去重指纹（SHA256）    |
| title            | 标题                  |
| link             | 原文链接              |
| doi              | DOI（可为空）         |
| authors          | 作者（可为空，文本）  |
| abstract         | 摘要（可为空，文本）  |
| journal_name     | 期刊名（可为空）      |
| published_at     | 发表时间（可为空）    |
| fetched_at       | 抓取入库时间          |
| notified         | 是否已通知            |

**代码中的作用**：
- 保存新条目并去重：`app/runner.py`（`save_new_entries`，基于 `fingerprint`）
- 列表查询与筛选：`app/web/routes.py`、`app/web/api.py`、`app/web/templates/entries.html`
- 通知发送后标记 `notified`：`app/runner.py`
- 与 `entry_contents` / `entry_structures` / `entry_types` 一对一关联

## entry_contents
**功能**：条目链接页面的正文内容/元数据提取结果（解析/抓取）。

| 字段           | 字段说明                        |
| -------------- | ------------------------------- |
| id             | 主键，自增                      |
| entry_id       | 所属条目（外键，唯一）          |
| provider       | 解析提供方（如 `exa` 等）       |
| request_id     | 提供方请求 ID（可为空）         |
| status         | 解析状态（success/failed 等）   |
| url            | 解析后的最终 URL（可为空）      |
| title          | 解析出的标题（可为空）          |
| author         | 解析出的作者（可为空）          |
| text           | 解析正文（截断后）              |
| raw_path       | 原始 JSON 路径（相对路径，可为空）|
| cost_total     | 调用总成本（可为空）            |
| cost_text      | 文本成本（可为空）              |
| search_time_ms | 解析耗时（可为空）              |
| fetched_at     | 内容抓取时间                    |

**代码中的作用**：
- 拉取正文与元数据并写库：`app/runner.py`（`fetch_parse_content_for_entries`）
- 接口预览与重试写入：`app/web/api.py`（`/entries/{id}/parse/preview` 等）
- 作为 `entry_structures` 的输入依赖（LLM 结构化前置条件）

## entry_structures
**功能**：LLM 生成的结构化摘要与站点类型判断结果。

| 字段             | 字段说明                                 |
|------------------|------------------------------------------|
| id               | 主键，自增                               |
| entry_id         | 所属条目（外键，唯一）                   |
| provider         | LLM 提供方标识                           |
| model            | 使用的模型名                             |
| base_url         | LLM API base URL（调试用）               |
| site_type        | 站点类型（paper / journal / news / ... / other） |
| site_type_reason | 站点类型判定原因（可为空）               |
| summary          | 摘要文本（可为空）                       |
| raw_json         | 原始 LLM 响应 JSON（可为空）             |
| status           | 处理状态（success / failed / pending）    |
| error_message    | 失败原因（可为空）                       |
| created_at       | 创建时间                                 |
| updated_at       | 更新时间                                 |

**代码中的作用**：
- 内容结构化抽取并写库：`app/runner.py`（`fetch_llm_structure_for_entries`）
- 接口预览与重试写入：`app/web/api.py`（`/entries/{id}/llm/preview` 等）
- 与 `entries` 一对一关联，用于侧栏摘要展示（前端调用接口）
- LLM 类型写入后同步到 `entry_types.llm_type`

## entry_types
**功能**：统一的文章类型分类表，汇总 Parse 推断、LLM 推断、用户手动覆盖三方类型。

| 字段              | 字段说明                                                    |
|-------------------|-------------------------------------------------------------|
| id                | 主键，自增                                                  |
| entry_id          | 所属条目（外键，唯一）                                      |
| effective_type    | 最终类型（按优先级计算：user > llm > parse）                |
| parse_type        | Parse 启发式推断的类型（可为空）                            |
| parse_reason      | Parse 类型推断原因（可为空）                                |
| parse_updated_at  | Parse 类型更新时间（可为空）                                |
| llm_type          | LLM 推断的类型（可为空）                                    |
| llm_reason        | LLM 类型推断原因（可为空）                                  |
| llm_updated_at    | LLM 类型更新时间（可为空）                                  |
| user_type         | 用户手动覆盖的类型（可为空）                                |
| user_reason       | 用户覆盖原因（可为空）                                      |
| user_updated_at   | 用户类型更新时间（可为空）                                  |
| created_at        | 创建时间                                                    |
| updated_at        | 更新时间                                                    |

**有效类型枚举**：`paper` / `journal` / `news` / `blog` / `docs` / `repository` / `forum` / `product` / `dataset` / `other`

**代码中的作用**：
- 类型计算与同步逻辑：`app/entry_type.py`（`resolve_effective_type`、`update_parse_type`、`update_llm_type`、`update_user_type`）
- Parse 内容写入后自动推断类型：`app/runner.py`、`app/web/api.py`（`/api/parse/fetch/{id}`）
- LLM 结构化写入后同步类型：`app/runner.py`、`app/web/api.py`（`/api/llm/struct/{id}`）
- 用户手动设置/清除类型：`app/web/api.py`（`PATCH /api/entries/{id}/type`、`DELETE /api/entries/{id}/type/user`）
- 前端展示与编辑：`app/web/templates/entries.html`（卡片 badge、预览侧栏类型编辑器）
- 与 `entries` 一对一关联（`Entry.type_info`）

## check_runs
**功能**：每次调度/手动检查的运行记录和统计。

| 字段                 | 字段说明                                   |
|----------------------|--------------------------------------------|
| id                   | 主键，自增                                 |
| started_at           | 开始时间                                   |
| completed_at         | 完成时间（可为空）                         |
| status               | 运行状态（running / completed / failed 等）|
| total_subscriptions  | 参与订阅数量                               |
| total_new_entries    | 新增条目数量                               |
| total_notifications  | 发送通知数量                               |
| error_message        | 错误信息（可为空）                         |

**代码中的作用**：
- 检查任务生命周期管理：`app/runner.py`（`run_check`）
- 首页与状态接口统计展示：`app/web/routes.py`、`app/web/api.py`

## notifications
**功能**：推送通知发送记录（成功或失败）。

| 字段             | 字段说明               |
|------------------|----------------------|
| id               | 主键，自增            |
| check_run_id     | 所属检查运行（可为空）|
| subscription_id  | 所属订阅（可为空）    |
| title            | 通知标题              |
| body             | 通知正文              |
| url              | 通知链接（可为空）    |
| sent_at          | 发送时间              |
| success          | 是否成功              |
| error_message    | 失败原因（可为空）    |

**代码中的作用**：
- 发送通知后落库：`app/notifier/bark.py`（`_record_notification`）
- 关联 `check_runs` / `subscriptions` 以便追踪来源

## app_config
**功能**：应用配置（运行配置 + 认证配置），单行表（`id=1`）。

| 字段         | 字段说明                                   |
|--------------|------------------------------------------|
| id           | 主键，固定为 1                           |
| runtime_json | 运行时配置 JSON（Bark / Exa / 调度等）    |
| auth_json    | 认证配置 JSON（密码 hash / salt / 会话密钥等） |
| created_at   | 创建时间                                 |
| updated_at   | 更新时间                                 |


**代码中的作用**：
- 初始化默认配置与首次管理员密码生成：`app/config_store.py`（`ensure_config`）
- 读取/更新运行配置与认证配置：`app/config_store.py`（`load_config_to_cache`、`update_runtime_config`、`update_auth_password` 等）
- 配置缓存与运行时读取：`app/config.py`

---

如需补充表字段说明（含索引/约束），或想加“数据流示意图”，告诉我需要的层级即可。

## tags
**功能**：用户定义的标签，用于分类和组织 Entry。

| 字段       | 字段说明                                   |
|------------|-------------------------------------------|
| id         | 主键，自增                                |
| name       | 标签显示名称（保留原始大小写）              |
| key        | 规范化后的唯一键（小写、去空白）            |
| created_at | 创建时间                                  |
| updated_at | 更新时间                                  |

**代码中的作用**：
- 标签列表查询：`app/web/api.py`（`GET /api/tags`）
- 标签页面展示：`app/web/routes.py`（`GET /tags`）、`app/web/templates/tags.html`
- 与 `entries` 通过 `entry_tags` 多对多关联

## entry_tags
**功能**：Entry 与 Tag 的多对多关联表。

| 字段     | 字段说明                |
|----------|------------------------|
| entry_id | Entry ID（联合主键）    |
| tag_id   | Tag ID（联合主键）      |

**代码中的作用**：
- 关联表在 `app/models.py` 中定义
- 标签读取/写入：`app/web/api.py`（`GET/PUT /api/entries/{id}/tags`）
- 按标签筛选 Entries：`app/web/routes.py`（`entries_page` 的 `tag_filter` 参数）
