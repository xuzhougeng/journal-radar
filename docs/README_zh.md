# Journal Radar

一个用于监测期刊更新（RSS/Atom 与 Crossref）并通过 Bark 推送到手机的 Web 应用。

## 功能特性

- **多源支持**：RSS/Atom 与 Crossref API
- **智能去重**：基于 DOI 或内容指纹避免重复通知
- **定时检查**：可配置每日检查时间
- **推送通知**：通过 Bark 推送到手机
- **Web 管理界面**：管理订阅、查看条目、配置设置
- **Web 端配置**：所有设置可在 Web UI 中修改

## 快速开始

### 1. 安装依赖

```bash
# 创建并激活虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或：venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行

```bash
# 开发环境
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 生产环境
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. 首次启动

首次启动时，应用会：
1. 自动创建数据库
2. 生成随机管理员密码
3. 在控制台输出该密码

**在启动日志中可看到类似内容：**

```
============================================================
FIRST TIME SETUP - ADMIN CREDENTIALS
============================================================
Username: admin
Password: <randomly-generated-password>
============================================================
Please save this password! It will not be shown again.
You can change it later in Settings > Reset Password.
============================================================
```

访问 `http://localhost:8000`，使用控制台显示的账号密码登录。

### 4. Web UI 配置

登录后进入 **Settings** 配置：

- **Bark 推送**：填写 Bark App 的 device key
- **Exa AI**（可选）：内容抽取 API key
- **Schedule**：设置检查时间与时区
- **Push Behavior**：配置合并推送策略
- **HTTP Client**：设置超时与 User-Agent
- **Authentication**：修改用户名或密码

## 配置说明

所有配置均存储在数据库并通过 Web UI 管理，无需 `.env` 文件。

### 默认值

首次启动默认值如下：

| Setting | Default |
|---------|---------|
| Check Time | 08:00 |
| Timezone | Asia/Shanghai |
| Bark Server | https://api.day.app |
| Request Timeout | 30 seconds |
| Merge Notifications | Yes |
| Max Entries per Message | 10 |

可在 `/settings` 页面修改。

## API 接口

除 `/healthz` 和 `/api/entries` 外，其它接口均需认证。

- `GET /healthz` - 健康检查
- `GET /` - 仪表盘（需登录）
- `GET /entries` - 条目列表（公开）
- `GET /settings` - 设置页（需登录）
- `GET /api/settings` - 获取设置
- `PUT /api/settings` - 更新设置
- `POST /api/settings/password` - 设置新密码
- `POST /api/settings/password/rotate` - 生成随机密码
- `GET /api/subscriptions` - 订阅列表
- `POST /api/subscriptions` - 新增订阅
- `DELETE /api/subscriptions/{id}` - 删除订阅
- `POST /api/check/run` - 手动触发检查
- `POST /api/push/test` - 测试 Bark 推送
- `GET /api/entries` - 最近条目（公开）
- `GET /api/runs` - 历史运行记录

## 订阅类型

### RSS/Atom

添加任意期刊的 RSS/Atom 链接，系统会定期检查更新。

### Crossref

通过 ISSN 添加期刊，系统会查询 Crossref API 获取最新论文。

## 项目结构

```
journal-monitor/
├── app/
│   ├── main.py          # FastAPI 入口
│   ├── config.py        # 配置模型与缓存
│   ├── config_store.py  # 数据库配置持久化
│   ├── db.py            # 数据库初始化
│   ├── models.py        # ORM 模型
│   ├── scheduler.py     # APScheduler 定时任务
│   ├── runner.py        # 检查运行逻辑
│   ├── exa_ai.py        # Exa AI 内容抽取
│   ├── sources/
│   │   ├── base.py      # Source 基类
│   │   ├── rss.py       # RSS 实现
│   │   └── crossref.py  # Crossref 实现
│   ├── notifier/
│   │   └── bark.py      # Bark 推送
│   └── web/
│       ├── routes.py    # Web 页面
│       ├── api.py       # API 接口
│       ├── auth.py      # 认证逻辑
│       ├── templates/   # Jinja2 模板
│       └── static/      # 静态资源
├── data/                # SQLite 数据库
├── requirements.txt
└── README.md
```

## 示例订阅

### RSS

| Journal | Feed URL |
|---------|----------|
| Nature | `https://www.nature.com/nature.rss` |
| Science | `https://www.science.org/rss/news_current.xml` |
| Cell | `https://www.cell.com/cell/rss/current` |
| PNAS | `https://www.pnas.org/rss/current.xml` |
| PLoS ONE | `https://journals.plos.org/plosone/feed/atom` |
| arXiv (cs.AI) | `http://export.arxiv.org/rss/cs.AI` |

### Crossref (ISSN)

| Journal | ISSN |
|---------|------|
| Nature | `0028-0836` |
| Science | `0036-8075` |
| Cell | `0092-8674` |
| PNAS | `0027-8424` |
| Nature Medicine | `1078-8956` |
| New England Journal of Medicine | `0028-4793` |

### Web UI 添加订阅

1. 登录应用
2. 进入 **Subscriptions**
3. 点击 **Add Subscription**
4. 选择订阅类型（RSS 或 Crossref）
5. 输入 Feed URL 或 ISSN
6. 点击 **Add**

## 故障排查

### 常见问题

1. **忘记管理员密码**
   - 推荐：使用 CLI 重置（数据保留）
     - 停止应用
     - 执行 `python -m app.cli reset-password`
     - 重启应用并使用新密码登录
   - 备选（破坏性）：删除本地数据重新生成
     - 删除数据库文件（`data/journal_monitor.sqlite3`）
     - 删除会话密钥（`data/.session_secret`）
     - 重启应用生成新账号

2. **Bark 推送不工作**
   - 在 Settings 填写 Bark 设备 key
   - 使用 “Send Test Notification” 按钮测试
   - 确认手机 Bark 已配置

3. **RSS 更新不及时**
   - 一些站点有缓存，更新可能延迟
   - 检查链接在浏览器中是否可访问
   - 查看日志中的 HTTP 错误

4. **Crossref 返回为空**
   - 校验 ISSN 是否正确（格式：`1234-5678`）
   - 可在 crossref.org 搜索验证期刊
   - 期刊必须注册在 Crossref

### 日志

日志输出到 stdout，可在控制台查看错误与运行状态。

## 安全说明

- 管理员密码使用 PBKDF2-SHA256（600,000 次迭代）哈希
- Session cookie 使用随机密钥签名
- 敏感 API key（Bark/Exa）存储在数据库中（不会写入日志）
- 认证始终开启，没有匿名管理员模式

## License

MIT
