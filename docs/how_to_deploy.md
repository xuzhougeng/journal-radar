# 如何部署

## CentOS 7 搭配宝塔面板

通过宝塔面板安装python 3.12


获取代码:

从 <https://github.com/xuzhougeng/journal-radar.git> 下载zip文件，上传到服务器


创建虚拟环境并安装依赖

```bash
/www/server/pyporject_evn/versions/3.12.0/bin/python3  -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

CentOS 7的机器安装 greenlet 编译阶段会报错，只能安装二进制

```bash
python -m pip install --only-binary=:all: greenlet
```

先确保能够正常的启动，首次启动会在控制台打印管理员账号与密码。

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

会有如下输出，其中Password是密码

```bash
2026-01-30 09:44:47.346 | INFO | app.config_store | FIRST TIME SETUP - ADMIN CREDENTIALS
2026-01-30 09:44:47.346 | INFO | app.config_store | ============================================================
2026-01-30 09:44:47.346 | INFO | app.config_store | Username: admin
2026-01-30 09:44:47.346 | INFO | app.config_store | Password: vou3kC9tcNhmmdKP
2026-01-30 09:44:47.346 | INFO | app.config_store | ============================================================
```

如果没看到可以重置密码

```bash
python -m app.cli reset-password
```


## 用 systemd 管理进程

创建服务文件：

```bash
sudo tee /etc/systemd/system/journal-radar.service >/dev/null <<'EOF'
[Unit]
Description=Journal Radar
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/journal-radar
Environment=HTTPS_ONLY=false
ExecStart=/opt/journal-radar/.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
Restart=always
RestartSec=3
User=www
Group=www

[Install]
WantedBy=multi-user.target
EOF
```

建议将代码放到 `/opt/journal-radar`，并赋予权限：

```bash
sudo mkdir -p /opt/journal-radar
sudo rsync -a --delete ./ /opt/journal-radar/
sudo chown -R www:www /opt/journal-radar
```

启动服务：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now journal-radar
sudo systemctl status journal-radar
```
重置密码后需要重启服务

```bash
sudo systemctl restart journal-radar
```

查看日志：

```bash
sudo journalctl -u journal-radar -f
```

## 数据与配置

- 数据库与 session secret 在 `data/` 目录（默认自动创建）
- 配置通过 Web UI 管理，无需 `.env`
- 建议定期备份 `data/` 目录

## Docker / Docker Compose 部署（推荐，解决 CentOS 7 旧 SQLite 问题）

该方案把运行环境（Python/SQLite）都放到容器里，宿主机只需要 Docker。
数据库与所有运行数据仍然落在项目目录下的 `data/`（通过 volume 挂载持久化）。

### 1) 准备目录

建议放到 `/opt/journal-monitor`：

```bash
sudo mkdir -p /opt/journal-monitor
sudo rsync -a --delete ./ /opt/journal-monitor/
sudo mkdir -p /opt/journal-monitor/data
sudo chown -R $USER:$USER /opt/journal-monitor
cd /opt/journal-monitor
```

### 2) 启动（构建并后台运行）

```bash
docker compose up -d --build
docker compose ps
```

首次启动同样会在日志里打印管理员账号/密码：

```bash
docker compose logs -f journal-monitor
```

### 3) 反向代理（Nginx / BT 面板）

容器默认只监听 `127.0.0.1:8000`，建议在宿主机 Nginx/宝塔里做反代到：

- `http://127.0.0.1:8000`

如果你是 HTTPS（强烈建议），保持默认即可；如果你明确要纯 HTTP，请确保容器里设置了 `HTTPS_ONLY=false`（`docker-compose.yml` 已默认设置）。

### 4) 用 systemd 托管 docker compose（可选）

创建 `/etc/systemd/system/journal-monitor.service`：

```bash
sudo tee /etc/systemd/system/journal-monitor.service >/dev/null <<'EOF'
[Unit]
Description=Journal Monitor (Docker Compose)
After=network.target docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=/opt/journal-monitor
RemainAfterExit=yes
ExecStart=/usr/bin/docker compose up -d --build
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF
```

启用并启动：

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now journal-monitor
sudo systemctl status journal-monitor
```

查看日志：

```bash
cd /opt/journal-monitor
docker compose logs -f journal-monitor
```

## 通过BT配置 Nginx 反向代理


## 网页配置

首先到 /subscriptions 页面 添加你感兴趣的期刊

接着到 exa.AI 申请API

到 /settings 页面的Parse/Content Fetching 填写EXA API KEY.

到Dashboard 点击Run Check Now

- Parse/Content Fetching

- Bark Push Notification

