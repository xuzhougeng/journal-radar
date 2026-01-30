# 如何部署

> 不建议在CentOS 7里面部署项目，你只会收获痛苦


推荐流程：**本地构建镜像** → **上传到服务器** → **服务器加载镜像** → 用 **docker compose / systemd** 管理运行。


该方案把运行环境（Python/SQLite）都放到容器里，服务器无需安装 Python/SQLite。
数据库与所有运行数据仍然落在项目目录下的 `data/`（通过 volume 挂载持久化）。


## 服务器上安装docker

```bash
sudo yum install -y yum-utils device-mapper-persistent-data lvm2

sudo yum-config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo

sudo yum install -y docker-ce docker-ce-cli containerd.io

sudo systemctl start docker
sudo systemctl enable docker
```

查看版本

```bash
sudo docker version
```


## 本地构建Docker镜像


在项目根目录执行（自行替换 tag，例如 `20260130`）：

```bash
docker build -t journal-monitor:20260130 .
```

本地测试（建议先验证能启动再导出上传）：

```bash
mkdir -p data
docker run --rm -it \
  -p 8000:8000 \
  -e HTTPS_ONLY=false \
  -v "$(pwd)/data:/data" \
  journal-monitor:20260130
```

验证健康检查：

```bash
curl -sS http://127.0.0.1:8000/healthz
```

导出镜像为 tar 包：

```bash
docker save -o journal-monitor_20260130.tar journal-monitor:20260130
```

## 上传到服务器并加载

使用scp或者FileZilla等工具

```bash
scp journal-monitor_20260130.tar root@<YOUR_SERVER>:/tmp/
```

## 服务器加载镜像

在服务器上加载：

```bash
cd /tmp
sudo docker load -i journal-monitor_20260130.tar
sudo docker images | grep journal-monitor
```

## 服务器部署

准备运行目录（服务器）

建议放到 `/opt/journal-monitor`：

```bash
sudo mkdir -p /opt/journal-monitor
sudo mkdir -p /opt/journal-monitor/data
```

在 `/opt/journal-monitor/`，新建 `docker-compose.yml` ，内容如下

```yaml
services:
  journal-monitor:
    image: journal-monitor:20260130
    container_name: journal-monitor-20260130
    restart: unless-stopped
    environment:
      - HTTPS_ONLY=false
      # - LOG_LEVEL=info
      # - TZ=Asia/Shanghai
    ports:
      - "127.0.0.1:8000:8000"
    volumes:
      - ./data:/data
```

### 启动Docker服务

```bash
cd /opt/journal-monitor
docker compose up -d
docker compose ps
```

首次启动同样会在日志里打印管理员账号/密码。

```bash
sudo docker compose logs -f journal-monitor

## 日志信息
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | #============================================================
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | FIRST TIME SETUP - ADMIN CREDENTIALS
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | #============================================================
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | Username: admin
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | Password: cR5dj4xBdqoGlkEg
#journal-monitor-20260130  | 2026-01-30 03:27:09.958 | INFO | app.config_store | ============================================================
```


数据路径:

- 数据库与 session secret 在 `data/` 目录（默认自动创建）
- 配置通过 Web UI 管理，无需 `.env`
- 建议定期备份 `data/` 目录

## 反向代理（Nginx / BT 面板）

容器默认只监听 `127.0.0.1:8000`，建议在宿主机 Nginx/宝塔里做反代到：

- `http://127.0.0.1:8000`

如果你是 HTTPS（强烈建议），保持默认即可；如果你明确要纯 HTTP，请确保容器里设置了 `HTTPS_ONLY=false`（`docker-compose.yml` 已默认设置）。



## 网页配置

步骤如下:

1. /subscriptions 页面 添加你感兴趣的期刊
1. /Dashboard 点击Run Check Now
1. /settings 按需配置


Setting配置内容
- Change Password 修改密码，更好记一点
- Parse/Content Fetching: 填写EXA API KEY. 依赖于 exa.AI 申请API. 如果
- Bark Push Notification: iOS系统的Bark提示
- LLM Structured Extraction: 通过LLM进行内容的解析
