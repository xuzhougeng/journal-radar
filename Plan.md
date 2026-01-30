# 期刊监督网站

## Goal

主任务: 开发一个可以方便我查看文献进展的网站

避免要做的:

- 多用户支持，这个绝对不做！

## TODO

- 支持系统提醒系统


## Milestone

- 2026-01-30 
 - 撰写了部署指南, 部署第一版本到腾讯云服务器上
 - 优化手机端的Entry页面
 - 调整entry的展示逻辑，未登录用户可以有Prview的权限

- 2026-01-29 
  - 使用LLM对网页进行解析，总结，优化了preview页面的展示效果
  - 解决Exa.AI 网页爬取失败的问题
  - 增加tag功能，允许用户对Entry里的文章进行标签化，并有一个页面支持按照标签筛选

- 2026-01-26 去掉了.env 而是通过数据库进行初始化
- 2026-01-25 项目初始化, 完成了第一版

## Note

千万不要在CentOS尝试手动部署，老老实实用Docker!

中间件的 SessionMiddleware 的 https_only 会导致cookie无法在http协议缓存，导致每次登录都失败。

```py
app.add_middleware(
    SessionMiddleware,
    secret_key=get_session_secret(),
    same_site="lax",
    https_only=StaticConfig.SESSION_COOKIE_HTTPS_ONLY,
)
```

Exa.AI爬取失败的时候，可以用设置参数 `livecrawl = "fallback", livecrawl_timeout = 15000`



