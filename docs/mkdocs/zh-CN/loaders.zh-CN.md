# 文档加载器

**Languages:** [English](../loaders.md) | 简体中文

`core/documentloaders` 提供 RAG 流水线最常用的三个加载器：HTML（goquery）、
HTTP 网页、PDF（ledongthuc/pdf）。三者都实现惰性 `LazyLoader` 接口，可与共享
的 `Load` / `LoadAndSplit` 辅助函数及递归分割器组合。

## 安装

```bash
go get github.com/projanvil/langchain-golang
```

## HTML 加载器

把 HTML 文档变成一个可读正文 `Document`（剥离脚本、样式与标签；标题折叠成
行）：

```go
import "github.com/projanvil/langchain-golang/core/documentloaders"

loader := documentloaders.NewHTMLLoader(strings.NewReader(htmlSource),
    documentloaders.WithMetadata(map[string]any{"source": "in-app page"}))
docs, err := documentloaders.Load(ctx, loader) // docs[0].Metadata["title"] 携带 <title>
```

## Web 加载器

经 HTTP 抓取页面并走共享的 HTML 提取——Python `WebBaseLoader` 的对应物
（相同的默认头模板；`USER_AGENT` 环境变量可覆盖默认 User-Agent）。元数据
沿用上游 `_build_metadata`：`source`、`title`、`description`、`language`：

```go
web := documentloaders.NewWebLoader(&http.Client{Timeout: 15 * time.Second},
    documentloaders.WithMaxBodyBytes(1 << 20)) // 1 MiB 上限，默认 10 MiB

// 一次性抓取：
docs, err := web.Load(ctx, "https://example.com/docs")

// 或先绑定 URL，再用共享的惰性辅助函数：
bound := web.ForURL("https://example.com/docs")
docs, err = documentloaders.Load(ctx, bound)
```

## PDF 加载器

逐页提取文本——每页一个文档，元数据带 0 起始的 `page` 与 `total_pages`
（Python `PyPDFLoader` parity；加密或畸形 PDF 会显式报错）：

```go
f, _ := os.Open("report.pdf")
defer f.Close()
loader := documentloaders.NewPDFLoader(f) // 任意 io.Reader
docs, err := documentloaders.Load(ctx, loader)
```

## 一步完成加载 + 分割

常见的 RAG 入口——先加载，再切成检索尺寸的块：

```go
splitter, _ := textsplitters.NewRecursiveCharacter(nil, false, textsplitters.Config{
    ChunkSize:    1000,
    ChunkOverlap: 200,
})
chunks, err := documentloaders.LoadAndSplit(ctx, loader, splitter)
store.AddDocuments(ctx, chunks)
```

`LoadAndSplit` 传 `nil` 分割器时会使用经
`RegisterDefaultTextSplitterFactory` 注册的默认工厂（未注册时报错）。

## 文件与 blob

`NewBlobFromPath(path, mimetype, metadata)` / `NewBlobFromData(...)` 提供带
来源标记的惰性输入，适合文件遍历式索引器；`blob.Reader()` 可喂给三个加载
器中的任何一个。

## 切换点

- **加载器**：三者共享 `LazyLoader` 形态——HTML 换成 web 或 PDF，分割/
  嵌入/存储阶段零改动。
- **分割器**：`LoadAndSplit` 接受任意 `TextSplitter`；默认递归字符分割，
  也可换基于 token 的分割器。
- **抓取策略**：向 `NewWebLoader` 注入自己的 `*http.Client`（超时、代理、
  认证）；用 `WithMaxBodyBytes` 封顶不受信的响应体。
