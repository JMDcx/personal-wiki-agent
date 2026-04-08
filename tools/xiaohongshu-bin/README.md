将本机小红书二进制放到这里：

- `xiaohongshu-mcp-darwin-arm64`
- `xiaohongshu-login-darwin-arm64`

示例：

```bash
chmod +x tools/xiaohongshu-bin/xiaohongshu-*-darwin-arm64
./tools/xiaohongshu-bin/xiaohongshu-login-darwin-arm64
./tools/xiaohongshu-bin/xiaohongshu-mcp-darwin-arm64
```

项目通过 `.env` 中的 `XHS_MCP_URL=http://127.0.0.1:18060/mcp` 连接本地服务。
