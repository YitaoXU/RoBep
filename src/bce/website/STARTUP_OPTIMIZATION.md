# BCE 服务器启动优化指南

## 🐛 启动慢的常见原因

### 主要延迟来源
1. **模型加载**：ReCEP模型文件较大（几百MB-几GB），首次加载需要时间
2. **CUDA初始化**：GPU设备检测和初始化
3. **ESM-C连接**：连接到 Evolutionary Scale API 服务器
4. **依赖加载**：PyTorch、Biotite等大型库的首次导入
5. **磁盘I/O**：从慢速存储设备读取模型文件

## 🚀 优化解决方案

### 1. 使用模型预加载（推荐）

使用新的 `--preload` 功能在服务器启动时预加载模型：

```bash
# 启动时预加载模型（默认开启）
python run_server.py --host 0.0.0.0 --port 8001

# 显式启用预加载
python run_server.py --host 0.0.0.0 --port 8001 --preload

# 指定GPU设备ID
python run_server.py --host 0.0.0.0 --port 8001 --device-id 1

# 使用自定义模型路径
python run_server.py --host 0.0.0.0 --port 8001 --model-path /path/to/your/model.bin

# 跳过预加载（原有行为）
python run_server.py --host 0.0.0.0 --port 8001 --no-preload
```

### 2. 诊断启动问题

运行诊断脚本来识别具体的延迟来源：

```bash
cd src/bce/website
python diagnose_startup.py
```

诊断脚本会测试：
- 模块导入时间
- CUDA初始化时间  
- 模型加载时间
- ESM-C连接时间
- 磁盘I/O性能
- 系统资源状态

### 3. 环境优化建议

#### 硬件优化
- **使用SSD存储**：提高模型文件读取速度
- **充足GPU内存**：避免CPU-GPU数据传输延迟
- **快速网络连接**：确保ESM-C API稳定连接

#### 软件优化
- **预热环境**：启动后运行一次小型预测来预热所有组件
- **缓存策略**：使用本地embeddings缓存减少API调用
- **进程复用**：在生产环境使用多worker模式

## 📊 性能对比

### 启动时间对比（典型场景）

| 启动方式 | 首次预测延迟 | 说明 |
|---------|-------------|------|
| 普通启动 | 15-30秒 | 模型在首次请求时加载 |
| 预加载启动 | 2-5秒 | 模型已预加载，快速响应 |

### 各组件典型加载时间

| 组件 | 正常时间 | 异常时间 | 可能原因 |
|------|---------|---------|----------|
| CUDA初始化 | < 1秒 | > 5秒 | GPU驱动问题 |
| 模型加载 | 3-8秒 | > 15秒 | 慢速存储/内存不足 |
| ESM-C连接 | 1-3秒 | > 10秒 | 网络延迟/API限制 |

## 🔧 故障排除

### 常见问题

1. **CUDA相关错误**
   ```bash
   # 检查CUDA状态
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **内存不足**
   ```bash
   # 使用CPU模式
   python run_server.py --device-id -1
   ```

3. **模型文件不存在**
   ```bash
   # 检查模型路径
   ls -la /path/to/model/file.bin
   ```

4. **网络连接问题**
   ```bash
   # 测试ESM-C连接
   curl -I https://forge.evolutionaryscale.ai
   ```

### 环境变量配置

```bash
# 设置GPU设备
export CUDA_VISIBLE_DEVICES=0

# 设置模型路径
export BCE_MODEL_PATH="/path/to/your/model.bin"

# 设置ESM token
export ESM_TOKEN="your_token_here"

# 启用调试日志
export BCE_LOG_LEVEL="debug"
```

## 📈 生产环境建议

### 服务器部署优化

```bash
# 生产环境启动命令
python run_server.py \
  --host 0.0.0.0 \
  --port 8001 \
  --workers 1 \
  --log-level info \
  --preload \
  --device-id 0
```

### 监控和维护

1. **定期检查GPU状态**
2. **监控内存使用**
3. **检查磁盘空间**
4. **监控API调用配额**

### 自动化脚本

创建启动脚本 `start_server.sh`：

```bash
#!/bin/bash
cd /path/to/BCE_prediction/src/bce/website
conda activate ReCEP

# 检查环境
python diagnose_startup.py

# 启动服务器
python run_server.py --host 0.0.0.0 --port 8001 --preload
```

## 💡 最佳实践

1. **首次部署**：先运行诊断脚本确认环境正常
2. **定期维护**：定期检查模型文件和缓存状态
3. **监控日志**：关注启动日志中的警告信息
4. **备份方案**：准备CPU模式作为GPU故障时的备选
5. **版本管理**：保持PyTorch和CUDA版本兼容性

通过这些优化措施，服务器启动后的首次预测响应时间可以从15-30秒降低到2-5秒。 