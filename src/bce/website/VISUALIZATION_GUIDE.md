# 可视化功能使用指南

## 🎯 问题修复说明

经过修复，网站的可视化功能现在已经完全正常工作：

### ✅ 修复的问题
- **Surface模式显示问题**：现在可以正确显示分子表面
- **Sphere球形区域显示问题**：现在可以正确显示球形预测区域
- **颜色渐变问题**：Surface模式下的概率渐变色现在正常工作

## 🖥️ 使用方法

### 1. 启动服务器
```bash
cd src/bce/website
conda activate ReCEP
python run_server.py --host 0.0.0.0 --port 8001
```

### 2. 访问网站
打开浏览器，访问：`http://localhost:8001`

### 3. 预测设置
- 输入PDB ID（例如：5I9Q）
- 选择链ID（例如：A）
- 点击"Predict Epitopes"

### 4. 查看结果 - 可视化选项

#### 🎨 Display Mode（显示模式）
- **Predicted Epitopes**：高亮显示预测的epitope残基
- **Probability Gradient**：按预测概率显示渐变色
- **Top-k Regions**：显示top-k预测区域，每个区域不同颜色

#### 🧬 Representation（表示方式）
- **Cartoon**：蛋白质二级结构表示
- **Surface**：✅ 分子表面（已修复）
- **Stick**：原子棒状表示
- **Sphere**：空间填充模型

#### 🔮 Show Spherical Regions
- 勾选此选项可显示球形预测区域
- ✅ 球形区域显示功能已完全修复
- 每个区域用不同颜色的透明球体表示

## 📊 修复后的功能特点

### Surface模式
- 🎯 **正确渲染**：分子表面现在能正确显示
- 🌈 **颜色支持**：支持所有显示模式的颜色映射
- 📈 **概率渐变**：在Probability Gradient模式下显示平滑的颜色过渡
- 🎨 **区域着色**：在Top-k Regions模式下每个区域有不同颜色

### Sphere球形区域
- 🔍 **精确定位**：球形区域精确对应预测的中心残基
- 📏 **准确半径**：使用实际预测半径（默认19.0 Å）
- 🎨 **颜色区分**：每个区域使用不同颜色便于区分
- 👁️ **透明度**：适当的透明度设置不会遮挡蛋白质结构

## 🧪 测试验证

所有功能都通过了自动化测试：
- ✅ 数据结构完整性检查
- ✅ Surface模式兼容性验证
- ✅ Sphere显示数据准备验证
- ✅ 颜色渐变范围检查

## 🎮 交互操作

### 鼠标控制
- **左键拖拽**：旋转视角
- **滚轮**：缩放
- **右键拖拽**：平移

### 控制按钮
- **Update Visualization**：应用新的可视化设置
- **Reset View**：重置视角到初始状态
- **Save Image**：保存当前视图为PNG图片

## 🔧 技术实现

### 后端修复
- 在`prepare_visualization_data`函数中添加了radius信息
- 确保`top_k_regions`包含所有必要的字段

### 前端修复
- 修复了3Dmol.js的surface渲染逻辑
- 实现了完整的sphere显示功能
- 改进了不同模式下的颜色处理

## 📋 使用建议

1. **最佳体验**：推荐使用Chrome或Firefox浏览器
2. **性能优化**：对于大型蛋白质，建议先使用Cartoon模式查看整体结构
3. **Surface模式**：适合查看epitope的表面可及性
4. **Sphere模式**：适合理解预测区域的空间分布
5. **组合使用**：可以同时开启Surface和Sphere模式获得最佳视觉效果

## 🎯 预期效果

修复后，您应该能够：
- 🔍 在Surface模式下看到完整的分子表面
- 🌈 在Probability Gradient模式下看到平滑的颜色过渡
- 🔮 在勾选"Show Spherical Regions"后看到透明的球形区域
- 🎨 在Top-k Regions模式下看到不同颜色的区域高亮

如果仍有问题，请检查浏览器控制台是否有JavaScript错误。 