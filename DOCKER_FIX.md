# RTMScore Dockerfile 依赖问题修复

## 问题描述
原始 Dockerfile 在构建时遇到以下错误：
```
ERROR: Could not find a version that satisfies the requirement numpy>=2.0.0rc1
```

## 根本原因
1. Python 3.8 不支持 numpy 2.0+
2. 某些包（如最新版 MDAnalysis）要求 numpy 2.0+
3. 包版本未锁定，导致安装了不兼容的版本

## 解决方案

### 修复的依赖版本
```
numpy: 1.20.3 - 1.23.x (Python 3.8 兼容)
scipy: 1.6.2 - 1.9.x
pandas: 1.3.0 - 1.x
scikit-learn: 0.24.2 - 1.2.x
matplotlib: 3.4.3 - 3.6.x
seaborn: 0.11.2 - 0.12.x
rdkit-pypi: 2021.9.5.1 (固定版本)
biopandas: 0.4.1 (固定版本)
MDAnalysis: 2.0.0 (固定版本)
ProDy: 2.1.0 (固定版本)
Cython: <3.0 (避免兼容性问题)
```

### Dockerfile 改进
1. **分层安装**: 先安装核心依赖（numpy, scipy），再安装其他包
2. **版本锁定**: 使用兼容 Python 3.8 的版本范围
3. **优化构建**: 移除不必要的 requirements.txt 复制，直接在 RUN 中指定版本

### 安装顺序
```dockerfile
1. 升级 pip, setuptools, wheel
2. 安装 numpy, scipy, Cython (核心依赖)
3. 安装 PyTorch 和 DGL
4. 安装化学和生物信息学库
5. 安装其他科学计算库
```

## 验证
构建命令：
```bash
docker build -t rtmscore:latest -f airdd/model/RTMScore/repo/Dockerfile airdd/model/RTMScore/repo/
```

## 兼容性矩阵

| 组件 | 版本 | Python 3.8 | 说明 |
|------|------|-----------|------|
| numpy | 1.20-1.23 | ✅ | 2.0+ 不支持 |
| PyTorch | 1.9.0 | ✅ | CUDA 11.1 |
| DGL | 0.7.0 | ✅ | CUDA 11.1 |
| RDKit | 2021.9.5.1 | ✅ | 化学库 |
| MDAnalysis | 2.0.0 | ✅ | 分子动力学 |

## 注意事项
- 如果需要更新依赖，请确保与 Python 3.8 兼容
- 考虑升级到 Python 3.10+ 以支持更新的库版本
- CUDA 版本必须与 PyTorch 和 DGL 匹配
