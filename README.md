给定背景图image和要被覆盖的范围mask，以及数张候选的用于填充的图片candidate，从candidate中选取区域，与image融合。

### 运行方式

```
python main.py --input_image [input_name]
```

### 代码主要构成

卷积版的L2距离（jittor聚合元算子实现）

graph-cut算法计算融合边界（使用了pymaxflow提供的api）

泊松融合（自行构造泊松矩阵，利用共轭梯度法求解泊松方程组Ax=b，利用了jsparse的spmm）