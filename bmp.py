import numpy as np
import sys
from loader import ReadBMPFile

# 命令行传入的文件路径
filePath = 'img1.bmp'
# 读取 BMP 文件
bmpFile = ReadBMPFile(filePath)
# R, G, B 三个通道 [0, 255]
R = bmpFile.R
G = bmpFile.G
B = bmpFile.B
# 显示图像
b = np.array(B, dtype = np.uint8)
g = np.array(G, dtype = np.uint8)
r = np.array(R, dtype = np.uint8)

print(r.shape)
print(g)
print(b)