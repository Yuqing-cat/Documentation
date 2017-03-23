Title: Classification
# 第三章、分类器
第一章提及了最常见的监督学习任务就是回归(预测数值)和分类(预测类别)。于是，我们在第二章中体验了一个完整的回归任务：预测房价。并使用了各种模型，如：线性回归，决策树和随机森林(这些在后面的章节中会详细讲解)。现在，我们就把目光投向分类系统。

## MNIST
本章将使用MNIST数据集：一个拥有7万张由高中生和美国人口普查局雇员书写的手写数字图片。每张图片都被标记了它所表示的数字。这个数据集被称为机器学习界的“Hello World”：每当人们有了新的分类算法，都会用MNIST来测试性能。

Scikit-Learn提供了很多下载热门数据集的帮助函数。通过以下代码，可以方便的下载MNIST数据集:
```Python
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST orignal')
mnist
```
通过Scikit-Learn下载的数据集通常都有一个相似的字典结构
* 一个 DESCR 键，描述数据集
* 一个 data 键， 包含每个实例一行每个属性一列的数组
* 一个 target 键， 包含标签数组

MNIST数据集有7万张图片，每个图片有784个特征值。这是因为每个图片由28x28的像素组成。从0(纯白)到255(纯黑)，一个特征值代表一个像素点。我们挑选其中一张图片，用reshape()将其还原成28x28的数组，并由Matplotlib的imshow()函数展示它。
```Python
%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt

some_digit = X[36000]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap = matplotlib.cm.binary,
           interpolation="nearest")
plt.axis("off")
plt.show()
```
再通过y[36000]比较相应的标签值。

## 训练二进制分类器
## 性能测试
### 交叉验证测量精度
### 含混矩阵(Confusion matrix)
### 查准率和查全率(Precision and Recall)
### 查准率/查全率的权衡
### ROC 曲线


## 多类分类
## 错误分析
## 多标签分类
## 多输出分类