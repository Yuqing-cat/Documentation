Title: Chapter two. End to End Machine Learning Project

在这一章里，你将扮演一个房地产公司新聘用的数据科学家
### 真实数据集
* [UC Irvine Machine Learning Repository](http://archive.ics.uci.edu/ml/)
* [Kaggle Dataset](https://www.kaggle.com/datasets)
* [Amazon’s AWS datasets](https://aws.amazon.com/fr/datasets/)
* Meta portals (they list open data repositories):
    - http://dataportals.org/
    - http://opendatamonitor.eu/
    - http://quandl.com/
* Other pages listing many popular open data repositories:
    - [Wikipedia’s list of Machine Learning datasets](https://goo.gl/SJHN2k)
    - [Quora.com question] (http://goo.gl/zDR78y)
    - [Datasets subreddit](https://www.reddit.com/r/datasets/)
本章中，我们选择了StatLib的加州住房价格数据集。该数据集基于1990年加州的人口普查数据。

### 宏观观察
#### 构成问题
Q - 我们的业务目标是什么。毕竟，建立模型不会是最终目的。公司期望如何使用并受益于这一模型将会决定如何构建模型、选择什么算法、怎样测试性能以及花费多少精力调优。
A - 模型的输出将会和其他的信号一起送到另一个机器学习系统中。这个下游系统将决定给定的区域是否值得投资。这将会直接影响公司的营收。
Q - 现有的解决方案是什么(如果有的话)。
A - 现在由专家手动估算区域住房价格。一个团队收集一个相关区域的最新信息，并使用一种复杂的规则来估计。成本高，耗时大，并且有15%的错误率。

有了这些信息，你现在可以来设计你的系统了。那么首先，你需要决定：
* 这是一个监督、非监督还是强化学习？
* 是分类、回归还是其他任务？
* 使用批量学习还是在线学习？

在我看来，这是一个典型的**监督学习**任务。因为你有被标记过的训练样本(每个实例都有一个预期的输出)。另外，这是一个典型的*回归*任务，因为你需要预测一个数值。更具体的来说，这是一个**多元回归**问题，因为系统将使用多个特征进行预测。最后，系统中不会有连续的数据流，而且不需要快速调整数据，所以使用**批量**学习就能够做的很好。

#### 选择性能指标
回归问题的典型性能指标是**均方根误差**(RMSE)。
例如，当RMSE = 50,000时，表示大约有68%的系统预测在落在$50,000之内，而95%的系统预测落在$100,000之内。
![RMSE][RMSE]
尽管RSME通常是回归问题的首选。但在某些情况下，我们也会倾向于其他的函数。例如，假设有很多离群区域，那么**平均绝对误差**(MAE)会是更好的选择。
![MAE][MAE]
#### 检查假设

### 获取数据
#### 创建工作区


[RMSE]:(Documentation\MLbooks\img\rmse.PNG)
[MAE]:(Documentation\MLbooks\img\mae.PNG)
