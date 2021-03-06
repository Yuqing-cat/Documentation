Title: Chapter two. End to End Machine Learning Project
# 第二章、端到端机器学习项目

在这一章里，你将扮演一个房地产公司新聘用的数据科学家
## 真实数据集
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

## 宏观观察
### 构成问题
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


## 实际操作
在这一部分中，我们将在Jupyter Notebook中观察、训练我们的数据集。具体步骤包括：
* 获取数据
    - 创建工作区：初始化python workspace
    - 下载数据： urllib; pandas
    - 观察数据： 
        - .head(); .info(); .value_counts(); .describe(); 
        - %matplotlib inline
            - hist(bins=a,figsize=(x,y))
    - 切割数据集，划分测试、训练两部分
        - from sklearn.model_selection import train_test_split
        - from sklearn.model_selection import StratfiledShuffleSplit
* 从数据可视化中获取洞察
    - 可视化地理数据
        - .plot(kind="scatter",x="longtitude,y="latitude",alpha= 0.4)
    - 观察相关系数
        - .corr()
        - 越1，越正相关
        - 越-1，越负相关
        - from pandas.tools.plotting import scatter_matrix
    - 尝试组合各种属性
        - 人均、平均
* 为机器学习算法准备数据
    - 数据清洗
        - 去零
            - from sklearn.preprocessing import Imputer
    - 文本、分类属性
        - from sklearn.preprocessing import LabelEncoder
        - from sklearn.preprocessing import OneHotEncoder
        - from sklearn.preprocessing import LabelBinarizer
    - 自定义转化
        - from sklearn.base import BaseEstimator, TransformerMixin
    - 特征缩放
    - 变化管道(Transformation pipelines)
        - from sklearn.pipeline import Pipeline
        - from sklearn.preprocessing import StandardScaler
* 选择并训练一个模型
    - 在训练集上训练和评估
        - from sklearn.linear_model import LinearRegression
        - from sklearn.tree import DecisionTreeRegressor
        - from sklearn.metrics import mean_squared_error
            - rmse = np.sqrt(mean_squared_error(labels, predictions))
    - 通过cross validation更好的评估
        - from sklearn.model_selection import cross_val_score
            - rmse_scores = np.sqrt(-cross_val_score(model, prepare_date, labels, scoring = "neg_mean_squared_error",cv=10))
* 调试你的模型(Fine-tune)
    - 网格搜索(Grid search)
        - from sklearn.model_selection import GridSearchCV
            - grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
            - grid_search.best_params_
            - grid_search.best_estimator_
    - 随机搜索(Randomized search)
    - 集成方法(Ensemble methods)
    - 分析最佳模型及其误差
        - feature_importances = grid_search.best_estimator_.feature_importances_
        - sorted(zip(feature_importances, attributes), reverse=True)
    - 在测试集评估你的系统
* 启动，监控和维护你的系统

    


    
        

[RMSE]:https://github.com/Yuqing-cat/Documentation/blob/master/MLbooks/img/rmse.png
[MAE]:https://github.com/Yuqing-cat/Documentation/blob/master/MLbooks/img/mae.png
