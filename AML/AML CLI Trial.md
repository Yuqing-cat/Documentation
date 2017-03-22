# DSVM & AML CLI Trial
This article is to record the steps of set-up AML CLI environment in chinese.
本文将会用中文记录搭建AML CML环境的步骤。
## DSVM创建与试用
### 创建一个Azure Linux DSVM
DSVM 是数据科学虚拟机的缩写。它是一个预安装了一系列常见的数据分析及机器学习工具的Azure 虚拟机镜像。
更多的内容可以参考这篇[介绍文章](https://github.com/Microsoft/azure-docs/blob/master/articles/machine-learning/machine-learning-data-science-linux-dsvm-intro.md)。
通过[Azure Portal](https://ms.portal.azure.com)新建虚拟机.
![创建Linux DSVM][create-DSVM]
选择合适的型号参数，等待几分钟，一个Linux DSVM创建完成。
### Azure Linux DSVM试用
Linux DSVM中预安装的数据科学工具有包含：
* Microsoft R Server 9.0
* Anaconda Python 2.7, 3.5
* JupyterHub
* PostgreSQL
* JuliaPro
* IDEs rne editors: Eclipse, Emacs, IntelliJ IDEA, PyCharm, Atom, VS Code
* Machine Learning Tools:AML, CNTK, Rattle, Xgboost, Vowpal Wabbit, Mxnet
* Git
* Spark local
* ...
通过终端工具远程登录虚拟机：
```sh
ssh <用户名>@<ip地址>
```
![登录Linux DSVM][login]
下载[样例数据](http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.DOCUMENTATION):
```sh
wget http://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data
```
由于样例数据没有表头信息，因此我们需要创建正确的表头：
```
echo 'word_freq_make, word_freq_address, word_freq_all, word_freq_3d,word_freq_our, word_freq_over, word_freq_remove, word_freq_internet,word_freq_order, word_freq_mail, word_freq_receive, word_freq_will,word_freq_people, word_freq_report, word_freq_addresses, word_freq_free,word_freq_business, word_freq_email, word_freq_you, word_freq_credit,word_freq_your, word_freq_font, word_freq_000, word_freq_money,word_freq_hp, word_freq_hpl, word_freq_george, word_freq_650, word_freq_lab,word_freq_labs, word_freq_telnet, word_freq_857, word_freq_data,word_freq_415, word_freq_85, word_freq_technology, word_freq_1999,word_freq_parts, word_freq_pm, word_freq_direct, word_freq_cs, word_freq_meeting,word_freq_original, word_freq_project, word_freq_re, word_freq_edu,word_freq_table, word_freq_conference, char_freq_semicolon, char_freq_leftParen,char_freq_leftBracket, char_freq_exclamation, char_freq_dollar, char_freq_pound, capital_run_length_average,capital_run_length_longest, capital_run_length_total, spam' > headers
```
合并数据集及表头文件
```
cat spambase.data >> headers
mv headers spambaseHeaders.data
```
在该数据集中，有一下集中类别的数据：
* word_freq_WORD: 表示在整个邮件中出现某个词汇的概率。
* word_freq_CHAR：表示在整个邮件中出现某个字符的概率。
* capital_run_length_longest: 表示在整个邮件中，大写字母的最长长度。
* capital_run_length_average: 表示在整个邮件中，大写字母的平均长度。
* capital_run_length_total: 表示在整个邮件中，大写字母的长度总和。
* spam: 邮件是否被拦截的标签(1 = 被拦截， 0 = 不被拦截)
### 利用Microsoft R open可视化数据集
完整的R脚本可以参考[这篇文章](https://github.com/Azure/Azure-MachineLearning-DataScience/blob/master/Data-Science-Virtual-Machine/Linux/samples/r-sample.r)
```
git clone https://github.com/Azure/Azure-MachineLearning-DataScience.git
```
在终端界面输入
```
R
```
进入R的交互界面后，我们将前面处理好的数据集赋值并设置随机数种子：
```R
data <- read.csv("spambaseHeaders.data")
set.seed(123)
```
通过summary()函数观察数据集的统计学信息：
```R
summary(data)
```
利用str()函数观察数据集的结构：
```R
str(data)
```
我们可以看到，spam这一列被读作了整型，而实际上它是一个分类变量(factor)。因此我们需要重新定义其类型：
```R
data$spam <- as.factor(data$spam)
```
基于ggplot2这个已经预装在虚机中的常用的图形库，我们可以对已有的统计性描述做一些探索性分析。接下来我们将以感叹号为例进行数据可视化的尝试。
```R
library(ggplot2)
ggplot(data) + geom_histogram(aes(x=char_freq_exclamation), binwidth=0.25)
```
得到了感叹号的频率直方图：
![histogram](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/histogram.PNG)
可以看出，零柱的数据量很大，我们应该过滤掉频率为零的情况
```R
email_with_exclamation = data[data$char_freq_exclamation > 0, ]
ggplot(email_with_exclamation) + geom_histogram(aes(x=char_freq_exclamation), binwidth=0.25)
```
![histogram2](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/histogram2.PNG)

取频率大于1的情况，并将数据集分为拦截与非拦截两类：
```R
ggplot(data[data$char_freq_exclamation > 1, ], aes(x=char_freq_exclamation)) + 
geom_density(lty=3) +
geom_density(aes(fill=spam, colour=spam), alpha=0.55) +
xlab("spam") +
ggtitle("Distribution of spam \nby frequency of !") +
labs(fill="spam", y="Density")
```
![distribution](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/Distribution.PNG)

### 训练并测试一个机器学习模型
接下来，我们来试着训练一个能够区分垃圾邮件的机器学习模型。我们将采用决策树和随机森林这两种模型进行训练。
在开始训练之前，我们需要将原始数据集分为训练集和测试集。其中，runif()函数能够生产均匀分布的随机数。
```R
rnd <- runif(dim(data)[1])
trainSet = subset(data, rnd <= 0.7)
testSet = subset(data, rnd > 0.7)
```
接下来利用[rpart()函数](https://cran.r-project.org/web/packages/rpart/rpart.pdf)创建决策树：
```R
model.rpart <- rpart(spam ~ ., method = "class", data = trainSet)
plot(model.rpart)
text(model.rpart)
```
![rpart](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/rpart.PNG)
通过一下两段代码，分别计算这一模型在训练集和测试集上的准确性：
```R
trainSetPred <- predict(model.rpart, newdata = trainSet, type = "class")
t <- table(`Actual Class` = trainSet$spam, `Predicted Class` = trainSetPred)
accuracy <- sum(diag(t))/sum(t)
accuracy
```
得到0.9030359。
```R
testSetPred <- predict(model.rpart, newdata = testSet, type = "class")
t <- table(`Actual Class` = testSet$spam, `Predicted Class` = testSetPred)
accuracy <- sum(diag(t))/sum(t)
accuracy
```
得到0.9024035。

现在我们换成随机森林模型。该模型能够训练大量决策树，并输出一个来自所有个体决策树的类。它矫正了决策树模型过度拟合训练数据集的弱点。
```R
require(randomForest)
trainVars <- setdiff(colnames(data), 'spam')
model.rf <- randomForest(x=trainSet[, trainVars], y=trainSet$spam)

trainSetPred <- predict(model.rf, newdata = trainSet[, trainVars], type = "class")
table(`Actual Class` = trainSet$spam, `Predicted Class` = trainSetPred)

testSetPred <- predict(model.rf, newdata = testSet[, trainVars], type = "class")
t <- table(`Actual Class` = testSet$spam, `Predicted Class` = testSetPred)
accuracy <- sum(diag(t))/sum(t)
accuracy
```
测试集准确率提高到了0.9504734。
### 将机器学习模型部署至Azure ML
首先，创建Azure Machine Learning Workspace。并在Machine Learning Studio界面的Settings中找到Workspace ID和Authorization Token，填入如下命令行中。
```sh
require(AzureML)
wsAuth = "<authorization-token>"
wsID = "<workspace-id>"
```
为了便于部署，我们将模型简化。
```R
colNames <- c("char_freq_dollar", "word_freq_remove", "word_freq_hp", "spam")
smallTrainSet <- trainSet[, colNames]
smallTestSet <- testSet[, colNames]
model.rpart <- rpart(spam ~ ., method = "class", data = smallTrainSet)
```
构造预测函数：
```R
predictSpam <- function(char_freq_dollar, word_freq_remove, word_freq_hp) {
    predictDF <- predict(model.rpart, data.frame("char_freq_dollar" = char_freq_dollar,
    "word_freq_remove" = word_freq_remove, "word_freq_hp" = word_freq_hp))
    return(colnames(predictDF)[apply(predictDF, 1, which.max)])
}
```
将该预测函数通过AzureML发布：
```R
spamWebService <- publishWebService("predictSpam","spamWebService",list("char_freq_dollar"="float", "word_freq_remove"="float","word_freq_hp"="float"),list("spam"="int"),wsID, wsAuth)
```
然而却出现了error。尚未明确原因。
[error](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/error_publish.PNG)

```R
ws <- workspace(id = "143e84a9e3ab470c9414c45363c61d9e",
authorization_token = "fz2QS5IEcCgdciS2GoLSoNpepVBfHUQMtdazsjFMeHoHVJEKQvcdiIv9kIi7wiy5TCKOKXgcZkNwOD0PsV9UJA==")
```
## 建立AML环境
从DSVM回归到AML CLI本身。
```sh
wget -q http://amlsamples.blob.core.windows.net/scripts/amlupdate.sh -O - | sudo bash -
sudo /opt/microsoft/azureml/initial\_setup.sh
```
成功后，登出再登入，使得设置生效。接下来就要创建AML CLI环境：
```sh
aml env setup
```
输入任意1-20位小写字母作为AML环境的名字，并选择订阅。相应的资源组、存储、容器注册表会自动创建。
通过以下命令保存环境变量。
```sh
source ~/.amlenvrc
cat < ~/.amlenvrc >> ~/.bashrc
```
## 访问Jupyter 
在浏览器中键入：
```
https://<ip地址>:8000
```
输入用户名密码后，登录至如下界面：
[Jupyter](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/Jupyter.PNG)
关于JupyterHub，请参照[这篇教程](http://jupyterhub-tutorial.readthedocs.io/en/latest/)

## spark实时场景
在JupyterHub中按照说明运行realtimewebservices.ipynb。
Cell -> run all；去掉In[6]第一行的注释(#%%save_file -f testing.py)并再次运行。
回到命令行界面，创建实时服务。
```sh
cd notebooks\azureml\realtime
aml env local
aml service create realtime -f testing.py -m housing.model -s webserviceschema.json -n mytestapp
```
页面会在‘create doker image’停留，
[realtime-01](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/realtime-01.PNG)
[realtime-02](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/realtime-02.PNG)
需要注意的是，教程中给出的score其实是label，即输入的最后一个数值，而非机器学习后的结果。
## spark批处理场景
同样的，在JupyterHub中按照说明运行batchwebservices.ipynb。
```sh
cd notebooks\azureml\batch

```
[batch-01](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/batch-01.PNG)
[batch-02](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/batch-02.PNG)

## 初窥CNTK
[Python API for CNTK](https://www.cntk.ai/pythondocs/gettingstarted.html)
在Python的交互界面中
```Python
import cnkt
cntk.__version__
```

```Python
cntk.minus([1,2,3],[4,5,6]).eval()
```
array([-3., -3., -3.], dtype=float32)


```Python
import numpy as np
x = cntk.input_variable(2)
y = cntk.input_variable(2)
x0 = np.asarray([[2.,1.]],dtype=np.float32)
y0 = np.asarray([[4.,6.]],dtype=np.float32)
cntk.squared_error(x, y).eval({x:x0, y:y0})
```
array([[ 29.]], dtype=float32)
两个数组的方差为(2-4)^2+(1-6)^2=29

```Python
import cntk as C
c = C.constant(3,shape=(2,3))
np.asarray(c)
np.ones_like(c)
```
array([[ 3.,  3.,  3.],
       [ 3.,  3.,  3.]], dtype=float32)
array([[ 1.,  1.,  1.],
       [ 1.,  1.,  1.]], dtype=float32)

在notebooks/CNTK中，涵盖了一系列包括图像分析、语言理解、强化学习等。

## 图像分类
```sh
cd .\notebooks\azureml
mkdir cntk
```
从[Github 文件夹](https://github.com/Azure/Spark-Operationalization-On-Azure/tree/master/samples/cntk/tutorials/realtime/files)复制resnet.dnn, driver.py, and score_file.py和car.png到cntk文件夹。
```sh
aml service create realtime -r cntk-py -f driver.py -m resnet.dnn -n cntksrvc2
```
[cntk_realtime](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/cntk_realtime.PNG)

[error_cntk](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/error_cntk.PNG)


```sh
aml service run realtime -n cntksrvc2 -d '{"input": "[\"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\"]"}'
```

## Tensorflow


[create-DSVM]:https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/Create-DSVM.PNG
[login]:https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/login.PNG