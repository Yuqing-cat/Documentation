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
[histogram](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/histogram.PNG)
可以看出，零柱的数据量很大，我们应该过滤掉频率为零的情况
```R
email_with_exclamation = data[data$char_freq_exclamation > 0, ]
ggplot(email_with_exclamation) + geom_histogram(aes(x=char_freq_exclamation), binwidth=0.25)
```
[histogram2](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/histogram2.PNG)
取频率大于1的情况，并将数据集分为拦截与非拦截两类：
```R
ggplot(data[data$char_freq_exclamation > 1, ], aes(x=char_freq_exclamation)) + 
geom_density(lty=3) +
geom_density(aes(fill=spam, colour=spam), alpha=0.55) +
xlab("spam") +
ggtitle("Distribution of spam \nby frequency of !") +
labs(fill="spam", y="Density")
```
[distribution](https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/distribution.PNG)
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
[rpart]=(https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/rpart.PNG)
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
## 建立AML CLI环境
从DSVM回归到AML CLI本身。
```sh
wget -q http://amlsamples.blob.core.windows.net/scripts/amlupdate.sh -O - | sudo bash -
sudo /opt/microsoft/azureml/initial\_setup.sh
```
成功后，登出再登入，使得设置生效。接下来就要创建AML CLI环境：
```sh
aml env setup
```



## Real-Time Scenario


[create-DSVM]:https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/Create-DSVM.PNG
[login]:https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/login.PNG