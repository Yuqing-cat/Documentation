# AML CLI Trial
This article is to record the steps of set-up AML CLI environment in chinese.
本文将会用中文记录搭建AML CML环境的步骤。
## DSVM创建与试用
### 创建一个Azure Linux DSVM
DSVM 是数据科学虚拟机的缩写。它是一个预安装了一系列常见的数据分析及机器学习工具的Azure 虚拟机镜像。
更多的内容可以参考这篇[介绍文章](https://github.com/Microsoft/azure-docs/blob/master/articles/machine-learning/machine-learning-data-science-linux-dsvm-intro.md)。
通过[Azure Portal](https://ms.portal.azure.com)新建
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
'''
ssh <用户名>@<ip地址>
'''
## Play with your DSVM
## Set up AML environment
## Real-Time Scenario


[create-DSVM]:https://github.com/Yuqing-cat/Documentation/blob/master/AML/img/Create-DSVM.PNG
