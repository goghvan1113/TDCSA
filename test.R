# Title   : test
# Author  : gaof23
# Time    : 2024/7/8
# Require : R 4.0.5 才能在pycharm画出图并且可以安装下面的包

# # 下面是测试这个版本的R能不能再Pycharm中画图
# mycars <- within(mtcars,{
#   vs <- factor(vs, labels = c('V', 'S'))
#   am <- factor(am, labels = c('automatic', 'manual'))
#   cyl <- ordered(cyl)
#   gear <- ordered(gear)
#   carb <- ordered(carb)
# })
#
# gears <- table(mycars$gear)
#
# barplot(gears, main='Title: Car gear distribution',xlab = 'Number of Gears', col = '#05ae99')
# am <- table(mycars&am)
# print(am)


# install.packages('tidyverse',type='binary') # 直接安装二进制包文件
# install.packages('tidytext',type='binary')
# install.packages('stringr',type='binary')
# install.packages('SentimentAnalysis',type='binary')
# if (!require("pacman")) install.packages("pacman")
# pacman::p_load(sentimentr, dplyr, magrittr)
# install.packages('textdata',type='binary')

library(tidyverse) # 数据处理和绘图
library(stringr) # 文本清理和正则表达式
library(tidytext) # 提供额外的文本挖掘功能
library(SentimentAnalysis)

# # 下面代码是测试ggplot()
# rm(list = ls())
# dat <- data.frame(
#   time = factor(c("Lunch","Dinner"), levels=c("Lunch","Dinner")),
#   total_bill = c(14.89, 17.23)
# )
# dat
# ggplot(data=dat, aes(x=time, y=total_bill, fill=time)) + geom_bar(stat="identity")

get_sentiments("afinn") %>% head()
# get_sentiments("bing")
# get_sentiments("nrc")

# # Create a vector of strings
# documents <- c("Wow, I really like the new light sabers!",
#                "That book was excellent.",
#                "R is a fantastic language.",
#                "The service in this restaurant was miserable.",
#                "This is neither positive or negative.",
#                "The waiter forget about my a dessert -- what a poor service!")
#
# # Analyze sentiment
# sentiment <- analyzeSentiment(documents)
# # Extract dictionary-based sentiment according to the QDAP dictionary
# sentiment$SentimentQDAP
# # View sentiment direction (i.e. positive, neutral and negative)
# convertToDirection(sentiment$SentimentQDAP)
# response <- c(+1, +1, +1, -1, 0, -1)
# compareToResponse(sentiment, response)