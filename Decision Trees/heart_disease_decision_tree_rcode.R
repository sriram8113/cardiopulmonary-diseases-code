## LIBRARIES
library(rpart)   ## FOR Decision Trees
library(rattle)  ## FOR Decision Tree Vis
library(rpart.plot)
library(RColorBrewer)
library(Cairo)
library(network)
library(ggplot2)
##If you install from the source....
#Sys.setenv(NOAWT=TRUE)
## ONCE: install.packages("wordcloud")
library(wordcloud)
## ONCE: install.packages("tm")

library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
#library(SnowballC)

library(proxy)
## ONCE: if needed:  install.packages("stringr")
library(stringr)
## ONCE: install.packages("textmineR")
library(textmineR)
library(igraph)
library(caret)
#library(lsa)

getwd()
#set working directory 
setwd("C:/Users/Sriram/Downloads")

#loading the dataset 
data <- read.csv('Cleaned_heart_cholestrol_data',stringsAsFactors=TRUE)
data

data[, "X"] <- NULL
data[, "index"] <- NULL
data

str(data)
data$target<-as.factor(data$target)
str(data)

apply(data, 2, table) 

GoPlot <- function(x) {
  
G <-ggplot(data=data, aes(.data[[x]], y="") ) +
    geom_bar(stat="identity", aes(fill =.data[[x]])) 
  
  return(G)
}

## Use the function in lappy
lapply(names(data), function(x) GoPlot(x))


##############################################################

(DataSize=nrow(data)) ## how many rows?
(TrainingSet_Size<-floor(DataSize*(3/4))) ## Size for training set
(TestSet_Size <- DataSize - TrainingSet_Size) ## Size for testing set

## Random sample WITHOUT replacement (why?)
## set a seed if you want it to be the same each time you
## run the code. The number (like 1234) does not matter
set.seed(1234)

## This is the sample of row numbers
(MyTrainSample <- sample(nrow(data),
                         TrainingSet_Size,replace=FALSE))

## Use the sample of row numbers to grab those rows only from
## the dataframe....
(MyTrainingSET <- data[MyTrainSample,])
table(MyTrainingSET$target)

## Use the NOT those row numbers (called -) to get the
## other row numbers not in the training to use to create
## the test set.

## Training and Testing datasets MUST be disjoint. Why?
(MyTestSET <- data[-MyTrainSample,])
table(MyTestSET$target)


(TestKnownLabels <- MyTestSET$target)
(MyTestSET <- MyTestSET[ , -which(names(MyTestSET) %in% c("target"))])


MyTrainingSET
str(MyTrainingSET)

## This code uses rpart to create decision tree
## Here, the ~ .  means to train using all data variables
## The MyTrainingSET#label tells it what the label is called
## In this dataset, the label is called "label".


DT <- rpart(MyTrainingSET$target ~ ., data = MyTrainingSET, method="class")
summary(DT)

DT2<-rpart(MyTrainingSET$target ~ ., data = MyTrainingSET,cp=.27, method="class")
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT2)


 ## This is the cp plot

## Let's make a third tree - here we use cp = 0 and 
## "information" instead of the default which is GINI
DT3<-rpart(MyTrainingSET$target ~ ., 
           data = MyTrainingSET,cp=0, method="class",
           parms = list(split="information"),minsplit=2)
## The small cp the larger the tree if cp is too small you have overfitting
summary(DT3)

DT$variable.importance
DT2$variable.importance
DT3$variable.importance

plotcp(DT)
plotcp(DT2)
plotcp(DT3)
#DT---------------------------------
(DT_Prediction= predict(DT, MyTestSET, type="class"))
## Confusion Matrix
conf_matrix<-table(DT_Prediction,TestKnownLabels) ## one way to make a confu mat
conf_matrix
## VIS..................
fancyRpartPlot(DT)

## DT2-----------------------------
### Example two with cp - a lower cp value is a bigger tree
(DT_Prediction2= predict(DT2, MyTestSET, type = "class"))
## ANother way to make a confusion matrix
table(DT_Prediction2,TestKnownLabels)
fancyRpartPlot(DT2)
## Example three with information gain and lower cp

##DT3---------------------------------------------------------
(DT_Prediction3= predict(DT3, MyTestSET, type = "class"))
table(DT_Prediction2,TestKnownLabels)
rattle::fancyRpartPlot(DT3,main="Decision Tree", cex=.5)


##########################################################################################
#plotting confusion matrix and classification report 

draw_confusion_matrix <- function(cm) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, 'Class1', cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, 'Class2', cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, 'Class1', cex=1.2, srt=90)
  text(140, 335, 'Class2', cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
} 

cm <- confusionMatrix(data = DT_Prediction, reference = TestKnownLabels)
cm2 <- confusionMatrix(data = DT_Prediction2, reference = TestKnownLabels)
cm3 <- confusionMatrix(data = DT_Prediction3, reference = TestKnownLabels)
draw_confusion_matrix(cm)
draw_confusion_matrix(cm2)
draw_confusion_matrix(cm3)
cm
cm3
############################################################################################
#feature importanace 

(vi_tree <- DT$variable.importance)

barplot(vi_tree, las = 1)

############################################################################################
(vi_tree2 <- DT2$variable.importance)

barplot(vi_tree2, las = 1)

(vi_tree3 <- DT3$variable.importance)

barplot(vi_tree3, las = 1)
