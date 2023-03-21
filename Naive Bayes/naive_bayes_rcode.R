library(tm)
#install.packages("tm")
library(stringr)
library(wordcloud)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering

library(naivebayes)
#Loading required packages
#install.packages('tidyverse')
library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
library(e1071)
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

StudentDF <- data

(Size <- (as.integer(nrow(StudentDF)/4)))  ## Test will be 1/4 of the data
(SAMPLE <- sample(nrow(StudentDF), Size))

(DF_Test_Student<-StudentDF[SAMPLE, ])
(DF_Train_Student<-StudentDF[-SAMPLE,])

head(DF_Train_Student, 10)
head(DF_Test_Student, 10)

str(DF_Test_Student$target)  ## Notice that the label is called "target" and
## is correctly set to type FACTOR. This is IMPORTANT!!
str(DF_Train_Student$target)  ## GOOD! Here "target" is also type FACTOR
##Check balance of test dataset
table(DF_Test_Student$target)


## Copy the Labels
(DF_Test_Student_Labels <- DF_Test_Student$target)
## Remove the labels
DF_Test_Student_NL<-DF_Test_Student[ , -which(names(DF_Test_Student) %in% c("target"))]
(DF_Test_Student_NL[1:5, 1:5])
## Check size
(ncol(DF_Test_Student_NL))
#(DF_Test_Student_NL)
## Train...--------------------------------
## Copy the Labels
(DF_Train_Student_Labels <- DF_Train_Student$target)
## Remove the labels
DF_Train_Student_NL<-DF_Train_Student[ , -which(names(DF_Train_Student) %in% c("target"))]
(DF_Train_Student_NL[1:5, 1:5])
## Check size
(ncol(DF_Train_Student_NL))
#(DF_Train_Student_NL)

head(DF_Test_Student_NL)  ## Testset
head(DF_Train_Student_NL)  ## Training set
## Label name is "Decision"
## Test labels:
head(DF_Test_Student_Labels)
head(DF_Train_Student_Labels)
############################ model ################################



(NB_e1071_Student1<-naiveBayes(DF_Train_Student_NL, DF_Train_Student_Labels))# instantiation
NB_e1071_Pred_Student1 <- predict(NB_e1071_Student1, DF_Test_Student_NL) # preditcions 
table(NB_e1071_Pred_Student1,DF_Test_Student_Labels)
(NB_e1071_Pred_Student1)

confusionMatrix(NB_e1071_Pred_Student1, DF_Test_Student_Labels)

############## model with laplace = 1 

(NB_e1071_Student<-naiveBayes(DF_Train_Student_NL, DF_Train_Student_Labels, laplace =1)) #instantiation
NB_e1071_Pred_Student <- predict(NB_e1071_Student, DF_Test_Student_NL) # predictions
table(NB_e1071_Pred_Student,DF_Test_Student_Labels)
(NB_e1071_Pred_Student)

confusionMatrix(NB_e1071_Pred_Student, DF_Test_Student_Labels)

#plotting confusion matrix 

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


cm <-confusionMatrix(NB_e1071_Pred_Student, DF_Test_Student_Labels)
cm1 <-confusionMatrix(NB_e1071_Pred_Student1, DF_Test_Student_Labels)

draw_confusion_matrix(cm)
draw_confusion_matrix(cm1)

cm
cm1
