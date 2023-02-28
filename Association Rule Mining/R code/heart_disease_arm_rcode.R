library(stats)  ## for dist
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/dist

## There are many clustering libraries
#install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)

library(amap)  ## for using Kmeans (notice the cap K)

library(factoextra) ## for cluster vis, silhouette, etc.
library(purrr)

#install.packages("stylo")
library(stylo)  ## for dist.cosine
#install.packages("philentropy")
library(philentropy)  ## for distance() which offers 46 metrics
## https://cran.r-project.org/web/packages/philentropy/vignettes/Distances.html
#install.packages('SnowballC')
library(SnowballC)
#install.packages('caTools')
library(caTools)
library(dplyr)
#install.packages('textstem')
library(textstem)
library(stringr)
#install.packages('wordcloud')
library(wordcloud)
#install.packages('RcolorBrewer')
#install.packages('tm')
#library(tm) ## to read in corpus (text data)

library(viridis)

library(TSP)
library(data.table)
library(ggplot2)
library(Matrix)
library(tcltk)
library(dplyr)
library(devtools)
library(purrr)
library(tidyr)
library(shiny)

## IF YOUR CODE BREAKS - TRY THIS
##
## Error in length(obj) : Method length not implemented for class rules 
## DO THIS: 
## (1) detach("package:arulesViz", unload=TRUE)
## (2) detach("package:arules", unload=TRUE)
## (3) library(arules)
## (4) library(arulesViz)
############################

library(arules)
library(arulesViz)
#install.packages("htmlwidgets")
library(htmlwidgets)

getwd()
#set working directory 
setwd("C:/Users/Sriram/Downloads")

#loading the dataset 
heart_data <- read.csv("heart_disease_data_arm")
heart_data

#removing first unkown column 
heart_data[, "X"] <- NULL
heart_data

heart_data[, "target_label"] <- NULL
heart_data[, "sex"] <- NULL
heart_data


summary(heart_data)

#removing row names and column names 
rownames(heart_data) <- NULL
colnames(heart_data) <- NULL

#replacing null values with empty
heart_data[is.na(heart_data)] <- ""

#saving cleaned data into csv file  
write.csv(heart_data, "heart_disease_data_arm_R", row.names = FALSE)

#converting data into transactions
heart_data <- read.transactions("heart_disease_data_arm_R",
                           rm.duplicates = FALSE, 
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep=",",  ## csv file
                           cols=NULL) ## The dataset has no row numbers

#inspecting first 5 transactions 
inspect(heart_data[1:5])

#frequency plot of the items in the transactions 
itemFrequencyPlot(heart_data, topN=20, type="absolute")

##########################Apriori########################################

Hrules = arules::apriori(heart_data, parameter = list(support=.25, 
                                                   confidence=.35, minlen=3))
#inspecting top 10 rules 
inspect(Hrules[1:20])

SortedRules_confi <- sort(Hrules, by="confidence", decreasing=TRUE)
SortedRules_lift <- sort(Hrules, by="lift", decreasing=TRUE)
SortedRules_supp <- sort(Hrules, by="support", decreasing=TRUE)
inspect(SortedRules_confi[1:20])
inspect(SortedRules_lift[1:20])
inspect(SortedRules_supp[1:20])
#plotting all the rules
plot(Hrules, method = "scatterplot")

#sorting rules with respect to confidence 
SortedRules <- sort(Hrules, by="confidence", decreasing=TRUE)

#inspecting top 20 rules 
inspect(SortedRules[1:20])
(summary(SortedRules))


plot(SortedRules_confi[1:20], method="graph", engine="htmlwidget")
plot(SortedRules_confi[1:20], method = "graph", asEdges = TRUE, limit = 10) 



