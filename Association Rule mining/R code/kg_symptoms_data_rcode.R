#Association rule mining 
#install.packages("arules")
#detach("package:arulesViz", unload=TRUE)
#detach("package:arules", unload=TRUE)
library(arules)
library(arulesViz)

#install.packages("TSP")
#install.packages("data.table")
## NOTE: If you are asked if you want to INSTALL FROM SOURCE - click YES!
#install.packages("arulesViz", dependencies = TRUE)
## IMPORTANT ## arules ONLY grabs rules with ONE item on the right
#install.packages("sp")


#install.packages("dplyr", dependencies = TRUE)
#install.packages("purrr", dependencies = TRUE)
#install.packages("devtools", dependencies = TRUE)
#install.packages("tidyr")
#install.packages("htmltools")
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

getwd()
setwd("C:/Users/Sriram/Downloads")
# Reading the data 

data1<- read.csv('dataset.csv')
data1

# Removing disease column

data1[, "Disease"] <- NULL

data1

#Removing row names and column names 

rownames(data1) <- NULL
colnames(data1) <- NULL

#Removing all the  null values 

data1[is.na(data1)] <- ""
data1

#storing cleaned data into another csv file 

write.csv(data1, "data1_ARM_final.csv", row.names = FALSE)

#converting the data type into transactions 

symptoms <- read.transactions("data1_ARM_final.csv",
                           rm.duplicates = FALSE, 
                           format = "basket",  ##if you use "single" also use cols=c(1,2)
                           sep=",",  ## csv file
                           cols=NULL) ## The dataset has no row numbers

#inspecting the transactions 

inspect(symptoms[1:50])


######################## Apriori ##################################


Srules1 = arules::apriori(symptoms, parameter = list(support=.035, 
                                                 confidence=.35, minlen=2))
inspect(Srules1[1:10])

#plotting all the rules
plot(Srules1, method = "scatterplot")

#plotting the frequent items in the transactions
itemFrequencyPlot(symptoms, topN=20, type="absolute")

#sorting rules wrto confidence
SortedRules1 <- sort(Srules1, by="confidence", decreasing=TRUE)

#inspecting top 20 rules 
inspect(SortedRules1[1:20])

#summary of the rules
(summary(SortedRules1))

#sorting rules with respect to lift 
subrules1 <- head(sort(SortedRules1, by="lift"),20)

#plotting 
plot(subrules1, method="graph", engine="htmlwidget")


########################## chestpain rules ###################################

havingchestpainRules <- apriori(data=symptoms,parameter = list(supp=.001, conf=.01, minlen=3),
                              appearance = list(default="lhs", rhs="chest_pain"),
                              control=list(verbose=FALSE))

#sorting with respect to confidence 
havingchestpainRules <- sort(havingchestpainRules, decreasing=TRUE, by="confidence")

#inspecting top 20 rules 
inspect(havingchestpainRules[1:20])

#plotting
plot(havingchestpainRules[1:30], method="graph", engine="htmlwidget")
plot(havingchestpainRules[1:30], method = "graph", asEdges = TRUE, limit = 10) 

###########################breathlessness rules #############################

breathlessnessRules <- apriori(data=symptoms,parameter = list(supp=.001, conf=.01, minlen=3),
                                appearance = list(default="lhs", rhs="breathlessness"),
                                control=list(verbose=FALSE))

#sorting with respect to confidence 
breathlessnessRules <- sort(breathlessnessRules, decreasing=TRUE, by="confidence")

#inspecting top 20 rules 
inspect(breathlessnessRules[1:20])

#plotting
plot(breathlessnessRules[1:30], method="graph", engine="htmlwidget")
plot(breathlessnessRules[1:30], method = "graph", asEdges = TRUE, limit = 10) 

########################### fast heart rate rules ##########################

fast_heart_rate_Rules <- apriori(data=symptoms,parameter = list(supp=.001, conf=.01, minlen=3),
                               appearance = list(default="lhs", rhs="fast_heart_rate"),
                               control=list(verbose=FALSE))

#sorting with respect to confidence 
fast_heart_rate_Rules <- sort(fast_heart_rate_Rules, decreasing=TRUE, by="confidence")

#inspecting top 20 rules 
inspect(fast_heart_rate_Rules[1:20])

#plotting
plot(fast_heart_rate_Rules[1:30], method="graph", engine="htmlwidget")
plot(fast_heart_rate_Rules[1:30], method = "graph", asEdges = TRUE, limit = 10) 
