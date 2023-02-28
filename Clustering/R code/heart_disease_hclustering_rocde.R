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
library(SnowballC)
library(caTools)
library(dplyr)
library(textstem)
library(stringr)
library(wordcloud)
library(tm)
library("ggplot2")
library("reshape2")
library("purrr")
library("dplyr")
# let's start with a dendrogram
library("dendextend")
 

getwd()

#setting working directory 
setwd("C:/Users/Sriram/Downloads")

#reading the data 
data <- read.csv('Cleaned_heart_cholestrol_data')
data


############################ making random sample data ######################
#selecting based on target column 
subset_data_0 <- data[data$target == 0,]
random_0 <- subset_data_0[sample(nrow(subset_data_0),30),]

subset_data_1<- data[data$target == 1,]
random_1<- subset_data_1[sample(nrow(subset_data_1),30),]

#row binding
new_random_data <- rbind(random_0, random_1)
new_random_data

#reseting index
new_data <- new_random_data                     # Duplicate data
rownames(new_data) <- NULL                 # Reset row names
new_data    


################# clustering numerical data using kmeans  ##########

#selecting only continous numeriacl variables
df <- data%>%select(age, trestbps, chol,thalach,oldpeak)
df

#summary of the data 
summary(df)

#scaling the data 
df1 <- scale(df)
df1

#calculating distance 
distance <- get_dist((df1))

#applying kmeans 
k2 <- kmeans(df1, 
             center = 2,
             nstart = 25  )
str(k2)

k2

#plotting clusters
fviz_cluster(k2, data = df1)

# 3clusters
k3 <- kmeans(df1, centers = 3, nstart = 25)

#4 clusters 
k4 <- kmeans(df1, centers = 4, nstart = 25)

#5 clusters 
k5 <- kmeans(df1, centers = 5, nstart = 25)



#2clusters
p1 <- fviz_cluster(k2, geom = "point", data = df1)+
  ggtitle("k = 2")

#3clusters
p2 <- fviz_cluster(k3, geom = "point", data = df1)+
  ggtitle("k = 3")

#4 clusters
p3 <- fviz_cluster(k4, geom = "point", data = df1)+
  ggtitle("k = 4")

#5clusetrs 
p4 <- fviz_cluster(k5, geom = "point", data = df1)+
  ggtitle("k = 5")

library(gridExtra)
grid.arrange(p1,p2,p3,p4, nrow = 2)

#################### Elbow method #################################

set.seed(123)
fviz_nbclust(df1, kmeans, method = "wss")

################## Average Silhouette Method########################

fviz_nbclust(df1, kmeans, method = "silhouette")

################## Gap Statistics###################################
set.seed(13)

gap_stat <- clusGap(df1, FUN = kmeans, nstart = 25, K.max = 10, B = 50)

print(gap_stat, method = "firstmax")

fviz_gap_stat(gap_stat)

######################################################################

#removing waste columns
data[, "X"] <- NULL
data[, 'index']<- NULL

data

#converting categorical labelled to factor form 

data$sex <- as.factor(data$sex)
data$cp <- as.factor(data$cp)
data$fbs <- as.factor(data$fbs)
data$restecg <- as.factor(data$restecg)
data$exang <- as.factor(data$exang)
data$slope <- as.factor(data$slope)
data$thal <- as.factor(data$thal)

#selecting numercial varaibles 
num_vars <- data[, c("age", "trestbps", "chol", "thalach", "oldpeak")]
num_vars

# Standardize the numerical variables
scaled_num_vars <- scale(num_vars)
scaled_num_vars


#########################HIERARCHIAL CLUSTERING ########################

df1

#distance matrix defaut is euclidean 
dist_matrix <- dist(df1)

# Perform hierarchical clustering using complete linkage
hc_complete <- hclust(dist_matrix, method = "complete")

hc_ward <- hclust(dist_matrix, method = "ward.D2")

plot(hc_complete, hang = -1, main = 'Hclust Complete')

plot(hc_ward, hang = -1, main = 'Hclust Ward' )


######################## cosine similarity ###########################

#install.packages("lsa")
library(lsa)

df1

#calculating cosine similarity and ward method 

(dist_C_smallCorp <- distance(as.matrix(df1), method="cosine"))
dist_C_smallCorp<- as.dist(dist_C_smallCorp)
dist_C_smallCorp
HClust_Ward_CosSim_SmallCorp <- hclust(dist_C_smallCorp, method="ward.D2")
plot(HClust_Ward_CosSim_SmallCorp,main = "Cosine Sim ward")

#calculating cosine similarity and complete method 

(dist_C_smallCorp <- distance(as.matrix(df1), method="cosine"))
dist_C_smallCorp<- as.dist(dist_C_smallCorp)
dist_C_smallCorp
HClust_Ward_CosSim_SmallCorp1 <- hclust(dist_C_smallCorp, method="complete")
plot(HClust_Ward_CosSim_SmallCorp1,main = "Cosine Sim complete")


#cosine similarity on sample data for better visulaisation

df2 <- new_data%>%select(age, trestbps, chol,thalach,oldpeak)
df2
df2 <- scale(df2)
df2

#cosine similairty and ward method
(dist_C_smallCorp1 <- distance(as.matrix(df2), method="cosine"))
dist_C_smallCorp1<- as.dist(dist_C_smallCorp1)
dist_C_smallCorp1

HClust_Ward_CosSim_SmallCorp2 <- hclust(dist_C_smallCorp1, method="ward.D2")
plot(HClust_Ward_CosSim_SmallCorp2,main = "Cosine Sim ward sample data")

#cosine similairty and complete  method

HClust_Ward_CosSim_SmallCorp3 <- hclust(dist_C_smallCorp1, method="complete")
plot(HClust_Ward_CosSim_SmallCorp3,main = "Cosine Sim complete sample data")






################### plotting both categorical and numerical using gower ##############

numeric_cols <- num_cols%>%select('age','trestbps','thalach','oldpeak','chol')
#selecting numerical varaibles 
num_cols <- data %>% 
  select_if(is.numeric)

#selcting categorical variables 
cag_cols <- data %>% select_if(is.factor)
cag_cols
num_cols

#scaling numerical varaibles 
scaled_num_cols <- scale(num_cols)

# Compute the dissimilarity matrix using the Gower distance
dissimilarity <- daisy(cbind(scaled_num_cols, cag_cols), metric = "gower")

# Perform hierarchical clustering using complete linkage
hc <- hclust(dissimilarity, method = "complete")
plot(hc)

################## Divisive ############################################################

gower.dist <- daisy(cbind(scaled_num_cols, cag_cols), metric = "gower")

divisive.clust <- diana(as.matrix(gower.dist), 
                        diss = TRUE, keep.diss = TRUE)
plot(divisive.clust, main = "Divisive")

dendro <- as.dendrogram(divisive.clust)
dendro.col <- dendro %>%
  set("branches_k_color", k = 7, value =   c("darkslategray", "darkslategray4", "darkslategray3", "gold3", "darkcyan", "cyan3", "gold3")) %>%
  set("branches_lwd", 0.6) %>%
  set("labels_colors", 
      value = c("darkslategray")) %>% 
  set("labels_cex", 0.5)
ggd1 <- as.ggdend(dendro.col)
ggplot(ggd1, theme = theme_minimal()) +
  labs(x = "Num. observations", y = "Height", title = "Dendrogram, k = 7, Divisive")

# Radial plot looks less cluttered (and cooler)
ggplot(ggd1, labels = T) + 
  scale_y_reverse(expand = c(0.2, 0)) +
  coord_polar(theta="x")


################ Agglomerative ###################################

aggl.clust.c <- hclust(gower.dist, method = "complete")
plot(aggl.clust.c, main = "Agglomerative, complete linkages")



dendro1 <- as.dendrogram(aggl.clust.c)
dendro1.col <- dendro1 %>%
  set("branches_k_color", k = 7, value =   c("darkslategray", "darkslategray4", "darkslategray3", "gold3", "darkcyan", "cyan3", "gold3")) %>%
  set("branches_lwd", 0.6) %>%
  set("labels_colors", 
      value = c("darkslategray")) %>% 
  set("labels_cex", 0.5)
ggd2 <- as.ggdend(dendro1.col)
ggplot(ggd2, theme = theme_minimal()) +
  labs(x = "Num. observations", y = "Height", title = "Dendrogram, k = 7, Agglomerative")

# Radial plot looks less cluttered (and cooler)
ggplot(ggd2, labels = T) + 
  scale_y_reverse(expand = c(0.2, 0)) +
  coord_polar(theta="x")


##################################################################################

