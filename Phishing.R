########################################################################################################
#
#
#                                      WEB PAGE PHISHING DETECTION
#
#
########################################################################################################

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(matrixStats)) install.packages("matrixStats", repos = "http://cran.us.r-project.org")
if(!require(RColorBrewer)) install.packages("RcolorBrewer", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(gam)) install.packages("gam", repos = "http://cran.us.r-project.org")
if(!require(naivebayes)) install.packages("naivebayes", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(matrixStats)
library(RColorBrewer)
library(caret)
library(gam)
library(randomForest)
library(naivebayes)


# #######################################################################################################
#
#     DATA CLEANING & PROCESSING
#
# #######################################################################################################

#     Data download
#------------------------------------------------------

# The Phishing Detection dataset contains information about web pages classified into legitimate or phishing
# It is is available from Kaggle: https://www.kaggle.com/shashwatwork/web-page-phishing-detection-dataset

# For simplicity, we made the file available in Github: 
# https://raw.githubusercontent.com/ruthparras/phishing/main/dataset_phishing.csv

file_url<- "https://raw.githubusercontent.com/ruthparras/phishing/main/dataset_phishing.csv"
phish<- read_csv(file_url)   # read csv file directly 


#    Data Cleaning
#-----------------------------------------------------

# remove variables that are not useful as predictors and convert character strings into factors.

url<- phish$url  # store the URL of the web page

phish <- phish %>% 
  select (status, everything(), -url) %>%  # move status to the front, and remove URL
  mutate(status=factor(status))  # convert character string into factors

# Detect observations with missing values (NA) and replace or remove
sum(is.na(phish))   


#    Removing predictors with close to zero variability   
#--------------------------------------------------------------

# Plot feature variability to identify which ones don't vary much; therefore having little predictive power

features<- phish %>% select(-status)  # remove prediction (status) and keep only features

sds <- apply(features, 2, sd) # calculate feature variability
qplot(sds, bins=10, main="feature variability", xlab="standard deviation", 
      ylab="count of features") +
  theme(plot.title =element_text(size=20, face ="bold" ), #change title font size
        axis.text=element_text(size=14), #change font size of axis text
        axis.title=element_text(size=16)) #change font size of axis titles

dev.copy(png, 'figures/sds.png') # save for loading into markdown
dev.off()

# count of features with standard deviations lower than 0.5.
index <- sds<0.5
sum(index)
qplot(sds[index], bins=30, main="features with sd < 0.5", xlab="standard deviation", 
      ylab="count of features" )+
  theme(plot.title =element_text(size=20, face ="bold" ), #change title font size
        axis.text=element_text(size=14), #change font size of axis text
        axis.title=element_text(size=16)) #change font size of axis titles

dev.copy(png,'figures/sds_zoom.png')  # save for loading into markdown
dev.off()

# use nearZeroVar to recommend features to be removed due to near zero variability

nzv <- nearZeroVar(phish)
length(nzv)  
phish<- phish[, - nzv]  

dim(phish)  #reduced number of features from 87 to 53, by removing those with low prediction value

save(phish, file="rdas/phish.rda")  # save for loading into rmarkdown


#   Data Scaling
#-----------------------------------------------------------

# A quick inspection of the data shows that different features have different ranges. 
summary(phish[1:10])

# Normalize data and bring it to a similar scale without distorting differences in the ranges of values
# Transform predictors into matrix, center and scale. 

phish_y <- phish$status
phish_x <- phish %>% select(-status) %>% as.matrix()

phish_x_centered <- sweep(phish_x, 2, colMeans(phish_x))   # subtract the mean of each predictor  
phish_x_scaled <- sweep(phish_x_centered, 2, colSds(phish_x), FUN = "/") # divide by sd of each predictor 


#   Data Imbalances
#-----------------------------------------------------------

# Validate that the dataset is balanced, containing the same proportion of phishing and legitimate URLs

mean(phish$status =="phishing")


#    Preparing training and testing sets
#-----------------------------------------------------------

# Create a random partition of the data with a 80% training and 20% test sets
set.seed(123, sample.kind = "Rounding")    # if using R 3.6 or later

test_index <- createDataPartition(phish_y, times = 1, p = 0.2, list = FALSE)  

x_test <- phish_x_scaled[test_index, ]
y_test <- phish_y[test_index]

x_train <- phish_x_scaled[-test_index, ]
y_train <- phish_y[-test_index]

save(x_train, file="rdas/x_train.rda")  # save for loading into rmarkdown
save(y_train, file="rdas/y_train.rda")  # save for loading into rmarkdown


###############################################################################################
# 
#
#     DATA EXPLORATION AND VISUALIZATION
#
#
###############################################################################################
# Explore the data in the training set (train_set) alone, to make sure that "unknown" information 
# from the testing set is not "leaked" and used during the prediction.

train_set<- data.frame( status = y_train, x_train)

dim(train_set)[1] # number of observations. Aprox 80% of phish dataset
dim(train_set)[2] # number of predictors

# visualize first few rows and columns of the scaled train_set 
train_set<- data.frame(status=y_train, x_train)
train_set[1:5, 1:5] 


#    Visualizing Features
#-----------------------------------------------------------------------

# Use a boxplot to visualize the first 10 features of the scaled train_set
boxplot_features <- train_set[,1:10] %>% gather(key = "var", value = "value", -status) %>%
  ggplot(aes(x=var, y=value, fill = status)) + geom_boxplot() + theme_gray (base_size= 20)+
  ggtitle("Boxpot of first 10 variables") + 
  xlab("features") + ylab("scaled value") + 
  theme(plot.title =element_text(size=20, face ="bold" ), #change title font size
        axis.text=element_text(size=12), #change font size of axis text
        axis.title=element_text(size=16)) #change font size of axis titles
boxplot_features


#  Visualizing the relationship among features
#------------------------------------------------------------------------

#  use a heatmap to visualize the relationship among features 

d_features <- dist(t(x_train))  # transpose x_train to calculate the distance among features
heatmap(as.matrix(d_features),
        col = colorRampPalette(brewer.pal(9,"Blues"))(9))

dev.copy(png,'figures/heatmap.png')  # save for loading into markdown
dev.off()

# Determine if some features are closer together and if there is opportunity for dimension reduction
h <- hclust(d_features)    

# summarize clustering information with a dendrogram
plot(h, cex = 0.65, main = "", xlab = "")  

# Cut into 10 groups
groups <- cutree (h, k=10)

#display which features in each group to identify similarities
split(names(groups), groups) 


#*****************    REDUCING DIMENSIONS WITH PRINCIPAL COMPONENT ANALYSIS (PCA) *********

#Perform a principal component analysis of the scaled train matrix

pca <- prcomp(x_train)  
summary(pca)  # summary of PC components

# plot "proportion of variance" and "cumulative proportion"  
pc_names <- 1:ncol(x_train) 
variance<- (pca$sdev^2)/sum(pca$sdev^2)

qplot(pc_names, variance,  xlab = "Principal Component", 
      ylab = "Proportion of Variance Explained") +
  theme(plot.title =element_text(size=20, face ="bold" ), #change title font size
        axis.text=element_text(size=12), #change font size of axis text
        axis.title=element_text(size=16)) #change font size of axis titles

dev.copy(png,'figures/pca_var.png')   # save for loading into markdown
dev.off()

qplot(pc_names, cumsum(variance),  xlab = "Principal Component", 
      ylab = "Cummulative Proportion of Variance Explained") +
  theme(plot.title =element_text(size=20, face ="bold" ), #change title font size
        axis.text=element_text(size=12), #change font size of axis text
        axis.title=element_text(size=16)) #change font size of axis titles

dev.copy(png,'figures/pca_cum.png')  # save for loading into markdown
dev.off()


#   Visualizing individual Principal Components 
#--------------------------------------------------------------------------

# Make a boxplot of the first 5PCs grouped by page status a 
boxplot_PC1_PC5 <- data.frame(status = y_train, pca$x[,1:5]) %>% 
  gather(key = "PC", value = "value", -status) %>% 
  ggplot(aes(PC, value, fill = status)) + 
  geom_boxplot() +
  ylim(-20,20) +
  theme(legend.position="top") +
  scale_fill_manual(breaks = c("legitimate", "phishing"), 
                    values=c("lightgreen", "lightcoral")) + # change color of legend
  ggtitle("Boxplot of first 5 PCs grouped by status") + 
  theme(plot.title =element_text(size=24, face ="bold" ), #change font size of all text
        axis.text=element_text(size=18), #change font size of axis text
        axis.title=element_blank(), # change font size of axis titles
        legend.text=element_text(size=20),  #change font size of legend
        legend.title= element_blank()) # remove legend title

boxplot_PC1_PC5

ggsave("figures/boxplot_PC1_PC5.png", boxplot_PC1_PC5, dpi=300)  # save for load into markdown


#    Visualizing  relationships among Principal Components
#---------------------------------------------------------------------------

#  visualize  relationships among first 8 components, with color representing the web page classification 
# For legibility, select 2,000 observations at random
PC1_PC2<- data.frame(pca$x[,1:2], status= y_train) %>%
  sample_n(2000) %>%    # select 2k observations at random
  ggplot(aes(PC1, PC2, fill = status)) +
  geom_point(cex=3, pch=21) +
  theme(legend.position="top") +
  scale_fill_manual(breaks = c("legitimate", "phishing"), 
                    values=c("lightgreen", "lightcoral")) + # change color of legend
  ggtitle("PC1 and PC2") + 
  theme(plot.title =element_text(size=26, face ="bold" ), #change font size of all text
        axis.text=element_text(size=18), #change font size of axis text
        axis.title=element_text(size=20), # change font size of axis titles
        legend.text=element_text(size=20), #change font size of legend
        legend.title= element_blank()) # remove legend title

PC3_PC4<- data.frame(pca$x[,3:4], status= y_train) %>%
  sample_n(2000) %>%    # select 2k observations at random
  ggplot(aes(PC3, PC4, fill = status)) +
  geom_point(cex=3, pch=21)+
  theme(legend.position="top") +
  scale_fill_manual(breaks = c("legitimate", "phishing"), 
                    values=c("lightgreen", "lightcoral")) + # change color of legend
  ggtitle("PC3 and PC4") + 
  theme(plot.title =element_text(size=26, face ="bold" ), #change font size of all text
        axis.text=element_text(size=18), #change font size of axis text
        axis.title=element_text(size=20), # change font size of axis titles
        legend.text=element_text(size=20),  #change font size of legend
        legend.title= element_blank()) # remove legend title

PC5_PC6<- data.frame(pca$x[,5:6], status= y_train) %>%
  sample_n(2000) %>%    # select 2k observations at random
  ggplot(aes(PC5, PC6, fill = status)) +
  geom_point(cex=3, pch=21)+
  theme(legend.position="top") +
  scale_fill_manual(breaks = c("legitimate", "phishing"), 
                    values=c("lightgreen", "lightcoral")) + # change color of legend
  ggtitle("PC5 and PC6") + 
  theme(plot.title =element_text(size=26, face ="bold" ), #change font size of all text
        axis.text=element_text(size=18), #change font size of axis text
        axis.title=element_text(size=20), # change font size of axis titles
        legend.text=element_text(size=20),  #change font size of legend
        legend.title= element_blank()) # remove legend title

PC7_PC8<- data.frame(pca$x[,7:8], status= y_train) %>%
  sample_n(2000) %>%    # select 2k observations at random
  ggplot(aes(PC7, PC8, fill = status)) +
  geom_point(cex=3, pch=21) +
  ylim(-3,3) +
  theme(legend.position="top") +
  scale_fill_manual(breaks = c("legitimate", "phishing"), 
                    values=c("lightgreen", "lightcoral")) + # change color of legend
  ggtitle("PC7 and PC8") + 
  theme(plot.title =element_text(size=26, face ="bold" ), #change font size of all text
        axis.text=element_text(size=18), #change font size of axis text
        axis.title=element_text(size=20), # change font size of axis titles
        legend.text=element_text(size=20),  #change font size of legend
        legend.title= element_blank()) # remove legend title

PC1_PC2
PC3_PC4
PC5_PC6
PC7_PC8

ggsave("figures/PC1_PC2.png", PC1_PC2, dpi=300)  # save for load into markdown
ggsave("figures/PC3_PC4.png", PC3_PC4, dpi=300)  # save for load into markdown
ggsave("figures/PC5_PC6.png", PC5_PC6, dpi=300)  # save for load into markdown
ggsave("figures/PC7_PC8.png", PC7_PC8, dpi=300)  # save for load into markdown


# ###################################################################################################
#
#  MODELING
#
#
# ###################################################################################################

# REDUCING DIMENSIONS in the TRAINING SET and TESTING SETS
#-----------------------------------------------------------------------------------------------

dim<- 20
x_train_pca <- pca$x[ ,1:dim] # reduce dimensions of training set 

# Perform the same PCA transformation on the test set
x_test_pca <- sweep (x_test, 2, colMeans(x_test)) %*% pca$rotation  # rotate test set to same space as train 
x_test_pca<- x_test_pca[ ,1:dim]  # reduce dimensions of testing set

#create a dataframe to store Accuracy results
results<- data.frame("MODEL"= character(0), 
                     "Accuracy"=integer(0), 
                     "Sensitivity"= integer(0), 
                     "Specificity"= integer(0))


#  TRAIN MODELS: Naive Bayes, LDA, QDA, kNN, and Random Forest
#------------------------------------------------------------------------------------------

# apply multiple ML models using train()
models <- c("naive_bayes", "lda", "qda", "knn", "rf") # 5 common models in caret

fits <- lapply(models, function(model){  # train the different models using default parameters.
  print(model)
  set.seed(123, sample.kind = "Rounding") # if using R 3.6 or later
  train(x_train_pca, y_train , method = model)
})

names(fits) <- models  # set the name of each model

# create matrix of predictions using the test set
predictions <- sapply (fits, function(fits){
  predict (fits, newdata= x_test_pca) })    # apply predict to each fitted model (fits) 

results <- sapply(c(1:5), function (col) {  # apply the confusion matrix to each column in predictions
  y_hat<-factor(predictions[,col])
  cm<- confusionMatrix(y_hat, reference = y_test)
  
  results[col, ]<- c(names(fits[col]),
                     round(cm$overall["Accuracy"],4),
                     round(cm$byClass["Sensitivity"],4), 
                     round(cm$byClass["Specificity"],4))
})

#  kNN MMODEL with cross-validation and tuning for k-neighbors 
#----------------------------------------------------------------------------------------------
# Let's use 10-fold cross validation to train a kNN model and optimize for K

k<- seq(1, 19, 1)
control<- trainControl(method = "cv", number = 10, p = .9) # 10 fold cross-val

set.seed(123, sample.kind = "Rounding")   # if using R 3.6 or later
train_knn <- train (x_train_pca, y_train, method = "knn",
                    tuneGrid = data.frame (k),
                    trControl=control)

ggplot( train_knn, highlight= TRUE) # highlights the k that optimizes the algorithm on the training set
k_opt <- train_knn$bestTune$k # value of k that maximizes accuracy

# fit kNN with optimized k to the entire training data
fit_knn <- knn3 (x_train_pca, y_train, k= k_opt)

y_hat_knn <- predict (fit_knn, x_test_pca, type="class") # make predictions against test set

cm<- confusionMatrix(y_hat_knn, y_test)
cm$overall["Accuracy"]
results2 <- c("tuned knn", 
              round(cm$overall["Accuracy"],4),
              round(cm$byClass["Sensitivity"],4), 
              round(cm$byClass["Specificity"],4))

results <- cbind (results, results2)


#   RANDOM FOREST MODEL with cross validation and  tuning
#*---------------------------------------------------------------------------------------------------- 
# Train a random forest model  using 5-fold cross validation and tune for mtry ranging from 1 to 10 
# with mtry the number of variables randomly sampled as candidates at each split

control<- trainControl(method = "cv", number = 5) # use 5-fold cross validation

set.seed(123, sample.kind = "Rounding")    # if using R 3.6 or later
train_rf<- train (x_train_pca, y_train, method = "rf", 
                  tuneGrid = data.frame (mtry = seq(1, 10, 1)), 
                  trControl=control ) 

mtry_plot<- ggplot( train_rf, highlight= TRUE) +  # highlight mtry value with highest accuracy
  geom_point(size=6) +
  xlab("Randomly Selected Predictors (mtry)") + 
  theme( axis.text=element_text(size=18), #change font size of axis text
         axis.title=element_text(size=26)) # change font size of axis titles
mtry_plot

ggsave("figures/mtry_plot.png", mtry_plot, dpi=300)  # save for load into markdown

train_rf$bestTune  # value of mtry that gives highest accuracy 

# Find the node size that maximizes accuracy
nodesize <- seq(1, 10, 1)
acc <- sapply(nodesize, function(ns){
  set.seed(123, sample.kind = "Rounding")    # if using R 3.6 or later
  train(x_train_pca, y_train,  method = "rf",  
        tuneGrid = data.frame(mtry = train_rf$bestTune$mtry),
        nodesize = ns)$results$Accuracy 
})

nodesize_plot <- ggplot (data.frame(acc, nodesize), aes(nodesize, acc)) + # plot nodesize accuracy
  geom_point(size=6) +
  xlab("Nodesize") + ylab("Accuracy") + 
  theme( axis.text=element_text(size=18), #change font size of axis text
         axis.title=element_text(size=26)) # change font size of axis titles
nodesize_plot

ggsave("figures/nodesize_plot.png", nodesize_plot, dpi=300)

ns <- nodesize[which.max(acc)]

# fit rf with optimized parameters using all training data
fit_rf <- randomForest(x_train_pca, y_train, mtry= train_rf$bestTune$mtry,
                       nodesize = ns, importance=TRUE)

y_hat_rf <- predict (fit_rf, x_test_pca)

cm<- confusionMatrix(y_hat_rf, y_test)
cm$overall["Accuracy"]
results2 <- c("tuned rf", 
              round(cm$overall["Accuracy"],4),
              round(cm$byClass["Sensitivity"],4), 
              round(cm$byClass["Specificity"],4))

results<- cbind (results, results2)

imp<- varImp(train_rf) # calculate importance of each PC component for determining the web page type


#    ENSEMBLE  - Majority prediction
#--------------------------------------------------------------------------

# generate a majority prediction combining the 3 best performing algorithms
ensemble <- cbind (lda = factor(predictions[,2]) =="phishing", 
                   tuned_knn = y_hat_knn == "phishing", 
                   tuned_rf = y_hat_rf == "phishing")

# predict "phishing" if that is what most models suggest.
y_hat_ensemble <- ifelse(rowMeans(ensemble) >  0.5, "phishing", "legitimate")

cm<-confusionMatrix(factor(y_hat_ensemble), y_test)
cm$overall["Accuracy"]   
results2 <- c("ensemble", 
              round(cm$overall["Accuracy"],4),
              round(cm$byClass["Sensitivity"],4), 
              round(cm$byClass["Specificity"],4))

results<- cbind (results, results2)
colnames(results) <- NULL
results_final<- as.data.frame(t(results)) %>% rename (MODEL = V1) %>% arrange(desc(Accuracy))
save(results_final, file="rdas/results_final.rda")  # save for loading into rmarkdow


