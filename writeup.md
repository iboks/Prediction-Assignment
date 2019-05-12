
```{r preliminaries, include=FALSE, cache=FALSE}
library(knitr)
opts_chunk$set(cache=TRUE)
set.seed(1)
```

The goal of the project
-----------------------

The description of the assignment contains the following information on the dataset:

> In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. 
The goal is 

> to predict the manner in which they did the exercise.

The training of the model
-------------------------

In the following, I describe the steps concerning the training of a predictive model.

### Read the data

First, the `.csv` file contain the training data is read into R. Here, unavailable values are set as `NA`.

```{r read.csv}
rawData <- read.csv("pml-training.csv", na.strings = c("NA", ""))
```

### Reduce the dataset

In the next step, I check the proportion of missing values (`NA`s) in the columns.

```{r check NAs}
propNAs <- colMeans(is.na(rawData))
table(propNAs)
```

There are `r sum(as.logical(propNAs))` columns in which almost all values (`r paste0(round(max(propNAs) * 100, 2), "%")`) are missing. If a column contains a large number of `NA`s, it will not be of great use for training the model. Hence, these columns will be removed. Only the columns without any `NA`s will be kept.

```{r remove NA columns}
# index of columns with NA values
idx <- !propNAs
# check
sum(idx)
# remove these columns  
rawDataReduced <- rawData[idx]
# check
ncol(rawDataReduced)
```

There are further unnecessary columns that can be removed. The column `X` contains the row numbers. The column `user_name` contains the name of the user. Of course, these variables cannot predictors for the type of exercise.

Furthermore, the three columns containing time stamps (`raw_timestamp_part_1`, `raw_timestamp_part_2`, and `cvtd_timestamp`) will not be used.

The factors `new_window` and `num_window` are not related to sensor data. They will be removed too.

```{r remove non-sensory columns}
# find columns not containing sensor measurement data
idx <- grep("^X$|user_name|timestamp|window", names(rawDataReduced))
# check
length(idx)
# remove columns
rawDataReduced2 <- rawDataReduced[-idx]
```

### Preparing the data for training

Now, the dataset contains one outcome column (`classe`) and `r ncol(rawDataReduced) - 1L` feature columns. The function `createDataPartition` of the `caret` package is used to split the data into a training and a cross-validation data set. Here, 70% of the data goes into the training set.

```{r data partition, message=FALSE}
library(caret)
```
```{r createDataPartition}
inTrain <- createDataPartition(y = rawDataReduced2$classe, p = 0.7, list = FALSE)
```

The index `inTrain` is used to split the data.

```{r separate datasets}
training <- rawDataReduced2[inTrain, ]
# the number of columns on the training set
nrow(training)
crossval <- rawDataReduced2[-inTrain, ]
# the number of rows in the cross-validation set
nrow(crossval)
```

### Train a model

I used the *random-forest* technique to generate a predictive model. In sum, 10 models were trained. I played around with the parameters passed to `trControl` and specified different models with bootstrapping (`method = "boot"`) and cross-validation (`method = "cv"`).

It took more than one day to train all models. Afterwards I tested their performance on the cross-validation dataset. It turned out that all models showed a good performance (because their accuracy was above 99%) though their training times were quite different.

Due to the similar performance, I will present the model with the shortest training time.

```{r load randomForest package, message=FALSE}
library(randomForest)
```

```{r train model}
trControl <- trainControl(method = "cv", number = 2)
modFit <- train(classe ~ ., data = training, method = "rf", prox = TRUE, trControl = trControl)
```

### Evaluate the model (out-of-sample error)

First, the final model is used to predict the outcome in the cross-validation dataset.

```{r predict}
pred <- predict(modFit, newdata = crossval)
```

Second, the function `confusionMatrix` is used to calculate the accuracy of the prediction.

```{r accuracy}
coMa <- confusionMatrix(pred, reference = crossval$classe)
acc <- coMa$overall["Accuracy"]
acc
```

The accuracy of the prediction is `r paste0(round(acc * 100, 2), "%")`. Hence, the *out-of-sample error* is `r paste0(round(100 - acc * 100, 2), "%")`.

### Variable importance

The five most important variables in the model and their relative importance values are:

```{r varImp}
vi <- varImp(modFit)$importance
vi[head(order(unlist(vi), decreasing = TRUE), 5L), , drop = FALSE]
```
#### The source of the data

The assignment is based on data of weight lifting exercises. It has been published:

Velloso, E.; Bulling, A.; Gellersen, H.; Ugulino, W.; Fuks, H. [Qualitative Activity Recognition of Weight Lifting Exercises](http://groupware.les.inf.puc-rio.br/har#ixzz34irPKNuZ). *Proceedings of 4th International Conference in Cooperation with SIGCHI (Augmented Human '13)*. Stuttgart, Germany: ACM SIGCHI, 2013.
