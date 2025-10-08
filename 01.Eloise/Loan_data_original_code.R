library(tidyverse)

loan <- read_csv("/kaggle/input/loan-approval-classification-data/loan_data.csv", col_types ="nffnnfnfnnnnff")
head(loan,10)

## the number of rows and columns
dim(loan)

str(loan)   ## structure of the data

glimpse(loan)   ### looking into the data

summary(loan)   ### summary of the whole data

loan %>% keep(is.numeric) %>% summary()  ### summary of the numeric data

loan %>%  keep(is.factor) %>%  summary()  ### summary of factors variables 

### looking for the variables with NAs
sapply(loan, function(x) sum(is.na(x)))

sum(is.na(loan))

### visualization


ggplot(loan, aes(x=person_age, fill=person_education, color=person_education)) + 
  geom_histogram(position="identity") 

library(Hmisc)
### histogram for all the numeric data
loan %>%  keep(is.numeric) %>% hist.data.frame()

### Barchart

loan %>% ggplot(aes(person_home_ownership, person_income, fill = person_gender)) + geom_bar(stat= "identity", position = "dodge")

loan %>% ggplot(aes(person_age, person_income, fill = person_home_ownership)) + geom_point() + stat_smooth()

loan %>% ggplot(aes(person_age, person_income, color = person_gender)) + geom_smooth()

loan %>% ggplot(aes(person_age, colour = person_education)) + geom_freqpoly(binwidth = 10)

### Data cleaning

loan <- loan %>% mutate(person_age =ifelse(person_age > 100, NA, person_age)) %>% 
  mutate(person_income =ifelse(person_income > 7200764, NA, person_income)) %>% 
  mutate(person_emp_exp =ifelse(person_emp_exp > 124, NA, person_emp_exp))

summary(loan)

loan %>% group_by(person_education) %>% summarise(Mean = mean(person_income, na.rm = TRUE)) %>% arrange(desc(Mean))

###  bar graph showing count
loan %>%  count(person_education) %>% ggplot(aes(person_education, n, fill = person_education)) + geom_bar(stat = "identity")

### bar chart for loan intent 
loan %>% count(loan_intent) %>% ggplot(aes(reorder(loan_intent,-n), n, fill = loan_intent)) + geom_bar(stat = "identity")

loan %>% count(loan_intent, person_gender) %>% ggplot(aes(reorder(loan_intent,-n), n, fill = person_gender)) + geom_bar(stat = "identity", position = "dodge")

table(loan$loan_intent, loan$person_gender)

loan %>% count(loan_intent, person_gender) %>% ggplot(aes(reorder(loan_intent,-n), n, fill = person_gender)) + geom_bar(stat = "identity", position = "dodge")

table(loan$loan_intent, loan$person_gender)

loan %>% ggplot(aes(person_income, person_age)) + geom_point(alpha = 0.2, aes(color = person_education), outlier.shape = NA)

loan %>% ggplot(aes(person_education,person_income)) + scale_y_log10() + geom_boxplot(outlier.shape = NA) + geom_jitter(alpha = 0.4, color = "tomato")

loan %>% filter(!is.na(person_age)) %>% select(person_age, person_income,person_education) %>% arrange(desc(person_income)) %>% head() 

set.seed(1234)
sample_index <- sample(nrow(loan), round(nrow(loan) * .75), replace = FALSE)

train_data <- loan[sample_index, ]
test_data <- loan[-sample_index, ]

round(prop.table(table(select(loan, loan_status))), 2)

round(prop.table(table(select(train_data, loan_status))), 2)

round(prop.table(table(select(test_data, loan_status))), 2)

### Training the model

library(rpart)
loan_mod <- rpart(loan_status ~ ., method = "class", data = train_data)

### Evaluating the model
library(rpart.plot)
rpart.plot(loan_mod)

### Prediction on test data

loan_pred <- predict(loan_mod,test_data, type = "class")

loan_pred_table <- table(test_data$loan_status, loan_pred)
loan_pred_table
sum(diag(loan_pred_table)) / nrow(test_data)

### Confusion Matrix
library(caret)
confusionMatrix(loan_pred, test_data$loan_status)

### Build the model for randomForest algorithm


loan <- na.omit(loan)

is.factor(loan$loan_status)

### Splitting the data
set.seed(1234)
sample_index <- sample(nrow(loan), round(nrow(loan) * .75), replace = FALSE)

train_data <- loan[sample_index, ]
test_data <- loan[-sample_index, ]

### Building the model

library(randomForest)

loan_forest <- randomForest(loan_status ~ ., data = train_data, mtry = 4, ntree = 2001, importance = TRUE)

loan_forest

### Prediction on train data

Randomforest_pred <- predict(loan_forest, train_data)

confusionMatrix(Randomforest_pred, train_data$loan_status)

### Prediction on test data
Randomforest_pred_test <- predict(loan_forest, test_data)

confusionMatrix(Randomforest_pred_test, test_data$loan_status)

### Important variable 

importance(loan_forest)

varImpPlot(loan_forest)

#### In conclusion, random forest algorithm is best suited for this data since it had a better accuracy of **92.84%** than that of decision tree algorithm which is **90.93%.**

