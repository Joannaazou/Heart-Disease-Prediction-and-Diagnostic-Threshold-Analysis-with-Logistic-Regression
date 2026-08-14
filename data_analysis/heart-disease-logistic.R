required_packages <- c(
  "caret",
  "car",
  "corrplot",
  "pheatmap",
  "DescTools",
  "pROC"
)

missing_packages <- required_packages[
  !required_packages %in% installed.packages()[, "Package"]
]

if (length(missing_packages) > 0) {
  install.packages(missing_packages)
}

invisible(lapply(required_packages, library, character.only = TRUE))

################################################################

#read and check data
heart_raw <- read.csv("heart_disease_cleveland.csv")
dim(heart_raw)
head(heart_raw)
colnames(heart_raw)
str(heart_raw)

#data cleaning
sum(is.na(heart_raw))
colSums(is.na(heart_raw))
heart <- na.omit(heart_raw)
sum(is.na(heart))

str(heart) #check the structure of the dataset again

#factorize
heart$sex <- factor(heart$sex)
heart$cp <- factor(heart$cp)
heart$fbs <- factor(heart$fbs)
heart$restecg <- factor(heart$restecg)
heart$exang <- factor(heart$exang)
heart$slope <- factor(heart$slope)
heart$thal <- factor(heart$thal)
heart$ca <- factor(heart$ca)
heart$disease <- factor(
  heart$disease,
  levels = c(0, 1),
  labels = c("No", "Yes"))

str(heart) #check the structure again

#exploratory analysis
table(heart$disease)
table(heart$thal, heart$disease)
table(heart$cp, heart$disease)
summary(heart)
barplot(
  table(heart$cp, heart$disease),
  beside = TRUE,
  legend = TRUE,
  xlab = "Chest Pain Type",
  ylab = "Number of Patients",
  main = "Chest Pain Type and Heart Disease"
) #seems cp has an infludence on disease
barplot(
  table(heart$thal, heart$disease),
  beside = TRUE,
  legend = TRUE,
  xlab = "Chest Pain Type",
  ylab = "Number of Patients",
  main = "Chest Pain Type and Heart Disease"
) 
barplot(
  table(heart$sex, heart$disease),
  beside = TRUE,
  legend = TRUE,
  xlab = "Chest Pain Type",
  ylab = "Number of Patients",
  main = "Chest Pain Type and Heart Disease"
) 


##for continuous variables
boxplot(age ~ disease, data = heart) #the distribution seems to be different
boxplot(chol ~ disease, data = heart) #the distribution is slightly different
boxplot(oldpeak ~ disease, data = heart) #different distribution

#look at the heatmap to briefly(!!) check for multicollinearity
##continuous variable -> spearman(look at the trend)
library(corrplot)
continuous_vars <- c(
  "age",
  "trestbps",
  "chol",
  "thalach",
  "oldpeak"
)
cor_cont <- cor(
  heart[, continuous_vars],
  method = "spearman",
  use = "complete.obs"
)
corrplot(
  cor_cont,
  method = "color",
  type = "lower",
  order = "hclust",
  addCoef.col = "black",
  tl.col = "black",
  tl.srt = 45,
  col = colorRampPalette(c("blue", "white", "red"))(200),
  title = "Continuous Variables: Spearman Correlation",
  mar = c(0, 0, 2, 0)
) #seems no suspicious variables

##categorical variables -> Cramer's V heatmap (chi-squared related)
library(pheatmap)
library(DescTools)
categorical_vars <- c(
  "sex", "cp", "fbs", "restecg",
  "exang", "slope", "thal"
)
cat_data <- heart[, categorical_vars]
cramers_v <- function(x, y) {
  CramerV(table(x, y), bias.correct = TRUE)
} ### Cramér's V function

### build Cramér's V matrix
n <- ncol(cat_data)
v_mat <- matrix(
  NA,
  nrow = n,
  ncol = n,
  dimnames = list(names(cat_data), names(cat_data))
)
for (i in seq_len(n)) {
  for (j in seq_len(n)) {
    v_mat[i, j] <- cramers_v(cat_data[[i]], cat_data[[j]])
  }
}
diag(v_mat) <- 1
### heatmap
pheatmap(
  v_mat,
  cluster_rows = TRUE,
  cluster_cols = TRUE,
  display_numbers = TRUE,
  number_format = "%.2f",
  color = colorRampPalette(c("white", "orange", "red"))(100),
  main = "Categorical Variables: Cramer's V"
) #seems there are no suspicious variables




#########################################################
############Building Models##############################
#########################################################

#seperate the dataset into training set and test set
set.seed(123)
train_index <- createDataPartition(
  heart$disease,
  p = 0.80,
  list = FALSE
)
train <- heart[train_index, ]
test  <- heart[-train_index, ]
prop.table(table(train$disease))
prop.table(table(test$disease)) #check the proportion
vif(model_full) #check multicollinearity -> no severe multicollinearity

#logistic regression of the training set with all variables
model_full <- glm(
  disease ~ age + sex + cp + trestbps + chol +
    fbs + restecg + thalach + exang +
    oldpeak + slope + ca + thal,
  data = train,
  family = binomial
)
summary(model_full) 

##odds ratio table
OR_full <- exp(
  cbind(
    OR = coef(model_full),
    confint(model_full)
  )
)
round(OR_full, 3)

#step regression based on AIC
model_step <- step(
  model_full,
  direction = "both"
)
summary(model_step)




##################################################
#########COMPARISON AND MODEL SELECTION###########
##################################################

summary(model_full)
summary(model_step)
formula(model_full)
formula(model_step)



###################1#########################
##################AIC########################
AIC(model_full)
AIC(model_step)
#> AIC(model_full)
#[1] 188.1036
#> AIC(model_step) ###reduced a little, which means the generalization ability of stepwise regression model could be better
#[1] 181.918


####################2########################
#test the model's performance on the test data
pred_full <- predict(
  model_full,
  newdata = test,
  type = "response"
)

pred_step <- predict(
  model_step,
  newdata = test,
  type = "response"
)

## draw the ROC(Receiver Operating Characteristics) curve, which  is important for clinical analysis cuase it accounts for FN,FP,TN,TP
library(pROC)

roc_full <- roc(
  response = test$disease,
  predictor = pred_full,
  levels = c("No", "Yes"),
  direction = "<"
)

plot(
  roc_full,
  main = "ROC Curve - Test Set - Full Logistic Regression"
) #the shape is not too close to the left-upper corner, but still very close, which means it's relatively good

auc_full <- auc(roc_full)

roc_step <- roc(
  response = test$disease,
  predictor = pred_step,
  levels = c("No", "Yes"),
  direction = "<"
)

plot(
  roc_step,
  main = "ROC Curve - Test Set - Stepwise Logistic Regression"
)

auc_step <- auc(roc_step)

##Draw the comparison graph
plot(
  roc_full,
  main = "Full vs Stepwise Logistic Regression"
)
plot(
  roc_step,
  add = TRUE,
  lty = 2
) #no obvious difference
legend(
  "bottomright",
  legend = c(
    paste0("Full Model, AUC = ", round(auc_full, 3)),
    paste0("Stepwise Model, AUC = ", round(auc_step, 3))
  ),
  lty = c(1, 2)
)

auc_step
auc_full
#The shape of the two ROC curve are similar, but the full model has a slightly higher auc value
#> auc_step
#Area under the curve: 0.9039
#> auc_full
#Area under the curve: 0.912


#######################3#####################
#repeated 5-cv to compare stepwise model and full model

set.seed(345)
ctrl <- trainControl(
  method = "repeatedcv",
  number = 5,
  repeats = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final"
)

full_formula <- disease ~ age + sex + cp + trestbps + chol +
  fbs + restecg + thalach + exang +
  oldpeak + slope + ca + thal

set.seed(123)

cv_full <- train(
  full_formula,
  data = train,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = ctrl
)

cv_full
cv_full$results
cv_full$resample
cv_full$results$ROC

step_formula <- formula(model_step)
set.seed(123)

cv_step <- train(
  step_formula,
  data = train,
  method = "glm",
  family = binomial,
  metric = "ROC",
  trControl = ctrl
)
cv_step
cv_full$results
cv_step$results

cv_comparison <- data.frame(
  Model = c("Full", "Stepwise"),
  Mean_AUC = c(
    cv_full$results$ROC,
    cv_step$results$ROC
  ),
  SD_AUC = c(
    cv_full$results$ROCSD,
    cv_step$results$ROCSD
  )
)

cv_comparison
#> cv_comparison
#Model  Mean_AUC     SD_AUC
#1     Full 0.8918531 0.04369856
#2 Stepwise 0.8979434 0.04128832
##stepwise seems to be better based on slightly larger AUC

#############################################################
#Based on all these information I select the stepwise model
#The AIC-selected stepwise model was preferred because it achieved a lower AIC and slightly better repeated cross-validation performance while using fewer predictors
#############################################################




#########################################################
############Selecting Threshold##############################
#########################################################



###########################0#################################
###############threshold sens-spec analysis################

thresholds <- seq(0.10, 0.90, by = 0.001)

threshold_results <- data.frame()

for (t in thresholds) {
  
  pred_class <- ifelse(
    pred_step >= t,
    "Yes",
    "No"
  )
  
  pred_class <- factor(
    pred_class,
    levels = c("No", "Yes")
  )
  
  actual <- factor(
    test$disease,
    levels = c("No", "Yes")
  )
  
  cm <- table(
    Prediction = pred_class,
    Actual = actual
  )
  
  TN <- cm["No", "No"]
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  TP <- cm["Yes", "Yes"]
  
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  threshold_results <- rbind(
    threshold_results,
    data.frame(
      threshold = t,
      TP = TP,
      FP = FP,
      TN = TN,
      FN = FN,
      sensitivity = sensitivity,
      specificity = specificity
    )
  )
}

head(threshold_results)

#plot the sensitivity-specifity graph

plot(
  threshold_results$threshold,
  threshold_results$sensitivity,
  type = "l",
  ylim = c(0, 1),
  xlab = "Threshold",
  ylab = "Metric",
  main = "Sensitivity and Specificity Across Thresholds"
)

lines(
  threshold_results$threshold,
  threshold_results$specificity,
  lty = 2
)

legend(
  "right",
  legend = c("Sensitivity", "Specificity"),
  lty = c(1, 2)
)



############################1#################################
###The youden method to select threshold (maximize sensitivity + specifity - 1)

best_coords_step <- coords(
  roc_step,
  x = "best",
  best.method = "youden",
  ret = c(
    "threshold",
    "sensitivity",
    "specificity"
  )
)

best_coords_step 
#threshold sensitivity specificity
#1 0.4897795   0.8148148     0.96875
####the sensitivity is not large enough!I want to focus on sensiticity cause it's a clinical analysis (C_FN >> C_FP)
best_threshold_step_youden <- best_coords_step$threshold
best_threshold_step_youden #0.4897795

#build the confusion matrix based on my selected threshold
pred_class_step <- ifelse(
  pred_step >= best_threshold_step,
  "Yes",
  "No"
)

pred_class_step <- factor(
  pred_class_step,
  levels = c("No", "Yes")
)
cm_step <- confusionMatrix(
  data = pred_class_step,
  reference = test$disease,
  positive = "Yes"
)

cm_step



############################2#################################
###another method to select the optimal threshold value:
###since it's a clinical analysis I want to focus on sensitivity cause FN cause a lot
###so I set the minimum acceptible sensitivity to be 0.85

eligible <- subset(
  threshold_results,
  sensitivity >= 0.85
)

best_coords_step_set <- eligible[
  which.max(eligible$specificity),
]

best_coords_step_set
#   threshold TP FP TN FN sensitivity specificity
#283     0.382 23  4 28  4   0.8518519       0.875
best_threshold_step_set <- best_coords_step_set["threshold"]
best_threshold_step_set



#############################3#################################
#determine threshold based on social utility, assume:
#U_TP = +10
#U_TN = +5
#U_FP = -2
#U_FN = -15
#U = 10TP + 5TN − 2FP − 15FN

#define social utility
thresholds_u <- sort(unique(pred_step))
utility_TP <- 10
utility_TN <- 5
utility_FP <- -2
utility_FN <- -15

#evaluate every threshold

utility_results <- data.frame()

for (t in thresholds_u) {
  
  pred_class <- ifelse(
    pred_step >= t,
    "Yes",
    "No"
  )
  
  pred_class <- factor(
    pred_class,
    levels = c("No", "Yes")
  )
  
  actual <- factor(
    test$disease,
    levels = c("No", "Yes")
  )
  
  # Confusion matrix
  cm <- table(
    Prediction = pred_class,
    Actual = actual
  )
  
  TN <- cm["No", "No"]
  FP <- cm["Yes", "No"]
  FN <- cm["No", "Yes"]
  TP <- cm["Yes", "Yes"]
  
  # Calculate utility
  total_utility <-
    utility_TP * TP +
    utility_TN * TN +
    utility_FP * FP +
    utility_FN * FN
  
  # Performance metrics
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  utility_results <- rbind(
    utility_results,
    data.frame(
      threshold = t,
      TP = TP,
      FP = FP,
      TN = TN,
      FN = FN,
      sensitivity = sensitivity,
      specificity = specificity,
      utility = total_utility
    )
  )
}

#find the threshold with maximum social utility 
best_utility <- utility_results[
  which.max(utility_results$utility),
]

best_utility
##> best_utility
##threshold TP FP TN FN sensitivity specificity utility
##30 0.3828664 23  4 28  4   0.8518519       0.875     302

#extract the best utility threshold
best_social_threshold <- best_utility$threshold

cat(
  "Social utility optimal threshold =",
  best_social_threshold,
  "\n"
)

#confusion matrix based on best utility threshold
pred_class_social <- ifelse(
  pred_step >= best_social_threshold,
  "Yes",
  "No"
)

pred_class_social <- factor(
  pred_class_social,
  levels = c("No", "Yes")
)

cm_social <- confusionMatrix(
  data = pred_class_social,
  reference = test$disease,
  positive = "Yes"
)

cm_social

#plot the utility out
plot(
  utility_results$threshold,
  utility_results$utility,
  type = "l",
  xlab = "Classification Threshold",
  ylab = "Total Social Utility",
  main = "Social Utility Across Classification Thresholds"
)

abline(
  v = best_social_threshold,
  lty = 2
)

#############################4#################################
#combine the threshold values into one graph
t_youden <- best_threshold_step_youden
t_sens85 <- best_threshold_step_set
t_utility <- best_utility$threshold

plot(
  threshold_results$threshold,
  threshold_results$sensitivity,
  type = "l",
  ylim = c(0, 1),
  xlab = "Classification Threshold",
  ylab = "Sensitivity / Specificity",
  main = "Threshold Sensitivity Analysis"
)

# Plot specificity

lines(
  threshold_results$threshold,
  threshold_results$specificity,
  lty = 2
)

# Youden threshold

abline(
  v = t_youden,
  col = "red",
  lty = 3,
  lwd = 2
)

# Sensitivity >= 0.85 threshold

abline(
  v = t_sens85,
  col = "blue",
  lty = 4,
  lwd = 2
)


# Social utility threshold

abline(
  v = t_utility,
  col = "darkgreen",
  lty = 5,
  lwd = 2
)

# Legend

legend(
  "bottomleft",
  legend = c(
    "Sensitivity",
    "Specificity",
    "Youden threshold",
    "Sensitivity ≥ 0.85 threshold",
    "Social utility threshold"
  ),
  lty = c(
    1,
    2,
    3,
    4,
    5
  ),
  col = c(
    "black",
    "black",
    "red",
    "blue",
    "darkgreen"
  ),
  lwd = c(
    1,
    1,
    2,
    2,
    2
  ),
  cex = 0.75,
  bty = "n"
)

