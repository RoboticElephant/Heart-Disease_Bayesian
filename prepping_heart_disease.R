# ---------- Libraries/Imports -----------
library(cmdstanr)
library(dplyr)
library(pROC)
check_cmdstan_toolchain(fix = TRUE, quiet = TRUE)

set.seed(123)

# -------- Description ---------
# Age: age of the patient [years]
# Sex: sex of the patient [M: Male, F: Female]
# ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
# RestingBP: resting blood pressure [mm Hg]
# Cholesterol: serum cholesterol [mg/dl]
# FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
# RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
# MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
# ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
# Oldpeak: oldpeak = ST [Numeric value measured in depression]
# ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
# HeartDisease: output class [1: heart disease, 0: Normal]

# Citation:
# fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.

# -------- Data Input ----------

df <- read.csv("data/heart.csv")

# -------- Clean Data and get it ready for Stan ----------

# convert text to numeric values. This is required for Bayes to use distributions with
df$Sex <- df$Sex %>% as.factor %>% as.numeric
df$ChestPainType <- df$ChestPainType %>% as.factor %>% as.numeric
df$RestingECG <- df$RestingECG %>% as.factor %>% as.numeric
df$ExerciseAngina <- df$ExerciseAngina %>% as.factor %>% as.numeric
df$ST_Slope <- df$ST_Slope %>% as.factor %>% as.numeric

plot(df$RestingBP, ylab = "Resting Blood Pressure", main = "Resting Blood Pressure with a zero Outlier")
# Can see that we have a single outlier. For this current case, we will just drop the outlier.
df <- df[-which.min(df$RestingBP), ]
rownames(df) <- 1:nrow(df)
plot(df$RestingBP, ylab = "Resting Blood Pressure", main = "Resting Blood Pressure (no Outliers)")

# Now plotting the cholesterol, we can see that there are values of 0, which shouldn't be.
plot(df$Cholesterol, ylab = 'Cholesterol', main = 'Multiple Incorrect Values for Cholesterol')
# We will get the index of these '0' values and try to impute them
idx.mis <- which(df$Cholesterol == 0)
# Get the observed values
idx.obs <- which(df$Cholesterol != 0)

# --------- Data ------------
# solving for HeartDisease
y.imp <- df$HeartDisease
X.chol_obs <- df$Cholesterol[idx.obs]
X.imp <- df[, !names(df) %in% c('HeartDisease', "Cholesterol")]


# --------------- Set up Impute Stan Model -------------------

# load in the data list
impute_dl <- list(N=nrow(X.imp), 
                  N_obs=length(idx.obs), N_mis=length(idx.mis),
                  ix_obs=idx.obs, ix_mis=idx.mis,
                  K=ncol(X.imp),
                  X=X.imp, x_obs=X.chol_obs,
                  y=y.imp
                  )


# compile the model
mod <- cmdstan_model('stan_models/cholesterol_impute.stan')

# build the model
fit.imp <- mod$sample(
  data = impute_dl,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500
)

# statistics
df.imp <- df
df.imp$Cholesterol[idx.mis] <- fit.imp$summary('x_mis')$mean

plot(df.imp$Cholesterol[idx.mis], main = "Imputed Cholesterol", ylab = 'Cholesterol')
plot(df.imp$Cholesterol, ylab = 'Cholesterol', main = 'All Cholesterol data points')


# --------- Split the data ---------

# Normally we would set up a model to be able to impute the missing/incorrect X values while predicting instead of imputing beforehand.
# The reason I have decided to split them was due to this data set is a combination of 5 different data sets from different areas that
# have been combined to create a single data set. These missing (because they are all relatively next to each other) means there is good
# reason that the dataset from a specific location is bad. More than likely the data would have these values and would not need to be
# imputed before running through the model. I could have made adjustments to the model to impute the missing x-values for a training
# dataset and then using this distribution determine what it would be for the test data set. I decided instead to impute all the values
# and the split them into training and test set, which could then be run through the model.

y <- df.imp$HeartDisease
X <- df.imp[, !names(df.imp) %in% c('HeartDisease')]

idx <- sample(1:nrow(X), nrow(X) * 0.8, replace = F)
X.train <- X[idx,]
y.train <- y[idx]
X.test <- X[-idx,]
y.test <- y[-idx]


# --------------- Setting up Logit Stan Model -------------------

# load in the data
data_list <- list(N=nrow(X.train), N_t=nrow(X.test), K=ncol(X.train),
                  X=X.train, X_test=X.test,
                  y=y.train, y_test=y.test
                  )

# compile the model
mod <- cmdstan_model('stan_models/heart_disease.stan')

# build the model
fit <- mod$sample(
  data = data_list,
  seed = 123,
  chains = 4,
  parallel_chains = 4,
  refresh = 500   # print update every 500 iterations
)

# statistics
ypred <- fit$summary('y_pred')$mean
# df.ypred <- fit$draws(format = "df") 
# df.ypred <- df.ypred[, grepl("y_pred", names(df.ypred))]

# ypred <- df.ypred %>% sapply(mean)

roc.score <- roc(y.test, ypred)
auc(roc.score)

# --------------------Saving fitted model objects-------------------
# save a fitted model object to disk and ensure that all of the contents are 
# available when reading the object back into R, we recommend using the 
# $save_object() method provided by CmdStanR
fit$save_object(file = "fit.RDS")

# can be read back in using readRDS
fit2 <- readRDS("fit.RDS")

