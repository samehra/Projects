library(readxl)
library(plyr)
library(dplyr)
library(ggplot2)
library(reshape)


setwd("D:/Desktop Data/QR/Inconvenience Score/CC Data Cases")
options(stringsAsFactors = FALSE)

## Inputting the datasets

#1 Complaints dataset

file_list <- list.files()
rm(cc_data)
for (file in file_list){
  
  # if the merged dataset doesn't exist, create it
  if (!exists("cc_data")){
    cc_data <- read_excel(file, col_names = T)
      }
  
  # if the merged dataset does exist, append to it
  else if (exists("cc_data")){
    temp_dataset <-read_excel(file, col_names = T)
    cc_data<-rbind(cc_data, temp_dataset)
    rm(temp_dataset)
  }
}

# FFP_Data = read.csv("D:/Desktop Data/QR/Inconvenience Score/FFP Flight Data.csv", sep = ",")
# FFP_Data$Flight.Date1 = as.POSIXct(strptime(FFP_Data$Flight.Date, "%Y/%m/%d %H:%M:%S"))
# 
# Travel_Freq = ddply(FFP_Data, .(FFP.Number), mutate, FltPerYear = sum(Activity.Count..Flight.))

#2 FFP flight data at month year level for 2016

FFP_FltActivity = read.csv("D:/Desktop Data/QR/Inconvenience Score/Flight_Activity_2016.csv", sep = ",")

FFP_Profile = read.csv("D:/Desktop Data/QR/Inconvenience Score/Profile_28dec16.csv", sep = ",")

#3 Inconvenience scores file

Cust_Inconv_Score = read_excel("D:/Desktop Data/QR/Inconvenience Score/ComplaintCategories.xlsx", col_names = T)

# ------------------------------------------------------------------------------------------------------ #
                                                ## Data Wrangling ##

names(cc_data) = make.names(names(cc_data))

FFP_CCData = subset(cc_data, !is.na(PC.Number) & (Case.Type == "Complaint" | Case.Type == "Compliment") & Case.Action.Type != "Duplicate")
FFP_CCData$Description = FFP_CCData$Description..Category.

library(tidyr)
FFP_CCData = FFP_CCData %>%
              separate(Description..Category., c("Business.Area", "Category1", "Category2", "Category3"), ">>")

FFP_CCData$HandlingTime = as.numeric(difftime(FFP_CCData$Resolved.On, FFP_CCData$Created.On, units = "days"))

TotalDays = as.numeric(difftime(max(FFP_CCData$Created.On), min(FFP_CCData$Created.On), units = "days"))

FFP_CCData$is.compliment = ifelse(FFP_CCData$Case.Type == "Compliment", 1, 0)

FFP_CCData = FFP_CCData %>% group_by(PC.Number) %>%
            mutate(avg_HandlingTime = mean(HandlingTime, na.rm = TRUE),
                   n_Clevel = sum(!is.na(C.level)),
                   n_compliment = sum(is.compliment == 1),
                   Last_CompDate = max(Created.On)
                   )

FFP_CCData$ClevelPerYear = FFP_CCData$n_Clevel*365/TotalDays

FFP_CCData$ComplimentPerYear = FFP_CCData$n_compliment*365/TotalDays

FFP_CCData$DaysSinceLastComp = as.numeric(difftime(max(FFP_CCData$Created.On),FFP_CCData$Last_CompDate, units = "days"))

#_____________________________________________________________________________________________________________________ #

FFP_FltSummary = FFP_FltActivity %>% group_by(FFP.Number) %>%
                summarise(FltPerYear = sum(Activity.Count..Flight., na.rm = TRUE),
                          FltPerYear_F = sum(Activity.Counts..First., na.rm = TRUE),
                          FltPerYear_J = sum(Activity.Counts..Business., na.rm = TRUE),
                          FltPerYear_Y = sum(Activity.Counts..Economy., na.rm = TRUE),
                          ATV = mean(ATV.Prorated.Revenue..USD., na.rm = TRUE)
                          )

FFP_CCData$PC.Number = as.numeric(FFP_CCData$PC.Number)

FFP_CCData$Category1 = if_else(FFP_CCData$Category1 == "Check-in", "Check-In",FFP_CCData$Category1)

t = inner_join(FFP_CCData, FFP_FltSummary, by = c("PC.Number" = "FFP.Number"))

t1 = left_join(t, Cust_Inconv_Score, by = c("Business.Area" = "Business.Area", "Category1" = "Category1", "Category2" = "Category2"))

Intmed = t1 %>% group_by(PC.Number, Business.Area, Category1, Category2, ClevelPerYear, ComplimentPerYear,
                     avg_HandlingTime, DaysSinceLastComp, FltPerYear, FltPerYear_F, FltPerYear_J, FltPerYear_Y,
                     ATV, Inconv.Score) %>%
            summarise(n_complaint = sum(is.compliment == 0))

Intmed = Intmed %>% group_by(PC.Number) %>%
            mutate(Avg_Inconv_Score = mean(Inconv.Score, na.rm = TRUE))

Intmed$CompPerYear = Intmed$n_complaint*365/TotalDays
Intmed$Description = paste(Intmed$Business.Area, Intmed$Category1, Intmed$Category2, sep='>>')
Intmed$Business.Area = NULL; Intmed$Category1 = NULL; Intmed$Category2 = NULL
Intmed$n_complaint = NULL; Intmed$Inconv.Score = NULL
Intmed = Intmed[c("PC.Number","ClevelPerYear","ComplimentPerYear","avg_HandlingTime","DaysSinceLastComp",
                  "FltPerYear","FltPerYear_F","FltPerYear_J","FltPerYear_Y","ATV","Avg_Inconv_Score","Description","CompPerYear")]

Molten = cast(Intmed, PC.Number+ClevelPerYear+ComplimentPerYear+avg_HandlingTime+DaysSinceLastComp+FltPerYear+FltPerYear_F+FltPerYear_J+FltPerYear_Y+ATV+Avg_Inconv_Score~ Description)

FFP_Profile = FFP_Profile[!(FFP_Profile$TIER == ""),]
Work_file = inner_join(Molten, FFP_Profile, by = c("PC.Number" = "FFP_NUMBER"))

Work_file[is.na(Work_file)] = 0
Work_file = subset(Work_file, Avg_Inconv_Score != 0)

Work_file$TIER = as.factor(Work_file$TIER)
Work_file$BG = ifelse(Work_file$TIER == "BG",1,0)
Work_file$SL = ifelse(Work_file$TIER == "SL",1,0)
Work_file$GL = ifelse(Work_file$TIER == "GL",1,0)
Work_file$PL = ifelse(Work_file$TIER == "PL",1,0)


Work_file$COUNTRY = as.factor(Work_file$COUNTRY)
Work_file$NATIONALITY = as.factor(Work_file$NATIONALITY)
Work_file$GENDER = as.factor(Work_file$GENDER)
Work_file$DOB1 = as.Date(format(as.Date(Work_file$DOB,"%d-%b-%y"),"19%y-%m-%d"))   ## If importing data with only two digits for the years, you will find that it assumes that years 69 to 99 are 1969-1999, while years 00 to 68 are 2000-2068 ##
Work_file$New_DOB = as.Date(ifelse(as.Date(Work_file$DOB,"%d-%b-%y") > Sys.Date(),format(Work_file$DOB1,"%Y-%m-%d"),format(as.Date(Work_file$DOB,"%d-%b-%y"),"%Y-%m-%d")))
Work_file$AGE = as.numeric(Sys.Date() - Work_file$New_DOB)/365
Work_file$AgeGroup = as.factor(ifelse(Work_file$AGE<31,"<=30",ifelse(Work_file$AGE<41,"31-40",ifelse(Work_file$AGE<51,"41-50",ifelse(Work_file$AGE<61,"51-60","60+")))))

## ------------------------------------------------------------------------------------------------------ ##
                                                ## Linear Regression ##

# Partion data into training set and test set

library(caTools)
set.seed(88)
split <- sample.split(Work_file$Avg_Inconv_Score, SplitRatio = 0.75)

train <- subset(Work_file, split == TRUE)
test <- subset(Work_file, split == FALSE)
test = subset(test, !is.na(Avg_Inconv_Score))

# Splitting the data for validation dataset

# split <- sample.split(train$Avg_Inconv_Score, SplitRatio = 0.75)
# 
# train1 <- subset(train, split == TRUE)
# val <- subset(train, split == FALSE)


lm_fit = lm(Avg_Inconv_Score ~., 
            data = train[,!colnames(train) %in% c("PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","DOB1","New_DOB","AGE","ENROLLMENT_DATE","PROMO_CODE","GENDER","NATIONALITY","COUNTRY","TIER")], na.action = na.exclude)

summary(lm_fit)

SSE_train <- sum(lm_fit$residuals^2)
MSE_train <- (1/2)*(SSE_train/nrow(train))

testPredict = predict(lm_fit, newdata = test)
SSE_test <- sum((testPredict - test$Avg_Inconv_Score)^2)
MSE_test <- (1/2)*(SSE_test/nrow(test))
SST_test <- sum((mean(test$Avg_Inconv_Score) - test$Avg_Inconv_Score)^2)
R2 <- 1 - SSE_test/SST_test

# Diagonostics

Cor_matrix = as.matrix(cor(Work_file))
corplot()
res = residuals(lm_fit)

qplot(Avg_Inconv_Score, res, data = train, geom = "line") +
  geom_smooth(method = "loess", span = 0.2)

qplot(Avg_Inconv_Score, res^2, data = train, geom = "line") +
  geom_smooth(method = "loess", span = 0.2)

plot(lm_fit$fitted, lm_fit$res)

#checking leverage

lev = hat(model.matrix(lm_fit))
plot(lev) 
temp = train[lev >0.6,]

# Normality of residuals

qqnorm(lm_fit$res)
qqline(lm_fit$res)
hist(lm_fit$res)

## ------------------------------------------------------------------------------------------------------ ##
                                                ## Ridge Regression ##

## Incorporating regularization to handle large no. of variables

library(glmnet)
x.full = as.matrix(Work_file[,!colnames(Work_file) %in% c("Avg_Inconv_Score","PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")])
y.full = Work_file$Avg_Inconv_Score

x.tr = as.matrix(train[,!colnames(train) %in% c("Avg_Inconv_Score","PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")])
y.tr = train$Avg_Inconv_Score

x.test = as.matrix(test[,!colnames(test) %in% c("Avg_Inconv_Score","PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")])
y.test = test$Avg_Inconv_Score

set.seed(10)
rr.cv <- cv.glmnet(x.tr, y=y.tr, alpha = 0)
plot(rr.cv)

rr.bestlam <- rr.cv$lambda.min
rr.goodlam <- rr.cv$lambda.1se

# predict validation set using best lambda and calculate RMSE
rr.fit <- glmnet(x.tr, y.tr, alpha = 0)
plot(rr.fit, xvar = "lambda", label = TRUE)

rr.pred <- predict(rr.fit, s = rr.bestlam, newx = x.test)
rr.full <- predict(rr.fit, s = rr.bestlam, newx = x.full)
rr.MSE_train = (1/2)*(mean((rr.pred - y.test)^2))
rr.MSE_full = (1/2)*(mean((rr.full - y.full)^2))
## ------------------------------------------------------------------------------------------------------ ##
                                                  ## Regression Trees ##

library(caret)
library(rpart)
library(rpart.plot)
library(e1071)

# Define cross-validation experiment
numFolds = trainControl( method = "cv", number = 10 )
cpGrid = expand.grid( .cp = seq(0.0001,0.001,0.0001))
#This will define our cp parameters to test as numbers from 0.01 to 0.5, in increments of 0.01.

# Perform the cross validation
save_CV<-train(Avg_Inconv_Score ~., data = Train[,!colnames(Train) %in% c("PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")], method = "rpart", trControl = numFolds, tuneGrid = cpGrid )
save_CV

# Create a new CART model
RegTreeCV = rpart(Avg_Inconv_Score ~., 
                      data = train[,!colnames(train) %in% c("PC.Number","ATV","FltPerYear","MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")], cp = 0.0001) # using the cp value got from Cross validation above
prp(RegTreeCV)
printcp(RegTreeCV)

# Make predictions (Out-of-Sample predictions of the Cross Validated CART model)
PredictCV = predict(RegTreeCV, newdata = test)
PredictCV.sse = sum((PredictCV - test$Avg_Inconv_Score)^2)
PredictCV.mse = (1/2)*(PredictCV.sse/nrow(test))

## ------------------------------------------------------------------------------------------------------ ##
                                            ## Export relevant files  ##

write.csv(Work_file[c("PC.Number","ClevelPerYear","ComplimentPerYear","avg_HandlingTime","DaysSinceLastComp",
                      "FltPerYear","FltPerYear_F","FltPerYear_J","FltPerYear_Y","ATV","Avg_Inconv_Score","Description","CompPerYear",
                      "MEMBER_TYPE","STATUS","DOB","GENDER","NATIONALITY","ENROLLMENT_DATE","PROMO_CODE","COUNTRY","TIER")]
                    , file= "D:/Desktop Data/QR/Inconvenience Score/R_OutputFile.csv")


write.csv(Work_file, file= "D:/Desktop Data/QR/Inconvenience Score/Work_file.csv")

