setwd("D:/Desktop Data/QR/Use Cases/Use Cases Data/02 Apr 2017")
options(stringsAsFactors = TRUE)

trim <- function (x) gsub("^\\s+|\\s+$", "", x)

NextDest = read.csv("Next Likely Destination-test.csv", header = TRUE, sep = ",")
# str(NextDest)
# NextDest$COUNTRY_CODE = trim(NextDest$COUNTRY_CODE)
# NextDest$NATIONALITY = trim(NextDest$NATIONALITY)

library(lubridate)

NextDest$FLIGHT_DATE = as.Date(NextDest$FIRST_FLIGHT_DATE, "%d/%m/%Y %H:%M:%S")
NextDest$FIRST_FLIGHT_DATE = parse_date_time(NextDest$FIRST_FLIGHT_DATE, "%d/%m/%Y %H:%M:%S"); NextDest$DOB = as.Date(NextDest$DOB, "%d/%m/%Y")
NextDest$FLIGHT_MONTH = format(NextDest$FIRST_FLIGHT_DATE,"%b")
NextDest$FLIGHT_Weekday = as.numeric(format(NextDest$FIRST_FLIGHT_DATE,"%u"))
NextDest$TOW = ifelse(NextDest$FLIGHT_Weekday<=2,"WeekStart",ifelse(NextDest$FLIGHT_Weekday<=4,"WeekMid","WeekEnd"))
NextDest$FLIGHT_WOY = format(NextDest$FIRST_FLIGHT_DATE,"%W")
NextDest$FLIGHT_DOM = as.numeric(format(NextDest$FIRST_FLIGHT_DATE,"%d"))
NextDest$TOM = ifelse(NextDest$FLIGHT_DOM<10,"MonthStart",ifelse(NextDest$FLIGHT_DOM<21,"MonthMid","MonthEnd"))
NextDest$Age = as.numeric((Sys.Date() - NextDest$DOB)/365)
NextDest$AgeGroup = ifelse(NextDest$Age<31,"<=30",ifelse(NextDest$Age<41,"31-40",ifelse(NextDest$Age<51,"41-50",ifelse(NextDest$Age<61,"51-60","60+"))))

NextDest = NextDest[!(NextDest$NATIONALITY == ""),]
NextDest = NextDest[!(is.na(NextDest$FLIGHT_DATE)),]

##======================= Identifying DESTINATION based on Length of Stay (LOS) ================================================##

library(dplyr)
NextDest = NextDest %>%
  group_by(PNR_NUM, CUSTOMER_PROFILE_ID) %>%
  arrange(PNR_NUM, CUSTOMER_PROFILE_ID, FLIGHT_DATE) %>%
  mutate(LOS = FLIGHT_DATE - lag(FLIGHT_DATE, default=NULL)) %>%
  mutate(maxLOS = max(LOS,na.rm = TRUE))

temp = subset(NextDest, (NextDest$LOS == NextDest$maxLOS & !is.na(NextDest$LOS)), select=c("PNR_NUM","CUSTOMER_PROFILE_ID","ORIGIN","LOS","ORIG_COUNTRY","ORIG_REGION"))

NextDest1 = left_join(NextDest, temp, by = c("PNR_NUM" = "PNR_NUM","CUSTOMER_PROFILE_ID" = "CUSTOMER_PROFILE_ID","maxLOS" = "LOS"))

NextDest1$Final_DEST = ifelse(is.na(NextDest1$ORIGIN.y),as.character(NextDest1$DESTINATION),as.character(NextDest1$ORIGIN.y))
NextDest1$Final_DEST_COUNTRY = ifelse(is.na(NextDest1$ORIG_COUNTRY.y),as.character(NextDest1$DEST_COUNTRY),as.character(NextDest1$ORIG_COUNTRY.y))
NextDest1$Final_DEST_REGION = ifelse(is.na(NextDest1$ORIG_REGION.y),as.character(NextDest1$DEST_REGION),as.character(NextDest1$ORIG_REGION.y))

NextDest1$ORIGIN.y = NULL;NextDest1$ORIG_COUNTRY.y = NULL;NextDest1$ORIG_REGION.y = NULL


##============================== Picking Origin based on earliest departure ==================##

NextDest1 = NextDest1 %>%
            group_by(PNR_NUM) %>%
            mutate(count = order(PNR_NUM, CUSTOMER_PROFILE_ID, FIRST_FLIGHT_DATE))
  
NextDest2 = NextDest1[(NextDest1$count == 1),]

NextDest2$Final_DEST_COUNTRY = ifelse(NextDest2$Final_DEST == "WDH", "NA",as.character(NextDest2$Final_DEST_COUNTRY))

# NextDest$COREqOrig = ifelse(NextDest$COUNTRY_CODE == NextDest$ORIG_COUNTRY,1,0)
# 
# NextDest = NextDest[!(NextDest$COREqOrig == 0),]

NextDest2$FLIGHT_MONTH = as.factor(NextDest2$FLIGHT_MONTH); NextDest2$TOW = as.factor(NextDest2$TOW); NextDest2$NATIONALITY = as.factor(NextDest2$NATIONALITY)
NextDest2$TOM = as.factor(NextDest2$TOM); NextDest2$AgeGroup = as.factor(NextDest2$AgeGroup); NextDest2$ORIG_COUNTRY.x = as.factor(NextDest2$ORIG_COUNTRY.x)
NextDest2$DESTINATION = as.factor(NextDest2$DESTINATION); NextDest2$ORIGIN.x = as.factor(NextDest2$ORIGIN.x); NextDest2$ORIG_REGION.x = as.factor(NextDest2$ORIG_REGION.x)
NextDest2$Final_DEST = as.factor(NextDest2$Final_DEST); NextDest2$Final_DEST_COUNTRY = as.factor(NextDest2$Final_DEST_COUNTRY); NextDest2$Final_DEST_REGION = as.factor(NextDest2$Final_DEST_REGION);

## Seperate estimation for F/J and Y cabins ##

library(caTools)
set.seed(3000) # to get the same split everytime

NextDest_Y = NextDest2[which(NextDest2$CABIN_CLASS == "Y"),]
NextDest_Y$Dest_DXB = ifelse(NextDest_Y$Final_DEST == "DXB",1,0)
NextDest_Y$Dest_DXB.Factor = as.factor(ifelse(NextDest_Y$Final_DEST == "DXB",1,-1))
spl = sample.split(NextDest_Y$Dest_DXB, SplitRatio = 0.8)
Train = subset(NextDest_Y, spl==TRUE)
Test = subset(NextDest_Y, spl==FALSE)

#===================== ## Classification Trees ## ==========================================#

library(rpart)
library(rpart.plot)

NextDest_Tree = rpart(Final_DEST ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + NATIONALITY,
                      data = Train, na.action = na.exclude, method = "class", cp = 0.001)

rpart.plot(NextDest_Tree)
prp(NextDest_Tree)

summary(NextDest_Tree)

PredictCART = predict(NextDest_Tree ,newdata=Test, type = 'response')
p.PredictCART <- apply(yhat.boost, 1, which.max)

dest.yhat.CART = colnames(PredictCART)[p.PredictCART]
sum(Test$Final_DEST == dest.yhat.CART)/nrow(Test)

#====================== ## Boosted Trees ## ===========================================#

# write.csv(NextDest_Y,file="D:/Desktop Data/QR/Use Cases/Use Cases Data/02 Apr 2017/NextDest_Y.csv")

library(gbm)
set.seed(3000)
boost.dest=gbm(Dest_DXB ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY,data=Train,distribution = 'multinomial',n.trees=100, interaction.depth=4, shrinkage = 0.005)
summary(boost.dest)

yhat.boost=predict(boost.dest ,newdata=Test, n.trees=100, type = 'response')
p.yhat.boost <- apply(yhat.boost, 1, which.max)
# yhat.boost[1:6,,]
# head(p.yhat.boost)
# head(Test$Final_DEST)
dest.yhat.boost = colnames(yhat.boost)[p.yhat.boost]
sum(Test$Dest_DXB == dest.yhat.boost)/nrow(Test)

table(Test$Dest_DXB,dest.yhat.boost)
#====================== ## Extra Gradient Boosted Trees ## ===========================================#

library(Matrix)
sparse_matrix <- sparse.model.matrix(Final_DEST ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY-1, data = Train)
output_matrix = sparse.model.matrix(~Final_DEST-1, data = Train)

library(xgboost)
xgb <- xgboost(data = sparse_matrix, 
               label = temp, 
               eta = 0.1,
               max_depth = 15, 
               nround=25, 
               subsample = 0.5,
               colsample_bytree = 0.5,
               seed = 1,
               eval_metric = "merror",
               objective = "multi:softprob",
               num_class = 78,
               nthread = 3
)
temp = as.matrix(output_matrix)

y_pred <- predict(xgb, data.matrix(Test))

table(Test$Final_DEST,Test$Final_DEST == dest.yhat.boost)

table(Test$Final_DEST)


#====================== ## Random FOrests ## ===========================================#

library(randomForest)
set.seed(3000)
rf.dest = randomForest(Final_DEST ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY, data=Train, importance =TRUE, ntree=10, nodesize=30)

#====================== ## Logisitic Regression ## ===========================================#

DXB_Log = glm(Dest_DXB ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY, data=Train, family=binomial)
summary(DXB_Log)

predictDXB_Log = predict(DXB_Log, type="response",newdata = Test)
table(Test$Dest_DXB, predictDXB_Log > 0.5)

library(ROCR)
ROCRpred = prediction(predictDXB_Log, Train$Dest_DXB)
ROCRperf = performance(ROCRpred, "tpr", "fpr") #This defines what we want to plot on x & y axis
plot(ROCRperf)
plot(ROCRperf, colorize=TRUE)
plot(ROCRperf, colorize=TRUE, print.cutoffs.at=seq(0,1,by=0.1), text.adj=c(-0.2,1.7))

auc = as.numeric(performance(ROCRpred, "auc")@y.values)
auc

#=================================== SVM ====================================================#

library(kernlab)
DXB_svm	=	ksvm(Dest_DXB.Factor ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY, data = Train, type="C-svc",kernel="rbfdot",C=10, scale=TRUE)

predDXB_svm <- predict(DXB_svm,Test[,c("FLIGHT_MONTH","TOW","TOM","AgeGroup","ORIG_COUNTRY.x","ORIGIN.x","NATIONALITY")])
sum(predDXB_svm==Test$Dest_DXB.Factor)/nrow(Test)

table(Test$Dest_DXB.Factor,predDXB_svm)



set.seed(1)
library(e1071 )

svmfit=svm(Dest_DXB.Factor ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY, data=Train, kernel="radial", cost=10,
           scale=FALSE, gamma = 1)
summary(svmfit)

tune.out=tune(svm, Dest_DXB.Factor ~ FLIGHT_MONTH + TOW + TOM + AgeGroup + ORIG_COUNTRY.x + ORIGIN.x + NATIONALITY, data=Train, kernel="radial",
                ranges=list(cost=c(0.1,1,10,100,1000),
                            gamma=c(0.5,1,2,3,4)))
summary(tune.out)

ypred=predict(svmfit ,Test)
table(Test$Dest_DXB.Factor,ypred)
