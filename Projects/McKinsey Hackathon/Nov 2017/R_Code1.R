library(ggplot2)
library(dplyr)
library(lubridate)

# inputting data
setwd("C:/Users/saura/Downloads/McKinsey Hackathon")
stringsAsFactors=FALSE
tr = read.csv('train_aWnotuB.csv')
te = read.csv('test_BdBKkAj.csv')

# settin date format
str(tr)
tr$NewDateTime = dmy_hm(tr$DateTime)
tr_trn = subset(tr, !(tr$YearMonth %in% c("2017-4","2017-5","2017-6")))
tr_cv = subset(tr, tr$YearMonth %in% c("2017-4","2017-5","2017-6"))

# SUbsetting data
tr1 = subset(tr, tr$Junction == 1)
tr1_trn = subset(tr1, !(tr1$YearMonth %in% c("2017-4","2017-5","2017-6")))
tr1_cv = subset(tr1, tr1$YearMonth %in% c("2017-4","2017-5","2017-6"))

#==================================================================================#
# Simple Linear Regression

#Adj R2 = 85
lm_tr1 = lm(Vehicles ~ factor(Year)+factor(Month)+factor(Day)+factor(Weekday)+factor(Hour),
            data = tr1_trn, na.action = na.exclude)

#Adj R2 = 61
lm_tr1 = lm(Vehicles ~ Year+Month+Day+Weekday+Hour,
            data = tr1_trn, na.action = na.exclude)

summary(lm_tr1)

SSE_tr1 <- sum(lm_tr1$residuals^2)
MSE_tr1 <- (1/2)*(SSE_tr1/nrow(tr1_trn))

#==================================================================================#
# Boosted Trees

library(gbm)
set.seed(3000)
boost.tr = gbm(Vehicles ~ Year+Month+Day+Weekday+Hour+Junction,
               data = tr,distribution = "gaussian",n.trees=1000,interaction.depth=3, shrinkage = 1)

summary(boost.tr)

yhat.boost=predict(boost.tr ,newdata = te, n.trees=1000)

SSE_test <- sum((yhat.boost - tr_cv$Vehicles)^2)
RMSE_test <- sqrt((SSE_test/nrow(tr_cv)))
SST_test <- sum((mean(tr_cv$Vehicles) - tr_cv$Vehicles)^2)
R2 <- 1 - SSE_test/SST_test
R2
RMSE_test

#==================================================================================#
# Moving Average

tr1$yday <- yday(tr1$NewDateTime)

library(data.table)
DT <- data.table(tr1)
DT[, index := .GRP, by = c("Junction","yday")]
tr1$DayNum = rep(1:366,each=24,length.out = nrow(tr1))
tr1$index = rep(1:nrow(tr1),each=24,length.out = nrow(tr1))
tr1 <- tr1[order(tr1$NewDateTime), ]

# the ma function expects a time series object so we first need to 
tr1_ts <- ts(tr1$Vehicles, start = 1, freq = 24)

# n is number of time points to average
n <- 24  # we have hourly data so this is a day
tr1$daily_ma <- stats::filter(tr1_ts, filter = rep(1, n)/n)
n <- 7 # we have daily data so this is a week
tr1$weekly_ma <- stats::filter(tr1_ts, filter = rep(1, n)/n)
n <- 30 # approximately a month
tr1$monthly_ma <- stats::filter(tr1_ts, filter = c(1/2, rep(1, n-1), 1/2)/n)
#n <- 365 # approximately a year
#tr1$annual_ma <- stats::filter(tr1_ts, filter = rep(1, n)/n)

qplot(NewDateTime, Vehicles, data = tr1, geom = "line", alpha = I(0.5)) +
  geom_line(aes(y = daily_ma, colour = "daily_ma"), size = 1) +
  geom_line(aes(y = weekly_ma, colour = "weekly_ma"), size = 1) +
  geom_line(aes(y = monthly_ma, colour = "monthly_ma"), size = 1) +
  scale_colour_brewer("Moving average", type = "qual", palette = 2) 

# add a straight line
qplot(NewDateTime, Vehicles, data = tr1, geom = "line") +
  geom_smooth(method = "lm")

# add a locally weighted regression line (loess) 
qplot(NewDateTime, Vehicles, data = tr1, geom = "line") +
  geom_smooth(method = "loess")

# change span of loess line
qplot(NewDateTime, Vehicles, data = tr1, geom = "line") +
  geom_smooth(method = "loess", span = 0.2)

# == Subtract == #
# calculate monthly means
day_fit <- lm(Vehicles ~ factor(yday), data = tr1, na.action = na.exclude)
tr1$day_avg <- predict(day_fit)
tr1$res <- residuals(day_fit)

Month_fit <- lm(Vehicles ~ factor(Month), data = tr1, na.action = na.exclude)
tr1$Month_fit_avg <- predict(Month_fit)
tr1$Month_res <- residuals(Month_fit)

qplot(NewDateTime, Vehicles, data = tr1, geom = "line", alpha = I(.3))+
  geom_line(aes(y=day_avg),size = 1)

qplot(NewDateTime, Vehicles, data = tr1, geom = "line") +
  geom_smooth(method = "loess", span = 0.2)

# === automatic approaches === stl and  decompose#
tr1_ts2 <- ts(tr1, start = 1, freq = 24*30.5)

plot(stl(tr1_ts, 24))
plot(decompose(tr1_ts))

# removing linear trend
tr1.df = tr1
tr1_trend <- loess(Vehicles ~ index, data = tr1.df, na.action = na.exclude)

tr1.df$trend <- predict(tr1_trend, newdata = tr1.df)
tr1.df$de_trend <- residuals(tr1_trend)
qplot(index, Vehicles, data = tr1.df, geom ="line", alpha = I(.3))+
  geom_line(aes(y=trend),size = 1)+
  ylab("Vehicles")  +
  xlab("Day")

# a multiplicative trend?
tr1.df$de_trend_mult <- tr1.df$Vehicles/tr1.df$trend

qplot(index, de_trend_mult, data = tr1.df, geom ="line") +
  ylab("Vehicles")  +
  xlab("Day")


#===================================================================================#
# compare to explicitly modelling linear trend
#fit_temp <- lm(temp ~ time, data= temp, na.action = na.exclude)

#tr1.df$lin_res <- residuals(tr1_trend)

tr1.df$lin_res <- tr1.df$de_trend_mult

qplot(index, lin_res, data = tr1.df, geom = "line")

acf(tr1.df$lin_res)
pacf(tr1.df$lin_res)

n <- nrow(tr1.df)
(fit_ma1 <- arima(tr1.df$Vehicles, order = c(0, 1, 1), xreg = 1:n))
(fit_ar2 <- arima(tr1.df$Vehicles, order = c(2, 1, 0), xreg = 1:n))
(fit_arma1 <- arima(tr1.df$Vehicles, order = c(1, 1, 1), xreg = 1:n))
(fit_ma2 <- arima(tr1.df$Vehicles, order = c(0, 1, 2), xreg = 1:n))

# MA(1) seems best, check residuals (a.k.a innovations)

# diagnostics
# is there any correlation left in the residuals
acf(residuals(fit_ma1))
pacf(residuals(fit_ma1))
# look good

# check normality
qqnorm(residuals(fit_ma1))
qqline(residuals(fit_ma1))
# a few outliers but mostly good

# a time plot of residuals
tr1.df$residuals <- residuals(fit_ma1)
qplot(index, residuals, data = tr1.df, geom = "line")

# our model looks good let's forecast Vehicle count 
pred.tr1 <- as.data.frame(predict(fit_ma1, n.ahead = 48, newxreg = (n + 1):(n+48)))
pred.tr1$index <- max(tr1.df$de_trend_mult) + (1:48)/12

qplot(index, Vehicles, data = tr1.df, geom = "line") + 
  geom_ribbon(aes(ymin = pred- 2*se, ymax = pred + 2*se, y = NULL), data = pred.tr1, alpha = 0.2) +
  geom_line(aes(y = pred), data = pred.tr1) +
  big_font

qplot(index, Vehicles, data = tr1.df, geom = "line") + 
  geom_ribbon(aes(ymin = exp(pred- 2*se), ymax = exp(pred + 2*se), y = NULL), data = pred.tr1, alpha = 0.2) +
  geom_line(aes(y = exp(pred)), data = pred.tr1) +
  big_font

#=============================================================================#
# Writing output file
output = as.data.frame(cbind(te$ID,yhat.boost))
str(output)
output = output %>% rename(ID = V1, Vehicles = yhat.boost)
write.csv(output,"C:/Users/saura/Downloads/McKinsey Hackathon/output4.csv",row.names = FALSE)
