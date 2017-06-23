library(h2o)

#Neural network with H2O package
#start h20 instance
localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
pathtodata <- paste0(normalizePath("D:/Practicum/Project Data/"),"/All Admits.xlsx", sheet = "Sheet 1")
write.table(x=Admits, file = pathtodata, row.names = F, col.names = T)
dat_h2o <- h2o.importFile(path = pathtodata, destination_frame = "Admits")
h2o.describe(dat_h2o)
dim(dat_h2o)

#model1 with a deep learning network (three layers of 50 nodes)
set.seed(54321)
model1 <- h2o.deeplearning(x=1:15,
                           y=16,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model1
plot(model1, main = "Model 1")

#model2 with three layers of 100 nodes
model2 <- h2o.deeplearning(x=1:15,
                           y=16,
                           training_frame = dat_h2o,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(100,100,100),
                           epochs = 50,
                           nfolds = 10)
model2
plot(model2, main = "Model 2")

#model3 with three layers of 50 nodes and a Tanh activation
model3 <- h2o.deeplearning(x=1:15,
                           y=16,
                           training_frame = dat_h2o,
                           activation = "TanhWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model3
plot(model3, main = "Model 3")

#model4 with three layers of 50 nodes and a Tanh activation, no dropout
set.seed(14)
model4 <- h2o.deeplearning(x=1:15,
                           y=16,
                           training_frame = dat_h2o,
                           activation = "Tanh",
                           input_dropout_ratio = 0.2,
                           hidden = c(50,50,50),
                           epochs = 50,
                           nfolds = 10)
model4
plot(model4, main = "Model 4")

#model5 with three layers of 100 nodes and a Tanh activation, no dropout
set.seed(14)
model5 <- h2o.deeplearning(x=1:15,
                           y=16,
                           training_frame = dat_h2o,
                           activation = "Tanh",
                           input_dropout_ratio = 0.2,
                           hidden = c(100,100,100),
                           epochs = 50,
                           nfolds = 10)
model5
plot(model5, main = "Model 5")


#Neural network with H2O package
#start h20 instance
#localH2O <- h2o.init(ip = "localhost", port = 54321, startH2O = TRUE)
df <- h2o.importFile(paste0(path = normalizePath("D:/Practicum/Project Data//"),"/All Admits.xlsx", sheet = "Sheet 1"))
dim(df)
df
splits <- h2o.splitFrame(df, c(0.6,0.2), seed=1234)
train  <- h2o.assign(splits[[1]], "train.hex") # 60%
valid  <- h2o.assign(splits[[2]], "valid.hex") # 20%
test   <- h2o.assign(splits[[3]], "test.hex")  # 20%


#model6 with a deep learning network (three layers of 50 nodes)
model6 <- h2o.deeplearning(x=2:16,
                           y=17,
                           set.seed(789),
                           training_frame = train,
                           validation_frame = test,
                           activation = "RectifierWithDropout",
                           input_dropout_ratio = 0.2,
                           hidden_dropout_ratios = c(0.5,0.5,0.5),
                           hidden = c(50,50,50),
                           epochs = 100,
                           nfolds = 10,
                           variable_importances = T)
model6
head(as.data.frame(h2o.varimp(model6)))

h2o.performance(model6, train = T)
h2o.performance(model6, valid = T)
h2o.performance(model6, newdata = train)
h2o.performance(model6, newdata = valid)
h2o.performance(model6, newdata = test)

DLpred <- h2o.predict(model6, test)
DLpred
test$Accuracy <- DLpred$predict == test$Enroll
1-mean(test$Accuracy)
plot(model6)

#h2o.logloss(model6, train = train, valid = valid, xval = FALSE)
