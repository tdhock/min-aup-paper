library(data.table)

## define variables which are specific to each data set.
prefix <- "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/"
esl.list <- list(
  "spam.data"=list(
    label.fun=function(dt)ncol(dt)),
  "SAheart.data"=list(
    ignore="row.names",
    transform=function(dt){
      dt[, famhist := ifelse(famhist=="Present", 1, 0)]
    },
    label.fun=function(dt)ncol(dt)),
  "zip.train.gz"=list( #TODO investigate why zero loss.
    label.fun=function(dt)1))
## code to run on each data set.
sets <- c("train", "test")
data.list <- list()
for(f in names(esl.list)){
  data.info <- esl.list[[f]]
  if(!file.exists(f)){
    u <- paste0(prefix, f)
    download.file(u, f)
  }
  full.dt <- data.table::fread(f)
  if(is.function(data.info$transform)){
    data.info$transform(full.dt)
  }
  label.col.id <- data.info$label.fun(full.dt)
  ignore.ids <- c(label.col.id, which(names(full.dt) %in% data.info$ignore))
  print(label.col.id)
  X.raw <- as.matrix(full.dt[, -ignore.ids, with=FALSE])
  y.vec <- full.dt[[label.col.id]]
  is.01 <- y.vec == 0 | y.vec == 1
  X.unsc <- X.raw[is.01, ]
  is.constant <- apply(X.unsc, 2, sd) == 0
  X.vary <- X.unsc[, !is.constant]
  range.list <- list()
  for(fn in c("min","max")){
    range.list[[fn]] <- matrix(
      apply(X.vary, 2, fn),
      nrow(X.vary),
      ncol(X.vary),
      byrow=TRUE)
  }
  X.all <- with(range.list, (X.vary-min)/(max-min)*2-1)
  y.all <- y.vec[is.01]
  set.seed(1)
  set.vec <- sample(rep(rep(sets, c(6, 1)), l=nrow(X.all)))
  for(set in sets){
    is.set <- set.vec==set
    data.list[[f]][[set]] <- list(
      x=unname(X.all[is.set,]),
      y=as.integer(y.all[is.set]))
  }
}

if(file.exists("figure-fashion-mnist-data.rds")){
  mnist.list <- readRDS("figure-fashion-mnist-data.rds")
}else{
  mnist.list <- list(
    fashion=keras::dataset_fashion_mnist(),
    MNIST=keras::dataset_mnist())
  saveRDS(data.list, "figure-fashion-mnist-data.rds")
}

for(mnist.name in names(mnist.list)){
  one.list <- mnist.list[[mnist.name]]
  for(set in sets){
    one.set <- one.list[[set]]
    is.01 <- one.set$y %in% 0:1
    data.list[[mnist.name]][[set]] <- list(
      x=matrix(one.set$x[is.01,,]/255*2-1, sum(is.01)),
      y=as.integer(one.set$y[is.01]))
  }
}

sapply(data.list, function(L)sapply(L, function(l)range(l$x)))
sapply(data.list, function(L)sapply(L, function(l)sum(0==apply(l$x, 2, sd))))
sapply(data.list, function(L)sapply(L, function(l)mean(l$y==0)))

subset.list <- function(L, n0, n1=n0){
  max.dt <- data.table(label=c(0,1), max.number=c(n0,n1))
  keep <- data.table(
    label=L$y,
    orig.index=seq_along(L$y)
  )[, label.number := 1:.N, by=label][
    max.dt, on="label"
  ][label.number <= max.number, orig.index]
  with(L, list(x=x[keep,], y=y[keep]))
}
set.prop.list <- list(
  train=c(0.01, 0.5),
  test=0.5)
prop.data.list <- list()
for(data.name in names(data.list)){
  one.data <- data.list[[data.name]]
  for(set in names(set.prop.list)){
    set.prop.vec <- set.prop.list[[set]]
    set.list <- one.data[[set]]
    y.counts <- table(set.list$y)
    balanced.list <- subset.list(set.list, min(y.counts))
    n0 <- sum(set.list$y==0)
    for(set.prop in set.prop.vec){
      n1 <- set.prop*n0/(1-set.prop)
      prop.data.list[[data.name]][[set]][[paste(set.prop)]] <-
        subset.list(balanced.list, n0, n1)
    }
  }
}

lapply(prop.data.list, sapply, sapply, function(L)mean(L$y==1))
lapply(prop.data.list, sapply, sapply, function(L)table(L$y))

get.loss.grad <- function(order.pred.vec, N.labels.vec, margin=1){
  stopifnot(N.labels.vec %in% c(0,1))
  labels.vec <- ifelse(N.labels.vec==1, 1, -1)
  label.tab <- table(labels.vec)
  denominator <- prod(label.tab)
  augmented.pred <- order.pred.vec+ifelse(N.labels.vec==1, 0, margin)
  grad.vec <- rep(NA_real_, length(order.pred.vec))
  sorted.indices <- order(augmented.pred)
  for(s in c(1, -1)){
    i <- if(s==1){
      sorted.indices
    }else{
      rev(sorted.indices)
    }
    pred.sorted <- order.pred.vec[i]
    labels.sorted <- labels.vec[i]
    I.coef <- ifelse(labels.sorted == s, 1, 0)
    z <- margin-s*pred.sorted
    quadratic <- cumsum(I.coef)
    linear <- cumsum(I.coef*s*2*z)
    constant <- cumsum(I.coef*z^2)
    ## below is only for gradient.
    grad.values <- 2*quadratic*pred.sorted + linear
    is.loss <- labels.sorted == -s
    grad.indices <- i[is.loss]
    grad.vec[grad.indices] <- grad.values[is.loss]
  }
  loss.values <- quadratic*pred.sorted^2 + linear*pred.sorted + constant
  list(
    loss=sum(loss.values[is.loss])/denominator,
    gradient=grad.vec/denominator)
}

learn <- function(subtrain, verbose=0, lambda=0, tol=1e-3){
  non.const.i <- which(apply(subtrain$x, 2, sd) > 0)
  full.weight <- rep(0, ncol(subtrain$x))
  X.subtrain <- subtrain$x[,non.const.i]
  weight.vec <- rep(0, ncol(X.subtrain))
  epoch <- 0
  crit <- Inf
  step.size <- 1
  weight.mat.list <- list()
  while(crit > tol){
    epoch <- epoch+1
    pred.vec <- X.subtrain %*% weight.vec
    loss.grad.list <- get.loss.grad(pred.vec, subtrain$y)
    loss.grad <- t(X.subtrain) %*% loss.grad.list$gradient
    pen.grad <- weight.vec * lambda/length(weight.vec)
    weight.grad <- loss.grad + pen.grad
    crit <- sum(abs(weight.grad))
    weight.after.step <- function(s)weight.vec - s * weight.grad
    cost.after.step <- function(s){
      w <- weight.after.step(s)
      loss <- get.loss.grad(X.subtrain %*% w, subtrain$y)$loss
      loss+0.5*sum(weight.vec^2)*lambda/length(weight.vec)
    }
    step.factor <- c(2, 1, 0.5, 0.1)
    step.candidates <- step.size*step.factor
    step.cost <- sapply(step.candidates, cost.after.step)
    step.size <- step.candidates[which.min(step.cost)]
    if(step.size<1e-3)step.size <- 0.1
    if(verbose)cat(sprintf(
      "epoch=%d cost=%f crit=%f best_step=%f\n",
      epoch,
      cost.after.step(0),
      crit,
      step.size))
    if(FALSE){
      roc.df <- WeightedROC::WeightedROC(pred.vec, subtrain$y)
      WeightedROC::WeightedAUC(roc.df)
    }
    weight.vec <- weight.after.step(step.size)
    full.weight[non.const.i] <- weight.vec
    weight.mat.list[[paste(epoch)]] <- full.weight
  }
  do.call(cbind, weight.mat.list)
}

loss.dt.list <- list()
for(data.name in names(prop.data.list)){
  one.data <- prop.data.list[[data.name]]
  for(prop.pos.train.labels in names(one.data$train)){
    train.list <- one.data$train[[prop.pos.train.labels]]
    set.seed(1)
    index.dt <- data.table(label=train.list$y)
    sv <- c("subtrain", "validation")
    index.dt[, set := sample(rep(sv, l=.N)), by=label]
    index.dt[, .(count=.N), by=.(label,set)]
    tlist <- list()
    for(s in sv){
      is.set <- index.dt$set==s
      tlist[[s]] <- with(train.list, list(
        x=x[is.set,],
        y=y[is.set]))
    }
    pen.weight.mat.list <- list()
    for(penalty in 10^seq(1, -5, by=-0.5)){
      print(penalty)
      learn.mat <- learn(tlist$subtrain, verbose=1, lambda=penalty)
      pen.weight.mat.list[[paste(penalty)]] <- learn.mat[,ncol(learn.mat)]
    }
    reg.type.list <- list(
      penalty=do.call(cbind, pen.weight.mat.list),
      epochs=learn(tlist$subtrain))
    for(reg.param in names(reg.type.list)){
      weight.mat <- reg.type.list[[reg.param]]
      for(set.name in names(tlist)){
        set.data <- tlist[[set.name]]
        pred.mat <- set.data$x %*% weight.mat
        loss <- apply(
          pred.mat, 2, function(pred)get.loss.grad(pred, set.data$y)$loss)
        loss.dt.list[[paste(
          data.name, prop.pos.train.labels, reg.param, set.name
        )]] <- data.table(
          data.name,
          prop.pos.train.labels,
          reg.param,
          set.name,
          reg.i=seq(0, 1, l=ncol(pred.mat)),
          reg.value=as.numeric(colnames(pred.mat)),
          loss)
      }
    }
  }
}

loss.dt <- do.call(rbind, loss.dt.list)
data.table::fwrite(loss.dt, "figure-l2-vs-early-stopping-data.csv")

