source("packages.R")

label.order.list <- list(
  bad=c(1, -1, 1, -1, -1),
  better=c(-1, -1, 1, -1, 1)
  ##best=c(-1, -1, -1, 1, 1, 1)
)
data.dt.list <- list()
pair.dt.list <- list()
for(quality.str in names(label.order.list)){
  quality <- factor(quality.str, names(label.order.list))
  label <- label.order.list[[quality.str]]
  is.positive <- label==1
  pred.list <- list(
    positive=which(is.positive),
    negative=which(!is.positive))
  x.args <- lapply(pred.list, seq_along)
  these.pairs <- data.table(do.call(expand.grid, x.args))
  for(pos.or.neg in c("positive", "negative")){
    indices <- these.pairs[[pos.or.neg]]
    pred.vals <- pred.list[[pos.or.neg]]
    set(these.pairs, j=paste0(pos.or.neg, ".pred"), value=pred.vals[indices])
  }
  ## assume pred is actually augmented predictions
  these.pairs[, diff.pred := negative.pred-positive.pred]
  these.pairs[, loss.clip := ifelse(diff.pred<0, 0, diff.pred)]
  pair.dt.list[[quality.str]] <- data.table(
    quality,
    these.pairs)
  these.data <- data.table(
    quality,
    pred=seq_along(label),
    label)
  data.dt.list[[quality.str]] <- these.data
}
data.dt <- do.call(rbind, data.dt.list)
pair.dt <- do.call(rbind, pair.dt.list)
both.dt <- rbind(
  data.table(pair.dt, loss="square"),
  data.table(pair.dt[loss.clip>0], loss="squared hinge"))
gg <- ggplot()+
  facet_grid(negative ~ quality, labeller=label_both)+
  geom_text(aes(
    pred, 0,
    label=label),
    data=data.dt)+
  geom_segment(aes(
    positive.pred, positive,
    size=loss, color=loss,
    xend=negative.pred, yend=positive),
    data=both.dt)+
  scale_size_manual(values=c("squared hinge"=1, square=2))+
  scale_color_manual(values=c("squared hinge"="red", square="black"))+
  scale_x_continuous("Augmented predictions")+
  scale_y_continuous("", breaks=0:2, labels=c("label", paste("positive:", 1:2)), limits=c(-0.5, 2))
png("figure-concept.png", width=7, height=3, res=200, units="in")
print(gg)
dev.off()
