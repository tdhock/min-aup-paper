library(data.table)
library(ggplot2)
test.dt <- data.table::fread("figure-l2-vs-early-stopping-spam-data.csv")
gg <- ggplot()+
  geom_point(aes(
    test.metric.value, reg.param),
    data=test.dt)+
  facet_grid(
    valid.metric.name ~ train.prop.pos + test.metric.name,
    labeller=label_both, scales="free")
test.dt[, selection := paste0(
  "\n", ifelse(valid.metric.name=="auc", "max(auc)", "min(loss)"))]
test.stats <- test.dt[test.metric.name=="auc", .(
  median=median(test.metric.value),
  q25=quantile(test.metric.value, 0.25),
  q75=quantile(test.metric.value, 0.75)
), by=.(selection, train.prop.pos, reg.param)]
gg <- ggplot()+
  geom_point(aes(
    median, reg.param),
    data=test.stats)+
  geom_segment(aes(
    q25, reg.param,
    xend=q75, yend=reg.param),
    data=test.stats)+
  facet_grid(
    selection ~ train.prop.pos,
    labeller=label_both, scales="free")+
  scale_x_continuous(
    "Test AUC, median and quartiles over 10-fold CV")
png(
  "figure-l2-vs-early-stopping-spam.png",
  width=7, height=2, res=200, units="in")
print(gg)
dev.off()
