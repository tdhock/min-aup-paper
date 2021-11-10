library(ggplot2)
loss.dt <- data.table::fread("figure-l2-vs-early-stopping-data.csv")

spam.valid.loss <- loss.dt[data.name=="spam.data"  & set.name=="validation"]
spam.valid.min <- spam.valid.loss[
  reg.param=="penalty",
  .SD[which.min(loss)],
  by=prop.pos.train.labels]
g <- ggplot()+
  geom_hline(aes(
    yintercept=loss, color=reg.param),
    data=spam.valid.min)+
  geom_line(aes(
    reg.i, loss, color=reg.param),
    data=spam.valid.loss)+
  facet_grid(
    data.name + prop.pos.train.labels ~ set.name,
    scales="free", labeller=label_both)+
  scale_y_log10()+
  scale_x_continuous(
    "Model complexity")
png(
  "figure-l2-vs-early-stopping-spam-validation.png",
  5, 5, res=200, units="in")
print(g)
dev.off()

g <- ggplot()+
  geom_line(aes(
    reg.i, loss, color=reg.param),
    data=loss.dt)+
  facet_grid(
    data.name + prop.pos.train.labels ~ set.name,
    scales="free", labeller=label_both)+
  scale_x_continuous(
    "Model complexity")+
  scale_y_log10()
png("figure-l2-vs-early-stopping.png", 7, 10, res=200, units="in")
print(g)
dev.off()
