options(tidyverse.quiet = TRUE)
library(tidyverse)
library(survival)
library(gtsummary)
library(foreign)
library(naniar)
library(smcfcs)
library(mice)
library(prodlim)
library(riskRegression)
library(splines)
library(dynpred)
library(progress)
library(wesanderson)
pal <- wes_palette(name = "FantasticFox1", n = 5)[-1]

p123 <- read.spss("P123 - age project - transfer file for Hein - 06102022.SAV", to.data.frame=TRUE)
p123 <- as_tibble(p123)
n <- nrow(p123)

# Make age categories
p123$age_4cat <- 1
p123$age_4cat[p123$age > 60] <- 2
p123$age_4cat[p123$age > 65] <- 3
p123$age_4cat[p123$age > 70] <- 4
p123$age_4cat <- factor(p123$age_4cat, levels=1:4,
                        labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
table(p123$age_4cat)

# Make competing risks outcomes
p123$orec012 <- as.numeric(p123$orecstat) - 1
p123$orec012[p123$orec012 == 2] <- 0 # these are the 999's
p123$orec012[p123$orec012 == 0 & (p123$survstat == "Dead" & p123$fupyrs <= p123$orecyrs)] <- 2
p123$adrec012 <- as.numeric(p123$adrecstat) - 1
p123$adrec012[p123$adrec012 == 2] <- 0 # these are the 999's
p123$adrec012[p123$adrec012 == 0 & (p123$survstat == "Dead" & p123$fupyrs <= p123$adrecyrs)] <- 2
p123$lrec012 <- as.numeric(p123$lrecstat) - 1
p123$lrec012[p123$lrec012 == 2] <- 0 # these are the 999's
p123$lrec012[p123$lrec012 == 0 & (p123$survstat == "Dead" & p123$fupyrs <= p123$lrecyrs)] <- 2
p123$prec012 <- as.numeric(p123$precstat) - 1
p123$prec012[p123$prec012 == 2] <- 0 # these are the 999's
p123$prec012[p123$prec012 == 0 & (p123$survstat == "Dead" & p123$fupyrs <= p123$precyrs)] <- 2
p123$vrec012 <- as.numeric(p123$vrecstat) - 1
p123$vrec012[p123$vrec012 == 2] <- 0 # these are the 999's
p123$vrec012[p123$vrec012 == 0 & (p123$survstat == "Dead" & p123$fupyrs <= p123$vrecyrs)] <- 2

# Number of events (0=censored, 1=event of interest, 2=competing risk)
table(p123$orec012)
table(p123$adrec012)
table(p123$lrec012)
table(p123$prec012)
table(p123$vrec012)

# Load imputed data
load("smcfcs_results.Rdata")
M <- length(imps_pilot$impDatasets) # number of imputations

predictors <- c("age", "stage", "Histgrade_3cat", "lvsi2",
                "molgroup", "treat")

# Change reference category for molgroup to NSMP for all imputed datasets
for (m in 1:M) {
  impm <- imps_pilot$impDatasets[[m]]
  lvls <- levels(impm$molgroup)
  impm$molgroup <- as.numeric(impm$molgroup)
  impm$molgroup[impm$molgroup == 4] <- 0
  impm$molgroup <- factor(impm$molgroup, levels=0:3, labels=lvls[c(4, 1:3)])
  imps_pilot$impDatasets[[m]] <- impm
}

###-----
# Overall recurrence

# First a plot with cumulative incidences per age category
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "age_4cat")])
ajfit <- prodlim(Hist(orecyrs, orec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
png("Orec.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Overall_recurrence.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Effect of age
imp0 <- as_tibble(p123)
imp0$lvsi2 <- imp0$LVSI_2cat
imp0$stage <- imp0$stage_4cat
imp0$molgroup <- fct_relevel(imp0$TCGA_4groups,
                             c("NSMP", "MMRd",
                               "POLE", "p53"))
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp5 <- imps_pilot$impDatasets[[5]]
imp5 <- merge(imp5, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp5$lvsi2 <- as.numeric(imp5$lvsi)
imp5$lvsi2[imp5$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp5$lvsi2 <- factor(imp5$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(orecyrs, orec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(orecyrs, orec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(orecyrs, orec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(orecyrs, orec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(orecyrs, orec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)

mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(orecyrs, orec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0)
mfit1_age <- coxph(Surv(orecyrs, orec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(orecyrs, orec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(orecyrs, orec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(orecyrs, orec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# Reviewer request, check PH assumption related to Fig 1A and B
c0 <- coxph(Surv(orecyrs, orec012==1) ~ age_4cat, data = imp0)
cox.zph(c0) # NEW!

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Orec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Orec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

# Apparent time-dependent AUC and Brier scores
score_vdata <- Score(
  list(mfit1),
  formula = Hist(orecyrs, orec012) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata

###-----
# Cancer-specific survival

load("p123sav.Rdata")
p123sav <- subset(p123sav, Cohortstudy != "MST")
# Make age categories
p123sav$age_4cat <- 1
p123sav$age_4cat[p123sav$age > 60] <- 2
p123sav$age_4cat[p123sav$age > 65] <- 3
p123sav$age_4cat[p123sav$age > 70] <- 4
p123sav$age_4cat <- factor(p123sav$age_4cat, levels=1:4,
                           labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
table(p123sav$age_4cat)
table(p123sav$survstat, p123sav$codec)
table(p123sav$codec, p123sav$study)
p123sav$codrec012 <- as.numeric(p123sav$codec) - 1
p123sav$codrec012[p123sav$codrec012 == 0] <- 2 # death unrelated and unknown cause of death are competing risk
p123sav$codrec012[p123sav$survstat == "Alive"] <- 0
p123sav$codrecyrs <- p123sav$fupyrs

# First a plot with cumulative incidences per age category
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123sav[, c("work_id", "age_4cat", "codrecyrs", "codrec012")])
ajfit <- prodlim(Hist(codrecyrs, codrec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))

png("Cancer_specific.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Cancer_specific.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Effect of age
imp0 <- p123sav
imp0$lvsi2 <- imp0$lvsi
imp0$molgroup <- imp0$TCGA_4groups_refMMRd
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp1 <- merge(imp1, p123sav[, c("work_id", "codrecyrs", "codrec012")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp2 <- merge(imp2, p123sav[, c("work_id", "codrecyrs", "codrec012")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp3 <- merge(imp3, p123sav[, c("work_id", "codrecyrs", "codrec012")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "Histgrade_3cat")])
imp4 <- merge(imp4, p123sav[, c("work_id", "codrecyrs", "codrec012")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade + lvsi2 + molgroup + received_treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age + 
                 stage + Histgrade + lvsi2 + molgroup + received_treat, data = imp0)
mfit1_age <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# Reviewer request, check PH assumption related to Fig 1A and B
c0 <- coxph(Surv(codrecyrs, codrec012 == 1) ~ age_4cat, data = p123sav)
cox.zph(c0) # NEW!

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Codrec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Codrec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

# Apparent time-dependent AUC and Brier scores
score_vdata <- Score(
  list(mfit1),
  formula = Hist(codrecyrs, codrec012 == 1) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata

###-----
# Distant recurrence

imp0 <- as_tibble(p123)
imp0$lvsi2 <- imp0$LVSI_2cat
imp0$stage <- imp0$stage_4cat
imp0$molgroup <- fct_relevel(imp0$TCGA_4groups,
                             c("NSMP", "MMRd",
                               "POLE", "p53"))
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "adrecyrs", "adrec012", "Histgrade_3cat")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "adrecyrs", "adrec012", "Histgrade_3cat")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "adrecyrs", "adrec012", "Histgrade_3cat")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "adrecyrs", "adrec012", "Histgrade_3cat")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# First a plot with cumulative incidences per age category
# Make age categories
imp1$age_4cat <- 1
imp1$age_4cat[imp1$age > 60] <- 2
imp1$age_4cat[imp1$age > 65] <- 3
imp1$age_4cat[imp1$age > 70] <- 4
imp1$age_4cat <- factor(imp1$age_4cat, levels=1:4,
                           labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
ajfit <- prodlim(Hist(adrecyrs, adrec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))

png("Distant_recurrence.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Distant_recurrence.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(adrecyrs, adrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(adrecyrs, adrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(adrecyrs, adrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(adrecyrs, adrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(adrecyrs, adrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(adrecyrs, adrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0)
mfit1_age <- coxph(Surv(adrecyrs, adrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(adrecyrs, adrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(adrecyrs, adrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(adrecyrs, adrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Adrec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Adrec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

# Apparent time-dependent AUC and Brier scores
score_vdata <- Score(
  list(mfit1),
  formula = Hist(adrecyrs, adrec012 == 1) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata

###-----
# Locoregional recurrence

imp0 <- as_tibble(p123)
imp0$lvsi2 <- imp0$LVSI_2cat
imp0$stage <- imp0$stage_4cat
imp0$molgroup <- fct_relevel(imp0$TCGA_4groups,
                             c("NSMP", "MMRd",
                               "POLE", "p53"))
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "lrecyrs", "lrec012", "Histgrade_3cat")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "lrecyrs", "lrec012", "Histgrade_3cat")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "lrecyrs", "lrec012", "Histgrade_3cat")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "lrecyrs", "lrec012", "Histgrade_3cat")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# First a plot with cumulative incidences per age category
# Make age categories
imp1$age_4cat <- 1
imp1$age_4cat[imp1$age > 60] <- 2
imp1$age_4cat[imp1$age > 65] <- 3
imp1$age_4cat[imp1$age > 70] <- 4
imp1$age_4cat <- factor(imp1$age_4cat, levels=1:4,
                        labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
ajfit <- prodlim(Hist(lrecyrs, lrec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))

png("Locoregional_recurrence.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Locoregional_recurrence.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(lrecyrs, lrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(lrecyrs, lrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(lrecyrs, lrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(lrecyrs, lrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(lrecyrs, lrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(lrecyrs, lrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0)
mfit1_age <- coxph(Surv(lrecyrs, lrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(lrecyrs, lrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(lrecyrs, lrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(lrecyrs, lrec012 == 1) ~ age + 
                     stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Lrec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Lrec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

# Apparent time-dependent AUC and Brier scores
score_vdata <- Score(
  list(mfit1),
  formula = Hist(lrecyrs, lrec012 == 1) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata

###-----
# Pelvic recurrence

imp0 <- as_tibble(p123)
imp0$lvsi2 <- imp0$LVSI_2cat
imp0$stage <- imp0$stage_4cat
imp0$molgroup <- fct_relevel(imp0$TCGA_4groups,
                             c("NSMP", "MMRd",
                               "POLE", "p53"))
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "precyrs", "prec012", "stage_2cat")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "precyrs", "prec012", "stage_2cat")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "precyrs", "prec012", "stage_2cat")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "precyrs", "prec012", "stage_2cat")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# First a plot with cumulative incidences per age category
# Make age categories
imp1$age_4cat <- 1
imp1$age_4cat[imp1$age > 60] <- 2
imp1$age_4cat[imp1$age > 65] <- 3
imp1$age_4cat[imp1$age > 70] <- 4
imp1$age_4cat <- factor(imp1$age_4cat, levels=1:4,
                        labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
ajfit <- prodlim(Hist(precyrs, prec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))

png("Pelvic_recurrence.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Pelvic_recurrence.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(precyrs, prec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(precyrs, prec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(precyrs, prec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(precyrs, prec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(precyrs, prec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp4)
mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(precyrs, prec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp0)
mfit1_age <- coxph(Surv(precyrs, prec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(precyrs, prec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(precyrs, prec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(precyrs, prec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Prec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Prec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

# Apparent time-dependent AUC and Brier scores
score_vdata <- Score(
  list(mfit1),
  formula = Hist(precyrs, prec012 == 1) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata

###-----
# Vaginal recurrence

imp0 <- as_tibble(p123)
imp0$lvsi2 <- imp0$LVSI_2cat
imp0$stage <- imp0$stage_4cat
imp0$molgroup <- fct_relevel(imp0$TCGA_4groups,
                             c("NSMP", "MMRd",
                               "POLE", "p53"))
imp1 <- imps_pilot$impDatasets[[1]]
imp1 <- merge(imp1, p123[, c("work_id", "treat", "treat_2cat", "vrecyrs", "vrec012", "stage_2cat")])
imp1$lvsi2 <- as.numeric(imp1$lvsi)
imp1$lvsi2[imp1$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp1$lvsi2 <- factor(imp1$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp2 <- imps_pilot$impDatasets[[2]]
imp2 <- merge(imp2, p123[, c("work_id", "treat", "treat_2cat", "vrecyrs", "vrec012", "stage_2cat")])
imp2$lvsi2 <- as.numeric(imp2$lvsi)
imp2$lvsi2[imp2$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp2$lvsi2 <- factor(imp2$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp3 <- imps_pilot$impDatasets[[3]]
imp3 <- merge(imp3, p123[, c("work_id", "treat", "treat_2cat", "vrecyrs", "vrec012", "stage_2cat")])
imp3$lvsi2 <- as.numeric(imp3$lvsi)
imp3$lvsi2[imp3$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp3$lvsi2 <- factor(imp3$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))
imp4 <- imps_pilot$impDatasets[[4]]
imp4 <- merge(imp4, p123[, c("work_id", "treat", "treat_2cat", "vrecyrs", "vrec012", "stage_2cat")])
imp4$lvsi2 <- as.numeric(imp4$lvsi)
imp4$lvsi2[imp4$lvsi2 == 1] <- 2 # Afwezig en Focaal aanwezig samen
imp4$lvsi2 <- factor(imp4$lvsi2, levels = 2:3, labels = c("Absent", "Present (substantial)"))

# First a plot with cumulative incidences per age category
# Make age categories
imp1$age_4cat <- 1
imp1$age_4cat[imp1$age > 60] <- 2
imp1$age_4cat[imp1$age > 65] <- 3
imp1$age_4cat[imp1$age > 70] <- 4
imp1$age_4cat <- factor(imp1$age_4cat, levels=1:4,
                        labels = c("< 60", "60 - 65", "65 - 70", "> 70"))
ajfit <- prodlim(Hist(vrecyrs, vrec012) ~ age_4cat, data = imp1)
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))

png("Vaginal_recurrence.png")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()
pdf("Vaginal_recurrence.pdf")
plot(ajfit, cause = 1, xlim = c(0, 10), ylim = c(0, 1), xlab = "Years since randomisation", col = pal,
     atrisk.at = seq(0, 10, by = 2))
dev.off()

# Natural splines
tmp <- ns(imp1$age, df = 3) # all imputed data sets have same ages (no missings)
knots <- attr(tmp, "knots") # save knots for later
Boundary.knots <- attr(tmp, "Boundary.knots") # save boundary knots for later

tmp <- as.data.frame(tmp)
names(tmp) <- paste("ns", 1:3, sep="")
imp0 <- cbind(imp0, tmp)
mfit0 <- coxph(Surv(vrecyrs, vrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + Histgrade_3cat + lvsi2 + molgroup + treat, data = imp0,
               x = TRUE)
imp1 <- cbind(imp1, tmp)
mfit1 <- coxph(Surv(vrecyrs, vrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp1,
               x = TRUE)
imp2 <- cbind(imp2, tmp)
mfit2 <- coxph(Surv(vrecyrs, vrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp2)
imp3 <- cbind(imp3, tmp)
mfit3 <- coxph(Surv(vrecyrs, vrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp3)
imp4 <- cbind(imp4, tmp)
mfit4 <- coxph(Surv(vrecyrs, vrec012 == 1) ~ ns1 + ns2 + ns3 + 
                 stage + lvsi2 + molgroup + treat_2cat, data = imp4)
mfit0 # NEW!
mfit1
mfit2
mfit3
mfit4
# mfit5 not done, used as fresh data set without ns1, ns2 and ns3 later on

mfit0_age <- coxph(Surv(vrecyrs, vrec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp0)
mfit1_age <- coxph(Surv(vrecyrs, vrec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp1)
mfit2_age <- coxph(Surv(vrecyrs, vrec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp2)
mfit3_age <- coxph(Surv(vrecyrs, vrec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp3)
mfit4_age <- coxph(Surv(vrecyrs, vrec012 == 1) ~ age + 
                     stage + lvsi2 + molgroup + treat, data = imp4)
mfit0_age
mfit1_age
mfit2_age
mfit3_age
mfit4_age

# For plot, prepare sequence of ages, and define ns values for them
ageseq <- seq(45, 85, by=0.25)
nage <- length(ageseq)
tmp <- as.matrix(ns(ageseq, df = 3, knots = knots, Boundary.knots = Boundary.knots))
# Extract regression coefficients and variance-covariance matrix of Cox model
bet <- mfit1$coef
sig <- mfit1$var

# Calculate linear predictor for age, relative to age=60, along with SE's
lp <- se.lp <- rep(0, nage)
for (j in 1:nage) {
  lp[j] <- (tmp[j, ] - tmp[61, ]) %*% bet[1:3]
  se.lp[j] <- sqrt( (tmp[j, ] - tmp[61, ]) %*% sig[1:3, 1:3] %*% (tmp[j, ] - tmp[61, ]) )
}
lp.low <- lp - qnorm(0.975) * se.lp
lp.upp <- lp + qnorm(0.975) * se.lp
# plot(ageseq, lp, ylim = range(c(lp.low, lp.upp)), type="l", lwd=2, col="blue",
#      xlab="Age", ylab="Log HR")
# lines(ageseq, lp.low, lty=2, col="blue")
# lines(ageseq, lp.upp, lty=2, col="blue")

# Plot with HR log-transformed
png("Vrec_HR_logtransform.png")
plot(ageseq, lp, ylim = c(log(0.25), log(4)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR", axes=FALSE)
lines(ageseq, lp.low, lty=2, col="blue")
lines(ageseq, lp.upp, lty=2, col="blue")
axis(1)
axis(2, at=log(c(0.25, 0.5, 1, 2, 4)), labels=c(0.25, 0.5, 1, 2, 4))
abline(h=0, lty=3)
box()
dev.off()

# Plot with HR untransformed
explp <- exp(lp)
explp.low <- exp(lp.low)
explp.upp <- exp(lp.upp)
png("Vrec_HR.png")
plot(ageseq, explp, ylim = range(c(explp.low, explp.upp)), type="l", lwd=2, col="blue",
     xlab="Age", ylab="HR")
lines(ageseq, explp.low, lty=2, col="blue")
lines(ageseq, explp.upp, lty=2, col="blue")
abline(h=1, lty=3)
dev.off()

score_vdata <- Score(
  list(mfit1),
  formula = Hist(vrecyrs, vrec012 == 1) ~ 1,
  cens.model = "km",
  data = imp1,
  conf.int = TRUE,
  times = c(3, 5, 10),
  metrics = c("auc", "brier"),
  summary = c("ipa"),
  cause = 1,
  plots = "calibration"
)
score_vdata
