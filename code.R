# Comment utiliser ce code ------------------------------------------------

# Ce programme permet de déterminer différents modèles pour la prédiction
# des mouvements piétons. Les trois modèles proposés dans ce code sont les
# suivants :
#   - régression multiple (pas à pas)
#   - régression relative au meilleur sous-ensemble
#   - régression LASSO
# 
# Chacun de ces trois modèles sont codés sous forme de fonction. Le résultat
# de chaque fonction est une liste contenant le tableau d'entrée avec les
# résultats calculés, la formule du modèle, le R2, l'EQM, la valeur mesurée
# maximale, la valeur calculée maximale et l'écart (en pourcents) entre ces
# deux valeurs.
# 
# Pour modifier les modèles, il est possible de jouer avec les paramètres
# des fonctions dans la partie 'Paramètres'.
# 
# À la fin, le tableau 'qualite' résume l'écart, le R2 et l'EQM de chaque
# modèle pour chaque période.

rm(list = ls()); graphics.off(); cat('\014') # tout réinitialiser
set.seed(90) # pour la reproductibilité

# Paramètres --------------------------------------------------------------

# modifier ces paramètres permet d'obtenir différents modèles

##### Pour la régression multiple :

# méthode de régression
# valeurs possibles : 'leapSeq', 'leapBackward', 'leapForward'
methode = 'leapSeq'

# mesure pour trouver le meilleur modèle
# valeurs possibles : 'Rsquared', 'RMSE'
metrique = 'Rsquared' 

# cross-validation : nombre de groupes k et nombre de répétitions r
k_multi = 10
r_multi = 10

# nombre maximal de variables m pour le modèle, pour chaque période
m_multi_jmat = 2
m_multi_jmidi = 2
m_multi_jsoir = 3
m_multi_smat = 2
m_multi_smidi = 2
m_multi_ssoir = 5


##### Pour la régression relative au meilleur sous-ensemble :

# cross-validation : nombre de groupes k et nombre de répétitions r
k_subset = 10
r_subset = 10

# nombre maximal de variables m pour le modèle, pour chaque période
m_subset_jmat = 3
m_subset_jmidi = 3
m_subset_jsoir = 3
m_subset_smat = 2
m_subset_smidi = 2
m_subset_ssoir = 2


##### Pour la régression LASSO :

# cross-validation : nombre de groupes k
k_lasso = 10

# Préparation -------------------------------------------------------------

# packages
library(caret)
library(leaps)
library(dplyr)
library(tidyverse)
library(funModeling)
library(ggplot2)
library(gridExtra)
library(plyr)
library(relaxo)
library(glmnet)

# tableau des données
d_full = read.csv('var.csv', sep = ',')

# tableau des données, nombre de piétons remplacés par leur logarithme naturel
d_full_log = d_full
d_full_log$jeu_7_9 = log(d_full_log$jeu_7_9, exp(1))
d_full_log$jeu_11_13 = log(d_full_log$jeu_11_13, exp(1))
d_full_log$jeu_17_19 = log(d_full_log$jeu_17_19, exp(1))
d_full_log$sam_7_9 = log(d_full_log$sam_7_9, exp(1))
d_full_log$sam_11_13 = log(d_full_log$sam_11_13, exp(1))
d_full_log$sam_17_19 = log(d_full_log$sam_17_19, exp(1))


# Fonction : Régression multiple ------------------------------------------

# http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/154-stepwise-regression-essentials-in-r/

multi_regression = function(donnees_full, donnees_periode, nom_periode, nouveau_nom, methode, metrique, k, r, nv_max){
  
  # cross-validation
  train_control = trainControl(method = 'repeatedcv', number = k, repeats = r)
  
  # calcul du modèle
  step_model = train(
    y = donnees_periode[, length(donnees_periode)],
    x = donnees_periode[, 1:length(donnees_periode)-1],
    method = methode,
    metric = metrique,
    tuneGrid = data.frame(nvmax = 1:nv_max),
    trControl = train_control
    )
  
  # meilleur modèle
  coef = coef(step_model$finalModel, step_model$bestTune[1, 1])
  
  # une boucle for qui met dans une variable string la formule du modèle
  i = 1
  model = ''
  
  for (e in coef){
    if (names(which(coef[i] == e)) == '(Intercept)'){
      model = paste(model, e)
    }
    else{
      model = paste(model, '+', e, '*', names(which(coef[i] == e)))
    }
    i = i + 1
  }
  model = paste('exp(', model, ')')
  
  # ajout des résultats dans donnees_full
  donnees_full = donnees_full %>% mutate(
    nouveau = case_when(
      is.na(donnees_full[[nom_periode]]) ~ eval(parse(text = model)),
      TRUE ~ exp(donnees_full[[nom_periode]]))
  )
  
  colnames(donnees_full)[length(donnees_full)] = nouveau_nom
  
  # tableau pour comparer les valeurs mesurées et valeurs calculées
  comp = donnees_full
  comp = comp %>% mutate(
    nouveau = case_when(
      TRUE ~ eval(parse(text = model)))
  )
  
  comp = data.frame(comp[[nom_periode]], log(comp$nouveau, exp(1)))
  colnames(comp) = c(nom_periode, nouveau_nom)
  comp = comp[complete.cases(comp[[nom_periode]]),]
  
  # corrélation
  cor_test = cor.test(comp[[nom_periode]], comp[[nouveau_nom]])
  
  # erreurs
  comp[comp == 0] = NA
  comp$absolue = abs(comp[[nom_periode]] - comp[[nouveau_nom]])
  comp$relative = comp$absolue / comp[[nom_periode]]
  somme_relative = 0
  for (e in c(comp$relative)[complete.cases(c(comp$relative))]){
    somme_relative = somme_relative + e^2
  }
  eqm = sqrt(somme_relative / length((comp$relative)[complete.cases(c(comp$relative))]))
  
  # valeur maximale
  max_mesure = exp(max(donnees_periode[[nom_periode]]))
  max_calcul = max(donnees_full[[nouveau_nom]])
  ecart = (max_calcul - max_mesure) / max_mesure
  
  # ce que la fonction retourne
  liste = list(donnees_full, cor_test$estimate^2, eqm, coef, max_mesure, max_calcul, ecart)
  names(liste) = c('Données', 'R2', 'EQM', 'Modèle', 'Mesure max.', 'Calcul max.', 'Écart')
  return(liste)
}


# Fonction : Régression relative au meilleur sous-ensemble ----------------

# http://www.sthda.com/english/articles/37-model-selection-essentials-in-r/155-best-subsets-regression-essentials-in-r/

subset_regression = function(donnees_full, donnees_periode, nom_periode, nouveau_nom, k, r, nv_max){
  
  # calcul des modèles
  models = regsubsets(
    y = donnees_periode[, length(donnees_periode)],
    x = donnees_periode[, 1:length(donnees_periode)-1],
    nvmax = nv_max
  )
  
  # fonction pour obtenir les formules des modèles
  get_model_formula = function(id, object, outcome){
    models = summary(object)$which[id,-1]
    predictors = names(which(models == TRUE))
    predictors = paste(predictors, collapse = '+')
    as.formula(paste0(outcome, '~', predictors))
  }
  
  # fonction pour obtenir l'erreur de cross-validation d'un modèle
  get_cv_error = function(model.formula, data){
    
    # cross-validation
    train_control = trainControl(method = 'repeatedcv', number = k, repeats = r)
    
    # calcul du modèle
    cv = train(
      model.formula,
      data = data,
      method = 'lm',
      trControl = train_control
    )
    
    # erreur de cross-validation
    cv$results$RMSE
  }
  
  # calculer l'erreur de cross-validation
  model.ids = 1:nv_max
  cv.errors =  map(model.ids, get_model_formula, models, nom_periode) %>%
    map(get_cv_error, data = donnees_periode) %>%
    unlist()
  
  # sélection du modèle qui minimise l'erreur de cross-validation
  coef = coef(models, which.min(cv.errors))
  
  # une boucle for qui met dans une variable string la formule du modèle
  i = 1
  model = ''
  
  for (e in coef){
    if (names(which(coef[i] == e)) == '(Intercept)'){
      model = paste(model, e)
    }
    else{
      model = paste(model, '+', e, '*', names(which(coef[i] == e)))
    }
    i = i + 1
  }
  model = paste('exp(', model, ')')
  
  # ajout des résultats dans donnees_full
  donnees_full = donnees_full %>% mutate(
    nouveau = case_when(
      is.na(donnees_full[[nom_periode]]) ~ eval(parse(text = model)),
      TRUE ~ exp(donnees_full[[nom_periode]])),
  )
  
  colnames(donnees_full)[length(donnees_full)] = nouveau_nom
  
  # tableau pour comparer les valeurs mesurées et valeurs calculées
  comp = donnees_full
  comp = comp %>% mutate(
    nouveau = case_when(
      TRUE ~ eval(parse(text = model)))
  )
  comp = data.frame(comp[[nom_periode]], log(comp$nouveau, exp(1)))
  colnames(comp) = c(nom_periode, nouveau_nom)
  comp = comp[complete.cases(comp[[nom_periode]]),]
  
  # corrélation
  cor_test = cor.test(comp[[nom_periode]], comp[[nouveau_nom]])
  
  # erreurs
  comp[comp == 0] = NA
  comp$absolue = abs(comp[[nom_periode]] - comp[[nouveau_nom]])
  comp$relative = comp$absolue / comp[[nom_periode]]
  somme_relative = 0
  for (e in c(comp$relative)[complete.cases(c(comp$relative))]){
    somme_relative = somme_relative + e^2
  }
  eqm = sqrt(somme_relative / length((comp$relative)[complete.cases(c(comp$relative))]))
  
  # valeur maximale
  max_mesure = exp(max(donnees_periode[[nom_periode]]))
  max_calcul = max(donnees_full[[nouveau_nom]])
  ecart = (max_calcul - max_mesure) / max_mesure
  
  # ce que la fonction retourne
  liste = list(donnees_full, cor_test$estimate^2, eqm, coef, max_mesure, max_calcul, ecart)
  names(liste) = c('Données', 'R2', 'EQM', 'Modèle', 'Mesure max.', 'Calcul max.', 'Écart')
  return(liste)
}


# Fonction : LASSO --------------------------------------------------------

# https://www.pluralsight.com/guides/linear-lasso-and-ridge-regression-with-r

lasso_regression = function(donnees_full, donnees_periode, nom_periode, nouveau_nom, nfolds){
  
  # nom des colonnes
  cols = names(donnees_periode[1:length(donnees_periode)-1])
  
  # les lambdas que la cross-validation va évaluer
  lambdas = 10^seq(2, -3, by = -0.1)
  
  # cross-validation pour trouver le meilleur lambda
  lasso_reg = cv.glmnet(
    as.matrix(donnees_periode[1:length(donnees_periode)-1]),
    as.matrix(donnees_periode[length(donnees_periode)]),
    alpha = 1, lambda = lambdas, standardize = TRUE, nfolds = nfolds
  )
  
  # modèle lasso avec le meilleur lambda
  lasso_model = glmnet(
    as.matrix(donnees_periode[1:length(donnees_periode)-1]),
    as.matrix(donnees_periode[length(donnees_periode)]),
    alpha = 1, lambda = lasso_reg$lambda.1se, standardize = TRUE
  )
  
  # une boucle for qui met dans une variable string la formule du modèle
  model = ''

  for (e in 1:length(cols)){
    model = paste(model, '+', lasso_model$beta[e], '*', cols[e])
  }

  model = paste('exp(', lasso_model$a0, model, ')')
  
  # ajout des résultats dans donnees_full
  donnees_full = donnees_full %>% mutate(
    nouveau = case_when(
    is.na(donnees_full[[nom_periode]]) ~ eval(parse(text = model)),
    TRUE ~ exp(donnees_full[[nom_periode]]))
    )
  
  colnames(donnees_full)[length(donnees_full)] = nouveau_nom
  
  # tableau pour comparer les valeurs mesurées et valeurs calculées
  comp = donnees_full
  comp = comp %>% mutate(
    nouveau = case_when(
      TRUE ~ eval(parse(text = model)))
  )
  
  comp = data.frame(comp[[nom_periode]], log(comp$nouveau, exp(1)))
  colnames(comp) = c(nom_periode, nouveau_nom)
  comp = comp[complete.cases(comp[[nom_periode]]),]
  
  # corrélation
  cor_test = cor.test(comp[[nom_periode]], comp[[nouveau_nom]])
  
  # erreurs
  comp[comp == 0] = NA
  comp$absolue = abs(comp[[nom_periode]] - comp[[nouveau_nom]])
  comp$relative = comp$absolue / comp[[nom_periode]]
  somme_relative = 0
  for (e in c(comp$relative)[complete.cases(c(comp$relative))]){
    somme_relative = somme_relative + e^2
  }
  eqm = sqrt(somme_relative / length((comp$relative)[complete.cases(c(comp$relative))]))
  
  # valeur maximale
  max_mesure = exp(max(donnees_periode[[nom_periode]]))
  max_calcul = max(donnees_full[[nouveau_nom]])
  ecart = (max_calcul - max_mesure) / max_mesure
  
  # ce que la fonction retourne
  liste = list(donnees_full, cor_test$estimate^2, eqm, model, max_mesure, max_calcul, ecart, lasso_reg$lambda.1se)
  names(liste) = c('Données', 'R2', 'EQM', 'Modèle', 'Mesure max.', 'Calcul max.', 'Écart', 'Lambda')
  return(liste)
}

# Calcul : Régression multiple --------------------------------------------

# extraction des données pour les segments avec comptages
d_jmat = subset(d_full_log, select = -c(fid, compteur, jeu_11_13, jeu_17_19, sam_7_9, sam_11_13, sam_17_19))
d_ped_jmat = d_jmat[complete.cases(d_jmat),]
d_jmidi = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_17_19, sam_7_9, sam_11_13, sam_17_19))
d_ped_jmidi = d_jmidi[complete.cases(d_jmidi),]
d_jsoir = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, sam_7_9, sam_11_13, sam_17_19))
d_ped_jsoir = d_jsoir[complete.cases(d_jsoir),]
d_smat = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_11_13, sam_17_19))
d_ped_smat = d_smat[complete.cases(d_smat),]
d_smidi = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_7_9, sam_17_19))
d_ped_smidi = d_smidi[complete.cases(d_smidi),]
d_ssoir = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_7_9, sam_11_13))
d_ped_ssoir = d_ssoir[complete.cases(d_ssoir),]

# calcul de la régression multiple pour les 6 périodes
multi_jmat = multi_regression(d_full_log, d_ped_jmat, 'jeu_7_9', 'jmat', methode, metrique, k_multi, r_multi, m_multi_jmat)
multi_jmidi = multi_regression(multi_jmat$Données, d_ped_jmidi, 'jeu_11_13', 'jmidi', methode, metrique, k_multi, r_multi, m_multi_jmidi)
multi_jsoir = multi_regression(multi_jmidi$Données, d_ped_jsoir, 'jeu_17_19', 'jsoir', methode, metrique, k_multi, r_multi, m_multi_jsoir)
multi_smat = multi_regression(multi_jsoir$Données, d_ped_smat, 'sam_7_9', 'smat', methode, metrique, k_multi, r_multi, m_multi_smat)
multi_smidi = multi_regression(multi_smat$Données, d_ped_smidi, 'sam_11_13', 'smidi', methode, metrique, k_multi, r_multi, m_multi_smidi)
multi_ssoir = multi_regression(multi_smidi$Données, d_ped_ssoir, 'sam_17_19', 'ssoir', methode, metrique, k_multi, r_multi, m_multi_ssoir)

# mise en forme des données
multi_resultats = subset(multi_ssoir$Données, select = c(fid, jmat, jmidi, jsoir, smat, smidi, ssoir))
multi_noms = c('jmat', 'jmidi', 'jsoir', 'smat', 'smidi', 'ssoir')
multi_ecart = c(multi_jmat$Écart, multi_jmidi$Écart, multi_jsoir$Écart, multi_smat$Écart, multi_smidi$Écart, multi_ssoir$Écart)
multi_r2 = c(multi_jmat$R2, multi_jmidi$R2, multi_jsoir$R2, multi_smat$R2, multi_smidi$R2, multi_ssoir$R2)
multi_eqm = c(multi_jmat$EQM, multi_jmidi$EQM, multi_jsoir$EQM, multi_smat$EQM, multi_smidi$EQM, multi_ssoir$EQM)
multi_qualite = data.frame(multi_noms, multi_ecart, multi_r2, multi_eqm)
names(multi_qualite) = c('noms', 'multi_ecart', 'multi_r2', 'multi_eqm')


# Calcul : Régression relative au meilleur sous-ensemble ------------------

# calcul de la régression relative au meilleur sous-ensemble pour les 6 périodes
subset_jmat = subset_regression(d_full_log, d_ped_jmat, 'jeu_7_9', 'jmat', k_subset, r_subset, m_subset_jmat)
subset_jmidi = subset_regression(subset_jmat$Données, d_ped_jmidi, 'jeu_11_13', 'jmidi', k_subset, r_subset, m_subset_jmidi)
subset_jsoir = subset_regression(subset_jmidi$Données, d_ped_jsoir, 'jeu_17_19', 'jsoir', k_subset, r_subset, m_subset_jsoir)
subset_smat = subset_regression(subset_jsoir$Données, d_ped_smat, 'sam_7_9', 'smat', k_subset, r_subset, m_subset_smat)
subset_smidi = subset_regression(subset_smat$Données, d_ped_smidi, 'sam_11_13', 'smidi', k_subset, r_subset, m_subset_smidi)
subset_ssoir = subset_regression(subset_smidi$Données, d_ped_ssoir, 'sam_17_19', 'ssoir', k_subset, r_subset, m_subset_ssoir)

# mise en forme des données
subset_resultats = subset(subset_ssoir$Données, select = c(fid, jmat, jmidi, jsoir, smat, smidi, ssoir))
subset_noms = c('jmat', 'jmidi', 'jsoir', 'smat', 'smidi', 'ssoir')
subset_ecart = c(subset_jmat$Écart, subset_jmidi$Écart, subset_jsoir$Écart, subset_smat$Écart, subset_smidi$Écart, subset_ssoir$Écart)
subset_r2 = c(subset_jmat$R2, subset_jmidi$R2, subset_jsoir$R2, subset_smat$R2, subset_smidi$R2, subset_ssoir$R2)
subset_eqm = c(subset_jmat$EQM, subset_jmidi$EQM, subset_jsoir$EQM, subset_smat$EQM, subset_smidi$EQM, subset_ssoir$EQM)
subset_qualite = data.frame(subset_noms, subset_ecart, subset_r2, subset_eqm)
names(subset_qualite) = c('noms', 'subset_ecart', 'subset_r2', 'subset_eqm')


# Calcul : LASSO ----------------------------------------------------------

# standardiser les valeurs, sauf le nombre de piétons
colonnes = names(subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_7_9, sam_11_13, sam_17_19)))
pre = preProcess(d_full_log[,colonnes], method = c('center', 'scale'))
d_full_log[,colonnes] = predict(pre, d_full_log[,colonnes])

# extraction des données pour les segments avec comptages (données standardisées)
d_jmat = subset(d_full_log, select = -c(fid, compteur, jeu_11_13, jeu_17_19, sam_7_9, sam_11_13, sam_17_19))
d_ped_jmat = d_jmat[complete.cases(d_jmat),]
d_jmidi = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_17_19, sam_7_9, sam_11_13, sam_17_19))
d_ped_jmidi = d_jmidi[complete.cases(d_jmidi),]
d_jsoir = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, sam_7_9, sam_11_13, sam_17_19))
d_ped_jsoir = d_jsoir[complete.cases(d_jsoir),]
d_smat = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_11_13, sam_17_19))
d_ped_smat = d_smat[complete.cases(d_smat),]
d_smidi = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_7_9, sam_17_19))
d_ped_smidi = d_smidi[complete.cases(d_smidi),]
d_ssoir = subset(d_full_log, select = -c(fid, compteur, jeu_7_9, jeu_11_13, jeu_17_19, sam_7_9, sam_11_13))
d_ped_ssoir = d_ssoir[complete.cases(d_ssoir),]

# calcul de la régression LASSO pour les 6 périodes
lasso_jmat = lasso_regression(d_full_log, d_ped_jmat, 'jeu_7_9', 'jmat', k_lasso)
lasso_jmidi = lasso_regression(lasso_jmat$Données, d_ped_jmidi, 'jeu_11_13', 'jmidi', k_lasso)
lasso_jsoir = lasso_regression(lasso_jmidi$Données, d_ped_jsoir, 'jeu_17_19', 'jsoir', k_lasso)
lasso_smat = lasso_regression(lasso_jsoir$Données, d_ped_smat, 'sam_7_9', 'smat', k_lasso)
lasso_smidi = lasso_regression(lasso_smat$Données, d_ped_smidi, 'sam_11_13', 'smidi', k_lasso)
lasso_ssoir = lasso_regression(lasso_smidi$Données, d_ped_ssoir, 'sam_17_19', 'ssoir', k_lasso)

# mise en forme des données
lasso_resultats = subset(lasso_ssoir$Données, select = c(fid, jmat, jmidi, jsoir, smat, smidi, ssoir))
lasso_noms = c('jmat', 'jmidi', 'jsoir', 'smat', 'smidi', 'ssoir')
lasso_ecart = c(lasso_jmat$Écart, lasso_jmidi$Écart, lasso_jsoir$Écart, lasso_smat$Écart, lasso_smidi$Écart, lasso_ssoir$Écart)
lasso_r2 = c(lasso_jmat$R2, lasso_jmidi$R2, lasso_jsoir$R2, lasso_smat$R2, lasso_smidi$R2, lasso_ssoir$R2)
lasso_eqm = c(lasso_jmat$EQM, lasso_jmidi$EQM, lasso_jsoir$EQM, lasso_smat$EQM, lasso_smidi$EQM, lasso_ssoir$EQM)
lasso_qualite = data.frame(lasso_noms, lasso_ecart, lasso_r2, lasso_eqm)
names(lasso_qualite) = c('noms', 'lasso_ecart', 'lasso_r2', 'lasso_eqm')


# Mise en commun ----------------------------------------------------------

# jointure des tableaux sur la qualité des modèles
qualite = merge(multi_qualite, subset_qualite, by = 'noms')
qualite = merge(qualite, lasso_qualite, by = 'noms')

# arrondi du nouveau tableau
numeric_columns = sapply(qualite, mode) == 'numeric'
qualite[numeric_columns] = round(qualite[numeric_columns], 4)


# Exportation des résultats -----------------------------------------------

# exportation des résultats pour QGIS
write.csv(multi_resultats, file = 'multi_resultats.csv', quote = FALSE)
write.csv(subset_resultats, file = 'subset_resultats.csv', quote = FALSE)
write.csv(lasso_resultats, file = 'lasso_resultats.csv', quote = FALSE)
