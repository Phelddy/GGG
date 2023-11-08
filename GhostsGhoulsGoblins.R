library(tidymodels)
library(vroom)

GGG_tr <- vroom("./GGG/train.csv")
GGG_te <- vroom("./GGG/test.csv")
GGG_missing <- vroom("./GGG/trainWithMissingValues.csv")

##MEAN Implementation
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_impute_mean(all_numeric_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

rmse_vec(baked_tr[is.na(GGG_missing)], baked_missing[is.na(GGG_missing)])
#.1523626

##Bagged Tree Implementation
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_impute_bag(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) %>%
  step_smote(all_outcomes(), neighbors = 10)
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

rmse_vec(baked_tr[is.na(GGG_missing)], baked_missing[is.na(GGG_missing)])


library(ggmosaic)


library(DataExplorer)
library(GGally)
library(patchwork)

glimpse(GGG_tr)
plot_correlation(GGG_tr)
plot_histogram(GGG_tr)
plot_missing(GGG_tr)

GGG_eda <- GGG_tr %>% mutate(type = as.factor(type), color = as.factor(color))
ggplot(data = GGG_tr) + geom_mosaic(aes(x = product(bone_length), fill = type))


#.111504

library(themis)



library(embed)
library(tidyverse)
library(tidymodels)

knn <- nearest_neighbor(neighbors = tune()) %>%
  set_mode("classification") %>%
  set_engine("kknn")



knn_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(knn)

tuning_grid <- grid_regular(neighbors(c(1,200)),
                            levels = 50)
###NOTE IT REALLY LIKES 57
folds <- vfold_cv(GGG_tr, v = 5, repeats = 1)

CV_results <- knn_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- knn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")



#####
tree_mod <- rand_forest(mtry = tune(),
                        min_n = tune(),
                        trees = 1000) %>%
  set_engine("ranger") %>%
  set_mode("classification")


##Workflow
forest_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(tree_mod)


tuning_grid <- grid_regular(mtry(range = c(1, 10)),
                            min_n(),
                            levels = 5)

folds <- vfold_cv(GGG_tr, v = 5, repeats = 1)

CV_results <- forest_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- forest_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  #step_impute_bag(all_numeric_predictors()) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_normalize(all_predictors()) #%>%
  #step_smote(all_outcomes(), neighbors = 10)
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)


###SVM

library(themis)
library(vroom)

library(tensorflow)
library(embed)
library(tidyverse)
library(tidymodels)
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_normalize(all_numeric_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

folds <- vfold_cv(GGG_tr, v = 5, repeats = 1)

CV_results <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- svm_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")

library(nnet)
###NEURAL NETWORKS
nn_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  update_role(id, new_role = "id") %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min = 0, max = 1)
prep <- prep(nn_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

nn_model <- mlp(hidden_units = tune(),
                epochs = 50) %>%
  set_engine("nnet") %>%
  set_mode("classification")

nn_tuneGrid <- grid_regular(hidden_units(range = c(1, 100)),
                            levels = 5)

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

tuned_nn <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(accuracy))

tuned_nn %>% collect_metrics() %>%
  filter(.metric =="accuracy") %>%
  ggplot(aes(x= hidden_units, y = mean)) + geom_line()

bestTune <- tuned_nn %>%
  select_best("accuracy")

final_wf <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")

#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")


###BOOOOOOSTING
library(themis)
library(vroom)

library(tensorflow)
library(embed)
library(tidyverse)
library(tidymodels)
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_normalize(all_numeric_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

library(bonsai)
library(lightgbm)

boost_model <- boost_tree(tree_depth = 250,
                          trees = tune(), 
                          learn_rate = tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(trees(),
                            learn_rate(),
                            levels = 5)

folds <- vfold_cv(GGG_tr, v = 5, repeats = 1)

CV_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")

####BARTBBARTBART
bart_model <- parsnip::bart(trees = tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

tuning_grid <- grid_regular(trees(),
                            levels = 5)

folds <- vfold_cv(GGG_tr, v = 5, repeats = 1)

CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")


###SVM Scribbling
library(themis)
library(vroom)

library(tensorflow)
library(embed)
library(tidyverse)
library(tidymodels)
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_normalize(all_numeric_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

svmRadial <- svm_rbf(rbf_sigma = .00318, cost = 2.41) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)



tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

final_wf <- svm_wf %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")


##NAIVE BAYES
library(tidymodels)
library(vroom)
library(discrim)
library(themis)



library(embed)
library(tidyverse)
library(tidymodels)

GGG_tr <- vroom("./GGG/train.csv")
GGG_te <- vroom("./GGG/test.csv")
GGG_missing <- vroom("./GGG/trainWithMissingValues.csv")

##MEAN Implementation
my_recipe <- recipe(type ~ ., data = GGG_tr) %>%
  step_impute_bag(all_numeric_predictors()) %>%
  step_normalize(all_numeric_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10)

folds <- vfold_cv(GGG_tr, v = 6, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

bestTune <- CV_results %>%
  select_best("accuracy")

final_wf <- nb_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = GGG_tr)

GGG_predictions <- final_wf %>% predict(new_data = GGG_te, type = "class")


######
#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")


###MODEL STACKING
library(stacks)

untunedModel <- control_stack_grid()
tunedModel <- control_stack_resamples()

nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>%
  set_mode("classification") %>%
  set_engine("naivebayes")

nb_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(nb_model)

tuning_grid <- grid_regular(Laplace(),
                            smoothness(),
                            levels = 10)

folds <- vfold_cv(GGG_tr, v = 6, repeats = 1)

CV_results <- nb_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc),
            control = untunedModel)

svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

tuning_grid_2 <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

CV_results_2 <- svm_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid_2,
            metrics = metric_set(roc_auc),
            control = untunedModel)

my_stack <- stacks() %>%
  add_candidates(CV_results) %>%
  add_candidates(CV_results_2)

stack_mod <- my_stack %>%
  blend_predictions() %>%
  fit_members()

stackData <- as_tibble(my_stack)

GGG_predictions <- predict(stack_mod,
                                new_data = GGG_te, type = "class")


