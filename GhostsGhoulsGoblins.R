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
  step_normalize(all_predictors())
prep <- prep(my_recipe)
baked_tr <- bake(prep, new_data = GGG_tr)
baked_missing <- bake(prep, new_data = GGG_missing)

rmse_vec(baked_tr[is.na(GGG_missing)], baked_missing[is.na(GGG_missing)])
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

#formats submissions properly
submission <- bind_cols(GGG_te %>% select(id), GGG_predictions$.pred_class)
submission <- submission %>% rename("id" = "id", "type" = "...2")
#writes onto a csv
vroom_write(submission, "./GGG/submission.csv", col_names = TRUE, delim = ", ")

