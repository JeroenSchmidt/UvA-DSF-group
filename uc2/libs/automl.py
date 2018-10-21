import h2o
from h2o.estimators.random_forest import H2ORandomForestEstimator
from h2o.estimators.xgboost import H2OXGBoostEstimator
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
from h2o.grid.grid_search import H2OGridSearch
from h2o.estimators.pca import H2OPrincipalComponentAnalysisEstimator
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
from h2o.automl import H2OAutoML

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def generate_predictions_automl(train, test, exclude_algos, nfolds, seed, max_models, target):
    
    # Identify predictors and response
    x = train.columns
    y = target
    
    print(x)
    
    x.remove(y)

    # For binary classification, response should be a factor
    #train[y] = train[y].asfactor()
    #test[y] = test[y].asfactor()

    # Number of CV folds (to generate level-one data for stacking)
    nfolds = nfolds

    # Run AutoML for 30 seconds
    aml = H2OAutoML(max_runtime_secs = 3600, nfolds=nfolds, seed=seed, exclude_algos=exclude_algos, max_models=max_models)
    aml.train(x = x, y = y,
              training_frame = train)

    # View the AutoML Leaderboard
    lb = aml.leaderboard

    print(lb)
    
    print_importances_best(lb.as_data_frame().model_id[0], target)

    # releases all objects
    
    h2o.remove_all()
    return lb

def print_importances_best(model_id, feature):

    stack_model = h2o.get_model(model_id)
    
    if (model_id.startswith('Stack')):

        metafit = h2o.get_model(stack_model.metalearner()['name'])
        dict_models = metafit.coef_norm()
        print('Stack is composed by:')
        print(dict_models)

        for key, value in dict_models.items():

            if ((value > 0) & (key != 'Intercept')):

                table = 'coefficients_table'
                
                base_learner = h2o.get_model(key)
                print(key)

                if (key.startswith('GBM') | key.startswith('DRF') | key.startswith('XRT')):

                    table = 'variable_importances'
                    df = pd.DataFrame(base_learner._model_json['output'][table].as_data_frame())
                    print(df)
                    
                    
                    
                else:
                    
                    df_coef = base_learner._model_json['output'][table].as_data_frame()
                    df_coef['abs_standardized_coef'] = [abs(x) for x in df_coef['standardized_coefficients']]
                    df = pd.DataFrame(df_coef.sort_values('abs_standardized_coef', ascending=False))
                    print(df)
                    
                df.to_csv('{}-{}.csv'.format(key, feature))
                    
        
    else:
        
        model = h2o.get_model(model_id)
            
        table = 'coefficients_table'

        if (model_id.startswith('GBM') | model_id.startswith('DRF') | model_id.startswith('XRT')):

            table = 'variable_importances'
            df = pd.DataFrame(model._model_json['output'][table].as_data_frame())
            print(df)

        else:

            df_coef = model._model_json['output'][table].as_data_frame()
            df_coef['abs_standardized_coefficients'] = [abs(x) for x in df_coef.standardized_coefficients]
            df = pd.DataFrame(df_coef.sort_values('abs_standardized_coefficients', ascending=False))
            print(df)
            
        df.to_csv('{}-{}.csv'.format(model_id, feature))