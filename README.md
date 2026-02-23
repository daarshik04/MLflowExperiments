# This is a MLflow experimentation project using the wine quality check data
import dagshub
dagshub.init(repo_owner='daarshik04', repo_name='MLflowExperiments', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)