import pandas as pd 
import logging 
import urllib.request
from credit_risk_model import config
import dill


def load_data_and_sanitize(file_name:str =config.FILE_NAME)->pd.DataFrame:
    
    if file_name.split('.')[-1]!='csv':
        raise ValueError('File must be a csv file')
    logging.info("Loading data from {file_name}")
    df = pd.read_csv(f"{config.PARENT_ABS_PATH}\data\{file_name}").rename(lambda x: x.lower().strip().replace(' ','_'),axis='columns')
    return df
    
def save_data(df : pd.DataFrame,file_name : str) -> None:
    """
    Saves a pandas DataFrame to a local csv file.
    
    Args:
        df (pd.DataFrame): The DataFrame to save.
        file_name (str): The name of the local csv file to save the DataFrame to.
    """
    logging.info('Saving data to %s', file_name)
    df.to_csv(f'{config.PARENT_ABS_PATH}/data/{file_name}', index=False)
    
def save_pipeline(pipeline, pipe_name: str) -> None:
    """
    Saves a pipeline to a pickle file.
    
    Args:
        pipeline (Pipeline): The pipeline to save.
    """
    logging.info('Saving pipeline to trained_models folder')
    with open(f'{config.PARENT_ABS_PATH}/credit_risk_model/trained_models/{pipe_name}.pkl', 'wb') as f:
        dill.dump(pipeline, f)
    print(f'Saved pipeline to trained_models/{pipe_name}.pkl')

def load_pipeline(pipe_name: str):
    """
    Loads a saved pipeline from a pickle file.

    Args:
        pipe_name (str): The name of the pipeline to load.

    Returns:
        Pipeline: The loaded pipeline.
    """
    with open(f'{config.PARENT_ABS_PATH}/credit_risk_model/trained_models/{pipe_name}.pkl', 'rb') as f:
        pipe = dill.load(f)
    return pipe
    
        
            