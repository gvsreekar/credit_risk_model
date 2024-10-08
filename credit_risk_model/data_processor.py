import pandas as pd 
import logging 
import urllib.request
from credit_risk_model import config
import joblib


def load_data_and_sanitize(file_name:str =config.FILE_NAME)->pd.DataFrame:
    
    if file_name.split('.')[-1]!='csv':
        raise ValueError('File must be a csv file')
    logging.info("Loading data from {file_name}")
    df = pd.read_csv(f"{config.PARENT_ABS_PATH}\data\{file_name}").rename(lambda x: x.lower().strip().replace(' ','_'),axis='columns')
    return df
    


    """
    This is a class to download data and load it into pandas DataFrame with some sanity operations
    """
    def __init__(self,file_path : str = 'data/raw',url:str = None, output_path : str = 'data/processed/'):
        if(url is None and file_path == 'data/raw') or (url is not None and file_path!='data/raw'):
            raise ValueError('Either url or file_path must/only be specified')
        
        self.file_path = f"{file_path}"+f"/{url.split('/')[-1]}" if url is not None else file_path # save non default user specified path
        self.url = url
        self.output_path = output_path
        print(self.file_path)
        
    def download_data(self)-> None:
        logging.info(f" Downloading data from {self.url}")
        urllib.request.urlretrieve(self.url, self.file_path)
            
    def load_data(self) -> pd.DataFrame:
        return pd.read_csv(self.file_path)
    
    def save_data(self,df:pd.DataFrame,file_name : str) -> None:
        logging.info(f" Saving the data to {self.output_path}")
        df.to_csv(self.output_path+file_name,index=False)
    
    def sanitize(self,df:pd.DataFrame)-> pd.DataFrame:
        """
        Rename columns to snake case and strip whitespace
        """
        return df.rename(lambda x: x.lower().strip().replace(' ','_'),axis='columns')
        
    
        
            