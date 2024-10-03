import pandas as pd 
import logging 

class DataHandler:
    
    """
    This is a class to download data and load it into pandas DataFrame with some sanity operations
    """
    def __init__(self,file_path : str = 'data/raw',url:str = None, output_path : str = 'data/processed/'):
        if(url is None and file_path == 'data/raw') or (url is not None and file_path!='data/raw'):
            raise ValueError('Either url or file_path must/only be specified')
        
        if url is not None:
            self.file_path = f"{file_path}"+f"/{url.split('/')[-1]}"
            self.url = url
        self.output_path = output_path
            
            