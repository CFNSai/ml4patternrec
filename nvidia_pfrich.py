'''
This is a sample to read the NVIDIA pfRICH
Optical simulation
'''

import gradientBoost_hybrid_model as gbhm
import pandas as pd
import numpy as np
import requests
import vector
from io import StringIO
import re

#Get Optix file from git
raw_git_url='https://raw.githubusercontent.com/BNLNPPS/esi-fastlight/320a08477ba12a52450d30a5c15c20faaf24e769/opticks_hits_output.txt'

#Read in NVIDIA Optix file as dataframe
try:
    response=requests.get(raw_git_url)
    # Raise an exception for bad status codes
    response.raise_for_status()  
    content=response.text
except requests.exceptions.RequestException as e:
    print(f"Error fetching data from GitHub: {e}")
    exit()

#Remove parenthenses from content
cleaned_content=re.sub(r'\(|\)|CreationProcessID\=','',content)
#Use StringIO to treat the string content as a file
file_content=StringIO(cleaned_content)
#Use `sep=r'\s+'` to handle varying whitespace delimiters
optix_df=pd.read_csv(file_content,sep=r'\s*[, ]\s*',header=None,engine='python')

#Set column headers
optix_df.columns=['time','wavelength','x','y','z','nx','ny','nz','polx','poly','polz','CreationProcessID']

#Calculate r
r=np.sqrt(optix_df['x']**2+optix_df['y']+optix_df['z']**2)
theta=np.arccos(optix_df['z']/r.replace(0,np.nan))

optix_df['theta']=theta
print(optix_df.head())
