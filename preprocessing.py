import pandas as pd
import numpy as np

def DataPreprocessing(train_path,test_path):
	#extracting data in pandas dataframe
	train_df=pd.read_csv(train_path)
	test_df=pd.read_csv(test_path)

	varieties = train_df.columns[13:]

	#Concatenate test and train data, hence introducing column variety in test data
	test['variety']="invalid"
	complete_df=pd.concat((train_df,test_df),axis=0)

	#FILLING THE MISSING VALUES
	#Country column has 39 missing values and since winery-country pair is unique, the missing values can be filled using the winery 
	#in the correspoding column

	complete_df.loc[9037, 'country']='Canada'
	complete_df.loc[14395,'country']='Austria'
	complete_df.loc[15896,'country']='New Zealand'
	complete_df.loc[23458,'country']='New Zealand'
	complete_df.loc[30076,'country']='South Africa'
	complete_df.loc[32137,'country']='Israel'
	complete_df.loc[49765,'country']='France'
	complete_df.loc[35098,'country']='Chile'
	complete_df.loc[35725,'country']='South Africa'
	complete_df.loc[47204,'country']='Israel'
	complete_df.loc[53566,'country']='Chile'
    complete_df.loc[64683,'country']='France'
	complete_df.loc[65423,'country']='Chile'
	complete_df.loc[67040,'country']='Greece'
	complete_df.loc[70050,'country']='Israel'
	complete_df.loc[73049,'country']='Austria'
	complete_df.loc[13171,'country']='Israel'
	# complete_df['country'] = complete_df.apply(lambda x: complete_df[complete_df['winery']==x['winery']].country.value_counts().index[0] if (pd.isnull(x['country'] and len(complete_df[complete_df['winery']==x['winery']].country.value_counts().index.tolist())>0)) else x['country'])

	complete_df.country.fillna("USA",inplace=True)

	#province coulmn has 39 missing values, we can use the winery column to find the province in which it is located
	complete_df.loc[9037,'province']="Ontario"
	complete_df.loc[14395,'province']='Südoststeiermark'
	complete_df.loc[15896,'province']="Canterbury"
	complete_df.loc[23458,'province']='Canterbury'
	complete_df.loc[30076,'province']='Hemel en Aarde'
	complete_df.loc[32137,'province']='Judean Hills'
	complete_df.loc[35098,'province']='Maipo Valley'
	complete_df.loc[35725,'province']='Hemel en Aarde'
	complete_df.loc[47204,'province']='Judean Hills'
	complete_df.loc[53566,'province']='Maipo Valley'
	complete_df.loc[49765,'province']='Bordeaux'
	complete_df.loc[64683,'province']='Bordeaux'
	complete_df.loc[70050,'province']='Judean Hills'
	complete_df.loc[13171,'province']='Judean Hills'
	complete_df.loc[65423,'province']='Maule Valley'
	complete_df.loc[73049,'province']='Südoststeiermark'
	complete_df.loc[67040,'province']='Vin de Pays de Velvendo'

	complete_df.province.fillna("Bordeaux",inplace=True)

	#region_1 column has many missing values. Through analyzing the review_titles we can infer the regions are usually mentioned 
	#brackets of review titles, hence missing regions can be extracted from there
	complete_df["expected_region"] = complete_df.apply(lambda x: x['review_title'].split("(")[1] if len(x["review_title"].split("("))>1 else "XXXX", axis=1)
	complete_df["expected_region"]=complete_df.apply(lambda x: x['expected_region'].split(")")[0], axis=1)

	complete_df['region_1']=complete_df.apply(lambda x: x['expected_region'] if pd.isnull(x['region_1']) else x['region_1'],axis=1)

	complete_df['region_1'] = complete_df.apply(lambda x: complete_df[complete_df.province == x['province']].region_1.value_counts().index.tolist()[0] if x['region_1']=='XXXX' else x['region_1'],axis=1)
	complete_df['region_1'] = complete_df.apply(lambda x: x['province'] if x['region_1']=='XXXX' else x['region_1'],axis=1)

	#price coulmn shows a strong corelation with region_1 column
	complete_df['price']=complete_df.apply(lambda x: complete_df[(complete_df.price.notnull()) & (complete_df.region_1==x['region_1'])].price.median() if np.isnan(x['price']) else x['price'], axis=1)
	complete_df['price']=complete_df.apply(lambda x: complete_df[(complete_df.price.notnull()) & (complete_df.province==x['province'])].price.median() if np.isnan(x['price']) else x['price'], axis=1)
	complete_df['price']=complete_df.apply(lambda x: complete_df[(complete_df.price.notnull()) & (complete_df.country==x['country'])].price.median() if np.isnan(x['price']) else x['price'], axis=1)


	return complete_df,test_df,varieties


