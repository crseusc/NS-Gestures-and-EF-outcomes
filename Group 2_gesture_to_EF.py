# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 18:41:31 2021

@author: Dani Kiyasseh (California Institute of Technology)
"""

from operator import itemgetter
#import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
import os

os.chdir('C:/Users/DaniK/OneDrive/Desktop')
rootpath = 'C:/Users/DaniK/OneDrive/Desktop/General/Caltech/Videos + Scripts'

#%%
glist = ['a', 'b', 'c', 'camera fell out for this time',
       'camera was taken out for a while', 'd', 'e', 'g', 'h', 'i', 'k',
       'l', 'm', 'nan', 'o', 'other', 'p', 'q', 'r', 's', 't', 'w']

glabels = ['assistant','burn dissect','cold cut','camera fell','camera out','pedicalize','hot cut','coagulate','hook','idle','clip','unknown','camera move',
		   'nan','coagulate then cut','other','peel','q','retraction','spread','2-hand spread','ineffective motion']

gmap = dict(zip(glist,glabels))

#%%
#import seaborn as sns
#import matplotlib.pyplot as plt
#def raster_plot(gdf):
#	pdf = gdf[gdf['Gesture'] == 'P-062_NVB_R']
#	gdf['Gesture'] = enc.transform(gdf['Gesture'])
#	fig,ax = plt.subplots(figsize=(12,8))
#	sns.scatterplot(x='CumTime',y='Gesture',data=pdf,ax=ax)
#	ax.set_yticks(range(len(enc.classes_)))
#	ax.set_yticklabels(glabels)
#%%
def loadUSCData():
	gdf = pd.read_csv(os.path.join(rootpath,'NS/NS_gestures_timestamps.csv'),index_col=0)
	gdf['Gesture'] = gdf['Gesture'].apply(lambda g:str(g).strip(' ')) # remove random spaces before or after gesture
	#gdf['Gesture'] = gdf['Gesture'].replace({' r':'r'})
	gdf['Video'] = gdf['Path'].apply(lambda path:path.split('\\')[-1])
	enc = LabelEncoder()
	enc.fit(gdf['Gesture'])
	
	ddf = pd.read_csv(os.path.join(rootpath,'Surgical Meta Information/USC/NS_clinical_data.csv'),index_col=0)
	ddf['PID'] = ddf.index
	ddf['ESI_12M'] = ddf['ESI @ 12 mo']
	pids = ddf['PID'].unique().tolist()
	gdf = gdf[gdf['PID'].isin(pids)]	
	return ddf, gdf, enc

def loadGronauData():
	gdf = pd.read_csv(os.path.join(rootpath,'NS/NS_Gronau_gestures_timestamps.csv'),index_col=0)
	gdf['Gesture'] = gdf['Gesture'].apply(lambda g:str(g).strip(' ')) # remove random spaces before or after gesture
	#gdf['Gesture'] = gdf['Gesture'].replace({' r':'r'})
	gdf['Video'] = gdf['Path'].apply(lambda path:path.split('\\')[-1])
	enc = LabelEncoder()
	enc.fit(gdf['Gesture'])
	
	ddf = pd.read_csv(os.path.join(rootpath,'Surgical Meta Information/Gronau/Clinical Data Gronau 30 cases (updated).csv'),index_col=0)
	ddf['PID'] = ddf.index
	ddf['ESI_12M'] = ddf['ESI_12m']
	pids = ddf['PID'].unique().tolist()
	gdf = gdf[gdf['PID'].isin(pids)]	
	return ddf, gdf, enc

def loadDFs(dataset='USC'):
	if dataset == 'USC':
		ddf, gdf, enc = loadUSCData()
	elif dataset == 'Gronau':
		ddf, gdf, enc = loadGronauData()	
	elif dataset == 'USC+Gronau':
		ddfA, gdfA, enc = loadUSCData()
		ddfB, gdfB, enc = loadGronauData()
		ddf = pd.concat((ddfA,ddfB),0)
		gdf = pd.concat((gdfA,gdfB),0)
		enc = LabelEncoder()
		enc.fit(gdf['Gesture'])
	
	return ddf, gdf, enc

#%%
def getGestDict(gdf,seq_len):
	colname = 'Video' #options: PID | Video
	seq_len = seq_len
	gest_dict = dict()
	for video in tqdm(gdf[colname].unique().tolist()):
		videodf = gdf[gdf[colname] == video]
		gest = videodf['Gesture'].to_numpy()
		ngest = len(gest)
		nchunks = ngest // seq_len
		assert nchunks > 0
		gest_list = []
		for i in range(nchunks):
			start = i*seq_len
			end = start + seq_len
			curr_gest = enc.transform(gest[start:end])
			gest_list.append(curr_gest)
		gest_dict[video] = np.array(gest_list)
	return gest_dict
	
#%%
def getLabelsDict(ddf,gest_dict):
	labels_dict = dict()
	for video,vals in tqdm(gest_dict.items()):
		y = ddf[ddf['PID'].str.contains(video.split('_')[0])]['ESI_12M'].iloc[0]
		#y = ddf[ddf['PID']==video]['ESI_12M'].iloc[0]
		labels_dict[video] = int(y)
	return labels_dict
	
def getCDF(ddf):
	cfeatures = ['Age','BMI','PSA','Nerve Sparing','Pre-op Gleason','ASA','Prostate volume (g)','Post-op Gleason','ECE','Radiation after surgery 1=Yes, 0=No','Postop ADT']
	cdf = ddf[cfeatures]
	for feature in cfeatures:
		if feature == 'Nerve Sparing':
			NSenc = LabelEncoder()
			cdf[feature] = NSenc.fit_transform(cdf[feature])
	return cdf, cfeatures
	
#%%
""" Combine DataFrames """
def getDF(ddf,gdf,seq_len):
	if include_gestures == True:
		gest_dict  = getGestDict(gdf,seq_len)
		labels_dict = getLabelsDict(ddf,gest_dict)
		gest_df = pd.DataFrame()
		for video,array in tqdm(gest_dict.items()):
			pid = video.split('_')[0]
			pids = [pid]*len(array)
			labels = [labels_dict[video]]*len(array)
			curr_df = pd.DataFrame(array)
		
			if include_clinical_features == True:	
				cdf, cfeatures = getCDF(ddf)
				curr_cdf = cdf.loc[pid,:].tolist()
				curr_cdf = pd.DataFrame([curr_cdf for _ in range(len(array))])
				curr_cdf.columns = cfeatures
				curr_df = pd.concat((curr_df,curr_cdf),1)
			
			""" Add Annotations and Meta Information """
			curr_df['label'] = labels
			curr_df['video'] = [video]*len(array)
			curr_df['pid'] = pids
				
			gest_df = pd.concat((gest_df,curr_df),0)
		df = gest_df.copy()
	elif include_clinical_features == True:
		cdf, cfeatures = getCDF(ddf)
		cdf['label'] = [ddf.loc[pid,'ESI_12M'] for pid in cdf.index]
		cdf['video'] = cdf.index #clinical features are not video dependent
		cdf['pid'] = cdf.index
		df = cdf.copy()
	
	return df
	
#%%
def getNFeatures(ddf,seq_len):
	if include_gestures == True:
		if include_clinical_features == True:
			_, cfeatures = getCDF(ddf)
			nfeatures = seq_len + len(cfeatures)
		else:
			nfeatures = seq_len
	elif include_clinical_features == True:
		_, cfeatures = getCDF(ddf)
		nfeatures = len(cfeatures)
	return nfeatures

def shuffleDF(gest_df_scaled,seed):
	gest_df_scaled = gest_df_scaled.sample(n=gest_df_scaled.shape[0],replace=False,random_state=seed)
	return gest_df_scaled
	
def scaleDF(gest_df,nfeatures):
	mins = gest_df.iloc[:,range(0,nfeatures)].min(axis=1)
	maxs = gest_df.iloc[:,range(0,nfeatures)].max(axis=1)
	mins.index,maxs.index = gest_df.index,gest_df.index
	gest_dfA = pd.DataFrame((gest_df.iloc[:,range(0,nfeatures)] - np.expand_dims(mins,1).repeat(nfeatures,axis=1)) / np.expand_dims(maxs - mins + 1e-8,1).repeat(nfeatures,axis=1))
	gest_df = pd.concat((gest_dfA,gest_df[['label','video','pid']]),1)
	return gest_df


#%%
def performTraining(df,nfeatures,include_gestures,include_clinical_features):
	seeds = 12 # 25 for USC data
	columns = ['Seed','Fold','AUC Train','AP Train','AUC Val','AP Val','AUC Test','AP Test']
	metrics = pd.DataFrame(columns=columns)
	for seed in range(seeds):
		#model = RandomForestClassifier()
		model = LogisticRegression(max_iter=500,random_state=seed)
		#model = SVC(kernel='rbf',probability=True)
		
		if include_gestures == True:
			if include_clinical_features == True:
				df_scaled = df.copy() #scaleDF(df,nfeatures) #do not scale
			else:
				df_scaled = scaleDF(df,nfeatures) #scale
		elif include_clinical_features == True:
			df_scaled = scaleDF(df,nfeatures) #scale
			
		df_scaled = shuffleDF(df_scaled,seed)
		#nrows = df_scaled.shape[0]
		nfolds = 8 # 4 for USC data
		videos = df_scaled['pid'].unique().tolist()
		
	#	test_videos = ['P-109',
	#					 'P-129',
	#					 'P-142',
	#					 'P-165',
	#					 'P-176',
	#					 'P-203',
	#					 'P-276',
	#					 'P-277',
	#					 'P-339',
	#					 'P-404']
	#	videos = list(set(videos) - set(test_videos))
		nvideos = len(videos)
		nvideos_per_fold = nvideos // nfolds
		
		#nsamples = gest_df.shape[0] // nfolds
		for k in tqdm(range(nfolds)):
			start = k*nvideos_per_fold
			end = start + nvideos_per_fold
			val_indices = list(range(start,end))
			if k == nfolds-1:
				test_indices = list(range(0,nvideos_per_fold))
			else:
				test_indices = list(range(end,end+nvideos_per_fold))
			train_indices = list(set(list(range(nvideos))) - set(val_indices).union(set(test_indices)))
	
			test_videos = list(itemgetter(*test_indices)(videos))
			val_videos = list(itemgetter(*val_indices)(videos))
			train_videos = list(itemgetter(*train_indices)(videos))
			
			test_fold = df_scaled[df_scaled['pid'].isin(test_videos)]
			val_fold = df_scaled[df_scaled['pid'].isin(val_videos)]
			train_folds = df_scaled[df_scaled['pid'].isin(train_videos)]
		
			xtrain,xval,xtest = train_folds.iloc[:,:nfeatures], val_fold.iloc[:,:nfeatures], test_fold.iloc[:,:nfeatures]
			ytrain,yval,ytest = train_folds.loc[:,'label'], val_fold.loc[:,'label'], test_fold.loc[:,'label']
			
			#xtrain = scaler.fit_transform(xtrain)
			#xval = scaler.transform(xval)
			#xtest = scaler.transform(xtest)
			
			model.fit(xtrain,ytrain)
			
			def collapse_probs(x,y,folds):
				probs = model.predict_proba(x)
				out_df = pd.DataFrame([probs[:,1],y]).T
				out_df.index = folds.index
				out_df[['video','pid']] = folds.loc[:,['video','pid']]
				out_df.columns = ['prob','label','video','pid']
				out_df['prob'] = out_df['prob'].astype(float)
				out_df['label'] = out_df['label'].astype(int)
				out_df = out_df.groupby(by=['pid']).mean()
				return out_df
			
			phases = ['train','val','test']
			xs = [xtrain,xval,xtest]
			ys = [ytrain,yval,ytest]
			folds = [train_folds,val_fold,test_fold]
			curr_metrics = [seed,k]
			for phase,x,y,fold in zip(phases,xs,ys,folds):
				out_df = collapse_probs(x,y,fold)
				auc = roc_auc_score(out_df['label'],out_df['prob'])
				ap = average_precision_score(out_df['label'],out_df['prob'])
				curr_metrics.extend([auc,ap])
				#out_df = collapse_probs(xval,yval,val_fold)
				#auc_val = roc_auc_score(out_df['label'],out_df['prob'])
				
				#""" Insert Evaluation on Held-Out Set """
				#out_df = collapse_probs(xtest,ytest,test_fold)
				#auc_test = roc_auc_score(out_df['label'],out_df['prob'])
				#""" End """
				
			curr_metrics = pd.DataFrame(curr_metrics).T #pd.DataFrame([seed,k,auc_train,ap_train,auc_val,ap_val,auc_test,ap_test]).T
			curr_metrics.columns = columns
			metrics = pd.concat((metrics,curr_metrics),0)
		
		metrics['AUC Train'] = metrics['AUC Train'].apply(lambda val:1-val if val < 0.5 else val)
		metrics['AUC Val'] = metrics['AUC Val'].apply(lambda val:1-val if val < 0.5 else val)
		metrics['AUC Test'] = metrics['AUC Test'].apply(lambda val:1-val if val < 0.5 else val)
		#print(metrics.mean())
	return metrics

def getSetting(include_gestures,include_clinical_features):
	if include_gestures == True:
		if include_clinical_features == True:
			setting = 'Clinical + Gestures'
		else:
			setting = 'Gestures'
	elif include_clinical_features == True:
		setting = 'Clinical'
	return setting
	
#%%
dataset = 'USC+Gronau' #options: USC | Gronau | USC+Gronau
metricsdf = pd.DataFrame()
for include_gestures in [False,True]:
	for include_clinical_features in [False,True]:
		if include_gestures == False and include_clinical_features == False:
			continue
		else:
			setting = getSetting(include_gestures,include_clinical_features)
			seq_len = 20
			ddf, gdf, enc = loadDFs(dataset=dataset)
			gest_dict = getGestDict(gdf,seq_len)
			labels_dict = getLabelsDict(ddf,gest_dict)
			df = getDF(ddf,gdf,seq_len)
			nfeatures = getNFeatures(ddf,seq_len)
			metrics = performTraining(df,nfeatures,include_gestures,include_clinical_features)
			metrics['setting'] = setting
			metricsdf = pd.concat((metricsdf,metrics),0)

#%%
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=2)
plt.style.use('seaborn-darkgrid')
fig,ax = plt.subplots(figsize=(12,8))
#metricsdf['setting'] = metricsdf['setting'].map({'Clinical':'Clinical Features','Gestures':'Gesture Sequences','Clinical + Gestures':'Combined'})
sns.violinplot(x='setting',y='AUC Test',data=metricsdf,palette='Set2',bw=0.2,ax=ax)
ax.set_xlabel('')
ax.set_ylabel('AUC')
ax.set_ylim([0,1.1])
#sns.violinplot(x='setting',y='AP Test',data=metricsdf,palette='Set2',ax=axes[1])
#axes[1].set_xlabel('')
