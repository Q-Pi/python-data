def model_building():
	import pandas as pd
	import seaborn as sns
	import matplotlib.pyplot as plt
	from sklearn.ensemble import RandomForestClassifier as RF
	from sklearn.model_selection import train_test_split as TTS
	from sklearn.pipeline import Pipeline
	from sklearn.model_selection import GridSearchCV
	from sklearn.decomposition import PCA

	#load data
	df = sns.load_dataset('iris')

	#isolation
	target_feature_name = 'species'
	Y = df[target_feature_name]
	X = df.drop(columns=target_feature_name)

	#labels
	labels = Y.astype('category').cat.categories.tolist()

	#tts
	X_tr, X_te, Y_tr, Y_te = TTS(X, Y, stratify=Y, random_state=42)

	#pipeline
	pipeline_details = [('PCA', PCA(random_state=20)),
	                    ('RF', RF())]
	pipeline = Pipeline(steps=pipeline_details)

	#hyperparameters
	hyperparameters = {}
	hyperparameters['PCA__n_components'] = [i for i in range(1, X_tr.shape[1]+1)]
	hyperparameters['RF__n_estimators']  = [i for i in range(10, 250 +1, 10)]

	#gridsearch
	hyperparameter_search = GridSearchCV(pipeline,
	                                     hyperparameters,
	                                     scoring='accuracy', 
	                                     cv=5)
	hyperparameter_search.fit(X_tr, Y_tr)
	#print('Meilleur score : {:.5f}'.format(hyperparameter_search.best_score_))
	#print('Meilleur paramètres : {}'.format(hyperparameter_search.best_params_))

	#pca
	N=hyperparameter_search.best_params_['PCA__n_components']
	pca = PCA(n_components=N, random_state=20)
	pca.fit(X_tr);

	X_tr_PCA = pca.transform(X_tr)
	X_te_PCA = pca.transform(X_te)

	#RandomForestClassifier
	N=hyperparameter_search.best_params_['RF__n_estimators']
	rf = RF(n_estimators=N)
	rf.fit(X_tr_PCA, Y_tr);

	#train check
	train_preds = rf.predict(X_tr_PCA)
	"""
	accuracy = lambda p, y : (p==y).sum()/len(y)
	print('Accuracy : {}'.format(accuracy(train_preds, Y_tr)))
	"""

	#validation
	df = pd.DataFrame(X_te_PCA)

	preds = rf.predict(df)
	proba = rf.predict_proba(df)

	df = pd.DataFrame(X_te)
	"""
	df['Predictions'] = preds
	for i in range(0, len(proba[0])):
	    df[labels[i]] = proba[:, i]
	df
	"""

	#Confusion Matrix
	"""
	from sklearn.metrics import confusion_matrix
	def show_cm(cm, labels):
	    df_cm = pd.DataFrame(cm, labels, labels)
	    sns.heatmap(df_cm, annot=True)
	    plt.show()

	cm_train = confusion_matrix(train_preds, Y_tr, labels=labels)
	show_cm(cm_train, labels)accuracy = lambda p, y : (p==y).sum()/len(y)
	print('Accuracy : {}'.format(accuracy(train_preds, Y_tr)))

	cm_test = confusion_matrix(preds, Y_te, labels=labels)
	show_cm(cm_test, labels)accuracy = lambda p, y : (p==y).sum()/len(y)
	print('Accuracy : {}'.format(accuracy(preds, Y_te)))
	"""

	#Saves
	from joblib import dump
	dump(rf, "utils/joblibs/model.joblib")

	dump(pca, "utils/joblibs/pca.joblib")

	dump(X_tr.columns.to_list(), "utils/joblibs/features.joblib")

	dump(X_tr.dtypes, "utils/joblibs/features_type.joblib")

	dump(labels, "utils/joblibs/labels.joblib")


def model_predict_proba(data):
	import pandas as pd
	from joblib import load
	from random import randrange

	#from model building
	model = load("ML/utils/joblibs/model.joblib")
	pca = load("ML/utils/joblibs/pca.joblib")
	features = load("ML/utils/joblibs/features.joblib")
	labels = load("ML/utils/joblibs/labels.joblib")

	#data: list -> DataFrame
	d = {}
	for i in range(0, len(features)):
		d[features[i]] = [data[i]]
	df = pd.DataFrame(data=d, columns=features)
	
	#pca & model
	X_PCA = pca.transform(df)
	proba = model.predict_proba(X_PCA)

	#return: list -> Json
	return proba[0]