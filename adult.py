from pandas import Series, concat, read_csv, isna, DataFrame, get_dummies
from sklearn import preprocessing
from sklearn.base import accuracy_score
from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import OneHotEncoder


def read_csv_1(data_file):
	dataFrame = read_csv(data_file, encoding='unicode_escape')
	dataFrame.drop('fnlwgt', axis=1, inplace=True)
	return dataFrame

# Return the number of rows in the pandas dataframe df.
def num_rows(df):
	return df[df.columns[0]].count()

# Return a list with the column names in the pandas dataframe df.
def column_names(df):
	return df.columns.to_list()

# Return the number of missing values in the pandas dataframe df.
def missing_values(df: DataFrame):
	count = 0
	nadf = df.isna()
	for column in nadf:
		count += list(nadf.get(column)).count(True)
	return count


# Return a list with the columns names containing at least one missing value in the pandas dataframe df.
def columns_with_missing_values(df):
	columns = []
	nadf = df.isna()
	for column in nadf:
		if True in list(nadf.get(column)):
			columns.append(column)
	return columns


# Returns the percentage of instances corresponding to persons whose education level is Bachelors or Masters
def bachelors_masters_percentage(df):
	# return round(list(df.get("education")).count("Bachelors") + list(df.get("education")).count("Masters")) / len(df.get("education")) * 100, 1)
	return round((list(df.get("education")).count("Bachelors") + list(df.get("education")).count("Masters")) / len(df.get("education")) * 100, 1)

# Return a pandas dataframe
def data_frame_without_missing_values(df: DataFrame):
	return df.dropna()

# Return a pandas dataframe
def one_hot_encoding(df):
	new_dataframe = df.copy()
	new_dataframe.drop('class', axis=1, inplace=True)
	columns = new_dataframe.select_dtypes(include=['object']).columns

	# encoder = preprocessing.OneHotEncoder(sparse_output=False)

	# one_hot_encoded = encoder.fit_transform(new_dataframe[columns])
	df_pandas_encoded = get_dummies(new_dataframe, columns=columns, drop_first=True)
	# one_hot_df = DataFrame(one_hot_encoded, columns=encoder.get_feature_names_out(columns))
	# df_encoded = concat([new_dataframe, one_hot_df], axis=1)

	# df_encoded = df_encoded.drop(columns, axis=1)
	return df_pandas_encoded


def label_encoding(df):
	new_dataframe = df.copy()
	label_encoder = preprocessing.LabelEncoder()
	new_dataframe['class']= label_encoder.fit_transform(new_dataframe['class'])
	return new_dataframe


def dt_predict(X_train,y_train):
	dtree = DecisionTreeClassifier()
	dtree = dtree.fit(X_train, y_train)
	prediction = dtree.predict(X_train)
	return Series(prediction)

# Evaluation
def dt_error_rate(y_pred, y_true):
	error_rate = 1 - accuracy_score(y_true, y_pred)
	return error_rate




