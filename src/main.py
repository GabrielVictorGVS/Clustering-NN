import argparse
import sys
import os
import pandas
import seaborn
import numpy
import tensorflow

from typing import Any

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, confusion_matrix, ConfusionMatrixDisplay

from matplotlib import pyplot

# Defines the global constant for the default algorithm dataset
DEFAULT_DATASET = 'iris'

# Defines the global constant for the default training dataset percentage
DEFAULT_TRAINING_PERCENTAGE = 70

# Defines the global constant for the cleveland dataset default path
DEFAULT_CLEVELAND_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'raw', 'cleveland.data')

# Defines the global constant for the algorithm selection flag
DEFAULT_IS_SUPERVISED_ALGORITHM = False

# Defines the global constant for the supervised prediction model learning rate
DEFAULT_NN_MODEL_LEARNING_RATE = 0.0005

# Defines the global constant for the supervised prediction model training epochs amount
DEFAULT_NN_MODEL_TRAINING_EPOCHS_AMOUNT = 1500

# Defines the global constant for the default k-cluster amount
DEFAULT_K_CLUSTER_AMOUNT = 3

# Defines the global constant for the default k-means maximum iterations
DEFAULT_MAX_ITERATIONS = 500

# Defines the global constant for the default random state
DEFAULT_RANDOM_STATE = None

# Creates and adds the necessary arguments to the parser
def preconfigure_parser() -> argparse.ArgumentParser:

	# Create the Argument Parser
	parser = argparse.ArgumentParser(description='Simple Clustering for the Iris Flower and the Cleveland Heart Disease Datasets')

	# Add the positional argument for the algorithm dataset
	parser.add_argument(
		'--dataset',
		nargs='?',
		default=DEFAULT_DATASET,
		choices=[
			'iris',
			'cleveland'
		],
		help=f'Choose the algorithm dataset (default: {DEFAULT_DATASET})'
	)

	# Lambda function for validating the tranining percentage input range
	def training_range_lambda(x):
		return int(x) if 0 <= int(x) <= 90 else (_ for _ in ()).throw(argparse.ArgumentTypeError(f'Percentage value must be between 0 and 70, but got {x}%%.'))

	# Lambda function for validating positive interger inputs
	def positive_interger_lambda(x):
		return int(x) if 0 <= int(x) else (_ for _ in ()).throw(argparse.ArgumentTypeError(f'Input value must be a positive interger, but got {x}.'))

	# Add the positional argument for the dataset training percentage
	parser.add_argument(
		'--training-percentage',
		type=training_range_lambda,
		nargs= '?',
		default=DEFAULT_TRAINING_PERCENTAGE,
		help=f'Dataset training percentage between 0%% and 90%% (default: {DEFAULT_TRAINING_PERCENTAGE}%%)'
	)

	# Add the positional argument for the cleveland dataset file path
	parser.add_argument(
		'--cleveland-path',
		type=str,
		nargs='?',
		default=DEFAULT_CLEVELAND_PATH,
		help=f'File path to the Cleveland Dataset comma-separated CSV file (default: {DEFAULT_CLEVELAND_PATH})'
	)

	# Add the positional argument for the algorithm selection flag
	parser.add_argument(
		'--supervised',
		action='store_true',
		help=f'Algorithm selection flag, either supervised with keras or non-supervised with k-means (default: {DEFAULT_IS_SUPERVISED_ALGORITHM})'
	)

	# Add the positional argument for the supervised model epochs amount
	parser.add_argument(
		'--epochs-amount',
		type=positive_interger_lambda,
		nargs='?',
		default=DEFAULT_NN_MODEL_TRAINING_EPOCHS_AMOUNT,
		help=f'Amount supervised model epochs (default: {DEFAULT_NN_MODEL_TRAINING_EPOCHS_AMOUNT})'
	)

	# Add the positional argument for the supervised model learning rate
	parser.add_argument(
		'--learning-rate',
		type=float,
		nargs='?',
		default=DEFAULT_NN_MODEL_LEARNING_RATE,
		help=f'Supervised model learning rate (default: {DEFAULT_NN_MODEL_LEARNING_RATE})'
	)

	# Add the positional argument for the k-cluster amount
	parser.add_argument(
		'--cluster-amount',
		type=positive_interger_lambda,
		nargs='?',
		default=DEFAULT_K_CLUSTER_AMOUNT,
		help=f'Amount of desired training k-clusters (default: {DEFAULT_K_CLUSTER_AMOUNT})'
	)

	# Add the positional argument for the maximum algorithm iterations
	parser.add_argument(
		'--max-iterations',
		type=positive_interger_lambda,
		nargs='?',
		default=DEFAULT_MAX_ITERATIONS,
		help=f'Amount of maximum K-Means algorithm iterations (default: {DEFAULT_MAX_ITERATIONS})'
	)

	# Add the positional argument for the algorithm random state
	parser.add_argument(
		'--random-state',
		type=positive_interger_lambda,
		nargs='?',
		default=DEFAULT_RANDOM_STATE,
		help=f'K-Means algorithm random state (default: {DEFAULT_RANDOM_STATE})'
	)

	# Returns the sucesfully created and configured parser
	return parser

# Loads and returns the dataset chosen by CLI arguments
def load_dataset(args : argparse.Namespace) -> pandas.DataFrame:

	# Verify the type of the 'args' argument
	if not isinstance(args, argparse.Namespace):

		# Print error message to the screen
		print(f'Error while Loading Dataset: Expected an argparse.Namespace, but got {type(args).__name__}.\nExiting...')

		# Exit with error code
		sys.exit(1)

	# Processing Branch for the Iris Dataset
	if ('iris' == args.dataset):

		# Loads the Iris Dataset
		iris = load_iris(as_frame=True)

		# Converts the Iris Dataset to a Pandas Dataframe
		dataframe = pandas.DataFrame(data=iris.data, columns=iris.feature_names)

		# Appends the labelled targets to the dataframe
		dataframe['species'] = iris.target

		# Returns the Iris Dataset as a Pandas Dataframe
		return dataframe

	# Processing Branch for the Cleveland Hearth Disease Dataset
	elif ('cleveland' == args.dataset):

		# Loads the Cleveland Dataset as a Pandas Dataframe
		dataframe = pandas.read_csv(args.cleveland_path, header = None)

		# Attributes labels to the dataset columns
		dataframe.columns = [
			'age',
			'sex',
			'cp',
			'trestbps',
			'chol',
			'fbs',
			'restecg',
			'thalach',
			'exang',
			'oldpeak',
			'slope',
			'ca',
			'thal',
			'diagnosis'
		]

		# Returns the Cleveland Dataset as a Pandas Dataframe
		return dataframe

	# Processing Branch for a Unknown Dataset
	else:

		# Print error message to the screen
		print('Error while Loading Dataset: Unknown Dataset.\nExiting...')

		# Exit with error code
		sys.exit(1)

# Sanitizes and returns the dataset chosen by CLI arguments
def sanitize_dataset(dataset : pandas.DataFrame, args : argparse.Namespace) -> pandas.DataFrame:

	# Verify the type of the arguments
	if (not isinstance(args, argparse.Namespace)) or (not isinstance(dataset, pandas.DataFrame)):

		# Print error message to the screen
		print('Error while Sanitizing Dataset: Conflicting argument data-types.\nExiting...')

		# Exit with error code
		sys.exit(1)

	# Processing Branch for the Iris Dataset
	if ('iris' == args.dataset):

		# Returns the Sanitized Iris Dataset as a Pandas Dataframe
		return dataset

	# Processing Branch for the Cleveland Hearth Disease Dataset
	elif ('cleveland' == args.dataset):

		# Substitutes all the dataset non-numeric values with NaN
		sanitized_dataset = dataset.apply(pandas.to_numeric, errors='coerce')

		# Removes all NaN values from the dataset in inplace mode
		sanitized_dataset.dropna(inplace = True)

		# Returns the Sanitized Cleveland Dataset as a Pandas Dataframe
		return sanitized_dataset

	# Processing Branch for a Unknown Dataset
	else:

		# Print error message to the screen
		print('Error while Sanitizing Dataset: Unknown Dataset.\nExiting...')

		# Exit with error code
		sys.exit(1)

# Shows the dataset as a pair-plot
def show_dataset(dataset : pandas.DataFrame, args : argparse.Namespace) -> None:

	# Verify the type of the arguments
	if not isinstance(args, argparse.Namespace) or not isinstance(dataset, pandas.DataFrame):

		# Print error message to the screen
		print('Error while Showing Dataset: Conflicting argument data-types.\nExiting...')

		# Exit with error code
		sys.exit(1)

	# Processing Branch for the Iris Dataset
	if ('iris' == args.dataset):

		# Defines target labels for the dataframe
		species = ['setosa', 'versicolor', 'virginica']

		# Creates a Plot Dataset
		plot_dataset = dataset

		# Loads the Iris Dataset
		iris = load_iris(as_frame=True)

		# Appends the labelled targets to the dataframe
		plot_dataset['species'] = iris.target.apply(lambda x: species[x])

		# Creates a pair-plot of the dataset
		seaborn.pairplot(plot_dataset, palette=seaborn.color_palette("magma", n_colors=3), hue='species')

		# Shows the dataset
		pyplot.show()

	# Processing Branch for the Cleveland Hearth Disease Dataset
	elif ('cleveland' == args.dataset):

		# Defines target labels for the dataframe
		diagnosis = {0: 'healthy', 1: 'sickness_1', 2: 'sickness_2', 3: 'sickness_3', 4: 'sickness_4'}

		# Creates a Plot Dataset
		plot_dataset = dataset

		# Applies the labels to the dataframe targets
		plot_dataset['diagnosis'] = dataset['diagnosis'].map(diagnosis)

		# Creates a subset for plotting
		plot_dataset = plot_dataset[['age', 'sex', 'trestbps', 'chol', 'diagnosis']]

		# Creates a pair-plot of the dataset
		seaborn.pairplot(plot_dataset, palette=seaborn.color_palette("magma", n_colors=5), hue='diagnosis')

		# Shows the dataset
		pyplot.show()

	# Processing Branch for a Unknown Dataset
	else:

		# Print error message to the screen
		print('Error while Showing Dataset: Unknown Dataset.\nExiting...')

		# Exit with error code
		sys.exit(1)

# Splits the dataset between data and target, and testing and training
def split_dataset(dataset : pandas.DataFrame, args : argparse.Namespace) -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:

	# Verify the type of the arguments
	if (not isinstance(args, argparse.Namespace)) or (not isinstance(dataset, pandas.DataFrame)):

		# Print error message to the screen
		print('Error while Splitting Dataset: Conflicting argument data-types.\nExiting...')

		# Exit with error code
		sys.exit(1)

	# Processing Branch for the Iris Dataset
	if ('iris' == args.dataset):

		# Extracts the indexes for the training dataset
		training_dataset = dataset.sample(frac=args.training_percentage/100, random_state=10)

		# Extracts the indexes for the testing dataset
		testing_dataset = dataset.drop(training_dataset.index)

		# Returns the splitten datasets
		return training_dataset.iloc[:, 0:4], training_dataset.loc[:, 'species'], testing_dataset.iloc[:, 0:4], testing_dataset.loc[:, 'species']

	# Processing Branch for the Cleveland Hearth Disease Dataset
	elif ('cleveland' == args.dataset):

		# Extracts the indexes for the training dataset
		training_dataset = dataset.sample(frac=args.training_percentage/100, random_state=10)

		# Extracts the indexes for the testing dataset
		testing_dataset = dataset.drop(training_dataset.index)

		# Returns the splitten datasets
		return training_dataset.iloc[:, 0:13], training_dataset.loc[:, 'diagnosis'], testing_dataset.iloc[:, 0:13], testing_dataset.loc[:, 'diagnosis']

	# Processing Branch for a Unknown Dataset
	else:

		# Print error message to the screen
		print('Error while Splitting Dataset: Unknown Dataset.\nExiting...')

		# Exit with error code
		sys.exit(1)

# Plots the model training loss and metrics, and saves them only in 'persistent' or 'inplace' processing mode.
def plot_metrics(history, metrics=None):

	# If no metrics keys are provided, plot all metrics excluding validation ones
	if metrics is None:
		metrics = [key for key in history.history.keys() if not key.startswith('val_')]

	# Iterates through and plots all available model metrics
	for metric in metrics:

		# Defines the metric color palette
		metric_color_palette = seaborn.color_palette("rocket", n_colors=2)

		# Defines the graph figure
		pyplot.figure(figsize=(10, 5))

		# Defines the capitalized and properly spaced metric name
		metric_name = metric.capitalize().replace('_', ' ')

		# Plot the training metric
		pyplot.plot(history.history[metric], label=f'Training {metric_name}', color=metric_color_palette[0])

		# Defines the graph title as the metric name
		pyplot.title(f'Model {metric_name}')

		# Defines the Y-axis label as the metric name
		pyplot.ylabel(f'Model {metric_name}')

		# Defines the X-axis label as 'Epochs'
		pyplot.xlabel('Epochs')

		# Enables the graph legend
		pyplot.legend()

		# Shows the graph figure on the screen
		pyplot.show()

# Benchmarks the supervised model
def benchmark_supervised_model(title : str, training_data : pandas.DataFrame, training_target : pandas.Series, labels : numpy.ndarray, model : Any) -> None:

	columns = None

	if ('iris' == args.dataset):

		columns = ['setosa', 'versicolor', 'virginica']

	elif ('cleveland' == args.dataset):

		columns = ['healthy', 'sickness_1', 'sickness_2', 'sickness_3', 'sickness_4']

	max_indices = numpy.argmax(labels, axis=1)

	max_column_names = numpy.array([columns[idx] for idx in max_indices])

	cm = confusion_matrix(training_target, max_column_names, labels=columns)

	cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)

	cm_display.plot(cmap='magma')

	pyplot.title(f'Matriz de Confusão - {title}')

	pyplot.xticks(ticks=numpy.arange(len(columns)), labels=columns)

	pyplot.yticks(ticks=numpy.arange(len(columns)), labels=columns, rotation= 90)

	pyplot.show()

	pca = PCA(n_components=2)

	pca_components = pca.fit_transform(training_data.values)

	pyplot.figure(figsize=(8, 6))

	scatter = pyplot.scatter(pca_components[:, 0], pca_components[:, 1], c=max_indices, cmap=seaborn.color_palette("flare", as_cmap=True))

	pyplot.title(f'PCA - {title}')

	pyplot.legend(handles=scatter.legend_elements()[0], labels=columns)

	pyplot.show()

# Plots the Elbow Method for K-Cluster amount approximation
def elbow_method(training_data : pandas.DataFrame, max_k_amount = 10) -> None:

	inertias = []

	for k in range(1, max_k_amount + 1):

		kmeans = KMeans(n_clusters=k, random_state=42)

		kmeans.fit(training_data)

		inertias.append(kmeans.inertia_)

	pyplot.figure(figsize=(8, 6))

	pyplot.plot(range(1, max_k_amount + 1), inertias, marker='o', linestyle='--')

	pyplot.title('Método de Cotovelo (Elbow)')

	pyplot.xlabel('Número de Clusters (k)')

	pyplot.ylabel('Inércia')

	pyplot.legend(['Inércia'])

	pyplot.show()

# Benchmarks the k-means model
def benchmark_kmeans(title : str, training_data : pandas.DataFrame, training_target : pandas.Series, labels : numpy.ndarray, model : KMeans) -> None:

	silhouette = silhouette_score(training_data, labels)

	homogeneity = homogeneity_score(training_target, labels)

	completeness = completeness_score(training_target, labels)

	metrics_text = f'Índice de Silhueta: {silhouette:.2f}\n'

	metrics_text += f'Inércia: {model.inertia_:.2f}\n'

	metrics_text += f'Homogeneidade: {homogeneity:.2f}\n'

	metrics_text += f'Complitude: {completeness:.2f}'

	pca = PCA(n_components=2)

	pca_components = pca.fit_transform(training_data.values)

	pyplot.figure(figsize=(8, 6))

	scatter = pyplot.scatter(pca_components[:, 0], pca_components[:, 1], c=labels, cmap=seaborn.color_palette("flare", as_cmap=True))

	pyplot.text(0.29, 0.97, metrics_text, ha='right', va='top', transform=pyplot.gca().transAxes, fontsize=10, color='black', bbox=dict(facecolor='white', alpha=0.5))

	pyplot.title(title)

	centers_pca = pca.transform(model.cluster_centers_)

	pyplot.scatter(centers_pca[:, 0], centers_pca[:, 1], s=150, alpha=0.75, marker='X')

	pyplot.legend(handles=scatter.legend_elements()[0], labels=[f'Cluster {i}' for i in range(model.n_clusters)])

	pyplot.show()

# Trains the model chosen by CLI arguments with the desired dataset
def train_model(training_data : pandas.DataFrame, training_target : pandas.Series, args : argparse.Namespace) -> Any:

	# Verify the type of the arguments
	if (not isinstance(args, argparse.Namespace)) or (not isinstance(training_data, pandas.DataFrame)) or (not isinstance(training_target, pandas.Series)):

		# Print error message to the screen
		print('Error while Training Model: Conflicting argument data-types.\nExiting...')

		# Exit with error code
		sys.exit(1)

	if (args.supervised):

		input_dimension = len(training_data.columns)

		output_dimension = len(numpy.unique(training_target))

		model = tensorflow.keras.Sequential(
			[
				tensorflow.keras.layers.Dense(input_dimension, input_dim= input_dimension, activation='relu', name = 'input_dense'),
				tensorflow.keras.layers.Dense(input_dimension*2, activation='relu', name = 'hidden_dense_1'),
				tensorflow.keras.layers.Dense(output_dimension, activation='softmax', name = 'output_dense')
			],
			name='supervised-clustering-nn'
		)

		optimizer=tensorflow.keras.optimizers.Adam(learning_rate=args.learning_rate)

		model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy', 'f1_score'])

		mapping = None

		if ('iris' == args.dataset):

			mapping = {'setosa': 0, 'versicolor': 1, 'virginica': 2}

		elif ('cleveland' == args.dataset):

			mapping = {'healthy': 0, 'sickness_1': 1, 'sickness_2': 2, 'sickness_3': 3, 'sickness_4': 4}

		numerical_training_target = training_target.map(mapping)

		history = model.fit(training_data, numerical_training_target, epochs=args.epochs_amount, verbose=1)

		plot_metrics(history, ['accuracy', 'loss'])

	else:

		# Executes and plots the elbow method for clustering amount approximation
		elbow_method(training_data)

		# Creates the k-means model and fits the training data
		model = KMeans(n_clusters=args.cluster_amount, max_iter=args.max_iterations, random_state=args.random_state).fit(training_data)

		# Benchmarks the training data target against the predicted labels
		benchmark_kmeans('PCA - Clusterização de Treino K-Means', training_data, training_target, model.labels_, model)

	return model

# Tests the model chosen by CLI arguments with the desired dataset
def test_model(testing_data : pandas.DataFrame, testing_target : pandas.Series, model : Any, args : argparse.Namespace) -> None:

	# Verify the type of the arguments
	if (not isinstance(args, argparse.Namespace)) or (not isinstance(testing_data, pandas.DataFrame)) or (not isinstance(testing_target, pandas.Series)):

		# Print error message to the screen
		print('Error while Testing Model: Conflicting argument data-types.\nExiting...')

		# Exit with error code
		sys.exit(1)

	if (args.supervised):

		history = model.predict(testing_data)

		benchmark_supervised_model('Clusterização de Teste Supervisionado', testing_data, testing_target, history, model)

	else:

		test_labels = model.fit_predict(testing_data)

		benchmark_kmeans('PCA - Clusterização de Teste K-Means', testing_data, testing_target, test_labels, model)

# Main file execution guard
if __name__ == '__main__':

	# Configures the matplotlib and seaborn palettes globally
	seaborn.set_palette("magma")

	# Creates and adds the necessary arguments to the parser
	parser = preconfigure_parser()

	# Parses the arguments
	args = parser.parse_args()

	# Loads the raw dataset
	raw_dataset = load_dataset(args)

	# Sanitizes the raw dataset
	sanitized_dataset = sanitize_dataset(raw_dataset, args)

	# Shows the complete dataset
	show_dataset(sanitized_dataset, args)

	# Splits the dataset between testing and training
	training_data, training_target, testing_data, testing_target = split_dataset(sanitized_dataset, args)

	# Trains the desired model and dataset and saves training data
	model = train_model(training_data, training_target, args)

	# Tests the desired model
	test_model(testing_data, testing_target, model, args)
