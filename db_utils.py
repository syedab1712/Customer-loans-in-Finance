import matplotlib.pyplot as plt
import pandas as pd
import psycopg2
from sqlalchemy import create_engine
import yaml
import seaborn as sns

# Load credentials from YAML file
with open('credentials.yaml', 'r') as f:
    credentials = yaml.safe_load(f)

class RDSDatabaseConnector:
    """
    Class to connect to an RDS database, extract data, and save it to a file.

    Attributes:
    - credentials: Dictionary containing the database credentials

    Methods:
    - __init__(self, credentials): Initializes an instance of the class 
    - _create_engine(self): Establishes a connection to the database
    - initialise_engine(self): Initializes the connection to the database
    - data_extraction(self, table_name='loan_payments'): Extracts data from the specified table
    - save_to_file(self, data, file_path='loan_payments_data.csv'): Saves the data to a CSV file
    - load_loan_data(self, file_path='loan_payments_data.csv'): Loads data from a CSV file
    """

    def __init__(self, credentials):
        self.credentials = credentials
        self.engine = None

    def _create_engine(self):
        return create_engine(f"postgresql+psycopg2://{self.credentials['RDS_USER']}:{self.credentials['RDS_PASSWORD']}@{self.credentials['RDS_HOST']}:{self.credentials['RDS_PORT']}/{self.credentials['RDS_DATABASE']}")

    def initialise_engine(self):
        self.engine = self._create_engine()

    def data_extraction(self, table_name='loan_payments'):
        query = f"SELECT * FROM {table_name};"
        return pd.read_sql(query, self.engine)

    def save_to_file(self, data, file_path='loan_payments_data.csv'):
        data.to_csv(file_path, index=False)

    def load_loan_data(self, file_path='loan_payments_data.csv'):
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return None

# Initialize the RDS connector and extract data
loan_data_connector = RDSDatabaseConnector(credentials)
loan_data_connector.initialise_engine()
loan_data = loan_data_connector.data_extraction()
loan_data_connector.save_to_file(loan_data, 'loan_payments_data.csv')

# Load data into a DataFrame
loans_df = loan_data_connector.load_loan_data('loan_payments_data.csv')
if loans_df is not None:
    loans_df.set_index('id', inplace=True)

# Clean term and employment_length columns
loans_df['term'] = loans_df['term'].str.replace('months', '', regex=True).str.strip()
loans_df['employment_length'] = loans_df['employment_length'].str.replace('years?', '', regex=True).str.strip()

# Convert columns to numeric, forcing errors to NaN
numeric_columns = ['funded_amount', 'term', 'int_rate']
loans_df[numeric_columns] = loans_df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Extract columns with null values
null_columns = loans_df.columns[loans_df.isnull().any()]
numeric_null_columns = loans_df[null_columns].select_dtypes(include=['number']).columns
print(numeric_null_columns)

# Create histograms of columns with null values
for column in ['funded_amount', 'term', 'int_rate']:
    loans_df[column].hist(bins=40)
    plt.title(f"Histogram of {column}")
    plt.show()

class DataFrameTransform:
    """
    Class to transform data for analysis.

    Attributes:
    - loans_df: DataFrame containing loan information

    Methods:
    - identify_skewed_columns: Identifies columns with significant skewness
    - visualize_skewness: Visualizes skewness for each column
    - transform_columns: Transforms columns where type is numeric
    - null_impute: Imputes null values with the median or mode
    - outlier_removal: Removes outliers from columns
    - save_transformed_data: Saves transformed data to a file
    """

    def __init__(self, loans_df):
        self.loans_df = loans_df

    def identify_skewed_columns(self, threshold=0.5):
        skewed_columns = self.loans_df.apply(lambda x: abs(x.skew()) > threshold)
        return skewed_columns[skewed_columns].index.tolist()

    def visualize_skewness(self, columns):
        for column in columns:
            self.loans_df[column].hist(bins=40)
            plt.title(f"Histogram of {column}")
            plt.show()

    def transform_columns(self, columns, transformation):
        for column in columns:
            if pd.api.types.is_numeric_dtype(self.loans_df[column]):
                self.loans_df[column] = transformation(self.loans_df[column])

    def null_impute(self):
        for column in self.loans_df.columns:
            if self.loans_df[column].dtype == 'float64' and self.loans_df[column].isnull().any():
                self.loans_df[column].fillna(self.loans_df[column].median(), inplace=True)
            else:
                self.loans_df[column].fillna(self.loans_df[column].mode()[0], inplace=True)

    def outlier_removal(self, column):
        Q1 = column.quantile(0.25)
        Q3 = column.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return column[(column >= lower_bound) & (column <= upper_bound)]

    def save_transformed_data(self, filename='transformed_data.csv'):
        self.loans_df.to_csv(filename, index=False)

transform = DataFrameTransform(loans_df)
skewed_columns = transform.identify_skewed_columns()
transform.visualize_skewness(skewed_columns)
transform.null_impute()
transform.save_transformed_data('transformed_data.csv')

class Plotter:
    """
    Class to plot data using various visualization methods.

    Attributes: 
    - loans_df: DataFrame containing loan information 

    Methods: 
    - null_percent_funded: Calculates the percentage of null values in funded_amount
    - null_percent_term: Calculates the percentage of null values in term
    - null_percent_int_rate: Calculates the percentage of null values in int_rate
    - skew_check: Visualizes the skew of numeric columns
    """

    def __init__(self, loans_df):
        self.loans_df = loans_df

    def plot_distribution(self, column):
        self.loans_df[column].hist(bins=40)
        plt.title(f"Histogram of {column}")
        plt.show()

    def null_percent_funded(self):
        return self.loans_df['funded_amount'].isnull().sum() * 100 / len(self.loans_df['funded_amount'])

    def null_percent_term(self):
        return self.loans_df['term'].isnull().sum() * 100 / len(self.loans_df['term'])

    def null_percent_int_rate(self):
        return self.loans_df['int_rate'].isnull().sum() * 100 / len(self.loans_df['int_rate'])

    def skew_check(self):
        numeric_data = self.loans_df.select_dtypes(include=['number']).columns
        f = pd.melt(self.loans_df, value_vars=numeric_data)
        g = sns.FacetGrid(f, col="variable", col_wrap=3, sharex=False, sharey=False)
        g.map(sns.histplot, "value", kde=True)
        plt.show()

plotting = Plotter(loans_df)
print(f"Null percent funded: {plotting.null_percent_funded()}")
print(f"Null percent term: {plotting.null_percent_term()}")
print(f"Null percent int_rate: {plotting.null_percent_int_rate()}")
plotting.skew_check()

class Analysis:
    """
    Class to analyze loans dataset and its impact on the loan company.

    Attributes:
    - loans_df: DataFrame containing loan information 

    Methods:
    - recovery: Calculates the recovery percentage
    - loss_charged_off: Calculates the revenue loss due to charged-off loans
    - sum_before_charged_off: Calculates the amount paid before loans were charged off
    - money_owed_choff_late: Calculates the revenue loss due to charged-off and late loans
    """

    def __init__(self, loans_df):
        self.loans_df = loans_df 

    def recovery(self):
        recovery_percentage = self.loans_df['total_payment'].sum() / self.loans_df['loan_amount'].sum() * 100
        print(f"Recovery Percentage: {recovery_percentage:.2f}%")
        return recovery_percentage

    def loss_charged_off(self):
        loan_loss = len(self.loans_df[self.loans_df['loan_status'] == 'Charged Off'])
        percentage_loan_loss = (loan_loss / len(self.loans_df)) * 100
        print(f"Charged Off Loans Percentage: {percentage_loan_loss:.2f}%")
        return percentage_loan_loss

    def sum_before_charged_off(self):
        charged_off = self.loans_df[self.loans_df['loan_status'] == 'Charged Off']
        total_payment_before_charged_off = charged_off['total_payment'].sum()
        print(f"Total Payment Before Charged Off: {total_payment_before_charged_off}")
        return total_payment_before_charged_off

    def money_owed_choff_late(self):
        charged_off = self.loans_df[self.loans_df['loan_status'] == 'Charged Off']
        late_loans = self.loans_df[self.loans_df['loan_status'].str.contains('Late')]
        total_money_owed = (charged_off['loan_amount'] - charged_off['total_payment']).sum() + (late_loans['loan_amount'] - late_loans['total_payment']).sum()
        print(f"Total Money Owed: {total_money_owed}")
        return total_money_owed

analysis = Analysis(loans_df)
analysis.recovery()
analysis.loss_charged_off()
analysis.sum_before_charged_off()
analysis.money_owed_choff_late()
