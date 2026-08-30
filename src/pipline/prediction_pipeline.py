import sys
from src.entity.config_entity import WinePredictorConfig
from src.entity.s3_estimator import Proj1Estimator
from src.exception import MyException
from src.logger import logging
from pandas import DataFrame


class WineData:
    def __init__(self,
                 fixed_acidity,
                 volatile_acidity,
                 citric_acid,
                 residual_sugar,
                 chlorides,
                 free_sulfur_dioxide,
                 total_sulfur_dioxide,
                 density,
                 pH,
                 sulphates,
                 alcohol
                 ):
        """
        Wine Data constructor
        Input: all features of the trained model for prediction
        """
        try:
            self.fixed_acidity = fixed_acidity
            self.volatile_acidity = volatile_acidity
            self.citric_acid = citric_acid
            self.residual_sugar = residual_sugar
            self.chlorides = chlorides
            self.free_sulfur_dioxide = free_sulfur_dioxide
            self.total_sulfur_dioxide = total_sulfur_dioxide
            self.density = density
            self.pH = pH
            self.sulphates = sulphates
            self.alcohol = alcohol

        except Exception as e:
            raise MyException(e, sys) from e

    def get_wine_input_data_frame(self) -> DataFrame:
        """
        This function returns a DataFrame from WineData class input
        """
        try:

            wine_input_dict = self.get_wine_data_as_dict()
            return DataFrame(wine_input_dict)

        except Exception as e:
            raise MyException(e, sys) from e

    def get_wine_data_as_dict(self):
        """
        This function returns a dictionary from WineData class input
        """
        logging.info("Entered get_wine_data_as_dict method as WineData class")

        try:
            input_data = {
                "fixed acidity": [self.fixed_acidity],
                "volatile acidity": [self.volatile_acidity],
                "citric acid": [self.citric_acid],
                "residual sugar": [self.residual_sugar],
                "chlorides": [self.chlorides],
                "free sulfur dioxide": [self.free_sulfur_dioxide],
                "total sulfur dioxide": [self.total_sulfur_dioxide],
                "density": [self.density],
                "pH": [self.pH],
                "sulphates": [self.sulphates],
                "alcohol": [self.alcohol]
            }

            logging.info("Created wine data dict")
            logging.info("Exited get_wine_data_as_dict method as WineData class")

            return input_data

        except Exception as e:
            raise MyException(e, sys) from e


class WineDataClassifier:
    def __init__(
            self,
            prediction_pipeline_config: WinePredictorConfig = WinePredictorConfig(),
    ) -> None:
        """
        :param prediction_pipeline_config: Configuration for prediction the value
        """
        try:
            self.prediction_pipeline_config = prediction_pipeline_config

        except Exception as e:
            raise MyException(e, sys)

    def predict(self, dataframe) -> str:
        """
        This is the method of WineDataClassifier
        Returns: Prediction in string format
        """
        try:
            logging.info("Entered predict method of WineDataClassifier class")

            model = Proj1Estimator(
                bucket_name=self.prediction_pipeline_config.model_bucket_name,
                model_path=self.prediction_pipeline_config.model_file_path,
            )

            result = model.predict(dataframe)

            return result

        except Exception as e:
            raise MyException(e, sys) from e
