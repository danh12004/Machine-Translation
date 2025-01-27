from src.transformer_MT import logger
from src.transformer_MT.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.transformer_MT.pipeline.data_preprocessing import DataPreprocessingTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Preprocessing stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_preprocessing = DataPreprocessingTrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e


