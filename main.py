from src.transformer_MT import logger
from src.transformer_MT.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.transformer_MT.pipeline.data_preprocessing import DataPreprocessingTrainingPipeline
from src.transformer_MT.pipeline.token_vocab_transform import TokenVocabTransformTrainingPipeline
from src.transformer_MT.pipeline.data_loader import DataLoaderTrainingPipeline
from src.transformer_MT.pipeline.training import TrainingPipeline

STAGE_NAME = "Data Ingestion"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Preprocessing"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_preprocessing = DataPreprocessingTrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Tokenizer Vocabulary Transform"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_preprocessing = TokenVocabTransformTrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Data Loader"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_preprocessing = DataLoaderTrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Training"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_preprocessing = TrainingPipeline()
    data_preprocessing.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e

