from src.transformer_MT import logger
from src.transformer_MT.pipeline.data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion stage"
try:
    logger.info(f">>>>>> stage {STAGE_NAME} start <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx===========x")
except Exception as e:
    logger.exception(e)
    raise e






# import os

# class DataIngestion:
#     def __init__(self, en_path, vi_path):
#         self.en_path = en_path
#         self.vi_path = vi_path

#     def load_data(self):
#         if not os.path.exists(self.en_path) or not os.path.exists(self.vi_path):
#             raise FileNotFoundError("One or both file not found")
        
#         with open(self.en_path, 'r', encoding='utf-8') as en_file, open(self.vi_path, 'r', encoding='utf-8') as vi_file:
#             english_lines = [line.strip() for line in en_file.readlines()]
#             vietnamses_lines = [line.strip() for line in vi_file.readlines()]

#         if len(english_lines) != len(vietnamses_lines):
#             raise ValueError("The number of lines in the English and Vietnamese files do not match.")


#         return [(en, vi) for en, vi in zip(english_lines, vietnamses_lines)]


