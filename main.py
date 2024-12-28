# Main file
# Use case: Pilot test, "integration test"

# === STEP 0: Imports
from modules.model_manager.model_manager import ModelManager
from modules.model_manager.helpers.configuration_loader import ConfigurationLoader as MMConfigLoader
from modules.model_manager.factories.model_factory import ModelFactory
from modules.model_manager.predictor.predictor import Predictor
from modules.model_manager.trainer.trainer import Trainer
from modules.processor.processor import Processor
from modules.processor.factories.split_strategy_factory import SplitStrategyFactory
from modules.processor.helpers.configuration_loader import ConfigurationLoader as PConfigLoader
from modules.dataset_generator.dataset_generator import DatasetGenerator
from modules.dataset_generator.factories.strategy_factory import StrategyFactory
from modules.dataset_generator.factories.join_factory import JoinFactory
from modules.dataset_generator.factories.feature_processor_factory import FeatureProcessorFactory
from modules.dataset_generator.factories.data_io_factory import DataIOFactory
from modules.dataset_generator.helpers.configuration_loader import ConfigurationLoader as DSConfigLoader
print("# === STEP 0: Imports: Starting ...")
print("# === STEP 0: Imports: Complete")

# === STEP 1: Dependency Instantiations And Global Variable Declarations
print("# === STEP 1: Dependency Instantiations And Global Variable Declarations: Starting ...")
# === DATASET GEN
yaml_path = '/Users/joaquinuriarte/Documents/GitHub/sports-betting/configs/model_v0.yaml'
ds_configuration_loader = DSConfigLoader()
data_factory, feature_processor_factory, join_factory, strategy_factory = DataIOFactory(
), FeatureProcessorFactory(), JoinFactory(), StrategyFactory()
# === PROCESSOR
p_configuration_loader = PConfigLoader()
split_strategy_factory = SplitStrategyFactory()
# === MODEL MANAGER
trainer = Trainer()
predictor = Predictor()
model_factory = ModelFactory()
mm_configuration_loader = MMConfigLoader()
print("# === STEP 1: Dependency Instantiations And Global Variable Declarations: Complete")

# === STEP 2: DATASET GEN
print("# === STEP 2: DATASET GEN: Starting ...")
dataset_generator = DatasetGenerator(yaml_path, ds_configuration_loader,
                                     data_factory, feature_processor_factory, join_factory, strategy_factory)
processed_dataset = dataset_generator.generate()
print("# === STEP 2: DATASET GEN: Complete")

# === STEP 3: PROCESSOR
print("# === STEP 3: PROCESSOR: Starting ...")
processor = Processor(yaml_path, p_configuration_loader,
                      processed_dataset, split_strategy_factory)
train_dataset, validation_dataset = processor.generate(val_dataset_flag=True)
print("# === STEP 3: PROCESSOR: Complete")

# === STEP 4: MODEL MANAGER
print("# === STEP 4: MODEL MANAGER: Starting ...")
model_manager = ModelManager(
    trainer, predictor, model_factory, mm_configuration_loader)
models_and_config = model_manager.create_models([yaml_path])
for item in models_and_config:
    model_manager.train(
        [item[0]], [tuple(train_dataset, validation_dataset)], save_after_training=True)
    # train needs to return the validation prediction. Otherwise we can call predict on validation as shown below
    validation_predictions = model_manager.predict(
        [item[0]], [validation_dataset.examples])
    print(f"Validation_predictions: {validation_predictions}")
    print(f"Validation_dataset: {validation_dataset}")
print("# === STEP 4: MODEL MANAGER: Complete")
