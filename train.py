train_record_fname = 'C:/ssd/train.tfrecord'
val_record_fname = 'C:/ssd/validation.tfrecord'
label_map_pbtxt_fname = 'C:/ssd/labelmap.pbtxt'

# Change the chosen_model variable to deploy different models available in the TF2 object detection zoo
chosen_model = 'ssd-mobilenet-v2-fpnlite-320'
MODELS_CONFIG = {
    'ssd-mobilenet-v2': {
        'model_name': 'ssd_mobilenet_v2_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_320x320_coco17_tpu-8.tar.gz',
    },
    'efficientdet-d0': {
        'model_name': 'efficientdet_d0_coco17_tpu-32',
        'base_pipeline_file': 'ssd_efficientdet_d0_512x512_coco17_tpu-8.config',
        'pretrained_checkpoint': 'efficientdet_d0_coco17_tpu-32.tar.gz',
    },
    'ssd-mobilenet-v2-fpnlite-320': {
        'model_name': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8',
        'base_pipeline_file': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.config',
        'pretrained_checkpoint': 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz',
    },
}
model_name = MODELS_CONFIG[chosen_model]['model_name']
pretrained_checkpoint = MODELS_CONFIG[chosen_model]['pretrained_checkpoint']
base_pipeline_file = MODELS_CONFIG[chosen_model]['base_pipeline_file']

# Set training parameters for the model
num_steps = 40000

if chosen_model == 'efficientdet-d0':
  batch_size = 1
else:
  batch_size = 1
  # Set file locations and get number of classes for config file
pipeline_fname = 'C:/ssd/models/mymodel/' + base_pipeline_file
fine_tune_checkpoint = 'C:/ssd/models/mymodel/' + model_name + '/checkpoint/ckpt-0'

def get_num_classes(pbtxt_fname):
    from object_detection.utils import label_map_util
    label_map = label_map_util.load_labelmap(pbtxt_fname)
    categories = label_map_util.convert_label_map_to_categories(
        label_map, max_num_classes=90, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    return len(category_index.keys())
num_classes = get_num_classes(label_map_pbtxt_fname)
print('Total classes:', num_classes)

pipeline_file = 'C:/ssd/models/mymodel/pipeline_file.config'
model_dir = 'C:/ssd/content/training/'

import subprocess

# Define the command to execute
command = [
    "python",
    "C:/ssd/models/research/object_detection/model_main_tf2.py",
    "--pipeline_config_path=" + pipeline_file,
    "--model_dir=" + model_dir,
    "--alsologtostderr",
    "--num_train_steps=" + str(num_steps),
    "--sample_1_of_n_eval_examples=1"
]

subprocess.run(command)

output_directory = 'C:/ssd/custom_model_lite'

# Path to training directory (the conversion script automatically chooses the highest checkpoint file)
last_model_path = 'C:/ssd/content/training'

command_tflite = [
    "python",
    "C:/ssd/models/research/object_detection/export_tflite_graph_tf2.py",
    "--trained_checkpoint_dir", last_model_path,
    "--output_directory", output_directory,
    "--pipeline_config_path", pipeline_file
]

subprocess.run(command_tflite)

# Convert exported graph file into TFLite model file
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model('C:/ssd/custom_model_lite/saved_model')
tflite_model = converter.convert()

with open('C:/ssd/custom_model_lite/detect.tflite', 'wb') as f:
  f.write(tflite_model)



