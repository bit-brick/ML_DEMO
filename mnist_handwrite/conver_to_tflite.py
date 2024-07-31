# convert_to_tflite.py

import tensorflow as tf

import numpy as np
def representative_dataset_gen():
    for _ in range(250):
        yield [np.random.uniform(0.0, 1.0, size=(1, 28, 28, 1)).astype(np.float32)] #需要輸入端大小
        

def convert_model_to_tflite(model_path, output_tflite_path):
    """
    Converts a saved Keras model to TFLite format with post-training quantization.
    
    Args:
        model_path (str): Path to the saved Keras model.
        output_tflite_path (str): Path where the TFLite model will be saved.
    """
    # Load the saved model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model to TFLite format with post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type  = tf.float32
    converter.inference_output_type = tf.float32
    converter.representative_dataset = representative_dataset_gen
    tflite_model = converter.convert()
    
    # Save the TFLite model
    with open(output_tflite_path, 'wb') as f:
        f.write(tflite_model)
        
    print(f"Quantized TFLite model saved at {output_tflite_path}")

# Call the function with your paths
convert_model_to_tflite('mnist_model.h5', 'mnist_model_quantized.tflite')