import pytest
import numpy as np
import sys, os
from tensorflow import keras
from tensorflow.keras.models import Sequential
import matplotlib

matplotlib.use('Agg')  # Prevent plots from popping up during tests
from types import SimpleNamespace
from pathlib import Path
# Adjust imports to match your DNN_binary_class.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import DNN_binary_class


#Test the function for data creation, training and testing sets
def test_Data_creation():

    At = [
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1'],
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1'],
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1']
    ]
    E = [
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1'],
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1'],
        ['0', '1', '1', '0', '1', '1', '0', '1', '0', '1', '1', '0', '1']
    ]
    sizextr = 2
    sizex = 2

    result=DNN_binary_class.Data_creation(sizextr,sizex,E,At)
    assert isinstance(result,tuple)


#Test define machine model from Keras
def test_define_model():
    size=1
    model=DNN_binary_class.define_model(size)

    # 1. Check type
    assert isinstance(model, Sequential)
    
    # 2. Check input shape
    assert model.input_shape == (None, size)
    
    # 3. Check output shape
    assert model.output_shape == (None, 1)
    
    # 4. Check number of layers
    assert len(model.layers) == 3  # Dense(50), Dense(20), Dense(1)
    
    # 5. Check activation functions of layers
    activations = [layer.activation.__name__ for layer in model.layers]
    assert activations == ['relu', 'relu', 'sigmoid']



#Test plot history function for machine learning model - accuracy and cross entropy
#Tests a mock of a history output, if it creates a file

def test_summarize_diagnostics_runs():
    # Create a mock Keras history object
    history = SimpleNamespace()
    history.history = {
        'loss': [0.5, 0.4, 0.3],
        'val_loss': [0.6, 0.45, 0.35],
        'accuracy': [0.7, 0.8, 0.85],
        'val_accuracy': [0.65, 0.78, 0.82]
    }

    # Override sys.argv[0] so the file saving works in tmp_path
    import sys

    #Name of location + namefile
    sys.argv[0] = str("test_script")
    filename = sys.argv[0].split('/')[-1] + "_plot.png"
    save_path = Path(filename)

    # Call the function
    DNN_binary_class.summarize_diagnostics(history)

    assert save_path.exists()
    assert save_path.suffix == ".png"

    #Deletes file
    save_path.unlink()

test_summarize_diagnostics_runs()