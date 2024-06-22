import argparse
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import requests
import pandas as pd
import io
import base64
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

app = Flask(__name__)
socketio = SocketIO(app)

# Define the neural network model
class SimpleNN(nn.Module):
    def __init__(self, hidden_neurons=8, output_neurons=10):
        super(SimpleNN, self).__init__()
        self.hidden = nn.Linear(64, hidden_neurons)  # Adjusted for 8x8 input
        self.output = nn.Linear(hidden_neurons, output_neurons)  # Adjusted for number of classes
        self.activation = nn.ReLU()  # Activation function

    def forward(self, x):
        x = self.activation(self.hidden(x))
        x = self.output(x)  # No activation here; will use CrossEntropyLoss
        return x

    def get_weight_bias_distributions(self):
        weights = self.hidden.weight.data.flatten().tolist()
        biases = self.hidden.bias.data.flatten().tolist()
        return weights, biases

    def get_activation_distribution(self, x):
        with torch.no_grad():
            hidden_activations = self.activation(self.hidden(x)).flatten().tolist()
        return hidden_activations

# Preprocess the Optical Recognition of Handwritten Digits data
def preprocess_digits(target_size=(8, 8), limit_per_digit=17, num_classes=10):
    # Download the dataset
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra'
    response = requests.get(url)
    data = pd.read_csv(io.StringIO(response.text), header=None)

    images = data.iloc[:, :-1].values.reshape(-1, 8, 8)
    labels = data.iloc[:, -1].values

    processed_images = []
    processed_labels = []

    digit_count = {digit: 0 for digit in range(num_classes)}

    for img, label in zip(images, labels):
        if label >= num_classes:
            continue  # Skip labels that are not in the desired range

        if digit_count[label] < limit_per_digit:
            img = img / 16.0  # Normalize pixel values to range [0, 1]
            img = 1 - img  # Invert the image
            processed_images.append(img.flatten())

            processed_labels.append(label)
            digit_count[label] += 1

            # Debugging: print the current count of each digit
            print(f"Added digit {label}: current count {digit_count}")

        if all(count >= limit_per_digit for count in digit_count.values()):
            break

    # Debugging: print final counts
    print(f"Final digit counts: {digit_count}")

    return np.array(processed_images), np.array(processed_labels)

# Training loop
def train(model, criterion, optimizer, dataloader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    training_metrics = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'training_accuracy': [], 'validation_accuracy': []}
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        training_loss = epoch_loss / len(dataloader)
        training_accuracy = 100. * correct / total
        
        # Validation metrics
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        validation_loss = val_loss / len(dataloader)
        validation_accuracy = 100. * val_correct / val_total
        
        # Log metrics
        training_metrics['epoch'].append(epoch)
        training_metrics['training_loss'].append(training_loss)
        training_metrics['validation_loss'].append(validation_loss)
        training_metrics['training_accuracy'].append(training_accuracy)
        training_metrics['validation_accuracy'].append(validation_accuracy)
        
        if epoch % 1 == 0:
            log_message = f"Epoch {epoch}, Training Loss: {training_loss}, Validation Loss: {validation_loss}, Training Accuracy: {training_accuracy}, Validation Accuracy: {validation_accuracy}"
            print(log_message)
            socketio.emit('log', {'message': log_message})
            socketio.emit('training_metrics', training_metrics)
    
    model.cpu()

@app.route('/')
def index():
    num_classes = args.num_classes
    return render_template('index.html', num_classes=num_classes)

@app.route('/predict', methods=['POST'])
def predict():
    input_grid = request.json['inputGrid']
    inverted_input_grid = 1 - np.array(input_grid).flatten()

    input_tensor = torch.tensor(inverted_input_grid[np.newaxis, :], dtype=torch.float32)
    with torch.no_grad():
        hidden_activations = model.activation(model.hidden(input_tensor)).numpy()
        output_activations = model(input_tensor).numpy()

    predicted_digit = int(np.argmax(output_activations))
    result = {
        'predictedDigit': predicted_digit,
        'hiddenActivations': hidden_activations.tolist(),
        'outputActivations': output_activations.tolist()
    }
    return jsonify(result)

@app.route('/train', methods=['POST'])
def train_model():
    global model, input_data, target_data
    epochs = int(request.json['epochs'])

    # Invert the input data for training
    inv_input_data = 1 - input_data

    dataset = TensorDataset(torch.tensor(inv_input_data, dtype=torch.float32), torch.tensor(target_data, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, criterion, optimizer, dataloader, epochs)
    model.eval()
    return jsonify({'message': 'Training completed'})

@app.route('/clear', methods=['POST'])
def clear_model_data():
    global model
    model = SimpleNN(args.hidden_neurons, args.num_classes)
    return jsonify({'message': 'Model data cleared'})

@app.route('/training_data', methods=['GET'])
def get_training_data():
    training_data = []
    for img in input_data:
        img = Image.fromarray((img.reshape(8, 8) * 255).astype(np.uint8))
        img = img.resize((28, 28), Image.NEAREST)
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
        training_data.append(img_str)
    return jsonify({'trainingData': training_data})

@app.route('/train_single', methods=['POST'])
def train_single_example():
    global model
    data = request.json
    input_grid = np.array(data['inputGrid'])
    digit = data['digit']

    # Invert the input data for training
    inv_input_data = 1 - input_grid.flatten()

    input_tensor = torch.tensor([inv_input_data], dtype=torch.float32)
    target_tensor = torch.tensor([digit], dtype=torch.long)

    dataset = TensorDataset(input_tensor, target_tensor)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train(model, criterion, optimizer, dataloader, epochs=1)
    model.eval()
    return jsonify({'message': f'Trained on single example of digit {digit}'})

@app.route('/load_training_image', methods=['POST'])
def load_training_image():
    index = request.json['index']
    img = input_data[index].reshape(8, 8)
    input_grid = img.tolist()
    return jsonify({'inputGrid': input_grid})

@app.route('/training_metrics', methods=['GET'])
def get_training_metrics():
    return jsonify(training_metrics)

@app.route('/distributions', methods=['POST'])
def get_distributions():
    input_grid = request.json['inputGrid']
    inverted_input_grid = 1 - np.array(input_grid).flatten()
    input_tensor = torch.tensor(inverted_input_grid[np.newaxis, :], dtype=torch.float32)

    weights, biases = model.get_weight_bias_distributions()
    activations = model.get_activation_distribution(input_tensor)
    
    with torch.no_grad():
        output = model(input_tensor)
        confidence = torch.nn.functional.softmax(output, dim=1).flatten().tolist()

    def create_histogram(data, title):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(data, bins=30)
        ax.set_title(title)
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    weight_hist = create_histogram(weights, 'Weight Distribution')
    bias_hist = create_histogram(biases, 'Bias Distribution')
    activation_hist = create_histogram(activations, 'Activation Distribution')

    return jsonify({
        'weightHist': weight_hist,
        'biasHist': bias_hist,
        'activationHist': activation_hist,
        'confidence': confidence
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple neural network on the Optical Recognition of Handwritten Digits dataset.')
    parser.add_argument('--hidden_neurons', type=int, default=16, help='Number of neurons in the hidden layer (default: 16)')
    parser.add_argument('--limit_per_digit', type=int, default=17, help='Number of digits per class for training (default: 17)')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of output neurons and number of digit classes (default: 10)')

    args = parser.parse_args()

    input_data, target_data = preprocess_digits(limit_per_digit=args.limit_per_digit, num_classes=args.num_classes)

    model = SimpleNN(hidden_neurons=args.hidden_neurons, output_neurons=args.num_classes)
    model.eval()

    training_metrics = {'epoch': [], 'training_loss': [], 'validation_loss': [], 'training_accuracy': [], 'validation_accuracy': []}

socketio.run(app, host='0.0.0.0', debug=True, port=8000)
