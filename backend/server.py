from flask import Flask, request, jsonify
import torch
from torch.utils.data import DataLoader
import pickle
from StockPriceDataset import StockPriceDataset
from lstm import LSTMModel
from torch import nn

app = Flask(__name__)

# request json format
# {
#     "company": "Apple",  #typical company name
#     "predict_type": "Day", # Day, Week, Month
#     "predict_period": 1, # 1, 2, 3, number of days, weeks, months
# }

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from POST request
    data = request.json
    company = data['company']
    predict_type = data['predict_type']
    predict_period = data['predict_period']

    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    # set directory for data and model
    data_dir = 'dataset/'
    model_dir = 'model/'    

    # calculate number of days to predict
    if predict_type.lower() == 'day':
        num_days = str(predict_period)
    elif predict_type.lower() == 'week':
        num_days = predict_period * 7
        num_days = str(num_days)
    elif predict_type.lower() == 'month':
        num_days = predict_period * 30
        num_days = str(num_days)
    else:
        return jsonify({'error': 'Invalid predict_type'})

    # Hyperparameters
    input_dim = 2 # number of features
    hidden_dim = 100 # number of hidden units
    num_layers = 3 # number of LSTM / GRU layers
    output_dim = int(num_days) # predict next specified days of stock price
    batch_size = 1

    # load test dataset
    test_dataset_name = 'test_dataset-'
    path = data_dir + test_dataset_name + company + '-' + num_days + '-' + '.pkl'
    with open(path, 'rb') as f:
        test = pickle.load(f)
    test_loader = DataLoader(test, batch_size=batch_size, shuffle=False)

    # initialize model, loss function, and optimizer
    model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim).to(device)
    criterion = nn.MSELoss()

    # load model weights
    model_name = 'lstm-'
    path = model_dir + model_name + company + '-' + num_days + '-' + '.pth'
    model.load_state_dict(torch.load(path, map_location=device))

    # predict
    model.eval()
    test_loss = 0.0
    num_test_batches = 0
    actual_values = []
    predicted_values = []

    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            targets = targets.squeeze(-1) # remove last dimension

            output = model(inputs)
            loss = criterion(output, targets)
            
            test_loss += loss.item()
            num_test_batches += 1

            # Record actual and predicted values
            actual_values.extend(targets.tolist())
            predicted_values.extend(output.tolist())

    test_loss /= num_test_batches

    # predicted_values = [item for sublist in predicted_values for item in sublist]
    # actual_values = [item for sublist in actual_values for item in sublist]

    actual_values = actual_values[0]
    predicted_values = predicted_values[0]
    # predicted_values = [item for sublist in predicted_values for item in sublist]
    # actual_values = [item for sublist in actual_values for item in sublist]

    return jsonify({
        'predicted_values': predicted_values, 
        'actual_values': actual_values
        })

if __name__ == '__main__':
    app.run(port=8080)