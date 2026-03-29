Data Structure: Historical data (e.g., closing prices, volume) is normalized (e.g., to range 




) and reshaped into chronological sequences (e.g., 60 days) to predict future values.
Architecture: Typically consists of input layers, multiple hidden LSTM layers (e.g., 50-512 units), Dropout layers to prevent overfitting, and Dense layers for the final output.
Training & Evaluation: Models are trained over multiple epochs using loss functions such as Mean Squared Error (MSE), with performance evaluated using Root Mean Squared Error (RMSE).
Input Features: In addition to price, models often incorporate technical indicators (e.g., moving averages) or sentiment analysis from news/social media
