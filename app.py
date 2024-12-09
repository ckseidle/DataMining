from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Read the CSV data
def load_player_data():
    try:
        df = pd.read_csv('data/yearly_player_data.csv')
        # Create a list of unique player names for the dropdown
        players = sorted(df['player_name'].unique())
        return df, players
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, []

# Define key statistics for comparison
COMPARISON_STATS = {
    'Games Per Season': 'games',
    'Fantasy Points (PPR)': 'fantasy_points_ppr',
    'Pass Attempts': 'pass_attempts',
    'Passing Yards': 'passing_yards',
    'Pass TDs': 'pass_td',
    'Interceptions': 'interception',
    'Targets': 'targets',
    'Receptions': 'receptions',
    'Receiving Yards': 'receiving_yards',
    'Receiving TDs': 'reception_td',
    'Rush Attempts': 'rush_attempts',
    'Rushing Yards': 'rushing_yards',
    'Rushing TDs': 'run_td',
    'Yards Per Game': 'ypg',
    'Points Per Game': 'ppg'
}

# Initialize the data
df, player_list = load_player_data()

def get_player_career_averages(player_data):
    # Select only the columns we need for comparison
    stats_to_average = list(COMPARISON_STATS.values())
    # Calculate mean for only the numeric columns we care about
    averages = player_data[stats_to_average].astype(float).mean()
    return averages

# Define which stats to predict
PREDICTION_STATS = {
    'Fantasy Points (PPR)': 'fantasy_points_ppr',
    'Games': 'games',
    'Passing Yards': 'passing_yards',
    'Pass TDs': 'pass_td',
    'Receiving Yards': 'receiving_yards',
    'Receiving TDs': 'reception_td',
    'Rushing Yards': 'rushing_yards',
    'Rushing TDs': 'run_td',
    'Points Per Game': 'ppg'
}

def predict_next_season(player_data):
    predictions = {}
    
    if len(player_data) < 2:  # Need at least 2 seasons for prediction
        return None
        
    # Sort by season
    player_data = player_data.sort_values('season')
    
    for stat_name, stat_column in PREDICTION_STATS.items():
        # Prepare data for prediction
        X = player_data['season'].values.reshape(-1, 1)
        y = player_data[stat_column].values
        
        # Create and fit the model
        model = LinearRegression()
        model.fit(X, y)
        
        # Predict next season
        next_season = player_data['season'].max() + 1
        prediction = model.predict([[next_season]])[0]
        
        # Ensure predictions are not negative
        prediction = max(0, prediction)
        
        # Round the prediction appropriately
        predictions[stat_name] = round(prediction, 2)
        
    return predictions

@app.route('/', methods=['GET', 'POST'])
def index():
    comparison_data = None
    
    if request.method == 'POST':
        player1 = request.form.get('player1')
        player2 = request.form.get('player2')
        
        # Get all seasons data for each player
        player1_data = df[df['player_name'] == player1]
        player2_data = df[df['player_name'] == player2]
        
        # Calculate career averages
        player1_averages = get_player_career_averages(player1_data)
        player2_averages = get_player_career_averages(player2_data)
        
        # Create comparison dictionary
        comparison_data = {
            'player1_name': player1,
            'player2_name': player2,
            'stats': {},
            'player1_seasons': len(player1_data),
            'player2_seasons': len(player2_data),
            'player1_years': f"{int(player1_data['season'].min())}-{int(player1_data['season'].max())}",
            'player2_years': f"{int(player2_data['season'].min())}-{int(player2_data['season'].max())}"
        }
        
        # Compile stats for comparison
        for stat_name, stat_column in COMPARISON_STATS.items():
            comparison_data['stats'][stat_name] = {
                'player1': round(float(player1_averages[stat_column]), 2),
                'player2': round(float(player2_averages[stat_column]), 2)
            }
    
    return render_template('index.html', 
                         players=player_list,
                         comparison_data=comparison_data)

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction_data = None
    error_message = None
    
    if request.method == 'POST':
        player_name = request.form.get('player')
        
        if player_name:
            # Get player's historical data
            player_data = df[df['player_name'] == player_name]
            
            if len(player_data) < 2:
                error_message = "Need at least 2 seasons of data for prediction"
            else:
                # Get predictions
                predictions = predict_next_season(player_data)
                
                if predictions:
                    prediction_data = {
                        'player_name': player_name,
                        'next_season': int(player_data['season'].max() + 1),
                        'predictions': predictions,
                        'last_season': {
                            stat_name: round(player_data[stat_column].iloc[-1], 2)
                            for stat_name, stat_column in PREDICTION_STATS.items()
                        }
                    }
                else:
                    error_message = "Unable to generate predictions"
    
    return render_template('predict.html', 
                         players=player_list,
                         prediction_data=prediction_data,
                         error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True)
