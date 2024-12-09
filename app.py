from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json

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
    
    if len(player_data) < 2:  
        return None
        
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

# Define stats available for trending
TREND_STATS = {
    'Fantasy Points (PPR)': 'fantasy_points_ppr',
    'Games Played': 'games',
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

def prepare_trend_data(player_data, stat_column):
    player_data = player_data.sort_values('season')
    
    # Convert values to lists
    labels = [int(x) for x in player_data['season'].tolist()]
    values = [float(x) for x in player_data[stat_column].tolist()]
    trend = []
    
    # Calculate trend line
    if len(labels) > 1:
        X = np.array(labels).reshape(-1, 1)
        y = np.array(values)
        model = LinearRegression()
        model.fit(X, y)
        trend = [float(x) for x in model.predict(X)]
    
    return {
        'labels': labels,
        'values': values,
        'trend': trend
    }

RANKABLE_STATS = {
    'Fantasy Points (PPR)': 'fantasy_points_ppr',
    'Passing Yards': 'passing_yards',
    'Pass TDs': 'pass_td',
    'Receiving Yards': 'receiving_yards',
    'Receiving TDs': 'reception_td',
    'Receptions': 'receptions',
    'Rushing Yards': 'rushing_yards',
    'Rushing TDs': 'run_td',
    'Total TDs': 'total_tds',
    'Total Yards': 'total_yards',
    'Points Per Game': 'ppg'
}

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

@app.route('/trends', methods=['GET', 'POST'])
def trends():
    labels = []
    values = []
    trend = []
    selected_player = None
    selected_stat = None
    
    if request.method == 'POST':
        player_name = request.form.get('player')
        stat_name = request.form.get('stat')
        
        if player_name and stat_name:
            player_data = df[df['player_name'] == player_name]
            stat_column = TREND_STATS[stat_name]
            
            if not player_data.empty:
                try:
                    data = prepare_trend_data(player_data, stat_column)
                    labels = data['labels']
                    values = data['values']
                    trend = data['trend']
                    selected_player = player_name
                    selected_stat = stat_name
                except Exception as e:
                    print(f"Error preparing trend data: {e}")
    
    return render_template('trends.html',
                         players=player_list,
                         stats=list(TREND_STATS.keys()),
                         labels=labels,
                         values=values,
                         trend=trend,
                         selected_player=selected_player,
                         selected_stat=selected_stat)

@app.route('/rankings', methods=['GET', 'POST'])
def rankings():
    rankings_data = None
    selected_stat = None
    selected_season = None
    
    seasons = sorted(df['season'].unique(), reverse=True)
    
    if request.method == 'POST':
        stat_name = request.form.get('stat')
        season = request.form.get('season')
        
        if stat_name and season:
            stat_column = RANKABLE_STATS[stat_name]
            season = int(season)
            
            season_data = df[df['season'] == season]
            
            # Get top 10 players for selected stat
            top_players = season_data.nlargest(10, stat_column)[
                ['player_name', 'team', stat_column, 'position']
            ].to_dict('records')
            
            rankings_data = top_players
            selected_stat = stat_name
            selected_season = season
    
    return render_template('rankings.html',
                         stats=list(RANKABLE_STATS.keys()),
                         seasons=seasons,
                         rankings_data=rankings_data,
                         selected_stat=selected_stat,
                         selected_season=selected_season,
                         RANKABLE_STATS=RANKABLE_STATS)

if __name__ == '__main__':
    app.run(debug=True)
