from flask import Flask, render_template
import pandas as pd

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

# Initialize the data
df, player_list = load_player_data()

@app.route('/')
def index():
    return render_template('index.html', players=player_list)

if __name__ == '__main__':
    app.run(debug=True)
