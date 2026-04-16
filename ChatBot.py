import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

# Load the dataset
data = pd.read_csv('chatbot_dataset.csv')

# Preprocess the data
nltk.download('punkt')
data['Question'] = data['Question'].apply(lambda x: ' '.join(nltk.word_tokenize(x.lower())))

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Question'], data['Answer'], test_size=0.2, random_state=42)

# Create a model pipeline
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X_train, y_train)

def get_response(question):
    question = ' '.join(nltk.word_tokenize(question.lower()))
    answer = model.predict([question])[0]
    return answer

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout with improved styling
app.layout = html.Div([
    html.Div(
        html.H1("Chatbot", style={'textAlign': 'center', 'color': '#2c3e50', 'font-family': 'Arial, sans-serif'}),
        style={'padding': '20px', 'backgroundColor': '#ecf0f1'}
    ),
    html.Div(
        dcc.Textarea(
            id='user-input',
            value='Type your question here...',
            style={
                'width': '100%', 'height': 100, 'padding': '10px', 'border': '2px solid #2c3e50',
                'borderRadius': '5px', 'font-family': 'Arial, sans-serif'
            }
        ),
        style={'margin': '20px'}
    ),
    html.Div(
        html.Button('Submit', id='submit-button', n_clicks=0, style={
            'padding': '10px 20px', 'font-size': '16px', 'color': '#fff', 'backgroundColor': '#2c3e50',
            'border': 'none', 'borderRadius': '5px', 'cursor': 'pointer', 'font-family': 'Arial, sans-serif'
        }),
        style={'textAlign': 'center'}
    ),
    html.Div(id='chatbot-output', style={'padding': '20px', 'font-family': 'Arial, sans-serif'}),
    html.Footer(
        html.P("Powered by Dash", style={'textAlign': 'center', 'color': '#95a5a6', 'font-size': '14px'}),
        style={'padding': '20px', 'backgroundColor': '#2c3e50'}
    )
], style={'font-family': 'Arial, sans-serif', 'backgroundColor': '#ecf0f1', 'maxWidth': '600px', 'margin': 'auto', 'border': '2px solid #2c3e50', 'borderRadius': '10px'})

# Define callback to update chatbot response
@app.callback(
    Output('chatbot-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('user-input', 'value')
)
def update_output(n_clicks, user_input):
    if n_clicks > 0:
        response = get_response(user_input)
        return html.Div([
            html.Div([
                html.P(f"You: {user_input}", style={'margin': '10px 0', 'font-weight': 'bold'}),
                html.P(f"Bot: {response}", style={
                    'margin': '10px 0', 'backgroundColor': '#f0f0f0', 'padding': '10px',
                    'borderRadius': '5px', 'borderLeft': '5px solid #2c3e50'
                })
            ], style={'backgroundColor': '#ecf0f1', 'padding': '15px', 'borderRadius': '10px', 'marginBottom': '20px'})
        ])
    return html.Div("Ask me something!", style={'textAlign': 'center', 'color': '#7f8c8d'})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)