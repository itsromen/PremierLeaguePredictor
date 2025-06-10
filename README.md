# PremierLeaguePredictor

This project is a machine learning application that predicts the outcome of Premier League football matches based on key match statistics. The system uses a Random Forest Classifier trained on historical match data to predict whether a match will result in a Home Win, Away Win, or Draw.



Features
- Predicts match outcomes based on:
  + Possession difference between teams
  + Shot difference between teams
  + Stadium attendance
- Provides prediction confidence levels for each possible outcome
- User-friendly GUI interface
- Sample data loading for quick testing



How to Run:

1- Install dependencies: pip install -r requirements.txt

2- Run the data preprocessing script: python preprocess.py

3- Train the machine learning model: python train_model.py

4- Launch the GUI application: python gui_app.py



Usage
- Enter the match statistics in the GUI:
  + Possession Difference (%): Home team possession minus away team possession
  + Shot Difference: Home team shots minus away team shots
  + Attendance: Total number of spectators at the match
- Click "Predict Outcome" to see the predicted result and confidence levels
- Use "Sample Data" to load example values or "Clear" to reset the inputs