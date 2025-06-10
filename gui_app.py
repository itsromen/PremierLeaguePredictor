import joblib
import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np

class PremierLeaguePredictor:
    def __init__(self, root):
        self.root = root
        self.root.title("Premier League Match Outcome Predictor")
        self.root.geometry("500x450")
        self.root.configure(bg='#f0f0f0')
        
        # Load trained model and scaler
        self.model = None
        self.scaler = None
        
        try:
            self.model = joblib.load('premier_league_model.pkl')
            print("Model loaded successfully")
        except FileNotFoundError:
            print("Model file not found - trying without scaler")
            messagebox.showwarning("Warning", "Model file not found! Please run train_model.py first.")
        
        try:
            self.scaler = joblib.load('scaler.pkl')
            print("Scaler loaded successfully")
        except FileNotFoundError:
            print("Scaler not found - will use original model without scaling")
            self.scaler = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Title
        title_label = tk.Label(
            self.root, 
            text="⚽ Premier League Match Predictor ⚽", 
            font=('Arial', 16, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=20)
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(padx=20, pady=10, fill='both', expand=True)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, text="Match Statistics", padding=20)
        input_frame.pack(fill='x', pady=10)
        
        # Possession difference
        ttk.Label(input_frame, text="Possession Difference (%):").grid(row=0, column=0, sticky='w', pady=5)
        self.possession_diff = ttk.Entry(input_frame, width=15)
        self.possession_diff.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(input_frame, text="(Home - Away)", font=('Arial', 8)).grid(row=0, column=2, sticky='w')
        
        # Shot difference
        ttk.Label(input_frame, text="Shot Difference:").grid(row=1, column=0, sticky='w', pady=5)
        self.shot_diff = ttk.Entry(input_frame, width=15)
        self.shot_diff.grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(input_frame, text="(Home - Away)", font=('Arial', 8)).grid(row=1, column=2, sticky='w')
        
        # Attendance
        ttk.Label(input_frame, text="Attendance:").grid(row=2, column=0, sticky='w', pady=5)
        self.attendance = ttk.Entry(input_frame, width=15)
        self.attendance.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(input_frame, text="(e.g., 50000)", font=('Arial', 8)).grid(row=2, column=2, sticky='w')
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(pady=20)
        
        # Predict button
        predict_btn = ttk.Button(
            button_frame, 
            text="🔮 Predict Outcome", 
            command=self.predict_outcome
        )
        predict_btn.pack(side='left', padx=10)
        
        # Clear button
        clear_btn = ttk.Button(
            button_frame, 
            text="🗑️ Clear", 
            command=self.clear_inputs
        )
        clear_btn.pack(side='left', padx=10)
        
        # Sample data button
        sample_btn = ttk.Button(
            button_frame, 
            text="📝 Sample Data", 
            command=self.load_sample_data
        )
        sample_btn.pack(side='left', padx=10)
        
        # Result frame
        result_frame = ttk.LabelFrame(main_frame, text="Prediction Result", padding=20)
        result_frame.pack(fill='x', pady=10)
        
        # Main result label - simplified approach
        self.result_text = tk.StringVar()
        self.result_text.set("Enter match statistics and click 'Predict Outcome'")
        
        self.result_label = tk.Label(
            result_frame, 
            textvariable=self.result_text,
            font=('Arial', 12, 'bold'),
            bg='white',
            relief='sunken',
            padx=10,
            pady=10,
            wraplength=400,
            justify='left'
        )
        self.result_label.pack(fill='x', pady=(0, 10))
        
        # Probability display frame
        self.prob_frame = ttk.Frame(result_frame)
        self.prob_frame.pack(fill='x')
        
        # Instructions
        instructions = tk.Label(
            main_frame,
            text="💡 Tip: Positive values favor home team, negative values favor away team",
            font=('Arial', 9),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        instructions.pack(pady=10)
        
    def validate_inputs(self):
        """Validate user inputs"""
        try:
            poss_diff = float(self.possession_diff.get())
            shot_diff = float(self.shot_diff.get())
            attend = int(self.attendance.get())
            
            # Basic validation
            if abs(poss_diff) > 100:
                raise ValueError("Possession difference should be between -100 and 100")
            if attend < 0:
                raise ValueError("Attendance cannot be negative")
                
            return poss_diff, shot_diff, attend
        except ValueError as e:
            if "could not convert" in str(e):
                messagebox.showerror("Input Error", "Please enter valid numbers in all fields")
            else:
                messagebox.showerror("Input Error", str(e))
            return None
            
    def predict_outcome(self):
        """Make prediction based on user inputs"""
        print("Predict button clicked!")  # Debug print
        
        if self.model is None:
            self.result_text.set("❌ No model loaded! Please run train_model.py first.")
            self.result_label.config(fg='red')
            return
        
        # Show processing message
        self.result_text.set("🔄 Processing...")
        self.result_label.config(fg='blue')
        self.root.update()  # Force immediate GUI update
        
        inputs = self.validate_inputs()
        if inputs is None:
            self.result_text.set("❌ Please enter valid inputs")
            self.result_label.config(fg='red')
            return
            
        poss_diff, shot_diff, attend = inputs
        print(f"Inputs: {poss_diff}, {shot_diff}, {attend}")  # Debug print
        
        try:
            # Create DataFrame
            features = pd.DataFrame([[poss_diff, shot_diff, attend]], 
                                  columns=['possession_difference', 'shot_difference', 'attendance'])
            
            # Scale features if scaler is available
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features)
                print("Using scaled features")
            else:
                features_scaled = features.values
                print("Using unscaled features")
            
            # Make prediction
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            
            print(f"Prediction: {prediction}")  # Debug print
            print(f"Probabilities: {probabilities}")  # Debug print
            
            # Update result label with prediction
            self.result_text.set(f"🎯 Predicted Outcome: {prediction}")
            
            # Set color based on prediction
            if prediction == 'Home Win':
                self.result_label.config(fg='#27ae60')  # Green
            elif prediction == 'Away Win':
                self.result_label.config(fg='#e74c3c')  # Red
            else:
                self.result_label.config(fg='#f39c12')  # Orange (Draw)
            
            # Clear previous probability display
            for widget in self.prob_frame.winfo_children():
                widget.destroy()
            
            # Get class names and display probabilities
            classes = self.model.classes_
            
            # Header for probabilities
            prob_header = tk.Label(
                self.prob_frame, 
                text="Confidence Levels:", 
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0'
            )
            prob_header.pack(pady=(10, 5))
            
            # Display each probability with visual bar
            for class_name, prob in zip(classes, probabilities):
                # Create frame for this probability
                prob_row_frame = tk.Frame(self.prob_frame, bg='#f0f0f0')
                prob_row_frame.pack(fill='x', pady=2)
                
                # Probability text
                prob_text = f"{class_name}: {prob:.1%}"
                prob_label = tk.Label(
                    prob_row_frame, 
                    text=prob_text, 
                    font=('Arial', 9),
                    bg='#f0f0f0',
                    width=20,
                    anchor='w'
                )
                prob_label.pack(side='left', padx=(10, 5))
                
                # Visual bar
                bar_frame = tk.Frame(prob_row_frame, bg='#f0f0f0')
                bar_frame.pack(side='left', fill='x', expand=True, padx=(0, 10))
                
                # Calculate bar width (max 200 pixels)
                bar_width = max(1, int(prob * 200))
                
                # Color based on outcome
                if class_name == 'Home Win':
                    bar_color = '#27ae60'
                elif class_name == 'Away Win':
                    bar_color = '#e74c3c'
                else:
                    bar_color = '#f39c12'
                
                # Create the bar using a Label with background color
                bar = tk.Label(
                    bar_frame,
                    text="",
                    bg=bar_color,
                    width=bar_width // 8,  # Approximate character width
                    height=1
                )
                bar.pack(side='left')
            
            # Force GUI update
            self.root.update_idletasks()
            self.root.update()
            
            print("GUI updated successfully")  # Debug print
                
        except Exception as e:
            print(f"Error during prediction: {e}")  # Debug print
            self.result_text.set(f"❌ Error: {str(e)}")
            self.result_label.config(fg='red')
            import traceback
            traceback.print_exc()
    
    def clear_inputs(self):
        """Clear all input fields"""
        self.possession_diff.delete(0, tk.END)
        self.shot_diff.delete(0, tk.END)
        self.attendance.delete(0, tk.END)
        
        # Reset result display
        self.result_text.set("Enter match statistics and click 'Predict Outcome'")
        self.result_label.config(fg='black')
        
        # Clear probabilities
        for widget in self.prob_frame.winfo_children():
            widget.destroy()
        
        # Force GUI refresh
        self.root.update_idletasks()
    
    def load_sample_data(self):
        """Load sample data for testing"""
        self.clear_inputs()
        
        self.possession_diff.insert(0, "15.5")
        self.shot_diff.insert(0, "8")
        self.attendance.insert(0, "45000")

if __name__ == "__main__":
    root = tk.Tk()
    app = PremierLeaguePredictor(root)
    root.mainloop()