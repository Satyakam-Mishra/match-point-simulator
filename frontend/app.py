import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import os
import sys

# Add the src directory to the path to import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import LIVE_POINT_WIN_PROB_DATASET

# ============= MAPPINGS =============
# Shot Type Mapping
SHOT_TYPE_MAPPING = {
    'forehand': 0,
    'backhand': 1,
    'slice': 2,
    'return': 3,
    'volley': 4,
    'overhead': 5,
    'lob': 6,
    'smash': 7,
    'chip': 8,
    'slice volley': 9,
    'put away': 10,
    'dropshot': 11,
    'approach': 12,
    'underspin': 13,
    'other': 14
}

# Reverse mapping for display
SHOT_TYPE_REVERSE = {v: k for k, v in SHOT_TYPE_MAPPING.items()}

# Direction Mapping (based on court directions)
DIRECTION_MAPPING = {
    'crosscourt': 1,
    'down the line': 2,
    'middle': 3
}

DIRECTION_REVERSE = {v: k for k, v in DIRECTION_MAPPING.items()}

# Serve Location Mapping
SERVE_LOCATION_MAPPING = {
    'Wide': 4,
    'Body/Line': 5,
    'T': 6
}

SERVE_LOCATION_REVERSE = {v: k for k, v in SERVE_LOCATION_MAPPING.items()}

# Position Information Mapping
POSITION_MAPPING = {
    'Baseline (=)': 0,
    'Net (+)': 1,
    'Behind Baseline (-)': 2
}

POSITION_REVERSE = {v: k for k, v in POSITION_MAPPING.items()}

# ============= HELPER FUNCTIONS =============
def point_parser(point):
    """Parse tennis point score into numeric representation"""
    if isinstance(point, str):
        if point == '0':
            return 0
        elif point == '15':
            return 1
        elif point == '30':
            return 2
        elif point == '40':
            return 3
        elif point == 'AD':
            return 4
    return None

def tiebreaker_point_parser(point):
    """Parse tiebreaker points (0-6+)"""
    try:
        return int(point)
    except:
        return -1

# Page configuration
st.set_page_config(
    page_title="Live Point Win Probability Predictor",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎾 Live Point Win Probability Predictor")
st.markdown("---")

# Load and train model (cached for performance)
@st.cache_resource
def load_or_train_model():
    """Load the dataset and train a Random Forest model if not already cached"""
    try:
        # Try to load the dataset
        df = pd.read_csv(LIVE_POINT_WIN_PROB_DATASET)
        
        # Separate features and target
        y = df["point_winner"]
        X = df.drop(columns=["point_winner"])
        
        # Fill missing values with -1
        X.fillna(-1, inplace=True)
        
        # Train a Random Forest model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=18,
            n_jobs=-1,
            random_state=42
        )
        model.fit(X, y)
        
        return model, X.columns.tolist()
    
    except FileNotFoundError:
        st.error(f"Dataset not found at {LIVE_POINT_WIN_PROB_DATASET}")
        st.info("Please ensure the dataset is generated before using this app.")
        return None, []

# Load model
model, feature_columns = load_or_train_model()

if model is None:
    st.stop()

# Create tabs for better organization
tab1, tab2 = st.tabs(["📊 Make Prediction", "ℹ️ Feature Guide"])

with tab1:
    st.subheader("Input Match and Point Information")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### **Player Information**")
        pl_0_hand = st.selectbox("Player 0 Hand", ["Right (R)", "Left (L)"], key="pl_0_hand")
        pl_0_hand = 0 if pl_0_hand == "Right (R)" else 1
        
        pl_1_hand = st.selectbox("Player 1 Hand", ["Right (R)", "Left (L)"], key="pl_1_hand")
        pl_1_hand = 0 if pl_1_hand == "Right (R)" else 1
    
    with col2:
        st.markdown("### **Match Information**")
        gender = st.selectbox("Gender", ["Men (M)", "Women (W)"], key="gender")
        gender = 0 if gender == "Men (M)" else 1
        
        best_of = st.number_input("Best Of", min_value=3, max_value=5, value=3, step=2, key="best_of")
    
    st.markdown("---")
    
    # Match Score Section
    st.markdown("### **Current Match Score**")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        set0 = st.number_input("Set 0", min_value=0, max_value = 2, value=0, key="set0")
    with col2:
        set1 = st.number_input("Set 1", min_value=0, max_value = 2, value=0, key="set1")
    with col3:
        game0 = st.number_input("Game 0", min_value=0, max_value = 5, value=0, key="game0")
    with col4:
        game1 = st.number_input("Game 1", min_value=0, max_value = 5, value=0, key="game1")
    
    st.markdown("---")
    
    # Surface Selection
    st.markdown("### **Surface**")
    surface = st.selectbox("Surface Type", ["Hard", "Clay", "Grass"], key="surface")
    surface_Hard = 1 if surface == "Hard" else 0
    surface_Clay = 1 if surface == "Clay" else 0
    surface_Grass = 1 if surface == "Grass" else 0
    
    st.markdown("---")
    
    # Current Point Information
    st.markdown("### **Current Point Status**")
    col1, col2, col3 = st.columns(3)
    
    # Tiebreaker checkbox first to determine point input type
    is_tiebreaker = st.checkbox("Is Tiebreaker", value=False, key="is_tiebreaker")
    is_tiebreaker_int = int(is_tiebreaker)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if is_tiebreaker:
            player_0_point = st.number_input("Player 0 Points (Tiebreak)", min_value=0, max_value=20, value=0, key="player_0_point")
        else:
            player_0_point_str = st.selectbox("Player 0 Points", ["0", "15", "30", "40", "AD"], key="player_0_point")
            player_0_point = point_parser(player_0_point_str)
    
    with col2:
        if is_tiebreaker:
            player_1_point = st.number_input("Player 1 Points (Tiebreak)", min_value=0, max_value=20, value=0, key="player_1_point")
        else:
            player_1_point_str = st.selectbox("Player 1 Points", ["0", "15", "30", "40", "AD"], key="player_1_point")
            player_1_point = point_parser(player_1_point_str)
    
    with col3:
        svr = st.selectbox("Serve/Return", ["Serve (0)", "Return (1)"], key="svr")
        svr = 0 if svr == "Serve (0)" else 1
    
    st.markdown("---")
    
    # Rally Information
    st.markdown("### **Rally Information**")
    col1, = st.columns(1)
    with col1:
        first_serve_in = st.selectbox("First Serve In", ["Yes (1)", "No (0)"], key="first_serve_in")
        first_serve_in = 1 if first_serve_in == "Yes (1)" else 0
    
    
    st.markdown("---")
    
    # Serve Shot Information
    st.markdown("### **Serve Shot**")
    col1, col2, col3 = st.columns(3)
    with col1:
        serve_location_option = st.selectbox("Serve Location", ["None"] + list(SERVE_LOCATION_MAPPING.keys()), key="serve_location")
        serve_location = SERVE_LOCATION_MAPPING.get(serve_location_option, -1) if serve_location_option != "None" else -1
    with col2:
        serve_shank_info = st.selectbox("Serve Shank Info", ["None", "No (0)", "Yes (1)"], key="serve_shank_info")
        serve_shank_info = -1 if serve_shank_info == "None" else 0 if serve_shank_info == "No (0)" else 1
    with col3:
        serve_position_info = st.selectbox("Serve Position", ["None"] + list(POSITION_MAPPING.keys()), key="serve_position_info")
        serve_position_info = POSITION_MAPPING.get(serve_position_info, -1) if serve_position_info != "None" else -1
    
    st.markdown("---")
    
    # Return Shot Information
    st.markdown("### **Return Shot**")
    col1, col2, col3 = st.columns(3)
    with col1:
        return_shot_type_option = st.selectbox("Return Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="return_shot_type")
        return_shot_type = SHOT_TYPE_MAPPING.get(return_shot_type_option, -1) if return_shot_type_option != "None" else -1
    with col2:
        return_direction_option = st.selectbox("Return Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="return_direction")
        return_direction = DIRECTION_MAPPING.get(return_direction_option, -1) if return_direction_option != "None" else -1
    with col3:
        return_depth = st.selectbox("Return Depth", ["None", "Shallow", "Medium", "Deep"], key="return_depth")
        return_depth = -1 if return_depth == "None" else 1 if return_depth == "Shallow" else 2 if return_depth == "Medium" else 3
    
    col1, col2 = st.columns(2)
    with col1:
        return_shank_info = st.selectbox("Return Shank Info", ["None", "No (0)", "Yes (1)"], key="return_shank_info")
        return_shank_info = -1 if return_shank_info == "None" else 0 if return_shank_info == "No (0)" else 1
    with col2:
        return_position_info = st.selectbox("Return Position", ["None"] + list(POSITION_MAPPING.keys()), key="return_position_info")
        return_position_info = POSITION_MAPPING.get(return_position_info, -1) if return_position_info != "None" else -1
    
    st.markdown("---")
    
    # Serve Plus One Information
    st.markdown("### **Serve Plus One Shot**")
    col1, col2, col3 = st.columns(3)
    with col1:
        serve_plus_one_shot_type_option = st.selectbox("Serve+1 Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="serve_plus_one_shot_type")
        serve_plus_one_shot_type = SHOT_TYPE_MAPPING.get(serve_plus_one_shot_type_option, -1) if serve_plus_one_shot_type_option != "None" else -1
    with col2:
        serve_plus_one_direction_option = st.selectbox("Serve+1 Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="serve_plus_one_direction")
        serve_plus_one_direction = DIRECTION_MAPPING.get(serve_plus_one_direction_option, -1) if serve_plus_one_direction_option != "None" else -1
    with col3:
        serve_plus_one_depth = st.selectbox("Serve+1 Depth", ["None", "Shallow", "Medium", "Deep"], key="serve_plus_one_depth")
        serve_plus_one_depth = -1 if serve_plus_one_depth == "None" else 1 if serve_plus_one_depth == "Shallow" else 2 if serve_plus_one_depth == "Medium" else 3
    
    col1, col2 = st.columns(2)
    with col1:
        serve_plus_one_shank_info = st.selectbox("Serve+1 Shank Info", ["None", "No (0)", "Yes (1)"], key="serve_plus_one_shank_info")
        serve_plus_one_shank_info = -1 if serve_plus_one_shank_info == "None" else 0 if serve_plus_one_shank_info == "No (0)" else 1
    with col2:
        serve_plus_one_position_info = st.selectbox("Serve+1 Position", ["None"] + list(POSITION_MAPPING.keys()), key="serve_plus_one_position_info")
        serve_plus_one_position_info = POSITION_MAPPING.get(serve_plus_one_position_info, -1) if serve_plus_one_position_info != "None" else -1
    
    st.markdown("---")
    
    # Shot 4 Information
    st.markdown("### **Shot 4**")
    col1, col2, col3 = st.columns(3)
    with col1:
        shot_4_shot_type_option = st.selectbox("Shot 4 Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="shot_4_shot_type")
        shot_4_shot_type = SHOT_TYPE_MAPPING.get(shot_4_shot_type_option, -1) if shot_4_shot_type_option != "None" else -1
    with col2:
        shot_4_shot_direction_option = st.selectbox("Shot 4 Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="shot_4_shot_direction")
        shot_4_shot_direction = DIRECTION_MAPPING.get(shot_4_shot_direction_option, -1) if shot_4_shot_direction_option != "None" else -1
    with col3:
        shot_4_shot_depth = st.selectbox("Shot 4 Depth", ["None", "Shallow", "Medium", "Deep"], key="shot_4_shot_depth")
        shot_4_shot_depth = -1 if shot_4_shot_depth == "None" else 1 if shot_4_shot_depth == "Shallow" else 2 if shot_4_shot_depth == "Medium" else 3
    
    col1, col2 = st.columns(2)
    with col1:
        shot_4_shank_info = st.selectbox("Shot 4 Shank Info", ["None", "No (0)", "Yes (1)"], key="shot_4_shank_info")
        shot_4_shank_info = -1 if shot_4_shank_info == "None" else 0 if shot_4_shank_info == "No (0)" else 1
    with col2:
        shot_4_position_info = st.selectbox("Shot 4 Position", ["None"] + list(POSITION_MAPPING.keys()), key="shot_4_position_info")
        shot_4_position_info = POSITION_MAPPING.get(shot_4_position_info, -1) if shot_4_position_info != "None" else -1
    
    st.markdown("---")
    
    # Shot 5 Information
    st.markdown("### **Shot 5**")
    col1, col2, col3 = st.columns(3)
    with col1:
        shot_5_shot_type_option = st.selectbox("Shot 5 Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="shot_5_shot_type")
        shot_5_shot_type = SHOT_TYPE_MAPPING.get(shot_5_shot_type_option, -1) if shot_5_shot_type_option != "None" else -1
    with col2:
        shot_5_shot_direction_option = st.selectbox("Shot 5 Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="shot_5_shot_direction")
        shot_5_shot_direction = DIRECTION_MAPPING.get(shot_5_shot_direction_option, -1) if shot_5_shot_direction_option != "None" else -1
    with col3:
        shot_5_shot_depth = st.selectbox("Shot 5 Depth", ["None", "Shallow", "Medium", "Deep"], key="shot_5_shot_depth")
        shot_5_shot_depth = -1 if shot_5_shot_depth == "None" else 1 if shot_5_shot_depth == "Shallow" else 2 if shot_5_shot_depth == "Medium" else 3
    
    col1, col2 = st.columns(2)
    with col1:
        shot_5_shank_info = st.selectbox("Shot 5 Shank Info", ["None", "No (0)", "Yes (1)"], key="shot_5_shank_info")
        shot_5_shank_info = -1 if shot_5_shank_info == "None" else 0 if shot_5_shank_info == "No (0)" else 1
    with col2:
        shot_5_position_info = st.selectbox("Shot 5 Position", ["None"] + list(POSITION_MAPPING.keys()), key="shot_5_position_info")
        shot_5_position_info = POSITION_MAPPING.get(shot_5_position_info, -1) if shot_5_position_info != "None" else -1
    
    st.markdown("---")
    
    # Shot 6 Information
    st.markdown("### **Shot 6**")
    col1, col2, col3 = st.columns(3)
    with col1:
        shot_6_shot_type_option = st.selectbox("Shot 6 Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="shot_6_shot_type")
        shot_6_shot_type = SHOT_TYPE_MAPPING.get(shot_6_shot_type_option, -1) if shot_6_shot_type_option != "None" else -1
    with col2:
        shot_6_shot_direction_option = st.selectbox("Shot 6 Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="shot_6_shot_direction")
        shot_6_shot_direction = DIRECTION_MAPPING.get(shot_6_shot_direction_option, -1) if shot_6_shot_direction_option != "None" else -1
    with col3:
        shot_6_shot_depth = st.selectbox("Shot 6 Depth", ["None", "Shallow", "Medium", "Deep"], key="shot_6_shot_depth")
        shot_6_shot_depth = -1 if shot_6_shot_depth == "None" else 1 if shot_6_shot_depth == "Shallow" else 2 if shot_6_shot_depth == "Medium" else 3
    
    col1, col2 = st.columns(2)
    with col1:
        shot_6_shank_info = st.selectbox("Shot 6 Shank Info", ["None", "No (0)", "Yes (1)"], key="shot_6_shank_info")
        shot_6_shank_info = -1 if shot_6_shank_info == "None" else 0 if shot_6_shank_info == "No (0)" else 1
    with col2:
        shot_6_position_info = st.selectbox("Shot 6 Position", ["None"] + list(POSITION_MAPPING.keys()), key="shot_6_position_info")
        shot_6_position_info = POSITION_MAPPING.get(shot_6_position_info, -1) if shot_6_position_info != "None" else -1
    
    st.markdown("---")
    
    # Prediction Button
    if st.button("🔮 Predict Point Winner", key="predict_button", use_container_width=True):
        # Create input dataframe with all features
        input_data = {
            'pl_0_hand': pl_0_hand,
            'pl_1_hand': pl_1_hand,
            'best_of': best_of,
            'gender': gender,
            'set0': set0,
            'set1': set1,
            'game0': game0,
            'game1': game1,
            'surface_Hard': surface_Hard,
            'surface_Clay': surface_Clay,
            'surface_Grass': surface_Grass,
            'player_0_point': player_0_point,
            'player_1_point': player_1_point,
            'svr': svr,
            'is_tiebreaker': is_tiebreaker_int,
            'first_serve_in': first_serve_in,
            'serve_location': serve_location,
            'serve_shank_info': serve_shank_info,
            'serve_position_info': serve_position_info,
            'return_shot_type': return_shot_type,
            'return_direction': return_direction,
            'return_depth': return_depth,
            'return_shank_info': return_shank_info,
            'return_position_info': return_position_info,
            'serve_plus_one_shot_type': serve_plus_one_shot_type,
            'serve_plus_one_direction': serve_plus_one_direction,
            'serve_plus_one_depth': serve_plus_one_depth,
            'serve_plus_one_shank_info': serve_plus_one_shank_info,
            'serve_plus_one_position_info': serve_plus_one_position_info,
            'shot_4_shot_type': shot_4_shot_type,
            'shot_4_shot_direction': shot_4_shot_direction,
            'shot_4_shot_depth': shot_4_shot_depth,
            'shot_4_shank_info': shot_4_shank_info,
            'shot_4_position_info': shot_4_position_info,
            'shot_5_shot_type': shot_5_shot_type,
            'shot_5_shot_direction': shot_5_shot_direction,
            'shot_5_shot_depth': shot_5_shot_depth,
            'shot_5_shank_info': shot_5_shank_info,
            'shot_5_position_info': shot_5_position_info,
            'shot_6_shot_type': shot_6_shot_type,
            'shot_6_shot_direction': shot_6_shot_direction,
            'shot_6_shot_depth': shot_6_shot_depth,
            'shot_6_shank_info': shot_6_shank_info,
            'shot_6_position_info': shot_6_position_info,
        }
        
        # Create dataframe and ensure correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required features are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = -1
        
        # Select only the required features in the correct order
        input_df = input_df[feature_columns]
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            
            st.markdown("---")
            st.markdown("## 🎯 **Prediction Results**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                winner = "Player 0" if prediction == 0 else "Player 1"
                st.metric("Predicted Point Winner", winner, delta=None)
            
            with col2:
                confidence = probabilities[int(prediction)] * 100
                st.metric("Confidence", f"{confidence:.2f}%", delta=None)
            
            # Display probability distribution
            st.markdown("### **Win Probability Distribution**")
            prob_data = pd.DataFrame({
                'Player': ['Player 0', 'Player 1'],
                'Win Probability': [probabilities[0] * 100, probabilities[1] * 100]
            })
            
            st.bar_chart(prob_data.set_index('Player'))
            
            # Display detailed prediction info
            st.markdown("### **Detailed Probabilities**")
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"**Player 0 Win Probability:** {probabilities[0]*100:.2f}%")
            with col2:
                st.info(f"**Player 1 Win Probability:** {probabilities[1]*100:.2f}%")
        
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")

with tab2:
    st.markdown("""
    ### **Feature Guide**
    
    #### **Player Information**
    - **Player Hand**: Right (R=0) or Left (L=1)
    
    #### **Match Information**
    - **Gender**: Men (M=0) or Women (W=1)
    - **Best Of**: Match format (e.g., Best of 3, Best of 5)
    - **Surface**: Hard, Clay, or Grass court
    
    #### **Match Score**
    - **Set 0/1**: Current set scores
    - **Game 0/1**: Current game scores
    
    #### **Point Status**
    - **Is Tiebreaker**: Check this if playing a tiebreaker
    - **Normal Game Points**: 0, 15, 30, 40, or AD (auto-mapped to 0, 1, 2, 3, 4)
    - **Tiebreaker Points**: Any number from 0 to 20+
    - **Serve/Return**: 0 for serve, 1 for return
    
    #### **Rally Information**
    - **First Serve In**: Whether first serve was in (Yes/No)
    
    #### **Serve Shot Information**
    - **Serve Location**: Wide (4), Body/Line (5), or T (6)
    - **Serve Shank Info**: Whether shot was a shank (0=No, 1=Yes)
    - **Serve Position**: Baseline (0), Net (1), or Behind Baseline (2)
    
    #### **Return Shot & All Subsequent Shots**
    - **Shot Type**: 
        - forehand (0), backhand (1), slice (2), return (3), volley (4)
        - overhead (5), lob (6), smash (7), chip (8), slice volley (9)
        - put away (10), dropshot (11), approach (12), underspin (13), other (14)
    - **Direction**: Crosscourt (1), Down the Line (2), Middle (3)
    - **Depth**: Shallow (1), Medium (2), Deep (3)
    - **Shank Info**: Whether shot was a shank (0=No, 1=Yes)
    - **Position**: Baseline (0), Net (1), or Behind Baseline (2)
    
    #### **Using None for Missing Values**
    If a particular shot hasn't happened yet in the rally, select "None" as the default value.
    This will be converted to -1 for the model input.
    """)

# Footer with attribution
st.markdown("---")
st.caption("Data provided by Jeff Sackmann's Match Charting Project (CC BY-NC-SA 4.0).")