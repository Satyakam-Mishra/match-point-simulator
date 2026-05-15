import streamlit as st
import pandas as pd
import numpy as np
import os
import sys
import pickle

# Add the src directory to the path to import modules FIRST
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import RANDOM_FOREST_LPWP_MODEL, FEATURE_COLUMNS_LPWP, LIVE_POINT_WIN_PROB_DATASET, RANDOM_FOREST_NBS_MODEL, FEATURE_COLUMNS_NBS, NEXT_BEST_SHOT_DATASET

# ============= PAGE CONFIGURATION =============
st.set_page_config(
    page_title="Tennis Analytics Platform",
    page_icon="🎾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

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

# Shank Info Mapping
SHANK_INFO_MAPPING = {
    'No': 0.0,
    'Yes': 1.0
}

SHANK_INFO_REVERSE = {v: k for k, v in SHANK_INFO_MAPPING.items()}

# Depth Mapping for opponent's shot
DEPTH_MAPPING = {
    'Shallow': 1,
    'Medium': 2,
    'Deep': 3
}

DEPTH_REVERSE = {v: k for k, v in DEPTH_MAPPING.items()}

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

# ============= HOME PAGE =============
def show_home_page():
    """Display the home page with project description and navigation"""
    st.title("🎾 Tennis Analytics Platform")
    st.markdown("---")
    
    st.markdown("""
    ## Welcome to the Match Point Simulator
    
    This comprehensive platform provides advanced analytics and predictions for tennis matches, 
    leveraging machine learning models trained on data from the **Jeff Sackmann's Match Charting Project**.
    
    ### 📊 Platform Overview
    
    Our platform offers several powerful tools designed for tennis enthusiasts, analysts, and coaches:
    
    - **Live Point Win Probability**: Get real-time predictions on the probability of the serving player winning the current point
    - **Match Analysis**: Deep dive into match statistics and trends
    - **Player Statistics**: Analyze individual player performance metrics
    - **Model Performance**: Explore model accuracy and performance metrics
    
    ### 🎯 Key Features
    
    - **Machine Learning Powered**: Uses Random Forest classification trained on thousands of professional matches
    - **Real-Time Predictions**: Input current match state to get instant win probability predictions
    - **Comprehensive Data**: Based on detailed point-by-point data including shot types, directions, and court positions
    - **Tennis-Specific Insights**: Considers serve locations, shot types, rally length, and more
    
    ### 🏆 Data Source
    
    Data provided by **Jeff Sackmann's Match Charting Project** (CC BY-NC-SA 4.0)
    
    ---
    
    ### 🚀 Get Started
    
    Select a page below to begin exploring:
    """)
    
    # Create 4 navigation buttons
    st.markdown("## Navigate to:")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #00D9FF; border-radius: 10px; text-align: center; margin: 10px 0;">
        <h3>📈 Live Point Win Probability</h3>
        <p>Predict the probability of the server winning the current point based on match and rally information.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Go to Live Predictor", key="btn_predictor", use_container_width=True):
            st.session_state.current_page = 'predictor'
            st.rerun()
    
    with col2:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #FF6B6B; border-radius: 10px; text-align: center; margin: 10px 0;">
        <h3>📊 Match Analysis</h3>
        <p>Analyze match statistics, patterns, and trends from professional tennis matches.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("📊 Go to Match Analysis", key="btn_analysis", use_container_width=True):
            st.session_state.current_page = 'analysis'
            st.rerun()
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #FFA500; border-radius: 10px; text-align: center; margin: 10px 0;">
        <h3>🎯 Next Best Shot Predictor</h3>
        <p>Predict the best next shot given the opponent's previous shot and match context.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🎯 Go to NBS Predictor", key="btn_nbs", use_container_width=True):
            st.session_state.current_page = 'nbs_predictor'
            st.rerun()
    
    with col4:
        st.markdown("""
        <div style="padding: 20px; border: 2px solid #95E1D3; border-radius: 10px; text-align: center; margin: 10px 0;">
        <h3>👤 Player Statistics</h3>
        <p>Explore detailed performance metrics and statistics for individual players.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("👤 Go to Player Stats", key="btn_player", use_container_width=True):
            st.session_state.current_page = 'player'
            st.rerun()

# ============= PREDICTOR PAGE =============
def show_predictor_page():
    """Display the live point win probability predictor"""
    if st.button("← Back to Home", key="back_home_predictor"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.title("🎾 Live Point Win Probability Predictor")
    st.markdown("---")
    
    # Load model (cached for performance)
    @st.cache_resource
    def load_model():
        """Load pre-trained model from pickle files"""
        
        # Construct absolute paths from config
        model_path = RANDOM_FOREST_LPWP_MODEL
        features_path = FEATURE_COLUMNS_LPWP
        
        # Try to load existing model
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
            st.success("✅ Model loaded successfully!")
            return model, feature_columns
        except FileNotFoundError:
            st.error("❌ Model files not found!")
            st.info("Please run `python src/lpwp_model.py` from the project root to generate the model files.")
            st.info(f"Expected files:")
            st.info(f"  - {model_path}")
            st.info(f"  - {features_path}")
            return None, []
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, []
    
    # Load model
    model, feature_columns = load_model()
    
    if model is None:
        st.stop()
    
    # Create tabs for input and guide
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

# ============= OTHER PAGES =============
def show_analysis_page():
    """Display the match analysis page"""
    if st.button("← Back to Home", key="back_home_analysis"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.title("📊 Match Analysis")
    st.markdown("---")
    st.info("Match Analysis page coming soon! This will provide detailed statistics and trends from professional tennis matches.")

def show_player_page():
    """Display the player statistics page"""
    if st.button("← Back to Home", key="back_home_player"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.title("👤 Player Statistics")
    st.markdown("---")
    st.info("Player Statistics page coming soon! Explore detailed performance metrics for individual players.")

def show_nbs_predictor_page():
    """Display the next best shot predictor"""
    if st.button("← Back to Home", key="back_home_nbs"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.title("🎯 Next Best Shot Predictor")
    st.markdown("---")
    
    # Load model (cached for performance)
    @st.cache_resource
    def load_nbs_model():
        """Load pre-trained NBS model from pickle files"""
        model_path = RANDOM_FOREST_NBS_MODEL
        features_path = FEATURE_COLUMNS_NBS
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            with open(features_path, 'rb') as f:
                feature_columns = pickle.load(f)
            st.success("✅ NBS Model loaded successfully!")
            return model, feature_columns
        except FileNotFoundError:
            st.error("❌ NBS Model files not found!")
            st.info("Please run `python src/nbs_model_bayesian_opt_04.py` from the project root to generate the model files.")
            st.info(f"Expected files:")
            st.info(f"  - {model_path}")
            st.info(f"  - {features_path}")
            return None, []
        except Exception as e:
            st.error(f"❌ Error loading model: {e}")
            return None, []
    
    model, feature_columns = load_nbs_model()
    
    if model is None:
        st.stop()
    
    tab1, tab2 = st.tabs(["📊 Make Prediction", "ℹ️ Feature Guide"])
    
    with tab1:
        st.subheader("Predict the Best Next Shot")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### **Player Information**")
            pl_0_hand = st.selectbox("Your Hand", ["Right (R)", "Left (L)"], key="nbs_pl_0_hand")
            pl_0_hand = 0 if pl_0_hand == "Right (R)" else 1
            
            pl_1_hand = st.selectbox("Opponent Hand", ["Right (R)", "Left (L)"], key="nbs_pl_1_hand")
            pl_1_hand = 0 if pl_1_hand == "Right (R)" else 1
        
        with col2:
            st.markdown("### **Match Information**")
            gender = st.selectbox("Gender", ["Men (M)", "Women (W)"], key="nbs_gender")
            gender = 0 if gender == "Men (M)" else 1
            
            best_of = st.number_input("Best Of", min_value=3, max_value=5, value=3, step=2, key="nbs_best_of")
        
        st.markdown("---")
        
        st.markdown("### **Current Match Score**")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            player_0_point = st.selectbox("Your Points", ["0", "15", "30", "40", "AD"], key="nbs_player_0_point", index=0)
            player_0_point = point_parser(player_0_point)
        with col2:
            player_1_point = st.selectbox("Opponent Points", ["0", "15", "30", "40", "AD"], key="nbs_player_1_point", index=0)
            player_1_point = point_parser(player_1_point)
        with col3:
            svr = st.selectbox("Serve/Return", ["Serve (0)", "Return (1)"], key="nbs_svr")
            svr = 0 if svr == "Serve (0)" else 1
        with col4:
            surface = st.selectbox("Surface", ["Hard", "Clay", "Grass"], key="nbs_surface")
            surface_Hard = 1 if surface == "Hard" else 0
            surface_Clay = 1 if surface == "Clay" else 0
            surface_Grass = 1 if surface == "Grass" else 0
        
        st.markdown("---")
        
        st.markdown("### **Current Rally Status**")
        col1, col2, col3 = st.columns(3)
        with col1:
            is_tiebreaker = st.checkbox("Is Tiebreaker", value=False, key="nbs_is_tiebreaker")
            is_tiebreaker = float(is_tiebreaker)
        with col2:
            first_serve_in = st.selectbox("First Serve In", ["Yes (1)", "No (0)"], key="nbs_first_serve_in")
            first_serve_in = 1.0 if first_serve_in == "Yes (1)" else 0.0
        with col3:
            shot_number = st.number_input("Shot Number (Your Current Shot)", min_value=1, max_value=20, value=2, key="nbs_shot_number")
        
        st.markdown("---")
        
        st.markdown("### **Opponent's Previous Shot**")
        st.info("Describe the shot your opponent just hit")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            prev_shot_type_option = st.selectbox("Opponent Shot Type", ["None"] + list(SHOT_TYPE_MAPPING.keys()), key="nbs_prev_shot_type")
            prev_shot_type = SHOT_TYPE_MAPPING.get(prev_shot_type_option, -1) if prev_shot_type_option != "None" else -1
        
        with col2:
            prev_shot_direction_option = st.selectbox("Opponent Shot Direction", ["None"] + list(DIRECTION_MAPPING.keys()), key="nbs_prev_shot_direction")
            prev_shot_direction = DIRECTION_MAPPING.get(prev_shot_direction_option, -1) if prev_shot_direction_option != "None" else -1
        
        with col3:
            prev_shot_depth_option = st.selectbox("Opponent Shot Depth", ["None", "Shallow", "Medium", "Deep"], key="nbs_prev_shot_depth")
            prev_shot_depth = -1 if prev_shot_depth_option == "None" else DEPTH_MAPPING.get(prev_shot_depth_option, -1)
        
        col1, col2 = st.columns(2)
        with col1:
            prev_shot_shank_info_option = st.selectbox("Opponent Shot - Shank", ["None", "No", "Yes"], key="nbs_prev_shot_shank")
            prev_shot_shank_info = -1 if prev_shot_shank_info_option == "None" else SHANK_INFO_MAPPING.get(prev_shot_shank_info_option, -1)
        
        with col2:
            prev_shot_position_info_option = st.selectbox("Opponent Position", ["None"] + list(POSITION_MAPPING.keys()), key="nbs_prev_shot_position")
            prev_shot_position_info = POSITION_MAPPING.get(prev_shot_position_info_option, -1) if prev_shot_position_info_option != "None" else -1
        
        st.markdown("---")
        
        if st.button("🎯 Predict Best Next Shot", key="nbs_predict_button", use_container_width=True):
            # Create input dataframe with all features
            input_data = {
                'pl_0_hand': pl_0_hand,
                'pl_1_hand': pl_1_hand,
                'best_of': best_of,
                'gender': gender,
                'point_number': 1,
                'svr': svr,
                'surface_Clay': surface_Clay,
                'surface_Grass': surface_Grass,
                'surface_Hard': surface_Hard,
                'player_0_point': player_0_point,
                'player_1_point': player_1_point,
                'is_tiebreaker': is_tiebreaker,
                'first_serve_in': first_serve_in,
                'shot_number': shot_number,
                'prev_shot_type': prev_shot_type,
                'prev_shot_direction': prev_shot_direction,
                'prev_shot_depth': prev_shot_depth,
                'prev_shot_shank_info': prev_shot_shank_info,
                'prev_shot_position_info': prev_shot_position_info,
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Ensure all required features are present
            for col in feature_columns:
                if col not in input_df.columns:
                    input_df[col] = -1
            
            input_df = input_df[feature_columns]
            
            try:
                predictions = model.predict(input_df)[0]
                probabilities = model.predict_proba(input_df)[0]
                
                st.markdown("---")
                st.markdown("## 🎯 **Recommended Next Shot**")
                
                # Parse y_string to display shot information
                shot_parts = predictions.split('_')
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if shot_parts[0] != 'NA':
                        shot_type_num = int(shot_parts[0])
                        shot_type_name = SHOT_TYPE_REVERSE.get(shot_type_num, 'Unknown')
                        st.info(f"**Recommended Shot Type**\n{shot_type_name.title()}")
                
                with col2:
                    if shot_parts[1] != 'NA':
                        shot_dir_num = int(shot_parts[1])
                        shot_dir_name = DIRECTION_REVERSE.get(shot_dir_num, 'Unknown')
                        st.info(f"**Recommended Direction**\n{shot_dir_name.title()}")
                
                with col3:
                    if shot_parts[2] != 'NA':
                        shot_depth_num = int(shot_parts[2])
                        shot_depth_name = DEPTH_REVERSE.get(shot_depth_num, 'Unknown')
                        st.info(f"**Recommended Depth**\n{shot_depth_name.title()}")
                
                st.markdown("---")
                
                # Display all possible shots with probabilities
                st.markdown("### **Top Recommended Shots**")
                
                # Get unique predictions and their probabilities
                unique_preds = model.classes_
                pred_data = pd.DataFrame({
                    'Shot Sequence': unique_preds,
                    'Probability (%)': probabilities * 100
                }).sort_values('Probability (%)', ascending=False).head(10)
                
                st.dataframe(pred_data, use_container_width=True)
                
                st.markdown("### **Shot Sequence Breakdown**")
                st.markdown(f"**Full Prediction:** `{predictions}`")
                st.markdown(f"**Confidence:** {max(probabilities) * 100:.2f}%")
            
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    
    with tab2:
        st.markdown("""
        ### **How to Use the Next Best Shot Predictor**
        
        This tool recommends the optimal next shot based on your opponent's previous shot and match context.
        
        #### **Input Guide**
        - **Your Hand & Opponent Hand**: Handedness of each player
        - **Gender**: Player classification (Men or Women)
        - **Best Of**: Match format (3 or 5 sets)
        - **Match Score**: Current points in the game
        - **Serve/Return**: Whether you're serving (0) or returning (1)
        - **Surface**: Court surface (Hard, Clay, or Grass)
        - **Is Tiebreaker**: Check if playing a tiebreaker
        - **First Serve In**: Whether first serve was in
        - **Shot Number**: Your current shot number in the rally
        
        #### **Opponent's Previous Shot**
        - **Shot Type**: What type of shot they hit (forehand, backhand, volley, etc.)
        - **Direction**: Where they aimed (crosscourt, down the line, middle)
        - **Depth**: How deep the shot landed (shallow, medium, deep)
        - **Shank**: Whether it was a mishit
        - **Position**: Where they were standing (baseline, net, behind baseline)
        
        #### **Output**
        The model will recommend the best shot type, direction, and depth for your next shot,
        along with a confidence score and alternative options.
        """)

def show_model_page():
    """Display the model performance page"""
    if st.button("← Back to Home", key="back_home_model"):
        st.session_state.current_page = 'home'
        st.rerun()
    
    st.title("🤖 Model Performance")
    st.markdown("---")
    st.info("Model Performance page coming soon! Understand model accuracy and feature importance metrics.")

# ============= MAIN APP LOGIC =============
if st.session_state.current_page == 'home':
    show_home_page()
elif st.session_state.current_page == 'predictor':
    show_predictor_page()
elif st.session_state.current_page == 'nbs_predictor':
    show_nbs_predictor_page()
elif st.session_state.current_page == 'analysis':
    show_analysis_page()
elif st.session_state.current_page == 'player':
    show_player_page()
elif st.session_state.current_page == 'model':
    show_model_page()