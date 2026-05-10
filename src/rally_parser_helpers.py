import pandas as pd
import numpy as np
import re
from collections import defaultdict
from config import FINAL_CLEANED_DATA, FEATURED_DATA

ground_strokes = ['f', 'b', 's', 'r', 'v', 'o', 'l', 'm', 'z', 'j', 'q', 't', 'p', 'u', 'y']

shot_direction = ['1', '2', '3']

serve_direction = ['4', '5', '6']

return_depth = ['7', '8', '9']

position_information = ['=', '+', '-']

terminal_symbols = ['*', '#', '@']

error_location = ['n', 'w', 'd', 'x']

shank = ['!']

def is_first_serve(first_serve, second_serve) -> bool:
    """Checks if the point starts with a second serve or a first serve."""
    # Treat NaN and empty strings as missing data
    first_serve_empty = pd.isna(first_serve) or first_serve == ""
    second_serve_empty = pd.isna(second_serve) or second_serve == ""
    
    if first_serve_empty and second_serve_empty:
        return np.nan  # Missing data  
    elif second_serve_empty:
        return True  # Only first serve info available, assume first serve
    else:
        return False # Second serve info available, so it's not a first serve
    
def make_shots(first_serve, second_serve) -> list:
    """Extracts shot types from the first and second serve strings."""
    shots = []
    is_first = is_first_serve(first_serve, second_serve)
    if is_first:
        sequence = first_serve
    else:
        sequence = second_serve
        
    new_shot = ""
    for char in sequence:
        if char in ground_strokes:
            if new_shot:  # Only append if not empty
                shots.append(new_shot)
            new_shot = char
        else:
            new_shot += char
    
    # Append the final shot
    if new_shot:
        shots.append(new_shot)
            
    return shots

def parse_serve_plus_one_terminal(shot) -> dict:
    """Parse serve plus one terminal"""
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    shot_dir = None
    shot_depth = None
    error_loc = None
    error_type = None
    winner = False
    shank_info = False
    position_info = []
    
    if '1' in shot:
        shot_dir = '1'
    elif '2' in shot:
        shot_dir = '2'
    elif '3' in shot:
        shot_dir = '3'
        
    if '7' in shot:
        shot_depth = '7'
    elif '8' in shot:
        shot_depth = '8'
    elif '9' in shot:
        shot_depth = '9'
    
    if 'n' in shot:
        error_loc = 'n'
        error_type = 'net'
    elif 'w' in shot:
        error_loc = 'w'
        error_type = 'wide'
    elif 'd' in shot:
        error_loc = 'd'
        error_type = 'deep'
    elif 'x' in shot:
        error_loc = 'x'
        error_type = 'deep_and_wide'
    
    if '*' in shot:
        winner = True
    elif '#' in shot:
        winner = False
        error_type = 'forced_error'
    elif '@' in shot:
        winner = False
        error_type = 'unforced_error'
        
    
             
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
        
    return {
        "shot_type": shot_type,
        "shot_direction": shot_dir,
        "shot_depth": shot_depth,
        "error_location": error_loc,
        "error_type": error_type,
        "winner": winner,
        "shank_info": shank_info,
        "position_info": position_info,
        "serve_plus_one": True
    }
    
def parse_serve_plus_one(shot) -> dict:
    """parse serve plus one shot - no terminal symbol, but can have shank and position info"""
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    shot_dir = None
    shot_depth = None
    
    shank_info = False
    position_info = []
    
    if '1' in shot:
        shot_dir = '1'
    elif '2' in shot:
        shot_dir = '2'
    elif '3' in shot:
        shot_dir = '3'
        
    if '7' in shot:
        shot_depth = '7'
    elif '8' in shot:
        shot_depth = '8'
    elif '9' in shot:
        shot_depth = '9'
             
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
    
    
        
    return {
        "shot_type": shot_type,
        "shot_direction": shot_dir,
        "shot_depth": shot_depth,
        "error_location": None,
        "error_type": None,
        "winner": False,
        "shank_info": shank_info,
        "position_info": position_info,
        "serve_plus_one": True
    }

def parse_terminal_shot(shot) -> dict:
    """Parse final shot in rally with terminal symbol"""
    # Check for terminal symbols
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    shot_dir = None
    shot_depth = None
    error_loc = None
    error_type = None
    winner = False
    shank_info = False
    position_info = []
    
    if '1' in shot:
        shot_dir = '1'
    elif '2' in shot:
        shot_dir = '2'
    elif '3' in shot:
        shot_dir = '3'
        
    if '7' in shot:
        shot_depth = '7'
    elif '8' in shot:
        shot_depth = '8'
    elif '9' in shot:
        shot_depth = '9'
    
    if 'n' in shot:
        error_loc = 'n'
        error_type = 'net'
    elif 'w' in shot:
        error_loc = 'w'
        error_type = 'wide'
    elif 'd' in shot:
        error_loc = 'd'
        error_type = 'deep'
    elif 'x' in shot:
        error_loc = 'x'
        error_type = 'deep_and_wide'
    
    if '*' in shot:
        winner = True
    elif '#' in shot:
        winner = False
        error_type = 'forced_error'
    elif '@' in shot:
        winner = False
        error_type = 'unforced_error'
        
    
             
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
        
    return {
        "shot_type": shot_type,
        "shot_direction": shot_dir,
        "shot_depth": shot_depth,
        "error_location": error_loc,
        "error_type": error_type,
        "winner": winner,
        "shank_info": shank_info,
        "position_info": position_info,
    }
    
    
def parse_shot(shot) -> dict:
    """Parse any middle rally shot (no terminal)"""
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    shot_dir = None
    shot_depth = None
    
    shank_info = False
    position_info = []
    
    if '1' in shot:
        shot_dir = '1'
    elif '2' in shot:
        shot_dir = '2'
    elif '3' in shot:
        shot_dir = '3'
        
    if '7' in shot:
        shot_depth = '7'
    elif '8' in shot:
        shot_depth = '8'
    elif '9' in shot:
        shot_depth = '9'
             
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
    
    
        
    return {
        "shot_type": shot_type,
        "shot_direction": shot_dir,
        "shot_depth": shot_depth,
        "error_location": None,
        "error_type": None,
        "winner": False,
        "shank_info": shank_info,
        "position_info": position_info
    }
    
def parse_return(shot) -> dict:
    """Parse return shot (middle of rally, no terminal)"""
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    ret_direction = None
    ret_depth = None
    
    shank_info = False
    position_info = []
    
    if '1' in shot:
        ret_direction = '1'
    elif '2' in shot:
        ret_direction = '2'
    elif '3' in shot:
        ret_direction = '3'
        
    if '7' in shot:
        ret_depth = '7'
    elif '8' in shot:
        ret_depth = '8'
    elif '9' in shot:
        ret_depth = '9'
             
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
        
    return {
        "shot_type": shot_type,
        "return_direction": ret_direction,
        "return_depth": ret_depth,
        "error_location": None,
        "error_type": None,
        "shank_info": shank_info,
        "position_info": position_info,
        "winner": False
    }

def parse_return_terminal(shot) -> dict:
    """Parse return shot with terminal - b3w# (error) or f38* (winner)"""
    shot_type = shot[0] if shot[0] in ground_strokes else np.nan
    ret_direction = None
    ret_depth = None
    error_loc = None
    error_type = None
    shank_info = False
    position_info = []
    winner = False
    
    if '1' in shot:
        ret_direction = '1'
    elif '2' in shot:
        ret_direction = '2'
    elif '3' in shot:
        ret_direction = '3'
        
    if '7' in shot:
        ret_depth = '7'
    elif '8' in shot:
        ret_depth = '8'
    elif '9' in shot:
        ret_depth = '9'
    
    if 'n' in shot:
        error_loc = 'n'
    elif 'w' in shot:
        error_loc = 'w'
    elif 'd' in shot:
        error_loc = 'd'
    elif 'x' in shot:
        error_loc = 'x'
        
    if '*' in shot:
        winner = True
    elif '#' in shot:
        error_type = 'forced_error'
    elif '@' in shot:
        error_type = 'unforced_error'
        
    if '!' in shot:
        shank_info = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
        
    return {
        "shot_type": shot_type,
        "return_direction": ret_direction,
        "return_depth": ret_depth,
        "error_location": error_loc,
        "error_type": error_type,
        "shank_info": shank_info,
        "position_info": position_info,
        "winner": winner
    }
        
    
    
def parse_serve(shot) -> dict:
    """Parse serve shot without terminal (rally continues)"""
    # Initialize all variables
    serve_dir = shot[0] if shot[0] in serve_direction else np.nan
    position_info = []
    shank_info = False
    
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
    
    if '!' in shot:
        shank_info = True
        
    return {
        "serve_location": serve_dir,
        "ace": False,
        "fault": False,
        "error_location": None,
        "shank_info": shank_info,
        "position_info": position_info
    }
    
def parse_serve_terminal(shot) -> dict:
    """Parse serve with terminal symbol - ace or fault"""
    # Initialize all variables
    serve_dir = shot[0] if shot[0] in serve_direction else np.nan
    error_loc = None
    fault = False
    ace = False
    shank_info = False
    position_info = []

    if 'n' in shot:
        error_loc = 'n'
        fault = True
    elif 'w' in shot:
        error_loc = 'w'
        fault = True
    elif 'd' in shot:
        error_loc = 'd'
        fault = True
    elif 'x' in shot:
        error_loc = 'x'
        fault = True
    
    if '!' in shot:
        shank_info = True
        
    if '*' in shot:
        ace = True
        fault = False
    elif '#' in shot:
        ace = False
        fault = True
    elif '@' in shot:
        ace = False
        fault = True
        
    if '+' in shot:
        position_info.append('+')
    if '-' in shot:
        position_info.append('-')
    if '=' in shot:
        position_info.append('=')
        
    return {
        "serve_location": serve_dir,
        "ace": ace,
        "fault": fault,
        "error_location": error_loc,
        "shank_info": shank_info,
        "position_info": position_info
    }
    

def parse_rally(shots) -> list:
    rally_length = len(shots)
    if rally_length == 1:
        return [parse_serve_terminal(shots[0])]
    elif rally_length == 2:
        return [parse_serve(shots[0]), parse_return_terminal(shots[1])] 
    elif rally_length == 3:
        return [parse_serve(shots[0]), parse_return(shots[1]), parse_serve_plus_one_terminal(shots[2])]
    elif rally_length >= 4:
        curr = [parse_serve(shots[0]), parse_return(shots[1]), parse_serve_plus_one(shots[2])]
        for shot in shots[3:-1]:
            curr.append(parse_shot(shot))
        curr.append(parse_terminal_shot(shots[-1]))
        return curr

