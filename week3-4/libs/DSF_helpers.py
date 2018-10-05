state_name_pairs = {
    'Hawaii': 'HI', 
    'Washington': 'WA', 
    'Montana': 'MT', 
    'Maine': 'ME', 
    'North Dakota': 'ND', 
    'South Dakota': 'SD', 
    'Wyoming': 'WY', 
    'Wisconsin': 'WI', 
    'Idaho': 'ID', 
    'Vermont': 'VT', 
    'Minnesota': 'MN', 
    'Oregon': 'OR', 
    'New Hampshire': 'NH', 
    'Iowa': 'IA', 
    'Massachusetts': 'MA', 
    'Nebraska': 'NE', 
    'New York': 'NY', 
    'Pennsylvania': 'PA', 
    'Connecticut': 'CT', 
    'Rhode Island': 'RI', 
    'New Jersey': 'NJ', 
    'Indiana': 'IN', 
    'Nevada': 'NV', 
    'Utah': 'UT', 
    'California': 'CA', 
    'Ohio': 'OH', 
    'Illinois': 'IL', 
    'District of Columbia': 'DC', 
    'Delaware': 'DE', 
    'West Virginia': 'WV', 
    'Maryland': 'MD', 
    'Colorado': 'CO', 
    'Kentucky': 'KY', 
    'Kansas': 'KS', 
    'Virginia': 'VA', 
    'Missouri': 'MO', 
    'Arizona': 'AZ', 
    'Oklahoma': 'OK', 
    'North Carolina': 'NC', 
    'Tennessee': 'TN', 
    'Texas': 'TX', 
    'New Mexico': 'NM', 
    'Alabama': 'AL',
    'Mississippi': 'MS',
    'Georgia': 'GA',
    'South Carolina': 'SC',
    'Arkansas': 'AR',
    'Louisiana': 'LA',
    'Florida': 'FL',
    'Michigan': 'MI',
    'Alaska': 'AK'}

import re

def get_state_ABRV(place_full_name, verbose=False):
    regex = r'([A-Z]){2,3}'
    match_ABRV = re.search(regex, place_full_name)
    
    if match_ABRV == None:
        if verbose: print("could not find: ",place_full_name)
        return place_full_name
    
    abrv = match_ABRV.group()
    
    
    if abrv == "USA":
        regex = r'(([A-Z][a-z]* [A-Z][a-z]*)|[A-Z][a-z]*)(?=,)'
        match_state_name = re.search(regex, place_full_name)
        
        if match_state_name == None:
            if verbose: print("could not find: ", place_full_name)
            return place_full_name
        
        state_name = match_state_name.group()
        abrv = state_name_to_abrv(state_name)
        
    return abrv

def state_name_to_abrv(state_name):
    abrv = state_name_pairs.get(state_name)
    return abrv