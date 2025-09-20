from dataclasses import dataclass, asdict
import random
import numpy as np

# p = PlayerPreferences()
# d = p.asdict()
# d['streets'] = 10
# e = PlayerPreferences(**d)
# print(e.streets)

@dataclass
class PlayerPreferences:
        
    # value of winning
    full_score : float = 500.0

    # value of direct posessions
    streets: float = 0
    villages: float = 1/3
    towns: float = 2/3

    # value of cards in hand and penalty for too many cards
    penalty_reference_for_too_many_cards : float = 7.0
    cards_in_hand: float = 0

    # value of direct options
    hand_for_street: float  = 0
    hand_for_village: float  = 0
    hand_for_town: float  = 0
    street_build_options: float  = 0
    village_build_options: float  = 0

    # value of current earning power
    cards_earning_power: float = 0

    # value of secondary options
    hand_for_street_missing_one: float = 0
    hand_for_village_missing_one: float = 0
    hand_for_town_missing_one: float = 0
    secondary_village_build_options: float = 0

    # value of earning power for direct options
    direct_options_earning_power: float = 0

    # value of tertiary options
    hand_for_village_missing_two: float = 0
    hand_for_town_missing_two: float = 0

    # value of secondary options earning power
    secondary_options_earning_power: float = 0

    # weight of cards in value calculation
    resource_type_weight: tuple = (float(1/5),0,float(1/5),float(1/5),float(1/5),float(1/5))

 
    def __post_init__(self):
        # behavior of different elements in normalization
        self.excluded_from_normalization = ['resource_type_weight','full_score','penalty_reference_for_too_many_cards']
        self.separate_weight_normalization = ['resource_type_weight']
        self.resource_type_weight = np.array(self.resource_type_weight)
    def asdict(self):
        return asdict(self)

    def copy(self):
        return PlayerPreferences(**self.asdict())
 
    def normalized(self):
        d = self.asdict()
        n = sum([v for k,v in d.items() if k not in self.excluded_from_normalization])
        for k,v in d.items():
            if k not in self.excluded_from_normalization:
                d[k] = float(v/n)
        
        for k in self.separate_weight_normalization:
            d[k] = np.array(d[k])/sum(d[k])
   
        return PlayerPreferences(**d)
    
    def randomize_values_for_appreciation(self,bandwidth):
        d = self.asdict()
        for k,v in d.items():
            if k not in self.excluded_from_normalization:
                d[k] = v * random.uniform(1.0 - bandwidth, 1.0+bandwidth)
        for k in self.separate_weight_normalization:
            arr = d[k]
            for t in range(len(arr)):
                arr[t] = arr[t] * random.uniform(1.0 - bandwidth, 1.0+bandwidth)
            d[k] = arr
        return PlayerPreferences(**d).normalized()
       
    
    def merge_values_for_appreciation(self,other_preference):
        d  = self.asdict()
        e  = other_preference.asdict()
        for k,v in d.items():
            if k not in self.excluded_from_normalization:
                d[k] = (d[k] + e[k])/2
        for k in self.separate_weight_normalization:
            arr_d = d[k]
            arr_e = e[k]
            for t in range(len(arr_d)):
                arr_d[t] = (arr_d[t]+arr_e[t])/2
            d[k] = arr_d
        return PlayerPreferences(**d).normalized()
    

d={'full_score': 0, 'streets': 0.1615594227747923, 'villages': 0.2961485945875825, 'towns': 0.24153440764240208, 
   'penalty_reference_for_too_many_cards': 7, 'cards_in_hand': 0.0013711796157184063, 
   'hand_for_street': 0.023639163335292715, 'hand_for_village': 0.03128547202248002, 
   'hand_for_town': 0.025491342005668043, 'street_build_options': 0.11401320207729074, 
   'village_build_options': 0.0014075613236915708, 'cards_earning_power': 0.05033539418219382, 
   'hand_for_street_missing_one': 0.002443204306152173, 'hand_for_village_missing_one': 0.002042238708140502, 
   'hand_for_town_missing_one': 0.0017278376628582053, 'secondary_village_build_options': 0.0, 
   'direct_options_earning_power': 0.026728644648182663, 'hand_for_village_missing_two': 0.0014441706982328611, 
   'hand_for_town_missing_two': 0.0036821948583871038, 'secondary_options_earning_power': 0.015145969550934273, 
   'resource_type_weight': np.array([0.27126462, 0.        , 0.13846242, 0.25696639, 0.15221156,
       0.18109501])}

optimized_1_with_0_for_full_score = PlayerPreferences(**d).normalized()