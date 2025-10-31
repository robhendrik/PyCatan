"""Player preference data and helpers.

This module defines the `PlayerPreferences` dataclass which captures
parameterized weights used by heuristic value functions (see
`value_utils.py`). It provides helpers to normalize, randomize and merge
preference sets for population-based tuning or sensitivity analysis.

Typical usage:
    p = PlayerPreferences()
    p2 = p.randomize_values_for_appreciation(0.1).normalized()

The module also exports a tuned default `optimized_1_with_0_for_full_score`.

Author: Rob Hendriks
Version: 1.0.0
"""

from dataclasses import dataclass, asdict
import random
import numpy as np

@dataclass
class PlayerPreferences:
    """
    PlayerPreferences captures scalar weights used by the project's
    heuristic value functions. Instances are plain dataclasses and can be
    converted to/from dictionaries for persistence, randomized for search,
    or normalized for comparison.

    Example:
        p = PlayerPreferences()
        d = p.asdict()
        d['streets'] = 10
        p2 = PlayerPreferences(**d)

    Attributes:
        full_score (float): Value assigned to winning the game.
        streets, villages, towns (float): Value per owned street/village/town.
        penalty_reference_for_too_many_cards (float): Reference used when
            applying a penalty for holding many resource cards.
        cards_in_hand (float): Weight applied to cards in hand.
        hand_for_* / *_build_options / *_missing_* (float): Weights for
            direct/secondary/tertiary build options and missing-card bonuses.
        cards_earning_power, direct_options_earning_power,
            secondary_options_earning_power (float): Weights for earning-power terms.
        resource_type_weight (np.ndarray): Per-resource type weights used
            when converting resource vectors to scalar values.
    """
        
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

    def asdict(self) -> dict:
        """Return a plain dict representation of the preferences.

        Returns:
            dict: Mapping of field names to values (compatible with the class
                constructor).
        """
        return asdict(self)

    def copy(self) -> 'PlayerPreferences':
        """Return a shallow copy of this PlayerPreferences instance.

        Returns:
            PlayerPreferences: New instance with the same field values.
        """
        return PlayerPreferences(**self.asdict())
 
    def normalized(self) -> 'PlayerPreferences':
        """Return a normalized copy of the preferences.

        Normalization scales all numeric preference fields (except those in
        ``excluded_from_normalization``) so their sum equals 1. Fields listed
        in ``separate_weight_normalization`` are normalized independently
        (useful for vector-like weights such as ``resource_type_weight``).

        Returns:
            PlayerPreferences: New normalized preferences instance.
        """
        d = self.asdict()
        n = sum([v for k,v in d.items() if k not in self.excluded_from_normalization])
        for k,v in d.items():
            if k not in self.excluded_from_normalization:
                d[k] = float(v/n)
        
        for k in self.separate_weight_normalization:
            d[k] = np.array(d[k])/sum(d[k])
   
        return PlayerPreferences(**d)
    
    def randomize_values_for_appreciation(self,bandwidth: float) -> 'PlayerPreferences':
        """Return a randomized (and normalized) variant of these preferences.

        Each numeric preference (except excluded keys) is multiplied by a
        random factor in [1-bandwidth, 1+bandwidth]. Vector-like weights in
        ``separate_weight_normalization`` are randomized element-wise.

        Args:
            bandwidth (float): Fractional perturbation range (0.0 -> no
                change, 0.1 -> ±10% randomization).

        Returns:
            PlayerPreferences: A normalized, randomized preferences instance.
        """
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
       
    def merge_values_for_appreciation(self,other_preference: 'PlayerPreferences') -> 'PlayerPreferences':
        """Merge these preferences with another and return a normalized result.

        The merge performed is a simple element-wise average of numeric
        fields (excluding those in ``excluded_from_normalization``). Vector
        fields listed in ``separate_weight_normalization`` are averaged
        element-wise. The result is normalized before being returned.

        Args:
            other_preference (PlayerPreferences): Other preferences to merge.

        Returns:
            PlayerPreferences: Normalized merged preferences.
        """
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