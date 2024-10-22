import pandas as pd
import itertools

class TVD:
    def __init__(self, original_data, perturbed_data, domains):
        self.original_data = original_data
        self.perturbed_data = perturbed_data
        self.domains = domains
        self.tvd = self.total_variation_distance()
        
    def total_variation_distance(self):
        # """
        # Computes the total variation distance between two (classical) probability
        # measures P(x) and Q(x).
        #     tvd(P,Q) = (1/2) \sum_x |P(x) - Q(x)|
        # """
        d = []
        for atr in self.original_data.columns:
            d.append(self.domains.get(atr))
        com = list(itertools.product(*d))
        tvd = 0 

        for co in com:
            original_query_condition = pd.Series([True] * len(self.original_data))
            perturbed_query_condition = pd.Series([True] * len(self.perturbed_data))
            
            for col, val in zip(self.original_data.columns, co):
                original_query_condition &= (self.original_data[col] == val)
                perturbed_query_condition &= (self.perturbed_data[col] == val)
            
            o_count = original_query_condition.sum()
            p_count = perturbed_query_condition.sum()
            o_count = o_count / (len(self.original_data))
            p_count = p_count / (len(self.perturbed_data))
            tvd += abs(o_count - p_count)

        tvd = tvd * (1/2) 
        return tvd
