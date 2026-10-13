from typing import List

class Solution:
    def TwoSum(self, nums: List[int], t : int) -> List[int]:
        check = {}
        
        for i, num in enumerate(nums):
            if t - num in check:
                return [check[t - num], i]
            else:
                check[num] = i
        
        return []