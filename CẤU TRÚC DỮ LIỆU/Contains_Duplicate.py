from typing import List

class Solution:
    def containsDup(self, nums : List[int]) -> bool:
        
        check = set()
        
        for i in nums:
            if i in check:
                return True
            else:
                check.add(i)
        return False