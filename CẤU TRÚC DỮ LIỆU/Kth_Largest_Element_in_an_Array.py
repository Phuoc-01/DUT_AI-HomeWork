import heapq
from typing import List

class Solution:
    def findKthLargest(self, nums: List[int], k: int) -> int:
        
        max_heap = [-n for n in nums]
        heapq.heapify(max_heap)
        
        for _ in range(k - 1):
            heapq.heappop(max_heap)
        
        return -max_heap[0]