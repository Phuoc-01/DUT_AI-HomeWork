from typing import Optional

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
        
class Solution:
    def reverse_linked_list(self, head: Optional[ListNode]) -> Optional[ListNode]:
        p = None
        h = head
        
        while h:
            t = h.next
            h.next = p
            p = h
            h = t
            
        return p