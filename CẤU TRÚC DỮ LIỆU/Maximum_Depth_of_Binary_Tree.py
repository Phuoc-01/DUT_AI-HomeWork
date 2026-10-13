from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        
class Solution:
    def MaxDepth(self, root: Optional[TreeNode]) -> int:
        if root is None:
            return 0
        
        l = self.MaxDepth(root.left)
        r = self.MaxDepth(root.right)
        
        return 1 + max(l,r)