class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        mapping = {")": "(", "}": "{", "]": "["}
        
        for i in s:
            if i in mapping:
                top = stack.pop() if stack else None
                
                if mapping[i] != top:
                    return False
            else:
                stack.append(i)
        
        return not stack