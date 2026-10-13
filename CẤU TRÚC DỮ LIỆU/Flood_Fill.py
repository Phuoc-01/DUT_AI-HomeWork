class Solution:
    def flood_fill(self, image: list[list[int]], sr: int, sc: int,color: int) -> list[list[int]]:
        m,n = len(image), len(image[0])
        
        o_color = image[sr][sc]
        
        if o_color == color:
            return image
        
        def dfs(x,y):
            if x < 0 or x >= m or y < 0 or y >= n or image[x][y] != o_color:
                return
        
            image[x][y] = color
            
            dfs(x+1,y)
            dfs(x-1,y)
            dfs(x,y+1)
            dfs(x,y-1)
        
        dfs(sr,sc)
        
        return image