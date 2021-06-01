function [x,y] = index2index(index,rows)
x=mod(index,rows);
y=floor(index/rows)+1;
if x == 0
    x = rows;
    y = y-1; 
end  
end