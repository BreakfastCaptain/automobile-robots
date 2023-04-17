% Laplacian operator sharpening
f = rgb2gray(imread('photo2.jpg'));
f = double(f);
[row,col] = size(f);
 
L = [0 -1 0;-1 4 -1;0 -1 0];    % Using L algorithm template
g = zeros(row,col);
for i=2:row-1
    for j=2:col-1   % The new result, in order not to affect the subsequent result
        g(i,j) = sum(sum([f(i-1,j-1) f(i-1,j) f(i-1,j+1); ...
                          f(i,j-1) f(i,j) f(i,j+1); ...
                          f(i+1,j-1) f(i+1,j) f(i+1,j+1)].*L));
    end
end
g1 = zeros(row,col);
for i=1:row
    for j=1:col
        g1(i,j) = g(i,j)+f(i,j);
                      
    end
end
 
imshow(uint8(g1));
