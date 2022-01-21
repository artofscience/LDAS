K = sparse(diag([2 2])); 
K = (K + K')/2;

F = [[1 2 4 4]; 
    [0 0 2 4]];


U = zeros(size(F));

[U, FF, UU] = ldas(K, F, U, [], [], [], 0)