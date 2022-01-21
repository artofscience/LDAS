function [U, FF, UU] = ldas(K, F, U, FF, UU, P, type)
    tol = 1e-6;
    for i = 1:size(F,2)
        b = F(:,i);
        if norm(b) < tol
            continue
        end
        [d, r] = gso(b,FF);
        if norm(r) > tol
            %fprintf("solve for load %d \n", i);
            FF = [FF r];
            sol = solve(K,r,P,type);
            UU = [UU sol];
            d = [d 1];
        end
        U(:,i) = sum(d.*UU,2);
    end
end

function [A,r] = gso(f, FF)
    A = [];
    r = f + 0;
    for load = FF
        a = dot(r, load) / dot(load, load);
        r = r - a*load;
        A = [A a];
    end
end

function x = solve(K, f, P, type)
    if type == 0
        if size(P) ~= size(K)
            P = chol(K);
        end
        x = P \ (P' \ f);
    else
        if size(P) ~= size(K)
            P = ichol(K);
        end
        res = 1e-6;
        [x,~] = pcg(K,f,res,100,P,P');
    end
end