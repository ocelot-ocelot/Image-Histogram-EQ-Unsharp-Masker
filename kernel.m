function result = kernel(s, a)
    
    if (abs(s) >= 0) & (abs(s) <= 1) 
        result = (a+2)*(abs(s)^3)-(a+3)*(abs(s)^2)+1;
        
    elseif (abs(s) > 1) & (abs(s) <= 2) 
        result = a*(abs(s)^3)-(5*a)*(abs(s)^2)+(8*a)*abs(s)-4*a;
    end
end