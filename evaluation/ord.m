function out = ord(val1,val2,delta)
ratio = (val1+eps)/(val2+eps);
    if ratio > delta
        out = 1;
    elseif ratio < 1/delta
        out = -1;
    else
        out = 0;
    end
end