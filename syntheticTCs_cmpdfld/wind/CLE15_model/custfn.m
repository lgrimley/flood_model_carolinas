%%

function neglog = custfn(xi, sig, data)
    custfn = -sum(w.*(-log(sig)+(1/xi-1).*log(1-xi.*data/sig))); 
end