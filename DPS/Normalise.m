function D=Normalise(D,b)
    [n,m]=size(D);
    for ii=1:m
        v=D(:,ii);
        minv=min(v);
        maxv=max(v);
        if minv==maxv
            v=ones(n,1)*.5;
        else
            v=b(1)+(b(2)-b(1))*(v-minv)/(maxv-minv); 
        end
       D(:,ii)=v;
    end
end

