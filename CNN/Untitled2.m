a1 = 'rawdata/';
a3 = '.png';
for a = 3744:1:5222
    a2= num2str(a);
    c1=[a1,a2];
    c2 = [a2,a3];
    fid=fopen(c1); 
    if fid == -1
        a = a+1;
        a2= num2str(a);
        c1=[a1,a2];
        c2 = [a2,a3];
        fid=fopen(c1); 
    end
    I = fread(fid);
    imagesc(reshape(I, 128, 128)'); 
    colormap(gray(256));
    saveas(1,c2);
end
