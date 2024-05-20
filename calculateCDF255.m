function intensity_cdf = calculateCDF255(img_y)

[pdf, freq] = calculatePDF(img_y);
intensity_cdf = zeros(1,256);
[lenx leny] = size(img_y);
n = lenx*leny;
sum_total = 0;

for k = 1:1:length(pdf)
    sum_total =sum_total + freq(k);             % increase cumulative sum over each element 
    cumulative(k) = sum_total;                  
    cdf(k) = cumulative(k)/n;                   % Assign the cdf value for kth element
    intensity_cdf(k) = round(cdf(k) * 255);     % Multiply with 255 and round for utf-8 format
end
end