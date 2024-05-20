function [pdf, freq] = calculatePDF(img_y)

[lenx leny] = size(img_y);
freq = zeros(1,256);
n1n2 = lenx*leny;

% Calculate PDF
for i = 1:1:lenx
    for j = 1:1:leny
        val = img_y(i,j);
        freq(val+1) = freq(val+1)+1;    % Construct frequency array for each intensity
        pdf(val+1) = freq(val+1)/n1n2;  % Increase corresponding frequency value in each iteration
    end
end
end