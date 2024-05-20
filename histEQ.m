function final_im = histEQ(img_y)

intensity_cdf = calculateCDF255(img_y);     % Get CDF map
[lenx leny] = size(img_y);
final_im = zeros(size(img_y));

for i = 1:1:lenx
    for j = 1:1:leny
        final_im(i,j) = intensity_cdf(img_y(i,j) + 1);  % Assign each pixel
    end
end
final_im = uint8(final_im); % Return image after Histogram Equalization
end