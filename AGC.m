function img_bright = AGC(img_y,low_threshold255)
% Apply Threshold to better brighten the image
img_y(img_y<low_threshold255) = 0; % Delete pixels which have intensity below threshold

minmin = double(min(min(img_y)));   % Get the minimum intensity after threshold filtering
maxmax = double(max(max(img_y)));   % Get the maximum intensity after threshold filtering

% Solve the Linear Mapping Equation
alfa = 255/(maxmax-minmin);
beta = (-minmin)*alfa;
intensity_diff = (double(maxmax-minmin))/256;

% Construct image by AGC Equation
img_bright = alfa*img_y+beta;

end