%%
%Problem 1

img = imread("mandi.tif");
img = img(1:1:2000,1:1:3000);
[lenx leny] = size(img);

% Seperate RGB Channels
img_g = img(2:2:end,1:2:end); 
img_r = img(1:2:end,2:2:end);
img_b = img(1:2:end,1:2:end);

% Interpolate red channel
img_r_2x = interp2x_me(img_r);

% Interpolate green channel
img_g_2x = interp2x_me(img_g);


% Interpolate blue channel
img_b_2x = interp2x_me(img_b);


% Normalize and Display Resulting Image
img_total = zeros([lenx leny 3]);
img_total(:,:,1) = img_r_2x(:,:);
img_total(:,:,2) = img_g_2x(:,:);
img_total(:,:,3) = img_b_2x(:,:);
img_total = img_total/255;



imshow(img_total);
title("Manual Interpolation and Demosaicing");
figure;
J = demosaic(img,"bggr");
imshow(J);
title("Demosaic by Matlab");

%%
%Problem 2

img_org = imresize(imread("casiopea.jpg"), [500 500]); 
img_ycbcr = rgb2ycbcr(img_org);
img_y = img_ycbcr(:,:,1);

prewitt_ker = [-1,0,1;-1,0,1;-1,0,1];   % Define a 3x3 prewitt kernel
sobel_ker = [-1,0,1;-2,0,2;-1,0,1];     % Define a 3x3 sobel kernel

gauss = fspecial('gaussian',3);         % Construct a 3x3 gaussian filter
[grad_x grad_y] = gradient(gauss);      % Take the gradient of gaussian filter in both dimensions (derivative of gaussian filter)

gauss_grad = conv2(grad_x,grad_y,'same');   % Construct seperable filter


prewitt_im = conv2(img_y,prewitt_ker,'same'); % Convolve image with prewitt kernel
sobel_im = conv2(img_y,sobel_ker,'same');     % Convolve image with sobel kernel
gauss_grad_im = conv2(img_y,gauss_grad,'same');   % Convolve image with derivative of gaussian kernel
edge_im = edge(img_y,'Canny');                    % Use edge function to detect edges

% Add Noise
noiseDB = 40; % Which results in 0.1 std
img_noise = imnoise(img_y, 'gaussian', 0, 0.08^2); % Add gaussian noise to the image

prewitt_im_noise = conv2(img_noise,prewitt_ker,'same'); % Convolve noisy image with prewitt kernel
sobel_im_noise = conv2(img_noise,sobel_ker,'same');     % Convolve noisy image with sobel kernel
gauss_grad_im_noise = conv2(img_noise,gauss_grad,'same');   % Convolve noisy image with derivative of gaussian kernel
edge_im_noise = edge(img_noise,'Canny');                    % Use edge function on noisy image to detect edges

% Plot the Results

tiledlayout(2,2);
nexttile;
imshow(sobel_im);
title("Sobel Kernel");
nexttile;
imshow(prewitt_im);
title("Prewitt Kernel");
nexttile;
imshow(gauss_grad_im);
title("Gaussian Kernel");
nexttile;
imshow(edge_im);
title("Edge Function");

figure;
tiledlayout(2,2);
nexttile;
imshow(sobel_im_noise);
title("Sobel Kernel - 40db Noise");
nexttile;
imshow(prewitt_im_noise);
title("Prewitt Kernel - 40db Noise");
nexttile;
imshow(gauss_grad_im_noise);
title("Gaussian Kernel - 40db Noise");
nexttile;
imshow(edge_im_noise);
title("Edge Function - 40db Noise");

%%
%Problem 3.1)
img_org = imread("casiopea.jpg");
img_ycbcr = rgb2ycbcr(img_org); %seperate the
img_y = img_ycbcr(:,:,1);

img_bright_y_03 = AGC(img_y,50); % Linear Automatic Gain Control with low-threshold of 50 intensity

img_ycbcr(:,:,1) = (img_bright_y_03); % Load the AGC adjusted Y channel to image
img_rgb = ycbcr2rgb(img_ycbcr); % Convert ycbcr to rgb space

% Plot the results
tiledlayout(1,2);
nexttile;
imshow(img_bright_y_03);
title("Image After AGC");
nexttile;
imshow(img_y);
title("Image Before AGC");

%%
%Problem 3.1) Histogram EQ

img_org = imread("casiopea.jpg");
img_ycbcr = rgb2ycbcr(img_org);
img_y = img_ycbcr(:,:,1);

% Calculating Histogram

[lenx leny] = size(img_y);
hist = zeros(1,256);

for i = 1:1:lenx
    for j = 1:1:leny
        val = img_y(i,j);
        hist(val) = hist(val) + 1;
    end
end

% Call the equalizer function on image
hist_norm_img = histEQ(img_y); 

% Plot the Results
imshow(hist_norm_img);
title("After Histogram EQ");

%% 
%Problem 3.1) Unsharp Masking on Y Channel

img_org = imread("casiopea.jpg");
img_ycbcr_03 = rgb2ycbcr(img_org);
img_ycbcr_08 = rgb2ycbcr(img_org);
img_y = img_ycbcr_03(:,:,1);

% Apply Threshold to better brighten the image
img_y(img_y<80) = 0;

h = fspecial('gaussian',9);
filtered = uint8(conv2(img_y,h,'same'));

% Apply Unsharp Masking
img_masked_y_03 = filtered + 3*(img_y - filtered);
img_masked_y_08 = filtered + 0.8*(img_y - filtered);


img_bright_y_03 = AGC(img_masked_y_03,50); % AGC on Y channel
img_bright_y_08 = AGC(img_masked_y_08,50); % AGC on Y channel
img_ycbcr_03(:,:,1) = img_bright_y_03;      % Reconstruct 3-Channel Image
img_ycbcr_08(:,:,1) = img_bright_y_08;      % Reconstruct 3-Channel Image

% Convert to the RGB space
img_rgb_03 = ycbcr2rgb(img_ycbcr_03);           
img_rgb_08 = ycbcr2rgb(img_ycbcr_08);

% Plot the Results
tiledlayout(2,1);
% nexttile;
% imshow(img_org);
% title("Original Image");
nexttile;
imshow(img_rgb_03);
title("Unsharp Masked Y-Channel Image, B=3");
nexttile;
imshow(img_rgb_08);
title("Unsharp Masked Y-Channel Image, B=0.8");


%%
% Problem 3.2) Unsharp Masking on Each Channel Applied Seperately

% Seperate the channels
img_y = img_ycbcr_03(:,:,1);
img_cb = img_ycbcr_03(:,:,2);
img_cr = img_ycbcr_03(:,:,3);

% Filter the channels seperately
filtered_y = uint8(conv2(img_y,h,'same'));
filtered_cb = uint8(conv2(img_cb,h,'same'));
filtered_cr = uint8(conv2(img_cr,h,'same'));

% Apply Unsharp Masking to the channels seperately
img_masked_y_03 = filtered + 3*(img_y - filtered_y);
img_masked_y_08 = filtered + 0.8*(img_y - filtered_y);
img_masked_cb_03 = filtered + 3*(img_cb - filtered_cb);
img_masked_cb_08 = filtered + 0.8*(img_cb - filtered_cb);
img_masked_cr_03 = filtered + 3*(img_cr - filtered_cr);
img_masked_cr_08 = filtered + 0.8*(img_cr - filtered_cr);

% img_bright_y_03 = AGC(img_masked_y_03,50);
% img_bright_y_08 = AGC(img_masked_y_08,50);
% img_bright_cb_03 = AGC(img_masked_cb_03,50);
% img_bright_cb_08 = AGC(img_masked_cb_08,50);
% img_bright_cr_03 = AGC(img_masked_cr_03,50);
% img_bright_cr_08 = AGC(img_masked_cr_08,50);

% Construct 3 Channel YCbCr Image
img_ycbcr_03(:,:,1) = img_masked_y_03;
img_ycbcr_08(:,:,1) = img_masked_y_08;
img_ycbcr_03(:,:,2) = img_masked_cb_03;
img_ycbcr_08(:,:,2) = img_masked_cb_08;
img_ycbcr_03(:,:,3) = img_masked_cr_03;
img_ycbcr_08(:,:,3) = img_masked_cr_08;

% Convert YCbCr Image to RGB colorspace
img_rgb_03 = ycbcr2rgb(img_ycbcr_03);
img_rgb_08 = ycbcr2rgb(img_ycbcr_08);

tiledlayout(2,1);
% nexttile;
% imshow(img_org);
% title("Original Image");
nexttile;
imshow(img_rgb_03);
title("Unsharp Mask on All Channels, B=3");
nexttile;
imshow(img_rgb_08);
title("Unsharp Mask on All Channels, B=0.8");

%%
% Function Definitions for Bicubic Interpolation

function img_inter = interp2x_me(img)

[lenx leny] = size(img);
img_inter = zeros(2*size(img));

for i = 1:1:(lenx-3)
    for j = 1:1:(leny-3)
        p = img(i:1:(i+3),j:1:(j+3));
        mat = zeros(2,2);
        for k = 1:1:2
            for l = 1:1:2
                mat(k,l) = bicubicInterpolate(p, 0.05*k, 0.05*l);
            end
        end

        j_ind = 2*j-1;
        j_ind_1 = 2*j;
        i_ind = 2*i -1;
        i_ind_1 = 2*i;
        img_inter(i_ind:1:i_ind_1,j_ind:1:j_ind_1) = mat(:,:);
        img_inter(2*i,2*j) = img(i,j);
    end
end
end

function result = cubicInterpolate(p, x)
    result = p(2) + 0.5 * x * (p(3) - p(1) + x * (2.0 * p(1) - 5.0 * p(2) + 4.0 * p(3) - p(4) + x * (3.0 * (p(2) - p(3)) + p(4) - p(1))));
end

function result = bicubicInterpolate(p, x, y)
    arr = zeros(1, 4);
    arr(1) = cubicInterpolate(p(1, :), y);
    arr(2) = cubicInterpolate(p(2, :), y);
    arr(3) = cubicInterpolate(p(3, :), y);
    arr(4) = cubicInterpolate(p(4, :), y);
    result = cubicInterpolate(arr, x);
end

