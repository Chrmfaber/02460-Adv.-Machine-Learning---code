% for outside
figure(4)
plot(a'*vp/max(a'*vp),'LineWidth',2)
title('Spectroscopy')
xlabel('Frequency')
ylabel('Intensity')
grid on

% for 'if test' part
figure(2)
imagesc(reshape(Dr,[10,10])')
title('Raman map')

% and this
figure(3)
imagesc(DD)
title('Raman data matrix')

% % for test
% figure(3)
% subplot(1,2,1)
% imagesc(reshape(gD(:,1),[10 10])')
% subplot(1,2,2)
% imagesc(reshape(gD(:,2),[10 10])')