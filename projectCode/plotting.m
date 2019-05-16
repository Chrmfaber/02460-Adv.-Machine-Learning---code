%% plot
if ~test
    load('datOMikkel.mat');
    
    f1 = figure(1);
    subplot(2,2,1)
    imagesc(X)
    title('Noisy data')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    subplot(2,2,2)
    imagesc(TX)
    title('Underlying data')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    subplot(2,2,3)
    imagesc(gD*gH)
    title('Reconstruction of data')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    subplot(2,2,4);
    imagesc(D*H)
    title('Article reconstruction')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    f2 = figure(2);
    subplot(2,2,1)
    plot(1:K, gD','linewidth',2); grid on;
    xlim([1 K]); %ylim([0 5]);
    set(gca,'Ytick',[0:0.5:2]);
    title('Colums of D');
    ax = gca;
    ax.FontSize = 20;
    ylabel('Weigth','FontSize',15)
    xlabel('Spectras','FontSize',15)
    h1 = text(-17, 0.55,'Our GPP','fontsize',20);
    set(h1, 'rotation', 90)

    subplot(2,2,2)
    plot(1:L, gH(1,:)','-','linewidth',2);grid on;
    hold on
    plot(1:L, gH(2,:)','--','linewidth',2);
    hold off
    xlim([1 L]); %ylim([0 5]);
    set(gca,'Ytick',[0:0.5:5]);
    title('Rows of H');
    ax = gca;
    ax.FontSize = 20;
    ylabel('Intensity','FontSize',15)
    xlabel('Freqencies','FontSize',15)

    subplot(2,2,3)
    plot(1:K, D','linewidth',2); grid on;
    xlim([1 K]); %ylim([0 5]);
    set(gca,'Ytick',[0:0.5:5]);
    ax = gca;
    ax.FontSize = 20;
    ylabel('Weigth','FontSize',15)
    xlabel('Spectras','FontSize',15)
    h1 = text(-17, 0.30,'Article GPP','fontsize',20);
    set(h1, 'rotation', 90)

    
    subplot(2,2,4)
    plot(1:L, H','linewidth',2); grid on;
    xlim([1 L]); %ylim([0 5]);
    set(gca,'Ytick',[0:1:5]);
    ax = gca;
    ax.FontSize = 20;
    ylabel('Intensity','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    

    movegui(f2,'west')
    movegui(f1,'east')
else
    f1 = figure(1);
    subplot(2,2,1)
    imagesc(X)
    title('Noisy data')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    subplot(2,2,2)
    imagesc(gD*gH)
    title('Reconstruction of data')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    subplot(2,2,3)
    plot(1:K, gD','linewidth',2); grid on;
    xlim([1 K]); %ylim([0 5]);
    set(gca,'Ytick',[0:2:5]);
    title('Colums of D');
    ax = gca;
    ax.FontSize = 20;
    ylabel('Weigth','FontSize',15)
    xlabel('Spectras','FontSize',15)
    
    
    subplot(2,2,4)
    plot(1:L, gH','linewidth',2); grid on;
    xlim([1 L]); %ylim([0 5]);
    title('Rows of H');
    ax = gca;
    ax.FontSize = 20;
    ylabel('Intensity','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    figure(2)
    imagesc(reshape(Dr,[10,10])')
    title('Raman map')
    ax = gca;
    ax.FontSize = 20;
    
    % and this
    figure(3)
    imagesc(DD)
    title('Raman data matrix')
    ax = gca;
    ax.FontSize = 20;
    ylabel('Spectras','FontSize',15)
    xlabel('Freqencies','FontSize',15)
    
    figure(4)
    plot(a'*vp/max(a'*vp),'LineWidth',2); grid on;
    title('Spectroscopy')
    ax = gca;
    ax.FontSize = 20;
    xlabel('Frequency','FontSize',15) 
    ylabel('Intensity','FontSize',15)
    
end

% f3 = figure(3);
% imagesc(gDmean*gHmean)


