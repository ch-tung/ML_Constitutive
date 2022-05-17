close all
clear

% load data
load('data.mat')

strain = ((1:size(epsilon,4))-1)/(size(epsilon,4)-1)*0.05;

for it = 1:size(epsilon,4)
    epsilon_t = epsilon(:,:,:,it);
    epsilon_t =  reshape(epsilon_t,size(epsilon_t,1),4);
    for j = 1:size(epsilon_t,1)
        p(j,:) = RSHEscore(epsilon_t(j,[1,4,2]));
    end
    p_ave(it,:) = mean(p);
    p_std(it,:) = std(p);
end

%% plot evolution

figure
hold on
box on

xlabel('strain (%)','FontSize',24)
ylabel('$p_{l,m}$','FontSize',24,'Interpreter','latex')

set(gca,'LineWidth',2)
set(gcf,'Position',[200,100,600,600])
set(gca,'FontSize',24,'FontName','Arial')

colors = [[0, 0.4470, 0.7410];[0.8500, 0.3250, 0.0980];[0.9290, 0.6940, 0.1250];[0.4940, 0.1840, 0.5560];[0.4660, 0.6740, 0.1880]];
for i = 1:5
%     errorbar(strain*100,p_ave(:,i),p_std(:,i),'.','LineWidth',2)
    patch([strain*100,fliplr(strain*100)],...
          [p_ave(:,i)'+p_std(:,i)',fliplr(p_ave(:,i)'-p_std(:,i)')],...
          colors(i,:),'FaceAlpha',0.25,'LineStyle','none')
end
for i = 1:5
	plot(strain*100,p_ave(:,i),'-','LineWidth',2)
end
xlim([0,5])
% ylim([-0.1,0.1])