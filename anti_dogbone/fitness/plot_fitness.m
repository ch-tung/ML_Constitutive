clear
close all

fitness_name = {'B?zier 2-fold','B?zier 3-fold','Free Harmonic Holes'};
mat_name = {'GWO126.mat','GWO127.mat','GWO128.mat'};
color_name = {'r','g','b'};

for j = 1:3
    figure(1)
    hold on
    box on
%     axis equal
    set(gcf,'Position',[200,100,600,600])
    
    data = load(mat_name{j});
    
    alpha = data.alpha_wolf;
    beta = data.beta_wolf;
    delta = data.delta_wolf;
    best = data.fitness;
    
    plot(1:length(best),best,'-','LineWidth',2,'DisplayName',fitness_name{j})
%     plot(1:length(alpha),alpha,'-r')
%     plot(1:length(beta),beta,'-g')
%     plot(1:length(delta),delta,'-b')
    
    xlim([0,500])
    ylim([0,1])

    legend('FontSize',18)
    
    xlabel('Generations','FontSize',24)
    ylabel('$\mathcal{S}_{\rm{cdf}}$','FontSize',24,'Interpreter','latex')
    
%     set(gca, 'XScale', 'log')
    set(gca, 'YScale', 'log')
    set(gca,'LineWidth',2)
    set(gcf,'Position',[200,100,800,600])
    set(gca,'FontSize',24,'FontName','Arial')
    
end