close all
clear

% plot strain evolutions
load('data.mat')

figure(1)
hold on
box on
grid on
view(-17.5,22.5)
axis equal

xlim([-.25,.25]/10)
ylim([-.1,.4]/10)
zlim([-.25,.25]/10)

xticks([-.5:.1:.5]/10)
yticks([-.5:.1:.5]/10)
zticks([-.5:.1:.5]/10)

xlabel('\epsilon_{xx}','FontSize',24)
ylabel('\epsilon_{yy}','FontSize',24)
zlabel('\epsilon_{xy}','FontSize',24)
set(gca,'LineWidth',2)
set(gcf,'Position',[200,100,600,600])
set(gca,'FontSize',24,'FontName','Arial')

color = parula(size(epsilon,4));

for it = 1:size(epsilon,4)
    epsilon_t = epsilon(:,:,:,it);
    epsilon_t_xx = epsilon_t(:,1,1);
    epsilon_t_xy = epsilon_t(:,1,2);
    epsilon_t_yy = epsilon_t(:,2,2);
    
    sc = scatter3(epsilon_t_xx,epsilon_t_yy,epsilon_t_xy,50,color(it,:),'.');
    sc.MarkerEdgeAlpha = .5;
end

% plot stress evolutions
load('data.mat')

figure(2)
hold on
box on
grid on
view(-17.5,22.5)
view(0,90)
axis equal

xlim([-.2,.2]/10)
ylim([-.2,.2]/10)
zlim([-.2,.2]/10)

xticks([-.5:.1:.5]/10)
yticks([-.5:.1:.5]/10)
zticks([-.5:.1:.5]/10)

xlabel('\sigma_{xx}','FontSize',24)
ylabel('\sigma_{yy}','FontSize',24)
zlabel('\sigma_{xy}','FontSize',24)
set(gca,'LineWidth',2)
set(gcf,'Position',[200,100,600,600])
set(gca,'FontSize',24,'FontName','Arial')

color = parula(size(sigma,4));

E_m = 1e6;

for it = 1:size(sigma,4)
    sigma_t = sigma(:,:,:,it)/E_m;
    sigma_t_xx = sigma_t(:,1,1);
    sigma_t_xy = sigma_t(:,1,2);
    sigma_t_yy = sigma_t(:,2,2);
    
    scatter3(sigma_t_xx,sigma_t_yy,sigma_t_xy,50,color(it,:),'.')
    sc.MarkerEdgeAlpha = .5;
end

% yield surface
syms x y z
f = sqrt(3/2*((x-(x+y)/2)^2+(y-(x+y)/2)^2+2*z^2)) - 0.01;
fi = fimplicit3(f);
fi.EdgeColor = 'none';
fi.FaceColor = '#134098';
fi.FaceAlpha = 0.1;
