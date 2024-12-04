%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author  :   Junming Yin
% Contact :   junmingy@email.arizona.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plotFunctions(Xtrain, ftrain, Xtest, ftest, R, a, b)

% plot the results, seed = 23
points  = 0:0.01:1;
figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,9), R + ftrain(:,1), '.k','MarkerSize',18); hold on;     % residual
[dumm I] = sort(Xtest(:,9));
plot(dumm, ftest(I,1), 'b-', 'LineWidth',5); hold on;
plot(points, -2*sin(2*(a:0.05:b)), '--r', 'LineWidth',5);              % true function
xlabel('$x_1$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_1$', 'Fontsize',25,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f1.pdf')


% 
figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,10), R + ftrain(:,2), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,10));
plot(dumm, ftest(I,2), 'b-', 'LineWidth',5); hold on;
plot(points, (a:0.05:b).^2 - 2.0833, '--r', 'LineWidth',5);
xlabel('$x_2$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_2$', 'Fontsize',25,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f2.pdf')


figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,11), R + ftrain(:,3), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,11));
plot(dumm, ftest(I,3), 'b-', 'LineWidth',5); hold on;
plot(points, 2*sin((a:0.05:b))./(2 - sin((a:0.05:b))) - 0.3716, '--r', 'LineWidth',5);
xlabel('$x_3$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_3$', 'Fontsize',25,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f3.pdf')


figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,12), R + ftrain(:,4), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,12));
plot(dumm, ftest(I,4), 'b-', 'LineWidth',5); hold on;
plot(points, exp(-(a:0.05:b)) - 2.4201, '--r', 'LineWidth',5);
xlabel('$x_4$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_4$', 'Fontsize',22,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f4.pdf')



figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,97), R + ftrain(:,5), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,97));
plot(dumm, ftest(I,5), 'b-', 'LineWidth',5); hold on;
plot(points, (a:0.05:b).^3 + 1.5*((a:0.05:b) - 1).^2 - 4.625, '--r', 'LineWidth',5);
xlabel('$x_5$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_5$', 'Fontsize',22,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f5.pdf')


figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,98), R + ftrain(:,6), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,98));
plot(dumm, ftest(I,6), 'b-', 'LineWidth',5); hold on;
plot(points, (a:0.05:b), '--r', 'LineWidth',5);
xlabel('$x_6$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_6$', 'Fontsize',22,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f6.pdf')



figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,99), R + ftrain(:,7), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,99));
plot(dumm, ftest(I,7), 'b-', 'LineWidth',5); hold on;
plot(points, 3*sin(exp(-0.5*(a:0.05:b))) - 1.8588, '--r', 'LineWidth',5);
xlabel('$x_7$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_7$', 'Fontsize',25,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f7.pdf')


figure
% set(gcf, 'Position', [216   791   525   493])
set(gcf, 'Color', 'w');
plot(Xtrain(:,100), R + ftrain(:,8), '.k','MarkerSize',18); hold on;
[dumm I] = sort(Xtest(:,100));
plot(dumm, ftest(I,8), 'b-', 'LineWidth',5); hold on;
plot(points, -5*normcdf((a:0.05:b), 0.5, 0.8) + 2.0015, '--r', 'LineWidth',5);
xlabel('$x_8$', 'Fontsize',25,'interpreter','latex');
ylabel('$f_8$', 'Fontsize',22,'interpreter','latex');
set(gca,'FontSize',18)
axis square
%export_fig(gcf, 'f8.pdf')


