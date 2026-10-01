function generate_final_graphs(outputDir)
%GENERATE_FINAL_GRAPHS Create the four-panel paper-style MATLAB figure.
%
% This file intentionally does no ML training and does not alter simulation
% results. It reads CSV outputs from underwater_ml_pipeline.py and produces
% raw PDR/ROR, loss-aware completion delay, and held-out ML accuracy graphs.
%
% Run from the ns-3.38 directory:
%   matlab -batch "generate_final_graphs"

if nargin == 0
    here = fileparts(mfilename('fullpath'));
    outputDir = fullfile(here, 'results', 'final_three_file_project');
end

networkFile = fullfile(outputDir, 'network_metrics_by_node.csv');
accuracyFile = fullfile(outputDir, 'ml_accuracy.csv');
assert(isfile(networkFile), 'Missing %s. Run underwater_ml_pipeline.py project first.', networkFile);
assert(isfile(accuracyFile), 'Missing %s. Run underwater_ml_pipeline.py project first.', accuracyFile);

network = readtable(networkFile, 'VariableNamingRule', 'preserve');
accuracy = readtable(accuracyFile, 'VariableNamingRule', 'preserve');
nodes = [25 50 75 100];
methods = ["SVM", "DTC", "RF", "Worst-OC baseline"];
labels = ["SVM", "DTC", "RF", "Baseline"];
colors = [0.84 0.15 0.16; 1.00 0.50 0.05; 0.12 0.47 0.71; 0.71 0.10 0.35];
markers = {'s', 'o', 'o', 'd'};

% Match the reference four-panel layout: no figure-wide title and one legend
% in the PDR panel only.
fig = figure('Color', 'w', 'Position', [60 60 1800 1225]);

subplot(2, 2, 1);
plotMetric(network, methods, labels, colors, markers, nodes, 'PDR (%)', ...
    'PDR vs Number of Nodes', 'Packet delivery ratio (%)', [0 102], true);

subplot(2, 2, 2);
plotMetric(network, methods, labels, colors, markers, nodes, ...
    'Loss-aware completion delay (ms)', 'End-to-end Delay vs Number of Nodes', ...
    'Average completion delay (ms)', [0 inf], false);

subplot(2, 2, 3);
plotMetric(network, methods, labels, colors, markers, nodes, 'ROR', ...
    'ROR vs Number of Nodes', 'Routing overhead ratio', [0 0.72], false);

subplot(2, 2, 4);
plotAccuracy(accuracy, colors(1:3, :));

exportgraphics(fig, fullfile(outputDir, 'model_comparison_matlab.png'), 'Resolution', 300);
exportgraphics(fig, fullfile(outputDir, 'model_comparison_matlab.pdf'), 'ContentType', 'vector');
fprintf('Saved:\n  %s\n  %s\n', ...
    fullfile(outputDir, 'model_comparison_matlab.png'), ...
    fullfile(outputDir, 'model_comparison_matlab.pdf'));
end


function plotMetric(network, methods, labels, colors, markers, nodes, column, titleText, yLabel, yLimits, addLegend)
hold on;
for i = 1:numel(methods)
    rows = network(string(network.Model) == methods(i), :);
    values = zeros(size(nodes));
    for j = 1:numel(nodes)
        values(j) = rows{rows.Nodes == nodes(j), column};
    end
    plot(nodes, values, ['-' markers{i}], 'Color', colors(i, :), ...
        'LineWidth', 2.4, 'MarkerSize', 8, 'MarkerFaceColor', colors(i, :), ...
        'DisplayName', labels(i));
end
title(titleText, 'FontWeight', 'bold', 'FontSize', 15);
xlabel('Number of nodes');
ylabel(yLabel);
xticks(nodes);
xlim([22 103]);
if isfinite(yLimits(2))
    ylim(yLimits);
else
    ylim([0 max(ylim)]);
end
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.0);
if addLegend
    legend('Location', 'best', 'Box', 'on');
end
end


function plotAccuracy(accuracy, colors)
methods = ["SVM", "DTC", "RF"];
values = zeros(1, numel(methods));
for i = 1:numel(methods)
    values(i) = accuracy{string(accuracy.Model) == methods(i), 'Binary accuracy (%)'};
end
b = bar(categorical(methods), values, 0.78, 'FaceColor', 'flat');
b.CData = colors;
for i = 1:numel(values)
    text(i, values(i) + 1.2, sprintf('%.1f%%', values(i)), ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
end
title('Held-out ML Accuracy', 'FontWeight', 'bold', 'FontSize', 15);
ylabel('Binary accuracy (%)');
ylim([0 100]);
grid on;
set(gca, 'FontSize', 12, 'LineWidth', 1.0);
end
