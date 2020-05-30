function v = train_mlp(x, layers, min_options, a0)
    if ~exist('a0', 'var') || isempty(a0)
        a0 = generate_startingpoint(layers);
    end

    orig_dist = pdist(x','euclidean');
    fun = @(weights)(lossfun(x, weights, layers, orig_dist));

    v = fminunc(fun, a0, min_options);
end

function y = generate_startingpoint(layers)
    input_layer = layers(1);
    hidden_layer = layers(2);
    output_layer = layers(3);

    bias = 1;
    size1 = (input_layer + bias) + hidden_layer;
    size2 = (hidden_layer + bias) + output_layer;
    
    y = randn(1, size1 + size2);
end

function stress = lossfun(x_mat, w, layers, orig_dist)
    nn_output_dist = nn_dist(x_mat, w, layers);
    if (length(nn_output_dist) ~= length(orig_dist))
        error('Validate: dist_vectors', 'Matrix sizes do not match.');
    end

    stress = 0;
    for i = 1: length(nn_output_dist)
        stress = stress + (nn_output_dist(i)^2 - orig_dist(i)^2)^2;
    end
end

function dist = nn_dist(x_mat, w, layers)
    n = length(x_mat);
    A = zeros(layers(3), n);

    for i = 1: n
        A(:, i) = mlp(x_mat(:, i), w, layers);
    end
    dist = pdist(A','euclidean');
end

function y = mlp(x, w, layers)
    validate_nn_parameters(layers);
    [w1, w2] = vector2weights(w, layers);
    y = w2*[1; afun(w1*[1;x])];
end

function y = afun(x)
    y = 1./(1+exp(-x)); % k = 1
end

function [w1, w2] = vector2weights(v, layers)
    validate_nn_parameters(layers);

    input_layer = layers(1);
    hidden_layer = layers(2);
    output_layer = layers(3);

    bias = 1;

    nn_tot_size = (input_layer + bias) + hidden_layer + ...
        (hidden_layer + bias) * output_layer;
    if (nn_tot_size ~= length(v))
        error('Vector size does not match with the neural network.')
    end

    v1_end = (input_layer + bias) + hidden_layer;
    v1 = v(1:v1_end);
    v2 = v(v1_end + 1:length(v));

    w1 = reshape(v1, hidden_layer, input_layer + bias);
    w2 = reshape(v2, output_layer, hidden_layer + bias);
end

function Y = multi_mlp(x_mat, w, neurons)
    n = length(x_mat(1, :));
    Y = zeros(neurons(3), n);
    for i = 1: n
        Y(:, i) = mlp(x_mat(:, i), w, neurons);
    end
end




