function [output] = inner_product_forward(input, layer, param)

d = size(input.data, 1);
k = size(input.data, 2); % batch size
n = size(param.w, 2);


output.data = ((param.w)'*input.data) + param.b';
output.height = size(output.data);
output.width = size(output.data);
output.channel = 500;
output.batch_size = 64;

end
