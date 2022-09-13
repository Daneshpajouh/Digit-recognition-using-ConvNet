function [output] = relu_forward(input)
output.height = input.height;
output.width = input.width;
output.channel = input.channel;
output.batch_size = input.batch_size;


output.data = zeros(size(input.data));

z = input.data;

output.data = max(0, z);

end
