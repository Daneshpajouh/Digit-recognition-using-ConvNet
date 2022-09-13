function [input_od] = relu_backward(output, input, layer)


input_od = zeros(size(input.data));

z = input.data;

 if z > 0
     input_od = output.diff;
 else
     input_od = 0.*output.diff;
 end
end
