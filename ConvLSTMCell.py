import torch
import torch.nn as nn
from HadamardProduct import HadamardProduct

# Doesn't include the feedback loop on the cells output
class ConvLSTMCell(nn.Module):

    def __init__(self, in_channels, out_channels, 
    kernel_size, padding, activation, frame_size):

        super(ConvLSTMCell, self).__init__()

        self.out_channels = out_channels      

        if activation == "tanh":
            self.activation = torch.tanh 
        elif activation == "relu":
            self.activation = torch.relu
        
        # Single convolutional layer for all convolutions
        self.conv = nn.Conv2d(
            in_channels=in_channels + out_channels, 
            out_channels=4 * out_channels, 
            kernel_size=kernel_size, 
            padding=padding)           

        # Initialize weights for Hadamard Products
        self.W_ci = HadamardProduct((out_channels, *frame_size))
        self.W_co = HadamardProduct((out_channels, *frame_size))
        self.W_cf = HadamardProduct((out_channels, *frame_size))

    def forward(self, X, H_prev, C_prev):

        # Concatenate along channels
        conv_input = torch.cat([X, H_prev], dim=1)

        # Do all convolutions at once for efficiency
        conv_output = self.conv(conv_input)

        # Split along channels
        i_conv, f_conv, C_conv, o_conv = torch.split(
            conv_output, self.out_channels, dim=1)

        input_gate = torch.sigmoid(i_conv + self.W_ci( C_prev ))
        forget_gate = torch.sigmoid(f_conv + self.W_cf( C_prev ))

        # Current Cell output
        C = forget_gate*C_prev + input_gate * self.activation(C_conv)

        output_gate = torch.sigmoid(o_conv + self.W_co( C ))

        # Current Hidden State
        H = output_gate * self.activation(C)

        return H, C
