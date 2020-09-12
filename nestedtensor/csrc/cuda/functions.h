#pragma once

void launch_softmax_backward(
    at::Tensor vals,
    int batch_size,
    int heads,
    int sequence_length);

void launch_softmax(
    at::Tensor vals,
    int batch_size,
    int heads,
    int sequence_length);
