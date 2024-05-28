- Simple GLA

Gating mechanism in https://arxiv.org/abs/2103.02143. Compared to GLA, the gating is head-wise instead of elementwise. As a result, we can adapt the RetNet kernel for training using matmul w/o numerical instability. It is faster than GLA but has less expressive power. I will use it as a baseline for the GLA.

$S_{t+1} = g_{t+1} \odot S_{t} + K_{t+1} V_{t+1}^{\top}$ where $g$ is a scalar.