#include <snn/snn.h>
#include <stdio.h>

int main(void) {
    snn_size_t layers[] = {2, 2};
    snn_feedforward_config_t cfg = snn_default_feedforward_config(layers, 2);
    snn_network_t *net = NULL;
    snn_state_t *state = NULL;
    float input[4] = {2.0f, 0.0f, 0.0f, 0.0f};
    uint8_t spikes[4] = {0};
    snn_status_t st = snn_build_feedforward(&cfg, NULL, &net);
    if (st != SNN_OK) {
        fprintf(stderr, "build failed: %s\n", snn_status_string(st));
        return 1;
    }
    st = snn_state_create(net, &state);
    if (st != SNN_OK) {
        fprintf(stderr, "state failed: %s\n", snn_status_string(st));
        snn_network_free(net);
        return 1;
    }
    (void)snn_step_cpu(net, state, input, spikes);
    printf("spikes: %u %u %u %u\n", spikes[0], spikes[1], spikes[2], spikes[3]);
    snn_state_free(state);
    snn_network_free(net);
    return 0;
}
