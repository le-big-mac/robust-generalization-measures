from enum import Enum

params = ["lr", "batch_norm", "batch_size", "model_depth", "dropout_prob"]
early_batches = (0, 1, 10, 100)
stop_types = ["final", "best", "_99"]


class ExperimentType(Enum):
    NO_DROPOUT = 1
    DROPOUT = 2
    BATCH_NORM = 3

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.value < other.value
        return TypeError


measure_types = {
    "margin":
        tuple([
            "margin",
            "inverse_margin",
            "fro_over_spec_over_margin_fft",
            "log_prod_of_fro_over_margin",
            "log_prod_of_spec_over_margin_fft",
            "log_spec_init_main_fft",
            "log_spec_orig_main_fft",
            "log_sum_of_fro_over_margin",
            "log_sum_of_spec_over_margin_fft",
            "path_norm_over_margin",
            "spec_init",
            "spec_orig"
        ]),
    "exponential_margin":
        tuple([
            "exponential_margin",
            "fro_over_spec_over_exponential_margin_fft",
            "log_prod_of_fro_over_exponential_margin",
            "log_prod_of_spec_over_exponential_margin_fft",
            "log_spec_init_main_exponential_margin_fft",
            "log_spec_orig_main_exponential_margin_fft",
            "log_sum_of_fro_over_exponential_margin",
            "log_sum_of_spec_over_exponential_margin_fft",
            "path_norm_over_exponential_margin",
            "spec_init_exponential_margin",
            "spec_orig_exponential_margin"
        ]),
    "normalized_exponential_margin":
        tuple([
            "normalized_exponential_margin",
            "fro_over_spec_over_normalized_exponential_margin_fft",
            "log_prod_of_fro_over_normalized_exponential_margin",
            "log_prod_of_spec_over_normalized_exponential_margin_fft",
            "log_spec_init_main_normalized_exponential_margin_fft",
            "log_spec_orig_main_normalized_exponential_margin_fft",
            "log_sum_of_fro_over_normalized_exponential_margin",
            "log_sum_of_spec_over_normalized_exponential_margin_fft",
            "path_norm_over_normalized_exponential_margin",
            "spec_init_normalized_exponential_margin",
            "spec_orig_normalized_exponential_margin"
        ]),
    "shifted_margin":
        tuple([
            "shifted_margin",
            "fro_over_spec_over_shifted_margin_fft",
            "log_prod_of_fro_over_shifted_margin",
            "log_prod_of_spec_over_shifted_margin_fft",
            "log_spec_init_main_shifted_margin_fft",
            "log_spec_orig_main_shifted_margin_fft",
            "log_sum_of_fro_over_shifted_margin",
            "log_sum_of_spec_over_shifted_margin_fft",
            "path_norm_over_shifted_margin",
            "spec_init_shifted_margin",
            "spec_orig_shifted_margin"
        ]),
    "pac_bayes":
        tuple([
            "pacbayes_flatness", 
            "pacbayes_init", 
            "pacbayes_init_full", 
            "pacbayes_init_main",
            "pacbayes_orig", 
            "pacbayes_orig_full", 
            "pacbayes_orig_main",
            "pacbayes_mag_flatness", 
            "pacbayes_mag_init", 
            "pacbayes_mag_init_full", 
            "pacbayes_mag_init_main", 
            "pacbayes_mag_orig", 
            "pacbayes_mag_orig_full", 
            "pacbayes_mag_orig_main",
        ]),
    "vc_dim":
        tuple([
            "vc_dim",
            "params"
        ]),
    "norms":
        tuple([
            "path_norm",
            "param_norm",
            "l2",
            "l2_dist",
            "log_sum_of_spec_fft",
            "log_prod_of_spec_fft",
            "log_prod_of_fro",
            "log_sum_of_fro",
            "fro_over_spec_fft",
            "fro_dist",
            "dist_spec_init_fft"
        ]),
    "optimisation":
        tuple([
            "accuracy/test",
            "accuracy/train",
            "cross_entropy_epoch_end/train",
            "cross_entropy_epoch_end/test",
            "sotl",
            "sotl-ema",
            "sotl-1",
            "sotl-2",
            "sotl-3",
            "sotl-5",
            "sotl-10",
            "_step",
            "epoch"
        ]),
    "hps":
        tuple([
            "lr",
            "batch_norm",
            "batch_size",
            "model_depth",
            "dropout_prob"
        ]),
    "oracles":
        tuple([
            "oracle 0.01"
            "oracle 0.02"
            "oracle 0.05"
        ]),
}
